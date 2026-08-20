from __future__ import annotations

from model2vec.utils import get_package_extras, importable

_REQUIRED_EXTRA = "onnx"

for extra_dependency in get_package_extras("model2vec", _REQUIRED_EXTRA):
    importable(extra_dependency, _REQUIRED_EXTRA)

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import torch
from skeletoken import TokenizerModel
from tokenizers import Tokenizer
from torch.export import Dim

from model2vec import StaticModel
from model2vec.inference import StaticModelPipeline
from model2vec.inference.mlp import Activation

logger = logging.getLogger(__name__)


def _dummy_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Create dummy, padded (batch_size, sequence_length) `input_ids`/`attention_mask` tensors for tracing.

    :return: A tuple of (input_ids, attention_mask), both of shape (2, 3).
    """
    input_ids = torch.zeros((2, 3), dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
    return input_ids, attention_mask


def _export_onnx(*args: Any, **kwargs: Any) -> None:
    """Call `torch.onnx.export`, silencing the benign "axis name will not be used" warning.

    `input_ids` and `attention_mask` deliberately share the same `Dim` objects (see `_dynamic_shapes`), so
    torch's exporter renames their shared symbol once and then warns, on the second input, that the (identical)
    rename is redundant. That's expected here and not a sign of a problem.

    :param *args: Positional arguments forwarded to `torch.onnx.export`.
    :param **kwargs: Keyword arguments forwarded to `torch.onnx.export`.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*will not be used, since it shares the same shape constraints")
        torch.onnx.export(*args, **kwargs)


def _dynamic_shapes() -> dict[str, dict[int, Dim]]:
    """Declare the dynamic (batch_size, sequence_length) axes shared by `input_ids` and `attention_mask`.

    :return: A `dynamic_shapes` mapping for `torch.onnx.export`.
    """
    batch_size = Dim("batch_size")
    sequence_length = Dim("sequence_length")
    return {
        "input_ids": {0: batch_size, 1: sequence_length},
        "attention_mask": {0: batch_size, 1: sequence_length},
    }


class TorchStaticModel(torch.nn.Module):
    def __init__(self, model: StaticModel) -> None:
        """Initialize the TorchStaticModel with a StaticModel instance."""
        super().__init__()
        embeddings = torch.from_numpy(model.embedding)
        if embeddings.dtype in {torch.int8, torch.uint8}:
            embeddings = embeddings.to(torch.float16)
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.normalize = model.normalize

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model, following the standard transformers `input_ids`/`attention_mask` signature.

        :param input_ids: The input token ids, of shape (batch_size, sequence_length).
        :param attention_mask: 1 for real tokens and 0 for padding, of shape (batch_size, sequence_length).
        :return: The embeddings, of shape (batch_size, embedding_dim).
        """
        mask = attention_mask.unsqueeze(-1).to(self.embeddings.weight.dtype)
        # Zero out padding
        embeddings = self.embeddings(input_ids) * mask
        embeddings = embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # Normalize if required
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings


class TorchStaticModelPipeline(torch.nn.Module):
    def __init__(self, pipeline: StaticModelPipeline) -> None:
        """Wrap a StaticModelPipeline (encoder + MLP head) as a single torch module."""
        super().__init__()
        self.encoder = TorchStaticModel(pipeline.model)
        self.activation = pipeline.head.activation
        # Rebuild the head Layers as nn.Linear. Layer stores weight as [out, in] and computes
        # x @ weight.T + bias, which matches nn.Linear(in, out) exactly.
        self.layers = torch.nn.ModuleList()
        for layer in pipeline.head.layers:
            weight = torch.from_numpy(layer.weight)
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight = torch.nn.Parameter(weight, requires_grad=False)
            linear.bias = torch.nn.Parameter(torch.from_numpy(layer.bias), requires_grad=False)
            self.layers.append(linear)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode the inputs and run them through the head, applying the output activation."""
        out = self.encoder(input_ids, attention_mask).float()
        *hidden_layers, last_layer = self.layers
        for layer in hidden_layers:
            out = torch.relu(layer(out))
        logits = last_layer(out)
        if self.activation == Activation.SOFTMAX:
            return torch.softmax(logits, dim=-1)
        if self.activation == Activation.SIGMOID:
            return torch.sigmoid(logits)
        return logits


def export_model_to_onnx(model: StaticModel | StaticModelPipeline, save_path: str | Path) -> None:
    """Export a StaticModel or a StaticModelPipeline to ONNX format and save tokenizer files.

    A classifier/regressor pipeline (one with a trained head) is exported with its head
    included. A plain encoder is exported as embeddings.

    :param model: The StaticModel or StaticModelPipeline instance to export.
    :param save_path: The directory to save the model and related files.
    """
    save_path = Path(save_path)
    if isinstance(model, StaticModelPipeline):
        _export_pipeline_to_onnx(model, save_path)
        return

    _export_encoder_to_onnx(model, save_path)


def _export_encoder_to_onnx(model: StaticModel, save_path: Path) -> None:
    """Export a plain StaticModel encoder to ONNX format and save tokenizer files.

    :param model: The StaticModel instance to export.
    :param save_path: The directory to save the model and related files.
    """
    save_path.mkdir(parents=True, exist_ok=True)

    torch_model = TorchStaticModel(model)
    torch_model.eval()

    # Prepare dummy input data, in the padded (batch_size, sequence_length) shape transformers-style ONNX runners expect
    input_ids, attention_mask = _dummy_inputs()

    # Export the model to ONNX
    onnx_model_path = save_path / "model.onnx"
    onnx_model_path.parent.mkdir(parents=True, exist_ok=True)
    _export_onnx(
        torch_model,
        (input_ids, attention_mask),
        str(onnx_model_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_shapes=_dynamic_shapes(),
    )

    logger.info(f"Model has been successfully exported to {onnx_model_path}")

    # Save the tokenizer files required for transformers.js, and a config.json for ONNX runtime providers
    _save_tokenizer_and_config(model.tokenizer, save_path)
    logger.info(f"Tokenizer files have been saved to {save_path}")


def _export_pipeline_to_onnx(pipeline: StaticModelPipeline, save_path: Path) -> None:
    """Export a StaticModelPipeline (encoder + classifier/regressor head) to ONNX format.

    The exported graph outputs class probabilities for classifiers (softmax/sigmoid heads) or
    raw predictions for regression/projector heads (identity activation).

    :param pipeline: The pretrained StaticModelPipeline.
    :param save_path: The directory to save the model and related files.
    """
    save_path.mkdir(parents=True, exist_ok=True)

    torch_model = TorchStaticModelPipeline(pipeline)
    torch_model.eval()

    # Prepare dummy input data, in the padded (batch_size, sequence_length) shape transformers-style ONNX runners expect
    input_ids, attention_mask = _dummy_inputs()

    # An identity head is a regressor/projector, so its output is a raw value rather than a probability
    output_name = "predictions" if pipeline.head.activation == Activation.IDENTITY else "probabilities"

    onnx_model_path = save_path / "model.onnx"
    onnx_model_path.parent.mkdir(parents=True, exist_ok=True)
    _export_onnx(
        torch_model,
        (input_ids, attention_mask),
        str(onnx_model_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=[output_name],
        dynamic_shapes=_dynamic_shapes(),
    )

    logger.info(f"Pipeline has been successfully exported to {onnx_model_path}")

    # Save the tokenizer files required for transformers.js, and a config.json for ONNX runtime providers
    _save_tokenizer_and_config(pipeline.model.tokenizer, save_path)
    logger.info(f"Tokenizer files have been saved to {save_path}")


def _resolve_pad_token_id(tokenizer: Tokenizer, tokenizer_model: TokenizerModel) -> int:
    """Resolve a pad token id for a StaticModel tokenizer, which may not have one registered as special.

    :param tokenizer: The tokenizer from the StaticModel.
    :param tokenizer_model: `tokenizer` wrapped as a `skeletoken.TokenizerModel`.
    :return: A vocabulary id to use as `pad_token_id`.
    """
    if tokenizer_model.pad_token_id is not None:
        return tokenizer_model.pad_token_id

    literal_pad_id = tokenizer.token_to_id("[PAD]")
    if literal_pad_id is not None:
        return literal_pad_id

    if tokenizer_model.unk_token_id is not None:
        return tokenizer_model.unk_token_id

    return 0


def _save_tokenizer_and_config(tokenizer: Tokenizer, save_directory: Path) -> None:
    """Save tokenizer files in a format compatible with Transformers, plus config.json and special_tokens_map.json.

    :param tokenizer: The tokenizer from the StaticModel.
    :param save_directory: The directory to save the tokenizer and config files.
    """
    tokenizer_model = TokenizerModel.from_tokenizer(tokenizer)
    tokenizer_model.to_transformers().save_pretrained(save_directory)

    pad_token_id = _resolve_pad_token_id(tokenizer, tokenizer_model)
    pad_token = tokenizer.id_to_token(pad_token_id) or ""

    config = {"pad_token_id": pad_token_id}
    (save_directory / "config.json").write_text(json.dumps(config, indent=2))

    special_tokens_map = {"pad_token": pad_token}
    (save_directory / "special_tokens_map.json").write_text(json.dumps(special_tokens_map, indent=2))
