from model2vec.utils import get_package_extras, importable

# Define the optional dependency group name
_REQUIRED_EXTRA = "onnx"

# Check if each dependency for the "onnx" group is importable
for extra_dependency in get_package_extras("model2vec", _REQUIRED_EXTRA):
    importable(extra_dependency, _REQUIRED_EXTRA)

import argparse
import logging
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from model2vec import StaticModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TorchStaticModel(torch.nn.Module):
    def __init__(self, model: StaticModel) -> None:
        """Initialize the TorchStaticModel with a StaticModel instance."""
        super().__init__()
        # Convert NumPy embeddings to a torch.nn.EmbeddingBag
        embeddings = torch.tensor(model.embedding, dtype=torch.float32)
        self.embedding_bag = torch.nn.EmbeddingBag.from_pretrained(embeddings, mode="mean", freeze=True)
        self.normalize = model.normalize
        # Save tokenizer attributes
        self.tokenizer = model.tokenizer
        self.unk_token_id = model.unk_token_id
        self.median_token_length = model.median_token_length

    def forward(self, input_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param input_ids: The input token ids.
        :param offsets: The offsets to compute the mean pooling.
        :return: The embeddings.
        """
        # Perform embedding lookup and mean pooling
        embeddings = self.embedding_bag(input_ids, offsets)
        # Normalize if required
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def tokenize(self, sentences: list[str], max_length: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize the input sentences.

        :param sentences: The input sentences.
        :param max_length: The maximum length of the input_ids.
        :return: The input_ids and offsets.
        """
        # Tokenization logic similar to your StaticModel
        if max_length is not None:
            m = max_length * self.median_token_length
            sentences = [sentence[:m] for sentence in sentences]
        encodings = self.tokenizer.encode_batch(sentences, add_special_tokens=False)
        encodings_ids = [encoding.ids for encoding in encodings]
        if self.unk_token_id is not None:
            # Remove unknown tokens
            encodings_ids = [
                [token_id for token_id in token_ids if token_id != self.unk_token_id] for token_ids in encodings_ids
            ]
        if max_length is not None:
            encodings_ids = [token_ids[:max_length] for token_ids in encodings_ids]
        # Flatten input_ids and compute offsets
        offsets = torch.tensor([0] + [len(ids) for ids in encodings_ids[:-1]], dtype=torch.long).cumsum(dim=0)
        input_ids = torch.tensor(
            [token_id for token_ids in encodings_ids for token_id in token_ids],
            dtype=torch.long,
        )
        return input_ids, offsets


def export_model_to_onnx(model_path: str, save_path: Path) -> None:
    """
    Export the StaticModel to ONNX format and save tokenizer files.

    :param model_path: The path to the pretrained StaticModel.
    :param save_path: The directory to save the model and related files.
    """
    save_path.mkdir(parents=True, exist_ok=True)

    # Load the StaticModel
    model = StaticModel.from_pretrained(model_path)
    torch_model = TorchStaticModel(model)

    # Save the model using save_pretrained
    model.save_pretrained(save_path)

    # Prepare dummy input data
    texts = ["hello", "hello world"]
    input_ids, offsets = torch_model.tokenize(texts)

    # Export the model to ONNX
    onnx_model_path = save_path / "onnx/model.onnx"
    onnx_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        torch_model,
        (input_ids, offsets),
        str(onnx_model_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "offsets"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "num_tokens"},
            "offsets": {0: "batch_size"},
            "embeddings": {0: "batch_size"},
        },
    )

    logger.info(f"Model has been successfully exported to {onnx_model_path}")

    # Save the tokenizer files required for transformers.js
    save_tokenizer(model.tokenizer, save_path)
    logger.info(f"Tokenizer files have been saved to {save_path}")


def save_tokenizer(tokenizer: Tokenizer, save_directory: Path) -> None:
    """
    Save tokenizer files in a format compatible with Transformers.

    :param tokenizer: The tokenizer from the StaticModel.
    :param save_directory: The directory to save the tokenizer files.
    """
    # Convert the tokenizers.Tokenizer to a PreTrainedTokenizerFast and save
    tokenizer_json_path = save_directory / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    # Load the tokenizer using PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    # Save the tokenizer files
    fast_tokenizer.save_pretrained(str(save_directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export StaticModel to ONNX format")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained StaticModel",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Directory to save the exported model and files",
    )
    args = parser.parse_args()

    export_model_to_onnx(args.model_path, Path(args.save_path))
