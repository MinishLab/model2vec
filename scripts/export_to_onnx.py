import argparse
import logging
from pathlib import Path

import torch

from model2vec import StaticModel

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
        input_ids = torch.tensor([token_id for token_ids in encodings_ids for token_id in token_ids], dtype=torch.long)
        return input_ids, offsets


def export_model_to_onnx(model_path: str, save_path: str) -> None:
    """
    Export the StaticModel to ONNX format.

    :param model_path: The path to the pretrained StaticModel.
    :param save_path: The path to save the exported ONNX model
    """
    # Convert the StaticModel to TorchStaticModel
    model = StaticModel.from_pretrained(model_path)
    torch_model = TorchStaticModel(model)

    # Prepare dummy input data
    texts = ["hello", "hello world"]
    input_ids, offsets = torch_model.tokenize(texts)

    # Export the model to ONNX
    torch.onnx.export(
        torch_model,
        (input_ids, offsets),
        save_path,
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

    logger.info(f"Model has been successfully exported to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export StaticModel to ONNX format")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the pretrained StaticModel")
    parser.add_argument("--save_path", type=Path, required=True, help="Path to save the exported ONNX model")
    args = parser.parse_args()

    export_model_to_onnx(args.model_path, args.save_path)
