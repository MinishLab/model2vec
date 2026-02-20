from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    embeddings: Path
    config: Path
    tokenizer: Path
    is_sentence_transformers: bool

    def with_parent(self, parent: Path) -> Layout:
        """Add a parent as a prefix."""
        return Layout(
            embeddings=parent / self.embeddings,
            config=parent / self.config,
            tokenizer=parent / self.tokenizer,
            is_sentence_transformers=self.is_sentence_transformers,
        )

    def is_valid(self) -> bool:
        """Check if all paths exist."""
        return all((self.embeddings.exists(), self.config.exists(), self.tokenizer.exists()))


FOLDER_LAYOUTS: tuple[Layout, ...] = (
    # model2vec
    Layout(
        config=Path("config.json"),
        embeddings=Path("model.safetensors"),
        tokenizer=Path("tokenizer.json"),
        is_sentence_transformers=False,
    ),
    # sentence-transformers
    Layout(
        config=Path("config_sentence_transformers.json"),
        embeddings=Path("model.safetensors"),
        tokenizer=Path("tokenizer.json"),
        is_sentence_transformers=True,
    ),
    # embeddings nested under 0_StaticEmbedding
    Layout(
        config=Path("config_sentence_transformers.json"),
        embeddings=Path("0_StaticEmbedding/model.safetensors"),
        tokenizer=Path("0_StaticEmbedding/tokenizer.json"),
        is_sentence_transformers=True,
    ),
)
