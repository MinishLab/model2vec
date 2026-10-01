from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from model2vec import StaticModel


@pytest.fixture
def concurrent_model() -> StaticModel:
    """建立无需下载的真实分词器和静态向量."""
    words = ["[UNK]", "a", "b", "c", "longwordnumberone", "longwordnumbertwo", "longwordnumberthree"]
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    vectors = np.arange(len(words) * 2, dtype=np.float32).reshape(len(words), 2)
    return StaticModel(vectors, tokenizer, max_length=2)


@pytest.mark.parametrize("short_method", ["encode", "encode_as_sequence"])
@pytest.mark.parametrize("long_method", ["encode", "encode_as_sequence"])
@pytest.mark.parametrize("long_limit", [3, None])
@pytest.mark.parametrize("parallel_batches", [False, True])
def test_concurrent_length_overrides(
    concurrent_model: StaticModel,
    monkeypatch: pytest.MonkeyPatch,
    short_method: str,
    long_method: str,
    long_limit: int | None,
    parallel_batches: bool,
) -> None:
    """不同长度及两种编码路径的重叠请求必须与顺序结果一致."""
    short_encode = getattr(concurrent_model, short_method)
    long_encode = getattr(concurrent_model, long_method)
    batch_options = {"batch_size": 1, "use_multiprocessing": parallel_batches, "multiprocessing_threshold": 0}
    expected_short = short_encode(["a b c"] * 2, max_length=1, **batch_options)
    expected_long = long_encode(["c b a"] * 2, max_length=long_limit, **batch_options)
    entered_short = Event()
    finished_long = Event()
    original_tokenize = concurrent_model._tokenize

    def scheduled_tokenize(sentences: list[str], tokenizer: Tokenizer) -> list[list[int]]:
        if sentences == ["a b c"]:
            entered_short.set()
            assert finished_long.wait(10), "长请求未完成"
        return original_tokenize(sentences, tokenizer)

    monkeypatch.setattr(concurrent_model, "_tokenize", scheduled_tokenize)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(short_encode, ["a b c"] * 2, max_length=1, **batch_options)
        try:
            assert entered_short.wait(10), "短请求未开始"
            actual_long = long_encode(["c b a"] * 2, max_length=long_limit, **batch_options)
        finally:
            finished_long.set()
        actual_short = pending.result(timeout=10)
    for expected, actual in zip(expected_short, actual_short):
        np.testing.assert_array_equal(actual, expected)
    for expected, actual in zip(expected_long, actual_long):
        np.testing.assert_array_equal(actual, expected)
    assert concurrent_model.tokenizer.truncation["max_length"] == 2


def test_default_encoding_reuses_tokenizer(concurrent_model: StaticModel, monkeypatch: pytest.MonkeyPatch) -> None:
    """默认路径不复制 tokenizer，避免给常规推理增加复制开销."""
    observed = []
    original_tokenize = concurrent_model._tokenize

    def observe(sentences: list[str], tokenizer: Tokenizer) -> list[list[int]]:
        observed.append(tokenizer)
        return original_tokenize(sentences, tokenizer)

    monkeypatch.setattr(concurrent_model, "_tokenize", observe)
    concurrent_model.encode(["a b c"], use_multiprocessing=False)
    assert observed == [concurrent_model.tokenizer]


def test_encoding_error_keeps_default_truncation(
    concurrent_model: StaticModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """覆盖长度的编码异常也不能改变模型默认配置."""

    def fail(*args: Any, **kwargs: Any) -> list[list[int]]:
        raise ValueError("分词失败")

    monkeypatch.setattr(concurrent_model, "_tokenize", fail)
    with pytest.raises(ValueError, match="分词失败"):
        concurrent_model.encode(["a b c"], max_length=None, use_multiprocessing=False)
    assert concurrent_model.tokenizer.truncation["max_length"] == 2
