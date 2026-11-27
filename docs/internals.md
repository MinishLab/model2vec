# Internals

## Rust backend

`StaticModel.encode` uses a Rust backend: [model2vec-rs](https://github.com/MinishLab/model2vec-rs). The Python `StaticModel` has embeddings, weights, token mapping and tokenizer as properties. It builds a `model2vec_rs.StaticModel` from them the first time you call `encode`. Tokenization, embedding lookup, pooling and normalization all happen in Rust.

When the dtype of the embeddings matches the backend's dtype, the Python model and the backend share the same memory: `model.embedding` and `model.weights` become _views_ of the backend's arrays, so the embeddings are only stored once.

If you change one of the properties through assignment, the Rust backend is invalidated, and reinstantiated on the next encode.

Here is a demonstration. It uses the private `_get_rust_backend` method to inspect the backend; don't rely on it in your own code.

```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("minishlab/blink-8M")
model.encode("hello")
# Now the Rust backend has been instantiated
backend_1 = model._get_rust_backend()
try:
    # Raises an error: the embeddings are read-only
    model.embedding *= 10
except ValueError:
    print("Error")
model.embedding = model.embedding * 10
# Rust backend is invalidated
model.encode("hello2")
# Rust backend is again up to date
backend_2 = model._get_rust_backend()
backend_1.embeddings.mean()
# ~-0.09
backend_2.embeddings.mean()
# ~-0.9

```

## Supported dtypes

The backend works with two embedding dtypes:

- `float32`
- `int8`

Models in any other dtype still load and encode, but the embeddings are cast to `float32` when the backend is built:

| Model dtype | Backend dtype | Memory | Output dtype of `encode` |
|-------------|---------------|--------|--------------------------|
| `int8`      | `int8`        | shared | `float32`                |
| `float32`   | `float32`     | shared | `float32`                |
| `float16`   | `float32`     | original + `float32` copy | `float16`  |
| `float64`   | `float32`     | original + `float32` copy | `float64`  |

As such, for `float16` and `float64` models you pay for an extra `float32` copy of the embedding matrix, the computation happens in `float32` anyway, and the result is cast back to the original dtype afterwards. A `float16` model therefore uses three times the memory of its embeddings on disk, and a `float64` model does not gain any precision.

## Picking a dtype

Our own models are not stored in `float16`, but some models published by others are. For those, and for `float64` models, it is better to quantize when loading:

- **`float16` models**: load them as `int8`. This is smaller than the `float16` original, uses the backend natively, and the loss in quality is usually negligible.
- **`float64` models**: load them as `float32`. The computation is done in `float32` regardless, so this only removes the extra copy.

```python
from model2vec import StaticModel

# A float16 model, loaded as int8
model = StaticModel.from_pretrained("path/to/float16-model", quantize_to="int8")

# A float64 model, loaded as float32
model = StaticModel.from_pretrained("path/to/float64-model", quantize_to="float32")
```

If you want to stay as close as possible to the original `float16` embeddings, load them as `float32` instead. This also avoids the extra copy, at the cost of twice the memory of the `float16` original.
