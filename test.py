from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

model_name = "onnxmodel"

TextEmbedding.add_custom_model(
    model=model_name,
    pooling=PoolingType.DISABLED,
    normalization=True,
    sources=ModelSource(url="onnxmodel"),
    dim=256,
    model_file="model.onnx",
)

model = TextEmbedding(model_name=model_name, threads=1, specific_model_path="onnxmodel")
