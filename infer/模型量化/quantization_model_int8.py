from onnxruntime.quantization import quantize_dynamic, QuantType

# 动态量化：自动将卷积、矩阵乘等算子量化为INT8
quantize_dynamic(
    model_input="model_fp32_simplified.onnx",  # 输入FP32模型
    model_output="model_int8.onnx",        # 输出INT8模型
    op_types_to_quantize=["Conv", "MatMul"],  # 仅量化计算密集型算子
    weight_type=QuantType.QInt8,               # 权重量化为INT8
    per_channel=False,                        # 简化量化（按张量而非通道）
    reduce_range=True                         # 缩小量化范围，减少精度损失
)
