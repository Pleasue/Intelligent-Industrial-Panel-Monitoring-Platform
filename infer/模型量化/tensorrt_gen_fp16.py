import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import onnx
from post_process import post_prosses, TextVisualizer
import cv2
import torch
# =========================================
# 配置路径
# =========================================
# ONNX_MODEL_PATH = "model_fp32_verified.onnx"
ONNX_MODEL_PATH = "model_fp32_simplified.onnx"
ENGINE_PATH = "model_fp32.trt"
IMAGE_FOLDER = "images/"
DIMS = (640, 640)
FP16_MODE = False

# =========================================
# TensorRT Logger
# =========================================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# =========================================
# 构建 engine 或加载已有 engine
# =========================================
def build_engine(onnx_file_path, engine_file_path, fp16=True):
    # if os.path.exists(engine_file_path):
    #     print(f"✅ Loading existing engine: {engine_file_path}")
    #     with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    #         engine = runtime.deserialize_cuda_engine(f.read())
    #     return engine

    # print(f"🔨 Building TensorRT engine from ONNX: {onnx_file_path}")
    # logger = trt.Logger(trt.Logger.ERROR)
    logger = trt.Logger(trt.Logger.WARNING)
    # logger = engine_file_path
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
    )
    network = builder.create_network(EXPLICIT_BATCH)

    parser = trt.OnnxParser(network, logger)
    
    onnx_model = onnx.load(onnx_file_path)

    if not parser.parse(onnx_model.SerializePartialToString()):
        error_msgs = ""
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config()
    config.max_workspace_size = 4<<30
    profile = builder.create_optimization_profile()

    profile.set_shape('input', [1,3 ,640 ,640], [1,3,640, 640], [1,3 ,640 ,640])
    config.add_optimization_profile(profile)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    # create engine
    with torch.cuda.device("cuda"):
        engine = builder.build_engine(network, config)

    with open(engine_file_path, mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")

    # with trt.Builder(TRT_LOGGER) as builder, \
    #      builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
    #      trt.OnnxParser(network, TRT_LOGGER) as parser, \
    #      builder.create_builder_config() as config:

    #     # 设置 workspace
    #     config.max_workspace_size = 4 << 30  # 4GB
    #     if fp16:
    #         config.set_flag(trt.BuilderFlag.FP16)

    #     # 解析 ONNX
    #     with open(onnx_file_path, 'rb') as f:
    #         if not parser.parse(f.read()):
    #             print("❌ Failed to parse ONNX model")
    #             for err in range(parser.num_errors):
    #                 print(parser.get_error(err))
    #             return None

    #     # 设置静态输入 shape
    #     for i in range(network.num_inputs):
    #         tensor = network.get_input(i)
    #         tensor.shape = (1, 3, DIMS[1], DIMS[0])

    #     # 构建 engine
    #     engine = builder.build_engine(network, config)
    #     if engine is None:model_fp16_official.engine'
    #         raise RuntimeError("❌ Failed to build TensorRT engine")

    #     # 保存 engine
    #     with open(engine_file_path, "wb") as f:
    #         f.write(engine.serialize())
    #     print(f"💾 Engine saved to {engine_file_path}")
    #     return engine

if __name__ == "__main__":
    build_engine(ONNX_MODEL_PATH, ENGINE_PATH, fp16=FP16_MODE)