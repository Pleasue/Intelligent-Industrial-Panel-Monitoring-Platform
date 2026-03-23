import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import onnx
# from post_process import post_prosses, TextVisualizer
import torch
# =========================================
# 配置路径
# =========================================
# ONNX_MODEL_PATH = "model_fp32_verified.onnx"
ONNX_MODEL_PATH = "model\\model_fp32_720.onnx"
ENGINE_PATH = "model_fp32_720.trt"
IMAGE_FOLDER = "images/"
DIMS = (1280, 720)
FP16_MODE = False

# =========================================
# TensorRT Logger
# =========================================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# =========================================
# 构建 engine 或加载已有 engine
# =========================================
def build_engine(onnx_file_path, engine_file_path, fp16=True):
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

    profile.set_shape('input', [1,3 ,720 ,1280], [1,3,720, 1280], [1,3 ,720 ,1280])
    config.add_optimization_profile(profile)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    # 关键：禁用 Myelin 引擎（彻底绕开 Myelin 相关逻辑）
    # config.clear_flag(trt.BuilderFlag.USE_MYELIN)  # 新增这一行，禁用 Myelin
    # create engine
    with torch.cuda.device("cuda"):
        engine = builder.build_engine(network, config)

    with open(engine_file_path, mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")

if __name__ == "__main__":
    build_engine(ONNX_MODEL_PATH, ENGINE_PATH, fp16=FP16_MODE)