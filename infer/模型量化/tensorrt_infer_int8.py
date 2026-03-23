import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time
import torch
from post_process import post_prosses, TextVisualizer

# =========================================
# 配置路径与参数（INT8 专用）
# =========================================
ENGINE_PATH = "model_int8.trt"  # 切换为 INT8 引擎
IMAGE_FOLDER = "images/"
DIMS = (640, 640)  # 需与校准和引擎构建时的形状一致
INT8_MODE = True
WARMUP_ITER = 10   # 热身推理次数
TEST_ITER = 50     # 正式测试次数（取平均值）

# # 校准预处理参数（必须与 INT8 校准阶段完全一致！）
# MEAN = [123.675, 116.28, 103.53]    # 与训练/校准一致的均值
# STD = [58.395, 57.12, 57.375]      # 与训练/校准一致的方差

# =========================================
# TensorRT Logger 与插件注册
# =========================================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")  # 注册插件（确保与构建时一致）

# =========================================
# INT8 推理专用函数（优化预处理与精度匹配）
# =========================================
def infer_int8(engine, image, image_path):
    context = engine.create_execution_context()
    # 配置动态形状（若引擎是动态的，需设置与输入匹配的形状）
    if engine.num_optimization_profiles > 0:
        input_idx = 0
        input_name = engine.get_binding_name(input_idx)
        # 设置实际输入形状 (batch=1, channel=3, height, width)
        context.set_binding_shape(input_idx, (1, 3, DIMS[1], DIMS[0]))
        print(f"⚙️ 动态形状配置: {input_name} -> (1,3,{DIMS[1]},{DIMS[0]})")

    # 图像预处理（与 INT8 校准严格一致，关键！）
    input_data = preprocess_int8(image)
    stream = cuda.Stream()
    bindings = []

    # 分配输入显存
    d_input = cuda.mem_alloc(input_data.nbytes)
    cuda.memcpy_htod_async(d_input, input_data, stream)
    bindings.append(int(d_input))

    # 分配输出显存（根据实际绑定形状动态获取）
    output_shapes = []
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            # 动态形状下需用 context 实际形状，而非 engine 静态形状
            shape = context.get_binding_shape(i)
            output_shapes.append(tuple(shape))
    
    d_outputs = []
    host_outputs = []
    for shape in output_shapes:
        host_out = np.empty(shape, dtype=np.float32)  # INT8 输出仍为 FP32（量化在引擎内部）
        d_out = cuda.mem_alloc(host_out.nbytes)
        d_outputs.append(d_out)
        host_outputs.append(host_out)
        bindings.append(int(d_out))

    # ===== 热身推理（排除初始化耗时）=====
    for _ in range(WARMUP_ITER):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for host_out, d_out in zip(host_outputs, d_outputs):
            cuda.memcpy_dtoh_async(host_out, d_out, stream)
    stream.synchronize()

    # ===== 正式推理（测量纯推理耗时）=====
    pure_infer_times = []
    for _ in range(TEST_ITER):
        # 拷贝输入到 GPU（不计入纯推理耗时）
        cuda.memcpy_htod_async(d_input, input_data, stream)
        
        # 记录纯推理时间（仅 GPU 执行推理的耗时）
        t0 = time.time()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()  # 等待 GPU 完成
        t1 = time.time()
        
        pure_infer_times.append((t1 - t0) * 1000)  # 转换为毫秒

        # 拷贝输出（不计入纯推理耗时）
        for host_out, d_out in zip(host_outputs, d_outputs):
            cuda.memcpy_dtoh_async(host_out, d_out, stream)
    stream.synchronize()

    # 统计纯推理耗时
    avg_pure_infer = np.mean(pure_infer_times)
    min_pure_infer = np.min(pure_infer_times)
    max_pure_infer = np.max(pure_infer_times)
    fps = TEST_ITER / (np.sum(pure_infer_times) / 1000)  # 吞吐量（FPS）

    # 打印单图统计
    print(f"\n【{os.path.basename(image_path)}】")
    print(f"纯推理耗时 - 平均: {avg_pure_infer:.2f} ms | 最小: {min_pure_infer:.2f} ms | 最大: {max_pure_infer:.2f} ms")
    print(f"吞吐量: {fps:.2f} FPS")

    return host_outputs

# =========================================
# INT8 专用预处理（与校准阶段严格一致）
# =========================================
def preprocess_int8(image):
    """预处理需与 INT8 校准完全一致，否则精度会下降"""
    # 1. 调整大小（与 DIMS 一致）
    resized = cv2.resize(image, DIMS)
    # 2. BGR→RGB（与 OpenCV 读取的格式匹配）
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # 3. 转换为 float32 并归一化（关键：使用与校准相同的均值和方差）
    # normalized = (rgb.astype(np.float32) - MEAN) / STD
    # 4. 维度转换为 (1, 3, H, W) 并确保内存连续
    input_data = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(input_data, dtype=np.float32)

# =========================================
# 主程序（适配 INT8 引擎，保留端到端统计）
# =========================================
if __name__ == "__main__":
    # 加载 INT8 引擎
    with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("❌ INT8 引擎加载失败，请检查引擎文件或插件是否匹配")
        exit(1)

    # 打印引擎信息（确认输入输出绑定）
    print(f"✅ 成功加载 INT8 TensorRT 引擎: {ENGINE_PATH}")
    print(f"绑定数量: {engine.num_bindings}")
    for i in range(engine.num_bindings):
        binding_name = engine.get_binding_name(i)
        binding_type = "输入" if engine.binding_is_input(i) else "输出"
        # 动态形状下，引擎的绑定形状可能为 -1，需在推理时设置
        binding_shape = engine.get_binding_shape(i)
        print(f"绑定 {i}: {binding_name} ({binding_type})，形状: {binding_shape}")

    # 加载测试图像
    image_files = [
        os.path.join(IMAGE_FOLDER, f)
        for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    if not image_files:
        print(f"⚠️ 在 {IMAGE_FOLDER} 中未找到图像文件")
        exit(0)
    print(f"\n📊 找到 {len(image_files)} 张测试图像\n")

    # 批量推理与统计
    total_end2end_time = 0.0
    for IMAGE_PATH in image_files:
        # 端到端计时开始（包含读取、预处理、推理、后处理）
        t_end2end_start = time.time()
        
        # 读取图像
        frame = cv2.imread(IMAGE_PATH)
        if frame is None:
            print(f"⚠️ 无法读取图像: {IMAGE_PATH}")
            continue
        
        # 执行 INT8 推理
        outputs = infer_int8(engine, frame, IMAGE_PATH)  # 直接传入 BGR 原图（预处理在内部完成）

        # 后处理与可视化（与原 FP16 流程完全兼容）
        outputs = [np.ascontiguousarray(o.astype(np.float32)) for o in outputs]
        predictions = [torch.from_numpy(p) for p in outputs]
        pr = post_prosses(predictions[0], predictions[2], predictions[1], predictions[3], DIMS)

        # 可视化结果（使用原图 resize 后的版本）
        frame_resized = cv2.resize(frame, DIMS)
        frame_tensor = torch.as_tensor(frame_resized)
        visualizer = TextVisualizer(frame_tensor)
        vis_output = visualizer.draw_instance_predictions(predictions=pr)
        out_img = vis_output.get_image()[:, :, ::-1]  # RGB→BGR 用于 OpenCV 显示

        # 端到端耗时统计
        t_end2end = (time.time() - t_end2end_start) * 1000
        total_end2end_time += t_end2end
        print(f"端到端耗时: {t_end2end:.2f} ms")

        # 显示结果（按 ESC 退出）
        cv2.imshow('INT8 TensorRT Inference', out_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 整体性能统计
    avg_end2end = total_end2end_time / len(image_files)
    print(f"\n📈 整体统计 - 平均端到端耗时: {avg_end2end:.2f} ms")
    cv2.destroyAllWindows()
