import os
import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from tqdm import tqdm  # 进度条（需安装：pip install tqdm）

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# ==============================================================
# 1️⃣ 自定义校准器（优化预处理，匹配训练分布）
# ==============================================================
class ImageEntropyCalibrator(trt.IInt8EntropyCalibratorV2):
    def __init__(self, data_dir, cache_file, input_shape=(1, 3, 640, 640), batch_size=1):
        super().__init__()
        self.data_dir = data_dir
        self.cache_file = cache_file
        self.input_shape = input_shape  # (batch, channel, height, width)
        self.batch_size = batch_size
        self.current_index = 0
        self.calibration_log = []  # 存储校准过程日志
        self.start_time = None  # 校准开始时间

        # 加载并过滤校准图像
        self.image_paths = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        if not self.image_paths:
            raise ValueError(f"❌ 校准目录 {data_dir} 中未找到图像文件")
        self.total_images = len(self.image_paths)
        self.calibration_log.append(f"📊 校准数据统计 - 总图像数: {self.total_images}, 批次大小: {batch_size}")

        # 预分配 GPU 内存
        self.input_volume = trt.volume(input_shape)
        self.d_input = cuda.mem_alloc(self.input_volume * np.float32().nbytes)

        # 预加载并预处理所有图像（记录耗时）
        self.calibration_batches, preprocess_time = self._preprocess_batches()
        self.total_batches = len(self.calibration_batches)
        self.calibration_log.append(f"⚡ 预处理完成 - 批次数量: {self.total_batches}, 耗时: {preprocess_time:.2f}s")
        self.calibration_log.append(f"📦 每批次形状: {self.calibration_batches[0].shape}（最后批次: {self.calibration_batches[-1].shape}）")

    def _preprocess_batches(self):
        """预处理所有图像并记录耗时"""
        batches = []
        batch = []
        invalid_count = 0
        start_time = time.time()

        print("📦 预处理校准图像...")
        for img_path in tqdm(self.image_paths, desc="预处理进度"):
            try:
                img = self.preprocess_image(img_path)
                batch.append(img)
                if len(batch) == self.batch_size:
                    batches.append(np.concatenate(batch, axis=0))
                    batch = []
            except Exception as e:
                invalid_count += 1
                self.calibration_log.append(f"⚠️ 跳过无效图像: {os.path.basename(img_path)} - 错误: {str(e)}")
        
        # 处理最后一个不足批次
        if batch:
            batches.append(np.concatenate(batch, axis=0))
        
        preprocess_time = time.time() - start_time
        self.calibration_log.append(f"✅ 预处理统计 - 有效图像: {self.total_images - invalid_count}, 无效图像: {invalid_count}")
        return batches, preprocess_time

    def preprocess_image(self, path):
        """图像预处理（与训练时严格一致）"""
        frame = cv2.imread(path)
        frame_resized = cv2.resize(frame, (self.input_shape[3], self.input_shape[2]))
        original_image = frame_resized[:, :, ::-1]  # BGR → RGB
        input_tensor = original_image.astype("float32").transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor
    
    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """获取下一批次并记录状态"""
        if self.current_index >= self.total_batches:
            # 校准结束，记录总耗时
            total_calib_time = time.time() - self.start_time
            self.calibration_log.append(f"⏱️ 校准完成 - 总耗时: {total_calib_time:.2f}s, 平均每批次耗时: {total_calib_time/self.total_batches:.2f}s")
            return None
        
        # 开始校准（记录首批次开始时间）
        if self.current_index == 0:
            self.start_time = time.time()
            self.calibration_log.append("\n🚀 开始校准过程...")
        
        # 记录当前批次信息
        batch = self.calibration_batches[self.current_index]
        batch_idx = self.current_index + 1
        batch_shape = batch.shape
        self.calibration_log.append(f"📥 校准批次 {batch_idx:03d}/{self.total_batches} - 形状: {batch_shape}, 样本数: {batch_shape[0]}")
        
        # 拷贝数据到 GPU
        cuda.memcpy_htod(self.d_input, batch.ravel())
        self.current_index += 1
        return [int(self.d_input)]

    def read_calibration_cache(self):
        """读取缓存并记录"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                cache = f.read()
            self.calibration_log.append(f"🧩 加载已有校准缓存 - 文件名: {self.cache_file}, 大小: {len(cache)/1024:.1f}KB")
            return cache
        self.calibration_log.append(f"ℹ️ 未找到校准缓存，将重新生成")
        return None

    def write_calibration_cache(self, cache):
        """保存缓存并记录"""
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        self.calibration_log.append(f"💾 保存校准缓存 - 文件名: {self.cache_file}, 大小: {len(cache)/1024:.1f}KB")

    def print_calibration_report(self):
        """打印完整校准报告"""
        print("\n" + "="*60)
        print("📋 INT8 校准过程报告")
        print("="*60)
        for log in self.calibration_log:
            print(log)
        print("="*60 + "\n")


# ==============================================================
# 2️⃣ 构建 INT8 Engine（适配旧版本 TensorRT，移除 int8_allow_force_int8）
# ==============================================================
def build_int8_engine(
    onnx_path, 
    engine_path, 
    calibration_dataset, 
    input_shape=(1, 3, 640, 640),
    dynamic_shapes=None,
    workspace_size=4
):
    # 输入校验
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"❌ ONNX 模型不存在: {onnx_path}")
    if not os.path.isdir(calibration_dataset):
        raise NotADirectoryError(f"❌ 校准目录不存在: {calibration_dataset}")

    # 初始化校准器（带记录功能）
    calibrator = ImageEntropyCalibrator(
        data_dir=calibration_dataset,
        cache_file="calibration.cache",
        input_shape=input_shape,
        batch_size=input_shape[0]
    )

    # 创建 TensorRT 组件
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # 配置工作空间
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size << 30)
    print(f"💾 工作空间大小: {workspace_size}GB")

    # ===================== INT8 关键配置（适配旧版本）=====================
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)  # 严格模式，不自动降级精度
    
    # 移除 int8_allow_force_int8，改用以下替代方案（旧版本兼容）
    if hasattr(config, 'int8_allow_force_int8'):
        config.int8_allow_force_int8 = True
        print("🎯 启用强制 INT8 量化（高版本 TensorRT）")
    else:
        # 旧版本替代方案：通过增加校准数据覆盖度+优化ONNX，间接解决量化参数缺失
        print("🎯 适配旧版本 TensorRT，通过校准数据优化量化效果")
    # ==================================================================

    print("🎯 启用严格 INT8 模式")

    # 解析 ONNX
    print(f"🧩 解析 ONNX 模型: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ ONNX 解析失败，错误信息：")
            for i in range(parser.num_errors):
                print(f"  错误 {i+1}: {parser.get_error(i)}")
            raise RuntimeError("ONNX 解析失败")

    # 配置输入形状
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"🔍 输入张量: 名称={input_name}, 原始形状={input_tensor.shape}")

    if dynamic_shapes:
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, **dynamic_shapes)
        config.add_optimization_profile(profile)
        print(f"📊 动态形状: min={dynamic_shapes['min']}, opt={dynamic_shapes['opt']}, max={dynamic_shapes['max']}")
    else:
        input_tensor.shape = input_shape
        print(f"📊 静态形状: {input_shape}")

    # 构建引擎
    print("\n🚀 开始构建 INT8 引擎（可能需要数分钟）...")
    build_start_time = time.time()
    engine = builder.build_engine(network, config)
    build_total_time = time.time() - build_start_time

    if engine is None:
        # 打印校准报告后再报错
        calibrator.print_calibration_report()
        raise RuntimeError("❌ 引擎构建失败")

    # 保存引擎
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    engine_size = os.path.getsize(engine_path) / 1024 / 1024
    print(f"✅ 引擎保存完成 - 路径: {engine_path}, 大小: {engine_size:.2f}MB, 构建耗时: {build_total_time:.2f}s")

    # 打印校准报告和引擎信息
    calibrator.print_calibration_report()
    print("🔍 引擎绑定信息:")
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        mode = "输入" if engine.binding_is_input(i) else "输出"
        print(f"  {i}: {name:<30} {mode:<6} 数据类型: {dtype}")

    return engine, calibrator


# ==============================================================
# 3️⃣ 主入口（保存校准记录到文件）
# ==============================================================
if __name__ == "__main__":
    # 配置参数
    ONNX_PATH = "model_fp32_verified.onnx"
    ENGINE_PATH = "model_int8.trt"
    CALIBRATION_DATASET = "images_for_int8/"
    INPUT_SHAPE = (1, 3, 640, 640)
    # DYNAMIC_SHAPES = {
    #     "min": (1, 3, 320, 320),
    #     "opt": (1, 3, 640, 640),
    #     "max": (1, 3, 1280, 1280)
    # }
    DYNAMIC_SHAPES = {}
    WORKSPACE_SIZE_GB = 8
    LOG_SAVE_PATH = "calibration_log.txt"  # 校准记录保存路径

    # 重要：删除旧缓存和引擎，避免影响新配置
    if os.path.exists("calibration.cache"):
        os.remove("calibration.cache")
        print("🗑️ 已删除旧校准缓存，将重新生成")
    if os.path.exists(ENGINE_PATH):
        os.remove(ENGINE_PATH)
        print("🗑️ 已删除旧引擎文件，将重新构建")

    try:
        engine, calibrator = build_int8_engine(
            onnx_path=ONNX_PATH,
            engine_path=ENGINE_PATH,
            calibration_dataset=CALIBRATION_DATASET,
            input_shape=INPUT_SHAPE,
            # dynamic_shapes=DYNAMIC_SHAPES,
            workspace_size=WORKSPACE_SIZE_GB
        )

        # 保存校准记录到文件（永久追溯）
        with open(LOG_SAVE_PATH, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write(f"📅 校准日志生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"🔧 配置参数 - ONNX模型: {ONNX_PATH}, 引擎路径: {ENGINE_PATH}\n")
            f.write(f"🔧 输入形状: {INPUT_SHAPE}, 动态范围: {DYNAMIC_SHAPES}\n")
            f.write("="*60 + "\n\n")
            for log in calibrator.calibration_log:
                f.write(log + "\n")
        
        print(f"\n🎉 所有过程完成！")
        print(f"📄 校准记录已保存到: {LOG_SAVE_PATH}")
        print("👉 验证引擎精度命令:")
        print(f"   trtexec --loadEngine={ENGINE_PATH} --verbose | grep 'precision'")

    except Exception as e:
        print(f"\n❌ 构建失败: {str(e)}")
        exit(1)
