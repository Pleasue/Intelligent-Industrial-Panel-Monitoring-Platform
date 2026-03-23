import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import os
import cv2
import torch
from typing import List, Tuple, Dict
from .config import ENGINE_PATH, INPUT_DIMS
from .post_processor import post_process, process_results

from .cuda_ctx import cuda_init, cuda_push, cuda_pop


cuda_init()

class TRTModelInferencer:
    """TensorRT模型推理器（优化上下文清理）"""
    def __init__(self, engine_path: str = ENGINE_PATH, input_dims: tuple = INPUT_DIMS):
        self.engine_path = engine_path
        self.input_dims = input_dims  # (width, height)
        self.engine = None
        self.context = None
        self.stream = None  # 不再手动创建，从cuda_ctx获取
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.input_names = []
        self.output_names = []
        self.is_destroyed = False  # 标记是否已销毁

        # # 手动创建context上下文
        # self.cfx = cuda.Device(0).make_context()
        cuda_push()

        # 初始化推理器
        self._init_plugins()
        self._init_engine()

        # self.cfx.pop()
        cuda_pop()

    def _init_plugins(self)->None:
        """初始化Tensorrt插件"""
        try:
            trt.init_libnvinfer_plugins(self.logger, "")
            print("✅ 插件初始化成功（已记录插件库）")
        except Exception as e:
            raise RuntimeError(f"插件初始化失败: {str(e)}")


    def _init_engine(self):
        """初始化TensorRT引擎"""
        try:
            # 加载引擎文件
            self._load_engine()
            # 初始化流
            self._init_stream()
            # 解析输入输出张量名称
            self._parse_io_tensors()
            print("✅ TensorRT推理器初始化成功")
        except Exception as e:
            raise RuntimeError(f"推理器初始化失败: {str(e)}")


    def _load_engine(self):
        """加载TensorRT引擎文件"""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"引擎文件不存在: {self.engine_path}")
          
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("引擎文件加载失败（可能是版本不兼容或文件损坏）")

        self.context = self.engine.create_execution_context()

    def _init_stream(self):
        """初始化CUDA流"""
        try:
            self.stream = cuda.Stream()
        except Exception as e:
            raise RuntimeError(f"CUDA流初始化失败:{str(e)}")
        
    def _parse_io_tensors(self):
        """解析输入输出张量名称"""
        """准备输入输出绑定（使用新的Tensor API兼容TensorRT 8.4+）"""
        self.input_names = []
        self.output_names = []

        for idx in range(self.engine.num_bindings):
            tensor_name = self.engine.get_binding_name(idx)
            if self.engine.binding_is_input(idx):
                self.input_names.append(tensor_name)
            else:
                self.output_names.append(tensor_name)

        if not self.input_names:
            raise RuntimeError("未找到输入张量（引擎文件可能有误）")
        if len(self.output_names) < 4:
            raise RuntimeError("引擎输出张量数量不足（需要至少4个输出）")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        :param image: 输入图像（BGR格式）
        :return: 预处理后的张量（NCHW格式，RGB通道）
        """
        # 缩放图像
        resized = cv2.resize(image, self.input_dims)
        # BGR转RGB
        rgb_img = resized[:, :, ::-1]
        # 转换为NCHW格式并归一化（根据模型训练时的预处理调整）
        return np.ascontiguousarray(
            rgb_img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
        )

    def infer(self, input_data: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """
        执行推理
        :param input_data: 预处理后的输入张量
        :return: 输出张量列表 + 推理时间（ms）
        """
        if self.is_destroyed:
            raise RuntimeError("推理器已销毁，无法执行推理")

        # self.cfx.push()
        cuda_push()
        try:
            # 获取输入张量的形状，并设置到执行上下文中
            input_name = self.input_names[0]
            self.context.set_input_shape(input_name, input_data.shape)

            # 分配输入内存
            d_input = cuda.mem_alloc(input_data.nbytes)
            cuda.memcpy_htod_async(d_input, input_data, self.stream)

            # 分配输出内存
            output_buffers = []
            d_outputs = []
            bindings = [None] * (len(self.input_names) + len(self.output_names))
            bindings[0] = int(d_input)

            # 为每个输出张量分配内存
            for i, output_name in enumerate(self.output_names):
                output_idx = self.engine.get_binding_index(output_name)
                output_shape = tuple(self.context.get_binding_shape(output_idx))
                host_output = np.empty(output_shape, dtype=np.float32)
                d_output = cuda.mem_alloc(host_output.nbytes)
                output_buffers.append(host_output)
                d_outputs.append(d_output)
                bindings[i + 1] = int(d_output)

            # 执行推理
            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
            self.stream.synchronize()

            # 拷贝输出数据到主机
            for host_out, d_out in zip(output_buffers, d_outputs):
                cuda.memcpy_dtoh_async(host_out, d_out, self.stream)
            self.stream.synchronize()


        except Exception as e:
            print(f"推理过程中发生错误: {str(e)}")
            raise

        finally:
            # 推理完成后，弹出设备上下文，确保资源释放
            # self.cfx.pop()  # 释放设备上下文，防止上下文泄漏
            cuda_pop()
            # # 释放临时内存（避免内存泄漏）
            # del d_input
            # for d_out in d_outputs:
            #     del d_out

        return output_buffers

    def process_image(self, image: np.ndarray) -> Dict:
        """
        端到端处理单张图像
        :param image: 输入图像（BGR格式）
        :return: 处理结果字典（包含标注图像、结构化数据、推理时间）
        """
        if self.is_destroyed:
            raise RuntimeError("推理器已销毁，无法处理图像")
        # 预处理
        h, w = image.shape[0], image.shape[1]
        input_data = self.preprocess(image)
        # 推理
        outputs = self.infer(input_data)
        # 后处理
        predictions = post_process(
            torch.from_numpy(outputs[0]),
            torch.from_numpy(outputs[2]),
            torch.from_numpy(outputs[1]),
            torch.from_numpy(outputs[3]),
            (w, h)
        )
        # 结构化数据
        structured_data = process_results(predictions)
        return structured_data
       
    def destroy(self):
        """显式释放资源"""
        if not self.is_destroyed:
            if hasattr(self, 'context') and self.context:
                del self.context
                self.context = None
            if hasattr(self, 'engine') and self.engine:
                del self.engine
                self.engine = None
            if hasattr(self, 'stream') and self.stream:
                del self.stream
                self.stream = None
            if hasattr(self, 'logger') and self.logger:
                del self.logger
                self.logger = None
            # 清除内存绑定
            self.input_names = []
            self.output_names = []
            self.is_destroyed = True
            # 设置资源清理标志
            print("资源已释放")

    def __del__(self):
        """销毁对象时释放资源"""
        self.destroy()