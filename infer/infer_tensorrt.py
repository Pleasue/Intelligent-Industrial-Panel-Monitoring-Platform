import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os 
import time 
import torch 
from .post_process import post_process, TextVisualizer
from typing import Tuple, List, Dict, Optional

class TRTModelInference:
    """Tensorrt模型推理封装类"""
    def __init__(self, engine_path: str, input_dims: Tuple[int, int] = (640, 640)):
        self.engine_path = engine_path 
        self.input_dims = input_dims 
        self.engine = None 
        self.context = None 
        self.stream = cuda.Stream() # 创建流 （cuda异步） 
        self.logger = trt.Logger(trt.Logger.WARNING)

        # 初始化资源
        self._init_plugins()
        self._load_engine()
        self._prepare_bindings()
    
    def _init_plugins(self)->None:
        """初始化Tensorrt插件"""
        try:
            trt.init_libnvinfer_plugins(self.logger, "")
        except Exception as e:
            raise RuntimeError(f"插件初始化失败:{str(e)}")
    
    def _load_engine(self)->None:
        """加载引擎"""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"引擎文件不存在:{self.engine_path}")
        try:
            with open(self.engine_path, "rb") as f, \
                trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError("引擎加载失败")
            self.context = self.engine.create_execution_context()
        except Exception as e:
            raise RuntimeError(f"引擎加载错误:{str(e)}")
    
    def _prepare_bindings(self)->None:
        """准备输入输出绑定（使用新的Tensor API兼容TensorRT 8.4+）"""
        self.input_binding_idx = None
        self.output_binding_indices = []
        self.output_shapes = []
        self.input_names = []
        self.output_names = []
        
        # 遍历所有绑定名称（新API推荐方式）
        for name in self.engine:
            # 判断是否为输入张量（新API）
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_binding_idx = self.engine[name]  # 获取绑定索引
                self.input_names.append(name)
            else:
                self.output_binding_indices.append(self.engine[name])
                self.output_names.append(name)
                # 获取输出张量形状（新API）
                self.output_shapes.append(tuple(self.engine.get_tensor_shape(name)))
        
        if self.input_binding_idx is None:
            raise RuntimeError("未找到输入绑定")
    
    def preprocess(self, image: np.ndarray)->np.ndarray:
        """图像预处理"""
        resized = cv2.resize(image, self.input_dims)
        rgb_img = resized[:, :, ::-1]
        input_data = np.ascontiguousarray(
            rgb_img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
        )
        
        return input_data

    def infer(self, input_data: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """执行推理（兼容新旧版本API，修复TypeError）"""
        # 1. 设置输入形状（新API）
        input_name = self.input_names[0]
        self.context.set_input_shape(input_name, input_data.shape)

        # 2. 分配输入内存并拷贝数据到GPU
        d_input = cuda.mem_alloc(input_data.nbytes)
        cuda.memcpy_htod_async(d_input, input_data, self.stream)

        # 3. 分配输出内存并构建绑定列表（兼容旧API）
        host_outputs = []
        d_outputs = []
        bindings = [0] * (len(self.input_names) + len(self.output_names))  # 绑定列表

        # 绑定输入
        bindings[self.engine[input_name]] = int(d_input)

        # 绑定输出并分配内存
        for output_name in self.output_names:
            output_idx = self.engine[output_name]  # 获取输出张量索引
            output_shape = tuple(self.context.get_tensor_shape(output_name))
            host_out = np.empty(output_shape, dtype=np.float32)
            d_out = cuda.mem_alloc(host_out.nbytes)
            host_outputs.append(host_out)
            d_outputs.append(d_out)
            bindings[output_idx] = int(d_out)  # 绑定输出索引

        # 4. 执行推理（使用旧API的bindings参数，兼容所有版本）
        self.context.execute_async_v2(
            bindings=bindings,  # 显式传入绑定列表
            stream_handle=self.stream.handle
        )
        self.stream.synchronize()  # 等待GPU推理完成

        # 5. 从GPU拷贝输出数据到CPU
        for host_out, d_out in zip(host_outputs, d_outputs):
            cuda.memcpy_dtoh_async(host_out, d_out, self.stream)
        self.stream.synchronize()

        return host_outputs

    def postprocess(self, outputs: List[np.ndarray])->Tuple[np.ndarray, Dict]:
        """后处理"""
        # 转化为torch张量
        outputs = [np.ascontiguousarray(o.astype(np.float32)) for o in outputs]
        predictions = [torch.from_numpy(p) for p in outputs]
        # 后处理
        pr = post_process(
            predictions[0], 
            predictions[2], 
            predictions[1], 
            predictions[3], 
            self.input_dims
        )
        
        # 可视化
        visualizer = TextVisualizer()
        vis_output = visualizer.process_results(predictions=pr)
        # vis_image = vis_output.get_image()[:, :, ::-1]  # RGB转BGR
        
        # 整理预测结果为字典(方便前端解析)
        # result_data = self._format_result(pr)
        
        return vis_output   

    def _format_result(self, predictions) -> Dict:
        """格式化预测结果为字典"""
        # 根据实际post_process输出格式进行调整
        result = {
            "texts": [predictions["ctrl_points"].shape],
            "bboxes": [predictions["recs"].shape],
            "scores": [predictions["pred_classes"].shape],
        }
        return result 
    
    def process_image(self, image: np.ndarray)->Dict:
        """端到端处理图像"""
        input_data = self.preprocess(image)
        outputs = self.infer(input_data)
        outputs = self.postprocess(outputs)
        return outputs

    def __del__(self):
        """释放资源"""
        if hasattr(self, 'context') and self.context:
            del self.context
        if hasattr(self, 'engine') and self.engine:
            del self.engine

if __name__ == "__main__":
    # 初始化推理器
    inferencer = TRTModelInference(
        engine_path="model_fp16_win_2.trt",
        input_dims=(640, 640)
    )

    image_path = "images\\image_00310.jpg"
    image = cv2.imread(image_path)
    result = inferencer.process_image(image)
    print(result)