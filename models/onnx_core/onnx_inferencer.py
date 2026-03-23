import onnxruntime as ort
import numpy as np
import cv2
import torch
from .post_processor import post_process, process_results
from .config import ENGINE_PATH, INPUT_DIMS
import cv2
import numpy as np

class ONNXModelInferencer:
    """onnx模型推理器"""
    def __init__(self, engine_path: str = ENGINE_PATH, input_dims: tuple = INPUT_DIMS):

        self.input_dims = input_dims  # (width, height)   
        # =========================================
        # 初始化 ONNXRuntime 推理会话（FP16 优化）
        # =========================================
        providers = ["CUDAExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        self.session = ort.InferenceSession(engine_path, sess_options, providers=providers)
        # 获取输入输出节点名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
    
    def process_image(self, image: np.ndarray):
        frame_resized = cv2.resize(image, self.input_dims)
        original_image = frame_resized[:, :, ::-1]  # BGR → RGB

        # 转换为 [1, 3, H, W] numpy
        input_tensor = original_image.astype("float32").transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # =========================================
        # 推理
        # =========================================
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        outputs = [np.ascontiguousarray(o.astype(np.float32)) for o in outputs]
        # =========================================
        # 后处理与可视化
        # =========================================
        predictions = [torch.from_numpy(p) for p in outputs]


        # 后处理
        predictions = post_process(
            predictions[0], 
            predictions[1], 
            predictions[2], 
            predictions[3], 
            self.input_dims,
        )        
        structured_data = process_results(predictions)
        return structured_data