import time 
import torch
from post_process import post_prosses, TextVisualizer
import cv2
import torch.onnx
import cv2

import onnxruntime as ort
import numpy as np

# 查看可用的执行 providers（确认GPU是否被支持）
print("可用执行 providers:", ort.get_available_providers())
# 输出应包含 ['CUDAExecutionProvider', 'CPUExecutionProvider'] 表示GPU可用

# 创建推理会话，指定使用CUDA执行（优先GPU）
session = ort.InferenceSession(
    "model//model_fp32_simplified.onnx",  # ONNX模型路径
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]  # 优先用GPU，失败则 fallback 到CPU
)

# 获取输入节点信息（名称、形状、数据类型）
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # 可能包含动态维度（如-1）
input_dtype = session.get_inputs()[0].type
print(input_shape, input_dtype)
# 获取输出节点信息
output_name = session.get_outputs()[0].name
output_shape = session.get_outputs()[0].shape
print(output_shape)

DIMS = (640, 640)
frame = cv2.imread("image_0143.jpg")

frame_rgb = cv2.resize(frame, DIMS)

original_image = frame_rgb[:, :, ::-1]
frame_rgb = original_image.astype("float32").transpose(2, 0, 1)

t = time.time()
# 以字典形式传入输入数据（key为输入节点名称，value为数据）
predictions = session.run(
    [output_name],  # 需返回的输出节点名称列表（可省略，默认返回所有输出）
    {input_name: frame_rgb}
)
frame_rgb = torch.as_tensor(frame_rgb)
# 输出结果为列表，对应请求的输出节点
result = predictions[0]
print("推理结果形状:", result.shape)
pr = post_prosses(predictions[0], predictions[1], predictions[2], predictions[3], DIMS)
frame_rgb = frame_rgb.permute(1, 2, 0)
visualizer = TextVisualizer(frame_rgb)
vis_output = visualizer.draw_instance_predictions(predictions=pr)
out_img = vis_output.get_image()[:, :, ::-1]
out = cv2.resize(out_img, (960, 540))
cv2.imshow('Camera', out)
cv2.waitKey (0) 
cv2.destroyAllWindows()
