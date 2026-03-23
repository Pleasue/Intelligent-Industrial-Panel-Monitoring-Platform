import onnxruntime as ort
import torch
from detectron2.structures import ImageList, Instances
from detectron2.data.detection_utils import read_image
from post_process import post_prosses
from edgespotter.onnx_model import SimpleONNXReadyModel
MODEL_PATH = 'model_infer.onnx'

# 使用 CUDA 加速
providers = ['CUDAExecutionProvider']
sess = ort.InferenceSession(MODEL_PATH, providers=providers)

import numpy as np
import cv2
import time

IMAGE_PATH = "image_0098.jpg"
# img = read_image(IMAGE_PATH, format="BGR")
input_name = sess.get_inputs()[0].name
print(input_name)
DIMS = [960, 540]

# 读取并预处理图像
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (960, 540))  # 注意宽度和高度的顺序
original_image = img[:, :, ::-1]  # BGR to RGB
img = original_image.astype("float32").transpose(2, 0, 1)  # HWC to CHW

# 使用onnx推理
inputs = {input_name: img}
output_names = [o.name for o in sess.get_outputs()]
print(output_names)
output = sess.run(output_names, inputs)
print("output", output[0])
# 使用模型推理
CHECKPOINT = "model_0007999.pth"
CONFIG = "configs/Base_det_export.yaml"#"/content/drive/MyDrive/deepsolo/Hebrew_colab_det.yaml"
model = SimpleONNXReadyModel(CONFIG, CHECKPOINT)
predictions = model(torch.as_tensor(img))
print("predictions", predictions[0])
# print(output)
# print('weights predicts: ', predictions[0].detach().cpu().numpy())
# print('onnx prediction: ', output[0])
# print('\nCorrect!' if (output[0].argmax(axis=2)[0] == np.array(predictions[0].detach().cpu())).all() else "Error!")
r = post_prosses(torch.as_tensor(output[0]), torch.as_tensor(output[1]), torch.as_tensor(output[2]), torch.as_tensor(output[3]), DIMS)
print(r)