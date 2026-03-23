import onnxruntime as ort
import numpy as np
import cv2
import torch
import time
import os
from post_process import post_process, TextVisualizer

import cv2
import numpy as np

def draw_detection_results(frame, processed_results):
    """
    在图像上绘制多边形检测框和文本内容（bbox 是闭合多边形坐标）
    :param frame: 输入图像（numpy.ndarray，OpenCV 读取的帧）
    :param processed_results: 检测结果列表（bbox 为多边形坐标）
    :return: 绘制后的图像
    """
    # 复制原图像（避免修改原图）
    img = frame.copy()
    
    # 定义绘制样式（可自定义）
    poly_color = (0, 255, 0)  # 多边形检测框颜色：绿色（BGR）
    poly_thickness = 2  # 多边形线宽
    text_color = (255, 255, 255)  # 文本颜色：白色
    text_bg_color = (0, 0, 255)  # 文本背景色：红色
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
    font_scale = 0.6  # 字体大小
    text_thickness = 1  # 文本线宽
    text_padding = 3  # 文本背景 padding

    # 遍历所有检测结果
    for result in processed_results:
        # 1. 提取关键数据（容错处理）
        text = result.get("text", "未知目标")
        score = result.get("score", 0.0)
        bbox = result.get("bbox", [])  # bbox：多边形坐标列表，如 [[x0,y0], [x1,y1], ...]
        ctrl_points = result.get("ctrl_points", [])  # 文本多边形（可选）

        # 跳过无效 bbox（顶点数 <3 无法构成多边形）
        if not bbox or len(bbox) < 3:
            continue

        # 2. 处理多边形 bbox：转为 numpy 整数数组（OpenCV 绘图要求）
        # bbox 格式：[[x0,y0], [x1,y1], ...] → 转为 (n,1,2) 维度
        poly_pts = np.array(bbox, dtype=np.int32)
        poly_pts = poly_pts.reshape((-1, 1, 2))  # 必须是 (顶点数, 1, 2) 格式

        # 3. 绘制多边形检测框（闭合）
        cv2.polylines(
            img,
            [poly_pts],  # 多边形顶点数组（注意外层是列表）
            isClosed=True,  # 闭合多边形
            color=poly_color,
            thickness=poly_thickness
        )

        # 4. 计算文本位置：基于多边形的左上角极值点
        # 提取所有顶点的 x 和 y 坐标
        poly_x = poly_pts[:, 0, 0]  # 所有顶点的 x 坐标
        poly_y = poly_pts[:, 0, 1]  # 所有顶点的 y 坐标
        # 左上角极值点：x 最小 + y 最小（文本放在该点上方）
        min_x = np.min(poly_x)
        min_y = np.min(poly_y)
        text_anchor_x = min_x  # 文本 x 锚点（和多边形左边界对齐）
        text_anchor_y = min_y  # 文本 y 锚点（多边形上边界）

        # 5. 绘制文本（含置信度）
        display_text = f"{text} ({score:.2f})"
        # 计算文本尺寸（宽、高）
        text_size, _ = cv2.getTextSize(display_text, font, font_scale, text_thickness)
        text_w, text_h = text_size

        # 文本背景框坐标（避免超出图像边界）
        bg_x1 = text_anchor_x
        bg_y1 = max(text_anchor_y - text_h - 2 * text_padding, 0)  # 上移文本，避免覆盖多边形
        bg_x2 = text_anchor_x + text_w + 2 * text_padding
        bg_y2 = text_anchor_y

        # 绘制文本背景框（填充）
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), text_bg_color, -1)

        # 绘制文本（居中对齐背景框）
        text_x = bg_x1 + text_padding
        text_y = bg_y1 + text_h + text_padding  # 文本 baseline 位置
        cv2.putText(
            img, display_text, (text_x, text_y),
            font, font_scale, text_color, text_thickness
        )

        # 6. 可选：绘制文本区域的多边形（ctrl_points）
        if ctrl_points and len(ctrl_points) >= 3:
            ctrl_pts = np.array(ctrl_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                img, [ctrl_pts], isClosed=True,
                color=(255, 0, 0),  # 蓝色
                thickness=1
            )

    return img

# =========================================
# 配置路径
# =========================================
ONNX_MODEL_PATH = "model_fp32_720.onnx"   # 可改成 model_fp32.onnx 或 model_int8.onnx
# ONNX_MODEL_PATH = "torch_model_fp32.onnx"
IMAGE_PATH = "image_0001.jpg"                       # 文件夹路径
DIMS = (1280, 720)

# =========================================
# 初始化 ONNXRuntime 推理会话（FP16 优化）
# =========================================
providers = ["CUDAExecutionProvider"]

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# sess_options.log_severity_level = 1  # 显示详细日志
# sess_options.log_verbosity_level = 10

session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=providers)

# 获取输入输出节点名称
input_name = session.get_inputs()[0].name
output_names = [out.name for out in session.get_outputs()]

print(f"✅ Loaded ONNX model: {ONNX_MODEL_PATH}")
print(f"🟢 Providers: {session.get_providers()}")
print(f"🔹 Input: {input_name}")
print(f"🔹 Outputs: {output_names}")


frame = cv2.imread(IMAGE_PATH)
frame_resized = cv2.resize(frame, DIMS)
original_image = frame_resized[:, :, ::-1]  # BGR → RGB

# 转换为 [1, 3, H, W] numpy
input_tensor = original_image.astype("float32").transpose(2, 0, 1)
input_tensor = np.expand_dims(input_tensor, axis=0)

# =========================================
# 推理
# =========================================
t0 = time.time()
outputs = session.run(output_names, {input_name: input_tensor})
t1 = time.time()
print(f"{IMAGE_PATH} ⏱ 推理耗时: {(t1 - t0)*1000:.2f} ms")


outputs = [np.ascontiguousarray(o.astype(np.float32)) for o in outputs]

# for i, out in enumerate(outputs):
#     print(f"[TensorRT] Output {i}: shape={out.shape}, dtype={out.dtype}, min={out.min()}, max={out.max()}")

# =========================================
# 后处理与可视化
# =========================================
predictions = [torch.from_numpy(p) for p in outputs]


# 后处理
pr = post_process(
    predictions[0], 
    predictions[1], 
    predictions[2], 
    predictions[3], 
    DIMS,
)

# 可视化
visualizer = TextVisualizer()
vis_output = visualizer.process_results(predictions=pr)
out_img = draw_detection_results(frame, vis_output)
# 本地调试：显示标注后的图像
cv2.imshow("Polygon Detection", out_img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
