import torch
import onnx
import onnxruntime
import numpy as np
# from edgespotter.model import build_model  # 按你的实际导入路径修改
from edgespotter.onnx_model import SimpleONNXReadyModel

CONFIG = "configs/Base_det_export.yaml"
CHECKPOINT = "model_0012000.pth"
# 1️⃣ 初始化模型
device =  "cpu"
model = SimpleONNXReadyModel(CONFIG, CHECKPOINT).to(device)
model.eval()

# 2️⃣ 构造虚拟输入
dummy_input = torch.randn(1, 3, 640, 640).to(device)
DIMS = (640, 640)

# 3️⃣ 导出前修复控制流问题
def fix_tracing(model):
    """将不稳定的 tensor 控制流转换为常量或 Python int"""
    import torch.nn as nn

    def _safe_as_int(x):
        if isinstance(x, torch.Tensor):
            return int(x.item()) if x.numel() == 1 else [int(xx.item()) for xx in x]
        return x

    for name, module in model.named_modules():
        if hasattr(module, "spatial_shapes"):
            module.spatial_shapes = _safe_as_int(module.spatial_shapes)
    return model

model = fix_tracing(model)

# 4️⃣ 导出为 ONNX
input_names = ["input"]
output_names = ["pred_0", "pred_1", "pred_2", "pred_3"]
dynamic_axes = {"input": {0: "batch", 2: "height", 3: "width"}}

torch.onnx.export(
    model,
    dummy_input,
    "model_fp32_verified.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    verbose=False,
)

print("✅ 导出成功: model_fp32_verified.onnx")

# 5️⃣ PyTorch 基线输出
with torch.no_grad():
    torch_outputs = model(dummy_input)
    torch_outs = [t.detach().cpu().numpy() for t in torch_outputs]

# 6️⃣ ONNX 推理验证
session = onnxruntime.InferenceSession("model_fp32_verified.onnx", providers=["CPUExecutionProvider"])
onnx_inputs = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
onnx_outputs = session.run(None, onnx_inputs)

# 7️⃣ 输出差异验证
for i in range(len(torch_outs)):
    diff = np.mean(np.abs(torch_outs[i] - onnx_outputs[i]))
    print(f"🔹 Output {i}: mean abs diff = {diff:.6f}")

print("✅ ONNX vs PyTorch 验证完成（若差异 < 1e-3 则视为一致）")
