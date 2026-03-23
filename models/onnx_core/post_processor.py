import numpy as np
import torch
from .config import CT_LABELS
from collections import defaultdict


def post_process(ctrl_point_cls: torch.Tensor, ctrl_point_coord: torch.Tensor,
                 ctrl_point_text: torch.Tensor, bd_points: torch.Tensor,
                 image_size: tuple) -> dict:
    """
    模型输出后处理
    :param ctrl_point_cls: 控制点分类输出
    :param ctrl_point_coord: 控制点坐标输出
    :param ctrl_point_text: 文本解码输出
    :param bd_points: 边界框输出
    :param image_size: 图像尺寸（宽, 高）
    :return: 处理后的预测结果字典
    """
    from .config import CONF_THRESHOLD
    
    # print("cls", ctrl_point_cls.shape)
    # print("coord", ctrl_point_coord.shape)
    # print("text", ctrl_point_text.shape)
    # print("bd", bd_points.shape)
    # 计算置信度并过滤
    prob = ctrl_point_cls.mean(-2).sigmoid()
    scores, _ = prob.max(-1)
    ctrl_point_text = torch.softmax(ctrl_point_text, dim=-1)

    results = defaultdict()
    scores_per_img = scores[0]
    ctrl_coord = ctrl_point_coord[0]
    ctrl_text = ctrl_point_text[0]
    bd = bd_points[0]
    mask = scores_per_img >= CONF_THRESHOLD
    if not mask.any():
        return results

    # 坐标缩放（映射到原始图像尺寸）
    ctrl_coord = ctrl_coord[mask]
    ctrl_coord[..., 0] *= image_size[0]
    ctrl_coord[..., 1] *= image_size[1]

    # 文本解码（取概率最高的字符）
    _, text_pred = ctrl_text[mask].topk(1)
    # 边界框坐标缩放
    bd = bd[mask]
    bd[..., 0::2] *= image_size[0]
    bd[..., 1::2] *= image_size[1]

    # print(text_pred.shape)
    results["scores"] = scores_per_img[mask]
    results["ctrl_points"] = ctrl_coord.flatten(1)
    results["recs"] = text_pred.squeeze(-1)
    results["bd"] = bd
    return results


def ctc_decode_recognition(rec: torch.Tensor) -> str:
    """CTC解码文本"""
    last_char = '###'
    text = ''
    for c in rec:
        c = int(c)
        if 0 <= c < len(CT_LABELS):
            if last_char != c:
                text += CT_LABELS[c]
                last_char = c
        else:
            last_char = '###'
    return text

def process_results(predictions: dict) -> list:
    """将预测结果转换为结构化数据（新增外接矩形计算）"""
    if not predictions:
        return []
    # print(predictions)
    recs = predictions["recs"].cpu()
    scores = predictions["scores"].cpu().numpy().tolist()  # 置信度分数
    bd_pts = np.asarray(predictions["bd"].cpu())
    structured_data = []
    for idx in range(len(recs)):
        rec = recs[idx]
        score = scores[idx]
        bd = bd_pts[idx] if bd_pts.size > 0 else None

        # 构建结构化结果
        item = {"text": ctc_decode_recognition(rec),}

        # ========== 关键修改：多边形转外接矩形 ==========
        if bd is not None and bd.size > 0:
            bd_split = np.hsplit(bd, 2)
            bbox = np.vstack([bd_split[0], bd_split[1][::-1]])  # 闭合多边形坐标
            # 4. 保留原多边形框（可选，如需删除原多边形可注释）
            item["bbox"] = bbox         # 外接矩形（核心新增）
            item["conf"] = round(score, 4)
        else:
            item["text"] = None  # 原多边形框（可选）
            item["bbox"] = None     # 外接矩形
            item["conf"] = None

        structured_data.append(item)
    return structured_data