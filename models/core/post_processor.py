import numpy as np
import torch
from shapely.geometry import LineString
from .config import CT_LABELS
from collections import defaultdict

class VisImage:
    """图像可视化辅助类"""
    def __init__(self, img: np.ndarray, scale: float = 1.0):
        self.img = img.astype("uint8")
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure()

    def _setup_figure(self):
        import matplotlib as mpl
        import matplotlib.figure as mplfigure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        mpl.use('Agg')  # 非交互模式
        self.fig = mplfigure.Figure(frameon=False)
        self.dpi = self.fig.get_dpi()
        self.fig.set_size_inches(
            (self.width * self.scale) / self.dpi,
            (self.height * self.scale) / self.dpi
        )
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0])
        self.ax.axis("off")
        self.ax.imshow(self.img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def get_image(self) -> np.ndarray:
        """获取绘制后的图像（RGB格式）"""
        s, (width, height) = self.canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype="uint8")
        img_rgba = buffer.reshape(height, width, 4)
        return img_rgba[..., :3].astype("uint8")

class TextVisualizer:
    """文本检测可视化与结构化数据处理类"""
    def __init__(self, img_rgb: np.ndarray, scale: float = 1.0):
        self.img = np.asarray(img_rgb).clip(0, 255).astype("uint8")
        self.output = VisImage(self.img, scale)

    def draw_polygon(self, segment: np.ndarray, color: tuple = (0.4, 0.6, 0.8), alpha: float = 0.4):
        """绘制多边形边界框"""
        import matplotlib.patches as mpl_patches
        import matplotlib.colors as mplc
        polygon = mpl_patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=mplc.to_rgb(color) + (1,),
            linewidth=2
        )
        self.output.ax.add_patch(polygon)

    def draw_text(self, text: str, position: np.ndarray, color: tuple = (0.58, 0, 0.83)):
        """绘制文本标签"""
        self.output.ax.text(
            position[0], position[1], text,
            size=16 * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment="left",
            color=color,
            zorder=10
        )

    def draw_instance_predictions(self, predictions: dict) -> np.ndarray:
        """绘制检测结果，返回标注后的图像（RGB）"""
        if not predictions:
            return self.img

        # 提取预测结果
        ctrl_pnts = predictions["ctrl_points"].cpu().numpy() if hasattr(predictions["ctrl_points"], 'cpu') else predictions["ctrl_points"]
        scores = predictions["scores"].cpu().tolist() if hasattr(predictions["scores"], 'cpu') else predictions["scores"]
        recs = predictions["recs"].cpu() if hasattr(predictions["recs"], 'cpu') else predictions["recs"]
        bd_pts = np.asarray(predictions["bd"].cpu()) if hasattr(predictions["bd"], 'cpu') else np.asarray(predictions["bd"])

        for ctrl_pnt, score, rec, bd in zip(ctrl_pnts, scores, recs, bd_pts):
            # 绘制边界框
            if bd is not None and bd.size > 0:
                bd_split = np.hsplit(bd, 2)
                bd_poly = np.vstack([bd_split[0], bd_split[1][::-1]])  # 闭合多边形
                self.draw_polygon(bd_poly)

            # 解码文本并绘制
            text = self._ctc_decode_recognition(rec)
            text_pos = bd_poly[0] - np.array([0, 15]) if (bd is not None and bd.size > 0) else self._get_center_point(ctrl_pnt)
            self.draw_text(text, text_pos)

        return self.output.get_image()

    def process_results(self, predictions: dict) -> list:
        """将预测结果转换为结构化数据（新增外接矩形计算）"""
        if not predictions:
            return []
        
        recs = predictions["recs"].cpu() if hasattr(predictions["recs"], 'cpu') else predictions["recs"]
        bd_pts = np.asarray(predictions["bd"].cpu()) if hasattr(predictions["bd"], 'cpu') else np.asarray(predictions["bd"])

        structured_data = []
        for idx in range(len(recs)):
            rec = recs[idx]
            bd = bd_pts[idx] if bd_pts.size > 0 else None

            # 构建结构化结果
            item = {"text": self._ctc_decode_recognition(rec),}

            # ========== 关键修改：多边形转外接矩形 ==========
            if bd is not None and bd.size > 0:
                # 1. 解析多边形所有顶点（原逻辑是分割x/y坐标，需合并为(x,y)顶点）
                bd_split = np.hsplit(bd, 2)  # bd_split[0] = 所有x坐标，bd_split[1] = 所有y坐标
                polygon_points = np.vstack([bd_split[0], bd_split[1]]).T  # 转换为 (n, 2) 格式的顶点列表（n为多边形顶点数）
                
                # 2. 计算外接矩形的边界（最小/最大x、y）
                min_x = np.min(polygon_points[:, 0])
                max_x = np.max(polygon_points[:, 0])
                min_y = np.min(polygon_points[:, 1])
                max_y = np.max(polygon_points[:, 1])
                
                # 3. 定义外接矩形的四个顶点（顺时针顺序：左上、右上、右下、左下）
                bbox_rect = [
                    [min_x, min_y],  # 左上
                    [max_x, min_y],  # 右上
                    [max_x, max_y],  # 右下
                    [min_x, max_y]   # 左下
                ]
                
                # 4. 保留原多边形框（可选，如需删除原多边形可注释）
                item["bbox"] = bbox_rect          # 外接矩形（核心新增）
            else:
                item["text"] = None  # 原多边形框（可选）
                item["bbox"] = None     # 外接矩形

            structured_data.append(item)
        return structured_data

    @staticmethod
    def _get_center_point(ctrl_pnt: np.ndarray) -> np.ndarray:
        """计算控制点的中心点"""
        line = ctrl_pnt.reshape(-1, 2)
        return np.array(LineString(line).interpolate(0.5, normalized=True).coords[0], dtype=np.int32)

    @staticmethod
    def _ctc_decode_recognition(rec: torch.Tensor) -> str:
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
            # 1. 解析多边形所有顶点（原逻辑是分割x/y坐标，需合并为(x,y)顶点）
            bd_split = np.hsplit(bd, 2)  # bd_split[0] = 所有x坐标，bd_split[1] = 所有y坐标
            polygon_points = np.vstack([bd_split[0], bd_split[1]]).T  # 转换为 (n, 2) 格式的顶点列表（n为多边形顶点数）
            
            # 2. 计算外接矩形的边界（最小/最大x、y）
            min_x = np.min(polygon_points[:, 0])
            max_x = np.max(polygon_points[:, 0])
            min_y = np.min(polygon_points[:, 1])
            max_y = np.max(polygon_points[:, 1])
            
            # 3. 定义外接矩形的四个顶点（顺时针顺序：左上、右上、右下、左下）
            bbox_rect = [
                [min_x, min_y],  # 左上
                [max_x, min_y],  # 右上
                [max_x, max_y],  # 右下
                [min_x, max_y]   # 左下
            ]
            
            # 4. 保留原多边形框（可选，如需删除原多边形可注释）
            item["bbox"] = bbox_rect          # 外接矩形（核心新增）
            item["conf"] = round(score, 4)
        else:
            item["text"] = None  # 原多边形框（可选）
            item["bbox"] = None     # 外接矩形
            item["conf"] = None

        structured_data.append(item)
    return structured_data