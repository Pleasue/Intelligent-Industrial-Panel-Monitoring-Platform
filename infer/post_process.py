import torch
from typing import List, Optional, Dict

import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.font_manager as mfm

from shapely.geometry import LineString

import numpy as np

class VisImage:
    def __init__(self, img, scale=1.0):

        self.img = img.astype("uint8")
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
       
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath):
        self.fig.savefig(filepath)

    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class TextVisualizer():
    def __init__(self, img_rgb=None, scale=1.0,):
        if img_rgb is not None:
            self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
            self.output = VisImage(self.img, scale)
        self._default_font_size = 16
        self.voc_size = 96
        self.CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']

    def process_results(self, predictions) -> List[Dict]:
        """
        纯数据处理：仅解析ctrl_pnts、scores、recs、bd_pts，不绘制图像
        返回结构化的检测结果列表
        
        Args:
            predictions: 模型输出的预测字典（包含ctrl_points、scores、recs、bd）
            
        Returns:
            List[Dict]: 每个元素包含单个检测结果的结构化数据
                - text: 解码后的文本
                - score: 置信度分数
                - ctrl_points: 控制点坐标 (N, 2)
                - bbox: 边界框坐标 (M, 2)（闭合多边形）
                - center_point: 控制点中点坐标 (2,)
        """
        # 提取核心数据并转换为numpy
        ctrl_pnts = predictions["ctrl_points"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy().tolist()  # 置信度分数
        recs = predictions["recs"].cpu()
        bd_pts = np.asarray(predictions["bd"].cpu())

        processed_results = []
        # 逐个处理每个检测结果
        for idx in range(len(ctrl_pnts)):
            ctrl_pnt = ctrl_pnts[idx]
            score = scores[idx]
            rec = recs[idx]
            bd = bd_pts[idx] if bd_pts.size > 0 else None

            # 解码文本
            text = self._ctc_decode_recognition(rec)
            
            # 处理控制点（reshape为(N, 2)格式）
            ctrl_points = ctrl_pnt.reshape(-1, 2).tolist()
            
            # 处理边界框（构建闭合多边形）
            bbox = None
            if bd is not None and bd.size > 0:
                bd_split = np.hsplit(bd, 2)
                bbox = np.vstack([bd_split[0], bd_split[1][::-1]]).tolist()  # 闭合多边形坐标
            
            # 计算控制点中点
            # center_point = self._get_center_point(ctrl_pnt).tolist()

            # 组装结构化结果
            processed_results.append({
                "text": text,
                "score": round(score, 4),  # 保留4位小数
                "ctrl_points": ctrl_points,
                "bbox": bbox,
                # "center_point": center_point
            })

        return processed_results

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
            if edge_color is None:
                # make edge color darker than the polygon color
                edge_color = color
            edge_color = mplc.to_rgb(edge_color) + (1,)

            polygon = mpl.patches.Polygon(
                segment,
                fill=True,
                facecolor=mplc.to_rgb(color) + (alpha,),
                edgecolor=edge_color,
                linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
            )
            self.output.ax.add_patch(polygon)
            return self.output

    def draw_instance_predictions(self, predictions):
        ctrl_pnts = predictions["ctrl_points"].cpu().numpy()
        scores = predictions["scores"].cpu().tolist()
        recs = predictions["recs"].cpu()
        bd_pts = np.asarray(predictions["bd"].cpu())
        self.overlay_instances(ctrl_pnts, scores, recs, bd_pts)

        return self.output

    def _process_ctrl_pnt(self, pnt):
        points = pnt.reshape(-1, 2)
        return points


    def _ctc_decode_recognition(self, rec):
        last_char = '###'
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    if self.voc_size == 37 or self.voc_size == 96:
                        s += self.CTLABELS[c]
                        last_char = c
                    else:
                        s += str(chr(self.CTLABELS[c]))
                        last_char = c
            else:
                last_char = '###'
        return s

    def overlay_instances(self, ctrl_pnts, scores, recs, bd_pnts, alpha=0.4):
        color = (100/255, 150/255, 200/255)
        for ctrl_pnt, score, rec, bd in zip(ctrl_pnts, scores, recs, bd_pnts):

            scores.append([score])
            # draw polygons
            if bd is not None:
                bd = np.hsplit(bd, 2)
                bd = np.vstack([bd[0], bd[1][::-1]])
                self.draw_polygon(bd, color, alpha=alpha)

            # draw center lines
            line = self._process_ctrl_pnt(ctrl_pnt)
            line_ = LineString(line)
            center_point = np.array(line_.interpolate(0.5, normalized=True).coords[0], dtype=np.int32)

            # draw text
            text = self._ctc_decode_recognition(rec)
            if self.voc_size == 37:
                text = text.upper()
            text = "{}".format(text)
            if bd is not None:
                text_pos = bd[0] - np.array([0,15])
            else:
                text_pos = center_point
            horiz_align = "left"
            font_size = self._default_font_size
            self.draw_text(
                        text,
                        text_pos,
                        color= (148/255, 0, 211/255),
                        horizontal_alignment=horiz_align,
                        font_size=font_size,
                        draw_chinese=False if self.voc_size == 37 or self.voc_size == 96 else True
                )
    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
        draw_chinese=False
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW
        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size
        
        x, y = position
        if draw_chinese:
            font_path = "./simsun.ttc"
            prop = mfm.FontProperties(fname=font_path)
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
                fontproperties=prop
            )
        else:
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
            )
        return self.output



def detector_postprocess(
    results: List[Dict], 
    output_height: int, 
    output_width: int, 
    min_size: Optional[int] = None, 
    max_size: Optional[int] = None
) -> List[Dict]:
    """
    Scale and align the detection results.
    """
    if min_size and max_size:
        # Eliminate the padding influence for ViTAE backbone results
        size = min_size * 1.0
        scale_img_size = min_size / min(output_width, output_height)
        if output_height < output_width:
            newh, neww = size, scale_img_size * output_width
        else:
            newh, neww = scale_img_size * output_height, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        scale_x, scale_y = (output_width / neww, output_height / newh)
    else:
        # Assuming results is a list of dictionaries, each containing 'image_size'
        image_size = results[0].get("image_size", (output_height, output_width))
        scale_x, scale_y = (output_width / image_size[1], output_height / image_size[0])

    # Scale points
    for result in results:
        if "ctrl_points" in result:
            ctrl_points = result["ctrl_points"]
            ctrl_points[:, 0::2] *= scale_x
            ctrl_points[:, 1::2] *= scale_y

        if "bd" in result and not isinstance(result["bd"], list):
            bd = result["bd"]
            bd[..., 0::2] *= scale_x
            bd[..., 1::2] *= scale_y

    return results

def post_process(
            ctrl_point_cls,
            ctrl_point_coord,
            ctrl_point_text,
            bd_points,
            image_sizes,
    ):
        ctrl_point_text = torch.softmax(ctrl_point_text, dim=-1)
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)
        
        result = {}
        selector = scores[0] >= 0.4
        scores_per_image = scores[0][selector]
        labels_per_image = labels[0][selector]
        ctrl_point_per_image = ctrl_point_coord[0][selector]
        ctrl_point_text_per_image = ctrl_point_text[0][selector]
        bd_points_per_image = bd_points[0][selector]

        result["height"] = image_sizes[1]
        result["width"] = image_sizes[0]
        result["scores"] = scores_per_image
        result["pred_classes"] = labels_per_image
        result["rec_scores"] = ctrl_point_text_per_image
        ctrl_point_per_image[..., 0] *= image_sizes[0]
        ctrl_point_per_image[..., 1] *= image_sizes[1]
        result["ctrl_points"] = ctrl_point_per_image.flatten(1)
        _, text_pred = ctrl_point_text_per_image.topk(1)
        result["recs"] = text_pred.squeeze(-1)
        bd_points_per_image[..., 0::2] *= image_sizes[0]
        bd_points_per_image[..., 1::2] *= image_sizes[1]
        result["bd"] = bd_points_per_image    
        
        return result


    
    
