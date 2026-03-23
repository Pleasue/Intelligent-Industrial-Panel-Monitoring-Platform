import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any


# ---------------------- 数据模型（与前端对齐） ----------------------
class TemplateTarget:
    """模板中的单个目标数据结构（适配JSON序列化）"""
    def __init__(self, text: str, userLabel: str, upperLimit: float, lowerLimit: float, confidence: float, bbox: List[List[float]]):
        self.text = text
        self.userLabel = userLabel
        self.upperLimit = upperLimit
        self.lowerLimit = lowerLimit
        self.confidence = confidence
        self.bbox = bbox

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

class Template:
    """前端发送的完整模板数据结构"""
    def __init__(self, templateId: str, templateName: str, createTime: str, content: List[TemplateTarget]):
        self.templateId = templateId
        self.templateName = templateName
        self.createTime = createTime
        self.content = content  # 列表：TemplateTarget实例

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Template":
        """从前端JSON数据构建Template实例"""
        content = [
            TemplateTarget(
                text=item["text"],
                userLabel=item["userLabel"],
                upperLimit=item["upperLimit"],
                lowerLimit=item["lowerLimit"],
                confidence=item["confidence"],
                bbox=item["bbox"]
            ) for item in data["content"]
        ]
        return Template(
            templateId=data["templateId"],
            templateName=data["templateName"],
            createTime=data["createTime"],
            content=content
        )

# 工具函数：从字符串中提取第一个连续数值（支持整数、小数、负数）
def extract_first_number(s: str) -> Optional[float]:
    import re
    # 正则匹配：-?\d+(\.\d+)? → 匹配负数、整数、小数（连续数字部分）
    num_match = re.search(r'-?\d+(\.\d+)?', s)
    if num_match:
        try:
            return float(num_match.group())  # 提取并转为float
        except (ValueError, TypeError):
            return None
    return None  # 无匹配的连续数值

class TrackedTarget:
    """跟踪结果（后端→前端）"""
    def __init__(self, trackId: str, matchedTemplateTargetId: str, text: str, bbox: List[List[float]],
                 confidence: float, upperLimit: float, lowerLimit: float, userLabel: str,
                 isOverLimit: bool, similarity: float):
        self.trackId = trackId
        self.matchedTemplateTargetId = matchedTemplateTargetId
        self.text = text
        self.bbox = bbox
        self.confidence = confidence
        self.upperLimit = upperLimit
        self.lowerLimit = lowerLimit
        self.userLabel = userLabel
        self.isOverLimit = isOverLimit
        self.similarity = similarity

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

# ---------------------- 跟踪上下文类 ----------------------
# 移除 @dataclass 装饰器
class TrackContext:
    """单个跟踪会话的上下文（手动定义 __init__，避免参数缺失）"""
    def __init__(self, session_id: str, template: Template):
        self.session_id = session_id  # 跟踪会话唯一ID
        self.template = template  # 前端发送的模板
        self.track_cache = {}  # 跨帧跟踪缓存
        self.next_track_id = 1  # 下一个可用跟踪ID
        self.last_active_time = datetime.now()  # 最后活跃时间
        self.track_mode = "filtered"  # 跟踪模式：filtered/direct
        # 直接在 __init__ 中构建特征库，无需外部传入
        self.feature_library = self._build_feature_library()

    def __post_init__(self):
        """初始化时构建模板特征库"""
        self.feature_library = self._build_feature_library()

    def _build_feature_library(self) -> List[Dict[str, Any]]:
        """构建模板特征库：提取核心特征（文本、尺寸、中心位置）"""
        feature_lib = []
        for idx, target in enumerate(self.template.content):
            # 计算目标尺寸（宽、高）
            bbox = target.bbox
            width = max(p[0] for p in bbox) - min(p[0] for p in bbox)
            height = max(p[1] for p in bbox) - min(p[1] for p in bbox)
            
            # 计算目标中心坐标
            center_x = (min(p[0] for p in bbox) + max(p[0] for p in bbox)) / 2
            center_y = (min(p[1] for p in bbox) + max(p[1] for p in bbox)) / 2

            feature_lib.append({
                "template_target_id": f"temp_{self.template.templateId}_{idx}",
                "text": target.text.strip().lower(),  # 文本特征（统一小写）
                "size": (width, height),  # 尺寸特征
                "center": (center_x, center_y),  # 位置特征
                "upper_limit": target.upperLimit,
                "lower_limit": target.lowerLimit,
                "label": target.userLabel,
                "confidence_threshold": 0.5  # 匹配置信度阈值
            })
        print(feature_lib)
        return feature_lib

    def match_infer_result(self, infer_result: Dict[str, Any]) -> Optional[TrackedTarget]:
        """
        核心匹配逻辑：当前跟踪目标 → 消失未超时目标 → 新模板匹配
        匹配规则：尺寸(40%)+位置(60%)，目标消失保留最后位置，复用首次ID
        """
        # 预处理推理结果
        infer_text = infer_result.get("text", "").strip().lower()
        infer_bbox = infer_result.get("bbox", [])
        infer_conf = infer_result.get("confidence", 0.0)

        # 跳过低置信度结果
        if infer_conf < 0.3 or not infer_bbox:
            return None

        # 计算推理目标特征
        infer_width = max(p[0] for p in infer_bbox) - min(p[0] for p in infer_bbox) if infer_bbox else 0
        infer_height = max(p[1] for p in infer_bbox) - min(p[1] for p in infer_bbox) if infer_bbox else 0
        infer_size = (infer_width, infer_height)

        infer_center = (
            (min(p[0] for p in infer_bbox) + max(p[0] for p in infer_bbox)) / 2 if infer_bbox else 0,
            (min(p[1] for p in infer_bbox) + max(p[1] for p in infer_bbox)) / 2 if infer_bbox else 0
        )

        # 寻找最佳匹配
        best_match = None
        max_similarity = 0.0
        
        for template_feature in self.feature_library:
            # 尺寸相似度
            size_similarity = self._calc_size_similarity(infer_size, template_feature["size"])
            # 位置相似度
            pos_similarity = self._calc_pos_similarity(infer_center, template_feature["center"])
            # 总相似度
            total_similarity = size_similarity * 0.4 + pos_similarity * 0.6
            if total_similarity > max_similarity and total_similarity >= 0.8:
                max_similarity = total_similarity
                best_match = template_feature

        # 无匹配结果→丢弃
        if not best_match:
            return None

        # ---------------------- 关键修改：错误判断逻辑 ----------------------
        # is_error = False  # 新增：统一错误状态（涵盖超限和文本不匹配）
        # template_label = best_match["label"].strip().lower()  # 模板目标的label（统一小写）

        # try:
        #     # 情况1：推理结果是数字
        #     infer_num = float(infer_text)
        #     # 错误条件：数值超限 OR 文本≠模板label
        #     is_over_limit = infer_num < best_match["lower_limit"] or infer_num > best_match["upper_limit"]
        #     is_text_mismatch = infer_text != template_label
        #     is_error = is_over_limit or is_text_mismatch
        # except (ValueError, TypeError):
        #     # 情况2：推理结果是文本
        #     # 错误条件：文本≠模板label
        #     is_text_mismatch = infer_text != template_label
        #     is_error = is_text_mismatch

        is_error = False  # 统一错误状态（涵盖超限、无有效数值、文本不匹配）
        template_label = best_match["label"].strip().lower()  # 模板label（统一小写，兼容大小写）
        lower_limit = best_match["lower_limit"]
        upper_limit = best_match["upper_limit"]

         # 生成跟踪ID（复用或新增）
        track_id = self._get_or_create_track_id(best_match["template_target_id"], infer_center)
        # 核心判断逻辑
        extracted_num = extract_first_number(infer_text)  # 提取推理结果中的连续数值
        is_label_match = infer_text.strip().lower() == template_label  # 文本是否完全匹配（统一小写，忽略首尾空格）
        is_num_in_range = False

        # 检查提取的数值是否在上下限内
        if extracted_num is not None:
            is_num_in_range = lower_limit <= extracted_num <= upper_limit

        # 错误条件：两个条件都不满足（既不匹配label，也无有效数值在范围内）
        is_error = not (is_label_match or is_num_in_range)


        # 构建跟踪结果
        tracked_target = TrackedTarget(
            trackId=track_id,
            matchedTemplateTargetId=best_match["template_target_id"],
            text=infer_result.get("text", ""),
            bbox=infer_bbox,
            confidence=infer_conf,
            upperLimit=best_match["upper_limit"],
            lowerLimit=best_match["lower_limit"],
            userLabel=best_match["label"],
            isOverLimit=is_error,
            similarity=round(max_similarity, 2)
        )

        # 更新缓存
        self.track_cache[track_id] = tracked_target
        self.last_active_time = datetime.now()

        return tracked_target

    def _calc_size_similarity(self, infer_size: tuple, template_size: tuple) -> float:
        """计算尺寸相似度（0-1）"""
        infer_w, infer_h = infer_size
        temp_w, temp_h = template_size
        if temp_w == 0 or temp_h == 0 or infer_w == 0 or infer_h == 0:
            return 0.0
        w_ratio = min(infer_w / temp_w, temp_w / infer_w)
        h_ratio = min(infer_h / temp_h, temp_h / infer_h)
        return (w_ratio + h_ratio) / 2

    def _calc_pos_similarity(self, infer_center: tuple, template_center: tuple) -> float:
        """计算位置相似度（0-1，最大有效距离300px）"""
        max_dist = 300.0
        dist = math.hypot(infer_center[0] - template_center[0], infer_center[1] - template_center[1])
        return 1.0 - (dist / max_dist) if dist <= max_dist else 0.0

    def _get_or_create_track_id(self, template_target_id: str, infer_center: tuple) -> str:
        """复用已有跟踪ID（距离<50px视为同一目标）"""
        for track_id, target in self.track_cache.items():
            if target.matchedTemplateTargetId == template_target_id:
                target_center = (
                    (min(p[0] for p in target.bbox) + max(p[0] for p in target.bbox)) / 2,
                    (min(p[1] for p in target.bbox) + max(p[1] for p in target.bbox)) / 2
                )
                if math.hypot(infer_center[0] - target_center[0], infer_center[1] - target_center[1]) < 50:
                    return track_id
        # 新增ID
        new_track_id = f"trk_{self.next_track_id}"
        self.next_track_id += 1
        return new_track_id


# ---------------------- 新增：直接返回原始结果（无过滤） ----------------------
def direct_return_result(infer_result: Dict[str, Any]) -> TrackedTarget:
    """
    不做任何跟踪过滤处理，直接返回原始推理结果
    仅统一格式（添加trackId、默认状态），不丢弃任何结果
    """
    # 提取原始推理结果字段
    infer_text = infer_result.get("text", "").strip()
    infer_bbox = infer_result.get("bbox", [])
    infer_conf = infer_result.get("confidence", 0.0)

    # # 生成默认跟踪ID（按顺序递增，保证格式统一）
    # track_id = f"trk_{self.next_track_id}"
    # self.next_track_id += 1

    # 构建统一格式的跟踪结果（默认状态：未匹配、不超限）
    tracked_target = TrackedTarget(
        trackId=0,
        matchedTemplateTargetId="",  # 无匹配的模板目标ID
        text=infer_text,
        bbox=infer_bbox,
        confidence=infer_conf,
        upperLimit=0.0,  # 默认无上限
        lowerLimit=0.0,  # 默认无下限
        userLabel="原始目标",  # 默认标签
        isOverLimit=False,  # 无上下限，默认不超限
        similarity=1.0  # 直接返回模式，相似度默认1.0
    )

    # # 更新缓存（可选：保留原始结果缓存，便于后续切换模式）
    # self.track_cache[track_id] = tracked_target
    # self.last_active_time = datetime.now()

    return tracked_target