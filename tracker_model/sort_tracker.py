import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# 卡尔曼滤波器定义（针对目标状态：x, y, w, h, vx, vy）
class KalmanBoxTracker:
    """
    单目标卡尔曼跟踪器，维护单个目标的状态估计
    状态向量：[x, y, w, h, vx, vy]（中心坐标、宽高、x/y方向速度）
    """
    count = 0  # 全局跟踪ID计数器

    def __init__(self, bbox):
        """
        初始化卡尔曼滤波器
        :param bbox: 初始检测框 [x1, y1, x2, y2]
        """
        # 初始化卡尔曼滤波器（6维状态，4维观测）
        self.kf = KalmanFilter(dim_x=6, dim_z=4)
        
        # 状态转移矩阵 F（匀速运动模型）
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 观测矩阵 H（仅观测x, y, w, h）
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        
        # 过程噪声协方差 Q（控制运动模型的不确定性）
        self.kf.Q *= 0.01
        # 观测噪声协方差 R（控制检测框的不确定性）
        self.kf.R = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 10]
        ])
        
        # 初始状态（将bbox转换为中心坐标+宽高）
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        self.kf.x = np.array([cx, cy, w, h, 0, 0])  # 初始速度为0
        
        # 状态协方差矩阵 P（初始不确定性）
        self.kf.P[4:, 4:] *= 1000.0  # 速度的初始不确定性大
        self.kf.P *= 10.0
        
        # 跟踪器属性
        self.id = KalmanBoxTracker.count  # 唯一ID
        KalmanBoxTracker.count += 1
        self.hits = 1  # 匹配成功次数
        self.age = 1  # 存活帧数
        self.time_since_update = 0  # 距离上次更新的帧数

    def update(self, bbox):
        """
        用新的检测框更新卡尔曼滤波器
        :param bbox: 新检测框 [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.hits += 1
        self.age += 1
        
        # 转换检测框为中心坐标+宽高
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        z = np.array([cx, cy, w, h])
        
        # 更新卡尔曼滤波器
        self.kf.update(z)

    def predict(self):
        """
        预测下一帧的目标状态
        :return: 预测的检测框 [x1, y1, x2, y2]
        """
        # 预测状态
        if (self.kf.x[4] + self.kf.x[0]) < 0:
            self.kf.x[4] = 0
        if (self.kf.x[5] + self.kf.x[1]) < 0:
            self.kf.x[5] = 0
        self.kf.predict()
        
        self.age += 1
        self.time_since_update += 1
        
        # 将预测的中心坐标+宽高转换为bbox
        cx, cy, w, h = self.kf.x[:4]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return [x1, y1, x2, y2]

    def get_state(self):
        """
        获取当前目标状态（bbox）
        :return: [x1, y1, x2, y2]
        """
        cx, cy, w, h = self.kf.x[:4]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return [x1, y1, x2, y2]

# SORT跟踪器主类
class SORT:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        初始化SORT多目标跟踪器
        :param max_age: 最大未更新帧数（超过则删除跟踪器）
        :param min_hits: 最小匹配次数（达到才显示跟踪ID）
        :param iou_threshold: IoU匹配阈值
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []  # 活跃跟踪器列表

    def update(self, dets):
        """
        用新的检测结果更新跟踪器
        :param dets: 检测框列表，格式 [[x1, y1, x2, y2], ...]
        :return: 跟踪结果，格式 [[x1, y1, x2, y2, track_id], ...]
        """
        # 步骤1：预测所有活跃跟踪器的状态
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # 删除无效跟踪器
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # 步骤2：匈牙利算法匹配检测框与跟踪器（IoU作为代价）
        matched, unmatched_dets, unmatched_trks = self._match(dets, trks)
        
        # 步骤3：更新匹配成功的跟踪器
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])
        
        # 步骤4：为未匹配的检测框创建新跟踪器
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)
        
        # 步骤5：筛选有效跟踪结果（删除过期/未稳定跟踪器）
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            pos = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hits >= self.min_hits or self.min_hits == 0):
                ret.append(np.concatenate((pos, [trk.id])).reshape(1, -1))
            # 删除超过最大未更新帧数的跟踪器
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def _match(self, dets, trks):
        """
        匈牙利算法匹配检测框与跟踪器
        :param dets: 检测框列表 (N,4)
        :param trks: 跟踪器预测框列表 (M,4)
        :return: matched, unmatched_dets, unmatched_trks
        """
        if len(trks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.empty((0, 1), dtype=int)
        
        # 计算IoU代价矩阵
        iou_matrix = self._iou_batch(dets, trks)
        
        # 匈牙利算法求解最优匹配
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(row_ind, col_ind)))
        
        # 筛选匹配成功的对（IoU >= 阈值）
        matched = []
        unmatched_dets = []
        for d in range(len(dets)):
            if d not in matched_indices[:, 0]:
                unmatched_dets.append(d)
        unmatched_trks = []
        for t in range(len(trks)):
            if t not in matched_indices[:, 1]:
                unmatched_trks.append(t)
        
        # 过滤低IoU匹配
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matched.append(m.reshape(1, 2))
        
        if len(matched) == 0:
            matched = np.empty((0, 2), dtype=int)
        else:
            matched = np.concatenate(matched, axis=0)
        
        return matched, np.array(unmatched_dets), np.array(unmatched_trks)

    def _iou_batch(self, dets, trks):
        """
        批量计算检测框与跟踪框的IoU
        :param dets: (N,4) 检测框
        :param trks: (M,4) 跟踪框
        :return: IoU矩阵 (N,M)
        """
        dets = np.asarray(dets)
        trks = np.asarray(trks)
        
        # 计算交集坐标
        ix1 = np.maximum(dets[:, 0][:, None], trks[:, 0])
        iy1 = np.maximum(dets[:, 1][:, None], trks[:, 1])
        ix2 = np.minimum(dets[:, 2][:, None], trks[:, 2])
        iy2 = np.minimum(dets[:, 3][:, None], trks[:, 3])
        
        # 计算交集面积
        iw = np.maximum(ix2 - ix1 + 1, 0.0)
        ih = np.maximum(iy2 - iy1 + 1, 0.0)
        inters = iw * ih
        
        # 计算并集面积
        dets_area = (dets[:, 2] - dets[:, 0] + 1) * (dets[:, 3] - dets[:, 1] + 1)
        trks_area = (trks[:, 2] - trks[:, 0] + 1) * (trks[:, 3] - trks[:, 1] + 1)
        union = dets_area[:, None] + trks_area - inters
        
        # 计算IoU
        iou = inters / union
        return iou

# ---------------------- 测试代码（工业面板字符跟踪示例） ----------------------
def test_sort_tracker():
    """
    测试SORT跟踪器：模拟工业面板字符检测+跟踪
    """
    # 初始化SORT跟踪器
    sort_tracker = SORT(max_age=5, min_hits=2, iou_threshold=0.2)
    
    # 模拟工业面板视频帧（生成3帧测试数据）
    # 每帧的检测框：模拟字符位置移动
    test_frames = [
        # 第1帧：2个字符检测框
        [[50, 50, 80, 80], [100, 60, 130, 90]],
        # 第2帧：字符轻微移动
        [[52, 51, 82, 81], [101, 61, 131, 91]],
        # 第3帧：字符继续移动
        [[54, 52, 84, 82], [102, 62, 132, 92]]
    ]
    
    # 逐帧处理
    for frame_idx, dets in enumerate(test_frames):
        print(f"\n===== 第{frame_idx+1}帧 =====")
        # 更新跟踪器
        track_results = sort_tracker.update(dets)
        
        # 打印跟踪结果
        print("检测框：", dets)
        print("跟踪结果（x1,y1,x2,y2,track_id）：")
        for res in track_results:
            x1, y1, x2, y2, track_id = res
            print(f"  ID-{int(track_id)}: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

if __name__ == "__main__":
    # 安装依赖：pip install numpy opencv-python scipy filterpy
    test_sort_tracker()