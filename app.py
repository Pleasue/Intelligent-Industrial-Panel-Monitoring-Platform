from flask import Flask, render_template, Response, jsonify, request
import numpy as np
import cv2
import threading
import queue
import atexit
import uuid
from typing import List, Dict, Optional

from models.load_model import GenModel, PanelModel
from tracker_model.tracker import TrackContext, Template, direct_return_result

import time
from datetime import datetime, timedelta 

import os
import base64

# from models.core.cuda_ctx import cuda_push, cuda_pop
# ======================== Flask 服务初始化 ========================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 支持中文输出


# ======================== 全局变量 ========================
cap = None  # 摄像头对象
inferencer = None  # 推理器
latest_results = []  # 最新检测结果缓存
_last_results = []  # 缓存上一帧

# camera_on = False
spotting_on = False  # 字符识别功能开关（默认关闭）

# 摄像头流
evt = threading.Event() # 控制当前流是否继续
cap_lock = threading.Lock()  # 线程锁（保证线程安全）
is_shutting_down = False  # 标记程序是否正在关闭

# 字符识别流
frame_queue = queue.Queue(maxsize=3) # 只保留3帧，防止爆内存
result_queue = queue.Queue(maxsize=1) # 只保留最新结果
infer_thread = None
infer_evt = threading.Event()

# 默认绘制开关
draw_on = False

# 本地视频流
UPLOAD_FOLDER = "uploads" # 存放上传视频
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 跟踪会话缓存（键：sessionId，值：跟踪上下文）
track_on = False # 跟踪功能开关（默认关闭）
cleanup_thread = None
track_sessions: Dict[str, "TrackContext"] = {}
track_lock = threading.Lock()  # 跟踪会话操作锁

# ======================== 模型工厂 ===========================
MODELS = {
    "PANEL": {"loader": lambda: PanelModel(), "name": "面板模型"},
    "GENERAL": {"loader": lambda: GenModel(), "name": "通用模型"},
    # "GENERAL": {"loader": lambda: }
}
current_key = "PANEL" # 默认模型
load_lock =  threading.RLock()

# ======================== 资源清理函数 ========================
def cleanup_all_resources():
    """程序退出时清理所有资源"""
    global cap, inferencer, is_shutting_down
    if is_shutting_down:
        return
    is_shutting_down = True

    print("\n📤 程序正在退出，清理资源中...")
    # 1. 停止摄像头
    with cap_lock:
        if cap and cap.isOpened():
            cap.release()
            cap = None
            print("✅ 摄像头资源已清理")
    # cuda_push()
    # 2. 销毁推理器
    if inferencer is not None:
        inferencer.destroy()
        inferencer = None
    # cuda_pop()
    # _release_cuda_ctx()
    print("✅ 推理器资源已清理")

# 注册程序退出时的清理函数
atexit.register(cleanup_all_resources)
    

# ======================== 初始化核心组件 ========================
def init_core_components():
    """初始化CUDA上下文和推理器"""
    global inferencer
    # 初始化推理器
    try:
        # edgespotter模型推理
        inferencer = MODELS[current_key]["loader"]() # 当前实例
        # inferencer = TRTModelInferencer()
    except Exception as e:
        print(f"❌ 推理器初始化失败: {str(e)}")
        # 初始化失败时清理资源
        cleanup_all_resources()

# 服务启动时初始化核心组件
init_core_components()

# ======================== 前端页面接口 ========================
@app.route("/")
def index():
    """前端主页"""
    return render_template("demo.html")


# ======================= 模型切换接口 ==========================
def reload_model(new_key):
    """热切换模型： 线程安全"""
    global inferencer, current_key
    if new_key not in MODELS:
        raise ValueError(f"不支持的模型 {new_key}")
    with load_lock:
        if inferencer:
            inferencer.destroy()
            inferencer = None
        # _release_cuda_ctx()
        current_key = new_key
        inferencer = MODELS[current_key]["loader"]() # 加载新模型
    print(f"[INFO] 已切换至 {MODELS[new_key]['name']}")

@app.route("/switch_model", methods=["POST"])
def switch_model():
    print("尝试切换模型")
    key = request.json.get("model") if request.is_json else request.args.get("model")
    try:
        reload_model(key)
        return jsonify({"code": 0, "msg": "切换成功", "now": key})
    except Exception as e:
        return jsonify({"code": 1, "msg": str(e)}), 400


# ======================== 字符识别功能控制接口 ========================
@app.route("/ctrl_spotting")
def ctrl_spotting():
    """控制字符识别功能"""
    global spotting_on
    spotting_on = request.args.get('on') == "1"
    # ===== 这里让开关立即生效 =====
    if spotting_on:
        start_inference_worker()   # 你的后台线程/进程启动函数
    else:
        stop_inference_worker()    # 安全停止函数
    #这里可以启停 OCR/识别线程、加载/卸载模型等
    return {"spotting": spotting_on}

# ===================== 字符识别启动/停止封装 =====================
def start_inference_worker():
    """开启识别线程（重复调用不会重复开）"""
    global infer_thread, infer_evt, cleanup_thread
    if infer_thread is not None and infer_thread.is_alive():
        print("推理线程已在运行")
        return
    infer_evt.clear()                                 # 清除退出标志
    infer_thread = threading.Thread(target=inference_worker, daemon=True)
    infer_thread.start()
    print("推理线程已启动")
    # 启动过期会话清理线程
    cleanup_thread = threading.Thread(target=cleanup_expired_sessions, daemon=True)
    cleanup_thread.start()
    print("会话清理线程已启动")

def stop_inference_worker():
    """停止识别线程（阻塞到线程真正退出）"""
    global infer_thread, cleanup_thread
    if infer_thread is None:
        return
    infer_evt.set()                                   # 通知线程退出
    infer_thread.join(timeout=1)                      # 最多等 1 秒
    if not infer_thread.is_alive():
        print("推理线程已安全退出")
    else:
        print("推理线程超时，强制结束")
    infer_thread = None
    cleanup_thread.join(timeout=1)                      # 最多等 1 秒
    if not cleanup_thread.is_alive():
        print("会话线程已安全退出")
    else:
        print("会话线程超时，强制结束")
    cleanup_thread = None
    
def inference_worker():
    """后台线程，不断取帧推理->放结果"""
    global track_on, result_queue, track_sessions, infer_evt
    while not infer_evt.is_set():
        # ctx = cuda.Device(0).make_context()   # 各线程自己 ctx
        try:
            frame = frame_queue.get(timeout=1) # 阻塞1s
            h, w, _ = frame.shape
            with load_lock:                     # 模型切换期间阻塞推理
                raw_results = inferencer.       process_image(frame) # 字符识别推理
            raw_results = process_results_for_table(raw_results)
            spotting_nums = len(raw_results)
            # print(raw_results)
            # 匹配模板 + 过滤非模板内容
            tracked_results = []
            for raw_res in raw_results:
                # 对每个推理结果进行模板匹配
                if track_on:
                    with track_lock:
                        # 遍历所有活跃会话（支持多前端同时跟踪）
                        for session in track_sessions.values():
                            matched_target = session.match_infer_result(raw_res)
                            if matched_target:
                                tracked_results.append(matched_target)
                else:
                    matched_target = direct_return_result(raw_res)
                    tracked_results.append(matched_target)

            if result_queue.full():
                result_queue.get()
            result_queue.put([tracked_results, h, w, spotting_nums])
        except queue.Empty:
            continue
        except Exception as e:
            print("推理线程异常:", e)
            result_queue.put([[], 720, 1280, 0])


# ======================== 字符识别功能打印接口 ========================

# def to_native(obj):
#     """递归把 numpy/torch 标量转成 Python 原生类型"""
#     if isinstance(obj, (np.generic, np.ndarray)):
#         return obj.tolist()               # 数组→list，标量→原生 float/int
#     if isinstance(obj, list):
#         return [to_native(i) for i in obj]
#     if isinstance(obj, dict):
#         return {k: to_native(v) for k, v in obj.items()}
#     return obj

# @app.route('/detection_results')
# def detection_results():
#     """浏览器轮询用的实时结果接口"""
#     global latest_results
#     latest_results = to_native(latest_results or [])  # 先洗一遍
#     return jsonify(latest_results)


# ======================== 字符识别检测框显示接口 ========================
@app.route('/ctrl_draw')
def ctrl_draw():
    """打开/关闭检测框"""
    global draw_on
    draw_on = request.args.get("on") == "1"
    return {"draw": draw_on}

def four_points_to_rect(bbox_4x2):
    """4×2 点 → [x1,y1,x2,y2]"""
    pts = np.array(bbox_4x2, dtype=int)
    return (*pts.min(axis=0), *pts.max(axis=0))

# ======================== 摄像头调用接口 ========================
@app.route("/ctrl_camera")
def ctrl_camera():
    """控制摄像头"""
    global evt
    camera_on = request.args.get('on') == "1"
    if camera_on:
        evt.set()
    else:
        evt.clear()
    return {"streaming": camera_on}

def gen_frames_from_camera():
    """生成带检测标注的视频流（MJPEG格式）"""
    global cap, inferencer, latest_results, spotting_on, frame_queue, result_queue, draw_on, _last_results
    # 程序关闭时退出循环
    with cap_lock:
        if cap is None or not cap.isOpened():  # 检查摄像头是否有效
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ 摄像头无法打开")
                return  # 返回空，避免后续操作
    try:
        while evt.is_set():
            with cap_lock:
                ret, frame = cap.read()
            if not ret:
                print("❌ 摄像头读取失败")
                break
            # 根据检测开关和推理器状态决定是否执行检测
            if spotting_on:
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    finally:
        # ===== 无论迭代器如何结束，都释放硬件 =====
        with cap_lock:
            if cap and cap.isOpened():
                cap.release()
                print('[camera released]')  # 调试用，可删
        evt.clear()                         # 保险：复位事件


# ======================== 本地视频文件上传接口 ========================
@app.route("/upload_video", methods=["POST"])
def upload_video():
    """前端上传本地视频文件"""
    if "video" not in request.files:
        return jsonify({"code": 1, "msg": "未选择文件"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"code": 1, "msg": "文件名空"}), 400
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return jsonify({"code": 0, "msg": "上传成功", "file": file.filename})


@app.route("/ctrl_playing")
def ctrl_playing():
    """播放上传的视频文件"""
    global evt
    play_on = request.args.get('on') == "1"
    if play_on:
        evt.set()
    else:
        evt.clear()
    return {"streaming": evt.is_set()}


# =============== 添加视频文件暂停播放功能 ============================
video_state = "break"
video_lock = threading.Lock()

@app.route("/ctrl_video")
def ctrl_video():
    global evt, video_state
    v_state = request.args.get("video_state")
    video_state = v_state
    if v_state == "play":
        if not evt.is_set():
            evt.set()
    return {"streaming": evt.is_set()}


def gen_frames_from_file(video_path: str):
    """逐帧读取本地文件 -> MJPEG"""
    global inferencer, spotting_on, draw_on, video_state
    last_frame = None
    cap_file = cv2.VideoCapture(video_path)
    # if not cap_file.isOpened():
    #     yield b''; return
    try:
        while evt.is_set():               # 受同一开关控制
            with video_lock:
                if video_state == "pause":
                    frame = last_frame
                else:
                    ret, frame = cap_file.read()
                    last_frame = frame
                    if not ret:
                        evt.clear()
                        # yield(b'--frame--\r\n')
                        print("✅ 视频文件播放结束")
                        break
            # results = []
            if spotting_on:
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    finally:
        cap_file.release()
# ======================== 视频流接口 ========================
@app.route("/video_stream")
def video_stream():
    """视频流接口（供前端展示）"""
    source = request.args.get("source", "camera")   # 默认摄像头
    if source == "file":
            filename = request.args.get("file")         # 例如 ?source=file&file=xxx.mp4
            if not filename:
                return "", 204
            path = os.path.join(UPLOAD_FOLDER, filename)
            if not os.path.exists(path):
                return jsonify({"code": 1, "msg": "文件不存在"}), 404
            if not evt.is_set():
                return "", 204 
            return Response(gen_frames_from_file(path),
                            mimetype="multipart/x-mixed-replace; boundary=frame")
    if not evt.is_set():
        return "", 204
    return Response(
        gen_frames_from_camera(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ======================== 处理冻结帧 ========================
@app.route('/process_freeze_frame', methods=['POST'])
def process_freeze_frame():
    global inferencer
    try:
        data = request.get_json()
        if not data or "image_base64" not in data:
            return jsonify({"success": False, "error": "缺少 image_base64 参数"}), 400

        # Base64 解码（原有逻辑不变）
        base64_str = data["image_base64"]
        base64_str = base64_str.replace(" ", "+").replace("\n", "").replace("\r", "")  # 处理非法字符
        try:
            image_bytes = base64.b64decode(base64_str)
        except Exception as e:
            print(f"Base64 解码失败：{str(e)}")
            return jsonify({"success": False, "error": "Base64 解码失败"}), 400
        img_np = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "图像解码失败"}), 400
        results = []
        cv2.imshow('img', img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        results = inferencer.process_image(img)
        results = process_results_for_table(results)
        return jsonify({
            "success": True,
            "frame_id": data.get("frame_id", ""),
            "results": results  # 列表类型，每个元素是dict
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def process_results_for_table(results):
    new_results = []
    for idx, obj in enumerate(results):
        # 每个目标生成一个dict，对应表格一行
        new_results.append({
            "obj": obj.get("object", f"目标{idx+1}"),
            "track_id": obj.get("track_id", f"track_{idx+1}"),
            "text": obj.get("text", None),  # 检测内容
            "upper_limit": obj.get("upper_limit", 20),
            "lower_limit": obj.get("lower_limit", 0),
            "confidence": round(obj.get("conf", 0), 2), # 置信度
            "bbox": obj.get("bbox").tolist(), # 监测框
        })

    # 若模型无返回结果，返回空列表
    return new_results if new_results else []

# ======================== 跟踪匹配接口 ====================================
@app.route("/api/track/init", methods=["POST"])
def init_track():
    """
    初始化跟踪会话，前端传入模板数据，后端创建跟踪上下文，返回会话ID
    """
    try:
        data = request.get_json()
        if not data or "template" not in data:
            return jsonify({"success": False, "error": "缺少模板数据"}), 400
        
        # 解析前端模板数据
        template = Template.from_dict(data["template"])
        client_id = data.get("clientId", f"client_{uuid.uuid4().hex[:8]}")
        
        # 生成唯一会话ID
        session_id = str(uuid.uuid4())
        # 创建跟踪上下文
        with track_lock:
            track_context = TrackContext(
                session_id=session_id,
                template=template,
            )
            track_sessions[session_id] = track_context
        
        # 返回结果给前端
        return jsonify({
            "success": True,
            "sessionId": session_id,
            "templateName": template.templateName,
            "targetCount": len(template.content),
            "message": "跟踪会话初始化成功"
        })
    except Exception as e:
        print("初始化跟踪异常:", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/ctrl_tracking")
def ctrl_track():
    """跟踪状态控制"""
    global track_on
    track_on = request.args.get('on') == "1"
    # ===== 这里让开关立即生效 =====
    if not track_on:
        stop_track()    # 安全停止函数
    return {"tracking": track_on}

@app.route("/api/track/stop", methods=["POST"])
def stop_track():
    """停止跟踪会话，释放资源"""
    try:
        data = request.get_json()
        session_id = data.get("sessionId")
        if not session_id:
            return jsonify({"success": False, "error": "缺少sessionId"}), 400
        
        # 删除会话
        with track_lock:
            if session_id in track_sessions:
                del track_sessions[session_id]
        
        return jsonify({"success": True, "message": "跟踪会话已停止"})
    except Exception as e:
        print("停止跟踪异常:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================ 仅识别或识别跟踪结果反馈接口===================================
@app.route("/api/track/results", methods=["GET"])
def get_track_results():
    """
    前端轮询获取跟踪结果（），仅返回当前会话的匹配结果（过滤其他会话数据）
    """
    global result_queue, track_on
    # 初始化默认值：避免变量未定义报错
    tracked_results = []
    w = 1280  # 图像宽度默认值
    h = 720  # 图像高度默认值
    spotting_nums = 0
    try:
        session_id = request.args.get("sessionId")
        if not result_queue.empty():
            all_results, h, w, spotting_nums = result_queue.get()
            # 在 print(all_results) 后添加：打印每个目标的 matchedTemplateTargetId
            # 过滤出当前会话的跟踪结果
            if track_on:
                if not session_id:
                    return jsonify({"success": False, "error": "缺少sessionId"}), 400
                
                # 验证会话是否存在
                with track_lock:
                    if session_id not in track_sessions:
                        return jsonify({"success": False, "error": "跟踪会话不存在"}), 404
                template_id = track_sessions[session_id].template.templateId
                tracked_results = [
                    res.to_dict() for res in all_results
                    if res.matchedTemplateTargetId.startswith(f"temp_{template_id}")
                ]
            else:
                tracked_results = [
                    res.to_dict() for res in all_results
                ]
        
        if current_key == "PANEL":
            w, h = 1280, 720 
        # 返回结构化结果
        return jsonify({
            "success": True,
            "sessionId": session_id,
            "trackedTargets": tracked_results,
            "width": w,
            "height": h,
            "spotting_nums": spotting_nums,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        print("获取跟踪结果异常:", e)
        return jsonify({"success": False, "error": str(e)}), 500

# ---------------------- 定时清理过期会话 ----------------------
def cleanup_expired_sessions():
    """每分钟清理10分钟未活跃的会话（避免内存泄漏）"""
    while not infer_evt.is_set():
        time.sleep(60)
        now = datetime.now()
        expired_sessions = []
        with track_lock:
            for sid, ctx in track_sessions.items():
                if (now - ctx.last_active_time) > timedelta(minutes=10):
                    expired_sessions.append(sid)
            # 删除过期会话
            for sid in expired_sessions:
                del track_sessions[sid]
        if expired_sessions:
            print(f"清理过期会话：{expired_sessions}")

# ======================== 服务启动 ========================
if __name__ == "__main__":
    # 解决matplotlib线程问题
    import matplotlib
    matplotlib.use('Agg')
    
    try:
        # 启动Flask服务（启用多线程支持）
        app.run(
            host="0.0.0.0",  # 允许外部访问
            port=8000,       # 服务端口
            debug=True,     # 生产环境关闭调试模式
            threaded=True,   # 启用多线程（支持并发访问）
            use_reloader=False  # 禁用重载器（避免重复初始化）
        )
    except KeyboardInterrupt:
        print("\n⚠️  收到退出信号，正在关闭服务...")
    finally:
        # 确保资源清理
        cleanup_all_resources()