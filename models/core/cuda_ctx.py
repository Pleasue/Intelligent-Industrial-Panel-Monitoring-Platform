import threading, atexit, signal, sys
import pycuda.driver as cuda
from pycuda.tools import clear_context_caches

_cuda_lock = threading.Lock()
_cuda_ctx  = None
_push_cnt = 0

def cuda_init():
    """整个进程只调用一次"""
    global _cuda_ctx
    with _cuda_lock:
        if _cuda_ctx is None:
            cuda.init()
            _cuda_ctx = cuda.Device(0).make_context()
    # 注册退出清理
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT,  lambda *_: (_cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), sys.exit(0)))

def _cleanup():
    global _push_cnt
    with _cuda_lock:
        print(f"[CUDA] 退出前 push 计数 = {_push_cnt}")
        while _push_cnt > 0:
            cuda.Context.pop()
            _push_cnt -= 1
        clear_context_caches()
    print("[CUDA] 上下文已空")

def cuda_push():
    global _push_cnt
    with _cuda_lock:
        _cuda_ctx.push()
        _push_cnt += 1

def cuda_pop():
    global _push_cnt
    with _cuda_lock:
        if _push_cnt > 0:
            cuda.Context.pop()
            _push_cnt -= 1