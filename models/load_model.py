
# from .core import (
#     TRTModelInferencer,
#     ENGINE_PATH, INPUT_DIMS
# )
from .gen_core import GENInferencer
from .onnx_core.onnx_inferencer import ONNXModelInferencer
# class BaseModel:
#     """统一标准，返回[{"text": None, "conf": None, "bbox": None,},{},...]"""
#     def __init__(self,):
#         self.engine: TRTModelInferencer = None
#         try:
#             self.engine= TRTModelInferencer(ENGINE_PATH, INPUT_DIMS)
#         except Exception as e:
#             raise RuntimeError(f"BaseModel初始化失败: {str(e)}")

#     def process_image(self, img):
#         if self.engine is None or self.engine.is_destroyed:
#             raise RuntimeError("模型未初始化或已销毁")
#         return self.engine.process_image(img)

#     def destroy(self) -> None:
#         """销毁模型资源（程序退出前必须调用）"""
#         print("🔧 清理BaseModel资源...")
#         if hasattr(self, 'engine') and self.engine:
#             self.engine.destroy()
#             del self.engine
#         # 强制清理CUDA上下文
#         print("✅ BaseModel资源清理完成")
    
#     def __del__(self,):
#         self.destroy()

class PanelModel:
    """统一标准，返回[{"text": None, "conf": None, "bbox": None,},{},...]"""
    def __init__(self):
        self.engine: ONNXModelInferencer = None
        try:
            self.engine = ONNXModelInferencer()
        except Exception as e:
            raise RuntimeError(f"PanelModel初始化失败: {str(e)}")
    
    def process_image(self, img):
        if self.engine is None:
            raise RuntimeError("模型未初始化或已销毁")
        output = self.engine.process_image(img)
        return output   
    
    def destroy(self) -> None:
        """销毁模型资源（程序退出前必须调用）"""
        print("🔧 清理GenModel资源...")
        if hasattr(self, 'engine') and self.engine:
            del self.engine
        # 强制清理CUDA上下文
        print("✅ GenModel资源清理完成")
    
    def __del__(self,):
        self.destroy()



class GenModel:
    """统一标准，返回[{"text": None, "conf": None, "bbox": None,},{},...]"""
    def __init__(self):
        self.engine: GENInferencer = None
        try:
            self.engine = GENInferencer()
        except Exception as e:
            raise RuntimeError(f"GenModel初始化失败: {str(e)}")
    
    def process_image(self, img):
        if self.engine is None:
            raise RuntimeError("模型未初始化或已销毁")
        output = self.engine.process_image(img)
        output = self.structured_data(output)
        return output
    
    def structured_data(self, output):
        structured_output = []
        box_list, texts, score_list = output
        for box, text, score in zip(box_list, texts, score_list):
            item = {"text": text, "conf": score, "bbox": box}
            structured_output.append(item)
        return structured_output
    
    def destroy(self) -> None:
        """销毁模型资源（程序退出前必须调用）"""
        print("🔧 清理GenModel资源...")
        if hasattr(self, 'engine') and self.engine:
            del self.engine
        # 强制清理CUDA上下文
        print("✅ GenModel资源清理完成")
    
    def __del__(self,):
        self.destroy()
    
        









        