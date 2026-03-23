# 模型配置
DET_PATH = "models\gen_core\det_db_mbv3_new.pth"  # TensorRT引擎路径
REC_PATH = "models\\gen_core\\ch_rec_moblie_crnn_mbv3.pth"
DICT_PATH = "models\\gen_core\\torchocr\\datasets\\alphabets\\ppocr_keys_v1.txt"
INPUT_DIMS = (1280, 720)                    # 模型输入尺寸
CONF_THRESHOLD = 0.8                       # 置信度阈值