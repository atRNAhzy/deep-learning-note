# 全局配置与超参数
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# 路径配置
DATA_ROOT = Path("../data")
OUTPUT_DIR = Path("../temp")

# 数据/模型通用
NUM_CLASSES = 10
INPUT_SIZE = 32
NUM_WORKERS = 4
DEVICE_FALLBACK_CPU = True  # 若 GPU 不可用或 OOM，允许回退 CPU

# 固定的模型结构（为简化搜索）
FIXED_MODEL = {
    "num_cnn_layers": 3,
    "cnn_channels": [64, 128, 256],
    "kernel_size": 3,
    "pooling_type": "max",  # {'max', 'avg', 'none'}
    "num_linear_layers": 2,
    "linear_hidden": [128, 64],
}

# 搜索空间（7 项基因）
BATCH_SIZE_CHOICES = [16, 32, 64]
LR_LOG10_RANGE = (-5.0, -2.0)           # 10**U(-5,-2)
DROPOUT_RANGE = (0.0, 0.7)              # U(0, 0.7)

# 模型开关（是否使用 BN / Dropout）默认值
# 说明：在遗传算法中，这4个开关来自个体基因。如果想忽略基因、统一用固定开关，
# 将 OVERRIDE_GENE_SWITCHES 设为 True 即可，值取自 DEFAULT_SWITCHES。
DEFAULT_SWITCHES = {
    "use_bn_cnn": False,
    "use_dropout_cnn": True,
    "use_bn_linear": False,
    "use_dropout_linear": False,
}
OVERRIDE_GENE_SWITCHES = False

# 训练默认参数（用于 GA 快速评估）
NUM_EPOCHS_EVAL = 15
SAMPLE_COUNT_TRAIN = 2000
SAMPLE_COUNT_TEST = 500

# 遗传算法参数
POP_SIZE = 12
NGEN = 10
CXPB = 0.8
MUTPB = 0.5
TOURN_SIZE = 2
# 每个基因被突变的概率
MUT_INDPB = 0.2

# 随机性
SEED_BASE = 42

@dataclass
class EvalConfig:
    num_epochs: int = NUM_EPOCHS_EVAL
    sample_count_train: Optional[int] = SAMPLE_COUNT_TRAIN
    sample_count_test: Optional[int] = SAMPLE_COUNT_TEST
    output_dir: Path = OUTPUT_DIR

