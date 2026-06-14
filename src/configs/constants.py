import re
from typing import Literal

### Main arg MODE types
Mode = Literal["pretrain", "downstream_eval"]

### Datasets
BASE_DATASETS = {"ptb_xl", "mimic_iv", "code15", "cpsc", "csn",}

### CLassification

# All allowed datasets, just add to the union
ALLOWED_DATA = set().union(BASE_DATASETS)

### Configs
CONFIG_FOLDER = "src/configs"

### Runs
RUNS_FOLDER = "src/runs"

DATA_DIR = "../data"

PTB_ORDER = ["I", "II", "III", "aVL", "aVR", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PTB_INDEPENDENT_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
PTB_INDEPENDENT_IDX = [0,1,6,7,8,9,10,11]

### Models

TRANSFORMER_MODELS = {
    "trans_discrete_decoder": {
        "find_unused_parameters": False,
    },
    "trans_continuous_nepa": {
        "find_unused_parameters": False,
    },
    "trans_continuous_dit": {
        "find_unused_parameters": False,
    },
}

MAE_MODELS = {
    "mae_vit": {
        "find_unused_parameters": False,
    },
}

MERL_MODEL = {
    "merl": {
        "find_unused_parameters": False,
    },
}

MLAE_MODELS = {
    "mlae": {
        "find_unused_parameters": False,
    },
}

MTAE_MODELS = {
    "mtae": {
        "find_unused_parameters": False,
    },
}

ST_MEM_MODELS = {
    "st_mem": {
        "find_unused_parameters": False,
    },
}

DBETA_MODELS = {
    "dbeta": {
        # MEM taps an intermediate cross-layer, and several modules feed multiple
        # losses -> the gradient-ready order is nondeterministic and DDP's bucket
        # rebuild desyncs, hanging in backward. static_graph records the graph once
        # and fixes the reduction order (and subsumes unused-param handling).
        "find_unused_parameters": False,
        "static_graph": True,
    },
}

# Encoders
ECG_ENCODERS = {
    "st_mem": {
        "find_unused_parameters": False,
        "strict": False,
        "model_hidden_size": 768,
        "projection_dim": 256,
        "encoder_input_len": None,
    },
    "mtae": {
        "find_unused_parameters": False,
        "strict": False,
        "model_hidden_size": 768,
        "projection_dim": 256,
        "encoder_input_len": None,
    },
    "mlae": {
        "find_unused_parameters": False,
        "strict": False,
        "model_hidden_size": 768,
        "projection_dim": 256,
        "encoder_input_len": None,
    },
}

def case_preserving_signal(m: re.Match) -> str:
    w = m.group(1)
    return "Signal" if w[0].isupper() else "signal"