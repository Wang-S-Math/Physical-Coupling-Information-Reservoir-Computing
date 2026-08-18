"""
Configuration for the PRC baseline experiments.

Abbreviations
-------------
llz_xx : linearly coupled Lorenz system (x-x)
llz_xy : nonlinear product-coupled Lorenz system (x*y)
lsl    : linearly coupled Rossler system (x-x)

The PRC hyperparameters follow the settings used for the paper comparison.
For the main dynamics-prediction comparison in the paper, M = 10.
"""

PRC_SYSTEM_CONFIGS = {
    "llz_xx": {
        "data_dir": "generate.data/llz.X-X",
        "data_pattern": "{M}coupled_lorenz_data.npy",
        "weight_dir": "result.date/llz.(x-x)/network",
        "weight_pattern": "{M}know_W_predicted.(x-x).npy",
        "pred_len": 1000,
    },

    "llz_xy": {
        "data_dir": "generate.data/llz.xy",
        "data_pattern": "{M}coupled_lorenz_data.xy.npy",
        "weight_dir": "result.date/llz.xy/network",
        "weight_pattern": "{M}know_W_predicted.(xy).npy",
        "pred_len": 1000,
    },

    "lsl": {
        "data_dir": "generate.data/lsl",
        "data_pattern": "{M}coupled_rossler_data.npy",
        "weight_dir": "result.date/lsl/network",
        "weight_pattern": "{M}rossler_know_W_predicted.npy",
        "pred_len": 10000,
    },
}


PRC_HYPERPARAMETERS = {
    "nn": 500,
    "grla": 1e-6,
    "grg": 0.7,
    "pbj": 0.95,
    "xshl": 0.03,
    "grso": 1.2,
    "she": 1000,
    "train_len": 15000,
    "vps_threshold": 0.5,
}


def get_prc_config(system, M):
    if system not in PRC_SYSTEM_CONFIGS:
        raise KeyError(
            f"Unknown system: {system}. "
            f"Choose from {tuple(PRC_SYSTEM_CONFIGS.keys())}."
        )

    cfg = dict(PRC_SYSTEM_CONFIGS[system])
    cfg.update(PRC_HYPERPARAMETERS)

    cfg["system"] = system
    cfg["M"] = M
    cfg["data_file"] = (
        f'{cfg["data_dir"]}/{cfg["data_pattern"].format(M=M)}'
    )
    cfg["weight_file"] = (
        f'{cfg["weight_dir"]}/{cfg["weight_pattern"].format(M=M)}'
    )

    return cfg
