"""Configuration for the PRC baseline experiments.

System-specific neighboring input components follow the coupling functions:
- llz_xx: linearly coupled Lorenz system, use x_j (component 0).
- llz_xy: product-coupled Lorenz system, use y_j (component 1).
- lsl: linearly coupled Rössler system, use x_j (component 0).

The output filenames match those loaded by figure.ipynb.
"""

PRC_SYSTEM_CONFIGS = {
    "llz_xx": {
        "data_dir": "generate.data/llz.X-X",
        "data_pattern": "{M}coupled_lorenz_data.npy",
        # Corrected repository directory: result.data (not result.date).
        "weight_dir": "result.data/llz.(x-x)/network",
        "weight_pattern": "{M}know_W_predicted.(x-x).npy",
        "neighbor_component": 0,
        "pred_len": 1000,
        "prediction_pattern": "{M}coupled_lorenz_pred.x-x.npy",
    },
    "llz_xy": {
        "data_dir": "generate.data/llz.xy",
        "data_pattern": "{M}coupled_lorenz_data.xy.npy",
        "weight_dir": "result.data/llz.xy/network",
        "weight_pattern": "{M}know_W_predicted.(xy).npy",
        "neighbor_component": 1,
        "pred_len": 1000,
        "prediction_pattern": "{M}coupled_lorenz_pred.xy.npy",
    },
    "lsl": {
        "data_dir": "generate.data/lsl",
        "data_pattern": "{M}coupled_rossler_data.npy",
        "weight_dir": "result.data/lsl/network",
        "weight_pattern": "{M}rossler_know_W_predicted.npy",
        "neighbor_component": 0,
        "pred_len": 10000,
        "prediction_pattern": "{M}coupled_lsl_pred.npy",
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
    "prediction_dir": "result.data/PRC",
}


def get_prc_config(system, M):
    if system not in PRC_SYSTEM_CONFIGS:
        raise KeyError(
            f"Unknown system: {system}. Choose from {tuple(PRC_SYSTEM_CONFIGS.keys())}."
        )

    cfg = dict(PRC_SYSTEM_CONFIGS[system])
    cfg.update(PRC_HYPERPARAMETERS)
    cfg["system"] = system
    cfg["M"] = M
    cfg["data_file"] = f'{cfg["data_dir"]}/{cfg["data_pattern"].format(M=M)}'
    cfg["weight_file"] = f'{cfg["weight_dir"]}/{cfg["weight_pattern"].format(M=M)}'
    cfg["prediction_file"] = (
        f'{cfg["prediction_dir"]}/{cfg["prediction_pattern"].format(M=M)}'
    )
    return cfg
