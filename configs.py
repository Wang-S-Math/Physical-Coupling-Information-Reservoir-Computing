from copy import deepcopy

# 1. Common settings for each dynamical system
SYSTEM_CONFIGS = {
    "llz_xx": {
        "data_dir": "generate.data/llz.X-X",
        "data_pattern": "{M}coupled_lorenz_data.npy",
        "adj_pattern": "{M}coupled_lorenz_adj_thresholded.npy",
        "result_dir": "result.date/llz.(x-x)",
        "dt": 0.01,
        "train_len": 15000,
        "sync_len": 1000,
        "pred_len": 1000,
        "res_size": 500,
        "spectral_radius": 0.95,
        "leakage": 0.25,
        "coupling_strength": 0.5,
        "coupling_dims": (1, 0, 0),
        "input_scale": {
            "known": 0.05,
            "unknown": 0.05,
        },
        "seed_base": {
            "known": 42,
            "unknown": 42,
        },
    },

    "llz_xy": {
        "data_dir": "generate.data/llz.xy",
        "data_pattern": "{M}coupled_lorenz_data.xy.npy",
        "adj_pattern": "{M}coupled_lorenz_adj_thresholded.xy.npy",
        "result_dir": "result.date/llz.xy",
        "dt": 0.01,
        "train_len": 15000,
        "sync_len": 1000,
        "pred_len": 1000,
        "res_size": 500,
        "spectral_radius": 0.95,
        "leakage": 0.20,
        "coupling_strength": 0.2,
        "coupling_dims": (1, 0, 0),
        "input_scale": {
            "known": 0.10,
            "unknown": 0.10,
        },
        "seed_base": {
            "known": 42,
            "unknown": 42,
        },
    },

    "lsl": {
        "data_dir": "generate.data/lsl",
        "data_pattern": "{M}coupled_rossler_data.npy",
        "adj_pattern": "{M}coupled_rossler_adj_thresholded.npy",
        "result_dir": "result.date/lsl",
        "dt": 0.01,
        "train_len": 15000,
        "sync_len": 1000,
        "pred_len": 10000,
        "res_size": 500,
        "spectral_radius": 0.95,
        "leakage": 0.25,
        "coupling_strength": 0.05,
        "coupling_dims": (1, 0, 0),
        "input_scale": {
            "known": 0.10,
            "unknown": 0.05,
        },
        "seed_base": {
            "known": 42,
            "unknown": 43,
        },
    },
}


# ---------------------------------------------------------------------
# 2. Network sizes
# ---------------------------------------------------------------------

AVAILABLE_SIZES = {
    "llz_xx": {
        "known": (5, 10, 20, 50, 100),
        "unknown": (5, 10, 20, 50, 100),
    },
    "llz_xy": {
        "known": (5, 10, 20, 50, 100),
        "unknown": (5, 10, 20, 50, 100, 700),
    },
    "lsl": {
        "known": (5, 10, 20, 50, 100),
        "unknown": (5, 10, 20, 50, 100),
    },
}

DEFAULT_REGULARIZATION = {
    "lambda_ridge": 1e-3,
    "lambda_net": 1e-7,
}

REGULARIZATION_BY_SIZE = {
    "llz_xx": {
        "known": {
            5:   {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            10:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            20:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            50:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            100: {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
        },
        "unknown": {
            5:   {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            10:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            20:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            50:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            100: {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
        },
    },

    "llz_xy": {
        "known": {
            5:   {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            10:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            20:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            50:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            100: {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
        },
        "unknown": {
            5:   {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            10:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            20:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            50:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            100: {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            700: {"lambda_ridge": 1e-7, "lambda_net": 1e-3},
        },
    },

    "lsl": {
        # Known topology: all sizes use the same regularization.
        "known": {
            5:   {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            10:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            20:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            50:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            100: {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
        },

        # Unknown topology: M=50 and M=100 use topology-specific settings.
        "unknown": {
            5:   {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            10:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            20:  {"lambda_ridge": 1e-3, "lambda_net": 1e-7},
            50:  {"lambda_ridge": 1e-4, "lambda_net": 1e-7},
            100: {"lambda_ridge": 1e-7, "lambda_net": 1e-3},
        },
    },
}


# 4. Hard thresholds for unknown-topology reconstruction

UNKNOWN_THRESHOLDS = {
    "llz_xx": {
        5: 0.005,
        10: 0.005,
        20: 0.005,
        50: 0.005,
        100: 0.005,
    },

    "llz_xy": {
        5: 0.005,
        10: 0.005,
        20: 0.005,
        50: 0.005,
        100: 0.005,
        700: 0.01,
    },

    "lsl": {
        5: 0.015,
        10: 0.015,
        20: 0.015,
        50: 0.020,
        100: 0.020,
    },
}


# 5. Evaluation / repeat settings

EVALUATION_CONFIG = {
    "vps_threshold": 0.5,
    "n_independent_runs": 20,
}

# 6. Helper function

def get_config(system, topology, M):
    if system not in SYSTEM_CONFIGS:
        raise KeyError(
            f"Unknown system: {system}. "
            f"Choose from {tuple(SYSTEM_CONFIGS.keys())}."
        )

    if topology not in ("known", "unknown"):
        raise ValueError("topology must be 'known' or 'unknown'.")

    if M not in AVAILABLE_SIZES[system][topology]:
        raise ValueError(
            f"M={M} is not listed for {system}/{topology}. "
            f"Available sizes: {AVAILABLE_SIZES[system][topology]}"
        )

    cfg = deepcopy(SYSTEM_CONFIGS[system])

    cfg["system"] = system
    cfg["topology"] = topology
    cfg["M"] = M

    cfg["data_file"] = (
        f'{cfg["data_dir"]}/{cfg["data_pattern"].format(M=M)}'
    )
    cfg["adj_file"] = (
        f'{cfg["data_dir"]}/{cfg["adj_pattern"].format(M=M)}'
    )

    cfg["input_scale"] = cfg["input_scale"][topology]
    cfg["seed_base"] = cfg["seed_base"][topology]

    regularization = REGULARIZATION_BY_SIZE[system][topology][M]
    cfg.update(regularization)

    if topology == "unknown":
        cfg["threshold"] = UNKNOWN_THRESHOLDS[system][M]
    else:
        cfg["threshold"] = None

    cfg.update(EVALUATION_CONFIG)

    return cfg


if __name__ == "__main__":
    # Example:
    # Nonlinear product-coupled Lorenz system,
    # unknown topology, M = 100.
    example = get_config("llz_xy", "unknown", 100)

    print("Example configuration:")
    for key, value in example.items():
        print(f"{key}: {value}")
