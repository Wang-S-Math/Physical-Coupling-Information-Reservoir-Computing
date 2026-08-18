import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy import linalg


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.linewidth": 1.1,
    "figure.dpi": 150,
})

DATA_FILE = "generate.data/llz.xy/10coupled_lorenz_data.xy.npy"
ADJ_FILE = "generate.data/llz.xy/10coupled_lorenz_adj_thresholded.xy.npy"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "noise_results_llz_xy")

DT = 0.01
TRAIN_LEN = 15000
SYNC_LEN = 1000
RES_SIZE = 500
COUPLING_STRENGTH = 0.2
LAMBDA_RIDGE = 1e-3
LAMBDA_NET = 1e-7
WEIGHT_THRESHOLD = 0.005

# 13个噪声水平：保留原来的3个低噪声点，之后从0.01到0.10按0.01递增
NOISE_LEVELS = np.array([
    0.0, 1e-4, 1e-3,
    0.01, 0.02, 0.03, 0.04, 0.05,
    0.06, 0.07, 0.08, 0.09, 0.10,
], dtype=float)
N_REPEATS = 20
RESERVOIR_BASE_SEED = 20260723
NOISE_BASE_SEED = 20260823
METRIC_NAMES = ["TPR", "Precision", "TNR", "weight_error"]


class Reservoir:
    def __init__(
        self,
        input_dim,
        res_size,
        seed,
        spectral_radius=0.95,
        leakage=0.2,
        input_scale=0.1,
    ):
        rng = np.random.default_rng(seed)
        self.alpha = leakage
        self.Win = rng.uniform(
            -input_scale, input_scale, (res_size, input_dim)
        )
        self.bias = rng.uniform(-input_scale, input_scale, res_size)

        w_res = rng.uniform(-1.0, 1.0, (res_size, res_size))
        radius = np.max(np.abs(linalg.eigvals(w_res, check_finite=False)))
        self.Wres = w_res * spectral_radius / radius

    def run_states(self, data):
        states = np.empty((len(data), self.Wres.shape[0]))
        state = np.zeros(self.Wres.shape[0])

        for t, u_t in enumerate(data):
            activation = self.Win @ u_t + self.Wres @ state + self.bias
            state = (
                (1.0 - self.alpha) * state
                + self.alpha * np.tanh(activation)
            )
            states[t] = state
        return states


def add_observation_noise(clean_data, sigma, seed):
    """x_noise(t)=x(t)+epsilon(t), epsilon~N(0,sigma^2)。"""
    if sigma == 0:
        return clean_data
    rng = np.random.default_rng(seed)
    return clean_data + rng.normal(0.0, sigma, clean_data.shape)


def reconstruct_network(time_series, reservoir_seed):
    _, n_nodes, _ = time_series.shape
    u_curr = time_series[SYNC_LEN:TRAIN_LEN - 1]
    u_next = time_series[SYNC_LEN + 1:TRAIN_LEN]
    derivative = (u_next - u_curr) / DT
    w_raw = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        reservoir = Reservoir(
            3, RES_SIZE, reservoir_seed + 1009 * i
        )
        states = reservoir.run_states(time_series[:TRAIN_LEN, i])
        r_i = states[SYNC_LEN:TRAIN_LEN - 1]
        neighbors = np.delete(np.arange(n_nodes), i)

        coupling = u_curr[:, i, 0, None] * u_curr[:, neighbors, 1]
        design = np.hstack((r_i, coupling))
        penalty = np.r_[
            np.full(RES_SIZE, LAMBDA_RIDGE),
            np.full(len(neighbors), LAMBDA_NET),
        ]

        matrix = design.T @ design
        matrix.flat[::matrix.shape[0] + 1] += penalty
        target = design.T @ derivative[:, i, 0]

        try:
            coefficients = linalg.solve(
                matrix, target, assume_a="pos", check_finite=False
            )
        except linalg.LinAlgError:
            coefficients = linalg.lstsq(
                matrix, target, check_finite=False
            )[0]

        w_raw[i, neighbors] = (
            coefficients[RES_SIZE:] / COUPLING_STRENGTH
        )

    w_pred = np.maximum(w_raw, 0.0)
    w_pred[w_pred < WEIGHT_THRESHOLD] = 0.0
    np.fill_diagonal(w_pred, 0.0)
    return w_pred


def evaluate_metrics(w_true, w_pred):
    # 排除自连接，避免主对角线被计入TN。
    mask = ~np.eye(w_true.shape[0], dtype=bool)
    true_edges = (np.abs(w_true) > 1e-6)[mask]
    pred_edges = (np.abs(w_pred) > WEIGHT_THRESHOLD)[mask]

    tp = np.count_nonzero(true_edges & pred_edges)
    fp = np.count_nonzero(~true_edges & pred_edges)
    fn = np.count_nonzero(true_edges & ~pred_edges)
    tn = np.count_nonzero(~true_edges & ~pred_edges)

    tpr = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0

    norm_true = np.linalg.norm(w_true, ord="fro")
    error = (
        np.linalg.norm(w_pred - w_true, ord="fro") / norm_true
        if norm_true else 0.0
    )
    return tpr, precision, tnr, error, tp, fp, fn


def save_csv(noise_levels, metrics, counts):
    os.makedirs(RESULT_DIR, exist_ok=True)
    all_file = os.path.join(RESULT_DIR, "noise_metrics_all_runs.csv")
    with open(all_file, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "noise_sigma", "repeat", *METRIC_NAMES, "TP", "FP", "FN"
        ])
        for i, sigma in enumerate(noise_levels):
            for repeat in range(N_REPEATS):
                writer.writerow([
                    sigma, repeat + 1,
                    *metrics[i, repeat], *counts[i, repeat],
                ])

    mean = np.mean(metrics, axis=1)
    std = np.std(metrics, axis=1, ddof=1)
    header = ["noise_sigma"]
    for name in METRIC_NAMES:
        header.extend((f"{name}_mean", f"{name}_std"))

    summary_file = os.path.join(RESULT_DIR, "noise_metrics_summary.csv")
    with open(summary_file, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for i, sigma in enumerate(noise_levels):
            row = [sigma]
            for column in range(len(METRIC_NAMES)):
                row.extend((mean[i, column], std[i, column]))
            writer.writerow(row)


def plot_results(noise_levels, metrics):
    mean = np.mean(metrics, axis=1)
    std = np.std(metrics, axis=1, ddof=1)

    # 参考论文图的横坐标表示方式：
    # 每个噪声水平等间距排列，标签仍显示真实噪声值。
    x_pos = np.arange(len(noise_levels))

    fig, ax_rate = plt.subplots(figsize=(11.5, 6.3))
    ax_error = ax_rate.twinx()

    rate_items = [
        (0, "TPR", "#2166AC", "^"),
        (2, "TNR", "#762A83", "s"),
        (1, "Precision", "#D6604D", "D"),
    ]
    lines = []

    for column, label, color, marker in rate_items:
        y_mean = mean[:, column]
        lower = np.clip(y_mean - std[:, column], 0.0, 1.0)
        upper = np.clip(y_mean + std[:, column], 0.0, 1.0)

        line, = ax_rate.plot(
            x_pos,
            y_mean,
            color=color,
            marker=marker,
            markersize=5.0,
            linewidth=2.2,
            label=label,
        )
        ax_rate.fill_between(
            x_pos,
            lower,
            upper,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        lines.append(line)

    error_color = "#4D9221"
    error_mean = mean[:, 3]
    error_lower = np.maximum(error_mean - std[:, 3], 0.0)
    error_upper = error_mean + std[:, 3]

    error_line, = ax_error.plot(
        x_pos,
        error_mean,
        color=error_color,
        marker="o",
        markersize=5.0,
        linewidth=2.2,
        label="Error",
    )
    ax_error.fill_between(
        x_pos,
        error_lower,
        error_upper,
        color=error_color,
        alpha=0.15,
        linewidth=0,
    )
    lines.append(error_line)

    ax_rate.set_xlabel(r"$\sigma_n$", fontsize=20)
    ax_rate.set_ylabel("TPR / TNR / Precision", fontsize=20)
    ax_error.set_ylabel(
        "Error",
        fontsize=20,
        color=error_color,
    )

    ax_rate.set_xlim(-0.35, len(noise_levels) - 0.65)
    ax_rate.set_ylim(0.0, 1.02)
    ax_error.set_ylim(bottom=0.0)

    # 13个噪声点全部显示，并采用参考图中的简洁科学计数法
    x_labels = [
        "0", "1e-4", "1e-3",
        "0.01", "0.02", "0.03", "0.04", "0.05",
        "0.06", "0.07", "0.08", "0.09", "0.1"
    ]
    ax_rate.set_xticks(x_pos)
    ax_rate.set_xticklabels(x_labels, fontsize=16)

    ax_rate.tick_params(axis="y", labelsize=18)
    ax_rate.tick_params(axis="x", pad=6)
    ax_error.tick_params(axis="y", labelsize=18, colors=error_color)
    ax_rate.grid(True, linestyle="--", alpha=0.10)

    fig.legend(
        handles=lines,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    os.makedirs(RESULT_DIR, exist_ok=True)
    fig.savefig(
        os.path.join(RESULT_DIR, "noise_robustness_llz_xy.pdf"),
        format="pdf",
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    raw_data = np.load(DATA_FILE, mmap_mode="r")
    true_adj = np.asarray(np.load(ADJ_FILE), dtype=float)

    if raw_data.ndim != 3 or raw_data.shape[2] != 3:
        raise ValueError("时间序列应为(time, oscillator, 3)的三维数组。")
    if raw_data.shape[0] < TRAIN_LEN:
        raise ValueError(
            f"数据长度为{raw_data.shape[0]}，小于TRAIN_LEN={TRAIN_LEN}。"
        )
    if true_adj.shape != (raw_data.shape[1], raw_data.shape[1]):
        raise ValueError(
            f"邻接矩阵{true_adj.shape}与振子数{raw_data.shape[1]}不一致。"
        )

    clean_data = np.asarray(raw_data[:TRAIN_LEN], dtype=float)
    n_noise = len(NOISE_LEVELS)
    metrics = np.full((n_noise, N_REPEATS, 4), np.nan)
    counts = np.full((n_noise, N_REPEATS, 3), -1, dtype=int)

    print(f"数据形状: {raw_data.shape}")
    print(f"结果目录: {RESULT_DIR}")
    print(f"噪声水平: {NOISE_LEVELS}")
    print(f"噪声点数: {n_noise}")
    print(f"每个噪声点重复次数: {N_REPEATS}")
    print(f"总实验次数: {n_noise * N_REPEATS}")

    for i, sigma in enumerate(NOISE_LEVELS):
        for repeat in range(N_REPEATS):
            # repeat不同，储备池和噪声不同；相同repeat跨噪声点保持一致。
            noisy_data = add_observation_noise(
                clean_data, sigma, NOISE_BASE_SEED + repeat
            )
            reservoir_seed = RESERVOIR_BASE_SEED + 100000 * repeat
            result = evaluate_metrics(
                true_adj,
                reconstruct_network(noisy_data, reservoir_seed),
            )

            metrics[i, repeat] = result[:4]
            counts[i, repeat] = result[4:]
            print(
                f"sigma={sigma:.4g}  "
                f"repeat={repeat + 1:02d}/{N_REPEATS}  "
                f"TPR={result[0]:.4f}  "
                f"Precision={result[1]:.4f}  "
                f"Error={result[3]:.6f}"
            )

    save_csv(NOISE_LEVELS, metrics, counts)
    plot_results(NOISE_LEVELS, metrics)

    print("\n各噪声点20次实验的平均结果")
    for sigma, values in zip(NOISE_LEVELS, np.mean(metrics, axis=1)):
        print(
            f"sigma={sigma:.4g}  "
            f"TPR={values[0]:.4f}  "
            f"Precision={values[1]:.4f}  "
            f"TNR={values[2]:.4f}  "
            f"Error={values[3]:.6f}"
        )
    print(f"\n结果已保存到: {RESULT_DIR}")


if __name__ == "__main__":
    main()
