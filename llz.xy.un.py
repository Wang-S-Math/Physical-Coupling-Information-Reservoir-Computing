

import numpy as np
import csv
import gc
from pathlib import Path
from scipy import linalg

from configs import get_config


# ==========================================
# 0. 评价指标 (Metrics)
# ==========================================
def evaluate_xu_metrics(W_true, W_pred, threshold=0.01):
    W_true = np.asarray(W_true, dtype=float)
    W_pred = np.asarray(W_pred, dtype=float)

    n = W_true.shape[0]
    mask = ~np.eye(n, dtype=bool)

    A_true = np.abs(W_true) > 1e-6
    A_pred = W_pred >= threshold

    yt = A_true[mask]
    yp = A_pred[mask]

    TP = np.sum(yt & yp)
    FN = np.sum(yt & (~yp))
    TN = np.sum((~yt) & (~yp))
    FP = np.sum((~yt) & yp)

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    diff_norm = np.linalg.norm(W_pred - W_true, "fro")
    true_norm = np.linalg.norm(W_true, "fro")
    Error = diff_norm / true_norm if true_norm > 0 else 0.0

    return TPR, TNR, Precision, Error
def calculate_vps(y_true, y_pred, threshold=0.5):
    """
    计算有效预测时间 (Valid Prediction Time / VPS)
    """
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    length = min(len(y_true_flat), len(y_pred_flat))
    y_true_flat = y_true_flat[:length]
    y_pred_flat = y_pred_flat[:length]

    std_val = np.std(y_true_flat, axis=0) + 1e-6
    diff = (y_true_flat - y_pred_flat) / std_val
    error_norm = np.sqrt(np.mean(diff ** 2, axis=1))

    indices = np.where(error_norm > threshold)[0]
    vps = indices[0] if len(indices) > 0 else length

    return vps, error_norm


# ==========================================
# 1. 核心类：统一的储备池 (Unified Reservoir)
# ==========================================
class Reservoir:
    def __init__(self, input_dim, res_size, spectral_radius=0.95, leakage=0.2, input_scale=0.1, seed=42):
        self.input_dim = input_dim
        self.res_size = res_size
        self.alpha = leakage
        self.input_scale = input_scale

        np.random.seed(seed)
        self.Win = (np.random.rand(res_size, input_dim) * 2 - 1) * input_scale
        W = np.random.rand(res_size, res_size) * 2 - 1
        radius = np.max(np.abs(linalg.eigvals(W)))
        self.Wres = W * (spectral_radius / radius)
        self.bias = (np.random.rand(res_size) * 2 - 1) * input_scale

        self.state = np.zeros(res_size)
        self.Wout = None

    def run_states(self, u_data):
        T = u_data.shape[0]
        states = np.zeros((T, self.res_size))
        r_curr = np.zeros(self.res_size)

        for t in range(T):
            u_t = u_data[t]
            pre_activation = np.dot(self.Win, u_t) + np.dot(self.Wres, r_curr) + self.bias
            r_curr = (1 - self.alpha) * r_curr + self.alpha * np.tanh(pre_activation)
            states[t] = r_curr

        self.state = r_curr
        return states

    def step(self, u_in):
        pre_activation = np.dot(self.Win, u_in) + np.dot(self.Wres, self.state) + self.bias
        self.state = (1 - self.alpha) * self.state + self.alpha * np.tanh(pre_activation)
        return self.state

    def set_readout_weights(self, Wout_trained):
        self.Wout = Wout_trained


# ==========================================
# 2. 辅助函数：物理耦合计算 (修改为 XY 乘积耦合)
# ==========================================
def get_coupling_term(u_state, adj_matrix, coupling_strength=1.0, coupling_mask=None):
    # === 修改开始: XY 乘积耦合公式 ===
    # 公式: sum(Aij * y_j * x_i)
    # u_state shape: (N_nodes, 3) -> x is col 0, y is col 1

    x = u_state[:, 0]
    y = u_state[:, 1]

    # 1. 计算邻居 y 的加权和: sum_j (A_ij * y_j)
    weighted_sum_y = (adj_matrix * coupling_strength) @ y

    # 2. 乘以自身的 x_i (逐元素相乘)
    coupling_term_x = weighted_sum_y * x

    # 3. 构造完整耦合向量 (仅作用于 x 轴)
    full_coupling = np.zeros_like(u_state)

    if coupling_mask is not None and coupling_mask[0] > 0:
        full_coupling[:, 0] = coupling_term_x

    return full_coupling
    # === 修改结束 ===


# ==========================================
# 3. 第一阶段：盲预测参数辨识 (Blind Identification)
# ==========================================
def stage_1_predict_weights_blind(time_series, dt=0.01, train_len=15000, sync_len=1000,
                                  res_size=500,
                                  lambda_ridge=1e-3,
                                  lambda_net=1e-7,
                                  coupling_dims=[1, 0, 0], coupling_strength=0.5,
                                  positive_only=True,
                                  threshold=0.005, seed_offset=0):
    """
    专门处理【未知结构】的情况，适配 XY 乘积耦合。
    """
    print("\n" + "=" * 60)
    print("STAGE 1: 盲预测参数辨识 (Blind Parameter Identification)")
    print(f"模式: [XY乘积耦合/未知结构] - 计算所有可能的连接并应用阈值。")


    steps, N_nodes, dim = time_series.shape
    coupling_mask = np.array(coupling_dims)
    U_curr = time_series[sync_len: train_len - 1]
    U_next = time_series[sync_len + 1: train_len]
    Y_target_deriv = (U_next - U_curr) / dt

    W_pred_matrix = np.zeros((N_nodes, N_nodes))
    trained_reservoirs = []

    for i in range(N_nodes):
        res = Reservoir(input_dim=dim, res_size=res_size, input_scale=0.1, seed=42 + i + seed_offset)
        r_full = res.run_states(time_series[:train_len, i, :])
        r_i = r_full[sync_len: train_len - 1]

        # === 核心逻辑：盲预测 (全连接假设) ===
        potential_neighbors = np.delete(np.arange(N_nodes), i)

        C_list = []
        for j in potential_neighbors:
            # === 修改开始: 使用 XY 乘积耦合特征 ===
            # 原逻辑: diff = (x_j - x_i)
            # 新逻辑: term = x_i * y_j

            x_i = U_curr[:, i, 0]  # 自身的 x
            y_j = U_curr[:, j, 1]  # 邻居的 y

            product_term = x_i * y_j

            # 构造耦合特征向量 (Time, 3)
            C_neighbor = np.zeros_like(U_curr[:, i, :])
            if coupling_mask[0] > 0:  # 确保 x 轴是允许耦合的
                C_neighbor[:, 0] = product_term

            C_list.append(C_neighbor)
            # === 修改结束 ===

        if len(C_list) > 0:
            C_matrix = np.hstack(C_list)
            L_i = np.hstack([r_i, C_matrix])
        else:
            L_i = r_i

        total_params = L_i.shape[1]
        Lambda = np.zeros((total_params, total_params))
        np.fill_diagonal(Lambda[:res_size, :res_size], lambda_ridge)
        np.fill_diagonal(Lambda[res_size:, res_size:], lambda_net)

        y_i = Y_target_deriv[:, i, :]

        A = np.dot(L_i.T, L_i) + Lambda
        B = np.dot(L_i.T, y_i)

        try:
            Xi = linalg.solve(A, B, assume_a='pos')
        except:
            Xi = linalg.lstsq(A, B)[0]

        W_readout_i = Xi[:res_size, :]
        res.set_readout_weights(W_readout_i.T)
        trained_reservoirs.append(res)

        if len(C_list) > 0:
            neighbor_params = Xi[res_size:, :]
            idx_ptr = 0
            for k, neighbor_idx in enumerate(potential_neighbors):
                w_block = neighbor_params[idx_ptr: idx_ptr + dim, :]
                valid_vals = []
                for d in range(dim):
                    if coupling_dims[d] > 0:
                        valid_vals.append(w_block[d, d])

                eff_weight_val = np.mean(valid_vals) if valid_vals else 0

                # 还原真实权重
                real_weight_val = eff_weight_val / coupling_strength

                W_pred_matrix[i, neighbor_idx] = real_weight_val
                idx_ptr += dim

    # === 阈值处理逻辑 ===
    if positive_only:
        W_pred_matrix = np.maximum(W_pred_matrix, 0.0)
    W_pred_matrix[W_pred_matrix < threshold] = 0.0
    print(f"参数辨识完成，已应用阈值截断 (Threshold={threshold})")

    return W_pred_matrix, trained_reservoirs


# ==========================================
# 4. 第二阶段：直接动力学预测
# ==========================================
def stage_2_direct_prediction(time_series, W_pred_real, trained_reservoirs, dt=0.01,
                              start_step=15000, pred_len=1000,
                              coupling_dims=[1, 0, 0], coupling_strength=0.2):
    print("\n" + "=" * 60)
    print("STAGE 2: 直接动力学预测 (Direct Prediction)")
    print("=" * 60)

    steps, N_nodes, dim = time_series.shape
    curr_u = time_series[start_step - 1].copy()
    predictions = np.zeros((pred_len, N_nodes, dim))

    print(f"开始闭环预测 (Steps={pred_len})...")

    for t in range(pred_len):
        curr_r_states = [res.state for res in trained_reservoirs]

        def get_derivative(u_val):
            phys_c = get_coupling_term(u_val, W_pred_real,
                                       coupling_strength=coupling_strength,
                                       coupling_mask=coupling_dims)
            d_dt = np.zeros_like(u_val)
            for i in range(N_nodes):
                intrinsic_dyn = trained_reservoirs[i].Wout @ curr_r_states[i]
                d_dt[i] = intrinsic_dyn + phys_c[i]
            return d_dt

        k1 = get_derivative(curr_u)
        u_k2 = curr_u + 0.5 * dt * k1
        k2 = get_derivative(u_k2)
        u_k3 = curr_u + 0.5 * dt * k2
        k3 = get_derivative(u_k3)
        u_k4 = curr_u + dt * k3
        k4 = get_derivative(u_k4)

        next_u = curr_u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        predictions[t] = next_u

        for i in range(N_nodes):
            trained_reservoirs[i].step(next_u[i])
        curr_u = next_u

    return predictions


# 5. 主程序入口


# ============================================================
# Main experiment
# ============================================================
if __name__ == "__main__":
    # Set M to the network size to be evaluated.
    # VPS is calculated only when M == 10; reconstruction metrics are
    # collected for every network size.
    M = 5  # Change to 10 for VPS runs; otherwise use the required network size
    N_RUNS = 20
    SEED_STEP = 10000

    cfg = get_config("llz_xy", "unknown", M)
    data_path = Path(cfg["data_file"])
    adj_path = Path(cfg["adj_file"])

    if not data_path.exists():
        raise FileNotFoundError(f"Time-series data not found: {data_path}")
    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacency matrix not found: {adj_path}")

    raw_data = np.load(data_path)
    true_adj = np.load(adj_path)
    if raw_data.shape[1] != M or true_adj.shape != (M, M):
        raise ValueError(
            f"Data/adjacency size mismatch: data={raw_data.shape}, "
            f"adjacency={true_adj.shape}, expected M={M}"
        )

    output_dir = Path(cfg["result_dir"]) / "20run_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = f"llz_xy_unknown_M{M}_20runs"
    raw_csv = output_dir / f"{file_stem}_raw.csv"
    summary_csv = output_dir / f"{file_stem}_summary.csv"

    columns = [
        "system", "topology", "M", "run", "seed_offset",
        "Error", "TPR", "TNR", "Precision", "VPS"
    ]
    rows = []
    print(f"Loaded data: {raw_data.shape}")
    print(f"System=llz_xy, topology=unknown, M={M}")
    print(f"Threshold={cfg['threshold']}")
    print(f"Running {N_RUNS} independent reservoir realizations.")
    print(f"Per-run results: {raw_csv}")

    # Write each completed run immediately, so completed results are retained
    # even if a later run is interrupted.
    with raw_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for run_idx in range(N_RUNS):
            run_no = run_idx + 1
            seed_offset = run_idx * SEED_STEP
            print(f"\n[Run {run_no}/{N_RUNS}] seed_offset={seed_offset}")

            W_predicted, trained_models = stage_1_predict_weights_blind(
                raw_data,
                dt=cfg["dt"],
                train_len=cfg["train_len"],
                sync_len=cfg["sync_len"],
                res_size=cfg["res_size"],
                lambda_ridge=cfg["lambda_ridge"],
                lambda_net=cfg["lambda_net"],
                coupling_dims=list(cfg["coupling_dims"]),
                coupling_strength=cfg["coupling_strength"],
                positive_only=True,
                threshold=cfg["threshold"],
                seed_offset=seed_offset,
            )

            tpr, tnr, precision, weight_error = evaluate_xu_metrics(
                true_adj,
                W_predicted,
                threshold=cfg["threshold"],
            )
            vps_value = ""

            # Dynamics prediction is required only for the three M=10 systems.
            if M == 10:
                y_pred_traj = stage_2_direct_prediction(
                    raw_data,
                    W_pred_real=W_predicted,
                    trained_reservoirs=trained_models,
                    dt=cfg["dt"],
                    start_step=cfg["train_len"],
                    pred_len=cfg["pred_len"],
                    coupling_dims=list(cfg["coupling_dims"]),
                    coupling_strength=cfg["coupling_strength"],
                )
                y_true_traj = raw_data[
                    cfg["train_len"]: cfg["train_len"] + cfg["pred_len"]
                ]
                vps_value, _ = calculate_vps(
                    y_true_traj, y_pred_traj, threshold=cfg["vps_threshold"]
                )

            row = {
                "system": "llz_xy",
                "topology": "unknown",
                "M": M,
                "run": run_no,
                "seed_offset": seed_offset,
                "Error": weight_error,
                "TPR": tpr,
                "TNR": tnr,
                "Precision": precision,
                "VPS": vps_value,
            }
            writer.writerow(row)
            f.flush()
            rows.append(row)
            if vps_value == "":
                print(
                    f"Run {run_no}: Error={weight_error:.8f}, "
                    f"TPR={tpr:.6f}, TNR={tnr:.6f}, Precision={precision:.6f}"
                )
            else:
                print(
                    f"Run {run_no}: Error={weight_error:.8f}, "
                    f"TPR={tpr:.6f}, TNR={tnr:.6f}, Precision={precision:.6f}, "
                    f"VPS={vps_value}"
                )

            # Release per-run model objects before constructing the next realization.
            if M == 10:
                del y_pred_traj, y_true_traj
            del trained_models, W_predicted
            gc.collect()

    # Sample standard deviation (ddof=1), matching the 20-realization summary.
    metrics = ["Error", "TPR", "TNR", "Precision"]
    if M == 10:
        metrics.append("VPS")
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "n", "mean", "std"])
        writer.writeheader()
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in rows], dtype=float)
            writer.writerow({
                "metric": metric,
                "n": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            })

    print("\n" + "=" * 65)
    print("20-run experiment completed.")
    print(f"Raw results: {raw_csv}")
    print(f"Summary:     {summary_csv}")
    print("=" * 65)
