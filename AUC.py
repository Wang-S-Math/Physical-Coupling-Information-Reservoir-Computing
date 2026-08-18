import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from pathlib import Path


# 0. 评价指标 (Metrics)
def evaluate_xu_metrics(W_true, W_pred, threshold=0.01):
    N = W_true.shape[0]
    A_true = (np.abs(W_true) > 1e-6).astype(int)
    A_pred = (np.abs(W_pred) > threshold).astype(int)

    P_count = np.sum(A_true)
    TP_count = np.sum(A_true * A_pred)
    TPR = TP_count / P_count if P_count > 0 else 0.0

    N_neg_count = np.sum(1 - A_true)
    TN_count = np.sum((1 - A_true) * (1 - A_pred))
    TNR = TN_count / N_neg_count if N_neg_count > 0 else 0.0

    diff_norm = np.linalg.norm(W_pred - W_true, 'fro')
    true_norm = np.linalg.norm(W_true, 'fro')
    Error = diff_norm / true_norm if true_norm > 0 else 0.0

    return TPR, TNR, Error


def calculate_vps(y_true, y_pred, threshold=0.5):
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    length = min(len(y_true_flat), len(y_pred_flat))
    y_true_flat = y_true_flat[:length]
    y_pred_flat = y_pred_flat[:length]

    # 按列(特征)计算标准差用于归一化, shape=(Features,)
    std_val = np.std(y_true_flat, axis=0) + 1e-6

    # 使用展平后的变量进行计算
    diff = (y_true_flat - y_pred_flat) / std_val

    # 每个时间步的全局误差 (所有节点和维度的均方根)
    error_norm = np.sqrt(np.mean(diff ** 2, axis=1))

    # 找到第一个超过阈值的时间点
    indices = np.where(error_norm > threshold)[0]
    vps = indices[0] if len(indices) > 0 else length

    return vps, error_norm



def evaluate_threshold_metrics(W_true, W_pred):
    """
    只统计非对角元素 (i != j)，与论文修改后的指标定义一致。
    W_pred 应该是已经按当前阈值截断后的矩阵。
    """
    W_true = np.asarray(W_true, dtype=float)
    W_pred = np.asarray(W_pred, dtype=float)

    n = W_true.shape[0]
    mask = ~np.eye(n, dtype=bool)

    A_true = np.abs(W_true) > 1e-6
    A_pred = np.abs(W_pred) > 1e-15

    yt = A_true[mask]
    yp = A_pred[mask]

    TP = np.sum(yt & yp)
    FN = np.sum(yt & (~yp))
    TN = np.sum((~yt) & (~yp))
    FP = np.sum((~yt) & yp)

    TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    TNR = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan
    Precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan

    diff = (W_pred - W_true)[mask]
    true = W_true[mask]
    denom = np.linalg.norm(true)
    Error = np.linalg.norm(diff) / denom if denom > 0 else np.nan

    return TPR, TNR, FPR, Precision, Error, TP, TN, FP, FN


def threshold_sensitivity_analysis(
        W_true,
        W_raw,
        thresholds=np.linspace(0.0, 0.5, 5001),
        selected_threshold=0.005,
        prefix="threshold_sensitivity_llz_xx_M100"):
    """
    直接对同一份“未经阈值截断”的连续权重矩阵 W_raw 扫描 xi。
    不重新训练 PCI-RC。
    """

    thresholds = np.asarray(thresholds, dtype=float)
    rows = []

    for xi in thresholds:
        W_thr = W_raw.copy()
        W_thr[np.abs(W_thr) < xi] = 0.0

        TPR, TNR, FPR, Precision, Error, TP, TN, FP, FN = \
            evaluate_threshold_metrics(W_true, W_thr)

        rows.append([
            xi, TPR, TNR, FPR, Precision, Error,
            TP, TN, FP, FN
        ])

    results = np.asarray(rows, dtype=float)

    # 保存数值结果
    header = (
        "threshold,TPR,TNR,FPR,Precision,"
        "RelativeWeightError,TP,TN,FP,FN"
    )
    np.savetxt(
        prefix + ".csv",
        results,
        delimiter=",",
        header=header,
        comments=""
    )

    # 找到最接近当前论文阈值 xi=0.005 的点
    idx = np.argmin(np.abs(results[:, 0] - selected_threshold))
    sel = results[idx]

    print("\n" + "=" * 70)
    print("Threshold sensitivity analysis")
    print("=" * 70)
    print(f"Selected xi = {sel[0]:.6f}")
    print(f"TPR        = {sel[1]:.6f}")
    print(f"TNR        = {sel[2]:.6f}")
    print(f"FPR        = {sel[3]:.6f}")
    print(f"Precision  = {sel[4]:.6f}")
    print(f"Weight Err = {sel[5]:.6f}")
    print(
        f"TP/TN/FP/FN = "
        f"{int(sel[6])}/{int(sel[7])}/{int(sel[8])}/{int(sel[9])}"
    )
    print("=" * 70)

    # 阈值敏感性图：主图 0~0.5，右下角插图 0~0.01
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    # ---- 主图：Threshold xi = 0~0.5 ----
    ax.plot(
        results[:, 0], results[:, 1],
        label="TPR", linewidth=2.2
    )
    ax.plot(
        results[:, 0], results[:, 2],
        label="TNR", linewidth=2.2
    )
    ax.plot(
        results[:, 0], results[:, 4],
        label="Precision", linewidth=2.2
    )
    ax.axvline(
        selected_threshold,
        linestyle="--",
        linewidth=1.6,
        label=fr"Selected $\xi={selected_threshold}$"
    )

    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(np.arange(0.0, 0.51, 0.1))
    ax.set_xlabel(r"Threshold $\xi$", fontsize=16)
    ax.set_ylabel("Metric Value", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(
        frameon=False,
        fontsize=13,
        loc="lower left"
    )

    # ---- 右下角局部放大图：Threshold xi = 0~0.01 ----
    axins = inset_axes(
        ax,
        width="31%",
        height="31%",
        loc="lower right",
        borderpad=2.2
    )

    zoom_min = 0.0
    zoom_max = 0.01
    zoom_mask = (
        (results[:, 0] >= zoom_min) &
        (results[:, 0] <= zoom_max)
    )

    axins.plot(
        results[zoom_mask, 0],
        results[zoom_mask, 1],
        linewidth=1.8
    )
    axins.plot(
        results[zoom_mask, 0],
        results[zoom_mask, 2],
        linewidth=1.8
    )
    axins.plot(
        results[zoom_mask, 0],
        results[zoom_mask, 4],
        linewidth=1.8
    )
    axins.axvline(
        selected_threshold,
        linestyle="--",
        linewidth=1.3
    )

    # 标出 xi = 0.005 处三个指标
    axins.scatter(
        [selected_threshold] * 3,
        [sel[1], sel[2], sel[4]],
        s=18,
        zorder=5
    )

    axins.set_xlim(0.0, 0.01)
    axins.set_xticks([0.0, 0.005, 0.01])
    axins.set_xticklabels(["0", "0.005", "0.01"])

    # 自动放大低阈值区域的纵轴，确保细微差异可见
    zoom_values = np.concatenate([
        results[zoom_mask, 1],
        results[zoom_mask, 2],
        results[zoom_mask, 4]
    ])
    finite_zoom = zoom_values[np.isfinite(zoom_values)]

    if finite_zoom.size > 0:
        ymin = max(0.0, np.min(finite_zoom) - 0.03)
        axins.set_ylim(ymin, 1.005)
    else:
        axins.set_ylim(0.9, 1.005)

    axins.tick_params(axis="both", labelsize=7, pad=1)
    axins.set_title(
        r"$0 \leq \xi \leq 0.01$",
        fontsize=8,
        pad=1.5
    )

    fig.tight_layout(pad=1.0)

    output_figure = prefix + ".pdf"
    fig.savefig(
        output_figure,
        dpi=600,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Threshold sensitivity figure saved to: {output_figure}")

    return results


# 1. 核心类：统一的储备池 (Unified Reservoir)
class Reservoir:
    def __init__(self, input_dim, res_size, spectral_radius=0.95, leakage=0.25, input_scale=0.1, seed=42):
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


# 2.物理耦合计算
def get_coupling_term(u_state, adj_matrix, coupling_strength=1.0, coupling_mask=None):
    effective_adj = adj_matrix * coupling_strength
    weighted_sum = effective_adj @ u_state
    row_sums = np.sum(effective_adj, axis=1)[:, np.newaxis]
    self_term = row_sums * u_state
    full_coupling = weighted_sum - self_term

    if coupling_mask is not None:
        mask = np.array(coupling_mask).reshape(1, -1)
        full_coupling = full_coupling * mask

    return full_coupling


# 3. 第一阶段：盲预测参数辨识 (Blind Identification)
def stage_1_predict_weights_blind(time_series, dt=0.01, train_len=15000, sync_len=1000,
                                  res_size=500,
                                  lambda_ridge=1e-3,
                                  lambda_net=1e-7,
                                  coupling_dims=[1, 0, 0], coupling_strength=0.5,
                                  positive_only=True,
                                  threshold=0.001):
    """
    专门处理【未知结构】的情况。
    假设所有节点之间都可能存在连接（全连接假设），通过岭回归和阈值截断来重构网络。
    """
    print("\n" + "=" * 60)
    print("STAGE 1: 盲预测参数辨识 (Blind Parameter Identification)")
    print(f"模式: [未知结构/全连接假设] - 计算所有可能的连接并应用阈值。")
    if positive_only:
        print(f"约束: [Positive Only] - 强制权重非负。")
    print("=" * 60)

    steps, N_nodes, dim = time_series.shape
    coupling_mask = np.array(coupling_dims)
    U_curr = time_series[sync_len: train_len - 1]
    U_next = time_series[sync_len + 1: train_len]
    Y_target_deriv = (U_next - U_curr) / dt

    W_pred_matrix = np.zeros((N_nodes, N_nodes))
    trained_reservoirs = []

    for i in range(N_nodes):
        res = Reservoir(input_dim=dim, res_size=res_size, input_scale=0.05, seed=42 + i)
        r_full = res.run_states(time_series[:train_len, i, :])
        r_i = r_full[sync_len: train_len - 1]

        # === 核心逻辑：盲预测 ===
        # 不使用 known_structure，假设除自己外的所有节点都是潜在邻居
        potential_neighbors = np.delete(np.arange(N_nodes), i)

        C_list = []
        for j in potential_neighbors:
            diff = (U_curr[:, j, :] - U_curr[:, i, :]) * coupling_mask
            C_list.append(diff)

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

    # --- 新增：打印未经截断的权重 ---
    print("\n[Raw Weights] 未经阈值截断的预测连接权重矩阵:")
    with np.printoptions(precision=5, suppress=True):
        print(W_pred_matrix)
    print("-" * 60)


    # 保留未经阈值截断的连续重构权重
    W_pred_raw = W_pred_matrix.copy()

    # 当前论文设定阈值下的截断结果
    W_pred_thresholded = W_pred_raw.copy()
    W_pred_thresholded[np.abs(W_pred_thresholded) < threshold] = 0.0
    print(f"参数辨识完成，已应用阈值截断 (Threshold={threshold})")

    return W_pred_raw, W_pred_thresholded, trained_reservoirs


# ==========================================
# 4. 第二阶段：直接动力学预测
# ==========================================
def stage_2_direct_prediction(time_series, W_pred_real, trained_reservoirs, dt=0.01,
                              start_step=15000, pred_len=1000,
                              coupling_dims=[1, 0, 0], coupling_strength=0.5):
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


if __name__ == "__main__":
    # --- 配置 ---
    DATA_FILE = 'generate.data/llz.X-X/10coupled_lorenz_data.npy'
    ADJ_FILE = 'generate.data/llz.X-X/10coupled_lorenz_adj_thresholded.npy'

    COUPLING_STRENGTH = 0.5
    DT = 0.01
    TRAIN_LEN = 15000
    PRED_LEN = 1000

    try:
        raw_data = np.load(DATA_FILE)
        true_adj = np.load(ADJ_FILE)
        print(f"数据加载成功: {raw_data.shape}")

    except Exception as e:
        print(f"错误: 无法加载数据文件 ({e})。请确保文件上传且路径正确。")
        steps, N, dim = 20000, 5, 3
        raw_data = np.random.rand(steps, N, dim)
        true_adj = np.eye(N)

    # ===== 第一步：参数辨识 (纯盲预测) =====
    W_predicted_raw, W_predicted, trained_models = stage_1_predict_weights_blind(
        raw_data,
        dt=DT,
        train_len=TRAIN_LEN,
        res_size=500,
        coupling_dims=[1, 0, 0],
        coupling_strength=COUPLING_STRENGTH,
        positive_only=True,
        threshold=0.005,  # 截断噪声
        lambda_ridge=1e-3,
        lambda_net=1e-7
    )

    # ===== 保存未经阈值截断的结果 =====
    Path('result.date/llz.(x-x)/network').mkdir(parents=True, exist_ok=True)

    raw_weights_name = (
        'result.date/llz.(x-x)/network/'
        '100.(X-X)unknow_W_predicted_RAW.npy'
    )
    np.save(raw_weights_name, W_predicted_raw)
    print(f"\n[Auto-Save] 未经阈值截断的连续权重已保存至: {raw_weights_name}")

    # ===== 直接利用 raw weights 做阈值敏感性分析 =====
    # 这里只改变 xi，不重新训练 PCI-RC。
    threshold_results = threshold_sensitivity_analysis(
        W_true=true_adj,
        W_raw=W_predicted_raw,
        thresholds=np.linspace(0.0, 0.5, 5001),
        selected_threshold=0.005,
        prefix='threshold_ROC'
    )

    # ===== 第二步：直接动力学预测 =====
    y_pred_traj = stage_2_direct_prediction(
        raw_data,
        W_pred_real=W_predicted,
        trained_reservoirs=trained_models,
        dt=DT,
        start_step=TRAIN_LEN,
        pred_len=PRED_LEN,
        coupling_dims=[1, 0, 0],
        coupling_strength=COUPLING_STRENGTH
    )

    Path('result.date/llz.(x-x)/network').mkdir(parents=True, exist_ok=True)
    Path('result.date/llz.(x-x)/dynamic').mkdir(parents=True, exist_ok=True)

    save_weights_name = 'result.date/llz.(x-x)/network/10ROC1.(X-X)unknow_W_predicted.npy'
    save_dynamics_name = 'result.date/llz.(x-x)/dynamic/10ROC1.(X-X)unknow_y_pred_traj.npy'

    np.save(save_weights_name, W_predicted)
    np.save(save_dynamics_name, y_pred_traj)
    print(f"\n[Auto-Save] 网络权重矩阵已保存至: {save_weights_name}")
    print(f"[Auto-Save] 预测动力学数据已保存至: {save_dynamics_name}\n")

    y_true_traj = raw_data[TRAIN_LEN: TRAIN_LEN + PRED_LEN]

    # ===== 计算 VPS 和 误差 =====
    VPS_THRESHOLD = 0.5
    vps, error_norm = calculate_vps(y_true_traj, y_pred_traj, threshold=VPS_THRESHOLD)

    # ===== 评估报告 =====
    print("\n" + "=" * 80)
    print("                评估报告 (EVALUATION REPORT)")
    print("=" * 80)

    # 计算 TPR/TNR
    TPR, TNR, Weight_Error = evaluate_xu_metrics(true_adj, W_predicted, threshold=0.005)

    print(f"  1. TPR:   {TPR:.4f}")
    print(f"  2. TNR:   {TNR:.4f}")
    print(f"  3. Struct Error: {Weight_Error:.6f}")
    print("-" * 40)
    print(f"动力学预测指标:")
    print(f"  4. VPS (有效预测步数): {vps} steps (Threshold={VPS_THRESHOLD})")

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print("\n真实网络权重 (Ground Truth Wij):")
    print(true_adj)
    print("\n预测网络权重 (Predicted Wij):")
    print(W_predicted)
    print("=" * 80 + "\n")
    print("\n完成。")