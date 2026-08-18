import numpy as np
import matplotlib.pyplot as plt
import os  # 添加 os 模块以处理路径


def generate_coupled_rossler_data(
        num_oscillators=10,
        steps=20000,
        dt=0.01,
        coupling_strength=0.05,
        sparsity=0.3,
        a=0.2, b=0.3, c=6  # 罗斯勒单系统混沌参数
):

    # 1. 随机生成连接矩阵
    adjacency_matrix = np.random.rand(num_oscillators, num_oscillators)

    # 应用稀疏度
    mask = np.random.rand(num_oscillators, num_oscillators) > (1 - sparsity)
    adjacency_matrix[mask] = 0
    np.fill_diagonal(adjacency_matrix, 0)  # 确保无自环

    # 创建阈值处理后的连接矩阵副本
    threshold = 0.01
    adjacency_matrix_thresholded = adjacency_matrix.copy()
    adjacency_matrix_thresholded[np.abs(adjacency_matrix_thresholded) < threshold] = 0
    # =======================================

    # 2. 初始化状态 (每个振子有 x, y, z 三个维度)
    current_state = np.random.rand(num_oscillators, 3) * 2

    # --- LLE 计算初始化 ---
    perturbation_size = 1e-8
    current_state_pert = current_state.copy()
    v0 = np.random.randn(num_oscillators, 3)
    v0 /= np.linalg.norm(v0)
    current_state_pert += perturbation_size * v0

    lle_sum = 0
    # ----------------------

    # 用于存储数据 (Steps, Oscillators, Dim)
    data_history = np.zeros((steps + 1, num_oscillators, 3))
    data_history[0] = current_state

    print(f"开始生成耦合罗斯勒数据 (使用阈值矩阵): {num_oscillators} 个振子, {steps} 个时间步...")

    # 3. 数值积分 (RK4 方法)
    for t in range(steps):
        def dynamics(state):
            x = state[:, 0]
            y = state[:, 1]
            z = state[:, 2]

            # 扩散耦合项 (作用于 x 分量)
            x_diff = x[None, :] - x[:, None]

            # === 修改：使用阈值后的矩阵计算耦合 ===
            coupling_term = np.sum(adjacency_matrix_thresholded * x_diff, axis=1)

            dx_dt = -y - z + coupling_strength * coupling_term
            dy_dt = x + a * y
            dz_dt = b + z * (x - c)

            return np.stack([dx_dt, dy_dt, dz_dt], axis=1)

        def rk4_step(s, h):
            k1 = dynamics(s)
            k2 = dynamics(s + 0.5 * h * k1)
            k3 = dynamics(s + 0.5 * h * k2)
            k4 = dynamics(s + h * k3)
            return s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 更新主轨道
        current_state = rk4_step(current_state, dt)
        # 更新扰动轨道
        current_state_pert = rk4_step(current_state_pert, dt)

        # 计算 LLE 增长率
        dist = np.linalg.norm(current_state_pert - current_state)
        lle_sum += np.log(dist / perturbation_size)

        # 重归一化
        current_state_pert = current_state + (current_state_pert - current_state) * (perturbation_size / dist)

        data_history[t + 1] = current_state

    final_lle = lle_sum / (steps * dt)
    time_series = data_history[1:]

    # 返回增加了 adjacency_matrix_thresholded
    return time_series, adjacency_matrix, adjacency_matrix_thresholded, final_lle


def calculate_synchronization_error(ts_data):

    #计算同步误差: 计算每个时间步所有振子状态的标准差均值。
    sync_error = np.mean(np.std(ts_data, axis=1), axis=1)
    return sync_error


# --- 主执行部分 ---
if __name__ == "__main__":
    np.random.seed(27)
    N_OSCILLATORS = 100
    STEPS = 30000
    DT = 0.01
    COUPLING = 0.05
    SPARSITY = 0.97# 越接近 1 越稀疏

    # 生成数据
    ts_data, true_weights, true_weights_thresholded, lle_val = generate_coupled_rossler_data(
        num_oscillators=N_OSCILLATORS,
        steps=STEPS,
        dt=DT,
        coupling_strength=COUPLING,
        sparsity=SPARSITY
    )

    # 计算同步误差
    sync_err_series = calculate_synchronization_error(ts_data)
    final_sync_err = np.mean(sync_err_series[-1000:])  # 取最后1000步的平均值

    # --- 1. 打印分析结果 ---
    print("\n" + "=" * 40)
    print("分析结果:")
    print(f"1. 最大李雅普诺夫指数 (LLE): {lle_val:.6f}")
    print(f"2. 最终平均同步误差: {final_sync_err:.6f}")

    if final_sync_err < 0.05:
        print("状态判定: 系统趋于同步 (Synchronized)")
    else:
        print("状态判定: 系统处于不同步状态 (Desynchronized)")

    if lle_val > 0.01:
        print("动力学判定: 混沌状态 (Chaotic)")
    else:
        print("动力学判定: 周期或稳定状态")
    print("=" * 40)

    # 确保目录存在
    save_dir = "../generate.data/lsl"
    os.makedirs(save_dir, exist_ok=True)

    file_ts = os.path.join(save_dir, "100coupled_rossler_data.npy")
    file_adj = os.path.join(save_dir, "100coupled_rossler_adj.npy")
    file_adj_thresh = os.path.join(save_dir, "100coupled_rossler_adj_thresholded.npy")

    np.save(file_ts, ts_data)
    np.save(file_adj, true_weights)
    np.save(file_adj_thresh, true_weights_thresholded)  # 保存阈值处理后的矩阵

    print("\n" + "=" * 40)
    print("数据已成功保存至本地:")
    print(f"1. 时间序列文件: {file_ts}")
    print(f"2. 原始连接矩阵: {file_adj}")
    print(f"3. 阈值连接矩阵: {file_adj_thresh}")
    print("=" * 40)

    # --- 3. 打印连接权重 ---
    print("\n真实连接矩阵权重 (Ground Truth - Thresholded):")
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    print(true_weights_thresholded)
    print("=" * 40)

    print("\n阈值处理后的连接矩阵权重 (Thresholded Adjacency Matrix Weights, threshold=0.001):")
    print(true_weights_thresholded)

    # --- 4. 可视化 ---
    print("\n正在生成可视化图表...")
    fig = plt.figure(figsize=(12, 4 * N_OSCILLATORS))
    gs = fig.add_gridspec(N_OSCILLATORS + 1, 2)

    colors = plt.cm.tab10(np.linspace(0, 1, N_OSCILLATORS))

    for i in range(N_OSCILLATORS):
        # 左侧：时间序列 (x 变量)
        ax_ts = fig.add_subplot(gs[i, 0])
        ax_ts.plot(ts_data[:15000, i, 0], lw=0.8, color=colors[i])
        ax_ts.set_ylabel(f'Osc {i + 1} (x)')
        ax_ts.grid(True, linestyle=':', alpha=0.6)
        if i == 0: ax_ts.set_title("Time Series Segment (x-axis)")

        # 右侧：吸引子 (x vs z)
        ax_ph = fig.add_subplot(gs[i, 1])
        ax_ph.plot(ts_data[:, i, 0], ts_data[:, i, 2], lw=0.2, alpha=0.6, color=colors[i])
        ax_ph.set_ylabel('z')
        ax_ph.grid(True, linestyle=':', alpha=0.6)
        if i == 0: ax_ph.set_title("Attractor Phase Portrait (x vs z)")

    # 底部：同步误差随时间的变化
    ax_sync = fig.add_subplot(gs[N_OSCILLATORS, :])
    ax_sync.plot(sync_err_series, color='black', lw=1)
    ax_sync.set_yscale('log')
    ax_sync.set_title("Synchronization Error Over Time (Log Scale)")
    ax_sync.set_xlabel("Time Steps")
    ax_sync.set_ylabel("Error E(t)")
    ax_sync.grid(True, which="both", linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()