import numpy as np
import matplotlib.pyplot as plt
import os

def generate_coupled_lorenz_data(
        num_oscillators=10,
        steps=20000,
        dt=0.01,
        coupling_strength=0.5,
        sparsity=0.97
):
    # 1. 随机生成连接矩阵
    adjacency_matrix = np.random.rand(num_oscillators, num_oscillators)

    # 应用稀疏度（0，1）
    mask = np.random.rand(num_oscillators, num_oscillators) > (1 - sparsity)
    adjacency_matrix[mask] = 0
    np.fill_diagonal(adjacency_matrix, 0)  # 确保无自环

    # 创建阈值处理后的连接矩阵副本
    threshold = 0.001
    adjacency_matrix_thresholded = adjacency_matrix.copy()
    adjacency_matrix_thresholded[np.abs(adjacency_matrix_thresholded) < threshold] = 0

    # 2. 定义洛伦兹参数 (保持混沌状态)
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    # 3. 初始化状态
    current_state = np.random.rand(num_oscillators, 3) * 20 - 10

    # LLE 计算初始化
    perturbation_size = 1e-8
    current_state_pert = current_state.copy()
    v0 = np.random.randn(num_oscillators, 3)
    v0 /= np.linalg.norm(v0)
    current_state_pert += perturbation_size * v0

    lle_sum = 0
    # ---------------------------

    # 用于存储数据
    data_history = np.zeros((steps + 1, num_oscillators, 3))
    data_history[0] = current_state
    print(f"开始生成耦合洛伦兹数据 (X对X耦合, 使用阈值后的矩阵): {num_oscillators} 个振子, {steps} 个时间步...")

    # 4. 数值积分 (RK4 方法)
    for t in range(steps):
        def dynamics(state):
            x = state[:, 0]
            y = state[:, 1]
            z = state[:, 2]

            x_diff = x[None, :] - x[:, None]

            # 计算耦合项: k * sum(Aij * (xj - xi))
            coupling_term = np.sum(adjacency_matrix_thresholded * x_diff, axis=1)

            # 将耦合项加到 dx_dt 上
            dx_dt = sigma * (y - x) + coupling_strength * coupling_term
            dy_dt = x * (rho - z) - y
            dz_dt = x * y - beta * z

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

        # 存储
        data_history[t + 1] = current_state

    # 计算平均 LLE
    final_lle = lle_sum / (steps * dt)
    # 去掉初始状态
    time_series = data_history[1:]

    return time_series, adjacency_matrix, adjacency_matrix_thresholded, final_lle


def calculate_synchronization_error(ts_data):
    sync_error = np.mean(np.std(ts_data, axis=1), axis=1)
    return sync_error


# --- 主执行部分 ---
if __name__ == "__main__":
    np.random.seed(4)  # 您可以更改括号中的数字来获得不同的固定结果
    # 设置参数
    N_OSCILLATORS = 5
    STEPS = 20000
    DT = 0.01
    COUPLING = 0.5

    # 生成数据
    ts_data, true_weights, true_weights_thresholded, lle_val = generate_coupled_lorenz_data(
        num_oscillators=N_OSCILLATORS,
        steps=STEPS,
        dt=DT,
        coupling_strength=COUPLING,
        sparsity=0.6
    )

    # 计算同步误差
    sync_err_series = calculate_synchronization_error(ts_data)
    final_sync_err = np.mean(sync_err_series[-1000:])  # 取最后1000步的平均值

    # --- 1. 打印分析结果 ---
    print("\n" + "=" * 40)
    print("分析结果 (X耦合X模式 - 使用阈值矩阵生成):")
    print(f"1. 最大李雅普诺夫指数 (LLE): {lle_val:.6f}")
    print(f"2. 最终平均同步误差: {final_sync_err:.6f}")

    # 判定逻辑
    if final_sync_err < 0.1:  # 洛伦兹吸引子范围大(-20~20)，误差阈值可稍宽
        print("状态判定: 系统趋于同步 (Synchronized)")
    else:
        print("状态判定: 系统处于不同步状态 (Desynchronized)")

    if lle_val > 0.01:
        print("动力学判定: 混沌状态 (Chaotic)")
    else:
        print("动力学判定: 周期或稳定状态")
    print("=" * 40)

    # 保存数据
    os.makedirs("generate.data/llz.X-X", exist_ok=True)

    file_ts = "../generate.data/llz.X-X/5coupled_lorenz_data.npy"
    file_adj = "../generate.data/llz.X-X/5coupled_lorenz_adj.npy"
    file_adj_thresholded = "../generate.data/llz.X-X/5coupled_lorenz_adj_thresholded.npy"

    np.save(file_ts, ts_data)
    np.save(file_adj, true_weights)
    np.save(file_adj_thresholded, true_weights_thresholded)
    print(f"数据已保存至: {file_ts}")
    print(f"原始连接矩阵已保存至: {file_adj}")
    print(f"阈值处理后的连接矩阵已保存至: {file_adj_thresholded}")

    # 打印权重
    print("\n原始连接矩阵权重 (Adjacency Matrix Weights):")
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(true_weights)

    print("\n阈值处理后的连接矩阵权重 (Thresholded Adjacency Matrix Weights, threshold=0.001):")
    print(true_weights_thresholded)

    # --- 可视化 ---
    print("\n正在生成可视化图表...")
    fig = plt.figure(figsize=(12, 4 * N_OSCILLATORS))
    gs = fig.add_gridspec(N_OSCILLATORS + 1, 2)  # 多加一行给同步误差图

    colors = plt.cm.tab10(np.linspace(0, 1, N_OSCILLATORS))

    for i in range(N_OSCILLATORS):
        # 1. 左侧：时间序列 (x 变量)
        ax_ts = fig.add_subplot(gs[i, 0])
        ax_ts.plot(ts_data[:2000, i, 0], lw=0.8, color=colors[i])
        ax_ts.set_ylabel(f'Osc {i + 1} (x)')
        ax_ts.grid(True, linestyle=':', alpha=0.6)

        if i == 0:
            ax_ts.set_title("Time Series Segment (x-axis)")

        # 2. 右侧：吸引子 (x vs z)
        ax_ph = fig.add_subplot(gs[i, 1])
        ax_ph.plot(ts_data[:, i, 0], ts_data[:, i, 2], lw=0.2, alpha=0.6, color=colors[i])
        ax_ph.set_ylabel('z')
        ax_ph.grid(True, linestyle=':', alpha=0.6)

        if i == 0:
            ax_ph.set_title("Attractor Phase Portrait (x vs z)")

    # 3. 底部：同步误差随时间的变化
    ax_sync = fig.add_subplot(gs[N_OSCILLATORS, :])
    ax_sync.plot(sync_err_series, color='black', lw=1)
    ax_sync.set_yscale('log')
    ax_sync.set_title(f"Synchronization Error Over Time (Final Error={final_sync_err:.4f})")
    ax_sync.set_xlabel("Time Steps")
    ax_sync.set_ylabel("Error E(t) [Log Scale]")
    ax_sync.grid(True, which="both", linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()