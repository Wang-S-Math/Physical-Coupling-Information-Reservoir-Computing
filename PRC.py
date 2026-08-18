
import os
import numpy as np
from numpy import zeros, tanh
from rcnp import rcnp

from prc_configs import get_prc_config


# 2. 补充函数 qiufei0
def qiufei0(ww):
    wl = []
    N = ww.shape[0]
    for i in range(N):
        # 找出第 i 行中非零的元素索引 (即 i 的邻居)
        # 使用 != 0 以支持负权重（排斥耦合）
        neighbors = np.where(ww[i] != 0)[0]
        wl.append(neighbors)
    return wl


def CCB4_2_2_2(y1, y2, ww, nn=500, grla=1e-6, grg=0.7, pbj=0.95, xshl=0.03, grso=1.2, she=1000, ycn=10000, oo1=1,
               oo2=0.5):
    if ycn is None:
        ycn = y2.shape[0]

    l1, l2 = y1.shape
    mn = int(l2 / 3)  # 节点数量

    print(f"启动 CCB4_2_2_2 (加权输入耦合版): Nodes={mn}, nn={nn}, 训练长度={l1}, 预测长度={ycn}")

    B = []
    # 动态调整输入缩放
    max_val = np.max(np.abs(y1))
    if max_val == 0: max_val = 1
    grgs = grso / max_val

    # 获取每个节点的邻居列表
    wl = qiufei0(ww)

    # --- 1. 训练阶段 ---
    for ii in range(mn):
        B1 = rcnp()

        u_self = y1[:, ii * 3: (ii + 1) * 3]

        neighbor_inputs = []
        neighbors = wl[ii]
        for jj in neighbors:
            w_val = ww[ii, jj]  # 获取具体权重
            # 提取邻居的 x 分量并加权
            u_neigh = y1[:, jj * 3 + 0] * w_val
            neighbor_inputs.append(u_neigh.reshape(-1, 1))

        if neighbor_inputs:
            # 拼接: (Time, 3 + num_neighbors)
            u_full = np.hstack([u_self] + neighbor_inputs)
        else:
            u_full = u_self

        n1 = u_full.shape[1]  # 当前储层的输入维度

        try:
            # 参数: nn, input_dim, self_dim=3, neigh_dim=1, scale
            B1.win = B1.fwin3(nn, n1, 3, 1, grgs)
        except AttributeError:
            # 如果没有 fwin3，回退到标准 fwin
            B1.win = B1.fwin(nn, n1, grgs)

        # 设置谱半径等参数
        current_xshl = xshl
        if nn * current_xshl < 1.5:
            current_xshl = 3.0 / nn
        B1.setwres(nn, current_xshl, pbj)
        B1.bk = B1.fbk(nn, oo1, oo2)

        B1.train(u_full, grg)
        B.append(B1)

    # 计算输出权重 Wout
    # CCB4 中，每个储层只负责预测自身的 3 个变量
    for ii in range(mn):
        # 目标: 自身的未来状态
        target = y1[she:, ii * 3: (ii + 1) * 3]
        # 状态: 自身的储层状态 (忽略前 she 步)
        states = B[ii].RR[she - 1: -1]
        B[ii].getwout(target, states, grla)

    print('已求解 wout (加权输入版)')

    # --- 2. 预测阶段 ---
    yc = zeros((ycn, l2))
    rrduo = zeros((mn, nn))  # 所有储层的状态

    # 初始化状态
    for ii in range(mn):
        rrduo[ii] = B[ii].rr.copy()

    for zz in range(ycn):
        # A. 计算当前步的预测输出 yc[zz]
        # Output = Wout * r(t)
        for ii in range(mn):
            yc[zz, ii * 3: (ii + 1) * 3] = B[ii].out @ rrduo[ii]

        # B. 更新储层状态 r(t+1)
        for ii in range(mn):
            u_self_t = yc[zz, ii * 3: (ii + 1) * 3]

            # 2. 邻居部分 (来自刚刚的预测，并加权)
            u_neigh_list = []
            neighbors = wl[ii]
            for jj in neighbors:
                w_val = ww[ii, jj]
                # 邻居 jj 的 x 分量
                val_neigh = yc[zz, jj * 3 + 0] * w_val
                u_neigh_list.append(val_neigh)

            if u_neigh_list:
                u_full_t = np.hstack([u_self_t, np.array(u_neigh_list)])
            else:
                u_full_t = u_self_t

            # 储层更新公式
            linear_part = B[ii].win @ u_full_t + B[ii].wres @ rrduo[ii] + B[ii].bk
            rrduo[ii] = (1 - B[ii].grg) * rrduo[ii] + B[ii].grg * tanh(linear_part)

    print('预测完成')

    # 调用比较函数 (使用第一个储层对象的工具函数)
    vps = B[0].fbijiao(yc, y2, 0.5)
    print(f"VPS: {vps} steps")
    return B, yc


# ============================================================
# Main program / 主程序
# ============================================================
if __name__ == "__main__":

    # ============================================================
    # System selection / 系统选择
    # Change SYSTEM below to run another dynamical system.
    # 如果想查看不同系统的 PRC 预测结果，请修改下面的 SYSTEM。（修改这里）
    #
    # Available options / 可选系统:
    #   "llz_xx" -> Lorenz (x-x)
    #   "llz_xy" -> Lorenz (x*y)
    #   "lsl"    -> Rossler (x-x)
    # ============================================================
    SYSTEM = "lsl"  # Change me / 修改这里
    M = 10

    cfg = get_prc_config(SYSTEM, M)

    data_file = cfg["data_file"]
    weight_file = cfg["weight_file"]

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Data file not found / 找不到数据文件: {data_file}"
        )

    if not os.path.exists(weight_file):
        raise FileNotFoundError(
            f"Weight file not found / 找不到重构权重文件: {weight_file}"
        )

    raw_data = np.load(data_file)
    weight_matrix = np.load(weight_file)

    print("=" * 70)
    print(f"System / 系统: {SYSTEM}")
    print(f"Network size / 网络规模: M={M}")
    print(f"Data file: {data_file}")
    print(f"Weighted network file: {weight_file}")
    print("=" * 70)

    if raw_data.ndim == 3:
        steps, n_nodes, n_dim = raw_data.shape
        flat_data = raw_data.reshape(steps, n_nodes * n_dim)
    elif raw_data.ndim == 2:
        flat_data = raw_data
    else:
        raise ValueError(
            f"Unexpected data shape / 数据维度异常: {raw_data.shape}"
        )

    train_len = cfg["train_len"]

    if len(flat_data) <= train_len:
        raise ValueError(
            f"Insufficient data length: {len(flat_data)} <= {train_len}. "
            "The paper setting requires a complete training segment."
        )

    y1_train = flat_data[:train_len]

    available_test = flat_data[train_len:]
    pred_len = min(cfg["pred_len"], len(available_test))
    y2_test = available_test[:pred_len]

    print(f"Training length / 训练长度: {len(y1_train)}")
    print(f"Prediction length / 预测长度: {len(y2_test)}")

    models, predictions = CCB4_2_2_2(
        y1_train,
        y2_test,
        weight_matrix,
        nn=cfg["nn"],
        grla=cfg["grla"],
        grg=cfg["grg"],
        pbj=cfg["pbj"],
        xshl=cfg["xshl"],
        grso=cfg["grso"],
        she=cfg["she"],
        ycn=len(y2_test),
    )

    print(f"Prediction shape / 预测结果维度: {predictions.shape}")
    print("No result files are written by this script.")
    print("本代码不保存新的预测结果文件。")
    print("Done.")
