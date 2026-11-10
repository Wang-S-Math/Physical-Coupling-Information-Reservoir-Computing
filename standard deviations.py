
import numpy as np
from pathlib import Path

# ==========================================
# 1. 读取原始无噪声数据
# 将 npy 文件放在本代码同一文件夹下
# ==========================================

data_path = (
    Path(__file__).resolve().parent
    / "generate.data"
    / "llz.xy"
    / "10coupled_lorenz_data.xy.npy"
)

data = np.load(data_path)

print("原始数据维度：", data.shape)

# ==========================================
# 2. 设置实验参数
# ==========================================

TRAIN_LEN = 15000
NOISE_STD = 0.10

# 取前15000个时间步
clean_data = data[:TRAIN_LEN]

# ==========================================
# 3. 分别计算 x、y、z 的标准差
# ==========================================

variables = ["x", "y", "z"]

print("\n========== 噪声水平计算结果 ==========")

for k, name in enumerate(variables):

    # 提取所有节点的第 k 个状态变量
    values = clean_data[:, :, k].reshape(-1)

    # 无噪声状态变量的标准差
    sigma = np.std(values, ddof=0)

    # 相对噪声水平
    ratio = NOISE_STD / sigma * 100

    print(f"\n{name} 变量：")
    print(f"无噪声标准差 = {sigma:.6f}")
    print(f"噪声标准差   = {NOISE_STD}")
    print(f"相对噪声水平 = {ratio:.4f}%")

print("\n========== 计算完成 ==========")
