# Re-import required libraries after execution environment reset
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.integrate import odeint

# === 讀入地震資料 ===
data = np.loadtxt("/Users/mac/Desktop/hw213/0602/Northridge.txt", skiprows=1)
time = data[:, 0]
acc = data[:, 1] * 9.81  # g → m/s²
dt = time[1] - time[0]
n_steps = len(time)

# === 結構與 TMD 參數 ===
m_floor = 8.46e7
m = m_floor
omega = np.array([0.4083, 1.1440, 1.6531])  # rad/s
zeta = np.array([0.0045, 0.0125, 0.0180])

# === 計算剛度：解聯立方程取得 k1, k2, k3 ===
lam = omega**2 * m
A = np.sum(lam)
B = lam[0]*lam[1] + lam[0]*lam[2] + lam[1]*lam[2]
Cval = lam[0] * lam[1] * lam[2]
def eqs_for_k(vars):
    k1_, k2_, k3_ = vars
    return [
        k1_ + 2*k2_ + 2*k3_ - A,
        k1_*k2_ + 2*k1_*k3_ + 3*k2_*k3_ - B,
        k1_*k2_*k3_ - Cval
    ]
initial_guess = [5e7, 8e7, 8e7]
k1_val, k2_val, k3_val = fsolve(eqs_for_k, initial_guess)

# === 結構剛度與質量矩陣 ===
K = np.array([
    [k1_val + k2_val, -k2_val, 0],
    [-k2_val, k2_val + k3_val, -k3_val],
    [0, -k3_val, k3_val]
])
M = np.diag([m, m, m])

# === 模態分析與正交化 ===
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(M) @ K)
idx = np.argsort(np.sqrt(np.real(eigvals)))
Phi = eigvecs[:, idx]
for i in range(3):
    Phi[:, i] /= np.sqrt(Phi[:, i].T @ M @ Phi[:, i])

# === 計算阻尼矩陣 ===
C = M @ Phi @ np.diag(2 * zeta * omega) @ (Phi.T @ M)

# === TMD 參數 ===
mu = 0.03
beta = 0.9592
zeta_d = 0.0857
omega_3 = omega[2]
m_d = mu * m
w_d = beta * omega_3
k_d = m_d * w_d**2
c_d = 2 * zeta_d * np.sqrt(m_d * k_d)

# === 擴增 M, K, C 矩陣為 4DOF ===
M4 = np.zeros((4, 4))
M4[:3, :3] = M
M4[3, 3] = m_d

K4 = np.zeros((4, 4))
K4[:3, :3] = K
K4[2, 2] += k_d
K4[2, 3] -= k_d
K4[3, 2] -= k_d
K4[3, 3] += k_d

C4 = np.zeros((4, 4))
C4[:3, :3] = C
C4[2, 2] += c_d
C4[2, 3] -= c_d
C4[3, 2] -= c_d
C4[3, 3] += c_d

# === 地震輸入（只對前三樓有影響）===
acc_interp = interp1d(time, acc, kind='linear', fill_value="extrapolate")
acc_uniform = acc_interp(time)
r = np.array([1, 1, 1, 0])

# === 整體運動方程（直接解 4DOF 系統）===
def system_eq(y, t, M, C, K, acc_func):
    u = y[:4]
    v = y[4:]
    acc = acc_func(t)
    a = np.linalg.solve(M, -C @ v - K @ u - M @ r * acc)
    return np.concatenate((v, a))

initial_conditions = np.zeros(8)  # [u1,u2,u3,uTMD,v1,v2,v3,vTMD]
acc_func = lambda t: np.interp(t, time, acc_uniform)
sol = odeint(system_eq, initial_conditions, time, args=(M4, C4, K4, acc_func))

# === 分離位移 ===
u1, u2, u3, uTMD = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

# === 繪圖並標記最大值 ===
colors = ['#E69F00', '#D55E00', '#CC79A7', '#0072B2']  # 為 TMD 增加顏色
labels = ["1F", "2F", "3F", "TMD"]
displacements = [u1, u2, u3, uTMD]

plt.figure(figsize=(15, 6))

for i, (u, label, color) in enumerate(zip(displacements, labels, colors)):
    # 繪製位移曲線
    plt.plot(time, u, label=label, color=color, linestyle='-' if i < 3 else '--')
    
    # 找到最大值的位置
    peak_value = np.max(np.abs(u))
    peak_time = time[np.argmax(np.abs(u))]
    actual_peak_value = u[np.argmax(np.abs(u))]  # 實際的尖峰值（可能為負）

    # 標記最大值並圈起來
    plt.scatter(peak_time, actual_peak_value, color="red", zorder=5)

# 圖表標題與標籤
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title("TMD 1 - Displacement with Peak Values")
plt.grid(True)
plt.legend()
plt.tight_layout()

# 儲存圖檔
plt.savefig("/Users/mac/Desktop/hw213/0602/TMD1_Displacement_with_Peaks.png", dpi=300)

# 顯示圖檔
plt.show()

# === 統計值 ===
def compute_stats(u, label):
    return {
        'label': label,
        'mean': np.mean(u),
        'rms': np.sqrt(np.mean(u**2)),
        'peak': np.max(np.abs(u))
    }

stats = [compute_stats(u1, '1F'),
         compute_stats(u2, '2F'),
         compute_stats(u3, '3F'),
         compute_stats(uTMD, 'TMD')]
stats
np.save('/Users/mac/Desktop/hw213/0602/time.npy', time)
np.save('/Users/mac/Desktop/hw213/0602/u1_hw5.npy', u1)
np.save('/Users/mac/Desktop/hw213/0602/u2_hw5.npy', u2)
np.save('/Users/mac/Desktop/hw213/0602/u3_hw5.npy', u3)

import pandas as pd

df_stats = pd.DataFrame(stats)
df_stats.to_csv("/Users/mac/Desktop/hw213/0602/TMD1_stats.csv", index=False)
print(df_stats)