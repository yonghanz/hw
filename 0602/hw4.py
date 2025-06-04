import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

# 讀取地震數據 (Northridge.txt)
file_path = '/Users/mac/Desktop/hw213/0602/Northridge.txt'
data = np.loadtxt(file_path, skiprows=1)

# 提取時間和加速度數據
time = data[:, 0]
acc = data[:, 1] * 9.81  # 轉換為 m/s^2（假設加速度是 g，並將其轉換為 m/s^2）

# 2. 定義建築物三層樓模態參數，求剛度矩陣 K 並建立 M、C
m_floor = 8.46e7  # 每層質量 [kg]
m = m_floor

omega = np.array([0.4083, 1.1440, 1.6531])  # Rad/s
zeta = np.array([0.0045, 0.0125, 0.0180])  # 無因次

# 計算模態本徵值 λᵢ = ωᵢ² * m
lam = omega**2 * m
A = np.sum(lam)  # λ1 + λ2 + λ3
B = lam[0]*lam[1] + lam[0]*lam[2] + lam[1]*lam[2]
Cval = lam[0] * lam[1] * lam[2]

# 三個故事彈簧剛度 k1, k2, k3 滿足：
#   (1)  k1 + 2 k2 + 2 k3 = A
#   (2)  k1 k2 + 2 k1 k3 + 3 k2 k3 = B
#   (3)  k1 k2 k3 = Cval
def eqs_for_k(vars):
    k1_, k2_, k3_ = vars
    return [
        k1_ + 2*k2_ + 2*k3_ - A,
        k1_*k2_ + 2*k1_*k3_ + 3*k2_*k3_ - B,
        k1_*k2_*k3_ - Cval
    ]

# 給定一組初值 (可視情況微調)
initial_guess = [5e7, 8e7, 8e7]
k1_val, k2_val, k3_val = fsolve(eqs_for_k, initial_guess)

# 建立 3×3 剛度矩陣 K（剪力模型）
K = np.array([
    [k1_val + k2_val, -k2_val, 0],
    [-k2_val, k2_val + k3_val, -k3_val],
    [0, -k3_val, k3_val]
])

# 3×3 質量矩陣 M
M = np.diag([m, m, m])

# 計算物理阻尼矩陣 C：先做特徵分解，取得模態向量 Φ
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(M).dot(K))
idx = np.argsort(np.sqrt(np.real(eigvals)))  # 依頻率排序
Phi = eigvecs[:, idx]

# 質量正交化 (Φᵀ M Φ = I)
for i in range(3):
    norm_i = np.sqrt(Phi[:, i].T @ M @ Phi[:, i])
    Phi[:, i] /= norm_i

# 組裝物理阻尼矩陣 C = M Φ (2 ζᵢ ωᵢ) Φᵀ M
C = M @ Phi @ np.diag(2 * zeta * omega) @ (Phi.T @ M)

# =============================================================================
# 3. 無 TMD 時，採用模態疊代進行 3DOF 系統時程積分
#    每個模態 i：
#      q̈ᵢ + 2 ζᵢ ωᵢ q̇ᵢ + ωᵢ² qᵢ = - Γᵢ ü_g(t)
#    其中 Γᵢ = φᵢᵀ M r, r=[1,1,1]ᵀ
#    最後合成 u_physical = Φ · q_modal
# =============================================================================
r = np.ones(3)  # 參與向量
Gamma = np.array([Phi[:, i].T @ M @ r for i in range(3)])

dt = time[1] - time[0]  # ≈ 0.01 s
n_steps = len(time)

# 插值加速度到 uniform grid（本檔案即已為 0.01s）
acc_interp = interp1d(time, acc, kind='linear', fill_value="extrapolate")
acc_uniform = acc_interp(time)

# Newmark‐β (Average Accel) for SDOF
def newmark_sdof(omega_i, zeta_i, Gamma_i, acc_g, dt):
    n = len(acc_g)
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    # 等效參數
    k_i = omega_i**2
    c_i = 2 * zeta_i * omega_i
    m_i = 1.0  # 模態品質量已設定為 1

    # 初始加速度
    a[0] = -Gamma_i * acc_g[0]

    beta = 0.25
    gamma = 0.5

    # 常數係數
    a_eff = m_i/(beta*dt*dt) + c_i*(gamma/(beta*dt))
    b_eff = m_i/(beta*dt) + c_i*((gamma/beta) - 1)
    c_eff = ((1/(2*beta) - 1)*m_i + dt*c_i*((gamma/(2*beta)) - 1))
    k_eff = k_i + a_eff

    for i in range(1, n):
        P_eff = -Gamma_i*acc_g[i] + a_eff*u[i-1] + b_eff*v[i-1] + c_eff*a[i-1]
        u[i] = P_eff / k_eff
        a[i] = (u[i] - u[i-1])*(1/(beta*dt*dt)) - v[i-1]*(1/(beta*dt)) \
               - a[i-1]*(1/(2*beta) - 1)
        v[i] = v[i-1] + dt*((1-gamma)*a[i-1] + gamma*a[i])

    return u, v, a

# 求解三個模態 qᵢ(t)
U_modal = np.zeros((3, n_steps))
for i in range(3):
    uq, vq, aq = newmark_sdof(omega[i], zeta[i], Gamma[i], acc_uniform, dt)
    U_modal[i, :] = uq

# 合成物理位移 u_no_tmd = Φ · q_modal
u_no_tmd = Phi.dot(U_modal)  # shape = (3, n_steps)

# 將 u_no_tmd 的每一行分別對應到 u1, u2, u3
u1, u2, u3 = u_no_tmd[0, :], u_no_tmd[1, :], u_no_tmd[2, :]

# 計算統計值
def compute_stats(u, label):
    return {
        'label': label,
        'mean': np.mean(u),
        'rms': np.sqrt(np.mean(u**2)),
        'peak': np.max(np.abs(u))
    }

stats = [
    compute_stats(u1, '1F'),
    compute_stats(u2, '2F'),
    compute_stats(u3, '3F')
]

# (c) 繪製未裝設 TMD 時各樓層位移時程，並儲存圖片
plt.figure(figsize=(10, 5))
colors = ['#E69F00', '#D55E00', '#CC79A7']
for i in range(3):
    plt.plot(time, u_no_tmd[i, :], label=f'Floor {i+1}', color=colors[i])
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('(c) Floor Displacements Without TMD')
plt.legend(loc='upper right')
plt.grid(which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Floor_Displacements_Without_TMD.png")  # 儲存圖片

# 儲存每層樓的位移數據
np.save('/Users/mac/Desktop/hw213/0602/u1_hw4.npy', u1)
np.save('/Users/mac/Desktop/hw213/0602/u2_hw4.npy', u2)
np.save('/Users/mac/Desktop/hw213/0602/u3_hw4.npy', u3)

# 儲存統計數據為 CSV
import pandas as pd
df_stats = pd.DataFrame(stats)
df_stats.to_csv("/Users/mac/Desktop/hw213/0602/noTMD_stats.csv", index=False)
print(df_stats)