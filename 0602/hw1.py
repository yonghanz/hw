import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import pandas as pd

# =============================================================================
# 1. 讀入地震歷時檔 (Northridge.txt)，跳過標頭，並轉換加速度單位
# =============================================================================
# 假設 Northridge.txt 第一列為標頭 "Time (s)  Acceleration (g)"，數值於第二列以後
data = np.loadtxt('/Users/mac/Desktop/hw213/0602/Northridge.txt', skiprows=1)
time    = data[:, 0]   # [s]
acc_g   = data[:, 1]   # [g]

g_const = 9.81
acc     = acc_g * g_const  # 轉成 [m/s²]

# (a) 繪製地震地表加速度時程圖，並儲存圖片
plt.figure(figsize=(10, 4))
plt.plot(time, acc, color='black', linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('(a) Ground Acceleration Time History')
plt.grid(which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Ground_Acceleration_Time_History.png")  # 儲存圖片

# (b) 計算地震地表加速度的平均值、均方根值、尖峰值
mean_acc = np.mean(acc)
rms_acc  = np.sqrt(np.mean(acc**2))
peak_acc = np.max(np.abs(acc))

print("(b) Ground Acceleration Statistics:")
print(f"    Mean (m/s²) = {mean_acc:.4f}")
print(f"    RMS  (m/s²) = {rms_acc:.4f}")
print(f"    Peak (m/s²) = {peak_acc:.4f}")

# =============================================================================
# 2. 定義建築物三層樓模態參數，求剛度矩陣 K 並建立 M、C
#    使用者給定：
#      - 質量 m = 8.46×10^7 kg（每層）
#      - 自然頻率 ω₁, ω₂, ω₃  (rad/s)
#      - 阻尼比     ζ₁, ζ₂, ζ₃
# =============================================================================
m_floor = 8.46e7  # 每層質量 [kg]
m       = m_floor

omega = np.array([0.4083, 1.1440, 1.6531])  # Rad/s
zeta  = np.array([0.0045, 0.0125, 0.0180])  # 無因次

# 計算模態本徵值 λᵢ = ωᵢ² * m
lam    = omega**2 * m
A      = np.sum(lam)                            # λ1 + λ2 + λ3
B      = lam[0]*lam[1] + lam[0]*lam[2] + lam[1]*lam[2]
Cval   = lam[0] * lam[1] * lam[2]

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
    [ k1_val + k2_val,    -k2_val,          0          ],
    [ -k2_val,            k2_val + k3_val, -k3_val    ],
    [ 0,                  -k3_val,          k3_val     ]
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
Gamma = np.array([ Phi[:, i].T @ M @ r for i in range(3) ])

dt      = time[1] - time[0]  # ≈ 0.01 s
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

    beta  = 0.25
    gamma = 0.5

    # 常數係數
    a_eff = m_i/(beta*dt*dt) + c_i*(gamma/(beta*dt))
    b_eff = m_i/(beta*dt)      + c_i*((gamma/beta) - 1)
    c_eff = ( (1/(2*beta) - 1)*m_i + dt*c_i*((gamma/(2*beta)) - 1) )
    k_eff = k_i + a_eff

    for i in range(1, n):
        P_eff = -Gamma_i*acc_g[i] + a_eff*u[i-1] + b_eff*v[i-1] + c_eff*a[i-1]
        u[i]   = P_eff / k_eff
        a[i]   = (u[i] - u[i-1])*(1/(beta*dt*dt)) - v[i-1]*(1/(beta*dt)) \
                 - a[i-1]*(1/(2*beta) - 1)
        v[i]   = v[i-1] + dt*((1-gamma)*a[i-1] + gamma*a[i])

    return u, v, a

# 求解三個模態 qᵢ(t)
U_modal = np.zeros((3, n_steps))
for i in range(3):
    uq, vq, aq = newmark_sdof(omega[i], zeta[i], Gamma[i], acc_uniform, dt)
    U_modal[i, :] = uq

# 合成物理位移 u_no_tmd = Φ · q_modal
u_no_tmd = Phi.dot(U_modal)  # shape = (3, n_steps)

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

# (e) 計算未裝 TMD 時各樓層位移之統計，並儲存資料
print("\n(e) Statistics Without TMD:")
statistics_no_tmd = []
for i in range(3):
    arr = u_no_tmd[i, :]
    mean_disp = np.mean(arr)
    rms_disp  = np.sqrt(np.mean(arr**2))
    peak_disp = np.max(np.abs(arr))
    statistics_no_tmd.append({
        'Label': f'Floor {i+1}',
        'Mean Displacement (m)': mean_disp,
        'RMS Displacement (m)': rms_disp,
        'Peak Displacement (m)': peak_disp
    })
    print(f"    Floor {i+1}: Mean = {mean_disp:.5f} m,  RMS = {rms_disp:.5f} m,  Peak = {peak_disp:.5f} m")

# 儲存未裝 TMD 的統計資料為 CSV
df_statistics_no_tmd = pd.DataFrame(statistics_no_tmd)

# =============================================================================
# 4. 裝設 TMD 時：建立 4×4 的 M_aug, C_aug, K_aug，以 Newmark‐β 直接積分
#    以第一組 TMD 參數（μ=0.03, 調諧頻率比=0.9592, ζ=0.0857）示範。
# =============================================================================
mu      = 0.03
m_t     = mu * m_floor
omega_t = 0.9592 * omega[0]
zeta_t  = 0.0857

k_t = m_t * omega_t**2
c_t = 2 * zeta_t * m_t * omega_t

# M_aug (4×4)
M_aug = np.zeros((4, 4))
M_aug[:3, :3] = M
M_aug[3, 3]   = m_t

# K_aug (4×4): TMD 串接在第 3 層 (index=2)
K_aug = np.zeros((4, 4))
K_aug[:3, :3] = K
K_aug[2, 2]  += k_t
K_aug[2, 3]   = -k_t
K_aug[3, 2]   = -k_t
K_aug[3, 3]   = k_t

# C_aug (4×4): 同樣在第 3 層串接阻尼 c_t
C_aug = np.zeros((4, 4))
C_aug[:3, :3] = C
C_aug[2, 2]  += c_t
C_aug[2, 3]   = -c_t
C_aug[3, 2]   = -c_t
C_aug[3, 3]   = c_t

# Newmark‐β for full N-DOF (4DOF)
def newmark_full(Mm, Cm, Km, acc_g, dt):
    n  = Mm.shape[0]
    nt = len(acc_g)
    u  = np.zeros((n, nt))
    v  = np.zeros((n, nt))
    a  = np.zeros((n, nt))

    beta  = 0.25
    gamma = 0.5

    # 有效剛質矩陣 (n×n)
    M_eff     = Mm + gamma*dt*Cm + beta*(dt**2)*Km
    M_eff_inv = np.linalg.inv(M_eff)

    # 初始加速度 (假設 u(0)=0, v(0)=0)
    a[:, 0] = np.linalg.inv(Mm) @ ( - Cm.dot(v[:, 0]) - Km.dot(u[:, 0]) - Mm.dot(np.ones(n)*acc_g[0]) )

    for i in range(1, nt):
        P_eff = - Mm.dot(np.ones(n)*acc_g[i]) \
                - Cm.dot( v[:, i-1] + (1-gamma)*dt*a[:, i-1] ) \
                - Km.dot( u[:, i-1] + dt*v[:, i-1] + 0.5*(1-2*beta)*(dt**2)*a[:, i-1] )
        u[:, i] = M_eff_inv.dot(P_eff)
        a[:, i] = ( u[:, i] - u[:, i-1] - dt*v[:, i-1] - 0.5*(1-2*beta)*(dt**2)*a[:, i-1] ) / (beta*(dt**2))
        v[:, i] = v[:, i-1] + dt*((1-gamma)*a[:, i-1] + gamma*a[:, i])

    return u, v, a

u_tmd, v_tmd, a_tmd = newmark_full(M_aug, C_aug, K_aug, acc_uniform, dt)

# (d) 繪製裝設 TMD 時，各樓層 + TMD 位移時程，並儲存圖片
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(time, u_tmd[i, :], label=f'Floor {i+1}', color=colors[i])
plt.plot(time, u_tmd[3, :], linestyle='--', color='#0072B2', label='TMD')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('(d) Floor and TMD Displacements With TMD (μ=0.03)')
plt.legend(loc='upper right')
plt.grid(which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Floor_and_TMD_Displacements_With_TMD.png")  # 儲存圖片

# (f) 計算裝設 TMD 時各樓層 + TMD 位移之統計，並儲存資料
labels = ['Floor 1', 'Floor 2', 'Floor 3', 'TMD']
statistics = []

print("\n(f) Statistics With TMD (μ=0.03):")
for i in range(4):
    arr = u_tmd[i, :]
    mean_disp = np.mean(arr)
    rms_disp  = np.sqrt(np.mean(arr**2))
    peak_disp = np.max(np.abs(arr))
    statistics.append({
        'Label': labels[i],
        'Mean Displacement (m)': mean_disp,
        'RMS Displacement (m)': rms_disp,
        'Peak Displacement (m)': peak_disp
    })
    print(f"    {labels[i]}: Mean = {mean_disp:.5f} m,  RMS = {rms_disp:.5f} m,  Peak = {peak_disp:.5f} m")

# 將統計結果儲存為 CSV 檔案
df_statistics = pd.DataFrame(statistics)
df_statistics.to_csv("TMD_Displacement_Statistics.csv", index=False)

plt.show()