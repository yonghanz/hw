import numpy as np
import matplotlib.pyplot as plt

# === 匯入 Northridge 地震資料 ===
def load_earthquake(filename):
    time = []
    ag_g = []
    with open(filename, "r") as f:
        next(f)  # 跳過標題行
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    time.append(float(parts[0]))
                    ag_g.append(float(parts[1]))
    time = np.array(time)
    ag = np.array(ag_g) * 9.81  # g 轉 m/s²
    return time, ag

# === 執行線性加速度法分析 ===
def linear_acceleration_method(m_floor, t, ag, mu, beta, zeta, k_story=1.2e8, w3=1.6531):
    dt = t[1] - t[0]
    N = len(t)
    n_dof = 4  # 三層 + TMD

    m_tmd = mu * m_floor
    m = np.array([m_floor, m_floor, m_floor, m_tmd])
    w_tmd = beta * w3
    k_tmd = m_tmd * w_tmd**2
    c_tmd = 2 * zeta * np.sqrt(k_tmd * m_tmd)

    # 勁度矩陣 K
    K = np.array([
        [k_story, -k_story, 0, 0],
        [-k_story, 2*k_story, -k_story, 0],
        [0, -k_story, k_story + k_tmd, -k_tmd],
        [0, 0, -k_tmd, k_tmd]
    ])

    # 質量與阻尼矩陣
    M = np.diag(m)
    C = np.zeros((4, 4))
    C[3, 3] = c_tmd

    # 地震力
    P = -np.outer(ag, m)

    # 初始化
    u = np.zeros((N, n_dof))
    v = np.zeros((N, n_dof))
    a = np.zeros((N, n_dof))

    β = 1/6
    γ = 0.5
    K_eff = M / (β * dt**2) + γ / (β * dt) * C + K
    K_eff_inv = np.linalg.inv(K_eff)

    for i in range(1, N):
        du = u[i-1]
        dv = v[i-1]
        da = a[i-1]

        u_hat = du + dt * dv + dt**2 * (0.5 - β) * da
        v_hat = dv + dt * (1 - γ) * da

        RHS = P[i] + M @ (u_hat / (β * dt**2)) + C @ (v_hat * γ / (β * dt))
        u[i] = K_eff_inv @ RHS
        a[i] = (u[i] - u_hat) / (β * dt**2)
        v[i] = v_hat + γ * dt * a[i]

    return u  # 回傳位移歷史：每行為 [1F, 2F, 3F, TMD]

# === 主程式 ===
# 載入地震資料
time, ag = load_earthquake("Northridge.txt")
m_floor = 8.46e7  # 每層樓質量

# 定義三組 TMD 參數
TMDs = [
    {"name": "TMD1 (μ=0.03)", "mu": 0.03, "beta": 0.9592, "zeta": 0.0857},
    {"name": "TMD2 (μ=0.1)",  "mu": 0.1,  "beta": 0.8789, "zeta": 0.1527},
    {"name": "TMD3 (μ=0.2)",  "mu": 0.2,  "beta": 0.7815, "zeta": 0.2098},
]

# 模擬並繪圖
plt.figure(figsize=(12, 8))
for tmd in TMDs:
    u = linear_acceleration_method(
        m_floor, time, ag,
        mu=tmd["mu"],
        beta=tmd["beta"],
        zeta=tmd["zeta"]
    )
    plt.plot(time, u[:, 3], label=f"TMD 位移 - {tmd['name']}", linewidth=2)
    # 若要畫樓層位移，也可加上：
    # plt.plot(time, u[:, 2], label=f"3F - {tmd['name']}")

plt.title("TMD 位移歷時比較（不同參數）", fontsize=14)
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

