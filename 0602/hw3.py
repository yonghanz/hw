import numpy as np
import matplotlib.pyplot as plt

# === 匯入 Northridge 地震資料 ===
def load_earthquake(filename):
    """
    讀取地震資料檔案，並將加速度從 g 單位轉換為 m/s²。
    """
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
    ag = np.array(ag_g) * 9.81  # 將 g 單位轉換為 m/s²
    return time, ag

# === 模擬三層樓 + TMD（四自由度） ===
def linear_acceleration_method(m_floor, t, ag, mu, beta, zeta, k_story=1.2e8, w3=1.6531):
    dt = t[1] - t[0]
    N = len(t)
    n_dof = 4

    m_tmd = mu * m_floor
    m = np.array([m_floor, m_floor, m_floor, m_tmd])
    w_tmd = beta * w3
    k_tmd = m_tmd * w_tmd**2
    c_tmd = 2 * zeta * np.sqrt(k_tmd * m_tmd)

    K = np.array([
        [k_story, -k_story, 0, 0],
        [-k_story, 2*k_story, -k_story, 0],
        [0, -k_story, k_story + k_tmd, -k_tmd],
        [0, 0, -k_tmd, k_tmd]
    ])

    M = np.diag(m)
    C = np.zeros((4, 4))
    C[3, 3] = c_tmd

    P = -np.outer(ag, m)

    u = np.zeros((N, n_dof))
    v = np.zeros((N, n_dof))
    a = np.zeros((N, n_dof))

    β = 1 / 6
    γ = 0.5
    K_eff = M / (β * dt**2) + γ / (β * dt) * C + K
    K_eff_inv = np.linalg.inv(K_eff)

    for i in range(1, N):
        du, dv, da = u[i - 1], v[i - 1], a[i - 1]
        u_hat = du + dt * dv + dt**2 * (0.5 - β) * da
        v_hat = dv + dt * (1 - γ) * da
        RHS = P[i] + M @ (u_hat / (β * dt**2)) + C @ (v_hat * γ / (β * dt))
        u[i] = K_eff_inv @ RHS
        a[i] = (u[i] - u_hat) / (β * dt**2)
        v[i] = v_hat + γ * dt * a[i]

    return u

# === 模擬未裝設 TMD（三自由度） ===
def simulate_without_tmd(m_floor, t, ag, k_story=1.2e8):
    dt = t[1] - t[0]
    N = len(t)
    n_dof = 3
    m = np.array([m_floor, m_floor, m_floor])
    K = np.array([
        [k_story, -k_story, 0],
        [-k_story, 2 * k_story, -k_story],
        [0, -k_story, k_story]
    ])
    M = np.diag(m)
    C = np.zeros((n_dof, n_dof))
    P = -np.outer(ag, m)

    u = np.zeros((N, n_dof))
    v = np.zeros((N, n_dof))
    a = np.zeros((N, n_dof))

    β = 1 / 6
    γ = 0.5
    K_eff = M / (β * dt ** 2) + γ / (β * dt) * C + K
    K_eff_inv = np.linalg.inv(K_eff)

    for i in range(1, N):
        du, dv, da = u[i - 1], v[i - 1], a[i - 1]
        u_hat = du + dt * dv + dt ** 2 * (0.5 - β) * da
        v_hat = dv + dt * (1 - γ) * da
        RHS = P[i] + M @ (u_hat / (β * dt ** 2)) + C @ (v_hat * γ / (β * dt))
        u[i] = K_eff_inv @ RHS
        a[i] = (u[i] - u_hat) / (β * dt ** 2)
        v[i] = v_hat + γ * dt * a[i]

    return u

# === 主程式 ===
time, ag = load_earthquake("/Users/mac/Desktop/hw213/0602/Northridge.txt")
m_floor = 8.46e7  # 每層樓質量

# 三組 TMD 參數
TMDs = [
    {"name": "TMD1 (μ=0.03)", "mu": 0.03, "beta": 0.9592, "zeta": 0.0857},
    {"name": "TMD2 (μ=0.1)",  "mu": 0.1,  "beta": 0.8789, "zeta": 0.1527},
    {"name": "TMD3 (μ=0.2)",  "mu": 0.2,  "beta": 0.7815, "zeta": 0.2098},
]

# 繪圖
plt.figure(figsize=(12, 8))
u_no_tmd = simulate_without_tmd(m_floor, time, ag)
plt.plot(time, u_no_tmd[:, 2], label="3F - 無 TMD", color="black", linewidth=2.5, linestyle=":")

for tmd in TMDs:
    u = linear_acceleration_method(m_floor, time, ag, tmd["mu"], tmd["beta"], tmd["zeta"])
    plt.plot(time, u[:, 2], label=f"3F - {tmd['name']}", linewidth=2)

plt.title("3F 位移歷時比較（含無 TMD）", fontsize=14)
plt.xlabel("Time (s)")
plt.ylabel("Displacement of 3F (m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
