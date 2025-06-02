import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 參數設定
m = 8.46e7  # 每層質量 (kg)
omega_1 = 0.4083  # 一樓自然頻率 (rad/s)
omega_2 = 1.1440  # 二樓自然頻率 (rad/s)
omega_3 = 1.6531  # 三樓自然頻率 (rad/s)
zeta_1 = 0.0045  # 一樓阻尼比
zeta_2 = 0.0125  # 二樓阻尼比
zeta_3 = 0.0180  # 三樓阻尼比

# 阻尼器參數
mass_ratio = 0.03  # 阻尼質量比
tuning_ratio = 0.9592  # 調諧頻率比
damping_ratio = 0.0857  # 阻尼比

# 阻尼器的質量
m_damping = mass_ratio * m

# 阻尼器的頻率 (調諧)
omega_tuned = tuning_ratio * omega_3

# 阻尼器的阻尼
zeta_damping = damping_ratio

# 計算剛度矩陣
k1 = m * omega_1**2  # 一樓剛度 (N/m)
k2 = m * omega_2**2  # 二樓剛度 (N/m)
k3 = m * omega_3**2  # 三樓剛度 (N/m)

# 計算阻尼矩陣
c1 = 2 * zeta_1 * omega_1 * m  # 一樓阻尼 (N·s/m)
c2 = 2 * zeta_2 * omega_2 * m  # 二樓阻尼 (N·s/m)
c3 = 2 * zeta_3 * omega_3 * m  # 三樓阻尼 (N·s/m)

# 阻尼器的影響，假設加在第三樓層
c_damping = 2 * zeta_damping * omega_tuned * m_damping

# 剛度矩陣
K = np.array([[k1, -k1, 0], [-k1, k1 + k2, -k2], [0, -k2, k2 + k3]])

# 阻尼矩陣
C = np.array([[c1, -c1, 0], [-c1, c1 + c2, -c2], [0, -c2, c2 + c3 + c_damping]])

# 質量矩陣
M = np.array([[m, 0, 0], [0, m, 0], [0, 0, m]])

# 地震加速度資料 (以示範數據為例)
time = np.linspace(0, 30, 1000)
acceleration = np.sin(0.2 * np.pi * time)  # 假設為簡單正弦波的地震加速度歷時

# 定義系統的運動方程 (二階微分方程)
def system_eq(y, t, M, C, K, acceleration):
    u1, u2, u3, v1, v2, v3 = y
    acc = np.interp(t, time, acceleration)  # 插值計算加速度

    # 運動方程：M * d^2u/dt^2 + C * du/dt + K * u = F
    dxdt = [v1, v2, v3, 
            np.linalg.inv(M)[0].dot(acc - np.dot(C, [v1, v2, v3]) - np.dot(K, [u1, u2, u3])), 
            np.linalg.inv(M)[1].dot(acc - np.dot(C, [v1, v2, v3]) - np.dot(K, [u1, u2, u3])), 
            np.linalg.inv(M)[2].dot(acc - np.dot(C, [v1, v2, v3]) - np.dot(K, [u1, u2, u3]))]

    return dxdt

# 初始條件
initial_conditions = [0, 0, 0, 0, 0, 0]  # 初始位移和速度均為0

# 求解運動方程
solution = odeint(system_eq, initial_conditions, time, args=(M, C, K, acceleration))

# 提取各樓層位移
u1 = solution[:, 0]
u2 = solution[:, 1]
u3 = solution[:, 2]

# 繪製位移歷時圖
plt.figure(figsize=(10, 6))
plt.plot(time, u1, label='一樓位移 (m)')
plt.plot(time, u2, label='二樓位移 (m)')
plt.plot(time, u3, label='三樓位移 (m)')
plt.xlabel('時間 (秒)')
plt.ylabel('位移 (米)')
plt.title('三層樓建築物與調諧質量阻尼器位移歷時圖')
plt.legend()
plt.grid(True)
plt.show()

