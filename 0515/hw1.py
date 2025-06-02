import numpy as np
import matplotlib.pyplot as plt

# === 基本參數 ===
W = 50_000  # 重量 [lb] (1 kip = 1000 lb)
g = 386.09  # 重力加速度 [in/s^2]
M = W / g   # 質量 [lb·s²/in]
xg_ddot_peak_g = 0.25  # 峰值地震加速度 [g]
xg_ddot_peak = xg_ddot_peak_g * g  # 換算為 [in/s^2]
t0 = 0.75  # 波形轉折點 [s]
dt = 0.01  # 時間間隔 [s]
T_total = 6  # 總時間長度 [s]

# === 時間軸 ===
t = np.arange(0, T_total + dt, dt)

# === 地震加速度波形 ===
xg_ddot = np.zeros_like(t)
xg_ddot[(t >= 0) & (t < t0)] = xg_ddot_peak
xg_ddot[(t >= t0) & (t < 2 * t0)] = -xg_ddot_peak
# 超過 2t0 則為 0，已初始化為 0

# === 地震力 Fg(t) = -M * xg_ddot(t) ===
Fg = -M * xg_ddot

# === 計算速度 x'(t) 和位移 x(t) ===
x_dot = np.cumsum(xg_ddot) * dt  # 速度 (數值積分)
x = np.cumsum(x_dot) * dt       # 位移 (數值積分)

# === 繪圖 ===
plt.figure(figsize=(10, 10))

# 地震加速度波形
plt.subplot(4, 1, 1)
plt.plot(t, xg_ddot, label=r"$\ddot{x}_g(t)$ [in/s²]", color='blue')
plt.title("Ground Acceleration $\ddot{x}_g(t)$")
plt.ylabel("Acceleration [in/s²]")
plt.grid(True)
plt.legend()

# 等效地震力
plt.subplot(4, 1, 2)
plt.plot(t, Fg, label=r"$F_g(t) = -M \cdot \ddot{x}_g(t)$", color='red')
plt.title("Equivalent Earthquake Load $F_g(t)$")
plt.ylabel("Force [lb]")
plt.grid(True)
plt.legend()

# 速度 x'(t)
plt.subplot(4, 1, 3)
plt.plot(t, x_dot, label=r"$x'(t)$ [in/s]", color='green')
plt.title("Velocity $x'(t)$")
plt.ylabel("Velocity [in/s]")
plt.grid(True)
plt.legend()

# 位移 x(t)
plt.subplot(4, 1, 4)
plt.plot(t, x, label=r"$x(t)$ [in]", color='purple')
plt.title("Displacement $x(t)$")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [in]")
plt.grid(True)
plt.legend()

plt.tight_layout()

# 儲存圖表為檔案
plt.savefig("output_plot.png", dpi=300)  # 儲存為 PNG 格式，解析度為 300 DPI

plt.show()