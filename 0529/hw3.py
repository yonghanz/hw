import numpy as np
import matplotlib.pyplot as plt

# 題目參數
m = 1.0             # ks^2/in
xi = 0.05           # 阻尼比
k1 = 631.65         # 初始剛度 [k/in]
k2 = 126.33         # 降伏後剛度 [k/in]
x_y = 1.0           # 降伏位移 [in]
dt = 0.005          # 時間間隔 [s]
n_steps = 6         # 前六步

# 初始條件
x = np.zeros(n_steps + 1)
v = np.zeros(n_steps + 1)
a = np.zeros(n_steps + 1)
Fs = np.zeros(n_steps + 1)
time = np.linspace(0, n_steps * dt, n_steps + 1)

x[0] = 0.0
v[0] = 40.0
Fs[0] = k1 * x[0]
a[0] = (Fs[0] - 2 * xi * np.sqrt(k1 * m) * v[0]) / m

# 平均加速度法常數
a0 = 1 / dt**2
a1 = 1 / (2 * dt)

# 時間步迴圈
for i in range(n_steps):
    # 確定當前剛度（根據歷史位移）
    if abs(x[i]) <= x_y:
        k_tangent = k1
    else:
        k_tangent = k2

    c = 2 * xi * np.sqrt(k_tangent * m)  # 更新阻尼（根據目前剛度）

    # 有效剛度
    k_eff = m * a0 + c * a1 + k_tangent

    # 有效載重（右手邊）
    rhs = m * (a0 * x[i] + a1 * v[i]) + c * (a1 * x[i] + 0.5 * v[i]) + Fs[i]

    # 位移更新
    x[i + 1] = rhs / k_eff

    # 彈簧力更新
    dx = x[i + 1]
    if abs(dx) <= x_y:
        Fs[i + 1] = k1 * dx
    else:
        sign = np.sign(dx)
        Fs[i + 1] = sign * (k1 * x_y + k2 * (abs(dx) - x_y))

    # 加速度與速度更新
    a[i + 1] = (Fs[i + 1] - c * v[i] - k_tangent * x[i + 1]) / m
    v[i + 1] = v[i] + dt * 0.5 * (a[i] + a[i + 1])

# -------------------------------
# 📊 圖表繪製
# -------------------------------
plt.figure(figsize=(12, 8))

# x(t)
plt.subplot(2, 3, 1)
plt.plot(time, x, 'o-', label='x(t)')
plt.xlabel('Time [s]')
plt.ylabel('Displacement [in]')
plt.title('Displacement x(t)')
plt.grid(True)

# v(t)
plt.subplot(2, 3, 2)
plt.plot(time, v, 'o-', label="v(t)", color='green')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [in/s]')
plt.title('Velocity v(t)')
plt.grid(True)

# a(t)
plt.subplot(2, 3, 3)
plt.plot(time, a, 'o-', label="a(t)", color='red')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [in/s²]')
plt.title('Acceleration a(t)')
plt.grid(True)

# Fs(t)
plt.subplot(2, 3, 4)
plt.plot(time, Fs, 'o-', label="Fs(t)", color='purple')
plt.xlabel('Time [s]')
plt.ylabel('Force Fs [k]')
plt.title('Spring Force Fs(t)')
plt.grid(True)

# Fs(x)
plt.subplot(2, 3, 5)
plt.plot(x, Fs, 'o-', label="Fs vs x", color='orange')
plt.xlabel('Displacement x [in]')
plt.ylabel('Force Fs [k]')
plt.title('Hysteresis Fs(x)')
plt.grid(True)

plt.tight_layout()
plt.show()
