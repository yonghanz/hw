import numpy as np
import matplotlib.pyplot as plt

# 題目參數
m = 1.0                # 質量 [ks^2/in]
xi = 0.05              # 阻尼比
k1 = 631.65            # 彈性剛度 [k/in]
k2 = 126.33            # 降伏後剛度 [k/in]
x_y = 1.0              # 降伏位移 [in]
dt = 0.005             # 時間間隔 [s]
n_steps = 6            # 前六個時間步

# 初始條件
x0 = 0.0               # 初始位移 [in]
v0 = 40.0              # 初始速度 [in/s]

# 衍生參數
omega_n = np.sqrt(k1 / m)
c = 2 * xi * np.sqrt(k1 * m)   # 阻尼係數 [k*s/in]

# 初始化陣列
x = np.zeros(n_steps + 1)
v = np.zeros(n_steps + 1)
a = np.zeros(n_steps + 1)
Fs = np.zeros(n_steps + 1)
time = np.linspace(0, n_steps * dt, n_steps + 1)

# 初始設定
x[0] = x0
v[0] = v0
Fs[0] = k1 * x0
a[0] = (Fs[0] - c * v[0]) / m

# 平均加速度法常數
a0 = 1 / dt**2
a1 = 1 / (2 * dt)

# 時間迴圈
for i in range(n_steps):
    # 決定當前剛度
    if abs(x[i]) < x_y:
        k_tangent = k1
    else:
        sign = np.sign(x[i])
        k_tangent = k2

    # 有效剛度
    k_eff = m * a0 + c * a1 + k_tangent

    # 有效力
    rhs = (m * (a0 * x[i] + a1 * v[i]) +
           c * (a1 * x[i] + 0.5 * v[i]) + Fs[i])

    # 解出下一步位移
    x[i + 1] = rhs / k_eff

    # 根據非線性模型更新 Fs
    dx = x[i + 1]
    if abs(dx) <= x_y:
        Fs[i + 1] = k1 * dx
    else:
        sign = np.sign(dx)
        Fs[i + 1] = sign * (k1 * x_y + k2 * (abs(dx) - x_y))

    # 更新加速度與速度
    a[i + 1] = (Fs[i + 1] - c * (v[i]) - k_tangent * x[i + 1]) / m
    v[i + 1] = v[i] + dt * 0.5 * (a[i] + a[i + 1])

# 繪圖
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(time, x, marker='o')
plt.title('x(t) 位移')
plt.xlabel('Time [s]')
plt.ylabel('Displacement [in]')

plt.subplot(2, 2, 2)
plt.plot(time, v, marker='o')
plt.title("x'(t) 速度")
plt.xlabel('Time [s]')
plt.ylabel('Velocity [in/s]')  # 修正此行

plt.subplot(2, 2, 3)
plt.plot(time, a, marker='o')
plt.title("x''(t) 加速度")
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [in/s²]')  # 修正此行

plt.subplot(2, 2, 4)
plt.plot(x, Fs, marker='o')
plt.title('F_s(x) 彈簧力-位移關係')
plt.xlabel('x(t) [in]')
plt.ylabel('F_s(t) [k]')

plt.tight_layout()
plt.show()
