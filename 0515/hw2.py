import numpy as np
import matplotlib.pyplot as plt

# 基本常數
g = 386.09  # in/s^2
W = 50      # kips
k = 100     # kips/in
xi = 0.12
xg0 = 0.25 * g  # in/s^2
t0 = 0.75       # seconds
dt = 0.01       # time step
theta = 1.4     # for Wilson method

# 時間軸
t = np.arange(0, 2 * t0 + dt, dt)
n = len(t)

# 計算質量與阻尼
m = W / g
omega_n = np.sqrt(k / m)
c = 2 * m * omega_n * xi

# 地震加速度
xg_ddot = np.ones(n) * xg0
xg_ddot[t >= t0] = -xg0

# 地震等效力
F_eff = -m * xg_ddot

# 初始化儲存結構
methods = ['average', 'linear', 'wilson', 'central']
results = {method: {'x': np.zeros(n), 'v': np.zeros(n), 'a': np.zeros(n)} for method in methods}

# 平均加速度法（Newmark: β = 1/4, γ = 1/2）
def newmark_average():
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    a[0] = (F_eff[0] - c * v[0] - k * x[0]) / m

    beta, gamma = 1/4, 1/2
    a1 = m / (beta * dt ** 2) + gamma * c / (beta * dt)
    a2 = m / (beta * dt) + (gamma / beta - 1) * c
    a3 = (1 / (2 * beta) - 1) * m + dt * (gamma / (2 * beta) - 1) * c

    keff = k + a1

    for i in range(n - 1):
        dp = F_eff[i + 1] + a1 * x[i] + a2 * v[i] + a3 * a[i]
        x[i + 1] = dp / keff
        a[i + 1] = (x[i + 1] - x[i]) / (beta * dt ** 2) - v[i] / (beta * dt) - a[i] / (2 * beta)
        v[i + 1] = v[i] + dt * ((1 - gamma) * a[i] + gamma * a[i + 1])
    return x, v, a

# 線性加速度法（Newmark: β = 1/6, γ = 1/2）
def newmark_linear():
    x, v, a = np.zeros(n), np.zeros(n), np.zeros(n)
    a[0] = (F_eff[0] - c * v[0] - k * x[0]) / m
    beta, gamma = 1/6, 1/2
    a1 = m / (beta * dt ** 2) + gamma * c / (beta * dt)
    a2 = m / (beta * dt) + (gamma / beta - 1) * c
    a3 = (1 / (2 * beta) - 1) * m + dt * (gamma / (2 * beta) - 1) * c
    keff = k + a1
    for i in range(n - 1):
        dp = F_eff[i + 1] + a1 * x[i] + a2 * v[i] + a3 * a[i]
        x[i + 1] = dp / keff
        a[i + 1] = (x[i + 1] - x[i]) / (beta * dt ** 2) - v[i] / (beta * dt) - a[i] / (2 * beta)
        v[i + 1] = v[i] + dt * ((1 - gamma) * a[i] + gamma * a[i + 1])
    return x, v, a

# Wilson-θ 法
def wilson_theta():
    x, v, a = np.zeros(n), np.zeros(n), np.zeros(n)
    a[0] = (F_eff[0] - c * v[0] - k * x[0]) / m
    for i in range(n - 1):
        dp = F_eff[i] + theta * (F_eff[i + 1] - F_eff[i])
        keff = m * (6 / (theta * dt) ** 2) + c * (3 / (theta * dt)) + k
        rhs = dp + m * (6 * x[i] / (theta * dt) ** 2 + 6 * v[i] / (theta * dt) + 3 * a[i]) + \
              c * (3 * x[i] / (theta * dt) + 2 * v[i] + 0.5 * theta * dt * a[i])
        x_theta = rhs / keff
        a_theta = (6 / (theta * dt) ** 2) * (x_theta - x[i]) - (6 / (theta * dt)) * v[i] - 3 * a[i]
        v_theta = v[i] + dt * (1 / 2) * (a[i] + a_theta)
        x[i + 1] = x[i] + dt * v[i] + dt ** 2 / 6 * (a[i] + 2 * a_theta)
        v[i + 1] = v_theta
        a[i + 1] = a_theta
    return x, v, a

# 中央差分法（顯式方法）
def central_difference():
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    a[0] = (F_eff[0] - c * v[0] - k * x[0]) / m
    x[1] = x[0] + dt * v[0] + 0.5 * dt ** 2 * a[0]
    for i in range(1, n - 1):
        x[i + 1] = (dt ** 2 / m * (F_eff[i] - c / dt * (x[i] - x[i - 1]) - k * x[i]) + 2 * x[i] - x[i - 1])
        v[i] = (x[i + 1] - x[i - 1]) / (2 * dt)
        a[i] = (x[i + 1] - 2 * x[i] + x[i - 1]) / dt ** 2
    return x, v, a

# 執行各方法
results['average']['x'], results['average']['v'], results['average']['a'] = newmark_average()
results['linear']['x'], results['linear']['v'], results['linear']['a'] = newmark_linear()
results['wilson']['x'], results['wilson']['v'], results['wilson']['a'] = wilson_theta()
results['central']['x'], results['central']['v'], results['central']['a'] = central_difference()

# 🔢 列出前6步的結果
print("First 6 time steps:")
for method in methods:
    print(f"\nMethod: {method}")
    print("t (s)\tx (in)\tv (in/s)\ta (in/s²)\tF_eff (kips)")
    for i in range(6):
        print(f"{t[i]:.2f}\t{results[method]['x'][i]:.5f}\t{results[method]['v'][i]:.5f}\t{results[method]['a'][i]:.5f}\t{F_eff[i]:.3f}")

# 📈 畫圖
for key in ['x', 'v', 'a']:
    plt.figure(figsize=(10, 4))
    for method in methods:
        plt.plot(t, results[method][key], label=method)
    plt.xlabel('Time (s)')
    plt.ylabel(key + '(t)')
    plt.title(f'Time History of {key}(t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# 🔨 畫 F_eff(t) 的圖
plt.figure(figsize=(10, 4))
plt.plot(t, F_eff, label=r'$F_{eff}(t)$', color='black')
plt.xlabel('Time (s)')
plt.ylabel(r'$F_{eff}$ (kips)')
plt.title('Effective Earthquake Force $F_{eff}(t)$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
import pandas as pd  # 確保已匯入 pandas

# 🔢 儲存前6步的結果為表格檔案
data = []
for method in methods:
    for i in range(6):
        data.append({
            'Method': method,
            'Time (s)': t[i],
            'Displacement x (in)': results[method]['x'][i],
            'Velocity v (in/s)': results[method]['v'][i],
            'Acceleration a (in/s²)': results[method]['a'][i],
            'Effective Force F_eff (kips)': F_eff[i]
        })

# 將資料轉換為 DataFrame
df = pd.DataFrame(data)

# 儲存為 CSV 檔案
df.to_csv("results_summary.csv", index=False)  # 儲存為 CSV 檔案

# 儲存為 Excel 檔案（可選）
df.to_excel("results_summary.xlsx", index=False)  # 儲存為 Excel 檔案

# 📈 儲存圖表
for key in ['x', 'v', 'a']:
    plt.figure(figsize=(10, 4))
    for method in methods:
        plt.plot(t, results[method][key], label=method)
    plt.xlabel('Time (s)')
    plt.ylabel(key + '(t)')
    plt.title(f'Time History of {key}(t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"time_history_{key}.png", dpi=300)  # 儲存圖表為 PNG 檔案
    plt.close()  # 關閉圖表，避免顯示

# 🔨 儲存 F_eff(t) 的圖
plt.figure(figsize=(10, 4))
plt.plot(t, F_eff, label=r'$F_{eff}(t)$', color='black')
plt.xlabel('Time (s)')
plt.ylabel(r'$F_{eff}$ (kips)')
plt.title('Effective Earthquake Force $F_{eff}(t)$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("effective_force.png", dpi=300)  # 儲存圖表為 PNG 檔案
plt.close()  # 關閉圖表