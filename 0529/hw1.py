import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager

# 設定中文字型
font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"  # 替換為宋體字型路徑
font_prop = font_manager.FontProperties(fname=font_path)
rcParams['font.sans-serif'] = [font_prop.get_name()]  # 使用字型名稱
rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# 系統參數
m = 1.0            # 質量 [ks^2/in]
xi = 0.05          # 阻尼比
k1 = 631.65        # 彈性剛度 [k/in]
k2 = 126.33        # 降伏後剛度 [k/in]
xy = 1.0           # 降伏位移 [in]

# 初始條件
x0 = 0.0           # 初始位移
v0 = 40.0          # 初始速度 [in/s]
dt = 0.005         # 時間間隔 [s]
n_steps = 6        # 計算步數

# 初始化陣列
t = np.linspace(0, dt * (n_steps - 1), n_steps)
x = np.zeros(n_steps)
v = np.zeros(n_steps)
a = np.zeros(n_steps)
Fs = np.zeros(n_steps)

# 初始剛度和阻尼
k = k1
c = 2 * xi * np.sqrt(k * m)

# 初始加速度
Fs[0] = k1 * x0
a[0] = (-c * v0 - Fs[0]) / m
x[0] = x0
v[0] = v0

# 平均加速度法常數
a0 = 1.0 / (dt ** 2)
a1 = 1.0 / (2 * dt)
ke = k + a0 * m + a1 * c  # 有效剛度（第一次迴圈使用 k1）

# 時間步迴圈
for i in range(1, n_steps):
    # 預測內力
    dp = m * (a0 * x[i - 1] + a1 * v[i - 1] + 0.25 * a[i - 1]) + \
         c * (a1 * x[i - 1] + 0.5 * v[i - 1]) + Fs[i - 1]
    
    dx = dp / ke
    x[i] = x[i - 1] + dx
    v[i] = v[i - 1] + dt * 0.5 * (a[i - 1] + (-c * v[i - 1] - Fs[i - 1]) / m)
    a[i] = (-c * v[i] - Fs[i - 1]) / m

    # 非線性處理：更新彈簧力與剛度
    if abs(x[i]) <= xy:
        Fs[i] = k1 * x[i]
        k = k1
    else:
        Fs[i] = k1 * xy + k2 * (x[i] - np.sign(x[i]) * xy)
        k = k2

# 繪圖
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, x, marker='o')
plt.title('位移 x(t)', fontproperties=font_prop)
plt.xlabel('時間 (秒)', fontproperties=font_prop)
plt.ylabel('位移 (英吋)', fontproperties=font_prop)

plt.subplot(2, 2, 2)
plt.plot(t, v, marker='o')
plt.title('速度 v(t)', fontproperties=font_prop)
plt.xlabel('時間 (秒)', fontproperties=font_prop)
plt.ylabel('速度 (英吋/秒)', fontproperties=font_prop)

plt.subplot(2, 2, 3)
plt.plot(t, a, marker='o')
plt.title('加速度 a(t)', fontproperties=font_prop)
plt.xlabel('時間 (秒)', fontproperties=font_prop)
plt.ylabel('加速度 (英吋/秒²)', fontproperties=font_prop)

plt.subplot(2, 2, 4)
plt.plot(t, Fs, marker='o')
plt.title('彈簧力 Fs(t)', fontproperties=font_prop)
plt.xlabel('時間 (秒)', fontproperties=font_prop)
plt.ylabel('彈簧力 (千磅)', fontproperties=font_prop)

plt.tight_layout()
plt.savefig("SDOF_Response.png")  # 儲存整體圖表為圖片
plt.show()

# F_s 對 x 圖
plt.figure()
plt.plot(x, Fs, marker='o')
plt.title('F_s 對 x 的關係', fontproperties=font_prop)
plt.xlabel('位移 x (英吋)', fontproperties=font_prop)
plt.ylabel('彈簧力 F_s (千磅)', fontproperties=font_prop)
plt.grid(True)
plt.savefig("Spring_Force_vs_Displacement.png")  # 儲存 F_s 對 x 圖為圖片
plt.show()

# 回傳計算結果供檢查
x, v, a, Fs
