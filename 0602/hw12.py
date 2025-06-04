# Re-import libraries after execution environment reset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === 讀取地震資料 ===
data = np.loadtxt("/Users/mac/Desktop/hw213/0602/Northridge.txt", skiprows=1)
time = data[:, 0]
acc_g = data[:, 1]  # 單位為 g
acc_mps2 = acc_g * 9.81  # 換算為 m/s²

# === (a) 繪製地震地表加速度歷時圖 ===
plt.figure(figsize=(12, 5))
plt.plot(time, acc_mps2, label="Ground Acceleration", color="steelblue")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Northridge")
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/mac/Desktop/hw213/0602/Ground_Acceleration.png", dpi=300)
plt.show()

# === (b) 計算統計量（平均值、RMS、尖峰值）===
mean_acc = np.mean(acc_mps2)
rms_acc = np.sqrt(np.mean(acc_mps2**2))
peak_acc = np.max(np.abs(acc_mps2))

# 定義 stats 並存入統計數據
stats = [
    {"Metric": "Mean (m/s²)", "Value": mean_acc},
    {"Metric": "RMS (m/s²)", "Value": rms_acc},
    {"Metric": "Peak (m/s²)", "Value": peak_acc}
]

# 儲存統計數據為 CSV
df_stats = pd.DataFrame(stats)
df_stats.to_csv("/Users/mac/Desktop/hw213/0602/a.csv", index=False)
print(df_stats)