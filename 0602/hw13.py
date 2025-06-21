import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 讀取時間和樓層位移數據
time = np.load('/Users/mac/Desktop/hw213/0602/time.npy')

# 讀取 hw4.py 的數據
u1_hw4 = np.load('/Users/mac/Desktop/hw213/0602/u1_hw4.npy')
u2_hw4 = np.load('/Users/mac/Desktop/hw213/0602/u2_hw4.npy')
u3_hw4 = np.load('/Users/mac/Desktop/hw213/0602/u3_hw4.npy')

# 讀取 hw5.py 的數據
u1_hw5 = np.load('/Users/mac/Desktop/hw213/0602/u1_hw5.npy')
u2_hw5 = np.load('/Users/mac/Desktop/hw213/0602/u2_hw5.npy')
u3_hw5 = np.load('/Users/mac/Desktop/hw213/0602/u3_hw5.npy')

# 讀取 hw6.py 的數據
u1_hw6 = np.load('/Users/mac/Desktop/hw213/0602/u1_hw6.npy')
u2_hw6 = np.load('/Users/mac/Desktop/hw213/0602/u2_hw6.npy')
u3_hw6 = np.load('/Users/mac/Desktop/hw213/0602/u3_hw6.npy')

# 讀取 hw7.py 的數據
u1_hw7 = np.load('/Users/mac/Desktop/hw213/0602/u1_hw7.npy')
u2_hw7 = np.load('/Users/mac/Desktop/hw213/0602/u2_hw7.npy')
u3_hw7 = np.load('/Users/mac/Desktop/hw213/0602/u3_hw7.npy')

# 定義計算統計值的函數
def compute_stats(u, label):
    return {
        'Label': label,
        'Mean': np.mean(u),
        'RMS': np.sqrt(np.mean(u**2)),
        'Peak': np.max(np.abs(u))
    }

# 計算統計值
stats = []

# No TMD
stats.append(compute_stats(u1_hw4, 'No TMD - 1F'))
stats.append(compute_stats(u2_hw4, 'No TMD - 2F'))
stats.append(compute_stats(u3_hw4, 'No TMD - 3F'))

# TMD1
stats.append(compute_stats(u1_hw5, 'TMD1 - 1F'))
stats.append(compute_stats(u2_hw5, 'TMD1 - 2F'))
stats.append(compute_stats(u3_hw5, 'TMD1 - 3F'))

# TMD2
stats.append(compute_stats(u1_hw6, 'TMD2 - 1F'))
stats.append(compute_stats(u2_hw6, 'TMD2 - 2F'))
stats.append(compute_stats(u3_hw6, 'TMD2 - 3F'))

# TMD3
stats.append(compute_stats(u1_hw7, 'TMD3 - 1F'))
stats.append(compute_stats(u2_hw7, 'TMD3 - 2F'))
stats.append(compute_stats(u3_hw7, 'TMD3 - 3F'))

# 將統計數據轉換為 DataFrame 並儲存為 CSV
df_stats = pd.DataFrame(stats)
df_stats.to_csv('/Users/mac/Desktop/hw213/0602/Displacement_Stats.csv', index=False)

# 輸出統計數據
print(df_stats)