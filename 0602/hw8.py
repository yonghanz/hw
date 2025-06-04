import numpy as np
import matplotlib.pyplot as plt

# 讀取時間和樓層位移數據
time = np.load('/Users/mac/Desktop/hw213/0602/time.npy')
u1_hw5 = np.load('/Users/mac/Desktop/hw213/0602/u1_hw5.npy')
u2_hw5 = np.load('/Users/mac/Desktop/hw213/0602/u2_hw5.npy')
u3_hw5 = np.load('/Users/mac/Desktop/hw213/0602/u3_hw5.npy')

u1_hw6 = np.load('/Users/mac/Desktop/hw213/0602/u1_hw6.npy')
u2_hw6 = np.load('/Users/mac/Desktop/hw213/0602/u2_hw6.npy')
u3_hw6 = np.load('/Users/mac/Desktop/hw213/0602/u3_hw6.npy')

u1_hw7 = np.load('/Users/mac/Desktop/hw213/0602/u1_hw7.npy')
u2_hw7 = np.load('/Users/mac/Desktop/hw213/0602/u2_hw7.npy')
u3_hw7 = np.load('/Users/mac/Desktop/hw213/0602/u3_hw7.npy')

# 繪製比較圖表
plt.figure(figsize=(12, 6))

# 比較 1F
plt.subplot(3, 1, 1)
plt.plot(time, u1_hw5, label='TMD1 - 1F', color='blue')
plt.plot(time, u1_hw6, label='TMD2 - 1F', color='orange')
plt.plot(time, u1_hw7, label='TMD3 - 1F', color='green')
plt.title('1F Displacement Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.grid()

# 比較 2F
plt.subplot(3, 1, 2)
plt.plot(time, u2_hw5, label='TMD1 - 2F', color='blue')
plt.plot(time, u2_hw6, label='TMD2 - 2F', color='orange')
plt.plot(time, u2_hw7, label='TMD3 - 2F', color='green')
plt.title('2F Displacement Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.grid()

# 比較 3F
plt.subplot(3, 1, 3)
plt.plot(time, u3_hw5, label='TMD1 - 3F', color='blue')
plt.plot(time, u3_hw6, label='TMD2 - 3F', color='orange')
plt.plot(time, u3_hw7, label='TMD3 - 3F', color='green')
plt.title('3F Displacement Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()