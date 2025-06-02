import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 基本參數
W = 50 * 1000  # [lb]
k = 100 * 1000  # [lb/in]
xi = 0.12
xddg0 = 0.25 * 386.4  # [in/s^2]
t0 = 0.75  # [s]
dt = 0.01  # [s]
g = 386.4  # [in/s^2]
m = W / g  # [slug]
c = 2 * xi * np.sqrt(k * m)

# 時間與地震輸入
n_step = 6
t = np.arange(0, n_step * dt, dt)

xddg = np.zeros_like(t)
for i in range(len(t)):
    if t[i] < t0:
        xddg[i] = xddg0
    elif t[i] < 2 * t0:
        xddg[i] = -xddg0
    else:
        xddg[i] = 0

F_eff = -m * xddg

# 初始條件
u0, v0 = 0.0, 0.0

# ------------------- 方法定義 -------------------
def newmark_solver(fe, m, c, k, u0, v0, dt, beta, gamma):
    n = len(fe)
    u, v, a = np.zeros(n), np.zeros(n), np.zeros(n)
    u[0], v[0] = u0, v0
    a[0] = (fe[0] - c*v[0] - k*u[0]) / m

    a0 = 1 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1 / (beta * dt)
    a3 = 1/(2*beta) - 1
    a4 = gamma/beta - 1
    a5 = dt * (gamma/(2*beta) - 1)
    keff = k + a0*m + a1*c

    for i in range(1, n):
        dp = fe[i] + m*(a0*u[i-1] + a2*v[i-1] + a3*a[i-1]) + c*(a1*u[i-1] + a4*v[i-1] + a5*a[i-1])
        u[i] = dp / keff
        v[i] = a1*(u[i] - u[i-1]) - a4*v[i-1] - a5*a[i-1]
        a[i] = a0*(u[i] - u[i-1]) - a2*v[i-1] - a3*a[i-1]
    return u, v, a

def wilson_theta(fe, m, c, k, u0, v0, dt, theta):
    n = len(fe)
    u, v, a = np.zeros(n), np.zeros(n), np.zeros(n)
    u[0], v[0] = u0, v0
    a[0] = (fe[0] - c*v[0] - k*u[0]) / m
    khat = k + 3*c/(theta*dt) + 6*m/(theta**2 * dt**2)

    for i in range(1, n):
        fe_theta = fe[i] + theta * (fe[i] - fe[i-1])
        dp = fe_theta + m * (6*u[i-1]/(theta**2 * dt**2) + 6*v[i-1]/(theta*dt) + 3*a[i-1]) + \
             c * (3*u[i-1]/(theta*dt) + 3*v[i-1] + theta*dt*a[i-1])
        du_theta = dp / khat

        u_theta = u[i-1] + theta * du_theta
        v_theta = (u_theta - u[i-1]) / (theta * dt)
        a_theta = (v_theta - v[i-1]) / (theta * dt)

        u[i] = u[i-1] + dt*v[i-1] + dt**2 * ((1 - 0.5/theta)*a[i-1] + 0.5/theta*a_theta)
        v[i] = v[i-1] + dt * ((1 - 1/theta)*a[i-1] + 1/theta*a_theta)
        a[i] = a[i-1] + (a_theta - a[i-1])/theta
    return u, v, a

def central_difference(fe, m, c, k, u0, v0, dt, n):
    u, v, a = np.zeros(n), np.zeros(n), np.zeros(n)
    u[0], v[0] = u0, v0
    a[0] = (fe[0] - c*v[0] - k*u[0]) / m
    u_m1 = u[0] - dt*v[0] + 0.5*dt**2 * a[0]

    for i in range(1, n):
        u[i] = (dt**2 / m)*(fe[i] - c*(u[i-1] - u_m1)/(2*dt) - k*u[i-1]) + 2*u[i-1] - u_m1
        u_m1 = u[i-1]

    for i in range(1, n-1):
        v[i] = (u[i+1] - u[i-1]) / (2*dt)
        a[i] = (u[i+1] - 2*u[i] + u[i-1]) / (dt**2)
    return u, v, a

# ------------------- 計算 -------------------
u_aam, v_aam, a_aam = newmark_solver(F_eff, m, c, k, u0, v0, dt, beta=1/4, gamma=1/2)
u_lam, v_lam, a_lam = newmark_solver(F_eff, m, c, k, u0, v0, dt, beta=1/6, gamma=1/2)
u_wil, v_wil, a_wil = wilson_theta(F_eff, m, c, k, u0, v0, dt, theta=1.4)
u_cdm, v_cdm, a_cdm = central_difference(F_eff, m, c, k, u0, v0, dt, n_step)

# 整理 DataFrame
df = pd.DataFrame({
    "Time": t,
    "F_eff": F_eff,
    "x_AAM": u_aam,
    "x_LAM": u_lam,
    "x_Wilson": u_wil,
    "x_CDM": u_cdm,
    "v_AAM": v_aam,
    "v_LAM": v_lam,
    "v_Wilson": v_wil,
    "v_CDM": v_cdm,
    "a_AAM": a_aam,
    "a_LAM": a_lam,
    "a_Wilson": a_wil,
    "a_CDM": a_cdm,
})

# 儲存 DataFrame 並顯示
print("SDOF Response Comparison")
print(df)  # 在終端機中顯示 DataFrame

# 儲存為 CSV 檔案
df.to_csv("SDOF_Response_Comparison.csv", index=False)
print("DataFrame has been saved to 'SDOF_Response_Comparison.csv'")

# 顯示速度結果（前六步）
T_v = pd.DataFrame({
    "Time": t[:6],
    "F_eff": F_eff[:6],
    "v_AAM": v_aam[:6],
    "v_LAM": v_lam[:6],
    "v_Wilson": v_wil[:6],
    "v_CDM": v_cdm[:6]
})
print("\n各方法速度結果（前六步）:")
print(T_v)

# 顯示加速度結果（前六步）
T_a = pd.DataFrame({
    "Time": t[:6],
    "F_eff": F_eff[:6],
    "a_AAM": a_aam[:6],
    "a_LAM": a_lam[:6],
    "a_Wilson": a_wil[:6],
    "a_CDM": a_cdm[:6]
})
print("\n各方法加速度結果（前六步）:")
print(T_a)

# 繪圖 - 位移、速度、加速度
plt.figure()
plt.plot(t, u_aam, 'o-', label='AAM')
plt.plot(t, u_lam, 's-', label='LAM')
plt.plot(t, u_wil, 'd-', label='Wilson')
plt.plot(t, u_cdm, '^-', label='CDM')
plt.xlabel("Time [s]"); plt.ylabel("Displacement x(t) [in]")
plt.title("SDOF Response - Displacement"); plt.grid(); plt.legend()
plt.show()

plt.figure()
plt.plot(t, v_aam, 'o-', label='AAM')
plt.plot(t, v_lam, 's-', label='LAM')
plt.plot(t, v_wil, 'd-', label='Wilson')
plt.plot(t, v_cdm, '^-', label='CDM')
plt.xlabel("Time [s]"); plt.ylabel("Velocity v(t) [in/s]")
plt.title("SDOF Response - Velocity"); plt.grid(); plt.legend()
plt.show()

plt.figure()
plt.plot(t, a_aam, 'o-', label='AAM')
plt.plot(t, a_lam, 's-', label='LAM')
plt.plot(t, a_wil, 'd-', label='Wilson')
plt.plot(t, a_cdm, '^-', label='CDM')
plt.xlabel("Time [s]"); plt.ylabel("Acceleration a(t) [in/s²]")
plt.title("SDOF Response - Acceleration"); plt.grid(); plt.legend()
plt.show()