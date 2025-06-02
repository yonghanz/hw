import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 基本參數
W = 50 * 1000       # [lb]
k = 100 * 1000      # [lb/in]
xi = 0.12
xddg0 = 0.25 * 386.4  # [in/s^2]
t0 = 0.75            # [s]
dt = 0.01            # [s]
g = 386.4            # [in/s^2]
m = W / g           # [slug]
c = 2 * xi * np.sqrt(k * m)

# 時間與地震輸入
n_step = 6
t = np.arange(n_step) * dt

xddg = np.zeros_like(t)
for i in range(n_step):
    if t[i] < t0:
        xddg[i] = xddg0
    elif t[i] < 2 * t0:
        xddg[i] = -xddg0
    else:
        xddg[i] = 0

F_eff = -m * xddg  # 等效外力

# 初始條件
u0 = 0
v0 = 0

# Newmark 方法
def newmark_solver(fe, m, c, k, u0, v0, dt, beta, gamma):
    n = len(fe)
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    u[0] = u0
    v[0] = v0
    a[0] = (fe[0] - c*v[0] - k*u[0]) / m

    a0 = 1 / (beta*dt**2)
    a1 = gamma / (beta*dt)
    a2 = 1 / (beta*dt)
    a3 = 1/(2*beta) - 1
    a4 = gamma/beta - 1
    a5 = dt*(gamma/(2*beta) - 1)
    keff = k + a0*m + a1*c

    for i in range(1, n):
        dp = fe[i] + m*(a0*u[i-1] + a2*v[i-1] + a3*a[i-1]) + \
                     c*(a1*u[i-1] + a4*v[i-1] + a5*a[i-1])
        u[i] = dp / keff
        v[i] = a1*(u[i] - u[i-1]) - a4*v[i-1] - a5*a[i-1]
        a[i] = a0*(u[i] - u[i-1]) - a2*v[i-1] - a3*a[i-1]
    return u, v, a

# Wilson-θ 方法
def wilson_theta(fe, m, c, k, u0, v0, dt, theta):
    n = len(fe)
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    u[0] = u0
    v[0] = v0
    a[0] = (fe[0] - c*v[0] - k*u[0]) / m

    khat = k + (3*c)/(theta*dt) + (6*m)/(theta**2*dt**2)

    for i in range(1, n):
        fe_theta = fe[i] + theta*(fe[i] - fe[i-1])
        dp = fe_theta + m*(6*u[i-1]/(theta**2*dt**2) + 6*v[i-1]/(theta*dt) + 3*a[i-1]) + \
             c*(3*u[i-1]/(theta*dt) + 3*v[i-1] + theta*dt*a[i-1])
        du_theta = dp / khat

        u_theta = u[i-1] + theta * du_theta
        v_theta = (u_theta - u[i-1]) / (theta * dt)
        a_theta = (v_theta - v[i-1]) / (theta * dt)

        u[i] = u[i-1] + dt*v[i-1] + dt**2*((1 - 0.5/theta)*a[i-1] + 0.5/theta*a_theta)
        v[i] = v[i-1] + dt*((1 - 1/theta)*a[i-1] + 1/theta*a_theta)
        a[i] = a[i-1] + (a_theta - a[i-1])/theta
    return u, v, a

# 中央差分法
def central_difference(fe, m, c, k, u0, v0, dt, n):
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    u[0] = u0
    v[0] = v0
    a[0] = (fe[0] - c*v[0] - k*u[0]) / m

    u_m1 = u[0] - dt*v[0] + 0.5*dt**2*a[0]

    for i in range(1, n):
        u[i] = (dt**2/m)*(fe[i] - c*(u[i-1] - u_m1)/(2*dt) - k*u[i-1]) + 2*u[i-1] - u_m1
        u_m1 = u[i-1]

    for i in range(1, n-1):
        v[i] = (u[i+1] - u[i-1]) / (2*dt)
        a[i] = (u[i+1] - 2*u[i] + u[i-1]) / (dt**2)
    return u, v, a

# 方法 1：AAM
u_aam, v_aam, a_aam = newmark_solver(F_eff, m, c, k, u0, v0, dt, beta=1/4, gamma=1/2)
# 方法 2：LAM
u_lam, v_lam, a_lam = newmark_solver(F_eff, m, c, k, u0, v0, dt, beta=1/6, gamma=1/2)
# 方法 3：Wilson-θ
u_wil, v_wil, a_wil = wilson_theta(F_eff, m, c, k, u0, v0, dt, theta=1.4)
# 方法 4：中央差分法
u_cdm, v_cdm, a_cdm = central_difference(F_eff, m, c, k, u0, v0, dt, n_step)

# 結果表格
T = pd.DataFrame({
    'Time': t,
    'F_eff': F_eff,
    'x_AAM': u_aam,
    'x_LAM': u_lam,
    'x_Wilson': u_wil,
    'x_CDM': u_cdm,
})
print("各方法位移結果（前六步）:")
print(T)

# 圖1：位移 x(t)
plt.figure()
plt.plot(t, u_aam, '-o', label='AAM')
plt.plot(t, u_lam, '-s', label='LAM')
plt.plot(t, u_wil, '-d', label='Wilson-θ')
plt.plot(t, u_cdm, '-^', label='CDM')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Displacement x(t) [in]')
plt.title('SDOF Response - Displacement')
plt.grid(True)

# 圖2：速度 v(t)
plt.figure()
plt.plot(t, v_aam, '-o', label='AAM')
plt.plot(t, v_lam, '-s', label='LAM')
plt.plot(t, v_wil, '-d', label='Wilson-θ')
plt.plot(t, v_cdm, '-^', label='CDM')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Velocity v(t) [in/s]')
plt.title('SDOF Response - Velocity')
plt.grid(True)

# 圖3：加速度 a(t)
plt.figure()
plt.plot(t, a_aam, '-o', label='AAM')
plt.plot(t, a_lam, '-s', label='LAM')
plt.plot(t, a_wil, '-d', label='Wilson-θ')
plt.plot(t, a_cdm, '-^', label='CDM')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration a(t) [in/s²]')
plt.title('SDOF Response - Acceleration')
plt.grid(True)

plt.show()