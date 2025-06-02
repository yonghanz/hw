import numpy as np
import matplotlib.pyplot as plt

# 使用黑白樣式
plt.style.use('grayscale')

# System parameters
m = 1.0                      # mass [ks^2/in]
xi = 0.05                    # damping ratio
k1 = 631.65                  # initial stiffness [k/in]
k2 = 126.33                  # post-yield stiffness [k/in]
x_y = 1.0                    # yield displacement [in]
dt = 0.005                   # time step [s]
t_max = 5.0                  # total simulation time [s]

# Damping coefficient based on initial stiffness
c = 2 * xi * np.sqrt(k1 * m)

# Newmark-beta parameters (average acceleration method)
beta = 1/4
gamma = 1/2

# Time array
t = np.arange(0, t_max + dt, dt)

# Pre-allocate arrays
n_steps = len(t)
x = np.zeros(n_steps)
v = np.zeros(n_steps)
a = np.zeros(n_steps)
F_s = np.zeros(n_steps)

# Initial conditions
x[0] = 0.0
v[0] = 40.0  # initial velocity [in/s]

# Compute initial acceleration (no external force)
a[0] = (-c * v[0] - k1 * x[0]) / m
F_s[0] = k1 * x[0]

# Force-displacement function with bilinear hysteresis
def restore_force(x_disp, plast_disp):
    rel_disp = x_disp - plast_disp
    if abs(rel_disp) <= x_y:
        stiffness = k1
    else:
        stiffness = k2
    return stiffness * rel_disp

# Track plastic displacement
plast_disp = 0.0

# Effective stiffness for Newmark
k_eff = m / (beta * dt**2) + gamma * c / (beta * dt)

# Time stepping
for i in range(1, n_steps):
    # Predictor
    x_pred = x[i-1] + dt * v[i-1] + dt**2 * (0.5 - beta) * a[i-1]
    v_pred = v[i-1] + dt * (1 - gamma) * a[i-1]

    # Restoring force at predictor
    F_restore_pred = restore_force(x_pred, plast_disp)
    
    # Solve for new displacement
    rhs = m * x_pred / (beta * dt**2) - c * v_pred - F_restore_pred
    x_new = rhs / k_eff
    v_new = v_pred + gamma * dt * ((x_new - x_pred) * (1/(beta * dt**2)) - a[i-1])
    a_new = (x_new - x[i-1] - dt * v[i-1] - 0.5 * dt**2 * a[i-1]) * 2 / dt**2

    # Update plastic displacement
    rel_disp = x_new - plast_disp
    if abs(rel_disp) > x_y:
        plast_disp += np.sign(rel_disp) * (abs(rel_disp) - x_y)

    # Store
    x[i] = x_new
    v[i] = v_new
    a[i] = a_new
    F_s[i] = restore_force(x_new, plast_disp)

# Plot (a) x(t)
plt.figure()
plt.plot(t, x, linestyle='-', color='black')
plt.xlabel('t [s]')
plt.ylabel('x(t) [in.]')
plt.title('(a) Displacement Time History')
plt.grid(True)
plt.savefig("Displacement_Time_History.png")  # 儲存圖片

# Plot (b) x_dot(t)
plt.figure()
plt.plot(t, v, linestyle='--', color='black')
plt.xlabel('t [s]')
plt.ylabel('ẋ(t) [in./s]')
plt.title('(b) Velocity Time History')
plt.grid(True)
plt.savefig("Velocity_Time_History.png")  # 儲存圖片

# Plot (c) x_ddot(t)
plt.figure()
plt.plot(t, a, linestyle='-.', color='black')
plt.xlabel('t [s]')
plt.ylabel('ẍ(t) [in./s²]')
plt.title('(c) Acceleration Time History')
plt.grid(True)
plt.savefig("Acceleration_Time_History.png")  # 儲存圖片

# Plot (d) F_s(t)
plt.figure()
plt.plot(t, F_s, linestyle='-', color='black')
plt.xlabel('t [s]')
plt.ylabel('F_s(t) [k]')
plt.title('(d) Restoring Force Time History')
plt.grid(True)
plt.savefig("Restoring_Force_Time_History.png")  # 儲存圖片

# Plot (e) F_s vs x hysteresis
plt.figure()
plt.plot(x, F_s, linestyle='-', color='black')
plt.xlabel('x [in.]')
plt.ylabel('F_s(x) [k]')
plt.title('(e) Hysteresis Loop')
plt.grid(True)
plt.savefig("Hysteresis_Loop.png")  # 儲存圖片

plt.tight_layout()
plt.show()

