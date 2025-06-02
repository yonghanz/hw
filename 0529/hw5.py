import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# System properties
m = 1.0  # ks^2/in
xi = 0.05
k1 = 631.65  # k/in
k2 = 126.33  # k/in
x_y = 1.0  # in
dt = 0.005  # s

# Derived quantities
omega_n = np.sqrt(k1/m)
c = 2 * m * omega_n * xi

# Newmark average acceleration parameters
beta = 1/4
gamma = 1/2
a0 = 1.0 / (beta * dt**2)
a1 = gamma / (beta * dt)
a2 = 1.0 / (beta * dt)
a3 = 1.0 / (2*beta) - 1
a4 = gamma/beta - 1
a5 = dt*(gamma/(2*beta) - 1)

# Initialize arrays
times = np.arange(0, 6*dt + 1e-12, dt)
n_steps = len(times)
x = np.zeros(n_steps)
v = np.zeros(n_steps)
a = np.zeros(n_steps)
F_spring = np.zeros(n_steps)

# Initial conditions
x[0] = 0.0
v[0] = 40.0  # in/s
a[0] = -(c*v[0] + (k1*x[0] if abs(x[0])<=x_y else k1*x_y + k2*(abs(x[0])-x_y))) / m

# Time-stepping
for i in range(n_steps-1):
    # Tangent stiffness
    k_t = k1 if abs(x[i]) <= x_y else k2
    # Effective stiffness
    K_eff = k_t + a1*c + a0*m
    # Effective load (no external load)
    P_eff = - (c*(a1*x[i] + a4*v[i] + a5*a[i]) + m*(a0*x[i] + a2*v[i] + a3*a[i]))
    dx = P_eff / K_eff
    x[i+1] = x[i] + dx
    a[i+1] = a0*dx - a2*v[i] - a3*a[i]
    v[i+1] = v[i] + dt*((1-gamma)*a[i] + gamma*a[i+1])
    # Spring force with bilinear law
    if abs(x[i+1]) <= x_y:
        F_spring[i+1] = k1 * x[i+1]
    else:
        F_spring[i+1] = np.sign(x[i+1])*(k1*x_y + k2*(abs(x[i+1]) - x_y))

# Prepare DataFrame
df = pd.DataFrame({
    'Time (s)': times,
    'Displacement x (in)': x,
    'Velocity v (in/s)': v,
    'Acceleration a (in/s^2)': a,
    'Spring Force Fs (k)': F_spring
})

# Display DataFrame (移除 ace_tools)
print("First Six Steps:")
print(df.head(6))

# Plotting time histories
plt.figure()
plt.plot(times, x)
plt.title("Displacement x(t)")
plt.xlabel("Time (s)")
plt.ylabel("x (in)")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(times, v)
plt.title("Velocity v(t)")
plt.xlabel("Time (s)")
plt.ylabel("v (in/s)")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(times, a)
plt.title("Acceleration a(t)")
plt.xlabel("Time (s)")
plt.ylabel("a (in/s^2)")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(times, F_spring)
plt.title("Spring Force Fs(t)")
plt.xlabel("Time (s)")
plt.ylabel("Fs (k)")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(x, F_spring)
plt.title("Force-Displacement Fs(x)")
plt.xlabel("x (in)")
plt.ylabel("Fs (k)")
plt.tight_layout()
plt.show()
