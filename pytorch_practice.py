import numpy as np
import matplotlib.pyplot as plt
eV   = 0.036749323858378
A0 = 0.00126
w0 = 0.26 * eV
A0 = 0.000126/w0
pi = np.pi

Tot_time = 2*pi/w0 * 20
nstep = 6000
t_int = Tot_time / nstep

t = np.linspace(0, Tot_time, nstep)
tm = t + 0.5 * t_int

t_center = Tot_time / 2

phase_env = pi * (tm - t_center) / Tot_time
envelope = np.cos(phase_env)**4

A_t = A0 * np.cos(w0*(tm - t_center)) * envelope
print(Tot_time/41.341374575751)
plt.figure(1,(9,4))
plt.plot(t, A_t)
plt.xlabel("Time (a.u.)")
plt.ylabel("A(t)")
plt.title("Vector Potential Pulse A(t)")
plt.grid(True)
plt.show()