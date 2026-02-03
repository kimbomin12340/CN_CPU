import numpy as np

def function(t, y):
    return (1 + 2*t)*np.sqrt(y)
ti = 0
yi = 1
h = 0.25

tr = 1
N = int((tr - ti) / h)

def RK4(ti, yi, h):
    k1 = function(ti, yi)
    k2 = function(ti + 1 / 2 * h, yi + 1 / 2 * k1 * h)
    k3 = function(ti + 1 / 2 * h, yi + 1 / 2 * k2 * h)
    k4 = function(ti + h, yi + k3 * h)
    return yi + 1/6 * (k1 + 2*k2 + 2*k3 + k4) * h
    yi /= np.linalg.norm(yi)
    ti += h


for it in range(nt):
    t = total_term[it]
    tm = t + 0.5 * t_int

    t_center = Tot_time / 2
    phase_env = pi * (tm - t_center) / Tot_time
    envelope = np.cos(phase_env) ** 4
    At = A0 * np.cos(w0 * (tm - t_center)) * envelope

    k2 = k - At
    Hk = genhk(k2)
    dHk = gendhk(k2)

    if it > 0:

        vec_up = RK4(t, vec_up, h)
        vec_down = RK4(t, vec_down, h)