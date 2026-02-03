def rk4n(H1, H2, H3, dt, psi):
    """
    One RK4 time step

    H1 = H(t)
    H2 = H(t + dt/2)
    H3 = H(t + dt)

    psi = wavefunction at time t
    """

    h = dt

    # k1
    k1 = h * fcn(H1, psi)
    x = psi + 0.5 * k1

    # k2
    k2 = h * fcn(H2, x)
    x = psi + 0.5 * k2

    # k3
    k3 = h * fcn(H2, x)
    x = psi + k3

    # k4
    k4 = h * fcn(H3, x)

    # update
    psi_new = psi + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    return psi_new