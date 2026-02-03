def fcn(H_t, psi):
    """
    Right-hand side of TDSE

    dpsi/dt = -i H(t) psi
    """
    return -1j * (H_t @ psi)