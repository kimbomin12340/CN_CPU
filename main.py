import numpy as np

# ============================================================
# modconstants.f90 (exact translation)
# ============================================================
astr = 1.8897259886          # bohr / angstrom
eV   = 0.036749323858378     # Hartree
pi   = np.pi
fs   = 41.341374575751
sol  = 137.0

kb = 8.6173e-5 * eV
delta = 1e-6
ci = 1j
c0 = 0.0 + 0.0j
c1 = 1.0 + 0.0j
e_fermi = 0*eV
w_func = 0*eV

# ============================================================
# lattice & TB parameters (modmain 대응)
# ============================================================
nrmax = 3

a = 2.46 * astr

a_1 = a * np.array([1.0, 0.0, 0.0])
a_2 = a * np.array([0.5, np.sqrt(3.0)/2.0, 0.0])

b_1 = (2*pi/a) * np.array([1.0, -1.0/np.sqrt(3.0), 0.0])
b_2 = (2*pi/a) * np.array([0.0,  2.0/np.sqrt(3.0), 0.0])

tau_1_vec = (2.0 * a_2 - a_1) / 3.0

# graphene monolayer atomic positions (A, B)
atom_pos = np.array([
    tau_1_vec,          # atom 1
    np.zeros(3)         # atom 2
])

# ============================================================
# Slater–Koster parameters
# ============================================================
# 에너지 scale parameters
v_pppi_0  = -2.7 * eV # 평면 내 Pz-Pz hopping
v_ppsgm_0 =  0.48 * eV # 층간 결합 ?? (그래핀은 2차원 물질 아닌가? 여기서 말하는 층은?)

#길이 scale parameters
dlt_0 = 0.184 * a # hopping의 decay length
a_0   = a / np.sqrt(3.0) # 그래핀 C-C 결합 길이 (평면 내 pi hopping의 기준거리)
d_0   = 3.35 * astr # graphite 층간거리 (sgm hopping의 기준거리)

# ============================================================
# unit cell area - 외적 크기
# ============================================================
cell_area = np.linalg.norm(np.cross(a_1, a_2))

# ============================================================
# hpp(d_vec)  --- exact Fortran translation
# ============================================================
def hpp(d_vec): # 두 atom 사이 거리 vec
    d  = np.linalg.norm(d_vec) # 거리 크기
    dz = d_vec[2] # z축 성분

    if d < delta: # hopping이라고 분류할 수 있는 최소 거리?
        return 0.0

    term_pi = (
        -v_pppi_0
        * np.exp(-(d - a_0) / dlt_0)
        * (1.0 - (dz / d)**2)
    )

    term_sg = (
        -v_ppsgm_0
        * np.exp(-(d - d_0) / dlt_0)
        * (dz / d)**2
    )

    return term_pi + term_sg

# ============================================================
# genhk(k_vec)  --- exact Fortran translation
# ============================================================
def genhk(k_vec): # k-공간에서의 hamiltonian
    """
    graphene tight-binding Hamiltonian H(k)

    k_vec    : complex ndarray (3, )
    atom_pos : ndarray (norb, 3)
    """

    norb = atom_pos.shape[0] # in u.c atom 수 알려줌 => 2
    h = np.zeros((norb, norb), dtype=np.complex128) # atom 수 만큼 정사각 행렬 생성

    cutoff = 1.1 * np.linalg.norm(tau_1_vec) # atom 1의 위치 벡터의 크기

    for i1 in range(norb): # range: 0,1
        for i2 in range(norb):
            if i1 != i2:
                for ir1 in range(-nrmax, nrmax + 1): # nrmax = 3으로 설정함 periodic boundary
                    for ir2 in range(-nrmax, nrmax + 1):

                        lr_vec = ir1 * a_1 + ir2 * a_2 # unit cell vector
                        d_vec  = lr_vec + atom_pos[i2] - atom_pos[i1] # i1 -> i2로 가는 hopping vector

                        if np.linalg.norm(d_vec) > cutoff:
                            continue

                        # cphase = k · d
                        cphase = (
                            k_vec[0]*d_vec[0]
                          + k_vec[1]*d_vec[1]
                          + k_vec[2]*d_vec[2]
                        )

                        phase = (
                            np.exp(1j * np.real(cphase))
                            * np.exp(-np.imag(cphase))
                        )

                        h[i1, i2] -= hpp(d_vec) * phase # hopping term * phase term => k-space에서의 hamiltonian

    # onsite term
    for i in range(norb):
        h[i, i] -= (e_fermi + w_func)

    return h

def genkline(hspts, n_p_to_p_grid):
    """
    Exact Python translation of Fortran subroutine genkline

    Parameters
    ----------
    hspts : ndarray (nhspts, 3)
        high-symmetry points
    n_p_to_p_grid : ndarray (nhspts-1, )
        number of points between hspts

    Returns
    -------
    kvecs : ndarray (nkline, 3)
        k-points along the line
    kline : ndarray (nkline, )
        cumulative k-distance
    """

    nhspts = hspts.shape[0] # 고대칭점 개수
    nkline = np.sum(n_p_to_p_grid) + 1 # 전체 k-point 개수 = 고대칭점 개수*각 고대칭점 당 존재하는 k값

    kvecs = np.zeros((nkline, 3)) # 3열이 되는 이유? -> 3차원이므로, 모든 k vector로 구성된 행렬
    kline = np.zeros(nkline) # 전체 k 개수 리스트

    i3 = 0
    kvecs[i3] = hspts[0]
    kline[i3] = 0.0

    for i1 in range(nhspts - 1):
        temp_v3 = (hspts[i1 + 1] - hspts[i1]) / n_p_to_p_grid[i1] # 구간별 선형 보간: 고대칭점 두 개를 잇는 직선 N등분 = 등간격!
        step_len = np.linalg.norm(temp_v3)

        for i2 in range(n_p_to_p_grid[i1]):
            i3 += 1
            kvecs[i3] = kvecs[i3 - 1] + temp_v3 # 실제 계산에 쓰일 k 벡터들
            kline[i3] = kline[i3 - 1] + step_len # 누적된 k-path 길이, x축

    return kvecs, kline

def make_k_grid(
    b_1, b_2,
    lineplotmode=True,
    nppx=64,
    nppy=64
):
    """
    Full Python translation of the Fortran k-grid generation logic.
    """

    if lineplotmode:
        # -------------------------
        # line plot mode
        # -------------------------
        nhspts = 2
        hspts = np.zeros((nhspts, 3)) # 3은 3차원 hspts[i] = (k_x, k_y, k_z)
        n_p_to_p_grid = np.zeros(nhspts - 1, dtype=int)

        hspts[0] = (
            0.66666666 * b_1
            + 0.33333333 * b_2
            - np.array([0.3, 0.0, 0.0])
        )
        hspts[1] = (
            0.66666666 * b_1
            + 0.33333333 * b_2
            + np.array([0.3, 0.0, 0.0])
        )

        origin_vec = 0.5 * (hspts[0] + hspts[1])
        print("(info) origin_vec:", origin_vec)

        n_p_to_p_grid[:] = 48

        k_vecs, kline = genkline(hspts, n_p_to_p_grid)

    else:
        # -------------------------
        # 2D grid mode
        # -------------------------
        origin_vec = 0.66666666 * b_1 + 0.33333333 * b_2
        width  = 0.08
        height = 0.08

        pxg = np.linspace(
            origin_vec[0] - width / 2,
            origin_vec[0] + width / 2,
            nppx
        )
        pyg = np.linspace(
            origin_vec[1] - height / 2,
            origin_vec[1] + height / 2,
            nppy
        )

        k_vecs = np.zeros((nppx * nppy, 3))
        idx = 0
        for ipy in range(nppy):
            for ipx in range(nppx):
                k_vecs[idx] = [pxg[ipx], pyg[ipy], 0.0]
                idx += 1

        kline = None

    return k_vecs, kline, origin_vec

# ============================================================
# example: graphene band at K
# ============================================================

import matplotlib.pyplot as plt

if __name__ == "__main__":

    # ---------------------------------------
    # k-path 생성 (Fortran lineplotmode = .true.)
    # ---------------------------------------
    k_vecs, kline, origin_vec = make_k_grid(
        b_1=b_1,
        b_2=b_2,
        lineplotmode=True
    )

    nk = k_vecs.shape[0]

    # ---------------------------------------
    # band 계산
    # ---------------------------------------
    # graphene monolayer → 2 bands
    nbands = 2
    bands = np.zeros((nk, nbands))

    for ik in range(nk):
        Hk = genhk(k_vecs[ik].astype(np.complex128))
        eigvals = np.linalg.eigvalsh(Hk)
        bands[ik, :] = eigvals




    # ---------------------------------------
    # band plot
    # ---------------------------------------
    plt.figure(figsize=(6, 4))

    for ib in range(nbands):
        plt.plot(kline, bands[:, ib], color="black", linewidth=1.5)

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)

    plt.xlabel(r"$k$ (a.u.)")
    plt.ylabel(r"Energy (Hartree)")
    plt.title("Graphene band (K-point line cut)")
    plt.tight_layout()
    plt.show()