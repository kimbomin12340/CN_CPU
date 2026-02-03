import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.fft import fft, fftfreq
from numba import njit, prange
import numba as nb
import time
nb.set_num_threads(12)
print("Num threads:", nb.get_num_threads())

# 1. 단위 변수 설정
astr = 1.8897259886          # bohr <- angstrom
eV   = 0.036749323858378     # eV -> Hartree
pi   = np.pi
fs   = 41.341374575751
sol  = 137.0
kB_au = 3.1668114e-6

kb = 8.6173e-5 * eV
delta = 1e-6
ci = 1j
c0 = 0.0 + 0.0j
c1 = 1.0 + 0.0j
hber = 1
e_fermi = 0*eV
w_func = 0*eV
c = 137
w0 = 0.26 * eV

A0 = 0.000126/w0
Temper = 300
# 2. graphite의 기본 구조 설정
nrmax = 3 # periodic boundary: 고체물리에서는 주로 3으로 초기 설정

a = 2.46 * astr # hexagonal structure lattice const

# primitive lattice vector in real space
a_1 = a * np.array([1.0, 0.0, 0.0])
a_2 = a * np.array([0.5, np.sqrt(3.0)/2.0, 0.0])
# primitive lattice vector in k-space
b_1 = (2*pi/a) * np.array([1.0, -1.0/np.sqrt(3.0), 0.0])
b_2 = (2*pi/a) * np.array([0.0, 2.0/np.sqrt(3.0), 0.0])

# sublattice vector = basis in unit cell
tau_1_vec = (2.0 * a_2 - a_1) / 3.0

# position of atoms in u.c
atom_pos = np.array([tau_1_vec, np.zeros(3)])
norb = atom_pos.shape[0]
cutoff = 1.1 * np.sqrt(tau_1_vec[0]**2 + tau_1_vec[1]**2 + tau_1_vec[2]**2)

# 3. 에너지 및 길이 scale parameters
v_pppi_0  = -2.7 * eV # 그래핀 평면 내 hopping
v_ppsgm_0 =  0.48 * eV # 층간 sgm 결합 = hopping

dlt_0 = 0.184 * a # hopping의 decay length
a_0   = a / np.sqrt(3.0) # 그래핀 C-C 결합 길이 (평면 내 pi hopping의 기준거리)
d_0   = 3.35 * astr # graphite 층간거리 (sgm hopping의 기준거리)


# 4. unit cell area
uc_area = np.linalg.norm(np.cross(a_1, a_2))


# 5. hpp 함수 정의: hopping 즉, overlap 부분에 존재하는 에너지
@njit(fastmath=True)
def hpp(d_vec): # d_vec = hopping하는 원자 사이의 거리 (오비탈은 고려하지 않음. just P_z)
    d = np.sqrt(d_vec[0]**2 + d_vec[1]**2 + d_vec[2]**2)
    dz = d_vec[2] # z 성분

    if d < delta: # hopping 최소 조건 만족 X
        return 0.0

    term_pi = -v_pppi_0 * np.exp(-(d - a_0) / dlt_0) * (1.0 - (dz / d)**2) # ?? 이 방향 의존성 term이 잘 와닫지 않느다.
    term_sgm = -v_ppsgm_0 * np.exp(-(d - d_0) / dlt_0) * (dz / d)**2

    return term_pi + term_sgm


# 6. hamiltonian in k-space
@njit(fastmath=True)
def genhk(k_vec):

    H = np.zeros((norb, norb), dtype=np.complex128) # hamiltonian 형태 형성: 오비탈 수 * 오비탈 수, basis = k_vec


    for i1 in range(norb):
        for i2 in range(norb):
            if i1 != i2: # off-diagonal part
                for ir1 in range(-nrmax, nrmax + 1): # periodic boundary 부분에서 (nrmax = 3 설정)
                    for ir2 in range(-nrmax, nrmax + 1):

                        Ux = ir1*a_1[0] + ir2*a_2[0]
                        Uy = ir1*a_1[1] + ir2*a_2[1]
                        Uz = ir1*a_1[2] + ir2*a_2[2]

                        dx = Ux + atom_pos[i2,0]-atom_pos[i1,0]
                        dy = Uy + atom_pos[i2,1]-atom_pos[i1,1]
                        dz = Uz + atom_pos[i2,2]-atom_pos[i1,2]

                        d = np.sqrt(dx*dx+dy*dy+dz*dz)
                        if d > cutoff:
                            continue

                        inexp = k_vec[0]*dx + k_vec[1]*dy + k_vec[2]*dz # k_vec * d_vec 내적 -> value
                        phase = np.exp(ci*np.real(inexp)) * np.exp(-np.imag(inexp)) # inexp에 허수 i 곱한 지수 term = phase % 진동 term이랑 감쇄 term이 존재
                        d_vec = np.array([dx, dy, dz])

                        H[i1, i2] -= hpp(d_vec) * phase # hopping은 -로 들어감!



    for i in range(norb): # diagonal part
        H[i, i] -= e_fermi + w_func

    return H

@njit(fastmath=True)
def gendhk(k_vec):

    dH = np.zeros((3, norb, norb), dtype=np.complex128)


    for i1 in range(norb):
        for i2 in range(norb):

            if i1 != i2:
                for ir1 in range(-nrmax, nrmax + 1):
                    for ir2 in range(-nrmax, nrmax + 1):
                        Ux = ir1*a_1[0] + ir2*a_2[0]
                        Uy = ir1*a_1[1] + ir2*a_2[1]
                        Uz = ir1*a_1[2] + ir2*a_2[2]

                        dx = Ux + atom_pos[i2,0]-atom_pos[i1,0]
                        dy = Uy + atom_pos[i2,1]-atom_pos[i1,1]
                        dz = Uz + atom_pos[i2,2]-atom_pos[i1,2]

                        d = np.sqrt(dx * dx + dy * dy + dz * dz)

                        if d > cutoff:
                            continue

                        inexp = k_vec[0] * dx + k_vec[1] * dy + k_vec[2] * dz  # k_vec * d_vec 내적 -> value
                        phase = np.exp(ci * np.real(inexp)) * np.exp(-np.imag(inexp))

                        d_vec = np.array([dx, dy, dz])

                        dH[:, i1, i2] -= ci * d_vec * hpp(d_vec) * phase

    # diagonal part: k 의존 X => zero vector
    return dH


    return k_vecs, kline, origin_vec

def bz_grid(b_1, b_2, Nk):
    k_list = []

    for ikx in range(-Nk//2, Nk//2):
        for iky in range(-Nk//2, Nk//2):
            k = (ikx/Nk) * b_1 + (iky/Nk) * b_2
            k_list.append(k)

    return np.array(k_list) # (Nk * Nk, 3) -> 수학 구조로 변경

def fourier(current, total_term, polarization):
    N = current.shape[0]
    T_tot = total_term[-1] - total_term[0]

    omega = 2 * np.pi * np.arange(N) / T_tot

    integrand = np.dot(current, polarization)

    p_fft_current = np.zeros(N, dtype=np.complex128)

    for i, w in enumerate(omega):
        integral = 0.0 + 0.0j

        for j in range(N-1):
            integral += 0.5 * (integrand[j] * np.exp(1j * w * total_term[j]) + integrand[j + 1] * np.exp(1j * w * total_term[j + 1])) * t_int

        p_fft_current[i] = integral


    return omega, p_fft_current

@njit(parallel=True)
def time_evolution(k_list, total_term, t_int, current):

    nk = k_list.shape[0]
    nt = total_term.shape[0]
    T_tot = total_term[-1] - total_term[0]
    current[:, :] = 0.0

    I = np.eye(2, dtype=np.complex128)

    current_tmp = np.zeros((nk, nt, 3), dtype=np.complex128)

    for ik in prange(nk):

        k = k_list[ik]

        H0 = genhk(k)
        eigvals, eigvecs = np.linalg.eigh(H0)

        vec_up = eigvecs[:,1].copy()
        vec_down = eigvecs[:,0].copy()

        x1 = (eigvals[1] - e_fermi) / (kB_au * Temper)
        x2 = (eigvals[0] - e_fermi) / (kB_au * Temper)



        f1 = 1.0 / (1.0 + np.exp(x1))
        f2 = 1.0 / (1.0 + np.exp(x2))

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

            if it>0:
                U_teo = np.linalg.solve(I - ci / hber * t_int / 2 * Hk, I + ci / hber * t_int / 2 * Hk)
                vec_up = U_teo @ vec_up
                vec_down = U_teo @ vec_down
            else:
                pass

            v1x = np.conj(vec_up) @ (dHk[0] @ vec_up)
            v1y = np.conj(vec_up) @ (dHk[1] @ vec_up)
            v1z = np.conj(vec_up) @ (dHk[2] @ vec_up)

            v2x = np.conj(vec_down) @ (dHk[0] @ vec_down)
            v2y = np.conj(vec_down) @ (dHk[1] @ vec_down)
            v2z = np.conj(vec_down) @ (dHk[2] @ vec_down)


            current_tmp[ik, it, 0] += v1x * f1 + v2x * f2
            current_tmp[ik, it, 1] += v1y * f1 + v2y * f2
            current_tmp[ik, it, 2] += v1z * f1 + v2z * f2

    for ik in range(nk):
        for it in range(nt):
            current[it, 0] += current_tmp[ik, it, 0]
            current[it, 1] += current_tmp[ik, it, 1]
            current[it, 2] += current_tmp[ik, it, 2]


def tanh_window(t, T_tot, smoothness):
    t_rel = t - T_tot / 2
    T_active = T_tot * 0.9  # 90% 영역까지는 1에 가깝게 유지
    return 0.5 * (np.tanh((t_rel + T_active/2) / smoothness) -
                  np.tanh((t_rel - T_active/2) / smoothness))




# 9. graphene band at K -> 그래핀이라 생략한 경우가 꽤 있음.
import matplotlib.pyplot as plt

if __name__ == "__main__":


    nk = 24
    Tot_time = 2*pi/w0 * 20
    nstep = 3000
    sig = Tot_time/12

    total_term = np.linspace(0, Tot_time, nstep)
    t_int = total_term[1] - total_term[0]

    ########################################
    k_list = bz_grid(b_1=b_1, b_2=b_2, Nk=nk)
    print("CPU K[0]:", k_list[0])
    current = np.zeros((len(total_term), 3), dtype=np.complex128)

    start = time.perf_counter()
    time_evolution(k_list, total_term , t_int, current) # -> current update
    plt.plot(total_term, current[:, 0].real)
    plt.show()
    # 시간에 따른 current 채우기 완료 -> fft 진행
    polarization = np.array([1, 0, 0]) # 보고 싶은 편광 방향

    win_func = tanh_window(total_term, Tot_time, smoothness=100)
    current_windowed = current * win_func[:, np.newaxis]

    omega, p_fft_current = fourier(current_windowed, total_term- Tot_time / 2, polarization)
    I_current = (omega ** 2) * np.abs(p_fft_current) ** 2
    angle_w = omega / w0
    end = time.perf_counter()
    print(f"걸린 시간: {end - start:.3f} 초")

    mask = (angle_w > 0) & (angle_w <= 50)  # 양수 주파수만 선택

    masks = [(angle_w > 0) & (angle_w <= 50),
             (angle_w > 0) & (angle_w <= 18),
             (angle_w > 0) & (angle_w <= 10)]

    for mask in masks:
        plt.figure(figsize=(6, 4))
        plt.semilogy(angle_w[mask], I_current[mask])
        plt.xlabel("Harmonic Order")
        plt.ylabel("Intensity")
        plt.grid(True, which='both', linestyle='--', alpha=0.3)

        # 홀수 harmonic 표시
        max_order = int(angle_w[mask].max())
        for n in range(1, max_order + 1, 2):
            plt.axvline(x=n, color='gray', linestyle='--', alpha=0.5)

        plt.show()