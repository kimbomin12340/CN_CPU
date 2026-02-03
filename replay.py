import numpy as np
from scipy.fft import fft, fftfreq
import time

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

A0 = 0.00126
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


# 3. 에너지 및 길이 scale parameters
v_pppi_0  = -2.7 * eV # 그래핀 평면 내 hopping
v_ppsgm_0 =  0.48 * eV # 층간 sgm 결합 = hopping

dlt_0 = 0.184 * a # hopping의 decay length
a_0   = a / np.sqrt(3.0) # 그래핀 C-C 결합 길이 (평면 내 pi hopping의 기준거리)
d_0   = 3.35 * astr # graphite 층간거리 (sgm hopping의 기준거리)


# 4. unit cell area
uc_area = np.linalg.norm(np.cross(a_1, a_2))


# 5. hpp 함수 정의: hopping 즉, overlap 부분에 존재하는 에너지
def hpp(d_vec): # d_vec = hopping하는 원자 사이의 거리 (오비탈은 고려하지 않음. just P_z)
    d = np.linalg.norm(d_vec) # size
    dz = d_vec[2] # z 성분

    if d < delta: # hopping 최소 조건 만족 X
        return 0.0
    # ??  hpp 함수 사용 전에 filtering 작업 한 번  하는데 또 해주는 이유? 좀 더 미세하게 걸러내기 위함인가? - 모델링에서 설정한 것

    term_pi = -v_pppi_0 * np.exp(-(d - a_0) / dlt_0) * (1.0 - (dz / d)**2) # ?? 이 방향 의존성 term이 잘 와닫지 않느다.
    term_sgm = -v_ppsgm_0 * np.exp(-(d - d_0) / dlt_0) * (dz / d)**2

    return term_pi + term_sgm
# ?? 여기에 사용한 수식을 좀 알아야 할 듯 이게 H_ij(R) term임 => 논문에서 가저온 것이므로 받아드리기


# 6. hamiltonian in k-space
def genhk(k_vec):
    norb = atom_pos.shape[0] # number of orbital in u.c => 2
    H = np.zeros((norb, norb), dtype=np.complex128) # hamiltonian 형태 형성: 오비탈 수 * 오비탈 수, basis = k_vec

    cutoff = 1.1 * np.linalg.norm(tau_1_vec) # size of pos_vec of atom 1 * 1.1 = 0.635*a % a가 안됨.

    for i1 in range(norb):
        for i2 in range(norb):
            if i1 != i2: # off-diagonal part
                for ir1 in range(-nrmax, nrmax + 1): # periodic boundary 부분에서 (nrmax = 3 설정)
                    for ir2 in range(-nrmax, nrmax + 1):

                        Uc_vec = ir1 * a_1 + ir2 * a_2 # unit cell vec: R
                        d_vec = Uc_vec + atom_pos[i2] - atom_pos[i1] # 서로 다른 두 atom 위치의 차이 = 상대거리
                        # ?? 앞에 조건문에 의해 periodic boundary 내 다른 u.c의 같은 위치에 있는 서로 다른 두 atom 간의 hopping은 고려 못하는거 아닌가?
                        # 어차피 이런 경우는 cutoff보다 거리가 커서 고려안해도 되는건가? -> 그런 것 같음 yes!
                        # 아 이게 밑에 i for문에서 고려되는 듯

                        if np.linalg.norm(d_vec) > cutoff:
                            continue # cutoff보다 상대거리가 길면 hopping 고려 X (진짜 타이트하다) -> 다음 반복문으로 넘어감.

                        inexp = k_vec[0]*d_vec[0] + k_vec[1]*d_vec[1] + k_vec[2]*d_vec[2] # k_vec * d_vec 내적 -> value
                        phase = np.exp(ci*np.real(inexp)) * np.exp(-np.imag(inexp)) # inexp에 허수 i 곱한 지수 term = phase % 진동 term이랑 감쇄 term이 존재

                        H[i1, i2] -= hpp(d_vec) * phase # hopping은 -로 들어감!



    for i in range(norb): # diagonal part
        H[i, i] -= e_fermi + w_func

    # ?? k 공간 헤밀토니안의 경우 같은 k 내에서 block diagonal화 된다고 했는데 지금 이 함수는 특정 k point에서의 헤밀토니안을 구하는 함수이다.
    # 이 헤밀토니안 내에서 대각부분과 아닌 부분이 또 나뉘어서 계산되는데 이게 가능한 이유가 저 cutoff 조건에 의해 대각 행렬에서는 phase term이 1 이 되고 (R이 무조건 0?) 그 말의 뜻은 on-site니깐 hpp 안쓰는 것?
    # k space basis를 보면 real space atomic orbital wave func의 sum으로 나타나 있기 때문에 on-site 뿐만 아니라 다른 u.c 내 같은 위치의 원자랑도 고려가되어야 한다.
    # 그러나 그래핀에서는 이 경우는 무조건 nearest neighborhood가 아니므로 hopping이 없다고 보고 위와 같이 on-site에 대한 에너지만 적었다 (다른 물질이라면 더 복잡;;)
    return H

def gendhk(k_vec):
    norb = atom_pos.shape[0]
    dH = np.zeros((3, norb, norb), dtype=np.complex128)
    cutoff = 1.1 * np.linalg.norm(tau_1_vec)

    for i1 in range(norb):
        for i2 in range(norb):

            if i1 != i2:
                for ir1 in range(-nrmax, nrmax + 1):
                    for ir2 in range(-nrmax, nrmax + 1):
                        Uc_vec = ir1 * a_1 + ir2 * a_2  # unit cell vec: R
                        d_vec = Uc_vec + atom_pos[i2] - atom_pos[i1]

                        if np.linalg.norm(d_vec) > cutoff:
                            continue

                        inexp = k_vec[0] * d_vec[0] + k_vec[1] * d_vec[1] + k_vec[2] * d_vec[2]  # k_vec * d_vec 내적 -> value
                        phase = np.exp(ci * np.real(inexp)) * np.exp(-np.imag(inexp))  # inexp에 허수 i 곱한 지수 term = phase % 진동 term이랑 감쇄 term이 존재
                        dH[:, i1, i2] -= ci * d_vec * hpp(d_vec) * phase

    # diagonal part: k 의존 X => zero vector
    return dH




def bz_grid(b_1, b_2, Nk):
    k_list = []

    for ikx in range(-Nk//2, Nk//2):
        for iky in range(-Nk//2, Nk//2):
            k = (ikx/Nk) * b_1 + (iky/Nk) * b_2
            k_list.append(k)

    return np.array(k_list) # (Nk * Nk, 3) -> 수학 구조로 변경

def U(t_int, Hk):
    I = np.eye(Hk.shape[0], dtype=np.complex128)
    return np.linalg.inv(I - ci / hber * t_int / 2 * Hk) @ (I + ci / hber * t_int / 2 * Hk)

def A(tm):
    t_center = Tot_time / 2
    phase_env = pi * (tm - t_center) / Tot_time
    envelope = np.cos(phase_env) ** 4
    At = A0 * np.cos(w0 * (tm - t_center)) * envelope
    return At

def fermi(e_fermi, E):
    return 1.0 / (1.0 + np.exp((E - e_fermi) / (kB_au * Temper)))

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

def tanh_window(t, T_tot, smoothness):
    t_rel = t - T_tot / 2
    T_active = T_tot * 0.9  # 90% 영역까지는 1에 가깝게 유지
    return 0.5 * (np.tanh((t_rel + T_active/2) / smoothness) -
                  np.tanh((t_rel - T_active/2) / smoothness))

# high harmonic generation
# => 강한 레이저를 원자/기체에 쏘면, 원래 주파수의 몇 배에 해당하는 빛이 생성되는 현상

# 9. graphene band at K -> 그래핀이라 생략한 경우가 꽤 있음.
import matplotlib.pyplot as plt

if __name__ == "__main__":

    nk = 24
    Tot_time = 2*pi/w0 * 20
    nstep = 2000
    total_term = np.linspace(0, Tot_time, nstep)
    t_int = total_term[1] - total_term[0]

    ########################################
    k_list = bz_grid(b_1=b_1, b_2=b_2, Nk=nk)
    current = np.zeros((len(total_term), 3), dtype=np.complex128)

    # time evolution op를 진행하기 위한 vec 저장공간
    vecs_ti_up = np.zeros((len(k_list), 2), dtype=np.complex128)
    vecs_ti_down = np.zeros((len(k_list), 2), dtype=np.complex128)

    # time evolution op를 진행하기 위한 vec 저장공간
    vecs_ti_up_t = vecs_ti_up
    vecs_ti_down_t = vecs_ti_down
    # Af = np.zeros(len(total_term))
    start = time.perf_counter()
    for it_idx, it in enumerate(total_term): # 각 시간별로 진행
        for ik, k in enumerate(k_list):
            if it_idx == 0: # 초기 eigen
                Hk = genhk(k.astype(np.complex128)) # k에 대한 헤밀토니안
                eigvals, eigvecs = np.linalg.eigh(Hk) # t = 0에서 고유값, 고유함수 저장

            # 각 시간 it에서의 시간 연산자 걸어준 eigenvecs => thetimevec
            HkT = genhk(k - A(it + (t_int / 2))) # 시간 it에서의 헤밀토니안
            dHkT = gendhk(k - A(it + (t_int / 2)))  # 시간 it에서의 헤밀토니안 미분

            # up/down 따로 고려해서 생각
            if it_idx == 0: #ti에서는 시간 연산자 안쓰니깐 따로 고려
                vecs_ti_up_t[ik] = eigvecs[:, 1]
                vecs_ti_down_t[ik] = eigvecs[:, 0]

            else: # 아니라면 시간 연산자 사용. 근데 이제 이전꺼에 계속 진행되는 감성으로 가야함.
                vecs_ti_up_t[ik] = U(t_int, HkT) @ vecs_ti_up_t[ik]
                vecs_ti_down_t[ik] = U(t_int, HkT) @ vecs_ti_down_t[ik]

            # 각 시간 it에서 구한 헤밀토니안 기댓값 -> x,y,z 3성분으로 분류 가능
            for coor in range(3):
                current[it_idx][coor] += (vecs_ti_up_t[ik].conj().T @ (dHkT[coor] @ vecs_ti_up_t[ik])) * fermi(e_fermi, eigvals[1])
                current[it_idx][coor] += (vecs_ti_down_t[ik].conj().T @ (dHkT[coor] @ vecs_ti_down_t[ik])) * fermi(e_fermi, eigvals[0])
            # 각 시간 it에서 모든 k에 대해 각 성분별로 속도 * fermi 분포 더함 -> current list에 저장.
        print("진행 시간:", it)

    # 시간에 따른 current 채우기 완료 -> fft 진행
    polarization = np.array([1, 0, 0])  # 보고 싶은 편광 방향

    win_func = tanh_window(total_term, Tot_time, smoothness=100)
    current_windowed = current * win_func[:, np.newaxis]

    omega, p_fft_current = fourier(current_windowed, total_term - Tot_time / 2, polarization)
    I_current = np.abs(omega ** 2 * p_fft_current) ** 2
    angle_w = omega / w0
    end = time.perf_counter()
    print(f"걸린 시간: {end - start:.3f} 초")

    mask = (angle_w > 0) & (angle_w <= 50)
    plt.semilogy(angle_w[mask], I_current[mask])
    plt.xlabel("harmonic order")
    plt.ylabel("Intensity")
    plt.show()







