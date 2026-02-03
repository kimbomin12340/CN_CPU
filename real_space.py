import numpy as np
from scipy.fft import fft, fftfreq

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
w = 0.26 * eV

A0 = 0.01
T = 300
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

                        H[i1, i2] -= hpp(d_vec) * phase # hopping은 -로 들어감

    for i in range(norb): # diagonal part
        H[i, i] -= e_fermi + w_func
    return H