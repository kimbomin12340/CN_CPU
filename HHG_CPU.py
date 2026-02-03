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
def hpp(d_vec): # d_vec (norb, norb, 2*nrmax+1, 2*nrmax+1, xyz)
    d = np.linalg.norm(d_vec, axis = -1) # 벡터 size
    dz = d_vec[..., 2] # z 성분

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
    Nk = k_vec.shape[0]
    H_total = np.zeros((Nk*norb, Nk*norb), dtype=np.complex64) # hamiltonian 형태 형성: 오비탈 수 * 오비탈 수, basis = k_vec

    cutoff = 1.1 * np.linalg.norm(tau_1_vec) # size of pos_vec of atom 1 * 1.1 = 0.635*a % a가 안됨.

    I1, I2, IR1, IR2 = np.meshgrid(
        np.arange(norb),
        np.arange(norb),
        np.arange(-nrmax, nrmax + 1),
        np.arange(-nrmax, nrmax + 1),
        indexing='ij'
    )


    Uc_vec = IR1[..., None] * a_1 + IR2[..., None] * a_2 # unit cell vec: R  # (norb, norb, 2*nrmax+1, 2*nrmax+1, 3) None: 차원 추가 1로
    d_vec = Uc_vec + atom_pos[I2] - atom_pos[I1] # 서로 다른 두 atom 위치의 차이 = 상대거리

    mask = (np.linalg.norm(d_vec, axis=-1) <= cutoff) & (I1 != I2) # i1 != i2 & cutoff

    # mask 조건에 따른 hpping 값
    hpp_vals = hpp(d_vec)  # shape: (norb, norb, 2*nrmax+1, 2*nrmax+1)
    hpp_vals[~mask] = 0

    # phase 계산: (Nk, norb, norb, 2*nrmax+1, 2*nrmax+1)
    phase = np.exp(1j * np.tensordot(k_vec, d_vec, axes=([1], [4])))  # (Nk, norb, norb, 2*nrmax+1, 2*nrmax+1)

    # 실제 hopping term
    hop_term = (hpp_vals[None, ...] * phase).sum(axis=(3,4))  # sum over IR1, IR2

    # hop_term: (Nk, norb, norb)
    Nk, norb = hop_term.shape[:2]

    # row, col index for each block
    block_offset = np.arange(Nk) * norb
    rows = block_offset[:, None] + np.arange(norb)  # shape (Nk, norb)
    cols = block_offset[:, None] + np.arange(norb)  # shape (Nk, norb)

    # broadcasting으로 모든 조합 만들기
    rows = rows[:, :, None]  # (Nk, norb, 1)
    cols = cols[:, None, :]  # (Nk, 1, norb)

    # H_total에 바로 대입
    H_total[rows, cols] = hop_term  # shape (Nk, norb, norb)

    return H_total

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
    # 1. k-index 배열 생성
    kx = np.linspace(-0.5, 0.5, Nk, endpoint=False) # -Nk//2 ~ Nk//2 범위
    ky = np.linspace(-0.5, 0.5, Nk, endpoint=False) # 마지막 포인트 포함 X

    # 2. 2D meshgrid 생성
    KX, KY = np.meshgrid(kx, ky, indexing='ij') # 행렬 생성

    # 3. k-vector 계산
    k_vectors = KX[..., None] * b_1 + KY[..., None] * b_2  # (Nk, Nk, 3)

    # 4. (Nk*Nk, 3) 형태로 reshape
    k_vectors = k_vectors.reshape(-1, 3) # -1은 자동 차원 계산 (마지막의 3차원을 제외하고)

    return k_vectors  # (Nk*Nk, 3)

def U(t_int, Hk):
    I = np.eye(Hk.shape[0], dtype=np.complex128)
    return np.linalg.inv(I - ci / hber * t_int / 2 * Hk) @ (I + ci / hber * t_int / 2 * Hk)

def A(ti, sig):
    return A0 * np.cos(w*ti) * np.exp(-np.abs(ti)/sig)

def fermi(e_fermi, E):
    return 1.0 / (1.0 + np.exp((E - e_fermi) / (kB_au * T)))

# high harmonic generation
# => 강한 레이저를 원자/기체에 쏘면, 원래 주파수의 몇 배에 해당하는 빛이 생성되는 현상

# 9. graphene band at K -> 그래핀이라 생략한 경우가 꽤 있음.
import matplotlib.pyplot as plt
import os
if __name__ == "__main__":

    nk = 36


    T = 2*pi/w * 15

    # print(2*pi/w/fs)
    # os.exit()

    t_0 = T/2
    sig = T/12
    nstep = 5000
    total_term = np.linspace(0, 2 * t_0, nstep)
    t_int = total_term[1] - total_term[0]
    print("omega_0", w/eV)
    print("omega:", 2 * pi / (t_int * 20) / eV)
    os.exit()
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
    for it_idx, it in enumerate(total_term): # 각 시간별로 진행
        for ik, k in enumerate(k_list):
            if it_idx == 0: # 초기 eigen
                Hk = genhk(k.astype(np.complex128)) # k에 대한 헤밀토니안
                eigvals, eigvecs = np.linalg.eigh(Hk) # t = 0에서 고유값, 고유함수 저장

            # 각 시간 it에서의 시간 연산자 걸어준 eigenvecs => thetimevec
            HkT = genhk((k - A(it + (t_int / 2), sig) / c).astype(np.complex128)) # 시간 it에서의 헤밀토니안
            dHkT = gendhk((k - A(it + (t_int / 2), sig) / c).astype(np.complex128))  # 시간 it에서의 헤밀토니안 미분

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
    polarization = np.array([1, 0, 0]) # 보고 싶은 편광 방향
    N = current.shape[0]
    # 끝 지점 window로 background 삭제
    window = np.blackman(N)
    current_w = current * window[:, None]

    fft_current = fft(current_w, axis = 0) # current fft 진행
    p_fft_current = np.dot(fft_current, polarization) / N # 특정 편광 방향만 보기 -> 내적 + 정규화

    # 주파수 축
    omega = fftfreq(N, t_int) * 2 * pi
    I_current = np.abs(omega**2 * p_fft_current)**2
    angle_w = omega / w

    mask = (angle_w > 0) & (angle_w <= 50)  # 양수 주파수만 선택
    plt.semilogy(angle_w[mask], I_current[mask])
    plt.xlabel("harmonic order")
    plt.ylabel("Intensity")
    plt.show()