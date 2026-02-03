import torch
import numpy as np

# 기본 세팅
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# 상수 세팅 템플릿
dtype_real = torch.float32
dtype_cplx = torch.complex64


# 1. 단위 변수 설정
# 변환자 (a. u.)
astr = torch.tensor(1.8897259886, device=device, dtype=dtype_real)        # bohr <- angstrom
eV   = torch.tensor(0.036749323858378, device=device, dtype=dtype_real)     # eV -> Hartree
fs   = torch.tensor(41.341374575751, device=device, dtype=dtype_real)
sol  = torch.tensor(137.0, device=device, dtype=dtype_real)
kB_au = torch.tensor(3.1668114e-6, device=device, dtype=dtype_real)

pi   = torch.pi.to(device=device, dtype=dtype_real)
kb =8.6173e-5 * eV
delta = torch.tensor(1e-6, device=device, dtype=dtype_real)
ci = torch.tensor(1j, device=device, dtype=dtype_cplx)
hbar = torch.tensor(1.0, device=device, dtype=dtype_real)
e_fermi = 0.0 * eV
w_func = 0.0 * eV
c = torch.tensor(137.0, device=device, dtype=dtype_real)

w = 0.26 * eV # 벡터장 각주파수
A0 = torch.tensor(0.01, device=device, dtype=dtype_real) # 벡터장 진폭
T = torch.tensor(300.0, device=device, dtype=dtype_real) # 페르미 디락 분포 속 온도[k]


# 2. graphite의 기본 구조 설정
nrmax = 3 # periodic boundary: 고체물리에서는 주로 3으로 초기 설정

a = 2.46 * astr # hexagonal structure lattice const

# primitive lattice vector in real space
sqrt3 = torch.sqrt(torch.tensor(3.0, device=device, dtype=dtype_real))
a_1 = a * torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype_real)
a_2 = a * torch.tensor([0.5, sqrt3/2.0, 0.0], device=device, dtype=dtype_real)
# primitive lattice vector in k-space
b_1 = (2.0 * pi / a) * torch.tensor([1.0, -1.0/sqrt3, 0.0], device=device, dtype=dtype_real)
b_2 = (2.0 * pi / a) * torch.tensor([0.0, 2.0/sqrt3, 0.0], device=device, dtype=dtype_real)

# sublattice vector = basis in unit cell
tau_1_vec = (2.0 * a_2 - a_1) / 3.0

# position of atoms in u.c
atom_pos = np.array([tau_1_vec, np.zeros(3)])


# 3. 에너지 및 길이 scale parameters
v_pppi_0  = -2.7 * eV # 그래핀 평면 내 hopping
v_ppsgm_0 =  0.48 * eV # 층간 sgm 결합 = hopping

dlt_0 = 0.184 * a # hopping의 decay length
a_0   = a / sqrt3 # 그래핀 C-C 결합 길이 (평면 내 pi hopping의 기준거리)
d_0   = 3.35 * astr # graphite 층간거리 (sgm hopping의 기준거리)


# 4. unit cell area (외적의 z성분)
uc_area = torch.abs(a_1[0]*a_2[1] - a_1[1]*a_2[0])


# 5. hpp 함수 정의: hopping 즉, overlap 부분에 존재하는 에너지
def hpp(d_vec): # d_vec = hopping하는 원자 사이의 거리 (오비탈은 고려하지 않음. just P_z)
    d = torch.linalg.norm(d_vec, dim=-1) # size
    dz = d_vec[..., 2] # z 성분

    eps = 1e-12
    d_safe = d +eps
    term_pi = -v_pppi_0 * torch.exp(-(d - a_0) / dlt_0) * (1.0 - (dz / d_safe)**2) # ?? 이 방향 의존성 term이 잘 와닫지 않느다.
    term_sgm = -v_ppsgm_0 * torch.exp(-(d - d_0) / dlt_0) * (dz / d_safe)**2

    h = term_pi + term_sgm

    return torch.where(d >= delta, h, torch.zeros_like(h)) # 조건을 위한 where함수


# 6. hamiltonian in k-space
def genhk(k_vec_batch):
    norb = atom_pos.shape[0] # number of orbital in u.c => 2
    H = torch.zeros((norb, norb), device=device, dtype=dtype_cplx) # hamiltonian 형태 형성: 오비탈 수 * 오비탈 수, basis = k_vec

    cutoff = 1.1 * torch.linalg.norm(tau_1_vec) # size of pos_vec of atom 1 * 1.1 = 0.635*a % a가 안됨.

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

                        if torch.linalg.norm(d_vec) > cutoff:
                            continue # cutoff보다 상대거리가 길면 hopping 고려 X (진짜 타이트하다) -> 다음 반복문으로 넘어감.

                        inexp = k_vec[0]*d_vec[0] + k_vec[1]*d_vec[1] + k_vec[2]*d_vec[2] # k_vec * d_vec 내적 -> value
                        phase = torch.exp(ci*torch.real(inexp)) * torch.exp(-torch.imag(inexp)) # inexp에 허수 i 곱한 지수 term = phase % 진동 term이랑 감쇄 term이 존재

                        H[i1, i2] -= hpp(d_vec) * phase # hopping은 -로 들어감!



    for i in range(norb): # diagonal part
        H[i, i] -= e_fermi + w_func

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


def bz_grid(b_1, b_2, Nk, device='cpu'):
    # 1. k-index 배열 생성
    kx = np.linspace(-0.5, 0.5, Nk, endpoint=False) # -Nk//2 ~ Nk//2 범위
    ky = np.linspace(-0.5, 0.5, Nk, endpoint=False) # 마지막 포인트 포함 X

    # 2. 2D meshgrid 생성
    KX, KY = np.meshgrid(kx, ky, indexing='ij') # 행렬 생성

    # 3. k-vector 계산
    k_vectors = KX[..., None] * b_1 + KY[..., None] * b_2  # (Nk, Nk, 3)

    # 4. (Nk*Nk, 3) 형태로 reshape
    k_vectors = k_vectors.reshape(-1, 3) # -1은 자동 차원 계산 (마지막의 3차원을 제외하고)

    # 5. torch tensor로 변환 (GPU 가능)
    k_vectors = torch.tensor(k_vectors, dtype=torch.float32, device=device)

    return k_vectors  # (Nk*Nk, 3)

def U(t_int, Hk):
    I = np.eye(Hk.shape[0], dtype=np.complex128)
    return np.linalg.inv(I - ci / hbar * t_int / 2 * Hk) @ (I + ci / hbar * t_int / 2 * Hk)

def A(ti, sig):
    return A0 * np.cos(w*ti) * np.exp(-np.abs(ti)/sig)

def fermi(e_fermi, E):
    return 1.0 / (1.0 + np.exp((E - e_fermi) / (kB_au * T)))

# high harmonic generation
# => 강한 레이저를 원자/기체에 쏘면, 원래 주파수의 몇 배에 해당하는 빛이 생성되는 현상

# 9. graphene band at K -> 그래핀이라 생략한 경우가 꽤 있음.
import matplotlib.pyplot as plt

if __name__ == "__main__":

    Nk = 36


    T = 2*pi/w * 15
    t_0 = T/2
    sig = T/12
    nstep = 5000
    total_term = np.linspace(0, 2 * t_0, nstep)
    t_int = total_term[1] - total_term[0]

    ########################################
    k_list = bz_grid(b_1=b_1, b_2=b_2, Nk=Nk)
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