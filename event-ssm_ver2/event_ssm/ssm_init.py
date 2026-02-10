import jax
from jax import random
import jax.numpy as np
from jax.nn.initializers import lecun_normal
from jax.numpy.linalg import eigh


def make_HiPPO(N):
    """
    Create a HiPPO-LegS matrix.
    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py

    :params N: int32, state size
    :returns: N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py

    :params N: int32, state size
    :returns: N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


# def make_DPLR_HiPPO(N):
#     """
#     Makes components needed for DPLR representation of HiPPO-LegS
#     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
#     Note, we will only use the diagonal part

#     :params N: int32, state size
#     :returns:   eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
#                 eigenvectors V, HiPPO B pre-conjugation
#     """
#     A, P, B = make_NPLR_HiPPO(N)

#     S = A + P[:, np.newaxis] * P[np.newaxis, :]

#     S_diag = np.diagonal(S)
#     Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

#     # Diagonalize S to V \Lambda V^*
#     Lambda_imag, V = eigh(S * -1j)

#     P = V.conj().T @ P
#     B_orig = B
#     B = V.conj().T @ B
#     return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig

def make_DPLR_HiPPO(N):
    """
    Modified version: Compute a real-valued HiPPO-LegS matrix.
    Instead of computing complex eigenvalues via eigh(S * -1j),
    we directly compute eigen-decomposition on S and use the real eigenvalues.
    """
    A, P, B = make_NPLR_HiPPO(N)
    S = A + P[:, np.newaxis] * P[np.newaxis, :]
    S_diag = np.diagonal(S)

    #############################################################################
    # # Use the mean of the diagonal as a baseline (same as original)
    # Lambda = np.mean(S_diag) * np.ones_like(S_diag)
    # # Instead of eigh(S * -1j), we compute eigen-decomposition on S directly.
    # Lambda_real, V = eigh(S)
    # # Use the real eigenvalues only
    # Lambda = Lambda_real  # Now Lambda is purely real.
    # # 返回 B, V 等不再进行复数操作，直接返回实数矩阵
    #############################################################################
    # 调试：检查 S 矩阵中是否有 NaN 或无穷大值
    print(f"[*] Debug: Checking matrix S for NaNs: {np.any(np.isnan(S))}, Infs: {np.any(np.isinf(S))}")

    # 规避：强制将 S 矩阵放到 CPU 上，以绕过 GPU 的 cuSolver 库
    try:
        cpu_device = jax.devices('cpu')[0]
        S_on_cpu = jax.device_put(S, cpu_device)
        print("[*] Debug: Forcing eigh computation on CPU to bypass cuSolver.")
        Lambda, V = eigh(S_on_cpu)
    except Exception as e:
        print(f"[错误] 即使在 CPU 上执行 eigh 也失败了: {e}")
        raise e


    return Lambda, P, B, V, B  # 此处 B 直接返回，无需 B_orig 的复数处理


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """
    Initialize the learnable timescale Delta by sampling
    uniformly between dt_min and dt_max.

    :params dt_min: float32, minimum value of log timestep
    :params dt_max: float32, maximum value of log timestep
    :returns: init function
    """
    def init(key, shape):
        return random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


# def init_log_steps(key, input):
#     """
#     Initialize an array of learnable timescale parameters

#     :params key: jax random
#     :params input: tuple containing the array shape H and
#                       dt_min and dt_max
#     :returns: initialized array of timescales (float32): (H,)
#      """
#     H, dt_min, dt_max = input
#     log_steps = []
#     for i in range(H):
#         key, skey = random.split(key)
#         log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
#         log_steps.append(log_step)

#     return np.array(log_steps)
def init_log_steps(key, input):
    H, dt_min, dt_max = input
    log_steps = []
    for i in range(H):
        key, skey = random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step[0])  # 提取标量，确保返回一维向量
    return np.array(log_steps)  # 结果形状为 (H,)




# def init_VinvB(init_fun, rng, shape, Vinv):
#     """
#     Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
#     Note we will parameterize this with two different matrices for complex numbers.

#     :params init_fun: function, the initialization function to use, e.g. lecun_normal()
#     :params rng: jax random key to be used with init function.
#     :params shape: tuple, desired shape (P,H)
#     :params Vinv: complex64, the inverse eigenvectors used for initialization
#     :returns: B_tilde (complex64) of shape (P,H,2)
#     """
#     B = init_fun(rng, shape)
#     VinvB = Vinv @ B
#     VinvB_real = VinvB.real
#     VinvB_imag = VinvB.imag
#     return np.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)

def init_VinvB(init_fun, rng, shape, Vinv):
    """
    Modified version: Directly initialize B as a real-valued matrix.
    The original version computed Vinv @ B and then split into real and imaginary parts.
    Here, we ignore Vinv and simply return the initialized B.
    """
    B = init_fun(rng, shape)
    return B


# def trunc_standard_normal(key, shape):
#     """
#     Sample C with a truncated normal distribution with standard deviation 1.

#     :params key: jax random key
#     :params shape: tuple, desired shape (H,P, _)
#     :returns: sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
#      """
#     H, P, _ = shape
#     Cs = []
#     for i in range(H):
#         key, skey = random.split(key)
#         C = lecun_normal()(skey, shape=(1, P, 2))
#         Cs.append(C)
#     return np.array(Cs)[:, 0]

def trunc_standard_normal(key, shape):
    """
    Modified version: Return a real-valued matrix.
    Instead of returning a tensor with an extra dimension for the imaginary part,
    we directly return the output of lecun_normal.
    """
    return nn.initializers.lecun_normal()(key, shape=shape)


# def init_CV(init_fun, rng, shape, V):
#     """
#     Initialize C_tilde=CV. First sample C. Then compute CV.
#     Note we will parameterize this with two different matrices for complex numbers.

#     :params init_fun: function, the initialization function to use, e.g. lecun_normal()
#     :params rng: jax random key to be used with init function.
#     :params shape: tuple, desired shape (H,P)
#     :params V: complex64, the eigenvectors used for initialization
#     :returns: C_tilde (complex64) of shape (H,P,2)
#      """
#     C_ = init_fun(rng, shape)
#     C = C_[..., 0] + 1j * C_[..., 1]
#     CV = C @ V
#     CV_real = CV.real
#     CV_imag = CV.imag
#     return np.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)

def init_CV(init_fun, rng, shape, V):
    """
    Modified version: Directly return a real-valued matrix.
    We simply use the initialization function without converting to complex numbers.
    """
    return init_fun(rng, shape)
