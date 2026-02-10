
from functools import partial
import jax
import jax.numpy as np
from jax.scipy.linalg import block_diag

from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal, glorot_normal

from .ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal, make_DPLR_HiPPO

from .layers import EventPooling


def discretize_zoh(Lambda, step_delta, time_delta):
    """
    Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    This is the default discretization method used by many SSM works including S5.

    :param Lambda: diagonal state matrix (P,)
    :param step_delta: discretization step sizes (P,)
    :param time_delta: (float32) discretization step sizes (P,)
    :return: discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Delta = step_delta * time_delta
    Lambda_bar = np.exp(Lambda * Delta)
    eps = 1e-6
    gamma_bar = np.where(np.abs(Lambda) > eps, (Lambda_bar - Identity) / Lambda, Delta)
    return Lambda_bar, gamma_bar


def discretize_dirac(Lambda, step_delta, time_delta):
    """
    Discretize a diagonalized, continuous-time linear SSM
    with dirac delta input spikes.
    :param Lambda: diagonal state matrix (P,)
    :param step_delta: discretization step sizes (P,)
    :param time_delta: (float32) discretization step sizes (P,)
    :return: discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)
    """
    Delta = step_delta * time_delta
    Lambda_bar = np.exp(Lambda * Delta)
    gamma_bar = 1.0
    return Lambda_bar, gamma_bar


def discretize_async(Lambda, step_delta, time_delta):
    """
    Discretize a diagonalized, continuous-time linear SSM
    with dirac delta input spikes and appropriate input normalization.

    :param Lambda: diagonal state matrix (P,)
    :param step_delta: discretization step sizes (P,)
    :param time_delta: (float32) discretization step sizes (P,)
    :return: discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    dt_total = step_delta * time_delta
    Lambda_bar = np.exp(Lambda * dt_total)
    eps = 1e-6
    exp_step = np.exp(Lambda * step_delta)
    gamma_bar = np.where(np.abs(Lambda) > eps, (exp_step - Identity) / Lambda, step_delta)
    return Lambda_bar, gamma_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """
    Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.

    :param q_i: tuple containing A_i and Bu_i at position i (P,), (P,)
    :param q_j: tuple containing A_j and Bu_j at position j (P,), (P,)
    :return: new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


# def apply_ssm(Lambda_elements, Bu_elements, C_tilde, conj_sym, stride=1):
#     """
#     Compute the LxH output of discretized SSM given an LxH input.

#     :param Lambda_elements: (complex64) discretized state matrix (L, P)
#     :param Bu_elements: (complex64) discretized inputs projected to state space (L, P)
#     :param C_tilde: (complex64) output matrix (H, P)
#     :param conj_sym: (bool) whether conjugate symmetry is enforced
#     :return: ys: (float32) the SSM outputs (S5 layer preactivations) (L, H)
#     """
#     remaining_timesteps = (Bu_elements.shape[0] // stride) * stride

#     _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

#     xs = xs[:remaining_timesteps:stride]

#     if conj_sym:
#         return jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
#     else:
#         return jax.vmap(lambda x: (C_tilde @ x).real)(xs)

def apply_ssm(Lambda_elements, Bu_elements, C_tilde, conj_sym, stride=1, return_x=False):
    """
    Compute the SSM output using parallel scan.
    如果 return_x 为 True，则同时返回中间状态 xs。
    """
    remaining_timesteps = (Bu_elements.shape[0] // stride) * stride
    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
    xs = xs[:remaining_timesteps:stride]
    if conj_sym:
        ys = jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)
    else:
        ys = jax.vmap(lambda x: (C_tilde @ x).real)(xs)
    if return_x:
        return ys, xs
    else:
        return ys


# class S5SSM(nn.Module):
#     H_in: int
#     H_out: int
#     P: int
#     block_size: int
#     C_init: str
#     discretization: str
#     dt_min: float
#     dt_max: float
#     conj_sym: bool = True
#     clip_eigs: bool = False
#     step_rescale: float = 1.0
#     stride: int = 1
#     pooling_mode: str = "last"
#     manual_lambda: float = None  # 新增参数：若不为 None，则使用手动设置的常数 lambda
#     debug: bool = False        # 新增调试选项：True 时返回中间调试信息

#     """
#     Event-based S5 module
    
#     :param H_in: int, SSM input dimension
#     :param H_out: int, SSM output dimension
#     :param P: int, SSM state dimension
#     :param block_size: int, block size for block-diagonal state matrix
#     :param C_init: str, initialization method for output matrix C
#     :param discretization: str, discretization method for event-based SSM
#     :param dt_min: float, minimum value of log timestep
#     :param dt_max: float, maximum value of log timestep
#     :param conj_sym: bool, whether to enforce conjugate symmetry in the state space operator
#     :param clip_eigs: bool, whether to clip eigenvalues of the state space operator
#     :param step_rescale: float, rescale factor for step size
#     :param stride: int, stride for subsampling layer
#     :param pooling_mode: str, pooling mode for subsampling layer
#     """

#     # def setup(self):
#     #     """
#     #     Initializes parameters once and performs discretization each time the SSM is applied to a sequence
#     #     """

#     #     # Initialize state matrix A using approximation to HiPPO-LegS matrix
#     #     Lambda, _, B, V, B_orig = make_DPLR_HiPPO(self.block_size)

#     #     blocks = self.P // self.block_size
#     #     block_size = self.block_size // 2 if self.conj_sym else self.block_size
#     #     local_P = self.P // 2 if self.conj_sym else self.P

#     #     Lambda = Lambda[:block_size]
#     #     V = V[:, :block_size]
#     #     Vc = V.conj().T

#     #     # If initializing state matrix A as block-diagonal, put HiPPO approximation
#     #     # on each block
#     #     Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
#     #     V = block_diag(*([V] * blocks))
#     #     Vinv = block_diag(*([Vc] * blocks))

#     #     state_str = f"SSM: {self.H_in} -> {self.P} -> {self.H_out}"
#     #     if self.stride > 1:
#     #         state_str += f" (stride {self.stride} with pooling mode {self.pooling_mode})"
#     #     print(state_str)

#     #     # Initialize diagonal state to state matrix Lambda (eigenvalues)
#     #     self.Lambda_re = self.param("Lambda_re", lambda rng, shape: Lambda.real, (None,))
#     #     self.Lambda_im = self.param("Lambda_im", lambda rng, shape: Lambda.imag, (None,))

#     #     if self.clip_eigs:
#     #         self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
#     #     else:
#     #         self.Lambda = self.Lambda_re + 1j * self.Lambda_im

#     #     # Initialize input to state (B) matrix
#     #     B_init = lecun_normal()
#     #     B_shape = (self.P, self.H_in)
#     #     self.B = self.param("B",
#     #                         lambda rng, shape: init_VinvB(B_init, rng, shape, Vinv),
#     #                         B_shape)

#     #     # Initialize state to output (C) matrix
#     #     if self.C_init in ["trunc_standard_normal"]:
#     #         C_init = trunc_standard_normal
#     #         C_shape = (self.H_out, self.P, 2)
#     #     elif self.C_init in ["lecun_normal"]:
#     #         C_init = lecun_normal()
#     #         C_shape = (self.H_out, self.P, 2)
#     #     elif self.C_init in ["complex_normal"]:
#     #         C_init = normal(stddev=0.5 ** 0.5)
#     #     else:
#     #         raise NotImplementedError(
#     #                "C_init method {} not implemented".format(self.C_init))

#     #     if self.C_init in ["complex_normal"]:
#     #         C = self.param("C", C_init, (self.H_out, local_P, 2))
#     #         self.C_tilde = C[..., 0] + 1j * C[..., 1]

#     #     else:
#     #         self.C = self.param("C",
#     #                             lambda rng, shape: init_CV(C_init, rng, shape, V),
#     #                             C_shape)

#     #         self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

#     #     # Initialize feedthrough (D) matrix
#     #     if self.H_in == self.H_out:
#     #         self.D = self.param("D", normal(stddev=1.0), (self.H_in,))
#     #     else:
#     #         self.D = self.param("D", glorot_normal(), (self.H_out, self.H_in))

#     #     # Initialize learnable discretization timescale value
#     #     self.log_step = self.param("log_step",
#     #                                init_log_steps,
#     #                                (local_P, self.dt_min, self.dt_max))

#     #     # pooling layer
#     #     self.pool = EventPooling(stride=self.stride, mode=self.pooling_mode)

#     #     # Discretize
#     #     if self.discretization in ["zoh"]:
#     #         self.discretize_fn = discretize_zoh
#     #     elif self.discretization in ["dirac"]:
#     #         self.discretize_fn = discretize_dirac
#     #     elif self.discretization in ["async"]:
#     #         self.discretize_fn = discretize_async
#     #     else:
#     #         raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

#     # def setup(self):
#     #     # Compute DPLR representation and keep only real parts.
#     #     Lambda, _, B, V, _ = make_DPLR_HiPPO(self.block_size)
#     #     blocks = self.P // self.block_size
#     #     block_size = self.block_size  # 不再拆分为对称部分
#     #     # 直接截取前 block_size 个数
#     #     Lambda = Lambda[:block_size]
#     #     # 将 Lambda 扩展为实数向量
#     #     Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
        
#     #     state_str = f"SSM: {self.H_in} -> {self.P} -> {self.H_out}"
#     #     if self.stride > 1:
#     #         state_str += f" (stride {self.stride} with pooling mode {self.pooling_mode})"
#     #     print(state_str)

#     #     # 直接以实数方式存储 Lambda
#     #     self.Lambda = self.param("Lambda", lambda rng, shape: Lambda, (None,))
        
#     #     # Initialize B as a real-valued matrix.
#     #     B_init = nn.initializers.lecun_normal()
#     #     B_shape = (self.P, self.H_in)
#     #     self.B = self.param("B", lambda rng, shape: init_VinvB(B_init, rng, shape, None), B_shape)
        
#     #     # Initialize C as a real-valued matrix.
#     #     if self.C_init in ["trunc_standard_normal", "lecun_normal", "complex_normal"]:
#     #         C_init = nn.initializers.lecun_normal()
#     #         C_shape = (self.H_out, self.P)
#     #     else:
#     #         raise NotImplementedError("C_init method {} not implemented".format(self.C_init))
#     #     self.C = self.param("C", C_init, C_shape)
#     #     # Directly use C as real-valued C_tilde.
#     #     self.C_tilde = self.C

#     #     # Initialize D as before.
#     #     if self.H_in == self.H_out:
#     #         self.D = self.param("D", nn.initializers.normal(stddev=1.0), (self.H_in,))
#     #     else:
#     #         self.D = self.param("D", nn.initializers.glorot_normal(), (self.H_out, self.H_in))
            
#     #     # self.log_step = self.param("log_step", init_log_steps, (self.P, self.dt_min, self.dt_max))
#     #     self.log_step = self.param("log_step", init_log_steps, (self.P, self.dt_min, self.dt_max))
#     #     self.pool = EventPooling(stride=self.stride, mode=self.pooling_mode)

#     #     # 离散化函数保持不变
#     #     if self.discretization in ["zoh"]:
#     #         self.discretize_fn = discretize_zoh
#     #     elif self.discretization in ["dirac"]:
#     #         self.discretize_fn = discretize_dirac
#     #     elif self.discretization in ["async"]:
#     #         self.discretize_fn = discretize_async
#     #     else:
#     #         raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

#     def setup(self):
#         if self.manual_lambda is not None:
#             # 使用手动设定的常数构造 Lambda 向量（每个状态的值均相同）
#             Lambda = self.manual_lambda * np.ones((self.P,))
#             # print(f"[S5SSM] Using manual lambda: {self.manual_lambda}, resulting in Lambda with shape {Lambda.shape} and values: {Lambda}")
#             # B 等其他参数仍从 DPLR 初始化中获取（忽略其 Lambda 部分）
#             _, _, B, V, _ = make_DPLR_HiPPO(self.block_size)
#         else:
#             Lambda, _, B, V, _ = make_DPLR_HiPPO(self.block_size)
#             blocks = self.P // self.block_size
#             block_size = self.block_size
#             Lambda = Lambda[:block_size]
#             Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
#             # print(f"[S5SSM] Using auto computed lambda, shape: {Lambda.shape}, values: {Lambda}")
        
#         state_str = f"SSM: {self.H_in} -> {self.P} -> {self.H_out}"
#         if self.stride > 1:
#             state_str += f" (stride {self.stride} with pooling mode {self.pooling_mode})"
#         print(state_str)

#         # self.Lambda = self.param("Lambda", lambda rng, shape: Lambda, (None,))

#         # 如果希望 Lambda 固定不变，则将它存入非训练变量集合中
#         self.Lambda = Lambda

#         B_init = nn.initializers.lecun_normal()
#         B_shape = (self.P, self.H_in)
#         self.B = self.param("B", lambda rng, shape: init_VinvB(B_init, rng, shape, None), B_shape)
#         if self.C_init in ["trunc_standard_normal", "lecun_normal", "complex_normal"]:
#             C_init = nn.initializers.lecun_normal()
#             C_shape = (self.H_out, self.P)
#         else:
#             raise NotImplementedError("C_init method {} not implemented".format(self.C_init))
#         self.C = self.param("C", C_init, C_shape)
#         self.C_tilde = self.C

#         if self.H_in == self.H_out:
#             self.D = self.param("D", nn.initializers.normal(stddev=1.0), (self.H_in,))
#         else:
#             self.D = self.param("D", nn.initializers.glorot_normal(), (self.H_out, self.H_in))

#         self.log_step = self.param("log_step", init_log_steps, (self.P, self.dt_min, self.dt_max))
#         self.pool = EventPooling(stride=self.stride, mode=self.pooling_mode)

#         if self.discretization in ["zoh"]:
#             self.discretize_fn = discretize_zoh
#         elif self.discretization in ["dirac"]:
#             self.discretize_fn = discretize_dirac
#         elif self.discretization in ["async"]:
#             self.discretize_fn = discretize_async
#         else:
#             raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))


#     # def __call__(self, input_sequence, integration_timesteps):
#     #     """
#     #     Compute the LxH output of the S5 SSM given an LxH input sequence using a parallel scan.

#     #     :param input_sequence: (float32) input sequence (L, H)
#     #     :param integration_timesteps: (float32) integration timesteps (L,)
#     #     :return: (float32) output sequence (L, H)
#     #     """

#     #     # discretize on the fly
#     #     B = self.B[..., 0] + 1j * self.B[..., 1]

#     #     def discretize_and_project_inputs(u, _timestep):
#     #         step = self.step_rescale * np.exp(self.log_step[:, 0])
#     #         Lambda_bar, gamma_bar = self.discretize_fn(self.Lambda, step, _timestep)
#     #         Bu = gamma_bar * (B @ u)
#     #         return Lambda_bar, Bu

#     #     Lambda_bar_elements, Bu_bar_elements = jax.vmap(discretize_and_project_inputs)(input_sequence, integration_timesteps)

#     #     ys = apply_ssm(
#     #         Lambda_bar_elements,
#     #         Bu_bar_elements,
#     #         self.C_tilde,
#     #         self.conj_sym,
#     #         stride=self.stride
#     #     )

#     #     if self.stride > 1:
#     #         input_sequence, _ = self.pool(input_sequence, integration_timesteps)

#     #     if self.H_in == self.H_out:
#     #         Du = jax.vmap(lambda u: self.D * u)(input_sequence)
#     #     else:
#     #         Du = jax.vmap(lambda u: self.D @ u)(input_sequence)

#     #     return ys + Du

#     # def __call__(self, input_sequence, integration_timesteps):
#     #     # Directly use real-valued B (no complex conversion)
#     #     B = self.B

#     #     def discretize_and_project_inputs(u, _timestep):
#     #         # Compute step as before, note self.log_step is now real-valued.
#     #         step = self.step_rescale * np.exp(self.log_step)
#     #         Lambda_bar, gamma_bar = self.discretize_fn(self.Lambda, step, _timestep)
#     #         # Compute projection in real domain.
#     #         Bu = gamma_bar * (B @ u)
#     #         return Lambda_bar, Bu

#     #     Lambda_bar_elements, Bu_bar_elements = jax.vmap(discretize_and_project_inputs)(
#     #         input_sequence, integration_timesteps
#     #     )

#     #     ys = apply_ssm(
#     #         Lambda_bar_elements,
#     #         Bu_bar_elements,
#     #         self.C_tilde,
#     #         self.conj_sym,  # 此参数在实数版本中无影响
#     #         stride=self.stride
#     #     )

#     #     if self.stride > 1:
#     #         input_sequence, _ = self.pool(input_sequence, integration_timesteps)

#     #     if self.H_in == self.H_out:
#     #         Du = jax.vmap(lambda u: self.D * u)(input_sequence)
#     #     else:
#     #         Du = jax.vmap(lambda u: self.D @ u)(input_sequence)

#     #     return ys + Du

#     def __call__(self, input_sequence, integration_timesteps):
#         B = self.B

#         def discretize_and_project_inputs(u, _timestep):
#             step = self.step_rescale * np.exp(self.log_step)
#             Lambda_bar, gamma_bar = self.discretize_fn(self.Lambda, step, _timestep)
#             Bu = gamma_bar * (B @ u)
#             return Lambda_bar, Bu

#         Lambda_bar_elements, Bu_bar_elements = jax.vmap(discretize_and_project_inputs)(
#             input_sequence, integration_timesteps
#         )

#         ys = apply_ssm(Lambda_bar_elements, Bu_bar_elements, self.C_tilde, self.conj_sym, stride=self.stride)

#         if self.stride > 1:
#             input_sequence, _ = self.pool(input_sequence, integration_timesteps)

#         if self.H_in == self.H_out:
#             Du = jax.vmap(lambda u: self.D * u)(input_sequence)
#         else:
#             Du = jax.vmap(lambda u: self.D @ u)(input_sequence)

#         return ys + Du

class S5SSM(nn.Module):
    H_in: int
    H_out: int
    P: int
    block_size: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    step_rescale: float = 1.0
    stride: int = 1
    pooling_mode: str = "last"
    manual_lambda: float = None  # 若不为 None，则使用手动设置的常数 lambda
    debug: bool = False        # 新增调试选项：True 时返回中间调试信息

    def setup(self):
        if self.manual_lambda is not None:
            Lambda = self.manual_lambda * np.ones((self.P,))
            _, _, B, V, _ = make_DPLR_HiPPO(self.block_size)
        else:
            Lambda, _, B, V, _ = make_DPLR_HiPPO(self.block_size)
            blocks = self.P // self.block_size
            block_size = self.block_size
            Lambda = Lambda[:block_size]
            Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
        state_str = f"SSM: {self.H_in} -> {self.P} -> {self.H_out}"
        if self.stride > 1:
            state_str += f" (stride {self.stride} with pooling mode {self.pooling_mode})"
        print(state_str)
        # Make Lambda trainable. Initialize from computed (or manual) values.
        self.Lambda = self.param("Lambda", lambda rng, shape: Lambda, (self.P,))

        B_init = nn.initializers.lecun_normal()
        B_shape = (self.P, self.H_in)
        self.B = self.param("B", lambda rng, shape: init_VinvB(B_init, rng, shape, None), B_shape)
        if self.C_init in ["trunc_standard_normal", "lecun_normal", "complex_normal"]:
            C_init = nn.initializers.lecun_normal()
            C_shape = (self.H_out, self.P)
        else:
            raise NotImplementedError("C_init method {} not implemented".format(self.C_init))
        self.C = self.param("C", C_init, C_shape)
        self.C_tilde = self.C

        if self.H_in == self.H_out:
            self.D = self.param("D", nn.initializers.normal(stddev=1.0), (self.H_in,))
        else:
            self.D = self.param("D", nn.initializers.glorot_normal(), (self.H_out, self.H_in))

        self.log_step = self.param("log_step", init_log_steps, (self.P, self.dt_min, self.dt_max))
        self.pool = EventPooling(stride=self.stride, mode=self.pooling_mode)

        if self.discretization in ["zoh"]:
            self.discretize_fn = discretize_zoh
        elif self.discretization in ["dirac"]:
            self.discretize_fn = discretize_dirac
        elif self.discretization in ["async"]:
            self.discretize_fn = discretize_async
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

    def __call__(self, input_sequence, integration_timesteps):
        B = self.B

        def discretize_and_project_inputs(u, _timestep):
            step = self.step_rescale * np.exp(self.log_step)
            Lambda_bar, gamma_bar = self.discretize_fn(self.Lambda, step, _timestep)
            Bu = gamma_bar * (B @ u)
            return Lambda_bar, Bu

        Lambda_bar_elements, Bu_bar_elements = jax.vmap(discretize_and_project_inputs)(
            input_sequence, integration_timesteps
        )

        if self.debug:
            ys, xs = apply_ssm(Lambda_bar_elements, Bu_bar_elements, self.C_tilde, self.conj_sym,
                                 stride=self.stride, return_x=True)
        else:
            ys = apply_ssm(Lambda_bar_elements, Bu_bar_elements, self.C_tilde, self.conj_sym,
                           stride=self.stride)

        if self.stride > 1:
            pooled_input, _ = self.pool(input_sequence, integration_timesteps)
            effective_inputs = pooled_input
        else:
            effective_inputs = input_sequence

        if self.H_in == self.H_out:
            Du = jax.vmap(lambda u: self.D * u)(effective_inputs)
        else:
            Du = jax.vmap(lambda u: self.D @ u)(effective_inputs)

        output = ys + Du

        if self.debug:
            # 取 xs 的前 5 个元素，并与对应的输入一起输出
            # 注意：xs 的 shape 为 (有效时间步数, P)
            x_first5 = xs[:, :5]
            # 这里构造 debug 信息字典
            debug_info = {"inputs": effective_inputs, "x_first5": x_first5}
            print("Debug Info: 每个 time step 的输入和 x 向量前 5 个元素：")
            for i in range(effective_inputs.shape[0]):
                print(f"Time step {i}: input = {effective_inputs[i]}, x[:5] = {x_first5[i]}")
            return output, debug_info
        else:
            return output



def init_S5SSM(
        C_init,
        dt_min,
        dt_max,
        conj_sym,
        clip_eigs,
):
    """
    Convenience function that will be used to initialize the SSM.
    Same arguments as defined in S5SSM above.
    """
    return partial(S5SSM,
                   C_init=C_init,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs
                   )
