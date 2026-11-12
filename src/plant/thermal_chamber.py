import numpy as np
from config.params import ThermalParams


class ThermalChamber:


    def __init__(self, params: ThermalParams):

        self.params = params
        self.T = float(params.T_init)
        self.t = 0.0

    def reset(self, T_init: float | None = None) -> float:

        self.T = float(T_init if T_init is not None else self.params.T_init)
        self.t = 0.0
        return self.T

    def derivative(self, T: float, u: float, T_amb: float | None = None) -> float:

        amb = self.params.T_amb if T_amb is None else T_amb

        u_clamped = float(np.clip(u, 0.0, 1.0))

        P_in = self.params.P_max * u_clamped
        Q_loss = (T - amb) / self.params.R_th
        dT_dt = (P_in - Q_loss) / self.params.C_th
        return dT_dt

    def step_euler(self, u: float, T_amb: float | None = None) -> float:

        dT = self.derivative(self.T, u, T_amb)
        self.T = self.T + self.params.dt * dT
        self.t += self.params.dt
        return self.T

    def step_rk4(self, u: float, T_amb: float | None = None) -> float:

        dt = self.params.dt
        k1 = self.derivative(self.T, u, T_amb)
        k2 = self.derivative(self.T + 0.5 * dt * k1, u, T_amb)
        k3 = self.derivative(self.T + 0.5 * dt * k2, u, T_amb)
        k4 = self.derivative(self.T + dt * k3, u, T_amb)

        self.T = self.T + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.t += dt
        return self.T

    def analytical_step_response(
        self, t_array: np.ndarray, u_step: float, T_0: float | None = None, T_amb: float | None = None
    ) -> np.ndarray:

        T_start = self.params.T_init if T_0 is None else T_0
        amb = self.params.T_amb if T_amb is None else T_amb
        u_val = float(np.clip(u_step, 0.0, 1.0))

        T_ss = amb + self.params.K_static * self.params.P_max * u_val
        tau = self.params.tau

        T_exact = T_ss + (T_start - T_ss) * np.exp(-t_array / tau)
        return T_exact

    def analytical_piecewise_response(
        self, t_array: np.ndarray, u_array: np.ndarray, T_0: float | None = None, T_amb: float | None = None
    ) -> np.ndarray:

        T_exact = np.zeros_like(t_array, dtype=float)
        T_curr = float(self.params.T_init if T_0 is None else T_0)
        amb = self.params.T_amb if T_amb is None else T_amb
        tau = self.params.tau

        T_exact[0] = T_curr
        for k in range(len(t_array) - 1):
            dt_step = t_array[k + 1] - t_array[k]
            u_val = float(np.clip(u_array[k], 0.0, 1.0))
            T_ss_k = amb + self.params.K_static * self.params.P_max * u_val
            T_curr = T_ss_k + (T_curr - T_ss_k) * np.exp(-dt_step / tau)
            T_exact[k + 1] = T_curr

        return T_exact
