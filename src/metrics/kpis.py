
import numpy as np


def compute_open_loop_metrics(T_sim: np.ndarray, T_analytical: np.ndarray) -> dict[str, float]:

    diff = T_sim - T_analytical
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    max_err = float(np.max(np.abs(diff)))
    return {
        "rmse_degC": rmse,
        "mae_degC": mae,
        "max_error_degC": max_err,
    }


def compute_closed_loop_kpis(
    t_arr: np.ndarray,
    T_arr: np.ndarray,
    u_arr: np.ndarray,
    T_ref: float,
    P_max: float,
    T_min_safe: float,
    T_max_safe: float,
    comfort_band: float = 0.5,
    shield_interventions: int = 0,
) -> dict[str, float]:

    dt = float(t_arr[1] - t_arr[0]) if len(t_arr) > 1 else 1.0

    # 1. KPI de Energía total consumida en kiloJoules
    energy_joules = float(np.sum(P_max * u_arr * dt))
    energy_kj = energy_joules / 1000.0
    energy_wh = energy_joules / 3600.0

    # 2. KPI de Conmutaciones
    du = np.abs(np.diff(u_arr))
    total_variation = float(np.sum(du))
    switching_count = int(np.sum(du > 0.05))

    # 3. KPI de Confort térmico
    steady_idx = len(T_arr) // 4
    rmse_comfort = float(np.sqrt(np.mean((T_arr[steady_idx:] - T_ref) ** 2)))
    in_comfort = np.abs(T_arr[steady_idx:] - T_ref) <= comfort_band
    comfort_pct = float(100.0 * np.mean(in_comfort))

    # 4. KPI de Seguridad
    violations = (T_arr < T_min_safe) | (T_arr > T_max_safe)
    violation_time_s = float(np.sum(violations) * dt)

    return {
        "energy_kJ": energy_kj,
        "energy_Wh": energy_wh,
        "total_variation_u": total_variation,
        "switching_count": switching_count,
        "rmse_comfort_degC": rmse_comfort,
        "comfort_band_pct": comfort_pct,
        "safety_violation_s": violation_time_s,
        "shield_interventions": int(shield_interventions),
    }
