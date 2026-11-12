import os
import sys
import csv
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.params import ThermalParams
from src.plant.thermal_chamber import ThermalChamber
from src.metrics.kpis import compute_open_loop_metrics


def run_step_experiment(params: ThermalParams, u_step: float = 0.5, duration_s: float = 720.0) -> dict:
    
    #Esta función ejecuta el ensayo de escalón constante en lazo abierto y lo compara con la solución analítica.
    #Aquí simulamos paso a paso con Euler y RK4 y medimos su diferencia frente a la exponencial teórica.
    
    n_steps = int(duration_s / params.dt) + 1
    t_arr = np.linspace(0.0, duration_s, n_steps)

    chamber_euler = ThermalChamber(params)
    chamber_rk4 = ThermalChamber(params)

    T_euler = np.zeros(n_steps)
    T_rk4 = np.zeros(n_steps)
    T_euler[0] = chamber_euler.reset()
    T_rk4[0] = chamber_rk4.reset()

    for k in range(n_steps - 1):
        T_euler[k + 1] = chamber_euler.step_euler(u_step)
        T_rk4[k + 1] = chamber_rk4.step_rk4(u_step)

    T_analytical = chamber_rk4.analytical_step_response(t_arr, u_step)

    metrics_euler = compute_open_loop_metrics(T_euler, T_analytical)
    metrics_rk4 = compute_open_loop_metrics(T_rk4, T_analytical)

    return {
        "t": t_arr,
        "u": np.full_like(t_arr, u_step),
        "T_euler": T_euler,
        "T_rk4": T_rk4,
        "T_analytical": T_analytical,
        "metrics_euler": metrics_euler,
        "metrics_rk4": metrics_rk4,
    }


def run_multistep_experiment(params: ThermalParams, duration_s: float = 1200.0) -> dict:
    
    #Esta función ejecuta un ensayo de escalones múltiples en lazo abierto (calentamiento y enfriamiento libre).
    #Aquí contrastamos la integración numérica frente a la solución analítica por tramos.
    
    n_steps = int(duration_s / params.dt) + 1
    t_arr = np.linspace(0.0, duration_s, n_steps)
    u_arr = np.zeros(n_steps)

    # Definimos tres escalones de potencia
    for k, t_val in enumerate(t_arr):
        if t_val < 400.0:
            u_arr[k] = 0.40
        elif t_val < 800.0:
            u_arr[k] = 0.80
        else:
            u_arr[k] = 0.00

    chamber_rk4 = ThermalChamber(params)
    T_rk4 = np.zeros(n_steps)
    T_rk4[0] = chamber_rk4.reset()

    for k in range(n_steps - 1):
        T_rk4[k + 1] = chamber_rk4.step_rk4(u_arr[k])

    T_analytical = chamber_rk4.analytical_piecewise_response(t_arr, u_arr)
    metrics_rk4 = compute_open_loop_metrics(T_rk4, T_analytical)

    return {
        "t": t_arr,
        "u": u_arr,
        "T_rk4": T_rk4,
        "T_analytical": T_analytical,
        "metrics_rk4": metrics_rk4,
    }


def save_plots_and_logs(params: ThermalParams, step_res: dict, multi_res: dict) -> tuple[str, str, str]:
    
    #Esta función genera la figura de evidencia del Primer Hito y guarda los registros CSV y JSON con fecha.
    #Aquí graficamos la superposición analítica vs. simulada y el error numérico residual.
    
    fig_dir = os.path.join(PROJECT_ROOT, "results", "figures")
    log_dir = os.path.join(PROJECT_ROOT, "results", "logs")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    timestamp_str = datetime.now().isoformat(timespec="seconds")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=160)
    fig.suptitle(
        f"Evidencia Hito 1 (S1): Cámara Térmica en Lazo Abierto vs. Respuesta Analítica\n"
        f"Fecha de ejecución: {timestamp_str} | tau = {params.tau:.1f} s, R_th = {params.R_th} °C/W, C_th = {params.C_th} J/°C",
        fontsize=11,
        fontweight="bold",
    )

    # Panel 1: Respuesta al escalón 50%
    ax1 = axes[0, 0]
    t = step_res["t"]
    T_ss = params.T_amb + params.K_static * params.P_max * 0.5
    T_tau = params.T_init + 0.63212 * (T_ss - params.T_init)

    ax1.plot(t, step_res["T_analytical"], "k-", lw=2.2, label="Respuesta Analítica Exacta")
    ax1.plot(t, step_res["T_rk4"], "r--", lw=1.8, label="Simulación Numérica (RK4)")
    ax1.plot(t[::15], step_res["T_euler"][::15], "b:", marker="o", ms=3.5, label="Simulación Numérica (Euler)")
    ax1.axhline(T_ss, color="gray", ls="-.", lw=1.0, label=f"Estado estacionario T_ss = {T_ss:.1f} °C")
    ax1.axvline(params.tau, color="green", ls=":", lw=1.2, label=f"Constante de tiempo tau = {params.tau:.0f} s ({T_tau:.2f} °C)")
    ax1.scatter([params.tau], [T_tau], color="green", zorder=5)
    ax1.set_title("(a) Escalón de Potencia u = 0.50 (50 W) en Lazo Abierto")
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("Temperatura [°C]")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="lower right", fontsize=8)

    # Panel 2: Error absoluto entre simulación numérica y solución analítica
    ax2 = axes[0, 1]
    err_euler = np.abs(step_res["T_euler"] - step_res["T_analytical"])
    err_rk4 = np.abs(step_res["T_rk4"] - step_res["T_analytical"])
    ax2.semilogy(t, np.maximum(err_euler, 1e-15), "b-", lw=1.5, label=f"Error |Euler - Analítica| (RMSE={step_res['metrics_euler']['rmse_degC']:.2e} °C)")
    ax2.semilogy(t, np.maximum(err_rk4, 1e-15), "r-", lw=1.5, label=f"Error |RK4 - Analítica| (RMSE={step_res['metrics_rk4']['rmse_degC']:.2e} °C)")
    ax2.set_title("(b) Error Absoluto de Integración vs. Solución Analítica (Escala Log)")
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylabel("Error Absoluto |T_sim - T_exacta| [°C]")
    ax2.grid(True, which="both", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 3: Ensayo de escalones múltiples 
    ax3 = axes[1, 0]
    tm = multi_res["t"]
    ax3.plot(tm, multi_res["T_analytical"], "k-", lw=2.2, label="Analítica por Tramos Exacta")
    ax3.plot(tm, multi_res["T_rk4"], "r--", lw=1.8, label="Simulación Cámara Térmica (RK4)")
    ax3.set_title("(c) Validación en Escalones Múltiples (40% -> 80% -> 0% Enfriamiento)")
    ax3.set_xlabel("Tiempo [s]")
    ax3.set_ylabel("Temperatura [°C]")
    ax3.grid(True, alpha=0.35)
    ax3.legend(loc="upper left", fontsize=8)

    # Panel 4: Perfil de potencia aplicada en escalones múltiples y error residual
    ax4 = axes[1, 1]
    ax4.step(tm, multi_res["u"] * params.P_max, "g-", where="post", lw=2.0, label="Potencia Calefactor P_in(t) [W]")
    ax4.set_title("(d) Señal de Entrada en Lazo Abierto P_in(t) [W]")
    ax4.set_xlabel("Tiempo [s]")
    ax4.set_ylabel("Potencia Térmica [W]")
    ax4.set_ylim(-5, 105)
    ax4.grid(True, alpha=0.35)
    ax4.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "hito1_lazo_abierto.png")
    plt.savefig(fig_path)
    plt.close(fig)

    # Guardamos registro CSV 
    csv_path = os.path.join(log_dir, "hito1_lazo_abierto_evidencia.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ejecucion", "tiempo_s", "u_escalon", "T_analitica_C", "T_sim_rk4_C", "T_sim_euler_C", "error_abs_rk4_C"])
        for k in range(0, len(t), 10):
            writer.writerow([
                timestamp_str,
                f"{t[k]:.1f}",
                f"{step_res['u'][k]:.2f}",
                f"{step_res['T_analytical'][k]:.6f}",
                f"{step_res['T_rk4'][k]:.6f}",
                f"{step_res['T_euler'][k]:.6f}",
                f"{abs(step_res['T_rk4'][k] - step_res['T_analytical'][k]):.6e}",
            ])

    json_path = os.path.join(log_dir, "hito1_resumen_metricas.json")
    summary_data = {
        "proyecto": "S1 - Control inteligente vs. clasico con capa de seguridad",
        "hito": "Hito 1 - Simulacion de la camara termica en lazo abierto contrastada con respuesta analitica",
        "timestamp": timestamp_str,
        "parametros_planta": {
            "R_th_C_per_W": params.R_th,
            "C_th_J_per_C": params.C_th,
            "tau_s": params.tau,
            "P_max_W": params.P_max,
            "T_amb_C": params.T_amb,
            "dt_s": params.dt,
        },
        "metricas_escalon_50pct": {
            "T_ss_teorica_C": float(T_ss),
            "T_en_tau_teorica_C": float(T_tau),
            "T_en_tau_sim_rk4_C": float(step_res["T_rk4"][int(params.tau / params.dt)]),
            "euler": step_res["metrics_euler"],
            "rk4": step_res["metrics_rk4"],
        },
        "metricas_multiescalon_rk4": multi_res["metrics_rk4"],
    }
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    return fig_path, csv_path, json_path


if __name__ == "__main__":
    params = ThermalParams()
    step_results = run_step_experiment(params, u_step=0.5, duration_s=720.0)
    multi_results = run_multistep_experiment(params, duration_s=1200.0)
    fig_p, csv_p, json_p = save_plots_and_logs(params, step_results, multi_results)

    print("EVIDENCIA GENERADA EXITOSAMENTE")
    print(f"Figura guardada en : {fig_p}")
    print(f"Log CSV guardado en: {csv_p}")
    print(f"Resumen JSON en    : {json_p}")
    print(f"RMSE RK4 vs Analítica (Escalón 50%): {step_results['metrics_rk4']['rmse_degC']:.4e} °C")
    print(f"RMSE Euler vs Analítica (Escalón 50%): {step_results['metrics_euler']['rmse_degC']:.4e} °C")
