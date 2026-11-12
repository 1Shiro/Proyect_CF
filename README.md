# Proyecto S1: Control Inteligente vs. Clásico con Capa de Seguridad

Simulación de una cámara térmica regulada por un agente de Aprendizaje por Refuerzo (**PPO**) compitiendo contra controladores clásicos (**On/Off** y **PI**), protegidos por un **Escudo de Seguridad (*Safety Shield*)** que impide salir de la banda de temperatura $[22\text{ °C}, 38\text{ °C}]$, evaluando energía, conmutaciones y confort tanto en régimen nominal como con parámetros perturbados en $\pm 20\%$.

## Estado Actual: Entrega 1 (Hito 1)

- [`config/params.py`](config/params.py): Parámetros físicos nominales ($R_{th}, C_{th}, \tau, P_{max}, T_{amb}$), límites de seguridad y generador de perturbaciones de $\pm 20\%$.
- [`src/plant/thermal_chamber.py`](src/plant/thermal_chamber.py): Ecuación diferencial de la cámara térmica, integradores numéricos (Euler y RK4) y solución analítica exacta.
- [`src/metrics/kpis.py`](src/metrics/kpis.py): Cálculo de métricas de validación en lazo abierto y KPIs del proyecto.
- [`scripts/hito1_lazo_abierto.py`](scripts/hito1_lazo_abierto.py): **Evidencia del Primer Hito (Sección 4)** — Simulación en lazo abierto contrastada con la respuesta analítica.

## Ejecución (Hito 1)

```bash
pip install -r requirements.txt
python scripts/hito1_lazo_abierto.py
```
