
from dataclasses import dataclass


@dataclass
class ThermalParams:
    """
    Esta clase almacena los parámetros físicos de la cámara térmica simulada.
    """
    # Resistencia térmica equivalente de las paredes de la cámara [°C/W]
    R_th: float = 0.50
    # Capacitancia térmica efectiva del volumen de aire y paredes internas [J/°C]
    C_th: float = 240.0
    # Potencia térmica máxima del actuador (calefactor eléctrico) [W]
    P_max: float = 100.0
    # Temperatura ambiente exterior [°C]
    T_amb: float = 20.0
    # Temperatura inicial de la cámara al comenzar la simulación [°C]
    T_init: float = 20.0
    # Período de muestreo discreto del sistema ciber-físico [s]
    dt: float = 1.0

    @property
    def tau(self) -> float:
        
        #Esta función calcula la constante de tiempo térmica tau = R_th * C_th [s].
        #Representa el tiempo que tarda la cámara en alcanzar el 63.2% de su cambio total.
        
        return self.R_th * self.C_th

    @property
    def K_static(self) -> float:
        
        #Esta función devuelve la ganancia estática térmica K = R_th [°C/W].
        #Indica cuántos °C sube la temperatura estacionaria por cada Watt aplicado.
        
        return self.R_th


@dataclass
class SafetyAndControlParams:
    
    #Esta clase define las referencias de confort y los límites de la capa de seguridad (Safety Shield).
    
    # Temperatura de referencia deseada (setpoint de confort) [°C]
    T_ref: float = 35.0
    # Banda de confort aceptable alrededor de la referencia (±0.5 °C)
    comfort_band: float = 0.5
    # Límite inferior estricto de la banda de seguridad [°C]
    T_min_safe: float = 22.0
    # Límite superior estricto de la banda de seguridad [°C]
    T_max_safe: float = 38.0
    # Margen preventivo del escudo de seguridad para actuar antes de cruzar el borde [°C]
    shield_margin: float = 0.25


def get_perturbed_params(base: ThermalParams, r_factor: float = 1.0, c_factor: float = 1.0) -> ThermalParams:
    
    #Esta función genera una copia de los parámetros físicos con perturbaciones (ej. ±20%).
    #Aquí multiplicamos R_th y C_th por sus factores respectivos para evaluar robustez.
    
    return ThermalParams(
        R_th=base.R_th * r_factor,
        C_th=base.C_th * c_factor,
        P_max=base.P_max,
        T_amb=base.T_amb,
        T_init=base.T_init,
        dt=base.dt,
    )
