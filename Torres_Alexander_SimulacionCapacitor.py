# Proyecto-simulacion-capacitor
#Alexander Torres Andrade - 31/01/2026 - TA II

import numpy as np
import matplotlib.pyplot as plt
import os
import time

# 1. --- CONFIGURACIÓN Y CARPETA DE SALIDA ---
# Crear carpeta para resultados
folder_name = "Resultados_Capacitor"
if not os.path.exists(folder_name):
    # Crear la carpeta si no existe
    os.makedirs(folder_name)

# Inicia la medición del tiempo total del programa
inicio_total = time.perf_counter()

# 2. --- PARÁMETROS DEL CIRCUITO ---
# Definición de constantes físicas.
R = 1000.0       # Ohms
C = 10e-6        # 10 uF
tau = R * C      
V_peak = 5.0     # 5V
freq = 20        # 20 Hz
periodo = 1/freq # Periodo de la señal cuadrada

# 3. --- PARÁMETROS DE SIMULACIÓN ---
# Configuración del tiempo de simulación.
dt = 0.0005      # Paso de tiempo
t_final = 0.2    # Tiempo total de simulación
t = np.arange(0, t_final, dt) # Vector de tiempo

# El paso de tiempo (dt) emula el 'Maximum Timestep' de simuladores como LTSpice.
# Un valor de 0.0005s asegura capturar la dinámica del capacitor (tau = 0.01s)
# cumpliendo con el criterio de estabilidad numérica.

# Función de voltaje de entrada (onda cuadrada).
def get_vin(time): 
    """ Simula una fuente de onda cuadrada: V_peak durante la mitad del periodo, 0V el resto"""
    return V_peak if (time % periodo) < (periodo / 2) else 0.0

# Ecuación diferencial.
def dv_dt(v_cap, time):
    """Representa la EDO del sistema: dVc/dt = (Vin - Vc) / (R*C)"""
    return (get_vin(time) - v_cap) / (R * C)

# 4. --- EJECUCIÓN DE MÉTODOS ---
# Algoritmo de integración numérica usando Euler y RK4.
v_euler = np.zeros(len(t))
v_rk4 = np.zeros(len(t))
v_input = np.array([get_vin(ti) for ti in t])

inicio_euler = time.perf_counter()
for i in range(1, len(t)):
    # Método de Euler: aproximación lineal simple
    v_euler[i] = v_euler[i-1] + dt * dv_dt(v_euler[i-1], t[i-1])
fin_euler = time.perf_counter()

inicio_rk4 = time.perf_counter()
for i in range(1, len(t)):
    # Método de Runge-Kutta de cuarto orden: promedio ponderado de 4 pendientes, más preciso.
    tn, vn = t[i-1], v_rk4[i-1]
    k1 = dv_dt(vn, tn)
    k2 = dv_dt(vn + 0.5 * dt * k1, tn + 0.5 * dt)
    k3 = dv_dt(vn + 0.5 * dt * k2, tn + 0.5 * dt)
    k4 = dv_dt(vn + dt * k3, tn + dt)
    v_rk4[i] = vn + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
fin_rk4 = time.perf_counter()

print(f"Tiempo de ejecución Euler: {fin_euler - inicio_euler:.6f} segundos")
print(f"Tiempo de ejecución RK4: {fin_rk4 - inicio_rk4:.6f} segundos")

# 5. --- CÁLCULO DE ERROR (MSE) ---
# Usamos RK4 como referencia de alta precisión para comparar a Euler, dada su alta convergencia.
error_abs = np.abs(v_rk4 - v_euler)
mse_euler = np.mean((v_rk4 - v_euler)**2)

fin_total = time.perf_counter()
print(f"Tiempo total de ejecución del programa: {fin_total - inicio_total:.6f} segundos")

# 6. --- GENERACIÓN DE RESULTADOS ---

# Generar tabla para el reporte.
print(f"{'Tiempo (s)':<12} | {'Vin (V)':<10} | {'V_Euler (V)':<12} | {'V_RK4 (V)':<10} | {'Error Abs':<12}")
print("-" * 65)
# A continuación, mostramos los primeros *50* pasos (*cantidad ajustable hasta 400*)
for i in range(0, 50): 
    print(f"{t[i]:<12.4f} | {v_input[i]:<10.2f} | {v_euler[i]:<12.4f} | {v_rk4[i]:<10.4f} | {error_abs[i]:<12.6f}")

# Guardar Gráfica
plt.figure(figsize=(10, 5))
plt.plot(t, v_input, 'k--', alpha=0.3, label='Entrada $V_{in}$')
plt.plot(t, v_euler, 'r-', label=f'Euler (MSE: {mse_euler:.5f})')
plt.plot(t, v_rk4, 'b-', label='Runge-Kutta 4 (Ref)')
plt.title('Simulación Numérica: Respuesta del Capacitor')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{folder_name}/grafica_comparativa.png")
print(f"-> Gráfica guardada en {folder_name}/")

# Guardar Reporte de Texto
with open(f"{folder_name}/reporte_errores.txt", "w") as f:
    f.write("REPORTE DE ANALISIS NUMERICO - CAPACITOR\n")
    f.write("========================================\n")
    f.write(f"Parametros: R={R}, C={C}, dt={dt}\n")
    f.write(f"Error Cuadratico Medio (Euler vs RK4): {mse_euler:.10f}\n")
    f.write(f"Error Absoluto Maximo: {np.max(error_abs):.10f}\n\n")
print(f"-> Reporte guardado en {folder_name}/")

plt.show()
