import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Directorios
input_dir = 'tras'
output_dir = 'absn'

# Constantes
h = 6.626e-34  # Constante de Planck (J·s)
c = 299792458  # Velocidad de la luz (m/s)
ev = 6.242e18  # Conversión de Joules a electronvolts (eV)

R2_tol = 0.8  # Umbral de R^2 para la segmentación
Dqtol = 5  # Umbral de diferencia de ángulo en grados

# Crear el directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Función para ajustar una línea recta y calcular R^2
def fit_and_evaluate(x, y):
    model = LinearRegression()
    # Convertir x a la forma correcta para el modelo
    x_reshaped = x.reshape(-1, 1)  
    model.fit(x_reshaped, y)
    y_pred = model.predict(x_reshaped)
    r2 = r2_score(y, y_pred)
    return model.coef_[0], model.intercept_, r2

# Función recursiva para la segmentación del gráfico TP_s vs hν
def segment_spectrum(x, y, r2_tol):
    segments = []
    coef, intercept, r2 = fit_and_evaluate(x, y)

    if r2 >= r2_tol:
        segments.append((x, y, coef, intercept, r2))
    else:
        # Bisectar los datos y ajustar a cada segmento
        mid_index = len(x) // 2
        left_segments = segment_spectrum(x[:mid_index], y[:mid_index], r2_tol)
        right_segments = segment_spectrum(x[mid_index:], y[mid_index:], r2_tol)
        segments.extend(left_segments + right_segments)

    return segments

# Función para fusionar segmentos basados en la inclinación
def merge_segments(segments, dqtol):
    merged_segments = []
    i = 0
    
    while i < len(segments):
        x_seg, y_seg, coef, intercept, r2 = segments[i]
        
        # Usar solo el coeficiente de la pendiente para calcular el ángulo
        angle_i = np.arctan(coef) * (180 / np.pi)  # Calcular el ángulo en grados
        
        j = i + 1
        while j < len(segments):
            x_next_seg, y_next_seg, coef_next, intercept_next, r2_next = segments[j]
            angle_j = np.arctan(coef_next) * (180 / np.pi)  # Calcular el ángulo en grados
            
            dqi = angle_j - angle_i  # Diferencia de ángulo
            if abs(dqi) < dqtol:  # Si la diferencia de ángulo es menor que el umbral
                # Fusionar segmentos
                x_seg = np.concatenate((x_seg, x_next_seg))
                y_seg = np.concatenate((y_seg, y_next_seg))
                coef, intercept, r2 = fit_and_evaluate(x_seg, y_seg)  # Ajustar nueva línea
                angle_i = np.arctan(coef) * (180 / np.pi)  # Actualizar ángulo
                j += 1
            else:
                break  # Salir del bucle si no se puede fusionar más

        merged_segments.append((x_seg, y_seg, coef, intercept, r2))
        i = j  # Avanzar al siguiente segmento no fusionado

    return merged_segments

for filename in os.listdir(input_dir):
    full_path = os.path.join(input_dir, filename)
    if os.path.isfile(full_path):
        # Leer y procesar datos
        num_filas = 877 - 86
        datos = pd.read_csv(full_path, skiprows=86, nrows=num_filas, header=None, delim_whitespace=True)

        x = datos[0].to_numpy() * 1e-9  # Convertir la longitud de onda de nm a metros
        y_1 = datos[1].to_numpy()  # Transmitancia
        y = -np.log10(y_1 * 0.01)  # Absorbancia

        # Cálculo de energía fotónica (hν) en eV
        x_prima = (h * c * ev) / x

        # Cálculo de (αhν)^2
        y_prima = (y * x_prima) ** 2

        # Aplicar el filtro de Savitzky-Golay para suavizar los datos
        y_filtrado = savgol_filter(y_prima, window_length=11, polyorder=2)

        # Ecuaciones (5) y (6) para calcular TP_s y φ
        TP_f = y_filtrado
        phi = np.max(x_prima) - np.min(x_prima)
        TP_s = phi * TP_f / np.max(TP_f)

        # Segmentación usando R^2_tol
        segments = segment_spectrum(x_prima, TP_s, R2_tol)
        # Fusionar segmentos usando Dqtol
        merged_segments = merge_segments(segments, Dqtol)

        # Graficar los segmentos fusionados
        plt.figure()
        for segment in merged_segments:
            x_seg, y_seg, coef, intercept, r2 = segment
            plt.plot(x_seg, y_seg, label=f'Segmento R^2={r2:.2f}')
            plt.plot(x_seg, coef * x_seg + intercept, '--', label=f'Ajuste lineal R^2={r2:.2f}')

        plt.xlabel('hν (eV)')
        plt.ylabel('TP_s')
        plt.title(f'Segmentación y fusión del gráfico de Tauc para {filename}')
        plt.legend()
        plt.show()

        # Guardar los datos procesados en el directorio de salida
        output_filename = os.path.join(output_dir, f"processed_{filename}")
        processed_data = pd.DataFrame({'hν (eV)': x_prima, 'TP_s': TP_s})
        processed_data.to_csv(output_filename, index=False)
