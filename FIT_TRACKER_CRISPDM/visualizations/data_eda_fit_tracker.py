"""
Script para generar evidencias visuales del preprocesamiento de datos (CRISP-DM)
- Gráfico de detección de repeticiones (picos y ventanas)
- Tabla de features estadísticas (snippet del dataset)
- Gráfico de distribución de clases (barras)
- Gráfico de cajas comparativo (boxplot de una feature clave por clase)

Guarda los gráficos en la carpeta ./evidencias_preprocesamiento/
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

# conf
DATASET_DIR = "FIT_TRACKER/pushup_model"  



N_HEAD = 5

# Configuración por modelo
import random
if "squat" in DATASET_DIR:
    FEATURE_CLAVE = "avg_knee_angle"
    FEATURE_CLAVE_EST = FEATURE_CLAVE + "_mean"
    CLASE_OBJETIVO = "label"
    # Buscar archivo CSV de resultado de squat
    squat_csvs = glob.glob("FIT_TRACKER/resultados_analisis/squat/squat_correcto/squat_correcto_front/*.csv")
    if not squat_csvs:
        raise FileNotFoundError("No se encontró ningún archivo de squat en resultados_analisis/squat/squat_correcto/squat_correcto_front/")
    RAW_SIGNAL_CSV = random.choice(squat_csvs)
elif "pushup" in DATASET_DIR:
    FEATURE_CLAVE = "shoulder_wrist_vertical_diff"
    FEATURE_CLAVE_EST = FEATURE_CLAVE + "_mean"
    CLASE_OBJETIVO = "class"
    # Buscar archivo CSV de resultado de pushup
    pushup_csvs = glob.glob("FIT_TRACKER/resultados_analisis/pushup/pushup_correcto/*.csv")
    if not pushup_csvs:
        raise FileNotFoundError("No se encontró ningún archivo de pushup en resultados_analisis/pushup/pushup_correcto/")
    RAW_SIGNAL_CSV = random.choice(pushup_csvs)
elif "plank" in DATASET_DIR:
    FEATURE_CLAVE = "shoulder_wrist_vertical_diff"
    FEATURE_CLAVE_EST = FEATURE_CLAVE + "_mean"
    CLASE_OBJETIVO = "class"
    # Buscar archivo CSV de resultado de plank
    plank_csvs = glob.glob("FIT_TRACKER/resultados_analisis/plank/plank_correcto/*.csv")
    if not plank_csvs:
        raise FileNotFoundError("No se encontró ningún archivo de plank en resultados_analisis/plank/plank_correcto/")
    RAW_SIGNAL_CSV = random.choice(plank_csvs)
else:
    raise ValueError("DATASET_DIR debe contener 'squat', 'pushup' o 'plank'")

os.makedirs(SALIDA_DIR, exist_ok=True)


#Gráfico de Detección/Segmentación 
df_signal = pd.read_csv(RAW_SIGNAL_CSV)
if FEATURE_CLAVE not in df_signal.columns:
    raise ValueError(f"No se encontró la columna {FEATURE_CLAVE} en {RAW_SIGNAL_CSV}")

signal = df_signal[FEATURE_CLAVE].values
frames = np.arange(len(signal))

smoothed = savgol_filter(signal, window_length=11 if len(signal) >= 11 else len(signal)//2*2+1, polyorder=3)

if "squat" in DATASET_DIR:
    # Para squat, detectar mínimos locales (repeticiones) en la señal suavizada
    from scipy.signal import find_peaks
    # Invertimos la señal para encontrar mínimos
    min_idxs, _ = find_peaks(-smoothed, distance=10, height=-120)  # height depende del rango de ángulos
    plt.figure(figsize=(12,5))
    plt.plot(frames, signal, label="Señal original (avg_knee_angle)", color="#7ec8e3", alpha=0.6)
    plt.plot(frames, smoothed, label="Señal suavizada", color="#005f73", linewidth=2)
    plt.scatter(min_idxs, smoothed[min_idxs], color="red", label="Mínimos detectados (repeticiones)")
    plt.title("Segmentación de repeticiones en squat por ángulo de rodilla")
    plt.xlabel("Frame")
    plt.ylabel("Ángulo de rodilla (avg_knee_angle)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA_DIR, "1_grafico_segmentacion_squat.png"))
    plt.close()
else:
    # Detección de picos estándar para pushup/plank
    peaks, _ = find_peaks(smoothed, distance=10, height=0.15)
    plt.figure(figsize=(12,5))
    plt.plot(frames, signal, label="Señal original", color="#7ec8e3", alpha=0.6)
    plt.plot(frames, smoothed, label="Señal suavizada", color="#005f73", linewidth=2)
    plt.scatter(peaks, smoothed[peaks], color="red", label="Picos detectados")
    plt.title("Detección de repeticiones en la señal cruda")
    plt.xlabel("Frame")
    plt.ylabel(FEATURE_CLAVE)
    handles, labels = plt.gca().get_legend_handles_labels()
    # Exclude the scatter label from the legend
    handles = handles[:2]
    labels = labels[:2]
    plt.legend(handles, labels)
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA_DIR, "1_grafico_deteccion_picos.png"))
    plt.close()

#Tabla de Features Estadísticas
# Buscar el dataset de entrenamiento
csvs = glob.glob(os.path.join(DATASET_DIR, "*_training_dataset.csv"))
if not csvs:
    raise FileNotFoundError("No se encontró *_training_dataset.csv en " + DATASET_DIR)
TRAINING_CSV = csvs[0]

df_train = pd.read_csv(TRAINING_CSV)
# Guardar snippet
df_train.head(N_HEAD).to_csv(os.path.join(SALIDA_DIR, "2_tabla_features_head.csv"), index=False)

#Gráfico de Distribución de Clases
plt.figure(figsize=(7,4))
df_train[CLASE_OBJETIVO].value_counts().plot(kind="bar", color="#219ebc")
plt.title("Distribución de clases en el dataset")
plt.xlabel("Clase")
plt.ylabel("Cantidad de repeticiones")
plt.tight_layout()
plt.savefig(os.path.join(SALIDA_DIR, "3A_distribucion_clases.png"))
plt.close()

#Gráfico de Cajas Comparativo
if FEATURE_CLAVE_EST in df_train.columns:
    plt.figure(figsize=(8,5))
    df_train.boxplot(column=FEATURE_CLAVE_EST, by=CLASE_OBJETIVO, grid=False, patch_artist=True,
                     boxprops=dict(facecolor="#90e0ef", color="#023047"),
                     medianprops=dict(color="#fb8500", linewidth=2))
    plt.title(f"Distribución de {FEATURE_CLAVE_EST} por clase")
    plt.suptitle("")
    plt.xlabel("Clase")
    plt.ylabel(FEATURE_CLAVE_EST)
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA_DIR, "3B_boxplot_feature_clave.png"))
    plt.close()
else:
    print(f"No se encontró la columna {FEATURE_CLAVE_EST} para el boxplot.")


#Distribución de clases: 1 gráfico por modelo
modelos = [
    ("pushup", "FIT_TRACKER/pushup_model", "shoulder_wrist_vertical_diff", "class", "#219ebc"),
    ("plank", "FIT_TRACKER/plank_model", "shoulder_wrist_vertical_diff", "class", "#43aa8b"),
    ("squat", "FIT_TRACKER/squat_model", "avg_knee_angle", "label", "#f8961e")
]
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#Gráfico de Clusters (PCA 2D) por modelo
for nombre, carpeta, feat, clase, color in modelos:
    csvs = glob.glob(os.path.join(carpeta, "*_training_dataset.csv"))
    if not csvs:
        continue
    df = pd.read_csv(csvs[0])
    # Selecciona solo columnas numéricas (features estadísticas)
    feature_cols = [col for col in df.columns if col.endswith('_mean') or col.endswith('_std') or col.endswith('_min') or col.endswith('_max') or col.endswith('_range')]
    if len(feature_cols) < 2:
        continue  # Necesitamos al menos 2 features para PCA
    X = df[feature_cols]
    y = df[clase]
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='Set1', s=60, edgecolor='k')
    plt.title(f'Separabilidad de Clases (PCA 2D) - {nombre}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Clase', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA_DIR, f"7_clusters_pca_{nombre}.png"))
    plt.close()
for nombre, carpeta, feat, clase, color in modelos:
    csvs = glob.glob(os.path.join(carpeta, "*_training_dataset.csv"))
    if not csvs:
        continue
    df = pd.read_csv(csvs[0])
    conteo = df[clase].value_counts().sort_index()
    plt.figure(figsize=(7,4))
    conteo.plot(kind="bar", color=color)
    plt.title(f"Distribución de clases en {nombre}")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad de repeticiones")
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA_DIR, f"4_distribucion_clases_{nombre}.png"))
    plt.close()

#Gráfico conjunto de 3 repeticiones de squat
if "squat" in DATASET_DIR:
    squat_csvs = glob.glob("FIT_TRACKER/resultados_analisis/squat/squat_correcto/squat_correcto_front/*.csv")
    if len(squat_csvs) >= 3:
        plt.figure(figsize=(12,6))
        colores = ["#7ec8e3", "#90e0ef", "#023047"]
        for i, path in enumerate(squat_csvs[:3]):
            df = pd.read_csv(path)
            signal = df["avg_knee_angle"].values if "avg_knee_angle" in df.columns else (df[["left_knee_angle","right_knee_angle"]].mean(axis=1).values)
            frames = np.arange(len(signal))
            smoothed = savgol_filter(signal, window_length=11 if len(signal) >= 11 else len(signal)//2*2+1, polyorder=3)
            min_idxs, _ = find_peaks(-smoothed, distance=10, height=-120)
            plt.plot(frames, smoothed, label=f"Repetición {i+1}", color=colores[i])
            plt.scatter(min_idxs, smoothed[min_idxs], color=colores[i], marker="x", s=60)
        plt.title("Segmentación de repeticiones en squat (3 ejemplos)")
        plt.xlabel("Frame")
        plt.ylabel("avg_knee_angle (suavizada)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SALIDA_DIR, "5_segmentacion_squat_3_reps.png"))
        plt.close()


#Gráfico de segmentación squat estilo pushup (1 repetición)
if "squat" in DATASET_DIR:
    # Usar el primer CSV de squat_correcto_front para consistencia
    squat_csvs = glob.glob("FIT_TRACKER/resultados_analisis/squat/squat_correcto/squat_correcto_front/*.csv")
    if squat_csvs:
        path = squat_csvs[0]
        df = pd.read_csv(path)
        signal = df["avg_knee_angle"].values if "avg_knee_angle" in df.columns else (df[["left_knee_angle","right_knee_angle"]].mean(axis=1).values)
        frames = np.arange(len(signal))
        smoothed = savgol_filter(signal, window_length=11 if len(signal) >= 11 else len(signal)//2*2+1, polyorder=3)
        min_idxs, _ = find_peaks(-smoothed, distance=10, height=-120)
        plt.figure(figsize=(12,5))
        plt.plot(frames, smoothed, label="Señal suavizada (avg_knee_angle)", color="#005f73", linewidth=2)
        plt.scatter(min_idxs, smoothed[min_idxs], color="red", marker="o", s=80, label="Mínimos detectados (repeticiones)")
        plt.title("Segmentación de repeticiones en squat (1 ejemplo)")
        plt.xlabel("Frame")
        plt.ylabel("Ángulo de rodilla (avg_knee_angle)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(SALIDA_DIR, "6_segmentacion_squat_1rep.png"))
        plt.close()


ANALISIS_DIR = "FIT_TRACKER/resultados_analisis/squat/analisis"
os.makedirs(ANALISIS_DIR, exist_ok=True)
csvs = glob.glob("FIT_TRACKER/resultados_analisis/squat/squat_correcto/squat_correcto_front/*.csv")
if csvs:
    path = csvs[0]
    df = pd.read_csv(path)
    signal = df["avg_knee_angle"].values if "avg_knee_angle" in df.columns else (df[["left_knee_angle","right_knee_angle"]].mean(axis=1).values)
    frames = np.arange(len(signal))
    smoothed = savgol_filter(signal, window_length=11 if len(signal) >= 11 else len(signal)//2*2+1, polyorder=3)
    min_idxs, _ = find_peaks(-smoothed, distance=10, height=-120)
    plt.figure(figsize=(12,5))
    plt.plot(frames, smoothed, label="Señal suavizada (avg_knee_angle)", color="#005f73", linewidth=2)
    plt.scatter(min_idxs, smoothed[min_idxs], color="red", marker="o", s=80, label="Mínimos detectados (repeticiones)")
    plt.title("Segmentación de repeticiones en squat (1 ejemplo)")
    plt.xlabel("Frame")
    plt.ylabel("Ángulo de rodilla (avg_knee_angle)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALISIS_DIR, "segmentacion_squat_1rep.png"))
    plt.close()

if len(csvs) >= 3:
    plt.figure(figsize=(12,6))
    colores = ["#7ec8e3", "#90e0ef", "#023047"]
    for i, path in enumerate(csvs[:3]):
        df = pd.read_csv(path)
        signal = df["avg_knee_angle"].values if "avg_knee_angle" in df.columns else (df[["left_knee_angle","right_knee_angle"]].mean(axis=1).values)
        frames = np.arange(len(signal))
        smoothed = savgol_filter(signal, window_length=11 if len(signal) >= 11 else len(signal)//2*2+1, polyorder=3)
        min_idxs, _ = find_peaks(-smoothed, distance=10, height=-120)
        plt.plot(frames, smoothed, label=f"Repetición {i+1}", color=colores[i])
        plt.scatter(min_idxs, smoothed[min_idxs], color=colores[i], marker="x", s=60)
    plt.title("Segmentación de repeticiones en squat (3 ejemplos)")
    plt.xlabel("Frame")
    plt.ylabel("avg_knee_angle (suavizada)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALISIS_DIR, "segmentacion_squat_3reps.png"))
    plt.close()

for i, path in enumerate(csvs):
    df = pd.read_csv(path)
    signal = df["avg_knee_angle"].values if "avg_knee_angle" in df.columns else (df[["left_knee_angle","right_knee_angle"]].mean(axis=1).values)
    frames = np.arange(len(signal))
    smoothed = savgol_filter(signal, window_length=11 if len(signal) >= 11 else len(signal)//2*2+1, polyorder=3)
    min_idxs, _ = find_peaks(-smoothed, distance=10, height=-120)
    plt.figure(figsize=(10,4))
    plt.plot(frames, smoothed, label="avg_knee_angle (suavizada)", color="#005f73", linewidth=2)
    plt.scatter(min_idxs, smoothed[min_idxs], color="red", marker="o", s=60, label="Mínimos (reps)")
    plt.title(f"Segmentación squat rep {i+1}")
    plt.xlabel("Frame")
    plt.ylabel("Ángulo de rodilla")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALISIS_DIR, f"segmentacion_squat_rep{i+1}.png"))
    plt.close()

print(f"Evidencias guardadas en: {SALIDA_DIR}/ y {ANALISIS_DIR}/")
