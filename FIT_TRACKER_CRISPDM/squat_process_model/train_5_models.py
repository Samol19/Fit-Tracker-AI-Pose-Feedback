import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight # Importante
from sklearn.pipeline import Pipeline # Importante para MLP

#Importar Modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Configuración
EXERCISE_NAME = 'squat'

#Variables Globales de Ruta
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_PATH, f'{EXERCISE_NAME}_training_dataset.csv')
OUTPUT_DIR = os.path.join(BASE_PATH, f'training_results_{EXERCISE_NAME}')
MODEL_DIR = os.path.join(BASE_PATH, 'models') # Directorio para modelos finales
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_preprocess_data(filepath):
    """Carga, codifica y divide los datos (sin ROS)."""
    print(f"Cargando dataset: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filepath}")
        return None, None, None, None, None
    
    df = df.fillna(0)
    
    X = df.drop(columns=['label', 'source_file', 'video_segment'], errors='ignore')
    y_raw = df['label']
    
    # Codificar etiquetas y guardar encoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    encoder_path = os.path.join(MODEL_DIR, f'{EXERCISE_NAME}_label_encoder.pkl')
    joblib.dump(le, encoder_path)
    print(f"Codificador de etiquetas guardado en: {encoder_path}")
    print("Clases encontradas:", dict(zip(le.classes_, le.transform(le.classes_))))
    
    # División estratificada de los datos originales (sin balancear)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Datos listos: {len(X_train)} de entrenamiento, {len(X_test)} de prueba.")
    print(f"Distribución de clases (entrenamiento) SIN BALANCEAR:", pd.Series(y_train).value_counts().to_dict())
    
    return X_train, X_test, y_train, y_test, le

def save_metrics(model_name, y_true, y_pred, train_time, infer_time, le):
    """Calcula y guarda un diccionario de métricas."""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, 
                                   target_names=le.classes_, 
                                   output_dict=True,
                                   zero_division=0)
    
    print(f"  Resultados para {model_name}:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Train Time: {train_time:.4f} s")
    print(f"    Inference Time: {infer_time:.4f} s")
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'precision_weighted': report['weighted avg']['precision'],
        'recall_weighted': report['weighted avg']['recall'],
        'f1_weighted': report['weighted avg']['f1-score'],
        'train_time_sec': train_time,
        'inference_time_sec': infer_time
    }
    return metrics

def plot_confusion_matrix(model_name, y_true, y_pred, le):
    """Guarda un gráfico de la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Matriz de Confusión - {model_name}', fontsize=16)
    plt.ylabel('Clase Verdadera', fontsize=12)
    plt.xlabel('Clase Predicha', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'cm_{model_name}.png'))
    plt.close()
    print(f"    Gráfico Matriz de Confusión guardado.")

def plot_feature_importance(model_name, model, feature_names):
    """
    Guarda un gráfico de importancia de características.
    Maneja 'feature_importances_' (árboles), 'coef_' (modelos lineales) y 'theta_' (GaussianNB).
    """
    importances = None
    xlabel = 'Importancia Relativa'

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        xlabel = 'Importancia Relativa (Gini/Gain)'
    elif hasattr(model, 'coef_'):
        if model.coef_.ndim > 1:
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            importances = np.abs(model.coef_)
        xlabel = 'Magnitud del Coeficiente'
    elif hasattr(model, 'theta_'):
        # Para GaussianNB, la "importancia" es la varianza de las medias
        # de las features entre las clases.
        importances = np.var(model.theta_, axis=0)
        xlabel = 'Varianza de Medias entre Clases (Theta)'
    else:
        print(f"    {model_name} no expone atributos de importancia. Se omite gráfico FI.")
        return

    if importances is None:
        return

    indices = np.argsort(importances)[-20:] # Top 20
    
    plt.figure(figsize=(12, 10))
    plt.title(f'Importancia de Características (Top 20) - {model_name}', fontsize=16)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel(xlabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'fi_{model_name}.png'))
    plt.close()
    print(f"    Gráfico Importancia de Características guardado.")

#Funciones de Entrenamiento

def train_random_forest(X_train, y_train, X_test, y_test, le):
    """Entrena, evalúa y guarda el RF usando class_weight.""" 
    print("\n1. Entrenando Random Forest (Modelo Principal)")
    model_name = 'RandomForest (Bal)'

    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    start_time = time.time()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train) # Entrenar con datos originales
    train_time = time.time() - start_time
    
    best_rf = grid_search.best_estimator_
    print(f"  Mejores parámetros: {grid_search.best_params_}")
    
    start_time = time.time()
    y_pred = best_rf.predict(X_test)
    infer_time = time.time() - start_time
    
    metrics = save_metrics(model_name, y_test, y_pred, train_time, infer_time, le)
    plot_confusion_matrix(model_name, y_test, y_pred, le)
    plot_feature_importance(model_name, best_rf, X_train.columns)
    # Guardar modelo principal
    model_path = os.path.join(MODEL_DIR, f'{EXERCISE_NAME}_model_rf.joblib')
    joblib.dump(best_rf, model_path)
    print(f"  ¡Modelo principal guardado en: {model_path}!")
    
    # Guardar también el scaler
    scaler = StandardScaler().fit(X_train)
    scaler_path = os.path.join(MODEL_DIR, f'{EXERCISE_NAME}_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler (para inferencia) guardado en: {scaler_path}")
    
    return metrics

def train_gaussian_nb(X_train, y_train, X_test, y_test, le):
    """Entrena y evalúa Gaussian Naive Bayes usando sample_weight.""" 
    print("\n2. Entrenando GaussianNB (Probabilístico)")
    model_name = 'GaussianNB (Bal)'
    
    # GNB es sensible a la escala
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # GNB no tiene 'class_weight', así que calculamos pesos manualmente
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    model = GaussianNB()
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    infer_time = time.time() - start_time
    
    metrics = save_metrics(model_name, y_test, y_pred, train_time, infer_time, le)
    plot_confusion_matrix(model_name, y_test, y_pred, le)
    plot_feature_importance(model_name, model, X_train.columns)
    
    return metrics

def train_mlp(X_train, y_train, X_test, y_test, le):
    """Entrena y evalúa Red Neuronal (MLP/FCNN) usando sample_weight.""" 
    print("\n3. Entrenando Red Neuronal MLP (Complejo)")
    model_name = 'MLPClassifier (FCNN-Bal)'
    
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train) # <-- AQUÍ
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = MLPClassifier(random_state=42, 
                           hidden_layer_sizes=(100, 50), 
                           max_iter=1000, 
                           early_stopping=True,
                           validation_fraction=0.1)
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights) 
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    infer_time = time.time() - start_time
    
    metrics = save_metrics(model_name, y_test, y_pred, train_time, infer_time, le)
    plot_confusion_matrix(model_name, y_test, y_pred, le)
    plot_feature_importance(model_name, model, X_train.columns)
    
    return metrics

def train_xgboost(X_train, y_train, X_test, y_test, le):
    """Entrena y evalúa XGBoost usando sample_weight.""" 
    print("\n4. Entrenando XGBoost (Ensamblador)")
    model_name = 'XGBoost (Bal)'
    
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    
    start_time = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weights) 
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start_time
    
    metrics = save_metrics(model_name, y_test, y_pred, train_time, infer_time, le)
    plot_confusion_matrix(model_name, y_test, y_pred, le)
    plot_feature_importance(model_name, model, X_train.columns)
    
    return metrics

def train_lightgbm(X_train, y_train, X_test, y_test, le):
    """Entrena y evalúa LightGBM usando class_weight.""" 
    print("\n5. Entrenando LightGBM (Ensamblador)")
    model_name = 'LightGBM (Bal)'
    
    model = LGBMClassifier(random_state=42, class_weight='balanced', verbosity=-1)
    
    start_time = time.time()
    model.fit(X_train, y_train) # Entrenar con datos originales
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start_time
    
    metrics = save_metrics(model_name, y_test, y_pred, train_time, infer_time, le)
    plot_confusion_matrix(model_name, y_test, y_pred, le)
    plot_feature_importance(model_name, model, X_train.columns)
    
    return metrics

def main():
    """Flujo principal de entrenamiento y comparación.""" 
    
    all_model_results = []
    
    # Cargar y preprocesar datos
    X_train, X_test, y_train, y_test, le = load_and_preprocess_data(DATASET_FILE)
    if X_train is None:
        print("Finalizando script debido a error en carga de datos.")
        return
        
    # Entrenar y evaluar modelos
    # El balanceo se maneja internamente en cada función
    all_model_results.append(train_random_forest(X_train, y_train, X_test, y_test, le))
    all_model_results.append(train_gaussian_nb(X_train, y_train, X_test, y_test, le))
    all_model_results.append(train_mlp(X_train, y_train, X_test, y_test, le))
    all_model_results.append(train_xgboost(X_train, y_train, X_test, y_test, le))
    all_model_results.append(train_lightgbm(X_train, y_train, X_test, y_test, le))
    
    # Guardar resultados comparativos
    results_df = pd.DataFrame(all_model_results)
    results_df = results_df.sort_values(by='f1_weighted', ascending=False)
    
    output_csv = os.path.join(OUTPUT_DIR, f'model_comparison_{EXERCISE_NAME}.csv')
    results_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*60)
    print(f"Comparación de Modelos para '{EXERCISE_NAME}' completada.")
    print(f"Resultados guardados en: {output_csv}")
    print("="*60)
    print(results_df.to_markdown(index=False, floatfmt=".4f"))

#main
if __name__ == '__main__':
    main()
