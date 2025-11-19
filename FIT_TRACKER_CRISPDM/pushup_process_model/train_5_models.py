import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Configuración
EXERCISE_NAME = 'pushup'

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_PATH, f'{EXERCISE_NAME}_training_dataset.csv')
OUTPUT_DIR = os.path.join(BASE_PATH, f'training_results_{EXERCISE_NAME}')
MODEL_DIR = os.path.join(BASE_PATH, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def save_metrics(model_name, y_true, y_pred, train_time, infer_time, le):
    """Calcula y guarda métricas."""
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
    """Guarda gráfico de matriz de confusión."""
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
    """Guarda gráfico de importancia de características."""
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
        importances = np.var(model.theta_, axis=0)
        xlabel = 'Varianza de Medias entre Clases (Theta)'
    else:
        print(f"    {model_name} no expone atributos de importancia. Se omite gráfico FI.")
        return

    if importances is None:
        return

    indices = np.argsort(importances)[-20:]
    
    plt.figure(figsize=(12, 10))
    plt.title(f'Importancia de Características (Top 20) - {model_name}', fontsize=16)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel(xlabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'fi_{model_name}.png'))
    plt.close()
    print(f"    Gráfico Importancia de Características guardado.")

def train_random_forest(X_train, y_train, X_test, y_test, le, feature_names):
    """Entrena RF usando ROS y class_weight."""
    print("\n1. Entrenando Random Forest (ROS+Bal)")
    model_name = 'RandomForest (ROS+Bal)'
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=2,
        class_weight='balanced'
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start_time
    
    metrics = save_metrics(model_name, y_test, y_pred, train_time, infer_time, le)
    plot_confusion_matrix(model_name, y_test, y_pred, le)
    plot_feature_importance(model_name, model, feature_names)
    
    model_path = os.path.join(MODEL_DIR, f'{EXERCISE_NAME}_model_rf.joblib')
    joblib.dump(model, model_path)
    print(f"  Modelo principal guardado en: {model_path}")
    
    return metrics

def train_gaussian_nb(X_train, y_train, X_test, y_test, le, feature_names):
    """Entrena Gaussian Naive Bayes usando ROS."""
    print("\n2. Entrenando GaussianNB (ROS)")
    model_name = 'GaussianNB (ROS)'
    
    model = GaussianNB()
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start_time
    
    metrics = save_metrics(model_name, y_test, y_pred, train_time, infer_time, le)
    plot_confusion_matrix(model_name, y_test, y_pred, le)
    plot_feature_importance(model_name, model, feature_names)
    
    return metrics

def train_mlp(X_train, y_train, X_test, y_test, le, feature_names):
    """Entrena MLP usando ROS."""
    print("\n3. Entrenando Red Neuronal MLP (ROS)")
    model_name = 'MLPClassifier (FCNN-ROS)'
    
    model = MLPClassifier(random_state=42,
                           hidden_layer_sizes=(100, 50), 
                           max_iter=1000, 
                           early_stopping=True,
                           validation_fraction=0.1)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start_time
    
    metrics = save_metrics(model_name, y_test, y_pred, train_time, infer_time, le)
    plot_confusion_matrix(model_name, y_test, y_pred, le)
    plot_feature_importance(model_name, model, feature_names)
    
    return metrics

def train_xgboost(X_train, y_train, X_test, y_test, le, feature_names):
    """Entrena XGBoost usando ROS."""
    print("\n4. Entrenando XGBoost (ROS)")
    model_name = 'XGBoost (ROS)'
    
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start_time
    
    metrics = save_metrics(model_name, y_test, y_pred, train_time, infer_time, le)
    plot_confusion_matrix(model_name, y_test, y_pred, le)
    plot_feature_importance(model_name, model, feature_names)
    
    return metrics

def train_lightgbm(X_train, y_train, X_test, y_test, le, feature_names):
    """Entrena LightGBM usando ROS."""
    print("\n5. Entrenando LightGBM (ROS)")
    model_name = 'LightGBM (ROS)'
    
    model = LGBMClassifier(random_state=42, verbosity=-1)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start_time
    
    metrics = save_metrics(model_name, y_test, y_pred, train_time, infer_time, le)
    plot_confusion_matrix(model_name, y_test, y_pred, le)
    plot_feature_importance(model_name, model, feature_names)
    
    return metrics

def main():
    """Flujo principal de entrenamiento y comparación."""
    
    all_model_results = []
    
    print(f"Iniciando pipeline de entrenamiento para: {EXERCISE_NAME}")
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"No se encontró el dataset en {DATASET_FILE}")
        return

    df = df.fillna(0)
    X = df.drop(columns=['class', 'source_file', 'video_segment'], errors='ignore')
    y_raw = df['class']
    feature_names = X.columns.tolist()
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    encoder_path = os.path.join(MODEL_DIR, f'{EXERCISE_NAME}_label_encoder.pkl')
    joblib.dump(le, encoder_path)
    print(f"Codificador de etiquetas guardado en: {encoder_path}")
    print("Clases encontradas:", dict(zip(le.classes_, le.transform(le.classes_))))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Aplicar RandomOversampling
    print("\nAplicando RandomOverSampler al set de entrenamiento...")
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    print("Distribución de clases (entrenamiento) ANTES de ROS:", pd.Series(y_train).value_counts().to_dict())
    print("Distribución de clases (entrenamiento) DESPUÉS de ROS:", pd.Series(y_train_res).value_counts().to_dict())
    
    print("\nAplicando StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_path = os.path.join(MODEL_DIR, f'{EXERCISE_NAME}_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler guardado en: {scaler_path}")
        
    all_model_results.append(train_random_forest(X_train_scaled, y_train_res, X_test_scaled, y_test, le, feature_names))
    all_model_results.append(train_gaussian_nb(X_train_scaled, y_train_res, X_test_scaled, y_test, le, feature_names))
    all_model_results.append(train_mlp(X_train_scaled, y_train_res, X_test_scaled, y_test, le, feature_names))
    all_model_results.append(train_xgboost(X_train_scaled, y_train_res, X_test_scaled, y_test, le, feature_names))
    all_model_results.append(train_lightgbm(X_train_scaled, y_train_res, X_test_scaled, y_test, le, feature_names))
    
    results_df = pd.DataFrame(all_model_results)
    results_df = results_df.sort_values(by='f1_weighted', ascending=False)
    
    output_csv = os.path.join(OUTPUT_DIR, f'model_comparison_{EXERCISE_NAME}.csv')
    results_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*60)
    print(f"Comparación de Modelos para '{EXERCISE_NAME}' completada.")
    print(f"Resultados guardados en: {output_csv}")
    print("="*60)
    print(results_df.to_markdown(index=False, floatfmt=".4f"))

if __name__ == '__main__':
    main()