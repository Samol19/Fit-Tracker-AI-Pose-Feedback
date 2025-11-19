import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

def train_squat_classifier(dataset_path):
    """Entrena un RandomForestClassifier, evalúa rendimiento y muestra análisis de errores."""
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset no encontrado en {dataset_path}")
        return

    # Carga de datos
    df = pd.read_csv(dataset_path)
    print("Iniciando entrenamiento del modelo de Squat...")

    # Preparación de datos
    # Seleccionar columnas de features (excluyendo label, source_file, frame)
    exclude_cols = ['label', 'source_file', 'frame']
    features = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nTotal de features: {len(features)}")
    print(f"Features: {features[:10]}...")  # Mostrar primeras 10
    
    X = df[features]
    y = df['label']
    sources = df['source_file'] if 'source_file' in df.columns else pd.Series(['unknown'] * len(df))

    # Codificar las etiquetas de texto a números usando LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Clases encontradas: {le.classes_}")
    print(f"Distribución de clases:")
    for clase in le.classes_:
        count = (y == clase).sum()
        print(f"  - {clase}: {count} muestras")

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test, sources_train, sources_test = train_test_split(
        X, y_encoded, sources, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\nTamaño del set de entrenamiento: {len(X_train)} muestras")
    print(f"Tamaño del set de prueba: {len(X_test)} muestras")

    # Escalado de Features
    print("\nAplicando StandardScaler a las features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenamiento del Modelo
    print("\nEntrenando el modelo RandomForest...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    print("Entrenamiento completado")

    # --- 4. Evaluación del Modelo ---
    print("\n--- Evaluación del Modelo ---")
    y_pred = model.predict(X_test_scaled)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPrecisión (Accuracy): {accuracy:.2f}")

    # Classification Report
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\nAnálisis de errores de clasificación")
    errors = 0
    for i, (real, pred) in enumerate(zip(y_test, y_pred)):
        if real != pred:
            errors += 1
            source_file = sources_test.iloc[i]
            real_class = le.inverse_transform([real])[0]
            pred_class = le.inverse_transform([pred])[0]
            print(f"Error #{errors}: Archivo '{source_file}'")
            print(f"  -> Clase Real: '{real_class}', Predicción: '{pred_class}'")
    
    if errors == 0:
        print("No se encontraron errores de clasificación en el conjunto de prueba")

    print("\nImportancia de Features:")
    importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances)

    print("\nMatriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print("(Filas: Clase Real, Columnas: Clase Predicha)")
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Clase Predicha')
    plt.ylabel('Clase Real')
    plt.title('Matriz de Confusión - Squat')
    plt.show()

    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, 'squat_classifier_model.pkl')
    encoder_path = os.path.join(script_dir, 'squat_label_encoder.pkl')
    scaler_path = os.path.join(script_dir, 'squat_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModelo guardado en: {model_path}")
    print(f"Label encoder guardado en: {encoder_path}")
    print(f"Scaler guardado en: {scaler_path}")


if __name__ == '__main__':
    dataset_csv_path = r'\FIT_TRACKER\squat_training_dataset.csv'
    train_squat_classifier(dataset_csv_path)
