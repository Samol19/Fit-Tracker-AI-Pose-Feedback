import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    """Carga el dataset, entrena un modelo de clasificación, evalúa y guarda."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(script_dir)
    dataset_path = os.path.join('FIT_TRACKER/plank_model/plank_training_dataset.csv')

    # Rutas para exportar a la API
    api_model_dir = os.path.join(base_path, 'API_FIT_TRACKER', 'model')
    os.makedirs(api_model_dir, exist_ok=True)
    model_output_path = os.path.join(api_model_dir, 'plank_classifier_model.pkl')
    encoder_output_path = os.path.join(api_model_dir, 'plank_label_encoder.pkl')
    scaler_output_path = os.path.join(api_model_dir, 'plank_scaler.pkl')

    if not os.path.exists(dataset_path):
        print(f"Error: El dataset '{os.path.basename(dataset_path)}' no se encontró.")
        print("Asegúrate de haber ejecutado 'build_training_dataset.py' primero.")
        return

    print("Cargando dataset...")
    df = pd.read_csv(dataset_path)

    # Separar características (X) y etiquetas (y)
    X = df.drop(['class', 'video_segment'], axis=1)
    y = df['class']

    # Normalización: Usar StandardScaler para que el API pueda usar el mismo preprocesamiento
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Codificar las etiquetas de texto a números
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Clases encontradas: {le.classes_}")

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

    print(f"Tamaño del set de entrenamiento: {len(X_train)} muestras")
    print(f"Tamaño del set de prueba: {len(X_test)} muestras")

    # Entrenar el modelo RandomForest
    # Usamos class_weight='balanced' por si hay desbalance en las clases
    print("\nEntrenando el modelo RandomForest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("¡Entrenamiento completado!")

    # Evaluar el modelo
    print("\nEvaluando el rendimiento del modelo...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nPrecisión (Accuracy): {accuracy:.2f}")
    print("\nReporte de Clasificación:")
    # Usamos los nombres originales de las clases en el reporte
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Guardar el modelo, el codificador y el scaler (si existe)
    joblib.dump(model, model_output_path)
    joblib.dump(le, encoder_output_path)
    if scaler is not None:
        joblib.dump(scaler, scaler_output_path)
        print(f"Scaler guardado en: {os.path.basename(scaler_output_path)}")
    print(f"\nModelo guardado en: {model_output_path}")
    print(f"Codificador de etiquetas guardado en: {encoder_output_path}")

    # Visualizar la Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Matriz de Confusión')
    plt.ylabel('Clase Verdadera')
    plt.xlabel('Clase Predicha')
    
    # Guardar la imagen de la matriz
    confusion_matrix_path = os.path.join(script_dir, 'plank_confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    print(f"Matriz de confusión guardada en: {os.path.basename(confusion_matrix_path)}")
    # plt.show() # Descomenta si quieres que se muestre la imagen al ejecutar

if __name__ == '__main__':
    train_model()