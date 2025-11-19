
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

def main():
    """Carga el dataset, entrena un RandomForest y guarda el modelo."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(script_dir)
    
    dataset_path = os.path.join(base_path, 'pushup_training_dataset.csv')
    model_output_path = os.path.join(script_dir, 'pushup_classifier_model.pkl')
    encoder_output_path = os.path.join(script_dir, 'pushup_label_encoder.pkl')
    scaler_output_path = os.path.join(script_dir, 'pushup_scaler.pkl')

    print("Iniciando entrenamiento del modelo...")
    print(f"Cargando dataset desde: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"No se encontró el dataset en {dataset_path}")
        return

    X = df.drop(['class', 'source_file'], axis=1)
    y = df['class']
    sources = df['source_file']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, encoder_output_path)
    print(f"Codificador de etiquetas guardado en: {encoder_output_path}")
    print("Clases encontradas y codificadas:", dict(zip(le.classes_, le.transform(le.classes_))))


    X_train, X_test, y_train, y_test, sources_train, sources_test = train_test_split(
        X, y_encoded, sources, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("\nAplicando oversampling para balancear las clases...")
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    print("Distribución de clases tras oversampling:", pd.Series(y_train_res).value_counts().to_dict())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_output_path)
    print(f"Scaler guardado en: {scaler_output_path}")

    print("\nEntrenando el modelo RandomForestClassifier...")
    
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        min_samples_leaf=2,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train_res)
    print("Entrenamiento completado")

    print("\nEvaluando el rendimiento del modelo...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPrecisión (Accuracy) en el conjunto de prueba: {accuracy:.2f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\nMatriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print("   (Columnas: Predicción / Filas: Realidad)")
    print(cm)

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

    # Visualizar la matriz de confusión
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Matriz de Confusión (Pesos Balanceados)')
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.show()

    joblib.dump(model, model_output_path)
    print(f"\nModelo entrenado guardado exitosamente en: {model_output_path}")

if __name__ == '__main__':
    main()