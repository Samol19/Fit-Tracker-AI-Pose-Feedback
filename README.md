# Fit Tracker AI: Aplicación móvil con inteligencia artificial para guiar y monitorear rutinas de acondicionamiento físico en adultos jóvenes de Lima Metropolitana.

![Estado del Proyecto](https://img.shields.io/badge/Estado-En_Desarrollo-yellow)
![Licencia](https://img.shields.io/badge/Licencia-MIT-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Solutions-4285F4?logo=google&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-%2302569B.svg?logo=Flutter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)
![Versión](https://img.shields.io/badge/Versión-1.0.0-green)

## Resumen del Proyecto

**Fit Tracker AI** es una solución tecnológica desarrollada como Trabajo de Investigación para optar por el grado de Bachiller en Ciencias de la Computación e Ingeniería de Software en la Universidad Peruana de Ciencias Aplicadas (UPC).

El proyecto aborda la problemática de las lesiones musculoesqueléticas derivadas de la ejecución incorrecta de ejercicios físicos sin supervisión profesional. La solución consiste en una aplicación móvil multiplataforma que integra técnicas de Visión Computacional (MediaPipe) y modelos de Aprendizaje Automático (Random Forest) para proporcionar retroalimentación técnica y biomecánica en tiempo real.

El sistema prioriza la seguridad del usuario mediante la detección de alta sensibilidad (Recall) de errores posturales críticos, tales como el valgo de rodilla en sentadillas o la hiperextensión lumbar en flexiones.

## Metodología

El desarrollo del componente de Inteligencia Artificial y la estructuración del proyecto se han regido bajo la metodología estándar **CRISP-DM (Cross-Industry Standard Process for Data Mining)**, abarcando las fases de:
1.  Comprensión del Negocio
2.  Comprensión de los Datos
3.  Preparación de los Datos
4.  Modelado
5.  Evaluación
6.  Despliegue

## Arquitectura y Tecnologías

La arquitectura del sistema sigue un patrón híbrido que combina el procesamiento en el dispositivo (Edge AI) para la extracción de características y servicios en la nube para la inferencia y gestión de datos.

### Cliente Móvil
* **Framework:** Flutter (Dart).
* **Visión Computacional:** Google MediaPipe Pose para la extracción de 33 puntos clave (landmarks) corporales.
* **Protocolo de Comunicación:** WebSockets para la transmisión de datos telemétricos de baja latencia.

### Backend e Inteligencia Artificial
* **Lenguaje:** Python 3.11.
* **Modelo de Clasificación:** Random Forest Classifier. Seleccionado por ofrecer el equilibrio óptimo entre precisión y eficiencia computacional frente a modelos como XGBoost, LightGBM y Redes Neuronales (MLP).
* **Ingeniería de Características:** Cálculo de vectores de ángulos biomecánicos mediante NumPy.
* **API:** FastAPI (Implementación asíncrona).
* **Infraestructura:** Contenerización con Docker y orquestación con Docker Compose.
* **Despliegue:** Máquina Virtual en Google Cloud Platform (GCP) con sistema operativo Ubuntu 24.04.

## Validación del Modelo (Resultados)

El modelo de clasificación ha sido sometido a una evaluación rigurosa utilizando un conjunto de datos propio y validado biomecánicamente. Los resultados obtenidos demuestran la viabilidad técnica del sistema para operar en tiempo real con alta fiabilidad.

### Métricas de Desempeño
Se alcanzó un rendimiento sobresaliente en los tres ejercicios evaluados:

| Ejercicio | Precisión Global (Accuracy) | Sensibilidad en Errores Críticos (Recall) |
| :--- | :---: | :---: |
| **Plank** | 100% | 100% |
| **Squat** | 100% | 100% |
| **Push-up** | 91.3% | >95% |

*Nota: La métrica de Recall para las clases de ejecución incorrecta se priorizó durante el entrenamiento (mediante balanceo de clases) para minimizar los Falsos Negativos, garantizando que el sistema alerte sobre ejecuciones lesivas.*

### Rendimiento Computacional
* **Latencia de Inferencia:** ~7 milisegundos (ms) en entorno de producción (GCP).

## Estructura del Repositorio

La organización del código fuente se distribuye de la siguiente manera:

```text
Fit-Tracker-AI/
├── API/                 # Implementación de los modelos en un API
├── FIT_TRACKER_CRISPDM/
│   ├── dataset/             # Implementación de la API con FastAPI y WebSockets
│   ├── models/          # Artefactos serializados (.pkl) listos para producción
│   ├── process_model/        # Scripts de entrenamiento y validación (CRISP-DM)
│   └── visualizations/   # Gráficas del analisis EDA y resultados
└── real_time_feedback   # Prototipo funcional de implementación en tiempo real de los modelos
