# MalarIA

## Clasificación de células sanguíneas mediante Deep Learning

Este proyecto implementa un modelo de clasificación de imágenes para diferenciar células sanguíneas *parasitadas por malaria* (Parasitized) de células *no infectadas* (Uninfected).

Para ello se utiliza el dataset *Cell Images for Detecting Malaria* disponible en Kaggle y una arquitectura *EfficientNet-B0* preentrenada, aplicando técnicas de *Transfer Learning* y *Fine-Tuning*.

El proyecto fue desarrollado en *Python con PyTorch* y está preparado para ejecutarse en *Google Colab utilizando GPU mediante CUDA*.

---

## Objetivo

El objetivo del proyecto es construir un clasificador capaz de recibir una imagen microscópica de una célula sanguínea y asignarla a una de las siguientes clases:

text
Parasitized
Uninfected


El flujo general del proyecto es:


Dataset ->
Carga y exploración ->
Preprocesamiento ->
División de los datos ->
EfficientNet-B0 ->
Transfer Learning ->
Fine-Tuning ->
Evaluación ->
Visualización de resultados


---

# Dataset

Se utiliza el dataset:

*Cell Images for Detecting Malaria*

Disponible en Kaggle bajo el identificador:

text
iarunava/cell-images-for-detecting-malaria


El dataset contiene imágenes reales de células sanguíneas distribuidas en dos carpetas:


Parasitized/
Uninfected/


El dataset no se almacena directamente dentro del repositorio. El código se encarga de descargarlo automáticamente durante la ejecución.

---

# Pasos realizados

## 1. Configuración del entorno

Primero se importan las librerías necesarias para trabajar con:

* imágenes;
* redes neuronales;
* GPU;
* visualización;
* descarga del dataset.

Las principales herramientas utilizadas son:


Python
PyTorch
Torchvision
Matplotlib
Pillow
KaggleHub
CUDA


También se fija una semilla aleatoria para facilitar la reproducibilidad de los resultados.

---

## 2. Verificación de GPU

El código comprueba que Google Colab tenga una GPU disponible mediante CUDA.

python
torch.cuda.is_available()


Si CUDA está disponible, el entrenamiento se realiza utilizando la GPU.

El programa también muestra:

Versión de PyTorch
GPU disponible
Memoria disponible


El uso de GPU permite reducir considerablemente el tiempo de entrenamiento.

---

## 3. Descarga del dataset

El dataset se descarga automáticamente desde Kaggle utilizando kagglehub.

Por lo tanto, no es necesario descargar manualmente las imágenes antes de ejecutar el proyecto.

Una vez descargado, el programa busca automáticamente las carpetas:


Parasitized
Uninfected


y prepara una estructura compatible con ImageFolder de PyTorch.

---

## 4. Exploración inicial de los datos

Después de cargar el dataset se verifica:

* número total de imágenes;
* clases disponibles;
* cantidad de imágenes por clase;
* estructura del dataset.

También se muestran varias imágenes reales utilizando *Matplotlib* para comprobar visualmente los datos.

Por ejemplo:

Parasitized       Uninfected

[imagen]          [imagen]
[imagen]          [imagen]


Esto permite comprobar que las imágenes se hayan cargado correctamente antes del entrenamiento.

---

## 5. Preprocesamiento

Antes de introducir las imágenes en la red neuronal se aplica un conjunto de transformaciones.

Todas las imágenes son redimensionadas a:


128 × 128 píxeles


Durante entrenamiento también se utilizan transformaciones como:


Horizontal Flip
Vertical Flip
Rotaciones


Estas operaciones se aplican sobre las imágenes originales durante el entrenamiento para aumentar la variabilidad de los datos.

Finalmente, las imágenes son convertidas a tensores y normalizadas.

---

## 6. División del dataset

El dataset se divide en tres grupos:


70 % → Entrenamiento
15 % → Validación
15 % → Prueba


### Entrenamiento

Se utiliza para actualizar los parámetros del modelo.

### Validación

Se utiliza durante el entrenamiento para observar el comportamiento del modelo sobre imágenes que no se están utilizando directamente para actualizar sus pesos.

### Prueba

Se utiliza únicamente al final para evaluar el rendimiento del modelo entrenado.

La división se realiza manteniendo la proporción de las dos clases.

---

## 7. Creación de DataLoaders

Después de dividir el dataset se crean DataLoader para:

Train
Validation
Test


Los DataLoaders permiten procesar las imágenes por grupos o *batches*.

El tamaño del batch se adapta a la memoria disponible de la GPU con el objetivo de aprovechar los recursos de Google Colab.

También se vuelve a visualizar un batch de imágenes para comprobar que el preprocesamiento funciona correctamente.

---

## 8. Carga del modelo

Para la clasificación se utiliza:

text
EfficientNet-B0


La red se carga con pesos previamente entrenados en ImageNet.

python
models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)


La capa de clasificación original se sustituye por una nueva capa con dos salidas:

Parasitized
Uninfected


De esta manera, EfficientNet-B0 se adapta al problema específico del proyecto.

---

## 9. Transfer Learning

El entrenamiento se realiza utilizando *Transfer Learning*.

En una primera etapa se congelan las capas encargadas de extraer características visuales y se entrena principalmente la nueva capa de clasificación.

Esta etapa permite aprovechar las características previamente aprendidas por EfficientNet-B0.

La primera fase se ejecuta durante:

35 épocas


---

## 10. Fine-Tuning

Posteriormente se inicia una segunda fase denominada *Fine-Tuning*.

En esta etapa se descongelan las últimas capas de EfficientNet-B0 para que puedan adaptarse mejor a las características específicas de las imágenes de células sanguíneas.

Esta segunda fase se ejecuta durante:

15 épocas


En total el modelo se entrena durante:


35 + 15 = 50 épocas


---

## 11. Entrenamiento

Durante cada época se realizan dos procesos.

### Entrenamiento

El modelo:

1. recibe un batch de imágenes;
2. genera una predicción;
3. compara la predicción con la etiqueta real;
4. calcula el error;
5. actualiza sus parámetros.

### Validación

Después se evalúa el modelo sobre el conjunto de validación sin actualizar sus parámetros.

Durante cada época se muestran resultados similares a:


Epoch 01/50
Train Loss: ...
Train Accuracy: ...
Validation Loss: ...
Validation Accuracy: ...


---

## 12. Optimización del entrenamiento

Para reducir el tiempo de ejecución en GPU se utiliza *Mixed Precision*.

Esto permite realizar determinadas operaciones con menor precisión numérica y aprovechar mejor la GPU.

También se utilizan:

text
CUDA
pin_memory
DataLoader con múltiples workers
batch size adaptado a la GPU


Estas configuraciones buscan mantener el tiempo de entrenamiento dentro de un rango razonable para Google Colab.

---

## 13. Guardado del mejor modelo

Durante el entrenamiento se compara el resultado obtenido en el conjunto de validación.

Cuando se obtiene una nueva mejor exactitud, el modelo se guarda automáticamente.

El archivo generado es:

best_malaria_efficientnet_b0.pt


Al finalizar las 50 épocas se vuelve a cargar este modelo para realizar la evaluación final.

---

## 14. Curvas de entrenamiento

Después del entrenamiento se generan gráficos con Matplotlib para mostrar:

Training Loss
Validation Loss
Training Accuracy
Validation Accuracy


Esto permite observar cómo cambia el desempeño del modelo a lo largo de las 50 épocas.

También se indica gráficamente el momento en el que comienza el Fine-Tuning.

---

## 15. Evaluación

El mejor modelo se evalúa utilizando el conjunto de prueba.

Se calculan diferentes métricas:

Accuracy
Precision
Recall
F1-score


Estas métricas permiten analizar el desempeño del modelo desde diferentes perspectivas y no depender únicamente del porcentaje total de predicciones correctas.

---

## 16. Matriz de confusión

También se genera una matriz de confusión.

Esta permite comparar:

Clase real
vs.
Clase predicha


y observar:

* clasificaciones correctas;
* falsos positivos;
* falsos negativos.

La matriz se genera utilizando PyTorch y se visualiza mediante Matplotlib.

---

## 17. Visualización de predicciones

El programa selecciona imágenes reales del conjunto de prueba y muestra:

Clase real: ...
Predicción: ...
Confianza: ...


Esto permite inspeccionar visualmente los resultados obtenidos por el modelo.

---

## 18. Grad-CAM

Finalmente se utiliza *Grad-CAM* para generar mapas de calor sobre algunas imágenes.

Estos mapas permiten visualizar qué zonas de una imagen tuvieron mayor influencia sobre la predicción realizada por EfficientNet-B0.

El resultado se muestra de forma similar a:

Imagen original ->

Predicción del modelo ->
       
Mapa Grad-CAM


Grad-CAM se utiliza únicamente como herramienta de interpretación visual del comportamiento del modelo.

---

# Resumen del flujo implementado


Descarga del dataset ->
        
Verificación de clases ->
        
Visualización de imágenes ->
        
Preprocesamiento ->
        
Train / Validation / Test ->
        
DataLoaders ->
        
EfficientNet-B0 preentrenada ->
        
Transfer Learning 
35 épocas ->
        
Fine-Tuning
15 épocas ->
        
Selección del mejor modelo ->
        
Evaluación en Test ->
        
Accuracy
Precision
Recall
F1-score ->
        
Matriz de confusión  ->
        
Predicciones visuales ->
      
Grad-CAM


---

# Cómo ejecutar el código

## 1. Abrir Google Colab

El proyecto está preparado para ejecutarse principalmente utilizando Google Colab.

Se debe abrir el archivo:


malaria_efficientnet.ipynb


en Google Colab.

---

## 2. Activar GPU

Antes de ejecutar el notebook se debe seleccionar una GPU.

En Google Colab:


Entorno de ejecución ->
        
Cambiar tipo de entorno de ejecución ->
        
Acelerador de hardware ->
        
T4 GPU


El código verifica automáticamente que CUDA esté disponible.

Si CUDA no está disponible, el entrenamiento no debería iniciarse.

---

## 3. Ejecutar el notebook

Ejecutar las celdas del notebook en orden.

También puede utilizarse:


Entorno de ejecución
→ Ejecutar todo


El código realizará automáticamente:


1. Instalación de KaggleHub
2. Descarga del dataset
3. Preparación de las imágenes
4. División de los datos
5. Visualización de ejemplos
6. Creación del modelo
7. Entrenamiento
8. Fine-Tuning
9. Evaluación
10. Visualización de resultados
11. Grad-CAM


---

## 4. Esperar el entrenamiento

El entrenamiento consta de:


50 épocas


divididas en:


35 épocas de Transfer Learning
15 épocas de Fine-Tuning


Durante la ejecución se muestra el progreso de cada época.

Por ejemplo:


Epoch 01/50 | Train Acc: ... | Val Acc: ...
Epoch 02/50 | Train Acc: ... | Val Acc: ...
...
Epoch 50/50 | Train Acc: ... | Val Acc: ...


---

## 5. Revisar los resultados

Al terminar se mostrarán:


Curvas de entrenamiento
Métricas finales
Matriz de confusión
Predicciones sobre imágenes
Grad-CAM
Tiempo de ejecución


También se genera el archivo con los mejores pesos:


best_malaria_efficientnet_b0.pt


---

# Requisitos

El entorno debe contar con:

Python 3
PyTorch
Torchvision
Matplotlib
Pillow
KaggleHub
CUDA


En Google Colab, la mayoría de estas dependencias ya se encuentran instaladas.

La dependencia adicional de KaggleHub se instala desde el propio notebook.

---

# Estructura del repositorio

Una estructura básica del repositorio puede ser:

MalarIA/
│
├── README.md
│
├── malaria_efficientnet.ipynb
│
├── results/
│   └── figures/
│
└── slides/


No es necesario almacenar el dataset dentro de GitHub, ya que se descarga automáticamente desde Kaggle.

---

# Consideraciones

Este proyecto fue desarrollado con fines académicos para aplicar técnicas de Deep Learning a imágenes biomédicas.

El modelo clasifica imágenes individuales del dataset utilizado y no debe interpretarse como un sistema clínico de diagnóstico de malaria.

Los resultados dependen de:

* las imágenes utilizadas;
* la división del dataset;
* la GPU disponible;
* los hiperparámetros;
* el proceso de entrenamiento.

---

# Tecnologías utilizadas


Python
PyTorch
Torchvision
EfficientNet-B0
Transfer Learning
Fine-Tuning
CUDA
Matplotlib
KaggleHub
Grad-CAM
Google Colab


---

## Autores


Cristhian Reaño Ccoscco - 20201181
José Zapata Castro - 20211845
Isaac Huamani Sulca - 20215421
Sebastian Saco Alvarado - 20221648
Carlos Camilo Vásquez Morales - 20202583
