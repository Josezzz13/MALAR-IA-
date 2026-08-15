<div align="center">

# 🩸 MalarIA
### Clasificación de células sanguíneas mediante Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU%20CUDA-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Proyecto-Académico-2E8B57?style=flat-square)

</div>

---

Este proyecto implementa un modelo de clasificación de imágenes para diferenciar células sanguíneas **parasitadas por malaria** (`Parasitized`) de células **no infectadas** (`Uninfected`).

Para ello se utiliza el dataset **Cell Images for Detecting Malaria** disponible en Kaggle y una arquitectura **EfficientNet-B0** preentrenada, aplicando técnicas de **Transfer Learning** y **Fine-Tuning**.

El proyecto fue desarrollado en **Python con PyTorch** y está preparado para ejecutarse en **Google Colab utilizando GPU mediante CUDA**.

## 📑 Tabla de contenidos

1. [Objetivo](#-objetivo)
2. [Dataset](#-dataset)
3. [Pasos realizados](#-pasos-realizados)
4. [Resumen del flujo implementado](#-resumen-del-flujo-implementado)
5. [Cómo ejecutar el código](#-cómo-ejecutar-el-código)
6. [Requisitos](#-requisitos)
7. [Estructura del repositorio](#-estructura-del-repositorio)
8. [Consideraciones](#-consideraciones)
9. [Tecnologías utilizadas](#-tecnologías-utilizadas)
10. [Autores](#-autores)

---

## 🎯 Objetivo

El objetivo del proyecto es construir un clasificador capaz de recibir una imagen microscópica de una célula sanguínea y asignarla a una de las siguientes clases:

```
Parasitized
Uninfected
```

El flujo general del proyecto es:

```
Dataset → Carga y exploración → Preprocesamiento → División de los datos →
EfficientNet-B0 → Transfer Learning → Fine-Tuning → Evaluación → Visualización de resultados
```

---

## 🦠 Dataset

Se utiliza el dataset **Cell Images for Detecting Malaria**, disponible en Kaggle bajo el identificador:

```
iarunava/cell-images-for-detecting-malaria
```

El dataset contiene imágenes reales de células sanguíneas distribuidas en dos carpetas:

```
Parasitized/
Uninfected/
```

> ⚠️ El dataset **no** se almacena directamente dentro del repositorio. El código se encarga de descargarlo automáticamente durante la ejecución.

---

## 🛠️ Pasos realizados

### 1️⃣ Configuración del entorno

Primero se importan las librerías necesarias para trabajar con imágenes, redes neuronales, GPU, visualización y descarga del dataset. Las principales herramientas utilizadas son:

| Herramienta | Uso |
|---|---|
| Python | Lenguaje base |
| PyTorch | Entrenamiento del modelo |
| Torchvision | Modelos preentrenados y transforms |
| Matplotlib | Visualización de resultados |
| Pillow | Manejo de imágenes |
| KaggleHub | Descarga automática del dataset |
| CUDA | Aceleración por GPU |

También se fija una semilla aleatoria para facilitar la reproducibilidad de los resultados.

### 2️⃣ Verificación de GPU

El código comprueba que Google Colab tenga una GPU disponible mediante CUDA:

```python
torch.cuda.is_available()
```

Si CUDA está disponible, el entrenamiento se realiza utilizando la GPU. El programa también muestra la versión de PyTorch, la GPU disponible y la memoria disponible. El uso de GPU permite reducir considerablemente el tiempo de entrenamiento.

### 3️⃣ Descarga del dataset

El dataset se descarga automáticamente desde Kaggle utilizando `kagglehub`, por lo que no es necesario descargarlo manualmente antes de ejecutar el proyecto. Una vez descargado, el programa busca automáticamente las carpetas `Parasitized` y `Uninfected`, y prepara una estructura compatible con `ImageFolder` de PyTorch.

### 4️⃣ Exploración inicial de los datos

Después de cargar el dataset se verifica el número total de imágenes, las clases disponibles, la cantidad de imágenes por clase y la estructura del dataset. También se muestran varias imágenes reales utilizando Matplotlib para comprobar visualmente los datos:

```
Parasitized       Uninfected
[imagen]          [imagen]
[imagen]          [imagen]
```

Esto permite comprobar que las imágenes se hayan cargado correctamente antes del entrenamiento.

### 5️⃣ Preprocesamiento

Antes de introducir las imágenes en la red neuronal se aplica un conjunto de transformaciones.

**📐 El resize a 128×128px.** Las imágenes del dataset vienen en tamaños variables — cada una es un recorte (*crop*) de una célula individual extraída de un frotis sanguíneo, con dimensiones que difieren célula a célula:

```python
IMAGE_SIZE = 128
...
transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
```

`Resize((128, 128))` estandariza todas las imágenes a un tamaño único, lo cual es un requisito obligatorio: la red (EfficientNet-B0) necesita un tensor de entrada de dimensión fija para poder procesar los *batches* en paralelo en la GPU. 128×128 fue el valor elegido como compromiso entre preservar suficiente detalle morfológico (la forma del parásito dentro de la célula) y mantener el costo computacional/memoria manejable.

**🔄 Por qué se aplicaron los flips y rotaciones.** Durante entrenamiento también se utilizan transformaciones de aumento de datos (*data augmentation*), aplicadas **solo** sobre el conjunto de entrenamiento (no en validación/test):

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomVerticalFlip(p=0.5)
transforms.RandomRotation(degrees=15)
```

La justificación técnica es la ausencia de orientación canónica: a diferencia de, por ejemplo, una foto de un rostro (donde "arriba" y "abajo" tienen significado), una célula bajo el microscopio no tiene una orientación biológica fija — puede aparecer rotada o reflejada en cualquier ángulo según cómo quedó posicionada en el portaobjetos y cómo se capturó el campo. Voltear la imagen horizontal o verticalmente no cambia la etiqueta (sigue siendo la misma célula, parasitada o no), pero sí le presenta al modelo variaciones geométricas adicionales del mismo ejemplo. Esto:

- Aumenta artificialmente la diversidad efectiva del set de entrenamiento sin necesidad de más datos reales.
- Fuerza al modelo a aprender características invariantes a la orientación (la forma/textura del parásito) en lugar de memorizar la pose específica de cada imagen.
- Reduce el riesgo de sobreajuste, dado que el modelo tiene alta capacidad (EfficientNet-B0 preentrenado) frente a un dataset de tamaño finito.

Finalmente, las imágenes son convertidas a tensores y normalizadas.

### 6️⃣ División del dataset

El dataset se divide en tres grupos, manteniendo la proporción de las dos clases (división estratificada):

| Subconjunto | Proporción | Rol |
|---|---|---|
| **Entrenamiento** | 70% | Se utiliza para actualizar los parámetros del modelo |
| **Validación** | 15% | Se utiliza durante el entrenamiento para observar el comportamiento del modelo sobre imágenes que no se usan para actualizar sus pesos |
| **Prueba** | 15% | Se utiliza únicamente al final para evaluar el rendimiento del modelo entrenado |

### 7️⃣ Creación de DataLoaders

Después de dividir el dataset se crean `DataLoader` para train, validation y test, que permiten procesar las imágenes por grupos o *batches*. El tamaño del batch se adapta a la memoria disponible de la GPU con el objetivo de aprovechar los recursos de Google Colab. También se vuelve a visualizar un batch de imágenes para comprobar que el preprocesamiento funciona correctamente.

### 8️⃣ Carga del modelo

Para la clasificación se utiliza **EfficientNet-B0**, cargada con pesos previamente entrenados en ImageNet:

```python
models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)
```

La capa de clasificación original se sustituye por una nueva capa con dos salidas (`Parasitized` / `Uninfected`). De esta manera, EfficientNet-B0 se adapta al problema específico del proyecto.

### 9️⃣ Transfer Learning

El entrenamiento se realiza utilizando **Transfer Learning**, en una estrategia de dos fases con bloques bien delimitados dentro de la arquitectura.

**Fase 1 — Feature Extraction.** Se congela todo el modelo (`param.requires_grad = False` en todos los parámetros) y solo se reemplaza y entrena el bloque **`classifier`**:

```python
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)
```

El bloque `classifier` (originalmente `Sequential(Dropout, Linear)` con salida a 1000 clases de ImageNet) se reemplaza por una capa `Linear` nueva con salida a 2 clases. Esta etapa permite aprovechar las características ya aprendidas por EfficientNet-B0 sin arriesgarse a modificarlas todavía.

La primera fase se ejecuta durante **35 épocas**.

### 🔟 Fine-Tuning

Posteriormente se inicia una segunda fase de **Fine-Tuning**, en la que se descongela adicionalmente el bloque **`model.features[-2:]`**: los **últimos dos bloques MBConv** del extractor de características de EfficientNet-B0 (las etapas convolucionales más profundas, cercanas a la salida):

```python
for param in model.features[-2:].parameters():
    param.requires_grad = True
```

Se eligen específicamente los últimos bloques (y no los primeros) porque en una CNN preentrenada las capas tempranas aprenden filtros genéricos de bajo nivel (bordes, texturas) que ya son útiles tal cual, mientras que las capas profundas codifican representaciones más abstractas y específicas del dominio original (ImageNet) — son las que conviene reajustar para especializarse en la morfología de células parasitadas.

Además, al iniciar esta fase se crea un **optimizador nuevo** con una tasa de aprendizaje 10 veces menor a la de la fase 1 (`lr=1e-4` vs `1e-3`), para hacer un ajuste fino con pasos pequeños en lugar de un reentrenamiento agresivo.

Esta segunda fase se ejecuta durante **15 épocas**.

> **¿Por qué 35 y 15, y no otro reparto?**
> El diseño usa épocas fijas por fase (sin *early stopping* ni *scheduler* de learning rate), y cada fase tiene un propósito distinto que justifica su duración:
> - **Fase 1 (35 épocas):** entrenar solo la capa `classifier` sobre características ya extraídas es una tarea de optimización simple y estable, con pocos parámetros entrenables frente al tamaño del dataset — el riesgo de sobreajuste es bajo, así que se le da más tiempo para encontrar el mejor límite de decisión posible.
> - **Fase 2 (15 épocas):** aquí sí se modifican pesos preentrenados en ImageNet que ya son útiles. Entrenar demasiadas épocas con el backbone descongelado es el escenario de mayor riesgo de sobreajuste y de "olvido catastrófico" de esas características genéricas transferibles. Por eso se usa una tasa de aprendizaje baja y una ventana corta — suficiente para especializar los bloques profundos sin darle margen al modelo para memorizar ruido del set de entrenamiento.

En total el modelo se entrena durante:

```
35 + 15 = 50 épocas
```

### 1️⃣1️⃣ Entrenamiento

Durante cada época se realizan dos procesos:

**Entrenamiento:** el modelo recibe un batch de imágenes, genera una predicción, compara la predicción con la etiqueta real, calcula el error y actualiza sus parámetros.

**Validación:** después se evalúa el modelo sobre el conjunto de validación sin actualizar sus parámetros.

Durante cada época se muestran resultados similares a:

```
Epoch 01/50
Train Loss: ...
Train Accuracy: ...
Validation Loss: ...
Validation Accuracy: ...
```

### 1️⃣2️⃣ Optimización del entrenamiento

Para reducir el tiempo de ejecución en GPU se utiliza **Mixed Precision**, que permite realizar determinadas operaciones con menor precisión numérica y aprovechar mejor la GPU. También se utilizan CUDA, `pin_memory`, DataLoader con múltiples *workers*, y un *batch size* adaptado a la GPU. Estas configuraciones buscan mantener el tiempo de entrenamiento dentro de un rango razonable para Google Colab.

### 1️⃣3️⃣ Guardado del mejor modelo

Durante el entrenamiento se compara el resultado obtenido en el conjunto de validación. Cuando se obtiene una nueva mejor exactitud, el modelo se guarda automáticamente:

```
best_malaria_efficientnet_b0.pt
```

Al finalizar las 50 épocas se vuelve a cargar este modelo para realizar la evaluación final.

### 1️⃣4️⃣ Curvas de entrenamiento

Después del entrenamiento se generan gráficos con Matplotlib para mostrar Training/Validation Loss y Training/Validation Accuracy, permitiendo observar cómo cambia el desempeño del modelo a lo largo de las 50 épocas. También se indica gráficamente el momento en el que comienza el Fine-Tuning.

### 1️⃣5️⃣ Evaluación

El mejor modelo se evalúa utilizando el conjunto de prueba, calculando **Accuracy**, **Precision**, **Recall** y **F1-score**. Estas métricas permiten analizar el desempeño del modelo desde diferentes perspectivas y no depender únicamente del porcentaje total de predicciones correctas.

### 1️⃣6️⃣ Matriz de confusión

También se genera una matriz de confusión que compara clase real vs. clase predicha, permitiendo observar clasificaciones correctas, falsos positivos y falsos negativos. La matriz se genera utilizando PyTorch y se visualiza mediante Matplotlib.

### 1️⃣7️⃣ Visualización de predicciones

El programa selecciona imágenes reales del conjunto de prueba y muestra la clase real, la predicción y la confianza del modelo, permitiendo inspeccionar visualmente los resultados obtenidos.

### 1️⃣8️⃣ Grad-CAM

Finalmente se utiliza **Grad-CAM** para generar mapas de calor sobre algunas imágenes, permitiendo visualizar qué zonas de una imagen tuvieron mayor influencia sobre la predicción realizada por EfficientNet-B0:

```
Imagen original → Predicción del modelo → Mapa Grad-CAM
```

Grad-CAM se utiliza únicamente como herramienta de interpretación visual del comportamiento del modelo.

---

## 🔁 Resumen del flujo implementado

```
Descarga del dataset
        ↓
Verificación de clases
        ↓
Visualización de imágenes
        ↓
Preprocesamiento (resize 128×128 + augmentación)
        ↓
Train / Validation / Test (70% / 15% / 15%)
        ↓
DataLoaders
        ↓
EfficientNet-B0 preentrenada
        ↓
Transfer Learning — 35 épocas (bloque classifier)
        ↓
Fine-Tuning — 15 épocas (bloque features[-2:])
        ↓
Selección del mejor modelo
        ↓
Evaluación en Test (Accuracy, Precision, Recall, F1-score)
        ↓
Matriz de confusión
        ↓
Predicciones visuales
        ↓
Grad-CAM
```

---

## ▶️ Cómo ejecutar el código

### 1. Abrir Google Colab

El proyecto está preparado para ejecutarse principalmente utilizando Google Colab. Se debe abrir el archivo `malaria_efficientnet.ipynb` en Google Colab.

### 2. Activar GPU

Antes de ejecutar el notebook se debe seleccionar una GPU:

```
Entorno de ejecución → Cambiar tipo de entorno de ejecución → Acelerador de hardware → T4 GPU
```

El código verifica automáticamente que CUDA esté disponible. Si CUDA no está disponible, el entrenamiento no debería iniciarse.

### 3. Ejecutar el notebook

Ejecutar las celdas del notebook en orden, o utilizar `Entorno de ejecución → Ejecutar todo`. El código realizará automáticamente:

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

### 4. Esperar el entrenamiento

El entrenamiento consta de **50 épocas**, divididas en **35 épocas de Transfer Learning** y **15 épocas de Fine-Tuning**. Durante la ejecución se muestra el progreso de cada época:

```
Epoch 01/50 | Train Acc: ... | Val Acc: ...
Epoch 02/50 | Train Acc: ... | Val Acc: ...
...
Epoch 50/50 | Train Acc: ... | Val Acc: ...
```

### 5. Revisar los resultados

Al terminar se mostrarán las curvas de entrenamiento, métricas finales, matriz de confusión, predicciones sobre imágenes, Grad-CAM y tiempo de ejecución. También se genera el archivo con los mejores pesos: `best_malaria_efficientnet_b0.pt`.

---

## 📦 Requisitos

El entorno debe contar con:

- Python 3
- PyTorch
- Torchvision
- Matplotlib
- Pillow
- KaggleHub
- CUDA

En Google Colab, la mayoría de estas dependencias ya se encuentran instaladas. La dependencia adicional de KaggleHub se instala desde el propio notebook.

---

## 🗂️ Estructura del repositorio

```
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
```

No es necesario almacenar el dataset dentro de GitHub, ya que se descarga automáticamente desde Kaggle.

---

## ⚠️ Consideraciones

Este proyecto fue desarrollado con fines académicos para aplicar técnicas de Deep Learning a imágenes biomédicas. **El modelo clasifica imágenes individuales del dataset utilizado y no debe interpretarse como un sistema clínico de diagnóstico de malaria.**

Los resultados dependen de:

- Las imágenes utilizadas
- La división del dataset
- La GPU disponible
- Los hiperparámetros
- El proceso de entrenamiento

---

## 🧰 Tecnologías utilizadas

`Python` · `PyTorch` · `Torchvision` · `EfficientNet-B0` · `Transfer Learning` · `Fine-Tuning` · `CUDA` · `Matplotlib` · `KaggleHub` · `Grad-CAM` · `Google Colab`

---

## 👥 Autores

| Nombre | Código | Participación |
|---|---|---|
| Cristhian Reaño Ccoscco | 20201181 | 100% |
| José Zapata Castro | 20211845 | 100% |
| Isaac Anthony Huamani Sulca | 20215421 | 100% |
| Sebastian Saco Alvarado | 20221648 | 100% |
| Carlos Camilo Vásquez Morales | 20202583 | 100% |
