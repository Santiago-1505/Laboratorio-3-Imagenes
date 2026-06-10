# Laboratorio 3 — Clasificación de Residuos con CoAtNet

Clasificación de imágenes de residuos usando la arquitectura **CoAtNet** sobre el dataset *Garbage Classification v2* (10 clases, ~13 348 imágenes). El proyecto incluye una explicación detallada de cada capa de la red y una implementación completa en PyTorch + timm.

---

## Integrantes

| Nombre | Correo |
|--------|--------|
| Ricardo Medina Herrera | ricardo.medinah@udea.edu.co |
| Santiago Villegas Naranjo | santiago.villegasn@udea.edu.co |

---

## Contenido del repositorio

| Archivo | Descripción |
|---------|-------------|
| `lab3_imagenes.ipynb` | Notebook principal: explicación de CoAtNet + implementación completa |
| `best_coatnet.tar.gz` | Pesos del mejor modelo entrenado |

---

## Red neuronal: CoAtNet

CoAtNet es una arquitectura híbrida que combina convoluciones (MBConv) y atención (Relative Self-Attention) de forma jerárquica en cinco etapas:

| Etapa | Tipo | Función principal |
|-------|------|-------------------|
| S0 — Convolution Stem | Conv 3×3 ×2 | Extracción de características locales básicas (bordes, texturas) |
| S1 | MBConv | Patrones locales de complejidad media |
| S2 | MBConv profundo | Último nivel puramente convolucional |
| S3 | Transformer + Relative Attention | Relaciones de largo alcance entre regiones |
| S4 | Transformer final | Representaciones semánticas completas |

La **Relative Self-Attention** es la innovación clave: añade un sesgo posicional relativo que combina la flexibilidad de la atención global con el sesgo inductivo traslacional de las CNN.

---

## Pipeline de entrenamiento

1. Descarga del dataset desde Kaggle (`kagglehub`)
2. Análisis exploratorio (EDA) y visualización de distribución por clase
3. Preprocesamiento e *data augmentation*
4. Carga del modelo CoAtNet preentrenado (`timm`)
5. Entrenamiento con scheduler y *early stopping*
6. Evaluación: accuracy, balanced accuracy, matriz de confusión, reporte de clasificación
7. Visualización de predicciones correctas e incorrectas

---

## Reproducir el experimento

```bash
# Clonar el repositorio
git clone https://github.com/Santiago-1505/Laboratorio-3-Imagenes.git
cd Laboratorio-3-Imagenes

# Abrir el notebook en Colab (badge en la primera celda)
# o ejecutar localmente con Jupyter
jupyter notebook lab3_imagenes.ipynb
```

Para cargar el mejor modelo guardado:

```python
import torch
import timm

model = timm.create_model('coatnet_0_rw_224', pretrained=False, num_classes=10)
# Extraer primero: tar -xzf best_coatnet.tar.gz
model.load_state_dict(torch.load('best_coatnet.pth', map_location='cpu'))
model.eval()
```

---

## Dataset

**Garbage Classification v2** — disponible en [Kaggle](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)

| Detalle | Valor |
|---------|-------|
| Clases | 10 |
| Imágenes totales | ~13 348 |
| Resolución usada | 256×256 → redimensionado a 224×224 |

---

## Framework y dependencias principales

- Python 3.x
- PyTorch
- [timm](https://github.com/huggingface/pytorch-image-models)
- torchvision
- kagglehub
- scikit-learn
- matplotlib / seaborn
