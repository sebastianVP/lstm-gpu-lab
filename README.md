# 🧠 LSTM GPU Lab | Laboratorio de LSTM acelerado con GPU para Clima Espacial, Meteorología y Economía

Este repositorio es un laboratorio personal para aprender, experimentar y optimizar redes LSTM(Long Short-Term Memory) usando aceleración por GPU con **NVIDIA GeForce RTX (MSI)** y **Python3.12**. Incluye ejemplos prácticos de predicción de series temporales.

This repository is a personal lab to learn, experiment, and optimize LSTM (Long Short-Term Memory) networks using GPU acceleration with **NVIDIA GeForce RTX (MSI)** and **Python 3.12**. It includes hands-on examples for time series prediction.

---

## 🚀 Características | Features

- Entrenamiento acelerado con CUDA 12.2 en GPU RTX.
- Comparaciones CPU vs GPU(tiempos y consumo)
- Casos de uso reales: predicción de clima espacial, etc.
- Documentación clara, ejemplos didácticos y notebooks listos para correr.
---

## 🧪 Contenido | Contents

| Carpeta / Folder | Descripción | GPU |
|------------------|-------------|-----|
| `01_espacio_lstm/` | Predicción de índice Kp o actividad ionosférica usando datos históricos | ✅ |
| `02_meteorologia_lstm/` | Pronóstico de temperatura, humedad o precipitaciones con datos climáticos | ✅ |
| `03_economia_lstm/` | Predicción de indicadores económicos como inflación o precios de acciones | ✅ |
| `utils/` | Utilidades: carga de datos, monitoreo de GPU, normalización y visualización | ✅ |

---

## 💻 Requisitos | Requirements

- Python 3.12  
- PyTorch (versión GPU) o TensorFlow (versión GPU)  
- NVIDIA GeForce RTX (ej. MSI RTX 3060)  
- CUDA Toolkit >= 12.2  
- NVIDIA Driver >= 535.230.02  

---
## 🎯 Objetivos | Goals
- Desarrollar modelos LSTM enfocados en dominios científicos y económicos
- Analizar y visualizar resultados con herramientas modernas
- Comparar rendimiento entre CPU y GPU
- Crear un portafolio robusto de proyectos reales
- Contribuir con la comunidad científica y técnica de habla hispana e inglesa