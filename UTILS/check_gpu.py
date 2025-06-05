
"""
VERIFICAR SI TIENES UNA GPU NVIDIA ACTIVA
-Verificar el modelo de la GPU
-Memoria
-Procesos
"""
import subprocess

def check_gpu():
    result = subprocess.run(["nvidia-smi"],capture_output=True,text=True)
    print(result.stdout)

check_gpu()
