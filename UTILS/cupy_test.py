"""
USO DE GPU con CuPy
CuPy es una libreria similar a NumPy, pero que usa la GPU de NVIDIA para hacer cálculos 
en paralelo con CUDA
"""
import numpy as np
import cupy as cp
import time

N = 10000  # TAMANO DE LA MATRIZ

# CPU con NumPy
start = time.time()
x_cpu = np.random.randn(N,N)
np.dot(x_cpu,x_cpu)
print("CPU time",time.time()-start)

# GPU con CuPy
start = time.time()
x_gpu = cp.random.randn(N,N)
cp.dot(x_gpu,x_gpu)
cp.cuda.Device(0).synchronize()
print("GPU time",time.time() -start)

