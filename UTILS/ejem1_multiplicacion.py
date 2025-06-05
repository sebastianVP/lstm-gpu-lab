import numpy as np
import cupy as cp
import time
N= 5000 

# NumPy(CPU)
start   = time.time()
a_cpu   = np.random.randn(N,N)
b_cpu   = np.random.randn(N,N)
res_cpu = np.dot(a_cpu,b_cpu)
print(f"CPU time: {time.time()-start:.4f} segundos")

# CuPy(GPU)
start    = time.time()
a_gpu    = cp.random.randn(N,N)
b_gpu    = cp.random.randn(N,N)

res_gpu  = cp.dot(a_gpu,b_gpu)
cp.cuda.Device(0).synchronize()

print(f"GPU time: {time.time()-start:.4f}segundos")