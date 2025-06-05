import time
import cupy as cp
import numpy as np


# CPU

start = time.time()
x_cpu = np.linspace(0,100,10**8)
y_cpu = np.sin(x_cpu)**2 + np.cos(x_cpu)**2

print(f"CPU tiempo (identidad trigonometrica):{time.time()-start:.4f}")

#CuPy(GPU)
start = time.time()
x_cpu = cp.linspace(0,100,10**8)
y_cpu = cp.sin(x_cpu)**2 + cp.cos(x_cpu)**2

cp.cuda.Device(0).synchronize()
print(f"GPU:{time.time()-start:.4f}")
