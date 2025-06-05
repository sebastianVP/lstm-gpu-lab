"""
(base) soporte@IGP-168:~/Documents/lstm-gpu-lab/UTILS$ python ejem3_modelos.py 
Usando: cpu
Tiempo Total entreamiento en cpu:197.8669 segundos
(base) soporte@IGP-168:~/Documents/lstm-gpu-lab/UTILS$ python ejem3_modelos.py 
Usando: cuda
Tiempo Total entreamiento en cuda:4.7899 segundos

"""

import torch
import torch.nn as nn
import torch.optim as optim
import time


# AQUI HABLAMOS AHORA DE UN DISPOSITIVO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = torch.device("cpu")

print("Usando:",device)


# DATOS ARTIFICIALES
x = torch.linspace(-1,1,10**7).view(-1,1).to(device)
y = 3*x + 0.5+0.1*torch.randn_like(x).to(device)
#print(x)
#print(y)
# RED SIMPLE
model = nn.Sequential(
    nn.Linear(1,64),
    nn.ReLU(),
    nn.Linear(64,1)
).to(device)

# Entrenamiento
optimizer = optim.Adam(model.parameters(),lr=0.01)
criterion = nn.MSELoss()

start = time.time()
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss   = criterion(output,y)
    loss.backward()
    optimizer.step()
print(f"Tiempo Total entreamiento en {device}:{time.time()-start:.4f} segundos")