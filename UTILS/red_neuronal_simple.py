import torch
import torch.nn as nn
import torch.optim as optim

# DETECTAR GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando: ", device)

# Datos sintéticos
x = torch.linspace(-1,1,100).view(-1,1).to(device)
y = 2*x+1+0.2*torch.rand(x.size()).to(device)

# Modelo Simple
model = nn.Sequential(
    nn.Linear(1,10),
    nn.ReLU(),
    nn.Linear(10,1)
).to(device)

# Entrenamiento
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output,y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch},Loss: {loss.item():.4f}")

