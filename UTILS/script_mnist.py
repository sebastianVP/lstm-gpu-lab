"""
pip install torch torchvision torchaudio

datasets.FashionMNIST
datasets.CIFAR10
datasets.CIFAR100

trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)


"""
import os
import torchvision
from torchvision import datasets,transforms

print(torchvision.__version__)

#RUTA DONDE SE ESPERA QUE ESTÉ EL DATASET MNIST
mnist_path = "./data/MNIST/processed/training.pt"

# VERIFICAR SI EL ARCHIVO EXISTE
if os.path.exists(mnist_path):
    print("El dataset MNIST ya está descargado")
else:
    print("Descargando el dataset MNIST...")

#TRANSFORMACION BASICA: Convierte la imagen a tensor
transform= transforms.ToTensor()

#CARGA DEL DATASET
trainset = datasets.MNIST(root="./data",train=True,download=True,transform=transform)
testset  = datasets.MNIST(root="./data",train=False,download=True,transform=transform)

print(f" Trainset size:{len(trainset)} muestras")
print(f" Testset  size:{len(testset)}  muestras")
