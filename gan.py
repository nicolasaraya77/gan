from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.optim as optim





class dset(Dataset):
    def __init__(self,data_dir,transform=None):

        """
        Args:
            data : path de operadores y porcentaje".
            matrix : matriz del grafo.
            transform (callable, optional): transformacion a ser aplicada.
        """
        self.data = pd.read_csv(data_dir)
        self.transform = transform
        #self.graph = pd.read_csv(graph_dir)



    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        m = self.data.iloc[index,9:17].values
        n = self.data.iloc[index,0:8].values
        #matrix = self.graph.iloc[index,0:8].values
        array = np.column_stack((n,m))


        return array

""" inicializacion de datos """



data = dset("data/dset.csv")
data_in_array = np.array(data)

print(data_in_array )

tensora = torch.from_numpy(data_in_array)
tensora = tensora.view(-1,2) #disminuye la dimension del tensor a filasX2columnas
print(tensora)
print(tensora.size())
print(tensora[:, 0])

plt.figure()

plt.title("IA Data")

plt.scatter(tensora[:, 0].numpy(),tensora[:, 1].numpy(),label="n")
plt.xlabel("operadores")
plt.ylabel("trabajo de operadores")
plt.show()






""" neuronas generadora y discriminadora """
class Generador(nn.Module):
    def __init__(self):
        super(Generador, self).__init__()

        self.main = nn.Sequential(


            nn.ReLU(True),
            nn.Linear(2,2,bias=True)
        )

    def forward(self, input):
        return self.main(input)



netG = Generador()
print(netG)




class Discriminador(nn.Module):
    def __init__(self):
        super(Discriminador, self).__init__()

        self.main = nn.Sequential(
        nn.ReLU(True),
        nn.Sigmoid(),

        nn.Tanh()

        )

    def forward(self, input):
        return self.main(input)



netD = Discriminador()
print(netD)
'''

""" funciones de perdida y optimizador """

# inicializar el criterio de perdida
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
#fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD, lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG, lr=0.0002, betas=(0.5, 0.999))
'''
