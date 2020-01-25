from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch

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

data = pd.read_csv('data/dset.csv')
cases = data.shape[0] - 1
len_of_n = 9
case = 0
for case in range(cases) :
    n = data.iloc[case+1,0:len_of_n]
    m = data.iloc[case+1,len_of_n:]
    print(n)
    print(m)
