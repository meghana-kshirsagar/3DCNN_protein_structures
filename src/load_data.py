import os
import pandas as pd
import numpy as np
import time

from sklearn.utils import class_weight

import torch
from torch.utils.data import Dataset, DataLoader

import h5py
from store_pytable import getH5column
import tables


def load_voxel_data(filepath):
    h5file = tables.open_file(filepath, mode="r")
    dataColumn = getH5column(h5file, "data")
    labelColumn = getH5column(h5file, "label")
    X=dataColumn[:]
    y=labelColumn[:]
    print('load_voxel_data: ',X.shape,y.shape)
    y = (y >= 116)  ### 87 or 116 -- 30A or 40A
    y = y.astype('long')
    return X, y


def load_mean(input_dir):
    mean_file = os.path.join(input_dir, 'X_mean.npy')
    if os.path.exists(mean_file):
        print('Reading mean from: ',mean_file)
        X_mean = np.load(mean_file)
    else:
        X_mean = 0
    return X_mean

def normalize(X, X_mean):
    return X-X_mean

 

class VoxelDataset(Dataset):
    def __init__(self, filepath, voxel_input_dir='/tmp/'):
        self.filepath = filepath
        start = time.time()
        self.X, self.y = load_voxel_data(filepath)
        print('Num pos:',sum(self.y==1),' Num neg:',sum(self.y==0))

        # load mean for normalization
        #self.X_mean = load_mean(voxel_input_dir)
        #self.X = normalize(self.X, self.X_mean)

        shuffle=False
        if shuffle:
            p = np.random.permutation(len(self.y))
            self.X = self.X[p,:,:,:,:]
            self.y = self.y[p]

        print('Took this long to load data: ',time.time()-start)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        voxel = self.X[idx,:]
        label = self.y[idx]
        #print('Getting index: ',idx,voxel.shape,label.shape)
        sample = {'grid': voxel, 'label': label}
        return sample
        
    def get_class_weights(self):
        class_weights=class_weight.compute_class_weight('balanced', classes=np.unique(self.y), y=self.y)
        class_weights=torch.tensor(class_weights,dtype=torch.float)
        print(class_weights)
        #class_weights = torch.tensor([0.5559, 4.9750], dtype=torch.float)
        return class_weights



class H5PYDataset(Dataset):
    
    def __init__(self, fn, voxel_input_dir='/tmp/'):
        self.fn = fn
        with h5py.File(fn, "r") as f:
            self.y = f["defaultNode/label"][:]
            self.length = len(self.y)
            print('Num pos:',sum(self.y==1),' Num neg:',sum(self.y==0))
            #self.y = (self.y >= 40)
            self.y = self.y.astype('long')

        # load mean for normalization
        #self.X_mean = load_mean(voxel_input_dir)
            
    def __getitem__(self, idx):    
        with h5py.File(self.fn, "r") as f:
            data = f["defaultNode"]["data"][idx]
            label = self.y[idx]

        return {
            "grid": data,
            #"grid": normalize(data, self.X_mean),
            "label": label
        }
    
    def __len__(self):
        return self.length
    
    def get_class_weights(self):
        class_weights=class_weight.compute_class_weight('balanced', classes=np.unique(self.y), y=self.y)
        class_weights=torch.tensor(class_weights,dtype=torch.float)
        print(class_weights)
        return class_weights
    

class ProteinDataset(Dataset):
    
    def __init__(self, pdb_list_infile, prefix, input_dir='/tmp/'):
        self.input_dir = input_dir
        # load list of proteins
        with open(pdb_list_infile,'r') as listin:
            files = [line.strip() for line in listin]
        self.proteins = files

        # load mean for normalization
        #self.X_mean = load_mean(input_dir)
            
        # load labels
        self.y = np.load(os.path.join(input_dir, format('%s_labels.npy' % prefix)))
        self.length = len(self.y)
        self.y = (self.y >= 40)
        self.y = self.y.astype('long')
        #print('LABELS shape: ',self.y.shape, 'num prots: ',self.length)

        # if shuffle, then make a random ordering
        #if shuffle:
        #    self.order = np.random.permutation(self.length)
        #    self.proteins = [self.proteins[x] for x in self.order]
        #    self.y = self.y[self.order, :]
            
    def __getitem__(self, idx):    
        start = time.time()
        fname = self.proteins[idx]
        fname = fname.split('/')[-1].replace('.pdb','_0.dat')
        fname = os.path.join(self.input_dir, fname)
        data = np.load(fname, allow_pickle=True)

        labels = self.y[idx, :data.shape[0]]

        #print(fname,data.shape, labels.shape)
        #print('Took this long to get idx: ',time.time()-start,flush=True)

        return {
            "grid": data,
            #"grid": normalize(data, self.X_mean),
            "label": labels
        }
    
    def __len__(self):
        return self.length
    
    def get_class_weights(self):
        class_weights=class_weight.compute_class_weight('balanced', classes=np.unique(self.y), y=self.y)
        class_weights=torch.tensor(class_weights,dtype=torch.float)
        print(class_weights)
        return class_weights
 


class NumpyDataset(Dataset):
    def __init__(self, filepath, voxel_input_dir='/tmp/'):
        self.filepath = filepath
        start = time.time()
        self.X = np.load(format('%s_X.npy' % filepath))
        self.y = np.load(format('%s_y.npy' % filepath))
        self.y = self.y.type(torch.LongTensor)

        # load mean for normalization
        self.X_mean = load_mean(voxel_input_dir)
        self.X = normalize(self.X, self.X_mean)
        print('Size:',self.X.shape,flush=True)

        print('Took this long to load data: ',time.time()-start,flush=True)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        voxel = self.X[idx,:]
        label = self.y[idx]
        #print('Getting index: ',idx,voxel.shape,label.shape)
        sample = {'grid': voxel, 'label': label}
        return sample

