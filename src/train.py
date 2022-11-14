import argparse

import os
import random
import pandas as pd
import numpy as np
import time

import torch

from load_data import VoxelDataset, H5PYDataset
from torch.utils.data import DataLoader
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
from torchvision import transforms

from model import Pockets3DCNN, get_old_params_3D_CNN

from sklearn.metrics import confusion_matrix, accuracy_score
from utils import get_early_prec, get_fmax, get_aucpr, get_auc


seed = 12306
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Current device: ' + str(device),flush=True)

#torch.cuda.set_per_process_memory_fraction(1, 1)
#torch.cuda.empty_cache()

# keep as many negatives as positives
def undersample_negatives(X, y, skew=1):
    npos = sum(y==1)
    npos = npos.numpy()
    negindices = np.where(y==0)[0]
    if (npos==0) or npos >= len(negindices):
        return X, y
    nneg = skew*npos
    negremove = len(negindices)-nneg
    #print('npos:',npos,'nneg:',len(negindices),' negremove:',negremove)
    if(negremove <= 0):
        return X, y
    negindices = negindices[np.random.choice(len(negindices),negremove,replace=False)] #remove fraction of elements
    keepindices = [i for i in range(0,X.shape[0]) if i not in negindices]
    X = X[keepindices, :]
    y = y[keepindices]
    #print('AFTER: ',X.shape, y.shape)
    return X, y


def data_augmentation(transf, X, y):
    X_new = transf(X)
    print(X_new.shape)
    X_new = np.row_stack((X, X_new))
    y_new = np.column_stack((y, y_new))
    print(X_new.shape)
    print(y_new.shape)
    return X_new, y_new


def save_results(output_dir, output_prefix, tr_losses, tr_accs, tr_aucprs, val_losses, val_accs, val_aucprs, val_f1, val_rocs):
    np.save(os.path.join(output_dir, output_prefix + '_train_loss.npy'), tr_losses)
    np.save(os.path.join(output_dir, output_prefix + '_train_acc.npy'), tr_accs)
    np.save(os.path.join(output_dir, output_prefix + '_train_aucpr.npy'), tr_aucprs)
    np.save(os.path.join(output_dir, output_prefix + '_val_loss.npy'), val_losses)
    np.save(os.path.join(output_dir, output_prefix + '_val_acc.npy'), val_accs)
    np.save(os.path.join(output_dir, output_prefix + '_val_aucpr.npy'), val_aucprs)
    np.save(os.path.join(output_dir, output_prefix + '_val_f1.npy'), val_f1)
    np.save(os.path.join(output_dir, output_prefix + '_val_rocs.npy'), val_rocs)


def train_3DCNN(model, opt):
    
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)
    
    print(model)
    model = model.to(device)
    
    start = time.time()
    print('Loading training data...')
    train_set = VoxelDataset(opt.train_path, opt.voxel_input_dir)
    #train_set = H5PYDataset(opt.train_path, opt.voxel_input_dir)
    class_weights = train_set.get_class_weights()
    train_loader = DataLoader(train_set, batch_size=opt.bs, shuffle=True, num_workers=2)
    print('Loading validation data...')
    val_set = VoxelDataset(opt.valid_path, opt.voxel_input_dir)
    #val_set = H5PYDataset(opt.valid_path, opt.voxel_input_dir)
    val_loader = DataLoader(val_set, batch_size=opt.bs, shuffle=True, num_workers=2)
    print('Took this long to load data: ',time.time()-start)

    ## define data augmentation
    augm_transform = Compose([
        #transforms.Lambda(lambda x: torch.Tensor(x)),
        transforms.RandomRotation(5),
        RandomTranslate([0.11, 0.11, 0.11]),
        RandomFlip()
        #transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
    ])
        
    tr_losses = np.zeros((opt.epoch,))
    tr_accs = np.zeros((opt.epoch,))
    tr_aucprs = np.zeros((opt.epoch,))
    val_losses = np.zeros((opt.epoch,))
    val_accs = np.zeros((opt.epoch,))
    val_aucprs = np.zeros((opt.epoch,))
    val_f1 = np.zeros((opt.epoch,))
    val_rocs = np.zeros((opt.epoch,))
    val_earlyprecs = np.zeros((opt.epoch,3))
        
    model.reset_parameters()
    
    criterion = torch.nn.CrossEntropyLoss() #weight=class_weights.to(device)) 
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr) #, weight_decay=1e-3)  ## L2 regularizer: 1e-5
    best_val_loss = 1e4
        
    for epoch in range(opt.epoch):
        s = time.time()
            
        model.train()
        losses = 0
        acc = 0
        y_preds = []
        y_trues = []
            
        for i, sampled_batch in enumerate(train_loader):
            data = sampled_batch['grid']
            y = sampled_batch['label'].squeeze()
            data = data.type(torch.FloatTensor)
            if opt.balancing:
                #data, y = data_augmentation(augm_transform, data, y)
                data, y = undersample_negatives(data, y, 2)
                #if(len(y)==0):
                #    continue
            y = y.to(device)
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            #print(output.shape, y.shape)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
                
            y_true = y.cpu().numpy()
            y_pred = output.data.cpu().numpy().argmax(axis=1)
            y_trues += y_true.tolist()
            y_preds += y_pred.tolist()
            acc += accuracy_score(y_true, y_pred)*100
            losses += loss.data.cpu().numpy()
                
        aucpr = get_aucpr(y_trues, y_preds)
        tr_losses[epoch] = losses/(i+1)
        tr_accs[epoch] = acc/(i+1)
        tr_aucprs[epoch] = aucpr
            
        model.eval()
        v_losses = 0
        v_acc = 0
        ## reset lists
        y_preds = []
        y_trues = []
            
        for j, sampled_batch in enumerate(val_loader):
            data = sampled_batch['grid']
            y = sampled_batch['label'].squeeze()
            data = data.type(torch.FloatTensor)
            y = y.to(device)
            data = data.to(device)
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, y)
                
            y_pred = output.data.cpu().numpy().argmax(axis=1)
            y_true = y.cpu().numpy()
            y_trues += y_true.tolist()
            y_preds += y_pred.tolist()
            v_acc += accuracy_score(y_true, y_pred)*100
            v_losses += loss.data.cpu().numpy()            
                
        cnf = confusion_matrix(y_trues, y_preds)        
        f1 = get_fmax(y_trues, y_preds)
        aucpr = get_aucpr(y_trues, y_preds)
        rocauc = get_auc(y_trues, y_preds)
        earlyprec = get_early_prec(y_trues, y_preds)
        val_aucprs[epoch] = aucpr
        val_rocs[epoch] = rocauc
        val_earlyprecs[epoch,:] = earlyprec
        earlyprec = " ".join([str(xx) for xx in earlyprec])
        val_losses[epoch] = v_losses/(j+1)
        val_accs[epoch] = v_acc/(j+1)
        val_f1[epoch] = f1
            
        current_val_loss = aucpr # v_losses/(j+1)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), os.path.join(opt.output_dir, format('%s_bs%s_lr%s_best_model.ckpt' % (opt.output_prefix,opt.bs,opt.lr))))
        save_results(opt.output_dir, opt.output_prefix, tr_losses, tr_accs, tr_aucprs, val_losses, val_accs, val_aucprs, val_f1, val_rocs)
            
        print('Epoch: {:03d} | time: {:.4f} seconds\nTrain Loss: {:.4f} | Train accuracy {:.4f}\n'
                'Valid. Loss: {:.4f} | Valid. accuracy {:.4f} | AUC-PR {:.4f} | ROC {:.4f} | ' 
                'F1 {:.4f} | Early-prec {:s}'.format(epoch+1, time.time()-s,
                        losses/(i+1), acc/(i+1), v_losses/(j+1), v_acc/(j+1), aucpr, rocauc, f1, earlyprec),flush=True)
        print('Validation confusion matrix:',flush=True)
        print(cnf,flush=True)

    save_results(opt.output_dir, opt.output_prefix, tr_losses, tr_accs, tr_aucprs, val_losses, val_accs, val_aucprs, val_f1, val_rocs)
    torch.save(model.state_dict(), os.path.join(opt.output_dir, format('%s_bs%s_lr%s_best_model.ckpt' % (opt.output_prefix,opt.bs,opt.lr))))

    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True, help='path to training h5 file')
    parser.add_argument('--valid_path', type=str, required=True, help='path to validation h5 file')
    parser.add_argument('--output_dir', type=str, required=False, help='output folder name')
    parser.add_argument('--output_prefix', type=str, required=False, help='output prefix')
    parser.add_argument('--voxel_input_dir', type=str, required=True, help='voxel folder name')
    parser.add_argument('--bs', type=int, required=True, help='batch size')
    parser.add_argument('--balancing', type=bool, required=False, default=False, help='use class balancing')
    parser.add_argument('--lr', type=float, required=True, help='learning rate')
    parser.add_argument('--epoch', type=int, required=True, help='number of epochs to train for')
    
    start_time = time.time()
    
    params = parser.parse_args()
    print(params,flush=True)
    
    n_outputs = 2
    input_shape, filters_shape, filter_sizes = get_old_params_3D_CNN()
    model = Pockets3DCNN(input_shape, filters_shape, filter_sizes, n_outputs)
    train_3DCNN(model, params)
    
    print('Total time taken: ',time.time()-start_time,flush=True)
