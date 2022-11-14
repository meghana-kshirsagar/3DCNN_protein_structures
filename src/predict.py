import argparse, os
import h5py

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Pockets3DCNN, get_old_params_3D_CNN
from load_data import VoxelDataset, H5PYDataset
from utils import get_pr_curve
from utils import get_early_prec, get_fmax, get_aucpr, get_auc
from sklearn import metrics
    
import matplotlib.pyplot as plt

def predict(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: ' + str(device))
    
    n_outputs = 2
    input_shape, filters_shape, filter_sizes = get_old_params_3D_CNN()
    model = Pockets3DCNN(input_shape, filters_shape, filter_sizes, n_outputs)
    model.load_state_dict(torch.load(params.model_path))
    model.to(device)
    
    test_set = VoxelDataset(params.test_path)
    test_loader = DataLoader(test_set, batch_size=params.bs, shuffle=False, num_workers=8)
    all_scores = []
    all_labels = []
    for i, sampled_batch in enumerate(test_loader):
        data = sampled_batch['grid']
        y = sampled_batch['label'].squeeze()
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        output = model(data)
        proba = F.softmax(output, dim=1)
        score = proba.data.cpu().numpy()
        all_scores += score[:,1].tolist()
        all_labels += y.numpy().tolist()
    
    print(len(all_scores), len(all_labels))
    combined = np.column_stack((all_labels,all_scores))
    print(combined.shape)
    rec, prec = get_pr_curve(all_labels, all_scores)
    rocauc = get_auc(all_labels, all_scores)
    aucpr = get_aucpr(all_labels, all_scores)
    earlyprec = get_early_prec(all_labels, all_scores)
    print('ROC: ',rocauc,'AUC-PR: ',aucpr,'Early-prec: ',earlyprec)
    
    plt.figure()
    plt.plot(rec,prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xticks([x/10 for x in list(range(0,11,1))])
    plt.yticks([x/10 for x in list(range(0,11,1))])
    plt.savefig(os.path.join(params.output_dir, params.output_prefix + '_pr_curve_new.jpg'))
    np.save(os.path.join(params.output_dir, params.output_prefix + '_labels_predictions.npy'), combined)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True, help='Input h5 file name')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save predictions')
    parser.add_argument('--output_prefix', type=str, required=False, help='output prefix')
    opt = parser.parse_args()
    opt.bs = 128

    predict(opt)
    
    
    
#from matplotlib.backends.backend_pdf import PdfPages
#with PdfPages('') as pdf:
#    pdf.savefig(fig)
