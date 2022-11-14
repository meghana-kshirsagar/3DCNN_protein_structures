import tensorflow as tf
import numpy as np
import mdtraj as md
import pandas as pd
import glob
import os, re


abbrev = {"ALA" : "A" , "ARG" : "R" , "ASN" : "N" , "ASP" : "D" , "CYS" : "C" , "CYM" : "C", "GLU" : "E" , "GLN" : "Q" , "GLY" : "G" , "HIS" : "H" , "ILE" : "I" , "LEU" : "L" , "LYS" : "K" , "MET" : "M" , "PHE" : "F" , "PRO" : "P" , "SER" : "S" , "THR" : "T" , "TRP" : "W" , "TYR" : "Y" , "VAL" : "V"}
lookup = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9, 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8, 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 'N': 2, 'Y': 18, 'M': 12}

#DATA_DIR = "../GVP/gvp/data"

def pockets_dataset(DATA_DIR, suffix):
    #will have [(xtc,pdb,index,residue,1/0),...]
    #suffix = '-protein-split'  # '-nsp10-nsp5'  # -TEM-1MY0-1BSQ-nsp5
    X_train = np.load(os.path.join(DATA_DIR,format("X-train%s.npy" % suffix)))
    y_train = np.load(os.path.join(DATA_DIR,format("y-train%s.npy" % suffix)),allow_pickle=True)
    trainset = list(zip(X_train,y_train))   

    X_validate = np.load(os.path.join(DATA_DIR,format("X-val%s.npy" % suffix)))
    y_validate = np.load(os.path.join(DATA_DIR,format("y-val%s.npy" % suffix)),allow_pickle=True)
    valset = list(zip(X_validate,y_validate))    

    #suffix = re.sub('-cv.*','',suffix)
    #print('new suffix, ',suffix)
    X_test = np.load(os.path.join(DATA_DIR,format("X-test%s.npy" % suffix)))
    y_test = np.load(os.path.join(DATA_DIR,format("y-test%s.npy" % suffix)),allow_pickle=True)
    testset = list(zip(X_test, y_test))    

#    trainset = DynamicLoader(trainset, batch_size)
#    valset = DynamicLoader(valset, batch_size)
#    testset = DynamicLoader(testset, batch_size)
    
#    output_types = (tf.float32, tf.int32, tf.int32, tf.float32)
#    trainset = tf.data.Dataset.from_generator(trainset.__iter__, output_types=output_types).prefetch(3)
#    valset = tf.data.Dataset.from_generator(valset.__iter__, output_types=output_types).prefetch(3)
#    testset = tf.data.Dataset.from_generator(testset.__iter__, output_types=output_types).prefetch(3)
    
    return trainset, valset, testset


## process each dataset, save to pdb file
def parse_dataset(dataset, training_dir, out_dir):
    #Batch will have [(xtc,pdb,index,residue,1/0),...]
    pdbs = []
    #can parallelize to improve speed
    for ex in dataset:
        x, y = ex
        x[1] = re.sub('.*msft','msft',x[1])
        pdb = md.load(os.path.join(training_dir, x[1]))
        prot_iis = pdb.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = pdb.atom_slice(prot_iis)
        pdbs.append(prot_bb)
    
    B = len(dataset)
    L_max = np.max([pdb.top.n_residues for pdb in pdbs])
    #X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    #S = np.zeros([B, L_max], dtype=np.int32)
    # -1 so we can distinguish 0 pocket volume and padded indices later 

    ## not using below code any more
    #y = np.zeros([B, L_max], dtype=np.int32)-1 

    resids = []
    pdb_list = []
    for i,ex in enumerate(dataset):
        x, targs = ex
        traj_fn, pdb_fn, traj_iis = x
        traj_iis = int(traj_iis)
        
        pdb = md.load(os.path.join(training_dir, pdb_fn))
        traj_fn = re.sub('.*msft','msft',traj_fn)
        struc = md.load_frame(os.path.join(training_dir, traj_fn),traj_iis,top=pdb)
        ## save to pdb
        PDB_ID = (pdb_fn.split('/')[-1]).split('.')[0]
        traj_name = traj_fn.split('/')[-1].replace('.xtc','')
        pdb_out = os.path.join(out_dir, format('%s_frame%d.pdb' % (traj_name, traj_iis)))
        print('Saving to: ',pdb_out)
        struc.save_pdb(pdb_out)
        pdb_list.append(pdb_out)
        
        prot_iis = struc.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = struc.atom_slice(prot_iis)
        l = prot_bb.top.n_residues
        #y[i, :l] = targs    ## not using this anymore
        
    return pdb_list #, y  ## not using this anymore

    

