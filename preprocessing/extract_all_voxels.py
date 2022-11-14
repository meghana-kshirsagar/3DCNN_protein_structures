from extract_all_res import extract_all_res
from extract_voxels import cut_box_site_test_pdb
from datasets import pockets_dataset, parse_dataset
import os, sys
import numpy as np

if __name__ == "__main__":
    foldnum = sys.argv[1]
    #train_dir = '../GVP/gvp/data/task1-folds-unique-cat-codes/'
    #output_dir = 'data/task1-folds-unique-cat-codes/'
    #suffix = format('-nearby-pv-procedure-min-rank-7-window-40-stride-1-final-task1-fold-%s' % foldnum)
    train_dir = '../GVP/gvp/data/new_data_october/'
    output_dir = 'data/new_data_october/'
    suffix = format('-nearby-pv-procedure-min-rank-7-window-40-stride-1-cv-fold-%s' % foldnum)
    pdb_dir = os.path.join(output_dir, 'pdb_files')
    ptf_dir = os.path.join(output_dir, 'ptf_files')
    
    trainset, valset, testset = pockets_dataset(train_dir, suffix)
    #trainset, valset, testset = pockets_dataset(os.path.join(train_dir,'data/new_data_october/'), suffix)
    print('Finished reading in pockets data...')
    
    outprefix = 'test'  # 'test' # 'val'
    pdb_list = parse_dataset(testset, train_dir, pdb_dir)
    #pdb_list = parse_dataset(trainset, train_dir, pdb_dir)
    #pdb_list = parse_dataset(valset, train_dir, pdb_dir)
    #pdb_list, labels = parse_dataset(trainset, train_dir, pdb_dir)
    with open(format('%s/%s_pdb_list_fold%s.txt' % (output_dir,outprefix, foldnum)),'w') as listoutput:
        for fn in pdb_list:
            listoutput.write(fn+'\n')
    print('Finished writing frames to PDB...')

    #with open(format('data/%s_pdb_list_fold0.txt' % outprefix),'r') as listinput:
    #    pdb_list = [line.strip() for line in listinput]
    #sys.exit(0)
    
    numpy_outdir = os.path.join(output_dir, 'numpy')
    #labels_out_file = os.path.join(numpy_outdir, format('%s_fold0_labels.npy' % outprefix))   ## not used any more
    #np.save(labels_out_file, labels)
    
    extract_all_res(pdb_list, ptf_dir)
    print('Finished converting to PTF files...')

    print('Converting to voxels now...')
    ## CONVERT TO VOXELS
    for pdb_fn in pdb_list:
        ptf_fn = pdb_fn.replace('pdb','ptf')
        ptf_id = (pdb_fn.split('/')[-1]).split('.')[0]
        out_file = numpy_outdir+'/'+ptf_id+'_0.dat'
        if not os.path.exists(out_file):
            print('Converting!!')
            cut_box_site_test_pdb(ptf_id,ptf_fn, pdb_fn, numpy_outdir)
    

