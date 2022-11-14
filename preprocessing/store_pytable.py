#!/usr/bin/env python
from __future__ import division;
from __future__ import print_function;
import tables;
import numpy as np
import sys
import os
import random

DEFAULT_NODE_NAME = "defaultNode";

def init_h5_file(toDiskName, groupName=DEFAULT_NODE_NAME, groupDescription=DEFAULT_NODE_NAME):
    """
        toDiskName: the name of the file on disk
    """
    import tables;
    h5file = tables.open_file(toDiskName, mode="w", title="Dataset")
    gcolumns = h5file.create_group(h5file.root, groupName, groupDescription)
    return h5file;

class InfoToInitArrayOnH5File(object):
    def __init__(self, name, shape, atomicType):
        """
            name: the name of this matrix
            shape: tuple indicating the shape of the matrix (similar to numpy shapes)
            atomicType: one of the pytables atomic types - eg: tables.Float32Atom() or tables.StringAtom(itemsize=length);
        """
        self.name = name;
        self.shape = shape;
        self.atomicType = atomicType;

def writeToDisk(theH5Column, whatToWrite, batch_size=5000):
    """
        Going to write to disk in batches of batch_size
    """ 
    data_size = len(whatToWrite);
    last = int(data_size / float(batch_size)) * batch_size
    for i in range(0, data_size, batch_size):
        stop = (i + data_size%batch_size if i >= last
                else i + batch_size)
        theH5Column.append(whatToWrite[i:stop]);
        h5file.flush()
    
def getH5column(h5file, columnName, nodeName=DEFAULT_NODE_NAME):
    node = h5file.get_node('/', DEFAULT_NODE_NAME)
    return getattr(node, columnName)


def initColumnsOnH5File(h5file, infoToInitArraysOnH5File, expectedRows, nodeName=DEFAULT_NODE_NAME, complib='blosc', complevel=5):
    """
        h5file: filehandle to the h5file, initialised with init_h5_file
        infoToInitArrayOnH5File: array of instances of InfoToInitArrayOnH5File
        expectedRows: this code is set up to work with EArrays, which can be extended after creation.
            (presumably, if your data is too big to fit in memory, you're going to have to use EArrays
            to write it in pieces). "sizeEstimate" is the estimated size of the final array; it
            is used by the compression algorithm and can have a significant impace on performance.
        nodeName: the name of the node being written to.
        complib: the docs seem to recommend blosc for compression...
        complevel: compression level. Not really sure how much of a difference this number makes...
    """
    gcolumns = h5file.get_node(h5file.root, nodeName);
    filters = tables.Filters(complib=complib, complevel=complevel);
    for infoToInitArrayOnH5File in infoToInitArraysOnH5File:
        finalShape = [0]; #in an eArray, the extendable dimension is set to have len 0
        finalShape.extend(infoToInitArrayOnH5File.shape);
        h5file.create_earray(gcolumns, infoToInitArrayOnH5File.name, atom=infoToInitArrayOnH5File.atomicType
                            , shape=finalShape, title=infoToInitArrayOnH5File.name #idk what title does...
                            , filters=filters, expectedrows=expectedRows);
    

if __name__ == "__main__":

    pdb_list_infile = sys.argv[1]
    input_dir = sys.argv[2]
    out_dir = sys.argv[3]
    label_file = sys.argv[4]
    outprefix = sys.argv[5]

    #input_dir = 'data/numpy'
  
    with open(pdb_list_infile,'r') as listin:
        files = [line.strip() for line in listin]
    
    # load labels
    y = np.load(label_file, allow_pickle=True)
    print(y.shape)
    
    ### debugging with smaller test set
    #randsamp = np.random.choice(range(len(files)),10)
    #files = [files[x] for x in randsamp]
    #y = y[randsamp,:]
    #print(y.shape)
    #################
    
    total_num = len(files)
    
    filename_out = format("%s/%s_data.h5" % (out_dir,outprefix))
    h5file = init_h5_file(filename_out)

    dataName = "data"
    dataShape = [4,20,20,20] 
    labelName = "label"
    labelShape = []
    dataInfo = InfoToInitArrayOnH5File(dataName, dataShape, tables.Float32Atom())
    labelInfo = InfoToInitArrayOnH5File(labelName, labelShape, tables.Float32Atom())
    
    numSamples = total_num*5
    initColumnsOnH5File(h5file, [dataInfo,labelInfo], numSamples)
    dataColumn = getH5column(h5file, dataName)
    labelColumn = getH5column(h5file, labelName)

    print('Writing # examples: ', total_num)
    labels = []
    for idx, fname in enumerate(files):
        fname = fname.split('/')[-1].replace('.pdb','_0.dat')
        fname = os.path.join(input_dir, fname)
        X=np.load(fname, allow_pickle=True)
        print('Now writing..',fname, X.shape,'Num labels:',len(y[idx]))
        writeToDisk(dataColumn, X)
        labels.append(y[idx])   ###, :X.shape[0]])   ## new format of labels has lists of values for each protein
        
    labels = np.concatenate(labels)
    print(labels.shape)
    writeToDisk(labelColumn, labels)

    h5file.close()
