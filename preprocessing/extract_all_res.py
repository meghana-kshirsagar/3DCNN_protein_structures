import math
import numpy
import os
import sys
#import theano
#import theano.tensor as T
import collections
import scipy
import gzip

X = ['H','K','R','D','E','S','T','N','Q','A','V','L','I','M','F','Y','W','P','G','C']
abrev={'HIS':'H','LYS':'K','ARG':'R','ASP':'D','GLU':'E','SER':'S','THR':'T','ASN':'N','GLN':'Q','ALA':'A','VAL':'V','LEU':'L','ILE':'I','MET':'M','PHE':'F','TYR':'Y','TRP': 'W','PRO': 'P','GLY': 'G', 'CYS': 'C', 'CYM': 'C'}

class PDB_atom:
	def __init__(self,atom_type,res,chain_ID,x,y,z,index,value):
		self.atom = atom_type
		self.res = res
		self.chain_ID = chain_ID
		self.x = x
		self.y = y
		self.z = z
		self.index = index
		self.value = value
	def __eq__(self, other): 
		return self.__dict__ == other.__dict__

def get_position_dict(all_PDB_atoms):
	get_position={}
	for a in all_PDB_atoms:
		get_position[a.atom]=[a.x,a.y,a.z]
	return get_position



def get_target_res(Query,chain_IDs, ID_dict):
	target_ATOM = 'CA'
	pos = []
	for i in range (0,len(Query)):
		TARGET_RES = Query[i]
		TARGET_CHAIN = chain_IDs[i]
		res_atoms = ID_dict[TARGET_CHAIN]
		get_position=get_position_dict(res_atoms) 
		if target_ATOM in get_position.keys():
			pos.append((TARGET_RES,TARGET_CHAIN,get_position[target_ATOM]))
	return pos
	
def grab_PDB(entry_list):
	ID_dict={}
	all_pos=[]
	all_lines=[]
	all_atom_type =[]
	PDB_entries = []
	atom_index = 0
	model_ID = 0
	MODELS = []
	SEQ= []
	chain_IDs=[]

	for line1 in entry_list:
		line=line1.split()
		
		if line[0]=="ATOM":
			atom=(line1[13:16].strip(' '))
			res=(line1[17:20])
			chain_ID=line1[21:26]
			chain=chain_ID[0]
			res_no=chain_ID[1:].strip(' ')
			res_no=int(res_no)
			chain_ID=(chain,res_no)
			new_pos=[float(line1[30:37]),float(line1[38:45]),float(line1[46:53])]
			all_pos.append(new_pos)
			all_lines.append(line1)
			all_atom_type.append(atom[0])
			if chain_ID not in ID_dict.keys():
				if res in abrev.keys():
					SEQ.append(abrev[res])
					chain_IDs.append(chain_ID)

				l=[]
				a=PDB_atom(atom_type=atom,res=res,chain_ID=chain_ID,x=new_pos[0],y=new_pos[1],z=new_pos[2],index=atom_index,value=1)
				l.append(a)
				ID_dict[chain_ID]=l
			else:
				ID_dict[chain_ID].append(PDB_atom(atom,res,chain_ID,new_pos[0],new_pos[1],new_pos[2],index=atom_index,value=1))

			PDB_entries.append(PDB_atom(atom,res,chain_ID,new_pos[0],new_pos[1],new_pos[2],index=atom_index,value=1))
			atom_index+=1

		if line[0]=="ENDMDL":
			break


	MODEL=[ID_dict, SEQ, chain_IDs]
	return MODEL

#def extract_all_res(pdb_list, pdb_dir, site):
def extract_all_res(pdb_list, ptf_dir):

	#pdb_list = open(pdb_list)
	
	for pdb_file_name in pdb_list:
		PDB_ID=(pdb_file_name.split('/')[-1]).split('.')[0]
		PDB_ID=PDB_ID.strip()
		ptf_name = os.path.join(ptf_dir,PDB_ID+'.ptf')
		ptf_file = open(ptf_name,'w')
		print(PDB_ID,pdb_file_name, ptf_name)

		if os.path.isfile(pdb_file_name):
			pdb_file=open(pdb_file_name)
			l=list(pdb_file)
			MODEL=grab_PDB(l)
			[ID_dict, SEQ, chain_IDs]=MODEL
			Query=SEQ
			print("".join(Query),len(Query))
			pos=get_target_res(Query,chain_IDs,ID_dict)
			for p_ in pos:
				TARGET_RES,TARGET_CHAIN,p=p_
				x=p[0]
				y=p[1]
				z=p[2]
				#print(">>",PDB_ID+'\t'+str(x)+'\t'+str(y)+'\t'+str(z)+'\n')
				ptf_file.write(PDB_ID+'\t'+str(x)+'\t'+str(y)+'\t'+str(z)+'\t'+TARGET_RES+'\t'+TARGET_CHAIN[0]+'\t'+str(TARGET_CHAIN[1])+'\n')

		else:
			print("PDB not found!")
			print(PDB_ID)
			
		ptf_file.close()

	return ptf_name
	

if __name__ == "__main__":
	pdb_filename = sys.argv[1]
	ptf_dir = sys.argv[2]
	extract_all_res([pdb_filename], ptf_dir)
