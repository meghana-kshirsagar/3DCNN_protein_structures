import sys
import numpy
import scipy.ndimage
import json
import os
from scipy import spatial
from Bio.PDB import PDBList


num_of_channels=4
x_dim=20
y_dim=20
z_dim=20
num_3d_pixel=20
atom_density=0

bias=[-0.558,-0.73,1.226]
resiName_to_label={'ILE': 12, 'GLN': 8, 'GLY': 18, 'GLU': 4, 'CYS': 19, 'HIS': 0, 'SER': 5, 'LYS': 1, 'PRO': 17, 'ASN': 7, 'VAL': 10, 'THR': 6, 'ASP': 3, 'TRP': 16, 'PHE': 14, 'ALA': 9, 'MET': 13, 'LEU': 11, 'ARG': 2, 'TYR': 15, 'CYM': 19}

label_atom_type_dict={
18:set(['N','CA','C','O']),
19:set(['N','CA','C','O','CB','SG']),
2:set(['N','CA','C','O','CB','CG','CD','NE','CZ','NH1','NH2']),
5:set(['N','CA','C','O','CB','OG']),
6:set(['N','CA','C','O','CB','OG1','CG2']),
1:set(['N','CA','C','O','CB','CG','CD','CE','NZ']),
13:set(['N','CA','C','O','CB','CG','SD','CE']),
9:set(['N','CA','C','O','CB']),
11:set(['N','CA','C','O','CB','CG','CD1','CD2']),
12:set(['N','CA','C','O','CB','CG1','CG2','CD1']),
10:set(['N','CA','C','O','CB','CG1','CG2']),
3:set(['N','CA','C','O','CB','CG','OD1','OD2']),
4:set(['N','CA','C','O','CB','CG','CD','OE1','OE2']),
0:set(['N','CA','C','O','CB','CG','ND1','CD2','CE1','NE2']),
7:set(['N','CA','C','O','CB','CG','OD1','ND2']),
17:set(['N','CA','C','O','CB','CG','CD']),
8:set(['N','CA','C','O','CB','CG','CD','OE1','NE2']),
14:set(['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ']),
16:set(['N','CA','C','O','CB','CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2']),
15:set(['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ','OH']),
}


class PDB_atom:
	def __init__(self,atom_type,res,chain_ID,x,y,z,index):
		self.atom = atom_type
		self.res = res
		self.chain_ID = chain_ID
		self.x = x
		self.y = y
		self.z = z
		self.index = index
	def __eq__(self, other): 
		return self.__dict__ == other.__dict__

def find_actual_pos(my_kd_tree,cor,PDB_entries):
	[d,i] = my_kd_tree.query(cor,k=1)
	return PDB_entries[i]

def get_position_dict(all_PDB_atoms):
	get_position={}
	for a in all_PDB_atoms:
		get_position[a.atom]=[a.x,a.y,a.z]
	return get_position

def center_and_transform(get_position):

	reference = get_position["CA"]
	axis_x = numpy.array(get_position["N"]) - numpy.array(get_position["CA"])  
	pseudo_axis_y = numpy.array(get_position["C"]) - numpy.array(get_position["CA"])  
	axis_z = numpy.cross(axis_x , pseudo_axis_y)
	if "CB" in get_position.keys():
		direction = numpy.array(get_position["CB"]) - numpy.array(get_position["CA"]) 
		axis_z *= numpy.sign( direction.dot(axis_z) ) 
	axis_y= numpy.cross(axis_z , axis_x)

	axis_x/=numpy.sqrt(sum(axis_x**2))
	axis_y/=numpy.sqrt(sum(axis_y**2))
	axis_z/=numpy.sqrt(sum(axis_z**2))

	
	transform=numpy.array([axis_x, axis_y, axis_z], 'float16').T
	return [reference,transform]

def PDB_is_in_list(PDB_entry,PDB_list):

	for element in PDB_list:
		if PDB_entry.x==element.x and PDB_entry.y==element.y and PDB_entry.z==element.z:
			#print("true"
			return True
	#print("false"
	return False

def grab_PDB(entry_list):

	ID_dict={}
	all_pos=[]
	all_lines=[]
	all_atom_type =[]
	PDB_entries = []
	atom_index = 0
	model_ID = 0
	MODELS = []

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

				l=[]
				a=PDB_atom(atom_type=atom,res=res,chain_ID=chain_ID,x=new_pos[0],y=new_pos[1],z=new_pos[2],index=atom_index)
				l.append(a)
				ID_dict[chain_ID]=l
			else:
				ID_dict[chain_ID].append(PDB_atom(atom,res,chain_ID,new_pos[0],new_pos[1],new_pos[2],index=atom_index))

			

			PDB_entries.append(PDB_atom(atom,res,chain_ID,new_pos[0],new_pos[1],new_pos[2],index=atom_index))
			atom_index+=1

		if line[0]=="ENDMDL":
			break

	
	MODEL=[ID_dict,all_pos,all_lines, all_atom_type, PDB_entries]
				
	return MODEL


def cut_box_site_test_pdb(ptf_ID,ptf_file_name,pdb_file_name, output_dir):
	print("Cutting local boxes around extracted positions in PDBs ...")
	pixel_size = 1
	std = 0.6
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	#if target_RES=='':
	#	ptf_ID = site_name
	#else:
	#	ptf_ID = target_RES +'_'+target_ATOM+'_'+site_name
	#extracted_ptf_file = open('data/ptf/'+ptf_ID+'_extracted.ptf','w')

	box_size=x_dim
	box_x_min=-box_size/2
	box_x_max=+box_size/2
	box_y_min=-box_size/2
	box_y_max=+box_size/2
	box_z_min=-box_size/2
	box_z_max=+box_size/2

	ptf_file=open(ptf_file_name)
	ptf_lines=list(ptf_file)


	pdb_group={}

	for line in ptf_lines:
		S=line.split()
		PDB_ID=S[0]
		x= float(S[1])
		y= float(S[2])
		z= float(S[3])
		if PDB_ID not in pdb_group:
			pdb_group[PDB_ID]=[]
		pdb_group[PDB_ID].append(line)


	dat_num=0
	extracted_box =[]
	for PDB_ID in pdb_group:
		print(PDB_ID)

		all_ptf_lines = pdb_group[PDB_ID]
		#pdb_file_name=pdb_dir+'/'+PDB_ID.lower()+'.pdb'
		pdb_file=open(pdb_file_name)
		l=list(pdb_file)
		MODEL=grab_PDB(l)

		[ID_dict,all_pos,all_lines,all_atom_type,PDB_entries]=MODEL

		
		all_pos = numpy.array(all_pos)
		my_kd_tree = scipy.spatial.KDTree(all_pos)

		for line in all_ptf_lines:
			S = line.split()
			PDB_ID = S[0]
			x= float(S[1])
			y= float(S[2])
			z= float(S[3])

			#pdb_file_name=pdb_dir+'/'+PDB_ID.lower()+'.pdb'
			pdb_file=open(pdb_file_name)
			l=list(pdb_file)
			MODEL=grab_PDB(l)

			[ID_dict,all_pos,all_lines,all_atom_type,PDB_entries]=MODEL
			all_pos = numpy.array(all_pos)
			my_kd_tree = scipy.spatial.KDTree(all_pos)
			
			actual_atom=find_actual_pos(my_kd_tree, [x,y,z], PDB_entries)

			chain_ID=actual_atom.chain_ID
			res_atoms=ID_dict[chain_ID]
			res=actual_atom.res
			if res in resiName_to_label.keys():
				label = resiName_to_label[res]
				get_position=get_position_dict(res_atoms)
				print(">>",res,label)
				if "CA" in set(get_position.keys()) and "C" in set(get_position.keys()) and "N" in set(get_position.keys()):

					[reference,transform]=center_and_transform(get_position)  
					transformed_pos = ((all_pos - reference).dot(transform))-bias
					x_index = numpy.intersect1d(numpy.where(transformed_pos[:,0]>box_x_min),numpy.where(transformed_pos[:,0]<box_x_max))
					y_index = numpy.intersect1d(numpy.where(transformed_pos[:,1]>box_y_min),numpy.where(transformed_pos[:,1]<box_y_max))
					z_index = numpy.intersect1d(numpy.where(transformed_pos[:,2]>box_z_min),numpy.where(transformed_pos[:,2]<box_z_max))

					final_index = numpy.intersect1d(x_index,y_index)
					final_index = numpy.intersect1d(final_index,z_index)
					final_index = final_index.tolist()
					final_index = [ ind for ind in final_index if (all_atom_type[ind] =='C' or all_atom_type[ind]=='O' or all_atom_type[ind]=='S' or all_atom_type[ind]=='N') ]
					box_ori = [PDB_entries[i] for i in final_index] 
					new_pos_in_box = transformed_pos[final_index]
					atom_count = len(box_ori)
			
					threshold=(box_size**3)*atom_density
					#print('>>',atom_count,threshold,box_size,atom_density)
					if atom_count > threshold:     #(box_size**3)*atom_density:
						samplega=numpy.zeros((num_of_channels,int(x_dim/pixel_size),int(y_dim/pixel_size),int(z_dim/pixel_size)))
						
						for i in range (0,len(box_ori)):
							atoms = box_ori[i]
							x=new_pos_in_box[i][0]
							y=new_pos_in_box[i][1]
							z=new_pos_in_box[i][2]

							x_new=x-box_x_min
							y_new=y-box_y_min
							z_new=z-box_z_min
							bin_x=int(numpy.floor(x_new/pixel_size))
							bin_y=int(numpy.floor(y_new/pixel_size))
							bin_z=int(numpy.floor(z_new/pixel_size))

							if(bin_x==num_3d_pixel):
								bin_x=num_3d_pixel-1
							if(bin_y==num_3d_pixel): 
								bin_y=num_3d_pixel-1
							if(bin_z==num_3d_pixel):
								bin_z=num_3d_pixel-1  
							
							#print('>',bin_x,bin_y,bin_z)
							if atoms.atom[0]=='O':
								samplega[0,bin_x,bin_y,bin_z] = samplega[0,bin_x,bin_y,bin_z] + 1
							elif atoms.atom[0]=='C':
								samplega[1,bin_x,bin_y,bin_z] = samplega[1,bin_x,bin_y,bin_z] + 1
							elif atoms.atom[0]=='N':
								samplega[2,bin_x,bin_y,bin_z] = samplega[2,bin_x,bin_y,bin_z] + 1
							elif atoms.atom[0]=='S':
								samplega[3,bin_x,bin_y,bin_z] = samplega[3,bin_x,bin_y,bin_z] + 1
                            ## consider adding hydrogens?


						X_smooth=numpy.zeros(samplega.shape, dtype='float32')

						for j in range (0,num_of_channels):
							X_smooth[j,:,:,:]=scipy.ndimage.filters.gaussian_filter(samplega[j,:,:,:], sigma=std, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
						extracted_box.append(X_smooth)
						#extracted_ptf_file.write(line)
							
						if(len(extracted_box)==1000):
							sample_time_t = numpy.array(extracted_box)
							sample_time_t.dump(output_dir+'/'+ptf_ID+'_'+str(dat_num)+'.dat')
							print("dumping extracted boxes " +ptf_ID+'_'+str(dat_num)+'.dat'+" in "+output_dir)
							extracted_box=[]
							dat_num+=1
						

	sample_time_t = numpy.array(extracted_box)
	sample_time_t.dump(output_dir+'/'+ptf_ID+'_'+str(dat_num)+'.dat')  
	print("dumping extracted boxes " +ptf_ID+'_'+str(dat_num)+'.dat'+" in "+output_dir)
	ptf_file.close()
	#extracted_ptf_file.close()
	print("finished extracting boxes")
	#return ptf_ID
	
		


if __name__ == "__main__":
	pdb_fn = sys.argv[1]
	numpy_outdir = sys.argv[2]
	ptf_fn = pdb_fn.replace('pdb','ptf')
	ptf_id = (pdb_fn.split('/')[-1]).split('.')[0]
	out_file = numpy_outdir+'/'+ptf_id+'_0.dat'
	print('Converting!!')
	cut_box_site_test_pdb(ptf_id,ptf_fn, pdb_fn, numpy_outdir)

