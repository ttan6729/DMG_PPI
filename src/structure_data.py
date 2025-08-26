import pdb2pqr
import numpy as np
from pdb2pqr.main import main_driver,drop_water,setup_molecule,non_trivial, print_pqr, print_pdb
import pdb2pqr.io
import argparse
import propka.lib
import freesasa
import Bio
import io
#from Bio.PDB import * 
from Bio.PDB import PDBParser
import subprocess
from colorama import Fore, Back, Style
from termcolor import colored, cprint


HBOND_DIST = 3.5  
HBOND_ANGLE = 120  
SALT_BRIDGE_DIST = 4.0 
HYDROPHOBIC_DIST = 4.5  
PI_STACK_DIST = 5.5   


donors = {"ARG": ["NE", "NH1", "NH2"], "ASN": ["ND2"], "GLN": ["NE2"],
		  "HIS": ["ND1", "NE2"], "LYS": ["NZ"], "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"]}
acceptors = {"ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"], "ASN": ["OD1"],
			 "GLN": ["OE1"], "HIS": ["ND1", "NE2"], "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"]}
positives = {"LYS": ["NZ"], "ARG": ["NH1", "NH2"]}
negatives = {"ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"]}
hydrophobics = {"ALA","VAL","LEU","ILE","MET","PHE","TRP","PRO"}
aromatics = {"PHE","TYR","TRP","HIS"}

def vector(a, b):
	return np.array(b.coord) - np.array(a.coord)

def angle(v1, v2):
	cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
	return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def detect_interactions(res1, res2):
	interactions = []

	atoms1 = {atom.get_name(): atom for atom in res1}
	atoms2 = {atom.get_name(): atom for atom in res2}

	resname1 = res1.get_resname()
	resname2 = res2.get_resname()


	for dres, datoms in donors.items():
		if resname1 == dres:
			for donor in datoms:
				if donor in atoms1:
					for ares, aatoms in acceptors.items():
						if resname2 == ares:
							for acc in aatoms:
								if acc in atoms2:
									d = atoms1[donor]
									a = atoms2[acc]
									dist = d - a
									if dist <= HBOND_DIST:
										return True

	for pres, patoms in positives.items():
		if resname1 == pres:
			for p in patoms:
				if p in atoms1:
					for nres, natoms in negatives.items():
						if resname2 == nres:
							for n in natoms:
								if n in atoms2:
									dist = atoms1[p] - atoms2[n]
									if dist <= SALT_BRIDGE_DIST:
										return True

	if resname1 in hydrophobics and resname2 in hydrophobics:
		for a1 in atoms1.values():
			for a2 in atoms2.values():
				dist = a1 - a2
				if dist <= HYDROPHOBIC_DIST:
					interactions.append(("hydrophobic", dist))
					break

	if resname1 in aromatics and resname2 in aromatics:
		for a1 in atoms1.values():
			for a2 in atoms2.values():
				dist = a1 - a2
				if dist <= PI_STACK_DIST:
					return True

	return False


def build_pdb2pqr_parser():
	pars = argparse.ArgumentParser(prog="pdb2pqr_prog",description='desc',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,conflict_handler="resolve",)
	pars.add_argument("--output_pqr", default='test.pqr',help="Output PQR path")
	pars.add_argument("--pdb_output", default='test.pdb',help="Output PQR path")
	pars.add_argument("--log-level",default="DEBUG",
		choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],)
	pars.add_argument("--assign_only", default=False) #?
	pars.add_argument("--clean", default=False)
	pars.add_argument("--userff", default=None)
	pars.add_argument("--ff",
		choices=[ff for ff in ["AMBER", "CHARMM", "PARSE", "TYL06", "PEOEPB", "SWANSON"]],
		default="AMBER",help="The forcefield to use.",) #parse is for high accuracy calculation
	pars.add_argument("--ffout", default=None)
	pars.add_argument("--usernames", default=False)
	pars.add_argument("--input_path",default=None,)
	pars.add_argument("--drop_water", default=False)
	pars.add_argument("--ligand", default=None)
	pars.add_argument("--neutraln", default=None)
	pars.add_argument("--neutralc", default=None)
	pars.add_argument("--debump", default=True)
	pars.add_argument("--pka_method", default='propka')
	pars.add_argument("--keep_chain", default=True) #?
	pars.add_argument("--ph", type=float, default=0.7)
	pars.add_argument("--opt", default=True)
	pars.add_argument("--include-header", default=True)
	pars.add_argument("--whitespace", default=False)
	pars.add_argument("--add-charge", default=True)
	pars = propka.lib.build_parser(pars)
	return pars

#add hydrogen to pdb structure using pdb2pqr package
def add_hydrogen(input_path,output='tmp'):
	parser = build_pdb2pqr_parser()
	pdb2pqr_args = parser.parse_args('')  # get default arguments
	pdb2pqr_args.input_path = input_path

	pdb2pqr_args.output_pqr = output + '.pqr'
	pdb2pqr_args.pdb_output = output + '.pdb'

	definition = pdb2pqr.io.get_definitions()
	pdblist, is_cif = pdb2pqr.io.get_molecule(pdb2pqr_args.input_path)
	if pdb2pqr_args.drop_water:
	   pdblist = drop_water(pdblist)
	biomolecule, definition, ligand = setup_molecule(pdblist, definition, pdb2pqr_args.ligand)		
	biomolecule.set_termini(pdb2pqr_args.neutraln, pdb2pqr_args.neutralc)
	biomolecule.update_bonds()
	if pdb2pqr_args.clean:
		results = {"header": "","missed_residues": None,"biomolecule": biomolecule,
			"lines": io.print_biomolecule_atoms(
				biomolecule.atoms, pdb2pqr_args.keep_chain),"pka_df": None,}
	else:
		results = non_trivial(args=pdb2pqr_args,biomolecule=biomolecule,ligand=ligand,
			definition=definition,is_cif=is_cif,)

	print_pqr(args=pdb2pqr_args,pqr_lines=results["lines"],header_lines=results["header"],
		missing_lines=results["missed_residues"],is_cif=is_cif,)
	if pdb2pqr_args.pdb_output:
		print_pdb(args=pdb2pqr_args,pdb_lines=pdb2pqr.io.print_biomolecule_atoms(
				biomolecule.atoms, chainflag=pdb2pqr_args.keep_chain, pdbfile=True),
			header_lines=results["header"],missing_lines=results["missed_residues"],is_cif=is_cif,)
	# pdblines = pdb2pqr.io.print_biomolecule_atoms(biomolecule.atoms, chainflag=pdb2pqr_args.keep_chain, pdbfile=True)
	# return io.StringIO(''.join(pdblines))
	return pdb2pqr_args.pdb_output,pdb2pqr_args.output_pqr

def count_residue(pdb_path):
	parser = Bio.PDB.PDBParser()
	structure = parser.get_structure('test', pdb_path) #pdb_path[pdb_path.rindex('/')+1:]
	model = structure[0]
	res_no = 0
	non_resi = 0
	for model in structure:
		for chain in model:
			for r in chain.get_residues():
				if r.id[0] == ' ':
					res_no +=1
				else:
					non_resi +=1
	return res_no



#cal the Solvent Accessible Surface Area using freesasa
def cal_sasa(pdb_path,pdb_structure):
	freesasa_structure = freesasa.Structure(pdb_path)
	areas = freesasa.calc(freesasa_structure).residueAreas()
	result = []
	for model in pdb_structure:
		for chain in model:
			chain_id = chain.get_id()
			for residue in chain:
				residue_id = residue.get_id()[1]  # Residue ID (sequence number)
				area = areas[chain_id][str(residue_id)]
				#print(area.__dir__()) #'residueType', 'residueNumber'
				result.append([area.total,area.mainChain,area.sideChain,area.polar,area.apolar,area.relativeTotal,area.relativeMainChain,area.relativeSideChain,area.relativePolar,area.relativeApolar])
	return np.array(result)

def cal_distance(atom1, atom2):
	return np.linalg.norm(atom1.coord - atom2.coord)

# cal the angle between three atoms
def cal_angle(atom1, atom2, atom3):
	vector1 = atom1.coord - atom2.coord
	vector2 = atom3.coord - atom2.coord
	cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
	angle = np.arccos(cos_theta) * 180.0 / np.pi
	return angle

def is_hbond(donor, acceptor, distance_cutoff=3.5, angle_cutoff=120):
	"""
	Check if a hydrogen bond exists between a donor and acceptor.
	Args:
		donor: Atom object (e.g., N or O with attached H).
		acceptor: Atom object (e.g., O or N).
		distance_cutoff: Maximum distance between donor and acceptor (Å).
		angle_cutoff: Minimum donor-hydrogen-acceptor angle (degrees).
	Returns:
		True if H-bond criteria are met, False otherwise.
	"""
	# Distance check
	distance = donor - acceptor
	if distance > distance_cutoff:
		return False
	
	# Angle check (if hydrogen is present)
	if "H" in donor.parent:
		hydrogen = donor.parent["H"]
		donor_pos = donor.get_coord()
		hydrogen_pos = hydrogen.get_coord()
		acceptor_pos = acceptor.get_coord()
		
		# Compute the angle
		vector1 = donor_pos - hydrogen_pos
		vector2 = acceptor_pos - hydrogen_pos
		angle = np.degrees(np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))))
		if angle > angle_cutoff:
			return True
	
	return False


#compute hydrogenbond within each resiude
def compute_intra_hydrogenbond(structure):
	result = []
	for model in structure:
		for chain in model:
			for residue in chain:
				donors = []
				acceptors = []
				count = 0
				for atom in residue:
					if atom.element == 'H':  # If it's a hydrogen atom
						# Check if it's part of a N-H or O-H group (donor)
						if atom.name.startswith('H') :#and (atom.get_parent().get_name() in ['N', 'O'])
							donors.append(atom)
					# Look for hydrogen bond acceptors (O, N, S)
					if atom.element in ['O', 'N', 'S']:  # Acceptors
						acceptors.append(atom)
				for donor in donors:
					for acceptor in acceptors:               
						if donor != acceptor and is_hbond(donor,acceptor):
							count+=1
						# distance = cal_distance(donor, acceptor)
						# angle = cal_angle(donor, acceptor, residue["N"])
						# if distance < 3.5 and angle > 120:
						# 	count += 1
				result.append(count)

	return np.array(result).reshape(-1,1)



import itertools
def compute_chemical_graph(structure):
	connections = []
	residues = []
	for model in structure:
		for chain in model:
			for residue in chain:
				residues.append(residue)

	for r1, r2 in itertools.combinations(residues, 2):

		if detect_interactions(r1, r2):
			id1 = r1.id[1]
			id2 = r2.id[1]
			connections.append([id1,id2])


	return connections


def compute_inter_hydrogenbond(structure):
	hydrogen_bonds = []
	donors, acceptors = [], []
	for model in structure:
		for chain in model:
			for residue in chain:
				rid = residue.id[1] - 1 
				if residue.id[0] == " ":  # Skip heteroatoms (e.g., water, ions)
					# Look for hydrogen bond donors (N-H, O-H)
					for atom in residue:
						#if atom.element == 'H':  # If it's a hydrogen atom
							# Check if it's part of a N-H or O-H group (donor)
							#if atom.name.startswith('H') :#and (atom.get_parent().get_name() in ['N', 'O'])
						if atom.get_name() in ["N", "O", "OH", "NH", "NH2"]:
							donors.append([atom,rid])

						# Look for hydrogen bond acceptors (O, N, S)
						if atom.get_name() in ["O", "N", "OH", "NH", "NH2"]:  # Acceptors
							acceptors.append([atom,rid])

	for donor in donors:
		for acceptor in acceptors:
			if donor[1] != acceptor[1]:
				atom1, atom2 = donor[0], acceptor[0]
				if is_hbond(atom1,atom2):
					hydrogen_bonds.append([donor[1],acceptor[1]])


	return hydrogen_bonds



def load_dx(file_path):
	origin,deltas,counts,data_start,data = None,[],None,None,None	
	with open(file_path, 'r') as f:
		lines = f.readlines()
		for i,line in enumerate(lines):
			# Skip comment lines
			if line.startswith('#'):
				continue
			parts = line.strip().split()
			if not parts:
				continue
			if parts[0] == 'origin':
				origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
			elif parts[0] == 'delta':
				delta = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
				deltas.append(delta)
			# Extract grid dimensions (counts)
			elif parts[0] == 'object' and parts[1] == '2': #and 'gridconnections' 
				for j, part in enumerate(parts):
					if part == 'counts':
						counts = [int(parts[j+1]), int(parts[j+2]), int(parts[j+3])]
						break
			# if origin is not None and len(deltas) == 3 and counts is not None:
			# 	break
			if "object 3" in line:
				data_start = i + 1
				break

		data = np.array([float(x) for line in lines[data_start:-5] for x in line.split()])
		data = data.reshape(counts[0],counts[1],counts[2])
	if len(deltas) == 3:
		# For axis-aligned grids, spacing is the magnitude of each delta vector
		spacing = np.array([np.linalg.norm(delta) for delta in deltas])
	else:
		raise ValueError("Invalid DX file: Could not find 3 delta vectors.")
	
	return data, origin, spacing


def cal_flexibility(structure,residue_num):
	# Dictionary to store B-factors (flexibility) for each residue
	residue_flexibility = []
	
	# Iterate over all models, chains, and residues in the structure
	for model in structure:
		for chain in model:
			for residue in chain:
				# Skip heteroatoms (e.g., water, ligands)
				if not residue.get_id()[0] == " ":
					continue
				b_factor_sum = 0
				atom_count = 0
				for atom in residue:
					b_factor_sum += atom.get_bfactor()
					atom_count += 1 
				if atom_count > 0:
					avg_b_factor = b_factor_sum / atom_count
					residue_flexibility.append(avg_b_factor)
					#residue_flexibility[residue.get_resname() + str(residue.get_id()[1])] = avg_b_factor
				else:
					residue_flexibility.append(0)

	residue_flexibility = np.array(residue_flexibility).reshape(residue_num,-1)
	return residue_flexibility


def get_residue_centroids(pqr_file):
	parser = PDBParser()
	structure = parser.get_structure("protein", pqr_file)
	centroids = {}
	for residue in structure.get_residues():
		res_id = residue.get_id()[1]
		atoms = [atom for atom in residue.get_atoms()]
		coords = np.array([atom.get_coord() for atom in atoms])
		centroid = np.mean(coords, axis=0)
		centroids[res_id] = centroid
	return centroids

def get_residue_potentials(centroids, esp_grid, origin, spacing):
	residue_esp = {}
	shape = esp_grid.shape
	for res_id, centroid in centroids.items():
		# Convert centroid to grid index
		grid_idx = ((centroid - origin) / spacing).astype(int)
		x, y, z = grid_idx
		if x < shape[0] and x >= 0 and y < shape[1] and y >=0 and z < shape[2] and z>=0:
			residue_esp[res_id] = esp_grid[x, y, z]
		else:
			residue_esp[res_id] = 0
	return residue_esp


import propka.run

def cal_pka_shifts(pdb_name,residue_num):
	pka_results = propka.run.single(pdb_name, write_pka=False)
	result = np.zeros((residue_num,3),dtype=float)

	tmp = []

	count = 0
	for group in pka_results.conformations['1A'].groups:
		if group.atom.residue_label.split()[1].isdigit():
			residue_id = int(group.atom.residue_label.split()[1])-1
		else:
			continue
		if group.pka_value is not None and group.intrinsic_pka is not None:
			pka_shift = group.pka_value - group.intrinsic_pka
			result[residue_id][0] = pka_shift
			result[residue_id][1] = group.pka_value
			result[residue_id][2] = group.intrinsic_pka
			#print(f'{residue_id} {group.pka_value} {group.intrinsic_pka}')

	return result


def cal_eps(prefix,residue_num):
	dx_path = f'{prefix}.pqr-PE0.dx'
	esp_grid , origin, spacing = load_dx(dx_path)
	centroids = get_residue_centroids(f'{prefix}.pqr')
	residue_esp = get_residue_potentials(centroids, esp_grid, origin, spacing)
	residue_num = len(centroids.keys())

	esp_vector = [ residue_esp[r] for r in range(1,residue_num+1)]
	esp_vector =  np.array(esp_vector).reshape(residue_num,-1)
	return esp_vector

# component 1: sasa Solvent-Accessible Surface Area，溶剂可及表面积 dim 10
# component 2 Electrostatic Potential，静电势 dim 1
# component 3 pka dim 3
# component 4 flexibility dim 1
def cal_properties(pdb_path='/home/user1/code/protein/data/structure_data/STRING/9606.ENSP00000000233.pdb',output_prefix=None):
	#cprint('Note, the preprocessing of protein structure requires pdb2pqr3.6.2 and apbs3.0','red')
	if output_prefix is None:
		output_prefix = pdb_path[0:pdb_path.rfind('.')]
	tmp_prefix = 'tmp'
	subprocess.run(f'pdb2pqr --ff=AMBER --with-ph=7.0 --apbs-input {tmp_prefix}.apbs --pdb-output {tmp_prefix}.pdb {pdb_path} {tmp_prefix}.pqr', shell=True)

	subprocess.run(f'apbs {tmp_prefix}.apbs', shell=True)

	residue_num = count_residue(pdb_path)
	residue_num2 = count_residue(f'{tmp_prefix}.pdb')

	print(f'{residue_num} {residue_num2}')
	#assert count_residue(f'{tmp_prefix}.pdb') == residue_num
	structure = PDBParser().get_structure('tmp', pdb_path)
	fMatrix = cal_sasa(pdb_path,structure)
	fMatrix = np.concatenate((fMatrix,cal_eps(tmp_prefix,residue_num)),axis=1)
	fMatrix = np.concatenate((fMatrix,cal_pka_shifts(pdb_path,residue_num)),axis=1)
	#fMatrix = np.concatenate((fMatrix,cal_flexibility(structure,residue_num)),axis=1)
	structure2 = PDBParser().get_structure('tmp2', f'{tmp_prefix}.pdb')
	#inter_hb = compute_inter_hydrogenbond(structure2) #list of connections
	inter_hb = compute_chemical_graph(structure2)

	return fMatrix,inter_hb

def transfer_charges(pdb_file, pqr_file, output_file='tmp2.pdb'):
	parser = Bio.PDB.PDBParser()
	structure = parser.get_structure("protein", pdb_file)
	# Load charges from PQR file
	charges = {}
	with open(pqr_file, "r") as f:
		for line in f:
			if line.startswith("ATOM"):
				atom_id = int(line[6:11])
				charge = float(line[55:62])
				charges[atom_id] = charge
	# Assign charges to PDB atoms
	for model in structure:
		for chain in model:
			for residue in chain:
				for atom in residue:
					if atom.serial_number in charges:
						atom.charge = charges[atom.serial_number]
	# Save the modified PDB file
	io = Bio.PDB.PDBIO()
	io.set_structure(structure)
	io.save(output_file)
	return output_file


