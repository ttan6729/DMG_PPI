import os
import re
from Bio import PDB
def processPDB(nameList,folder=None):
	parser = PDB.PDBParser(QUIET=True)
	for seqName in nameList:
		fName = f'{folder}/{seqName}.pdb'
		if not os.path.isfile(fName):
			print(f'{fName} not found')
			exit()
		read_atoms(fName)
		structure_data = parser.get_structure("protein", fName)
		count = 0
		for model in structure_data:
			for chain in model:
				for residue in chain:
					for atom in residue:
						if count <= 50:
							#print(f"Atom: {atom.get_name()}, Residue: {residue.get_resname()}, Position: {atom.get_coord()}")
							count += 1

	return


def read_atoms(fName, chain=".", model=1):
    pattern = re.compile(chain)

    current_model = model
    atoms = []
    ajs = []
    line_count = 0

    file = open(fName,'r')
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("ATOM"):
            type = line[12:16].strip()
            chain = line[21:22]
            line_count += 1
            if type == "CA" and re.match(pattern, chain):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ajs_id = line[17:20]
                atoms.append((x, y, z))
                ajs.append(ajs_id)
        # elif line.startswith("MODEL"):
        #     current_model = int(line[10:14].strip())
    # print(line_count)
    # print(len(atoms))
    file.close()
    return atoms, ajs