import numpy as np    
from pssmgen import PSSM
import os


def GenPSSM(fa_dir="/mnt/data6t/EC/PPIKG/multiSet/FastaDir", out_dir="/mnt/data6t/EC/PPIKG/multiSet/script/PSSM-Data"):
	gen = PSSM(work_dir='7CEI')
	gen.configure(blast_exe='/mnt/data6t/EC/PPIKG/ncbi-blast-2.13.0/bin/psiblast',
			database='/mnt/data6t/EC/PPIKG/multiSet/data/SHS-DB/SHS',
			num_threads = 4, evalue=0.0001, comp_based_stats='T',
			max_target_seqs=2000, num_iterations=3, outfmt=7,
			save_each_pssm=True, save_pssm_after_last_round=True)
	gen.get_pssm(fasta_dir=fa_dir,out_dir=out_dir, run= True )
	
	# map PSSM and PDB to get consisitent/mapped PSSM files
	#gen.map_pssm(pssm_dir='pssm_raw', pdb_dir='pdb', out_dir='pssm', chain=('A','B'))
	# write consistent/mapped PDB files and move inconsistent ones to another folder for backup
	#gen.get_mapped_pdb(pdbpssm_dir='pssm', pdb_dir='pdb', pdbnonmatch_dir='pdb_nonmatch')

def CalPSSM(prefix='PSSM-Data', fNum = 0):
	chaNum = 20
	result = []
	AA = 'ARNDCQEGHILKMFPSTWYV'
	for i in range(fNum):
		tmp = np.zeros(chaNum)
		fName = prefix+str(i+1)+'.pssm'
		if not os.path.isfile(fName):
			print(f'PSSM file {fName} is not found ,please read mannual and change mode')
			result.append(tmp)
			continue
			
		with open(fName) as f:
			lines = f.readlines()[3:-6]
			for line in lines:
				values = line.strip().split()[2:2+chaNum]
				values= np.array(values,dtype=float)
				maxValue = np.max(values)
				minValue = np.min(values)
				gap = maxValue - minValue
				if gap != 0:
					for i in range(chaNum):
						tmp[i] += -1 + 2*(values[i]-maxValue)/gap
			result.append(tmp)
	return np.array(result,dtype=float)			





#iLearn  return a list of multiple lines in PSSM files
# def PSSM(fastas, **kw):
#     if check_sequences.check_fasta_with_equal_length(fastas) == False:
#         print('Error: for "PSSM" encoding, the input fasta sequences should be with equal length. \n\n')
#         return 0

#     pssmDir = kw['path']
#     if pssmDir == None:
#         print('Error: please specify the directory of predicted protein disorder files by "--path" \n\n')
#         return 0

#     AA = 'ARNDCQEGHILKMFPSTWYV'

#     encodings = []
#     header = ['#', 'label']
#     for p in range(1, len(fastas[0][1]) + 1):
#         for aa in AA:
#             header.append('Pos.'+str(p) + '.' + aa)
#     encodings.append(header)

#     for i in fastas:
#         name, sequence, label = i[0], i[1], i[2]
#         code = [name, label]
#         if os.path.exists(pssmDir+'/'+name+'.pssm') == False:
#             print('Error: pssm profile for protein ' + name + ' does not exist.')
#             sys.exit(1)
#         with open(pssmDir+'/'+name+'.pssm') as f:
#             records = f.readlines()[3: -6]

#         proteinSeq = ''
#         pssmMatrix = []
#         for line in records:
#             array = line.strip().split()
#             pssmMatrix.append(array[2:22])
#             proteinSeq = proteinSeq + array[1]

#         pos = proteinSeq.find(sequence)
#         if pos == -1:
#             print('Warning: could not find the peptide in proteins.\n\n')
#         else:
#             for p in range(pos, pos + len(sequence)):
#                 code = code + pssmMatrix[p]
#         encodings.append(code)

#     return encodings

#DELPHI
# def load_fasta_and_compute(seq_fn, out_base_fn, raw_pssm_dir):
#     fin = open(seq_fn, "r")
#     # fout = open(out_fn, "w")
#     while True:
#         Pid = fin.readline().rstrip("\n")[1:]
#         line_Pseq = fin.readline().rstrip("\n")
#         if not line_Pseq:
#             break
#         pssm_fn = raw_pssm_dir + "/" + Pid + ".fasta.pssm"
#         LoadPSSMandPrintFeature(pssm_fn, out_base_fn, Pid, line_Pseq)
#         # fout.write(line_Pid)
#         # fout.write(line_Pseq)

#         # fout.write(",".join(map(str,Feature)) + "\n")
#     fin.close()
#     # fout.close()


# def extract_lines(pssmFile):
#     fin = open(pssmFile)
#     pssmLines = []
#     if fin == None:
#         return
#     for i in range(3):
#         fin.readline()  # exclude the first three lines
#     while True:
#         psspLine = fin.readline()
#         if psspLine.strip() == '' or psspLine.strip() == None:
#             break
#         pssmLines.append(psspLine)
#     fin.close()
#     return pssmLines



# def LoadPSSMandPrintFeature(pssm_fn, out_base_fn, Pid, line_Pseq):
#     print(Pid)
#     global min_value, max_value
#     fin = open(pssm_fn, "r")
#     pssmLines=extract_lines(pssm_fn)
#     # print(pssmLines)
#     seq_len = len(pssmLines)
#     if (seq_len == len(line_Pseq)):
#         # exit(1)
#         pssm_np_2D = np.zeros(shape=(20, seq_len))
#         for i in range(seq_len):

#             # fist 69 chars are what we need
#             # print(pssmLines[i][9:70])
#             #
#             values_20 = pssmLines[i].split()[2:22]
#             # print(values_20)
#             for aa_index in range(20):
#                 pssm_np_2D[aa_index][i] = (float(values_20[aa_index]) - min_value)/(max_value - min_value)
#                 # max_value = max(max_value, float(values_20[aa_index]))
#                 # min_value = min(min_value, float(values_20[aa_index]))
#         fin.close()

#         # print to feature file
#         for i in range(1, 21):
#             out_fn = out_base_fn + str(i) + ".txt"
#             fout = open(out_fn, "a+")
#             fout.write(">" + Pid + "\n")
#             fout.write(line_Pseq + "\n")
#             fout.write(",".join(map(str,pssm_np_2D[i-1])) + "\n")
#             fout.close()
#     else:
#         print("length doesn't match for protein ", Pid)
#         print("PSSM file has ", seq_len," lines, but sequence length is ", len(line_Pseq))