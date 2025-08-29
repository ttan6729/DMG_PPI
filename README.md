# DMG-PPI, dual channel framework for learning compelx interaction patterns and PPI prediction
This repository contains an official implementation of DMG_PPI 
----
## Environments
- python3.11
- torch2.1.2
- dgl 2.3.0+cuda121
- numpy1.26.4
----
### Dataset
To test the performance of DMG-PPI, download the processed dataset from https://drive.google.com/file/d/1gj00JePblQfEakdcol4nsvT1WZcm8vRh/view?usp=sharing and extract the ZIP file in the same directory as main.py.

### Usage
```
usage: PPIM [-h] [-m M] [-t T] [-i I] [-i1 I1] [-i2 I2] [-i3 I3] [-i4 I4] [-i5 I5] [-struct_path STRUCT_PATH] [-o O] [-e E] [-b B] [-ln LN] [-Loss LOSS] [-ff FF] 

options:
  -h, --help    show this help message and exit
  -t T          model name, default: DMG 
  -m M          mode, optinal value: read,bfs,dfs,rand,
  -i I			path of sequnce and relation file
  -i1 I1        sequence file (when -i is not provided)
  -i2 I2        relation file when -i is not provided)
  -i3 I3        file path of test set indices (for read mode)
  -e E          epochs
  -b B          batch size
  -o O 			output path
  -ln LN        number of dual-channel gnn layers
  -Loss LOSS    loss function

```
### Sample command for training and testing
```
python3 main.py -m bfs -t DMG -i 27K.txt -i4 features/27K  -ln 3 -e 100 -o ../result/test
python3 main.py -m bfs -t DMG -i1 data/SHS27k.seqs.tsv -i2 data/SHS27k.actions.txt -i4 features/27K  -ln 3 -e 100-o ../result/test
```

