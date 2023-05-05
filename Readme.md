## Code Written by Nakul Gopalan and Dilip 
## Edited by Ifrah Idrees 

# Installation 
Install the environment using environment.yml
`conda env create -f environment.yml`
# Running Code 
Run 

`python torch_seq2seq.py --src_path ../../data/hard_pc_src_syn.txt --tar_path ../../data/hard_pc_tar_syn.txt`

# Data used for evaluation in lggltl paper:
Clean Up Domain:
1. Expanded: '../../data/hard_pc_src_syn.txt', '../../data/hard_pc_tar_syn.txt'
2. Original: '../../data/hard_pc_src.txt', '../../data/hard_pc_tar.txt' 

Manipulation Domain:
1. Original: '../../data/hard_pc_src2.txt', '../../data/hard_pc_tar2.txt'
2. Expanded: '../../data/hard_pc_src_syn2.txt', '../../data/hard_pc_tar_syn2.txt'


