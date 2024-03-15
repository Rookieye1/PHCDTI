# PHCDTI
Implementation of some SOTA CTR algorithm based on FuxiCTR open source framework

The raw data is too large to upload, we recommend users to customize their own dataset in the ```dataset.yaml```
## Environment
```
pip install -r requirements.txt
```
## Usage 
All experiments were conducted on the open source framework Huawei FuxiCTR.

The paper : https://dl.acm.org/doi/epdf/10.1145/3459637.3482486

Code : https://github.com/xue-pai/FuxiCTR

For training and Evaluating the model :
```
python main_experiments.py
```
## Resources:
+ README.md: this file.
+ data: The datasets used in paper.
	+ DrugBank.csv: 
	+ KIBA.csv: 
	+ Davis.csv
	In the directory of data, we now have the original data "DrugBank/KIBA/Davis.csv" as follows:

	```
	Drug_ID Protein_ID Drug_SMILES Amino_acid_sequence interaction
	DB00303 P45059 [H][C@]12[C@@H]... MVKFNSSRKSGKSKKTIRKLT... 1
	DB00114 P19113 CC1=NC=C(COP(O)... MMEPEEYRERGREMVDYICQY... 1
	DB00117 P19113 N[C@@H](CC1=CNC... MMEPEEYRERGREMVDYICQY... 1
	...
	...
	...
	DB00441 P48050 NC1=NC(=O)N(C=C... MHGHSRNGQAHVPRRKRRNRF... 0
	DB08532 O00341 FC1=CC=CC=C1C1=... MVPHAILARGRDVCRRNGLLI... 0

	```



## Cite
All experiments were conducted on the open source framework Huawei FuxiCTR, for continuting development, please cite the following papers :
```
@inproceedings{2021Open,
  title={Open Benchmarking for Click-Through Rate Prediction.},
  author={ Zhu, J.  and  Liu, J.  and  Yang, S.  and  Zhang, Q.  and  He, X. },
  booktitle={Conference on Information and Knowledge Management},
  year={2021},
}
```
# Run:

python HpyerAttentionDTI_main.py
