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
