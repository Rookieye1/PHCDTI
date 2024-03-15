import pandas as pd
import numpy as np


# with open('E:/dataset/药物与蛋白质/KIBA.txt','r') as f:
#     datas = f.readlines()
#


import pandas as pd

# 读取txt数据
data = pd.read_csv('D:/临时代码空间/DrugBank.txt', delimiter='\t')

# 将数据保存为csv文件
data.to_csv('D:/临时代码空间/DrugBannk.csv', index=False)