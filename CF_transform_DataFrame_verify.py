import pandas as pd
import numpy as np
pf = pd.read_excel('Dictionary2.xlsx', sheet_name='sheet1',index_col='ID')
pf_dict = pf.to_dict()
pf_change = pf_dict['名字']
pf_2 = pd.read_csv('data_decomposition1_SUM.csv', index_col=0, encoding='UTF-8')
SUM = 0
ret = list(pf_2.columns)

for i in range(len(pf_2.columns)):
    for j in pf_change.keys():
        a = pf_2.columns[i]
        b = pf_change[j]
        if pf_2.columns[i] == j:
            ret[i] = pf_change[j]
            SUM = SUM + 1
pf_2.columns = ret
print(pf_2.columns)
print(pf_change.keys())
print(SUM)
pf_2.to_csv('data_decomposition1_SUM2.csv', encoding='utf8')