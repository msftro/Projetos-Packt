# %%
import pandas as pd

df = pd.read_csv('data/Chapter_1_cleaned_data.csv')
df.head()

# %%
pay_feats = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
df[pay_feats].describe()

# %%
df[pay_feats[0]].value_counts().sort_index()

# %%
df[pay_feats[0]].hist()

# %%
import numpy as np

pay_1_bins = np.array(range(-2, 10)) - 0.5
pay_1_bins

# %%
import matplotlib.pyplot as plt

df[pay_feats[0]].hist(bins=pay_1_bins)
plt.ylabel('PAY_1')
plt.xlabel('Number of accounts')

# %
import matplotlib as mpl

mpl.rcParams['font.size'] = 4
df[pay_feats].hist(bins=pay_1_bins, layout=(2,3))

# %%
df.loc[df['PAY_2'] == 2, ['PAY_2', 'PAY_3']].head()

# %%
bill_amt = ['BILL_AMT1', 
              'BILL_AMT2', 
              'BILL_AMT3', 
              'BILL_AMT4', 
              'BILL_AMT5', 
              'BILL_AMT6']

pay_amt = ['PAY_AMT1',
           'PAY_AMT2',
           'PAY_AMT3',
           'PAY_AMT4',
           'PAY_AMT5',
           'PAY_AMT6']

# %%
df[bill_amt].describe()

# %%
df[bill_amt].hist(bins=20, layout=(2,3))

# %%
df[pay_amt].describe()

# %%
df[pay_amt].hist(bins=20, layout=(2,3), xrot=45)

# %%
pay_zero_mask = df[pay_amt] == 0
pay_zero_mask.sum()

# %%
df[pay_amt][~pay_zero_mask].apply(np.log).hist(layout=(2,3))

