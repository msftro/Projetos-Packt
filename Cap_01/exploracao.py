# %%
import pandas as pd

# Carrega os dados do arquivo Excel para um DataFrame
df = pd.read_excel('../Cap_01/default_of_credit_card_clients__courseware_version_1_21_19.xls')

# %%
# Retorna o número de colunas do DataFrame
df.shape[1]

# %%
# Retorna o número de linhas do DataFrame
df.shape[0]

# %%
# Exibe os nomes das colunas do DataFrame
df.columns

# %%
# Exibe as primeiras 5 linhas do DataFrame
df.head()

# %%
# Verifica o número de IDs únicos na coluna 'ID'
df['ID'].nunique()

# %%
# Conta a frequência de cada ID no DataFrame e exibe os 5 primeiros
id_counts = df['ID'].value_counts()
id_counts.head()

# %%
# Conta a frequência das contagens de IDs (ex.: quantos IDs aparecem uma vez, quantos aparecem duas vezes, etc.)
id_counts.value_counts()

# %%
# Cria uma máscara para identificar IDs que aparecem exatamente duas vezes
dupe_mask = id_counts == 2
dupe_mask.head()  # Exibe as primeiras 5 linhas da máscara

# %%
# Cria uma lista de IDs que aparecem exatamente duas vezes
dupe_ids = list(id_counts.index[dupe_mask])
dupe_ids[:5]  # Exibe os primeiros 5 IDs duplicados

# %%
# Seleciona todas as linhas do DataFrame para os 3 primeiros IDs duplicados encontrados
df.loc[df['ID'].isin(dupe_ids[0:3]), :].head(10)

# %%
# Cria uma máscara que verifica se os valores em cada célula são zero
df_zero_mask = df == 0
df_zero_mask  # Exibe a máscara

# %%
# Remove a primeira coluna ('ID') e verifica se todas as outras colunas em uma linha são zero
df_zero_mask = df_zero_mask.iloc[:, 1:].all(axis=1)
sum(df_zero_mask)  # Conta o número de linhas onde todas as colunas (exceto 'ID') são zero

# %%
# Cria uma cópia do DataFrame original excluindo as linhas onde todas as colunas (exceto 'ID') são zero
df_clean_1 = df[~df_zero_mask].copy()
df_clean_1.shape  # Mostra as dimensões do DataFrame limpo

# %%
# Verifica o número de IDs únicos no DataFrame limpo
df_clean_1['ID'].nunique()

# %%
# Exibe informações sobre o DataFrame limpo, incluindo o tipo de dados de cada coluna
df_clean_1.info()

# %%
# Exibe as primeiras 5 linhas da coluna 'PAY_1'
df_clean_1['PAY_1'].head()

# %%
# Conta os valores distintos na coluna 'PAY_1'
df_clean_1['PAY_1'].value_counts()

# %%
# Cria uma máscara para identificar linhas onde a coluna 'PAY_1' não contém o valor 'Not available'
valid_pay_1_mask = df_clean_1['PAY_1'] != 'Not available'
valid_pay_1_mask[:5]  # Exibe as primeiras 5 linhas da máscara

# %%
# Conta o número de linhas onde a coluna 'PAY_1' não contém 'Not available'
sum(valid_pay_1_mask)

# %%
# Cria uma cópia do DataFrame limpo excluindo as linhas onde 'PAY_1' contém 'Not available'
df_clean_2 = df_clean_1.loc[valid_pay_1_mask, :].copy()
df_clean_2.shape  # Mostra as dimensões do DataFrame final limpo

# %%
df_clean_2['PAY_1'].value_counts()

# %%
df_clean_2['PAY_1'] = df_clean_2['PAY_1'].astype('int64')

# %%
df_clean_2[['PAY_1', 'PAY_2']].info()

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl

# %%
mpl.rcParams['figure.dpi'] = 400

df_clean_2[['LIMIT_BAL', 'AGE']].hist()

# %%
df_clean_2[['LIMIT_BAL', 'AGE']].describe()

# %%
df_clean_2['EDUCATION'].value_counts()

# %%
df_clean_2['EDUCATION'] = df_clean_2['EDUCATION'].replace(to_replace=[0, 5, 6], value=4)
df_clean_2['EDUCATION'].value_counts()

# %%
df_clean_2['MARRIAGE'].value_counts()

# %%
df_clean_2['MARRIAGE'] = df_clean_2['MARRIAGE'].replace(0, 3)
df_clean_2['MARRIAGE'].value_counts()

# %%
df_clean_2.groupby('EDUCATION').agg({'default payment next month':'mean'}).plot.bar(legend=False)
plt.ylabel('Default rate')
plt.xlabel('Education level: orginal encoding')

# %%
df_clean_2['EDUCATION_CAT'] = 'none'
df_clean_2[['EDUCATION', 'EDUCATION_CAT']].head()

# %%
cat_mapping = {
    1: 'graduate school',
    2: 'university',
    3: 'high school',
    4: 'others'
}

df_clean_2['EDUCATION_CAT'] = df_clean_2['EDUCATION'].map(cat_mapping).astype('category')
df_clean_2[['EDUCATION', 'EDUCATION_CAT']].head(10)

# %%
edu_ohe = pd.get_dummies(df_clean_2['EDUCATION_CAT']).astype(int)
edu_ohe.head(10)

# %%
