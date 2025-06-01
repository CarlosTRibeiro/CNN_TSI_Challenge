import pandas as pd

# CSVs de imagens
df_gray_8x8 = pd.read_csv('Dataset/hmnist_8_8_L.csv', sep=',')
df_rgb_8x8 = pd.read_csv('Dataset/hmnist_8_8_RGB.csv', sep=',')
df_gray_28x28 = pd.read_csv('Dataset/hmnist_28_28_L.csv', sep=',')
df_rgb_28x28 = pd.read_csv('Dataset/hmnist_28_28_RGB.csv', sep=',')

# CSV de variáveis demográficas
df_meta = pd.read_csv('Dataset/HAM10000_metadata.csv')

# Verificando os primeiros registros para cada DataFrame
print(df_rgb_8x8.head())
print(df_gray_8x8.head())
print(df_rgb_28x28.head())
print(df_gray_28x28.head())
print(df_meta.head())

# Verificar se há valores nulos
print(df_rgb_8x8.isnull().sum())
print(df_gray_8x8.isnull().sum())
print(df_rgb_28x28.isnull().sum())
print(df_gray_28x28.isnull().sum())
print(df_meta.isnull().sum())

# Verificar se todos os DataFrames têm o mesmo número de linhas
print(len(df_rgb_8x8), len(df_gray_8x8), len(df_rgb_28x28), len(df_gray_28x28), len(df_meta))


# Remover a coluna 'label' dos DataFrames de imagem
df_rgb_8x8 = df_rgb_8x8.drop('label', axis=1)
df_gray_8x8 = df_gray_8x8.drop('label', axis=1)
df_rgb_28x28 = df_rgb_28x28.drop('label', axis=1)
df_gray_28x28 = df_gray_28x28.drop('label', axis=1)

# Visualizar os tipos de dados
print(df_meta.dtypes)

# Converter sexo para valores numéricos
df_meta['sex'] = df_meta['sex'].map({'male': 1, 'female': 0})

# Lidar com valores faltantes (se necessário)
df_meta['age'].fillna(df_meta['age'].median(), inplace=True)
df_meta['sex'].fillna(df_meta['sex'].mode()[0], inplace=True)
df_meta['localization'].fillna('unknown', inplace=True)

# One-hot encoding para 'localization'
df_meta = pd.get_dummies(df_meta, columns=['localization'])

# Definir os diagnósticos malignos
malignant_cases = ['mel', 'bcc', 'akiec']

# Criar a nova coluna 'malignancy'
df_meta['malignancy'] = df_meta['dx'].apply(lambda x: 1 if x in malignant_cases else 0)

# Verificar o resultado
print(df_meta[['dx', 'malignancy']].head())


print(df_rgb_8x8.shape)
print(df_gray_8x8.shape)
print(df_rgb_28x28.shape)
print(df_gray_28x28.shape)
print(df_meta.head())

# Verificar a quantidade de casos malignos e benignos
print(df_meta['malignancy'].value_counts())


# Salvando os DataFrames como CSV (sem o índice)
df_rgb_8x8.to_csv('rgb_8x8.csv', index=False, sep=',')
df_gray_8x8.to_csv('gray_8x8.csv', index=False, sep=',')
df_rgb_28x28.to_csv('rgb_28x28.csv', index=False, sep=',')
df_gray_28x28.to_csv('gray_28x28.csv', index=False, sep=',')
df_meta.to_csv('meta.csv', index=False, sep=',')