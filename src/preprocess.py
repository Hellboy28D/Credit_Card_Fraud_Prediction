from utils import *


credit_card_data = pd.read_csv('creditcard.csv')

print(credit_card_data.head())
print(credit_card_data.tail())
print(credit_card_data.info())
print(credit_card_data.isnull().sum())
print(credit_card_data['Class'].value_counts())

# Seprating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

print(legit.Amount.describe())
print(fraud.Amount.describe())

# Compaer the values for both transaction
print(credit_card_data.groupby('Class').mean())

# *** Under-Sampling ***

legit_sample = legit.sample(n = 492)
print(legit_sample.head( ))

# Concatenating two dataframes
new_dataset = pd.concat([legit_sample, fraud], axis = 0)
print(new_dataset.groupby('Class').mean())
