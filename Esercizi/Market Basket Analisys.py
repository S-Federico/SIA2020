import mlxtend
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules


def hot_encode(x):
    if(x<=0):
        return 0
    if(x>=1):
        return 1
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)  
data = pd.read_csv("online-retail-dataset.csv", sep=',')
print(data['Description'].head())

#Rimuoviamo gli spazi extra dalle descrizioni. Strip rimuove il carattere
# passato come argomento dall'inizio e la fine della stringa

data['Description'] = data['Description'].str.strip()
print()
print(data['Description'].head())

#Droppiamo le transazioni senza numero di fattura e a credito, indicate con c
data.dropna(axis=0,subset=['InvoiceNo'],inplace=True) #droppa i nulli
data['InvoiceNo'] = data['InvoiceNo'].astype('str')
data = data[~data['InvoiceNo'].str.contains('C')] # '~' significa not

#dividiamo le transazioni per i paesi di interesse
basket_France = (data[data['Country']=="France"]
                 .groupby(['InvoiceNo','Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))

basket_UK = (data[data['Country']=="United Kingdom"]
                 .groupby(['InvoiceNo','Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))

#I valori per le transazioni sono stati portati tutti a 1
basket_encoded=basket_France.applymap(hot_encode)
basket_France=basket_encoded

basket_encoded=basket_UK.applymap(hot_encode)
basket_UK=basket_encoded
print(basket_UK)

#Costruzione del modello
frq_items=apriori(basket_France,min_support=0.05,use_colnames=True)
#Determiniamo le regole di associazione
rules=association_rules(frq_items,metric="lift",min_threshold=1)
rules=rules.sort_values(['confidence','lift'],ascending=[False,False])
print(rules.head())
