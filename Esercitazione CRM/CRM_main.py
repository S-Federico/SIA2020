"""
Si consideri il set di dati CRM, che contiene alcune informazioni sui clienti di una piccola
azienda. Si vogliono analizzare tali dati e ricavare un modello previsivo in grado di
distinguere i clienti abituali dai clienti occasionali, allo scopo di intraprendere opportune
politiche di marketing mirato sui primi. Gli attributi presenti in questo insieme di dati
appartengono essenzialmente a due gruppi: informazioni sul primo acquisto del cliente e
informazioni personali.
"""

import itertools as it

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split

from Utils_esercitazioni.Classification_utils import train_classifiers, pretty_print_classifiers_sorted, validate_model

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

def main():
    #Lettura dei dati
    crm_df = pd.read_excel('CRM data.xlsx')
    print(crm_df.head())
    print(crm_df.describe())

    #Impostiamo i valori mancanti a 0
    crm_df = crm_df.fillna(0)

    for row in crm_df.itertuples():
        # Controllo sul primo ordine e numero di prodotti
        if row[3] <= 0 or row[4] <= 0:
            crm_df = crm_df.drop(row[0])
            continue

        for n in it.chain([2], range(5, 15)):
            if row[n] not in [0, 1]:
                crm_df = crm_df.drop(row[0])
                break

        # Controllo sulle dimensioni dell'azienda
        dims = [row[9], row[10], row[15]]
        if dims.count(1) != 1:
            crm_df = crm_df.drop(row[0])
            continue

        # Controllo sulla zona dell'azienda
        zones = [row[6], row[13], row[14]]
        if zones.count(1) != 1:
            crm_df = crm_df.drop(row[0])
            continue

        # Controllo sull'età
        ages = [row[7], row[8], row[11]]
        if ages.count(1) != 1:
            crm_df = crm_df.drop(row[0])
            continue

    # Controllo sulle istanze per duplicati
    duplicated_rows = crm_df[crm_df.duplicated(['client_code'])]

    # Droppiamo i duplicati
    crm_df.drop_duplicates(subset=['client_code'], inplace=True, keep='first')

    # Controlliamo quante istanze abbiamo droppato
    print("\nNew dimensions: ")
    print(crm_df.describe())

    # Rimuoviamo la colonna di id, non utile ai fini dell'analisi
    del crm_df['client_code']

    # Mettiamo l'attributo 'first_amount-spent' in bin di uguale dimensione
    pd.cut(crm_df['first_amount_spent'], 10)

    # Mostriamo la matrice di correlazione
    corr_mat = crm_df.corr()
    sn.heatmap(corr_mat, annot=True, square=True, fmt='.2f')
    plt.title('Matrice di correlazione tra gli attributi')
    plt.show()

    print(crm_df.dtypes)

    # Divisione in train e test set
    X = crm_df
    Y = X.pop('Y').values

    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size=0.3, random_state=0)

    # Allenamento e analisi dei risultati
    clss = train_classifiers(X_train, Y_train, X_test, Y_test, 5, True)
    best_model_name=pretty_print_classifiers_sorted(clss, "test_score")

    # Analisi del modello migliore
    validate_model(best_model_name, X_train, Y_train)

if __name__ == '__main__':
    main()