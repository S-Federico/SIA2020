'''

Un’azienda di collocamento possiede una banca dati contenente le informazioni relative a 50000 infermiere.

Si vogliono contattare tramite posta tutte le infermiere appartenenti alla classe “priority”.

Tali infermiere saranno tutte inviate ad un processo di selezione degli ospedali che hanno fatto richiesta di nuove infermiere.

L’operazione di invio delle lettere ha un costo fisso di 10,000 € e un costo individuale (per ogni infermiera contattata) pari a 5 €.

Gli ospedali prenderanno in prova le infermiere selezionate dalla ditta e alla fine del periodo di prova pagheranno alla ditta 10 € per
ogni infermiera che risulterà essere effettivamente una infermiera classificabile come appartenente alla classe “priority”.

Decidere quale modello, tra quelli studiati permette di ottenere potenzialmente il profitto maggiore.

'''

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

def one_hot_encoder(x):
    if(x=="priority"):
        x=1
    else:
        x=0

def main():
    data=pd.read_csv("nursery data.csv")
    print(data.head(10))
    x=data
    y=data["class"]
    x_train,x_test,y_train,y_test = train_test_split(test_size=0.3,random_state=0)

    clf= DecisionTreeClassifier(max_depth=9)
    clf.fit(x_train,y_train)
    y_pred= clf.predict(x_test)

    print(accuracy_score(y_test,y_pred))



if __name__ == '__main__':
    main()