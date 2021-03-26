from sklearn import  datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import numpy as np

iris=datasets.load_iris()
#riduciamo la matrice. Prendiamo tutte le righe (:) ma solo le colonne 2 e 3

X=iris.data[:,[2,3]]
Y=iris.target

#ora dividiamo il dtataset in training e test

#X sono le istanze, usate come data,
# Y sono le  etichette, usate come target
#Testsize indica quanto usiamo per il test, in questo caso 30%
#Random_state è l'indicatore di randomizzazione, serve a dire in che modo selezionare le istanze che vanno nei due set
X_train,X_test,Y_train,Y_test = \
    train_test_split(X,Y,test_size=0.3,random_state=0)


#albero di classificazione di profondità massima pari a 5
clf=DecisionTreeClassifier(max_depth=5 )

#facciamo il training del modello
clf.fit(X_train,Y_train)

#Facciamo la predizione
Y_pred=clf.predict(X_test)

#Facciamo la verifica confrontando con Y_test

a_s=accuracy_score(Y_test,Y_pred)

print(a_s)

report = classification_report(Y_pred,Y_test)
print(report)

