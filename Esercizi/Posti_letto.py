import pandas as p
import matplotlib.pyplot as plt

#read csv
csv = p.read_csv('postiletto.csv',sep=';')

#ispezione dei dati e dei tipi
print(csv.head())
print( csv.dtypes)

#forziamo la conversione numerica
csv= csv.convert_dtypes(convert_integer=True,convert_floating=True)

#ispezione dei dati e dei tipi
#print(csv.head())
#print( csv.dtypes)

#Analizziamo il totale postiletto in un anno per regione(credo)

#con questo selezioniamo la colonna totale posti letto e solo per l'anno 2014
csv2014 = csv[csv['Anno']==2014]
beds2014 = csv2014['Totale posti letto']

#descriptive stats
#print (beds2014.describe())

#plot
#histogram = beds2014.hist(bins=50)

#aggiungiamo dettagli al grafico
#histogram.set_title('Distribuzione di letti per ospedale - 2014')
#histogram.set_xlabel('Numero di letti')
#histogram.set_ylabel('count')

#mostriamo il plot
#plt.show()

#ordiniamo per numero di letti
csv2014=csv2014.sort_values('Totale posti letto', ascending=False)

#mostriamo i nomi degli ospedali e i posti letto totali
print(csv2014[['Denominazione Struttura/Stabilimento','Totale posti letto']])

#raggruppiamo il numero di letti per regione
bedsByRegion=csv[['Descrizione Regione','Totale posti letto']].groupby('Descrizione Regione')

#sommiamo i letti in ogni regione e sortiamo
summedandsortedbeds=bedsByRegion.sum().sort_values('Totale posti letto')

#horizontal plot
summedandsortedbeds.plot.barh()

#show plot
plt.show()