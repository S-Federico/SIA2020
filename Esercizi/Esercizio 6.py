# Scrivere un programma che legge input 5 numeri interi da tastiera. Ogni numero letto viene inserito in una lista
# solo se non e’ un duplicato di un numero gia’ letto. Se e’ un duplicato, il programma continua la lettura da
# tastiera finche’ un numero non duplicato viene digitato. Dopo aver letto i 5 valori, il programma ne calcola la
# media e la stampa a video.

def avrg(a):
    m=0
    for n in a:
       m=n+m
    return m / len(a)

def main():
    a=[];i=0
    while (i<5):
        n=int (input("Inserire cinque numeri, senza ripetizioni"))
        if n not in a:
            a.append(n)
            i=i+1
    print(a)
    print("La media è : " + str(avrg(a)))


if __name__ == '__main__':
    main();