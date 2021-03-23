# Scrivere una funzione check_sum_present(li, value) che prende in input una lista di interi (si assume senza
# elementi duplicati) ed un valore e verifica se nella lista ci sono due elementi a,b tali che a+b=value. La funzione
# deve restituire la lista di tutte le coppie [a,b] che soddisfano la condizione.

# lo so fa schifo sto codice, ma volevo fare in fretta e così funziona
def chksum_present(li,value):
    results=[]
    for n in li:
        for m in li:
            if (n!=m):
                 if(n+m==value):
                    if [m,n] not in results:
                        results.append([n,m])
    return results

def  main():
    li=[1,2,3,4,5]
    value=5
    r=chksum_present(li,value)
    print (r)

if __name__ == '__main__':
    main()