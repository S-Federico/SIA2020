#Scrivere una funzione check_string(str) che riceve in input una stringa, e ritorna 0 se la stringa str e ben formata, 1 altrimenti.
#Assumiamo che una stringa sia ben formata se contiene solo caratteri minuscoli

import string

def chkstr(str):
    for c in str:
        if c not in string.ascii_lowercase:
            return 1
    return 0

def main():
    print (chkstr("ciao"))
    print (chkstr("cIAO"))

if __name__ == "__main__":
    main()