# scrivere il codice per la funzione fattoriale

def fact(x):
    if x >= 0:
        if x == 0:
            return 1
        else:
            return x * fact(x - 1)
    else:
        print("Factorial function not defined for negative numbers")
        return 0


def main():
    print(fact(1))
    print(fact(2))
    print(fact(-1))


if __name__ == "__main__":
    main()
