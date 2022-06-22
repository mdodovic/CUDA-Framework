import matplotlib.pyplot as plt


def main():

    test = ["30x30/1000", "500x500/10", "1000x1000/100", "1000x1000/1000"]
    gameoflife_total = [5.5, 32.4, 111.6226415, 169.6081871]
    gameoflife_evolve = [4.833333333, 153.0744681, 169.9098837, 167.9005848]

    plt.plot(test, gameoflife_total)
    plt.title("Ubrzanje izvrsavanja cele aplikacije")
    plt.xlabel("Test primer")
    plt.ylabel("Ubrzanje")
    plt.savefig("./game_of_life_total.png", dpi = 90)
    plt.show()

    plt.plot(test, gameoflife_evolve)
    plt.title("Ubrzanje funkcije evolve()")
    plt.xlabel("Test primer")
    plt.ylabel("Ubrzanje")
    plt.savefig("./game_of_life_evolve.png", dpi = 90)
    plt.show()


if __name__ == "__main__":
    main()