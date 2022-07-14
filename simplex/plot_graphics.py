import matplotlib.pyplot as plt


def main():

    test = [50000, 100000, 1000000, 10000000]
    hotspot = [1.136876006, 1.155664222, 1.157585578, 1.150718317]

    plt.plot(test, hotspot)
    plt.title("Ubrzanje aplikacije")
    plt.xlabel("Test primer")
    plt.ylabel("Ubrzanje")
    plt.savefig("./simplex.png", dpi = 90)
    plt.show()


if __name__ == "__main__":
    main()