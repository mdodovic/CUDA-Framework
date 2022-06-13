import matplotlib.pyplot as plt


def main():

    num_of_threads = [2, 4, 8]

    gameoflife_mpi_30_30_1000 = [1.514, 1.585, 1.595]
    gameoflife_mpi_500_500_10 = [1.936, 2.115, 2.146]
    gameoflife_mpi_1000_1000_100 = [1.834, 2.16, 2.151]
    gameoflife_mpi_1000_1000_1000 = [1.834, 2.16, 2.151]

    plt.plot(num_of_threads, gameoflife_mpi_30_30_1000)
    plt.plot(num_of_threads, gameoflife_mpi_500_500_10)
    plt.plot(num_of_threads, gameoflife_mpi_1000_1000_100)
    plt.plot(num_of_threads, gameoflife_mpi_1000_1000_1000)
    plt.xlabel("Broj niti")
    plt.ylabel("Ubrzanje")
    plt.legend(["30x30/1000", "500x500/10", "1000x1000/100", "1000x1000/1000"], loc="best")
    plt.savefig("./game_of_life.png", dpi = 90)
    plt.show()


if __name__ == "__main__":
    main()