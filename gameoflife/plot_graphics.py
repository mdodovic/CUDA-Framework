import matplotlib.pyplot as plt


def main():

    simplex_modified_mpi_50000 = [1.514, 1.585, 1.595]
    simplex_modified_mpi_100000 = [1.936, 2.115, 2.146]
    simplex_modified_mpi_1000000 = [1.834, 2.16, 2.151]

    plt.plot(num_of_threads, simplex_modified_mpi_50000)
    plt.plot(num_of_threads, simplex_modified_mpi_100000)
    plt.plot(num_of_threads, simplex_modified_mpi_1000000)
    plt.xlabel("Broj niti")
    plt.ylabel("Ubrzanje")
    plt.legend(["50000 iterations", "100000 iterations", "1000000 iterations"], loc="best")
    plt.savefig("simplex_modified.png", dpi = 90)
    plt.show()


if __name__ == "__main__":
    main()