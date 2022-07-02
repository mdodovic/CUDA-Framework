import matplotlib.pyplot as plt


def main():

    test = ["32x32/8192", "256x256/8192", "1024x1024/4096", "1024x1024/8192", "1024x1024/16384", "1024x1024/32768"]
    hotspot = [1.281690141, 16.07084469, 21.33158813, 21.0401659, 21.81525183, 20.83893718]

    plt.plot(test, hotspot)
    plt.title("Ubrzanje aplikacije")
    plt.xlabel("Test primer")
    plt.ylabel("Ubrzanje")
    plt.xticks(rotation=-45)
    plt.savefig("./hotspot.png", dpi = 90)
    plt.show()


if __name__ == "__main__":
    main()