from src import load_data


def main():
    data = load_data.load_housingdata()
    print(data.head())


if __name__ == "__main__":
    main()
