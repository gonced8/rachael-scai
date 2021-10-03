from sys import argv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyze(filename):
    # Load data
    data = pd.read_csv(filename, sep="\t", index_col=False)

    # Consider only rows/samples with all metrics
    data = pd.DataFrame.dropna(data)

    sns.displot(data=data["ROUGE1-R"], kind="kde")
    plt.show()
    print(data)
    data["ROUGE1-R"]


def main(filenames):
    for filename in filenames:
        analyze(filename)


if __name__ == "__main__":
    if len(argv) <= 1:
        print("Please specify CSV file to load.")
        print(f"Example: {sys.argv[0]} example_file.csv")
    else:
        main(argv[1:])
