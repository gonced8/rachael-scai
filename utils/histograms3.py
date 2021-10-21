import json
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATS = False
SAMPLE = False
SHOW = True
SAVE = False

plt.rcParams.update({"font.size": 16})
plt.rcParams["font.family"] = "serif"


def split_df(df):
    df1, df2 = np.array_split(df, 2)
    return df1, df2


def stats(data):
    for k, v in data.items():
        if k.count(",") == 2:
            print(
                f"{k}: {v['ROUGE1-R'].mean():.3f} {v['MRR'].mean():.3f} {v['F1'].mean():.3f} {v['Exact match'].mean():.3f} len={len(v)}"
            )


def sample_from_each_case(data):
    for k, v in data.items():
        if k.count(",") == 2:
            print(k)
            print(v.sample(4, ignore_index=True))


def get_success_fail(data, metric, threshold=None, quartile=4):
    if threshold is None:
        *fail, success = np.array_split(data.sort_values(by=metric), quartile)
        fail = pd.concat(fail)
    else:
        fail = data[data[metric] < threshold]
        success = data[data[metric] >= threshold]

    return success, fail


def analyze(filename):
    # Load data
    data = pd.read_csv(filename, sep="\t", index_col=False)

    print(f"Data: {data.size} entries")

    # Consider only rows/samples with all metrics
    data = pd.DataFrame.dropna(data)

    # Split data
    splits = {"original": data}

    splits["ROUGE1-R success"], splits["ROUGE1-R fail"] = get_success_fail(
        splits["original"], ["ROUGE1-R", "MRR", "F1"]
    )

    mrr_threshold = 1 / 4
    (
        splits["ROUGE1-R success, MRR success"],
        splits["ROUGE1-R success, MRR fail"],
        splits["ROUGE1-R fail, MRR success"],
        splits["ROUGE1-R fail, MRR fail"],
    ) = (
        *get_success_fail(splits["ROUGE1-R success"], "MRR", mrr_threshold),
        *get_success_fail(splits["ROUGE1-R fail"], "MRR", mrr_threshold),
        # *get_success_fail(splits["ROUGE1-R success"], ["MRR", "ROUGE1-R", "F1"]),
        # *get_success_fail(splits["ROUGE1-R fail"], ["MRR", "ROUGE1-R", "F1"]),
    )

    # F1 3rd quartile
    f1_q3 = splits["original"]["F1"].quantile(0.75)

    (
        splits["ROUGE1-R success, MRR success, F1 success"],
        splits["ROUGE1-R success, MRR success, F1 fail"],
        splits["ROUGE1-R success, MRR fail, F1 success"],
        splits["ROUGE1-R success, MRR fail, F1 fail"],
        splits["ROUGE1-R fail, MRR success, F1 success"],
        splits["ROUGE1-R fail, MRR success, F1 fail"],
        splits["ROUGE1-R fail, MRR fail, F1 success"],
        splits["ROUGE1-R fail, MRR fail, F1 fail"],
    ) = (
        *get_success_fail(splits["ROUGE1-R success, MRR success"], "F1", f1_q3),
        *get_success_fail(splits["ROUGE1-R success, MRR fail"], "F1", f1_q3),
        *get_success_fail(splits["ROUGE1-R fail, MRR success"], "F1", f1_q3),
        *get_success_fail(splits["ROUGE1-R fail, MRR fail"], "F1", f1_q3),
    )

    # Get means and medians
    metrics = ["ROUGE1-R", "MRR", "F1", "Exact match"]
    medians = {
        description: {metric: split[metric].median() for metric in metrics}
        for description, split in splits.items()
    }
    means = {
        description: {metric: split[metric].mean() for metric in metrics}
        for description, split in splits.items()
    }

    print("medians", json.dumps(medians, indent=4), sep="\n")
    print("means", json.dumps(means, indent=4), sep="\n")

    # ROUGE1-R 3rd quartile
    rouge_q3 = splits["original"]["ROUGE1-R"].quantile(0.75)

    # Plot ROUGE1-R
    fig1, ax1 = plt.subplots()

    ax1.axvline(rouge_q3, color="k", linestyle="dashed", linewidth=2)
    ax1.hist(
        splits["original"]["ROUGE1-R"],
        bins=20,
        weights=np.ones(len(splits["original"])) / len(splits["original"]),
        edgecolor="white",
        color="#979797",
    )

    ax1.legend(["threshold (Q3)"])
    ax1.set_xlabel("ROUGE1-R")
    ax1.set_ylabel("Relative Frequency")

    # Plot MRR
    fig2, ax2 = plt.subplots()

    length = len(splits["original"])

    ax2.axvline(mrr_threshold, color="k", linestyle="dashed", linewidth=2)
    ax2.hist(
        [splits["ROUGE1-R fail"]["MRR"], splits["ROUGE1-R success"]["MRR"]],
        # bins=[0, 0.1, 1 / 3, 0.9, 1],
        bins=10,
        weights=[
            # np.ones(len(splits["ROUGE1-R fail"])) / length,
            # np.ones(len(splits["ROUGE1-R success"])) / length,
            np.ones(len(splits["ROUGE1-R fail"])) / len(splits["ROUGE1-R fail"]),
            np.ones(len(splits["ROUGE1-R success"])) / len(splits["ROUGE1-R success"]),
        ],
        stacked=True,
        edgecolor="white",
        color=["#F2AD00", "#00A08A"],
    )

    ax2.legend(["threshold (1/4)", "rewriting $x$", "rewriting $\checkmark$"])
    ax2.set_xlabel("MRR@10")
    ax2.set_ylabel("Relative Frequency")

    # Plot F1
    fig3, ax3 = plt.subplots()

    ax3.axvline(f1_q3, color="k", linestyle="dashed", linewidth=2)
    ax3.hist(
        splits["original"]["F1"],
        bins=20,
        weights=np.ones(len(splits["original"])) / len(splits["original"]),
        edgecolor="white",
        color="#979797",
    )

    ax3.legend(["threshold (Q3)"])
    ax3.set_xlabel("F1")
    ax3.set_ylabel("Relative Frequency")

    # Plot F1 (ROUGE1-R fail)
    fig4, ax4 = plt.subplots()

    length = len(splits["ROUGE1-R fail"])

    ax4.hist(
        [
            splits["ROUGE1-R fail, MRR fail"]["F1"],
            splits["ROUGE1-R fail, MRR success"]["F1"],
        ],
        bins=10,
        weights=[
            # np.ones(len(splits["ROUGE1-R fail, MRR fail"])) / length,
            # np.ones(len(splits["ROUGE1-R fail, MRR success"])) / length,
            np.ones(len(splits["ROUGE1-R fail, MRR fail"]))
            / len(splits["ROUGE1-R fail, MRR fail"]),
            np.ones(len(splits["ROUGE1-R fail, MRR success"]))
            / len(splits["ROUGE1-R fail, MRR success"]),
        ],
        stacked=True,
        edgecolor="white",
        color=["#d22d36", "#046C9A"],
        align="left",
        rwidth=0.5,
    )

    # Plot F1 (ROUGE1-R success)
    length = len(splits["ROUGE1-R success"])

    ax4.hist(
        [
            splits["ROUGE1-R success, MRR fail"]["F1"],
            splits["ROUGE1-R success, MRR success"]["F1"],
        ],
        bins=10,
        weights=[
            # np.ones(len(splits["ROUGE1-R success, MRR fail"])) / length,
            # np.ones(len(splits["ROUGE1-R success, MRR success"])) / length,
            np.ones(len(splits["ROUGE1-R success, MRR fail"]))
            / len(splits["ROUGE1-R success, MRR fail"]),
            np.ones(len(splits["ROUGE1-R success, MRR success"]))
            / len(splits["ROUGE1-R success, MRR success"]),
        ],
        stacked=True,
        edgecolor="white",
        color=["#F98400", "#5BBCD6"],
        align="mid",
        rwidth=0.5,
    )

    ax4.legend(
        [
            "rewriting $x$\n retrieval $x$",
            "rewriting $x$\n retrieval $\checkmark$",
            "rewriting $\checkmark$\n retrieval $x$",
            "rewriting $\checkmark$\n retrieval $\checkmark$",
        ]
    )
    ax4.set_xlabel("F1")
    ax4.set_ylabel("Relative Frequency")

    # Print stats
    if STATS:
        stats(splits)

    # Sample samples of each case
    if SAMPLE:
        sample_from_each_case(splits)

    if SHOW:
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        fig4.tight_layout()
        plt.show()

    if SAVE:
        fig1.savefig("plots/rouge1-r.pdf", bbox_inches="tight")
        fig2.savefig("plots/mrr.pdf", bbox_inches="tight")
        fig3.savefig("plots/f1.pdf", bbox_inches="tight")
        fig4.savefig("plots/f1_hist.pdf", bbox_inches="tight")


def main(filenames):
    for filename in filenames:
        analyze(filename)


if __name__ == "__main__":
    if len(argv) <= 1:
        print("Please specify CSV file to load.")
        print(f"Example: {argv[0]} example_file.csv")
    else:
        main(argv[1:])
