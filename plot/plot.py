import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHOW = True
SAVE = True

epoch_filename = "data/run-.-tag-epoch.csv"
rouge1_filename = "data/run-.-tag-rouge1.csv"
rouge2_filename = "data/run-.-tag-rouge2.csv"
rougeL_filename = "data/run-.-tag-rougeL.csv"
train_loss_filename = "data/run-.-tag-train_loss.csv"
val_loss_filename = "data/run-.-tag-val_loss.csv"

epoch = pd.read_csv(epoch_filename)
rouge1 = pd.read_csv(rouge1_filename)
rouge2 = pd.read_csv(rouge2_filename)
rougeL = pd.read_csv(rougeL_filename)
train_loss = pd.read_csv(train_loss_filename)
val_loss = pd.read_csv(val_loss_filename)

step_to_epoch = epoch["Value"].iloc[-1] / epoch["Step"].iloc[-1]
train_epoch = train_loss["Step"] * step_to_epoch
val_epoch = val_loss["Step"] * step_to_epoch

fig1, ax1 = plt.subplots(1, 1)
ax1.plot(train_epoch, train_loss["Value"], "--", lw=1, label="train")
ax1.plot(val_epoch, val_loss["Value"], lw=3, label="val")
ax1.set_xlabel("Epoch", fontsize=14)
ax1.set_ylabel("Loss", fontsize=14)
ax1.tick_params(axis="x", labelsize=13)
ax1.tick_params(axis="y", labelsize=13)
ax1.grid()
ax1.legend(fontsize=14)
fig1.tight_layout()

fig2, ax2 = plt.subplots(1, 1)
ax2.plot(val_epoch, rouge1["Value"], lw=3, label="1")
ax2.plot(val_epoch, rouge2["Value"], ls="dashed", lw=3, label="2")
ax2.plot(val_epoch, rougeL["Value"], ls="dotted", lw=3, label="L")
ax2.set_xlabel("Epoch", fontsize=14)
ax2.set_ylabel("ROUGEx-F1", fontsize=14)
ax2.tick_params(axis="x", labelsize=13)
ax2.tick_params(axis="y", labelsize=13)
ax2.grid()
ax2.legend(fontsize=14)
fig2.tight_layout()

if SAVE:
    fig1.savefig("figs/loss.pdf", bbox_inches="tight")
    fig2.savefig("figs/rouge.pdf", bbox_inches="tight")

if SHOW:
    plt.show()
