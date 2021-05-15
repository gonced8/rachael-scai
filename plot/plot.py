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

epoch = pd.read_csv(epoch_filename)["Value"]
rouge1 = pd.read_csv(rouge1_filename)["Value"]
rouge2 = pd.read_csv(rouge2_filename)["Value"]
rougeL = pd.read_csv(rougeL_filename)["Value"]
train_loss = pd.read_csv(train_loss_filename)["Value"]
val_loss = pd.read_csv(val_loss_filename)["Value"]

train_epoch = np.linspace(epoch.iloc[0], epoch.iloc[-1], len(train_loss))
val_epoch = np.linspace(epoch.iloc[0], epoch.iloc[-1], len(val_loss))

fig1, ax1 = plt.subplots(1, 1)
ax1.plot(train_epoch, train_loss, "--", lw=1, label="train")
ax1.plot(val_epoch, val_loss, lw=2, label="val")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid()
ax1.legend()
fig1.tight_layout()

fig2, ax2 = plt.subplots(1, 1)
ax2.plot(val_epoch, rouge1, lw=2, label="rouge1")
ax2.plot(val_epoch, rouge2, lw=2, label="rouge2")
ax2.plot(val_epoch, rougeL, lw=2, label="rougeL")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Rouge")
ax2.grid()
ax2.legend()
fig2.tight_layout()

if SAVE:
    fig1.savefig("figs/loss.pdf")
    fig2.savefig("figs/rouge.pdf")

if SHOW:
    plt.show()
