import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.35)
df = pd.read_csv("pretrained_model/training_stats_single_mask.csv")
df1 = df.copy()[["step", "training_loss"]]
df1.rename({"training_loss": "loss"}, axis=1, inplace=True)
df1["legend"] = "train"
df2 = df.copy()[["step", "testing_loss"]]
df2.rename({"testing_loss": "loss"}, axis=1, inplace=True)
df2["legend"] = "test"
df = pd.concat([df1, df2])
df["loss"] = df["loss"]*16

plt.figure(figsize=(10, 5))
p = sns.lineplot(data=df, x="step", y="loss", hue="legend")
p.set_xlabel("Batch (batch_size=16)")
p.set_ylabel("Loss")
plt.savefig("pretrained_model/training_stats_single_mask.png")
# plt.show()

sns.set(font_scale=1.35)
df = pd.read_csv("pretrained_model/training_stats_multiple_masks.csv")
df1 = df.copy()[["step", "training_loss"]]
df1.rename({"training_loss": "loss"}, axis=1, inplace=True)
df1["legend"] = "train"
df2 = df.copy()[["step", "testing_loss"]]
df2.rename({"testing_loss": "loss"}, axis=1, inplace=True)
df2["legend"] = "test"
df = pd.concat([df1, df2])
df["loss"] = df["loss"]*16

plt.figure(figsize=(10, 5))
p = sns.lineplot(data=df, x="step", y="loss", hue="legend")
p.set_xlabel("Batch (batch_size=16)")
p.set_ylabel("Loss")
plt.savefig("pretrained_model/training_stats_multiple_masks.png")

sns.set(font_scale=1.35)
df = pd.read_csv(
    "pretrained_model/training_stats_multiple_masks_partially_wrong.csv")
df1 = df.copy()[["step", "training_loss"]]
df1.rename({"training_loss": "loss"}, axis=1, inplace=True)
df1["legend"] = "train"
df2 = df.copy()[["step", "testing_loss"]]
df2.rename({"testing_loss": "loss"}, axis=1, inplace=True)
df2["legend"] = "test"
df = pd.concat([df1, df2])
df["loss"] = df["loss"]*16

plt.figure(figsize=(10, 5))
p = sns.lineplot(data=df, x="step", y="loss", hue="legend")
p.set_xlabel("Batch (batch_size=16)")
p.set_ylabel("Loss")
plt.savefig("pretrained_model/training_stats_multiple_masks_partially_wrong.png")

sns.set(font_scale=1.35)
df = pd.read_csv(
    "pretrained_model/training_stats_multiple_masks_partially_wrong_all_articles.csv")
df = df.copy()[["step", "training_loss"]]
df.rename({"training_loss": "loss"}, axis=1, inplace=True)
df["legend"] = "train"
df["loss"] = df["loss"]*16

plt.figure(figsize=(10, 5))
p = sns.lineplot(data=df, x="step", y="loss", hue="legend")
p.set_xlabel("Batch (batch_size=16)")
p.set_ylabel("Loss")
plt.savefig(
    "pretrained_model/training_stats_multiple_masks_partially_wrong_all_articles.png")
