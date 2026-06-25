import pandas as pd
import matplotlib.pyplot as plt
import os


os.makedirs("results", exist_ok=True)


df = pd.read_csv(
    "training_log_15ue.monitor.csv",
    skiprows=1
)


df["smoothed"] = df["r"].rolling(
    window=5,
    min_periods=1
).mean()


plt.figure(figsize=(8, 5))

plt.plot(
    df.index,
    df["r"],
    alpha=0.3,
    label="Raw Reward"
)

plt.plot(
    df.index,
    df["smoothed"],
    linewidth=3,
    label="Moving Average"
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Reward Convergence")
plt.legend()
plt.grid(True)


plt.savefig(
    "results/reward_curve_smoothed.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved: results/reward_curve_smoothed.png")

plt.show()