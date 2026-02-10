import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_path = "/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/slurm_logs/train_37069531_ReBu97ct.out"

epochs, val_loss, val_acc = [], [], []
pat = re.compile(r"\| Eval epoch (\d+) \s*\| val_loss ([\d\.]+) \s*\| val_acc ([\d\.]+)%")

with open(log_path, "r") as f:
    for line in f:
        m = pat.search(line)
        if m:
            epochs.append(int(m.group(1)))
            val_loss.append(float(m.group(2)))
            val_acc.append(float(m.group(3)))

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

axes[0].plot(epochs, val_loss, label="val_loss")
axes[0].set_ylabel("val_loss")
axes[0].grid(True)

axes[1].plot(epochs, val_acc, color="tab:orange", label="val_acc")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("val_acc (%)")
axes[1].grid(True)

plt.tight_layout()
out = "/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/outputs/val_loss_val_acc_train_37069531.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved to {out}")