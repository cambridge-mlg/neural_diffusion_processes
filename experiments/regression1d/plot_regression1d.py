from pathlib import Path
import matplotlib.pyplot as plt
from tueplots import bundles
import numpy as  np

DATASETS = [
    "se",
    "matern",
    "weaklyperiodic",
    "sawtooth",
    "mixture",
]

RENAMES = {
    "se": "Squared Exponential",
    "sawtooth": "Sawtooth",
    "matern": r"Mat\'ern--$\frac52$",
    "weaklyperiodic": "Weakly Periodic",
    "mixture": "Mixture",
}

def display_name(x):
    return RENAMES.get(x, "****" + x + "****")


def plot_dataset(dataset: str, axes, legend=False):
    if Path(f"./plot_model_{dataset}.npz").exists():
        print("Loading plot data")
        npz = np.load(f"./plot_model_{dataset}.npz")
        x_dataset = npz["x_dataset"]
        ys_dataset = npz["ys_dataset"]
        x_prior = npz["x_prior"]
        ys_prior = npz["ys_prior"]
        x_post = npz["x_post"]
        ys_post = npz["ys_post"]
        x_context = npz["x_context"]
        y_context = npz["y_context"]
        try:
            m = npz["m"]
            v = npz["v"]
        except:
            m = None
            v = None

    lo, hi = -2, 2
    ns = 3 if dataset == "sawtooth" else 10
    a = .25 if dataset == "sawtooth" else 1.

    axes[0].plot(x_dataset, ys_dataset[:ns, :, 0].T, color="C0", alpha=.3)
    axes[1].plot(x_prior, ys_prior[:ns, -1, :, 0].T, "C0", alpha=.3)
    if m is not None and v is not None:
        axes[2].plot(x_post, m, "k", lw=1, alpha=.8)
        axes[2].fill_between(x_post.ravel(), m.ravel() - 1.96 * v.ravel(), m.ravel() + 1.96 * v.ravel(), color="k", alpha=.1, label="True Posterior")
        if legend: axes[2].legend()
    axes[2].plot(x_post, ys_post[:ns, :, 0].T, "C0", alpha=.3)
    axes[2].plot(x_context, y_context, "C3o", markersize=2)
    axes[0].set_ylabel(display_name(dataset))


    for ax in axes:
        ax.set_xlim(lo, hi)
        # ax.set_ylim(-3*a, 3*a)


plt.rcParams.update(bundles.neurips2023(nrows=len(DATASETS), ncols=3))
# def neurips2023(*, usetex=True, rel_width=1.0, nrows=1, ncols=1, family="serif"):
fig, axes = plt.subplots(nrows=len(DATASETS), ncols=3, sharex=True, sharey='row')
for i, dataset in enumerate(DATASETS):
    plot_dataset(dataset, axes[i], legend=i==0)

axes[0, 0].set_title("Data Prior")
axes[0, 1].set_title("Model Prior")
axes[0, 2].set_title("Model Posterior")

# align y labels
fig.align_ylabels()

plt.savefig("plot_regression1d.pdf", bbox_inches="tight")
plt.savefig("plot_regression1d.png", bbox_inches="tight", dpi=200)
plt.show()