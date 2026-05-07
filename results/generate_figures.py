import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modern, clean theme with sharper typography
sns.set_theme(style="ticks", context="paper", font_scale=1.4)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


def plot_per_subject_metrics(results_data, out_dir):
    if "per_subject" not in results_data:
        return

    subjects = []
    accs = []
    kappas = []

    for sub, metrics in results_data["per_subject"].items():
        subjects.append(f"Sub {sub}")
        accs.append(metrics.get("accuracy", 0))
        kappas.append(metrics.get("kappa", 0))

    df_metrics = pd.DataFrame({"Subject": subjects, "Accuracy": accs, "Kappa": kappas})

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_metrics,
        x="Subject",
        y="Accuracy",
        palette="crest",
        edgecolor=".1",
        linewidth=1.5,
    )

    if "mean_accuracy" in results_data:
        mean_acc = results_data["mean_accuracy"]
        plt.axhline(
            mean_acc,
            color="#e63946",
            linestyle="--",
            linewidth=2.5,
            label=f"Mean Acc: {mean_acc:.3f}",
        )
        plt.legend(frameon=True, fancybox=True, shadow=True)

    plt.ylim(0, 1.05)
    plt.title("Per-Subject Accuracy", pad=15)
    plt.ylabel("Accuracy")
    plt.xlabel("Subject")
    plt.xticks(rotation=45)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "per_subject_accuracy.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot Kappa
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_metrics,
        x="Subject",
        y="Kappa",
        palette="flare",
        edgecolor=".1",
        linewidth=1.5,
    )

    if "mean_kappa" in results_data:
        mean_kappa = results_data["mean_kappa"]
        plt.axhline(
            mean_kappa,
            color="#1d3557",
            linestyle="--",
            linewidth=2.5,
            label=f"Mean Kappa: {mean_kappa:.3f}",
        )
        plt.legend(frameon=True, fancybox=True, shadow=True)

    plt.ylim(0, 1.05)
    plt.title("Per-Subject Cohen's Kappa", pad=15)
    plt.ylabel("Kappa")
    plt.xlabel("Subject")
    plt.xticks(rotation=45)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "per_subject_kappa.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_learning_curves(history_files, out_dir):
    if not history_files:
        return

    all_data = []
    for f in history_files:
        sub_name = os.path.basename(f).replace(".json", "")
        with open(f, "r") as file:
            hist = json.load(file)
            for entry in hist:
                entry["subject"] = sub_name
                all_data.append(entry)

    df = pd.DataFrame(all_data)

    metrics_to_plot = [
        ("train_loss", "Training Loss", "#2a9d8f"),
        ("test_accuracy", "Test Accuracy", "#457b9d"),
        ("test_kappa", "Test Kappa", "#e07a5f"),
    ]

    for col, title, sharp_color in metrics_to_plot:
        if col in df.columns:
            plt.figure(figsize=(10, 6))

            # Mean and SD across subjects
            mean_df = df.groupby("epoch")[col].mean()
            std_df = df.groupby("epoch")[col].std()

            plt.plot(
                mean_df.index,
                mean_df.values,
                label="Mean",
                color=sharp_color,
                linewidth=3,
            )
            plt.fill_between(
                mean_df.index,
                mean_df.values - std_df.values,
                mean_df.values + std_df.values,
                color=sharp_color,
                alpha=0.15,
                label="±1 SD",
                edgecolor="none",
            )

            plt.title(f"{title} vs Epochs", pad=15)
            plt.xlabel("Epoch")
            plt.ylabel(title)
            plt.legend(frameon=True, fancybox=True, shadow=True)
            sns.despine()
            plt.grid(axis="y", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(
                os.path.join(out_dir, f"learning_curve_{col}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # Plot individual subjects
            plt.figure(figsize=(10, 6))
            n_subs = len(df["subject"].unique())
            sub_palette = sns.color_palette("husl", n_subs)

            for idx, sub in enumerate(df["subject"].unique()):
                sub_data = df[df["subject"] == sub]
                plt.plot(
                    sub_data["epoch"],
                    sub_data[col],
                    alpha=0.7,
                    label=sub,
                    linewidth=2,
                    color=sub_palette[idx],
                )

            plt.title(f"Individual Subjects: {title}", pad=15)
            plt.xlabel("Epoch")
            plt.ylabel(title)
            plt.legend(
                bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True, shadow=True
            )
            sns.despine()
            plt.grid(axis="y", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(
                os.path.join(out_dir, f"learning_curve_individual_{col}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


def main():
    base_dir = os.getcwd()

    for item in os.listdir(base_dir):
        # Exclude pretrained and ignore non-directories
        if item == "pretrain_cross_dataset" or not os.path.isdir(item):
            continue

        print(f"Processing folder: {item}")
        exp_dir = os.path.join(base_dir, item)

        # Look for subdirectories containing the results (e.g. bnci2014_...)
        for sub_item in os.listdir(exp_dir):
            sub_dir = os.path.join(exp_dir, sub_item)
            if not os.path.isdir(sub_dir):
                continue

            print(f"  Found sub-folder: {sub_item}")

            results_files = glob.glob(os.path.join(sub_dir, "*results.json"))
            history_files = glob.glob(os.path.join(sub_dir, "*history.json"))

            if not results_files and not history_files:
                continue

            fig_dir = os.path.join(sub_dir, "figures")
            os.makedirs(fig_dir, exist_ok=True)

            if results_files:
                with open(results_files[0], "r") as f:
                    results_data = json.load(f)
                plot_per_subject_metrics(results_data, fig_dir)
                print(f"    -> Generated per-subject metrics plots in {fig_dir}")

            if history_files:
                plot_learning_curves(history_files, fig_dir)
                print(f"    -> Generated learning curve plots in {fig_dir}")


if __name__ == "__main__":
    main()
