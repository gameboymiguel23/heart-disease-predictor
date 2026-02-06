import os
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_target_distribution(df, target_col: str, output_dir: str = "outputs") -> str:
    ensure_dir(output_dir)
    outpath = os.path.join(output_dir, "target_distribution.png")

    counts = df[target_col].value_counts().sort_index()
    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Target distribution")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath


def plot_correlation_heatmap(df, output_dir: str = "outputs") -> str:
    ensure_dir(output_dir)
    outpath = os.path.join(output_dir, "correlation_heatmap.png")

    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values)
    plt.title("Correlation heatmap")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath
