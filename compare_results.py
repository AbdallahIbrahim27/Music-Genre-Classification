"""
compare_results.py
------------------
Runs BOTH approaches and produces a side-by-side comparison chart.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from tabular_approach import main as run_tabular
from cnn_approach     import main as run_cnn


def main():
    print("=" * 70)
    print("  MUSIC GENRE CLASSIFICATION — FULL COMPARISON")
    print("=" * 70)

    print("\n[1/2] Running Tabular Approach (MFCC + Scikit-learn) ...")
    tabular_results = run_tabular()

    print("\n[2/2] Running CNN Approach (Mel Spectrogram + Keras) ...")
    cnn_results = run_cnn()

    # ── Build merged comparison chart ─────────────────────────────────────────
    all_models = {}
    for name, acc in tabular_results.items():
        all_models[f'[Tabular]\n{name}'] = acc
    for name, acc in cnn_results.items():
        all_models[f'[CNN]\n{name}'] = acc

    labels  = list(all_models.keys())
    values  = [v * 100 for v in all_models.values()]
    colors  = (['#2196F3', '#03A9F4'] * 10)[:len(tabular_results)]  + \
              (['#9C27B0', '#E91E63'] * 10)[:len(cnn_results)]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor='white', linewidth=1.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Music Genre Classification — All Models Compared', fontsize=15, fontweight='bold')
    ax.axhline(y=np.mean(values), color='gray', linestyle='--', alpha=0.6,
               label=f'Mean: {np.mean(values):.1f}%')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    legend_patches = [
        mpatches.Patch(color='#2196F3', label='Tabular / ML'),
        mpatches.Patch(color='#9C27B0', label='Image / CNN'),
    ]
    ax.legend(handles=legend_patches + [plt.Line2D([0], [0], color='gray',
              linestyle='--', label=f'Mean: {np.mean(values):.1f}%')],
              loc='upper right')

    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=150)
    plt.close()

    # ── Print final table ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'Model':<35} {'Accuracy':>10}")
    print("  " + "-" * 47)
    for label, acc in zip(labels, values):
        clean = label.replace('\n', ' ')
        print(f"  {clean:<35} {acc:>9.2f}%")
    best_label = labels[np.argmax(values)].replace('\n', ' ')
    print("=" * 70)
    print(f"  🏆 Best overall: {best_label}  ({max(values):.2f}%)")
    print("  📊 final_comparison.png saved")
    print("=" * 70)


if __name__ == '__main__':
    main()
