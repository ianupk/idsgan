"""
Evaluate IDSGAN and reproduce key paper results.

Reproduces:
    Table 2   – Detection rates before and after adversarial generation
    Figure 2  – DR comparison bar charts (saved as PNG + PDF)
    Figure 3  – EIR robustness bar charts (saved as PNG + PDF)

Run train_ids.py and train_idsgan.py before this script.

Usage
-----
    python evaluate.py
"""

import os

import matplotlib
matplotlib.use("Agg")           # non-interactive backend; must be set before pyplot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate

from src import config, feature_utils
from src.data_preprocessing import prepare_datasets
from src.ids_models import load_ids_model, predict_ids
from src.idsgan import IDSGAN
from src.metrics import detection_rate, evasion_increase_rate


# Helper – load a trained IDSGAN and run inference
def generate_adversarial(
    attack_data: np.ndarray,
    attack_category: str,
    model_name: str,
    unmodified_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate adversarial examples using a saved IDSGAN generator.

    Parameters
    ----------
    attack_data:
        Original malicious traffic records (N × 122).
    attack_category:
        Primary attack category used for the functional mask ("DoS" or "U2R").
    model_name:
        IDS model name; used to locate the correct .pth checkpoint.
    unmodified_mask:
        Optional boolean mask (122,) to override the default functional mask
        (used for robustness evaluation, Section 4.3 of the paper).

    Returns
    -------
    np.ndarray
        Adversarial traffic records (N × 122).
    """
    attack_group = "DoS" if attack_category == "DoS" else "U2R_R2L"
    model_path   = os.path.join(
        config.MODELS_DIR, f"idsgan_{attack_group}_{model_name}.pth"
    )

    # Build IDSGAN shell for inference (ids_model=None; training not needed here)
    idsgan = IDSGAN(attack_category=attack_category, ids_model=None)

    if os.path.exists(model_path):
        idsgan.load(model_path)
    else:
        print(
            f"  WARNING: generator checkpoint not found at {model_path}.\n"
            f"           Using an untrained generator – results will be random."
        )

    if unmodified_mask is not None:
        original_mask        = idsgan.functional_mask
        idsgan.functional_mask = unmodified_mask

    adv_data = idsgan.generate_adversarial(attack_data)

    if unmodified_mask is not None:
        idsgan.functional_mask = original_mask

    return adv_data


# Plotting helpers
def _plot_figure2(results: dict) -> None:
    """Figure 2: original DR vs adversarial DR per IDS model."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    groups = [("DoS", "(a) DoS"), ("U2R_R2L", "(b) U2R & R2L")]
    width  = 0.35

    for ax, (group_name, title) in zip(axes, groups):
        models = [m for m in config.IDS_MODELS if (group_name, m) in results]
        if not models:
            ax.set_title(f"{title} – No Data")
            continue

        orig_drs = [results[(group_name, m)]["orig_dr"] * 100 for m in models]
        adv_drs  = [results[(group_name, m)]["adv_dr"]  * 100 for m in models]
        x        = np.arange(len(models))

        bars1 = ax.bar(x - width / 2, orig_drs, width,
                       label="Original DR",    color="#4C72B0",
                       edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, adv_drs,  width,
                       label="Adversarial DR", color="#DD8452",
                       edgecolor="black", linewidth=0.5)

        ax.set_ylabel("Detection Rate (%)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right")

        for bar in bars1:
            h = bar.get_height()
            if h > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, h - 2,
                        f"{h:.1f}", ha="center", va="top",
                        fontsize=8, color="white")
        for bar in bars2:
            h = bar.get_height()
            if h > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Figure 2: Detection Rate Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    base = os.path.join(config.FIGURES_DIR, "figure_2_detection_rates")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.pdf",           bbox_inches="tight")
    plt.close()
    print("  Saved Figure 2.")


def _plot_figure3(results: dict) -> None:
    """Figure 3: EIR before and after adding unmodified features."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    groups = [("DoS", "(a) DoS"), ("U2R_R2L", "(b) U2R & R2L")]
    width  = 0.35

    for ax, (group_name, title) in zip(axes, groups):
        models = [m for m in config.IDS_MODELS if (group_name, m) in results]
        if not models:
            ax.set_title(f"{title} – No Data")
            continue

        eirs        = [results[(group_name, m)]["eir"]        * 100 for m in models]
        eirs_robust = [results[(group_name, m)]["eir_robust"] * 100 for m in models]
        x           = np.arange(len(models))

        ax.bar(x - width / 2, eirs,        width,
               label="Before adding unmodified",
               color="#55A868", edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, eirs_robust, width,
               label="After adding unmodified",
               color="#C44E52", edgecolor="black", linewidth=0.5)

        ax.set_ylabel("Evasion Increase Rate (%)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 110)
        ax.legend(loc="lower right")

    plt.suptitle("Figure 3: EIR Robustness Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    base = os.path.join(config.FIGURES_DIR, "figure_3_eir_robustness")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.pdf",           bbox_inches="tight")
    plt.close()
    print("  Saved Figure 3.")


# Main evaluation
def main() -> None:
    print("=" * 60)
    print("  IDSGAN Evaluation")
    print("=" * 60)

    # Data
    print("\nLoading preprocessed data...")
    data          = prepare_datasets()
    test_attack_X = data["test_attack_X"]

    # Build per-group test arrays
    test_data_per_group: dict[str, np.ndarray] = {}
    for group_name, group_cats in config.ATTACK_GROUPS.items():
        group_data = [
            test_attack_X[cat]
            for cat in group_cats
            if cat in test_attack_X
        ]
        if group_data:
            test_data_per_group[group_name] = np.concatenate(group_data, axis=0)

    # Table 2
    print("\n" + "=" * 60)
    print("  Table 2: IDSGAN Performance (DoS and U2R & R2L)")
    print("=" * 60)

    table_rows: list[list] = []
    plot_results: dict     = {}

    for group_name in ["DoS", "U2R_R2L"]:
        if group_name not in test_data_per_group:
            print(f"  No test data for {group_name}. Skipping.")
            continue

        X_attack     = test_data_per_group[group_name]
        y_true       = np.ones(len(X_attack), dtype=int)
        primary_cat  = "DoS" if group_name == "DoS" else "U2R"

        for model_name in config.IDS_MODELS:
            model_path = os.path.join(config.MODELS_DIR, f"{model_name}_ids.pkl")
            if not os.path.exists(model_path):
                print(f"  SKIP {model_name}: checkpoint not found (run train_ids.py).")
                continue
            try:
                ids_model = load_ids_model(model_name, model_path)
            except Exception as exc:
                print(f"  SKIP {model_name}: could not load – {exc}.")
                continue

            # Original DR
            orig_preds = predict_ids(ids_model, X_attack)
            orig_dr    = detection_rate(y_true, orig_preds)

            # Scenario ×: only default functional features unmodified
            adv_data = generate_adversarial(X_attack, primary_cat, model_name)
            adv_dr   = detection_rate(y_true, predict_ids(ids_model, adv_data))
            eir      = evasion_increase_rate(orig_dr, adv_dr)

            # Scenario ✓: 50 % of non-functional features additionally frozen
            added_mask      = feature_utils.get_added_unmodified_features(
                primary_cat, fraction=0.5, seed=config.RANDOM_SEED
            )
            adv_data_robust = generate_adversarial(
                X_attack, primary_cat, model_name, unmodified_mask=added_mask
            )
            adv_dr_robust   = detection_rate(y_true, predict_ids(ids_model, adv_data_robust))
            eir_robust      = evasion_increase_rate(orig_dr, adv_dr_robust)

            plot_results[(group_name, model_name)] = dict(
                orig_dr=orig_dr,
                adv_dr=adv_dr,
                eir=eir,
                adv_dr_robust=adv_dr_robust,
                eir_robust=eir_robust,
            )

            table_rows.append([
                group_name, model_name,
                f"{orig_dr       * 100:.2f}%",
                f"{adv_dr        * 100:.2f}%",
                f"{eir           * 100:.2f}%",
                f"{adv_dr_robust * 100:.2f}%",
                f"{eir_robust    * 100:.2f}%",
            ])

    print(tabulate(
        table_rows,
        headers=[
            "Attack", "IDS",
            "Original DR",
            "Adv DR (×)", "EIR (×)",
            "Adv DR (✓)", "EIR (✓)",
        ],
        tablefmt="grid",
    ))

    # Figures
    if plot_results:
        print(f"\nGenerating figures → {config.FIGURES_DIR}")
        _plot_figure2(plot_results)
        _plot_figure3(plot_results)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
