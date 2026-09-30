"""
Train all seven black-box IDS classifiers on the NSL-KDD dataset.

The trained models are saved to results/models/ as .pkl files and are
required before running train_idsgan.py, evaluate.py, or run_baselines.py.

Usage
-----
    python train_ids.py

Output
------
    results/models/<MODEL>_ids.pkl  for each of SVM NB MLP LR DT RF KNN
"""

import os

import numpy as np
from tabulate import tabulate

from src.config import IDS_MODELS, MODELS_DIR, ATTACK_GROUPS
from src.data_preprocessing import prepare_datasets
from src.ids_models import (
    create_ids_model,
    evaluate_ids,
    predict_ids,
    save_ids_model,
    train_ids_model,
)
from src.metrics import detection_rate


def main() -> None:
    print("=" * 60)
    print("  Training Black-Box IDS Models")
    print("=" * 60)

    # Data loading
    print("\nLoading and preprocessing data...")
    data = prepare_datasets()

    ids_train_X      = data["ids_train_X"]
    ids_train_y      = data["ids_train_y_bin"]
    test_X           = data["test_X"]
    test_y           = data["test_y_bin"]
    test_categories  = data["test_y_multi"]

    # Train each model
    overall_results: list[list] = []
    category_dr: dict[str, list[float]] = {g: [] for g in ATTACK_GROUPS}

    print(f"\nTraining {len(IDS_MODELS)} IDS models...\n")

    for model_name in IDS_MODELS:
        print(f"  [{model_name}] Training ...", end=" ", flush=True)

        model = create_ids_model(model_name)
        model = train_ids_model(model, ids_train_X, ids_train_y)

        # Overall classification metrics
        metrics = evaluate_ids(model, test_X, test_y)
        overall_results.append([
            model_name,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1']:.4f}",
        ])

        # Per-category detection rates (used in Table 2)
        preds = predict_ids(model, test_X)
        for group_name, group_cats in ATTACK_GROUPS.items():
            mask = np.isin(test_categories, group_cats)
            if np.any(mask):
                group_y_true = np.ones(int(mask.sum()), dtype=int)
                group_y_pred = preds[mask]
                dr = detection_rate(group_y_true, group_y_pred)
            else:
                dr = 0.0
            category_dr[group_name].append(dr)

        # Persist
        save_path = os.path.join(MODELS_DIR, f"{model_name}_ids.pkl")
        save_ids_model(model, model_name, save_path)
        print("done.")

    # Print results
    print("\n" + "=" * 60)
    print("  Overall IDS Model Performance")
    print("=" * 60)
    print(tabulate(
        overall_results,
        headers=["Model", "Accuracy", "Precision", "Recall", "F1"],
        tablefmt="grid",
    ))

    print("\n" + "=" * 60)
    print("  Original Detection Rates per Attack Group (Table 2 – rows 1–2)")
    print("=" * 60)
    cat_table = [
        [model_name] + [f"{category_dr[g][i] * 100:.2f}%" for g in ATTACK_GROUPS]
        for i, model_name in enumerate(IDS_MODELS)
    ]
    print(tabulate(
        cat_table,
        headers=["Model"] + list(ATTACK_GROUPS.keys()),
        tablefmt="grid",
    ))

    print(f"\nAll models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
