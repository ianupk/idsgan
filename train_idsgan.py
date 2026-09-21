"""
Train IDSGAN adversarial generator for each (attack group, IDS model) pair.

Run train_ids.py first to produce the black-box IDS checkpoints that
IDSGAN uses as its discriminator signal.

Usage
-----
    # Train all 14 combinations (2 attack groups × 7 IDS models)
    python train_idsgan.py

    # Single attack group against one IDS model
    python train_idsgan.py --attack dos --ids_model MLP

    # Custom epoch count
    python train_idsgan.py --epochs 50

Arguments
---------
    --attack    {dos, u2r_r2l, all}   Attack group to train for (default: all)
    --ids_model {SVM,NB,MLP,LR,DT,RF,KNN,all}
                                       Target IDS model (default: all)
    --epochs    int                    Training epochs (default: config.EPOCHS)

Output
------
    results/models/idsgan_<ATTACK>_<MODEL>.pth  generator + discriminator weights
"""

import argparse
import os

import numpy as np

from src.config import (
    ATTACK_GROUPS,
    BATCH_SIZE,
    EPOCHS,
    IDS_MODELS,
    MODELS_DIR,
)
from src.data_preprocessing import prepare_datasets
from src.ids_models import load_ids_model
from src.idsgan import IDSGAN

# ---------------------------------------------------------------------------
# CLI → internal attack-group name mapping
# Keys match the --attack argument values; values match ATTACK_GROUPS keys.
# ---------------------------------------------------------------------------
_ATTACK_CLI_MAP: dict[str, list[str]] = {
    "dos":     ["DoS"],
    "u2r_r2l": ["U2R_R2L"],
    "all":     list(ATTACK_GROUPS.keys()),  # ["DoS", "U2R_R2L"]
}

# Map each ATTACK_GROUP key to the primary attack category name used for
# the functional-feature mask inside the Generator.
_PRIMARY_ATTACK: dict[str, str] = {
    "DoS":     "DoS",
    "U2R_R2L": "U2R",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train IDSGAN adversarial generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="all",
        choices=list(_ATTACK_CLI_MAP.keys()),
        help="Attack group to train for.",
    )
    parser.add_argument(
        "--ids_model",
        type=str,
        default="all",
        choices=IDS_MODELS + ["all"],
        help="Black-box IDS model to target.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    attacks_to_train = _ATTACK_CLI_MAP[args.attack]
    models_to_train  = IDS_MODELS if args.ids_model == "all" else [args.ids_model]

    print("=" * 60)
    print("  IDSGAN Training")
    print("=" * 60)
    print(f"  Attack groups : {attacks_to_train}")
    print(f"  IDS models    : {models_to_train}")
    print(f"  Epochs        : {args.epochs}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    print("\nLoading and preprocessing data...")
    data = prepare_datasets()

    normal_data = data["gan_normal_X"]

    # Concatenate per-group attack arrays
    attack_data_dict: dict[str, np.ndarray] = {}
    for group_name, group_cats in ATTACK_GROUPS.items():
        group_data = [
            data["gan_attack_X"][cat]
            for cat in group_cats
            if cat in data["gan_attack_X"]
        ]
        if group_data:
            attack_data_dict[group_name] = np.concatenate(group_data, axis=0)

    print(f"  Normal data        : {normal_data.shape}")
    for k, v in attack_data_dict.items():
        print(f"  Attack data [{k:8s}]: {v.shape}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for attack_group in attacks_to_train:
        if attack_group not in attack_data_dict:
            print(f"\nWARNING: No attack data found for '{attack_group}'. Skipping.")
            continue

        attack_data   = attack_data_dict[attack_group]
        primary_cat   = _PRIMARY_ATTACK[attack_group]

        for model_name in models_to_train:
            print(f"\n{'=' * 60}")
            print(f"  [{attack_group} vs {model_name}]")
            print(f"{'=' * 60}")

            # Load pre-trained black-box IDS
            model_path = os.path.join(MODELS_DIR, f"{model_name}_ids.pkl")
            if not os.path.exists(model_path):
                print(
                    f"  ERROR: {model_path} not found.\n"
                    f"  Please run 'python train_ids.py' first."
                )
                continue
            try:
                ids_model = load_ids_model(model_name, model_path)
                print(f"  Loaded {model_name} IDS checkpoint.")
            except Exception as exc:
                print(f"  ERROR: Could not load {model_name}: {exc}")
                continue

            # Train
            idsgan = IDSGAN(attack_category=primary_cat, ids_model=ids_model)
            idsgan.train(normal_data, attack_data, args.epochs, BATCH_SIZE)

            # Save
            save_path = os.path.join(MODELS_DIR, f"idsgan_{attack_group}_{model_name}.pth")
            idsgan.save(save_path)
            print(f"  Saved generator checkpoint → {save_path}")

            # Quick validation on held-out test samples
            test_attack = data.get("test_attack_X", {})
            test_samples = [
                test_attack[cat]
                for cat in ATTACK_GROUPS[attack_group]
                if cat in test_attack
            ]
            if test_samples:
                X_test = np.concatenate(test_samples, axis=0)
                sample  = X_test[: min(100, len(X_test))]
                adv_sample = idsgan.generate_adversarial(sample)

                orig_dr = float(np.mean(ids_model.predict(sample) == 1))
                adv_dr  = float(np.mean(ids_model.predict(adv_sample) == 1))

                print(f"  Validation ({len(sample)} samples):")
                print(f"    Original DR   : {orig_dr * 100:.2f}%")
                print(f"    Adversarial DR: {adv_dr  * 100:.2f}%")

    print("\n" + "=" * 60)
    print("  Training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
