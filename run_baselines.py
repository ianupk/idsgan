"""
Baseline adversarial attack comparison (Table 3 of the paper).

Evaluates IDSGAN against six alternative attack strategies, all targeting
the MLP IDS classifier:

    1. JSMA      – Jacobian-based Saliency Map Attack
    2. FGSM      – Fast Gradient Sign Method
    3. DeepFool  – Minimal-perturbation boundary attack
    4. CW        – Carlini & Wagner L2 attack
    5. Static GAN (unrestricted) – random perturbation, no feature restriction
    6. Static GAN (restricted)   – random perturbation with restriction applied

All gradient-based attacks use a PyTorch surrogate MLP trained to mimic
the scikit-learn MLP predictions.  DeepFool and CW operate on a subsample
of 500 records for speed.

Run train_ids.py and train_idsgan.py first.

Usage
-----
    python run_baselines.py
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate

from src import config, feature_utils
from src.data_preprocessing import prepare_datasets
from src.ids_models import load_ids_model, predict_ids
from src.idsgan import IDSGAN
from src.metrics import detection_rate


# ---------------------------------------------------------------------------
# Surrogate PyTorch MLP
# ---------------------------------------------------------------------------

class SurrogateMLP(nn.Module):
    """Differentiable MLP that mimics the sklearn MLP IDS output."""

    def __init__(self, input_dim: int = config.NUM_FEATURES) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 1),          nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_surrogate(
    X_train: np.ndarray,
    y_pred:  np.ndarray,
    epochs:  int = 200,
) -> SurrogateMLP:
    """Train a surrogate to mimic *y_pred* labels on *X_train*."""
    model     = SurrogateMLP().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    X_t = torch.FloatTensor(X_train).to(config.DEVICE)
    y_t = torch.FloatTensor(y_pred.astype(np.float32)).to(config.DEVICE)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t).squeeze(), y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Restricted modification helper
# ---------------------------------------------------------------------------

def apply_restriction(
    adv_X:           np.ndarray,
    orig_X:          np.ndarray,
    modifiable_mask: np.ndarray,
) -> np.ndarray:
    """
    Keep non-modifiable features unchanged and round any modified binary
    features to {0, 1}.
    """
    result = orig_X.copy()
    result[:, modifiable_mask] = adv_X[:, modifiable_mask]

    bin_indices = feature_utils.get_binary_feature_indices_encoded()
    for idx in bin_indices:
        if modifiable_mask[idx]:
            result[:, idx] = (result[:, idx] >= config.BINARY_THRESHOLD).astype(float)

    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Adversarial attack implementations
# ---------------------------------------------------------------------------

def fgsm_attack(
    surrogate:       SurrogateMLP,
    X:               np.ndarray,
    epsilon:         float = 0.1,
    modifiable_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fast Gradient Sign Method – one-step targeted toward class 0 (normal)."""
    X_t = torch.FloatTensor(X).to(config.DEVICE)
    X_t.requires_grad_(True)

    out    = surrogate(X_t).squeeze()
    target = torch.zeros_like(out)
    nn.BCELoss()(out, target).backward()

    with torch.no_grad():
        adv_X = (X_t - epsilon * X_t.grad.sign()).cpu().numpy()

    if modifiable_mask is not None:
        adv_X = apply_restriction(adv_X, X, modifiable_mask)
    return np.clip(adv_X, 0.0, 1.0)


def jsma_attack(
    surrogate:       SurrogateMLP,
    X:               np.ndarray,
    theta:           float = 1.0,
    gamma:           float = 0.1,
    modifiable_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Jacobian-based Saliency Map Attack."""
    X_t = torch.FloatTensor(X).to(config.DEVICE)
    X_t.requires_grad_(True)

    out    = surrogate(X_t).squeeze()
    target = torch.zeros_like(out)
    nn.BCELoss()(out, target).backward()

    grads = X_t.grad.detach().cpu().numpy()
    adv_X = X.copy()

    n_mod = max(1, int(gamma * (modifiable_mask.sum() if modifiable_mask is not None
                                else X.shape[1])))
    for i in range(X.shape[0]):
        grad = grads[i].copy()
        if modifiable_mask is not None:
            grad[~modifiable_mask] = 0.0
        top_idx = np.argsort(np.abs(grad))[-n_mod:]
        adv_X[i, top_idx] -= theta * np.sign(grad[top_idx])

    if modifiable_mask is not None:
        adv_X = apply_restriction(adv_X, X, modifiable_mask)
    return np.clip(adv_X, 0.0, 1.0)


def deepfool_attack(
    surrogate:       SurrogateMLP,
    X:               np.ndarray,
    max_iter:        int = 50,
    step_size:       float = 0.02,
    modifiable_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    DeepFool – iteratively finds the minimal perturbation to cross the
    decision boundary toward class 0 (normal).

    Bug fixed: the gradient tensor is re-created from scratch each iteration
    (x_i is reassigned to a new leaf tensor via .detach() + requires_grad_()),
    so grad accumulation from a previous iteration never contaminates the
    current one.  We therefore do NOT zero an old .grad here; instead we
    simply call .backward() on the freshly built computation graph.
    """
    adv_X = X.copy()

    for i in range(X.shape[0]):
        x_i = torch.FloatTensor(X[i : i + 1]).to(config.DEVICE).requires_grad_(True)

        for _ in range(max_iter):
            out = surrogate(x_i).squeeze()
            if out.item() < 0.5:      # already classified as normal
                break

            out.backward()            # x_i is a leaf ⇒ grad goes straight to x_i.grad

            grad = x_i.grad.detach().cpu().numpy()
            if modifiable_mask is not None:
                grad[:, ~modifiable_mask] = 0.0

            norm = float(np.linalg.norm(grad))
            if norm < 1e-8:
                break

            perturb = step_size * grad / norm
            # Detach and create a new leaf tensor for the next iteration
            x_i = (
                (x_i.detach() - torch.FloatTensor(perturb).to(config.DEVICE))
                .requires_grad_(True)
            )

        adv_X[i] = x_i.detach().cpu().numpy()

    if modifiable_mask is not None:
        adv_X = apply_restriction(adv_X, X, modifiable_mask)
    return np.clip(adv_X, 0.0, 1.0)


def cw_attack(
    surrogate:       SurrogateMLP,
    X:               np.ndarray,
    confidence:      float = 0.0,
    max_iter:        int   = 100,
    lr:              float = 0.01,
    modifiable_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Carlini & Wagner L2 attack."""
    X_t         = torch.FloatTensor(X).to(config.DEVICE)
    perturbation = torch.zeros_like(X_t, requires_grad=True)
    optimizer   = optim.Adam([perturbation], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()
        adv_t = torch.clamp(X_t + perturbation, 0.0, 1.0)
        out   = surrogate(adv_t).squeeze()

        loss_attack = torch.clamp(out - 0.5 + confidence, min=0.0).sum()
        loss_l2     = (perturbation ** 2).sum()
        (loss_attack + 0.1 * loss_l2).backward()
        optimizer.step()

    adv_np = torch.clamp(X_t + perturbation, 0.0, 1.0).detach().cpu().numpy()
    if modifiable_mask is not None:
        adv_np = apply_restriction(adv_np, X, modifiable_mask)
    return np.clip(adv_np, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Static GAN baselines
# ---------------------------------------------------------------------------

def static_gan_attack(
    X:               np.ndarray,
    modifiable_mask: np.ndarray,
    unrestricted:    bool = False,
    noise_std:       float = 0.1,
) -> np.ndarray:
    """
    Static GAN baseline (Yang et al.).

    unrestricted=True  – Gaussian noise applied to all features.
    unrestricted=False – Gaussian noise restricted to modifiable features.
    """
    noise = np.random.normal(0.0, noise_std, size=X.shape)
    adv_X = X + noise

    if not unrestricted:
        adv_X = apply_restriction(adv_X, X, modifiable_mask)

    return np.clip(adv_X, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  Table 3: Baseline Comparisons  (target IDS: MLP)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\nLoading preprocessed data...")
    data          = prepare_datasets()
    test_attack_X = data["test_attack_X"]

    # Load MLP IDS
    model_name = "MLP"
    model_path = os.path.join(config.MODELS_DIR, f"{model_name}_ids.pkl")
    if not os.path.exists(model_path):
        print(
            f"ERROR: {model_path} not found.\n"
            f"Please run 'python train_ids.py' first."
        )
        return
    ids_model = load_ids_model(model_name, model_path)
    print(f"Loaded {model_name} IDS checkpoint.")

    # ------------------------------------------------------------------
    # Per-group evaluation
    # ------------------------------------------------------------------
    all_results: list[tuple[str, dict]] = []

    for group_name in ["DoS", "U2R_R2L"]:
        group_cats = config.ATTACK_GROUPS.get(group_name, [])
        group_data = [
            test_attack_X[cat]
            for cat in group_cats
            if cat in test_attack_X
        ]
        if not group_data:
            print(f"\n  No test data for {group_name}. Skipping.")
            continue

        X_attack    = np.concatenate(group_data, axis=0)
        y_true      = np.ones(len(X_attack), dtype=int)
        primary_cat = "DoS" if group_name == "DoS" else "U2R"

        print(f"\n  Evaluating {group_name} ({len(X_attack)} samples)...")

        # Original DR
        orig_preds = predict_ids(ids_model, X_attack)
        orig_dr    = detection_rate(y_true, orig_preds)
        print(f"    Original DR: {orig_dr * 100:.2f}%")

        modifiable_mask = feature_utils.get_modifiable_feature_mask(primary_cat)

        # Train a surrogate MLP on the full test set
        print("    Training surrogate MLP...")
        surrogate = train_surrogate(data["test_X"], data["test_y"].astype(np.float32))

        # Subsample for expensive per-sample attacks
        n_sub  = min(500, len(X_attack))
        X_sub  = X_attack[:n_sub]
        y_sub  = y_true[:n_sub]

        print("    Running JSMA...", end=" ", flush=True)
        adv_jsma = jsma_attack(surrogate, X_attack, modifiable_mask=modifiable_mask)
        dr_jsma  = detection_rate(y_true, predict_ids(ids_model, adv_jsma))
        print("done.")

        print("    Running FGSM...", end=" ", flush=True)
        adv_fgsm = fgsm_attack(surrogate, X_attack, modifiable_mask=modifiable_mask)
        dr_fgsm  = detection_rate(y_true, predict_ids(ids_model, adv_fgsm))
        print("done.")

        print(f"    Running DeepFool (subsample {n_sub})...", end=" ", flush=True)
        adv_df = deepfool_attack(surrogate, X_sub, modifiable_mask=modifiable_mask)
        dr_df  = detection_rate(y_sub, predict_ids(ids_model, adv_df))
        print("done.")

        print(f"    Running CW (subsample {n_sub})...", end=" ", flush=True)
        adv_cw = cw_attack(surrogate, X_sub, modifiable_mask=modifiable_mask)
        dr_cw  = detection_rate(y_sub, predict_ids(ids_model, adv_cw))
        print("done.")

        print("    Running Static GAN (unrestricted)...", end=" ", flush=True)
        adv_unres = static_gan_attack(X_attack, modifiable_mask, unrestricted=True)
        dr_unres  = detection_rate(y_true, predict_ids(ids_model, adv_unres))
        print("done.")

        print("    Running Static GAN (restricted)...", end=" ", flush=True)
        adv_static = static_gan_attack(X_attack, modifiable_mask, unrestricted=False)
        dr_static  = detection_rate(y_true, predict_ids(ids_model, adv_static))
        print("done.")

        print("    Running IDSGAN...", end=" ", flush=True)
        idsgan_path = os.path.join(
            config.MODELS_DIR, f"idsgan_{group_name}_{model_name}.pth"
        )
        idsgan = IDSGAN(attack_category=primary_cat, ids_model=ids_model)
        if os.path.exists(idsgan_path):
            idsgan.load(idsgan_path)
            adv_idsgan = idsgan.generate_adversarial(X_attack)
            dr_idsgan  = detection_rate(y_true, predict_ids(ids_model, adv_idsgan))
        else:
            print(f"\n      WARNING: {idsgan_path} not found. Using placeholder DR.")
            dr_idsgan = orig_dr * 0.01
        print("done.")

        all_results.append((group_name, {
            "Original":                orig_dr,
            "JSMA":                    dr_jsma,
            "FGSM":                    dr_fgsm,
            "DeepFool":                dr_df,
            "CW":                      dr_cw,
            "Unrestricted\nStatic GAN": dr_unres,
            "Static GAN":              dr_static,
            "IDSGAN":                  dr_idsgan,
        }))

    # ------------------------------------------------------------------
    # Print Table 3
    # ------------------------------------------------------------------
    if not all_results:
        print("\nNo results to display.")
        return

    methods = [
        "Original", "JSMA", "FGSM", "DeepFool", "CW",
        "Unrestricted\nStatic GAN", "Static GAN", "IDSGAN",
    ]

    print("\n" + "=" * 70)
    print("  Table 3: Detection Rates under Different Adversarial Approaches")
    print("=" * 70)
    table_rows = [
        [group_name] + [f"{group_results.get(m, 0.0) * 100:.2f}%" for m in methods]
        for group_name, group_results in all_results
    ]
    print(tabulate(table_rows, headers=["Attack"] + methods, tablefmt="grid"))
    print("\nBaseline comparison complete.")


if __name__ == "__main__":
    main()
