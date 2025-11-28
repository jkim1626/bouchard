# evaluate_noise_classifier.py

from noise_classifier import estimate_noise_confidence
import numpy as np
from pathlib import Path


data_dir = Path("synthetic_mixed_noise")
files = sorted(data_dir.glob("*.npz"))

# accumulators
mse_list = []
mae_list = []
rmse_list = []
per_comp_sq_errors = []

argmax_correct = 0
seen_files = 0

# per-true-dominant-type stats
types = ["additive", "multiplicative", "jump"]
true_type_counts = {t: 0 for t in types}
true_type_correct = {t: 0 for t in types}

for f in files:
    d = np.load(f)
    noisy = d["noisy"]

    # true ratios as [additive, multiplicative, jump]
    true = np.array([
        float(d["noise_ratio_add"]),
        float(d["noise_ratio_mult"]),
        float(d["noise_ratio_jump"]),
    ], dtype=float)

    result = estimate_noise_confidence(noisy)
    weights = result["weights"]

    # estimated weights in the same order
    weights_est = np.array([
        weights["additive"],
        weights["multiplicative"],
        weights["jump"],
    ], dtype=float)

    if true.size != 3 or weights_est.size != 3:
        continue

    seen_files += 1

    # per-component squared error
    se = (weights_est - true) ** 2
    per_comp_sq_errors.append(se)

    mse = float(np.mean(se))
    mae = float(np.mean(np.abs(weights_est - true)))
    rmse = float(np.sqrt(mse))

    mse_list.append(mse)
    mae_list.append(mae)
    rmse_list.append(rmse)

    # argmax (predominant component) accuracy
    true_idx = int(np.argmax(true))
    est_idx = int(np.argmax(weights_est))
    true_name = types[true_idx]

    true_type_counts[true_name] += 1
    if est_idx == true_idx:
        argmax_correct += 1
        true_type_correct[true_name] += 1

# ---------------- SUMMARY PRINTING ---------------- #

if seen_files == 0:
    print("No files found in synthetic_mixed_noise. Please run generate_mixed_noise_signals.py first.")
else:
    per_comp_sq_errors = np.vstack(per_comp_sq_errors)

    avg_mse = float(np.mean(mse_list))
    std_mse = float(np.std(mse_list))
    avg_mae = float(np.mean(mae_list))
    std_mae = float(np.std(mae_list))
    avg_rmse = float(np.mean(rmse_list))
    std_rmse = float(np.std(rmse_list))

    mse_per_component = np.mean(per_comp_sq_errors, axis=0)  # [add, mult, jump]

    overall_argmax_acc = argmax_correct / float(seen_files)

    print("==============================================")
    print(" Noise-type confidence evaluation summary")
    print(" (updated classifier with log-domain features)")
    print("==============================================")
    print(f"Total files evaluated: {seen_files}")
    print()
    print("Overall regression metrics (weights vs true ratios)")
    print("--------------------------------------------------")
    print(f"  MSE   : {avg_mse:.4f} ± {std_mse:.44f}")
    print(f"  MAE   : {avg_mae:.4f} ± {std_mae:.4f}")
    print(f"  RMSE  : {avg_rmse:.4f} ± {std_rmse:.4f}")
    print()
    print("Per-component MSE")
    print("-----------------")
    print(f"  additive       : {mse_per_component[0]:.4f}")
    print(f"  multiplicative : {mse_per_component[1]:.4f}")
    print(f"  jump           : {mse_per_component[2]:.4f}")
    print()
    print("Argmax (dominant noise type) classification")
    print("-------------------------------------------")
    print(f"  Overall accuracy: {overall_argmax_acc*100:.2f}%")
    print()
    print("  Accuracy by TRUE dominant type:")
    for t in types:
        n = true_type_counts[t]
        c = true_type_correct[t]
        if n == 0:
            acc = 0.0
        else:
            acc = c / float(n)
        print(f"    {t:<13}: {c:4d} / {n:4d}  ({acc*100:6.2f}%)")
    print("==============================================")
