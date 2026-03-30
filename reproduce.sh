#!/bin/bash
# Reproduce all paper experiments.
#
# Usage:
#   bash reproduce.sh                        # core experiments (~12h on 1x GPU)
#   bash reproduce.sh --ecg-only             # ECG experiments only (~4h)
#   bash reproduce.sh --cifar-only           # CIFAR experiments only (~8h)
#   bash reproduce.sh --quick                # Quick sanity check (~30min)
#   bash reproduce.sh --baseline-n20         # 20-seed Baseline comparison (~10h)
#   bash reproduce.sh --basis-comparison     # Basis comparison (4×5 seeds, ~6h)
#   bash reproduce.sh --golden-baselines     # Golden deterministic baselines (~1h)
#   bash reproduce.sh --ood-eval             # OOD detection evaluation (~30min)
#   bash reproduce.sh --all                  # Everything (~30h)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-data}"

# Flags
ECG_ONLY=false
CIFAR_ONLY=false
QUICK=false
BASELINE_N20=false
BASIS_COMPARISON=false
GOLDEN_BASELINES=false
OOD_EVAL=false
ALL=false

for arg in "$@"; do
    case $arg in
        --ecg-only) ECG_ONLY=true ;;
        --cifar-only) CIFAR_ONLY=true ;;
        --quick) QUICK=true ;;
        --baseline-n20) BASELINE_N20=true ;;
        --basis-comparison) BASIS_COMPARISON=true ;;
        --golden-baselines) GOLDEN_BASELINES=true ;;
        --ood-eval) OOD_EVAL=true ;;
        --all) ALL=true ;;
    esac
done

if [ "$ALL" = true ]; then
    BASELINE_N20=true
    BASIS_COMPARISON=true
    GOLDEN_BASELINES=true
    OOD_EVAL=true
fi

# If any specific flag is set, skip core experiments unless --all or no flags
SPECIFIC_FLAG=false
if [ "$BASELINE_N20" = true -o "$BASIS_COMPARISON" = true -o "$GOLDEN_BASELINES" = true -o "$OOD_EVAL" = true ]; then
    SPECIFIC_FLAG=true
fi
RUN_CORE=true
if [ "$SPECIFIC_FLAG" = true ] && [ "$ALL" = false ]; then
    RUN_CORE=false
fi

# Seeds used across all multi-seed experiments
SEEDS_5="42 123 456 789 1337"
SEEDS_20="2 3 7 11 13 17 19 23 29 31 37 41 42 43 47 53 123 456 789 1337"

echo "============================================"
echo "RepNet Paper Reproduction"
echo "$(date)"
echo "Device: $(${PYTHON} -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
echo "Flags: ecg_only=$ECG_ONLY cifar_only=$CIFAR_ONLY quick=$QUICK"
echo "       baseline_n20=$BASELINE_N20 basis=$BASIS_COMPARISON golden=$GOLDEN_BASELINES ood=$OOD_EVAL"
echo "============================================"

mkdir -p results

# ── 1. Verify determinism ──────────────────────────────────
if [ "$CIFAR_ONLY" = false ] && [ "$RUN_CORE" = true ]; then
    echo ""
    echo "=== Step 1: Verify bit-identical determinism ==="
    VERIFY_EPOCHS=5
    if [ "$QUICK" = true ]; then VERIFY_EPOCHS=2; fi
    $PYTHON experiments/verify_determinism.py \
        --model conformer --data-dir "$DATA_DIR/ptb-xl" --epochs $VERIFY_EPOCHS
fi

# ── 2. ECG experiments ─────────────────────────────────────
if [ "$CIFAR_ONLY" = false ] && [ "$RUN_CORE" = true ]; then
    ECG_EPOCHS=85
    if [ "$QUICK" = true ]; then ECG_EPOCHS=10; fi

    echo ""
    echo "=== Step 2a: Conformer mixed-basis multi-seed (5 seeds) ==="
    for seed in $SEEDS_5; do
        $PYTHON experiments/train_ecg.py \
            --model conformer --name conformer_mixed_s${seed} --data-dir "$DATA_DIR/ptb-xl" \
            --epochs $ECG_EPOCHS --seed $seed --mixed-bases --class-weight sqrt
    done

    echo ""
    echo "=== Step 2b: Conformer Kaiming multi-seed (5 seeds) ==="
    for seed in $SEEDS_5; do
        $PYTHON experiments/train_ecg.py \
            --model conformer --name conformer_kaiming_s${seed} --data-dir "$DATA_DIR/ptb-xl" \
            --epochs $ECG_EPOCHS --seed $seed --no-det-init --class-weight sqrt
    done

    echo ""
    echo "=== Step 2c: Conformer (golden ratio, mixed-basis) ==="
    $PYTHON experiments/train_ecg.py \
        --model conformer --name conformer_mixed_golden --data-dir "$DATA_DIR/ptb-xl" \
        --epochs $ECG_EPOCHS --batch-order golden --mixed-bases --class-weight sqrt

    echo ""
    echo "=== Step 2d: Baseline CNN (5 seeds, mixed-basis + Kaiming) ==="
    for seed in $SEEDS_5; do
        $PYTHON experiments/train_ecg.py \
            --model baseline --name baseline_mixed_s${seed} --data-dir "$DATA_DIR/ptb-xl" \
            --epochs $ECG_EPOCHS --seed $seed --mixed-bases --class-weight sqrt
        $PYTHON experiments/train_ecg.py \
            --model baseline --name baseline_kaiming_s${seed} --data-dir "$DATA_DIR/ptb-xl" \
            --epochs $ECG_EPOCHS --seed $seed --no-det-init --class-weight sqrt
    done

    echo ""
    echo "=== Step 2e: Cross-validation (Conformer) ==="
    $PYTHON experiments/cross_validate.py \
        --model conformer --data-dir "$DATA_DIR/ptb-xl" --epochs $ECG_EPOCHS --mixed-bases

    echo ""
    echo "=== Step 2f: OOD detection ==="
    CKPTS=""
    for f in best_conformer_mixed_golden.pt best_baseline_mixed_s42.pt best_conformer_dct_s42.pt; do
        if [ -f "$f" ]; then
            [ -n "$CKPTS" ] && CKPTS="$CKPTS,"
            CKPTS="$CKPTS$f"
        fi
    done
    if [ -n "$CKPTS" ]; then
        OOD_ARGS="--checkpoints $CKPTS --data-dir $DATA_DIR/ptb-xl"
        if [ -d "$DATA_DIR/nstdb" ]; then
            OOD_ARGS="$OOD_ARGS --nstdb-path $DATA_DIR/nstdb"
        fi
        $PYTHON experiments/eval_ood.py $OOD_ARGS
    else
        echo "  No checkpoints found for OOD eval (skipping)"
    fi
fi

# ── 3. Conformer n=20 (full multi-seed) ───────────────────
if [ "$CIFAR_ONLY" = false ] && [ "$BASELINE_N20" = true -o "$ALL" = true ]; then
    ECG_EPOCHS=85
    if [ "$QUICK" = true ]; then ECG_EPOCHS=10; fi

    echo ""
    echo "=== Step 3a: Conformer mixed-basis multi-seed (20 seeds) ==="
    for seed in $SEEDS_20; do
        $PYTHON experiments/train_ecg.py \
            --model conformer --name conformer_mixed_s${seed} --data-dir "$DATA_DIR/ptb-xl" \
            --epochs $ECG_EPOCHS --seed $seed --mixed-bases --class-weight sqrt
    done

    echo ""
    echo "=== Step 3b: Conformer Kaiming multi-seed (20 seeds) ==="
    for seed in $SEEDS_20; do
        $PYTHON experiments/train_ecg.py \
            --model conformer --name conformer_kaiming_s${seed} --data-dir "$DATA_DIR/ptb-xl" \
            --epochs $ECG_EPOCHS --seed $seed --no-det-init --class-weight sqrt
    done
fi

# ── 4. Baseline n=20 ──────────────────────────────────────
if [ "$CIFAR_ONLY" = false ] && [ "$BASELINE_N20" = true -o "$ALL" = true ]; then
    ECG_EPOCHS=85
    if [ "$QUICK" = true ]; then ECG_EPOCHS=10; fi

    echo ""
    echo "=== Step 4: Baseline CNN multi-seed (20 seeds) ==="
    for seed in $SEEDS_20; do
        $PYTHON experiments/train_ecg.py \
            --model baseline --name baseline_mixed_s${seed} --data-dir "$DATA_DIR/ptb-xl" \
            --epochs $ECG_EPOCHS --seed $seed --mixed-bases --class-weight sqrt
        $PYTHON experiments/train_ecg.py \
            --model baseline --name baseline_kaiming_s${seed} --data-dir "$DATA_DIR/ptb-xl" \
            --epochs $ECG_EPOCHS --seed $seed --no-det-init --class-weight sqrt
    done
fi

# ── 5. Basis comparison ───────────────────────────────────
if [ "$CIFAR_ONLY" = false ] && [ "$BASIS_COMPARISON" = true -o "$ALL" = true ]; then
    ECG_EPOCHS=85
    if [ "$QUICK" = true ]; then ECG_EPOCHS=10; fi

    echo ""
    echo "=== Step 5: Basis comparison (Conformer, 5 seeds × 4 bases) ==="
    for basis in dct hadamard hartley sinusoidal; do
        for seed in $SEEDS_5; do
            $PYTHON experiments/train_ecg.py \
                --model conformer --name conformer_${basis}_s${seed} \
                --data-dir "$DATA_DIR/ptb-xl" \
                --basis $basis --seed $seed --epochs $ECG_EPOCHS --class-weight sqrt
        done
    done
fi

# ── 6. Golden deterministic baselines ─────────────────────
if [ "$CIFAR_ONLY" = false ] && [ "$GOLDEN_BASELINES" = true -o "$ALL" = true ]; then
    ECG_EPOCHS=85
    if [ "$QUICK" = true ]; then ECG_EPOCHS=10; fi

    echo ""
    echo "=== Step 6: Golden deterministic baselines ==="

    # Conformer mixed-basis golden (fully deterministic)
    $PYTHON experiments/train_ecg.py --model conformer --name conformer_mixed_golden \
        --data-dir "$DATA_DIR/ptb-xl" \
        --batch-order golden --epochs $ECG_EPOCHS --mixed-bases --class-weight sqrt

    # Conformer Kaiming golden (essential control)
    $PYTHON experiments/train_ecg.py --model conformer --name conformer_kaiming_golden \
        --data-dir "$DATA_DIR/ptb-xl" \
        --no-det-init --seed 42 --batch-order golden --epochs $ECG_EPOCHS --class-weight sqrt

    # Baseline mixed-basis golden
    $PYTHON experiments/train_ecg.py --model baseline --name baseline_mixed_golden \
        --data-dir "$DATA_DIR/ptb-xl" \
        --batch-order golden --epochs $ECG_EPOCHS --mixed-bases --class-weight sqrt

    # Baseline Kaiming golden
    $PYTHON experiments/train_ecg.py --model baseline --name baseline_kaiming_golden \
        --data-dir "$DATA_DIR/ptb-xl" \
        --no-det-init --seed 42 --batch-order golden --epochs $ECG_EPOCHS --class-weight sqrt
fi

# ── 7. OOD evaluation (2×2 design) ───────────────────────
if [ "$CIFAR_ONLY" = false ] && [ "$OOD_EVAL" = true -o "$ALL" = true ]; then
    echo ""
    echo "=== Step 7: OOD evaluation (2×2 design) ==="
    GOLDEN_CKPTS=""
    for f in best_conformer_mixed_golden.pt best_conformer_kaiming_golden.pt \
             best_baseline_mixed_golden.pt best_baseline_kaiming_golden.pt; do
        if [ -f "$f" ]; then
            [ -n "$GOLDEN_CKPTS" ] && GOLDEN_CKPTS="$GOLDEN_CKPTS,"
            GOLDEN_CKPTS="$GOLDEN_CKPTS$f"
        fi
    done
    if [ -n "$GOLDEN_CKPTS" ]; then
        OOD_ARGS="--checkpoints $GOLDEN_CKPTS --data-dir $DATA_DIR/ptb-xl"
        if [ -d "$DATA_DIR/nstdb" ]; then
            OOD_ARGS="$OOD_ARGS --nstdb-path $DATA_DIR/nstdb"
        fi
        $PYTHON experiments/eval_ood.py $OOD_ARGS
    else
        echo "  No golden checkpoints found (run --golden-baselines first)"
    fi
fi

# ── 8. MedMNIST experiments ──────────────────────────────
if [ "$ECG_ONLY" = false ] && [ "$CIFAR_ONLY" = false ] && [ "$RUN_CORE" = true ]; then
    MEDMNIST_EPOCHS=200
    if [ "$QUICK" = true ]; then MEDMNIST_EPOCHS=20; fi

    echo ""
    echo "=== Step 8: MedMNIST (DCT vs Kaiming, 5 seeds) ==="
    for dataset in pathmnist dermamnist bloodmnist; do
        for init in dct kaiming; do
            for seed in $SEEDS_5; do
                echo "  $dataset | $init | seed=$seed"
                $PYTHON experiments/train_medmnist.py \
                    --dataset $dataset --init $init --seed $seed \
                    --epochs $MEDMNIST_EPOCHS --data-dir "$DATA_DIR/medmnist"
            done
        done
        # Also run bottleneck-free variant
        for init in dct kaiming; do
            for seed in $SEEDS_5; do
                echo "  $dataset | $init | no-bottleneck | seed=$seed"
                $PYTHON experiments/train_medmnist.py \
                    --dataset $dataset --init $init --seed $seed --no-bottleneck \
                    --epochs $MEDMNIST_EPOCHS --data-dir "$DATA_DIR/medmnist"
            done
        done
    done
fi

# ── 9. CIFAR experiments ─────────────────────────────────
if [ "$ECG_ONLY" = false ] && [ "$RUN_CORE" = true ]; then
    CIFAR_EPOCHS=200
    if [ "$QUICK" = true ]; then CIFAR_EPOCHS=20; fi

    echo ""
    echo "=== Step 9: CIFAR-100 (DCT vs Kaiming, 5 of 20 seeds) ==="
    $PYTHON experiments/train_cifar100.py --init dct --epochs $CIFAR_EPOCHS --data-dir "$DATA_DIR/cifar"
    for seed in $SEEDS_5; do
        $PYTHON experiments/train_cifar100.py --init kaiming --seed $seed --epochs $CIFAR_EPOCHS --data-dir "$DATA_DIR/cifar"
    done
fi

# ── 10. Verify paper numbers ────────────────────────────────
echo ""
echo "=== Step 10: Verify paper numbers from evidence ==="
if [ -f "evidence/compute_paper_numbers.py" ]; then
    $PYTHON evidence/compute_paper_numbers.py --verify
else
    echo "  compute_paper_numbers.py not found (skipping verification)"
fi

echo ""
echo "============================================"
echo "ALL DONE: $(date)"
echo "Results in: $SCRIPT_DIR/results/"
echo "============================================"
ls -la results/*.json 2>/dev/null || echo "(no results yet)"
