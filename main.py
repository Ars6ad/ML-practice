import os
import argparse
import torch

from src.utils import load_config, set_seed, get_device
from src.train import train_experiment
from src.evaluate import evaluate_checkpoint
from src.robustness import evaluate_robustness
from scripts.audit_dataset import audit_dataset
from scripts.compare_results import compare_experiments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Robust Deep Learning-Based Postharvest Fruit Quality Classification"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["baseline", "augmentation", "class_weighted", "all"],
        help="Experiment to run (default: all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override maximum training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run dataset audit and exit",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate existing checkpoints without retraining",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip environmental perturbation robustness analysis",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force retraining even if checkpoint exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Apply CLI overrides
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.seed is not None:
        config["training"]["seed"] = args.seed

    set_seed(config["training"]["seed"])

    print("\n" + "=" * 80)
    print(" ROBUST POSTHARVEST FRUIT QUALITY CLASSIFICATION")
    print(" Transfer Learning with DenseNet121 under Class Imbalance & Environmental Variation")
    print("=" * 80)
    get_device(verbose=True)

    # 1. Dataset Audit
    print("\n--- STEP 1: Dataset Audit and Validation ---")
    audit_dataset(args.config)

    if args.audit_only:
        print("\nAudit completed. Exiting as --audit-only was specified.")
        return

    # Determine experiments to run
    if args.experiment == "all":
        experiments_to_run = ["baseline", "augmentation", "class_weighted"]
    else:
        experiments_to_run = [args.experiment]

    checkpoints = {}
    ckpt_dir = config["paths"]["checkpoints_dir"]

    # 2. Training
    if not args.evaluate_only:
        print("\n--- STEP 2: Model Training ---")
        for exp in experiments_to_run:
            res = train_experiment(
                config=config,
                experiment_type=exp,
                resume=(not args.no_resume),
            )
            checkpoints[exp] = res["checkpoint_path"]
    else:
        print("\n--- Skipping Training (--evaluate-only specified) ---")
        for exp in experiments_to_run:
            checkpoints[exp] = os.path.join(ckpt_dir, f"{exp}_best.pth")

    # 3. Evaluation on Untouched Test Set
    print("\n--- STEP 3: Evaluation on Untouched Test Split ---")
    for exp, ckpt_path in checkpoints.items():
        if os.path.exists(ckpt_path):
            evaluate_checkpoint(ckpt_path, config=config, split="test")
        else:
            print(f"Warning: Checkpoint for {exp} not found at {ckpt_path}. Skipping evaluation.")

    # 4. Comparative Analysis & Visualizations
    if len(experiments_to_run) > 1 or args.experiment == "all":
        print("\n--- STEP 4: Comparative Statistical Analysis ---")
        compare_experiments(args.config)

    # 5. Environmental Robustness Analysis
    if not args.skip_robustness:
        available_ckpts = {
            exp: os.path.join(ckpt_dir, f"{exp}_best.pth")
            for exp in ["baseline", "augmentation", "class_weighted"]
            if os.path.exists(os.path.join(ckpt_dir, f"{exp}_best.pth"))
        }
        if len(available_ckpts) > 0:
            print("\n--- STEP 5: Environmental Robustness Perturbation Evaluation ---")
            rob_df = evaluate_robustness(available_ckpts, config)
            from src.visualization import plot_robustness_comparison
            plots_dir = config["paths"]["plots_dir"]
            plot_robustness_comparison(rob_df, save_path=os.path.join(plots_dir, "robustness_comparison.png"))

    print("\n" + "=" * 80)
    print(" ALL WORKFLOW STAGES FINISHED SUCCESSFULLY!")
    print(" Outputs, models, plots, and reports generated in 'outputs/'")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
