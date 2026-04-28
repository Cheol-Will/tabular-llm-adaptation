from __future__ import annotations

import argparse
import os
from pathlib import Path

from analysis_utils import analyze_hpo, analyze_reg_dist

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Modular analysis framework for TabArena experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--analysis_type",
        type=str,
        required=True,
        choices=["hpo", "reg-dist"],
        help="Type of analysis to perform",
    )

    parser.add_argument(
        "--task_id",
        type=str,
        default=None,
        help="Task ID to analyze (required for reg-dist)",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., TFMLLM, FTTransformer)",
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name (directory under results/)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )

    return parser

def main():
    """Main entry point - routes to appropriate analysis function."""
    parser = get_parser()
    args = parser.parse_args()

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "evals" / args.exp_name

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"TabArena Analysis Framework")
    print(f"{'='*80}")
    print(f"Analysis type: {args.analysis_type}")
    print(f"Output directory: {output_dir}")

    # Route to appropriate analysis function
    if args.analysis_type == "hpo":
        analyze_hpo(
            model=args.model,
            exp_name=args.exp_name,
            output_dir=output_dir,
        )
    elif args.analysis_type == "reg-dist":
        if args.task_id is None:
            raise ValueError("--task_id is required for reg-dist analysis")
        analyze_reg_dist(
            model=args.model,
            exp_name=args.exp_name,
            task_id=args.task_id,
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Unknown analysis type: {args.analysis_type}")


if __name__ == "__main__":
    main()
