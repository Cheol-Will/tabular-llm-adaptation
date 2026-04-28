from __future__ import annotations

import argparse
import os
from pathlib import Path

from analysis_utils import analyze_hpo, analyze_reg_dist, analyze_attn_map
from utils import get_parser


def main():
    """Main entry point - routes to appropriate analysis function."""
    parser = get_parser()
    parser.add_argument(
        "--analysis_type",
        type=str,
        required=True,
        choices=["hpo", "reg-dist", "attn-map"],
        help="Type of analysis to perform",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        required=True,
    )
    args = parser.parse_args()

    # Set up output directory
    output_dir = Path(__file__).parent / "evals" / args.exp_name / "analysis"

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
        if args.task_ids is None:
            raise ValueError("--task_ids is required for reg-dist analysis")
        analyze_reg_dist(
            model=args.model,
            exp_name=args.exp_name,
            task_id=args.task_ids,
            output_dir=output_dir,
        )
    elif args.analysis_type == "attn-map":
        analyze_attn_map(
            args=args,
            model=args.model,
            exp_name=None,
            task_id=args.task_id,
            output_dir=None,
        )
    else:
        raise ValueError(f"Unknown analysis type: {args.analysis_type}")


if __name__ == "__main__":
    main()
