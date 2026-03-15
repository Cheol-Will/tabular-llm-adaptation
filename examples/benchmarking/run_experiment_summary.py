from pathlib import Path
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.website.website_format import format_leaderboard

def main():
    res_root = Path(__file__).parent / "results"
    
    folders = [
        "ft_transformer",
        # "ours",
    ]
    
    paths = [str(res_root / f) for f in folders if (res_root / f).exists()]
    print(f"Loading: {paths}")

    # Load and merge results from specified paths
    combined = EndToEnd.from_path_raw_to_results(path_raw=paths, cache=True)

    # Compare with internal Paper Baselines
    leaderboard = combined.compare_on_tabarena(
        output_dir=res_root / "summary",
        only_valid_tasks=True,
        use_model_results=True
    )

    # Print the final leaderboard
    print(format_leaderboard(leaderboard).to_markdown(index=False))

if __name__ == "__main__":
    main()