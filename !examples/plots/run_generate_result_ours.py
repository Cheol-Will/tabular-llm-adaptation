from __future__ import annotations

from pathlib import Path

from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection

if __name__ == "__main__":
    save_path = "output_leaderboard"  # folder to save all figures and tables

    output_path_verified = Path(save_path) / "verified"
    output_path_unverified = Path(save_path) / "unverified"

    target_methods = [
        "LinearModel",
        "RandomForest",
        "CatBoost", 
        "RealMLP_GPU", 
        "RealTabPFN-v2.5"
    ]
    
    target_metadata_lst = [
        m for m in tabarena_method_metadata_collection.method_metadata_lst 
        if m.method in target_methods
    ]
    
    tabarena_context = TabArenaContext(methods=target_metadata_lst, include_unverified=False)
    tabarena_context_unverified = TabArenaContext(methods=target_metadata_lst, include_unverified=True)

    leaderboard_verified = tabarena_context.compare(output_dir=output_path_verified)
    leaderboard_unverified = tabarena_context_unverified.compare(output_dir=output_path_unverified)

    leaderboard_verified = leaderboard_verified[
        ~leaderboard_verified['method'].str.contains('ensemble', case=False, na=False)
    ]
    leaderboard_unverified = leaderboard_unverified[
        ~leaderboard_unverified['method'].str.contains('ensemble', case=False, na=False)
    ]

    leaderboard_website_verified = tabarena_context.leaderboard_to_website_format(leaderboard=leaderboard_verified)
    leaderboard_website_unverified = tabarena_context_unverified.leaderboard_to_website_format(leaderboard=leaderboard_unverified)

    print("Verified Leaderboard (Default & Tuned only):")
    print(leaderboard_website_verified.to_markdown(index=False))
    print("\n" + "-"*60 + "\n")

    print("Unverified Leaderboard (Default & Tuned only):")
    print(leaderboard_website_unverified.to_markdown(index=False))
    print("")