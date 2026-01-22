import asyncio
import json
import os
from pathlib import Path
from collections import defaultdict

import chess
import chess.engine
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils as st_utils

from utils import (
    load_positions,
    get_move_from_model,
    centipawn_loss_to_elo_estimate,
    evaluate_with_stockfish,
    MODEL,
    INFERENCE_URL,
)


def select_best_move(
    moves: list[tuple[str, chess.Move | None, str]],
    engine: chess.engine.SimpleEngine,
    board: chess.Board
) -> tuple[int, dict]:
    """
    Select the best move from N candidates based on centipawn loss.

    Returns:
        (best_index, evaluation_dict) where evaluation_dict contains
        the stockfish evaluation for the best move.
    """
    best_idx = 0
    best_eval = None
    best_cpl = float('inf')

    for idx, (response, move, reasoning) in enumerate(moves):
        result = {
            "position": board.fen(),
            "move": move.uci() if move else None,
            "response": response,
            "reasoning": reasoning,
        }
        evaluated = evaluate_with_stockfish(result, engine)

        if evaluated["centipawn_loss"] < best_cpl:
            best_cpl = evaluated["centipawn_loss"]
            best_eval = evaluated
            best_idx = idx

    return best_idx, best_eval


async def run_bon_experiment(
    test_mode: bool = False,
    concurrency: int = 200,
    num_positions: int = 100,
):
    """
    Run Best-of-N experiment.

    For each position and each reasoning mode, sample N times and pick the best.
    """
    st_utils.setup_environment()

    # Use OpenRouter API (caching works with different seeds for BoN)
    api = InferenceAPI(
        cache_dir=Path(".cache"),
        openai_base_url=INFERENCE_URL,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    all_positions = load_positions()

    if test_mode:
        positions = all_positions[:5]
        n_values = [2, 4]
        cot_modes = [False]  # Only low reasoning in test mode
    else:
        positions = all_positions[:num_positions]
        n_values = [1, 2, 4, 8, 16]
        cot_modes = [False]  # Only low reasoning

    print(f"Running BoN experiment on {len(positions)} positions")
    print(f"N values: {n_values}")
    print(f"Reasoning modes: low only")
    print("=" * 60)

    # Build all task specifications upfront
    # Each task: (pos_idx, fen, n, use_cot, seed)
    # We need separate API calls for each (position, n, cot, seed) combination
    task_specs = []
    for pos_idx, (fen, description) in enumerate(positions):
        for n in n_values:
            for use_cot in cot_modes:
                for seed in range(1, n + 1):
                    task_specs.append((pos_idx, fen, n, use_cot, seed))

    print(f"Launching {len(task_specs)} API calls with concurrency={concurrency}...")

    semaphore = asyncio.Semaphore(concurrency)

    async def get_single_move(pos_idx, fen, n, use_cot, seed):
        """Get a single move from the model."""
        async with semaphore:
            board = chess.Board(fen)
            response, move, reasoning = await get_move_from_model(api, board, use_cot, seed=seed)
            return {
                "pos_idx": pos_idx,
                "fen": fen,
                "n": n,
                "use_cot": use_cot,
                "seed": seed,
                "response": response,
                "move": move,
                "reasoning": reasoning,
            }

    # Run all API calls concurrently
    tasks = [get_single_move(*spec) for spec in task_specs]
    api_results = await tqdm_asyncio.gather(*tasks, desc="API calls")

    # Group results by (pos_idx, n, use_cot)
    grouped = defaultdict(list)
    for r in api_results:
        key = (r["pos_idx"], r["n"], r["use_cot"])
        grouped[key].append(r)

    # Now evaluate with Stockfish sequentially
    print("Evaluating with Stockfish...")
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")

    # Results structure: {(n, use_cot): [list of CPL values]}
    results = {(n, cot): [] for n in n_values for cot in cot_modes}

    # Process each group
    for (pos_idx, n, use_cot), moves_data in tqdm(sorted(grouped.items()), desc="Stockfish eval"):
        fen = moves_data[0]["fen"]
        board = chess.Board(fen)

        # Convert to format expected by select_best_move
        moves = [(m["response"], m["move"], m["reasoning"]) for m in moves_data]

        # Select best move using Stockfish
        best_idx, best_eval = select_best_move(moves, engine, board)

        cpl = best_eval["centipawn_loss"]
        results[(n, use_cot)].append(cpl)

        mode_str = "medium" if use_cot else "low"
        print(f"  Pos {pos_idx+1}, N={n}, {mode_str}: CPL={cpl:.0f} (best of {n})")

    engine.quit()

    # Compute statistics and plot
    compute_and_plot_bon_results(results, n_values, cot_modes)

    return results


def compute_and_plot_bon_results(
    results: dict,
    n_values: list[int],
    cot_modes: list[bool],
    output_path: str = "outputs/experiment2_1_results.png"
):
    """Compute statistics and create the BoN plot."""

    # Compute mean and std for each configuration
    stats = {}
    for (n, cot), cpl_values in results.items():
        stats[(n, cot)] = {
            "mean_cpl": np.mean(cpl_values),
            "std_cpl": np.std(cpl_values),
            "elo": centipawn_loss_to_elo_estimate(np.mean(cpl_values)),
        }

    # Print results
    print("\n" + "=" * 60)
    print("BEST-OF-N RESULTS")
    print("=" * 60)

    for cot in cot_modes:
        mode_str = "MEDIUM REASONING" if cot else "LOW REASONING"
        print(f"\n{mode_str}:")
        for n in n_values:
            s = stats[(n, cot)]
            print(f"  N={n:2d}: CPL={s['mean_cpl']:.1f} (+/- {s['std_cpl']:.1f}), ~{s['elo']} ELO")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"medium": "#2ecc71", "low": "#e74c3c"}
    markers = {"medium": "o", "low": "s"}

    for cot in cot_modes:
        mode_str = "medium" if cot else "low"
        label = "Medium Reasoning" if cot else "Low Reasoning"

        means = [stats[(n, cot)]["mean_cpl"] for n in n_values]
        stds = [stats[(n, cot)]["std_cpl"] for n in n_values]

        ax.errorbar(
            n_values, means, yerr=stds,
            label=label, color=colors[mode_str],
            marker=markers[mode_str], markersize=8,
            capsize=5, linewidth=2
        )

    ax.set_xlabel("N (Best-of-N samples)", fontsize=12)
    ax.set_ylabel("Average Centipawn Loss", fontsize=12)
    ax.set_title("Best-of-N Sampling: Effect on Move Quality", fontsize=14, fontweight='bold')

    # Set x-axis to log scale since N doubles each time
    ax.set_xscale('log', base=2)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])

    # Add ELO reference lines
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='~2000 ELO')
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='~1700 ELO')

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Invert y-axis so lower (better) CPL is at top
    ax.invert_yaxis()

    plt.tight_layout()

    # Ensure output directory exists
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")

    # Save detailed results
    results_serializable = {
        f"n={n}_cot={cot}": {
            "cpl_values": cpl_values,
            "mean": stats[(n, cot)]["mean_cpl"],
            "std": stats[(n, cot)]["std_cpl"],
            "elo": stats[(n, cot)]["elo"],
        }
        for (n, cot), cpl_values in results.items()
    }

    with open("outputs/experiment2_1_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Detailed results saved to outputs/experiment2_1_results.json")

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode (5 positions, N=2,4 only)")
    parser.add_argument("--n", type=int, default=100, help="Number of positions to evaluate")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent API calls")
    args = parser.parse_args()

    if args.test:
        print("Running in TEST MODE (5 positions, N=[2,4], low reasoning only)")
    asyncio.run(run_bon_experiment(
        test_mode=args.test,
        num_positions=args.n,
        concurrency=args.concurrency
    ))
