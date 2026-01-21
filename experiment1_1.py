"""
Experiment 1-1: Evaluate GPTOSS-20b chess move quality on static positions.

This script:
1. Loads chess positions from a dataset
2. Has GPTOSS-20b suggest moves with and without chain-of-thought
3. Evaluates move quality using Stockfish
4. Outputs results as a bar graph with ELO context
"""

import asyncio
import json
import os
import re
from pathlib import Path

import chess
import chess.engine
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

from chess_utils import evaluate_with_stockfish

# OpenRouter configuration
INFERENCE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-oss-20b"

def load_positions(path: str = "outputs/boards.json") -> list[tuple[str, str]]:
    """Load positions from boards.json file (list of FEN strings)."""
    with open(path) as f:
        fens = json.load(f)
    return [(fen, f"Position {i+1}") for i, fen in enumerate(fens)]


def parse_move_from_response(response: str, board: chess.Board) -> chess.Move | None:
    """
    Extract a chess move from the model's response.
    Expects format: \\box{e2e4}
    """
    text = response.strip().lower()

    # Look for \box{...} format
    box_pattern = r'\\box\{([a-h][1-8][a-h][1-8][qrbn]?)\}'
    matches = re.findall(box_pattern, text)
    if len(matches) == 1:
        move = chess.Move.from_uci(matches[0])
        if move in board.legal_moves:
            return move

    return None


async def get_move_from_model(api: InferenceAPI, board: chess.Board, use_cot: bool, model_id: str = MODEL, seed: int | None = None) -> tuple[str, chess.Move | None, str]:
    """
    Ask the model for a chess move.
    Returns (response_text, parsed_move, reasoning_trace).
    
    Args:
        seed: Optional seed for reproducibility. Different seeds give different cached results.
    """
    board_text = board.fen()
    prompt_text = f"""{board_text}\n\n What is your move? Output your move like this: \\box{{e2e4}}"""
    prompt = Prompt(messages=[ChatMessage(content=prompt_text, role=MessageRole.user)])

    # Use reasoning={"effort": "low"} to disable CoT for GPT models
    # Note: For OpenAI reasoning models, use max_completion_tokens instead of max_tokens
    extra_kwargs = {"extra_body": { "reasoning": {"effort": "high"}}} 
    if not use_cot: 
        extra_kwargs["extra_body"] = {"reasoning": {"effort": "low"}}

    response = await api(
        model_id=model_id,
        prompt=prompt,
        temperature=1,
        seed=seed,
        force_provider="openai",  # Use OpenRouter via OpenAI-compatible API
        **extra_kwargs,
    )

    response_text = response[0].completion
    parsed_move = parse_move_from_response(response_text, board)
    
    # Extract reasoning trace
    reasoning = ""
    try:
        reasoning = response[0].generated_content[0].content['message'].get('reasoning', '')
    except (IndexError, KeyError, TypeError):
        pass

    return response_text, parsed_move, reasoning


def compute_and_plot_results(all_results: list, output_path: str = "outputs/experiment1_1_results.png"):
    """Compute statistics, print results, and create bar graph."""
    print(f"Saved {len(all_results)} model outputs to model_outputs_visual/")

    # Organize results by mode
    results = {"with_cot": [], "without_cot": []}
    for result in all_results:
        mode = result["mode"]
        result_copy = {k: v for k, v in result.items() if k not in ("mode", "response")}
        results[mode].append(result_copy)

        # Print progress
        if result["valid_move"]:
            print(f"  {result['description']} ({mode}): {result.get('move', 'N/A')} (loss: {result['centipawn_loss']})")
        else:
            print(f"  {result['description']} ({mode}): INVALID MOVE")

    # Calculate statistics (only for modes that have results)
    stats = {}
    raw_reasoning_words = {}  # Store raw values for scatter plot
    active_modes = [m for m in ["with_cot", "without_cot"] if results[m]]
    for mode in active_modes:
        cpl_values = [r["centipawn_loss"] for r in results[mode]]
        valid_moves = sum(1 for r in results[mode] if r["valid_move"])
        best_moves = sum(1 for r in results[mode] if r.get("is_best", False))
        # Count reasoning words (approximate by splitting on whitespace)
        reasoning_words = [len(r.get("reasoning", "").split()) for r in results[mode]]
        raw_reasoning_words[mode] = reasoning_words

        avg_cpl = np.mean(cpl_values)
        stats[mode] = {
            "avg_centipawn_loss": avg_cpl,
            "std_centipawn_loss": np.std(cpl_values),
            "valid_move_rate": valid_moves / len(results[mode]),
            "best_move_rate": best_moves / len(results[mode]),
            "estimated_elo": centipawn_loss_to_elo_estimate(avg_cpl),
            "avg_reasoning_tokens": np.mean(reasoning_words),
            "std_reasoning_tokens": np.std(reasoning_words),
        }

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    mode_labels = {"with_cot": "HIGH REASONING", "without_cot": "LOW REASONING"}
    for mode in active_modes:
        s = stats[mode]
        print(f"\n{mode_labels.get(mode, mode)}:")
        print(f"  Average centipawn loss: {s['avg_centipawn_loss']:.1f} (+/- {s['std_centipawn_loss']:.1f})")
        print(f"  Valid move rate: {s['valid_move_rate']*100:.1f}%")
        print(f"  Best move rate: {s['best_move_rate']*100:.1f}%")
        print(f"  Estimated ELO: ~{s['estimated_elo']}")
        print(f"  Avg reasoning words: {s['avg_reasoning_tokens']:.1f} (+/- {s['std_reasoning_tokens']:.1f})")

    # Create bar graph (only if we have both modes for comparison)
    if len(active_modes) == 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        modes = ["High Reasoning", "Low Reasoning"]
        x = np.arange(len(modes))

        # Centipawn loss
        ax1 = axes[0]
        cpl_values = [stats["with_cot"]["avg_centipawn_loss"], stats["without_cot"]["avg_centipawn_loss"]]
        cpl_stds = [stats["with_cot"]["std_centipawn_loss"], stats["without_cot"]["std_centipawn_loss"]]
        ax1.bar(x, cpl_values, yerr=cpl_stds, capsize=5, color=["#2ecc71", "#e74c3c"])
        ax1.set_ylabel("Centipawn Loss")
        ax1.set_title("Average Centipawn Loss\n(lower is better)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(modes)
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='~2000 ELO')
        ax1.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='~1700 ELO')
        ax1.legend()

        # Valid move rate
        ax2 = axes[1]
        valid_rates = [stats["with_cot"]["valid_move_rate"]*100, stats["without_cot"]["valid_move_rate"]*100]
        ax2.bar(x, valid_rates, color=["#2ecc71", "#e74c3c"])
        ax2.set_ylabel("Valid Move Rate (%)")
        ax2.set_title("Valid Move Rate")
        ax2.set_xticks(x)
        ax2.set_xticklabels(modes)
        ax2.set_ylim(0, 105)

        # Reasoning words (log scale)
        ax3 = axes[2]
        token_values = [stats["with_cot"]["avg_reasoning_tokens"], stats["without_cot"]["avg_reasoning_tokens"]]
        ax3.bar(x, token_values, color=["#2ecc71", "#e74c3c"], alpha=0.7)
        # Add raw datapoints as scatter
        for i, mode in enumerate(["with_cot", "without_cot"]):
            jitter = np.random.uniform(-0.15, 0.15, len(raw_reasoning_words[mode]))
            # Add 1 to avoid log(0) issues
            values = [max(v, 1) for v in raw_reasoning_words[mode]]
            ax3.scatter(x[i] + jitter, values, color="black", alpha=0.4, s=15, zorder=3)
        ax3.set_yscale('log', base=2)
        # Set tick marks at powers of 2
        max_val = max(max(raw_reasoning_words["with_cot"]), max(raw_reasoning_words["without_cot"]), 1)
        powers = [2**i for i in range(0, int(np.log2(max_val)) + 2)]
        ax3.set_yticks(powers)
        ax3.set_yticklabels([str(p) for p in powers])
        ax3.set_ylabel("Reasoning Words")
        ax3.set_title("Average Reasoning Words (log scale)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(modes)

        plt.suptitle("GPT-OSS-20B Chess Performance: High vs Low Reasoning", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {output_path}")
    else:
        print("\nSkipping plot (need both CoT modes for comparison)")

    # Add ELO context
    print("\n" + "=" * 60)
    print("ELO CONTEXT")
    print("=" * 60)
    print(
        "Centipawn Loss to ELO (approximate):\n"
        "  - Grandmaster (2500+): 10-20 CPL\n"
        "  - Master (2200-2500): 20-40 CPL\n"
        "  - Expert (2000-2200): 40-60 CPL\n"
        "  - Class A (1800-2000): 60-90 CPL\n"
        "  - Class B (1600-1800): 90-120 CPL\n"
        "  - Class C (1400-1600): 120-160 CPL\n"
        "  - Class D (1200-1400): 160-200 CPL\n"
        "  - Beginner (<1200): 200+ CPL"
    )

    # Save detailed results
    with open("outputs/experiment1_1_results.json", "w") as f:
        json.dump({"results": results, "stats": stats}, f, indent=2)
    print(f"Detailed results saved to outputs/experiment1_1_results.json")

    return results, stats


def centipawn_loss_to_elo_estimate(avg_cpl: float) -> int:
    """
    Rough estimate of ELO based on average centipawn loss.
    Based on empirical data from chess databases.
    """
    # Approximate mapping (varies by source):
    # Grandmaster (2500+): 10-20 CPL
    # Master (2200-2500): 20-40 CPL
    # Expert (2000-2200): 40-60 CPL
    # Class A (1800-2000): 60-90 CPL
    # Class B (1600-1800): 90-120 CPL
    # Class C (1400-1600): 120-160 CPL
    # Class D (1200-1400): 160-200 CPL
    # Beginner (<1200): 200+ CPL

    if avg_cpl < 15:
        return 2600
    elif avg_cpl < 25:
        return 2400
    elif avg_cpl < 40:
        return 2200
    elif avg_cpl < 60:
        return 2000
    elif avg_cpl < 90:
        return 1800
    elif avg_cpl < 120:
        return 1600
    elif avg_cpl < 160:
        return 1400
    elif avg_cpl < 200:
        return 1200
    else:
        return 1000


def write_result_to_file(result: dict, idx: int, output_dir: Path):
    """Write a single result to model_outputs_visual folder."""
    mode = result["mode"]
    desc = result["description"].replace(" ", "_").replace(".", "")
    filename = f"{idx:02d}_{desc}_{mode}.txt"
    with open(output_dir / filename, "w") as f:
        f.write(f"Position: {result['description']}\n")
        f.write(f"FEN: {result['position']}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Valid move: {result.get('valid_move', 'pending')}\n")
        f.write("=" * 60 + "\n")
        f.write("REASONING TRACE:\n")
        f.write("=" * 60 + "\n")
        f.write(result.get("reasoning", "") + "\n")
        f.write("=" * 60 + "\n")
        f.write("MODEL RESPONSE:\n")
        f.write("=" * 60 + "\n")
        f.write(result.get("response", "") + "\n")


async def get_model_move(api: InferenceAPI, fen: str, description: str, use_cot: bool, idx: int, output_dir: Path) -> dict:
    """Get move from model (no Stockfish evaluation yet)."""
    board = chess.Board(fen)
    mode = "with_cot" if use_cot else "without_cot"

    response, move, reasoning = await get_move_from_model(api, board, use_cot)

    result = {
        "position": fen,
        "description": description,
        "mode": mode,
        "move": move.uci() if move else None,
        "response": response or "",
        "reasoning": reasoning or "",
        "idx": idx,
    }
    
    # Write result immediately
    write_result_to_file(result, idx, output_dir)
    
    return result


async def run_experiment(test_mode: bool = False, concurrency: int = 100, num_positions: int = 500):
    """Run the experiment with concurrent API calls."""
    utils.setup_environment()

    # Use OpenRouter API
    api = InferenceAPI(
        cache_dir=Path(".cache"),
        openai_base_url=INFERENCE_URL,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    all_positions = load_positions()
    if test_mode:
        positions = all_positions[:3]
    else:
        positions = all_positions[:num_positions]

    print(f"Running experiment on {len(positions)} positions with concurrency={concurrency}...")
    print("=" * 60)

    # Create output directory
    output_dir = Path("model_outputs_visual")
    output_dir.mkdir(exist_ok=True)

    # Create task specifications with indices
    task_specs = []
    cot_modes = [False] if test_mode else [True, False]  # Skip CoT in test mode
    idx = 0
    for fen, description in positions:
        for use_cot in cot_modes:
            task_specs.append((fen, description, use_cot, idx))
            idx += 1

    semaphore = asyncio.Semaphore(concurrency)  # Shared semaphore for rate limiting

    async def limited_task(fen, description, use_cot, idx):
        async with semaphore:
            return await get_model_move(api, fen, description, use_cot, idx, output_dir)

    # Run all API calls concurrently with progress bar
    print(f"Launching {len(task_specs)} API calls...")
    tasks = [limited_task(fen, desc, cot, i) for fen, desc, cot, i in task_specs]
    model_results = await tqdm_asyncio.gather(*tasks, desc="API calls")

    # Now evaluate with Stockfish sequentially (avoids crashes)
    print("Evaluating moves with Stockfish...")
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    all_results = []
    for result in tqdm(model_results, desc="Stockfish eval"):
        try:
            evaluated = evaluate_with_stockfish(result, engine)
        except chess.engine.EngineTerminatedError:
            # Engine crashed, restart it and mark this result as failed
            print(f"\nStockfish crashed on {result['description']}, restarting...")
            try:
                engine.quit()
            except:
                pass
            engine = chess.engine.SimpleEngine.popen_uci("stockfish")
            evaluated = {
                **result,
                "valid_move": False,
                "centipawn_loss": 500,  # Penalty
                "error": "stockfish_crash",
            }
        # Update the file with valid_move info after Stockfish evaluation
        write_result_to_file(evaluated, evaluated["idx"], output_dir)
        all_results.append(evaluated)
    engine.quit()

    # Compute stats, plot, and save results
    results, stats = compute_and_plot_results(all_results)

    return results, stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode (3 positions only)")
    parser.add_argument("--n", type=int, default=10, help="Number of positions to evaluate")
    args = parser.parse_args()
    
    if args.test:
        print("Running in TEST MODE (3 positions only)")
    asyncio.run(run_experiment(test_mode=args.test, num_positions=args.n))
