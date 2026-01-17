"""
Experiment 1-1: Evaluate Qwen3-8B chess move quality on static positions.

This script:
1. Loads chess positions from a dataset
2. Has Qwen suggest moves with and without chain-of-thought
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

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Sample chess positions (FEN strings) - a mix of opening, middlegame, and endgame
SAMPLE_POSITIONS = [
    # Opening positions
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "After 1.e4"),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "After 1.e4 e5"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Italian Game start"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Italian Game"),
    ("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3", "Bishops Opening"),

    # Middlegame positions
    ("r1bq1rk1/ppp2ppp/2n2n2/3pp3/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQ d6 0 7", "Queens Gambit Declined"),
    ("r2q1rk1/pppbbppp/2n1pn2/3p4/3P4/2NBPN2/PPP2PPP/R1BQ1RK1 w - - 6 8", "Closed position"),
    ("r1b2rk1/pp1nqppp/2pbpn2/3p4/2PP4/2N1PN2/PPQ2PPP/R1B1KB1R w KQ - 4 9", "Typical middlegame"),
    ("r4rk1/pp1bppbp/2np1np1/q7/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 3 11", "Dragon Sicilian"),
    ("r1bqr1k1/ppp2ppp/2np1n2/2b1p3/2B1P3/2PP1N2/PP3PPP/RNBQ1RK1 w - - 0 8", "Giuoco Piano"),

    # Tactical positions
    ("r2qkb1r/pp2pppp/2n2n2/3p4/3P2b1/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 4 6", "Pin on f3"),
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5", "Development position"),
    ("2r2rk1/pp2ppbp/1qnp1np1/8/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 5 13", "Opposite castling"),

    # Endgame positions
    ("8/5k2/8/8/8/8/5PP1/4K3 w - - 0 1", "King and pawns endgame"),
    ("8/8/8/3k4/8/8/3K4/3R4 w - - 0 1", "Rook vs King"),
    ("8/8/4k3/8/2B5/8/8/4K3 w - - 0 1", "Bishop vs King"),
    ("5k2/5p2/5P2/5K2/8/8/8/8 w - - 0 1", "Pawn endgame"),
    ("8/8/8/8/8/5K2/4Q3/5k2 w - - 0 1", "Queen vs King"),
    ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "Rook endgame"),
    ("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1", "Opposition endgame"),
]

def evaluate_move(board: chess.Board, move: chess.Move, engine: chess.engine.SimpleEngine) -> dict:
    """
    Evaluate a move using Stockfish.
    Returns centipawn loss and other metrics.
    """
    # Get evaluation before the move
    info_before = engine.analyse(board, chess.engine.Limit(depth=15))
    score_before = info_before["score"].relative.score(mate_score=10000)

    # Find best move
    best_move_info = engine.play(board, chess.engine.Limit(depth=15))
    best_move = best_move_info.move

    # Make the actual move and evaluate
    board_copy = board.copy()
    board_copy.push(move)
    info_after = engine.analyse(board_copy, chess.engine.Limit(depth=15))
    # Note: score is now from opponent's perspective, so negate it
    score_after = -info_after["score"].relative.score(mate_score=10000)

    # Make the best move and evaluate
    board_best = board.copy()
    board_best.push(best_move)
    info_best = engine.analyse(board_best, chess.engine.Limit(depth=15))
    score_best = -info_best["score"].relative.score(mate_score=10000)

    centipawn_loss = score_best - score_after

    return {
        "move": move.uci(),
        "best_move": best_move.uci(),
        "is_best": move == best_move,
        "centipawn_loss": max(0, centipawn_loss),  # Loss should be non-negative
        "score_after": score_after,
        "score_best": score_best,
    }


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
        move = chess.Move.from_uci(match)
        if move in board.legal_moves:
            return move

    return None


async def get_move_from_qwen( api: InferenceAPI, board: chess.Board, use_cot: bool, model_id: str = "Qwen/Qwen3-8b",) -> tuple[str, chess.Move | None]:
    """
    Ask Qwen for a chess move.
    Returns (response_text, parsed_move).
    """
    board_text = board.fen()

    if use_cot:
        prompt_text = f"""{board_text}\n\n What is your move? Output your move like this: \\box{{e2e4}}"""
    else:
        prompt_text = f"""{board_text}\n\n What is your move? Output your move like this: \\box{{e2e4}} /no_think"""

    prompt = Prompt(messages=[ChatMessage(content=prompt_text, role=MessageRole.user)])

    response = await api(
        model_id=model_id,
        prompt=prompt,
        max_tokens=2000,  # Qwen3-8b is a reasoning model, needs tokens for thinking
        temperature=0,
        force_provider="openai",  # Use OpenRouter via OpenAI-compatible API
    )

    response_text = response[0].completion
    parsed_move = parse_move_from_response(response_text, board)

    return response_text, parsed_move


def compute_and_plot_results(results: dict, output_path: str = "experiment1_1_results.png"):
    """Compute statistics, print results, and create bar graph."""
    # Calculate statistics
    stats = {}
    for mode in ["with_cot", "without_cot"]:
        cpl_values = [r["centipawn_loss"] for r in results[mode]]
        valid_moves = sum(1 for r in results[mode] if r["valid_move"])
        best_moves = sum(1 for r in results[mode] if r.get("is_best", False))

        avg_cpl = np.mean(cpl_values)
        stats[mode] = {
            "avg_centipawn_loss": avg_cpl,
            "std_centipawn_loss": np.std(cpl_values),
            "valid_move_rate": valid_moves / len(results[mode]),
            "best_move_rate": best_moves / len(results[mode]),
            "estimated_elo": centipawn_loss_to_elo_estimate(avg_cpl),
        }

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for mode in ["with_cot", "without_cot"]:
        s = stats[mode]
        print(f"\n{mode.upper().replace('_', ' ')}:")
        print(f"  Average centipawn loss: {s['avg_centipawn_loss']:.1f} (+/- {s['std_centipawn_loss']:.1f})")
        print(f"  Valid move rate: {s['valid_move_rate']*100:.1f}%")
        print(f"  Best move rate: {s['best_move_rate']*100:.1f}%")
        print(f"  Estimated ELO: ~{s['estimated_elo']}")

    # Create bar graph
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    modes = ["With CoT", "Without CoT"]
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

    # Best move rate
    ax3 = axes[2]
    best_rates = [stats["with_cot"]["best_move_rate"]*100, stats["without_cot"]["best_move_rate"]*100]
    ax3.bar(x, best_rates, color=["#2ecc71", "#e74c3c"])
    ax3.set_ylabel("Best Move Rate (%)")
    ax3.set_title("Best Move Rate")
    ax3.set_xticks(x)
    ax3.set_xticklabels(modes)
    ax3.set_ylim(0, 105)

    plt.suptitle("Qwen3-8B Chess Performance: With vs Without Chain-of-Thought", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")

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
    with open("experiment1_1_results.json", "w") as f:
        json.dump({"results": results, "stats": stats}, f, indent=2)
    print(f"Detailed results saved to experiment1_1_results.json")

    return stats


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


async def evaluate_single_position( api: InferenceAPI, fen: str, description: str, use_cot: bool, engine: chess.engine.SimpleEngine,) -> dict:
    """Evaluate a single position with the model."""
    board = chess.Board(fen)
    mode = "with_cot" if use_cot else "without_cot"

    response, move = await get_move_from_qwen(api, board, use_cot)

    if move is None:
        return {
            "position": fen,
            "description": description,
            "mode": mode,
            "valid_move": False,
            "centipawn_loss": 500,  # Penalty for invalid move
            "response": response or "",
        }
    else:
        eval_result = evaluate_move(board, move, engine)
        return {
            "position": fen,
            "description": description,
            "mode": mode,
            "valid_move": True,
            "response": response,
            **eval_result,
        }


async def run_experiment(test_mode: bool = False, concurrency: int = 100):
    """Run the experiment with concurrent API calls."""
    utils.setup_environment()

    # Use OpenRouter API
    api = InferenceAPI(
        cache_dir=Path(".cache"),
        openai_base_url=OPENROUTER_BASE_URL,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # Initialize Stockfish
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")

    positions = SAMPLE_POSITIONS[:3] if test_mode else SAMPLE_POSITIONS

    print(f"Running experiment on {len(positions)} positions with concurrency={concurrency}...")
    print("=" * 60)

    # Create task specifications
    task_specs = []
    for fen, description in positions:
        for use_cot in [True, False]:
            task_specs.append((fen, description, use_cot))

    async def limited_task(fen, description, use_cot):
        async with asyncio.Semaphore(concurrency):
            return await evaluate_single_position(api, fen, description, use_cot, engine)

    # Run all tasks concurrently with limited concurrency
    print(f"Launching {len(task_specs)} API calls...")
    all_results = await asyncio.gather(*[limited_task(fen, desc, cot) for fen, desc, cot in task_specs])

    # Save model outputs to folder
    output_dir = Path("model_outputs_visual")
    output_dir.mkdir(exist_ok=True)
    for i, result in enumerate(all_results):
        mode = result["mode"]
        desc = result["description"].replace(" ", "_").replace(".", "")
        filename = f"{i:02d}_{desc}_{mode}.txt"
        with open(output_dir / filename, "w") as f:
            f.write(f"Position: {result['description']}\n")
            f.write(f"FEN: {result['position']}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Valid move: {result['valid_move']}\n")
            f.write("=" * 60 + "\n")
            f.write("MODEL RESPONSE:\n")
            f.write("=" * 60 + "\n")
            f.write(result.get("response", "") + "\n")
    print(f"Saved {len(all_results)} model outputs to {output_dir}/")

    # Organize results by mode
    results = {"with_cot": [], "without_cot": []}
    for result in all_results:
        mode = result.pop("mode")
        result.pop("response", None)  # Remove response from results dict
        results[mode].append(result)

        # Print progress
        if result["valid_move"]:
            print(f"  {result['description']} ({mode}): {result['move']} (loss: {result['centipawn_loss']})")
        else:
            print(f"  {result['description']} ({mode}): INVALID MOVE")

    engine.quit()

    # Compute stats, plot, and save results
    stats = compute_and_plot_results(results)

    return results, stats


if __name__ == "__main__":
    import sys
    test_mode = "--test" in sys.argv
    if test_mode:
        print("Running in TEST MODE (3 positions only)")
    asyncio.run(run_experiment(test_mode=test_mode))
