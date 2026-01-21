"""
Experiment 2: RL training on chess using GRPO with Tinker.

Trains a model to play better chess using GRPO (Group Relative Policy Optimization).
Uses Stockfish evaluation as the reward signal.

Outputs a graph of: Training Steps vs Chess Performance (avg centipawn loss, valid move rate)
"""

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import tinker
from tinker import ServiceClient, Datum, ModelInput, SamplingParams, AdamParams, TensorData


@dataclass
class Config:
    batch_size: int = 8
    group_size: int = 4
    learning_rate: float = 4e-5
    lora_rank: int = 32
    max_steps: int = 10
    eval_positions: int = 20
    base_model: str = "openai/gpt-oss-20b"


def load_positions(path: str = "outputs/boards.json") -> list[str]:
    """Load chess positions (FEN strings) from the dataset."""
    with open(path) as f:
        return json.load(f)


def parse_move_from_response(response: str, board: chess.Board) -> chess.Move | None:
    """Extract a chess move from the model's response. Expects format: \\box{e2e4}"""
    text = response.strip().lower()
    box_pattern = r'\\box\{([a-h][1-8][a-h][1-8][qrbn]?)\}'
    matches = re.findall(box_pattern, text)
    if len(matches) >= 1:
        move = chess.Move.from_uci(matches[0])
        if move in board.legal_moves:
            return move
    return None


def compute_reward_and_cpl(
    fen: str, move_str: str | None, engine: chess.engine.SimpleEngine
) -> tuple[float, float, bool]:
    """
    Compute reward and centipawn loss for a move.
    Returns (reward, centipawn_loss, is_valid).
    """
    if move_str is None:
        return -1.0, 500.0, False

    board = chess.Board(fen)
    move = chess.Move.from_uci(move_str)

    if move not in board.legal_moves:
        return -1.0, 500.0, False

    # Get best move evaluation
    best_move_info = engine.play(board, chess.engine.Limit(depth=12))
    best_move = best_move_info.move

    # Evaluate position after our move
    board_after = board.copy()
    board_after.push(move)
    info_after = engine.analyse(board_after, chess.engine.Limit(depth=12))
    score_after = -info_after["score"].relative.score(mate_score=10000)

    # Evaluate position after best move
    board_best = board.copy()
    board_best.push(best_move)
    info_best = engine.analyse(board_best, chess.engine.Limit(depth=12))
    score_best = -info_best["score"].relative.score(mate_score=10000)

    centipawn_loss = max(0, score_best - score_after)

    # Convert CPL to reward: 0 CPL -> 1.0, 200+ CPL -> -1.0
    reward = 1.0 - (centipawn_loss / 100.0)
    reward = max(-1.0, min(1.0, reward))

    return reward, centipawn_loss, True


def build_prompt(fen: str) -> str:
    """Build the prompt for the model given a chess position."""
    return f"{fen}\n\nWhat is your move? Output your move like this: \\box{{e2e4}}"


def sample_responses_sync(
    sampling_client,
    tokenizer,
    positions: list[str],
    group_size: int,
    max_tokens: int = 128,
) -> list[dict]:
    """Sample multiple responses per position for GRPO training."""
    all_samples = []

    for fen in tqdm(positions, desc="Sampling"):
        prompt = build_prompt(fen)
        prompt_input = ModelInput.from_ints(tokenizer.encode(prompt))
        params = SamplingParams(max_tokens=max_tokens, temperature=1.0)

        response = sampling_client.sample(
            prompt=prompt_input,
            num_samples=group_size,
            sampling_params=params,
        ).result()

        for seq in response.sequences:
            response_text = tokenizer.decode(seq.tokens)
            all_samples.append({
                "fen": fen,
                "prompt": prompt,
                "prompt_input": prompt_input,
                "response_tokens": seq.tokens,
                "response_text": response_text,
                "logprobs": seq.logprobs,
            })

    return all_samples


def evaluate_samples_with_stockfish(
    samples: list[dict],
    engine: chess.engine.SimpleEngine,
) -> list[dict]:
    """Evaluate all sampled moves with Stockfish and compute rewards."""
    for sample in samples:
        board = chess.Board(sample["fen"])
        move = parse_move_from_response(sample["response_text"], board)
        move_str = move.uci() if move else None
        reward, cpl, valid = compute_reward_and_cpl(sample["fen"], move_str, engine)
        sample["move"] = move_str
        sample["reward"] = reward
        sample["centipawn_loss"] = cpl
        sample["valid_move"] = valid
    return samples


def compute_grpo_advantages(samples: list[dict]) -> list[dict]:
    """Compute GRPO-style advantages: advantage = (reward - mean) / std within each group."""
    # Group samples by position
    groups = {}
    for sample in samples:
        fen = sample["fen"]
        if fen not in groups:
            groups[fen] = []
        groups[fen].append(sample)

    # Compute advantages within each group
    for group in groups.values():
        rewards = [s["reward"] for s in group]
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-8

        for sample in group:
            sample["advantage"] = (sample["reward"] - mean_reward) / std_reward

    return samples


def train_step_sync(
    training_client,
    samples: list[dict],
    learning_rate: float,
) -> dict:
    """Execute one GRPO training step."""
    # Prepare training data
    training_data = []
    for sample in samples:
        advantages = [sample["advantage"]] * len(sample["response_tokens"])
        datum = Datum(
            model_input=sample["prompt_input"],
            loss_fn_inputs={
                "target_tokens": TensorData(data=list(sample["response_tokens"]), dtype="int64"),
                "logprobs": TensorData(data=list(sample["logprobs"]), dtype="float32"),
                "advantages": TensorData(data=advantages, dtype="float32"),
            },
        )
        training_data.append(datum)

    # Forward-backward pass
    fwd_bwd_future = training_client.forward_backward(
        data=training_data,
        loss_fn="ppo",
        loss_fn_config={
            "clip_low_threshold": 0.8,
            "clip_high_threshold": 1.2,
        },
    )

    # Optimizer step
    optim_future = training_client.optim_step(AdamParams(learning_rate=learning_rate))

    fwd_bwd_result = fwd_bwd_future.result()
    optim_future.result()

    return {
        "loss": fwd_bwd_result.loss,
        "mean_reward": np.mean([s["reward"] for s in samples]),
        "mean_cpl": np.mean([s["centipawn_loss"] for s in samples]),
        "valid_rate": np.mean([s["valid_move"] for s in samples]),
    }


def evaluate_model(
    sampling_client,
    tokenizer,
    positions: list[str],
    engine: chess.engine.SimpleEngine,
    max_tokens: int = 128,
) -> dict:
    """Evaluate model on a set of positions (single sample per position)."""
    samples = sample_responses_sync(
        sampling_client, tokenizer, positions, group_size=1, max_tokens=max_tokens
    )
    samples = evaluate_samples_with_stockfish(samples, engine)

    return {
        "avg_cpl": np.mean([s["centipawn_loss"] for s in samples]),
        "valid_rate": np.mean([s["valid_move"] for s in samples]),
        "avg_reward": np.mean([s["reward"] for s in samples]),
    }


def plot_training_curve(metrics: list[dict], output_path: str):
    """Plot training steps vs chess performance."""
    steps = list(range(len(metrics)))
    avg_cpls = [m["eval_cpl"] for m in metrics]
    valid_rates = [m["eval_valid_rate"] * 100 for m in metrics]
    rewards = [m["mean_reward"] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Centipawn loss over training
    ax1 = axes[0]
    ax1.plot(steps, avg_cpls, "b-o", linewidth=2, markersize=6)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Avg Centipawn Loss")
    ax1.set_title("Chess Performance vs Training Steps\n(lower is better)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Expert (~2000 ELO)')
    ax1.axhline(y=100, color='orange', linestyle='--', alpha=0.5, label='Class B (~1700 ELO)')
    ax1.legend()

    # Valid move rate
    ax2 = axes[1]
    ax2.plot(steps, valid_rates, "g-o", linewidth=2, markersize=6)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Valid Move Rate (%)")
    ax2.set_title("Valid Move Rate vs Training Steps")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    # Training reward
    ax3 = axes[2]
    ax3.plot(steps, rewards, "r-o", linewidth=2, markersize=6)
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Mean Reward")
    ax3.set_title("Training Reward vs Steps")
    ax3.grid(True, alpha=0.3)

    plt.suptitle("GRPO Training Progress", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def run_grpo_training(config: Config):
    """Main GRPO training loop."""
    print(f"Starting GRPO training with config:")
    print(f"  base_model: {config.base_model}")
    print(f"  batch_size: {config.batch_size}, group_size: {config.group_size}")
    print(f"  learning_rate: {config.learning_rate}, max_steps: {config.max_steps}")

    # Load data
    positions = load_positions()
    print(f"Loaded {len(positions)} positions")

    # Split into train and eval
    np.random.seed(42)
    np.random.shuffle(positions)
    eval_positions = positions[:config.eval_positions]
    train_positions = positions[config.eval_positions:]

    # Initialize Tinker
    print("Initializing Tinker clients...")
    service_client = ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
        rank=config.lora_rank,
    )
    sampling_client = service_client.create_sampling_client(
        base_model=config.base_model
    )
    tokenizer = sampling_client.get_tokenizer()

    # Initialize Stockfish
    print("Initializing Stockfish...")
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")

    # Training metrics
    metrics = []

    # Initial evaluation
    print("\n=== Initial Evaluation ===")
    eval_result = evaluate_model(sampling_client, tokenizer, eval_positions, engine)
    print(f"Initial: CPL={eval_result['avg_cpl']:.1f}, Valid={eval_result['valid_rate']*100:.1f}%")

    for step in range(config.max_steps):
        print(f"\n=== Step {step + 1}/{config.max_steps} ===")

        # Sample batch of positions
        batch_positions = np.random.choice(
            train_positions, size=min(config.batch_size, len(train_positions)), replace=False
        ).tolist()

        # Sample responses
        print("Sampling responses...")
        samples = sample_responses_sync(
            sampling_client, tokenizer, batch_positions, config.group_size
        )

        # Evaluate with Stockfish
        print("Evaluating with Stockfish...")
        samples = evaluate_samples_with_stockfish(samples, engine)

        # Compute GRPO advantages
        samples = compute_grpo_advantages(samples)

        # Training step
        print("Training...")
        step_metrics = train_step_sync(training_client, samples, config.learning_rate)

        # Get updated sampling client with new weights
        sampling_client = training_client.save_weights_and_get_sampling_client(
            f"checkpoint-{step+1}"
        )

        # Evaluate on held-out positions
        print("Evaluating...")
        eval_result = evaluate_model(sampling_client, tokenizer, eval_positions, engine)

        step_metrics["eval_cpl"] = eval_result["avg_cpl"]
        step_metrics["eval_valid_rate"] = eval_result["valid_rate"]
        step_metrics["step"] = step + 1
        metrics.append(step_metrics)

        print(f"  Loss: {step_metrics['loss']:.4f}")
        print(f"  Train reward: {step_metrics['mean_reward']:.3f}")
        print(f"  Eval CPL: {eval_result['avg_cpl']:.1f}, Valid: {eval_result['valid_rate']*100:.1f}%")

    engine.quit()

    # Save results
    results_path = "outputs/experiment2_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "base_model": config.base_model,
                "batch_size": config.batch_size,
                "group_size": config.group_size,
                "learning_rate": config.learning_rate,
                "max_steps": config.max_steps,
            },
            "metrics": metrics,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot training curve
    plot_training_curve(metrics, "outputs/experiment2_training_curve.png")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GRPO training for chess with Tinker")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--group-size", type=int, default=4, help="GRPO group size")
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--eval-positions", type=int, default=20, help="Positions for evaluation")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="Base model")
    args = parser.parse_args()

    config = Config(
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.lr,
        max_steps=args.steps,
        eval_positions=args.eval_positions,
        base_model=args.model,
    )

    run_grpo_training(config)
