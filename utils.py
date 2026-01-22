"""Shared utilities for chess experiments."""

import json
import re

import chess
import chess.engine

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

# OpenRouter configuration
INFERENCE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-oss-20b"
MODEL = "Qwen/Qwen3-30B-A3B"


def load_positions(path: str = "outputs/boards.json") -> list[tuple[str, str]]:
    """Load positions from boards.json file (list of FEN strings)."""
    with open(path) as f:
        fens = json.load(f)
    return [(fen, f"Position {i+1}") for i, fen in enumerate(fens)]


def parse_move_from_response(response: str, board: chess.Board) -> chess.Move | None:
    """
    Extract a chess move from the model's response.
    Expects format: [move: e2e4]
    """
    text = response.strip().lower()

    # Look for [move: ...] format
    move_pattern = r'\[move:\s*([a-h][1-8][a-h][1-8][qrbn]?)\]'
    matches = re.findall(move_pattern, text)
    if len(matches) >= 1:
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
    prompt_text = f"""Here is a chess board:\n {board_text}
    Please quickly [ie /nothink] output a good move (and nothing else). 
    If your move is e2e4 you write [move: e2e4]."""
    prompt = Prompt(messages=[ChatMessage(content=prompt_text, role=MessageRole.user)])

    # Use reasoning={"effort": "low"} to disable CoT for GPT models
    # Note: For OpenAI reasoning models, use max_completion_tokens instead of max_tokens
    extra_kwargs = {"extra_body": { "reasoning": {"effort": "medium"}}} 
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


def centipawn_loss_to_elo_estimate(avg_cpl: float) -> int:
    """
    Rough estimate of ELO based on average centipawn loss.
    Based on empirical data from chess databases.
    """
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


def evaluate_with_stockfish(result: dict, engine: chess.engine.SimpleEngine) -> dict:
    """
    Evaluate a model's move with Stockfish.
    
    Args:
        result: dict with "position" (FEN) and "move" (UCI string or None)
        engine: Stockfish engine instance
    
    Returns:
        result dict merged with evaluation data (centipawn_loss, best_move, etc.)
    """
    board = chess.Board(result["position"])
    move_str = result["move"]
    
    if move_str is None:
        return {
            **result,
            "valid_move": False,
            "centipawn_loss": 500,  # Penalty for invalid move
        }
    
    move = chess.Move.from_uci(move_str)
    
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
        **result,
        "valid_move": True,
        "move": move.uci(),
        "best_move": best_move.uci(),
        "is_best": move == best_move,
        "centipawn_loss": max(0, centipawn_loss),
        "score_after": score_after,
        "score_best": score_best,
    }
