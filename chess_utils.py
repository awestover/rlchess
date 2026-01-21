"""Chess move evaluation utilities using Stockfish."""

import chess
import chess.engine


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
