#!/usr/bin/env python3
"""
Sample "interesting" chess positions by random play + Stockfish filtering.

Requires:
  pip install python-chess

Usage:
  python sample_positions.py --n 50
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import chess
import chess.engine


@dataclass(frozen=True)
class Filters:
    min_plies: int = 1
    max_plies: int = 80
    min_total_pieces: int = 6
    nodes: int = 12000


def total_piece_count(board: chess.Board) -> int:
    # count all pieces on board (including kings)
    return len(board.piece_map())


def weighted_random_move(board: chess.Board) -> chess.Move:
    """Bias random playouts a bit toward forcing moves so we reach richer middlegames."""
    moves = list(board.legal_moves)
    if not moves:
        raise RuntimeError("No legal moves")

    weights = []
    for mv in moves:
        w = 1.0
        if board.is_capture(mv):
            w *= 2.5
        board.push(mv)
        try:
            if board.is_check():
                w *= 2.0
        finally:
            board.pop()
        # small preference for developing moves early: avoid too many pawn shuffles
        piece = board.piece_at(mv.from_square)
        if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            w *= 1.2
        weights.append(w)

    return random.choices(moves, weights=weights, k=1)[0]


def score_to_cp(score: chess.engine.PovScore, turn: chess.Color) -> Optional[int]:
    """
    Convert engine score to centipawns if possible.
    Returns None for mate scores (you can choose to keep/discard those).
    """
    s = score.pov(turn)
    if s.is_mate():
        return None
    return int(s.score(mate_score=100000))


def analyze_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    filters: Filters,
) -> Optional[Tuple[str, int]]:
    """
    Returns (fen, cp_eval) if it passes filters, else None.
    cp_eval is from side-to-move POV.
    """
    if board.is_game_over(claim_draw=True):
        return None
    if total_piece_count(board) < filters.min_total_pieces:
        return None

    limit = chess.engine.Limit(nodes=filters.nodes)
    info = engine.analyse(board, limit)

    sc = info.get("score")
    if sc is None:
        return None
    cp = score_to_cp(sc, board.turn)
    if cp is None:
        # skip mate positions
        return None

    return (board.fen(), cp)


def sample_one(board_rng: random.Random, filters: Filters) -> chess.Board:
    board = chess.Board()
    plies = board_rng.randint(filters.min_plies, filters.max_plies)
    for _ in range(plies):
        if board.is_game_over(claim_draw=True):
            break
        mv = weighted_random_move(board)
        board.push(mv)
    return board


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="How many positions to output")
    args = ap.parse_args()

    filters = Filters()
    output = "outputs/boards.json"

    rng = random.Random(0)
    
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")  # UCI via python-chess
    try:
        found = 0
        attempts = 0
        positions = []
        while found < args.n:
            attempts += 1
            board = sample_one(rng, filters)
            res = analyze_position(engine, board, filters)
            if res is None:
                continue
            fen, cp = res
            found += 1
            print(f"{found}/{args.n}  eval_cp={cp:+d}")
            positions.append(fen)
        with open(output, "w") as f:
            json.dump(positions, f, indent=2)
        print(f"Saved {found} positions to {output}")
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
