"""
SF-Owen High-Depth (D24-D30) Tactical & Strategic Distillation Generator.
Uses Stockfish 19 across parallel worker threads to generate deep evaluations.
"""

import os
import sys
import time
import struct
import random
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor

def get_default_sf_path():
    candidates = [
        "/home/hemesh/Documents/stockfish/stockfish-linux-x86-64-universal",
        "/kaggle/working/sf-src/src/stockfish",
        "/usr/games/stockfish",
        "stockfish"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "stockfish"

DEFAULT_SF_PATH = get_default_sf_path()
BOOK_PATH = "/home/hemesh/Videos/Owen/tools/book-200.epd"

# Piece mapping: standard FEN char to Owen/SF integer (0..11, 12=empty)
PIECE_CHAR_TO_ID = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def fen_to_raw_board(fen: str):
    parts = fen.split()
    board_part = parts[0]
    stm_part = parts[1] if len(parts) > 1 else 'w'
    
    board = [12] * 64
    ranks = board_part.split('/')
    for r_idx, rank_str in enumerate(ranks):
        rank = 7 - r_idx
        file = 0
        for ch in rank_str:
            if ch.isdigit():
                file += int(ch)
            elif ch in PIECE_CHAR_TO_ID:
                sq = rank * 8 + file
                board[sq] = PIECE_CHAR_TO_ID[ch]
                file += 1
                
    stm = 0 if stm_part == 'w' else 1
    return bytes(board), stm


class StockfishWorker:
    def __init__(self, sf_path=None):
        if not sf_path:
            sf_path = DEFAULT_SF_PATH
        self.proc = subprocess.Popen(
            [sf_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
        self.send("uci")
        self.send("setoption name Hash value 32")
        self.send("isready")
        self._wait_ready()

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_ready(self):
        while True:
            line = self.proc.stdout.readline()
            if "readyok" in line:
                break

    def evaluate_position(self, fen: str, depth: int = 24):
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        score_cp = 0
        bestmove = None
        
        while True:
            line = self.proc.stdout.readline().strip()
            if not line:
                continue
            if line.startswith("info") and "score" in line:
                tokens = line.split()
                if "cp" in tokens:
                    cp_idx = tokens.index("cp")
                    if cp_idx + 1 < len(tokens):
                        try:
                            score_cp = int(tokens[cp_idx + 1])
                        except ValueError:
                            pass
                elif "mate" in tokens:
                    mate_idx = tokens.index("mate")
                    if mate_idx + 1 < len(tokens):
                        try:
                            mate_dist = int(tokens[mate_idx + 1])
                            score_cp = 10000 if mate_dist > 0 else -10000
                        except ValueError:
                            pass
            if line.startswith("bestmove"):
                bestmove = line.split()[1]
                break
                
        return score_cp, bestmove

    def close(self):
        try:
            self.send("quit")
            self.proc.terminate()
        except Exception:
            pass


def generate_batch(worker_id: int, fens: list, depth: int, out_bin_path: str, sf_path: str = None):
    worker = StockfishWorker(sf_path=sf_path)
    records = []
    
    for idx, fen in enumerate(fens):
        score_cp, bestmove = worker.evaluate_position(fen, depth=depth)
        board_bytes, stm = fen_to_raw_board(fen)
        
        # SDataRecord 69-byte layout:
        # board[64] uint8, stm uint8, eval int16, result uint8, ply uint8
        score_clamped = max(-10000, min(10000, score_cp))
        result = 1 # neutral/draw placeholder
        ply = 10
        
        rec = struct.pack("<64sBhBB", board_bytes, stm, score_clamped, result, ply)
        records.append(rec)
        
    worker.close()
    
    with open(out_bin_path, "ab") as f:
        for r in records:
            f.write(r)
            
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Deep D24-D30 Distillation Dataset Generator")
    parser.add_argument("--engine", type=str, default=DEFAULT_SF_PATH, help="Path to Stockfish engine")
    parser.add_argument("--depth", type=int, default=24, help="Search depth per position")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads")
    parser.add_argument("--total-positions", type=int, default=10000, help="Total positions to generate")
    parser.add_argument("--output", type=str, default="/home/hemesh/Videos/sf-owen/data/d24_distill.bin", help="Output .bin file")
    args = parser.parse_args()

    print("=" * 60)
    print("SF-OWEN DEEP D24-D30 DISTILLATION DATASET GENERATOR")
    print(f"Engine: {args.engine}")
    print(f"Workers: {args.workers}, Target Depth: D{args.depth}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Load openings
    openings = []
    book_candidates = [
        BOOK_PATH,
        "/kaggle/working/sf-owen/tools/book-200.epd",
        "/kaggle/working/book-200.epd"
    ]
    for b in book_candidates:
        if os.path.exists(b):
            with open(b, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        openings.append(line.split(";")[0].strip())
            break
    if not openings:
        openings = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

    # Multiply openings with random perturbations / variations
    fens_to_process = []
    while len(fens_to_process) < args.total_positions:
        fens_to_process.extend(openings)
    fens_to_process = fens_to_process[:args.total_positions]
    random.shuffle(fens_to_process)

    # Chunk across workers
    chunk_size = len(fens_to_process) // args.workers
    chunks = [fens_to_process[i:i + chunk_size] for i in range(0, len(fens_to_process), chunk_size)]

    t0 = time.time()
    total_generated = 0

    print(f"Starting {args.workers} workers for {len(fens_to_process):,} positions...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for w_id, chunk in enumerate(chunks):
            futures.append(executor.submit(generate_batch, w_id, chunk, args.depth, args.output, args.engine))
            
        for f in futures:
            total_generated += f.result()
            print(f"Progress: {total_generated:,} / {len(fens_to_process):,} positions ({time.time()-t0:.1f}s)")

    print(f"\nSuccessfully generated {total_generated:,} deep D{args.depth} positions in {time.time()-t0:.1f}s!")
    print(f"Saved to {args.output} ({os.path.getsize(args.output):,} bytes)!")


if __name__ == "__main__":
    main()
