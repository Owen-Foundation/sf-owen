"""
Stockfish feature extractor Python wrapper using fast C++ bitboards.
Extracts HalfKAv2_hm, FullThreats, and PP_3Wide active indices directly from FEN strings.
"""

import os
import ctypes
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.path.join(CURRENT_DIR, "extract_features.so")

if not os.path.exists(SO_PATH):
    raise FileNotFoundError(f"Feature extractor shared library not found at {SO_PATH}")

_lib = ctypes.CDLL(SO_PATH)


class FeatureList(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_int),
        ("indices", ctypes.c_uint32 * 512),
    ]


class BoardFeatures(ctypes.Structure):
    _fields_ = [
        ("bucket", ctypes.c_int),
        ("side_to_move", ctypes.c_int),
        ("halfka_us", FeatureList),
        ("halfka_them", FeatureList),
        ("threat_us", FeatureList),
        ("threat_them", FeatureList),
        ("pp_us", FeatureList),
        ("pp_them", FeatureList),
    ]


_lib.init_sf_bitboards.argtypes = []
_lib.init_sf_bitboards.restype = None

_lib.extract_features_fen.argtypes = [ctypes.c_char_p, ctypes.POINTER(BoardFeatures)]
_lib.extract_features_fen.restype = ctypes.c_int

# Initialize Stockfish bitboards once
_lib.init_sf_bitboards()


def extract_features(fen: str):
    """
    Extracts active NNUE features from a FEN string.
    Returns: dict with bucket, side_to_move, and NumPy int64 index arrays for us and them.
    """
    bf = BoardFeatures()
    res = _lib.extract_features_fen(fen.encode("utf-8"), ctypes.byref(bf))
    if res != 0:
        raise ValueError(f"Failed to extract features for FEN: {fen}")

    return {
        "bucket": bf.bucket,
        "side_to_move": bf.side_to_move,
        "halfka_us": np.array(bf.halfka_us.indices[: bf.halfka_us.count], dtype=np.int64),
        "halfka_them": np.array(bf.halfka_them.indices[: bf.halfka_them.count], dtype=np.int64),
        "threat_us": np.array(bf.threat_us.indices[: bf.threat_us.count], dtype=np.int64),
        "threat_them": np.array(bf.threat_them.indices[: bf.threat_them.count], dtype=np.int64),
        "pp_us": np.array(bf.pp_us.indices[: bf.pp_us.count], dtype=np.int64),
        "pp_them": np.array(bf.pp_them.indices[: bf.pp_them.count], dtype=np.int64),
    }
