"""
SF-Owen NNUE Binary Loader & Exporter
Handles reading and writing Stockfish NNUE .nnue files with LEB128 compression.
"""

import os
import struct
import ctypes
import numpy as np
import torch

# Load the fast C LEB128 decoder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.path.join(CURRENT_DIR, "leb128_fast.so")

if not os.path.exists(SO_PATH):
    # Compile on the fly if needed
    c_src = os.path.join(CURRENT_DIR, "leb128_fast.c")
    os.system(f"gcc -O3 -fPIC -shared '{c_src}' -o '{SO_PATH}'")

_lib = ctypes.CDLL(SO_PATH)
_lib.decode_leb128_i16.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
_lib.decode_leb128_i16.restype = ctypes.c_int
_lib.decode_leb128_i32.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
_lib.decode_leb128_i32.restype = ctypes.c_int

_lib.encode_leb128_i16.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
_lib.encode_leb128_i16.restype = ctypes.c_int

_lib.encode_leb128_i32.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
_lib.encode_leb128_i32.restype = ctypes.c_int


def _read_leb128(f, count, dtype=np.int16):
    magic = f.read(17)
    if magic != b"COMPRESSED_LEB128":
        raise ValueError(f"Invalid LEB128 magic: {magic}")
    bytes_left = struct.unpack("<I", f.read(4))[0]
    compressed = f.read(bytes_left)
    if len(compressed) != bytes_left:
        raise ValueError(f"Truncated LEB128 stream: expected {bytes_left}, got {len(compressed)}")
    
    out = np.empty(count, dtype=dtype)
    if dtype == np.int16:
        res = _lib.decode_leb128_i16(compressed, bytes_left, out.ctypes.data, count)
    elif dtype == np.int32:
        res = _lib.decode_leb128_i32(compressed, bytes_left, out.ctypes.data, count)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")
    
    if res != 0:
        raise RuntimeError(f"LEB128 decoding failed with error code {res}")
    return out


def _write_leb128(f, arr):
    f.write(b"COMPRESSED_LEB128")
    flat = np.ascontiguousarray(arr).ravel()
    count = len(flat)
    
    # Preallocate buffer (LEB128 takes max 5 bytes per 32-bit int, average ~1.5 bytes)
    max_buf_len = count * 4 + 1024
    buf = bytearray(max_buf_len)
    buf_arr = (ctypes.c_char * max_buf_len).from_buffer(buf)
    out_len = ctypes.c_uint32(0)
    
    if flat.dtype == np.int16:
        res = _lib.encode_leb128_i16(flat.ctypes.data, count, buf_arr, max_buf_len, ctypes.byref(out_len))
    elif flat.dtype == np.int32:
        res = _lib.encode_leb128_i32(flat.ctypes.data, count, buf_arr, max_buf_len, ctypes.byref(out_len))
    else:
        # Cast to int32
        arr32 = flat.astype(np.int32)
        res = _lib.encode_leb128_i32(arr32.ctypes.data, count, buf_arr, max_buf_len, ctypes.byref(out_len))
        
    if res != 0:
        raise RuntimeError(f"Fast LEB128 encoding failed: {res}")
        
    encoded_len = out_len.value
    f.write(struct.pack("<I", encoded_len))
    f.write(memoryview(buf)[:encoded_len])


def load_sf_nnue(file_path: str):
    """
    Loads a Stockfish NNUE binary (.nnue) into a structured Python dictionary containing
    all weights and biases as NumPy arrays / PyTorch tensors.
    """
    weights = {}
    with open(file_path, "rb") as f:
        version, net_hash, desc_len = struct.unpack("<III", f.read(12))
        desc = f.read(desc_len).decode("utf-8", errors="ignore")
        
        weights["header"] = {
            "version": version,
            "hash": net_hash,
            "description": desc
        }
        
        ft_hash = struct.unpack("<I", f.read(4))[0]
        weights["ft_hash"] = ft_hash
        
        # Feature Transformer Weights
        weights["ft_biases"] = _read_leb128(f, 1024, np.int16)
        weights["threat_weights"] = np.frombuffer(f.read(59808 * 1024), dtype=np.int8).reshape(59808, 1024).copy()
        weights["threat_psqt"] = _read_leb128(f, 59808 * 8, np.int32).reshape(59808, 8)
        weights["pp_weights"] = np.frombuffer(f.read(4560 * 1024), dtype=np.int8).reshape(4560, 1024).copy()
        weights["pp_psqt"] = _read_leb128(f, 4560 * 8, np.int32).reshape(4560, 8)
        weights["halfka_weights"] = _read_leb128(f, 22528 * 1024, np.int16).reshape(22528, 1024)
        weights["halfka_psqt"] = _read_leb128(f, 22528 * 8, np.int32).reshape(22528, 8)
        
        # 8 King-Piece Layer Stacks
        stacks = []
        for stack_idx in range(8):
            stack_hash = struct.unpack("<I", f.read(4))[0]
            fc0_b = np.frombuffer(f.read(32 * 4), dtype=np.int32).copy()
            fc0_w = np.frombuffer(f.read(32 * 1024), dtype=np.int8).reshape(1024, 32).T.copy()
            
            fc1_b = np.frombuffer(f.read(32 * 4), dtype=np.int32).copy()
            fc1_w = np.frombuffer(f.read(32 * 64), dtype=np.int8).reshape(32, 64).copy()
            
            fc2_b = np.frombuffer(f.read(1 * 4), dtype=np.int32).copy()
            fc2_w = np.frombuffer(f.read(1 * 128), dtype=np.int8).reshape(1, 128).copy()
            
            stacks.append({
                "hash": stack_hash,
                "fc0_bias": fc0_b,
                "fc0_weight": fc0_w,
                "fc1_bias": fc1_b,
                "fc1_weight": fc1_w,
                "fc2_bias": fc2_b,
                "fc2_weight": fc2_w,
            })
        
        weights["layer_stacks"] = stacks
        remaining = f.read()
        if len(remaining) != 0:
            raise ValueError(f"Extra {len(remaining)} trailing bytes in NNUE file!")
            
    return weights


def save_sf_nnue(weights: dict, file_path: str, description: str = "SF-Owen fine-tuned net"):
    """
    Saves weights dictionary back to Stockfish NNUE binary (.nnue) format.
    """
    desc_bytes = description.encode("utf-8")
    with open(file_path, "wb") as f:
        # Header
        ver = weights["header"].get("version", 0x6a448afa)
        net_hash = weights["header"].get("hash", 0xa85b2205)
        f.write(struct.pack("<III", ver, net_hash, len(desc_bytes)))
        f.write(desc_bytes)
        
        # FT Hash
        f.write(struct.pack("<I", weights.get("ft_hash", 0xcb685313)))
        
        # FT Weights
        _write_leb128(f, weights["ft_biases"].astype(np.int16))
        f.write(weights["threat_weights"].astype(np.int8).tobytes())
        _write_leb128(f, weights["threat_psqt"].astype(np.int32))
        f.write(weights["pp_weights"].astype(np.int8).tobytes())
        _write_leb128(f, weights["pp_psqt"].astype(np.int32))
        _write_leb128(f, weights["halfka_weights"].astype(np.int16))
        _write_leb128(f, weights["halfka_psqt"].astype(np.int32))
        
        # Stacks
        for stack in weights["layer_stacks"]:
            f.write(struct.pack("<I", stack.get("hash", 0x63337116)))
            f.write(stack["fc0_bias"].astype(np.int32).tobytes())
            f.write(stack["fc0_weight"].astype(np.int8).tobytes())
            f.write(stack["fc1_bias"].astype(np.int32).tobytes())
            f.write(stack["fc1_weight"].astype(np.int8).tobytes())
            f.write(stack["fc2_bias"].astype(np.int32).tobytes())
            f.write(stack["fc2_weight"].astype(np.int8).tobytes())
