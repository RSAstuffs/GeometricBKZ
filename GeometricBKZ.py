"""
Standalone BKZ Module
=====================

Custom BKZ implementation using geometric SVP oracle.
Ported from GeometricLLL.
"""

import numpy as np
from typing import Optional, Tuple
import math
from geometric_lll import GeometricLLL

# Optional numba JIT for hot inner loops (not required)
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

INT64_LIMIT = 1 << 62


def _try_cast_int64(basis: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Try to cast a basis to np.int64 safely.

    Returns (basis_array, used_int64_flag). If any entry would overflow
    int64 (or is non-integer), returns the original array and False.
    """
    # If it's already int64, keep it
    if isinstance(basis, np.ndarray) and basis.dtype == np.int64:
        return basis, True

    # If shapeless or empty, return as-is
    try:
        flat = basis.flat
    except Exception:
        return basis, False

    # Check every entry fits in signed 62-bit to leave headroom
    for x in flat:
        # allow numpy ints too
        if not isinstance(x, (int, np.integer)):
            return basis, False
        if abs(int(x)) >= INT64_LIMIT:
            return basis, False

    # Safe to cast
    try:
        arr = np.array(basis, dtype=np.int64, copy=True)
        return arr, True
    except Exception:
        return basis, False


def _reduce_vector_int64(v, w):
    n = v.shape[0]
    dot = 0
    norm_sq = 0
    for i in range(n):
        dot += v[i] * w[i]
        norm_sq += w[i] * w[i]
    if norm_sq == 0:
        return v.copy()
    if dot >= 0:
        coeff = (dot + norm_sq // 2) // norm_sq
    else:
        coeff = -((-dot + norm_sq // 2) // norm_sq)
    if coeff == 0:
        return v.copy()
    res = v.copy()
    for i in range(n):
        res[i] = res[i] - coeff * w[i]
    return res

if NUMBA_AVAILABLE:
    # If numba is installed, JIT the int64 reduction for extra speed
    try:
        _reduce_vector_int64 = njit(cache=True)(_reduce_vector_int64)
    except Exception:
        # ignore numba failures and keep Python version
        pass

def bkz_reduce(basis: np.ndarray, block_size: int = 20, max_tours: int = 10, verbose: bool = True, N: int = 1) -> np.ndarray:
    """
    Custom BKZ reduction using geometric SVP oracle.
    
    Args:
        basis: Input lattice basis (n x m matrix)
        block_size: Block size for BKZ
        max_tours: Maximum number of tours
        verbose: Print progress
        
    Returns:
        BKZ-reduced basis
    """
    if basis is None or len(basis) == 0:
        return basis
    
    # Try to use int64 fast path when possible
    basis = np.array(basis, copy=True)
    basis, _ = _try_cast_int64(basis)
    n = len(basis)
    
    block_size = min(block_size, n)
    
    if verbose:
        print(f"[*] Running Geometric BKZ on {n}x{basis.shape[1]} lattice...")
        print(f"[*] Block size: {block_size}, Max tours: {max_tours}")
    
    # Initial geometric reduction using GeometricLLL to match geometric_lll.py
    try:
        g_full = GeometricLLL(N, basis=basis.copy())
        basis = g_full.run_geometric_reduction(verbose=False, num_passes=1)
    except Exception:
        basis = _geometric_lll_reduce(basis, verbose=False)
    
    best_basis = basis.copy()
    best_norm = _get_shortest_norm(basis)
    
    for tour in range(max_tours):
        if verbose:
            print(f"\n[*] === BKZ TOUR {tour + 1}/{max_tours} ===")
        
        tour_improved = False
        
        # Process each block
        for k in range(n - 1):
            block_end = min(k + block_size, n)
            block_len = block_end - k
            
            if block_len < 2:
                continue
            
            # Extract block
            block = basis[k:block_end].copy()

            # Use GeometricLLL on the block (projecting would be more accurate)
            try:
                block_lll = GeometricLLL(N, basis=block)
                # First try the expand-and-recompress routine which can expose
                # shorter vectors on structured lattices.
                try:
                    expanded = block_lll._expand_and_recompress_geometric(verbose=False)
                except Exception:
                    expanded = None

                if expanded is not None and len(expanded) > 0:
                    # Run geometric reduction on the expanded basis to refine it
                    block_lll.basis = expanded
                    reduced_block = block_lll.run_geometric_reduction(verbose=False, num_passes=4)
                else:
                    # Fallback: plain geometric reduction
                    reduced_block = block_lll.run_geometric_reduction(verbose=False, num_passes=4)

                # Get shortest vector
                shortest_idx = 0
                shortest_norm = _vector_norm_sq(reduced_block[0])
                for i in range(1, len(reduced_block)):
                    norm = _vector_norm_sq(reduced_block[i])
                    if norm > 0 and (shortest_norm == 0 or norm < shortest_norm):
                        shortest_norm = norm
                        shortest_idx = i

                shortest_vector = reduced_block[shortest_idx].copy()
            except Exception:
                # fallback
                shortest_vector = _geometric_svp_oracle(block)
                shortest_norm = _vector_norm_sq(shortest_vector)
            
            # Check if this is shorter than current first vector in block
            current_norm = _vector_norm_sq(basis[k])
            
            if shortest_norm > 0 and shortest_norm < current_norm:
                # Insert shortest vector at the beginning of block
                # Shift vectors
                for i in range(block_end - 1, k, -1):
                    basis[i] = basis[i-1].copy()
                basis[k] = shortest_vector
                
                # Re-reduce affected portion
                for i in range(k, min(k + block_size + 1, n)):
                    for j in range(i):
                        basis[i] = _reduce_vector(basis[i], basis[j])
                
                tour_improved = True
                
                if verbose:
                    bits = shortest_norm.bit_length() // 2
                    print(f"[*] Block {k}: found shorter vector ~2^{bits} bits")
            elif verbose:
                print(f"[*] Block {k}: no shorter vector found")        # Full re-reduction
        for i in range(1, n):
            for j in range(i):
                basis[i] = _reduce_vector(basis[i], basis[j])
        
        # Track best
        current_best = _get_shortest_norm(basis)
        if current_best and (best_norm is None or current_best < best_norm):
            best_norm = current_best
            best_basis = basis.copy()
            if verbose:
                bits = best_norm.bit_length() // 2
                print(f"[*] ★ New best: ~2^{bits} bits")
        
        if not tour_improved:
            if verbose:
                print(f"[*] No improvement in tour {tour + 1}, stopping early")
            break
    
    if verbose:
        if best_norm:
            bits = best_norm.bit_length() // 2
            print(f"\n[*] BKZ complete. Best shortest: ~2^{bits} bits")
        else:
            print(f"\n[*] BKZ complete.")
    
    return best_basis

def _geometric_lll_reduce(basis: np.ndarray, verbose: bool = False, num_passes: int = 1) -> np.ndarray:
    """Full geometric LLL reduction with hierarchical compression."""
    for pass_num in range(num_passes):
        basis = _single_geometric_pass(basis, verbose and pass_num == 0)
    return basis

def _single_geometric_pass(basis: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Single pass of geometric LLL reduction with hierarchical compression."""
    if basis is None or len(basis) == 0:
        return basis

    basis = np.array(basis, copy=True)
    basis, used_int64 = _try_cast_int64(basis)
    n = len(basis)
    m = basis.shape[1] if len(basis.shape) > 1 else n

    if verbose:
        print(f"[*] Hierarchical Geometric Compression on {n}x{m} lattice...")

    def compress_square(v0, v1, v2, v3):
        """Compress 4 vectors geometrically - O(1) operation"""
        # Invert to point same direction as v0
        if np.dot(v0, v1) < 0: v1 = -v1
        if np.dot(v0, v2) < 0: v2 = -v2
        if np.dot(v0, v3) < 0: v3 = -v3

        # Fuse A-B: reduce v1 against v0
        d00 = np.dot(v0, v0)
        if d00 > 0:
            r = (np.dot(v1, v0) + d00 // 2) // d00
            if r != 0: v1 = v1 - r * v0

        # Fuse C-D: reduce v3 against v2
        d22 = np.dot(v2, v2)
        if d22 > 0:
            r = (np.dot(v3, v2) + d22 // 2) // d22
            if r != 0: v3 = v3 - r * v2

        # Compress to point: reduce v2 against v0
        if d00 > 0:
            r = (np.dot(v2, v0) + d00 // 2) // d00
            if r != 0: v2 = v2 - r * v0

        # Also reduce v3 against v0
        if d00 > 0:
            r = (np.dot(v3, v0) + d00 // 2) // d00
            if r != 0: v3 = v3 - r * v0

        return v0, v1, v2, v3

    def compress_pair(v0, v1):
        """Compress 2 vectors - O(1)"""
        if np.dot(v0, v1) < 0: v1 = -v1
        d00 = np.dot(v0, v0)
        if d00 > 0:
            r = (np.dot(v1, v0) + d00 // 2) // d00
            if r != 0: v1 = v1 - r * v0
        return v0, v1

    # === HIERARCHICAL COMPRESSION ===
    # Process in groups of 4 (like the geometric square)

    # Level 1: Compress all groups of 4
    i = 0
    while i + 3 < n:
        basis[i], basis[i+1], basis[i+2], basis[i+3] = compress_square(
            basis[i], basis[i+1], basis[i+2], basis[i+3]
        )
        i += 4

    # Handle remaining 2-3 vectors
    if i + 1 < n:
        basis[i], basis[i+1] = compress_pair(basis[i], basis[i+1])
        if i + 2 < n:
            basis[i], basis[i+2] = compress_pair(basis[i], basis[i+2])

    # Level 2: Compress across groups (reduce each group leader against first)
    for i in range(4, n, 4):
        if np.dot(basis[0], basis[i]) < 0:
            basis[i] = -basis[i]
        d00 = np.dot(basis[0], basis[0])
        if d00 > 0:
            r = (np.dot(basis[i], basis[0]) + d00 // 2) // d00
            if r != 0:
                basis[i] = basis[i] - r * basis[0]

    # Sort by norm
    norms = [(np.dot(basis[i], basis[i]), i) for i in range(n)]
    norms.sort()
    basis = np.array([basis[idx] for _, idx in norms], dtype=object)

    if verbose:
        shortest = norms[0][0]
        bits = shortest.bit_length() // 2 if shortest and shortest > 0 else 0
        print(f"[*] Shortest: ~2^{bits} bits")

    return basis

def _geometric_svp_oracle(block_basis):
    """
    Geometric SVP Oracle using GeometricLLL.run_geometric_reduction (same as geometric_lll.py).
    Falls back to returning the shortest existing vector if anything fails.
    """
    if block_basis is None or len(block_basis) == 0:
        return np.zeros(block_basis.shape[1] if hasattr(block_basis, 'shape') else 0, dtype=object)

    # Try to use the GeometricLLL reduction on the block (matches geometric_lll.run_bkz)
    try:
        block = np.array(block_basis, dtype=object).copy()
        # GeometricLLL expects an N parameter; use 1 as a dummy since run_geometric_reduction
        # does not depend on N for pure geometric compression.
        g = GeometricLLL(1, basis=block)
        reduced_block = g.run_geometric_reduction(verbose=False, num_passes=4)

        # Choose shortest
        shortest_idx = 0
        shortest_norm = _vector_norm_sq(reduced_block[0])
        for i in range(1, len(reduced_block)):
            norm = _vector_norm_sq(reduced_block[i])
            if norm > 0 and (shortest_norm == 0 or norm < shortest_norm):
                shortest_norm = norm
                shortest_idx = i

        return reduced_block[shortest_idx].copy()
    except Exception:
        # Fallback: pick shortest existing vector
        norms = [_vector_norm_sq(v) for v in block_basis]
        min_idx = int(np.argmin(norms))
        return block_basis[min_idx].copy()

def _reduce_vector(v, w):
    """Reduce v with respect to w."""
    # Fast int64 path using numba
    try:
        if isinstance(v, np.ndarray) and isinstance(w, np.ndarray) and v.dtype == np.int64 and w.dtype == np.int64:
            return _reduce_vector_int64(v, w)
    except Exception:
        # fall through to generic path
        pass

    # Generic fallback (object or float arrays)
    dot = np.dot(v, w)
    norm_sq = np.dot(w, w)
    if norm_sq == 0:
        return v
    coeff = round(dot / norm_sq)
    return v - coeff * w

def _vector_norm_sq(v):
    """Squared norm of vector."""
    return np.dot(v, v)

def _get_shortest_norm(basis) -> Optional[int]:
    """Get squared norm of shortest non-zero vector."""
    shortest = None
    for v in basis:
        norm_sq = _vector_norm_sq(v)
        if norm_sq > 0:
            if shortest is None or norm_sq < shortest:
                shortest = norm_sq
    return shortest
