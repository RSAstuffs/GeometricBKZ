"""
Standalone BKZ Module - Optimized
==================================

Custom BKZ implementation using geometric SVP oracle.
Optimized for speed with GSO caching, local re-reduction, and early termination.
"""

import numpy as np
from typing import Optional, Tuple, List
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


class GSOCache:
    """Cache for Gram-Schmidt orthogonalization coefficients."""
    
    def __init__(self, basis: np.ndarray):
        self.n = len(basis)
        self.norm_cache = np.zeros(self.n, dtype=object)
        self.needs_update = [True] * self.n
        self.basis = basis
        self._update_all()
    
    def _update_all(self):
        """Recompute all norms."""
        for i in range(self.n):
            if self.needs_update[i]:
                self.norm_cache[i] = _vector_norm_sq(self.basis[i])
                self.needs_update[i] = False
    
    def update_range(self, start: int, end: int):
        """Mark range as needing update."""
        for i in range(max(0, start), min(self.n, end)):
            self.needs_update[i] = True
    
    def get_norm(self, i: int):
        """Get cached norm, updating if necessary."""
        if self.needs_update[i]:
            self.norm_cache[i] = _vector_norm_sq(self.basis[i])
            self.needs_update[i] = False
        return self.norm_cache[i]
    
    def get_shortest_index(self) -> Tuple[int, Optional[int]]:
        """Get index and norm of shortest non-zero vector."""
        shortest_idx = 0
        shortest_norm = None
        
        for i in range(self.n):
            norm = self.get_norm(i)
            if norm > 0:
                if shortest_norm is None or norm < shortest_norm:
                    shortest_norm = norm
                    shortest_idx = i
        
        return shortest_idx, shortest_norm


def bkz_reduce(basis: np.ndarray, block_size: int = 20, max_tours: int = 10, 
               verbose: bool = True, N: int = 1) -> np.ndarray:
    """
    Custom BKZ reduction using geometric SVP oracle - OPTIMIZED VERSION.
    
    Optimizations:
    - GSO caching to avoid redundant norm computations
    - Local re-reduction instead of full O(n²) re-reduction
    - Early termination in SVP oracle
    - Block change tracking to skip unchanged blocks
    - Adaptive block processing with jump strategy
    - Pruned expansion for faster convergence
    
    Args:
        basis: Input lattice basis (n x m matrix)
        block_size: Block size for BKZ
        max_tours: Maximum number of tours
        verbose: Print progress
        N: Parameter for GeometricLLL (default 1)
        
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
        print(f"[*] Running Optimized Geometric BKZ on {n}x{basis.shape[1]} lattice...")
        print(f"[*] Block size: {block_size}, Max tours: {max_tours}")
    
    # Initial geometric reduction using GeometricLLL
    try:
        g_full = GeometricLLL(N, basis=basis.copy())
        basis = g_full.run_geometric_reduction(verbose=False, num_passes=1)
    except Exception:
        basis = _geometric_lll_reduce(basis, verbose=False)
    
    # Initialize GSO cache
    gso_cache = GSOCache(basis)
    
    best_basis = basis.copy()
    _, best_norm = gso_cache.get_shortest_index()
    
    # Track which blocks need processing
    block_changed = [True] * n
    
    for tour in range(max_tours):
        if verbose:
            print(f"\n[*] === BKZ TOUR {tour + 1}/{max_tours} ===")
        
        tour_improved = False
        blocks_processed = 0
        blocks_skipped = 0
        
        # Jump strategy: process blocks with adaptive stepping
        k = 0
        while k < n - 1:
            block_end = min(k + block_size, n)
            block_len = block_end - k
            
            if block_len < 2:
                k += 1
                continue
            
            # Skip unchanged blocks (after first tour)
            if tour > 0 and not block_changed[k]:
                blocks_skipped += 1
                k += max(1, block_size // 2)
                continue
            
            blocks_processed += 1
            
            # Extract block
            block = basis[k:block_end].copy()

            # Adaptive number of passes based on block size
            num_passes = max(1, min(4, 20 // block_len))

            # Use GeometricLLL on the block
            try:
                block_lll = GeometricLLL(N, basis=block)
                
                # Try expand-and-recompress with aggressive pruning
                try:
                    expanded = block_lll._expand_and_recompress_geometric(verbose=False)
                    
                    # OPTIMIZATION: Prune expanded basis aggressively
                    if expanded is not None and len(expanded) > 2 * block_size:
                        norms = [(np.dot(v, v), idx) for idx, v in enumerate(expanded)]
                        norms.sort()
                        expanded = np.array([expanded[idx] for _, idx in norms[:2 * block_size]])
                except Exception:
                    expanded = None

                if expanded is not None and len(expanded) > 0:
                    block_lll.basis = expanded
                    reduced_block = block_lll.run_geometric_reduction(verbose=False, num_passes=num_passes)
                else:
                    reduced_block = block_lll.run_geometric_reduction(verbose=False, num_passes=num_passes)

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
                shortest_vector = _geometric_svp_oracle(block, num_passes=num_passes)
                shortest_norm = _vector_norm_sq(shortest_vector)
            
            # Check if this is shorter than current first vector in block
            current_norm = gso_cache.get_norm(k)
            
            if shortest_norm > 0 and shortest_norm < current_norm:
                # Insert shortest vector at the beginning of block
                for i in range(block_end - 1, k, -1):
                    basis[i] = basis[i-1].copy()
                basis[k] = shortest_vector
                
                # OPTIMIZATION: Local re-reduction only (not full O(n²))
                local_end = min(k + block_size + 5, n)
                for i in range(k + 1, local_end):
                    for j in range(i):
                        basis[i] = _reduce_vector(basis[i], basis[j])
                
                # Update GSO cache for affected range
                gso_cache.update_range(k, local_end)
                
                # Mark affected blocks as changed
                for idx in range(max(0, k - block_size), min(n, k + block_size)):
                    block_changed[idx] = True
                
                tour_improved = True
                
                if verbose:
                    bits = shortest_norm.bit_length() // 2
                    print(f"[*] Block {k}: found shorter vector ~2^{bits} bits")
                
                # Jump back on improvement
                k = max(0, k - block_size // 2)
            else:
                # Mark this block as unchanged
                block_changed[k] = False
                
                if verbose and blocks_processed % 10 == 0:
                    print(f"[*] Block {k}: no improvement")
                
                # Jump forward
                k += max(1, block_size // 2)
        
        if verbose:
            print(f"[*] Processed {blocks_processed} blocks, skipped {blocks_skipped}")
        
        # Track best
        _, current_best = gso_cache.get_shortest_index()
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


def _geometric_svp_oracle(block_basis, num_passes: int = 4):
    """
    Geometric SVP Oracle using GeometricLLL.run_geometric_reduction.
    Falls back to returning the shortest existing vector if anything fails.
    
    OPTIMIZED: Uses adaptive num_passes parameter.
    """
    if block_basis is None or len(block_basis) == 0:
        return np.zeros(block_basis.shape[1] if hasattr(block_basis, 'shape') else 0, dtype=object)

    try:
        block = np.array(block_basis, dtype=object).copy()
        g = GeometricLLL(1, basis=block)
        reduced_block = g.run_geometric_reduction(verbose=False, num_passes=num_passes)

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
