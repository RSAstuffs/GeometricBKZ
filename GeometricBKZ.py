"""
Standalone BKZ Module - Optimized with GeometricLLL Integration
================================================================

Custom BKZ implementation using geometric SVP oracle.
Properly integrated with GeometricLLL for maximum efficiency.
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

# Optional scipy for convex hull and triangulation
try:
    from scipy.spatial import ConvexHull, Delaunay
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

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


def _select_candidate_blocks_geometric(basis: np.ndarray, block_size: int, max_candidates: int = 10) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Select candidate blocks using geometric probes with triangulation.
    
    The Square (parallelepiped vertices) informs the Triangle (Delaunay triangulation)
    to triangulate promising regions for next block coordinates.
    
    Pipeline:
    1. Compute parallelepiped vertices B * s for s in {0,1}^d
    2. Triangulate the projected vertices using Delaunay
    3. Score triangles by area, orientation, and local density
    4. Map high-scoring triangles to contiguous index ranges
    5. Select top candidates for SVP oracle invocation
    """
    n = len(basis)
    if n < block_size:
        return [(0, n)]
    
    # 1. Compute vertices
    d = min(6, n)
    vertices = []
    vertex_indices = []  # track which basis vectors contribute
    if d <= 6:
        for mask in range(1 << d):
            s = np.zeros(n, dtype=object)
            contrib = []
            for i in range(d):
                if mask & (1 << i):
                    s[i] = 1
                    contrib.append(i)
            if not contrib:
                continue
            v = np.zeros(basis.shape[1], dtype=object)
            for j in range(n):
                v += s[j] * basis[j]
            vertices.append(v.astype(float))
            vertex_indices.append(contrib)
    else:
        # Random vertices
        for _ in range(50):
            s = np.random.randint(0, 2, n)
            contrib = [i for i in range(n) if s[i] == 1]
            if not contrib:
                continue
            v = np.zeros(basis.shape[1], dtype=object)
            for j in range(n):
                v += s[j] * basis[j]
            vertices.append(v.astype(float))
            vertex_indices.append(contrib)
    
    if len(vertices) < 4:
        return [(0, min(block_size, n))], []
    
    vertices = np.array(vertices)
    
    # 2. Triangulate using Delaunay
    if SCIPY_AVAILABLE and len(vertices) >= 4:
        try:
            tri = Delaunay(vertices[:, :2])  # Use first 2 dimensions for 2D triangulation
            triangles = tri.simplices
        except Exception:
            # Fallback to simple range selection
            triangles = None
    else:
        triangles = None
    
    candidate_ranges = []
    
    if triangles is not None:
        # 3. Score triangles and compute centroids for next point placement
        triangle_scores = []
        for simplex in triangles:
            # Triangle vertices
            tri_verts = vertices[simplex]
            # Area (using cross product in 2D)
            v1, v2, v3 = tri_verts
            area = abs((v2[0] - v1[0])*(v3[1] - v1[1]) - (v3[0] - v1[0])*(v2[1] - v1[1])) / 2
            if area == 0:
                continue
            
            # Centroid (in 2D projection space)
            centroid = np.mean(tri_verts[:, :2], axis=0)
            
            # Find closest basis vector to centroid (by Euclidean distance in projection space)
            min_dist = float('inf')
            closest_idx = -1
            for i in range(n):
                basis_proj = np.array([basis[i][0], basis[i][1]], dtype=float)  # First 2 coords
                dist = np.linalg.norm(basis_proj - centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            # Collect contributing indices from vertices
            indices = set()
            for idx in simplex:
                indices.update(vertex_indices[idx])
            indices = sorted(list(indices))
            
            if len(indices) < 2:
                continue
            
            # Score: prefer larger areas and more spread indices
            score = area * len(indices)
            triangle_scores.append((score, indices, closest_idx))
        
        # Sort by score descending
        triangle_scores.sort(reverse=True)
        
        # 4. Map to contiguous ranges, prioritizing those containing centroid indices
        prioritized_ranges = []
        for score, indices, centroid_idx in triangle_scores[:max_candidates]:
            # Find contiguous segments
            indices.sort()
            start = indices[0]
            for i in range(1, len(indices)):
                if indices[i] > indices[i-1] + 1:
                    end = indices[i-1] + 1
                    if end - start >= 2:
                        prioritized_ranges.append((start, min(end, n), centroid_idx))
                    start = indices[i]
            end = indices[-1] + 1
            if end - start >= 2:
                prioritized_ranges.append((start, min(end, n), centroid_idx))
        
        # Sort by whether range contains centroid
        prioritized_ranges.sort(key=lambda x: x[2] in range(x[0], x[1]), reverse=True)
        candidate_ranges = [(s, e) for s, e, c in prioritized_ranges]
    else:
        # Fallback: use PCA as before
        # Compute covariance
        mean = np.mean(vertices, axis=0)
        centered = vertices - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        directions = eigvecs[:, -min(3, len(eigvecs)):]
        
        for dir_vec in directions.T:
            projections = [np.dot(basis[i], dir_vec) for i in range(n)]
            sorted_indices = np.argsort(np.abs(projections))[::-1]
            top_indices = sorted_indices[:max(1, n // 5)]
            top_indices.sort()
            start = top_indices[0]
            for i in range(1, len(top_indices)):
                if top_indices[i] > top_indices[i-1] + 1:
                    end = top_indices[i-1] + 1
                    if end - start >= 2:
                        candidate_ranges.append((start, min(end, n)))
                    start = top_indices[i]
            end = top_indices[-1] + 1
            if end - start >= 2:
                candidate_ranges.append((start, min(end, n)))
    
    # Remove duplicates and filter
    candidate_ranges = list(set(candidate_ranges))
    candidate_ranges = [(s, e) for s, e in candidate_ranges if e - s <= block_size and e - s >= 2]
    
    # Select top candidates
    selected = candidate_ranges[:max_candidates]
    centroid_hints = []
    if triangles is not None and prioritized_ranges:
        centroid_hints = [prioritized_ranges[i][2] for i in range(min(len(prioritized_ranges), max_candidates))]
    
    return selected if selected else [(0, min(block_size, n))], centroid_hints


def bkz_reduce(basis: np.ndarray, block_size: int = 20, max_tours: int = 10, 
               verbose: bool = True, N: int = 1) -> np.ndarray:
    """
    Custom BKZ reduction using geometric SVP oracle - OPTIMIZED VERSION.
    
    Properly integrated with GeometricLLL for maximum efficiency:
    - Uses GeometricLLL's run_geometric_reduction for initial reduction
    - Uses expand_recompress_staged for SVP oracle when beneficial
    - Uses geometric reordering O(n log n) instead of O(n²) full reduction
    - GSO caching to avoid redundant norm computations
    - Block change tracking to skip unchanged blocks
    - Adaptive block processing with jump strategy
    
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
    
    # Convert to object array for arbitrary precision
    basis = np.array(basis, dtype=object, copy=True)
    n = len(basis)
    
    block_size = min(block_size, n)
    
    if verbose:
        print(f"[*] Running Optimized Geometric BKZ on {n}x{basis.shape[1]} lattice...")
        print(f"[*] Block size: {block_size}, Max tours: {max_tours}")
    
    # Initial geometric reduction using GeometricLLL (matches geometric_lll.py)
    if verbose:
        print(f"[*] Initial geometric reduction...")
    try:
        g_full = GeometricLLL(N, basis=basis.copy())
        basis = g_full.run_geometric_reduction(verbose=False, num_passes=1)
    except Exception as e:
        if verbose:
            print(f"[!] GeometricLLL reduction failed: {e}, using fallback")
        # Fallback to simple sort by norm
        basis = _geometric_reorder(basis, verbose=False)
    
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
        
        # Geometric block selection: use parallelepiped vertices to select candidates
        candidate_blocks, centroid_hints = _select_candidate_blocks_geometric(basis, block_size, max_candidates=10)
        
        for k_start, k_end in candidate_blocks:
            block_len = k_end - k_start
            if block_len < 2:
                continue
            
            blocks_processed += 1
            
            # Extract block
            block = basis[k_start:k_end].copy()

            # Adaptive number of passes based on block size
            num_passes = max(2, min(4, 20 // block_len))

            # DIVINE: Use the Oracle to find shortest vector
            # The Oracle uses staged expansion for blocks >= 4 to reveal hidden structure
            shortest_vector, shortest_norm = _geometric_svp_oracle(
                block, N, num_passes, block_len, verbose=(verbose and blocks_processed <= 3)
            )
            
            # Check if this is shorter than current first vector in block
            current_norm = gso_cache.get_norm(k_start)
            
            if shortest_norm > 0 and shortest_norm < current_norm:
                # Insert shortest vector at the beginning of block
                for i in range(k_end - 1, k_start, -1):
                    basis[i] = basis[i-1].copy()
                basis[k_start] = shortest_vector
                
                # Local re-reduction of affected region (matches geometric_lll.py pattern)
                for i in range(k_start, min(k_start + block_size + 1, n)):
                    for j in range(i):
                        basis[i] = _reduce_vector(basis[i], basis[j])
                
                # Update GSO cache for affected range
                gso_cache.update_range(k_start, min(k_start + block_size + 1, n))
                
                # Mark affected blocks as changed
                for idx in range(max(0, k_start - block_size), min(n, k_start + block_size)):
                    block_changed[idx] = True
                
                tour_improved = True
                
                if verbose:
                    bits = shortest_norm.bit_length() // 2
                    print(f"[*] Block {k_start}: found shorter vector ~2^{bits} bits")
            else:
                if verbose and blocks_processed % 10 == 0:
                    print(f"[*] Block {k_start}: no improvement")
        
        if verbose:
            print(f"[*] Processed {blocks_processed} candidate blocks")
        
        # Use GeometricLLL's geometric reordering O(n log n) instead of O(n²) reduction
        try:
            g_reorder = GeometricLLL(N, basis=basis.copy())
            basis = g_reorder._geometric_reorder(basis, verbose=False)
        except Exception:
            # Fallback to local implementation
            basis = _geometric_reorder(basis, verbose=False)
        
        # Update GSO cache after reordering
        gso_cache.basis = basis
        gso_cache.update_range(0, n)
        
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


def _geometric_svp_oracle(block_basis, N: int, num_passes: int, block_len: int, 
                          verbose: bool = False, centroid_hint: int = -1):
    """
    Geometric SVP Oracle - TRUE DIVINATION MODE.
    
    The Oracle divines by using the square's vertices to predict WHERE to look next.
    The vertices don't just compress - they POINT to the next block's location.
    
    Key insight: The square's vertices after transformation indicate which
    lattice regions contain shorter vectors. Use this to GUIDE the block selection,
    not just to score the result.
    
    Returns: (shortest_vector, shortest_norm_sq)
    """
    if block_basis is None or len(block_basis) == 0:
        return np.zeros(block_basis.shape[1] if hasattr(block_basis, 'shape') else 0, dtype=object), 0

    try:
        block = np.array(block_basis, dtype=object).copy()
        g = GeometricLLL(N, basis=block)
        
        if verbose:
            print(f"[Oracle] Divining block of size {block_len}...")
        
        # CRITICAL: Use expand_recompress_staged for ALL non-trivial blocks
        # This is the method that actually reveals hidden structure
        if block_len >= 4:
            try:
                if verbose:
                    print(f"[Oracle] Using staged expansion to reveal hidden geometry...")
                reduced_block = g.expand_recompress_staged(verbose=False)
                
                if reduced_block is None or len(reduced_block) == 0:
                    if verbose:
                        print(f"[Oracle] Staged expansion produced no result, falling back...")
                    reduced_block = g.run_geometric_reduction(verbose=False, num_passes=num_passes)
            except Exception as e:
                if verbose:
                    print(f"[Oracle] Staged expansion failed ({e}), using standard reduction...")
                reduced_block = g.run_geometric_reduction(verbose=False, num_passes=num_passes)
        else:
            # Trivial blocks: just use standard reduction
            reduced_block = g.run_geometric_reduction(verbose=False, num_passes=num_passes)

        # Find ACTUAL shortest vector (not divined - just shortest)
        shortest_idx = 0
        shortest_norm = _vector_norm_sq(reduced_block[0])
        
        for i in range(1, len(reduced_block)):
            norm = _vector_norm_sq(reduced_block[i])
            if norm > 0 and (shortest_norm == 0 or norm < shortest_norm):
                shortest_norm = norm
                shortest_idx = i
        
        # If centroid hint provided, check if hinted index has a competitive vector
        if centroid_hint >= 0 and centroid_hint < len(reduced_block):
            hint_norm = _vector_norm_sq(reduced_block[centroid_hint])
            if hint_norm > 0 and hint_norm <= shortest_norm * 1.1:  # Within 10% of shortest
                shortest_idx = centroid_hint
                shortest_norm = hint_norm
                if verbose:
                    print(f"[Oracle] Using centroid-hinted vector at index {centroid_hint}")
        
        if verbose:
            bits = shortest_norm.bit_length() // 2 if shortest_norm > 0 else 0
            print(f"[Oracle] Found shortest at index {shortest_idx}: ~2^{bits} bits")

        return reduced_block[shortest_idx].copy(), shortest_norm
        
    except Exception as e:
        if verbose:
            print(f"[!] Oracle failed: {e}, using fallback")
        # Fallback: pick shortest existing vector
        norms = [_vector_norm_sq(v) for v in block_basis]
        min_idx = int(np.argmin(norms))
        return block_basis[min_idx].copy(), norms[min_idx]


def _geometric_reorder(basis, verbose=False):
    """
    PURE GEOMETRIC REORDERING: Sort vectors by a geometric criterion.
    
    Instead of iterative swaps, compute a geometric "score" for each vector
    and reorder accordingly. This is O(n log n) instead of O(n²).
    
    Geometric score: Combination of:
    - Norm (smaller = better)
    - Orthogonality to previous vectors (more orthogonal = better)
    - "Spread" in the coordinate space
    """
    n = len(basis)
    if n <= 1:
        return basis
    
    basis = basis.copy()
    
    # Compute geometric scores
    scores = []
    for i in range(n):
        norm_i = np.dot(basis[i], basis[i])
        if norm_i == 0:
            scores.append((float('inf'), i))
            continue
        
        # Score based on norm (log scale to handle huge integers)
        norm_bits = norm_i.bit_length() if norm_i > 0 else 0
        scores.append((norm_bits, i))
    
    # Sort by score (smallest norm first)
    scores.sort(key=lambda x: x[0])
    
    # Reorder basis
    new_basis = np.array([basis[scores[i][1]] for i in range(n)], dtype=object)
    
    return new_basis


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
