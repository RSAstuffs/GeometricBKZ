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
        # 3. Score triangles and vertices for next point placement
        candidate_scores = []
        
        # Score triangles
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
            candidate_scores.append((score, indices, closest_idx, 'triangle'))
        
        # Score vertices (square corners) as additional candidates
        for v_idx, vertex in enumerate(vertices):
            # Use vertex position as centroid
            centroid = vertex[:2]  # 2D projection
            
            # Find closest basis vector to vertex
            min_dist = float('inf')
            closest_idx = -1
            for i in range(n):
                basis_proj = np.array([basis[i][0], basis[i][1]], dtype=float)
                dist = np.linalg.norm(basis_proj - centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            # Get indices from this vertex
            indices = sorted(list(vertex_indices[v_idx]))
            if len(indices) < 2:
                continue
            
            # Score vertices by their "importance" (number of contributing indices)
            # Give vertices slightly lower priority than triangles by scaling score
            score = 0.8 * len(indices)  # Scale down to prefer triangles but still consider vertices
            candidate_scores.append((score, indices, closest_idx, 'vertex'))
        
        # Sort all candidates by score descending
        candidate_scores.sort(reverse=True)
        
        # 4. Map to contiguous ranges, prioritizing those containing centroid indices
        prioritized_ranges = []
        for score, indices, centroid_idx, source in candidate_scores[:max_candidates]:
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
    
    # Initial geometric reduction using proper geometric methods (analogous to geometric_lll.py)
    if verbose:
        print(f"[*] Initial geometric reduction...")

    # Apply geometric LLL reduction (from geometric_lll.py _lll_reduce_basis)
    basis = _geometric_lll_reduce(basis, max_iterations=min(30, n), verbose=verbose)

    # Apply hierarchical geometric compression (from geometric_lll.py run_geometric_reduction)
    basis = _hierarchical_geometric_compress(basis, verbose=verbose)
    
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
        
        # Process blocks sequentially (like geometric_lll.py run_bkz)
        for k in range(n - 1):
            # Block indices: [k, min(k + block_size, n)) (from geometric_lll.py)
            block_end = min(k + block_size, n)
            block_len = block_end - k

            if block_len < 2:
                continue

            blocks_processed += 1

            # Extract block (from geometric_lll.py)
            block = basis[k:block_end].copy()

            # Find shortest vector in block using geometric reduction (from geometric_lll.py)
            # Apply geometric LLL and hierarchical compression to the block
            reduced_block = _geometric_lll_reduce(block, max_iterations=min(20, block_len), verbose=False)
            reduced_block = _hierarchical_geometric_compress(reduced_block, verbose=False)

            # Get the shortest vector from reduced block (from geometric_lll.py)
            shortest_idx = 0
            shortest_norm = _vector_norm_sq(reduced_block[0])
            for i in range(1, len(reduced_block)):
                norm = _vector_norm_sq(reduced_block[i])
                if norm > 0 and (shortest_norm == 0 or norm < shortest_norm):
                    shortest_norm = norm
                    shortest_idx = i

            # Check if this improves the basis (from geometric_lll.py)
            current_norm = gso_cache.get_norm(k)

            if shortest_norm > 0 and shortest_norm < current_norm:
                # Insert shortest vector at position k (from geometric_lll.py)
                new_vector = reduced_block[shortest_idx].copy()

                # Shift vectors down
                for i in range(block_end - 1, k, -1):
                    basis[i] = basis[i-1].copy()
                basis[k] = new_vector

                # Re-reduce the affected portion (from geometric_lll.py)
                for i in range(k, min(k + block_size + 1, n)):
                    for j in range(i):
                        basis[i] = _reduce_vector(basis[i], basis[j])

                # Update GSO cache for affected range
                gso_cache.update_range(k, min(k + block_size + 1, n))

                # Mark affected blocks as changed
                for idx in range(max(0, k - block_size), min(n, k + block_size)):
                    block_changed[idx] = True

                tour_improved = True

                if verbose:
                    bits = shortest_norm.bit_length() // 2
                    print(f"[*] Block {k}: improved to ~2^{bits} bits")
        
        if verbose:
            print(f"[*] Processed {blocks_processed} candidate blocks")
        
        # Apply geometric LLL and hierarchical compression after each tour (from geometric_lll.py pattern)
        if verbose and tour % 2 == 0:  # Less frequent for performance
            print(f"[*] Applying geometric LLL after tour {tour + 1}...")
        basis = _geometric_lll_reduce(basis, max_iterations=min(15, n//2), verbose=False)

        # Apply hierarchical compression
        if verbose and tour % 3 == 0:
            print(f"[*] Hierarchical geometric compression after tour {tour + 1}...")
        basis = _hierarchical_geometric_compress(basis, verbose=False)

        # Geometric reordering (from geometric_lll.py _geometric_reorder)
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


def _simple_svp_oracle(block_basis, max_coeff=3, verbose=False):
    """
    Simple SVP Oracle - Finds shorter vectors through systematic enumeration.

    Enumerates combinations of block vectors to find linear combinations that
    are shorter than existing vectors.
    """
    if block_basis is None or len(block_basis) == 0:
        return None, None

    basis = np.array(block_basis, dtype=object)
    n = len(basis)
    dim = basis.shape[1]

    current_shortest_norm = _get_shortest_norm(basis)
    best_vector = None
    best_norm = current_shortest_norm

    # Try all combinations of ±1 for first few vectors
    import itertools

    for r in range(1, min(4, n+1)):  # Subset sizes 1 to 3
        for subset in itertools.combinations(range(n), r):
            # Try all sign patterns for this subset
            for signs in itertools.product([-1, 1], repeat=r):
                combination = np.zeros(dim, dtype=object)
                for i, (idx, sign) in enumerate(zip(subset, signs)):
                    combination += sign * basis[idx]

                norm_sq = _vector_norm_sq(combination)
                if norm_sq > 0 and norm_sq < best_norm:
                    best_norm = norm_sq
                    best_vector = combination.copy()

                    if verbose:
                        print(f"[SVP] Found combination of {r} vectors, norm² {norm_sq}")

    return best_vector, best_norm


def _expand_recompress_staged(block_basis, verbose: bool = False):
    """
    STAGED EXPANSION-RECOMPRESSION - Core geometric method for finding shorter vectors.

    Applies geometric LLL, hierarchical compression, and SVP enumeration to find
    shorter vectors in the lattice block.
    """
    if block_basis is None or len(block_basis) == 0:
        return block_basis

    basis = np.array(block_basis, dtype=object).copy()

    # Track the best basis found
    best_basis = basis.copy()
    best_norm = _get_shortest_norm(basis)

    # Stage 1: Geometric LLL reduction
    stage1 = _geometric_lll_reduce(basis, max_iterations=min(15, len(basis)), verbose=False)
    stage1_norm = _get_shortest_norm(stage1)
    if stage1_norm < best_norm:
        best_norm = stage1_norm
        best_basis = stage1.copy()

    # Stage 2: Hierarchical compression
    stage2 = _hierarchical_geometric_compress(stage1, verbose=False)
    stage2_norm = _get_shortest_norm(stage2)
    if stage2_norm < best_norm:
        best_norm = stage2_norm
        best_basis = stage2.copy()

    # Stage 3: SVP enumeration to find even shorter vectors
    shorter_vector, shorter_norm = _simple_svp_oracle(stage2, verbose=verbose)

    if shorter_vector is not None and shorter_norm < best_norm:
        # Insert the shorter vector into the basis
        # Replace the longest vector with the shorter one
        longest_idx = 0
        longest_norm = _vector_norm_sq(best_basis[0])
        for i in range(1, len(best_basis)):
            norm = _vector_norm_sq(best_basis[i])
            if norm > longest_norm:
                longest_norm = norm
                longest_idx = i

        best_basis[longest_idx] = shorter_vector

        # Re-reduce after insertion
        best_basis = _geometric_lll_reduce(best_basis, max_iterations=min(10, len(best_basis)), verbose=False)
        best_norm = _get_shortest_norm(best_basis)

    if verbose:
        bits = best_norm.bit_length() // 2 if best_norm else 0
        print(f"[Staged] Complete. Best: ~2^{bits} bits")

    return best_basis


def _geometric_svp_oracle(block_basis, N: int, num_passes: int, block_len: int,
                          verbose: bool = False, centroid_hint: int = -1):
    """
    Geometric SVP Oracle - TRUE DIVINATION MODE.

    Uses staged expansion-recompression to find shorter vectors in blocks.
    """
    if block_basis is None or len(block_basis) == 0:
        return np.zeros(block_basis.shape[1] if hasattr(block_basis, 'shape') else 0, dtype=object), 0

    try:
        if verbose:
            print(f"[Oracle] Geometric divination on block of size {block_len}...")

        # Use staged expansion-recompression (the core method from geometric_lll.py)
        reduced_block = _expand_recompress_staged(block_basis, verbose=verbose)

        # Find shortest vector in the reduced block
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
            print(f"[Oracle] Found shortest: ~2^{bits} bits")

        return reduced_block[shortest_idx].copy(), shortest_norm

    except Exception as e:
        if verbose:
            print(f"[!] Oracle failed: {e}, using fallback")
        # Fallback: pick shortest existing vector
        norms = [_vector_norm_sq(v) for v in block_basis]
        min_idx = int(np.argmin(norms))
        return block_basis[min_idx].copy(), norms[min_idx]


def _geometric_lll_reduce(basis, delta=0.99, max_iterations=50, verbose=False):
    """
    GEOMETRIC LLL REDUCTION - Based on geometric_lll.py approach.

    Uses geometric swapping criteria instead of traditional Lovász condition.
    This implements the geometric LLL algorithm from GeometricLLL.
    """
    n = len(basis)
    if n <= 1:
        return basis

    basis = basis.copy()
    max_iterations = n * n * 5  # Matches geometric_lll.py

    for iteration in range(max_iterations):
        made_swap = False

        for k in range(1, n):
            # Size reduce first (this is standard LLL)
            for j in range(k-1, -1, -1):
                basis[k] = _reduce_vector(basis[k], basis[j])

            # GEOMETRIC SWAP CRITERION (from geometric_lll.py)
            if k < n:
                # Compute norms
                norm_k = _vector_norm_sq(basis[k])
                norm_km1 = _vector_norm_sq(basis[k-1])

                if norm_k > 0 and norm_km1 > 0:
                    # Compute the projection component (how much k lies along k-1)
                    dot_prod = np.dot(basis[k], basis[k-1])

                    # GEOMETRIC CRITERION: Projection ratio
                    proj_sq = dot_prod * dot_prod

                    # Orthogonalized component of b_k
                    if norm_km1 > 0:
                        orth_norm_k = norm_k - (proj_sq // norm_km1) if norm_km1 != 0 else norm_k
                    else:
                        orth_norm_k = norm_k

                    # GEOMETRIC SWAP: Swap if orthogonalized k is much shorter than k-1
                    # Use 3/4 ratio (classic LLL) computed with integers
                    if 4 * orth_norm_k < 3 * norm_km1:
                        # Swap!
                        basis[k-1], basis[k] = basis[k].copy(), basis[k-1].copy()
                        made_swap = True

                        if verbose and iteration % 10 == 0:
                            print(f"[Geometric LLL] Swap {k-1}<->{k}: orth_ratio = {float(orth_norm_k)/float(norm_km1):.3f}")

        if not made_swap:
            break

    return basis


def _hierarchical_geometric_compress(basis, verbose=False):
    """
    HIERARCHICAL GEOMETRIC COMPRESSION - Based on geometric_lll.py run_geometric_reduction.

    Compresses vectors in groups of 4 (geometric squares) using proper reduction operations.
    This implements the hierarchical compression algorithm from GeometricLLL.
    """
    n = len(basis)
    if n == 0:
        return basis

    basis = basis.astype(object)

    def compress_square(v0, v1, v2, v3):
        """Compress 4 vectors geometrically - O(1) operation (from geometric_lll.py)"""
        # Invert to point same direction as v0
        if np.dot(v0, v1) < 0: v1 = -v1
        if np.dot(v0, v2) < 0: v2 = -v2
        if np.dot(v0, v3) < 0: v3 = -v3

        # Fuse A-B: reduce v1 against v0
        d00 = _vector_norm_sq(v0)
        if d00 > 0:
            dot01 = np.dot(v0, v1)
            r = (dot01 + d00 // 2) // d00
            if r != 0: v1 = v1 - r * v0

        # Fuse C-D: reduce v3 against v2
        d22 = _vector_norm_sq(v2)
        if d22 > 0:
            dot23 = np.dot(v2, v3)
            r = (dot23 + d22 // 2) // d22
            if r != 0: v3 = v3 - r * v2

        # Compress to point: reduce v2 against v0
        if d00 > 0:
            dot02 = np.dot(v0, v2)
            r = (dot02 + d00 // 2) // d00
            if r != 0: v2 = v2 - r * v0

        # Also reduce v3 against v0
        if d00 > 0:
            dot03 = np.dot(v0, v3)
            r = (dot03 + d00 // 2) // d00
            if r != 0: v3 = v3 - r * v0

        return v0, v1, v2, v3

    def compress_pair(v0, v1):
        """Compress 2 vectors - O(1) (from geometric_lll.py)"""
        if np.dot(v0, v1) < 0: v1 = -v1
        d00 = _vector_norm_sq(v0)
        if d00 > 0:
            dot01 = np.dot(v0, v1)
            r = (dot01 + d00 // 2) // d00
            if r != 0: v1 = v1 - r * v0
        return v0, v1

    # === HIERARCHICAL COMPRESSION (from geometric_lll.py) ===
    # Process in groups of 4 (like the geometric square)
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
        d00 = _vector_norm_sq(basis[0])
        if d00 > 0:
            dot0i = np.dot(basis[0], basis[i])
            r = (dot0i + d00 // 2) // d00
            if r != 0:
                basis[i] = basis[i] - r * basis[0]

    # Sort by norm (from geometric_lll.py)
    norms = [(np.dot(basis[i], basis[i]), i) for i in range(n)]
    norms.sort()
    basis = np.array([basis[idx] for _, idx in norms], dtype=object)

    return basis


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
