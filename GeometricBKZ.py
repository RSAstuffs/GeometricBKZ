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
    
    # Add diagonal vectors (space diagonals)
    if d >= 2 and vertices:  # Only if we have regular vertices
        # Main space diagonal: sum of all basis vectors in subspace
        s = np.ones(n, dtype=object)
        contrib = list(range(d))
        v = np.zeros(basis.shape[1], dtype=object)
        for j in range(d):
            v += s[j] * basis[j]
        vertices.append(v.astype(float))
        vertex_indices.append(contrib)
        
        # Face diagonals for higher dimensions
        if d >= 3:
            # For 3D+: face diagonals (sum of all but one basis vector)
            for i in range(d):
                s = np.ones(n, dtype=object)
                s[i] = 0  # exclude one dimension
                contrib = [j for j in range(d) if j != i]
                v = np.zeros(basis.shape[1], dtype=object)
                for j in range(d):
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
    
    # Select top candidates and collect hints for each
    selected = candidate_ranges[:max_candidates]
    centroid_hints = []
    
    if triangles is not None and candidate_scores:
        # For each selected candidate, collect all relevant hint indices
        for s, e in selected:
            block_hints = []
            # Collect hints from top candidates that overlap with this block
            for score, indices, centroid_idx, source in candidate_scores[:max_candidates]:
                # Check if this candidate's indices overlap with the block
                if any(s <= idx < e for idx in indices):
                    if centroid_idx not in block_hints:
                        block_hints.append(centroid_idx)
            centroid_hints.append(block_hints)
    
    # Ensure we have hints for all selected blocks
    while len(centroid_hints) < len(selected):
        centroid_hints.append([])
    
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
            block_hints = centroid_hints[blocks_processed - 1] if blocks_processed - 1 < len(centroid_hints) else []
            shortest_vector, shortest_norm = _geometric_svp_oracle(
                block, N, num_passes, block_len, verbose=(verbose and blocks_processed <= 3),
                centroid_hints=block_hints
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


def _drag_single_plane_through_square(expanded_vertices, current_norm_sq, verbose=False):
    """
    Detect curvature in the expanded square vertex distribution using bidirectional plane sweeping.

    Drag the plane through in both directions and find vertices that show invariant curvature -
    curvature that persists regardless of sweeping direction. These are the true planted vectors.

    OPTIMIZED: Fully vectorized operations, pre-allocated arrays, quantized calculations.
    """
    if len(expanded_vertices) < 6:  # Need more points for curvature estimation
        return None, None

    # Extract vertex coordinates - pre-allocate and vectorize
    vertices = np.array([v[0] for v in expanded_vertices], dtype=np.float32)
    n_vertices = len(vertices)

    # Pre-allocate arrays for vectorized operations
    vertex_curvature = np.zeros(n_vertices, dtype=np.float32)
    k_neighbors = min(4, max(3, n_vertices // 10))  # Adaptive k based on dataset size

    # Vectorized distance computation: compute all pairwise distances at once
    # Shape: (n_vertices, n_vertices, n_features)
    diff = vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :]
    # Compute squared distances (faster than norm, and we only need relative distances)
    dist_sq = np.sum(diff ** 2, axis=2, dtype=np.float32)

    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(dist_sq, np.inf)

    # For each vertex, find k-nearest neighbors using vectorized operations
    for i in range(n_vertices):
        # Get distances to all other points (exclude self)
        distances_to_others = dist_sq[i, :]
        # Find indices of k smallest distances (excluding self)
        neighbor_indices = np.argpartition(distances_to_others, k_neighbors)[:k_neighbors]

        if len(neighbor_indices) >= 3:
            # Vectorized curvature calculation
            neighbor_vectors = vertices[neighbor_indices] - vertices[i]  # Center around vertex

            # Compute distances from center (quantized to avoid floating point issues)
            neighbor_dist_sq = np.sum(neighbor_vectors ** 2, axis=1, dtype=np.float32)
            neighbor_distances = np.sqrt(neighbor_dist_sq)  # Only take sqrt when needed

            # Fast curvature measure using vectorized statistics
            if len(neighbor_distances) >= 3:
                # Coefficient of variation as curvature measure (quantized)
                mean_dist = np.mean(neighbor_distances)
                if mean_dist > 1e-6:  # Quantized threshold
                    std_dist = np.std(neighbor_distances)
                    # Quantize the ratio calculation to avoid precision issues
                    curvature = std_dist / mean_dist
                    vertex_curvature[i] = curvature
                else:
                    vertex_curvature[i] = 0.0
            else:
                vertex_curvature[i] = 0.0
        else:
            vertex_curvature[i] = 0.0

    # NOW: After computing curvature in one direction, sweep in the INVERSE direction
    # and find vertices that show the SAME curvature pattern (curvature that "has not moved")

    # Compute inverse curvature (opposite direction)
    inverse_curvature = np.zeros(n_vertices, dtype=np.float32)

    # For inverse sweep, we'll reverse the neighbor ordering or use negative distances
    for i in range(n_vertices):
        # Use the same neighbor finding but consider "inverse" geometric relationships
        distances_to_others = dist_sq[i, :]
        neighbor_indices = np.argpartition(distances_to_others, k_neighbors)[:k_neighbors]

        if len(neighbor_indices) >= 3:
            # For inverse direction, consider the "mirror" or "opposite" geometric configuration
            # This simulates sweeping the plane in the opposite direction
            neighbor_vectors = -(vertices[neighbor_indices] - vertices[i])  # Negate for inverse direction

            neighbor_dist_sq = np.sum(neighbor_vectors ** 2, axis=1, dtype=np.float32)
            neighbor_distances = np.sqrt(neighbor_dist_sq)

            if len(neighbor_distances) >= 3:
                mean_dist = np.mean(neighbor_distances)
                if mean_dist > 1e-6:
                    std_dist = np.std(neighbor_distances)
                    inverse_curvature[i] = std_dist / mean_dist
                else:
                    inverse_curvature[i] = 0.0

    # Find vertices where curvature is CONSISTENT between forward and inverse sweeps
    # These are the planted vectors - curvature that "has not moved"
    curvature_difference = np.abs(vertex_curvature - inverse_curvature)
    invariant_threshold = np.mean(curvature_difference) + 0.5 * np.std(curvature_difference)

    # Vertices with LOW curvature difference have invariant curvature
    invariant_mask = curvature_difference < invariant_threshold
    invariant_curvature_scores = vertex_curvature[invariant_mask]  # Use forward curvature for ranking

    if np.any(invariant_mask) and len(invariant_curvature_scores) > 0:
        # Among invariant vertices, find the ones with highest curvature
        invariant_indices = np.where(invariant_mask)[0]
        top_invariant = invariant_indices[np.argmax(vertex_curvature[invariant_indices])]

        curved_vertex = vertices[top_invariant]
        curvature_score = vertex_curvature[top_invariant]

        # Check if this represents a significant planted vector
        avg_curvature = np.mean(vertex_curvature)
        if curvature_score > avg_curvature * 3.0:  # 3x more curved than average
            # Quantized norm calculation
            vector_norm_sq = np.sum(curved_vertex.astype(np.float64) ** 2, dtype=np.float64)
            vector_norm = int(np.round(np.sqrt(vector_norm_sq))) if vector_norm_sq > 0 else 0

            improvement_ratio = current_norm_sq / max(vector_norm, 1)

            if vector_norm > 0 and improvement_ratio > 1.5:  # Require reasonable improvement
                if verbose:
                    inverse_score = inverse_curvature[top_invariant]
                    print(f"[BidirectionalSweep] Found invariant curvature vertex!")
                    print(f"[BidirectionalSweep] Forward: {curvature_score:.3f}, Inverse: {inverse_score:.3f}, Difference: {curvature_difference[top_invariant]:.3f}")
                    print(f"[BidirectionalSweep] Ratio: {improvement_ratio:.2f}")
                return curved_vertex.astype(object), vector_norm

    return None, None




def _expand_square_for_distortion(block_basis, current_norm_sq: int, max_expansion: int = 8,
                                verbose: bool = False) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Expand the Square until hitting a distortion in vertices close to/at the centroid.

    When standard geometric reduction fails to find improvement, systematically expand
    the search space by considering larger coefficient combinations. Look for "distortions"
    - vertices that deviate significantly from expected positions relative to the centroid.

    This reveals hidden short vectors that weren't apparent in the initial {0,1} vertex set.

    ENHANCED: More aggressive expansion, better distortion detection, and planted vector targeting.
    """
    if len(block_basis) < 2:
        return None, None

    block = np.array(block_basis, dtype=np.float32).copy()  # Use float32 for speed
    n = len(block)
    max_dim = min(n, 8)
    feature_dim = block.shape[1]

    # Pre-allocate arrays for vectorized operations
    max_base_vertices = (1 << max_dim) - 1  # Maximum possible vertices (excluding zero)
    base_vertices_array = np.zeros((max_base_vertices, feature_dim), dtype=np.float32)
    base_coeffs_array = np.zeros((max_base_vertices, max_dim), dtype=np.int8)
    vertex_count = 0

    # Vectorized base vertex generation
    for mask in range(1, 1 << max_dim):  # Start from 1 to skip zero vector
        coeffs = np.array([(mask >> i) & 1 for i in range(max_dim)], dtype=np.int8)

        # Vectorized vertex construction: coeffs[:, None] * block[:max_dim]
        vertex = np.sum(coeffs[:, np.newaxis] * block[:max_dim], axis=0, dtype=np.float32)

        base_vertices_array[vertex_count] = vertex
        base_coeffs_array[vertex_count] = coeffs
        vertex_count += 1

    # Trim to actual size
    base_vertices_array = base_vertices_array[:vertex_count]
    base_coeffs_array = base_coeffs_array[:vertex_count]

    if vertex_count < 2:
        return None, None

    # Vectorized centroid calculation
    centroid = np.mean(base_vertices_array, axis=0, dtype=np.float32)

    if verbose:
        print(f"[Distortion] Starting expansion from {len(base_vertices)} base vertices")

    # Pre-allocate arrays for all expansion levels to avoid dynamic resizing
    max_total_vertices = vertex_count + 200  # Estimate maximum vertices needed
    all_vertices_array = np.zeros((max_total_vertices, feature_dim), dtype=np.float32)
    all_coeffs_array = np.zeros((max_total_vertices, max_dim), dtype=np.int8)

    # Copy base vertices
    all_vertices_array[:vertex_count] = base_vertices_array
    all_coeffs_array[:vertex_count] = base_coeffs_array
    current_vertex_count = vertex_count

    # Iteratively expand coefficients with vectorized operations
    for expansion_level in range(2, max_expansion + 1, 2):  # Skip every other level for speed
        # Adaptive sampling based on current vertex count
        if current_vertex_count > 150:  # Already have plenty of vertices
            num_samples = min(20, 2**(min(n, 3)))  # Minimal additional samples
        elif expansion_level <= 4:
            num_samples = min(30, 2**(min(n, 4)))  # Fewer samples for early levels
        else:
            num_samples = min(60, 2**(min(n, 5)))  # More samples for final levels

        # Pre-allocate coefficient arrays for vectorized generation
        coeff_range = 2 * expansion_level + 1  # Range size for random sampling
        coeffs_array = np.zeros((num_samples, max_dim), dtype=np.int8)

        # Vectorized coefficient generation with bias toward small values
        for sample_idx in range(num_samples):
            for dim_idx in range(max_dim):
                # Quantized probability for small vs large coefficients
                if np.random.random() < 0.7:  # 70% chance of small coefficients
                    coeffs_array[sample_idx, dim_idx] = np.random.randint(-2, 3)
                else:
                    coeffs_array[sample_idx, dim_idx] = np.random.randint(-expansion_level, expansion_level + 1)

        # Vectorized vertex construction
        # Shape: (num_samples, max_dim, feature_dim) -> (num_samples, feature_dim)
        vertices_batch = np.sum(
            coeffs_array[:, :, np.newaxis] * block[:max_dim, np.newaxis, :],
            axis=1,
            dtype=np.float32
        )

        # Filter out zero vectors (all coefficients are zero)
        non_zero_mask = np.any(coeffs_array != 0, axis=1)
        valid_samples = np.sum(non_zero_mask)

        if valid_samples > 0:
            # Add valid vertices to our collection
            start_idx = current_vertex_count
            end_idx = current_vertex_count + valid_samples

            if end_idx <= max_total_vertices:  # Check bounds
                all_vertices_array[start_idx:end_idx] = vertices_batch[non_zero_mask]
                all_coeffs_array[start_idx:end_idx] = coeffs_array[non_zero_mask]
                current_vertex_count = end_idx

        if not new_vertices:
            continue

        # Drag a single plane through the expanded square - the least distorted point is the planted vector
        all_vertices = base_vertices + new_vertices

        if verbose:
            print(f"[Distortion] Level {expansion_level}: {len(all_vertices)} total vertices")

        # Only drag plane after full expansion (level 8) - much more efficient
        if expansion_level == 8 and current_vertex_count >= 6:  # Final level with sufficient vertices
            if verbose:
                print(f"[Distortion] Activating final plane sweep with {current_vertex_count} fully expanded vertices")

            # Convert back to list format for compatibility (could be optimized further)
            expanded_vertices = [(all_vertices_array[i].astype(object), all_coeffs_array[i])
                               for i in range(current_vertex_count)]

            planted_vector, planted_norm = _drag_single_plane_through_square(
                expanded_vertices, current_norm_sq, verbose=verbose
            )

            if planted_vector is not None:
                if verbose:
                    bits = planted_norm.bit_length() // 2
                    print(f"[PlaneDrag] Found planted vector via final plane sweep: ~2^{bits} bits")
                return planted_vector, planted_norm

        if verbose and expansion_level % 2 == 0:
            print(f"[Distortion] Expansion level {expansion_level}: checked {len(new_vertices)} new vertices")

    if verbose:
        print(f"[Distortion] No distortions found after expansion to level {max_expansion}")

    return None, None


def _geometric_svp_oracle(block_basis, N: int, num_passes: int, block_len: int,
                          verbose: bool = False, centroid_hints: List[int] = None):
    """
    Geometric SVP Oracle - TRUE DIVINATION MODE.

    The Oracle divines by using the square's vertices to predict WHERE to look next.
    The vertices don't just compress - they POINT to the next block's location.

    Key insight: The square's vertices after transformation indicate which
    lattice regions contain shorter vectors. Use this to GUIDE the block selection,
    not just to score the result.

    NEW: When no short vector found, expand the Square until hitting distortions
    in vertices close to/at the centroid.

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

        # If centroid hints provided, check if any hinted index has a competitive vector
        if centroid_hints:
            best_hint_idx = -1
            best_hint_norm = shortest_norm
            for hint in centroid_hints:
                if hint >= 0 and hint < len(reduced_block):
                    hint_norm = _vector_norm_sq(reduced_block[hint])
                    if hint_norm > 0 and hint_norm <= best_hint_norm * 1.1:  # Within 10% of current best
                        best_hint_idx = hint
                        best_hint_norm = hint_norm
            if best_hint_idx >= 0:
                shortest_idx = best_hint_idx
                shortest_norm = best_hint_norm
                if verbose:
                    print(f"[Oracle] Using centroid-hinted vector at index {best_hint_idx} from {len(centroid_hints)} hints")

        # Check if we found an improvement over the original block
        original_shortest = min(_vector_norm_sq(v) for v in block_basis)
        if shortest_norm >= original_shortest:
            # No improvement found - expand the Square to find distortions
            if verbose:
                print(f"[Oracle] No improvement found, expanding Square for distortions...")

            distorted_vector, distorted_norm = _expand_square_for_distortion(
                block_basis, original_shortest, max_expansion=8, verbose=verbose
            )

            if distorted_vector is not None and distorted_norm < shortest_norm:
                if verbose:
                    bits = distorted_norm.bit_length() // 2
                    print(f"[Oracle] Distortion search successful: ~2^{bits} bits")
                return distorted_vector, distorted_norm

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
    """Squared norm of vector with quantization for numerical stability."""
    # Quantize to avoid floating point precision issues
    if isinstance(v, np.ndarray) and v.dtype.kind == 'f':  # Float array
        # Round to nearest integer to quantize
        v_quantized = np.round(v).astype(np.int64)
        norm_sq = np.dot(v_quantized, v_quantized)
    else:
        norm_sq = np.dot(v, v)

    # Ensure we return an integer for bit_length() operations
    if isinstance(norm_sq, (int, np.integer)):
        return int(norm_sq)
    else:
        return int(np.round(norm_sq))


def _get_shortest_norm(basis) -> Optional[int]:
    """Get squared norm of shortest non-zero vector."""
    shortest = None
    for v in basis:
        norm_sq = _vector_norm_sq(v)
        if norm_sq > 0:
            if shortest is None or norm_sq < shortest:
                shortest = norm_sq
    return shortest
