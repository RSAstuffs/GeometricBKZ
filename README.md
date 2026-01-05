# BKZ (custom geometric) — differences vs Standard BKZ

This document explains concrete implementation and algorithmic differences between the custom Geometric/BKZ code in this repository (files like `geometric_lll.py` and `bkz.py`) and a standard BKZ implementation as found in libraries like fpylll.

It is written to help reviewers, collaborators, and anyone trying to interpret benchmark results in this project.

## High-level summary

- Standard BKZ: repeatedly processes blocks of lattice vectors after orthogonal projection (Gram–Schmidt), and calls a (typically exact or pruned) SVP oracle inside each block (e.g. enumeration, fplll's BKZ). It uses exact/float Gram–Schmidt to project and reduce blocks; block SVP is typically solved using optimized C/C++ code.

- Custom Geometric BKZ (this repo): uses a heuristic geometric SVP oracle derived from the `GeometricLLL` class. The oracle performs hierarchical/combinatorial "compressions" (square/pair compress, expand→recompress heuristics) that aim to expose short linear combinations in structured lattices. The BKZ driver uses these geometric reductions instead of classical enumeration-based SVP.

The practical effect: the custom code can sometimes find shorter vectors on specially-constructed or planted lattices (Coppersmith-style) but will not always match classical BKZ on arbitrary random lattices. It is also currently slower (Python-level code) compared with optimized C/C++ implementations.

## Detailed differences

### 1) Block projection (Gram–Schmidt)

- Standard BKZ: before solving SVP on block [k, k + b), the block is orthogonally projected onto the complement of the first k vectors using Gram–Schmidt (floating-point GS or exact rational GS). This isolates the block's sublattice and makes block-SVP mathematically equivalent to the intended BKZ operation.

- Custom BKZ: for simplicity and to match the geometric code's internal expectations, the driver often passes the raw block (without full orthogonal projection) to the oracle. This is a significant algorithmic departure and can alter which vectors are considered "shortest" in the projected sense.

Implication: omitting projection can both speed up code paths (less work) and make results not strictly comparable to standard BKZ. For fair benchmarking, implement block projection before calling the oracle.

### 2) The SVP oracle

- Standard BKZ: uses an exact or pruned enumeration/LP-based SVP solver (often in compiled C/C++). This solver exhaustively searches combinations up to pruning heuristics and returns a true shortest vector in the projected block (within the limits of the algorithm).

- Custom Geometric BKZ: the oracle is heuristic and geometric:
  - Hierarchical compression: group vectors into 4s (A,B,C,D), invert vertex directions to align, do A-B and C-D fusions, compress to a point, then reduce across group leaders.
  - Expand→recompress: intentionally expand a compressed state into a line/triangle/square configuration, then run rotating compress passes and keep the best result. This perturbation can expose short combinations in structured lattices (e.g., Coppersmith-type lattices) but is not guaranteed to find exact block-SVP.

Implication: geometric oracle is powerful on certain structure; it is heuristic and may miss exact block-SVP solutions that enumeration would find (or may find different short vectors because of lack of projection).

### 3) Arithmetic and dtype handling

- Standard BKZ libraries typically use compiled types with careful handling of big integers and floating-point GS (or exact rationals) to avoid overflow and to exploit speed.

- Custom BKZ: originally used object-dtype numpy arrays (Python big ints). The repository now includes a safe int64 fast-path and attempt to use numeric `np.int64` arrays when entries fit in 64-bit, with fallbacks to Python ints for larger values. Inner reductions now have an optimized int64 reducer and (optionally) a JIT via numba.

Implication: int64 fast-path accelerates many operations but risks overflow for large lattices. The code intentionally checks bounds and falls back to object dtype when values are too big.

### 4) Language and performance

- Standard BKZ (fpylll etc.): core loops and enumeration implemented in C/C++ and tuned; performance is orders of magnitude better for large or many-block instances.

- Custom BKZ: implementation is in Python + NumPy, with some inner hot paths optionally JIT-able with numba. As a result, custom BKZ is slower but easier to modify and instrument.

### 5) Heuristics and patterns

- Custom BKZ's geometric methods are designed to exploit geometric and combinatorial structure (e.g., compressions that emulate collapsing squares into points). This can be advantageous for Coppersmith-like lattices where small-root structure exists.

- Standard BKZ is general-purpose and tuned to perform well across random and structured lattices; its SVP oracle is more principled (enumeration/pruning) while being slower per block.

## When the custom BKZ tends to win

- Planted or structured lattices where expand→recompress reveals combinations that enumeration with small block sizes misses.
- Problems specifically designed with geometric collapse / compression in mind (Coppersmith small-root lattices, some factoring instances).

## When the standard BKZ tends to win

- Random lattices or worst-case lattices where an exact/pruned enumeration is needed to find the shortest vector.
- Large-scale runs where performance matters (compiled code is many times faster).

If you want, I can add a short script that enforces projected-block BKZ for both oracles and runs side-by-side comparisons with consistent parameters.
