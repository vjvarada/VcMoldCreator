# Fast C++ Algorithms for VcMoldCreator

This directory contains optimized C++ implementations of performance-critical algorithms
using pybind11 for Python bindings. These provide **10-20x speedups** over the pure Python
implementations.

## Optimized Algorithms

### 1. Dijkstra Escape Labeling
- **Function**: `dijkstra_escape_labeling()`
- **Purpose**: Finds shortest weighted paths from interior vertices to H1/H2 boundaries
- **Speedup**: ~15-20x faster than Python heapq-based implementation
- **Used by**: Parting surface computation, mold half classification

### 2. Edge Boundary Label Computation
- **Function**: `compute_edge_boundary_labels()`
- **Purpose**: Labels edges based on vertex boundary membership and surface geometry
- **Speedup**: ~10-15x faster with optimized hash maps
- **Used by**: Tetrahedral mesh visualization, edge classification

## Building

### Prerequisites

```bash
pip install pybind11 numpy
```

### Windows (Visual Studio)

Make sure you have Visual Studio Build Tools installed with C++ support.

```powershell
cd desktop_app/core
python setup_cpp.py build_ext --inplace
```

### Linux/macOS

```bash
cd desktop_app/core
python setup_cpp.py build_ext --inplace
```

### Verify Installation

```python
from core import fast_algorithms
print("C++ module loaded successfully!")

# Check available functions
print(dir(fast_algorithms))
```

## Usage

The Python code automatically uses C++ implementations when available:

```python
from core.tetrahedral_mesh import run_dijkstra_escape_labeling, compute_edge_boundary_labels

# These will use C++ if compiled, Python fallback otherwise
labels = compute_edge_boundary_labels(tet_result)
escape_labels, indices, distances, dests, paths = run_dijkstra_escape_labeling(tet_result)
```

Check implementation status:

```python
from core.fast_algorithms_wrapper import get_implementation_info
print(get_implementation_info())
# {'cpp_available': True, 'dijkstra': 'C++', 'edge_labels': 'C++', 'speedup_estimate': '10-20x'}
```

## Performance Comparison

| Algorithm | Python Time | C++ Time | Speedup |
|-----------|-------------|----------|---------|
| Dijkstra (50K vertices) | ~800ms | ~50ms | 16x |
| Edge Labels (100K edges) | ~200ms | ~15ms | 13x |

## Troubleshooting

### "No module named 'fast_algorithms'"
The C++ module hasn't been compiled. Run the build command above.

### Compilation errors on Windows
Make sure Visual Studio Build Tools are installed:
```powershell
winget install Microsoft.VisualStudio.2022.BuildTools
```
Then install the "Desktop development with C++" workload.

### Compilation errors on Linux
Install the required packages:
```bash
sudo apt-get install build-essential python3-dev
```

### OpenMP warnings
OpenMP is optional and used for potential future parallelization. The code will work without it.

## Development

To modify the C++ code:

1. Edit `cpp/fast_algorithms.cpp`
2. Rebuild: `python setup_cpp.py build_ext --inplace`
3. Restart Python to reload the module

## Files

```
core/
├── cpp/
│   └── fast_algorithms.cpp    # C++ implementations with pybind11 bindings
├── setup_cpp.py               # Build script
├── fast_algorithms_wrapper.py # Python wrapper with fallback
└── README_CPP.md             # This file
```
