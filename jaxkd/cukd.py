import ctypes
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax import Array


def init() -> None:
    """
    Initialize the cudaKDTree library by loading the shared object file.
    This function should be called before using any cudaKDTree functionality.
    """
    root_path = Path(__file__).parent
    if not (root_path / "external").exists():
        raise RuntimeError("External library directory not found. Please install from source.")
    if not (root_path / "external" / "build").exists():
        raise RuntimeError("Build directory not found. Please run `cmake -B build` in 'external'.")
    so_path = next((Path(__file__).parent / "external" / "build").glob("libjaxcukd*.so"), None)
    if so_path is None:
        raise RuntimeError(
            "'libjaxcukd*.so' not found. Please run `cmake --build build in 'external'."
        )
    libjaxcukd = ctypes.cdll.LoadLibrary(so_path)
    jax.ffi.register_ffi_target(
        "build_and_query", jax.ffi.pycapsule(libjaxcukd.build_and_query), platform="gpu"
    )


@Partial(jax.jit, static_argnames=("k",))
def build_and_query(points: Array, queries: Array, k: int = 1) -> tuple[Array, Array]:
    """
    Build a k-d tree and query neighbors by calling cudaKDTree with JAX FFI.

    Args:
        points: (N, d) Points to build the k-d tree from.
        queries: (Q, d) Query points to find neighbors for.
        k (int): Number of neighbors to return.

    Returns:
        neighbors: (Q, k) Indices of the k nearest neighbors for each query point.
        distances: (Q, k) Distances to the k nearest neighbors for each query point.
    """
    call = jax.ffi.ffi_call(
        "build_and_query",
        jax.ShapeDtypeStruct((len(queries), k), jnp.int32),
        vmap_method="sequential",
    )
    neighbors = call(points, queries, k=np.int32(k))
    distances = jnp.linalg.norm(points[neighbors] - queries[:, jnp.newaxis], axis=-1)
    return neighbors, distances
