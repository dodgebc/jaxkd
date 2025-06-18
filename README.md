# JAX *k*-D
Find *k*-nearest neighbors using a *k*-d tree in JAX!

This is an implementation of two GPU-friendly tree algorithms [[1](https://arxiv.org/abs/2211.00120), [2](https://arxiv.org/abs/2210.12859)] using only JAX primitives. The core `build_tree`, `query_neighbors`, and `count_neighbors` operations are compatible with JIT and automatic differentiation. They are reasonably fast when vectorized on GPU/TPU, but will be slower than SciPy's `KDTree` on CPU. For small problems where a pairwise distance matrix fits in memory, check whether brute force is faster (see `jaxkd.extras`).

If neighbor search is the performance bottleneck and you only use Nvidia GPUs, there is a [CUDA extension](https://github.com/dodgebc/jaxkd-cuda)  available that can be installed as an optional dependency (see below). The intention is to match the behavior of the pure-JAX version and integrate seamlessly with the `cuda=True` argument. Building the extension will require CMake and NVCC installed on your system. There may be some rough edges and the internal workings may change.

For even more power and flexibility, consider binding the original [cudaKDTree](https://github.com/ingowald/cudaKDTree) library to JAX. Functionality will be different as described in the [jaxkd-cuda](https://github.com/dodgebc/jaxkd-cuda) repository, where example bindings can also be found. Be warned that they will not spark joy. The advantage of the pure-JAX version is that it is portable and easy to use, with the ability to scale up to larger problems without the complexity of integrating non-JAX libraries. Try it out!

<a target="_blank" href="https://colab.research.google.com/github/dodgebc/jaxkd/blob/main/demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


## Usage

```python
import jax
import jaxkd as jk

kp, kq = jax.random.split(jax.random.key(83))
points = jax.random.normal(kp, shape=(100_000, 3))
queries = jax.random.normal(kq, shape=(10_000, 3))

tree = jk.build_tree(points)
counts = jk.count_neighbors(tree, queries, r=0.1)
neighbors, distances = jk.query_neighbors(tree, queries, k=10)
```

There is also a one-step `build_and_query` for convenience, and all these functions accept `cuda=True` to use the CUDA extension if it is installed.

Additional helpful functionality can be found in `jaxkd.extras`.
- `query_neighbors_pairwise` and `count_neighbors_pairwise` for brute-force neighbor searches
- `k_means` for clustering using *k*-means++ initialization, thanks to [@NeilGirdhar](https://github.com/NeilGirdhar) for contributions

Suggestions and contributions for other extras are always welcome!

## Installation
To install, use `pip`. The only dependency is `jax`.
```
python -m pip install jaxkd
```
Or with the CUDA extension.
```
python -m pip install jaxkd[cuda]
```
Or just grab `tree.py`.