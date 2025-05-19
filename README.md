# JAX $k$-D
Find $k$-nearest neighbors using a $k$-d tree in JAX!

This is an implementation of two GPU-friendly tree algorithms [[1](https://arxiv.org/abs/2211.00120), [2](https://arxiv.org/abs/2210.12859)] using only XLA primitives. It is convenient and lightweight, but the CUDA-based [cudaKDTree](https://github.com/ingowald/cudaKDTree) may be a better choice depending on the application.

The core `build_tree`, `query_neighbors`, and `count_neighbors` operations are compatible with JIT and automatic differentiation. They are reasonably fast when vectorized on GPU, but will be much slower on CPU than `scipy.spatial.KDTree`. For small problems where a pairwise distance matrix fits in memory, check whether brute force is faster (see `extras.py`). The main advantage of `jaxkd` is the ability to scale up to larger problems without the complexity of integrating non-JAX libraries, especially when the neighbor search should not be the primary computational load.

## Usage

```python
import jax
import jaxkd as jk

kp, kq = jax.random.split(jax.random.key(83))
points = jax.random.normal(kp, shape=(100_000, 3))
queries = jax.random.normal(kq, shape=(10_000, 3))

tree = jk.build_tree(points)
counts = jk.count_neighbors(tree, queries, 0.1)
neighbors, distances = jk.query_neighbors(tree, queries, 10)
```

There is also `jaxkd.extras` with simple brute force versions of these for comparison, as well as a starter k-means implementation. More suggestions welcome!

## Installation
To install, use `pip`. The only dependency is `jax`.
```
python -m pip install jaxkd
```
Or just grab `tree.py`.

## Notes
- The `demo.ipynb` notebook in the source repository has some additional examples, including gradient-based optimization using neighbors and iterative clustering with $k$-means.
- The `query_neighbors` function is intended for small values of $k$ and does not use a max heap for simplicity.
- Some common $k$-d tree operations such as ball search are not implemented because they do not return a fixed size array. But there are probably others which could be implemented if there is a need. Suggestions welcome!
- Only the Euclidean distance is currently supported, this relatively easy to change if needed.