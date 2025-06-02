# JAX *k*-D
Find *k*-nearest neighbors using a *k*-d tree in JAX!

This is an implementation of two GPU-friendly tree algorithms [[1](https://arxiv.org/abs/2211.00120), [2](https://arxiv.org/abs/2210.12859)] using only JAX primitives. The core `build_tree`, `query_neighbors`, and `count_neighbors` operations are compatible with JIT and automatic differentiation. They are reasonably fast when vectorized on GPU/TPU, but will be slower than SciPy's `KDTree` on CPU. For small problems where a pairwise distance matrix fits in memory, check whether brute force is faster (see `jaxkd.extras`).

If neighbor search is the performance bottleneck and you only use Nvidia GPUs, consider binding the lower-level [cudaKDTree](https://github.com/ingowald/cudaKDTree) library to JAX. This can be done with `jaxkd.cukd`, though it requires some [setup](##Performance). Be warned that this approach will not spark joy. The advantage of the pure JAX version is that it is portable and easy to use, with the ability to scale up to larger problems without the complexity of integrating non-JAX libraries. Try it out!

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

Additional helpful functionality can be found in `jaxkd.extras`.
- `query_neighbors_pairwise` and `count_neighbors_pairwise` for brute-force neighbor searches
- `k_means` for clustering using *k*-means++ initialization, thanks to [@NeilGirdhar](https://github.com/NeilGirdhar) for contributions

Suggestions and contributions for other extras are always welcome!

## Installation
To install, use `pip`. The only dependency is `jax`.
```
python -m pip install jaxkd
```
Or just grab `tree.py`.

## Performance

The relevant baseline for `jaxkd` is the original CUDA-based [cudaKDTree](https://github.com/ingowald/cudaKDTree) library, which implements the same algorithms at a lower level and can be bound to `jax` using the "[foreign function interface](https://docs.jax.dev/en/latest/ffi.html)". There will be a significant performance gain at the cost of portability and ease of use. There are also more features available in cudaKDTree if you are willing to your own bindings. Below are some rough numbers for speed on an H100 (`cukd` compiled with fixed *k*=16 hence the identical times). Try it out for yourself at the end of the [demo notebook](https://colab.research.google.com/github/dodgebc/jaxkd/blob/main/demo.ipynb)!

| Milliseconds      | Build tree      | Query *k*=1     | Query *k*=4     | Query *k*=16    |
|-------------------|-----------------|-----------------|-----------------|-----------------|
| `jaxkd` (100K, 3) | 11              | 18              | 36              | 70              |
| `jaxkd` (1M, 3)   | 67              | 53              | 266             | 773             |
| `jaxkd` (10M, 3)  | 811             | 491             | 2885            | 9285            |
| `cukd`  (100K, 3) | 2.4             | 0.8             | 0.8             | 0.8             |
| `cukd`  (1M, 3)   | 10              | 6               | 6               | 6               |
| `cukd`  (10M, 3)  | 67              | 128             | 128             | 128             |

If you decide you need the performance, the `external` folder above contains a minimal example of the required CUDA code, inspired by [this repo](https://github.com/EiffL/JaxKDTree) and updated to work with the current JAX FFI API. Assuming you have CMake and CUDA Toolkit, you should be able to build it from source by running these commands from the root of the repository.

```
cmake -S external -B external/build
cmake --build external/build
cp external/build/libjaxcukd.so jaxkd/
```

You can then `pip install .` and then use `jk.cukd` like this!

```python
jk.cukd.init()
jk.cukd.query_neighbors(points, queries, k=4)
```

Feel free to walk through this in the demo notebook on Google Colab first. Be warned that this is experimental code. It will be brittle and requires recompilation to change the dimension of input or the maximum number of neighbors returned. Ideally someone with more CUDA development experience would write a fully-featured interface to the library, but alas.