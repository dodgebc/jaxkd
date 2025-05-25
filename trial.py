import argparse
import time

import jax

import jaxkd as jk

key = jax.random.key(83)

argparse = argparse.ArgumentParser(description="Simple speed trials")
argparse.add_argument(
    "--points", type=int, default=1024, help="Number of random points to generate"
)
argparse.add_argument("--dims", type=int, default=2, help="Number of dimensions")
argparse.add_argument("--neighbors", type=int, default=4, help="Number of neighbors to query")
argparse.add_argument(
    "--queries", type=int, default=0, help="Number of queries to run (0 for all points)"
)
argparse.add_argument(
    "--batches", type=int, default=1, help="Number of batches to evaluate queries in"
)
argparse.add_argument("--trials", type=int, default=2, help="Number of trials to run")
args = argparse.parse_args()
print(args)

for i in range(args.trials):
    print(f"\nTrial {i + 1}")
    start = time.time()
    key, subkey = jax.random.split(key)
    points = jax.random.normal(subkey, shape=(args.points, args.dims))
    tree = jk.build_tree(points)
    tree[1].block_until_ready()
    print(f"Built kd-tree in {time.time() - start:.3f} seconds")

    start = time.time()
    queries = points[: args.queries] if args.queries > 0 else points

    if args.batches == 1:
        neighbors, distances = jk.query_neighbors(tree, queries, k=args.neighbors)
    else:
        print(tree)
        neighbors, distances = jax.lax.map(
            lambda q: jk.tree._single_query_neighbors(tree, q, k=args.neighbors),  # noqa: F821
            queries,
            batch_size=len(points) // args.batches,
        )
    neighbors.block_until_ready()
    print(f"Found query neighbors in {time.time() - start:.3f} seconds")

    del points, tree, queries, neighbors, distances
