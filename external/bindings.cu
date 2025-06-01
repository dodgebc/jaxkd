#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "cukd/builder.h"
#include "cukd/knn.h"

// maximum number of neighbors to return, must be known at compile time
#define FIXED_K 16

namespace ffi = xla::ffi;

using data_t = float3;
using data_traits = cukd::default_data_traits<float3>;

// CUDA KNN kernel, adapted from cukd sample code
__global__ void d_knn(
    const cukd::SpatialKDTree<float3, data_traits> tree,
    const float3* d_queries,
    size_t numQueries,
    int32_t k,
    int32_t* d_results)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numQueries) return;

    // make fixed candidate list for small k, maximum 16 neighbors by default
    // can also set a max radius less than infinity if desired
    cukd::FixedCandidateList<FIXED_K> result(INFINITY);

    // perform stack-based KNN search
    cukd::stackBased::knn<decltype(result), float3, data_traits>(result, tree, d_queries[tid]);

    // transfer indices to results
    for (int i = 0; i < k; i++) {
        d_results[tid * k + i] = result.decode_pointID(result.entry[i]);
    }
}

// Build tree and perform queries
ffi::Error build_and_query_impl(
    ffi::Buffer<ffi::F32> points,
    ffi::Buffer<ffi::F32> queries,
    int32_t k,
    ffi::ResultBuffer<ffi::S32> neighbors)
{
    // checks
    if (k > FIXED_K) {
        return ffi::Error::InvalidArgument("knn compiled for k <= " + std::to_string(FIXED_K) + " but got k = " + std::to_string(k) + " must recompile with larger FIXED_K");
    } else if (k < 1) {
        return ffi::Error::InvalidArgument("k must be at least 1, got k = " + std::to_string(k));
    }
    if (points.dimensions()[1] != 3 || queries.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("points and queries must be 3D points, must recompile for other dimensions");
    }

    // extract dimensions and pointers from input arrays
    size_t numPoints = points.dimensions()[0];
    size_t numQueries = queries.dimensions()[0];
    float3* input_points = reinterpret_cast<float3*>(points.typed_data());
    float3* input_queries = reinterpret_cast<float3*>(queries.typed_data());
    int32_t* output_neighbors = neighbors->typed_data();

    // build spatial k-d tree
    cukd::SpatialKDTree<float3, data_traits> tree;
    cukd::BuildConfig buildConfig{};
    buildTree(tree, input_points, numPoints, buildConfig);
    CUKD_CUDA_SYNC_CHECK();

    // run knn kernel
    int threadsPerBlock = 256;
    int numBlocks = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    d_knn<<<numBlocks, threadsPerBlock>>>(tree, input_queries, numQueries, k, output_neighbors);
    cudaDeviceSynchronize();

    cukd::free(tree);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    build_and_query, build_and_query_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()  // points
        .Arg<ffi::Buffer<ffi::F32>>()  // queries
        .Attr<int32_t>("k")            // k
        .Ret<ffi::Buffer<ffi::S32>>()  // neighbors
);







// Test build tree (for benchmarking or debugging)
ffi::Error build_test_impl(
    ffi::Buffer<ffi::F32> points,
    ffi::ResultBuffer<ffi::S32> ids)
{
    // checks
    if (points.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("points must be 3D points, must recompile for other dimensions");
    }

    // extract dimensions and pointers from input arrays
    size_t numPoints = points.dimensions()[0];
    float3* input_points = reinterpret_cast<float3*>(points.typed_data());
    int32_t* output_ids = ids->typed_data();

    // build spatial k-d tree
    cukd::SpatialKDTree<float3, data_traits> tree;
    cukd::BuildConfig buildConfig{};
    buildTree(tree, input_points, numPoints, buildConfig);
    CUKD_CUDA_SYNC_CHECK();

    // copy primitive IDs to output
    cudaMemcpy(output_ids, tree.primIDs, numPoints * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    cukd::free(tree);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    build_test, build_test_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()  // points
        .Ret<ffi::Buffer<ffi::S32>>()  // ids
);