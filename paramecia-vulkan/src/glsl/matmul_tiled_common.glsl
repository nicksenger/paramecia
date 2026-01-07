// Shared helper functions for hierarchical tiled matmul kernels
// This file contains common utilities for 3-level tiling (Block → Warp → Thread)

#ifndef MATMUL_TILED_COMMON_GLSL
#define MATMUL_TILED_COMMON_GLSL

// ============================================================================
// Tile Size Configuration (via specialization constants or defines)
// ============================================================================
// These should be defined before including this file, or use defaults:
//   BM, BN, BK: Block tile sizes
//   WM, WN: Warp tile sizes
//   TM, TN: Thread tile sizes
//   WMITER: Warp M iterations

#ifndef BM
#define BM 16  // Block M (decode-optimized default)
#endif
#ifndef BN
#define BN 64  // Block N (decode-optimized default)
#endif
#ifndef BK
#define BK 32  // Block K (aligns with Q8_0 block size)
#endif
#ifndef WM
#define WM 16  // Warp M
#endif
#ifndef WN
#define WN 32  // Warp N
#endif
#ifndef TM
#define TM 8   // Thread M tile
#endif
#ifndef TN
#define TN 8   // Thread N tile
#endif
#ifndef WMITER
#define WMITER 1  // Warp M iterations
#endif

// ============================================================================
// Shared Memory Layout
// ============================================================================
// Padding to avoid bank conflicts (works on NVIDIA 32-way, AMD 32-way, Intel 16-way)
#define SHMEM_STRIDE_A (BK + 8)
#define SHMEM_STRIDE_B (BN + 8)

// Total shared memory size (in bytes):
// (BM × SHMEM_STRIDE_A + BK × SHMEM_STRIDE_B) × 4 bytes

// ============================================================================
// Workgroup Configuration
// ============================================================================
// Workgroup size is determined by tile sizes:
//   - Warps per workgroup = (BM/WM) × (BN/WN)
//   - Threads per workgroup = warps × 32 (subgroup size)
//   - For decode config (16×64): 1×2 = 2 warps = 64 threads
//   - For prefill config (64×64): 2×2 = 4 warps = 128 threads

// Compute number of warps
#define WARPS_M (BM / WM / WMITER)
#define WARPS_N (BN / WN)
#define NWARPS (WARPS_M * WARPS_N)

// Subgroup size (assume 32 for NVIDIA/AMD, works on Intel too)
#define WARP_SIZE 32

// Total threads per workgroup
#define THREADS_PER_WG (NWARPS * WARP_SIZE)

// ============================================================================
// Thread/Warp Coordinate Helpers
// ============================================================================

/// Compute warp and thread tile coordinates from local invocation IDs.
///
/// For 2D layout:
///   warp_id = y (0 to NWARPS-1)
///   lane_id = x (0 to 31)
///
/// Warp grid layout (warp M, warp N):
///   warp_m_idx = warp_id / WARPS_N
///   warp_n_idx = warp_id % WARPS_N
///
/// Thread tile position within warp:
///   thread_m_in_warp = (lane_id / (WN / TN)) * TM
///   thread_n_in_warp = (lane_id % (WN / TN)) * TN
///
/// Global thread tile coordinates:
///   thread_m = warp_m_base + thread_m_in_warp
///   thread_n = warp_n_base + thread_n_in_warp
void compute_tile_coords(
    uint warp_id,
    uint lane_id,
    uint wg_m,
    uint wg_n,
    out uint thread_m,
    out uint thread_n,
    out uint warp_m_base,
    out uint warp_n_base
) {
    // Warp position in the block
    uint warp_m_idx = warp_id / WARPS_N;
    uint warp_n_idx = warp_id % WARPS_N;

    // Warp base coordinates (in global M,N space)
    warp_m_base = wg_m + warp_m_idx * WM * WMITER;
    warp_n_base = wg_n + warp_n_idx * WN;

    // Number of threads per warp dimension
    uint threads_m = WM / TM;  // e.g., 16/8 = 2 threads along M
    uint threads_n = WN / TN;  // e.g., 32/8 = 4 threads along N

    // Map lane_id to 2D thread grid within warp
    // Note: Not all lanes may be active if threads_per_warp < WARP_SIZE
    uint thread_m_in_warp = (lane_id / threads_n) * TM;
    uint thread_n_in_warp = (lane_id % threads_n) * TN;

    // Global thread tile coordinates
    thread_m = warp_m_base + thread_m_in_warp;
    thread_n = warp_n_base + thread_n_in_warp;
}

/// Simpler version for common case where we just need thread coordinates
void get_thread_tile_coords(
    uint warp_id,
    uint lane_id,
    uint wg_m,
    uint wg_n,
    out uint thread_m,
    out uint thread_n
) {
    uint warp_m_base, warp_n_base;
    compute_tile_coords(warp_id, lane_id, wg_m, wg_n, thread_m, thread_n, warp_m_base, warp_n_base);
}

// ============================================================================
// Bounds Checking Helpers
// ============================================================================

/// Check if tile coordinates are within bounds
bool in_bounds(uint m, uint n, uint M, uint N) {
    return m < M && n < N;
}

/// Check if shared memory coordinates are within tile
bool in_tile_bounds(uint tile_m, uint tile_n, uint tile_M, uint tile_N) {
    return tile_m < tile_M && tile_n < tile_N;
}

// ============================================================================
// Cooperative Loading Helpers
// ============================================================================

/// Get the number of elements each thread should load for cooperative loading
/// Total elements = tile_elements
/// Each of THREADS_PER_WG threads loads: ceil(tile_elements / THREADS_PER_WG)
uint elements_per_thread(uint tile_elements) {
    return (tile_elements + THREADS_PER_WG - 1) / THREADS_PER_WG;
}

/// Convert flat thread ID to cooperative loading coordinates
/// For a tile of size [rows, cols], each thread loads elements at stride THREADS_PER_WG
void cooperative_load_coords(uint tid, uint cols, out uint row, out uint col) {
    row = tid / cols;
    col = tid % cols;
}

#endif // MATMUL_TILED_COMMON_GLSL
