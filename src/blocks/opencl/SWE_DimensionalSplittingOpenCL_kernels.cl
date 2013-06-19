/// Returns the 1D array index for a 2D grid stored in column major order
/**
 * @param x The x position
 * @param y the y position
 * @param rows The number of rows in the grid
 */
inline size_t colMajor(size_t x, size_t y, size_t rows)
{
    return (x * rows) + y;
}

/// Returns the 1D array index for a 2D grid stored in row major order
/**
 * @param x The x position
 * @param y the y position
 * @param cols The number of columns in the grid
 */
inline size_t rowMajor(size_t x, size_t y, size_t cols)
{
    return (y * cols) + x;
}

/// Compute net updates (X-Sweep)
/**
 * Kernel Range should be set to (#cols-1, #rows)
 * 
 * @param h                     Pointer to water heights
 * @param hu                    Pointer to horizontal water momentums
 * @param b                     Pointer to bathymetry
 * @param hNetUpdatesLeft       Pointer to left going water updates
 * @param hNetUpdatesRight      Pointer to right going water updates
 * @param huNetUpdatesLeft      Pointer to left going momentum updates
 * @param huNetUpdatesRight     Pointer to right going momentum updates
 * @param maxWaveSpeed          Pointer to global maximum wavespeed
 */
__kernel void dimensionalSplitting_XSweep_netUpdates(
    __global float* h,
    __global float* hu,
    __global float* b,
    __global float* hNetUpdatesLeft,
    __global float* hNetUpdatesRight,
    __global float* huNetUpdatesLeft,
    __global float* huNetUpdatesRight,
    __global float* maxWaveSpeed)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t rows = get_global_size(1);
    
    size_t leftId = colMajor(x, y, rows);
    size_t rightId = colMajor(x+1, y, rows);
    
    computeNetUpdates(
        h[leftId], h[rightId],
        hu[leftId], hu[rightId],
        b[leftId], b[rightId],
        &(hNetUpdatesLeft[leftId]), &(hNetUpdatesRight[leftId]),
        &(huNetUpdatesLeft[leftId]), &(huNetUpdatesRight[leftId]),
        &(maxWaveSpeed[leftId])
    );
}

/// Compute net updates (Y-Sweep)
/**
 * Kernel Range should be set to (#cols, #rows-1)
 * 
 * @param h                     Pointer to water heights
 * @param hv                    Pointer to vertical water momentums
 * @param b                     Pointer to bathymetry
 * @param hNetUpdatesLeft       Pointer to left going water updates
 * @param hNetUpdatesRight      Pointer to right going water updates
 * @param huNetUpdatesLeft      Pointer to left going momentum updates
 * @param huNetUpdatesRight     Pointer to right going momentum updates
 * @param maxWaveSpeed          Pointer to global maximum wavespeed
 */
__kernel void dimensionalSplitting_YSweep_netUpdates(
    __global float* h,
    __global float* hv,
    __global float* b,
    __global float* hNetUpdatesLeft,
    __global float* hNetUpdatesRight,
    __global float* hvNetUpdatesLeft,
    __global float* hvNetUpdatesRight,
    __global float* maxWaveSpeed)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t rows = get_global_size(1);
    
    size_t leftId = colMajor(x, y, rows+1);
    size_t rightId = colMajor(x, y+1, rows+1);
    size_t updateId = colMajor(x, y, rows);
    
    computeNetUpdates(
        h[leftId], h[rightId],
        hv[leftId], hv[rightId],
        b[leftId], b[rightId],
        &(hNetUpdatesLeft[updateId]), &(hNetUpdatesRight[updateId]),
        &(hvNetUpdatesLeft[updateId]), &(hvNetUpdatesRight[updateId]),
        &(maxWaveSpeed[updateId])
    );
}

/// Update Unknowns (X-Sweep)
/**
 * Kernel Range should be set to (#cols-2, #rows)
 * 
 * @param dt_dx                 The desired update step
 * @param h                     Pointer to water heights
 * @param hu                    Pointer to horizontal water momentums
 * @param hNetUpdatesLeft       Pointer to left going water updates
 * @param hNetUpdatesRight      Pointer to right going water updates
 * @param huNetUpdatesLeft      Pointer to left going momentum updates
 * @param huNetUpdatesRight     Pointer to right going momentum updates
 */
__kernel void dimensionalSplitting_XSweep_updateUnknowns(
    float dt_dx,
    __global float* h,
    __global float* hu,
    __global float* hNetUpdatesLeft,
    __global float* hNetUpdatesRight,
    __global float* huNetUpdatesLeft,
    __global float* huNetUpdatesRight)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t rows = get_global_size(1);
    
    size_t leftId = colMajor(x, y, rows); // [x][y]
    size_t rightId = colMajor(x+1, y, rows); // [x+1][y]
    
    // update heights
    h[rightId] -= dt_dx * (hNetUpdatesRight[leftId] + hNetUpdatesLeft[rightId]);
    // Update momentum in x-direction
    hu[rightId] -= dt_dx * (huNetUpdatesRight[leftId] + huNetUpdatesLeft[rightId]);
    
    // Catch negative heights
    h[rightId] = fmax(h[rightId], 0.f);
}

/// Update Unknowns (Y-Sweep)
/**
 * Kernel Range should be set to (#cols, #rows-2)
 * 
 * @param dt_dy                 The desired update step
 * @param h                     Pointer to water heights
 * @param hv                    Pointer to vertical water momentums
 * @param hNetUpdatesLeft       Pointer to left going water updates
 * @param hNetUpdatesRight      Pointer to right going water updates
 * @param hvNetUpdatesLeft      Pointer to left going momentum updates
 * @param hvNetUpdatesRight     Pointer to right going momentum updates
 */
__kernel void dimensionalSplitting_YSweep_updateUnknowns(
    float dt_dy,
    __global float* h,
    __global float* hv,
    __global float* hNetUpdatesLeft,
    __global float* hNetUpdatesRight,
    __global float* hvNetUpdatesLeft,
    __global float* hvNetUpdatesRight)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t rows = get_global_size(1);
    
    size_t cellId = colMajor(x, y+1, rows+2);
    size_t leftId = colMajor(x, y, rows+1); // [x][y]
    size_t rightId = colMajor(x, y+1, rows+1); // [x][y+1]
    
    // update heights
    h[cellId] -= dt_dy * (hNetUpdatesRight[leftId] + hNetUpdatesLeft[rightId]);
    // Update momentum in x-direction
    hv[cellId] -= dt_dy * (hvNetUpdatesRight[leftId] + hvNetUpdatesLeft[rightId]);
    
    // Catch negative heights
    h[cellId] = fmax(h[cellId], 0.f);
}

/// Kernel to reduce the maximum value of an array (or linearized 2D grid) (CPU Version)
/**
 * Notes:
 * - The work group size MUST be a power of 2 to work properly!
 * - This kernel is destructive, e.g. the values-Array will be overriden
 * - The maximum for each work group can be read from values[groupId*groupSize*stride]
 * - This kernel is NOT suited for execution on CPUs (due to limited work group size)
 * 
 * @param values Pointer to the array
 * @param length Number of elements in value array
 * @param stride The stride of values to take into account
 * @param scratch Pointer to local scratch memory (at least sizeof(float)*workgroupsize)
 */
__kernel void reduceMaximum(
    __global float* values,
    __const uint length,
    __const uint stride,
    __local float* scratch)
{
    size_t global_id = get_global_id(0);
    size_t source_id = stride*global_id;
    size_t local_id = get_local_id(0);
    
    if(source_id < length) {
        scratch[local_id] = values[source_id];
    } else {
        scratch[local_id] = -INFINITY;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint i = 2; i <= get_local_size(0); i <<= 1) {
        // Fast modulo operation (for i being a power of two)
        if((local_id & (i-1)) == 0) {
            scratch[local_id] = fmax(scratch[ local_id ], scratch[ local_id + (i>>1) ]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if(local_id == 0) {
        values[source_id] = scratch[0];
    }
}


/// Kernel to reduce the maximum value of an array (or linearized 2D grid) (CPU Version)
/**
 * Notes:
 * - The global ND range MUST NOT exceed ceil(size/block)
 * - This kernel is destructive, e.g. the values-Array will be overriden
 * - The maximum for each block can be read from values[globalId*block*stride]
 * - This kernel is ONLY suited for execution on CPUs
 * 
 * @param values Pointer to the array
 * @param length Number of elements in value array
 * @param block The block size to process
 * @param stride The stride of values to take into account
 */
__kernel void reduceMaximumCPU(
    __global float* values,
    __const uint length,
    __const uint block,
    __const uint stride)
{
    size_t global_id = get_global_id(0);
    
    uint start = global_id * block * stride;
    uint end = (global_id+1) * block * stride;
    // make sure we stay inside array bounds
    end = min(end, length);
    
    float acc = -INFINITY;
    
    for(uint i = start; i < end; i += stride) {
        acc = fmax(acc, values[i]);
    }
    
    values[start] = acc;
}
