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
 * @param h                             Pointer to global water heights memory
 * @param hu                            Pointer to global horizontal water momentums memory
 * @param b                             Pointer to global bathymetry memory
 * @param hNetUpdatesLeft               Pointer to global left going water updates memory
 * @param hNetUpdatesRight              Pointer to global right going water updates memory
 * @param huNetUpdatesLeft              Pointer to global left going momentum updates memory
 * @param huNetUpdatesRight             Pointer to global right going momentum updates memory
 * @param maxWaveSpeed                  Pointer to global maximum wavespeed memory
 * @param hScratch                      Pointer to local water heights scratch memory (LOCAL ONLY)
 * @param huScratch                     Pointer to local horizontal water momentums scratch memory (LOCAL ONLY)
 * @param bScratch                      Pointer to local bathymetry scratch memory (LOCAL ONLY)
 * @param hNetUpdatesLeftScratch        Pointer to local left going water updates scratch memory (LOCAL ONLY)
 * @param hNetUpdatesRightScratch       Pointer to local right going water updates scratch memory (LOCAL ONLY)
 * @param huNetUpdatesLeftScratch       Pointer to local left going momentum updates scratch memory (LOCAL ONLY)
 * @param huNetUpdatesRightScratch      Pointer to local right going momentum updates scratch memory (LOCAL ONLY)
 * @param maxWaveSpeedScratch           Pointer to local maximum wavespeed scratch memory (LOCAL ONLY)
 * @param edges                         Number of edges (LOCAL ONLY)
 * @param rows                          Number of rows (LOCAL ONLY)
 */
__kernel void dimensionalSplitting_XSweep_netUpdates(
    __global float* h,
    __global float* hu,
    __global float* b,
    __global float* hNetUpdatesLeft,
    __global float* hNetUpdatesRight,
    __global float* huNetUpdatesLeft,
    __global float* huNetUpdatesRight,
    __global float* maxWaveSpeed
#ifdef MEM_LOCAL
    ,
    __local float* hScratch,
    __local float* huScratch,
    __local float* bScratch,
    __local float* hNetUpdatesLeftScratch,
    __local float* hNetUpdatesRightScratch,
    __local float* huNetUpdatesLeftScratch,
    __local float* huNetUpdatesRightScratch,
    __local float* maxWaveSpeedScratch,
    __const uint edges, // cols-1
    __const uint rows
#endif
)
{    
#ifndef MEM_LOCAL
    // GLOBAL
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
#else
    // LOCAL
    uint localsize = get_local_size(0);
    uint gid = get_group_id(0);
    uint offset = gid * localsize * rows + get_group_id(1);
    // Number of floats to load (make sure we stay in bounds)
    uint num = min(localsize, edges-(gid * localsize)) + 1;
    
    event_t event[4];
    // local dst, global src, num floats, stride
    event[0] = async_work_group_strided_copy(hScratch, h+offset, num, rows, 0);
    event[1] = async_work_group_strided_copy(huScratch, hu+offset, num, rows, 0);
    event[2] = async_work_group_strided_copy(bScratch, b+offset, num, rows, 0);
    // wait for memory
    wait_group_events(3, event);
    
    uint id = get_local_id(0);
    
    if(get_global_id(0) < edges) {
        computeNetUpdates(
            hScratch[id], hScratch[id+1],
            huScratch[id], huScratch[id+1],
            bScratch[id], bScratch[id+1],
            &(hNetUpdatesLeftScratch[id]), &(hNetUpdatesRightScratch[id]),
            &(huNetUpdatesLeftScratch[id]), &(huNetUpdatesRightScratch[id]),
            &(maxWaveSpeedScratch[id])
        );
    }
    
    // Make sure all calculations have finished
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // write local memory back to global memory
    num--; // we're writing one update less than number of cells
    event[0] = async_work_group_strided_copy(hNetUpdatesLeft+offset, hNetUpdatesLeftScratch, num, rows, 0);
    event[1] = async_work_group_strided_copy(hNetUpdatesRight+offset, hNetUpdatesRightScratch, num, rows, 0);
    event[2] = async_work_group_strided_copy(huNetUpdatesLeft+offset, huNetUpdatesLeftScratch, num, rows, 0);
    event[3] = async_work_group_strided_copy(huNetUpdatesRight+offset, huNetUpdatesRightScratch, num, rows, 0);
    
    // Reduce maximum
    // TODO: reduce maximum in local memory
    // maxWaveSpeed[gid] = maxWave;
    event_t waveEvent = async_work_group_strided_copy(maxWaveSpeed+offset, maxWaveSpeedScratch, num, rows, 0);
    
    // Wait for async transfers
    wait_group_events(4, event);
    wait_group_events(1, &waveEvent); // TODO: remove this
#endif
}

/// Compute net updates (Y-Sweep)
/**
 * Kernel Range should be set to (#cols, #rows-1)
 * 
 * @param h                             Pointer to water heights
 * @param hv                            Pointer to vertical water momentums
 * @param b                             Pointer to bathymetry
 * @param hNetUpdatesLeft               Pointer to left going water updates
 * @param hNetUpdatesRight              Pointer to right going water updates
 * @param huNetUpdatesLeft              Pointer to left going momentum updates
 * @param huNetUpdatesRight             Pointer to right going momentum updates
 * @param maxWaveSpeed                  Pointer to global maximum wavespeed
 * @param hScratch                      Pointer to local water heights scratch memory (LOCAL ONLY)
 * @param hvScratch                     Pointer to local horizontal water momentums scratch memory (LOCAL ONLY)
 * @param bScratch                      Pointer to local bathymetry scratch memory (LOCAL ONLY)
 * @param hNetUpdatesLeftScratch        Pointer to local left going water updates scratch memory (LOCAL ONLY)
 * @param hNetUpdatesRightScratch       Pointer to local right going water updates scratch memory (LOCAL ONLY)
 * @param hvNetUpdatesLeftScratch       Pointer to local left going momentum updates scratch memory (LOCAL ONLY)
 * @param hvNetUpdatesRightScratch      Pointer to local right going momentum updates scratch memory (LOCAL ONLY)
 * @param maxWaveSpeedScratch           Pointer to local maximum wavespeed scratch memory (LOCAL ONLY)
 * @param cols                          Number of columns (LOCAL ONLY)
 * @param edges                         Number of edges (LOCAL ONLY)
 */
__kernel void dimensionalSplitting_YSweep_netUpdates(
    __global float* h,
    __global float* hv,
    __global float* b,
    __global float* hNetUpdatesLeft,
    __global float* hNetUpdatesRight,
    __global float* hvNetUpdatesLeft,
    __global float* hvNetUpdatesRight,
    __global float* maxWaveSpeed
#ifdef MEM_LOCAL
    ,
    __local float* hScratch,
    __local float* hvScratch,
    __local float* bScratch,
    __local float* hNetUpdatesLeftScratch,
    __local float* hNetUpdatesRightScratch,
    __local float* hvNetUpdatesLeftScratch,
    __local float* hvNetUpdatesRightScratch,
    __local float* maxWaveSpeedScratch,
    __const uint cols,
    __const uint edges // rows-1
#endif
)
{
#ifndef MEM_LOCAL
    // GLOBAL
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t rows = get_global_size(1);
    
    size_t leftId = colMajor(x, y, rows+1);
    size_t rightId = colMajor(x, y+1, rows+1);
    size_t updateId = colMajor(x, y, rows);
    
    computeNetUpdatesGlobal(
        h[leftId], h[rightId],
        hv[leftId], hv[rightId],
        b[leftId], b[rightId],
        &(hNetUpdatesLeft[updateId]), &(hNetUpdatesRight[updateId]),
        &(hvNetUpdatesLeft[updateId]), &(hvNetUpdatesRight[updateId]),
        &(maxWaveSpeed[updateId])
    );
#else
    // LOCAL
    uint localsize = get_local_size(1);
    uint gid = get_group_id(1);
    uint offset = get_group_id(0)*(edges+1) + gid * localsize;
    // Number of floats to load (make sure we stay in bounds)
    uint num = min(localsize, edges-(gid * localsize)) + 1;

    event_t event[4];
    // local dst, global src, num floats
    event[0] = async_work_group_copy(hScratch, h+offset, num, 0);
    event[1] = async_work_group_copy(hvScratch, hv+offset, num, 0);
    event[2] = async_work_group_copy(bScratch, b+offset, num, 0);
    // wait for memory
    wait_group_events(3, event);

    uint id = get_local_id(1);

    if(get_global_id(1) < edges) {
        computeNetUpdates(
            hScratch[id], hScratch[id+1],
            hvScratch[id], hvScratch[id+1],
            bScratch[id], bScratch[id+1],
            &(hNetUpdatesLeftScratch[id]), &(hNetUpdatesRightScratch[id]),
            &(hvNetUpdatesLeftScratch[id]), &(hvNetUpdatesRightScratch[id]),
            &(maxWaveSpeedScratch[id])
        );
    }

    // Make sure all calculations have finished
    barrier(CLK_LOCAL_MEM_FENCE);

    // write local memory back to global memory
    num--; // we're writing one update less than number of cells
    offset = get_group_id(0)*edges + gid * localsize;
    event[0] = async_work_group_copy(hNetUpdatesLeft+offset, hNetUpdatesLeftScratch, num, 0);
    event[1] = async_work_group_copy(hNetUpdatesRight+offset, hNetUpdatesRightScratch, num, 0);
    event[2] = async_work_group_copy(hvNetUpdatesLeft+offset, hvNetUpdatesLeftScratch, num, 0);
    event[3] = async_work_group_copy(hvNetUpdatesRight+offset, hvNetUpdatesRightScratch, num, 0);

    // Reduce maximum
    // TODO: reduce maximum in local memory
    // maxWaveSpeed[gid] = maxWave;
    event_t waveEvent = async_work_group_copy(maxWaveSpeed+offset, maxWaveSpeedScratch, num, 0);

    // Wait for async transfers
    wait_group_events(4, event);
    wait_group_events(1, &waveEvent); // TODO: remove this
#endif
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

/// Kernel setting boundary conditions (OUTFLOW or WALL) on the left boundary
/**
 * Kernel range should be set to (#rows)
 * 
 * @param h         Pointer to water heights
 * @param hu        Pointer to horizontal water momentums
 * @param hv        Pointer to vertical water momentums
 * @param left      Sign Sign of the horizontal momentum at left boundary (-1 for WALL, +1 for OUTFLOW)
 */
__kernel void setLeftBoundary(
    __global float* h,
    __global float* hu,
    __global float* hv,
    __const float leftSign)
{
    uint j = get_global_id(0);
    
    size_t srcId = colMajor(1, j, get_global_size(0));
    size_t dstId = colMajor(0, j, get_global_size(0));
    
    h[dstId] = h[srcId];
    hu[dstId] = leftSign*hu[srcId];
    hv[dstId] = hv[srcId];
    
    if(j == 0) {
        // first row, set corner
        srcId = colMajor(1, 1, get_global_size(0));
        h[dstId] = h[srcId];
        hu[dstId] = hu[srcId];
        hv[dstId] = hv[srcId];
    }
    
    if(j == get_global_size(0)-1) {
        // last row, set corner
        srcId = colMajor(1, j-1, get_global_size(0));
        h[dstId] = h[srcId];
        hu[dstId] = hu[srcId];
        hv[dstId] = hv[srcId];
    }
}

/// Kernel setting boundary conditions (OUTFLOW or WALL) on the right boundary
/**
 * Kernel range should be set to (#rows)
 * 
 * @param h         Pointer to water heights
 * @param hu        Pointer to horizontal water momentums
 * @param hv        Pointer to vertical water momentums
 * @param cols      Total number of columns in the grid (including ghosts)
 * @param rightSign Sign of the horizontal momentum at right boundary (-1 for WALL, +11 for OUTFLOW)
 */
__kernel void setRightBoundary(
    __global float* h,
    __global float* hu,
    __global float* hv,
    __const uint cols,
    __const float rightSign)
{
    uint j = get_global_id(0);
    
    size_t srcId = colMajor((cols-1)-1, j, get_global_size(0));
    size_t dstId = colMajor((cols-1), j, get_global_size(0));
    
    h[dstId] = h[srcId];
    hu[dstId] = rightSign*hu[srcId];
    hv[dstId] = hv[srcId];
    
    if(j == 0) {
        // first row, set corner
        srcId = colMajor((cols-1)-1, 1, get_global_size(0));
        h[dstId] = h[srcId];
        hu[dstId] = hu[srcId];
        hv[dstId] = hv[srcId];
    }
    
    if(j == get_global_size(0)-1) {
        // last row, set corner
        srcId = colMajor((cols-1)-1, j-1, get_global_size(0));
        h[dstId] = h[srcId];
        hu[dstId] = hu[srcId];
        hv[dstId] = hv[srcId];
    }
}

/// Kernel setting boundary conditions (OUTFLOW or WALL) on the top/bottom boundaries
/**
 * Kernel range should be set to (#cols)
 * 
 * @param h         Pointer to water heights
 * @param hu        Pointer to horizontal water momentums
 * @param hv        Pointer to vertical water momentums
 * @param rows      Total number of rows in the grid (including ghosts)
 * @param bottomSign  Sign of the vertical momentum at bottom boundary (-1 for WALL, +1 for OUTFLOW)
 * @param topSign   Sign of the vertical momentum at top boundary (-1 for WALL, +1 for OUTFLOW)

 */
__kernel void setBottomTopBoundary(
    __global float* h,
    __global float* hu,
    __global float* hv,
    __const uint rows,
    __const float bottomSign,
    __const float topSign)
{
    uint i = get_global_id(0);
    
    uint srcId = colMajor(i, 1, rows);
    uint dstId = colMajor(i, 0, rows);
    
    h[dstId] = h[srcId];
    hu[dstId] = hu[srcId];
    hv[dstId] = bottomSign*hv[srcId];
    
    srcId = colMajor(i, (rows-1)-1, rows);
    dstId = colMajor(i, (rows-1), rows);
    
    h[dstId] = h[srcId];
    hu[dstId] = hu[srcId];
    hv[dstId] = topSign*hv[srcId];
}
