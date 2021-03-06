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

/// Kernel to reduce the maximum value of an array in local memory
/**
 * After return of the function, the result can be read from the first
 * array element.
 * Note that ALL local memory operations on the array values must have
 * finished before calling this function -> use local memory fence.
 *
 * @param values The array values in local memory
 * @param length The array length (length MUST BE a power of two)
 * @param local_id Local ID of current work item
 */
void localReduceMaximum(__local float* values, unsigned int length, unsigned int local_id)
{
    for(unsigned int i = 2; i <= length; i <<= 1) {
        // Fast modulo operation (for i being a power of two)
        if((local_id & (i-1)) == 0) {
            values[local_id] = fmax(values[ local_id ], values[ local_id + (i>>1) ]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/// Compute net updates (X-Sweep)
/**
 * Kernel Range should be set to (#cols-1, #rows)
 *
 * If maxWaveSpeed is reduced globally, the dimensions for maxWaveSpeeds
 * are (cols-1)*rows (aka edges*rows). For net-updates, the dimensions are
 * cols*rows (aka (edges+1)*rows) since we need to keep room for the net-updates
 * at the edge that are copied from the "next" device.
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
    size_t edges = get_global_size(0);
    
    size_t leftId = colMajor(x, y, rows);
    size_t rightId = colMajor(x+1, y, rows);
    size_t updateId = rowMajor(x, y, edges+1); // leave room for edge-update from next device
    size_t waveId = rowMajor(x, y, edges); // leave room for edge-update from next device
    computeNetUpdates(
        h[leftId], h[rightId],
        hu[leftId], hu[rightId],
        b[leftId], b[rightId],
        &(hNetUpdatesLeft[updateId]), &(hNetUpdatesRight[updateId]),
        &(huNetUpdatesLeft[updateId]), &(huNetUpdatesRight[updateId]),
        &(maxWaveSpeed[waveId])
    );
#else
    // LOCAL
    size_t localsize = get_local_size(0);
    size_t gid = get_group_id(0);
    size_t start = gid*localsize;
    size_t offset = colMajor(start, get_group_id(1), rows);
    // Number of floats to load (make sure we stay in bounds)
    size_t num = min(localsize, edges-start) + 1;
    
    event_t event[4];
    // local dst, global src, num floats, stride
    event[0] = async_work_group_strided_copy(hScratch, h+offset, num, rows, 0);
    event[1] = async_work_group_strided_copy(huScratch, hu+offset, num, rows, 0);
    event[2] = async_work_group_strided_copy(bScratch, b+offset, num, rows, 0);
    // wait for memory
    wait_group_events(3, event);
    
    size_t id = get_local_id(0);
    
    if(id < num) {
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
    offset = rowMajor(start, get_group_id(1), edges+1);
    
    event[0] = async_work_group_copy(hNetUpdatesLeft+offset, hNetUpdatesLeftScratch, num, 0);
    event[1] = async_work_group_copy(hNetUpdatesRight+offset, hNetUpdatesRightScratch, num, 0);
    event[2] = async_work_group_copy(huNetUpdatesLeft+offset, huNetUpdatesLeftScratch, num, 0);
    event[3] = async_work_group_copy(huNetUpdatesRight+offset, huNetUpdatesRightScratch, num, 0);
    
    // Reduce maximum
#ifdef LOCAL_REDUCE
    // fill buffer with neutral element -INFINITY to a length of a power of two
    if(id >= num)
        maxWaveSpeedScratch[id] = -INFINITY;
    // reduce maximum locally
    localReduceMaximum(maxWaveSpeedScratch, localsize, id);
    // Store maximum of group
    if(id == 0)
        maxWaveSpeed[rowMajor(gid, get_group_id(1), get_num_groups(0))] = maxWaveSpeedScratch[0];
#else
    offset = rowMajor(start, get_group_id(1), edges);
    event_t waveEvent = async_work_group_copy(maxWaveSpeed+offset, maxWaveSpeedScratch, num, 0);
    wait_group_events(1, &waveEvent);
#endif
    
    // Wait for async transfers
    wait_group_events(4, event);
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
    
    computeNetUpdates(
        h[leftId], h[rightId],
        hv[leftId], hv[rightId],
        b[leftId], b[rightId],
        &(hNetUpdatesLeft[updateId]), &(hNetUpdatesRight[updateId]),
        &(hvNetUpdatesLeft[updateId]), &(hvNetUpdatesRight[updateId]),
        &(maxWaveSpeed[updateId])
    );
#else
    // LOCAL
    size_t localsize = get_local_size(1);
    size_t gid = get_group_id(1);
    size_t start = gid*localsize;
    size_t offset = colMajor(get_group_id(0), start, edges+1);
    // Number of floats to load (make sure we stay in bounds)
    size_t num = min(localsize, edges-start) + 1;

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
    offset = colMajor(get_group_id(0), start, edges);
    event[0] = async_work_group_copy(hNetUpdatesLeft+offset, hNetUpdatesLeftScratch, num, 0);
    event[1] = async_work_group_copy(hNetUpdatesRight+offset, hNetUpdatesRightScratch, num, 0);
    event[2] = async_work_group_copy(hvNetUpdatesLeft+offset, hvNetUpdatesLeftScratch, num, 0);
    event[3] = async_work_group_copy(hvNetUpdatesRight+offset, hvNetUpdatesRightScratch, num, 0);

    // Reduce maximum
#ifdef DEBUG
#ifdef LOCAL_REDUCE
    // fill buffer with neutral element -INFINITY to a length of a power of two
    if(id >= num)
        maxWaveSpeedScratch[id] = -INFINITY;
    // reduce maximum locally
    localReduceMaximum(maxWaveSpeedScratch, localsize, id);
    // Store maximum of group
    if(id == 0)
        maxWaveSpeed[rowMajor(get_group_id(0), gid, get_num_groups(0))] = maxWaveSpeedScratch[0];
#else
    event_t waveEvent = async_work_group_copy(maxWaveSpeed+offset, maxWaveSpeedScratch, num, 0);
    wait_group_events(1, &waveEvent);
#endif
#endif
    // Wait for async transfers
    wait_group_events(4, event);
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
    __global float* huNetUpdatesRight
#ifdef MEM_LOCAL
    ,
    __local float* hScratch,
    __local float* huScratch,
    __local float* hNetUpdatesLeftScratch,
    __local float* hNetUpdatesRightScratch,
    __local float* huNetUpdatesLeftScratch,
    __local float* huNetUpdatesRightScratch,
    __const uint edges, // cols-2
    __const uint rows
#endif
        )
{
#ifndef MEM_LOCAL
    // GLOBAL
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t rows = get_global_size(1);
    size_t cols = get_global_size(0)+1;
    
    size_t cellId = colMajor(x+1, y, rows);
    size_t leftId = rowMajor(x, y, cols);
    size_t rightId = rowMajor(x+1, y, cols);
    
    // update heights
    h[cellId] -= dt_dx * (hNetUpdatesRight[leftId] + hNetUpdatesLeft[rightId]);
    // Update momentum in x-direction
    hu[cellId] -= dt_dx * (huNetUpdatesRight[leftId] + huNetUpdatesLeft[rightId]);
    
    // Catch negative heights
    h[cellId] = fmax(h[cellId], 0.f);
#else
    // LOCAL
    size_t id = get_local_id(0);
    size_t gid = get_group_id(0);
    size_t localsize = get_local_size(0);
    size_t start = gid*localsize;
    size_t cols = edges+1;
    
    size_t cellOffset = colMajor(start+1, get_group_id(1), rows); // skip ghost column
    size_t leftOffset = rowMajor(start, get_group_id(1), cols);
    size_t rightOffset = leftOffset+1;
    
    size_t num = min(localsize, edges-start);
    
    event_t event[6];
    event[0] = async_work_group_copy(hNetUpdatesLeftScratch, hNetUpdatesLeft+rightOffset, num, 0);
    event[1] = async_work_group_copy(hNetUpdatesRightScratch, hNetUpdatesRight+leftOffset, num, 0);
    event[2] = async_work_group_copy(huNetUpdatesLeftScratch, huNetUpdatesLeft+rightOffset, num, 0);
    event[3] = async_work_group_copy(huNetUpdatesRightScratch, huNetUpdatesRight+leftOffset, num, 0);
    event[4] = async_work_group_strided_copy(hScratch, h+cellOffset, num, rows, 0);
    event[5] = async_work_group_strided_copy(huScratch, hu+cellOffset, num, rows, 0);
    wait_group_events(6, event);
    
    // make sure we stay inside bounds
    if(id < num) {
        // update heights
        hScratch[id] -= dt_dx * (hNetUpdatesRightScratch[id] + hNetUpdatesLeftScratch[id]);
        // Update momentum in x-direction
        huScratch[id] -= dt_dx * (huNetUpdatesRightScratch[id] + huNetUpdatesLeftScratch[id]);
        
        // Catch negative heights
        hScratch[id] = fmax(hScratch[id], 0.f);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    event[0] = async_work_group_strided_copy(h+cellOffset, hScratch, num, rows, 0);
    event[1] = async_work_group_strided_copy(hu+cellOffset, huScratch, num, rows, 0);
    wait_group_events(2, event);
#endif
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
    __global float* hvNetUpdatesRight
#ifdef MEM_LOCAL
    ,
    __local float* hScratch,
    __local float* hvScratch,
    __local float* hNetUpdatesLeftScratch,
    __local float* hNetUpdatesRightScratch,
    __local float* hvNetUpdatesLeftScratch,
    __local float* hvNetUpdatesRightScratch,
    __const uint cols,
    __const uint edges // rows-1
#endif
)
{
#ifndef MEM_LOCAL
    // GLOBAL 
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t edges = get_global_size(1)+1;
    size_t rows = edges+1;
    
    size_t cellId = colMajor(x, y+1, rows);
    size_t leftId = colMajor(x, y, edges);
    size_t rightId = colMajor(x, y+1, edges);
    
    // update heights
    h[cellId] -= dt_dy * (hNetUpdatesRight[leftId] + hNetUpdatesLeft[rightId]);
    // Update momentum in x-direction
    hv[cellId] -= dt_dy * (hvNetUpdatesRight[leftId] + hvNetUpdatesLeft[rightId]);
    
    // Catch negative heights
    h[cellId] = fmax(h[cellId], 0.f);
#else
    // LOCAL
    size_t id = get_local_id(1);
    size_t gid = get_group_id(1);
    size_t localsize = get_local_size(1);
    size_t start = gid*localsize;
    
    size_t cellOffset = colMajor(get_group_id(0), start+1, edges+1); // skip ghost column
    size_t leftOffset = colMajor(get_group_id(0), start, edges);
    size_t rightOffset = leftOffset+1;
    
    size_t num = min(localsize, edges-1-start);
    
    event_t event[6];
    event[0] = async_work_group_copy(hScratch, h+cellOffset, num, 0);
    event[1] = async_work_group_copy(hvScratch, hv+cellOffset, num, 0);
    event[2] = async_work_group_copy(hNetUpdatesLeftScratch, hNetUpdatesLeft+rightOffset, num, 0);
    event[3] = async_work_group_copy(hNetUpdatesRightScratch, hNetUpdatesRight+leftOffset, num, 0);
    event[4] = async_work_group_copy(hvNetUpdatesLeftScratch, hvNetUpdatesLeft+rightOffset, num, 0);
    event[5] = async_work_group_copy(hvNetUpdatesRightScratch, hvNetUpdatesRight+leftOffset, num, 0);
    
    wait_group_events(6, event);
    
    // make sure we stay inside bounds
    if(id < num) {
        // update heights
        hScratch[id] -= dt_dy * (hNetUpdatesRightScratch[id] + hNetUpdatesLeftScratch[id]);
        // Update momentum in x-direction
        hvScratch[id] -= dt_dy * (hvNetUpdatesRightScratch[id] + hvNetUpdatesLeftScratch[id]);
        
        // Catch negative heights
        hScratch[id] = fmax(hScratch[id], 0.f);
    }
    
    // make sure all operations on local memory have finished
    barrier(CLK_LOCAL_MEM_FENCE);
    
    event[0] = async_work_group_copy(h+cellOffset, hScratch,  num, 0);
    event[1] = async_work_group_copy(hv+cellOffset, hvScratch, num, 0);
    wait_group_events(2, event);
#endif
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
    __const unsigned int length,
    __const unsigned int stride,
    __local float* scratch)
{
    unsigned int global_id = get_global_id(0);
    unsigned int source_id = stride*global_id;
    unsigned int local_id = get_local_id(0);
    
    if(source_id < length) {
        scratch[local_id] = values[source_id];
    } else {
        scratch[local_id] = -INFINITY;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    localReduceMaximum(scratch, get_local_size(0), local_id);
    
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
    __const unsigned int length,
    __const unsigned int block,
    __const unsigned int stride)
{
    unsigned int global_id = get_global_id(0);
    
    unsigned long blockstride = (unsigned long)block * (unsigned long)stride;
    
    unsigned long start = blockstride * (unsigned long)global_id;
    unsigned long end = start+blockstride;
    
    // make sure we stay inside array bounds
    end = min(end, (unsigned long)length);
    
    float acc = -INFINITY;
    
    for(unsigned long i = start; i < end; i += stride) {
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


 
/// Write the left-most column of a buffer into a copy-column buffer that is transferred to another device
/**
 * Note that the source buffer is assumed to be in row-major order
 * 
 * @param source The source buffer
 * @param copyBuffer The column buffer
 * @param cols The number of columns of the source buffer
 */
__kernel void writeNetUpdatesEdgeCopy(__global float* source, __global float* copyBuffer, __const uint cols)
{
    size_t sourceId = rowMajor(0, get_global_id(0), (size_t)cols);
    size_t destinationId = get_global_id(0);
    
    copyBuffer[destinationId] = source[sourceId];
}

/// read the right-most column from a copy-column buffer into a destination buffer
/**
 * Note that the destination buffer is assumed to be in row-major order
 * 
 * @param source The destination buffer
 * @param copyBuffer The column buffer
 * @param cols The number of columns of the destination buffer
 */
__kernel void readNetUpdatesEdgeCopy(__global float* destination, __global float* copyBuffer, __const uint cols)
{
    size_t sourceId = get_global_id(0);
    size_t destinationId = rowMajor(cols-1, get_global_id(0), (size_t)cols);
    
    destination[destinationId] = copyBuffer[sourceId];
}
