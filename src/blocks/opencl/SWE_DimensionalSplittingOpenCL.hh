#ifndef SWE_DIMENSIONALSPLITTINGOPENCL_HH_
#define SWE_DIMENSIONALSPLITTINGOPENCL_HH_

#include "OpenCLWrapper.hh"
#include "blocks/SWE_Block.hh"
#include "tools/help.hh"

//! Type to set options for kernel optimization types (e.g. memory)
typedef enum {MEM_LOCAL, MEM_GLOBAL} KernelType;

/**
 * OpenCL Dimensional Splitting Block
 *
 * The two dimensional wave propagation is split into a X and a Y-Sweep
 * where we compute the wave propagation in the x sweep first
 * and then the y sweep.
 */
class SWE_DimensionalSplittingOpenCL : public SWE_Block, public OpenCLWrapper {
protected:
    //! h variable buffers on computing device
    std::vector<cl::Buffer> hd;
    //! hu variable buffers on computing device
    std::vector<cl::Buffer> hud;
    //! hv variable buffers on computing device
    std::vector<cl::Buffer> hvd;
    //! b variable buffers on computing device
    std::vector<cl::Buffer> bd;
    
    //! internal buffers for h net updates (left) on computing device
    std::vector<cl::Buffer> hNetUpdatesLeft;
    //! internal buffers for h net updates (right) on computing device
    std::vector<cl::Buffer> hNetUpdatesRight;
    //! internal buffers for hu net updates (left) on computing device
    std::vector<cl::Buffer> huNetUpdatesLeft;
    //! internal buffers for hu net updates (right) on computing device
    std::vector<cl::Buffer> huNetUpdatesRight;
    //! internal buffers for computed wavespeeds
    std::vector<cl::Buffer> waveSpeeds;
    
    //! SubBuffer column chunk size
    unsigned int chunkSize;
    
    //! Buffer chunk sizes (start column index and length) for multiple devices
    std::vector< std::pair<size_t, size_t> > bufferChunks;    
    
    //! Number of devices that should be used
    unsigned int useDevices;
    
    //! The kernel memory type to be used
    KernelType kernelType;
    
    //! The kernel type used for reductions (e.g. maxWaveSpeed reduction)
    KernelType kernelReduceType;
    
    /// Reduce maximum value in an OpenCL buffer (overwrites the buffer!)
    /**
     * @param queue The command queue to perform the reduction on
     * @param buffer The buffer to reduce
     * @param length The length of the array
     * @param waitEvent OpenCL queue event to wait for before starting
     * @param event The event returned by the last reduce operation (should be waited on to read result)
     */
    void reduceMaximum(cl::CommandQueue &queue,
                        cl::Buffer &buffer,
                        size_t length,
                        cl::Event *waitEvent = NULL,
                        cl::Event *event = NULL);
   
    /// Create OpenCL device buffers for h, hu, hv, and b variables
    void createBuffers();
    
    /// Calculate buffer chunk sizes for splitting domain among multiple devices
    /**
     * @param cols Total number of columns (including ghosts)
     * @param deviceCount Total number of devices to be used
     */
    void calculateBufferChunks(size_t cols, size_t deviceCount);
    
    /// Sync specified OpenCL buffers from compute devices to host memory
    /**
     * @param buffers A list of device buffers and corresponding host pointers
     */
    void syncBuffersBeforeRead(std::vector< std::pair< std::vector<cl::Buffer>*, float*> > &buffers);
    
    /// Sync specified OpenCL buffers from host memory to compute devices
    /**
     * @param buffers A list of device buffers and corresponding host pointers
     */
    void syncBuffersAfterWrite(std::vector< std::pair< std::vector<cl::Buffer>*, float*> > &buffers);
    
    /// Get the properties to be used for OpenCL Command Queues (e.g. out-of-order execution)
    inline cl_command_queue_properties getCommandQueueProperties() {
        cl_command_queue_properties properties = 0;
        
        // Enable out of order execution
//#ifndef NDEBUG
//        // Enable profiling in debug mode
          properties |= CL_QUEUE_PROFILING_ENABLE;
//#endif
        return properties;
    }
    
public:    
    /// Dimensional Splitting Constructor (OpenCL)
    /**
     * @param l_nx The grid size in x-direction (excluding ghost cells)
     * @param l_ny The grid size in y-direction (excluding ghost cells)
     * @param l_dx The mesh size of the Cartesian grid in x-direction
     * @param l_dy The mesh size of the Cartesian grid in y-direction
     * @param preferredDeviceType The preferred OpenCL device type to use for computation
     * @param maxDevices Maximum number of computing devices to be used (0 = unlimited)
     * @param kernelType The kernel memory type to use (MEM_GLOBAL or MEM_LOCAL)
     * @param workGroupSize The maximum work group size to use (should be a power of two)
     */
    SWE_DimensionalSplittingOpenCL(int l_nx, int l_ny,
        float l_dx, float l_dy,
        cl_device_type preferredDeviceType = 0,
        unsigned int maxDevices = 0,
        KernelType kernelType = MEM_GLOBAL,
        size_t workGroupSize = 1024);
    
    /// Print information about OpenCL devices used
    void printDeviceInformation();
    
    /// Print information about OpenCL kernel execution and memory operations
    void printProfilingInformation();
    
    /// Set conditions according to boundary types
    /**
     * The values will be updated using an OpenCL kernel in device memory
     * to avoid a memory transfer from host to device
     */
    void setBoundaryConditions();
    
    /// Simulate a single timestep.
    /**
     * @param dt The timestep
     */
    void simulateTimestep(float dt);
    
    /// Simulate from a start to an end time
    /**
     * @param	tStart The time where the simulation is started
     * @param	tEnd The time of the next checkpoint 
     * @return	The actual end time reached
     */
    float simulate(float tStart, float tEnd);
    
    /// Compute the numerical fluxes for every edge and store the net updates in member variables.
    /**
     * First, we're computing all updates in x direction (X-Sweep) 
     * and store intermediate heights (used in the Y-Sweep) in the 
     * hStar member variable.
     * Then, we're computing all updates in y direction (Y-Sweep).
     */
    void computeNumericalFluxes();
    
    /// Update unknowns using the computed net-updates
    /**
     * Update heights and momentums using the previously computed 
     * netupdates in computeNumericalFluxes using the supplied timestep.
     *
     * @param dt The timestep
     */
    void updateUnknowns(float dt);
    
   /**
    * Update OpenCL buffers on computing device
    * after an external update of the water height (e.g. read scenario)
    */
    void synchAfterWrite();
    
    /**
     * Update OpenCL water height buffer on computing device
     * after an external update of the water height (e.g. read scenario)
     */
    void synchWaterHeightAfterWrite();

    /**
     * Update OpenCL buffers on computing device after an external
     * update of the hu and hv variables (e.g. read scenario)
     */
    void synchDischargeAfterWrite();

    /**
     * Update OpenCL bathymetry buffer on computing device
     * after an external update of the bathymetry (e.g. read scenario)
     */
    void synchBathymetryAfterWrite();
    
   /**
    * Update host-side variables from OpenCL buffers on computing device
    * before external read (e.g. write output)
    */
    void synchBeforeRead();
    
    /**
     * Update host-side water height variable from OpenCL water height
     * buffer on computing device before external read (e.g. write output)
     */
    void synchWaterHeightBeforeRead();
    
    /**
     * Update host-side water hu and hv variables from OpenCL hu and hv buffers
     * on computing device before external read (e.g. write output)
     */
    void synchDischargeBeforeRead();
    
    /**
     * Update host-side bathymetry variable from OpenCL bathymetry
     * buffer on computing device before external read (e.g. write output)
     */
    void synchBathymetryBeforeRead();
};

#endif /* SWE_DIMENSIONALSPLITTINGOPENCL_HH_ */
