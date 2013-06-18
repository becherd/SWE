#ifndef SWE_DIMENSIONALSPLITTINGOPENCL_HH_
#define SWE_DIMENSIONALSPLITTINGOPENCL_HH_

#include "OpenCLWrapper.hh"
#include "blocks/SWE_Block.hh"
#include "tools/help.hh"

/**
 * OpenCL Dimensional Splitting Block
 *
 * The two dimensional wave propagation is split into a X and a Y-Sweep
 * where we compute the wave propagation in the x sweep first
 * and then the y sweep.
 */
class SWE_DimensionalSplittingOpenCL : public SWE_Block, public OpenCLWrapper {
protected:
    //! h variable buffer on computing device
    cl::Buffer hd;
    //! hu variable buffer on computing device
    cl::Buffer hud;
    //! hv variable buffer on computing device
    cl::Buffer hvd;
    //! b variable buffer on computing device
    cl::Buffer bd;
    
    //! internal buffer for h net updates (left) on computing device
    cl::Buffer hNetUpdatesLeft;
    //! internal buffer for h net updates (right) on computing device
    cl::Buffer hNetUpdatesRight;
    //! internal buffer for hu net updates (left) on computing device
    cl::Buffer huNetUpdatesLeft;
    //! internal buffer for hu net updates (right) on computing device
    cl::Buffer huNetUpdatesRight;
    //! internal buffer for computed wavespeeds
    cl::Buffer waveSpeeds;
    
    //! Whether computing devices and host have a unified memory
    bool unifiedMemory;
    
    /// Reduce maximum value in an OpenCL buffer (overwrites the buffer!)
    /**
     * @param queue The command queue to perform the reduction on
     * @param buffer The buffer the reduce
     * @param length The length of the array
     * @return The reduced maximum value
     */
    float reduceMaximum(cl::CommandQueue &queue, cl::Buffer &buffer, unsigned int length);
   
    /// Create OpenCL device buffers for h, hu, hv, and b variables
    void createBuffers();
    
public:
    /// Dimensional Splitting Constructor (OpenCL)
    /**
     * @param l_nx The grid size in x-direction (excluding ghost cells)
     * @param l_ny The grid size in y-direction (excluding ghost cells)
     * @param l_dx The mesh size of the Cartesian grid in x-direction
     * @param l_dy The mesh size of the Cartesian grid in y-direction
     * @param preferredDeviceType The preferred OpenCL device type to use for computation
     */
    SWE_DimensionalSplittingOpenCL(int l_nx, int l_ny,
        float l_dx, float l_dy,
        cl_device_type preferredDeviceType = 0);
    
    /// Print information about OpenCL devices used
    void printDeviceInformation();
    
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
