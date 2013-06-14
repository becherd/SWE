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
};

#endif /* SWE_DIMENSIONALSPLITTINGOPENCL_HH_ */
