#ifndef SWE_DIMENSIONALSPLITTING_HH_
#define SWE_DIMENSIONALSPLITTING_HH_

#include "blocks/SWE_Block.hh"
#include "tools/help.hh"
#include "fwave_solver/FWave.hpp"

/**
 * Dimensional Splitting Block
 *
 * The two dimensional wave propagation is split into a X and a Y-Sweep
 * where we compute the wave propagation in the x sweep first
 * and then the  y sweep.
 */
class SWE_DimensionalSplitting : public SWE_Block {
private:
#ifndef USEOPENMP
    //! The solver used for local edge Riemann problems
	solver::FWave<float> dimensionalSplittingSolver;
#endif
        
    //! net-updates for the heights of the cells on the left sides of the vertical edges.
    Float2D hNetUpdatesLeft;
    //! net-updates for the heights of the cells on the right sides of the vertical edges.
    Float2D hNetUpdatesRight;

    //! net-updates for the x-momentums of the cells on the left sides of the vertical edges.
    Float2D huNetUpdatesLeft;
    //! net-updates for the x-momentums of the cells on the right sides of the vertical edges.
    Float2D huNetUpdatesRight;

    //! net-updates for the heights of the cells below the horizontal edges.
    Float2D hNetUpdatesBelow;
    //! net-updates for the heights of the cells above the horizontal edges.
    Float2D hNetUpdatesAbove;

    //! net-updates for the y-momentums of the cells below the horizontal edges.
    Float2D hvNetUpdatesBelow;
    //! net-updates for the y-momentums of the cells above the horizontal edges.
    Float2D hvNetUpdatesAbove;
    
    //! intermediate height of the cells after the x-sweep has been performed.
	Float2D hStar;
	
public:
    /// Dimensional Splitting Constructor
    /**
     * @param l_nx The grid size in x-direction (excluding ghost cells)
     * @param l_ny The grid size in y-direction (excluding ghost cells)
     * @param l_dx The mesh size of the Cartesian grid in x-direction
     * @param l_dy The mesh size of the Cartesian grid in y-direction
     */
    SWE_DimensionalSplitting(int l_nx, int l_ny,
        float l_dx, float l_dy);
    
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

#endif /* SWE_DIMENSIONALSPLITTING_HH_ */
