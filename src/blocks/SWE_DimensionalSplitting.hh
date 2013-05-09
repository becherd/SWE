#ifndef SWE_DIMENSIONALSPLITTING_HH_
#define SWE_DIMENSIONALSPLITTING_HH_

#include "blocks/SWE_Block.hh"
#include "tools/help.hh"
#include "fwave_solver/FWave.hpp"

class SWE_DimensionalSplitting : public SWE_Block {
private:
    //! The solver used for local edge Riemann problems
	solver::FWave<float> dimensionalSplittingSolver;
    
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
    SWE_DimensionalSplitting(int l_nx, int l_ny,
        float l_dx, float l_dy);
    
    void simulateTimestep(float dt);
    float simulate(float tStart, float tEnd);
    void computeNumericalFluxes();
    void updateUnknowns(float dt);
    
};

#endif /* SWE_DIMENSIONALSPLITTING_HH_ */
