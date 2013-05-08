#ifndef SWE_DIMENSIONALSPLITTING_HH_
#define SWE_DIMENSIONALSPLITTING_HH_

#include "blocks/SWE_Block.hh"
#include "tools/help.hh"
#include "solvers/FWave.cpp"
#include "types.h"

class SWE_DimensionalSplitting : public SWE_Block {
private:

	solver::FWave<float> m_solver;

	Float2D m_hNetUpdatesLeft;
	Float2D m_hNetUpdatesRight;
	Float2D m_huNetUpdatesLeft;
	Float2D m_huNetUpdatesRight;
	Float2D m_hNetUpdatesBelow;
	Float2D m_hNetUpdatesAbove;
	Float2D m_hvNetUpdatesBelow;
	Float2D m_hvNetUpdatesAbove;

	Float2D m_hStar;
	Float2D m_huStar;
	
public:
    SWE_DimensionalSplitting(int l_nx, int l_ny,
        float l_dx, float l_dy);

    
    virtual void simulateTimestep(float dt) = 0;
    virtual float simulate(float tStart, float tEnd) = 0;
    virtual void computeNumericalFluxes() = 0;
    virtual void updateUnknowns(float dt) = 0;
    
};

#endif /* SWE_DIMENSIONALSPLITTING_HH_ */
