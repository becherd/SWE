#ifndef SWE_DIMENSIONALSPLITTING_CPP_
#define SWE_DIMENSIONALSPLITTING_CPP_

#include <cassert>

#include "SWE_DimensionalSplitting.hh"
#include "tools/help.hh"



SWE_DimensionalSplitting::SWE_DimensionalSplitting(int l_nx, int l_ny,
    float l_dx, float l_dy):
    SWE_Block(l_nx, l_ny, l_dx, l_dy),
    hNetUpdatesLeft  (nx+1, ny+2),
    hNetUpdatesRight (nx+1, ny+2),
    huNetUpdatesLeft (nx+1, ny+2),
    huNetUpdatesRight(nx+1, ny+2),
    hNetUpdatesBelow (nx, ny+1),
    hNetUpdatesAbove (nx, ny+1),
    hvNetUpdatesBelow(nx, ny+1),
    hvNetUpdatesAbove(nx, ny+1),

    hStar(nx, ny+2)
{
}

void SWE_DimensionalSplitting::computeNumericalFluxes()
{
    float maxWaveSpeed, maxEdgeSpeed = 0.f;
    
    // X-sweep
    for(int j = 0; j < ny+2; j++) {
        for(int i = 0; i < nx+1; i++) {
            dimensionalSplittingSolver.computeNetUpdates( h[i][j], h[i+1][j],
                    hu[i][j], hu[i+1][j],
                    b[i][j], b[i+1][j],
                    hNetUpdatesLeft[i][j], hNetUpdatesRight[i][j],
                    huNetUpdatesLeft[i][j], huNetUpdatesRight[i][j],
                    maxEdgeSpeed );
            // Update maxWaveSpeed (x direction)
            if (maxEdgeSpeed > maxWaveSpeed)
                maxWaveSpeed = maxEdgeSpeed;
        }
    }
    
    assert(maxWaveSpeed > 0.0);
    
    // Compute CFL condition (slightly pessimistic)
    maxTimestep = dx/maxWaveSpeed * .4f;
    
    assert(std::isfinite(maxTimestep));
    
    // Update Q* (h*)
    for (int j = 0; j < ny+2; j++) {
        for (int i = 0; i < nx; i++) {
            hStar[i][j] =  h[i+1][j] - maxTimestep/dx * (hNetUpdatesRight[i][j] + hNetUpdatesLeft[i+1][j]);
        }
    }
    
    // Y-sweep
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny+1; j++) {
            dimensionalSplittingSolver.computeNetUpdates( hStar[i][j], hStar[i][j+1],
                    hv[i+1][j], hv[i+1][j+1],
                    b[i+1][j], b[i+1][j+1],
                    hNetUpdatesBelow[i][j], hNetUpdatesAbove[i][j],
                    hvNetUpdatesBelow[i][j], hvNetUpdatesAbove[i][j],
                    maxEdgeSpeed );
            // Update maxWaveSpeed (y direction)
            if (maxEdgeSpeed > maxWaveSpeed)
                maxWaveSpeed = maxEdgeSpeed;
        }
    }
    
  #ifndef NDEBUG
    assert(maxWaveSpeed > 0.0);
    
    // Check if the CFL condition is also satisfied for y direction
    float maxTimestepY = .5f * dy / maxWaveSpeed;
    if(maxTimestepY >= maxTimestep) {
        // OK, everything's fine
    } else {
        // Oops, CFL condition is NOT satisfied
        std::cerr << "WARNING: CFL condition is not satisfied in y-sweep: "
                  << maxTimestepY << " < " << maxTimestep << std::endl;
    }
  #endif

}

void SWE_DimensionalSplitting::updateUnknowns(float dt)
{
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            // Update heights
            h[i+1][j+1]  = hStar[i][j+1] - dt/dy * (hNetUpdatesAbove[i][j] + hNetUpdatesBelow[i][j+1]);
            // Update momentum in x-direction
            hu[i+1][j+1] -= dt/dx * (huNetUpdatesLeft[i+1][j+1] + huNetUpdatesRight[i][j+1]);
            // Update momentum in y-direction
            hv[i+1][j+1] -= dt/dy * (hvNetUpdatesBelow[i][j+1] + hvNetUpdatesAbove[i][j]);
        }
    }
}

void SWE_DimensionalSplitting::simulateTimestep(float dt)
{
    computeNumericalFluxes();
    updateUnknowns(dt);
}

float SWE_DimensionalSplitting::simulate(float tStart,float tEnd)
{
    float t = tStart;
    do {
        //set values in ghost cells
        setGhostLayer();
        
        // compute net updates for every edge
        computeNumericalFluxes();
        //execute a wave propagation time step
        updateUnknowns(maxTimestep);
        t += maxTimestep;
        
        std::cout << "Simulation at time " << t << std::endl << std::flush;
    } while(t < tEnd);

    return t;
}

#endif /* SWE_DIMENSIONALSPLITTING_CPP_ */
