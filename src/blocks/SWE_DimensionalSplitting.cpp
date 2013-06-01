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
    
    /**
     * **X-Sweep**
     *
     * Iterate through every row (including ghost-only rows) and compute 
     * the left and right net updates for each edge. NetUpdatesLeft[i][j]
     * denotes the left-going update from cell i+1 to cell i in row j,
     * while NetUpdatesRight[i][j] denotes the right-going update from cell i
     * to cell i+1 in row j
     */
    for(int j = 0; j < ny+2; j++) {
        for(int i = 0; i < nx+1; i++) {
            dimensionalSplittingSolver.computeNetUpdates( h[i][j], h[i+1][j],
                    hu[i][j], hu[i+1][j],
                    b[i][j], b[i+1][j],
                    hNetUpdatesLeft[i][j], hNetUpdatesRight[i][j],
                    huNetUpdatesLeft[i][j], huNetUpdatesRight[i][j],
                    maxEdgeSpeed );
            
            // Update maxWaveSpeed (x direction)
            // maxWaveSpeed is likely to be greater than maxEdgeSpeed
            if (maxEdgeSpeed < maxWaveSpeed) {
                // nothing to do
            } else {
                maxWaveSpeed = maxEdgeSpeed;
            }
        }
    }
    
    assert(maxWaveSpeed > 0.0);
    
    // Compute CFL condition (slightly pessimistic)
    maxTimestep = dx/maxWaveSpeed * .4f;
    
    assert(std::isfinite(maxTimestep));
    
    /**
     * **Update intermediate heights (hStar)**
     *
     * Compute the intermediate heights resulting from the X-Sweep using
     * the left- and right-going net updates. Note that hStar does not include
     * the ghost cells at the left and right boundary of the block. Therefore
     * the cell hStar[i][j] corresponds to h[i+1][j] since indexing begins with 0
     * in hStar, similarly hStar contains two cells less than h in horizontal (x)
     * direction 
     */
    for (int j = 0; j < ny+2; j++) {
        for (int i = 0; i < nx; i++) {
            hStar[i][j] =  h[i+1][j] - maxTimestep/dx * (hNetUpdatesRight[i][j] + hNetUpdatesLeft[i+1][j]);
            
            // catch negative heights
            if(hStar[i][j] > 0.0) {
                // nothing to do
            } else {
                hStar[i][j] = 0.0;
            }
        }
    }
    
    /**
     * **Y-Sweep**
     *
     * Iterate through every column of hStar (therefore excluding the left and right
     * ghost columns) and compute all the vertical (above- and below-going) net
     * updates. NetUpdatesBelow[i][j] denotes the updates going from cell j+1 to cell j
     * in the (i+1)-th column , while NetUpdatesAbove[i][j] denotes the updates going from cell 
     * j to j+1 in the (i+1)-th column of the block
     */
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny+1; j++) {
            dimensionalSplittingSolver.computeNetUpdates( hStar[i][j], hStar[i][j+1],
                    hv[i+1][j], hv[i+1][j+1],
                    b[i+1][j], b[i+1][j+1],
                    hNetUpdatesBelow[i][j], hNetUpdatesAbove[i][j],
                    hvNetUpdatesBelow[i][j], hvNetUpdatesAbove[i][j],
                    maxEdgeSpeed );
#ifndef NDEBUG
            // Update maxWaveSpeed (y direction)
            // maxWaveSpeed is likely to be greater than maxEdgeSpeed
            if (maxEdgeSpeed < maxWaveSpeed) {
                // nothing to do
            } else {
                maxWaveSpeed = maxEdgeSpeed;
            }
#endif
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
    /**
     * Iterate through every cell inside the block (excluding ghost cells)
     * and compute the resulting height, horizontal and vertical momentum
     * using the left, right, above and below net updates
     */
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            // Update heights
            h[i+1][j+1]  = hStar[i][j+1] - dt/dy * (hNetUpdatesAbove[i][j] + hNetUpdatesBelow[i][j+1]);
            // Update momentum in x-direction
            hu[i+1][j+1] -= dt/dx * (huNetUpdatesLeft[i+1][j+1] + huNetUpdatesRight[i][j+1]);
            // Update momentum in y-direction
            hv[i+1][j+1] -= dt/dy * (hvNetUpdatesBelow[i][j+1] + hvNetUpdatesAbove[i][j]);
            
            // catch negative heights
            if(h[i+1][j+1] > 0.0) {
                // nothing to do
            } else {
                h[i+1][j+1] = 0.0;
                hu[i+1][j+1] = 0.0;
                hv[i+1][j+1] = 0.0;
            }
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
