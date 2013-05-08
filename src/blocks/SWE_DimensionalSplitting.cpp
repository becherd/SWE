#ifndef SWE_DIMENSIONALSPLITTING_CPP_
#define SWE_DIMENSIONALSPLITTING_CPP_

#include "SWE_DimensionalSplitting.hh"
#include "tools/help.hh"


SWE_DimensionalSplitting::SWE_DimensionalSplitting(int l_nx, int l_ny,
        float l_dx, float l_dy):
	SWE_Block(l_nx, l_ny, l_dx, l_dy),
  	m_hNetUpdatesLeft  (nx+1, ny),
 	m_hNetUpdatesRight (nx+1, ny),
 	m_huNetUpdatesLeft (nx+1, ny),
 	m_huNetUpdatesRight(nx+1, ny),
	m_hNetUpdatesBelow (nx, ny+1),
 	m_hNetUpdatesAbove (nx, ny+1),
	m_hvNetUpdatesBelow(nx, ny+1),
	m_hvNetUpdatesAbove(nx, ny+1),

	m_hStar(nx+1, ny),
	m_huStar(nx+1, ny)
{}




void SWE_DimensionalSplitting::computeNumericalFluxes()
{
	float maxWaveSpeed = 0.f;

    // x-sweep
    for( unsigned int j = 1; i < ny+1; i++ ) {
        for(unsigned int i = 0; j < nx+1; ){
		m_solver.computeNetUpdates( m_h[i-1][j], m_h[i][j],
				m_hu[i-1][j], m_hu[i][j],
				m_b[i-1][j], m_b[i][j],
				m_hNetUpdatesLeft[i-1][j], m_hNetUpdatesRight[i][j],
				m_huNetUpdatesLeft[i-1][j], m_huNetUpdatesRight[i][j],
				maxEdgeSpeed );
		}
    }

    // y-sweep
	for( unsigned int i = 1; i < ny+1; i++ ) {
        for(unsigned int j = 0; j < nx+1; ){
		m_solver.computeNetUpdates( m_hStar[i-1][j], m_hStar[i][j],
				m_hu[i-1][j], m_hu[i][j],
				m_b[i-1][j], m_b[i][j],
				m_hNetUpdatesAbove[i][j-1], m_hNetUpdatesBelow[i][j],
				m_hvNetUpdatesAbove[i][j-1], m_hvNetUpdatesBelow[i][j],
				maxEdgeSpeed );
  		}
	}
		// Update maxWaveSpeed
		if (maxEdgeSpeed > maxWaveSpeed)
			maxWaveSpeed = maxEdgeSpeed;

	// Compute CFL condition
	T maxTimeStep = m_cellSize/maxWaveSpeed * .4f;

	return maxTimeStep;
}



void SWE_DimensionalSplitting::updateUnknowns(T dt)
{
	// Loop over all cells
	//x-sweep: Q*
	for (unsigned int j = 1; i < m_size+1; i++) {
		for (unsigned int i = 1; i < m_size+1; i++) {
   	    	m_hStar[i][j] =  m_h[i][j] - dt/m_cellSize * (m_hNetUpdatesRight[i-1][j] + m_hNetUpdatesLeft[i][j]);
   	    	m_huStar[i][j] = m_hv[i][j] - dt/m_cellSize * (m_huNetUpdatesRight[i-1][j] + m_huNetUpdatesLeft[i][j]);
		}
	}
	
	//y-sweep: Q
	for (unsigned int i = 1; i < m_size+1; i++) {
		for (unsigned int j = 1; i < m_size+1; i++) {
       		m_h[i][j] = m_hStar[i][j] - dt/m_cellSize * (m_hNetUpdatesBelow[i][j-1] + m_hNetUpdatesAbove[i][j]);
        	m_hv[i][j] = m_huStar[i][j] - dt/m_cellSize * (m_hvNetUpdatesBelow[i][j-1] + m_hvNetUpdatesAbove[i][j]);
		}
	}
}



void SWE_DimensionalSplitting::simulateTimestep(float dt) {
	computeNumericalFluxes();
	updateUnknowns(dt);
}


#endif /* SWE_DIMENSIONALSPLITTING_CPP_ */
