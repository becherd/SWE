#ifndef __SWE_ARTIFICIALTSUNAMISCENARIO_HH
#define __SWE_ARTIFICIALTSUNAMISCENARIO_HH

#include "SWE_Scenario.hh"

#define _USE_MATH_DEFINES
#include <cmath>

/**
 * Scenario "Artificial Tsunami"
 */
class SWE_ArtificialTsunamiScenario : public SWE_Scenario {

  private:

	/**
     * @return displacement at pos
     */
	float getDisplacement(float x, float y) {
        if(std::fabs(x) > 500.0 || std::fabs(y) > 500.0)
            return 0.0;
        
        double dx = sin( (x / 500.0 + 1.0) * M_PI);
        double dy = 1.0 - (y*y)/(500.0*500.0);
        
        return 5.0 * dx * dy;
	}

	/**
     * @return bathymetry before earthquake at pos
     */
	float getInitialBathymetry(float x, float y) {
		return -100.0f;
    };

  public:

    /**
     * @return bathymetry at pos
     */
    float getBathymetry(float x, float y) {
		return getInitialBathymetry(x,y) + getDisplacement(x,y);
    };

	 /**
     * @return Initial water height at pos
     */
    float getWaterHeight(float x, float y) { 
        return -getInitialBathymetry(x, y);
    };
	
	/**
     * @return time when to end simulation
     */
	float endSimulation() {
		return 50.0f;
	};

   /**
    * Determines the type (e.g. reflecting wall or outflow) of a certain boundary
    *
    * @param edge The boundary edge
    * @return The type of the specified boundary (e.g. OUTFLOW or WALL)
    */
	BoundaryType getBoundaryType(BoundaryEdge edge) {
			return OUTFLOW;
	};
    
    /** Get the boundary positions
     *
     * @param i_edge which edge
     * @return value in the corresponding dimension
     */
    float getBoundaryPos(BoundaryEdge i_edge) {
       if ( i_edge == BND_LEFT )
         return -5000.0f;
       else if ( i_edge == BND_RIGHT)
         return 5000.0f;
       else if ( i_edge == BND_BOTTOM )
         return -5000.0f;
       else
         return 5000.0f; 
    };
};

#endif /* SWE_ARTIFICIALTSUNAMISCENARIO_HH_ */
