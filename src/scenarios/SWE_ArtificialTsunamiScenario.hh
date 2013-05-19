#ifndef __SWE_ARTIFICIALTSUNAMISCENARIO_HH
#define __SWE_ARTIFICIALTSUNAMISCENARIO_HH

#include "SWE_Scenario.hh"

#define _USE_MATH_DEFINES
#include <cmath>

class SWE_ArtificialTsunamiScenario : public SWE_Scenario {

  private:
	float getDisplacement(float x, float y){
		x=x-5000.0f;
		y=y-5000.0f;
		if (x >= -500.0f && x <= 500.0f && y >= -500.0f && y <= 500.0f)
			return 5.0f*sin(((x/500.0f)+1.0f)*M_PI)*((-1)*(y/500.0f)*(y/500.0f)+1.0f);
		return 0;
	}

	float getBathymetrybeforeEarthquake(float x, float y) {
		return -100.0f;
    };

  public:

    /**
     * @return bathymetry at pos
     */
    float getBathymetry(float x, float y) {
		return getBathymetrybeforeEarthquake(x,y) + getDisplacement(x,y);
    };

	 /**
     * @return Initial water height at pos
     */
    float getWaterHeight(float x, float y) { 
        return -getBathymetrybeforeEarthquake(x, y);
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
		if (edge == BND_RIGHT)
			return OUTFLOW;
		return WALL;
	};
    
    /** Get the boundary positions
     *
     * @param i_edge which edge
     * @return value in the corresponding dimension
     */
    float getBoundaryPos(BoundaryEdge i_edge) {
       if ( i_edge == BND_LEFT )
         return 0.0f;
       else if ( i_edge == BND_RIGHT)
         return 10000.0f;
       else if ( i_edge == BND_BOTTOM )
         return 0.0f;
       else
         return 10000.0f; 
    };
};

#endif /* SWE_ARTIFICIALTSUNAMISCENARIO_HH_ */
