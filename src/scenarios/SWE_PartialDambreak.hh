#ifndef __SWE_PARTIALDAMBREAK_HH
#define __SWE_PARTIALDAMBREAK_HH

#include <cmath>

#include "SWE_Scenario.hh"

/**
 * Scenario "Partial Dambreak"
 */
class SWE_PartialDambreak : public SWE_Scenario {

  public:

    /**
     * @return bathymetry at pos
     */
    float getBathymetry(float x, float y) {
		//Position of the dam
		if (x >= 97.5f && x <= 102.5f && (y < 95.0f || y > 170.0f))
			return 1.0f;
		return -10.0f;
    };

	 /**
     * @return Initial water height at pos
     */
    float getWaterHeight(float x, float y) { 
		if (x < 97.5f)
            return 10.0f;
        else if (x <= 102.5f)
            return 0.0f;
        return 5.0f;
    };
	
	/**
     * @return time when to end simulation
     */
	virtual float endSimulation() { 
		return 20.0f; 
	};


	/**
     * @return boundary type at edge
     */
	virtual BoundaryType getBoundaryType(BoundaryEdge edge) { 
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
         return 400.0f;
       else if ( i_edge == BND_BOTTOM )
         return 0.0f;
       else
         return 200.0f; 
    };
};




#endif /* SWE_PARTIALDAMBREAK_HH_ */
