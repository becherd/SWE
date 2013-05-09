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
		if (x > 97.5f && x < 297.5f && y < 95.0f && y > 170.0f)
			return 10.0f;
		return -10.0f;
    };

	 /**
     * @return Initial water height at pos
     */
    float getWaterHeight(float x, float y) { 
	if (x<=97.5f)
		return 10.0f;
	return 5.0f;
    };
	

    virtual BoundaryType getBoundaryType(BoundaryEdge edge) { 
	if (edge == BND_RIGHT)
		return OUTFLOW;
	return WALL;
 };
};




#endif /* SWE_PARTIALDAMBREAK_HH_ */
