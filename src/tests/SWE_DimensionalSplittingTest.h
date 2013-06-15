
#include <cxxtest/TestSuite.h>
#include "blocks/SWE_DimensionalSplitting.hh"

#define private public
#define protected public

#include "DamBreak1DTestScenario.hh"

/**
 * Unit test to check SWE_DimensionalSplitting against a 1D solution by simulating a 1D
 * scenario in two dimensions
 */
class SWE_DimensionalSplittingTest : public CxxTest::TestSuite {
    private:
        /** tolerance for assertions */
        const static float TOLERANCE = 1e-5;
        /** relative tolerance for assertions */
        const static float REL_TOLERANCE = 0.025;
        
        /** Number of cells */
        const static int SIZE = 50;
        
        /** Number of timesteps to compute */
        const static unsigned int TIMESTEPS = 50;
        
        /**
         * Simulate a one dimensional DamBreak in two dimensions
         * and check the results
         * @param dir The direction of the dambreak (1 for X, 0 for Y)
         */
        void testDamBreak(unsigned int dir) {
            // Init dimsplitting
            SWE_DimensionalSplitting dimensionalSplitting(SIZE, SIZE, 1.f, 1.f);
            
            // Init testing scenario
            DamBreak1DTestScenario scenario(dir);
            
            // Init with test scenario
            dimensionalSplitting.initScenario(0.f, 0.f, scenario);
            
            // passed simulation time
            float t = 0.0;
            
            for(unsigned int step = 0; step < TIMESTEPS; step++) {
                
                // set values in ghost cells:
                dimensionalSplitting.setGhostLayer();
                
                // compute numerical flux on each edge
                dimensionalSplitting.computeNumericalFluxes();
            
                // update the cell values with maximum timestep
                t += dimensionalSplitting.getMaxTimestep();
                dimensionalSplitting.updateUnknowns(dimensionalSplitting.getMaxTimestep());
                
                // Cells have been updates, check the results
                if(dir == DamBreak1DTestScenario::DIR_X) {
                    // DamBreak is in X direction
                    // all values in one column should have (roughly) the same value, since
                    // we're x direction only
                    for(int i = 1; i <= SIZE; i++) {
                        for(int j = 1; j < SIZE; j++) {
                            TS_ASSERT_DELTA(
                                dimensionalSplitting.getWaterHeight()[i][j],
                                dimensionalSplitting.getWaterHeight()[i][j+1],
                                TOLERANCE);
                        }
                    }
                } else {
                    // DamBreak is in Y direction
                    // all values in one row should have (roughly) the same value, since
                    // we're y direction only
                    for(int j = 1; j <= SIZE; j++) {
                        for(int i = 1; i < SIZE; i++) {
                            TS_ASSERT_DELTA(
                                dimensionalSplitting.getWaterHeight()[i][j],
                                dimensionalSplitting.getWaterHeight()[i+1][j],
                                TOLERANCE);
                        }
                    }
                }
                
                
                // Cross-check the values with the results from SWE1D
                // Note that we have to check the timecode, since same number of 
                // timesteps does not imply that the same time has passed (due
                // to the slightly pessimistic CFL criterion)
                
                // Check if we have exceeded the pre-computed cross-checked time from SWE1D
                // If so, skip the tests, since they will be useless anyways
                if(t > DamBreak1DTestScenario::checkTimecodes[TIMESTEPS-1]) {
                    TS_WARN("Exceeded cross-check simulation time");
                    break;
                } else {
                    // find the nearest timestep index (in terms of time)
                    unsigned int index = 0;
                    float current = DamBreak1DTestScenario::checkTimecodes[index];
                    float next = DamBreak1DTestScenario::checkTimecodes[index+1];
                    while(index < TIMESTEPS-1 && std::fabs(current-t) > std::fabs(next-t)) {
                        index++;
                        current = DamBreak1DTestScenario::checkTimecodes[index];
                        next = DamBreak1DTestScenario::checkTimecodes[index+1];
                    }
                    
                    // compare each cell of the 2D result to the "nearest" (in terms of time) 1D result
                    for(int i = 1; i <= SIZE; i++) {
                        float height;
                        if(dir == DamBreak1DTestScenario::DIR_X) {
                            height = dimensionalSplitting.getWaterHeight()[i][1];
                        } else {
                            height = dimensionalSplitting.getWaterHeight()[1][i];
                        }
                        float check = DamBreak1DTestScenario::check[index][i-1];
                        TS_ASSERT_DELTA(
                            (height - check) / check,
                            0.0,
                            REL_TOLERANCE);
                    }

                }
            }
        }
    public:
        /// Simulate the 1D DamBreak in Y direction
        void testDamBreakY() {
            testDamBreak(DamBreak1DTestScenario::DIR_Y);
        }
        /// Simulate the 1D DamBreak in X direction
        void testDamBreakX() {
            testDamBreak(DamBreak1DTestScenario::DIR_X);
        }
};
