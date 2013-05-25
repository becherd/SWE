
#include <cxxtest/TestSuite.h>

#define private public
#define protected public

#include "scenarios/SWE_CheckpointTsunamiScenario.hh"

/**
 * Unit test to check correct behavior of SWE_CheckpointTsunamiScenario reading
 * data from NetCDF checkpoint file
 */
class SWE_CheckpointTsunamiScenarioTest : public CxxTest::TestSuite {
    private:
        //! The scenario instance to test on
        SWE_CheckpointTsunamiScenario *scenario;
        
        //! numerical tolerance for assertions
        static const double TOLERANCE = 1e-5;
        
    public:

        /// Set Up called before each test case (create scenario)
        void setUp() {
            // Create scenario
            // Note: CHECKPOINT_FILE env variables must be passed when compiling the tests
            scenario = new SWE_CheckpointTsunamiScenario(std::string(CHECKPOINT_FILE));
        }
        
        /// Tear Down called after each test case (delete scenario object)
        void tearDown() {
            delete scenario;
        }
        
        /// Test opening input files and reading attributes
        /**
         * Test correctness of
         *     - dimension and variable IDs
         *     - dimension lengths
         *     - dimension minimum, maximum and step size
         */
        void testLoadInputFiles() {
            // loadInputFiles is called by the constructor
            
            // Check Dimension and variable identifiers
            TSM_ASSERT_EQUALS("time-Dimension ID", scenario->time_id, 0);
            TSM_ASSERT_EQUALS("x-Dimension ID", scenario->x_id, 1);
            TSM_ASSERT_EQUALS("y-Dimension ID", scenario->y_id, 2);
            TSM_ASSERT_EQUALS("h-Varible ID", scenario->h_id, 3);
            TSM_ASSERT_EQUALS("hu-Varible ID", scenario->hu_id, 4);
            TSM_ASSERT_EQUALS("hv-Varible ID", scenario->hv_id, 5);
            TSM_ASSERT_EQUALS("b-Varible ID", scenario->b_id, 6);
            
            // Check dimension sizes
            TSM_ASSERT_EQUALS("time-Dimension Length", scenario->time_len, 5);
            TSM_ASSERT_EQUALS("x-Dimension Length", scenario->x_len, 80);
            TSM_ASSERT_EQUALS("y-Dimension Length", scenario->y_len, 40);
            
            // Check min/max/step width
            TSM_ASSERT_EQUALS("x-Dimension Minimum", scenario->x_min, 2.5);
            TSM_ASSERT_EQUALS("x-Dimension Maximum", scenario->x_max, 397.5);
            TSM_ASSERT_EQUALS("x-Dimension Step", scenario->x_step, 5.0);
            TSM_ASSERT_EQUALS("y-Dimension Minimum", scenario->y_min, 2.5);
            TSM_ASSERT_EQUALS("y-Dimension Maximum", scenario->y_max, 197.5);
            TSM_ASSERT_EQUALS("y-Dimension Step", scenario->y_step, 5.0);
        }
        
        /// Test correctness of index calculation in a single dimension (e.g. x or y dimension)
        void testGetIndex1D() {
            /// Test increasing dimension values
            size_t len = 6;
            float step, origin;
            
            // float dim[] = {-25.0, -15.0, -5.0, 5.0, 15.0, 25.0};
            step = 10.0;
            origin = -30.0;
            TSM_ASSERT_EQUALS("Round up", scenario->getIndex1D(2.5-origin, step, len), 3);
            TSM_ASSERT_EQUALS("Round down", scenario->getIndex1D(19.5-origin, step, len), 4);
            TSM_ASSERT_EQUALS("Round down", scenario->getIndex1D(-2.5-origin, step, len), 2);
            TSM_ASSERT_EQUALS("Above upper", scenario->getIndex1D(32.5-origin, step, len), 5);
            TSM_ASSERT_EQUALS("Below lower", scenario->getIndex1D(-35.5-origin, step, len), 0);
            TSM_ASSERT_EQUALS("Edge", scenario->getIndex1D(0.0-origin, step, len), 3);
        }
        
        /// Test correct positions of boundaries (= computational domain size)
        void testGetBoundaryPos() {
            TSM_ASSERT_EQUALS("Left", scenario->getBoundaryPos(BND_LEFT), 0.0);
            TSM_ASSERT_EQUALS("Right", scenario->getBoundaryPos(BND_RIGHT), 400.0);
            TSM_ASSERT_EQUALS("Bottom", scenario->getBoundaryPos(BND_BOTTOM), 0.0);
            TSM_ASSERT_EQUALS("Top", scenario->getBoundaryPos(BND_TOP), 200.0);
        }
        
        /// Test correctness of boundary type reading
        void testGetBoundaryType() {
            // Set boundary types
            TSM_ASSERT_EQUALS("Left", scenario->getBoundaryType(BND_LEFT), WALL);
            TSM_ASSERT_EQUALS("Right", scenario->getBoundaryType(BND_RIGHT), OUTFLOW);
            TSM_ASSERT_EQUALS("Bottom", scenario->getBoundaryType(BND_BOTTOM), WALL);
            TSM_ASSERT_EQUALS("Top", scenario->getBoundaryType(BND_TOP), WALL);
        }
        
        /// Test correctness of end simulation time reading
        void testEndSimulation() {
            TS_ASSERT_EQUALS(scenario->endSimulation(), 100.0);
        }
        
        /// Test correctness of reading the total number of checkpoints to be written
        void testGetNumberOfCheckpoints() {
            TS_ASSERT_EQUALS(scenario->getNumberOfCheckpoints(), 10);
        }
        
        /// Test correctness of reading the number and timestep of the last checkpoint in the file
        void testGetLastCheckpoint() {
            int checkpoint;
            float timestep;
            scenario->getLastCheckpoint(checkpoint, timestep);
            TS_ASSERT_EQUALS(checkpoint, 4);
            TS_ASSERT_DELTA(timestep, 40.00757, TOLERANCE);
        }
        
        /// Test reading of number of cells in x and y direction
        void testGetNumberOfCells() {
            int x,y;
            scenario->getNumberOfCells(x,y);
            TS_ASSERT_EQUALS(x,80);
            TS_ASSERT_EQUALS(y,40);
        }
        
        /// Test reading of bathymetry data from the checkpoint file
        void testGetBathymetry() {
            // x below, x above, y below, y above, inside
            TSM_ASSERT_DELTA("X (below)", scenario->getBathymetry(-10.0, 102.5), -10.0, TOLERANCE);
            TSM_ASSERT_DELTA("X (above)", scenario->getBathymetry(425.0, 57.5), -10.0, TOLERANCE);
            TSM_ASSERT_DELTA("Y (below)", scenario->getBathymetry(202.5, -10.0), -10.0, TOLERANCE);
            TSM_ASSERT_DELTA("Y (above)", scenario->getBathymetry(247.5, 250.0), -10.0, TOLERANCE);
            TSM_ASSERT_DELTA("Inside", scenario->getBathymetry(143.5, 79.5), -10.0, TOLERANCE);
        }
        
        /// Test reading of water heights from the checkpoint file
        void testGetWaterHeight() {
            TSM_ASSERT_DELTA("X (below)", scenario->getWaterHeight(-10.0, 102.5), 6.02871, TOLERANCE);
            TSM_ASSERT_DELTA("X (above)", scenario->getWaterHeight(425.0, 57.5), 6.40594, TOLERANCE);
            TSM_ASSERT_DELTA("Y (below)", scenario->getWaterHeight(202.5, -10.0), 6.32772, TOLERANCE);
            TSM_ASSERT_DELTA("Y (above)", scenario->getWaterHeight(247.5, 250.0), 5.55896, TOLERANCE);
            TSM_ASSERT_DELTA("Inside", scenario->getWaterHeight(143.5, 79.5), 5.85992, TOLERANCE);
        }
        
        /// Test reading of horizontal velocity from the checkpoint file
        void testGetVeloc_u() {
            TSM_ASSERT_DELTA("X (below)", scenario->getVeloc_u(-10.0, 102.5), 0.06399229685952716253, TOLERANCE);
            TSM_ASSERT_DELTA("X (above)", scenario->getVeloc_u(425.0, 57.5), 1.55515505921067009682, TOLERANCE);
            TSM_ASSERT_DELTA("Y (below)", scenario->getVeloc_u(202.5, -10.0), 0.39563697508739324749, TOLERANCE);
            TSM_ASSERT_DELTA("Y (above)", scenario->getVeloc_u(247.5, 250.0), 0.31298480291277505145, TOLERANCE);
            TSM_ASSERT_DELTA("Inside", scenario->getVeloc_u(143.5, 79.5), -0.16962381738999849827, TOLERANCE);
        }
        
        /// Test reading of vertical velocity from the checkpoint file
        void testGetVeloc_v() {
            TSM_ASSERT_DELTA("X (below)", scenario->getVeloc_v(-10.0, 102.5), -0.23309630086701798561, TOLERANCE);
            TSM_ASSERT_DELTA("X (above)", scenario->getVeloc_v(425.0, 57.5), -0.54300695916602403394, TOLERANCE);
            TSM_ASSERT_DELTA("Y (below)", scenario->getVeloc_v(202.5, -10.0), -0.00480432130372393216, TOLERANCE);
            TSM_ASSERT_DELTA("Y (above)", scenario->getVeloc_v(247.5, 250.0), 0.01913361492077654813, TOLERANCE);
            TSM_ASSERT_DELTA("Inside", scenario->getVeloc_v(143.5, 79.5), 0.4315860967385220276, TOLERANCE);
        }
};
