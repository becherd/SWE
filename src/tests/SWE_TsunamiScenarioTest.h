
#include <cxxtest/TestSuite.h>

#define private public
#define protected public

#include "scenarios/SWE_TsunamiScenario.hh"

/**
 * Unit test to check correct behavior of SWE_TsunamiScenario reading
 * bathymetry and displacement data from NetCDF files
 */
class SWE_TsunamiScenarioTest : public CxxTest::TestSuite {
    private:
        //! The scenario instance to test on
        SWE_TsunamiScenario *scenario;
        
    public:

        /// Set Up called before each test case (create scenario)
        void setUp() {
            // Create scenario
            // Note: {BATHYMETRY,DISPLACEMENT}_FILE env variables must be passed when compiling the tests
            scenario = new SWE_TsunamiScenario(std::string(BATHYMETRY_FILE), std::string(DISPLACEMENT_FILE));
        }
        
        /// Tear Down called after each test case (delete scenario object)
        void tearDown() {
            delete scenario;
        }
        
        /// Test opening input files and reading attributes
        /**
         * Test correctness of
         *     - dimension IDs
         *     - variable lengths
         *     - left, right, bottom and top boundary
         *     - step with between cells in x and y domain
         * for both bathymetry and displacement files
         */
        void testLoadInputFiles() {
            // loadInputFiles is called by the constructor
            
            // Check Dimension identifiers for bathymetry
            TSM_ASSERT_EQUALS("Bathymetry x-Dimension ID", scenario->bathymetry_x_id, 0);
            TSM_ASSERT_EQUALS("Bathymetry y-Dimension ID", scenario->bathymetry_y_id, 1);
            TSM_ASSERT_EQUALS("Bathymetry z-Dimension ID", scenario->bathymetry_z_id, 2);
            
            // Check Dimension identifiers for displacement
            TSM_ASSERT_EQUALS("Displacement x-Dimension ID", scenario->displacement_x_id, 0);
            TSM_ASSERT_EQUALS("Displacement y-Dimension ID", scenario->displacement_y_id, 1);
            TSM_ASSERT_EQUALS("Displacement z-Dimension ID", scenario->displacement_z_id, 2);
            
            // Check dimension sizes
            TSM_ASSERT_EQUALS("Bathymetry x-Dimension Length", scenario->bathymetry_x_len, 100);
            TSM_ASSERT_EQUALS("Bathymetry y-Dimension Length", scenario->bathymetry_y_len, 50);
            TSM_ASSERT_EQUALS("Displacement x-Dimension Length", scenario->displacement_x_len, 20);
            TSM_ASSERT_EQUALS("Displacement y-Dimension Length", scenario->displacement_y_len, 10);
            
            // Check min/max/step width for bathymetry
            TSM_ASSERT_EQUALS("Bathymetry x-Dimension Left", scenario->bathymetry_left, -250);
            TSM_ASSERT_EQUALS("Bathymetry x-Dimension Right", scenario->bathymetry_right, 750);
            TSM_ASSERT_EQUALS("Bathymetry x-Dimension Step", scenario->bathymetry_x_step, 10);
            TSM_ASSERT_EQUALS("Bathymetry y-Dimension Bottom", scenario->bathymetry_bottom, -1250);
            TSM_ASSERT_EQUALS("Bathymetry y-Dimension Top", scenario->bathymetry_top, 1250);
            TSM_ASSERT_EQUALS("Bathymetry y-Dimension Step", scenario->bathymetry_y_step, 50);
            
            // Check min/max/step width for displacement
            TSM_ASSERT_EQUALS("Displacement x-Dimension Left", scenario->displacement_left, 150);
            TSM_ASSERT_EQUALS("Displacement x-Dimension Right", scenario->displacement_right, 350);
            TSM_ASSERT_EQUALS("Displacement x-Dimension Step", scenario->displacement_x_step, 10);
            TSM_ASSERT_EQUALS("Displacement y-Dimension Bottom", scenario->displacement_bottom, -500);
            TSM_ASSERT_EQUALS("Displacement y-Dimension Top", scenario->displacement_top, 500);
            TSM_ASSERT_EQUALS("Displacement y-Dimension Step", scenario->displacement_y_step, 100);
        }
        
        /// Test correctness of index calculation in a single dimension (e.g. x or y dimension)
        void testGetIndex1D() {
            /// Test increasing dimension values
            size_t len = 6;
            float step, origin;
            
            // Increasing order, uniform cell spacing
            float dim1[] = {-25.0, -15.0, -5.0, 5.0, 15.0, 25.0};
            step = 10.0;
            origin = -30.0;
            TSM_ASSERT_EQUALS("[INC|UNIFORM] Round up", scenario->getIndex1D(2.5, origin, step, dim1, len), 3);
            TSM_ASSERT_EQUALS("[INC|UNIFORM] Round down", scenario->getIndex1D(19.5, origin, step, dim1, len), 4);
            TSM_ASSERT_EQUALS("[INC|UNIFORM] Round down", scenario->getIndex1D(-2.5, origin, step, dim1, len), 2);
            TSM_ASSERT_EQUALS("[INC|UNIFORM] Above upper", scenario->getIndex1D(32.5, origin, step, dim1, len), 5);
            TSM_ASSERT_EQUALS("[INC|UNIFORM] Below lower", scenario->getIndex1D(-35.5, origin, step, dim1, len), 0);
            TSM_ASSERT_EQUALS("[INC|UNIFORM] Edge", scenario->getIndex1D(0.0, origin, step, dim1, len), 3);
            
            // Decreasing order, uniform cell spacing
            float dim2[] = {25.0, 15.0, 5.0, -5.0, -15.0, -25.0};
            step = -10.0;
            origin = 30.0;
            TSM_ASSERT_EQUALS("[DEC|UNIFORM] Round up", scenario->getIndex1D(2.5, origin, step, dim2, len), 2);
            TSM_ASSERT_EQUALS("[DEC|UNIFORM] Round down", scenario->getIndex1D(-2.5, origin, step, dim2, len), 3);
            TSM_ASSERT_EQUALS("[DEC|UNIFORM] Above upper", scenario->getIndex1D(32.5, origin, step, dim2, len), 0);
            TSM_ASSERT_EQUALS("[DEC|UNIFORM] Below lower", scenario->getIndex1D(-35.5, origin, step, dim2, len), 5);
            TSM_ASSERT_EQUALS("[DEC|UNIFORM] Edge", scenario->getIndex1D(0.0, origin, step, dim2, len), 3);
            
            // Increasing order, non-uniform cell spacing
            float dim3[] = {-25.0, -10.0, -3.25, 3.25, 10.0, 25.0};
            step = 10.0;
            origin = -30.0;
            TSM_ASSERT_EQUALS("[INC|NON-UNIFORM] Round up", scenario->getIndex1D(19.5, origin, step, dim3, len), 5);
            TSM_ASSERT_EQUALS("[INC|NON-UNIFORM] Round down", scenario->getIndex1D(-19.5, origin, step, dim3, len), 0);
            TSM_ASSERT_EQUALS("[INC|NON-UNIFORM] Above upper", scenario->getIndex1D(32.5, origin, step, dim3, len), 5);
            TSM_ASSERT_EQUALS("[INC|NON-UNIFORM] Below lower", scenario->getIndex1D(-35.5, origin, step, dim3, len), 0);
            TSM_ASSERT_EQUALS("[INC|NON-UNIFORM] Edge", scenario->getIndex1D(17.5, origin, step, dim3, len), 4);
            
            // Decreasing order, non-uniform cell spacing
            float dim4[] = {25.0, 10.0, 3.25, -3.25, -10.0, -25.0};
            step = -10.0;
            origin = 30.0;
            TSM_ASSERT_EQUALS("[DEC|NON-UNIFORM] Round up", scenario->getIndex1D(19.5, origin, step, dim4, len), 0);
            TSM_ASSERT_EQUALS("[DEC|NON-UNIFORM] Round down", scenario->getIndex1D(-19.5, origin, step, dim4, len), 5);
            TSM_ASSERT_EQUALS("[DEC|NON-UNIFORM] Above upper", scenario->getIndex1D(32.5, origin, step, dim4, len), 0);
            TSM_ASSERT_EQUALS("[DEC|NON-UNIFORM] Below lower", scenario->getIndex1D(-35.5, origin, step, dim4, len), 5);
            TSM_ASSERT_EQUALS("[DEC|NON-UNIFORM] Edge", scenario->getIndex1D(17.5, origin, step, dim4, len), 1);
        }
        
        /// Test correctness of the binary index search algorithm to handle non-uniformly spaced input data cells
        void testBinaryIndexSearch() {
            float values[] = {-100.0, -90.0, -50.0, -20.0, 0.0, 30.0, 45.0, 75.0, 100.0};
            size_t len = 9;
            
            TS_ASSERT_EQUALS(scenario->binaryIndexSearch(-120.0, values, len, 0, len-1), 0);
            TS_ASSERT_EQUALS(scenario->binaryIndexSearch(120.0, values, len, 0, len-1), 8);
            TS_ASSERT_EQUALS(scenario->binaryIndexSearch(-99.0, values, len, 0, len-1), 0);
            TS_ASSERT_EQUALS(scenario->binaryIndexSearch(-91.0, values, len, 0, len-1), 1);
            TS_ASSERT_EQUALS(scenario->binaryIndexSearch(-70.0, values, len, 0, len-1), 1);
            TS_ASSERT_EQUALS(scenario->binaryIndexSearch(80.0, values, len, 0, len-1), 7);
            TS_ASSERT_EQUALS(scenario->binaryIndexSearch(20.0, values, len, 0, len-1), 5);
        }
        
        /// Test correct positions of boundaries (= computational domain size)
        void testGetBoundaryPos() {
            TSM_ASSERT_EQUALS("Left", scenario->getBoundaryPos(BND_LEFT), -250);
            TSM_ASSERT_EQUALS("Right", scenario->getBoundaryPos(BND_RIGHT), 750);
            TSM_ASSERT_EQUALS("Bottom", scenario->getBoundaryPos(BND_BOTTOM), -1250);
            TSM_ASSERT_EQUALS("Top", scenario->getBoundaryPos(BND_TOP), 1250);
        }
        
        /// Test correctness of boundary type setting and reading
        void testGetBoundaryType() {
            // Set boundary types
            BoundaryType boundaryTypes[] = {WALL, OUTFLOW, OUTFLOW, WALL};
            scenario->setBoundaryTypes(boundaryTypes);
            
            TSM_ASSERT_EQUALS("Left", scenario->getBoundaryType(BND_LEFT), WALL);
            TSM_ASSERT_EQUALS("Right", scenario->getBoundaryType(BND_RIGHT), OUTFLOW);
            TSM_ASSERT_EQUALS("Bottom", scenario->getBoundaryType(BND_BOTTOM), OUTFLOW);
            TSM_ASSERT_EQUALS("Top", scenario->getBoundaryType(BND_TOP), WALL);
        }
        
        /// Test correctnes of bathymetry data reading from file
        void testGetInitialBathymetry() {
            
            // x-Position is outside computational domain (should give nearest cell value)
            TSM_ASSERT_EQUALS("x-Pos Outside Domain (lower)", scenario->getInitialBathymetry(-500.0, 25.0), -6.125);
            TSM_ASSERT_EQUALS("x-Pos Outside Domain (upper)", scenario->getInitialBathymetry(1500.0, 25.0), 18.625);
            
            // y-Position is outside computational domain (should give nearest cell value)
            TSM_ASSERT_EQUALS("y-Pos Outside Domain (lower)", scenario->getInitialBathymetry(105.0, -2000.0), -128.625);
            TSM_ASSERT_EQUALS("y-Pos Outside Domain (upper)", scenario->getInitialBathymetry(205.0, 3000.0), 251.125);
            
            //
            // From now on, both x- and y-Pos are inside the domain
            //
            
            // Request exact position is in the dataset
            TSM_ASSERT_EQUALS("Exact Position", scenario->getInitialBathymetry(115.0, 25.0), 2.875);
            
            // Exact position is NOT in dataset, but somewhere in a cell
            TSM_ASSERT_EQUALS("Cell Position", scenario->getInitialBathymetry(-122.5, 105.0), -15.625);
            
            // Exact position is NOT in dataset, but in a edge in x-direction
            // We're using a simple 'round up' approach for now
            TSM_ASSERT_EQUALS("X-Edge Position", scenario->getInitialBathymetry(420.0, -570.0), -244.375);
            
            // Exact position is NOT in dataset, but in a edge in y-direction
            // We're using a simple 'round up' approach for now
            TSM_ASSERT_EQUALS("Y-Edge Position", scenario->getInitialBathymetry(652.5, 50.0), 49.125);
            
            // Exact position is NOT in dataset and it's on the boundary in x-direction
            TSM_ASSERT_EQUALS("Lower X-Boundary Position", scenario->getInitialBathymetry(-250.0, -70.0), 18.375);
            TSM_ASSERT_EQUALS("Upper X-Boundary Position", scenario->getInitialBathymetry(750.0, -70.0), -55.875);
            
            // Exact position is NOT in dataset and it's on the boundary in x-direction
            TSM_ASSERT_EQUALS("Lower Y-Boundary Position", scenario->getInitialBathymetry(2.0, -1250.0), -6.125);
            TSM_ASSERT_EQUALS("Upper Y-Boundary Position", scenario->getInitialBathymetry(24.0, 1250.0), 30.625);
        }
        
        /// Test correctnes of displacement data reading from file
        void testGetDisplacement() {
            
            // x-Position is outside displacement domain (should give a zero displacement)
            TSM_ASSERT_EQUALS("x-Pos Outside Displacement (lower)", scenario->getDisplacement(-500.0, 25.0), 0.0);
            TSM_ASSERT_EQUALS("x-Pos Outside Displacement (upper)", scenario->getDisplacement(1500.0, 25.0), 0.0);
            
            // y-Position is outside displacement domain (should give a zero displacement)
            TSM_ASSERT_EQUALS("y-Pos Outside Displacement (lower)", scenario->getDisplacement(105.0, -2000.0), 0.0);
            TSM_ASSERT_EQUALS("y-Pos Outside Displacement (upper)", scenario->getDisplacement(205.0, 3000.0), 0.0);
            
            //
            // From now on, both x- and y-Pos are inside the displacement domain
            //
            
            // Request exact position is in the dataset
            TSM_ASSERT_EQUALS("Exact Position", scenario->getDisplacement(175.0, 250.0), 212.5);
            
            // Exact position is NOT in dataset, but somewhere in a cell
            TSM_ASSERT_EQUALS("Cell Position", scenario->getDisplacement(187.5, 175.0), 167.5);
            
            // Exact position is NOT in dataset, but in a edge in x-direction
            // We're using a simple 'round up' approach for now
            TSM_ASSERT_EQUALS("X-Edge Position", scenario->getDisplacement(200.0, 175.0), 177.5);
            
            // Exact position is NOT in dataset, but in a edge in y-direction
            // We're using a simple 'round up' approach for now
            TSM_ASSERT_EQUALS("Y-Edge Position", scenario->getDisplacement(302.5, 200.0), 277.5);
            
            // Exact position is NOT in dataset and it's on the boundary in x-direction
            TSM_ASSERT_EQUALS("Lower X-Boundary Position", scenario->getDisplacement(150.0, -70.0), 0.0);
            TSM_ASSERT_EQUALS("Upper X-Boundary Position", scenario->getDisplacement(350.0, -70.0), 0.0);
            
            // Exact position is NOT in dataset and it's on the boundary in x-direction
            TSM_ASSERT_EQUALS("Lower Y-Boundary Position", scenario->getDisplacement(225.0, -500.0), 0.0);
            TSM_ASSERT_EQUALS("Upper Y-Boundary Position", scenario->getDisplacement(325.0, 500.0), 0.0);
        }
        
        /// Test bathymetry calculation based on read bathymetry and displacement data
        void testGetBathymetry() {
            TSM_ASSERT_EQUALS("Bathymetry > 20m", scenario->getBathymetry(155.0, 475.0), 376.125);
            TSM_ASSERT_EQUALS("0m < Bathymetry <= 20m", scenario->getBathymetry(-155.0, -50.0), 20.0);
            TSM_ASSERT_EQUALS("-20m <= Bathymetry < 0m", scenario->getBathymetry(-155.0, 50.0), -20.0);
            TSM_ASSERT_EQUALS("Bathymetry < -20m", scenario->getBathymetry(155.0, -425.0), -213.375);            
        }
        
        /// Test water height calculation based in read bathymetry data
        void testGetWaterHeight() {
            TSM_ASSERT_EQUALS("Wet Cell (Bathymetry < 20m)", scenario->getWaterHeight(-122.5, 105.0), 20.0);
            TSM_ASSERT_EQUALS("Wet Cell (Bathymetry > 20m)", scenario->getWaterHeight(-122.5, 305.0), 40.625);
            TSM_ASSERT_EQUALS("Dry Cell", scenario->getWaterHeight(92.5, 85.0), 0.0);
        }
};
