
#include <cxxtest/TestSuite.h>

#define private public
#define protected public

#include "writer/CoarseGridWrapper.hh"

/**
 * Unit test for checking the CoarseGridWrapper class to transform
 * a refined grid into a coarse grid for writing output
 */
class CoarseGridWrapperTest : public CxxTest::TestSuite {
    private:
        //! The grid wrapper instance to test on
        io::CoarseGridWrapper *wrapper;
        //! The grid wrapper instance to test on (alternative testing sequence)
        io::CoarseGridWrapper *altWrapper;
        //! The grid wrapper instance to test on (coarseness factor = 1)
        io::CoarseGridWrapper *identicalWrapper;
        
        //! The grid used in testing
        Float2D *grid;
        
        //! The boundary size (ghost cells) at left, right, bottom, top boundary
        io::BoundarySize boundary;
        
        //! number of columns in the refined grid
        static const unsigned int cols = 12;
        //! number of rows in the refined grid
        static const unsigned int rows = 18;
        //! coarseness factor
        static const float coarseness = 1.5f;
        
        //! number of columns in the refined grid (alternative testing sequence)
        static const unsigned int altCols = 15;
        //! number of rows in the refined grid (alternative testing sequence)
        static const unsigned int altRows = 13;
        //! coarseness factor (alternative testing sequence)
        static const float altCoarseness = 4.0f;
    public:

        /// Set Up called before each test case (create wrapper)
        void setUp() {
            // boundary size (ghost cells) at left, right, bottom, top boundary
            // we set crazy numbers here because we should be independent of boundary size
            boundary[0] = 1;
            boundary[1] = 4;
            boundary[2] = 3;
            boundary[3] = 2;
                        
            unsigned int totalCols = cols + boundary[0] + boundary[1];
            unsigned int totalRows = rows + boundary[2] + boundary[3];
            
            // create grid
            grid = new Float2D(totalCols, totalRows);
            
            // init grid with sample values
            for(unsigned int i = 0; i < totalCols; i++) {
                for(unsigned int j = 0; j < totalRows; j++) {
                    // In the refined grid
                    // 
                    float value = float(2*(int(i)-int(boundary[0])) + 0.5 * (int(j)-int(boundary[2])));
                    (*grid)[i][j] = value;
                }
            }
            
            // create wrapper to test on
            wrapper = new io::CoarseGridWrapper(*grid, boundary, cols, rows, coarseness);
            // create wrapper to test on (alterative testing sequence)
            altWrapper = new io::CoarseGridWrapper(*grid, boundary, altCols, altRows, altCoarseness);
            // create wrapper to test on (coarseness factor = 1)
            identicalWrapper = new io::CoarseGridWrapper(*grid, boundary, cols, rows, 1.0);
        }
        
        /// Tear Down called after each test case (delete wrapper object)
        void tearDown() {
            delete wrapper;
            delete altWrapper;
            delete identicalWrapper;
            delete grid;
        }
        
        /// Test constructor with basic setup
        void testCoarseGridWrapper() {
            TS_ASSERT_EQUALS(wrapper->stepWidthX, 1.5f);
            TS_ASSERT_EQUALS(wrapper->stepWidthY, 1.5f);
            
            TS_ASSERT_EQUALS(altWrapper->stepWidthX, 3.75f);
            TS_ASSERT_EQUALS(altWrapper->stepWidthY, 3.25f);
            
            TS_ASSERT_EQUALS(identicalWrapper->stepWidthX, 1.f);
            TS_ASSERT_EQUALS(identicalWrapper->stepWidthY, 1.f);
        }
        
        /// Test reading of coarse values averaged over all refined values in the coarse cell
        void testGetElem() {
            // 1*0 + 0.5*0,5 + 0.5*2 + 0.5^2 * 2,5 = 1.875
            // 1 + 0.5 + 0.5 + 0.25 = 2.25
            TS_ASSERT_DELTA(wrapper->getElem(0,0), 1.875/2.25, 1e-5);
            
            // 0.25 * 2.5 + 0.5 * 3 + 0.5 * 4.5 + 1 * 5 = 9.375
            // 1 + 0.5 + 0.5 + 0.25 = 2.25
            TS_ASSERT_DELTA(wrapper->getElem(1,1), 9.375/2.25, 1e-5);
            
            // 0.25 * 28 + 0.5 * 28.5 + 0.5 * 30 + 1 * 30.5 = 66.75
            // 1 + 0.5 + 0.5 + 0.25 = 2.25
            TS_ASSERT_DELTA(wrapper->getElem(7,11), 66.75/2.25, 1e-5);
            
            // test identical wrapper
            // should give exactly the same values as in the refined grid
            for(unsigned int i = 0; i < cols; i++) {
                for(unsigned int j = 0; j < rows; j++) {
                    TS_ASSERT_EQUALS(identicalWrapper->getElem(i,j), (*grid)[i+boundary[0]][j+boundary[2]]);
                }
            }
        }
        
        /// Test reading of number of coarse grid columns
        void testGetCols() {
            TS_ASSERT_EQUALS(wrapper->getCols(), 8);
            TS_ASSERT_EQUALS(altWrapper->getCols(), 4);
            TS_ASSERT_EQUALS(identicalWrapper->getCols(), cols);
        }
        
        /// Test reading of number of coarse grid rows
        void testGetRows() {
            TS_ASSERT_EQUALS(wrapper->getRows(), 12);
            TS_ASSERT_EQUALS(altWrapper->getRows(), 4);
            TS_ASSERT_EQUALS(identicalWrapper->getRows(), rows);
        }
};
