
#include <cxxtest/TestSuite.h>

#define private public
#define protected public

#include "blocks/opencl/SWE_DimensionalSplittingOpenCL.hh"

// #include "DamBreak1DTestScenario.hh"

/**
 * Unit test to check SWE_DimensionalSplittingOpenCL
 */
class SWE_DimensionalSplittingOpenCLTest : public CxxTest::TestSuite {
private:
    //! SWE_DimensionalSplittingOpenCL instance to test on
    SWE_DimensionalSplittingOpenCL *block;
    
public:
    
    /// Test maximum reduction of an array (may yield different results on GPU and CPU hardware!)
    void testReduceMaximum() {
        
        block = new SWE_DimensionalSplittingOpenCL(10, 10, 1.0, 1.0);
        
        // testing array
        unsigned int size = 73*16+3;
        
        float values[size];
        
        // actual maximum value
        float max = -INFINITY;
        
        // init random seed
        srand((unsigned)time(0));
        
        // fill values array with random values
        for(unsigned int i = 0; i < size; i++) {
            float f = (rand() % 100) * ((float)rand()/(float)RAND_MAX);
            
            max = std::max(f, max);
            values[i] = f;
        }
        
        cl::Buffer valuesBuf(block->context, (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR), size*sizeof(float), values);
        
        float result = block->reduceMaximum(block->queues[0], valuesBuf, size);
        
        TS_ASSERT_EQUALS(result, max);
    }
};
