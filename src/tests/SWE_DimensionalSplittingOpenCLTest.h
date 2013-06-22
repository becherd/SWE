
#include <cxxtest/TestSuite.h>

#define private public
#define protected public

#include "blocks/opencl/SWE_DimensionalSplittingOpenCL.hh"

#include "DamBreak1DTestScenario.hh"

/**
 * Unit test to check SWE_DimensionalSplittingOpenCL
 */
class SWE_DimensionalSplittingOpenCLTest : public CxxTest::TestSuite {
private:
    //! SWE_DimensionalSplittingOpenCL instance to test on
    SWE_DimensionalSplittingOpenCL *block;
    
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
       SWE_DimensionalSplittingOpenCL dimensionalSplitting(SIZE, SIZE, 1.f, 1.f);
       
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
           
           const Float2D &h = dimensionalSplitting.getWaterHeight();
           
           // Cells have been updates, check the results
           if(dir == DamBreak1DTestScenario::DIR_X) {
               // DamBreak is in X direction
               // all values in one column should have (roughly) the same value, since
               // we're x direction only
               for(int i = 1; i <= SIZE; i++) {
                   for(int j = 1; j < SIZE; j++) {
                       TS_ASSERT_DELTA(
                           h[i][j],
                           h[i][j+1],
                           TOLERANCE);
                   }
               }
               TS_ASSERT_DELTA(DamBreak1DTestScenario::checkTimecodes[step], t, TOLERANCE);
           } else {
               // DamBreak is in Y direction
               // all values in one row should have (roughly) the same value, since
               // we're y direction only
               for(int j = 1; j <= SIZE; j++) {
                   for(int i = 1; i < SIZE; i++) {
                       TS_ASSERT_DELTA(
                           h[i][j],
                           h[i+1][j],
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
                       height = h[i][1];
                   } else {
                       height = h[1][i];
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
        
        cl::Event e;
        block->reduceMaximum(block->queues[0], valuesBuf, size, NULL, &e);
        e.wait();
        
        float result;
        block->queues[0].enqueueReadBuffer(valuesBuf, CL_TRUE, 0, sizeof(cl_float), &result);
        
        TS_ASSERT_EQUALS(result, max);
    }
    
    /// Test splitting of computational domain into buffers for multiple computing devices
    void testCalculateBufferChunks() {
        block = new SWE_DimensionalSplittingOpenCL(100, 100, 1.0, 1.0);
        // clear data (automatically calculated from constructor)
        block->bufferChunks.clear();
        
        // Test single device
        block->calculateBufferChunks(99, 1);
        TS_ASSERT_EQUALS(block->bufferChunks.size(), 1);
        TS_ASSERT_EQUALS(block->bufferChunks[0].first, 0);
        TS_ASSERT_EQUALS(block->bufferChunks[0].second, 99);
        
        delete block;
        
        block = new SWE_DimensionalSplittingOpenCL(100, 100, 1.0, 1.0);
        block->bufferChunks.clear();
        
        // Test multiple devices
        block->calculateBufferChunks(100, 3);
        TS_ASSERT_EQUALS(block->chunkSize, 34);
        TS_ASSERT_EQUALS(block->bufferChunks.size(), 3);
        TS_ASSERT_EQUALS(block->bufferChunks[0].first, 0);
        TS_ASSERT_EQUALS(block->bufferChunks[0].second, 35);
        TS_ASSERT_EQUALS(block->bufferChunks[1].first, 34);
        TS_ASSERT_EQUALS(block->bufferChunks[1].second, 35);
        TS_ASSERT_EQUALS(block->bufferChunks[2].first, 68);
        TS_ASSERT_EQUALS(block->bufferChunks[2].second, 32);
    }
    
    /// Simulate the 1D DamBreak in Y direction
    void testDamBreakY() {
        testDamBreak(DamBreak1DTestScenario::DIR_Y);
    }
    /// Simulate the 1D DamBreak in X direction
    void testDamBreakX() {
        testDamBreak(DamBreak1DTestScenario::DIR_X);
    }
};
