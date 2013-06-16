
#include <cxxtest/TestSuite.h>

#define private public
#define protected private
#include "blocks/opencl/OpenCLWrapper.hh"
#include "kernels/kernels.h"

/**
 * Unit test to check OpenCL Kernels
 */
class SWE_DimensionalSplittingOpenCLKernelsTest : public CxxTest::TestSuite {
    private:
        //! OpenCL Wrapper instance to test on
        OpenCLWrapper *wrapper;
        
        void _runComputeNetUpdates(
            const char* text,
            float hLeft, float hRight,
            float huLeft, float huRight,
            float bLeft, float bRight,
            float expectedHNetUpdateLeft, float expectedHNetUpdateRight,
            float expectedHuNetUpdateLeft, float expectedHuNetUpdateRight,
            float expectedMaxWaveSpeed) {
            
            float hNetUpdateLeft, hNetUpdateRight, huNetUpdateLeft, huNetUpdateRight, maxWaveSpeed;
            
            cl::Buffer hLeftBuf(wrapper->context, CL_MEM_WRITE_ONLY, sizeof(float));
            cl::Buffer hRightBuf(wrapper->context, CL_MEM_WRITE_ONLY, sizeof(float));
            cl::Buffer huLeftBuf(wrapper->context, CL_MEM_WRITE_ONLY, sizeof(float));
            cl::Buffer huRightBuf(wrapper->context, CL_MEM_WRITE_ONLY, sizeof(float));
            cl::Buffer maxWaveBuf(wrapper->context, CL_MEM_WRITE_ONLY, sizeof(float));
            
            cl::Kernel *k = &(wrapper->kernels["computeNetUpdates"]);
            k->setArg(0, hLeft);
            k->setArg(1, hRight);
            k->setArg(2, huLeft);
            k->setArg(3, huRight);
            k->setArg(4, bLeft);
            k->setArg(5, bRight);
            k->setArg(6, hLeftBuf);
            k->setArg(7, hRightBuf);
            k->setArg(8, huLeftBuf);
            k->setArg(9, huRightBuf);
            k->setArg(10, maxWaveBuf);
            
            wrapper->queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(1), cl::NullRange);

            wrapper->queues[0].enqueueReadBuffer(hLeftBuf, CL_BLOCKING, 0, sizeof(float), &hNetUpdateLeft);
            wrapper->queues[0].enqueueReadBuffer(hRightBuf, CL_BLOCKING, 0, sizeof(float), &hNetUpdateRight);
            wrapper->queues[0].enqueueReadBuffer(huLeftBuf, CL_BLOCKING, 0, sizeof(float), &huNetUpdateLeft);
            wrapper->queues[0].enqueueReadBuffer(huRightBuf, CL_BLOCKING, 0, sizeof(float), &huNetUpdateRight);
            wrapper->queues[0].enqueueReadBuffer(maxWaveBuf, CL_BLOCKING, 0, sizeof(float), &maxWaveSpeed);
            
            float delta = 1e-3;
            
            TSM_ASSERT_DELTA(text, hNetUpdateLeft, expectedHNetUpdateLeft, delta);
            TSM_ASSERT_DELTA(text, hNetUpdateRight, expectedHNetUpdateRight, delta);
            TSM_ASSERT_DELTA(text, huNetUpdateLeft, expectedHuNetUpdateLeft, delta);
            TSM_ASSERT_DELTA(text, huNetUpdateRight, expectedHuNetUpdateRight, delta);
            TSM_ASSERT_DELTA(text, maxWaveSpeed, expectedMaxWaveSpeed, delta);
        }
        
        void _runSweep(const char* kernelName,
            int sourceCount, int updateCount,
            int kernelRangeX, int kernelRangeY,
            float* h,
            float* hu,
            float* b,
            float* expectedHNetUpdateLeft, float* expectedHNetUpdateRight,
            float* expectedHuNetUpdateLeft, float* expectedHuNetUpdateRight,
            float* expectedMaxWaveSpeed)
        {
            // h, hu and b input buffers
            cl::Buffer hBuf(wrapper->context, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), sourceCount*sizeof(float), h);
            cl::Buffer huBuf(wrapper->context, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), sourceCount*sizeof(float), hu);
            cl::Buffer bBuf(wrapper->context, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), sourceCount*sizeof(float), b);
            
            // net update and maxwave buffers
            cl::Buffer hLeftBuf(wrapper->context, CL_MEM_WRITE_ONLY, updateCount*sizeof(float));
            cl::Buffer hRightBuf(wrapper->context, CL_MEM_WRITE_ONLY, updateCount*sizeof(float));
            cl::Buffer huLeftBuf(wrapper->context, CL_MEM_WRITE_ONLY, updateCount*sizeof(float));
            cl::Buffer huRightBuf(wrapper->context, CL_MEM_WRITE_ONLY, updateCount*sizeof(float));
            cl::Buffer maxWaveBuf(wrapper->context, CL_MEM_WRITE_ONLY, updateCount*sizeof(float));
            
            cl::Kernel *k = &(wrapper->kernels[kernelName]);
            k->setArg(0, hBuf);
            k->setArg(1, huBuf);
            k->setArg(2, bBuf);
            k->setArg(3, hLeftBuf);
            k->setArg(4, hRightBuf);
            k->setArg(5, huLeftBuf);
            k->setArg(6, huRightBuf);
            k->setArg(7, maxWaveBuf);
            
            wrapper->queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(kernelRangeX,kernelRangeY), cl::NullRange);
            
            
            float   hNetUpdateLeft[updateCount], hNetUpdateRight[updateCount],
                    huNetUpdateLeft[updateCount], huNetUpdateRight[updateCount],
                    maxWaveSpeed[updateCount];
            
            wrapper->queues[0].enqueueReadBuffer(hLeftBuf, CL_BLOCKING, 0, updateCount*sizeof(float), &hNetUpdateLeft);
            wrapper->queues[0].enqueueReadBuffer(hRightBuf, CL_BLOCKING, 0, updateCount*sizeof(float), &hNetUpdateRight);
            wrapper->queues[0].enqueueReadBuffer(huLeftBuf, CL_BLOCKING, 0, updateCount*sizeof(float), &huNetUpdateLeft);
            wrapper->queues[0].enqueueReadBuffer(huRightBuf, CL_BLOCKING, 0, updateCount*sizeof(float), &huNetUpdateRight);
            wrapper->queues[0].enqueueReadBuffer(maxWaveBuf, CL_BLOCKING, 0, updateCount*sizeof(float), &maxWaveSpeed);
            
            float delta = 1e-3;
            
            for(int i = 0; i < updateCount; i++) {
                TSM_ASSERT_DELTA("h net update left", hNetUpdateLeft[i], expectedHNetUpdateLeft[i], delta);
                TSM_ASSERT_DELTA("h net update right", hNetUpdateRight[i], expectedHNetUpdateRight[i], delta);
                TSM_ASSERT_DELTA("hu net update left", huNetUpdateLeft[i], expectedHuNetUpdateLeft[i], delta);
                TSM_ASSERT_DELTA("hu net update right", huNetUpdateRight[i], expectedHuNetUpdateRight[i], delta);
                TSM_ASSERT_DELTA("max wave speed", maxWaveSpeed[i], expectedMaxWaveSpeed[i], delta);
            }
        }
        
        void _runUpdate(const char* kernelName,
            int sourceCount, int updateCount,
            int kernelRangeX, int kernelRangeY,
            float ds_dt,
            float* h, float* hu,
            float* hNetUpdateLeft, float* hNetUpdateRight,
            float* huNetUpdateLeft, float* huNetUpdateRight,
            float* expectedH, float* expectedHu)
        {
            // h, hu and b input buffers
            cl::Buffer hBuf(wrapper->context, (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR), sourceCount*sizeof(float), h);
            cl::Buffer huBuf(wrapper->context, (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR), sourceCount*sizeof(float), hu);
            
            // net update and maxwave buffers
            cl::Buffer hLeftBuf(wrapper->context, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), updateCount*sizeof(float), hNetUpdateLeft);
            cl::Buffer hRightBuf(wrapper->context, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), updateCount*sizeof(float), hNetUpdateRight);
            cl::Buffer huLeftBuf(wrapper->context, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), updateCount*sizeof(float), huNetUpdateLeft);
            cl::Buffer huRightBuf(wrapper->context, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), updateCount*sizeof(float), huNetUpdateRight);
            
            cl::Kernel *k = &(wrapper->kernels[kernelName]);
            k->setArg(0, ds_dt);
            k->setArg(1, hBuf);
            k->setArg(2, huBuf);
            k->setArg(3, hLeftBuf);
            k->setArg(4, hRightBuf);
            k->setArg(5, huLeftBuf);
            k->setArg(6, huRightBuf);
            
            wrapper->queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(kernelRangeX,kernelRangeY), cl::NullRange);
            
            
            float hResult[sourceCount], huResult[sourceCount];
            
            wrapper->queues[0].enqueueReadBuffer(hBuf, CL_BLOCKING, 0, sourceCount*sizeof(float), hResult);
            wrapper->queues[0].enqueueReadBuffer(huBuf, CL_BLOCKING, 0, sourceCount*sizeof(float), huResult);
            
            float delta = 1e-3;
            
            for(int i = 0; i < sourceCount; i++) {
                
                if(expectedH[i] != -INFINITY)
                    TSM_ASSERT_DELTA("h", hResult[i], expectedH[i], delta);
                if(expectedHu[i] != -INFINITY)
                    TSM_ASSERT_DELTA("hu", huResult[i], expectedHu[i], delta);
            }
        }
        
    public:
        void setUp() {
            wrapper = new OpenCLWrapper();
            
            cl::Program::Sources kernelSources;
            getKernelSources(kernelSources);
            wrapper->buildProgram(kernelSources);
        }
        
        void tearDown() {
            delete wrapper;
        }
        
        /// Test kernel function calculating the X-Sweep net updates
        void testXSweep() {            
            int x = 4;
            int y = 4;
    
            int srcCount = x*y;
            int updCount = (x-1)*y;
            
            // Note that all values are stored in column major order (as in Float2D)
            float h[] = {
                15.0, 10.0, 12.0, 11.0,
                12.0, 11.0, 13.0,  9.0,
                13.0,  7.0, 10.5,  8.0,
                12.5,  8.5,  9.0,  10.0
            };
    
            float hu[] = {
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            };
    
            float b[] = {
                -5.0, -2.0, -3.0, -5.0,
                -5.6, -1.5, -2.7, -3.4,
                -4.3, -2.2, -4.0, -2.3,
                -6.6, -3.1, -0.5, -1.0,
            };
            
            float expectedHNetUpdateLeft[] = {
                20.7145, -7.61185, -7.19785, 1.98091, 
                -12.7347, 22.0812, 20.3989, -0.456578, 
                15.6573, -2.61581, -9.77995, -15.5039
            };
            float expectedHNetUpdateRight[] = {
                -20.7145, 7.61185, 7.19785, -1.98091, 
                12.7347, -22.0812, -20.3989, 0.456578, 
                -15.6573, 2.61581, 9.77995, 15.5039 
            };
            float expectedHuNetUpdateLeft[] = {
                -238.383, 77.2538, 79.7063, -19.62, 
                141.019, -207.481, -219.008, 4.16926, 
                -175.109, 22.8083, 95.6475, 145.679
            };
            float expectedHuNetUpdateRight[] = {
                -238.383, 77.2538, 79.7063, -19.62, 
                141.019, -207.481, -219.008, 4.16926, 
                -175.109, 22.8083, 95.6475, 145.679 
            };
            float expectedMaxWaveSpeed[] = {
                11.508, 10.1491, 11.0736, 9.90454, 
                11.0736, 9.39628, 10.7363, 9.13154, 
                11.1838, 8.71938, 9.77995, 9.39628
            };
    
            _runSweep(  "dimensionalSplitting_XSweep_netUpdates",
                        srcCount, updCount, x-1, y,
                        h, hu, b,
                        expectedHNetUpdateLeft, expectedHNetUpdateRight,
                        expectedHuNetUpdateLeft, expectedHuNetUpdateRight,
                        expectedMaxWaveSpeed);
        }
        
        /// Test kernel function calculating the Y-Sweep net updates
        void testYSweep() {            
            int x = 4;
            int y = 4;
    
            int srcCount = x*y;
            int updCount = x*(y-1);
            
            // Note that all values are stored in column major order (as in Float2D)
            float h[] = {
                15.0, 10.0, 12.0, 11.0,
                12.0, 11.0, 13.0,  9.0,
                13.0,  7.0, 10.5,  8.0,
                12.5,  8.5,  9.0,  10.0
            };
    
            float hu[] = {
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            };
    
            float b[] = {
                -5.0, -2.0, -3.0, -5.0,
                -5.6, -1.5, -2.7, -3.4,
                -4.3, -2.2, -4.0, -2.3,
                -6.6, -3.1, -0.5, -1.0,
            };
            
            float expectedHNetUpdateLeft[] = {
                11.0736, -5.19399, 15.9322, -16.4632, 
                -4.33996, 24.4117, 19.3139, -7.87512, 
                3.81036, 2.53728, -14.3605, -2.41344
            };
            float expectedHNetUpdateRight[] = {
                -11.0736, 5.19399, -15.9322, 16.4632, 
                4.33996, -24.4117, -19.3139, 7.87512, 
                -3.81036, -2.53728, 14.3605, 2.41344 
            };
            float expectedHuNetUpdateLeft[] = {
                -122.625, 53.955, -169.223, 174.863, 
                47.088, -253.588, -191.295, 72.9619, 
                -36.297, -25.7513, 133.048, 23.2988
            };
            float expectedHuNetUpdateRight[] = {
                -122.625, 53.955, -169.223, 174.863, 
                47.088, -253.588, -191.295, 72.9619, 
                -36.297, -25.7513, 133.048, 23.2988 
            };
            float expectedMaxWaveSpeed[] = {
                11.0736, 10.388, 10.6214, 10.6214, 
                10.8499, 10.388, 9.90454, 9.26485, 
                9.52589, 10.1491, 9.26485, 9.65376
            };
    
            _runSweep(  "dimensionalSplitting_YSweep_netUpdates",
                        srcCount, updCount, x, y-1,
                        h, hu, b,
                        expectedHNetUpdateLeft, expectedHNetUpdateRight,
                        expectedHuNetUpdateLeft, expectedHuNetUpdateRight,
                        expectedMaxWaveSpeed);
        }
        
        void testXUpdateUnknowns() {
            int x = 4;
            int y = 4;
            
            int srcCount = x*y;
            int updCount = (x-1)*y;
            
            // Note that all values are stored in column major order (as in Float2D)
            float h[] = {
                15.0, 10.0, 12.0, 11.0,
                12.0, 11.0, 13.0,  9.0,
                13.0,  7.0, 10.5,  8.0,
                12.5,  8.5,  9.0,  10.0
            };
    
            float hu[] = {
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            };
            
            float hNetUpdateLeft[] = {
                20.7145, -7.61185, -7.19785, 1.98091, 
                -12.7347, 22.0812, 20.3989, -0.456578, 
                15.6573, -2.61581, -9.77995, -15.5039
            };
            float hNetUpdateRight[] = {
                -20.7145, 7.61185, 7.19785, -1.98091, 
                12.7347, -22.0812, -20.3989, 0.456578, 
                -15.6573, 2.61581, 9.77995, 15.5039 
            };
            float huNetUpdateLeft[] = {
                -238.383, 77.2538, 79.7063, -19.62, 
                141.019, -207.481, -219.008, 4.16926, 
                -175.109, 22.8083, 95.6475, 145.679
            };
            float huNetUpdateRight[] = {
                -238.383, 77.2538, 79.7063, -19.62, 
                141.019, -207.481, -219.008, 4.16926, 
                -175.109, 22.8083, 95.6475, 145.679 
            };
            
            // -INFINITY implies "don't care"
            float expectedH[] = {
                -INFINITY, -INFINITY, -INFINITY, -INFINITY, 
                28.7246, 0.0, 0.0, 10.2187, 
                0.0, 19.3485, 25.5894, 15.5236,
                -INFINITY, -INFINITY, -INFINITY, -INFINITY
            };
            
            // -INFINITY implies "don't care"
            float expectedHu[] = {
                -INFINITY, -INFINITY, -INFINITY, -INFINITY, 
                48.6821, 65.1139, 69.651, 7.72536, 
                17.0449, 92.3366, 61.6804, -74.9239,
                -INFINITY, -INFINITY, -INFINITY, -INFINITY
            };
            
            float dt_dx = 0.5;
            
            _runUpdate( "dimensionalSplitting_XSweep_updateUnknowns",
                    srcCount, updCount, x-2, y,
                    dt_dx,
                    h, hu,
                    hNetUpdateLeft, hNetUpdateRight,
                    huNetUpdateLeft, huNetUpdateRight,
                    expectedH, expectedHu
                );
        }
        
        void testYUpdateUnknowns() {
            int x = 4;
            int y = 4;
            
            int srcCount = x*y;
            int updCount = x*(y-1);
            
            // Note that all values are stored in column major order (as in Float2D)
            float h[] = {
                15.0, 10.0, 12.0, 11.0,
                12.0, 11.0, 13.0,  9.0,
                13.0,  7.0, 10.5,  8.0,
                12.5,  8.5,  9.0,  10.0
            };
    
            float hu[] = {
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            };
            
            float hNetUpdateLeft[] = {
                11.0736, -5.19399, 15.9322, -16.4632, 
                -4.33996, 24.4117, 19.3139, -7.87512, 
                3.81036, 2.53728, -14.3605, -2.41344
            };
            float hNetUpdateRight[] = {
                -11.0736, 5.19399, -15.9322, 16.4632, 
                4.33996, -24.4117, -19.3139, 7.87512, 
                -3.81036, -2.53728, 14.3605, 2.41344 
            };
            float huNetUpdateLeft[] = {
                -122.625, 53.955, -169.223, 174.863, 
                47.088, -253.588, -191.295, 72.9619, 
                -36.297, -25.7513, 133.048, 23.2988
            };
            float huNetUpdateRight[] = {
                -122.625, 53.955, -169.223, 174.863, 
                47.088, -253.588, -191.295, 72.9619, 
                -36.297, -25.7513, 133.048, 23.2988 
            };
            
            // -INFINITY implies "don't care"
            float expectedH[] = {
                -INFINITY, 18.1338, 1.43693, -INFINITY, 
                -INFINITY, 0.0, 15.5489, -INFINITY, 
                -INFINITY, 7.63654, 18.9489, -INFINITY, 
                -INFINITY, 8.5, 9, -INFINITY
            };
            
            // -INFINITY implies "don't care"
            float expectedHu[] = {
                -INFINITY, 34.335, 57.6338, -INFINITY, 
                -INFINITY, 103.25, 222.442, -INFINITY, 
                -INFINITY, 31.0241, -53.6484, -INFINITY, 
                -INFINITY, 0, 0, -INFINITY
            };
            
            float dt_dx = 0.5;
            
            _runUpdate( "dimensionalSplitting_YSweep_updateUnknowns",
                    srcCount, updCount, x, y-2,
                    dt_dx,
                    h, hu,
                    hNetUpdateLeft, hNetUpdateRight,
                    huNetUpdateLeft, huNetUpdateRight,
                    expectedH, expectedHu
                );
        }
        
        /// Test the computeNetUpdates kernel
        void testComputeNetUpdates() {
            _runComputeNetUpdates(
                "Regular",
                10.0, 12.5, 5.0, -3.5, -50.0, -50.0, // h left/right, hu left/right, bathymetry left/right
                -17.34505918808096570196, 8.84505918808096570196, // h update left/right
                180.68503796971846054652, 93.70121203028153945348, // hu update left/right
                10.59362182183554874211 // wave speed
            );
                
            _runComputeNetUpdates(
                "SupersonicRight",
                4.5, 2.5, 20.0, 22.5, -50.0, -50.0, // h left/right, hu left/right, bathymetry left/right
                0.0, 2.5, // h update left/right
                0.0, 44.94111111111111111111, // hu update left/right
                12.24950641851166448956 // wave speed
            );
                
            _runComputeNetUpdates(
                "SupersonicLeft",
                7.5, 1.4, -27.3, -25.2, -50.0, -50.0,
                2.1, 0.0,
                87.93555, 0.0,
                14.57956803440405980804
            );
                
            _runComputeNetUpdates(
                "Steady",
                12.0, 12.0, 14.0, 14.0, -50.0, -50.0,
                0.0, 0.0,
                0.0, 0.0,
                12.0165514586817413307
            );
            
            float h = 5.0;
            float hu = h * sqrt(9.81 * h);
            _runComputeNetUpdates(
                "LambdaZero",
                h, h, hu, hu, -50.0, -50.0,
                0.0, 0.0,
                0.0, 0.0,
                14.00714103591450242095
            );
            
            _runComputeNetUpdates(
                "ZeroHeightLeft",
                0.0, 5.0, 0.0, 2.5, -50.0, -50.0,
                -11.13068051441438335285, 13.63068051441438335285,
                49.55681948558561664715, 74.31818051441438335285,
                5.45227220576575334114
            );
            
            _runComputeNetUpdates(
                "ZeroHeightRight",
                5.0, 0.0, 2.5, 0.0, -50.0, -50.0,
                11.13068051441438335285, -13.63068051441438335285,
                -49.55681948558561664715, -74.31818051441438335285,
                5.45227220576575334114
            );
            
            _runComputeNetUpdates(
                "ZeroHeightBoth",
                0.0, 0.0, 2.5, 1.5, -50.0, -50.0,
                0.0, 0.0, 0.0, 0.0, 0.0
            );
            
            _runComputeNetUpdates(
                "DryLeft",
                10.0, 5.0, 10.0, -2.5, 0.0, -50.0,
                0.0, -2.5,
                0.0, -17.50892629489312802619,
                7.00357051795725121047
            );
            
            _runComputeNetUpdates(
                "DryRight",
                12.5, 5.0, 6.5, 10.0, -50.0, 1.0,
                -6.5, 0.0,
                71.97851241863782780576, 0.0,
                11.07361729517505043166
            );
            _runComputeNetUpdates(
                "DryBoth",
                4.5, 3.5, 2.5, 1.5, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0
            );
        }
};
