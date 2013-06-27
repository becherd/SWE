
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
        //! OpenCL Wrapper with "local memory option"
        OpenCLWrapper *wrapperLocal;
        
        //! Kernel direction (X)
        const static unsigned int DIR_X = 1;
        //! Kernel direction (Y)
        const static unsigned int DIR_Y = 2;
        
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

            wrapper->queues[0].enqueueReadBuffer(hLeftBuf, CL_TRUE, 0, sizeof(float), &hNetUpdateLeft);
            wrapper->queues[0].enqueueReadBuffer(hRightBuf, CL_TRUE, 0, sizeof(float), &hNetUpdateRight);
            wrapper->queues[0].enqueueReadBuffer(huLeftBuf, CL_TRUE, 0, sizeof(float), &huNetUpdateLeft);
            wrapper->queues[0].enqueueReadBuffer(huRightBuf, CL_TRUE, 0, sizeof(float), &huNetUpdateRight);
            wrapper->queues[0].enqueueReadBuffer(maxWaveBuf, CL_TRUE, 0, sizeof(float), &maxWaveSpeed);
            
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
            unsigned int kernelDirection,
            float* h,
            float* hu,
            float* b,
            float* expectedHNetUpdateLeft, float* expectedHNetUpdateRight,
            float* expectedHuNetUpdateLeft, float* expectedHuNetUpdateRight,
            float* expectedMaxWaveSpeed)
        {
            // 
            // GLOBAL MEMORY
            // 
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
            
            wrapper->queues[0].enqueueReadBuffer(hLeftBuf, CL_TRUE, 0, updateCount*sizeof(float), &hNetUpdateLeft);
            wrapper->queues[0].enqueueReadBuffer(hRightBuf, CL_TRUE, 0, updateCount*sizeof(float), &hNetUpdateRight);
            wrapper->queues[0].enqueueReadBuffer(huLeftBuf, CL_TRUE, 0, updateCount*sizeof(float), &huNetUpdateLeft);
            wrapper->queues[0].enqueueReadBuffer(huRightBuf, CL_TRUE, 0, updateCount*sizeof(float), &huNetUpdateRight);
            wrapper->queues[0].enqueueReadBuffer(maxWaveBuf, CL_TRUE, 0, updateCount*sizeof(float), &maxWaveSpeed);
            
            float delta = 1e-3;
            
            for(int i = 0; i < updateCount; i++) {
                TSM_ASSERT_DELTA("[global] h net update left", hNetUpdateLeft[i], expectedHNetUpdateLeft[i], delta);
                TSM_ASSERT_DELTA("[global] h net update right", hNetUpdateRight[i], expectedHNetUpdateRight[i], delta);
                TSM_ASSERT_DELTA("[global] hu net update left", huNetUpdateLeft[i], expectedHuNetUpdateLeft[i], delta);
                TSM_ASSERT_DELTA("[global] hu net update right", huNetUpdateRight[i], expectedHuNetUpdateRight[i], delta);
                TSM_ASSERT_DELTA("[global] max wave speed", maxWaveSpeed[i], expectedMaxWaveSpeed[i], delta);
            }
            
            // 
            // LOCAL MEMORY
            // 
            // h, hu and b input buffers
            cl::Buffer hBufLocal(wrapperLocal->context, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), sourceCount*sizeof(float), h);
            cl::Buffer huBufLocal(wrapperLocal->context, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), sourceCount*sizeof(float), hu);
            cl::Buffer bBufLocal(wrapperLocal->context, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), sourceCount*sizeof(float), b);
            
            // net update and maxwave buffers
            cl::Buffer hLeftBufLocal(wrapperLocal->context, CL_MEM_WRITE_ONLY, updateCount*sizeof(float));
            cl::Buffer hRightBufLocal(wrapperLocal->context, CL_MEM_WRITE_ONLY, updateCount*sizeof(float));
            cl::Buffer huLeftBufLocal(wrapperLocal->context, CL_MEM_WRITE_ONLY, updateCount*sizeof(float));
            cl::Buffer huRightBufLocal(wrapperLocal->context, CL_MEM_WRITE_ONLY, updateCount*sizeof(float));
            cl::Buffer maxWaveBufLocal(wrapperLocal->context, CL_MEM_WRITE_ONLY, updateCount*sizeof(float));
            
            k = &(wrapperLocal->kernels[kernelName]);
            size_t groupSize = wrapperLocal->getKernelGroupSize(*k, wrapperLocal->devices[0]);
            k->setArg(0, hBufLocal);
            k->setArg(1, huBufLocal);
            k->setArg(2, bBufLocal);
            k->setArg(3, hLeftBufLocal);
            k->setArg(4, hRightBufLocal);
            k->setArg(5, huLeftBufLocal);
            k->setArg(6, huRightBufLocal);
            k->setArg(7, maxWaveBufLocal);
            k->setArg(8, cl::__local((groupSize+1)*sizeof(cl_float)));
            k->setArg(9, cl::__local((groupSize+1)*sizeof(cl_float)));
            k->setArg(10, cl::__local((groupSize+1)*sizeof(cl_float)));
            k->setArg(11, cl::__local(groupSize*sizeof(cl_float)));
            k->setArg(12, cl::__local(groupSize*sizeof(cl_float)));
            k->setArg(13, cl::__local(groupSize*sizeof(cl_float)));
            k->setArg(14, cl::__local(groupSize*sizeof(cl_float)));
            k->setArg(15, cl::__local(groupSize*sizeof(cl_float)));
            k->setArg(16, kernelRangeX);
            k->setArg(17, kernelRangeY);
            
            cl::NDRange globalRange;
            cl::NDRange localRange;
            
            if(kernelDirection == DIR_X) {
                globalRange = cl::NDRange(wrapperLocal->getKernelRange(groupSize, kernelRangeX), kernelRangeY);
                localRange = cl::NDRange(groupSize, 1);
            } else {
                globalRange = cl::NDRange(kernelRangeX, wrapperLocal->getKernelRange(groupSize, kernelRangeY));
                localRange = cl::NDRange(1, groupSize);
            }
            
            wrapperLocal->queues[0].enqueueNDRangeKernel(*k, cl::NullRange, globalRange, localRange);
            
            wrapperLocal->queues[0].enqueueReadBuffer(hLeftBufLocal, CL_TRUE, 0, updateCount*sizeof(float), &hNetUpdateLeft);
            wrapperLocal->queues[0].enqueueReadBuffer(hRightBufLocal, CL_TRUE, 0, updateCount*sizeof(float), &hNetUpdateRight);
            wrapperLocal->queues[0].enqueueReadBuffer(huLeftBufLocal, CL_TRUE, 0, updateCount*sizeof(float), &huNetUpdateLeft);
            wrapperLocal->queues[0].enqueueReadBuffer(huRightBufLocal, CL_TRUE, 0, updateCount*sizeof(float), &huNetUpdateRight);
            wrapperLocal->queues[0].enqueueReadBuffer(maxWaveBufLocal, CL_TRUE, 0, updateCount*sizeof(float), &maxWaveSpeed);
            
            for(int i = 0; i < updateCount; i++) {
                TSM_ASSERT_DELTA("[local] h net update left", hNetUpdateLeft[i], expectedHNetUpdateLeft[i], delta);
                TSM_ASSERT_DELTA("[local] h net update right", hNetUpdateRight[i], expectedHNetUpdateRight[i], delta);
                TSM_ASSERT_DELTA("[local] hu net update left", huNetUpdateLeft[i], expectedHuNetUpdateLeft[i], delta);
                TSM_ASSERT_DELTA("[local] hu net update right", huNetUpdateRight[i], expectedHuNetUpdateRight[i], delta);
                TSM_ASSERT_DELTA("[local] max wave speed", maxWaveSpeed[i], expectedMaxWaveSpeed[i], delta);
            }
        }
        
        void _runUpdate(const char* kernelName,
            int sourceCount, int updateCount,
            int kernelRangeX, int kernelRangeY,
            unsigned int kernelDirection,
            float ds_dt,
            float* h, float* hu,
            float* hNetUpdateLeft, float* hNetUpdateRight,
            float* huNetUpdateLeft, float* huNetUpdateRight,
            float* expectedH, float* expectedHu)
        {
            // GLOBAL
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
            
            wrapper->queues[0].enqueueReadBuffer(hBuf, CL_TRUE, 0, sourceCount*sizeof(float), hResult);
            wrapper->queues[0].enqueueReadBuffer(huBuf, CL_TRUE, 0, sourceCount*sizeof(float), huResult);
            
            float delta = 1e-3;
            
            for(int i = 0; i < sourceCount; i++) {
                
                if(expectedH[i] != -INFINITY)
                    TSM_ASSERT_DELTA("[global] h", hResult[i], expectedH[i], delta);
                if(expectedHu[i] != -INFINITY)
                    TSM_ASSERT_DELTA("[global] hu", huResult[i], expectedHu[i], delta);
            }
            
            // LOCAL
            
            // h, hu and b input buffers
            cl::Buffer hBufLocal(wrapperLocal->context, (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR), sourceCount*sizeof(float), h);
            cl::Buffer huBufLocal(wrapperLocal->context, (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR), sourceCount*sizeof(float), hu);
            
            // net update and maxwave buffers
            cl::Buffer hLeftBufLocal(wrapperLocal->context, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), updateCount*sizeof(float), hNetUpdateLeft);
            cl::Buffer hRightBufLocal(wrapperLocal->context, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), updateCount*sizeof(float), hNetUpdateRight);
            cl::Buffer huLeftBufLocal(wrapperLocal->context, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), updateCount*sizeof(float), huNetUpdateLeft);
            cl::Buffer huRightBufLocal(wrapperLocal->context, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), updateCount*sizeof(float), huNetUpdateRight);
            
            k = &(wrapperLocal->kernels[kernelName]);
            size_t groupSize = wrapperLocal->getKernelGroupSize(*k, wrapperLocal->devices[0]);
            k->setArg(0, ds_dt);
            k->setArg(1, hBufLocal);
            k->setArg(2, huBufLocal);
            k->setArg(3, hLeftBufLocal);
            k->setArg(4, hRightBufLocal);
            k->setArg(5, huLeftBufLocal);
            k->setArg(6, huRightBufLocal);
            k->setArg(7, cl::__local(groupSize*sizeof(cl_float)));
            k->setArg(8, cl::__local(groupSize*sizeof(cl_float)));
            k->setArg(9, cl::__local(groupSize*sizeof(cl_float)));
            k->setArg(10, cl::__local(groupSize*sizeof(cl_float)));
            k->setArg(11, cl::__local(groupSize*sizeof(cl_float)));
            k->setArg(12, cl::__local(groupSize*sizeof(cl_float)));
            if(kernelDirection == DIR_X) {
                // x-2, y
                k->setArg(13, (unsigned int)kernelRangeX);
                k->setArg(14, (unsigned int)kernelRangeY);
            } else {
                // x, y-2
                k->setArg(13, (unsigned int)kernelRangeX);
                k->setArg(14, (unsigned int)kernelRangeY+1);
            }
            
            cl::NDRange globalRange;
            cl::NDRange localRange;
            
            if(kernelDirection == DIR_X) {
                globalRange = cl::NDRange(wrapperLocal->getKernelRange(groupSize, kernelRangeX), kernelRangeY);
                localRange = cl::NDRange(groupSize, 1);
            } else {
                globalRange = cl::NDRange(kernelRangeX, wrapperLocal->getKernelRange(groupSize, kernelRangeY));
                localRange = cl::NDRange(1, groupSize);
            }
            wrapperLocal->queues[0].enqueueNDRangeKernel(*k, cl::NullRange, globalRange, localRange);
            
            
            wrapperLocal->queues[0].enqueueReadBuffer(hBufLocal, CL_TRUE, 0, sourceCount*sizeof(float), hResult);
            wrapperLocal->queues[0].enqueueReadBuffer(huBufLocal, CL_TRUE, 0, sourceCount*sizeof(float), huResult);
            
            for(int i = 0; i < sourceCount; i++) {
                
                if(expectedH[i] != -INFINITY)
                    TSM_ASSERT_DELTA("[local] h", expectedH[i], hResult[i], delta);
                if(expectedHu[i] != -INFINITY)
                    TSM_ASSERT_DELTA("[local] hu", expectedHu[i], huResult[i], delta);
            }
        }
        
    public:
        void setUp() {
            
            
            cl::Program::Sources kernelSources;
            getKernelSources(kernelSources);
            wrapper = new OpenCLWrapper();
            wrapper->buildProgram(kernelSources);
            
            wrapperLocal = new OpenCLWrapper();
            wrapperLocal->buildProgram(kernelSources, "-D MEM_LOCAL");
        }
        
        void tearDown() {
            delete wrapper;
            delete wrapperLocal;
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
                20.7145, -12.7347, 15.6573,
                -7.61185, 22.0812, -2.61581,
                -7.19785, 20.3989, -9.77995,
                1.98091, -0.456578, -15.5039
            };
            
            float expectedHNetUpdateRight[] = {
                -20.7145, 12.7347, -15.6573,
                7.61185, -22.0812, 2.61581,
                7.19785, -20.3989, 9.77995,
                -1.98091, 0.456578, 15.5039 
            };
            
            float expectedHuNetUpdateLeft[] = {
                -238.383, 141.019, -175.109,
                77.2538, -207.481, 22.8083, 
                79.7063, -219.008, 95.6475, 
                -19.62, 4.16926, 145.679
            };
            float expectedHuNetUpdateRight[] = {
                -238.383, 141.019, -175.109, 
                77.2538, -207.481, 22.8083,
                79.7063, -219.008, 95.6475, 
                -19.62, 4.16926, 145.679 
            };
            float expectedMaxWaveSpeed[] = {
                11.508, 11.0736, 11.1838, 
                10.1491, 9.39628, 8.71938, 
                11.0736, 10.7363, 9.77995,
                9.90454, 9.13154, 9.39628
            };
    
            _runSweep(  "dimensionalSplitting_XSweep_netUpdates",
                        srcCount, updCount, x-1, y, DIR_X,
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
                        srcCount, updCount, x, y-1, DIR_Y,
                        h, hu, b,
                        expectedHNetUpdateLeft, expectedHNetUpdateRight,
                        expectedHuNetUpdateLeft, expectedHuNetUpdateRight,
                        expectedMaxWaveSpeed);
        }
        
        /// Test Kernel for update of unkown values (h, hu) from calculated net updates (X direction)
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
                20.7145, -12.7347, 15.6573,
                -7.61185, 22.0812, -2.61581,
                -7.19785, 20.3989, -9.77995,
                1.98091, -0.456578, -15.5039
            };
            
            float hNetUpdateRight[] = {
                -20.7145, 12.7347, -15.6573,
                7.61185, -22.0812, 2.61581,
                7.19785, -20.3989, 9.77995,
                -1.98091, 0.456578, 15.5039 
            };
            
            float huNetUpdateLeft[] = {
                -238.383, 141.019, -175.109,
                77.2538, -207.481, 22.8083, 
                79.7063, -219.008, 95.6475, 
                -19.62, 4.16926, 145.679
            };
            float huNetUpdateRight[] = {
                -238.383, 141.019, -175.109, 
                77.2538, -207.481, 22.8083,
                79.7063, -219.008, 95.6475, 
                -19.62, 4.16926, 145.679 
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
                    srcCount, updCount, x-2, y, DIR_X,
                    dt_dx,
                    h, hu,
                    hNetUpdateLeft, hNetUpdateRight,
                    huNetUpdateLeft, huNetUpdateRight,
                    expectedH, expectedHu
                );
        }
        
        /// Test Kernel for update of unkown values (h, hv) from calculated net updates (Y direction)
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
                11.0736, -5.19399, 15.9322,
                -16.4632, -4.33996, 24.4117,
                19.3139, -7.87512, 3.81036,
                2.53728, -14.3605, -2.41344
            };
            float hNetUpdateRight[] = {
                -11.0736, 5.19399, -15.9322,
                16.4632, 4.33996, -24.4117,
                -19.3139, 7.87512, -3.81036,
                -2.53728, 14.3605, 2.41344 
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
                -INFINITY, 4.93836, 0.0, -INFINITY, 
                -INFINITY, 20.5945, 4.65726, -INFINITY, 
                -INFINITY, 16.9489, 3.02646, -INFINITY
            };
            
            // -INFINITY implies "don't care"
            float expectedHu[] = {
                -INFINITY, 34.335, 57.6338, -INFINITY, 
                -INFINITY, -110.976, 103.25, -INFINITY, 
                -INFINITY, 59.1666, -18.3324, -INFINITY, 
                -INFINITY, -53.6484, -78.1734, -INFINITY
            };
            
            float dt_dx = 0.5;
            
            _runUpdate( "dimensionalSplitting_YSweep_updateUnknowns",
                    srcCount, updCount, x, y-2, DIR_Y,
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
        
        /// Test Kernel for reduction of maximum in an array (GPU version)
        void testReduceMaximum() {
            
            // testing array
            unsigned int size = 73*16+3;
            unsigned int workGroup = 16; // working group size
            unsigned int groupCount = (unsigned int) std::ceil((float)size/workGroup);
            unsigned int globalSize = workGroup*groupCount;
            
            float values[size];
            float values2[size];
            
            // actual maximum value
            float max[groupCount];
            for(unsigned int i = 0; i < groupCount; i++)
                max[i] = -INFINITY;
            
            // init random seed
            srand((unsigned)time(0));
            
            // fill values array with random values
            for(unsigned int i = 0; i < size; i++) {
                float f = (rand() % 100) * ((float)rand()/(float)RAND_MAX);
                
                unsigned int group = i/workGroup;
                max[group] = std::max(f, max[group]);
                values[i] = f;
            }
            
            cl::Buffer valuesBuf(wrapper->context, (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR), size*sizeof(float), values);
        
            cl::Kernel *k = &(wrapper->kernels["reduceMaximum"]);
            k->setArg(0, valuesBuf);
            k->setArg(1, size);
            k->setArg(2, 1);
            k->setArg(3, cl::__local(sizeof(cl_float) * workGroup));
            
            if(k->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(wrapper->devices[0]) <= 1) {
                TS_SKIP("Kernel cannot be executed on this device due to maximum work group size of 1");
            } else {
                try {
                    wrapper->queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(workGroup));
                    wrapper->queues[0].enqueueReadBuffer(valuesBuf, CL_TRUE, 0, size*sizeof(float), values2);
                } catch(cl::Error &e) {
                    wrapper->handleError(e);
                }
            
                for(unsigned int i = 0; i < groupCount; i++) {
                    TS_ASSERT_EQUALS(values2[workGroup*i], max[i]);
                }
            }
        }
        
        /// Test Kernel for reduction of maximum in an array (CPU version)
        void testReduceMaximumCPU() {
            // testing array
            unsigned int size = 73*16+3;
            unsigned int workGroup = 16; // working group size
            unsigned int groupCount = (unsigned int) std::ceil((float)size/workGroup);
            
            float values[size];
            float values2[size];
            
            // actual maximum value
            float max[groupCount];
            for(unsigned int i = 0; i < groupCount; i++)
                max[i] = -INFINITY;
            
            // init random seed
            srand((unsigned)time(0));
            
            // fill values array with random values
            for(unsigned int i = 0; i < size; i++) {
                float f = (rand() % 100) * ((float)rand()/(float)RAND_MAX);
                
                unsigned int group = i/workGroup;
                max[group] = std::max(f, max[group]);
                values[i] = f;
            }
            
            cl::Buffer valuesBuf(wrapper->context, (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR), size*sizeof(float), values);
        
            cl::Kernel *k = &(wrapper->kernels["reduceMaximumCPU"]);
            k->setArg(0, valuesBuf);
            k->setArg(1, size);
            k->setArg(2, workGroup);
            k->setArg(3, 1);
            
            try {
                wrapper->queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(groupCount), cl::NullRange);
                wrapper->queues[0].enqueueReadBuffer(valuesBuf, CL_TRUE, 0, size*sizeof(float), values2);
            } catch(cl::Error &e) {
                wrapper->handleError(e);
            }
        
            for(unsigned int i = 0; i < groupCount; i++) {
                TS_ASSERT_EQUALS(values2[workGroup*i], max[i]);
            }
        }
};
