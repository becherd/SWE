
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
