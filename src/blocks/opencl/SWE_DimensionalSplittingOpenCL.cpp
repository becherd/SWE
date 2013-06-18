#ifndef SWE_DIMENSIONALSPLITTINGOPENCL_CPP_
#define SWE_DIMENSIONALSPLITTINGOPENCL_CPP_

#include <cassert>
#include <cmath>

#include "SWE_DimensionalSplittingOpenCL.hh"
#include "tools/help.hh"

// Note: kernels/kernels.h is created during build process
// from the OpenCL kernels
#include "kernels/kernels.h"

SWE_DimensionalSplittingOpenCL::SWE_DimensionalSplittingOpenCL(int l_nx, int l_ny,
    float l_dx, float l_dy,
    cl_device_type preferredDeviceType):
    SWE_Block(l_nx, l_ny, l_dx, l_dy),
    OpenCLWrapper(preferredDeviceType)
{
    cl::Program::Sources kernelSources;
    getKernelSources(kernelSources);
    buildProgram(kernelSources);
    
    unifiedMemory = devices[0].getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>();
    
    createBuffers();
}

void SWE_DimensionalSplittingOpenCL::printDeviceInformation()
{
    std::cout << "Using " << devices.size() << " OpenCL devices of type ";
    switch(deviceType) {
        case CL_DEVICE_TYPE_CPU:
            std::cout << "CPU";
            break;
        case CL_DEVICE_TYPE_GPU:
            std::cout << "GPU";
            break;
        case CL_DEVICE_TYPE_ACCELERATOR:
            std::cout << "ACCELERATOR";
            break;
        case CL_DEVICE_TYPE_DEFAULT:
            std::cout << "DEFAULT";
            break;
        default:
            std::cout << "UNKNOWN";
    }
    std::cout << ":" << std::endl;
    
    for(unsigned int i = 0; i < devices.size(); i++) {
        std::string deviceName, deviceVendor;
        try {
            devices[i].getInfo(CL_DEVICE_NAME, &deviceName);
            devices[i].getInfo(CL_DEVICE_VENDOR, &deviceVendor);
            std::cout << "    (" << i << ") " << deviceVendor << " " << deviceName << std::endl;
        } catch(cl::Error &e) {
            std::cerr << "Unable to query device info:" << e.what() << " (" << e.err() << ")" << std::endl;
        }
    }
    
    std::cout << std::endl;
}

float SWE_DimensionalSplittingOpenCL::reduceMaximum(cl::CommandQueue &queue, cl::Buffer &buffer, unsigned int length) {
    cl::Kernel *k;
    float result;
    
    cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
    
    if(device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU) {
        // Use CPU optimized kernel
        k = &(kernels["reduceMaximumCPU"]);
        unsigned int stride = 1;
        unsigned int block = std::min(std::max(length/1024, (unsigned int)16), (unsigned int)8192);
        
        unsigned int items = (unsigned int)ceil((float)length/(float)(block*stride));
        
        while(items > 1) {
            k->setArg(0, buffer);
            k->setArg(1, length);
            k->setArg(2, block);
            k->setArg(3, stride);
            
            items = (unsigned int)ceil((float)length/(float)(block*stride));
            
            queue.enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(items), cl::NullRange);
            
            stride *= block;
        }
    } else {
        // Use GPU optimized kernel
        k = &(kernels["reduceMaximum"]);
        unsigned int stride = 1;
        // get optimal work group size
        unsigned int workGroup = k->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
        assert(workGroup > 1);
        
        unsigned int groupCount = (unsigned int)ceil((float)length/(float)(workGroup*stride));
        unsigned int globalSize = workGroup*groupCount;
        
        while(groupCount > 1) {
            k->setArg(0, buffer);
            k->setArg(1, length);
            k->setArg(2, stride);
            k->setArg(3, cl::__local(workGroup*sizeof(cl_float)));
            
            groupCount = (unsigned int)ceil((float)length/(float)(workGroup*stride));
            globalSize = workGroup*groupCount;
            
            queue.enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(workGroup));
            
            stride *= workGroup;
        }
    }
    
    // TODO: use events for synchronization
    queues[0].finish();
    
    // read result
    queue.enqueueReadBuffer(buffer, CL_BLOCKING, 0, sizeof(float), &result);
    return result;
}

void SWE_DimensionalSplittingOpenCL::createBuffers()
{
    cl_mem_flags flags = (unifiedMemory) ? CL_MEM_USE_HOST_PTR : 0;
    size_t bufferSize = h.getRows() * h.getCols() * sizeof(cl_float);
    hd = cl::Buffer(context, (CL_MEM_READ_WRITE | flags), bufferSize, (unifiedMemory ? h.elemVector() : NULL));
    hud = cl::Buffer(context, (CL_MEM_READ_WRITE | flags), bufferSize, (unifiedMemory ? hu.elemVector() : NULL));
    hvd = cl::Buffer(context, (CL_MEM_READ_WRITE | flags), bufferSize, (unifiedMemory ? hv.elemVector() : NULL));
    bd = cl::Buffer(context, (CL_MEM_READ_ONLY | flags), bufferSize, (unifiedMemory ? b.elemVector() : NULL));
    
    // These buffers are named for Xsweep but will also be used in the Ysweep
    hNetUpdatesLeft = cl::Buffer(context, CL_MEM_READ_WRITE, bufferSize);
    hNetUpdatesRight = cl::Buffer(context, CL_MEM_READ_WRITE, bufferSize);
    huNetUpdatesLeft = cl::Buffer(context, CL_MEM_READ_WRITE, bufferSize);
    huNetUpdatesRight = cl::Buffer(context, CL_MEM_READ_WRITE, bufferSize);
    waveSpeeds = cl::Buffer(context, CL_MEM_READ_WRITE, bufferSize);
}

void SWE_DimensionalSplittingOpenCL::synchWaterHeightAfterWrite()
{
    if(unifiedMemory) return;
    queues[0].enqueueWriteBuffer(hd, CL_BLOCKING, 0, h.getRows()*h.getCols()*sizeof(cl_float), h.elemVector());
}

void SWE_DimensionalSplittingOpenCL::synchDischargeAfterWrite()
{
    if(unifiedMemory) return;
    // TODO: use async, non-blocking read
    queues[0].enqueueWriteBuffer(hud, CL_BLOCKING, 0, hu.getRows()*hu.getCols()*sizeof(cl_float), hu.elemVector());
    queues[0].enqueueWriteBuffer(hvd, CL_BLOCKING, 0, hv.getRows()*hv.getCols()*sizeof(cl_float), hv.elemVector());
}

void SWE_DimensionalSplittingOpenCL::synchBathymetryAfterWrite()
{
    if(unifiedMemory) return;
    queues[0].enqueueWriteBuffer(bd, CL_BLOCKING, 0, b.getRows()*b.getCols()*sizeof(cl_float), b.elemVector());
}

void SWE_DimensionalSplittingOpenCL::synchWaterHeightBeforeRead()
{
    if(unifiedMemory) return;
    queues[0].enqueueReadBuffer(hd, CL_BLOCKING, 0, h.getRows()*h.getCols()*sizeof(cl_float), h.elemVector());
}

void SWE_DimensionalSplittingOpenCL::synchDischargeBeforeRead()
{
    if(unifiedMemory) return;
    // TODO: use async, non-blocking read
    queues[0].enqueueReadBuffer(hud, CL_BLOCKING, 0, hu.getRows()*hu.getCols()*sizeof(cl_float), hu.elemVector());
    queues[0].enqueueReadBuffer(hvd, CL_BLOCKING, 0, hv.getRows()*hv.getCols()*sizeof(cl_float), hv.elemVector());
}

void SWE_DimensionalSplittingOpenCL::synchBathymetryBeforeRead()
{
    if(unifiedMemory) return;
    queues[0].enqueueReadBuffer(bd, CL_BLOCKING, 0, b.getCols()*b.getRows()*sizeof(cl_float), b.elemVector());
}

void SWE_DimensionalSplittingOpenCL::computeNumericalFluxes()
{
    // Pointer to kernel object
    cl::Kernel *k;    
    try {
        // enqueue X-Sweep Kernel
        k = &(kernels["dimensionalSplitting_XSweep_netUpdates"]);
        k->setArg(0, hd);
        k->setArg(1, hud);
        k->setArg(2, bd);
        k->setArg(3, hNetUpdatesLeft);
        k->setArg(4, hNetUpdatesRight);
        k->setArg(5, huNetUpdatesLeft);
        k->setArg(6, huNetUpdatesRight);
        k->setArg(7, waveSpeeds);

        queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getCols()-1, h.getRows()), cl::NullRange);

        // TODO: use events for synchronization
        queues[0].finish();

        // reduce waveSpeed Maximum
        float maxWaveSpeed = reduceMaximum(queues[0], waveSpeeds, (h.getCols()-1) * h.getRows());
        // calculate maximum timestep
        maxTimestep = dx/maxWaveSpeed * 0.4f;

        // enqueue updateUnknowns Kernel (X-Sweep)
        k = &(kernels["dimensionalSplitting_XSweep_updateUnknowns"]);
        float dt_dx = maxTimestep / dx;
        k->setArg(0, dt_dx);
        k->setArg(1, hd);
        k->setArg(2, hud);
        k->setArg(3, hNetUpdatesLeft);
        k->setArg(4, hNetUpdatesRight);
        k->setArg(5, huNetUpdatesLeft);
        k->setArg(6, huNetUpdatesRight);

        queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getCols()-2, h.getRows()), cl::NullRange);

        // TODO: use events for synchronization
        queues[0].finish();

        // enqueue Y-Sweep Kernel
        k = &(kernels["dimensionalSplitting_YSweep_netUpdates"]);
        k->setArg(0, hd);
        k->setArg(1, hvd);
        k->setArg(2, bd);
        k->setArg(3, hNetUpdatesLeft);
        k->setArg(4, hNetUpdatesRight);
        k->setArg(5, huNetUpdatesLeft);
        k->setArg(6, huNetUpdatesRight);
        k->setArg(7, waveSpeeds);
        
        queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getCols(), h.getRows()-1), cl::NullRange);

    
        // TODO: use events for synchronization
        queues[0].finish();

        // enqueue netUpdate Kernel (Y-Sweep)
        k = &(kernels["dimensionalSplitting_YSweep_updateUnknowns"]);
        float dt_dy = maxTimestep / dy;
        k->setArg(0, dt_dy);
        k->setArg(1, hd);
        k->setArg(2, hvd);
        k->setArg(3, hNetUpdatesLeft);
        k->setArg(4, hNetUpdatesRight);
        k->setArg(5, huNetUpdatesLeft);
        k->setArg(6, huNetUpdatesRight);

        queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getCols(), h.getRows()-2), cl::NullRange);

        // TODO: use events for synchronization
        queues[0].finish();
    } catch(cl::Error &e) {
        handleError(e);
    }
}

void SWE_DimensionalSplittingOpenCL::updateUnknowns(float dt)
{
    // TODO
}

void SWE_DimensionalSplittingOpenCL::simulateTimestep(float dt)
{
    computeNumericalFluxes();
    updateUnknowns(dt);
}

float SWE_DimensionalSplittingOpenCL::simulate(float tStart,float tEnd)
{
    float t = tStart;
    do {
        //set values in ghost cells
        setGhostLayer();
        
        // compute net updates for every edge
        computeNumericalFluxes();
        //execute a wave propagation time step
        updateUnknowns(maxTimestep);
        t += maxTimestep;
        
        std::cout << "Simulation at time " << t << std::endl << std::flush;
    } while(t < tEnd);

    return t;
}

#endif /* SWE_DIMENSIONALSPLITTINGOPENCL_CPP_ */