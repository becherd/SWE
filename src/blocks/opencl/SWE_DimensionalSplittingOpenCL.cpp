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
    
    // read result
    queue.enqueueReadBuffer(buffer, CL_BLOCKING, 0, sizeof(float), &result);
    return result;
}

void SWE_DimensionalSplittingOpenCL::createBuffers()
{
    cl_mem_flags flags = getBufferMemoryFlags(true);
    hd = cl::Buffer(context, (CL_MEM_READ_WRITE | flags), h.getRows()*h.getCols(), h.elemVector());
    hud = cl::Buffer(context, (CL_MEM_READ_WRITE | flags), hu.getRows()*hu.getCols(), hu.elemVector());
    hvd = cl::Buffer(context, (CL_MEM_READ_WRITE | flags), hv.getRows()*hv.getCols(), hv.elemVector());
    bd = cl::Buffer(context, (CL_MEM_READ_ONLY | flags), b.getRows()*b.getCols(), b.elemVector());
}

void SWE_DimensionalSplittingOpenCL::synchWaterHeightAfterWrite()
{
    if(unifiedMemory) return;
    queues[0].enqueueWriteBuffer(hd, CL_BLOCKING, 0, h.getCols()*h.getRows(), h.elemVector());
}

void SWE_DimensionalSplittingOpenCL::synchDischargeAfterWrite()
{
    if(unifiedMemory) return;
    // TODO: use async, non-blocking read
    queues[0].enqueueWriteBuffer(hud, CL_BLOCKING, 0, hu.getCols()*hu.getRows(), hu.elemVector());
    queues[0].enqueueWriteBuffer(hvd, CL_BLOCKING, 0, hv.getCols()*hv.getRows(), hv.elemVector());
}

void SWE_DimensionalSplittingOpenCL::synchBathymetryAfterWrite()
{
    if(unifiedMemory) return;
    queues[0].enqueueWriteBuffer(bd, CL_BLOCKING, 0, b.getCols()*b.getRows(), b.elemVector());
}

void SWE_DimensionalSplittingOpenCL::synchWaterHeightBeforeRead()
{
    if(unifiedMemory) return;
    queues[0].enqueueReadBuffer(hd, CL_BLOCKING, 0, h.getCols()*h.getRows(), h.elemVector());
}

void SWE_DimensionalSplittingOpenCL::synchDischargeBeforeRead()
{
    if(unifiedMemory) return;
    // TODO: use async, non-blocking read
    queues[0].enqueueReadBuffer(hud, CL_BLOCKING, 0, hu.getCols()*hu.getRows(), hu.elemVector());
    queues[0].enqueueReadBuffer(hvd, CL_BLOCKING, 0, hv.getCols()*hv.getRows(), hv.elemVector());
}

void SWE_DimensionalSplittingOpenCL::synchBathymetryBeforeRead()
{
    if(unifiedMemory) return;
    queues[0].enqueueReadBuffer(bd, CL_BLOCKING, 0, b.getCols()*b.getRows(), b.elemVector());
}

void SWE_DimensionalSplittingOpenCL::computeNumericalFluxes()
{
    // TODO
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
