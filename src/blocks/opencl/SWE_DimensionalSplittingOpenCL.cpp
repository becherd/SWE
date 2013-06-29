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
    cl_device_type preferredDeviceType,
    unsigned int maxDevices,
    KernelType _kernelType,
    size_t _workGroupSize):
    SWE_Block(l_nx, l_ny, l_dx, l_dy),
    OpenCLWrapper(preferredDeviceType, getCommandQueueProperties(), _workGroupSize),
    kernelType(_kernelType)
{
    cl::Program::Sources kernelSources;
    getKernelSources(kernelSources);
    
    std::string memOpts, reduceOpts;
    if(kernelType == MEM_GLOBAL) {
        memOpts = std::string("-D MEM_GLOBAL ");
    } else if(kernelType == MEM_LOCAL) {
        memOpts = std::string("-D MEM_LOCAL ");
    }
    
    if(devices[0].getInfo<CL_DEVICE_TYPE>() != CL_DEVICE_TYPE_CPU && kernelType == MEM_LOCAL) {
        // reduce locally
        kernelReduceType = MEM_LOCAL;
        reduceOpts = std::string("-D LOCAL_REDUCE ");
    } else {
        // reduce globally
        kernelReduceType = MEM_GLOBAL;
        reduceOpts = std::string("-D GLOBAL_REDUCE ");
    }
    
#ifdef NDEBUG
    std::string debugOpts = std::string(" ");
#else
    std::string debugOpts = std::string("-D DEBUG ");
#endif
    
    std::string options = memOpts + debugOpts + reduceOpts;
    
    buildProgram(kernelSources, options);
    
    if(maxDevices == 0)
        useDevices = devices.size();
    else
        useDevices = std::min((size_t)maxDevices, devices.size());
    
    createBuffers();
}

void SWE_DimensionalSplittingOpenCL::printDeviceInformation()
{
    std::cout << "Found " << devices.size() << " OpenCL devices of type ";
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
    
    std::cout << "Using " << useDevices << " of " << devices.size() << " OpenCL devices." << std::endl;
    
    std::cout << "Using ";
    if(kernelType == MEM_GLOBAL) std::cout << "global";
    else std::cout << "local";
    std::cout << " memory." << std::endl;
    
    std::cout << "Using ";
    if(kernelReduceType == MEM_GLOBAL) std::cout << "global";
    else std::cout << "local";
    std::cout << " maximum reduction." << std::endl;
    
    if(kernelType == MEM_LOCAL)
        std::cout << "Maximum work group size: " << workGroupSize << std::endl;
    std::cout << std::endl;
}

void SWE_DimensionalSplittingOpenCL::printProfilingInformation()
{
    std::cout << "OpenCL Kernel and Memory Operation Profiling:" << std::endl;
    std::map<std::string, profilingInfo>::const_iterator opItr;
    for(opItr = profilingEvents.begin(); opItr != profilingEvents.end(); ++opItr) {
        std::cout << "  " << opItr->first << ": " << std::endl;
        
        std::map< ProfilingState, cl_ulong >::const_iterator stateItr;
        for(stateItr = opItr->second.second.begin(); stateItr != opItr->second.second.end(); ++stateItr) {
            std::cout << "    ";
            if(stateItr->first == PROFILING_QUEUE)
                std::cout << "QUEUE";
            else if(stateItr->first == PROFILING_SUBMIT)
                std::cout << "SUBMIT";
            else
                std::cout << "EXEC";
            
            std::cout << ": " << float(stateItr->second)/1.0e9 << " seconds" << std::endl;
        }
    }
}
 
void SWE_DimensionalSplittingOpenCL::reduceMaximum(cl::CommandQueue &queue, cl::Buffer &buffer, size_t length, cl::Event *waitEvent, cl::Event *event) {
    cl::Kernel *k;
    
    // List of events a kernel has to wait for
    std::vector<cl::Event> waitList;
    
    if(waitEvent != NULL)
        waitList.push_back(*waitEvent);
    
    cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
    
    if(device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU) {
        // Use CPU optimized kernel
        k = &(kernels["reduceMaximumCPU"]);
        size_t stride = 1;
        size_t block = std::min(std::max((size_t)length/1024, (size_t)16), (size_t)8192);
        size_t items;
        
        do {
            items = (size_t)ceil((float)length/(float)(block*stride));
            
            try {
                k->setArg(0, buffer);
                k->setArg(1, (cl_uint)length);
                k->setArg(2, (cl_uint)block);
                k->setArg(3, (cl_uint)stride);
            } catch(cl::Error &e) {
                handleError(e, "Unable to set reduceMaximumCPU kernel arguments");
            }
            
            cl::Event localEvent;
            cl::Event *e = (items > 1) ? &localEvent : event;
            try {
                queue.enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(items), cl::NullRange, &waitList, e);  
                addProfilingEvent(*e, "reduceMaximumCPU");
            } catch(cl::Error &e) {
                handleError(e, "Unable to enqueue reduceMaximumCPU kernel");
            }
            if(items > 1) {
                waitList.clear();
                waitList.push_back(*e);
            }
            
            stride *= block;
            // avoid stride overflow
            stride = std::min(stride, length);
        } while(items > 1);
    } else {
        // Use GPU optimized kernel
        k = &(kernels["reduceMaximum"]);
        size_t stride = 1;
        // get optimal work group size
        size_t workGroup = getKernelGroupSize(*k, device);
        assert(workGroup > 1);
        
        size_t groupCount, globalSize;
        
        do {
            try {
                k->setArg(0, buffer);
                k->setArg(1, (cl_uint)length);
                k->setArg(2, (cl_uint)stride);
                k->setArg(3, cl::__local(workGroup*sizeof(cl_float)));
            } catch(cl::Error &e) {
                handleError(e, "Unable to set reduceMaximum kernel arguments");
            }
            groupCount = (size_t)ceil((float)length/(float)(workGroup*stride));
            globalSize = workGroup*groupCount;
            
            
            cl::Event localEvent;
            cl::Event *e = (groupCount > 1) ? &localEvent : event;
            try {
                queue.enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(workGroup), &waitList, e);                
                addProfilingEvent(*e, "reduceMaximum");
            } catch(cl::Error &e) {
                handleError(e, "Unable to enqueue reduceMaximum kernel");
            }
            if(groupCount > 1) {
                waitList.clear();
                waitList.push_back(*e);
            }
            
            stride *= workGroup;
            // avoid stride overflow
            stride = std::min(stride, length);
        } while(groupCount > 1);
    }
    
    queue.flush();
}

void SWE_DimensionalSplittingOpenCL::calculateBufferChunks(size_t cols, size_t deviceCount) {
    
    /**
    Example: 11 Columns, 3 Devices, Chunksize = 4
    
    <pre>
    Updates       *-----*-----*--0--*-----*                                           
                                          *-----*-----*--1--*-----*             
                                                                  *--2--*-----*      
                                                                               
    Edges         0     1     2     3     4     5     6     7     8     9
            |     |     |     |     |     |     |     |     |     |     |     |
    Cells   |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |
            |     |     |     |     |     |     |     |     |     |     |     |
                                                                               
    Vars    +--------------0--------------+                                    
                                    +---------------1-------------+            
                                                            +---------2-------+
    </pre>
    
    In this example, the net updates at edges 4 and 8 must be copied
    to the update buffers of device 0 and 1 respectively. After applying 
    the netupdates on devices 0 and 1, the overlapping variable columns 
    for h and hu must be copied back to the varible buffers of device 1
    and 2 respectively. Note that hv does not have to be copied since 
    the overlapping net updates only affect the X-Sweep (and not the Y-Sweep)
    */
    
    chunkSize = static_cast <size_t> (std::ceil(float(cols) / deviceCount));
    
    size_t start = 0;
    size_t end = 0;
    
    while(end < cols-1) {
        end = start + chunkSize;
        end = std::min(end, cols-1);
        
        bufferChunks.push_back( std::make_pair(start, end-start+1) );
        
        start = end;
    }
}

void SWE_DimensionalSplittingOpenCL::createBuffers()
{
    size_t y = h.getRows();
    size_t colSize = y * sizeof(cl_float);
    
    calculateBufferChunks(h.getCols(), useDevices);
    
    for(unsigned int i = 0; i < useDevices; i++) {
        size_t size = colSize * bufferChunks[i].second;
        try {
            hd.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, size));
            hud.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, size));
            hvd.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, size));
            bd.push_back(cl::Buffer(context, CL_MEM_READ_ONLY, size));
        } catch(cl::Error &e) {
            handleError(e, "Unable to create variable buffers");
        }
        
        // These buffers are named for Xsweep but will also be used in the Ysweep
        try {
            hNetUpdatesLeft.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, size));
            hNetUpdatesRight.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, size));
            huNetUpdatesLeft.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, size));
            huNetUpdatesRight.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, size));
            waveSpeeds.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, size));
        } catch(cl::Error &e) {
            handleError(e, "Unable to create update buffers");
        }
        
        if(i < useDevices-1) {
            // create edge copy buffers
            try {
                hNetUpdatesLeftEdgeCopy.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, colSize));
                huNetUpdatesLeftEdgeCopy.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, colSize));
            } catch(cl::Error &e) {
                handleError(e, "Unable to create edge copy buffers");
            }
        }
    }
}

void SWE_DimensionalSplittingOpenCL::setBoundaryConditions()
{
    cl::Kernel *k;
    std::vector<cl::Event> waitList;
    cl::Event event;
    
    // Set boundary conditions at top and bottom boundary
    k = &(kernels["setBottomTopBoundary"]);
    for(unsigned int i = 0; i < useDevices; i++) {
        try {
            k->setArg(0, hd[i]);
            k->setArg(1, hud[i]);
            k->setArg(2, hvd[i]);
            k->setArg(3, h.getRows());
            k->setArg(4, (boundary[BND_BOTTOM] == OUTFLOW) ? 1.f : -1.f);
            k->setArg(5, (boundary[BND_TOP] == OUTFLOW) ? 1.f : -1.f);
        } catch(cl::Error &e) {
            handleError(e, "Unable to set setBottomTopBoundary kernel arguments");
        }
        
        size_t length = bufferChunks[i].second;
        try {
            queues[i].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(length), cl::NullRange, NULL, &event);
            addProfilingEvent(event, "set top/bottom boundary");
        } catch(cl::Error &e) {
            handleError(e, "Unable to enqueue setBottomTopBoundary kernel");
        }

        waitList.push_back(event);
    }
    
    // Set boundary conditions at left boundary
    k = &(kernels["setLeftBoundary"]);
    try {
        k->setArg(0, hd[0]);
        k->setArg(1, hud[0]);
        k->setArg(2, hvd[0]);
        k->setArg(3, (boundary[BND_LEFT] == OUTFLOW) ? 1.f : -1.f);
    } catch(cl::Error &e) {
        handleError(e, "Unable to set setLeftBoundary kernel arguments");
    }
    
    try {
        queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getRows()), cl::NullRange, &waitList, &event);
        addProfilingEvent(event, "set left boundary");
    } catch(cl::Error &e) {
        handleError(e, "Unable to enqueue setLeftBoundary kernel");
    }
    
    waitList.push_back(event);
    
    // Set boundary conditions at right boundary
    k = &(kernels["setRightBoundary"]);
    try {
        k->setArg(0, hd[useDevices-1]);
        k->setArg(1, hud[useDevices-1]);
        k->setArg(2, hvd[useDevices-1]);
        k->setArg(3, (unsigned int)bufferChunks[useDevices-1].second);
        k->setArg(4, (boundary[BND_RIGHT] == OUTFLOW) ? 1.f : -1.f);
    } catch(cl::Error &e) {
        handleError(e, "Unable to set setRightBoundary kernel arguments");
    }
    
    try {
        queues[useDevices-1].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getRows()), cl::NullRange, &waitList, &event);
        addProfilingEvent(event, "set right boundary");
    } catch(cl::Error &e) {
        handleError(e, "Unable to enqueue setRightBoundary kernel");
    }
    
    waitList.push_back(event);
    
    for(unsigned int i = 0; i < useDevices; i++)
        queues[i].flush();
    
    cl::Event::waitForEvents(waitList);
}

void SWE_DimensionalSplittingOpenCL::syncBuffersBeforeRead(std::vector< std::pair< std::vector<cl::Buffer>*, float*> > &buffers) {
    std::vector<cl::Event> events;
    size_t y = h.getRows();
    size_t colSize = sizeof(cl_float) * y;
    
    for(unsigned int i = 0; i < buffers.size(); i++) {
        for(unsigned int j = 0; j < useDevices; j++) {
            cl::Event event;
            size_t start = bufferChunks[j].first;
            size_t length = bufferChunks[j].second;
            size_t size = ((j == useDevices-1) ? length : (length-1)) * colSize;
            float* dst = buffers[i].second + (start*y);
            queues[j].enqueueReadBuffer((*buffers[i].first)[j], CL_FALSE, 0, size, dst, NULL, &event);
            addProfilingEvent(event, "sync before read");
            events.push_back(event);
        }
    }
    
    // Wait until all transfers have finished
    cl::Event::waitForEvents(events);
}

void SWE_DimensionalSplittingOpenCL::syncBuffersAfterWrite(std::vector< std::pair< std::vector<cl::Buffer>*, float*> > &buffers) {
    std::vector<cl::Event> events;
    size_t y = h.getRows();
    size_t colSize = sizeof(cl_float) * y;
    
    for(unsigned int i = 0; i < buffers.size(); i++) {
        for(unsigned int j = 0; j < useDevices; j++) {
            cl::Event event;
            size_t start = bufferChunks[j].first;
            size_t length = bufferChunks[j].second;
            size_t size = length * colSize;
            float* src = buffers[i].second + (start*y);
            queues[j].enqueueWriteBuffer((*buffers[i].first)[j], CL_FALSE, 0, size, src, NULL, &event);
            addProfilingEvent(event, "sync after write");
            events.push_back(event);
        }
    }
    
    // Wait until all transfers have finished
    cl::Event::waitForEvents(events);
}

void SWE_DimensionalSplittingOpenCL::synchAfterWrite()
{
    std::vector< std::pair< std::vector<cl::Buffer>*, float*> > buffers;
    buffers.push_back(std::make_pair(&hd, h.elemVector()));
    buffers.push_back(std::make_pair(&hud, hu.elemVector()));
    buffers.push_back(std::make_pair(&hvd, hv.elemVector()));
    buffers.push_back(std::make_pair(&bd, b.elemVector()));
    syncBuffersAfterWrite(buffers);
}

void SWE_DimensionalSplittingOpenCL::synchWaterHeightAfterWrite()
{
    std::vector< std::pair< std::vector<cl::Buffer>*, float*> > buffers;
    buffers.push_back(std::make_pair(&hd, h.elemVector()));
    syncBuffersAfterWrite(buffers);
}

void SWE_DimensionalSplittingOpenCL::synchDischargeAfterWrite()
{
    std::vector< std::pair< std::vector<cl::Buffer>*, float*> > buffers;
    buffers.push_back(std::make_pair(&hud, hu.elemVector()));
    buffers.push_back(std::make_pair(&hvd, hv.elemVector()));
    syncBuffersAfterWrite(buffers);
}

void SWE_DimensionalSplittingOpenCL::synchBathymetryAfterWrite()
{
    std::vector< std::pair< std::vector<cl::Buffer>*, float*> > buffers;
    buffers.push_back(std::make_pair(&bd, b.elemVector()));
    syncBuffersAfterWrite(buffers);
}

void SWE_DimensionalSplittingOpenCL::synchBeforeRead()
{
    std::vector< std::pair< std::vector<cl::Buffer>*, float*> > buffers;
    buffers.push_back(std::make_pair(&hd, h.elemVector()));
    buffers.push_back(std::make_pair(&hud, hu.elemVector()));
    buffers.push_back(std::make_pair(&hvd, hv.elemVector()));
    buffers.push_back(std::make_pair(&bd, b.elemVector()));
    syncBuffersBeforeRead(buffers);
}

void SWE_DimensionalSplittingOpenCL::synchWaterHeightBeforeRead()
{
    std::vector< std::pair< std::vector<cl::Buffer>*, float*> > buffers;
    buffers.push_back(std::make_pair(&hd, h.elemVector()));
    syncBuffersBeforeRead(buffers);
}

void SWE_DimensionalSplittingOpenCL::synchDischargeBeforeRead()
{
    std::vector< std::pair< std::vector<cl::Buffer>*, float*> > buffers;
    buffers.push_back(std::make_pair(&hud, hu.elemVector()));
    buffers.push_back(std::make_pair(&hvd, hv.elemVector()));
    syncBuffersBeforeRead(buffers);
}

void SWE_DimensionalSplittingOpenCL::synchBathymetryBeforeRead()
{
    std::vector< std::pair< std::vector<cl::Buffer>*, float*> > buffers;
    buffers.push_back(std::make_pair(&bd, b.elemVector()));
    syncBuffersBeforeRead(buffers);
}

void SWE_DimensionalSplittingOpenCL::computeNumericalFluxes()
{
    // Pointer to kernel object
    cl::Kernel *k;
    // Number of rows
    size_t y = h.getRows();
    // Size of a column (in bytes)
    size_t colSize = y*sizeof(cl_float);
    // Length variable (number of cols per chunk)
    size_t length;
    
    // Event waitlist for various kernel enqueues
    std::vector<cl::Event> waitList;
    // Device specific event waitList
    std::vector< std::vector<cl::Event> > deviceWaitList;
    for(unsigned int i = 0; i < useDevices; i++)
        deviceWaitList.push_back(std::vector<cl::Event>());
    
    try {
        // enqueue X-Sweep Kernel
        k = &(kernels["dimensionalSplitting_XSweep_netUpdates"]);
        
        for(unsigned int i = 0; i < useDevices; i++) {
            
            length = bufferChunks[i].second;
            
            size_t groupSize, globalSize;
            cl::NDRange globalRange, localRange;
            if(kernelType == MEM_LOCAL) {
                groupSize = getKernelGroupSize(*k, devices[i]);
                globalSize = getKernelRange(groupSize, length-1);
                globalRange = cl::NDRange(globalSize, y);
                localRange = cl::NDRange(groupSize, 1);
            } else {
                globalRange = cl::NDRange(length-1, y);
                localRange = cl::NullRange;
            }
            k->setArg(0, hd[i]);
            k->setArg(1, hud[i]);
            k->setArg(2, bd[i]);
            k->setArg(3, hNetUpdatesLeft[i]);
            k->setArg(4, hNetUpdatesRight[i]);
            k->setArg(5, huNetUpdatesLeft[i]);
            k->setArg(6, huNetUpdatesRight[i]);
            k->setArg(7, waveSpeeds[i]);
            if(kernelType == MEM_LOCAL) {
                k->setArg(8, cl::__local((groupSize+1)*sizeof(cl_float)));
                k->setArg(9, cl::__local((groupSize+1)*sizeof(cl_float)));
                k->setArg(10, cl::__local((groupSize+1)*sizeof(cl_float)));
                k->setArg(11, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(12, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(13, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(14, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(15, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(16, (unsigned int)length-1);
                k->setArg(17, (unsigned int)y);
            }
            
            cl::Event sweepEvent;
            queues[i].enqueueNDRangeKernel(*k, cl::NullRange, globalRange, localRange, NULL, &sweepEvent);
            addProfilingEvent(sweepEvent, "X-Sweep");
            
            // reduce waveSpeed Maximum
            cl::Event maximumEvent;
            size_t maxWaveSpeedLength;
            if(kernelReduceType == MEM_LOCAL) {
                // local reduction
                maxWaveSpeedLength = y * globalSize/groupSize;
            } else {
                // global reduction
                maxWaveSpeedLength = (length-1) * y;
            }
            reduceMaximum(queues[i], waveSpeeds[i], maxWaveSpeedLength, &sweepEvent, &maximumEvent);
            waitList.push_back(maximumEvent);
        }
        
        cl::Event::waitForEvents(waitList);
        waitList.clear();
        
        // Read maximum
        float maxWaveSpeed = -INFINITY;
        for(unsigned int i = 0; i < useDevices; i++) {
            float result;
            cl::Event e;
            queues[i].enqueueReadBuffer(waveSpeeds[i], CL_TRUE, 0, sizeof(cl_float), &result, NULL, &e);
            addProfilingEvent(e, "read maxWaveSpeed (X)");
            maxWaveSpeed = std::max(maxWaveSpeed, result);
        }
        
        // calculate maximum timestep
        maxTimestep = dx/maxWaveSpeed * 0.4f;
        
        // Copy net update buffers at edges from device n+1 to device n
        for(unsigned int i = 0; i < useDevices-1; i++) {            
            cl::Event e;
            std::vector<cl::Event> copyWaitListH, copyWaitListHu;
            unsigned int cols;
            
            cols = (unsigned int)bufferChunks[i+1].second;
            k = &(kernels["writeNetUpdatesEdgeCopy"]);
            // READ MEMORY TO EDGE COPY ON DEVICE i+1
            // write h
            k->setArg(0, hNetUpdatesLeft[i+1]);
            k->setArg(1, hNetUpdatesLeftEdgeCopy[i]);
            k->setArg(2, cols);
            queues[i+1].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getRows()), cl::NullRange, NULL, &e);
            addProfilingEvent(e, "writeNetUpdatesEdgeCopy");
            copyWaitListH.push_back(e);
            
            // write hu
            k->setArg(0, huNetUpdatesLeft[i+1]);
            k->setArg(1, huNetUpdatesLeftEdgeCopy[i]);
            k->setArg(2, cols);
            queues[i+1].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getRows()), cl::NullRange, NULL, &e);
            addProfilingEvent(e, "writeNetUpdatesEdgeCopy");
            copyWaitListHu.push_back(e);
            
            // READ EDGE COPY ON DEVICE i INTO MEMORY
            cols = (unsigned int)bufferChunks[i].second;
            k = &(kernels["readNetUpdatesEdgeCopy"]);
            // read h
            k->setArg(0, hNetUpdatesLeft[i]);
            k->setArg(1, hNetUpdatesLeftEdgeCopy[i]);
            k->setArg(2, cols);
            queues[i].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getRows()), cl::NullRange, &copyWaitListH, &e);
            addProfilingEvent(e, "readNetUpdatesEdgeCopy");
            deviceWaitList[i].push_back(e);
            
            // read hu
            k->setArg(0, huNetUpdatesLeft[i]);
            k->setArg(1, huNetUpdatesLeftEdgeCopy[i]);
            k->setArg(2, cols);
            queues[i].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getRows()), cl::NullRange, &copyWaitListHu, &e);
            addProfilingEvent(e, "readNetUpdatesEdgeCopy");
            deviceWaitList[i].push_back(e);
        }
        
        // enqueue updateUnknowns Kernel (X-Sweep)
        k = &(kernels["dimensionalSplitting_XSweep_updateUnknowns"]);
        float dt_dx = maxTimestep / dx;
        for(unsigned int i = 0; i < useDevices; i++) {
            
            length = bufferChunks[i].second;
            
            size_t groupSize;
            cl::NDRange globalRange, localRange;
            if(kernelType == MEM_LOCAL) {
                groupSize = getKernelGroupSize(*k, devices[i]);
                globalRange = cl::NDRange(getKernelRange(groupSize, length-1), y);
                localRange = cl::NDRange(groupSize, 1);
            } else {
                globalRange = cl::NDRange(length-1, y);
                localRange = cl::NullRange;
            }
            
            k->setArg(0, dt_dx);
            k->setArg(1, hd[i]);
            k->setArg(2, hud[i]);
            k->setArg(3, hNetUpdatesLeft[i]);
            k->setArg(4, hNetUpdatesRight[i]);
            k->setArg(5, huNetUpdatesLeft[i]);
            k->setArg(6, huNetUpdatesRight[i]);
            if(kernelType == MEM_LOCAL) {
                k->setArg(7, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(8, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(9, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(10, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(11, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(12, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(13, (unsigned int)length-1);
                k->setArg(14, (unsigned int)y);
            }
            
            cl::Event e;
            queues[i].enqueueNDRangeKernel(*k, cl::NullRange, globalRange, localRange, &deviceWaitList[i], &e);
            addProfilingEvent(e, "X-Update");
            waitList.push_back(e);
        }
        
        // Copy updates edges between buffers from device n to n+1
        for(unsigned int i = 0; i < useDevices-1; i++) {
            // We're initiating a copy from device n (to fetch data from device n+1)
            length = bufferChunks[i].second;
            size_t offset = (length-1)*colSize;
            cl::Event e;
            deviceWaitList[i+1].clear();
            
            queues[i+1].enqueueCopyBuffer(hd[i], hd[i+1], offset, 0, colSize, &waitList, &e);
            deviceWaitList[i+1].push_back(e);
            addProfilingEvent(e, "copy edges");
            queues[i+1].enqueueCopyBuffer(hud[i], hud[i+1], offset, 0, colSize, &waitList, &e);
            deviceWaitList[i+1].push_back(e);
            addProfilingEvent(e, "copy edges");
            // Note that we do not need to copy hvd, since vertical momentum is not updated in the X-Sweep
        }
        
        waitList.clear();
        
        // enqueue Y-Sweep Kernel
        k = &(kernels["dimensionalSplitting_YSweep_netUpdates"]);
        for(unsigned int i = 0; i < useDevices; i++) {
            
            length = bufferChunks[i].second;
            
            size_t groupSize, globalSize;
            cl::NDRange globalRange, localRange;
            if(kernelType == MEM_LOCAL) {
                groupSize = getKernelGroupSize(*k, devices[i]);
                globalSize = getKernelRange(groupSize, y-1);
                globalRange = cl::NDRange(length, globalSize);
                localRange = cl::NDRange(1, groupSize);                
            } else {
                globalRange = cl::NDRange(length, y-1);
                localRange = cl::NullRange;
            }
            
            k->setArg(0, hd[i]);
            k->setArg(1, hvd[i]);
            k->setArg(2, bd[i]);
            k->setArg(3, hNetUpdatesLeft[i]);
            k->setArg(4, hNetUpdatesRight[i]);
            k->setArg(5, huNetUpdatesLeft[i]);
            k->setArg(6, huNetUpdatesRight[i]);
            k->setArg(7, waveSpeeds[i]);
            if(kernelType == MEM_LOCAL) {
                k->setArg(8, cl::__local((groupSize+1)*sizeof(cl_float)));
                k->setArg(9, cl::__local((groupSize+1)*sizeof(cl_float)));
                k->setArg(10, cl::__local((groupSize+1)*sizeof(cl_float)));
                k->setArg(11, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(12, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(13, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(14, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(15, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(16, (unsigned int)length);
                k->setArg(17, (unsigned int)y-1);
            }
            
            cl::Event e;
            queues[i].enqueueNDRangeKernel(*k, cl::NullRange, globalRange, localRange, &deviceWaitList[i], &e);
            addProfilingEvent(e, "Y-Sweep");
            deviceWaitList[i].clear();
            deviceWaitList[i].push_back(e);
            
#ifndef NDEBUG
            // reduce waveSpeed Maximum
            cl::Event maximumEvent;
            size_t maxWaveSpeedLength;
            if(kernelReduceType == MEM_LOCAL) {
                // local reduction
                maxWaveSpeedLength = length * globalSize/groupSize;
            } else {
                // global reduction
                maxWaveSpeedLength = length * (y-1);
            }
            reduceMaximum(queues[i], waveSpeeds[i], maxWaveSpeedLength, &e, &maximumEvent);
            waitList.push_back(maximumEvent);
#endif
        }
        
#ifndef NDEBUG
        cl::Event::waitForEvents(waitList);
        waitList.clear();
        // Read maximum
        float maxWaveSpeedY = -INFINITY;
        for(unsigned int i = 0; i < useDevices; i++) {
            float result;
            cl::Event e;
            queues[i].enqueueReadBuffer(waveSpeeds[i], CL_TRUE, 0, sizeof(cl_float), &result, &waitList, &e);
            addProfilingEvent(e, "read maxWaveSpeed (Y)");
            maxWaveSpeedY = std::max(maxWaveSpeedY, result);
        }
        
        // Check CFL condition
        float maxTimestepY = .5f * dy / maxWaveSpeed;
        if(maxTimestepY >= maxTimestep) {
            // OK, everything's fine
        } else {
            // Oops, CFL condition is NOT satisfied
            std::cerr << "WARNING: CFL condition is not satisfied in y-sweep: "
                      << maxTimestepY << " < " << maxTimestep << std::endl;
        }
#endif

        // enqueue netUpdate Kernel (Y-Sweep)
        k = &(kernels["dimensionalSplitting_YSweep_updateUnknowns"]);
        float dt_dy = maxTimestep / dy;
        for(unsigned int i = 0; i < useDevices; i++) {
            
            length = bufferChunks[i].second;
            
            size_t groupSize;
            cl::NDRange globalRange, localRange;
            if(kernelType == MEM_LOCAL) {
                groupSize = getKernelGroupSize(*k, devices[i]);
                globalRange = cl::NDRange(length, getKernelRange(groupSize, y-2));
                localRange = cl::NDRange(1, groupSize);
            } else {
                globalRange = cl::NDRange(length, y-2);
                localRange = cl::NullRange;
            }
            
            k->setArg(0, dt_dy);
            k->setArg(1, hd[i]);
            k->setArg(2, hvd[i]);
            k->setArg(3, hNetUpdatesLeft[i]);
            k->setArg(4, hNetUpdatesRight[i]);
            k->setArg(5, huNetUpdatesLeft[i]);
            k->setArg(6, huNetUpdatesRight[i]);
            if(kernelType == MEM_LOCAL) {
                k->setArg(7, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(8, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(9, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(10, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(11, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(12, cl::__local(groupSize*sizeof(cl_float)));
                k->setArg(13, (unsigned int)length);
                k->setArg(14, (unsigned int)y-1);
            }
            
            cl::Event e;
            queues[i].enqueueNDRangeKernel(*k, cl::NullRange, globalRange, localRange, &deviceWaitList[i], &e);
            addProfilingEvent(e, "Y-Update");
            waitList.push_back(e);
        }
        
        // Copy updated edge columns after Y-Sweep so we ensure that the overlapping 
        // edge columns really have identical values
        for(unsigned int i = 0; i < useDevices-1; i++) {
            // We're initiating a copy from device n+1 (to fetch data from device n)
            length = bufferChunks[i].second;
            size_t offset = (length-1)*colSize;
            cl::Event e;
            deviceWaitList[i+1].clear();
            
            queues[i+1].enqueueCopyBuffer(hd[i], hd[i+1], offset, 0, colSize, &waitList, &e);
            addProfilingEvent(e, "copy edges (after Y-update)");
            waitList.push_back(e);
            queues[i+1].enqueueCopyBuffer(hvd[i], hvd[i+1], offset, 0, colSize, &waitList, &e);
            addProfilingEvent(e, "copy edges (after Y-update)");
            waitList.push_back(e);
            // Note that we do not need to copy hvd, since vertical momentum is not updated in the X-Sweep
        }
        
        for(unsigned int i = 0; i < useDevices; i++)
            queues[i].flush();
        
        cl::Event::waitForEvents(waitList);
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
