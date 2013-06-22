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
    OpenCLWrapper(preferredDeviceType, getCommandQueueProperties())
{
    cl::Program::Sources kernelSources;
    getKernelSources(kernelSources);
    buildProgram(kernelSources);
    
    // TODO: enable multiple devices
    useDevices = 1;
    // useDevices = devices.size();
    
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
    
    std::cout << std::endl;
}

void SWE_DimensionalSplittingOpenCL::reduceMaximum(cl::CommandQueue &queue, cl::Buffer &buffer, unsigned int length, cl::Event *waitEvent, cl::Event *event) {
    cl::Kernel *k;
    
    // List of events a kernel has to wait for
    std::vector<cl::Event> waitList;
    
    if(waitEvent != NULL)
        waitList.push_back(*waitEvent);
    
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
            
            cl::Event *e;
            cl::Event localEvent;
            if(items > 1)
                e = &localEvent;
            else
                e = event;    
            queue.enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(items), cl::NullRange, &waitList, e);
            if(items > 1) {
                waitList.clear();
                waitList.push_back(*e);
            }
            
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
        
        assert(groupCount > 1);
        
        while(groupCount > 1) {
            k->setArg(0, buffer);
            k->setArg(1, length);
            k->setArg(2, stride);
            k->setArg(3, cl::__local(workGroup*sizeof(cl_float)));
            
            groupCount = (unsigned int)ceil((float)length/(float)(workGroup*stride));
            globalSize = workGroup*groupCount;
            
            cl::Event *e;
            cl::Event localEvent;
            if(groupCount > 1)
                e = &localEvent;
            else
                e = event;  
            queue.enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(workGroup), &waitList, e);
            if(groupCount > 1) {
                waitList.clear();
                waitList.push_back(*e);
            }
            
            stride *= workGroup;
        }
    }
    
    queue.flush();
}

void SWE_DimensionalSplittingOpenCL::calculateBufferChunks(size_t cols, size_t deviceCount) {
    
    /**
    Example: 11 Columns, 3 Devices, Chunksize = 4
    
    <pre>
    Updates       *-----*-----*--0--*-----*                                           
                                          *-----*-----*--1--*-----*             
                                                                  *--2--*      
                                                                               
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
    
    calculateBufferChunks(y, useDevices);
    
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
        
        if(i == useDevices-1)
            size -= colSize; // NetUpdates of last device is one column smaller
        
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
    }
}

void SWE_DimensionalSplittingOpenCL::setBoundaryConditions()
{
    cl::Kernel *k;
    std::vector<cl::Event> waitList;
    cl::Event event;
    
    // Set boundary conditions at left and right boundary
    k = &(kernels["setLeftRightBoundary"]);
    k->setArg(0, hd[0]);
    k->setArg(1, hud[0]);
    k->setArg(2, hvd[0]);
    k->setArg(3, h.getCols());
    k->setArg(4, (boundary[BND_LEFT] == OUTFLOW) ? 1.f : -1.f);
    k->setArg(5, (boundary[BND_RIGHT] == OUTFLOW) ? 1.f : -1.f);
    
    queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getRows()), cl::NullRange, NULL, &event);
    waitList.push_back(event);
    
    // Set boundary conditions at top and bottom boundary
    k = &(kernels["setBottomTopBoundary"]);
    k->setArg(0, hd[0]);
    k->setArg(1, hud[0]);
    k->setArg(2, hvd[0]);
    k->setArg(3, h.getRows());
    k->setArg(4, (boundary[BND_BOTTOM] == OUTFLOW) ? 1.f : -1.f);
    k->setArg(5, (boundary[BND_TOP] == OUTFLOW) ? 1.f : -1.f);
    
    queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getCols()), cl::NullRange, &waitList);
    
    queues[0].finish();
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
    
    // Event waitlist for various kernel enqueues
    std::vector<cl::Event> waitList;
    cl::Event xSweepNetUpdatesEvent;
    cl::Event ySweepNetUpdatesEvent;
    cl::Event xSweepUpdateUnknownsEvent;
    cl::Event ySweepUpdateUnknownsEvent;
    try {
        // enqueue X-Sweep Kernel
        k = &(kernels["dimensionalSplitting_XSweep_netUpdates"]);
        k->setArg(0, hd[0]);
        k->setArg(1, hud[0]);
        k->setArg(2, bd[0]);
        k->setArg(3, hNetUpdatesLeft[0]);
        k->setArg(4, hNetUpdatesRight[0]);
        k->setArg(5, huNetUpdatesLeft[0]);
        k->setArg(6, huNetUpdatesRight[0]);
        k->setArg(7, waveSpeeds[0]);

        queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getCols()-1, h.getRows()), cl::NullRange, NULL, &xSweepNetUpdatesEvent);
        
        // reduce waveSpeed Maximum
        cl::Event maximumEvent;
        reduceMaximum(queues[0], waveSpeeds[0], (h.getCols()-1) * h.getRows(), &xSweepNetUpdatesEvent, &maximumEvent);
        maximumEvent.wait();
        
        // Read maximum
        float maxWaveSpeed;
        queues[0].enqueueReadBuffer(waveSpeeds[0], CL_TRUE, 0, sizeof(cl_float), &maxWaveSpeed);
        
        // calculate maximum timestep
        maxTimestep = dx/maxWaveSpeed * 0.4f;
        
        // enqueue updateUnknowns Kernel (X-Sweep)
        k = &(kernels["dimensionalSplitting_XSweep_updateUnknowns"]);
        float dt_dx = maxTimestep / dx;
        k->setArg(0, dt_dx);
        k->setArg(1, hd[0]);
        k->setArg(2, hud[0]);
        k->setArg(3, hNetUpdatesLeft[0]);
        k->setArg(4, hNetUpdatesRight[0]);
        k->setArg(5, huNetUpdatesLeft[0]);
        k->setArg(6, huNetUpdatesRight[0]);

        queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getCols()-2, h.getRows()), cl::NullRange, NULL, &xSweepUpdateUnknownsEvent);
        waitList.push_back(xSweepUpdateUnknownsEvent);

        // enqueue Y-Sweep Kernel
        k = &(kernels["dimensionalSplitting_YSweep_netUpdates"]);
        k->setArg(0, hd[0]);
        k->setArg(1, hvd[0]);
        k->setArg(2, bd[0]);
        k->setArg(3, hNetUpdatesLeft[0]);
        k->setArg(4, hNetUpdatesRight[0]);
        k->setArg(5, huNetUpdatesLeft[0]);
        k->setArg(6, huNetUpdatesRight[0]);
        k->setArg(7, waveSpeeds[0]);
        
        queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getCols(), h.getRows()-1), cl::NullRange, &waitList, &ySweepNetUpdatesEvent);
        waitList.clear();
        waitList.push_back(ySweepNetUpdatesEvent);

        // enqueue netUpdate Kernel (Y-Sweep)
        k = &(kernels["dimensionalSplitting_YSweep_updateUnknowns"]);
        float dt_dy = maxTimestep / dy;
        k->setArg(0, dt_dy);
        k->setArg(1, hd[0]);
        k->setArg(2, hvd[0]);
        k->setArg(3, hNetUpdatesLeft[0]);
        k->setArg(4, hNetUpdatesRight[0]);
        k->setArg(5, huNetUpdatesLeft[0]);
        k->setArg(6, huNetUpdatesRight[0]);
        
        queues[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(h.getCols(), h.getRows()-2), cl::NullRange, &waitList, &ySweepUpdateUnknownsEvent);
        
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
