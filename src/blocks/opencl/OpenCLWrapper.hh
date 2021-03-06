#ifndef OPENCLWRAPPER_HH
#define OPENCLWRAPPER_HH

#define __CL_ENABLE_EXCEPTIONS 1
#include <CL/cl.hpp>
#include <cassert>
#include <iostream>
#include <map>
#include <cmath>

#include <pthread.h>

//! Type to identifiy different execution states (queued<->submitted, submitted<->start, start<->end)
typedef enum { PROFILING_QUEUE, PROFILING_SUBMIT, PROFILING_EXEC } ProfilingState;
//! Profiling information passed to the profiling callback function
typedef std::pair< pthread_mutex_t* , std::map<ProfilingState, cl_ulong> > profilingInfo;

/// OpenCL Wrapper
/**
 * Simplifies much of the commonly needed boilerplate code
 * to set up a computing context, devices, command queues
 * and program kernels using a single device type but
 * an arbitrary number of computing devices and kernel
 * functions
 */
class OpenCLWrapper {
protected:
    //! The OpenCL platform used
    cl::Platform platform;
    //! List of OpenCL device types (in descending priority)
    std::vector<cl_device_type> deviceTypes;
    //! Number of computing devices available for each device type
    std::map<cl_device_type, int> deviceTypeCount;
    
    //! The OpenCL computing context
    cl::Context context;
    //! The OpenCL device type used in the context
    cl_device_type deviceType;
    //! List of devices in the OpenCL context
    std::vector<cl::Device> devices;
    //! List of command queues corresponding to the OpenCL devices
    std::vector<cl::CommandQueue> queues;
    
    //! The OpenCL program
    cl::Program program;
    //! OpenCL Kernels in the program identified by kernel function name
    std::map<std::string, cl::Kernel> kernels;
    
    //! Kernel and memory profiling information
    std::map < std::string, profilingInfo > profilingEvents;

    //! Mutex for exclusive access to profilingEvents
    pthread_mutex_t profilingMutex;
    
    //! Maximum work group size to use for kernel execution
    size_t workGroupSize;
    
    /// Display info about an OpenCL Exception and exit application
    /**
     * @param e The OpenCL Error
     */
    void handleError(cl::Error &e, const char* msg = "") {
        std::cerr << "OpenCL Error: " << msg
                  << " in " << e.what() << " (" << e.err() << ")" << std::endl;
        exit(-1);
    }
    
    /// Setup OpenCL computing platform and read available devices on the platform
    void setupPlatform() {
        std::vector<cl::Platform> platforms;
        std::vector<cl::Device> platformDevices;
        try {
            cl::Platform::get(&platforms);
            if(platforms.size() < 1) {
                // No platforms found
                std::cerr << "No OpenCL Platforms found" << std::endl;
                exit(-1);
            }
        
            // We assume one platform for now
            assert(platforms.size() == 1);
            platform = platforms[0];
        
            // Get a list of devices on the platform
            platform.getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
        
            if(platformDevices.size() < 1) {
                // No platforms found
                std::cerr << "No OpenCL Devices found" << std::endl;
                exit(-1);
            }
            
            for(unsigned int i = 0; i < platformDevices.size(); i++) {
                // for each type, check if device is of this type
                // if so, increase counter for type
                cl_device_type _deviceType;
                platformDevices[i].getInfo(CL_DEVICE_TYPE, &_deviceType);
            
                for(unsigned int j = 0; j < deviceTypes.size(); j++) {
                    if((_deviceType & deviceTypes[j]) == deviceTypes[j]) {
                        // device i is of type j
                        deviceTypeCount[deviceTypes[j]]++;
                    }
                }
            }
        } catch(cl::Error &e) {
            handleError(e);
        }
    }
    
    /// Setup OpenCL computing context and command queues for each device
    /**
     * This sets up an OpenCL computing context. If there is at least one device
     * of the preferred device type available on the platform, the context is created
     * using that type of device. If no preferred type is supplied or there is no 
     * device of that type available on the platform, the "best" available device type
     * is chosen.
     * Note that the context contains ALL computing devices of that type. So if
     * there are two GPUs available on the platform, both GPUs will be used in
     * the context.
     * The queues will be creating using the supplied queue options
     * (for instance out-of-order exec)
     *
     * @param preferredDeviceType The preferred OpenCL device type (CPU, GPU, ..)
     * @param queueProperties OpenCL queue options for device command queues
     */
    void setupContext(  cl_device_type preferredDeviceType = 0,
                        cl_command_queue_properties queueProperties = 0) {
        // reset device type
        deviceType = 0;
        
        // check if preferred type has devices
        if(deviceTypeCount[preferredDeviceType] > 0) {
            deviceType = preferredDeviceType;
        } else {
            // go through type list and choose first device type
            // (list is ordered in descending priority)
            for(unsigned int j = 0; j < deviceTypes.size(); j++) {
                if(deviceTypeCount[deviceTypes[j]] > 0) {
                    deviceType = deviceTypes[j];
                    break;
                }
            }
        }
        
        if(deviceType == 0) {
            // no matching device found
            std::cerr << "No OpenCL Devices found" << std::endl;
            exit(-1);
        }
        
        // create context of chosen device type
        try {        
            // TODO: supply callback function
            cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};        
            context = cl::Context(deviceType, properties);
            
            // Read devices in context
            context.getInfo(CL_CONTEXT_DEVICES, &devices);
            
            // for each computing device, create an in-order command queue
            for(unsigned int i = 0; i < devices.size(); i++) { 
                queues.push_back(cl::CommandQueue(context, devices[i], queueProperties));
            }
        } catch (cl::Error &e) {
            std::cerr << "Error: Unable to create OpenCL context: Error " << e.err() << std::endl;
            std::cerr << "This usually means that there are no computing devices of the specified type" << std::endl;
            handleError(e);
        }
    }
    
   /// Calculate optimal work group size for kernel and device
   /**
    * @param kernel The kernel object to execute
    * @param device The device to execute the kernel on
    * @return The work group size
    */
   inline size_t getKernelGroupSize( const cl::Kernel &kernel, const cl::Device &device ) {
       return std::min(workGroupSize, kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device));
   }
   
   /// Calculate kernel range so that the group size divides kernel range
   /**
    * @param groupSize The group size to use
    * @param range The minimum range to execute on
    * @return The kernel range to use
    */
   inline size_t getKernelRange(size_t groupSize, size_t range) {
       return groupSize * (size_t)ceil(float(range) / groupSize);
   }
    
public:
    /// Constructor
    /**
     * @param preferredDeviceType The preferred OpenCL device type (CPU, GPU, ..)
     * @param queueProperties OpenCL queue options for device command queues
     * @param _workGroupSize The maximum work group size to use for kernel execution
     */
    OpenCLWrapper(  cl_device_type preferredDeviceType = 0,
                    cl_command_queue_properties queueProperties = 0,
                    size_t _workGroupSize = 1024) : workGroupSize(_workGroupSize){
        // List of available OpenCL device types, highest priority first
        deviceTypes.push_back(CL_DEVICE_TYPE_ACCELERATOR);
        deviceTypes.push_back(CL_DEVICE_TYPE_GPU);
        deviceTypes.push_back(CL_DEVICE_TYPE_CPU);
        deviceTypes.push_back(CL_DEVICE_TYPE_DEFAULT);
        
        setupPlatform();
        setupContext(preferredDeviceType, queueProperties);
        
        // init profiling mutex
        pthread_mutex_init(&profilingMutex, NULL);
    }
    
    /// Destructor
    ~OpenCLWrapper() {
        pthread_mutex_destroy(&profilingMutex);
    }
    
    /// Get pointer to profiling info supplied to profiling callback for a certain description (kernel name)
    /**
     * @param description The description (e.g. Kernel name)
     */
    inline profilingInfo* getProfilingCallbackInfo(const char* description) {
        profilingInfo *info = &profilingEvents[std::string(description)];
        info->first = &profilingMutex;
        return info;
    }
    
    /// Add an OpenCL event ot the list of profiled events
    /**
     * @param e The OpenCL event to profile
     * @param description The description (e.g. Kernel name)
     */
    inline void addProfilingEvent(cl::Event &e, const char* description) {
#ifdef OPENCL_PROFILING
        e.setCallback(CL_COMPLETE, OpenCLWrapper::eventProfilingCallback, (void*)getProfilingCallbackInfo(description));
#endif
    }
    
    /// Callback function to profile events, e.g. measure kernel execution time
    static void eventProfilingCallback(cl_event event, cl_int command_exec_status, void *user_data)
    {
        cl_ulong queueTime, submitTime, startTime, endTime;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queueTime, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submitTime, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
        
        profilingInfo *info = (profilingInfo *)user_data;
    
        pthread_mutex_lock(info->first);
        (info->second)[PROFILING_QUEUE] += (submitTime - queueTime);
        (info->second)[PROFILING_SUBMIT] += (startTime - submitTime);
        (info->second)[PROFILING_EXEC] += (endTime - startTime);
        pthread_mutex_unlock(info->first);
    }
    
    void buildProgram(cl::Program::Sources &kernelSources, const std::string &options = std::string()) {
        try {        
            program = cl::Program(context, kernelSources);
            program.build(devices, options.c_str());
        
            std::vector<cl::Kernel> _kernels;
            program.createKernels(&_kernels);
            
            for(unsigned int i = 0; i < _kernels.size(); i++) {
                std::string kernelName;
                _kernels[i].getInfo(CL_KERNEL_FUNCTION_NAME, &kernelName);
                kernels[kernelName] = _kernels[i];
            }
        } catch(cl::Error &e) {
            if(e.err() != CL_BUILD_PROGRAM_FAILURE) {
                handleError(e);
            }
        
            // Kernel build failure
            std::cerr << "Error building OpenCL program:" << std::endl;
            for(unsigned int i = 0; i < devices.size(); i++) {
                cl_build_status buildStatus;
                buildStatus = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[i]);
                std::cerr << "Device " << i << ": ";
                switch(buildStatus) {
                    case CL_BUILD_NONE:
                        std::cerr << "NONE";
                        break;
                    case CL_BUILD_ERROR:
                        std::cerr << "ERROR";
                        break;
                    case CL_BUILD_SUCCESS:
                        std::cerr << "SUCCESS";
                        break;
                    case CL_BUILD_IN_PROGRESS:
                        std::cerr << "IN_PROGRESS";
                        break;
                    default:
                        std::cerr << "UNKNOWN";
                        break;
                }
                std::cerr << std::endl;
                std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i]);
                std::cerr << std::endl;
            }
            exit(-1); 
        }
    }
};

#endif /* OPENCLWRAPPER_HH */