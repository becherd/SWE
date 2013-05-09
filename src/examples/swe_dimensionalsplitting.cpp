#include <iostream>

#include "blocks/SWE_DimensionalSplitting.hh"
#include "scenarios/SWE_simple_scenarios.hh"
#include "scenarios/SWE_PartialDambreak.hh"

#ifdef WRITENETCDF
#include "writer/NetCdfWriter.hh"
#else
#include "writer/VtkWriter.hh"
#endif

#include "tools/help.hh"
#include "tools/Logger.hh"
#include "tools/ProgressBar.hh"

int main( int argc, char** argv ) {
    if(argc != 4 && argc != 5) {
        std::cout << "Usage: ./SWE_gnu_debug_none_dimsplit <x> <y> <name> [checkpoints]" << std::endl
                  << "    e.g. ./SWE_gnu_debug_none_dimsplit 100 200 test 50" << std::endl
                  << "    For a block size of 100 x 200 and writing 50 checkpoints with basename 'test'" << std::endl;
        return 1;
    }
    
    //! Number of cells in x direction
    int l_nX;
    
    //! Number of cells in y direction
    int l_nY;
    
    //! l_baseName of the plots.
    std::string l_baseName;
    
    l_nY = l_nX = atoi(argv[1]);
    l_nY = atoi(argv[2]);
    l_baseName = std::string(argv[3]);    
    
    //! number of checkpoints for visualization (at each checkpoint in time, an output file is written).
    int l_numberOfCheckPoints = 20;
    if(argc == 5)
        l_numberOfCheckPoints = atoi(argv[4]);
    
    // create a simple artificial scenario
    SWE_PartialDambreak l_scenario;

    //! size of a single cell in x- and y-direction
    float l_dX, l_dY;
    
    // compute the size of a single cell
    l_dX = (l_scenario.getBoundaryPos(BND_RIGHT) - l_scenario.getBoundaryPos(BND_LEFT) )/l_nX;
    l_dY = (l_scenario.getBoundaryPos(BND_TOP) - l_scenario.getBoundaryPos(BND_BOTTOM) )/l_nY;
    
    SWE_DimensionalSplitting l_dimensionalSplitting(l_nX, l_nY, l_dX, l_dY);
    
    //! origin of the simulation domain in x- and y-direction
    float l_originX, l_originY;

    // get the origin from the scenario
    l_originX = l_scenario.getBoundaryPos(BND_LEFT);
    l_originY = l_scenario.getBoundaryPos(BND_BOTTOM);

    // initialize the wave propagation block
    l_dimensionalSplitting.initScenario(l_originX, l_originY, l_scenario);
    
    //! time when the simulation ends.
    float l_endSimulation = l_scenario.endSimulation();
    
    //! checkpoints when output files are written.
    float* l_checkPoints = new float[l_numberOfCheckPoints+1];
    
    // compute the checkpoints in time
    for(int cp = 0; cp <= l_numberOfCheckPoints; cp++) {
        l_checkPoints[cp] = cp*(l_endSimulation/l_numberOfCheckPoints);
    }
    
    // Init fancy progressbar
    tools::ProgressBar progressBar(l_endSimulation);
    
    // write the output at time zero
    tools::Logger::logger.printOutputTime((float) 0.);
    progressBar.update(0.);
    
    std::string l_fileName = generateBaseFileName(l_baseName,0,0);
    //boundary size of the ghost layers
    io::BoundarySize l_boundarySize = {{1, 1, 1, 1}};
  #ifdef WRITENETCDF
    //construct a NetCdfWriter
    io::NetCdfWriter l_writer( l_fileName,
        l_dimensionalSplitting.getBathymetry(),
  		l_boundarySize,
  		l_nX, l_nY,
  		l_dX, l_dY,
  		l_originX, l_originY);
  #else
    // consturct a VtkWriter
    io::VtkWriter l_writer( l_fileName,
  		l_dimensionalSplitting.getBathymetry(),
  		l_boundarySize,
  		l_nX, l_nY,
  		l_dX, l_dY );
  #endif
    // Write zero time step
    l_writer.writeTimeStep( l_dimensionalSplitting.getWaterHeight(),
                                l_dimensionalSplitting.getDischarge_hu(),
                                l_dimensionalSplitting.getDischarge_hv(),
                                (float) 0.);
    
    /**
     * Simulation.
     */
    // print the start message and reset the wall clock time
    progressBar.clear();
    tools::Logger::logger.printStartMessage();
    tools::Logger::logger.initWallClockTime(time(NULL));
    
    //! simulation time.
    float l_t = 0.0;
    progressBar.update(l_t);
    
    unsigned int l_iterations = 0;
    
    // loop over checkpoints
    for(int c = 1; c <= l_numberOfCheckPoints; c++) {
        
        // do time steps until next checkpoint is reached
        while( l_t < l_checkPoints[c] ) {
            // set values in ghost cells:
            l_dimensionalSplitting.setGhostLayer();
            
            // reset the cpu clock
            tools::Logger::logger.resetCpuClockToCurrentTime();
            
            // approximate the maximum time step
            // TODO: This calculation should be replaced by the usage of the wave speeds occuring during the flux computation
            // Remark: The code is executed on the CPU, therefore a "valid result" depends on the CPU-GPU-synchronization.
  //        _wavePropgationBlock.computeMaxTimestep();
            
            // compute numerical flux on each edge
            l_dimensionalSplitting.computeNumericalFluxes();
            
            //! maximum allowed time step width.
            float l_maxTimeStepWidth = l_dimensionalSplitting.getMaxTimestep();
            
            // update the cell values
            l_dimensionalSplitting.updateUnknowns(l_maxTimeStepWidth);
            
            // update the cpu time in the logger
            tools::Logger::logger.updateCpuTime();
            
            // update simulation time with time step width.
            l_t += l_maxTimeStepWidth;
            l_iterations++;
            
            // print the current simulation time
            progressBar.clear();
            tools::Logger::logger.printSimulationTime(l_t);
            progressBar.update(l_t);
        }
        // print current simulation time of the output
        progressBar.clear();
        tools::Logger::logger.printOutputTime(l_t);
        progressBar.update(l_t);
        
        // write output
        l_writer.writeTimeStep( l_dimensionalSplitting.getWaterHeight(),
                              l_dimensionalSplitting.getDischarge_hu(),
                              l_dimensionalSplitting.getDischarge_hv(),
                              l_t);
    }
    
    /**
     * Finalize.
     */
    // write the statistics message
    progressBar.clear();
    tools::Logger::logger.printStatisticsMessage();
    
    // print the cpu time
    tools::Logger::logger.printCpuTime();
    
    // print the wall clock time (includes plotting)
    tools::Logger::logger.printWallClockTime(time(NULL));
    
    // printer iteration counter
    tools::Logger::logger.printIterationsDone(l_iterations);
    
    return 0;
}

