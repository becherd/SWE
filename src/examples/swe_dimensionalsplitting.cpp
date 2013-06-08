#include <iostream>
#include <unistd.h>

#include "blocks/SWE_DimensionalSplitting.hh"
#include "scenarios/SWE_Scenario.hh"
#include "scenarios/SWE_PartialDambreak.hh"
#include "scenarios/SWE_ArtificialTsunamiScenario.hh"

#ifdef WRITENETCDF
#include "writer/NetCdfWriter.hh"

#include "scenarios/SWE_TsunamiScenario.hh"
#include "scenarios/SWE_CheckpointTsunamiScenario.hh"
#else
#include "writer/VtkWriter.hh"
#endif

#include "tools/help.hh"
#include "tools/Logger.hh"
#include "tools/ProgressBar.hh"

int main( int argc, char** argv ) {
    
    //! Number of cells in x direction
    int l_nX = 0;
    
    //! Number of cells in y direction
    int l_nY = 0;

	//! coarseness factor
	float l_coarseness = 1.0;
    
    //! l_baseName of the plots.
    std::string l_baseName;
    
    //! bathymetry input file name
    std::string l_bathymetryFileName;
    
    //! displacement input file name
    std::string l_displacementFileName;
    
    //! checkpoint input file name
    std::string l_checkpointFileName;
    
    //! the total simulation time
    int l_simulationTime = 0.0;
    
    //! type of boundary conditions at LEFT, RIGHT, TOP, and BOTTOM boundary
    BoundaryType l_boundaryTypes[4];
    //! whether to override the scenario-defined conditions (true) or not (false)
    bool l_overwriteBoundaryTypes = false;
    
    //! List of defined scenarios
    typedef enum {
        SCENARIO_TSUNAMI, SCENARIO_CHECKPOINT_TSUNAMI,
        SCENARIO_ARTIFICIAL_TSUNAMI, SCENARIO_PARTIAL_DAMBREAK
    } ScenarioName;
    
    //! the name of the chosen scenario
    ScenarioName l_scenarioName;
#ifdef WRITENETCDF
    l_scenarioName = SCENARIO_TSUNAMI;
#else
    l_scenarioName = SCENARIO_PARTIAL_DAMBREAK;
#endif
    
    //! number of checkpoints for visualization (at each checkpoint in time, an output file is written).
    int l_numberOfCheckPoints = 20;
    
    // Option Parsing
    // REQUIRED
    // -x <num>        // Number of cells in x-dir
    // -y <num>        // Number of cells in y-dir
    // -o <file>       // Output file basename
    // OPTIONAL (may be required for certain scenarios)
    // -i <file>       // initial bathymetry data file name (REQUIRED for certain scenarios)
    // -d <file>       // input displacement data file name (REQUIRED for certain scenarios)
    // -c <file>       // checkpoints data file name
    // -f <float>      // output coarseness factor
    // -n <num>        // Number of checkpoints
    // -t <float>      // Simulation time in seconds
    // -s <scenario>   // Artificial scenario name ("artificialtsunami", "partialdambreak")
    // -b <code>       // Boundary conditions, "w" or "o"
    //                 // 1 value: for all
    //                 // 2 values: first is left/right, second is top/bottom
    //                 // 4 values: left, right, bottom, top
    int c;
    int showUsage = 0;
    std::string optstr;
    while ((c = getopt(argc, argv, "x:y:o:i:d:c:n:t:b:s:f:")) != -1) {
        switch(c) {
            case 'x':
                l_nX = atoi(optarg);
                break;
            case 'y':
                l_nY = atoi(optarg);
                break;
            case 'o':
                l_baseName = std::string(optarg);
                break;
#ifdef WRITENETCDF
            case 'i':
                l_bathymetryFileName = std::string(optarg);
                break;
            case 'd':
                l_displacementFileName = std::string(optarg);
                break;
            case 'c':
                l_checkpointFileName = std::string(optarg);
                break;
#endif
            case 'n':
                l_numberOfCheckPoints = atoi(optarg);
                break;
            case 't':
                l_simulationTime = atof(optarg);
                break;
            case 'b':
                optstr = std::string(optarg);
                l_overwriteBoundaryTypes = true;
                switch(optstr.length()) {
                    case 1:
                        // one option for all boundaries
                        for(int i = 0; i < 4; i++)
                            l_boundaryTypes[i] = (optstr[0] == 'w') ? WALL : OUTFLOW;
                        break;
                    case 2:
                        // first: left/right, second: top/bottom
                        for(int i = 0; i < 2; i++)
                            l_boundaryTypes[i] = (optstr[0] == 'w') ? WALL : OUTFLOW;
                        for(int i = 2; i < 4; i++)
                            l_boundaryTypes[i] = (optstr[1] == 'w') ? WALL : OUTFLOW;
                        break;
                    case 4:
                        // left right bottom top
                        for(int i = 0; i < 4; i++)
                            l_boundaryTypes[i] = (optstr[i] == 'w') ? WALL : OUTFLOW;
                        break;
                    default:
                        std::cerr << "Invalid option argument: Invalid boundary specification (-b)" << std::endl;
                        showUsage = 1;
                        break;
                }
                break;
            case 's':
                optstr = std::string(optarg);
                if(optstr == "artificialtsunami") {
                    l_scenarioName = SCENARIO_ARTIFICIAL_TSUNAMI;
                } else if(optstr == "partialdambreak") {
                    l_scenarioName = SCENARIO_PARTIAL_DAMBREAK;
                } else {
                    std::cerr << "Invalid option argument: Unknown scenario (-s)" << std::endl;
                    showUsage = 1;
                }
                break;
            case 'f':
                l_coarseness = atof(optarg);
                break;
            default:
                showUsage = 1;
                break;
        }
    }
    
    // Do several checks on supplied options
    if(!showUsage) {
        // Check for required arguments x and y cells unless we can get the info from a checkpoint file
        if((l_nX == 0 || l_nY == 0) && l_checkpointFileName.empty()) {
            std::cerr << "Missing required arguments: number of cells in X (-x) and Y (-y) direction" << std::endl;
            showUsage = 1;
        }
        // Check for required output base file name
        if(l_baseName.empty() && l_checkpointFileName.empty()) {
            std::cerr << "Missing required argument: base name of output file (-o)" << std::endl;
            showUsage = 1;
        }
        // Check for valid number of checkpoints
        if(l_numberOfCheckPoints <= 0) {
            std::cerr << "Invalid option argument: Number of checkpoints must be greater than zero (-n)" << std::endl;
            showUsage = 1;
        }
        // Check if a checkpoint-file is given as input. If so, switch to checkpoint scenario
        if(!l_checkpointFileName.empty()) {
            l_scenarioName = SCENARIO_CHECKPOINT_TSUNAMI;
            
            // We handle the file name of checkpoint and output data files without the ".nc"
            // extension internally, so we're removing the extension here in case it is supplied
            int cpLength = l_checkpointFileName.length();
            if(l_checkpointFileName.substr(cpLength-3, 3).compare(".nc") == 0) {
                l_checkpointFileName.erase(cpLength-3, 3);
            }
            
            if(l_nX > 0 || l_nY > 0)
                std::cerr << "WARNING: Supplied number of grid cells will be ignored (reading from checkpoint)" << std::endl;
            if(l_simulationTime > 0.0)
                std::cerr << "WARNING: Supplied simulation time will be ignored (reading from checkpoint)" << std::endl;
        }
        
        if(l_scenarioName == SCENARIO_TSUNAMI) {
            // We've got no checkpoint and no artificial scenario
            // => Bathymetry and displacement data must be supplied
            if(l_bathymetryFileName.empty() || l_displacementFileName.empty()) {
                std::cerr << "Missing required argument: bathymetry (-i) and displacement (-d) files must be supplied" << std::endl;
                showUsage = 1;
            }
        } else {
            if(!l_bathymetryFileName.empty() || !l_displacementFileName.empty())
                std::cerr << "WARNING: Supplied bathymetry and displacement data will be ignored" << std::endl;
        }
    }
    
    if(showUsage) {
        std::cout << "Usage:" << std::endl;
        std::cout << "Simulating a tsunami with bathymetry and displacement input:" << std::endl;
        std::cout << "    ./SWE_<opt> -i <bathymetryfile> -d <displacementfile> [OPTIONS]" << std::endl;
        std::cout << "Resuming a crashed simulation from checkpoint file:" << std::endl;
        std::cout << "    ./SWE_<opt> -c <checkpointfile> [-o <outputfile>]" << std::endl;
        std::cout << "Simulating an artificial scenario:" << std::endl;
        std::cout << "    ./SWE_<opt> -s <scenarioname> [OPTIONS]" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "    -o <filename>   The output file base name" << std::endl;
        std::cout << "        Note: If the file already exists it is assumed to be a checkpointfile" << std::endl;
        std::cout << "        from which to resume simulation. Input options are ignored then." << std::endl;
        std::cout << "    -x <num>        The number of cells in x-direction" << std::endl;
        std::cout << "    -y <num>        The number of cells in y-direction" << std::endl;
        std::cout << "    -n <num>        Number of checkpoints to be written" << std::endl;
        std::cout << "    -t <time>       Total simulation time" << std::endl;
        std::cout << "    -f <num>        Coarseness factor" << std::endl;
        std::cout << "    -b <code>       Boundary Conditions" << std::endl;
        std::cout << "                    Codes: Combination of 'w' (WALL) and 'o' (OUTFLOW)" << std::endl;
        std::cout << "                      One char: Option for ALL boundaries" << std::endl;
        std::cout << "                      Two chars: Options for left/right and top/bottom boundaries" << std::endl;
        std::cout << "                      Four chars: Options for left, right, bottom, top boundaries" << std::endl;
        std::cout << "    -i <filename>   Name of bathymetry data file" << std::endl;
        std::cout << "    -d <filename>   Name of displacement data file" << std::endl;
        std::cout << "    -c <filename>   Name of checkpointfile" << std::endl;
        std::cout << "    -s <scenario>   Name of artificial scenario" << std::endl;
        std::cout << "                    Scenarios: 'artificialtsunami', 'partialdambreak'" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "Notes when using a checkpointfile:" << std::endl;
        std::cout << "    -x, -y, -n, -t, -b, -i, -d, -s are ignored (values are read from checkpointfile)" << std::endl;
        std::cout << "    An output file (-o) can be specified. In that case, the checkpointfile" << std::endl;
        std::cout << "    is copied to that location and output is appended to the output file." << std::endl;
        std::cout << "    If no output file is specified, output is appended to the checkpointfile." << std::endl;
        std::cout << "" << std::endl;
        std::cout << "Example: " << std::endl; 
        std::cout << "./SWE_<compiler>_<build>_none_dimsplit -x 100 -y 200 -o out -i b.nc -d d.nc -n 50 -b owwo" << std::endl;
        std::cout << "    will simulate a tsunami scenario using bathymetry from 'b.nc' and displacements ";
        std::cout << "from 'd.nc' on a grid of size 100 x 200 using outflow conditions for left and ";
        std::cout << "top boundary and wall conditions for right and bottom boundary, writing 50 checkpoints ";
        std::cout << "to out_<num>.nc" << std::endl;
        
        return 0;
    }
    
    //! output file basename (with block coordinates)
    std::string l_outputFileName = generateBaseFileName(l_baseName,0,0);
    
#ifdef WRITENETCDF
    if(l_scenarioName != SCENARIO_CHECKPOINT_TSUNAMI) {
        // This is a tsunami scenario, check if the output file (with .nc-extension) exists
        // In that case switch to checkpoint scenario
        int ncOutputFile;
        int status = nc_open((l_outputFileName + ".nc").c_str(), NC_NOWRITE, &ncOutputFile);
        if(status == NC_NOERR) {
            // Output file exists and is a NetCDF file => switch to checkpointing
            l_scenarioName = SCENARIO_CHECKPOINT_TSUNAMI;
            l_checkpointFileName = l_outputFileName;
            nc_close(ncOutputFile);
        }
    }
#endif
    
    //! Pointer to instance of chosen scenario
    SWE_Scenario *l_scenario;
    
    // Create scenario according to chosen options
    switch(l_scenarioName) {
#ifdef WRITENETCDF
        case SCENARIO_TSUNAMI:
            l_scenario = new SWE_TsunamiScenario(l_bathymetryFileName, l_displacementFileName);
            
            // overwrite boundary conditions from scenario in case they have 
            // been explicitly set using command line arguments
            if(l_overwriteBoundaryTypes)
                ((SWE_TsunamiScenario *)l_scenario)->setBoundaryTypes(l_boundaryTypes);
            break;
        case SCENARIO_CHECKPOINT_TSUNAMI:
            l_scenario = new SWE_CheckpointTsunamiScenario(l_checkpointFileName + ".nc");
            
            // Read number if grid cells from checkpoint
            ((SWE_CheckpointTsunamiScenario *)l_scenario)->getNumberOfCells(l_nX, l_nY);
            
            if(l_overwriteBoundaryTypes)
                std::cerr << "WARNING: Loading checkpointed Simulation does not support "
                          << "explicitly setting boundary conditions" << std::endl;
            break;
#endif
        case SCENARIO_ARTIFICIAL_TSUNAMI:
            l_scenario = new SWE_ArtificialTsunamiScenario();

            // overwrite boundary conditions from scenario in case they have 
            // been explicitly set using command line arguments
            if(l_overwriteBoundaryTypes)
                ((SWE_ArtificialTsunamiScenario *)l_scenario)->setBoundaryTypes(l_boundaryTypes);
            break;
        case SCENARIO_PARTIAL_DAMBREAK:
            l_scenario = new SWE_PartialDambreak();
            if(l_overwriteBoundaryTypes)
                std::cerr << "WARNING: PartialDambreak-Scenario does not support "
                          << "explicitly setting boundary conditions" << std::endl;
            break;
        default:
            std::cerr << "Invalid Scenario" << std::endl;
            exit(1);
            break;
    }

    //! size of a single cell in x- and y-direction
    float l_dX, l_dY;
    
    // compute the size of a single cell
    l_dX = (l_scenario->getBoundaryPos(BND_RIGHT) - l_scenario->getBoundaryPos(BND_LEFT) )/l_nX;
    l_dY = (l_scenario->getBoundaryPos(BND_TOP) - l_scenario->getBoundaryPos(BND_BOTTOM) )/l_nY;
    
    //! Dimensional Splitting Block
    SWE_DimensionalSplitting l_dimensionalSplitting(l_nX, l_nY, l_dX, l_dY);
    
    //! origin of the simulation domain in x- and y-direction
    float l_originX, l_originY;

    // get the origin from the scenario
    l_originX = l_scenario->getBoundaryPos(BND_LEFT);
    l_originY = l_scenario->getBoundaryPos(BND_BOTTOM);

    // initialize the wave propagation block
    l_dimensionalSplitting.initScenario(l_originX, l_originY, *l_scenario);
    
    //! time when the simulation ends.
    float l_endSimulation;
    if(l_simulationTime <= 0.0) {
        // We haven't got a valid simulation time as arguments, use the pre-defied one from scenario
        l_endSimulation = l_scenario->endSimulation();
    } else {
        // Use time given from command line
        l_endSimulation = l_simulationTime;
    }
    
    //! simulation time.
    float l_t = 0.0;
    
    //! checkpoint counter
    int l_checkpoint = 1;

#ifdef WRITENETCDF
    if(l_scenarioName == SCENARIO_CHECKPOINT_TSUNAMI) {
        // read total number of checkpoints
        l_numberOfCheckPoints = ((SWE_CheckpointTsunamiScenario *)l_scenario)->getNumberOfCheckpoints();
        
        // load last checkpoint and timestep from scenario (checkpoint-file)
        ((SWE_CheckpointTsunamiScenario *)l_scenario)->getLastCheckpoint(l_checkpoint, l_t);
        l_checkpoint++;
        
        // forace coarseness of 1 if reading from checkpoint data
        l_coarseness = 1.0;
    }
#endif
    
    // read actual boundary types (command line merged with scenario)
    l_boundaryTypes[BND_LEFT] = l_scenario->getBoundaryType(BND_LEFT);
    l_boundaryTypes[BND_RIGHT] = l_scenario->getBoundaryType(BND_RIGHT);
    l_boundaryTypes[BND_BOTTOM] = l_scenario->getBoundaryType(BND_BOTTOM);
    l_boundaryTypes[BND_TOP] = l_scenario->getBoundaryType(BND_TOP);
    
    //! checkpoints when output files are written.
    float* l_checkPoints = new float[l_numberOfCheckPoints+1];
    
    // compute the checkpoints in time
    for(int cp = 0; cp <= l_numberOfCheckPoints; cp++) {
        l_checkPoints[cp] = cp*(l_endSimulation/l_numberOfCheckPoints);
    }
    
    // Init fancy progressbar
    tools::ProgressBar progressBar(l_endSimulation);
    
    // write the output at time zero
    tools::Logger::logger.printOutputTime((float) l_t);
    progressBar.update(l_t);
    
    //boundary size of the ghost layers
    io::BoundarySize l_boundarySize = {{1, 1, 1, 1}};
    
    // Delete scenarioto free resources and close opened files
    delete l_scenario;
    
#ifdef WRITENETCDF
    if(l_scenarioName == SCENARIO_CHECKPOINT_TSUNAMI) {
        if(l_baseName.empty()) {
            // If there is no output file name given, use the checkpoint file
            l_outputFileName = l_checkpointFileName;
        } else if(l_outputFileName.compare(l_checkpointFileName) != 0) {
            // output file name given and it is not equal to the checkpoint file
            // therefore, we have to make a copy of our checkpointfile
            // in order to continue the simulation
            std::ifstream src((l_checkpointFileName + ".nc").c_str());
            std::ofstream dst((l_outputFileName + ".nc").c_str());
            dst << src.rdbuf();
        }
    }

    //construct a NetCdfWriter
    io::NetCdfWriter l_writer( l_outputFileName,
        l_dimensionalSplitting.getBathymetry(),
  		l_boundarySize,
  		l_nX, l_nY,
  		l_dX, l_dY,
  		l_originX, l_originY,
        l_coarseness);
        
        l_writer.writeSimulationInfo(l_numberOfCheckPoints, l_endSimulation, l_boundaryTypes);
#else
    // consturct a VtkWriter
    io::VtkWriter l_writer( l_outputFileName,
  		l_dimensionalSplitting.getBathymetry(),
  		l_boundarySize,
  		l_nX, l_nY,
  		l_dX, l_dY,
        0, 0,
        l_coarseness);
#endif
    if(l_scenarioName != SCENARIO_CHECKPOINT_TSUNAMI) {
        // Write zero time step
        l_writer.writeTimeStep( l_dimensionalSplitting.getWaterHeight(),
                                l_dimensionalSplitting.getDischarge_hu(),
                                l_dimensionalSplitting.getDischarge_hv(), 
                                (float) 0.);
    }
        
    /**
     * Simulation.
     */
    // print the start message and reset the wall clock time
    progressBar.clear();
    tools::Logger::logger.printStartMessage();
    tools::Logger::logger.initWallClockTime(time(NULL));
    
    progressBar.update(l_t);
    
    unsigned int l_iterations = 0;
    
    // loop over checkpoints
    while(l_checkpoint <= l_numberOfCheckPoints) {
        
        // do time steps until next checkpoint is reached
        while( l_t < l_checkPoints[l_checkpoint] ) {
            // set values in ghost cells:
            l_dimensionalSplitting.setGhostLayer();
            
            // reset the cpu clock
            tools::Logger::logger.resetCpuClockToCurrentTime();
            
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
        
        l_checkpoint++;
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
    
    // print average time per cell per iteration
    tools::Logger::logger.printAverageCPUTimePerCellPerIteration(l_iterations, l_nX*(l_nY+2)); 
    
    return 0;
}
