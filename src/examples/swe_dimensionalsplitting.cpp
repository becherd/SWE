#include <iostream>
#include <unistd.h>

#include "blocks/SWE_DimensionalSplitting.hh"
#include "scenarios/SWE_Scenario.hh"
#include "scenarios/SWE_PartialDambreak.hh"
#include "scenarios/SWE_TsunamiScenario.hh"
#include "scenarios/SWE_ArtificialTsunamiScenario.hh"

#ifdef WRITENETCDF
#include "writer/NetCdfWriter.hh"
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
    BoundaryType l_boundaryConditions[4];
    //! whether to override the scenario-defined conditions (1) or not (0)
    int l_overwriteBoundaryConditions = 0;
    
    //! List of defined scenarios
    typedef enum {
        SCENARIO_TSUNAMI, SCENARIO_CHECKPOINT_TSUNAMI, 
        SCENARIO_ARTIFICIAL_TSUNAMI, SCENARIO_PARTIAL_DAMBREAK
    } ScenarioName;
    
    //! the name of the chosen scenario
    ScenarioName l_scenarioName = SCENARIO_TSUNAMI;
    
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
    while ((c = getopt(argc, argv, "x:y:o:i:d:c:n:t:b:s:")) != -1) {
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
            case 'i':
                l_bathymetryFileName = std::string(optarg);
                break;
            case 'd':
                l_displacementFileName = std::string(optarg);
                break;
            case 'c':
                l_checkpointFileName = std::string(optarg);
                break;
            case 'n':
                l_numberOfCheckPoints = atoi(optarg);
                break;
            case 't':
                l_simulationTime = atof(optarg);
                break;
            case 'b':
                optstr = std::string(optarg);
                switch(optstr.length()) {
                    case 1:
                        // one option for all boundaries
                        for(int i = 0; i < 4; i++)
                            l_boundaryConditions[i] = (optstr[0] == 'w') ? WALL : OUTFLOW;
                        break;
                    case 2:
                        // first: left/right, second: top/bottom
                        for(int i = 0; i < 2; i++)
                            l_boundaryConditions[i] = (optstr[0] == 'w') ? WALL : OUTFLOW;
                        for(int i = 2; i < 4; i++)
                            l_boundaryConditions[i] = (optstr[1] == 'w') ? WALL : OUTFLOW;
                        break;
                    case 4:
                        // left right bottom top
                        for(int i = 0; i < 4; i++)
                            l_boundaryConditions[i] = (optstr[i] == 'w') ? WALL : OUTFLOW;
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
            default:
                showUsage = 1;
                break;
        }
    }
    
    // Do several checks on supplied options
    if(!showUsage) {
        // Check for required arguments x and y cells
        if(l_nX == 0 || l_nY == 0) {
            std::cerr << "Missing required arguments: number of cells in X (-x) and Y (-y) direction" << std::endl;
            showUsage = 1;
        }
        // Check for required output base file name
        if(l_baseName.empty()) {
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
        }
        
        if(l_scenarioName == SCENARIO_TSUNAMI) {
            // We've got no checkpoint and no artificial scenario
            // => Bathymetry and displacement data must be supplied
            if(l_bathymetryFileName.empty() || l_displacementFileName.empty()) {
                std::cerr << "Missing required argument: bathymetry (-i) and displacement (-d) files must be supplied" << std::endl;
                showUsage = 1;
            }
        }
    }
    
    if(showUsage) {
        std::cout << "Usage: ./SWE_<compiler>_<build>_none_dimsplit [OPTIONS]" << std::endl;
        std::cout << "  Available Options:" << std::endl;
        std::cout << "  REQUIRED:" << std::endl;
        std::cout << "    -x <num>      Number of grid cells in x-direction" << std::endl;
        std::cout << "    -y <num>      Number of grid cells in y-direction" << std::endl;
        std::cout << "    -o <file>     Base name of the output file(s)" << std::endl;
        std::cout << "  OPTIONAL:" << std::endl;
        std::cout << "    -i <file>     Name of bathymetry data file" << std::endl;
        std::cout << "    -d <file>     Name of displacement data file" << std::endl;
        std::cout << "    -c <file>     Name of checkpoint file to continue simulation" << std::endl;
        std::cout << "    -t <float>    Simulation time in seconds" << std::endl;
        std::cout << "    -n <num>      Number of checkpoints to be written" << std::endl;
        std::cout << "    -s <scenario> Name of artificial scenario" << std::endl;
        std::cout << "                  Scenarios: 'artificialtsunami', 'partialdambreak'" << std::endl;
        std::cout << "    -b <code>     Boundary Conditions" << std::endl;
        std::cout << "                  Codes: Combination of 'w' (WALL) and 'o' (OUTFLOW)" << std::endl;
        std::cout << "                    One char: Option for ALL boundaries" << std::endl;
        std::cout << "                    Two chars: Options for left/right and top/bottom boundaries" << std::endl;
        std::cout << "                    Four chars: Options for left, right, bottom, top boundaries" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "Example: " << std::endl; 
        std::cout << "./SWE_<compiler>_<build>_none_dimsplit -x 100 -y 200 -o out -i b.nc -d d.nc -n 50 -b owwo" << std::endl;
        std::cout << "    will simulate a tsunami scenario using bathymetry from 'b.nc' and displacements ";
        std::cout << "from 'd.nc' on a grid of size 100 x 200 using outflow conditions for left and ";
        std::cout << "top boundary and wall conditions for right and bottom boundary, writing 50 checkpoints ";
        std::cout << "to out_<num>.<ext>" << std::endl;
        
        return 0;
    }
    
    //! Pointer to instance of chosen scenario
    SWE_Scenario *l_scenario;
    
    // Create scenario according to chosen options
    switch(l_scenarioName) {
        case SCENARIO_TSUNAMI:
            // overwrite boundary conditions from scenario in case they have 
            // been explicitly set using command line arguments
            if(!l_overwriteBoundaryConditions) {
                l_scenario = new SWE_TsunamiScenario(l_bathymetryFileName, l_displacementFileName);
            } else {
                l_scenario = new SWE_TsunamiScenario(l_bathymetryFileName, l_displacementFileName, l_boundaryConditions);
            }
            break;
        case SCENARIO_CHECKPOINT_TSUNAMI:
            // TODO: Implement checkpointed tsunami
            std::cerr << "Checkpointed Tsunami is not implemented yet :(" << std::endl;
            abort();
            break;
        case SCENARIO_ARTIFICIAL_TSUNAMI:
            // overwrite boundary conditions from scenario in case they have 
            // been explicitly set using command line arguments
            if(!l_overwriteBoundaryConditions) {
                l_scenario = new SWE_ArtificialTsunamiScenario();
            } else {
                l_scenario = new SWE_ArtificialTsunamiScenario(l_boundaryConditions);
            }
            break;
        case SCENARIO_PARTIAL_DAMBREAK:
            l_scenario = new SWE_PartialDambreak();
            if(l_overwriteBoundaryConditions) {
                std::cerr << "WARNING: PartialDambreak-Scenario does not support "
                          << "explicitly setting boundary conditions" << std::endl;
            }
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
    
    std::string l_outputFileName = generateBaseFileName(l_baseName,0,0);
    //boundary size of the ghost layers
    io::BoundarySize l_boundarySize = {{1, 1, 1, 1}};
#ifdef WRITENETCDF
    //construct a NetCdfWriter
    io::NetCdfWriter l_writer( l_outputFileName,
        l_dimensionalSplitting.getBathymetry(),
  		l_boundarySize,
  		l_nX, l_nY,
  		l_dX, l_dY,
  		l_originX, l_originY);
#else
    // consturct a VtkWriter
    io::VtkWriter l_writer( l_outputFileName,
  		l_dimensionalSplitting.getBathymetry(),
  		l_boundarySize,
  		l_nX, l_nY,
  		l_dX, l_dY );
#endif
    // Write zero time step
    l_writer.writeTimeStep( l_dimensionalSplitting.getWaterHeight(),
                                l_dimensionalSplitting.getDischarge_hu(),
                                l_dimensionalSplitting.getDischarge_hv(), 
                                (float) 0.,
								l_numberOfCheckPoints,
								l_endSimulation);
    
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
    
    // free scenario object
    delete l_scenario;
    
    return 0;
}
