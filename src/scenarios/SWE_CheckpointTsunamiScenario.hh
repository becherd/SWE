#ifndef __SWE_CHECKPOINTTSUNAMISCENARIO_HH
#define __SWE_CHECKPOINTTSUNAMISCENARIO_HH

#include <iostream>
#include <cassert>

#include <netcdf.h>

#include "SWE_Scenario.hh"

/**
 * Scenario "Checkpoint Tsunami"
 *
 * A Scenario to load a netCDF file with checkpoint data from a previous, unsuccessful run to resume the simulation
 *
 */
class SWE_CheckpointTsunamiScenario : public SWE_Scenario {

protected:
    //! numerical tolerance used in certain comparisons
    const static float tolerance = 1e-10;
    
    //! The NetCDF checkpoint file ID
    int file_id;
    //! The NetCDF dimension x ID
    int x_id;
    //! The NetCDF dimension y ID
    int y_id;
    //! The NetCDF height ID
    int h_id;
    //! The NetCDF hu ID
    int hu_id;
    //! The NetCDF hv ID
    int hv_id;
    //! The NetCDF b(x,y) ID
    int b_id;
    //! The NetCDF time ID
    int time_id;
    
    //! The NetCDF length of x dimension
    size_t x_len;
    //! The NetCDF length of y dimension
    size_t y_len;
    //! The NetCDF length of time dimension
    size_t time_len;
    
    //! The NetCDF minimum value in x dimension
    float x_min;
    //! The NetCDF maximum value in x dimension
    float x_max;
    //! The NetCDF step width in x dimension (step width between two cells)
    float x_step;
    //! The NetCDF minimum value in y dimension
    float y_min;
    //! The NetCDF maximum value in y dimension
    float y_max;
    //! The NetCDF step width in y dimension (step width between two cells)
    float y_step;
    
    /**
     * Load the checkpoint file
     *
     * @param checkpointFileName The file name of the checkpoint data file
     */
    void loadInputFiles(std::string checkpointFileName) {
        // We can store the return values of the netCDF methods here
        int retval;
        
        // Index for adressing 1D-variable values
        size_t index[1];
        
        /**
         * Load the bathymetry file
         */
        // Open the file
        retval = nc_open(checkpointFileName.c_str(), NC_NOWRITE, &file_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        // Read ID for time, x, y, h, hu, hv and b variables
        retval = nc_inq_unlimdim(file_id, &time_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_dimid(file_id, "x", &x_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_dimid(file_id, "y", &y_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_varid(file_id, "h", &h_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_varid(file_id, "hu", &hu_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_varid(file_id, "hv", &hv_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_varid(file_id, "b", &b_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        // Read Dimensions for x and y variable
        retval = nc_inq_dimlen(file_id, x_id, &x_len);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_dimlen(file_id, y_id, &y_len);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_dimlen(file_id, time_id, &time_len);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        // We should have more than a single cell
        assert(x_len >= 2);
        assert(y_len >= 2);
        
        // And we should have at least a single timestep
        assert(time_len >= 1);
        
        // Read minimum value for x and y variable
        // Note: We assume the x and y dimensions are stored in ascending order
        index[0] = 0;
        nc_get_var1_float(file_id, x_id, (const size_t *)index, &x_min);
        nc_get_var1_float(file_id, y_id, (const size_t *)index, &y_min);
        
        // Read maximum value for x and y variable
        index[0] = x_len - 1;
        nc_get_var1_float(file_id, x_id, (const size_t *)index, &x_max);
        index[0] = y_len - 1;
        nc_get_var1_float(file_id, y_id, (const size_t *)index, &y_max);
        
        // Calculate step width (cell size) for x and y variable
        // Note: We assume the step width remains constaint over the complete domain,
        // which is true since we use equally spaced cells in our simulation
        x_step = (x_max - x_min) / (x_len - 1);
        y_step = (y_max - y_min) / (y_len - 1);
        
        // Step size should be greater than zero
        assert(x_step > 0.0); assert(y_step > 0.0);
    }
    
    void handleNetCDFError(int status) {
        std::cerr << "NetCDF Error: " << nc_strerror(status) << std::endl;
        exit(status);
    }
    
    size_t getIndex1D(float relativePosition, float stepWidth, size_t length) {
        size_t index;
        
        if(relativePosition >= tolerance) {
            index = static_cast <size_t> (std::floor(relativePosition / stepWidth));
            
            // make sure the index stays inside variable index bounds
            if(index >= length)
                index = length-1;
        } else {
            // requested coordinate is below our lower domain bound
            index = 0;
        }
        return index;
    }
    
    void getIndex(float x, float y, size_t index[2]) {
        // Y index
        index[0] = getIndex1D(y-getBoundaryPos(BND_BOTTOM), y_step, y_len);
        // X index
        index[1] = getIndex1D(x-getBoundaryPos(BND_LEFT), x_step, x_len);
    }
    
    float readFloatValue(int varid, float x, float y, int isTimeDependend = 1) {
        // Index array for reading values from NetCDF
        int indexLength = isTimeDependend ? 3 : 2;
        
        size_t index[indexLength];
        if(isTimeDependend) {
            getIndex(x, y, (index+1));
            // We want the value at the latest time step
            index[0] = time_len - 1; 
        } else {
            getIndex(x, y, index);
        }
        
        float value;
        int status = nc_get_var1_float(file_id, varid, (const size_t *)index, &value);
        if(status != NC_NOERR) handleNetCDFError(status);
        return value;
    }

public:

    SWE_CheckpointTsunamiScenario(std::string checkpointFileName)
    : SWE_Scenario() {
        loadInputFiles(checkpointFileName);
    }
    
    ~SWE_CheckpointTsunamiScenario() {
        // Close open NetCDF handle
        nc_close(file_id);
    }
    
    /**
     * @return bathymetry at pos
     */
    float getBathymetry(float x, float y) {
        return readFloatValue(b_id, x, y, 0);
    };

     /**
     * @return water height at pos
     */
    float getWaterHeight(float x, float y) {
        return readFloatValue(h_id, x, y);
    };
    
    /**
     * @return velocity in x-direction at pos
     */
    float getVeloc_u(float x, float y) {
        float height = getWaterHeight(x, y);
        if(height >= tolerance)
            return readFloatValue(hu_id, x, y) / height;
        return 0.0;
    };
   
    /**
     * @return velocity in y-direction at pos
     */
    float getVeloc_v(float x, float y) {
        float height = getWaterHeight(x, y);
        if(height >= tolerance)
            return readFloatValue(hv_id, x, y) / height;
        return 0.0;
    };
    
    void getNumberOfCells(int &nX, int &nY) {
        nX = x_len;
        nY = y_len;
    }
    
    /**
     * @return time when to end simulation
     */
    float endSimulation() {
        float endSimulation;
        int status = nc_get_att_float(file_id, NC_GLOBAL, "endSimulation", &endSimulation);
        if(status == NC_NOERR)
            return endSimulation;
        
        std::cerr << "WARNING: Checkpointfile does not contain a 'endSimulation' attribute!" << std::endl;
        return 50.0f;
    };

    /**
     * Determines the type (e.g. reflecting wall or outflow) of a certain boundary
     *
     * @param edge The boundary edge
     * @return The type of the specified boundary (e.g. OUTFLOW or WALL)
     */
    BoundaryType getBoundaryType(BoundaryEdge edge) {
        // TODO: read boundary type from netCDF file
        return OUTFLOW;
    };
    
    /** Get the boundary positions
     *
     * @param i_edge which edge
     * @return value in the corresponding dimension
     */
    float getBoundaryPos(BoundaryEdge i_edge) {
        if ( i_edge == BND_LEFT )
            return x_min - (x_step / 2);
        else if ( i_edge == BND_RIGHT)
            return x_max + (x_step / 2);
        else if ( i_edge == BND_BOTTOM )
            return y_min - (y_step / 2);
        else
            return y_max + (y_step / 2);
    };
};

#endif /* __SWE_CHECKPOINTTSUNAMISCENARIO_HH */
