#ifndef __SWE_TSUNAMISCENARIO_HH
#define __SWE_TSUNAMISCENARIO_HH

#include <iostream>
#include <cassert>

#include <netcdf.h>

#include "SWE_Scenario.hh"

/**
 * Scenario "Tsunami"
 *
 * A generic Tsunami Scenario loading bathymetry and displacement data from
 * from NetCDF files.
 */
class SWE_TsunamiScenario : public SWE_Scenario {

protected:
    //! numerical tolerance used in certain comparisons
    const static float tolerance = 1e-10;
    
    //! Boundary types for each boundary (left, right, bottom, top)
    BoundaryType boundaryTypes[4];
    
    //! The NetCDF bathymetry file ID
    int bathymetry_file_id;
    //! The NetCDF bathymetry z(x,y) ID
    int bathymetry_z_id;
    //! The NetCDF bathymetry dimension x ID
    int bathymetry_x_id;
    //! The NetCDF bathymetry dimension y ID
    int bathymetry_y_id;
    //! The NetCDF bathymetry length of x dimension
    size_t bathymetry_x_len;
    //! The NetCDF bathymetry length of y dimension
    size_t bathymetry_y_len;
    //! The NetCDF bathymetry minimum value in x dimension
    float bathymetry_x_min;
    //! The NetCDF bathymetry maximum value in x dimension
    float bathymetry_x_max;
    //! The NetCDF bathymetry step width in x dimension (step width between two cells)
    float bathymetry_x_step;
    //! The NetCDF bathymetry minimum value in y dimension
    float bathymetry_y_min;
    //! The NetCDF bathymetry maximum value in y dimension
    float bathymetry_y_max;
    //! The NetCDF bathymetry step width in y dimension (step width between two cells)
    float bathymetry_y_step;
    
    
    //! The NetCDF displacement file ID
    int displacement_file_id;
    //! The NetCDF displacement z(x,y) ID
    int displacement_z_id;
    //! The NetCDF displacement dimension x ID
    int displacement_x_id;
    //! The NetCDF displacement dimension y ID
    int displacement_y_id;
    //! The NetCDF displacement length of x dimension
    size_t displacement_x_len;
    //! The NetCDF displacement length of y dimension
    size_t displacement_y_len; 
    //! The NetCDF displacement minimum value in x dimension
    float displacement_x_min;
    //! The NetCDF displacement maximum value in x dimension
    float displacement_x_max;
    //! The NetCDF displacement step width in x dimension (step width between two cells)
    float displacement_x_step;
    //! The NetCDF displacement minimum value in y dimension
    float displacement_y_min;
    //! The NetCDF displacement maximum value in y dimension
    float displacement_y_max;
    //! The NetCDF displacement step width in y dimension (step width between two cells)
    float displacement_y_step;
    
    
    /**
     * Load both the bathymetry and displacement file
     *
     * @param bathymetryFileName The file name of the bathymetry data file
     * @param displacementFileName The file name of the displacement data file
     */
    void loadInputFiles(std::string bathymetryFileName, std::string displacementFileName) {
        // We can store the return values of the netCDF methods here
        int retval;
        
        // Index for adressing 1D-variable values
        size_t index[1];
        
        /**
         * Load the bathymetry file
         */
        // Open the file
        retval = nc_open(bathymetryFileName.c_str(), NC_NOWRITE, &bathymetry_file_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        // Read ID for x and y and z variables
        retval = nc_inq_dimid(bathymetry_file_id, "x", &bathymetry_x_id);    
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_dimid(bathymetry_file_id, "y", &bathymetry_y_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_varid(bathymetry_file_id, "z", &bathymetry_z_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        // Read Dimensions for x and y variable
        retval = nc_inq_dimlen(bathymetry_file_id, bathymetry_x_id, &bathymetry_x_len);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_dimlen(bathymetry_file_id, bathymetry_y_id, &bathymetry_y_len);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        // We should have more than a single cell
        assert(bathymetry_x_len >= 2);
        assert(bathymetry_y_len >= 2);
        
        // Read minimum value for x and y variable
        // Note: We assume the x and y dimensions are stored in ascending order
        index[0] = 0;
        nc_get_var1_float(bathymetry_file_id, bathymetry_x_id, (const size_t *)index, &bathymetry_x_min);
        nc_get_var1_float(bathymetry_file_id, bathymetry_y_id, (const size_t *)index, &bathymetry_y_min);
        
        // Read maximum value for x and y variable
        index[0] = bathymetry_x_len - 1;
        nc_get_var1_float(bathymetry_file_id, bathymetry_x_id, (const size_t *)index, &bathymetry_x_max);
        index[0] = bathymetry_y_len - 1;
        nc_get_var1_float(bathymetry_file_id, bathymetry_y_id, (const size_t *)index, &bathymetry_y_max);
        
        // Calculate step width (cell size) for x and y variable
        // Note: We assume the step width remains constaint over the complete domain
        bathymetry_x_step = (bathymetry_x_max - bathymetry_x_min) / (bathymetry_x_len - 1);
        bathymetry_y_step = (bathymetry_y_max - bathymetry_y_min) / (bathymetry_y_len - 1);
        
        // Step size should be greater than zero
        assert(bathymetry_x_step > 0.0); assert(bathymetry_y_step > 0.0);
        
        /**
         * Load the displacement file
         */
        // Open the file
        retval = nc_open(displacementFileName.c_str(), NC_NOWRITE, &displacement_file_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        // Read ID for x and y and z variables
        retval = nc_inq_dimid(displacement_file_id, "x", &displacement_x_id);    
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_dimid(displacement_file_id, "y", &displacement_y_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_varid(displacement_file_id, "z", &displacement_z_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        // Read Dimensions for x and y variable
        retval = nc_inq_dimlen(displacement_file_id, displacement_x_id, &displacement_x_len);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_dimlen(displacement_file_id, displacement_y_id, &displacement_y_len);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        // We should have more than a single cell
        assert(bathymetry_x_len >= 2);
        assert(bathymetry_y_len >= 2);
        // The dimensions of the displacement file cannot be bigger than the dimensions of the bathymetry file
        assert(displacement_x_len <= bathymetry_x_len);
        assert(displacement_y_len <= bathymetry_y_len);
        
        // Read minimum value for x and y variable
        // Note: We assume the x and y dimensions are stored in ascending order
        index[0] = 0;
        nc_get_var1_float(displacement_file_id, displacement_x_id, (const size_t *)index, &displacement_x_min);
        nc_get_var1_float(displacement_file_id, displacement_y_id, (const size_t *)index, &displacement_y_min);
        
        // Read maximum value for x and y variable
        index[0] = displacement_x_len - 1;
        nc_get_var1_float(displacement_file_id, displacement_x_id, (const size_t *)index, &displacement_x_max);
        index[0] = displacement_y_len - 1;
        nc_get_var1_float(displacement_file_id, displacement_y_id, (const size_t *)index, &displacement_y_max);
        
        // Calculate step width (cell size) for x and y variable
        // Note: We assume the step width remains constaint over the complete domain
        displacement_x_step = (displacement_x_max - displacement_x_min) / (displacement_x_len - 1);
        displacement_y_step = (displacement_y_max - displacement_y_min) / (displacement_y_len - 1);
        
        // Step size should be greater than zero
        assert(displacement_x_step > 0.0); assert(displacement_y_step > 0.0);
    }
    
    void handleNetCDFError(int status) {
        std::cerr << "NetCDF Error: " << nc_strerror(status) << std::endl;
        exit(status);
    }
    
    void getInitialBathymetryIndex(float x, float y, size_t index[2]) {
        // Find the nearest cell in the bathymetry data
        size_t xIndex, yIndex;
        
        // relative position in the domain
        float relPosX = x - getBoundaryPos(BND_LEFT);
        float relPosY = y - getBoundaryPos(BND_BOTTOM);
        
        if(relPosX >= tolerance) {
            xIndex = static_cast <size_t> (std::floor(relPosX / bathymetry_x_step));
            
            // make sure the index stays inside variable index bounds
            if(xIndex >= bathymetry_x_len)
                xIndex = bathymetry_x_len-1;
        } else {
            // requested coordninate is below our lower domain bound
            xIndex = 0;
        }
        
        if(relPosY >= tolerance) {
            yIndex = static_cast <size_t> (std::floor(relPosY / bathymetry_y_step));
            
            // make sure the index stays inside variable index bounds
            if(yIndex >= bathymetry_y_len)
                yIndex = bathymetry_y_len-1;
        } else {
            // requested coordninate is below our lower domain bound
            yIndex = 0;
        }
        
        index[0] = yIndex;
        index[1] = xIndex;
    }
    
    int getDisplacementIndex(float x, float y, size_t index[2]) {
        float borderXMin = displacement_x_min - (displacement_x_step / 2);
        float borderXMax = displacement_x_max + (displacement_x_step / 2);
        float borderYMin = displacement_y_min - (displacement_y_step / 2);
        float borderYMax = displacement_y_max + (displacement_y_step / 2);
        
        // Check if we're outside the displacement data domain
        if(x >= borderXMax || x <= borderXMin || y >= borderYMax || y <= borderYMin)
            return 0;
        
        // We're inside displacement data domain
        // Find the nearest cell in the displacement data
        size_t xIndex = static_cast <size_t> (std::floor((x - borderXMin) / displacement_x_step));
        size_t yIndex = static_cast <size_t> (std::floor((y - borderYMin) / displacement_y_step));
        
        // Check index bounds
        assert(xIndex >= 0); assert(xIndex < displacement_x_len);
        assert(yIndex >= 0); assert(yIndex < displacement_y_len);
        
        index[0] = yIndex;
        index[1] = xIndex;
        
        return 1;
    }
    
    float getInitialBathymetry(float x, float y) {
        // Index array for reading values from NetCDF
        size_t index[2];
        getInitialBathymetryIndex(x, y, index);
        
        float bathymetry;
        int status = nc_get_var1_float(bathymetry_file_id, bathymetry_z_id, (const size_t *)index, &bathymetry);
        if(status != NC_NOERR) handleNetCDFError(status);
        return bathymetry;
    }
    
    float getDisplacement(float x, float y) {
        // Index array for reading values from NetCDF
        size_t index[2];
        int hasDisplacement = getDisplacementIndex(x, y, index);
        if(!hasDisplacement)
            return 0.0;
        
        float displacement;
        int status = nc_get_var1_float(displacement_file_id, displacement_z_id, (const size_t *)index, &displacement);
        if(status != NC_NOERR) handleNetCDFError(status);
        return displacement;
    }

public:

    SWE_TsunamiScenario(std::string bathymetryFileName, std::string displacementFileName)
    : SWE_Scenario() {
        loadInputFiles(bathymetryFileName, displacementFileName);
        
        for(int i = 0; i < 4; i++)
            boundaryTypes[i] = OUTFLOW;
    }
    
    SWE_TsunamiScenario(std::string bathymetryFileName, std::string displacementFileName, BoundaryType* _boundaryTypes) : SWE_Scenario() {
            loadInputFiles(bathymetryFileName, displacementFileName);
            
            for(int i = 0; i < 4; i++)
                boundaryTypes[i] = _boundaryTypes[i];
    }
    
    ~SWE_TsunamiScenario() {
        // Close open NetCDF handles
        nc_close(bathymetry_file_id);
        nc_close(displacement_file_id);
    }
    
    /**
     * @return bathymetry at pos
     */
    float getBathymetry(float x, float y) {
        float bathymetry = getInitialBathymetry(x,y) + getDisplacement(x,y);

        // Test if the bathymetry value is between -20 metres and 20 metres
        if(std::fabs(bathymetry) >= 20.0)
            return bathymetry;
        else if(bathymetry >= 0.0)
            return 20.0;
        return -20.0;
    };

     /**
     * @return Initial water height at pos
     */
    float getWaterHeight(float x, float y) {
        // bathymetry is rounded to avoid bathymetry around zero
        // therefore, we get the rounded bathymetry and subtract the displacement
        float height = -(getBathymetry(x,y) - getDisplacement(x,y));
        if(height >= 0.0)
            return height;
        return 0.0;
    };
    
    /**
     * @return time when to end simulation
     */
    float endSimulation() {
        return 50.0f;
    };

   /**
    * Determines the type (e.g. reflecting wall or outflow) of a certain boundary
    *
    * @param edge The boundary edge
    * @return The type of the specified boundary (e.g. OUTFLOW or WALL)
    */
    BoundaryType getBoundaryType(BoundaryEdge edge) {
        return boundaryTypes[edge];
    };
    
    /** Get the boundary positions
     *
     * @param i_edge which edge
     * @return value in the corresponding dimension
     */
    float getBoundaryPos(BoundaryEdge i_edge) {
        if ( i_edge == BND_LEFT )
            return bathymetry_x_min - (bathymetry_x_step / 2);
        else if ( i_edge == BND_RIGHT)
            return bathymetry_x_max + (bathymetry_x_step / 2);
        else if ( i_edge == BND_BOTTOM )
            return bathymetry_y_min - (bathymetry_y_step / 2);
        else
            return bathymetry_y_max + (bathymetry_y_step / 2);
    };
};

#endif /* SWE_TSUNAMISCENARIO_HH_ */
