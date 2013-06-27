#ifndef __SWE_TSUNAMISCENARIO_HH
#define __SWE_TSUNAMISCENARIO_HH

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <netcdf.h>

#include "SWE_Scenario.hh"

/**
 * Scenario "Tsunami"
 *
 * A generic Tsunami Scenario loading bathymetry and displacement data from
 * NetCDF files.
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
    
    //! The NetCDF bathymetry values of x dimensions
    float *bathymetry_x_values;
    //! The NetCDF bathymetry left boundary (in x dimension)
    float bathymetry_left;
    //! The NetCDF bathymetry right boundary (in x dimension)
    float bathymetry_right;
    //! The NetCDF bathymetry step width in x dimension (step width between two cells)
    float bathymetry_x_step;
    
    //! The NetCDF bathymetry values of y dimensions
    float *bathymetry_y_values;
    //! The NetCDF bathymetry bottom boundary (in y dimension)
    float bathymetry_bottom;
    //! The NetCDF bathymetry top boundary (in y dimension)
    float bathymetry_top;
    //! The NetCDF bathymetry step width in y dimension (step width between two cells)
    float bathymetry_y_step;
#ifdef NETCDF_CACHE
    //! NetCDF bathymetry z data as cache (for fast read access)
    float *bathymetry_z_cache;
#endif
    
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
    
    //! The NetCDF displacement values of x dimensions
    float *displacement_x_values;
    //! The NetCDF displacement left boundary (in x dimension)
    float displacement_left;
    //! The NetCDF displacement right boundary (in x dimension)
    float displacement_right;
    //! The NetCDF displacement step width in x dimension (step width between two cells)
    float displacement_x_step;
    
    //! The NetCDF displacement values of y dimensions
    float *displacement_y_values;
    //! The NetCDF displacement bottom boundary (in y dimension)
    float displacement_bottom;
    //! The NetCDF displacement top boundary (in y dimension)
    float displacement_top;
    //! The NetCDF displacement step width in y dimension (step width between two cells)
    float displacement_y_step;
#ifdef NETCDF_CACHE
    //! NetCDF displacement z data as cache (for fast read access)
    float *displacement_z_cache;
#endif
    
    /// Load both the bathymetry and displacement file
    /**
     * @param bathymetryFileName The file name of the bathymetry data file
     * @param displacementFileName The file name of the displacement data file
     */
    void loadInputFiles(std::string bathymetryFileName, std::string displacementFileName) {
        // We can store the return values of the netCDF methods here
        int retval;
        
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
        
        // Allocate some memory for the dimensions
        bathymetry_x_values = new float[bathymetry_x_len];
        bathymetry_y_values = new float[bathymetry_y_len];
        
#ifdef NETCDF_CACHE
        // Allocate memory for netcdf bathymetry cache
        bathymetry_z_cache = new float[bathymetry_y_len*bathymetry_x_len];
        // Load complete var
        nc_get_var_float(bathymetry_file_id, bathymetry_z_id, bathymetry_z_cache);
#endif
        
        // Read dimensions from file
        retval = nc_get_var_float(bathymetry_file_id, bathymetry_x_id, bathymetry_x_values);
        retval = nc_get_var_float(bathymetry_file_id, bathymetry_y_id, bathymetry_y_values);
        
        // COARDS note: x and y dimensions may be either monotonically increasing
        // or monotonically decreasing
        
        // Calculate step width (cell size) for x and y variable
        // Note: This calculation is only correct in case we have equally spaced cells in the file
        // However, the error should be negligible in case the assumption of equally spaced
        // cells holds not to be true
        // Note: the step size is negative in case the values are in decreasing order
        bathymetry_x_step = (bathymetry_x_values[bathymetry_x_len-1] - bathymetry_x_values[0]) / (bathymetry_x_len-1);
        bathymetry_y_step = (bathymetry_y_values[bathymetry_y_len-1] - bathymetry_y_values[0]) / (bathymetry_y_len-1);

        // Step width should not be zero
        assert(bathymetry_x_step != 0.0); assert(bathymetry_y_step != 0.0);
        
        // Calculate the left, right, bottom and top end of the domain, since the
        // values denote the center value of a cell, we have to add half a
        // cell of margin to all ends.
        bathymetry_left = bathymetry_x_values[0] - bathymetry_x_step/2;
        bathymetry_right = bathymetry_x_values[bathymetry_x_len-1] + bathymetry_x_step/2;
        bathymetry_bottom = bathymetry_y_values[0] - bathymetry_y_step/2;
        bathymetry_top = bathymetry_y_values[bathymetry_y_len-1] + bathymetry_y_step/2;
        
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
        assert(displacement_x_len >= 2);
        assert(displacement_y_len >= 2);
        // The dimensions of the displacement file cannot be bigger than the dimensions of the bathymetry file
        assert(displacement_x_len <= bathymetry_x_len);
        assert(displacement_y_len <= bathymetry_y_len);
        
        // Allocate some memory for the dimensions
        displacement_x_values = new float[displacement_x_len];
        displacement_y_values = new float[displacement_y_len];
        
#ifdef NETCDF_CACHE
        // Allocate memory for netcdf bathymetry cache
        displacement_z_cache = new float[displacement_y_len*displacement_x_len];
        // Load complete var
        nc_get_var_float(displacement_file_id, displacement_z_id, displacement_z_cache);
#endif
        
        // Read dimensions from file
        retval = nc_get_var_float(displacement_file_id, displacement_x_id, displacement_x_values);
        retval = nc_get_var_float(displacement_file_id, displacement_y_id, displacement_y_values);
        
        // COARDS note: x and y dimensions may be either monotonically increasing
        // or monotonically decreasing
        
        // Calculate step width (cell size) for x and y variable
        // Note: This calculation is only correct in case we have equally spaced cells in the file
        // However, the error should be negligible in case the assumption of equally spaced
        // cells holds not to be true
        // Note: the step size is negative in case the values are in decreasing order
        displacement_x_step = (displacement_x_values[displacement_x_len-1] - displacement_x_values[0]) / (displacement_x_len-1);
        displacement_y_step = (displacement_y_values[displacement_y_len-1] - displacement_y_values[0]) / (displacement_y_len-1);
        
        // Step width should not be zero
        assert(displacement_x_step != 0.0); assert(displacement_y_step != 0.0);
        
        // Calculate the left, right, bottom and top end of the domain, since the
        // values denote the center value of a cell, we have to add half a
        // cell of margin to all ends.
        displacement_left = displacement_x_values[0] - displacement_x_step/2;
        displacement_right = displacement_x_values[displacement_x_len-1] + displacement_x_step/2;
        displacement_bottom = displacement_y_values[0] - displacement_y_step/2;
        displacement_top = displacement_y_values[displacement_y_len-1] + displacement_y_step/2;
    }
    
    /// Abort execution with netCDF error message
    /**
     * @param status The error status returned by a netCDF function call
     */
    void handleNetCDFError(int status) {
        std::cerr << "NetCDF Error: " << nc_strerror(status) << std::endl;
        exit(status);
    }
    
    /// Calculate the nearest cell index for a position inside a domain for a single dimension
    /**
     * @param position The position inside the domain
     * @param origin The origin of the domain
     * @param stepWidth The assumed step width between cells
     * @param values Array of dimension data (the center position of each cell)
     * @param length The total length of the values array
     * @return The index of the nearest cell in the domain
     */
    size_t getIndex1D(float position, float origin, float stepWidth, float *values, size_t length) {
        size_t index;
        
        // calculate the relative positon from the origin (e.g. left-boundary in x-dimension)
        float relativePosition = position - origin;
        
        // Let's assume all the cells are spaced equally
        // We need to check the signs of the relative position and the step width
        // If both have the same sign, the relative position has an index greater than or equal zero
        // E.g. a negative step width implies a decreasing order of values
        // Therefore a positive relative position means the position is higher than
        // the left (=highest valued) boundary and therefore does not lie between the boundaries
        if(std::signbit(relativePosition) == std::signbit(stepWidth)) {
            index = static_cast <size_t> (std::floor(relativePosition / stepWidth));
            
            // make sure the index stays inside variable index bounds
            if(index >= length)
                index = length-1;
        } else {
            // requested coordinate is below our lower domain bound
            index = 0;
        }
#ifdef DISABLE_NONUNIFORM_NETCDF_CELLS
        return index;
#else
        // Let's validate the assumption of equally spaced cells
        float distance = std::fabs(position - values[index]);
        bool indexIsCorrect = true;
        if(index >= 1)
            indexIsCorrect = indexIsCorrect && (distance <= std::fabs(position - values[index-1]));
        if(index <= length-2)
            indexIsCorrect = indexIsCorrect && (distance <= std::fabs(position - values[index+1]));
        
        if(indexIsCorrect)
            return index;
        
        // Oh, the assumption of equally spaced cells seems to hold not to be true :(
        // Lets do a binary search over all the values to find the nearest cell index
        return binaryIndexSearch(position, values, length, 0, length-1);
#endif
    }
    
    /// Perform an extended binary search on dimension data to find the nearest cell
    /**
     * @param position The position inside the domain
     * @param values Array of dimension data (the center position of each cell)
     * @param length The total length of the values array
     * @param start The start index from where to search on
     * @param end The end index till where to search on
     * @return The index of the nearest cell in the domain
     */
    size_t binaryIndexSearch(float position, float *values, size_t length, size_t start, size_t end) {
        assert(start >= 0);
        assert(end <= length-1);
        assert(start <= end);        
        
        if(start == end)
            return start;
        
        // Start in the middle of the remaining values
        size_t searchIndex = (start+end)/2;
        // Calculate distance to chosen cell
        float distance = std::fabs(position - values[searchIndex]);
        
        if(searchIndex >= 1) {
            // in case there is a left-sided cell, check if the distance to the left
            // cell is less than the distance to the chosen cell. If so, we can continue
            // to search only on the left side of our chosen cell index
            if(distance > std::fabs(position - values[searchIndex-1]))
                return binaryIndexSearch(position, values, length, start, searchIndex-1);
        }
        if(searchIndex <= length-2) {
            // in case there is a right-sided cell, check if the distance to the right
            // cell is less than the distance to the chosen cell. If so, we can continue
            // to search only on the right side of our chosen cell index
            if(distance > std::fabs(position - values[searchIndex+1]))
                return binaryIndexSearch(position, values, length, searchIndex+1, end);
        }
        
        // In case neither the left-hand nor the right-hand cell of our chosen cell are
        // "nearer" to the position in the domain, we already found the optimal cell index
        return searchIndex;
    }
    
    /// Read the bathymetry value at a specified netCDF index
    /**
     * This is a wrapper around a NetCDF library function to load data from the
     * NetCDF file. Since loading every value at once is very slow, we are 
     * caching the data file in-memory for fast access (if NETCDF_CACHE is set)
     * 
     * @param x The x index
     * @param y The y index
     */
    float readBathymetryValue(size_t x, size_t y) {
#ifdef NETCDF_CACHE
        return bathymetry_z_cache[y*bathymetry_x_len + x];
#else
        float bathymetry;
        int status = nc_get_var1_float(bathymetry_file_id, bathymetry_z_id, (const size_t *)index, &bathymetry);
        if(status != NC_NOERR) handleNetCDFError(status);
        return bathymetry;
#endif
    }
    
    /// Read the bathymetry value at a specified netCDF index
    /**
     * This is a wrapper around a NetCDF library function to load data from the
     * NetCDF file. Since loading every value at once is very slow, we are 
     * caching the data file in-memory for fast access (if NETCDF_CACHE is set)
     * 
     * @param x The x index
     * @param y The y index
     */
    float readDisplacementValue(size_t x, size_t y) {
#ifdef NETCDF_CACHE
        return displacement_z_cache[y*displacement_x_len + x];
#else
        float displacement;
        int status = nc_get_var1_float(displacement_file_id, displacement_z_id, (const size_t *)index, &displacement);
        if(status != NC_NOERR) handleNetCDFError(status);
        return displacement;
#endif
    }
    
    /// Checks if a supplied value lies between two boundaries 
    /**
     * @param value The value to perform the boundary check on
     * @param left The left boundary value (may be greater than right boundary)
     * @param right The right boundary value (may be less than left boundary)
     * @return whether True if value lies between boundaries, False if not
     */
    int isBetween(float value, float left, float right) {
        if(left < right)
            return (value > left && value < right);
        return (value < left && value > right);
    }
    
    /// Read the initial bathymetry data (before earthquake) from the input file
    /**
     * @param x The x position in the domain
     * @param y The y posiition in the domain
     * @return The z value (bathymetry) read from data file
     */
    float getInitialBathymetry(float x, float y) {
        // Indices for reading values from NetCDF
        size_t yIndex = getIndex1D(y, bathymetry_bottom, bathymetry_y_step, bathymetry_y_values, bathymetry_y_len);
        size_t xIndex = getIndex1D(x, bathymetry_left, bathymetry_x_step, bathymetry_x_values, bathymetry_x_len);
         
        return readBathymetryValue(xIndex, yIndex);
    }
    
    /// Read the displacement data (caused by earthquake) from the input file
    /**
     * @param x The x position in the domain
     * @param y The y posiition in the domain
     * @return The z value (displacement) read from data file
     */
    float getDisplacement(float x, float y) {
        // Check if we're outside the displacement data domain
        if(!isBetween(x, displacement_left, displacement_right) || !isBetween(y, displacement_bottom, displacement_top))
            return 0.0;
        
        // Indices for reading values from NetCDF
        size_t yIndex = getIndex1D(y, displacement_bottom, displacement_y_step, displacement_y_values, displacement_y_len);
        size_t xIndex = getIndex1D(x, displacement_left, displacement_x_step, displacement_x_values, displacement_x_len);
         
        return readDisplacementValue(xIndex, yIndex);
    }

public:
    /// Constructor
    /**
     * @param bathymetryFileName The file name of the bathymetry data file to load
     * @param displacementFileName The file name of the displacement data file to load
     */
    SWE_TsunamiScenario(std::string bathymetryFileName, std::string displacementFileName)
    : SWE_Scenario() {
        loadInputFiles(bathymetryFileName, displacementFileName);
        
        // Set default boundary types
        for(int i = 0; i < 4; i++)
            boundaryTypes[i] = OUTFLOW;
    }
    
    ~SWE_TsunamiScenario() {
        // Close open NetCDF handles
        nc_close(bathymetry_file_id);
        nc_close(displacement_file_id);
        
        delete[] bathymetry_x_values;
        delete[] bathymetry_y_values;
        delete[] displacement_x_values;
        delete[] displacement_y_values;
#ifdef NETCDF_CACHE
        delete[] bathymetry_z_cache;
        delete[] displacement_z_cache;
#endif
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
        return 50.f;
    };
    
    /// Override default boundary types
    /**
     * @param _boundaryTypes An array holding the left, right, bottom, top boundary types
     */
    void setBoundaryTypes(BoundaryType* _boundaryTypes) {
        for(int i = 0; i < 4; i++)
            boundaryTypes[i] = _boundaryTypes[i];
    };

    /// Get the type (e.g. reflecting wall or outflow) of a certain boundary
    /**
     * @param edge The boundary edge
     * @return The type of the specified boundary (e.g. OUTFLOW or WALL)
     */
    BoundaryType getBoundaryType(BoundaryEdge edge) {
        return boundaryTypes[edge];
    };
    
    /// Get the boundary position of a certain boundary
    /**
     * @param i_edge which edge
     * @return value in the corresponding dimension
     */
    float getBoundaryPos(BoundaryEdge i_edge) {
        if ( i_edge == BND_LEFT )
            return (bathymetry_left < bathymetry_right) ? bathymetry_left : bathymetry_right;
        else if ( i_edge == BND_RIGHT)
            return (bathymetry_left > bathymetry_right) ? bathymetry_left : bathymetry_right;
        else if ( i_edge == BND_BOTTOM )
            return (bathymetry_bottom < bathymetry_top) ? bathymetry_bottom : bathymetry_top;
        else
            return (bathymetry_bottom > bathymetry_top) ? bathymetry_bottom : bathymetry_top;
    };
};

#endif /* SWE_TSUNAMISCENARIO_HH_ */
