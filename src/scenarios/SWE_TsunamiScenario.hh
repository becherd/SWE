#ifndef __SWE_TSUNAMISCENARIO_HH
#define __SWE_TSUNAMISCENARIO_HH

#include "SWE_Scenario.hh"
#include <cassert>
#include <cstdlib>
#include <string>
#include <cstring>
#include <iostream>
#include <netcdf.h>
#include <stdio.h>

/**
 * Scenario "Tsunami"
 *
 * A generic Tsunami Scenario loading bathymetry and displacement data from
 * from NetCDF files.
 */
class SWE_TsunamiScenario : public SWE_Scenario {

protected:

    float bathymetry_xy;
    float displacement_xy;
    
    //! The NetCDF bathymetry file ID
    int bathymetry_file_id;
    //! The NetCDF bathymetry z(x,y) ID
    int bathymetry_z_id;
    //! The NetCDF bathymetry dimension x ID
    int bathymetry_dimx_id;
    //! The NetCDF bathymetry dimension y ID
    int bathymetry_dimy_id;
    //! The NetCDF bathymetry length of x dimension
    size_t bathymetry_len_x;
    //! The NetCDF bathymetry length of y dimension
    size_t bathymetry_len_y; 
    
    //! The NetCDF displacement file ID
    int displacement_file_id;
    //! The NetCDF displacement z(x,y) ID
    int displacement_z_id;
    //! The NetCDF displacement dimension x ID
    int displacement_dimx_id;
    //! The NetCDF displacement dimension y ID
    int displacement_dimy_id;
    //! The NetCDF displacement length of x dimension
    size_t displacement_len_x;
    //! The NetCDF displacement length of y dimension
    size_t displacement_len_y; 
    
    
    /**
     * Load both the bathymetry and displacement file
     *
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
        
        retval = nc_inq_varid(bathymetry_file_id, "z", &bathymetry_z_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        retval = nc_inq_dimid(bathymetry_file_id, "x", &bathymetry_dimx_id);    
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_dimid(bathymetry_file_id, "y", &bathymetry_dimy_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        retval = nc_inq_dimlen(bathymetry_file_id, bathymetry_dimx_id, &bathymetry_len_x);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_dimlen(bathymetry_file_id, bathymetry_dimy_id, &bathymetry_len_y);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        /**
         * Load the displacement file
         */
        
        // Open the file
          retval = nc_open(displacementFileName.c_str(), NC_NOWRITE, &displacement_file_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
           retval = nc_inq_varid(displacement_file_id, "z", &displacement_z_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        retval = nc_inq_dimid(displacement_file_id, "x", &displacement_dimx_id);    
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_dimid(displacement_file_id, "y", &displacement_dimy_id);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        retval = nc_inq_dimlen(displacement_file_id, displacement_dimx_id, &displacement_len_x);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        retval = nc_inq_dimlen(displacement_file_id, displacement_dimy_id, &displacement_len_y);
        if(retval != NC_NOERR) handleNetCDFError(retval);
        
        // The dimensions of the displacement file cannot be bigger than the dimensions of the bathymetry file
        assert(displacement_len_x <= bathymetry_len_x);
        assert(displacement_len_y <= bathymetry_len_y);
        
        
        // TODO: Determine coordinate system mapping between bathymetry and displacement 
        // data files (displacement domain is a subset of bathymetry domain)
        
        /**
         * Since we have two different coordinate systems, we have to shift the indices from the displacement coordinate system to     
         * the bathymetry coordinate system so that the displacement applies in the middle of the entire domain. 
         * tx is the translation in x-direction, ty the translation in y-direction.
         */ 
        // int tx = (int) (bathymetry_len_x - displacement_len_x)*0.5f;
        // int ty = (int) (bathymetry_len_y - displacement_len_y)*0.5f;
        // 
        // if (x >= tx && x <= (bathymetry_len_x - tx) && y >= ty && y <= (bathymetry_len_y - ty)){
        //     displacement_xy = displacement_in[int(x)-tx][int(y) -ty];
        // }
        // else displacement_xy = 0;

    }
    
    void handleNetCDFError(int status) {
        std::cerr << "NetCDF Error: " << nc_strerror(status) << std::endl;
        abort();
    }
    
    float getInitialBathymetry(float x, float y) {
        float initialBathymetry;
        // TODO: map (x,y) position in domain to NetCDF index
        size_t index[] = {0,0};
        int status = nc_get_var1_float(bathymetry_file_id, bathymetry_z_id, (const size_t *)index, &initialBathymetry);
        if(status != NC_NOERR) handleNetCDFError(status);
        return initialBathymetry;
    }
    
    float getDisplacement(float x, float y) {
        float initialDisplacement;
        // TODO: map (x,y) position in domain to NetCDF index
        size_t index[] = {0,0};
        int status = nc_get_var1_float(displacement_file_id, displacement_z_id, (const size_t *)index, &initialDisplacement);
        if(status != NC_NOERR) handleNetCDFError(status);
        return initialDisplacement;
    }

public:
    
    SWE_TsunamiScenario(std::string bathymetryFileName, std::string displacementFileName) {
        loadInputFiles(bathymetryFileName, displacementFileName);
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
        bathymetry_xy = getInitialBathymetry(x,y) + getDisplacement(x,y);

        // Test if the bathymetry value is between -20 metres and 20 metres
        if (bathymetry_xy >= -20.0f && bathymetry_xy < 0.0f)
            return -20.0f;
        else if (bathymetry_xy <= 20.0f && bathymetry_xy >= 0.0f)
            return 20.0f;
        else return bathymetry_xy;
    };

     /**
     * @return Initial water height at pos
     */
    float getWaterHeight(float x, float y) { 
        return -getInitialBathymetry(x,y);
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
        return OUTFLOW;
    };
    
    /** Get the boundary positions
     *
     * @param i_edge which edge
     * @return value in the corresponding dimension
     */
    float getBoundaryPos(BoundaryEdge i_edge) {
        if ( i_edge == BND_LEFT )
            return 0.0f;
        else if ( i_edge == BND_RIGHT)
            return 1000.0f;
        else if ( i_edge == BND_BOTTOM )
            return 0.0f;
        else
            return 1000.0f; 
    };
};

#endif /* SWE_TSUNAMISCENARIO_HH_ */
