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

#define BATHYMETRY_FILE "artificialtsunami_bathymetry_1000.nc"
#define DISPLACEMENT_FILE "artificialtsunami_displ_1000.nc"

using namespace std;


/**

 */
class SWE_TsunamiScenario : public SWE_Scenario {



private:

	float bathymetry_xy;
	float displacement_xy;
	

	/**
	 *
	 * Load both the bathymetry and displacement file
	 *
	 */
	void loadInputFiles(float x, float y){

#ifndef WRITENETCDF
        std::cerr << "NetCDF support must be enabled (writeNetCDF=yes) in order to read bathymetry data" << std::endl;
        abort();
#else
       	/**
       	 * Load the bathymetry file
       	 */
		int 
			/* The bathymetry file ID */
			bathymetry_file_id,
			/* The z(x,y) ID */
			bathymetry_z_id,
			/* The dimension x ID */
			bathymetry_dimx_id,
			/* The dimension y ID */
			bathymetry_dimy_id;
		size_t 
			bathymetry_len_x,
			bathymetry_len_y; 

		// We can store the return values of the netCDF methods here
		int retval;

		// Open the file
  		retval = nc_open(BATHYMETRY_FILE, NC_NOWRITE, &bathymetry_file_id);

 	  	retval = nc_inq_varid(bathymetry_file_id, "z", &bathymetry_z_id);

		retval= nc_inq_dimid(bathymetry_file_id, "x", &bathymetry_dimx_id);	
		retval= nc_inq_dimid(bathymetry_file_id, "y", &bathymetry_dimy_id);

		retval= nc_inq_dimlen(bathymetry_file_id, bathymetry_dimx_id, &bathymetry_len_x);
		retval= nc_inq_dimlen(bathymetry_file_id, bathymetry_dimy_id, &bathymetry_len_y);

		// Array in which we will store the bathymetry data from the input file
		float bathymetry_in[bathymetry_len_x][bathymetry_len_y];

		// Read the bathymetry data and store it into bathymetry_in
		retval = nc_get_var_float(bathymetry_file_id, bathymetry_z_id, &bathymetry_in[0][0]);

		//make sure we stay into array bounds
		if ((int) x > 1000) x=1000; 
		if ((int) x < 0) x=0;  
		if ((int) y > 1000) y=1000; 
		if ((int) y < 0) y=0;  

		bathymetry_xy = bathymetry_in[(int) x][(int) y];

		/**
		 * Load the displacement file
		 */

		int 
			/* The displacement file ID */
			displacement_file_id,
			/* The z(x,y) ID */
			displacement_z_id,
			/* The dimension x ID */
			displacement_dimx_id,
			/* The dimension y ID */
			displacement_dimy_id;

		size_t
			/* The size in x-dimension */
			displacement_len_x,
			/* The size in y-dimension */
			displacement_len_y;

		// Open the file
  		retval = nc_open(DISPLACEMENT_FILE, NC_NOWRITE, &displacement_file_id);

 	  	retval = nc_inq_varid(displacement_file_id, "z", &displacement_z_id);

		retval= nc_inq_dimid(displacement_file_id, "x", &displacement_dimx_id);	
		retval= nc_inq_dimid(displacement_file_id, "y", &displacement_dimy_id);

		retval= nc_inq_dimlen(displacement_file_id, displacement_dimx_id, &displacement_len_x);
		retval= nc_inq_dimlen(displacement_file_id, displacement_dimy_id, &displacement_len_y);

		//The dimensions of the displacement file cannot be bigger than the dimensions of the bathymetry file
		assert(displacement_len_x <= bathymetry_len_x);
		assert(displacement_len_y <= bathymetry_len_y);

		// Array in which we will store the displacement data from the input file
		float displacement_in[displacement_len_x][displacement_len_y];

		// Read the displacement data and store it into displacement_in
		retval = nc_get_var_float(displacement_file_id, displacement_z_id, &displacement_in[0][0]);

		/**
		 * Since we have two different coordinate systems, we have to shift the indices from the displacement coordinate system to 	
		 * the bathymetry coordinate system so that the displacement applies in the middle of the entire domain. 
		 * tx is the translation in x-direction, ty the translation in y-direction.
		 */ 
		int tx= (int) (bathymetry_len_x - displacement_len_x)*0.5f;
		int ty= (int) (bathymetry_len_y - displacement_len_y)*0.5f;

		if (x >= tx && x <= (bathymetry_len_x - tx) && y >= ty && y <= (bathymetry_len_y - ty)){
			displacement_xy = displacement_in[int(x)-tx][int(y) -ty];
		}
		else displacement_xy = 0;
#endif;
}

public:

    /**
     * @return bathymetry at pos
     */
    float getBathymetry(float x, float y) {

		loadInputFiles(x, y);

		bathymetry_xy=bathymetry_xy+displacement_xy;

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
		loadInputFiles(x, y);
		return -bathymetry_xy;
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
