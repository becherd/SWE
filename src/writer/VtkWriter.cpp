/**
 * @file
 * This file is part of SWE.
 *
 * @author Sebastian Rettenberger (rettenbs AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Sebastian_Rettenberger,_M.Sc.)
 *
 * @section LICENSE
 *
 * SWE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SWE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SWE.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * @section DESCRIPTION
 */

#include <cassert>
#include <fstream>
#include "VtkWriter.hh"

/**
 * Creates a vtk file for each time step.
 * Any existing file will be replaced.
 *
 * @param i_baseName base name of the netCDF-file to which the data will be written to.
 * @param i_nX number of cells in the horizontal direction.
 * @param i_nY number of cells in the vertical direction.
 * @param i_dX cell size in x-direction.
 * @param i_dY cell size in y-direction.
 * @param i_offsetX x-offset of the block
 * @param i_offsetY y-offset of the block
 * @param i_dynamicBathymetry
 *
 * @todo This version can only handle a boundary layer of size 1
 */
io::VtkWriter::VtkWriter( const std::string &i_baseName,
		const Float2D &i_b,
		const BoundarySize &i_boundarySize,
		int i_nX, int i_nY,
		float i_dX, float i_dY, float i_coarseness,
		int i_offsetX, int i_offsetY) :
  io::Writer(i_baseName, i_b, i_boundarySize, i_nX, i_nY, i_coarseness),
  dX(i_dX), dY(i_dY),
  offsetX(i_offsetX), offsetY(i_offsetY)
{
}

void io::VtkWriter::writeTimeStep(
		const Float2D &i_h,
        const Float2D &i_hu,
        const Float2D &i_hv,
        float i_time)
{
	std::ofstream vtkFile(generateFileName().c_str());
	assert(vtkFile.good());

	//Grid wrapper for each grid
	CoarseGridWrapper gridWrapperH(i_h, boundarySize, nX, nY, coarseness);
	CoarseGridWrapper gridWrapperHu(i_hu, boundarySize, nX, nY, coarseness);
	CoarseGridWrapper gridWrapperHv(i_hv, boundarySize, nX, nY, coarseness);
	CoarseGridWrapper gridWrapperB(b, boundarySize, nX, nY, coarseness);

	// VTK header
	vtkFile << "<?xml version=\"1.0\"?>" << std::endl
			<< "<VTKFile type=\"StructuredGrid\">" << std::endl
			<< "<StructuredGrid WholeExtent=\"" << offsetX << " " << offsetX+coarseX
				<< " " << offsetY << " " << offsetY+coarseY << " 0 0\">" << std::endl
	        << "<Piece Extent=\"" << offsetX << " " << offsetX+coarseX
	        	<< " " << offsetY << " " << offsetY+coarseY << " 0 0\">" << std::endl;

	vtkFile << "<Points>" << std::endl
			<< "<DataArray NumberOfComponents=\"3\" type=\"Float32\" format=\"ascii\">" << std::endl;

	//Grid points
	for (unsigned int j=0; j < coarseX+1; j++)
	      for (unsigned int i=0; i < coarseY+1; i++)
	    	  vtkFile << (offsetX+i)*(static_cast <int> (std::ceil(float(dX) / coarseness))) << " " << (offsetY+j)*(static_cast <int> (std::ceil(float(dY) / coarseness))) <<" 0" << std::endl;

	vtkFile << "</DataArray>" << std::endl
			<< "</Points>" << std::endl;

	vtkFile << "<CellData>" << std::endl;

	// Water surface height h
	vtkFile << "<DataArray Name=\"h\" type=\"Float32\" format=\"ascii\">" << std::endl;
	for (unsigned int j=0; j < coarseY; j++)
		for (unsigned int i=0; i < coarseX+0; i++)
			vtkFile << gridWrapperH.getElem(i, j) << std::endl;
	vtkFile << "</DataArray>" << std::endl;

	// Momentums
	vtkFile << "<DataArray Name=\"hu\" type=\"Float32\" format=\"ascii\">" << std::endl;
	for (unsigned int j=0; j < coarseY; j++)
		for (unsigned int i=0; i < coarseX; i++)
			vtkFile << gridWrapperHu.getElem(i, j) << std::endl;
	vtkFile << "</DataArray>" << std::endl;

	vtkFile << "<DataArray Name=\"hv\" type=\"Float32\" format=\"ascii\">" << std::endl;
	for (unsigned int j=0; j < coarseY; j++)
		for (unsigned int i=0; i<coarseX; i++)
			vtkFile << gridWrapperHv.getElem(i, j) << std::endl;
	vtkFile << "</DataArray>" << std::endl;

	// Bathymetry
	vtkFile << "<DataArray Name=\"b\" type=\"Float32\" format=\"ascii\">" << std::endl;
	for (unsigned int j=0; j<coarseY; j++)
		for (unsigned int i=0; i<coarseX; i++)
			vtkFile << gridWrapperB.getElem(i, j) << std::endl;
	vtkFile << "</DataArray>" << std::endl;

	vtkFile << "</CellData>" << std::endl
			<< "</Piece>" << std::endl;

	vtkFile << "</StructuredGrid>" << std::endl
			<< "</VTKFile>" << std::endl;

	// Increament time step
	timeStep++;
}
