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

#ifndef WRITER_HH_
#define WRITER_HH_

#include "tools/help.hh"
#include "writer/BoundarySize.hh"
#include "writer/CoarseGridWrapper.hh"

namespace io {
	class Writer;
}

class io::Writer
{
protected:
	//! file name of the data file
	const std::string fileName;

    //! (Reference) to bathymetry data
    const Float2D &b;

	//! Boundary layer size
	const BoundarySize boundarySize;

    //! dimensions of the refined/internal grid in x- and y-direction.
    const unsigned int nX, nY;
    
	//! Coarseness factor
	float coarseness;
    
    //! dimensions of the coarse/output grid in x- and y-direction
    unsigned int coarseX, coarseY;

    //! current time step
    size_t timeStep;

public:
    /**
	 * @param i_boundarySize size of the boundaries.
	 */
	Writer(const std::string &i_fileName,
		const Float2D &i_b,
		const BoundarySize &i_boundarySize,
		int i_nX, int i_nY,
        float i_coarseness = 1.f)
		: fileName(i_fileName),
		  b(i_b),
		  boundarySize(i_boundarySize),
		  nX(i_nX), nY(i_nY),
          coarseness(i_coarseness),
		  timeStep(0)
	{
        CoarseGridWrapper gridWrapper(b, boundarySize, nX, nY, coarseness);
        coarseX = gridWrapper.getCols();
        coarseY = gridWrapper.getRows();
	}

	virtual ~Writer() {}

	/**
	 * Writes one time step
	 *
	 * @param i_h water heights at a given time step.
	 * @param i_hu momentums in x-direction at a given time step.
	 * @param i_hv momentums in y-direction at a given time step.
	 * @param i_time simulation time of the time step.
	 */
	virtual void writeTimeStep(
			const Float2D &i_h,
            const Float2D &i_hu,
            const Float2D &i_hv,
            float i_time) = 0;
};

#endif // WRITER_HH_
