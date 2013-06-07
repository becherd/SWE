
#ifndef BOUNDARYSIZE_HH_
#define BOUNDARYSIZE_HH_

namespace io {
	struct BoundarySize;
}

/**
 * This struct is used so we can initialize this array
 * in the constructor.
 */
struct io::BoundarySize
{
	 /**
	  * boundarySize[0] == left
	  * boundarySize[1] == right
	  * boundarySize[2] == bottom
	  * boundarySize[3] == top
	  */
	int boundarySize[4];

	int& operator[](unsigned int i)
	{
		return boundarySize[i];
	}

	int operator[](unsigned int i) const
	{
		return boundarySize[i];
	}
};

#endif // BOUDARYSIZE_HH_
