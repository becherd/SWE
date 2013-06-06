#ifndef __COARSEGRIDWRAPPER_HH
#define __COARSEGRIDWRAPPER_HH

#include <cassert>
#include "Writer.hh"
#include "tools/help.hh"

/**
 * CoarseGridWrapper is a wrapper class around Float2D
 * to map from a coarse grid to a refined internal grid
 * represented by a Float2D object.
 */
class CoarseGridWrapper {
    private:
        //! Reference to the refined grid to wrap around
        const Float2D &grid;
        //! Size of the boundary (ghost cells) at left, right, bottom, top boundary
        io::BoundarySize boundarySize;
        
        //! Refined grid size in x-direction (excluding boundary/ghost cells)
        unsigned int refinedX;
        //! Refined grid size in y-direction (excluding boundary/ghost cells)
        unsigned int refinedY;
        
        //! coarseness factor
        /** describes how coarse the grid will be in relation to the refined grid */
        float coarseness;
                
        //! Coarse grid size in x-direction
        unsigned int coarseX;
        //! Coarse grid size in y-direction
        unsigned int coarseY;
        
        //! step width of coarse grid in x direction
        /** A stepwidth of n means that a single coarse cell contains n refined cells in that direction */
        float stepWidthX;
        //! step width of coarse grid in y direction
        /** A stepwidth of n means that a single coarse cell contains n refined cells in that direction */
        float stepWidthY;
    public:
        /**
         * Constructor
         *
         * @param i_grid The refined grid to wrap around
         * @param i_nX Refined grid size in X-direction (excluding ghost cells)
         * @param i_nY Refined grid size in Y-direction (excluding ghost cells)
         * @param i_boundarySize Number of ghost cells at left, right, bottom, top boundary
         * @param i_coarseness The coarseness factor
         */
        CoarseGridWrapper(  const Float2D &i_grid,
                            const io::BoundarySize &i_boundarySize,
                            unsigned int i_nX, unsigned int i_nY,
                            float i_coarseness) :
                grid(i_grid),
                boundarySize(i_boundarySize),
                refinedX(i_nX), refinedY(i_nY),
                coarseness(i_coarseness) {
            assert(coarseness >= 1.f);
            // calculate number of rows/cols in the coarse grid
            coarseX = static_cast <int> (std::ceil(float(refinedX) / coarseness));
            coarseY = static_cast <int> (std::ceil(float(refinedY) / coarseness));
            assert(coarseX > 0); assert(coarseY > 0);
            
            // calculate step width of the coarse grid
            stepWidthX = float(refinedX) / float(coarseX);
            stepWidthY = float(refinedY) / float(coarseY);
            assert(stepWidthX >= 1.f); assert(stepWidthY >= 1.f);
        }
        
        /// Read the value of a coarse cell averaged over the refined values
        /**
         * @param y The x-coordinate of the requested coarse cell
         * @param y The y-coordinate of the requested coarse cell
         * @return The weighted average value of all refined cells belonging to the coarse cell
         */
        inline float getElem(unsigned int x, unsigned int y) {
            assert(x >= 0); assert(y >= 0);
            assert(x < coarseX); assert(y < coarseY);
            
            float lowerX = float(x) * stepWidthX;
            float upperX = float(x+1) * stepWidthX;
            float lowerY = float(y) * stepWidthY;
            float upperY = float(y+1) * stepWidthY;
            
            unsigned int lowerIndexX = static_cast <unsigned int> (std::floor(lowerX));
            unsigned int upperIndexX = static_cast <unsigned int> (std::ceil(upperX));
            unsigned int lowerIndexY = static_cast <unsigned int> (std::floor(lowerY));
            unsigned int upperIndexY = static_cast <unsigned int> (std::ceil(upperY));
            assert(lowerIndexX <= refinedX);
            assert(upperIndexX <= refinedX);
            assert(lowerIndexY <= refinedY);
            assert(upperIndexY <= refinedY);
            
            float lowerFractionX = 1.f - (lowerX - float(lowerIndexX));
            float upperFractionX = 1.f - (float(upperIndexX) - upperX);
            float lowerFractionY = 1.f - (lowerY - float(lowerIndexY));
            float upperFractionY = 1.f - (float(upperIndexY) - upperY);
            assert(lowerFractionX <= 1.f);
            assert(upperFractionX <= 1.f);
            assert(lowerFractionY <= 1.f);
            assert(upperFractionY <= 1.f);
            
            // for each cell being at least partially in the coarse cell
            // calculate cell size and add area and value
            float value = 0.f;
            float area = 0.f;
            
            // The grid value of a certain cell is weighted by the the fraction of the
            // refined cell that belongs to the coarse cell
            // At the end, the sum of weighted values is divided by the sum of the weights
            // to get the weighted average of all refined cells in a coarse cell
            for(unsigned int i = lowerIndexX; i < upperIndexX; i++) {
                for(unsigned int j = lowerIndexY; j < upperIndexY; j++) {
                    // what fraction of the current refined cell is inside the coarse cell?
                    float fraction = 1.f;
                    
                    if(i == lowerIndexX) fraction *= lowerFractionX;
                    if(i == upperIndexX-1) fraction *= upperFractionX;
                    
                    if(j == lowerIndexY) fraction *= lowerFractionY;
                    if(j == upperIndexY-1) fraction *= upperFractionY;
                    
                    // Add boundary size (left/bottom) to indices
                    unsigned int xCoord = i + boundarySize[0];
                    unsigned int yCoord = j + boundarySize[2];
                    
                    // area is the sum of the weights (fractions)
                    area += fraction;
                    // value is the sum of weighted values
                    value += (fraction * grid[xCoord][yCoord]);
                }
            }
            
            assert(area > 0.f);
            return (value / area);
        }
        
        /**
         * @return The number of rows in the coarse grid
         */
        inline int getRows() { return coarseY; }
        
        /**
         * @return The number of columns in the coarse grid
         */
        inline int getCols() { return coarseX; }
};

#endif
