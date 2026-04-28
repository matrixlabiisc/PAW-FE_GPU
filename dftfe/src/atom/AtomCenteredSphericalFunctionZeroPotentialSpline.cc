// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//

#include "AtomCenteredSphericalFunctionZeroPotentialSpline.h"
#include "vector"
namespace dftfe
{
  AtomCenteredSphericalFunctionZeroPotentialSpline::
    AtomCenteredSphericalFunctionZeroPotentialSpline(std::string filename,
                                                     double      truncationTol,
                                                     bool consider0thEntry)
  {
    d_lQuantumNumber = 0;
    std::vector<std::vector<double>> radialFunctionData(0);
    unsigned int                     fileReadFlag =
      dftUtils::readPsiFile(2, radialFunctionData, filename);
    d_DataPresent = fileReadFlag == 1 ? true : false;
    if (!d_DataPresent)
      std::cout << "Warning file not present: " << filename << std::endl;
    d_cutOff = 0.0;
    d_rMin   = 0.0;
    if (fileReadFlag == 1)
      {
        unsigned int        numRows = radialFunctionData.size() - 1;
        std::vector<double> xData(numRows), yData(numRows);

        unsigned int maxRowId = 0;
        for (unsigned int irow = 0; irow < numRows; ++irow)
          {
            xData[irow] = radialFunctionData[irow][0];
            yData[irow] = radialFunctionData[irow][1];

            if (std::fabs(yData[irow]) > truncationTol)
              maxRowId = irow + 1;
          }

        if (!consider0thEntry)
          yData[0] = yData[1];

        alglib::real_1d_array x;
        x.setcontent(numRows, &xData[0]);
        alglib::real_1d_array y;
        y.setcontent(numRows, &yData[0]);
        alglib::ae_int_t natural_bound_type_L = 1;
        alglib::ae_int_t natural_bound_type_R = 1;
        spline1dbuildcubic(x,
                           y,
                           numRows,
                           natural_bound_type_L,
                           0.0,
                           natural_bound_type_R,
                           0.0,
                           d_radialSplineObject);
        d_cutOff = xData[maxRowId];
        // std::cout << "Cutoff Radius: " << d_cutOff << " " << maxRowId
        //           << std::endl;
        d_rMin = xData[0];
      }
  }



} // end of namespace dftfe
