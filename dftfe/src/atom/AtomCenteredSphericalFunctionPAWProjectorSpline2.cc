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

#include "AtomCenteredSphericalFunctionPAWProjectorSpline2.h"
#include "vector"
namespace dftfe
{
  AtomCenteredSphericalFunctionPAWProjectorSpline2::
    AtomCenteredSphericalFunctionPAWProjectorSpline2(std::string  filename,
                                                     unsigned int l,
                                                     int          colIndex,
                                                     int          totalColSize,
                                                     double       rmaxAug,
                                                     bool consider0thEntry,
                                                     dftfe::Int numThreads)
  {
    d_lQuantumNumber = l;
    std::vector<std::vector<double>> radialFunctionData(0);
    dftUtils::readFile(totalColSize, radialFunctionData, filename);
    d_DataPresent = true;
    d_cutOff      = rmaxAug;
    d_rMin        = 0.0;
    d_numThreads  = numThreads;

    unsigned int        numRows = radialFunctionData.size() - 1;
    std::vector<double> xData(numRows), yData(numRows);

    unsigned int maxRowId = 0;
    for (unsigned int irow = 0; irow < numRows; ++irow)
      {
        xData[irow] = radialFunctionData[irow][0];
        yData[irow] = radialFunctionData[irow][colIndex] * xData[irow];
      }



    alglib::real_1d_array x;
    x.setcontent(numRows, &xData[0]);
    alglib::real_1d_array y;
    y.setcontent(numRows, &yData[0]);
    alglib::ae_int_t natural_bound_type_L = 1;
    alglib::ae_int_t natural_bound_type_R = 1;
    d_radialSplineObject.clear();
    d_radialSplineObject.resize(d_numThreads);
    for (dftfe::Int threadId = 0; threadId < d_numThreads; threadId++)
      spline1dbuildcubic(x,
                         y,
                         numRows,
                         natural_bound_type_L,
                         radialFunctionData[0][colIndex],
                         natural_bound_type_R,
                         0.0,
                         d_radialSplineObject[threadId]);
    d_rMin = xData[0];
    d_y0   = radialFunctionData[0][colIndex];
  }

  double
  AtomCenteredSphericalFunctionPAWProjectorSpline2::getRadialValue(
    double     r,
    dftfe::Int threadId) const
  {
    if (r >= d_cutOff)
      return 0.0;

    if (std::fabs(r - d_rMin) <= 1E-8)
      return d_y0;

    double v = alglib::spline1dcalc(d_radialSplineObject[threadId], r);
    return v / r;
  }

  std::vector<double>
  AtomCenteredSphericalFunctionPAWProjectorSpline2::getDerivativeValue(
    double r) const
  {
    std::vector<double> Value(3, 0.0);
    if (r >= d_cutOff)
      return Value;

    if (r <= d_rMin)
      r = d_rMin;
    alglib::spline1ddiff(
      d_radialSplineObject[0], r, Value[0], Value[1], Value[2]);

    return Value;
  }

  double
  AtomCenteredSphericalFunctionPAWProjectorSpline2::getrMinVal() const
  {
    return d_rMin;
  }
} // end of namespace dftfe
