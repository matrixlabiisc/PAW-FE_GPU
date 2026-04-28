// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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

#include "AtomCenteredSphericalFunctionGaussian.h"
#include "vector"
namespace dftfe
{
  AtomCenteredSphericalFunctionGaussian::AtomCenteredSphericalFunctionGaussian(
    double       RcParameter,
    double       RmaxParameter,
    unsigned int lParameter,
    double       normalizationConstant)
  {
    d_lQuantumNumber        = lParameter;
    d_Rc                    = RcParameter;
    d_cutOff                = RmaxParameter;
    d_NormalizationConstant = normalizationConstant;
    d_rMinVal               = getRadialValue(0.0);
  }

  double
  AtomCenteredSphericalFunctionGaussian::getRadialValue(
    double     r,
    dftfe::Int threadId) const
  {
    if (r >= d_cutOff)
      return 0.0;
    double Value = pow(r, 2 * d_lQuantumNumber) * std::exp(-pow((r / d_Rc), 2));
    Value /= d_NormalizationConstant;
    return Value;
  }
  std::vector<double>
  AtomCenteredSphericalFunctionGaussian::getDerivativeValue(double r) const
  {
    if (r >= d_cutOff)
      return std::vector<double>(3, 0.0);
    double derivativeValue =
      std::exp(-pow((r / d_Rc), 2)) *
      (2 * d_lQuantumNumber * pow(r, 2 * d_lQuantumNumber - 1) -
       2 * pow(r, 2 * d_lQuantumNumber + 1) / pow(d_Rc, 2));
    derivativeValue /= d_NormalizationConstant;

    std::vector<double> Vec(3, 0);
    Vec[0] = getRadialValue(r);
    Vec[1] = derivativeValue;

    return (Vec);
  }

  double
  AtomCenteredSphericalFunctionGaussian::getrMinVal() const
  {
    return d_rMinVal;
  }
} // end of namespace dftfe
