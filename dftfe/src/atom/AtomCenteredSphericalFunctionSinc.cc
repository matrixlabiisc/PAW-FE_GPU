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

#include "AtomCenteredSphericalFunctionSinc.h"
#include "vector"
namespace dftfe
{
  AtomCenteredSphericalFunctionSinc::AtomCenteredSphericalFunctionSinc(
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
  AtomCenteredSphericalFunctionSinc::getRadialValue(double     r,
                                                    dftfe::Int threadId) const
  {
    if (r >= d_cutOff)
      return 0.0;
    double sincValue = boost::math::sinc_pi(r / d_Rc * M_PI);
    double Value     = r > d_Rc ? 0.0 : sincValue * sincValue;
    if (d_lQuantumNumber > 0)
      Value *= pow(r, d_lQuantumNumber);
    Value /= d_NormalizationConstant;
    return Value;
  }

  std::vector<double>
  AtomCenteredSphericalFunctionSinc::getDerivativeValue(double r) const
  {
    std::vector<double> Value(3, 0.0);
    if (r >= d_cutOff)
      return Value;

    Value[0]               = getRadialValue(r);
    double derivativeValue = 0.0;
    if (r <= 1E-8)
      derivativeValue = 0.0;
    else
      {
        derivativeValue =
          (std::cos(r / d_Rc * M_PI) - boost::math::sinc_pi(r / d_Rc * M_PI)) /
          r;
        derivativeValue /= d_NormalizationConstant * M_PI / d_Rc;
      }
    Value[1] = derivativeValue;
    // Caution second derivative not implemented
    return Value;
  }

  double
  AtomCenteredSphericalFunctionSinc::getrMinVal() const
  {
    return d_rMinVal;
  }
} // end of namespace dftfe
