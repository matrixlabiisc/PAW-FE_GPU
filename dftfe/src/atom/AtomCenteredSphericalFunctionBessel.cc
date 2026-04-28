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

#include "AtomCenteredSphericalFunctionBessel.h"
#include "vector"
namespace dftfe
{
  AtomCenteredSphericalFunctionBessel::AtomCenteredSphericalFunctionBessel(
    double       RcParameter,
    double       RmaxParameter,
    unsigned int lParameter,
    double       normalizationConstant)
  {
    d_lQuantumNumber = lParameter;
    d_Rc             = RcParameter;
    d_cutOff         = RmaxParameter;
    using namespace boost::math::quadrature;
    AssertThrow(
      d_lQuantumNumber <= 2,
      dealii::ExcMessage(
        "DFT-FE Error:  Bessel functions only ti;; LQuantumNo 2 is defined"));
    std::vector<double> q1 = {3.141592653589793 / d_Rc,
                              4.493409457909095 / d_Rc,
                              5.76345919689455 / d_Rc};
    std::vector<double> q2 = {6.283185307179586 / d_Rc,
                              7.7252518369375 / d_Rc,
                              9.095011330476355 / d_Rc};

    d_NormalizationConstant = normalizationConstant;
    d_rMinVal               = getRadialValue(0.0);
  }

  double
  AtomCenteredSphericalFunctionBessel::getRadialValue(double     r,
                                                      dftfe::Int threadId) const
  {
    double              Value = 0.0;
    std::vector<double> q1    = {3.141592653589793 / d_Rc,
                                 4.493409457909095 / d_Rc,
                                 5.76345919689455 / d_Rc};
    std::vector<double> q2    = {6.283185307179586 / d_Rc,
                                 7.7252518369375 / d_Rc,
                                 9.095011330476355 / d_Rc};
    double              derJ1 =
      d_lQuantumNumber > 0 ?
                     std::sph_bessel(d_lQuantumNumber - 1, q1[d_lQuantumNumber] * d_Rc) -
          double(d_lQuantumNumber + 1) / (q1[d_lQuantumNumber] * d_Rc) *
            std::sph_bessel(d_lQuantumNumber, q1[d_lQuantumNumber] * d_Rc) :
                     -std::sph_bessel(1, q1[d_lQuantumNumber] * d_Rc);
    double derJ2 =
      d_lQuantumNumber > 0 ?
        std::sph_bessel(d_lQuantumNumber - 1, q2[d_lQuantumNumber] * d_Rc) -
          double(d_lQuantumNumber + 1) / (q2[d_lQuantumNumber] * d_Rc) *
            std::sph_bessel(d_lQuantumNumber, q2[d_lQuantumNumber] * d_Rc) :
        -std::sph_bessel(1, q2[d_lQuantumNumber] * d_Rc);
    double alpha = -q1[d_lQuantumNumber] / q2[d_lQuantumNumber] * derJ1 / derJ2;
    Value =
      r > d_Rc ?
        0.0 :
        (std::sph_bessel(d_lQuantumNumber, q1[d_lQuantumNumber] * r) +
         alpha * (std::sph_bessel(d_lQuantumNumber, q2[d_lQuantumNumber] * r)));
    return Value / d_NormalizationConstant;
  }
  std::vector<double>
  AtomCenteredSphericalFunctionBessel::getDerivativeValue(double r) const
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

  double
  AtomCenteredSphericalFunctionBessel::getrMinVal() const
  {
    return d_rMinVal;
  }
} // end of namespace dftfe
