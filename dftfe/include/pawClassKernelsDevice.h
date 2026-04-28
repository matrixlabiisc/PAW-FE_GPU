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
// @author Kartick Ramakrishnan
//
#ifndef DFTFE_PAWCLASSDEVICEKERNELS_H
#define DFTFE_PAWCLASSDEVICEKERNELS_H

#include <deviceKernelsGeneric.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherHelpers.h>

namespace dftfe
{
  namespace pawClassKernelsDevice
  {
    void
    LDAContributiontoDeltaVxc(const dftfe::uInt numberOfRadialPoints,
                              const dftfe::uInt numberOfProjectors,
                              const dftfe::uInt numberOfAtoms,
                              const dftfe::uInt sphericalQuadBatchSize,
                              const dftfe::uInt spinIndex,
                              const double     *sphericalQuadWeights,
                              const double     *productOfSphericalHarmonics,
                              const double     *radialMesh,
                              const double     *rabValues,
                              const double     *productOfAllElectronWfc,
                              const double     *productOfPseudoSmoothWfc,
                              const double     *allElectronPDEX,
                              const double     *allElectronPDEC,
                              const double     *pseudoSmoothPDEX,
                              const double     *pseudoSmoothPDEC,
                              double           *deltaVxcIJRadialValues);

    void
    combineGradDensityContributions(
      const dftfe::uInt numberEntries,
      const double     *atomDensityGradientAllElectron_0,
      const double     *atomDensityGradientAllElectron_1,
      const double     *atomDensityGradientAllElectron_2,
      const double     *atomDensityGradientSmooth_0,
      const double     *atomDensityGradientSmooth_1,
      const double     *atomDensityGradientSmooth_2,
      double           *atomGradDensityAllelectron,
      double           *atomGradDensitySmooth);
    void
    GGAContributiontoDeltaVxc(const dftfe::uInt numberOfRadialPoints,
                              const dftfe::uInt numberOfProjectors,
                              const dftfe::uInt numberOfAtoms,
                              const dftfe::uInt sphericalQuadBatchSize,
                              const dftfe::uInt spinIndex,
                              const double     *sphericalQuadWeights,
                              const double     *productOfSphericalHarmonics,
                              const double     *GradPhiSphericalHarmonics,
                              const double     *GradThetaSphericalHarmonics,
                              const double     *radialMesh,
                              const double     *rabValues,
                              const double     *productOfAllElectronWfcValue,
                              const double     *productOfPseudoSmoothWfcValue,
                              const double     *productOfAllElectronWfcDer,
                              const double     *productOfPseudoSmoothWfcDer,
                              const double     *gradDensityXCSpinIndexAE,
                              const double     *gradDensityXCOtherSpinIndexAE,
                              const double     *gradDensityXCSpinIndexPS,
                              const double     *gradDensityXCOtherSpinIndexPS,
                              const double     *allElectronPDEX,
                              const double     *allElectronPDEC,
                              const double     *pseudoSmoothPDEX,
                              const double     *pseudoSmoothPDEC,
                              double           *deltaVxcIJRadialValues);

    void
    groupSimpsonIntegral(const dftfe::uInt numberOfRadialPoints,
                         const dftfe::uInt numberOfProjectors,
                         const dftfe::uInt numberOfAtoms,
                         const dftfe::uInt numberOfQuadPoints,
                         const double     *simpsonQuadWeights,
                         const double     *integrandValues,
                         double           *outputIntegralValues);


  } // namespace pawClassKernelsDevice



} // namespace dftfe

#endif // DFTFE_PAWCLASSDEVICEKERNELS_H
