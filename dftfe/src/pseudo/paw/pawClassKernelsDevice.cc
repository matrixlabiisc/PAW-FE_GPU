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
#include <pawClassKernelsDevice.h>
#include <BLASWrapper.h>
namespace dftfe
{

  DFTFE_CREATE_KERNEL(
    void,
    combineGradDensityContributionsKernel,
    {
      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += nThreadsPerBlock * nThreadBlock)
        {
          atomGradDensityAllelectron[3 * index] =
            atomDensityGradientAllElectron_0[index];

          atomGradDensityAllelectron[3 * index + 1] =
            atomDensityGradientAllElectron_1[index];

          atomGradDensityAllelectron[3 * index + 2] =
            atomDensityGradientAllElectron_2[index];

          atomGradDensitySmooth[3 * index] = atomDensityGradientSmooth_0[index];

          atomGradDensitySmooth[3 * index + 1] =
            atomDensityGradientSmooth_1[index];

          atomGradDensitySmooth[3 * index + 2] =
            atomDensityGradientSmooth_2[index];
        }
    },
    const dftfe::uInt numberEntries,
    const double     *atomDensityGradientAllElectron_0,
    const double     *atomDensityGradientAllElectron_1,
    const double     *atomDensityGradientAllElectron_2,
    const double     *atomDensityGradientSmooth_0,
    const double     *atomDensityGradientSmooth_1,
    const double     *atomDensityGradientSmooth_2,
    double           *atomGradDensityAllelectron,
    double           *atomGradDensitySmooth);



  DFTFE_CREATE_KERNEL(
    void,
    groupSimpsonIntegralKernel,
    {
      const dftfe::uInt npjsq = numberOfProjectors * numberOfProjectors;
      const dftfe::uInt numberEntries =
        numberOfAtoms * numberOfRadialPoints * npjsq;
      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += nThreadsPerBlock * nThreadBlock)
        {
          const dftfe::uInt iAtom   = index / (npjsq * numberOfRadialPoints);
          const dftfe::uInt index1  = index % (npjsq * numberOfRadialPoints);
          const dftfe::uInt indexIJ = index1 / npjsq;
          const dftfe::uInt rpoint  = index1 % npjsq;
          double            temp    = rpoint < numberOfQuadPoints ?
                                        simpsonQuadWeights[rpoint] * integrandValues[index] :
                                        0.0;
          dftfe::utils::atomicAddWrapper(
            &outputIntegralValues[iAtom * npjsq + indexIJ], temp);
        }
    },
    const dftfe::uInt numberOfRadialPoints,
    const dftfe::uInt numberOfProjectors,
    const dftfe::uInt numberOfAtoms,
    const dftfe::uInt numberOfQuadPoints,
    const double     *simpsonQuadWeights,
    const double     *integrandValues,
    double           *outputIntegralValues);



  DFTFE_CREATE_KERNEL(
    void,
    LDAContributiontoDeltaVxcKernel,
    {
      const dftfe::uInt numberEntries =
        numberOfProjectors * numberOfProjectors * numberOfRadialPoints *
        sphericalQuadBatchSize * numberOfAtoms;
      const dftfe::uInt npjsq = numberOfProjectors * numberOfProjectors;
      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += nThreadsPerBlock * nThreadBlock)
        {
          const dftfe::uInt iAtom =
            index / (npjsq * numberOfRadialPoints * sphericalQuadBatchSize);
          const dftfe::uInt index1 =
            index % (npjsq * numberOfRadialPoints * sphericalQuadBatchSize);
          const dftfe::uInt qpoint  = index1 / (npjsq * numberOfRadialPoints);
          const dftfe::uInt index2  = index1 % (npjsq * numberOfRadialPoints);
          const dftfe::uInt rpoint  = index2 / (npjsq);
          const dftfe::uInt indexIJ = index2 % (npjsq);
          double            temp =
            radialMesh[rpoint] > 1E-8 ?
                         ((allElectronPDEX[iAtom * numberOfRadialPoints *
                                  sphericalQuadBatchSize +
                                qpoint * numberOfRadialPoints + rpoint] +
                allElectronPDEC[iAtom * numberOfRadialPoints *
                                  sphericalQuadBatchSize +
                                qpoint * numberOfRadialPoints + rpoint]) *
                 productOfAllElectronWfc[index2] -
               (pseudoSmoothPDEX[iAtom * numberOfRadialPoints *
                                   sphericalQuadBatchSize +
                                 qpoint * numberOfRadialPoints + rpoint] +
                pseudoSmoothPDEC[iAtom * numberOfRadialPoints *
                                   sphericalQuadBatchSize +
                                 qpoint * numberOfRadialPoints + rpoint]) *
                 productOfPseudoSmoothWfc[index2]) *
                rabValues[rpoint] * radialMesh[rpoint] * radialMesh[rpoint] *
                sphericalQuadWeights[qpoint] *
                productOfSphericalHarmonics[indexIJ + qpoint * npjsq] :
                         0.0;
          dftfe::utils::atomicAddWrapper(
            &deltaVxcIJRadialValues[iAtom * numberOfRadialPoints * npjsq +
                                    (indexIJ * numberOfRadialPoints + rpoint)],
            temp);
        }
    },
    const dftfe::uInt numberOfRadialPoints,
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


  DFTFE_CREATE_KERNEL(
    void,
    GGAContributiontoDeltaVxcKernel,
    {
      const dftfe::uInt numberEntries =
        numberOfProjectors * numberOfProjectors * numberOfRadialPoints *
        sphericalQuadBatchSize * numberOfAtoms;
      const dftfe::uInt npjsq = numberOfProjectors * numberOfProjectors;
      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += nThreadsPerBlock * nThreadBlock)
        {
          const dftfe::uInt iAtom =
            index / (npjsq * numberOfRadialPoints * sphericalQuadBatchSize);
          const dftfe::uInt index1 =
            index % (npjsq * numberOfRadialPoints * sphericalQuadBatchSize);
          const dftfe::uInt qpoint  = index1 / (npjsq * numberOfRadialPoints);
          const dftfe::uInt index2  = index1 % (npjsq * numberOfRadialPoints);
          const dftfe::uInt rpoint  = index2 / (npjsq);
          const dftfe::uInt indexij = index2 % (npjsq);
          const dftfe::uInt indexji =
            (indexij / numberOfProjectors) +
            (indexij % numberOfProjectors) * numberOfProjectors;
          const dftfe::uInt indexIJ   = rpoint * npjsq + indexij;
          const dftfe::uInt indexJI   = rpoint * npjsq + indexji;
          double            ValueAE   = 0.0;
          double            ValuePS   = 0.0;
          double            temp      = 0.0;
          double            termAE    = 0.0;
          double            termOffAE = 0.0;
          double            termPS    = 0.0;
          double            termOffPS = 0.0;
          termAE    = (allElectronPDEX[3 * iAtom * sphericalQuadBatchSize *
                                      numberOfRadialPoints +
                                    3 * rpoint + 2 * spinIndex +
                                    qpoint * numberOfRadialPoints * 3] +
                    allElectronPDEC[3 * iAtom * sphericalQuadBatchSize *
                                      numberOfRadialPoints +
                                    3 * rpoint + 2 * spinIndex +
                                    qpoint * numberOfRadialPoints * 3]);
          termOffAE = (allElectronPDEX[3 * iAtom * sphericalQuadBatchSize *
                                         numberOfRadialPoints +
                                       3 * rpoint + 1 +
                                       qpoint * numberOfRadialPoints * 3] +
                       allElectronPDEC[3 * iAtom * sphericalQuadBatchSize *
                                         numberOfRadialPoints +
                                       3 * rpoint + 1 +
                                       qpoint * numberOfRadialPoints * 3]);
          termPS    = (pseudoSmoothPDEX[3 * iAtom * sphericalQuadBatchSize *
                                       numberOfRadialPoints +
                                     3 * rpoint + 2 * spinIndex +
                                     qpoint * numberOfRadialPoints * 3] +
                    pseudoSmoothPDEC[3 * iAtom * sphericalQuadBatchSize *
                                       numberOfRadialPoints +
                                     3 * rpoint + 2 * spinIndex +
                                     qpoint * numberOfRadialPoints * 3]);
          termOffPS = (pseudoSmoothPDEX[3 * iAtom * sphericalQuadBatchSize *
                                          numberOfRadialPoints +
                                        3 * rpoint + 1 +
                                        qpoint * numberOfRadialPoints * 3] +
                       pseudoSmoothPDEC[3 * iAtom * sphericalQuadBatchSize *
                                          numberOfRadialPoints +
                                        3 * rpoint + 1 +
                                        qpoint * numberOfRadialPoints * 3]);

          ValueAE +=
            (2 * termAE *
               gradDensityXCSpinIndexAE[3 * iAtom * sphericalQuadBatchSize *
                                          numberOfRadialPoints +
                                        3 * rpoint + 0 +
                                        qpoint * numberOfRadialPoints * 3] +
             termOffAE *
               gradDensityXCOtherSpinIndexAE
                 [3 * iAtom * sphericalQuadBatchSize * numberOfRadialPoints +
                  3 * rpoint + 0 + qpoint * numberOfRadialPoints * 3]) *
            (productOfAllElectronWfcDer[indexIJ] *
               productOfSphericalHarmonics[indexij + qpoint * npjsq] +
             productOfAllElectronWfcDer[indexJI] *
               productOfSphericalHarmonics[indexji + qpoint * npjsq]);
          ValueAE +=
            (2 * termAE *
               gradDensityXCSpinIndexAE[3 * iAtom * sphericalQuadBatchSize *
                                          numberOfRadialPoints +
                                        3 * rpoint + 1 +
                                        qpoint * numberOfRadialPoints * 3] +
             termOffAE *
               gradDensityXCOtherSpinIndexAE
                 [3 * iAtom * sphericalQuadBatchSize * numberOfRadialPoints +
                  3 * rpoint + 1 + qpoint * numberOfRadialPoints * 3]) *
            (productOfAllElectronWfcValue[indexIJ] *
               GradThetaSphericalHarmonics[indexij + qpoint * npjsq] +
             productOfAllElectronWfcValue[indexJI] *
               GradThetaSphericalHarmonics[indexji + qpoint * npjsq]);
          ValueAE +=
            (2 * termAE *
               gradDensityXCSpinIndexAE[3 * iAtom * sphericalQuadBatchSize *
                                          numberOfRadialPoints +
                                        3 * rpoint + 2 +
                                        qpoint * numberOfRadialPoints * 3] +
             termOffAE *
               gradDensityXCOtherSpinIndexAE
                 [3 * iAtom * sphericalQuadBatchSize * numberOfRadialPoints +
                  3 * rpoint + 2 + qpoint * numberOfRadialPoints * 3]) *
            (productOfAllElectronWfcValue[indexIJ] *
               GradPhiSphericalHarmonics[indexij + qpoint * npjsq] +
             productOfAllElectronWfcValue[indexJI] *
               GradPhiSphericalHarmonics[indexji + qpoint * npjsq]);


          ValuePS +=
            (2 * termPS *
               gradDensityXCSpinIndexPS[3 * iAtom * sphericalQuadBatchSize *
                                          numberOfRadialPoints +
                                        3 * rpoint + 0 +
                                        qpoint * numberOfRadialPoints * 3] +
             termOffPS *
               gradDensityXCOtherSpinIndexPS
                 [3 * iAtom * sphericalQuadBatchSize * numberOfRadialPoints +
                  3 * rpoint + 0 + qpoint * numberOfRadialPoints * 3]) *
            (productOfPseudoSmoothWfcDer[indexIJ] *
               productOfSphericalHarmonics[indexij + qpoint * npjsq] +
             productOfPseudoSmoothWfcDer[indexJI] *
               productOfSphericalHarmonics[indexji + qpoint * npjsq]);
          ValuePS +=
            (2 * termPS *
               gradDensityXCSpinIndexPS[3 * iAtom * sphericalQuadBatchSize *
                                          numberOfRadialPoints +
                                        3 * rpoint + 1 +
                                        qpoint * numberOfRadialPoints * 3] +
             termOffPS *
               gradDensityXCOtherSpinIndexPS
                 [3 * iAtom * sphericalQuadBatchSize * numberOfRadialPoints +
                  3 * rpoint + 1 + qpoint * numberOfRadialPoints * 3]) *
            (productOfPseudoSmoothWfcValue[indexIJ] *
               GradThetaSphericalHarmonics[indexij + qpoint * npjsq] +
             productOfPseudoSmoothWfcValue[indexJI] *
               GradThetaSphericalHarmonics[indexji + qpoint * npjsq]);
          ValuePS +=
            (2 * termPS *
               gradDensityXCSpinIndexPS[3 * iAtom * sphericalQuadBatchSize *
                                          numberOfRadialPoints +
                                        3 * rpoint + 2 +
                                        qpoint * numberOfRadialPoints * 3] +
             termOffPS *
               gradDensityXCOtherSpinIndexPS
                 [3 * iAtom * sphericalQuadBatchSize * numberOfRadialPoints +
                  3 * rpoint + 2 + qpoint * numberOfRadialPoints * 3]) *
            (productOfPseudoSmoothWfcValue[indexIJ] *
               GradPhiSphericalHarmonics[indexij + qpoint * npjsq] +
             productOfPseudoSmoothWfcValue[indexJI] *
               GradPhiSphericalHarmonics[indexji + qpoint * npjsq]);

          temp = radialMesh[rpoint] > 1E-8 ?
                   (ValueAE - ValuePS) * rabValues[rpoint] *
                     radialMesh[rpoint] * radialMesh[rpoint] *
                     sphericalQuadWeights[qpoint] :
                   0.0;
          dftfe::utils::atomicAddWrapper(
            &deltaVxcIJRadialValues[iAtom * numberOfRadialPoints * npjsq +
                                    (indexij * numberOfRadialPoints + rpoint)],
            temp);
        }
    },
    const dftfe::uInt numberOfRadialPoints,
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
                              double           *deltaVxcIJRadialValues)
    {
      const dftfe::uInt npjsq = numberOfProjectors * numberOfProjectors;
      DFTFE_LAUNCH_KERNEL(
        LDAContributiontoDeltaVxcKernel,
        (dftfe::utils::DEVICE_BLOCK_SIZE + npjsq * numberOfRadialPoints *
                                             sphericalQuadBatchSize *
                                             numberOfAtoms) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>::d_streamId,
        numberOfRadialPoints,
        numberOfProjectors,
        numberOfAtoms,
        sphericalQuadBatchSize,
        spinIndex,
        dftfe::utils::makeDataTypeDeviceCompatible(sphericalQuadWeights),
        dftfe::utils::makeDataTypeDeviceCompatible(productOfSphericalHarmonics),
        dftfe::utils::makeDataTypeDeviceCompatible(radialMesh),
        dftfe::utils::makeDataTypeDeviceCompatible(rabValues),
        dftfe::utils::makeDataTypeDeviceCompatible(productOfAllElectronWfc),
        dftfe::utils::makeDataTypeDeviceCompatible(productOfPseudoSmoothWfc),
        dftfe::utils::makeDataTypeDeviceCompatible(allElectronPDEX),
        dftfe::utils::makeDataTypeDeviceCompatible(allElectronPDEC),
        dftfe::utils::makeDataTypeDeviceCompatible(pseudoSmoothPDEX),
        dftfe::utils::makeDataTypeDeviceCompatible(pseudoSmoothPDEC),
        dftfe::utils::makeDataTypeDeviceCompatible(deltaVxcIJRadialValues));
    }

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
                              double           *deltaVxcIJRadialValues)
    {
      const dftfe::uInt npjsq = numberOfProjectors * numberOfProjectors;
      DFTFE_LAUNCH_KERNEL(
        GGAContributiontoDeltaVxcKernel,
        (dftfe::utils::DEVICE_BLOCK_SIZE + npjsq * numberOfRadialPoints *
                                             sphericalQuadBatchSize *
                                             numberOfAtoms) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>::d_streamId,
        numberOfRadialPoints,
        numberOfProjectors,
        numberOfAtoms,
        sphericalQuadBatchSize,
        spinIndex,
        dftfe::utils::makeDataTypeDeviceCompatible(sphericalQuadWeights),
        dftfe::utils::makeDataTypeDeviceCompatible(productOfSphericalHarmonics),
        dftfe::utils::makeDataTypeDeviceCompatible(GradPhiSphericalHarmonics),
        dftfe::utils::makeDataTypeDeviceCompatible(GradThetaSphericalHarmonics),
        dftfe::utils::makeDataTypeDeviceCompatible(radialMesh),
        dftfe::utils::makeDataTypeDeviceCompatible(rabValues),
        dftfe::utils::makeDataTypeDeviceCompatible(
          productOfAllElectronWfcValue),
        dftfe::utils::makeDataTypeDeviceCompatible(
          productOfPseudoSmoothWfcValue),
        dftfe::utils::makeDataTypeDeviceCompatible(productOfAllElectronWfcDer),
        dftfe::utils::makeDataTypeDeviceCompatible(productOfPseudoSmoothWfcDer),
        dftfe::utils::makeDataTypeDeviceCompatible(gradDensityXCSpinIndexAE),
        dftfe::utils::makeDataTypeDeviceCompatible(
          gradDensityXCOtherSpinIndexAE),
        dftfe::utils::makeDataTypeDeviceCompatible(gradDensityXCSpinIndexPS),
        dftfe::utils::makeDataTypeDeviceCompatible(
          gradDensityXCOtherSpinIndexPS),
        dftfe::utils::makeDataTypeDeviceCompatible(allElectronPDEX),
        dftfe::utils::makeDataTypeDeviceCompatible(allElectronPDEC),
        dftfe::utils::makeDataTypeDeviceCompatible(pseudoSmoothPDEX),
        dftfe::utils::makeDataTypeDeviceCompatible(pseudoSmoothPDEC),
        dftfe::utils::makeDataTypeDeviceCompatible(deltaVxcIJRadialValues));
    }

    void
    groupSimpsonIntegral(const dftfe::uInt numberOfRadialPoints,
                         const dftfe::uInt numberOfProjectors,
                         const dftfe::uInt numberOfAtoms,
                         const dftfe::uInt numberOfQuadPoints,
                         const double     *simpsonQuadWeights,
                         const double     *integrandValues,
                         double           *outputIntegralValues)
    {
      const dftfe::uInt numberEntries = numberOfRadialPoints *
                                        numberOfProjectors *
                                        numberOfProjectors * numberOfAtoms;

      DFTFE_LAUNCH_KERNEL(
        groupSimpsonIntegralKernel,
        (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>::d_streamId,
        numberOfRadialPoints,
        numberOfProjectors,
        numberOfAtoms,
        numberOfQuadPoints,
        dftfe::utils::makeDataTypeDeviceCompatible(simpsonQuadWeights),
        dftfe::utils::makeDataTypeDeviceCompatible(integrandValues),
        dftfe::utils::makeDataTypeDeviceCompatible(outputIntegralValues));
    }



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
      double           *atomGradDensitySmooth)
    {
      DFTFE_LAUNCH_KERNEL(
        combineGradDensityContributionsKernel,
        (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>::d_streamId,
        numberEntries,
        dftfe::utils::makeDataTypeDeviceCompatible(
          atomDensityGradientAllElectron_0),
        dftfe::utils::makeDataTypeDeviceCompatible(
          atomDensityGradientAllElectron_1),
        dftfe::utils::makeDataTypeDeviceCompatible(
          atomDensityGradientAllElectron_2),
        dftfe::utils::makeDataTypeDeviceCompatible(atomDensityGradientSmooth_0),
        dftfe::utils::makeDataTypeDeviceCompatible(atomDensityGradientSmooth_1),
        dftfe::utils::makeDataTypeDeviceCompatible(atomDensityGradientSmooth_2),
        dftfe::utils::makeDataTypeDeviceCompatible(atomGradDensityAllelectron),
        dftfe::utils::makeDataTypeDeviceCompatible(atomGradDensitySmooth));
    }



  } // namespace pawClassKernelsDevice
} // namespace dftfe
