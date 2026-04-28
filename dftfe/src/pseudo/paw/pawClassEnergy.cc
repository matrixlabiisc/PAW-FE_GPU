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
#include <pawClass.h>
#if defined(DFTFE_WITH_DEVICE)
#  include <pawClassKernelsDevice.h>
#endif
namespace dftfe
{

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::
    evaluateNonLocalHamiltonianElectrostaticsValue(
      const distributedCPUVec<double> &phiTotNodalValues,
      const dftfe::uInt                dofHandlerId)
  {
    // FE Electrostatics

    double            alpha = 1.0;
    double            beta  = 1.0;
    const dftfe::uInt numberNodesPerElement =
      d_BasisOperatorElectroHostPtr->nDofsPerCell();
    const dftfe::uInt numberQuadraturePoints =
      (d_jxwcompensationCharge.begin()->second).size();
    const dftfe::uInt numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();
    d_nonLocalHamiltonianElectrostaticValue.clear();
    for (dftfe::uInt iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        const dftfe::uInt atomId = atomIdsInCurrentProcess[iAtom];
        const dftfe::uInt Znum   = atomicNumber[atomId];
        const dftfe::uInt NumTotalSphericalFunctions =
          d_atomicShapeFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
            Znum);
        d_nonLocalHamiltonianElectrostaticValue[atomId] =
          std::vector<double>(NumTotalSphericalFunctions, 0.0);
      }
    dealii::FEEvaluation<3, -1> feEvalObj(
      d_BasisOperatorElectroHostPtr->matrixFreeData(),
      dofHandlerId,
      d_compensationChargeQuadratureIdElectro);

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    int                                         iElem = 0;
    for (std::set<dftfe::uInt>::iterator it =
           d_atomicShapeFnsContainer->d_feEvaluationMap.begin();
         it != d_atomicShapeFnsContainer->d_feEvaluationMap.end();
         ++it)
      {
        std::vector<double> phiValuesQuadPoints(numberQuadraturePoints, 0.0);
        dftfe::uInt         cell = *it;
        feEvalObj.reinit(cell);
        feEvalObj.read_dof_values_plain(phiTotNodalValues);
        feEvalObj.evaluate(dealii::EvaluationFlags::values);
        for (dftfe::uInt iSubCell = 0;
             iSubCell < d_BasisOperatorElectroHostPtr->matrixFreeData()
                          .n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            subCellPtr =
              d_BasisOperatorElectroHostPtr->matrixFreeData().get_cell_iterator(
                cell, iSubCell, dofHandlerId);
            dealii::CellId subCellId = subCellPtr->id();
            dftfe::uInt    cellIndex =
              d_BasisOperatorElectroHostPtr->cellIndex(subCellId);
            if (d_atomicShapeFnsContainer->atomSupportInElement(cellIndex))
              {
                double *tempVec = phiValuesQuadPoints.data();


                for (dftfe::uInt q_point = 0; q_point < numberQuadraturePoints;
                     ++q_point)
                  {
                    tempVec[q_point] = feEvalObj.get_value(q_point)[iSubCell];
                  }

                std::vector<dftfe::Int> atomIdsInElem =
                  d_atomicShapeFnsContainer->getAtomIdsInElement(cellIndex);
                for (int iAtom = 0; iAtom < atomIdsInElem.size(); iAtom++)
                  {
                    const dftfe::uInt atomId = atomIdsInElem[iAtom];
                    const dftfe::uInt Znum   = atomicNumber[atomId];
                    const dftfe::uInt NumTotalSphericalFunctions =
                      d_atomicShapeFnsContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    std::vector<double> gLValues =
                      d_gLValuesQuadPoints[std::make_pair(atomId, cellIndex)];
                    iElem++;
                    d_BLASWrapperHostPtr->xgemm(
                      'N',
                      'N',
                      1,
                      NumTotalSphericalFunctions,
                      numberQuadraturePoints,
                      &alpha,
                      phiValuesQuadPoints.data(),
                      1,
                      &gLValues[0],
                      numberQuadraturePoints,
                      &beta,
                      d_nonLocalHamiltonianElectrostaticValue[atomId].data(),
                      1);



                  } // iAtom
              }     // if
          }         // subcell
      }             // FEEval iterator
    // for (std::map<dftfe::uInt, std::vector<double>>::iterator it =
    //        d_nonLocalHamiltonianElectrostaticValue.begin();
    //      it != d_nonLocalHamiltonianElectrostaticValue.end();
    //      ++it)
    //   {
    //     dftfe::uInt        atomId  = it->first;
    //     std::vector<double> entries = it->second;
    //     for (int i = 0; i < entries.size(); i++)
    //       std::cout << entries[i] << " ";
    //     std::cout << std::endl;
    //   }
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::computeDeltaExchangeCorrelationEnergy(
    double     &DeltaExchangeCorrelationVal,
    TypeOfField typeOfField)
  {
    double TotalDeltaExchangeCorrelationVal = 0.0;

    const bool isGGA =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);
    std::vector<double>              quad_weights;
    std::vector<std::vector<double>> quad_points;
    getSphericalQuadratureRule(quad_weights, quad_points);
    double TotalDeltaXC = 0.0;

    int         numberofSphericalValues = quad_weights.size();
    dftfe::uInt atomId                  = 0;
    dftfe::uInt numSpinComponents       = D_ij.size();
    if (d_LocallyOwnedAtomId.size() > 0)
      {
        const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        std::vector<dftfe::uInt> atomicNumbers =
          d_atomicProjectorFnsContainer->getAtomicNumbers();
        for (int iAtomList = 0; iAtomList < d_LocallyOwnedAtomId.size();
             iAtomList++)
          {
            std::unordered_map<
              std::string,
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>>
              AEdensityProjectionInputs;
            std::unordered_map<
              std::string,
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>>
              PSdensityProjectionInputs;
            atomId                         = d_LocallyOwnedAtomId[iAtomList];
            const dftfe::uInt   Znum       = atomicNumbers[atomId];
            const dftfe::uInt   RmaxIndex  = d_RmaxAugIndex[Znum];
            std::vector<double> radialGrid = d_radialMesh[Znum];
            std::vector<double> rab        = d_radialJacobianData[Znum];
            dftfe::uInt         RadialMeshSize = radialGrid.size();
            const dftfe::uInt   numberofValues =
              std::min(RmaxIndex + 5, RadialMeshSize);
            AEdensityProjectionInputs["quadpts"] =
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                3 * numberofValues, 0.0);
            PSdensityProjectionInputs["quadpts"] =
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                3 * numberofValues, 0.0);
            AEdensityProjectionInputs["quadWt"] =
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                numberofValues, 0.0);
            PSdensityProjectionInputs["quadWt"] =
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                numberofValues, 0.0);
            std::pair<dftfe::uInt, dftfe::uInt> quadIndexes =
              std::make_pair(0, numberofValues);
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              &PSdensityValsForXC = PSdensityProjectionInputs["densityFunc"];
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              &AEdensityValsForXC = AEdensityProjectionInputs["densityFunc"];
            PSdensityValsForXC.resize(2 * numberofValues, 0.0);
            AEdensityValsForXC.resize(2 * numberofValues, 0.0);
            if (isGGA)
              {
                dftfe::utils::MemoryStorage<double,
                                            dftfe::utils::MemorySpace::HOST>
                  &PSdensityGradForXC =
                    PSdensityProjectionInputs["gradDensityFunc"];
                dftfe::utils::MemoryStorage<double,
                                            dftfe::utils::MemorySpace::HOST>
                  &AEdensityGradForXC =
                    AEdensityProjectionInputs["gradDensityFunc"];
                PSdensityGradForXC.resize(2 * numberofValues * 3, 0.0);
                AEdensityGradForXC.resize(2 * numberofValues * 3, 0.0);
              }
            TotalDeltaExchangeCorrelationVal -= d_coreXC[Znum];
            const dftfe::uInt numberOfProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            const dftfe::uInt numberOfRadialProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
            const dftfe::uInt numberOfProjectorsSq =
              numberOfProjectors * numberOfProjectors;
            double              Yi, Yj;
            std::vector<double> Dij(numberOfProjectors * numberOfProjectors * 2,
                                    0.0);


            // Compute Dij_up and Dij_down
            // For spin unpolarised case, Dij_up = Dij_down
            if (numSpinComponents == 1)
              {
                std::vector<double> DijComp =
                  D_ij[0][typeOfField].find(atomId)->second;
                for (dftfe::uInt iComp = 0; iComp < 2; iComp++)
                  {
                    for (dftfe::uInt iProj = 0;
                         iProj < numberOfProjectors * numberOfProjectors;
                         iProj++)
                      {
                        Dij[iProj +
                            iComp * numberOfProjectors * numberOfProjectors] +=
                          0.5 * (DijComp[iProj]);
                      }
                  }
              }
            else if (numSpinComponents == 2)
              {
                dftfe::uInt shift = numberOfProjectors * numberOfProjectors;
                std::vector<double> DijRho =
                  D_ij[0][typeOfField].find(atomId)->second;
                std::vector<double> DijMagZ =
                  D_ij[1][typeOfField].find(atomId)->second;
                for (dftfe::uInt iProj = 0;
                     iProj < numberOfProjectors * numberOfProjectors;
                     iProj++)
                  {
                    Dij[iProj] += 0.5 * (DijRho[iProj] + DijMagZ[iProj]);
                    Dij[iProj + shift] +=
                      0.5 * (DijRho[iProj] - DijMagZ[iProj]);
                  }
              }

            if (!isGGA)
              {
                double Yi, Yj;
                for (int qpoint = 0; qpoint < numberofSphericalValues; qpoint++)
                  {
                    std::vector<double> SphericalHarmonics(numberOfProjectors *
                                                             numberOfProjectors,
                                                           0.0);
                    // help me.. A better strategy to store this

                    std::vector<double> productOfAEpartialWfc =
                      d_productOfAEpartialWfc[Znum];
                    std::vector<double> productOfPSpartialWfc =
                      d_productOfPSpartialWfc[Znum];
                    double              quadwt = quad_weights[qpoint];
                    std::vector<double> DijYij(numberOfProjectors *
                                                 numberOfProjectors * 2,
                                               0.0);
                    dftfe::uInt shift = numberOfProjectors * numberOfProjectors;
                    int         projIndexI = 0;
                    for (int iProj = 0; iProj < numberOfRadialProjectors;
                         iProj++)
                      {
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>
                          sphFn_i =
                            sphericalFunction.find(std::make_pair(Znum, iProj))
                              ->second;
                        const int lQuantumNo_i = sphFn_i->getQuantumNumberl();
                        for (int mQuantumNumber_i = -lQuantumNo_i;
                             mQuantumNumber_i <= lQuantumNo_i;
                             mQuantumNumber_i++)
                          {
                            sphericalHarmonicUtils::getSphericalHarmonicVal(
                              quad_points[qpoint][0],
                              quad_points[qpoint][1],
                              lQuantumNo_i,
                              mQuantumNumber_i,
                              Yi);

                            int projIndexJ = 0;
                            for (int jProj = 0;
                                 jProj < numberOfRadialProjectors;
                                 jProj++)
                              {
                                std::shared_ptr<
                                  AtomCenteredSphericalFunctionBase>
                                  sphFn_j = sphericalFunction
                                              .find(std::make_pair(Znum, jProj))
                                              ->second;
                                const int lQuantumNo_j =
                                  sphFn_j->getQuantumNumberl();
                                for (int mQuantumNumber_j = -lQuantumNo_j;
                                     mQuantumNumber_j <= lQuantumNo_j;
                                     mQuantumNumber_j++)
                                  {
                                    sphericalHarmonicUtils::
                                      getSphericalHarmonicVal(
                                        quad_points[qpoint][0],
                                        quad_points[qpoint][1],
                                        lQuantumNo_j,
                                        mQuantumNumber_j,
                                        Yj);

                                    SphericalHarmonics[projIndexI *
                                                         numberOfProjectors +
                                                       projIndexJ] = Yi * Yj;
                                    SphericalHarmonics[projIndexJ *
                                                         numberOfProjectors +
                                                       projIndexI] = Yi * Yj;
                                    DijYij[projIndexI * numberOfProjectors +
                                           projIndexJ] =
                                      Yi * Yj *
                                      Dij[projIndexI * numberOfProjectors +
                                          projIndexJ];
                                    DijYij[projIndexI * numberOfProjectors +
                                           projIndexJ + shift] =
                                      Yi * Yj *
                                      Dij[projIndexI * numberOfProjectors +
                                          projIndexJ + shift];
                                    DijYij[projIndexJ * numberOfProjectors +
                                           projIndexI] =
                                      Yi * Yj *
                                      Dij[projIndexJ * numberOfProjectors +
                                          projIndexI];
                                    DijYij[projIndexJ * numberOfProjectors +
                                           projIndexI + shift] =
                                      Yi * Yj *
                                      Dij[projIndexJ * numberOfProjectors +
                                          projIndexI + shift];
                                    projIndexJ++;
                                  } // mQuantumNumber_j

                              } // jProj
                            projIndexI++;
                          } // mQuantumNumber_i



                      } // iProj
                    const char        transA = 'N', transB = 'N';
                    const double      AlphaRho = 1, BetaRho = 0.5;
                    const dftfe::uInt inc = 1;
                    for (dftfe::uInt iComp = 0; iComp < 2; iComp++)
                      {
                        std::vector<double> atomDensityAllelectron =
                          d_atomTypeCoreFlagMap[Znum] ?
                            d_atomCoreDensityAE[Znum] :
                            std::vector<double>(numberofValues, 0.0);
                        std::vector<double> atomDensitySmooth =
                          d_atomTypeCoreFlagMap[Znum] ?
                            d_atomCoreDensityPS[Znum] :
                            std::vector<double>(numberofValues, 0.0);

                        d_BLASWrapperHostPtr->xgemm(transA,
                                                    transB,
                                                    inc,
                                                    numberofValues,
                                                    numberOfProjectorsSq,
                                                    &AlphaRho,
                                                    &DijYij[iComp * shift],
                                                    inc,
                                                    &productOfAEpartialWfc[0],
                                                    numberOfProjectorsSq,
                                                    &BetaRho,
                                                    &atomDensityAllelectron[0],
                                                    inc);
                        d_BLASWrapperHostPtr->xgemm(transA,
                                                    transB,
                                                    inc,
                                                    numberofValues,
                                                    numberOfProjectorsSq,
                                                    &AlphaRho,
                                                    &DijYij[iComp * shift],
                                                    inc,
                                                    &productOfPSpartialWfc[0],
                                                    numberOfProjectorsSq,
                                                    &BetaRho,
                                                    &atomDensitySmooth[0],
                                                    inc);


                        for (int iRad = 0; iRad < numberofValues; iRad++)
                          {
                            PSdensityValsForXC[iRad + iComp * numberofValues] =
                              atomDensitySmooth[iRad];
                            AEdensityValsForXC[iRad + iComp * numberofValues] =
                              atomDensityAllelectron[iRad];
                          }
                      } // iComp
                    d_auxDensityMatrixXCAEPtr->projectDensityStart(
                      AEdensityProjectionInputs);
                    d_auxDensityMatrixXCAEPtr->projectDensityEnd(
                      d_mpiCommParent);
                    d_auxDensityMatrixXCPSPtr->projectDensityStart(
                      PSdensityProjectionInputs);
                    d_auxDensityMatrixXCPSPtr->projectDensityEnd(
                      d_mpiCommParent);
                    std::unordered_map<xcRemainderOutputDataAttributes,
                                       dftfe::utils::MemoryStorage<
                                         double,
                                         dftfe::utils::MemorySpace::HOST>>
                      xDataOutPS;
                    std::unordered_map<xcRemainderOutputDataAttributes,
                                       dftfe::utils::MemoryStorage<
                                         double,
                                         dftfe::utils::MemorySpace::HOST>>
                      cDataOutPS;
                    std::unordered_map<xcRemainderOutputDataAttributes,
                                       dftfe::utils::MemoryStorage<
                                         double,
                                         dftfe::utils::MemorySpace::HOST>>
                      xDataOutAE;
                    std::unordered_map<xcRemainderOutputDataAttributes,
                                       dftfe::utils::MemoryStorage<
                                         double,
                                         dftfe::utils::MemorySpace::HOST>>
                      cDataOutAE;
                    dftfe::utils::MemoryStorage<double,
                                                dftfe::utils::MemorySpace::HOST>
                      &xEnergyOutPS =
                        xDataOutPS[xcRemainderOutputDataAttributes::e];
                    dftfe::utils::MemoryStorage<double,
                                                dftfe::utils::MemorySpace::HOST>
                      &cEnergyOutPS =
                        cDataOutPS[xcRemainderOutputDataAttributes::e];
                    dftfe::utils::MemoryStorage<double,
                                                dftfe::utils::MemorySpace::HOST>
                      &xEnergyOutAE =
                        xDataOutAE[xcRemainderOutputDataAttributes::e];
                    dftfe::utils::MemoryStorage<double,
                                                dftfe::utils::MemorySpace::HOST>
                      &cEnergyOutAE =
                        cDataOutAE[xcRemainderOutputDataAttributes::e];

                    d_excManagerPtr->getExcSSDFunctionalObj()
                      ->computeRhoTauDependentXCData(*d_auxDensityMatrixXCPSPtr,
                                                     quadIndexes,
                                                     xDataOutPS,
                                                     cDataOutPS);
                    d_excManagerPtr->getExcSSDFunctionalObj()
                      ->computeRhoTauDependentXCData(*d_auxDensityMatrixXCAEPtr,
                                                     quadIndexes,
                                                     xDataOutAE,
                                                     cDataOutAE);
                    std::function<double(const dftfe::uInt &)> Integral =
                      [&](const dftfe::uInt &rpoint) {
                        double Val1 =
                          (xEnergyOutAE[rpoint] + cEnergyOutAE[rpoint]);
                        double Val2 =
                          (xEnergyOutPS[rpoint] + cEnergyOutPS[rpoint]);
                        double Value = rab[rpoint] * (Val1 - Val2) *
                                       pow(radialGrid[rpoint], 2);
                        return (Value);
                      };

                    double RadialIntegral =
                      simpsonIntegral(0, RmaxIndex + 1, Integral);
                    TotalDeltaXC += RadialIntegral * quadwt * 4.0 * M_PI;
                    TotalDeltaExchangeCorrelationVal +=
                      RadialIntegral * quadwt * 4.0 * M_PI;


                  } // qpoint

              } // LDA case
            else
              {
                double Yi, Yj;
                dftfe::utils::MemoryStorage<double,
                                            dftfe::utils::MemorySpace::HOST>
                  &PSdensityGradForXC =
                    PSdensityProjectionInputs["gradDensityFunc"];
                dftfe::utils::MemoryStorage<double,
                                            dftfe::utils::MemorySpace::HOST>
                  &AEdensityGradForXC =
                    AEdensityProjectionInputs["gradDensityFunc"];
                for (int qpoint = 0; qpoint < numberofSphericalValues; qpoint++)
                  {
                    std::vector<double> SphericalHarmonics(numberOfProjectors *
                                                             numberOfProjectors,
                                                           0.0);
                    std::vector<double> GradThetaSphericalHarmonics(
                      numberOfProjectors * numberOfProjectors, 0.0);
                    std::vector<double> GradPhiSphericalHarmonics(
                      numberOfProjectors * numberOfProjectors, 0.0);


                    std::vector<double> productOfAEpartialWfc =
                      d_productOfAEpartialWfc[Znum];
                    std::vector<double> productOfPSpartialWfc =
                      d_productOfPSpartialWfc[Znum];

                    std::vector<double> productOfAEpartialWfcValue =
                      d_productOfAEpartialWfcValue[Znum];
                    std::vector<double> productOfPSpartialWfcValue =
                      d_productOfPSpartialWfcValue[Znum];
                    std::vector<double> productOfAEpartialWfcDer =
                      d_productOfAEpartialWfcDer[Znum];
                    std::vector<double> productOfPSpartialWfcDer =
                      d_productOfPSpartialWfcDer[Znum];
                    double              quadwt = quad_weights[qpoint];
                    std::vector<double> DijYij(numberOfProjectors *
                                                 numberOfProjectors * 2,
                                               0.0);
                    std::vector<double> DijGradThetaYij(
                      numberOfProjectors * numberOfProjectors * 2, 0.0);
                    std::vector<double> DijGradPhiYij(numberOfProjectors *
                                                        numberOfProjectors * 2,
                                                      0.0);

                    int         projIndexI = 0;
                    dftfe::uInt shift = numberOfProjectors * numberOfProjectors;
                    for (int iProj = 0; iProj < numberOfRadialProjectors;
                         iProj++)
                      {
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>
                          sphFn_i =
                            sphericalFunction.find(std::make_pair(Znum, iProj))
                              ->second;
                        const int lQuantumNo_i = sphFn_i->getQuantumNumberl();
                        for (int mQuantumNumber_i = -lQuantumNo_i;
                             mQuantumNumber_i <= lQuantumNo_i;
                             mQuantumNumber_i++)
                          {
                            sphericalHarmonicUtils::getSphericalHarmonicVal(
                              quad_points[qpoint][0],
                              quad_points[qpoint][1],
                              lQuantumNo_i,
                              mQuantumNumber_i,
                              Yi);

                            int projIndexJ = 0;
                            for (int jProj = 0;
                                 jProj < numberOfRadialProjectors;
                                 jProj++)
                              {
                                std::shared_ptr<
                                  AtomCenteredSphericalFunctionBase>
                                  sphFn_j = sphericalFunction
                                              .find(std::make_pair(Znum, jProj))
                                              ->second;
                                const int lQuantumNo_j =
                                  sphFn_j->getQuantumNumberl();
                                for (int mQuantumNumber_j = -lQuantumNo_j;
                                     mQuantumNumber_j <= lQuantumNo_j;
                                     mQuantumNumber_j++)
                                  {
                                    sphericalHarmonicUtils::
                                      getSphericalHarmonicVal(
                                        quad_points[qpoint][0],
                                        quad_points[qpoint][1],
                                        lQuantumNo_j,
                                        mQuantumNumber_j,
                                        Yj);

                                    std::vector<double> gradYj =
                                      derivativeOfRealSphericalHarmonic(
                                        lQuantumNo_j,
                                        mQuantumNumber_j,
                                        quad_points[qpoint][0],
                                        quad_points[qpoint][1]);
                                    SphericalHarmonics[projIndexI *
                                                         numberOfProjectors +
                                                       projIndexJ] = Yi * Yj;
                                    DijYij[projIndexI * numberOfProjectors +
                                           projIndexJ] =
                                      Yi * Yj *
                                      Dij[projIndexI * numberOfProjectors +
                                          projIndexJ];
                                    DijYij[projIndexI * numberOfProjectors +
                                           projIndexJ + shift] =
                                      Yi * Yj *
                                      Dij[projIndexI * numberOfProjectors +
                                          projIndexJ + shift];

                                    GradThetaSphericalHarmonics
                                      [projIndexI * numberOfProjectors +
                                       projIndexJ] = Yi * gradYj[0];
                                    double temp =
                                      std::abs(std::sin(
                                        quad_points[qpoint][0])) <= 1E-8 ?
                                        0.0 :
                                        Yi * gradYj[1] /
                                          std::sin(quad_points[qpoint][0]);
                                    GradPhiSphericalHarmonics
                                      [projIndexI * numberOfProjectors +
                                       projIndexJ] = temp;

                                    DijGradThetaYij[projIndexI *
                                                      numberOfProjectors +
                                                    projIndexJ] =
                                      Dij[projIndexI * numberOfProjectors +
                                          projIndexJ] *
                                      Yi * gradYj[0];
                                    DijGradThetaYij[projIndexI *
                                                      numberOfProjectors +
                                                    projIndexJ + shift] =
                                      Dij[projIndexI * numberOfProjectors +
                                          projIndexJ + shift] *
                                      Yi * gradYj[0];
                                    DijGradPhiYij[projIndexI *
                                                    numberOfProjectors +
                                                  projIndexJ] =
                                      Dij[projIndexI * numberOfProjectors +
                                          projIndexJ] *
                                      temp;
                                    DijGradPhiYij[projIndexI *
                                                    numberOfProjectors +
                                                  projIndexJ + shift] =
                                      Dij[projIndexI * numberOfProjectors +
                                          projIndexJ + shift] *
                                      temp;
                                    projIndexJ++;
                                  } // mQuantumNumber_j

                              } // jProj
                            projIndexI++;
                          } // mQuantumNumber_i



                      } // iProj
                    // COmputing Density Values at radial points
                    const char        transA = 'N', transB = 'N';
                    const double      AlphaRho = 1.0, BetaRho = 0.5;
                    const dftfe::uInt inc          = 1;
                    const double      AlphaGradRho = 2.0;
                    const double      BetaGradRho  = 0.5;
                    for (dftfe::uInt iComp = 0; iComp < 2; iComp++)
                      {
                        std::vector<double> atomDensityAllelectron =
                          d_atomTypeCoreFlagMap[Znum] ?
                            d_atomCoreDensityAE[Znum] :
                            std::vector<double>(numberofValues, 0.0);
                        std::vector<double> atomDensitySmooth =
                          d_atomTypeCoreFlagMap[Znum] ?
                            d_atomCoreDensityPS[Znum] :
                            std::vector<double>(numberofValues, 0.0);

                        std::vector<double> atomDensityGradientAllElectron_0 =
                          d_atomTypeCoreFlagMap[Znum] ?
                            d_gradCoreAE[Znum] :
                            std::vector<double>(numberofValues, 0.0);
                        std::vector<double> atomDensityGradientSmooth_0 =
                          d_atomTypeCoreFlagMap[Znum] ?
                            d_gradCorePS[Znum] :
                            std::vector<double>(numberofValues, 0.0);

                        std::vector<double> atomDensityGradientAllElectron_1 =
                          std::vector<double>(numberofValues, 0.0);
                        std::vector<double> atomDensityGradientSmooth_1 =
                          std::vector<double>(numberofValues, 0.0);
                        std::vector<double> atomDensityGradientAllElectron_2 =
                          std::vector<double>(numberofValues, 0.0);
                        std::vector<double> atomDensityGradientSmooth_2 =
                          std::vector<double>(numberofValues, 0.0);

                        d_BLASWrapperHostPtr->xgemm(transA,
                                                    transB,
                                                    inc,
                                                    numberofValues,
                                                    numberOfProjectorsSq,
                                                    &AlphaRho,
                                                    &DijYij[iComp * shift],
                                                    inc,
                                                    &productOfAEpartialWfc[0],
                                                    numberOfProjectorsSq,
                                                    &BetaRho,
                                                    &atomDensityAllelectron[0],
                                                    inc);
                        d_BLASWrapperHostPtr->xgemm(transA,
                                                    transB,
                                                    inc,
                                                    numberofValues,
                                                    numberOfProjectorsSq,
                                                    &AlphaRho,
                                                    &DijYij[iComp * shift],
                                                    inc,
                                                    &productOfPSpartialWfc[0],
                                                    numberOfProjectorsSq,
                                                    &BetaRho,
                                                    &atomDensitySmooth[0],
                                                    inc);

                        // component 0
                        d_BLASWrapperHostPtr->xgemm(
                          transA,
                          transB,
                          inc,
                          numberofValues,
                          numberOfProjectorsSq,
                          &AlphaGradRho,
                          &DijYij[iComp * shift], // to be changed
                          inc,
                          &productOfAEpartialWfcDer[0], // to be changed
                          numberOfProjectorsSq,
                          &BetaGradRho,
                          &atomDensityGradientAllElectron_0[0], // to be changed
                          inc);
                        d_BLASWrapperHostPtr->xgemm(
                          transA,
                          transB,
                          inc,
                          numberofValues,
                          numberOfProjectorsSq,
                          &AlphaGradRho,
                          &DijYij[iComp * shift], // to be changed
                          inc,
                          &productOfPSpartialWfcDer[0], // to be changed
                          numberOfProjectorsSq,
                          &BetaGradRho,
                          &atomDensityGradientSmooth_0[0], // to be changed
                          inc);

                        // component 1
                        d_BLASWrapperHostPtr->xgemm(
                          transA,
                          transB,
                          inc,
                          numberofValues,
                          numberOfProjectorsSq,
                          &AlphaGradRho,
                          &DijGradThetaYij[iComp * shift], // to be changed
                          inc,
                          &productOfAEpartialWfcValue[0],
                          numberOfProjectorsSq,
                          &BetaGradRho,
                          &atomDensityGradientAllElectron_1[0], // to be changed
                          inc);
                        d_BLASWrapperHostPtr->xgemm(
                          transA,
                          transB,
                          inc,
                          numberofValues,
                          numberOfProjectorsSq,
                          &AlphaGradRho,
                          &DijGradThetaYij[iComp * shift], // to be changed
                          inc,
                          &productOfPSpartialWfcValue[0], // to be changed
                          numberOfProjectorsSq,
                          &BetaGradRho,
                          &atomDensityGradientSmooth_1[0], // to be changed
                          inc);

                        // component 2
                        d_BLASWrapperHostPtr->xgemm(
                          transA,
                          transB,
                          inc,
                          numberofValues,
                          numberOfProjectorsSq,
                          &AlphaGradRho,
                          &DijGradPhiYij[iComp * shift], // to be changed
                          inc,
                          &productOfAEpartialWfcValue[0], // to be changed
                          numberOfProjectorsSq,
                          &BetaGradRho,
                          &atomDensityGradientAllElectron_2[0], // to be changed
                          inc);
                        d_BLASWrapperHostPtr->xgemm(
                          transA,
                          transB,
                          inc,
                          numberofValues,
                          numberOfProjectorsSq,
                          &AlphaGradRho,
                          &DijGradPhiYij[iComp * shift], // to be changed
                          inc,
                          &productOfPSpartialWfcValue[0], // to be changed
                          numberOfProjectorsSq,
                          &BetaGradRho,
                          &atomDensityGradientSmooth_2[0], // to be changed
                          inc);



                        for (int iRad = 0; iRad < numberofValues; iRad++)
                          {
                            // LDA constribution
                            PSdensityValsForXC[iRad + iComp * numberofValues] =
                              atomDensitySmooth[iRad];
                            AEdensityValsForXC[iRad + iComp * numberofValues] =
                              atomDensityAllelectron[iRad];


                            // GGA contribution
                            PSdensityGradForXC[3 * iRad + 0 +
                                               3 * iComp * numberofValues] =
                              atomDensityGradientSmooth_0[iRad];
                            PSdensityGradForXC[3 * iRad + 1 +
                                               3 * iComp * numberofValues] =
                              atomDensityGradientSmooth_1[iRad];
                            PSdensityGradForXC[3 * iRad + 2 +
                                               3 * iComp * numberofValues] =
                              atomDensityGradientSmooth_2[iRad];


                            AEdensityGradForXC[3 * iRad + 0 +
                                               3 * iComp * numberofValues] =
                              atomDensityGradientAllElectron_0[iRad];
                            AEdensityGradForXC[3 * iRad + 1 +
                                               3 * iComp * numberofValues] =
                              atomDensityGradientAllElectron_1[iRad];
                            AEdensityGradForXC[3 * iRad + 2 +
                                               3 * iComp * numberofValues] =
                              atomDensityGradientAllElectron_2[iRad];
                          }
                      } // iComp
                    d_auxDensityMatrixXCAEPtr->projectDensityStart(
                      AEdensityProjectionInputs);
                    d_auxDensityMatrixXCAEPtr->projectDensityEnd(
                      d_mpiCommParent);
                    d_auxDensityMatrixXCPSPtr->projectDensityStart(
                      PSdensityProjectionInputs);
                    d_auxDensityMatrixXCPSPtr->projectDensityEnd(
                      d_mpiCommParent);
                    std::unordered_map<xcRemainderOutputDataAttributes,
                                       dftfe::utils::MemoryStorage<
                                         double,
                                         dftfe::utils::MemorySpace::HOST>>
                      xDataOutPS;
                    std::unordered_map<xcRemainderOutputDataAttributes,
                                       dftfe::utils::MemoryStorage<
                                         double,
                                         dftfe::utils::MemorySpace::HOST>>
                      cDataOutPS;
                    std::unordered_map<xcRemainderOutputDataAttributes,
                                       dftfe::utils::MemoryStorage<
                                         double,
                                         dftfe::utils::MemorySpace::HOST>>
                      xDataOutAE;
                    std::unordered_map<xcRemainderOutputDataAttributes,
                                       dftfe::utils::MemoryStorage<
                                         double,
                                         dftfe::utils::MemorySpace::HOST>>
                      cDataOutAE;
                    dftfe::utils::MemoryStorage<double,
                                                dftfe::utils::MemorySpace::HOST>
                      &xEnergyOutPS =
                        xDataOutPS[xcRemainderOutputDataAttributes::e];
                    dftfe::utils::MemoryStorage<double,
                                                dftfe::utils::MemorySpace::HOST>
                      &cEnergyOutPS =
                        cDataOutPS[xcRemainderOutputDataAttributes::e];
                    dftfe::utils::MemoryStorage<double,
                                                dftfe::utils::MemorySpace::HOST>
                      &xEnergyOutAE =
                        xDataOutAE[xcRemainderOutputDataAttributes::e];
                    dftfe::utils::MemoryStorage<double,
                                                dftfe::utils::MemorySpace::HOST>
                      &cEnergyOutAE =
                        cDataOutAE[xcRemainderOutputDataAttributes::e];

                    d_excManagerPtr->getExcSSDFunctionalObj()
                      ->computeRhoTauDependentXCData(*d_auxDensityMatrixXCPSPtr,
                                                     quadIndexes,
                                                     xDataOutPS,
                                                     cDataOutPS);
                    d_excManagerPtr->getExcSSDFunctionalObj()
                      ->computeRhoTauDependentXCData(*d_auxDensityMatrixXCAEPtr,
                                                     quadIndexes,
                                                     xDataOutAE,
                                                     cDataOutAE);
                    std::function<double(const dftfe::uInt &)> Integral =
                      [&](const dftfe::uInt &rpoint) {
                        double Val1 =
                          (xEnergyOutAE[rpoint] + cEnergyOutAE[rpoint]);
                        double Val2 =
                          (xEnergyOutPS[rpoint] + cEnergyOutPS[rpoint]);
                        double Value = rab[rpoint] * (Val1 - Val2) *
                                       pow(radialGrid[rpoint], 2);
                        return (Value);
                      };

                    double RadialIntegral =
                      simpsonIntegral(0, RmaxIndex + 1, Integral);
                    TotalDeltaXC += RadialIntegral * quadwt * 4.0 * M_PI;
                    TotalDeltaExchangeCorrelationVal +=
                      RadialIntegral * quadwt * 4.0 * M_PI;
                  }
              }
          } // iAtomList
      }     // If locallyOwned
    MPI_Barrier(d_mpiCommParent);
    DeltaExchangeCorrelationVal =
      (dealii::Utilities::MPI::sum(TotalDeltaExchangeCorrelationVal,
                                   d_mpiCommParent));

    return (dealii::Utilities::MPI::sum(TotalDeltaXC, d_mpiCommParent));
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeCoreDeltaExchangeCorrelationEnergy()
  {
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        double     TotalDeltaXC = 0.0;
        const bool isGGA =
          (d_excManagerPtr->getExcSSDFunctionalObj()
             ->getDensityBasedFamilyType() == densityFamilyType::GGA);

        if (d_atomTypeCoreFlagMap[*it])
          {
            dftfe::uInt         Znum       = *it;
            std::vector<double> RadialMesh = d_radialMesh[Znum];
            dftfe::uInt         RmaxIndex  = d_RmaxAugIndex[Znum];
            std::unordered_map<
              std::string,
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>>
                                AEdensityProjectionInputs;
            std::vector<double> rab            = d_radialJacobianData[Znum];
            dftfe::uInt         RadialMeshSize = RadialMesh.size();
            const dftfe::uInt   numberofValues = RadialMesh.size();
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              atomCoreDensityAE(numberofValues, 0.0);
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              gradCoreAE(numberofValues, 0.0);
            if (d_atomTypeCoreFlagMap[Znum])
              {
                atomCoreDensityAE.copyFrom(d_atomCoreDensityAE[Znum]);
                gradCoreAE.copyFrom(d_gradCoreAE[Znum]);
              }

            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              atomDensityAllelectron = atomCoreDensityAE;
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              atomGradAllelectron = gradCoreAE;
            AEdensityProjectionInputs["quadpts"] =
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                3 * numberofValues, 0.0);
            AEdensityProjectionInputs["quadWt"] =
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                numberofValues, 0.0);
            std::pair<dftfe::uInt, dftfe::uInt> quadIndexes =
              std::make_pair(0, numberofValues);
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              &AEdensityValsForXC = AEdensityProjectionInputs["densityFunc"];
            AEdensityValsForXC.resize(2 * numberofValues, 0.0);
            if (isGGA)
              {
                dftfe::utils::MemoryStorage<double,
                                            dftfe::utils::MemorySpace::HOST>
                  &AEdensityGradForXC =
                    AEdensityProjectionInputs["gradDensityFunc"];
                AEdensityGradForXC.resize(2 * numberofValues * 3, 0.0);
              }
            for (int iRad = 0; iRad < numberofValues; iRad++)
              {
                AEdensityValsForXC[iRad] = atomDensityAllelectron[iRad] / 2.0;
                AEdensityValsForXC[iRad + numberofValues] =
                  atomDensityAllelectron[iRad] / 2.0;
                if (isGGA)
                  {
                    dftfe::utils::MemoryStorage<double,
                                                dftfe::utils::MemorySpace::HOST>
                      &AEdensityGradForXC =
                        AEdensityProjectionInputs["gradDensityFunc"];
                    AEdensityGradForXC[3 * iRad + 0] =
                      atomGradAllelectron[iRad] / 2.0;
                    AEdensityGradForXC[numberofValues * 3 + 3 * iRad + 0] =
                      atomGradAllelectron[iRad] / 2.0;
                  }
              }
            d_auxDensityMatrixXCAEPtr->projectDensityStart(
              AEdensityProjectionInputs);
            d_auxDensityMatrixXCAEPtr->projectDensityEnd(d_mpiCommParent);
            std::unordered_map<
              xcRemainderOutputDataAttributes,
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>>
              xDataOutAE;
            std::unordered_map<
              xcRemainderOutputDataAttributes,
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>>
              cDataOutAE;
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              &xEnergyOutAE = xDataOutAE[xcRemainderOutputDataAttributes::e];
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              &cEnergyOutAE = cDataOutAE[xcRemainderOutputDataAttributes::e];
            d_excManagerPtr->getExcSSDFunctionalObj()
              ->computeRhoTauDependentXCData(*d_auxDensityMatrixXCAEPtr,
                                             quadIndexes,
                                             xDataOutAE,
                                             cDataOutAE);
            double                                     RadialIntegral = 0.0;
            std::function<double(const dftfe::uInt &)> Integral =
              [&](const dftfe::uInt &rpoint) {
                double Val1 = (xEnergyOutAE[rpoint] + cEnergyOutAE[rpoint]);

                double Value = rab[rpoint] * (Val1)*pow(RadialMesh[rpoint], 2);
                return (Value);
              };

            RadialIntegral = simpsonIntegral(0, numberofValues - 2, Integral);

            TotalDeltaXC = RadialIntegral;
          } // if core
        d_coreXC[*it] = TotalDeltaXC * (4 * M_PI);
        pcout << "Core contribution: " << d_coreXC[*it] << std::endl;
      } // *it
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<double>
  pawClass<ValueType, memorySpace>::getDeltaEnergy(TypeOfField typeOfField)
  {
    // 0Th Term is Delta ZeroijDij
    // 1st Term is Delta KEijDij
    // 2nd Term is Delta CijDij + Delta CijklDIjDkl + DeltaC0
    // 4th Term is is Delta CijDij + Delta CijklDIjDkl + DeltaC0val
    // 5th Term is Delta Exc
    // 6th Term is Delta Exc valence
    // 7th termis \Delta Hij Dij
    // 8th term is coreControbution difference in zeroPotential
    std::map<dftfe::uInt, std::vector<ValueType>> nonLocalHij, nonLocalHijUp,
      nonLocalHijDown;
    if (d_dftParamsPtr->spinPolarized == 0)
      {
        nonLocalHij = d_atomicNonLocalPseudoPotentialConstants
          [CouplingType::HamiltonianEntries];
      }
    else if (d_dftParamsPtr->spinPolarized == 1)
      {
        pcout << "Computing nonLocal Hamiltonaian for spin Up: " << std::endl;
        initialiseExchangeCorrelationEnergyCorrection(0);
        computeNonlocalPseudoPotentialConstants(
          CouplingType::HamiltonianEntries, 0);
        nonLocalHijUp = d_atomicNonLocalPseudoPotentialConstants
          [CouplingType::HamiltonianEntries];

        pcout << "Computing nonLocal Hamiltonaian for spin Down: " << std::endl;
        initialiseExchangeCorrelationEnergyCorrection(1);
        computeNonlocalPseudoPotentialConstants(
          CouplingType::HamiltonianEntries, 1);
        nonLocalHijDown = d_atomicNonLocalPseudoPotentialConstants
          [CouplingType::HamiltonianEntries];
      }

    std::vector<double> dETerms(9, 0.0);
    double              totalZeroPotentialContribution              = 0.0;
    double              totalKineticEnergyContribution              = 0.0;
    double              totalKineticEnergyContributionValence       = 0.0;
    double              totalColEnergyContribution                  = 0.0;
    double              totalColValenceEnergyContribution           = 0.0;
    double              totalExchangeCorrelationContribution        = 0.0;
    double              totalExchangeCorrelationValenceContribution = 0.0;
    double              totalnonLocalHamiltonianContribution        = 0.0;
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    totalExchangeCorrelationContribution =
      computeDeltaExchangeCorrelationEnergy(
        totalExchangeCorrelationValenceContribution, typeOfField);
    std::map<TypeOfField, std::map<dftfe::uInt, std::vector<double>>> &D_ijRho =
      D_ij[0];
    for (int iAtom = 0; iAtom < atomicNumber.size(); iAtom++)
      {
        dftfe::uInt       Znum   = atomicNumber[iAtom];
        dftfe::uInt       atomId = iAtom;
        const dftfe::uInt numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const dftfe::uInt numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        std::vector<double>    DijRho = D_ijRho[typeOfField][atomId];
        std::vector<double>    DijUp, DijDown;
        std::vector<ValueType> deltaHij, deltaHijUp, deltaHijDown;
        std::vector<double>    Zeroij = d_zeroPotentialij[Znum];
        std::vector<double>    KEij   = d_KineticEnergyCorrectionTerm[Znum];
        std::vector<double>    Cij    = d_deltaCij[Znum];
        std::vector<double>    Cijkl  = d_deltaCijkl[Znum];
        if (d_dftParamsPtr->spinPolarized == 0)
          {
            deltaHij = nonLocalHij[atomId];
          }
        else if (d_dftParamsPtr->spinPolarized == 1)
          {
            deltaHijUp   = nonLocalHijUp[atomId];
            deltaHijDown = nonLocalHijDown[atomId];
            DijUp.resize(numberOfProjectors * numberOfProjectors, 0.0);
            DijDown.resize(numberOfProjectors * numberOfProjectors, 0.0);
            std::vector<double> DijMagZ =
              D_ij[1][typeOfField].find(atomId)->second;
            for (dftfe::uInt iProj = 0;
                 iProj < numberOfProjectors * numberOfProjectors;
                 iProj++)
              {
                DijUp[iProj] += 0.5 * (DijRho[iProj] + DijMagZ[iProj]);
                DijDown[iProj] += 0.5 * (DijRho[iProj] - DijMagZ[iProj]);
              }
          }
        double C                               = d_deltaC[Znum];
        double Cvalence                        = d_deltaValenceC[Znum];
        double KEcontribution                  = 0.0;
        double ZeroPotentialContribution       = 0.0;
        double ColoumbContribution             = 0.0;
        double ColoumbContributionValence      = 0.0;
        double nonLocalHamiltonianContribution = 0.0;
        pcout << "Energy nonlocal term: " << std::endl;
        for (int iProj = 0; iProj < numberOfProjectors; iProj++)
          {
            for (int jProj = 0; jProj < numberOfProjectors; jProj++)
              {
                if (d_dftParamsPtr->spinPolarized == 0)
                  {
                    nonLocalHamiltonianContribution +=
                      DijRho[iProj * numberOfProjectors + jProj] *
                      std::real(deltaHij[iProj * numberOfProjectors + jProj]);
                  }
                else if (d_dftParamsPtr->spinPolarized == 1)
                  {
                    nonLocalHamiltonianContribution +=
                      DijUp[iProj * numberOfProjectors + jProj] *
                      std::real(deltaHijUp[iProj * numberOfProjectors + jProj]);
                    nonLocalHamiltonianContribution +=
                      DijDown[iProj * numberOfProjectors + jProj] *
                      std::real(
                        deltaHijDown[iProj * numberOfProjectors + jProj]);
                  }
                KEcontribution += DijRho[iProj * numberOfProjectors + jProj] *
                                  KEij[iProj * numberOfProjectors + jProj];
                ZeroPotentialContribution +=
                  DijRho[iProj * numberOfProjectors + jProj] *
                  Zeroij[iProj * numberOfProjectors + jProj];
                ColoumbContribution +=
                  DijRho[iProj * numberOfProjectors + jProj] *
                  Cij[iProj * numberOfProjectors + jProj];
                double CijkContribution = 0.0;
                for (int lProj = 0; lProj < numberOfProjectors; lProj++)
                  {
                    for (int kProj = 0; kProj < numberOfProjectors; kProj++)
                      {
                        dftfe::uInt index = pow(numberOfProjectors, 3) * iProj +
                                            pow(numberOfProjectors, 2) * jProj +
                                            numberOfProjectors * kProj + lProj;
                        CijkContribution +=
                          DijRho[kProj * numberOfProjectors + lProj] *
                          Cijkl[index];
                      }
                  }
                ColoumbContribution +=
                  CijkContribution * DijRho[iProj * numberOfProjectors + jProj];
              } // jProj
          }     // iProj
        ColoumbContributionValence = ColoumbContribution;
        ColoumbContribution += C;
        ColoumbContributionValence += Cvalence;

        totalZeroPotentialContribution += ZeroPotentialContribution;
        totalKineticEnergyContribution +=
          KEcontribution + d_coreKE.find(Znum)->second;
        totalKineticEnergyContributionValence += KEcontribution;
        totalColEnergyContribution += ColoumbContribution;
        totalColValenceEnergyContribution += ColoumbContributionValence;
        totalnonLocalHamiltonianContribution += nonLocalHamiltonianContribution;
        // pcout << "-------------------------" << std::endl;
        // pcout << totalnonLocalHamiltonianContribution << " "
        //       << totalZeroPotentialContribution << " "
        //       << ZeroPotentialContribution << " "
        //       << totalZeroPotentialContribution << std::endl;



      } // iAtom
    // 0Th Term is Delta ZeroijDij
    // 1st Term is Delta KEijDij
    // 2nd Term is Delta KE valence
    // 3rd Term is Delta CijDij + Delta CijklDIjDkl + DeltaC0
    // 4th Term is is Delta CijDij + Delta CijklDIjDkl + DeltaC0val
    // 5th Term is Delta Exc
    // 6th Term is Delta Exc valence
    // 7th termis \Delta Hij Dij
    dETerms[0] = totalZeroPotentialContribution;
    dETerms[1] = totalKineticEnergyContribution;
    dETerms[2] = totalKineticEnergyContributionValence;
    dETerms[3] = totalColEnergyContribution;
    dETerms[4] = totalColValenceEnergyContribution;
    dETerms[5] = totalExchangeCorrelationContribution;
    dETerms[6] = totalExchangeCorrelationValenceContribution;
    dETerms[7] = totalnonLocalHamiltonianContribution;
    pcout << "dETerms " << std::endl;
    for (int i = 0; i <= 8; i++)
      pcout << dETerms[i] << " ";
    pcout << std::endl;



    return dETerms;
  }
  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::DEVICE>;
#endif

} // namespace dftfe
