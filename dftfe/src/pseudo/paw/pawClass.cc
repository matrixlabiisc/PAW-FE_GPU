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
#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
#endif
namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeCompensationChargeL0()
  {
    // Timer for compensation chargeL0
    // dftUtils::printCurrentMemoryUsage(d_mpiCommParent,
    //                                   "PAWClass Compensation Charge L0
    //                                   Begin");
    MPI_Barrier(d_mpiCommParent);
    double timeStart = MPI_Wtime();
    d_bl0QuadValuesAllAtoms.clear();
    d_rhoCoreAtomsRefinedValues.clear();
    d_gradRhoCoreAtomsRefinedValues.clear();
    d_HessianRhoCoreAtomsRefinedValues.clear();
    d_rhoCoreRefinedValues.clear();
    // Reinit the basis operator for electrostatics
    d_BasisOperatorElectroHostPtr->reinit(
      0, 0, d_compensationChargeQuadratureIdElectro, false);
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                     &quadPoints = d_BasisOperatorElectroHostPtr->quadPoints();
    const dftfe::uInt numberNodesPerElement =
      d_BasisOperatorElectroHostPtr->nDofsPerCell();

    const dftfe::uInt numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();

    const dftfe::uInt numberQuadraturePoints =
      d_BasisOperatorElectroHostPtr->nQuadsPerCell();

    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const std::vector<double> &atomCoordinates =
      d_atomicShapeFnsContainer->getAtomCoordinates();
    const std::map<dftfe::uInt, std::vector<double>> &periodicImageCoord =
      d_atomicShapeFnsContainer->getPeriodicImageCoordinatesList();

    const std::map<dftfe::uInt, std::vector<dftfe::Int>> &imageIdsMap =
      d_atomicShapeFnsContainer->getImageIds();
    const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction = d_atomicShapeFnsContainer->getSphericalFunctions();
    const std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();
    MPI_Barrier(d_mpiCommParent);
    // Initialisation of the map

    dftfe::uInt numberOfCells = d_BasisOperatorElectroHostPtr->nCells();

    // Pre-fill maps for all cells to ensure safe access later.
    for (int iElem = 0; iElem < numberOfCells; iElem++)
      {
        const dealii::CellId cell_id =
          d_BasisOperatorElectroHostPtr->cellID(iElem);
        if (d_atomicShapeFnsContainer->atomSupportInElement(iElem))
          {
            d_bl0QuadValuesAllAtoms[cell_id] =
              std::vector<double>(numberQuadraturePoints, 0.0);
            d_rhoCoreRefinedValues[cell_id] =
              std::vector<double>(numberQuadraturePoints, 0.0);
          }
      } // iElem


#pragma omp parallel for num_threads(d_nOMPThreads)
    for (dftfe::uInt iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        const dftfe::uInt atomId = atomIdsInCurrentProcess[iAtom];

        // FIX: Safely perform all map lookups before using the results.
        const auto it_imageIds = imageIdsMap.find(atomId);
        if (it_imageIds == imageIdsMap.end())
          continue;
        const std::vector<dftfe::Int> &imageIds = it_imageIds->second;

        const auto it_imageCoords = periodicImageCoord.find(atomId);
        if (it_imageCoords == periodicImageCoord.end())
          continue;
        const std::vector<double> &imageCoordinates = it_imageCoords->second;

        const dftfe::uInt Znum = atomicNumber[atomId];

        const auto it_sphFn = sphericalFunction.find(std::make_pair(Znum, 0));
        if (it_sphFn == sphericalFunction.end())
          continue;
        std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
          it_sphFn->second;

        const dftfe::uInt imageIdsSize = imageCoordinates.size() / 3;

        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        dealii::Point<3> nuclearCoordinates(atomCoordinates[3 * atomId + 0],
                                            atomCoordinates[3 * atomId + 1],
                                            atomCoordinates[3 * atomId + 2]);
        double           dL0     = d_DeltaL0coeff[Znum];
        double           RmaxAug = d_RmaxAug[Znum];

        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             iElemComp++)
          {
            const dftfe::uInt elementIndex =
              elementIndexesInAtomCompactSupport[iElemComp];
            typename dealii::DoFHandler<3>::active_cell_iterator cell =
              d_BasisOperatorElectroHostPtr->getCellIterator(elementIndex);

            if (cell->is_locally_owned())
              {
                // FIX: Use .at() for safe access to pre-filled maps.
                std::vector<double> &quadvalues =
                  d_bl0QuadValuesAllAtoms.at(cell->id());
                std::vector<double> &rhoCoreRefinedValues =
                  d_rhoCoreRefinedValues.at(cell->id());
                for (int iImageAtomCount = 0; iImageAtomCount < imageIdsSize;
                     ++iImageAtomCount)
                  {
                    std::vector<double> *rhoCoreAtomRefinedValuesPtr = nullptr;
                    std::vector<double> *gradRhoCoreAtomRefinedValuesPtr =
                      nullptr;
                    std::vector<double> *HessianRhoCoreAtomRefinedValuesPtr =
                      nullptr;

// FIX: Protect non-thread-safe map insertion with a critical section.
#pragma omp critical
                    {
                      std::vector<double> &vec =
                        d_rhoCoreAtomsRefinedValues[imageIds[iImageAtomCount]]
                                                   [cell->id()];
                      std::vector<double> &gradVec =
                        d_gradRhoCoreAtomsRefinedValues
                          [imageIds[iImageAtomCount]][cell->id()];
                      std::vector<double> &HessianVec =
                        d_HessianRhoCoreAtomsRefinedValues
                          [imageIds[iImageAtomCount]][cell->id()];
                      if (vec.empty())
                        {
                          vec.resize(numberQuadraturePoints, 0.0);
                          gradVec.resize(3 * numberQuadraturePoints, 0.0);
                          HessianVec.resize(3 * 3 * numberQuadraturePoints,
                                            0.0);
                        }
                      rhoCoreAtomRefinedValuesPtr        = &vec;
                      gradRhoCoreAtomRefinedValuesPtr    = &gradVec;
                      HessianRhoCoreAtomRefinedValuesPtr = &HessianVec;
                    }

                    dealii::Point<3> chargePoint(0.0, 0.0, 0.0);
                    if (iImageAtomCount == 0)
                      {
                        chargePoint = nuclearCoordinates;
                      }
                    else
                      {
                        chargePoint[0] =
                          imageCoordinates[3 * iImageAtomCount + 0];
                        chargePoint[1] =
                          imageCoordinates[3 * iImageAtomCount + 1];
                        chargePoint[2] =
                          imageCoordinates[3 * iImageAtomCount + 2];
                      }

                    double x[3], r, theta, phi;

                    for (dftfe::uInt iQuadPoint = 0;
                         iQuadPoint < numberQuadraturePoints;
                         ++iQuadPoint)
                      {
                        const double *quad_point_ptr =
                          quadPoints.data() +
                          elementIndex * numberQuadraturePoints * 3 +
                          iQuadPoint * 3;
                        x[0] = quad_point_ptr[0] - chargePoint[0];
                        x[1] = quad_point_ptr[1] - chargePoint[1];
                        x[2] = quad_point_ptr[2] - chargePoint[2];

                        sphericalHarmonicUtils::convertCartesianToSpherical(
                          x, r, theta, phi);
                        double rdensity = r;
                        if (rdensity < 1E-4)
                          {
                            rdensity = 1E-4;
                            x[0]     = (1.0e-4) / std::sqrt(3.0);
                            x[1]     = (1.0e-4) / std::sqrt(3.0);
                            x[2]     = (1.0e-4) / std::sqrt(3.0);
                          }
                        if (r <= sphFn->getRadialCutOff())
                          {
                            double radialVal = sphFn->getRadialValue(r);
                            double term      = dL0 * radialVal / sqrt(4 * M_PI);

// FIX: Use atomic operations for all writes to shared memory.
#pragma omp atomic update
                            quadvalues[iQuadPoint] += term;
                          }

                        std::vector<double> densityValue;
                        getRadialCoreDensity(Znum, rdensity, densityValue);

#pragma omp atomic update
                        rhoCoreRefinedValues[iQuadPoint] += densityValue[0];


                        (*rhoCoreAtomRefinedValuesPtr)[iQuadPoint] +=
                          densityValue[0];

                        (*gradRhoCoreAtomRefinedValuesPtr)[3 * iQuadPoint +
                                                           0] +=
                          densityValue[1] * x[0];

                        (*gradRhoCoreAtomRefinedValuesPtr)[3 * iQuadPoint +
                                                           1] +=
                          densityValue[1] * x[1];

                        (*gradRhoCoreAtomRefinedValuesPtr)[3 * iQuadPoint +
                                                           2] +=
                          densityValue[1] * x[2];
                        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                          {
                            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                              {
                                double temp = (densityValue[2] -
                                               densityValue[1] / rdensity) *
                                              (x[iDim] / rdensity) *
                                              (x[jDim] / rdensity);
                                if (iDim == jDim)
                                  temp += densityValue[1] / rdensity;

                                (*HessianRhoCoreAtomRefinedValuesPtr)
                                  [9 * iQuadPoint + 3 * iDim + jDim] += temp;
                              }
                          }

                      } // quad loop
                  }     // image atom loop

                // This operation is safe because the key is unique to the
                // thread.

              } // cell locallyOwned
          }     // iElemComp
      }         // iAtom

    // Timer for compensation chargeL0
    // Reinit the basis operator for electrostatics
    initRhoCoreCorrectionValues();
    MPI_Barrier(d_mpiCommParent);
    double timeEnd = MPI_Wtime();
    pcout << "computeCompensationChargeL0 timer: " << timeEnd - timeStart
          << std::endl;
    MPI_Barrier(d_mpiCommParent);
    // dftUtils::printCurrentMemoryUsage(d_mpiCommParent,
    //                                   "PAWClass Compensation Charge L0 End");
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initRhoCoreCorrectionValues()
  {
    // Timer for compensation chargeL0
    // dftUtils::printCurrentMemoryUsage(d_mpiCommParent,
    //                                   "PAWClass Compensation Charge L0
    //                                   Begin");
    MPI_Barrier(d_mpiCommParent);
    d_rhoCoreCorrectionValues.clear();
    d_rhoCoreAtomsCorrectionValues.clear();
    d_gradRhoCoreAtomsCorrectionValues.clear();
    d_HessianRhoCoreAtomsCorrectionValues.clear();
    // Reinit the basis operator for electrostatics
    d_BasisOperatorHostPtr->reinit(0, 0, d_densityQuadratureId, false);
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &quadPoints = d_BasisOperatorHostPtr->quadPoints();

    const dftfe::uInt numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();


    const dftfe::uInt numberQuadraturePoints =
      d_BasisOperatorHostPtr->nQuadsPerCell();

    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const std::vector<double> &atomCoordinates =
      d_atomicShapeFnsContainer->getAtomCoordinates();
    const std::map<dftfe::uInt, std::vector<double>> &periodicImageCoord =
      d_atomicShapeFnsContainer->getPeriodicImageCoordinatesList();

    const std::map<dftfe::uInt, std::vector<dftfe::Int>> &imageIdsMap =
      d_atomicShapeFnsContainer->getImageIds();
    const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction = d_atomicShapeFnsContainer->getSphericalFunctions();
    const std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();
    MPI_Barrier(d_mpiCommParent);
    // Initialisation of the map

    dftfe::uInt numberOfCells = d_BasisOperatorHostPtr->nCells();

    // Initialize the map for cells owned by this process. This ensures that
    // .at() in the parallel section will not fail for locally owned cells.
    for (int iElem = 0; iElem < numberOfCells; iElem++)
      {
        if (d_atomicShapeFnsContainer->atomSupportInElement(iElem))
          {
            d_rhoCoreCorrectionValues[d_BasisOperatorHostPtr->cellID(iElem)] =
              std::vector<double>(numberQuadraturePoints, 0.0);
          }
      } // iElem


#pragma omp parallel for num_threads(d_nOMPThreads)
    for (dftfe::uInt iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        const dftfe::uInt atomId = atomIdsInCurrentProcess[iAtom];

        // FIX: Safely get data before the main work by checking find() results.
        const auto it_imageIds = imageIdsMap.find(atomId);
        if (it_imageIds == imageIdsMap.end())
          continue;
        const std::vector<dftfe::Int> &imageIds = it_imageIds->second;

        const auto it_imageCoords = periodicImageCoord.find(atomId);
        if (it_imageCoords == periodicImageCoord.end())
          continue;
        const std::vector<double> &imageCoordinates = it_imageCoords->second;

        const dftfe::uInt Znum = atomicNumber[atomId];

        const auto it_sphFn = sphericalFunction.find(std::make_pair(Znum, 0));
        if (it_sphFn == sphericalFunction.end())
          continue;
        std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
          it_sphFn->second;

        const dftfe::uInt imageIdsSize = imageCoordinates.size() / 3;

        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        dealii::Point<3> nuclearCoordinates(atomCoordinates[3 * atomId + 0],
                                            atomCoordinates[3 * atomId + 1],
                                            atomCoordinates[3 * atomId + 2]);
        double           RmaxAug = d_RmaxAug[Znum];

        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             iElemComp++)
          {
            const dftfe::uInt elementIndex =
              elementIndexesInAtomCompactSupport[iElemComp];
            typename dealii::DoFHandler<3>::active_cell_iterator cell =
              d_BasisOperatorHostPtr->getCellIterator(elementIndex);
            if (cell->is_locally_owned())
              {
                // FIX: Use .at() for read-only access, which is faster and safe
                // because we pre-filled the map for all locally owned cells.
                std::vector<double> &rhoCoreCorrectionValues =
                  d_rhoCoreCorrectionValues.at(cell->id());

                for (int iImageAtomCount = 0; iImageAtomCount < imageIdsSize;
                     ++iImageAtomCount)
                  {
                    std::vector<double> *rhoCoreAtomCorrectionValuesPtr =
                      nullptr;
                    std::vector<double> *gradRhoCoreAtomCorrectionValuesPtr =
                      nullptr;
                    std::vector<double> *HessianRhoCoreAtomCorrectionValuesPtr =
                      nullptr;
// FIX: Use a critical section to protect non-thread-safe map insertion.
#pragma omp critical
                    {
                      // This block ensures only one thread at a time can create
                      // a new entry in the map, preventing data structure
                      // corruption.
                      std::vector<double> &vec = d_rhoCoreAtomsCorrectionValues
                        [imageIds[iImageAtomCount]][cell->id()];
                      std::vector<double> &gradvec =
                        d_gradRhoCoreAtomsCorrectionValues
                          [imageIds[iImageAtomCount]][cell->id()];
                      std::vector<double> &Hessianvec =
                        d_HessianRhoCoreAtomsCorrectionValues
                          [imageIds[iImageAtomCount]][cell->id()];
                      if (vec.empty())
                        {
                          vec.resize(numberQuadraturePoints, 0.0);
                          gradvec.resize(3 * numberQuadraturePoints, 0.0);
                          Hessianvec.resize(9 * numberQuadraturePoints, 0.0);
                        }
                      rhoCoreAtomCorrectionValuesPtr        = &vec;
                      gradRhoCoreAtomCorrectionValuesPtr    = &gradvec;
                      HessianRhoCoreAtomCorrectionValuesPtr = &Hessianvec;
                    }

                    dealii::Point<3> chargePoint(0.0, 0.0, 0.0);
                    if (iImageAtomCount == 0)
                      {
                        chargePoint = nuclearCoordinates;
                      }
                    else
                      {
                        chargePoint[0] =
                          imageCoordinates[3 * iImageAtomCount + 0];
                        chargePoint[1] =
                          imageCoordinates[3 * iImageAtomCount + 1];
                        chargePoint[2] =
                          imageCoordinates[3 * iImageAtomCount + 2];
                      }
                    double x[3];
                    double r, theta, phi;

                    for (dftfe::uInt iQuadPoint = 0;
                         iQuadPoint < numberQuadraturePoints;
                         ++iQuadPoint)
                      {
                        const double *quad_point_ptr =
                          quadPoints.data() +
                          elementIndex * numberQuadraturePoints * 3 +
                          iQuadPoint * 3;
                        x[0] = quad_point_ptr[0] - chargePoint[0];
                        x[1] = quad_point_ptr[1] - chargePoint[1];
                        x[2] = quad_point_ptr[2] - chargePoint[2];

                        sphericalHarmonicUtils::convertCartesianToSpherical(
                          x, r, theta, phi);
                        if (r < 1E-4)
                          {
                            r    = 1E-4;
                            x[0] = (1.0e-4) / std::sqrt(3.0);
                            x[1] = (1.0e-4) / std::sqrt(3.0);
                            x[2] = (1.0e-4) / std::sqrt(3.0);
                          }
                        std::vector<double> densityValue;
                        getRadialCoreDensity(Znum, r, densityValue);


#pragma omp atomic update
                        rhoCoreCorrectionValues[iQuadPoint] += densityValue[0];


                        (*rhoCoreAtomCorrectionValuesPtr)[iQuadPoint] +=
                          densityValue[0];

                        (*gradRhoCoreAtomCorrectionValuesPtr)[3 * iQuadPoint +
                                                              0] +=
                          densityValue[1] * x[0];

                        (*gradRhoCoreAtomCorrectionValuesPtr)[3 * iQuadPoint +
                                                              1] +=
                          densityValue[1] * x[1];

                        (*gradRhoCoreAtomCorrectionValuesPtr)[3 * iQuadPoint +
                                                              2] +=
                          densityValue[1] * x[2];
                        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                          {
                            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                              {
                                double temp =
                                  (densityValue[2] - densityValue[1] / r) *
                                  (x[iDim] / r) * (x[jDim] / r);
                                if (iDim == jDim)
                                  temp += densityValue[1] / r;

                                (*HessianRhoCoreAtomCorrectionValuesPtr)
                                  [9 * iQuadPoint + 3 * iDim + jDim] += temp;
                              }
                          }

                      } // quad loop
                  }     // image atom loop
              }         // cell locallyOwned
          }             // iElemComp
      }                 // iAtom

    MPI_Barrier(d_mpiCommParent);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeCompensationChargeL0(
    const std::vector<double> &scalingFactorPerAtom)
  {
    // Timer for compensation chargeL0
    // dftUtils::printCurrentMemoryUsage(d_mpiCommParent,
    //                                   "PAWClass Compensation Charge L0
    //                                   Begin");
    MPI_Barrier(d_mpiCommParent);
    double timeStart = MPI_Wtime();
    d_bl0QuadValuesAllAtoms.clear();
    d_imageIdFromAtomCompactSupportMap.clear();
    d_g0ValuesQuadPoints.clear();
    d_BasisOperatorElectroHostPtr->reinit(
      0, 0, d_compensationChargeQuadratureIdElectro, false);
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                     &quadPoints = d_BasisOperatorElectroHostPtr->quadPoints();
    const dftfe::uInt numberNodesPerElement =
      d_BasisOperatorElectroHostPtr->nDofsPerCell();
    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      numberNodesPerElement);
    const dftfe::uInt numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    const std::map<dftfe::uInt, std::vector<dftfe::Int>> &imageIdsMap =
      d_atomicShapeFnsContainer->getImageIds();
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                     &JxW = d_BasisOperatorElectroHostPtr->JxWBasisData();
    const dftfe::uInt numberQuadraturePoints =
      d_BasisOperatorElectroHostPtr->nQuadsPerCell();

    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const std::vector<double> &atomCoordinates =
      d_atomicShapeFnsContainer->getAtomCoordinates();
    const std::map<dftfe::uInt, std::vector<double>> &periodicImageCoord =
      d_atomicShapeFnsContainer->getPeriodicImageCoordinatesList();
    const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction = d_atomicShapeFnsContainer->getSphericalFunctions();
    const std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();
    MPI_Barrier(d_mpiCommParent);
    // Initialisation of the map

    dftfe::uInt numberOfCells = d_BasisOperatorElectroHostPtr->nCells();

    for (int iElem = 0; iElem < numberOfCells; iElem++)
      {
        if (d_atomicShapeFnsContainer->atomSupportInElement(iElem))
          {
            d_bl0QuadValuesAllAtoms[d_BasisOperatorElectroHostPtr->cellID(
              iElem)] = std::vector<double>(numberQuadraturePoints, 0.0);
          }
      } // iElem
    for (dftfe::uInt iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        const dftfe::uInt        atomId = atomIdsInCurrentProcess[iAtom];
        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        const dftfe::uInt numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        for (int iElem = 0; iElem < numberElementsInAtomCompactSupport; iElem++)
          {
            const dftfe::uInt elementIndex =
              elementIndexesInAtomCompactSupport[iElem];
            d_g0ValuesQuadPoints[std::make_pair(atomId, elementIndex)] =
              std::vector<double>();
            d_imageIdFromAtomCompactSupportMap[std::make_pair(atomId,
                                                              elementIndex)] =
              -1;
          }
      }

#pragma omp        parallel for num_threads(d_nOMPThreads)
    for (dftfe::uInt iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        const dftfe::uInt   atomId = atomIdsInCurrentProcess[iAtom];
        const dftfe::uInt   Znum   = atomicNumber[atomId];
        const double        scalingFactor = scalingFactorPerAtom[9 * atomId];
        std::vector<double> imageCoordinates =
          periodicImageCoord.find(atomId)->second;
        const auto it_imageIds = imageIdsMap.find(atomId);
        if (it_imageIds == imageIdsMap.end())
          continue;
        const std::vector<dftfe::Int> &imageIds = it_imageIds->second;
        const dftfe::uInt imageIdsSize = imageCoordinates.size() / 3;
        std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
          sphericalFunction.find(std::make_pair(Znum, 0))->second;
        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        dealii::Point<3> nuclearCoordinates(atomCoordinates[3 * atomId + 0],
                                            atomCoordinates[3 * atomId + 1],
                                            atomCoordinates[3 * atomId + 2]);
        double           dL0 = d_DeltaL0coeff[Znum];
        // pcout << "Delta dL0: " << dL0 << " Znum: " << Znum << std::endl;
        double RmaxAug = d_RmaxAug[Znum];

        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             iElemComp++)
          {
            const dftfe::uInt elementIndex =
              elementIndexesInAtomCompactSupport[iElemComp];
            typename dealii::DoFHandler<3>::active_cell_iterator cell =
              d_BasisOperatorElectroHostPtr->getCellIterator(elementIndex);
            if (cell->is_locally_owned())
              {
                std::vector<double> &quadvalues =
                  d_bl0QuadValuesAllAtoms.find(cell->id())->second;
                std::vector<double> quadValuesAtom(numberQuadraturePoints, 0.0);
                for (int iImageAtomCount = 0; iImageAtomCount < imageIdsSize;
                     ++iImageAtomCount)
                  {
                    bool             compactSupport = false;
                    dealii::Point<3> chargePoint(0.0, 0.0, 0.0);
                    if (iImageAtomCount == 0)
                      {
                        chargePoint = nuclearCoordinates;
                      }
                    else
                      {
                        chargePoint[0] =
                          imageCoordinates[3 * iImageAtomCount + 0];
                        chargePoint[1] =
                          imageCoordinates[3 * iImageAtomCount + 1];
                        chargePoint[2] =
                          imageCoordinates[3 * iImageAtomCount + 2];
                      }
                    double x[3];
                    double sphericalHarmonicVal, radialVal,
                      sphericalFunctionValue;
                    double      r, theta, phi, angle;
                    dftfe::uInt atomImageIndex = atomId;
                    for (dftfe::uInt iQuadPoint = 0;
                         iQuadPoint < numberQuadraturePoints;
                         ++iQuadPoint)
                      {
                        x[0] = *(quadPoints.data() +
                                 elementIndex * numberQuadraturePoints * 3 +
                                 iQuadPoint * 3 + 0) -
                               chargePoint[0];
                        x[1] = *(quadPoints.data() +
                                 elementIndex * numberQuadraturePoints * 3 +
                                 iQuadPoint * 3 + 1) -
                               chargePoint[1];
                        x[2] = *(quadPoints.data() +
                                 elementIndex * numberQuadraturePoints * 3 +
                                 iQuadPoint * 3 + 2) -
                               chargePoint[2];
                        sphericalHarmonicUtils::convertCartesianToSpherical(
                          x, r, theta, phi);
                        if (r <= sphFn->getRadialCutOff())
                          {
                            radialVal =
                              sphFn->getRadialValue(r) / scalingFactor;
#pragma omp atomic update
                            quadvalues[iQuadPoint] +=
                              dL0 * radialVal / sqrt(4 * M_PI);

                            quadValuesAtom[iQuadPoint] +=
                              dL0 * radialVal / sqrt(4 * M_PI) *
                              JxW[elementIndex * numberQuadraturePoints +
                                  iQuadPoint];
                            compactSupport = true;
                          } // inside r <= Rmax


                      } // quad loop
                    if (compactSupport)
                      d_imageIdFromAtomCompactSupportMap
                        .find(std::make_pair(atomId, elementIndex))
                        ->second = imageIds[iImageAtomCount];
                  } // image atom loop
                d_g0ValuesQuadPoints.find(std::make_pair(atomId, elementIndex))
                  ->second = quadValuesAtom;
              } // cell locallyOwned

          } // iElemComp

      } // iAtom
    // Timer for compensation chargeL0
    MPI_Barrier(d_mpiCommParent);
    double timeEnd = MPI_Wtime();
    pcout << "computeCompensationChargeL0 timer: " << timeEnd - timeStart
          << std::endl;
    MPI_Barrier(d_mpiCommParent);
    // dftUtils::printCurrentMemoryUsage(d_mpiCommParent,
    //                                   "PAWClass Compensation Charge L0
    //                                   End");
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeCompensationCharge(
    TypeOfField typeOfField)
  {
    const dftfe::uInt numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    dealii::QIterated<3> quadratureHigh(
      dealii::QGauss<1>(d_dftParamsPtr->QuadratureOrderNuclearCharge),
      d_dftParamsPtr->QuadratureCopyNuclearCharge);
    const dftfe::uInt numberQuadraturePoints = quadratureHigh.size();



    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const dftfe::uInt one = 1;
    (*d_bQuadValuesAllAtoms).clear();

    MPI_Barrier(d_mpiCommParent);
    if (d_bl0QuadValuesAllAtoms.size() > 0)
      {
        for (std::map<dealii::CellId, std::vector<double>>::iterator it =
               d_bl0QuadValuesAllAtoms.begin();
             it != d_bl0QuadValuesAllAtoms.end();
             ++it)
          {
            std::vector<double> &ValueL0 = it->second;
            std::vector<double>  Temp;
            (*d_bQuadValuesAllAtoms).find(it->first)->second.clear();
            // for (dftfe::uInt q_point = 0; q_point < numberQuadraturePoints;
            //      q_point++)
            //   {
            //     Temp.push_back(ValueL0[q_point]);
            //   }
            (*d_bQuadValuesAllAtoms)[it->first] = ValueL0;
          }
      }
    MPI_Barrier(d_mpiCommParent);
    const std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();
    MPI_Barrier(d_mpiCommParent);
    std::map<TypeOfField, std::map<dftfe::uInt, std::vector<double>>> &D_ijRho =
      D_ij[0];
    double temp = 0.0;
    for (dftfe::uInt iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        dftfe::uInt atomId = atomIdsInCurrentProcess[iAtom];
        dftfe::uInt Znum   = atomicNumber[atomId];
        dftfe::uInt numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const dftfe::uInt   npjsq = numberOfProjectors * numberOfProjectors;
        std::vector<double> Tij   = d_ProductOfQijShapeFnAtQuadPoints[atomId];
        std::vector<double> Dij   = D_ijRho[typeOfField][atomId];
        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        for (dftfe::uInt iElem = 0; iElem < numberElementsInAtomCompactSupport;
             iElem++)
          {
            const dftfe::uInt elementIndex =
              elementIndexesInAtomCompactSupport[iElem];
            std::vector<double> &quadvalues =
              (*d_bQuadValuesAllAtoms)
                .find(d_BasisOperatorElectroHostPtr->cellID(elementIndex))
                ->second;
            for (dftfe::uInt q_point = 0; q_point < numberQuadraturePoints;
                 ++q_point)
              {
                dftfe::uInt loc =
                  iElem * npjsq * numberQuadraturePoints + q_point * npjsq;
                d_BLASWrapperHostPtr->xdot(
                  npjsq, &Tij[loc], 1, &Dij[0], 1, &temp);
                quadvalues[q_point] += temp;

              } // q_point

          } // iElem
      }     // iAtom loop
    MPI_Barrier(d_mpiCommParent);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeCompensationChargeMemoryOpt(
    TypeOfField typeOfField)
  {
    // dftUtils::printCurrentMemoryUsage(
    //   d_mpiCommParent, "PAWClass Compensation Charge MemOpt Begin");
    const dftfe::uInt numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    dealii::QIterated<3> quadratureHigh(
      dealii::QGauss<1>(d_dftParamsPtr->QuadratureOrderNuclearCharge),
      d_dftParamsPtr->QuadratureCopyNuclearCharge);
    const dftfe::uInt numberQuadraturePoints = quadratureHigh.size();


    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const dftfe::uInt one = 1;
    (*d_bQuadValuesAllAtoms).clear();

    MPI_Barrier(d_mpiCommParent);
    if (d_bl0QuadValuesAllAtoms.size() > 0)
      {
        for (std::map<dealii::CellId, std::vector<double>>::iterator it =
               d_bl0QuadValuesAllAtoms.begin();
             it != d_bl0QuadValuesAllAtoms.end();
             ++it)
          {
            std::vector<double> &ValueL0 = it->second;
            (*d_bQuadValuesAllAtoms).find(it->first)->second.clear();
            (*d_bQuadValuesAllAtoms)[it->first] = ValueL0;
          }
      }
    MPI_Barrier(d_mpiCommParent);
    const std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();
    MPI_Barrier(d_mpiCommParent);
    std::map<TypeOfField, std::map<dftfe::uInt, std::vector<double>>> &D_ijRho =
      D_ij[0];
    for (dftfe::uInt iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        dftfe::uInt atomId = atomIdsInCurrentProcess[iAtom];
        dftfe::uInt Znum   = atomicNumber[atomId];
        dftfe::uInt numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        dftfe::uInt         npjsq = numberOfProjectors * numberOfProjectors;
        std::vector<double> gLValuesAtQuadPoints =
          d_shapeFnAtQuadPoints[atomId];
        std::vector<double> Dij = D_ijRho[typeOfField][atomId];
        std::vector<double> Tij = d_productOfMultipoleClebshGordon[Znum];
        dftfe::uInt         numShapeFunctions =
          d_atomicShapeFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
            Znum);
        std::vector<double> deltaL(numShapeFunctions, 0.0);
        const char          transA = 'N', transB = 'N';
        const double        Alpha = 1.0, Beta = 0.0;
        const dftfe::uInt   inc  = 1;
        double              temp = 0.0;
        d_BLASWrapperHostPtr->xgemm(transA,
                                    transB,
                                    inc,
                                    numShapeFunctions,
                                    npjsq,
                                    &Alpha,
                                    &Dij[0],
                                    inc,
                                    &Tij[0],
                                    npjsq,
                                    &Beta,
                                    &deltaL[0],
                                    inc);



        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        for (dftfe::uInt iElem = 0; iElem < numberElementsInAtomCompactSupport;
             iElem++)
          {
            const dftfe::uInt elementIndex =
              elementIndexesInAtomCompactSupport[iElem];
            std::vector<double> &quadvalues =
              (*d_bQuadValuesAllAtoms)
                .find(d_BasisOperatorElectroHostPtr->cellID(elementIndex))
                ->second;
            for (dftfe::uInt q_point = 0; q_point < numberQuadraturePoints;
                 ++q_point)
              {
                dftfe::uInt loc =
                  iElem * numShapeFunctions * numberQuadraturePoints +
                  q_point * numShapeFunctions;
                d_BLASWrapperHostPtr->xdot(numShapeFunctions,
                                           &gLValuesAtQuadPoints[loc],
                                           1,
                                           &deltaL[0],
                                           1,
                                           &temp);

                quadvalues[q_point] += temp;
              } // q_point

          } // iElem
      }     // iAtom loop
    MPI_Barrier(d_mpiCommParent);
    // dftUtils::printCurrentMemoryUsage(
    //   d_mpiCommParent, "PAWClass Compensation Charge MemOpt End");
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::
    computeAtomDependentCompensationChargeMemoryOpt(TypeOfField typeOfField)
  {
    // dftUtils::printCurrentMemoryUsage(
    //   d_mpiCommParent, "PAWClass Compensation Charge MemOpt Begin");
    d_bAtomsValuesQuadPoints.clear();
    const dftfe::uInt numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    dealii::QIterated<3> quadratureHigh(
      dealii::QGauss<1>(d_dftParamsPtr->QuadratureOrderNuclearCharge),
      d_dftParamsPtr->QuadratureCopyNuclearCharge);
    const dftfe::uInt numberQuadraturePoints = quadratureHigh.size();
    d_BasisOperatorElectroHostPtr->reinit(
      0, 0, d_compensationChargeQuadratureIdElectro, false);
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const dftfe::uInt              one = 1;
    const std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();
    MPI_Barrier(d_mpiCommParent);
    std::map<TypeOfField, std::map<dftfe::uInt, std::vector<double>>> &D_ijRho =
      D_ij[0];
    const std::map<dftfe::uInt, std::vector<dftfe::Int>> &imageIdsMap =
      d_atomicShapeFnsContainer->getImageIds();
    for (dftfe::uInt iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        dftfe::uInt atomId      = atomIdsInCurrentProcess[iAtom];
        dftfe::uInt Znum        = atomicNumber[atomId];
        const auto  it_imageIds = imageIdsMap.find(atomId);
        if (it_imageIds == imageIdsMap.end())
          continue;
        const std::vector<dftfe::Int> &imageIds = it_imageIds->second;
        dftfe::uInt                    numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        dftfe::uInt         npjsq = numberOfProjectors * numberOfProjectors;
        std::vector<double> gLValuesAtQuadPoints =
          d_shapeFnAtQuadPoints[atomId];
        std::vector<double> Dij = D_ijRho[typeOfField][atomId];
        std::vector<double> Tij = d_productOfMultipoleClebshGordon[Znum];
        dftfe::uInt         numShapeFunctions =
          d_atomicShapeFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
            Znum);
        std::vector<double> deltaL(numShapeFunctions, 0.0);
        const char          transA = 'N', transB = 'N';
        const double        Alpha = 1.0, Beta = 0.0;
        const dftfe::uInt   inc  = 1;
        double              temp = 0.0;
        d_BLASWrapperHostPtr->xgemm(transA,
                                    transB,
                                    inc,
                                    numShapeFunctions,
                                    npjsq,
                                    &Alpha,
                                    &Dij[0],
                                    inc,
                                    &Tij[0],
                                    npjsq,
                                    &Beta,
                                    &deltaL[0],
                                    inc);



        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        for (dftfe::uInt iElem = 0; iElem < numberElementsInAtomCompactSupport;
             iElem++)
          {
            const dftfe::uInt elementIndex =
              elementIndexesInAtomCompactSupport[iElem];
            if (d_g0ValuesQuadPoints.find(
                  std::make_pair(atomId, elementIndex)) ==
                d_g0ValuesQuadPoints.end())
              continue;
            dftfe::Int imageId = d_imageIdFromAtomCompactSupportMap
                                   .find(std::make_pair(atomId, elementIndex))
                                   ->second;
            std::vector<double> quadValues =
              d_g0ValuesQuadPoints.find(std::make_pair(atomId, elementIndex))
                ->second;
            const std::vector<double> &jxw =
              d_jxwcompensationCharge
                .find(d_BasisOperatorElectroHostPtr
                        ->d_cellIndexToCellIdMap[elementIndex])
                ->second;
            for (dftfe::uInt q_point = 0; q_point < numberQuadraturePoints;
                 ++q_point)
              {
                dftfe::uInt loc =
                  iElem * numShapeFunctions * numberQuadraturePoints +
                  q_point * numShapeFunctions;
                d_BLASWrapperHostPtr->xdot(numShapeFunctions,
                                           &gLValuesAtQuadPoints[loc],
                                           1,
                                           &deltaL[0],
                                           1,
                                           &temp);

                quadValues[q_point] += temp * jxw[q_point];
              } // q_point
            d_bAtomsValuesQuadPoints[imageId][elementIndex] = quadValues;


          } // iElem

      } // iAtom loop
    MPI_Barrier(d_mpiCommParent);
    // dftUtils::printCurrentMemoryUsage(
    //   d_mpiCommParent, "PAWClass Compensation Charge MemOpt End");
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeCompensationChargeCoeffMemoryOpt()
  {
    // dftUtils::printCurrentMemoryUsage(
    //   d_mpiCommParent, "PAWClass Compensation Charge Coeff MemOpt Begin");

    double timeStart = MPI_Wtime();
    d_BasisOperatorElectroHostPtr->reinit(
      0, 0, d_compensationChargeQuadratureIdElectro, false);
    const dftfe::uInt numberNodesPerElement =
      d_BasisOperatorElectroHostPtr->nDofsPerCell();
    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      numberNodesPerElement);
    const dftfe::uInt numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    dealii::QIterated<3> quadratureHigh(
      dealii::QGauss<1>(d_dftParamsPtr->QuadratureOrderNuclearCharge),
      d_dftParamsPtr->QuadratureCopyNuclearCharge);
    dealii::FEValues<3> fe_values(
      d_BasisOperatorElectroHostPtr->matrixFreeData()
        .get_dof_handler(d_BasisOperatorElectroHostPtr->d_dofHandlerID)
        .get_fe(),
      quadratureHigh,
      dealii::update_JxW_values | dealii::update_quadrature_points);
    d_jxwcompensationCharge.clear();
    const dftfe::uInt numberQuadraturePoints = quadratureHigh.size();
    if (d_bl0QuadValuesAllAtoms.size() > 0)
      {
        for (std::map<dealii::CellId, std::vector<double>>::iterator it =
               d_bl0QuadValuesAllAtoms.begin();
             it != d_bl0QuadValuesAllAtoms.end();
             ++it)
          {
            const dftfe::uInt cellIndex =
              d_BasisOperatorElectroHostPtr->d_cellIdToCellIndexMap[it->first];
            dealii::DoFHandler<3>::active_cell_iterator cell =
              d_BasisOperatorElectroHostPtr->getCellIterator(cellIndex);
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                std::vector<double> jxw(numberQuadraturePoints, 0.0);
                for (dftfe::uInt iQuad = 0; iQuad < numberQuadraturePoints;
                     iQuad++)
                  {
                    jxw[iQuad] = fe_values.JxW(iQuad);
                  }
                d_jxwcompensationCharge[it->first] = jxw;


              } // cell
          }     // it
      }         // if
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const std::vector<double> &atomCoordinates =
      d_atomicShapeFnsContainer->getAtomCoordinates();
    const std::map<dftfe::uInt, std::vector<double>> &periodicImageCoord =
      d_atomicShapeFnsContainer->getPeriodicImageCoordinatesList();
    const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction = d_atomicShapeFnsContainer->getSphericalFunctions();
    const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      projectorFunction =
        d_atomicProjectorFnsContainer->getSphericalFunctions();
    const std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();
    pcout << "Debug Line 1083" << std::endl;
    d_shapeFnAtQuadPoints.clear();
    d_gLValuesQuadPoints.clear();
    std::vector<
      std::map<std::pair<dftfe::uInt, dftfe::uInt>, std::vector<double>>>
      local_gLValues(d_nOMPThreads);
    std::vector<std::map<dftfe::uInt, std::vector<double>>> local_shapeFn(
      d_nOMPThreads);
    std::vector<double> IntegralValueperAtom(9 * atomicNumber.size(), 0.0);
    std::vector<double> IntegralValue(9, 0.0);
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &quadPoints = d_BasisOperatorElectroHostPtr->quadPoints();
#pragma omp parallel for num_threads(d_nOMPThreads)
    for (dftfe::uInt iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        const dftfe::uInt   atomId = atomIdsInCurrentProcess[iAtom];
        const dftfe::uInt   Znum   = atomicNumber[atomId];
        std::vector<double> imageCoordinates =
          periodicImageCoord.find(atomId)->second;
        const dftfe::uInt imageIdsSize = imageCoordinates.size() / 3;
        std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
          sphericalFunction.find(std::make_pair(Znum, 0))->second;
        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        dealii::Point<3>    nuclearCoordinates(atomCoordinates[3 * atomId + 0],
                                            atomCoordinates[3 * atomId + 1],
                                            atomCoordinates[3 * atomId + 2]);
        double              RmaxAug   = d_RmaxAug[Znum];
        std::vector<double> multipole = d_multipole[Znum];
        const dftfe::uInt   NumRadialSphericalFunctions =
          d_atomicShapeFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        const dftfe::uInt NumTotalSphericalFunctions =
          d_atomicShapeFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
            Znum);
        const dftfe::uInt NumProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const dftfe::uInt NumRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        std::vector<double> tempCoeff(numberElementsInAtomCompactSupport *
                                        numberQuadraturePoints *
                                        NumTotalSphericalFunctions,
                                      0.0);
        // pcout << "DEBUG Line 622"
        //       << "iAtom and elements: " << iAtom << " "
        //       << numberElementsInAtomCompactSupport << std::endl;
        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             iElemComp++)
          {
            const dftfe::uInt elementIndex =
              elementIndexesInAtomCompactSupport[iElemComp];
            typename dealii::DoFHandler<3>::active_cell_iterator cell =
              d_BasisOperatorElectroHostPtr->getCellIterator(elementIndex);

            if (cell->is_locally_owned())
              {
                std::vector<double> gLValuesQuadPoints(
                  numberQuadraturePoints * NumTotalSphericalFunctions, 0.0);

                const std::vector<double> &jxw =
                  d_jxwcompensationCharge
                    .find(d_BasisOperatorElectroHostPtr
                            ->d_cellIndexToCellIdMap[elementIndex])
                    ->second;

                // New loop order: qPoint -> iImageAtomCount -> alpha ->
                // mQuantumNumber
                for (int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints;
                     ++iQuadPoint)
                  {
                    const double qp_x =
                      *(quadPoints.data() +
                        elementIndex * numberQuadraturePoints * 3 +
                        iQuadPoint * 3 + 0);
                    const double qp_y =
                      *(quadPoints.data() +
                        elementIndex * numberQuadraturePoints * 3 +
                        iQuadPoint * 3 + 1);
                    const double qp_z =
                      *(quadPoints.data() +
                        elementIndex * numberQuadraturePoints * 3 +
                        iQuadPoint * 3 + 2);

                    for (int iImageAtomCount = 0;
                         iImageAtomCount < imageIdsSize;
                         ++iImageAtomCount)
                      {
                        dealii::Point<3> chargePoint(0.0, 0.0, 0.0);
                        if (iImageAtomCount == 0)
                          {
                            chargePoint = nuclearCoordinates;
                          }
                        else
                          {
                            chargePoint[0] =
                              imageCoordinates[3 * iImageAtomCount + 0];
                            chargePoint[1] =
                              imageCoordinates[3 * iImageAtomCount + 1];
                            chargePoint[2] =
                              imageCoordinates[3 * iImageAtomCount + 2];
                          }
                        double x[3];
                        x[0] = qp_x - chargePoint[0];
                        x[1] = qp_y - chargePoint[1];
                        x[2] = qp_z - chargePoint[2];

                        double r, theta, phi;
                        sphericalHarmonicUtils::convertCartesianToSpherical(
                          x, r, theta, phi);

                        dftfe::uInt Lindex = 0;
                        for (dftfe::uInt alpha = 0;
                             alpha < NumRadialSphericalFunctions;
                             ++alpha)
                          {
                            std::shared_ptr<AtomCenteredSphericalFunctionBase>
                              sphFn = sphericalFunction
                                        .find(std::make_pair(Znum, alpha))
                                        ->second;
                            int lQuantumNumber = sphFn->getQuantumNumberl();

                            // Check radial cutoff before entering the m-loop
                            if (r <= sphFn->getRadialCutOff())
                              {
                                double radialVal = sphFn->getRadialValue(r);
                                for (int mQuantumNumber = -lQuantumNumber;
                                     mQuantumNumber <= lQuantumNumber;
                                     mQuantumNumber++)
                                  {
                                    double sphericalHarmonicVal,
                                      sphericalFunctionValue;
                                    sphericalHarmonicUtils::
                                      getSphericalHarmonicVal(
                                        theta,
                                        phi,
                                        lQuantumNumber,
                                        mQuantumNumber,
                                        sphericalHarmonicVal);

                                    sphericalFunctionValue =
                                      radialVal * sphericalHarmonicVal;

                                    dftfe::uInt loc =
                                      iElemComp * (numberQuadraturePoints *
                                                   NumTotalSphericalFunctions) +
                                      iQuadPoint * NumTotalSphericalFunctions +
                                      Lindex;

                                    tempCoeff[loc] += sphericalFunctionValue;

                                    gLValuesQuadPoints
                                      [Lindex * numberQuadraturePoints +
                                       iQuadPoint] +=
                                      jxw[iQuadPoint] * sphericalFunctionValue;

                                    IntegralValueperAtom[atomId * 9 + Lindex] +=
                                      sphericalFunctionValue * jxw[iQuadPoint] *
                                      pow(r, lQuantumNumber) *
                                      sphericalHarmonicVal;

                                    Lindex++;
                                  } // mQuantumNumber
                              }     // inside r <= Rmax
                            else
                              {
                                // If outside cutoff, advance Lindex by the
                                // number of m values to keep it synchronized.
                                Lindex += (2 * lQuantumNumber + 1);
                              }
                          } // alpha
                      }     // image atom loop
                  }         // quad loop
                local_gLValues[omp_get_thread_num()]
                              [std::make_pair(atomId, elementIndex)] =
                                gLValuesQuadPoints;
              } // cell is locally owned
          }     // iElemComp
        local_shapeFn[omp_get_thread_num()][atomId] = tempCoeff;


      } // iAtom
    for (int i = 0; i < d_nOMPThreads; ++i)
      {
        d_gLValuesQuadPoints.insert(local_gLValues[i].begin(),
                                    local_gLValues[i].end());
        d_shapeFnAtQuadPoints.insert(local_shapeFn[i].begin(),
                                     local_shapeFn[i].end());
      }
    int iAtom = 0;
    MPI_Allreduce(MPI_IN_PLACE,
                  &IntegralValueperAtom[0],
                  9 * atomicNumber.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpiCommParent);

    // pcout << "Indvidual shape function errors: " << std::endl;
    for (iAtom = 0; iAtom < atomicNumber.size(); iAtom++)
      {
        // pcout << "atomId: " << iAtom << " with Znum: " << atomicNumber[iAtom]
        //       << " ";
        for (int L = 0; L < 9; L++)
          {
            // pcout << IntegralValueperAtom[iAtom * 9 + L] << " ";
            IntegralValue[L] += IntegralValueperAtom[iAtom * 9 + L] - 1;
          }
        // pcout << std::endl;
      }
    for (int L = 0; L < 9; L++)
      pcout << "Net error for L: " << L << " " << IntegralValue[L] << std::endl;
    MPI_Barrier(d_mpiCommParent);
    double timeEnd = MPI_Wtime();
    pcout << "computeCompensationChargeCoeffMemoryOpt Timer: "
          << timeEnd - timeStart << std::endl;
    computeCompensationChargeL0(IntegralValueperAtom);

    for (dftfe::uInt atomId = 0; atomId < atomicNumber.size(); atomId++)
      {
        std::vector<double> &tempCoeff =
          d_shapeFnAtQuadPoints.find(atomId)->second;
        if (d_shapeFnAtQuadPoints.find(atomId) != d_shapeFnAtQuadPoints.end())
          {
            std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
              d_atomicShapeFnsContainer
                ->d_elementIndexesInAtomCompactSupport[atomId];
            const dftfe::uInt numberElementsInAtomCompactSupport =
              elementIndexesInAtomCompactSupport.size();
            for (dftfe::uInt iElemComp = 0;
                 iElemComp < numberElementsInAtomCompactSupport;
                 iElemComp++)
              {
                const dftfe::uInt elementIndex =
                  elementIndexesInAtomCompactSupport[iElemComp];
                std::vector<double> &gLValuesQuadPoints =
                  d_gLValuesQuadPoints.find(std::pair(atomId, elementIndex))
                    ->second;
                for (dftfe::uInt iQuadPoint = 0;
                     iQuadPoint < numberQuadraturePoints;
                     iQuadPoint++)
                  {
                    for (dftfe::uInt Lindex = 0; Lindex < 9; Lindex++)
                      {
                        const double scalingFactor =
                          IntegralValueperAtom[9 * atomId + Lindex];
                        dftfe::uInt loc =
                          iElemComp * (numberQuadraturePoints * 9) +
                          iQuadPoint * 9 + Lindex;
                        tempCoeff[loc] /= scalingFactor;
                        gLValuesQuadPoints[Lindex * numberQuadraturePoints +
                                           iQuadPoint] /= scalingFactor;
                      } // Lindex
                  }     // iQuadPoint
              }         // iElemeComp
          }             // if atomId present
      }                 // atomId
    // dftUtils::printCurrentMemoryUsage(
    //   d_mpiCommParent, "PAWClass Compensation Charge Coeff MemOpt End");
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<ValueType, memorySpace> &
  pawClass<ValueType, memorySpace>::getCouplingMatrix(CouplingType couplingtype)
  {
    if (couplingtype == CouplingType::HamiltonianEntries)
      {
        std::vector<ValueType> Entries;
        if (!d_HamiltonianCouplingMatrixEntriesUpdated)
          {
            // dftfe::utils::MemoryStorage<ValueType,
            //                             dftfe::utils::MemorySpace::HOST>
            //                                 couplingEntriesHost;
            const std::vector<dftfe::uInt> atomIdsInProcessor =
              d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
            std::vector<dftfe::uInt> atomicNumber =
              d_atomicProjectorFnsContainer->getAtomicNumbers();
            for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
              {
                dftfe::uInt atomId = atomIdsInProcessor[iAtom];
                dftfe::uInt Zno    = atomicNumber[atomId];
                dftfe::uInt numberSphericalFunctions =
                  d_atomicProjectorFnsContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
                for (dftfe::uInt alpha_i = 0;
                     alpha_i < numberSphericalFunctions;
                     alpha_i++)
                  {
                    for (dftfe::uInt alpha_j = 0;
                         alpha_j < numberSphericalFunctions;
                         alpha_j++)
                      {
                        dftfe::uInt index =
                          alpha_i * numberSphericalFunctions + alpha_j;
                        ValueType V = d_atomicNonLocalPseudoPotentialConstants
                          [CouplingType::HamiltonianEntries][atomId][index];
                        Entries.push_back(V);
                      }
                  }
              }
          }
        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          {
            if (!d_HamiltonianCouplingMatrixEntriesUpdated)
              {
                d_couplingMatrixEntries[couplingtype].resize(Entries.size());
                d_couplingMatrixEntries[couplingtype].copyFrom(Entries);
                d_HamiltonianCouplingMatrixEntriesUpdated = true;
              }

            return d_couplingMatrixEntries[couplingtype];
          }
#if defined(DFTFE_WITH_DEVICE)
        else
          {
            if (!d_HamiltonianCouplingMatrixEntriesUpdated)
              {
                std::vector<ValueType> EntriesPadded;
                d_nonLocalOperator->paddingCouplingMatrix(
                  Entries, EntriesPadded, CouplingStructure::dense);
                d_couplingMatrixEntries[couplingtype].resize(
                  EntriesPadded.size());
                d_couplingMatrixEntries[couplingtype].copyFrom(EntriesPadded);
                d_HamiltonianCouplingMatrixEntriesUpdated = true;
              }
            return d_couplingMatrixEntries[couplingtype];
          }
#endif
      }
    else if (couplingtype == CouplingType::OverlapEntries)
      {
        std::vector<ValueType> Entries;
        if (!d_overlapCouplingMatrixEntriesUpdated)
          {
            const std::vector<dftfe::uInt> atomIdsInProcessor =
              d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
            std::vector<dftfe::uInt> atomicNumber =
              d_atomicProjectorFnsContainer->getAtomicNumbers();
            for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
              {
                dftfe::uInt atomId = atomIdsInProcessor[iAtom];
                dftfe::uInt Zno    = atomicNumber[atomId];
                dftfe::uInt numberSphericalFunctions =
                  d_atomicProjectorFnsContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
                for (dftfe::uInt alpha_i = 0;
                     alpha_i < numberSphericalFunctions;
                     alpha_i++)
                  {
                    for (dftfe::uInt alpha_j = 0;
                         alpha_j < numberSphericalFunctions;
                         alpha_j++)
                      {
                        dftfe::uInt index =
                          alpha_i * numberSphericalFunctions + alpha_j;
                        ValueType V = d_atomicNonLocalPseudoPotentialConstants
                          [CouplingType::OverlapEntries][Zno][index];
                        Entries.push_back(V);
                      }
                  }
              }
          }
        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          {
            if (!d_overlapCouplingMatrixEntriesUpdated)
              {
                d_couplingMatrixEntries[couplingtype].resize(Entries.size());
                d_couplingMatrixEntries[couplingtype].copyFrom(Entries);
                d_overlapCouplingMatrixEntriesUpdated = true;
              }

            return d_couplingMatrixEntries[couplingtype];
          }
#if defined(DFTFE_WITH_DEVICE)
        else
          {
            if (!d_overlapCouplingMatrixEntriesUpdated)
              {
                std::vector<ValueType> EntriesPadded;
                d_nonLocalOperator->paddingCouplingMatrix(
                  Entries, EntriesPadded, CouplingStructure::dense);
                d_couplingMatrixEntries[couplingtype].resize(
                  EntriesPadded.size());
                d_couplingMatrixEntries[couplingtype].copyFrom(EntriesPadded);
                d_overlapCouplingMatrixEntriesUpdated = true;
              }
            return d_couplingMatrixEntries[couplingtype];
          }
#endif
      }
    else if (couplingtype == CouplingType::inverseOverlapEntries)
      {
        std::vector<ValueType> Entries;
        if (!d_inverseCouplingMatrixEntriesUpdated)
          {
            const std::vector<dftfe::uInt> atomIdsInProcessor =
              d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
            std::vector<dftfe::uInt> atomicNumber =
              d_atomicProjectorFnsContainer->getAtomicNumbers();
            for (int kPoint = 0; kPoint < d_kpointWeights.size(); kPoint++)
              {
                for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
                  {
                    dftfe::uInt atomId = atomIdsInProcessor[iAtom];
                    dftfe::uInt Zno    = atomicNumber[atomId];
                    dftfe::uInt numberSphericalFunctions =
                      d_atomicProjectorFnsContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
                    for (dftfe::uInt alpha_i = 0;
                         alpha_i < numberSphericalFunctions;
                         alpha_i++)
                      {
                        for (dftfe::uInt alpha_j = 0;
                             alpha_j < numberSphericalFunctions;
                             alpha_j++)
                          {
                            dftfe::uInt index =
                              alpha_i * numberSphericalFunctions + alpha_j;
                            ValueType V =
                              d_atomicNonLocalPseudoPotentialConstants
                                [CouplingType::inverseOverlapEntries][atomId]
                                [kPoint * numberSphericalFunctions *
                                   numberSphericalFunctions +
                                 index];
                            Entries.push_back(V);
                          } // alpha_j
                      }     // alpha_i
                  }         // iAtom
              }             // kPoint
          }
        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          {
            if (!d_inverseCouplingMatrixEntriesUpdated)
              {
                d_couplingMatrixEntries[couplingtype].resize(Entries.size());
                d_couplingMatrixEntries[couplingtype].copyFrom(Entries);
                d_inverseCouplingMatrixEntriesUpdated = true;
              }

            return d_couplingMatrixEntries[couplingtype];
          }
#if defined(DFTFE_WITH_DEVICE)
        else
          {
            if (!d_inverseCouplingMatrixEntriesUpdated)
              {
                std::vector<ValueType> EntriesPadded;
                d_nonLocalOperator->paddingCouplingMatrix(
                  Entries, EntriesPadded, CouplingStructure::dense);
                d_couplingMatrixEntries[couplingtype].resize(
                  EntriesPadded.size());
                d_couplingMatrixEntries[couplingtype].copyFrom(EntriesPadded);
                d_inverseCouplingMatrixEntriesUpdated = true;
              }
            return d_couplingMatrixEntries[couplingtype];
          }
#endif
      }



    // return d_couplingMatrixEntries[couplingtype];
  } // namespace dftfe


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<
    typename dftfe::dataTypes::singlePrecType<ValueType>::type,
    memorySpace> &
  pawClass<ValueType, memorySpace>::getCouplingMatrixSinglePrec(
    CouplingType couplingtype)
  {
    if (couplingtype == CouplingType::HamiltonianEntries)
      {
        if (!d_HamiltonianCouplingMatrixEntriesUpdatedSinglePrec)
          {
            dftfe::utils::MemoryStorage<ValueType, memorySpace>
              couplingEntries =
                getCouplingMatrix(CouplingType::HamiltonianEntries);
            dftfe::utils::MemoryStorage<
              typename dftfe::dataTypes::singlePrecType<ValueType>::type,
              memorySpace>
              couplingEntriesSinglePrec;
            couplingEntriesSinglePrec.resize(couplingEntries.size());
            if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
              d_BLASWrapperHostPtr->copyValueType1ArrToValueType2Arr(
                couplingEntriesSinglePrec.size(),
                couplingEntries.data(),
                couplingEntriesSinglePrec.data());
#if defined(DFTFE_WITH_DEVICE)
            else
              d_BLASWrapperDevicePtr->copyValueType1ArrToValueType2Arr(
                couplingEntriesSinglePrec.size(),
                couplingEntries.data(),
                couplingEntriesSinglePrec.data());
#endif

            d_couplingMatrixEntriesSinglePrec
              [CouplingType::HamiltonianEntries] = couplingEntriesSinglePrec;
            d_HamiltonianCouplingMatrixEntriesUpdatedSinglePrec = true;
          }
      }
    else if (couplingtype == CouplingType::OverlapEntries)
      {
        if (!d_overlapCouplingMatrixEntriesUpdatedSinglePrec)
          {
            dftfe::utils::MemoryStorage<ValueType, memorySpace>
              couplingEntries = getCouplingMatrix(CouplingType::OverlapEntries);
            dftfe::utils::MemoryStorage<
              typename dftfe::dataTypes::singlePrecType<ValueType>::type,
              memorySpace>
              couplingEntriesSinglePrec;
            couplingEntriesSinglePrec.resize(couplingEntries.size());
            if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
              d_BLASWrapperHostPtr->copyValueType1ArrToValueType2Arr(
                couplingEntriesSinglePrec.size(),
                couplingEntries.data(),
                couplingEntriesSinglePrec.data());
#if defined(DFTFE_WITH_DEVICE)
            else
              d_BLASWrapperDevicePtr->copyValueType1ArrToValueType2Arr(
                couplingEntriesSinglePrec.size(),
                couplingEntries.data(),
                couplingEntriesSinglePrec.data());
#endif

            d_couplingMatrixEntriesSinglePrec[CouplingType::OverlapEntries] =
              couplingEntriesSinglePrec;
            d_overlapCouplingMatrixEntriesUpdatedSinglePrec = true;
          }
      }
    else if (couplingtype == CouplingType::inverseOverlapEntries)
      {
        if (!d_inverseCouplingMatrixEntriesUpdatedSinglePrec)
          {
            dftfe::utils::MemoryStorage<ValueType, memorySpace>
              couplingEntries =
                getCouplingMatrix(CouplingType::inverseOverlapEntries);
            dftfe::utils::MemoryStorage<
              typename dftfe::dataTypes::singlePrecType<ValueType>::type,
              memorySpace>
              couplingEntriesSinglePrec;
            couplingEntriesSinglePrec.resize(couplingEntries.size());
            if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
              d_BLASWrapperHostPtr->copyValueType1ArrToValueType2Arr(
                couplingEntriesSinglePrec.size(),
                couplingEntries.data(),
                couplingEntriesSinglePrec.data());
#if defined(DFTFE_WITH_DEVICE)
            else
              d_BLASWrapperDevicePtr->copyValueType1ArrToValueType2Arr(
                couplingEntriesSinglePrec.size(),
                couplingEntries.data(),
                couplingEntriesSinglePrec.data());
#endif

            d_couplingMatrixEntriesSinglePrec
              [CouplingType::inverseOverlapEntries] = couplingEntriesSinglePrec;
            d_inverseCouplingMatrixEntriesUpdatedSinglePrec = true;
          }
      }



    return d_couplingMatrixEntriesSinglePrec[couplingtype];
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeDij(
    const bool        isDijOut,
    const dftfe::uInt startVectorIndex,
    const dftfe::uInt vectorBlockSize,
    const dftfe::uInt spinIndex,
    const dftfe::uInt kpointIndex)
  {
    const std::vector<dftfe::uInt> atomIdsInProcessor =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<dftfe::uInt> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    char transB = 'N';
#ifdef USE_COMPLEX
    char transA = 'C';
#else
    char transA = 'T';
#endif
    const ValueType beta  = 0.0;
    const ValueType alpha = 1.0;
    for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
      {
        const dftfe::uInt atomId = atomIdsInProcessor[iAtom];
        const dftfe::uInt Znum   = atomicNumber[atomId];
        const dftfe::uInt numberSphericalFunctions =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        if (startVectorIndex == 0 && spinIndex == 0)
          {
            D_ij[0][isDijOut ? TypeOfField::Out : TypeOfField::In][atomId] =
              std::vector<double>(numberSphericalFunctions *
                                    numberSphericalFunctions,
                                  0.0);
            if (d_dftParamsPtr->spinPolarized == 1)
              D_ij[1][isDijOut ? TypeOfField::Out : TypeOfField::In][atomId] =
                std::vector<double>(numberSphericalFunctions *
                                      numberSphericalFunctions,
                                    0.0);
          }
        std::vector<ValueType> tempDij(numberSphericalFunctions *
                                         numberSphericalFunctions,
                                       0.0);

        // if (d_verbosity >= 5)
        //   {
        //     std::cout << "U Matrix Entries for spinIndex: " <<spinIndex<<
        //     std::endl; for (int i = 0; i < numberSphericalFunctions *
        //     vectorBlockSize; i++)
        //       pcout << *(d_nonLocalOperator
        //                        ->getCconjtansXLocalDataStructure(iAtom) +
        //                      i)
        //                 << std::endl;
        //}
        const ValueType *Umatrix =
          d_nonLocalOperator->getCconjtansXLocalDataStructure(iAtom);
        d_BLASWrapperHostPtr->xgemm(transA,
                                    transB,
                                    numberSphericalFunctions,
                                    numberSphericalFunctions,
                                    vectorBlockSize,
                                    &alpha,
                                    Umatrix,
                                    vectorBlockSize,
                                    Umatrix,
                                    vectorBlockSize,
                                    &beta,
                                    &tempDij[0],
                                    numberSphericalFunctions);

        std::transform(
          D_ij[0][isDijOut ? TypeOfField::Out : TypeOfField::In][atomId].data(),
          D_ij[0][isDijOut ? TypeOfField::Out : TypeOfField::In][atomId]
              .data() +
            numberSphericalFunctions * numberSphericalFunctions,
          tempDij.data(),
          D_ij[0][isDijOut ? TypeOfField::Out : TypeOfField::In][atomId].data(),
          [](auto &p, auto &q) { return p + dftfe::utils::realPart(q); });
        if (d_dftParamsPtr->spinPolarized == 1)
          {
            if (spinIndex == 0)
              {
                std::transform(
                  D_ij[1][isDijOut ? TypeOfField::Out : TypeOfField::In][atomId]
                    .data(),
                  D_ij[1][isDijOut ? TypeOfField::Out : TypeOfField::In][atomId]
                      .data() +
                    numberSphericalFunctions * numberSphericalFunctions,
                  tempDij.data(),
                  D_ij[1][isDijOut ? TypeOfField::Out : TypeOfField::In][atomId]
                    .data(),
                  [](auto &p, auto &q) {
                    return p + dftfe::utils::realPart(q);
                  });
              }
            else if (spinIndex == 1)
              {
                std::transform(
                  D_ij[1][isDijOut ? TypeOfField::Out : TypeOfField::In][atomId]
                    .data(),
                  D_ij[1][isDijOut ? TypeOfField::Out : TypeOfField::In][atomId]
                      .data() +
                    numberSphericalFunctions * numberSphericalFunctions,
                  tempDij.data(),
                  D_ij[1][isDijOut ? TypeOfField::Out : TypeOfField::In][atomId]
                    .data(),
                  [](auto &p, auto &q) {
                    return p - dftfe::utils::realPart(q);
                  });
              }
          }
      }
  }
  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
