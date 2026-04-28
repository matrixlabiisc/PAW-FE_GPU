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
#include <unistd.h>
#if defined(DFTFE_WITH_DEVICE)
#  include <pawClassKernelsDevice.h>
#endif
// unsigned long long
// getTotalSystemMemory()
// {
//   long pages     = sysconf(_SC_PHYS_PAGES);
//   long page_size = sysconf(_SC_PAGE_SIZE);
//   return pages * page_size;
// }
// unsigned long long
// getTotalAvaliableMemory()
// {
//   long pages     = sysconf(_SC_AVPHYS_PAGES);
//   long page_size = sysconf(_SC_PAGE_SIZE);
//   return pages * page_size;
// }
namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  pawClass<ValueType, memorySpace>::pawClass(
    const MPI_Comm                           &mpi_comm_parent,
    const std::string                        &scratchFolderName,
    dftParameters                            *dftParamsPtr,
    const std::set<dftfe::uInt>              &atomTypes,
    const bool                                floatingNuclearCharges,
    const dftfe::uInt                         nOMPThreads,
    const std::map<dftfe::uInt, dftfe::uInt> &atomAttributes,
    const bool                                reproducibleOutput,
    const dftfe::Int                          verbosity,
    const bool                                useDevice,
    const bool                                memOptMode)
    : d_mpiCommParent(mpi_comm_parent)
    , d_this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent))
    , d_n_mpi_processes(
        dealii::Utilities::MPI::n_mpi_processes(mpi_comm_parent))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_dftParamsPtr(dftParamsPtr)
  {
    d_dftfeScratchFolderName = scratchFolderName;
    d_atomTypes              = atomTypes;
    d_floatingNuclearCharges = floatingNuclearCharges;
    d_nOMPThreads            = nOMPThreads;
    d_reproducible_output    = reproducibleOutput;
    d_verbosity              = verbosity;
    d_atomTypeAtributes      = atomAttributes;
    d_useDevice              = useDevice;
    d_memoryOptMode          = memOptMode;
    d_integralCoreDensity    = 0.0;
    d_hasSOC                 = false;
    D_ij.clear();
    D_ij.resize((d_dftParamsPtr->spinPolarized == 1) ? 2 : 1);

    d_auxDensityMatrixXCPSPtr =
      std::make_shared<AuxDensityMatrixRadial<memorySpace>>();
    d_auxDensityMatrixXCAEPtr =
      std::make_shared<AuxDensityMatrixRadial<memorySpace>>();
    // dftUtils::printCurrentMemoryUsage(d_mpiCommParent,
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        char DerivativeFileName[256];
        strcpy(DerivativeFileName,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) + "/" +
                "derivatives.dat")
                 .c_str());
        std::vector<std::vector<double>> derivativeData(0);
        dftUtils::readFile(2, derivativeData, DerivativeFileName);
        dftfe::uInt         RmaxIndex = 0;
        dftfe::uInt         maxIndex  = derivativeData.size();
        std::vector<double> RadialMesh(maxIndex, 0.0);
        std::vector<double> radialDerivative(maxIndex, 0.0);
        char                pseudoAtomDataFile[256];
        strcpy(pseudoAtomDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/PseudoAtomDat")
                 .c_str());
        std::ifstream readPseudoDataFileNames(pseudoAtomDataFile);
        char          isCore;
        double        coreKE;
        double        RmaxAug;
        if (readPseudoDataFileNames.is_open())
          {
            readPseudoDataFileNames >> isCore;
            readPseudoDataFileNames >> RmaxAug;
            if (isCore == 'T')
              readPseudoDataFileNames >> coreKE;
            else
              coreKE = 0.0;
            d_RmaxAug[*it] = RmaxAug;
            d_coreKE[*it]  = coreKE;
          }
        double deltaR = 1000;
        for (dftfe::Int iRow = 0; iRow < maxIndex; iRow++)
          {
            RadialMesh[iRow]       = derivativeData[iRow][0];
            radialDerivative[iRow] = derivativeData[iRow][1];
            if (std::fabs(RadialMesh[iRow] - RmaxAug) < deltaR)
              {
                RmaxIndex = iRow;
                deltaR    = std::fabs(RadialMesh[iRow] - RmaxAug);
              }
          }
        d_radialMesh[*it]         = RadialMesh;
        d_radialJacobianData[*it] = radialDerivative;
        pcout << "PAW Initialization: Rmax Index is: " << RmaxIndex
              << std::endl;
        pcout
          << "PAW Initialization: Difference in RmaxAug with xml augmentation Radius: "
          << deltaR << std::endl;
        d_RmaxAugIndex[*it] = RmaxIndex;
        pcout
          << "PAW Initializaion: Warning! make sure the above value is not large!!"
          << std::endl;
        double Rold = d_RmaxAug[*it];
        if (deltaR > 1E-8)
          d_RmaxAug[*it] = RadialMesh[RmaxIndex];
        pcout << "PAW Initialization: Warning!! PAW RmaxAug is reset to: "
              << d_RmaxAug[*it] << " from: " << Rold << std::endl;
      }
    // dftUtils::printCurrentMemoryUsage(d_mpiCommParent, "PAWClass
    // Constructor");
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType,
           memorySpace>::createAtomCenteredSphericalFunctionsForShapeFunctions()
  {
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        char        pseudoAtomDataFile[256];
        dftfe::uInt cumulativeSplineId = 0;
        strcpy(pseudoAtomDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/PseudoAtomDat")
                 .c_str());

        dftfe::uInt   Znum = *it;
        std::ifstream readPseudoDataFileNames(pseudoAtomDataFile);
        for (dftfe::Int i = 0; i <= 2; i++)
          {
            std::string temp;
            std::getline(readPseudoDataFileNames, temp);
          }
        dftfe::uInt shapeFnType;
        readPseudoDataFileNames >> shapeFnType;
        double rc;
        readPseudoDataFileNames >> rc;
        pcout << "Shape Fn Type and rc: " << shapeFnType << " " << rc
              << std::endl;
        dftfe::uInt lmaxAug = d_dftParamsPtr->noShapeFnsInPAW;
        double      rmaxAug = d_RmaxAug[*it];
        char        shapeFnFile[256];
        strcpy(shapeFnFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/shape_functions.dat")
                 .c_str());
        std::vector<std::vector<double>> shaveFnValues(0);
        dftUtils::readFile(lmaxAug + 1, shaveFnValues, shapeFnFile);
        std::vector<double> radialMesh   = d_radialMesh[*it];
        std::vector<double> jacobianData = d_radialJacobianData[*it];
        dftfe::uInt         rmaxAugIndex = d_RmaxAugIndex[*it];
        dftfe::uInt         numValues    = radialMesh.size();
        double              RmaxShapeFn;
        for (dftfe::uInt lQuantumNo = 0; lQuantumNo < lmaxAug; lQuantumNo++)
          {
            double normalizationalizationConstant = 0.0;
            std::function<double(const dftfe::uInt &)> f =
              [&](const dftfe::uInt &i) {
                double Value = jacobianData[i] *
                               shaveFnValues[i][lQuantumNo + 1] *
                               pow(radialMesh[i], lQuantumNo + 2);
                return (Value);
              };
            pcout << "Computing Normalization Constant for ShapeFn:  "
                  << lQuantumNo << " of Znum: " << Znum << " ";
            normalizationalizationConstant =
              simpsonIntegral(0, rmaxAugIndex, f);
            pcout << "Normalization Constant Value: "
                  << normalizationalizationConstant << std::endl;
            if (shapeFnType == 0)
              {
                // Bessel Function
                pcout << "Bessel function: " << lQuantumNo << std::endl;
                d_atomicShapeFnsMap[std::make_pair(Znum, lQuantumNo)] =
                  std::make_shared<AtomCenteredSphericalFunctionBessel>(
                    rc, rmaxAug, lQuantumNo, normalizationalizationConstant);
                RmaxShapeFn = rc;
              }
            else if (shapeFnType == 1)
              {
                // Gauss Function
                if (d_verbosity >= 5)
                  pcout << "Gauss function: " << lQuantumNo << std::endl;
                d_atomicShapeFnsMap[std::make_pair(Znum, lQuantumNo)] =
                  std::make_shared<AtomCenteredSphericalFunctionGaussian>(
                    rc, rmaxAug, lQuantumNo, normalizationalizationConstant);
                RmaxShapeFn = rmaxAug;
              }
            else if (shapeFnType == 2)
              {
                // sinc Function
                if (d_verbosity >= 5)
                  pcout << "sinc function: " << lQuantumNo << std::endl;
                d_atomicShapeFnsMap[std::make_pair(Znum, lQuantumNo)] =
                  std::make_shared<AtomCenteredSphericalFunctionSinc>(
                    rc, rmaxAug, lQuantumNo, normalizationalizationConstant);
                RmaxShapeFn = rc;
              }
            else
              {
                d_atomicShapeFnsMap[std::make_pair(Znum, lQuantumNo)] =
                  std::make_shared<
                    AtomCenteredSphericalFunctionProjectorSpline>(shapeFnFile,
                                                                  lQuantumNo,
                                                                  0,
                                                                  lQuantumNo +
                                                                    1,
                                                                  lmaxAug + 1,
                                                                  1E-12,
                                                                  true);
                RmaxShapeFn = rmaxAug;
              }
          }
        // Determine the RmaxAugIndex for shape functions
        double      deltaR    = 1000;
        dftfe::uInt RmaxIndex = 0;
        for (dftfe::Int iRow = 0; iRow < numValues; iRow++)
          {
            if (std::fabs(radialMesh[iRow] - RmaxShapeFn) < deltaR)
              {
                RmaxIndex = iRow;
                deltaR    = std::fabs(radialMesh[iRow] - RmaxShapeFn);
              }
          }
        d_RmaxAugIndexShapeFn[*it] = RmaxIndex;
        pcout << "PAW Initialization: Rmax Index for ShapeFn is: " << RmaxIndex
              << std::endl;
        std::vector<double> shapeFnGridData(lmaxAug * numValues, 0.0);
        for (dftfe::uInt lQuantumNo = 0; lQuantumNo < lmaxAug; lQuantumNo++)
          {
            for (dftfe::Int iRow = 0; iRow < numValues; iRow++)
              {
                shapeFnGridData[lQuantumNo * numValues + iRow] =
                  d_atomicShapeFnsMap[std::make_pair(Znum, lQuantumNo)]
                    ->getRadialValue(radialMesh[iRow]);
              }
          }
        d_atomicShapeFn[*it] = shapeFnGridData;



      } //*it
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType,
           memorySpace>::createAtomCenteredSphericalFunctionsForDensities()
  {
    d_atomicCoreDensityVector.clear();
    d_atomicCoreDensityVector.resize(d_nOMPThreads);
    d_atomicValenceDensityVector.clear();
    d_atomicValenceDensityVector.resize(d_nOMPThreads);

    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        dftfe::uInt atomicNumber = *it;
        char        valenceDataFile[256];
        strcpy(valenceDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/pseudo_valence_density.dat")
                 .c_str());
        char coreDataFileAE[256];
        strcpy(coreDataFileAE,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/ae_core_density.dat")
                 .c_str());
        char coreDataFilePS[256];
        strcpy(coreDataFilePS,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/pseudo_core_density.dat")
                 .c_str());

        for (dftfe::uInt i = 0; i < d_nOMPThreads; i++)
          {
            d_atomicValenceDensityVector[i][*it] = std::make_shared<
              AtomCenteredSphericalFunctionValenceDensitySpline>(
              valenceDataFile, 1E-10 * sqrt(4 * M_PI), true);
            d_atomicCoreDensityVector[i][*it] =
              std::make_shared<AtomCenteredSphericalFunctionCoreDensitySpline>(
                coreDataFilePS, 1E-12, true);
          }
        if (d_atomicCoreDensityVector[0][atomicNumber]->isDataPresent())
          d_atomTypeCoreFlagMap[atomicNumber] = true;
        else
          d_atomTypeCoreFlagMap[atomicNumber] = false;
        std::vector<double> radialMesh     = d_radialMesh[*it];
        dftfe::uInt         numValues      = radialMesh.size();
        std::vector<double> jacobianValues = d_radialJacobianData[*it];
        std::vector<double> radialAECoreDensity(numValues, 0.00);
        std::vector<double> radialPSCoreDensity(numValues, 0.00);

        if (d_atomicCoreDensityVector[0][atomicNumber]->isDataPresent())
          {
            std::vector<std::vector<double>> AECoreDensityData(0),
              PSCoreDensityData(0);
            dftUtils::readFile(2, AECoreDensityData, coreDataFileAE);
            dftUtils::readFile(2, PSCoreDensityData, coreDataFilePS);

            for (dftfe::uInt iRow = 0; iRow < numValues; iRow++)
              {
                radialAECoreDensity[iRow] = AECoreDensityData[iRow][1];
                radialPSCoreDensity[iRow] = PSCoreDensityData[iRow][1];
              }

            d_radialCoreDerAE[*it] =
              radialDerivativeOfMeshData(radialMesh,
                                         jacobianValues,
                                         radialAECoreDensity);

            d_radialCoreDerPS[*it] =
              radialDerivativeOfMeshData(radialMesh,
                                         jacobianValues,
                                         radialPSCoreDensity);
          }
        else
          {
            d_radialCoreDerAE[*it] = std::vector<double>(numValues, 0.0);
            d_radialCoreDerPS[*it] = std::vector<double>(numValues, 0.0);
          }
        d_atomCoreDensityAE[*it]          = radialAECoreDensity;
        d_atomCoreDensityPS[*it]          = radialPSCoreDensity;
        double charge                     = double(*it) / sqrt(4.0 * M_PI);
        d_DeltaL0coeff[*it]               = -charge;
        d_integralCoreDensityPerAtom[*it] = 0.0;
        if (d_atomTypeCoreFlagMap[atomicNumber])
          {
            std::function<double(const dftfe::uInt &)> f =
              [&](const dftfe::uInt &i) {
                double Value = jacobianValues[i] * radialAECoreDensity[i] *
                               pow(radialMesh[i], 2);

                return (Value);
              };
            double Q1 = simpsonIntegral(0, radialAECoreDensity.size() - 1, f);
            pcout
              << "PAW Initialization: Integral All Electron CoreDensity from Radial integration: "
              << Q1 * sqrt(4 * M_PI) << std::endl;
            std::function<double(const dftfe::uInt &)> g =
              [&](const dftfe::uInt &i) {
                double Value = jacobianValues[i] * radialPSCoreDensity[i] *
                               pow(radialMesh[i], 2);
                return (Value);
              };
            double Q2 = simpsonIntegral(0, radialPSCoreDensity.size() - 1, g);
            pcout
              << "PAW Initialization: Integral Pseudo Smooth CoreDensity from Radial integration: "
              << Q2 * sqrt(4 * M_PI) << std::endl;
            d_integralCoreDensityPerAtom[*it] = Q2 * sqrt(4 * M_PI);
            d_DeltaL0coeff[*it] += (Q1 - Q2);
            d_Ncore[*it]      = Q1 * sqrt(4 * M_PI);
            d_NtildeCore[*it] = Q2 * sqrt(4 * M_PI);
          }
        pcout << "PAW Initialisation: Value of deltaL0 for ZNum: " << *it << " "
              << d_DeltaL0coeff[*it] << std::endl;
      } //*it loop
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialise(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
      basisOperationsHostPtr,
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::DEVICE>>
      basisOperationsDevicePtr,
#endif
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      basisOperationsElectroHostPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      BLASWrapperPtrHost,
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      BLASWrapperPtrDevice,
#endif
    dftfe::uInt                              densityQuadratureId,
    dftfe::uInt                              localContributionQuadratureId,
    dftfe::uInt                              sparsityPatternQuadratureId,
    dftfe::uInt                              nlpspQuadratureId,
    dftfe::uInt                              densityQuadratureIdElectro,
    std::shared_ptr<excManager<memorySpace>> excFunctionalPtr,
    const std::vector<std::vector<double>>  &atomLocations,
    dftfe::uInt                              numEigenValues,
    dftfe::uInt compensationChargeQuadratureIdElectro,
    std::map<dealii::CellId, std::vector<double>>     &bQuadValuesAllAtoms,
    std::map<dealii::CellId, std::vector<dftfe::Int>> &bQuadAtomIdsAllAtoms,
    std::map<dealii::CellId, std::vector<dftfe::Int>>
              &bQuadAtomIdsAllAtomsImages,
    const bool singlePrecNonLocalOperator,
    const bool floatingNuclearCharges,
    const bool computeForce,
    const bool computeStress)
  {
    MPI_Barrier(d_mpiCommParent);
    d_BasisOperatorHostPtr        = basisOperationsHostPtr;
    d_BLASWrapperHostPtr          = BLASWrapperPtrHost;
    d_BasisOperatorElectroHostPtr = basisOperationsElectroHostPtr;
#if defined(DFTFE_WITH_DEVICE)
    d_BLASWrapperDevicePtr   = BLASWrapperPtrDevice;
    d_BasisOperatorDevicePtr = basisOperationsDevicePtr;
#endif

    d_bQuadValuesAllAtoms        = &bQuadValuesAllAtoms;
    d_bQuadAtomIdsAllAtoms       = &bQuadAtomIdsAllAtoms;
    d_bQuadAtomIdsAllAtomsImages = &bQuadAtomIdsAllAtomsImages;
    std::vector<dftfe::uInt> atomicNumbers;
    for (dftfe::Int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        atomicNumbers.push_back(atomLocations[iAtom][0]);
      }
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        dftfe::uInt Znum = *it;
        for (dftfe::Int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
          {
            if (Znum == atomLocations[iAtom][0])
              {
                d_valenceElectrons[Znum] = atomLocations[iAtom][1];
                break;
              }
          }
      }
    dftUtils::printCurrentMemoryUsage(d_mpiCommParent,
                                      "PAWClass Initialise Begin");

    d_densityQuadratureId           = densityQuadratureId;
    d_localContributionQuadratureId = localContributionQuadratureId;
    d_densityQuadratureIdElectro    = densityQuadratureIdElectro;
    d_sparsityPatternQuadratureId   = sparsityPatternQuadratureId;
    d_nlpspQuadratureId             = nlpspQuadratureId;
    d_excManagerPtr                 = excFunctionalPtr;
    d_numEigenValues                = numEigenValues;
    d_compensationChargeQuadratureIdElectro =
      compensationChargeQuadratureIdElectro;
    d_singlePrecNonLocalOperator = singlePrecNonLocalOperator;
    // Read Derivative File



    // Reading Core Density Data
    createAtomCenteredSphericalFunctionsForDensities();
    // Reading Projectors/partial and PS partial waves Data
    createAtomCenteredSphericalFunctionsForProjectors();
    // Rading Shapefunctions Data
    createAtomCenteredSphericalFunctionsForShapeFunctions();
    // Reading ZeroPotential Data
    createAtomCenteredSphericalFunctionsForZeroPotential();
    d_atomicProjectorFnsContainer =
      std::make_shared<AtomCenteredSphericalFunctionContainer>();

    d_atomicProjectorFnsContainer->init(atomicNumbers, d_atomicProjectorFnsMap);

    d_atomicShapeFnsContainer =
      std::make_shared<AtomCenteredSphericalFunctionContainer>();

    d_atomicShapeFnsContainer->init(atomicNumbers, d_atomicShapeFnsMap);

    if (!d_useDevice)
      {
        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          d_nonLocalOperator = std::make_shared<
            AtomicCenteredNonLocalOperator<ValueType, memorySpace>>(
            d_BLASWrapperHostPtr,
            d_BasisOperatorHostPtr,
            d_atomicProjectorFnsContainer,
            d_mpiCommParent,
            d_memoryOptMode,
            floatingNuclearCharges,
            false,
            computeForce,
            computeStress);
        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          if (d_singlePrecNonLocalOperator)
            d_nonLocalOperatorSinglePrec =
              std::make_shared<AtomicCenteredNonLocalOperator<
                typename dftfe::dataTypes::singlePrecType<ValueType>::type,
                memorySpace>>(d_BLASWrapperHostPtr,
                              d_BasisOperatorHostPtr,
                              d_atomicProjectorFnsContainer,
                              d_mpiCommParent,
                              d_memoryOptMode);
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          d_nonLocalOperator = std::make_shared<
            AtomicCenteredNonLocalOperator<ValueType, memorySpace>>(
            d_BLASWrapperDevicePtr,
            d_BasisOperatorDevicePtr,
            d_atomicProjectorFnsContainer,
            d_mpiCommParent,
            d_memoryOptMode,
            floatingNuclearCharges,
            false,
            computeForce,
            computeStress);
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          if (d_singlePrecNonLocalOperator)
            d_nonLocalOperatorSinglePrec =
              std::make_shared<AtomicCenteredNonLocalOperator<
                typename dftfe::dataTypes::singlePrecType<ValueType>::type,
                memorySpace>>(d_BLASWrapperDevicePtr,
                              d_BasisOperatorDevicePtr,
                              d_atomicProjectorFnsContainer,
                              d_mpiCommParent,
                              d_memoryOptMode);
      }
#endif

    computeRadialMultipoleData();
    computeMultipoleInverse();

    computeNonlocalPseudoPotentialConstants(CouplingType::OverlapEntries);

    initialiseKineticEnergyCorrection();
    initialiseColoumbicEnergyCorrection();
    initialiseZeroPotential();

    initialiseDataonRadialMesh();

    computeCoreDeltaExchangeCorrelationEnergy();
    MPI_Barrier(d_mpiCommParent);
    // dftUtils::printCurrentMemoryUsage(d_mpiCommParent,
    //                                   "PAWClass Initialise End");
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseNonLocalContribution(
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<std::vector<double>> &periodicCoords,
    const std::vector<double>              &kPointWeights,
    const std::vector<double>              &kPointCoordinates,
    const bool                              updateNonlocalSparsity,
    const dftfe::uInt                       dofHanderId)
  {
    // dftUtils::printCurrentMemoryUsage(d_mpiCommParent,
    //                                   "PAWClass Initialise Non Local Begin");
    std::vector<dftfe::uInt> atomicNumbers;
    std::vector<double>      atomCoords;
    for (dftfe::Int iAtom = 0;
         iAtom < d_atomLocationsInterestPseudopotential.size();
         iAtom++)
      {
        atomicNumbers.push_back(
          d_atomLocationsInterestPseudopotential[iAtom][0]);
        for (dftfe::Int dim = 2; dim < 5; dim++)
          atomCoords.push_back(
            d_atomLocationsInterestPseudopotential[iAtom][dim]);
      }

    d_kpointWeights = kPointWeights;

    createAtomTypesList(d_atomLocationsInterestPseudopotential);

    d_atomicProjectorFnsContainer->initaliseCoordinates(atomCoords,
                                                        periodicCoords,
                                                        imageIds);
    d_atomicShapeFnsContainer->initaliseCoordinates(atomCoords,
                                                    periodicCoords,
                                                    imageIds);
    dftUtils::printCurrentMemoryUsage(
      d_mpiCommParent, "PAWClass Initialise Sparsity Update Begin");
    if (updateNonlocalSparsity)
      {
        computeAugmentationOverlap();

        d_HamiltonianCouplingMatrixEntriesUpdated           = false;
        d_overlapCouplingMatrixEntriesUpdated               = false;
        d_inverseCouplingMatrixEntriesUpdated               = false;
        d_HamiltonianCouplingMatrixEntriesUpdatedSinglePrec = false;
        d_overlapCouplingMatrixEntriesUpdatedSinglePrec     = false;
        d_inverseCouplingMatrixEntriesUpdatedSinglePrec     = false;
        MPI_Barrier(d_mpiCommParent);
        double InitTime = MPI_Wtime();
        d_atomicProjectorFnsContainer->computeSparseStructure(
          d_BasisOperatorHostPtr, d_sparsityPatternQuadratureId, 1.02, 1);
        MPI_Barrier(d_mpiCommParent);
        pcout << "Computing sparse structure for shapeFunctions: " << std::endl;
        d_atomicShapeFnsContainer->computeSparseStructure(
          d_BasisOperatorElectroHostPtr, 3, 1.02, 1);
        d_atomicShapeFnsContainer->computeFEEvaluationMaps(
          d_BasisOperatorElectroHostPtr,
          d_compensationChargeQuadratureIdElectro,
          dofHanderId);
        MPI_Barrier(d_mpiCommParent);
        double TotalTime = MPI_Wtime() - InitTime;
        if (d_verbosity >= 2)
          pcout
            << "pawclass: Time taken for computeSparseStructureNonLocalProjectors: "
            << TotalTime << std::endl;
      }
    dftUtils::printCurrentMemoryUsage(
      d_mpiCommParent, "PAWClass Initialise Sparsity Update End");
    MPI_Barrier(d_mpiCommParent);
    double InitTimeTotal = MPI_Wtime();
    dftUtils::printCurrentMemoryUsage(
      d_mpiCommParent, "PAWClass Initialise C matrix Compute Begin");
    d_nonLocalOperator->intitialisePartitionerKPointsAndComputeCMatrixEntries(
      updateNonlocalSparsity,
      kPointWeights,
      kPointCoordinates,
      d_BasisOperatorHostPtr,
      d_BLASWrapperHostPtr,
      d_nlpspQuadratureId);
    dftUtils::printCurrentMemoryUsage(
      d_mpiCommParent, "PAWClass Initialise C matrix Compute End");
    if (d_singlePrecNonLocalOperator)
      d_nonLocalOperatorSinglePrec
        ->copyPartitionerKPointsAndComputeCMatrixEntries(updateNonlocalSparsity,
                                                         kPointWeights,
                                                         kPointCoordinates,
                                                         d_BasisOperatorHostPtr,
                                                         d_BLASWrapperHostPtr,
                                                         d_nlpspQuadratureId,
                                                         d_nonLocalOperator);
    dftUtils::printCurrentMemoryUsage(
      d_mpiCommParent, "PAWClass Initialise C matrix single Prec  End");
    if (d_dftParamsPtr->loadDeltaSinvData)
      {
        dftfe::Int flag = loadDeltaSinverseEntriesFromFile();
        MPI_Barrier(d_mpiCommParent);
        if (flag == 0)
          {
            pcout << "Some issue in reading Sinv: " << std::endl;
            std::exit(0);
          }
      }
    else
      {
        computeNonlocalPseudoPotentialConstants(
          CouplingType::inverseOverlapEntries);
        if (d_dftParamsPtr->saveDeltaSinvData)
          {
            saveDeltaSinverseEntriesToFile();
          }
      }


    pcout << "-----Compensation Charge---" << std::endl;
    computeCompensationChargeL0();

    computeproductOfCGMultipole();
    computeCompensationChargeCoeffMemoryOpt();



    // checkOverlapAugmentation();
    MPI_Barrier(d_mpiCommParent);
    double TotalTime = MPI_Wtime() - InitTimeTotal;
    if (d_verbosity >= 2)
      pcout << "pawclass: Time taken for non local psp init: " << TotalTime
            << std::endl;
    // dftUtils::printCurrentMemoryUsage(d_mpiCommParent,
    //                                   "PAWClass Initialise Non Local End");
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseNonLocalContribution(
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<std::vector<double>> &periodicCoords,
    const std::vector<double>              &kPointWeights,
    const std::vector<double>              &kPointCoordinates,
    const bool                              updateNonlocalSparsity,
    const std::map<dftfe::uInt, std::vector<dftfe::Int>> &sparsityPattern,
    const std::vector<std::vector<dealii::CellId>>
      &elementIdsInAtomCompactSupport,
    const std::vector<std::vector<dftfe::uInt>>
                                   &elementIndexesInAtomCompactSupport,
    const std::vector<dftfe::uInt> &atomIdsInCurrentProcess,
    dftfe::uInt                     numberElements)
  {}



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeNonlocalPseudoPotentialConstants(
    CouplingType couplingtype,
    dftfe::uInt  s)
  {
    if (couplingtype == CouplingType::OverlapEntries)
      {
        for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
             it != d_atomTypes.end();
             ++it)
          {
            const dftfe::uInt   Znum           = *it;
            std::vector<double> multipoleTable = d_multipole[Znum];
            const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                           std::shared_ptr<AtomCenteredSphericalFunctionBase>>
              sphericalFunction =
                d_atomicProjectorFnsContainer->getSphericalFunctions();
            dftfe::uInt numberOfRadialProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
            dftfe::uInt numTotalProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            std::vector<ValueType> fullMultipoleTable(numTotalProjectors *
                                                        numTotalProjectors,
                                                      0.0);
            dftfe::Int             projectorIndex_i = 0;
            for (dftfe::uInt alpha_i = 0; alpha_i < numberOfRadialProjectors;
                 alpha_i++)
              {
                std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
                  sphericalFunction.find(std::make_pair(Znum, alpha_i))->second;
                dftfe::Int lQuantumNo_i = sphFn_i->getQuantumNumberl();

                for (dftfe::Int mQuantumNo_i = -lQuantumNo_i;
                     mQuantumNo_i <= lQuantumNo_i;
                     mQuantumNo_i++)
                  {
                    dftfe::Int projectorIndex_j = 0;
                    for (dftfe::uInt alpha_j = 0;
                         alpha_j < numberOfRadialProjectors;
                         alpha_j++)
                      {
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>
                          sphFn_j = sphericalFunction
                                      .find(std::make_pair(Znum, alpha_j))
                                      ->second;
                        dftfe::Int lQuantumNo_j = sphFn_j->getQuantumNumberl();
                        for (dftfe::Int mQuantumNo_j = -lQuantumNo_j;
                             mQuantumNo_j <= lQuantumNo_j;
                             mQuantumNo_j++)
                          {
                            fullMultipoleTable[projectorIndex_i *
                                                 numTotalProjectors +
                                               projectorIndex_j] =
                              sqrt(4 * M_PI) *
                              multipoleTable[alpha_i *
                                               numberOfRadialProjectors +
                                             alpha_j] *
                              gaunt(lQuantumNo_i,
                                    lQuantumNo_j,
                                    0,
                                    mQuantumNo_i,
                                    mQuantumNo_j,
                                    0);

                            projectorIndex_j++;
                          } // mQuantumeNo_j

                      } // alpha_j


                    projectorIndex_i++;
                  } // mQuantumNo_i
              }     // alpha_i

            d_atomicNonLocalPseudoPotentialConstants
              [CouplingType::OverlapEntries][Znum] = fullMultipoleTable;
            if (d_verbosity >= 5)
              {
                pcout << "NonLocal Overlap Matrrix for Znum: " << Znum
                      << std::endl;
                for (dftfe::Int i = 0; i < numTotalProjectors; i++)
                  {
                    for (dftfe::Int j = 0; j < numTotalProjectors; j++)
                      pcout << d_atomicNonLocalPseudoPotentialConstants
                                 [CouplingType::OverlapEntries][Znum]
                                 [i * numTotalProjectors + j]
                            << " ";
                    pcout << std::endl;
                  }
                pcout << "----------------------------" << std::endl;
              }

          } //*it
        d_overlapCouplingMatrixEntriesUpdated = false;
      }
    else if (couplingtype == CouplingType::inverseOverlapEntries)
      {
        if (d_dftParamsPtr->ApproxDelta)
          {
            std::map<dftfe::uInt, std::vector<double>> PijMatrix;
            std::vector<dftfe::uInt>                   atomicNumber =
              d_atomicProjectorFnsContainer->getAtomicNumbers();

            const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                           std::shared_ptr<AtomCenteredSphericalFunctionBase>>
              sphericalFunction =
                d_atomicProjectorFnsContainer->getSphericalFunctions();
            const dftfe::uInt numKpoints = d_kpointWeights.size();
            for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
                 it != d_atomTypes.end();
                 ++it)
              {
                dftfe::uInt       atomicNumber = *it;
                const dftfe::uInt numberOfRadialProjectors =
                  d_atomicProjectorFnsContainer
                    ->getTotalNumberOfRadialSphericalFunctionsPerAtom(
                      atomicNumber);
                const dftfe::uInt numberOfProjectors =
                  d_atomicProjectorFnsContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber);
                std::vector<double> radialPijTable(numberOfRadialProjectors *
                                                     numberOfRadialProjectors,
                                                   0.0);
                std::vector<double> proj         = d_radialProjVal[*it];
                std::vector<double> radialMesh   = d_radialMesh[*it];
                std::vector<double> jacobianData = d_radialJacobianData[*it];
                dftfe::uInt         meshSize     = radialMesh.size();
                dftfe::uInt         rmaxAugIndex = d_RmaxAugIndex[*it];
                for (dftfe::uInt alpha_i = 0;
                     alpha_i < numberOfRadialProjectors;
                     alpha_i++)
                  {
                    for (dftfe::uInt alpha_j = 0;
                         alpha_j < numberOfRadialProjectors;
                         alpha_j++)
                      {
                        double Value = 0.0;
                        Value        = multipoleIntegrationGrid(proj.data() +
                                                           (alpha_i)*meshSize,
                                                         proj.data() +
                                                           (alpha_j)*meshSize,
                                                         radialMesh,
                                                         jacobianData,
                                                         0,
                                                         0,
                                                         rmaxAugIndex);

                        radialPijTable[alpha_i * numberOfRadialProjectors +
                                       alpha_j] = Value;

                      } // alpha_j
                  }     // alpha_i
                std::vector<double> PijTable(numberOfProjectors *
                                               numberOfProjectors,
                                             0.0);
                int                 projIndexI = 0;
                for (int iProj = 0; iProj < numberOfRadialProjectors; iProj++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
                      sphericalFunction
                        .find(std::make_pair(atomicNumber, iProj))
                        ->second;
                    int lQuantumNo_i = sphFn_i->getQuantumNumberl();
                    for (int mQuantumNumber_i = -lQuantumNo_i;
                         mQuantumNumber_i <= lQuantumNo_i;
                         mQuantumNumber_i++)
                      {
                        int projIndexJ = 0;
                        for (int jProj = 0; jProj < numberOfRadialProjectors;
                             jProj++)
                          {
                            std::shared_ptr<AtomCenteredSphericalFunctionBase>
                              sphFn_j =
                                sphericalFunction
                                  .find(std::make_pair(atomicNumber, jProj))
                                  ->second;
                            int lQuantumNo_j = sphFn_j->getQuantumNumberl();
                            for (int mQuantumNumber_j = -lQuantumNo_j;
                                 mQuantumNumber_j <= lQuantumNo_j;
                                 mQuantumNumber_j++)
                              {
                                PijTable[projIndexI * numberOfProjectors +
                                         projIndexJ] =
                                  gaunt(lQuantumNo_i,
                                        lQuantumNo_j,
                                        0,
                                        mQuantumNumber_i,
                                        mQuantumNumber_j,
                                        0) *
                                  radialPijTable[iProj *
                                                   numberOfRadialProjectors +
                                                 jProj] *
                                  sqrt(4 * M_PI);
                                projIndexJ++;
                              } // mQuantumNumber_j

                          } // jProj
                        projIndexI++;
                      } // mQuantumNumber_i



                  } // iProj
                PijMatrix[atomicNumber] = PijTable;
              }



            for (dftfe::Int iAtom = 0; iAtom < atomicNumber.size(); iAtom++)
              {
                dftfe::uInt Znum = atomicNumber[iAtom];
                dftfe::uInt numberOfProjectors =
                  d_atomicProjectorFnsContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                std::vector<ValueType> deltaMatrixFull(
                  numKpoints * numberOfProjectors * numberOfProjectors, 0.0);
                for (dftfe::Int kPoint = 0; kPoint < numKpoints; kPoint++)
                  {
                    std::vector<ValueType> deltaMatrix2(numberOfProjectors *
                                                          numberOfProjectors,
                                                        0.0);
                    std::vector<double>    multipoleInverse =
                      d_multipoleInverse[Znum];
                    for (dftfe::uInt iProj = 0; iProj < numberOfProjectors;
                         iProj++)
                      {
                        for (dftfe::uInt jProj = 0; jProj < numberOfProjectors;
                             jProj++)
                          {
                            deltaMatrix2[iProj * numberOfProjectors + jProj] =
                              multipoleInverse[iProj * numberOfProjectors +
                                               jProj] +
                              PijMatrix[Znum]
                                       [iProj * numberOfProjectors + jProj];
                          }
                      }

                    dftfe::linearAlgebraOperations::inverse(&deltaMatrix2[0],
                                                            numberOfProjectors);
                    // Copy into each kPoint location
                    d_BLASWrapperHostPtr->xcopy(
                      numberOfProjectors * numberOfProjectors,
                      &deltaMatrix2[0],
                      1,
                      &deltaMatrixFull[kPoint * numberOfProjectors *
                                       numberOfProjectors],
                      1);
                  }
                d_atomicNonLocalPseudoPotentialConstants
                  [CouplingType::inverseOverlapEntries][iAtom] =
                    deltaMatrixFull;
              }
          }

        else
          {
            pcout
              << "PAWClass: Delta Sinv coupling matrix computed using the global P matrix "
              << std::endl;

            MPI_Barrier(d_mpiCommParent);
            double timerStart = MPI_Wtime();

            const dftfe::uInt numberNodesPerElement =
              d_BasisOperatorHostPtr->nDofsPerCell();
            const ValueType                 alpha1 = 1.0;
            const std::vector<dftfe::uInt> &atomicNumber =
              d_atomicShapeFnsContainer->getAtomicNumbers();
            dftfe::uInt              totalEntries = 0;
            std::vector<dftfe::uInt> startIndexAllAtoms(atomicNumber.size(), 0);
            for (dftfe::Int iAtom = 0; iAtom < atomicNumber.size(); iAtom++)
              {
                startIndexAllAtoms[iAtom] = totalEntries;
                dftfe::uInt Znum          = atomicNumber[iAtom];
                dftfe::uInt numProj =
                  d_atomicProjectorFnsContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                totalEntries += numProj * numProj;
              }
            MPI_Barrier(d_mpiCommParent);
            const dftfe::uInt      numKpoints = d_kpointWeights.size();
            std::vector<ValueType> PijMatrix(totalEntries * numKpoints, 0.0);
            const dftfe::uInt      natoms = atomicNumber.size();
            const dftfe::uInt      ndofs = d_BasisOperatorHostPtr->nOwnedDofs();
            std::vector<dftfe::uInt> relAtomdIdsInCurrentProcs =
              relevantAtomdIdsInCurrentProcs();
            dftfe::uInt              totalProjectorsInProcessor = 0;
            std::vector<dftfe::uInt> startIndexProcessorVec(
              relAtomdIdsInCurrentProcs.size(), 0);
            dftfe::uInt startIndex = 0;
            for (dftfe::Int iAtom = 0; iAtom < relAtomdIdsInCurrentProcs.size();
                 iAtom++)
              {
                startIndexProcessorVec[iAtom] = startIndex;
                dftfe::uInt atomId = relAtomdIdsInCurrentProcs[iAtom];
                dftfe::uInt Znum   = atomicNumber[atomId];
                startIndex +=
                  d_atomicProjectorFnsContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
              }
            totalProjectorsInProcessor = startIndex;
            std::vector<ValueType> processorLocalPTransPMatrix(
              totalProjectorsInProcessor * totalProjectorsInProcessor, 0.0);
            std::vector<dftfe::uInt> numProjList;
            for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
                 it != d_atomTypes.end();
                 ++it)
              {
                numProjList.push_back(
                  d_atomicProjectorFnsContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(*it));
              }

            std::map<dftfe::uInt,
                     dftfe::linearAlgebra::
                       MultiVector<ValueType, dftfe::utils::MemorySpace::HOST>>
              Pmatrix;
            const dftfe::utils::MemoryStorage<double,
                                              dftfe::utils::MemorySpace::HOST>
              DminusHalf =
                d_BasisOperatorHostPtr->inverseSqrtMassVectorBasisData();
            for (dftfe::Int i = 0; i < numProjList.size(); i++)
              {
                if (Pmatrix.find(numProjList[i]) == Pmatrix.end())
                  {
                    Pmatrix[numProjList[i]] = dftfe::linearAlgebra::
                      MultiVector<ValueType, dftfe::utils::MemorySpace::HOST>();
                    Pmatrix[numProjList[i]].reinit(
                      d_BasisOperatorHostPtr->mpiPatternP2P, numProjList[i]);
                  }
              }

            for (dftfe::Int kPoint = 0; kPoint < d_kpointWeights.size();
                 kPoint++)
              {
                std::vector<ValueType> processorLocalPmatrix(
                  ndofs * totalProjectorsInProcessor, 0.0);
                dftfe::uInt projStartIndex = 0;
                for (dftfe::uInt atomId = 0; atomId < atomicNumber.size();
                     atomId++)
                  {
                    dftfe::uInt Znum = atomicNumber[atomId];
                    dftfe::uInt numProj =
                      d_atomicProjectorFnsContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    Pmatrix[numProj].setValue(0);
                    if (d_atomicProjectorFnsContainer
                          ->atomIdPresentInCurrentProcessor(atomId))
                      {
                        std::vector<dftfe::uInt>
                          elementIndexesInAtomCompactSupport =
                            d_atomicProjectorFnsContainer
                              ->d_elementIndexesInAtomCompactSupport[atomId];
                        dftfe::Int numberElementsInAtomCompactSupport =
                          elementIndexesInAtomCompactSupport.size();

                        for (dftfe::Int iElem = 0;
                             iElem < numberElementsInAtomCompactSupport;
                             iElem++)
                          {
                            dftfe::uInt elementIndex =
                              elementIndexesInAtomCompactSupport[iElem];
                            // convert this to a ValueType* for better
                            // access. IMPORTANT...
                            std::vector<ValueType> CMatrixEntries =
                              d_nonLocalOperator->getCmatrixEntries(
                                kPoint, atomId, elementIndex);
                            // pcout << "CMatrix: " << iElem << " " <<
                            // elementIndex
                            //       << std::endl;
                            for (dftfe::uInt iDof = 0;
                                 iDof < numberNodesPerElement;
                                 iDof++)
                              {
                                dftfe::uInt dofIndex =
                                  d_BasisOperatorHostPtr
                                    ->d_cellDofIndexToProcessDofIndexMap
                                      [elementIndex * numberNodesPerElement +
                                       iDof];
                                d_BLASWrapperHostPtr->xaxpy(
                                  numProj,
                                  &alpha1,
                                  &CMatrixEntries[iDof * numProj],
                                  1,
                                  Pmatrix[numProj].data() +
                                    (dofIndex * numProj),
                                  1);
                              } // iDof


                          } // iElem



                      } // if atomId present
                    d_BasisOperatorHostPtr
                      ->d_constraintInfo[d_BasisOperatorHostPtr->d_dofHandlerID]
                      .distribute_slave_to_master(Pmatrix[numProj]);
                    Pmatrix[numProj].accumulateAddLocallyOwned();
                    Pmatrix[numProj].zeroOutGhosts();
                    if (std::find(relAtomdIdsInCurrentProcs.begin(),
                                  relAtomdIdsInCurrentProcs.end(),
                                  atomId) != relAtomdIdsInCurrentProcs.end())
                      {
#pragma omp parallel for num_threads(d_nOMPThreads)
                        for (dftfe::Int iDof = 0;
                             iDof < Pmatrix[numProj].locallyOwnedSize();
                             iDof++)
                          {
                            const ValueType scalingCoeff =
                              *(DminusHalf.data() + iDof);
                            d_BLASWrapperHostPtr->xaxpy(
                              numProj,
                              &scalingCoeff,
                              Pmatrix[numProj].data() + iDof * numProj,
                              1,
                              &processorLocalPmatrix
                                [iDof * totalProjectorsInProcessor +
                                 projStartIndex],
                              1);
                          } // iDof
                        projStartIndex += numProj;
                      } // if

                  } // atomId
                char      transA = 'N';
                ValueType alpha  = 1.0;
                ValueType beta   = 0.0;
#ifdef USE_COMPLEX
                char transB = 'C';
#else
                char transB = 'T';
#endif
                if (totalProjectorsInProcessor > 0)
                  d_BLASWrapperHostPtr->xgemm(transA,
                                              transB,
                                              totalProjectorsInProcessor,
                                              totalProjectorsInProcessor,
                                              ndofs,
                                              &alpha,
                                              &processorLocalPmatrix[0],
                                              totalProjectorsInProcessor,
                                              &processorLocalPmatrix[0],
                                              totalProjectorsInProcessor,
                                              &beta,
                                              &processorLocalPTransPMatrix[0],
                                              totalProjectorsInProcessor);
#pragma omp parallel for num_threads(d_nOMPThreads)
                for (dftfe::Int iAtom = 0;
                     iAtom < relAtomdIdsInCurrentProcs.size();
                     iAtom++)
                  {
                    dftfe::uInt atomId = relAtomdIdsInCurrentProcs[iAtom];
                    dftfe::uInt Znum   = atomicNumber[atomId];
                    dftfe::uInt numProj_i =
                      d_atomicProjectorFnsContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    dftfe::uInt startIndexGlobal_i =
                      d_totalProjectorStartIndex[atomId];
                    dftfe::uInt startIndexProcessor_i =
                      startIndexProcessorVec[iAtom];
                    for (dftfe::Int iProj = 0; iProj < numProj_i; iProj++)
                      {
                        for (dftfe::Int jProj = 0; jProj < numProj_i; jProj++)
                          {
                            PijMatrix[kPoint * totalEntries +
                                      (startIndexAllAtoms[atomId] +
                                       iProj * numProj_i + jProj)] =
                              processorLocalPTransPMatrix
                                [(startIndexProcessor_i + iProj) *
                                   totalProjectorsInProcessor +
                                 (startIndexProcessor_i + jProj)];
                          }
                      }


                  } // iAtom
              }     // kPoint

            MPI_Allreduce(MPI_IN_PLACE,
                          &PijMatrix[0],
                          totalEntries * numKpoints,
                          dataTypes::mpi_type_id(&PijMatrix[0]),
                          MPI_SUM,
                          d_mpiCommParent);
            // #pragma omp parallel for num_threads(d_nOMPThreads)
            for (dftfe::Int iAtom = 0; iAtom < atomicNumber.size(); iAtom++)
              {
                dftfe::uInt Znum = atomicNumber[iAtom];
                dftfe::uInt numberOfProjectors =
                  d_atomicProjectorFnsContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                std::vector<ValueType> deltaMatrixFull(
                  numKpoints * numberOfProjectors * numberOfProjectors, 0.0);
                for (dftfe::Int kPoint = 0; kPoint < numKpoints; kPoint++)
                  {
                    std::vector<ValueType> deltaMatrix2(numberOfProjectors *
                                                          numberOfProjectors,
                                                        0.0);
                    std::vector<double>    multipoleInverse =
                      d_multipoleInverse[Znum];
                    dftfe::uInt start = startIndexAllAtoms[iAtom];
                    for (dftfe::uInt iProj = 0; iProj < numberOfProjectors;
                         iProj++)
                      {
                        for (dftfe::uInt jProj = 0; jProj < numberOfProjectors;
                             jProj++)
                          {
                            deltaMatrix2[iProj * numberOfProjectors + jProj] =
                              multipoleInverse[iProj * numberOfProjectors +
                                               jProj] +
                              PijMatrix[kPoint * totalEntries + start +
                                        iProj * numberOfProjectors + jProj];
                          }
                      }

                    dftfe::linearAlgebraOperations::inverse(&deltaMatrix2[0],
                                                            numberOfProjectors);
                    // Copy into each kPoint location
                    d_BLASWrapperHostPtr->xcopy(
                      numberOfProjectors * numberOfProjectors,
                      &deltaMatrix2[0],
                      1,
                      &deltaMatrixFull[kPoint * numberOfProjectors *
                                       numberOfProjectors],
                      1);
                  }
                d_atomicNonLocalPseudoPotentialConstants
                  [CouplingType::inverseOverlapEntries][iAtom] =
                    deltaMatrixFull;
              }
            MPI_Barrier(d_mpiCommParent);
            double timerEnd = MPI_Wtime();
            pcout
              << "PAWClass: Delta Sinv coupling matrix computed using the global P matrix: "
              << (timerEnd - timerStart) << std::endl;
            MPI_Barrier(d_mpiCommParent);
          }
        // Pmatrix.clear();
        d_inverseCouplingMatrixEntriesUpdated = false;
        const std::vector<dftfe::uInt> &atomicNumber =
          d_atomicShapeFnsContainer->getAtomicNumbers();
        if (d_verbosity >= 5)
          {
            for (dftfe::Int iAtom = 0; iAtom < atomicNumber.size(); iAtom++)
              {
                const dftfe::uInt Znum = atomicNumber[iAtom];
                dftfe::uInt       numberOfProjectors =
                  d_atomicProjectorFnsContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                const std::vector<ValueType> inverseOverlapEntries =
                  d_atomicNonLocalPseudoPotentialConstants
                    [CouplingType::inverseOverlapEntries][iAtom];
                if (d_verbosity >= 5)
                  {
                    pcout << "inverseOverlapEntries for iAtom: " << iAtom
                          << std::endl;
                    for (dftfe::Int i = 0; i < numberOfProjectors; i++)
                      {
                        for (dftfe::Int j = 0; j < numberOfProjectors; j++)
                          pcout
                            << inverseOverlapEntries[i * numberOfProjectors + j]
                            << " ";
                        pcout << std::endl;
                      }
                    pcout << "------------------------------" << std::endl;
                  }
              }
          }
      }
    else if (couplingtype == CouplingType::HamiltonianEntries)
      {
        MPI_Barrier(d_mpiCommParent);
        dftfe::uInt       one     = 1;
        const char        transA  = 'N';
        const char        transB  = 'N';
        const char        transB2 = 'T';
        const double      alpha   = 1.0;
        const double      beta    = 1.0;
        const dftfe::uInt inc     = 1;
        std::map<TypeOfField, std::map<dftfe::uInt, std::vector<double>>>
          &D_ijRho = D_ij[0];
        const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          shapeFunction = d_atomicShapeFnsContainer->getSphericalFunctions();
        const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          projectorFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
             it != d_atomTypes.end();
             ++it)
          {
            dftfe::uInt              atomType  = *it;
            std::vector<dftfe::uInt> atomLists = d_atomTypesList[atomType];
            dftfe::uInt              Znum      = *it;
            const dftfe::uInt        numberOfProjectorFns =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(*it);
            const dftfe::uInt npjsq =
              numberOfProjectorFns * numberOfProjectorFns;
            std::vector<double> KEContribution =
              d_KineticEnergyCorrectionTerm[Znum];
            std::vector<double> C_ijcontribution  = d_deltaCij[Znum];
            std::vector<double> Cijkl             = d_deltaCijkl[Znum];
            std::vector<double> zeroPotentialAtom = d_zeroPotentialij[Znum];
            std::vector<double> multipoleValue    = d_multipole[Znum];
            std::vector<double> nonLocalHamiltonianVector(npjsq *
                                                            atomLists.size(),
                                                          0.0);
            std::vector<double> deltaColoumbicEnergyDijVector(
              npjsq * atomLists.size(), 0.0);
            std::vector<double> NonLocalComputedContributions(
              npjsq * atomLists.size(), 0.0);
            std::vector<double> LocalContribution(npjsq, 0.0);
            dftfe::Int          numRadProj =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
            const dftfe::uInt noOfShapeFns =
              d_atomicShapeFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            const dftfe::Int numRadShapeFns =
              d_atomicShapeFnsContainer
                ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
            std::vector<double> FullMultipoleTable(npjsq * noOfShapeFns, 0.0);
            dftfe::Int          projectorIndex_i = 0;
            for (dftfe::Int alpha_i = 0; alpha_i < numRadProj; alpha_i++)
              {
                std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
                  projectorFunction.find(std::make_pair(Znum, alpha_i))->second;
                dftfe::Int lQuantumNumber_i = sphFn_i->getQuantumNumberl();
                for (dftfe::Int mQuantumNo_i = -lQuantumNumber_i;
                     mQuantumNo_i <= lQuantumNumber_i;
                     mQuantumNo_i++)
                  {
                    dftfe::Int projectorIndex_j = 0;
                    for (dftfe::Int alpha_j = 0; alpha_j < numRadProj;
                         alpha_j++)
                      {
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>
                          sphFn_j = projectorFunction
                                      .find(std::make_pair(Znum, alpha_j))
                                      ->second;
                        dftfe::Int lQuantumNumber_j =
                          sphFn_j->getQuantumNumberl();
                        for (dftfe::Int mQuantumNo_j = -lQuantumNumber_j;
                             mQuantumNo_j <= lQuantumNumber_j;
                             mQuantumNo_j++)
                          {
                            dftfe::Int LshapeFnIndex = 0;
                            for (dftfe::Int L = 0; L < numRadShapeFns; L++)
                              {
                                std::shared_ptr<
                                  AtomCenteredSphericalFunctionBase>
                                  sphFn =
                                    shapeFunction.find(std::make_pair(Znum, L))
                                      ->second;
                                dftfe::Int lQuantumNumber_L =
                                  sphFn->getQuantumNumberl();
                                for (dftfe::Int mQuantumNo_L =
                                       -lQuantumNumber_L;
                                     mQuantumNo_L <= lQuantumNumber_L;
                                     mQuantumNo_L++)
                                  {
                                    FullMultipoleTable[projectorIndex_i *
                                                         numberOfProjectorFns *
                                                         noOfShapeFns +
                                                       projectorIndex_j *
                                                         noOfShapeFns +
                                                       LshapeFnIndex] =
                                      multipoleValue[L * numRadProj *
                                                       numRadProj +
                                                     alpha_i * numRadProj +
                                                     alpha_j] *
                                      gaunt(lQuantumNumber_i,
                                            lQuantumNumber_j,
                                            lQuantumNumber_L,
                                            mQuantumNo_i,
                                            mQuantumNo_j,
                                            mQuantumNo_L);

                                    LshapeFnIndex++;
                                  } // mQuantumNo_L
                              }     // L

                            projectorIndex_j++;
                          } // mQuantumNo_j
                      }     // alpha_j

                    projectorIndex_i++;
                  } // mQuantumNo_i

              } // alpha_i

            for (dftfe::Int iProj = 0; iProj < npjsq; iProj++)
              {
                LocalContribution[iProj] = KEContribution[iProj] +
                                           C_ijcontribution[iProj] -
                                           zeroPotentialAtom[iProj];
              }

            for (dftfe::Int i = 0; i < atomLists.size(); i++)
              {
                for (dftfe::Int iAtomList = 0;
                     iAtomList < d_LocallyOwnedAtomId.size();
                     iAtomList++)
                  {
                    dftfe::uInt atomId = d_LocallyOwnedAtomId[iAtomList];

                    if (atomLists[i] == atomId)
                      {
                        // Cijkl Contribution
                        std::vector<double> CijklContribution(npjsq, 0.0);
                        std::vector<double> Dij =
                          D_ijRho[TypeOfField::In][atomId];
                        d_BLASWrapperHostPtr->xgemm(transA,
                                                    transB,
                                                    inc,
                                                    npjsq,
                                                    npjsq,
                                                    &alpha,
                                                    &Dij[0],
                                                    inc,
                                                    &Cijkl[0],
                                                    npjsq,
                                                    &beta,
                                                    &CijklContribution[0],
                                                    inc);
                        d_BLASWrapperHostPtr->xgemm(transA,
                                                    transB2,
                                                    inc,
                                                    npjsq,
                                                    npjsq,
                                                    &alpha,
                                                    &Dij[0],
                                                    inc,
                                                    &Cijkl[0],
                                                    npjsq,
                                                    &beta,
                                                    &CijklContribution[0],
                                                    inc);
                        std::vector<double> XCcontribution =
                          d_ExchangeCorrelationEnergyCorrectionTerm[atomId];
                        for (dftfe::Int iProj = 0; iProj < npjsq; iProj++)
                          {
                            NonLocalComputedContributions[i * npjsq + iProj] =
                              CijklContribution[iProj] + XCcontribution[iProj];
                          }

                      } // Accessing locally owned atom in the current AtomList
                  }     // AtomList in locallyOwnedLoop
                std::vector<double> nonLocalElectrostatics =
                  d_nonLocalHamiltonianElectrostaticValue[atomLists[i]];
                std::vector<double> NonLocalElectorstaticsContributions(npjsq,
                                                                        0.0);
                if (nonLocalElectrostatics.size() > 0)
                  d_BLASWrapperHostPtr->xgemm(
                    transA,
                    transB,
                    inc,
                    npjsq,
                    noOfShapeFns,
                    &alpha,
                    &nonLocalElectrostatics[0],
                    inc,
                    &FullMultipoleTable[0],
                    noOfShapeFns,
                    &beta,
                    &NonLocalElectorstaticsContributions[0],
                    inc);
                for (dftfe::Int iProj = 0; iProj < npjsq; iProj++)
                  {
                    NonLocalComputedContributions[i * npjsq + iProj] +=
                      NonLocalElectorstaticsContributions[iProj];
                    // deltaColoumbicEnergyDijVector[i * npjsq + iProj] +=
                    //   NonLocalElectorstaticsContributions[iProj];
                  }
              } // i in AtomList Loop
            MPI_Allreduce(MPI_IN_PLACE,
                          &NonLocalComputedContributions[0],
                          npjsq * atomLists.size(),
                          MPI_DOUBLE,
                          MPI_SUM,
                          d_mpiCommParent);

            for (dftfe::Int i = 0; i < atomLists.size(); i++)
              {
                std::vector<ValueType> ValueAtom(npjsq, 0.0);
                std::vector<double>    deltaColoumbicEnergyDij(npjsq, 0.0);
                dftfe::uInt            iAtom = atomLists[i];

                for (dftfe::Int iProj = 0; iProj < npjsq; iProj++)
                  {
                    ValueAtom[iProj] =
                      NonLocalComputedContributions[i * npjsq + iProj] +
                      LocalContribution[iProj];
                  }
                d_atomicNonLocalPseudoPotentialConstants
                  [CouplingType::HamiltonianEntries][atomLists[i]] = ValueAtom;
                // pcout << "NonLocal Hamiltonian Matrix for iAtom: "
                //       << atomLists[i] << " "
                //       << d_atomicNonLocalPseudoPotentialConstants
                //            [CouplingType::HamiltonianEntries][atomLists[i]]
                //              .size()
                //       << std::endl;
                // pcout << "Non Local Ham: " << std::endl;
                // for (dftfe::Int iProj = 0; iProj < numberOfProjectorFns;
                //      iProj++)
                //   {
                //     for (dftfe::Int jProj = 0; jProj < numberOfProjectorFns;
                //          jProj++)
                //       pcout
                //         << d_atomicNonLocalPseudoPotentialConstants
                //              [CouplingType::HamiltonianEntries][atomLists[i]]
                //              [iProj * numberOfProjectorFns + jProj]
                //         << " ";
                //     pcout << std::endl;
                //   }
                // pcout << "----------------------------" << std::endl;
              } // i in atomList

          } // *it
        d_HamiltonianCouplingMatrixEntriesUpdated           = false;
        d_HamiltonianCouplingMatrixEntriesUpdatedSinglePrec = false;
      }
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType,
           memorySpace>::createAtomCenteredSphericalFunctionsForProjectors()
  {
    d_atomicProjectorFnsVector.clear();
    std::vector<std::vector<dftfe::Int>> projectorIdDetails;
    std::vector<std::vector<dftfe::Int>> atomicFunctionIdDetails;
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        char        pseudoAtomDataFile[256];
        dftfe::uInt cumulativeSplineId = 0;
        strcpy(pseudoAtomDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/PseudoAtomDat")
                 .c_str());

        dftfe::uInt   Znum = *it;
        std::ifstream readPseudoDataFileNames(pseudoAtomDataFile);
        dftfe::uInt   numberOfProjectors;
        for (dftfe::Int i = 0; i <= 4; i++)
          {
            std::string temp;
            std::getline(readPseudoDataFileNames, temp);
          }
        readPseudoDataFileNames >> numberOfProjectors;
        std::vector<dftfe::uInt> projectorPerOrbital(4, 0);
        readPseudoDataFileNames >> projectorPerOrbital[0];
        readPseudoDataFileNames >> projectorPerOrbital[1];
        readPseudoDataFileNames >> projectorPerOrbital[2];
        readPseudoDataFileNames >> projectorPerOrbital[3];
        dftfe::uInt totalProjectors =
          projectorPerOrbital[0] + projectorPerOrbital[1] +
          projectorPerOrbital[2] + projectorPerOrbital[3];
        pcout << "Znum: " << *it
              << " has no. of radial projectors to be: " << numberOfProjectors
              << std::endl;
        pcout << " Projector l = 0 has: " << projectorPerOrbital[0]
              << " components" << std::endl;
        pcout << " Projector l = 1 has: " << projectorPerOrbital[1]
              << " components" << std::endl;
        pcout << " Projector l = 2 has: " << projectorPerOrbital[2]
              << " components" << std::endl;
        pcout << " Projector l = 3 has: " << projectorPerOrbital[3]
              << " components" << std::endl;
        if (totalProjectors == numberOfProjectors)
          pcout
            << "PAW::Initialization total Radial Projectors in pseudopotential file: "
            << totalProjectors << std::endl;
        else
          AssertThrow(
            false,
            dealii::ExcMessage(
              "PAW::Initialization No. of radial projectors mismatch. Check input data "));
        std::vector<double> radialMesh   = d_radialMesh[*it];
        std::vector<double> jacobianData = d_radialJacobianData[*it];
        dftfe::uInt         meshSize     = radialMesh.size();
        dftfe::uInt         alpha        = 0;
        std::vector<double> radialValuesAE(meshSize * numberOfProjectors);
        std::vector<double> radialValuesPS(meshSize * numberOfProjectors);
        std::vector<double> radialProjector(meshSize * numberOfProjectors);
        std::vector<double> radialDerivativeAE(meshSize * numberOfProjectors);
        std::vector<double> radialDerivativePS(meshSize * numberOfProjectors);
        for (dftfe::uInt lQuantumNo = 0; lQuantumNo < 4; lQuantumNo++)
          {
            if (projectorPerOrbital[lQuantumNo] == 0)
              continue;
            else
              {
                dftfe::uInt noOfProjectors = projectorPerOrbital[lQuantumNo];
                char        projectorFile[256];
                char        AEpartialWaveFile[256];
                char        PSpartialWaveFile[256];
                strcpy(projectorFile,
                       (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                        "/proj_l" + std::to_string(lQuantumNo) + ".dat")
                         .c_str());
                strcpy(AEpartialWaveFile,
                       (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                        "/allelectron_partial_l" + std::to_string(lQuantumNo) +
                        ".dat")
                         .c_str());
                strcpy(PSpartialWaveFile,
                       (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                        "/smooth_partial_l" + std::to_string(lQuantumNo) +
                        ".dat")
                         .c_str());
                std::vector<std::vector<double>> allElectronPartialData(0);
                dftUtils::readFile(noOfProjectors + 1,
                                   allElectronPartialData,
                                   AEpartialWaveFile);
                std::vector<std::vector<double>> projectorData(0);
                dftUtils::readFile(noOfProjectors + 1,
                                   projectorData,
                                   projectorFile);
                std::vector<std::vector<double>> smoothPartialData(0);
                dftUtils::readFile(noOfProjectors + 1,
                                   smoothPartialData,
                                   PSpartialWaveFile);

                for (dftfe::Int j = 1; j < noOfProjectors + 1; j++)
                  {
                    dftfe::uInt         startIndex = alpha * meshSize;
                    std::vector<double> aePhi(meshSize, 0.0);
                    std::vector<double> psPhi(meshSize, 0.0);
                    std::vector<double> proj(meshSize, 0.0);
                    for (dftfe::Int iRow = 0; iRow < meshSize; iRow++)
                      {
                        aePhi[iRow] = allElectronPartialData[iRow][j];
                        psPhi[iRow] = smoothPartialData[iRow][j];
                        proj[iRow]  = projectorData[iRow][j];
                      }
                    std::vector<double> functionDerivativesAE =
                      radialDerivativeOfMeshData(radialMesh,
                                                 jacobianData,
                                                 aePhi);
                    std::vector<double> functionDerivativesPS =
                      radialDerivativeOfMeshData(radialMesh,
                                                 jacobianData,
                                                 psPhi);

                    for (dftfe::Int iRow = 0; iRow < meshSize; iRow++)
                      {
                        radialValuesAE[startIndex + iRow]  = aePhi[iRow];
                        radialValuesPS[startIndex + iRow]  = psPhi[iRow];
                        radialProjector[startIndex + iRow] = proj[iRow];
                        radialDerivativeAE[startIndex + iRow] =
                          functionDerivativesAE[iRow];
                        radialDerivativePS[startIndex + iRow] =
                          functionDerivativesPS[iRow];
                      }
                    d_atomicProjectorFnsMap[std::make_pair(Znum, alpha)] =
                      std::make_shared<
                        AtomCenteredSphericalFunctionPAWProjectorSpline2>(
                        projectorFile,
                        lQuantumNo,
                        j,
                        noOfProjectors + 1,
                        d_RmaxAug[*it],
                        true,
                        d_nOMPThreads);
                    pcout
                      << "Projector cutoff-radius: " << Znum << " " << alpha
                      << " "
                      << d_atomicProjectorFnsMap[std::make_pair(Znum, alpha)]
                           ->getRadialCutOff()
                      << std::endl;
                    d_atomicAEPartialWaveFnsMap[std::make_pair(Znum, alpha)] =
                      std::make_shared<
                        AtomCenteredSphericalFunctionPAWProjectorSpline>(
                        AEpartialWaveFile,
                        lQuantumNo,
                        0,
                        j,
                        noOfProjectors + 1,
                        d_RmaxAug[*it],
                        true);
                    d_atomicPSPartialWaveFnsMap[std::make_pair(Znum, alpha)] =
                      std::make_shared<
                        AtomCenteredSphericalFunctionPAWProjectorSpline>(
                        PSpartialWaveFile,
                        lQuantumNo,
                        0,
                        j,
                        noOfProjectors + 1,
                        d_RmaxAug[*it],
                        true);

                    alpha++;
                  }
              }
          }
        d_radialWfcValAE[*it] = radialValuesAE;
        d_radialWfcValPS[*it] = radialValuesPS;
        d_radialProjVal[*it]  = radialProjector;
        d_radialWfcDerAE[*it] = radialDerivativeAE;
        d_radialWfcDerPS[*it] = radialDerivativePS;



      } // for loop *it
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType,
           memorySpace>::createAtomCenteredSphericalFunctionsForZeroPotential()
  {
    d_atomicZeroPotVector.clear();
    d_atomicZeroPotVector.resize(d_nOMPThreads);

    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        dftfe::uInt atomicNumber = *it;
        char        LocalDataFile[256];
        strcpy(LocalDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/zeroPotential.dat")
                 .c_str());
        std::ifstream readFile(LocalDataFile);
        if (!readFile.fail())
          {
            for (dftfe::uInt i = 0; i < d_nOMPThreads; i++)
              d_atomicZeroPotVector[i][*it] = std::make_shared<
                AtomCenteredSphericalFunctionZeroPotentialSpline>(LocalDataFile,
                                                                  1E-12,
                                                                  true);
            std::vector<std::vector<double>> zeroPotentialData(0);
            dftUtils::readFile(2, zeroPotentialData, LocalDataFile);
            dftfe::uInt         numValues = zeroPotentialData.size();
            std::vector<double> zeroPotential(numValues, 0.0);
            for (dftfe::Int iRow = 0; iRow < numValues; iRow++)
              zeroPotential[iRow] = zeroPotentialData[iRow][1];
            d_zeroPotentialRadialValues[*it] = zeroPotential;
            double Znum                      = double(*it);
            double Ncore                     = d_Ncore[*it];
            double NtildeCore                = d_NtildeCore[*it];
            double NValence                  = Znum - Ncore;
            char   valenceDensityDataFile[256];

            std::vector<double> radialMesh     = d_radialMesh[*it];
            std::vector<double> jacobianValues = d_radialJacobianData[*it];
            strcpy(valenceDensityDataFile,
                   (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                    "/pseudo_valence_density.dat")
                     .c_str());
            std::vector<std::vector<double>> valenceDensityData(0);
            dftfe::dftUtils::readFile(2,
                                      valenceDensityData,
                                      valenceDensityDataFile);
            std::vector<double> radialValenceDensity(numValues, 0.0);
            for (dftfe::Int irow = 0; irow < numValues; irow++)
              radialValenceDensity[irow] = valenceDensityData[irow][1];
            std::function<double(const dftfe::uInt &)> g =
              [&](const dftfe::uInt &i) {
                double Value = jacobianValues[i] * radialValenceDensity[i] *
                               pow(radialMesh[i], 2);
                return (Value);
              };
            double Q2 = simpsonIntegral(0, radialValenceDensity.size() - 1, g);
            pcout
              << "PAW Initialization: Integral Pseudo Valence Density from Radial integration: "
              << Q2 * sqrt(4 * M_PI) << std::endl;


            double              NTildeValence = Q2 * sqrt(4 * M_PI);
            std::vector<double> shapeFnRadial = d_atomicShapeFn[*it];
          }
        else
          {
            strcpy(LocalDataFile,
                   (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                    "/localPotential.dat")
                     .c_str());
            std::vector<std::vector<double>> localPotentialData(0);
            dftUtils::readFile(2, localPotentialData, LocalDataFile);
            dftfe::uInt numValues = localPotentialData.size();
            // std::vector<double> zeroPotential =
            //   computeZeroPotentialFromLocalPotential(d_radialMesh[*it],
            //                                          d_radialJacobianData[*it],
            //                                          radialLocalPotentialData,
            //                                          d_atomCoreDensityAE[*it],
            //                                          d_atomCoreDensityPS[*it],
            //                                          d_atomicShapeFn[*it],
            //                                          radialValenceDensity,
            //                                          Znum,
            //                                          Ncore,
            //                                          NtildeCore,
            //                                          NValence,
            //                                          NTildeValence);
          }

      } //*it loop
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::getRadialValenceDensity(dftfe::uInt Znum,
                                                            double      rad)
  {
    dftfe::uInt threadId = omp_get_thread_num();
    double      Value =
      d_atomicValenceDensityVector[threadId][Znum]->getRadialValue(rad) /
      (sqrt(4 * M_PI));

    return (Value);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::getRadialValenceDensity(
    dftfe::uInt          Znum,
    double               rad,
    std::vector<double> &Val)
  {
    dftfe::uInt threadId = omp_get_thread_num();
    Val.clear();
    Val = d_atomicValenceDensityVector[threadId][Znum]->getDerivativeValue(rad);
    for (dftfe::Int i = 0; i < Val.size(); i++)
      Val[i] /= sqrt(4 * M_PI);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::getRmaxValenceDensity(dftfe::uInt Znum)
  {
    dftfe::uInt threadId = omp_get_thread_num();
    return (d_atomicValenceDensityVector[threadId][Znum]->getRadialCutOff());
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::getRmaxCoreDensity(dftfe::uInt Znum)
  {
    dftfe::uInt threadId = omp_get_thread_num();

    return (d_atomicCoreDensityVector[threadId][Znum]->getRadialCutOff());
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::getRadialCoreDensity(dftfe::uInt Znum,
                                                         double      rad)
  {
    dftfe::uInt threadId = omp_get_thread_num();
    double      Value    = 0.0;
    if (d_atomTypeCoreFlagMap[Znum])
      Value = d_atomicCoreDensityVector[threadId][Znum]->getRadialValue(rad);
    Value /= sqrt(4 * M_PI);
    return (Value);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::map<dealii::CellId, std::vector<double>> &
  pawClass<ValueType, memorySpace>::getRhoCoreRefinedValues()
  {
    return d_rhoCoreRefinedValues;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::map<dealii::CellId, std::vector<double>> &
  pawClass<ValueType, memorySpace>::getRhoCoreCorrectionValues()
  {
    return d_rhoCoreCorrectionValues;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>> &
  pawClass<ValueType, memorySpace>::getRhoCoreAtomRefinedValues()
  {
    return d_rhoCoreAtomsRefinedValues;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>> &
  pawClass<ValueType, memorySpace>::getGradRhoCoreAtomRefinedValues()
  {
    return d_gradRhoCoreAtomsRefinedValues;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>> &
  pawClass<ValueType, memorySpace>::getHessianRhoCoreAtomRefinedValues()
  {
    return d_HessianRhoCoreAtomsRefinedValues;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::map<dftfe::Int, std::map<dftfe::uInt, std::vector<double>>> &
  pawClass<ValueType, memorySpace>::getAtomDependentCompensationCharegValues()
  {
    return d_bAtomsValuesQuadPoints;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>> &
  pawClass<ValueType, memorySpace>::getRhoCoreAtomCorrectionValues()
  {
    return d_rhoCoreAtomsCorrectionValues;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>> &
  pawClass<ValueType, memorySpace>::getGradRhoCoreAtomCorrectionValues()
  {
    return d_gradRhoCoreAtomsCorrectionValues;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>> &
  pawClass<ValueType, memorySpace>::getHessianRhoCoreAtomCorrectionValues()
  {
    return d_HessianRhoCoreAtomsCorrectionValues;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::getRadialCoreDensity(
    dftfe::uInt          Znum,
    double               rad,
    std::vector<double> &Val)
  {
    dftfe::uInt threadId = omp_get_thread_num();
    Val.clear();
    if (d_atomTypeCoreFlagMap[Znum])
      Val = d_atomicCoreDensityVector[threadId][Znum]->getDerivativeValue(rad);
    else
      Val = std::vector<double>(3, 0.0);
    for (dftfe::Int i = 0; i < Val.size(); i++)
      Val[i] /= sqrt(4 * M_PI);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::getRadialZeroPotential(
    dftfe::uInt          Znum,
    double               rad,
    std::vector<double> &Val)
  {
    dftfe::uInt threadId = omp_get_thread_num();
    Val.clear();
    Val = d_atomicZeroPotVector[threadId][Znum]->getDerivativeValue(rad);
    for (dftfe::Int i = 0; i < Val.size(); i++)
      Val[i] /= sqrt(4 * M_PI);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::getRmaxZeroPotential(dftfe::uInt Znum)
  {
    return (d_atomicZeroPotVector[0][Znum]->getRadialCutOff());
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::getMaxAugmentationRadius()
  {
    double maxAugmentationRadius = 0.0;
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        if (d_RmaxAug[*it] > maxAugmentationRadius)
          maxAugmentationRadius = d_RmaxAug[*it];
      }
    pcout << "PAWClass:: Max augmentation radius: " << maxAugmentationRadius
          << std::endl;
    return (maxAugmentationRadius);
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  bool
  pawClass<ValueType, memorySpace>::coreNuclearDensityPresent(dftfe::uInt Znum)
  {
    return (d_atomTypeCoreFlagMap[Znum]);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::setImageCoordinates(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<std::vector<double>> &periodicCoords,
    std::vector<dftfe::uInt>               &imageIdsTemp,
    std::vector<double>                    &imageCoordsTemp)
  {
    imageIdsTemp.clear();
    imageCoordsTemp.clear();
    imageCoordsTemp.resize(imageIds.size() * 3, 0.0);
    std::vector<dftfe::uInt> imageLoc(int(atomLocations.size()), 0.0);
    for (dftfe::Int jImage = 0; jImage < imageIds.size(); jImage++)
      {
        dftfe::uInt atomId = (imageIds[jImage]);
        imageIdsTemp.push_back(atomId);
        dftfe::Int startLoc = imageLoc[atomId];
        imageCoordsTemp[3 * jImage + 0] =
          periodicCoords[atomId][3 * startLoc + 0];
        imageCoordsTemp[3 * jImage + 1] =
          periodicCoords[atomId][3 * startLoc + 1];
        imageCoordsTemp[3 * jImage + 2] =
          periodicCoords[atomId][3 * startLoc + 2];
        imageLoc[atomId] += 1;
      }
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
  pawClass<ValueType, memorySpace>::getNonLocalOperator()
  {
    return d_nonLocalOperator;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::shared_ptr<AtomicCenteredNonLocalOperator<
    typename dftfe::dataTypes::singlePrecType<ValueType>::type,
    memorySpace>>
  pawClass<ValueType, memorySpace>::getNonLocalOperatorSinglePrec()
  {
    return d_nonLocalOperatorSinglePrec;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  pawClass<ValueType, memorySpace>::getTotalNumberOfAtomsInCurrentProcessor()
  {
    return d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess().size();
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  pawClass<ValueType, memorySpace>::getAtomIdInCurrentProcessor(
    dftfe::uInt iAtom)
  {
    std::vector<dftfe::uInt> atomIdList =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    return (atomIdList[iAtom]);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  pawClass<ValueType, memorySpace>::getTotalNumberOfSphericalFunctionsForAtomId(
    dftfe::uInt atomId)
  {
    std::vector<dftfe::uInt> atomicNumbers =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    return (
      d_atomicProjectorFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
        atomicNumbers[atomId]));
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseColoumbicEnergyCorrection()
  {
    pcout << "Initalising Delta C Correction Term" << std::endl;
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        dftfe::uInt atomicNumber = *it;

        const dftfe::uInt numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber);
        const dftfe::uInt numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(atomicNumber);
        const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();

        const dftfe::uInt numRadialShapeFunctions =
          d_atomicShapeFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(atomicNumber);
        const dftfe::uInt numShapeFunctions =
          d_atomicShapeFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
            atomicNumber);

        dftfe::uInt RmaxIndex          = d_RmaxAugIndex[atomicNumber];
        dftfe::uInt RmaxIndexShapeFn   = d_RmaxAugIndexShapeFn[atomicNumber];
        std::vector<double> radialMesh = d_radialMesh[atomicNumber];
        const dftfe::uInt   meshSize   = radialMesh.size();
        std::vector<double> rab        = d_radialJacobianData[atomicNumber];

        std::vector<double> multipoleTable = d_multipole[atomicNumber];
        std::vector<double> Delta_Cij(numberOfProjectors * numberOfProjectors,
                                      0.0);
        std::vector<double> Delta_Cijkl(pow(numberOfProjectors, 4), 0.0);
        double              DeltaC        = 0.0;
        double              DeltaCValence = 0.0;
        std::map<int, int>  mapOfRadProjLval;
        std::vector<std::vector<dftfe::Int>> projectorDetailsOfAtom;
        for (dftfe::Int iProj = 0; iProj < numberOfRadialProjectors; iProj++)
          {
            const std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
              sphericalFunction.find(std::make_pair(atomicNumber, iProj))
                ->second;
            const dftfe::Int lQuantumNo = sphFn->getQuantumNumberl();
            mapOfRadProjLval[iProj]     = lQuantumNo;
            std::vector<dftfe::Int> temp(3, 0);
            for (dftfe::Int mQuantumNumber = -lQuantumNo;
                 mQuantumNumber <= lQuantumNo;
                 mQuantumNumber++)
              {
                temp[0] = iProj;
                temp[1] = lQuantumNo;
                temp[2] = mQuantumNumber;
                projectorDetailsOfAtom.push_back(temp);
              }
          }

        std::vector<double> psCoreDensity = d_atomCoreDensityPS[*it];
        std::vector<double> aeCoreDensity = d_atomCoreDensityAE[*it];
        std::vector<double> shapeFnRadial = d_atomicShapeFn[*it];
        std::vector<double> NcorePotential, tildeNCorePotential,
          tildeNCorePotentialFull;
        if (d_atomTypeCoreFlagMap[*it])
          {
            oneTermPoissonPotential(&aeCoreDensity[0],
                                    0,
                                    0,
                                    RmaxIndex,
                                    2,
                                    radialMesh,
                                    rab,
                                    NcorePotential);
            oneTermPoissonPotential(&psCoreDensity[0],
                                    0,
                                    0,
                                    RmaxIndex,
                                    2,
                                    radialMesh,
                                    rab,
                                    tildeNCorePotential);
            oneTermPoissonPotential(&psCoreDensity[0],
                                    0,
                                    0,
                                    meshSize - 1,
                                    2,
                                    radialMesh,
                                    rab,
                                    tildeNCorePotentialFull);
          }
        std::vector<std::vector<double>> gLPotential;
        for (dftfe::Int lShapeFn = 0; lShapeFn < numRadialShapeFunctions;
             lShapeFn++)
          {
            std::vector<double> tempPotential;
            oneTermPoissonPotential(&shapeFnRadial[lShapeFn * meshSize],
                                    lShapeFn,
                                    0,
                                    RmaxIndexShapeFn,
                                    2,
                                    radialMesh,
                                    rab,
                                    tempPotential);
            gLPotential.push_back(tempPotential);
          }
        double ShapeFn0PseudoElectronDensityContribution = 0.0,
               AllElectronDensityContribution            = 0.0,
               PseudoElectronDensityContribution         = 0.0,
               ShapeFnContribution[numShapeFunctions];
        double ShapeFn0PseudoElectronDensityContributionFull = 0.0;
        double PseudoElectronDensityContributionFull         = 0.0;
        if (d_atomTypeCoreFlagMap[*it])
          {
            std::function<double(const dftfe::uInt &)> Integral1 =
              [&](const dftfe::uInt &i) {
                double Value =
                  rab[i] * gLPotential[0][i] * psCoreDensity[i] * radialMesh[i];

                return (Value);
              };
            ShapeFn0PseudoElectronDensityContribution =
              simpsonIntegral(0, RmaxIndex, Integral1);
            ShapeFn0PseudoElectronDensityContributionFull =
              simpsonIntegral(0, meshSize - 1, Integral1);

            std::function<double(const dftfe::uInt &)> Integral2 =
              [&](const dftfe::uInt &i) {
                double Value = rab[i] * tildeNCorePotential[i] *
                               psCoreDensity[i] * radialMesh[i];
                return (Value);
              };
            std::function<double(const dftfe::uInt &)> Integral2Full =
              [&](const dftfe::uInt &i) {
                double Value = rab[i] * tildeNCorePotentialFull[i] *
                               psCoreDensity[i] * radialMesh[i];
                return (Value);
              };
            PseudoElectronDensityContribution =
              simpsonIntegral(0, RmaxIndex, Integral2);
            PseudoElectronDensityContributionFull =
              simpsonIntegral(0, meshSize - 1, Integral2Full);
            std::function<double(const dftfe::uInt &)> Integral3 =
              [&](const dftfe::uInt &i) {
                double Value =
                  rab[i] * NcorePotential[i] * aeCoreDensity[i] * radialMesh[i];
                return (Value);
              };
            AllElectronDensityContribution =
              simpsonIntegral(0, RmaxIndex, Integral3);
          }
        dftfe::Int lshapeFn = 0;
        for (dftfe::Int L = 0; L < numRadialShapeFunctions; L++)
          {
            std::function<double(const dftfe::uInt &)> IntegralLoop =
              [&](const dftfe::uInt &i) {
                double Value = rab[i] * gLPotential[L][i] *
                               shapeFnRadial[L * meshSize + i] * radialMesh[i];
                return (Value);
              };
            double ValTempShapeFnContribution =
              simpsonIntegral(0, RmaxIndex, IntegralLoop);

            for (dftfe::Int m = -L; m <= L; m++)
              {
                ShapeFnContribution[lshapeFn] = ValTempShapeFnContribution;
                lshapeFn++;
              }
          }
        std::map<std::pair<int, int>, std::vector<double>> phiIphiJPotentialAE,
          phiIphiJPotentialPS;
        std::vector<double> allElectronPhiIphiJCoreDensityContribution(
          numberOfRadialProjectors * numberOfRadialProjectors, 0.0);
        std::vector<double> pseudoSmoothPhiIphiJCoreDensityContribution(
          numberOfRadialProjectors * numberOfRadialProjectors, 0.0);

        std::vector<double> pseudoSmoothPhiIphiJgLContribution(
          numberOfRadialProjectors * numberOfRadialProjectors *
            numShapeFunctions,
          0.0);
        std::vector<double> integralAllElectronPhiIphiJContribution(
          numberOfRadialProjectors * numberOfRadialProjectors, 0.0);

        std::vector<double> psPhi = d_radialWfcValPS[*it];
        std::vector<double> aePhi = d_radialWfcValAE[*it];

        for (dftfe::Int iProj = 0; iProj < numberOfRadialProjectors; iProj++)
          {
            dftfe::Int l_i = mapOfRadProjLval[iProj];

            for (dftfe::Int jProj = 0; jProj <= iProj; jProj++)
              {
                dftfe::Int       l_j = mapOfRadProjLval[jProj];
                const dftfe::Int index2 =
                  jProj * numberOfRadialProjectors + iProj;
                const dftfe::Int index1 =
                  iProj * numberOfRadialProjectors + jProj;
                // dftfe::Int lmin =
                //   std::min(std::abs(l_i - l_j), std::abs(l_i + l_j));
                // dftfe::Int lmax =
                //   std::max(std::abs(l_i - l_j), std::abs(l_i + l_j));
                dftfe::Int lmin = std::abs(l_i - l_j);
                dftfe::Int lmax = l_i + l_j;
                for (dftfe::Int lShapeFn = lmin; lShapeFn <= lmax; lShapeFn++)
                  {
                    std::vector<double> tempPotentialAE, tempPotentialPS;
                    twoTermPoissonPotential(&aePhi[iProj * meshSize],
                                            &aePhi[jProj * meshSize],
                                            lShapeFn,
                                            0,
                                            RmaxIndex,
                                            2,
                                            radialMesh,
                                            rab,
                                            tempPotentialAE);
                    twoTermPoissonPotential(&psPhi[iProj * meshSize],
                                            &psPhi[jProj * meshSize],
                                            lShapeFn,
                                            0,
                                            RmaxIndex,
                                            2,
                                            radialMesh,
                                            rab,
                                            tempPotentialPS);
                    phiIphiJPotentialAE[std::make_pair(index1, lShapeFn)] =
                      tempPotentialAE;
                    phiIphiJPotentialAE[std::make_pair(index2, lShapeFn)] =
                      tempPotentialAE;
                    phiIphiJPotentialPS[std::make_pair(index1, lShapeFn)] =
                      tempPotentialPS;
                    phiIphiJPotentialPS[std::make_pair(index2, lShapeFn)] =
                      tempPotentialPS;
                  }
                double              tempAE, tempPS;
                std::vector<double> tempPotentialPS =
                  phiIphiJPotentialPS[std::make_pair(index1, 0)];
                std::vector<double> tempPotentialAE =
                  phiIphiJPotentialAE[std::make_pair(index1, 0)];
                if (d_atomTypeCoreFlagMap[*it])
                  {
                    std::function<double(const dftfe::uInt &)> IntegralLoop1 =
                      [&](const dftfe::uInt &i) {
                        double Value = rab[i] * aeCoreDensity[i] *
                                       tempPotentialAE[i] * radialMesh[i];
                        return (Value);
                      };
                    tempAE = tempPotentialAE.size() == 0 ?
                               0.0 :
                               simpsonIntegral(0, RmaxIndex, IntegralLoop1);

                    std::function<double(const dftfe::uInt &)> IntegralLoop2 =
                      [&](const dftfe::uInt &i) {
                        double Value = rab[i] * psCoreDensity[i] *
                                       tempPotentialPS[i] * radialMesh[i];
                        return (Value);
                      };
                    tempPS = tempPotentialPS.size() == 0 ?
                               0.0 :
                               simpsonIntegral(0, RmaxIndex, IntegralLoop2);


                    allElectronPhiIphiJCoreDensityContribution[index1] = tempAE;
                    allElectronPhiIphiJCoreDensityContribution[index2] = tempAE;
                    pseudoSmoothPhiIphiJCoreDensityContribution[index1] =
                      tempPS;
                    pseudoSmoothPhiIphiJCoreDensityContribution[index2] =
                      tempPS;
                  } // if core present

                integralAllElectronPhiIphiJContribution[index1] =
                  integralOfProjectorsInAugmentationSphere(
                    &aePhi[iProj * meshSize],
                    &aePhi[jProj * meshSize],
                    radialMesh,
                    rab,
                    0,
                    RmaxIndex);
                integralAllElectronPhiIphiJContribution[index2] =
                  integralAllElectronPhiIphiJContribution[index1];
                dftfe::Int shapeFnIndex = 0;
                for (dftfe::Int L = 0; L < numRadialShapeFunctions; L++)
                  {
                    std::function<double(const dftfe::uInt &)> IntegralLoop =
                      [&](const dftfe::uInt &i) {
                        double Value = rab[i] * gLPotential[L][i] *
                                       psPhi[iProj * meshSize + i] *
                                       psPhi[jProj * meshSize + i] *
                                       radialMesh[i];
                        return (Value);
                      };
                    double ValTempShapeFnContribution =
                      simpsonIntegral(0, RmaxIndex, IntegralLoop);
                    for (dftfe::Int m = -L; m <= L; m++)
                      {
                        pseudoSmoothPhiIphiJgLContribution
                          [iProj * numberOfRadialProjectors *
                             numShapeFunctions +
                           jProj * numShapeFunctions + shapeFnIndex] =
                            ValTempShapeFnContribution;
                        pseudoSmoothPhiIphiJgLContribution
                          [jProj * numberOfRadialProjectors *
                             numShapeFunctions +
                           iProj * numShapeFunctions + shapeFnIndex] =
                            ValTempShapeFnContribution;
                        shapeFnIndex++;
                      }
                  }


              } // jProj
          }     // iProj
        // Computing Delta C0 Term
        double dL0       = d_DeltaL0coeff[*it];
        double valueTemp = 0.0;

        valueTemp = 0.5 * (AllElectronDensityContribution);
        // pcout << " Core-Core contribution: " << valueTemp << std::endl;
        DeltaC += valueTemp;


        valueTemp = -0.5 * (PseudoElectronDensityContribution);
        // pcout << " - psedo-pseduo contribution: " << valueTemp << std::endl;
        DeltaC += valueTemp;
        valueTemp = -0.5 * (PseudoElectronDensityContributionFull);
        // pcout << " - psedo-pseduo contribution Full: " << valueTemp
        //<< std::endl;
        DeltaCValence += valueTemp;
        valueTemp = -0.5 * (dL0 * dL0 * ShapeFnContribution[0]);
        // pcout << " -g_L(x)g_L(x) contribution: " << valueTemp << std::endl;
        DeltaC += valueTemp;
        DeltaCValence += valueTemp;

        valueTemp =
          -(d_DeltaL0coeff[*it]) * (ShapeFn0PseudoElectronDensityContribution);
        // pcout << " -g_L(x)-pseudo contribution: " << valueTemp << std::endl;
        DeltaC += valueTemp;
        // DeltaCValence += valueTemp;
        valueTemp = -(d_DeltaL0coeff[*it]) *
                    (ShapeFn0PseudoElectronDensityContributionFull);
        // pcout << " -g_L(x)-pseudo contribution Full: " << valueTemp
        //      << std::endl;
        DeltaCValence += valueTemp;
        valueTemp =
          -sqrt(4 * M_PI) * (*it) *
          integralOfDensity(&aeCoreDensity[0], radialMesh, rab, 0, RmaxIndex);

        // pcout << " integral core/r: " << valueTemp << std::endl;
        DeltaC += valueTemp;

        // pcout << "Start of Filling in entries to Delta C_ij matrices"
        //<< std::endl;

        for (dftfe::Int i = 0; i < numberOfProjectors; i++)
          {
            dftfe::Int l_i           = projectorDetailsOfAtom[i][1];
            dftfe::Int m_i           = projectorDetailsOfAtom[i][2];
            dftfe::Int radProjIndexI = projectorDetailsOfAtom[i][0];

            for (dftfe::Int j = 0; j < numberOfProjectors; j++)
              {
                dftfe::Int l_j           = projectorDetailsOfAtom[j][1];
                dftfe::Int m_j           = projectorDetailsOfAtom[j][2];
                dftfe::Int radProjIndexJ = projectorDetailsOfAtom[j][0];
                double     GauntValueij  = gaunt(l_i, l_j, 0, m_i, m_j, 0);
                if (d_atomTypeCoreFlagMap[*it])
                  {
                    Delta_Cij[i * numberOfProjectors + j] +=
                      GauntValueij *
                      (allElectronPhiIphiJCoreDensityContribution
                         [radProjIndexI * numberOfRadialProjectors +
                          radProjIndexJ] -
                       pseudoSmoothPhiIphiJCoreDensityContribution
                         [radProjIndexI * numberOfRadialProjectors +
                          radProjIndexJ]);
                  }
                if (l_i == l_j && m_i == m_j)
                  {
                    Delta_Cij[i * numberOfProjectors + j] +=
                      -(double(atomicNumber)) *
                      integralAllElectronPhiIphiJContribution
                        [radProjIndexI * numberOfRadialProjectors +
                         radProjIndexJ];
                  }
                double multipoleValue =
                  multipoleTable[radProjIndexI * numberOfRadialProjectors +
                                 radProjIndexJ];
                Delta_Cij[i * numberOfProjectors + j] -=
                  multipoleValue * GauntValueij *
                  (dL0 * ShapeFnContribution[0]);

                if (d_atomTypeCoreFlagMap[*it])
                  {
                    Delta_Cij[i * numberOfProjectors + j] -=
                      multipoleValue *
                      ShapeFn0PseudoElectronDensityContribution * GauntValueij;
                  }
                Delta_Cij[i * numberOfProjectors + j] -=
                  GauntValueij * dL0 *
                  pseudoSmoothPhiIphiJgLContribution
                    [radProjIndexI * numShapeFunctions *
                       numberOfRadialProjectors +
                     radProjIndexJ * numShapeFunctions + 0];
              } // j
          }     // i
        pcout << "Start of Filling in entries to Delta C_ijkl matrices"
              << std::endl;
        for (dftfe::Int iProj = 0; iProj < numberOfProjectors; iProj++)
          {
            dftfe::Int l_i           = projectorDetailsOfAtom[iProj][1];
            dftfe::Int m_i           = projectorDetailsOfAtom[iProj][2];
            dftfe::Int radProjIndexI = projectorDetailsOfAtom[iProj][0];

            for (dftfe::Int jProj = 0; jProj < numberOfProjectors; jProj++)
              {
                dftfe::Int l_j           = projectorDetailsOfAtom[jProj][1];
                dftfe::Int m_j           = projectorDetailsOfAtom[jProj][2];
                dftfe::Int radProjIndexJ = projectorDetailsOfAtom[jProj][0];
                const dftfe::Int index_ij =
                  numberOfRadialProjectors * radProjIndexI + radProjIndexJ;

                for (dftfe::Int kProj = 0; kProj < numberOfProjectors; kProj++)
                  {
                    dftfe::Int l_k           = projectorDetailsOfAtom[kProj][1];
                    dftfe::Int m_k           = projectorDetailsOfAtom[kProj][2];
                    dftfe::Int radProjIndexK = projectorDetailsOfAtom[kProj][0];
                    for (dftfe::Int lProj = 0; lProj < numberOfProjectors;
                         lProj++)
                      {
                        dftfe::Int l_l = projectorDetailsOfAtom[lProj][1];
                        dftfe::Int m_l = projectorDetailsOfAtom[lProj][2];
                        dftfe::Int radProjIndexL =
                          projectorDetailsOfAtom[lProj][0];
                        const dftfe::Int index =
                          pow(numberOfProjectors, 3) * iProj +
                          pow(numberOfProjectors, 2) * jProj +
                          pow(numberOfProjectors, 1) * kProj + lProj;
                        const dftfe::Int index_ijkl =
                          pow(numberOfRadialProjectors, 3) * radProjIndexI +
                          pow(numberOfRadialProjectors, 2) * radProjIndexJ +
                          pow(numberOfRadialProjectors, 1) * radProjIndexK +
                          radProjIndexL;

                        double     radValijkl = 0.0;
                        dftfe::Int lmin =
                          std::min(std::abs(l_i - l_j), std::abs(l_k - l_l));
                        dftfe::Int lmax = std::max((l_i + l_j), (l_k + l_l));
                        for (dftfe::Int lprojShapeFn = lmin;
                             lprojShapeFn <= lmax;
                             lprojShapeFn++)
                          {
                            bool flag = false;
                            for (dftfe::Int mprojShapeFn = -lprojShapeFn;
                                 mprojShapeFn <= lprojShapeFn;
                                 mprojShapeFn++)
                              {
                                double CG1, CG2;
                                CG1 = gaunt(l_i,
                                            l_j,
                                            lprojShapeFn,
                                            m_i,
                                            m_j,
                                            mprojShapeFn);
                                CG2 = gaunt(l_k,
                                            l_l,
                                            lprojShapeFn,
                                            m_k,
                                            m_l,
                                            mprojShapeFn);
                                if (std::fabs(CG1 * CG2) > 1E-10)
                                  flag = true;
                              } // mproj
                            if (flag)
                              {
                                if (phiIphiJPotentialAE
                                      .find(
                                        std::make_pair(index_ij, lprojShapeFn))
                                      ->second.size() > 0)
                                  {
                                    std::vector<double> potentialPhiIPhiJ =
                                      phiIphiJPotentialAE
                                        .find(std::make_pair(index_ij,
                                                             lprojShapeFn))
                                        ->second;
                                    std::vector<double>
                                      potentialTildePhiITildePhiJ =
                                        phiIphiJPotentialPS
                                          .find(std::make_pair(index_ij,
                                                               lprojShapeFn))
                                          ->second;


                                    std::function<double(const dftfe::uInt &)>
                                      IntegralContribution = [&](
                                                               const dftfe::uInt
                                                                 &i) {
                                        double Value1 =
                                          rab[i] * potentialPhiIPhiJ[i] *
                                          aePhi[radProjIndexK * meshSize + i] *
                                          aePhi[radProjIndexL * meshSize + i] *
                                          radialMesh[i];
                                        double Value2 =
                                          rab[i] *
                                          potentialTildePhiITildePhiJ[i] *
                                          psPhi[radProjIndexK * meshSize + i] *
                                          psPhi[radProjIndexL * meshSize + i] *
                                          radialMesh[i];
                                        return (Value1 - Value2);
                                      };


                                    double TotalValue =
                                      simpsonIntegral(0,
                                                      RmaxIndex,
                                                      IntegralContribution);
                                    double TotalContribution = 0.0;

                                    for (dftfe::Int mprojShapeFn =
                                           -lprojShapeFn;
                                         mprojShapeFn <= lprojShapeFn;
                                         mprojShapeFn++)
                                      {
                                        double CG1, CG2;
                                        CG1 = gaunt(l_i,
                                                    l_j,
                                                    lprojShapeFn,
                                                    m_i,
                                                    m_j,
                                                    mprojShapeFn);
                                        CG2 = gaunt(l_k,
                                                    l_l,
                                                    lprojShapeFn,
                                                    m_k,
                                                    m_l,
                                                    mprojShapeFn);
                                        if (std::fabs(CG1 * CG2) > 1E-10)
                                          TotalContribution +=
                                            (TotalValue)*CG1 * CG2;

                                      } // mproj
                                    Delta_Cijkl[index] +=
                                      0.5 * TotalContribution;
                                  }

                                else
                                  {
                                    pcout
                                      << "Mising Entries for lproj: " << lProj
                                      << " " << index_ij << std::endl;
                                  }
                              }
                          }
                        double     val           = 0;
                        dftfe::Int lShapeFnIndex = 0;
                        for (dftfe::Int L = 0; L < numRadialShapeFunctions; L++)
                          {
                            dftfe::Int lQuantumNo = L;
                            for (dftfe::Int mQuantumNo = -lQuantumNo;
                                 mQuantumNo <= lQuantumNo;
                                 mQuantumNo++)
                              {
                                double multipoleValue1 =
                                  multipoleTable[lQuantumNo *
                                                   numberOfRadialProjectors *
                                                   numberOfRadialProjectors +
                                                 radProjIndexI *
                                                   numberOfRadialProjectors +
                                                 radProjIndexJ];
                                double multipoleValue2 =
                                  multipoleTable[lQuantumNo *
                                                   numberOfRadialProjectors *
                                                   numberOfRadialProjectors +
                                                 radProjIndexK *
                                                   numberOfRadialProjectors +
                                                 radProjIndexL];
                                double GauntValueijL = gaunt(
                                  l_i, l_j, lQuantumNo, m_i, m_j, mQuantumNo);
                                double GauntValueklL = gaunt(
                                  l_k, l_l, lQuantumNo, m_k, m_l, mQuantumNo);
                                val += multipoleValue2 * GauntValueklL *
                                       pseudoSmoothPhiIphiJgLContribution
                                         [radProjIndexI * numShapeFunctions *
                                            numberOfRadialProjectors +
                                          radProjIndexJ * numShapeFunctions +
                                          lShapeFnIndex] *
                                       GauntValueijL;

                                val += 0.5 * multipoleValue1 * GauntValueijL *
                                       multipoleValue2 * GauntValueklL *
                                       ShapeFnContribution[lShapeFnIndex];

                                lShapeFnIndex++;
                              } // mQuantumNo
                          }     // L
                        Delta_Cijkl[index] -= val;

                      } // lProj
                  }     // kProj


              } // j
          }     // i

        // Copying the data to class
        d_deltaCij[*it]      = Delta_Cij;
        d_deltaCijkl[*it]    = Delta_Cijkl;
        d_deltaC[*it]        = DeltaC;
        d_deltaValenceC[*it] = DeltaCValence;
        // pcout << "** Delta C0 Term: " << DeltaC << std::endl;
        // pcout << "** Delta C0 Valence Term: " << DeltaCValence << std::endl;
        // printing the entries
        if (d_verbosity >= 5)
          {
            pcout << "Delta Cij Term: " << std::endl;
            for (dftfe::Int i = 0; i < numberOfProjectors; i++)
              {
                for (dftfe::Int j = 0; j < numberOfProjectors; j++)
                  pcout << Delta_Cij[i * numberOfProjectors + j] << " ";
                pcout << std::endl;
              }
            pcout << "Delta sum_klCijkl: " << std::endl;
            for (dftfe::Int i = 0; i < numberOfProjectors; i++)
              {
                for (dftfe::Int j = 0; j < numberOfProjectors; j++)
                  {
                    double Value = 0.0;
                    for (dftfe::Int k = 0; k < numberOfProjectors; k++)
                      {
                        for (dftfe::Int l = 0; l < numberOfProjectors; l++)
                          {
                            Value +=
                              Delta_Cijkl[i * pow(numberOfProjectors, 3) +
                                          j * pow(numberOfProjectors, 2) +
                                          k * numberOfProjectors + l];
                          }
                      }
                    pcout << Value << " ";
                  }
                pcout << std::endl;
              }
          }

      } //*it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseZeroPotential()
  {
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        dftfe::uInt       atomicNumber = *it;
        const dftfe::uInt numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber);
        const dftfe::uInt numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(atomicNumber);
        const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();

        dftfe::uInt         RmaxIndex  = d_RmaxAugIndex[atomicNumber];
        std::vector<double> radialMesh = d_radialMesh[atomicNumber];
        std::vector<double> rab        = d_radialJacobianData[atomicNumber];

        std::vector<double> tempZeroPotentialIJ(numberOfProjectors *
                                                  numberOfProjectors,
                                                0.0);
        std::vector<double> radialIntegralData(numberOfRadialProjectors *
                                                 numberOfRadialProjectors,
                                               0.0);
        std::vector<double> radialPSWaveFunctionsData =
          d_radialWfcValPS[atomicNumber];
        std::vector<double> zeroPotentialData =
          d_zeroPotentialRadialValues[atomicNumber];
        for (dftfe::Int i = 0; i < numberOfRadialProjectors; i++)
          {
            for (dftfe::Int j = 0; j <= i; j++)
              {
                radialIntegralData[i * numberOfRadialProjectors + j] =
                  threeTermIntegrationOverAugmentationSphere(
                    &radialPSWaveFunctionsData[i * radialMesh.size()],
                    &radialPSWaveFunctionsData[j * radialMesh.size()],
                    &zeroPotentialData[0],
                    radialMesh,
                    rab,
                    0,
                    RmaxIndex);
                radialIntegralData[j * numberOfRadialProjectors + i] =
                  radialIntegralData[i * numberOfRadialProjectors + j];
              } // j
          }     // i

        dftfe::Int projIndexI = 0;
        if (d_verbosity >= 5)
          pcout << "Delta Zero potential " << std::endl;
        for (dftfe::Int iProj = 0; iProj < numberOfRadialProjectors; iProj++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
              sphericalFunction.find(std::make_pair(atomicNumber, iProj))
                ->second;
            const dftfe::Int lQuantumNo_i = sphFn_i->getQuantumNumberl();
            for (dftfe::Int mQuantumNumber_i = -lQuantumNo_i;
                 mQuantumNumber_i <= lQuantumNo_i;
                 mQuantumNumber_i++)
              {
                dftfe::Int projIndexJ = 0;
                for (dftfe::Int jProj = 0; jProj < numberOfRadialProjectors;
                     jProj++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_j =
                      sphericalFunction
                        .find(std::make_pair(atomicNumber, jProj))
                        ->second;
                    const dftfe::Int lQuantumNo_j =
                      sphFn_j->getQuantumNumberl();
                    for (dftfe::Int mQuantumNumber_j = -lQuantumNo_j;
                         mQuantumNumber_j <= lQuantumNo_j;
                         mQuantumNumber_j++)
                      {
                        tempZeroPotentialIJ[projIndexI * numberOfProjectors +
                                            projIndexJ] =
                          gaunt(lQuantumNo_i,
                                lQuantumNo_j,
                                0,
                                mQuantumNumber_i,
                                mQuantumNumber_j,
                                0) *
                          radialIntegralData[iProj * numberOfRadialProjectors +
                                             jProj];
                        if (d_verbosity >= 5)
                          pcout << tempZeroPotentialIJ[projIndexI *
                                                         numberOfProjectors +
                                                       projIndexJ]
                                << " ";
                        projIndexJ++;
                      } // mQuantumNumber_j

                  } // jProj
                if (d_verbosity >= 5)
                  pcout << std::endl;
                projIndexI++;
              } // mQuantumNumber_i



          } // iProj
        double zeroPotentialEnergy = 0.0;
        if (d_atomTypeCoreFlagMap[*it])
          {
            std::vector<double> NtildeCore = d_atomCoreDensityPS[*it];
            std::function<double(const dftfe::uInt &)> Integral =
              [&](const dftfe::uInt &rpoint) {
                double Val1  = NtildeCore[rpoint] * zeroPotentialData[rpoint];
                double Value = rab[rpoint] * (Val1)*pow(radialMesh[rpoint], 2);
                return (Value);
              };
            zeroPotentialEnergy = simpsonIntegral(0, RmaxIndex, Integral);
          }

        d_zeroPotentialCoreEnergyContribution[*it] = zeroPotentialEnergy;
        pcout << "PAWClass Init: zero potential energy: " << zeroPotentialEnergy
              << std::endl;
        d_zeroPotentialij[*it] = tempZeroPotentialIJ;
      } //*it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::
    initialiseExchangeCorrelationEnergyCorrection(dftfe::uInt spinIndex)
  {
    bool isGGA =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);
    dftfe::uInt                      numSpinComponents  = D_ij.size();
    double                           timegradRhoCompute = 0.0;
    double                           timeDensityCompute = 0.0;
    double                           timeXCManager      = 0.0;
    double                           timeSimpsonInt     = 0.0;
    std::vector<double>              quad_weights;
    std::vector<std::vector<double>> quad_points;
    getSphericalQuadratureRule(quad_weights, quad_points);
    double DijCreation, LDAContribution, Part0Contribution, PartAContribution,
      PartBContribution, PartCContribution, VxcCompute;
    double            TimerStart, TimerEnd;
    const dftfe::uInt numberofSphericalValues = quad_weights.size();
    const dftfe::uInt sphericalQuadNoOfBatches =
      (d_dftParamsPtr->useDevicePAWXCEvaluation) ? 1 : numberofSphericalValues;
    const dftfe::uInt sphericalQuadBatchSize =
      numberofSphericalValues / sphericalQuadNoOfBatches;
    const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction =
        d_atomicProjectorFnsContainer->getSphericalFunctions();
    std::vector<dftfe::uInt> atomicNumbers =
      d_atomicProjectorFnsContainer->getAtomicNumbers();

    if (d_LocallyOwnedAtomId.size() > 0)
      {
        d_ExchangeCorrelationEnergyCorrectionTerm.clear();
        for (dftfe::Int iAtomList = 0; iAtomList < d_LocallyOwnedAtomId.size();
             iAtomList++)
          {
            const dftfe::uInt atomId = d_LocallyOwnedAtomId[iAtomList];
            const dftfe::uInt Znum   = atomicNumbers[atomId];
            const dftfe::uInt numberOfProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            d_ExchangeCorrelationEnergyCorrectionTerm[atomId] =
              std::vector<double>(numberOfProjectors * numberOfProjectors, 0.0);
          }

        dftfe::Int iAtomList = 0;
        std::unordered_map<
          std::string,
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
          AEdensityProjectionInputs;
        std::unordered_map<
          std::string,
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
                    PSdensityProjectionInputs;
        dftfe::uInt ZNumOld = 0;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          atomDensityAllelectron, atomDensitySmooth;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          atomGradDensityAllelectron, atomGradDensitySmooth;
#if defined(DFTFE_WITH_DEVICE)
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          SimpsonWeightsDevice, atomCoreDensityAllelectronDevice,
          atomCoreDensitySmoothDevice, productOfAEpartialWfcDevice,
          productOfPSpartialWfcDevice, radialValuesDevice, rabValuesDevice,
          atomDensityAEDevice, atomDensityPSDevice, deltaVxcIJValues,
          pseudoSmoothPDEC, pseudoSmoothPDEX, allElectronPDEC, allElectronPDEX,
          DijYijDevice, sphericalQuadWeightsDevice,
          productOfSphericalHarmonicsDevice;

        // GGA Terms
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          atomDensityGradientAllElectron_0Device,
          atomDensityGradientSmooth_0Device,
          atomDensityGradientAllElectron_1Device,
          atomDensityGradientSmooth_1Device,
          atomDensityGradientAllElectron_2Device,
          atomDensityGradientSmooth_2Device, GradThetaSphericalHarmonicsDevice,
          GradPhiSphericalHarmonicsDevice, productOfAEpartialWfcValueDevice,
          productOfPSpartialWfcValueDevice, productOfAEpartialWfcDerDevice,
          productOfPSpartialWfcDerDevice, DijGradThetaYijDevice,
          DijGradPhiYijDevice, gradCoreAllElectronDevice,
          gradCorePseudoSmoothDevice;

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          atomGradDensityAllelectronDevice, atomGradDensitySmoothDevice;

        if (d_dftParamsPtr->useDevicePAWXCEvaluation)
          sphericalQuadWeightsDevice.resize(sphericalQuadBatchSize, 0.0);
#endif
        // #pragma omp parallel for num_threads(d_nOMPThreads) private( \
          //   iAtomList, AEdensityProjectionInputs, PSdensityProjectionInputs)
        double dataAllocationTime, densityComputeTime, gradDensityTime,
          libxcTime, ldaTime, ggaTime;
        dataAllocationTime = 0.0;
        densityComputeTime = 0.0;
        gradDensityTime    = 0.0;
        libxcTime          = 0.0;
        ldaTime            = 0.0;
        ggaTime            = 0.0;
        for (std::map<dftfe::uInt, std::vector<dftfe::uInt>>::iterator it =
               d_LocallyOwnedAtomIdMapWithAtomType.begin();
             it != d_LocallyOwnedAtomIdMapWithAtomType.end();
             ++it)
          {
            double                   timeTemp = MPI_Wtime();
            std::vector<dftfe::uInt> locallyOwnedAtomId =
              d_LocallyOwnedAtomIdMapWithAtomType.find(it->first)->second;
            const dftfe::uInt Znum = it->first;
            const dftfe::uInt atomBatchSize =
              (d_dftParamsPtr->useDevicePAWXCEvaluation) ?
                locallyOwnedAtomId.size() :
                1;

            const std::vector<double> RadialMesh =
              d_radialMesh.find(Znum)->second;
            const dftfe::uInt RmaxIndex = d_RmaxAugIndex.find(Znum)->second;
            const std::vector<double> rab =
              d_radialJacobianData.find(Znum)->second;
            const dftfe::uInt RadialMeshSize = RadialMesh.size();
            const dftfe::uInt numberofValues = RmaxIndex + 5;
            AEdensityProjectionInputs["quadpts"] =
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                3 * numberofValues * sphericalQuadBatchSize * atomBatchSize,
                0.0);
            PSdensityProjectionInputs["quadpts"] =
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                3 * numberofValues * sphericalQuadBatchSize * atomBatchSize,
                0.0);
            AEdensityProjectionInputs["quadWt"] =
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                numberofValues * sphericalQuadBatchSize * atomBatchSize, 0.0);
            PSdensityProjectionInputs["quadWt"] =
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                numberofValues * sphericalQuadBatchSize * atomBatchSize, 0.0);
            std::pair<dftfe::uInt, dftfe::uInt> radialIndex =
              std::make_pair<dftfe::uInt, dftfe::uInt>(
                0, numberofValues * sphericalQuadBatchSize * atomBatchSize);
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              &PSdensityValsForXC = PSdensityProjectionInputs["densityFunc"];
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              &AEdensityValsForXC = AEdensityProjectionInputs["densityFunc"];
            PSdensityValsForXC.resize(2 * numberofValues *
                                        sphericalQuadBatchSize * atomBatchSize,
                                      0.0);
            AEdensityValsForXC.resize(2 * numberofValues *
                                        sphericalQuadBatchSize * atomBatchSize,
                                      0.0);

            const std::vector<double> SimpsonWeights =
              simpsonIntegralWeights(0, RmaxIndex);
            std::vector<double> integralValue(RmaxIndex + 1, 0.0);
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
                PSdensityGradForXC.resize(2 * numberofValues *
                                            sphericalQuadBatchSize *
                                            atomBatchSize * 3,
                                          0.0);
                AEdensityGradForXC.resize(2 * numberofValues *
                                            sphericalQuadBatchSize *
                                            atomBatchSize * 3,
                                          0.0);
              }
            const dftfe::uInt numberOfProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            const dftfe::uInt numberOfRadialProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
            const dftfe::uInt numberOfProjectorsSq =
              numberOfProjectors * numberOfProjectors;
            if (d_dftParamsPtr->useDevicePAWXCEvaluation)
              {
#if defined(DFTFE_WITH_DEVICE)
                // Transfer appropriate Data and reserve on GPUs
                double              Yi, Yj;
                std::vector<double> productOfAEpartialWfc =
                  d_productOfAEpartialWfc[Znum];
                std::vector<double> productOfPSpartialWfc =
                  d_productOfPSpartialWfc[Znum];
                std::vector<double> SphericalHarmonics(numberOfProjectors *
                                                         numberOfProjectors *
                                                         sphericalQuadBatchSize,
                                                       0.0);
                std::vector<double> GradThetaSphericalHarmonics(
                  numberOfProjectors * numberOfProjectors *
                    sphericalQuadBatchSize,
                  0.0);
                std::vector<double> GradPhiSphericalHarmonics(
                  numberOfProjectors * numberOfProjectors *
                    sphericalQuadBatchSize,
                  0.0);
                const char   transA = 'N', transB = 'N';
                const double Alpha = 1, Beta = 0.0;
                const double AlphaRho = 1.0, BetaRhoGPU = 0.0, BetaRho = 1.0;
                const double AlphaGradRho   = 2.0;
                const double BetaGradRhoGPU = 0.0, BetaGradRho = 1.0;
                const dftfe::uInt inc = 1;
                SimpsonWeightsDevice.resize(SimpsonWeights.size(), 0.0);
                SimpsonWeightsDevice.copyFrom(SimpsonWeights);
                atomCoreDensityAllelectronDevice.resize(
                  numberofValues * sphericalQuadBatchSize * atomBatchSize * 2,
                  0.0); // Double

                atomCoreDensitySmoothDevice.resize(numberofValues *
                                                     sphericalQuadBatchSize *
                                                     atomBatchSize * 2,
                                                   0.0); // Double
                if (d_atomTypeCoreFlagMap[Znum])
                  {
                    atomCoreDensityAllelectronDevice.copyFrom(
                      d_atomCoreDensityAE[Znum], numberofValues, 0, 0);
                    atomCoreDensitySmoothDevice.copyFrom(
                      d_atomCoreDensityPS[Znum], numberofValues, 0, 0);

                    for (dftfe::Int q = 1; q < sphericalQuadBatchSize * 2;
                         q++) // to be modified
                      {
                        d_BLASWrapperDevicePtr->xaxpy(
                          numberofValues,
                          &Alpha,
                          atomCoreDensityAllelectronDevice.data(),
                          inc,
                          atomCoreDensityAllelectronDevice.data() +
                            q * numberofValues,
                          inc);
                        d_BLASWrapperDevicePtr->xaxpy(
                          numberofValues,
                          &Alpha,
                          atomCoreDensitySmoothDevice.data(),
                          inc,
                          atomCoreDensitySmoothDevice.data() +
                            q * numberofValues,
                          inc);
                      }
                    for (dftfe::Int iAtom = 1; iAtom < atomBatchSize; iAtom++)
                      {
                        d_BLASWrapperDevicePtr->xaxpy(
                          numberofValues * sphericalQuadBatchSize * 2,
                          &Alpha,
                          atomCoreDensityAllelectronDevice.data(),
                          inc,
                          atomCoreDensityAllelectronDevice.data() +
                            iAtom * numberofValues * sphericalQuadBatchSize * 2,
                          inc);
                        d_BLASWrapperDevicePtr->xaxpy(
                          numberofValues * sphericalQuadBatchSize * 2,
                          &Alpha,
                          atomCoreDensitySmoothDevice.data(),
                          inc,
                          atomCoreDensitySmoothDevice.data() +
                            iAtom * numberofValues * sphericalQuadBatchSize * 2,
                          inc);
                      }
                  }

                productOfAEpartialWfcDevice.resize(
                  d_productOfAEpartialWfc[Znum].size(), 0.0);
                productOfAEpartialWfcDevice.copyFrom(
                  d_productOfAEpartialWfc[Znum]);
                productOfPSpartialWfcDevice.resize(
                  d_productOfAEpartialWfc[Znum].size(), 0.0);
                productOfPSpartialWfcDevice.copyFrom(
                  d_productOfPSpartialWfc[Znum]);
                radialValuesDevice.resize(RadialMesh.size());
                rabValuesDevice.resize(RadialMesh.size());
                radialValuesDevice.copyFrom(RadialMesh);
                rabValuesDevice.copyFrom(rab);

                atomDensityAEDevice.resize(numberofValues *
                                             sphericalQuadBatchSize *
                                             atomBatchSize * 2,
                                           0.0);
                atomDensityPSDevice.resize(numberofValues *
                                             sphericalQuadBatchSize *
                                             atomBatchSize * 2,
                                           0.0);
                allElectronPDEX.resize((isGGA ? 3 : 1) * numberofValues *
                                         sphericalQuadBatchSize * atomBatchSize,
                                       0.0);
                allElectronPDEC.resize((isGGA ? 3 : 1) * numberofValues *
                                         sphericalQuadBatchSize * atomBatchSize,
                                       0.0);
                pseudoSmoothPDEX.resize((isGGA ? 3 : 1) * numberofValues *
                                          sphericalQuadBatchSize *
                                          atomBatchSize,
                                        0.0);
                pseudoSmoothPDEC.resize((isGGA ? 3 : 1) * numberofValues *
                                          sphericalQuadBatchSize *
                                          atomBatchSize,
                                        0.0);
                deltaVxcIJValues.resize(numberofValues * numberOfProjectors *
                                          numberOfProjectors * atomBatchSize,
                                        0.0);
                DijYijDevice.resize(numberOfProjectors * numberOfProjectors *
                                      sphericalQuadBatchSize * atomBatchSize *
                                      2,
                                    0.0);
                productOfSphericalHarmonicsDevice.resize(
                  numberOfProjectors * numberOfProjectors *
                    sphericalQuadBatchSize,
                  0.0);
                atomDensitySmooth.resize(numberofValues *
                                           sphericalQuadBatchSize *
                                           atomBatchSize * 2,
                                         0.0);
                atomDensityAllelectron.resize(numberofValues *
                                                sphericalQuadBatchSize *
                                                atomBatchSize * 2,
                                              0.0);

                if (isGGA)
                  {
                    atomDensityGradientAllElectron_0Device.resize(
                      numberofValues * sphericalQuadBatchSize * atomBatchSize *
                        2,
                      0.0);
                    atomDensityGradientSmooth_0Device.resize(
                      numberofValues * sphericalQuadBatchSize * atomBatchSize *
                        2,
                      0.0);
                    atomDensityGradientAllElectron_1Device.resize(
                      numberofValues * sphericalQuadBatchSize * atomBatchSize *
                        2,
                      0.0);
                    atomDensityGradientSmooth_1Device.resize(
                      numberofValues * sphericalQuadBatchSize * atomBatchSize *
                        2,
                      0.0);
                    atomDensityGradientAllElectron_2Device.resize(
                      numberofValues * sphericalQuadBatchSize * atomBatchSize *
                        2,
                      0.0);
                    atomDensityGradientSmooth_2Device.resize(
                      numberofValues * sphericalQuadBatchSize * atomBatchSize *
                        2,
                      0.0);
                    GradThetaSphericalHarmonicsDevice.resize(
                      numberOfProjectorsSq * sphericalQuadBatchSize *
                        atomBatchSize,
                      0.0);
                    GradPhiSphericalHarmonicsDevice.resize(
                      numberOfProjectorsSq * sphericalQuadBatchSize *
                        atomBatchSize,
                      0.0);
                    productOfAEpartialWfcValueDevice.resize(
                      d_productOfAEpartialWfcValue[Znum].size(), 0.0);
                    productOfPSpartialWfcValueDevice.resize(
                      d_productOfAEpartialWfcValue[Znum].size(), 0.0);
                    productOfAEpartialWfcDerDevice.resize(
                      d_productOfAEpartialWfcValue[Znum].size(), 0.0);
                    productOfPSpartialWfcDerDevice.resize(
                      d_productOfAEpartialWfcValue[Znum].size(), 0.0);
                    gradCoreAllElectronDevice.resize(numberofValues *
                                                       sphericalQuadBatchSize *
                                                       atomBatchSize * 2,
                                                     0.0);
                    gradCorePseudoSmoothDevice.resize(numberofValues *
                                                        sphericalQuadBatchSize *
                                                        atomBatchSize * 2,
                                                      0.0);
                    if (d_atomTypeCoreFlagMap[Znum])
                      {
                        gradCoreAllElectronDevice.copyFrom(d_gradCoreAE[Znum],
                                                           numberofValues,
                                                           0,
                                                           0);
                        gradCorePseudoSmoothDevice.copyFrom(d_gradCorePS[Znum],
                                                            numberofValues,
                                                            0,
                                                            0);
                        for (dftfe::Int q = 1; q < sphericalQuadBatchSize * 2;
                             q++) // to be modified
                          {
                            d_BLASWrapperDevicePtr->xaxpy(
                              numberofValues,
                              &Alpha,
                              gradCoreAllElectronDevice.data(),
                              inc,
                              gradCoreAllElectronDevice.data() +
                                q * numberofValues,
                              inc);
                            d_BLASWrapperDevicePtr->xaxpy(
                              numberofValues,
                              &Alpha,
                              gradCorePseudoSmoothDevice.data(),
                              inc,
                              gradCorePseudoSmoothDevice.data() +
                                q * numberofValues,
                              inc);
                          }
                        for (dftfe::Int iAtom = 1; iAtom < atomBatchSize;
                             iAtom++)
                          {
                            d_BLASWrapperDevicePtr->xaxpy(
                              numberofValues * sphericalQuadBatchSize * 2,
                              &Alpha,
                              gradCoreAllElectronDevice.data(),
                              inc,
                              gradCoreAllElectronDevice.data() +
                                iAtom * numberofValues *
                                  sphericalQuadBatchSize * 2,
                              inc);
                            d_BLASWrapperDevicePtr->xaxpy(
                              numberofValues * sphericalQuadBatchSize * 2,
                              &Alpha,
                              gradCorePseudoSmoothDevice.data(),
                              inc,
                              gradCorePseudoSmoothDevice.data() +
                                iAtom * numberofValues *
                                  sphericalQuadBatchSize * 2,
                              inc);
                          }
                      }
                    productOfAEpartialWfcValueDevice.copyFrom(
                      d_productOfAEpartialWfcValue[Znum]);
                    productOfPSpartialWfcValueDevice.copyFrom(
                      d_productOfPSpartialWfcValue[Znum]);
                    productOfAEpartialWfcDerDevice.copyFrom(
                      d_productOfAEpartialWfcDer[Znum]);
                    productOfPSpartialWfcDerDevice.copyFrom(
                      d_productOfPSpartialWfcDer[Znum]);

                    DijGradThetaYijDevice.resize(numberOfProjectorsSq *
                                                   sphericalQuadBatchSize *
                                                   atomBatchSize * 2,
                                                 0.0);
                    DijGradPhiYijDevice.resize(numberOfProjectorsSq *
                                                 sphericalQuadBatchSize *
                                                 atomBatchSize * 2,
                                               0.0);

                    atomGradDensityAllelectron.resize(3 * numberofValues *
                                                        sphericalQuadBatchSize *
                                                        atomBatchSize * 2,
                                                      0.0);
                    atomGradDensitySmooth.resize(3 * numberofValues *
                                                   sphericalQuadBatchSize *
                                                   atomBatchSize * 2,
                                                 0.0);
                    atomGradDensityAllelectronDevice.resize(
                      3 * numberofValues * sphericalQuadBatchSize *
                        atomBatchSize * 2,
                      0.0);
                    atomGradDensitySmoothDevice.resize(
                      3 * numberofValues * sphericalQuadBatchSize *
                        atomBatchSize * 2,
                      0.0);
                  } // GGA
                std::vector<double> spericalQuadWeights(sphericalQuadBatchSize);
                for (dftfe::Int iQuadPoint = 0;
                     iQuadPoint < sphericalQuadBatchSize;
                     iQuadPoint++)
                  {
                    dftfe::uInt qpoint = iQuadPoint;
                    spericalQuadWeights[iQuadPoint] =
                      quad_weights[qpoint] * 4 * M_PI;
                    dftfe::Int projIndexI = 0;
                    for (dftfe::Int iProj = 0; iProj < numberOfRadialProjectors;
                         iProj++)
                      {
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>
                          sphFn_i =
                            sphericalFunction.find(std::make_pair(Znum, iProj))
                              ->second;
                        const dftfe::Int lQuantumNo_i =
                          sphFn_i->getQuantumNumberl();
                        for (dftfe::Int mQuantumNumber_i = -lQuantumNo_i;
                             mQuantumNumber_i <= lQuantumNo_i;
                             mQuantumNumber_i++)
                          {
                            sphericalHarmonicUtils::getSphericalHarmonicVal(
                              quad_points[qpoint][0],
                              quad_points[qpoint][1],
                              lQuantumNo_i,
                              mQuantumNumber_i,
                              Yi);

                            dftfe::Int projIndexJ = 0;
                            for (dftfe::Int jProj = 0;
                                 jProj < numberOfRadialProjectors;
                                 jProj++)
                              {
                                std::shared_ptr<
                                  AtomCenteredSphericalFunctionBase>
                                  sphFn_j = sphericalFunction
                                              .find(std::make_pair(Znum, jProj))
                                              ->second;
                                const dftfe::Int lQuantumNo_j =
                                  sphFn_j->getQuantumNumberl();
                                for (dftfe::Int mQuantumNumber_j =
                                       -lQuantumNo_j;
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
                                                       projIndexJ +
                                                       iQuadPoint *
                                                         numberOfProjectorsSq] =
                                      Yi * Yj;
                                    if (isGGA)
                                      {
                                        std::vector<double> gradYj =
                                          derivativeOfRealSphericalHarmonic(
                                            lQuantumNo_j,
                                            mQuantumNumber_j,
                                            quad_points[qpoint][0],
                                            quad_points[qpoint][1]);

                                        GradThetaSphericalHarmonics
                                          [projIndexI * numberOfProjectors +
                                           projIndexJ +
                                           iQuadPoint * numberOfProjectorsSq] =
                                            Yi * gradYj[0];
                                        double temp =
                                          std::abs(std::sin(
                                            quad_points[qpoint][0])) <= 1E-8 ?
                                            0.0 :
                                            Yi * gradYj[1] /
                                              std::sin(quad_points[qpoint][0]);
                                        GradPhiSphericalHarmonics
                                          [projIndexI * numberOfProjectors +
                                           projIndexJ +
                                           iQuadPoint * numberOfProjectorsSq] =
                                            temp;
                                      }

                                    projIndexJ++;
                                  } // mQuantumNumber_j

                              } // jProj
                            projIndexI++;
                          } // mQuantumNumber_i
                      }     // iProj
                  }         // iQuadPoint
                sphericalQuadWeightsDevice.copyFrom(spericalQuadWeights);
                productOfSphericalHarmonicsDevice.copyFrom(SphericalHarmonics);
                if (isGGA)
                  {
                    GradThetaSphericalHarmonicsDevice.copyFrom(
                      GradThetaSphericalHarmonics);
                    GradPhiSphericalHarmonicsDevice.copyFrom(
                      GradPhiSphericalHarmonics);
                  }

                for (iAtomList = 0; iAtomList < locallyOwnedAtomId.size();
                     iAtomList += atomBatchSize)
                  {
                    std::vector<double> Delta_Excij(numberOfProjectors *
                                                      numberOfProjectors *
                                                      atomBatchSize,
                                                    0.0);
                    std::vector<double> Delta_ExcijDensity(
                      numberOfProjectors * numberOfProjectors * atomBatchSize,
                      0.0);
                    std::vector<double> Delta_ExcijSigma(numberOfProjectors *
                                                           numberOfProjectors *
                                                           atomBatchSize,
                                                         0.0);
                    double              RadialIntegralGGA = 0.0;
                    double              RadialIntegralLDA = 0.0;
                    std::vector<double> DijYij(numberOfProjectors *
                                                 numberOfProjectors *
                                                 sphericalQuadBatchSize *
                                                 atomBatchSize * 2,
                                               0.0);
                    std::vector<double> DijGradThetaYij(
                      numberOfProjectors * numberOfProjectors *
                        sphericalQuadBatchSize * atomBatchSize * 2,
                      0.0);
                    std::vector<double> DijGradPhiYij(numberOfProjectors *
                                                        numberOfProjectors *
                                                        sphericalQuadBatchSize *
                                                        atomBatchSize * 2,
                                                      0.0);
                    for (dftfe::uInt iAtom = 0; iAtom < atomBatchSize; iAtom++)
                      {
                        const dftfe::uInt   atomId = locallyOwnedAtomId[iAtom];
                        std::vector<double> Dij(numberOfProjectors *
                                                  numberOfProjectors * 2,
                                                0.0);


                        // Compute Dij_up and Dij_down
                        // For spin unpolarised case, Dij_up = Dij_down
                        if (numSpinComponents == 1)
                          {
                            std::vector<double> DijComp =
                              D_ij[0][TypeOfField::In].find(atomId)->second;
                            for (dftfe::uInt iComp = 0; iComp < 2; iComp++)
                              {
                                for (dftfe::uInt iProj = 0;
                                     iProj <
                                     numberOfProjectors * numberOfProjectors;
                                     iProj++)
                                  {
                                    Dij[iProj + iComp * numberOfProjectors *
                                                  numberOfProjectors] +=
                                      0.5 * (DijComp[iProj]);
                                  }
                              }
                          }
                        else if (numSpinComponents == 2)
                          {
                            dftfe::uInt shift =
                              numberOfProjectors * numberOfProjectors;
                            std::vector<double> DijRho =
                              D_ij[0][TypeOfField::In].find(atomId)->second;
                            std::vector<double> DijMagZ =
                              D_ij[1][TypeOfField::In].find(atomId)->second;
                            for (dftfe::uInt iProj = 0;
                                 iProj <
                                 numberOfProjectors * numberOfProjectors;
                                 iProj++)
                              {
                                Dij[iProj] +=
                                  0.5 * (DijRho[iProj] + DijMagZ[iProj]);
                                Dij[iProj + shift] +=
                                  0.5 * (DijRho[iProj] - DijMagZ[iProj]);
                              }
                          }
                        dftfe::uInt shift = sphericalQuadBatchSize *
                                            numberOfProjectorsSq *
                                            atomBatchSize;
                        for (dftfe::uInt iQuad = 0;
                             iQuad < sphericalQuadBatchSize;
                             iQuad++)
                          {
                            for (dftfe::Int proj = 0;
                                 proj < numberOfProjectorsSq;
                                 proj++)
                              {
                                DijYij[iAtom * sphericalQuadBatchSize *
                                         numberOfProjectorsSq +
                                       iQuad * numberOfProjectorsSq + proj] =
                                  Dij[proj] *
                                  SphericalHarmonics[proj +
                                                     iQuad *
                                                       numberOfProjectorsSq];
                                DijYij[iAtom * sphericalQuadBatchSize *
                                         numberOfProjectorsSq +
                                       iQuad * numberOfProjectorsSq + proj +
                                       shift] =
                                  Dij[proj + numberOfProjectorsSq] *
                                  SphericalHarmonics[proj +
                                                     iQuad *
                                                       numberOfProjectorsSq];

                                DijGradThetaYij[iAtom * sphericalQuadBatchSize *
                                                  numberOfProjectorsSq +
                                                iQuad * numberOfProjectorsSq +
                                                proj] =
                                  Dij[proj] *
                                  GradThetaSphericalHarmonics
                                    [proj + iQuad * numberOfProjectorsSq];
                                DijGradThetaYij[iAtom * sphericalQuadBatchSize *
                                                  numberOfProjectorsSq +
                                                iQuad * numberOfProjectorsSq +
                                                proj + shift] =
                                  Dij[proj + numberOfProjectorsSq] *
                                  GradThetaSphericalHarmonics
                                    [proj + iQuad * numberOfProjectorsSq];
                                DijGradPhiYij[iAtom * sphericalQuadBatchSize *
                                                numberOfProjectorsSq +
                                              iQuad * numberOfProjectorsSq +
                                              proj] =
                                  Dij[proj] *
                                  GradPhiSphericalHarmonics
                                    [proj + iQuad * numberOfProjectorsSq];
                                DijGradPhiYij[iAtom * sphericalQuadBatchSize *
                                                numberOfProjectorsSq +
                                              iQuad * numberOfProjectorsSq +
                                              proj + shift] =
                                  Dij[proj + numberOfProjectorsSq] *
                                  GradPhiSphericalHarmonics
                                    [proj + iQuad * numberOfProjectorsSq];
                              }
                          }

                      } // iAtom


                    DijYijDevice.copyFrom(DijYij);

                    if (isGGA)
                      {
                        DijGradThetaYijDevice.copyFrom(DijGradThetaYij);
                        DijGradPhiYijDevice.copyFrom(DijGradPhiYij);
                      }
                    dataAllocationTime += MPI_Wtime() - timeTemp;
                    timeTemp = MPI_Wtime();
                    // GeMM call for GPUs
                    d_BLASWrapperDevicePtr->xgemmStridedBatched(
                      transA,
                      transB,
                      inc,
                      numberofValues,
                      numberOfProjectorsSq,
                      &AlphaRho,
                      DijYijDevice.data(),
                      inc,
                      numberOfProjectorsSq,
                      productOfAEpartialWfcDevice.data(),
                      numberOfProjectorsSq,
                      0,
                      &BetaRhoGPU,
                      atomDensityAEDevice.data(),
                      inc,
                      numberofValues,
                      sphericalQuadBatchSize * atomBatchSize * 2);
                    d_BLASWrapperDevicePtr->xgemmStridedBatched(
                      transA,
                      transB,
                      inc,
                      numberofValues,
                      numberOfProjectorsSq,
                      &AlphaRho,
                      DijYijDevice.data(),
                      inc,
                      numberOfProjectorsSq,
                      productOfPSpartialWfcDevice.data(),
                      numberOfProjectorsSq,
                      0,
                      &BetaRhoGPU,
                      atomDensityPSDevice.data(),
                      inc,
                      numberofValues,
                      sphericalQuadBatchSize * atomBatchSize * 2);
                    if (d_atomTypeCoreFlagMap[Znum])
                      {
                        double scaleAlpha = 0.5;
                        d_BLASWrapperDevicePtr->xaxpy(
                          numberofValues * sphericalQuadBatchSize *
                            atomBatchSize * 2,
                          &scaleAlpha,
                          atomCoreDensityAllelectronDevice.data(),
                          inc,
                          atomDensityAEDevice.data(),
                          inc);
                        d_BLASWrapperDevicePtr->xaxpy(
                          numberofValues * sphericalQuadBatchSize *
                            atomBatchSize * 2,
                          &scaleAlpha,
                          atomCoreDensitySmoothDevice.data(),
                          inc,
                          atomDensityPSDevice.data(),
                          inc);
                      }
                    atomDensityAllelectron.copyFrom(atomDensityAEDevice);
                    atomDensitySmooth.copyFrom(atomDensityPSDevice);
                    densityComputeTime += MPI_Wtime() - timeTemp;
                    timeTemp = MPI_Wtime();
                    if (isGGA)
                      {
                        d_BLASWrapperDevicePtr->xgemmStridedBatched(
                          transA,
                          transB,
                          inc,
                          numberofValues,
                          numberOfProjectorsSq,
                          &AlphaGradRho,
                          DijYijDevice.data(),
                          inc,
                          numberOfProjectorsSq,
                          productOfAEpartialWfcDerDevice.data(),
                          numberOfProjectorsSq,
                          0,
                          &BetaGradRhoGPU,
                          atomDensityGradientAllElectron_0Device.data(),
                          inc,
                          numberofValues,
                          sphericalQuadBatchSize * atomBatchSize * 2);
                        d_BLASWrapperDevicePtr->xgemmStridedBatched(
                          transA,
                          transB,
                          inc,
                          numberofValues,
                          numberOfProjectorsSq,
                          &AlphaGradRho,
                          DijYijDevice.data(),
                          inc,
                          numberOfProjectorsSq,
                          productOfPSpartialWfcDerDevice.data(),
                          numberOfProjectorsSq,
                          0,
                          &BetaGradRhoGPU,
                          atomDensityGradientSmooth_0Device.data(),
                          inc,
                          numberofValues,
                          sphericalQuadBatchSize * atomBatchSize * 2);

                        d_BLASWrapperDevicePtr->xgemmStridedBatched(
                          transA,
                          transB,
                          inc,
                          numberofValues,
                          numberOfProjectorsSq,
                          &AlphaGradRho,
                          DijGradThetaYijDevice.data(),
                          inc,
                          numberOfProjectorsSq,
                          productOfAEpartialWfcValueDevice.data(),
                          numberOfProjectorsSq,
                          0,
                          &BetaGradRhoGPU,
                          atomDensityGradientAllElectron_1Device.data(),
                          inc,
                          numberofValues,
                          sphericalQuadBatchSize * atomBatchSize * 2);
                        d_BLASWrapperDevicePtr->xgemmStridedBatched(
                          transA,
                          transB,
                          inc,
                          numberofValues,
                          numberOfProjectorsSq,
                          &AlphaGradRho,
                          DijGradThetaYijDevice.data(),
                          inc,
                          numberOfProjectorsSq,
                          productOfPSpartialWfcValueDevice.data(),
                          numberOfProjectorsSq,
                          0,
                          &BetaGradRhoGPU,
                          atomDensityGradientSmooth_1Device.data(),
                          inc,
                          numberofValues,
                          sphericalQuadBatchSize * atomBatchSize * 2);

                        d_BLASWrapperDevicePtr->xgemmStridedBatched(
                          transA,
                          transB,
                          inc,
                          numberofValues,
                          numberOfProjectorsSq,
                          &AlphaGradRho,
                          DijGradPhiYijDevice.data(),
                          inc,
                          numberOfProjectorsSq,
                          productOfAEpartialWfcValueDevice.data(),
                          numberOfProjectorsSq,
                          0,
                          &BetaGradRhoGPU,
                          atomDensityGradientAllElectron_2Device.data(),
                          inc,
                          numberofValues,
                          sphericalQuadBatchSize * atomBatchSize * 2);
                        d_BLASWrapperDevicePtr->xgemmStridedBatched(
                          transA,
                          transB,
                          inc,
                          numberofValues,
                          numberOfProjectorsSq,
                          &AlphaGradRho,
                          DijGradPhiYijDevice.data(),
                          inc,
                          numberOfProjectorsSq,
                          productOfPSpartialWfcValueDevice.data(),
                          numberOfProjectorsSq,
                          0,
                          &BetaGradRhoGPU,
                          atomDensityGradientSmooth_2Device.data(),
                          inc,
                          numberofValues,
                          sphericalQuadBatchSize * atomBatchSize * 2);
                        if (d_atomTypeCoreFlagMap[Znum])
                          {
                            double scaleAlpha = 0.5;
                            d_BLASWrapperDevicePtr->xaxpy(
                              numberofValues * sphericalQuadBatchSize *
                                atomBatchSize * 2,
                              &scaleAlpha,
                              gradCoreAllElectronDevice.data(),
                              inc,
                              atomDensityGradientAllElectron_0Device.data(),
                              inc);
                            d_BLASWrapperDevicePtr->xaxpy(
                              numberofValues * sphericalQuadBatchSize *
                                atomBatchSize * 2,
                              &scaleAlpha,
                              gradCorePseudoSmoothDevice.data(),
                              inc,
                              atomDensityGradientSmooth_0Device.data(),
                              inc);
                          }
                        pawClassKernelsDevice::combineGradDensityContributions(
                          numberofValues * sphericalQuadBatchSize *
                            atomBatchSize * 2,
                          atomDensityGradientAllElectron_0Device.data(),
                          atomDensityGradientAllElectron_1Device.data(),
                          atomDensityGradientAllElectron_2Device.data(),
                          atomDensityGradientSmooth_0Device.data(),
                          atomDensityGradientSmooth_1Device.data(),
                          atomDensityGradientSmooth_2Device.data(),
                          atomGradDensityAllelectronDevice.data(),
                          atomGradDensitySmoothDevice.data());
                        atomGradDensityAllelectron.copyFrom(
                          atomGradDensityAllelectronDevice);
                        atomGradDensitySmooth.copyFrom(
                          atomGradDensitySmoothDevice);
                      } // GGA
                    gradDensityTime += MPI_Wtime() - timeTemp;
                    if (!isGGA)
                      {
                        // Replace This with a simple copyFrom
                        for (dftfe::Int iAtom = 0; iAtom < atomBatchSize;
                             iAtom++)
                          {
                            for (dftfe::Int iQuadPoint = 0;
                                 iQuadPoint < sphericalQuadBatchSize;
                                 iQuadPoint++)
                              {
                                for (dftfe::Int iRad = 0; iRad < numberofValues;
                                     iRad++)
                                  {
                                    dftfe::uInt indexUp =
                                      iRad + numberofValues * iQuadPoint +
                                      iAtom * numberofValues *
                                        sphericalQuadBatchSize;
                                    dftfe::uInt indexDown =
                                      indexUp + sphericalQuadBatchSize *
                                                  atomBatchSize *
                                                  numberofValues;
                                    PSdensityValsForXC[indexUp] =
                                      atomDensitySmooth[indexUp];
                                    PSdensityValsForXC[indexDown] =
                                      atomDensitySmooth[indexDown];
                                    AEdensityValsForXC[indexUp] =
                                      atomDensityAllelectron[indexUp];
                                    AEdensityValsForXC[indexDown] =
                                      atomDensityAllelectron[indexDown];
                                  } // iRad
                              }     // iQuadPoint
                          }         // iAtom

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


                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST>
                          &pdexDensitySpinUpPS = xDataOutPS
                            [xcRemainderOutputDataAttributes::pdeDensitySpinUp];
                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            &pdexDensitySpinDownPS =
                              xDataOutPS[xcRemainderOutputDataAttributes::
                                           pdeDensitySpinDown];
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST>
                          &pdecDensitySpinUpPS = cDataOutPS
                            [xcRemainderOutputDataAttributes::pdeDensitySpinUp];
                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            &pdecDensitySpinDownPS =
                              cDataOutPS[xcRemainderOutputDataAttributes::
                                           pdeDensitySpinDown];

                        d_excManagerPtr->getExcSSDFunctionalObj()
                          ->computeRhoTauDependentXCData(
                            *d_auxDensityMatrixXCPSPtr,
                            radialIndex,
                            xDataOutPS,
                            cDataOutPS);
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



                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST>
                          &pdexDensitySpinUpAE = xDataOutAE
                            [xcRemainderOutputDataAttributes::pdeDensitySpinUp];
                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            &pdexDensitySpinDownAE =
                              xDataOutAE[xcRemainderOutputDataAttributes::
                                           pdeDensitySpinDown];
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST>
                          &pdecDensitySpinUpAE = cDataOutAE
                            [xcRemainderOutputDataAttributes::pdeDensitySpinUp];
                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            &pdecDensitySpinDownAE =
                              cDataOutAE[xcRemainderOutputDataAttributes::
                                           pdeDensitySpinDown];

                        d_excManagerPtr->getExcSSDFunctionalObj()
                          ->computeRhoTauDependentXCData(
                            *d_auxDensityMatrixXCAEPtr,
                            radialIndex,
                            xDataOutAE,
                            cDataOutAE);



                        allElectronPDEX.copyFrom((spinIndex == 0 ?
                                                    pdexDensitySpinUpAE :
                                                    pdexDensitySpinDownAE));
                        allElectronPDEC.copyFrom((spinIndex == 0 ?
                                                    pdecDensitySpinUpAE :
                                                    pdecDensitySpinDownAE));
                        pseudoSmoothPDEX.copyFrom((spinIndex == 0 ?
                                                     pdexDensitySpinUpPS :
                                                     pdexDensitySpinDownPS));
                        pseudoSmoothPDEC.copyFrom((spinIndex == 0 ?
                                                     pdecDensitySpinUpPS :
                                                     pdecDensitySpinDownPS));



                        deltaVxcIJValues.setValue(0.0);
                        pawClassKernelsDevice::LDAContributiontoDeltaVxc(
                          numberofValues,
                          numberOfProjectors,
                          atomBatchSize,
                          sphericalQuadBatchSize,
                          spinIndex,
                          sphericalQuadWeightsDevice.data(),
                          productOfSphericalHarmonicsDevice.data(),
                          radialValuesDevice.data(),
                          rabValuesDevice.data(),
                          productOfAEpartialWfcDevice.data(),
                          productOfPSpartialWfcDevice.data(),
                          allElectronPDEX.data(),
                          allElectronPDEC.data(),
                          pseudoSmoothPDEX.data(),
                          pseudoSmoothPDEC.data(),
                          deltaVxcIJValues.data());


                        for (dftfe::Int iAtom = 0; iAtom < atomBatchSize;
                             iAtom++)
                          {
                            for (dftfe::Int i = 0; i < numberOfProjectors; i++)
                              {
                                // Proj J

                                for (dftfe::Int j = 0; j < numberOfProjectors;
                                     j++)
                                  {
                                    const dftfe::uInt one = 1;
                                    if (j <= i)
                                      {
                                        const dftfe::uInt indexIJ =
                                          i * numberOfProjectors + j;
                                        d_BLASWrapperDevicePtr->xdot(
                                          RmaxIndex + 1,
                                          SimpsonWeightsDevice.data(),
                                          one,
                                          deltaVxcIJValues.data() +
                                            iAtom * numberOfProjectorsSq *
                                              numberofValues +
                                            indexIJ * numberofValues,
                                          one,
                                          &RadialIntegralLDA);
                                        Delta_Excij[iAtom *
                                                      numberOfProjectorsSq +
                                                    (i * numberOfProjectors +
                                                     j)] = RadialIntegralLDA;
                                      }
                                  } // Proj J
                              }     // Proj I
                          }         // iAtom
                      }             //! GGA
                    else if (isGGA)
                      {
                        timeTemp = MPI_Wtime();
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST> &PSdensityGradForXC =
                          PSdensityProjectionInputs["gradDensityFunc"];
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST> &AEdensityGradForXC =
                          AEdensityProjectionInputs["gradDensityFunc"];

                        for (dftfe::uInt iAtom = 0; iAtom < atomBatchSize;
                             iAtom++)
                          {
                            for (dftfe::uInt iQuadPoint = 0;
                                 iQuadPoint < sphericalQuadBatchSize;
                                 iQuadPoint++)
                              {
                                for (dftfe::uInt iRad = 0;
                                     iRad < numberofValues;
                                     iRad++)
                                  {
                                    dftfe::uInt indexUp =
                                      iRad + numberofValues * iQuadPoint +
                                      iAtom * numberofValues *
                                        sphericalQuadBatchSize;
                                    dftfe::uInt indexDown =
                                      indexUp + sphericalQuadBatchSize *
                                                  atomBatchSize *
                                                  numberofValues;
                                    // LDA Contribution
                                    PSdensityValsForXC[indexUp] =
                                      atomDensitySmooth[indexUp];
                                    PSdensityValsForXC[indexDown] =
                                      atomDensitySmooth[indexDown];
                                    AEdensityValsForXC[indexUp] =
                                      atomDensityAllelectron[indexUp];
                                    AEdensityValsForXC[indexDown] =
                                      atomDensityAllelectron[indexDown];
                                    // GGA Contribution

                                    PSdensityGradForXC[3 * indexUp + 0] =
                                      atomGradDensitySmooth[3 * indexUp + 0];

                                    PSdensityGradForXC[3 * indexUp + 1] =
                                      atomGradDensitySmooth[3 * indexUp + 1];

                                    PSdensityGradForXC[3 * indexUp + 2] =
                                      atomGradDensitySmooth[3 * indexUp + 2];

                                    AEdensityGradForXC[3 * indexUp + 0] =
                                      atomGradDensityAllelectron[3 * indexUp +
                                                                 0];
                                    AEdensityGradForXC[3 * indexUp + 1] =
                                      atomGradDensityAllelectron[3 * indexUp +
                                                                 1];
                                    AEdensityGradForXC[3 * indexUp + 2] =
                                      atomGradDensityAllelectron[3 * indexUp +
                                                                 2];

                                    PSdensityGradForXC[3 * indexDown + 0] =
                                      atomGradDensitySmooth[3 * indexDown + 0];
                                    PSdensityGradForXC[3 * indexDown + 1] =
                                      atomGradDensitySmooth[3 * indexUp + 1];
                                    PSdensityGradForXC[3 * indexDown + 2] =
                                      atomGradDensitySmooth[3 * indexDown + 2];

                                    AEdensityGradForXC[3 * indexDown + 0] =
                                      atomGradDensityAllelectron[3 * indexDown +
                                                                 0];
                                    AEdensityGradForXC[3 * indexDown + 1] =
                                      atomGradDensityAllelectron[3 * indexDown +
                                                                 1];
                                    AEdensityGradForXC[3 * indexDown + 2] =
                                      atomGradDensityAllelectron[3 * indexDown +
                                                                 2];
                                  } // iRad
                              }     // iQuadPoint
                          }         // iAtom
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


                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST>
                          &pdexDensitySpinUpPS = xDataOutPS
                            [xcRemainderOutputDataAttributes::pdeDensitySpinUp];
                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            &pdexDensitySpinDownPS =
                              xDataOutPS[xcRemainderOutputDataAttributes::
                                           pdeDensitySpinDown];
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST>
                          &pdecDensitySpinUpPS = cDataOutPS
                            [xcRemainderOutputDataAttributes::pdeDensitySpinUp];
                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            &pdecDensitySpinDownPS =
                              cDataOutPS[xcRemainderOutputDataAttributes::
                                           pdeDensitySpinDown];

                        xDataOutPS[xcRemainderOutputDataAttributes::pdeSigma] =
                          dftfe::utils::MemoryStorage<
                            double,
                            dftfe::utils::MemorySpace::HOST>();
                        cDataOutPS[xcRemainderOutputDataAttributes::pdeSigma] =
                          dftfe::utils::MemoryStorage<
                            double,
                            dftfe::utils::MemorySpace::HOST>();

                        d_excManagerPtr->getExcSSDFunctionalObj()
                          ->computeRhoTauDependentXCData(
                            *d_auxDensityMatrixXCPSPtr,
                            radialIndex,
                            xDataOutPS,
                            cDataOutPS);
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST> &pdexSigmaPS =
                          xDataOutPS[xcRemainderOutputDataAttributes::pdeSigma];
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST> &pdecSigmaPS =
                          cDataOutPS[xcRemainderOutputDataAttributes::pdeSigma];



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
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST>
                          &pdexDensitySpinUpAE = xDataOutAE
                            [xcRemainderOutputDataAttributes::pdeDensitySpinUp];
                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            &pdexDensitySpinDownAE =
                              xDataOutAE[xcRemainderOutputDataAttributes::
                                           pdeDensitySpinDown];
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST>
                          &pdecDensitySpinUpAE = cDataOutAE
                            [xcRemainderOutputDataAttributes::pdeDensitySpinUp];
                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            &pdecDensitySpinDownAE =
                              cDataOutAE[xcRemainderOutputDataAttributes::
                                           pdeDensitySpinDown];
                        xDataOutAE[xcRemainderOutputDataAttributes::pdeSigma] =
                          dftfe::utils::MemoryStorage<
                            double,
                            dftfe::utils::MemorySpace::HOST>();
                        cDataOutAE[xcRemainderOutputDataAttributes::pdeSigma] =
                          dftfe::utils::MemoryStorage<
                            double,
                            dftfe::utils::MemorySpace::HOST>();


                        std::unordered_map<DensityDescriptorDataAttributes,
                                           dftfe::utils::MemoryStorage<
                                             double,
                                             dftfe::utils::MemorySpace::HOST>>
                          densityDataAE;
                        std::unordered_map<DensityDescriptorDataAttributes,
                                           dftfe::utils::MemoryStorage<
                                             double,
                                             dftfe::utils::MemorySpace::HOST>>
                          densityDataPS;
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST> &densitySpinUpAE =
                          densityDataAE
                            [DensityDescriptorDataAttributes::valuesSpinUp];
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST> &densitySpinDownAE =
                          densityDataAE
                            [DensityDescriptorDataAttributes::valuesSpinDown];
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST>
                          &gradDensitySpinUpAE = densityDataAE
                            [DensityDescriptorDataAttributes::gradValuesSpinUp];
                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            &gradDensitySpinDownAE =
                              densityDataAE[DensityDescriptorDataAttributes::
                                              gradValuesSpinDown];

                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST> &densitySpinUpPS =
                          densityDataPS
                            [DensityDescriptorDataAttributes::valuesSpinUp];
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST> &densitySpinDownPS =
                          densityDataPS
                            [DensityDescriptorDataAttributes::valuesSpinDown];
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST>
                          &gradDensitySpinUpPS = densityDataPS
                            [DensityDescriptorDataAttributes::gradValuesSpinUp];
                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            &gradDensitySpinDownPS =
                              densityDataPS[DensityDescriptorDataAttributes::
                                              gradValuesSpinDown];
                        d_auxDensityMatrixXCAEPtr->applyLocalOperations(
                          radialIndex, densityDataAE);
                        d_auxDensityMatrixXCPSPtr->applyLocalOperations(
                          radialIndex, densityDataPS);
                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            &gradDensityXCSpinIndexAE =
                              spinIndex == 0 ? gradDensitySpinUpAE :
                                               gradDensitySpinDownAE;
                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            &gradDensityXCOtherSpinIndexAE =
                              spinIndex == 0 ? gradDensitySpinDownAE :
                                               gradDensitySpinUpAE;
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::DEVICE>
                          gradDensityXCSpinIndexAEDevice(
                            gradDensityXCSpinIndexAE.size());
                        gradDensityXCSpinIndexAEDevice.copyFrom(
                          gradDensityXCSpinIndexAE);
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::DEVICE>
                          gradDensityXCOtherSpinIndexAEDevice(
                            gradDensityXCOtherSpinIndexAE.size());
                        gradDensityXCOtherSpinIndexAEDevice.copyFrom(
                          gradDensityXCOtherSpinIndexAE);

                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            &gradDensityXCSpinIndexPS =
                              spinIndex == 0 ? gradDensitySpinUpPS :
                                               gradDensitySpinDownPS;
                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            &gradDensityXCOtherSpinIndexPS =
                              spinIndex == 0 ? gradDensitySpinDownPS :
                                               gradDensitySpinUpPS;
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::DEVICE>
                          gradDensityXCSpinIndexPSDevice(
                            gradDensityXCSpinIndexPS.size());
                        gradDensityXCSpinIndexPSDevice.copyFrom(
                          gradDensityXCSpinIndexPS);
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::DEVICE>
                          gradDensityXCOtherSpinIndexPSDevice(
                            gradDensityXCOtherSpinIndexPS.size());
                        gradDensityXCOtherSpinIndexPSDevice.copyFrom(
                          gradDensityXCOtherSpinIndexPS);
                        d_excManagerPtr->getExcSSDFunctionalObj()
                          ->computeRhoTauDependentXCData(
                            *d_auxDensityMatrixXCAEPtr,
                            radialIndex,
                            xDataOutAE,
                            cDataOutAE);
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST> &pdexSigmaAE =
                          xDataOutAE[xcRemainderOutputDataAttributes::pdeSigma];
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST> &pdecSigmaAE =
                          cDataOutAE[xcRemainderOutputDataAttributes::pdeSigma];
                        libxcTime += MPI_Wtime() - timeTemp;
                        timeTemp = MPI_Wtime();

                        allElectronPDEX.copyFrom((spinIndex == 0 ?
                                                    pdexDensitySpinUpAE :
                                                    pdexDensitySpinDownAE));
                        allElectronPDEC.copyFrom((spinIndex == 0 ?
                                                    pdecDensitySpinUpAE :
                                                    pdecDensitySpinDownAE));
                        pseudoSmoothPDEX.copyFrom((spinIndex == 0 ?
                                                     pdexDensitySpinUpPS :
                                                     pdexDensitySpinDownPS));
                        pseudoSmoothPDEC.copyFrom((spinIndex == 0 ?
                                                     pdecDensitySpinUpPS :
                                                     pdecDensitySpinDownPS));
                        deltaVxcIJValues.setValue(0.0);
                        pawClassKernelsDevice::LDAContributiontoDeltaVxc(
                          numberofValues,
                          numberOfProjectors,
                          atomBatchSize,
                          sphericalQuadBatchSize,
                          spinIndex,
                          sphericalQuadWeightsDevice.data(),
                          productOfSphericalHarmonicsDevice.data(),
                          radialValuesDevice.data(),
                          rabValuesDevice.data(),
                          productOfAEpartialWfcDevice.data(),
                          productOfPSpartialWfcDevice.data(),
                          allElectronPDEX.data(),
                          allElectronPDEC.data(),
                          pseudoSmoothPDEX.data(),
                          pseudoSmoothPDEC.data(),
                          deltaVxcIJValues.data());
                        for (dftfe::uInt iAtom = 0; iAtom < atomBatchSize;
                             iAtom++)
                          {
                            for (dftfe::Int i = 0; i < numberOfProjectors; i++)
                              {
                                // Proj J

                                for (dftfe::Int j = 0; j < numberOfProjectors;
                                     j++)
                                  {
                                    const dftfe::uInt one = 1;
                                    if (j <= i)
                                      {
                                        const dftfe::uInt indexIJ =
                                          i * numberOfProjectors + j;
                                        d_BLASWrapperDevicePtr->xdot(
                                          RmaxIndex + 1,
                                          SimpsonWeightsDevice.data(),
                                          one,
                                          deltaVxcIJValues.data() +
                                            iAtom * numberOfProjectorsSq *
                                              numberofValues +
                                            indexIJ * numberofValues,
                                          one,
                                          &RadialIntegralLDA);
                                        Delta_ExcijDensity
                                          [iAtom * numberOfProjectorsSq +
                                           (i * numberOfProjectors + j)] +=
                                          RadialIntegralLDA;
                                      }
                                  } // Proj J
                              }     // Proj I
                          }         // iAtom
                        ldaTime += MPI_Wtime() - timeTemp;
                        //@Kartick some hardcoding for only spin unpolarised
                        // are
                        // done below..

                        timeTemp = MPI_Wtime();
                        allElectronPDEX.copyFrom(pdexSigmaAE);
                        allElectronPDEC.copyFrom(pdecSigmaAE);
                        pseudoSmoothPDEX.copyFrom(pdexSigmaPS);
                        pseudoSmoothPDEC.copyFrom(pdecSigmaPS);
                        deltaVxcIJValues.setValue(0.0);
                        pawClassKernelsDevice::GGAContributiontoDeltaVxc(
                          numberofValues,
                          numberOfProjectors,
                          atomBatchSize,
                          sphericalQuadBatchSize,
                          spinIndex,
                          sphericalQuadWeightsDevice.data(),
                          productOfSphericalHarmonicsDevice.data(),
                          GradPhiSphericalHarmonicsDevice.data(),
                          GradThetaSphericalHarmonicsDevice.data(),
                          radialValuesDevice.data(),
                          rabValuesDevice.data(),
                          productOfAEpartialWfcValueDevice.data(),
                          productOfPSpartialWfcValueDevice.data(),
                          productOfAEpartialWfcDerDevice.data(),
                          productOfPSpartialWfcDerDevice.data(),
                          gradDensityXCSpinIndexAEDevice.data(),
                          gradDensityXCOtherSpinIndexAEDevice.data(),
                          gradDensityXCSpinIndexPSDevice.data(),
                          gradDensityXCOtherSpinIndexPSDevice.data(),
                          allElectronPDEX.data(),
                          allElectronPDEC.data(),
                          pseudoSmoothPDEX.data(),
                          pseudoSmoothPDEC.data(),
                          deltaVxcIJValues.data());
                        for (dftfe::Int iAtom = 0; iAtom < atomBatchSize;
                             iAtom++)
                          {
                            for (dftfe::Int i = 0; i < numberOfProjectors; i++)
                              {
                                // Proj J

                                for (dftfe::Int j = 0; j < numberOfProjectors;
                                     j++)
                                  {
                                    const dftfe::uInt one = 1;
                                    if (true)
                                      {
                                        const dftfe::uInt indexIJ =
                                          i * numberOfProjectors + j;
                                        d_BLASWrapperDevicePtr->xdot(
                                          RmaxIndex + 1,
                                          SimpsonWeightsDevice.data(),
                                          one,
                                          deltaVxcIJValues.data() +
                                            iAtom * numberOfProjectorsSq *
                                              numberofValues +
                                            indexIJ * numberofValues,
                                          one,
                                          &RadialIntegralGGA);
                                        Delta_ExcijSigma
                                          [iAtom * numberOfProjectorsSq +
                                           (i * numberOfProjectors + j)] +=
                                          RadialIntegralGGA;
                                      }
                                  } // Proj J
                              }     // Proj I
                          }         // iAtom
                        ggaTime += MPI_Wtime() - timeTemp;
                      } // GGA
                    for (dftfe::Int iAtom = 0; iAtom < atomBatchSize; iAtom++)
                      {
                        const dftfe::uInt   atomId = locallyOwnedAtomId[iAtom];
                        std::vector<double> DeltaExc(numberOfProjectorsSq, 0.0);
                        for (dftfe::Int i = 0; i < numberOfProjectors; i++)
                          {
                            for (dftfe::Int j = 0; j <= i; j++)
                              {
                                if (!isGGA)
                                  {
                                    Delta_Excij[iAtom * numberOfProjectorsSq +
                                                j * numberOfProjectors + i] =
                                      Delta_Excij[iAtom * numberOfProjectorsSq +
                                                  i * numberOfProjectors + j];
                                  }
                                else
                                  {
                                    double temp =
                                      Delta_ExcijDensity
                                        [iAtom * numberOfProjectorsSq +
                                         i * numberOfProjectors + j] +
                                      Delta_ExcijSigma[iAtom *
                                                         numberOfProjectorsSq +
                                                       i * numberOfProjectors +
                                                       j];
                                    Delta_Excij[iAtom * numberOfProjectorsSq +
                                                j * numberOfProjectors + i] =
                                      temp;
                                    Delta_Excij[iAtom * numberOfProjectorsSq +
                                                i * numberOfProjectors + j] =
                                      temp;
                                  } // else

                              } // jProj

                          } // iProj
                        for (dftfe::Int i = 0; i < numberOfProjectors; i++)
                          {
                            for (dftfe::Int j = 0; j < numberOfProjectors; j++)
                              {
                                DeltaExc[i * numberOfProjectors + j] =
                                  Delta_Excij[iAtom * numberOfProjectorsSq +
                                              i * numberOfProjectors + j];
                              }
                          }

                        //
                        if (d_verbosity >= 5)
                          {
                            pcout << " Delta XC for iAtom: " << atomId
                                  << std::endl;
                            for (dftfe::Int iProj = 0;
                                 iProj < numberOfProjectors;
                                 iProj++)
                              {
                                for (dftfe::Int jProj = 0;
                                     jProj < numberOfProjectors;
                                     jProj++)
                                  pcout << DeltaExc[iProj * numberOfProjectors +
                                                    jProj]
                                        << " ";
                                pcout << std::endl;
                              }
                          }
                        d_ExchangeCorrelationEnergyCorrectionTerm.find(atomId)
                          ->second = DeltaExc;
                      } // iAtom
                  }     // iAtomList
#endif
              } // Use Device
            else
              {
                for (iAtomList = 0; iAtomList < locallyOwnedAtomId.size();
                     iAtomList += atomBatchSize)
                  {
                    std::vector<double> Delta_Excij(numberOfProjectors *
                                                      numberOfProjectors *
                                                      atomBatchSize,
                                                    0.0);
                    std::vector<double> Delta_ExcijDensity(
                      numberOfProjectors * numberOfProjectors * atomBatchSize,
                      0.0);
                    std::vector<double> Delta_ExcijSigma(numberOfProjectors *
                                                           numberOfProjectors *
                                                           atomBatchSize,
                                                         0.0);
                    double              RadialIntegralGGA = 0.0;
                    double              RadialIntegralLDA = 0.0;
                    const dftfe::uInt   atomId = locallyOwnedAtomId[iAtomList];

                    std::vector<double> Dij(numberOfProjectors *
                                              numberOfProjectors * 2,
                                            0.0);


                    // Compute Dij_up and Dij_down
                    // For spin unpolarised case, Dij_up = Dij_down
                    if (numSpinComponents == 1)
                      {
                        std::vector<double> DijComp =
                          D_ij[0][TypeOfField::In].find(atomId)->second;
                        for (dftfe::uInt iComp = 0; iComp < 2; iComp++)
                          {
                            for (dftfe::uInt iProj = 0;
                                 iProj <
                                 numberOfProjectors * numberOfProjectors;
                                 iProj++)
                              {
                                Dij[iProj + iComp * numberOfProjectors *
                                              numberOfProjectors] +=
                                  0.5 * (DijComp[iProj]);
                              }
                          }
                      }
                    else if (numSpinComponents == 2)
                      {
                        dftfe::uInt shift =
                          numberOfProjectors * numberOfProjectors;
                        std::vector<double> DijRho =
                          D_ij[0][TypeOfField::In].find(atomId)->second;
                        std::vector<double> DijMagZ =
                          D_ij[1][TypeOfField::In].find(atomId)->second;
                        for (dftfe::uInt iProj = 0;
                             iProj < numberOfProjectors * numberOfProjectors;
                             iProj++)
                          {
                            Dij[iProj] +=
                              0.5 * (DijRho[iProj] + DijMagZ[iProj]);
                            Dij[iProj + shift] +=
                              0.5 * (DijRho[iProj] - DijMagZ[iProj]);
                          }
                      }
                    if (!isGGA)
                      {
                        double Yi, Yj;

                        for (dftfe::Int qpoint = 0;
                             qpoint < numberofSphericalValues;
                             qpoint++)
                          {
                            std::vector<double> SphericalHarmonics(
                              numberOfProjectors * numberOfProjectors, 0.0);
                            // help me.. A better strategy to store this

                            std::vector<double> productOfAEpartialWfc =
                              d_productOfAEpartialWfc[Znum];
                            std::vector<double> productOfPSpartialWfc =
                              d_productOfPSpartialWfc[Znum];
                            double              quadwt = quad_weights[qpoint];
                            std::vector<double> DijYij(numberOfProjectors *
                                                         numberOfProjectors * 2,
                                                       0.0);


                            dftfe::Int shift =
                              numberOfProjectors * numberOfProjectors;
                            dftfe::Int projIndexI = 0;
                            for (dftfe::Int iProj = 0;
                                 iProj < numberOfRadialProjectors;
                                 iProj++)
                              {
                                std::shared_ptr<
                                  AtomCenteredSphericalFunctionBase>
                                  sphFn_i = sphericalFunction
                                              .find(std::make_pair(Znum, iProj))
                                              ->second;
                                const dftfe::Int lQuantumNo_i =
                                  sphFn_i->getQuantumNumberl();
                                for (int mQuantumNumber_i = -lQuantumNo_i;
                                     mQuantumNumber_i <= lQuantumNo_i;
                                     mQuantumNumber_i++)
                                  {
                                    sphericalHarmonicUtils::
                                      getSphericalHarmonicVal(
                                        quad_points[qpoint][0],
                                        quad_points[qpoint][1],
                                        lQuantumNo_i,
                                        mQuantumNumber_i,
                                        Yi);

                                    dftfe::Int projIndexJ = 0;
                                    for (dftfe::Int jProj = 0;
                                         jProj < numberOfRadialProjectors;
                                         jProj++)
                                      {
                                        std::shared_ptr<
                                          AtomCenteredSphericalFunctionBase>
                                          sphFn_j =
                                            sphericalFunction
                                              .find(std::make_pair(Znum, jProj))
                                              ->second;
                                        const dftfe::Int lQuantumNo_j =
                                          sphFn_j->getQuantumNumberl();
                                        for (dftfe::Int mQuantumNumber_j =
                                               -lQuantumNo_j;
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

                                            SphericalHarmonics
                                              [projIndexI * numberOfProjectors +
                                               projIndexJ] = Yi * Yj;
                                            SphericalHarmonics
                                              [projIndexJ * numberOfProjectors +
                                               projIndexI] = Yi * Yj;

                                            DijYij[projIndexI *
                                                     numberOfProjectors +
                                                   projIndexJ] =
                                              Yi * Yj *
                                              Dij[projIndexI *
                                                    numberOfProjectors +
                                                  projIndexJ]; //@Kartick this
                                                               // is
                                                               // not needed
                                            DijYij[projIndexJ *
                                                     numberOfProjectors +
                                                   projIndexI] =
                                              Yi * Yj *
                                              Dij[projIndexJ *
                                                    numberOfProjectors +
                                                  projIndexI];

                                            DijYij[projIndexI *
                                                     numberOfProjectors +
                                                   projIndexJ + shift] =
                                              Yi * Yj *
                                              Dij[projIndexI *
                                                    numberOfProjectors +
                                                  projIndexJ +
                                                  shift]; //@Kartick this is
                                                          // not needed
                                            DijYij[projIndexJ *
                                                     numberOfProjectors +
                                                   projIndexI + shift] =
                                              Yi * Yj *
                                              Dij[projIndexJ *
                                                    numberOfProjectors +
                                                  projIndexI + shift];

                                            projIndexJ++;
                                          } // mQuantumNumber_j

                                      } // jProj
                                    projIndexI++;
                                  } // mQuantumNumber_i



                              } // iProj


                            const char     transA = 'N', transB = 'N';
                            const double   Alpha = 1, Beta = 0.0;
                            const unsigned inc   = 1;
                            const double   Beta2 = 0.5;
                            for (dftfe::Int iComp = 0; iComp < 2; iComp++)
                              {
                                std::vector<double> atomDensityAllelectron =
                                  d_atomTypeCoreFlagMap[Znum] ?
                                    d_atomCoreDensityAE[Znum] :
                                    std::vector<double>(numberofValues, 0.0);
                                std::vector<double> atomDensitySmooth =
                                  d_atomTypeCoreFlagMap[Znum] ?
                                    d_atomCoreDensityPS[Znum] :
                                    std::vector<double>(numberofValues, 0.0);
                                d_BLASWrapperHostPtr->xgemm(
                                  transA,
                                  transB,
                                  inc,
                                  numberofValues,
                                  numberOfProjectorsSq,
                                  &Alpha,
                                  &DijYij[shift * iComp],
                                  inc,
                                  &productOfAEpartialWfc[0],
                                  numberOfProjectorsSq,
                                  &Beta2,
                                  &atomDensityAllelectron[0],
                                  inc);
                                d_BLASWrapperHostPtr->xgemm(
                                  transA,
                                  transB,
                                  inc,
                                  numberofValues,
                                  numberOfProjectorsSq,
                                  &Alpha,
                                  &DijYij[shift * iComp],
                                  inc,
                                  &productOfPSpartialWfc[0],
                                  numberOfProjectorsSq,
                                  &Beta2,
                                  &atomDensitySmooth[0],
                                  inc);

                                for (dftfe::Int iRad = 0; iRad < numberofValues;
                                     iRad++)
                                  {
                                    PSdensityValsForXC[iRad +
                                                       iComp * numberofValues] =
                                      atomDensitySmooth[iRad];
                                    AEdensityValsForXC[iRad +
                                                       iComp * numberofValues] =
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


                            std::unordered_map<
                              xcRemainderOutputDataAttributes,
                              dftfe::utils::MemoryStorage<
                                double,
                                dftfe::utils::MemorySpace::HOST>>
                              xDataOutPS;
                            std::unordered_map<
                              xcRemainderOutputDataAttributes,
                              dftfe::utils::MemoryStorage<
                                double,
                                dftfe::utils::MemorySpace::HOST>>
                              cDataOutPS;


                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdexDensitySpinUpPS =
                                xDataOutPS[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinUp];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdexDensitySpinDownPS =
                                xDataOutPS[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinDown];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdecDensitySpinUpPS =
                                cDataOutPS[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinUp];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdecDensitySpinDownPS =
                                cDataOutPS[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinDown];

                            d_excManagerPtr->getExcSSDFunctionalObj()
                              ->computeRhoTauDependentXCData(
                                *d_auxDensityMatrixXCPSPtr,
                                radialIndex,
                                xDataOutPS,
                                cDataOutPS);
                            std::unordered_map<
                              xcRemainderOutputDataAttributes,
                              dftfe::utils::MemoryStorage<
                                double,
                                dftfe::utils::MemorySpace::HOST>>
                              xDataOutAE;
                            std::unordered_map<
                              xcRemainderOutputDataAttributes,
                              dftfe::utils::MemoryStorage<
                                double,
                                dftfe::utils::MemorySpace::HOST>>
                              cDataOutAE;


                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdexDensitySpinUpAE =
                                xDataOutAE[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinUp];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdexDensitySpinDownAE =
                                xDataOutAE[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinDown];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdecDensitySpinUpAE =
                                cDataOutAE[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinUp];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdecDensitySpinDownAE =
                                cDataOutAE[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinDown];

                            d_excManagerPtr->getExcSSDFunctionalObj()
                              ->computeRhoTauDependentXCData(
                                *d_auxDensityMatrixXCAEPtr,
                                radialIndex,
                                xDataOutAE,
                                cDataOutAE);

                            const dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdexDensitySpinIndexAE =
                                spinIndex == 0 ? pdexDensitySpinUpAE :
                                                 pdexDensitySpinDownAE;
                            const dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdecDensitySpinIndexAE =
                                spinIndex == 0 ? pdecDensitySpinUpAE :
                                                 pdecDensitySpinDownAE;

                            const dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdexDensitySpinIndexPS =
                                spinIndex == 0 ? pdexDensitySpinUpPS :
                                                 pdexDensitySpinDownPS;
                            const dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdecDensitySpinIndexPS =
                                spinIndex == 0 ? pdecDensitySpinUpPS :
                                                 pdecDensitySpinDownPS;

                            for (dftfe::Int i = 0; i < numberOfProjectors; i++)
                              {
                                // Proj J

                                for (dftfe::Int j = 0; j <= i; j++)
                                  {
                                    const dftfe::uInt one = 1;
                                    for (dftfe::Int rpoint = 0;
                                         rpoint < RmaxIndex + 1;
                                         rpoint++)
                                      {
                                        dftfe::uInt index =
                                          rpoint * numberOfProjectorsSq +
                                          i * numberOfProjectors + j;
                                        double Val1 =
                                          productOfAEpartialWfc[index] *
                                          (pdexDensitySpinIndexAE[rpoint] +
                                           pdecDensitySpinIndexAE[rpoint]);
                                        double Val2 =
                                          productOfPSpartialWfc[index] *
                                          (pdexDensitySpinIndexPS[rpoint] +
                                           pdecDensitySpinIndexPS[rpoint]);
                                        integralValue[rpoint] =
                                          rab[rpoint] * (Val1 - Val2) *
                                          pow(RadialMesh[rpoint], 2);
                                      }
                                    d_BLASWrapperHostPtr->xdot(
                                      RmaxIndex + 1,
                                      &SimpsonWeights[0],
                                      one,
                                      &integralValue[0],
                                      one,
                                      &RadialIntegralLDA);
                                    RadialIntegralLDA +=
                                      SimpsonResidual(0,
                                                      RmaxIndex + 1,
                                                      integralValue);
                                    Delta_Excij[i * numberOfProjectors + j] +=
                                      RadialIntegralLDA * quadwt * 4.0 * M_PI *
                                      SphericalHarmonics[i *
                                                           numberOfProjectors +
                                                         j];
                                  } // Proj J
                              }     // Proj I

                          } // qpoint
                      }     // LDA case
                    else
                      {
                        double Yi, Yj;
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST> &PSdensityGradForXC =
                          PSdensityProjectionInputs["gradDensityFunc"];
                        dftfe::utils::MemoryStorage<
                          double,
                          dftfe::utils::MemorySpace::HOST> &AEdensityGradForXC =
                          AEdensityProjectionInputs["gradDensityFunc"];
                        for (int qpoint = 0; qpoint < numberofSphericalValues;
                             qpoint++)
                          {
                            std::vector<double> SphericalHarmonics(
                              numberOfProjectors * numberOfProjectors, 0.0);
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
                            std::vector<double> DijGradPhiYij(
                              numberOfProjectors * numberOfProjectors * 2, 0.0);

                            int        projIndexI = 0;
                            dftfe::Int shift =
                              numberOfProjectors * numberOfProjectors;
                            for (int iProj = 0;
                                 iProj < numberOfRadialProjectors;
                                 iProj++)
                              {
                                std::shared_ptr<
                                  AtomCenteredSphericalFunctionBase>
                                  sphFn_i = sphericalFunction
                                              .find(std::make_pair(Znum, iProj))
                                              ->second;
                                const int lQuantumNo_i =
                                  sphFn_i->getQuantumNumberl();
                                for (int mQuantumNumber_i = -lQuantumNo_i;
                                     mQuantumNumber_i <= lQuantumNo_i;
                                     mQuantumNumber_i++)
                                  {
                                    sphericalHarmonicUtils::
                                      getSphericalHarmonicVal(
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
                                          sphFn_j =
                                            sphericalFunction
                                              .find(std::make_pair(Znum, jProj))
                                              ->second;
                                        const int lQuantumNo_j =
                                          sphFn_j->getQuantumNumberl();
                                        for (int mQuantumNumber_j =
                                               -lQuantumNo_j;
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
                                            SphericalHarmonics
                                              [projIndexI * numberOfProjectors +
                                               projIndexJ] = Yi * Yj;
                                            DijYij[projIndexI *
                                                     numberOfProjectors +
                                                   projIndexJ] =
                                              Yi * Yj *
                                              Dij[projIndexI *
                                                    numberOfProjectors +
                                                  projIndexJ];

                                            DijYij[projIndexI *
                                                     numberOfProjectors +
                                                   projIndexJ + shift] =
                                              Yi * Yj *
                                              Dij[projIndexI *
                                                    numberOfProjectors +
                                                  projIndexJ + shift];

                                            GradThetaSphericalHarmonics
                                              [projIndexI * numberOfProjectors +
                                               projIndexJ] = Yi * gradYj[0];
                                            double temp =
                                              std::abs(std::sin(
                                                quad_points[qpoint][0])) <=
                                                  1E-8 ?
                                                0.0 :
                                                Yi * gradYj[1] /
                                                  std::sin(
                                                    quad_points[qpoint][0]);
                                            GradPhiSphericalHarmonics
                                              [projIndexI * numberOfProjectors +
                                               projIndexJ] = temp;

                                            DijGradThetaYij
                                              [projIndexI * numberOfProjectors +
                                               projIndexJ] =
                                                Dij[projIndexI *
                                                      numberOfProjectors +
                                                    projIndexJ] *
                                                Yi * gradYj[0];
                                            DijGradPhiYij[projIndexI *
                                                            numberOfProjectors +
                                                          projIndexJ] =
                                              Dij[projIndexI *
                                                    numberOfProjectors +
                                                  projIndexJ] *
                                              temp;

                                            DijGradThetaYij
                                              [projIndexI * numberOfProjectors +
                                               projIndexJ + shift] =
                                                Dij[projIndexI *
                                                      numberOfProjectors +
                                                    projIndexJ + shift] *
                                                Yi * gradYj[0];
                                            DijGradPhiYij[projIndexI *
                                                            numberOfProjectors +
                                                          projIndexJ + shift] =
                                              Dij[projIndexI *
                                                    numberOfProjectors +
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
                            double            time1        = MPI_Wtime();
                            for (dftfe::Int iComp = 0; iComp < 2; iComp++)
                              {
                                std::vector<double> atomDensityAllelectron =
                                  d_atomTypeCoreFlagMap[Znum] ?
                                    d_atomCoreDensityAE[Znum] :
                                    std::vector<double>(numberofValues, 0.0);
                                std::vector<double> atomDensitySmooth =
                                  d_atomTypeCoreFlagMap[Znum] ?
                                    d_atomCoreDensityPS[Znum] :
                                    std::vector<double>(numberofValues, 0.0);

                                std::vector<double>
                                  atomDensityGradientAllElectron_0 =
                                    d_atomTypeCoreFlagMap[Znum] ?
                                      d_gradCoreAE[Znum] :
                                      std::vector<double>(numberofValues, 0.0);
                                std::vector<double>
                                  atomDensityGradientSmooth_0 =
                                    d_atomTypeCoreFlagMap[Znum] ?
                                      d_gradCorePS[Znum] :
                                      std::vector<double>(numberofValues, 0.0);

                                std::vector<double>
                                  atomDensityGradientAllElectron_1 =
                                    std::vector<double>(numberofValues, 0.0);
                                std::vector<double>
                                  atomDensityGradientSmooth_1 =
                                    std::vector<double>(numberofValues, 0.0);
                                std::vector<double>
                                  atomDensityGradientAllElectron_2 =
                                    std::vector<double>(numberofValues, 0.0);
                                std::vector<double>
                                  atomDensityGradientSmooth_2 =
                                    std::vector<double>(numberofValues, 0.0);

                                d_BLASWrapperHostPtr->xgemm(
                                  transA,
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
                                d_BLASWrapperHostPtr->xgemm(
                                  transA,
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
                                timeDensityCompute += (MPI_Wtime() - time1);
                                // component 0
                                double time2 = MPI_Wtime();
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
                                  &atomDensityGradientAllElectron_0[0],
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
                                  &atomDensityGradientSmooth_0[0], // to be
                                                                   // changed
                                  inc);

                                // component 1
                                d_BLASWrapperHostPtr->xgemm(
                                  transA,
                                  transB,
                                  inc,
                                  numberofValues,
                                  numberOfProjectorsSq,
                                  &AlphaGradRho,
                                  &DijGradThetaYij[iComp *
                                                   shift], // to be changed
                                  inc,
                                  &productOfAEpartialWfcValue[0],
                                  numberOfProjectorsSq,
                                  &BetaGradRho,
                                  &atomDensityGradientAllElectron_1[0],
                                  inc);
                                d_BLASWrapperHostPtr->xgemm(
                                  transA,
                                  transB,
                                  inc,
                                  numberofValues,
                                  numberOfProjectorsSq,
                                  &AlphaGradRho,
                                  &DijGradThetaYij[iComp *
                                                   shift], // to be changed
                                  inc,
                                  &productOfPSpartialWfcValue[0], // to be
                                                                  // changed
                                  numberOfProjectorsSq,
                                  &BetaGradRho,
                                  &atomDensityGradientSmooth_1[0], // to be
                                                                   // changed
                                  inc);

                                // component 2
                                d_BLASWrapperHostPtr->xgemm(
                                  transA,
                                  transB,
                                  inc,
                                  numberofValues,
                                  numberOfProjectorsSq,
                                  &AlphaGradRho,
                                  &DijGradPhiYij[iComp *
                                                 shift], // to be changed
                                  inc,
                                  &productOfAEpartialWfcValue[0], // to be
                                                                  // changed
                                  numberOfProjectorsSq,
                                  &BetaGradRho,
                                  &atomDensityGradientAllElectron_2
                                    [0], // to be
                                         // changed
                                  inc);
                                d_BLASWrapperHostPtr->xgemm(
                                  transA,
                                  transB,
                                  inc,
                                  numberofValues,
                                  numberOfProjectorsSq,
                                  &AlphaGradRho,
                                  &DijGradPhiYij[iComp *
                                                 shift], // to be changed
                                  inc,
                                  &productOfPSpartialWfcValue[0], // to be
                                                                  // changed
                                  numberOfProjectorsSq,
                                  &BetaGradRho,
                                  &atomDensityGradientSmooth_2[0], // to be
                                                                   // changed
                                  inc);
                                timegradRhoCompute += (MPI_Wtime() - time2);

                                for (int iRad = 0; iRad < numberofValues;
                                     iRad++)
                                  {
                                    // LDA constribution

                                    PSdensityValsForXC[iRad +
                                                       numberofValues * iComp] =
                                      atomDensitySmooth[iRad];
                                    AEdensityValsForXC[iRad +
                                                       numberofValues * iComp] =
                                      atomDensityAllelectron[iRad];

                                    // GGA contribution
                                    PSdensityGradForXC[3 * iRad + 0 +
                                                       3 * iComp *
                                                         numberofValues] =
                                      atomDensityGradientSmooth_0[iRad];
                                    PSdensityGradForXC[3 * iRad + 1 +
                                                       3 * iComp *
                                                         numberofValues] =
                                      atomDensityGradientSmooth_1[iRad];
                                    PSdensityGradForXC[3 * iRad + 2 +
                                                       3 * iComp *
                                                         numberofValues] =
                                      atomDensityGradientSmooth_2[iRad];


                                    AEdensityGradForXC[3 * iRad + 0 +
                                                       3 * iComp *
                                                         numberofValues] =
                                      atomDensityGradientAllElectron_0[iRad];
                                    AEdensityGradForXC[3 * iRad + 1 +
                                                       3 * iComp *
                                                         numberofValues] =
                                      atomDensityGradientAllElectron_1[iRad];
                                    AEdensityGradForXC[3 * iRad + 2 +
                                                       3 * iComp *
                                                         numberofValues] =
                                      atomDensityGradientAllElectron_2[iRad];
                                  }
                              }
                            double time3 = MPI_Wtime();
                            d_auxDensityMatrixXCAEPtr->projectDensityStart(
                              AEdensityProjectionInputs);
                            d_auxDensityMatrixXCAEPtr->projectDensityEnd(
                              d_mpiCommParent);
                            d_auxDensityMatrixXCPSPtr->projectDensityStart(
                              PSdensityProjectionInputs);
                            d_auxDensityMatrixXCPSPtr->projectDensityEnd(
                              d_mpiCommParent);


                            std::unordered_map<
                              xcRemainderOutputDataAttributes,
                              dftfe::utils::MemoryStorage<
                                double,
                                dftfe::utils::MemorySpace::HOST>>
                              xDataOutPS;
                            std::unordered_map<
                              xcRemainderOutputDataAttributes,
                              dftfe::utils::MemoryStorage<
                                double,
                                dftfe::utils::MemorySpace::HOST>>
                              cDataOutPS;


                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdexDensitySpinUpPS =
                                xDataOutPS[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinUp];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdexDensitySpinDownPS =
                                xDataOutPS[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinDown];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdecDensitySpinUpPS =
                                cDataOutPS[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinUp];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdecDensitySpinDownPS =
                                cDataOutPS[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinDown];

                            xDataOutPS
                              [xcRemainderOutputDataAttributes::pdeSigma] =
                                dftfe::utils::MemoryStorage<
                                  double,
                                  dftfe::utils::MemorySpace::HOST>();
                            cDataOutPS
                              [xcRemainderOutputDataAttributes::pdeSigma] =
                                dftfe::utils::MemoryStorage<
                                  double,
                                  dftfe::utils::MemorySpace::HOST>();

                            d_excManagerPtr->getExcSSDFunctionalObj()
                              ->computeRhoTauDependentXCData(
                                *d_auxDensityMatrixXCPSPtr,
                                radialIndex,
                                xDataOutPS,
                                cDataOutPS);
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST> &pdexSigmaPS =
                              xDataOutPS
                                [xcRemainderOutputDataAttributes::pdeSigma];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST> &pdecSigmaPS =
                              cDataOutPS
                                [xcRemainderOutputDataAttributes::pdeSigma];



                            std::unordered_map<
                              xcRemainderOutputDataAttributes,
                              dftfe::utils::MemoryStorage<
                                double,
                                dftfe::utils::MemorySpace::HOST>>
                              xDataOutAE;
                            std::unordered_map<
                              xcRemainderOutputDataAttributes,
                              dftfe::utils::MemoryStorage<
                                double,
                                dftfe::utils::MemorySpace::HOST>>
                              cDataOutAE;
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdexDensitySpinUpAE =
                                xDataOutAE[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinUp];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdexDensitySpinDownAE =
                                xDataOutAE[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinDown];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdecDensitySpinUpAE =
                                cDataOutAE[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinUp];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdecDensitySpinDownAE =
                                cDataOutAE[xcRemainderOutputDataAttributes::
                                             pdeDensitySpinDown];
                            xDataOutAE
                              [xcRemainderOutputDataAttributes::pdeSigma] =
                                dftfe::utils::MemoryStorage<
                                  double,
                                  dftfe::utils::MemorySpace::HOST>();
                            cDataOutAE
                              [xcRemainderOutputDataAttributes::pdeSigma] =
                                dftfe::utils::MemoryStorage<
                                  double,
                                  dftfe::utils::MemorySpace::HOST>();
                            const dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdexDensitySpinIndexAE =
                                spinIndex == 0 ? pdexDensitySpinUpAE :
                                                 pdexDensitySpinDownAE;
                            const dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdecDensitySpinIndexAE =
                                spinIndex == 0 ? pdecDensitySpinUpAE :
                                                 pdecDensitySpinDownAE;

                            const dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdexDensitySpinIndexPS =
                                spinIndex == 0 ? pdexDensitySpinUpPS :
                                                 pdexDensitySpinDownPS;
                            const dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &pdecDensitySpinIndexPS =
                                spinIndex == 0 ? pdecDensitySpinUpPS :
                                                 pdecDensitySpinDownPS;

                            d_excManagerPtr->getExcSSDFunctionalObj()
                              ->computeRhoTauDependentXCData(
                                *d_auxDensityMatrixXCAEPtr,
                                radialIndex,
                                xDataOutAE,
                                cDataOutAE);
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST> &pdexSigmaAE =
                              xDataOutAE
                                [xcRemainderOutputDataAttributes::pdeSigma];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST> &pdecSigmaAE =
                              cDataOutAE
                                [xcRemainderOutputDataAttributes::pdeSigma];

                            timeXCManager += (MPI_Wtime() - time3);

                            double time4 = MPI_Wtime();

                            for (int i = 0; i < numberOfProjectors; i++)
                              {
                                // Proj J

                                for (int j = 0; j <= i; j++)
                                  {
                                    const dftfe::uInt one = 1;
                                    for (int rpoint = 0; rpoint < RmaxIndex + 1;
                                         rpoint++)
                                      {
                                        dftfe::uInt index =
                                          rpoint * numberOfProjectorsSq +
                                          i * numberOfProjectors + j;
                                        double Val1 =
                                          productOfAEpartialWfc[index] *
                                          (pdexDensitySpinIndexAE[rpoint] +
                                           pdecDensitySpinIndexAE[rpoint]);
                                        double Val2 =
                                          productOfPSpartialWfc[index] *
                                          (pdexDensitySpinIndexPS[rpoint] +
                                           pdecDensitySpinIndexPS[rpoint]);
                                        integralValue[rpoint] =
                                          std::fabs(RadialMesh[rpoint]) > 1E-8 ?
                                            rab[rpoint] * (Val1 - Val2) *
                                              pow(RadialMesh[rpoint], 2) :
                                            0.0;
                                      }
                                    d_BLASWrapperHostPtr->xdot(
                                      RmaxIndex + 1,
                                      &SimpsonWeights[0],
                                      one,
                                      &integralValue[0],
                                      one,
                                      &RadialIntegralLDA);
                                    // RadialIntegralLDA +=
                                    //   SimpsonResidual(0,
                                    //                   RmaxIndex + 1,
                                    //                   integralValue);

                                    Delta_ExcijDensity[i * numberOfProjectors +
                                                       j] +=
                                      RadialIntegralLDA * quadwt * 4.0 * M_PI *
                                      SphericalHarmonics[i *
                                                           numberOfProjectors +
                                                         j];
                                  } // Proj J
                              }     // Proj I


                            std::unordered_map<
                              DensityDescriptorDataAttributes,
                              dftfe::utils::MemoryStorage<
                                double,
                                dftfe::utils::MemorySpace::HOST>>
                              densityDataAE, densityDataPS;
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &gradDensitySpinUpAE =
                                densityDataAE[DensityDescriptorDataAttributes::
                                                gradValuesSpinUp];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &gradDensitySpinDownAE =
                                densityDataAE[DensityDescriptorDataAttributes::
                                                gradValuesSpinDown];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &gradDensitySpinUpPS =
                                densityDataPS[DensityDescriptorDataAttributes::
                                                gradValuesSpinUp];
                            dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &gradDensitySpinDownPS =
                                densityDataPS[DensityDescriptorDataAttributes::
                                                gradValuesSpinDown];

                            d_auxDensityMatrixXCAEPtr->applyLocalOperations(
                              radialIndex, densityDataAE);
                            d_auxDensityMatrixXCPSPtr->applyLocalOperations(
                              radialIndex, densityDataPS);

                            const dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &gradDensityXCSpinIndexAE =
                                spinIndex == 0 ? gradDensitySpinUpAE :
                                                 gradDensitySpinDownAE;
                            const dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &gradDensityXCOtherSpinIndexAE =
                                spinIndex == 0 ? gradDensitySpinDownAE :
                                                 gradDensitySpinUpAE;

                            const dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &gradDensityXCSpinIndexPS =
                                spinIndex == 0 ? gradDensitySpinUpPS :
                                                 gradDensitySpinDownPS;
                            const dftfe::utils::MemoryStorage<
                              double,
                              dftfe::utils::MemorySpace::HOST>
                              &gradDensityXCOtherSpinIndexPS =
                                spinIndex == 0 ? gradDensitySpinDownPS :
                                                 gradDensitySpinUpPS;

                            for (int i = 0; i < numberOfProjectors; i++)
                              {
                                // Proj J
                                for (int j = 0; j < numberOfProjectors; j++)
                                  {
                                    const dftfe::uInt one = 1;
                                    dftfe::uInt       indexij =
                                      i * numberOfProjectors + j;
                                    dftfe::uInt indexji =
                                      j * numberOfProjectors + i;
                                    for (dftfe::uInt rpoint = 0;
                                         rpoint < RmaxIndex + 1;
                                         rpoint++)
                                      {
                                        dftfe::uInt indexIJ =
                                          rpoint * numberOfProjectorsSq +
                                          i * numberOfProjectors + j;
                                        dftfe::uInt indexJI =
                                          rpoint * numberOfProjectorsSq +
                                          j * numberOfProjectors + i;
                                        double termAE =
                                          (pdexSigmaAE[3 * rpoint +
                                                       2 * spinIndex] +
                                           pdecSigmaAE[3 * rpoint +
                                                       2 * spinIndex]);
                                        double termOffAE =
                                          (pdexSigmaAE[3 * rpoint + 1] +
                                           pdecSigmaAE[3 * rpoint + 1]);
                                        double termPS =
                                          (pdexSigmaPS[3 * rpoint +
                                                       2 * spinIndex] +
                                           pdecSigmaPS[3 * rpoint +
                                                       2 * spinIndex]);
                                        double termOffPS =
                                          (pdexSigmaPS[3 * rpoint + 1] +
                                           pdecSigmaPS[3 * rpoint + 1]);
                                        double Val1 = 0.0;
                                        double Val2 = 0.0;
                                        Val1 +=
                                          (2 * termAE *
                                             gradDensityXCSpinIndexAE[3 *
                                                                        rpoint +
                                                                      0] +
                                           termOffAE *
                                             gradDensityXCOtherSpinIndexAE
                                               [3 * rpoint + 0]) *
                                          (productOfAEpartialWfcDer[indexIJ] *
                                             SphericalHarmonics[indexij] +
                                           productOfAEpartialWfcDer[indexJI] *
                                             SphericalHarmonics[indexji]);
                                        Val1 +=
                                          (2 * termAE *
                                             gradDensityXCSpinIndexAE[3 *
                                                                        rpoint +
                                                                      1] +
                                           termOffAE *
                                             gradDensityXCOtherSpinIndexAE
                                               [3 * rpoint + 1]) *
                                          (productOfAEpartialWfcValue[indexIJ] *
                                             GradThetaSphericalHarmonics
                                               [indexij] +
                                           productOfAEpartialWfcValue[indexJI] *
                                             GradThetaSphericalHarmonics
                                               [indexji]);
                                        Val1 +=
                                          (2 * termAE *
                                             gradDensityXCSpinIndexAE[3 *
                                                                        rpoint +
                                                                      2] +
                                           termOffAE *
                                             gradDensityXCOtherSpinIndexAE
                                               [3 * rpoint + 2]) *
                                          (productOfAEpartialWfcValue[indexIJ] *
                                             GradPhiSphericalHarmonics
                                               [indexij] +
                                           productOfAEpartialWfcValue[indexJI] *
                                             GradPhiSphericalHarmonics
                                               [indexji]);


                                        Val2 +=
                                          (2 * termPS *
                                             gradDensityXCSpinIndexPS[3 *
                                                                        rpoint +
                                                                      0] +
                                           termOffPS *
                                             gradDensityXCOtherSpinIndexPS
                                               [3 * rpoint + 0]) *
                                          (productOfPSpartialWfcDer[indexIJ] *
                                             SphericalHarmonics[indexij] +
                                           productOfPSpartialWfcDer[indexJI] *
                                             SphericalHarmonics[indexji]);
                                        Val2 +=
                                          (2 * termPS *
                                             gradDensityXCSpinIndexPS[3 *
                                                                        rpoint +
                                                                      1] +
                                           termOffPS *
                                             gradDensityXCOtherSpinIndexPS
                                               [3 * rpoint + 1]) *
                                          (productOfPSpartialWfcValue[indexIJ] *
                                             GradThetaSphericalHarmonics
                                               [indexij] +
                                           productOfPSpartialWfcValue[indexJI] *
                                             GradThetaSphericalHarmonics
                                               [indexji]);
                                        Val2 +=
                                          (2 * termPS *
                                             gradDensityXCSpinIndexPS[3 *
                                                                        rpoint +
                                                                      2] +
                                           termOffPS *
                                             gradDensityXCOtherSpinIndexPS
                                               [3 * rpoint + 2]) *
                                          (productOfPSpartialWfcValue[indexIJ] *
                                             GradPhiSphericalHarmonics
                                               [indexij] +
                                           productOfPSpartialWfcValue[indexJI] *
                                             GradPhiSphericalHarmonics
                                               [indexji]);

                                        integralValue[rpoint] =
                                          std::fabs(RadialMesh[rpoint]) > 1E-8 ?
                                            rab[rpoint] * (Val1 - Val2) *
                                              pow(RadialMesh[rpoint], 2) :
                                            0.0;
                                      }
                                    d_BLASWrapperHostPtr->xdot(
                                      RmaxIndex + 1,
                                      &SimpsonWeights[0],
                                      one,
                                      &integralValue[0],
                                      one,
                                      &RadialIntegralGGA);

                                    // RadialIntegralGGA +=
                                    //   SimpsonResidual(0,
                                    //                   RmaxIndex + 1,
                                    //                   integralValue);

                                    Delta_ExcijSigma[i * numberOfProjectors +
                                                     j] +=
                                      RadialIntegralGGA * quadwt * 4.0 * M_PI;
                                  } // Proj J
                              }     // Proj I

                            timeSimpsonInt += MPI_Wtime() - time4;

                          } // qpoint

                      } // GGA case
                    for (int i = 0; i < numberOfProjectors; i++)
                      {
                        for (int j = 0; j <= i; j++)
                          {
                            if (!isGGA)
                              {
                                Delta_Excij[j * numberOfProjectors + i] =
                                  Delta_Excij[i * numberOfProjectors + j];
                              }
                            else
                              {
                                double temp =
                                  Delta_ExcijDensity[i * numberOfProjectors +
                                                     j] +
                                  Delta_ExcijSigma[i * numberOfProjectors + j];
                                Delta_Excij[j * numberOfProjectors + i] = temp;
                                Delta_Excij[i * numberOfProjectors + j] = temp;
                              } // else

                          } // jProj
                      }     // iProj

                    //
                    d_ExchangeCorrelationEnergyCorrectionTerm.find(atomId)
                      ->second = Delta_Excij;
                    if (d_verbosity >= 5)
                      {
                        pcout << " Delta XC for iAtom: " << atomId << std::endl;
                        for (int iProj = 0; iProj < numberOfProjectors; iProj++)
                          {
                            for (int jProj = 0; jProj < numberOfProjectors;
                                 jProj++)
                              pcout << Delta_Excij[iProj * numberOfProjectors +
                                                   jProj]
                                    << " ";
                            pcout << std::endl;
                          }
                      }

                  } // iAtomList

              } // use CPUs

            ZNumOld = Znum;
          } // it
        if (d_verbosity >= 5)
          {
            for (int i = 0; i < d_n_mpi_processes; i++)
              {
                if (i == d_this_mpi_process)
                  {
                    std::cout
                      << "MPI Process XC time splits: " << d_this_mpi_process
                      << " " << d_LocallyOwnedAtomId.size() << " "
                      << dataAllocationTime << " " << densityComputeTime << " "
                      << gradDensityTime << " " << libxcTime << " " << ldaTime
                      << " " << ggaTime << " " << std::endl;
                  }
              }
          }
      }
    // dftUtils::printCurrentMemoryUsage(d_mpiCommParent, "PAWClass delta XC
    // End");
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseKineticEnergyCorrection()
  {
    pcout << "PAWClass Init: Reading KE_ij correction terms from XML file..."
          << std::endl;
    const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction =
        d_atomicProjectorFnsContainer->getSphericalFunctions();
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        dftfe::uInt atomicNumber = *it;
        char        keFileName[256];
        strcpy(keFileName,
               (d_dftfeScratchFolderName + "/z" + std::to_string(atomicNumber) +
                "/" + "KineticEnergyij.dat")
                 .c_str());

        std::vector<double> KineticEnergyij;
        dftUtils::readFile(KineticEnergyij, keFileName);
        const dftfe::uInt numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber);
        const dftfe::uInt numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(atomicNumber);
        std::vector<double> Tij(numberOfProjectors * numberOfProjectors, 0.0);
        AssertThrow(
          KineticEnergyij.size() ==
            numberOfRadialProjectors * numberOfRadialProjectors,
          dealii::ExcMessage(
            "PAW::Initialization Kinetic Nergy correction term mismatch in number of entries"));
        dftfe::uInt projIndex_i = 0;
        for (dftfe::uInt alpha_i = 0; alpha_i < numberOfRadialProjectors;
             alpha_i++)
          {
            // pcout << "Alpha_i: " << alpha_i << std::endl;
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
              sphericalFunction.find(std::make_pair(atomicNumber, alpha_i))
                ->second;
            int lQuantumNo_i = sphFn_i->getQuantumNumberl();
            for (int mQuantumNo_i = -lQuantumNo_i; mQuantumNo_i <= lQuantumNo_i;
                 mQuantumNo_i++)
              {
                dftfe::uInt projIndex_j = 0;
                for (dftfe::uInt alpha_j = 0;
                     alpha_j < numberOfRadialProjectors;
                     alpha_j++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_j =
                      sphericalFunction
                        .find(std::make_pair(atomicNumber, alpha_j))
                        ->second;
                    int lQuantumNo_j = sphFn_j->getQuantumNumberl();
                    for (int mQuantumNo_j = -lQuantumNo_j;
                         mQuantumNo_j <= lQuantumNo_j;
                         mQuantumNo_j++)
                      {
                        if (lQuantumNo_i == lQuantumNo_j &&
                            mQuantumNo_i == mQuantumNo_j)
                          Tij[projIndex_i * numberOfProjectors + projIndex_j] =
                            KineticEnergyij[alpha_i * numberOfRadialProjectors +
                                            alpha_j];
                        projIndex_j++;
                      } // mQuantumNo_j
                  }     // alpha_j

                projIndex_i++;
              } // mQuantumNo_i
          }     // alpha_i
        d_KineticEnergyCorrectionTerm[*it] = Tij;
        if (d_verbosity >= 5)
          for (int i = 0; i < numberOfProjectors; i++)
            {
              for (int j = 0; j < numberOfProjectors; j++)
                pcout
                  << d_KineticEnergyCorrectionTerm[*it]
                                                  [i * numberOfProjectors + j]
                  << " ";
              pcout << std::endl;
            }
      } //*it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeRadialMultipoleData()
  {
    const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction =
        d_atomicProjectorFnsContainer->getSphericalFunctions();
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        dftfe::uInt       atomicNumber = *it;
        const dftfe::uInt numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(atomicNumber);
        const dftfe::uInt   lmaxAug = d_dftParamsPtr->noShapeFnsInPAW;
        std::vector<double> multipoleTable(lmaxAug * numberOfRadialProjectors *
                                             numberOfRadialProjectors,
                                           0.0);
        std::vector<double> aePhi        = d_radialWfcValAE[*it];
        std::vector<double> psPhi        = d_radialWfcValPS[*it];
        std::vector<double> radialMesh   = d_radialMesh[*it];
        std::vector<double> jacobianData = d_radialJacobianData[*it];
        dftfe::uInt         meshSize     = radialMesh.size();
        dftfe::uInt         rmaxAugIndex = d_RmaxAugIndex[*it];
        std::vector<double> aePartialWfcIntegralIJ(numberOfRadialProjectors *
                                                   numberOfRadialProjectors);
        for (dftfe::uInt L = 0; L < lmaxAug; L++)
          {
            for (dftfe::uInt alpha_i = 0; alpha_i < numberOfRadialProjectors;
                 alpha_i++)
              {
                std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
                  sphericalFunction.find(std::make_pair(atomicNumber, alpha_i))
                    ->second;
                int lQuantumNo_i = sphFn_i->getQuantumNumberl();
                for (dftfe::uInt alpha_j = 0;
                     alpha_j < numberOfRadialProjectors;
                     alpha_j++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_j =
                      sphericalFunction
                        .find(std::make_pair(atomicNumber, alpha_j))
                        ->second;
                    int    lQuantumNo_j = sphFn_j->getQuantumNumberl();
                    double Value        = 0.0;
                    if (L >= std::abs(lQuantumNo_i - lQuantumNo_j) &&
                        L <= (lQuantumNo_i + lQuantumNo_j))
                      {
                        Value = multipoleIntegrationGrid(aePhi.data() +
                                                           (alpha_i)*meshSize,
                                                         aePhi.data() +
                                                           (alpha_j)*meshSize,
                                                         radialMesh,
                                                         jacobianData,
                                                         L,
                                                         0,
                                                         rmaxAugIndex) -
                                multipoleIntegrationGrid(psPhi.data() +
                                                           (alpha_i)*meshSize,
                                                         psPhi.data() +
                                                           (alpha_j)*meshSize,
                                                         radialMesh,
                                                         jacobianData,
                                                         L,
                                                         0,
                                                         rmaxAugIndex);
                      }
                    multipoleTable[L * numberOfRadialProjectors *
                                     numberOfRadialProjectors +
                                   alpha_i * numberOfRadialProjectors +
                                   alpha_j] = Value;
                    if (L == 0)
                      {
                        aePartialWfcIntegralIJ[alpha_i *
                                                 numberOfRadialProjectors +
                                               alpha_j] =
                          multipoleIntegrationGrid(aePhi.data() +
                                                     (alpha_i)*meshSize,
                                                   aePhi.data() +
                                                     (alpha_j)*meshSize,
                                                   radialMesh,
                                                   jacobianData,
                                                   L,
                                                   0,
                                                   rmaxAugIndex);
                      }

                  } // alpha_j
              }     // alpha_i
          }         // L



        d_multipole[*it]              = multipoleTable;
        d_aePartialWfcIntegralIJ[*it] = aePartialWfcIntegralIJ;
      } //*it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeInverseOfMultipoleData()
  {
    pcout << "PAWClass: Computing inverse multipole data from XML file..."
          << std::endl;
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        dftfe::uInt         atomicNumber   = *it;
        std::vector<double> multipoleTable = d_multipole[atomicNumber];
        const dftfe::uInt   numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber);
        const dftfe::uInt numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(atomicNumber);
        const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        std::vector<double> Multipole(numberOfProjectors * numberOfProjectors,
                                      0.0);
        int                 projIndexI = 0;
        for (int iProj = 0; iProj < numberOfRadialProjectors; iProj++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
              sphericalFunction.find(std::make_pair(atomicNumber, iProj))
                ->second;
            int lQuantumNo_i = sphFn_i->getQuantumNumberl();
            for (int mQuantumNumber_i = -lQuantumNo_i;
                 mQuantumNumber_i <= lQuantumNo_i;
                 mQuantumNumber_i++)
              {
                int projIndexJ = 0;
                for (int jProj = 0; jProj < numberOfRadialProjectors; jProj++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_j =
                      sphericalFunction
                        .find(std::make_pair(atomicNumber, jProj))
                        ->second;
                    int lQuantumNo_j = sphFn_j->getQuantumNumberl();
                    for (int mQuantumNumber_j = -lQuantumNo_j;
                         mQuantumNumber_j <= lQuantumNo_j;
                         mQuantumNumber_j++)
                      {
                        Multipole[projIndexI * numberOfProjectors +
                                  projIndexJ] =
                          gaunt(lQuantumNo_i,
                                lQuantumNo_j,
                                0,
                                mQuantumNumber_i,
                                mQuantumNumber_j,
                                0) *
                          multipoleTable[iProj * numberOfRadialProjectors +
                                         jProj] *
                          sqrt(4 * M_PI);
                        projIndexJ++;
                      } // mQuantumNumber_j

                  } // jProj
                projIndexI++;
              } // mQuantumNumber_i



          } // iProj
        const char          uplo = 'L';
        const int           N    = numberOfProjectors;
        std::vector<double> A    = Multipole;


        dftfe::linearAlgebraOperations::inverse(&A[0], N);
        d_multipoleInverse[atomicNumber] = A;


      } //*it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::createAtomTypesList(
    const std::vector<std::vector<double>> &atomLocations)
  {
    d_nProjPerTask    = 0;
    d_nProjSqTotal    = 0;
    d_totalProjectors = 0;
    pcout << "Creating Atom Type List " << std::endl;
    d_LocallyOwnedAtomId.clear();
    d_totalProjectorStartIndex.clear();
    d_projectorStartIndex.clear();
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        unsigned                 atomType = *it;
        std::vector<dftfe::uInt> atomLocation;
        for (int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
          {
            if (atomLocations[iAtom][0] == atomType)
              atomLocation.push_back(iAtom);
          }
        d_atomTypesList[atomType] = atomLocation;
      }
    if (d_n_mpi_processes > atomLocations.size())
      {
        if (d_this_mpi_process >= atomLocations.size())
          {
          }
        else
          d_LocallyOwnedAtomId.push_back(d_this_mpi_process);
      }
    else
      {
        int no_atoms       = atomLocations.size() / d_n_mpi_processes;
        int remainderAtoms = atomLocations.size() % d_n_mpi_processes;
        for (int i = 0; i < no_atoms; i++)
          {
            d_LocallyOwnedAtomId.push_back(d_this_mpi_process * no_atoms + i);
          }
        if (d_this_mpi_process < remainderAtoms)
          d_LocallyOwnedAtomId.push_back(d_n_mpi_processes * no_atoms +
                                         d_this_mpi_process);
      }
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();

    for (dftfe::uInt i = 0; i < d_LocallyOwnedAtomId.size(); i++)
      {
        dftfe::uInt atomId = d_LocallyOwnedAtomId[i];
        dftfe::uInt Znum   = atomicNumber[atomId];
        d_nProjPerTask += d_atomicProjectorFnsContainer
                            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
      }

    for (dftfe::uInt i = 0; i < atomicNumber.size(); i++)
      {
        dftfe::uInt Znum = atomicNumber[i];
        dftfe::uInt numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        d_totalProjectorStartIndex.push_back(d_totalProjectors);
        d_projectorStartIndex.push_back(d_nProjSqTotal);
        d_nProjSqTotal += (numberOfProjectors * (numberOfProjectors + 1)) / 2;
        d_totalProjectors += numberOfProjectors;
      }
    d_LocallyOwnedAtomIdMapWithAtomType.clear();

    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        unsigned                 atomType = *it;
        std::vector<dftfe::uInt> atomLocation;
        for (int iAtom = 0; iAtom < d_LocallyOwnedAtomId.size(); iAtom++)
          {
            dftfe::uInt atomId = d_LocallyOwnedAtomId[iAtom];
            if (atomLocations[atomId][0] == atomType)
              {
                atomLocation.push_back(atomId);
                // std::cout << "DEBUG: " << atomType << " "
                //       << atomLocations[atomId][0] << " " << atomId <<"
                //       "<<d_this_mpi_process<< std::endl;
              }
          }
        if (atomLocation.size() > 0)
          d_LocallyOwnedAtomIdMapWithAtomType[atomType] = atomLocation;
      }
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::initialiseDataonRadialMesh()
  {
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        dftfe::uInt         Znum           = *it;
        std::vector<double> radialMesh     = d_radialMesh[*it];
        std::vector<double> jacobianData   = d_radialJacobianData[*it];
        const dftfe::uInt   rmaxAugIndex   = d_RmaxAugIndex[*it];
        const dftfe::uInt   radialMeshSize = radialMesh.size();
        const dftfe::uInt   numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const dftfe::uInt numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        std::vector<double> productOfAEpartialWfc(
          radialMeshSize * numberOfProjectors * numberOfProjectors, 0.0);
        std::vector<double> productOfPSpartialWfc(
          radialMeshSize * numberOfProjectors * numberOfProjectors, 0.0);

        // Core densit changes
        for (int rPoint = 0; rPoint < radialMeshSize; rPoint++)
          {
            double r                = radialMesh[rPoint];
            int    projectorIndex_i = 0;
            for (int alpha_i = 0; alpha_i < numberOfRadialProjectors; alpha_i++)
              {
                std::shared_ptr<AtomCenteredSphericalFunctionBase> AEsphFn_i =
                  d_atomicAEPartialWaveFnsMap
                    .find(std::make_pair(Znum, alpha_i))
                    ->second;
                std::shared_ptr<AtomCenteredSphericalFunctionBase> PSsphFn_i =
                  d_atomicPSPartialWaveFnsMap
                    .find(std::make_pair(Znum, alpha_i))
                    ->second;
                int    lQuantumNo_i  = AEsphFn_i->getQuantumNumberl();
                double radialValAE_i = AEsphFn_i->getRadialValue(r);
                double radialValPS_i = PSsphFn_i->getRadialValue(r);
                for (int mQuantumNo_i = -lQuantumNo_i;
                     mQuantumNo_i <= lQuantumNo_i;
                     mQuantumNo_i++)
                  {
                    int projectorIndex_j = 0;
                    for (int alpha_j = 0; alpha_j < numberOfRadialProjectors;
                         alpha_j++)
                      {
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>
                          AEsphFn_j = d_atomicAEPartialWaveFnsMap
                                        .find(std::make_pair(Znum, alpha_j))
                                        ->second;
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>
                          PSsphFn_j = d_atomicPSPartialWaveFnsMap
                                        .find(std::make_pair(Znum, alpha_j))
                                        ->second;
                        double radialValAE_j = AEsphFn_j->getRadialValue(r);
                        double radialValPS_j = PSsphFn_j->getRadialValue(r);
                        int    lQuantumNo_j  = AEsphFn_j->getQuantumNumberl();
                        for (int mQuantumNo_j = -lQuantumNo_j;
                             mQuantumNo_j <= lQuantumNo_j;
                             mQuantumNo_j++)
                          {
                            dftfe::uInt index =
                              rPoint * numberOfProjectors * numberOfProjectors +
                              projectorIndex_i * numberOfProjectors +
                              projectorIndex_j;
                            productOfAEpartialWfc[index] =
                              radialValAE_j * radialValAE_i;
                            productOfPSpartialWfc[index] =
                              radialValPS_i * radialValPS_j;
                            projectorIndex_j++;

                          } // mQuantumNo_j
                      }     // alpha_j

                    projectorIndex_i++;
                  } // mQuantumNo_i



              } // alpha_i

          } // rPoint
        d_productOfAEpartialWfc[*it] = productOfAEpartialWfc;
        d_productOfPSpartialWfc[*it] = productOfPSpartialWfc;
        for (int rPoint = 0; rPoint < radialMeshSize; rPoint++)
          {
            d_atomCoreDensityAE[*it][rPoint] /= sqrt(4 * M_PI);
            d_atomCoreDensityPS[*it][rPoint] /= sqrt(4 * M_PI);
          }
        const bool isGGA =
          (d_excManagerPtr->getExcSSDFunctionalObj()
             ->getDensityBasedFamilyType() == densityFamilyType::GGA);
        if (isGGA)
          {
            dftfe::uInt         npj_2 = pow(numberOfProjectors, 2);
            std::vector<double> productValDerPS(npj_2 * radialMeshSize, 0.0);
            std::vector<double> productValDerAE(npj_2 * radialMeshSize, 0.0);
            std::vector<double> derAECore(radialMeshSize, 0.0);
            std::vector<double> derPSCore(radialMeshSize, 0.0);
            std::vector<double> derAECoreSq(radialMeshSize, 0.0);
            std::vector<double> productValsPS(npj_2 * radialMeshSize, 0.0);
            std::vector<double> productValsAE(npj_2 * radialMeshSize, 0.0);
            std::vector<double> derCoreRhoAE = d_radialCoreDerAE[*it];
            std::vector<double> derCoreRhoPS = d_radialCoreDerPS[*it];
            std::vector<double> derWfcAE     = d_radialWfcDerAE[*it];
            std::vector<double> derWfcPS     = d_radialWfcDerPS[*it];
            std::vector<double> WfcAE        = d_radialWfcValAE[*it];
            std::vector<double> WfcPS        = d_radialWfcValPS[*it];

            // map of projectroIndex tot radialProjectorId
            std::vector<dftfe::uInt> projectorIndexRadialIndexMap(
              numberOfProjectors);
            dftfe::uInt projectorIndex = 0;
            for (dftfe::uInt alpha = 0; alpha < numberOfRadialProjectors;
                 alpha++)
              {
                std::shared_ptr<AtomCenteredSphericalFunctionBase> AEsphFn =
                  d_atomicProjectorFnsMap.find(std::make_pair(Znum, alpha))
                    ->second;
                int lQuantumNo = AEsphFn->getQuantumNumberl();
                for (int mQuantumNo = -lQuantumNo; mQuantumNo <= lQuantumNo;
                     mQuantumNo++)
                  {
                    projectorIndexRadialIndexMap[projectorIndex] = alpha;
                    projectorIndex++;
                  }
              }

            for (int rpoint = 0; rpoint < radialMeshSize; rpoint++)
              {
                double r = radialMesh[rpoint];
                // CoreDensity Changes Pending
                if (d_atomTypeCoreFlagMap[*it])
                  {
                    derAECore[rpoint] =
                      1 / sqrt(4 * M_PI) * derCoreRhoAE[rpoint];
                    derPSCore[rpoint] =
                      1 / sqrt(4 * M_PI) * derCoreRhoPS[rpoint];
                    derAECoreSq[rpoint] = derAECore[rpoint] * derAECore[rpoint];
                  }
                for (int projectorIndex_i = 0;
                     projectorIndex_i < numberOfProjectors;
                     projectorIndex_i++)
                  {
                    dftfe::uInt alpha_i =
                      projectorIndexRadialIndexMap[projectorIndex_i];
                    for (int projectorIndex_j = 0;
                         projectorIndex_j < numberOfProjectors;
                         projectorIndex_j++)
                      {
                        dftfe::uInt alpha_j =
                          projectorIndexRadialIndexMap[projectorIndex_j];
                        dftfe::uInt index =
                          rpoint * numberOfProjectors * numberOfProjectors +
                          projectorIndex_i * numberOfProjectors +
                          projectorIndex_j;

                        double DerAEij =
                          WfcAE[alpha_i * radialMeshSize + rpoint] *
                          derWfcAE[alpha_j * radialMeshSize + rpoint];
                        double DerPSij =
                          WfcPS[alpha_i * radialMeshSize + rpoint] *
                          derWfcPS[alpha_j * radialMeshSize + rpoint];
                        productValDerAE[index] = DerAEij;
                        productValDerPS[index] = DerPSij;

                        double ValAEij =
                          WfcAE[alpha_i * radialMeshSize + rpoint] *
                          WfcAE[alpha_j * radialMeshSize + rpoint];
                        double ValPSij =
                          WfcPS[alpha_i * radialMeshSize + rpoint] *
                          WfcPS[alpha_j * radialMeshSize + rpoint];
                        productValsAE[index] = r <= 1E-8 ? 0.0 : ValAEij / r;
                        productValsPS[index] = r <= 1E-8 ? 0.0 : ValPSij / r;

                      } // projectorIndex_j
                  }     // projectorIndex_i



              } // rPoint
            d_gradCoreAE[*it]                 = derAECore;
            d_gradCorePS[*it]                 = derPSCore;
            d_gradCoreSqAE[*it]               = derAECoreSq;
            d_productOfAEpartialWfcDer[*it]   = productValDerAE;
            d_productOfPSpartialWfcDer[*it]   = productValDerPS;
            d_productOfAEpartialWfcValue[*it] = productValsAE;
            d_productOfPSpartialWfcValue[*it] = productValsPS;

          } // isGGA

      } //*it
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::map<dftfe::uInt, dftfe::uInt> &
  pawClass<ValueType, memorySpace>::getPSPAtomIdToGlobalIdMap()
  {
    return d_atomIdPseudopotentialInterestToGlobalId;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeDijFromPSIinitialGuess(
    const dftfe::utils::MemoryStorage<ValueType, memorySpace> *X,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<ValueType, double, memorySpace>>
                                           &basisOperationsPtr,
    const std::vector<std::vector<double>> &atomLocations,
    const dftfe::uInt                       numberOfElectrons,
    const dftfe::uInt                       totalNumWaveFunctions,
    const dftfe::uInt                       quadratureIndex,
    const std::vector<double>              &kPointWeights,
    const MPI_Comm                         &interpoolcomm,
    const MPI_Comm                         &interBandGroupComm)
  {
    MPI_Barrier(d_mpiCommParent);
    const dftfe::uInt numKPoints             = kPointWeights.size();
    const dftfe::uInt numLocalDofs           = basisOperationsPtr->nOwnedDofs();
    const dftfe::uInt totalLocallyOwnedCells = basisOperationsPtr->nCells();
    const dftfe::uInt numNodesPerElement = basisOperationsPtr->nDofsPerCell();
    // band group parallelization data structures
    const dftfe::uInt numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
    const dftfe::uInt bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
                                               totalNumWaveFunctions,
                                               bandGroupLowHighPlusOneIndices);

    const dftfe::uInt BVec = std::min(d_dftParamsPtr->chebyWfcBlockSize,
                                      bandGroupLowHighPlusOneIndices[1]);

    dftfe::utils::MemoryStorage<ValueType, memorySpace> tempCellNodalData;

    const double spinPolarizedFactor =
      (d_dftParamsPtr->spinPolarized == 1) ? 1.0 : 2.0;
    const dftfe::uInt numSpinComponents =
      (d_dftParamsPtr->spinPolarized == 1) ? 2 : 1;
    std::vector<dftfe::uInt> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    bool                                       flag = false;
    std::map<dftfe::uInt, std::vector<double>> occupancyMatrixMap;
    if (d_dftParamsPtr->startingWFCType == "RANDOM")
      {
        // Read file..

        {
          for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
               it != d_atomTypes.end();
               ++it)
            {
              dftfe::uInt       Znum = *it;
              const dftfe::uInt numberOfRadialProjectors =
                d_atomicProjectorFnsContainer
                  ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
              const dftfe::uInt numberOfProjectors =
                d_atomicProjectorFnsContainer
                  ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
              std::vector<double>              occupancyEntries;
              std::vector<std::vector<double>> occupancyVector(0);
              char                             inicialOccupancyDataFile[256];
              strcpy(inicialOccupancyDataFile,
                     (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                      "/initOccupancyFile.dat")
                       .c_str());
              dftUtils::readFile(1, occupancyVector, inicialOccupancyDataFile);
              double totalOccupancy      = 0.0;
              double numValenceElectrons = d_valenceElectrons[Znum];
              for (int i = 0; i < occupancyVector.size(); i++)
                {
                  {
                    occupancyEntries.push_back(occupancyVector[i][0]);
                    totalOccupancy += occupancyVector[i][0];
                  }

                } // i
              pcout << "Znum: " << Znum << " has " << numValenceElectrons
                    << " and total initial occpuancy: " << totalOccupancy
                    << " from: " << occupancyEntries.size() << " orbitals!"
                    << " while total projectors is: " << numberOfProjectors
                    << " conditon check: "
                    << (std::fabs(totalOccupancy - numValenceElectrons) <=
                          1E-3 &&
                        occupancyEntries.size() == numberOfProjectors)
                    << std::endl;
              if (std::fabs(totalOccupancy - numValenceElectrons) <= 1E-3 &&
                  occupancyEntries.size() == numberOfProjectors)
                {
                  flag                     = true;
                  occupancyMatrixMap[Znum] = occupancyEntries;
                }
              else
                flag = false;
            }
        }
        if (flag == false)
          pcout
            << "DFTFE-Warning: Initial WFC are RANDOM, however initial Dij is set based on initial PSI. Check INITAL FRACTIONAL OCCUPANCY LIST file "
            << std::endl;
      }

    if (flag)
      {
        for (dftfe::uInt atomId = 0; atomId < atomicNumber.size(); atomId++)
          {
            dftfe::uInt Znum = atomicNumber[atomId];
            dftfe::uInt numberOfProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            dftfe::uInt         startIndex = d_projectorStartIndex[atomId];
            std::vector<double> Dij(numberOfProjectors * numberOfProjectors,
                                    0.0);
            std::vector<double> occupancyEntries =
              occupancyMatrixMap.find(Znum)->second;
            dftfe::uInt index = 0;
            for (int i = 0; i < numberOfProjectors; i++)
              {
                Dij[i * numberOfProjectors + i] = occupancyEntries[i];
              }
            D_ij[0][TypeOfField::In][atomId] = Dij;
          }
      }

    else if (!flag)
      {
        const ValueType zero                    = 0;
        const ValueType scalarCoeffAlphaRho     = 1.0;
        const ValueType scalarCoeffBetaRho      = 1.0;
        const ValueType scalarCoeffAlphaGradRho = 1.0;
        const ValueType scalarCoeffBetaGradRho  = 1.0;

        const dftfe::uInt cellsBlockSize =
          memorySpace == dftfe::utils::MemorySpace::DEVICE ?
            basisOperationsPtr->nCells() :
            1;
        const dftfe::uInt numCellBlocks =
          totalLocallyOwnedCells / cellsBlockSize;
        MPI_Barrier(d_mpiCommParent);
        const dftfe::uInt remCellBlockSize =
          totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;
        d_BasisOperatorHostPtr->reinit(BVec, cellsBlockSize, quadratureIndex);
        const dftfe::uInt numQuadPoints =
          d_BasisOperatorHostPtr->nQuadsPerCell();
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          partialOccupVecHost(BVec, 0.0);
#if defined(DFTFE_WITH_DEVICE)
        dftfe::utils::MemoryStorage<double, memorySpace> partialOccupVec(
          partialOccupVecHost.size());
#else
        auto &partialOccupVec = partialOccupVecHost;
#endif

        dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
          *flattenedArrayBlock;
        dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
                    projectorKetTimesVector;
        dftfe::uInt previousSize = 0;
        bool        startFlag    = true;
        for (dftfe::uInt kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
          {
            dftfe::uInt numberOfRemainingElectrons = numberOfElectrons;
            for (dftfe::uInt spinIndex = 0; spinIndex < numSpinComponents;
                 ++spinIndex)
              {
                d_nonLocalOperator->initialiseOperatorActionOnX(kPoint);

                for (dftfe::uInt jvec = 0; jvec < totalNumWaveFunctions;
                     jvec += BVec)
                  {
                    const dftfe::uInt currentBlockSize =
                      std::min(BVec, totalNumWaveFunctions - jvec);
                    basisOperationsPtr->reinit(currentBlockSize,
                                               cellsBlockSize,
                                               quadratureIndex);
                    d_nonLocalOperator->initialiseFlattenedDataStructure(
                      currentBlockSize, projectorKetTimesVector);



                    flattenedArrayBlock = &(
                      basisOperationsPtr->getMultiVector(currentBlockSize, 0));
                    d_nonLocalOperator->initialiseFlattenedDataStructure(
                      currentBlockSize, projectorKetTimesVector);
                    if ((jvec + currentBlockSize) <=
                          bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId +
                                                         1] &&
                        (jvec + currentBlockSize) >
                          bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                      {
                        // compute occupancy Vector
                        for (dftfe::uInt iEigenVec = 0;
                             iEigenVec < currentBlockSize;
                             ++iEigenVec)
                          {
                            double OccupancyFactor = 0.0;
                            if (numberOfRemainingElectrons == 1)
                              {
                                OccupancyFactor            = 0.5;
                                numberOfRemainingElectrons = 0;
                              }
                            else if (numberOfRemainingElectrons > 1)
                              {
                                OccupancyFactor = 1.0;
                                numberOfRemainingElectrons -=
                                  spinPolarizedFactor;
                              }



                            *(partialOccupVecHost.begin() + iEigenVec) =
                              OccupancyFactor * kPointWeights[kPoint] *
                              spinPolarizedFactor;
                          }
#if defined(DFTFE_WITH_DEVICE)
                        partialOccupVec.copyFrom(partialOccupVecHost);
#endif

                        if (memorySpace == dftfe::utils::MemorySpace::HOST)
                          for (dftfe::uInt iNode = 0; iNode < numLocalDofs;
                               ++iNode)
                            std::memcpy(flattenedArrayBlock->data() +
                                          iNode * currentBlockSize,
                                        X->data() +
                                          numLocalDofs * totalNumWaveFunctions *
                                            (numSpinComponents * kPoint +
                                             spinIndex) +
                                          iNode * totalNumWaveFunctions + jvec,
                                        currentBlockSize * sizeof(ValueType));
#if defined(DFTFE_WITH_DEVICE)
                        else if (memorySpace ==
                                 dftfe::utils::MemorySpace::DEVICE)
                          d_BLASWrapperDevicePtr
                            ->stridedCopyToBlockConstantStride(
                              currentBlockSize,
                              totalNumWaveFunctions,
                              numLocalDofs,
                              jvec,
                              X->data() +
                                numLocalDofs * totalNumWaveFunctions *
                                  (numSpinComponents * kPoint + spinIndex),
                              flattenedArrayBlock->data());
#endif

                        flattenedArrayBlock->updateGhostValues();
                        basisOperationsPtr->distribute(*(flattenedArrayBlock));


                        for (int iblock = 0; iblock < (numCellBlocks + 1);
                             iblock++)
                          {
                            const dftfe::uInt currentCellsBlockSize =
                              (iblock == numCellBlocks) ? remCellBlockSize :
                                                          cellsBlockSize;
                            if (currentCellsBlockSize > 0)
                              {
                                const dftfe::uInt startingCellId =
                                  iblock * cellsBlockSize;
                                if (currentCellsBlockSize * currentBlockSize !=
                                    previousSize)
                                  {
                                    tempCellNodalData.resize(
                                      currentCellsBlockSize * currentBlockSize *
                                      numNodesPerElement);
                                    if constexpr (dftfe::utils::MemorySpace::
                                                    DEVICE == memorySpace)
                                      {
                                        d_nonLocalOperator
                                          ->initialiseCellWaveFunctionPointers(
                                            tempCellNodalData,
                                            currentCellsBlockSize);
                                      }

                                    previousSize =
                                      currentCellsBlockSize * currentBlockSize;
                                  }
                                if (
                                  d_nonLocalOperator
                                    ->getTotalNonLocalElementsInCurrentProcessor() >
                                  0)
                                  {
                                    basisOperationsPtr
                                      ->extractToCellNodalDataKernel(
                                        *(flattenedArrayBlock),
                                        tempCellNodalData.data(),
                                        std::pair<dftfe::uInt, dftfe::uInt>(
                                          startingCellId,
                                          startingCellId +
                                            currentCellsBlockSize));

                                    d_nonLocalOperator->applyCconjtransOnX(
                                      tempCellNodalData.data(),
                                      std::pair<dftfe::uInt, dftfe::uInt>(
                                        startingCellId,
                                        startingCellId +
                                          currentCellsBlockSize));
                                    // Call apply CconjTranspose
                                  }



                              } // non-trivial cell block check
                          }     // cells block loop

                        d_nonLocalOperator->applyAllReduceOnCconjtransX(
                          projectorKetTimesVector);
                        d_nonLocalOperator
                          ->copyBackFromDistributedVectorToLocalDataStructure(
                            projectorKetTimesVector, partialOccupVec);
                        computeDij(false,
                                   startFlag ? 0 : 1,
                                   currentBlockSize,
                                   spinIndex,
                                   kPoint);
                        // Call computeDij
                      } // if
                    startFlag = false;
                  } // jVec
              }     // spinIndex
          }         // kPoiintIndex


        MPI_Barrier(d_mpiCommParent);
        communicateDijAcrossAllProcessors(TypeOfField::In,
                                          0,
                                          interpoolcomm,
                                          interBandGroupComm);


        MPI_Barrier(d_mpiCommParent);
      }
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        pcout << "Initialising Magnetization value for Dij..... " << std::endl;
        std::vector<dftfe::uInt> atomicNumber =
          d_atomicProjectorFnsContainer->getAtomicNumbers();
        const std::vector<dftfe::uInt> ownedAtomIds =
          d_nonLocalOperator->getOwnedAtomIdsInCurrentProcessor();
        for (dftfe::uInt chargeId = 0; chargeId < atomicNumber.size();
             chargeId++)
          {
            dftfe::uInt       Znum = atomicNumber[chargeId];
            const dftfe::uInt numberOfProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            std::vector<double> occupancyEntries;
            double              rhoAtomFactor  = 0.0;
            double              magZAtomFactor = 0.0;
            if (atomLocations[chargeId].size() == 5)
              {
                rhoAtomFactor  = 1.0;
                magZAtomFactor = 0.0;
              }
            else if (atomLocations[chargeId].size() == 6)
              {
                rhoAtomFactor  = 1.0;
                magZAtomFactor = atomLocations[chargeId][5];
              }
            else if (atomLocations[chargeId].size() == 7)
              {
                rhoAtomFactor  = atomLocations[chargeId][6];
                magZAtomFactor = atomLocations[chargeId][5];
              }
            std::vector<double> Dij(numberOfProjectors * numberOfProjectors,
                                    0.0);
            std::vector<double> DijRho = D_ij[0][TypeOfField::In][chargeId];
            std::vector<double> DijMagZ(numberOfProjectors * numberOfProjectors,
                                        0.0);
            for (int i = 0; i < numberOfProjectors * numberOfProjectors; i++)
              {
                DijRho[i] *= rhoAtomFactor;
                DijMagZ[i] = magZAtomFactor * DijRho[i];
              }
            D_ij[1][TypeOfField::In][chargeId] = DijMagZ;
          }
      }
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const bool
  pawClass<ValueType, memorySpace>::hasSOC() const
  {
    return d_hasSOC;
  }
  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
