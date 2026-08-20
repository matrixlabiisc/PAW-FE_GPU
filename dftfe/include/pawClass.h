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
// @author  Kartick Ramakrishnan
//

#ifndef DFTFE_PAWCLASS_H
#define DFTFE_PAWCLASS_H

#include "vector"
#include "map"
#include "AtomCenteredSphericalFunctionValenceDensitySpline.h"
#include "AtomCenteredSphericalFunctionCoreDensitySpline.h"
#include "AtomCenteredSphericalFunctionLocalPotentialSpline.h"
#include "AtomCenteredSphericalFunctionProjectorSpline.h"
#include "AtomCenteredSphericalFunctionPAWProjectorSpline.h"
#include "AtomCenteredSphericalFunctionPAWProjectorSpline2.h"
#include "AtomCenteredSphericalFunctionZeroPotentialSpline.h"
#include "AtomCenteredSphericalFunctionContainer.h"
#include "AtomicCenteredNonLocalOperator.h"
#include "AtomCenteredSphericalFunctionGaussian.h"
#include "AtomCenteredSphericalFunctionSinc.h"
#include "AtomCenteredSphericalFunctionBessel.h"
#include "wigner/gaunt.hpp"
#include <memory>
#include <MemorySpaceType.h>
#include <headers.h>
#include <TypeConfig.h>
#include <dftUtils.h>
#include "FEBasisOperations.h"
#include <BLASWrapper.h>
#include <xc.h>
#include <mixingClass.h>
#include <excManager.h>
#include <AuxDensityMatrixRadial.h>
#include <pseudopotentialBaseClass.h>
#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
#endif
namespace dftfe
{

  enum class TypeOfField
  {
    In,
    Out,
    Residual
  };


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class pawClass : public pseudopotentialBaseClass<ValueType, memorySpace>
  {
  public:
    pawClass(const MPI_Comm                           &mpi_comm_parent,
             const std::string                        &scratchFolderName,
             dftParameters                            *dftParamsPtr,
             const std::set<dftfe::uInt>              &atomTypes,
             const bool                                floatingNuclearCharges,
             const dftfe::uInt                         nOMPThreads,
             const std::map<dftfe::uInt, dftfe::uInt> &atomAttributes,
             const bool                                reproducibleOutput,
             const dftfe::Int                          verbosity,
             const bool                                useDevice,
             const bool                                memOptMode);
    /**
     * @brief Initialises all the data members with addresses/values to/of dftClass.
     * @param[in] densityQuadratureId quadratureId for density.
     * @param[in] localContributionQuadratureId quadratureId for local/zero
     * potential
     * @param[in] nuclearChargeQuadratureIdElectro quadratureId for nuclear
     * charges
     * @param[in] densityQuadratureIdElectro quadratureId for density in
     * Electrostatics mesh
     * @param[in] excFunctionalPtr address XC functional pointer
     * @param[in] numEigenValues number of eigenvalues
     * @param[in] atomLocations atomic Coordinates
     * @param[in] imageIds image IDs of periodic cell
     * @param[in] periodicCoords coordinates of image atoms
     */

    void
    initialise(
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
        basisOperationsHostPtr,
#if defined(DFTFE_WITH_DEVICE)
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<ValueType,
                                        double,
                                        dftfe::utils::MemorySpace::DEVICE>>
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
      const bool computeStress);

    /**
     * @brief Initialises all the data members with addresses/values to/of dftClass.
     * @param[in] densityQuadratureId quadratureId for density.
     * @param[in] localContributionQuadratureId quadratureId for local/zero
     * potential
     * @param[in] nuclearChargeQuadratureIdElectro quadratureId for nuclear
     * charges
     * @param[in] densityQuadratureIdElectro quadratureId for density in
     * Electrostatics mesh
     * @param[in] bQuadValuesAllAtoms address of nuclear charge field
     * @param[in] excFunctionalPtr address XC functional pointer
     * @param[in] numEigenValues number of eigenvalues
     * @param[in] atomLocations atomic Coordinates
     * @param[in] imageIds image IDs of periodic cell
     * @param[in] periodicCoords coordinates of image atoms
     */
    void
    initialiseNonLocalContribution(
      const std::vector<dftfe::Int>          &imageIds,
      const std::vector<std::vector<double>> &periodicCoords,
      const std::vector<double>              &kPointWeights,
      const std::vector<double>              &kPointCoordinates,
      const bool                              updateNonlocalSparsity,
      const dftfe::uInt                       dofHanderId);


    void
    initialiseNonLocalContribution(
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
      dftfe::uInt                     numberElements);



    void
    computeCompensationCharge(TypeOfField typeOfField);

    void
    computeCompensationChargeMemoryOpt(TypeOfField typeOfField);
    void
    computeAtomDependentCompensationChargeMemoryOpt(TypeOfField typeOfField);
    /**
     * @brief pawclass omputecompensationchargel0:
     *
     */
    double
    TotalCompensationCharge();

    void
    computeDij(const bool        isDijOut,
               const dftfe::uInt startVectorIndex,
               const dftfe::uInt vectorBlockSize,
               const dftfe::uInt spinIndex,
               const dftfe::uInt kpointIndex);

    /**
     * @brief Initialises local potential
     */
    void
    initLocalPotential();

    void
    getRadialValenceDensity(dftfe::uInt          Znum,
                            double               rad,
                            std::vector<double> &Val);

    double
    getRadialValenceDensity(dftfe::uInt Znum, double rad);

    double
    getRmaxValenceDensity(dftfe::uInt Znum);

    void
    getRadialCoreDensity(dftfe::uInt          Znum,
                         double               rad,
                         std::vector<double> &Val);

    const std::map<dealii::CellId, std::vector<double>> &
    getRhoCoreCorrectionValues();

    const std::map<dealii::CellId, std::vector<double>> &
    getRhoCoreRefinedValues();

    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>> &
    getRhoCoreAtomCorrectionValues();
    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>> &
    getGradRhoCoreAtomCorrectionValues();
    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>> &
    getHessianRhoCoreAtomCorrectionValues();

    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>> &
    getRhoCoreAtomRefinedValues();

    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>> &
    getGradRhoCoreAtomRefinedValues();

    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>> &
    getHessianRhoCoreAtomRefinedValues();
    const std::map<dftfe::Int, std::map<dftfe::uInt, std::vector<double>>> &
    getAtomDependentCompensationCharegValues();

    double
    getRadialCoreDensity(dftfe::uInt Znum, double rad);

    double
    getRmaxCoreDensity(dftfe::uInt Znum);

    void
    getRadialZeroPotential(dftfe::uInt          Znum,
                           double               rad,
                           std::vector<double> &Val);

    double
    getRmaxZeroPotential(dftfe::uInt Znum);

    double
    getMaxAugmentationRadius();

    bool
    coreNuclearDensityPresent(dftfe::uInt Znum);
    // Returns the number of Projectors for the given atomID in cooridnates List
    dftfe::uInt
    getTotalNumberOfSphericalFunctionsForAtomId(dftfe::uInt atomId);
    // Returns the Total Number of atoms with support in the processor
    dftfe::uInt
    getTotalNumberOfAtomsInCurrentProcessor();
    // Returns the atomID in coordinates list for the iAtom index.
    dftfe::uInt
    getAtomIdInCurrentProcessor(dftfe::uInt iAtom);


    const dftfe::utils::MemoryStorage<ValueType, memorySpace> &
    getCouplingMatrix(
      CouplingType couplingtype = CouplingType::HamiltonianEntries);

    const dftfe::utils::MemoryStorage<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace> &
    getCouplingMatrixSinglePrec(
      CouplingType couplingtype = CouplingType::HamiltonianEntries);


    const std::shared_ptr<
      AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
    getNonLocalOperator();

    const std::shared_ptr<AtomicCenteredNonLocalOperator<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace>>
    getNonLocalOperatorSinglePrec();

    void
    evaluateNonLocalHamiltonianElectrostaticsValue(
      const distributedCPUVec<double> &phiTotNodalValues,
      const dftfe::uInt                dofHandlerId);

    // void
    // evaluateZeroPotentialCoreDensityContribution(
    //   const std::map<dealii::CellId, std::vector<double>> &coreRho,
    //   const std::map<dealii::CellId, std::vector<double>> &zeroPotential,
    //   const dftfe::utils::MemoryStorage<double,
    //   dftfe::utils::MemorySpace::HOST>
    //     &rhoInQuadValues);

    void
    initialiseExchangeCorrelationEnergyCorrection(dftfe::uInt spinIndex = 0);



    void
    computeNonlocalPseudoPotentialConstants(CouplingType couplingtype,
                                            dftfe::uInt  s = 0);



    double
    computeDeltaExchangeCorrelationEnergy(double &DeltaExchangeCorrelationVal,
                                          TypeOfField typeOfField);



    std::vector<double>
    getDijWeights();

    std::vector<double>
    DijVectorForMixing(TypeOfField typeOfField, mixingVariable mixVar);

    double
    densityScalingFactor(const std::vector<std::vector<double>> &atomLocations);

    void
    communicateDijAcrossAllProcessors(
      TypeOfField      typeOfField,
      const dftfe::Int iComp,
      const MPI_Comm  &interpoolcomm,
      const MPI_Comm  &interBandGroupComm,
      const bool       communicateDijAcrossAllProcessors = true);

    void
    computeDijFromPSIinitialGuess(
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
      const MPI_Comm                         &interBandGroupComm);

    void
    chargeNeutrality(double      integralRhoValue,
                     TypeOfField typeOfField,
                     bool        computeCompCharge = true);

    void
    fillDijMatrix(TypeOfField                typeOfField,
                  const dftfe::Int           iComp,
                  const std::vector<double> &DijVector,
                  const MPI_Comm            &interpoolcomm,
                  const MPI_Comm            &interBandGroupComm);
    std::vector<double>
    getDeltaEnergy(TypeOfField typeOfField = TypeOfField::Out);

    void
    computeIntegralCoreDensity(
      const std::map<dealii::CellId, std::vector<double>> &rhoCore);

    double
    computeNormDij(std::vector<double> &DijResidual);

    void
    saveDijEntriesToFile(const MPI_Comm &mpiCommParent,
                         TypeOfField     typeOfField);

    void
    loadDijEntriesFromFile();


    void
    printAtomsForces();

    std::vector<double>
    getAtomsForces();

    void
    computeIntegralAEdensityInsideAugmentationSphere();

    void
    zeroPotentialContribution(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityQuadValues,
      const std::map<dealii::CellId, std::vector<double>>
                       &zeroPotentalAtQuadPoints,
      const dftfe::uInt quadratureId,
      TypeOfField       typeOfField);

    double
    computePAWCorrectionContribution(dftfe::uInt index,
                                     TypeOfField typeOfField);

    void
    determineAtomsOfInterstPseudopotential(
      const std::vector<std::vector<double>> &atomCoordinates);
    const std::map<dftfe::uInt, dftfe::uInt> &
    getPSPAtomIdToGlobalIdMap();

    const bool
    hasSOC() const;

  private:
    void
    createAtomTypesList(const std::vector<std::vector<double>> &atomLocations);

    void
    initialiseDataonRadialMesh();

    void
    initialiseColoumbicEnergyCorrection();

    void
    initialiseZeroPotential();



    void
    initialiseKineticEnergyCorrection();

    void
    computeRadialMultipoleData();

    // void
    // computeAtomDependentSinverseCouplingMatrix();

    void
    computeMultipoleInverse();

    void
    computeInverseOfMultipoleData();

    void
    computeCompensationChargeL0();

    void
    initRhoCoreCorrectionValues();

    void
    computeCompensationChargeL0(
      const std::vector<double> &scalingFactorPerAtom);

    void
    computeCompensationChargeCoeffMemoryOpt();

    void
    computeproductOfCGMultipole();

    void
    saveDeltaSinverseEntriesToFile();



    int
    loadDeltaSinverseEntriesFromFile();
    // std::map<dftfe::uInt, std::vector<double>> d_sinverseCouplingMatrix;
    std::map<dftfe::uInt, std::vector<double>> d_KineticEnergyCorrectionTerm;
    std::map<dftfe::uInt, std::vector<double>> d_zeroPotentialij;
    std::map<dftfe::uInt, double> d_zeroPotentialCoreEnergyContribution;
    std::map<dftfe::uInt, std::vector<double>>
      d_ExchangeCorrelationEnergyCorrectionTerm;
    std::map<dftfe::uInt, std::vector<double>> d_ColoumbicEnergyCorrectionTerm;
    std::map<dftfe::uInt, std::vector<double>> d_DeltaColoumbicEnergyDij;
    std::map<dftfe::uInt, double> d_coreKE, d_deltaC, d_coreXC, d_deltaValenceC;
    std::map<dftfe::uInt, std::vector<double>> d_deltaCij, d_deltaCijkl;
    std::map<dftfe::uInt, std::vector<double>>
                             d_nonLocalHamiltonianElectrostaticValue;
    dftfe::uInt              d_nProjPerTask, d_nProjSqTotal, d_totalProjectors;
    std::vector<dftfe::uInt> d_projectorStartIndex;
    std::vector<dftfe::uInt> d_totalProjectorStartIndex;
    double                   d_TotalCompensationCharge;
    double d_integralCoreDensity, d_integrealCoreDensityRadial;
    std::map<dftfe::uInt, double>                 d_integralCoreDensityPerAtom;
    std::map<dealii::CellId, std::vector<double>> d_jxwcompensationCharge;

    // Force
    std::vector<double> d_totalforceOnAtoms;

    /**
     * @brief Converts the periodic image data structure to relevant form for the container class
     * @param[in] atomLocations atomic Coordinates
     * @param[in] imageIds image IDs of periodic cell
     * @param[in] periodicCoords coordinates of image atoms
     * @param[out] imageIdsTemp image IDs of periodic cell
     * @param[out] imageCoordsTemp coordinates of image atoms
     */
    void
    setImageCoordinates(const std::vector<std::vector<double>> &atomLocations,
                        const std::vector<dftfe::Int>          &imageIds,
                        const std::vector<std::vector<double>> &periodicCoords,
                        std::vector<dftfe::uInt>               &imageIdsTemp,
                        std::vector<double> &imageCoordsTemp);
    /**
     * @brief Creating Density splines for all atomTypes
     */
    void
    createAtomCenteredSphericalFunctionsForDensities();

    void
    createAtomCenteredSphericalFunctionsForShapeFunctions();


    void
    createAtomCenteredSphericalFunctionsForProjectors();
    void
    createAtomCenteredSphericalFunctionsForZeroPotential();

    std::vector<double>
    computeZeroPotentialFromLocalPotential(
      const std::vector<double> &radialMesh,
      const std::vector<double> &rab,
      const double              *localPotential,
      const double              *coreDensityAE,
      const double              *coreDensityPS,
      const double              *shapeFunction0,
      const double              *valenceDensity,
      const double               Znum,
      const double               Ncore,
      const double               NtildeCore,
      const double               NValence,
      const double               NtildeValence);

    std::vector<double>
    computeLocalPotentialFromZeroPotential(
      const std::vector<double> &radialMesh,
      const std::vector<double> &rab,
      const double              *zeroPotential,
      const double              *coreDensityAE,
      const double              *coreDensityPS,
      const double              *shapeFunction0,
      const double              *valenceDensity,
      const double               Znum,
      const double               Ncore,
      const double               NtildeCore,
      const double               NValence,
      const double               NtildeValence);



    std::complex<double>
    computeTransformationExtries(dftfe::Int l, dftfe::Int mu, dftfe::Int m);
    /**
     * @brief factorial: Recursive method to find factorial
     *
     *
     *  @param[in] n
     *
     */

    double
    factorial(double n)
    {
      if (n < 0)
        {
          // pcout<<"-ve Factorial"<<std::endl;
          return -100.;
        }
      return (n == 1. || n == 0.) ? 1. : factorial(n - 1) * n;
    }
    // Utils Functions
    double
    gaunt(dftfe::Int l_i,
          dftfe::Int l_j,
          dftfe::Int l,
          dftfe::Int m_i,
          dftfe::Int m_j,
          dftfe::Int m);
    double
    multipoleIntegrationGrid(double              *f1,
                             double              *f2,
                             std::vector<double> &radial,
                             std::vector<double> &rab,
                             const dftfe::Int     L,
                             const dftfe::uInt    rminIndex,
                             const dftfe::uInt    rmaxIndex);
    double
    simpsonIntegral(const dftfe::uInt                           startIndex,
                    const dftfe::uInt                           endIndex,
                    std::function<double(const dftfe::uInt &)> &IntegrandValue);

    const std::vector<double>
    simpsonIntegralWeights(const dftfe::uInt startIndex,
                           const dftfe::uInt EndIndex);

    double
    SimpsonResidual(const dftfe::uInt          startIndex,
                    const dftfe::uInt          EndIndex,
                    const std::vector<double> &integrandValue);

    // COmputes \int{f1(r)*f2(r)*f3(r)*r^2dr*J_r}
    double
    threeTermIntegrationOverAugmentationSphere(double              *f1,
                                               double              *f2,
                                               double              *f3,
                                               std::vector<double> &radial,
                                               std::vector<double> &rab,
                                               const dftfe::uInt    rminIndex,
                                               const dftfe::uInt    rmaxIndex);
    // Computes the potential due to charge fun
    void
    oneTermPoissonPotential(const double              *fun,
                            const dftfe::uInt          l,
                            const dftfe::uInt          rminIndex,
                            const dftfe::uInt          rmaxIndex,
                            const dftfe::Int           powerofR,
                            const std::vector<double> &radial,
                            const std::vector<double> &rab,
                            std::vector<double>       &Potential);

    void
    twoTermPoissonPotential(const double              *fun1,
                            const double              *fun2,
                            const dftfe::uInt          l,
                            const dftfe::uInt          rminIndex,
                            const dftfe::uInt          rmaxIndex,
                            const dftfe::Int           powerofR,
                            const std::vector<double> &radial,
                            const std::vector<double> &rab,
                            std::vector<double>       &Potential);

    double
    integralOfProjectorsInAugmentationSphere(const double        *f1,
                                             const double        *f2,
                                             std::vector<double> &radial,
                                             std::vector<double> &rab,
                                             const dftfe::uInt    rminIndex,
                                             const dftfe::uInt    rmaxIndex);

    double
    integralOfDensity(const double        *f1,
                      std::vector<double> &radial,
                      std::vector<double> &rab,
                      const dftfe::uInt    rminIndex,
                      const dftfe::uInt    rmaxIndex);
    void
    getSphericalQuadratureRule(std::vector<double>              &quad_weights,
                               std::vector<std::vector<double>> &quad_points);

    void
    computeCoreDeltaExchangeCorrelationEnergy();


    void
    computeAugmentationOverlap();

    void
    checkOverlapAugmentation();

    std::vector<dftfe::uInt>
    relevantAtomdIdsInCurrentProcs();

    std::vector<dftfe::uInt>
    relevantAtomdIdsInCurrentProcs(
      std::vector<dftfe::uInt>                         &numberOfCellsPerAtom,
      std::vector<std::pair<dftfe::uInt, dftfe::uInt>> &atomNumberNCellsPair);

    std::vector<double>
    derivativeOfRealSphericalHarmonic(dftfe::uInt lQuantumNo,
                                      dftfe::Int  mQuantumNo,
                                      double      theta,
                                      double      phi);
    std::vector<double>
    radialDerivativeOfMeshData(const std::vector<double> &r,
                               const std::vector<double> &rab,
                               const std::vector<double> &functionValue);

    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      d_BLASWrapperHostPtr;
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      d_BLASWrapperDevicePtr;
#endif
    std::vector<std::vector<double>> d_nonLocalPseudoPotentialConstants;
    std::map<CouplingType, std::map<dftfe::uInt, std::vector<ValueType>>>
      d_atomicNonLocalPseudoPotentialConstants;
    std::map<CouplingType, dftfe::utils::MemoryStorage<ValueType, memorySpace>>
      d_couplingMatrixEntries;
    std::map<CouplingType,
             dftfe::utils::MemoryStorage<
               typename dftfe::dataTypes::singlePrecType<ValueType>::type,
               memorySpace>>
         d_couplingMatrixEntriesSinglePrec;
    bool d_HamiltonianCouplingMatrixEntriesUpdated,
      d_overlapCouplingMatrixEntriesUpdated,
      d_inverseCouplingMatrixEntriesUpdated;
    bool d_HamiltonianCouplingMatrixEntriesUpdatedSinglePrec,
      d_overlapCouplingMatrixEntriesUpdatedSinglePrec,
      d_inverseCouplingMatrixEntriesUpdatedSinglePrec;
    std::vector<std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicWaveFnsVector;
    std::shared_ptr<AtomCenteredSphericalFunctionContainer>
      d_atomicProjectorFnsContainer;

    std::shared_ptr<AtomCenteredSphericalFunctionContainer>
      d_atomicShapeFnsContainer;

    std::map<std::pair<dftfe::uInt, dftfe::uInt>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicProjectorFnsMap, d_atomicAEPartialWaveFnsMap,
      d_atomicPSPartialWaveFnsMap;

    std::map<std::pair<dftfe::uInt, dftfe::uInt>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicShapeFnsMap;
    // parallel communication objects
    const MPI_Comm    d_mpiCommParent;
    const dftfe::uInt d_this_mpi_process;
    const dftfe::uInt d_n_mpi_processes;

    // conditional stream object
    dealii::ConditionalOStream pcout;
    bool                       d_useDevice;
    bool                       d_memoryOptMode;
    dftfe::uInt                d_densityQuadratureId;
    dftfe::uInt                d_compensationChargeQuadratureIdElectro;
    dftfe::uInt                d_localContributionQuadratureId;
    dftfe::uInt                d_nuclearChargeQuadratureIdElectro;
    dftfe::uInt                d_densityQuadratureIdElectro;
    dftfe::uInt                d_sparsityPatternQuadratureId;
    dftfe::uInt                d_nlpspQuadratureId;
    bool                       d_singlePrecNonLocalOperator;
    // dftfe::uInt                d_dofHandlerID;
    std::shared_ptr<excManager<memorySpace>> d_excManagerPtr;
    dftParameters                           *d_dftParamsPtr;
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
      d_BasisOperatorHostPtr;
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::DEVICE>>
      d_BasisOperatorDevicePtr;
#endif
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      d_BasisOperatorElectroHostPtr;

    std::map<dftfe::uInt, std::vector<double>>
      d_ProductOfQijShapeFnAtQuadPoints;
    std::map<dftfe::uInt, std::vector<double>> d_shapeFnAtQuadPoints;

    std::vector<
      std::map<TypeOfField, std::map<dftfe::uInt, std::vector<double>>>>
                                               D_ij;
    std::map<dftfe::uInt, std::vector<double>> d_multipole, d_multipoleInverse;
    std::map<dftfe::uInt, std::vector<double>> d_aePartialWfcIntegralIJ;
    std::map<dftfe::uInt, std::vector<double>> d_productOfMultipoleClebshGordon;
    std::vector<double> d_deltaInverseMatrix, d_deltaMatrix;
    std::map<std::pair<dftfe::uInt, dftfe::uInt>, std::vector<double>>
      d_gLValuesQuadPoints;
    std::map<std::pair<dftfe::uInt, dftfe::uInt>, std::vector<double>>
      d_g0ValuesQuadPoints;
    std::map<dftfe::Int, std::map<dftfe::uInt, std::vector<double>>>
      d_bAtomsValuesQuadPoints;
    std::map<std::pair<dftfe::uInt, dftfe::uInt>, dftfe::Int>
                                       d_imageIdFromAtomCompactSupportMap;
    std::map<dftfe::uInt, double>      d_DeltaL0coeff, d_NtildeCore, d_Ncore;
    std::map<dftfe::uInt, double>      d_RmaxAug, d_RminAug, d_RmaxComp;
    std::map<dftfe::uInt, dftfe::uInt> d_RmaxAugIndex;
    std::map<dftfe::uInt, dftfe::uInt> d_RmaxAugIndexShapeFn;

    // Radial Data on meshGrid
    std::map<dftfe::uInt, std::vector<double>> d_radialMesh,
      d_radialJacobianData;
    std::map<dftfe::uInt, double> d_radialValueCoreSmooth0,
      d_radialValueCoreAE0;
    std::map<dftfe::uInt, std::vector<double>> d_PSWfc0, d_AEWfc0;
    std::map<dftfe::uInt, std::vector<double>> d_productOfAEpartialWfc,
      d_productOfPSpartialWfc, d_atomCoreDensityAE, d_atomCoreDensityPS,
      d_atomicShapeFn;

    std::map<dftfe::uInt, std::vector<double>> d_gradCoreAE, d_gradCorePS,
      d_gradCoreSqAE;
    std::map<dftfe::uInt, std::vector<double>> d_productOfPSpartialWfcDer,
      d_productOfAEpartialWfcDer; // \phi^a_i(r)\frac{d\phi^a_j(r)}{dr}
    std::map<dftfe::uInt, std::vector<double>> d_productOfPSpartialWfcValue,
      d_productOfAEpartialWfcValue; // \frac{\phi^a_i(r)\phi^a_j(r)}{r}
    std::map<dftfe::uInt, std::vector<double>> d_radialWfcDerAE,
      d_radialWfcValAE, d_radialWfcDerPS, d_radialWfcValPS, d_radialProjVal,
      d_radialCoreDerAE, d_radialCoreDerPS;
    std::map<dftfe::uInt, std::vector<double>> d_zeroPotentialRadialValues;
    // Total Comepsantion charge field
    std::map<dealii::CellId, std::vector<double>>     *d_bQuadValuesAllAtoms;
    std::map<dealii::CellId, std::vector<dftfe::Int>> *d_bQuadAtomIdsAllAtoms;
    std::map<dealii::CellId, std::vector<dftfe::Int>>
      *d_bQuadAtomIdsAllAtomsImages;
    // Total Compensation charge field only due to the g_0(r)Delta_0 component
    std::map<dealii::CellId, std::vector<double>> d_bl0QuadValuesAllAtoms;
    std::map<dealii::CellId, std::vector<double>> d_rhoCoreRefinedValues;
    std::map<dealii::CellId, std::vector<double>> d_rhoCoreCorrectionValues;
    std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
      d_rhoCoreAtomsRefinedValues, d_rhoCoreAtomsCorrectionValues;
    std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
      d_gradRhoCoreAtomsRefinedValues, d_gradRhoCoreAtomsCorrectionValues;
    std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
      d_HessianRhoCoreAtomsRefinedValues, d_HessianRhoCoreAtomsCorrectionValues;
    distributedCPUVec<ValueType>                    Pmatrix;
    std::map<dftfe::uInt, bool>                     d_atomTypeCoreFlagMap;
    bool                                            d_floatingNuclearCharges;
    dftfe::Int                                      d_verbosity;
    std::set<dftfe::uInt>                           d_atomTypes;
    std::map<dftfe::uInt, dftfe::uInt>              d_valenceElectrons;
    std::map<dftfe::uInt, std::vector<dftfe::uInt>> d_atomTypesList;
    std::map<dftfe::uInt, std::vector<dftfe::uInt>>
                                     d_LocallyOwnedAtomIdMapWithAtomType;
    std::vector<dftfe::uInt>         d_LocallyOwnedAtomId;
    std::string                      d_dftfeScratchFolderName;
    std::vector<dftfe::Int>          d_imageIds;
    std::vector<std::vector<double>> d_imagePositions;
    dftfe::uInt                      d_numEigenValues;
    dftfe::uInt                      d_nOMPThreads;

    std::vector<double> d_kpointWeights;

    std::shared_ptr<AuxDensityMatrix<memorySpace>> d_auxDensityMatrixXCPSPtr;
    std::shared_ptr<AuxDensityMatrix<memorySpace>> d_auxDensityMatrixXCAEPtr;
    // Creating Object for Atom Centerd Nonlocal Operator
    std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
      d_nonLocalOperator;
    // Creating Object for Atom Centerd Nonlocal Operator SinglePrec
    std::shared_ptr<AtomicCenteredNonLocalOperator<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace>>
      d_nonLocalOperatorSinglePrec;

    std::vector<std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicProjectorFnsVector;
    std::vector<
      std::map<dftfe::uInt, std::shared_ptr<AtomCenteredSphericalFunctionBase>>>
      d_atomicZeroPotVector;
    std::vector<
      std::map<dftfe::uInt, std::shared_ptr<AtomCenteredSphericalFunctionBase>>>
      d_atomicValenceDensityVector;
    std::vector<
      std::map<dftfe::uInt, std::shared_ptr<AtomCenteredSphericalFunctionBase>>>
         d_atomicCoreDensityVector;
    bool d_reproducible_output;
    std::map<unsigned int, std::map<unsigned int, std::array<double, 4>>>
         d_atomicProjectorFnsljmValues;
    bool d_hasSOC;
    /// FIXME: eventually it should be a map of atomic number to struct-
    /// {valence number, mesh input etc}
    std::map<dftfe::uInt, dftfe::uInt> d_atomTypeAtributes;
    std::vector<std::vector<double>>   d_atomLocationsInterestPseudopotential;
    std::map<dftfe::uInt, dftfe::uInt>
      d_atomIdPseudopotentialInterestToGlobalId;

  }; // end of class

} // end of namespace dftfe
// #include "../src/pseudo/paw/pawClassInit.t.cc"
// #include "../src/pseudo/paw/pawClass.t.cc"
// #include "../src/pseudo/paw/pawClassUtils.t.cc"
// #include "../src/pseudo/paw/pawClassEnergy.t.cc"
#endif //  DFTFE_PSEUDOPOTENTIALBASE_H
