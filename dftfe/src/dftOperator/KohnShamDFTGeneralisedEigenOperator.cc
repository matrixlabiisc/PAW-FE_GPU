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
// @author Nikhil Kodali
//

#include <KohnShamDFTGeneralisedEigenOperator.h>
#include <ExcDFTPlusU.h>
#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
#endif
#if defined(DFTFE_WITH_DEVICE)
#  include <DeviceAPICalls.h>
#endif

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  KohnShamDFTGeneralisedEigenOperator<memorySpace>::
    KohnShamDFTGeneralisedEigenOperator(
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        BLASWrapperPtr,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        BLASWrapperPtrHost,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
        basisOperationsPtr,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtrHost,
      std::shared_ptr<
        dftfe::pseudopotentialBaseClass<dataTypes::number, memorySpace>>
                                               pseudopotentialClassPtr,
      std::shared_ptr<excManager<memorySpace>> excManagerPtr,
      dftParameters                           *dftParamsPtr,
      const dftfe::uInt                        densityQuadratureID,
      const dftfe::uInt                        lpspQuadratureID,
      const dftfe::uInt                        feOrderPlusOneQuadratureID,
      const MPI_Comm                          &mpi_comm_parent,
      const MPI_Comm                          &mpi_comm_domain)
    : KohnShamDFTBaseOperator<memorySpace>(BLASWrapperPtr,
                                           BLASWrapperPtrHost,
                                           basisOperationsPtr,
                                           basisOperationsPtrHost,
                                           pseudopotentialClassPtr,
                                           excManagerPtr,
                                           dftParamsPtr,
                                           densityQuadratureID,
                                           lpspQuadratureID,
                                           feOrderPlusOneQuadratureID,
                                           mpi_comm_parent,
                                           mpi_comm_domain)
  {}
  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTGeneralisedEigenOperator<memorySpace>::
    reinitNonLocalOperatorBlockVector(const dftfe::uInt numberWavefunctions)
  {
    if (d_dftParamsPtr->isPseudopotential)
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          {
            d_pseudopotentialNonLocalOperator->initialiseFlattenedDataStructure(
              numberWavefunctions,
              d_pseudopotentialNonLocalProjectorTimesVectorBlock);
            d_pseudopotentialNonLocalOperator
              ->initialiseCellWaveFunctionPointers(d_cellWaveFunctionMatrixSrc,
                                                   d_cellsBlockSizeHX);
          }
        else
          {
            d_pseudopotentialNonLocalOperator->initialiseFlattenedDataStructure(
              numberWavefunctions,
              d_pseudopotentialNonLocalProjectorTimesVectorBlock);
          }
      }
    if (d_dftParamsPtr->isPseudopotential && d_dftParamsPtr->useSinglePrecCheby)
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          {
            d_pseudopotentialNonLocalOperatorSinglePrec
              ->initialiseFlattenedDataStructure(
                numberWavefunctions,
                d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec);
            d_pseudopotentialNonLocalOperatorSinglePrec
              ->initialiseCellWaveFunctionPointers(
                d_cellWaveFunctionMatrixSrcSinglePrec, d_cellsBlockSizeHX);
          }
        else
          d_pseudopotentialNonLocalOperatorSinglePrec
            ->initialiseFlattenedDataStructure(
              numberWavefunctions,
              d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec);
        if (d_dftParamsPtr->communPrecCheby == "BF16")
          d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
            .setCommunicationPrecision(
              dftfe::utils::mpi::communicationPrecision::half);
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTGeneralisedEigenOperator<memorySpace>::overlapMatrixTimesX(
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    const double                                                       scalarOX,
    const double                                                       scalarY,
    const double                                                       scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const bool useApproximateMatrixEntries)
  {
    const dftfe::uInt numCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt numberWavefunctions = src.numVectors();
    const double      one(1.0);
    if (d_numVectorsInternal != numberWavefunctions)
      reinitNumberWavefunctions(numberWavefunctions);

    if (d_basisOperationsPtr->d_nVectors != numberWavefunctions)
      d_basisOperationsPtr->reinit(numberWavefunctions,
                                   d_cellsBlockSizeHX,
                                   d_densityQuadratureID,
                                   false,
                                   false);

    d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * src.numVectors(),
                            scalarX,
                            src.data(),
                            scalarY,
                            dst.data());
    src.updateGhostValues();
    d_basisOperationsPtr->distribute(src);
    const dataTypes::number scalarCoeffAlpha = 1.0,
                            scalarCoeffBeta  = dataTypes::number(0.0);

    if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
      if (d_dftParamsPtr->isPseudopotential)
        d_pseudopotentialNonLocalOperator->initialiseOperatorActionOnX(
          d_kPointIndex);
    const bool hasNonlocalComponents =
      d_dftParamsPtr->isPseudopotential &&
      (d_pseudopotentialNonLocalOperator
         ->getTotalNonLocalElementsInCurrentProcessor() > 0);
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<dftfe::uInt, dftfe::uInt> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
        d_BLASWrapperPtr->stridedCopyToBlock(
          numberWavefunctions,
          numDoFsPerCell * (cellRange.second - cellRange.first),
          src.data(),
          d_cellWaveFunctionMatrixSrc.data() +
            (d_dftParamsPtr->memOptMode ?
               0 :
               cellRange.first * numDoFsPerCell * numberWavefunctions),
          d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
              .data() +
            cellRange.first * numDoFsPerCell);
        if (hasNonlocalComponents)
          d_pseudopotentialNonLocalOperator->applyCconjtransOnX(
            d_cellWaveFunctionMatrixSrc.data() +
              (d_dftParamsPtr->memOptMode ?
                 0 :
                 cellRange.first * numDoFsPerCell * numberWavefunctions),
            cellRange);
      }
    d_pseudopotentialNonLocalProjectorTimesVectorBlock.setValue(0);
    d_pseudopotentialNonLocalOperator->applyAllReduceOnCconjtransX(
      d_pseudopotentialNonLocalProjectorTimesVectorBlock);
    d_pseudopotentialNonLocalOperator->applyVOnCconjtransX(
      CouplingStructure::dense,
      d_pseudopotentialClassPtr->getCouplingMatrix(
        CouplingType::OverlapEntries),
      d_pseudopotentialNonLocalProjectorTimesVectorBlock,
      true);

    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<dftfe::uInt, dftfe::uInt> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
        if (d_dftParamsPtr->memOptMode)
          {
            d_BLASWrapperPtr->stridedCopyToBlock(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              src.data(),
              d_cellWaveFunctionMatrixSrc.data(),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
          }
        if (!useApproximateMatrixEntries)
          {
            d_BLASWrapperPtr->xgemmStridedBatched(
              'N',
              'N',
              numberWavefunctions,
              numDoFsPerCell,
              numDoFsPerCell,
              &scalarCoeffAlpha,
              d_cellWaveFunctionMatrixSrc.data() +
                (d_dftParamsPtr->memOptMode ?
                   0 :
                   cellRange.first * numDoFsPerCell * numberWavefunctions),
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              d_basisOperationsPtr->cellMassMatrix().data() +
                cellRange.first * numDoFsPerCell * numDoFsPerCell,
              numDoFsPerCell,
              numDoFsPerCell * numDoFsPerCell,
              &scalarCoeffBeta,
              d_cellWaveFunctionMatrixDst.data(),
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              cellRange.second - cellRange.first);
          }
        else
          d_cellWaveFunctionMatrixDst.setValue(0.0);
        if (hasNonlocalComponents)
          d_pseudopotentialNonLocalOperator->applyCOnVCconjtransX(
            d_cellWaveFunctionMatrixDst.data(), cellRange);
        d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
          numberWavefunctions,
          numDoFsPerCell * (cellRange.second - cellRange.first),
          scalarOX,
          d_cellWaveFunctionMatrixDst.data(),
          dst.data(),
          d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
              .data() +
            cellRange.first * numDoFsPerCell);
      }

    d_basisOperationsPtr->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
      .distribute_slave_to_master(dst);
    src.zeroOutGhosts();
    inverseMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
    dst.accumulateAddLocallyOwned();
    dst.zeroOutGhosts();
    if (useApproximateMatrixEntries)
      {
        const dftfe::uInt blockSize = src.numVectors();
        d_BLASWrapperPtr->stridedBlockAxpy(
          blockSize,
          src.locallyOwnedSize(),
          src.data(),
          d_basisOperationsPtr->massVectorBasisData().data(),
          scalarOX,
          dst.data());
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTGeneralisedEigenOperator<memorySpace>::overlapInverseMatrixTimesX(
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    const double scalarOinvX,
    const double scalarY,
    const double scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst)
  {
    // Add Assert that scalarX and scalarY has to be 0

    const dftfe::uInt numCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt numberWavefunctions = src.numVectors();
    const bool        hasNonlocalComponents =
      d_dftParamsPtr->isPseudopotential &&
      (d_pseudopotentialNonLocalOperator
         ->getTotalNonLocalElementsInCurrentProcessor() > 0);
    dst = src;
    src.updateGhostValues();
    inverseMassVectorScaledConstraintsNoneDataInfoPtr->distribute(src);
    if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
      d_pseudopotentialNonLocalOperator->initialiseOperatorActionOnX(
        d_kPointIndex);
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<dftfe::uInt, dftfe::uInt> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
        // Can be optimized. Not needed for all cells
        d_BLASWrapperPtr->stridedBlockScaleCopy(
          numberWavefunctions,
          numDoFsPerCell * (cellRange.second - cellRange.first),
          1.0,
          d_basisOperationsPtr->cellInverseMassVectorBasisData().data() +
            cellRange.first * numDoFsPerCell,
          src.data(),
          d_cellWaveFunctionMatrixSrc.data() +
            (d_dftParamsPtr->memOptMode ?
               0 :
               cellRange.first * numDoFsPerCell * numberWavefunctions),
          d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
              .data() +
            cellRange.first * numDoFsPerCell);
        if (hasNonlocalComponents)
          d_pseudopotentialNonLocalOperator->applyCconjtransOnX(
            d_cellWaveFunctionMatrixSrc.data() +
              (d_dftParamsPtr->memOptMode ?
                 0 :
                 cellRange.first * numDoFsPerCell * numberWavefunctions),
            cellRange);
      }
    d_pseudopotentialNonLocalProjectorTimesVectorBlock.setValue(0);
    d_pseudopotentialNonLocalOperator->applyAllReduceOnCconjtransX(
      d_pseudopotentialNonLocalProjectorTimesVectorBlock, false);
    d_pseudopotentialNonLocalOperator->applyVOnCconjtransX(
      CouplingStructure::dense,
      d_pseudopotentialClassPtr->getCouplingMatrix(
        CouplingType::inverseOverlapEntries),
      d_pseudopotentialNonLocalProjectorTimesVectorBlock,
      true,
      d_kPointIndex);

    // VC^TMinvX is done
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        d_cellWaveFunctionMatrixDst.setValue(0);
        std::pair<dftfe::uInt, dftfe::uInt> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
        if (hasNonlocalComponents)
          {
            d_pseudopotentialNonLocalOperator->applyCOnVCconjtransX(
              d_cellWaveFunctionMatrixDst.data(), cellRange);
            d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              -1.0,
              d_cellWaveFunctionMatrixDst.data(),
              dst.data(),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
          }
      }
    d_basisOperationsPtr->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
      .distribute_slave_to_master(dst);
    src.zeroOutGhosts();
    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
    dst.accumulateAddLocallyOwned();
    dst.zeroOutGhosts();
    d_BLASWrapperPtr->stridedBlockScale(
      numberWavefunctions,
      dst.locallyOwnedSize(),
      1.0,
      d_basisOperationsPtr->inverseMassVectorBasisData().data(),
      dst.data());
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTGeneralisedEigenOperator<memorySpace>::overlapInverseMatrixTimesX(
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &src,
    const double scalarOinvX,
    const double scalarY,
    const double scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &dst)
  {
    // Add Assert that scalarX and scalarY has to be 0

    const dftfe::uInt numCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt numberWavefunctions = src.numVectors();
    const bool        hasNonlocalComponents =
      d_dftParamsPtr->isPseudopotential &&
      (d_pseudopotentialNonLocalOperatorSinglePrec
         ->getTotalNonLocalElementsInCurrentProcessor() > 0);
    dst = src;
    src.updateGhostValues();
    inverseMassVectorScaledConstraintsNoneDataInfoPtr->distribute(src);
    if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
      d_pseudopotentialNonLocalOperatorSinglePrec->initialiseOperatorActionOnX(
        d_kPointIndex);
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<dftfe::uInt, dftfe::uInt> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
        // Can be optimized. Not needed for all cells
        d_BLASWrapperPtr->stridedBlockScaleCopy(
          numberWavefunctions,
          numDoFsPerCell * (cellRange.second - cellRange.first),
          1.0,
          d_basisOperationsPtr->cellInverseMassVectorBasisData().data() +
            cellRange.first * numDoFsPerCell,
          src.data(),
          d_cellWaveFunctionMatrixSrcSinglePrec.data() +
            (d_dftParamsPtr->memOptMode ?
               0 :
               cellRange.first * numDoFsPerCell * numberWavefunctions),
          d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
              .data() +
            cellRange.first * numDoFsPerCell);
        if (hasNonlocalComponents)
          d_pseudopotentialNonLocalOperatorSinglePrec->applyCconjtransOnX(
            d_cellWaveFunctionMatrixSrcSinglePrec.data() +
              (d_dftParamsPtr->memOptMode ?
                 0 :
                 cellRange.first * numDoFsPerCell * numberWavefunctions),
            cellRange);
      }
    d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec.setValue(0);
    d_pseudopotentialNonLocalOperatorSinglePrec->applyAllReduceOnCconjtransX(
      d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec, false);
    d_pseudopotentialNonLocalOperatorSinglePrec->applyVOnCconjtransX(
      CouplingStructure::dense,
      d_pseudopotentialClassPtr->getCouplingMatrixSinglePrec(
        CouplingType::inverseOverlapEntries),
      d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec,
      true,
      d_kPointIndex);

    // VC^TMinvX is done
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        d_cellWaveFunctionMatrixDstSinglePrec.setValue(0);
        std::pair<dftfe::uInt, dftfe::uInt> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
        if (hasNonlocalComponents)
          {
            d_pseudopotentialNonLocalOperatorSinglePrec->applyCOnVCconjtransX(
              d_cellWaveFunctionMatrixDstSinglePrec.data(), cellRange);
            d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              -1.0,
              d_cellWaveFunctionMatrixDstSinglePrec.data(),
              dst.data(),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
          }
      }
    d_basisOperationsPtr->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
      .distribute_slave_to_master(dst);
    src.zeroOutGhosts();
    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
    dst.accumulateAddLocallyOwned();
    dst.zeroOutGhosts();
    d_BLASWrapperPtr->stridedBlockScale(
      numberWavefunctions,
      dst.locallyOwnedSize(),
      1.0,
      d_basisOperationsPtr->inverseMassVectorBasisData().data(),
      dst.data());
  }



  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTGeneralisedEigenOperator<memorySpace>::HXCheby(
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    const double                                                       scalarHX,
    const double                                                       scalarY,
    const double                                                       scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &tempVec,
    const HXChebyOperations operations)
  {
    const dftfe::uInt numCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt numberWavefunctions = src.numVectors();
    if (d_numVectorsInternal != numberWavefunctions)
      reinitNumberWavefunctions(numberWavefunctions);

    if (d_basisOperationsPtr->d_nVectors != numberWavefunctions)
      d_basisOperationsPtr->reinit(numberWavefunctions,
                                   d_cellsBlockSizeHX,
                                   d_densityQuadratureID,
                                   false,
                                   false);
    const bool hasNonlocalComponents =
      d_dftParamsPtr->isPseudopotential &&
      (d_pseudopotentialNonLocalOperator
         ->getTotalNonLocalElementsInCurrentProcessor() > 0);
    const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                            scalarCoeffBeta  = dataTypes::number(0.0);
    const dftfe::uInt blockSize              = src.numVectors();
    if (operations == HXChebyOperations::SinvXOnlySinvLocX ||
        operations == HXChebyOperations::All)
      {
        d_BLASWrapperPtr->stridedBlockAxpBy(
          blockSize,
          src.locallyOwnedSize(),
          src.data(),
          d_basisOperationsPtr->inverseMassVectorBasisData().data(),
          1.0,
          0.0,
          tempVec.data());
      }
    if (operations == HXChebyOperations::All)
      {
        tempVec.updateGhostValues();
      }
    if (operations == HXChebyOperations::SinvXOnlyExtractionApplyConjTransOnX ||
        operations == HXChebyOperations::All)
      {
        d_basisOperationsPtr->distribute(tempVec);
        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          if (d_dftParamsPtr->isPseudopotential)
            d_pseudopotentialNonLocalOperator->initialiseOperatorActionOnX(
              d_kPointIndex);
        for (dftfe::uInt iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
            d_BLASWrapperPtr->stridedCopyToBlock(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              tempVec.data(),
              d_cellWaveFunctionMatrixSrc.data() +
                (d_dftParamsPtr->memOptMode ?
                   0 :
                   cellRange.first * numDoFsPerCell * numberWavefunctions),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
            d_pseudopotentialNonLocalOperator->applyCconjtransOnX(
              d_cellWaveFunctionMatrixSrc.data() +
                (d_dftParamsPtr->memOptMode ?
                   0 :
                   cellRange.first * numDoFsPerCell * numberWavefunctions),
              cellRange);
          }
        d_pseudopotentialNonLocalProjectorTimesVectorBlock.setValue(0);
        d_pseudopotentialNonLocalOperator->applyAllReduceOnCconjtransX(
          d_pseudopotentialNonLocalProjectorTimesVectorBlock, true);
      }
    if (operations == HXChebyOperations::All)
      {
        d_pseudopotentialNonLocalProjectorTimesVectorBlock
          .accumulateAddLocallyOwnedBegin();
        d_pseudopotentialNonLocalProjectorTimesVectorBlock
          .accumulateAddLocallyOwnedEnd();
        d_pseudopotentialNonLocalProjectorTimesVectorBlock
          .updateGhostValuesBegin();
        d_pseudopotentialNonLocalProjectorTimesVectorBlock
          .updateGhostValuesEnd();
      }
    if (operations == HXChebyOperations::SinvXOnlyApplyVOnCconjTransXAssembly ||
        operations == HXChebyOperations::All)
      {
        tempVec.zeroOutGhosts();
        inverseMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(tempVec);
        d_pseudopotentialNonLocalOperator->applyVOnCconjtransX(
          CouplingStructure::dense,
          d_pseudopotentialClassPtr->getCouplingMatrix(
            CouplingType::inverseOverlapEntries),
          d_pseudopotentialNonLocalProjectorTimesVectorBlock,
          true,
          d_kPointIndex);
        for (dftfe::uInt iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));

            {
              d_cellWaveFunctionMatrixDst.setValue(0.0);
              d_pseudopotentialNonLocalOperator->applyCOnVCconjtransX(
                d_cellWaveFunctionMatrixDst.data(), cellRange);
              d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
                numberWavefunctions,
                numDoFsPerCell * (cellRange.second - cellRange.first),
                -1.0,
                d_basisOperationsPtr->cellInverseMassVectorBasisData().data() +
                  cellRange.first * numDoFsPerCell,
                d_cellWaveFunctionMatrixDst.data(),
                tempVec.data(),
                d_basisOperationsPtr
                    ->d_flattenedCellDofIndexToProcessDofIndexMap.data() +
                  cellRange.first * numDoFsPerCell);
            }
          }
        inverseMassVectorScaledConstraintsNoneDataInfoPtr
          ->distribute_slave_to_master(tempVec);
      }

    if (operations == HXChebyOperations::All)
      {
        tempVec.accumulateAddLocallyOwned();
        tempVec.zeroOutGhosts();
      }

    if (operations == HXChebyOperations::HXOnlyAxpy ||
        operations == HXChebyOperations::All)
      {
        d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * src.numVectors(),
                                scalarX,
                                src.data(),
                                scalarY,
                                dst.data());
      }


    if (operations == HXChebyOperations::All)
      {
        tempVec.updateGhostValues();
      }

    if (operations == HXChebyOperations::HXOnlyExtractionApplyConjTransOnX ||
        operations == HXChebyOperations::All)
      {
        d_basisOperationsPtr->distribute(tempVec);
        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          d_pseudopotentialNonLocalOperator->initialiseOperatorActionOnX(
            d_kPointIndex);
        for (dftfe::uInt iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
            d_BLASWrapperPtr->stridedCopyToBlock(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              tempVec.data(),
              d_cellWaveFunctionMatrixSrc.data() +
                (d_dftParamsPtr->memOptMode ?
                   0 :
                   cellRange.first * numDoFsPerCell * numberWavefunctions),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
            if (hasNonlocalComponents)
              d_pseudopotentialNonLocalOperator->applyCconjtransOnX(
                d_cellWaveFunctionMatrixSrc.data() +
                  (d_dftParamsPtr->memOptMode ?
                     0 :
                     cellRange.first * numDoFsPerCell * numberWavefunctions),
                cellRange);
          }
        d_pseudopotentialNonLocalProjectorTimesVectorBlock.setValue(0);
        d_pseudopotentialNonLocalOperator->applyAllReduceOnCconjtransX(
          d_pseudopotentialNonLocalProjectorTimesVectorBlock, true);
      }
    if (operations == HXChebyOperations::All)
      {
        d_pseudopotentialNonLocalProjectorTimesVectorBlock
          .accumulateAddLocallyOwnedBegin();
        d_pseudopotentialNonLocalProjectorTimesVectorBlock
          .accumulateAddLocallyOwnedEnd();
        d_pseudopotentialNonLocalProjectorTimesVectorBlock
          .updateGhostValuesBegin();
        d_pseudopotentialNonLocalProjectorTimesVectorBlock
          .updateGhostValuesEnd();
      }
    if (operations ==
          HXChebyOperations::HXOnlyHlocXApplyConVCconjTransXAssembly ||
        operations == HXChebyOperations::All)
      {
        if (!d_dftParamsPtr->memOptMode)
          {
            tempVec.zeroOutGhosts();
            inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(
              tempVec);
          }
        d_pseudopotentialNonLocalOperator->applyVOnCconjtransX(
          CouplingStructure::dense,
          d_pseudopotentialClassPtr->getCouplingMatrix(
            CouplingType::HamiltonianEntries),
          d_pseudopotentialNonLocalProjectorTimesVectorBlock,
          true);
        for (dftfe::uInt iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
            if (d_dftParamsPtr->memOptMode)
              {
                d_BLASWrapperPtr->stridedCopyToBlock(
                  numberWavefunctions,
                  numDoFsPerCell * (cellRange.second - cellRange.first),
                  tempVec.data(),
                  d_cellWaveFunctionMatrixSrc.data(),
                  d_basisOperationsPtr
                      ->d_flattenedCellDofIndexToProcessDofIndexMap.data() +
                    cellRange.first * numDoFsPerCell);
              }
            d_BLASWrapperPtr->xgemmStridedBatched(
              'N',
              'N',
              numberWavefunctions,
              numDoFsPerCell,
              numDoFsPerCell,
              &scalarCoeffAlpha,
              d_cellWaveFunctionMatrixSrc.data() +
                (d_dftParamsPtr->memOptMode ?
                   0 :
                   cellRange.first * numDoFsPerCell * numberWavefunctions),
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              d_cellHamiltonianMatrix[d_HamiltonianIndex].data() +
                cellRange.first * numDoFsPerCell * numDoFsPerCell,
              numDoFsPerCell,
              numDoFsPerCell * numDoFsPerCell,
              &scalarCoeffBeta,
              d_cellWaveFunctionMatrixDst.data(),
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              cellRange.second - cellRange.first);
            d_pseudopotentialNonLocalOperator->applyCOnVCconjtransX(
              d_cellWaveFunctionMatrixDst.data(), cellRange);
            d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              scalarHX,
              d_cellWaveFunctionMatrixDst.data(),
              dst.data(),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
          }
        if (d_dftParamsPtr->memOptMode)
          {
            tempVec.zeroOutGhosts();
            inverseMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(
              tempVec);
          }
        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
          .distribute_slave_to_master(dst);
      }
    if (operations == HXChebyOperations::All)
      {
        dst.accumulateAddLocallyOwned();
        dst.zeroOutGhosts();
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTGeneralisedEigenOperator<memorySpace>::
    overlapSqrtInverseMatrixTimesX(
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
      const double scalarOinvX,
      const double scalarY,
      const double scalarX,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst)
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTGeneralisedEigenOperator<memorySpace>::HXCheby(
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &src,
    const double scalarHX,
    const double scalarY,
    const double scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &dst,
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
                           &tempVec,
    const HXChebyOperations operations)
  {
    const dftfe::uInt numCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt numberWavefunctions = src.numVectors();
#if defined(DFTFE_WITH_DEVICE)
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        if (d_dftParamsPtr->tensorOpType == "TF32")
          d_BLASWrapperPtr->setTensorOpDataType(
            dftfe::linearAlgebra::tensorOpDataType::tf32);
        if (d_dftParamsPtr->tensorOpType == "BF16")
          d_BLASWrapperPtr->setTensorOpDataType(
            dftfe::linearAlgebra::tensorOpDataType::bf16);
      }
#endif
    if (d_numVectorsInternal != numberWavefunctions)
      reinitNumberWavefunctions(numberWavefunctions);

    if (d_basisOperationsPtr->d_nVectors != numberWavefunctions)
      d_basisOperationsPtr->reinit(numberWavefunctions,
                                   d_cellsBlockSizeHX,
                                   d_densityQuadratureID,
                                   false,
                                   false);
    const bool hasNonlocalComponents =
      d_dftParamsPtr->isPseudopotential &&
      (d_pseudopotentialNonLocalOperatorSinglePrec
         ->getTotalNonLocalElementsInCurrentProcessor() > 0);
    const dataTypes::numberFP32 scalarCoeffAlpha = dataTypes::numberFP32(1.0),
                                scalarCoeffBeta  = dataTypes::numberFP32(0.0);

    const dftfe::uInt blockSize = src.numVectors();
    // dftfe::utils::deviceSynchronize();
    // computing_timer.enter_subsection("HXChebySP: SlocinvX");
    if (operations == HXChebyOperations::SinvXOnlySinvLocX ||
        operations == HXChebyOperations::All)
      {
        d_BLASWrapperPtr->stridedBlockAxpBy(
          blockSize,
          src.locallyOwnedSize(),
          src.data(),
          d_basisOperationsPtr->inverseMassVectorBasisData().data(),
          1.0,
          0.0,
          tempVec.data());
      }
    // dftfe::utils::deviceSynchronize();
    // computing_timer.leave_subsection("HXChebySP: SlocinvX");

    // dftfe::utils::deviceSynchronize();
    // computing_timer.enter_subsection(
    //   "HXChebySP: tempVecUpdateGhostValues SinvX");
    if (operations == HXChebyOperations::All)
      {
        tempVec.updateGhostValues();
      }
    // dftfe::utils::deviceSynchronize();
    // computing_timer.leave_subsection(
    //   "HXChebySP: tempVecUpdateGhostValues SinvX");

    // dftfe::utils::deviceSynchronize();
    // computing_timer.enter_subsection(
    //   "HXChebySP: extraction of tempVec and CconjTransX SinvX");
    if (operations == HXChebyOperations::SinvXOnlyExtractionApplyConjTransOnX ||
        operations == HXChebyOperations::All)
      {
        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
          .distribute(tempVec);
        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          if (d_dftParamsPtr->isPseudopotential)
            d_pseudopotentialNonLocalOperatorSinglePrec
              ->initialiseOperatorActionOnX(d_kPointIndex);
        for (dftfe::uInt iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
            d_BLASWrapperPtr->stridedCopyToBlock(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              tempVec.data(),
              d_cellWaveFunctionMatrixSrcSinglePrec.data() +
                (d_dftParamsPtr->memOptMode ?
                   0 :
                   cellRange.first * numDoFsPerCell * numberWavefunctions),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
            d_pseudopotentialNonLocalOperatorSinglePrec->applyCconjtransOnX(
              d_cellWaveFunctionMatrixSrcSinglePrec.data() +
                (d_dftParamsPtr->memOptMode ?
                   0 :
                   cellRange.first * numDoFsPerCell * numberWavefunctions),
              cellRange);
          }
        d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec.setValue(
          0);
        d_pseudopotentialNonLocalOperatorSinglePrec
          ->applyAllReduceOnCconjtransX(
            d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec, true);
      }
    // dftfe::utils::deviceSynchronize();
    // computing_timer.leave_subsection(
    //   "HXChebySP: extraction of tempVec and CconjTransX SinvX");


    if (operations == HXChebyOperations::All)
      {
        // dftfe::utils::deviceSynchronize();
        // computing_timer.enter_subsection("HXChebySP: nonLocal AllReduce
        // SinvX");
        d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
          .accumulateAddLocallyOwnedBegin();
        d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
          .accumulateAddLocallyOwnedEnd();
        d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
          .updateGhostValuesBegin();
        d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
          .updateGhostValuesEnd();
        // dftfe::utils::deviceSynchronize();
        // computing_timer.leave_subsection("HXChebySP: nonLocal AllReduce
        // SinvX");
      }



    if (operations == HXChebyOperations::SinvXOnlyApplyVOnCconjTransXAssembly ||
        operations == HXChebyOperations::All)
      {
        // dftfe::utils::deviceSynchronize();
        // computing_timer.enter_subsection(
        //   "HXChebySP: applyVOnCconjtransX SinvX");
        tempVec.zeroOutGhosts();
        inverseMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(tempVec);
        d_pseudopotentialNonLocalOperatorSinglePrec->applyVOnCconjtransX(
          CouplingStructure::dense,
          d_pseudopotentialClassPtr->getCouplingMatrixSinglePrec(
            CouplingType::inverseOverlapEntries),
          d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec,
          true,
          d_kPointIndex);
        // dftfe::utils::deviceSynchronize();
        // computing_timer.leave_subsection(
        //   "HXChebySP: applyVOnCconjtransX SinvX");
        // dftfe::utils::deviceSynchronize();
        // computing_timer.enter_subsection(
        //   "HXChebySP: applyCOnVCconjtransX and Assembly SinvX");
        for (dftfe::uInt iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
            {
              d_cellWaveFunctionMatrixDstSinglePrec.setValue(0);
              d_pseudopotentialNonLocalOperatorSinglePrec->applyCOnVCconjtransX(
                d_cellWaveFunctionMatrixDstSinglePrec.data(), cellRange);
              d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
                numberWavefunctions,
                numDoFsPerCell * (cellRange.second - cellRange.first),
                -1.0,
                d_basisOperationsPtr->cellInverseMassVectorBasisData().data() +
                  cellRange.first * numDoFsPerCell,
                d_cellWaveFunctionMatrixDstSinglePrec.data(),
                tempVec.data(),
                d_basisOperationsPtr
                    ->d_flattenedCellDofIndexToProcessDofIndexMap.data() +
                  cellRange.first * numDoFsPerCell);
            }
          }
        inverseMassVectorScaledConstraintsNoneDataInfoPtr
          ->distribute_slave_to_master(tempVec);
        // dftfe::utils::deviceSynchronize();
        // computing_timer.leave_subsection(
        //   "HXChebySP: applyCOnVCconjtransX and Assembly SinvX");
      }


    // dftfe::utils::deviceSynchronize();
    // computing_timer.enter_subsection("HXChebySP: accumulateAdd SinvX");
    if (operations == HXChebyOperations::All)
      {
        tempVec.accumulateAddLocallyOwned();
        tempVec.zeroOutGhosts();
      }
    // dftfe::utils::deviceSynchronize();
    // computing_timer.leave_subsection("HXChebySP: accumulateAdd SinvX");

    // dftfe::utils::deviceSynchronize();
    // computing_timer.enter_subsection("HXChebySP: axpBY HX");
    if (operations == HXChebyOperations::HXOnlyAxpy ||
        operations == HXChebyOperations::All)
      {
        d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * src.numVectors(),
                                scalarX,
                                src.data(),
                                scalarY,
                                dst.data());
      }
    // dftfe::utils::deviceSynchronize();
    // computing_timer.leave_subsection("HXChebySP: axpBY HX");

    // dftfe::utils::deviceSynchronize();
    // computing_timer.enter_subsection("HXChebySP: tempVecUpdateGhost HX");
    if (operations == HXChebyOperations::All)
      {
        tempVec.updateGhostValues();
      }
    // dftfe::utils::deviceSynchronize();
    // computing_timer.leave_subsection("HXChebySP: tempVecUpdateGhost HX");

    // dftfe::utils::deviceSynchronize();
    // computing_timer.enter_subsection(
    //   "HXChebySP: extraction and CconjtransX HX");
    if (operations == HXChebyOperations::HXOnlyExtractionApplyConjTransOnX ||
        operations == HXChebyOperations::All)
      {
        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
          .distribute(tempVec);

        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          d_pseudopotentialNonLocalOperatorSinglePrec
            ->initialiseOperatorActionOnX(d_kPointIndex);
        for (dftfe::uInt iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
            d_BLASWrapperPtr->stridedCopyToBlock(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              tempVec.data(),
              d_cellWaveFunctionMatrixSrcSinglePrec.data() +
                (d_dftParamsPtr->memOptMode ?
                   0 :
                   cellRange.first * numDoFsPerCell * numberWavefunctions),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
            if (hasNonlocalComponents)
              d_pseudopotentialNonLocalOperatorSinglePrec->applyCconjtransOnX(
                d_cellWaveFunctionMatrixSrcSinglePrec.data() +
                  (d_dftParamsPtr->memOptMode ?
                     0 :
                     cellRange.first * numDoFsPerCell * numberWavefunctions),
                cellRange);
          }
        d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec.setValue(
          0);
        d_pseudopotentialNonLocalOperatorSinglePrec
          ->applyAllReduceOnCconjtransX(
            d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec, true);
      }
    // dftfe::utils::deviceSynchronize();
    // computing_timer.leave_subsection(
    //   "HXChebySP: extraction and CconjtransX HX");


    if (operations == HXChebyOperations::All)
      {
        // dftfe::utils::deviceSynchronize();
        // computing_timer.enter_subsection("HXChebySP: AllReduce HX");

        d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
          .accumulateAddLocallyOwnedBegin();
        d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
          .accumulateAddLocallyOwnedEnd();
        d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
          .updateGhostValuesBegin();
        d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
          .updateGhostValuesEnd();
        // dftfe::utils::deviceSynchronize();
        // computing_timer.leave_subsection("HXChebySP: AllReduce HX");
      }

    if (operations ==
          HXChebyOperations::HXOnlyHlocXApplyConVCconjTransXAssembly ||
        operations == HXChebyOperations::All)
      {
        // dftfe::utils::deviceSynchronize();
        // computing_timer.enter_subsection("HXChebySP: applyVonCtransX HX");
        if (!d_dftParamsPtr->memOptMode)
          {
            tempVec.zeroOutGhosts();
            inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(
              tempVec);
          }


        d_pseudopotentialNonLocalOperatorSinglePrec->applyVOnCconjtransX(
          CouplingStructure::dense,
          d_pseudopotentialClassPtr->getCouplingMatrixSinglePrec(
            CouplingType::HamiltonianEntries),
          d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec,
          true);

        // dftfe::utils::deviceSynchronize();
        // computing_timer.leave_subsection("HXChebySP: applyVonCtransX HX");

        // dftfe::utils::deviceSynchronize();
        // computing_timer.enter_subsection("HXChebySP: HlocX and assembly HX");
        for (dftfe::uInt iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
            if (d_dftParamsPtr->memOptMode)
              {
                d_BLASWrapperPtr->stridedCopyToBlock(
                  numberWavefunctions,
                  numDoFsPerCell * (cellRange.second - cellRange.first),
                  tempVec.data(),
                  d_cellWaveFunctionMatrixSrcSinglePrec.data(),
                  d_basisOperationsPtr
                      ->d_flattenedCellDofIndexToProcessDofIndexMap.data() +
                    cellRange.first * numDoFsPerCell);
              }
            d_BLASWrapperPtr->xgemmStridedBatched(
              'N',
              'N',
              numberWavefunctions,
              numDoFsPerCell,
              numDoFsPerCell,
              &scalarCoeffAlpha,
              d_cellWaveFunctionMatrixSrcSinglePrec.data() +
                (d_dftParamsPtr->memOptMode ?
                   0 :
                   cellRange.first * numDoFsPerCell * numberWavefunctions),
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              d_cellHamiltonianMatrixSinglePrec[d_HamiltonianIndex].data() +
                cellRange.first * numDoFsPerCell * numDoFsPerCell,
              numDoFsPerCell,
              numDoFsPerCell * numDoFsPerCell,
              &scalarCoeffBeta,
              d_cellWaveFunctionMatrixDstSinglePrec.data(),
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              cellRange.second - cellRange.first);
            d_pseudopotentialNonLocalOperatorSinglePrec->applyCOnVCconjtransX(
              d_cellWaveFunctionMatrixDstSinglePrec.data(), cellRange);
            d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              scalarHX,
              d_cellWaveFunctionMatrixDstSinglePrec.data(),
              dst.data(),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
          }
        if (d_dftParamsPtr->memOptMode)
          {
            tempVec.zeroOutGhosts();
            inverseMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(
              tempVec);
          }
        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
          .distribute_slave_to_master(dst);
        // dftfe::utils::deviceSynchronize();
        // computing_timer.leave_subsection("HXChebySP: HlocX and assembly HX");
      }

    // dftfe::utils::deviceSynchronize();
    // computing_timer.enter_subsection("HXChebySP: dst accumulate Add HX");

    if (operations == HXChebyOperations::All)
      {
        dst.accumulateAddLocallyOwned();
        dst.zeroOutGhosts();
      }
      // dftfe::utils::deviceSynchronize();
      // computing_timer.leave_subsection("HXChebySP: dst accumulate Add HX");
#if defined(DFTFE_WITH_DEVICE)
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      d_BLASWrapperPtr->setTensorOpDataType(
        dftfe::linearAlgebra::tensorOpDataType::fp32);
#endif
  }
  template class KohnShamDFTGeneralisedEigenOperator<
    dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class KohnShamDFTGeneralisedEigenOperator<
    dftfe::utils::MemorySpace::DEVICE>;
#endif


} // namespace dftfe
