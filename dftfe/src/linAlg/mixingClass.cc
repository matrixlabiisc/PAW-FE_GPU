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
// @author Vishal Subramanian
//

#include <mixingClass.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>

#include <cmath>

namespace dftfe
{
  MixingScheme::MixingScheme(
    const MPI_Comm   &mpi_comm_parent,
    const MPI_Comm   &mpi_comm_domain,
    const dftfe::uInt verbosity,
    const std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      &blasWrapperHost)
    : d_mpi_comm_domain(mpi_comm_domain)
    , d_mpi_comm_parent(mpi_comm_parent)
    , d_blasWrapperHostPtr(blasWrapperHost)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_verbosity(verbosity)
  {
    AssertThrow(
      blasWrapperHost != nullptr,
      dealii::ExcMessage(
        "DFT-FE Error: MixingScheme requires a non-null host BLASWrapper."));
  }

  void
  MixingScheme::addMixingVariable(
    const mixingVariable mixingVariableList,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                &weightDotProducts,
    const bool   performMPIReduce,
    const double mixingValue,
    const bool   adaptMixingValue)
  {
    d_variableHistoryIn[mixingVariableList] = std::deque<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>();
    d_variableHistoryResidual[mixingVariableList] = std::deque<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>();
    d_vectorDotProductWeights[mixingVariableList] = weightDotProducts;

    d_sqrtVectorDotProductWeights[mixingVariableList].resize(
      weightDotProducts.size());
    for (dftfe::uInt q = 0; q < weightDotProducts.size(); ++q)
      {
        const double wq = weightDotProducts[q];
        d_sqrtVectorDotProductWeights[mixingVariableList][q] =
          (wq > 0.0) ? std::sqrt(wq) : 0.0;
      }

    d_performMPIReduce[mixingVariableList]     = performMPIReduce;
    d_mixingParameter[mixingVariableList]      = mixingValue;
    d_adaptMixingParameter[mixingVariableList] = adaptMixingValue;
    d_anyMixingParameterAdaptive =
      adaptMixingValue || d_anyMixingParameterAdaptive;
    d_adaptiveMixingParameterDecLastIteration = false;
    d_adaptiveMixingParameterDecAllIterations = true;
    d_adaptiveMixingParameterIncAllIterations = true;
    dftfe::uInt weightDotProductsSize         = weightDotProducts.size();
    MPI_Allreduce(MPI_IN_PLACE,
                  &weightDotProductsSize,
                  1,
                  dftfe::dataTypes::mpi_type_id(&weightDotProductsSize),
                  MPI_MAX,
                  d_mpi_comm_domain);
    if (weightDotProductsSize > 0)
      {
        d_performMixing[mixingVariableList] = true;
      }
    else
      {
        d_performMixing[mixingVariableList] = false;
      }
  }

  void
  MixingScheme::computeMixingMatrices(
    const std::deque<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &inHist,
    const std::deque<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &residualHist,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &weightDotProducts,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                        &sqrtWeightDotProducts,
    const bool           isPerformMixing,
    const bool           isMPIAllReduce,
    std::vector<double> &A,
    std::vector<double> &c)
  {
    std::vector<double> Adensity;
    Adensity.resize(A.size());
    std::fill(Adensity.begin(), Adensity.end(), 0.0);

    std::vector<double> cDensity;
    cDensity.resize(c.size());
    std::fill(cDensity.begin(), cDensity.end(), 0.0);

    dftfe::Int  N             = inHist.size() - 1;
    dftfe::uInt numQuadPoints = 0;
    if (N > 0)
      numQuadPoints = inHist[0].size();

    if (isPerformMixing)
      {
        AssertThrow(numQuadPoints == weightDotProducts.size(),
                    dealii::ExcMessage(
                      "DFT-FE Error: The size of the weight dot products vec "
                      "does not match the size of the vectors in history."
                      "Please resize the vectors appropriately."));
        AssertThrow(
          numQuadPoints == sqrtWeightDotProducts.size(),
          dealii::ExcMessage(
            "DFT-FE Error: sqrt(weight) vector size does not match "
            "weight dot products (must be set in addMixingVariable)."));
        if (N > 0 && numQuadPoints > 0)
          {
            // Weighted Gram matrix G = X^T X and c = X^T y with
            // X_{qm} = sqrt(w_q) * (F_n - F_{n-1-m})_q
            // y_q = sqrt(w_q) * (F_n)_q
            // (column-major X: numQuadPoints rows, N columns, lda =
            // numQuadPoints). sqrt(w) is precomputed in addMixingVariable
            // (d_sqrtVectorDotProductWeights).
            d_mixingBlasWeightedFn.resize(numQuadPoints);
            d_mixingBlasMatrixX.resize(static_cast<size_t>(numQuadPoints) *
                                       static_cast<size_t>(N));

            const double *const Fn      = residualHist[N].data();
            const double *const sqrtW   = sqrtWeightDotProducts.data();
            const dftfe::uInt   nqp     = numQuadPoints;
            const double        neg_one = -1.0;

            d_blasWrapperHostPtr->hadamardProduct(
              nqp, sqrtW, Fn, d_mixingBlasWeightedFn.data());

            for (dftfe::Int m = 0; m < N; ++m)
              {
                const double *const Fprev = residualHist[N - 1 - m].data();
                double *const       col =
                  d_mixingBlasMatrixX.data() +
                  static_cast<size_t>(m) * static_cast<size_t>(numQuadPoints);
                d_blasWrapperHostPtr->xcopy(nqp, Fn, 1, col, 1);
                d_blasWrapperHostPtr->xaxpy(nqp, &neg_one, Fprev, 1, col, 1);
                d_blasWrapperHostPtr->hadamardProduct(nqp, sqrtW, col, col);
              }

            const double       alpha  = 1.0;
            const double       beta   = 0.0;
            const unsigned int nHist  = static_cast<unsigned int>(N);
            const unsigned int kDim   = numQuadPoints;
            const unsigned int lda    = numQuadPoints;
            const unsigned int ldc    = nHist;
            unsigned int       incx   = 1;
            unsigned int       incy   = 1;
            char               transA = 'T';
            char               transB = 'N';
            char               transV = 'T';

            // Adensity = X^T * X (Fortran column-major layout, same as dgesv).
            dgemm_(&transA,
                   &transB,
                   &nHist,
                   &nHist,
                   &kDim,
                   &alpha,
                   d_mixingBlasMatrixX.data(),
                   &lda,
                   d_mixingBlasMatrixX.data(),
                   &lda,
                   &beta,
                   Adensity.data(),
                   &ldc);

            // cDensity = X^T * (sqrt(w) ⊙ F_n)
            const unsigned int mDgemv = numQuadPoints;
            const unsigned int nDgemv = nHist;
            dgemv_(&transV,
                   &mDgemv,
                   &nDgemv,
                   &alpha,
                   d_mixingBlasMatrixX.data(),
                   &lda,
                   d_mixingBlasWeightedFn.data(),
                   &incx,
                   &beta,
                   cDensity.data(),
                   &incy);
          }

        dftfe::uInt aSize = Adensity.size();
        dftfe::uInt cSize = cDensity.size();

        std::vector<double> ATotal(aSize), cTotal(cSize);
        std::fill(ATotal.begin(), ATotal.end(), 0.0);
        std::fill(cTotal.begin(), cTotal.end(), 0.0);
        if (isMPIAllReduce)
          {
            if (aSize > 0)
              MPI_Allreduce(&Adensity[0],
                            &ATotal[0],
                            aSize,
                            MPI_DOUBLE,
                            MPI_SUM,
                            d_mpi_comm_domain);
            if (cSize > 0)
              MPI_Allreduce(&cDensity[0],
                            &cTotal[0],
                            cSize,
                            MPI_DOUBLE,
                            MPI_SUM,
                            d_mpi_comm_domain);
          }
        else
          {
            ATotal = Adensity;
            cTotal = cDensity;
          }
        for (dftfe::uInt i = 0; i < aSize; i++)
          {
            A[i] += ATotal[i];
          }

        for (dftfe::uInt i = 0; i < cSize; i++)
          {
            c[i] += cTotal[i];
          }
      }
  }

  dftfe::uInt
  MixingScheme::lengthOfHistory()
  {
    return d_variableHistoryIn[mixingVariable::rho].size();
  }

  // Fucntion to compute the mixing coefficients based on anderson scheme
  void
  MixingScheme::computeAndersonMixingCoeff(
    const std::vector<mixingVariable> mixingVariablesList)
  {
    // initialize data structures
    // assumes rho is a mixing variable
    int N = d_variableHistoryIn[mixingVariable::rho].size() - 1;
    MPI_Barrier(d_mpi_comm_parent);
    double startTime = MPI_Wtime();

    if (N > 0)
      {
        int              NRHS = 1, lda = N, ldb = N, info;
        std::vector<int> ipiv(N);
        d_A.resize(lda * N);
        d_c.resize(ldb * NRHS);
        for (dftfe::Int i = 0; i < lda * N; i++)
          d_A[i] = 0.0;
        for (dftfe::Int i = 0; i < ldb * NRHS; i++)
          d_c[i] = 0.0;

        for (const auto &key : mixingVariablesList)
          {
            computeMixingMatrices(d_variableHistoryIn[key],
                                  d_variableHistoryResidual[key],
                                  d_vectorDotProductWeights[key],
                                  d_sqrtVectorDotProductWeights[key],
                                  d_performMixing[key],
                                  d_performMPIReduce[key],
                                  d_A,
                                  d_c);
          }

        dgesv_(&N, &NRHS, &d_A[0], &lda, &ipiv[0], &d_c[0], &ldb, &info);
      }
    MPI_Barrier(d_mpi_comm_parent);
    double dt    = MPI_Wtime() - startTime;
    double dtMax = 0.0;
    // MPI_Allreduce(&dt,
    //               &dtMax,
    //               1,
    //               dftfe::dataTypes::mpi_type_id(&dt),
    //               MPI_MAX,
    //               d_mpi_comm_parent);
    //pcout << "Timer for computeAndersonMixingCoeff: " << dtMax << std::endl;

    d_cFinal = 1.0;
    for (dftfe::Int i = 0; i < N; i++)
      d_cFinal -= d_c[i];
    computeAdaptiveAndersonMixingParameter();
  }


  // Fucntion to compute the mixing parameter based on an adaptive anderson
  // scheme, algorithm 1 in [CPC. 292, 108865 (2023)]
  void
  MixingScheme::computeAdaptiveAndersonMixingParameter()
  {
    double ci = 1.0;
    if (d_anyMixingParameterAdaptive &&
        d_variableHistoryIn[mixingVariable::rho].size() > 1)
      {
        double bii   = std::abs(d_cFinal);
        double gbase = 1.0;
        double gpv   = 0.02;
        double ggap  = 0.0;
        double gi =
          gpv * ((double)d_variableHistoryIn[mixingVariable::rho].size()) +
          gbase;
        double x = std::abs(bii) / gi;
        if (x < 0.5)
          ci = 1.0 / (2.0 + std::log(0.5 / x));
        else if (x <= 2.0)
          ci = x;
        else
          ci = 2.0 + std::log(x / 2.0);
        double pi = 0.0;
        if (ci < 1.0 == d_adaptiveMixingParameterDecLastIteration)
          if (ci < 1.0)
            if (d_adaptiveMixingParameterDecAllIterations)
              pi = 1.0;
            else
              pi = 2.0;
          else if (d_adaptiveMixingParameterIncAllIterations)
            pi = 1.0;
          else
            pi = 2.0;
        else
          pi = 3.0;

        ci                                        = std::pow(ci, 1.0 / pi);
        d_adaptiveMixingParameterDecLastIteration = ci < 1.0;
        d_adaptiveMixingParameterDecAllIterations =
          d_adaptiveMixingParameterDecAllIterations & ci < 1.0;
        d_adaptiveMixingParameterIncAllIterations =
          d_adaptiveMixingParameterIncAllIterations & ci >= 1.0;
      }
    MPI_Bcast(&ci, 1, MPI_DOUBLE, 0, d_mpi_comm_parent);
    for (const auto &[key, value] : d_variableHistoryIn)
      if (d_adaptMixingParameter[key])
        {
          d_mixingParameter[key] *= ci;
        }
    if (d_verbosity > 0)
      for (const auto &[key, value] : d_variableHistoryIn)
        {
          if (key == mixingVariable::rho &&
              d_adaptMixingParameter[mixingVariable::rho])
            pcout << "Adaptive Anderson mixing parameter for Rho: "
                  << d_mixingParameter[mixingVariable::rho] << std::endl;
          else if (key == mixingVariable::rho)
            pcout << "Anderson mixing parameter for Rho: "
                  << d_mixingParameter[mixingVariable::rho] << std::endl;
          if (key == mixingVariable::magZ &&
              d_adaptMixingParameter[mixingVariable::magZ])
            pcout << "Adaptive Anderson mixing parameter for magZ: "
                  << d_mixingParameter[mixingVariable::magZ] << std::endl;
          else if (key == mixingVariable::magZ)
            pcout << "Anderson mixing parameter for magZ: "
                  << d_mixingParameter[mixingVariable::magZ] << std::endl;
        }
  }

  // Fucntions to add to the history
  void
  MixingScheme::addVariableToInHist(const mixingVariable mixingVariableName,
                                    const double        *inputVariableToInHist,
                                    const dftfe::uInt    length)
  {
    d_variableHistoryIn[mixingVariableName].push_back(
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
        length));
    std::memcpy(d_variableHistoryIn[mixingVariableName].back().data(),
                inputVariableToInHist,
                length * sizeof(double));
  }

  void
  MixingScheme::addVariableToResidualHist(
    const mixingVariable mixingVariableName,
    const double        *inputVariableToResidualHist,
    const dftfe::uInt    length)
  {
    d_variableHistoryResidual[mixingVariableName].push_back(
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
        length));
    std::memcpy(d_variableHistoryResidual[mixingVariableName].back().data(),
                inputVariableToResidualHist,
                length * sizeof(double));
  }

  // Computes the new variable after mixing.
  void
  MixingScheme::mixVariable(mixingVariable    mixingVariableName,
                            double           *outputVariable,
                            const dftfe::uInt lenVar)
  {
    dftfe::uInt N = d_variableHistoryIn[mixingVariableName].size() - 1;
    // Assumes the variable is present otherwise will lead to a seg fault
    AssertThrow(
      lenVar == d_variableHistoryIn[mixingVariableName][0].size(),
      dealii::ExcMessage(
        "DFT-FE Error: The size of the input variables in history does not match the provided size."));

    const double mixingParam = d_mixingParameter[mixingVariableName];
    const double cFinal      = d_cFinal;

    // output = inBar + mixingParam * residualBar
    // inBar = cFinal * in[N] + sum_i c[i] * in[N-1-i]
    // residualBar = cFinal * res[N] + sum_i c[i] * res[N-1-i]
    d_blasWrapperHostPtr->xcopy(
      lenVar,
      d_variableHistoryIn[mixingVariableName][N].data(),
      1,
      outputVariable,
      1);
    d_blasWrapperHostPtr->xscal(outputVariable, cFinal, lenVar);

    d_mixingBlasTemp.resize(lenVar);
    d_blasWrapperHostPtr->xcopy(
      lenVar,
      d_variableHistoryResidual[mixingVariableName][N].data(),
      1,
      d_mixingBlasTemp.data(),
      1);
    d_blasWrapperHostPtr->xscal(d_mixingBlasTemp.data(), cFinal, lenVar);

    for (dftfe::Int i = 0; i < static_cast<dftfe::Int>(N); ++i)
      {
        const double ci = d_c[i];
        d_blasWrapperHostPtr->xaxpy(
          lenVar,
          &ci,
          d_variableHistoryIn[mixingVariableName][N - 1 - i].data(),
          1,
          outputVariable,
          1);
        d_blasWrapperHostPtr->xaxpy(
          lenVar,
          &ci,
          d_variableHistoryResidual[mixingVariableName][N - 1 - i].data(),
          1,
          d_mixingBlasTemp.data(),
          1);
      }

    d_blasWrapperHostPtr->xaxpy(
      lenVar, &mixingParam, d_mixingBlasTemp.data(), 1, outputVariable, 1);
  }

  void
  MixingScheme::getOptimizedResidual(mixingVariable    mixingVariableName,
                                     double           *outputVariable,
                                     const dftfe::uInt lenVar)
  {
    dftfe::uInt N = d_variableHistoryIn[mixingVariableName].size() - 1;
    // Assumes the variable is present otherwise will lead to a seg fault
    AssertThrow(
      lenVar == d_variableHistoryIn[mixingVariableName][0].size(),
      dealii::ExcMessage(
        "DFT-FE Error: The size of the input variables in history does not match the provided size."));

    const double cFinal = d_cFinal;
    d_blasWrapperHostPtr->xcopy(
      lenVar,
      d_variableHistoryResidual[mixingVariableName][N].data(),
      1,
      outputVariable,
      1);
    d_blasWrapperHostPtr->xscal(outputVariable, cFinal, lenVar);

    for (dftfe::Int i = 0; i < static_cast<dftfe::Int>(N); ++i)
      {
        const double ci = d_c[i];
        d_blasWrapperHostPtr->xaxpy(
          lenVar,
          &ci,
          d_variableHistoryResidual[mixingVariableName][N - 1 - i].data(),
          1,
          outputVariable,
          1);
      }
  }

  void
  MixingScheme::mixPreconditionedResidual(mixingVariable    mixingVariableName,
                                          double           *inputVariable,
                                          double           *outputVariable,
                                          const dftfe::uInt lenVar)
  {
    dftfe::uInt N = d_variableHistoryIn[mixingVariableName].size() - 1;
    // Assumes the variable is present otherwise will lead to a seg fault
    AssertThrow(
      lenVar == d_variableHistoryIn[mixingVariableName][0].size(),
      dealii::ExcMessage(
        "DFT-FE Error: The size of the input variables in history does not match the provided size."));

    const double mixingParam = d_mixingParameter[mixingVariableName];
    const double cFinal      = d_cFinal;

    d_blasWrapperHostPtr->xcopy(
      lenVar,
      d_variableHistoryIn[mixingVariableName][N].data(),
      1,
      outputVariable,
      1);
    d_blasWrapperHostPtr->xscal(outputVariable, cFinal, lenVar);

    for (dftfe::Int i = 0; i < static_cast<dftfe::Int>(N); ++i)
      {
        const double ci = d_c[i];
        d_blasWrapperHostPtr->xaxpy(
          lenVar,
          &ci,
          d_variableHistoryIn[mixingVariableName][N - 1 - i].data(),
          1,
          outputVariable,
          1);
      }

    d_blasWrapperHostPtr->xaxpy(
      lenVar, &mixingParam, inputVariable, 1, outputVariable, 1);
  }

  // Clears the history
  // But it does not clear the list of variables
  // and its corresponding JxW values
  void
  MixingScheme::clearHistory()
  {
    for (const auto &[key, value] : d_variableHistoryIn)
      {
        d_variableHistoryIn[key].clear();
        d_variableHistoryResidual[key].clear();
      }
  }


  // Deletes old history.
  // This is not recursively
  // If the length is greater or equal to mixingHistory then the
  // oldest history is deleted
  void
  MixingScheme::popOldHistory(dftfe::uInt mixingHistory)
  {
    if (d_variableHistoryIn[mixingVariable::rho].size() > mixingHistory)
      {
        for (const auto &[key, value] : d_variableHistoryIn)
          {
            d_variableHistoryIn[key].pop_front();
            d_variableHistoryResidual[key].pop_front();
          }
      }
  }

} // namespace dftfe
