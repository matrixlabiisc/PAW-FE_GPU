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
namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::gaunt(dftfe::Int l_i,
                                          dftfe::Int l_j,
                                          dftfe::Int l,
                                          dftfe::Int m_i,
                                          dftfe::Int m_j,
                                          dftfe::Int m)
  {
    bool flagm      = !(m == (m_i + m_j) || m == m_i - m_j || m == -m_i + m_j ||
                   m == -m_i - m_j);
    dftfe::Int lmax = l_i + l_j;
    dftfe::Int k    = std::max(std::abs(l_i - l_j),
                            std::min(std::abs(m_i + m_j), std::abs(m_i - m_j)));
    dftfe::Int lmin = (k + lmax) % 2 == 0 ? k : k + 1;
    /*if (flagm || ((l_i + l_j + l) % 2 == 1) ||
        l < lmin || l > lmax)
      {
        return 0.0;
      } */


    dftfe::Int flag1 = m_i == 0 ? 0 : 1;
    dftfe::Int flag2 = m_j == 0 ? 0 : 1;
    dftfe::Int flag3 = m == 0 ? 0 : 1;
    dftfe::Int flag  = flag1 + flag2 + flag3;

    if (flag < 2)
      {
        double gauntvalue = wigner::gaunt<double>(l_i, l_j, l, m_i, m_j, m);
        // double gauntvalue = gauntcomplex(l_i, l_j, l, m_i, m_j, m);
        if (flag == 0)
          {
            return gauntvalue;
          }
        else
          return 0.0;
      }

    if (flag == 3)
      {
        std::complex<double> U1 =
          (computeTransformationExtries(l, m, -(m_i + m_j))) *
          computeTransformationExtries(l_i, m_i, m_i) *
          computeTransformationExtries(l_j, m_j, m_j);
        std::complex<double> U2 =
          (computeTransformationExtries(l, m, -(m_i - m_j))) *
          computeTransformationExtries(l_i, m_i, m_i) *
          computeTransformationExtries(l_j, m_j, -m_j);

        double value =
          2 * U1.real() *
            wigner::gaunt<double>(l_i, l_j, l, m_i, m_j, -(m_i + m_j)) +
          2 * U2.real() *
            wigner::gaunt<double>(l_i, l_j, l, m_i, -m_j, -(m_i - m_j));
        return (value);
      }
    if (flag == 2)
      {
        dftfe::Int l1, l2, l3, m1, m2;
        if (flag1 == 0)
          {
            l3 = l_i;
            l2 = l_j;
            l1 = l;
            m2 = m_j;
            m1 = m;
          }
        else if (flag2 == 0)
          {
            l3 = l_j;
            l2 = l_i;
            l1 = l;
            m2 = m_i;
            m1 = m;
          }
        else
          {
            l3 = l;
            l2 = l_i;
            l1 = l_j;
            m2 = m_i;
            m1 = m_j;
          }
        std::complex<double> U = (computeTransformationExtries(l1, m1, -m2)) *
                                 computeTransformationExtries(l2, m2, m2);
        double value =
          2 * U.real() * wigner::gaunt<double>(l3, l2, l1, 0, m2, -m2);
        return (value);
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::complex<double>
  pawClass<ValueType, memorySpace>::computeTransformationExtries(dftfe::Int l,
                                                                 dftfe::Int mu,
                                                                 dftfe::Int m)
  {
    std::complex<double> U(0.0, 0.0);
    dftfe::Int           delta_mu0deltam0 = (m == 0 && mu == 0) ? 1 : 0;
    U.real(delta_mu0deltam0 +
           1 / sqrt(2) *
             ((mu > 0 ? 1 : 0) * (m == mu ? 1 : 0) +
              (mu > 0 ? 1 : 0) * pow(-1, m) * (m == -mu ? 1 : 0)));
    U.imag(1 / sqrt(2) *
           ((-mu > 0 ? 1 : 0) * pow(-1, m) * (m == mu ? 1 : 0) -
            (-mu > 0 ? 1 : 0) * (m == -mu ? 1 : 0)));
    return (U);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::threeTermIntegrationOverAugmentationSphere(
    double              *f1,
    double              *f2,
    double              *f3,
    std::vector<double> &radial,
    std::vector<double> &rab,
    const dftfe::uInt    rminIndex,
    const dftfe::uInt    rmaxIndex)
  {
    double                                     IntOut = 0.0;
    std::function<double(const dftfe::uInt &)> integrationValue =
      [&](const dftfe::uInt &i) {
        double Value = rab[i] * f3[i] * f2[i] * f1[i] * radial[i] * radial[i];
        return (Value);
      };
    double Q1 = simpsonIntegral(rminIndex, rmaxIndex, integrationValue);
    IntOut += Q1;
    return (IntOut);
  }
  // template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  // double
  // pawClass<ValueType, memorySpace>::simpsonIntegral(
  //   const dftfe::uInt                           startIndex,
  //   const dftfe::uInt                           endIndex,
  //   std::function<double(const dftfe::uInt &)> &IntegrandValue)
  // {
  //   const dftfe::uInt endIndex_2 = endIndex - startIndex;
  //   if (endIndex_2 == 0)
  //     return 0.0;
  //   else if (endIndex_2 == 1)
  //     return 0.5 * (IntegrandValue(startIndex) + IntegrandValue(EndIndex));
  //   else
  //     {
  //       std::vector<double> simfact(endIndex_2, 0.0);
  //       simfact[EndIndex - 1] = 1.0 / 3.0;
  //       simfact[startIndex]   = 0.0;
  //       dftfe::uInt ir_last   = 0;
  //       for (dftfe::uInt i = endIndex_2 - 1; i >= 2; i -= 2)
  //         {
  //           simfact[i - 1] = 4.0 / 3.0;
  //           simfact[i - 2] = 2.0 / 3.0;
  //           ir_last        = i - 2;
  //         }
  //       simfact[ir_last] *= 0.5;
  //       double            IntegralResult = 0.0;
  //       const dftfe::uInt one            = 1;
  //       for (dftfe::uInt i = 0; i < endIndex_2; i++)
  //         IntegralResult += simfact[i] * IntegrandValue(i + startIndex);
  //       //
  //       d_BLASWrapperHostPtr->xdot((EndIndex-startIndex),&simfact[0],&one,&IntegrandValue[0],&one,&IntegralResult);
  //       double residual = 0.0;
  //       if ((EndIndex - startIndex) % 2 != 0)
  //         return (IntegralResult);
  //       else
  //         {
  //           residual = 1.0 / 3.0 *
  //                      (IntegrandValue(startIndex) * 1.25 +
  //                       2.0 * IntegrandValue(startIndex + 1) -
  //                       0.25 * IntegrandValue(startIndex + 2));
  //           if (std::fabs(residual) > 1E-8)
  //             pcout << "DEBUG: Residual activated: " << residual << " "
  //                   << IntegralResult << std::endl;
  //           return (IntegralResult + residual);
  //         }
  //     }
  // }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::simpsonIntegral(
    const dftfe::uInt                           startIndex,
    const dftfe::uInt                           endIndex,
    std::function<double(const dftfe::uInt &)> &IntegrandValue)
  {
    const dftfe::uInt N = endIndex - startIndex; // number of intervals
    if (N == 0)
      return 0.0;
    if (N == 1)
      return 0.5 * (IntegrandValue(startIndex) + IntegrandValue(endIndex));

    double result = 0.0;
    if (N % 2 == 0)
      {
        // Pure Simpson's rule
        result += IntegrandValue(startIndex) + IntegrandValue(endIndex);
        for (dftfe::uInt i = 1; i < N; ++i)
          result += (i % 2 == 0 ? 2.0 : 4.0) * IntegrandValue(startIndex + i);
        result *= (1.0 / 3.0); // dr already in f()
      }
    else
      {
        // Simpson on first N-1 intervals
        result +=
          IntegrandValue(startIndex) + IntegrandValue(startIndex + N - 1);
        for (dftfe::uInt i = 1; i < N - 1; ++i)
          result += (i % 2 == 0 ? 2.0 : 4.0) * IntegrandValue(startIndex + i);
        result *= (1.0 / 3.0); // dr already in f()

        // Last trapezoid
        result +=
          0.5 * (IntegrandValue(startIndex + N - 1) + IntegrandValue(endIndex));
      }
    return result;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<double>
  pawClass<ValueType, memorySpace>::simpsonIntegralWeights(
    const dftfe::uInt startIndex,
    const dftfe::uInt EndIndex)
  {
    const dftfe::uInt   nPoints = EndIndex - startIndex + 1;
    std::vector<double> simfact(nPoints, 0.0);

    if (nPoints == 1)
      {
        simfact[0] = 1.0; // degenerate case
        return simfact;
      }
    else if (nPoints == 2)
      {
        // Trapezoidal fallback
        simfact[0] = simfact[1] = 0.5;
        return simfact;
      }
    else
      {
        // Standard Simpson's weights
        simfact[0]           = 1.0 / 3.0;
        simfact[nPoints - 1] = 1.0 / 3.0;

        for (int i = 1; i < (int)nPoints - 1; ++i)
          {
            if (i % 2 == 0)
              simfact[i] = 2.0 / 3.0;
            else
              simfact[i] = 4.0 / 3.0;
          }

        // If odd number of intervals, apply standard Simpson's rule
        if ((nPoints - 1) % 2 != 0)
          {
            // Odd number of intervals — fallback on trapezoid for last segment
            simfact[nPoints - 2] = 0.5;
            simfact[nPoints - 1] = 0.5;
          }
        return simfact;
      }
  }
  // template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  // const std::vector<double>
  // pawClass<ValueType, memorySpace>::simpsonIntegralWeights(
  //   const dftfe::uInt startIndex,
  //   const dftfe::uInt EndIndex)
  // {
  //   std::vector<double> simfact((EndIndex - startIndex), 0.0);
  //   // std::cout<<"Start and EndIndex: "<<startIndex<<"
  //   "<<EndIndex<<std::endl; if (startIndex == EndIndex)
  //     return simfact;
  //   else if (startIndex + 1 == EndIndex)
  //     return std::vector<double>(2, 0.5);
  //   else
  //     {
  //       // std::cout<<"Here: "<<std::endl;
  //       std::vector<double> simfact((EndIndex - startIndex), 0.0);
  //       simfact[EndIndex - startIndex - 1] = 1.0 / 3.0;
  //       simfact[0]                         = 0.0;
  //       dftfe::uInt ir_last                = 0;
  //       for (dftfe::uInt i = (EndIndex - startIndex) - 1; i >= 2; i -= 2)
  //         {
  //           simfact[i - 1] = 4.0 / 3.0;
  //           simfact[i - 2] = 2.0 / 3.0;
  //           ir_last        = i - 2;
  //         }
  //       simfact[ir_last] *= 0.5;
  //       return simfact;
  //     }
  // }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::SimpsonResidual(
    const dftfe::uInt          startIndex,
    const dftfe::uInt          EndIndex,
    const std::vector<double> &integrandValue)
  {
    double residual = 0.0;
    if ((EndIndex - startIndex) % 2 != 0)
      return (residual);
    else
      {
        residual = 1.0 / 3.0 *
                   (integrandValue[startIndex] * 1.25 +
                    2.0 * integrandValue[startIndex + 1] -
                    0.25 * integrandValue[startIndex + 2]);
        if (std::fabs(residual) > 1E-8)
          pcout << "DEBUG: Residual activated: " << residual << std::endl;
        return (residual);
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::oneTermPoissonPotential(
    const double              *fun,
    const dftfe::uInt          l,
    const dftfe::uInt          rminIndex,
    const dftfe::uInt          rmaxIndex,
    const dftfe::Int           powerofR,
    const std::vector<double> &radial,
    const std::vector<double> &rab,
    std::vector<double>       &Potential)
  {
    const size_t N = radial.size();
    if (N == 0)
      return;
    if (rminIndex >= N || rmaxIndex >= N || rminIndex > rmaxIndex)
      {
        throw std::invalid_argument("Invalid rmin/rmax indices");
      }
    if (rab.size() != N)
      throw std::invalid_argument("rab size mismatch");

    Potential.assign(N, double(0));

    // tiny for numerical safety
    const double tiny = std::numeric_limits<double>::epsilon() * double(1e6);
    const double prefac =
      double(4.0) * M_PI / (double(2.0) * double(l) + double(1.0));

    // Temporary arrays
    std::vector<double> Integral1(N, double(0));
    std::vector<double> Integral2(N, double(0));
    std::vector<double> aa(N, double(0));
    std::vector<double> bb(N, double(0));

    // build aa and bb on the same index convention as ABINIT (skip i=0 if you
    // like)
    for (size_t i = 0; i < N; ++i)
      {
        const double r = radial[i];
        const double g = fun[i];
        // rad^l and rad^(l+1) handled via pow; guard r=0
        double radL =
          (r > 0.0) ? std::pow(r, static_cast<double>(l)) : double(0.0);
        double radL1 = (r > 0.0) ? radL * r : double(0.0); // r^(l+1)
        // aa = g * r^{powerofR} * r^l * rab[i]
        // bb = g * r^{powerofR} / r^{l+1} * rab[i]
        double rpow = (r > 0.0 || powerofR >= 0) ?
                        std::pow(r, static_cast<double>(powerofR)) :
                        std::pow(r + tiny, static_cast<double>(powerofR));
        aa[i]       = g * rpow * radL * rab[i];
        // avoid divide-by-zero: if r==0 set bb[i] = 0 (integral handled via
        // neighbouring points)
        if (r > 0.0)
          bb[i] = g * rpow / radL1 * rab[i];
        else
          bb[i] = double(0.0);
      }

    // --- Integral1: forward cumulative using ABINIT index-Simpson pattern ---
    // Initialize Integral1[rminIndex] = 0
    Integral1[rminIndex] = double(0.0);

    // start i = rminIndex + 2 (we compute Integral1[i] from Integral1[i-2])
    // and fill also Integral1[i-1] with the staggered formula
    for (int ii = static_cast<int>(rminIndex) + 2;
         ii <= static_cast<int>(rmaxIndex);
         ii += 2)
      {
        int i = ii;
        // Simpson on [i-2, i] -> adds (aa[i-2] + 4*aa[i-1] + aa[i]) / 3
        Integral1[i] =
          Integral1[i - 2] +
          (aa[i - 2] + double(4.0) * aa[i - 1] + aa[i]) / double(3.0);
        // Staggered value for i-1 (mid index) using ABINIT's 3-point
        // correction: Integral1[i-1] = Integral1[i-2] + (1/3)*(1.25 aa[i-2]
        // + 2.0 aa[i-1] - 0.25 aa[i])
        Integral1[i - 1] =
          Integral1[i - 2] + (double(1.25) * aa[i - 2] +
                              double(2.0) * aa[i - 1] - double(0.25) * aa[i]) /
                               double(3.0);
      }

    // If the interval count (rmax-rmin) is odd, the above loop will not set
    // Integral1[rmaxIndex] when rmaxIndex - rminIndex is odd (i.e. an extra
    // interval). ABINIT applies a special correction:
    if (((int)rmaxIndex - (int)rminIndex) % 2 != 0)
      {
        // set last point using the same staggering formula (forward)
        int i = static_cast<int>(rmaxIndex);
        if (i >= 2)
          {
            Integral1[i] = Integral1[i - 1]; // default fallback, then correct:
            Integral1[i] = Integral1[i - 1] +
                           (double(1.25) * aa[i - 2] + double(2.0) * aa[i - 1] -
                            double(0.25) * aa[i]) /
                             double(3.0);
          }
      }

    // --- Integral2: reverse cumulative, mirror pattern ---
    Integral2[rmaxIndex] = double(0.0);

    for (int ii = static_cast<int>(rmaxIndex) - 2;
         ii >= static_cast<int>(rminIndex);
         ii -= 2)
      {
        int i = ii;
        // Simpson on [i, i+2]: adds (bb[i+2] + 4*bb[i+1] + bb[i]) / 3 to
        // Integral2[i+2] Here accumulate backward: Integral2[i] =
        // Integral2[i+2] + (bb[i+2] + 4*bb[i+1] + bb[i]) / 3
        Integral2[i] =
          Integral2[i + 2] +
          (bb[i + 2] + double(4.0) * bb[i + 1] + bb[i]) / double(3.0);
        // staggered for i+1
        Integral2[i + 1] =
          Integral2[i + 2] + (double(1.25) * bb[i + 2] +
                              double(2.0) * bb[i + 1] - double(0.25) * bb[i]) /
                               double(3.0);
      }

    if (((int)rmaxIndex - (int)rminIndex) % 2 != 0)
      {
        // handle leftmost point if needed (mirror of forward scheme)
        int i = static_cast<int>(rminIndex);
        if (i + 2 < static_cast<int>(N))
          {
            Integral2[i] = Integral2[i + 1] +
                           (double(1.25) * bb[i + 2] + double(2.0) * bb[i + 1] -
                            double(0.25) * bb[i]) /
                             double(3.0);
          }
      }

    // --- Assemble potential ---
    for (size_t i = rminIndex; i < N; ++i)
      {
        if (i <= rmaxIndex)
          {
            double r = radial[i];
            if (r <= tiny)
              {
                // safe small-r handling: use limit from nearby point
                // (you can replace with analytic small-r limit if desired)
                if (i + 1 <= rmaxIndex)
                  Potential[i] = Potential[i + 1];
                else
                  Potential[i] = double(0.0);
              }
            else
              {
                double term1 =
                  Integral1[i] / std::pow(r, static_cast<double>(l));
                double term2 =
                  Integral2[i] * std::pow(r, static_cast<double>(l + 1));
                Potential[i] = prefac * (term1 + term2);
              }
          }
        else
          {
            // for r > rmax, set to constant = value at rmaxIndex
            Potential[i] = Potential[rmaxIndex];
          }
      }
  }

  // }
  // template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  // void pawClass<ValueType, memorySpace>::oneTermPoissonPotential(
  //     const double              *fun,
  //     const dftfe::uInt          l,
  //     const dftfe::uInt          rminIndex,
  //     const dftfe::uInt          rmaxIndex,
  //     const dftfe::Int           powerofR,
  //     const std::vector<double> &radial,
  //     const std::vector<double> &rab,
  //     std::vector<double>       &Potential)
  // {
  //     Potential.assign(radial.size(), 0.0);

  //     const double rmin = radial[rminIndex];
  //     const double rmax = radial[rmaxIndex];

  //     std::vector<double> Integral1(radial.size(), 0.0);
  //     std::vector<double> Integral2(radial.size(), 0.0);
  //     std::vector<double> radL(radial.size(), 0.0);
  //     std::vector<double> radL1(radial.size(), 0.0);
  //     std::vector<double> aa(radial.size(), 0.0);
  //     std::vector<double> bb(radial.size(), 0.0);

  //     // Precompute powers
  //     for (dftfe::Int i = 0; i < (dftfe::Int)radial.size(); i++)
  //     {
  //         double r   = radial[i];
  //         double g_y = fun[i];
  //         radL[i]    = std::pow(r, l);
  //         radL1[i]   = radL[i] * r;
  //         aa[i]      = g_y * std::pow(r, powerofR) * radL[i] * rab[i];
  //         bb[i]      = g_y * std::pow(r, powerofR) / radL1[i] * rab[i];
  //     }

  //     const dftfe::Int N = (dftfe::Int)rmaxIndex - (dftfe::Int)rminIndex;

  //     // ---- Simpson integration for Integral1 ----
  //     {
  //         double sum1 = 0.0;
  //         dftfe::Int i = (dftfe::Int)rminIndex;
  //         // Main Simpson loop for even chunks
  //         for (; i + 2 <= (dftfe::Int)rmaxIndex - 1; i += 2)
  //         {
  //             sum1 += (aa[i] + 4.0 * aa[i + 1] + aa[i + 2]) / 3.0;
  //             Integral1[i + 2] = sum1;
  //         }
  //         // Handle last odd interval (Simpson 3/8)
  //         if (i + 3 <= (dftfe::Int)rmaxIndex)
  //         {
  //             sum1 += (3.0 / 8.0) * (aa[i] + 3.0 * aa[i + 1] + 3.0 * aa[i +
  //             2] + aa[i + 3]); Integral1[i + 3] = sum1;
  //         }
  //     }

  //     // ---- Simpson integration for Integral2 (reverse) ----
  //     {
  //         double sum2 = 0.0;
  //         dftfe::Int i = (dftfe::Int)rmaxIndex;
  //         for (; i - 2 >= (dftfe::Int)rminIndex + 1; i -= 2)
  //         {
  //             sum2 += (bb[i] + 4.0 * bb[i - 1] + bb[i - 2]) / 3.0;
  //             Integral2[i - 2] = sum2;
  //         }
  //         if (i - 3 >= (dftfe::Int)rminIndex)
  //         {
  //             sum2 += (3.0 / 8.0) * (bb[i] + 3.0 * bb[i - 1] + 3.0 * bb[i -
  //             2] + bb[i - 3]); Integral2[i - 3] = sum2;
  //         }
  //     }

  //     // ---- Assemble potential ----
  //     for (dftfe::Int i = (dftfe::Int)rminIndex; i <
  //     (dftfe::Int)radial.size(); i++)
  //     {
  //         if (radial[i] > 1E-10)
  //         {
  //             double Value1 = Integral1[i];
  //             double Value2 = Integral2[i];
  //             Potential[i] = (4 * M_PI / (2 * double(l) + 1.0)) *
  //                            (Value1 / std::pow(radial[i], l) +
  //                             Value2 * std::pow(radial[i], l + 1));
  //         }
  //         if (i > (dftfe::Int)rmaxIndex)
  //             Potential[i] = Potential[(dftfe::Int)rmaxIndex];
  //     }
  // }
  // template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  // void
  // pawClass<ValueType, memorySpace>::twoTermPoissonPotential(
  //     const double              *fun1,
  //     const double              *fun2,
  //     const dftfe::uInt          l,
  //     const dftfe::uInt          rminIndex,
  //     const dftfe::uInt          rmaxIndex,
  //     const dftfe::Int           powerofR,
  //     const std::vector<double> &radial,
  //     const std::vector<double> &rab,
  //     std::vector<double>       &Potential)
  // {
  //     Potential.assign(radial.size(), 0.0);

  //     std::vector<double> Integral1(radial.size(), 0.0);
  //     std::vector<double> Integral2(radial.size(), 0.0);
  //     std::vector<double> radL(radial.size(), 0.0);
  //     std::vector<double> radL1(radial.size(), 0.0);
  //     std::vector<double> aa(radial.size(), 0.0);
  //     std::vector<double> bb(radial.size(), 0.0);

  //     // Precompute terms
  //     for (dftfe::Int i = 0; i < (dftfe::Int)radial.size(); i++)
  //     {
  //         double r   = radial[i];
  //         double g_y = fun1[i] * fun2[i] * std::pow(r, powerofR);
  //         radL[i]    = std::pow(r, l);
  //         radL1[i]   = radL[i] * r;
  //         aa[i]      = g_y * radL[i] * rab[i];
  //         bb[i]      = g_y / radL1[i] * rab[i];
  //     }

  //     const dftfe::Int N = (dftfe::Int)rmaxIndex - (dftfe::Int)rminIndex;

  //     // ---- Forward Simpson integration for Integral1 ----
  //     {
  //         double sum1 = 0.0;
  //         dftfe::Int i = (dftfe::Int)rminIndex;
  //         for (; i + 2 <= (dftfe::Int)rmaxIndex - 1; i += 2)
  //         {
  //             sum1 += (aa[i] + 4.0 * aa[i + 1] + aa[i + 2]) / 3.0;
  //             Integral1[i + 2] = sum1;
  //         }
  //         // Handle last odd chunk with Simpson 3/8
  //         if (i + 3 <= (dftfe::Int)rmaxIndex)
  //         {
  //             sum1 += (3.0 / 8.0) *
  //                     (aa[i] + 3.0 * aa[i + 1] + 3.0 * aa[i + 2] + aa[i +
  //                     3]);
  //             Integral1[i + 3] = sum1;
  //         }
  //     }

  //     // ---- Reverse Simpson integration for Integral2 ----
  //     {
  //         double sum2 = 0.0;
  //         dftfe::Int i = (dftfe::Int)rmaxIndex;
  //         for (; i - 2 >= (dftfe::Int)rminIndex + 1; i -= 2)
  //         {
  //             sum2 += (bb[i] + 4.0 * bb[i - 1] + bb[i - 2]) / 3.0;
  //             Integral2[i - 2] = sum2;
  //         }
  //         if (i - 3 >= (dftfe::Int)rminIndex)
  //         {
  //             sum2 += (3.0 / 8.0) *
  //                     (bb[i] + 3.0 * bb[i - 1] + 3.0 * bb[i - 2] + bb[i -
  //                     3]);
  //             Integral2[i - 3] = sum2;
  //         }
  //     }

  //     // ---- Assemble potential ----
  //     for (dftfe::Int i = (dftfe::Int)rminIndex; i <
  //     (dftfe::Int)radial.size(); i++)
  //     {
  //         if (radial[i] > 1E-10)
  //         {
  //             double Value1 = Integral1[i];
  //             double Value2 = Integral2[i];
  //             Potential[i] = (4 * M_PI / (2.0 * double(l) + 1.0)) *
  //                            (Value1 / std::pow(radial[i], l) +
  //                             Value2 * std::pow(radial[i], l + 1));
  //         }
  //         if (i > (dftfe::Int)rmaxIndex)
  //             Potential[i] = Potential[(dftfe::Int)rmaxIndex];
  //     }
  // }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::twoTermPoissonPotential(
    const double              *fun1,
    const double              *fun2,
    const dftfe::uInt          l,
    const dftfe::uInt          rminIndex,
    const dftfe::uInt          rmaxIndex,
    const dftfe::Int           powerofR,
    const std::vector<double> &radial,
    const std::vector<double> &rab,
    std::vector<double>       &Potential)
  {
    const size_t N = radial.size();
    if (N == 0)
      return;
    if (rminIndex >= N || rmaxIndex >= N || rminIndex > rmaxIndex)
      {
        throw std::invalid_argument("Invalid rmin/rmax indices");
      }
    if (rab.size() != N)
      throw std::invalid_argument("rab size mismatch");

    Potential.assign(N, double(0));

    // tiny for numerical safety
    const double tiny = std::numeric_limits<double>::epsilon() * double(1e6);
    const double prefac =
      double(4.0) * M_PI / (double(2.0) * double(l) + double(1.0));

    // Temporary arrays
    std::vector<double> Integral1(N, double(0));
    std::vector<double> Integral2(N, double(0));
    std::vector<double> aa(N, double(0));
    std::vector<double> bb(N, double(0));

    // build aa and bb on the same index convention as ABINIT (skip i=0 if you
    // like)
    for (size_t i = 0; i < N; ++i)
      {
        const double r = radial[i];
        const double g = fun1[i] * fun2[i];
        // rad^l and rad^(l+1) handled via pow; guard r=0
        double radL =
          (r > 0.0) ? std::pow(r, static_cast<double>(l)) : double(0.0);
        double radL1 = (r > 0.0) ? radL * r : double(0.0); // r^(l+1)
        // aa = g * r^{powerofR} * r^l * rab[i]
        // bb = g * r^{powerofR} / r^{l+1} * rab[i]
        double rpow = (r > 0.0 || powerofR >= 0) ?
                        std::pow(r, static_cast<double>(powerofR)) :
                        std::pow(r + tiny, static_cast<double>(powerofR));
        aa[i]       = g * rpow * radL * rab[i];
        // avoid divide-by-zero: if r==0 set bb[i] = 0 (integral handled via
        // neighbouring points)
        if (r > 0.0)
          bb[i] = g * rpow / radL1 * rab[i];
        else
          bb[i] = double(0.0);
      }

    // --- Integral1: forward cumulative using ABINIT index-Simpson pattern ---
    // Initialize Integral1[rminIndex] = 0
    Integral1[rminIndex] = double(0.0);

    // start i = rminIndex + 2 (we compute Integral1[i] from Integral1[i-2])
    // and fill also Integral1[i-1] with the staggered formula
    for (int ii = static_cast<int>(rminIndex) + 2;
         ii <= static_cast<int>(rmaxIndex);
         ii += 2)
      {
        int i = ii;
        // Simpson on [i-2, i] -> adds (aa[i-2] + 4*aa[i-1] + aa[i]) / 3
        Integral1[i] =
          Integral1[i - 2] +
          (aa[i - 2] + double(4.0) * aa[i - 1] + aa[i]) / double(3.0);
        // Staggered value for i-1 (mid index) using ABINIT's 3-point
        // correction: Integral1[i-1] = Integral1[i-2] + (1/3)*(1.25 aa[i-2]
        // + 2.0 aa[i-1] - 0.25 aa[i])
        Integral1[i - 1] =
          Integral1[i - 2] + (double(1.25) * aa[i - 2] +
                              double(2.0) * aa[i - 1] - double(0.25) * aa[i]) /
                               double(3.0);
      }

    // If the interval count (rmax-rmin) is odd, the above loop will not set
    // Integral1[rmaxIndex] when rmaxIndex - rminIndex is odd (i.e. an extra
    // interval). ABINIT applies a special correction:
    if (((int)rmaxIndex - (int)rminIndex) % 2 != 0)
      {
        // set last point using the same staggering formula (forward)
        int i = static_cast<int>(rmaxIndex);
        if (i >= 2)
          {
            Integral1[i] = Integral1[i - 1]; // default fallback, then correct:
            Integral1[i] = Integral1[i - 1] +
                           (double(1.25) * aa[i - 2] + double(2.0) * aa[i - 1] -
                            double(0.25) * aa[i]) /
                             double(3.0);
          }
      }

    // --- Integral2: reverse cumulative, mirror pattern ---
    Integral2[rmaxIndex] = double(0.0);

    for (int ii = static_cast<int>(rmaxIndex) - 2;
         ii >= static_cast<int>(rminIndex);
         ii -= 2)
      {
        int i = ii;
        // Simpson on [i, i+2]: adds (bb[i+2] + 4*bb[i+1] + bb[i]) / 3 to
        // Integral2[i+2] Here accumulate backward: Integral2[i] =
        // Integral2[i+2] + (bb[i+2] + 4*bb[i+1] + bb[i]) / 3
        Integral2[i] =
          Integral2[i + 2] +
          (bb[i + 2] + double(4.0) * bb[i + 1] + bb[i]) / double(3.0);
        // staggered for i+1
        Integral2[i + 1] =
          Integral2[i + 2] + (double(1.25) * bb[i + 2] +
                              double(2.0) * bb[i + 1] - double(0.25) * bb[i]) /
                               double(3.0);
      }

    if (((int)rmaxIndex - (int)rminIndex) % 2 != 0)
      {
        // handle leftmost point if needed (mirror of forward scheme)
        int i = static_cast<int>(rminIndex);
        if (i + 2 < static_cast<int>(N))
          {
            Integral2[i] = Integral2[i + 1] +
                           (double(1.25) * bb[i + 2] + double(2.0) * bb[i + 1] -
                            double(0.25) * bb[i]) /
                             double(3.0);
          }
      }

    // --- Assemble potential ---
    for (size_t i = rminIndex; i < N; ++i)
      {
        if (i <= rmaxIndex)
          {
            double r = radial[i];
            if (r <= tiny)
              {
                // safe small-r handling: use limit from nearby point
                // (you can replace with analytic small-r limit if desired)
                if (i + 1 <= rmaxIndex)
                  Potential[i] = Potential[i + 1];
                else
                  Potential[i] = double(0.0);
              }
            else
              {
                double term1 =
                  Integral1[i] / std::pow(r, static_cast<double>(l));
                double term2 =
                  Integral2[i] * std::pow(r, static_cast<double>(l + 1));
                Potential[i] = prefac * (term1 + term2);
              }
          }
        else
          {
            // for r > rmax, set to constant = value at rmaxIndex
            Potential[i] = Potential[rmaxIndex];
          }
      }
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::integralOfProjectorsInAugmentationSphere(
    const double        *f1,
    const double        *f2,
    std::vector<double> &radial,
    std::vector<double> &rab,
    const dftfe::uInt    rminIndex,
    const dftfe::uInt    rmaxIndex)
  {
    double value = 0.0;

    std::function<double(const dftfe::uInt &)> integral =
      [&](const dftfe::uInt &i) {
        if (radial[i] < 1E-9)
          return 0.0;

        double Value = rab[i] * f2[i] * f1[i] * radial[i];
        return (Value);
      };
    value = simpsonIntegral(rminIndex, rmaxIndex, integral);
    return (value);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::integralOfDensity(
    const double        *f1,
    std::vector<double> &radial,
    std::vector<double> &rab,
    const dftfe::uInt    rminIndex,
    const dftfe::uInt    rmaxIndex)
  {
    double value = 0.0;

    std::function<double(const dftfe::uInt &)> integral =
      [&](const dftfe::uInt &i) {
        double Value = rab[i] * f1[i] * radial[i];
        return (Value);
      };

    value = simpsonIntegral(rminIndex, rmaxIndex, integral);
    return (value);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::getSphericalQuadratureRule(
    std::vector<double>              &quad_weights,
    std::vector<std::vector<double>> &quad_points)
  {
    std::vector<std::vector<double>> quadratureData;
    char                             quadratureFileName[256];
    if (d_dftParamsPtr->sphericalQuadrature == 0)
      sprintf(quadratureFileName,
              "%s/data/quadratureRules/quadRule86.txt",
              DFTFE_PATH);
    else if (d_dftParamsPtr->sphericalQuadrature == 1)
      sprintf(quadratureFileName,
              "%s/data/quadratureRules/quadRule50.txt",
              DFTFE_PATH);
    else if (d_dftParamsPtr->sphericalQuadrature == 2)
      sprintf(quadratureFileName,
              "%s/data/quadratureRules/quadRule74.txt",
              DFTFE_PATH);
    else if (d_dftParamsPtr->sphericalQuadrature == 3)
      sprintf(quadratureFileName,
              "%s/data/quadratureRules/quadRule110.txt",
              DFTFE_PATH);
    dftUtils::readFile(3, quadratureData, quadratureFileName);
    dftfe::Int numRows = quadratureData.size();
    for (dftfe::Int i = 0; i < numRows; i++)
      {
        quad_weights.push_back(quadratureData[i][2]);
        std::vector<double> temp(2, 0);
        temp[1] = (quadratureData[i][0] + 180) / 180 * M_PI;
        temp[0] = quadratureData[i][1] / 180 * M_PI;
        quad_points.push_back(temp);
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<double>
  pawClass<ValueType, memorySpace>::derivativeOfRealSphericalHarmonic(
    dftfe::uInt lQuantumNo,
    dftfe::Int  mQuantumNo,
    double      theta,
    double      phi)
  {
    std::vector<double> RSH(2, 0.0);
    if (lQuantumNo == 0)
      return (RSH);
    double sphericalHarmonicValue, sphericalHarmonicValue1;
    sphericalHarmonicUtils::getSphericalHarmonicVal(
      theta, phi, lQuantumNo, -mQuantumNo, sphericalHarmonicValue);
    // RSH[1] = -std::abs(m) * sphericalHarmonicValue;
    RSH[1] = -mQuantumNo * sphericalHarmonicValue;

    sphericalHarmonicUtils::getSphericalHarmonicVal(
      theta, phi, lQuantumNo, mQuantumNo, sphericalHarmonicValue);
    sphericalHarmonicUtils::getSphericalHarmonicVal(
      theta, phi, lQuantumNo + 1, mQuantumNo, sphericalHarmonicValue1);
    if (std::fabs(std::sin(theta)) > 1E-8)
      RSH[0] = -double(lQuantumNo + 1) * std::cos(theta) / std::sin(theta) *
                 sphericalHarmonicValue +
               sqrt(double(2 * lQuantumNo + 1.0)) /
                 sqrt(double(2 * lQuantumNo + 3.0)) *
                 sqrt(double((lQuantumNo + 1) * (lQuantumNo + 1) -
                             mQuantumNo * mQuantumNo)) *
                 sphericalHarmonicValue1 / std::sin(theta);
    else
      RSH[0] = 0.0;


    return (RSH);
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<double>
  pawClass<ValueType, memorySpace>::DijVectorForMixing(TypeOfField typeOfField,
                                                       mixingVariable mixVar)
  {
    std::vector<dftfe::uInt> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    const std::vector<dftfe::uInt> ownedAtomIds =
      d_nonLocalOperator->getOwnedAtomIdsInCurrentProcessor();
    std::map<TypeOfField, std::map<dftfe::uInt, std::vector<double>>> &D_ijMix =
      mixVar == mixingVariable::DijMatrixRho ? D_ij[0] : D_ij[1];
    std::vector<double> DijVector;
    for (dftfe::uInt iAtom = 0; iAtom < ownedAtomIds.size(); iAtom++)
      {
        dftfe::uInt atomId = ownedAtomIds[iAtom];
        dftfe::uInt Znum   = atomicNumber[atomId];
        dftfe::uInt numProj =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        dftfe::uInt numRadProj =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);

        std::vector<double> Dij_in  = D_ijMix[TypeOfField::In][atomId];
        std::vector<double> Dij_out = D_ijMix[TypeOfField::Out][atomId];
        if (typeOfField == TypeOfField::In)
          {
            for (dftfe::Int iProj = 0; iProj < numProj; iProj++)
              {
                for (dftfe::Int jProj = 0; jProj < numProj; jProj++)
                  {
                    DijVector.push_back(Dij_in[iProj * numProj + jProj]);
                  }
              }
          }
        else if (typeOfField == TypeOfField::Out)
          {
            for (dftfe::Int iProj = 0; iProj < numProj; iProj++)
              {
                for (dftfe::Int jProj = 0; jProj < numProj; jProj++)
                  {
                    DijVector.push_back(Dij_out[iProj * numProj + jProj]);
                  }
              }
          }
        else
          {
            for (dftfe::Int iProj = 0; iProj < numProj; iProj++)
              {
                for (dftfe::Int jProj = 0; jProj < numProj; jProj++)
                  {
                    DijVector.push_back(Dij_out[iProj * numProj + jProj] -
                                        Dij_in[iProj * numProj + jProj]);
                  }
              }
          }
      }

    MPI_Barrier(d_mpiCommParent);
    return (DijVector);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<double>
  pawClass<ValueType, memorySpace>::radialDerivativeOfMeshData(
    const std::vector<double> &r,
    const std::vector<double> &rab,
    const std::vector<double> &functionValue)
  {
    alglib::real_1d_array       x, y;
    alglib::spline1dinterpolant p1;
    dftfe::uInt                 size = r.size();
    x.setcontent(size, &r[0]);
    y.setcontent(size, &functionValue[0]);
    alglib::ae_int_t natural_bound_type = 1;
    alglib::ae_int_t dir_bound_type     = 0;
    alglib::spline1dbuildcubic(x,
                               y,
                               size,
                               dir_bound_type,
                               functionValue[0],
                               natural_bound_type,
                               0.0,
                               p1);
    std::vector<double> der(size, 0.0);
    std::vector<double> coeff(5, 0.0);
    coeff[0] = -25.0 / 12.0;
    coeff[1] = 4.0;
    coeff[2] = -3.0;
    coeff[3] = 4.0 / 3.0;
    coeff[4] = -1.0 / 4.0;
    MPI_Barrier(d_mpiCommParent);

    for (dftfe::uInt i = 0; i < size - 4; i++)
      {
        double Value, derivativeValue, radialDensitySecondDerivative;
        der[i] =
          (coeff[0] * functionValue[i] + coeff[1] * functionValue[i + 1] +
           coeff[2] * functionValue[i + 2] + coeff[3] * functionValue[i + 3] +
           coeff[4] * functionValue[i + 4]) /
          rab[i];
        alglib::spline1ddiff(
          p1, r[i], Value, derivativeValue, radialDensitySecondDerivative);
      }

    return (der);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::multipoleIntegrationGrid(
    double              *f1,
    double              *f2,
    std::vector<double> &radial,
    std::vector<double> &rab,
    const dftfe::Int     L,
    const dftfe::uInt    rminIndex,
    const dftfe::uInt    rmaxIndex)
  {
    std::function<double(const dftfe::uInt &)> integrationValue =
      [&](const dftfe::uInt &i) {
        double Value = rab[i] * f2[i] * f1[i];
        Value *= pow(radial[i], L + 2);
        return (Value);
      };

    double IntegralResult =
      simpsonIntegral(rminIndex, rmaxIndex, integrationValue);


    return (IntegralResult);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::densityScalingFactor(
    const std::vector<std::vector<double>> &atomLocations)
  {
    double scaleFactor  = 0.0;
    double numElectrons = 0;
    std::map<TypeOfField, std::map<dftfe::uInt, std::vector<double>>> &D_ijRho =
      D_ij[0];
    for (dftfe::Int atomId = 0; atomId < atomLocations.size(); atomId++)
      {
        dftfe::uInt Znum = atomLocations[atomId][0];
        numElectrons += atomLocations[atomId][1];
        const dftfe::uInt numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const dftfe::uInt numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        std::vector<double> Dij              = D_ijRho[TypeOfField::In][atomId];
        std::vector<double> multipoleTable   = d_multipole[Znum];
        dftfe::uInt         projectorIndex_i = 0;
        for (dftfe::Int alpha_i = 0; alpha_i < numberOfRadialProjectors;
             alpha_i++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> AEsphFn_i =
              d_atomicAEPartialWaveFnsMap.find(std::make_pair(Znum, alpha_i))
                ->second;
            dftfe::Int lQuantumNo_i = AEsphFn_i->getQuantumNumberl();
            for (dftfe::Int mQuantumNo_i = -lQuantumNo_i;
                 mQuantumNo_i <= lQuantumNo_i;
                 mQuantumNo_i++)
              {
                dftfe::uInt projectorIndex_j = 0;
                for (dftfe::Int alpha_j = 0; alpha_j < numberOfRadialProjectors;
                     alpha_j++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase>
                      AEsphFn_j = d_atomicAEPartialWaveFnsMap
                                    .find(std::make_pair(Znum, alpha_j))
                                    ->second;
                    dftfe::Int lQuantumNo_j = AEsphFn_j->getQuantumNumberl();
                    for (dftfe::Int mQuantumNo_j = -lQuantumNo_j;
                         mQuantumNo_j <= lQuantumNo_j;
                         mQuantumNo_j++)
                      {
                        scaleFactor +=
                          gaunt(lQuantumNo_i,
                                lQuantumNo_j,
                                0,
                                mQuantumNo_i,
                                mQuantumNo_j,
                                0) *
                          Dij[projectorIndex_i * numberOfProjectors +
                              projectorIndex_j] *
                          multipoleTable[alpha_i * numberOfRadialProjectors +
                                         alpha_j];
                        projectorIndex_j++;
                      }
                  }
                // pcout<<std::endl;
                projectorIndex_i++;
              }
          }
      }
    pcout << "Number of Electrons: " << numElectrons << std::endl;
    pcout << "sqrt(4*M_PI)*DeltaijDij: " << sqrt(4 * M_PI) * scaleFactor
          << std::endl;
    pcout << "Scaling Factor for Init Rho: "
          << numElectrons - sqrt(4 * M_PI) * scaleFactor << std::endl;
    return (numElectrons - sqrt(4 * M_PI) * scaleFactor);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::communicateDijAcrossAllProcessors(
    TypeOfField      typeOfField,
    const dftfe::Int iComp,
    const MPI_Comm  &interpoolcomm,
    const MPI_Comm  &interBandGroupComm,
    const bool       communicateAcrossPool)
  {
    const std::vector<dftfe::uInt> ownedAtomIds =
      d_nonLocalOperator->getOwnedAtomIdsInCurrentProcessor();
    std::vector<double>      DijTotalVector(d_nProjSqTotal, 0.0);
    std::vector<dftfe::uInt> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    std::map<TypeOfField, std::map<dftfe::uInt, std::vector<double>>>
      &D_ijVector = D_ij[iComp];
    if (ownedAtomIds.size() > 0)
      {
        for (dftfe::Int iAtom = 0; iAtom < ownedAtomIds.size(); iAtom++)
          {
            dftfe::uInt         atomId     = ownedAtomIds[iAtom];
            dftfe::uInt         Znum       = atomicNumber[atomId];
            dftfe::uInt         startIndex = d_projectorStartIndex[atomId];
            std::vector<double> Dij        = D_ijVector[typeOfField][atomId];
            dftfe::uInt         numberOfProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            dftfe::uInt index = 0;
            for (dftfe::uInt i = 0; i < numberOfProjectors; i++)
              {
                for (dftfe::uInt j = 0; j <= i; j++)
                  {
                    DijTotalVector[(startIndex + index)] =
                      Dij[i * numberOfProjectors + j];
                    index++;
                  }
              }
          }
      }
    MPI_Barrier(d_mpiCommParent);
    MPI_Allreduce(MPI_IN_PLACE,
                  &DijTotalVector[0],
                  d_nProjSqTotal,
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpiCommParent);
    int size;
    MPI_Comm_size(interpoolcomm, &size);
    if (size > 1 && communicateAcrossPool)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      &DijTotalVector[0],
                      d_nProjSqTotal,
                      MPI_DOUBLE,
                      MPI_SUM,
                      interpoolcomm);
        if (d_this_mpi_process != 0)
          {
            DijTotalVector.clear();
            DijTotalVector.resize(d_nProjSqTotal, 0.0);
          }
        MPI_Allreduce(MPI_IN_PLACE,
                      &DijTotalVector[0],
                      d_nProjSqTotal,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpiCommParent);
      }
    dftfe::Int rank = dealii::Utilities::MPI::this_mpi_process(interpoolcomm);

    MPI_Barrier(d_mpiCommParent);
    for (dftfe::uInt atomId = 0; atomId < atomicNumber.size(); atomId++)
      {
        dftfe::uInt Znum = atomicNumber[atomId];
        dftfe::uInt numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        dftfe::uInt         startIndex = d_projectorStartIndex[atomId];
        std::vector<double> Dij(numberOfProjectors * numberOfProjectors, 0.0);
        dftfe::uInt         index = 0;
        for (dftfe::Int i = 0; i < numberOfProjectors; i++)
          {
            for (dftfe::Int j = 0; j <= i; j++)
              {
                Dij[i * numberOfProjectors + j] =
                  DijTotalVector[(startIndex + index)];
                Dij[j * numberOfProjectors + i] =
                  DijTotalVector[(startIndex + index)];
                index++;
              }
          }

        if (d_verbosity >= 5 && (rank == 0))
          {
            MPI_Barrier(d_mpiCommParent);
            pcout << "---------------MATRIX METHOD ------------------------"
                  << std::endl;


            pcout
              << "------------------------------------------------------------"
              << std::endl;
            pcout << "D_ij of atom: " << atomId
                  << " for spin channel: " << iComp << " with Z:" << Znum
                  << std::endl;
            dftfe::Int numberProjectorFunctions = numberOfProjectors;
            for (dftfe::Int i = 0; i < numberProjectorFunctions; i++)
              {
                for (dftfe::Int j = 0; j < numberProjectorFunctions; j++)
                  pcout << Dij[i * numberProjectorFunctions + j] << " ";
                pcout << std::endl;
              }
            pcout
              << "------------------------------------------------------------"
              << std::endl;
          }


        D_ijVector[typeOfField][atomId] = Dij;
      }

    d_HamiltonianCouplingMatrixEntriesUpdated = false;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType,
           memorySpace>::computeIntegralAEdensityInsideAugmentationSphere()
  {
    std::vector<dftfe::uInt> atomicNumbers =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    std::vector<double> integralValues(atomicNumbers.size(), 0.0);
    std::map<TypeOfField, std::map<dftfe::uInt, std::vector<double>>> &D_ijRho =
      D_ij[0];
    const std::vector<dftfe::uInt> ownedAtomIds =
      d_nonLocalOperator->getOwnedAtomIdsInCurrentProcessor();
    for (dftfe::uInt iAtom = 0; iAtom < ownedAtomIds.size(); iAtom++)
      {
        dftfe::uInt         atomId = ownedAtomIds[iAtom];
        dftfe::uInt         Znum   = atomicNumbers[atomId];
        std::vector<double> radialIntegralAEPartialWFCIJ =
          d_aePartialWfcIntegralIJ[Znum];
        dftfe::uInt       rmaxAugIndex = d_RmaxAugIndex[Znum];
        const dftfe::uInt numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const dftfe::uInt numberOfRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        std::vector<double> integralAEPartialWFCIJ(numberOfProjectors *
                                                     numberOfProjectors,
                                                   0.0);
        int                 projIndexI = 0;
        for (int iProj = 0; iProj < numberOfRadialProjectors; iProj++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
              sphericalFunction.find(std::make_pair(Znum, iProj))->second;
            int lQuantumNo_i = sphFn_i->getQuantumNumberl();
            for (int mQuantumNumber_i = -lQuantumNo_i;
                 mQuantumNumber_i <= lQuantumNo_i;
                 mQuantumNumber_i++)
              {
                int projIndexJ = 0;
                for (int jProj = 0; jProj < numberOfRadialProjectors; jProj++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_j =
                      sphericalFunction.find(std::make_pair(Znum, jProj))
                        ->second;
                    int lQuantumNo_j = sphFn_j->getQuantumNumberl();
                    for (int mQuantumNumber_j = -lQuantumNo_j;
                         mQuantumNumber_j <= lQuantumNo_j;
                         mQuantumNumber_j++)
                      {
                        integralAEPartialWFCIJ[projIndexI * numberOfProjectors +
                                               projIndexJ] =
                          gaunt(lQuantumNo_i,
                                lQuantumNo_j,
                                0,
                                mQuantumNumber_i,
                                mQuantumNumber_j,
                                0) *
                          radialIntegralAEPartialWFCIJ
                            [iProj * numberOfRadialProjectors + jProj] *
                          sqrt(4 * M_PI);
                        projIndexJ++;
                      } // mQuantumNumber_j

                  } // jProj
                projIndexI++;
              } // mQuantumNumber_i



          } // iProj
        std::vector<double> Dij   = D_ijRho[TypeOfField::Out][atomId];
        double              Value = 0.0;
        for (dftfe::uInt i = 0; i < numberOfProjectors * numberOfProjectors;
             i++)
          {
            Value += Dij[i] * integralAEPartialWFCIJ[i];
          }
        if (d_atomTypeCoreFlagMap[Znum])
          {
            double              coredensity  = 0.0;
            std::vector<double> radialMesh   = d_radialMesh[Znum];
            std::vector<double> jacobianData = d_radialJacobianData[Znum];
            std::vector<double> radialAECoreDensity = d_atomCoreDensityAE[Znum];
            std::function<double(const dftfe::uInt &)> f =
              [&](const dftfe::uInt &i) {
                double Value = jacobianData[i] * radialAECoreDensity[i] *
                               pow(radialMesh[i], 2);

                return (Value);
              };
            coredensity = simpsonIntegral(0, rmaxAugIndex, f);
            //Value += coredensity * sqrt(4 * M_PI);
          }
        integralValues[atomId] = Value;

      } //
    MPI_Allreduce(MPI_IN_PLACE,
                  &integralValues[0],
                  atomicNumbers.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpiCommParent);
    // Account for K-point and band parallelization
    pcout << "Integrated electronic density in atomic spheres" << std::endl;
    pcout << "-----------------------------------------------" << std::endl;
    pcout << "AtomID     "
          << " RMaxAug    "
          << " Integrated Value" << std::endl;
    for (dftfe::Int iAtom = 0; iAtom < atomicNumbers.size(); iAtom++)
      {
        dftfe::uInt Znum = atomicNumbers[iAtom];
        pcout << iAtom << "   " << d_RmaxAug[Znum] << " "
              << integralValues[iAtom] << std::endl;
      }
    pcout << "-----------------------------------------------" << std::endl;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeMultipoleInverse()
  {
    pcout << "PAWClass Init: computing Inverse Multipole Table" << std::endl;
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
        const dftfe::uInt numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber);
        std::vector<double> fullMultipoleTable(numberOfProjectors *
                                                 numberOfProjectors,
                                               0.0);
        std::vector<double> multipoleTable   = d_multipole[*it];
        dftfe::Int          projectorIndex_i = 0;
        for (dftfe::Int alpha_i = 0; alpha_i < numberOfRadialProjectors;
             alpha_i++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
              sphericalFunction.find(std::make_pair(atomicNumber, alpha_i))
                ->second;
            dftfe::Int lQuantumNumber_i = sphFn_i->getQuantumNumberl();
            for (dftfe::Int mQuantumNumber_i = -lQuantumNumber_i;
                 mQuantumNumber_i <= lQuantumNumber_i;
                 mQuantumNumber_i++)
              {
                dftfe::Int projectorIndex_j = 0;
                for (dftfe::Int alpha_j = 0; alpha_j < numberOfRadialProjectors;
                     alpha_j++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_j =
                      sphericalFunction
                        .find(std::make_pair(atomicNumber, alpha_j))
                        ->second;
                    dftfe::Int lQuantumNumber_j = sphFn_j->getQuantumNumberl();
                    for (dftfe::Int mQuantumNumber_j = -lQuantumNumber_j;
                         mQuantumNumber_j <= lQuantumNumber_j;
                         mQuantumNumber_j++)
                      {
                        fullMultipoleTable[projectorIndex_i *
                                             numberOfProjectors +
                                           projectorIndex_j] =
                          sqrt(4 * M_PI) *
                          gaunt(lQuantumNumber_i,
                                lQuantumNumber_j,
                                0,
                                mQuantumNumber_i,
                                mQuantumNumber_j,
                                0) *
                          multipoleTable[alpha_i * numberOfRadialProjectors +
                                         alpha_j];
                        // pcout
                        //   << alpha_i << " " << alpha_j << " "
                        //   << projectorIndex_i << " " << projectorIndex_j <<
                        //   "
                        //   "
                        //   << gaunt(lQuantumNumber_i,
                        //            lQuantumNumber_j,
                        //            0,
                        //            mQuantumNumber_i,
                        //            mQuantumNumber_j,
                        //            0)
                        //   << " "
                        //   << multipoleTable[alpha_i *
                        //   numberOfRadialProjectors +
                        //                     alpha_j]
                        //   << " "
                        //   << fullMultipoleTable[projectorIndex_i *
                        //                           numberOfProjectors +
                        //                         projectorIndex_j]
                        //   << std::endl;
                        projectorIndex_j++;
                      } // mQuantumNumber_j
                  }     // alpha_j
                projectorIndex_i++;
              } // mQuantumNumber_i
          }     // alpha_i
        const char          uplo = 'L';
        const dftfe::Int    N    = numberOfProjectors;
        std::vector<double> A    = fullMultipoleTable;
        if (d_verbosity >= 5)
          {
            pcout << "Multipole Table: " << std::endl;
            for (dftfe::Int i = 0; i < numberOfProjectors; i++)
              {
                for (dftfe::Int j = 0; j < numberOfProjectors; j++)
                  pcout << A[i * numberOfProjectors + j] << " ";
                pcout << std::endl;
              }
          }

        dftfe::linearAlgebraOperations::inverse(&A[0], N);
        d_multipoleInverse[atomicNumber] = A;
        if (d_verbosity >= 5)
          {
            pcout << "Multipole Table Inverse: " << std::endl;
            for (dftfe::Int i = 0; i < numberOfProjectors; i++)
              {
                for (dftfe::Int j = 0; j < numberOfProjectors; j++)
                  pcout << A[i * numberOfProjectors + j] << " ";
                pcout << std::endl;
              }
          }


      } //*it
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::TotalCompensationCharge()
  {
    double            normValue = 0.0;
    const dftfe::uInt numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    const dftfe::uInt numberQuadraturePoints =
      (d_jxwcompensationCharge.begin()->second).size();
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    dftfe::uInt iElem = 0;

    for (std::map<dealii::CellId, std::vector<double>>::iterator it =
           (*d_bQuadValuesAllAtoms).begin();
         it != (*d_bQuadValuesAllAtoms).end();
         ++it)
      {
        const dealii::CellId      cellId = it->first;
        const std::vector<double> Temp =
          (*d_bQuadValuesAllAtoms).find(it->first)->second;
        const dftfe::uInt elementIndex =
          d_BasisOperatorElectroHostPtr->cellIndex(cellId);
        const std::vector<double> jxw =
          d_jxwcompensationCharge.find(it->first)->second;
        for (dftfe::uInt q_point = 0; q_point < numberQuadraturePoints;
             q_point++)
          {
            normValue += Temp[q_point] * jxw[q_point];
          }
        iElem++;
      }


    d_TotalCompensationCharge =
      dealii::Utilities::MPI::sum(normValue, d_mpiCommParent);
    pcout << "Total Compensation Charge: " << d_TotalCompensationCharge
          << std::endl;
    return d_TotalCompensationCharge;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::chargeNeutrality(double integralRhoValue,
                                                     TypeOfField typeOfField,
                                                     bool computeCompCharge)
  {
    if (computeCompCharge)
      {
        computeCompensationChargeMemoryOpt(TypeOfField::In);
      }
    double integralCompCharge = TotalCompensationCharge();
    pcout << "----------------------------------------------------"
          << std::endl;
    pcout << "Integral nTilde : " << integralRhoValue << std::endl;
    pcout << "Integral nTilde + nTildeCore: "
          << d_integralCoreDensity + integralRhoValue << std::endl;
    pcout << "Inegral Comp charge: " << integralCompCharge << std::endl;
    pcout << "Charge Neutrality error: "
          << (integralRhoValue + d_integralCoreDensity + integralCompCharge)
          << std::endl;
    pcout << "----------------------------------------------------"
          << std::endl;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::fillDijMatrix(
    TypeOfField                typeOfField,
    const dftfe::Int           iComp,
    const std::vector<double> &DijVector,
    const MPI_Comm            &interpoolcomm,
    const MPI_Comm            &interBandGroupComm)
  {
    std::vector<dftfe::uInt> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    {
      std::map<TypeOfField, std::map<dftfe::uInt, std::vector<double>>>
        &D_ijVector = D_ij[iComp];
      for (dftfe::Int iAtom = 0; iAtom < atomicNumber.size(); iAtom++)
        {
          dftfe::uInt Znum   = atomicNumber[iAtom];
          dftfe::uInt atomId = iAtom;
          dftfe::uInt numberOfProjectors =
            d_atomicProjectorFnsContainer
              ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
          D_ijVector[typeOfField][atomId] =
            std::vector<double>(numberOfProjectors * numberOfProjectors, 0.0);
        }

      // std::vector<double> DijTotalVector(d_nProjSqTotal, 0.0);
      const std::vector<dftfe::uInt> ownedAtomIds =
        d_nonLocalOperator->getOwnedAtomIdsInCurrentProcessor();
      if (ownedAtomIds.size() > 0)
        {
          dftfe::uInt index = 0;
          for (dftfe::Int iAtom = 0; iAtom < ownedAtomIds.size(); iAtom++)
            {
              dftfe::uInt atomId = ownedAtomIds[iAtom];
              dftfe::uInt Znum   = atomicNumber[atomId];
              dftfe::uInt numberOfProjectors =
                d_atomicProjectorFnsContainer
                  ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
              std::vector<double> Dij(numberOfProjectors * numberOfProjectors,
                                      0.0);
              for (dftfe::uInt i = 0; i < numberOfProjectors; i++)
                {
                  for (dftfe::uInt j = 0; j < numberOfProjectors; j++)
                    {
                      Dij[i * numberOfProjectors + j] = DijVector[index];
                      index++;
                    }
                }
              D_ijVector[typeOfField][atomId] = Dij;
            }
        }
    }
    communicateDijAcrossAllProcessors(
      typeOfField, iComp, interpoolcomm, interBandGroupComm, false);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<double>
  pawClass<ValueType, memorySpace>::getDijWeights()
  {
    std::vector<double>            weights;
    const std::vector<dftfe::uInt> ownedAtomIds =
      d_nonLocalOperator->getOwnedAtomIdsInCurrentProcessor();
    for (dftfe::uInt iAtom = 0; iAtom < ownedAtomIds.size(); iAtom++)
      {
        dftfe::uInt              atomId = ownedAtomIds[iAtom];
        std::vector<dftfe::uInt> atomicNumber =
          d_atomicProjectorFnsContainer->getAtomicNumbers();
        dftfe::uInt Znum = atomicNumber[atomId];
        dftfe::uInt numProj =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        dftfe::uInt numRadProj =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        std::vector<double> multipoleTable = d_multipole[Znum];

        std::vector<double> multipoleTableVal(numProj * numProj, 0.0);
        dftfe::Int          projectorIndex_i = 0;
        for (dftfe::Int alpha_i = 0; alpha_i < numRadProj; alpha_i++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_i =
              sphericalFunction.find(std::make_pair(Znum, alpha_i))->second;
            dftfe::Int lQuantumNo_i = sphFn_i->getQuantumNumberl();
            for (dftfe::Int mQuantumNo_i = -lQuantumNo_i;
                 mQuantumNo_i <= lQuantumNo_i;
                 mQuantumNo_i++)
              {
                dftfe::Int projectorIndex_j = 0;
                for (dftfe::Int alpha_j = 0; alpha_j < numRadProj; alpha_j++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn_j =
                      sphericalFunction.find(std::make_pair(Znum, alpha_j))
                        ->second;
                    dftfe::Int lQuantumNo_j = sphFn_j->getQuantumNumberl();
                    for (dftfe::Int mQuantumNo_j = -lQuantumNo_j;
                         mQuantumNo_j <= lQuantumNo_j;
                         mQuantumNo_j++)
                      {
                        weights.push_back(
                          pow(multipoleTable[alpha_i * numRadProj + alpha_j] *
                                gaunt(lQuantumNo_i,
                                      lQuantumNo_j,
                                      0,
                                      mQuantumNo_i,
                                      mQuantumNo_j,
                                      0) *
                                sqrt(4 * M_PI),
                              2));
                        projectorIndex_j++;
                      }
                  }
                projectorIndex_i++;
              }
          }
      }

    MPI_Barrier(d_mpiCommParent);
    return weights;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeIntegralCoreDensity(
    const std::map<dealii::CellId, std::vector<double>> &rhoCore)
  {
    d_BasisOperatorElectroHostPtr->reinit(0, 0, d_densityQuadratureIdElectro);

    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                JxwVector     = d_BasisOperatorElectroHostPtr->JxWBasisData();
    dftfe::uInt numQuadPoints = d_BasisOperatorElectroHostPtr->nQuadsPerCell();
    MPI_Barrier(d_mpiCommParent);
    double totalCoreDensity = 0.0;
    if (rhoCore.size() > 0)
      {
        for (std::map<dealii::CellId, std::vector<double>>::const_iterator it =
               rhoCore.cbegin();
             it != rhoCore.cend();
             ++it)
          {
            const std::vector<double> &Value = it->second;
            const std::vector<double> &ValueCorrection =
              d_rhoCoreCorrectionValues.find(it->first) !=
                  d_rhoCoreCorrectionValues.end() ?
                d_rhoCoreCorrectionValues.find(it->first)->second :
                std::vector<double>(numQuadPoints, 0.0);
            dftfe::uInt cellIndex =
              d_BasisOperatorHostPtr->cellIndex(it->first);
            for (dftfe::Int qpoint = 0; qpoint < numQuadPoints; qpoint++)
              totalCoreDensity += (Value[qpoint] - ValueCorrection[qpoint]) *
                                  JxwVector[cellIndex * numQuadPoints + qpoint];
          }
      }
    d_BasisOperatorElectroHostPtr->reinit(
      0, 0, d_compensationChargeQuadratureIdElectro);
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                JxwVectorComp = d_BasisOperatorElectroHostPtr->JxWBasisData();
    dftfe::uInt numQuadPointsComp =
      d_BasisOperatorElectroHostPtr->nQuadsPerCell();
    if (d_rhoCoreRefinedValues.size() > 0)
      {
        for (std::map<dealii::CellId, std::vector<double>>::const_iterator it =
               d_rhoCoreRefinedValues.cbegin();
             it != d_rhoCoreRefinedValues.cend();
             ++it)
          {
            const std::vector<double> &Value = it->second;
            dftfe::uInt                cellIndex =
              d_BasisOperatorHostPtr->cellIndex(it->first);
            for (dftfe::Int qpoint = 0; qpoint < numQuadPointsComp; qpoint++)
              totalCoreDensity +=
                (Value[qpoint]) *
                JxwVectorComp[cellIndex * numQuadPointsComp + qpoint];
          }
      }
    d_integralCoreDensity =
      dealii::Utilities::MPI::sum(totalCoreDensity, d_mpiCommParent);
    d_integrealCoreDensityRadial = 0.0;
    std::vector<dftfe::uInt> atomicNumbers =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    for (dftfe::Int iAtom = 0; iAtom < atomicNumbers.size(); iAtom++)
      {
        dftfe::uInt Znum = atomicNumbers[iAtom];
        d_integrealCoreDensityRadial += d_integralCoreDensityPerAtom[Znum];
      }
    pcout
      << "PAW Class: Error in integralCoreDensity with radial data and FEM: "
      << std::fabs(d_integralCoreDensity - d_integrealCoreDensityRadial)
      << std::endl;
    if (std::fabs(d_integralCoreDensity - d_integrealCoreDensityRadial) > 1E-4)
      pcout << "PAW Class: Warning!! Increase density quadrature rule: "
            << std::endl;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<dftfe::uInt>
  pawClass<ValueType, memorySpace>::relevantAtomdIdsInCurrentProcs(
    std::vector<dftfe::uInt>                         &numberOfCellsPerAtom,
    std::vector<std::pair<dftfe::uInt, dftfe::uInt>> &atomNumberNCellsPair)
  {
    numberOfCellsPerAtom.clear();
    atomNumberNCellsPair.clear();
    const dftfe::uInt numberNodesPerElement =
      d_BasisOperatorHostPtr->nDofsPerCell();
    std::vector<dftfe::uInt> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    numberOfCellsPerAtom.resize(atomicNumber.size(), 0);
    const std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    for (dftfe::Int iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        dftfe::uInt              atomId = atomIdsInCurrentProcess[iAtom];
        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomicProjectorFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        numberOfCellsPerAtom[atomId] +=
          elementIndexesInAtomCompactSupport.size();
      }
    MPI_Barrier(d_mpiCommParent);
    MPI_Allreduce(MPI_IN_PLACE,
                  &numberOfCellsPerAtom[0],
                  atomicNumber.size(),
                  MPI_UNSIGNED,
                  MPI_SUM,
                  d_mpiCommParent);
    MPI_Barrier(d_mpiCommParent);
    for (dftfe::Int iAtom = 0; iAtom < atomicNumber.size(); iAtom++)
      {
        pcout << "Total cells in compact support: " << iAtom << " "
              << atomicNumber[iAtom] << " " << numberOfCellsPerAtom[iAtom]
              << std::endl;
      }
    pcout << "---------" << std::endl;
    MPI_Barrier(d_mpiCommParent);
    std::vector<dftfe::uInt> globalAtomsOfInt;
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        std::set<dftfe::uInt> nCells;
        for (dftfe::Int iAtom = 0; iAtom < atomicNumber.size(); iAtom++)
          {
            if (*it == atomicNumber[iAtom])
              {
                nCells.insert(numberOfCellsPerAtom[iAtom]);
              }
          }
        for (std::set<dftfe::uInt>::iterator it1 = nCells.begin();
             it1 != nCells.end();
             it1++)
          {
            for (dftfe::uInt iAtom = 0; iAtom < atomicNumber.size(); iAtom++)
              {
                if (*it == atomicNumber[iAtom] &&
                    *it1 == numberOfCellsPerAtom[iAtom])
                  {
                    globalAtomsOfInt.push_back(iAtom);
                    atomNumberNCellsPair.push_back(
                      std::make_pair(iAtom, numberOfCellsPerAtom[iAtom]));
                    break;
                  }
              }
          }
      }
    const dftfe::uInt natoms = globalAtomsOfInt.size();
    pcout << "PAW Class: No. of atoms to be considered for apprx Sinv: "
          << natoms << std::endl;
    dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::HOST>
      atomOwnedVector;
    atomOwnedVector.reinit(d_BasisOperatorHostPtr->mpiPatternP2P, natoms);
    atomOwnedVector.setValue(0);


    for (dftfe::Int iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        dftfe::uInt atomId = atomIdsInCurrentProcess[iAtom];

        if (std::find(globalAtomsOfInt.begin(),
                      globalAtomsOfInt.end(),
                      atomId) != globalAtomsOfInt.end())
          {
            dftfe::uInt atomIndex = std::find(globalAtomsOfInt.begin(),
                                              globalAtomsOfInt.end(),
                                              atomId) -
                                    globalAtomsOfInt.begin();
            if (atomIndex > natoms)
              std::cout << "Line 1312" << std::endl;
            std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
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
                for (dftfe::uInt iDof = 0; iDof < numberNodesPerElement; iDof++)
                  {
                    dftfe::uInt dofIndex =
                      d_BasisOperatorHostPtr->d_cellDofIndexToProcessDofIndexMap
                        [elementIndex * numberNodesPerElement + iDof];
                    *(atomOwnedVector.data() + (dofIndex * natoms) +
                      atomIndex) += 1.0;
                  } // iDof


              } // iElem
          }

      } // iAtom
    d_BasisOperatorHostPtr
      ->d_constraintInfo[d_BasisOperatorHostPtr->d_dofHandlerID]
      .distribute_slave_to_master(atomOwnedVector);
    atomOwnedVector.accumulateAddLocallyOwned();
    atomOwnedVector.zeroOutGhosts();
    std::vector<double> atomsPresent(natoms, 0.0);
    for (dftfe::uInt iDof = 0; iDof < atomOwnedVector.locallyOwnedSize();
         iDof++)
      {
        std::transform(atomOwnedVector.data() + iDof * natoms,
                       atomOwnedVector.data() + iDof * natoms + natoms,
                       atomsPresent.data(),
                       atomsPresent.data(),
                       [](auto &p, auto &q) { return p + q; });
      }
    std::vector<dftfe::uInt> totalAtomIdsInProcessor;
    for (dftfe::uInt iAtom = 0; iAtom < natoms; iAtom++)
      {
        if (atomsPresent[iAtom] > 0)
          totalAtomIdsInProcessor.push_back(globalAtomsOfInt[iAtom]);
      }
    return (totalAtomIdsInProcessor);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<dftfe::uInt>
  pawClass<ValueType, memorySpace>::relevantAtomdIdsInCurrentProcs()
  {
    std::vector<dftfe::uInt> atomicNumberTemp =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    std::vector<dftfe::uInt> numberOfCellsPerAtom;
    numberOfCellsPerAtom.resize(atomicNumberTemp.size(), 0);
    const std::vector<dftfe::uInt> atomIdsInCurrentProcessTemp =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    for (dftfe::Int iAtom = 0; iAtom < atomIdsInCurrentProcessTemp.size();
         iAtom++)
      {
        dftfe::uInt              atomId = atomIdsInCurrentProcessTemp[iAtom];
        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomicProjectorFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        numberOfCellsPerAtom[atomId] +=
          elementIndexesInAtomCompactSupport.size();
      }
    MPI_Barrier(d_mpiCommParent);
    MPI_Allreduce(MPI_IN_PLACE,
                  &numberOfCellsPerAtom[0],
                  atomicNumberTemp.size(),
                  MPI_UNSIGNED,
                  MPI_SUM,
                  d_mpiCommParent);
    MPI_Barrier(d_mpiCommParent);
    // for (dftfe::Int iAtom = 0; iAtom < atomicNumberTemp.size(); iAtom++)
    //   {
    //     pcout << "Total cells in compact support: " << iAtom << " "
    //           << atomicNumberTemp[iAtom] << " " <<
    //           numberOfCellsPerAtom[iAtom]
    //           << std::endl;
    //   }
    // pcout << "---------" << std::endl;
    const dftfe::uInt numberNodesPerElement =
      d_BasisOperatorHostPtr->nDofsPerCell();
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const dftfe::uInt natoms = atomicNumber.size();
    dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::HOST>
      atomOwnedVector;
    atomOwnedVector.reinit(d_BasisOperatorHostPtr->mpiPatternP2P, natoms);
    atomOwnedVector.setValue(0);
    std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();

    for (dftfe::Int iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        dftfe::uInt atomId = atomIdsInCurrentProcess[iAtom];

        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomicProjectorFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        dftfe::Int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();

        for (dftfe::Int iElem = 0; iElem < numberElementsInAtomCompactSupport;
             iElem++)
          {
            dftfe::uInt elementIndex =
              elementIndexesInAtomCompactSupport[iElem];
            for (dftfe::uInt iDof = 0; iDof < numberNodesPerElement; iDof++)
              {
                dftfe::uInt dofIndex =
                  d_BasisOperatorHostPtr->d_cellDofIndexToProcessDofIndexMap
                    [elementIndex * numberNodesPerElement + iDof];
                *(atomOwnedVector.data() + (dofIndex * natoms) + atomId) += 1.0;

              } // iDof


          } // iElem

      } // iAtom
    d_BasisOperatorHostPtr
      ->d_constraintInfo[d_BasisOperatorHostPtr->d_dofHandlerID]
      .distribute_slave_to_master(atomOwnedVector);
    atomOwnedVector.accumulateAddLocallyOwned();
    atomOwnedVector.zeroOutGhosts();
    std::vector<double> atomsPresent(natoms, 0.0);
    for (dftfe::uInt iDof = 0; iDof < atomOwnedVector.locallyOwnedSize();
         iDof++)
      {
        std::transform(atomOwnedVector.data() + iDof * natoms,
                       atomOwnedVector.data() + iDof * natoms + natoms,
                       atomsPresent.data(),
                       atomsPresent.data(),
                       [](auto &p, auto &q) { return p + q; });
      }
    std::vector<dftfe::uInt> totalAtomIdsInProcessor;
    for (dftfe::Int iAtom = 0; iAtom < natoms; iAtom++)
      {
        if (atomsPresent[iAtom] > 0)
          totalAtomIdsInProcessor.push_back(iAtom);
      }
    // std::cout << "Number of relevant atoms and local atoms in procs: "
    //           << totalAtomIdsInProcessor.size() << " "
    //           << atomIdsInCurrentProcess.size() << " " << d_this_mpi_process
    //           << std::endl;
    return (totalAtomIdsInProcessor);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::computeNormDij(
    std::vector<double> &DijResidual)
  {
    std::vector<double> Weights = getDijWeights();
    AssertThrow(DijResidual.size() == Weights.size(),
                dealii::ExcMessage("PAW:: Mixing issue for Dij "));
    double norm = 0.0;
    for (dftfe::uInt i = 0; i < DijResidual.size(); i++)
      norm += DijResidual[i] * DijResidual[i] * Weights[i];
    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, d_mpiCommParent);
    return sqrt(norm);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeAugmentationOverlap()
  {
    double     maxOverlap = 0.0;
    dftfe::Int srcAtom    = -1;
    dftfe::Int dstAtom    = -1;
    if (d_LocallyOwnedAtomId.size() > 0)
      {
        const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        std::vector<dftfe::uInt> atomicNumbers =
          d_atomicProjectorFnsContainer->getAtomicNumbers();
        const std::vector<double> &atomCoordinates =
          d_atomicProjectorFnsContainer->getAtomCoordinates();
        const std::map<dftfe::uInt, std::vector<double>> &periodicImageCoord =
          d_atomicProjectorFnsContainer->getPeriodicImageCoordinatesList();
        std::vector<double> dCord(3, 0.0);

        for (dftfe::Int iAtomList = 0; iAtomList < d_LocallyOwnedAtomId.size();
             iAtomList++)
          {
            dftfe::uInt         atomId     = d_LocallyOwnedAtomId[iAtomList];
            const dftfe::uInt   Znum       = atomicNumbers[atomId];
            const double        rmaxAugsrc = d_RmaxAug[Znum];
            std::vector<double> sourceCoord(3, 0.0);
            sourceCoord[0] = atomCoordinates[3 * atomId + 0];
            sourceCoord[1] = atomCoordinates[3 * atomId + 1];
            sourceCoord[2] = atomCoordinates[3 * atomId + 2];
            for (dftfe::uInt iAtom = 0; iAtom < atomicNumbers.size(); iAtom++)
              {
                if (iAtom != atomId)
                  {
                    std::vector<double> imageCoordinates =
                      periodicImageCoord.find(iAtom)->second;
                    const double rmaxAugDst   = d_RmaxAug[atomicNumbers[iAtom]];
                    double       idleDistance = rmaxAugsrc + rmaxAugDst;
                    dftfe::uInt  imageIdsSize = imageCoordinates.size() / 3;
                    for (dftfe::Int iImage = 0; iImage < imageIdsSize; iImage++)
                      {
                        if (iImage == 0)
                          {
                            dCord[0] = (sourceCoord[0] -
                                        imageCoordinates[3 * iImage + 0]);
                            dCord[1] = (sourceCoord[1] -
                                        imageCoordinates[3 * iImage + 1]);
                            dCord[2] = (sourceCoord[2] -
                                        imageCoordinates[3 * iImage + 2]);
                          }
                        else
                          {
                            dCord[0] =
                              (sourceCoord[0] - atomCoordinates[3 * iAtom + 0]);
                            dCord[1] =
                              (sourceCoord[1] - atomCoordinates[3 * iAtom + 1]);
                            dCord[2] =
                              (sourceCoord[2] - atomCoordinates[3 * iAtom + 2]);
                          }

                        double distance =
                          std::sqrt(dCord[0] * dCord[0] + dCord[1] * dCord[1] +
                                    dCord[2] * dCord[2]);
                        double ratio =
                          (idleDistance - distance) / idleDistance * 100;
                        if (maxOverlap < ratio)
                          {
                            maxOverlap = ratio;
                            srcAtom    = atomId;
                            dstAtom    = iAtom;
                          }
                      }
                  }
              }
          }
      }
    double maxOverlapOverall;
    MPI_Allreduce(
      &maxOverlap, &maxOverlapOverall, 1, MPI_DOUBLE, MPI_MAX, d_mpiCommParent);
    pcout << "Max Overlap in system: " << maxOverlapOverall << std::endl;
    if (std::fabs(maxOverlapOverall - maxOverlap) < 1E-8 &&
        maxOverlapOverall > 1E-8)
      {
        std::cout << "Overlap between atoms: " << srcAtom << " and " << dstAtom
                  << " is: " << maxOverlap << std::flush << std::endl;
      }
    MPI_Barrier(d_mpiCommParent);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::checkOverlapAugmentation()
  {
    const dftfe::uInt totalLocallyOwnedCells = d_BasisOperatorHostPtr->nCells();
    const dftfe::uInt nodesPerElement = d_BasisOperatorHostPtr->nDofsPerCell();
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    std::vector<dftfe::Int>        elementsPerAtom(atomicNumber.size(), 0);
    const std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    for (dftfe::Int iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        dftfe::uInt              atomId = atomIdsInCurrentProcess[iAtom];
        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomicProjectorFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        elementsPerAtom[atomId] = elementIndexesInAtomCompactSupport.size();
        // if (atomId == 0)
        //   std::cout << "Rank and No of elements: " << d_this_mpi_process <<
        //   "
        //   "
        //             << elementIndexesInAtomCompactSupport.size() <<
        //             std::endl;
      }
    MPI_Allreduce(MPI_IN_PLACE,
                  &elementsPerAtom[0],
                  atomicNumber.size(),
                  MPI_INT,
                  MPI_SUM,
                  d_mpiCommParent);

    // for (dftfe::Int iAtom = 0; iAtom < atomicNumber.size(); iAtom++)
    //   {
    //     pcout << "Number of elements for AtomID: " << iAtom << " "
    //           << elementsPerAtom[iAtom] << std::endl;
    //   }
    // std::cout << std::flush;
    MPI_Barrier(d_mpiCommParent);
    for (dftfe::uInt iCell = 0; iCell < totalLocallyOwnedCells; iCell++)
      {
        const std::vector<dftfe::Int> atomIdsInCell =
          d_atomicProjectorFnsContainer->getAtomIdsInElement(iCell);
        if (atomIdsInCell.size() > 1)
          {
            std::cout << "More than 1 atom present in iCell in rank: " << iCell
                      << " " << d_this_mpi_process << " "
                      << atomIdsInCell.size() << std::endl;
            std::vector<std::vector<ValueType>> CMatrixEntries;
            for (dftfe::Int iAtom = 0; iAtom < atomIdsInCell.size(); iAtom++)
              {
                CMatrixEntries.push_back(d_nonLocalOperator->getCmatrixEntries(
                  0, atomIdsInCell[iAtom], iCell));
              }
            for (dftfe::Int iNode = 0; iNode < nodesPerElement; iNode++)
              {
                std::vector<double> Values(atomIdsInCell.size(), 0.0);
                for (dftfe::Int iAtom = 0; iAtom < atomIdsInCell.size();
                     iAtom++)
                  {
                    dftfe::uInt atomId = atomIdsInCell[iAtom];
                    dftfe::uInt Znum   = atomicNumber[atomId];
                    dftfe::uInt numProj =
                      d_atomicProjectorFnsContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    for (dftfe::Int iProj = 0; iProj < numProj; iProj++)
                      {
                        if (Values[iAtom] <
                            std::abs(
                              (CMatrixEntries[iAtom][iNode * numProj + iProj])))
                          Values[iAtom] = std::abs(
                            (CMatrixEntries[iAtom][iNode * numProj + iProj]));
                      }
                  }
                std::sort(Values.begin(), Values.end(), std::greater<double>());
                // if (Values[0] > 1E-8)
                //   std::cout << "PAW Warning: Nodal overlap of atoms "
                //             << Values[0] << " " << Values[1] << std::endl;
              }
          }
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeproductOfCGMultipole()
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
        std::vector<double> multipole = d_multipole[*it];
        const dftfe::uInt   NumRadialSphericalFunctions =
          d_atomicShapeFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        const dftfe::uInt NumTotalSphericalFunctions =
          d_atomicShapeFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
            Znum);
        const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicShapeFnsContainer->getSphericalFunctions();
        const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          projectorFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        std::vector<double> productValues(
          NumTotalSphericalFunctions * numberOfProjectors * numberOfProjectors);
        dftfe::uInt Lindex = 0;
        for (dftfe::uInt alpha = 0; alpha < NumRadialSphericalFunctions;
             ++alpha)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
              sphericalFunction.find(std::make_pair(Znum, alpha))->second;
            dftfe::Int lQuantumNumber = sphFn->getQuantumNumberl();

            for (dftfe::Int mQuantumNumber = -lQuantumNumber;
                 mQuantumNumber <= lQuantumNumber;
                 mQuantumNumber++)
              {
                dftfe::uInt alpha_i = 0;
                for (dftfe::Int i = 0; i < numberOfRadialProjectors; i++)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> projFnI =
                      projectorFunction.find(std::make_pair(Znum, i))->second;
                    dftfe::Int l_i = projFnI->getQuantumNumberl();
                    for (dftfe::Int m_i = -l_i; m_i <= l_i; m_i++)
                      {
                        dftfe::uInt alpha_j = 0;
                        for (dftfe::Int j = 0; j < numberOfRadialProjectors;
                             j++)
                          {
                            std::shared_ptr<AtomCenteredSphericalFunctionBase>
                              projFnJ =
                                projectorFunction.find(std::make_pair(Znum, j))
                                  ->second;
                            dftfe::Int l_j = projFnJ->getQuantumNumberl();
                            for (dftfe::Int m_j = -l_j; m_j <= l_j; m_j++)
                              {
                                double multipolevalue =
                                  multipole[lQuantumNumber *
                                              numberOfRadialProjectors *
                                              numberOfRadialProjectors +
                                            i * numberOfRadialProjectors + j];
                                double      Cijl = gaunt(l_i,
                                                    l_j,
                                                    lQuantumNumber,
                                                    m_i,
                                                    m_j,
                                                    mQuantumNumber);
                                dftfe::uInt loc  = Lindex * numberOfProjectors *
                                                    numberOfProjectors +
                                                  alpha_i * numberOfProjectors +
                                                  alpha_j;
                                productValues[loc] = Cijl * multipolevalue;

                                alpha_j++;
                              } // m_j


                          } // j
                        alpha_i++;
                      } // m_i

                  } // i

                Lindex++;
              } // mQuantumNumber

          } // lQuantumNumber

        d_productOfMultipoleClebshGordon[Znum] = productValues;
      }
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::saveDeltaSinverseEntriesToFile()
  {
#ifdef USE_COMPLEX
    pcout << "Not available for complex" << std::endl;
    std::exit(0);
#else
    if (d_this_mpi_process == 0)
      {
        pcout << "Saving DeltaSinverse Entries: " << std::endl;
        const std::vector<dftfe::uInt> &atomicNumber =
          d_atomicProjectorFnsContainer->getAtomicNumbers();
        for (dftfe::uInt atomId = 0; atomId < atomicNumber.size(); atomId++)
          {
            dftfe::uInt Znum = atomicNumber[atomId];
            dftfe::uInt numberOfProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            // ReadFile from file
            char SinverseFileName[256];
            strcpy(SinverseFileName,
                   ("sinverseAtomId" + std::to_string(atomId)).c_str());
            std::vector<ValueType> SinverseEntries =
              d_atomicNonLocalPseudoPotentialConstants
                [CouplingType::inverseOverlapEntries][atomId];
            dftUtils::writeDataIntoFile(SinverseEntries,
                                        SinverseFileName,
                                        d_mpiCommParent);
          }
      }
#endif
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  int
  pawClass<ValueType, memorySpace>::loadDeltaSinverseEntriesFromFile()
  {
#ifdef USE_COMPLEX
    pcout << "Not available for complex" << std::endl;
    std::exit(0);
#else

    pcout << "Loading DeltaSinverse Entries: " << std::endl;
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    for (dftfe::uInt atomId = 0; atomId < atomicNumber.size(); atomId++)
      {
        dftfe::uInt Znum = atomicNumber[atomId];
        dftfe::uInt numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        // ReadFile from file
        char SinverseFileName[256];
        strcpy(SinverseFileName,
               ("sinverseAtomId" + std::to_string(atomId)).c_str());
        std::vector<double> SinverseEntries;
        dftUtils::readFile(SinverseEntries, SinverseFileName);
        if (SinverseEntries.size() != numberOfProjectors * numberOfProjectors)
          {
            pcout << "AtomID " << atomId
                  << " Sinverse coupling matrix not found" << std::endl;
            return (0);
          }
        std::vector<ValueType> SinverseEntriesCopy(SinverseEntries.begin(),
                                                   SinverseEntries.end());
        d_atomicNonLocalPseudoPotentialConstants
          [CouplingType::inverseOverlapEntries][atomId] = SinverseEntriesCopy;
      }
    return (1);
#endif
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::saveDijEntriesToFile(
    const MPI_Comm &mpiCommParent,
    TypeOfField     typeOfField)
  {
    dftfe::uInt thisMpiRankParent =
      dealii::Utilities::MPI::this_mpi_process(mpiCommParent);
    if (thisMpiRankParent == 0)
      {
        pcout << "Saving Dij Out Entries: " << std::endl;
        const std::vector<dftfe::uInt> &atomicNumber =
          d_atomicProjectorFnsContainer->getAtomicNumbers();
        for (dftfe::Int iComp = 0; iComp < D_ij.size(); iComp++)
          {
            std::map<TypeOfField, std::map<dftfe::uInt, std::vector<double>>>
              &D_ijComp = D_ij[iComp];
            for (dftfe::uInt atomId = 0; atomId < atomicNumber.size(); atomId++)
              {
                dftfe::uInt Znum = atomicNumber[atomId];
                dftfe::uInt numberOfProjectors =
                  d_atomicProjectorFnsContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                // ReadFile from file
                char DijFileName[256];
                strcpy(DijFileName,
                       ("Dij" + std::to_string(iComp) + "AtomId" +
                        std::to_string(atomId))
                         .c_str());
                std::vector<double> Dij = D_ijComp[typeOfField][atomId];
                dftUtils::writeDataIntoFile(Dij, DijFileName, mpiCommParent);
              }
          }
      }
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::loadDijEntriesFromFile()
  {
    pcout << "Loading Dij Entries: " << std::endl;
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    for (dftfe::Int iComp = 0; iComp < D_ij.size(); iComp++)
      {
        std::map<TypeOfField, std::map<dftfe::uInt, std::vector<double>>>
          &D_ijComp = D_ij[iComp];
        for (dftfe::uInt atomId = 0; atomId < atomicNumber.size(); atomId++)
          {
            dftfe::uInt Znum = atomicNumber[atomId];
            dftfe::uInt numberOfProjectors =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            // ReadFile from file
            char DijFileName[256];
            strcpy(DijFileName,
                   ("Dij" + std::to_string(iComp) + "AtomId" +
                    std::to_string(atomId))
                     .c_str());
            std::vector<double> DijEntries;
            dftUtils::readFile(DijEntries, DijFileName);
            if (DijEntries.size() != numberOfProjectors * numberOfProjectors)
              {
                pcout << "AtomID " << atomId << " Dij matrix not found"
                      << std::endl;
              }

            D_ijComp[TypeOfField::In][atomId] = DijEntries;
          }
      }
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<double>
  pawClass<ValueType, memorySpace>::computeZeroPotentialFromLocalPotential(
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
    const double               NtildeValence)
  {
    const dftfe::uInt   numEntries = radialMesh.size();
    std::vector<double> xcInputDensity1(numEntries, 0.0);
    std::vector<double> xcInputDensity2(numEntries, 0.0);
    std::vector<double> HartreeDensity(numEntries, 0.0);

    std::vector<double> exchangePotentialVal1(numEntries);
    std::vector<double> corrPotentialVal1(numEntries);
    std::vector<double> exchangePotentialVal2(numEntries);
    std::vector<double> corrPotentialVal2(numEntries);


    double scalarFactorXC      = (NValence - NtildeValence);
    double scalarFactorHartree = (Ncore - NtildeCore - Znum);
    pcout << "Density data " << std::endl;
    for (dftfe::Int irad = 0; irad < numEntries; irad++)
      {
        xcInputDensity2[irad] = (valenceDensity[irad] + coreDensityPS[irad] +
                                 scalarFactorXC * shapeFunction0[irad]) /
                                sqrt(4 * M_PI);
        xcInputDensity1[irad] =
          (valenceDensity[irad] + coreDensityPS[irad]) / sqrt(4 * M_PI);
        HartreeDensity[irad] =
          (coreDensityPS[irad] + scalarFactorHartree * shapeFunction0[irad]);
        pcout << xcInputDensity1[irad] << " " << xcInputDensity2[irad] << " "
              << HartreeDensity[irad] << std::endl;
      }
    pcout << "----------------------------------------" << std::endl;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<double>
  pawClass<ValueType, memorySpace>::computeLocalPotentialFromZeroPotential(
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
    const double               NtildeValence)
  {
    const dftfe::uInt   numEntries = radialMesh.size();
    std::vector<double> xcInputDensity1(numEntries, 0.0);
    std::vector<double> xcInputDensity2(numEntries, 0.0);
    std::vector<double> HartreeDensity(numEntries, 0.0);
    std::vector<double> HartreePotential(numEntries, 0.0);
    std::vector<double> localPotential(numEntries, 0.0);


    std::vector<double> exchangePotentialVal1(numEntries);
    std::vector<double> corrPotentialVal1(numEntries);
    std::vector<double> exchangePotentialVal2(numEntries);
    std::vector<double> corrPotentialVal2(numEntries);

    pcout << "Nvalence NtildeValence Ncore NTildeCore Znum " << NValence << " "
          << NtildeValence << " " << Ncore << " " << NtildeCore << " " << Znum
          << std::endl;
    double scalarFactorXC      = (NValence - NtildeValence);
    double scalarFactorHartree = (Ncore - NtildeCore - Znum);
    pcout << "ScalarFactorXC: " << scalarFactorXC << std::endl;
    pcout << "ScalarFactorHartree: " << scalarFactorHartree << std::endl;
    pcout << "Density data " << numEntries << std::endl;
    for (dftfe::Int irad = 0; irad < numEntries; irad++)
      {
        xcInputDensity2[irad] = (valenceDensity[irad] + coreDensityPS[irad] +
                                 scalarFactorXC * shapeFunction0[irad]) /
                                sqrt(4 * M_PI);
        xcInputDensity1[irad] =
          (valenceDensity[irad] + coreDensityPS[irad]) / sqrt(4 * M_PI);
        HartreeDensity[irad] =
          (coreDensityPS[irad] + scalarFactorHartree * shapeFunction0[irad]);
        pcout << xcInputDensity1[irad] << " " << xcInputDensity2[irad] << " "
              << HartreeDensity[irad] << std::endl;
      }
    pcout << "----------------------------------------" << std::endl;
    oneTermPoissonPotential(&HartreeDensity[0],
                            0,
                            0,
                            numEntries - 2,
                            2,
                            radialMesh,
                            rab,
                            HartreePotential);
    pcout << "Local Potential: " << std::endl;
    for (dftfe::Int irad = 0; irad < numEntries; irad++)
      {
        localPotential[irad] =
          zeroPotential[irad] + HartreePotential[irad] +
          (exchangePotentialVal1[irad] + corrPotentialVal1[irad] -
           exchangePotentialVal2[irad] - corrPotentialVal2[irad]) *
            sqrt(4 * M_PI);
        pcout << radialMesh[irad] << " " << zeroPotential[irad] << " "
              << HartreePotential[irad] << " "
              << (exchangePotentialVal1[irad] + corrPotentialVal1[irad] -
                  exchangePotentialVal2[irad] - corrPotentialVal2[irad]) *
                   sqrt(4 * M_PI)
              << " " << localPotential[irad] << std::endl;
      }
    return (localPotential);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::zeroPotentialContribution(
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityQuadValues,
    const std::map<dealii::CellId, std::vector<double>>
                     &zeroPotentalAtQuadPoints,
    const dftfe::uInt quadratureId,
    TypeOfField       typeOfField)
  {
    double result = 0.0;
    d_BasisOperatorElectroHostPtr->reinit(0, 0, quadratureId, false);
    std::map<TypeOfField, std::map<dftfe::uInt, std::vector<double>>> &D_ijRho =
      D_ij[0];
    const dftfe::uInt nQuadsPerCell =
      d_BasisOperatorElectroHostPtr->nQuadsPerCell();
    for (dftfe::uInt iCell = 0; iCell < d_BasisOperatorElectroHostPtr->nCells();
         ++iCell)
      {
        const auto cellIt = zeroPotentalAtQuadPoints.find(
          d_BasisOperatorElectroHostPtr->cellID(iCell));
        if (cellIt != zeroPotentalAtQuadPoints.end())
          {
            const std::vector<double> &cellFieldValues = cellIt->second;
            for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
              result += cellFieldValues[iQuad] *
                        densityQuadValues[0][iCell * nQuadsPerCell + iQuad] *
                        d_BasisOperatorElectroHostPtr
                          ->JxWBasisData()[iCell * nQuadsPerCell + iQuad];
          }
      }
    // MPI all reduce to get the result across all processes
    double totalLocalPseudoPotentialEnergy =
      dealii::Utilities::MPI::sum(result, d_mpiCommParent);
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    double totalZeroPotentialContribution = 0.0;
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
        std::vector<double> Dij    = D_ijRho[typeOfField][atomId];
        std::vector<double> Zeroij = d_zeroPotentialij[Znum];
        double              ZeroPotentialContribution = 0.0;
        for (int iProj = 0; iProj < numberOfProjectors; iProj++)
          {
            for (int jProj = 0; jProj < numberOfProjectors; jProj++)
              {
                ZeroPotentialContribution +=
                  Dij[iProj * numberOfProjectors + jProj] *
                  Zeroij[iProj * numberOfProjectors + jProj];

              } // jProj
          }     // iProj

        totalZeroPotentialContribution += ZeroPotentialContribution;
      } // iAtom
    pcout << "Initial Zero potetential " << totalZeroPotentialContribution
          << " " << totalLocalPseudoPotentialEnergy << " "
          << (totalLocalPseudoPotentialEnergy - totalZeroPotentialContribution)
          << std::endl;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  pawClass<ValueType, memorySpace>::computePAWCorrectionContribution(
    dftfe::uInt index,
    TypeOfField typeOfField)
  {
    std::map<dftfe::uInt, std::vector<double>> Dij = D_ij[index][typeOfField];
    double                                     totalValue = 0.0;
    std::vector<dftfe::uInt>                   atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    const std::vector<dftfe::uInt> ownedAtomIds =
      d_nonLocalOperator->getOwnedAtomIdsInCurrentProcessor();
    for (dftfe::uInt iAtom = 0; iAtom < ownedAtomIds.size(); iAtom++)
      {
        dftfe::uInt       atomId = ownedAtomIds[iAtom];
        dftfe::uInt       Znum   = atomicNumber[atomId];
        const dftfe::uInt numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        std::vector<ValueType> overlapMatrixEntries =
          d_atomicNonLocalPseudoPotentialConstants[CouplingType::OverlapEntries]
                                                  [Znum];
        double value = 0.0;
        for (dftfe::Int iProj = 0;
             iProj < numberOfProjectors * numberOfProjectors;
             iProj++)
          {
            value += dftfe::utils::realPart(overlapMatrixEntries[iProj]) *
                     Dij[atomId][iProj];
          }
        totalValue += value;
      }
    MPI_Allreduce(
      MPI_IN_PLACE, &totalValue, 1, MPI_DOUBLE, MPI_SUM, d_mpiCommParent);
    return (totalValue);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::determineAtomsOfInterstPseudopotential(
    const std::vector<std::vector<double>> &atomCoordinates)
  {
    d_atomLocationsInterestPseudopotential.clear();
    d_atomIdPseudopotentialInterestToGlobalId.clear();
    dftfe::uInt atomIdPseudo = 0;
    // pcout<<"Atoms of interest: "<<std::endl;
    for (dftfe::uInt iAtom = 0; iAtom < atomCoordinates.size(); iAtom++)
      {
        if (true)
          {
            d_atomLocationsInterestPseudopotential.push_back(
              atomCoordinates[iAtom]);
            d_atomIdPseudopotentialInterestToGlobalId[atomIdPseudo] = iAtom;
            // pcout<<iAtom<<" "<<atomIdPseudo<<" ";
            // for(dftfe::Int i = 0; i <
            // d_atomLocationsInterestPseudopotential[atomIdPseudo].size();
            // i++)
            //   pcout<<d_atomLocationsInterestPseudopotential[atomIdPseudo][i]<<"
            //   ";
            // pcout<<std::endl;
            atomIdPseudo++;
          }
      }
  }


  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
