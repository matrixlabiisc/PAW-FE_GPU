//
// Created by Kartick Ramakrishnan.
//

#include "AuxDensityMatrixRadial.h"
#include <Exceptions.h>
#include <iostream>

namespace dftfe
{
  namespace
  {
    void
    fillDensityAttributeData(
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &attributeData,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                                                &values,
      const std::pair<dftfe::uInt, dftfe::uInt> &indexRange)
    {
      dftfe::uInt startIndex = indexRange.first;
      dftfe::uInt endIndex   = indexRange.second;

      attributeData.resize(endIndex - startIndex);
      if (startIndex > endIndex || endIndex > values.size())
        {
          std::cout << "CHECK1A: " << startIndex << std::endl;
          std::cout << "CHECK1B: " << endIndex << std::endl;
          std::cout << "CHECK1C: " << values.size() << std::endl;
          throw std::invalid_argument("Invalid index range for densityData");
        }

      for (dftfe::uInt i = startIndex; i < endIndex; ++i)
        {
          attributeData[i - startIndex] = values[i];
        }
    }
  } // namespace
  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixRadial<memorySpace>::applyLocalOperations(
    const std::pair<dftfe::uInt, dftfe::uInt> &quadIndexRange,
    std::unordered_map<
      WfcDescriptorDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &wfcData)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixRadial<memorySpace>::applyLocalOperations(
    const std::pair<dftfe::uInt, dftfe::uInt> &quadIndexRange,
    std::unordered_map<
      DensityDescriptorDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityData)
  {
    std::pair<dftfe::uInt, dftfe::uInt> indexRangeVal;
    std::pair<dftfe::uInt, dftfe::uInt> indexRangeGrad;

    dftfe::uInt minIndex = quadIndexRange.first;
    indexRangeVal.first  = minIndex;
    indexRangeVal.second = quadIndexRange.second;

    indexRangeGrad.first  = quadIndexRange.first * 3;
    indexRangeGrad.second = quadIndexRange.second * 3;

    if (densityData.find(DensityDescriptorDataAttributes::valuesTotal) !=
        densityData.end())
      {
        fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::valuesTotal],
          d_densityValsTotalAllQuads,
          indexRangeVal);
      }

    if (densityData.find(DensityDescriptorDataAttributes::valuesSpinUp) !=
        densityData.end())
      {
        fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::valuesSpinUp],
          d_densityValsSpinUpAllQuads,
          indexRangeVal);
      }

    if (densityData.find(DensityDescriptorDataAttributes::valuesSpinDown) !=
        densityData.end())
      {
        fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::valuesSpinDown],
          d_densityValsSpinDownAllQuads,
          indexRangeVal);
      }

    if (densityData.find(DensityDescriptorDataAttributes::gradValuesSpinUp) !=
        densityData.end())
      {
        fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::gradValuesSpinUp],
          d_gradDensityValsSpinUpAllQuads,
          indexRangeGrad);
      }

    if (densityData.find(DensityDescriptorDataAttributes::gradValuesSpinDown) !=
        densityData.end())
      {
        fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::gradValuesSpinDown],
          d_gradDensityValsSpinDownAllQuads,
          indexRangeGrad);
      }
  }



  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixRadial<memorySpace>::evalOverlapMatrixStart(
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &quadpts,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &quadWt)
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixRadial<memorySpace>::evalOverlapMatrixEnd(
    const MPI_Comm &mpiComm)
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixRadial<memorySpace>::projectDensityMatrixStart(
    const std::unordered_map<std::string, std::vector<dataTypes::number>>
      &projectionInputsDataType,
    const std::unordered_map<
      std::string,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
                    &projectionInputsReal,
    const dftfe::Int iSpin)
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixRadial<memorySpace>::projectDensityMatrixEnd(
    const MPI_Comm &mpiComm)
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixRadial<memorySpace>::projectDensityStart(
    const std::unordered_map<
      std::string,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &projectionInputs)
  {
    d_quadPointsAll  = projectionInputs.find("quadpts")->second;
    d_quadWeightsAll = projectionInputs.find("quadWt")->second;
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &densityVals       = projectionInputs.find("densityFunc")->second;
    const dftfe::uInt nQ = d_quadWeightsAll.size();
    d_densityValsTotalAllQuads.resize(nQ, 0);
    d_densityValsSpinUpAllQuads.resize(nQ, 0);
    d_densityValsSpinDownAllQuads.resize(nQ, 0);
    for (dftfe::uInt iquad = 0; iquad < nQ; iquad++)
      d_densityValsSpinUpAllQuads[iquad] = densityVals[iquad];

    for (dftfe::uInt iquad = 0; iquad < nQ; iquad++)
      d_densityValsSpinDownAllQuads[iquad] = densityVals[nQ + iquad];

    for (dftfe::uInt iquad = 0; iquad < nQ; iquad++)
      d_densityValsTotalAllQuads[iquad] = d_densityValsSpinUpAllQuads[iquad] +
                                          d_densityValsSpinDownAllQuads[iquad];

    if (projectionInputs.find("gradDensityFunc") != projectionInputs.end())
      {
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>
          &gradDensityVals = projectionInputs.find("gradDensityFunc")->second;
        d_gradDensityValsSpinUpAllQuads.resize(nQ * 3, 0);
        d_gradDensityValsSpinDownAllQuads.resize(nQ * 3, 0);

        for (dftfe::uInt iquad = 0; iquad < nQ; iquad++)
          for (dftfe::uInt idim = 0; idim < 3; idim++)
            d_gradDensityValsSpinUpAllQuads[3 * iquad + idim] =
              gradDensityVals[3 * iquad + idim];

        for (dftfe::uInt iquad = 0; iquad < nQ; iquad++)
          for (dftfe::uInt idim = 0; idim < 3; idim++)
            d_gradDensityValsSpinDownAllQuads[3 * iquad + idim] =
              gradDensityVals[3 * nQ + 3 * iquad + idim];
      }
    if (projectionInputs.find("tauFunc") != projectionInputs.end())
      {
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>
          &tauVals = projectionInputs.find("tauFunc")->second;
        d_tauValsTotalAllQuads.resize(nQ, 0);
        d_tauValsSpinUpAllQuads.resize(nQ, 0);
        d_tauValsSpinDownAllQuads.resize(nQ, 0);
        for (dftfe::uInt iquad = 0; iquad < nQ; iquad++)
          {
            d_tauValsSpinUpAllQuads[iquad]   = tauVals[iquad];
            d_tauValsSpinDownAllQuads[iquad] = tauVals[nQ + iquad];
            d_tauValsTotalAllQuads[iquad] =
              tauVals[iquad] + tauVals[nQ + iquad];
          }
      }
  }



  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixRadial<memorySpace>::projectDensityEnd(
    const MPI_Comm &mpiComm)
  {}

  template class AuxDensityMatrixRadial<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class AuxDensityMatrixRadial<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
