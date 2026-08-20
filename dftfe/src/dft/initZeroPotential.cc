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
// @author Phani Motamarri
//

//
// Initialize rho by reading in single-atom electron-density and fit a spline
//
#include <dftParameters.h>
#include <dft.h>
#include <fileReaders.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::initZeroPotential()
  {
    // clear existing data
    d_zeroPotential.clear();
    d_zeroPotentialAtoms.clear();
    d_gradZeroPotentialAtoms.clear();

    // Reading single atom rho initial guess
    pcout << std::endl
          << "Computing ZeroPotential at QuadPoints....." << std::endl;

    std::map<dftfe::uInt, double> outerMostPointZeroPotential;
    const double                  truncationTol = 1e-12;
    dftfe::uInt                   fileReadFlag  = 0;

    double maxZeroPotentialTail = 0.0;
    // loop over atom types
    for (std::set<dftfe::uInt>::iterator it = atomTypes.begin();
         it != atomTypes.end();
         it++)
      {
        outerMostPointZeroPotential[*it] =
          d_pawClassPtr->getRmaxZeroPotential(*it);
        if (outerMostPointZeroPotential[*it] > maxZeroPotentialTail)
          maxZeroPotentialTail = outerMostPointZeroPotential[*it];
        if (d_dftParamsPtr->verbosity >= 4)
          pcout << " Atomic number: " << *it
                << " Outermost Point ZeroPotential: "
                << outerMostPointZeroPotential[*it] << std::endl;
      }

    const double cellCenterCutOff = maxZeroPotentialTail + 5.0;
    //
    // Initialize rho
    //
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_lpspQuadratureId);
    dealii::FEValues<3> fe_values(FE,
                                  quadrature_formula,
                                  dealii::update_quadrature_points);
    const dftfe::uInt   n_q_points = quadrature_formula.size();
    //
    // get number of global charges
    //
    const dftfe::Int numberGlobalCharges = atomLocations.size();

    //
    // get number of image charges used only for periodic
    //
    const dftfe::Int numberImageCharges = d_imageIdsPAW.size();

    //
    // loop over elements
    //
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    dealii::Tensor<1, 3, double> zeroTensor1;
    for (dftfe::uInt i = 0; i < 3; i++)
      zeroTensor1[i] = 0.0;

    // loop over elements
    //
    cell = dofHandler.begin_active();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            std::vector<double> &zeroPotentialQuadValues =
              d_zeroPotential[cell->id()];
            std::vector<double> zeroPotentialAtom(n_q_points, 0.0);
            std::vector<dealii::Tensor<1, 3, double>> gradzeroPotentialAtom(
              n_q_points, zeroTensor1);
            zeroPotentialQuadValues.resize(n_q_points, 0.0);



            // loop over atoms
            for (dftfe::uInt iAtom = 0; iAtom < atomLocations.size(); ++iAtom)
              {
                dealii::Point<3> atom(atomLocations[iAtom][2],
                                      atomLocations[iAtom][3],
                                      atomLocations[iAtom][4]);
                bool             isZeroPotentialDataInCell = false;


                if (atom.distance(cell->center()) > cellCenterCutOff)
                  continue;

                // loop over quad points
                for (dftfe::uInt q = 0; q < n_q_points; ++q)
                  {
                    dealii::Point<3> quadPoint = fe_values.quadrature_point(q);
                    dealii::Tensor<1, 3, double> diff = quadPoint - atom;
                    double distanceToAtom = quadPoint.distance(atom);
                    double value          = 0.0;
                    double radialFirstDerivative;
                    if (distanceToAtom <=
                        outerMostPointZeroPotential[atomLocations[iAtom][0]])
                      {
                        std::vector<double> Vec;
                        d_pawClassPtr->getRadialZeroPotential(
                          atomLocations[iAtom][0], distanceToAtom, Vec);

                        value                 = Vec[0];
                        radialFirstDerivative = Vec[1];

                        isZeroPotentialDataInCell = true;
                      }
                    else
                      {
                        value                 = 0.0;
                        radialFirstDerivative = 0.0;
                      }
                    zeroPotentialQuadValues[q] += value;
                    zeroPotentialAtom[q] = value;
                    if (distanceToAtom > 1e-14)
                      gradzeroPotentialAtom[q] =
                        radialFirstDerivative * diff / distanceToAtom;
                    else
                      gradzeroPotentialAtom[q] = zeroTensor1;



                  } // end loop over quad points
                if (isZeroPotentialDataInCell)
                  {
                    std::vector<double> &ZeroPotentialAtomCell =
                      d_zeroPotentialAtoms[iAtom][cell->id()];
                    std::vector<double> &gradZeroPotentialAtomCell =
                      d_gradZeroPotentialAtoms[iAtom][cell->id()];
                    gradZeroPotentialAtomCell.resize(3 * n_q_points, 0.0);
                    ZeroPotentialAtomCell.resize(n_q_points, 0.0);

                    for (dftfe::uInt q = 0; q < n_q_points; ++q)
                      {
                        ZeroPotentialAtomCell[q] = zeroPotentialAtom[q];
                        gradZeroPotentialAtomCell[3 * q + 0] =
                          gradzeroPotentialAtom[q][0];
                        gradZeroPotentialAtomCell[3 * q + 1] =
                          gradzeroPotentialAtom[q][1];
                        gradZeroPotentialAtomCell[3 * q + 2] =
                          gradzeroPotentialAtom[q][2];
                      } // q_point loop
                  }     // if loop
              }         // loop over atoms

            // loop over image charges
            for (dftfe::uInt iImageCharge = 0;
                 iImageCharge < numberImageCharges;
                 ++iImageCharge)
              {
                const dftfe::Int masterAtomId = d_imageIdsPAW[iImageCharge];
                dealii::Point<3> imageAtom(
                  d_imagePositionsPAW[iImageCharge][0],
                  d_imagePositionsPAW[iImageCharge][1],
                  d_imagePositionsPAW[iImageCharge][2]);

                if (imageAtom.distance(cell->center()) > cellCenterCutOff)
                  continue;

                bool isZeroPotentialDataInCell = false;

                // loop over quad points
                for (dftfe::uInt q = 0; q < n_q_points; ++q)
                  {
                    dealii::Point<3> quadPoint = fe_values.quadrature_point(q);
                    dealii::Tensor<1, 3, double> diff = quadPoint - imageAtom;
                    double distanceToAtom = quadPoint.distance(imageAtom);



                    double value                 = 0.0;
                    double radialFirstDerivative = 0.0;
                    if (distanceToAtom <=
                        outerMostPointZeroPotential[atomLocations[masterAtomId]
                                                                 [0]])
                      {
                        std::vector<double> Vec;
                        d_pawClassPtr->getRadialZeroPotential(
                          atomLocations[masterAtomId][0], distanceToAtom, Vec);
                        value                 = Vec[0];
                        radialFirstDerivative = Vec[1];
                        // pcout<<distanceToAtom<<" "<<value<<std::endl;
                        isZeroPotentialDataInCell = true;
                      }

                    zeroPotentialQuadValues[q] += value;
                    zeroPotentialAtom[q] = value;
                    if (distanceToAtom > 1e-14)
                      gradzeroPotentialAtom[q] =
                        radialFirstDerivative * diff / distanceToAtom;
                    else
                      gradzeroPotentialAtom[q] = zeroTensor1;

                  } // quad point loop

                if (isZeroPotentialDataInCell)
                  {
                    std::vector<double> &ZeroPotentialAtomCell =
                      d_zeroPotentialAtoms[numberGlobalCharges + iImageCharge]
                                          [cell->id()];
                    ZeroPotentialAtomCell.resize(n_q_points);
                    std::vector<double> &gradZeroPotentialAtomCell =
                      d_gradZeroPotentialAtoms[numberGlobalCharges +
                                               iImageCharge][cell->id()];
                    gradZeroPotentialAtomCell.resize(3 * n_q_points, 0.0);
                    ZeroPotentialAtomCell.resize(n_q_points, 0.0);
                    for (dftfe::uInt q = 0; q < n_q_points; ++q)
                      {
                        ZeroPotentialAtomCell[q] = zeroPotentialAtom[q];
                        gradZeroPotentialAtomCell[3 * q + 0] =
                          gradzeroPotentialAtom[q][0];
                        gradZeroPotentialAtomCell[3 * q + 1] =
                          gradzeroPotentialAtom[q][1];
                        gradZeroPotentialAtomCell[3 * q + 2] =
                          gradzeroPotentialAtom[q][2];
                      } // q_point loop
                  }     // if loop

              } // end of image charges

          } // cell locally owned check

      } // cell loop
  }
#include "dft.inst.cc"
} // namespace dftfe
