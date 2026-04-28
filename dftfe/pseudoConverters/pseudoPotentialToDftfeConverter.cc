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
// @author Kartikey Srivastava, Kartick Ramakrishnan
//
#include <iostream>
#include <vector>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <stdexcept>
#include <cmath>


namespace dftfe
{
  namespace pseudoUtils
  {
    dftfe::Int
    XmlTagOccurance(std::vector<std::string> tag_name, std::string file_path_in)
    {
      xmlDocPtr  doc;
      xmlNodePtr cur;
      doc = xmlParseFile(file_path_in.c_str());
      cur = xmlDocGetRootElement(doc);
      // Finding the tag
      dftfe::Int occurance = 0;
      for (dftfe::Int i = 0; i < tag_name.size(); i++)
        {
          cur                 = cur->children;
          const xmlChar *temp = (const xmlChar *)tag_name[i].c_str();
          while (cur != NULL)
            {
              if ((!xmlStrcmp(cur->name, temp)))
                {
                  occurance = occurance + 1;
                }
              cur = cur->next;
            }
        }
      return occurance;
    }

    std::vector<double>
    XmlTagReaderMain(std::vector<std::string> tag_name,
                     std::string              file_path_in)
    {
      xmlDocPtr  doc;
      xmlNodePtr cur;
      doc = xmlParseFile(file_path_in.c_str());
      cur = xmlDocGetRootElement(doc);
      // Finding the tag

      for (dftfe::Int i = 0; i < tag_name.size(); i++)
        {
          cur                 = cur->children;
          const xmlChar *temp = (const xmlChar *)tag_name[i].c_str();
          while (cur != NULL)
            {
              if ((!xmlStrcmp(cur->name, temp)))
                {
                  break;
                }
              cur = cur->next;
            }
        }
      // If tag not found
      if (cur == NULL)
        {
          std::cout << "Tag not found";
          return std::vector<double>();
        }
      else
        {
          // Extracting main data
          xmlChar *key;
          key = xmlNodeListGetString(doc, cur->xmlChildrenNode, 1);
          std::string         main_str = (char *)key;
          std::vector<double> main;
          std::stringstream   ss;
          ss << main_str;
          double temp_str;
          while (!ss.eof())
            {
              ss >> temp_str;
              main.push_back(temp_str);
            }
          main.pop_back();
          return main;
        }
    }

    std::vector<double>
    XmlTagReaderMainMulti(std::vector<std::string> tag_name,
                          std::string              file_path_in,
                          dftfe::Int               occurance)
    {
      xmlDocPtr  doc;
      xmlNodePtr cur;
      doc = xmlParseFile(file_path_in.c_str());
      cur = xmlDocGetRootElement(doc);
      // Finding the tag
      dftfe::Int flag = 0;
      for (dftfe::Int i = 0; i < tag_name.size(); i++)
        {
          cur                 = cur->children;
          const xmlChar *temp = (const xmlChar *)tag_name[i].c_str();
          while (cur != NULL)
            {
              if ((!xmlStrcmp(cur->name, temp)))
                {
                  flag = flag + 1;
                  if (flag == occurance)
                    break;
                }
              cur = cur->next;
            }
        }
      // If tag not found
      if (cur == NULL)
        {
          std::cout << "Tag not found";
          return std::vector<double>();
        }
      else
        {
          // Extracting main data
          xmlChar *key;
          key = xmlNodeListGetString(doc, cur->xmlChildrenNode, 1);
          std::string         main_str = (char *)key;
          std::vector<double> main;
          std::stringstream   ss;
          ss << main_str;
          double temp_str;
          while (!ss.eof())
            {
              ss >> temp_str;
              main.push_back(temp_str);
            }
          main.pop_back();
          return main;
        }
    }
    void
    XmlTagReaderAttr(std::vector<std::string>  tag_name,
                     std::string               file_path_in,
                     std::vector<std::string> *attr_type,
                     std::vector<std::string> *attr_value)
    {
      xmlDocPtr  doc;
      xmlNodePtr cur;
      doc = xmlParseFile(file_path_in.c_str());
      cur = xmlDocGetRootElement(doc);

      // Finding the tag
      for (dftfe::Int i = 0; i < tag_name.size(); i++)
        {
          cur                 = cur->children;
          const xmlChar *temp = (const xmlChar *)tag_name[i].c_str();
          while (cur != NULL)
            {
              if ((!xmlStrcmp(cur->name, temp)))
                {
                  break;
                }
              cur = cur->next;
            }
        }

      // If tag not found
      if (cur == NULL)
        {
          std::cout << "Tag not found";
          return;
        }
      else
        {
          // Extracting Attribute data
          xmlAttr *attribute = cur->properties;
          if (attribute == NULL)
            {
              std::cout << "Tag does not have attributes";
              return;
            }
          else
            {
              for (xmlAttrPtr attr = cur->properties; NULL != attr;
                   attr            = attr->next)
                {
                  (*attr_type).push_back((char *)(attr->name));
                  xmlChar *value = xmlNodeListGetString(doc, attr->children, 1);
                  (*attr_value).push_back((char *)value);
                }
            }
        }
    }

    void
    XmlTagReaderAttrMulti(std::vector<std::string>  tag_name,
                          std::string               file_path_in,
                          std::vector<std::string> *attr_type,
                          std::vector<std::string> *attr_value,
                          dftfe::Int                occurance)
    {
      xmlDocPtr  doc;
      xmlNodePtr cur;
      doc             = xmlParseFile(file_path_in.c_str());
      cur             = xmlDocGetRootElement(doc);
      dftfe::Int flag = 0;
      // Finding the tag
      for (dftfe::Int i = 0; i < tag_name.size(); i++)
        {
          cur                 = cur->children;
          const xmlChar *temp = (const xmlChar *)tag_name[i].c_str();
          while (cur != NULL)
            {
              if ((!xmlStrcmp(cur->name, temp)))
                {
                  flag = flag + 1;
                  if (flag == occurance)
                    break;
                }
              cur = cur->next;
            }
        }

      // Extracting Attribute data
      xmlAttr *attribute = cur->properties;
      if (attribute == NULL)
        {
          std::cout << "Tag does not have attributes";
          return;
        }
      else
        {
          for (xmlAttrPtr attr = cur->properties; NULL != attr;
               attr            = attr->next)
            {
              (*attr_type).push_back((char *)(attr->name));
              xmlChar *value = xmlNodeListGetString(doc, attr->children, 1);
              (*attr_value).push_back((char *)value);
            }
        }
    }

    void
    XmlTagReaderMultiAttr(std::vector<std::string>  tag_name,
                          std::string               file_path_in,
                          std::vector<std::string> *attr_type,
                          std::vector<std::string> *attr_value,
                          dftfe::Int                occurance)
    {
      xmlDocPtr  doc;
      xmlNodePtr cur;
      doc             = xmlParseFile(file_path_in.c_str());
      cur             = xmlDocGetRootElement(doc);
      dftfe::Int flag = 0;
      // Finding the tag
      for (dftfe::Int i = 0; i < tag_name.size(); i++)
        {
          cur                 = cur->children;
          const xmlChar *temp = (const xmlChar *)tag_name[i].c_str();
          while (cur != NULL)
            {
              if ((!xmlStrcmp(cur->name, temp)) && i == (tag_name.size() - 1))
                {
                  flag = flag + 1;
                  if (flag == occurance)
                    break;
                }

              if ((!xmlStrcmp(cur->name, temp)) && i != (tag_name.size() - 1))
                {
                  break;
                }

              cur = cur->next;
            }
        }

      // Extracting Attribute data
      xmlAttr *attribute = cur->properties;
      if (attribute == NULL)
        {
          std::cout << "Tag does not have attributes";
          return;
        }
      else
        {
          for (xmlAttrPtr attr = cur->properties; NULL != attr;
               attr            = attr->next)
            {
              (*attr_type).push_back((char *)(attr->name));
              xmlChar *value = xmlNodeListGetString(doc, attr->children, 1);
              (*attr_value).push_back((char *)value);
            }
        }
    }
    dftfe::Int
    xmlNodeChildCount(std::vector<std::string> tag_name,
                      std::string              file_path_in)
    {
      xmlDocPtr  doc;
      xmlNodePtr cur;
      doc = xmlParseFile(file_path_in.c_str());
      cur = xmlDocGetRootElement(doc);

      // Finding the tag
      for (dftfe::Int i = 0; i < tag_name.size(); i++)
        {
          cur                 = cur->children;
          const xmlChar *temp = (const xmlChar *)tag_name[i].c_str();
          while (cur != NULL)
            {
              if ((!xmlStrcmp(cur->name, temp)))
                {
                  break;
                }
              cur = cur->next;
            }
        }
      // Counting children of current node
      dftfe::Int child_count = xmlChildElementCount(cur);
      return child_count;
    }


    //                                                                DFT-FE
    //                                                                Formatting
    //                                                                (UPF file)
    void
    xmltoSummaryFile(std::string file_path_in, std::string file_path_out)
    {
      // List of momentum values
      bool socFlag = false;
      {
        std::vector<std::string> header_tag;
        std::vector<std::string> attr_type;
        std::vector<std::string> attr_value;
        header_tag.push_back("PP_HEADER");
        XmlTagReaderAttr(header_tag, file_path_in, &attr_type, &attr_value);
        std::string to_search = "has_so";
        dftfe::uInt index     = 0;
        auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
        if (it == attr_type.end())
          {
            throw std::invalid_argument(
              "spin orbit coupling attribute not found");
          }
        else
          {
            index = std::distance(attr_type.begin(), it);
          }
        socFlag = attr_value[index] == "T";
      }
      std::vector<std::string> tag_name_parent;
      tag_name_parent.push_back("PP_NONLOCAL");
      std::vector<dftfe::Int> ang_mom_list;
      for (dftfe::Int i = 1;
           i < xmlNodeChildCount(tag_name_parent, file_path_in);
           i++)
        {
          std::string pp_beta_str = "PP_BETA.";
          pp_beta_str += std::to_string(i);
          std::vector<std::string> tag_name;
          tag_name.push_back("PP_NONLOCAL");
          tag_name.push_back(pp_beta_str);
          std::vector<std::string> attr_type;
          std::vector<std::string> attr_value;
          XmlTagReaderAttr(tag_name, file_path_in, &attr_type, &attr_value);
          dftfe::uInt index     = 0;
          std::string to_search = "angular_momentum";
          auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it == attr_type.end())
            {
              throw std::invalid_argument(
                "angular momentum attribute not found");
            }
          else
            {
              index = std::distance(attr_type.begin(), it);
              ang_mom_list.push_back(std::stoi(attr_value[index]));
            }
        }
      // Unique angular momentum values
      std::vector<dftfe::Int> ang_mom_unique_list;
      auto                    is_unique =
        std::adjacent_find(ang_mom_list.begin(), ang_mom_list.end()) ==
        ang_mom_list.end();
      if (!is_unique)
        {
          ang_mom_unique_list = ang_mom_list;
          std::sort(ang_mom_unique_list.begin(), ang_mom_unique_list.end());
          auto it =
            ang_mom_unique_list.erase(std::unique(ang_mom_unique_list.begin(),
                                                  ang_mom_unique_list.end()));
          ang_mom_unique_list.resize(distance(ang_mom_unique_list.begin(), it));
        }
      else
        {
          ang_mom_unique_list = ang_mom_list;
        }

      std::vector<double> tot_ang_mom_list;
      if (socFlag)
        {
          std::vector<std::string> tag_name_parent;
          tag_name_parent.push_back("PP_SPIN_ORB");
          for (int i = 1; i < ang_mom_list.size() + 1; i++)
            {
              std::string pp_beta_str = "PP_RELBETA.";
              pp_beta_str += std::to_string(i);
              std::vector<std::string> tag_name;
              tag_name.push_back("PP_SPIN_ORB");
              tag_name.push_back(pp_beta_str);
              std::vector<std::string> attr_type;
              std::vector<std::string> attr_value;
              XmlTagReaderAttr(tag_name, file_path_in, &attr_type, &attr_value);
              dftfe::uInt index     = 0;
              std::string to_search = "jjj";
              auto        it =
                std::find(attr_type.begin(), attr_type.end(), to_search);
              if (it == attr_type.end())
                {
                  throw std::invalid_argument(
                    "total angular momentum attribute not found");
                }
              else
                {
                  index = std::distance(attr_type.begin(), it);
                  tot_ang_mom_list.push_back(std::stod(attr_value[index]));
                }
            }
        }
      // Multiplicity of unique angular momentum values
      std::vector<dftfe::Int> ang_mom_multiplicity_list;
      for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          dftfe::Int count = 0;
          for (dftfe::Int j = 0; j < ang_mom_list.size(); j++)
            {
              if (ang_mom_list[j] == ang_mom_unique_list[i])
                {
                  count++;
                }
            }
          ang_mom_multiplicity_list.push_back(count);
        }
      dftfe::Int                       row_index = 0;
      dftfe::Int                       index     = 0;
      std::vector<std::vector<double>> out_proj_arr;
      for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          dftfe::Int l = ang_mom_unique_list[i];
          for (dftfe::Int j = 0; j < ang_mom_multiplicity_list[i]; j++)
            {
              dftfe::Int m = -l;
              for (dftfe::Int k = 0; k < 2 * l + 1; k++)
                {
                  out_proj_arr.push_back((std::vector<double>()));
                  out_proj_arr[row_index].push_back(index);
                  out_proj_arr[row_index].push_back(l);
                  if (socFlag)
                    out_proj_arr[row_index].push_back(tot_ang_mom_list[index]);
                  out_proj_arr[row_index].push_back(m);
                  m++;
                  row_index++;
                }
              index++;
            }
        }
      // Writing the supplementary

      std::fstream file;
      file_path_out.append("/PseudoAtomDat");
      file.open(file_path_out, std::ios::out);
      if (file.is_open())
        {
          // Total projector
          file << out_proj_arr.size() << std::endl;
          file << socFlag << std::endl;
          // Projector data
          dftfe::Int m = out_proj_arr.size();
          dftfe::Int n = out_proj_arr[0].size();
          for (dftfe::Int i = 0; i < m; i++)
            {
              for (dftfe::Int j = 0; j < n; j++)
                file << out_proj_arr[i][j] << " ";
              file << std::endl;
            }

          for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
            {
              std::string proj_str = "proj_l";
              proj_str += std::to_string(ang_mom_unique_list[i]);
              file << proj_str << ".dat" << std::endl;
              file << ang_mom_multiplicity_list[i] << std::endl;
            }
          // Name for D_ij file
          file << "denom.dat" << std::endl;

          // Orbitals
          std::vector<std::string> pswfc_tag;
          pswfc_tag.push_back("PP_PSWFC");
          for (dftfe::Int i = 1;
               i <= xmlNodeChildCount(pswfc_tag, file_path_in);
               i++)
            {
              // Reading chi data
              std::string pp_chi_str = "PP_CHI.";
              pp_chi_str += std::to_string(i);
              std::vector<std::string> chi_tag;
              chi_tag.push_back("PP_PSWFC");
              chi_tag.push_back(pp_chi_str);
              std::vector<std::string> attr_type;
              std::vector<std::string> attr_value;
              XmlTagReaderAttr(chi_tag, file_path_in, &attr_type, &attr_value);
              dftfe::uInt index     = 0;
              std::string to_search = "label";
              auto        it =
                std::find(attr_type.begin(), attr_type.end(), to_search);
              if (it == attr_type.end())
                {
                  throw std::invalid_argument(
                    "orbital label attribute not found");
                }
              else
                {
                  index = std::distance(attr_type.begin(), it);
                }
              std::string orbital_string = attr_value[index];
              for (auto &w : orbital_string)
                {
                  w = tolower(w);
                  if (w == 's')
                    w = '0';
                  else if (w == 'p')
                    w = '1';
                  else if (w == 'd')
                    w = '2';
                  else if (w == 'f')
                    w = '3';
                }
              file << "psi" + orbital_string + ".inp" << std::endl;
            }
        }
      file.close();
    }

    void
    xmltoProjectorFile(std::string file_path_in, std::string file_path_out)
    {
      // List of momentum values
      std::vector<std::string> tag_name_parent;
      tag_name_parent.push_back("PP_NONLOCAL");
      std::vector<dftfe::Int> ang_mom_list;
      for (dftfe::Int i = 1;
           i < xmlNodeChildCount(tag_name_parent, file_path_in);
           i++)
        {
          std::string pp_beta_str = "PP_BETA.";
          pp_beta_str += std::to_string(i);
          std::vector<std::string> tag_name;
          tag_name.push_back("PP_NONLOCAL");
          tag_name.push_back(pp_beta_str);
          std::vector<std::string> attr_type;
          std::vector<std::string> attr_value;
          XmlTagReaderAttr(tag_name, file_path_in, &attr_type, &attr_value);
          dftfe::uInt index     = 0;
          std::string to_search = "angular_momentum";
          auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it == attr_type.end())
            {
              throw std::invalid_argument(
                "angular momentum attribute not found");
            }
          else
            {
              index = std::distance(attr_type.begin(), it);
              ang_mom_list.push_back(std::stoi(attr_value[index]));
            }
        }

      // Unique angular momentum values
      std::vector<dftfe::Int> ang_mom_unique_list;
      auto                    is_unique =
        std::adjacent_find(ang_mom_list.begin(), ang_mom_list.end()) ==
        ang_mom_list.end();
      if (!is_unique)
        {
          ang_mom_unique_list = ang_mom_list;
          std::sort(ang_mom_unique_list.begin(), ang_mom_unique_list.end());
          auto it =
            ang_mom_unique_list.erase(std::unique(ang_mom_unique_list.begin(),
                                                  ang_mom_unique_list.end()));
          ang_mom_unique_list.resize(distance(ang_mom_unique_list.begin(), it));
        }
      else
        {
          ang_mom_unique_list = ang_mom_list;
        }

      // Multiplicity of unique angular momentum values
      std::vector<dftfe::Int> ang_mom_multiplicity_list;
      for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          dftfe::Int count = 0;
          for (dftfe::Int j = 0; j < ang_mom_list.size(); j++)
            {
              if (ang_mom_list[j] == ang_mom_unique_list[i])
                {
                  count++;
                }
            }
        }
      // Beta index for same angular momentum
      std::vector<std::vector<dftfe::Int>> beta_index;
      for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          beta_index.push_back((std::vector<dftfe::Int>()));
          for (dftfe::Int j = 0; j < ang_mom_list.size(); j++)
            {
              if (ang_mom_list[j] == ang_mom_unique_list[i])
                {
                  beta_index[i].push_back(j + 1);
                }
            }
        }

      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("PP_MESH");
      radial_tag.push_back("PP_R");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Extracting projector data according to angular momentum
      for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          std::vector<std::vector<double>> beta_values;
          std::string                      proj_str = "/proj_l";
          proj_str += std::to_string(ang_mom_unique_list[i]);
          proj_str += ".dat";
          for (dftfe::Int j = 0; j < beta_index[i].size(); j++)
            {
              std::string pp_beta_str = "PP_BETA.";
              pp_beta_str += std::to_string(beta_index[i][j]);
              std::vector<std::string> beta_tag;
              beta_tag.push_back("PP_NONLOCAL");
              beta_tag.push_back(pp_beta_str);
              beta_values.push_back(std::vector<double>());
              beta_values[j] = XmlTagReaderMain(beta_tag, file_path_in);
              std::vector<double> trial =
                XmlTagReaderMain(beta_tag, file_path_in);
            }

          std::fstream file;
          file.open(file_path_out + proj_str, std::ios::out);
          file << std::setprecision(15);
          if (file.is_open())
            {
              for (dftfe::Int l = 0; l < radial_coord.size(); l++)
                {
                  if (l == 0)
                    {
                      file << radial_coord[0] << " ";
                      for (dftfe::Int m = 0; m < beta_values.size(); m++)
                        {
                          if (m != (beta_values.size() - 1))
                            file << beta_values[m][1] / radial_coord[1] << " ";
                          else
                            file << beta_values[m][1] / radial_coord[1]
                                 << std::endl;
                        }
                    }
                  else
                    {
                      file << radial_coord[l] << " ";
                      for (dftfe::Int m = 0; m < beta_values.size(); m++)
                        {
                          if (m != (beta_values.size() - 1))
                            file << beta_values[m][l] / radial_coord[l] << " ";
                          else
                            file << beta_values[m][l] / radial_coord[l]
                                 << std::endl;
                        }
                    }
                }
            }
          file.close();
        }
    }

    void
    xmltoLocalPotential(std::string file_path_in, std::string file_path_out)
    {
      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("PP_MESH");
      radial_tag.push_back("PP_R");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Extracting local potential data
      std::vector<double>      local_pot_values;
      std::vector<std::string> local_por_tag;
      local_por_tag.push_back("PP_LOCAL");
      local_pot_values = XmlTagReaderMain(local_por_tag, file_path_in);

      // Writing the local potential data
      std::fstream file;
      file.open(file_path_out + "/locPot.dat", std::ios::out);
      file << std::setprecision(12);
      if (file.is_open())
        {
          for (dftfe::Int l = 0; l < radial_coord.size(); l++)
            {
              file << radial_coord[l] << " " << local_pot_values[l] / 2
                   << std::endl;
            }
        }
      file.close();
    }

    void
    xmltoDenomFile(std::string file_path_in, std::string file_path_out)
    {
      // Extracting Diagonal Matrix
      std::vector<double>      diagonal_mat;
      std::vector<std::string> diagonal_tag;
      diagonal_tag.push_back("PP_NONLOCAL");
      diagonal_tag.push_back("PP_DIJ");
      diagonal_mat = XmlTagReaderMain(diagonal_tag, file_path_in);

      std::vector<std::string> tag_name_parent;
      tag_name_parent.push_back("PP_NONLOCAL");
      dftfe::Int n = xmlNodeChildCount(tag_name_parent, file_path_in) - 1;

      // Writing the denom.dat
      std::fstream file;
      file.open(file_path_out + "/denom.dat", std::ios::out);
      file << std::setprecision(12);
      if (file.is_open())
        {
          for (dftfe::Int l = 0; l < diagonal_mat.size(); l++)
            {
              if (l != 0 & (l % n == 0))
                file << std::endl;
              file << diagonal_mat[l] / 2 << " ";
            }
        }
      file.close();
    }
    void
    xmltoCoreDensityFile(std::string file_path_in, std::string file_path_out)
    {
      std::vector<std::string> header_tag;
      std::vector<std::string> attr_type;
      std::vector<std::string> attr_value;
      header_tag.push_back("PP_HEADER");
      XmlTagReaderAttr(header_tag, file_path_in, &attr_type, &attr_value);
      dftfe::uInt index     = 0;
      std::string to_search = "core_correction";
      auto        it = std::find(attr_type.begin(), attr_type.end(), to_search);
      if (it == attr_type.end())
        {
          throw std::invalid_argument("core correction attribute not found");
        }
      else
        {
          index = std::distance(attr_type.begin(), it);
        }

      if (attr_value[index] == "T")
        {
          // Extracting radial coordinates
          std::vector<double>      radial_coord;
          std::vector<std::string> radial_tag;
          radial_tag.push_back("PP_MESH");
          radial_tag.push_back("PP_R");
          radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

          // Extracting non local core correction
          std::vector<double>      nlcc_values;
          std::vector<std::string> nlcc_tag;
          nlcc_tag.push_back("PP_NLCC");
          nlcc_values = XmlTagReaderMain(nlcc_tag, file_path_in);

          // Writing coreDensity.inp
          std::fstream file;
          file.open(file_path_out + "/coreDensity.inp", std::ios::out);
          file << std::setprecision(12);
          if (file.is_open())
            {
              for (dftfe::Int l = 0; l < radial_coord.size(); l++)
                {
                  file << radial_coord[l] << " " << nlcc_values[l] << std::endl;
                }
            }
          file.close();
        }
      else
        throw std::invalid_argument("core_correction set false");
    }

    void
    xmltoDensityFile(std::string file_path_in, std::string file_path_out)
    {
      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("PP_MESH");
      radial_tag.push_back("PP_R");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Extracting valence density
      std::vector<double>      rhoatom_values;
      std::vector<std::string> rhoatom_tag;
      rhoatom_tag.push_back("PP_RHOATOM");
      rhoatom_values = XmlTagReaderMain(rhoatom_tag, file_path_in);

      // Writing density.inp
      double       pi = 2 * acos(0.0);
      std::fstream file;
      file.open(file_path_out + "/density.inp", std::ios::out);
      file << std::setprecision(15);
      if (file.is_open())
        {
          for (dftfe::Int l = 0; l < radial_coord.size(); l++)
            {
              if (l == 0)
                file << radial_coord[0] << " " << rhoatom_values[0]
                     << std::endl;
              else
                file << radial_coord[l] << " "
                     << rhoatom_values[l] /
                          (4 * pi * std::pow(radial_coord[l], 2))
                     << std::endl;
            }
        }
      file.close();
    }
    void
    xmltoOrbitalFile(std::string file_path_in, std::string file_path_out)
    { // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("PP_MESH");
      radial_tag.push_back("PP_R");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);
      std::vector<std::string> pswfc_tag;
      pswfc_tag.push_back("PP_PSWFC");
      for (dftfe::Int i = 1; i <= xmlNodeChildCount(pswfc_tag, file_path_in);
           i++)
        {
          // Reading chi data
          std::string pp_chi_str = "PP_CHI.";
          pp_chi_str += std::to_string(i);
          std::vector<std::string> chi_tag;
          chi_tag.push_back("PP_PSWFC");
          chi_tag.push_back(pp_chi_str);
          std::vector<double> chi_values =
            XmlTagReaderMain(chi_tag, file_path_in);
          std::vector<std::string> attr_type;
          std::vector<std::string> attr_value;
          XmlTagReaderAttr(chi_tag, file_path_in, &attr_type, &attr_value);
          dftfe::uInt index     = 0;
          std::string to_search = "label";
          auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it == attr_type.end())
            {
              throw std::invalid_argument("orbital label attribute not found");
            }
          else
            {
              index = std::distance(attr_type.begin(), it);
            }
          std::string orbital_string_nl = attr_value[index];
          for (auto &w : orbital_string_nl)
            {
              w = tolower(w);
            }
          char n = orbital_string_nl[0];
          char l;
          if (orbital_string_nl[1] == 's')
            {
              l = '0';
            }
          if (orbital_string_nl[1] == 'p')
            {
              l = '1';
            }
          if (orbital_string_nl[1] == 'd')
            {
              l = '2';
            }
          if (orbital_string_nl[1] == 'f')
            {
              l = '3';
            }
          std::string  orbital_string = "psi";
          std::fstream file;
          file.open(file_path_out + "/" + orbital_string + n + l + ".inp",
                    std::ios::out);
          file << std::setprecision(12);
          if (file.is_open())
            {
              for (dftfe::Int l = 0; l < chi_values.size(); l++)
                {
                  file << radial_coord[l] << " " << chi_values[l] << std::endl;
                }
            }
          file.close();
        }
    }


    //                                                                              PAW Extenstion


    void
    xmltoZeroPotFile(std::string file_path_in, std::string file_path_out)
    {
      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("radial_grid");
      radial_tag.push_back("values");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Extracting valence density
      std::vector<double>      zero_pot_values;
      std::vector<std::string> zero_pot_tag;
      zero_pot_tag.push_back("zero_potential");
      zero_pot_values = XmlTagReaderMain(zero_pot_tag, file_path_in);

      // Writing zero_potential.dat
      std::fstream file;
      file.open(file_path_out + "/zeroPotential.dat", std::ios::out);
      file << std::fixed
           << std::setprecision(std::numeric_limits<double>::max_digits10);
      if (file.is_open())
        {
          for (dftfe::Int l = 0; l < radial_coord.size(); l++)
            {
              file << radial_coord[l] << " " << zero_pot_values[l] << std::endl;
            }
        }
      file.close();
    }

    void
    xmltoAllElecCoreDensityFile(std::string file_path_in,
                                std::string file_path_out)
    {
      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("radial_grid");
      radial_tag.push_back("values");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Extracting valence density
      std::vector<double>      ae_density_values;
      std::vector<std::string> ae_density_tag;
      ae_density_tag.push_back("ae_core_density");
      ae_density_values = XmlTagReaderMain(ae_density_tag, file_path_in);

      // Writing all electron core density file
      std::fstream file;
      file.open(file_path_out + "/ae_core_density.dat", std::ios::out);
      file << std::setprecision(21);
      if (file.is_open())
        {
          for (dftfe::Int l = 0; l < radial_coord.size(); l++)
            {
              file << radial_coord[l] << " " << ae_density_values[l]
                   << std::endl;
            }
        }
      file.close();
    }

    void
    xmltoPseudoCoreDensityFile(std::string file_path_in,
                               std::string file_path_out)
    {
      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("radial_grid");
      radial_tag.push_back("values");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Extracting valence density
      std::vector<double>      pseudo_density_values;
      std::vector<std::string> pseudo_density_tag;
      pseudo_density_tag.push_back("pseudo_core_density");
      pseudo_density_values =
        XmlTagReaderMain(pseudo_density_tag, file_path_in);

      // Writing Pseudo core density file
      std::fstream file;
      file.open(file_path_out + "/pseudo_core_density.dat", std::ios::out);
      file << std::setprecision(18);
      if (file.is_open())
        {
          for (dftfe::Int l = 0; l < radial_coord.size(); l++)
            {
              file << radial_coord[l] << " " << pseudo_density_values[l]
                   << std::endl;
            }
        }
      file.close();
    }
    void
    xmltoKineticDifFile(std::string file_path_in, std::string file_path_out)
    {
      // Extracting Diagonal Matrix
      std::vector<double>      diagonal_mat;
      std::vector<std::string> diagonal_tag;
      diagonal_tag.push_back("kinetic_energy_differences");
      diagonal_mat = XmlTagReaderMain(diagonal_tag, file_path_in);

      // Writing the denom.dat
      std::fstream file;
      file.open(file_path_out + "/KineticEnergyij.dat", std::ios::out);
      file << std::setprecision(18);
      if (file.is_open())
        {
          for (dftfe::Int l = 0; l < diagonal_mat.size(); l++)
            {
              file << diagonal_mat[l] << std::endl;
            }
        }
      file.close();
    }

    void
    xmltoPseudoValDensityFile(std::string file_path_in,
                              std::string file_path_out)
    {
      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("radial_grid");
      radial_tag.push_back("values");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Extracting valence density
      std::vector<double>      pseudo_val_density_values;
      std::vector<std::string> pseudo_val_density_tag;
      pseudo_val_density_tag.push_back("pseudo_valence_density");
      pseudo_val_density_values =
        XmlTagReaderMain(pseudo_val_density_tag, file_path_in);

      // Writing Pseudo Valence Density File
      std::fstream file;
      file.open(file_path_out + "/pseudo_valence_density.dat", std::ios::out);
      file << std::fixed
           << std::setprecision(std::numeric_limits<double>::max_digits10);
      if (file.is_open())
        {
          for (dftfe::Int l = 0; l < radial_coord.size(); l++)
            {
              file << radial_coord[l] << " " << pseudo_val_density_values[l]
                   << std::endl;
            }
        }
      file.close();
    }

    void
    xmltoProjectorPAWFile(
      std::string                               file_path_in,
      std::string                               file_path_out,
      const std::map<std::string, std::string> &mapStateIndexToQuantumNumber)
    {
      // List of momentum values
      std::vector<std::string> tag_name;
      tag_name.push_back("projector_function");
      dftfe::Int no_projector_func = XmlTagOccurance(tag_name, file_path_in);
      std::vector<dftfe::Int>  ang_mom_list;
      std::vector<std::string> ang_mom_list_str;
      for (dftfe::Int i = 0; i < no_projector_func; i++)
        {
          std::vector<std::string> header_tag;
          std::vector<std::string> attr_type;
          std::vector<std::string> attr_value;
          header_tag.push_back("projector_function");
          XmlTagReaderAttrMulti(
            header_tag, file_path_in, &attr_type, &attr_value, i + 1);
          dftfe::uInt index     = 0;
          std::string to_search = "state";
          auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it == attr_type.end())
            {
              std::cout << "state attribute not found";
              return;
            }
          else
            {
              index = std::distance(attr_type.begin(), it);
              ang_mom_list_str.push_back(
                mapStateIndexToQuantumNumber.find(attr_value[index])->second);
            }
        }

      for (dftfe::Int j = 0; j < ang_mom_list_str.size(); j++)
        {
          for (dftfe::Int k = 0; k < ang_mom_list_str[j].size(); k++)
            {
              if (ang_mom_list_str[j][k] == 's')
                {
                  ang_mom_list.push_back(0);
                  break;
                }
              if (ang_mom_list_str[j][k] == 'p')
                {
                  ang_mom_list.push_back(1);
                  break;
                }
              if (ang_mom_list_str[j][k] == 'd')
                {
                  ang_mom_list.push_back(2);
                  break;
                }
              if (ang_mom_list_str[j][k] == 'f')
                {
                  ang_mom_list.push_back(3);
                  break;
                }
            }
        }
      // Unique angular momentum values
      std::vector<dftfe::Int> ang_mom_unique_list;
      auto                    is_unique =
        std::adjacent_find(ang_mom_list.begin(), ang_mom_list.end()) ==
        ang_mom_list.end();
      if (!is_unique)
        {
          ang_mom_unique_list = ang_mom_list;
          std::sort(ang_mom_unique_list.begin(), ang_mom_unique_list.end());
          auto it =
            ang_mom_unique_list.erase(std::unique(ang_mom_unique_list.begin(),
                                                  ang_mom_unique_list.end()));
          ang_mom_unique_list.resize(distance(ang_mom_unique_list.begin(), it));
        }
      else
        {
          ang_mom_unique_list = ang_mom_list;
        }

      // Beta index for same angular momentum
      std::vector<std::vector<dftfe::Int>> beta_index;
      for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          beta_index.push_back((std::vector<dftfe::Int>()));
          for (dftfe::Int j = 0; j < ang_mom_list.size(); j++)
            {
              if (ang_mom_list[j] == ang_mom_unique_list[i])
                {
                  beta_index[i].push_back(j + 1);
                }
            }
        }

      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("radial_grid");
      radial_tag.push_back("values");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Extracting projector data according to angular momentum
      for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          std::vector<std::vector<double>> beta_values;
          std::string                      proj_str = "/proj_l";
          proj_str += std::to_string(ang_mom_unique_list[i]);
          proj_str += ".dat";
          for (dftfe::Int j = 0; j < beta_index[i].size(); j++)
            {
              std::vector<std::string> beta_tag;
              beta_tag.push_back("projector_function");
              beta_values.push_back(std::vector<double>());
              beta_values[j] =
                XmlTagReaderMainMulti(beta_tag, file_path_in, beta_index[i][j]);
            }

          std::fstream file;
          file.open(file_path_out + proj_str, std::ios::out);
          file << std::fixed
               << std::setprecision(std::numeric_limits<double>::max_digits10);
          if (file.is_open())
            {
              for (dftfe::Int l = 0; l < radial_coord.size(); l++)
                {
                  file << radial_coord[l] << " ";
                  for (dftfe::Int m = 0; m < beta_values.size(); m++)
                    {
                      if (m != (beta_values.size() - 1))
                        file << beta_values[m][l] << " ";
                      else
                        file << beta_values[m][l] << std::endl;
                    }
                }
            }
          file.close();
        }
    }

    void
    createStateIndexToQuantumNumberMap(
      std::string                         file_path_in,
      std::string                         file_path_out,
      std::map<std::string, std::string> &mapStateToQuantumNumber)
    {
      mapStateToQuantumNumber.clear();
      file_path_out.append("/initOccupancyFile.dat");
      // Extracting radial coordinates

      // Finding the Orbitals to prdftfe::Int


      std::vector<std::string> header_tag;
      header_tag.push_back("valence_states");
      std::vector<std::string> ppw_tag;
      ppw_tag.push_back("pseudo_partial_wave");
      std::vector<std::string> valence_tag;
      valence_tag.push_back("valence_states");
      valence_tag.push_back("state");
      dftfe::Int no_states = xmlNodeChildCount(header_tag, file_path_in);
      dftfe::Int extraOrbitalIndex = 0;
      std::vector<double> occupancyVector;
      occupancyVector.clear();
      for (dftfe::Int i = 0; i < no_states; i++)
        {
          std::vector<std::string> attr_type;
          std::vector<std::string> attr_value;
          bool                     filledOrbital = true;
          XmlTagReaderMultiAttr(
            valence_tag, file_path_in, &attr_type, &attr_value, i + 1);
          std::string orbitalId;
          std::string nIndex;
          std::string lIndex;
          dftfe::Int  lQuantumNo;
          dftfe::Int  nQuandumNo;
          dftfe::uInt index_2     = 0;
          std::string to_search_2 = "id";
          auto        it_2 =
            std::find(attr_type.begin(), attr_type.end(), to_search_2);
          index_2   = std::distance(attr_type.begin(), it_2);
          orbitalId = (attr_value[index_2]);

          dftfe::uInt index_3     = 0;
          std::string to_search_3 = "n";
          auto        it_3 =
            std::find(attr_type.begin(), attr_type.end(), to_search_3);
          // Check if "n" is present
          if (it_3 != attr_type.end())
            {
              nIndex = attr_value[index_3];
            }
          else
            {
              filledOrbital = false;
              extraOrbitalIndex += 1;
            }


          dftfe::uInt index_4     = 0;
          std::string to_search_4 = "l";
          auto        it_4 =
            std::find(attr_type.begin(), attr_type.end(), to_search_4);
          index_4    = std::distance(attr_type.begin(), it_4);
          lQuantumNo = std::stoi(attr_value[index_4]);
          if (lQuantumNo == 0)
            lIndex = "s";
          else if (lQuantumNo == 1)
            lIndex = "p";
          else if (lQuantumNo == 2)
            lIndex = "d";
          else if (lQuantumNo == 3)
            lIndex = "f";
          else
            {
              // Assert Throw
            }
          if (filledOrbital)
            mapStateToQuantumNumber[orbitalId] = (nIndex + lIndex);
          else
            mapStateToQuantumNumber[orbitalId] =
              (lIndex + std::to_string(extraOrbitalIndex));
          std::string to_search_5 = "f";
          dftfe::uInt index_5     = 0;
          auto        it_5 =
            std::find(attr_type.begin(), attr_type.end(), to_search_5);
          if (it_5 != attr_type.end())
            {
              index_5                  = std::distance(attr_type.begin(), it_5);
              dftfe::Int totalOrbitals = 2 * lQuantumNo + 1;
              double     frac = std::stod(attr_value[index_5]) / totalOrbitals;
              for (dftfe::Int mIndex = -lQuantumNo; mIndex <= lQuantumNo;
                   mIndex++)
                occupancyVector.push_back(frac);
            }
          else
            {
              dftfe::Int totalOrbitals = 2 * lQuantumNo + 1;
              double     frac          = 0.0;
              for (dftfe::Int mIndex = -lQuantumNo; mIndex <= lQuantumNo;
                   mIndex++)
                occupancyVector.push_back(frac);
            }
        }
      std::fstream file;
      file.open(file_path_out, std::ios::out);
      if (file.is_open())
        {
          for (dftfe::uInt i = 0; i < occupancyVector.size(); i++)
            file << occupancyVector[i] << std::endl;
        }
      file.close();
    }



    void
    xmltoSmoothPartialPAWFile(
      std::string                               file_path_in,
      std::string                               file_path_out,
      const std::map<std::string, std::string> &mapStateIndexToQuantumNumber)
    {
      // List of momentum values
      std::vector<std::string> tag_name;
      tag_name.push_back("pseudo_partial_wave");
      dftfe::Int no_projector_func = XmlTagOccurance(tag_name, file_path_in);
      std::vector<dftfe::Int>  ang_mom_list;
      std::vector<std::string> ang_mom_list_str;
      for (dftfe::Int i = 0; i < no_projector_func; i++)
        {
          std::vector<std::string> header_tag;
          std::vector<std::string> attr_type;
          std::vector<std::string> attr_value;
          header_tag.push_back("pseudo_partial_wave");
          XmlTagReaderAttrMulti(
            header_tag, file_path_in, &attr_type, &attr_value, i + 1);
          dftfe::uInt index     = 0;
          std::string to_search = "state";
          auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it == attr_type.end())
            {
              std::cout << "state attribute not found";
              return;
            }
          else
            {
              index = std::distance(attr_type.begin(), it);
              ang_mom_list_str.push_back(
                mapStateIndexToQuantumNumber.find(attr_value[index])->second);
            }
        }

      for (dftfe::Int j = 0; j < ang_mom_list_str.size(); j++)
        {
          for (dftfe::Int k = 0; k < ang_mom_list_str[j].size(); k++)
            {
              if (ang_mom_list_str[j][k] == 's')
                {
                  ang_mom_list.push_back(0);
                  break;
                }
              if (ang_mom_list_str[j][k] == 'p')
                {
                  ang_mom_list.push_back(1);
                  break;
                }
              if (ang_mom_list_str[j][k] == 'd')
                {
                  ang_mom_list.push_back(2);
                  break;
                }
              if (ang_mom_list_str[j][k] == 'f')
                {
                  ang_mom_list.push_back(3);
                  break;
                }
            }
        }
      // Unique angular momentum values
      std::vector<dftfe::Int> ang_mom_unique_list;
      auto                    is_unique =
        std::adjacent_find(ang_mom_list.begin(), ang_mom_list.end()) ==
        ang_mom_list.end();
      if (!is_unique)
        {
          ang_mom_unique_list = ang_mom_list;
          std::sort(ang_mom_unique_list.begin(), ang_mom_unique_list.end());
          auto it =
            ang_mom_unique_list.erase(std::unique(ang_mom_unique_list.begin(),
                                                  ang_mom_unique_list.end()));
          ang_mom_unique_list.resize(distance(ang_mom_unique_list.begin(), it));
        }
      else
        {
          ang_mom_unique_list = ang_mom_list;
        }

      // Beta index for same angular momentum
      std::vector<std::vector<dftfe::Int>> beta_index;
      for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          beta_index.push_back((std::vector<dftfe::Int>()));
          for (dftfe::Int j = 0; j < ang_mom_list.size(); j++)
            {
              if (ang_mom_list[j] == ang_mom_unique_list[i])
                {
                  beta_index[i].push_back(j + 1);
                }
            }
        }

      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("radial_grid");
      radial_tag.push_back("values");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Extracting projector data according to angular momentum
      for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          std::vector<std::vector<double>> beta_values;
          std::string                      proj_str = "/smooth_partial_l";
          proj_str += std::to_string(ang_mom_unique_list[i]);
          proj_str += ".dat";
          for (dftfe::Int j = 0; j < beta_index[i].size(); j++)
            {
              std::vector<std::string> beta_tag;
              beta_tag.push_back("pseudo_partial_wave");
              beta_values.push_back(std::vector<double>());
              beta_values[j] =
                XmlTagReaderMainMulti(beta_tag, file_path_in, beta_index[i][j]);
            }

          std::fstream file;
          file.open(file_path_out + proj_str, std::ios::out);
          file << std::fixed
               << std::setprecision(std::numeric_limits<double>::max_digits10);
          if (file.is_open())
            {
              for (dftfe::Int l = 0; l < radial_coord.size(); l++)
                {
                  file << radial_coord[l] << " ";
                  for (dftfe::Int m = 0; m < beta_values.size(); m++)
                    {
                      if (m != (beta_values.size() - 1))
                        file << beta_values[m][l] << " ";
                      else
                        file << beta_values[m][l] << std::endl;
                    }
                }
            }
          file.close();
        }
    }

    void
    xmltoAllElecPartialPAWFile(
      std::string                               file_path_in,
      std::string                               file_path_out,
      const std::map<std::string, std::string> &mapStateIndexToQuantumNumber)
    {
      // List of momentum values
      std::vector<std::string> tag_name;
      tag_name.push_back("ae_partial_wave");
      dftfe::Int no_projector_func = XmlTagOccurance(tag_name, file_path_in);
      std::vector<dftfe::Int>  ang_mom_list;
      std::vector<std::string> ang_mom_list_str;
      for (dftfe::Int i = 0; i < no_projector_func; i++)
        {
          std::vector<std::string> header_tag;
          std::vector<std::string> attr_type;
          std::vector<std::string> attr_value;
          header_tag.push_back("ae_partial_wave");
          XmlTagReaderAttrMulti(
            header_tag, file_path_in, &attr_type, &attr_value, i + 1);
          dftfe::uInt index     = 0;
          std::string to_search = "state";
          auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it == attr_type.end())
            {
              std::cout << "state attribute not found";
              return;
            }
          else
            {
              index = std::distance(attr_type.begin(), it);
              ang_mom_list_str.push_back(
                mapStateIndexToQuantumNumber.find(attr_value[index])->second);
            }
        }

      for (dftfe::Int j = 0; j < ang_mom_list_str.size(); j++)
        {
          for (dftfe::Int k = 0; k < ang_mom_list_str[j].size(); k++)
            {
              if (ang_mom_list_str[j][k] == 's')
                {
                  ang_mom_list.push_back(0);
                  break;
                }
              if (ang_mom_list_str[j][k] == 'p')
                {
                  ang_mom_list.push_back(1);
                  break;
                }
              if (ang_mom_list_str[j][k] == 'd')
                {
                  ang_mom_list.push_back(2);
                  break;
                }
              if (ang_mom_list_str[j][k] == 'f')
                {
                  ang_mom_list.push_back(3);
                  break;
                }
            }
        }
      // Unique angular momentum values
      std::vector<dftfe::Int> ang_mom_unique_list;
      auto                    is_unique =
        std::adjacent_find(ang_mom_list.begin(), ang_mom_list.end()) ==
        ang_mom_list.end();
      if (!is_unique)
        {
          ang_mom_unique_list = ang_mom_list;
          std::sort(ang_mom_unique_list.begin(), ang_mom_unique_list.end());
          auto it =
            ang_mom_unique_list.erase(std::unique(ang_mom_unique_list.begin(),
                                                  ang_mom_unique_list.end()));
          ang_mom_unique_list.resize(distance(ang_mom_unique_list.begin(), it));
        }
      else
        {
          ang_mom_unique_list = ang_mom_list;
        }

      // Beta index for same angular momentum
      std::vector<std::vector<dftfe::Int>> beta_index;
      for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          beta_index.push_back((std::vector<dftfe::Int>()));
          for (dftfe::Int j = 0; j < ang_mom_list.size(); j++)
            {
              if (ang_mom_list[j] == ang_mom_unique_list[i])
                {
                  beta_index[i].push_back(j + 1);
                }
            }
        }

      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("radial_grid");
      radial_tag.push_back("values");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Extracting projector data according to angular momentum
      for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          std::vector<std::vector<double>> beta_values;
          std::string                      proj_str = "/allelectron_partial_l";
          proj_str += std::to_string(ang_mom_unique_list[i]);
          proj_str += ".dat";
          for (dftfe::Int j = 0; j < beta_index[i].size(); j++)
            {
              std::vector<std::string> beta_tag;
              beta_tag.push_back("ae_partial_wave");
              beta_values.push_back(std::vector<double>());
              beta_values[j] =
                XmlTagReaderMainMulti(beta_tag, file_path_in, beta_index[i][j]);
            }

          std::fstream file;
          file.open(file_path_out + proj_str, std::ios::out);
          file << std::fixed
               << std::setprecision(std::numeric_limits<double>::max_digits10);
          if (file.is_open())
            {
              for (dftfe::Int l = 0; l < radial_coord.size(); l++)
                {
                  file << radial_coord[l] << " ";
                  for (dftfe::Int m = 0; m < beta_values.size(); m++)
                    {
                      if (m != (beta_values.size() - 1))
                        file << beta_values[m][l] << " ";
                      else
                        file << beta_values[m][l] << std::endl;
                    }
                }
            }
          file.close();
        }
    }
    void
    xmltoShapeFunctionFile(std::string file_path_in, std::string file_path_out)
    {
      // LMaxAug
      dftfe::Int LmaxAug = 2;

      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("radial_grid");
      radial_tag.push_back("values");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Defining RmaxAug and RmaxComp
      std::vector<std::string> header_tag;
      std::vector<std::string> attr_type;
      std::vector<std::string> attr_value;
      header_tag.push_back("paw_radius");
      XmlTagReaderAttr(header_tag, file_path_in, &attr_type, &attr_value);
      double RmaxAug = std::stod(attr_value[0]);

      double     dR     = 1000.0;
      dftfe::Int RIndex = 0;
      double     RmaxComp;
      for (dftfe::Int i = 0; i < radial_coord.size(); i++)
        {
          if (std::fabs(radial_coord[i] - RmaxAug) < dR)
            {
              dR     = std::fabs(radial_coord[i] - RmaxAug);
              RIndex = i;
            }
        }
      if (radial_coord[RIndex] - RmaxAug > 1E-6)
        {
          RmaxComp = radial_coord[RIndex - 1];
        }
      else
        RmaxComp = radial_coord[RIndex];

      // Extracting the shape function type
      std::vector<std::string> header_tag_2;
      std::vector<std::string> attr_type_2;
      std::vector<std::string> attr_value_2;
      header_tag_2.push_back("shape_function");
      XmlTagReaderAttr(header_tag_2, file_path_in, &attr_type_2, &attr_value_2);
      dftfe::uInt index     = 0;
      std::string to_search = "type";
      auto it = std::find(attr_type_2.begin(), attr_type_2.end(), to_search);
      if (it == attr_type_2.end())
        {
          std::cout << "type attribute not found";
          return;
        }
      else
        {
          index = std::distance(attr_type_2.begin(), it);
        }
      std::string shape_function_type = attr_value_2[index];

      // Extracting rc
      index     = 0;
      to_search = "rc";
      it        = std::find(attr_type_2.begin(), attr_type_2.end(), to_search);
      if (it == attr_type_2.end())
        {
          std::cout << "rc attribute not found";
          return;
        }
      else
        {
          index = std::distance(attr_type_2.begin(), it);
        }

      double rc = std::stod(attr_value_2[index]);

      // Evaluating expressions based on type of shape function
      std::vector<std::vector<double>> shape_function_values;
      if (shape_function_type == "gauss")
        {
          for (dftfe::Int l = 0; l <= LmaxAug; l++)
            {
              shape_function_values.push_back(std::vector<double>());
              for (dftfe::Int r = 0; r < radial_coord.size(); r++)
                {
                  double Value = 0;
                  if (radial_coord[r] < RmaxComp)
                    {
                      Value = pow(radial_coord[r], l) *
                              (std::exp(-pow(radial_coord[r] / rc, 2)));
                    }

                  shape_function_values[l].push_back(Value);
                }
            }
        }
      if (shape_function_type == "sinc")
        {
          double pi = 2 * acos(0.0);
          for (dftfe::Int l = 0; l <= LmaxAug; l++)
            {
              shape_function_values.push_back(std::vector<double>());
              for (dftfe::Int r = 0; r < radial_coord.size(); r++)
                {
                  double Value = 0;
                  if (radial_coord[r] < std::min(rc, RmaxComp))
                    {
                      if (radial_coord[r] == 0)
                        {
                          Value = pow(radial_coord[r], l) * 1;
                        }
                      else
                        {
                          Value = pow(radial_coord[r], l) *
                                  (pow(sin(pi * (radial_coord[r] / rc)) /
                                         (pi * (radial_coord[r] / rc)),
                                       2));
                        }
                    }

                  shape_function_values[l].push_back(Value);
                }
            }
        }
      if (shape_function_type == "bessel")
        {
          std::vector<double> q1 = {3.141592653589793 / rc,
                                    4.493409457909095 / rc,
                                    5.76345919689455 / rc};
          std::vector<double> q2 = {6.283185307179586 / rc,
                                    7.7252518369375 / rc,
                                    9.095011330476355 / rc};
          for (dftfe::Int l = 0; l <= LmaxAug; l++)
            {
              shape_function_values.push_back(std::vector<double>());
              double derJ1 = l > 0 ? std::sph_bessel(l - 1, q1[l] * rc) -
                                       double(l + 1) / (q1[l] * rc) *
                                         std::sph_bessel(l, q1[l] * rc) :
                                     -std::sph_bessel(l + 1, q1[l] * rc);
              double derJ2 = l > 0 ? std::sph_bessel(l - 1, q2[l] * rc) -
                                       double(l + 1) / (q2[l] * rc) *
                                         std::sph_bessel(l, q2[l] * rc) :
                                     -std::sph_bessel(l + 1, q2[l] * rc);
              double alpha = -q1[l] / q2[l] * derJ1 / derJ2;
              for (dftfe::Int r = 0; r < radial_coord.size(); r++)
                {
                  double Value = 0;
                  if (radial_coord[r] <= std::min(rc, RmaxComp))
                    {
                      Value =
                        (std::sph_bessel(l, q1[l] * radial_coord[r]) +
                         alpha * (std::sph_bessel(l, q2[l] * radial_coord[r])));
                    }

                  shape_function_values[l].push_back(Value);
                }
            }
        }

      // Writing to the file
      std::fstream file;
      file.open(file_path_out + "/shape_functions.dat", std::ios::out);
      file << std::fixed
           << std::setprecision(std::numeric_limits<double>::max_digits10);
      if (file.is_open())
        {
          for (dftfe::Int l = 0; l < radial_coord.size(); l++)
            {
              file << radial_coord[l] << " ";
              for (dftfe::Int m = 0; m <= LmaxAug; m++)
                {
                  if (m != (LmaxAug))
                    file << shape_function_values[m][l] << " ";
                  else
                    file << shape_function_values[m][l] << std::endl;
                }
            }
        }
      file.close();
    }
    void
    xmltoDerivativeFile(std::string file_path_in, std::string file_path_out)
    {
      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("radial_grid");
      radial_tag.push_back("values");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Extracting derivative values
      std::vector<double>      derivative_values;
      std::vector<std::string> derivative_tag;
      derivative_tag.push_back("radial_grid");
      derivative_tag.push_back("derivatives");
      derivative_values = XmlTagReaderMain(derivative_tag, file_path_in);

      // Writing derivatives.dat
      std::fstream file;
      file.open(file_path_out + "/derivatives.dat", std::ios::out);
      file << std::fixed
           << std::setprecision(std::numeric_limits<double>::max_digits10);
      if (file.is_open())
        {
          for (dftfe::Int l = 0; l < radial_coord.size(); l++)
            {
              file << radial_coord[l] << " " << derivative_values[l]
                   << std::endl;
            }
        }
      file.close();
    }
    void
    xmltoOrbitalPAW(
      std::string                               file_path_in,
      std::string                               file_path_out,
      const std::map<std::string, std::string> &mapStateIndexToQuantumNumber)
    {
      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("radial_grid");
      radial_tag.push_back("values");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Finding the Orbitals to prdftfe::Int
      std::vector<dftfe::Int>  n;
      std::vector<dftfe::Int>  l;
      std::vector<std::string> orbitals_to_print;
      std::vector<std::string> header_tag;
      header_tag.push_back("valence_states");
      std::vector<std::string> ppw_tag;
      ppw_tag.push_back("pseudo_partial_wave");
      std::vector<std::string> valence_tag;
      valence_tag.push_back("valence_states");
      valence_tag.push_back("state");
      dftfe::Int no_states = xmlNodeChildCount(header_tag, file_path_in);
      for (dftfe::Int i = 0; i < no_states; i++)
        {
          std::vector<std::string> attr_type;
          std::vector<std::string> attr_value;
          XmlTagReaderMultiAttr(
            valence_tag, file_path_in, &attr_type, &attr_value, i + 1);
          dftfe::uInt index     = 0;
          std::string to_search = "f";
          auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it != attr_type.end())
            {
              index = std::distance(attr_type.begin(), it);
              if (std::stod(attr_value[index]) != 0)
                {
                  dftfe::uInt index_2     = 0;
                  std::string to_search_2 = "id";
                  auto        it_2 =
                    std::find(attr_type.begin(), attr_type.end(), to_search_2);
                  index_2 = std::distance(attr_type.begin(), it_2);
                  orbitals_to_print.push_back(
                    mapStateIndexToQuantumNumber.find(attr_value[index_2])
                      ->second);

                  dftfe::uInt index_3     = 0;
                  std::string to_search_3 = "n";
                  auto        it_3 =
                    std::find(attr_type.begin(), attr_type.end(), to_search_3);
                  index_3 = std::distance(attr_type.begin(), it_3);
                  n.push_back(std::stoi(attr_value[index_3]));

                  dftfe::uInt index_4     = 0;
                  std::string to_search_4 = "l";
                  auto        it_4 =
                    std::find(attr_type.begin(), attr_type.end(), to_search_4);
                  index_4 = std::distance(attr_type.begin(), it_4);
                  l.push_back(std::stoi(attr_value[index_4]));
                }
            }
        }


      for (dftfe::Int orb = 0; orb < orbitals_to_print.size(); orb++)
        {
          for (dftfe::Int j = 0; j < no_states; j++)
            {
              std::vector<std::string> attr_type;
              std::vector<std::string> attr_value;

              XmlTagReaderAttrMulti(
                ppw_tag, file_path_in, &attr_type, &attr_value, j + 1);
              std::vector<double> main_value =
                XmlTagReaderMainMulti(ppw_tag, file_path_in, j + 1);
              dftfe::uInt index     = 0;
              std::string to_search = "state";
              auto        it =
                std::find(attr_type.begin(), attr_type.end(), to_search);
              if (it != attr_type.end())
                {
                  index = std::distance(attr_type.begin(), it);
                  if (attr_value[index] == orbitals_to_print[orb])
                    {
                      std::string orb_str = "/psi";
                      orb_str             = orb_str + std::to_string(n[orb]) +
                                std::to_string(l[orb]) + ".inp";
                      // Writing psi_nl.inp
                      std::fstream file;
                      file.open(file_path_out + orb_str, std::ios::out);
                      file << std::fixed
                           << std::setprecision(
                                std::numeric_limits<double>::max_digits10);
                      if (file.is_open())
                        {
                          for (dftfe::Int l = 0; l < radial_coord.size(); l++)
                            {
                              file << radial_coord[l] << " " << main_value[l]
                                   << std::endl;
                            }
                        }
                      file.close();
                    }
                }
            }
        }
    }
    void
    xmltoSummaryPAWFile(
      std::string                               file_path_in,
      std::string                               file_path_out,
      const std::map<std::string, std::string> &mapStateIndexToQuantumNumber)
    {
      // List of momentum values
      std::vector<std::string> tag_name;
      tag_name.push_back("projector_function");
      dftfe::Int no_projector_func = XmlTagOccurance(tag_name, file_path_in);
      std::vector<dftfe::Int>  ang_mom_list;
      std::vector<std::string> ang_mom_list_str;
      for (dftfe::Int i = 0; i < no_projector_func; i++)
        {
          std::vector<std::string> header_tag;
          std::vector<std::string> attr_type;
          std::vector<std::string> attr_value;
          header_tag.push_back("projector_function");
          XmlTagReaderAttrMulti(
            header_tag, file_path_in, &attr_type, &attr_value, i + 1);
          dftfe::uInt index     = 0;
          std::string to_search = "state";
          auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it == attr_type.end())
            {
              std::cout << "state attribute not found";
              return;
            }
          else
            {
              index = std::distance(attr_type.begin(), it);
              ang_mom_list_str.push_back(
                mapStateIndexToQuantumNumber.find(attr_value[index])->second);
            }
        }

      for (dftfe::Int j = 0; j < ang_mom_list_str.size(); j++)
        {
          for (dftfe::Int k = 0; k < ang_mom_list_str[j].size(); k++)
            {
              if (ang_mom_list_str[j][k] == 's')
                {
                  ang_mom_list.push_back(0);
                  break;
                }
              if (ang_mom_list_str[j][k] == 'p')
                {
                  ang_mom_list.push_back(1);
                  break;
                }
              if (ang_mom_list_str[j][k] == 'd')
                {
                  ang_mom_list.push_back(2);
                  break;
                }
              if (ang_mom_list_str[j][k] == 'f')
                {
                  ang_mom_list.push_back(3);
                  break;
                }
            }
        }
      // Unique angular momentum values
      std::vector<dftfe::Int> ang_mom_unique_list;
      auto                    is_unique =
        std::adjacent_find(ang_mom_list.begin(), ang_mom_list.end()) ==
        ang_mom_list.end();
      if (!is_unique)
        {
          ang_mom_unique_list = ang_mom_list;
          std::sort(ang_mom_unique_list.begin(), ang_mom_unique_list.end());
          auto it =
            ang_mom_unique_list.erase(std::unique(ang_mom_unique_list.begin(),
                                                  ang_mom_unique_list.end()));
          ang_mom_unique_list.resize(distance(ang_mom_unique_list.begin(), it));
        }
      else
        {
          ang_mom_unique_list = ang_mom_list;
        }

      std::vector<dftfe::Int> ang_mom_multiplicity_list;
      for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          dftfe::Int count = 0;
          for (dftfe::Int j = 0; j < ang_mom_list.size(); j++)
            {
              if (ang_mom_list[j] == ang_mom_unique_list[i])
                {
                  count++;
                }
            }
          ang_mom_multiplicity_list.push_back(count);
        }

      dftfe::Int                           row_index = 0;
      dftfe::Int                           index     = 0;
      std::vector<std::vector<dftfe::Int>> out_proj_arr;
      for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          dftfe::Int l = ang_mom_unique_list[i];
          for (dftfe::Int j = 0; j < ang_mom_multiplicity_list[i]; j++)
            {
              dftfe::Int m = -l;
              for (dftfe::Int k = 0; k < 2 * l + 1; k++)
                {
                  out_proj_arr.push_back((std::vector<dftfe::Int>()));
                  out_proj_arr[row_index].push_back(index);
                  out_proj_arr[row_index].push_back(l);
                  out_proj_arr[row_index].push_back(m);
                  m++;
                  row_index++;
                }
              index++;
            }
        }
      // Extracting rmaxaug
      std::vector<std::string> header_tag_r;
      std::vector<std::string> attr_type_r;
      std::vector<std::string> attr_value_r;
      header_tag_r.push_back("paw_radius");
      XmlTagReaderAttr(header_tag_r, file_path_in, &attr_type_r, &attr_value_r);

      double       RmaxAug = std::stod(attr_value_r[0]);
      std::fstream file;
      file_path_out.append("/PseudoAtomDat");
      file.open(file_path_out, std::ios::out);
      file << std::fixed
           << std::setprecision(std::numeric_limits<double>::max_digits10);

      if (file.is_open())
        {
          // nlcc Flag
          std::vector<std::string> header_tag_nlcc;
          std::vector<std::string> attr_type_nlcc;
          std::vector<std::string> attr_value_nlcc;
          header_tag_nlcc.push_back("atom");
          XmlTagReaderAttr(header_tag_nlcc,
                           file_path_in,
                           &attr_type_nlcc,
                           &attr_value_nlcc);
          dftfe::uInt index_2     = 0;
          std::string to_search_2 = "core";
          auto        it_2        = std::find(attr_type_nlcc.begin(),
                                attr_type_nlcc.end(),
                                to_search_2);
          if (it_2 == attr_type_nlcc.end())
            {
              std::cout << "core attribute not found";
              return;
            }
          else
            {
              index_2 = std::distance(attr_type_nlcc.begin(), it_2);
              if (std::stod(attr_value_nlcc[index_2]) > 1E-1)
                file << "T" << std::endl;
              else
                file << "F" << std::endl;
            }

          // RmaxAug
          file << RmaxAug << std::endl;

          // ke_core
          std::vector<std::string> header_tag_ke;
          std::vector<std::string> attr_type_ke;
          std::vector<std::string> attr_value_ke;
          header_tag_ke.push_back("core_energy");
          XmlTagReaderAttr(header_tag_ke,
                           file_path_in,
                           &attr_type_ke,
                           &attr_value_ke);
          dftfe::uInt index     = 0;
          std::string to_search = "kinetic";
          auto        it =
            std::find(attr_type_ke.begin(), attr_type_ke.end(), to_search);
          if (it == attr_type_ke.end())
            {
              std::cout << "kinetic attribute not found";
              return;
            }
          else
            {
              index = std::distance(attr_type_ke.begin(), it);
              file << std::stod(attr_value_ke[index]) << std::endl;
            }

          // Shape Function

          // Extracting the shape function type
          std::vector<std::string> header_tag_2;
          std::vector<std::string> attr_type_2;
          std::vector<std::string> attr_value_2;
          header_tag_2.push_back("shape_function");
          XmlTagReaderAttr(header_tag_2,
                           file_path_in,
                           &attr_type_2,
                           &attr_value_2);
          dftfe::uInt index_3     = 0;
          std::string to_search_3 = "type";
          auto        it_3 =
            std::find(attr_type_2.begin(), attr_type_2.end(), to_search_3);
          if (it_3 == attr_type_2.end())
            {
              std::cout << "type attribute not found";
              return;
            }
          else
            {
              index_3 = std::distance(attr_type_2.begin(), it_3);
            }
          std::string shape_function_type = attr_value_2[index_3];

          dftfe::uInt index_4     = 0;
          std::string to_search_4 = "rc";
          auto        it_4 =
            std::find(attr_type_2.begin(), attr_type_2.end(), to_search_4);
          if (it_4 == attr_type_2.end())
            {
              std::cout << "rc attribute not found";
              return;
            }
          else
            {
              index_4 = std::distance(attr_type_2.begin(), it_4);
            }

          double rc = std::stod(attr_value_2[index_4]);

          if (shape_function_type == "sinc")
            {
              file << "2" << std::endl;
              file << rc << std::endl;
            }

          else if (shape_function_type == "gauss")
            {
              file << "1" << std::endl;
              file << rc << std::endl;
            }

          else
            {
              file << "0" << std::endl;
              file << rc << std::endl;
            }
        }

      // l count
      file << ang_mom_list.size() << std::endl;

      for (dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          file << ang_mom_multiplicity_list[i] << std::endl;
        }

      if (ang_mom_unique_list.size() != 4)
        {
          for (dftfe::Int s = 0; s < (4 - ang_mom_unique_list.size()); s++)
            {
              file << 0 << std::endl;
            }
        }
      // // Projector data


      // dftfe::Int m = out_proj_arr.size();
      // dftfe::Int n = out_proj_arr[0].size();
      // for (dftfe::Int i = 0; i < m; i++) {
      //     for (dftfe::Int j = 0; j < n; j++)
      //         file << out_proj_arr[i][j] << " ";
      //     file << std::endl;
      // }
      // for(dftfe::Int i = 0; i < ang_mom_unique_list.size(); i++){
      //     file <<
      //     "proj_l"+std::to_string(ang_mom_unique_list[i])+".dat"<<std::endl;
      //     file <<
      //     "allelectron_partial_l"+std::to_string(ang_mom_unique_list[i])+".dat"<<std::endl;
      //     file <<
      //     "smooth_partial_l"+std::to_string(ang_mom_unique_list[i])+".dat"<<std::endl;
      //     file << ang_mom_multiplicity_list[i]<< std::endl;
      // }


      // // nlcc Flag
      // std::vector<std::string> header_tag_nlcc;
      // std::vector<std::string> attr_type_nlcc;
      // std::vector<std::string> attr_value_nlcc;
      // header_tag_nlcc.push_back("atom");
      // XmlTagReaderAttr(header_tag_nlcc,file_path_in,&attr_type_nlcc,&attr_value_nlcc);
      // dftfe::uInt index_2 = 0;
      // std::string to_search_2 = "core";
      // auto it_2 =
      // std::find(attr_type_nlcc.begin(),attr_type_nlcc.end(),to_search_2); if
      // (it_2 == attr_type_nlcc.end()) {   std::cout<<"core attribute not
      // found";
      //     return;
      // }
      // else
      // {
      //     index_2 = std::distance(attr_type_nlcc.begin(), it_2);
      //     if(std::stod(attr_value_nlcc[index_2])>  1E-1)
      //         file<<"T"<<std::endl;
      //     else
      //         file<<"F"<<std::endl;
      // }
    }


    dftfe::Int
    pseudoPotentialTypeCheck(const std::string file_path_in)
    {
      auto       file_type = file_path_in.substr(file_path_in.size() - 3, 3);
      dftfe::Int pawFlag   = -1;
      std::vector<std::string> header_tag;
      std::vector<std::string> attr_type;
      std::vector<std::string> attr_value;

      if (file_type == "upf")
        {
          header_tag.push_back("PP_HEADER");
          XmlTagReaderAttr(header_tag, file_path_in, &attr_type, &attr_value);
          std::string to_search = "is_paw";
          dftfe::uInt index     = 0;
          auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it == attr_type.end())
            {
              std::cout << "PAW attribute not found";
              return (-1);
            }
          else
            {
              index = std::distance(attr_type.begin(), it);
            }
          if (attr_value[index] == "T")
            pawFlag = 1;
          else
            pawFlag = 0;
        }
      else if (file_type == "xml")
        {
          std::cout << "Line 712" << std::endl;
          header_tag.push_back("paw_radius");
          XmlTagReaderAttr(header_tag, file_path_in, &attr_type, &attr_value);
          if (attr_type.size() > 0)
            pawFlag = 1;
          else
            pawFlag = 0;
          std::cout << "PAW Flag: " << pawFlag << std::endl;
        }
      else
        std::cout << "Unrecognised file format..." << std::endl;
      return pawFlag;
    }

    dftfe::Int
    pseudoPotentialToDftfeParser(const std::string file_path_in,
                                 const std::string file_path_out,
                                 const dftfe::Int  verbosity,
                                 dftfe::uInt      &nlccFlag,
                                 dftfe::uInt      &socFlag,
                                 dftfe::uInt      &pawFlag)
    {
      auto file_type = file_path_in.substr(file_path_in.size() - 3, 3);
      if (file_type == "upf")
        {
          xmltoSummaryFile(file_path_in, file_path_out);
          xmltoProjectorFile(file_path_in, file_path_out);
          xmltoLocalPotential(file_path_in, file_path_out);
          xmltoDenomFile(file_path_in, file_path_out);
          xmltoDensityFile(file_path_in, file_path_out);
          xmltoOrbitalFile(file_path_in, file_path_out);


          std::vector<std::string> header_tag;
          std::vector<std::string> attr_type;
          std::vector<std::string> attr_value;
          header_tag.push_back("PP_HEADER");
          XmlTagReaderAttr(header_tag, file_path_in, &attr_type, &attr_value);
          // NLCC
          dftfe::uInt index     = 0;
          std::string to_search = "core_correction";
          auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it == attr_type.end())
            {
              throw std::invalid_argument(
                "core correction attribute not found");
            }
          else
            {
              index = std::distance(attr_type.begin(), it);
            }
          if (attr_value[index] == "T")
            {
              nlccFlag = 1;
              xmltoCoreDensityFile(file_path_in, file_path_out);
            }
          else
            nlccFlag = 0;

          // SOC
          to_search = "has_so";
          it        = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it == attr_type.end())
            {
              throw std::invalid_argument(
                "spin orbit coupling attribute not found");
            }
          else
            {
              index = std::distance(attr_type.begin(), it);
            }
          if (attr_value[index] == "T")
            socFlag = 1;
          else
            socFlag = 0;

          // PAW
          to_search = "is_paw";
          it        = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it == attr_type.end())
            {
              throw std::invalid_argument("PAW attribute not found");
            }
          else
            {
              index = std::distance(attr_type.begin(), it);
            }
          if (attr_value[index] == "T")
            pawFlag = 1;
          else
            pawFlag = 0;
          return (0);
        }
      else if (file_type == "xml")
        {
          std::map<std::string, std::string> mapStateIndexToQuantumNumber;
          createStateIndexToQuantumNumberMap(file_path_in,
                                             file_path_out,
                                             mapStateIndexToQuantumNumber);
          xmltoZeroPotFile(file_path_in, file_path_out);
          xmltoKineticDifFile(file_path_in, file_path_out);
          xmltoPseudoValDensityFile(file_path_in, file_path_out);
          xmltoProjectorPAWFile(file_path_in,
                                file_path_out,
                                mapStateIndexToQuantumNumber);
          xmltoSmoothPartialPAWFile(file_path_in,
                                    file_path_out,
                                    mapStateIndexToQuantumNumber);
          xmltoAllElecPartialPAWFile(file_path_in,
                                     file_path_out,
                                     mapStateIndexToQuantumNumber);
          xmltoShapeFunctionFile(file_path_in, file_path_out);
          xmltoDerivativeFile(file_path_in, file_path_out);
          xmltoOrbitalPAW(file_path_in,
                          file_path_out,
                          mapStateIndexToQuantumNumber);
          xmltoSummaryPAWFile(file_path_in,
                              file_path_out,
                              mapStateIndexToQuantumNumber);

          // nlcc Flag
          std::vector<std::string> header_tag_nlcc;
          std::vector<std::string> attr_type_nlcc;
          std::vector<std::string> attr_value_nlcc;
          header_tag_nlcc.push_back("atom");
          XmlTagReaderAttr(header_tag_nlcc,
                           file_path_in,
                           &attr_type_nlcc,
                           &attr_value_nlcc);
          dftfe::uInt index_2     = 0;
          std::string to_search_2 = "core";
          auto        it_2        = std::find(attr_type_nlcc.begin(),
                                attr_type_nlcc.end(),
                                to_search_2);
          if (it_2 == attr_type_nlcc.end())
            {
              std::cout << "core attribute not found";
              return (-1);
            }
          else
            {
              index_2 = std::distance(attr_type_nlcc.begin(), it_2);
              if (std::stod(attr_value_nlcc[index_2]) > 1E-1)
                {
                  nlccFlag = 1;
                  xmltoAllElecCoreDensityFile(file_path_in, file_path_out);
                  xmltoPseudoCoreDensityFile(file_path_in, file_path_out);
                }
              else
                nlccFlag = 0;
            }

          // PAW flag
          pawFlag = 1;
          return (0);
        }

      else
        return (-1);
    }


  } // namespace pseudoUtils
} // namespace dftfe
