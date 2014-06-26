/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut f�r Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef PCLFUNCTIONS_H
#define PCLFUNCTIONS_H

#include "pclDefines.h"
#include "pclStructures.h"

#include "../common/sharedStructures.h"

namespace ito 
{

class DataObject; //forward declaration

namespace pclHelper
{
    void POINTCLOUD_EXPORT PointXYZRGBtoXYZRGBA (const pcl::PointXYZRGB& in, pcl::PointXYZRGBA&  out);
    void POINTCLOUD_EXPORT PointXYZRGBAtoXYZRGB (const pcl::PointXYZRGBA& in, pcl::PointXYZRGB&  out);
    void POINTCLOUD_EXPORT PointCloudXYZRGBtoXYZRGBA(const pcl::PointCloud<pcl::PointXYZRGB>& in, pcl::PointCloud<pcl::PointXYZRGBA>& out);

#if PCL_VERSION_COMPARE(>=,1,7,0)
    ito::RetVal POINTCLOUD_EXPORT pointCloud2ToPCLPointCloud(const pcl::PCLPointCloud2 &msg, PCLPointCloud *pc);
    ito::RetVal POINTCLOUD_EXPORT pclPointCloudToPointCloud2(const PCLPointCloud &pc, pcl::PCLPointCloud2 &msg);
    ito::tPCLPointType POINTCLOUD_EXPORT guessPointType(const pcl::PCLPointCloud2 &msg);
#else
    ito::RetVal POINTCLOUD_EXPORT pointCloud2ToPCLPointCloud(const sensor_msgs::PointCloud2 &msg, PCLPointCloud *pc);
    ito::RetVal POINTCLOUD_EXPORT pclPointCloudToPointCloud2(const PCLPointCloud &pc, sensor_msgs::PointCloud2 &msg);
    ito::tPCLPointType POINTCLOUD_EXPORT guessPointType(const sensor_msgs::PointCloud2 &msg);
#endif

    ito::RetVal POINTCLOUD_EXPORT pointCloudFromXYZ(const DataObject* mapX, const DataObject* mapY, const DataObject* mapZ, PCLPointCloud &out, bool deleteNaN = false);
    ito::RetVal POINTCLOUD_EXPORT pointCloudFromXYZI(const DataObject* mapX, const DataObject* mapY, const DataObject* mapZ, const DataObject* mapI, PCLPointCloud &out, bool deleteNaN = false);
    ito::RetVal POINTCLOUD_EXPORT pointCloudFromDisparity(const DataObject* mapDisp, PCLPointCloud &out, bool deleteNaN = false);
    ito::RetVal POINTCLOUD_EXPORT pointCloudFromDisparityI(const DataObject* mapDisp, const DataObject *mapI, PCLPointCloud &out, bool deleteNaN = false);
    ito::RetVal POINTCLOUD_EXPORT pointCloudFromDisparityRGBA(const DataObject* mapDisp, const DataObject *mapColor, PCLPointCloud &out, bool deleteNaN = false);

    ito::RetVal POINTCLOUD_EXPORT pointCloudToDObj(const PCLPointCloud *pc, DataObject &out);

    ito::RetVal POINTCLOUD_EXPORT dataObj4x4ToEigenAffine3f(const DataObject *in, Eigen::Affine3f &out);
    ito::RetVal POINTCLOUD_EXPORT eigenAffine3fToDataObj4x4(const Eigen::Affine3f *in, DataObject &out);

    //template<typename _Tp, int _Rows, int _Cols> ito::RetVal eigenMatrixToDataObj(const Eigen::Matrix<_Tp,_Rows,_Cols> &mat, DataObject &out);

    //ito::RetVal writeBinary(const std::string &filename, const ito::PCLPointCloud &cloud);
    //ito::RetVal readBinary(const std::string &filename, ito::PCLPointCloud &cloud);

} //end namespace pclHelper

} //end namespace ito

#endif