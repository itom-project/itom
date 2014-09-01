/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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

#include "pclFunctions.h"

#include "../common/typeDefs.h"
#include "../DataObject/dataobj.h"
#include "../DataObject/dataObjectFuncs.h"

#include <pcl/io/io.h>

#if PCL_VERSION_COMPARE(>=,1,7,0)
    #include <pcl/conversions.h>
#else
    #include <pcl/ros/conversions.h>
#endif

namespace ito 
{

namespace pclHelper
{

//------------------------------------------------------------------------------------------------------------------------------
//! converts pcl::PointXYZRGB to pcl::PointXYZRGBA
/*!
    \param [in] in is the input point
    \param [in/out] out is the converted output point, where alpha is set to 255 (no transparency)
    \sa PointXYZRGBAtoXYZRGB
*/
void PointXYZRGBtoXYZRGBA (const pcl::PointXYZRGB& in, pcl::PointXYZRGBA&  out)
{
    out.x = in.x; out.y = in.y; out.z = in.z;
    out.r = in.r; out.g = in.g; out.b = in.b; out.PCLALPHA = 255;
}

//------------------------------------------------------------------------------------------------------------------------------
//! converts pcl::PointXYZRGBA to pcl::PointXYZRGB
/*!
    \param [in] in is the input point
    \param [in/out] out is the converted output point
    \sa PointXYZRGBtoXYZRGBA
*/
void PointXYZRGBAtoXYZRGB (const  pcl::PointXYZRGBA& in, pcl::PointXYZRGB&  out)
{
    out.x = in.x; out.y = in.y; out.z = in.z;
    out.r = in.r; out.g = in.g; out.b = in.b;
}

//------------------------------------------------------------------------------------------------------------------------------
void PointCloudXYZRGBtoXYZRGBA(const pcl::PointCloud<pcl::PointXYZRGB>& in, pcl::PointCloud<pcl::PointXYZRGBA>& out)
{
    out.width = in.width;
    out.height = in.height;

    for (size_t i = 0; i < in.points.size(); i++)
    {
        pcl::PointXYZRGBA p;
        ito::pclHelper::PointXYZRGBtoXYZRGBA (in.points[i], p);
        out.points.push_back (p);
    }
}

//------------------------------------------------------------------------------------------------------------------------------
#if PCL_VERSION_COMPARE(>=,1,7,0)
ito::RetVal pointCloud2ToPCLPointCloud(const pcl::PCLPointCloud2 &msg, PCLPointCloud *pc)
{
    RetVal retval = retOk;
    if (!pc)
    {
        return RetVal(retError, 0, "PCLPointCloud is NULL");
    }

    ito::tPCLPointType pointType = pc->getType();
    pcl::MsgFieldMap field_map;

    switch(pointType)
    {
    case ito::pclXYZ:
        pcl::createMapping<pcl::PointXYZ>(msg.fields, field_map);
        pcl::fromPCLPointCloud2(msg, *(pc->toPointXYZ()), field_map);
        break;
    case ito::pclXYZI:
        pcl::createMapping<pcl::PointXYZI>(msg.fields, field_map);
        pcl::fromPCLPointCloud2(msg, *(pc->toPointXYZI()), field_map);
        break;
    case ito::pclXYZRGBA:
        pcl::createMapping<pcl::PointXYZRGBA>(msg.fields, field_map);
        pcl::fromPCLPointCloud2(msg, *(pc->toPointXYZRGBA()), field_map);
        break;
    case ito::pclXYZNormal:
        pcl::createMapping<pcl::PointNormal>(msg.fields, field_map);
        pcl::fromPCLPointCloud2(msg, *(pc->toPointXYZNormal()), field_map);
        break;
    case ito::pclXYZINormal:
        pcl::createMapping<pcl::PointXYZINormal>(msg.fields, field_map);
        pcl::fromPCLPointCloud2(msg, *(pc->toPointXYZINormal()), field_map);
        break;
    case ito::pclXYZRGBNormal:
        pcl::createMapping<pcl::PointXYZRGBNormal>(msg.fields, field_map);
        pcl::fromPCLPointCloud2(msg, *(pc->toPointXYZRGBNormal()), field_map);
        break;
    default:
        retval += RetVal(retError, 0, "given point cloud cannot be converted into desired type");
        break;
    }

    return retval;
}
#else
ito::RetVal pointCloud2ToPCLPointCloud(const sensor_msgs::PointCloud2 &msg, PCLPointCloud *pc)
{
    RetVal retval = retOk;
    if (!pc)
    {
        return RetVal(retError, 0, "PCLPointCloud is NULL");
    }

    ito::tPCLPointType pointType = pc->getType();
    pcl::MsgFieldMap field_map;

    switch(pointType)
    {
    case ito::pclXYZ:
        pcl::createMapping<pcl::PointXYZ>(msg.fields, field_map);
        pcl::fromROSMsg(msg, *(pc->toPointXYZ()), field_map);
        break;
    case ito::pclXYZI:
        pcl::createMapping<pcl::PointXYZI>(msg.fields, field_map);
        pcl::fromROSMsg(msg, *(pc->toPointXYZI()), field_map);
        break;
    case ito::pclXYZRGBA:
        pcl::createMapping<pcl::PointXYZRGBA>(msg.fields, field_map);
        pcl::fromROSMsg(msg, *(pc->toPointXYZRGBA()), field_map);
        break;
    case ito::pclXYZNormal:
        pcl::createMapping<pcl::PointNormal>(msg.fields, field_map);
        pcl::fromROSMsg(msg, *(pc->toPointXYZNormal()), field_map);
        break;
    case ito::pclXYZINormal:
        pcl::createMapping<pcl::PointXYZINormal>(msg.fields, field_map);
        pcl::fromROSMsg(msg, *(pc->toPointXYZINormal()), field_map);
        break;
    case ito::pclXYZRGBNormal:
        pcl::createMapping<pcl::PointXYZRGBNormal>(msg.fields, field_map);
        pcl::fromROSMsg(msg, *(pc->toPointXYZRGBNormal()), field_map);
        break;
    default:
        retval += RetVal(retError, 0, "given point cloud cannot be converted into desired type");
        break;
    }

    return retval;
}
#endif

//------------------------------------------------------------------------------------------------------------------------------
#if PCL_VERSION_COMPARE(>=,1,7,0)
ito::RetVal pclPointCloudToPointCloud2(const PCLPointCloud &pc, pcl::PCLPointCloud2 &msg)
{
    RetVal retval = retOk;
    
    ito::tPCLPointType pointType = pc.getType();
    pcl::MsgFieldMap field_map;

    switch(pointType)
    {
    case ito::pclXYZ:
        pcl::toPCLPointCloud2(*(pc.toPointXYZ()), msg);
        break;
    case ito::pclXYZI:
        pcl::toPCLPointCloud2(*(pc.toPointXYZI()), msg);
        break;
    case ito::pclXYZRGBA:
        pcl::toPCLPointCloud2(*(pc.toPointXYZRGBA()), msg);
        break;
    case ito::pclXYZNormal:
        pcl::toPCLPointCloud2(*(pc.toPointXYZNormal()), msg);
        break;
    case ito::pclXYZINormal:
        pcl::toPCLPointCloud2(*(pc.toPointXYZINormal()), msg);
        break;
    case ito::pclXYZRGBNormal:
        pcl::toPCLPointCloud2(*(pc.toPointXYZRGBNormal()), msg);
        break;
    case ito::pclInvalid:
        msg = pcl::PCLPointCloud2();
        break;
    default:
        retval += RetVal(retError, 0, "given point cloud cannot be converted into sensor_msgs::PointCloud2");
        break;
    }

    return retval;
}
#else
ito::RetVal pclPointCloudToPointCloud2(const PCLPointCloud &pc, sensor_msgs::PointCloud2 &msg)
{
    RetVal retval = retOk;
    
    ito::tPCLPointType pointType = pc.getType();
    pcl::MsgFieldMap field_map;

    switch(pointType)
    {
    case ito::pclXYZ:
        pcl::toROSMsg(*(pc.toPointXYZ()), msg);
        break;
    case ito::pclXYZI:
        pcl::toROSMsg(*(pc.toPointXYZI()), msg);
        break;
    case ito::pclXYZRGBA:
        pcl::toROSMsg(*(pc.toPointXYZRGBA()), msg);
        break;
    case ito::pclXYZNormal:
        pcl::toROSMsg(*(pc.toPointXYZNormal()), msg);
        break;
    case ito::pclXYZINormal:
        pcl::toROSMsg(*(pc.toPointXYZINormal()), msg);
        break;
    case ito::pclXYZRGBNormal:
        pcl::toROSMsg(*(pc.toPointXYZRGBNormal()), msg);
        break;
    case ito::pclInvalid:
        msg = sensor_msgs::PointCloud2();
        break;
    default:
        retval += RetVal(retError, 0, "given point cloud cannot be converted into sensor_msgs::PointCloud2");
        break;
    }

    return retval;
}
#endif

//------------------------------------------------------------------------------------------------------------------------------
#if PCL_VERSION_COMPARE(>=,1,7,0)
ito::tPCLPointType guessPointType(const pcl::PCLPointCloud2 &msg)
#else
ito::tPCLPointType guessPointType(const sensor_msgs::PointCloud2 &msg)
#endif
{
    if (pcl::getFieldIndex(msg,"x") >= 0 && pcl::getFieldIndex(msg,"y") >= 0 && pcl::getFieldIndex(msg,"z") >= 0)
    {
        bool rgb = (pcl::getFieldIndex(msg,"rgb") >= 0);
        bool rgba = (pcl::getFieldIndex(msg,"rgba") >= 0);
        bool normal = (pcl::getFieldIndex(msg, "normal_x") >= 0 && pcl::getFieldIndex(msg,"normal_y") >= 0 && pcl::getFieldIndex(msg,"normal_z") >= 0 && pcl::getFieldIndex(msg,"curvature") >= 0);

        //hack, since ply-files sometimes call normal_i ni. rename it now. (maybe this is fixed in pcl 1.6)
        normal |= (pcl::getFieldIndex(msg, "nx") >= 0 && pcl::getFieldIndex(msg,"ny") >= 0 && pcl::getFieldIndex(msg,"nz") >= 0 && pcl::getFieldIndex(msg,"curvature") >= 0);
        bool intensity = (pcl::getFieldIndex(msg, "intensity") >= 0);

        //pclInvalid      = 0x0000, /*!< invalid point */
        //pclXYZ          = 0x0001, /*!< point with x,y,z-value */
        //pclXYZI         = 0x0002, /*!< point with x,y,z and intensity value */
        //pclXYZRGBA      = 0x0004, /*!< point with x,y,z and r,g,b,a */
        //pclXYZNormal    = 0x0008, /*!< point with x,y,z value, its normal vector nx,ny,nz and a curvature value */
        //pclXYZINormal   = 0x0010, /*!< point with the same values than pclXYZNormal and an additional intensity value */
        //pclXYZRGBNormal = 0x0020  /*!< point with x,y,z and r,g,b and normal vector (including curvature) */
        if (!rgb && !rgba && !normal && !intensity)
        {
            return ito::pclXYZ;
        }
        else if (!rgb && !rgba && !normal && intensity)
        {
            return ito::pclXYZI;
        }
        else if ((rgb || rgba) && !normal && !intensity)
        {
            return ito::pclXYZRGBA;
        }
        else if (!rgb && !rgba && normal && !intensity)
        {
            return ito::pclXYZNormal;
        }
        else if (!rgb && !rgba && normal && intensity)
        {
            return ito::pclXYZINormal;
        }
        else if ((rgb || rgba) && normal && !intensity)
        {
            return ito::pclXYZRGBNormal;
        } 
    }

    return ito::pclInvalid;
}

//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal pointCloudFromXYZ(const DataObject* mapX, const DataObject* mapY, const DataObject* mapZ, PCLPointCloud &out, bool deleteNaN /*= false*/)
{
    RetVal retval = retOk;
    bool isDense = true;

    retval += ito::dObjHelper::verify2DDataObject(mapZ, "Z", 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 1, ito::tFloat32);
    retval += ito::dObjHelper::verify2DDataObject(mapX, "X", mapZ->getSize(0), mapZ->getSize(0), mapZ->getSize(1), mapZ->getSize(1), 1, ito::tFloat32);
    retval += ito::dObjHelper::verify2DDataObject(mapY, "Y", mapZ->getSize(0), mapZ->getSize(0), mapZ->getSize(1), mapZ->getSize(1), 1, ito::tFloat32);

    if (!retval.containsError())
    {
        uint32_t width, height;
        cv::Mat *x = reinterpret_cast<cv::Mat*>(mapX->get_mdata()[ mapX->seekMat(0) ]);
        cv::Mat *y = reinterpret_cast<cv::Mat*>(mapY->get_mdata()[ mapY->seekMat(0) ]);
        cv::Mat *z = reinterpret_cast<cv::Mat*>(mapZ->get_mdata()[ mapZ->seekMat(0) ]);

        ito::float32 *xRow, *yRow, *zRow;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        pcl::PointXYZ point;
        ito::PCLPointCloud pointCloud;

        width = mapZ->getSize(1);
        height = mapZ->getSize(0);

        if (deleteNaN)
        {
            pointCloud = ito::PCLPointCloud(ito::pclXYZ);
            cloud = pointCloud.toPointXYZ();
            pointCloud.reserve(width*height);

            size_t counter = 0;

            for (int i = 0; i < x->rows; i++)
            {
                xRow = x->ptr<ito::float32>(i);
                yRow = y->ptr<ito::float32>(i);
                zRow = z->ptr<ito::float32>(i);

                for (int j = 0; j < x->cols; j++)
                {
                    if (!(pcl_isnan(zRow[j]) || pcl_isnan(yRow[j]) || pcl_isnan(xRow[j])))
                    {
                        point.x = xRow[j];
                        point.y = yRow[j];
                        point.z = zRow[j];
                        (*cloud).push_back(point);
                        counter++;
                    }
                }
            }

            cloud->is_dense = false;
            cloud->resize(counter);
        }
        else
        {
            pointCloud = ito::PCLPointCloud(width, height, ito::pclXYZ, ito::PCLPoint(point));

            cloud = pointCloud.toPointXYZ();

            size_t counter = 0;

            for (int i = 0; i < x->rows; i++)
            {
                xRow = x->ptr<ito::float32>(i);
                yRow = y->ptr<ito::float32>(i);
                zRow = z->ptr<ito::float32>(i);

                for (int j = 0; j < x->cols; j++)
                {
                    point.x = xRow[j];
                    point.y = yRow[j];
                    point.z = zRow[j];

                    if (!pcl_isfinite(point.z) || !pcl_isfinite(point.x) || !pcl_isfinite(point.y))
                    {
                        isDense = false;
                    }

                    cloud->at(i * width + j) = point;
                    //cloud->at(j,i) = point;
                    counter++;
                }
            }

            cloud->is_dense = isDense;
        }

        out = pointCloud;
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal pointCloudFromXYZI(const DataObject* mapX, const DataObject* mapY, const DataObject* mapZ, const DataObject* mapI, PCLPointCloud &out, bool deleteNaN /*= false*/)
{
    RetVal retval = retOk;
    bool isDense = true;

    retval += ito::dObjHelper::verify2DDataObject(mapZ, "Z", 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 1, ito::tFloat32);
    retval += ito::dObjHelper::verify2DDataObject(mapX, "X", mapZ->getSize(0), mapZ->getSize(0), mapZ->getSize(1), mapZ->getSize(1), 1, ito::tFloat32);
    retval += ito::dObjHelper::verify2DDataObject(mapY, "Y", mapZ->getSize(0), mapZ->getSize(0), mapZ->getSize(1), mapZ->getSize(1), 1, ito::tFloat32);
    retval += ito::dObjHelper::verify2DDataObject(mapI, "I", mapZ->getSize(0), mapZ->getSize(0), mapZ->getSize(1), mapZ->getSize(1), 1, ito::tFloat32);

    if (!retval.containsError())
    {
        uint32_t width, height;
        cv::Mat *x = reinterpret_cast<cv::Mat*>(mapX->get_mdata()[ mapX->seekMat(0) ]);
        cv::Mat *y = reinterpret_cast<cv::Mat*>(mapY->get_mdata()[ mapY->seekMat(0) ]);
        cv::Mat *z = reinterpret_cast<cv::Mat*>(mapZ->get_mdata()[ mapZ->seekMat(0) ]);
        cv::Mat *intensity = reinterpret_cast<cv::Mat*>(mapI->get_mdata()[ mapI->seekMat(0) ]);

        ito::float32 *xRow, *yRow, *zRow, *iRow;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
        pcl::PointXYZI point;
        ito::PCLPointCloud pointCloud;

        width = mapZ->getSize(1);
        height = mapZ->getSize(0);

        if (deleteNaN)
        {
            pointCloud = ito::PCLPointCloud(ito::pclXYZI);
            cloud = pointCloud.toPointXYZI();
            pointCloud.reserve(width*height);

            size_t counter = 0;

            for (int i = 0; i < x->rows; i++)
            {
                xRow = x->ptr<ito::float32>(i);
                yRow = y->ptr<ito::float32>(i);
                zRow = z->ptr<ito::float32>(i);
                iRow = intensity->ptr<ito::float32>(i);

                for (int j = 0; j < x->cols; j++)
                {
                    if (!(pcl_isnan(zRow[j]) || pcl_isnan(yRow[j]) || pcl_isnan(xRow[j])))
                    {
                        point.x = xRow[j];
                        point.y = yRow[j];
                        point.z = zRow[j];
                        point.intensity = iRow[j];
                        (*cloud).push_back(point);
                        counter++;
                    }
                }
            }

            cloud->is_dense = false;
            cloud->resize(counter);
        }
        else
        {
            pointCloud = ito::PCLPointCloud(width, height, ito::pclXYZI, ito::PCLPoint(point));

            cloud = pointCloud.toPointXYZI();

            size_t counter = 0;

            for (int i = 0; i < x->rows; i++)
            {
                xRow = x->ptr<ito::float32>(i);
                yRow = y->ptr<ito::float32>(i);
                zRow = z->ptr<ito::float32>(i);
                iRow = intensity->ptr<ito::float32>(i);

                for (int j = 0; j < x->cols; j++)
                {
                    point.x = xRow[j];
                    point.y = yRow[j];
                    point.z = zRow[j];
                    point.intensity = iRow[j];

                    if (!pcl_isfinite(point.z) || !pcl_isfinite(point.x) || !pcl_isfinite(point.y))
                    {
                        isDense = false;
                    }

                    //cloud->at(j,i) = point;
                    cloud->at(i * width + j) = point;
                    counter++;
                }
            }

            cloud->is_dense = isDense;
        }

        out = pointCloud;
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal pointCloudFromDisparity(const DataObject* mapDisp, PCLPointCloud &out, bool deleteNaN /*= false*/)
{
    return pointCloudFromDisparityI(mapDisp, NULL, out, deleteNaN);
}


//------------------------------------------------------------------------------------------------------------------------------
template<typename _TpM> void fromDataObj(const cv::Mat *mapDisp, const ito::float32 firstX, const ito::float32 stepX, const ito::float32 firstY, const ito::float32 stepY, const bool deleteNaN, ito::PCLPointCloud &out, bool &isDense)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointXYZ point;
            
    int width = mapDisp->cols;
    int height = mapDisp->rows;

    if (deleteNaN)
    {
        out = ito::PCLPointCloud(ito::pclXYZ);
        cloud = out.toPointXYZ();
        out.reserve(width * height);
        size_t counter = 0;

        for (int i = 0; i < height; i++)
        {
            _TpM *zRow = (_TpM*)mapDisp->ptr<_TpM>(i);

            for (int j = 0; j < width; j++)
            {
                if (!(pcl_isnan(zRow[j])))
                {
                    point.x = firstX + j * stepX;
                    point.y = firstY + i * stepY;
                    point.z = zRow[j];
                    (*cloud).push_back(point);
                    counter++;
                }
            }
        }

        cloud->is_dense = false;
        cloud->resize(counter);
    }
    else
    {
        out = ito::PCLPointCloud(width, height, ito::pclXYZ, ito::PCLPoint(point));
        cloud = out.toPointXYZ();

        #if (USEOMP)
        #pragma omp parallel num_threads(NTHREADS)
        {
        #pragma omp for schedule(guided)
        #endif
        for (int i = 0; i < height; i++)
        {
            _TpM *zRow = (_TpM*)mapDisp->ptr<ito::float32>(i);

            for (int j = 0; j < width; j++)
            {
                point.x = firstX + j * stepX;
                point.y = firstY + i * stepY;
                point.z = zRow[j];

                if (!pcl_isfinite(point.z))
                {
                    isDense = false;
                }

                //cloud->at(j,i) = point;
                cloud->at(i * width + j) = point;
//                counter++;
            }
        }
        #if (USEOMP)
        }
        #endif

        cloud->is_dense = isDense;
    }
}

//------------------------------------------------------------------------------------------------------------------------------
template<typename _TpM> ito::RetVal fromDataObj(const cv::Mat *mapDisp, const ito::DataObject *mapI, const ito::float32 firstX, const ito::float32 stepX, const ito::float32 firstY, const ito::float32 stepY, const bool deleteNaN, ito::PCLPointCloud &out, bool &isDense)
{
    cv::Mat *intensity = reinterpret_cast<cv::Mat*>(mapI->get_mdata()[ mapI->seekMat(0) ]);

    switch(mapI->getType())
    {
        case ito::tUInt8:
        {
            fromDataObj<_TpM, ito::uint8>(mapDisp, intensity, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
        }
        break;
        case ito::tInt8:
        {
            fromDataObj<_TpM, ito::int8>(mapDisp, intensity, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
        }
        break;
        case ito::tUInt16:
        {
            fromDataObj<_TpM, ito::uint16>(mapDisp, intensity, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
        }
        break;
        case ito::tInt16:
        {
            fromDataObj<_TpM, ito::int16>(mapDisp, intensity, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
        }
        break;
        case ito::tUInt32:
        {
            fromDataObj<_TpM, ito::uint32>(mapDisp, intensity, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
        }
        break;
        case ito::tInt32:
        {
            fromDataObj<_TpM, ito::int32>(mapDisp, intensity, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
        }
        break;
        case ito::tFloat32:
        {
            fromDataObj<_TpM, ito::float32>(mapDisp, intensity, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
        }
        break;
        case ito::tFloat64:
        {
            fromDataObj<_TpM, ito::float64>(mapDisp, intensity, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
        }
        break;
        default:
            return ito::RetVal(ito::retError, 0, "Unknown type or type not implemented");
    }
    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------
template<typename _TpM, typename _TpI> void fromDataObj(const cv::Mat *mapDisp, const cv::Mat *mapInt, const ito::float32 firstX, const ito::float32 stepX, const ito::float32 firstY, const ito::float32 stepY, const bool deleteNaN, ito::PCLPointCloud &out, bool &isDense)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    pcl::PointXYZI point;
            
    int width = mapDisp->cols;
    int height = mapDisp->rows;

    if (deleteNaN)
    {
        out = ito::PCLPointCloud(ito::pclXYZI);
        cloud = out.toPointXYZI();
        out.reserve(width*height);
        size_t counter = 0;

        for (int i = 0; i < height; i++)
        {
            _TpM *zRow = (_TpM*)mapDisp->ptr<_TpM>(i);
            _TpI *iRow = (_TpI*)mapInt->ptr<_TpI>(i);

            for (int j = 0; j < width; j++)
            {
                if (!(pcl_isnan(zRow[j])))
                {
                    point.x = firstX + j * stepX;
                    point.y = firstY + i * stepY;
                    point.z = zRow[j];
                    point.intensity = iRow[j];
                    (*cloud).push_back(point);
                    counter++;
                }
            }
        }

        cloud->is_dense = false;
        cloud->resize(counter);
    }
    else
    {
        out = ito::PCLPointCloud(width, height, ito::pclXYZI, ito::PCLPoint(point));
        cloud = out.toPointXYZI();

        #if (USEOMP)
        #pragma omp parallel num_threads(NTHREADS)
        {
        #pragma omp for schedule(guided)
        #endif
        for (int i = 0; i < height; i++)
        {
            _TpM *zRow = (_TpM*)mapDisp->ptr<_TpM>(i);
            _TpI *iRow = (_TpI*)mapInt->ptr<_TpI>(i);

            for (int j = 0; j < width; j++)
            {
                point.x = firstX + j * stepX;
                point.y = firstY + i * stepY;
                point.z = zRow[j];
                point.intensity = iRow[j];

                if (!pcl_isfinite(point.z))
                {
                    isDense = false;
                }

                //cloud->at(j,i) = point;
                cloud->at(i * width + j) = point;
            }
        }
        #if (USEOMP)
        }
        #endif

        cloud->is_dense = isDense;
    }
}

//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal pointCloudFromDisparityI(const DataObject* mapDisp, const DataObject *mapI, PCLPointCloud &out, bool deleteNaN /*= false*/)
{
    RetVal retval = retOk;
    float firstX = 0.0;
    float stepX = 1.0;
    float firstY = 0.0;
    float stepY = 1.0;
    bool isDense = true;

    retval += ito::dObjHelper::verify2DDataObject(mapDisp, "disparityMap", 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 8, ito::tInt8, ito::tUInt8, ito::tInt16, ito::tUInt16, 
        ito::tInt32, ito::tUInt32, ito::tFloat32, ito::tFloat64);
    if (mapI)
    {
        retval += ito::dObjHelper::verify2DDataObject(mapI, "intensityMap", 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 8, ito::tInt8, ito::tUInt8, ito::tInt16, ito::tUInt16, 
        ito::tInt32, ito::tUInt32, ito::tFloat32, ito::tFloat64);

        if (mapI->getSize(0) != mapDisp->getSize(0) || mapI->getSize(1) != mapDisp->getSize(1))
        {
            retval += ito::RetVal(ito::retError, 0, "disparityMap and intensityMap must have the same size");
        }
    }

    if (retval == retOk)
    {
        bool checkScale = true;
        firstX = cv::saturate_cast<float>(mapDisp->getPixToPhys(mapDisp->getDims()-1, 0, checkScale));
        stepX = cv::saturate_cast<float>(mapDisp->getPixToPhys(mapDisp->getDims()-1, 1, checkScale)) - firstX;
        firstY = cv::saturate_cast<float>(mapDisp->getPixToPhys(mapDisp->getDims()-2, 0, checkScale));
        stepY = cv::saturate_cast<float>(mapDisp->getPixToPhys(mapDisp->getDims()-2, 1, checkScale)) - firstY;
    }

    if (retval == ito::retOk)
    {
//        uint32_t width, height;
//        ito::float32 *zRow;
//        ito::float32 *iRow;

        cv::Mat *z = reinterpret_cast<cv::Mat*>(mapDisp->get_mdata()[ mapDisp->seekMat(0) ]);

        if (mapI == NULL)
        {
            switch(mapDisp->getType())
            {
                case ito::tUInt8:
                {
                    fromDataObj<ito::uint8>(z, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                case ito::tInt8:
                {
                    fromDataObj<ito::int8>(z, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                case ito::tUInt16:
                {
                    fromDataObj<ito::uint16>(z, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                case ito::tInt16:
                {
                    fromDataObj<ito::int16>(z, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                case ito::tUInt32:
                {
                    fromDataObj<ito::uint32>(z, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                case ito::tInt32:
                {
                    fromDataObj<ito::int32>(z, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                case ito::tFloat32:
                {
                    fromDataObj<ito::float32>(z, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                case ito::tFloat64:
                {
                    fromDataObj<ito::float64>(z, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                default:
                    return ito::RetVal(ito::retError, 0, "Unknown type or type not implemented");
            }
        }
        else
        {
            switch(mapDisp->getType())
            {
                case ito::tUInt8:
                {
                    return fromDataObj<ito::uint8>(z, mapI, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                case ito::tInt8:
                {
                    return fromDataObj<ito::int8>(z, mapI, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                case ito::tUInt16:
                {
                    return fromDataObj<ito::uint16>(z, mapI, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                case ito::tInt16:
                {
                    return fromDataObj<ito::int16>(z, mapI, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                case ito::tUInt32:
                {
                    return fromDataObj<ito::uint32>(z, mapI, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                case ito::tInt32:
                {
                    return fromDataObj<ito::int32>(z, mapI, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                case ito::tFloat32:
                {
                    return fromDataObj<ito::float32>(z, mapI, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                case ito::tFloat64:
                {
                    return fromDataObj<ito::float64>(z, mapI, firstX, stepX, firstY, stepY, deleteNaN, out, isDense);
                }
                break;
                default:
                    return ito::RetVal(ito::retError, 0, "Unknown type or type not implemented");
            }
        }
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal pointCloudFromDisparityRGBA(const DataObject* mapDisp, const DataObject *mapColor, PCLPointCloud &out, bool deleteNaN /*= false*/)
{
    RetVal retval = retOk;
    float firstX = 0.0;
    float stepX = 1.0;
    float firstY = 0.0;
    float stepY = 1.0;
    bool isDense = true;

    retval += ito::dObjHelper::verify2DDataObject(mapDisp, "disparityMap", 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 1, ito::tFloat32);
    if (mapColor)
    {
        retval += ito::dObjHelper::verify2DDataObject(mapColor, "colorMap", 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 1, ito::tRGBA32);

        if (mapColor->getSize(0) != mapDisp->getSize(0) || mapColor->getSize(1) != mapDisp->getSize(1))
        {
            retval += ito::RetVal(ito::retError, 0, "disparityMap and colorMap must have the same size");
        }
    }

    if (retval == retOk)
    {
        bool checkScale = true;
        firstX = cv::saturate_cast<float>(mapDisp->getPixToPhys(mapDisp->getDims()-1, 0, checkScale));
        stepX = cv::saturate_cast<float>(mapDisp->getPixToPhys(mapDisp->getDims()-1, 1, checkScale)) - firstX;
        firstY = cv::saturate_cast<float>(mapDisp->getPixToPhys(mapDisp->getDims()-2, 0, checkScale));
        stepY = cv::saturate_cast<float>(mapDisp->getPixToPhys(mapDisp->getDims()-2, 1, checkScale)) - firstY;
    }

    if (retval == ito::retOk)
    {
        uint32_t width, height;
        ito::float32 *zRow;
        ito::Rgba32 *cRow;

        cv::Mat *z = reinterpret_cast<cv::Mat*>(mapDisp->get_mdata()[ mapDisp->seekMat(0) ]);

        if (mapColor == NULL)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
            pcl::PointXYZ point;
            
            width = mapDisp->getSize(1);
            height = mapDisp->getSize(0);

            if (deleteNaN)
            {
                out = ito::PCLPointCloud(ito::pclXYZ);
                cloud = out.toPointXYZ();
                out.reserve(width*height);
                size_t counter = 0;

                for (int i = 0; i < z->rows; i++)
                {
                    zRow = z->ptr<ito::float32>(i);

                    for (int j = 0; j < z->cols; j++)
                    {
                        if (!(pcl_isnan(zRow[j])))
                        {
                            point.x = firstX + j * stepX;
                            point.y = firstY + i * stepY;
                            point.z = zRow[j];
                            (*cloud).push_back(point);
                            counter++;
                        }
                    }
                }

                cloud->is_dense = false;
                cloud->resize(counter);
            }
            else
            {
                out = ito::PCLPointCloud(width, height, ito::pclXYZ, ito::PCLPoint(point));
                cloud = out.toPointXYZ();
                size_t counter = 0;

                for (int i = 0; i < z->rows; i++)
                {
                    zRow = z->ptr<ito::float32>(i);

                    for (int j = 0; j < z->cols; j++)
                    {
                        point.x = firstX + j * stepX;
                        point.y = firstY + i * stepY;
                        point.z = zRow[j];

                        if (!pcl_isfinite(point.z))
                        {
                            isDense = false;
                        }

                        //cloud->at(j,i) = point;
                        cloud->at(i * width + j) = point;
                        counter++;
                    }
                }

                cloud->is_dense = isDense;
            }
        }
        else
        {
            cv::Mat *color = reinterpret_cast<cv::Mat*>(mapColor->get_mdata()[ mapColor->seekMat(0) ]);

            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
            pcl::PointXYZRGBA point;
            
            width = mapDisp->getSize(1);
            height = mapDisp->getSize(0);

            if (deleteNaN)
            {
                out = ito::PCLPointCloud(ito::pclXYZRGBA);
                cloud = out.toPointXYZRGBA();
                out.reserve(width*height);
                size_t counter = 0;

                for (int i = 0; i < z->rows; i++)
                {
                    zRow = z->ptr<ito::float32>(i);
                    cRow = color->ptr<ito::Rgba32>(i);

                    for (int j = 0; j < z->cols; j++)
                    {
                        if (!(pcl_isnan(zRow[j])))
                        {
                            point.x = firstX + j * stepX;
                            point.y = firstY + i * stepY;
                            point.z = zRow[j];
                            point.rgba = cRow[j].rgba;
                            (*cloud).push_back(point);
                            counter++;
                        }
                    }
                }

                cloud->is_dense = false;
                cloud->resize(counter);
            }
            else
            {
                out = ito::PCLPointCloud(width, height, ito::pclXYZRGBA, ito::PCLPoint(point));
                cloud = out.toPointXYZRGBA();
                size_t counter = 0;

                for (int i = 0; i < z->rows; i++)
                {
                    zRow = z->ptr<ito::float32>(i);
                    cRow = color->ptr<ito::Rgba32>(i);

                    for (int j = 0; j < z->cols; j++)
                    {
                        point.x = firstX + j * stepX;
                        point.y = firstY + i * stepY;
                        point.z = zRow[j];
                        point.rgba = cRow[j].rgba;

                        if (!pcl_isfinite(point.z))
                        {
                            isDense = false;
                        }

                        //cloud->at(j,i) = point;
                        cloud->at(i * width + j) = point;
                        counter++;
                    }
                }

                cloud->is_dense = isDense;
            }
        }
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal pointCloudToDObj(const PCLPointCloud *pc, DataObject &out)
{
    if (pc == NULL)
    {
        return RetVal(retError, 0, "PCLPointCloud is NULL");
    }
    
    if (pc->getType() == ito::pclXYZ)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = pc->toPointXYZ();
        pcl::PointCloud<pcl::PointXYZ>::VectorType points = cloud->points;
        out = DataObject(3, (int)cloud->size(), ito::tFloat32);
        pcl::PointXYZ *point;

        ito::float32 *xRow = (ito::float32*)out.rowPtr(0, 0);
        ito::float32 *yRow = (ito::float32*)out.rowPtr(0, 1);
        ito::float32 *zRow = (ito::float32*)out.rowPtr(0, 2);

        for (size_t i = 0; i < points.size(); i++)
        {
            point = &(points[i]);
            xRow[i] = point->x;
            yRow[i] = point->y;
            zRow[i] = point->z;
        }
    }
    else if (pc->getType() == ito::pclXYZI)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = pc->toPointXYZI();
        pcl::PointCloud<pcl::PointXYZI>::VectorType points = cloud->points;
        out = DataObject(4, (int)cloud->size(), ito::tFloat32);
        pcl::PointXYZI *point;

        ito::float32 *xRow = (ito::float32*)out.rowPtr(0, 0);
        ito::float32 *yRow = (ito::float32*)out.rowPtr(0, 1);
        ito::float32 *zRow = (ito::float32*)out.rowPtr(0, 2);
        ito::float32 *iRow = (ito::float32*)out.rowPtr(0, 3);

        for (size_t i = 0; i < points.size(); i++)
        {
            point = &(points[i]);
            xRow[i] = point->x;
            yRow[i] = point->y;
            zRow[i] = point->z;
            iRow[i] = point->intensity;
        }
    }
    else if (pc->getType() == ito::pclXYZRGBA)
    {
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud = pc->toPointXYZRGBA();
        pcl::PointCloud<pcl::PointXYZRGBA>::VectorType points = cloud->points;
        out = DataObject(7, (int)cloud->size(), ito::tFloat32);
        pcl::PointXYZRGBA *point;

        ito::float32 *xRow = (ito::float32*)out.rowPtr(0, 0);
        ito::float32 *yRow = (ito::float32*)out.rowPtr(0, 1);
        ito::float32 *zRow = (ito::float32*)out.rowPtr(0, 2);
        ito::float32 *rRow = (ito::float32*)out.rowPtr(0, 3);
        ito::float32 *gRow = (ito::float32*)out.rowPtr(0, 4);
        ito::float32 *bRow = (ito::float32*)out.rowPtr(0, 5);
        ito::float32 *aRow = (ito::float32*)out.rowPtr(0, 6);

        for (size_t i = 0; i < points.size(); i++)
        {
            point = &(points[i]);
            xRow[i] = point->x;
            yRow[i] = point->y;
            zRow[i] = point->z;
            rRow[i] = point->r;
            gRow[i] = point->g;
            bRow[i] = point->b;
            aRow[i] = point->a;
        }
    }
    else if (pc->getType() == ito::pclXYZNormal)
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud = pc->toPointXYZNormal();
        pcl::PointCloud<pcl::PointNormal>::VectorType points = cloud->points;
        out = DataObject(7, (int)cloud->size(), ito::tFloat32);
        pcl::PointNormal *point;

        ito::float32 *xRow = (ito::float32*)out.rowPtr(0, 0);
        ito::float32 *yRow = (ito::float32*)out.rowPtr(0, 1);
        ito::float32 *zRow = (ito::float32*)out.rowPtr(0, 2);
        ito::float32 *nxRow = (ito::float32*)out.rowPtr(0, 3);
        ito::float32 *nyRow = (ito::float32*)out.rowPtr(0, 4);
        ito::float32 *nzRow = (ito::float32*)out.rowPtr(0, 5);
        ito::float32 *ncurvRow = (ito::float32*)out.rowPtr(0, 6);

        for (size_t i = 0; i < points.size(); i++)
        {
            point = &(points[i]);
            xRow[i] = point->x;
            yRow[i] = point->y;
            zRow[i] = point->z;
            nxRow[i] = point->normal_x;
            nyRow[i] = point->normal_y;
            nzRow[i] = point->normal_z;
            ncurvRow[i] = point->curvature;
        }
    }
    else if (pc->getType() == ito::pclXYZINormal)
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud = pc->toPointXYZINormal();
        pcl::PointCloud<pcl::PointXYZINormal>::VectorType points = cloud->points;
        out = DataObject(8, (int)cloud->size(), ito::tFloat32);
        pcl::PointXYZINormal *point;

        ito::float32 *xRow = (ito::float32*)out.rowPtr(0, 0);
        ito::float32 *yRow = (ito::float32*)out.rowPtr(0, 1);
        ito::float32 *zRow = (ito::float32*)out.rowPtr(0, 2);
        ito::float32 *nxRow = (ito::float32*)out.rowPtr(0, 3);
        ito::float32 *nyRow = (ito::float32*)out.rowPtr(0, 4);
        ito::float32 *nzRow = (ito::float32*)out.rowPtr(0, 5);
        ito::float32 *ncurvRow = (ito::float32*)out.rowPtr(0, 6);
        ito::float32 *iRow = (ito::float32*)out.rowPtr(0, 7);

        for (size_t i = 0; i < points.size(); i++)
        {
            point = &(points[i]);
            xRow[i] = point->x;
            yRow[i] = point->y;
            zRow[i] = point->z;
            nxRow[i] = point->normal_x;
            nyRow[i] = point->normal_y;
            nzRow[i] = point->normal_z;
            ncurvRow[i] = point->curvature;
            iRow[i] = point->intensity;
            
        }
    }
    else if (pc->getType() == ito::pclXYZRGBNormal)
    {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud = pc->toPointXYZRGBNormal();
        pcl::PointCloud<pcl::PointXYZRGBNormal>::VectorType points = cloud->points;
        out = DataObject(11, (int)cloud->size(), ito::tFloat32);
        pcl::PointXYZRGBNormal *point;

        ito::float32 *xRow = (ito::float32*)out.rowPtr(0, 0);
        ito::float32 *yRow = (ito::float32*)out.rowPtr(0, 1);
        ito::float32 *zRow = (ito::float32*)out.rowPtr(0, 2);
        ito::float32 *nxRow = (ito::float32*)out.rowPtr(0, 3);
        ito::float32 *nyRow = (ito::float32*)out.rowPtr(0, 4);
        ito::float32 *nzRow = (ito::float32*)out.rowPtr(0, 5);
        ito::float32 *ncurvRow = (ito::float32*)out.rowPtr(0, 6);
        ito::float32 *rRow = (ito::float32*)out.rowPtr(0, 7);
        ito::float32 *gRow = (ito::float32*)out.rowPtr(0, 8);
        ito::float32 *bRow = (ito::float32*)out.rowPtr(0, 9);
        ito::float32 *aRow = (ito::float32*)out.rowPtr(0, 10);

        for (size_t i = 0; i < points.size(); i++)
        {
            point = &(points[i]);
            xRow[i] = point->x;
            yRow[i] = point->y;
            zRow[i] = point->z;
            nxRow[i] = point->normal_x;
            nyRow[i] = point->normal_y;
            nzRow[i] = point->normal_z;
            ncurvRow[i] = point->curvature;
            rRow[i] = point->r;
            gRow[i] = point->g;
            bRow[i] = point->b;
            aRow[i] = point->a;
            
        }
    }
    else if (pc->getType() == ito::pclInvalid)
    {
        out = DataObject();
    }
    else
    {
        return RetVal(retError, 0, "point clouds with RGB content cannot be converted to data object");
    }

    return retOk;
}

//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal dataObj4x4ToEigenAffine3f(const DataObject *in, Eigen::Affine3f &out)
{

    RetVal retval;
    ito::DataObject* in2 = NULL;

    if (in)
    {
        retval += ito::dObjHelper::verify2DDataObject(in, "transform", 4, 4, 4, 4, 7, ito::tInt8, ito::tUInt8, ito::tInt16, ito::tUInt16, ito::tInt32, ito::tUInt32, ito::tFloat32);
        if (!retval.containsError())
        {
            const ito::float32* r0 = (const ito::float32*)in->rowPtr(0, 0);
            const ito::float32* r1 = (const ito::float32*)in->rowPtr(0, 1);
            const ito::float32* r2 = (const ito::float32*)in->rowPtr(0, 2);
            const ito::float32* r3 = (const ito::float32*)in->rowPtr(0, 3);

            if (in->getType() != ito::tFloat32)
            {
                in2 = new ito::DataObject();
                retval += in->convertTo(*in2, ito::tFloat32);

                if (retval == retOk)
                {
                    r0 = (const ito::float32*)in2->rowPtr(0, 0);
                    r1 = (const ito::float32*)in2->rowPtr(0, 1);
                    r2 = (const ito::float32*)in2->rowPtr(0, 2);
                    r3 = (const ito::float32*)in2->rowPtr(0, 3);
                }
            }

            if (!retval.containsError())
            {
                Eigen::Matrix4f homMat;

                homMat << r0[0], r0[1], r0[2], r0[3],
                          r1[0], r1[1], r1[2], r1[3],
                          r2[0], r2[1], r2[2], r2[3],
                          r3[0], r3[1], r3[2], r3[3];

                out = homMat;
            }
        }
    }
    else
    {
        retval += RetVal(retError,0,"dataObject must not be NULL");
    }

    if (in2)
    {
        delete in2;
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal eigenAffine3fToDataObj4x4(const Eigen::Affine3f *in, DataObject &out)
{
    RetVal retval;
    out = DataObject();
    retval += out.eye(4, ito::tFloat32);

    if (in->Options & Eigen::ColMajor)
    {
        retval += ito::RetVal(ito::retError, 0, "affine3f object must be rowMajor");
    }

    if (!retval.containsError())
    {
        ito::float32 *r0 = (ito::float32*)out.rowPtr(0, 0);
        ito::float32 *r1 = (ito::float32*)out.rowPtr(0, 1);
        ito::float32 *r2 = (ito::float32*)out.rowPtr(0, 2);

        if (in->Mode == Eigen::AffineCompact) //in is a 3x4 matrix
        {
            const ito::float32 *d = in->data();
            r0[0] = d[0];
            r1[0] = d[1];
            r2[0] = d[2];

            r0[1] = d[3];
            r1[1] = d[4];
            r2[1] = d[5];

            r0[2] = d[6];
            r1[2] = d[7];
            r2[2] = d[8];

            r0[3] = d[9];
            r1[3] = d[10];
            r2[3] = d[11];
        }
        else
        {
            retval += RetVal(retError, 0, "Eigen transform object must have a type AffineCompact.");
        }
    }

    return retval;
}



////------------------------------------------------------------------------------------------------------------------------------
//ito::RetVal writeBinary(const std::string &filename, const ito::PCLPointCloud &cloud)
//{
//    pcl::PCDWriter w;
//    int ret;
//
//    switch(cloud.getType())
//    {
//    case ito::pclXYZ:
//        ret = pcl::io::savePCDFile<pcl::PointXYZ>(filename, *cloud.toPointXYZ(), true);
//        break;
//    case ito::pclXYZI:
//        ret = pcl::io::savePCDFile<pcl::PointXYZI>(filename, *cloud.toPointXYZI(), true);
//        break;
//    case ito::pclXYZRGBA:
//        ret = pcl::io::savePCDFile<pcl::PointXYZRGBA>(filename, *cloud.toPointXYZRGBA(), true);
//        break;
//    case ito::pclXYZNormal:
//        ret = pcl::io::savePCDFile<pcl::PointNormal>(filename, *cloud.toPointXYZNormal(), true);
//        break;
//    case ito::pclXYZINormal:
//        ret = pcl::io::savePCDFile<pcl::PointXYZINormal>(filename, *cloud.toPointXYZINormal(), true);
//        break;
//    case ito::pclXYZRGBNormal:
//        ret = pcl::io::savePCDFile<pcl::PointXYZRGBNormal>(filename, *cloud.toPointXYZRGBNormal(), true);
//        break;
//    default:
//        return RetVal(retError,0,"invalid point cloud");
//    }
//
//    return ito::retOk;
//}
//
////------------------------------------------------------------------------------------------------------------------------------
//ito::RetVal readBinary(const std::string &filename, ito::PCLPointCloud &cloud)
//{
//    return ito::retOk;
//}

} //end namespace pclHelper
} //end namespace ito
