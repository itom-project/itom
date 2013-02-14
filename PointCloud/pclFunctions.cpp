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

#include <pcl/io/pcd_io.h>

namespace ito 
{

namespace pclHelper
{

//------------------------------------------------------------------------------------------------------------------------------
void PointXYZRGBtoXYZRGBA (pcl::PointXYZRGB& in, pcl::PointXYZRGBA&  out)
{
    out.x = in.x; out.y = in.y; out.z = in.z;
    out.r = in.r; out.g = in.g; out.b = in.b; out.PCLALPHA = 255;
}

//------------------------------------------------------------------------------------------------------------------------------
void PointXYZRGBAtoXYZRGB (pcl::PointXYZRGBA& in, pcl::PointXYZRGB&  out)
{
    out.x = in.x; out.y = in.y; out.z = in.z;
    out.r = in.r; out.g = in.g; out.b = in.b;
}

//------------------------------------------------------------------------------------------------------------------------------
void PointCloudXYZRGBtoXYZRGBA( pcl::PointCloud<pcl::PointXYZRGB>& in, pcl::PointCloud<pcl::PointXYZRGBA>& out)
{
    out.width = in.width;
    out.height = in.height;

    for(size_t i = 0; i < in.points.size() ; i++)
    {
        pcl::PointXYZRGBA p;
        ito::pclHelper::PointXYZRGBtoXYZRGBA (in.points[i], p);
        out.points.push_back (p);
    }

}

//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal pointCloudFromXYZ(const DataObject* mapX, const DataObject* mapY, const DataObject* mapZ, PCLPointCloud &out, bool deleteNaN /*= false*/)
{
    RetVal retval = retOk;
    bool isDense = true;

    retval += ito::dObjHelper::verify2DDataObject(mapZ, "Z", 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 1, ito::tFloat32);
    retval += ito::dObjHelper::verify2DDataObject(mapX, "X", mapZ->getSize(0,true), mapZ->getSize(0,true), mapZ->getSize(1,true), mapZ->getSize(1,true), 1, ito::tFloat32);
    retval += ito::dObjHelper::verify2DDataObject(mapY, "Y", mapZ->getSize(0,true), mapZ->getSize(0,true), mapZ->getSize(1,true), mapZ->getSize(1,true), 1, ito::tFloat32);

    if(!retval.containsError())
    {
        uint32_t width, height;
        cv::Mat *x = reinterpret_cast<cv::Mat*>(mapX->get_mdata()[ mapX->seekMat(0) ]);
        cv::Mat *y = reinterpret_cast<cv::Mat*>(mapY->get_mdata()[ mapY->seekMat(0) ]);
        cv::Mat *z = reinterpret_cast<cv::Mat*>(mapZ->get_mdata()[ mapZ->seekMat(0) ]);

        ito::float32 *xRow, *yRow, *zRow;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        pcl::PointXYZ point;
        ito::PCLPointCloud pointCloud;

        width = mapZ->getSize(1,true);
        height = mapZ->getSize(0,true);

        if(deleteNaN)
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
                    if(!(pcl_isnan(zRow[j]) || pcl_isnan(yRow[j]) || pcl_isnan(xRow[j])))
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

            pointCloud = ito::PCLPointCloud(width, height, ito::pclXYZ, ito::PCLPoint(point) );

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

                    if(!pcl_isfinite(point.z) || !pcl_isfinite(point.x) || !pcl_isfinite(point.y))
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
    retval += ito::dObjHelper::verify2DDataObject(mapX, "X", mapZ->getSize(0,true), mapZ->getSize(0,true), mapZ->getSize(1,true), mapZ->getSize(1,true), 1, ito::tFloat32);
    retval += ito::dObjHelper::verify2DDataObject(mapY, "Y", mapZ->getSize(0,true), mapZ->getSize(0,true), mapZ->getSize(1,true), mapZ->getSize(1,true), 1, ito::tFloat32);
    retval += ito::dObjHelper::verify2DDataObject(mapI, "I", mapZ->getSize(0,true), mapZ->getSize(0,true), mapZ->getSize(1,true), mapZ->getSize(1,true), 1, ito::tFloat32);

    if(!retval.containsError())
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

        width = mapZ->getSize(1,true);
        height = mapZ->getSize(0,true);

        if(deleteNaN)
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
                    if(!(pcl_isnan(zRow[j]) || pcl_isnan(yRow[j]) || pcl_isnan(xRow[j])))
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

            pointCloud = ito::PCLPointCloud(width, height, ito::pclXYZI, ito::PCLPoint(point) );

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

                    if(!pcl_isfinite(point.z) || !pcl_isfinite(point.x) || !pcl_isfinite(point.y))
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
ito::RetVal pointCloudFromDisparityI(const DataObject* mapDisp, const DataObject *mapI, PCLPointCloud &out, bool deleteNaN /*= false*/)
{
    RetVal retval = retOk;
    float firstX = 0.0;
    float stepX = 1.0;
    float firstY = 0.0;
    float stepY = 1.0;
    bool isDense = true;

    retval += ito::dObjHelper::verify2DDataObject(mapDisp, "disparityMap", 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 1, ito::tFloat32);
    if(mapI)
    {
        retval += ito::dObjHelper::verify2DDataObject(mapI, "intensityMap", 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 1, ito::tFloat32);

        if( mapI->getSize(0) != mapDisp->getSize(0) || mapI->getSize(1) != mapDisp->getSize(1) )
        {
            retval += ito::RetVal(ito::retError,0,"disparityMap and intensityMap must have the same size");
        }
    }

    if(retval == retOk)
    {
        bool checkScale = true;
        firstX = cv::saturate_cast<float>(mapDisp->getPixToPhys(mapDisp->getDims()-1, 0, checkScale, false));
        stepX = cv::saturate_cast<float>(mapDisp->getPixToPhys(mapDisp->getDims()-1, 1, checkScale, false)) - firstX;
        firstY = cv::saturate_cast<float>(mapDisp->getPixToPhys(mapDisp->getDims()-2, 0, checkScale, false));
        stepY = cv::saturate_cast<float>(mapDisp->getPixToPhys(mapDisp->getDims()-2, 1, checkScale, false)) - firstY;
    }

    if(retval == ito::retOk)
    {
        uint32_t width, height;
        ito::float32 *zRow;
        ito::float32 *iRow;

        cv::Mat *z = reinterpret_cast<cv::Mat*>(mapDisp->get_mdata()[ mapDisp->seekMat(0) ]);

        if(mapI == NULL)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
            pcl::PointXYZ point;
            
            width = mapDisp->getSize(1,true);
            height = mapDisp->getSize(0,true);

            if(deleteNaN)
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
                        if(!(pcl_isnan(zRow[j])))
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
                out = ito::PCLPointCloud(width, height, ito::pclXYZ, ito::PCLPoint(point) );
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

                        if(!pcl_isfinite(point.z))
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
            cv::Mat *intensity = reinterpret_cast<cv::Mat*>(mapI->get_mdata()[ mapI->seekMat(0) ]);

            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
            pcl::PointXYZI point;
            
            width = mapDisp->getSize(1,true);
            height = mapDisp->getSize(0,true);

            if(deleteNaN)
            {
                out = ito::PCLPointCloud(ito::pclXYZI);
                cloud = out.toPointXYZI();
                out.reserve(width*height);
                size_t counter = 0;

                for (int i = 0; i < z->rows; i++)
                {
                    zRow = z->ptr<ito::float32>(i);
                    iRow = intensity->ptr<ito::float32>(i);

                    for (int j = 0; j < z->cols; j++)
                    {
                        if(!(pcl_isnan(zRow[j])))
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
                out = ito::PCLPointCloud(width, height, ito::pclXYZI, ito::PCLPoint(point) );
                cloud = out.toPointXYZI();
                size_t counter = 0;

                for (int i = 0; i < z->rows; i++)
                {
                    zRow = z->ptr<ito::float32>(i);
                    iRow = intensity->ptr<ito::float32>(i);

                    for (int j = 0; j < z->cols; j++)
                    {
                        point.x = firstX + j * stepX;
                        point.y = firstY + i * stepY;
                        point.z = zRow[j];
                        point.intensity = iRow[j];

                        if(!pcl_isfinite(point.z))
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
    if(pc == NULL)
    {
        return RetVal(retError,0,"PCLPointCloud is NULL");
    }
    
    if(pc->getType() == ito::pclXYZ)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = pc->toPointXYZ();
        pcl::PointCloud<pcl::PointXYZ>::VectorType points = cloud->points;
        out = DataObject(3, cloud->size(), ito::tFloat32);
        pcl::PointXYZ *point;

        ito::float32 *xRow = (ito::float32*)out.rowPtr(0,0);
        ito::float32 *yRow = (ito::float32*)out.rowPtr(0,1);
        ito::float32 *zRow = (ito::float32*)out.rowPtr(0,2);

        for(size_t i = 0 ; i < points.size() ; i++)
        {
            point = &(points[i]);
            xRow[i] = point->x;
            yRow[i] = point->y;
            zRow[i] = point->z;
        }
    }
    else if(pc->getType() == ito::pclXYZI)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = pc->toPointXYZI();
        pcl::PointCloud<pcl::PointXYZI>::VectorType points = cloud->points;
        out = DataObject(4, cloud->size(), ito::tFloat32);
        pcl::PointXYZI *point;

        ito::float32 *xRow = (ito::float32*)out.rowPtr(0,0);
        ito::float32 *yRow = (ito::float32*)out.rowPtr(0,1);
        ito::float32 *zRow = (ito::float32*)out.rowPtr(0,2);
        ito::float32 *iRow = (ito::float32*)out.rowPtr(0,3);

        for(size_t i = 0 ; i < points.size() ; i++)
        {
            point = &(points[i]);
            xRow[i] = point->x;
            yRow[i] = point->y;
            zRow[i] = point->z;
            iRow[i] = point->intensity;
        }
    }
    else if(pc->getType() == ito::pclXYZNormal)
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud = pc->toPointXYZNormal();
        pcl::PointCloud<pcl::PointNormal>::VectorType points = cloud->points;
        out = DataObject(6, cloud->size(), ito::tFloat32);
        pcl::PointNormal *point;

        ito::float32 *xRow = (ito::float32*)out.rowPtr(0,0);
        ito::float32 *yRow = (ito::float32*)out.rowPtr(0,1);
        ito::float32 *zRow = (ito::float32*)out.rowPtr(0,2);
        ito::float32 *nxRow = (ito::float32*)out.rowPtr(0,3);
        ito::float32 *nyRow = (ito::float32*)out.rowPtr(0,4);
        ito::float32 *nzRow = (ito::float32*)out.rowPtr(0,5);

        for(size_t i = 0 ; i < points.size() ; i++)
        {
            point = &(points[i]);
            xRow[i] = point->x;
            yRow[i] = point->y;
            zRow[i] = point->z;
            nxRow[i] = point->normal_x;
            nyRow[i] = point->normal_y;
            nzRow[i] = point->normal_z;
        }
    }
    else if(pc->getType() == ito::pclXYZINormal)
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud = pc->toPointXYZINormal();
        pcl::PointCloud<pcl::PointXYZINormal>::VectorType points = cloud->points;
        out = DataObject(7, cloud->size(), ito::tFloat32);
        pcl::PointXYZINormal *point;

        ito::float32 *xRow = (ito::float32*)out.rowPtr(0,0);
        ito::float32 *yRow = (ito::float32*)out.rowPtr(0,1);
        ito::float32 *zRow = (ito::float32*)out.rowPtr(0,2);
        ito::float32 *nxRow = (ito::float32*)out.rowPtr(0,3);
        ito::float32 *nyRow = (ito::float32*)out.rowPtr(0,4);
        ito::float32 *nzRow = (ito::float32*)out.rowPtr(0,5);
        ito::float32 *iRow = (ito::float32*)out.rowPtr(0,6);

        for(size_t i = 0 ; i < points.size() ; i++)
        {
            point = &(points[i]);
            xRow[i] = point->x;
            yRow[i] = point->y;
            zRow[i] = point->z;
            nxRow[i] = point->normal_x;
            nyRow[i] = point->normal_y;
            nzRow[i] = point->normal_z;
            iRow[i] = point->intensity;
        }
    }
    else if(pc->getType() == ito::pclInvalid)
    {
        out = DataObject();
    }
    else
    {
        return RetVal(retError,0,"point clouds with RGB content cannot be converted to data object");
    }

    return retOk;
}

//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal dataObj4x4ToEigenAffine3f(const DataObject *in, Eigen::Affine3f &out)
{

    RetVal retval;
    ito::DataObject* in2 = NULL;

    if(in)
    {
        retval += ito::dObjHelper::verify2DDataObject(in, "transform", 4, 4, 4, 4, 7, ito::tInt8, ito::tUInt8, ito::tInt16, ito::tUInt16, ito::tInt32, ito::tUInt32, ito::tFloat32);
        if(!retval.containsError())
        {
            const ito::float32* r0 = (const ito::float32*)in->rowPtr(0,0);
            const ito::float32* r1 = (const ito::float32*)in->rowPtr(0,1);
            const ito::float32* r2 = (const ito::float32*)in->rowPtr(0,2);
            const ito::float32* r3 = (const ito::float32*)in->rowPtr(0,3);

            if(in->getType() != ito::tFloat32)
            {
                in2 = new ito::DataObject();
                retval += in->convertTo(*in2, ito::tFloat32);

                if(retval == retOk)
                {
                    r0 = (const ito::float32*)in2->rowPtr(0,0);
                    r1 = (const ito::float32*)in2->rowPtr(0,1);
                    r2 = (const ito::float32*)in2->rowPtr(0,2);
                    r3 = (const ito::float32*)in2->rowPtr(0,3);
                }
            }

            if(!retval.containsError())
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

    if(in2)
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
    if(!retval.containsError())
    {
        ito::float32 *r0 = (ito::float32*)out.rowPtr(0,0);
        ito::float32 *r1 = (ito::float32*)out.rowPtr(0,1);
        ito::float32 *r2 = (ito::float32*)out.rowPtr(0,2);

        if(in->Mode == Eigen::AffineCompact) //in is a 3x4 matrix
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
            retval += RetVal(retError,0,"Eigen transform object must have a type AffineCompact.");
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