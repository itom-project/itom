/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
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

template<typename _Tp> bool is_finite(const _Tp &value) { return std::isfinite(value); }
template<> bool is_finite<ito::uint8>(const ito::uint8 &/*value*/) { return true; }
template<> bool is_finite<ito::int8>(const ito::int8 &/*value*/) { return true; }
template<> bool is_finite<ito::uint16>(const ito::uint16 &/*value*/) { return true; }
template<> bool is_finite<ito::int16>(const ito::int16 &/*value*/) { return true; }
template<> bool is_finite<ito::uint32>(const ito::uint32 &/*value*/) { return true; }
template<> bool is_finite<ito::int32>(const ito::int32 &/*value*/) { return true; }

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
ito::RetVal POINTCLOUD_EXPORT normalsAtCogFromPolygonMesh(const PCLPolygonMesh &mesh, PCLPointCloud &out, const std::vector<int> &indices /*= std::vector<int>()*/)
{
    ito::RetVal retval;

    if (mesh.valid())
    {
        ito::tPCLPointType t = ito::pclHelper::guessPointType(mesh.polygonMesh()->cloud);
        ito::tPCLPointType t_out = ito::pclInvalid;

        switch (t)
        {
        case ito::pclXYZ:
        case ito::pclXYZNormal:
            t_out = ito::pclXYZNormal;
            break;
        case ito::pclXYZI:
        case ito::pclXYZINormal:
            t_out = ito::pclXYZINormal;
            break;
        case ito::pclXYZRGBA:
        case ito::pclXYZRGBNormal:
            t_out = ito::pclXYZRGBNormal;
            break;
        }

        if (t_out != ito::pclInvalid)
        {
            out = ito::PCLPointCloud(t_out);
            const pcl::Vertices *v;
#if PCL_VERSION_COMPARE(>=,1,12,0)
            const pcl::index_t *v_;
#else
            const uint32_t *v_;
#endif
            Eigen::Vector3f a, b;
            Eigen::Vector3f cog;           //center of gravity of all mesh segments that should be covered
            Eigen::Vector3f normal;    //normal vector (given in coordinate frame phi) of each center of gravity
            bool indexed_mode = (indices.size() > 0);
            int next_index = 0;
            int index;
            bool has_next;
            int count = 0;
            pcl::PolygonMeshConstPtr meshPtr = mesh.polygonMesh();
            size_t nrPolygons = meshPtr->polygons.size();

            if (indices.size() > 0)
            {
                for (int i = 0; i < indices.size(); ++i)
                {
                    if (indices[i] < 0 || indices[i] >= nrPolygons)
                    {
                        retval += ito::RetVal(ito::retError, 0, "indices contain invalid values. out of bounds.");
                        t_out = ito::pclInvalid;
                        break;
                    }
                }
            }

            if (t_out == ito::pclXYZNormal)
            {
                if (t == ito::pclXYZ)
                {
                    pcl::PointCloud<pcl::PointXYZ>::Ptr points_(new pcl::PointCloud<pcl::PointXYZ>());
                    pcl::PointCloud<pcl::PointNormal>::Ptr out_points = out.toPointXYZNormal();
                    pcl::PointNormal pt;
                    pt.curvature = 0.0;
#if PCL_VERSION_COMPARE(>=,1,7,0)
                    pcl::fromPCLPointCloud2(meshPtr->cloud, *points_);
#else
					pcl::fromROSMsg(meshPtr->cloud, *points_);
#endif
                    has_next = indexed_mode ? (indices.size() > next_index) : (nrPolygons > next_index);
                    out_points->reserve(indices.size() > 0 ? indices.size() : nrPolygons);

                    while (has_next)
                    {
                        index = indexed_mode ? indices[next_index] : next_index;

                        v = &(meshPtr->polygons[index]);
                        if( v->vertices.size() >= 3 )
                        {
                            v_ = &(v->vertices.front());
                            cog = Eigen::Vector3f::Zero();
                            for(size_t j = 0; j < v->vertices.size() ; j++)
                            {
                                cog += points_->points[*v_].getVector3fMap();
                                v_++;
                            }
                            v_ = &(v->vertices.front());
                            a = points_->points[*(v_+1)].getVector3fMap() - points_->points[*v_].getVector3fMap();
                            b = points_->points[*(v_+2)].getVector3fMap() - points_->points[*v_].getVector3fMap();
                            //     P1
                            //    /  \
                            //   P0 - P2
                            // normal points out of screen if polygon is [0,1,2]
                            normal = a.cross(b).normalized(); //normal points outside of the surface
                            cog /= static_cast<float>( v->vertices.size() );
                            pt.x = cog[0];
                            pt.y = cog[1];
                            pt.z = cog[2];
                            pt.normal_x = normal[0];
                            pt.normal_y = normal[1];
                            pt.normal_z = normal[2];
                            out_points->push_back(pt);
                            count++;
                        }

                        next_index ++;
                        has_next = indexed_mode ? (indices.size() > next_index) : (nrPolygons > next_index);
                    }

                    out_points->resize(count);
                    out_points->is_dense = true;
                    out_points->height = 1; //unorganized
                    out_points->width = count;
                }
                else
                {
                    pcl::PointCloud<pcl::PointNormal>::Ptr points_(new pcl::PointCloud<pcl::PointNormal>());
                    pcl::PointCloud<pcl::PointNormal>::Ptr out_points = out.toPointXYZNormal();
                    pcl::PointNormal pt;
                    float curvature;
#if PCL_VERSION_COMPARE(>=,1,7,0)
                    pcl::fromPCLPointCloud2(meshPtr->cloud, *points_);
#else
					pcl::fromROSMsg(meshPtr->cloud, *points_);
#endif
                    has_next = indexed_mode ? (indices.size() > next_index) : (nrPolygons > next_index);
                    out_points->reserve(indices.size() > 0 ? indices.size() : nrPolygons);

                    while (has_next)
                    {
                        index = indexed_mode ? indices[next_index] : next_index;

                        v = &(meshPtr->polygons[index]);
                        if( v->vertices.size() >= 3 )
                        {
                            v_ = &(v->vertices.front());
                            cog = Eigen::Vector3f::Zero();
                            curvature = 0.0;
                            for(size_t j = 0; j < v->vertices.size() ; j++)
                            {
                                cog += points_->points[*v_].getVector3fMap();
                                curvature += points_->points[*v_].curvature;
                                v_++;
                            }
                            v_ = &(v->vertices.front());
                            a = points_->points[*(v_+1)].getVector3fMap() - points_->points[*v_].getVector3fMap();
                            b = points_->points[*(v_+2)].getVector3fMap() - points_->points[*v_].getVector3fMap();
                            //     P1
                            //    /  \
                            //   P0 - P2
                            // normal points out of screen if polygon is [0,1,2]
                            normal = a.cross(b).normalized(); //normal points outside of the surface
                            cog /= static_cast<float>( v->vertices.size() );
                            pt.x = cog[0];
                            pt.y = cog[1];
                            pt.z = cog[2];
                            pt.normal_x = normal[0];
                            pt.normal_y = normal[1];
                            pt.normal_z = normal[2];
                            pt.curvature = curvature / v->vertices.size();
                            out_points->push_back(pt);
                            count++;
                        }

                        next_index ++;
                        has_next = indexed_mode ? (indices.size() > next_index) : (nrPolygons > next_index);
                    }

                    out_points->resize(count);
                    out_points->is_dense = true;
                    out_points->height = 1; //unorganized
                    out_points->width = count;
                }
            }
            else if (t_out == ito::pclXYZINormal)
            {
                if (t == ito::pclXYZI)
                {
                    pcl::PointCloud<pcl::PointXYZI>::Ptr points_(new pcl::PointCloud<pcl::PointXYZI>());
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr out_points = out.toPointXYZINormal();
                    pcl::PointXYZINormal pt;
                    pt.curvature = 0.0;
#if PCL_VERSION_COMPARE(>=,1,7,0)
                    pcl::fromPCLPointCloud2(meshPtr->cloud, *points_);
#else
					pcl::fromROSMsg(meshPtr->cloud, *points_);
#endif
                    has_next = indexed_mode ? (indices.size() > next_index) : (nrPolygons > next_index);
                    out_points->reserve(indices.size() > 0 ? indices.size() : nrPolygons);

                    float intensity;

                    while (has_next)
                    {
                        index = indexed_mode ? indices[next_index] : next_index;

                        v = &(meshPtr->polygons[index]);
                        if( v->vertices.size() >= 3 )
                        {
                            v_ = &(v->vertices.front());
                            cog = Eigen::Vector3f::Zero();
                            intensity = 0.0;
                            for(size_t j = 0; j < v->vertices.size() ; j++)
                            {
                                cog += points_->points[*v_].getVector3fMap();
                                intensity += points_->points[*v_].intensity;
                                v_++;
                            }
                            v_ = &(v->vertices.front());
                            a = points_->points[*(v_+1)].getVector3fMap() - points_->points[*v_].getVector3fMap();
                            b = points_->points[*(v_+2)].getVector3fMap() - points_->points[*v_].getVector3fMap();
                            //     P1
                            //    /  \
                            //   P0 - P2
                            // normal points out of screen if polygon is [0,1,2]
                            normal = a.cross(b).normalized(); //normal points outside of the surface
                            cog /= static_cast<float>( v->vertices.size() );
                            pt.x = cog[0];
                            pt.y = cog[1];
                            pt.z = cog[2];
                            pt.normal_x = normal[0];
                            pt.normal_y = normal[1];
                            pt.normal_z = normal[2];
                            pt.intensity = intensity / v->vertices.size();
                            out_points->push_back(pt);
                            count++;
                        }

                        next_index ++;
                        has_next = indexed_mode ? (indices.size() > next_index) : (nrPolygons > next_index);
                    }

                    out_points->resize(count);
                    out_points->is_dense = true;
                    out_points->height = 1; //unorganized
                    out_points->width = count;
                }
                else
                {
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr points_(new pcl::PointCloud<pcl::PointXYZINormal>());
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr out_points = out.toPointXYZINormal();
                    pcl::PointXYZINormal pt;
                    float curvature;
#if PCL_VERSION_COMPARE(>=,1,7,0)
                    pcl::fromPCLPointCloud2(meshPtr->cloud, *points_);
#else
					pcl::fromROSMsg(meshPtr->cloud, *points_);
#endif
                    has_next = indexed_mode ? (indices.size() > next_index) : (nrPolygons > next_index);
                    out_points->reserve(indices.size() > 0 ? indices.size() : nrPolygons);

                    float intensity;

                    while (has_next)
                    {
                        index = indexed_mode ? indices[next_index] : next_index;

                        v = &(meshPtr->polygons[index]);
                        if( v->vertices.size() >= 3 )
                        {
                            v_ = &(v->vertices.front());
                            cog = Eigen::Vector3f::Zero();
                            intensity = 0.0;
                            curvature = 0.0;
                            for(size_t j = 0; j < v->vertices.size() ; j++)
                            {
                                cog += points_->points[*v_].getVector3fMap();
                                intensity += points_->points[*v_].intensity;
                                curvature += points_->points[*v_].curvature;
                                v_++;
                            }
                            v_ = &(v->vertices.front());
                            a = points_->points[*(v_+1)].getVector3fMap() - points_->points[*v_].getVector3fMap();
                            b = points_->points[*(v_+2)].getVector3fMap() - points_->points[*v_].getVector3fMap();
                            //     P1
                            //    /  \
                            //   P0 - P2
                            // normal points out of screen if polygon is [0,1,2]
                            normal = a.cross(b).normalized(); //normal points outside of the surface
                            cog /= static_cast<float>( v->vertices.size() );
                            pt.x = cog[0];
                            pt.y = cog[1];
                            pt.z = cog[2];
                            pt.normal_x = normal[0];
                            pt.normal_y = normal[1];
                            pt.normal_z = normal[2];
                            pt.intensity = intensity / v->vertices.size();
                            pt.curvature = curvature / v->vertices.size();
                            out_points->push_back(pt);
                            count++;
                        }

                        next_index ++;
                        has_next = indexed_mode ? (indices.size() > next_index) : (nrPolygons > next_index);
                    }

                    out_points->resize(count);
                    out_points->is_dense = true;
                    out_points->height = 1; //unorganized
                    out_points->width = count;
                }
            }
            else if (t_out == ito::pclXYZRGBNormal)
            {
                if (t == ito::pclXYZRGBA)
                {
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points_(new pcl::PointCloud<pcl::PointXYZRGBA>());
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr out_points = out.toPointXYZRGBNormal();
                    pcl::PointXYZRGBNormal pt;
                    pt.curvature = 0.0;
#if PCL_VERSION_COMPARE(>=,1,7,0)
                    pcl::fromPCLPointCloud2(meshPtr->cloud, *points_);
#else
					pcl::fromROSMsg(meshPtr->cloud, *points_);
#endif
                    has_next = indexed_mode ? (indices.size() > next_index) : (nrPolygons > next_index);
                    out_points->reserve(indices.size() > 0 ? indices.size() : nrPolygons);

                    float red,green,blue,alpha;

                    while (has_next)
                    {
                        index = indexed_mode ? indices[next_index] : next_index;

                        v = &(meshPtr->polygons[index]);
                        if( v->vertices.size() >= 3 )
                        {
                            v_ = &(v->vertices.front());
                            cog = Eigen::Vector3f::Zero();
                            red = 0.0; green = 0.0; blue = 0.0; alpha = 0.0;
                            for(size_t j = 0; j < v->vertices.size() ; j++)
                            {
                                cog += points_->points[*v_].getVector3fMap();
                                red += points_->points[*v_].r;
                                green += points_->points[*v_].g;
                                blue += points_->points[*v_].b;
                                alpha += points_->points[*v_].a;
                                v_++;
                            }
                            v_ = &(v->vertices.front());
                            a = points_->points[*(v_+1)].getVector3fMap() - points_->points[*v_].getVector3fMap();
                            b = points_->points[*(v_+2)].getVector3fMap() - points_->points[*v_].getVector3fMap();
                            //     P1
                            //    /  \
                            //   P0 - P2
                            // normal points out of screen if polygon is [0,1,2]
                            normal = a.cross(b).normalized(); //normal points outside of the surface
                            cog /= static_cast<float>( v->vertices.size() );
                            pt.x = cog[0];
                            pt.y = cog[1];
                            pt.z = cog[2];
                            pt.normal_x = normal[0];
                            pt.normal_y = normal[1];
                            pt.normal_z = normal[2];
                            pt.r = red / v->vertices.size();
                            pt.g = green / v->vertices.size();
                            pt.b = blue / v->vertices.size();
                            pt.a = alpha / v->vertices.size();
                            out_points->push_back(pt);
                            count++;
                        }

                        next_index ++;
                        has_next = indexed_mode ? (indices.size() > next_index) : (nrPolygons > next_index);
                    }

                    out_points->resize(count);
                    out_points->is_dense = true;
                    out_points->height = 1; //unorganized
                    out_points->width = count;
                }
                else
                {
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr points_(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr out_points = out.toPointXYZRGBNormal();
                    pcl::PointXYZRGBNormal pt;
                    float curvature;
#if PCL_VERSION_COMPARE(>=,1,7,0)
                    pcl::fromPCLPointCloud2(meshPtr->cloud, *points_);
#else
					pcl::fromROSMsg(meshPtr->cloud, *points_);
#endif
                    has_next = indexed_mode ? (indices.size() > next_index) : (nrPolygons > next_index);
                    out_points->reserve(indices.size() > 0 ? indices.size() : nrPolygons);

                    float red,green,blue,alpha;

                    while (has_next)
                    {
                        index = indexed_mode ? indices[next_index] : next_index;

                        v = &(meshPtr->polygons[index]);
                        if( v->vertices.size() >= 3 )
                        {
                            v_ = &(v->vertices.front());
                            cog = Eigen::Vector3f::Zero();
                            red = 0.0; green = 0.0; blue = 0.0; alpha = 0.0;
                            curvature = 0.0;
                            for(size_t j = 0; j < v->vertices.size() ; j++)
                            {
                                cog += points_->points[*v_].getVector3fMap();
                                red += points_->points[*v_].r;
                                green += points_->points[*v_].g;
                                blue += points_->points[*v_].b;
                                alpha += points_->points[*v_].a;
                                curvature += points_->points[*v_].curvature;
                                v_++;
                            }
                            v_ = &(v->vertices.front());
                            a = points_->points[*(v_+1)].getVector3fMap() - points_->points[*v_].getVector3fMap();
                            b = points_->points[*(v_+2)].getVector3fMap() - points_->points[*v_].getVector3fMap();
                            //     P1
                            //    /  \
                            //   P0 - P2
                            // normal points out of screen if polygon is [0,1,2]
                            normal = a.cross(b).normalized(); //normal points outside of the surface
                            cog /= static_cast<float>( v->vertices.size() );
                            pt.x = cog[0];
                            pt.y = cog[1];
                            pt.z = cog[2];
                            pt.normal_x = normal[0];
                            pt.normal_y = normal[1];
                            pt.normal_z = normal[2];
                            pt.r = red / v->vertices.size();
                            pt.g = green / v->vertices.size();
                            pt.b = blue / v->vertices.size();
                            pt.a = alpha / v->vertices.size();
                            pt.curvature = curvature / v->vertices.size();
                            out_points->push_back(pt);
                            count++;
                        }

                        next_index ++;
                        has_next = indexed_mode ? (indices.size() > next_index) : (nrPolygons > next_index);
                    }

                    out_points->resize(count);
                    out_points->is_dense = true;
                    out_points->height = 1; //unorganized
                    out_points->width = count;
                }
            }

        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, "no corresponding cloud type with normal vectors could be derived from cloud type of given mesh");
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, "invalid mesh");
    }
    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> ito::RetVal readXYZData(const cv::Mat *x, const cv::Mat *y, const cv::Mat *z, pcl::PointCloud<pcl::PointXYZ>::Ptr  &cloud, const bool deleteNaNorInf)
{
    pcl::PointXYZ point;
    int width = z->cols;
    int height = z->rows;
    const _Tp *xRow, *yRow, *zRow;
    size_t counter = 0;
    bool organized = true;

    if (deleteNaNorInf)
    {
        cloud->reserve(width * height);
        cloud->is_dense = true; //always dense since nan or inf not included in cloud

        for (int i = 0; i < x->rows; i++)
        {
            xRow = x->ptr<_Tp>(i);
            yRow = y->ptr<_Tp>(i);
            zRow = z->ptr<_Tp>(i);

            for (int j = 0; j < x->cols; j++)
            {
                if ((std::isfinite(zRow[j]) && std::isfinite(yRow[j]) && std::isfinite(xRow[j])))
                {
                    point.x = xRow[j];
                    point.y = yRow[j];
                    point.z = zRow[j];
                    (*cloud).push_back(point);
                    counter++;
                }
                else
                {
                    organized = false;
                }
            }
        }

        cloud->resize(counter);
    }
    else
    {
        cloud->resize(width * height);
        cloud->is_dense = true;

        for (int i = 0; i < x->rows; i++)
        {
            xRow = x->ptr<_Tp>(i);
            yRow = y->ptr<_Tp>(i);
            zRow = z->ptr<_Tp>(i);

            for (int j = 0; j < x->cols; j++)
            {
                point.x = xRow[j];
                point.y = yRow[j];
                point.z = zRow[j];

                if (!std::isfinite(point.z) || !std::isfinite(point.x) || !std::isfinite(point.y))
                {
                    cloud->is_dense = false;
                }

                cloud->at(i * width + j) = point;
                counter++;
            }
        }
    }

	if (organized)
	{
		cloud->width = width;
		cloud->height = height;
	}
	else
	{
		cloud->width = counter;
		cloud->height = 1;
	}

    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> ito::RetVal readXYZIData(const cv::Mat *x, const cv::Mat *y, const cv::Mat *z, const cv::Mat *inten, pcl::PointCloud<pcl::PointXYZI>::Ptr  &cloud, const bool deleteNaNorInf)
{
    pcl::PointXYZI point;
    int width = z->cols;
    int height = z->rows;
    const _Tp *xRow, *yRow, *zRow, *iRow;
    size_t counter = 0;
    bool organized = true;

    if (deleteNaNorInf)
    {
        cloud->reserve(width * height);
        cloud->is_dense = true; //always dense since nan or inf not included in cloud

        for (int i = 0; i < x->rows; i++)
        {
            xRow = x->ptr<_Tp>(i);
            yRow = y->ptr<_Tp>(i);
            zRow = z->ptr<_Tp>(i);
            iRow = inten->ptr<_Tp>(i);

            for (int j = 0; j < x->cols; j++)
            {
                if ((std::isfinite(zRow[j]) && std::isfinite(yRow[j]) && std::isfinite(xRow[j])))
                {
                    point.x = xRow[j];
                    point.y = yRow[j];
                    point.z = zRow[j];
                    point.intensity = iRow[j];
                    (*cloud).push_back(point);
                    counter++;
                }
                else
                {
                    organized = false;
                }
            }
        }

        cloud->resize(counter);
    }
    else
    {
        cloud->is_dense = true;
        cloud->resize(width * height);
        for (int i = 0; i < x->rows; i++)
        {
            xRow = x->ptr<_Tp>(i);
            yRow = y->ptr<_Tp>(i);
            zRow = z->ptr<_Tp>(i);
            iRow = inten->ptr<_Tp>(i);

            for (int j = 0; j < x->cols; j++)
            {
                point.x = xRow[j];
                point.y = yRow[j];
                point.z = zRow[j];
                point.intensity = iRow[j];

                if (!std::isfinite(point.z) || !std::isfinite(point.x) || !std::isfinite(point.y))
                {
                    cloud->is_dense = false;
                }

                cloud->at(i * width + j) = point;
                counter++;
            }
        }
    }

	if (organized)
	{
		cloud->width = width;
		cloud->height = height;
	}
	else
	{
		cloud->width = counter;
		cloud->height = 1;
	}

    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> ito::RetVal readXYZRGBAData(const cv::Mat *x, const cv::Mat *y, const cv::Mat *z, const cv::Mat *color, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr  &cloud, const bool deleteNaNorInf)
{
    pcl::PointXYZRGBA point;
    int width = z->cols;
    int height = z->rows;
    const _Tp *xRow, *yRow, *zRow;
    const ito::Rgba32 *cRow;
    size_t counter = 0;
    bool organized = true;

    if (deleteNaNorInf)
    {
        cloud->reserve(width * height);
        cloud->is_dense = true; //always dense since nan or inf not included in cloud

        for (int i = 0; i < x->rows; i++)
        {
            xRow = x->ptr<_Tp>(i);
            yRow = y->ptr<_Tp>(i);
            zRow = z->ptr<_Tp>(i);
            cRow = color->ptr<ito::Rgba32>(i);

            for (int j = 0; j < x->cols; j++)
            {
                if ((std::isfinite(zRow[j]) && std::isfinite(yRow[j]) && std::isfinite(xRow[j])))
                {
                    point.x = xRow[j];
                    point.y = yRow[j];
                    point.z = zRow[j];
                    point.rgba = cRow[j].rgba;
                    (*cloud).push_back(point);
                    counter++;
                }
                else
                {
                    organized = false;
                }
            }
        }

        cloud->resize(counter);
    }
    else
    {
        cloud->resize(width * height);
        cloud->is_dense = true;

        for (int i = 0; i < x->rows; i++)
        {
            xRow = x->ptr<_Tp>(i);
            yRow = y->ptr<_Tp>(i);
            zRow = z->ptr<_Tp>(i);
            cRow = color->ptr<ito::Rgba32>(i);

            for (int j = 0; j < x->cols; j++)
            {
                point.x = xRow[j];
                point.y = yRow[j];
                point.z = zRow[j];
                point.rgba = cRow[j].rgba;

                if (!std::isfinite(point.z) || !std::isfinite(point.x) || !std::isfinite(point.y))
                {
                    cloud->is_dense = false;
                }

                cloud->at(i * width + j) = point;
                counter++;
            }
        }
    }


	if ((counter == (width * height)) && organized)
	{
		cloud->width = width;
		cloud->height = height;
	}
	else
	{
		cloud->width = counter;
		cloud->height = 1;
	}

    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal pointCloudFromXYZ(const DataObject* mapX, const DataObject* mapY, const DataObject* mapZ, PCLPointCloud &out, bool deleteNaNorInf /*= false*/)
{
    RetVal retval = retOk;

    retval += ito::dObjHelper::verify2DDataObject(mapZ, "Z", 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 8, ito::tFloat32, ito::tFloat64, ito::tUInt8, ito::tInt8, ito::tUInt16, ito::tInt16, ito::tUInt32, ito::tInt32);
    retval += ito::dObjHelper::verify2DDataObject(mapX, "X", mapZ->getSize(0), mapZ->getSize(0), mapZ->getSize(1), mapZ->getSize(1), 8, ito::tFloat32, ito::tFloat64, ito::tUInt8, ito::tInt8, ito::tUInt16, ito::tInt16, ito::tUInt32, ito::tInt32);
    retval += ito::dObjHelper::verify2DDataObject(mapY, "Y", mapZ->getSize(0), mapZ->getSize(0), mapZ->getSize(1), mapZ->getSize(1), 8, ito::tFloat32, ito::tFloat64, ito::tUInt8, ito::tInt8, ito::tUInt16, ito::tInt16, ito::tUInt32, ito::tInt32);

    if (!retval.containsError())
    {
        const cv::Mat *x = mapX->get_mdata()[ mapX->seekMat(0) ];
        const cv::Mat *y = mapY->get_mdata()[ mapY->seekMat(0) ];
        const cv::Mat *z = mapZ->get_mdata()[ mapZ->seekMat(0) ];

		cv::Mat x_f32, y_f32, z_f32;
		x->convertTo(x_f32, CV_32FC1, 1.0, 0.0);
		y->convertTo(y_f32, CV_32FC1, 1.0, 0.0);
		z->convertTo(z_f32, CV_32FC1, 1.0, 0.0);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        pcl::PointXYZ point;
        ito::PCLPointCloud pointCloud;
        pointCloud = ito::PCLPointCloud(ito::pclXYZ);
        cloud = pointCloud.toPointXYZ();
		retval += readXYZData<ito::float32>(&x_f32, &y_f32, &z_f32, cloud, deleteNaNorInf);
        out = pointCloud;
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal pointCloudFromXYZI(const DataObject* mapX, const DataObject* mapY, const DataObject* mapZ, const DataObject* mapI, PCLPointCloud &out, bool deleteNaNorInf /*= false*/)
{
    RetVal retval = retOk;

    retval += ito::dObjHelper::verify2DDataObject(mapZ, "Z", 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 8, ito::tFloat32, ito::tFloat64, ito::tUInt8, ito::tInt8, ito::tUInt16, ito::tInt16, ito::tUInt32, ito::tInt32);
    retval += ito::dObjHelper::verify2DDataObject(mapX, "X", mapZ->getSize(0), mapZ->getSize(0), mapZ->getSize(1), mapZ->getSize(1), 8, ito::tFloat32, ito::tFloat64, ito::tUInt8, ito::tInt8, ito::tUInt16, ito::tInt16, ito::tUInt32, ito::tInt32);
    retval += ito::dObjHelper::verify2DDataObject(mapY, "Y", mapZ->getSize(0), mapZ->getSize(0), mapZ->getSize(1), mapZ->getSize(1), 8, ito::tFloat32, ito::tFloat64, ito::tUInt8, ito::tInt8, ito::tUInt16, ito::tInt16, ito::tUInt32, ito::tInt32);
    retval += ito::dObjHelper::verify2DDataObject(mapI, "I", mapZ->getSize(0), mapZ->getSize(0), mapZ->getSize(1), mapZ->getSize(1), 8, ito::tFloat32, ito::tFloat64, ito::tUInt8, ito::tInt8, ito::tUInt16, ito::tInt16, ito::tUInt32, ito::tInt32);

    if (!retval.containsError())
    {
        const cv::Mat *x = mapX->get_mdata()[ mapX->seekMat(0) ];
        const cv::Mat *y = mapY->get_mdata()[ mapY->seekMat(0) ];
        const cv::Mat *z = mapZ->get_mdata()[ mapZ->seekMat(0) ];
        const cv::Mat *intensity = mapI->get_mdata()[ mapI->seekMat(0) ];

		cv::Mat x_f32, y_f32, z_f32, i_f32;
		x->convertTo(x_f32, CV_32FC1, 1.0, 0.0);
		y->convertTo(y_f32, CV_32FC1, 1.0, 0.0);
		z->convertTo(z_f32, CV_32FC1, 1.0, 0.0);
		intensity->convertTo(i_f32, CV_32FC1, 1.0, 0.0);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
        pcl::PointXYZI point;
        ito::PCLPointCloud pointCloud;
        pointCloud = ito::PCLPointCloud(ito::pclXYZI);
        cloud = pointCloud.toPointXYZI();

        retval += readXYZIData<ito::float32>(&x_f32, &y_f32, &z_f32, &i_f32, cloud, deleteNaNorInf);

        out = pointCloud;
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal pointCloudFromXYZRGBA(const DataObject* mapX, const DataObject* mapY, const DataObject* mapZ, const DataObject* mapColor, PCLPointCloud &out, bool deleteNaNorInf /*= false*/)
{
    RetVal retval = retOk;

    retval += ito::dObjHelper::verify2DDataObject(mapZ, "Z", 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 8, ito::tFloat32, ito::tFloat64, ito::tUInt8, ito::tInt8, ito::tUInt16, ito::tInt16, ito::tUInt32, ito::tInt32);
    retval += ito::dObjHelper::verify2DDataObject(mapX, "X", mapZ->getSize(0), mapZ->getSize(0), mapZ->getSize(1), mapZ->getSize(1), 8, ito::tFloat32, ito::tFloat64, ito::tUInt8, ito::tInt8, ito::tUInt16, ito::tInt16, ito::tUInt32, ito::tInt32);
    retval += ito::dObjHelper::verify2DDataObject(mapY, "Y", mapZ->getSize(0), mapZ->getSize(0), mapZ->getSize(1), mapZ->getSize(1), 8, ito::tFloat32, ito::tFloat64, ito::tUInt8, ito::tInt8, ito::tUInt16, ito::tInt16, ito::tUInt32, ito::tInt32);
    retval += ito::dObjHelper::verify2DDataObject(mapColor, "Color", mapZ->getSize(0), mapZ->getSize(0), mapZ->getSize(1), mapZ->getSize(1), 1, ito::tRGBA32);

    if (!retval.containsError())
    {
        const cv::Mat *x = mapX->get_mdata()[ mapX->seekMat(0) ];
        const cv::Mat *y = mapY->get_mdata()[ mapY->seekMat(0) ];
        const cv::Mat *z = mapZ->get_mdata()[ mapZ->seekMat(0) ];
        const cv::Mat *color = mapColor->get_mdata()[ mapColor->seekMat(0) ]; //always rgba32

		cv::Mat x_f32, y_f32, z_f32;
		x->convertTo(x_f32, CV_32FC1, 1.0, 0.0);
		y->convertTo(y_f32, CV_32FC1, 1.0, 0.0);
		z->convertTo(z_f32, CV_32FC1, 1.0, 0.0);

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
        pcl::PointXYZRGBA point;
        ito::PCLPointCloud pointCloud;
        pointCloud = ito::PCLPointCloud(ito::pclXYZRGBA);
        cloud = pointCloud.toPointXYZRGBA();

        retval += readXYZRGBAData<ito::float32>(&x_f32, &y_f32, &z_f32, color, cloud, deleteNaNorInf);

        out = pointCloud;
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal pointCloudFromDisparity(const DataObject* mapDisp, PCLPointCloud &out, bool deleteNaNorInf /*= false*/)
{
    return pointCloudFromDisparityI(mapDisp, NULL, out, deleteNaNorInf);
}

//------------------------------------------------------------------------------------------------------------------------------
template<typename _TpM, typename _TpI> void fromDataObj(const cv::Mat *mapDisp, const cv::Mat *mapInt, const ito::float32 firstX, const ito::float32 stepX,
    const ito::float32 firstY, const ito::float32 stepY,
    const ito::float32 minI, const ito::float32 scaleI,
    const bool deleteNaNorInf, ito::PCLPointCloud &out)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    pcl::PointXYZI point;

    int width = mapDisp->cols;
    int height = mapDisp->rows;

    if (deleteNaNorInf)
    {
        out = ito::PCLPointCloud(ito::pclXYZI);
        cloud = out.toPointXYZI();
        out.reserve(width*height);
        size_t counter = 0;
        cloud->is_dense = true; //cloud is always dense since nan and inf values are deleted
        bool organized = true;

        for (int i = 0; i < height; i++)
        {
            const _TpM *zRow = mapDisp->ptr<_TpM>(i);
            const _TpI *iRow = mapInt->ptr<_TpI>(i);

            for (int j = 0; j < width; j++)
            {
                if (is_finite<_TpM>(zRow[j]))
                {
                    point.x = firstX + j * stepX;
                    point.y = firstY + i * stepY;
                    point.z = zRow[j];
                    point.intensity = (iRow[j] - minI) * scaleI;
                    (*cloud).push_back(point);
                    counter++;
                }
                else
                {
                    organized = false;
                }
            }
        }

        cloud->resize(counter);

        if (organized)
        {
            cloud->width = width;
            cloud->height = height;
        }
        else
        {
            cloud->width = counter;
            cloud->height = 1;
        }
    }
    else
    {
        out = ito::PCLPointCloud(width, height, ito::pclXYZI, ito::PCLPoint(point));
        cloud = out.toPointXYZI();
        cloud->is_dense = true;

        for (int i = 0; i < height; i++)
        {
            _TpM *zRow = (_TpM*)mapDisp->ptr<_TpM>(i);
            _TpI *iRow = (_TpI*)mapInt->ptr<_TpI>(i);

            for (int j = 0; j < width; j++)
            {
                point.x = firstX + j * stepX;
                point.y = firstY + i * stepY;
                point.z = zRow[j];
                point.intensity = (iRow[j] - minI) * scaleI;

                if (!std::isfinite(point.z))
                {
                    cloud->is_dense = false;
                }

                cloud->at(i * width + j) = point;
            }
        }

        //always organized
        cloud->height = height;
        cloud->width = width;
    }
}


//------------------------------------------------------------------------------------------------------------------------------
template<typename _TpM> ito::RetVal fromDataObj1(const cv::Mat *mapDisp, const ito::float32 firstX, const ito::float32 stepX, const ito::float32 firstY, const ito::float32 stepY, const bool deleteNaNorInf, ito::PCLPointCloud &out)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointXYZ point;

    int width = mapDisp->cols;
    int height = mapDisp->rows;

    if (deleteNaNorInf)
    {
        out = ito::PCLPointCloud(ito::pclXYZ);
        cloud = out.toPointXYZ();
        cloud->is_dense = true; //cloud is always dense since nan or inf are not included
        out.reserve(width * height);
        size_t counter = 0;
        bool organized = true;

        for (int i = 0; i < height; i++)
        {
            const _TpM *zRow = mapDisp->ptr<_TpM>(i);

            for (int j = 0; j < width; j++)
            {
                if (is_finite<_TpM>(zRow[j]))
                {
                    point.x = firstX + j * stepX;
                    point.y = firstY + i * stepY;
                    point.z = zRow[j];
                    (*cloud).push_back(point);
                    counter++;
                }
                else
                {
                    organized = false; //at least one nan value --> cloud is not organized
                }
            }
        }

        cloud->resize(counter);

        if (organized)
        {
            cloud->width = width;
            cloud->height = height;
        }
        else
        {
            cloud->width = counter;
            cloud->height = 1;
        }
    }
    else
    {
        out = ito::PCLPointCloud(width, height, ito::pclXYZ, ito::PCLPoint(point));
        cloud = out.toPointXYZ();
        cloud->is_dense = true;

        for (int i = 0; i < height; i++)
        {
            _TpM *zRow = (_TpM*)mapDisp->ptr<ito::float32>(i);

            for (int j = 0; j < width; j++)
            {
                point.x = firstX + j * stepX;
                point.y = firstY + i * stepY;
                point.z = zRow[j];

                if (!std::isfinite(point.z))
                {
                    cloud->is_dense = false;
                }

                cloud->at(i * width + j) = point;
            }
        }

        cloud->height = height;
        cloud->width = width;
    }

    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------
template<typename _TpM> ito::RetVal fromDataObj1(const cv::Mat *mapDisp, const ito::DataObject *mapI, const ito::float32 firstX, const ito::float32 stepX, const ito::float32 firstY, const ito::float32 stepY, const bool deleteNaNorInf, ito::PCLPointCloud &out)
{
    const cv::Mat *intensity = mapI->get_mdata()[ mapI->seekMat(0) ];

    switch(mapI->getType())
    {
        case ito::tUInt8:
        {
            // 8 bit intensity we always scale between 0 - 255
            fromDataObj<_TpM, ito::uint8>(mapDisp, intensity, firstX, stepX, firstY, stepY, 0.0, 1.0 / 255.0, deleteNaNorInf, out);
        }
        break;
        case ito::tInt8:
        {
            fromDataObj<_TpM, ito::int8>(mapDisp, intensity, firstX, stepX, firstY, stepY, -128.0, 1.0 / 255.0, deleteNaNorInf, out);
        }
        break;
        case ito::tUInt16:
        {
            ito::float64 minVal, maxVal;
            ito::uint32 minLoc[3], maxLoc[3];
            ito::dObjHelper::minMaxValue(mapI, minVal, &minLoc[0], maxVal, &maxLoc[0]);
            if (maxVal <= 1024.0)       //guess 10 bit image
                maxVal = 1024.0;
            else if (maxVal <= 4096.0)  //guess 12 bit image
                maxVal = 4096.0;
            else if (maxVal <= 16384.0) //guess 14 bit image
                maxVal = 16384.0;
            else
                maxVal = 65535.0;       //guess 16 bit image
            fromDataObj<_TpM, ito::uint16>(mapDisp, intensity, firstX, stepX, firstY, stepY, 0.0, 1.0 / maxVal, deleteNaNorInf, out);
        }
        break;
        case ito::tInt16:
        {
            ito::float64 minVal, maxVal;
            ito::uint32 minLoc[3], maxLoc[3];
            ito::dObjHelper::minMaxValue(mapI, minVal, &minLoc[0], maxVal, &maxLoc[3]);
            if (maxVal <= 511.0)       //guess 10 bit image
            {
                minVal = -512.0;
                maxVal = 1024.0;
            }
            else if (maxVal <= 2047.0)  //guess 12 bit image
            {
                minVal = -2048.0;
                maxVal = 4096.0;
            }
            else if (maxVal <= 8191.0) //guess 14 bit image
            {
                minVal = -8192.0;
                maxVal = 16384.0;
            }
            else
            {
                minVal = -32768.0;
                maxVal = 65536.0;      //guess 16 bit image
            }
            fromDataObj<_TpM, ito::int16>(mapDisp, intensity, firstX, stepX, firstY, stepY, minVal, 1.0 / maxVal, deleteNaNorInf, out);
        }
        break;
        case ito::tUInt32:
        {
            ito::float64 minVal, maxVal;
            ito::uint32 minLoc[3], maxLoc[3];
            ito::dObjHelper::minMaxValue(mapI, minVal, &minLoc[0], maxVal, &maxLoc[0]);
            fromDataObj<_TpM, ito::uint32>(mapDisp, intensity, firstX, stepX, firstY, stepY, minVal, 1.0 / (maxVal - minVal), deleteNaNorInf, out);
        }
        break;
        case ito::tInt32:
        {
            ito::float64 minVal, maxVal;
            ito::uint32 minLoc[3], maxLoc[3];
            ito::dObjHelper::minMaxValue(mapI, minVal, &minLoc[0], maxVal, &maxLoc[0]);
            fromDataObj<_TpM, ito::int32>(mapDisp, intensity, firstX, stepX, firstY, stepY, minVal, 1.0 / (maxVal - minVal), deleteNaNorInf, out);
        }
        break;
        case ito::tFloat32:
        {
            ito::float64 minVal, maxVal;
            ito::uint32 minLoc[3], maxLoc[3];
            ito::dObjHelper::minMaxValue(mapI, minVal, &minLoc[0], maxVal, &maxLoc[0]);
            fromDataObj<_TpM, ito::float32>(mapDisp, intensity, firstX, stepX, firstY, stepY, minVal, 1.0 / (maxVal - minVal), deleteNaNorInf, out);
        }
        break;
        case ito::tFloat64:
        {
            ito::float64 minVal, maxVal;
            ito::uint32 minLoc[3], maxLoc[3];
            ito::dObjHelper::minMaxValue(mapI, minVal, &minLoc[0], maxVal, &maxLoc[0]);
            fromDataObj<_TpM, ito::float64>(mapDisp, intensity, firstX, stepX, firstY, stepY, minVal, 1.0 / (maxVal - minVal), deleteNaNorInf, out);
        }
        break;
        default:
            return ito::RetVal(ito::retError, 0, "Unknown type or type not implemented");
    }
    return ito::retOk;
}



//------------------------------------------------------------------------------------------------------------------------------
ito::RetVal pointCloudFromDisparityI(const DataObject* mapDisp, const DataObject *mapI, PCLPointCloud &out, bool deleteNaNorInf /*= false*/)
{
    RetVal retval = retOk;
    float firstX = 0.0;
    float stepX = 1.0;
    float firstY = 0.0;
    float stepY = 1.0;

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

        const cv::Mat *z = mapDisp->get_mdata()[ mapDisp->seekMat(0) ];

        if (mapI == NULL)
        {
            switch(mapDisp->getType())
            {
                case ito::tUInt8:
                {
                    fromDataObj1<ito::uint8>(z, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
                }
                break;
                case ito::tInt8:
                {
                    fromDataObj1<ito::int8>(z, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
                }
                break;
                case ito::tUInt16:
                {
                    fromDataObj1<ito::uint16>(z, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
                }
                break;
                case ito::tInt16:
                {
                    fromDataObj1<ito::int16>(z, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
                }
                break;
                case ito::tUInt32:
                {
                    fromDataObj1<ito::uint32>(z, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
                }
                break;
                case ito::tInt32:
                {
                    fromDataObj1<ito::int32>(z, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
                }
                break;
                case ito::tFloat32:
                {
                    fromDataObj1<ito::float32>(z, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
                }
                break;
                case ito::tFloat64:
                {
                    fromDataObj1<ito::float64>(z, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
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
                    return fromDataObj1<ito::uint8>(z, mapI, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
                }
                break;
                case ito::tInt8:
                {
                    return fromDataObj1<ito::int8>(z, mapI, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
                }
                break;
                case ito::tUInt16:
                {
                    return fromDataObj1<ito::uint16>(z, mapI, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
                }
                break;
                case ito::tInt16:
                {
                    return fromDataObj1<ito::int16>(z, mapI, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
                }
                break;
                case ito::tUInt32:
                {
                    return fromDataObj1<ito::uint32>(z, mapI, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
                }
                break;
                case ito::tInt32:
                {
                    return fromDataObj1<ito::int32>(z, mapI, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
                }
                break;
                case ito::tFloat32:
                {
                    return fromDataObj1<ito::float32>(z, mapI, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
                }
                break;
                case ito::tFloat64:
                {
                    return fromDataObj1<ito::float64>(z, mapI, firstX, stepX, firstY, stepY, deleteNaNorInf, out);
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
ito::RetVal pointCloudFromDisparityRGBA(const DataObject* mapDisp, const DataObject *mapColor, PCLPointCloud &out, bool deleteNaNorInf /*= false*/)
{
    RetVal retval = retOk;
    float firstX = 0.0;
    float stepX = 1.0;
    float firstY = 0.0;
    float stepY = 1.0;

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
        const ito::float32 *zRow;
        const ito::Rgba32 *cRow;

        const cv::Mat *z = mapDisp->get_mdata()[ mapDisp->seekMat(0) ];

        if (mapColor == NULL)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
            pcl::PointXYZ point;

            width = mapDisp->getSize(1);
            height = mapDisp->getSize(0);

            if (deleteNaNorInf)
            {
                out = ito::PCLPointCloud(ito::pclXYZ);
                cloud = out.toPointXYZ();
                out.reserve(width*height);
                size_t counter = 0;
                cloud->is_dense = true; //always dense since no nan or inf values will be in the cloud

                for (int i = 0; i < z->rows; i++)
                {
                    zRow = z->ptr<ito::float32>(i);

                    for (int j = 0; j < z->cols; j++)
                    {
                        if (!(std::isnan(zRow[j])))
                        {
                            point.x = firstX + j * stepX;
                            point.y = firstY + i * stepY;
                            point.z = zRow[j];
                            (*cloud).push_back(point);
                            counter++;
                        }
                    }
                }

                cloud->resize(counter);

                if (counter == (width*height))
                {
                    //organized
                    cloud->height = height;
                    cloud->width = width;
                }
                else
                {
                    //non-organized
                    cloud->height = 1;
                    cloud->width = counter;
                }
            }
            else
            {
                out = ito::PCLPointCloud(width, height, ito::pclXYZ, ito::PCLPoint(point));
                cloud = out.toPointXYZ();
                cloud->is_dense = true;
                //organized
                cloud->height = height;
                cloud->width = width;

                for (int i = 0; i < z->rows; i++)
                {
                    zRow = z->ptr<ito::float32>(i);

                    for (int j = 0; j < z->cols; j++)
                    {
                        point.x = firstX + j * stepX;
                        point.y = firstY + i * stepY;
                        point.z = zRow[j];

                        if (!std::isfinite(point.z))
                        {
                            cloud->is_dense = false;
                        }

                        cloud->at(i * width + j) = point;
                    }
                }
            }
        }
        else
        {
            const cv::Mat *color = mapColor->get_mdata()[ mapColor->seekMat(0) ];

            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
            pcl::PointXYZRGBA point;

            width = mapDisp->getSize(1);
            height = mapDisp->getSize(0);

            if (deleteNaNorInf)
            {
                out = ito::PCLPointCloud(ito::pclXYZRGBA);
                cloud = out.toPointXYZRGBA();
                out.reserve(width*height);
                size_t counter = 0;
                cloud->is_dense = true; //always dense since no nan or inf values will be in the cloud

                for (int i = 0; i < z->rows; i++)
                {
                    zRow = z->ptr<ito::float32>(i);
                    cRow = color->ptr<ito::Rgba32>(i);

                    for (int j = 0; j < z->cols; j++)
                    {
                        if (!(std::isnan(zRow[j])))
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

                cloud->resize(counter);

                if (counter == (width*height))
                {
                    //organized
                    cloud->height = height;
                    cloud->width = width;
                }
                else
                {
                    //non-organized
                    cloud->height = 1;
                    cloud->width = counter;
                }
            }
            else
            {
                out = ito::PCLPointCloud(width, height, ito::pclXYZRGBA, ito::PCLPoint(point));
                cloud = out.toPointXYZRGBA();
                cloud->is_dense = true;
                //organized
                cloud->height = height;
                cloud->width = width;

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

                        if (!std::isfinite(point.z))
                        {
                            cloud->is_dense = false;
                        }

                        cloud->at(i * width + j) = point;
                    }
                }
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

    if (in)
    {
        ito::DataObject in2 = ito::dObjHelper::squeezeConvertCheck2DDataObject(in, "transform", ito::Range(4, 4), ito::Range(4, 4), retval, ito::tFloat32, 8, ito::tInt8, ito::tUInt8, ito::tInt16, ito::tUInt16, ito::tInt32, ito::tUInt32, ito::tFloat32, ito::tFloat64);

        if (!retval.containsError())
        {
            const ito::float32* r0 = in2.rowPtr<ito::float32>(0, 0);
            const ito::float32* r1 = in2.rowPtr<ito::float32>(0, 1);
            const ito::float32* r2 = in2.rowPtr<ito::float32>(0, 2);
            const ito::float32* r3 = in2.rowPtr<ito::float32>(0, 3);

            Eigen::Matrix4f homMat;

            homMat << r0[0], r0[1], r0[2], r0[3],
                        r1[0], r1[1], r1[2], r1[3],
                        r2[0], r2[1], r2[2], r2[3],
                        r3[0], r3[1], r3[2], r3[3];

            out = homMat;
        }
    }
    else
    {
        retval += RetVal(retError,0,"dataObject 'transform' must not be NULL");
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

} //end namespace pclHelper
} //end namespace ito
