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

#include "pclStructures.h"

#include <pcl/exceptions.h>
#include <pcl/common/common.h>
#include <pcl/common/io.h>

#include "pclFunctions.h"

namespace ito {

#define PCLMAKEFUNCLIST(FuncName) static t##FuncName fList##FuncName[] =   \
{                                                                       \
   FuncName<pcl::PointXYZ>,                                             \
   FuncName<pcl::PointXYZI>,                                            \
   FuncName<pcl::PointXYZRGBA>,                                         \
   FuncName<pcl::PointNormal>,                                          \
   FuncName<pcl::PointXYZINormal>,                                      \
   FuncName<pcl::PointXYZRGBNormal>                                    \
};




const pcl::PointXYZ & PCLPoint::getPointXYZ() const
{
    if(m_type != ito::pclXYZ) throw pcl::PCLException("point is not of type PointXYZ",__FILE__,"getPointXYZ",__LINE__);
    return *reinterpret_cast<pcl::PointXYZ*>(m_genericPoint);
}

const pcl::PointXYZI & PCLPoint::getPointXYZI() const
{
    if(m_type != ito::pclXYZI) throw pcl::PCLException("point is not of type PointXYZI",__FILE__,"getPointXYZI",__LINE__);
    return *reinterpret_cast<pcl::PointXYZI*>(m_genericPoint);
}

const pcl::PointXYZRGBA & PCLPoint::getPointXYZRGBA() const
{
    if(m_type != ito::pclXYZRGBA) throw pcl::PCLException("point is not of type PointXYZRGBA",__FILE__,"getPointXYZRGBA",__LINE__);
    return *reinterpret_cast<pcl::PointXYZRGBA*>(m_genericPoint);
}

const pcl::PointNormal & PCLPoint::getPointXYZNormal() const
{
    if(m_type != ito::pclXYZNormal) throw pcl::PCLException("point is not of type PointXYZNormal",__FILE__,"getPointXYZNormal",__LINE__);
    return *reinterpret_cast<pcl::PointNormal*>(m_genericPoint);
}

const pcl::PointXYZINormal & PCLPoint::getPointXYZINormal() const
{
    if(m_type != ito::pclXYZINormal) throw pcl::PCLException("point is not of type PointXYZINormal",__FILE__,"getPointXYZINormal",__LINE__);
    return *reinterpret_cast<pcl::PointXYZINormal*>(m_genericPoint);
}

const pcl::PointXYZRGBNormal & PCLPoint::getPointXYZRGBNormal() const
{
    if(m_type != ito::pclXYZRGBNormal) throw pcl::PCLException("point is not of type PointXYZRGBNormal",__FILE__,"getPointXYZRGBNormal",__LINE__);
    return *reinterpret_cast<pcl::PointXYZRGBNormal*>(m_genericPoint);
}


pcl::PointXYZ & PCLPoint::getPointXYZ()
{
    if(m_type != ito::pclXYZ) throw pcl::PCLException("point is not of type PointXYZ",__FILE__,"getPointXYZ",__LINE__);
    return *reinterpret_cast<pcl::PointXYZ*>(m_genericPoint);
}

pcl::PointXYZI & PCLPoint::getPointXYZI()
{
    if(m_type != ito::pclXYZI) throw pcl::PCLException("point is not of type PointXYZI",__FILE__,"getPointXYZI",__LINE__);
    return *reinterpret_cast<pcl::PointXYZI*>(m_genericPoint);
}

pcl::PointXYZRGBA & PCLPoint::getPointXYZRGBA()
{
    if(m_type != ito::pclXYZRGBA) throw pcl::PCLException("point is not of type PointXYZRGBA",__FILE__,"getPointXYZRGBA",__LINE__);
    return *reinterpret_cast<pcl::PointXYZRGBA*>(m_genericPoint);
}

pcl::PointNormal & PCLPoint::getPointXYZNormal()
{
    if(m_type != ito::pclXYZNormal) throw pcl::PCLException("point is not of type PointXYZNormal",__FILE__,"getPointXYZNormal",__LINE__);
    return *reinterpret_cast<pcl::PointNormal*>(m_genericPoint);
}

pcl::PointXYZINormal & PCLPoint::getPointXYZINormal()
{
    if(m_type != ito::pclXYZINormal) throw pcl::PCLException("point is not of type PointXYZINormal",__FILE__,"getPointXYZINormal",__LINE__);
    return *reinterpret_cast<pcl::PointXYZINormal*>(m_genericPoint);
}

pcl::PointXYZRGBNormal & PCLPoint::getPointXYZRGBNormal()
{
    if(m_type != ito::pclXYZRGBNormal) throw pcl::PCLException("point is not of type PointXYZRGBNormal",__FILE__,"getPointXYZRGBNormal",__LINE__);
    return *reinterpret_cast<pcl::PointXYZRGBNormal*>(m_genericPoint);
}


void PCLPoint::copyFromVoidPtrAndType(void* ptr, ito::tPCLPointType type)
{
    if(m_genericPoint)
    {
        //delete old point
        switch(m_type)
        {
            case ito::pclInvalid:
                break;
            case ito::pclXYZ:
                delete reinterpret_cast<pcl::PointXYZ*>(m_genericPoint);
                break;
            case ito::pclXYZI:
                delete reinterpret_cast<pcl::PointXYZI*>(m_genericPoint);
                break;
            case ito::pclXYZRGBA:
                delete reinterpret_cast<pcl::PointXYZRGBA*>(m_genericPoint);
                break;
            case ito::pclXYZNormal:
                delete reinterpret_cast<pcl::PointNormal*>(m_genericPoint);
                break;
            case ito::pclXYZINormal:
                delete reinterpret_cast<pcl::PointXYZINormal*>(m_genericPoint);
                break;
            case ito::pclXYZRGBNormal:
                delete reinterpret_cast<pcl::PointXYZRGBNormal*>(m_genericPoint);
                break;
        }
        m_genericPoint = NULL;
    }

    m_type = type;
    switch(type)
    {
        case ito::pclXYZ:
            m_genericPoint = reinterpret_cast<void*>( new pcl::PointXYZ( *reinterpret_cast<pcl::PointXYZ*>(ptr) ) );
            break;
        case ito::pclXYZI:
            m_genericPoint = reinterpret_cast<void*>( new pcl::PointXYZI( *reinterpret_cast<pcl::PointXYZI*>(ptr) ) );
            break;
        case ito::pclXYZRGBA:
            m_genericPoint = reinterpret_cast<void*>( new pcl::PointXYZRGBA( *reinterpret_cast<pcl::PointXYZRGBA*>(ptr) ) );
            break;
        case ito::pclXYZNormal:
            m_genericPoint = reinterpret_cast<void*>( new pcl::PointNormal( *reinterpret_cast<pcl::PointNormal*>(ptr) ) );
            break;
        case ito::pclXYZINormal:
            m_genericPoint = reinterpret_cast<void*>( new pcl::PointXYZINormal( *reinterpret_cast<pcl::PointXYZINormal*>(ptr) ) );
            break;
        case ito::pclXYZRGBNormal:
            m_genericPoint = reinterpret_cast<void*>( new pcl::PointXYZRGBNormal( *reinterpret_cast<pcl::PointXYZRGBNormal*>(ptr) ) );
            break;
        case ito::pclInvalid:
            m_genericPoint = NULL;
            break;
    }
}

bool PCLPoint::getXYZ(float &x, float &y, float &z)
{
    switch(m_type)
    {
        case ito::pclXYZ:
            {
                pcl::PointXYZ &p = getPointXYZ();
                x = p.x; y = p.y; z = p.z;
                return true;
            }
        case ito::pclXYZI:
            {
                pcl::PointXYZI &p = getPointXYZI();
                x = p.x; y = p.y; z = p.z;
                return true;
            }
        case ito::pclXYZRGBA:
            {
                pcl::PointXYZRGBA &p = getPointXYZRGBA();
                x = p.x; y = p.y; z = p.z;
                return true;
            }
        case ito::pclXYZNormal:
            {
                pcl::PointNormal &p = getPointXYZNormal();
                x = p.x; y = p.y; z = p.z;
                return true;
            }
        case ito::pclXYZINormal:
            {
                pcl::PointXYZINormal &p = getPointXYZINormal();
                x = p.x; y = p.y; z = p.z;
                return true;
            }
        case ito::pclXYZRGBNormal:
            {
                pcl::PointXYZRGBNormal &p = getPointXYZRGBNormal();
                x = p.x; y = p.y; z = p.z;
                return true;
            }
        default:
            x=y=z = 0.0;
            return false;
    }
    return false;
}

bool PCLPoint::setXYZ(float x, float y, float z, int mask)
{
    switch(m_type)
    {
        case ito::pclXYZ:
            {
                pcl::PointXYZ &p = getPointXYZ();
                if(mask & 0x01) p.x = x;
                if(mask & 0x02) p.y = y;
                if(mask & 0x04) p.z = z;
                return true;
            }
        case ito::pclXYZI:
            {
                pcl::PointXYZI &p = getPointXYZI();
                if(mask & 0x01) p.x = x;
                if(mask & 0x02) p.y = y;
                if(mask & 0x04) p.z = z;
                return true;
            }
        case ito::pclXYZRGBA:
            {
                pcl::PointXYZRGBA &p = getPointXYZRGBA();
                if(mask & 0x01) p.x = x;
                if(mask & 0x02) p.y = y;
                if(mask & 0x04) p.z = z;
                return true;
            }
        case ito::pclXYZNormal:
            {
                pcl::PointNormal &p = getPointXYZNormal();
                if(mask & 0x01) p.x = x;
                if(mask & 0x02) p.y = y;
                if(mask & 0x04) p.z = z;
                return true;
            }
        case ito::pclXYZINormal:
            {
                pcl::PointXYZINormal &p = getPointXYZINormal();
                if(mask & 0x01) p.x = x;
                if(mask & 0x02) p.y = y;
                if(mask & 0x04) p.z = z;
                return true;
            }
        case ito::pclXYZRGBNormal:
            {
                pcl::PointXYZRGBNormal &p = getPointXYZRGBNormal();
                if(mask & 0x01) p.x = x;
                if(mask & 0x02) p.y = y;
                if(mask & 0x04) p.z = z;
                return true;
            }
        default:
            return false;
    }
    return false;
}

bool PCLPoint::getNormal(float &nx, float &ny, float &nz)
{
    switch(m_type)
    {
        case ito::pclXYZNormal:
            {
                pcl::PointNormal &p = getPointXYZNormal();
                nx = p.normal_x; ny = p.normal_y; nz = p.normal_z;
                return true;
            }
        case ito::pclXYZINormal:
            {
                pcl::PointXYZINormal &p = getPointXYZINormal();
                nx = p.normal_x; ny = p.normal_y; nz = p.normal_z;
                return true;
            }
        case ito::pclXYZRGBNormal:
            {
                pcl::PointXYZRGBNormal &p = getPointXYZRGBNormal();
                nx = p.normal_x; ny = p.normal_y; nz = p.normal_z;
                return true;
            }
        default:
            nx=ny=nz = 0.0;
            return false;
    }
    return false;
}

bool PCLPoint::setNormal(float nx, float ny, float nz, int mask)
{
    switch(m_type)
    {
        case ito::pclXYZNormal:
            {
                pcl::PointNormal &p = getPointXYZNormal();
                if(mask & 0x01) p.normal_x = nx;
                if(mask & 0x02) p.normal_y = ny;
                if(mask & 0x04) p.normal_z = nz;
                return true;
            }
        case ito::pclXYZINormal:
            {
                pcl::PointXYZINormal &p = getPointXYZINormal();
                if(mask & 0x01) p.normal_x = nx;
                if(mask & 0x02) p.normal_y = ny;
                if(mask & 0x04) p.normal_z = nz;
                return true;
            }
        case ito::pclXYZRGBNormal:
            {
                pcl::PointXYZRGBNormal &p = getPointXYZRGBNormal();
                if(mask & 0x01) p.normal_x = nx;
                if(mask & 0x02) p.normal_y = ny;
                if(mask & 0x04) p.normal_z = nz;
                return true;
            }
        default:
            return false;
    }
    return false;
}

bool PCLPoint::getRGBA(uint8_t &r, uint8_t &g, uint8_t &b, uint8_t &a)
{
    switch(m_type)
    {
        case ito::pclXYZRGBA:
            {
                pcl::PointXYZRGBA &p = getPointXYZRGBA();
                r = p.r; g = p.g ; b = p.b ; a = p.PCLALPHA;
                return true;
            }
        case ito::pclXYZRGBNormal:
            {
                pcl::PointXYZRGBNormal &p = getPointXYZRGBNormal();
                r = p.r; g = p.g ; b = p.b ; a = p.PCLALPHA;
                return true;
            }
        default:
            r=g=b=a = 0;
            return false;
    }
    return false;
}

bool PCLPoint::setRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a, int mask)
{
    switch(m_type)
    {
        case ito::pclXYZRGBA:
            {
                pcl::PointXYZRGBA &p = getPointXYZRGBA();
                if(mask & 0x01) p.r = r;
                if(mask & 0x02) p.g = g;
                if(mask & 0x04) p.b = b;
                if(mask & 0x08) p.PCLALPHA = a;
                return true;
            }
        case ito::pclXYZRGBNormal:
            {
                pcl::PointXYZRGBNormal &p = getPointXYZRGBNormal();
                if(mask & 0x01) p.r = r;
                if(mask & 0x02) p.g = g;
                if(mask & 0x04) p.b = b;
                if(mask & 0x08) p.PCLALPHA = a;
                return true;
            }
        default:
            return false;
    }
    return false;
}

bool PCLPoint::getIntensity(float &intensity)
{
    switch(m_type)
    {
        case ito::pclXYZI:
            intensity = getPointXYZI().intensity;
            return true;
        case ito::pclXYZINormal:
            intensity = getPointXYZINormal().intensity;
            return true;
        default:
            intensity = 0;
            return false;
    }
    return false;
}

bool PCLPoint::setIntensity(float intensity)
{
    switch(m_type)
    {
        case ito::pclXYZI:
            getPointXYZI().intensity = intensity;
            return true;
        case ito::pclXYZINormal:
            getPointXYZINormal().intensity = intensity;
            return true;
        default:
            return false;
    }
    return false;
}

bool PCLPoint::getCurvature(float &curvature)
{
    switch(m_type)
    {
        case ito::pclXYZNormal:
            curvature = getPointXYZNormal().curvature;
            return true;
        case ito::pclXYZINormal:
            curvature = getPointXYZINormal().curvature;
            return true;
        case ito::pclXYZRGBNormal:
            curvature = getPointXYZRGBNormal().curvature;
            return true;
        default:
            curvature = 0;
            return false;
    }
    return false;
}

bool PCLPoint::setCurvature(float curvature)
{
    switch(m_type)
    {
        case ito::pclXYZNormal:
            getPointXYZNormal().curvature = curvature;
            return true;
        case ito::pclXYZINormal:
            getPointXYZINormal().curvature = curvature;
            return true;
        case ito::pclXYZRGBNormal:
            getPointXYZRGBNormal().curvature = curvature;
            return true;
        default:
            return false;
    }
    return false;
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> pcl::PointCloud<_Tp>* getPointCloudPtrInternal(ito::PCLPointCloud &pc) { return NULL; }
template<> pcl::PointCloud<pcl::PointXYZ>* getPointCloudPtrInternal(ito::PCLPointCloud &pc) { return pc.m_pcXYZ.get(); }
template<> pcl::PointCloud<pcl::PointXYZI>* getPointCloudPtrInternal(ito::PCLPointCloud &pc) { return pc.m_pcXYZI.get(); }
template<> pcl::PointCloud<pcl::PointXYZRGBA>* getPointCloudPtrInternal(ito::PCLPointCloud &pc) { return pc.m_pcXYZRGBA.get(); }
template<> pcl::PointCloud<pcl::PointNormal>* getPointCloudPtrInternal(ito::PCLPointCloud &pc) { return pc.m_pcXYZNormal.get(); }
template<> pcl::PointCloud<pcl::PointXYZINormal>* getPointCloudPtrInternal(ito::PCLPointCloud &pc) { return pc.m_pcXYZINormal.get(); }
template<> pcl::PointCloud<pcl::PointXYZRGBNormal>* getPointCloudPtrInternal(ito::PCLPointCloud &pc) { return pc.m_pcXYZRGBNormal.get(); }

template<typename _Tp> const pcl::PointCloud<_Tp>* getPointCloudPtrInternal(const ito::PCLPointCloud &pc) { return NULL; }
template<> const pcl::PointCloud<pcl::PointXYZ>* getPointCloudPtrInternal(const ito::PCLPointCloud &pc) { return pc.m_pcXYZ.get(); }
template<> const pcl::PointCloud<pcl::PointXYZI>* getPointCloudPtrInternal(const ito::PCLPointCloud &pc) { return pc.m_pcXYZI.get(); }
template<> const pcl::PointCloud<pcl::PointXYZRGBA>* getPointCloudPtrInternal(const ito::PCLPointCloud &pc) { return pc.m_pcXYZRGBA.get(); }
template<> const pcl::PointCloud<pcl::PointNormal>* getPointCloudPtrInternal(const ito::PCLPointCloud &pc) { return pc.m_pcXYZNormal.get(); }
template<> const pcl::PointCloud<pcl::PointXYZINormal>* getPointCloudPtrInternal(const ito::PCLPointCloud &pc) { return pc.m_pcXYZINormal.get(); }
template<> const pcl::PointCloud<pcl::PointXYZRGBNormal>* getPointCloudPtrInternal(const ito::PCLPointCloud &pc) { return pc.m_pcXYZRGBNormal.get(); }

template<typename _Tp> _Tp* getPointPtrInternal(ito::PCLPoint &point) { return NULL; }
template<> pcl::PointXYZ* getPointPtrInternal(ito::PCLPoint &point) { return &point.getPointXYZ(); }
template<> pcl::PointXYZI* getPointPtrInternal(ito::PCLPoint &point) { return &point.getPointXYZI(); }
template<> pcl::PointXYZRGBA* getPointPtrInternal(ito::PCLPoint &point) { return &point.getPointXYZRGBA(); }
template<> pcl::PointNormal* getPointPtrInternal(ito::PCLPoint &point) { return &point.getPointXYZNormal(); }
template<> pcl::PointXYZINormal* getPointPtrInternal(ito::PCLPoint &point) { return &point.getPointXYZINormal(); }
template<> pcl::PointXYZRGBNormal* getPointPtrInternal(ito::PCLPoint &point) { return &point.getPointXYZRGBNormal(); }

template<typename _Tp> const _Tp* getPointPtrInternal(const ito::PCLPoint &point) { return NULL; }
template<> const pcl::PointXYZ* getPointPtrInternal(const ito::PCLPoint &point) { return &point.getPointXYZ(); }
template<> const pcl::PointXYZI* getPointPtrInternal(const ito::PCLPoint &point) { return &point.getPointXYZI(); }
template<> const pcl::PointXYZRGBA* getPointPtrInternal(const ito::PCLPoint &point) { return &point.getPointXYZRGBA(); }
template<> const pcl::PointNormal* getPointPtrInternal(const ito::PCLPoint &point) { return &point.getPointXYZNormal(); }
template<> const pcl::PointXYZINormal* getPointPtrInternal(const ito::PCLPoint &point) { return &point.getPointXYZINormal(); }
template<> const pcl::PointXYZRGBNormal* getPointPtrInternal(const ito::PCLPoint &point) { return &point.getPointXYZRGBNormal(); }


//-------------------------------------------------------------------------------------------------
PCLPointCloud::PCLPointCloud(uint32_t width_, uint32_t height_, ito::tPCLPointType type_,  const PCLPoint &value_)
{
    m_type = type_;

    if (value_.getType() == ito::pclInvalid)
    {
        switch(type_)
        {
            case ito::pclXYZ:
                m_pcXYZ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>(width_,height_));
                break;
            case ito::pclXYZI:
                m_pcXYZI = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>(width_,height_));
                break;
            case ito::pclXYZRGBA:
                m_pcXYZRGBA = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>(width_,height_));
                break;
            case ito::pclXYZNormal:
                m_pcXYZNormal = pcl::PointCloud<pcl::PointNormal>::Ptr(new pcl::PointCloud<pcl::PointNormal>(width_,height_));
                break;
            case ito::pclXYZINormal:
                m_pcXYZINormal = pcl::PointCloud<pcl::PointXYZINormal>::Ptr(new pcl::PointCloud<pcl::PointXYZINormal>(width_,height_));
                break;
            case ito::pclXYZRGBNormal:
                m_pcXYZRGBNormal = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>(width_,height_));
                break;
            default:
                break;
        }
    }
    else
    {
        switch(type_)
        {
            case ito::pclXYZ:
                m_pcXYZ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>(width_,height_,value_.getPointXYZ()));
                break;
            case ito::pclXYZI:
                m_pcXYZI = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>(width_,height_,value_.getPointXYZI()));
                break;
            case ito::pclXYZRGBA:
                m_pcXYZRGBA = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>(width_,height_,value_.getPointXYZRGBA()));
                break;
            case ito::pclXYZNormal:
                m_pcXYZNormal = pcl::PointCloud<pcl::PointNormal>::Ptr(new pcl::PointCloud<pcl::PointNormal>(width_,height_,value_.getPointXYZNormal()));
                break;
            case ito::pclXYZINormal:
                m_pcXYZINormal = pcl::PointCloud<pcl::PointXYZINormal>::Ptr(new pcl::PointCloud<pcl::PointXYZINormal>(width_,height_,value_.getPointXYZINormal()));
                break;
            case ito::pclXYZRGBNormal:
                m_pcXYZRGBNormal = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>(width_,height_,value_.getPointXYZRGBNormal()));
                break;
            default:
                break;
        }
    }
}

//---------------------------------------------------------------------------------------------------------------
PCLPointCloud::PCLPointCloud (PCLPointCloud &pc)
{
    m_type = pc.m_type;
    switch(pc.m_type)
    {
        case ito::pclXYZ:
            m_pcXYZ = pc.toPointXYZ();
            break;
        case ito::pclXYZI:
            m_pcXYZI = pc.toPointXYZI();
            break;
        case ito::pclXYZRGBA:
            m_pcXYZRGBA = pc.toPointXYZRGBA();
            break;
        case ito::pclXYZNormal:
            m_pcXYZNormal = pc.toPointXYZNormal();
            break;
        case ito::pclXYZINormal:
            m_pcXYZINormal = pc.toPointXYZINormal();
            break;
        case ito::pclXYZRGBNormal:
            m_pcXYZRGBNormal = pc.toPointXYZRGBNormal();
            break;
        default:
            break;
    }
}

//---------------------------------------------------------------------------------------------------------------
PCLPointCloud::PCLPointCloud (const PCLPointCloud &pc)
{
    m_type = pc.m_type;
    switch(pc.m_type)
    {
        case ito::pclXYZ:
            m_pcXYZ = pc.toPointXYZ();
            break;
        case ito::pclXYZI:
            m_pcXYZI = pc.toPointXYZI();
            break;
        case ito::pclXYZRGBA:
            m_pcXYZRGBA = pc.toPointXYZRGBA();
            break;
        case ito::pclXYZNormal:
            m_pcXYZNormal = pc.toPointXYZNormal();
            break;
        case ito::pclXYZINormal:
            m_pcXYZINormal = pc.toPointXYZINormal();
            break;
        case ito::pclXYZRGBNormal:
            m_pcXYZRGBNormal = pc.toPointXYZRGBNormal();
            break;
        default:
            break;
    }
}

//---------------------------------------------------------------------------------------------------------------
PCLPointCloud::PCLPointCloud (const PCLPointCloud &pc, const std::vector< int > &indices) :
    m_type(ito::pclInvalid)
{
    size_t size = pc.size();
    if (indices.size() > size)
    {
        throw pcl::PCLException("indices vector is longer than the number of points in the given point cloud",__FILE__, "PCLPointCloud", __LINE__);
    }

    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (indices[i] < 0 || indices[i] >= size)
        {
            throw pcl::PCLException("indices vector contain invalid values.",__FILE__, "PCLPointCloud", __LINE__);
        }
    }

    m_type = pc.m_type;
    switch(pc.m_type)
    {
        case ito::pclXYZ:
            m_pcXYZ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>(*pc.toPointXYZ(), indices));
            break;
        case ito::pclXYZI:
            m_pcXYZI = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>(*pc.toPointXYZI(), indices));
            break;
        case ito::pclXYZRGBA:
            m_pcXYZRGBA = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>(*pc.toPointXYZRGBA(), indices));
            break;
        case ito::pclXYZNormal:
            m_pcXYZNormal = pcl::PointCloud<pcl::PointNormal>::Ptr(new pcl::PointCloud<pcl::PointNormal>(*pc.toPointXYZNormal(), indices));
            break;
        case ito::pclXYZINormal:
            m_pcXYZINormal = pcl::PointCloud<pcl::PointXYZINormal>::Ptr(new pcl::PointCloud<pcl::PointXYZINormal>(*pc.toPointXYZINormal(), indices));
            break;
        case ito::pclXYZRGBNormal:
            m_pcXYZRGBNormal = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>(*pc.toPointXYZRGBNormal(), indices));
            break;
    }
}

//---------------------------------------------------------------------------------------------------------------
//! make a deep copy of this point cloud
PCLPointCloud PCLPointCloud::copy() const
{
    //this is an inefficient way to do a deep-copy
    std::vector<int> indices;
    indices.resize(size());
    for (size_t i = 0; i < size(); ++i)
    {
        indices[i] = i;
    }

    return PCLPointCloud(*this, indices);
}

//---------------------------------------------------------------------------------------------------------------
void PCLPointCloud::setInvalid()
{
    createEmptyPointCloud(ito::pclInvalid);
}

//---------------------------------------------------------------------------------------------------------------
void PCLPointCloud::createEmptyPointCloud(ito::tPCLPointType type)
{
    //clear existing data
    switch(m_type)
    {
        case ito::pclXYZ: m_pcXYZ = pcl::PointCloud<pcl::PointXYZ>::Ptr(); break;
        case ito::pclXYZI: m_pcXYZI = pcl::PointCloud<pcl::PointXYZI>::Ptr(); break;
        case ito::pclXYZRGBA: m_pcXYZRGBA = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(); break;
        case ito::pclXYZNormal: m_pcXYZNormal = pcl::PointCloud<pcl::PointNormal>::Ptr(); break;
        case ito::pclXYZINormal: m_pcXYZINormal = pcl::PointCloud<pcl::PointXYZINormal>::Ptr(); break;
        case ito::pclXYZRGBNormal: m_pcXYZRGBNormal = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(); break;
        default: break;
    }

    m_type = type;
    switch(type)
    {
    case ito::pclXYZ:
        m_pcXYZ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>()); break;
    case ito::pclXYZI:
        m_pcXYZI = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>()); break;
    case ito::pclXYZRGBA:
        m_pcXYZRGBA = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>()); break;
    case ito::pclXYZNormal:
        m_pcXYZNormal = pcl::PointCloud<pcl::PointNormal>::Ptr(new pcl::PointCloud<pcl::PointNormal>()); break;
    case ito::pclXYZINormal:
        m_pcXYZINormal = pcl::PointCloud<pcl::PointXYZINormal>::Ptr(new pcl::PointCloud<pcl::PointXYZINormal>()); break;
    case ito::pclXYZRGBNormal:
        m_pcXYZRGBNormal = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>()); break;
    default:
        m_type = ito::pclInvalid; break;
    }
}



//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> bool IsOrganizedFunc(const ito::PCLPointCloud *pc)
{
   const pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
   if(temp)
   {
       return temp->isOrganized();
   }
   throw pcl::PCLException("shared pointer is NULL",__FILE__, "IsOrganized", __LINE__);
}

typedef bool (*tIsOrganizedFunc)(const ito::PCLPointCloud *pc);
PCLMAKEFUNCLIST(IsOrganizedFunc)

bool PCLPointCloud::isOrganized() const
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListIsOrganizedFunc[idx](this);
    throw pcl::PCLException("invalid point cloud",__FILE__, "isOrganized", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> void SetWidthFunc(ito::PCLPointCloud *pc, uint32_t width)
{
   pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
   if(temp)
   {
       temp->width = width;
       return;
   }
   throw pcl::PCLException("shared pointer is NULL",__FILE__, "set_width", __LINE__);
}

typedef void (*tSetWidthFunc)(ito::PCLPointCloud *pc, uint32_t width);
PCLMAKEFUNCLIST(SetWidthFunc)

void PCLPointCloud::set_width(uint32_t width)
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListSetWidthFunc[idx](this, width);
    throw pcl::PCLException("invalid point cloud",__FILE__, "set_width", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> uint32_t GetWidthFunc(const ito::PCLPointCloud *pc)
{
   const pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
   if(temp)
   {
       return temp->width;
   }
   throw pcl::PCLException("shared pointer is NULL",__FILE__, "width", __LINE__);
}

typedef uint32_t (*tGetWidthFunc)(const ito::PCLPointCloud *pc);
PCLMAKEFUNCLIST(GetWidthFunc)

uint32_t PCLPointCloud::width() const
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListGetWidthFunc[idx](this);
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> size_t GetSizeFunc(const ito::PCLPointCloud *pc)
{
   const pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
   if(temp)
   {
       return temp->size();
   }
   throw pcl::PCLException("shared pointer is NULL",__FILE__, "size", __LINE__);
}

typedef size_t (*tGetSizeFunc)(const ito::PCLPointCloud *pc);
PCLMAKEFUNCLIST(GetSizeFunc)

size_t PCLPointCloud::size() const
{
    int idx = getFuncListIndex();
    if(idx >= 0)
        return fListGetSizeFunc[idx](this);
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> uint32_t GetHeightFunc(const ito::PCLPointCloud *pc)
{
   const pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
   if(temp)
   {
       return temp->height;
   }
   throw pcl::PCLException("shared pointer is NULL",__FILE__, "height", __LINE__);
}

typedef uint32_t (*tGetHeightFunc)(const ito::PCLPointCloud *pc);
PCLMAKEFUNCLIST(GetHeightFunc)

uint32_t PCLPointCloud::height() const
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListGetHeightFunc[idx](this);
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> void SetHeightFunc(ito::PCLPointCloud *pc, uint32_t height)
{
   pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
   if(temp)
   {
       temp->height = height;
       return;
   }
   throw pcl::PCLException("shared pointer is NULL",__FILE__, "set_height", __LINE__);
}

typedef void (*tSetHeightFunc)(ito::PCLPointCloud *pc, uint32_t height);
PCLMAKEFUNCLIST(SetHeightFunc)

void PCLPointCloud::set_height(uint32_t height)
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListSetHeightFunc[idx](this, height);
    throw pcl::PCLException("invalid point cloud",__FILE__, "set_height", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> bool GetDenseFunc(const ito::PCLPointCloud *pc)
{
   const pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
   if(temp)
   {
       return temp->is_dense;
   }
   throw pcl::PCLException("shared pointer is NULL",__FILE__, "is_dense", __LINE__);
}

typedef bool (*tGetDenseFunc)(const ito::PCLPointCloud *pc);
PCLMAKEFUNCLIST(GetDenseFunc)

bool PCLPointCloud::is_dense() const
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListGetDenseFunc[idx](this);
    throw pcl::PCLException("invalid point cloud",__FILE__, "is_dense", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> void SetDenseFunc(ito::PCLPointCloud *pc, bool dense)
{
   pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
   if(temp)
   {
       temp->is_dense = dense;
       return;
   }
   throw pcl::PCLException("shared pointer is NULL",__FILE__, "set_dense", __LINE__);
}

typedef void (*tSetDenseFunc)(ito::PCLPointCloud *pc, bool dense);
PCLMAKEFUNCLIST(SetDenseFunc)
void PCLPointCloud::set_dense(bool dense)
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListSetDenseFunc[idx](this, dense);
    throw pcl::PCLException("invalid point cloud",__FILE__, "set_dense", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> void ScaleXYZFunc(ito::PCLPointCloud *pc, ito::float32 scaleX, ito::float32 scaleY, ito::float32 scaleZ)
{
   pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
   if(temp)
   {
       for (int i = 0; i < temp->points.size(); ++i)
       {
           temp->points[i].x *= scaleX;
           temp->points[i].y *= scaleY;
           temp->points[i].z *= scaleZ;
       }
       return;
   }
   throw pcl::PCLException("shared pointer is NULL",__FILE__, "scaleXYZ", __LINE__);
}

typedef void(*tScaleXYZFunc)(ito::PCLPointCloud *pc, ito::float32 scaleX, ito::float32 scaleY, ito::float32 scaleZ);
PCLMAKEFUNCLIST(ScaleXYZFunc)
void PCLPointCloud::scaleXYZ(float32 scaleX, float32 scaleY, float32 scaleZ)
{
    int idx = getFuncListIndex();
    if (idx >= 0)    return fListScaleXYZFunc[idx](this, scaleX, scaleY, scaleZ);
    throw pcl::PCLException("invalid point cloud",__FILE__, "scaleXYZ", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> void MoveXYZFunc(ito::PCLPointCloud *pc, ito::float32 dX, ito::float32 dY, ito::float32 dZ)
{
   pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
   if(temp)
   {
       for (int i = 0; i < temp->points.size(); ++i)
       {
           temp->points[i].x += dX;
           temp->points[i].y += dY;
           temp->points[i].z += dZ;
       }
       return;
   }
   throw pcl::PCLException("shared pointer is NULL",__FILE__, "scaleXYZ", __LINE__);
}

typedef void(*tMoveXYZFunc)(ito::PCLPointCloud *pc, ito::float32 dX, ito::float32 dY, ito::float32 dZ);
PCLMAKEFUNCLIST(MoveXYZFunc)
void PCLPointCloud::moveXYZ(float32 dX, float32 dY, float32 dZ)
{
    int idx = getFuncListIndex();
    if (idx >= 0)    return fListMoveXYZFunc[idx](this, dX, dY, dZ);
    throw pcl::PCLException("invalid point cloud",__FILE__, "moveXYZ", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------
#if PCL_VERSION_COMPARE(>=,1,7,0)
    template<typename _Tp> pcl::PCLHeader GetHeaderFunc(const ito::PCLPointCloud *pc)
    {
       const pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
       if(temp)
       {
           return temp->header;
       }
       throw pcl::PCLException("shared pointer is NULL",__FILE__, "header", __LINE__);
    }

    typedef pcl::PCLHeader (*tGetHeaderFunc)(const ito::PCLPointCloud *pc);

    PCLMAKEFUNCLIST(GetHeaderFunc)

    pcl::PCLHeader PCLPointCloud::header() const
    {
        int idx = getFuncListIndex();
        if(idx >= 0)    return fListGetHeaderFunc[idx](this);
        throw pcl::PCLException("invalid point cloud",__FILE__, "header", __LINE__);
    }
#else
    template<typename _Tp> std_msgs::Header GetHeaderFunc(const ito::PCLPointCloud *pc)
    {
       const pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
       if(temp)
       {
           return temp->header;
       }
       throw pcl::PCLException("shared pointer is NULL",__FILE__, "header", __LINE__);
    }

    typedef std_msgs::Header (*tGetHeaderFunc)(const ito::PCLPointCloud *pc);

    PCLMAKEFUNCLIST(GetHeaderFunc)

    std_msgs::Header PCLPointCloud::header() const
    {
        int idx = getFuncListIndex();
        if(idx >= 0)    return fListGetHeaderFunc[idx](this);
        throw pcl::PCLException("invalid point cloud",__FILE__, "header", __LINE__);
    }
#endif





//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> std::string GetFieldsListFunc(const ito::PCLPointCloud *pc)
{
   const pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
   if(temp)
   {
       return pcl::getFieldsList(*temp);
   }
   throw pcl::PCLException("shared pointer is NULL",__FILE__, "header", __LINE__);
}

typedef std::string (*tGetFieldsListFunc)(const ito::PCLPointCloud *pc);
PCLMAKEFUNCLIST(GetFieldsListFunc)

std::string PCLPointCloud::getFieldsList() const
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListGetFieldsListFunc[idx](this);
    throw pcl::PCLException("invalid point cloud",__FILE__, "header", __LINE__);
}


//----------------------------------------------------------------------------------------------------------------------------------
#if PCL_VERSION_COMPARE(>=,1,7,0)
template<typename _Tp> std::vector<pcl::PCLPointField> GetFieldsInfoFunc(const ito::PCLPointCloud *pc)
{
   const pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
   if(temp)
   {
       std::vector<pcl::PCLPointField> fields;
       pcl::getFields<_Tp>(fields);
       return fields;
   }
   throw pcl::PCLException("shared pointer is NULL",__FILE__, "header", __LINE__);
}

typedef std::vector<pcl::PCLPointField> (*tGetFieldsInfoFunc)(const ito::PCLPointCloud *pc);
PCLMAKEFUNCLIST(GetFieldsInfoFunc)

std::vector<pcl::PCLPointField> PCLPointCloud::getFieldsInfo() const
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListGetFieldsInfoFunc[idx](this);
    throw pcl::PCLException("invalid point cloud",__FILE__, "header", __LINE__);
}
#else
template<typename _Tp> std::vector<sensor_msgs::PointField> GetFieldsInfoFunc(const ito::PCLPointCloud *pc)
{
   const pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
   if(temp)
   {
       std::vector<sensor_msgs::PointField> fields;
       pcl::getFields<_Tp>(fields);
       return fields;
   }
   throw pcl::PCLException("shared pointer is NULL",__FILE__, "header", __LINE__);
}

typedef std::vector<sensor_msgs::PointField> (*tGetFieldsInfoFunc)(const ito::PCLPointCloud *pc);
PCLMAKEFUNCLIST(GetFieldsInfoFunc)

std::vector<sensor_msgs::PointField> PCLPointCloud::getFieldsInfo() const
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListGetFieldsInfoFunc[idx](this);
    throw pcl::PCLException("invalid point cloud",__FILE__, "header", __LINE__);
}
#endif

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> unsigned char* GenericPointAccessFunc(const ito::PCLPointCloud *pc, size_t &strideBytes)
{
   const pcl::PointCloud<_Tp>* temp = getPointCloudPtrInternal<_Tp >(*pc);
   if(temp)
   {
       strideBytes = sizeof(_Tp);
       return (unsigned char*)(temp->points.data());
   }
   throw pcl::PCLException("shared pointer is NULL",__FILE__, "header", __LINE__);
}

typedef unsigned char* (*tGenericPointAccessFunc)(const ito::PCLPointCloud *pc, size_t &strideBytes);
PCLMAKEFUNCLIST(GenericPointAccessFunc)

unsigned char* PCLPointCloud::genericPointAccess(size_t &strideBytes) const
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListGenericPointAccessFunc[idx](this, strideBytes);
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> void PcAddFunc(ito::PCLPointCloud *pc1, const ito::PCLPointCloud *pc2, ito::PCLPointCloud *pcRes)
{
    pcl::PointCloud<_Tp>* pc1_ = getPointCloudPtrInternal<_Tp >(*pc1);
    const pcl::PointCloud<_Tp>* pc2_ = getPointCloudPtrInternal<_Tp >(*pc2);
    pcl::PointCloud<_Tp>* pcRes_ = getPointCloudPtrInternal<_Tp >(*pcRes);

    if(pc1_ == NULL || pc2_ == NULL || pcRes_ == NULL)
    {
        throw pcl::PCLException("shared pointer is NULL",__FILE__, "PcAddFunc", __LINE__);
    }
    *pcRes_ = *pc1_ + *pc2_;
}

typedef void (*tPcAddFunc)(ito::PCLPointCloud *pc1, const ito::PCLPointCloud *pc2, ito::PCLPointCloud *pcRes);
PCLMAKEFUNCLIST(PcAddFunc)

PCLPointCloud & PCLPointCloud::operator+= (const PCLPointCloud &rhs)
{
    if(getType() == ito::pclInvalid)
    {
        createEmptyPointCloud(rhs.getType());
    }

    if(getType() != rhs.getType())
    {
        throw pcl::PCLException("type of point clouds are not equal",__FILE__, "operator+=", __LINE__);
    }

    int idx = getFuncListIndex();
    if(idx >= 0)
    {
        fListPcAddFunc[idx](this, &rhs, this);
        return *this;
    }
    throw pcl::PCLException("invalid point cloud",__FILE__, "operator+=", __LINE__);
}

const PCLPointCloud PCLPointCloud::operator+ (const PCLPointCloud &rhs)
{
    if(getType() != ito::pclInvalid)
    {
        if(getType() != rhs.getType())
        {
            throw pcl::PCLException("type of point clouds are not equal",__FILE__, "operator+=", __LINE__);
        }
        PCLPointCloud result(getType());
        int idx = getFuncListIndex();
        if(idx >= 0)
        {
            fListPcAddFunc[idx](this, &rhs, &result);
            return result;
        }
    }
    else
    {
        const ito::tPCLPointType newType = rhs.getType();
        PCLPointCloud result(newType);
        PCLPointCloud temp(newType);
        int idx = getFuncListIndex(newType);
        if(idx >= 0)
        {
            fListPcAddFunc[idx](&temp, &rhs, &result);
            return result;
        }
    }
    throw pcl::PCLException("invalid point cloud",__FILE__, "operator+", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------


PCLPointCloud & PCLPointCloud::operator= (const PCLPointCloud &copy)
{
    if (this != &copy)
    {
        setInvalid();
        m_type = copy.getType();

        switch(copy.getType())
        {
            case ito::pclXYZ:
                m_pcXYZ = copy.m_pcXYZ;
                break;
            case ito::pclXYZI:
                m_pcXYZI = copy.m_pcXYZI;
                break;
            case ito::pclXYZRGBA:
                m_pcXYZRGBA = copy.m_pcXYZRGBA;
                break;
            case ito::pclXYZNormal:
                m_pcXYZNormal = copy.m_pcXYZNormal;
                break;
            case ito::pclXYZINormal:
                m_pcXYZINormal = copy.m_pcXYZINormal;
                break;
            case ito::pclXYZRGBNormal:
                m_pcXYZRGBNormal = copy.m_pcXYZRGBNormal;
                break;
        }
    }

    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> const ito::PCLPoint GetItemFunc(const ito::PCLPointCloud *pc, size_t n)
{
    const pcl::PointCloud<_Tp>* pc_ = getPointCloudPtrInternal<_Tp >(*pc);
    if(pc_ == NULL)
    {
        throw pcl::PCLException("shared pointer is NULL",__FILE__, "GetItemFunc", __LINE__);
    }
    const _Tp &point = pc_->at(n);
    return ito::PCLPoint(point);
}

typedef const ito::PCLPoint (*tGetItemFunc)(const ito::PCLPointCloud *pc, size_t n);
PCLMAKEFUNCLIST(GetItemFunc)

const ito::PCLPoint PCLPointCloud::operator[] (size_t n) const
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListGetItemFunc[idx](this, n);
    throw pcl::PCLException("invalid point cloud",__FILE__, "operator[]", __LINE__);
}

const ito::PCLPoint PCLPointCloud::at (size_t n) const
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListGetItemFunc[idx](this, n);
    throw pcl::PCLException("invalid point cloud",__FILE__, "at(size_t n)", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> void SetItemFunc(ito::PCLPointCloud * /*pc*/, size_t /*n*/, ito::PCLPoint & /*point*/)
{
    throw pcl::PCLException("not implemented",__FILE__,"SetItemFunc",__LINE__);
}

template<> void SetItemFunc<pcl::PointXYZ>(ito::PCLPointCloud *pc, size_t n, ito::PCLPoint &point)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr a = pc->m_pcXYZ;
    a->at(n) = point.getPointXYZ();
}

template<> void SetItemFunc<pcl::PointXYZI>(ito::PCLPointCloud *pc, size_t n, ito::PCLPoint &point)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr a = pc->m_pcXYZI;
    a->at(n) = point.getPointXYZI();
}

template<> void SetItemFunc<pcl::PointXYZRGBA>(ito::PCLPointCloud *pc, size_t n, ito::PCLPoint &point)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr a = pc->m_pcXYZRGBA;
    a->at(n) = point.getPointXYZRGBA();
}

template<> void SetItemFunc<pcl::PointNormal>(ito::PCLPointCloud *pc, size_t n, ito::PCLPoint &point)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr a = pc->m_pcXYZNormal;
    a->at(n) = point.getPointXYZNormal();
}

template<> void SetItemFunc<pcl::PointXYZINormal>(ito::PCLPointCloud *pc, size_t n, ito::PCLPoint &point)
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr a = pc->m_pcXYZINormal;
    a->at(n) = point.getPointXYZINormal();
}

template<> void SetItemFunc<pcl::PointXYZRGBNormal>(ito::PCLPointCloud *pc, size_t n, ito::PCLPoint &point)
{
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr a = pc->m_pcXYZRGBNormal;
    a->at(n) = point.getPointXYZRGBNormal();
}

typedef void (*tSetItemFunc)(ito::PCLPointCloud *pc, size_t n, ito::PCLPoint &point);
PCLMAKEFUNCLIST(SetItemFunc)


void PCLPointCloud::set_item(size_t n, PCLPoint &point)
{
    if(getType() != point.getType())
    {
        throw pcl::PCLException("point cloud and point must have the same type",__FILE__, "set_item", __LINE__);
    }

    int idx = getFuncListIndex();
    if(idx >= 0)
    {
        fListSetItemFunc[idx](this, n, point);
        return;
    }
    throw pcl::PCLException("invalid point cloud",__FILE__, "set_item", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> void PushBackFunc(ito::PCLPointCloud * /*pc*/, const ito::PCLPoint & /*point*/)
{
    throw pcl::PCLException("not implemented",__FILE__,"PushBackFunc",__LINE__);
}

template<> void PushBackFunc<pcl::PointXYZ>(ito::PCLPointCloud *pc, const ito::PCLPoint &point)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr a = pc->m_pcXYZ;
    a->push_back(point.getPointXYZ());
}

template<> void PushBackFunc<pcl::PointXYZI>(ito::PCLPointCloud *pc, const ito::PCLPoint &point)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr a = pc->m_pcXYZI;
    a->push_back(point.getPointXYZI());
}

template<> void PushBackFunc<pcl::PointXYZRGBA>(ito::PCLPointCloud *pc, const ito::PCLPoint &point)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr a = pc->m_pcXYZRGBA;
    a->push_back(point.getPointXYZRGBA());
}

template<> void PushBackFunc<pcl::PointNormal>(ito::PCLPointCloud *pc, const ito::PCLPoint &point)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr a = pc->m_pcXYZNormal;
    a->push_back(point.getPointXYZNormal());
}

template<> void PushBackFunc<pcl::PointXYZINormal>(ito::PCLPointCloud *pc, const ito::PCLPoint &point)
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr a = pc->m_pcXYZINormal;
    a->push_back(point.getPointXYZINormal());
}

template<> void PushBackFunc<pcl::PointXYZRGBNormal>(ito::PCLPointCloud *pc, const ito::PCLPoint &point)
{
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr a = pc->m_pcXYZRGBNormal;
    a->push_back(point.getPointXYZRGBNormal());
}

typedef void (*tPushBackFunc)(ito::PCLPointCloud *pc, const ito::PCLPoint &point);
PCLMAKEFUNCLIST(PushBackFunc)

void PCLPointCloud::push_back(const ito::PCLPoint &pt)
{
    if(getType() == ito::pclInvalid) createEmptyPointCloud(pt.getType());
    if(getType() != pt.getType())
    {
        throw pcl::PCLException("point cloud and point must have the same type",__FILE__, "push_back", __LINE__);
    }

    int idx = getFuncListIndex();
    if(idx >= 0)
    {
        fListPushBackFunc[idx](this, pt);
        return;
    }
    throw pcl::PCLException("invalid point cloud",__FILE__, "push_back", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> bool EmptyFunc(const ito::PCLPointCloud *pc)
{
    const pcl::PointCloud<_Tp>* pc_ = getPointCloudPtrInternal<_Tp >(*pc);
    if(pc_ == NULL)
    {
        throw pcl::PCLException("shared pointer is NULL",__FILE__, "EmptyFunc", __LINE__);
    }
    return pc_->empty();
}

typedef bool (*tEmptyFunc)(const ito::PCLPointCloud *pc);
PCLMAKEFUNCLIST(EmptyFunc)

bool PCLPointCloud::empty() const
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListEmptyFunc[idx](this);
    throw pcl::PCLException("invalid point cloud",__FILE__, "empty", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> void ReserveResizeFunc(ito::PCLPointCloud *pc, size_t n, bool reserveNotResize)
{
    pcl::PointCloud<_Tp>* pc_ = getPointCloudPtrInternal<_Tp >(*pc);
    if(pc_ == NULL)
    {
        throw pcl::PCLException("shared pointer is NULL",__FILE__, "ReserveResizeFunc", __LINE__);
    }

    if(reserveNotResize)
    {
        pc_->reserve(n);
    }
    else
    {
        pc_->resize(n);
    }
}

typedef void (*tReserveResizeFunc)(ito::PCLPointCloud *pc, size_t n, bool reserveNotResize);
PCLMAKEFUNCLIST(ReserveResizeFunc)
void PCLPointCloud::reserve(size_t n)
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListReserveResizeFunc[idx](this, n, true);
    throw pcl::PCLException("invalid point cloud",__FILE__, "reserve", __LINE__);
}

void PCLPointCloud::resize(size_t n)
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListReserveResizeFunc[idx](this, n, false);
    throw pcl::PCLException("invalid point cloud",__FILE__, "resize", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> void ClearFunc(ito::PCLPointCloud *pc)
{
    pcl::PointCloud<_Tp>* pc_ = getPointCloudPtrInternal<_Tp >(*pc);
    if(pc_ == NULL)
    {
        throw pcl::PCLException("shared pointer is NULL",__FILE__, "ClearFunc", __LINE__);
    }
    pc_->clear();
}

typedef void (*tClearFunc)(ito::PCLPointCloud *pc);
PCLMAKEFUNCLIST(ClearFunc)

void PCLPointCloud::clear()
{
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListClearFunc[idx](this);
    throw pcl::PCLException("invalid point cloud",__FILE__, "clear", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> void EraseFunc(ito::PCLPointCloud *pc, uint32_t startIndex, uint32_t endIndex) //endIndex is always one index after the last index to erase
{
    pcl::PointCloud<_Tp>* pc_ = getPointCloudPtrInternal<_Tp >(*pc);
    if(pc_ == NULL)
    {
        throw pcl::PCLException("shared pointer is NULL",__FILE__, "ClearFunc", __LINE__);
    }

//    if(startIndex < 0) startIndex = 0; // gcc complains startIndex is unsigned, so this line is sensless
    if(endIndex > pc_->size()) endIndex = pc_->size();

    pc_->erase( pc_->points.begin() + startIndex, pc_->points.begin() + endIndex);
}

typedef void (*tEraseFunc)(ito::PCLPointCloud *pc, uint32_t startIndex, uint32_t endIndex);
PCLMAKEFUNCLIST(EraseFunc)

void PCLPointCloud::erase(uint32_t startIndex, uint32_t endIndex)
{
    if(startIndex > endIndex) std::swap( startIndex, endIndex );
    int idx = getFuncListIndex();
    if(idx >= 0)    return fListEraseFunc[idx](this, startIndex, endIndex);
    throw pcl::PCLException("invalid point cloud",__FILE__, "erase", __LINE__);
}

//----------------------------------------------------------------------------------------------------------------------------------

template<typename _Tp> void InsertFunc(ito::PCLPointCloud * /*pc*/, uint32_t /*index*/, const ito::PCLPoint& /*point*/)
{
    throw pcl::PCLException("not implemented",__FILE__,"InsertFunc",__LINE__);
}

template<> void InsertFunc<pcl::PointXYZ>(ito::PCLPointCloud *pc, uint32_t index, const ito::PCLPoint& point)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr a = pc->m_pcXYZ;
//    if(index < 0 || index > a->size())
    if(index > a->size())
    {
        throw pcl::PCLException("index is out of bounds",__FILE__, "InsertFunc", __LINE__);
    }
    a->insert(a->points.begin() + index, point.getPointXYZ());
}

template<> void InsertFunc<pcl::PointXYZI>(ito::PCLPointCloud *pc, uint32_t index, const ito::PCLPoint& point)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr a = pc->m_pcXYZI;
//    if(index < 0 || index > a->size())
    if(index > a->size())
    {
        throw pcl::PCLException("index is out of bounds",__FILE__, "InsertFunc", __LINE__);
    }
    a->insert(a->points.begin() + index, point.getPointXYZI());
}

template<> void InsertFunc<pcl::PointXYZRGBA>(ito::PCLPointCloud *pc, uint32_t index, const ito::PCLPoint& point)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr a = pc->m_pcXYZRGBA;
//    if(index < 0 || index > a->size())
    if(index > a->size())
    {
        throw pcl::PCLException("index is out of bounds",__FILE__, "InsertFunc", __LINE__);
    }
    a->insert(a->points.begin() + index, point.getPointXYZRGBA());
}

template<> void InsertFunc<pcl::PointNormal>(ito::PCLPointCloud *pc, uint32_t index, const ito::PCLPoint& point)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr a = pc->m_pcXYZNormal;
//    if(index < 0 || index > a->size())
    if(index > a->size())
    {
        throw pcl::PCLException("index is out of bounds",__FILE__, "InsertFunc", __LINE__);
    }
    a->insert(a->points.begin() + index, point.getPointXYZNormal());
}

template<> void InsertFunc<pcl::PointXYZINormal>(ito::PCLPointCloud *pc, uint32_t index, const ito::PCLPoint& point)
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr a = pc->m_pcXYZINormal;
//    if(index < 0 || index > a->size())
    if(index > a->size())
    {
        throw pcl::PCLException("index is out of bounds",__FILE__, "InsertFunc", __LINE__);
    }
    a->insert(a->points.begin() + index, point.getPointXYZINormal());
}

template<> void InsertFunc<pcl::PointXYZRGBNormal>(ito::PCLPointCloud *pc, uint32_t index, const ito::PCLPoint& point)
{
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr a = pc->m_pcXYZRGBNormal;
//    if(index < 0 || index > a->size())
    if(index > a->size())
    {
        throw pcl::PCLException("index is out of bounds",__FILE__, "InsertFunc", __LINE__);
    }
    a->insert(a->points.begin() + index, point.getPointXYZRGBNormal());
}

typedef void (*tInsertFunc)(ito::PCLPointCloud *pc, uint32_t index, const ito::PCLPoint& point);
PCLMAKEFUNCLIST(InsertFunc)

void PCLPointCloud::insert(uint32_t index, const ito::PCLPoint& point)
{
    if(getType() == ito::pclInvalid) createEmptyPointCloud(point.getType());
    if(getType() != point.getType())
    {
        throw pcl::PCLException("point cloud and point must have the same type",__FILE__, "push_back", __LINE__);
    }

    int idx = getFuncListIndex();
    if(idx >= 0)    return fListInsertFunc[idx](this, index, point);
    throw pcl::PCLException("invalid point cloud",__FILE__, "insert", __LINE__);
}




//-----------------------------------------------------------------------------------------------------------------------------------------------------

PCLPolygonMesh::PCLPolygonMesh() : m_valid(false)
{
    m_polygonMesh = pcl::PolygonMesh::Ptr(); //new pcl::PolygonMesh());
}

PCLPolygonMesh::PCLPolygonMesh(pcl::PolygonMesh::Ptr polygonMesh) : m_valid(true), m_polygonMesh(polygonMesh)
{
    if(polygonMesh.get() == NULL)
    {
        m_valid = false;
    }
}

PCLPolygonMesh::PCLPolygonMesh(PCLPolygonMesh &mesh)
{
    m_valid = mesh.m_valid;
    m_polygonMesh = mesh.m_polygonMesh;
}

PCLPolygonMesh::PCLPolygonMesh(const PCLPolygonMesh &mesh)
{
    m_valid = mesh.m_valid;
    m_polygonMesh = mesh.m_polygonMesh;
}

PCLPolygonMesh::PCLPolygonMesh(PCLPolygonMesh &mesh, const std::vector<uint32_t> &polygonIndices)
{
    m_valid = mesh.m_valid;
    m_polygonMesh = pcl::PolygonMesh::Ptr(new pcl::PolygonMesh());

    pcl::PolygonMesh *input = mesh.m_polygonMesh.get();
    pcl::PolygonMesh *output = m_polygonMesh.get();

    if(m_valid && input && output && input->polygons.size() > 0)
    {
        //bucket for polygons (1: index in polygonIndices -> take it into output, 0: ignore it)
        uint8_t *bucket = (uint8_t*)malloc( input->polygons.size() * sizeof(uint8_t) );
        memset(bucket, 0, input->polygons.size() * sizeof(uint8_t) );
        uint32_t maxIdx = input->polygons.size() - 1;

        for(size_t idx = 0 ; idx < polygonIndices.size(); idx++)
        {
            if( polygonIndices[idx] <= maxIdx )
            {
                bucket[ polygonIndices[idx] ] ++;
            }
        }

        //the following algorithm has been taken from pcl::surface::SimplificationRemoveUnusedVertices::simplify (version 1.6.0)
        unsigned int nr_points = input->cloud.width * input->cloud.height;

        std::vector<int> new_indices (nr_points, -1);
        std::vector<int> indices;
        indices.reserve (nr_points);

        // mark all points in triangles as being used
        for (size_t p = 0; p < input->polygons.size (); ++p)
        {
            if (bucket[p] > 0)
            {
                for (size_t point = 0; point < input->polygons[p].vertices.size (); ++point)
                {
                    if (new_indices[ input->polygons[p].vertices[point] ] == -1)
                    {
                        new_indices[ input->polygons[p].vertices[point]] = static_cast<int> (indices.size ());
                        indices.push_back (input->polygons[p].vertices[point]);
                    }
                }
            }
        }

        // copy cloud information
        output->header = input->header;
        output->cloud.data.clear ();
        output->cloud.header = input->cloud.header;
        output->cloud.fields = input->cloud.fields;
        output->cloud.row_step = input->cloud.row_step;
        output->cloud.point_step = input->cloud.point_step;
        output->cloud.is_bigendian = input->cloud.is_bigendian;
        output->cloud.height = 1; // cloud is no longer organized
        output->cloud.width = static_cast<int> (indices.size ());
        output->cloud.row_step = output->cloud.point_step * output->cloud.width;
        output->cloud.data.resize (output->cloud.width * output->cloud.height * output->cloud.point_step);
        output->cloud.is_dense = false;
        output->polygons.clear ();

        // copy (only!) used points
        for (size_t i = 0; i < indices.size (); ++i)
        {
            memcpy (&output->cloud.data[i * output->cloud.point_step], &input->cloud.data[indices[i] * output->cloud.point_step], output->cloud.point_step);
        }

        // copy mesh information (and update indices)
        output->polygons.reserve (input->polygons.size ());
        for (size_t p = 0; p < input->polygons.size (); ++p)
        {
            if (bucket[p] > 0)
            {
                pcl::Vertices corrected_polygon;
                corrected_polygon.vertices.resize (input->polygons[p].vertices.size () );
                for (size_t point = 0; point < input->polygons[p].vertices.size(); ++point)
                {
                    corrected_polygon.vertices[point] = new_indices[input->polygons[p].vertices[point]];
                }
                output->polygons.push_back (corrected_polygon);
            }
        }

        free(bucket);
        bucket = NULL;
    }
}

PCLPolygonMesh::PCLPolygonMesh(const PCLPointCloud &cloud, const std::vector<pcl::Vertices> &polygons) :
    m_valid(true)
{
    m_polygonMesh = pcl::PolygonMesh::Ptr(new pcl::PolygonMesh());
    m_polygonMesh->header = cloud.header();
    m_polygonMesh->polygons = polygons;
#if PCL_VERSION_COMPARE(>=,1,7,0)
    pcl::PCLPointCloud2 msg;
#else
    sensor_msgs::PointCloud2 msg;
#endif
    ito::pclHelper::pclPointCloudToPointCloud2(cloud, msg);
    m_polygonMesh->cloud = msg;
}

PCLPolygonMesh::~PCLPolygonMesh()
{
    m_polygonMesh.reset();
}


PCLPolygonMesh & PCLPolygonMesh::operator= (const PCLPolygonMesh &copy)
{
    m_valid = copy.m_valid;
    m_polygonMesh = copy.m_polygonMesh;
    return *this;
}

size_t PCLPolygonMesh::height() const
{
    pcl::PolygonMesh *mesh = m_polygonMesh.get();
    if(mesh == NULL)
    {
        return 0;
    }
    return mesh->cloud.height;
}

size_t PCLPolygonMesh::width() const
{
    pcl::PolygonMesh *mesh = m_polygonMesh.get();
    if(mesh == NULL)
    {
        return 0;
    }
    return mesh->cloud.width;
}

std::string PCLPolygonMesh::getFieldsList() const
{
    pcl::PolygonMesh *mesh = m_polygonMesh.get();
    if(mesh == NULL)
    {
        return "";
    }
#if PCL_VERSION_COMPARE(>=,1,7,0)
    std::vector< pcl::PCLPointField> fields = mesh->cloud.fields;
#else
    std::vector< ::sensor_msgs::PointField> fields = mesh->cloud.fields;
#endif
    std::string output;
    for(int i = 0; i < ((int)(fields.size())-1); i++)
    {
        output += fields[i].name + ";";
    }

    if(fields.size() > 0)
    {
        output += fields[ (int)(fields.size()) - 1 ].name;
    }

    return output;
}

std::ostream& PCLPolygonMesh::streamOut(std::ostream& out)
{
    pcl::PolygonMesh *mesh = m_polygonMesh.get();
    if(m_valid && mesh)
    {
        /*out << "header: " << std::endl;
        out << mesh->header;*/
#if PCL_VERSION_COMPARE(>=,1,7,0)
        pcl::PCLPointCloud2 *c = &(mesh->cloud);
        pcl::PCLPointField *f;
#else
        sensor_msgs::PointCloud2 *c = &(mesh->cloud);
        sensor_msgs::PointField *f;
#endif
        out << "points:\n------------\n" << std::endl;
        out << " size: [" << c->height << " x " << c->width << "]\n" << std::endl;
        if(c->is_bigendian)
        {
            out << " big_endian: true\n" << std::endl;
        }
        else
        {
            out << " big_endian: false\n" << std::endl;
        }
        if(c->is_dense)
        {
            out << " dense: true\n" << std::endl;
        }
        else
        {
            out << " dense: false\n" << std::endl;
        }
        out << " \nfields[]:\n" << std::endl;
        for (size_t i = 0; i < c->fields.size (); ++i)
        {
            f = &(c->fields[i]);
            switch(f->datatype)
            {
            case 1:
                out << " [" << i << "]: '" << f->name <<  "' int8\n" << std::endl;
                break;
            case 2:
                out << " [" << i << "]: '" << f->name <<  "' uint8\n" << std::endl;
                break;
            case 3:
                out << " [" << i << "]: '" << f->name <<  "' int16\n" << std::endl;
                break;
            case 4:
                out << " [" << i << "]: '" << f->name <<  "' uint16\n" << std::endl;
                break;
            case 5:
                out << " [" << i << "]: '" << f->name <<  "' int32\n" << std::endl;
                break;
            case 6:
                out << " [" << i << "]: '" << f->name <<  "' uint32\n" << std::endl;
                break;
            case 7:
                out << " [" << i << "]: '" << f->name <<  "' float32\n" << std::endl;
                break;
            case 8:
                out << " [" << i << "]: '" << f->name <<  "' float64\n" << std::endl;
                break;
            }
        }

        out << "\npoints[]:\n" << std::endl;
        uint32_t counter = 0;
        uint8_t *ptr = &(c->data[0]);

        for(uint32_t h = 0 ; h < c->height ; h++)
        {
            for(uint32_t w = 0 ; w < c->width ; w++)
            {
                out << " " << counter << " [" << h << "," << w << "]: ";

                for(size_t i = 0; i < c->fields.size (); ++i)
                {
                    f = &(c->fields[i]);

                    switch(f->datatype)
                    {
                    case 1:
                        out << (ito::int8)(*(ito::int8*)(ptr+f->offset)) << " ";
                        break;
                    case 2:
                        out << (ito::uint8)(*(ito::uint8*)(ptr+f->offset)) << " ";
                        break;
                    case 3:
                        out << (ito::uint16)(*(ito::uint16*)(ptr+f->offset)) << " ";
                        break;
                    case 4:
                        out << (ito::int16)(*(ito::int16*)(ptr+f->offset)) << " ";
                        break;
                    case 5:
                        out << (ito::int32)(*(ito::int32*)(ptr+f->offset)) << " ";
                        break;
                    case 6:
                        out << (ito::uint32)(*(ito::uint32*)(ptr+f->offset)) << " ";
                        break;
                    case 7:
                        out << (ito::float32)(*(ito::float32*)(ptr+f->offset)) << " ";
                        break;
                    case 8:
                        out << (ito::float64)(*(ito::float64*)(ptr+f->offset)) << " ";
                        break;
                    }
                }
                ptr += c->point_step;
                out << "\n" << std::endl;
                counter++;
            }
        }

        out << "\npolygons\n------------\n" << std::endl;
        for (size_t i = 0; i < mesh->polygons.size (); ++i)
        {
          out << " polygon[" << i << "]: ";
          for( size_t j = 0 ; j < mesh->polygons[i].vertices.size(); j++)
          {
              out << mesh->polygons[i].vertices[j] << " ";
          }
          out << "\n" << std::endl; //<< std::endl;
        }
    }
    return out;
}

} //end namespace ito
