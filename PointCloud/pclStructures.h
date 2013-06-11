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

#ifndef PCLSTRUCTURES_H
#define PCLSTRUCTURES_H


#include "../common/typeDefs.h"

#include <vector>

#ifndef linux
#pragma warning( disable: 4996) //supress deprecated warning of pcl (which occur very often)
#endif
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#ifndef linux
#pragma warning( default: 4996) //show 4996 warnings again
#endif



namespace ito
{

#if PCL_VERSION_COMPARE(>,1,5,1)
	#define PCLALPHA a
#else
	#define PCLALPHA _unused
#endif

class PCLPoint
{
public:
    PCLPoint() : m_genericPoint(NULL), m_type(ito::pclInvalid) {}

    PCLPoint(ito::tPCLPointType type) : m_genericPoint(NULL), m_type(type)
    {
        switch(m_type)
        {
            case ito::pclXYZ:
                m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZ());
                break;
            case ito::pclXYZI:
                m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZI());
                break;
            case ito::pclXYZRGBA:
                m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZRGBA());
                break;
            case ito::pclXYZNormal:
                m_genericPoint = reinterpret_cast<void*>(new pcl::PointNormal());
                break;
            case ito::pclXYZINormal:
                m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZINormal());
                break;
            case ito::pclXYZRGBNormal:
                m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZRGBNormal());
                break;
            default:
                //m_genericPoint = NULL;
                break;
        }
    }
    void copyFromVoidPtrAndType(void* ptr, ito::tPCLPointType type);

    PCLPoint(const pcl::PointXYZ &point) : m_genericPoint(NULL), m_type(ito::pclXYZ) 
    { 
        m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZ(point));
    }

    PCLPoint(const pcl::PointXYZI &point) : m_genericPoint(NULL), m_type(ito::pclXYZI) 
    { 
        m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZI(point));
    }

    PCLPoint(const pcl::PointXYZRGBA &point) : m_genericPoint(NULL), m_type(ito::pclXYZRGBA) 
    { 
        m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZRGBA(point));
    }

    PCLPoint(const pcl::PointNormal &point) : m_genericPoint(NULL), m_type(ito::pclXYZNormal) 
    { 
        m_genericPoint = reinterpret_cast<void*>(new pcl::PointNormal(point));
    }

    PCLPoint(const pcl::PointXYZINormal &point) : m_genericPoint(NULL), m_type(ito::pclXYZINormal) 
    { 
        m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZINormal(point));
    }

    PCLPoint(const pcl::PointXYZRGBNormal &point) : m_genericPoint(NULL), m_type(ito::pclXYZRGBNormal) 
    { 
        m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZRGBNormal(point));
    }

    PCLPoint ( const PCLPoint & p ) : m_genericPoint(NULL), m_type(ito::pclInvalid)
    {
        copyFromVoidPtrAndType( p.m_genericPoint, p.m_type);
    }

    PCLPoint & operator= ( const PCLPoint & p )
    {
        copyFromVoidPtrAndType( p.m_genericPoint, p.m_type);
        return *this;
    }

    ~PCLPoint() 
    {
        if(m_genericPoint)
        {
            switch(m_type)
            {
                case ito::pclXYZ: delete reinterpret_cast<pcl::PointXYZ*>(m_genericPoint); break;
                case ito::pclXYZI: delete reinterpret_cast<pcl::PointXYZI*>(m_genericPoint); break;
                case ito::pclXYZRGBA: delete reinterpret_cast<pcl::PointXYZRGBA*>(m_genericPoint); break;
                case ito::pclXYZNormal: delete reinterpret_cast<pcl::PointNormal*>(m_genericPoint); break;
                case ito::pclXYZINormal: delete reinterpret_cast<pcl::PointXYZINormal*>(m_genericPoint); break;
                case ito::pclXYZRGBNormal: delete reinterpret_cast<pcl::PointXYZRGBNormal*>(m_genericPoint); break;
                default: break;
            }
            m_genericPoint = NULL;
        }
    }

    inline ito::tPCLPointType getType() const { return m_type; }

    const pcl::PointXYZ & getPointXYZ() const;
    const pcl::PointXYZI & getPointXYZI() const;
    const pcl::PointXYZRGBA & getPointXYZRGBA() const;
    const pcl::PointNormal & getPointXYZNormal() const;
    const pcl::PointXYZINormal & getPointXYZINormal() const;
    const pcl::PointXYZRGBNormal & getPointXYZRGBNormal() const;

    pcl::PointXYZ & getPointXYZ();
    pcl::PointXYZI & getPointXYZI();
    pcl::PointXYZRGBA & getPointXYZRGBA();
    pcl::PointNormal & getPointXYZNormal();
    pcl::PointXYZINormal & getPointXYZINormal();
    pcl::PointXYZRGBNormal & getPointXYZRGBNormal();

    bool getXYZ(float &x, float &y, float &z);
    bool setXYZ(float x, float y, float z, int mask = 0xFFFF);
    bool getNormal(float &nx, float &ny, float &nz);
    bool setNormal(float nx, float ny, float nz, int mask = 0xFFFF);
    bool getRGBA(uint8_t &r, uint8_t &g, uint8_t &b, uint8_t &a);
    bool setRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a, int mask = 0xFFFF);
    bool getIntensity(float &intensity);
    bool setIntensity(float intensity);
    bool getCurvature(float &curvature);
    bool setCurvature(float curvature);

    
private:
    template<typename _Tp> friend _Tp* getPointPtrInternal(ito::PCLPoint &point);
    template<typename _Tp> friend const _Tp* getPointPtrInternal(const ito::PCLPoint &point);

    void *m_genericPoint;
    ito::tPCLPointType m_type;
};

class PCLPointCloud
{
public:
    PCLPointCloud() : m_type(ito::pclInvalid) {};
    PCLPointCloud(ito::tPCLPointType type) : m_type(type) 
    {
        createEmptyPointCloud(type);
    };
    PCLPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pclPtr) : m_pcXYZ(pclPtr), m_type(ito::pclXYZ) {};
    PCLPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pclPtr) : m_pcXYZI(pclPtr), m_type(ito::pclXYZI) {};
    PCLPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pclPtr) : m_pcXYZRGBA(pclPtr), m_type(ito::pclXYZRGBA) {};
    PCLPointCloud(pcl::PointCloud<pcl::PointNormal>::Ptr pclPtr) : m_pcXYZNormal(pclPtr), m_type(ito::pclXYZNormal) {};
    PCLPointCloud(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pclPtr) : m_pcXYZINormal(pclPtr), m_type(ito::pclXYZINormal) {};
    PCLPointCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclPtr) : m_pcXYZRGBNormal(pclPtr), m_type(ito::pclXYZRGBNormal) {};

    PCLPointCloud(uint32_t width_, uint32_t height_, ito::tPCLPointType type_, const PCLPoint &value_ = PCLPoint());
    PCLPointCloud (PCLPointCloud &pc);
 	PCLPointCloud (const PCLPointCloud &pc);
 	PCLPointCloud (const PCLPointCloud &pc, const std::vector< int > &indices);

    ~PCLPointCloud(){};

    inline ito::tPCLPointType getType() const 
    { 
    return m_type; 
    };

    inline pcl::PointCloud<pcl::PointXYZ>::Ptr toPointXYZ() const                   
    { 
        if(m_type == ito::pclXYZ) return m_pcXYZ;
        throw pcl::PCLException("point cloud has not the desired type PointXYZ",__FILE__, "toPointXYZ", __LINE__);
    };
    inline pcl::PointCloud<pcl::PointXYZI>::Ptr toPointXYZI() const                 
    { 
        if(m_type == ito::pclXYZI) return m_pcXYZI;
        throw pcl::PCLException("point cloud has not the desired type PointXYZI",__FILE__, "toPointXYZI", __LINE__);
    };
    inline pcl::PointCloud<pcl::PointXYZRGBA>::Ptr toPointXYZRGBA() const           
    { 
        if(m_type == ito::pclXYZRGBA) return m_pcXYZRGBA;
        throw pcl::PCLException("point cloud has not the desired type PointXYZRGBA",__FILE__, "toPointXYZRGBA", __LINE__);
    };
    inline pcl::PointCloud<pcl::PointNormal>::Ptr toPointXYZNormal() const          
    { 
        if(m_type == ito::pclXYZNormal) return m_pcXYZNormal;
        throw pcl::PCLException("point cloud has not the desired type PointXYZNormal",__FILE__, "toPointXYZNormal", __LINE__);
    };
    inline pcl::PointCloud<pcl::PointXYZINormal>::Ptr toPointXYZINormal() const     
    { 
        if(m_type == ito::pclXYZINormal) return m_pcXYZINormal;
        throw pcl::PCLException("point cloud has not the desired type PointXYZINormal",__FILE__, "toPointXYZINormal", __LINE__);
    };
    inline pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr toPointXYZRGBNormal() const 
    { 
        if(m_type == ito::pclXYZRGBNormal) return m_pcXYZRGBNormal;
        throw pcl::PCLException("point cloud has not the desired type PointXYZRGBNormal",__FILE__, "toPointXYZRGBNormal", __LINE__);
    };

    PCLPointCloud & operator+= (const PCLPointCloud &rhs);
    const PCLPointCloud operator+ (const PCLPointCloud &rhs);
    PCLPointCloud & operator= (const PCLPointCloud &copy);

    const PCLPoint operator[] (size_t n) const;
    const PCLPoint at (size_t n) const;
    
    void set_item(size_t n, PCLPoint &point);

    bool isOrganized() const;
    uint32_t width() const;
    uint32_t height() const;
    size_t size() const;
    bool is_dense() const;
    void set_width(uint32_t width);
    void set_height(uint32_t height);
    void set_dense(bool dense);
    std_msgs::Header header() const;

    std::string getFieldsList() const;

    void push_back(const ito::PCLPoint &pt);
    bool empty() const;
    void reserve(size_t n);
    void resize(size_t n);
    void clear();
    void erase(uint32_t startIndex, uint32_t endIndex);
    void insert(uint32_t index, const ito::PCLPoint& point);

protected:
    void setInvalid();
    void createEmptyPointCloud(ito::tPCLPointType type);
    
private:

    inline int getFuncListIndex() const
    {
        switch(m_type)
        {
        case ito::pclXYZ: return 0;
        case ito::pclXYZI: return 1;
        case ito::pclXYZRGBA: return 2;
        case ito::pclXYZNormal: return 3;
        case ito::pclXYZINormal: return 4;
        case ito::pclXYZRGBNormal: return 5;
        default: return -1;
        }
    };

    inline int getFuncListIndex(const ito::tPCLPointType &type) const
    {
        switch(type)
        {
        case ito::pclXYZ: return 0;
        case ito::pclXYZI: return 1;
        case ito::pclXYZRGBA: return 2;
        case ito::pclXYZNormal: return 3;
        case ito::pclXYZINormal: return 4;
        case ito::pclXYZRGBNormal: return 5;
        default: return -1;
        }
    };

    template<typename _Tp> friend pcl::PointCloud<_Tp>* getPointCloudPtrInternal(ito::PCLPointCloud &pc);
    template<typename _Tp> friend const pcl::PointCloud<_Tp>* getPointCloudPtrInternal(const ito::PCLPointCloud &pc);
    template<typename _Tp> friend std_msgs::Header GetHeaderFunc(ito::PCLPointCloud &pc);
    template<typename _Tp> friend uint32_t GetWidthFunc(const ito::PCLPointCloud *pc);
    template<typename _Tp> friend uint32_t GetHeightFunc(const ito::PCLPointCloud *pc);
    template<typename _Tp> friend void SetHeightFunc(ito::PCLPointCloud *pc, uint32_t height);
    template<typename _Tp> friend bool GetDenseFunc(const ito::PCLPointCloud *pc);
    template<typename _Tp> friend void SetDenseFunc(ito::PCLPointCloud *pc, bool dense);
    template<typename _Tp> friend std_msgs::Header GetHeaderFunc(const ito::PCLPointCloud *pc);
    template<typename _Tp> friend void SetWidthFunc(ito::PCLPointCloud *pc, uint32_t width);
    template<typename _Tp> friend void PcAddFunc(ito::PCLPointCloud *pc1, const ito::PCLPointCloud *pc2, ito::PCLPointCloud *pcRes);
    template<typename _Tp> friend void SetItemFunc(ito::PCLPointCloud *pc, size_t n, ito::PCLPoint &point);
    template<typename _Tp> friend void PushBackFunc(ito::PCLPointCloud * pc, const ito::PCLPoint & point);
    template<typename _Tp> friend bool EmptyFunc(const ito::PCLPointCloud *pc);
    template<typename _Tp> friend void ReserveResizeFunc(ito::PCLPointCloud *pc, size_t n, bool reserveNotResize);
    template<typename _Tp> friend void ClearFunc(ito::PCLPointCloud *pc);
    template<typename _Tp> friend void EraseFunc(ito::PCLPointCloud *pc, uint32_t startIndex, uint32_t endIndex);
    template<typename _Tp> friend void InsertFunc(ito::PCLPointCloud *pc, uint32_t index, const ito::PCLPoint& point);
    template<typename _Tp> friend std::string GetFieldsListFunc(const ito::PCLPointCloud *pc);


    pcl::PointCloud<pcl::PointXYZ>::Ptr m_pcXYZ;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_pcXYZI;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr m_pcXYZRGBA;
    pcl::PointCloud<pcl::PointNormal>::Ptr m_pcXYZNormal;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr m_pcXYZINormal;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr m_pcXYZRGBNormal;

    ito::tPCLPointType m_type;
};

class PCLPolygonMesh
{
public:
    PCLPolygonMesh();
    PCLPolygonMesh(pcl::PolygonMesh::Ptr polygonMesh);
    PCLPolygonMesh(PCLPolygonMesh &mesh);
    PCLPolygonMesh(PCLPolygonMesh &mesh, const std::vector<uint32_t> &polygonIndices);
    PCLPolygonMesh(const PCLPointCloud &cloud, const std::vector<pcl::Vertices> &polygons);
    PCLPolygonMesh(const PCLPolygonMesh &mesh);
    ~PCLPolygonMesh();

    inline pcl::PolygonMesh::Ptr polygonMesh() { return m_polygonMesh; }
    inline void setPolygonMesh(pcl::PolygonMesh::Ptr &mesh) { m_polygonMesh = mesh; }
    PCLPolygonMesh & operator= (const PCLPolygonMesh &copy);

    inline bool valid() { return m_valid; }

    size_t height() const;
    size_t width() const;
    std::string getFieldsList() const;

    std::ostream& streamOut(std::ostream& out);
    
protected:

private:
    bool m_valid;
    pcl::PolygonMesh::Ptr m_polygonMesh;
};

} //end namespace ito




#endif
