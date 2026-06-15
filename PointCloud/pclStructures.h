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

#ifndef PCLSTRUCTURES_H
#define PCLSTRUCTURES_H

#include "pclDefines.h"
#include "../common/typeDefs.h"

#include <vector>

#ifdef WIN32
#pragma warning( disable: 4996) //supress deprecated warning of pcl (which occur very often)
#endif
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#ifdef WIN32
#pragma warning( default: 4996) //show 4996 warnings again
#endif



namespace ito
{

#if PCL_VERSION_COMPARE(>,1,5,1)
    #define PCLALPHA a
#else
    #define PCLALPHA _unused
#endif

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class PCLPoint
    \brief generic class that covers one single point of different possible types provided by the Point Cloud Library (PCL).

    The possible types are compatible to PCLPointCloud and described in ito::tPCLPointType:

    * ito::pclXYZ: x, y, z
    * ito::pclXYZI: x, y, z, intensity
    * ito::pclXYZRGBA: x, y, z, color (red, green, blue, alpha)
    * ito::pclXYZNormal: x, y, z, normal_x, normal_y, normal_z, curvature
    * ito::pclXYZINormal
    * ito::pclXYZRGBNormal

    The specific type is saved in the member m_type whereas m_genericPoint points to the value.
*/
class POINTCLOUD_EXPORT PCLPoint
{
public:
    //! empty constructor creates invalid point type
    PCLPoint() : m_genericPoint(NULL), m_type(ito::pclInvalid) {}

    //! constructor with desired point type. The specific point is created but not initialized with desired values.
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

    //! helper function to copy the content of a given pointer representing a specific point of given type into this object.
    void copyFromVoidPtrAndType(void* ptr, ito::tPCLPointType type);

    //! copy constructor from point of PCL type pcl::PointXYZ
    PCLPoint(const pcl::PointXYZ &point) : m_genericPoint(NULL), m_type(ito::pclXYZ)
    {
        m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZ(point));
    }

    //! copy constructor from point of PCL type pcl::PointXYZI
    PCLPoint(const pcl::PointXYZI &point) : m_genericPoint(NULL), m_type(ito::pclXYZI)
    {
        m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZI(point));
    }

    //! copy constructor from point of PCL type pcl::PointXYZRGBA
    PCLPoint(const pcl::PointXYZRGBA &point) : m_genericPoint(NULL), m_type(ito::pclXYZRGBA)
    {
        m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZRGBA(point));
    }

    //! copy constructor from point of PCL type pcl::PointNormal
    PCLPoint(const pcl::PointNormal &point) : m_genericPoint(NULL), m_type(ito::pclXYZNormal)
    {
        m_genericPoint = reinterpret_cast<void*>(new pcl::PointNormal(point));
    }

    //! copy constructor from point of PCL type pcl::PointXYZINormal
    PCLPoint(const pcl::PointXYZINormal &point) : m_genericPoint(NULL), m_type(ito::pclXYZINormal)
    {
        m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZINormal(point));
    }

    //! copy constructor from point of PCL type pcl::PointXYZRGBNormal
    PCLPoint(const pcl::PointXYZRGBNormal &point) : m_genericPoint(NULL), m_type(ito::pclXYZRGBNormal)
    {
        m_genericPoint = reinterpret_cast<void*>(new pcl::PointXYZRGBNormal(point));
    }

    //! copy constructor from another instance of PCLPoint
    PCLPoint ( const PCLPoint & p ) : m_genericPoint(NULL), m_type(ito::pclInvalid)
    {
        copyFromVoidPtrAndType( p.m_genericPoint, p.m_type);
    }

    //! assigns the given PCLPoint data to this object
    PCLPoint & operator= ( const PCLPoint & p )
    {
        copyFromVoidPtrAndType( p.m_genericPoint, p.m_type);
        return *this;
    }

    //! destructor
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

    //! returns type of covered point type or ito::pclInvalid if invalid point.
    inline ito::tPCLPointType getType() const { return m_type; }

    //! if this object covers a point of type ito::PointXYZ, this point is returned as pcl::PointXYZ object.
    /*!
    \throws pcl::PCLException if this point is not of type ito::PointXYZ
    */
    const pcl::PointXYZ & getPointXYZ() const;

    //! if this object covers a point of type ito::PointXYZI, this point is returned as pcl::PointXYZI object.
    /*!
    \throws pcl::PCLException if this point is not of type ito::PointXYZI
    */
    const pcl::PointXYZI & getPointXYZI() const;

    //! if this object covers a point of type ito::PointXYZRGBA, this point is returned as pcl::PointXYZRGBA object.
    /*!
    \throws pcl::PCLException if this point is not of type ito::PointXYZRGBA
    */
    const pcl::PointXYZRGBA & getPointXYZRGBA() const;

    //! if this object covers a point of type ito::PointXYZNormal, this point is returned as pcl::PointNormal object.
    /*!
    \throws pcl::PCLException if this point is not of type ito::PointXYZNormal
    */
    const pcl::PointNormal & getPointXYZNormal() const;

    //! if this object covers a point of type ito::PointXYZINormal, this point is returned as pcl::PointXYZINormal object.
    /*!
    \throws pcl::PCLException if this point is not of type ito::PointXYZINormal
    */
    const pcl::PointXYZINormal & getPointXYZINormal() const;

    //! if this object covers a point of type ito::PointXYZRGBNormal, this point is returned as pcl::PointXYZRGBNormal object.
    /*!
    \throws pcl::PCLException if this point is not of type ito::PointXYZRGBNormal
    */
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

    void *m_genericPoint;      /*!< generic pointer that holds an instance of the corresponding classes pcl::PointXYZ, pcl::PointNormal... (depending on m_type) */
    ito::tPCLPointType m_type; /*!< type covered by this object */
};

// Forward declaration of friend methods
#ifdef __APPLE__
    class PCLPointCloud;
    template<typename _Tp> pcl::PointCloud<_Tp>* getPointCloudPtrInternal(ito::PCLPointCloud &pc);
    template<typename _Tp> const pcl::PointCloud<_Tp>* getPointCloudPtrInternal(const ito::PCLPointCloud &pc);

#if PCL_VERSION_COMPARE(>=,1,7,0)
    template<typename _Tp> pcl::PCLHeader GetHeaderFunc(ito::PCLPointCloud &pc);
    template<typename _Tp> pcl::PCLHeader GetHeaderFunc(const ito::PCLPointCloud *pc);
#else
    template<typename _Tp> std_msgs::Header GetHeaderFunc(ito::PCLPointCloud &pc);
    template<typename _Tp> std_msgs::Header GetHeaderFunc(const ito::PCLPointCloud *pc);
#endif
    template<typename _Tp> uint32_t GetWidthFunc(const ito::PCLPointCloud *pc);
    template<typename _Tp> uint32_t GetHeightFunc(const ito::PCLPointCloud *pc);
    template<typename _Tp> void SetHeightFunc(ito::PCLPointCloud *pc, uint32_t height);
    template<typename _Tp> bool GetDenseFunc(const ito::PCLPointCloud *pc);
    template<typename _Tp> void SetDenseFunc(ito::PCLPointCloud *pc, bool dense);
    template<typename _Tp> void SetWidthFunc(ito::PCLPointCloud *pc, uint32_t width);
    template<typename _Tp> void PcAddFunc(ito::PCLPointCloud *pc1, const ito::PCLPointCloud *pc2, ito::PCLPointCloud *pcRes);
    template<typename _Tp> void SetItemFunc(ito::PCLPointCloud *pc, size_t n, ito::PCLPoint &point);
    template<typename _Tp> void PushBackFunc(ito::PCLPointCloud * pc, const ito::PCLPoint & point);
    template<typename _Tp> bool EmptyFunc(const ito::PCLPointCloud *pc);
    template<typename _Tp> void ReserveResizeFunc(ito::PCLPointCloud *pc, size_t n, bool reserveNotResize);
    template<typename _Tp> void ClearFunc(ito::PCLPointCloud *pc);
    template<typename _Tp> void EraseFunc(ito::PCLPointCloud *pc, uint32_t startIndex, uint32_t endIndex);
    template<typename _Tp> void InsertFunc(ito::PCLPointCloud *pc, uint32_t index, const ito::PCLPoint& point);
    template<typename _Tp> std::string GetFieldsListFunc(const ito::PCLPointCloud *pc);
#endif // __APPLE__

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class PCLPointCloud
    \brief generic class that covers one single point cloud of different possible types provided by the Point Cloud Library (PCL).

    The possible types are compatible to PCLPoint and described in ito::tPCLPointType:

    * ito::pclXYZ: x, y, z
    * ito::pclXYZI: x, y, z, intensity
    * ito::pclXYZRGBA: x, y, z, color (red, green, blue, alpha)
    * ito::pclXYZNormal: x, y, z, normal_x, normal_y, normal_z, curvature
    * ito::pclXYZINormal
    * ito::pclXYZRGBNormal

    The specific type is saved in the member m_type whereas different members are available each one holding a shared pointer to
    the point cloud of one specific type. Only one of those shared pointers can contain a valid point cloud.
*/
class POINTCLOUD_EXPORT PCLPointCloud
{
public:
    //! constructor for an empty, invalid point cloud
    PCLPointCloud() : m_type(ito::pclInvalid) {};

    //! constructor for an empty point cloud of the desired type
    PCLPointCloud(ito::tPCLPointType type) : m_type(type)
    {
        createEmptyPointCloud(type);
    };
    //! constructor from given shared pointer of pcl::PointCloud<pcl::PointXYZ>
    PCLPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pclPtr) : m_pcXYZ(pclPtr), m_type(ito::pclXYZ) {};

    //! constructor from given shared pointer of pcl::PointCloud<pcl::PointXYZI>
    PCLPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pclPtr) : m_pcXYZI(pclPtr), m_type(ito::pclXYZI) {};

    //! constructor from given shared pointer of pcl::PointCloud<pcl::PointXYZRGBA>
    PCLPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pclPtr) : m_pcXYZRGBA(pclPtr), m_type(ito::pclXYZRGBA) {};

    //! constructor from given shared pointer of pcl::PointCloud<pcl::PointNormal>
    PCLPointCloud(pcl::PointCloud<pcl::PointNormal>::Ptr pclPtr) : m_pcXYZNormal(pclPtr), m_type(ito::pclXYZNormal) {};

    //! constructor from given shared pointer of pcl::PointCloud<pcl::PointXYZINormal>
    PCLPointCloud(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pclPtr) : m_pcXYZINormal(pclPtr), m_type(ito::pclXYZINormal) {};

    //! constructor from given shared pointer of pcl::PointCloud<pcl::PointXYZRGBNormal>
    PCLPointCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclPtr) : m_pcXYZRGBNormal(pclPtr), m_type(ito::pclXYZRGBNormal) {};

    //! creates a point cloud with given width_, height_ and type_ and sets all points to the given value_.
    /*!
    \throws pcl::PCLException if type of value_ is not invalid but does not fit to desired type_
    */
    PCLPointCloud(uint32_t width_, uint32_t height_, ito::tPCLPointType type_, const PCLPoint &value_ = PCLPoint());

    //! copy constructor
    PCLPointCloud(PCLPointCloud &pc);

    //! copy constructor
    PCLPointCloud(const PCLPointCloud &pc);

    //! copy constructor that creates the point cloud from a given point cloud and a vector of point indices that should be copied only.
    /*!
    \throws pcl::PCLException if indices contain invalid values or are longer than the number of points in pc.
    */
    PCLPointCloud(const PCLPointCloud &pc, const std::vector< int > &indices);

    //! destructor
    ~PCLPointCloud(){};

    //! returns type of covered point cloud or ito::pclInvalid if invalid point cloud.
    inline ito::tPCLPointType getType() const
    {
        return m_type;
    };

    //! if this cloud has color components returns != 0, else 0
    inline int hasRGB() const { return m_type & (ito::pclXYZRGBNormal | ito::pclXYZRGBA); }

    //! if this cloud has the normal components returns != 0, else 0
    inline int hasNormal() const { return m_type & (ito::pclXYZINormal | ito::pclXYZNormal | ito::pclXYZRGBNormal); }

    //! if this cloud has the intensity component returns != 0, else 0
    inline int hasIntensity() const { return m_type & ( ito::pclXYZI | ito::pclXYZINormal ); }

    //! if this cloud has the curvature component returns != 0, else 0
    inline int hasCurvature() const { return m_type & (ito::pclXYZINormal | ito::pclXYZNormal | ito::pclXYZRGBNormal); }

    //! returns a shared pointer to the internal pcl::PointCloud<pcl::PointXYZ> cloud.
    /*!
    \throws pcl::PCLException if this point cloud does not contain the cloud of the desired type.
    */
    inline pcl::PointCloud<pcl::PointXYZ>::Ptr toPointXYZ() const
    {
        if(m_type == ito::pclXYZ) return m_pcXYZ;
        throw pcl::PCLException("point cloud has not the desired type PointXYZ",__FILE__, "toPointXYZ", __LINE__);
    };

    //! returns a shared pointer to the internal pcl::PointCloud<pcl::PointXYZI> cloud.
    /*!
    \throws pcl::PCLException if this point cloud does not contain the cloud of the desired type.
    */
    inline pcl::PointCloud<pcl::PointXYZI>::Ptr toPointXYZI() const
    {
        if(m_type == ito::pclXYZI) return m_pcXYZI;
        throw pcl::PCLException("point cloud has not the desired type PointXYZI",__FILE__, "toPointXYZI", __LINE__);
    };

    //! returns a shared pointer to the internal pcl::PointCloud<pcl::PointXYZRGBA> cloud.
    /*!
    \throws pcl::PCLException if this point cloud does not contain the cloud of the desired type.
    */
    inline pcl::PointCloud<pcl::PointXYZRGBA>::Ptr toPointXYZRGBA() const
    {
        if(m_type == ito::pclXYZRGBA) return m_pcXYZRGBA;
        throw pcl::PCLException("point cloud has not the desired type PointXYZRGBA",__FILE__, "toPointXYZRGBA", __LINE__);
    };

    //! returns a shared pointer to the internal pcl::PointCloud<pcl::PointNormal> cloud.
    /*!
    \throws pcl::PCLException if this point cloud does not contain the cloud of the desired type.
    */
    inline pcl::PointCloud<pcl::PointNormal>::Ptr toPointXYZNormal() const
    {
        if(m_type == ito::pclXYZNormal) return m_pcXYZNormal;
        throw pcl::PCLException("point cloud has not the desired type PointXYZNormal",__FILE__, "toPointXYZNormal", __LINE__);
    };

    //! returns a shared pointer to the internal pcl::PointCloud<pcl::PointXYZINormal> cloud.
    /*!
    \throws pcl::PCLException if this point cloud does not contain the cloud of the desired type.
    */
    inline pcl::PointCloud<pcl::PointXYZINormal>::Ptr toPointXYZINormal() const
    {
        if(m_type == ito::pclXYZINormal) return m_pcXYZINormal;
        throw pcl::PCLException("point cloud has not the desired type PointXYZINormal",__FILE__, "toPointXYZINormal", __LINE__);
    };

    //! returns a shared pointer to the internal pcl::PointCloud<pcl::PointXYZRGBNormal> cloud.
    /*!
    \throws pcl::PCLException if this point cloud does not contain the cloud of the desired type.
    */
    inline pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr toPointXYZRGBNormal() const
    {
        if(m_type == ito::pclXYZRGBNormal) return m_pcXYZRGBNormal;
        throw pcl::PCLException("point cloud has not the desired type PointXYZRGBNormal",__FILE__, "toPointXYZRGBNormal", __LINE__);
    };

    //! returns a constant shared pointer to the internal pcl::PointCloud<pcl::PointXYZ> cloud.
    /*!
    \throws pcl::PCLException if this point cloud does not contain the cloud of the desired type.
    */
    inline pcl::PointCloud<pcl::PointXYZ>::ConstPtr toPointXYZConst() const
    {
        if(m_type == ito::pclXYZ) return m_pcXYZ;
        throw pcl::PCLException("point cloud has not the desired type PointXYZ",__FILE__, "toPointXYZ", __LINE__);
    };

    //! returns a constant shared pointer to the internal pcl::PointCloud<pcl::PointXYZI> cloud.
    /*!
    \throws pcl::PCLException if this point cloud does not contain the cloud of the desired type.
    */
    inline pcl::PointCloud<pcl::PointXYZI>::ConstPtr toPointXYZIConst() const
    {
        if(m_type == ito::pclXYZI) return m_pcXYZI;
        throw pcl::PCLException("point cloud has not the desired type PointXYZI",__FILE__, "toPointXYZI", __LINE__);
    };

    //! returns a constant shared pointer to the internal pcl::PointCloud<pcl::PointXYZRGBA> cloud.
    /*!
    \throws pcl::PCLException if this point cloud does not contain the cloud of the desired type.
    */
    inline pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr toPointXYZRGBAConst() const
    {
        if(m_type == ito::pclXYZRGBA) return m_pcXYZRGBA;
        throw pcl::PCLException("point cloud has not the desired type PointXYZRGBA",__FILE__, "toPointXYZRGBA", __LINE__);
    };

    //! returns a constant shared pointer to the internal pcl::PointCloud<pcl::PointNormal> cloud.
    /*!
    \throws pcl::PCLException if this point cloud does not contain the cloud of the desired type.
    */
    inline pcl::PointCloud<pcl::PointNormal>::ConstPtr toPointXYZNormalConst() const
    {
        if(m_type == ito::pclXYZNormal) return m_pcXYZNormal;
        throw pcl::PCLException("point cloud has not the desired type PointXYZNormal",__FILE__, "toPointXYZNormal", __LINE__);
    };

    //! returns a constant shared pointer to the internal pcl::PointCloud<pcl::PointXYZINormal> cloud.
    /*!
    \throws pcl::PCLException if this point cloud does not contain the cloud of the desired type.
    */
    inline pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr toPointXYZINormalConst() const
    {
        if(m_type == ito::pclXYZINormal) return m_pcXYZINormal;
        throw pcl::PCLException("point cloud has not the desired type PointXYZINormal",__FILE__, "toPointXYZINormal", __LINE__);
    };

    //! returns a constant shared pointer to the internal pcl::PointCloud<pcl::PointXYZRGBNormal> cloud.
    /*!
    \throws pcl::PCLException if this point cloud does not contain the cloud of the desired type.
    */
    inline pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr toPointXYZRGBNormalConst() const
    {
        if(m_type == ito::pclXYZRGBNormal) return m_pcXYZRGBNormal;
        throw pcl::PCLException("point cloud has not the desired type PointXYZRGBNormal",__FILE__, "toPointXYZRGBNormal", __LINE__);
    };

    //! appends another point cloud to this point cloud
    /*!
    \throws pcl::PCLException if both point clouds have different types or one of both clouds is invalid.
    */
    PCLPointCloud & operator+= (const PCLPointCloud &rhs);

    //! returns a point cloud consisting of points of this and another point cloud.
    /*!
    \throws pcl::PCLException if both point clouds have different types or one of both clouds is invalid.
    */
    const PCLPointCloud operator+ (const PCLPointCloud &rhs);

    //! assigment operator
    PCLPointCloud & operator= (const PCLPointCloud &copy);

    //! make a deep copy of this point cloud
    PCLPointCloud copy() const;

    //! returns point at given position n.
    /*!
    \throws pcl::PCLException if cloud is invalid or n is out of bounds
    */
    const PCLPoint operator[] (size_t n) const;

    //! returns point at given position n.
    /*!
    \throws pcl::PCLException if cloud is invalid or n is out of bounds
    */
    const PCLPoint at (size_t n) const;

    //! sets the point at position n to the value given by point.
    /*!
    \throws pcl::PCLException if cloud is invalid, n is out of bounds or the type of point does not fit to this point cloud.
    */
    void set_item(size_t n, PCLPoint &point);

    //! returns true if this point cloud is organized, hence its height is > 1. In organized clouds adjacent points are pretended to be neighbours in the world, too.
    bool isOrganized() const;

    //! returns width of this point cloud
    uint32_t width() const;

    //! returns height of this point cloud (1 if not organized, else > 1)
    uint32_t height() const;

    //! returns the number of points in this point cloud (width() * height())
    size_t size() const;

    //! returns true if this point cloud only contains points with valid components, else at least one point with nan or inf component is present
    bool is_dense() const;

    //! setter to set the width of the point cloud
    void set_width(uint32_t width);

    //! setter to set the height of the point cloud
    void set_height(uint32_t height);

    //! setter to set the dense property of this point cloud
    void set_dense(bool dense);

    //! scale every point in this cloud by scaleX, scaleY and scaleZ in X, Y and Z direction. Other point properties are not influenced by this.
    void scaleXYZ(float32 scaleX, float32 scaleY, float32 scaleZ);

    //! scale every point in this cloud by scaleX, scaleY and scaleZ in X, Y and Z direction. Other point properties are not influenced by this.
    void moveXYZ(float32 dX, float32 dY, float32 dZ);

    //! returns the header structure of this point cloud for a conversion to the old pcl::PCLPointCloud2 structure used in the pcl::PolygonMesh.
#if PCL_VERSION_COMPARE(>=,1,7,0)
    pcl::PCLHeader header() const;
#else
    std_msgs::Header header() const;
#endif

    //! returns a space separated string with the names of all components in the point cloud, e.g. "x y z"
    std::string getFieldsList() const;

    //! returns vector with information about all fields contained in the specific point cloud (each info struct contains the name, offset, datatype, ... of any field)
#if PCL_VERSION_COMPARE(>=,1,7,0)
    std::vector<pcl::PCLPointField> getFieldsInfo() const;
#else
	std::vector<sensor_msgs::PointField> getFieldsInfo() const;
#endif

    //! returns the pointer to the first point in the current cloud or NULL if the cloud is invalid, strideBytes is the number of bytes to jump from one point to the next one.
    unsigned char* genericPointAccess(size_t &strideBytes) const;

    //! adds a new point to this cloud
    void push_back(const ito::PCLPoint &pt);

    //! returns true if point cloud is empty, else false
    bool empty() const;

    //! reserves a space of n elements for the point cloud without resizing it
    void reserve(size_t n);

    //! resizes the point cloud to n points. Each point is arbitrarily initialized
    void resize(size_t n);

    //! clears all points in the point cloud
    void clear();

    //! clears the points from [startIndex, endIndex]
    void erase(uint32_t startIndex, uint32_t endIndex);

    //! inserts point at given index
    void insert(uint32_t index, const ito::PCLPoint& point);

protected:
    //! clears this point cloud and sets the type to ito::pclInvalid. An existing point cloud is implicitely deleted.
    void setInvalid();

    //! creates an empty point cloud of given type. An existing point cloud is implicitely deleted.
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

#if PCL_VERSION_COMPARE(>=,1,7,0)
    template<typename _Tp> friend pcl::PCLHeader GetHeaderFunc(ito::PCLPointCloud &pc);
    template<typename _Tp> friend pcl::PCLHeader GetHeaderFunc(const ito::PCLPointCloud *pc);
#else
    template<typename _Tp> friend std_msgs::Header GetHeaderFunc(ito::PCLPointCloud &pc);
    template<typename _Tp> friend std_msgs::Header GetHeaderFunc(const ito::PCLPointCloud *pc);
#endif
    template<typename _Tp> friend uint32_t GetWidthFunc(const ito::PCLPointCloud *pc);
    template<typename _Tp> friend uint32_t GetHeightFunc(const ito::PCLPointCloud *pc);
    template<typename _Tp> friend void SetHeightFunc(ito::PCLPointCloud *pc, uint32_t height);
    template<typename _Tp> friend bool GetDenseFunc(const ito::PCLPointCloud *pc);
    template<typename _Tp> friend void SetDenseFunc(ito::PCLPointCloud *pc, bool dense);
    template<typename _Tp> friend void SetWidthFunc(ito::PCLPointCloud *pc, uint32_t width);
    template<typename _Tp> friend void PcAddFunc(ito::PCLPointCloud *pc1, const ito::PCLPointCloud *pc2, ito::PCLPointCloud *pcRes);
    template<typename _Tp> friend void SetItemFunc(ito::PCLPointCloud *pc, size_t n, ito::PCLPoint &point);
    template<typename _Tp> friend void PushBackFunc(ito::PCLPointCloud * pc, const ito::PCLPoint & point);
    template<typename _Tp> friend bool EmptyFunc(const ito::PCLPointCloud *pc);
    template<typename _Tp> friend void ReserveResizeFunc(ito::PCLPointCloud *pc, size_t n, bool reserveNotResize);
    template<typename _Tp> friend void ClearFunc(ito::PCLPointCloud *pc);
    template<typename _Tp> friend void EraseFunc(ito::PCLPointCloud *pc, uint32_t startIndex, uint32_t endIndex);
    template<typename _Tp> friend void InsertFunc(ito::PCLPointCloud *pc, uint32_t index, const ito::PCLPoint& point);
    template<typename _Tp> friend void ScaleXYZFunc(ito::PCLPointCloud *pc, ito::float32 scaleX, ito::float32 scaleY, ito::float32 scaleZ);
    template<typename _Tp> friend std::string GetFieldsListFunc(const ito::PCLPointCloud *pc);


    pcl::PointCloud<pcl::PointXYZ>::Ptr m_pcXYZ;                   /*!< shared pointer to point cloud of type pcl::PointXYZ. Contains valid data if m_type is ito::pclXYZ.*/
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_pcXYZI;                 /*!< shared pointer to point cloud of type pcl::PointXYZI. Contains valid data if m_type is ito::pclXYZI.*/
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr m_pcXYZRGBA;           /*!< shared pointer to point cloud of type pcl::PointXYZRGBA. Contains valid data if m_type is ito::pclXYZRGBA.*/
    pcl::PointCloud<pcl::PointNormal>::Ptr m_pcXYZNormal;          /*!< shared pointer to point cloud of type pcl::PointNormal. Contains valid data if m_type is ito::pclXYZNormal.*/
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr m_pcXYZINormal;     /*!< shared pointer to point cloud of type pcl::PointXYZINormal. Contains valid data if m_type is ito::pclXYZINormal.*/
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr m_pcXYZRGBNormal; /*!< shared pointer to point cloud of type pcl::PointXYZRGBNormal. Contains valid data if m_type is ito::pclXYZRGBNormal.*/

    ito::tPCLPointType m_type;                                     /*!< type of point cloud covered by this instance */
};

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class PCLPolygonMesh
    \brief generic class that covers a shared pointer to pcl::PolygonMesh that is a class for a polygonal mesh provided by the point cloud library (PCL)
*/
class POINTCLOUD_EXPORT PCLPolygonMesh
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
    inline pcl::PolygonMesh::ConstPtr polygonMesh() const { return m_polygonMesh; }

    inline void setPolygonMesh(pcl::PolygonMesh::Ptr &mesh) { m_polygonMesh = mesh; }
    PCLPolygonMesh & operator= (const PCLPolygonMesh &copy);

    inline bool valid() const { return m_valid; }

    size_t height() const;
    size_t width() const;
    std::string getFieldsList() const;

    std::ostream& streamOut(std::ostream& out);

protected:

private:
    bool m_valid;                        /*!< true if m_polygonMesh is a valid polygonal mesh, else false */
    pcl::PolygonMesh::Ptr m_polygonMesh; /*!< shared pointer to pcl::PolygonMesh object */
};

} //end namespace ito




#endif
