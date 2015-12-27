/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#include "../AbstractDObjPCLFigure.h"

#include <qmetaobject.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
AbstractDObjPclFigure::AbstractDObjPclFigure(const QString &itomSettingsFile, const ito::ParamBase::Type inpType, AbstractFigure::WindowMode windowMode /*= AbstractFigure::ModeStandaloneInUi*/, QWidget *parent /*= 0*/) :
    AbstractFigure(itomSettingsFile, windowMode, parent),
    m_inpType(inpType)
{
    m_pInput.insert("pointCloud", new ito::Param("pointCloud", ito::ParamBase::PointCloudPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
    m_pInput.insert("polygonMesh", new ito::Param("polygonMesh", ito::ParamBase::PolygonMeshPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
    m_pInput.insert("dataObject", new ito::Param("dataObject", ito::ParamBase::DObjPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractDObjPclFigure::AbstractDObjPclFigure(const QString &itomSettingsFile, AbstractFigure::WindowMode windowMode /*= AbstractFigure::ModeStandaloneInUi*/, QWidget *parent /*= 0*/) :
    AbstractFigure(itomSettingsFile, windowMode, parent),
    m_inpType(0)
{
    m_pInput.insert("pointCloud", new ito::Param("pointCloud", ito::ParamBase::PointCloudPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
    m_pInput.insert("polygonMesh", new ito::Param("polygonMesh", ito::ParamBase::PolygonMeshPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
    m_pInput.insert("dataObject", new ito::Param("dataObject", ito::ParamBase::DObjPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractDObjPclFigure::~AbstractDObjPclFigure()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractDObjPclFigure::update(void)
{
    //!> do the real update work, here the transformation from source to displayed takes place
    return applyUpdate();  
}

//----------------------------------------------------------------------------------------------------------------------------------
QSharedPointer<ito::DataObject> AbstractDObjPclFigure::getDataObject(void) const 
{
    ito::DataObject *dObj = m_pInput["dataObject"]->getVal<ito::DataObject*>();
    if (dObj)
    {
        return QSharedPointer<ito::DataObject>(new ito::DataObject(*dObj)); 
    }
    return QSharedPointer<ito::DataObject>();
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjPclFigure::setDataObject(QSharedPointer<ito::DataObject> source) 
{ 
    ito::RetVal retval = ito::retOk;
    QSharedPointer<ito::DataObject> oldSource; //possible backup for previous source, this backup must be alive until updateParam with the new one has been completely propagated

    if (m_dataPointerDObj.contains("dataObject"))
    {
        //check if pointer of shared incoming data object is different to pointer of previous data object
        //if so, free previous
        if (m_dataPointerDObj["dataObject"].data() != source.data())
        {
            oldSource = m_dataPointerDObj["dataObject"];
            m_dataPointerDObj["dataObject"] = source;
        }  
    }
    else
    {
        m_dataPointerDObj["dataObject"] = source;
    }
            
    ito::ParamBase thisParam("dataObject", ito::ParamBase::DObjPtr, (const char*)source.data());
    m_inpType = ito::ParamBase::DObjPtr;
    retval += updateParam(&thisParam, 1);

    updatePropertyDock();
}

#ifdef USEPCL
//----------------------------------------------------------------------------------------------------------------------------------
QSharedPointer<ito::PCLPointCloud> AbstractDObjPclFigure::getPointCloud(void) const 
{
    ito::PCLPointCloud *pc = m_pInput["pointCloud"]->getVal<ito::PCLPointCloud*>();
    if (pc)
    {
        return QSharedPointer<ito::PCLPointCloud>(new ito::PCLPointCloud(*pc)); 
    }
    return QSharedPointer<ito::PCLPointCloud>();
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjPclFigure::setPointCloud(QSharedPointer<ito::PCLPointCloud> source) 
{ 
    ito::RetVal retval = ito::retOk;
    QSharedPointer<ito::PCLPointCloud> oldSource; //possible backup for previous source, this backup must be alive until updateParam with the new one has been completely propagated

    if (m_dataPointerDObj.contains("pointCloud"))
    {
        //check if pointer of shared incoming data object is different to pointer of previous data object
        //if so, free previous
        if (m_dataPointerPC["pointCloud"].data() != source.data())
        {
            oldSource = m_dataPointerPC["pointCloud"];
            m_dataPointerPC["pointCloud"] = source;
        }  
    }
    else
    {
        m_dataPointerPC["pointCloud"] = source;
    }
            
    ito::ParamBase thisParam("pointCloud", ito::ParamBase::PointCloudPtr, (const char*)source.data());
    m_inpType = ito::ParamBase::PointCloudPtr;
    retval += updateParam(&thisParam, 1);

    updatePropertyDock();
}

//----------------------------------------------------------------------------------------------------------------------------------
QSharedPointer<ito::PCLPolygonMesh> AbstractDObjPclFigure::getPolygonMesh(void) const 
{
    ito::PCLPolygonMesh *pm = m_pInput["polygonMesh"]->getVal<ito::PCLPolygonMesh*>();
    if (pm)
    {
        return QSharedPointer<ito::PCLPolygonMesh>(new ito::PCLPolygonMesh(*pm)); 
    }
    return QSharedPointer<ito::PCLPolygonMesh>();
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjPclFigure::setPolygonMesh(QSharedPointer<ito::PCLPolygonMesh> source) 
{ 
    ito::RetVal retval = ito::retOk;
    QSharedPointer<ito::PCLPolygonMesh> oldSource; //possible backup for previous source, this backup must be alive until updateParam with the new one has been completely propagated

    if (m_dataPointerPM.contains("polygonMesh"))
    {
        //check if pointer of shared incoming data object is different to pointer of previous data object
        //if so, free previous
        if (m_dataPointerPM["polygonMesh"].data() != source.data())
        {
            oldSource = m_dataPointerPM["polygonMesh"];
            m_dataPointerPM["polygonMesh"] = source;

        }  
    }
    else
    {
        m_dataPointerPM["polygonMesh"] = source;
    }
            
    ito::ParamBase thisParam("polygonMesh", ito::ParamBase::PolygonMeshPtr, (const char*)source.data());
    m_inpType = ito::ParamBase::PolygonMeshPtr;
    retval += updateParam(&thisParam, 1);

    updatePropertyDock();
}

#endif // USEPCL
//----------------------------------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractDObjPclFigure::setLinePlot(const double /*x0*/, const double /*y0*/, const double /*x1*/, const double /*y1*/, const int /*destID*/)
{
    return ito::RetVal(ito::retError, 0, tr("Function \'spawnLinePlot\' not supported from this plot widget").toLatin1().data());

}

//----------------------------------------------------------------------------------------------------------------------------------
ito::AutoInterval AbstractDObjPclFigure::getXAxisInterval(void) const
{
    return ito::AutoInterval();
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjPclFigure::setXAxisInterval(ito::AutoInterval)
{
    return;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::AutoInterval AbstractDObjPclFigure::getYAxisInterval(void) const
{
    return ito::AutoInterval();
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjPclFigure::setYAxisInterval(ito::AutoInterval)
{
    return;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::AutoInterval AbstractDObjPclFigure::getZAxisInterval(void) const
{
    return ito::AutoInterval();
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjPclFigure::setZAxisInterval(ito::AutoInterval)
{
    return;
}

//----------------------------------------------------------------------------------------------------------------------------------
QString AbstractDObjPclFigure::getColorMap(void) const
{
    return QString();
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjPclFigure::setColorMap(QString)
{
    return;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! plot-specific render function to enable more complex printing in subfigures ...
QPixmap AbstractDObjPclFigure::renderToPixMap(const int xsize, const int ysize, const int resolution)
{
    QPixmap emptyMap(xsize, ysize);
    emptyMap.fill(Qt::green);
    return emptyMap;
}

} //end namespace ito
