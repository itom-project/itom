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

#ifndef ABSTRACTDOBJPCLFIGURE_H
#define ABSTRACTDOBJPCLFIGURE_H

#include "AbstractFigure.h"
#include "../DataObject/dataobj.h"
#if defined USEPCL || ITOM_POINTCLOUDLIBRARY
#include "../PointCloud/pclStructures.h"
#endif
#include "../common/sharedStructuresQt.h"
#include "../common/addInInterface.h"
#include "../common/interval.h"
#include "../common/qtMetaTypeDeclarations.h"

#include <qpointer.h>
#include <qpixmap.h>


#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito {

class ITOMCOMMONQT_EXPORT AbstractDObjPclFigure : public AbstractFigure
{
    Q_OBJECT
    Q_PROPERTY(QSharedPointer<ito::DataObject> dataObject READ getDataObject WRITE setDataObject DESIGNABLE false USER false)   
#ifdef USEPCL //this symbol is automatically defined if the itom SDK is compiled with PCL support (set in itom_sdk.cmake)
    Q_PROPERTY(QSharedPointer<ito::PCLPointCloud> pointCloud READ getPointCloud WRITE setPointCloud DESIGNABLE false USER false)
    Q_PROPERTY(QSharedPointer<ito::PCLPolygonMesh> polygonMesh READ getPolygonMesh WRITE setPolygonMesh DESIGNABLE false USER false)
#endif

    Q_PROPERTY(ito::AutoInterval xAxisInterval READ getXAxisInterval WRITE setXAxisInterval DESIGNABLE true USER true)
    Q_PROPERTY(ito::AutoInterval yAxisInterval READ getYAxisInterval WRITE setYAxisInterval DESIGNABLE true USER true)
    Q_PROPERTY(ito::AutoInterval zAxisInterval READ getZAxisInterval WRITE setZAxisInterval DESIGNABLE true USER true)
    Q_PROPERTY(QString colorMap READ getColorMap WRITE setColorMap DESIGNABLE true USER true)

    Q_CLASSINFO("prop://dataObject", "Sets the input data object for this plot.")
    Q_CLASSINFO("prop://polygonMesh", "Sets the input polygon mesh for this plot.")
    Q_CLASSINFO("prop://pointCloud", "Sets the input point cloud for this plot.")

    Q_CLASSINFO("prop://xAxisInterval", "Sets the visible range of the displayed x-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].")
    Q_CLASSINFO("prop://yAxisInterval", "Sets the visible range of the displayed y-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].")
    Q_CLASSINFO("prop://zAxisInterval", "Sets the visible range of the displayed z-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].")
    Q_CLASSINFO("prop://colorMap", "Color map (string) that should be used to colorize a non-color data object.")
    
    Q_CLASSINFO("slot://setLinePlot", "This (virtual) slot can be invoked by python to trigger a lineplot.")

public:
    AbstractDObjPclFigure(const QString &itomSettingsFile, const ito::ParamBase::Type inpType, AbstractFigure::WindowMode windowMode = AbstractFigure::ModeStandaloneInUi, QWidget *parent = 0) : 
        AbstractFigure(itomSettingsFile, windowMode, parent),
        m_inpType(inpType)
    {
        m_pInput.insert("pointCloud", new ito::Param("pointCloud", ito::ParamBase::PointCloudPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));       
        m_pInput.insert("polygonMesh", new ito::Param("polygonMesh", ito::ParamBase::PolygonMeshPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));       
        m_pInput.insert("dataObject", new ito::Param("dataObject", ito::ParamBase::DObjPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
    }

    AbstractDObjPclFigure(const QString &itomSettingsFile, AbstractFigure::WindowMode windowMode = AbstractFigure::ModeStandaloneInUi, QWidget *parent = 0) :
        AbstractFigure(itomSettingsFile, windowMode, parent),
        m_inpType(0)
    {
        m_pInput.insert("pointCloud", new ito::Param("pointCloud", ito::ParamBase::PointCloudPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
        m_pInput.insert("polygonMesh", new ito::Param("polygonMesh", ito::ParamBase::PolygonMeshPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
        m_pInput.insert("dataObject", new ito::Param("dataObject", ito::ParamBase::DObjPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
    }
    
    virtual ~AbstractDObjPclFigure() 
    {
    }

    ito::RetVal update(void);

    virtual void setDataObject(QSharedPointer<ito::DataObject>);
    virtual QSharedPointer<ito::DataObject> getDataObject(void) const;

    virtual inline ito::AutoInterval getXAxisInterval(void) const { return ito::AutoInterval(); }
    virtual inline void setXAxisInterval(ito::AutoInterval) { return; }
        
    virtual inline ito::AutoInterval getYAxisInterval(void) const { return ito::AutoInterval(); }
    virtual inline void setYAxisInterval(ito::AutoInterval) { return; }
        
    virtual inline ito::AutoInterval getZAxisInterval(void) const { return ito::AutoInterval(); }
    virtual inline void setZAxisInterval(ito::AutoInterval) { return; }
        
    virtual inline QString getColorMap(void) const { return QString(); }
    virtual inline void setColorMap(QString) { return; }

    //! plot-specific render function to enable more complex printing in subfigures ...
    virtual inline QPixmap renderToPixMap(const int xsize, const int ysize, const int resolution) 
    {
        QPixmap emptyMap(xsize, ysize);
        emptyMap.fill(Qt::green);
        return emptyMap;
    } 

#ifdef USEPCL
    virtual void setPointCloud(QSharedPointer<ito::PCLPointCloud>);
    virtual QSharedPointer<ito::PCLPointCloud> getPointCloud(void) const;

    virtual void setPolygonMesh(QSharedPointer<ito::PCLPolygonMesh>);
    virtual QSharedPointer<ito::PCLPolygonMesh> getPolygonMesh(void) const;
#endif

protected:
    QHash<QString, QSharedPointer<ito::DataObject> > m_dataPointerDObj;
    int m_inpType;
#ifdef USEPCL
    QHash<QString, QSharedPointer<ito::PCLPointCloud> > m_dataPointerPC;
    QHash<QString, QSharedPointer<ito::PCLPolygonMesh> > m_dataPointerPM;
#endif

signals:

public slots:
    //this source is invoked by any connected camera
/*    
    virtual void setSource(QSharedPointer<ito::DataObject> source, ItomSharedSemaphore *waitCond);
    virtual void setSource(QSharedPointer<ito::PCLPointCloud> source, ItomSharedSemaphore *waitCond);
    virtual void setSource(QSharedPointer<ito::PCLPolygonMesh> source, ItomSharedSemaphore *waitCond);
*/
    //this can be invoked by python to trigger a lineplot
    virtual ito::RetVal setLinePlot(const double x0, const double y0, const double x1, const double y1, const int destID = -1);
};

} // namespace ito


#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif //ABSTRACTDOBJPCLFIGURE_H
