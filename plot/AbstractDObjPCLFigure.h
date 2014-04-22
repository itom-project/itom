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

#ifndef ABSTRACTDOBJPCLFIGURE_H
#define ABSTRACTDOBJPCLFIGURE_H

#include "AbstractFigure.h"
#include "../DataObject/dataobj.h"
#if defined USEPCL || ITOM_POINTCLOUDLIBRARY
#include "../PointCloud/pclStructures.h"
#endif
#include "../common/sharedStructuresQt.h"
#include "../common/addInInterface.h"

#include <qpointer.h>

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito {

class ITOMCOMMONQT_EXPORT AbstractDObjPclFigure : public AbstractFigure
{
    Q_OBJECT
    Q_PROPERTY(QSharedPointer<ito::DataObject> dataObject READ getDataObject WRITE setDataObject DESIGNABLE false USER false)
//    Q_PROPERTY(QSharedPointer<ito::DataObject> displayed READ getDisplayed DESIGNABLE false USER false)
//    Q_PROPERTY(QPointer<ito::AddInDataIO> camera READ getCamera WRITE setCamera DESIGNABLE false USER false)    
#ifdef USEPCL
    Q_PROPERTY(QSharedPointer<ito::PCLPointCloud> pointCloud READ getPointCloud WRITE setPointCloud DESIGNABLE false USER false)
    Q_PROPERTY(QSharedPointer<ito::PCLPolygonMesh> polygonMesh READ getPolygonMesh WRITE setPolygonMesh DESIGNABLE false USER false)
#endif

    Q_PROPERTY(QPointF xAxisInterval READ getXAxisInterval WRITE setXAxisInterval DESIGNABLE true USER true)
    Q_PROPERTY(QPointF yAxisInterval READ getYAxisInterval WRITE setYAxisInterval DESIGNABLE true USER true)
    Q_PROPERTY(QPointF zAxisInterval READ getZAxisInterval WRITE setZAxisInterval DESIGNABLE true USER true)
    Q_PROPERTY(QString colorMap READ getColorMap WRITE setColorMap DESIGNABLE true USER true)

    Q_CLASSINFO("prop://source", "Sets the input data object for this plot.")
    Q_CLASSINFO("prop://displayed", "This returns the currently displayed data object [read only].")
//    Q_CLASSINFO("prop://camera", "Use this property to set a camera/grabber to this plot (live image).")
    Q_CLASSINFO("prop://xAxisInterval", "Sets the visible range of the displayed x-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].")
    Q_CLASSINFO("prop://yAxisInterval", "Sets the visible range of the displayed y-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].")
    Q_CLASSINFO("prop://zAxisInterval", "Sets the visible range of the displayed z-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].")
    Q_CLASSINFO("prop://colorMap", "Color map (string) that should be used to colorize a non-color data object.")
    

public:
    AbstractDObjPclFigure(const QString &itomSettingsFile, const ito::ParamBase::Type inpType, AbstractFigure::WindowMode windowMode = AbstractFigure::ModeStandaloneInUi, QWidget *parent = 0) : 
        AbstractFigure(itomSettingsFile, windowMode, parent),
        m_inpType(inpType)
    {
        m_pInput.insert("pointCloud", new ito::Param("pointCloud", ito::ParamBase::PointCloudPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
//            m_pOutput.insert("displayed", new ito::Param("displayed", ito::ParamBase::PointCloudPtr, NULL, QObject::tr("Actual output data of plot").toLatin1().data()));        
        m_pInput.insert("polygonMesh", new ito::Param("polygonMesh", ito::ParamBase::PolygonMeshPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
//            m_pOutput.insert("displayed", new ito::Param("displayed", ito::ParamBase::PolygonMeshPtr, NULL, QObject::tr("Actual output data of plot").toLatin1().data()));        
        m_pInput.insert("dataObject", new ito::Param("dataObject", ito::ParamBase::DObjPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
//            m_pOutput.insert("displayed", new ito::Param("displayed", ito::ParamBase::DObjPtr, NULL, QObject::tr("Actual output data of plot").toLatin1().data()));
    }
    
    virtual ~AbstractDObjPclFigure() 
    {
//        removeLiveSource();
    }

    ito::RetVal update(void);

    virtual void setDataObject(QSharedPointer<ito::DataObject>);
    virtual QSharedPointer<ito::DataObject> getDataObject(void) const;
#ifdef USEPCL
    virtual void setPointCloud(QSharedPointer<ito::PCLPointCloud>);
    virtual QSharedPointer<ito::PCLPointCloud> getPointCloud(void) const;

    virtual void setPolygonMesh(QSharedPointer<ito::PCLPolygonMesh>);
    virtual QSharedPointer<ito::PCLPolygonMesh> getPolygonMesh(void) const;
#endif

    virtual inline QPointF getXAxisInterval(void) const { return QPointF(); }
    virtual inline void setXAxisInterval(QPointF) { return; }
        
    virtual inline QPointF getYAxisInterval(void) const { return QPointF(); }
    virtual inline void setYAxisInterval(QPointF) { return; }
        
    virtual inline QPointF getZAxisInterval(void) const { return QPointF(); }
    virtual inline void setZAxisInterval(QPointF) { return; }
        
    virtual inline QString getColorMap(void) const { return QString(); }
    virtual inline void setColorMap(QString) { return; }

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
