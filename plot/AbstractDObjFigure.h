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

#ifndef ABSTRACTDOBJFIGURE_H
#define ABSTRACTDOBJFIGURE_H

#include "AbstractFigure.h"
#include "../DataObject/dataobj.h"
#include "../common/sharedStructuresQt.h"
#include "../common/addInInterface.h"

#include <qpointer.h>

namespace ito {

class ITOMSHAREDDESIGNER_EXPORT AbstractDObjFigure : public AbstractFigure
{
    Q_OBJECT
    Q_PROPERTY(QSharedPointer<ito::DataObject> source READ getSource WRITE setSource DESIGNABLE false USER false)
    Q_PROPERTY(QSharedPointer<ito::DataObject> displayed READ getDisplayed DESIGNABLE false USER false)
    Q_PROPERTY(QPointer<ito::AddInDataIO> camera READ getCamera WRITE setCamera DESIGNABLE false USER false)

    Q_PROPERTY(QPointF xAxisInterval READ getXAxisInterval WRITE setXAxisInterval DESIGNABLE true USER true)
    Q_PROPERTY(QPointF yAxisInterval READ getYAxisInterval WRITE setYAxisInterval DESIGNABLE true USER true)
    Q_PROPERTY(QPointF zAxisInterval READ getZAxisInterval WRITE setZAxisInterval DESIGNABLE true USER true)
    Q_PROPERTY(QString colorMap READ getColorMap WRITE setColorMap DESIGNABLE true USER true)

    Q_CLASSINFO("prop://source", "Sets the input data object for this plot.")
    Q_CLASSINFO("prop://displayed", "This returns the currently displayed data object [read only].")
    Q_CLASSINFO("prop://camera", "Use this property to set a camera/grabber to this plot (live image).")
    Q_CLASSINFO("prop://xAxisInterval", "Sets the visible range of the displayed x-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].")
    Q_CLASSINFO("prop://yAxisInterval", "Sets the visible range of the displayed y-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].")
    Q_CLASSINFO("prop://zAxisInterval", "Sets the visible range of the displayed z-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].")
    Q_CLASSINFO("prop://colorMap", "Color map (string) that should be used to colorize a non-color data object.")
    

public:
    AbstractDObjFigure(const QString &itomSettingsFile, AbstractFigure::WindowMode windowMode = AbstractFigure::ModeStandaloneInUi, QWidget *parent = 0) : 
        AbstractFigure(itomSettingsFile, windowMode, parent),
        m_cameraConnected(false)
    {
        m_pInput.insert("source", new ito::Param("source", ito::ParamBase::DObjPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
        m_pOutput.insert("displayed", new ito::Param("displayed", ito::ParamBase::DObjPtr, NULL, QObject::tr("Actual output data of plot").toLatin1().data()));
    }

    virtual ~AbstractDObjFigure() 
    {
        removeLiveSource();
    }

    ito::RetVal update(void);

    virtual QSharedPointer<ito::DataObject> getSource(void) const;
    virtual void setSource(QSharedPointer<ito::DataObject> source);

    virtual QSharedPointer<ito::DataObject> getDisplayed(void); // { return QSharedPointer<ito::DataObject>(m_pOutput["displayed"]->getVal<ito::DataObject*>()); }

    virtual QPointer<ito::AddInDataIO> getCamera(void) const;
    virtual void setCamera( QPointer<ito::AddInDataIO> camera );

    virtual inline QPointF getXAxisInterval(void) const { return QPointF(); }
    virtual inline void setXAxisInterval(QPointF) { return; }
        
    virtual inline QPointF getYAxisInterval(void) const { return QPointF(); }
    virtual inline void setYAxisInterval(QPointF) { return; }
        
    virtual inline QPointF getZAxisInterval(void) const { return QPointF(); }
    virtual inline void setZAxisInterval(QPointF) { return; }
        
    virtual inline QString getColorMap(void) const { return QString(); }
    virtual inline void setColorMap(QString) { return; }

protected:
    QHash<QString, QSharedPointer<ito::DataObject> > m_dataPointer;
    bool m_cameraConnected;

    RetVal removeLiveSource();

signals:

public slots:
    //this source is invoked by any connected camera
    virtual void setSource(QSharedPointer<ito::DataObject> source, ItomSharedSemaphore *waitCond);

    //this can be invoked by python to trigger a lineplot
    virtual ito::RetVal setLinePlot(const double x0, const double y0, const double x1, const double y1, const int destID = -1);
};

} // namespace ito

#endif //ABSTRACTDOBJFIGURE_H
