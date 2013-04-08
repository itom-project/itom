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

class AbstractDObjFigure : public AbstractFigure
{
    Q_OBJECT
    Q_PROPERTY(QSharedPointer<ito::DataObject> source READ getSource WRITE setSource DESIGNABLE false)
    Q_PROPERTY(QSharedPointer<ito::DataObject> displayed READ getDisplayed DESIGNABLE false)
    Q_PROPERTY(QPointer<ito::AddInDataIO> camera READ getCamera WRITE setCamera DESIGNABLE false)

    Q_PROPERTY(QPointF XAxisInterval READ getXAxisInterval WRITE setXAxisInterval DESIGNABLE true)
    Q_PROPERTY(QPointF YAxisInterval READ getYAxisInterval WRITE setYAxisInterval DESIGNABLE true)
    Q_PROPERTY(QPointF ZAxisInterval READ getZAxisInterval WRITE setZAxisInterval DESIGNABLE true)
    Q_PROPERTY(QString ColorPalette READ getColorPalette WRITE setColorPalette DESIGNABLE true)
    

public:
    AbstractDObjFigure(const QString &itomSettingsFile, AbstractFigure::WindowMode windowMode = AbstractFigure::ModeStandaloneInUi, QWidget *parent = 0) : 
        AbstractFigure(itomSettingsFile, windowMode, parent),
        m_cameraConnected(false)
    {
        m_pInput.insert("source", new ito::Param("source", ito::ParamBase::DObjPtr, NULL, QObject::tr("Source data for plot").toAscii().data()));
        m_pOutput.insert("displayed", new ito::Param("displayed", ito::ParamBase::DObjPtr, NULL, QObject::tr("Actual output data of plot").toAscii().data()));
    }

    virtual ~AbstractDObjFigure() 
    {
        removeLiveSource();
    }

    ito::RetVal update(void);

    virtual QSharedPointer<ito::DataObject> getSource(void);
    void setSource(QSharedPointer<ito::DataObject> source);

    virtual QSharedPointer<ito::DataObject> getDisplayed(void); // { return QSharedPointer<ito::DataObject>(m_pOutput["displayed"]->getVal<ito::DataObject*>()); }

    QPointer<ito::AddInDataIO> getCamera(void);
    void setCamera( QPointer<ito::AddInDataIO> camera );

    virtual inline QPointF getXAxisInterval(void) { return QPointF(); }
    virtual inline void setXAxisInterval(QPointF) { return; }
        
    virtual inline QPointF getYAxisInterval(void) { return QPointF(); }
    virtual inline void setYAxisInterval(QPointF) { return; }
        
    virtual inline QPointF getZAxisInterval(void) { return QPointF(); }
    virtual inline void setZAxisInterval(QPointF) { return; }
        
    virtual inline QString getColorPalette(void) { return QString(); }
    virtual inline void setColorPalette(QString) { return; }

protected:
    QHash<QString, QSharedPointer<ito::DataObject> > m_dataPointer;
    bool m_cameraConnected;

    RetVal removeLiveSource();

signals:

public slots:
    //this source is invoked by any connected camera
    virtual void setSource(QSharedPointer<ito::DataObject> source, ItomSharedSemaphore *waitCond);
        
};

} // namespace ito

#endif //ABSTRACTDOBJFIGURE_H
