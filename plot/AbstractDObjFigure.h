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

#ifndef ABSTRACTDOBJFIGURE_H
#define ABSTRACTDOBJFIGURE_H

#include "AbstractFigure.h"
#include "../DataObject/dataobj.h"
#include "../common/sharedStructuresQt.h"
#include "../common/addInInterface.h"
#include "../common/interval.h"
#include "../common/qtMetaTypeDeclarations.h"

#include <qpointer.h>
#include <qpixmap.h>


#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONPLOT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito {

class ITOMCOMMONPLOT_EXPORT AbstractDObjFigure : public AbstractFigure
{
    Q_OBJECT
    Q_PROPERTY(QSharedPointer<ito::DataObject> source READ getSource WRITE setSource DESIGNABLE false USER false)
    Q_PROPERTY(QSharedPointer<ito::DataObject> displayed READ getDisplayed DESIGNABLE false USER false)
    Q_PROPERTY(QPointer<ito::AddInDataIO> camera READ getCamera WRITE setCamera DESIGNABLE false USER false)

    Q_PROPERTY(ito::AutoInterval xAxisInterval READ getXAxisInterval WRITE setXAxisInterval DESIGNABLE true USER true)
    Q_PROPERTY(ito::AutoInterval yAxisInterval READ getYAxisInterval WRITE setYAxisInterval DESIGNABLE true USER true)
    Q_PROPERTY(ito::AutoInterval zAxisInterval READ getZAxisInterval WRITE setZAxisInterval DESIGNABLE true USER true)
    Q_PROPERTY(QString colorMap READ getColorMap WRITE setColorMap DESIGNABLE true USER true)

    Q_CLASSINFO("prop://source", "Sets the input data object for this plot.")
    Q_CLASSINFO("prop://displayed", "This returns the currently displayed data object [read only].")
    Q_CLASSINFO("prop://camera", "Use this property to set a camera/grabber to this plot (live image).")
    Q_CLASSINFO("prop://xAxisInterval", "Sets the visible range of the displayed x-axis (in coordinates of the data object). Set it to 'auto' if range should be automatically set [default].")
    Q_CLASSINFO("prop://yAxisInterval", "Sets the visible range of the displayed y-axis (in coordinates of the data object). Set it to 'auto' if range should be automatically set [default].")
    Q_CLASSINFO("prop://zAxisInterval", "Sets the visible range of the displayed z-axis (in coordinates of the data object). Set it to 'auto' if range should be automatically set [default].")
    Q_CLASSINFO("prop://colorMap", "Color map (string) that should be used to colorize a non-color data object.")

    Q_CLASSINFO("slot://setSource", "This slot can be implemented by any plot plugin to send a dataObject to the plot. Here it is not required and therefore not implemented.")
    Q_CLASSINFO("slot://setLinePlot", "This slot can be implemented by any plot plugin to force the plot to open a line plot. Here it is not required and therefore not implemented.")

public:
    AbstractDObjFigure(const QString &itomSettingsFile, AbstractFigure::WindowMode windowMode = AbstractFigure::ModeStandaloneInUi, QWidget *parent = 0);

    virtual ~AbstractDObjFigure();

    //! overload of AbstractNode::update
    ito::RetVal update(void);

    virtual QSharedPointer<ito::DataObject> getSource(void) const;
    virtual ito::RetVal setSource(QSharedPointer<ito::DataObject> source);

    virtual ito::RetVal setAxisData(QSharedPointer<ito::DataObject> data, Qt::Axis axis);
    virtual QSharedPointer<ito::DataObject> getAxisData(Qt::Axis axis) const;

    virtual QSharedPointer<ito::DataObject> getDisplayed(void); // { return QSharedPointer<ito::DataObject>(m_pOutput["displayed"]->getVal<ito::DataObject*>()); }

    virtual QPointer<ito::AddInDataIO> getCamera(void) const;
    virtual ito::RetVal setCamera( QPointer<ito::AddInDataIO> camera );

    virtual ito::AutoInterval getXAxisInterval(void) const;
    virtual void setXAxisInterval(ito::AutoInterval);

    virtual ito::AutoInterval getYAxisInterval(void) const;
    virtual void setYAxisInterval(ito::AutoInterval);

    virtual ito::AutoInterval getZAxisInterval(void) const;
    virtual void setZAxisInterval(ito::AutoInterval);

    virtual QString getColorMap(void) const;
    virtual void setColorMap(QString);

    //! plot-specific render function to enable more complex printing in subfigures ...
    virtual QPixmap renderToPixMap(const int xsize, const int ysize, const int resolution);

protected:
    QHash<QString, QSharedPointer<ito::DataObject> > m_dataPointer;
    bool m_cameraConnected;

    RetVal removeLiveSource();

signals:

public slots:
    //this source is invoked by any connected camera
    virtual ITOM_PYNOTACCESSIBLE void setSource(QSharedPointer<ito::DataObject> source, ItomSharedSemaphore *waitCond);

    //this can be invoked by python to trigger a lineplot
    virtual ito::RetVal setLinePlot(const double x0, const double y0, const double x1, const double y1, const int destID = -1);
};

} // namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif //ABSTRACTDOBJFIGURE_H
