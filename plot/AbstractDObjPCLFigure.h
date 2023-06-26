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


#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONPLOT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito {

    class ITOMCOMMONPLOT_EXPORT AbstractDObjPclFigure : public AbstractFigure
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
    AbstractDObjPclFigure(const QString &itomSettingsFile, const ito::ParamBase::Type inpType, AbstractFigure::WindowMode windowMode = AbstractFigure::ModeStandaloneInUi, QWidget *parent = 0);

    AbstractDObjPclFigure(const QString &itomSettingsFile, AbstractFigure::WindowMode windowMode = AbstractFigure::ModeStandaloneInUi, QWidget *parent = 0);

    virtual ~AbstractDObjPclFigure();

    //! overload of AbstractNode::update
    ito::RetVal update(void);

    virtual ito::RetVal setDataObject(QSharedPointer<ito::DataObject>);
    virtual QSharedPointer<ito::DataObject> getDataObject(void) const;

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

#ifdef USEPCL
    virtual ito::RetVal setPointCloud(QSharedPointer<ito::PCLPointCloud>);
    virtual QSharedPointer<ito::PCLPointCloud> getPointCloud(void) const;

    virtual ito::RetVal setPolygonMesh(QSharedPointer<ito::PCLPolygonMesh>);
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
    //this can be invoked by python to trigger a lineplot
    virtual ito::RetVal setLinePlot(const double x0, const double y0, const double x1, const double y1, const int destID = -1);
};

} // namespace ito


#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif //ABSTRACTDOBJPCLFIGURE_H
