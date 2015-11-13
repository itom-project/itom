/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom.
  
    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef FIGUREWIDGET_H
#define FIGUREWIDGET_H

#include "abstractDockWidget.h"

#include "common/sharedStructures.h"
#include "DataObject/dataobj.h"
#if ITOM_POINTCLOUDLIBRARY > 0
#include "../../PointCloud/pclStructures.h"
#endif
#include "common/addInInterface.h"

#include <qgridlayout.h>
#include <qsharedpointer.h>
#include <qpointer.h>
#include <qaction.h>
#include <qmenu.h>
#include <qevent.h>
#include <qsignalmapper.h>

namespace ito {

class FigureWidget : public AbstractDockWidget
{
    Q_OBJECT

    Q_PROPERTY(QRect geometry READ geometry WRITE setGeometry)


public:
    FigureWidget(const QString &title, bool docked, bool isDockAvailable, int rows, int cols, QWidget *parent = 0, Qt::WindowFlags flags = 0);
    ~FigureWidget();

    RetVal plot(QSharedPointer<ito::DataObject> dataObj, int areaRow, int areaCol, const QString &className, QWidget **canvasWidget);
#if ITOM_POINTCLOUDLIBRARY > 0
    RetVal plot(QSharedPointer<ito::PCLPointCloud> dataObj, int areaRow, int areaCol, const QString &className, QWidget **canvasWidget);
    RetVal plot(QSharedPointer<ito::PCLPolygonMesh> dataObj, int areaRow, int areaCol, const QString &className, QWidget **canvasWidget);
#endif
    RetVal liveImage(QPointer<AddInDataIO> cam, int areaRow, int areaCol, const QString &className, QWidget **canvasWidget);

    RetVal loadDesignerWidget(int areaRow, int areaCol, const QString &className, QWidget **canvasWidget);

    QWidget *getSubplot(int index) const;

    RetVal changeCurrentSubplot(int newIndex);

    //---------------------------------
    // setter / getter
    //---------------------------------
    void setFigHandle(QSharedPointer<unsigned int> figHandle) { m_guardedFigHandle = figHandle; }


    inline int rows() const { return m_rows; };
    inline int cols() const { return m_cols; };

protected:

    QWidget* prepareWidget(const QString &plotClassName, int areaRow, int areaCol, RetVal &retval);

    void createActions();
    void createMenus();
    void createToolBars();
    void createStatusBar();
    void updateActions();
    void updatePythonActions(){ updateActions(); }

    void closeEvent(QCloseEvent *event); // { event->accept(); };

    //bool eventFilter(QObject *obj, QEvent *event);

    QSharedPointer<ito::Param> getParamByInvoke(ito::AddInBase* addIn, const QString &paramName, ito::RetVal &retval);

private:
    QGridLayout *m_pGrid;
    QWidget *m_pCenterWidget;

    QMenu *m_menuWindow;
    QMenu *m_menuSubplot;
    QAction *m_firstSysAction;

    QActionGroup *m_pSubplotActions;

    int m_rows;
    int m_cols;
    int m_curIdx;

    QSharedPointer<unsigned int> m_guardedFigHandle; //this figure holds it own reference, this is deleted if this figure is closed by a close-event or if the close-method is called.

    QMap< QObject*, QList<QAction*> > m_menuStack;

    QVector<QWidget*> m_widgets;

signals:
    
private slots:
    void mnu_subplotActionsTriggered(QAction *action);
};

} //end namespace ito

#endif
