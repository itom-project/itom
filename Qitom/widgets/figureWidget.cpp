/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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

#include "../global.h"

#include "figureWidget.h"
#include "../AppManagement.h"
#include "../organizer/designerWidgetOrganizer.h"
#include "../organizer/uiOrganizer.h"
#include "../helper/guiHelper.h"
#include "plot/AbstractDObjFigure.h"
#include "plot/AbstractDObjPCLFigure.h"

#include <qlayoutitem.h>
#include <qactiongroup.h>


namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------

FigureWidget::FigureWidget(
        const QString &title,
        bool docked, bool isDockAvailable,
        int rows, int cols,
        QWidget* parent,
        Qt::WindowFlags /*flags*/) :
        AbstractDockWidget(docked, isDockAvailable,
                            floatingWindow, movingEnabled,
                            title, "", parent),
    m_pGrid(NULL),
    m_pCenterWidget(NULL),
    m_menuWindow(NULL),
    m_menuSubplot(NULL),
    m_pSubplotActions(NULL),
    m_firstSysAction(NULL),
    m_rows(rows),
    m_cols(cols),
    m_curIdx(-1),
    m_currentBorderColor(QColor(41, 128, 185)) /*blue from start screen of itom*/
{

    AbstractDockWidget::init();

    QWidget *temp;
    int idx = 0;

    m_pCenterWidget = new QWidget(this);
    m_pGrid = new QGridLayout(m_pCenterWidget);
    m_pGrid->setSpacing(0);
    m_pGrid->setContentsMargins(0,0,0,0);
    m_pCenterWidget->setLayout(m_pGrid);

    m_widgets.fill(NULL, rows * cols);

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            temp = new QWidget(m_pCenterWidget);
            temp->setContentsMargins(1, 1, 1, 1); //for margin of selected subplot area
            temp->setObjectName(QString("emptyWidget%1").arg(m_cols * r + c));
            m_widgets[idx] = temp;
            m_pGrid->addWidget(m_widgets[idx], r, c);
            idx++;
        }
    }

    changeCurrentSubplot(0);

    resize(defaultSize());

    setContentWidget(m_pCenterWidget);
    m_pCenterWidget->setContentsMargins(0,0,0,0);
    setFocusPolicy(Qt::StrongFocus);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
/*!

*/
FigureWidget::~FigureWidget()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
//!> returns the default size of this figure widget
QSize FigureWidget::defaultSize() const
{
    float f = ito::GuiHelper::screenDpiFactor();
    return QSize(750 * f, 450 * f);
}

//----------------------------------------------------------------------------------------------------------------------------------
void FigureWidget::setCurrentBorderColor(QColor color)
{
    m_currentBorderColor = color;
}

//----------------------------------------------------------------------------------------------------------------------------------------
void FigureWidget::closeEvent(QCloseEvent *event) // { event->accept(); };
{
    //usually any figure-instances (in python) keep references (QSharedPointer<unsigned int>) to this figure. If the last python-instance is closed,
    //the deleter-method of their guarded figure handle deletes this figure, if this figure does not keep its own reference. This is only the case,
    //if a figure is directly created by the plot of liveImage command from the module 'itom'. Then no corresponding figure-instance is created and
    //the figure is only closed if the user closes it or the static close-method of class 'figure'.
    m_guardedFigHandle.clear();

    event->accept();
}

//----------------------------------------------------------------------------------------------------------------------------------------
void FigureWidget::createActions()
{
    QAction *temp = NULL;
    if (m_rows > 1 || m_cols > 1)
    {
        m_pSubplotActions = new QActionGroup(this);
        m_pSubplotActions->setExclusive(true);
        connect(m_pSubplotActions, SIGNAL(triggered(QAction*)), this, SLOT(mnu_subplotActionsTriggered(QAction*)));

        for (int r = 0; r < m_rows; r++)
        {
            for (int c = 0; c < m_cols; c++)
            {
                temp = new QAction(tr("subplot %1 (empty)").arg(c + r * m_cols),this);
                temp->setData(c + r * m_cols);
                temp->setCheckable(true);
                m_pSubplotActions->addAction(temp);
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------------
void FigureWidget::createMenus()
{
    if (m_pSubplotActions)
    {
        m_menuSubplot = new QMenu(tr("&Subplots"), this);
        m_menuSubplot->addActions(m_pSubplotActions->actions());
        m_firstSysAction = getMenuBar()->addMenu(m_menuSubplot);
    }

    //create main menus
    m_menuWindow = new QMenu(tr("&Windows"), this);
    if (m_actStayOnTop)
    {
        m_menuWindow->addAction(m_actStayOnTop);
    }
    if (m_actStayOnTopOfApp)
    {
        m_menuWindow->addAction(m_actStayOnTopOfApp);
    }

    QAction *act = getMenuBar()->addMenu(m_menuWindow);
    if (!m_firstSysAction) m_firstSysAction = act;
}

//----------------------------------------------------------------------------------------------------------------------------------------
void FigureWidget::createToolBars()
{
}

//----------------------------------------------------------------------------------------------------------------------------------------
void FigureWidget::createStatusBar()
{
}

//----------------------------------------------------------------------------------------------------------------------------------------
void FigureWidget::updateActions()
{
}

//----------------------------------------------------------------------------------------------------------------------------------------
#if ITOM_POINTCLOUDLIBRARY > 0
RetVal FigureWidget::plot(
        QSharedPointer<ito::PCLPointCloud> pc,
        int areaRow, int areaCol,
        const QString &className,
        QWidget **canvasWidget)
{
    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    RetVal retval;
    QString plotClassName;
    int idx = areaCol + areaRow * m_cols;

    *canvasWidget = NULL;

    if (dwo)
    {
        /* className can be

        * an empty string, then the right category is guessed from the dimension of the pointCloud
        * 2.5D, or 2.5d -> then the default plot of the corresponding category is used
        * a className, then the className is searched within the category, guessed from the
        *     dimension of the pointCloud (if not found, a warning is returned and the default of the category is used)
        * 2.5d:className -> the className is searched within the given category,
        *     if not found, the default class from the category is used
        */
        if (className.compare("2.5d", Qt::CaseInsensitive) == 0)
        {
            plotClassName = dwo->getFigureClass("PerspectivePlot", "", retval);
        }
        else if (className.startsWith("2.5d:", Qt::CaseInsensitive))
        {
            plotClassName = dwo->getFigureClass("PerspectivePlot", className.mid(5), retval);
        }
        else
        {
            plotClassName = dwo->getFigureClass("PerspectivePlot", className, retval);
        }

        QWidget *destWidget = prepareWidget(plotClassName, areaRow, areaCol, retval);

        if (!retval.containsError() && destWidget)
        {
            if (destWidget->inherits("ito::AbstractDObjPclFigure"))
            {
                ito::AbstractDObjPclFigure *dObjPclFigure = qobject_cast<ito::AbstractDObjPclFigure*>(destWidget);
                dObjPclFigure->setPointCloud(pc);
                *canvasWidget = destWidget;

                connect(dObjPclFigure, &AbstractFigure::windowTitleModified,
                    [this](QString title) {this->setAdvancedWindowTitle(title);});
            }
            else
            {
                retval += RetVal::format(retError, 0, tr("designer widget of class '%s' cannot plot objects of type pointCloud").toLatin1().data(), plotClassName.toLatin1().data());
                DELETE_AND_SET_NULL(destWidget);
            }

            if (idx == m_curIdx)
            {
                changeCurrentSubplot(idx);
            }
        }
        else if (retval.containsError())
        {
            DELETE_AND_SET_NULL(destWidget);
        }
    }
    else
    {
        retval += RetVal(retError, 0, tr("designerWidgetOrganizer is not available").toLatin1().data());
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------------
RetVal FigureWidget::plot(
        QSharedPointer<ito::PCLPolygonMesh> pm,
        int areaRow, int areaCol,
        const QString &className,
        QWidget **canvasWidget)
{
    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    RetVal retval;
    QString plotClassName;
    int idx = areaCol + areaRow * m_cols;

    *canvasWidget = NULL;

    if (dwo)
    {
        /* className can be

        * an empty string, then the right category is guessed from the dimension of the polygonMesh
        * 2.5D, or 2.5d -> then the default plot of the corresponding category is used
        * a className, then the className is searched within the category, guessed from the
        *     dimension of the polygonMesh (if not found, a warning is returned and the default of the category is used)
        * 2.5d:className -> the className is searched within the given category,
        *     if not found, the default class from the category is used
        */
        if (className.compare("2.5d", Qt::CaseInsensitive) == 0)
        {
            plotClassName = dwo->getFigureClass("PerspectivePlot", "", retval);
        }
        else if (className.startsWith("2.5d:", Qt::CaseInsensitive))
        {
            plotClassName = dwo->getFigureClass("PerspectivePlot", className.mid(5), retval);
        }
        else
        {
            plotClassName = dwo->getFigureClass("PerspectivePlot", className, retval);
        }

        QWidget *destWidget = prepareWidget(plotClassName, areaRow, areaCol, retval);

        if (!retval.containsError() && destWidget)
        {
            if (destWidget->inherits("ito::AbstractDObjPclFigure"))
            {
                ito::AbstractDObjPclFigure *dObjPclFigure = qobject_cast<ito::AbstractDObjPclFigure*>(destWidget);
                dObjPclFigure->setPolygonMesh(pm);
                *canvasWidget = destWidget;

                connect(dObjPclFigure, &AbstractFigure::windowTitleModified,
                    [this](QString title) {this->setAdvancedWindowTitle(title);});
            }
            else
            {
                retval += RetVal::format(retError, 0, tr("designer widget of class '%s' cannot plot objects of type polygonMesh").toLatin1().data(), plotClassName.toLatin1().data());
                DELETE_AND_SET_NULL(destWidget);
            }

            if (idx == m_curIdx)
            {
                changeCurrentSubplot(idx);
            }
        }
        else if (retval.containsError())
        {
            DELETE_AND_SET_NULL(destWidget);
        }
    }
    else
    {
        retval += RetVal(retError, 0, tr("designerWidgetOrganizer is not available").toLatin1().data());
    }

    return retval;
}
#endif // #if ITOM_POINTCLOUDLIBRARY > 0

//----------------------------------------------------------------------------------------------------------------------------------------
RetVal FigureWidget::plot(
        QSharedPointer<ito::DataObject> dataObj,
        QSharedPointer<ito::DataObject> xAxisObj,
        int areaRow, int areaCol,
        const QString &className,
        QWidget **canvasWidget)
{
    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    RetVal retval;
    QString plotClassName;
    int idx = areaCol + areaRow * m_cols;

    *canvasWidget = NULL;

    if (dwo)
    {
        /* className can be

            * an empty string, then the right category is guessed from the dimension of the dataObject
            * 1D, 1d, 2D, 2d, 2.5D, or 2.5d -> then the default plot of the corresponding category is used
            * a className, then the className is searched within the category, guessed from the dimension of the dataObject (if not found, a warning is returned and the default of the category is used)
            * 1d:className, 2d:className, 2.5d:className -> the className is searched within the given category, if not found, the default class from the category is used
        */
        if (className.compare("1d", Qt::CaseInsensitive) == 0)
        {
            plotClassName = dwo->getFigureClass("DObjStaticLine", "", retval);
        }
        else if (className.compare("2d", Qt::CaseInsensitive) == 0)
        {
            plotClassName = dwo->getFigureClass("DObjStaticImage", "", retval);
        }
        else if (className.compare("2.5d", Qt::CaseInsensitive) == 0)
        {
            plotClassName = dwo->getFigureClass("PerspectivePlot", "", retval);
        }
        else if (className.startsWith("1d:", Qt::CaseInsensitive))
        {
            plotClassName = dwo->getFigureClass("DObjStaticLine", className.mid(3), retval);
        }
        else if (className.startsWith("2d:", Qt::CaseInsensitive))
        {
            plotClassName = dwo->getFigureClass("DObjStaticImage", className.mid(3), retval);
        }
        else if (className.startsWith("2.5d:", Qt::CaseInsensitive))
        {
            plotClassName = dwo->getFigureClass("PerspectivePlot", className.mid(5), retval);
        }
        else
        {
            int dims = dataObj->getDims();
            int sizex = dataObj->getSize(dims - 1);
            int sizey = dataObj->getSize(dims - 2);
            if ((dims == 1) || ((dims > 1) && ((sizex == 1) || (sizey == 1))))
            {
                plotClassName = dwo->getFigureClass("DObjStaticLine", className, retval);

            }
            else
            {
                plotClassName = dwo->getFigureClass("DObjStaticImage", className, retval);
            }
        }

        QWidget *destWidget = prepareWidget(plotClassName, areaRow, areaCol, retval);

        if (!retval.containsError() && destWidget && destWidget->inherits("ito::AbstractFigure"))
        {
            connect((ito::AbstractFigure*)destWidget, &AbstractFigure::windowTitleModified,
                [this](QString title) {this->setAdvancedWindowTitle(title);});
        }

        if (!retval.containsError() && destWidget)
        {
            if (destWidget->inherits("ito::AbstractDObjFigure"))
            {
                ito::AbstractDObjFigure *dObjFigure = NULL;
                dObjFigure = (ito::AbstractDObjFigure*)(destWidget);
                if (xAxisObj) //here: the xAxis object is set first, the source is not set, yet, therefore the update() method of the plot will quit in this first run and will be executed, once the source is available (below)
                {
                    dObjFigure->setAxisData(xAxisObj, Qt::XAxis);
                }
                dObjFigure->setSource(dataObj);

                *canvasWidget = destWidget;
            }
            else if (destWidget->inherits("ito::AbstractDObjPclFigure"))
            {
                ito::AbstractDObjPclFigure *dObjPclFigure = NULL;
                dObjPclFigure = (ito::AbstractDObjPclFigure*)(destWidget);
                dObjPclFigure->setDataObject(dataObj);
                *canvasWidget = destWidget;
            }
            else
            {
                retval += RetVal::format(retError, 0, tr("designer widget of class '%s' cannot plot objects of type dataObject").toLatin1().data(), plotClassName.toLatin1().data());
                DELETE_AND_SET_NULL(destWidget);
            }

            if (idx == m_curIdx && !retval.containsError())
            {
                changeCurrentSubplot(idx);
            }
        }
        else if (retval.containsError())
        {
            DELETE_AND_SET_NULL(destWidget);
        }
    }
    else
    {
        retval += RetVal(retError, 0, tr("designerWidgetOrganizer is not available").toLatin1().data());
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------------
RetVal FigureWidget::liveImage(
        QPointer<AddInDataIO> cam,
        int areaRow, int areaCol,
        const QString &className,
        QWidget **canvasWidget)
{
    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    RetVal retval;
    QString plotClassName;
//    bool exists = false;
    int idx = areaCol + areaRow * m_cols;

    *canvasWidget = NULL;

    if (!dwo)
    {
        retval += RetVal(retError, 0, tr("designerWidgetOrganizer is not available").toLatin1().data());
    }
    else if (cam.isNull())
    {
        retval += RetVal(retError, 0, tr("camera is not available any more").toLatin1().data());
    }
    else
    {
        //get grabDepth
        bool setDepth = false;
        bool isLine = false;
        ito::AutoInterval bitRange (0.0, 1.0);
        QSharedPointer<ito::Param> bpp = getParamByInvoke(cam.data(), "bpp", retval);

        if (!retval.containsError())
        {
            if (bpp->getVal<int>() == 8)
            {
                setDepth = true;
                bitRange.setMaximum(255.0);
            }
            else if (bpp->getVal<int>() < 17)
            {
                setDepth = true;
                bitRange.setMaximum((float)((1 << bpp->getVal<int>())-1));
            }
            else if (bpp->getVal<int>() == 32)
            {
                // ToDo define float32 and int32 behavior!
            }
            else if (bpp->getVal<int>() == 64)
            {
                // ToDo define float64 behavior!
            }

        }

        //get size of camera image
        QSharedPointer<ito::Param> sizex = getParamByInvoke(cam.data(), "sizex", retval);
        QSharedPointer<ito::Param> sizey = getParamByInvoke(cam.data(), "sizey", retval);

        if (!retval.containsError())
        {
            /* className can be

                * an empty string, then the right category is guessed from the dimension of the dataObject
                * 1D, 1d, 2D, 2d, 2.5D, or 2.5d -> then the default plot of the corresponding category is used
                * a className, then the className is searched within the category, guessed from the dimension of the dataObject (if not found, a warning is returned and the default of the category is used)
                * 1d:className, 2d:className, 2.5d:className -> the className is searched within the given category, if not found, the default class from the category is used
            */
            if (className.compare("1d", Qt::CaseInsensitive) == 0)
            {
                plotClassName = dwo->getFigureClass("DObjLiveLine", "", retval);
            }
            else if (className.compare("2d", Qt::CaseInsensitive) == 0)
            {
                plotClassName = dwo->getFigureClass("DObjLiveImage", "", retval);
            }
            else if (className.compare("2.5d", Qt::CaseInsensitive) == 0)
            {
                plotClassName = dwo->getFigureClass("PerspectivePlot", "", retval);
            }
            else if (className.startsWith("1d:", Qt::CaseInsensitive))
            {
                plotClassName = dwo->getFigureClass("DObjLiveLine", className.mid(3), retval);
            }
            else if (className.startsWith("2d:", Qt::CaseInsensitive))
            {
                plotClassName = dwo->getFigureClass("DObjLiveImage", className.mid(3), retval);
            }
            else if (className.startsWith("2.5d:", Qt::CaseInsensitive))
            {
                plotClassName = dwo->getFigureClass("PerspectivePlot", className.mid(5), retval);
            }
            else
            {
                if (sizex->getVal<int>() == 1 || sizey->getVal<int>() == 1)
                {
                    plotClassName = dwo->getFigureClass("DObjLiveLine", className, retval);
                    isLine = true;
                }
                else
                {
                    plotClassName = dwo->getFigureClass("DObjLiveImage", className, retval);
                }
            }
        }

        QWidget *destWidget = prepareWidget(plotClassName, areaRow, areaCol, retval);

        if (!retval.containsError() && destWidget && destWidget->inherits("ito::AbstractFigure"))
        {
            connect((ito::AbstractFigure*)destWidget, &AbstractFigure::windowTitleModified,
                [this](QString title) {this->setAdvancedWindowTitle(title);});
        }

        if (!retval.containsError() && destWidget)
        {
            ito::AbstractDObjFigure *dObjFigure = NULL;
            ito::AbstractDObjPclFigure *dObjPclFigure = NULL;

            if (destWidget->inherits("ito::AbstractDObjFigure"))
            {
                dObjFigure = (ito::AbstractDObjFigure*)(destWidget);

                //check if dObjFigure has property "yAxisFlipped" and flip it, if so.
                QVariant yAxisFlipped = dObjFigure->property("yAxisFlipped");
                if (yAxisFlipped.isValid())
                {
                    dObjFigure->setProperty("yAxisFlipped", true);
                }

                if (setDepth)
                {
                    if (isLine) dObjFigure->setYAxisInterval(bitRange);
                    else dObjFigure->setZAxisInterval(bitRange);
                }

                dObjFigure->setCamera(cam);
                *canvasWidget = destWidget;
            }
            else if (destWidget->inherits("ito::AbstractDObjPclFigure"))
            {
                dObjPclFigure = (ito::AbstractDObjPclFigure*)(destWidget);

                //check if dObjFigure has property "yAxisFlipped" and flip it, if so.
                QVariant yAxisFlipped = dObjPclFigure->property("yAxisFlipped");
                if (yAxisFlipped.isValid())
                {
                    dObjPclFigure->setProperty("yAxisFlipped", true);
                }

                if (setDepth)
                {
                    if (isLine) dObjPclFigure->setYAxisInterval(bitRange);
                    else dObjPclFigure->setZAxisInterval(bitRange);
                }

//                dObjPclFigure->setCamera(cam);
                *canvasWidget = destWidget;
            }
            else
            {
                retval += RetVal::format(retError, 0, tr("designer widget of class '%s' cannot plot objects of type dataObject").toLatin1().data(), plotClassName.toLatin1().data());
                DELETE_AND_SET_NULL(destWidget);
            }

            if (idx == m_curIdx && !retval.containsError())
            {
                changeCurrentSubplot(idx);
            }
        }
        else if (retval.containsError())
        {
            DELETE_AND_SET_NULL(destWidget);
        }
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------------
RetVal FigureWidget::loadDesignerWidget(
        int areaRow, int areaCol,
        const QString &className,
        QWidget **canvasWidget)
{
    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    RetVal retval;
    int idx = areaCol + areaRow * m_cols;

    *canvasWidget = NULL;

    if (dwo)
    {
        QWidget *destWidget = prepareWidget(className, areaRow, areaCol, retval);

        if (!retval.containsError() && destWidget)
        {
            *canvasWidget = destWidget;

            if (idx == m_curIdx && !retval.containsError())
            {
                changeCurrentSubplot(idx);
            }
        }
        else if (retval.containsError())
        {
            DELETE_AND_SET_NULL(destWidget);
        }
    }
    else
    {
        retval += RetVal(retError, 0, tr("designerWidgetOrganizer is not available").toLatin1().data());
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------------
QWidget* FigureWidget::prepareWidget(
        const QString &plotClassName,
        int areaRow, int areaCol,
        RetVal &retval)
{
    UiOrganizer *uiOrg = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    bool exists = false;
    QWidget *destinationWidget = NULL;
    QMenuBar *menubar = getMenuBar();
    QAction *insertAction;
    QList<QAction*> menusActions;
    QList<AbstractFigure::ToolBarItem> toolbars;
    int idx = areaCol + areaRow * m_cols;

    if (areaRow < 0 || areaRow >= m_rows)
    {
        retval += ito::RetVal::format(ito::retError, 0, tr("areaRow out of range [0,%i]").toLatin1().data(), m_rows-1);
    }

    if (areaCol < 0 || areaCol >= m_cols)
    {
        retval += ito::RetVal::format(ito::retError, 0, tr("arealCol out of range [0,%i]").toLatin1().data(), m_cols-1);
    }

    if (!retval.containsError())
    {
        if (uiOrg)
        {
            QWidget *currentItemWidget = m_widgets[idx];
            if (currentItemWidget)
            {
                const QMetaObject* meta = currentItemWidget->metaObject();
                if (QString::compare(plotClassName, meta->className(), Qt::CaseInsensitive) == 0)
                {
                    destinationWidget = currentItemWidget;
                    exists = true;
                }
            }

            if (exists == false)
            {
                QWidget* newWidget = uiOrg->loadDesignerPluginWidget(plotClassName, retval, AbstractFigure::ModeInItomFigure, m_pCenterWidget);
                if (newWidget)
                {
                    newWidget->setObjectName(QString("plot%1x%2").arg(areaRow).arg(areaCol));
                    newWidget->setContentsMargins(1, 1, 1, 1); //for margin of selected subplot area

                    ito::AbstractFigure *figWidget = NULL;
                    if (newWidget->inherits("ito::AbstractFigure"))
                    {
                        figWidget = (ito::AbstractFigure*)(newWidget);

                        QList<QMenu*> menus = figWidget->getMenus();
                        menusActions.clear();

                        foreach(QMenu* m, menus)
                        {
                            insertAction = menubar->insertMenu(m_firstSysAction, m);
                            menusActions.append(insertAction);
                        }
                        m_menuStack[figWidget] = menusActions;

                        foreach(const AbstractFigure::ToolBarItem &t, figWidget->getToolbars())
                        {
                            t.toolbar->setVisible(false);
                        }

                        foreach(const AbstractFigure::ToolboxItem &t, figWidget->getToolboxes())
                        {
                            if (t.toolbox && getCanvas())
                            {
                                getCanvas()->addDockWidget(t.area, t.toolbox);
                            }
                        }

                        setMenusVisible(figWidget, false);
                    }

                    if (m_pSubplotActions)
                    {
                        int idx = areaCol + areaRow * m_cols;
                        m_pSubplotActions->actions()[ idx ]->setText(tr("subplot %1").arg(idx));
                    }

                    QWidget *oldWidget = m_widgets[idx];
                    m_pGrid->addWidget(newWidget, areaRow, areaCol, 1, 1);
                    m_widgets[idx] = newWidget;
                    destinationWidget = newWidget;

                    if (oldWidget)
                    {
                        menusActions = m_menuStack[oldWidget];
                        foreach(QAction* a, menusActions)
                        {
                            menubar->removeAction(a);
                        }
                        m_menuStack.remove(oldWidget);

                        oldWidget->deleteLater();
                    }
                }
                else
                {
                    retval += RetVal::format(retError, 0, tr("could not create designer widget of class '%s'").toLatin1().data(), plotClassName.toLatin1().data());
                }
            }
        }
        else
        {
            retval += RetVal(retError, 0, tr("designerWidgetOrganizer or uiOrganizer is not available").toLatin1().data());
        }
    }

    return destinationWidget;
}

//----------------------------------------------------------------------------------------------------------------------------------------
QWidget *FigureWidget::getSubplot(int index) const
{
    if (m_pGrid)
    {
        int column = index % m_cols;
        int row = (index - column) / m_cols;
        QLayoutItem *item = m_pGrid->itemAtPosition(row,column); //(index);
        if (item)
        {
            return item->widget();
        }
    }
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------------
QSharedPointer<ito::Param> FigureWidget::getParamByInvoke(ito::AddInBase* addIn, const QString &paramName, ito::RetVal &retval)
{
    QSharedPointer<ito::Param> result;

    if (addIn == NULL)
    {
        retval += RetVal(retError, 0, tr("addInBase pointer is NULL").toLatin1().data());
    }
    else
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        ito::Param param = addIn->getParamRec(paramName);
        if (param.getName() != NULL)   // Parameter is defined
        {
            result = QSharedPointer<ito::Param>(new ito::Param(param));
            QMetaObject::invokeMethod(addIn, "getParam", Q_ARG(QSharedPointer<ito::Param>, result), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
            if (!locker.getSemaphore()->wait(PLUGINWAIT))
            {
                retval += RetVal::format(retError, 0, tr("timeout while getting parameter '%s' from plugin").toLatin1().data(), paramName.toLatin1().data());
            }
            else
            {
                retval += locker.getSemaphore()->returnValue;
            }
        }
        else
        {
            retval += RetVal::format(retError, 0, tr("parameter '%s' is not defined in plugin").toLatin1().data(), paramName.toLatin1().data());
        }
    }

    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------------
//bool FigureWidget::eventFilter(QObject *obj, QEvent *event)
//{
//    switch(event->type())
//    {
//    case QEvent::MouseButtonDblClick:
//    case QEvent::MouseButtonPress:
//    case QEvent::MouseButtonRelease:
//    case QEvent::MouseMove:
//        return false; //don't forward event to plot widgets
//    }
//
//    return QObject::eventFilter(obj,event);
//}

//----------------------------------------------------------------------------------------------------------------------------------------
void FigureWidget::setMenusVisible(const QWidget *widget, bool visible)
{
    if (m_menuStack.contains((QObject*)widget))
    {
        QList<QAction*> actions = m_menuStack[(QObject*)widget];
        foreach(QAction *a, actions)
        {
            a->setVisible(visible);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------------
RetVal FigureWidget::changeCurrentSubplot(int newIndex)
{
    QWidget *widget = NULL;
    int idx;
    ito::AbstractFigure *figWidget = NULL;
//    QMainWindow *mainWin = getCanvas();
    QString key_;

    for (int r = 0; r < m_rows; r++)
    {
        for (int c = 0; c < m_cols; c++)
        {
            idx = m_cols * r + c;
            widget = m_widgets[idx];
            if (widget && idx == newIndex)
            {
                //if (idx != m_curIdx)
                //{
                    //set new toolbars
                    QList< AbstractFigure::ToolBarItem > toolbars;
                    if (widget->inherits("ito::AbstractFigure"))
                    {
                        figWidget = (ito::AbstractFigure*)(widget);
                        toolbars = figWidget->getToolbars();

                        QList< AbstractFigure::ToolBarItem >::iterator i;
                        for (i = toolbars.begin(); i != toolbars.end(); ++i)
                        {
                            key_ = QString("%1_%2").arg(size_t(figWidget)).arg(i->key);
                            addToolBar(i->toolbar, key_, i->area, i->section);
                            i->toolbar->setVisible(i->visible);
                        }
                    }
                //}

                if (m_rows > 1 || m_cols > 1)
                {
                    widget->setStyleSheet(QString("QWidget#%1 { border: 1px solid %2 } ").arg(widget->objectName()).arg(m_currentBorderColor.name()));
                }

                if (m_pSubplotActions) m_pSubplotActions->actions()[ idx ]->setChecked(true);

                setMenusVisible(widget, true);
            }
            else if (widget)
            {
                if (idx == m_curIdx)
                {
                    //remove toolbars from this
                    QList< AbstractFigure::ToolBarItem > toolbars;
                    if (widget->inherits("ito::AbstractFigure"))
                    {
                        figWidget = (ito::AbstractFigure*)(widget);
                        toolbars = figWidget->getToolbars();

                        QList< AbstractFigure::ToolBarItem >::iterator i;
                        for (i = toolbars.begin(); i != toolbars.end(); ++i)
                        {
                            key_ = QString("%1_%2").arg(size_t(figWidget)).arg(i->key);
                            removeToolBar(key_);
                        }
                    }
                }

                if (m_rows > 1 || m_cols > 1)
                {
                    widget->setContentsMargins(2, 2, 2, 2);
                    widget->setStyleSheet(QString("QWidget#%1 { border: 2px none } ").arg(widget->objectName()));
                }

                setMenusVisible(widget, false);
            }
        }
    }

    m_curIdx = newIndex;

    return retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------------
void FigureWidget::mnu_subplotActionsTriggered(QAction *action)
{
    if (action)
    {
        changeCurrentSubplot(action->data().toInt());

    }
}
} //end namespace ito
