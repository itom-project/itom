/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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
#include "plot/AbstractDObjFigure.h"


namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------

FigureWidget::FigureWidget(const QString &title, bool docked, bool isDockAvailable, QWidget *parent, Qt::WindowFlags /*flags*/)
    : AbstractDockWidget(docked, isDockAvailable, floatingWindow, movingEnabled, title, parent),
    m_pGrid(NULL),
    m_pCenterWidget(NULL)
{

    AbstractDockWidget::init();

    m_pCenterWidget = new QWidget(this);
    m_pGrid = new QGridLayout(m_pCenterWidget);
    m_pGrid->setSpacing(0);
    m_pGrid->setContentsMargins(0,0,0,0);
    m_pCenterWidget->setLayout(m_pGrid);

    resizeDockWidget(700,400);

    setContentWidget(m_pCenterWidget);
    m_pCenterWidget->setContentsMargins(0,0,0,0);
    //m_pCenterWidget->setStyleSheet( "background-color:#ffccee" );

    setFocusPolicy(Qt::StrongFocus);
//    setAcceptDrops(true);
    
}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
/*!
    cancels connections and closes every tab.
*/
FigureWidget::~FigureWidget()
{


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
}

//----------------------------------------------------------------------------------------------------------------------------------------
void FigureWidget::createMenus()
{
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
RetVal FigureWidget::plot(QSharedPointer<ito::DataObject> dataObj, int areaRow, int areaCol, QString className, QPoint &newAreas)
{
    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    UiOrganizer *uiOrg = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    RetVal retval;
    QString plotClassName;
    bool exists = false;

    if(dwo && uiOrg)
    {
        int dims = dataObj->getDims();
        int sizex = dataObj->getSize(dims - 1);
        int sizey = dataObj->getSize(dims - 2);
        if ((dims == 1) || ((dims > 1) && ((sizex == 1) || (sizey == 1))))
        {
            plotClassName = dwo->getFigureClass("DObjStaticLine", plotClassName, retval);
            
        }
        else
        {
            plotClassName = dwo->getFigureClass("DObjStaticImage", plotClassName, retval);
            //not 1D so try 2D ;-) new 2dknoten()
        }

        QLayoutItem *currentItem = m_pGrid->itemAtPosition(areaRow,areaCol);
        QWidget *currentItemWidget = currentItem ? currentItem->widget() : NULL;
        if(currentItemWidget)
        {
            const QMetaObject* meta = currentItemWidget->metaObject();
            if(QString::compare(plotClassName, meta->className(), Qt::CaseInsensitive) == 0)
            {
                exists = true;
            }
        }

        if(exists == false)
        {
            QWidget* newWidget = uiOrg->loadDesignerPluginWidget(plotClassName, retval, m_pCenterWidget);
            if(newWidget)
            {
                QWidget *oldWidget = currentItem ? currentItemWidget : NULL;
                m_pGrid->addWidget(newWidget, areaRow, areaCol, 1, 1);
                currentItemWidget = newWidget;
                if(oldWidget) oldWidget->deleteLater();
            }
            else
            {
                retval += RetVal::format(retError,0,"could not create designer widget of class '%s'", plotClassName.toAscii().data());
            }
        }

        if(!retval.containsError() && currentItemWidget)
        {
            ito::AbstractDObjFigure *dObjFigure = NULL;
            if (currentItemWidget->inherits("ito::AbstractDObjFigure"))
            {
                dObjFigure = (ito::AbstractDObjFigure*)(currentItemWidget);
                dObjFigure->setSource(dataObj);
            }
            else
            {
                retval += RetVal::format(retError,0,"designer widget of class '%s' cannot plot objects of type dataObject", plotClassName.toAscii().data());
            }
        }
    }
    else
    {
        retval += RetVal(retError,0,"designerWidgetOrganizer or uiOrganizer is not available");
    }

    newAreas.setX(m_pGrid->columnCount());
    newAreas.setY(m_pGrid->rowCount());

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------------
RetVal FigureWidget::liveImage(QPointer<AddInDataIO> cam, int areaRow, int areaCol, QString className, QPoint &newAreas)
{
    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    UiOrganizer *uiOrg = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    RetVal retval;
    QString plotClassName;
    bool exists = false;

    if(cam.isNull())
    {
        retval += RetVal(retError,0,"camera is not available any more");
    }
    else if(dwo && uiOrg)
    {
        //get size of camera image
        QSharedPointer<ito::Param> sizex = getParamByInvoke(cam.data(), "sizex", retval);
        QSharedPointer<ito::Param> sizey = getParamByInvoke(cam.data(), "sizey", retval);
        
        if(!retval.containsError())
        {
            if(sizex->getVal<int>() == 1 || sizey->getVal<int>() == 1)
            {
                plotClassName = dwo->getFigureClass("DObjLiveLine", plotClassName, retval);
            }
            else
            {
                plotClassName = dwo->getFigureClass("DObjLiveImage", plotClassName, retval);
            }
        }

        QLayoutItem *currentItem = m_pGrid->itemAtPosition(areaRow,areaCol);
        QWidget *currentItemWidget = currentItem ? currentItem->widget() : NULL;
        if(currentItemWidget)
        {
            const QMetaObject* meta = currentItemWidget->metaObject();
            if(QString::compare(plotClassName, meta->className(), Qt::CaseInsensitive) == 0)
            {
                exists = true;
            }
        }

        if(exists == false)
        {
            QWidget* newWidget = uiOrg->loadDesignerPluginWidget(plotClassName, retval, m_pCenterWidget);
            if(newWidget)
            {
                QWidget *oldWidget = currentItem ? currentItemWidget : NULL;
                m_pGrid->addWidget(newWidget, areaRow, areaCol, 1, 1);
                currentItemWidget = newWidget;
                if(oldWidget) oldWidget->deleteLater();
            }
            else
            {
                retval += RetVal::format(retError,0,"could not create designer widget of class '%s'", plotClassName.toAscii().data());
            }
        }

        if(!retval.containsError() && currentItemWidget)
        {
            ito::AbstractDObjFigure *dObjFigure = NULL;
            if (currentItemWidget->inherits("ito::AbstractDObjFigure"))
            {
                dObjFigure = (ito::AbstractDObjFigure*)(currentItemWidget);
                dObjFigure->setCamera(cam);
            }
            else
            {
                retval += RetVal::format(retError,0,"designer widget of class '%s' cannot plot objects of type dataObject", plotClassName.toAscii().data());
            }
        }
    }
    else
    {
        retval += RetVal(retError,0,"designerWidgetOrganizer or uiOrganizer is not available");
    }

    newAreas.setX(m_pGrid->columnCount());
    newAreas.setY(m_pGrid->rowCount());

    return retval;
}



QSharedPointer<ito::Param> FigureWidget::getParamByInvoke(ito::AddInBase* addIn, const QString &paramName, ito::RetVal &retval)
{
    QSharedPointer<ito::Param> result;

    if(addIn == NULL)
    {
        retval += RetVal(retError,0,"addInBase pointer is NULL");
    }
    else
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        ito::Param param = addIn->getParamRec(paramName);
        if ( param.getName() != NULL)   // Parameter is defined
        {
            result = QSharedPointer<ito::Param>(new ito::Param(param));
            QMetaObject::invokeMethod(addIn, "getParam", Q_ARG(QSharedPointer<ito::Param>, result), Q_ARG(ItomSharedSemaphore *, locker.getSemaphore()));
            if (!locker.getSemaphore()->wait(5000) )
            {
                retval += RetVal::format(retError,0,"timeout while getting parameter '%s' from plugin", paramName.toAscii().data());
            }
            else
            {
                retval += locker.getSemaphore()->returnValue;
            }
        }
        else
        {
            retval += RetVal::format(retError,0,"parameter '%s' is not defined in plugin", paramName.toAscii().data());
        }
    }

    return result;
}

} //end namespace ito
