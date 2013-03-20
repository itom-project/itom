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
    m_pGrid = new QGridLayout(this);
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


void FigureWidget::createActions()
{
}

void FigureWidget::createMenus()
{
}

void FigureWidget::createToolBars()
{
}

void FigureWidget::createStatusBar()
{
}

void FigureWidget::updateActions()
{
}


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
        if(currentItem && currentItem->widget())
        {
            const QMetaObject* meta = currentItem->widget()->metaObject();
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
                m_pGrid->addWidget(newWidget, areaRow, areaCol, 1, 1);
                currentItem = m_pGrid->itemAtPosition(areaRow,areaCol);
            }
            else
            {
                retval += RetVal::format(retError,0,"could not create designer widget of class '%s'", plotClassName.toAscii().data());
            }
        }

        if(!retval.containsError() && currentItem)
        {
            ito::AbstractDObjFigure *dObjFigure = NULL;
            if (currentItem->widget()->inherits("ito::AbstractDObjFigure"))
            {
                dObjFigure = (ito::AbstractDObjFigure*)(currentItem->widget());
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




} //end namespace ito
