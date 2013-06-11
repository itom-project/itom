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

#include "dialogProperties.h"

#include "../global.h"
#include <qmetaobject.h>

#include "widgetPropEditorAPI.h"
#include "widgetPropEditorCalltips.h"
#include "widgetPropEditorStyles.h"
#include "widgetPropEditorAutoCompletion.h"
#include "widgetPropEditorGeneral.h"
#include "widgetPropGeneralLanguage.h"
#include "widgetPropPythonStartup.h"
#include "widgetPropConsoleWrap.h"
#include "widgetPropFigurePlugins.h"
#include "widgetPropGeneralApplication.h"

#include "AppManagement.h"

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
DialogProperties::DialogProperties(QWidget * parent, Qt::WindowFlags f) :
    QDialog(parent, f)
{
    setWindowTitle(tr("Properties"));

    m_pStackedWidget = new QStackedWidget();
    m_pEmptyPage = new QWidget(m_pStackedWidget);
    m_pStackedWidget->addWidget(m_pEmptyPage);
    
    m_pCategories = new QTreeWidget();
    m_pCategories->setColumnCount(1);
    m_pCategories->setHeaderHidden(true);
    m_pCategories->setSortingEnabled(false);

    m_pCategories->setMinimumWidth(200);

    connect(m_pCategories, SIGNAL(currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)), this, SLOT(categoryChanged(QTreeWidgetItem*,QTreeWidgetItem*)));

    m_pButtonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel | QDialogButtonBox::Apply , Qt::Horizontal);
    connect(m_pButtonBox, SIGNAL(accepted()), this, SLOT(accepted()));
    connect(m_pButtonBox, SIGNAL(rejected()), this, SLOT(rejected()));
    connect(m_pButtonBox->button(QDialogButtonBox::Apply), SIGNAL(clicked()), this, SLOT(apply()));

    m_pLine = new QFrame();
    m_pLine->setFrameShape(QFrame::HLine);
    m_pLine->setFrameShadow(QFrame::Sunken);

    m_pPageTitle = new QLabel("page title");

    QVBoxLayout *m_pVerticalLayoutRight;
    QVBoxLayout *m_pVerticalLayout;
    QWidget *m_pSplitterRightWidget = new QWidget();

    m_pVerticalLayoutRight = new QVBoxLayout();

    m_pVerticalLayoutRight->addWidget(m_pPageTitle);
    m_pVerticalLayoutRight->addWidget(m_pLine);
    m_pVerticalLayoutRight->addWidget(m_pStackedWidget);
    

    m_pSplitterRightWidget->setLayout(m_pVerticalLayoutRight);
    m_pSplitterRightWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    m_pSplitter = new QSplitter(Qt::Horizontal);
    m_pSplitter->addWidget(m_pCategories);
    m_pSplitter->addWidget(m_pSplitterRightWidget);
    m_pSplitter->setStretchFactor(1, 10);

    m_pVerticalLayout = new QVBoxLayout(this);
    m_pVerticalLayout->addWidget(m_pSplitter);
    m_pVerticalLayout->addWidget(m_pButtonBox);

    setLayout(m_pVerticalLayout);

    initPages();

    resize(700, 450);
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogProperties::~DialogProperties()
{
    PropertyPage page;
    foreach(page, m_pages)
    {
        if (page.m_widget)
        {
            delete page.m_widget;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogProperties::initPages()
{
    //-----------------------------------------------------------------------------------------------------
    // Please insert property pages here:
    // 
    // Step 1: #include PropertyWidget at the top of this file
    // Step 2: if necessary, add parent page to the m_pages-map below
    // Step 3: add property page to the m_pages -map
    //
    // Important: The key of the m_pages-map and the third entry in the PropertyPage-struct must have the same value.
    // This value gives the tree-structure of the property-widgets. Every / (slash) indicates a new child of the parent tree
    //-----------------------------------------------------------------------------------------------------


    m_pages["04_editor"] = PropertyPage(tr("Editor"), tr("Editor - please choose subpage"), "04_editor", NULL, QIcon(":/application/icons/preferences-general.png"));
    m_pages["04_editor/general"] = PropertyPage(tr("General"), tr("Editor - General"), "04_editor/general", new WidgetPropEditorGeneral(), QIcon(":/application/icons/preferences-general.png"));
    m_pages["04_editor/api"] = PropertyPage(tr("API"), tr("Editor - API files"), "04_editor/api", new WidgetPropEditorAPI(), QIcon(":/application/icons/preferences-general.png"));
    m_pages["04_editor/calltips"] = PropertyPage(tr("Calltips"), tr("Editor - calltips"), "04_editor/calltips", new WidgetPropEditorCalltips(), QIcon(":/application/icons/preferences-general.png"));
    m_pages["04_editor/autocompletion"] = PropertyPage(tr("Auto Completion"), tr("Editor - auto completion"), "04_editor/autocompletion", new WidgetPropEditorAutoCompletion(), QIcon(":/application/icons/preferences-general.png"));
    m_pages["04_editor/styles"] = PropertyPage(tr("Styles"), tr("Editor - styles"), "04_editor/styles", new WidgetPropEditorStyles(), QIcon(":/application/icons/preferences-general.png"));
    m_pages["01_console"] = PropertyPage(tr("Console"), tr("Console - please choose subpage"), "01_console", NULL, QIcon(":/application/icons/editSmartIndent.png"));
    m_pages["01_console/lineWrap"] = PropertyPage(tr("Line Wrap"), tr("Console - Line Wrap"), "01_console/lineWrap", new WidgetPropConsoleWrap(), QIcon(":/application/icons/editSmartIndent.png"));
    m_pages["03_python"] = PropertyPage(tr("Python"), tr("Python - please choose subpage"), "03_python", NULL, QIcon(":/application/icons/preferences-python.png"));
    m_pages["03_python/startup"] = PropertyPage(tr("Startup"), tr("Python - startups"), "03_python/startup", new WidgetPropPythonStartup(), QIcon(":/application/icons/preferences-python.png"));
    m_pages["00_general"] = PropertyPage(tr("General"), tr("General - please choose subpage"), "00_general", NULL, QIcon(":/application/icons/itomicon/curAppIcon.png"));
    m_pages["00_general/language"] = PropertyPage(tr("Language"), tr("General - language"), "00_general/language", new WidgetPropGeneralLanguage(), QIcon(":/application/icons/preferences-general.png"));
    m_pages["00_general/application"] = PropertyPage(tr("Application"), tr("General - application"), "00_general/application", new WidgetPropGeneralApplication(), QIcon(":/application/icons/itomicon/curAppIcon.png"));
    m_pages["05_plots"] = PropertyPage(tr("Plots and Figures"), tr("Plots and Figures - please choose subpage"), "05_plots", NULL, QIcon(":/plots/icons/itom_icons/3d.png"));
    m_pages["05_plots/defaults"] = PropertyPage(tr("Default Plots"), tr("Plots and Figures - Defaults"), "05_plots/defaults", new WidgetPropFigurePlugins(), QIcon(":/plots/icons/itom_icons/2d.png"));

    PropertyPage page;
    QStringList pathes;

    foreach(page, m_pages)
    {
        pathes = page.m_fullname.split("/");
        addPage(page, m_pCategories->invisibleRootItem(), pathes);

        if (page.m_widget)
        {
            page.m_widget->readSettings();
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogProperties::addPage(PropertyPage page, QTreeWidgetItem *parent, QStringList remainingPathes)
{
    QTreeWidgetItem *newItem;
    bool found = false;

    if (remainingPathes.length() == 1)
    {
        newItem = new QTreeWidgetItem();
        newItem->setText(0, page.m_name);
        newItem->setData(1, Qt::DisplayRole, page.m_fullname);

        if (!page.m_icon.isNull())
        {
            newItem->setIcon(0, page.m_icon);
        }

        parent->addChild(newItem);
        parent->setExpanded(true);

        if (page.m_widget)
        {
            m_pStackedWidget->addWidget(qobject_cast<QWidget*>(page.m_widget));
        }
    }
    else if (remainingPathes.length() > 0)
    {
        for (int i = 0 ; i < parent->childCount() ; i++)
        {
            if (parent->child(i)->data(1, Qt::DisplayRole).toString() == remainingPathes[0])
            {
                remainingPathes.pop_front();
                addPage(page, parent->child(i), remainingPathes);
                
                found = true;
                break;
            }
        }

        if (!found)
        {
            qDebug() << "it was not possible to find a parent property page with name " << remainingPathes[0] << ". This should be a child of " << parent->data(0, Qt::DisplayRole).toString();
        }
    }
    else
    {
        qDebug() << "the given fullname of a new property page is empty.";
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogProperties::categoryChanged(QTreeWidgetItem *current, QTreeWidgetItem * /*previous*/)
{
    bool found = false;

    if (current)
    {
        QString key = current->data(1, Qt::DisplayRole).toString();

        QMap<QString, PropertyPage>::iterator it = m_pages.find(key);

        if ( it != m_pages.end() )
        {
            m_pPageTitle->setText(it->m_title);

            if (it->m_widget)
            {
                it->m_visited = true;
                m_pStackedWidget->setCurrentWidget(qobject_cast<QWidget*>(it->m_widget));
                found = true;
            }
        }
    }

    if (!found)
    {
        m_pStackedWidget->setCurrentWidget(m_pEmptyPage);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogProperties::accepted()
{
    apply();
    close();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogProperties::rejected()
{
    close();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogProperties::apply()
{
    PropertyPage page;
    QStringList pathes;

    foreach(page, m_pages)
    {
        if (page.m_widget && page.m_visited)
        {
            page.m_widget->writeSettings();
        }
    }

    QObject *mainApplication = AppManagement::getMainApplication();

    QMetaObject::invokeMethod(mainApplication, "_propertiesChanged");

}

} //end namespace ito
