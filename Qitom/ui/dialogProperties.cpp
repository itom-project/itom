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

#include "dialogProperties.h"

#include "../global.h"
#include <qmetaobject.h>
#include <qdialogbuttonbox.h>
#include <qscrollarea.h>

#include "widgetPropEditorCalltips.h"
#include "widgetPropEditorStyles.h"
#include "widgetPropEditorAutoCompletion.h"
#include "widgetPropEditorAutoCodeFormat.h"
#include "widgetPropEditorGotoAssignment.h"
#include "widgetPropEditorDocstringGenerator.h"
#include "widgetPropEditorCodeCheckers.h"
#include "widgetPropEditorGeneral.h"
#include "widgetPropEditorScripts.h"
#include "widgetPropGeneralLanguage.h"
#include "widgetPropPythonStartup.h"
#include "widgetPropPythonGeneral.h"
#include "widgetPropConsoleGeneral.h"
#include "widgetPropConsoleWrap.h"
#include "widgetPropConsoleLastCommand.h"
#include "widgetPropFigurePlugins.h"
#include "widgetPropGeneralApplication.h"
#include "widgetPropHelpDock.h"
#include "widgetPropGeneralStyles.h"
#include "widgetPropPluginsAlgorithms.h"
#include "widgetPropPluginsActuators.h"
#include "widgetPropWorkspaceUnpack.h"
#include "widgetPropGeneralPlotSettings.h"
#include "widgetPropPalettes.h"

#include "AppManagement.h"
#include "../helper/guiHelper.h"

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
DialogProperties::DialogProperties(QWidget* parent, Qt::WindowFlags f) :
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

	float screenFactorDpi = GuiHelper::screenDpiFactor();

    m_pCategories->setMinimumWidth(200 * screenFactorDpi);

    connect(
        m_pCategories, SIGNAL(currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)),
        this, SLOT(categoryChanged(QTreeWidgetItem*, QTreeWidgetItem*)));

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
	m_pSplitter->setChildrenCollapsible(false);

    m_pVerticalLayout = new QVBoxLayout(this);
    m_pVerticalLayout->addWidget(m_pSplitter);
    m_pVerticalLayout->addWidget(m_pButtonBox);

    setLayout(m_pVerticalLayout);

    initPages();

	resize(950 * screenFactorDpi, 450 * screenFactorDpi);

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("DialogProperties");
    QString key = settings.value("PropertyTreeNode", "00_general").toString();
    settings.endGroup();

    selectTabByKey(key);
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogProperties::~DialogProperties()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("DialogProperties");
    settings.setValue("PropertyTreeNode", m_CurrentPropertyKey);
    settings.endGroup();

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


    m_pages["04_editor"] = PropertyPage(tr("Editor"), tr("Editor - Please Choose Subpage"), "04_editor", NULL, QIcon(":/application/icons/preferences-general.png"));
    m_pages["04_editor/01general"] = PropertyPage(tr("General"), tr("Editor - General"), "04_editor/01general", new WidgetPropEditorGeneral(), QIcon(":/application/icons/preferences-general.png"));
    m_pages["04_editor/02scripts"] = PropertyPage(tr("Script Editors"), tr("Editor - Scripts"), "04_editor/02scripts", new WidgetPropEditorScripts(), QIcon(":/application/icons/preferences-general.png"));
	m_pages["04_editor/03syntax"] = PropertyPage(tr("Syntax and Style Checks"), tr("Editor - Syntax and Style Checks"), "04_editor/03syntax", new WidgetPropEditorCodeCheckers(), QIcon(":/application/icons/preferences-general.png"));
    m_pages["04_editor/04calltips"] = PropertyPage(tr("Calltips and Help Tooltips"), tr("Editor - Calltips and Help Tooltips"), "04_editor/04calltips", new WidgetPropEditorCalltips(), QIcon(":/application/icons/preferences-general.png"));
    m_pages["04_editor/05autocompletion"] = PropertyPage(tr("Auto Completion"), tr("Editor - Auto Completion"), "04_editor/05autocompletion", new WidgetPropEditorAutoCompletion(), QIcon(":/application/icons/preferences-general.png"));
    m_pages["04_editor/06gotoassignment"] = PropertyPage(tr("Goto Assignment"), tr("Editor - Goto Assignment"), "04_editor/06gotoassignment", new WidgetPropEditorGotoAssignment(), QIcon(":/application/icons/preferences-general.png"));
    m_pages["04_editor/07autocodeformat"] = PropertyPage(tr("Auto Code Format"), tr("Editor - Auto Code Format"), "04_editor/07autocodeformat", new WidgetPropEditorAutoCodeFormat(), QIcon(":/application/icons/preferences-general.png"));
    m_pages["04_editor/08styles"] = PropertyPage(tr("Styles"), tr("Editor - Styles"), "04_editor/08styles", new WidgetPropEditorStyles(), QIcon(":/application/icons/preferences-general.png"));
    m_pages["04_editor/09docstringGenerator"] = PropertyPage(tr("Docstring Generator"), tr("Editor - Docstring Generator"), "04_editor/09docstringGenerator", new WidgetPropEditorDocstringGenerator(), QIcon(":/application/icons/preferences-general.png"));
    m_pages["01_console"] = PropertyPage(tr("Console"), tr("Console - Please Choose Subpage"), "01_console", NULL, QIcon(":/application/icons/editSmartIndent.png"));
    m_pages["01_console/01general"] = PropertyPage(tr("General"), tr("Console - General"), "01_console/01general", new WidgetPropConsoleGeneral(), QIcon(":/application/icons/editSmartIndent.png"));
    m_pages["01_console/02lineWrap"] = PropertyPage(tr("Line Wrap"), tr("Console - Line Wrap"), "01_console/02lineWrap", new WidgetPropConsoleWrap(), QIcon(":/application/icons/editSmartIndent.png"));
    m_pages["01_console/03commandHistory"] = PropertyPage(tr("Command History"), tr("Console - Command History"), "01_console/03commandHistory", new WidgetPropConsoleLastCommand(), QIcon(":/application/icons/editSmartIndent.png"));
    m_pages["03_python"] = PropertyPage(tr("Python"), tr("Python - Please Choose Subpage"), "03_python", NULL, QIcon(":/application/icons/preferences-python.png"));
    m_pages["03_python/01general"] = PropertyPage(tr("General"), tr("Python - General"), "03_python/01general", new WidgetPropPythonGeneral(), QIcon(":/application/icons/preferences-python.png"));
    m_pages["03_python/02startup"] = PropertyPage(tr("Startup"), tr("Python - Startups"), "03_python/02startup", new WidgetPropPythonStartup(), QIcon(":/application/icons/preferences-python.png"));
    m_pages["00_general"] = PropertyPage(tr("General"), tr("General - Please Choose Subpage"), "00_general", NULL, QIcon(":/application/icons/itomicon/itomLogo3_64.png"));
    m_pages["00_general/01application"] = PropertyPage(tr("Application"), tr("General - Application"), "00_general/01application", new WidgetPropGeneralApplication(), QIcon(":/application/icons/itomicon/itomLogo3_64.png"));
    m_pages["00_general/02language"] = PropertyPage(tr("Language"), tr("General - Language"), "00_general/02language", new WidgetPropGeneralLanguage(), QIcon(":/classNavigator/icons/global.png"));
    m_pages["00_general/03helpViewer"]  = PropertyPage(tr("Plugin Help Viewer"), tr("General - Plugin Help Viewer"), "00_general/03helpViewer" , new WidgetPropHelpDock(), QIcon(":/plugins/icons/plugin.png"));
    m_pages["00_general/04styles"] = PropertyPage(tr("Styles and Themes"), tr("General - Styles and Themes"), "00_general/04styles", new WidgetPropGeneralStyles(), QIcon(":/application/icons/color-icon.png"));
    m_pages["05_workspace"] = PropertyPage(tr("Workspace"), tr("Workspace - Please Choose Subpage"), "05_workspace", NULL, QIcon(":/workspace/icons/import-prop-icon.png"));
    m_pages["05_workspace/01unpack"] = PropertyPage(tr("Import To Workspace"), tr("Workspace - Import"), "05_workspace/01unpack", new WidgetPropWorkspaceUnpack(), QIcon(":/workspace/icons/import-prop-icon.png"));
    m_pages["06_plugins"] = PropertyPage(tr("Plugins"), tr("Plugins - Please Choose Subpage"), "06_plugins", NULL, QIcon(":/plugins/icons/plugin.png"));
    m_pages["06_plugins/02algorithms"] = PropertyPage(tr("Algorithms and Filters"), tr("Plugins - Algorithms And Filters"), "06_plugins/02algorithms", new WidgetPropPluginsAlgorithms(), QIcon(":/plugins/icons/pluginAlgo.png"));
    m_pages["06_plugins/03actuators"] = PropertyPage(tr("Actuators"), tr("Plugins - Actuators"), "06_plugins/03actuators", new WidgetPropPluginsActuators(), QIcon(":/plugins/icons/pluginActuator.png"));
    m_pages["07_plots"] = PropertyPage(tr("Plots And Figures"), tr("Plots and Figures - Please Choose Subpage"), "07_plots", NULL, QIcon(":/plots/icons/itom_icons/3d.png"));
    m_pages["07_plots/01defaults"] = PropertyPage(tr("Default Plots"), tr("Plots and Figures - Defaults"), "07_plots/01defaults", new WidgetPropFigurePlugins(), QIcon(":/plots/icons/itom_icons/2d.png"));
    m_pages["07_plots/02defaultSettings"] = PropertyPage(tr("Default Style Settings"), tr("Plots And Figures - Default Style Settings"), "07_plots/02defaultSettings", new WidgetPropGeneralPlotSettings(), QIcon(":/plots/icons/itom_icons/2d.png"));
    m_pages["07_plots/03palettes"] = PropertyPage(tr("Palettes Settings"), tr("Plots and Figures - Palettes Settings"), "07_plots/03palettes", new WidgetPropPalettes(), QIcon(":/plots/icons/itom_icons/color.png"));

    PropertyPage page;
    QStringList pathes;

    foreach(page, m_pages)
    {
        pathes = page.m_fullname.split("/");
        addPage(page, m_pCategories->invisibleRootItem(), pathes);

        /*
        if (page.m_widget)
        {
            page.m_widget->readSettings();
        }
        */
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
        for (int i = 0; i < parent->childCount(); i++)
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
bool DialogProperties::selectTabByKey(QString &key, QTreeWidgetItem *parent /*= NULL*/)
{
    bool found = false;

    if (parent == NULL)
    {
        parent = m_pCategories->invisibleRootItem();
    }
    else
    {
        QString currentKey = parent->data(1, Qt::DisplayRole).toString();
        if (QString::compare(key, currentKey, Qt::CaseInsensitive) == 0)
        {
            found = true;

            m_pCategories->setCurrentItem(parent);
        }
    }

    if (!found) //search all childs...
    {
        for(int i = 0; i < parent->childCount(); ++i)
        {
            found = selectTabByKey(key, parent->child(i));
            if (found)
            {
                break;
            }
        }
    }

    return found;
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogProperties::categoryChanged(QTreeWidgetItem *current, QTreeWidgetItem * /*previous*/)
{
    bool found = false;

    if (current)
    {
        m_CurrentPropertyKey = current->data(1, Qt::DisplayRole).toString();
        QMap<QString, PropertyPage>::iterator it = m_pages.find(m_CurrentPropertyKey);

        if (it != m_pages.end())
        {
            m_pPageTitle->setText(it->m_title);

            if (it->m_widget)
            {
                if (it->m_visited == false && it->m_widget)
                {
                    it->m_widget->readSettings();
                }

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

    QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
    QMetaObject::invokeMethod(mainApplication, "_propertiesChanged");
    QApplication::restoreOverrideCursor();
}

} //end namespace ito
