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

#define ITOM_IMPORT_API
#include "../../common/apiFunctionsInc.h"
#undef ITOM_IMPORT_API
#define ITOM_IMPORT_PLOTAPI
#include "../../common/apiFunctionsGraphInc.h"
#undef ITOM_IMPORT_PLOTAPI
#include "../AbstractFigure.h"
#include "../../common/typeDefs.h"
#include "../../common/addInInterface.h"
#include "QPropertyEditor/QPropertyEditorWidget.h"
#include "itomWidgets/searchBox.h"

#include <qaction.h>
#include <qtoolbar.h>
#include <qmenu.h>
#include <qmenubar.h>
#include <qevent.h>
#include <qsettings.h>
#include <qshortcut.h>
#include <qmainwindow.h>
#include <qscreen.h>


namespace ito
{

class PropertyEditorWindow : public QMainWindow
{
    Q_OBJECT

public:
    PropertyEditorWindow(QWidget *parent = nullptr);


    QPropertyEditorWidget *propertyEditor() const;

private:
    QPropertyEditorWidget *m_pEditor;

    float screenDpiFactor();
};

//-------------------------------------------------------------------------------------
PropertyEditorWindow::PropertyEditorWindow(QWidget *parent /*= nullptr*/) :
    QMainWindow(parent)
{
    Qt::WindowFlags flags = windowFlags();
    flags |= Qt::Widget;
    flags &= (~Qt::Window);
    setWindowFlags(flags);

    m_pEditor = new QPropertyEditorWidget(this);
    setCentralWidget(m_pEditor);

    QToolBar *toolbar = addToolBar(tr("Options"));
    int size = 16 * screenDpiFactor();
    toolbar->setIconSize(QSize(size, size));

    toolbar->addActions(m_pEditor->actions());

    SearchBox *searchBox = new SearchBox(this);
    searchBox->setShowSearchIcon(true);
    searchBox->setText(m_pEditor->nameFilterPattern());
    connect(searchBox, &SearchBox::textEdited, [=](const QString &text) {
        m_pEditor->setNameFilterPattern(QString("*%1*").arg(text));
    });
    toolbar->addWidget(searchBox);
}

//-------------------------------------------------------------------------------------
QPropertyEditorWidget* PropertyEditorWindow::propertyEditor() const
{
    return m_pEditor;
}

//-------------------------------------------------------------------------------------
float PropertyEditorWindow::screenDpiFactor()
{
    float dpi = 0.0;
    const QScreen *screen = nullptr;
    const QPoint point = geometry().topLeft();

#if (QT_VERSION >= QT_VERSION_CHECK(5,10,0))
    screen = QGuiApplication::screenAt(point);
#else
    // this is a copy of the future implementation of the screenAt method.
    QVarLengthArray<const QScreen*, 8> visitedScreens;

    for (const QScreen *scr : QGuiApplication::screens())
    {
        if (visitedScreens.contains(scr))
        {
            continue;
        }

        // The virtual siblings include the screen itself, so iterate directly
        for (QScreen *sibling : scr->virtualSiblings())
        {
            if (sibling->geometry().contains(point))
            {
                screen = sibling;
                break;
            }

            visitedScreens.append(sibling);
        }

        if (screen != nullptr)
        {
            break;
        }
    }
#endif

    if (screen)
    {
        dpi = screen->logicalDotsPerInch();
    }

    if (dpi <= 0.0)
    {
        dpi = 96.0;
    }

    return qBound(1.0, dpi / 96.0, 1.e10);
}

//------------------------------------------------------------------------------------------------------------------------
class AbstractFigurePrivate
{
public:
    AbstractFigurePrivate() :
        propertyDock(NULL),
        propertyEditorWidget(NULL),
        propertyObservedObject(NULL),
        toolbarsVisible(true),
		windowTitleSuffix(""),
        pMainParent(NULL)
    {
    }

    QList<QMenu*> menus;
    QList<AbstractFigure::ToolBarItem> toolbars;
    QList<AbstractFigure::ToolboxItem> toolboxes;
    QHash<QAction*, QShortcut*> shortcutActions;

    QDockWidget *propertyDock;
    QPropertyEditorWidget *propertyEditorWidget;

	QObject *propertyObservedObject;
    bool toolbarsVisible;
	QString windowTitleSuffix; //cache of current window title suffix (e.g. Figure 102 - Suffix)

    AbstractFigure::WindowMode windowMode;
    QString itomSettingsFile;
    QWidget *pMainParent; //the parent of this figure is only set to m_mainParent, if the stay-on-top behaviour is set to the right value
};

//----------------------------------------------------------------------------------------------------------------------------------
AbstractFigure::AbstractFigure(const QString &itomSettingsFile, WindowMode windowMode, QWidget *parent) :
    QMainWindow(parent),
    AbstractNode(),
    d_ptr(new AbstractFigurePrivate()),
    m_apiFunctionsGraphBasePtr(NULL),
    m_apiFunctionsBasePtr(NULL)
{
    Q_D(AbstractFigure);

    d->windowMode = windowMode;
    d->pMainParent = parent;
    d->itomSettingsFile = itomSettingsFile;

    initialize();
}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractFigure::~AbstractFigure()
{
    Q_D(AbstractFigure);

    if (d->propertyEditorWidget)
    {
        //unregister object in order to prevent a possible crash, if
        //the object is currently being deleted and the property editor
        //tries to update its representation.
        d->propertyEditorWidget->setObject(NULL);
        d->propertyEditorWidget->deleteLater();
    }

    //clear toolbars and menus. toolbars and toolboxes are only added
    //to the main window of the plot in the window modes standaloneInUi or
    //standaloneWindow. If so, they are deleted by the destructor of
    //the main window. Else they have to be deleted here.
	if (d->windowMode == ModeInItomFigure)
	{
		foreach(ToolBarItem t, d->toolbars)
		{
			if (t.toolbar)
			{
				t.toolbar->deleteLater();
			}
		}
	}

	if (d->windowMode == ModeInItomFigure)
	{
        foreach(ToolboxItem t, d->toolboxes)
        {
            if (t.toolbox)
            {
                t.toolbox->deleteLater();
            }
        }
    }

    foreach(QMenu *m, d->menus)
    {
        m->deleteLater();
    }

    d->menus.clear();
    d->toolbars.clear();
    d->toolboxes.clear();

    d->propertyDock = NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractFigure::initialize()
{
    Q_D(AbstractFigure);

    //in all modes, plot is either embedded in itom figureWidget or in external ui-file. Therefore, it is always considered to be a widget
    switch (d->windowMode)
    {
        case AbstractFigure::ModeInItomFigure:
        case AbstractFigure::ModeStandaloneInUi:
            setWindowFlags(Qt::Widget);
            setAttribute(Qt::WA_DeleteOnClose, false);
            menuBar()->setVisible(false);
            break;
        case AbstractFigure::ModeStandaloneWindow:
            setWindowFlags(Qt::Window);
            setAttribute(Qt::WA_DeleteOnClose, true);
            menuBar()->setVisible(true);
            break;
    }

    d->propertyDock = new QDockWidget(tr("Properties"), this);
    d->propertyDock->setVisible(false);
    d->propertyDock->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetMovable);

    auto propEditorWindow = new PropertyEditorWindow(d->propertyDock);
    d->propertyEditorWidget = propEditorWindow->propertyEditor();
    d->propertyDock->setWidget(propEditorWindow);
	addToolbox(d->propertyDock, "properties", Qt::RightDockWidgetArea);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
int AbstractFigure::getPlotID()
{
    Q_D(const AbstractFigure);

    if (!ito::ITOM_API_FUNCS_GRAPH)
        return 0;
    ito::uint32 thisID = 0;
    ito::RetVal retval = apiGetFigureIDbyHandle(this, thisID);

    if (retval.containsError())
    {
        return 0;
    }
    return thisID;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setPropertyObservedObject(QObject* obj)
{
    Q_D(AbstractFigure);

    d->propertyObservedObject = obj;
    if (d->propertyEditorWidget)
    {
        d->propertyEditorWidget->setObject(obj);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::updatePropertyDock()
{
    Q_D(AbstractFigure);

    if (d->propertyEditorWidget && d->propertyObservedObject)
    {
        d->propertyEditorWidget->updateObject(d->propertyObservedObject);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::addMenu(QMenu *menu)
{
    Q_D(AbstractFigure);

    //never adds to menuBar()
    d->menus.append(menu);
}

//----------------------------------------------------------------------------------------------------------------------------------
QList<QMenu*> AbstractFigure::getMenus() const
{
    Q_D(const AbstractFigure);

    if (d->windowMode == AbstractFigure::ModeStandaloneInUi)
    {
        //in standalone mode, this plugin handles its own menus and toolbars
        return QList<QMenu*>();
    }
    else
    {
        return d->menus;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QList<AbstractFigure::ToolBarItem> AbstractFigure::getToolbars() const
{
    Q_D(const AbstractFigure);

    if (d->windowMode == AbstractFigure::ModeStandaloneInUi)
    {
        //in standalone mode, this plugin handles its own menus and toolbars
        return QList<AbstractFigure::ToolBarItem>();
    }
    else
    {
        return d->toolbars;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::addToolBar(QToolBar *toolbar, const QString &key, Qt::ToolBarArea area /*= Qt::TopToolBarArea*/, int section /*= 1*/)
{
    Q_D(AbstractFigure);

    ToolBarItem item;
    item.key = key;
    item.area = area;
    item.toolbar = toolbar;
    item.visible = d->toolbarsVisible;
    item.section = section;

    int maxSection = 1;

    //get highest section for same area
    foreach (const ToolBarItem &titem, d->toolbars)
    {
        if (titem.area == area)
        {
            maxSection = std::max(maxSection, titem.section);
        }
    }
	//this signal is established in order to check if the toolbar war already destroyed
	bool test = connect(toolbar, SIGNAL(destroyed(QObject*)), this, SLOT(toolBarDestroyed(QObject*)));

    d->toolbars.append(item);

    if (d->windowMode == AbstractFigure::ModeStandaloneInUi || d->windowMode == AbstractFigure::ModeStandaloneWindow)
    {
        if (maxSection < section)
        {
            QMainWindow::addToolBarBreak(area);
        }

        QMainWindow::addToolBar(area, toolbar);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::addToolBarBreak(const QString &key, Qt::ToolBarArea area /*= Qt::TopToolBarArea*/)
{
    Q_D(AbstractFigure);

    ToolBarItem item;
    item.key = key;
    item.area = area;
    item.toolbar = NULL;
    item.visible = d->toolbarsVisible;
    item.section = 1;

    //get highest section for same area
    foreach(const ToolBarItem &titem, d->toolbars)
    {
        if (titem.area == area)
        {
            item.section = std::max(item.section, titem.section);
        }
    }

    d->toolbars.append(item);

    if (d->windowMode == AbstractFigure::ModeStandaloneInUi || d->windowMode == AbstractFigure::ModeStandaloneWindow)
    {
        QMainWindow::addToolBarBreak(area);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::showToolBar(const QString &key)
{
    Q_D(AbstractFigure);

    QList<AbstractFigure::ToolBarItem>::iterator i;

    for (i = d->toolbars.begin(); i != d->toolbars.end(); ++i)
    {
        if (i->key == key)
        {
            i->visible = true;
            i->toolbar->setVisible(true && d->toolbarsVisible);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::hideToolBar(const QString &key)
{
    Q_D(AbstractFigure);

    QList<AbstractFigure::ToolBarItem>::iterator i;

    for (i = d->toolbars.begin(); i != d->toolbars.end(); ++i)
    {
        if (i->key == key)
        {
            i->visible = false;
            i->toolbar->setVisible(false /*&& m_toolbarsVisible*/); //always false
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setApiFunctionGraphBasePtr(void **apiFunctionGraphBasePtr)
{
    this->importItomApiGraph(apiFunctionGraphBasePtr);
    m_apiFunctionsGraphBasePtr = apiFunctionGraphBasePtr;
    ito::ITOM_API_FUNCS_GRAPH = apiFunctionGraphBasePtr;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setApiFunctionBasePtr(void **apiFunctionBasePtr)
{
    this->importItomApi(apiFunctionBasePtr);
    m_apiFunctionsBasePtr = apiFunctionBasePtr;
    ito::ITOM_API_FUNCS = apiFunctionBasePtr;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool AbstractFigure::event(QEvent *e)
{
    //the event User+123 is emitted by UiOrganizer, if the API has been prepared and can
    //transmitted to the plugin. This assignment cannot be done directly, since
    //the array ITOM_API_FUNCS is in another scope if called from itom. By sending an
    //event from itom to the plugin, this method is called and ITOM_API_FUNCS is in the
    //right scope. The methods above only set the pointers in the "wrong"-itom-scope (which
    //also is necessary if any methods of the plugin are directly called from itom).
    if (e->type() == (QEvent::User+123))
    {
        //importItomApi(m_apiFunctionsBasePtr);
        //importItomPlotApi(m_apiFunctionsGraphBasePtr);
        init();
    }

    return QMainWindow::event(e);
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setToolbarVisible(bool visible)
{
    Q_D(AbstractFigure);

    QList<AbstractFigure::ToolBarItem>::iterator i;

    for (i = d->toolbars.begin(); i != d->toolbars.end(); ++i)
    {
        if (i->toolbar)
        {
            i->toolbar->setVisible(visible && (*i).visible);
        }
    }

    d->toolbarsVisible = visible;
    updatePropertyDock();
}

//----------------------------------------------------------------------------------------------------------------------------------
bool AbstractFigure::getToolbarVisible() const
{
    Q_D(const AbstractFigure);

    return d->toolbarsVisible;
}

//----------------------------------------------------------------------------------------------------------------------------------
QDockWidget* AbstractFigure::getPropertyDockWidget() const
{
    Q_D(const AbstractFigure);

    return d->propertyDock;
}

//----------------------------------------------------------------------------------------------------------------------------------
QList<AbstractFigure::ToolboxItem> AbstractFigure::getToolboxes() const
{
    Q_D(const AbstractFigure);

    if (d->windowMode == AbstractFigure::ModeStandaloneInUi)
    {
        //in standalone mode, this plugin handles its own menus and toolbars
        return QList<AbstractFigure::ToolboxItem>();
    }
    else
    {
        return d->toolboxes;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::addToolbox(QDockWidget *toolbox, const QString &key, Qt::DockWidgetArea area /*= Qt::RightDockWidgetArea*/)
{
    Q_D(AbstractFigure);

    ToolboxItem item;
    item.key = key;
    item.area = area;
    item.toolbox = toolbox;
    d->toolboxes.append(item);
	//this signal is established in order to check if the docking widget already has been deleted while destruction of mainWindows
	bool test = connect(toolbox, SIGNAL(destroyed(QObject*)), this, SLOT(toolBoxDestroyed(QObject*)));

    switch (d->windowMode)
    {
    case AbstractFigure::ModeInItomFigure:
        /*default if figure is used for plotting data in itom, may also be part of a subfigure area.
        Then, the created DockWidget should be used by the outer window and managed/displayed by it */
        break;
    case AbstractFigure::ModeStandaloneInUi:
        /*figure is contained in an user interface. Then the dock widget is dock with floating mode (default) */
        QMainWindow::addDockWidget(Qt::RightDockWidgetArea, toolbox);
        toolbox->setFloating(true);
        break;

    case AbstractFigure::ModeStandaloneWindow:
        QMainWindow::addDockWidget(Qt::RightDockWidgetArea, toolbox);
        break;
    }
}


//----------------------------------------------------------------------------------------------------------------------------------
bool AbstractFigure::removeToolbox(const QString &key)
{
    Q_D(AbstractFigure);

	bool state = false;
	bool found = true;
	while (found == true)
	{
		int index = 0;
		found = false;
		foreach (ToolboxItem item, d->toolboxes)
		{
			if (item.toolbox == NULL)
			{
				continue;
			}
			if (item.key == key)
			{
				if (item.toolbox->isVisible())
				{
					item.toolbox->hide();
				}
				if (d->windowMode != AbstractFigure::ModeInItomFigure)
				{
					QMainWindow::removeDockWidget(item.toolbox);
				}
				d->toolboxes.removeAt(index);
				state = true;
				found = true;
				break;
			}
			index++;
		}
	}

	return state;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::mnuShowProperties(bool checked)
{
    Q_D(AbstractFigure);

    if (d->propertyDock)
    {
        d->propertyDock->setVisible(checked);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::toolBoxDestroyed(QObject *object)
{
    Q_D(AbstractFigure);

	if (object == NULL)
	{
		return;
	}
	int index = 0;
	foreach (ToolboxItem item, d->toolboxes)
	{
		if (item.toolbox == object)
		{
			d->toolboxes.removeAt(index);
			break;
		}
		index++;
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::toolBarDestroyed(QObject *object)
{
    Q_D(AbstractFigure);

	if (object == NULL)
	{
		return;
	}
	int index = 0;
	foreach (ToolBarItem item, d->toolbars)
	{
		if (item.toolbar == object)
		{
			d->toolbars.removeAt(index);
			break;
		}
		index++;
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractFigure::registerShortcutActions()
{
    Q_D(AbstractFigure);

    QShortcut *shortcut;
    QAction *a;
    QWidget *p = centralWidget();
    foreach(QObject *o, children())
    {
        a = qobject_cast<QAction*>(o);

        if (a && d->shortcutActions.contains(a))
        {
            d->shortcutActions[a]->deleteLater(); //delete a previous shortcut
        }

        if (a && a->shortcut().isEmpty() == false)
        {
            shortcut = new QShortcut(a->shortcut(), p);
            shortcut->setContext(Qt::WidgetWithChildrenShortcut);
            connect(shortcut, SIGNAL(activated()), a, SLOT(trigger()));

            QString text2 = a->text();
            QString text3 = a->text();
            text3.replace("&", "");
            text2 += "\t" + a->shortcut().toString(QKeySequence::NativeText);
            text3 += " (" + a->shortcut().toString(QKeySequence::NativeText) + ")";
            a->setText(text2);
            a->setToolTip(text3);
            a->setShortcut(QKeySequence());
            shortcut->setEnabled(a->isEnabled());

            connect(a, SIGNAL(changed()), this, SLOT(actionChanged())); //to be notified if e.g. the enable property of the action changed
            d->shortcutActions[a] = shortcut;
        }
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::actionChanged()
{
    Q_D(AbstractFigure);

    QObject *s = sender(); //action where any property like enabled changed...
    QAction *a = qobject_cast<QAction*>(s);

    if (a && d->shortcutActions.contains(a))
    {
        d->shortcutActions[a]->setEnabled(a->isEnabled());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setWindowTitleExtension(const QString& title)
{
    Q_D(AbstractFigure);

	if (d->windowTitleSuffix != title)
	{
		d->windowTitleSuffix = title;

		if (title != "")
		{
			emit windowTitleModified(tr(" - ") + title);
		}
		else
		{
			emit windowTitleModified("");
		}
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractFigure::WindowMode AbstractFigure::getWindowMode() const
{
    Q_D(const AbstractFigure);

    return d->windowMode;
}

//----------------------------------------------------------------------------------------------------------------------------------
QString AbstractFigure::getItomSettingsFile() const
{
    Q_D(const AbstractFigure);

    return d->itomSettingsFile;
}

} //end namespace ito

#include "AbstractFigure.moc"
