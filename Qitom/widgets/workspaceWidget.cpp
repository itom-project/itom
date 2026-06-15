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

#include "workspaceWidget.h"
#include "../python/pythonDataObject.h"
#include "../python/pythonEngineInc.h"

#include "../AppManagement.h"
#include "../ui/dialogVariableDetail.h"
#include "../ui/dialogVariableDetailDataObject.h"
#include "helper/guiHelper.h"

#include "dataobj.h"

#include <qdrag.h>
#include <qmessagebox.h>
#include <qsettings.h>
#include <qsharedpointer.h>
#include <qstringlist.h>


namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class WorkspaceWidget
    \brief the workspaceWidget displays either a global or a local workspace given by a python
   dictionary. This widget is inherited from QTreeWidget.
*/

//! constructor
/*!
    \param globalNotLocal true: this widget shows a global python dictionary, false: local
    \param parent parent-widget
*/
WorkspaceWidget::WorkspaceWidget(bool globalNotLocal, QWidget* parent) :
    QTreeWidget(parent), m_globalNotLocal(globalNotLocal), m_workspaceContainer(NULL),
    m_displayItemDetails(NULL)
{
    QStringList headers;

    setDragDropMode(QAbstractItemView::DragOnly);

    setColumnCount(3);
    setEditTriggers(QAbstractItemView::NoEditTriggers);

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    if (m_globalNotLocal)
    {
        headers << tr("Globals") << tr("Value") << tr("Type");
        settings.beginGroup("itomGlobalWorkspaceDockWidget");
    }
    else
    {
        headers << tr("Locals") << tr("Value") << tr("Type");
        settings.beginGroup("itomLocalWorkspaceDockWidget");
    }

    setHeaderLabels(headers);
    setSortingEnabled(true);
    sortByColumn(0, Qt::AscendingOrder);
    setTextElideMode(Qt::ElideMiddle);

    // dont expand or collapse items on double-click (since
    // double-click opens the content information dialog)
    setExpandsOnDoubleClick(false);

    clear();
    m_itemHash.clear();

    QIcon icon(":/application/icons/preferences-python.png");
    const float screenFactor = GuiHelper::screenDpiFactor();
    m_dragPixmap = icon.pixmap(screenFactor * 22, screenFactor * 22);

    /* '__', 'NoneType', 'type',\
         'bool', 'int', 'long', 'float', 'complex',\
         'str', 'unicode', 'tuple', 'list',\
         'dict', 'dict-proxy', 'set', 'file', 'xrange',\
         'slice', 'buffer', 'class', 'instance',\
         'instance method', 'property', 'generator',\
         'function', 'builtin_function_or_method', 'code', 'module',\
         'ellipsis', 'traceback', 'frame', 'other']*/

    m_workspaceContainer = new ito::PyWorkspaceContainer(m_globalNotLocal);

    connect(
        m_workspaceContainer,
        &PyWorkspaceContainer::updateAvailable,
        this,
        &WorkspaceWidget::workspaceContainerUpdated);

    // when double-clicking on an item, its content is showed in
    // DialogVariableDetail-Dialog
    connect(
        this,
        SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)),
        this,
        SLOT(itemDoubleClicked(QTreeWidgetItem*, int)));

    connect(
        this, SIGNAL(itemExpanded(QTreeWidgetItem*)), this, SLOT(itemExpanded(QTreeWidgetItem*)));

    m_displayItemDetails =
        new QAction(QIcon(":/misc/icons/displayDetails.svg"), tr("Display details"), this);
    connect(m_displayItemDetails, SIGNAL(triggered()), this, SLOT(displayItemDetails()));

    int size = settings.beginReadArray("ColWidth");

    for (int i = 0; i < size; ++i)
    {
        settings.setArrayIndex(i);
        setColumnWidth(i, settings.value("width", 100).toInt());
        setColumnHidden(i, columnWidth(i) == 0);
    }

    settings.endArray();
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
Qt::DropActions WorkspaceWidget::supportedDragActions() const
{
    return supportedDropActions() | Qt::CopyAction;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
WorkspaceWidget::~WorkspaceWidget()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);

    if (m_globalNotLocal)
    {
        settings.beginGroup("itomGlobalWorkspaceDockWidget");
    }
    else
    {
        settings.beginGroup("itomLocalWorkspaceDockWidget");
    }

    settings.beginWriteArray("ColWidth");

    for (int i = 0; i < columnCount(); i++)
    {
        settings.setArrayIndex(i);
        settings.setValue("width", columnWidth(i));
    }

    settings.endArray();
    settings.endGroup();

    disconnect(
        m_workspaceContainer,
        &PyWorkspaceContainer::updateAvailable,
        this,
        &WorkspaceWidget::workspaceContainerUpdated);

    m_workspaceContainer->deleteLater();

    disconnect(
        this,
        SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)),
        this,
        SLOT(itemDoubleClicked(QTreeWidgetItem*, int)));

    disconnect(
        this, SIGNAL(itemExpanded(QTreeWidgetItem*)), this, SLOT(itemExpanded(QTreeWidgetItem*)));
}

//----------------------------------------------------------------------------------------------------------------------------------
QStringList WorkspaceWidget::mimeTypes() const
{
    QStringList types = QTreeWidget::mimeTypes();

    if (types.contains("text/plain") == false)
    {
        types.append("text/plain");
    }

    return types;
}

//----------------------------------------------------------------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
QMimeData* WorkspaceWidget::mimeData(const QList<QTreeWidgetItem*> items) const
#else
QMimeData* WorkspaceWidget::mimeData(const QList<QTreeWidgetItem*>& items) const
#endif
{
    QMimeData* mimeData = QTreeWidget::mimeData(items);
    QStringList texts;

    foreach (const QTreeWidgetItem* item, items)
    {
        texts.append(getPythonReadableName(item));
    }

    // text in mimeData must be UTF8 encoded, not Latin1 (since it could also be read by other
    // applications).
    mimeData->setData("text/plain", texts.join("\n").toUtf8());
    return mimeData;
}

//----------------------------------------------------------------------------------------------------------------------------------
QString WorkspaceWidget::getPythonReadableName(const QTreeWidgetItem* item) const
{
    QString name;
    const QTreeWidgetItem* tempItem = nullptr;

    if (item)
    {
        QByteArray type = item->data(0, RoleType).toByteArray();

        if (item->parent() == nullptr)
        {
            name = item->text(0);
        }
        else
        {
            tempItem = item;

            while (tempItem->parent() != nullptr)
            {
                type = tempItem->data(0, RoleType).toByteArray();

                if (type[0] == PY_DICT || type[0] == PY_MAPPING || type[0] == PY_LIST_TUPLE)
                {
                    if (type[1] == PY_NUMBER)
                    {
                        name.prepend("[" + tempItem->text(0) + "]");
                    }
                    else if (type[1] == PY_STRING)
                    {
                        name.prepend("[\"" + tempItem->text(0) + "\"]");
                    }
                    else
                    {
                        name.prepend("[" + tempItem->text(0) + "]");
                    }
                }
                else if (type[0] == PY_ATTR)
                {
                    name.prepend("." + tempItem->text(0));
                }

                tempItem = tempItem->parent();
            }

            name.prepend(tempItem->text(0));
        }
    }

    return name;
}

//----------------------------------------------------------------------------------------------------------------------------------
int WorkspaceWidget::numberOfSelectedMainItems() const
{
    int counter = 0;
    QList<QTreeWidgetItem*> items = selectedItems();

    foreach (const QTreeWidgetItem* item, items)
    {
        if (item->parent() == nullptr)
        {
            counter++;
        }
    }

    return counter;
}

//----------------------------------------------------------------------------------------------------------------------------------
int WorkspaceWidget::numberOfSelectedItems(bool ableToBeRenamed /*= false*/) const
{
    int counter = 0;
    QList<QTreeWidgetItem*> items = selectedItems();

    if (ableToBeRenamed)
    {
        QByteArray type;

        foreach (const QTreeWidgetItem* item, items)
        {
            type = item->data(0, RoleType).toByteArray();

            if (type[0] != PY_LIST_TUPLE)
            {
                // only variables and not index-entries of lists or tuples can be renamed
                counter++;
            }
        }
    }
    else
    {
        counter = items.count();
    }

    return counter;
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceWidget::updateView(
    const QHash<QString, ito::PyWorkspaceItem*>& items,
    const QString& baseName,
    QTreeWidgetItem* parent)
{
    QHash<QString, QTreeWidgetItem*>::const_iterator it;
    QString hashName;
    QTreeWidgetItem* actItem;
    QTreeWidgetItem* tempItem;

    foreach (const ito::PyWorkspaceItem* item, items)
    {
        hashName = baseName + ito::PyWorkspaceContainer::delimiter + item->m_key;
        it = m_itemHash.constFind(hashName);

        if (it != m_itemHash.constEnd())
        {
            actItem = *it;
        }
        else
        {
            actItem = new WorkspaceTreeItem(parent, 0);
            m_itemHash[hashName] = actItem;

            if (parent == nullptr)
            {
                addTopLevelItem(actItem);
                actItem->setFlags(
                    Qt::ItemIsDragEnabled | Qt::ItemIsEditable | Qt::ItemIsSelectable |
                    Qt::ItemIsEnabled);
            }
            else
            {
                actItem->setFlags(
                    Qt::ItemIsDragEnabled | Qt::ItemIsEditable | Qt::ItemIsSelectable |
                    Qt::ItemIsEnabled); // Qt::ItemIsDragEnabled | Qt::ItemIsEnabled);
            }
        }

        actItem->setText(0, item->m_name); // name of variable, key-word of dictionary of index
                                           // number of sequences (list, tuple..)
        actItem->setText(1, item->m_value); // content of variable
        actItem->setText(2, item->m_type); // data type
        actItem->setData(0, RoleFullName, hashName);
        actItem->setData(0, RoleCompatibleTypes, item->m_compatibleParamBaseType);

        // m_key is ab:name where a is
        // [PY_LIST_TUPLE,PY_MAPPING,PY_DICT,PY_ATTR]
        // and b is [PY_NUMBER or PY_STRING]
        actItem->setData(0, RoleType, item->m_key.left(2).toLatin1());

        if (item->m_childState == ito::PyWorkspaceItem::stateNoChilds)
        {
            actItem->setChildIndicatorPolicy(QTreeWidgetItem::DontShowIndicator);

            while (actItem->childCount() > 0)
            {
                tempItem = actItem->child(0);
                recursivelyDeleteHash(tempItem->data(0, RoleFullName).toString());
                actItem->removeChild(actItem->child(0));
            }
        }
        else
        {
            actItem->setChildIndicatorPolicy(QTreeWidgetItem::ShowIndicator);

            if (item->m_childs.count() == 0) // item has children, but they are not shown yet
            {
                while (actItem->childCount() > 0)
                {
                    tempItem = actItem->child(0);
                    recursivelyDeleteHash(tempItem->data(0, RoleFullName).toString());
                    actItem->removeChild(actItem->child(0));
                }
            }
            else
            {
                updateView(item->m_childs, hashName, actItem);
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceWidget::workspaceContainerUpdated(
    PyWorkspaceItem* rootItem, QString fullNameRoot, QStringList recentlyDeletedFullNames)
{
    QTreeWidgetItem* parent = nullptr;
    QTreeWidgetItem* temp;

    if (m_workspaceContainer)
    {
        if (m_workspaceContainer->isRoot(rootItem))
        {
            if (rootItem->m_childs.count() == 0)
            {
                m_itemHash.clear();
                clear();
                recentlyDeletedFullNames.clear();
            }

            parent = nullptr;
        }
        else
        {
            auto it = m_itemHash.constFind(fullNameRoot);

            if (it == m_itemHash.constEnd())
            {
                return; // error
            }

            parent = *it;
        }

        QHash<QString, QTreeWidgetItem*>::const_iterator it;

        foreach (const QString& deleteHashName, recentlyDeletedFullNames)
        {
            temp = nullptr;
            it = m_itemHash.constFind(deleteHashName);

            if (it != m_itemHash.constEnd())
            {
                temp = (*it);
            }

            recursivelyDeleteHash(deleteHashName);

            DELETE_AND_SET_NULL(temp);
        }

        if (m_workspaceContainer->m_accessMutex.tryLock(1000))
        {
            updateView(rootItem->m_childs, fullNameRoot, parent);
            m_workspaceContainer->m_accessMutex.unlock();
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceWidget::recursivelyDeleteHash(QTreeWidgetItem* item)
{
    if (item)
    {
        QString fullName = item->data(0, RoleFullName).toString();
        recursivelyDeleteHash(fullName);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceWidget::recursivelyDeleteHash(const QString& fullBaseName)
{
    QHash<QString, QTreeWidgetItem*>::iterator it = m_itemHash.find(fullBaseName);
    if (it != m_itemHash.end())
    {
        for (int i = 0; i < it.value()->childCount(); i++)
        {
            recursivelyDeleteHash(it.value()->child(i));
        }

        m_itemHash.erase(it);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceWidget::displayItemDetails()
{
    QList<QTreeWidgetItem*> itemList = selectedItems();
    foreach (QTreeWidgetItem* item, itemList)
    {
        if (item)
        {
            itemDoubleClicked(item, 0);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot, invoked if item is double-clicked
/*!
    opens instance of DialogVariableDetail to show detailed information about the double-clicked
   variable

    \param item QTreeWidgetItem which has been clicked
    \sa DialogVariableDetail
*/
void WorkspaceWidget::itemDoubleClicked(QTreeWidgetItem* item, int /*column*/)
{
    QString extendedValue = "";
    QString name;
    QSharedPointer<QString> tempValue;
    QTreeWidgetItem* tempItem = nullptr;
    QString fullName("empty item");
    QByteArray type;
    QSharedPointer<ito::ParamBase> value;
    QStringList key;
    QVector<int> paramBaseTypes; // Type of ParamBase, which is compatible to this value,
    QSharedPointer<ito::DataObject> dObj;
    const ito::DataObject* obj = nullptr;

    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    ito::PyWorkspaceItem* item2 = nullptr;

    if (item)
    {
        fullName = item->data(0, RoleFullName).toString();
        type = item->data(0, RoleType).toByteArray();
        item2 = m_workspaceContainer->getItemByFullName(fullName);
        extendedValue = item2->m_extendedValue;

        name = getPythonReadableName(item);
    }

    // ask python to get extendedValue, since this value has been complex
    // such that it hasn't been evaluated at runtime before
    if (extendedValue == "")
    {
        if (eng)
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            tempValue = QSharedPointer<QString>(new QString());
            QMetaObject::invokeMethod(
                eng,
                "workspaceGetValueInformation",
                Q_ARG(PyWorkspaceContainer*, m_workspaceContainer),
                Q_ARG(QString, fullName),
                Q_ARG(QSharedPointer<QString>, tempValue),
                Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

            if (!locker.getSemaphore()->waitAndProcessEvents(1000))
            {
                extendedValue = tr("timeout while asking python for detailed information");
            }
            else
            {
                extendedValue = *tempValue;
            }
        }
    }

    if (item2->m_compatibleParamBaseType == ito::ParamBase::DObjPtr)
    {
        key.append(item->data(0, WorkspaceWidget::RoleFullName).toString());
        paramBaseTypes.append(item->data(0, WorkspaceWidget::RoleCompatibleTypes).toInt());

        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QSharedPointer<SharedParamBasePointerVector> values(new SharedParamBasePointerVector());
        QMetaObject::invokeMethod(
            eng,
            "getParamsFromWorkspace",
            Q_ARG(bool, m_globalNotLocal),
            Q_ARG(QStringList, key),
            Q_ARG(QVector<int>, paramBaseTypes),
            Q_ARG(QSharedPointer<SharedParamBasePointerVector>, values),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        if (!locker.getSemaphore()->wait(5000))
        {
            extendedValue = tr("timeout while asking python for detailed information");
        }

        for (int i = 0; i < values->size(); ++i)
        {
            if (values->at(i)->getType() == (ito::ParamBase::DObjPtr & ito::paramTypeMask))
            {
                obj = (*values)[i]->getVal<const ito::DataObject*>();
                dObj = QSharedPointer<ito::DataObject>(new ito::DataObject(*obj));
                break;
            }
        }

        ito::RetVal retval = locker->returnValue;

        if (dObj.isNull())
        {
            retval += ito::RetVal(
                ito::retError, 0, "The object could not be transformed into a dataObject.");
        }

        if (retval.containsError())
        {
            extendedValue = tr("No table view possible, since the given object could not be "
                               "converted to a dataObject.") +
                "\n\n" + extendedValue;

            DialogVariableDetail* dlg =
                new DialogVariableDetail(name, item->text(2), extendedValue, this);
            dlg->setAttribute(Qt::WA_DeleteOnClose, true);
            dlg->setModal(false);
            dlg->show();
            dlg->raise();
            dlg->activateWindow();
        }
        else
        {
            DialogVariableDetailDataObject* dlg = new DialogVariableDetailDataObject(
                name,
                item->text(2),
                PythonDataObject::typeNumberToName(obj->getType()),
                dObj,
                this);
            dlg->setAttribute(Qt::WA_DeleteOnClose, true);
            dlg->setModal(false);
            dlg->show();
            dlg->raise();
            dlg->activateWindow();
        }
    }
    else
    {
        DialogVariableDetail* dlg =
            new DialogVariableDetail(name, item->text(2), extendedValue, this);
        dlg->setAttribute(Qt::WA_DeleteOnClose, true);
        dlg->setModal(false);
        dlg->show();
        dlg->raise();
        dlg->activateWindow();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceWidget::itemExpanded(QTreeWidgetItem* item)
{
    QString fullName = item->data(0, RoleFullName).toString();
    m_workspaceContainer->m_accessMutex.lock();
    m_workspaceContainer->m_expandedFullNames.insert(fullName);
    m_workspaceContainer->m_accessMutex.unlock();

    if (item->childCount() == 0) // childs have not been submitted by python yet
    {
        m_workspaceContainer->emitGetChildNodes(m_workspaceContainer, fullName);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceWidget::itemCollapsed(QTreeWidgetItem* item)
{
    QString fullName = item->data(0, RoleFullName).toString();
    m_workspaceContainer->m_accessMutex.lock();
    m_workspaceContainer->m_expandedFullNames.remove(fullName);
    m_workspaceContainer->m_accessMutex.unlock();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceWidget::startDrag(Qt::DropActions supportedActions)
{
    // QTreeWidget::startDrag(supportedActions);
    QList<QTreeWidgetItem*> items = selectedItems();
    if (items.count() > 0)
    {
        QMimeData* data = mimeData(items);
        if (!data)
        {
            return;
        }

        QRect rect;
        QDrag* drag = new QDrag(this);
        drag->setPixmap(m_dragPixmap);
        drag->setMimeData(data);
        Qt::DropAction defaultDropAction = Qt::IgnoreAction;

        if (this->defaultDropAction() != Qt::IgnoreAction &&
            (supportedActions & this->defaultDropAction()))
        {
            defaultDropAction = this->defaultDropAction();
        }
        else if (
            supportedActions & Qt::CopyAction && dragDropMode() != QAbstractItemView::InternalMove)
        {
            defaultDropAction = Qt::CopyAction;
        }
        drag->exec(supportedActions, defaultDropAction);
    }
}

} // end namespace ito
