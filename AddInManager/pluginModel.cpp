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

#include "pluginModel.h"
#include "addInManager.h"
#include <qfileinfo.h>
#include <qmimedata.h>

#include "../common/semVerVersion.h"

namespace ito {

class PlugInModelPrivate
{
public:
    PlugInModelPrivate() : m_pAIM(nullptr), m_bgColorItemsWithPythonRef(QColor(255, 255, 175)){};
    ~PlugInModelPrivate(){};

public:
    QList<QString> m_headers; //!<  string list of names of column headers
    QList<QVariant> m_alignment; //!<  list of alignments for the corresponding headers

    int m_treeFixNodes[6];
    QModelIndex m_treeFixIndizes[6];

    QIcon m_iconActuator;
    QIcon m_iconGrabber;
    QIcon m_iconADDA;
    QIcon m_iconRawIO;
    QIcon m_iconFilter;
    QIcon m_iconDataIO;
    QIcon m_iconAlgo;
    QIcon m_iconWidget;
    QIcon m_iconPlots;

    QColor m_bgColorItemsWithPythonRef;

    ito::AddInManager* m_pAIM;
};

//-------------------------------------------------------------------------------------
/** constructor
 *
 *   contructor, creating column headers for the tree view
 */
PlugInModel::PlugInModel(ito::AddInManager* addInManager, QObject* parent /*= nullptr*/) :
    QAbstractItemModel(parent), d_ptr(new PlugInModelPrivate())
{
    Q_D(PlugInModel);

    d->m_pAIM = addInManager;

    d->m_headers << tr("Name") << tr("Type") << tr("Version") << tr("Filename") << tr("Author")
                 << tr("min. itom Version") << tr("max. itom Version") << tr("Description");
    d->m_alignment << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft)
                   << QVariant(Qt::AlignCenter) << QVariant(Qt::AlignLeft)
                   << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignRight)
                   << QVariant(Qt::AlignRight) << QVariant(Qt::AlignLeft);

    d->m_iconActuator = QIcon(":/plugins/icons/pluginActuator.png");
    d->m_iconGrabber = QIcon(":/plugins/icons/pluginGrabber.png");
    d->m_iconADDA = QIcon(":/plugins/icons/pluginADDA.png");
    d->m_iconRawIO = QIcon(":/plugins/icons/pluginRawIO.png");
    d->m_iconFilter = QIcon(":/plugins/icons/pluginFilter.png");
    d->m_iconDataIO = QIcon(":/plugins/icons/plugin.png");
    d->m_iconAlgo = QIcon(":/plugins/icons/pluginAlgo.png");
    d->m_iconWidget = QIcon(":/plugins/icons/window.png");

    d->m_treeFixNodes[0] = typeDataIO;
    d->m_treeFixNodes[1] = typeActuator;
    d->m_treeFixNodes[2] = typeAlgo;
    d->m_treeFixNodes[3] = typeDataIO | typeGrabber;
    d->m_treeFixNodes[4] = typeDataIO | typeADDA;
    d->m_treeFixNodes[5] = typeDataIO | typeRawIO;

    d->m_treeFixIndizes[0] = createIndex(0, 0, &(d->m_treeFixNodes[0])); // level 0
    d->m_treeFixIndizes[1] = createIndex(1, 0, &(d->m_treeFixNodes[1])); // level 0
    d->m_treeFixIndizes[2] = createIndex(2, 0, &(d->m_treeFixNodes[2])); // level 0
    d->m_treeFixIndizes[3] = createIndex(0, 0, &(d->m_treeFixNodes[3])); // level 1
    d->m_treeFixIndizes[4] = createIndex(1, 0, &(d->m_treeFixNodes[4])); // level 1
    d->m_treeFixIndizes[5] = createIndex(2, 0, &(d->m_treeFixNodes[5])); // level 1
}

//-------------------------------------------------------------------------------------
/** destructor - clean up, clear header and alignment list
 *
 */
PlugInModel::~PlugInModel()
{
    Q_D(PlugInModel);

    d->m_headers.clear();
    d->m_alignment.clear();
    return;
}

QModelIndex PlugInModel::getTypeNode(const int type) const
{
    Q_D(const PlugInModel);

    for (unsigned int i = 0; i < (sizeof(d->m_treeFixNodes) / sizeof(d->m_treeFixNodes[0])); i++)
    {
        if (type == d->m_treeFixNodes[i])
        {
            return d->m_treeFixIndizes[i];
        }
    }
    return QModelIndex();
}

//-------------------------------------------------------------------------------------
/**
 *   @param
 *   @return
 */
Qt::ItemFlags PlugInModel::flags(const QModelIndex& index) const
{
    Q_D(const PlugInModel);

    if (!index.isValid())
    {
        return Qt::NoItemFlags;
    }

    tItemType itemType;
    size_t itemInternalData;

    if (!getModelIndexInfo(index, itemType, itemInternalData))
    {
        return Qt::NoItemFlags;
    }

    if (itemType == itemInstance)
    {
        ito::AddInBase* aib = (ito::AddInBase*)(itemInternalData);

        if (d->m_pAIM->isPluginInstanceDead(aib))
        {
            return Qt::ItemIsSelectable;
        }
        else
        {
            return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
        }
    }

    if (!getIsAlgoPlugIn(itemType, itemInternalData) &&
        (itemType & (itemPlugin | itemFilter | itemWidget)))
    {
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsDragEnabled;
    }

    return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

//-------------------------------------------------------------------------------------
/** return parent element
 *   @param [in] index   the element's index for which the parent should be returned
 *   @return     the parent element.
 *
 */
QModelIndex PlugInModel::parent(const QModelIndex& index) const
{
    Q_D(const PlugInModel);

    if (!index.isValid())
    {
        return QModelIndex();
    }

    tItemType itemType;
    size_t itemInternalData;

    if (!getModelIndexInfo(index, itemType, itemInternalData))
    {
        return QModelIndex();
    }

    switch (itemType)
    {
    case itemCatDataIO:
    case itemCatActuator:
    case itemCatAlgo: {
        return QModelIndex();
    }
    case itemSubCategoryDataIO_Grabber:
    case itemSubCategoryDataIO_ADDA:
    case itemSubCategoryDataIO_RawIO: {
        return d->m_treeFixIndizes[0];
    }
    case itemPlugin: {
        ito::AddInInterfaceBase* aiib = (ito::AddInInterfaceBase*)(itemInternalData);
        return getTypeNode(aiib->getType());
        /*for (int i = 0 ; i < sizeof(m_treeFixNodes) / sizeof(m_treeFixNodes[0]) ; i++)
        {
            if (aiib->getType() == m_treeFixNodes[i])
            {
                return m_treeFixIndizes[i];
            }
        }*/
    }
    case itemInstance: {
        ito::AddInBase* aib = (ito::AddInBase*)(itemInternalData);
        ito::AddInInterfaceBase* aiib = aib->getBasePlugin();

        if (aiib->getType() & ito::typeActuator)
        {
            for (int i = 0; i < d->m_pAIM->getActList()->count(); i++)
            {
                if (d->m_pAIM->getActList()->at(i) == (QObject*)aiib)
                {
                    return createIndex(i, 0, (void*)aiib);
                }
            }

            return QModelIndex();
        }
        else if (aiib->getType() & ito::typeDataIO)
        {
            int rowCounter = -1;
            ito::AddInInterfaceBase* aiib2 = nullptr;

            for (int i = 0; i < d->m_pAIM->getDataIOList()->count(); i++)
            {
                aiib2 = (ito::AddInInterfaceBase*)d->m_pAIM->getDataIOList()->at(i);

                if (aiib2->getType() == aiib->getType())
                {
                    rowCounter++;
                }

                if (aiib2 == aiib)
                {
                    return createIndex(rowCounter, 0, (void*)aiib);
                }
            }

            return QModelIndex();
        }
        else
        {
            return QModelIndex();
        }
    }
    case itemFilter: {
        ito::AddInAlgo::FilterDef* filter = (ito::AddInAlgo::FilterDef*)(itemInternalData);

        for (int i = 0; i < d->m_pAIM->getAlgList()->count(); i++)
        {
            if (d->m_pAIM->getAlgList()->at(i) == (QObject*)filter->m_pBasePlugin)
            {
                return createIndex(i, 0, (void*)filter->m_pBasePlugin);
            }
        }

        return QModelIndex();
    }
    case itemWidget: {
        ito::AddInAlgo::AlgoWidgetDef* widget = (ito::AddInAlgo::AlgoWidgetDef*)(itemInternalData);

        for (int i = 0; i < d->m_pAIM->getAlgList()->count(); i++)
        {
            if (d->m_pAIM->getAlgList()->at(i) == (QObject*)widget->m_pBasePlugin)
            {
                return createIndex(i, 0, (void*)widget->m_pBasePlugin);
            }
        }
        return QModelIndex();
    }
    default: {
        return QModelIndex();
    }
    }
}

//-------------------------------------------------------------------------------------
/** return number of rows
 *   @param [in] parent parent of current item
 *   @return     returns 0 for all child-child elements, the number of instances for child elements
 * (a plugin) and the number of plugins for a root element
 */
int PlugInModel::rowCount(const QModelIndex& parent) const
{
    Q_D(const PlugInModel);

    if (parent.isValid() == false)
    {
        return 3;
    }
    else // parent valid
    {
        tItemType parentType;
        size_t parentInternalData;

        if (!getModelIndexInfo(parent, parentType, parentInternalData))
        {
            return 0;
        }

        switch (parentType)
        {
        case itemCatDataIO: {
            return 3; // three sub-categories of dataIO
        }
        case itemCatActuator: {
            return d->m_pAIM->getActList()->count();
        }
        case itemCatAlgo: {
            return d->m_pAIM->getAlgList()->count();
        }
        case itemSubCategoryDataIO_Grabber:
        case itemSubCategoryDataIO_ADDA:
        case itemSubCategoryDataIO_RawIO: {
            int counter = 0;
            const QList<QObject*>* dataIOs = d->m_pAIM->getDataIOList();
            ito::AddInInterfaceBase* aiib = nullptr;
            for (int i = 0; i < dataIOs->count(); i++)
            {
                aiib = qobject_cast<ito::AddInInterfaceBase*>(dataIOs->at(i));
                if ((size_t)aiib->getType() == parentInternalData /*subtype*/)
                {
                    counter++;
                }
            }
            return counter;
        }
        case itemPlugin: {
            ito::AddInInterfaceBase* aiib = (ito::AddInInterfaceBase*)(parentInternalData);
            if (aiib->getType() & ito::typeAlgo)
            {
                ito::AddInAlgo* aia = (ito::AddInAlgo*)(aiib->getInstList()[0]);
                QHash<QString, ito::AddInAlgo::FilterDef*> filters;
                QHash<QString, ito::AddInAlgo::AlgoWidgetDef*> widgets;
                aia->getFilterList(filters);
                aia->getAlgoWidgetList(widgets);
                return filters.size() + widgets.size();
            }
            else
            {
                return aiib->getInstList().count();
            }
        }
        /*case itemInstance:
        case itemFilter:
        case itemWidget:
        case itemUnknown:*/
        default: {
            return 0;
        }
        }
    }
}

//-------------------------------------------------------------------------------------
/** return column count
 *   @param [in] parent parent of current item
 *   @return     2 for child elements (instances) and the header size for root elements (plugins)
 */
int PlugInModel::columnCount(const QModelIndex& /*parent*/) const
{
    Q_D(const PlugInModel);

    return d->m_headers.size();
}

//-------------------------------------------------------------------------------------
/** return current index element
 *   @param [in] row row of current element
 *   @param [in] column column of current element
 *   @param [in] parent  parent of current element
 *   @return QModelIndex - element at current index
 *
 *   This method returns the QModelIndex for the current element. As the tree structure is not
 * cached it has to be "calculated" on each call. An invalid parent means were in the top most
 * "plane" of the tree, i.e. the plugin-plane. If the passed index is out of range we return an
 * empty element. Otherwise a new element marked as root level element (i.e. interal pointer =
 * ROOTPOINTER) is returned. If the parent element is valid the index for an instance is requested.
 * In that case it is first checked if the index for a child child element is queried. In that case
 * again an empty element is returned else the plugin for the selected instance is searched in the
 * plugin lists and an according index is created.
 */
QModelIndex PlugInModel::index(int row, int column, const QModelIndex& parent) const
{
    Q_D(const PlugInModel);

    if (!hasIndex(row, column, parent))
    {
        return QModelIndex();
    }

    if (parent.isValid() == false)
    {
        if (row >= 0 && row <= 2)
        {
            return createIndex(row, column, (void*)&d->m_treeFixNodes[row]);
        }

        return QModelIndex();
    }
    else // parent valid
    {
        tItemType parentType;
        size_t parentInternalData;

        if (!getModelIndexInfo(parent, parentType, parentInternalData))
        {
            return QModelIndex();
        }

        switch (parentType)
        {
        case itemCatDataIO: {
            return createIndex(row, column, (void*)(&d->m_treeFixNodes[row + 3]));
        }
        case itemCatActuator: {
            return createIndex(row, column, (void*)d->m_pAIM->getActList()->at(row));
        }
        case itemCatAlgo: {
            return createIndex(row, column, (void*)d->m_pAIM->getAlgList()->at(row));
        }
        case itemSubCategoryDataIO_Grabber:
        case itemSubCategoryDataIO_ADDA:
        case itemSubCategoryDataIO_RawIO: {
            int counter = -1;
            const QList<QObject*>* dataIOs = d->m_pAIM->getDataIOList();
            ito::AddInInterfaceBase* aiib = nullptr;

            for (int i = 0; i < dataIOs->count(); i++)
            {
                aiib = qobject_cast<ito::AddInInterfaceBase*>(dataIOs->at(i));
                if ((size_t)aiib->getType() == parentInternalData /*subtype*/)
                {
                    counter++;
                    if (counter == row)
                    {
                        return createIndex(row, column, (void*)aiib);
                    }
                }
            }
        }
        case itemPlugin: {
            ito::AddInInterfaceBase* aiib = (ito::AddInInterfaceBase*)(parentInternalData);

            if (aiib->getType() & ito::typeAlgo)
            {
                ito::AddInAlgo* aia = (ito::AddInAlgo*)(aiib->getInstList()[0]);
                QHash<QString, ito::AddInAlgo::FilterDef*> filters;
                QHash<QString, ito::AddInAlgo::AlgoWidgetDef*> widgets;
                aia->getFilterList(filters);
                aia->getAlgoWidgetList(widgets);

                if (filters.count() > row)
                {
                    return createIndex(row, column, (void*)filters.values()[row]);
                }
                else
                {
                    aia->getAlgoWidgetList(widgets);
                    // qDebug() << "AlgoWidget: r" << row << ", c:" << column << ", p:" <<
                    // (void*)widgets.values()[row - filters.count()];
                    return createIndex(row, column, (void*)widgets.values()[row - filters.count()]);
                }
            }
            else
            {
                return createIndex(row, column, (void*)aiib->getInstList()[row]);
            }
        }
        /*case itemInstance:
        case itemFilter:
        case itemWidget:
        case itemUnknown:*/
        default: {
            return QModelIndex();
        }
        }
    }
}

//-------------------------------------------------------------------------------------
/** return the header / captions for the tree view model
 *
 */
QVariant PlugInModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    Q_D(const PlugInModel);

    if (role == Qt::DisplayRole && orientation == Qt::Horizontal)
    {
        if (section >= 0 && section < d->m_headers.size())
        {
            return d->m_headers.at(section);
        }
        return QVariant();
    }

    return QVariant();
}

//-------------------------------------------------------------------------------------
QColor PlugInModel::backgroundColorInstancesWithPythonRef() const
{
    Q_D(const PlugInModel);

    return d->m_bgColorItemsWithPythonRef;
}

//-------------------------------------------------------------------------------------
void PlugInModel::setBackgroundColorInstancesWithPythonRef(const QColor& bgColor)
{
    Q_D(PlugInModel);

    if (d->m_bgColorItemsWithPythonRef != bgColor)
    {
        beginResetModel();
        d->m_bgColorItemsWithPythonRef = bgColor;
        endResetModel();
    }
}

//-------------------------------------------------------------------------------------
/**
 *   @param index
 *   @param type
 *   @param internalData
 *   @return bool
 *
 *
 */
bool PlugInModel::getModelIndexInfo(
    const QModelIndex& index, tItemType& type, size_t& internalData) const
{
    Q_D(const PlugInModel);

    type = itemUnknown;
    internalData = 0;

    if (!index.isValid())
    {
        return false;
    }

    // table of type vs. internalData
    // itemUnknown           -> 0
    // itemCatDataIO         -> ito::typeDataIO
    // itemCatActuator       -> ito::typeActuator
    // itemCatAlgo           -> ito::typeAlgo
    // itemSubCategoryDataIO_... -> Content of m_treeFixNodes[3,4,5] (OR-combination of
    // ito::typeDataIO and sub-type) itemPlugin            -> Pointer to corresponding
    // AddInInterfaceBase itemInstance          -> Pointer to corresponding AddInBase itemFilter ->
    // Pointer to corresponding FilterDef itemWidget            -> Pointer to corresponding
    // AlgoWidgetDef

    // check if item is of type itemCategory or itemSubCategory
    const int* ptr1 = &d->m_treeFixNodes[0];
    void* internalPtr = index.internalPointer();
    QObject* obj = nullptr;
    //    int rowIndex;

    if (internalPtr >= ptr1 && internalPtr <= (ptr1 + sizeof(d->m_treeFixNodes)))
    {
        switch (*(int*)index.internalPointer())
        {
        case typeAlgo: {
            type = itemCatAlgo;
            internalData = ito::typeAlgo;
            break;
        }
        case typeDataIO: {
            type = itemCatDataIO;
            internalData = ito::typeDataIO;
            break;
        }
        case typeActuator: {
            type = itemCatActuator;
            internalData = ito::typeActuator;
            break;
        }
        case typeDataIO | typeGrabber: {
            // sub category of dataIO
            type = itemSubCategoryDataIO_Grabber;
            internalData = *(int*)index.internalPointer();
            break;
        }
        case typeDataIO | typeADDA: {
            // sub category of dataIO
            type = itemSubCategoryDataIO_ADDA;
            internalData = *(int*)index.internalPointer();
            break;
        }
        case typeDataIO | typeRawIO: {
            // sub category of dataIO
            type = itemSubCategoryDataIO_RawIO;
            internalData = *(int*)index.internalPointer();
            break;
        }
        default: {
            return false;
        }
        }
        return true;
    }
    else
    {
        // check if item is a filter
        // check type of element
        const QHash<QString, ito::AddInAlgo::FilterDef*>* filters = d->m_pAIM->getFilterList();
        QHash<QString, ito::AddInAlgo::FilterDef*>::const_iterator i = filters->constBegin();
        while (i != filters->constEnd())
        {
            // check if index corresponds to this filter
            if ((void*)i.value() == internalPtr)
            {
                type = itemFilter;
                internalData = (size_t)(i.value());
                return true;
            }
            ++i;
        }

        // check if item is a widget
        const QHash<QString, ito::AddInAlgo::AlgoWidgetDef*>* widgets =
            d->m_pAIM->getAlgoWidgetList();
        QHash<QString, ito::AddInAlgo::AlgoWidgetDef*>::const_iterator j = widgets->constBegin();
        while (j != widgets->constEnd())
        {
            // check if index corresponds to this widget
            if ((void*)j.value() == internalPtr)
            {
                type = itemWidget;
                internalData = (size_t)(j.value());
                return true;
            }
            ++j;
        }

        // if the element is no filter and no widget, it only can be a plugin (the DLL itself) or an
        // instance of a plugin
        obj = (QObject*)internalPtr;

        ito::AddInInterfaceBase* aiib = qobject_cast<ito::AddInInterfaceBase*>(obj);
        if (aiib) // it is a plugin
        {
            type = itemPlugin;
            internalData = (size_t)aiib;
            return true;
        }

        if (obj->inherits("ito::AddInBase"))
        {
            ito::AddInBase* aib = reinterpret_cast<ito::AddInBase*>(obj);
            if (aib) // it is an instance
            {
                type = itemInstance;
                internalData = (size_t)aib;
                return true;
            }
        }
    }

    return false;
}

//-------------------------------------------------------------------------------------
/**
 *   @param index
 *   @param name
 *   @param itemType
 *   @param role
 *   @param icon
 *   @return QVariant
 *
 *
 */
QVariant PlugInModel::getFixedNodeInfo(
    const QModelIndex& index,
    const QVariant& name,
    const tItemType& itemType,
    const int& role,
    const QIcon icon) const
{
    Q_D(const PlugInModel);

    if (role == Qt::DisplayRole)
    {
        if (index.column() == 0)
        {
            return name;
        }
        else
        {
            return QVariant();
        }
    }
    else if (role == Qt::DecorationRole)
    {
        if (index.column() == 0)
        {
            return icon;
        }
        else
        {
            return QVariant();
        }
    }
    else if (role == Qt::ToolTipRole)
    {
        if (index.column() == 0)
        {
            return name;
        }
        else
        {
            return QVariant();
        }
    }
    else if (role == Qt::TextAlignmentRole)
    {
        if (index.column() == 0)
        {
            return d->m_alignment[index.column()];
        }
        else
        {
            return QVariant();
        }
    }
    else if (role == Qt::UserRole + 1) // returns type (OR-combination)
    {
        switch (itemType)
        {
        case itemCatDataIO: {
            return ito::typeDataIO;
        }
        case itemCatActuator: {
            return ito::typeActuator;
        }
        case itemCatAlgo: {
            return ito::typeAlgo;
        }
        case itemSubCategoryDataIO_Grabber: {
            return ito::typeDataIO | ito::typeGrabber;
        }
        case itemSubCategoryDataIO_ADDA: {
            return ito::typeDataIO | ito::typeADDA;
        }
        case itemSubCategoryDataIO_RawIO: {
            return ito::typeDataIO | ito::typeRawIO;
        }
        default: {
            return QVariant();
        }
        }
        return itemType;
    }
    else if (role == Qt::UserRole + 2)
    {
        // returns true if item is a category or sub-category, else  false
        return true;
    }
    else if (role == Qt::UserRole + 3)
    {
        // returns tItemType
        return itemType;
    }

    return QVariant();
}

//-------------------------------------------------------------------------------------
/**
 *   @param index
 *   @param role
 *   @return QVariant
 *
 *
 */
QVariant PlugInModel::getPluginNodeInfo(const QModelIndex& index, const int& role) const
{
    Q_D(const PlugInModel);

    ito::AddInInterfaceBase* aib = (ito::AddInInterfaceBase*)index.internalPointer();

    if (role == Qt::DisplayRole)
    {
        switch (index.column())
        {
        case 0: // name
        {
            return aib->objectName();
        }
        case 1: // type
            switch (aib->getType())
            {
            case typeActuator: {
                return tr("Actuator");
            }
            case typeDataIO | typeGrabber: {
                return tr("Grabber");
            }
            case typeDataIO | typeADDA: {
                return tr("ADDA");
            }
            case typeDataIO | typeRawIO: {
                return tr("Raw IO");
            }
            case typeAlgo: {
                return tr("Algorithm");
            }
            default:
                return QVariant();
            }
        case 2: // version
        {
            SemVerVersion version = SemVerVersion::fromInt(aib->getVersion());
            return version.toString();
        }
        case 3: // filename
        {
            QFileInfo filename;
            filename = QFileInfo(aib->getFilename());
            return filename.fileName();
        }
        case 4: // autor
        {
            return aib->getAuthor();
        }
        case 5: // minversion
        {
            SemVerVersion version = SemVerVersion::fromInt(aib->getMinItomVer());
            if (version.toInt() != MINVERSION)
            {
                return version.toString();
            }
            return QString("-");
        }
        case 6: // maxversion
        {
            SemVerVersion version = SemVerVersion::fromInt(aib->getMaxItomVer());
            if (version.toInt() != MAXVERSION)
            {
                return version.toString();
            }
            return QString("-");
        }
        case 7: // description
        {
            return aib->getDescription();
        }
        default: {
            return QVariant();
        }
        }
    }
    else if (role == Qt::DecorationRole)
    {
        if (index.column() == 0)
        {
            switch (aib->getType())
            {
            case typeActuator:
                return d->m_iconActuator;
                break;
            case typeDataIO | typeGrabber:
                return d->m_iconGrabber;
                break;
            case typeDataIO | typeADDA:
                return d->m_iconADDA;
                break;
            case typeDataIO | typeRawIO:
                return d->m_iconRawIO;
                break;
            case typeAlgo:
                return d->m_iconAlgo;
                break;
            default:
                return QVariant();
            }
        }
        else
        {
            return QVariant();
        }
    }
    else if (role == Qt::ToolTipRole)
    {
        switch (index.column())
        {
        case 0: // name
            return aib->objectName();
        case 3: // filename
            return aib->getFilename();
        case 7: // description
            return aib->getDescription();
        default:
            return QVariant();
        }
    }
    else if (role == Qt::TextAlignmentRole)
    {
        if (index.column() >= 0 && index.column() < d->m_alignment.size())
        {
            return d->m_alignment[index.column()];
        }
        else
        {
            return QVariant();
        }
    }
    else if (role == Qt::UserRole + 1)
    {
        // returns type (OR-combination)
        return aib->getType();
    }
    else if (role == Qt::UserRole + 2)
    {
        // returns true if item is a category or sub-category, else false
        return false;
    }
    else if (role == Qt::UserRole + 3) // returns tItemType
    {
        return itemPlugin;
    }

    return QVariant();
}

//-------------------------------------------------------------------------------------
/**
 *   @param index
 *   @param role
 *   @return QVariant
 *
 *
 */
QVariant PlugInModel::getInstanceNodeInfo(const QModelIndex& index, const int& role) const
{
    Q_D(const PlugInModel);

    ito::AddInBase* ai = (ito::AddInBase*)index.internalPointer();

    if (role == Qt::DisplayRole)
    {
        switch (index.column())
        {
        case 0: // name
        {
            QString ident = ai->getIdentifier();
            if (ident.size() > 0)
            {
                return ident;
            }
            return QString("ID: %1").arg(ai->getID());
        }
        case 7: // description
        {
            // return aib->getDescription();
        }
        default: {
            return QVariant();
        }
        }
    }
    else if (role == Qt::ToolTipRole)
    {
        if (index.column() == 0)
        {
            QString ident = ai->getIdentifier();

            if (ident.size() > 0)
            {
                return ident;
            }

            return QString("ID: %1").arg(ai->getID());
        }
        else
        {
            return QVariant();
        }
    }
    else if (role == Qt::TextAlignmentRole)
    {
        if (index.column() >= 0 && index.column() < d->m_alignment.size())
        {
            return d->m_alignment[index.column()];
        }
        else
        {
            return QVariant();
        }
    }
    else if (role == Qt::UserRole + 1)
    {
        // returns type (OR-combination)
        return ai->getBasePlugin()->getType();
    }
    else if (role == Qt::UserRole + 2)
    {
        // returns true if item is a category or sub-category, else false
        return false;
    }
    else if (role == Qt::UserRole + 3) // returns tItemType
    {
        return itemInstance;
    }
    else if (role == Qt::BackgroundRole)
    {
        ito::AddInBase* ais = (ito::AddInBase*)index.internalPointer();

        if (ais)
        {
            if (ais->createdByGUI() == 0 || ais->getRefCount() > 0)
            {
                return d->m_bgColorItemsWithPythonRef;
            }
        }
        else
        {
            return QVariant();
        }
    }

    return QVariant();
}

//-------------------------------------------------------------------------------------
/**
 *   @param index
 *   @param role
 *   @param filterNotWidget
 *   @return QVariant
 *
 *
 */
QVariant PlugInModel::getFilterOrWidgetNodeInfo(
    const QModelIndex& index, const int& role, bool filterNotWidget) const
{
    Q_D(const PlugInModel);
    QString* name;
    QString* description;

    if (filterNotWidget)
    {
        ito::AddInAlgo::FilterDef* filterDef =
            (ito::AddInAlgo::FilterDef*)(index.internalPointer());
        name = &(filterDef->m_name);
        description = &(filterDef->m_description);
    }
    else
    {
        ito::AddInAlgo::AlgoWidgetDef* widgetDef =
            (ito::AddInAlgo::AlgoWidgetDef*)(index.internalPointer());
        name = &(widgetDef->m_name);
        description = &(widgetDef->m_description);
    }

    if (role == Qt::DisplayRole)
    {
        switch (index.column())
        {
        case 0: // name
        {
            return *name;
        }
        case 1: // type
        {
            if (filterNotWidget)
            {
                return tr("Filter");
            }
            return tr("Widget");
        }
        case 7: // description
        {
            int firstLineBreak = description->indexOf('\n');
            QString shortDesc;
            if (firstLineBreak > 0)
            {
                shortDesc = description->left(firstLineBreak);
            }
            else
            {
                shortDesc = *description;
            }
            return shortDesc;
        }
        default: {
            return QVariant();
        }
        }
    }
    else if (role == Qt::DecorationRole)
    {
        if (index.column() == 0)
        {
            if (filterNotWidget)
            {
                return d->m_iconFilter;
            }
            else
            {
                return d->m_iconWidget;
            }
        }
        return QVariant();
    }
    else if (role == Qt::ToolTipRole)
    {
        switch (index.column())
        {
        case 0: // name
        case 7: // description
        {
            QString text = *name;
            if (description->size() > 0)
            {
                QString lb;
                lb.fill('-', text.length() * 1.5);

                QStringList desc = description->split('\n');
                text += "\n" + lb;

                foreach (const QString& s, desc)
                {
                    if (s.size() < 200)
                    {
                        text += "\n" + s;
                    }
                    else
                    {
                        QStringList words = s.split(" ");
                        int curLen = 0;
                        text += "\n";
                        foreach (const QString& w, words)
                        {
                            curLen += w.size();
                            if (curLen < 200)
                            {
                                text += w + " ";
                                curLen++;
                            }
                            else
                            {
                                curLen = w.size() + 1;
                                text += "\n" + w + " ";
                            }
                        }
                    }
                }
            }
            return text;
        }
        default: {
            return QVariant();
        }
        }
    }
    else if (role == Qt::TextAlignmentRole)
    {
        if (index.column() >= 0 && index.column() < d->m_alignment.size())
        {
            return d->m_alignment[index.column()];
        }
        else
        {
            return QVariant();
        }
    }
    else if (role == Qt::UserRole + 1) // returns type (OR-combination)
    {
        return ito::typeAlgo;
    }
    else if (role == Qt::UserRole + 2) // returns true if item is a category or sub-category, else
                                       // false
    {
        return false;
    }
    else if (role == Qt::UserRole + 3) // returns tItemType
    {
        if (filterNotWidget)
        {
            return itemFilter;
        }
        else
        {
            return itemWidget;
        }
    }

    return QVariant();
}

//-------------------------------------------------------------------------------------
/** return data elements for a given row
 *   @param [in] index   index for which the data elements should be delivered
 *   @param [in] role    the current role of the model
 *   @return data of the selected element, depending on the element's row and column (passed in
 * index.row and index.column)
 *
 *   This method is actually used to fill the tree view. It returns the data for the selected
 * element, depending as well on the column of the selected element, passed in index.column. The
 * method here is divded into two parts. The first one handels requests for root elements (plugins)
 * the second one is used for child elements (instances of plugins).
 */
QVariant PlugInModel::data(const QModelIndex& index, int role) const
{
    Q_D(const PlugInModel);

    tItemType itemType;
    size_t itemInternalData;

    if (!getModelIndexInfo(index, itemType, itemInternalData))
    {
        return QVariant();
    }

    switch (itemType)
    {
    case itemCatDataIO: {
        return getFixedNodeInfo(index, tr("DataIO"), itemType, role, d->m_iconDataIO);
    }
    case itemCatActuator: {
        return getFixedNodeInfo(index, tr("Actuator"), itemType, role, d->m_iconActuator);
    }
    case itemCatAlgo: {
        return getFixedNodeInfo(index, tr("Algorithm"), itemType, role, d->m_iconAlgo);
    }
    case itemSubCategoryDataIO_Grabber: {
        return getFixedNodeInfo(index, tr("Grabber"), itemType, role, d->m_iconGrabber);
    }
    case itemSubCategoryDataIO_ADDA: {
        return getFixedNodeInfo(index, tr("ADDA"), itemType, role, d->m_iconADDA);
    }
    case itemSubCategoryDataIO_RawIO: {
        return getFixedNodeInfo(index, tr("Raw IO"), itemType, role, d->m_iconRawIO);
    }
    case itemPlugin: {
        ito::AddInInterfaceBase* aiib = (ito::AddInInterfaceBase*)(itemInternalData);

        for (int i = 0; i < sizeof(d->m_treeFixNodes) / sizeof(d->m_treeFixNodes[0]); i++)
        {
            if (aiib->getType() == d->m_treeFixNodes[i])
            {
                return getPluginNodeInfo(index, role);
            }
        }
        break;
    }
    case itemInstance: {
        ito::AddInBase* aib = (ito::AddInBase*)(itemInternalData);
        ito::AddInInterfaceBase* aiib = aib->getBasePlugin();

        if (aiib->getType() & ito::typeActuator)
        {
            int count = d->m_pAIM->getActList()->count();

            for (int i = 0; i < count; i++)
            {
                if (d->m_pAIM->getActList()->at(i) == (QObject*)aiib)
                {
                    return getInstanceNodeInfo(index, role);
                }
            }
        }
        else if (aiib->getType() & ito::typeDataIO)
        {
            int count = d->m_pAIM->getDataIOList()->count();

            for (int i = 0; i < count; i++)
            {
                if (d->m_pAIM->getDataIOList()->at(i) == (QObject*)aiib)
                {
                    return getInstanceNodeInfo(index, role);
                }
            }
        }
        break;
    }
    case itemFilter: {
        return getFilterOrWidgetNodeInfo(index, role, true);
    }
    case itemWidget: {
        return getFilterOrWidgetNodeInfo(index, role, false);
    }
        // default:
        //{
        //    return QVariant();
        //}
    }

    return QVariant();
}

//-------------------------------------------------------------------------------------
/**
 *   @param ai
 *   @return QModelIndex
 *
 *
 */
QModelIndex PlugInModel::getIndexByAddIn(ito::AddInBase* ai) const
{
    if (ai == nullptr)
    {
        return QModelIndex();
    }

    QModelIndex baseIndex = getIndexByAddInInterface(ai->getBasePlugin());

    if (baseIndex.isValid() == false)
    {
        return QModelIndex();
    }

    int rows = rowCount(baseIndex);
    QModelIndex idx;

    for (int i = 0; i < rows; i++)
    {
        idx = index(i, 0, baseIndex);

        if (idx.isValid() && idx.internalPointer() == (void*)ai)
        {
            return index(i, 0, baseIndex);
        }
    }
    return QModelIndex();
}

//-------------------------------------------------------------------------------------
/**
 *   @param aib
 *   @return QModelIndex
 *
 *
 */
QModelIndex PlugInModel::getIndexByAddInInterface(AddInInterfaceBase* aib) const
{
    Q_D(const PlugInModel);

    if (aib)
    {
        const QList<QObject*>* list = nullptr;

        switch (aib->getType())
        {
        case ito::typeActuator: {
            list = d->m_pAIM->getActList();

            for (int i = 0; i < list->count(); i++)
            {
                if (qobject_cast<ito::AddInInterfaceBase*>(list->at(i)) == aib)
                {
                    return index(i, 0, d->m_treeFixIndizes[1]);
                }
            }
        }
        break;
        case ito::typeAlgo: {
            list = d->m_pAIM->getAlgList();

            for (int i = 0; i < list->count(); i++)
            {
                if (qobject_cast<ito::AddInInterfaceBase*>(list->at(i)) == aib)
                {
                    return index(i, 0, d->m_treeFixIndizes[2]);
                }
            }
        }
        break;
        case ito::typeDataIO | ito::typeGrabber:
        case ito::typeDataIO | ito::typeRawIO:
        case ito::typeDataIO | ito::typeADDA: {
            list = d->m_pAIM->getDataIOList();
            int countGrabber = 0;
            int countRawIO = 0;
            int countADDA = 0;
            ito::AddInInterfaceBase* aib2 = nullptr;

            for (int i = 0; i < list->count(); i++)
            {
                aib2 = qobject_cast<ito::AddInInterfaceBase*>(list->at(i));

                if (aib2 == aib)
                {
                    if (aib->getType() & ito::typeGrabber)
                    {
                        return index(countGrabber, 0, d->m_treeFixIndizes[3]);
                    }
                    else if (aib->getType() & ito::typeRawIO)
                    {
                        return index(countRawIO, 0, d->m_treeFixIndizes[5]);
                    }
                    else if (aib->getType() & ito::typeADDA)
                    {
                        return index(countADDA, 0, d->m_treeFixIndizes[4]);
                    }
                    else
                    {
                        return QModelIndex();
                    }
                }

                if (aib2->getType() & ito::typeGrabber)
                {
                    countGrabber++;
                }

                if (aib2->getType() & ito::typeRawIO)
                {
                    countRawIO++;
                }

                if (aib2->getType() & ito::typeADDA)
                {
                    countADDA++;
                }
            }
        }
        break;
        }
    }

    return QModelIndex();
}

//-------------------------------------------------------------------------------------
/**
 *   @param itemType
 *   @param internalData
 *   @return bool
 *
 *
 */
bool PlugInModel::getIsAlgoPlugIn(tItemType& itemType, size_t& internalData) const
{
    Q_D(const PlugInModel);

    if (itemType == PlugInModel::itemPlugin)
    {
        ito::AddInInterfaceBase* aiib = (ito::AddInInterfaceBase*)(internalData);
        return (aiib->getType() == d->m_treeFixNodes[2]);
    }

    return false;
}

//-------------------------------------------------------------------------------------
/**
 *   @param itemType
 *   @param internalData
 *   @return bool
 *
 *
 */
bool PlugInModel::getIsGrabberInstance(tItemType& itemType, size_t& internalData) const
{
    if (itemType == PlugInModel::itemInstance) // internalData can be casted to AddInBase
    {
        ito::AddInBase* aib = (ito::AddInBase*)(internalData);

        if (aib->inherits("ito::AddInGrabber"))
        {
            return true;
        }
    }
    return false;
}

//-------------------------------------------------------------------------------------
/**
 *   @param addInInterface
 *   @param beginOperation
 *   @return bool
 *
 *
 */
bool PlugInModel::insertInstance(ito::AddInInterfaceBase* addInInterface, bool beginOperation)
{
    if (beginOperation)
    {
        QModelIndex baseIndex = getIndexByAddInInterface(addInInterface);

        if (baseIndex.isValid())
        {
            int rows = rowCount(baseIndex);
            beginInsertRows(baseIndex, rows, rows); // append element
            return true;
        }

        return false;
    }
    else
    {
        endInsertRows();
        return true;
    }
}

//-------------------------------------------------------------------------------------
/**
 *   @param addInInterface
 *   @param addInInstance
 *   @param beginOperation
 *   @return bool
 *
 *
 */
bool PlugInModel::deleteInstance(ito::AddInBase* addInInstance, const bool beginOperation)
{
    if (beginOperation)
    {
        QModelIndex index = getIndexByAddIn(addInInstance);

        if (index.isValid())
        {
            QModelIndex parentIdx = parent(index);
            int i = index.row();
            beginRemoveRows(parentIdx, i, i);
            return true;
        }

        return false;
    }
    else
    {
        endRemoveRows();
        return true;
    }
}

//-------------------------------------------------------------------------------------
/**
 *   @param beginOperation
 *   @return bool
 *
 *
 */
bool PlugInModel::resetModel(bool beginOperation)
{
    if (beginOperation)
    {
        beginResetModel();
    }
    else
    {
        endResetModel();
    }

    return true;
}

//-------------------------------------------------------------------------------------
QString PlugInModel::getInitCommand(const QModelIndex& item) const
{
    Q_D(const PlugInModel);

    tItemType type;
    size_t internalData;
    getModelIndexInfo(item, type, internalData);
    QString command;

    if (type & itemPlugin) // check if item is a plugin
    {
        ito::AddInInterfaceBase* aiib = (ito::AddInInterfaceBase*)(internalData);
        int aiibType = aiib->getType();

        if (aiibType & d->m_treeFixNodes[0]) // dataIO
        {
            QVector<ito::Param>* mandParam(aiib->getInitParamsMand());
            QStringList listParam;

            foreach (const auto& item, *mandParam)
            {
                listParam.append(item.getName());
            }

            if (listParam.length() > 0)
            {
                listParam[0].prepend(", ");
            }

            command = QString("dataIO(\"%1\"%2)")
                          .arg(aiib->objectName(), listParam.join(", "))
                          .toLatin1()
                          .data();
        }
        else if (aiibType & d->m_treeFixNodes[1]) // actuator
        {
            QVector<ito::Param>* mandParam(aiib->getInitParamsMand());
            QStringList listParam;

            foreach (const auto& item, *mandParam)
            {
                listParam.append(item.getName());
            }

            if (listParam.length() > 0)
            {
                listParam[0].prepend(", ");
            }

            command = QString("actuator(\"%1\"%2)")
                          .arg(aiib->objectName(), listParam.join(", "))
                          .toLatin1()
                          .data();
        }
    }
    else if (type & itemFilter)
    {
        ito::AddInAlgo::FilterDef* fd = (AddInAlgo::FilterDef*)(internalData);

        QStringList listParam;
        AddInAlgo::t_filterParam paramFunc = fd->m_paramFunc;
        QVector<ito::Param> mandParams;
        QVector<ito::Param> optParams;
        QVector<ito::Param> outParams;
        paramFunc(&mandParams, &optParams, &outParams);

        foreach (const auto& item, mandParams)
        {
            listParam.append(item.getName());
        }

        command = QString("algorithms.%1(%2)").arg(fd->m_name, listParam.join(", "));
    }
    else if (type & itemWidget)
    {
        QStringList listParam;
        AddInAlgo::AlgoWidgetDef* aw = (AddInAlgo::AlgoWidgetDef*)(internalData);
        AddInAlgo::t_filterParam paramFunc = aw->m_paramFunc;
        QVector<ito::Param> mandParams;
        QVector<ito::Param> optParams;
        QVector<ito::Param> outParams;
        paramFunc(&mandParams, &optParams, &outParams);

        foreach (const auto& item, mandParams)
        {
            listParam.append(item.getName());
        }

        if (listParam.length() > 0)
        {
            listParam[0].prepend(", ");
        }
        command = QString("ui.createNewPluginWidget(\"%1%2\")")
                      .arg(aw->m_name, listParam.join(", "))
                      .toLatin1()
                      .data();
    }
    return command;
}

//-------------------------------------------------------------------------------------
QMimeData* PlugInModel::mimeData(const QModelIndexList& indexes) const
{
    QMimeData* mimeData = QAbstractItemModel::mimeData(indexes);
    QStringList texts;

    foreach (const QModelIndex& item, indexes)
    {
        if (item.column() == 0)
        {
            texts.append(getInitCommand(item));
        }
    }

    // text in mimeData must be UTF8 encoded, not Latin1 (since it could also be read by other
    // applications).
    mimeData->setData("text/plain", texts.join("\n").toUtf8());
    return mimeData;
}
} // end namespace ito
