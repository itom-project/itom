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

#pragma once

#if !defined(Q_MOC_RUN) || defined(ADDINMGR_DLL) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

#include "addInMgrDefines.h"
#include "../common/addInInterface.h"

#include <qabstractitemmodel.h>
#include <qscopedpointer.h>
#include <qcolor.h>
#include <qicon.h>

namespace ito
{
    class AddInBase;
    class AddInManager;

    /**
    * PluginLoadStatusFlag enumeration
    * This enum holds the four possible return states for loaded DLLs Ok, Warning, Error and Ignored
    */
    enum tPluginLoadStatusFlag
    {
        plsfOk       = 0x001,  /*!< ok */
        plsfWarning  = 0x002,  /*!< warning */
        plsfError    = 0x004,  /*!< error */
        plsfIgnored  = 0x008,  /*!< ignored */
        plsfRelDbg   = 0x100,   /*!< is Dbg version */
        plsfIncompatible = 0x200, /*!< incompatible to this itom version */
    };
    Q_DECLARE_FLAGS(PluginLoadStatusFlags, tPluginLoadStatusFlag)


    /*!
        \class PluginLoadStatus
        \brief This struct provides a structure for saving the load status of any plugins or designerWidgets
    */
    struct PluginLoadStatus
    {
        PluginLoadStatus() : filename("") {}
        QString filename;
        QList< QPair<ito::PluginLoadStatusFlags, QString> > messages;
    };

    class PlugInModelPrivate; //forward declaration

    /** @class PlugInModel
    *   @brief class for visualizing the available (loaded) plugins
    *
    *   The PlugInModel supplies a widget showing the available plugins (libraries) with their name, filename, version and so on.
    *   In addition below each plugin its running instances are shown and if a plugin offers a configuration dialog it can be opened
    *   using a right click on the instance and selecting "open configuration dialog" in the context menu. The tree view is
    *   automatically updated when a new instance is created or an existing one had been deleted.
    */
    class ADDINMGR_EXPORT PlugInModel : public QAbstractItemModel
    {
        Q_OBJECT

        public:
            PlugInModel(ito::AddInManager *addInManager, QObject *parent = NULL);
            ~PlugInModel();

            enum tItemType {
                itemUnknown = 0x0000,
                itemCatDataIO = 0x0001,
                itemCatActuator = 0x0002,
                itemCatAlgo = 0x0004,
                itemSubCategoryDataIO_Grabber = 0x0008,
                itemSubCategoryDataIO_ADDA = 0x0010,
                itemSubCategoryDataIO_RawIO = 0x0020,
                itemPlugin = 0x0040,
                itemInstance = 0x0080,
                itemFilter = 0x0100,
                itemWidget = 0x0200,
                itemCatMainAll = itemCatDataIO | itemCatActuator | itemCatAlgo,
                itemCatSubAll = itemSubCategoryDataIO_Grabber | itemSubCategoryDataIO_ADDA | itemSubCategoryDataIO_RawIO,
                itemCatAll = itemCatMainAll | itemCatSubAll
            };

            QVariant data(const QModelIndex &index, int role) const;
            Qt::ItemFlags flags(const QModelIndex &index) const;
            QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;
            QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
            QModelIndex parent(const QModelIndex &index) const;
            int rowCount(const QModelIndex &parent = QModelIndex()) const;
            int columnCount(const QModelIndex &parent = QModelIndex()) const;
            int update(void) { emit(beginResetModel()); emit(endResetModel()); return 0; };

            bool insertInstance(ito::AddInInterfaceBase* addInInterface, bool beginOperation);
            bool deleteInstance(ito::AddInBase *addInInstance, const bool beginOperation);
            bool resetModel(bool beginOperation);

            QModelIndex getIndexByAddIn(ito::AddInBase *ai) const;
            QModelIndex getIndexByAddInInterface(ito::AddInInterfaceBase *aib) const;
            bool getModelIndexInfo(const QModelIndex &index, tItemType &type, size_t &internalData) const;

            bool getIsAlgoPlugIn(tItemType &itemType, size_t &internalData) const;
            bool getIsGrabberInstance(tItemType &itemType, size_t &internalData) const;

            QModelIndex getTypeNode(const int type) const;

            //!< returns the background color for instances, that have been created by Python or which have at least one current reference by Python code
            QColor backgroundColorInstancesWithPythonRef() const;
            void setBackgroundColorInstancesWithPythonRef(const QColor &bgColor);

        protected:
            QVariant getFixedNodeInfo(const QModelIndex &index, const QVariant &name, const tItemType &itemType, const int &role, const QIcon icon) const;
            QVariant getPluginNodeInfo(const QModelIndex &index, const int &role) const;
            QVariant getInstanceNodeInfo(const QModelIndex &index, const int &role) const;
            QVariant getFilterOrWidgetNodeInfo(const QModelIndex &index, const int &role, bool filterNotWidget) const;
            QMimeData* mimeData(const QModelIndexList &indexes) const;

        private:
            QScopedPointer<PlugInModelPrivate> d_ptr; //!> self-managed pointer to the private class container (deletes itself if d_ptr is destroyed)
            QString getInitCommand(const QModelIndex & item) const;
            Q_DECLARE_PRIVATE(PlugInModel);
    };

}; // namespace ito

#endif // #if !defined(Q_MOC_RUN) || defined(ADDINMGR_DLL)
