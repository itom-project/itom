/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

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

#ifndef DIALOGPLUGINPICKER_H
#define DIALOGPLUGINPICKER_H

#include "../../common/addInInterface.h"
#include "../../common/sharedStructures.h"

#include <QtGui>
#include <qdialog.h>
#include <qvector.h>

#include "../organizer/addInManager.h"
#include <qsortfilterproxymodel.h>

#include "ui_dialogPluginPicker.h"

namespace ito {

class PickerSortFilterProxyModel : public QSortFilterProxyModel
{
public:
    PickerSortFilterProxyModel(QObject *parent = 0) : QSortFilterProxyModel(parent), m_minimumMask(0x0), m_pluginName(QString::Null()), m_showPluginsWithoutInstance(false) {};
    ~PickerSortFilterProxyModel() {};

    inline void setPluginMinimumMask( const int minimumMask ) 
    { 
        m_minimumMask = minimumMask; 
        invalidateFilter(); 
    };

    inline void setPluginName( QString &name )
    {
        m_pluginName = name;
        invalidateFilter();
    }

    inline void showPluginsWithoutInstance(bool show) 
    { 
        m_showPluginsWithoutInstance = show; 
        invalidateFilter(); 
    };

protected:
    bool filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const
    {
        QModelIndex idx = sourceModel()->index(sourceRow, 0, sourceParent);
        int type = sourceModel()->data(idx, Qt::UserRole + 1).toInt();


        if(!m_showPluginsWithoutInstance)
        {
            int itemType = sourceModel()->data(idx, Qt::UserRole + 3).toInt();
            if(itemType == ito::PlugInModel::itemPlugin && sourceModel()->hasChildren(idx) == false) 
            {
                return false;
            }
        }

        if(!m_pluginName.isNull())
        {
            if( QString::compare( sourceModel()->data(idx, Qt::DisplayRole).toString(), m_pluginName, Qt::CaseInsensitive ) != 0)
            {
                return false;
            }
        }

        if(type == ito::typeAlgo) 
        {
            return false; //never allow algorithms or widgets
        }

        return (type & m_minimumMask) == m_minimumMask;
    }

private:
    int m_minimumMask;
    QString m_pluginName; //default QString::Null()
    bool m_showPluginsWithoutInstance;
};

} //end namespace ito

namespace ito {


class DialogPluginPicker : public QDialog 
{
    Q_OBJECT

public:
    DialogPluginPicker(bool allowNewInstances, ito::AddInBase *currentItem, int minimumPluginTypeMask = 0x0, QString pluginName = QString::Null(), QWidget *parent = NULL );
    ~DialogPluginPicker() {};

    ito::AddInBase* getSelectedInstance();

protected:

private:

    Ui::DialogPluginPicker ui;
    PickerSortFilterProxyModel *m_pFilterModel;

private slots:
    void itemClicked(const QModelIndex &index);
    void itemDblClicked(const QModelIndex &index);
    void showPluginsWithoutInstance(bool checked); 
    void createNewInstance(bool checked);

};

} //end namespace ito

#endif
