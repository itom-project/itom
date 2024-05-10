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

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#pragma once

#include "ui_paramInputDialog.h"

#include "common/paramMeta.h"
#include "common/param.h"

#include <qdialog.h>
#include <qstyleditemdelegate.h>
#include <qmetatype.h>
#include <qlist.h>
#include <qpair.h>
#include <qregularexpression.h>

Q_DECLARE_METATYPE(ito::complex128)

//-------------------------------------------------------------------------------------
class QListWidgetItem;

namespace ito {


//! declaration of delegate class
class LineEditDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    explicit LineEditDelegate(const ito::ParamMeta *meta, int paramType, QObject *parent = 0);

    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;

    void setEditorData(QWidget *editor, const QModelIndex &index) const;
    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
    void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const;

private:
    QSharedPointer<ito::ParamMeta> m_meta;
    int m_paramType;
};

//-------------------------------------------------------------------------------------
class ParamInputDialog: public QDialog
{
    Q_OBJECT

public:
    explicit ParamInputDialog(const Param& param, QWidget *parent = nullptr);
    ~ParamInputDialog();

    Param getItems(RetVal &retValue) const;

private slots:
    void on_newListItemButton_clicked();
    void on_deleteListItemButton_clicked();
    void on_moveListItemUpButton_clicked();
    void on_moveListItemDownButton_clicked();
    void on_listWidget_currentRowChanged();
    void on_listWidget_itemDoubleClicked(QListWidgetItem *item);
    void on_buttonBox_clicked(QAbstractButton* btn);

protected:
    virtual void setItemData(int role, const QVariant &v);
    virtual QVariant getItemData(int role) const;
    QList<QPair<QString, QVariant>> parseListItems(const ito::Param &param) const;
    void updateButtonState();

private:
    void updateEditor();
    Ui::paramInputDialog ui;
    bool m_updating;
    QString m_newItemText;
    size_t m_minSize;
    size_t m_maxSize;
    size_t m_stepSize;
    Param m_param;
    QRegularExpression m_RegExp;
    LineEditDelegate *m_lineEditDel;
};

} //end namespace ito
