/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2021, Institut fuer Technische Optik (ITO),
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

#include "dialogVariableDetailDataObject.h"
#include "dataobj.h"
#include <qsharedpointer.h>
#include <qstandarditemmodel.h>

#include <qclipboard.h>
#include <qmap.h>
#include <qspinbox.h>
#include <qboxlayout.h>
#include <qmainwindow.h>
#include <qtoolbar.h>

namespace ito {

//----------------------------------------------------------------------------------------------
DialogVariableDetailDataObject::DialogVariableDetailDataObject(
    const QString& name,
    const QString& type,
    const char* dtype,
    QSharedPointer<ito::DataObject> dObj,
    QWidget* parent) :
    QDialog(parent),
    m_pAxesRanges(nullptr), m_dObj(dObj)
{
    ui.setupUi(this);

    ui.txtName->setText(name);
    ui.txtType->setText(type);
    ui.txtDType->setText(dtype);

    QMainWindow *tableMain = new QMainWindow(this);
    tableMain->setWindowFlag(Qt::Widget, true);
    QToolBar *tb = tableMain->addToolBar("myToolbar");

    QVBoxLayout *tableLayout = qobject_cast<QVBoxLayout*>(ui.tabTable->layout());
    tableLayout->insertWidget(0, tableMain);
    tableLayout->removeWidget(ui.dataTable);
    tableMain->setCentralWidget(ui.dataTable);
    tb->addActions(ui.dataTable->actions());

    ui.dataTable->setAlignment(Qt::AlignVCenter | Qt::AlignRight);
    ui.dataTable->setReadOnly(true);

    connect(
        ui.dataTable,
        &DataObjectTable::selectionInformationChanged,
        ui.lblSelectionInformation,
        &QLabel::setText);

    ui.metaWidget->setData(m_dObj);
    ui.metaWidget->setReadOnly(true);

    // add combobox for displayed axis and slicing of dataObject axis
    int dims = m_dObj->getDims();

    QString dObjSize;

    if (dims > 0)
    {
        dObjSize = "[";

        for (int dim = 0; dim < dims; dim++)
        {
            if (dim < dims - 1)
            {
                dObjSize.append(QString("%1 x ").arg(m_dObj->getSize(dim)));
            }
            else
            {
                dObjSize.append(QString("%1").arg(m_dObj->getSize(dim)));
            }
        }

        dObjSize.append("]");
    }
    else
    {
        dObjSize = tr("<empty>");
    }

    ui.txtDSize->setText(dObjSize);

    if (dims < 3)
    {
        ui.dataTable->setData(QSharedPointer<ito::DataObject>(m_dObj));
        ui.frameSlicing->setVisible(false);
    }
    else
    {
        int col = dims - 1;
        int row = dims - 2;
        bool valid = true;

        m_pAxesRanges = new ito::Range[dims];

        QStringList items;
        QString itemDescription;
        std::string axisDescription;
        for (int idx = 0; idx < dims; idx++)
        {
            axisDescription = m_dObj->getAxisDescription(idx, valid);

            if (axisDescription.empty())
            {
                itemDescription = tr("Axis %1").arg(idx);
            }
            else
            {
                itemDescription =
                    tr("%1 (Axis %2)").arg(QString::fromUtf8(axisDescription.data())).arg(idx);
            }

            items += itemDescription;
        }

        ui.comboBoxDisplayedCol->blockSignals(true);
        ui.comboBoxDisplayedRow->blockSignals(true);

        ui.comboBoxDisplayedRow->addItems(items);
        ui.comboBoxDisplayedRow->setCurrentIndex(row);

        ui.comboBoxDisplayedCol->addItems(items);
        ui.comboBoxDisplayedCol->setCurrentIndex(col);


        QStandardItemModel* model =
            qobject_cast<QStandardItemModel*>(ui.comboBoxDisplayedRow->model());
        QStandardItem* item =
            model->item(ui.comboBoxDisplayedRow->count() - 1); // last axis not allow for rows
        item->setFlags(item->flags() & ~Qt::ItemIsEnabled);

        model = qobject_cast<QStandardItemModel*>(ui.comboBoxDisplayedCol->model());
        item = model->item(0); // first axis not allow for cols
        item->setFlags(item->flags() & ~Qt::ItemIsEnabled);

        ui.comboBoxDisplayedCol->blockSignals(false);
        ui.comboBoxDisplayedRow->blockSignals(false);

        addSlicingWidgets();
        changeDisplayedAxes(-1);
    }
}

//----------------------------------------------------------------------------------------------
DialogVariableDetailDataObject::~DialogVariableDetailDataObject()
{
    DELETE_AND_SET_NULL_ARRAY(m_pAxesRanges);
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::on_btnCopyClipboard_clicked()
{
    QClipboard* clipboard = QApplication::clipboard();
    clipboard->setText(ui.txtName->text(), QClipboard::Clipboard);
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::on_comboBoxDisplayedRow_currentIndexChanged(int idx)
{
    deleteSlicingWidgets();
    addSlicingWidgets();
    changeDisplayedAxes(0); // 0: row, 1: col, -1: default
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::on_comboBoxDisplayedCol_currentIndexChanged(int idx)
{
    deleteSlicingWidgets();
    addSlicingWidgets();
    changeDisplayedAxes(1); // 0: row, 1: col, -1: default
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::spinBoxValueChanged(int idx)
{
    changeDisplayedAxes(-1); // 0: row, 1: col, -1: default
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::deleteSlicingWidgets()
{
    while (ui.horizontalLayoutSlicing->count() != 0)
    {
        QLayoutItem* item = ui.horizontalLayoutSlicing->takeAt(0);
        delete item->widget();
        delete item;
    }
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::addSlicingWidgets()
{
    int dims = m_dObj->getDims();
    int col = ui.comboBoxDisplayedCol->currentIndex();

    int row = ui.comboBoxDisplayedRow->currentIndex();

    ui.horizontalLayoutSlicing->addWidget(new QLabel("[", this));

    m_spinBoxToIdxMap.clear();

    for (int idx = 0; idx < dims; idx++)
    {
        if ((idx == row) || (idx == col)) // row
        {
            ui.horizontalLayoutSlicing->addWidget(new QLabel(":", this));
        }
        else
        {
            QSpinBox* spinBox = new QSpinBox(this);
            spinBox->setMaximum(m_dObj->getSize(idx) - 1);
            connect(spinBox, SIGNAL(valueChanged(int)), this, SLOT(spinBoxValueChanged(int)));
            ui.horizontalLayoutSlicing->addWidget(spinBox);
            m_spinBoxToIdxMap.insert(idx, spinBox);
        }

        if (idx < dims - 1)
        {
            ui.horizontalLayoutSlicing->addWidget(new QLabel(", ", this));
        }
    }

    ui.horizontalLayoutSlicing->addWidget(new QLabel("]", this));
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::changeDisplayedAxes(int isColNotRow = -1)
{
    ui.comboBoxDisplayedCol->blockSignals(true);
    ui.comboBoxDisplayedRow->blockSignals(true);

    int dims = m_dObj->getDims();
    int col = ui.comboBoxDisplayedCol->currentIndex();
    int row = ui.comboBoxDisplayedRow->currentIndex();

    if (row == col)
    {
        if (isColNotRow) // 0: row, 1: col, -1: default
        {
            row = col - 1;
            ui.comboBoxDisplayedRow->setCurrentIndex(row);
        }
        else
        {
            col = row + 1;
            ui.comboBoxDisplayedCol->setCurrentIndex(col);
        }
        deleteSlicingWidgets();
        addSlicingWidgets();
    }

    for (int idx = 0; idx < dims; idx++)
    {
        if (idx == row)
        {
            m_pAxesRanges[idx] = ito::Range(ito::Range::all());
        }
        else if (idx == col)
        {
            m_pAxesRanges[idx] = ito::Range(ito::Range::all());
        }
        else
        {
            int index;
            if (m_spinBoxToIdxMap.empty())
            {
                index = 0;
            }
            else
            {
                index = m_spinBoxToIdxMap.value(idx)->value();
            }
            m_pAxesRanges[idx] = ito::Range(index, index + 1);
        }
    }

    ui.dataTable->setData(
        QSharedPointer<ito::DataObject>(new ito::DataObject(m_dObj->at(m_pAxesRanges).squeeze())));

    ui.comboBoxDisplayedCol->blockSignals(false);
    ui.comboBoxDisplayedRow->blockSignals(false);
}

} // end namespace ito
