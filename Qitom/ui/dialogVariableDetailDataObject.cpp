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

#include <qclipboard.h>
#include <qmap.h>
#include <qspinbox.h>

namespace ito {

//----------------------------------------------------------------------------------------------
DialogVariableDetailDataObject::DialogVariableDetailDataObject(
    const QString& name,
    const QString& type,
    const char* dtype,
    QSharedPointer<ito::DataObject> dObj,
    QWidget* parent) :
    QDialog(parent),
    m_AxesRanges(nullptr)
{
    ui.setupUi(this);

    m_dObj = dObj;

    ui.txtName->setText(name);
    ui.txtType->setText(type);
    ui.txtDType->setText(dtype);

    ui.dataTable->setReadOnly(true);

    ui.metaWidget->setData(m_dObj);
    ui.metaWidget->setReadOnly(true);

    // add combobox for displayed axis and slicing of dataObject axis
    int dims = m_dObj->getDims();
    int col = dims - 2;
    int row = dims - 1;
    bool valid = true;

    m_AxesRanges = new ito::Range[dims];

    QStringList items;
    QString itemDescription;
    std::string axisDescription;
    for (int idx = 0; idx < dims; idx++)
    {
        itemDescription = "";
        axisDescription = m_dObj->getAxisDescription(idx, valid);
        if (!axisDescription.empty())
        {
            itemDescription += QString::fromUtf8(axisDescription.data());
            itemDescription += " (";
        }

        itemDescription += QString::number(idx);

        if (!axisDescription.empty())
        {
            itemDescription += ")";
        }

        items += itemDescription;
    }

    ui.comboBoxDisplayedRow->addItems(items);
    ui.comboBoxDisplayedRow->setCurrentIndex(col);
    ui.comboBoxDisplayedCol->addItems(items);
    ui.comboBoxDisplayedCol->setCurrentIndex(row);

    addSlicingWidgets();
    changeDisplayedAxes();

    connect(
        ui.comboBoxDisplayedRow,
        SIGNAL(currentIndexChanged(int)),
        this,
        SLOT(comboBoxCurrentIndexChanged(int)));
    connect(
        ui.comboBoxDisplayedCol,
        SIGNAL(currentIndexChanged(int)),
        this,
        SLOT(comboBoxCurrentIndexChanged(int)));
}

//----------------------------------------------------------------------------------------------
DialogVariableDetailDataObject::~DialogVariableDetailDataObject()
{
    DELETE_AND_SET_NULL_ARRAY(m_AxesRanges);
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::on_btnCopyClipboard_clicked()
{
    QClipboard* clipboard = QApplication::clipboard();
    clipboard->setText(ui.txtName->text(), QClipboard::Clipboard);
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::comboBoxCurrentIndexChanged(int idx)
{
    deleteSlicingWidgets();
    addSlicingWidgets();
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::spinBoxValueChanged(int idx)
{
    changeDisplayedAxes();
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

    ui.horizontalLayoutSlicing->addWidget(new QLabel("["));

    m_SpinBoxToIdxMap.clear();

    for (int idx = 0; idx < dims; idx++)
    {
        if ((idx == row) || (idx == col)) // row
        {
            ui.horizontalLayoutSlicing->addWidget(new QLabel(":"));
        }
        else if (m_dObj->getSize(idx) == 1)
        {
            ui.horizontalLayoutSlicing->addWidget(new QLabel("0"));
        }
        else
        {
            QSpinBox* spinBox = new QSpinBox();
            spinBox->setMaximum(m_dObj->getSize(idx) - 1);
            connect(spinBox, SIGNAL(valueChanged(int)), this, SLOT(spinBoxValueChanged(int)));
            ui.horizontalLayoutSlicing->addWidget(spinBox);
            m_SpinBoxToIdxMap.insert(idx, spinBox);
        }

        if (idx < dims - 1)
        {
            ui.horizontalLayoutSlicing->addWidget(new QLabel(", "));
        }
    }

    ui.horizontalLayoutSlicing->addWidget(new QLabel("]"));
    changeDisplayedAxes();
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::changeDisplayedAxes()
{
    int dims = m_dObj->getDims();
    int col = ui.comboBoxDisplayedCol->currentIndex();
    int row = ui.comboBoxDisplayedRow->currentIndex();

    for (int idx = 0; idx < dims; idx++)
    {
        if (idx == row)
        {
            m_AxesRanges[idx] = ito::Range(ito::Range::all());
        }
        else if (idx == col)
        {
            m_AxesRanges[idx] = ito::Range(ito::Range::all());
        }
        else
        {
            int index;
            if (m_SpinBoxToIdxMap.empty())
            {
                index = 0;
            }
            else
            {
                int index = m_SpinBoxToIdxMap.value(idx)->value();
            }
            m_AxesRanges[idx] = ito::Range(index, index + 1);
        }
    }

    ui.dataTable->setData(
        QSharedPointer<ito::DataObject>(new ito::DataObject(m_dObj->at(m_AxesRanges).squeeze())));
}

} // end namespace ito
