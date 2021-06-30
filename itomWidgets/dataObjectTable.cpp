/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2021, Institut fuer Technische Optik (ITO),
   Universitaet Stuttgart, Germany

   This file is part of itom.

   itom is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   itom is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with itom. If not, see <http://www.gnu.org/licenses/>.

   In addition, as a special exception, the Institut fuer Technische
   Optik (ITO) gives you certain additional rights.
   These rights are described in the ITO LGPL Exception version 1.0,
   which can be found in the file LGPL_EXCEPTION.txt in this package.
*********************************************************************** */

#include "dataObjectTable.h"
#include <qapplication.h>
#include <qclipboard.h>
#include <qdebug.h>
#include <qevent.h>
#include <qheaderview.h>
#include <qinputdialog.h>
#include <qmenu.h>
#include <qscrollbar.h>

#include "dataObjectDelegate.h"
#include "dataObjectModel.h"

#include "common/typeDefs.h"


//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
DataObjectTable::DataObjectTable(QWidget* parent /*= 0*/) : QTableView(parent)
{
    m_pModel = new DataObjectModel();
    m_pDelegate = new DataObjectDelegate(this);

    setModel(m_pModel);
    setItemDelegate(m_pDelegate);

    setEditTriggers(QAbstractItemView::AnyKeyPressed | QAbstractItemView::DoubleClicked);

    connect(this, SIGNAL(activated(QModelIndex)), this, SLOT(_activated(QModelIndex)));
    connect(this, SIGNAL(clicked(QModelIndex)), this, SLOT(_clicked(QModelIndex)));
    connect(this, SIGNAL(doubleClicked(QModelIndex)), this, SLOT(_doubleClicked(QModelIndex)));
    connect(this, SIGNAL(entered(QModelIndex)), this, SLOT(_entered(QModelIndex)));
    connect(this, SIGNAL(pressed(QModelIndex)), this, SLOT(_pressed(QModelIndex)));

    horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    setContextMenuPolicy(Qt::DefaultContextMenu);

    setSelectionMode(QAbstractItemView::ExtendedSelection);

    emit selectionInformationChanged("");
}

//-------------------------------------------------------------------------------------
DataObjectTable::~DataObjectTable()
{
    m_pDelegate->deleteLater();
    m_pModel->deleteLater();
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setData(QSharedPointer<ito::DataObject> dataObj)
{
    emit selectionInformationChanged("");

    m_pModel->setDataObject(dataObj);

    horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
}

//-------------------------------------------------------------------------------------
QSharedPointer<ito::DataObject> DataObjectTable::getData() const
{
    return m_pModel->getDataObject();
}

//-------------------------------------------------------------------------------------
bool DataObjectTable::getReadOnly() const
{
    return m_pModel->getReadOnly();
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setReadOnly(bool value)
{
    m_pModel->setReadOnly(value);
}

//-------------------------------------------------------------------------------------
double DataObjectTable::getMin() const
{
    return m_pDelegate->m_min;
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setMin(double value)
{
    m_pDelegate->m_min = value;
}

//-------------------------------------------------------------------------------------
double DataObjectTable::getMax() const
{
    return m_pDelegate->m_max;
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setMax(double value)
{
    m_pDelegate->m_max = value;
}

//-------------------------------------------------------------------------------------
int DataObjectTable::getDecimals() const
{
    return m_pModel->getDecimals();
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setDecimals(int value)
{
    m_pModel->setDecimals(value);
}

//-------------------------------------------------------------------------------------
int DataObjectTable::getEditorDecimals() const
{
    return m_pDelegate->m_editorDecimals;
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setEditorDecimals(int value)
{
    m_pDelegate->m_editorDecimals = value;
}

//-------------------------------------------------------------------------------------
QHeaderView::ResizeMode DataObjectTable::getHorizontalResizeMode() const
{
    return horizontalHeader()->sectionResizeMode(0);
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setHorizontalResizeMode(QHeaderView::ResizeMode mode)
{
    return horizontalHeader()->setSectionResizeMode(mode);
}

//-------------------------------------------------------------------------------------
QHeaderView::ResizeMode DataObjectTable::getVerticalResizeMode() const
{
    return verticalHeader()->sectionResizeMode(0);
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setVerticalResizeMode(QHeaderView::ResizeMode mode)
{
    return verticalHeader()->setSectionResizeMode(mode);
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setHorizontalLabels(QStringList value)
{
    m_pModel->setHeaderLabels(Qt::Horizontal, value);
    horizontalHeader()->repaint();
}

//-------------------------------------------------------------------------------------
QStringList DataObjectTable::getHorizontalLabels() const
{
    return m_pModel->getHorizontalHeaderLabels();
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setVerticalLabels(QStringList value)
{
    m_pModel->setHeaderLabels(Qt::Vertical, value);
    verticalHeader()->repaint();
}

//-------------------------------------------------------------------------------------
QStringList DataObjectTable::getVerticalLabels() const
{
    return m_pModel->getVerticalHeaderLabels();
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setSuffixes(QStringList value)
{
    m_pDelegate->m_suffixes = value;
    m_pModel->setSuffixes(value);
}

//-------------------------------------------------------------------------------------
QStringList DataObjectTable::getSuffixes() const
{
    return m_pModel->getSuffixes();
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setDefaultCols(int value)
{
    m_pModel->setDefaultGrid(m_pModel->getDefaultRows(), value);
}

//-------------------------------------------------------------------------------------
int DataObjectTable::getDefaultCols() const
{
    return m_pModel->getDefaultCols();
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setDefaultRows(int value)
{
    m_pModel->setDefaultGrid(value, m_pModel->getDefaultCols());
}

//-------------------------------------------------------------------------------------
int DataObjectTable::getDefaultRows() const
{
    return m_pModel->getDefaultRows();
}

//-------------------------------------------------------------------------------------
Qt::Alignment DataObjectTable::getAlignment() const
{
    return m_pModel->getAlignment();
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setAlignment(Qt::Alignment alignment)
{
    m_pModel->setAlignment(alignment);
}

//-------------------------------------------------------------------------------------
QSize DataObjectTable::sizeHint() const
{
    QHeaderView* hHeader = horizontalHeader();
    QHeaderView* vHeader = verticalHeader();

    /*QScrollBar *hScrollBar = horizontalScrollBar();
    QScrollBar *vScrollBar = verticalScrollBar();*/

    int h = 25;
    int w = 15;
    h += m_pModel->getDefaultRows() * vHeader->defaultSectionSize();
    w += m_pModel->getDefaultCols() * hHeader->defaultSectionSize();

    if (vHeader->isVisible())
    {
        w += vHeader->sizeHint().width();
    }
    if (hHeader->isVisible())
    {
        h += hHeader->sizeHint().height();
    }

    // if (vScrollBar->isVisible())
    //{
    //    w += vScrollBar->sizeHint().width();
    //}
    // if (hScrollBar->isVisible())
    //{
    //    h += hScrollBar->sizeHint().height();
    //}

    return QSize(w, h);
}

bool sortByRowAndColumn(const QModelIndex& idx1, const QModelIndex& idx2)
{
    if (idx1.row() == idx2.row())
    {
        return idx1.column() < idx2.column();
    }

    return idx1.row() < idx2.row();
}

//-------------------------------------------------------------------------------------
void DataObjectTable::keyPressEvent(QKeyEvent* e)
{
    if (e->matches(QKeySequence::Copy))
    {
        copySelectionToClipboard();
        e->accept();
    }
    else
    {
        QTableView::keyPressEvent(e);
    }
}

//-------------------------------------------------------------------------------------
void DataObjectTable::contextMenuEvent(QContextMenuEvent* event)
{
    QMenu contextMenu(this);
    contextMenu.addAction(
        QIcon(":/itomDesignerPlugins/general/icons/clipboard.png"),
        "copy selection",
        this,
        SLOT(copySelectionToClipboard()));
    contextMenu.addAction(
        QIcon(":/itomDesignerPlugins/general/icons/clipboard.png"),
        "copy all",
        this,
        SLOT(copyAllToClipboard()));
    contextMenu.addSeparator();
    contextMenu.addAction(
        QIcon(":/itomDesignerPlugins/general/icons/decimals.png"),
        "decimals...",
        this,
        SLOT(setDecimalsGUI()));
    contextMenu.exec(event->globalPos());

    event->accept();
}

//-------------------------------------------------------------------------------------
void DataObjectTable::copySelectionToClipboard()
{
    QStringList items;
    int currentRow = 0;
    QModelIndexList selected = selectedIndexes();
    qSort(selected.begin(), selected.end(), sortByRowAndColumn);

    if (selected.size() > 0)
    {
        int firstRow = selected[0].row();
        int lastRow = selected[selected.size() - 1].row();
        int firstCol = INT_MAX;
        int lastCol = 0;
        foreach (const QModelIndex& idx, selected)
        {
            firstCol = std::min(firstCol, idx.column());
            lastCol = std::max(lastCol, idx.column());
        }
        int cols = 1 + lastCol - firstCol;
        int rows = 1 + lastRow - firstRow;

        items.reserve(rows * cols);
        int currentIdx = 0;
        int lastIdx = 0;

        foreach (const QModelIndex& idx, selected)
        {
            currentIdx = cols * (idx.row() - firstRow) + (idx.column() - firstCol);
            while (lastIdx < currentIdx)
            {
                items.append("");
                lastIdx++;
            }

            items.append(m_pModel->data(idx, DataObjectModel::displayRoleWithoutSuffix).toString());
            lastIdx++;
        }

        while (items.size() < rows)
        {
            items.append("");
        }

        QStringList final;
        for (int i = 0; i < rows; ++i)
        {
            final.append(QStringList(items.mid(i * cols, cols)).join(";"));
        }

        QApplication::clipboard()->setText(final.join("\n"));
    }
}


//-------------------------------------------------------------------------------------
void DataObjectTable::copyAllToClipboard()
{
    int rows = m_pModel->rowCount();
    int cols = m_pModel->columnCount();

    QStringList colHeaders;
    colHeaders << ""; // for the top left corner
    for (int i = 0; i < cols; ++i)
    {
        colHeaders << QString("\"%1\"").arg(
            m_pModel->headerData(i, Qt::Horizontal, Qt::DisplayRole).toString());
    }

    QStringList rowHeaders;
    for (int i = 0; i < rows; ++i)
    {
        rowHeaders << QString("\"%1\"").arg(
            m_pModel->headerData(i, Qt::Vertical, Qt::DisplayRole).toString());
    }

    QStringList final;
    final << colHeaders.join(";");

    for (int r = 0; r < rows; ++r)
    {
        QStringList rowData;
        rowData << rowHeaders[r];

        for (int c = 0; c < cols; ++c)
        {
            rowData << m_pModel
                           ->data(
                               m_pModel->index(r, c),
                               DataObjectModel::preciseDisplayRoleWithoutSuffix)
                           .toString();
        }
        final << rowData.join(";");
    }

    QApplication::clipboard()->setText(final.join("\n"));
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setDecimalsGUI()
{
    bool ok;
    int decimals = getDecimals();
    int newDecimals = QInputDialog::getInt(
        this, tr("number of decimals"), tr("set number of decimals"), decimals, 0, 15, 1, &ok);

    if (ok)
    {
        int editorDecimals = getEditorDecimals();
        setDecimals(newDecimals);
        setEditorDecimals(std::max(0, editorDecimals + (newDecimals - decimals)));
    }
}


//-------------------------------------------------------------------------------------
template <typename _Tp>
void gatherSelectionInformation(
    const ito::DataObject* dObj, const QModelIndexList& indexes, QStringList& infos)
{
    bool hasStd = false;
    _Tp minimum = std::numeric_limits<_Tp>::max();
    _Tp maximum = std::numeric_limits<_Tp>::min();

    QString v = QString::fromLocal8Bit(dObj->getValueUnit().data());

    if (v != "")
    {
        v.prepend(" ");
    }

    if (std::is_floating_point<_Tp>())
    {
        maximum = -minimum;
    }

    double sum = 0;
    _Tp value;
    QVector<_Tp> values;
    values.reserve(indexes.size());

    foreach (const QModelIndex& idx, indexes)
    {
        value = dObj->at<_Tp>(idx.row(), idx.column());
        values.push_back(value);
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        sum += value;
    }

    double mean = (double)sum / indexes.size();
    double std = 0.0;

    if (indexes.size() > 1)
    {
        hasStd = true;
        double cum = 0.0;

        foreach (const _Tp& v, values)
        {
            cum += std::pow((v - mean), 2);
        }

        std = std::sqrt(cum / (indexes.size() - 1));
    }
    else
    {
        hasStd = false;
    }

    if (std::is_floating_point<_Tp>())
    {
        infos.append(QObject::tr("Minimum: %1%2").arg((double)minimum, 0, 'g').arg(v));
        infos.append(QObject::tr("Maximum: %1%2").arg((double)maximum, 0, 'g').arg(v));
    }
    else
    {
        infos.append(QObject::tr("Minimum: %1%2").arg((qint64)minimum).arg(v));
        infos.append(QObject::tr("Maximum: %1%2").arg((qint64)maximum).arg(v));
    }

    infos.append(QObject::tr("Mean: %1%2").arg(mean, 0, 'g').arg(v));

    if (hasStd)
    {
        infos.append(QObject::tr("StdDev: %1%2").arg(std, 0, 'g').arg(v));
    }
}

template <>
void gatherSelectionInformation<ito::Rgba32>(
    const ito::DataObject* dObj, const QModelIndexList& indexes, QStringList& infos)
{
}

template <>
void gatherSelectionInformation<ito::complex64>(
    const ito::DataObject* dObj, const QModelIndexList& indexes, QStringList& infos)
{
    ito::complex128 sum = 0;
    ito::complex64 value;

    QString v = QString::fromLocal8Bit(dObj->getValueUnit().data());

    if (v != "")
    {
        v.prepend(" ");
    }

    foreach (const QModelIndex& idx, indexes)
    {
        value = dObj->at<ito::complex64>(idx.row(), idx.column());
        sum += value;
    }

    sum /= indexes.size();
    infos.append(QObject::tr("Mean: %1%2%3%4")
                     .arg(sum.real(), 0, 'g')
                     .arg(sum.imag() >= 0 ? "+" : "")
                     .arg(sum.imag(), 0, 'g')
                     .arg(v));
}

template <>
void gatherSelectionInformation<ito::complex128>(
    const ito::DataObject* dObj, const QModelIndexList& indexes, QStringList& infos)
{
    ito::complex128 sum = 0;
    ito::complex128 value;

    QString v = QString::fromLocal8Bit(dObj->getValueUnit().data());

    if (v != "")
    {
        v.prepend(" ");
    }

    foreach (const QModelIndex& idx, indexes)
    {
        value = dObj->at<ito::complex128>(idx.row(), idx.column());
        sum += value;
    }

    sum /= indexes.size();
    infos.append(QObject::tr("Mean: %1%2%3%4")
                     .arg(sum.real(), 0, 'g')
                     .arg(sum.imag() >= 0 ? "+" : "")
                     .arg(sum.imag(), 0, 'g')
                     .arg(v));
}

//-------------------------------------------------------------------------------------
void DataObjectTable::selectionChanged(
    const QItemSelection& selected, const QItemSelection& deselected)
{
    QTableView::selectionChanged(selected, deselected);

    const QModelIndexList& indexes = selectedIndexes(); // selected.indexes();

    if (indexes.size() == 0)
    {
        emit selectionInformationChanged("");
    }
    else
    {
        const auto& dObjPtr = getData();

        if (dObjPtr)
        {
            int count = indexes.size();
            QStringList infos;

            infos.append(tr("Count: %1").arg(count));

            switch (dObjPtr->getType())
            {
            case ito::tUInt8:
                gatherSelectionInformation<ito::uint8>(dObjPtr.data(), indexes, infos);
                break;
            case ito::tInt8:
                gatherSelectionInformation<ito::int8>(dObjPtr.data(), indexes, infos);
                break;
            case ito::tUInt16:
                gatherSelectionInformation<ito::uint16>(dObjPtr.data(), indexes, infos);
                break;
            case ito::tInt16:
                gatherSelectionInformation<ito::int16>(dObjPtr.data(), indexes, infos);
                break;
            case ito::tUInt32:
                gatherSelectionInformation<ito::uint32>(dObjPtr.data(), indexes, infos);
                break;
            case ito::tInt32:
                gatherSelectionInformation<ito::int32>(dObjPtr.data(), indexes, infos);
                break;
            case ito::tFloat32:
                gatherSelectionInformation<ito::float32>(dObjPtr.data(), indexes, infos);
                break;
            case ito::tFloat64:
                gatherSelectionInformation<ito::float64>(dObjPtr.data(), indexes, infos);
                break;
            case ito::tComplex64:
                gatherSelectionInformation<ito::complex64>(dObjPtr.data(), indexes, infos);
                break;
            case ito::tComplex128:
                gatherSelectionInformation<ito::complex128>(dObjPtr.data(), indexes, infos);
                break;
            case ito::tRGBA32:
                gatherSelectionInformation<ito::Rgba32>(dObjPtr.data(), indexes, infos);
                break;
            default:
                infos.append(tr("No further information due to unsupported array type."));
                break;
            }

            // should not happen
            emit selectionInformationChanged(infos.join(", "));
        }
        else
        {
            qDebug() << "error gathering selection information.";
        }
    }
}
