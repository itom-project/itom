/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2022, Institut fuer Technische Optik (ITO),
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
#include <qactiongroup.h>
#include <qapplication.h>
#include <qclipboard.h>
#include <qdebug.h>
#include <qdir.h>
#include <qevent.h>
#include <qfiledialog.h>
#include <qheaderview.h>
#include <qinputdialog.h>
#include <qmenu.h>
#include <qmessagebox.h>
#include <qregion.h>
#include <qscrollbar.h>

#include "dataObjectDelegate.h"
#include "dataObjectModel.h"
#include "dialogHeatmapConfiguration.h"

#include "common/typeDefs.h"
#include "common/helperDatetime.h"


class DataObjectTablePrivate
{
public:
    DataObjectTablePrivate() :
        m_pActNumberFormatStandard(nullptr), m_pActNumberFormatScientific(nullptr),
        m_pActNumberFormatAuto(nullptr), m_pActCopySelection(nullptr),
        m_pActClearSelection(nullptr), m_pActExportCsv(nullptr),
        m_pActResizeColumnsToContent(nullptr), m_pActHeatmapOff(nullptr), m_pActHeatmapRgb(nullptr),
        m_pActHeatmapRYG(nullptr), m_pActHeatmapGYR(nullptr), m_pActHeatmapRWG(nullptr),
        m_pActHeatmapGWR(nullptr), m_pMenuHeatmap(nullptr), m_pActHeatmapConfig(nullptr)
    {
    }

    struct CellItem
    {
        QVariant value; // int, double or string allowed
        QColor bgColor;
    };

    QAction* m_pActNumberFormatStandard;
    QAction* m_pActNumberFormatScientific;
    QAction* m_pActNumberFormatAuto;

    QAction* m_pActCopySelection;
    QAction* m_pActExportCsv;
    QAction* m_pActClearSelection;
    QAction* m_pActResizeColumnsToContent;

    QAction* m_pActHeatmapOff;
    QAction* m_pActHeatmapRgb;
    QAction* m_pActHeatmapRYG;
    QAction* m_pActHeatmapGYR;
    QAction* m_pActHeatmapRWG;
    QAction* m_pActHeatmapGWR;
    QAction* m_pActHeatmapConfig;

    QMenu* m_pMenuHeatmap;

    ito::RetVal copyToClipboard(const QVector<CellItem>& items, int rows, int cols);
    ito::RetVal saveToCsv(
        const QString& filename,
        const QVector<CellItem>& items,
        int rows,
        int cols,
        char format,
        int precision,
        QChar decimalSign,
        const QByteArray& valueSeparator,
        const QByteArray& rowSeparator);
    ito::RetVal getSelectedItems(
        DataObjectModel* model,
        const QModelIndexList& selectedIndices,
        QVector<CellItem>& items,
        int& rows,
        int& cols);
};

//------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
ito::RetVal DataObjectTablePrivate::copyToClipboard(
    const QVector<CellItem>& items, int rows, int cols)
{
    // items must be sorted row by row
    if (items.size() != rows * cols)
    {
        return ito::RetVal(
            ito::retError, 0, "copy to clipboard not possible due to inconsistent data.");
    }

    if (items.size() == 0)
    {
        return ito::retOk;
    }

    QMimeData* mime = new QMimeData();

    // 1. text format
    QStringList rowTexts;
    QStringList rowsHtml;
    QStringList columnTexts;
    QStringList columnsHtml;
    QLocale locale;
    QString attributes;

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            const CellItem& item = items[r * cols + c];

            if (item.bgColor.isValid())
            {
                attributes = QString(" bgcolor=\"%1\"").arg(item.bgColor.name());
            }
            else
            {
                attributes = "";
            }

            if (item.value.isValid())
            {
                if (item.value.type() == QVariant::LongLong ||
                    item.value.type() == QVariant::String)
                {
                    columnTexts.append(item.value.toString());
                    columnsHtml.append(
                        QString("<td%1>%2</td>").arg(attributes).arg(item.value.toString()));
                }
                else if (item.value.type() == QVariant::Double)
                {
                    columnTexts.append(locale.toString(item.value.toDouble(), 'f', 8));
                    columnsHtml.append(
                        QString("<td%1>%2</td>").arg(attributes).arg(columnTexts.last()));
                }
                else
                {
                    return ito::RetVal(ito::retError, 0, "invalid type in item value");
                }
            }
            else
            {
                columnTexts.append("");
                columnsHtml.append("<td></td>");
            }
        }

        rowTexts.append(columnTexts.join("\t"));
        columnTexts.clear();
        rowsHtml.append(QString("<tr>%1</tr>").arg(columnsHtml.join("")));
        columnsHtml.clear();
    }

    mime->setText(rowTexts.join("\n"));

    QString html =
        QString("<html><body><table cellspacing=\"0\" border=\"0\">%1</table></body></html>")
            .arg(rowsHtml.join(""));
    mime->setHtml(html);

    QApplication::clipboard()->setMimeData(mime);

    return ito::retOk;
}

//-------------------------------------------------------------------------------------
ito::RetVal DataObjectTablePrivate::saveToCsv(
    const QString& filename,
    const QVector<CellItem>& items,
    int rows,
    int cols,
    char format,
    int precision,
    QChar decimalSign,
    const QByteArray& valueSeparator,
    const QByteArray& rowSeparator)
{
    // items must be sorted row by row
    if (items.size() != rows * cols)
    {
        return ito::RetVal(
            ito::retError, 0, "export to CSV not possible due to inconsistent data.");
    }

    if (items.size() == 0)
    {
        return ito::retOk;
    }

    QList<QByteArray> rowTexts;
    QList<QByteArray> columnTexts;
    QByteArray itemText;


    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            const CellItem& item = items[r * cols + c];

            if (item.value.isValid())
            {
                if (item.value.type() == QVariant::LongLong ||
                    item.value.type() == QVariant::String)
                {
                    itemText = item.value.toByteArray();
                }
                else if (item.value.type() == QVariant::Double)
                {
                    itemText = QByteArray::number(item.value.toDouble(), format, precision);

                    if (decimalSign != '.')
                    {
                        itemText = itemText.replace('.', decimalSign.toLatin1());
                    }
                }
                else
                {
                    return ito::RetVal(ito::retError, 0, "invalid type in item value");
                }

                columnTexts.append(itemText);
            }
            else
            {
                columnTexts.append("");
            }
        }

        rowTexts.append(columnTexts.join(valueSeparator));
        columnTexts.clear();
    }

    QFile file(filename);

    if (file.open(QIODevice::ReadWrite | QIODevice::Truncate))
    {
        file.write(rowTexts.join(rowSeparator));
        file.close();
    }
    else
    {
        return ito::RetVal::format(
            ito::retError,
            0,
            "Could not open file '%s' for writing the CSV data.",
            filename.toLatin1().data());
    }

    return ito::retOk;
}

//-------------------------------------------------------------------------------------
bool sortByRowAndColumn(const QModelIndex& idx1, const QModelIndex& idx2)
{
    if (idx1.row() == idx2.row())
    {
        return idx1.column() < idx2.column();
    }

    return idx1.row() < idx2.row();
}

//-------------------------------------------------------------------------------------
ito::RetVal DataObjectTablePrivate::getSelectedItems(
    DataObjectModel* model,
    const QModelIndexList& selectedIndices,
    QVector<CellItem>& items,
    int& rows,
    int& cols)
{
    items.clear();
    rows = 0;
    cols = 0;

    QModelIndexList selected = selectedIndices;

    if (selected.size() > 0)
    {
        std::sort(selected.begin(), selected.end(), sortByRowAndColumn);

        int firstRow = selected[0].row();
        int lastRow = selected[selected.size() - 1].row();
        int firstCol = INT_MAX;
        int lastCol = 0;

        foreach (const QModelIndex& idx, selected)
        {
            firstCol = std::min(firstCol, idx.column());
            lastCol = std::max(lastCol, idx.column());
        }

        cols = 1 + lastCol - firstCol;
        rows = 1 + lastRow - firstRow;
        items.resize(cols * rows);
        int currentIdx = 0;

        foreach (const QModelIndex& idx, selected)
        {
            currentIdx = cols * (idx.row() - firstRow) + (idx.column() - firstCol);
            items[currentIdx].bgColor = model->data(idx, Qt::BackgroundRole).value<QColor>();
            items[currentIdx].value =
                model->data(idx, DataObjectModel::longlongDoubleOrStringRoleWithoutSuffix);
        }
    }
    else
    {
        rows = model->rowCount();
        cols = model->columnCount();

        if (rows * cols > 0)
        {
            items.resize(cols * rows);
            int currentIdx = 0;

            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    currentIdx = r * cols + c;
                    QModelIndex idx = model->index(r, c);
                    items[currentIdx].bgColor =
                        model->data(idx, Qt::BackgroundRole).value<QColor>();
                    items[currentIdx].value =
                        model->data(idx, DataObjectModel::longlongDoubleOrStringRoleWithoutSuffix);
                }
            }
        }
    }

    return ito::retOk;
}


QHash<DataObjectTable*, DataObjectTablePrivate*> DataObjectTable::PrivateHash =
    QHash<DataObjectTable*, DataObjectTablePrivate*>();


//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
DataObjectTable::DataObjectTable(QWidget* parent /*= 0*/) : QTableView(parent)
{
    PrivateHash[this] = new DataObjectTablePrivate();

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

    createActions();

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

    delete PrivateHash[this];
    PrivateHash.remove(this);
}

//-------------------------------------------------------------------------------------
void DataObjectTable::createActions()
{
    DataObjectTablePrivate* d = PrivateHash[this];

    d->m_pActCopySelection = new QAction(
        QIcon(":/files/icons/tableExportSelection.svg"),
        tr("Copy Table / Selection to Clipboard"),
        this);
    connect(
        d->m_pActCopySelection,
        &QAction::triggered,
        this,
        &DataObjectTable::copySelectionToClipboard);
    d->m_pActCopySelection->setStatusTip(tr("Copy the entire table or the current selection (and "
                                            "optional heatmap background) to the clipboard."));
    addAction(d->m_pActCopySelection);

    d->m_pActExportCsv = new QAction(
        QIcon(":/files/icons/exportToCsv.svg"), tr("Export Table / Selection to CSV File"), this);
    connect(d->m_pActExportCsv, &QAction::triggered, this, &DataObjectTable::saveSelectionToCSV);
    addAction(d->m_pActExportCsv);

    QAction* a = new QAction(this);
    a->setSeparator(true);
    addAction(a);

    d->m_pActClearSelection =
        new QAction(QIcon(":/table/icons/clearSelection.svg"), tr("Clear Selection"), this);
    d->m_pActClearSelection->setEnabled(false);
    connect(d->m_pActClearSelection, &QAction::triggered, this, &DataObjectTable::clearSelection);
    addAction(d->m_pActClearSelection);

    a = new QAction(QIcon(":general/icons/decimals.png"), tr("Decimals..."), this);
    connect(a, &QAction::triggered, this, &DataObjectTable::setDecimalsGUI);
    a->setStatusTip(tr("Set the number of decimals."));
    addAction(a);

    d->m_pActNumberFormatStandard = new QAction(tr("Standard"), this);
    d->m_pActNumberFormatStandard->setData(NumberFormat::Standard);
    d->m_pActNumberFormatStandard->setCheckable(true);
    d->m_pActNumberFormatScientific = new QAction(tr("Scientific"), this);
    d->m_pActNumberFormatScientific->setData(NumberFormat::Scientific);
    d->m_pActNumberFormatScientific->setCheckable(true);
    d->m_pActNumberFormatAuto = new QAction(tr("Auto"), this);
    d->m_pActNumberFormatAuto->setData(NumberFormat::Auto);
    d->m_pActNumberFormatAuto->setCheckable(true);

    QActionGroup* ag = new QActionGroup(this);
    ag->addAction(d->m_pActNumberFormatStandard);
    ag->addAction(d->m_pActNumberFormatScientific);
    ag->addAction(d->m_pActNumberFormatAuto);
    d->m_pActNumberFormatStandard->setChecked(true);
    connect(ag, &QActionGroup::triggered, this, &DataObjectTable::numberFormatTriggered);

    QMenu* numberFormatMenu = new QMenu(tr("Number Format"), this);
    numberFormatMenu->setIcon(QIcon(":general/icons/number_format.png"));
    numberFormatMenu->addActions(ag->actions());

    addAction(numberFormatMenu->menuAction());

    d->m_pActResizeColumnsToContent = new QAction(
        QIcon(":/misc/icons/resizeColumnsToContent.svg"), tr("Resize Columns to Content"), this);
    connect(
        d->m_pActResizeColumnsToContent,
        &QAction::triggered,
        this,
        &DataObjectTable::resizeColumnsToContents);
    d->m_pActResizeColumnsToContent->setStatusTip(
        tr("Resizes all columns to fit the current content."));
    addAction(d->m_pActResizeColumnsToContent);


    d->m_pActHeatmapOff = new QAction(QIcon(":/misc/icons/heatmap_off.svg"), tr("Off"), this);
    d->m_pActHeatmapOff->setData(HeatmapType::Off);
    d->m_pActHeatmapOff->setCheckable(true);

    d->m_pActHeatmapRgb =
        new QAction(QIcon(":/application/icons/color-icon.png"), tr("Real Color"), this);
    d->m_pActHeatmapRgb->setData(HeatmapType::RealColor);
    d->m_pActHeatmapRgb->setCheckable(true);

    d->m_pActHeatmapRYG =
        new QAction(QIcon(":/misc/icons/heatmap_ryg.svg"), tr("Red-Yellow-Green"), this);
    d->m_pActHeatmapRYG->setData(HeatmapType::RedYellowGreen);
    d->m_pActHeatmapRYG->setCheckable(true);

    d->m_pActHeatmapGYR =
        new QAction(QIcon(":/misc/icons/heatmap_gyr.svg"), tr("Green-Yellow-Red"), this);
    d->m_pActHeatmapGYR->setData(HeatmapType::GreenYellowRed);
    d->m_pActHeatmapGYR->setCheckable(true);

    d->m_pActHeatmapRWG =
        new QAction(QIcon(":/misc/icons/heatmap_rwg.svg"), tr("Red-White-Green"), this);
    d->m_pActHeatmapRWG->setData(HeatmapType::RedWhiteGreen);
    d->m_pActHeatmapRWG->setCheckable(true);

    d->m_pActHeatmapGWR =
        new QAction(QIcon(":/misc/icons/heatmap_gwr.svg"), tr("Green-White-Red"), this);
    d->m_pActHeatmapGWR->setData(HeatmapType::GreenWhiteRed);
    d->m_pActHeatmapGWR->setCheckable(true);

    QActionGroup* ag2 = new QActionGroup(this);
    ag2->addAction(d->m_pActHeatmapOff);
    ag2->addAction(d->m_pActHeatmapRgb);
    ag2->addAction(d->m_pActHeatmapRYG);
    ag2->addAction(d->m_pActHeatmapGYR);
    ag2->addAction(d->m_pActHeatmapRWG);
    ag2->addAction(d->m_pActHeatmapGWR);
    d->m_pActHeatmapOff->setChecked(true);
    connect(ag2, &QActionGroup::triggered, this, &DataObjectTable::heatmapTriggered);

    d->m_pMenuHeatmap = new QMenu(tr("Heatmap"), this);
    d->m_pMenuHeatmap->setIcon(QIcon(":/misc/icons/heatmap.svg"));
    d->m_pMenuHeatmap->addActions(ag2->actions());
    d->m_pMenuHeatmap->addSeparator();

    d->m_pActHeatmapConfig =
        new QAction(QIcon(":/application/icons/adBlockAction.png"), tr("Configure..."), this);
    connect(d->m_pActHeatmapConfig, &QAction::triggered, this, &DataObjectTable::configureHeatmap);
    d->m_pMenuHeatmap->addAction(d->m_pActHeatmapConfig);


    addAction(d->m_pMenuHeatmap->menuAction());
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setData(QSharedPointer<ito::DataObject> dataObj)
{
    DataObjectTablePrivate* d = PrivateHash[this];

    emit selectionInformationChanged("");

    m_pModel->setDataObject(dataObj);

    int type = dataObj->getType();
    d->m_pMenuHeatmap->setEnabled(type != ito::tComplex64 && type != ito::tComplex128);
    d->m_pActHeatmapRgb->setVisible(type == ito::tRGBA32);
    d->m_pActHeatmapGWR->setVisible(type != ito::tRGBA32);
    d->m_pActHeatmapRWG->setVisible(type != ito::tRGBA32);
    d->m_pActHeatmapGYR->setVisible(type != ito::tRGBA32);
    d->m_pActHeatmapRYG->setVisible(type != ito::tRGBA32);

    horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);

    selectionChanged(QItemSelection(), QItemSelection());
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
    QModelIndexList indices = selectedIndexes();
    m_pModel->setDecimals(value);
    restoreSelection(indices);
}

//-------------------------------------------------------------------------------------
void DataObjectTable::restoreSelection(const QModelIndexList& indices)
{
    QItemSelection selection;
    QModelIndex idxTL, idxBR; // topLeft, bottomRight

    // workaround to get from single model indices to
    // a small list of compact QItemSelectionRanges. For
    // this workaround, methods from QRegion are used.
    QRegion regions;

    foreach (const QModelIndex& idx, indices)
    {
        regions += QRect(idx.column(), idx.row(), 1, 1);
    }

#if (QT_VERSION >= QT_VERSION_CHECK(5,8,0))
    auto it = regions.begin();
    auto it_end = regions.end();
#else
    QVector<QRect> rects = regions.rects();
    auto it = rects.constBegin();
    auto it_end = rects.constEnd();
#endif

    for (; it != it_end; ++it)
    {
        idxTL = m_pModel->index(it->top(), it->left());
        idxBR = m_pModel->index(it->bottom(), it->right());
        selection.append(QItemSelectionRange(idxTL, idxBR));
    }

    selectionModel()->select(selection, QItemSelectionModel::Select);
    selectionChanged(selectionModel()->selection(), QItemSelection());
}

//-------------------------------------------------------------------------------------
void DataObjectTable::setNumberFormat(const NumberFormat& format)
{
    QModelIndexList indices = selectedIndexes();
    const char numberFormats[3] = {'f', 'e', 'g'};
    m_pModel->setNumberFormat(numberFormats[format]);

    DataObjectTablePrivate* priv = PrivateHash[this];

    switch (format)
    {
    case Standard:
        priv->m_pActNumberFormatStandard->setChecked(true);
        break;
    case Scientific:
        priv->m_pActNumberFormatScientific->setChecked(true);
        break;
    default:
        priv->m_pActNumberFormatAuto->setChecked(true);
        break;
    }

    restoreSelection(indices);
}

//-------------------------------------------------------------------------------------
DataObjectTable::NumberFormat DataObjectTable::getNumberFormat() const
{
    char nf = m_pModel->getNumberFormat();

    switch (nf)
    {
    case 'f':
        return Standard;
    case 'e':
        return Scientific;
    default:
        return Auto;
    }
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
    contextMenu.addActions(actions());
    contextMenu.exec(event->globalPos());
    event->accept();
}


//-------------------------------------------------------------------------------------
void DataObjectTable::copySelectionToClipboard()
{
    DataObjectTablePrivate* d = PrivateHash[this];

    QVector<DataObjectTablePrivate::CellItem> items;
    int rows;
    int cols;
    ito::RetVal retval = d->getSelectedItems(m_pModel, selectedIndexes(), items, rows, cols);

    if (!retval.containsError())
    {
        retval += d->copyToClipboard(items, rows, cols);
    }

    if (retval.containsError())
    {
        QMessageBox::critical(
            this,
            tr("Error copying to clipboard"),
            tr("The table could not be copied to the clipboard. Reason: %1")
                .arg(retval.errorMessage()));
    }
}

//-------------------------------------------------------------------------------------
void DataObjectTable::saveSelectionToCSV()
{
    QDir current = QDir::currentPath();

    QString filename = QFileDialog::getSaveFileName(
        this, tr("CSV File"), current.filePath("export.csv"), tr("CSV Files (*.csv *.txt)"));

    if (filename == "")
    {
        return;
    }
    else
    {
        DataObjectTablePrivate* d = PrivateHash[this];

        QVector<DataObjectTablePrivate::CellItem> items;
        int rows;
        int cols;
        ito::RetVal retval = d->getSelectedItems(m_pModel, selectedIndexes(), items, rows, cols);

        if (!retval.containsError())
        {
            retval += d->saveToCsv(filename, items, rows, cols, 'f', 6, '.', "\t", "\n");
        }

        if (retval.containsError())
        {
            QMessageBox::critical(
                this,
                tr("Error exporting to CSV"),
                tr("The table could not be exported to the CSV file. Reason: %1")
                    .arg(retval.errorMessage()));
        }
    }
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
typename std::enable_if<std::is_floating_point<_Tp>::value, bool>::type isNaN(const _Tp& value)
{
    return std::isnan(value);
}

template <typename _Tp>
typename std::enable_if<!std::is_floating_point<_Tp>::value, bool>::type isNaN(const _Tp& value)
{
    return false;
}


//-------------------------------------------------------------------------------------
template <typename _Tp>
void gatherSelectionInformation(
    const ito::DataObject* dObj, const QModelIndexList& indexes, QStringList& infos)
{
    if (indexes.size() == 0)
    {
        infos.clear();
        return;
    }

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

        if (values.size() == 0)
        {
            minimum = value;
            maximum = value;
        }

        if (!std::is_floating_point<_Tp>() || !isNaN(value))
        {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            values.push_back(value);
            sum += value;
        }
    }

    double mean =
        values.size() > 0 ? (double)sum / values.size() : std::numeric_limits<_Tp>::quiet_NaN();
    double std = 0.0;

    if (values.size() > 1)
    {
        hasStd = true;
        double cum = 0.0;

        foreach (const _Tp& v, values)
        {
            cum += std::pow((v - mean), 2);
        }

        std = std::sqrt(cum / (values.size() - 1));
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

template <>
void gatherSelectionInformation<ito::DateTime>(
    const ito::DataObject* dObj, const QModelIndexList& indexes, QStringList& infos)
{
    if (indexes.size() == 0)
    {
        infos.clear();
        return;
    }

    ito::DateTime minimum;
    ito::DateTime maximum;

    QString v = QString::fromLocal8Bit(dObj->getValueUnit().data());

    if (v != "")
    {
        v.prepend(" ");
    }

    ito::DateTime value;
    size_t count = 0;

    foreach(const QModelIndex& idx, indexes)
    {
        value = dObj->at<ito::DateTime>(idx.row(), idx.column());

        if (count == 0)
        {
            minimum = value;
            maximum = value;
        }
        else
        {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }

        count++;
    }

    if (count > 0)
    {
        QString minStr = ito::datetime::toQDateTime(minimum).toString(Qt::ISODateWithMs);
        QString maxStr = ito::datetime::toQDateTime(maximum).toString(Qt::ISODateWithMs);
        infos.append(QObject::tr("Minimum: %1%2").arg(minStr).arg(v));
        infos.append(QObject::tr("Maximum: %1%2").arg(maxStr).arg(v));
    }
}

template <>
void gatherSelectionInformation<ito::TimeDelta>(
    const ito::DataObject* dObj, const QModelIndexList& indexes, QStringList& infos)
{
    if (indexes.size() == 0)
    {
        infos.clear();
        return;
    }

    ito::TimeDelta minimum;
    ito::TimeDelta maximum;

    QString v = QString::fromLocal8Bit(dObj->getValueUnit().data());

    if (v != "")
    {
        v.prepend(" ");
    }

    ito::TimeDelta value;
    size_t count = 0;

    foreach(const QModelIndex& idx, indexes)
    {
        value = dObj->at<ito::TimeDelta>(idx.row(), idx.column());

        if (count == 0)
        {
            minimum = value;
            maximum = value;
        }
        else
        {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }

        count++;
    }

    if (count > 0)
    {
        auto timeDeltaToString = [](const ito::TimeDelta &td)
        {
            int days, seconds, useconds;
            ito::timedelta::toDSU(td, days, seconds, useconds);

            int sec = seconds % 60;
            seconds -= sec;
            int minutes = seconds / 60;
            int min = minutes % 60;
            minutes -= min;
            int hour = minutes / 60;
            QLatin1Char fill('0');

            QString result;

            if (days != 0)
            {
                result = QObject::tr("%1 days ").arg(days);
            }

            if ((days >= 0) && (sec < 0 || min < 0 || hour < 0 || useconds < 0))
            {
                result += "-";
            }

            result += QObject::tr("%1:%2:%3")
                .arg(std::abs(hour), 2, 10, fill)
                .arg(std::abs(min), 2, 10, fill)
                .arg(std::abs(sec), 2, 10, fill);

            if (useconds != 0)
            {
                if (useconds % 1000 == 0)
                {
                    result += QString(".%1").arg(std::abs(useconds / 1000), 3, 10, fill);
                }
                else
                {
                    result += QString(".%1").arg(std::abs(useconds), 6, 10, fill);
                }
            }

            return result;
        };

        QString minStr = timeDeltaToString(minimum);
        QString maxStr = timeDeltaToString(maximum);
        infos.append(QObject::tr("Minimum: %1%2").arg(minStr).arg(v));
        infos.append(QObject::tr("Maximum: %1%2").arg(maxStr).arg(v));
    }
}

//-------------------------------------------------------------------------------------
void DataObjectTable::selectionChanged(
    const QItemSelection& selected, const QItemSelection& deselected)
{
    DataObjectTablePrivate* d = PrivateHash[this];

    QTableView::selectionChanged(selected, deselected);

    const QModelIndexList& indexes = selectedIndexes(); // selected.indexes();

    if (d->m_pActClearSelection)
    {
        d->m_pActClearSelection->setEnabled(indexes.size() > 0);
    }

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
            case ito::tDateTime:
                gatherSelectionInformation<ito::DateTime>(dObjPtr.data(), indexes, infos);
                break;
            case ito::tTimeDelta:
                gatherSelectionInformation<ito::TimeDelta>(dObjPtr.data(), indexes, infos);
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

//-------------------------------------------------------------------------------------
void DataObjectTable::numberFormatTriggered(QAction* a)
{
    setNumberFormat((NumberFormat)a->data().toInt());
}

//-------------------------------------------------------------------------------------
void DataObjectTable::heatmapTriggered(QAction* a)
{
    a->setChecked(true);

    QModelIndexList indices = selectedIndexes();

    m_pModel->setHeatmapType(a->data().toInt());

    restoreSelection(indices);
}

//-------------------------------------------------------------------------------------
void DataObjectTable::configureHeatmap()
{
    ito::AutoInterval interval = m_pModel->getHeatmapInterval();

    ito::DialogHeatmapConfiguration dlg(interval, this);

    if (dlg.exec() == QDialog::Accepted)
    {
        interval = dlg.getInterval();
        QModelIndexList indices = selectedIndexes();
        m_pModel->setHeatmapInterval(interval);
        restoreSelection(indices);
    }
}
