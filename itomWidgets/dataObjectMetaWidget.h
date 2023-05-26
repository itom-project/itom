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
*********************************************************************** */

#ifndef DATAOBJECTMETAWIDGET_H
#define DATAOBJECTMETAWIDGET_H

#include "DataObject/dataobj.h"

#include <qtreewidget.h>
#include <qsharedpointer.h>

#include "commonWidgets.h"


class ITOMWIDGETS_EXPORT DataObjectMetaWidget : public QTreeWidget
{
    Q_OBJECT

    Q_PROPERTY(QSharedPointer<ito::DataObject> data READ getData WRITE setData DESIGNABLE false);
    Q_PROPERTY(bool readOnlyEnabled READ getReadOnly WRITE setReadOnly DESIGNABLE true);
    Q_PROPERTY(bool detailedInfo READ getDetailedStatus WRITE setDetailedStatus DESIGNABLE true);
    Q_PROPERTY(bool previewEnabled READ getPreviewStatus WRITE setPreviewStatus DESIGNABLE true);
    Q_PROPERTY(int previewSize READ getPreviewSize WRITE setPreviewSize DESIGNABLE true);
    Q_PROPERTY(int decimals READ getDecimals WRITE setDecimals DESIGNABLE true);
    //Q_PROPERTY(QString colorBar READ getColorMap WRITE setColorMap DESIGNABLE true)

    Q_CLASSINFO("prop://data", "The dataObject to read the meta data from.")
    Q_CLASSINFO("prop://readOnlyEnabled", "Enable / disable modification of meta data, (not supported).")
    Q_CLASSINFO("prop://decimals", "Number of decimals to show.")
    Q_CLASSINFO("prop://previewEnabled", "Add a preview to the meta data.")
    Q_CLASSINFO("prop://previewSize", "Set the preview size.")
    Q_CLASSINFO("prop://detailedInfo", "Toggle between basic and detailed metaData.")
    //Q_CLASSINFO("prop://colorBar", "Name of the color bar for the preview.")


public:
    DataObjectMetaWidget(QWidget* parent = 0);
    ~DataObjectMetaWidget();

    void setData(QSharedPointer<ito::DataObject> dataObj);
    QSharedPointer<ito::DataObject> getData() const;

    bool getReadOnly() const {return m_readOnly;};
    void setReadOnly(const bool value);

    int getDecimals() const {return m_decimals;};
    void setDecimals(const int value);

    bool getPreviewStatus() const {return m_preview;};
    void setPreviewStatus(const bool value);

    int getPreviewSize() const {return m_previewSize;};
    void setPreviewSize(const int value);

    bool getDetailedStatus() const {return m_detailedStatus;};
    void setDetailedStatus(const bool value);
    //QString getColorMap() const {return m_colorBarName;};
    //void setColorMap(const QString &name);

    virtual QSize sizeHint() const;


protected:

private:
    bool m_readOnly;
    bool m_preview;
    bool m_detailedStatus;
    int m_previewSize;
    int m_decimals;
    QString m_colorBarName;
    QVector<ito::uint32> m_colorTable;

    ito::DataObject m_data;
};

#endif
