/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
    Universit�t Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut f�r Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef PICKERLEGENDWIDGET_H
#define PICKERLEGENDWIDGET_H

#ifdef __APPLE__
extern "C++" {
#endif

#include "../common/commonGlobal.h"
#include "../common/typeDefs.h"
#include "../common/shape.h"

#include <qpoint.h>
#include <qvector3d.h>
#include <qvector4d.h>
#if QT_VERSION < 0x050000
#include <qtreewidget.h>
#include <qhash.h>
#include <qpixmap.h>
#else
#include <QtWidgets/qtreewidget.h>
#include <QtGui/qpixmap.h>
//
#endif

class ITOMCOMMONQT_EXPORT PickerInfoWidget : public QTreeWidget
{
    Q_OBJECT
        
    public:        
		PickerInfoWidget(QWidget* parent = NULL);

    private:
        QHash< int, int> m_relationHash;

    public slots:
		void updatePicker(const int index, const QPointF position);
		void updatePickers(const QVector<int> indices, const QVector< QPointF> positions);
		void updatePicker(const int index, const QVector3D position);
        void updatePickers(const QVector<int> indices, const QVector< QVector3D> positions);

		void updateChildPlot(const int index, int type, const QVector4D positionAndDirection);
		void updateChildPlots(const QVector<int> indices, const QVector<int> type, const QVector<QVector4D> positionAndDirection);
		void removeChildPlot(int index);
		void removeChildPlots();

        void removePicker(int index);
        void removePickers();

		QPixmap renderToPixMap(const int xsize, const int ysize, const int resolution);

    private slots:
};

#ifdef __APPLE__
}
#endif

#endif // MARKERLEGENDWIDGET_H