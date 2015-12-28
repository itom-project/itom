/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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

#ifndef MARKERLEGENDWIDGET_H
#define MARKERLEGENDWIDGET_H

#ifdef __APPLE__
extern "C++" {
#endif

#include "../common/commonGlobal.h"
#include "../common/typeDefs.h"
#include "../common/shape.h"

#if QT_VERSION < 0x050000
#include <qtreewidget.h>
#include <qhash.h>
#else
#include <QtWidgets/qtreewidget.h>
//
#endif

class ITOMCOMMONQT_EXPORT MarkerInfoWidget : public QTreeWidget
{
    Q_OBJECT
        
    public:        
		MarkerInfoWidget(QWidget* parent = NULL);

    private:

	public slots:
		void updateMarker(const ito::Shape element);
		void updateMarkers(const QVector< ito::Shape > elements);

        void removeMarker(int index);
        void removeMarkers();

    private slots:
};

#ifdef __APPLE__
}
#endif

#endif // MARKERLEGENDWIDGET_H