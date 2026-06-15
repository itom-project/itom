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

#ifndef QTHELPURLSCHEMEHANDLER_H
#define QTHELPURLSCHEMEHANDLER_H

#include "../global.h"

#ifdef ITOM_USEHELPVIEWER

#include <qwebengineurlschemehandler.h>
#include <qbytearray.h>

class QHelpEngine;

namespace ito {

class QtHelpUrlSchemeHandler : public QWebEngineUrlSchemeHandler
{
	Q_OBJECT
public:
	QtHelpUrlSchemeHandler(QHelpEngine *helpEngine, QObject *parent = 0);
	~QtHelpUrlSchemeHandler();

	void requestStarted(QWebEngineUrlRequestJob* request);

private:
	QHelpEngine *m_pHelpEngine;

	QByteArray mimeFromUrl(const QUrl &url);
};

} //end namespace ito

#endif
#endif
