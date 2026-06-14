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

#include "qtHelpUrlSchemeHandler.h"

#ifdef ITOM_USEHELPVIEWER

#include <qhelpengine.h>
#include <qbuffer.h>
#include <qmetaobject.h>
#include <qurl.h>
#include <qwebengineurlrequestjob.h>

namespace ito {

struct ExtensionMap {
	const char *extension;
	const char *mimeType;
} extensionMap[] = {
	{ ".bmp", "image/bmp" },
	{ ".css", "text/css" },
	{ ".gif", "image/gif" },
	{ ".html", "text/html" },
	{ ".htm", "text/html" },
	{ ".ico", "image/x-icon" },
	{ ".jpeg", "image/jpeg" },
	{ ".jpg", "image/jpeg" },
	{ ".js", "application/x-javascript" },
	{ ".mng", "video/x-mng" },
	{ ".pbm", "image/x-portable-bitmap" },
	{ ".pgm", "image/x-portable-graymap" },
	{ ".pdf", "application/pdf" },
	{ ".png", "image/png" },
	{ ".ppm", "image/x-portable-pixmap" },
	{ ".rss", "application/rss+xml" },
	{ ".svg", "image/svg+xml" },
	{ ".svgz", "image/svg+xml" },
	{ ".text", "text/plain" },
	{ ".tif", "image/tiff" },
	{ ".tiff", "image/tiff" },
	{ ".txt", "text/plain" },
	{ ".xbm", "image/x-xbitmap" },
	{ ".xml", "text/xml" },
	{ ".xpm", "image/x-xpm" },
	{ ".xsl", "text/xsl" },
	{ ".xhtml", "application/xhtml+xml" },
	{ ".wml", "text/vnd.wap.wml" },
	{ ".wmlc", "application/vnd.wap.wmlc" },
	{ "about:blank", 0 },
	{ 0, 0 }
};

//--------------------------------------------------------------------------------
QtHelpUrlSchemeHandler::QtHelpUrlSchemeHandler(QHelpEngine *helpEngine, QObject *parent /*= 0*/) :
	QWebEngineUrlSchemeHandler(parent),
	m_pHelpEngine(helpEngine)
{

}

//--------------------------------------------------------------------------------
QtHelpUrlSchemeHandler::~QtHelpUrlSchemeHandler()
{

}

//--------------------------------------------------------------------------------
QByteArray QtHelpUrlSchemeHandler::mimeFromUrl(const QUrl &url)
{
	const QString &path = url.path();
	const int index = path.lastIndexOf(QLatin1Char('.'));
	const QByteArray &ext = path.mid(index).toUtf8().toLower();

	const ExtensionMap *e = extensionMap;
	while (e->extension)
	{
		if (ext == e->extension)
		{
			return e->mimeType;
		}
		++e;
	}
	return "application/octet-stream";
}

//--------------------------------------------------------------------------------
void QtHelpUrlSchemeHandler::requestStarted(QWebEngineUrlRequestJob* request)
{
	QUrl url = request->requestUrl();
	QByteArray ba = m_pHelpEngine->fileData(url);
	QBuffer *buffer = new QBuffer;
	connect(request, SIGNAL(destroyed()), buffer, SLOT(deleteLater()));

	buffer->open(QIODevice::WriteOnly);
	buffer->write(ba.data(), ba.size());
	buffer->close();

	request->reply(mimeFromUrl(url), buffer);
}

} //end namespace ito

#endif
