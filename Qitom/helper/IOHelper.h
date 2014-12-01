/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#ifndef IOHELPER_H
#define IOHELPER_H

#include "../global.h"
#include "../common/sharedStructures.h"
#include "../common/sharedStructuresQt.h"
#include "../../common/addInInterface.h"

#include <qstring.h>
#include <qobject.h>
#include <qicon.h>

namespace ito {

class IOHelper : public QObject
{
    Q_OBJECT
public:
    enum IOFilter
    {
        IOInput = 0x001,        /*!< consider algorithms for file input */
        IOOutput = 0x002,       /*!< consider algorithms for file output */
        IOPlugin = 0x004,       /*!< consider algorithms provided by plugins */
        IOAllFiles = 0x008,     /*!< add the "All Files (*.*)" filter */
        IOWorkspace = 0x010,     /*!< only consider filters which can be imported or exported from python workspace */
        IOMimeDataObject = 0x020,
        IOMimePointCloud = 0x040,
        IOMimePolygonMesh = 0x080,
        IOMimeAll = IOMimeDataObject | IOMimePointCloud | IOMimePolygonMesh
    };
    Q_DECLARE_FLAGS(IOFilters, IOFilter)

    enum SearchFolder
    {
        SFResources = 0x001,
        SFDirect = 0x002,
        SFCurrent = 0x004,
        SFAppDir = 0x008,
        SFAppDirQItom = 0x010,
        SFAll = SFResources | SFDirect | SFCurrent | SFAppDir | SFAppDirQItom
    };
    Q_DECLARE_FLAGS(SearchFolders, SearchFolder)


    static RetVal openGeneralFile(const QString &generalFileName, bool openUnknownsWithExternalApp = true, bool showMessages = false, QWidget* parent = NULL, const char* errorSlotMemberOfParent = NULL, bool globalNotLocalWorkspace = true);

    static RetVal uiExportPyWorkspaceVars(bool globalNotLocal, const QStringList &varNames, QVector<int> compatibleParamBaseTypes, QString defaultPath = QString::Null(), QWidget* parent = NULL);
    static RetVal exportPyWorkspaceVars(const QString &filename, bool globalNotLocal, const QStringList &varNames);

    static RetVal uiImportPyWorkspaceVars(bool globalNotLocal, IOFilters IOfilters, QString defaultPath = QString::Null(), QWidget* parent = NULL);
    static RetVal importPyWorkspaceVars(const QString &filename, bool globalNotLocal);

    static RetVal uiOpenPythonScript(QString defaultPath = QString::Null(), QWidget* parent = NULL);
    static RetVal openPythonScript(const QString &filename);

    static RetVal uiOpenFileWithFilter(const ito::AddInAlgo::FilterDef *filter, const QString &filename, QWidget *parent = NULL, bool globalNotLocal = true);
    static RetVal uiSaveFileWithFilter(QSharedPointer<ito::ParamBase> &value, const QString &filename, QWidget *parent = NULL);

    static RetVal openUIFile(const QString &filename, QWidget* parent = NULL, const char* errorSlotMemberOfParent = NULL);

    static QString getFileFilters(const IOFilters &IOfilters, QStringList *allPatterns = NULL);

    static bool fileFitsToFileFilters(const QString &filename, const IOFilters &IOfilters);

    static void elideFilepathMiddle(QString &path, int pixelLength);

    static QIcon searchIcon(const QString &filename, SearchFolders searchFolders = SFAll, const QIcon &fallbackIcon = QIcon());

private:
    IOHelper() {};
    ~IOHelper() {};
    IOHelper(const IOHelper &) : QObject() {};

};

} //end namespace ito

#endif
