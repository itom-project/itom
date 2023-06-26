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

#ifndef IOHELPER_H
#define IOHELPER_H

#include "../global.h"
#include "../common/sharedStructures.h"
#include "../common/sharedStructuresQt.h"
#include "../../common/addInInterface.h"

#include <qstring.h>
#include <qobject.h>
#include <qicon.h>
#include <qlist.h>

namespace ito {



class IOHelper : public QObject
{
    Q_OBJECT
public:

    /**
    * IOFilter enumeration
    * This enum contains flags to filter out input/output algorithms for various objects
    */
    enum IOFilter
    {
        IOInput = 0x001,        /*!< consider algorithms for file input */
        IOOutput = 0x002,       /*!< consider algorithms for file output */
        IOPlugin = 0x004,       /*!< consider algorithms provided by plugins */
        IOAllFiles = 0x008,     /*!< add the "All Files (*.*)" filter */
        IOWorkspace = 0x010,     /*!< only consider filters which can be imported or exported from python workspace */
        IOMimeDataObject = 0x020, /*!< consider algorithms that allow saving or loading data objects */
        IOMimePointCloud = 0x040, /*!< consider algorithms that allow saving or loading point clouds */
        IOMimePolygonMesh = 0x080, /*!< consider algorithms that allow saving or loading polygonal meshes */
        IOMimeAll = IOMimeDataObject | IOMimePointCloud | IOMimePolygonMesh /*!< or-combination of IOMimeDataObject, IOMimePointCloud and IOMimePolygonMesh */
    };
    Q_DECLARE_FLAGS(IOFilters, IOFilter)

    /**
    * SearchFolder enumeration
    * This enumeration contains values to describe specific directories that are searched for files (e.g. icons)
    */
    enum SearchFolder
    {
        SFResources = 0x001,   /*!< search the resource files for the file */
        SFDirect = 0x002,      /*!< consider the file as absolute file path */
        SFCurrent = 0x004,     /*!< look for the given file in the current directory */
        SFAppDir = 0x008,      /*!< look for the file in the application directory of itom */
        SFAppDirQItom = 0x010, /*!< look for the file in the Qitom subdirectory of the application directory*/
        SFAll = SFResources | SFDirect | SFCurrent | SFAppDir | SFAppDirQItom /*!< or-combination of all available search folders */
    };
    Q_DECLARE_FLAGS(SearchFolders, SearchFolder)

    /** This struct defines all necessary information for a possible charset encoding.
    */
    struct CharsetEncodingItem
    {
        CharsetEncodingItem() :
            encodingName(""), displayName(""), displayNameShort(""), bom(""), userDefined(false)
        {}

        //!< the name that is accepted by QTextCodec or QStringConverter.
        //!< This is also used as value in the itom settings.
        QString encodingName;

        //!< the name used as display name in a config dialog
        QString displayName;

        //!< the short version, e.g. for a status bar
        QString displayNameShort;

        //!< all possible aliases, e.g. in Python scripts (coding=...)
        QStringList aliases;

        //!< the Byte-Order-Mark used to automatically detect this encoding (or empty, if not used)
        QByteArray bom;

        //!< false if part of the officially supported list of encodings, else true
        bool userDefined;
    };


    static RetVal openGeneralFile(const QString &generalFileName, bool openUnknownsWithExternalApp = true, bool showMessages = false, QWidget* parent = NULL, const char* errorSlotMemberOfParent = NULL, bool globalNotLocalWorkspace = true);

    static RetVal uiExportPyWorkspaceVars(bool globalNotLocal, const QStringList &varNames, QVector<int> compatibleParamBaseTypes, QString defaultPath = QString(), QWidget* parent = NULL);
    static RetVal exportPyWorkspaceVars(const QString &filename, bool globalNotLocal, const QStringList &varNames);

    static RetVal uiImportPyWorkspaceVars(bool globalNotLocal, const IOFilters &IOfilters, QString defaultPath = QString(), QWidget* parent = NULL);
    static RetVal importPyWorkspaceVars(const QString &filename, bool globalNotLocal, QWidget* parent = NULL);

    static RetVal openPythonScript(const QString &filename);

    static RetVal uiOpenFileWithFilter(const ito::AddInAlgo::FilterDef *filter, const QString &filename, QWidget *parent = NULL, bool globalNotLocal = true);
    static RetVal uiSaveFileWithFilter(QSharedPointer<ito::ParamBase> &value, const QString &filename, QWidget *parent = NULL);

    static RetVal openUIFile(const QString &filename, QWidget* parent = NULL, const char* errorSlotMemberOfParent = NULL);

    static QString getFileFilters(const IOFilters &IOfilters, QStringList *allPatterns = NULL);

    static bool fileFitsToFileFilters(const QString &filename, const IOFilters &IOfilters);

    static void elideFilepathMiddle(QString &path, int pixelLength);

    static QIcon searchIcon(const QString &filename, const SearchFolders &searchFolders = SFAll, const QIcon &fallbackIcon = QIcon());

    static QString getAllItomFilesName() { return allItomFilesName; } /*!< name of file filter that bundles are readable files of itom, usually 'Itom Files'. */

    static QList<CharsetEncodingItem> getSupportedScriptEncodings();
    static CharsetEncodingItem getDefaultScriptEncoding();
    static CharsetEncodingItem getEncodingFromAlias(const QString &alias, bool* found = nullptr);

private:
    IOHelper() {}; /*!< private constructor since this class only contains static method and no instance must be created */
    ~IOHelper() {}; /*!< private destructor */
    IOHelper(const IOHelper &) : QObject() {};

    static QString allItomFilesName;
    static QList<CharsetEncodingItem> supportedScriptEncodings;

};

} //end namespace ito

Q_DECLARE_METATYPE(ito::IOHelper::CharsetEncodingItem)
Q_DECLARE_OPERATORS_FOR_FLAGS(ito::IOHelper::IOFilters)

#endif
