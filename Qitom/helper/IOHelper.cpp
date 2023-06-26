/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2022, Institut fuer Technische Optik (ITO),
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

#include "../python/pythonEngineInc.h"
#include "../python/pythonEngine.h"

#if ITOM_POINTCLOUDLIBRARY > 0
    #include "../../PointCloud/pclStructures.h"
#endif

#include "../organizer/scriptEditorOrganizer.h"
#include "../organizer/processOrganizer.h"
#include "../../AddInManager/addInManager.h"
#include "../../AddInManager/algoInterfaceValidator.h"
#include "compatHelper.h"
#include "../ui/dialogOpenFileWithFilter.h"
#include "../ui/dialogSaveFileWithFilter.h"

#include "../AppManagement.h"
#include "IOHelper.h"

#include <qdir.h>
#include <qfile.h>
#include <qfiledialog.h>
#include <qfileinfo.h>
#include <qdesktopservices.h>
#include <qurl.h>
#include <qprocess.h>
#include <QtNetwork/qtcpsocket.h>
#include <QtNetwork/qhostaddress.h>
#include <qinputdialog.h>
#include <qmessagebox.h>
#include <qsettings.h>
#include <qregularexpression.h>

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    #include <qtextcodec.h>
#else
    #include <QStringDecoder>
#endif

namespace ito {

/*!
    \class IOHelper
    \brief This class contains several static methods to load or save various file formats.

    The methods in this class can be used to save or load data objects, point clouds or polygonal meshes
    to or from various file formats. The algorithms for most file formats are not directly supported
    by this class, but algorithm plugins are scanned and checked if they support loading or saving from or
    to different formats. If so, the specific method in the plugin is called by methods defined in this class.

    Most methods can be operated with or without GUI support, hence either message boxes are displayed or
    the communication is done by RetVal only.
*/

//! name of set of all itom files (used in file open dialog or file system dialog)
QString IOHelper::allItomFilesName = QObject::tr("Itom Files");

//! the list of all officially supported encodings for Python scripts
//! This will be filled by the first call to IOHelper::getSupportedScriptEncodings
QList<IOHelper::CharsetEncodingItem> IOHelper::supportedScriptEncodings;

//------------------------------------------------------------------------------------------------------------------------------------------------
//! method to load any supported file
/*!
    This method tries to load any given file that is directly or indirectly supported.
    Indirect support means that algorithm plugins are checked for their support for the
    given file format. If there is a corresponding method found, it is used to open the file.

    Possible file formats are:

    * .py -> open the python file in a script editor
    * .idc -> loads the content of the 'itom data collection' file to the global or local python workspace
    * .mat -> load the content of the Matlab file to the global or local python workspace using Scipy (only if the package Scipy is available)
    * .ui -> open the ui file in the QtDesigner application
    * else -> tries to find at least algorithm that supports this file ending and can load it to a data object, point cloud or polygonal mesh (to global or local workspace)

    If two or more algorithms pretend to be able to load the file format, a dialog appears where the user can select the desired filter.

    \param generalFileName is the file name to load. If the file name is not absolute, it is considered to be relative to the current directory.
    \param openUnknownsWithExternalApp is a boolean variable that indicates if an unsupported or unknown file format is opened with the external
               application that is officially connected with this file format
    \param showMessages if true, an error or warning during the execution of this method will be displayed in a message box.
    \param parent is the widget this method should be related to. Dialogs or messages are then displayed using this parent.
    \param errorSlotMemberOfParent is only considered for ui-files. Pass a SLOT(myMethod(QProcess::ProcessError)) description such that errors
               occurred in the QtDesigner will call the given slot. Else pass NULL.
    \param globalNotLocalWorkspace is only considered when files are opened that load data objects, point clouds or polygonal meshes to the Python
               workspace. If true, the object is loaded to the global workspace, else to the local (only allowed if a local workspace is currently available)
    \return success of loading as RetVal
    \sa openPythonScript, importPyWorkspaceVars, openUIFile, uiOpenFileWithFilter
*/
/*static */RetVal IOHelper::openGeneralFile(const QString &generalFileName, bool openUnknownsWithExternalApp, bool showMessages, QWidget* parent, const char* errorSlotMemberOfParent, bool globalNotLocalWorkspace /*= true*/)
{
    QFile file(generalFileName);
    ito::RetVal retval(ito::retOk);

    if (!file.exists())
    {
        retval += RetVal(retError, 1001, tr("File %1 does not exist").arg(generalFileName).toLatin1().data());
        goto end;
    }
    else
    {
        QFileInfo fileinfo(file);
        QString suffix = fileinfo.suffix().toLower();

        if (suffix == "py")
        {
            retval += openPythonScript(generalFileName);
            goto end;
        }
        else if (suffix == "idc") //itom data collection
        {
            retval += importPyWorkspaceVars(generalFileName, globalNotLocalWorkspace, parent);
            goto end;
        }
        else if (suffix == "mat") //matlab file
        {
            retval += importPyWorkspaceVars(generalFileName, globalNotLocalWorkspace, parent);
            goto end;
        }
        else if (suffix == "ui") //UI file
        {
            retval += openUIFile(generalFileName, parent, errorSlotMemberOfParent);
            goto end;
        }
        else //check whether there is a plugin which can open this file
        {
            ito::AddInManager *AIM = static_cast<ito::AddInManager*>(AppManagement::getAddInManager());
            ito::AddInAlgo::FilterDef *filter = NULL;
            if (AIM)
            {
                QList<ito::AddInAlgo::FilterDef*> filters = AIM->getFilterByInterface(ito::AddInAlgo::iReadDataObject, suffix);
                filters << AIM->getFilterByInterface(ito::AddInAlgo::iReadPointCloud, suffix);
                filters << AIM->getFilterByInterface(ito::AddInAlgo::iReadPolygonMesh, suffix);

                if (filters.size() == 1)
                {
                    filter = filters[0];
                }
                else if (filters.size() > 1)
                {
                    bool ok = false;
                    QStringList items;
                    foreach(const ito::AddInAlgo::FilterDef* f, filters)
                    {
                        items.append(f->m_name);
                    }
                    QString result = QInputDialog::getItem(parent, tr("Multiple plugins"), tr("Multiple plugins provide methods to load the file of type '%1'. Please choose one.").arg(suffix), items, 0, false, &ok);
                    if (ok)
                    {
                        filter = filters[items.indexOf(result)];
                    }
                    else
                    {
                        return ito::retOk;
                    }
                }

                if (filter)
                {
                    retval += uiOpenFileWithFilter(filter, generalFileName, parent, globalNotLocalWorkspace);
                    goto end;
                }
            }
        }

        if (openUnknownsWithExternalApp)
        {
            QUrl url = QUrl::fromLocalFile(generalFileName);
            if (QDesktopServices::openUrl(url))
            {
                return RetVal(retOk);
            }
            else
            {
                retval += RetVal(retError, 1002, tr("File '%1' could not be opened with registered external application").arg(generalFileName).toLatin1().data());
                goto end;
            }
        }
        else
        {
            retval += RetVal(retError, 1002, tr("File %1 can not be opened with this application").arg(generalFileName).toLatin1().data());
            goto end;
        }
    }

end:
    if (retval.containsWarningOrError())
    {
        if (showMessages)
        {
            QMessageBox msgBox(parent);

            if (retval.containsError())
            {
                msgBox.setIcon(QMessageBox::Critical);
                msgBox.setWindowTitle(tr("Error"));
            }
            else
            {
                msgBox.setIcon(QMessageBox::Warning);
                msgBox.setWindowTitle(tr("Warning"));
            }

            if (retval.hasErrorMessage())
            {
                QString errStr = QLatin1String(retval.errorMessage());
                msgBox.setText(errStr);
            }
            else
            {
                msgBox.setText(tr("Unknown error or warning when opening the file."));
            }

            msgBox.exec();
        }
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------
//! export one or more variables from a python workspace
/*!
    This method allows exporting one or more variables from the global or local python workspace to
    a user defined file. A file save dialog is shown to the user in order to select the desired file name.
    Depending on the type of the given variables, the dialog only allows inserting supported file formats.

    One or multiple variables can be saved in idc (itom data collection) or mat (Matlab) containers using
    the method exportPyWorkspaceVars. Single variables can also be exported to suitable file formats using
    the method uiSaveFileWithFilter. This finally uses a suitable filter method from an algorithm plugin.

    In case of an export based on a plugin filter, the data related to the variable name is obtained
    from the workspace by invoking the slot getParamsFromWorkspace in the class PythonEngine.

    \param globalNotLocal defines if the variables are exported from the global (true) or local (false) workspace
    \param varNames is a list if one or multiple variable names within the workspace. These can be single variable names (direct child of local or global workspace, or a full item name to any subitem as it is used in workspaceWidget)
    \param compatibleParamBaseTypes is a vector of the same size than varNames. A value can be ito::ParamBase::DObjPtr, ito::ParamBase::PointCloudPtr
              or ito::ParamBase::PolygonMeshPtr to describe the type of the variable or 0 if the variable covers another object. This information is
              used to set the filters in the file save dialog.
    \param defaultPath is the default path that is pre-set in the file save dialog.
    \param parent is the parent widget of the file save dialog.
    \return success of the export as RetVal
    \sa exportPyWorkspaceVars, uiSaveFileWithFilter, getParamsFromWorkspace
*/
/*static */RetVal IOHelper::uiExportPyWorkspaceVars(bool globalNotLocal, const QStringList &varNames, QVector<int> compatibleParamBaseTypes, QString defaultPath, QWidget* parent)
{
    static QString uiExportPyWorkspaceDefaultPath;

    if (defaultPath.isNull() || defaultPath.isEmpty())
    {
        if(uiExportPyWorkspaceDefaultPath == "")
        {
            uiExportPyWorkspaceDefaultPath = QDir::currentPath();
        }
        defaultPath = uiExportPyWorkspaceDefaultPath;
    }

    if (compatibleParamBaseTypes.size() == 0)
    {
        compatibleParamBaseTypes.fill(varNames.size(), 0);
    }

    Q_ASSERT(varNames.size() == compatibleParamBaseTypes.size());
    IOFilters filters = IOHelper::IOOutput | IOHelper::IOWorkspace | IOHelper::IOPlugin;

    //multiple files can only be saved to idc or mat containers, only one single file can be saved using an plugin filter.
    //Its search can then be limited to a mime type.
    if (varNames.size() == 1)
    {
        switch(compatibleParamBaseTypes[0])
        {
        case ito::ParamBase::DObjPtr:
            filters |= IOHelper::IOMimeDataObject;
            break;
        case ito::ParamBase::PointCloudPtr:
            filters |= IOHelper::IOMimePointCloud;
            break;
        case ito::ParamBase::PolygonMeshPtr:
            filters |= IOHelper::IOMimePolygonMesh;
            break;
        }
    }

    QString filterString = IOHelper::getFileFilters(filters);
    static QString selectedFilter; //this will save the recently selected filter for the next time
    QString filename = QFileDialog::getSaveFileName(parent, tr("Save selected variables as..."), defaultPath, filterString, &selectedFilter);

    if (filename.isEmpty())
    {
        return RetVal(retOk);
    }
    else
    {
        QFileInfo info(filename);
        QString suffix = info.suffix().toLower();

        uiExportPyWorkspaceDefaultPath = info.path(); //save directory as default for next call to this export dialog

        if (suffix == "idc" || suffix == "mat")
        {
            return exportPyWorkspaceVars(filename, globalNotLocal, varNames);
        }
        else
        {
            //try to open it with filters
            RetVal retVal;
            QSharedPointer<ito::ParamBase> value;
            PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
            if (eng == NULL)
            {
                return RetVal(retError, 1, tr("Python engine not available").toLatin1().data());
            }

            if (eng->isPythonBusy() && !eng->isPythonDebuggingAndWaiting())
            {
                return RetVal(retError, 2, tr("Variables cannot be exported since python is busy right now").toLatin1().data());
            }

            QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
            //get values from workspace
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            QSharedPointer<SharedParamBasePointerVector> values(new SharedParamBasePointerVector());
            QMetaObject::invokeMethod(eng, "getParamsFromWorkspace",Q_ARG(bool,globalNotLocal), Q_ARG(QStringList, varNames), Q_ARG(QVector<int>, compatibleParamBaseTypes), Q_ARG(QSharedPointer<SharedParamBasePointerVector>, values), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
            if (!locker.getSemaphore()->wait(5000))
            {
                retVal += RetVal(retError, 0, tr("Timeout while getting value from workspace").toLatin1().data());
            }
            else
            {
                retVal += locker.getSemaphore()->returnValue;
            }

            if (values->size() != varNames.size())
            {
                retVal += RetVal(retError, 0, tr("The number of values returned from workspace does not correspond to requested number").toLatin1().data());
            }
            QApplication::restoreOverrideCursor();

            for (int i=0;i<varNames.size() && !retVal.containsError();i++)
            {
                retVal += uiSaveFileWithFilter((*values)[0], filename, parent);
            }

            return retVal;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! export one or more variables from a python workspace to an idc or mat file
/*!
    This method exports one or more variables from the global or local python workspace
    to a idc (itom data collection) or mat (Matlab) container file. Other file suffixes
    will return an error. For the export, the slots pickleVariables (idc) or saveMatlabVariables (mat),
    defined in class PythonEngine, are invoked. Mat is only supported if the Python package Scipy is available.
    The idc format is written via the Python module pickle.

    \param filename is the filename to the idc or mat file
    \param globalNotLocal defines if the variables are exported from the global (true) or local (false) workspace
    varNames is a list if one or multiple variable names within the workspace. These can be single variable names (direct child of local or global workspace, or a full item name to any subitem as it is used in workspaceWidget)
    \return success of the export as RetVal
    \sa uiExportPyWorkspaceVars
*/
/*static */RetVal IOHelper::exportPyWorkspaceVars(const QString &filename, bool globalNotLocal, const QStringList &varNames)
{
    RetVal retValue(retOk);

    QFile file(filename);

    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return RetVal(retError, 3, tr("File cannot be opened").toLatin1().data());
    }
    file.close();

    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (eng == NULL)
    {
        return RetVal(retError, 1, tr("Python engine not available").toLatin1().data());
    }

    if (eng->isPythonBusy() && !eng->isPythonDebuggingAndWaiting())
    {
        return RetVal(retError, 2, tr("Variables cannot be exported since python is busy right now").toLatin1().data());
    }

    QFileInfo info(file);
    QString suffix = info.suffix().toLower();

    if (suffix == "idc")
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore(1));

        QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
        QApplication::processEvents(QEventLoop::ExcludeSocketNotifiers); //the WaitCursor only becomes visible if the event loop of the main thread is called once.
                                                                         //(it is not allowed to filter  QEventLoop::ExcludeUserInputEvents here out, since mouse events
                                                                         //have to be passed to the operating system. Else the cursor is not changed. - at least with Windows)

        QMetaObject::invokeMethod(eng, "pickleVariables", Q_ARG(bool,globalNotLocal), Q_ARG(QString,filename), Q_ARG(QStringList,varNames), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        if (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginFileSaveLoad))
        {
            retValue += RetVal(retError, 2, tr("Timeout while pickling variables").toLatin1().data());
        }
        else
        {
            retValue += locker.getSemaphore()->returnValue;
        }

        QApplication::restoreOverrideCursor();
    }
    else if (suffix == "mat")
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore(1));

        QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
        QApplication::processEvents(QEventLoop::ExcludeSocketNotifiers); //the WaitCursor only becomes visible if the event loop of the main thread is called once.
                                                                         //(it is not allowed to filter  QEventLoop::ExcludeUserInputEvents here out, since mouse events
                                                                         //have to be passed to the operating system. Else the cursor is not changed. - at least with Windows)

        QMetaObject::invokeMethod(eng, "saveMatlabVariables", Q_ARG(bool,globalNotLocal), Q_ARG(QString,filename), Q_ARG(QStringList,varNames), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        if (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginFileSaveLoad))
        {
            retValue += RetVal(retError, 2, tr("Timeout while saving variables to matlab file").toLatin1().data());
        }
        else
        {
            retValue += locker.getSemaphore()->returnValue;
        }

        QApplication::restoreOverrideCursor();
    }
    else
    {
        retValue += RetVal(retError, 0, tr("Suffix must be *.idc or *.mat").toLatin1().data());
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! open a file load dialog and let the user selected a file that is opened and load to a python workspace
/*!
    This method opens a file load dialog and let the user selected a file. The file filters can
    be adjusted, such that for instance only file formats that contain data objects, point clouds and/or polygonal meshes
    are allowed. The selected file is then opened using openGeneralFile. If loading the file using a plugin filter
    requires further mandatory or optional parameters, a generic parameter input dialog is shown, too.

    \param globalNotLocal defines if the variables are loaded to the global (true) or local (false) workspace
    \param IOfilters is an or combination of IOFilter to adjust the supported file formats.
    \param defaultPath is the default path that is pre-set in the file load dialog.
    \param parent is the parent widget of the file load dialog.
    \return success of the import as RetVal
    \sa openGeneralFile
*/
/*static */RetVal IOHelper::uiImportPyWorkspaceVars(bool globalNotLocal, const IOFilters &IOfilters, QString defaultPath, QWidget* parent)
{

    static QString uiImportPyWorkspaceDefaultPath;

    if (defaultPath.isNull() || defaultPath.isEmpty())
    {
        if(uiImportPyWorkspaceDefaultPath == "")
        {
            uiImportPyWorkspaceDefaultPath = QDir::currentPath();
        }
        defaultPath = uiImportPyWorkspaceDefaultPath;
    }

    QString filters = IOHelper::getFileFilters(IOfilters & (~IOOutput));
    static QString selectedFilter; //this variable will contain the last selected filter which is the default for the next time

    QString filename = QFileDialog::getOpenFileName(parent, tr("Import data"), defaultPath, filters, &selectedFilter);

    if (filename.isEmpty())
    {
        return RetVal(retOk);
    }
    else
    {
        QFile file(filename);
        if (file.exists())
        {
            uiImportPyWorkspaceDefaultPath = QFileInfo(filename).canonicalPath(); //save directory as default for next call to this export dialog

            QFileInfo info(filename);
            return openGeneralFile(filename, false, false, parent, nullptr, globalNotLocal);
        }
        else
        {
            return RetVal(retError, 1, tr("File not found").toLatin1().data());
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! import an idc or mat file and load the content to a python workspace
/*!
    Import an idc (itom data collection) or mat (Matlab) ocntainer to the global or local
    python workspace. This is done by an invokation of the slot unpickleVariables or loadMatlabVariables
    of the class PythonEngine.

    \param filename is the filename with a suffix idc or mat (only supported if Scipy is available)
    \param globalNotLocal defines if the file is loaded to the global (true) or local (false) workspace
    \return success of the import as RetVal
    \sa unpickleVariables, loadMatlabVariables
*/
/*static */RetVal IOHelper::importPyWorkspaceVars(const QString &filename, bool globalNotLocal, QWidget* parent /*= NULL*/)
{
    RetVal retValue(retOk);

    QFile file(filename);

    if (!file.open(QIODevice::ReadOnly))
    {
        return RetVal(retError, 3, tr("File cannot be opened").toLatin1().data());
    }
    file.close();

    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (eng == NULL)
    {
        return RetVal(retError, 1, tr("Python engine not available").toLatin1().data());
    }

    if (eng->isPythonBusy() && !eng->isPythonDebuggingAndWaiting())
    {
        //if multiple files are dragged into the workspace, the python-free signal is sometimes received in the upcoming event-lopp. Therefore, process this and verify
        //again if python is still busy.

        QApplication::processEvents(QEventLoop::ExcludeSocketNotifiers);
        if (eng->isPythonBusy() && !eng->isPythonDebuggingAndWaiting())
        {
            return RetVal(retError, 2, tr("Variables cannot be imported since python is busy right now").toLatin1().data());
        }
    }

    QFileInfo info(file);
    QString suffix = info.suffix().toLower();

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("Workspace");
    bool unpackDict = settings.value("importIdcMatUnpackDict", "true").toBool();
    QString packedVarname = "";
    settings.endGroup();

    if (!unpackDict)
    {
        QRegularExpression regExp("^[a-zA-Z][a-zA-Z0-9_]*$");
        QString defaultName = info.completeBaseName();

        if (defaultName.indexOf(regExp) == -1)
        {
            //defaultName.prepend("var");
            defaultName.replace("-", "_");
			defaultName.replace(".", "_");
			defaultName.replace(" ", "_");

            if (defaultName.indexOf(regExp) == -1)
            {
                defaultName = "varName";
            }
        }

        bool ok = true;

        if (parent)
        {
            parent->activateWindow();
        }

        packedVarname = QInputDialog::getText(parent, tr("Variable name of imported dictionary"), tr("Please indicate a variable name for the dictionary in file '%1' \n(name must start with a letter followed by numbers or letters).").arg(info.fileName()), QLineEdit::Normal, defaultName, &ok);

        if (!ok)
        {
            return ito::retOk;
        }

        if (packedVarname.indexOf(regExp) == -1)
        {
            return RetVal(retError, 0, tr("Invalid variable name").toLatin1().data());
        }
    }

    if (suffix == "idc")
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore(1));

        QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
        QApplication::processEvents(QEventLoop::ExcludeSocketNotifiers); //the WaitCursor only becomes visible if the event loop of the main thread is called once.
                                                                         //(it is not allowed to filter  QEventLoop::ExcludeUserInputEvents here out, since mouse events
                                                                         //have to be passed to the operating system. Else the cursor is not changed. - at least with Windows)

        QMetaObject::invokeMethod(eng, "unpickleVariables", Q_ARG(bool,globalNotLocal), Q_ARG(QString,filename), Q_ARG(QString,packedVarname), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
        if (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginFileSaveLoad))
        {
            retValue += RetVal(retError, 2, tr("Timeout while unpickling variables").toLatin1().data());
        }
        else
        {
            retValue += locker.getSemaphore()->returnValue;
        }

        QApplication::restoreOverrideCursor();
    }
    else if (suffix == "mat")
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore(1));

        QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
        QApplication::processEvents(QEventLoop::ExcludeSocketNotifiers); //the WaitCursor only becomes visible if the event loop of the main thread is called once.
                                                                         //(it is not allowed to filter  QEventLoop::ExcludeUserInputEvents here out, since mouse events
                                                                        //have to be passed to the operating system. Else the cursor is not changed. - at least with Windows)

        QMetaObject::invokeMethod(eng, "loadMatlabVariables", Q_ARG(bool, globalNotLocal), Q_ARG(QString, filename), Q_ARG(QString, packedVarname), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        if (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginFileSaveLoad))
        {
            retValue += RetVal(retError, 2, tr("Timeout while loading matlab variables").toLatin1().data());
        }
        else
        {
            retValue += locker.getSemaphore()->returnValue;
        }

        QApplication::restoreOverrideCursor();
    }
    else
    {
        retValue += RetVal(retError, 0, tr("Suffix must be *.idc or *.mat").toLatin1().data());
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! open a given python file in a script editor window
/*!
    Open the given python file (suffix *.py) using the slot openScript of the
    class ScriptEditorOrganizer.

    \param filename is the filename to the py file
    \return retOk in case of success and retError in case of an error or timeout.
    \sa openScript
*/
/*static */RetVal IOHelper::openPythonScript(const QString &filename)
{
    QFile file(filename);
    RetVal retValue(retOk);

    if (file.exists())
    {
        QObject *seo = AppManagement::getScriptEditorOrganizer();

        if (seo != NULL)
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore(1));

            QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
            QApplication::processEvents(QEventLoop::ExcludeSocketNotifiers); //the WaitCursor only becomes visible if the event loop of the main thread is called once.
                                                                             //(it is not allowed to filter  QEventLoop::ExcludeUserInputEvents here out, since mouse events
                                                                             //have to be passed to the operating system. Else the cursor is not changed. - at least with Windows)

            QMetaObject::invokeMethod(seo,"openScript",Q_ARG(QString, filename),Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
            if (!locker.getSemaphore()->wait(20000))
            {
                retValue += RetVal(retError, 2, tr("Timeout while opening script").toLatin1().data());
            }
            QApplication::restoreOverrideCursor();

            return retValue;
        }

        return RetVal(retError);
    }
    else
    {
        return RetVal(retError);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! open ui file in an instance of QtDesigner
/*!
    Tries to open the given ui file in a new or already opened instance of QtDesigner.
    The designer folder of itom is passed as plugin path to the QtDesigner such that
    itom designer plugins are also considered as widgets in the QtDesigner application.

    It is possible to pass a slot with a single argument of type QProcess::ProcessError.
    If this slot is given, it is connected to the error signal of the QtDesigner process such that
    error during the startup... of QtDesigner can be appropriately handled.

    \param filename is the filename to the ui file
    \param parent is the widget where the slot given by errorSlotMemberOfParent is defined
    \param errorSlotMemberOfParent is SLOT(myMethod(QProcess::ProcessError)) description such that errors
               occurred in the QtDesigner will call the given slot. Else pass NULL.
    \return retOk in case of success and retError in case of an error or timeout.
*/
/*static */RetVal IOHelper::openUIFile(const QString &filename, QWidget* parent, const char* errorSlotMemberOfParent)
{
    ProcessOrganizer *po = qobject_cast<ProcessOrganizer*>(AppManagement::getProcessOrganizer());
    if (po)
    {
        bool existingProcess = false;
        QProcess *process = po->getProcess("designer", true, existingProcess, true);

        if (existingProcess && process->state() == QProcess::Running)
        {
            //the output message is mainly useful in order to get the socket-number of the designer (started as server)
            //the designer returns this number over stdOut. When opening another ui-file, this ui-file is send over a socket
            //connection
            QString tcpPort = po->getStandardOutputBuffer("designer");
            bool done = false;
            if (tcpPort != "")
            {
                QTcpSocket socket;
                quint16 tcpPort2 = tcpPort.toInt();
                if (tcpPort2 != 0)
                {
                    QString ipAddress = QHostAddress(QHostAddress::LocalHost).toString();
                    socket.connectToHost(ipAddress, tcpPort2, QIODevice::WriteOnly);
                    bool result = socket.waitForConnected(30000);
                    if (result)
                    {
                        QByteArray filename_ = filename.toLatin1() + "\n";
                        socket.write(filename_);
                        done = true;
                        socket.disconnectFromHost();
                        socket.waitForDisconnected(250);

                        //Q_PID processID = process->pid();
                        po->bringWindowsOnTop("Qt Designer");
                    }
                }
            }

            if (!done)
            {
                process = po->getProcess("designer", false, existingProcess, true);
                //create new process for designer, since sending data to existing one failed
                QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
                QString appPath = QDir::cleanPath(QCoreApplication::applicationDirPath());
                env.insert("QT_PLUGIN_PATH", appPath);

#ifdef WIN32
                QString pathEnv = env.value("Path");
                pathEnv.prepend(appPath + ";");
                env.insert("Path", pathEnv);
#endif

                process->setProcessEnvironment(env);

                if (errorSlotMemberOfParent != nullptr)
                {
                    connect(process, SIGNAL(errorOccurred(QProcess::ProcessError)), parent, errorSlotMemberOfParent);
                }

                po->clearStandardOutputBuffer("designer");

                QStringList arguments;
                arguments << "-server" << filename;

                QString app = ProcessOrganizer::getAbsQtToolPath( "designer" );
                process->start(app, arguments);
            }
        }
        else
        {
            QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
            QString appPath = QDir::cleanPath(QCoreApplication::applicationDirPath());
            env.insert("QT_PLUGIN_PATH", appPath);

#ifndef WIN32
            QString pathEnv = env.value("PATH");
            pathEnv.prepend(appPath + ":");
            env.insert("PATH", pathEnv);
#else
            QString pathEnv = env.value("Path");
            pathEnv.prepend(appPath + ";");
            env.insert("Path", pathEnv);
#endif

            process->setProcessEnvironment(env);

            if (errorSlotMemberOfParent != nullptr)
            {
                connect(process, SIGNAL(errorOccurred(QProcess::ProcessError)), parent, errorSlotMemberOfParent);
            }

            po->clearStandardOutputBuffer("designer");

            QStringList arguments;
            arguments << "-server" << filename;
            QString app = ProcessOrganizer::getAbsQtToolPath( "designer" );
            //qDebug() << app << arguments;
            process->start(app, arguments);
        }
    }

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! open a file using a filter method from an algorithm plugin and shows an import dialog
/*!
    This method tries to open a given file using a given filter method from an algorithm plugin.
    If the file could be successfully loaded (to a data object, point cloud or polygonal mesh), it is
    imported to the global or local python workspace. The given filter method must support one of the
    filter interfaces ito::AddInAlgo::iReadDataObject, ito::AddInAlgo::iReadPointCloud or ito::AddInAlgo::iReadPolygonMesh.

    The load and preview of the file as well as an input mask for the variable name of the imported data
    is done by a dialog of class DialogOpenFileWithFilter. This dialog let the user also indicate required
    mandatory or optional parameters for the load. The variable name can also be validated and checked for duplicates.

    \param filter is a pointer to the ito::AddInAlgo::FilterDef structures that indicates the desired plugin filter method.
    \param filename is the name of the file
    \param parent is the parent widget of the load and preview dialog.
    \param globalNotLocal defines if the file should be loaded to the global (true) or local python workspace (false)
    \return success of the import as RetVal
    \sa putParamsToWorkspace, uiSaveFileWithFilter, DialogOpenFileWithFilter
*/
/*static*/ RetVal IOHelper::uiOpenFileWithFilter(const ito::AddInAlgo::FilterDef *filter, const QString &filename, QWidget *parent /*= NULL*/, bool globalNotLocal /*= true*/)
{
    RetVal retval;
    ito::AddInManager *AIM = static_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    PythonEngine *pyEng = static_cast<PythonEngine*>(AppManagement::getPythonEngine());
    QVector<ito::ParamBase> autoMand;
    QVector<ito::ParamBase> autoOut;
    QVector<ito::Param> userMand;
    QVector<ito::Param> userOpt;
    DialogOpenFileWithFilter *dialog = NULL;
    QStringList pythonVarNames;
    bool putParamsToWorkspace = false;
    QVector<SharedParamBasePointer> values;

    DialogOpenFileWithFilter::CheckVarname varnameCheck = (globalNotLocal) ? DialogOpenFileWithFilter::CheckGlobalWorkspace : DialogOpenFileWithFilter::CheckLocalWorkspace;

    ito::DataObject dObj;
#if ITOM_POINTCLOUDLIBRARY > 0
    ito::PCLPointCloud pointCloud;
    ito::PCLPolygonMesh polygonMesh;
#endif

    if (AIM && pyEng)
    {
        const ito::FilterParams *fp = AIM->getHashedFilterParams(filter->m_paramFunc);
        autoOut.clear();
        autoMand.clear();
        const AlgoInterfaceValidator *aiv = AIM->getAlgoInterfaceValidator();
        if (aiv)
        {
            retval += aiv->getInterfaceParameters(filter->m_interface, autoMand, autoOut);
            if (!retval.containsError())
            {
                userOpt = fp->paramsOpt;
                userMand = fp->paramsMand.mid(autoMand.size());
            }

            switch (filter->m_interface)
            {
            case ito::AddInAlgo::iReadDataObject:
                {
                    //1. dataObject
                    autoMand[0].setVal<void*>(&dObj);
                    //2. filename
                    autoMand[1].setVal<char*>(filename.toLatin1().data());

                    dialog = new DialogOpenFileWithFilter(filename, filter, autoMand, autoOut, userMand, userOpt, retval, varnameCheck, parent);
                    if (!retval.containsError())
                    {
                        int result = dialog->exec();
                        if (result == QDialog::Accepted)
                        {
                            //autoMand = dialog->getAutoMand();
                            //autoOut = dialog->getAutoOut();
                            pythonVarNames << dialog->getPythonVariable();
                            values << SharedParamBasePointer(new ito::ParamBase(autoMand[0]));
                            putParamsToWorkspace = true;
                        }
                    }
                }
                break;
#if ITOM_POINTCLOUDLIBRARY > 0
            case ito::AddInAlgo::iReadPointCloud:
                {
                    //1. pointCloud
                    autoMand[0].setVal<void*>(&pointCloud);
                    //2. filename
                    autoMand[1].setVal<char*>(filename.toLatin1().data());

                    dialog = new DialogOpenFileWithFilter(filename, filter, autoMand, autoOut, userMand, userOpt, retval, varnameCheck, parent);
                    if (!retval.containsError())
                    {
                        int result = dialog->exec();
                        if (result == QDialog::Accepted)
                        {
                            //autoMand = dialog->getAutoMand();
                            //autoOut = dialog->getAutoOut();
                            pythonVarNames << dialog->getPythonVariable();
                            values << SharedParamBasePointer(new ito::ParamBase(autoMand[0]));
                            putParamsToWorkspace = true;
                        }
                    }
                }
                break;
            case ito::AddInAlgo::iReadPolygonMesh:
                {
                    //1. polygonMesh
                    autoMand[0].setVal<void*>(&polygonMesh);
                    //2. filename
                    autoMand[1].setVal<char*>(filename.toLatin1().data());

                    dialog = new DialogOpenFileWithFilter(filename, filter, autoMand, autoOut, userMand, userOpt, retval, varnameCheck, parent);
                    if (!retval.containsError())
                    {
                        int result = dialog->exec();
                        if (result == QDialog::Accepted)
                        {
                            //autoMand = dialog->getAutoMand();
                            //autoOut = dialog->getAutoOut();
                            pythonVarNames << dialog->getPythonVariable();
                            values << SharedParamBasePointer(new ito::ParamBase(autoMand[0]));
                            putParamsToWorkspace = true;
                        }
                    }
                }
                break;
#else
            case ito::AddInAlgo::iReadPointCloud:
            case ito::AddInAlgo::iReadPolygonMesh:
                retval += ito::RetVal(ito::retError, 0, tr("PolygonMesh and PointCloud not available since support of PointCloudLibrary is disabled in this version.").toLatin1().data());
                break;
#endif
            default:
                retval += ito::RetVal(ito::retError, 0, tr("The algorithm interface is not supported").toLatin1().data());
            }

            if (!retval.containsError() && putParamsToWorkspace)
            {
                ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

                QMetaObject::invokeMethod(pyEng, "putParamsToWorkspace", Q_ARG(bool, globalNotLocal), Q_ARG(QStringList, pythonVarNames), Q_ARG(QVector<SharedParamBasePointer>, values), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
                if (locker.getSemaphore()->wait(AppManagement::timeouts.pluginFileSaveLoad) == false)
                {
                    QMessageBox::critical(parent, tr("Timeout while sending values to python"), tr("A timeout occurred while content of loaded file has been sent to python workspace"));
                }
                else
                {
                    retval += locker.getSemaphore()->returnValue;
                    if (retval != ito::retOk)
                    {
                        QString title;
                        QString text;
                        if (retval.hasErrorMessage()) text = QString("\n%1").arg(QLatin1String(retval.errorMessage()));
                        if (retval.containsError())
                        {
                            text.prepend(tr("An error occured while importing the loaded file into the python workspace."));
                            QMessageBox::critical(parent, tr("Error while sending values to python"), text);
                        }
                        else if (retval.containsWarning())
                        {
                            text.prepend(tr("A warning occured while importing the loaded file into the python workspace."));
                            QMessageBox::warning(parent, tr("Warning while sending values to python"), text);
                        }
                    }
                }
            }

            DELETE_AND_SET_NULL(dialog);
        }
        else
        {
            retval += RetVal(retError, 0, tr("AlgoInterfaceValidator not available.").toLatin1().data());
        }

        if (!retval.containsError())
        {

        }
    }
    else
    {
        retval += RetVal(retError, 0, tr("AddInManager or PythonEngine not available").toLatin1().data());
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! save a file using a filter method from an algorithm plugin and shows an export dialog
/*!
    This method tries to save a given data object, point cloud or polygonal mesh using a given filter method from an algorithm plugin.
    The given filter method must support one of the filter interfaces ito::AddInAlgo::iWriteDataObject,
    ito::AddInAlgo::iWritePointCloud or ito::AddInAlgo::iWritePolygonMesh.

    If the export requires further mandatory or optional parameters, an export dialog (class DialogSaveFileWithFilter)
    is shown.

    \param value is the export object as shared pointer of ParamBase. Only the types ito::ParamBase::DObjPtr, ito::ParamBase::PointCloudPtr
               and ito::ParamBase::PolygonMeshPtr are supported.
    \param filename is the name of the file
    \param parent is the parent widget of the possible export dialog.
    \return success of the export as RetVal
    \sa DialogSaveFileWithFilter, uiOpenFileWithFilter
*/
/*static*/ RetVal IOHelper::uiSaveFileWithFilter(QSharedPointer<ito::ParamBase> &value, const QString &filename, QWidget *parent /*= NULL*/)
{

    ito::AddInManager *AIM = static_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    ito::AddInAlgo::FilterDef *filter = NULL;
    QList<ito::AddInAlgo::FilterDef*> filters;
    QFileInfo info(filename);
    QString suffix = info.suffix().toLower();

    if (AIM)
    {
        switch(value->getType())
        {
        case ito::ParamBase::DObjPtr:
            filters = AIM->getFilterByInterface(ito::AddInAlgo::iWriteDataObject, suffix);
            break;
        case ito::ParamBase::PointCloudPtr:
            filters = AIM->getFilterByInterface(ito::AddInAlgo::iWritePointCloud, suffix);
            break;
        case ito::ParamBase::PolygonMeshPtr:
            filters = AIM->getFilterByInterface(ito::AddInAlgo::iWritePolygonMesh, suffix);
            break;
        default:
            return RetVal(retError, 0, tr("There is no plugin interface able to save the requested file type").toLatin1().data());
        }

        if (filters.size() == 1)
        {
            filter = filters[0];
        }
        else if (filters.size() > 1)
        {
            bool ok = false;
            QStringList items;
            foreach(const ito::AddInAlgo::FilterDef* f, filters)
            {
                items.append(f->m_name);
            }
            QString result = QInputDialog::getItem(parent, tr("Multiple plugins"), tr("Multiple plugins provide methods to save the file of type '%1'. Please choose one.").arg(suffix), items, 0, false, &ok);
            if (ok)
            {
                filter = filters[items.indexOf(result)];
            }
            else
            {
                return ito::retOk; //user cancel
            }
        }

        QVector<ito::ParamBase> autoMand;
        QVector<ito::ParamBase> autoOut;
        QVector<ito::ParamBase> paramsOpt;
        QVector<ito::ParamBase> paramsMand;
        QVector<ito::Param> userMand;
        QVector<ito::Param> userOpt;
        ito::RetVal retval;

        const ito::FilterParams *fp = AIM->getHashedFilterParams(filter->m_paramFunc);
        const AlgoInterfaceValidator *aiv = AIM->getAlgoInterfaceValidator();

        if (aiv)
        {
            retval += aiv->getInterfaceParameters(filter->m_interface, autoMand, autoOut);
            if (!retval.containsError())
            {
                switch(filter->m_interface)
                {
                case ito::AddInAlgo::iWriteDataObject:
                case ito::AddInAlgo::iWritePolygonMesh:
                case ito::AddInAlgo::iWritePointCloud:
                    autoMand[0] = *value;
                    autoMand[1].setVal<char*>(filename.toLatin1().data());
                    break;
                default:
                    retval += ito::RetVal(retError, 0, tr("Algorithm interface not supported").toLatin1().data());
                }

                if (!retval.containsError())
                {
                    userOpt = fp->paramsOpt;
                    userMand = fp->paramsMand.mid(autoMand.size());

                    ito::AddInAlgo::FilterDefExt *filterExt = dynamic_cast<ito::AddInAlgo::FilterDefExt*>(filter);
                    QSharedPointer<ito::FunctionCancellationAndObserver> emptyObserver; //no observer initialized

                    if (userMand.size() > 0 || userOpt.size() > 0)
                    {
                        DialogSaveFileWithFilter *dialog = new DialogSaveFileWithFilter(filename, filter, autoMand, autoOut, userMand, userOpt, true, parent);
                        if (dialog->exec() == QDialog::Accepted)
                        {
                            QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
                            QApplication::processEvents(QEventLoop::ExcludeSocketNotifiers); //the WaitCursor only becomes visible if the event loop of the main thread is called once.
                                                                                             //(it is not allowed to filter  QEventLoop::ExcludeUserInputEvents here out, since mouse events
                                                                                             //have to be passed to the operating system. Else the cursor is not changed. - at least with Windows)

                            dialog->getParameters(paramsMand, paramsOpt);
                            paramsMand = autoMand + paramsMand;

                            if (filterExt)
                            {
                                retval += filterExt->m_filterFuncExt(&paramsMand, &paramsOpt, &autoOut, emptyObserver);
                            }
                            else
                            {
                                retval += filter->m_filterFunc(&paramsMand, &paramsOpt, &autoOut);
                            }

                            QApplication::restoreOverrideCursor();
                        }

                        DELETE_AND_SET_NULL(dialog);
                    }
                    else
                    {
                        QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
                        QApplication::processEvents(QEventLoop::ExcludeSocketNotifiers); //the WaitCursor only becomes visible if the event loop of the main thread is called once.
                                                                                         //(it is not allowed to filter  QEventLoop::ExcludeUserInputEvents here out, since mouse events
                                                                                         //have to be passed to the operating system. Else the cursor is not changed. - at least with Windows)

                        if (filterExt)
                        {
                            retval += filterExt->m_filterFuncExt(&paramsMand, &paramsOpt, &autoOut, emptyObserver);
                        }
                        else
                        {
                            retval += filter->m_filterFunc(&paramsMand, &paramsOpt, &autoOut);
                        }

                        QApplication::restoreOverrideCursor();
                    }
                }
            }
            else
            {
                retval += RetVal(retError, 0, tr("Error while getting mand and out parameters from algorithm interface").toLatin1().data());
            }
        }
        else
        {
            retval += RetVal(retError, 0, tr("AlgoInterfaceValidator not available").toLatin1().data());
        }
        return retval;
    }
    else
    {
        return RetVal(retError, 0, tr("AddInManager not available").toLatin1().data());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Returns a list of all file endings that correspond to itom
/*!
    This function returns a QString that contains a semicolon separated list of all fileendings
    that were passed by the iofilters parameter

    \param IOfilters filters that contain the filenendings
    \param allPatterns Pointer to QStringList  (standard = 0)
    \return returns a QString with fileendings (semicolon separated)
*/
/*static*/ QString IOHelper::getFileFilters(const IOFilters &IOfilters, QStringList *allPatterns /*= NULL*/)
{
    QStringList filter;

    if (IOfilters.testFlag(ito::IOHelper::IOWorkspace) == false)
    {
        filter << tr("Python Scripts (*.py)") << tr("itom Data Collection (*.idc)") << tr("Matlab Matrix (*.mat)") << tr("User Interfaces (*.ui)");
    }
    else
    {
        filter << tr("Itom Data Collection (*.idc)") << tr("Matlab Matrix (*.mat)");
    }

    if (IOfilters.testFlag(ito::IOHelper::IOPlugin))
    {
        ito::AddInManager *AIM = static_cast<ito::AddInManager*>(AppManagement::getAddInManager());
        QList<ito::AddInAlgo::FilterDef*> filterPlugins;
        if (AIM)
        {
            if (IOfilters.testFlag(ito::IOHelper::IOInput))
            {
                if (IOfilters.testFlag(ito::IOHelper::IOMimeDataObject))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iReadDataObject, QString());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iReadDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
                if (IOfilters.testFlag(ito::IOHelper::IOMimePointCloud))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iReadPointCloud, QString());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iReadDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
                if (IOfilters.testFlag(ito::IOHelper::IOMimePolygonMesh))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iReadPolygonMesh, QString());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iReadDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
            }

            if (IOfilters.testFlag(ito::IOHelper::IOOutput))
            {
                if (IOfilters.testFlag(ito::IOHelper::IOMimeDataObject))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iWriteDataObject, QString());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iWriteDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
                if (IOfilters.testFlag(ito::IOHelper::IOMimePointCloud))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iWritePointCloud, QString());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iReadDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
                if (IOfilters.testFlag(ito::IOHelper::IOMimePolygonMesh))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iWritePolygonMesh, QString());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iReadDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
            }

            //add here further filters
        }
    }

    //delete duplicates
    filter.removeDuplicates();

    //get all file-patterns from all filters and merge them together to one entry containing all, that is then added as 'Itom Files'
    QRegularExpression reg("^.*\\((.*)\\)$");
    QStringList _allPatterns;
    QRegularExpressionMatch match;

    foreach(const QString &item, filter)
    {
        match = reg.match(item);

        if(match.hasMatch())
        {
            _allPatterns.append( match.captured(1).trimmed().split(" "));
        }
    }

    _allPatterns.removeDuplicates();

	filter << QString("%1 (%2)").arg(allItomFilesName).arg(_allPatterns.join(" "));

    if(allPatterns)
    {
        *allPatterns = _allPatterns;
    }

    if (IOfilters.testFlag(ito::IOHelper::IOAllFiles))
    {
        filter << tr("All Files (*.*)");
    }

    return filter.join(";;");
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Checks if a file fits to a filter
/*!
    This function checks if a fileending of a file fits to a given set of filters.

    \param filename pass the filename as a QString
    \param IOfilters pass the filterset that the filename shopuld be checked against
    \return returns true if the file fits to the filters, else false is returned
*/
/*static*/ bool IOHelper::fileFitsToFileFilters(const QString &filename, const IOFilters &IOfilters)
{
    QStringList allPatterns;
    getFileFilters(IOfilters, &allPatterns);
    QRegularExpression reg;

    foreach(const QString &pat, allPatterns)
    {
        reg.setPattern(CompatHelper::regExpAnchoredPattern(CompatHelper::wildcardToRegularExpression(pat)));

        if(filename.indexOf(reg) >= 0)
        {
            return true;
        }
    }

    return false;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Shortens paths so that menus can display them without becoming too big
/*!
    This functio is used to shorten paths so they fit into a menu or something
    compareable.

    Example:
    D:/testdir1/testdir2/file.py
    becomes D:/...ir2/file.py

    If the pixelLength is shorter than the filename a minimum is returned:
    D:/...file.py
    Even if the minimum is longer than the pixelLength.

    \param path The path that is supposed to be shortened
    \param pixelLength The length the path has to have after shortening
*/
/*static*/ void IOHelper::elideFilepathMiddle(QString &path, int pixelLength)
{
    QFont font;
    QFontMetrics fontm(font);

#if (QT_VERSION >= QT_VERSION_CHECK(5, 11, 0))
        int width = fontm.horizontalAdvance(path);
#else
        int width = fontm.width(path);
#endif

    if (width > pixelLength)
    {
        bool end = false;

#if (QT_VERSION >= QT_VERSION_CHECK(5, 11, 0))
        while (width > pixelLength - fontm.horizontalAdvance("...") && end == false)
#else
        while (width > pixelLength - fontm.width("...") && end == false)
#endif
        {
            int index = path.indexOf(QDir::separator(), 0)+1;
            if (index == 0 || index == path.lastIndexOf(QDir::separator()))
            {
                end = true;
            }
            path.remove(index, 1);
#if (QT_VERSION >= QT_VERSION_CHECK(5, 11, 0))
            width = fontm.horizontalAdvance(path);
#else
            width = fontm.width(path);
#endif
        }
        path.insert(path.indexOf(QDir::separator(),0)+1, "...");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! search an icon file in different locations, open and return it
/*!
    \param filename is the relative or absolute filename of the icon
    \param searchFolders is a bitmask that defines the locations that are searched for the filename
    \param fallbackIcon let you define an alternative icon that is returned if filename is not found in any location
    \return loaded icon or invalid QIcon
*/
/*static*/ QIcon IOHelper::searchIcon(const QString &filename, const SearchFolders &searchFolders /*= SFAll*/, const QIcon &fallbackIcon /*= QIcon()*/)
{
    QIcon icon;
    bool found = false;
    QDir dir;

    if (searchFolders & SFResources)
    {
        if (filename.startsWith(":"))
        {
            icon = QIcon(filename);
            if (icon.isNull() == false && icon.availableSizes().size() > 0)
            {
                found = true;
            }
        }
    }

    if (!found && (searchFolders & SFDirect))
    {
        icon = QIcon(filename);

        if (icon.isNull() == false && icon.availableSizes().size() > 0)
        {
            found = true;
        }
    }

    if (!found && (searchFolders & SFCurrent))
    {
        dir = QDir::current();

        if (!filename.isEmpty() && dir.exists() && dir.exists(filename))
        {
            icon = QIcon(dir.absoluteFilePath(filename));

            if (icon.isNull() == false && icon.availableSizes().size() > 0)
            {
                found = true;
            }
        }
    }

    if (!found && (searchFolders & SFAppDir))
    {
        dir.setPath(QCoreApplication::applicationDirPath());

        if (!filename.isEmpty() && dir.exists() && dir.exists(filename))
        {
            icon = QIcon(dir.absoluteFilePath(filename));

            if (icon.isNull() == false && icon.availableSizes().size() > 0)
            {
                found = true;
            }
        }
    }

    if (!found && (searchFolders & SFAppDirQItom))
    {
        dir.setPath(QCoreApplication::applicationDirPath());
        dir.cd("Qitom");

        if (!filename.isEmpty() && dir.exists() && dir.exists(filename))
        {
            icon = QIcon(dir.absoluteFilePath(filename));

            if (icon.isNull() == false && icon.availableSizes().size() > 0)
            {
                found = true;
            }
        }
    }

    //nothing valid found, return to fallback icon
    if (!found || icon.isNull() == true || icon.availableSizes().size() == 0)
    {
        icon = fallbackIcon;
    }

    return icon;
}

//-------------------------------------------------------------------------------------
//! return a list of default encodings, that are officially supported by Qt (as well as Python).
/*
    \return a map with the official name as key and a string list of aliases as value. The
        aliases are only in small letters, one has to convert a comparison string to small
        letters first. The aliases are mostly aliases used in Python scripts (e.g. #coding=... line).
*/
/*static*/ QList<IOHelper::CharsetEncodingItem> IOHelper::getSupportedScriptEncodings()
{
    if (supportedScriptEncodings.size() > 0)
    {
        return supportedScriptEncodings;
    }

    // aliases contain the name as well as all aliases
    // from https://docs.python.org/3.11/library/codecs.html#standard-encodings

    auto item = CharsetEncodingItem();
    item.aliases = QStringList() << "latin-1" << "latin1" << "latin_1" << "iso-8859-15" << "iso8859-1" << "8859" << "cp819" << "latin" << "L1";
    item.bom = "";
    item.encodingName = "Latin1";
    item.displayName = "Latin 1";
    item.displayNameShort = "Latin 1";
    supportedScriptEncodings.append(item);

    item = CharsetEncodingItem();
    item.aliases = QStringList() << "utf8" << "utf-8" << "utf_8" << "u8" << "utf" << "cp650001";
    item.bom = "";
    item.encodingName = "UTF-8";
    item.displayName = "UTF-8";
    item.displayNameShort = "UTF-8";
    supportedScriptEncodings.append(item);

    item = CharsetEncodingItem();
    item.aliases = QStringList() << "utf_8_sig";
    item.bom = QByteArray::fromHex("EFBBBF");
    item.encodingName = "UTF-8"; //do not change or remove, see getDefaultScriptEncoding
    item.displayName = "UTF-8 with BOM";
    item.displayNameShort = "UTF-8 BOM";
    supportedScriptEncodings.append(item);

#if 0
#if Q_BYTE_ORDER == Q_LITTLE_ENDIAN
    item = CharsetEncodingItem();
    item.aliases = QStringList() << "utf-16be" << "utf_16_be";
    item.bom = "";
    item.encodingName = "UTF-16BE";
    item.displayName = tr("UTF-16BE (not recommended for Python scripts)");
    item.displayNameShort = "UTF-16BE";
    supportedScriptEncodings.append(item);

    item = CharsetEncodingItem();
    item.aliases = QStringList() << "utf-16le" << "utf_16_le" << "utf16" << "utf-16" << "utf_16" << "u16";
    item.bom = "";
    item.encodingName = "UTF-16LE";
    item.displayName = tr("UTF-16LE (not recommended for Python scripts)");
    item.displayNameShort = "UTF-16LE";
    supportedScriptEncodings.append(item);

    item = CharsetEncodingItem();
    item.aliases = QStringList() << "utf-32be" << "utf_32_be";
    item.bom = "";
    item.encodingName = "UTF-32BE";
    item.displayName = tr("UTF-32BE (not recommended for Python scripts)");
    item.displayNameShort = "UTF-32BE";
    supportedScriptEncodings.append(item);

    item = CharsetEncodingItem();
    item.aliases = QStringList() << "utf-32le" << "utf_32_le" << "utf32" << "utf-32" << "utf_32" << "u32";
    item.bom = "";
    item.encodingName = "UTF-32LE";
    item.displayName = tr("UTF-32LE (not recommended for Python scripts)");
    item.displayNameShort = "UTF-32LE";
    supportedScriptEncodings.append(item);
#else
    item = CharsetEncodingItem();
    item.aliases = QStringList() << "utf-16be" << "utf_16_be" << "utf16" << "utf-16" << "utf_16" << "u16";
    item.bom = "";
    item.encodingName = "UTF-16BE";
    item.displayName = tr("UTF-16BE (not recommended for Python scripts)");
    item.displayNameShort = "UTF-16BE";
    supportedScriptEncodings.append(item);

    item = CharsetEncodingItem();
    item.aliases = QStringList() << "utf-16le" << "utf_16_le";
    item.bom = "";
    item.encodingName = "UTF-16LE";
    item.displayName = tr("UTF-16LE (not recommended for Python scripts)");
    item.displayNameShort = "UTF-16LE";
    supportedScriptEncodings.append(item);

    item = CharsetEncodingItem();
    item.aliases = QStringList() << "utf-32be" << "utf_32_be" << "utf32" << "utf-32" << "utf_32" << "u32";
    item.bom = "";
    item.encodingName = "UTF-32BE";
    item.displayName = tr("UTF-32BE (not recommended for Python scripts)");
    item.displayNameShort = "UTF-32BE";
    supportedScriptEncodings.append(item);

    item = CharsetEncodingItem();
    item.aliases = QStringList() << "utf-32le" << "utf_32_le";
    item.bom = "";
    item.encodingName = "UTF-32LE";
    item.displayName = tr("UTF-32LE (not recommended for Python scripts)");
    item.displayNameShort = "UTF-32LE";
    supportedScriptEncodings.append(item);
#endif

    item = CharsetEncodingItem();
    item.aliases = QStringList();
    item.bom = QByteArray::fromHex("FEFF");
    item.encodingName = "UTF-16BE";
    item.displayName = "UTF-16BE with BOM (not recommended for Python scripts)";
    item.displayNameShort = tr("UTF-16BE BOM");
    supportedScriptEncodings.append(item);

    item = CharsetEncodingItem();
    item.aliases = QStringList();
    item.bom = QByteArray::fromHex("FFFE");
    item.encodingName = "UTF-16LE";
    item.displayName = "UTF-16LE with BOM (not recommended for Python scripts)";
    item.displayNameShort = tr("UTF-16LE BOM");
    supportedScriptEncodings.append(item);

    item = CharsetEncodingItem();
    item.aliases = QStringList();
    item.bom = QByteArray::fromHex("0000FEFF");
    item.encodingName = "UTF-32BE";
    item.displayName = tr("UTF-32BE with BOM (not recommended for Python scripts)");
    item.displayNameShort = tr("UTF-32BE BOM");
    supportedScriptEncodings.append(item);

    item = CharsetEncodingItem();
    item.aliases = QStringList();
    item.bom = QByteArray::fromHex("FFFE0000");
    item.encodingName = "UTF-32LE";
    item.displayName = tr("UTF-32LE with BOM (not recommended for Python scripts)");
    item.displayNameShort = tr("UTF-32LE BOM");
    supportedScriptEncodings.append(item);
#endif

    return supportedScriptEncodings;
}

//-------------------------------------------------------------------------------------
/*static*/ IOHelper::CharsetEncodingItem IOHelper::getDefaultScriptEncoding()
{
    auto encodings = getSupportedScriptEncodings();

    foreach(const auto &enc, encodings)
    {
        if (enc.encodingName == "UTF-8" && enc.bom == "")
        {
            return enc;
        }
    }

    return CharsetEncodingItem();
}

//-------------------------------------------------------------------------------------
/*static*/ IOHelper::CharsetEncodingItem IOHelper::getEncodingFromAlias(const QString &alias, bool* found /*= nullptr*/)
{
    auto defaultEncodings = getSupportedScriptEncodings();
    auto it = defaultEncodings.constBegin();

    while (!found && it != defaultEncodings.constEnd())
    {
        foreach(const QString &s, it->aliases)
        {
            if (QString::compare(alias, s, Qt::CaseInsensitive) == 0)
            {
                if (found)
                {
                    *found = true;
                }

                return *it;
            }
        }

        ++it;
    }

    // alias not found, create a user defined one, as long as QTextCodec supports it.
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QTextCodec *tc = QTextCodec::codecForName(alias.toLatin1());

    if (tc)
#else
    QStringDecoder decoder(alias.toLatin1());

    if (decoder.isValid())
#endif
    {
        CharsetEncodingItem item;
        item.encodingName = alias;
        item.displayName = item.displayNameShort = alias;
        item.userDefined = true;

        if (found)
        {
            *found = true;
        }

        return item;
    }

    if (found)
    {
        *found = false;
    }

    return CharsetEncodingItem();
}

} //end namespace ito
