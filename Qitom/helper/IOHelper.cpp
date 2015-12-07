/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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
#include "../organizer/addInManager.h"
#include "../organizer/algoInterfaceValidator.h"

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

namespace ito {

/*static */RetVal IOHelper::openGeneralFile(const QString &generalFileName, bool openUnknownsWithExternalApp, bool showMessages, QWidget* parent, const char* errorSlotMemberOfParent, bool globalNotLocalWorkspace /*= true*/)
{
    QFile file(generalFileName);
    ito::RetVal retval(ito::retOk);

    if (!file.exists())
    {
        retval += RetVal(retError, 1001, tr("file %1 does not exist").arg(generalFileName).toLatin1().data());
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
            retval += importPyWorkspaceVars(generalFileName, globalNotLocalWorkspace);
            goto end;
        }
        else if (suffix == "mat") //matlab file
        {
            retval += importPyWorkspaceVars(generalFileName, globalNotLocalWorkspace);
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
            retval += RetVal(retError, 1002, tr("file %1 can not be opened with this application").arg(generalFileName).toLatin1().data());
            goto end;
        }
    }

end:
    if (retval.containsWarningOrError())
    {
        if (showMessages)
        {
            QMessageBox msgBox(parent);
            if (retval.hasErrorMessage())
            {
                QString errStr = QLatin1String(retval.errorMessage());
                msgBox.setText(errStr);
            }
            else
                msgBox.setText("unknown error opening file");
            msgBox.exec();
        }
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
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
    IOFilters filters = (IOFilters) (IOHelper::IOOutput | IOHelper::IOWorkspace | IOHelper::IOPlugin);

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
            //QDir::setCurrent(info.path());
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
                return RetVal(retError, 1, tr("python engine not available").toLatin1().data());
            }

            if (eng->isPythonBusy() && !eng->isPythonDebuggingAndWaiting())
            {
                return RetVal(retError, 2, tr("variables cannot be exported since python is busy right now").toLatin1().data());
            }

            QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
            //get values from workspace
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            QSharedPointer<SharedParamBasePointerVector> values(new SharedParamBasePointerVector());
            QMetaObject::invokeMethod(eng, "getParamsFromWorkspace",Q_ARG(bool,globalNotLocal), Q_ARG(QStringList, varNames), Q_ARG(QVector<int>, compatibleParamBaseTypes), Q_ARG(QSharedPointer<SharedParamBasePointerVector>, values), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
            if (!locker.getSemaphore()->wait(5000))
            {
                retVal += RetVal(retError, 0, tr("timeout while getting value from workspace").toLatin1().data());
            }
            else
            {
                retVal += locker.getSemaphore()->returnValue;
            }
            
            if (values->size() != varNames.size())
            {
                retVal += RetVal(retError, 0, tr("the number of values returned from workspace does not correspond to requested number").toLatin1().data());
            }
            QApplication::restoreOverrideCursor();

            for (int i=0;i<varNames.size() && !retVal.containsError();i++)
            {
                retVal += uiSaveFileWithFilter((*values)[0], filename,parent);
            }

            /*if (retVal.containsError())
            {
                QString text = tr("An error occurred while saving to file.");
                if (retVal.errorMessage()) text.append("\n").append(retVal.errorMessage());
                QMessageBox::critical(parent, tr("Error while saving file"), text);
            }*/

            return retVal;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static */RetVal IOHelper::exportPyWorkspaceVars(const QString &filename, bool globalNotLocal, const QStringList &varNames)
{
    RetVal retValue(retOk);

    QFile file(filename);

    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return RetVal(retError, 3, tr("file cannot be opened").toLatin1().data());
    }
    file.close();

    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (eng == NULL)
    {
        return RetVal(retError, 1, tr("python engine not available").toLatin1().data());
    }

    if (eng->isPythonBusy() && !eng->isPythonDebuggingAndWaiting())
    {
        return RetVal(retError, 2, tr("variables cannot be exported since python is busy right now").toLatin1().data());
    }

    QFileInfo info(file);
    QString suffix = info.suffix().toLower();

    if (suffix == "idc")
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore(1));

        QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
        QMetaObject::invokeMethod(eng, "pickleVariables", Q_ARG(bool,globalNotLocal), Q_ARG(QString,filename), Q_ARG(QStringList,varNames), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        if (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginFileSaveLoad))
        {
            retValue += RetVal(retError, 2, tr("timeout while pickling variables").toLatin1().data());
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
        QMetaObject::invokeMethod(eng, "saveMatlabVariables", Q_ARG(bool,globalNotLocal), Q_ARG(QString,filename), Q_ARG(QStringList,varNames), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        if (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginFileSaveLoad))
        {
            retValue += RetVal(retError, 2, tr("timeout while saving variables to matlab file").toLatin1().data());
        }
        else
        {
            retValue += locker.getSemaphore()->returnValue;
        }

        QApplication::restoreOverrideCursor();
    }
    else
    {
        retValue += RetVal(retError, 0, tr("suffix must be *.idc or *.mat").toLatin1().data());
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static */RetVal IOHelper::uiImportPyWorkspaceVars(bool globalNotLocal, IOFilters IOfilters, QString defaultPath, QWidget* parent)
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

    IOfilters &= ~ito::IOHelper::IOOutput;
    QString filters = IOHelper::getFileFilters(IOfilters);
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
            //QDir::setCurrent(QFileInfo(filename).path());

            uiImportPyWorkspaceDefaultPath = QFileInfo(filename).canonicalPath(); //save directory as default for next call to this export dialog

            QFileInfo info(filename);
            return openGeneralFile(filename, false, true, parent, NULL, globalNotLocal);
        }
        else
        {
            return RetVal(retError, 1, tr("file not found").toLatin1().data());
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static */RetVal IOHelper::importPyWorkspaceVars(const QString &filename, bool globalNotLocal)
{
    RetVal retValue(retOk);

    QFile file(filename);

    if (!file.open(QIODevice::ReadOnly))
    {
        return RetVal(retError, 3, tr("file cannot be opened").toLatin1().data());
    }
    file.close();

    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (eng == NULL)
    {
        return RetVal(retError, 1, tr("python engine not available").toLatin1().data());
    }

    if (eng->isPythonBusy() && !eng->isPythonDebuggingAndWaiting())
    {
        return RetVal(retError, 2, tr("variables cannot be imported since python is busy right now").toLatin1().data());
    }

    QFileInfo info(file);
    QString suffix = info.suffix().toLower();

    if (suffix == "idc")
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore(1));

        QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
        QMetaObject::invokeMethod(eng, "unpickleVariables", Q_ARG(bool,globalNotLocal), Q_ARG(QString,filename), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
        if (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginFileSaveLoad))
        {
            retValue += RetVal(retError, 2, tr("timeout while unpickling variables").toLatin1().data());
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
        QMetaObject::invokeMethod(eng, "loadMatlabVariables", Q_ARG(bool,globalNotLocal), Q_ARG(QString,filename), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
        
        if (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginFileSaveLoad))
        {
            retValue += RetVal(retError, 2, tr("timeout while loading matlab variables").toLatin1().data());
        }
        else
        {
            retValue += locker.getSemaphore()->returnValue;
        }

        QApplication::restoreOverrideCursor();
    }
    else
    {
        retValue += RetVal(retError, 0, tr("suffix must be *.idc or *.mat").toLatin1().data());
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static */RetVal IOHelper::uiOpenPythonScript(QString defaultPath, QWidget* parent)
{
    QString fileName;

    if (defaultPath.isNull() || defaultPath.isEmpty()) defaultPath = QDir::currentPath();

    fileName = QFileDialog::getOpenFileName(parent, tr("open python script"), defaultPath, tr("python (*.py)"));

    QFileInfo info(fileName);

    if (fileName.isEmpty())
    {
        return RetVal(retOk);
    }
    else
    {
        QDir::setCurrent(QFileInfo(fileName).path());
        return openPythonScript(fileName);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
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
            QMetaObject::invokeMethod(seo,"openScript",Q_ARG(QString, filename),Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
            if (!locker.getSemaphore()->wait(20000))
            {
                retValue += RetVal(retError, 2, tr("timeout while opening script").toLatin1().data());
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
/*static */RetVal IOHelper::openUIFile(const QString &filename, QWidget* parent, const char* errorSlotMemberOfParent)
{
    ProcessOrganizer *po = qobject_cast<ProcessOrganizer*>(AppManagement::getProcessOrganizer());
    if (po)
    {
        bool existingProcess = false;
        QProcess *process = po->getProcess("designer", true, existingProcess, false);

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
                process = po->getProcess("designer", false, existingProcess, false);
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

                if (errorSlotMemberOfParent != NULL)
                {
                    connect(process, SIGNAL(error(QProcess::ProcessError)), parent, errorSlotMemberOfParent);
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

            if (errorSlotMemberOfParent != NULL)
            {
                connect(process, SIGNAL(error(QProcess::ProcessError)), parent, errorSlotMemberOfParent);
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

                    dialog = new DialogOpenFileWithFilter(filename, filter, autoMand, autoOut, userMand, userOpt, retval, parent );
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

                    dialog = new DialogOpenFileWithFilter(filename, filter, autoMand, autoOut, userMand, userOpt, retval, parent );
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

                    dialog = new DialogOpenFileWithFilter(filename, filter, autoMand, autoOut, userMand, userOpt, retval, parent);
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
                        
                QMetaObject::invokeMethod(pyEng, "putParamsToWorkspace", Q_ARG(bool,globalNotLocal), Q_ARG(QStringList, pythonVarNames), Q_ARG(QVector<SharedParamBasePointer>, values), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
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
        case ito::ParamBase::DObjPtr & ito::paramTypeMask:
            filters = AIM->getFilterByInterface(ito::AddInAlgo::iWriteDataObject, suffix);
            break;
        case ito::ParamBase::PointCloudPtr & ito::paramTypeMask:
            filters = AIM->getFilterByInterface(ito::AddInAlgo::iWritePointCloud, suffix);
            break;
        case ito::ParamBase::PolygonMeshPtr & ito::paramTypeMask:
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
                    retval += ito::RetVal(retError, 0, tr("algorithm interface not supported").toLatin1().data());
                }

                if (!retval.containsError())
                {
                    userOpt = fp->paramsOpt;
                    userMand = fp->paramsMand.mid(autoMand.size());

                    if (userMand.size() > 0 || userOpt.size() > 0)
                    {
                        DialogSaveFileWithFilter *dialog = new DialogSaveFileWithFilter(filename, filter, autoMand, autoOut, userMand, userOpt, parent);
                        if (dialog->exec() == QDialog::Accepted)
                        {
                            QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
                            dialog->getParameters(paramsMand, paramsOpt);
                            paramsMand = autoMand + paramsMand;
                            retval += filter->m_filterFunc(&paramsMand, &paramsOpt, &autoOut);
                            QApplication::restoreOverrideCursor();
                        }

                        DELETE_AND_SET_NULL(dialog);
                    }
                    else
                    {
                        QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
                        retval += filter->m_filterFunc(&autoMand, &paramsOpt, &autoOut);
                        QApplication::restoreOverrideCursor();
                    }
                }
            }
            else
            {
                retval += RetVal(retError, 0, tr("error while getting mand and out parameters from algorithm interface").toLatin1().data());
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
        filter << tr("Python Scripts (*.py)") << tr("Itom Data Collection (*.idc)") << tr("Matlab Matrix (*.mat)") << tr("User Interfaces (*.ui)");
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
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iReadDataObject, QString::Null());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iReadDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
                if (IOfilters.testFlag(ito::IOHelper::IOMimePointCloud))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iReadPointCloud, QString::Null());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iReadDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
                if (IOfilters.testFlag(ito::IOHelper::IOMimePolygonMesh))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iReadPolygonMesh, QString::Null());
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
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iWriteDataObject, QString::Null());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iWriteDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
                if (IOfilters.testFlag(ito::IOHelper::IOMimePointCloud))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iWritePointCloud, QString::Null());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iReadDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
                if (IOfilters.testFlag(ito::IOHelper::IOMimePolygonMesh))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iWritePolygonMesh, QString::Null());
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
    QRegExp reg("^.*\\((.*)\\)$");
    QStringList _allPatterns;

    foreach(const QString &item, filter)
    {
        if( reg.indexIn(item) >= 0 )
        {
            _allPatterns.append( reg.cap(1).trimmed().split(" ") );
        }
    }

    _allPatterns.removeDuplicates();

    filter << tr("Itom Files (%1)").arg(_allPatterns.join(" "));

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
    QRegExp reg;
    reg.setPatternSyntax( QRegExp::Wildcard );

    foreach(const QString &pat, allPatterns)
    {
        reg.setPattern(pat);
        if(reg.exactMatch(filename))
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
    int width = fontm.width(path);
    if (width > pixelLength)
    {
        bool end = false;
        while(width > pixelLength - fontm.width("...") && end == false)
        {
            int index = path.indexOf(QDir::separator(), 0)+1;
            if (index == 0 || index == path.lastIndexOf(QDir::separator()))
            {
                end = true;
            }
            path.remove(index, 1);
            width = fontm.width(path);
        }
        path.insert(path.indexOf(QDir::separator(),0)+1, "...");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ QIcon IOHelper::searchIcon(const QString &filename, SearchFolders searchFolders /*= SFAll*/, const QIcon &fallbackIcon /*= QIcon()*/)
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
        dir = QCoreApplication::applicationDirPath();
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
        dir = QCoreApplication::applicationDirPath();
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

} //end namespace ito
