/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2015, Institut für Technische Optik (ITO),
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

//import this before any qobject stuff
#include "../python/pythonEngine.h"

#include "pipManager.h"

#include "../../common/sharedStructures.h"
#include "../AppManagement.h"
#include <qdir.h>

namespace ito 
{

//----------------------------------------------------------------------------------------------------------------------------------
/** constructor
*
*   contructor, creating column headers for the tree view
*/
PipManager::PipManager(QObject *parent /*= 0*/) :
    QAbstractItemModel(parent),
    m_currentTask(taskNo),
    m_pipAvailable(false),
    m_pipVersion(0x000000)
{
    m_headers << tr("Name") << tr("Version") << tr("Location") << tr("Requires") << tr("Updates") << tr("Summary") << tr("Homepage") << tr("License");
    m_alignment << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft);

    connect(&m_pipProcess, SIGNAL(error(QProcess::ProcessError)), this, SLOT(processError(QProcess::ProcessError)));
    connect(&m_pipProcess, SIGNAL(finished(int,QProcess::ExitStatus)), this, SLOT(processFinished(int,QProcess::ExitStatus)));
    connect(&m_pipProcess, SIGNAL(readyReadStandardError()), this, SLOT(processReadyReadStandardError()));
    connect(&m_pipProcess, SIGNAL(readyReadStandardOutput()), this, SLOT(processReadyReadStandardOutput()));

    const PythonEngine *pyeng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (pyeng)
    {
        m_pythonPath = pyeng->getPythonPathPrefix();

        if (m_pythonPath != "")
        {
            QDir pythonPath(m_pythonPath);
            if (pythonPath.exists())
            {
#ifdef WIN32
                m_pythonPath = pythonPath.absoluteFilePath("python.exe");
#endif
            }
            else
            {
                m_pythonPath = "";
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/** destructor - clean up, clear header and alignment list
*
*/
PipManager::~PipManager()
{
    if (m_pipProcess.state() == QProcess::Running || m_pipProcess.state() == QProcess::Starting)
    {
        m_pipProcess.kill();
        m_pipProcess.waitForFinished(2000);
    }
}
//----------------------------------------------------------------------------------------------------------------------------------
/** return parent element
*   @param [in] index   the element's index for which the parent should be returned
*   @return     the parent element. 
*
*/
QModelIndex PipManager::parent(const QModelIndex &index) const
{
    return QModelIndex();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return number of rows
*   @param [in] parent parent of current item
*   @return     returns number of users
*/
int PipManager::rowCount(const QModelIndex &parent) const
{
    return m_pythonPackages.length();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return the header / captions for the tree view model
*
*/
QVariant PipManager::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role == Qt::DisplayRole && orientation == Qt::Horizontal )
    {
        if (section >= 0 && section < m_headers.size())
        {
            return m_headers.at(section);
        }
        return QVariant();
    }
    return QVariant();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return data elements for a given row
*   @param [in] index   index for which the data elements should be delivered
*   @param [in] role    the current role of the model 
*   @return data of the selected element, depending on the element's row and column (passed in index.row and index.column)
*
*/
QVariant PipManager::data(const QModelIndex &index, int role) const
{
    if(!index.isValid())
    {
        return QVariant();
    }
 
    if(role == Qt::DisplayRole || role == Qt::ToolTipRole)
    {
        const PythonPackage &package = m_pythonPackages[index.row()];
        switch (index.column())
        {
            case 0:
                return package.m_name;
            case 1:
                return package.m_version;
            case 2:
                return package.m_location;
            case 3:
                return package.m_requires;
            case 4:
                {
                    if (package.m_status == PythonPackage::Uptodate)
                    {
                        return tr("up to date");
                    }
                    else if (package.m_status == PythonPackage::Outdated)
                    {
                        return tr("new version %1 available").arg(package.m_newVersion);
                    }
                    else
                    {
                        return tr("unknown");
                    }
                }
            case 5:
                return package.m_summary;
            case 6:
                return package.m_homepage;
            case 7:
                return package.m_license;
            default:
                return QVariant();
        }
    }
    
    return QVariant();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return column count
*   @param [in] parent parent of current item
*   @return     2 for child elements (instances) and the header size for root elements (plugins)
*/
int PipManager::columnCount(const QModelIndex & /*parent*/) const
{
    return m_headers.size();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return current index element
*   @param [in] row row of current element
*   @param [in] column column of current element
*   @param [in] parent  parent of current element
*   @return QModelIndex - element at current index
*
*   returns the passed row as index, as the users are arranged in a simple one dimensional list
*/
QModelIndex PipManager::index(int row, int column, const QModelIndex &parent) const
{
    if(parent.isValid() || row < 0 || row >= m_pythonPackages.length() || column < 0 || column >= m_headers.size())
    {
        return QModelIndex();
    }
    
    return createIndex(row, column);
}

//----------------------------------------------------------------------------------------------------------------------------------
bool PipManager::isPipStarted() const
{
    return m_pipProcess.pid() != 0;
}


//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::checkPipAvailable(const PipGeneralOptions &options /*= PipGeneralOptions()*/)
{
    if (m_pythonPath == "")
    {
        emit outputAvailable("Python is not available\n", false);
        return;
    }

    if (m_currentTask == taskNo)
    {
        emit pipRequestStarted(taskCheckAvailable, "Check connection to pip and get version...\n");
        clearBuffers();
        m_currentTask = taskCheckAvailable;

        QStringList arguments;
        arguments << "-m" << "pip" << "-V";
        arguments << parseGeneralOptions(options, true);
        m_pipProcess.start(m_pythonPath, arguments);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::listAvailablePackages(const PipGeneralOptions &options /*= PipGeneralOptions()*/)
{
    if (m_pythonPath == "")
    {
        emit outputAvailable("Python is not available\n", false);
        return;
    }

    if (m_pipAvailable == false)
    {
        emit outputAvailable("pip is not available\n", false);
        return;
    }


    //list consists of two steps:
    //1. get package names using list
    //2. get more information using show package1 package2 ...
    if (m_currentTask == taskNo)
    {
        emit pipRequestStarted(taskListPackages1, "Get list of installed packages... (step 1)\n", true);
        clearBuffers();
        m_currentTask = taskListPackages1;
        m_generalOptionsCache = options;

        QStringList arguments;
        arguments << "-m" << "pip" << "list";
        arguments << parseGeneralOptions(options);
        m_pipProcess.start(m_pythonPath, arguments);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::listAvailablePackages2(const QStringList &names)
{
    if (m_pythonPath == "")
    {
        emit outputAvailable("Python is not available\n", false);
        return;
    }

    if (m_pipAvailable == false)
    {
        emit outputAvailable("pip is not available\n", false);
        return;
    }

    //list consists of two steps:
    //1. get package names using freeze
    //2. get more information using show package1 package2 ...
    if (m_currentTask == taskNo)
    {
        emit pipRequestStarted(taskListPackages2, "Get list of installed packages... (step 2)\n", true);
        clearBuffers();
        m_currentTask = taskListPackages2;

        QStringList arguments;
        arguments << "-m" << "pip" << "show" << names;
        arguments << parseGeneralOptions(m_generalOptionsCache);
        m_pipProcess.start(m_pythonPath, arguments);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::checkPackageUpdates(const PipGeneralOptions &options /*= PipGeneralOptions()*/)
{
    if (m_pythonPath == "")
    {
        emit outputAvailable("Python is not available\n", false);
        return;
    }

    if (m_pipAvailable == false)
    {
        emit outputAvailable("pip is not available\n", false);
        return;
    }

    if (m_currentTask == taskNo)
    {
        emit pipRequestStarted(taskCheckUpdates, "Check online (pypi.python.org) if newer versions of packages are available...\n");
        clearBuffers();
        m_currentTask = taskCheckUpdates;

        QStringList arguments;
        arguments << "-m" << "pip" << "list" << "--outdated";
        arguments << parseGeneralOptions(options);
        m_pipProcess.start(m_pythonPath, arguments);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::installPackage(const PipInstall &installSettings, const PipGeneralOptions &options /*= PipGeneralOptions()*/)
{
    if (m_pythonPath == "")
    {
        emit outputAvailable("Python is not available\n", false);
        return;
    }

    if (m_pipAvailable == false)
    {
        emit outputAvailable("pip is not available\n", false);
        return;
    }

    if (m_currentTask == taskNo)
    {
        emit pipRequestStarted(taskInstall, "Install package...\n");
        clearBuffers();
        m_currentTask = taskInstall;

        QStringList arguments;
        arguments << "-m" << "pip" << "install";

        if (installSettings.upgrade)
        {
            arguments << "--upgrade";
        }
        if (!installSettings.installDeps)
        {
            arguments << "--no-deps";
        }
        if (installSettings.ignoreIndex)
        {
            arguments << "--no-index";
        }
        if (installSettings.findLinks != "")
        {
            arguments << "--find-links" << installSettings.findLinks;
        }

        if (installSettings.type == PipInstall::typeWhl)
        {
            arguments << "--use-wheel";
        }
        else
        {
            arguments << "--no-use-wheel";
        }

        arguments << parseGeneralOptions(options);

        arguments << installSettings.packageName;
        m_pipProcess.start(m_pythonPath, arguments);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::uninstallPackage(const QString &packageName, const PipGeneralOptions &options /*= PipGeneralOptions()*/)
{
    if (m_pythonPath == "")
    {
        emit outputAvailable("Python is not available\n", false);
        return;
    }

    if (m_pipAvailable == false)
    {
        emit outputAvailable("pip is not available\n", false);
        return;
    }

    if (m_currentTask == taskNo)
    {
        emit pipRequestStarted(taskUninstall, QString("Uninstall package %1...\n").arg(packageName));
        clearBuffers();
        m_currentTask = taskUninstall;

        QStringList arguments;
        arguments << "-m" << "pip" << "uninstall" << "--yes";

        arguments << parseGeneralOptions(options);

        arguments << packageName;
        m_pipProcess.start(m_pythonPath, arguments);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::processError(QProcess::ProcessError error)
{
    if (m_currentTask != taskNo)
    {
        switch (error)
        {
        case QProcess::FailedToStart:
            emit outputAvailable(tr("Could not start python pip\n"), false);
            break;
        case QProcess::ReadError:
            emit outputAvailable(tr("An error occurred when attempting to read from the process.\n"), false);
            break;
        default:
            emit outputAvailable(tr("other error"), false);
            break;
        }
    }

    finalizeTask();
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::processFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (exitStatus == QProcess::CrashExit)
    {
        Task temp = m_currentTask;
        m_currentTask = taskNo;

        emit pipRequestFinished(temp, tr("Python pip crashed during execution\n"), false);

        if (temp != taskNo)
        {
            processReadyReadStandardError();
            processReadyReadStandardOutput();
        }

        clearBuffers();
    }
    else
    {
        finalizeTask();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::processReadyReadStandardError()
{
    QByteArray str = m_pipProcess.readAllStandardError();
    if (str.length() > 0)
    {
        m_standardErrorBuffer += str;
        emit outputAvailable(str, false);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::processReadyReadStandardOutput()
{
    QByteArray str = m_pipProcess.readAllStandardOutput();
    if (str.length() > 0)
    {
        m_standardOutputBuffer += str;
        emit outputAvailable(str, true);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::finalizeTask()
{
    Task temp = m_currentTask;
    m_currentTask = taskNo;

    if (temp != taskNo)
    {
        processReadyReadStandardError();
        processReadyReadStandardOutput();

        QString error = m_standardErrorBuffer;
        QString output = m_standardOutputBuffer;

        if (temp == taskCheckAvailable)
        {
            QRegExp reg("pip ((\\d+)\\.(\\d+)\\.(\\d+)) from(.*)");
            if (reg.indexIn(output) != -1)
            {
                m_pipVersion = CREATEVERSION(reg.cap(2).toInt(), reg.cap(3).toInt(), reg.cap(4).toInt());
                QString version = reg.cap(1);
                emit pipVersion(version);
                m_pipAvailable = true;
                emit pipRequestFinished(temp, "", true);
            }
            else
            {
                m_pipAvailable = false;
                emit pipRequestFinished(temp, "Package pip is not available. Install Python pip first (see https://pip.pypa.io/en/latest/installing.html).\n", false);
            }
        }
        else if (temp == taskListPackages1)
        {
            if (error != "")
            {
                emit pipRequestFinished(temp, "Error obtaining list of packages (freeze)\n", false);
            }
            else
            {
                QStringList packages_out;
                int idx;
                QStringList packages = output.split("\n");
                foreach (const QString &p, packages)
                {
                    idx = p.indexOf(" (");
                    if (idx != -1)
                    {
                        packages_out.append(p.left(idx));
                    }
                }

                if (packages_out.length() > 0)
                {
                    listAvailablePackages2(packages_out);
                }
            }
        }
        else if (temp == taskListPackages2)
        {
            if (error != "")
            {
                emit pipRequestFinished(temp, "Error obtaining list of packages (show)\n", false);
            }
            else
            {
                beginResetModel();
                m_pythonPackages.clear();

                QStringList lines = output.split("\r\n");
                if (lines.length() == 0)
                {
                    lines = output.split("\n");
                }

                PythonPackage package;
                bool package_started = false;
                int pos;
                QString key, value;
                QStringList keys;
                keys << "Name" << "Version" << "Summary" << "Home-page" << "License" << "Location" << "Requires";

                foreach (const QString &line, lines)
                {
                    if (line == "---")
                    {
                        if (package_started)
                        {
                            m_pythonPackages << package;
                        }

                        package_started = true;
                        package = PythonPackage(); //start new, empty package structure
                    }
                    else if (line != "")
                    {
                        //check if line consists of key: value
                        pos = line.indexOf(": ");
                        if (pos != -1)
                        {
                            key = line.left(pos);
                            value = line.mid(pos+2);

                            switch (keys.indexOf(key))
                            {
                            case 0: //Name
                                package.m_name = value;
                                break;
                            case 1: //Version
                                package.m_version = value;
                                break;
                            case 2: //Summary
                                package.m_summary = value;
                                break;
                            case 3: //Home-page
                                package.m_homepage = value;
                                break;
                            case 4: //License
                                package.m_license = value;
                                break;
                            case 5: //Location
                                package.m_location = value;
                                break;
                            case 6: //Requires
                                package.m_requires = value;
                                break;
                            }

                        }
                    }
                }

                if (package_started)
                {
                    m_pythonPackages << package;
                }

                endResetModel();
                emit pipRequestFinished(temp, "List of packages obtained.\n", true);
            }
        }
        else if (temp == taskCheckUpdates)
        {
            if (error != "")
            {
                emit pipRequestFinished(temp, "Error obtaining list of outdated packages (list)\n", false);
            }
            else
            {
                QRegExp rx("(\\S+) \\(Current: (\\S)+ Latest: (\\S+) (\\[\\S+\\])\\)");
                int pos = 0;
                QMap<QString,QString> outdated;

                while ((pos = rx.indexIn(output, pos)) != -1)
                {
                    outdated[rx.cap(1)] = rx.cap(3);
                    pos += rx.matchedLength();
                }

                for (int i = 0; i < m_pythonPackages.length(); ++i)
                {
                    if (outdated.contains(m_pythonPackages[i].m_name))
                    {
                        m_pythonPackages[i].m_newVersion = outdated[m_pythonPackages[i].m_name];
                        m_pythonPackages[i].m_status = PythonPackage::Outdated;
                    }
                    else
                    {
                        m_pythonPackages[i].m_status = PythonPackage::Uptodate;
                    }
                }

                emit dataChanged(createIndex(0,4), createIndex(m_pythonPackages.length()-1, 4));

                emit pipRequestFinished(temp, "Packages checked.\n", true);
            }
        }
        else if (temp == taskInstall)
        {
            if (error != "")
            {
                emit pipRequestFinished(temp, "Error installing package\n", false);
            }
            else
            {
                listAvailablePackages();
            }
        }
        else if (temp == taskUninstall)
        {
            if (error != "")
            {
                emit pipRequestFinished(temp, "Error uninstalling package\n", false);
            }
            else
            {
                listAvailablePackages();
            }
        }
    }

    clearBuffers();
}

//-----------------------------------------------------------------------------------------
QStringList PipManager::parseGeneralOptions(const PipGeneralOptions &options, bool ignoreRetries /*= false*/) const
{
    QStringList output;

    if (options.isolated)
    {   
        output << "--isolated";
    }

    if (options.logPath != "")
    {
        output << "--log" << QString("\"%1\"").arg(options.logPath);
    }

    if (options.proxy != "")
    {
        output << "--proxy" << options.proxy;
    }

    if (options.timeout >= 0)
    {
        output << "--timeout" << QString("%1").arg(options.timeout);
    }

    if (options.retries > 0 && !ignoreRetries && MAJORVERSION(m_pipVersion) >= 6)
    {
        output << "--retries" << QString("%1").arg(options.retries);
    }
    
    return output;
}

//-----------------------------------------------------------------------------------------
void PipManager::clearBuffers()
{
    m_standardOutputBuffer.clear();
    m_standardErrorBuffer.clear();
}

//-----------------------------------------------------------------------------------------
void PipManager::interruptPipProcess()
{
    if (m_pipProcess.state() == QProcess::Running || m_pipProcess.state() == QProcess::Starting)
    {
        m_pipProcess.kill();
    }
}

//-----------------------------------------------------------------------------------------
bool PipManager::isPackageInUseByOther(const QModelIndex &index)
{
    if (!index.isValid() || index.row() < 0 || index.row() >= m_pythonPackages.size())
    {
        return false;
    }

    QString other = m_pythonPackages[index.row()].m_name;
    QStringList requires;
    foreach (const PythonPackage &pp, m_pythonPackages)
    {
        requires = pp.m_requires.split(", ");
        if (requires.contains(other, Qt::CaseInsensitive))
        {
            return true;
        }
    }

    return false;
}

} //end namespace ito