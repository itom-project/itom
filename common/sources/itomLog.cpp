#include "../itomLog.h"
#include <qdatetime.h>
#include <qdir.h>
#include <qstandardpaths.h>

#ifdef Q_OS_WIN
#include <qt_windows.h>
#endif


namespace ito {


bool Logger::s_handlerRegistered = false;
QVector<Logger*> Logger::s_instances = QVector<Logger*>();


//----------------------------------------------------------------------------------------------------------------------------------
Logger::Logger(QString logFileName, QString logFileDir, int fileSizeBytes, int backupCount)
{
    if (logFileDir.isEmpty())
    {
        logFileDir = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
        logFileDir += "/qitom/log";
    }
    this->m_logFile.setFileName(logFileDir + "/" + logFileName);
    this->initFiles(fileSizeBytes, backupCount);
    this->m_logFile.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text);

    // first lines in log file
    this->m_logFile.write("------------------------------------------------------------------------------------------\n");
    this->m_logFile.write(
        QString(QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") + " Starting itom... \n")
            .toLatin1()
            .constData());
    this->m_logFile.write("------------------------------------------------------------------------------------------\n");

    this->m_messageStream = new QTextStream(&this->m_logFile);
    if (!s_handlerRegistered)
    {
        qInstallMessageHandler(this->s_messageHandler);
        s_handlerRegistered = true;
    }

    s_instances.append(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
Logger::~Logger()
{
    if (this->m_messageStream != nullptr)
    {
        delete this->m_messageStream;
        this->m_messageStream = nullptr;
    }
    this->m_logFile.close();

    s_instances.remove(s_instances.indexOf(this));
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal Logger::copyLog(QString directory, ItomSharedSemaphore* waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    RetVal retVal(ito::retOk);

    QFileInfoList list = this->listBackups();
    list.append(QFileInfo(this->m_logFile.fileName()));
    QListIterator<QFileInfo> i(list);
    while (i.hasNext())
    {
        QFileInfo info = i.next();
        QFile file(info.absoluteFilePath());
        QString targetFileName = directory + "/" + info.fileName();
        if (QFile::exists(targetFileName))
        {
            retVal += ito::RetVal(
                ito::retError,
                0,
                tr("The file already exists: %1").arg(targetFileName).toLatin1().data());
            continue;
        }
        bool success = file.copy(directory + "/" + info.fileName());
        if (!success)
        {
            retVal += ito::RetVal(
                ito::retError,
                0,
                tr("The file could not be copied to: %1").arg(targetFileName).toLatin1().data());
        }
    }
    if (waitCond)
    {
        waitCond->returnValue = retVal;
        waitCond->release();
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
 * @brief a static message handler to be registered with qInstallMessageHandler
 *
 * qInstallMessageHandler can only register a static method so this is used to call handleMessage on
 * every Logger instance.
 *
 * Under Windows, the message is additionally sent to the debugger (e.g. the output window
 * of Visual Studio), if a debugger is currently attached. This is necessary, since this
 * message handler replaces the default handler of Qt, that would do this on its own.
 */
void Logger::s_messageHandler(QtMsgType type, const QMessageLogContext& context, const QString& msg)
{
    s_outputToDebugger(type, context, msg);

    QVectorIterator<Logger*> i(s_instances);
    while (i.hasNext())
    {
        i.next()->handleMessage(type, context, msg);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
 * @brief sends the given message to an attached debugger (Windows only).
 *
 * The message is formatted as 'file(line): [type] message', such that a double click
 * on the entry in the output window of Visual Studio jumps to the corresponding
 * position in the source code. An user defined QT_MESSAGE_PATTERN is respected.
 *
 * This method does nothing on other operating systems or if no debugger is attached.
 */
void Logger::s_outputToDebugger(
    QtMsgType type, const QMessageLogContext& context, const QString& msg)
{
#ifdef Q_OS_WIN
    if (!IsDebuggerPresent())
    {
        // avoid any overhead if itom is not started from a debugger
        return;
    }

    QString text;

    if (qEnvironmentVariableIsSet("QT_MESSAGE_PATTERN"))
    {
        // the user defined an own pattern, that should be respected.
        text = qFormatLogMessage(type, context, msg);
    }
    else
    {
        QString typeName;

        switch (type)
        {
        case QtDebugMsg:
            typeName = "qDebug";
            break;
        case QtWarningMsg:
            typeName = "qWarning";
            break;
        case QtCriticalMsg:
            typeName = "qCritical";
            break;
        case QtFatalMsg:
            typeName = "qFatal";
            break;
        default:
            typeName = "qInfo";
            break;
        }

        if (context.file)
        {
            // this format allows Visual Studio to jump to the source code position
            // if the line in the output window is double-clicked.
            text = QString("%1(%2): [%3] %4")
                       .arg(context.file)
                       .arg(context.line)
                       .arg(typeName, msg);
        }
        else
        {
            text = QString("[%1] %2").arg(typeName, msg);
        }
    }

    text += "\n";

    OutputDebugStringW(reinterpret_cast<const wchar_t*>(text.utf16()));
#else
    Q_UNUSED(type)
    Q_UNUSED(context)
    Q_UNUSED(msg)
#endif
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
 * @brief Initializes the log files
 *
 * Prepares m_logFile for usage by performing a backup and deleting old backups if required.
 *
 * @param fileSizeBytes the file size in bytes above which a new file will be created
 * @param backupCount the number of old log files to be kept
 */
void Logger::initFiles(int fileSizeBytes, int backupCount)
{
    QFileInfo info(this->m_logFile.fileName());
    QDir dir = info.absoluteDir();
    this->m_logFile.setFileName(info.absoluteFilePath()); // ensure that the path is absolute
    if (!dir.exists())
    {
        dir.mkpath(dir.absolutePath());
        return; // there is no old log file yet
    }
    if (!this->m_logFile.exists() || fileSizeBytes < 1 || backupCount < 1)
    {
        return; // there is no log file or only one log file should be used
    }
    if (info.size() < fileSizeBytes)
    {
        return; // the log file is too small to start a new one
    }

    // delete one more backup file because a new one will be created afterwards
    this->deleteOldBackups(backupCount - 1);
    this->storeBackupFile();
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
 * @brief Lists all existing backup files whit the form <logFile>_<date>.<suffix>
 */
QFileInfoList Logger::listBackups()
{
    QFileInfo info(this->m_logFile.fileName());
    QDir dir = info.absoluteDir();
    QString test = info.absolutePath();

    dir.setFilter(QDir::Files | QDir::Hidden | QDir::NoSymLinks);
    QStringList nameFilters;
    nameFilters << info.baseName() + "_????_??_??__??_??_??." + info.suffix();
    dir.setNameFilters(nameFilters);
    dir.setSorting(QDir::Name); // sorting by name means sorting by date with this file names

    return dir.entryInfoList();
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
 * @brief Deletes the excess backup files
 *
 * @param backupCount the number of backup files to be kept
 */
void Logger::deleteOldBackups(int backupCount)
{
    QFileInfoList list = this->listBackups();
    if (list.length() < backupCount)
    {
        return; // no backups have to be deleted
    }

    for (int i = 0; i < (list.size() - backupCount); ++i)
    {
        QFile file(list[i].absoluteFilePath());
        file.remove();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
 * @brief Creates a backup of m_logFile.
 *
 * m_logFile is renamed by prepending the current date and afterward m_log file is set to the old
 * name again to create a new log file.
 */
void Logger::storeBackupFile()
{
    QFileInfo info(this->m_logFile.fileName());
    QDir dir = info.absoluteDir();
    QString backupName =
        QString(dir.absolutePath() + "/" + info.baseName() + "_%1__%2." + info.suffix())
            .arg(QDate::currentDate().toString("yyyy_MM_dd"))
            .arg(QTime::currentTime().toString("hh_mm_ss"));
    this->m_logFile.rename(backupName);
    this->m_logFile.setFileName(info.absoluteFilePath());
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
 * @brief Handles a qDebug message
 *
 * The incoming message will be parsed and stored to the log file in a specific format.
 */
void Logger::handleMessage(QtMsgType type, const QMessageLogContext& context, const QString& msg)
{
    this->m_msgOutputProtection.lock();

    switch (type)
    {
    case QtDebugMsg:
        (*this->m_messageStream) << "[qDebug    "
                                 << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss")
                                 << "] - " << msg << "     (File: " << context.file
                                 << " Line: " << context.line << " Function: " << context.function
                                 << ")\n";
        break;
    case QtWarningMsg:
        (*this->m_messageStream) << "[qWarning  "
                                 << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss")
                                 << "] - " << msg << "     (File: " << context.file
                                 << " Line: " << context.line << " Function: " << context.function
                                 << ")\n";
        break;
    case QtCriticalMsg:
        (*this->m_messageStream) << "[qCritical "
                                 << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss")
                                 << "] - " << msg << "     (File: " << context.file
                                 << " Line: " << context.line << " Function: " << context.function
                                 << ")\n";
        break;
    case QtFatalMsg:
        (*this->m_messageStream) << "[qFatal    "
                                 << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss")
                                 << "] - " << msg << "     (File: " << context.file
                                 << " Line: " << context.line << " Function: " << context.function
                                 << ")\n";
        abort();
    }

    this->m_messageStream->flush();
    this->m_msgOutputProtection.unlock();
}

//----------------------------------------------------------------------------------------------------------------------------------
void Logger::writePythonLog(QString msg)
{
    this->m_msgOutputProtection.lock();
    QString indentMsg = "    " + msg.trimmed();
    indentMsg.replace('\n', "\n    ");
    (*this->m_messageStream) << "[Python    "
                             << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "]\n"
                             << indentMsg << "\n";

    this->m_messageStream->flush();
    this->m_msgOutputProtection.unlock();
}


} // namespace ito
