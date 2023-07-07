#include "../itomLog.h"
#include <qdatetime.h>
#include <qdir.h>
#include <qfileinfo.h>


namespace ito {


bool Logger::s_handlerRegistered = false;
QVector<Logger*> Logger::s_instances = QVector<Logger*>();


//----------------------------------------------------------------------------------------------------------------------------------
Logger::Logger(QString logFile, int fileSizeBytes, int backupCount)
{
    this->m_logFile.setFileName(logFile);
    this->initFiles(fileSizeBytes, backupCount);
    this->m_logFile.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text);

    // first lines in log file
    this->m_logFile.write(
        "------------------------------------------------------------------------------------------\n");
    this->m_logFile.write(
        QString(QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") + " Starting itom... \n")
            .toLatin1()
            .constData());
    this->m_logFile.write(
        "------------------------------------------------------------------------------------------\n");

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
void Logger::s_messageHandler(QtMsgType type, const QMessageLogContext& context, const QString& msg)
{
    QVectorIterator<Logger*> i(s_instances);
    while (i.hasNext())
    {
        i.next()->handleMessage(type, context, msg);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void Logger::initFiles(int fileSizeBytes, int backupCount)
{
    QFileInfo info(this->m_logFile.fileName());
    QDir dir = info.absoluteDir();
    if (!dir.exists())
    {
        dir.mkpath(dir.absolutePath());
        return;
    }
    if (!this->m_logFile.exists() || fileSizeBytes < 1 || backupCount < 1)
    {
        return;
    }
    if (info.size() < fileSizeBytes)
    {
        return;
    }

    this->deleteOldBackups(backupCount - 1);
    this->storeBackupFile();
}

//----------------------------------------------------------------------------------------------------------------------------------
void Logger::deleteOldBackups(int backupCount)
{
    QFileInfo info(this->m_logFile.fileName());
    QDir dir = info.absoluteDir();

    // list backup files in the form <logFile>_<date>.<suffix>
    dir.setFilter(QDir::Files | QDir::Hidden | QDir::NoSymLinks);
    QStringList filters;
    filters << info.baseName() + "_????_??_??__??_??_??." + info.suffix();
    dir.setNameFilters(filters);
    dir.setSorting(QDir::Name);
    QFileInfoList list = dir.entryInfoList();

    if (list.length() < backupCount)
    {
        return;
    }

    for (int i = 0; i < (list.size() - backupCount); ++i)
    {
        QFile file(list[i].absoluteFilePath());
        file.remove();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void Logger::storeBackupFile()
{
    QFileInfo info(this->m_logFile.fileName());
    QDir dir = info.absoluteDir();
    QString backupName =
        QString(dir.absolutePath() + "/" + info.baseName() + "_%1__%2." + info.suffix())
            .arg(QDate::currentDate().toString("yyyy_MM_dd"))
            .arg(QTime::currentTime().toString("hh_mm_ss"));
    this->m_logFile.rename(backupName);
    this->m_logFile.setFileName(info.fileName());
}

//----------------------------------------------------------------------------------------------------------------------------------
void Logger::handleMessage(
    QtMsgType type, const QMessageLogContext& context, const QString& msg)
{
    this->m_msgOutputProtection.lock();

    switch (type)
    {
    case QtDebugMsg:
        (*this->m_messageStream) << "[qDebug    "
                         << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - "
                         << msg << "     (File: " << context.file << " Line: " << context.line
                         << " Function: " << context.function << ")\n";
        break;
    case QtWarningMsg:
        (*this->m_messageStream) << "[qWarning  "
                         << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - "
                         << msg << "     (File: " << context.file << " Line: " << context.line
                         << " Function: " << context.function << ")\n";
        break;
    case QtCriticalMsg:
        (*this->m_messageStream) << "[qCritical "
                         << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - "
                         << msg << "     (File: " << context.file << " Line: " << context.line
                         << " Function: " << context.function << ")\n";
        break;
    case QtFatalMsg:
        (*this->m_messageStream) << "[qFatal    "
                         << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - "
                         << msg << "     (File: " << context.file << " Line: " << context.line
                         << " Function: " << context.function << ")\n";
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
                             << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss")
                     << "]\n"
                     << indentMsg << "\n";

    this->m_messageStream->flush();
    this->m_msgOutputProtection.unlock();
}

} // namespace ito
