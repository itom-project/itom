#include "../itomLog.h"
#include <qdatetime.h>


namespace ito {


bool Logger::s_handlerRegistered = false;
QVector<Logger*> Logger::s_instances = QVector<Logger*>();


Logger::Logger(QString logFile)
{
    this->m_logFile.setFileName(logFile);
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

void Logger::s_messageHandler(QtMsgType type, const QMessageLogContext& context, const QString& msg)
{
    QVectorIterator<Logger*> i(s_instances);
    while (i.hasNext())
    {
        i.next()->handleMessage(type, context, msg);
    }
}

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
