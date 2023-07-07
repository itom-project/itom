
#ifndef ITOM_LOG_H
#define ITOM_LOG_H


#include "retVal.h"

#include <qfile.h>
#include <qmutex.h>
#include <qobject.h>
#include <qtextstream.h>
#include <qvector.h>


namespace ito {


/**
 * @class Logger
 * @brief Writes a log file.
 *
 * The logger redirects qDebug messages to a given log file and offers a method to write messages
 * from python to the same file. It offers the possibility for log rotation where a new file is
 * opened on instantiation if the old file is bigger than a given size. The number of old log files
 * to be kept is configurable.
 */
class ITOMCOMMONQT_EXPORT Logger : QObject
{
    Q_OBJECT

public:
    /**
     * @brief Construct a new Logger object
     *
     * @param logFile the log will be written to this file
     * @param fileSizeBytes the file size in bytes above which a new file will be created
     * @param backupCount the number of old log files to be kept
     */
    Logger(QString logFile, int fileSizeBytes = 0, int backupCount = 0);
    ~Logger();

public slots:
    /**
     * @brief Writes a message from python to the log
     *
     * The message will be prepended with [Python  <date> ].
     *
     * @param msg the message to be written, can contain newline characters
     */
    void writePythonLog(QString msg);

private:
    static bool s_handlerRegistered;
    static QVector<Logger*> s_instances;
    QFile m_logFile;
    QTextStream* m_messageStream;
    QMutex m_msgOutputProtection;

    static void s_messageHandler(
        QtMsgType type, const QMessageLogContext& context, const QString& msg);

    void initFiles(int fileSize, int backupCount);
    void deleteOldBackups(int backupCount);
    void storeBackupFile();
    void handleMessage(QtMsgType type, const QMessageLogContext& context, const QString& msg);
};


} // namespace ito


#endif // ITOM_LOG_H
