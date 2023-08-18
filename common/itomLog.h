
#ifndef ITOM_LOG_H
#define ITOM_LOG_H


#include "retVal.h"
#include "sharedStructuresQt.h"

#include <qfile.h>
#include <qfileinfo.h>
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
class ITOMCOMMONQT_EXPORT Logger : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief Construct a new Logger object
     *
     * @param logFileName the name of the log file
     * @param fileSizeBytes the file size in bytes above which a new file will be created
     * @param backupCount the number of old log files to be kept
     * @param logFileDir the log will be written to this directory; a user directory is used if not
     *          given
     */
    Logger(
        QString logFileName, QString logFileDir = "", int fileSizeBytes = 0, int backupCount = 0);
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

    /**
     * @brief Copies all log files to the given directory.
     */
    RetVal copyLog(QString directory, ItomSharedSemaphore* waitCond = nullptr);

private:
    static bool s_handlerRegistered;
    static QVector<Logger*> s_instances;
    QFile m_logFile;
    QTextStream* m_messageStream;
    QMutex m_msgOutputProtection;

    static void s_messageHandler(
        QtMsgType type, const QMessageLogContext& context, const QString& msg);

    void initFiles(int fileSize, int backupCount);
    QFileInfoList listBackups();
    void deleteOldBackups(int backupCount);
    void storeBackupFile();
    void handleMessage(QtMsgType type, const QMessageLogContext& context, const QString& msg);
};


} // namespace ito


#endif // ITOM_LOG_H
