#ifndef PYTHON_LOG_H
#define PYTHON_LOG_H


#include "common/itomLog.h"
#include "common/typeDefs.h"

#include <qobject.h>
#include <qpointer.h>
#include <qtimer.h>


namespace ito {


/**
 * @class PythonLogger
 * @brief Copies python errors to the logger.
 *
 * This gets the logger and the python error stream from AppManagement and copies any error to
 * the logger.
 */
class PythonLogger : public QObject
{
    Q_OBJECT

public:
    PythonLogger();

    /**
     * @brief initializes the PythonLogger
     *
     * This has to be called after the logger and the python error stream are registered in the
     * AppManagement. If not logger or error stream can be retrieved nothing happens.
     */
    void init();

private:
    QString m_receiveStreamBuffer;
    QTimer m_receiveStreamBufferTimer;
    QPointer<Logger> m_logger;

    void processStreamBuffer();

private slots:
    void receiveStream(QString text, ito::tStreamMessageType msgType);
};


} // namespace ito


#endif // PYTHON_LOG_H
