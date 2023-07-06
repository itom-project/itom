#ifndef PYTHON_LOG_H
#define PYTHON_LOG_H


#include "common/typeDefs.h"
#include "common/itomLog.h"
#include <qobject.h>
#include <qtimer.h>


namespace ito {


class PythonLogger : public QObject
{
    Q_OBJECT

public:
    PythonLogger();
    void init();

private:
    QString m_receiveStreamBuffer;
    QTimer m_receiveStreamBufferTimer;
    Logger* m_logger;

    void processStreamBuffer();

private slots:
    void receiveStream(QString text, ito::tStreamMessageType msgType);
};


} // namespace ito


#endif // PYTHON_LOG_H
