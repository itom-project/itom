
#ifndef ITOM_LOG_H
#define ITOM_LOG_H


#include "retVal.h"
#include <qfile.h>
#include <qobject.h>
#include <qmutex.h>
#include <qtextstream.h>
#include <qvector.h>


// only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib
//#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)


namespace ito {


class ITOMCOMMONQT_EXPORT Logger : QObject
{
    Q_OBJECT

public:
    Logger(QString logFile);
    ~Logger();

public slots:
    void writePythonLog(QString msg);

private:
    static bool s_handlerRegistered;
    static QVector<Logger*> s_instances;
    QFile m_logFile;
    QTextStream* m_messageStream;
    QMutex m_msgOutputProtection;

    static void s_messageHandler(
        QtMsgType type, const QMessageLogContext& context, const QString& msg);

    void handleMessage(QtMsgType type, const QMessageLogContext& context, const QString& msg);
};


} // namespace ito


#endif // ITOM_LOG_H
