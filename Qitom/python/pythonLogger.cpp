#include "pythonLogger.h"
#include "AppManagement.h"
#include "qDebugStream.h"


namespace ito {


//-------------------------------------------------------------------------------------
PythonLogger::PythonLogger() : m_logger(nullptr)
{
}

//-------------------------------------------------------------------------------------
void PythonLogger::init()
{
    this->m_logger = qobject_cast<Logger*>(AppManagement::getLogger());
    if (!this->m_logger)
    {
        return;
    }
    QDebugStream* cErrStream = qobject_cast<QDebugStream*>(AppManagement::getCerrStream());
    if (!cErrStream)
    {
        return;
    }

    connect(
        &m_receiveStreamBufferTimer, &QTimer::timeout, this, &PythonLogger::processStreamBuffer);
    m_receiveStreamBufferTimer.setInterval(50);
    m_receiveStreamBufferTimer.start();

    connect(cErrStream, &QDebugStream::flushStream, this, &PythonLogger::receiveStream);
}

//-------------------------------------------------------------------------------------
void PythonLogger::receiveStream(QString text, ito::tStreamMessageType msgType)
{
    if (msgType != ito::tStreamMessageType::msgStreamErr)
    {
        processStreamBuffer();
    }
    else
    {
        m_receiveStreamBuffer += text;
    }
}

//-------------------------------------------------------------------------------------
void PythonLogger::processStreamBuffer()
{
    if (m_receiveStreamBuffer == "" || this->m_logger == nullptr)
    {
        return;
    }
    this->m_logger->writePythonLog(m_receiveStreamBuffer);
    m_receiveStreamBuffer = "";
}


} // namespace ito
