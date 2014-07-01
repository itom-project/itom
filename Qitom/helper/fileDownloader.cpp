
#include "fileDownloader.h"

#include <qdebug.h>
#include <qnetworkcookiejar.h>

//-------------------------------------------------------------------------------------
FileDownloader::FileDownloader(QUrl Url, int nrOfAllowedRedirects /*= 0*/, QObject *parent) :
    QObject(parent),
    m_pCurrentNetworkReply(NULL),
    m_bytesReceived(-1),
    m_bytesTotal(-1),
    m_nrOfAllowedRedirects(nrOfAllowedRedirects)
{
    connect(&m_WebCtrl, SIGNAL(finished(QNetworkReply*)), SLOT(fileDownloaded(QNetworkReply*)));
    m_WebCtrl.setCookieJar(new QNetworkCookieJar(&m_WebCtrl));

    QNetworkRequest request(Url);
    request.setRawHeader("User-Agent", "Wget/1.12 (linux-gnu)");
    m_pCurrentNetworkReply = m_WebCtrl.get(request);

    connect(m_pCurrentNetworkReply, SIGNAL(downloadProgress(qint64, qint64)), this, SLOT(downloadProgress(qint64,qint64)));
}

//-------------------------------------------------------------------------------------
FileDownloader::~FileDownloader()
{
     if (m_pCurrentNetworkReply)
     {
         m_pCurrentNetworkReply->deleteLater();
     }
}

//-------------------------------------------------------------------------------------
//! Aboarts a download
/*!

*/
void FileDownloader::abortDownload()
{
    if (m_pCurrentNetworkReply)
    {
        m_pCurrentNetworkReply->abort();
    }
}

//-------------------------------------------------------------------------------------
//! This functions returns the downloadprogress of a file.
/*!
    The progress is returned in percent as an integer between 0 and 100.
   
    \return downloadprogress in percent.
*/
int FileDownloader::getDownloadProgress()
{
    if (m_bytesTotal > 0)
    {
        return 100.0 * (double)m_bytesReceived / m_bytesTotal;
    }
    return 0;
}

//-------------------------------------------------------------------------------------
//! Returns the status of a download.
/*!
    The status is returned as a Filedownloader::Status. This enumeration
    has different states: sRunning, sAborted, sFinished, sError
    This function also handles redirects!

    \param errorMsg The function stores the message in Qstring thst it can be displayed
    \return statusmessage of the downlaod
*/
FileDownloader::Status FileDownloader::getStatus(QString &errorMsg)
{
    Status status;
    if (m_pCurrentNetworkReply)
    {
        if (m_pCurrentNetworkReply->error() != QNetworkReply::NoError)
        {
            status = sError;
            errorMsg = m_pCurrentNetworkReply->errorString();
        }
        else if (m_pCurrentNetworkReply->isRunning())
        {
            status = sRunning;
        }
        else if (m_pCurrentNetworkReply->isFinished())
        {
            // finished... but check for redirect
            int redirect = checkRedirect(errorMsg);
            if (redirect == 1) //redirection started
            {
                status = sRunning;
            }
            else if (redirect == 0)
            {
                qDebug() << m_pCurrentNetworkReply->error();
                status = sFinished;
            }
            else //error in redirect
            {
                status = sError;
            }
        }
    }
    else
    {
        errorMsg = tr("no network reply instance available");
        status = sError;
    }

    return status;
}

//-------------------------------------------------------------------------------------
//! Setting method to set the meber variables.
/*!
    \param bytesReceived received bytes
    \param bytesTotal total size of the downloading file bytes
    
    \sa getDownloadProgress
*/
void FileDownloader::downloadProgress(qint64 bytesReceived, qint64 bytesTotal)
{
    m_bytesReceived = bytesReceived;
    m_bytesTotal = bytesTotal;
}

//-------------------------------------------------------------------------------------
//! This function is called when the download is finished. 
/*!
    It stores the downloaded data in a member variable and 

    \param pReply
    
*/
void FileDownloader::fileDownloaded(QNetworkReply* pReply)
{
    if (pReply->error() == QNetworkReply::NoError)
    {
        m_DownloadedData = pReply->readAll();
        QString msg;
        int t = checkRedirect(msg);
    }
    else
    {
       int bp = 0; // Error occured
    }
}

//-------------------------------------------------------------------------------------
//! This function returns the downloaded data. 
/*! 

    \return returns downloaded data as a QByteArray.
*/
QByteArray FileDownloader::downloadedData() const
{
    return m_DownloadedData;
}

//-------------------------------------------------------------------------------------
//! This function checks if the given error message is caused by a redirect 
/*! 
    \param errorMsg
    \return returns status of redirection. 0 = ok (no redirection, 1 = redirection, -1 = number of redirections exceeded
*/
int FileDownloader::checkRedirect(QString &errorMsg)
{
    if (m_pCurrentNetworkReply)
    {
        int i = m_pCurrentNetworkReply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        QUrl redirect = m_pCurrentNetworkReply->attribute(QNetworkRequest::RedirectionTargetAttribute).toUrl();

        if (redirect.isValid())
        {
            if (m_nrOfAllowedRedirects > 0)
            {
                m_nrOfAllowedRedirects--;
                QUrl newUrl = redirect;

                if (newUrl.isRelative())
                {
                    newUrl = m_pCurrentNetworkReply->url().resolved(newUrl);
                }

                qDebug() << newUrl;
                QNetworkRequest request(newUrl);
                request.setRawHeader("User-Agent", "Wget/1.12 (linux-gnu)");
                m_pCurrentNetworkReply->deleteLater();
                m_pCurrentNetworkReply = m_WebCtrl.get(request);
                return 1;
            }
            else
            {
                errorMsg = tr("Requested URL forces a redirection. Maximum number of redirections exceeded.");
                return -1;
            }
        }
        else
        {
            //no redirect, all good
            errorMsg = "";
            return 0;
        }
    }
    else
    {
        errorMsg = tr("no network reply instance available");
        return -1;
    }
}


