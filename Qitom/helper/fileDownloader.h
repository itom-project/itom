#ifndef FILEDOWNLOADER_H
#define FILEDOWNLOADER_H
     
#include <qobject.h>
#include <qbytearray.h>
#include <qnetworkaccessmanager.h>
#include <qnetworkrequest.h>
#include <qnetworkreply.h>
#include <qurl.h>
     
class FileDownloader : public QObject
{
    Q_OBJECT
public:
    explicit FileDownloader(QUrl imageUrl, int nrOfAllowedRedirects = 0, QObject *parent = 0);
     
    virtual ~FileDownloader();

    enum Status { sRunning, sAborted, sFinished, sError };
     
    QByteArray downloadedData() const;

    void abortDownload();

    int getDownloadProgress();

    Status getStatus(QString &errorMsg);
     
signals:
    //void downloaded();
     
private slots:
     
    void fileDownloaded(QNetworkReply* pReply);
    void downloadProgress(qint64 bytesReceived, qint64 bytesTotal);
     
private:

    int checkRedirect(QString &errorMsg);
     
    QNetworkAccessManager m_WebCtrl;
     
    QByteArray m_DownloadedData;
    
    QNetworkReply *m_pCurrentNetworkReply;

    qint64 m_bytesReceived;
    qint64 m_bytesTotal;

    int m_nrOfAllowedRedirects;
     
};
     
#endif // FILEDOWNLOADER_H
