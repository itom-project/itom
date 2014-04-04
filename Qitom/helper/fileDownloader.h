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
    explicit FileDownloader(QUrl imageUrl, QObject *parent = 0);
     
    virtual ~FileDownloader();
     
    QByteArray downloadedData() const;
     
signals:
    void downloaded();
     
private slots:
     
    void fileDownloaded(QNetworkReply* pReply);
     
private:
     
    QNetworkAccessManager m_WebCtrl;
     
    QByteArray m_DownloadedData;
     
};
     
#endif // FILEDOWNLOADER_H
