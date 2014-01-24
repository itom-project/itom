#ifndef HELPTREEDOCKWIDGET_H
#define HELPTREEDOCKWIDGET_H

#include <QtSql>
#include <qwidget.h>
#include <qstandarditemmodel.h>
#include "../../common/sharedStructures.h"
#include "ui_helpTreeDockWidget.h"
#include <qtimer.h>
#include <qfuturewatcher.h>
#include <qmovie.h>
#include "../widgets/abstractDockWidget.h"

class LeafFilterProxyModel; //forward declaration

class HelpTreeDockWidget : public QWidget
{
    Q_OBJECT

public:
    HelpTreeDockWidget(QWidget *parent, ito::AbstractDockWidget *dock = 0, Qt::WFlags flags = 0);
    ~HelpTreeDockWidget();

    enum itemType {typeSqlItem = 1, typeFilter = 2, typeWidget = 3, typeFPlugin = 4, typeWPlugin = 5};

public slots:
    void navigateBackwards();
    void navigateForwards();
    void expandTree();
    void collapseTree();
    void reloadDB();
    void liveFilter(const QString &filterText);
    void showTreeview();
    void unshowTreeview();
    void propertiesChanged();
    ito::RetVal showFilterWidgetPluginHelp(const QString &filtername, itemType type);

private slots:
    void on_treeView_clicked(QModelIndex i);
    void on_splitter_splitterMoved ( int pos, int index );
    void on_textBrowser_anchorClicked(const QUrl & link);   

    void dbLoaderFinished(int index);

private:

    

    struct SqlItem
    {
        int type;
        QString prefix;
        QString name;
        QString path;
    };

    struct DisplayBool
    {
        bool Filters;
        bool Modules;
    };

    Ui::HelpTreeDockWidget ui;
    static void createFilterNode(QStandardItemModel* model);
    static void createItemRek(QStandardItemModel* model, QStandardItem& parent, const QString parentPath, QList<SqlItem> &items, const QMap<int,QIcon> *iconGallery);
    static ito::RetVal loadDBinThread(const QString &path, const QStringList &includedDBs, QStandardItemModel *mainModel, const QMap<int,QIcon> *iconGallery, const DisplayBool &show);
    static ito::RetVal readSQL(const QString &filter, const QString &file, QList<SqlItem> &items);

    void CreateItem(QStandardItemModel& model, QStringList &items);
    void saveIni();
    void loadIni();
    ito::RetVal displayHelp(const QString &path, const int newpage);
    QStringList separateLink(const QUrl &link);
    ito::RetVal highlightContent(const QString &prefix , const QString &name , const QString &param , const QString &shortDesc, const QString &helpText, const QString &error, QTextDocument *document, bool htmlNotPlainText = true);
    QModelIndex findIndexByName(const QString &modelName);

    QString parseFilterContent(const QString &input);
    ito::RetVal parseParamVector(const QString &sectionname, const QVector<ito::Param> &paramVector, QString &content);
    QString parseParam(const QString &tmpl, const ito::Param &param);

    QFutureWatcher<ito::RetVal> dbLoaderWatcher;

    // Variables
    QStandardItemModel        *m_pMainModel;
    LeafFilterProxyModel      *m_pMainFilterModel;
    ito::AbstractDockWidget   *m_pParent;
    QStringList                m_history;
    QStringList                m_includedDBs;
    QString                    m_dbPath;


    QMovie                    *m_previewMovie;

    QMap<int, QIcon> m_iconGallery;
    DisplayBool m_showSelection;
    int m_historyIndex;
    int m_autoCollTime;
    double m_percWidthVi;
    double m_percWidthUn;
    bool m_treeVisible;
    bool m_plaintext;
    bool m_openLinks;
    bool m_autoCollTree;
    bool m_forced;
protected:
    bool eventFilter(QObject *obj, QEvent *event);
};

#endif // HELPTREEDOCKWIDGET_H
