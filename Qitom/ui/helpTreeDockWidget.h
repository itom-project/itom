#ifndef HELPTREEDOCKWIDGET_H
#define HELPTREEDOCKWIDGET_H

#include <QtSql/qsql.h>
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
    HelpTreeDockWidget(QWidget *parent, ito::AbstractDockWidget *dock = 0, Qt::WindowFlags flags = 0);
    ~HelpTreeDockWidget();

    enum itemType {typeSqlItem = 1, typeFilter = 2, typeWidget = 3, typeFPlugin = 4, typeWPlugin = 5, typeCategory = 6, typeDataIO = 7, typeActuator = 8};

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
    void showPluginInfo(QString name, int type, const QModelIndex modelIndex, bool fromLink);
    ito::RetVal showFilterWidgetPluginHelp(const QString &filtername, itemType type);

private slots:
    void on_splitter_splitterMoved ( int pos, int index );
    void on_textBrowser_anchorClicked(const QUrl & link);   

    void dbLoaderFinished(int index);

    void on_treeView_expanded(const QModelIndex &index);
    void on_treeView_collapsed(const QModelIndex &index);

    void selectedItemChanged(const QModelIndex &current, const QModelIndex &previous);

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
        bool Widgets;
        bool Modules;
        bool DataIO;
    };

    
    static void createFilterWidgetNode(int fOrW, QStandardItemModel* model, const QMap<int,QIcon> *iconGallery);
    static void createItemRek(QStandardItemModel* model, QStandardItem& parent, const QString parentPath, QList<SqlItem> &items, const QMap<int,QIcon> *iconGallery);
    static ito::RetVal loadDBinThread(const QString &path, const QStringList &includedDBs, QStandardItemModel *mainModel, const QMap<int,QIcon> *iconGallery, const DisplayBool &show);
    static ito::RetVal readSQL(const QString &filter, const QString &file, QList<SqlItem> &items);

    void CreateItem(QStandardItemModel& model, QStringList &items);
    void saveIni();
    void loadIni();
    ito::RetVal displayHelp(const QString &path);
    QStringList separateLink(const QUrl &link);
    ito::RetVal highlightContent(const QString &prefix , const QString &name , const QString &param , const QString &shortDesc, const QString &helpText, const QString &error, QTextDocument *document);
    QModelIndex findIndexByPath(const int type, QStringList path, QStandardItem* current);

    QString parseFilterWidgetContent(const QString &input);
    ito::RetVal parseParamVector(const QString &sectionname, const QVector<ito::Param> &paramVector, QString &content);
    QString parseParam(const QString &tmpl, const ito::Param &param);

    QFutureWatcher<ito::RetVal> dbLoaderWatcher;

    // Const
    static const int m_urPath = Qt::UserRole + 1;
    static const int m_urType = Qt::UserRole + 2;

    // Variables
    Ui::HelpTreeDockWidget ui;

    QStandardItemModel        *m_pMainModel;
    LeafFilterProxyModel      *m_pMainFilterModel;
    ito::AbstractDockWidget   *m_pParent;
    QList<QModelIndex>         m_history;
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
    bool m_internalCall;
protected:
    bool eventFilter(QObject *obj, QEvent *event);
};

#endif // HELPTREEDOCKWIDGET_H
