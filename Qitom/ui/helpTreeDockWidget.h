#ifndef HELPTREEDOCKWIDGET_H
#define HELPTREEDOCKWIDGET_H

#if QT_VERSION < 0x050800
    #include <QtSql/qsql.h>
#else
    #include <QtSql/qtsqlglobal.h>
#endif
#include <qwidget.h>
#include <qstandarditemmodel.h>
#include "../../common/sharedStructures.h"
#include "ui_helpTreeDockWidget.h"
#include <qtimer.h>
#include <qfuturewatcher.h>
#include <qmovie.h>
#include "../widgets/abstractDockWidget.h"

class QShowEvent; //forward declaration

namespace ito
{

class LeafFilterProxyModel; //forward declaration

class HelpTreeDockWidget : public QWidget
{
    Q_OBJECT

public:
    HelpTreeDockWidget(QWidget *parent, ito::AbstractDockWidget *dock = 0, Qt::WindowFlags flags = 0);
    ~HelpTreeDockWidget();

    enum itemType { typeSqlItem = 1, typeFilter = 2, typeWidget = 3, typeFPlugin = 4, typeWPlugin = 5, typeCategory = 6, typeDataIO = 7, typeActuator = 8 };

    enum iconType 
    {
        iconFilter = 100, 
        iconPluginAlgo = 101,
        iconPluginFilter = 102,
        iconWidget = 103,
        iconPluginDataIO = 104,
        iconPluginGrabber = 105,
        iconPluginAdda = 106,
        iconPluginRawIO = 107,
        iconPluginActuator = 108
    };

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
    void showPluginInfo(const QString &name, int type, const QModelIndex &modelIndex, bool fromLink);
    ito::RetVal showFilterWidgetPluginHelp(const QString &filtername, itemType type);

private slots:
    void on_splitter_splitterMoved ( int pos, int index );
    void on_helpTreeContent_anchorClicked(const QUrl & link);

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
    static void createItemRek(QStandardItem& parent, const QString &parentPath, QList<SqlItem> &items, const QMap<int,QIcon> *iconGallery);
    static ito::RetVal loadDBinThread(const QString &path, const QStringList &includedDBs, QStandardItemModel *mainModel, const QMap<int,QIcon> *iconGallery, const DisplayBool &show);
    static ito::RetVal readSQL(const QString &file, QList<SqlItem> &items);

    void CreateItem(QStandardItemModel& model, QStringList &items);
    void saveIni();
    void loadIni();
    ito::RetVal displayHelp(const QString &path);
    QStringList separateLink(const QUrl &link);
    ito::RetVal highlightContent(const QString &prefix , const QString &name , const QString &param , const QString &shortDesc, const QString &helpText, const QString &error, QTextDocument *document);
    QModelIndex findIndexByPath(const int type, QStringList path, const QStandardItem* current);

    QString parseFilterWidgetContent(const QString &input);
    ito::RetVal parseParamVector(const QString &sectionname, const QVector<ito::Param> &paramVector, QString &content);
    QString parseParam(const QString &tmpl, const ito::Param &param);

    QFutureWatcher<ito::RetVal> dbLoaderWatcher;

    // Const
    static const int m_urPath = Qt::UserRole + 1;
    static const int m_urType = Qt::UserRole + 2;

    QString minText(int minimum) const;
    QString minText(double minimum) const;
    QString minText(char minimum) const;
    QString maxText(int minimum) const;
    QString maxText(double minimum) const;
    QString maxText(char minimum) const;
    QString minmaxText(int value) const;
    QString minmaxText(double value) const;
    QString minmaxText(char value) const;
    
    // Variables
    Ui::HelpTreeDockWidget   ui;                
    QStandardItemModel      *m_pMainModel;          /*!< Model to store the tree with all database entries*/
    LeafFilterProxyModel    *m_pMainFilterModel;    /*!< Filtered Tree Model (between the model and the tree*/
    ito::AbstractDockWidget *m_pParent;             /*!< pointer to helpDockWidget with Toolbar*/
    QList<QModelIndex>       m_history;             /*!< List to store the adresses of the last visited pages */
    QStringList              m_includedDBs; 
    QString                  m_dbPath;              /*!< path from where the databases are loaded */
    QMovie                  *m_previewMovie;        /*!< turning circle to show "wait" status*/    
    QMap<int, QIcon>         m_iconGallery;
    DisplayBool              m_showSelection;
    int                      m_historyIndex;        
    int                      m_autoCollTime;        /*!< after this time the tree automatically becomes smaller*/
    double                   m_treeWidthVisible;    /*!< width of tree while visible (in percent of the total width)*/
    double                   m_treeWidthInvisible;  /*!< width of tree while small (in percent of the total width)*/
    bool                     m_treeVisible;
    bool                     m_plaintext;           /*!< true: html code is displayed, false: normal help with style is displayed*/
    bool                     m_openLinks;           /*!< decides if external links open when clicked*/
    bool                     m_autoCollTree;
    bool                     m_forced;
    bool                     m_internalCall;        /*!< If a page is called by the history buttons, this bool prevents from that this page is stored in the historylist again*/
    bool                     m_doingExpandAll;      /*!< if expand all is executed from somewhere, the slots on_treeView_expanded or on_treeView_collapsed should not be called to avoid crazy never-ending loops in Qt5, debug.*/

protected:
    bool eventFilter(QObject *obj, QEvent *event);
    void showEvent(QShowEvent *event);
};

} //end namespace ito

#endif // HELPTREEDOCKWIDGET_H
