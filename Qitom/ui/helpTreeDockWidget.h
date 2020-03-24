#ifndef HELPTREEDOCKWIDGET_H
#define HELPTREEDOCKWIDGET_H

#include "../../common/sharedStructures.h"

#include <qwidget.h>
#include <qstandarditemmodel.h>
#include <qtimer.h>
#include <qfuturewatcher.h>
#include <qmovie.h>

#include "../widgets/abstractDockWidget.h"

#include "ui_helpTreeDockWidget.h"

class QShowEvent; //forward declaration

namespace ito
{

class LeafFilterProxyModel; //forward declaration

class HelpTreeDockWidget : public QWidget
{
    Q_OBJECT

    Q_FLAGS(State States)

public:
    HelpTreeDockWidget(QWidget *parent, ito::AbstractDockWidget *dock = 0, Qt::WindowFlags flags = 0);
    ~HelpTreeDockWidget();

    enum HelpItemType 
    {  
        typeFilter = 2, 
        typeWidget = 3, 
        typeFPlugin = 4, 
        typeWPlugin = 5, 
        typeCategory = 6, 
        typeDataIO = 7, 
        typeActuator = 8 
    };

    enum IconType 
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

    enum State
    {
        stateIdle = 0x00,
        stateVisible = 0x01,
        stateContentLoaded = 0x02
    };
    Q_DECLARE_FLAGS(States, State)
    
        

public slots:
    void navigateBackwards();
    void navigateForwards();
    void expandTree();
    void collapseTree();
    void reloadHelpResources();
    void liveFilter(const QString &filterText);
    void propertiesChanged();
    void showPluginInfo(const QString &name, HelpItemType type, const QModelIndex &modelIndex, bool fromLink);
    ito::RetVal showFilterWidgetPluginHelp(const QString &filtername, HelpItemType type);

private slots:
    void on_splitter_splitterMoved ( int pos, int index );
    void on_helpTreeContent_anchorClicked(const QUrl & link);

    void loadHelpResourcesFinished(int index);

    void on_treeView_expanded(const QModelIndex &index);
    void on_treeView_collapsed(const QModelIndex &index);

    void selectedItemChanged(const QModelIndex &current, const QModelIndex &previous);

private:

    static void createFilterWidgetNode(int fOrW, QStandardItemModel* model, const QMap<int,QIcon> *iconGallery);
    static ito::RetVal loadHelpResources(QStandardItemModel *mainModel, const QMap<int,QIcon> *iconGallery);

    void storeSettings();
    void restoreSettings();
    QStringList separateLink(const QUrl &link);
    ito::RetVal highlightContent(const QString &prefix , const QString &name , const QString &param , const QString &shortDesc, const QString &helpText, const QString &error, QTextDocument *document, const QMap<QString, QImage> &images);
    QModelIndex findIndexByPath(const int type, QStringList path, const QStandardItem* current);

    QString parseFilterWidgetContent(const QString &input);
    ito::RetVal parseParamVector(const QString &sectionname, const QVector<ito::Param> &paramVector, QString &content);
    QString parseParam(const QString &tmpl, const ito::Param &param);

    // Const
    static const int rolePath = Qt::UserRole + 1;
    static const int roleType = Qt::UserRole + 2;
    static const int roleFilename = Qt::UserRole + 3;

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
    QMovie                  *m_previewMovie;        /*!< turning circle to show "wait" status*/    
    QMap<int, QIcon>         m_iconGallery;
    int                      m_historyIndex;        
    int                      m_autoCollTime;        /*!< after this time the tree automatically becomes smaller*/
    double                   m_treeWidthVisible;    /*!< width of tree while visible (in percent of the total width)*/
    double                   m_treeWidthInvisible;  /*!< width of tree while small (in percent of the total width)*/
    bool                     m_treeVisible;
    bool                     m_plaintext;           /*!< true: html code is displayed, false: normal help with style is displayed*/
    bool                     m_autoCollTree;
    bool                     m_internalCall;        /*!< If a page is called by the history buttons, this bool prevents from that this page is stored in the historylist again*/
    bool                     m_doingExpandAll;      /*!< if expand all is executed from somewhere, the slots on_treeView_expanded or on_treeView_collapsed should not be called to avoid crazy never-ending loops in Qt5, debug.*/
    States                   m_state;               /*!< stateIdle if the widget is not visible yet and no content has been loaded, stateVisible if it became visible for the first time, stateContentLoaded if all contents have been loaded.*/
    QString                  m_filterTextPending;
    int                      m_filterTextPendingTimer;

    QFutureWatcher<ito::RetVal> m_loaderWatcher;
    QMutex m_dbLoaderMutex;

protected:
    void showEvent(QShowEvent *event);
    void timerEvent(QTimerEvent *event);
};

} //end namespace ito

#endif // HELPTREEDOCKWIDGET_H
