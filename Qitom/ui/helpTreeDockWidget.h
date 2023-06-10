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

    // the following properties replace $<name>$ tags in the help_style.css file,
    // where <name> is the name of the property
    Q_PROPERTY(QColor backgroundColorHeading READ backgroundColorHeading WRITE setBackgroundColorHeading) // default: #efefef;
    Q_PROPERTY(QColor textColorHeading READ textColorHeading WRITE setTextColorHeading) // default: #0c3762;
    Q_PROPERTY(QColor linkColor READ linkColor WRITE setLinkColor); // default: #dc3c01
    Q_PROPERTY(QColor backgroundParamName READ backgroundParamName WRITE setBackgroundParamName); // default: #dcb8aa
    Q_PROPERTY(QColor textColorSection READ textColorSection WRITE setTextColorSection); // default: #dc3c01
    Q_PROPERTY(QColor backgroundColorSection READ backgroundColorSection WRITE setBackgroundColorSection); // default: #eeeeee

    Q_FLAGS(State States)

public:
    HelpTreeDockWidget(
        QWidget* parent,
        ito::AbstractDockWidget* dock = 0,
        Qt::WindowFlags flags = Qt::WindowFlags());
    ~HelpTreeDockWidget();

    enum HelpItemType
    {
        typeFilter = 2,   /* a filter method from an algorithm plugin */
        typeWidget = 3,   /* a widget method from an algorithm plugin */
        typeFPlugin = 4,  /* an algorithm plugin in the filter section */
        typeWPlugin = 5,  /* an algorithm plugin in the widget section */
        typeCategory = 6, /* a category */
        typeDataIO = 7,   /* a dataIO plugin */
        typeActuator = 8  /* an actuator plugin */
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


    QColor backgroundColorHeading() const;
    void setBackgroundColorHeading(const QColor &color);

    QColor textColorHeading() const;
    void setTextColorHeading(const QColor &color);

    QColor backgroundColorSection() const;
    void setBackgroundColorSection(const QColor &color);

    QColor textColorSection() const;
    void setTextColorSection(const QColor &color);

    QColor linkColor() const;
    void setLinkColor(const QColor &color);

    QColor backgroundParamName() const;
    void setBackgroundParamName(const QColor &color);



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

    void loadAndProcessCssStyleSheet();

    // Const
    static const int rolePath = Qt::UserRole + 1;
    static const int roleType = Qt::UserRole + 2;
    static const int roleFilename = Qt::UserRole + 3;

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

    QColor m_backgroundColorHeading; // default: #efefef;
    QColor m_textColorHeading; // default: #0c3762;
    QColor m_linkColor; // default: #dc3c01
    QColor m_backgroundParamName; // default: #dcb8aa
    QColor m_textColorSection; // default: #dc3c01
    QColor m_backgroundColorSection; // default: #eeeeee

    QFutureWatcher<ito::RetVal> m_loaderWatcher;
    QMutex m_dbLoaderMutex;

protected:
    void showEvent(QShowEvent *event);
    void timerEvent(QTimerEvent *event);
};

} //end namespace ito

#endif // HELPTREEDOCKWIDGET_H
