#ifndef HELPTREEDOCKWIDGET_H
#define HELPTREEDOCKWIDGET_H

#include <QtSql>
#include <qwidget.h>
#include <qstandarditemmodel.h>
#include "common\sharedStructures.h"
#include "ui_helpTreeDockWidget.h"
#include <qtimer.h>

class LeafFilterProxyModel; //forward declaration

class HelpTreeDockWidget : public QWidget
{
    Q_OBJECT

public:
    HelpTreeDockWidget(QWidget *parent = 0, Qt::WFlags flags = 0);
    ~HelpTreeDockWidget();

public slots:
	void navigateBackwards();
	void navigateForwards();
	void expandTree();
	void collapseTree();
	void reloadDB();
    void liveFilter(const QString &filtertext);
	void showTreeview();
	void unshowTreeview();

private slots:
    void on_treeView_clicked(QModelIndex i);
    void on_textBrowser_anchorClicked(const QUrl & link);   

private:
    void CreateItemRek(QStandardItemModel& model, QStandardItem& parent, const QString parentPath, QList<QString> &items);
    void CreateItem(QStandardItemModel& model, QList<QString> &items);
    ito::RetVal DisplayHelp(const QString &path, const int newpage);
	QTimer *TreeCloseTimer;
	QStringList SeparateLink(const QUrl &link);

    ito::RetVal readSQL(const QString &filter, const QString &file, QList<QString> &items);
    QTextDocument* HighlightContent(const QString &Helptext, const QString &Prefix , const QString &Name , const QString &Param , const QString &ShortDesc, const QString &Error);
	QModelIndex FindIndexByName(const QString Modelname);

    Ui::HelpTreeDockWidget ui;

    QStandardItemModel *m_pMainModel;
    LeafFilterProxyModel *m_pMainFilterModel;
    QStringList *m_pHistory;
	int m_pHistoryIndex;
	QString m_dbPath;
	QList<QSqlDatabase> m_DBList;
protected:
	bool eventFilter(QObject *obj, QEvent *event);
	void leaveEvent( QEvent * event );
	void enterEvent( QEvent * event );
};

#endif // HELPTREEDOCKWIDGET_H
