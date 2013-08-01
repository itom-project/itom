#ifndef HELPTREEDOCKWIDGET_H
#define HELPTREEDOCKWIDGET_H

#include <qwidget.h>
#include <qstandarditemmodel.h>

#include "ui_helpTreeDockWidget.h"

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

private slots:
    void on_treeView_clicked(QModelIndex i);
    void on_textBrowser_anchorClicked(const QUrl & link);    

private:
    void CreateItemRek(QStandardItemModel& model, QStandardItem& parent, const QString parentPath, QList<QString> &items);
    void CreateItem(QStandardItemModel& model, QList<QString> &items);
    void DisplayHelp(const QString &path, const int newpage);

	QStringList SeparateLink(const QUrl &link);
    QList<QString> ReadSQL(const QString &filter);
    QTextDocument* HighlightContent(const QString &Helptext, const QString &Prefix , const QString &Name , const QString &Param , const QString &ShortDesc, const QString &Error);
	QModelIndex FindIndexByName(const QString Modelname);

    Ui::HelpTreeDockWidget ui;

    QStandardItemModel *m_pMainModel;
    LeafFilterProxyModel *m_pMainFilterModel;
    QStringList *m_pHistory;


};

#endif // HELPTREEDOCKWIDGET_H
