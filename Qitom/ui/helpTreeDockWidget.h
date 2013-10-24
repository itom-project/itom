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
    void liveFilter(const QString &filterText);
	void showTreeview();
	void unshowTreeview();
	void propertiesChanged();

private slots:
    void on_treeView_clicked(QModelIndex i);
	void on_splitter_splitterMoved ( int pos, int index );
    void on_textBrowser_anchorClicked(const QUrl & link);   

	void dbLoaderFinished(int index);

private:
	Ui::HelpTreeDockWidget ui;
    static void createItemRek(QStandardItemModel* model, QStandardItem& parent, const QString parentPath, QStringList &items, const QMap<QString,QIcon> *iconGallery);
	static ito::RetVal loadDBinThread(const QString &path, const QStringList &includedDBs, QStandardItemModel *mainModel, const QMap<QString,QIcon> *iconGallery);
	static ito::RetVal readSQL(const QString &filter, const QString &file, QList<QString> &items);

	void CreateItem(QStandardItemModel& model, QStringList &items);
    void saveIni();
	void loadIni();
	ito::RetVal displayHelp(const QString &path, const int newpage);
	QStringList separateLink(const QUrl &link);
    QTextDocument* highlightContent(const QString &helpText, const QString &prefix , const QString &name , const QString &param , const QString &shortDesc, const QString &error);
	QModelIndex findIndexByName(const QString &modelName);

	QFutureWatcher<ito::RetVal> dbLoaderWatcher;

	// Variables
    QStandardItemModel		*m_pMainModel;
    LeafFilterProxyModel	*m_pMainFilterModel;
    QStringList				m_history;
	QStringList				m_includedDBs;
	QString					m_dbPath;

	QMovie					*m_previewMovie;

    QMap<QString, QIcon> m_iconGallery;
	
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
