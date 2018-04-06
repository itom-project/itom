/****************************************************************************
**
** Copyright (C) 2009 Nokia Corporation and/or its subsidiary(-ies).
** Contact: Nokia Corporation (qt-info@nokia.com)
**
** This file is part of the documentation of Qt. It was originally
** published as part of Qt Quarterly.
**
** $QT_BEGIN_LICENSE:LGPL$
** No Commercial Usage
** This file contains pre-release code and may not be distributed.
** You may use this file in accordance with the terms and conditions
** contained in the Technology Preview License Agreement accompanying
** this package.
**
** GNU Lesser General Public License Usage
** Alternatively, this file may be used under the terms of the GNU Lesser
** General Public License version 2.1 as published by the Free Software
** Foundation and appearing in the file LICENSE.LGPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU Lesser General Public License version 2.1 requirements
** will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
**
** In addition, as a special exception, Nokia gives you certain
** additional rights.  These rights are described in the Nokia Qt LGPL
** Exception version 1.1, included in the file LGPL_EXCEPTION.txt in this
** package.
**
** If you have questions regarding the use of this file, please contact
** Nokia at qt-info@nokia.com.
**
**
**
**
**
**
**
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include <QtGui>
#include <qapplication.h>

#include "highlighter.h"
#include "textedit.h"
#include "codeeditor.h"
#include "qcompleter.h"
#include "treemodelcompleter.h"
#include <qtreeview.h>
#include <qheaderview.h>
#include <qgridlayout.h>
#include <qlineedit.h>

class DumpModel : public QStandardItemModel
{
public:
    DumpModel(QObject *parent) : QStandardItemModel(parent) {}

    QVariant data(const QModelIndex & index, int role = Qt::DisplayRole) const
    {
        QVariant test = QStandardItemModel::data(index, role);
        //qDebug() << test;
        return test;
    }

};

QAbstractItemModel *modelFromFile(const QString& fileName, QCompleter *completer)
{
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly))
        return new QStringListModel(completer);

#ifndef QT_NO_CURSOR
    QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
#endif
    QStringList words;

    DumpModel *model = new DumpModel(completer);
    QVector<QStandardItem *> parents(10);
    parents[0] = model->invisibleRootItem();

    while (!file.atEnd()) {
        QString line = file.readLine();
        QString trimmedLine = line.trimmed();
        if (line.isEmpty() || trimmedLine.isEmpty())
            continue;

        QRegularExpression re("^\\s+");
        QRegularExpressionMatch match = re.match(line);
        int nonws = match.capturedStart();
        int level = 0;
        if (nonws == -1) {
            level = 0;
        } else {
            if (line.startsWith("\t")) {
                level = match.capturedLength();
            } else {
                level = match.capturedLength()/4;
            }
        }

        if (level+1 >= parents.size())
            parents.resize(parents.size()*2);

        QStandardItem *item = new QStandardItem;
        item->setText(trimmedLine);
        parents[level]->appendRow(item);
        parents[level+1] = item;
    }

#ifndef QT_NO_CURSOR
    QApplication::restoreOverrideCursor();
#endif

    return model;
}

int main(int argv, char **args)
{
    Q_INIT_RESOURCE(customcompleter);

    QApplication app(argv, args);

    //TextEdit edit;
    CodeEditor edit;

    QTextDocument *document= edit.document();
    QTextOption op = document->defaultTextOption();
    op.setFlags(op.flags() | QTextOption::ShowTabsAndSpaces | QTextOption::ShowLineAndParagraphSeparators | QTextOption::AddSpaceForLineAndParagraphSeparators);
    document->setDefaultTextOption(op);

    edit.setWordWrapMode(QTextOption::WrapAnywhere);
    edit.setLineWrapMode(QPlainTextEdit::WidgetWidth);

    Highlighter highlighter(edit.document());

    TreeModelCompleter *completer = new TreeModelCompleter(&edit);
    completer->setModel(modelFromFile(":/resources/wordlist.txt", completer));
    completer->setSeparator(QLatin1String("."));
    completer->setModelSorting(QCompleter::CaseInsensitivelySortedModel);
    completer->setCompletionMode(QCompleter::UnfilteredPopupCompletion);
    //completer->setCompletionMode(QCompleter::InlineCompletion);
    completer->setCompletionMode(QCompleter::PopupCompletion);
    completer->setCompletionColumn(0);
    completer->setCompletionRole(Qt::DisplayRole);
    completer->setCaseSensitivity(Qt::CaseInsensitive);
    completer->setWrapAround(false);
    edit.setCompleter(completer);

    QTreeView *treeView = new QTreeView;
    treeView->setModel(completer->model());
    treeView->header()->hide();
    treeView->expandAll();

    QLineEdit *lineEdit = new QLineEdit();
    lineEdit->setCompleter(completer);

    QWidget *widget = new QWidget();

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(treeView, 0,0);
    layout->addWidget(&edit, 1,0);
    layout->addWidget(lineEdit, 2,0);

    widget->setLayout(layout);


    widget->setWindowTitle("Code Editor Example");
    widget->show();

    return app.exec();
}

