/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom.
  
    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef ABSTRACTPYSCINTILLAWIDGET
#define ABSTRACTPYSCINTILLAWIDGET

#include "../common/sharedStructures.h"

// Under Windows, define QSCINTILLA_MAKE_DLL to create a Scintilla DLL, or
// define QSCINTILLA_DLL to link against a Scintilla DLL, or define neither
// to either build or link against a static Scintilla library.
//!< this text is coming from qsciglobal.h
#define QSCINTILLA_DLL  //http://www.riverbankcomputing.com/pipermail/qscintilla/2007-March/000034.html

#include <Qsci/qsciscintilla.h>
#include <Qsci/qscilexerpython.h>
#include <Qsci/qsciapis.h>
#include <qevent.h>
#include "../organizer/qsciApiManager.h"
#include <qstringlist.h>
#include <qrect.h>

namespace ito {

class AbstractPyScintillaWidget : public QsciScintilla
{
    Q_OBJECT //-> see: #define QSCINTILLA_DLL  //http://www.riverbankcomputing.com/pipermail/qscintilla/2007-March/000034.html in sharedStructures.h

public:
    AbstractPyScintillaWidget(QWidget* parent = NULL);
    ~AbstractPyScintillaWidget();

    QString getWordAtPosition(const int &line, const int &index);

protected:

    enum tUserSelectionState { selNo, selRange };

    void init();
    //bool event ( QEvent * event );

    virtual void loadSettings(); //overwrite this method if you want to load further settings

//    void mouseReleaseEvent(QMouseEvent * event);
//    void keyReleaseEvent(QKeyEvent * event);
    void checkUserSelectionState();
    QString formatPythonCodePart(const QString &text, int &lineCount);
    QString formatConsoleCodePart(const QString &text);

    tUserSelectionState m_userSelectionState;

private:
    QsciLexerPython* qSciLex;
    QsciApiManager *m_pApiManager;

    QStringList m_installedApiFiles;
    bool m_textIndicatorActive;
    int m_textIndicatorNr; //number of indicator which marks all appearances of the currently selected text
    int getSpaceTabCount(const QString &s);
    bool haveToIndention(QString s);

public slots:
    void selectionChanged();
    void reloadSettings() { loadSettings(); };

signals:
    void userSelectionChanged(int lineFrom, int indexFrom, int lineTo, int indexTo);

};

} //end namespace ito

#endif