/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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

#ifndef SCRIPTEDITORPRINTER_H
#define SCRIPTEDITORPRINTER_H

#include <qprinter.h>
#include <qtextdocument.h>

QT_BEGIN_NAMESPACE
class QRect;
class QPainter;
QT_END_NAMESPACE


namespace ito
{

class CodeEditor;

class ScriptEditorPrinter : public QPrinter
{
public:
    ScriptEditorPrinter(QPrinter::PrinterMode mode = QPrinter::ScreenResolution);

    virtual ~ScriptEditorPrinter();

    virtual QRect formatPage( QPainter &painter, bool drawing, const QRect &area, int pageNumber, int pageCount);

    //! Return the number of points to add to each font when printing.
    //!
    //! \sa setMagnification()
    int magnification() const {return m_magnification;}

    //! Sets the number of points to add to each font when printing to \a
    //! magnification.
    //!
    //! \sa magnification()
    virtual void setMagnification(int magnification);

    //! Print a range of lines from the Scintilla instance \a qsb.  \a from is
    //! the first line to print and a negative value signifies the first line
    //! of text.  \a to is the last line to print and a negative value
    //! signifies the last line of text.  true is returned if there was no
    //! error.
    virtual int printRange(CodeEditor *editor, int from = -1, int to = -1);

    virtual void paintPage(int pageNumber, int pageCount, QPainter* painter, QTextDocument* doc, const QRectF& textRect, float scale);

private:
    int m_magnification;
    int m_alphaLevel;

    ScriptEditorPrinter(const ScriptEditorPrinter &);
    ScriptEditorPrinter &operator=(const ScriptEditorPrinter &);
};

} //end namespace ito

#endif
