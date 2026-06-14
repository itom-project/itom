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

    Further hints:
    ------------------------

    This file belongs to the code editor of itom. The code editor is
    in major parts a fork / rewritten version of the python-based source
    code editor PyQode from Colin Duquesnoy and others
    (see https://github.com/pyQode). PyQode itself is licensed under
    the MIT License (MIT).

    Some parts of the code editor of itom are also inspired by the
    source code editor of the Spyder IDE (https://github.com/spyder-ide),
    also licensed under the MIT License and developed by the Spyder Project
    Contributors.

*********************************************************************** */

#ifndef CODEEDITOR_H
#define CODEEDITOR_H

#include <qcolor.h>
#include <qevent.h>
#include <qpair.h>
#include <qplaintextedit.h>
#include <qpoint.h>
#include <qset.h>
#include <qtextobject.h>

#include "syntaxHighlighter/syntaxHighlighterBase.h"
#include "textDecoration.h"

class QMenu; // forward declaration
class QMimeData; // forward declaration

namespace ito {

struct VisibleBlock
{
    int topPosition;
    int lineNumber;
    QTextBlock textBlock;
};

class PanelsManager; // forward declaration
class TextDecorationsManager; // forward declaration
class DelayJobRunnerBase; // forward declaration
class ModesManager; // forward declaration
class SyntaxHighlighterBase; // forward declaration
class TextBlockUserData;


/*
The editor widget is a simple extension to QPlainTextEdit.
It adds a few utility signals/methods and introduces the concepts of
**Managers, Modes and Panels**.
A **mode/panel** is an editor extension that, once added to a CodeEdit
instance, may modify its behaviour and appearance:
    * **Modes** are simple objects which connect to the editor signals to
    append new behaviours (such as automatic indentation, code completion,
    syntax checking,...)
    * **Panels** are the combination of a **Mode** and a **QWidget**.
    They are displayed in the CodeEdit's content margins.
    When you install a Panel on a CodeEdit, you can choose to install it in
    one of the four following zones:
        .. image:: _static/editor_widget.png
            :align: center
            :width: 600
            :height: 450
A **manager** is an object that literally manage a specific aspect of
:class:`pyqode.core.api.CodeEdit`. There are managers to manage the list of
modes/panels, to open/save file and to control the backend:
    - :attr:`pyqode.core.api.CodeEdit.file`:
        File manager. Use it to open/save files or access the opened file
        attribute.
    - :attr:`pyqode.core.api.CodeEdit.backend`:
        Backend manager. Use it to start/stop the backend or send a work
        request.
    - :attr:`pyqode.core.api.CodeEdit.modes`:
        Modes manager. Use it to append/remove modes on the editor.
    - :attr:`pyqode.core.api.CodeEdit.panels`:
        Modes manager. Use it to append/remove panels on the editor.
Starting from version 2.1, CodeEdit defines the
:attr:`pyqode.core.api.CodeEdit.mimetypes` class attribute that can be used
by IDE to determine which editor to use for a given mime type. This
property is a list of supported mimetypes. An empty list means the
CodeEdit is generic. **Code editors specialised for a specific language
should define the mime types they support!**
*/
class CodeEditor : public QPlainTextEdit
{
    Q_OBJECT
public:
    enum EdgeMode
    {
        EdgeNone, //!< Long lines are not marked.
        EdgeLine, //!< A vertical line is drawn at the column set by setEdgeColumn(). This is
                  //!< recommended for monospace fonts.
        EdgeBackground //!< The background color of characters after the column limit is changed to
                       //!< the color set by setEdgeColor(). This is recommended for proportional
                       //!< fonts.
    };

    CodeEditor(QWidget* parent = NULL, bool createDefaultActions = true);
    virtual ~CodeEditor();

    bool useSpacesInsteadOfTabs() const;
    void setUseSpacesInsteadOfTabs(bool value);

    bool selectLineOnCopyEmpty() const;
    void setSelectLineOnCopyEmpty(bool value);

    bool showContextMenu() const;
    void setShowContextMenu(bool value);

    bool showWhitespaces() const;
    void setShowWhitespaces(bool value);

    QString fontName() const;
    void setFontName(const QString& value);

    int fontSize() const;
    void setFontSize(int fontSize);

    int zoomLevel() const;
    void setZoomLevel(int value);

    int tabLength() const;
    void setTabLength(int value);

    QColor background() const;
    void setBackground(const QColor& value);

    QColor foreground() const;
    void setForeground(const QColor& value);

    QColor selectionForeground() const;
    void setSelectionForeground(const QColor& value);

    QColor selectionBackground() const;
    void setSelectionBackground(const QColor& value);

    QColor whitespacesForeground() const;
    void setWhitespacesForeground(const QColor& value);

    bool saveOnFocusOut() const;
    void setSaveOnFocusOut(bool value);

    EdgeMode edgeMode() const;
    void setEdgeMode(EdgeMode mode);

    int edgeColumn() const;
    void setEdgeColumn(int column);

    QColor edgeColor() const;
    void setEdgeColor(const QColor& color);

    bool showIndentationGuides() const;
    void setShowIndentationGuides(bool value);

    QColor indentationGuidesColor() const;
    void setIndentationGuidesColor(const QColor& color);

    QList<VisibleBlock> visibleBlocks() const;
    bool dirty() const;

    int firstVisibleLine() const;
    void setFirstVisibleLine(int line);

    bool isModified() const;
    void setModified(bool modified);

    QString wordSeparators() const
    {
        return m_wordSeparators;
    }

    bool isUndoAvailable() const
    {
        return m_undoAvailable;
    }
    bool isRedoAvailable() const
    {
        return m_redoAvailable;
    }

    void setMouseCursor(const QCursor& cursor);

    void cursorPosition(int& line, int& column) const;

    void setViewportMargins(int left, int top, int right, int bottom);

    QRectF blockBoundingGeometry(const QTextBlock& block) const { return QPlainTextEdit::blockBoundingGeometry(block); }
    QPointF contentOffset() const { return QPlainTextEdit::contentOffset(); }

    PanelsManager* panels() const;
    TextDecorationsManager* decorations() const;
    ModesManager* modes() const;

    SyntaxHighlighterBase* syntaxHighlighter() const;

    int currentLineNumber() const;
    int currentColumnNumber() const;
    int lineNbrFromPosition(int yPos) const;
    int lineCount() const;
    int lineLength(int line) const;
    QTextCursor selectWholeLine(int line = -1, bool applySelection = true);
    QTextCursor selectLines(int start = 0, int end = -1, bool applySelection = true);
    QPair<int, int> selectionRange() const; // start, end
    void getSelection(int* lineFrom, int* indexFrom, int* lineTo, int* indexTo);
    void setSelection(int lineFrom, int indexFrom, int lineTo, int indexTo);
    bool hasSelectedText() const;
    int linePosFromNumber(int lineNumber) const;
    void lineIndexFromPosition(const QPoint& pos, int* line, int* column) const;
    void lineIndexFromPosition(int pos, int* line, int* column) const;
    void getCursorPosition(int* line, int* column) const;
    QTextCursor setCursorPosition(int line, int column, bool applySelection = true);
    void unfoldCursorPosition(); // make sures, that the line of the current cursor is unfold
    void ensureLineVisible(int line);
    QTextCursor gotoLine(int line, int column, bool move = true);

    void removeSelectedText();

    bool findFirst(
        const QString& expr,
        bool re,
        bool cs,
        bool wo,
        bool wrap,
        bool forward = true,
        int line = -1,
        int index = -1,
        bool show = true);
    bool findNext();
    void replace(const QString& text);

    void endUndoAction()
    {
        textCursor().endEditBlock();
    }
    void beginUndoAction()
    {
        textCursor().beginEditBlock();
    }

    QString selectedText() const;
    int length() const
    {
        return toPlainText().size();
    }
    int positionFromLineIndex(int line, int column) const;

    int lineIndent(int lineNumber = -1) const;
    int lineIndent(const QTextBlock* lineNbr) const;
    QString lineText(int lineIdx) const;
    void markWholeDocDirty();
    void callResizeEvent(QResizeEvent* evt)
    {
        resizeEvent(evt);
    }

    virtual QString codeText(int& /*line*/, int& /*column*/) const
    {
        return toPlainText();
    } // usually this is the same than toPlainText(), however in the console widget, the codeText()
      // contains all the history of input code within this session!

    void indent();
    void unindent();

    virtual void cut();
    virtual void copy();

    void resetStylesheet();
    void rehighlight();
    void rehighlightBlock(int lineFromIdx, int lineToIdx /*=-1*/);

    void showTooltip(const QPoint& pos, const QString& tooltip);
    void showTooltip(
        const QPoint& pos, const QString& tooltip, const TextDecoration::Ptr& senderDeco);

    void setPlainText(
        const QString& text, const QString& mimeType = "", const QString& encoding = "");
    void insertAt(const QString& text, int line, int index);
    void append(const QString& text);

    bool isCommentOrString(
        const QTextCursor& cursor,
        const QList<StyleItem::StyleType>& formats = QList<StyleItem::StyleType>());
    bool isCommentOrString(
        const QTextBlock& block,
        const QList<StyleItem::StyleType>& formats = QList<StyleItem::StyleType>());
    bool isNumber(const QTextCursor& cursor) const;
    bool isNumber(const QTextBlock& block) const;

    QTextCursor wordUnderCursor(bool selectWholeWord) const;
    QTextCursor wordUnderCursor(const QTextCursor& cursor, bool selectWholeWord) const;
    QString wordAtPosition(int line, int index, bool selectWholeWord) const;
    QTextCursor wordUnderMouseCursor() const;

    TextBlockUserData* getTextBlockUserData(int lineIndex, bool createIfNotExist = true);
    TextBlockUserData* getTextBlockUserData(QTextBlock& block, bool createIfNotExist = true);
    QSet<TextBlockUserData*>& textBlockUserDataList()
    {
        return m_textBlockUserDataList;
    }
    const QSet<TextBlockUserData*>& textBlockUserDataList() const
    {
        return m_textBlockUserDataList;
    }
    const TextBlockUserData* getConstTextBlockUserData(int lineIndex) const;

    virtual bool removeTextBlockUserData(TextBlockUserData* userData);

    bool bookmarksAvailable() const;
    bool breakpointsAvailable() const;

    void callWheelEvent(QWheelEvent* e);

    void reportPositionAsGoBackNavigationItem(
        const QTextCursor& cursor, const QString& reason) const;

protected:
    struct CursorPosition
    {
        CursorPosition() : editorUID(-1){};
        CursorPosition(const QTextCursor& textCursor, int UID = -1) :
            cursor(textCursor), editorUID(UID){};

        void invalidate()
        {
            cursor = QTextCursor();
            editorUID = -1;
        }

        QTextCursor cursor;
        int editorUID; //! the UID of the editor where the cursor click comes from
    };

    CodeEditor& operator=(const CodeEditor&)
    {
        return *this;
    };

    QMenu* contextMenu() const
    {
        return m_pContextMenu;
    }

    void showTooltipDelayJobRunner(QList<QVariant> args);

    void initSettings();
    void initStyle();

    QString previousLineText() const;
    QString currentLineText() const;

    void setWhitespacesFlags(bool show);
    void updateTabStopAndIndentationWidth();

    void updateVisibleBlocks();

    void doHomeKey(QEvent* event = NULL, bool select = false);

    QTextCursor moveCursorTo(int line) const;

    virtual void reportGoBackNavigationCursorMovement(
        const CursorPosition& cursor, const QString& origin) const;

    virtual void contextMenuAboutToShow(int contextMenuLine);

    virtual void resizeEvent(QResizeEvent* e);
    virtual void closeEvent(QCloseEvent* e);
    virtual void keyPressEvent(QKeyEvent* e);
    virtual void keyReleaseEvent(QKeyEvent* e);
    virtual void mouseDoubleClickEvent(QMouseEvent* e);
    virtual void mousePressEvent(QMouseEvent* e);
    virtual void mouseReleaseEvent(QMouseEvent* e);
    virtual void mouseMoveEvent(QMouseEvent* e);
    virtual void showEvent(QShowEvent* e);
    virtual void paintEvent(QPaintEvent* e);
    virtual void wheelEvent(QWheelEvent* e);
    virtual void contextMenuEvent(QContextMenuEvent* e);

    virtual void focusInEvent(QFocusEvent* e);
    virtual void focusOutEvent(QFocusEvent* e);

    virtual bool eventFilter(QObject* obj, QEvent* e);

    virtual bool keyPressInternalEvent(QKeyEvent* e)
    {
        return true;
    };

private:
    struct FindOptions
    {
        FindOptions() : valid(false), forward(true){};
        bool valid;
        QString expr;
        bool re;
        bool cs;
        bool wo;
        bool wrap;
        bool forward;
        bool show;
    };

    FindOptions m_lastFindOptions;
    bool m_showCtxMenu;
    int m_defaultFontSize;
    bool m_useSpacesInsteadOfTabs;
    QColor m_whitespacesForeground;
    QColor m_selBackground;
    QColor m_selForeground;
    QColor m_background;
    QColor m_foreground;
    bool m_showWhitespaces;
    int m_tabLength;
    int m_zoomLevel;
    int m_fontSize;
    QString m_fontFamily;
    bool m_selectLineOnCopyEmpty;
    QString m_wordSeparators;
    QPoint m_lastMousePos;
    int m_prevTooltipBlockNbr;
    int m_indentationBarWidth;
    int m_minLineJumpsForGoBackNavigationReport;

    EdgeMode m_edgeMode;
    int m_edgeColumn;
    QColor m_edgeColor;

    bool m_showIndentationGuides;
    QColor m_indentationGuidesColor;

    bool m_redoAvailable;
    bool m_undoAvailable;

    // flags/working variables
    QList<VisibleBlock> m_visibleBlocks;
    QSet<TextBlockUserData*> m_textBlockUserDataList;

    QMenu* m_pContextMenu;

    PanelsManager* m_pPanels;
    TextDecorationsManager* m_pDecorations;
    ModesManager* m_pModes;

    DelayJobRunnerBase* m_pTooltipsRunner;

private slots:
    void emitDirtyChanged(bool state);
    void setUndoAvailable(bool available);
    void setRedoAvailable(bool available);

signals:
    void dirtyChanged(bool state); // Signal emitted when the dirty state changed
    void painted(QPaintEvent* e);
    void keyPressed(QKeyEvent* e);
    void keyReleased(QKeyEvent* e);
    void postKeyPressed(QKeyEvent* e); // Signal emitted at the end of the key_pressed event
    void mouseDoubleClicked(
        QMouseEvent* e); // Signal emitted when a mouse double click event occured
    void mousePressed(QMouseEvent* e); // Signal emitted when a mouse button is pressed
    void mouseReleased(QMouseEvent* e); // Signal emitted when a key is released
    void mouseMoved(QMouseEvent* e); // Signal emitted when the mouse_moved
    void mouseWheelActivated(QWheelEvent* e);

    void focusedIn(QFocusEvent* e); // Signal emitted when focusInEvent is is called

    void indentRequested(); // Signal emitted when the user press the TAB key
    void unindentRequested(); // Signal emitted when the user press the BACK-TAB (Shift+TAB) key

    void updateRequest();

    void updateActions();

    void newTextSet(); //!< Signal emitted when a new text is set on the widget
};

} // end namespace ito

#endif
