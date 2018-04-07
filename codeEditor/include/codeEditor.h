#ifndef CODEEDITOR_H
#define CODEEDITOR_H

#include "qplaintextedit.h"
#include "qcolor.h"
#include "qset.h"
#include "qpair.h"
#include "qtextobject.h"

struct VisibleBlock
{
    int topPosition;
    int lineNumber;
    QTextBlock textBlock;
};

class PanelsManager; //forward declaration
class TextDecorationsManager; //forward declaration

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
public:
    CodeEditor(QWidget *parent = NULL, bool createDefaultActions = true);
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

    int zoomLevel() const;
    void setZoomLevel(int value);

    int tabLength() const;
    void setTabLength(int value);

    QList<VisibleBlock> visibleBlocks() const;
    bool dirty() const;

    void setMouseCursor(const QCursor &cursor);

    void cursorPosition(int &line, int &column) const;

    void setViewportMargins(int left, int top, int right, int bottom);

    PanelsManager* panels() const;
    TextDecorationsManager* decorations() const;

protected:
    void initSettings();
    void initStyle();
    void initActions(bool createStandardActions);

    void setWhitespacesFlags(bool show);

    virtual void resizeEvent(QResizeEvent *e);
    virtual void closeEvent(QCloseEvent *e);
    virtual void keyPressEvent(QKeyEvent *e);
    virtual void keyReleaseEvent(QKeyEvent *e);
    virtual void mouseDoubleClickEvent(QMouseEvent *e);
    virtual void showEvent(QShowEvent *e);

private:
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
    QByteArray m_wordSeparators;

    //flags/working variables
    bool m_cleaning;
    QSet<int> m_modifiedLines; //(line)
    bool m_dirty;
    QList<VisibleBlock> m_visibleBlocks;

    PanelsManager *m_pPanels;
    TextDecorationsManager *m_pDecorations;

private slots:
    void emitDirtyChanged(bool state);
    void onTextChanged();

signals:
    void dirtyChanged(bool state); //Signal emitted when the dirty state changed
    void painted(QPaintEvent *e);
    void keyPressed(QKeyEvent *e);
    void keyReleased(QKeyEvent *e);
    void postKeyPressed(QKeyEvent *e); // Signal emitted at the end of the key_pressed event
    void mouseDoubleClicked(QMouseEvent *e); // Signal emitted when a mouse double click event occured
    void mousePressed(QMouseEvent *e); // Signal emitted when a mouse button is pressed
    void mouseReleased(QMouseEvent *e); //Signal emitted when a key is released
    void mouseMoved(QMouseEvent *e); // Signal emitted when the mouse_moved

    void blockCountChanged();
    void updateRequest();
};

#endif