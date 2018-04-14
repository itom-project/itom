

#include <QtGui>
#include <qapplication.h>

#include "codeEditor.h"
#include "modes/caretLineHighlight.h"
#include "syntaxHighlighter/pythonSyntaxHighlighter.h"
#include "managers/modesManager.h"

int main(int argv, char **args)
{
    //Q_INIT_RESOURCE(customcompleter);

    QApplication app(argv, args);

    //TextEdit edit;
    CodeEditor editor;

    editor.resize(800, 600);
    //editor.modes()->append(Mode::Ptr(new CaretLineHighlighterMode("description of caret line highlighter mode")));
    editor.modes()->append(Mode::Ptr(new PythonSyntaxHighlighter(editor.document(), "description of PythonSyntaxHighlighter")));
    //editor.appendPlainText("\n\n\n\n\n\n\n\n\n\n");
    //editor.appendPlainText("(----(j\njj)\n)");

    editor.setWindowTitle("Code Editor Example");
    editor.show();

    return app.exec();
}

