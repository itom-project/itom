#ifndef UTILS_H
#define UTILS_H

#include <qcolor.h>
#include <qtextedit.h>
#include <qchar.h>

class CodeEditor;

namespace Utils
{
    /*
    Stores information about a parenthesis in a line of code.
    */
    class ParenthesisInfo
    {
    public:
        ParenthesisInfo(int pos, const QChar &chara) :
            position(pos),
            character(chara)
        {
            //: Position of the parenthesis, expressed as a number of character
            //: The parenthesis character, one of "(", ")", "{", "}", "[", "]"
        }

        int position;
        QChar character;
    };

    /*
    Return color that is lighter or darker than the base color.*/
    QColor driftColor(const QColor &baseColor, int factor = 110);

    QList<ParenthesisInfo> listSymbols(CodeEditor *editor, const QTextBlock &block, const char* character);
    void getBlockSymbolData(CodeEditor *editor, const QTextBlock &block, QList<ParenthesisInfo> &parentheses, QList<ParenthesisInfo> &squareBrackets, QList<ParenthesisInfo> &braces);

    int getFoldLvl(const QTextBlock &block);
    void setFoldLvl(QTextBlock &block, int val);
    bool isFoldTrigger(const QTextBlock &block);
    void setFoldTrigger(QTextBlock &block, int val);
    bool isCollapsed(const QTextBlock &block);
    void setCollapsed(QTextBlock &block, int val);
};

#endif