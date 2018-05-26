#ifndef UTILS_H
#define UTILS_H

#include <qcolor.h>
#include <qtextedit.h>
#include <qchar.h>

namespace ito {

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
    Helps retrieving the various part of the user state bitmask.
   
    This helper should be used to replace calls to
    
    ``QTextBlock.setUserState``/``QTextBlock.getUserState`` as well as
    ``QSyntaxHighlighter.setCurrentBlockState``/
    ``QSyntaxHighlighter.currentBlockState`` and
    ``QSyntaxHighlighter.previousBlockState``.
    
    The bitmask is made up of the following fields:

        - bit0 -> bit26: User state (for syntax highlighting)
        - bit26: fold trigger state
        - bit27-bit29: fold level (8 level max)
        - bit30: fold trigger flag
        - bit0 -> bit15: 16 bits for syntax highlighter user state (
          for syntax highlighting)
        - bit16-bit25: 10 bits for the fold level (1024 levels)
        - bit26: 1 bit for the fold trigger flag (trigger or not trigger)
        - bit27: 1 bit for the fold trigger state (expanded/collapsed)
    */
    class TextBlockHelper
    {
    public:
        static int getState(const QTextBlock &block);
        static void setState(QTextBlock &block, int state);
        static int getFoldLvl(const QTextBlock &block);
        static void setFoldLvl(QTextBlock &block, int val);
        static bool isFoldTrigger(const QTextBlock &block);
        static void setFoldTrigger(QTextBlock &block, int val);
        static bool isCollapsed(const QTextBlock &block);
        static void setCollapsed(QTextBlock &block, int val);
    };

    /*
    Return color that is lighter or darker than the base color.*/
    QColor driftColor(const QColor &baseColor, int factor = 110);

    QList<ParenthesisInfo> listSymbols(CodeEditor *editor, const QTextBlock &block, const char* character);
    void getBlockSymbolData(CodeEditor *editor, const QTextBlock &block, QList<ParenthesisInfo> &parentheses, QList<ParenthesisInfo> &squareBrackets, QList<ParenthesisInfo> &braces);

    QString lstrip(const QString &string);
    QString rstrip(const QString &string);
    QString strip(const QString &string);
    int numlines(const QString &string);

    
};

} //end namespace ito

#endif