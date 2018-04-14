#include "textDecorationsManager.h"

#include "codeEditor.h"
#include "panel.h"


#include <assert.h>
#include <vector>



//---------------------------------------------------------------------
//---------------------------------------------------------------------
TextDecorationsManager::TextDecorationsManager(CodeEditor *editor, QObject *parent /*= NULL*/) : 
    Manager(editor, parent)
{
}

    //---------------------------------------------------------------------
TextDecorationsManager::~TextDecorationsManager()
{
}

//---------------------------------------------------------------------
QList<QTextEdit::ExtraSelection> TextDecorationsManager::getExtraSelections() const
{
    QList<QTextEdit::ExtraSelection> s;
    for (int i = 0; i < m_decorations.size(); ++i)
    {
        if (m_decorations[i].isNull() == false)
        {
            s << *(static_cast<const QTextEdit::ExtraSelection*>(m_decorations[i].data()));
        }
    }
    return s;
}

bool sortDecorationsByDrawOrder(const TextDecoration::Ptr &a, const TextDecoration::Ptr &b)
{
    return a->drawOrder() < b->drawOrder();
}

//---------------------------------------------------------------------
/*
Adds a text decoration on a CodeEdit instance

:param decoration: Text decoration to add
:type decoration: pyqode.core.api.TextDecoration
*/
bool TextDecorationsManager::append(TextDecoration::Ptr decoration)
{
    if (m_decorations.contains(decoration))
    {
        return false;
    }

    m_decorations.append(decoration);
    std::sort(m_decorations.begin(), m_decorations.end(), sortDecorationsByDrawOrder);
    editor()->setExtraSelections(getExtraSelections());
    return true;
}


//---------------------------------------------------------------------
/*
Removes a text decoration from the editor.

:param decoration: Text decoration to remove
:type decoration: pyqode.core.api.TextDecoration
*/
bool TextDecorationsManager::remove(TextDecoration::Ptr decoration)
{
   if (m_decorations.removeOne(decoration))
   {
       editor()->setExtraSelections(getExtraSelections());
       return true;
   }
   return false;
}

//---------------------------------------------------------------------
/*
Removes all text decoration from the editor.
*/
void TextDecorationsManager::clear()
{
    m_decorations.clear();
    editor()->setExtraSelections(QList<QTextEdit::ExtraSelection>());
}