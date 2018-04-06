#include "manager.h"

//---------------------------------------------------
Manager::Manager(CodeEditor *editor, QObject *parent) : m_pEditor(editor), QObject(parent)
{
}

//---------------------------------------------------
Manager::~Manager()
{
}

//---------------------------------------------------
CodeEditor* Manager::editor() const
{
    return m_pEditor;
}