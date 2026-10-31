/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#include "projectTreeModel.h"

#include "../helper/editorFileType.h"

#include <qdir.h>
#include <qicon.h>
#include <qobject.h>

#include <algorithm>

namespace ito {

//-------------------------------------------------------------------------------------
ProjectTreeModel::Node::~Node()
{
    qDeleteAll(children);
}

//-------------------------------------------------------------------------------------
int ProjectTreeModel::Node::row() const
{
    return parent ? parent->children.indexOf(const_cast<Node*>(this)) : -1;
}

//-------------------------------------------------------------------------------------
ProjectTreeModel::ProjectTreeModel(QObject* parent) :
    QAbstractItemModel(parent), m_showHidden(false), m_changeInProgress(false),
    m_pWatcher(new QFileSystemWatcher(this))
{
    connect(
        m_pWatcher,
        &QFileSystemWatcher::directoryChanged,
        this,
        &ProjectTreeModel::directoryChanged);
}

//-------------------------------------------------------------------------------------
ProjectTreeModel::~ProjectTreeModel()
{
    qDeleteAll(m_roots);
}

//-------------------------------------------------------------------------------------
void ProjectTreeModel::setProject(const ProjectFile& project)
{
    setExcludes(project.excludes());
    setRootFolders(project.folders());
}

//-------------------------------------------------------------------------------------
void ProjectTreeModel::setRootFolders(const QList<ProjectFolder>& folders)
{
    m_changeInProgress = true;
    beginResetModel();

    qDeleteAll(m_roots);
    m_roots.clear();

    if (!m_pWatcher->directories().isEmpty())
    {
        m_pWatcher->removePaths(m_pWatcher->directories());
    }

    foreach (const ProjectFolder& f, folders)
    {
        Node* node = new Node();
        node->path = ProjectFile::normalize(f.path);
        node->name = f.name.isEmpty() ? QFileInfo(node->path).fileName() : f.name;

        if (node->name.isEmpty())
        {
            node->name = node->path; // e.g. a drive root
        }

        node->isDir = true;
        node->isRoot = true;
        m_roots.append(node);
    }

    endResetModel();
    m_changeInProgress = false;
}

//-------------------------------------------------------------------------------------
void ProjectTreeModel::clear()
{
    setRootFolders(QList<ProjectFolder>());
}

//-------------------------------------------------------------------------------------
QList<ProjectFolder> ProjectTreeModel::rootFolders() const
{
    QList<ProjectFolder> folders;

    foreach (const Node* n, m_roots)
    {
        ProjectFolder f;
        f.path = n->path;
        f.name = n->name;
        folders.append(f);
    }

    return folders;
}

//-------------------------------------------------------------------------------------
/*static*/ QList<QRegularExpression> ProjectTreeModel::toRegularExpressions(
    const QStringList& patterns)
{
    QList<QRegularExpression> result;

    foreach (const QString& p, patterns)
    {
        const QString pattern = p.trimmed();

        if (!pattern.isEmpty())
        {
            // own wildcard conversion ('*' and '?'), since the Qt helpers for this
            // only exist since Qt 5.12
            QString regex = "\\A";

            foreach (const QChar& c, pattern)
            {
                if (c == '*')
                {
                    regex += ".*";
                }
                else if (c == '?')
                {
                    regex += ".";
                }
                else
                {
                    regex += QRegularExpression::escape(QString(c));
                }
            }

            regex += "\\z";
            result.append(QRegularExpression(regex, QRegularExpression::CaseInsensitiveOption));
        }
    }

    return result;
}

//-------------------------------------------------------------------------------------
void ProjectTreeModel::setExcludes(const QStringList& patterns)
{
    m_excludes = patterns;
    m_excludeRegex = toRegularExpressions(patterns);
    refresh();
}

//-------------------------------------------------------------------------------------
void ProjectTreeModel::setNameFilters(const QStringList& patterns)
{
    m_nameFilters = patterns;
    m_nameFilterRegex = toRegularExpressions(patterns);
    refresh();
}

//-------------------------------------------------------------------------------------
void ProjectTreeModel::setShowHiddenFiles(bool show)
{
    if (m_showHidden != show)
    {
        m_showHidden = show;
        refresh();
    }
}

//-------------------------------------------------------------------------------------
ProjectTreeModel::Node* ProjectTreeModel::nodeFromIndex(const QModelIndex& index) const
{
    if (!index.isValid())
    {
        return nullptr;
    }

    return static_cast<Node*>(index.internalPointer());
}

//-------------------------------------------------------------------------------------
QModelIndex ProjectTreeModel::indexFromNode(Node* node) const
{
    if (!node)
    {
        return QModelIndex();
    }

    const int row = node->isRoot ? m_roots.indexOf(node) : node->row();
    return row >= 0 ? createIndex(row, 0, node) : QModelIndex();
}

//-------------------------------------------------------------------------------------
QString ProjectTreeModel::filePath(const QModelIndex& index) const
{
    const Node* n = nodeFromIndex(index);
    return n ? n->path : QString();
}

//-------------------------------------------------------------------------------------
QFileInfo ProjectTreeModel::fileInfo(const QModelIndex& index) const
{
    return QFileInfo(filePath(index));
}

//-------------------------------------------------------------------------------------
bool ProjectTreeModel::isDir(const QModelIndex& index) const
{
    const Node* n = nodeFromIndex(index);
    return n ? n->isDir : false;
}

//-------------------------------------------------------------------------------------
ProjectTreeModel::NodeType ProjectTreeModel::nodeType(const QModelIndex& index) const
{
    const Node* n = nodeFromIndex(index);

    if (!n || !n->isDir)
    {
        return FileNode;
    }

    return n->isRoot ? RootFolderNode : FolderNode;
}

//-------------------------------------------------------------------------------------
QModelIndex ProjectTreeModel::indexForPath(const QString& path, bool load)
{
    const QString abs = ProjectFile::normalize(path);

    if (abs.isEmpty())
    {
        return QModelIndex();
    }

    // the longest matching root wins (roots may be nested)
    Node* best = nullptr;

    foreach (Node* r, m_roots)
    {
        if ((abs == r->path || abs.startsWith(r->path + "/")) &&
            (!best || r->path.size() > best->path.size()))
        {
            best = r;
        }
    }

    if (!best)
    {
        return QModelIndex();
    }

    Node* node = best;

    if (abs != best->path)
    {
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
        const QStringList parts = abs.mid(best->path.size() + 1).split('/', Qt::SkipEmptyParts);
#else
        const QStringList parts = abs.mid(best->path.size() + 1).split('/', QString::SkipEmptyParts);
#endif

        foreach (const QString& part, parts)
        {
            if (!node->populated)
            {
                if (!load)
                {
                    return QModelIndex();
                }

                populate(node);
            }

            Node* next = nullptr;

            foreach (Node* c, node->children)
            {
                if (c->name == part)
                {
                    next = c;
                    break;
                }
            }

            if (!next)
            {
                return QModelIndex();
            }

            node = next;
        }
    }

    return indexFromNode(node);
}

//-------------------------------------------------------------------------------------
QModelIndex ProjectTreeModel::index(int row, int column, const QModelIndex& parent) const
{
    if (column != 0 || row < 0)
    {
        return QModelIndex();
    }

    if (!parent.isValid())
    {
        return row < m_roots.size() ? createIndex(row, 0, m_roots[row]) : QModelIndex();
    }

    const Node* p = nodeFromIndex(parent);

    if (!p || row >= p->children.size())
    {
        return QModelIndex();
    }

    return createIndex(row, 0, p->children[row]);
}

//-------------------------------------------------------------------------------------
QModelIndex ProjectTreeModel::parent(const QModelIndex& index) const
{
    const Node* n = nodeFromIndex(index);

    if (!n || n->isRoot || !n->parent)
    {
        return QModelIndex();
    }

    return indexFromNode(n->parent);
}

//-------------------------------------------------------------------------------------
int ProjectTreeModel::rowCount(const QModelIndex& parent) const
{
    if (parent.column() > 0)
    {
        return 0;
    }

    if (!parent.isValid())
    {
        return m_roots.size();
    }

    const Node* p = nodeFromIndex(parent);
    return p ? p->children.size() : 0;
}

//-------------------------------------------------------------------------------------
int ProjectTreeModel::columnCount(const QModelIndex& /*parent*/) const
{
    return 1;
}

//-------------------------------------------------------------------------------------
bool ProjectTreeModel::hasChildren(const QModelIndex& parent) const
{
    if (!parent.isValid())
    {
        return !m_roots.isEmpty();
    }

    const Node* p = nodeFromIndex(parent);

    if (!p || !p->isDir)
    {
        return false;
    }

    // an unread folder may have children; show the expand indicator
    return p->populated ? !p->children.isEmpty() : true;
}

//-------------------------------------------------------------------------------------
bool ProjectTreeModel::canFetchMore(const QModelIndex& parent) const
{
    const Node* p = nodeFromIndex(parent);
    return p && p->isDir && !p->populated;
}

//-------------------------------------------------------------------------------------
void ProjectTreeModel::fetchMore(const QModelIndex& parent)
{
    Node* p = nodeFromIndex(parent);

    if (p && p->isDir && !p->populated)
    {
        populate(p);
    }
}

//-------------------------------------------------------------------------------------
Qt::ItemFlags ProjectTreeModel::flags(const QModelIndex& index) const
{
    if (!index.isValid())
    {
        return Qt::NoItemFlags;
    }

    return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

//-------------------------------------------------------------------------------------
QVariant ProjectTreeModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (section == 0 && orientation == Qt::Horizontal && role == Qt::DisplayRole)
    {
        return tr("Name");
    }

    return QVariant();
}

//-------------------------------------------------------------------------------------
QVariant ProjectTreeModel::data(const QModelIndex& index, int role) const
{
    const Node* n = nodeFromIndex(index);

    if (!n)
    {
        return QVariant();
    }

    switch (role)
    {
    case Qt::DisplayRole:
        return n->name;
    case Qt::ToolTipRole:
        return QDir::toNativeSeparators(n->path);
    case Qt::DecorationRole:
        return icon(n);
    case FilePathRole:
        return n->path;
    case NodeTypeRole:
        return (int)nodeType(index);
    case IsDirRole:
        return n->isDir;
    default:
        return QVariant();
    }
}

//-------------------------------------------------------------------------------------
QIcon ProjectTreeModel::icon(const Node* node) const
{
    if (node->isDir)
    {
        return m_iconProvider.icon(QFileIconProvider::Folder);
    }

    const EditorFileType::Type type = EditorFileType::fromFilename(node->path);

    if (type != EditorFileType::Unknown)
    {
        return EditorFileType::icon(type);
    }

    return m_iconProvider.icon(QFileInfo(node->path));
}

//-------------------------------------------------------------------------------------
/*static*/ bool ProjectTreeModel::lessThan(const Node* a, const Node* b)
{
    if (a->isDir != b->isDir)
    {
        return a->isDir; // folders first
    }

    const int cmp = QString::compare(a->name, b->name, Qt::CaseInsensitive);

    if (cmp != 0)
    {
        return cmp < 0;
    }

    return a->name < b->name; // stable for names that only differ in case
}

//-------------------------------------------------------------------------------------
bool ProjectTreeModel::isExcluded(const QFileInfo& fi) const
{
    const QString name = fi.fileName();

    foreach (const QRegularExpression& re, m_excludeRegex)
    {
        if (re.match(name).hasMatch())
        {
            return true;
        }
    }

    if (!fi.isDir() && !m_nameFilterRegex.isEmpty())
    {
        bool accepted = false;

        foreach (const QRegularExpression& re, m_nameFilterRegex)
        {
            if (re.match(name).hasMatch())
            {
                accepted = true;
                break;
            }
        }

        if (!accepted)
        {
            return true;
        }
    }

    return false;
}

//-------------------------------------------------------------------------------------
//! reads the entries of the folder of parent (sorted, filtered) as new, unlinked nodes
QList<ProjectTreeModel::Node*> ProjectTreeModel::readFolder(Node* parent) const
{
    QList<Node*> nodes;
    QDir dir(parent->path);
    QDir::Filters filters = QDir::AllEntries | QDir::NoDotAndDotDot | QDir::System;

    if (m_showHidden)
    {
        filters |= QDir::Hidden;
    }

    foreach (const QFileInfo& fi, dir.entryInfoList(filters, QDir::NoSort))
    {
        if (isExcluded(fi))
        {
            continue;
        }

        Node* n = new Node();
        n->path = parent->path + "/" + fi.fileName();
        n->name = fi.fileName();
        n->isDir = fi.isDir();
        n->parent = parent;
        nodes.append(n);
    }

    std::sort(nodes.begin(), nodes.end(), lessThan);
    return nodes;
}

//-------------------------------------------------------------------------------------
void ProjectTreeModel::populate(Node* node)
{
    if (!node || !node->isDir || node->populated || m_changeInProgress)
    {
        return;
    }

    QList<Node*> entries = readFolder(node);
    const QModelIndex parentIndex = indexFromNode(node);

    if (!entries.isEmpty())
    {
        m_changeInProgress = true;
        beginInsertRows(parentIndex, 0, entries.size() - 1);
        node->children = entries;
        endInsertRows();
        m_changeInProgress = false;
    }

    node->populated = true;
    watch(node);
    emit folderLoaded(node->path);
}

//-------------------------------------------------------------------------------------
void ProjectTreeModel::refresh(const QModelIndex& index)
{
    if (!index.isValid())
    {
        foreach (Node* r, m_roots)
        {
            refreshAll(r);
        }
    }
    else
    {
        refreshAll(nodeFromIndex(index));
    }
}

//-------------------------------------------------------------------------------------
//! refreshes node and all its loaded sub folders
void ProjectTreeModel::refreshAll(Node* node)
{
    if (!node || !node->isDir || !node->populated)
    {
        return;
    }

    refreshNode(node);

    foreach (Node* c, node->children)
    {
        refreshAll(c);
    }
}

//-------------------------------------------------------------------------------------
//! re-reads one folder and applies the difference to the children (insert / remove rows)
void ProjectTreeModel::refreshNode(Node* node)
{
    if (!node || !node->isDir || !node->populated || m_changeInProgress)
    {
        return;
    }

    m_changeInProgress = true;

    const QModelIndex parentIndex = indexFromNode(node);
    QList<Node*> fresh = readFolder(node);

    // 1. remove children that no longer exist (or changed their type)
    for (int i = node->children.size() - 1; i >= 0; --i)
    {
        Node* old = node->children[i];
        bool found = false;

        foreach (const Node* f, fresh)
        {
            if (f->name == old->name && f->isDir == old->isDir)
            {
                found = true;
                break;
            }
        }

        if (!found)
        {
            beginRemoveRows(parentIndex, i, i);
            node->children.removeAt(i);
            endRemoveRows();
            unwatch(old);
            delete old;
        }
    }

    // 2. insert new entries at their sorted position
    foreach (Node* f, fresh)
    {
        bool exists = false;

        foreach (const Node* c, node->children)
        {
            if (c->name == f->name && c->isDir == f->isDir)
            {
                exists = true;
                break;
            }
        }

        if (exists)
        {
            delete f;
            continue;
        }

        int pos = 0;

        while (pos < node->children.size() && lessThan(node->children[pos], f))
        {
            ++pos;
        }

        beginInsertRows(parentIndex, pos, pos);
        node->children.insert(pos, f);
        endInsertRows();
    }

    m_changeInProgress = false;
    emit folderLoaded(node->path);
}

//-------------------------------------------------------------------------------------
void ProjectTreeModel::directoryChanged(const QString& path)
{
    QModelIndex idx = indexForPath(path, false);

    if (idx.isValid())
    {
        refreshNode(nodeFromIndex(idx));
    }
    else
    {
        m_pWatcher->removePath(path);
    }
}

//-------------------------------------------------------------------------------------
void ProjectTreeModel::watch(Node* node)
{
    if (node && node->isDir && QFileInfo(node->path).isDir() &&
        !m_pWatcher->directories().contains(node->path))
    {
        m_pWatcher->addPath(node->path);
    }
}

//-------------------------------------------------------------------------------------
//! stops watching node and all its sub folders
void ProjectTreeModel::unwatch(Node* node)
{
    if (!node || !node->isDir)
    {
        return;
    }

    if (m_pWatcher->directories().contains(node->path))
    {
        m_pWatcher->removePath(node->path);
    }

    foreach (Node* c, node->children)
    {
        unwatch(c);
    }
}

} // namespace ito
