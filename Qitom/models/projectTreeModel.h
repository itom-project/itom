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

#pragma once

#include "../organizer/projectFile.h"

#include <qabstractitemmodel.h>
#include <qfileiconprovider.h>
#include <qfileinfo.h>
#include <qfilesystemwatcher.h>
#include <qlist.h>
#include <qregularexpression.h>
#include <qstring.h>
#include <qstringlist.h>

namespace ito {

//! Tree model of the root folders of a project (project mode of the file browser).
/*!
    Every root folder of the project is a top level item. Folders are read lazily
    (canFetchMore / fetchMore) with QDir, so a project with many files stays fast.
    Loaded folders are watched with QFileSystemWatcher and refreshed on changes.

    Entries are sorted: folders first, then files, both case insensitive by name.
    Exclude patterns (wildcards, applied to the file or folder name) hide entries
    completely, name filters (wildcards) only hide files.

    The model has one column. It does not support editing, drag and drop or removal
    through the model; file operations are done directly on the file system and
    the model picks them up through the watcher (or an explicit refresh()).
*/
class ProjectTreeModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    enum NodeType
    {
        RootFolderNode = 0, //!< a root folder of the project
        FolderNode = 1,
        FileNode = 2
    };
    Q_ENUM(NodeType)

    enum Roles
    {
        FilePathRole = Qt::UserRole + 1, //!< absolute path (QString)
        NodeTypeRole = Qt::UserRole + 2, //!< NodeType (int)
        IsDirRole = Qt::UserRole + 3 //!< bool
    };

    explicit ProjectTreeModel(QObject* parent = nullptr);
    ~ProjectTreeModel();

    //! replaces the root folders, excludes and name of the model by the ones of the project
    void setProject(const ProjectFile& project);

    //! replaces the root folders (the display name of a folder is optional)
    void setRootFolders(const QList<ProjectFolder>& folders);

    //! removes all root folders
    void clear();

    QList<ProjectFolder> rootFolders() const;

    //! wildcard patterns of files and folders that are not shown (e.g. "__pycache__", "*.pyc")
    void setExcludes(const QStringList& patterns);
    QStringList excludes() const
    {
        return m_excludes;
    }

    //! wildcard patterns of files that are shown (empty list: all files)
    void setNameFilters(const QStringList& patterns);
    QStringList nameFilters() const
    {
        return m_nameFilters;
    }

    void setShowHiddenFiles(bool show);
    bool showHiddenFiles() const
    {
        return m_showHidden;
    }

    //! absolute path of the index or an empty string for an invalid index
    QString filePath(const QModelIndex& index) const;
    QFileInfo fileInfo(const QModelIndex& index) const;
    bool isDir(const QModelIndex& index) const;
    NodeType nodeType(const QModelIndex& index) const;

    //! returns the index of a loaded entry by its absolute path, or an invalid index.
    //! Folders on the way are loaded if load is true.
    QModelIndex indexForPath(const QString& path, bool load = false);

    //! re-reads the folder of the index (or all root folders for an invalid index).
    //! Only already loaded folders are re-read.
    void refresh(const QModelIndex& index = QModelIndex());

    // QAbstractItemModel
    QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;
    QModelIndex parent(const QModelIndex& index) const override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
    Qt::ItemFlags flags(const QModelIndex& index) const override;
    bool hasChildren(const QModelIndex& parent = QModelIndex()) const override;
    bool canFetchMore(const QModelIndex& parent) const override;
    void fetchMore(const QModelIndex& parent) override;

signals:
    //! emitted after a folder has been read (initially or by a refresh)
    void folderLoaded(const QString& path);

private slots:
    void directoryChanged(const QString& path);

private:
    struct Node
    {
        QString path; //!< absolute path
        QString name; //!< file name, or display name of a root folder
        bool isDir;
        bool isRoot;
        bool populated; //!< children have been read from disk
        Node* parent;
        QList<Node*> children;

        Node() : isDir(false), isRoot(false), populated(false), parent(nullptr)
        {
        }
        ~Node();

        int row() const;
    };

    Node* nodeFromIndex(const QModelIndex& index) const;
    QModelIndex indexFromNode(Node* node) const;
    QList<Node*> readFolder(Node* parent) const;
    bool isExcluded(const QFileInfo& fi) const;
    void populate(Node* node);
    void refreshNode(Node* node);
    void refreshAll(Node* node);
    void watch(Node* node);
    void unwatch(Node* node);
    QIcon icon(const Node* node) const;

    static bool lessThan(const Node* a, const Node* b);
    static QList<QRegularExpression> toRegularExpressions(const QStringList& patterns);

    QList<Node*> m_roots;
    QStringList m_excludes;
    QStringList m_nameFilters;
    QList<QRegularExpression> m_excludeRegex;
    QList<QRegularExpression> m_nameFilterRegex;
    bool m_showHidden;
    bool m_changeInProgress; //!< guards against re-entrant structure changes (e.g. fetchMore from a slot)
    QFileSystemWatcher* m_pWatcher;
    QFileIconProvider m_iconProvider;
};

} // namespace ito
