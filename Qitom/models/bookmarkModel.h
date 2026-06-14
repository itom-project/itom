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

#ifndef BOOKMARKMODEL_H
#define BOOKMARKMODEL_H

#include "../common/sharedStructures.h"

#include <qabstractitemmodel.h>
#include <qlist.h>
#include <qaction.h>
#include <qicon.h>
#include <qstring.h>
#include <QDebug>


namespace ito {

//! item of BookmarkModel
/*!
    this struct corresponds to one item in the BookmarkModel
*/
struct BookmarkItem
{
    /*! constructor fills struct with default values */
    BookmarkItem(): filename(""), lineIdx(-1), enabled(true)  {}
    BookmarkItem(const QString &filename_, int lineidx_) : filename(filename_), lineIdx(lineidx_), enabled(true) {}
    QString filename;       /*!<  filename of corresponding python file */
    int lineIdx;            /*!<  line number */
    bool enabled;           /*!<  indicates whether breakpoint is actually enabled */

    bool isValid() const { return lineIdx != -1; }
};

} //end namespace ito

Q_DECLARE_METATYPE(ito::BookmarkItem) //must be outside of namespace

namespace ito
{

QDataStream &operator<<(QDataStream &out, const BookmarkItem &obj);
QDataStream &operator>>(QDataStream &in, BookmarkItem &obj);

class BookmarkModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    enum BookmarkRole
    {
        RoleFilename = Qt::UserRole + 1,
        RoleLineIdx = Qt::UserRole + 2,
        RoleEnabled = Qt::UserRole + 3
    };

    BookmarkModel();
    ~BookmarkModel();

    RetVal saveState();
    RetVal restoreState();

    QVariant data(const QModelIndex &index, int role) const;
    QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
    QModelIndex parent(const QModelIndex &index) const;
    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;

    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;

    RetVal addBookmark(const BookmarkItem &item);
    RetVal deleteBookmark(const BookmarkItem &item);
    RetVal deleteBookmarks(const QList<BookmarkItem> &items);
    RetVal deleteAllBookmarks();
    RetVal changeBookmark(const BookmarkItem &item, const QString &newFilename, int newLineNo);

    QList<BookmarkItem> getBookmarks(const QString &filenameFilter = QString()) const;

    QAction *bookmarkNextAction() const { return m_pBookmarkNext; }
    QAction *bookmarkPreviousAction() const { return m_pBookmarkPrevious; }
    QAction *bookmarkClearAllAction() const { return m_pBookmarkClearAll; }

    bool bookmarkExists(const BookmarkItem &item) const;
    bool bookmarkExists(const QString &filename, int lineno) const;

    void gotoBookmark(const QModelIndex &index);

protected:
    const BookmarkItem& itemFromModelIndex(const QModelIndex &index) const;
    QModelIndex modelIndexFromItem(const BookmarkItem &item) const;
    void updateActions();

private:
    QList<BookmarkItem> m_bookmarks;    /*!<  list of bookmarks (BookmarkItem) which are currently available in this application */
    QList<QString> m_headers;               /*!<  string list of names of column headers */
    QList<QVariant> m_alignment;            /*!<  list of alignments for the corresponding headers */
    Qt::CaseSensitivity m_filenameCaseSensitivity;

    int m_currentIndex;
    QAction *m_pBookmarkNext;
    QAction *m_pBookmarkPrevious;
    QAction *m_pBookmarkClearAll;
    BookmarkItem m_invalidBookmarkItem;

Q_SIGNALS:
    void bookmarkAdded(const BookmarkItem &item);   /*!<  emitted if bookmark has been added to model at position row */
    void bookmarkDeleted(const BookmarkItem &item); /*!<  emitted if bookmark has been deleted */
    void gotoBookmark(const BookmarkItem &item);    /*!< emitted if the cursor should jump to a certain bookmark */

public Q_SLOTS:
    void clearAllBookmarks();
    void gotoNextBookmark();
    void gotoPreviousBookmark();
};

} //end namespace ito


#endif
