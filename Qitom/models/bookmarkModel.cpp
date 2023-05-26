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

#include "bookmarkModel.h"

#include "../AppManagement.h"

#include <algorithm>

#include <qfileinfo.h>
#include <qdir.h>
#include <qsize.h>
#include <qsettings.h>
#include <qdatastream.h>
#include <qicon.h>
#include <qsize.h>
#include <qhash.h>

namespace ito
{
    QDataStream &operator<<(QDataStream &out, const BookmarkItem &obj)
    {
        out << obj.filename << obj.lineIdx << obj.enabled;
        return out;
    }

    QDataStream &operator>>(QDataStream &in, BookmarkItem &obj)
    {
        in >> obj.filename >> obj.lineIdx >> obj.enabled;
        return in;
    }

/*!
    \class BookmarkModel
    \brief model for management of all bookmarks. This model will be displayed by a viewer-widget in the main window
*/

//-------------------------------------------------------------------------------------------------------
//! constructor
/*!
    initializes headers and its alignment
*/
BookmarkModel::BookmarkModel() : QAbstractItemModel(), m_currentIndex(-1)
{
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    // must not be called any more in Qt6, since this is automatically done then.
    qRegisterMetaTypeStreamOperators<ito::BookmarkItem>("BookmarkItem");
#else
    qRegisterMetaType<ito::BookmarkItem>("BookmarkItem");
#endif

    m_headers   << tr("Bookmarks") ;
    m_alignment << QVariant(Qt::AlignLeft);

#ifndef WIN32
    m_filenameCaseSensitivity = Qt::CaseSensitive;
#else
    m_filenameCaseSensitivity = Qt::CaseInsensitive;
#endif

    m_pBookmarkNext = new QAction(QIcon(":/bookmark/icons/bookmarkNext.png"), tr("&Next Bookmark"), this);
    connect(m_pBookmarkNext, SIGNAL(triggered()), this, SLOT(gotoNextBookmark()));

    m_pBookmarkPrevious = new QAction(QIcon(":/bookmark/icons/bookmarkPrevious.png"), tr("&Previous Bookmark"), this);
    connect(m_pBookmarkPrevious, SIGNAL(triggered()), this, SLOT(gotoPreviousBookmark()));

    m_pBookmarkClearAll = new QAction(QIcon(":/bookmark/icons/bookmarkClearAll.png"), tr("&Clear All Bookmarks"), this);
    connect(m_pBookmarkClearAll, SIGNAL(triggered()), this, SLOT(clearAllBookmarks()));

    updateActions();
}

//-------------------------------------------------------------------------------------------------------
//! destructor
BookmarkModel::~BookmarkModel()
{
    m_headers.clear();
    m_alignment.clear();
    m_bookmarks.clear();
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//! Saves the breakpoint model into the settings
/*!

*/
RetVal BookmarkModel::saveState()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    int counter = 0;
    settings.beginGroup("BookmarkModel");

    settings.beginWriteArray("bookmarkStorage");

    foreach(const BookmarkItem &item, m_bookmarks)
    {
        QFileInfo fi(item.filename);

        if (fi.exists())
        {
            settings.setArrayIndex(counter++);
            settings.setValue("item", QVariant::fromValue<BookmarkItem>(item));
        }
    }

    settings.endArray();
    settings.endGroup();

    return ito::retOk;
}

//-------------------------------------------------------------------------------------------------------
//! Restores the breakpoint model from the settings
/*!

*/
RetVal BookmarkModel::restoreState()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("BookmarkModel");

    int size = settings.beginReadArray("bookmarkStorage");
    for (int i = 0; i < size; ++i)
    {
        settings.setArrayIndex(i);
        QVariant v = settings.value("item");

        if (v.canConvert<BookmarkItem>())
        {
            BookmarkItem item = v.value<BookmarkItem>();

            QFileInfo fi(item.filename);
            if (fi.exists())
            {
                addBookmark(item);
            }
        }
    }

    settings.endArray();
    settings.endGroup();

    return ito::retOk;
}

//-------------------------------------------------------------------------------------------------------
//! adds given bookmark to model
/*!
    if added, the signal bookmarkAdded is emitted.

    \param[in] BookmarkItem to be appended
    \return retOk if this item could be appended, retError if it already exists.
*/
RetVal BookmarkModel::addBookmark(const BookmarkItem &item)
{
    if (bookmarkExists(item))
    {
        return RetVal(ito::retError, 0, "bookmark item already exists.");
    }

    // Parent does not exist yet, so create it
    beginInsertRows(QModelIndex(), m_bookmarks.size(), m_bookmarks.size() + 1);
    m_bookmarks.append(item);
    emit bookmarkAdded(item);
    endInsertRows();

    updateActions();

    return RetVal(retOk);
}

//-------------------------------------------------------------------------------------------------------
//! delete a given bookmark
/*!
    emits bookmarkDeleted if deletion has been successful.

    \param index QModelIndex of bookmark which should be deleted
    \return retOk in case of success, if QModelIndex not valid retError
*/
RetVal BookmarkModel::deleteBookmark(const BookmarkItem &item)
{
    QModelIndex index = modelIndexFromItem(item);

    if (index.isValid())
    {
        beginRemoveRows(parent(index), index.row(), index.row());
        emit bookmarkDeleted(item);
        m_bookmarks.removeAt(index.row());
        endRemoveRows();

        updateActions();
    }

    return retOk;
}

//-------------------------------------------------------------------------------------------------------
//! delete the given bookmarks
/*!
    emits bookmarkDeleted if deletion has been successful.

    \param index QModelIndex of bookmark which should be deleted
    \return retOk in case of success, if QModelIndex not valid retError
*/
RetVal BookmarkModel::deleteBookmarks(const QList<BookmarkItem> &items)
{
    foreach(const BookmarkItem &item, items)
    {
        QModelIndex index = modelIndexFromItem(item);

        if (index.isValid())
        {
            beginRemoveRows(parent(index), index.row(), index.row());
            emit bookmarkDeleted(item);
            m_bookmarks.removeAt(index.row());
            endRemoveRows();
        }
    }

    updateActions();

    return retOk;
}

//-------------------------------------------------------------------------------------------------------
//! delete all bookmarks
/*!
    \param indizes list of QModelIndex
    \return retOk
    \sa deleteBookmark
*/
RetVal BookmarkModel::deleteAllBookmarks()
{
    beginResetModel();
    foreach (const BookmarkItem &item, m_bookmarks)
    {
        if (item.isValid())
        {
            emit bookmarkDeleted(item);
        }
    }

    m_bookmarks.clear();
    endResetModel();

    updateActions();

    return retOk;
}

//-------------------------------------------------------------------------------------------------------
RetVal BookmarkModel::changeBookmark(const BookmarkItem &item, const QString &newFilename, int newLineNo)
{
    QModelIndex index = modelIndexFromItem(item);

    if (index.isValid())
    {
        m_bookmarks[index.row()].filename = newFilename;
        m_bookmarks[index.row()].lineIdx = newLineNo;

        emit dataChanged(createIndex(index.row(), 0), createIndex(index.row(), 0));

        return retOk;
    }
    else
    {
        return RetVal(retError, 0, "bookmark item does not exist");
    }
}

//-------------------------------------------------------------------------------------------------------
//! counts number of bookmarks in this model
/*!
    \return number of elements
*/
int BookmarkModel::rowCount(const QModelIndex &parent) const
{
    return m_bookmarks.count();
}

//-------------------------------------------------------------------------------------------------------
//! counts number of columns in this model (corresponds to number of header-elements)
/*!
    \return number of columns
*/
int BookmarkModel::columnCount(const QModelIndex &parent) const
{
    return m_headers.size();
}

//-------------------------------------------------------------------------------------------------------
//! overwritten data method of QAbstractItemModel
/*!
    data method will be called by View-Widget in order to fill the table.

    \param index QModelIndex of item, whose content should be returned
    \return content of desired item and column
*/
QVariant BookmarkModel::data(const QModelIndex &index, int role) const
{
    if(!index.isValid())
    {
        return QVariant();
    }

    const BookmarkItem& item = itemFromModelIndex(index);

    //security check
    if (!item.isValid() || index.column() != 0)
    {
        return QVariant();
    }

    if (role == Qt::DisplayRole)
    {
        return tr("%1, Line %2").arg(QFileInfo(item.filename).fileName()).arg(item.lineIdx + 1);
    }
    else if (role == Qt::DecorationRole)
    {
        return QIcon(":/bookmark/icons/bookmark.png");
    }
    else if (role == Qt::ToolTipRole || role == RoleFilename)
    {
        return item.filename;
    }
    else if (role == Qt::TextAlignmentRole)
    {
        return QVariant(Qt::Alignment(Qt::AlignLeft | Qt::AlignVCenter));
    }
    else if (role == RoleLineIdx)
    {
        return item.lineIdx;
    }
    else if (role == RoleEnabled)
    {
        return item.enabled;
    }

    return QVariant();
}

//-------------------------------------------------------------------------------------------------------
//! returns QModelIndex for given row and column
/*!
    \param row row of desired entry, corresponds to index in m_bookmarks list
    \param column column of desired entry
    \param parent since this model is no tree model, parent always points to a "virtual" root element
    \return empty QModelIndex if row or column are out of bound, else returns new valid QModelIndex for that combination of row and column
*/
QModelIndex BookmarkModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!parent.isValid()) //root item
    {
        if (row < 0 || column < 0 || column >= m_headers.size() || row >= m_bookmarks.count())
        {
            return QModelIndex();
        }
        else
        {
            return createIndex(row, column, (void*)NULL); //internalId of root item is NULL
        }
    }

    return QModelIndex();
}

//-------------------------------------------------------------------------------------------------------
//! returns parent of given QModelIndex
/*!
    since this model is not a tree model, returns always an empty QModelIndex
*/
QModelIndex BookmarkModel::parent(const QModelIndex &index) const
{
    return QModelIndex();
}

//-------------------------------------------------------------------------------------------------------
//! returns header element at given position
/*!
    \param section position in m_headers list
    \param orientation the model's orientation should be horizontal, no other orientation is supported
    \param role model is only prepared for DisplayRole
    \return name of header or empty QVariant value (if no header element available)
*/
QVariant BookmarkModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if( role == Qt::DisplayRole && orientation == Qt::Horizontal )
    {
        if(section >= 0 && section < m_headers.size())
        {
            return m_headers.at(section);
        }
    }

    return QVariant();
}

//-------------------------------------------------------------------------------------------------------
//! returns BookmarkItem for given QModelIndex
/*!
    \param index given QModelIndex
    \return BookmarkItem to the QModelIndex taht was given
*/
const BookmarkItem& BookmarkModel::itemFromModelIndex(const QModelIndex &index) const
{
    if (index.isValid() && !index.parent().isValid())
    {
        if (index.row() >= 0 && index.row() < m_bookmarks.size())
        {
            return m_bookmarks[index.row()];
        }
    }

    return m_invalidBookmarkItem;
}

//-------------------------------------------------------------------------------------------------------
QModelIndex BookmarkModel::modelIndexFromItem(const BookmarkItem &item) const
{
    if (item.isValid())
    {
        for (int i = 0; i < m_bookmarks.size(); ++i)
        {
            const BookmarkItem &it = m_bookmarks[i];

            if (it.enabled == item.enabled &&
                it.lineIdx == item.lineIdx &&
                QString::compare(it.filename, item.filename, m_filenameCaseSensitivity) == 0)
            {
                return createIndex(i, 0);
            }
        }
    }

    return QModelIndex();
}

//-------------------------------------------------------------------------------------------------------
bool BookmarkModel::bookmarkExists(const BookmarkItem &item) const
{
    foreach(const BookmarkItem &it, m_bookmarks)
    {
        if (it.lineIdx == item.lineIdx && \
            QString::compare(it.filename, item.filename, m_filenameCaseSensitivity) == 0 && \
            it.enabled == item.enabled)
        {
            return true;
        }
    }

    return false;
}

//-------------------------------------------------------------------------------------------------------
bool BookmarkModel::bookmarkExists(const QString &filename, int lineno) const
{
    foreach(const BookmarkItem &it, m_bookmarks)
    {
        if (it.lineIdx == lineno && \
            QString::compare(it.filename, filename, m_filenameCaseSensitivity) == 0)
        {
            return true;
        }
    }

    return false;
}

//-------------------------------------------------------------------------------------------------------
QList<BookmarkItem> BookmarkModel::getBookmarks(const QString &filenameFilter /*= QString()*/) const
{
    QList<BookmarkItem> items;

    if (filenameFilter == "")
    {
        items = m_bookmarks;
    }
    else
    {
        foreach(const BookmarkItem &item, m_bookmarks)
        {
            if (QString::compare(item.filename, filenameFilter, m_filenameCaseSensitivity) == 0)
            {
                items.append(item);
            }
        }
    }

    return items;
}

//-------------------------------------------------------------------------------------------------------
void BookmarkModel::clearAllBookmarks()
{
    deleteAllBookmarks();
}

//-------------------------------------------------------------------------------------------------------
void BookmarkModel::gotoNextBookmark()
{
    if (m_bookmarks.size() > 0)
    {
        m_currentIndex++;

        if (m_currentIndex >= m_bookmarks.size())
        {
            m_currentIndex = 0;
        }

        gotoBookmark(createIndex(m_currentIndex, 0));
    }
}

//-------------------------------------------------------------------------------------------------------
void BookmarkModel::gotoPreviousBookmark()
{
    if (m_bookmarks.size() > 0)
    {
        m_currentIndex--;

        if (m_currentIndex < 0)
        {
            m_currentIndex = m_bookmarks.size() - 1;
        }

        gotoBookmark(createIndex(m_currentIndex, 0));
    }
}

//--------------------------------------------------------------------------------------------------------
void BookmarkModel::gotoBookmark(const QModelIndex &index)
{
    BookmarkItem item = itemFromModelIndex(index);
    if (item.isValid())
    {
        emit gotoBookmark(item);
    }
}

//--------------------------------------------------------------------------------------------------------
void BookmarkModel::updateActions()
{
    m_pBookmarkClearAll->setEnabled(m_bookmarks.size() > 0);
    m_pBookmarkNext->setEnabled(m_bookmarks.size() > 0);
    m_pBookmarkPrevious->setEnabled(m_bookmarks.size() > 0);
}

} //end namespace ito
