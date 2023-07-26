/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.

    This file is a port and modified version of the
    CTK Common Toolkit (http://www.commontk.org)
*********************************************************************** */

// Qt includes
#include <QDebug>
#include <QDir>
#include <QRegularExpression>
#include <QString>
#include <QStringList>

#include "utils.h"

// STD includes
#include <algorithm>
#include <limits>

#ifdef _MSC_VER
  #pragma warning(disable: 4996)
#endif

//------------------------------------------------------------------------------
void ctk::qListToSTLVector(const QStringList& list,
                                 std::vector<char*>& vector)
{
  // Resize if required
  if (list.count() != static_cast<int>(vector.size()))
    {
    vector.resize(list.count());
    }
  for (int i = 0; i < list.count(); ++i)
    {
    // Allocate memory
    char* str = new char[list[i].size()+1];
    strcpy(str, list[i].toLatin1());
    vector[i] = str;
    }
}

//------------------------------------------------------------------------------
namespace
{
/// Convert QString to std::string
static std::string qStringToSTLString(const QString& qstring)
{
  return qstring.toStdString();
}
}

//------------------------------------------------------------------------------
void ctk::qListToSTLVector(const QStringList& list,
                                 std::vector<std::string>& vector)
{
  // To avoid unnessesary relocations, let's reserve the required amount of space
  vector.reserve(list.size());
  std::transform(list.begin(),list.end(),std::back_inserter(vector),&qStringToSTLString);
}

//------------------------------------------------------------------------------
void ctk::stlVectorToQList(const std::vector<std::string>& vector,
                                 QStringList& list)
{
  std::transform(vector.begin(),vector.end(),std::back_inserter(list),&QString::fromStdString);
}

//-----------------------------------------------------------------------------
const char *ctkNameFilterRegExp =
  "^(.*)\\(([a-zA-Z0-9_.*? +;#\\-\\[\\]@\\{\\}/!<>\\$%&=^~:\\|]*)\\)$";
const char *ctkValidWildCard =
  "^[\\w\\s\\.\\*\\_\\~\\$\\[\\]]+$";

//-----------------------------------------------------------------------------
QStringList ctk::nameFilterToExtensions(const QString& nameFilter)
{
  QRegularExpression regexp(QString::fromLatin1(ctkNameFilterRegExp));
  int i = nameFilter.indexOf(regexp);
  if (i < 0)
    {
    QRegularExpression isWildCard(QString::fromLatin1(ctkValidWildCard));
    if (nameFilter.indexOf(isWildCard) >= 0)
      {
      return QStringList(nameFilter);
      }
    return QStringList();
    }
  QStringList captured = regexp.namedCaptureGroups();
  QString f;

  if (captured.size() > 2)
  {
      f = captured[2];
  }
  else
  {
      return QStringList();
  }

#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
    return f.split(QLatin1Char(' '), Qt::SkipEmptyParts);
#else
    return f.split(QLatin1Char(' '), QString::SkipEmptyParts);
#endif
}

//-----------------------------------------------------------------------------
QStringList ctk::nameFiltersToExtensions(const QStringList& nameFilters)
{
  QStringList extensions;
  foreach(const QString& nameFilter, nameFilters)
    {
    extensions << nameFilterToExtensions(nameFilter);
    }
  return extensions;
}

//-----------------------------------------------------------------------------
QString ctk::extensionToRegExp(const QString& extension)
{
  // typically *.jpg
  QRegularExpression extensionExtractor("\\*\\.(\\w+)");
  int pos = extension.indexOf(extensionExtractor);
  if (pos < 0)
    {
    return QString();
    }

  QStringList captured = extensionExtractor.namedCaptureGroups();

  if (captured.size() > 1)
  {
      return ".*\\." + captured[1] + "?$";
  }

  return QString();
}

//-----------------------------------------------------------------------------
QRegularExpression ctk::nameFiltersToRegExp(const QStringList& nameFilters)
{
  QString pattern;
  foreach(const QString& nameFilter, nameFilters)
    {
    foreach(const QString& extension, nameFilterToExtensions(nameFilter))
      {
      QString regExpExtension = extensionToRegExp(extension);
      if (!regExpExtension.isEmpty())
        {
        if (pattern.isEmpty())
          {
          pattern = "(";
          }
        else
          {
          pattern += "|";
          }
        pattern +=regExpExtension;
        }
      }
    }
  if (pattern.isEmpty())
    {
    pattern = ".+";
    }
  else
    {
    pattern += ")";
    }
  return QRegularExpression(pattern);
}

//-----------------------------------------------------------------------------
int ctk::significantDecimals(double value, int defaultDecimals)
{
  if (value == 0.
      || qAbs(value) == std::numeric_limits<double>::infinity())
    {
    return 0;
    }
  if (value != value) // is NaN
    {
    return -1;
    }
  QString number = QString::number(value, 'f', 16);
  QString fractional = number.section('.', 1, 1);
  Q_ASSERT(fractional.length() == 16);
  QChar previous;
  int previousRepeat=0;
  bool only0s = true;
  bool isUnit = value > -1. && value < 1.;
  for (int i = 0; i < fractional.length(); ++i)
    {
    QChar digit = fractional.at(i);
    if (digit != '0')
      {
      only0s = false;
      }
    // Has the digit been repeated too many times ?
    if (digit == previous && previousRepeat == 2 &&
        !only0s)
      {
      if (digit == '0' || digit == '9')
        {
        return i - previousRepeat;
        }
      return i;
      }
    // Last digit
    if (i == fractional.length() - 1)
      {
      // If we are here, that means that the right number of significant
      // decimals for the number has not been figured out yet.
      if (previousRepeat > 2 && !(only0s && isUnit) )
        {
        return i - previousRepeat;
        }
      // If defaultDecimals has been provided, just use it.
      if (defaultDecimals >= 0)
        {
        return defaultDecimals;
        }
      return fractional.length();
      }
    // get ready for next
    if (previous != digit)
      {
      previous = digit;
      previousRepeat = 1;
      }
    else
      {
      ++previousRepeat;
      }
    }
  Q_ASSERT(false);
  return fractional.length();
}

//-----------------------------------------------------------------------------
int ctk::orderOfMagnitude(double value)
{
  value = qAbs(value);
  if (value == 0.
      || value == std::numeric_limits<double>::infinity()
      || value != value // is NaN
      || value < std::numeric_limits<double>::epsilon() // is tool small to compute
  )
    {
    return std::numeric_limits<int>::min();
    }
  double magnitude = 1.00000000000000001;
  int magnitudeOrder = 0;

  int magnitudeStep = 1;
  double magnitudeFactor = 10;

  if (value < 1.)
    {
    magnitudeOrder = -1;
    magnitudeStep = -1;
    magnitudeFactor = 0.1;
    }

  double epsilon = std::numeric_limits<double>::epsilon();
  while ( (magnitudeStep > 0 && value >= magnitude) ||
          (magnitudeStep < 0 && value < magnitude - epsilon))
    {
    magnitude *= magnitudeFactor;
    magnitudeOrder += magnitudeStep;
    }
  // we went 1 order too far, so decrement it
  return magnitudeOrder - magnitudeStep;
}

//-----------------------------------------------------------------------------
double ctk::closestPowerOfTen(double _value)
{
  const double sign = _value >= 0. ? 1 : -1;
  const double value = qAbs(_value);
  if (value == 0.
      || value == std::numeric_limits<double>::infinity()
      || value != value // is NaN
      || value < std::numeric_limits<double>::epsilon() // is denormalized
  )
    {
    return _value;
    }

  double magnitude = 1.;
  double nextMagnitude = magnitude;

  if (value >= 1.)
    {
    do
      {
      magnitude = nextMagnitude;
      nextMagnitude *= 10.;
      }
    while ( (value - magnitude)  > (nextMagnitude - value) );
    }
  else
    {
    do
      {
      magnitude = nextMagnitude;
      nextMagnitude /= 10.;
      }
    while ( (value - magnitude)  < (nextMagnitude - value) );
    }
  return magnitude * sign;
}

//-----------------------------------------------------------------------------
bool ctk::removeDirRecursively(const QString & dirName)
{
  bool result = false;
  QDir dir(dirName);

  if (dir.exists(dirName))
    {
    foreach (QFileInfo info, dir.entryInfoList(QDir::NoDotAndDotDot | QDir::System | QDir::Hidden  | QDir::AllDirs | QDir::Files, QDir::DirsFirst))
      {
      if (info.isDir())
        {
        result = ctk::removeDirRecursively(info.absoluteFilePath());
        }
      else
        {
        result = QFile::remove(info.absoluteFilePath());
        }

      if (!result)
        {
        return result;
        }
      }
    result = dir.rmdir(dirName);
    }

  return result;
}

//-----------------------------------------------------------------------------
bool ctk::copyDirRecursively(const QString &srcPath, const QString &dstPath)
{
  // See http://stackoverflow.com/questions/2536524/copy-directory-using-qt
  if (!QFile::exists(srcPath))
    {
    qCritical() << "ctk::copyDirRecursively: Failed to copy nonexistent directory" << srcPath;
    return false;
    }

  QDir srcDir(srcPath);
  if (!srcDir.relativeFilePath(dstPath).startsWith(".."))
    {
    qCritical() << "ctk::copyDirRecursively: Cannot copy directory" << srcPath << "into itself" << dstPath;
    return false;
    }


  QDir parentDstDir(QFileInfo(dstPath).path());
  if (!QFile::exists(dstPath) && !parentDstDir.mkdir(QFileInfo(dstPath).fileName()))
    {
    qCritical() << "ctk::copyDirRecursively: Failed to create destination directory" << QFileInfo(dstPath).fileName();
    return false;
    }

  foreach(const QFileInfo &info, srcDir.entryInfoList(QDir::Dirs | QDir::Files | QDir::NoDotAndDotDot))
    {
    QString srcItemPath = srcPath + "/" + info.fileName();
    QString dstItemPath = dstPath + "/" + info.fileName();
    if (info.isDir())
      {
      if (!ctk::copyDirRecursively(srcItemPath, dstItemPath))
        {
        qCritical() << "ctk::copyDirRecursively: Failed to copy files from " << srcItemPath << " into " << dstItemPath;
        return false;
        }
      }
    else if (info.isFile())
      {
      if (!QFile::copy(srcItemPath, dstItemPath))
        {
        return false;
        }
      }
    else
      {
      qWarning() << "ctk::copyDirRecursively: Unhandled item" << info.filePath();
      }
    }
  return true;
}

//-----------------------------------------------------------------------------
QString ctk::qtHandleToString(Qt::HANDLE handle)
{
  QString str;
  QTextStream s(&str);
  s << handle;
  return str;
}

//-----------------------------------------------------------------------------
qint64 ctk::msecsTo(const QDateTime& t1, const QDateTime& t2)
{
  QDateTime utcT1 = t1.toUTC();
  QDateTime utcT2 = t2.toUTC();

  return static_cast<qint64>(utcT1.daysTo(utcT2)) * static_cast<qint64>(1000*3600*24)
      + static_cast<qint64>(utcT1.time().msecsTo(utcT2.time()));
}
