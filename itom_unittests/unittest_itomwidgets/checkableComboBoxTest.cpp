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


#include "gtest/gtest.h"

#include "checkableComboBox.h"
#include "widgetTestHelpers.h"

#include <QAbstractItemView>
#include <QSignalSpy>
#include <QStandardItemModel>
#include <QtTest/QTest>

using namespace itomWidgetsTest;

namespace {

class CheckableComboBoxTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        combo.addItems(QStringList() << "red" << "green" << "blue" << "alpha");
    }

    //! opens the popup and clicks on the given row of the list view.
    void clickItemInPopup(int row)
    {
        if (!combo.view()->isVisible())
        {
            combo.showPopup();
            ASSERT_TRUE(QTest::qWaitForWindowExposed(combo.view()->window()));
        }

        QAbstractItemView* view = combo.view();
        const QModelIndex index = view->model()->index(row, 0, combo.rootModelIndex());
        const QPoint pos = view->visualRect(index).center();

        QTest::mouseClick(view->viewport(), Qt::LeftButton, Qt::NoModifier, pos);
    }

    CheckableComboBox combo;
};

} // namespace

//--------------------------------------------------------------------------------------------------
// state logic
//--------------------------------------------------------------------------------------------------

TEST_F(CheckableComboBoxTest, InitiallyNothingIsChecked)
{
    EXPECT_EQ(combo.count(), 4);
    EXPECT_TRUE(combo.noneChecked());
    EXPECT_FALSE(combo.allChecked());
    EXPECT_TRUE(combo.getCheckedIndices().isEmpty());
}

TEST_F(CheckableComboBoxTest, SetCheckedIndicesReplacesTheSelection)
{
    combo.setCheckedIndices(QVector<int>() << 0 << 2);
    EXPECT_EQ(combo.getCheckedIndices(), QVector<int>() << 0 << 2);

    combo.setCheckedIndices(QVector<int>() << 3);
    EXPECT_EQ(combo.getCheckedIndices(), QVector<int>() << 3);
    EXPECT_FALSE(combo.noneChecked());
    EXPECT_FALSE(combo.allChecked());
}

TEST_F(CheckableComboBoxTest, SetCheckedIndicesIgnoresInvalidIndices)
{
    combo.setCheckedIndices(QVector<int>() << -1 << 1 << 99);

    EXPECT_EQ(combo.getCheckedIndices(), QVector<int>() << 1);
}

TEST_F(CheckableComboBoxTest, AllChecked)
{
    combo.setCheckedIndices(QVector<int>() << 0 << 1 << 2 << 3);

    EXPECT_TRUE(combo.allChecked());
    EXPECT_FALSE(combo.noneChecked());
}

TEST_F(CheckableComboBoxTest, SetIndexStateTogglesSingleItems)
{
    combo.setIndexState(1, true);
    combo.setIndexState(2, true);
    combo.setIndexState(1, false);

    EXPECT_EQ(combo.getCheckedIndices(), QVector<int>() << 2);
    EXPECT_EQ(combo.checkState(combo.model()->index(2, 0)), Qt::Checked);
    EXPECT_EQ(combo.checkState(combo.model()->index(1, 0)), Qt::Unchecked);
}

TEST_F(CheckableComboBoxTest, CheckedIndicesProperty)
{
    ASSERT_TRUE(combo.setProperty("checkedIndices", QVariant::fromValue(QVector<int>() << 0 << 3)));

    const QVector<int> indices = combo.property("checkedIndices").value<QVector<int>>();
    EXPECT_EQ(indices, QVector<int>() << 0 << 3);
}

TEST_F(CheckableComboBoxTest, CheckedIndexesChangedIsEmittedOncePerChange)
{
    QSignalSpy spy(&combo, SIGNAL(checkedIndexesChanged()));

    combo.setIndexState(0, true);
    EXPECT_EQ(spy.count(), 1);

    combo.setIndexState(0, true); // no change
    EXPECT_EQ(spy.count(), 1);

    combo.setIndexState(0, false);
    EXPECT_EQ(spy.count(), 2);
}

TEST_F(CheckableComboBoxTest, CustomCheckableModel)
{
    QStandardItemModel model;
    for (const char* name : {"a", "b", "c"})
    {
        model.appendRow(new QStandardItem(QString::fromLatin1(name)));
    }

    combo.setModel(&model);
    combo.setCheckableModel(&model);
    combo.setIndexState(2, true);

    EXPECT_EQ(combo.count(), 3);
    EXPECT_EQ(combo.getCheckedIndices(), QVector<int>() << 2);
    EXPECT_EQ(model.item(2)->checkState(), Qt::Checked);
}

//--------------------------------------------------------------------------------------------------
// mouse interaction in the popup
//--------------------------------------------------------------------------------------------------

TEST_F(CheckableComboBoxTest, ClickInPopupTogglesItem)
{
    ASSERT_TRUE(showAndWait(combo, QSize(200, 30)));

    clickItemInPopup(1);
    EXPECT_EQ(combo.getCheckedIndices(), QVector<int>() << 1);

    clickItemInPopup(1);
    EXPECT_TRUE(combo.noneChecked());
}

TEST_F(CheckableComboBoxTest, PopupStaysOpenForMultipleSelection)
{
    ASSERT_TRUE(showAndWait(combo, QSize(200, 30)));

    clickItemInPopup(0);
    EXPECT_TRUE(combo.view()->isVisible());

    clickItemInPopup(2);
    clickItemInPopup(3);

    EXPECT_EQ(combo.getCheckedIndices(), QVector<int>() << 0 << 2 << 3);

    combo.hidePopup();
    EXPECT_FALSE(combo.view()->isVisible());
}

// Regression test: CheckableComboBox::eventFilter formerly toggled view()->currentIndex()
// instead of the clicked item. A click on a disabled item then toggled the previously
// current item (row 0).
TEST_F(CheckableComboBoxTest, DisabledItemCannotBeToggledByMouse)
{
    QStandardItemModel* model = qobject_cast<QStandardItemModel*>(combo.model());
    ASSERT_NE(model, nullptr);
    model->item(1)->setEnabled(false);

    ASSERT_TRUE(showAndWait(combo, QSize(200, 30)));
    clickItemInPopup(1);

    EXPECT_TRUE(combo.noneChecked());
}
