#include <qcoreapplication.h>
#include "gtest/gtest.h"

#include "../AddInManager/addInManager.h"


TEST(AddInManagerTest, General)
{
    int argc = 0;
    QCoreApplication a(argc, NULL);

    ito::AddInManager* aim = new ito::AddInManager("", NULL, NULL, NULL); // &a);
    EXPECT_FALSE(aim == NULL);

    std::cout << aim->getNumTotItems() << "\n" << std::endl;
    aim->closeInstance();
}

