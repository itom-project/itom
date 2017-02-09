#include <qcoreapplication.h>
#include "gtest/gtest.h"

#include "../AddInManager/addInManager.h"


TEST(AddInManagerTest, General)
{
    int argc = 0;
    QCoreApplication a(argc, NULL);

    //ito::AddInManager* aim = ito::AddInManager::createInstance("", NULL, NULL, NULL); // &a);
    //EXPECT_NE(aim, NULL);
    //AddInManager::closeInstance();

    ito::AddInManager* aim = new ito::AddInManager("", NULL, NULL, NULL); // &a);
    EXPECT_FALSE(aim == NULL);
    aim->closeInstance();
}

