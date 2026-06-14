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

#include "gtest/gtest.h"

#include "helperCommon.h"
#include "param.h"

#include <qmap.h>
#include <qstringlist.h>


TEST(HelperCommonTest, CheckParamVector)
{
    QVector<ito::Param>* v = nullptr;
    QVector<ito::Param>* v2 = nullptr;

    EXPECT_TRUE(ito::checkParamVector(v).containsError());
    EXPECT_TRUE(ito::checkParamVectors(v2, v2, v2).containsError());

    QVector<ito::Param> w;
    QVector<ito::Param> w1, w2, w3;
    w1 << ito::Param("test", ito::ParamBase::Int, 0, nullptr, "");
    w2 << ito::Param("test", ito::ParamBase::Int, 0, nullptr, "");
    w3 << ito::Param("test", ito::ParamBase::Int, 0, nullptr, "");

    EXPECT_TRUE(ito::checkParamVector(&w) == ito::retOk);
    EXPECT_TRUE(ito::checkParamVectors(&w1, &w2, &w3) == ito::retOk);

    EXPECT_EQ(w1.size(), 0);
    EXPECT_EQ(w2.size(), 0);
    EXPECT_EQ(w3.size(), 0);
}

TEST(HelperCommonTest, GetParamByName)
{
    QVector<ito::Param> params;
    params << ito::Param("p_1", ito::ParamBase::Int, 1, nullptr, "");
    params << ito::Param("p_2", ito::ParamBase::Int, 2, nullptr, "");
    params << ito::Param("p_3", ito::ParamBase::Int, 3, nullptr, "");

    QVector<ito::ParamBase> paramsBase;

    foreach (const auto& p, params)
    {
        paramsBase << p;
    }

    ito::RetVal retval;

    // check for empty vectors
    ito::Param* param = ito::getParamByName((QVector<ito::Param>*)nullptr, "p", &retval);
    EXPECT_EQ(param, nullptr);
    EXPECT_TRUE(retval == ito::retError);

    param = ito::getParamByName((QVector<ito::Param>*)nullptr, "p", nullptr);
    EXPECT_EQ(param, nullptr);

    retval = ito::retOk;
    ito::ParamBase* paramBase =
        ito::getParamByName((QVector<ito::ParamBase>*)nullptr, "p", &retval);
    EXPECT_EQ(paramBase, nullptr);
    EXPECT_TRUE(retval == ito::retError);

    paramBase = ito::getParamByName((QVector<ito::ParamBase>*)nullptr, "p", nullptr);
    EXPECT_EQ(paramBase, nullptr);

    // check for valid vectors
    retval = ito::retOk;
    param = ito::getParamByName(&params, "P_1", &retval);
    EXPECT_EQ(param, nullptr);
    EXPECT_TRUE(retval == ito::retError);

    retval = ito::retOk;
    param = ito::getParamByName(&params, "p_1", &retval);
    EXPECT_NE(param, nullptr);
    EXPECT_EQ(param->getVal<int>(), 1);
    EXPECT_TRUE(retval == ito::retOk);

    retval = ito::retOk;
    paramBase = ito::getParamByName(&paramsBase, "P_1", &retval);
    EXPECT_EQ(paramBase, nullptr);
    EXPECT_TRUE(retval == ito::retError);

    retval = ito::retOk;
    paramBase = ito::getParamByName(&paramsBase, "p_1", &retval);
    EXPECT_NE(paramBase, nullptr);
    EXPECT_EQ(paramBase->getVal<int>(), 1);
    EXPECT_TRUE(retval == ito::retOk);
}

TEST(HelperCommonTest, ParseParamName)
{
    // valid keys
    QStringList keys;
    keys << "key"
         << "key123"
         << "a345"
         << "param_123_234";

    QString name, additionalTag;
    bool hasIndex;
    int index;

    foreach (const QString& key, keys)
    {
        ito::RetVal retVal = ito::parseParamName(key, name, hasIndex, index, additionalTag);

        EXPECT_TRUE(retVal == ito::retOk);
        EXPECT_EQ(name, key);
        EXPECT_FALSE(hasIndex);
        EXPECT_EQ(index, -1);
        EXPECT_EQ(additionalTag, "");

        // with index, no suffix
        retVal = ito::parseParamName(key + "[0]", name, hasIndex, index, additionalTag);

        EXPECT_TRUE(retVal == ito::retOk);
        EXPECT_EQ(name, key);
        EXPECT_TRUE(hasIndex);
        EXPECT_EQ(index, 0);
        EXPECT_EQ(additionalTag, "");

        retVal = ito::parseParamName(key + "[20]", name, hasIndex, index, additionalTag);

        EXPECT_TRUE(retVal == ito::retOk);
        EXPECT_EQ(name, key);
        EXPECT_TRUE(hasIndex);
        EXPECT_EQ(index, 20);
        EXPECT_EQ(additionalTag, "");

        // with suffix
        retVal = ito::parseParamName(key + ":sdf-w234_24sd", name, hasIndex, index, additionalTag);

        EXPECT_TRUE(retVal == ito::retOk);
        EXPECT_EQ(name, key);
        EXPECT_FALSE(hasIndex);
        EXPECT_EQ(index, -1);
        EXPECT_EQ(additionalTag, "sdf-w234_24sd");

        retVal =
            ito::parseParamName(key + "[0]:sdf-w234_24sd", name, hasIndex, index, additionalTag);

        EXPECT_TRUE(retVal == ito::retOk);
        EXPECT_EQ(name, key);
        EXPECT_TRUE(hasIndex);
        EXPECT_EQ(index, 0);
        EXPECT_EQ(additionalTag, "sdf-w234_24sd");

        retVal =
            ito::parseParamName(key + "[20]:sdf-w234_24sd", name, hasIndex, index, additionalTag);

        EXPECT_TRUE(retVal == ito::retOk);
        EXPECT_EQ(name, key);
        EXPECT_TRUE(hasIndex);
        EXPECT_EQ(index, 20);
        EXPECT_EQ(additionalTag, "sdf-w234_24sd");
    }

    // invalid keys
    EXPECT_TRUE(ito::parseParamName("23s", name, hasIndex, index, additionalTag) == ito::retError);
    EXPECT_TRUE(ito::parseParamName("a-b", name, hasIndex, index, additionalTag) == ito::retError);
    EXPECT_TRUE(ito::parseParamName("a?d", name, hasIndex, index, additionalTag) == ito::retError);
    EXPECT_TRUE(
        ito::parseParamName("test[a]", name, hasIndex, index, additionalTag) == ito::retError);
    EXPECT_TRUE(
        ito::parseParamName("test[2.0]", name, hasIndex, index, additionalTag) == ito::retError);
    EXPECT_TRUE(
        ito::parseParamName("test[]", name, hasIndex, index, additionalTag) == ito::retError);
    EXPECT_TRUE(
        ito::parseParamName(" test [0]:suffix", name, hasIndex, index, additionalTag) ==
        ito::retError);
}

TEST(HelperCommonTest, GetParamValue)
{
    QMap<QString, ito::Param> params;

    params["p_1"] = ito::Param("p_1", ito::ParamBase::Int, 1, nullptr, "");
    params["p_2"] = ito::Param("p_2", ito::ParamBase::Int, 2, nullptr, "");
    params["p_3"] = ito::Param("p_3", ito::ParamBase::Int, 3, nullptr, "");

    int arr[] = {1, 2, 3};
    params["array"] = ito::Param("array", ito::ParamBase::IntArray, 3, arr, "");

    ito::ByteArray stringList[] = {"hello", "test", "string"};
    params["stringlist"] = ito::Param("stringlist", ito::ParamBase::StringList, 3, stringList, "");

    QString name;
    int index;
    ito::Param value;

    EXPECT_TRUE(ito::getParamValue(nullptr, "key", value, name, index) == ito::retError);

    ito::RetVal retVal = ito::getParamValue(&params, "none", value, name, index);
    EXPECT_TRUE(retVal == ito::retError);

    // integer parameter
    retVal = ito::getParamValue(&params, "p_2", value, name, index);
    EXPECT_TRUE(retVal == ito::retOk);
    EXPECT_EQ(value, params["p_2"]);
    EXPECT_EQ(index, -1);
    EXPECT_EQ(name, "p_2");

    retVal = ito::getParamValue(&params, "p_2:suffix", value, name, index);
    EXPECT_TRUE(retVal == ito::retOk);
    EXPECT_EQ(value, params["p_2"]);
    EXPECT_EQ(index, -1);
    EXPECT_EQ(name, "p_2");

    retVal = ito::getParamValue(&params, "p_2[0]", value, name, index);
    EXPECT_TRUE(retVal == ito::retWarning); // index ignored
    EXPECT_EQ(value, params["p_2"]);
    EXPECT_EQ(index, 0);
    EXPECT_EQ(name, "p_2");

    // int array
    retVal = ito::getParamValue(&params, "array", value, name, index);
    EXPECT_TRUE(retVal == ito::retOk);
    EXPECT_EQ(value, params["array"]);
    EXPECT_EQ(index, -1);
    EXPECT_EQ(name, "array");

    retVal = ito::getParamValue(&params, "array:suffix", value, name, index);
    EXPECT_TRUE(retVal == ito::retOk);
    EXPECT_EQ(value, params["array"]);
    EXPECT_EQ(index, -1);
    EXPECT_EQ(name, "array");

    retVal = ito::getParamValue(&params, "array[1]", value, name, index);
    EXPECT_TRUE(retVal == ito::retOk);
    EXPECT_EQ(value, params["array"][1]);
    EXPECT_EQ(index, 1);
    EXPECT_EQ(name, "array");

    // string list
    retVal = ito::getParamValue(&params, "stringlist", value, name, index);
    EXPECT_TRUE(retVal == ito::retOk);
    EXPECT_EQ(value, params["stringlist"]);
    EXPECT_EQ(index, -1);
    EXPECT_EQ(name, "stringlist");

    retVal = ito::getParamValue(&params, "stringlist:suffix", value, name, index);
    EXPECT_TRUE(retVal == ito::retOk);
    EXPECT_EQ(value, params["stringlist"]);
    EXPECT_EQ(index, -1);
    EXPECT_EQ(name, "stringlist");

    retVal = ito::getParamValue(&params, "stringlist[1]", value, name, index);
    EXPECT_TRUE(retVal == ito::retOk);
    EXPECT_EQ(value, params["stringlist"][1]);
    EXPECT_EQ(index, 1);
    EXPECT_EQ(name, "stringlist");
}


TEST(HelperCommonTest, SetParamValue)
{
    QMap<QString, ito::Param> params;

    params["p_1"] = ito::Param("p_1", ito::ParamBase::Int, 1, nullptr, "");
    params["p_2"] = ito::Param("p_2", ito::ParamBase::Int, 2, nullptr, "");
    params["p_3"] = ito::Param("p_3", ito::ParamBase::Int, 3, nullptr, "");

    int arr[] = {1, 2, 3};
    int arrNew[] = { -5, 3, 0 };
    params["array"] = ito::Param("array", ito::ParamBase::IntArray, 3, arr, "");

    ito::ByteArray stringList[] = {"hello", "test", "string"};
    ito::ByteArray stringListNew[] = { "hello_", "_test", "str_ing" };
    params["stringlist"] = ito::Param("stringlist", ito::ParamBase::StringList, 3, stringList, "");

    QString name;
    int index;
    ito::Param value;

    // invalid parameter keys
    EXPECT_TRUE(ito::setParamValue(nullptr, "key", value, name, index) == ito::retError);

    ito::RetVal retVal = ito::setParamValue(&params, "none", value, name, index);
    EXPECT_TRUE(retVal == ito::retError);

    // integer parameter
    retVal = ito::setParamValue(
        &params, "p_2", ito::ParamBase("idle", ito::ParamBase::Int, 5), name, index);
    EXPECT_TRUE(retVal == ito::retOk);
    EXPECT_EQ(params["p_2"].getVal<int>(), 5);
    EXPECT_EQ(index, -1);
    EXPECT_EQ(name, "p_2");

    retVal = ito::setParamValue(
        &params, "p_2", ito::ParamBase("idle", ito::ParamBase::Double, -7.5), name, index);
    EXPECT_TRUE(retVal == ito::retError); // wrong type

    retVal = ito::setParamValue(
        &params, "p_2", ito::ParamBase("idle", ito::ParamBase::IntArray, 0, (ito::int32*)nullptr), name, index);
    EXPECT_TRUE(retVal == ito::retError); // wrong type

    retVal = ito::setParamValue(&params, "p_2:suffix", ito::ParamBase("idle", ito::ParamBase::Int, 7), name, index);
    EXPECT_TRUE(retVal == ito::retOk);
    EXPECT_EQ(params["p_2"].getVal<int>(), 7);
    EXPECT_EQ(index, -1);
    EXPECT_EQ(name, "p_2");

    retVal = ito::setParamValue(&params, "p_2[0]", ito::ParamBase("idle", ito::ParamBase::Int, 3), name, index);
    EXPECT_TRUE(retVal == ito::retWarning); // index ignored
    EXPECT_EQ(params["p_2"].getVal<int>(), 3);
    EXPECT_EQ(index, 0);
    EXPECT_EQ(name, "p_2");

    // int array
    retVal = ito::setParamValue(&params, "array", ito::ParamBase("idle", ito::ParamBase::IntArray, 3, arrNew), name, index);
    EXPECT_TRUE(retVal == ito::retOk);
    EXPECT_EQ(0, params["array"].getVal<const int*>()[2]);
    EXPECT_EQ(index, -1);
    EXPECT_EQ(name, "array");

    retVal = ito::setParamValue(&params, "array[1]", ito::ParamBase("idle", ito::ParamBase::Int, 30), name, index);
    EXPECT_TRUE(retVal == ito::retOk);
    EXPECT_EQ(30, params["array"][1].getVal<int>());
    EXPECT_EQ(index, 1);
    EXPECT_EQ(name, "array");

    // string list
    retVal = ito::setParamValue(&params, "stringlist", ito::ParamBase("idle", ito::ParamBase::StringList, 3, stringListNew), name, index);
    EXPECT_TRUE(retVal == ito::retOk);
    EXPECT_EQ("str_ing", params["stringlist"].getVal<const ito::ByteArray*>()[2]);
    EXPECT_EQ(index, -1);
    EXPECT_EQ(name, "stringlist");

    retVal = ito::setParamValue(&params, "stringlist[1]", ito::ParamBase("idle", ito::ParamBase::String, "xyz"), name, index);
    EXPECT_TRUE(retVal == ito::retOk);
    EXPECT_STREQ("xyz", params["stringlist"][1].getVal<const char*>());
    EXPECT_EQ(index, 1);
    EXPECT_EQ(name, "stringlist");
}
