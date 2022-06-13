
#include "../AddInManager/paramHelper.h"
#include "../DataObject/dataobj.h"

#include "gtest/gtest.h"

using namespace ito;


TEST(ParamHelperTest, CompareMetaParamNoMeta)
{
    // two times no meta information is equal
    RetVal ret;
    auto result = ParamHelper::compareMetaParam(nullptr, nullptr, "tmpl", "real", ret);

    EXPECT_EQ(result, tCmpEqual);
    EXPECT_EQ(ret, retOk);
};

TEST(ParamHelperTest, CompareMetaParamPartialNullptr)
{
    DoubleMeta dm(0.0, 100.0, 2.0, "category");
    // two times no meta information is equal
    RetVal ret;
    auto result = ParamHelper::compareMetaParam(&dm, nullptr, "tmpl", "real", ret);

    EXPECT_EQ(result, tCmpCompatible);
    EXPECT_EQ(ret, retOk);

    ret = retOk;
    result = ParamHelper::compareMetaParam(nullptr, &dm, "tmpl", "real", ret);

    EXPECT_EQ(result, tCmpFailed);
    EXPECT_EQ(ret, retError);
};

TEST(ParamHelperTest, CompareMetaParamUnequalType)
{
    DoubleMeta doubleMeta(0.0, 100.0, 2.0, "category");
    IntMeta intMeta(-4, 5, 3);
    StringMeta stringMeta(StringMeta::String, "hello");
    HWMeta hwMeta("DummyGrabber", "category");
    DObjMeta dObjMeta(1, 5, "category");
    CharArrayMeta charArrayMeta(0, 100, 2, 10, 20, 2);

    RetVal ret;
    auto result = ParamHelper::compareMetaParam(&doubleMeta, &intMeta, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpFailed);
    EXPECT_EQ(ret, retError);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&intMeta, &doubleMeta, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpFailed);
    EXPECT_EQ(ret, retError);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&intMeta, &hwMeta, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpFailed);
    EXPECT_EQ(ret, retError);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&dObjMeta, &stringMeta, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpFailed);
    EXPECT_EQ(ret, retError);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&charArrayMeta, &intMeta, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpFailed);
    EXPECT_EQ(ret, retError);
}


TEST(ParamHelperTest, CompareMetaDObjMeta)
{
    DObjMeta dObjMeta1(1, 5, "category1");
    DObjMeta dObjMeta2(1, 6, "category2");
    DObjMeta dObjMeta3(2, 6, "category3");
    DObjMeta dObjMeta4(1, 5, "category4");
    dObjMeta4.appendAllowedDataType(ito::tUInt8);
    dObjMeta4.appendAllowedDataType(ito::tFloat32);
    dObjMeta4.appendAllowedDataType(ito::tInt32);
    dObjMeta4.appendAllowedDataType(ito::tDateTime);
    dObjMeta4.appendAllowedDataType(ito::tInt32);

    DObjMeta dObjMeta5(1, 5, "category4");
    dObjMeta5.appendAllowedDataType(ito::tUInt8);
    dObjMeta5.appendAllowedDataType(ito::tInt32);
    dObjMeta5.appendAllowedDataType(ito::tDateTime);

    DObjMeta dObjMeta6(1, 5, "category4");
    dObjMeta6.appendAllowedDataType(ito::tComplex128);
    
    RetVal ret = retOk;
    auto result = ParamHelper::compareMetaParam(&dObjMeta1, &dObjMeta1, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpEqual);
    EXPECT_EQ(ret, retOk);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&dObjMeta1, nullptr, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpCompatible);
    EXPECT_EQ(ret, retOk);

    ret = retOk;
    result = ParamHelper::compareMetaParam(nullptr, &dObjMeta1, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpFailed);
    EXPECT_EQ(ret, retError);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&dObjMeta1, &dObjMeta2, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpCompatible);
    EXPECT_EQ(ret, retOk);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&dObjMeta2, &dObjMeta1, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpFailed);
    EXPECT_EQ(ret, retError);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&dObjMeta1, &dObjMeta3, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpFailed);
    EXPECT_EQ(ret, retError);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&dObjMeta3, &dObjMeta1, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpFailed);
    EXPECT_EQ(ret, retError);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&dObjMeta1, &dObjMeta4, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpFailed);
    EXPECT_EQ(ret, retError);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&dObjMeta4, &dObjMeta1, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpCompatible);
    EXPECT_EQ(ret, retOk);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&dObjMeta4, &dObjMeta4, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpEqual);
    EXPECT_EQ(ret, retOk);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&dObjMeta4, &dObjMeta5, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpFailed);
    EXPECT_EQ(ret, retError);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&dObjMeta5, &dObjMeta4, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpCompatible);
    EXPECT_EQ(ret, retOk);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&dObjMeta4, &dObjMeta6, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpFailed);
    EXPECT_EQ(ret, retError);

    ret = retOk;
    result = ParamHelper::compareMetaParam(&dObjMeta6, &dObjMeta4, "tmpl", "real", ret);
    EXPECT_EQ(result, tCmpFailed);
    EXPECT_EQ(ret, retError);
}

TEST(ParamHelperTest, ValidateDataObjParam)
{
    DObjMeta dobjMeta(2, 4, "category");
    dobjMeta.appendAllowedDataType(ito::tFloat32);
    dobjMeta.appendAllowedDataType(ito::tComplex128);

    auto retVal = ParamHelper::validateDObjMeta(&dobjMeta, nullptr, true);
    EXPECT_EQ(retVal, retError);

    retVal = ParamHelper::validateDObjMeta(&dobjMeta, nullptr, false);
    EXPECT_EQ(retVal, retOk);

    retVal = ParamHelper::validateDObjMeta(nullptr, nullptr, false);
    EXPECT_EQ(retVal, retOk);

    DataObject dobj;
    retVal = ParamHelper::validateDObjMeta(&dobjMeta, &dobj, false);
    EXPECT_EQ(retVal, retError);

    DataObject dobj2(50, 30, ito::tFloat32);
    retVal = ParamHelper::validateDObjMeta(&dobjMeta, &dobj2, false);
    EXPECT_EQ(retVal, retOk);

    DataObject dobj4(50, 30, ito::tFloat64);
    retVal = ParamHelper::validateDObjMeta(&dobjMeta, &dobj4, false);
    EXPECT_EQ(retVal, retError);

    int sizes[] = { 1,2,3,4,5 };
    DataObject dobj3(5, sizes, ito::tFloat32);
    retVal = ParamHelper::validateDObjMeta(&dobjMeta, &dobj3, false);
    EXPECT_EQ(retVal, retError);
}

TEST(ParamHelperTest, ConvertDObjParam)
{
    DObjMeta dobjMeta(2, 4, "category");
    dobjMeta.appendAllowedDataType(ito::tFloat32);
    dobjMeta.appendAllowedDataType(ito::tComplex128);

    DataObject dObj(2, 4, ito::tComplex128);
    ParamBase p1("param", ParamBase::DObjPtr | ParamBase::In, nullptr);
    p1.setVal<DataObject*>(&dObj);

    bool ok;
    ParamBase result = ParamHelper::convertParam(p1, ParamBase::DObjPtr, &ok);

    EXPECT_EQ(ok, true);
    EXPECT_EQ(&dObj, result.getVal<const DataObject*>());

    result = ParamHelper::convertParam(p1, ParamBase::ComplexArray, &ok);

    EXPECT_EQ(ok, false);
    EXPECT_EQ(result.getType(), 0);
}

TEST(ParamHelperTest, ValidateAndCastDObjParam)
{
    DObjMeta dobjMeta(2, 4, "category");
    dobjMeta.appendAllowedDataType(ito::tFloat32);
    dobjMeta.appendAllowedDataType(ito::tComplex128);

    DataObject dObj(2, 4, ito::tComplex128);
    ParamBase p1("param:suffix", ParamBase::DObjPtr | ParamBase::In, nullptr);
    p1.setVal<DataObject*>(&dObj);

    ParamBase p2("param[0]", ParamBase::DObjPtr | ParamBase::In, nullptr);
    p1.setVal<DataObject*>(&dObj);

    ParamBase p3("param", ParamBase::Int, 0);

    Param templateParam("tmpl", ParamBase::DObjPtr, nullptr, "");
    templateParam.setMeta(&dobjMeta, false);

    RetVal ret = ParamHelper::validateAndCastParam(templateParam, p1, false);
    EXPECT_EQ(ret, retOk);

    ret = ParamHelper::validateParam(templateParam, p1, false);
    EXPECT_EQ(ret, retOk);

    ret = ParamHelper::validateAndCastParam(templateParam, p2, false);
    EXPECT_EQ(ret, retError);

    ret = ParamHelper::validateParam(templateParam, p2, false);
    EXPECT_EQ(ret, retError);

    ret = ParamHelper::validateAndCastParam(templateParam, p3, true);
    EXPECT_EQ(ret, retError);

    ret = ParamHelper::validateParam(templateParam, p3, true);
    EXPECT_EQ(ret, retError);

    dObj = DataObject();
    p1.setVal<DataObject*>(&dObj);

    ret = ParamHelper::validateAndCastParam(templateParam, p1, false);
    EXPECT_EQ(ret, retError);

    ret = ParamHelper::validateParam(templateParam, p1, false);
    EXPECT_EQ(ret, retError);
}

TEST(ParamHelperTest, ParseParamNameTest)
{
    QString paramName;
    bool hasIndex;
    int index;
    QString additionalTag;

    RetVal ret = ParamHelper::parseParamName("p2aram_Name34", paramName, hasIndex, index, additionalTag);
    EXPECT_EQ(ret, retOk);
    EXPECT_EQ(paramName, "p2aram_Name34");
    EXPECT_FALSE(hasIndex);
    EXPECT_EQ(index, -1);
    EXPECT_EQ(additionalTag, "");

    ret = ParamHelper::parseParamName("p2aram_Name34[45]", paramName, hasIndex, index, additionalTag);
    EXPECT_EQ(ret, retOk);
    EXPECT_EQ(paramName, "p2aram_Name34");
    EXPECT_TRUE(hasIndex);
    EXPECT_EQ(index, 45);
    EXPECT_EQ(additionalTag, "");

    ret = ParamHelper::parseParamName("p2aram_Name34[45]:suffix", paramName, hasIndex, index, additionalTag);
    EXPECT_EQ(ret, retOk);
    EXPECT_EQ(paramName, "p2aram_Name34");
    EXPECT_TRUE(hasIndex);
    EXPECT_EQ(index, 45);
    EXPECT_EQ(additionalTag, "suffix");

    ret = ParamHelper::parseParamName("p2aram_Name34:suffix", paramName, hasIndex, index, additionalTag);
    EXPECT_EQ(ret, retOk);
    EXPECT_EQ(paramName, "p2aram_Name34");
    EXPECT_FALSE(hasIndex);
    EXPECT_EQ(index, -1);
    EXPECT_EQ(additionalTag, "suffix");

    ret = ParamHelper::parseParamName("2p2aram_Name34", paramName, hasIndex, index, additionalTag);
    EXPECT_EQ(ret, retError);

    ret = ParamHelper::parseParamName("p2aram Name34", paramName, hasIndex, index, additionalTag);
    EXPECT_EQ(ret, retError);

    ret = ParamHelper::parseParamName("p2aram/Name34", paramName, hasIndex, index, additionalTag);
    EXPECT_EQ(ret, retError);
}
