#include "paramMeta.h"
#include "byteArray.h"
#include "gtest/gtest.h"

using namespace ito;

template <typename _Tp> void AssignmentAndCopyConstrImpl(_Tp meta, _Tp altMeta, const char* text)
{
    ParamMeta base;
    EXPECT_NE(base, meta) << text;

    EXPECT_FALSE(meta == altMeta);
    EXPECT_TRUE(meta != altMeta);
    EXPECT_TRUE(meta.getType() == altMeta.getType());

    _Tp meta2 = meta;
    EXPECT_TRUE(meta2 == meta) << text;
    EXPECT_TRUE(meta2.getType() == meta.getType()) << text;

    _Tp meta3(meta);
    EXPECT_TRUE(meta3 == meta) << text;
    EXPECT_TRUE(meta3.getType() == meta.getType()) << text;

    meta3 = altMeta;
    EXPECT_TRUE(meta3 == altMeta) << text;
    EXPECT_TRUE(meta3.getType() == altMeta.getType()) << text;

    meta3 = meta;
    EXPECT_FALSE(meta3 == altMeta) << text;

    std::vector<ito::ParamMeta> vector;
    vector.push_back(meta);
    vector.push_back(altMeta);
    EXPECT_TRUE(vector[0].getType() == meta.getType()) << text;
    EXPECT_TRUE(vector[1].getType() == meta.getType()) << text;

    ParamMeta base2;
    EXPECT_TRUE(base2.getType() == ito::ParamMeta::rttiUnknown) << text;
    base2 = meta;
    EXPECT_TRUE(base2.getType() == meta.getType()) << text;
}

TEST(ParamMetaTest, AssignmentAndCopyConstructor)
{
    AssignmentAndCopyConstrImpl<CharMeta>(
        CharMeta(2, 40, 6, "category"), CharMeta(3, 41, 1, "category"), "CharMeta");
    AssignmentAndCopyConstrImpl<IntMeta>(
        IntMeta(2, 40, 6, "category"), IntMeta(3, 41, 2, "category"), "IntMeta");
    AssignmentAndCopyConstrImpl<DoubleMeta>(
        DoubleMeta(2.0, 40.0, 6.0, "category"),
        DoubleMeta(2.01, 40.01, 6.01, "category"),
        "DoubleMeta");
    StringMeta sm(StringMeta::Wildcard, "category");
    sm.addItem("*.jpg");
    AssignmentAndCopyConstrImpl<StringMeta>(
        sm, StringMeta(StringMeta::String, "category2"), "StringMeta");
    AssignmentAndCopyConstrImpl<HWMeta>(
        HWMeta("sdf", "category"), HWMeta("sdf", "category2"), "HWMeta");
    AssignmentAndCopyConstrImpl<DObjMeta>(
        DObjMeta(ito::tUInt8 | ito::tInt32, 2, 32, "category"),
        DObjMeta(ito::tUInt16 | ito::tInt32, 2, 32, "category"),
        "DObjMeta");
    AssignmentAndCopyConstrImpl<CharArrayMeta>(
        CharArrayMeta(2, 200, 10, 20, 50, 6, "category"),
        CharArrayMeta(3, 200, 10, 20, 50, 6, "category"),
        "CharArrayMeta");
    AssignmentAndCopyConstrImpl<IntArrayMeta>(
        IntArrayMeta(2, 200, 10, 20, 50, 6, "category"),
        IntArrayMeta(2, 200, 10, 21, 50, 6, "category"),
        "IntArrayMeta");
    AssignmentAndCopyConstrImpl<DoubleArrayMeta>(
        DoubleArrayMeta(2.0, 200.0, 10.0, 20, 50, 6, "category"),
        DoubleArrayMeta(2.0, 200.0, 10.0, 20, 50, 7, "category3"),
        "DoubleArrayMeta");
    AssignmentAndCopyConstrImpl<StringListMeta>(
        StringListMeta(StringMeta::RegExp, 3, 6, 3, "category"),
        StringListMeta(StringMeta::Wildcard, 3, 6, 3, "category"),
        "StringListMeta");

    AssignmentAndCopyConstrImpl<DoubleIntervalMeta>(
        DoubleIntervalMeta(1.0, 3.0, 0.01, 2, 6, 2, "category"),
        DoubleIntervalMeta(1.0, 3.0, 0.011, 2, 6, 2, "category"),
        "DoubleIntervalMeta");
    AssignmentAndCopyConstrImpl<RangeMeta>(
        RangeMeta(1, 3, 1, 2, 6, 2, "category"),
        RangeMeta(1, 3, 2, 2, 6, 2, "category"),
        "RangeMeta");
    AssignmentAndCopyConstrImpl<IntervalMeta>(
        IntervalMeta(1, 3, 1, 2, 6, 2, "category"),
        IntervalMeta(1, 3, 2, 2, 6, 2, "category"),
        "IntervalMeta");

    RangeMeta width(512, 1024, 64, "width");
    RangeMeta height(64, 128, 2, "height");
    RangeMeta height2(64, 256, 2, "height");
    AssignmentAndCopyConstrImpl<RectMeta>(
        RectMeta(width, height, "category"), RectMeta(width, height2, "category"), "RectMeta");
}
