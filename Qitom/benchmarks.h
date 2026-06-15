/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include <qmap.h>
#include <qhash.h>

#include <qtextstream.h>
#include <qfile.h>
#include <qdatetime.h>
#include <qdir.h>
#include <qmutex.h>
#include <qregexp.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include "DataObject\dataobj.h"
#include "common/typeDefs.h"
#include "common/color.h"

void benchmarkTest1()
{
    int64 start, ende;
    double freq = cv::getTickFrequency();

    //1
    int size = 1000000;
    int temp;

    start = cv::getTickCount();
    std::vector<int> a1;
    a1.resize(size);
    for(int i=0;i<size;i++)
    {
        a1[i]=2;
        temp=a1[i];
    }
    a1.clear();
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    int* a2 = new int[size];
    for(int i=0;i<size;i++)
    {
        a2[i]=2;
        temp=a2[i];
    }
    delete[] a2;
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;
}

void benchmarkTest2()
{
    qDebug("benchmarkTest2");
    int64 start, ende;
    double freq = cv::getTickFrequency();


    //2
    int *test = (int*)(new cv::Mat());
    int size = 1000000;
    cv::Mat* ptr = NULL;

    start = cv::getTickCount();
    for(int i=0;i<size;i++)
    {
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    for(int i=0;i<size;i++)
    {
        ptr = (cv::Mat*)test;
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    for(int i=0;i<size;i++)
    {
        ptr = reinterpret_cast<cv::Mat*>(test);
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;


}

void benchmarkTest3()
{
    ito::DataObject *do1 = NULL; //new ito::DataObject(10000,100,100,ito::tFloat32);
    ito::DataObject *do2 = NULL;//new ito::DataObject(*do1);

    qDebug("benchmarkTest3");
    int64 start, ende;
    double freq = cv::getTickFrequency();

    start = cv::getTickCount();
    do1 = new ito::DataObject(10000,100,100,ito::tFloat32);
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    do2 = new ito::DataObject(*do1);
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    delete do2;
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    delete do1;
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    //int i=1;
};

void benchmarkTest4()
{
    int64 start, ende;
    double freq = cv::getTickFrequency();
    QString str1 = "guten tag kih ihiu oiuziuzt iztfzutfu iztuztriuz iuztiuztiuztzutut";
    QString str2 = "guten tag kih ihiu oiuziuzt iztfzutfu iztuztriuz iuztiuztiuztzutut";
    QByteArray ba1 = str1.toLatin1();
    QByteArray ba2 = str2.toLatin1();
    char *c1 = ba1.data();
    char *c2 = ba2.data();
    int num = 10000000;
    int c = -num;
    size_t size = sizeof(char) * std::min( strlen(c1),strlen(c2));

    qDebug() << "benchmarkTest4: " << num;
    c = 0;
    start = cv::getTickCount();
    for(int i = 0; i< num;i++)
    {
        if(str1 == str2) {c++;}else{c--;}
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq << " result: " << c;
    c = 0;
    start = cv::getTickCount();
    for(int i = 0; i< num;i++)
    {
        if(ba1 == ba2) {c++;}else{c--;}
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq << " result: " << c;
    c = 0;
    start = cv::getTickCount();
    for(int i = 0; i< num;i++)
    {
        if(strcmp(c1,c2)) {c++;}else{c--;}
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq << " result: " << c;
    c = 0;
    start = cv::getTickCount();
    for(int i = 0; i< num;i++)
    {
        if(memcmp(c1,c2,size)) {c++;}else{c--;}
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq << " result: " << c;

    //int i=1;
};

void benchmarkTest5()
{
    ito::DataObject *do1 = NULL; //new ito::DataObject(10000,100,100,ito::tFloat32);
    ito::DataObject *do2 = NULL;//new ito::DataObject(*do1);

    qDebug("benchmarkTest5");
    int64 start, ende;
    double freq = cv::getTickFrequency();
    size_t j = 0;

    start = cv::getTickCount();
    for (size_t i = 0 ; i < 1000000; i++)
    {
        j += i;
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    j = 0;
    start = cv::getTickCount();
    for (size_t i = 0 ; i < 1000000; ++i)
    {
        j += i;
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;
};

typedef struct
{
    union
    {
        union
        {
            struct
            {
                ito::uint8 b;
                ito::uint8 g;
                ito::uint8 r;
                ito::uint8 a;
            };
            float rgb;
        };
        ito::uint32 rgba;
    };

}
rgba32_;

void benchmarkTestColor()
{
    int64 start, ende;
    double freq = cv::getTickFrequency();
    size_t j = 0;

    ito::Rgba32 c1, c2;

    start = cv::getTickCount();
    for (size_t i = 0 ; i < 1000000; i++)
    {
        ito::Rgba32 e1;
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    for (size_t i = 0 ; i < 1000000; i++)
    {
        c1 = ito::Rgba32(12,13,14,15);
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    for (size_t i = 0 ; i < 1000000; i++)
    {
        c2 = c1;
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    for (size_t i = 0 ; i < 1000000; i++)
    {
        unsigned int argb = c2.argb();
        argb = argb+2;
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;



    start = cv::getTickCount();
    for (size_t i = 0 ; i < 1000000; i++)
    {
        rgba32_ e1;
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    rgba32_ d1, d2;
    start = cv::getTickCount();
    for (size_t i = 0 ; i < 1000000; i++)
    {
        d1.r = 13;
        d1.a = 12;
        d1.g = 14;
        d1.b = 15;
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();

    for (size_t i = 0 ; i < 1000000; i++)
    {
        d2 = d1;
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    for (size_t i = 0 ; i < 1000000; i++)
    {
        unsigned int argb = d2.rgba;
        argb = argb+2;
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;


    qDebug() << "array construction";
    start = cv::getTickCount();
    ito::Rgba32 h1[100000];
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    rgba32_ h2[100000];
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    ito::uint32 h3[100000];
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;
}

void dataObjectDStackMemoryLeak()
{
    for (int i = 0; i < 100; ++i)
    {
        qDebug() << "round " << i;
        int n = 50;
        ito::DataObject *mats = new ito::DataObject[n];
        for (int i = 0; i < n; ++i)
        {
            mats[i] = ito::DataObject(1000,1000, ito::tFloat64);
        }

        {
            ito::DataObject result = ito::DataObject::stack(mats, n);

            delete[] mats;
            mats = NULL;
        }

        qDebug() << "round " << i << " finished";
    }

}


void startBenchmarks()
{
    benchmarkTest1();
    benchmarkTest2();
    benchmarkTest3();
    benchmarkTest4();
    benchmarkTest5();
    benchmarkTestColor();
}

#endif
