#ifndef CHARBASEDFOLDDETECTOR_H
#define CHARBASEDFOLDDETECTOR_H

#include "foldDetector.h"

/*
This module contains the code folding API.
*/

class CharBasedFoldDetectorPrivate;


/*
Fold detector based on trigger charachters (e.g. a { increase fold level
    and } decrease fold level).
*/
class CharBasedFoldDetector : public FoldDetector
{
    Q_OBJECT
public:
    CharBasedFoldDetector(QChar openChars = '{', QChar closeChars = '}', QObject *parent = NULL);

    virtual ~CharBasedFoldDetector();


    virtual int detectFoldLevel(const QTextBlock &previousBlock, const QTextBlock &block);
private:
    CharBasedFoldDetectorPrivate *d_ptr;
    Q_DECLARE_PRIVATE(CharBasedFoldDetector);
};


#endif