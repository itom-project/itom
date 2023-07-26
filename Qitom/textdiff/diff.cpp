/* diff - compute a shortest edit script(SES) given two sequences
 * Copyright(c) 2004 Michael B. Allen <mba2000 ioplex.com>
 *
 * The MIT License
 *
 * Permission is hereby granted,free of charge,to any person obtaining a
 * copy of this software and associated documentation files(the "Software"),
 * to deal in the Software without restriction,including without limitation
 * the rights to use,copy,modify,merge,publish,distribute,sublicense,
 * and/or sell copies of the Software,and to permit persons to whom the
 * Software is furnished to do so,subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS",WITHOUT WARRANTY OF ANY KIND,EXPRESS OR
 * IMPLIED,INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,DAMAGES OR
 * OTHER LIABILITY,WHETHER IN AN ACTION OF CONTRACT,TORT OR OTHERWISE,
 * ARISING FROM,OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/* This algorithm is basically Myers' solution to SES/LCS with
 * the Hirschberg linear space refinement as described in the
 * following publication:
 *
 *   E. Myers,``An O(ND) Difference Algorithm and Its Variations,''
 *   Algorithmica 1,2(1986),251-266.
 *   http://www.cs.arizona.edu/people/gene/PAPERS/diff.ps
 *
 * This is the same algorithm used by GNU diff(1).
 */

//#include <windows.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <errno.h>

#include "diff.h"
#include "varray.h"

#define FV(k) _v(ctx,(k),0)
#define RV(k) _v(ctx,(k),1)



typedef struct
{
    void *context;
    struct varray *buf;
    struct varray *DiffEditArray;
    int DiffEditArrayIndex;
    int dmax;
} MatchContext;

typedef struct{
    int x,y,u,v;
} MiddleSnake;

static void _setv(MatchContext *ctx,int k,int r,int val)
{
    int j;
    int *i;
                /* Pack -N to N into 0 to N * 2
                 */
    j=k <= 0 ? -k * 4+r : k * 4 +(r - 2);

    i=(int *)varray_get(ctx->buf,j);
    *i=val;
}
static int _v(MatchContext *ctx,int k,int r)
{
    int j;

    j=k <= 0 ? -k * 4+r : k * 4 +(r - 2);

    return *((int *)varray_get(ctx->buf,j));
}

static int _find_middle_snake(const void *a,int aoff,int n,
        const void *b,int boff,int m,
        MatchContext *ctx,
        MiddleSnake *ms)
{
    int delta,odd,mid,d;

    delta=n - m;
    odd=delta & 1;
    mid=(n+m) / 2;
    mid += odd;

    _setv(ctx,1,0,0);
    _setv(ctx,delta - 1,1,n);

    for(d=0; d <= mid; d++)
    {
        int k,x,y;

        if((2 * d - 1) >= ctx->dmax) {
            return ctx->dmax;
        }

        for(k=d; k >= -d; k -= 2) {
            if(k==-d ||(k != d && FV(k - 1)<FV(k+1))) {
                x=FV(k+1);
            } else {
                x=FV(k - 1)+1;
            }
            y=x - k;

            ms->x=x;
            ms->y=y;
            const unsigned char *a0=(const unsigned char *)a+aoff;
            const unsigned char *b0=(const unsigned char *)b+boff;
            while(x<n && y<m && a0[x]==b0[y]) {
                x++; y++;
            }
            _setv(ctx,k,0,x);

            if(odd && k >=(delta -(d - 1)) && k <=(delta +(d - 1))) {
                if(x >= RV(k)) {
                    ms->u=x;
                    ms->v=y;
                    return 2 * d - 1;
                }
            }
        }
        for(k=d; k >= -d; k -= 2) {
            int kr=(n - m)+k;

            if(k==d ||(k != -d && RV(kr - 1)<RV(kr+1))) {
                x=RV(kr - 1);
            } else {
                x=RV(kr+1) - 1;
            }
            y=x - kr;

            ms->u=x;
            ms->v=y;
            const unsigned char *a0=(const unsigned char *)a+aoff;
            const unsigned char *b0=(const unsigned char *)b+boff;
            while(x > 0 && y > 0 && a0[x - 1]==b0[y - 1]) {
                x--; y--;
            }
            _setv(ctx,kr,1,x);

            if(!odd && kr >= -d && kr <= d) {
                if(x <= FV(kr)) {
                    ms->x=x;
                    ms->y=y;
                    return 2 * d;
                }
            }
        }
    }

    errno=EFAULT;

    return -1;
}


static void AddEdit(MatchContext *ctx,int op,int off,int len)
{
    DiffEdit *e;

    if(len==0 || ctx->DiffEditArray==NULL)
    {
        return;
    }
    /* Add an edit to the SES(or
    * coalesce if the op is the same)
    */
    e=(DiffEdit *)varray_get(ctx->DiffEditArray,ctx->DiffEditArrayIndex);
    if(e->op==op)
    {
        e->len += len;
    }else if(e)
    {
        if(e->op)
        {
            ctx->DiffEditArrayIndex++;
            e=(DiffEdit *)varray_get(ctx->DiffEditArray,ctx->DiffEditArrayIndex);
        }
        if(e)
        {
            e->op=op;
            e->off=off;
            e->len=len;
        }
    }
}

static int CalculateSES(const void *a,int aoff,int n,
        const void *b,int boff,int m,
        MatchContext *ctx)
{
    MiddleSnake ms;
    int d;

    if(n==0)
    {
        AddEdit(ctx,DIFF_INSERT,boff,m);
        d=m;
    } else if(m==0) {
        AddEdit(ctx,DIFF_DELETE,aoff,n);
        d=n;
    } else {
        /* Find the middle "snake" around which we
        * recursively solve the sub-problems.
        */
        d=_find_middle_snake(a,aoff,n,b,boff,m,ctx,&ms);
        if(d==-1) {
            return -1;
        } else if(d >= ctx->dmax) {
            return ctx->dmax;
        } else if(ctx->DiffEditArray==NULL) {
            return d;
        } else if(d > 1) {
            if(CalculateSES(a,aoff,ms.x,b,boff,ms.y,ctx)==-1) {
                return -1;
            }

            AddEdit(ctx,DIFF_MATCH,aoff+ms.x,ms.u - ms.x);

            aoff += ms.u;
            boff += ms.v;
            n -= ms.u;
            m -= ms.v;
            if(CalculateSES(a,aoff,n,b,boff,m,ctx)==-1) {
                return -1;
            }
        } else {
            int x=ms.x;
            int u=ms.u;

                 /* There are only 4 base cases when the
                  * edit distance is 1.
                  *
                  * n > m   m > n
                  *
                  *   -       |
                  *    \       \    x != u
                  *     \       \
                  *
                  *   \       \
                  *    \       \    x==u
                  *     -       |
                  */

            if(m > n) {
                if(x==u) {
                    AddEdit(ctx,DIFF_MATCH,aoff,n);
                    AddEdit(ctx,DIFF_INSERT,boff +(m - 1),1);
                } else {
                    AddEdit(ctx,DIFF_INSERT,boff,1);
                    AddEdit(ctx,DIFF_MATCH,aoff,n);
                }
            } else {
                if(x==u) {
                    AddEdit(ctx,DIFF_MATCH,aoff,m);
                    AddEdit(ctx,DIFF_DELETE,aoff +(n - 1),1);
                } else {
                    AddEdit(ctx,DIFF_DELETE,aoff,1);
                    AddEdit(ctx,DIFF_MATCH,aoff+1,m);
                }
            }
        }
    }

    return d;
}

int DiffArray(
    const void *a,int aoff,int n,
    const void *b,int boff,int m,
    void *context,int dmax,struct varray *ses,int *sn,struct varray *buf)
{
    MatchContext ctx;
    int d,x,y;
    DiffEdit *e=NULL;
    struct varray tmp;

    ctx.context=context;
    if(buf)
    {
        ctx.buf=buf;
    }else
    {
        varray_init(&tmp,sizeof(int));
        ctx.buf=&tmp;
    }
    ctx.DiffEditArray=ses;
    ctx.DiffEditArrayIndex=0;
    ctx.dmax=dmax?dmax:INT_MAX;
    if(ses&&sn)
    {
        if((e=(DiffEdit *)varray_get(ses,0))==NULL)
        {
            if(!buf)
            {
                varray_deinit(&tmp);
            }
            return -1;
        }
        e->op=0;
    }

         /* The CalculateSES function assumes the SES will begin or end with a delete
          * or insert. The following will insure this is true by eating any
          * beginning matches. This is also a quick to process sequences
          * that match entirely.
          */
    x=y=0;
    const unsigned char *a0=(const unsigned char *)a+aoff;
    const unsigned char *b0=(const unsigned char *)b+boff;
    while(x<n && y<m && a0[x]==b0[y])
    {
        x++; y++;
    }
    AddEdit(&ctx,DIFF_MATCH,aoff,x);
    if((d=CalculateSES(a,aoff+x,n-x,b,boff+y,m-y,&ctx))==-1)
    {
        if(!buf)
        {
            varray_deinit(&tmp);
        }
        return -1;
    }
    if(ses&&sn) {
        *sn=e->op ? ctx.DiffEditArrayIndex+1 : 0;
    }

    if(!buf)
    {
        varray_deinit(&tmp);
    }

    return d;
}

int GetStringSimilarity(const char *a,const char *b)
{
    int n,m,d;
    int sn,i;
    struct varray ses;

    varray_init(&ses,sizeof(DiffEdit));
    n=(int)strlen(a);
    m= (int)strlen(b);

    if ((d=DiffArray(
        a,0,n,
        b,0,m,
        NULL,0,&ses,&sn,NULL)) == -1)
    {
        varray_deinit(&ses);
        return 0;
    }

    int match_len=0;
    int unmatch_len=0;

    for (i=0; i < sn; i++) {
        DiffEdit *e=(DiffEdit *)varray_get(&ses,i);
        switch (e->op)
        {
            case DIFF_MATCH:
                //fwrite(a + e->off,1,e->len,stdout);
                match_len+=e->len*2;
                break;
            case DIFF_INSERT:
                unmatch_len+=e->len;
                break;
            case DIFF_DELETE:
                unmatch_len+=e->len;
                break;
        }
    }

    /*case DIFF_MATCH:
        printf("MAT: ");
        fwrite(a + e->off, 1, e->len, stdout);
        break;
    case DIFF_INSERT:
        printf("INS: ");
        fwrite(b + e->off, 1, e->len, stdout);
        break;
    case DIFF_DELETE:
        printf("DEL: ");
        fwrite(a + e->off, 1, e->len, stdout);
        break;*/

    varray_deinit(&ses);

    if(match_len+unmatch_len==0)
    {
        return 100;
    }

    return (match_len*100)/(match_len+unmatch_len);
}

// return a vector whose size equals the length of text. Every item
// gives the line index of the character in text at the same position
QVector<int> FillLineIdxs(const QByteArray &text)
{
    QByteArrayList lines = text.split('\n');
    QVector<int> indexes;
    indexes.resize(text.size() + 1); // +1 for a newline at the end of the text
    int x = 0;
    int i;
    int lineIdx = 0;

    foreach(const auto &line, lines)
    {
        for (i = 0; i <= line.size(); ++i)
        {
            indexes[x++] = lineIdx;
        }

        lineIdx++;
    }

    return indexes;
}

//!< compares oldText with newText, where both strings can have \n as line separator.
/* This method returns the difference between two texts on a line-based level.

The mapping returns a vector, whose size corresponds to the number of lines in oldText.
Every line is set to -1, if this line is not found any more in newText, else to
the new line index [0..] in newText. mapping is only valid if this method returns true.
In case of false, no mapping could be found.

*/
bool GetLineMapping(const QByteArray &oldText, const QByteArray &newText, QVector<int> &mapping)
{
    int n, m, d;
    int sn, i;
    struct varray ses;
    const char* a = oldText.constData();
    const char* b = newText.constData();

    QVector<int> oldTextLineIdxs = FillLineIdxs(oldText);
    QVector<int> newTextLineIdxs = FillLineIdxs(newText);
    QByteArrayList oldLines = oldText.split('\n');

    varray_init(&ses, sizeof(DiffEdit));
    n = (int)strlen(a);
    m = (int)strlen(b);

    if ((d = DiffArray(
        a, 0, n,
        b, 0, m,
        NULL, 0, &ses, &sn, NULL)) == -1)
    {
        varray_deinit(&ses);
        return false;
    }

    int match_len = 0;
    int unmatch_len = 0;
    int a_pos = 0;
    int b_pos = 0;

    mapping.clear();
    mapping.fill(-1, oldLines.size());

    for (i = 0; i < sn; i++)
    {
        DiffEdit *e = (DiffEdit *)varray_get(&ses, i);

        switch (e->op)
        {
        case DIFF_MATCH:
            // both texts are equal
            //printf("MAT: ");
            //fwrite(a + e->off, 1, e->len, stdout);

            for (int i = a_pos, j = b_pos; i < a_pos + e->len; ++i, ++j)
            {
                mapping[oldTextLineIdxs[i]] = newTextLineIdxs[j];
            }

            a_pos += e->len;
            b_pos += e->len;

            break;
        case DIFF_INSERT:
            // some portion of b is added to a, a is unchanged
            //printf("INS: ");
            //fwrite(b + e->off, 1, e->len, stdout);

            b_pos += e->len;
            break;
        case DIFF_DELETE:
            // some part of a is removed
            //printf("DEL: ");
            //fwrite(a + e->off, 1, e->len, stdout);

            for (int i = a_pos; i < a_pos + e->len; ++i)
            {
                mapping[oldTextLineIdxs[i]] = -1;
            }

            a_pos += e->len;

            break;
        }
    }

    varray_deinit(&ses);

    return true;
}
