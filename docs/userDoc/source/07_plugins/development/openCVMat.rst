.. |pt| replace:: Python   

OpenCV matrices **cv::Mat** and **cv::Mat_**
============================================

In OpenCV the main matrix class is called **Mat** and is contained in the OpenCV-namespace **cv**. This matrix is not templated but nevertheless can contain
different data types. These are indicated by a certain type-number. Additionally, OpenCV provides a templated class called **Mat_**, which is derived from
**Mat**. Internally, the template-data type is analysed and tried to convert to the corresponding type-number. If this is not possible the user-defined type-number
is assumed.

In the following section of the manual, a short introduction for the use of the class **Mat** is given.

Creating a matrix of class **cv::Mat**
--------------------------------------

//tell us something about the different possibilities how to construct a cv::Mat (different constructors, their parameters, examples...)

To create and manipulate multidimensional matrices. You can create a Mat object in multiple ways:
Lets start with two-dimensional matrices

.. code-block:: c++
    :linenos:
    
    cv::Mat A = cv::Mat(int rows, int cols, int type);
    cv::Mat A = cv::Mat(3,3,CV_32FC2);

    cv::Mat A = cv::Mat(int rows, int cols, int type, const Scalar& s);
    cv::Mat A = cv::Mat(3,3,CV_32FC3,cv::Scalar(0,0,255));

For three dimensional and multichannel images we first define their size: row and column count wise.
Then we need to specify the data type to use for storing the elements and the number of channels per matrix point.
To do this we have multiple definitions made according to the following convention.
CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]
For instance, CV_32FC3 means we use floating char types that are 32 bit long and each pixel has three items of this to form the three channels. This are predefined for up to four channel numbers. The Scalar is four element short vector. Specify this and you can initialize all matrix points with a custom value.

Create a multi-dimensional array: create a 100x100x100 8-bit array

.. code-block:: c++
    :linenos:
    
    cv::Mat A = cv::Mat(int ndims, const int* sizes, int type)
    int Sz[] = {100,100,100}; 
    cv::Mat A = cv::Mat(3, Sz, CV_8U, cv::Scalar::all(0)); 
    
It passes the number of dimensions =1 to the Mat constructor but the created array will be 2-dimensional with the number of columns set to 1. So, Mat::dimsis always >= 2 (can also be 0 when the array is empty).
Use a copy constructor or assignment operator where there can be an array or expression on the right side (seebelow). As noted in the introduction, the array assignment is an O(1) operation because it only copies the headerand increases the reference counter. The Mat::clone() method can be used to get a full (deep) copy of the array when you need it.

Another approach

.. code-block:: c++
    :linenos:

    cv::Mat A = cv::Mat(const Mat& m, const Range& rowRange, const Range& colRange)
    cv::Mat A = cv::Mat(originalMatrix, cv::Range::all(), cv::Range(1,3));

rowRange - Range of the m rows to take. As usual, the range start is inclusive and the range end is exclusive. Use Range::all() to take all the rows.
colRange - Range of the m columns to take. Use Range::all() to take all the columns.
ranges - Array of selected ranges of m along each dimensionality.
 
Possible parameter names for matrix **cv::Mat**
-------------------------------------------------

The parameters which you add to the **cv::Mat**, must have a name which fits to the following rules:

* The name starts with a lower or upper case character, hence a value between *a-z* or *A-Z*
* The first character of the name can be followed by an infinite number of alpha-numerical characters (including characters like *_* or *-*).
* To create Mat object.
    using create (nrows, ncols, type)
    where    
    
    * nrows is number of rows
    * ncols is number of columns
    
    type is specified value such as 
    
    #. CV_8UC1 means 8-bit single-channel array,
    #. CV_32FC2 means 2-channel(i.e. complex) floating-point array
    #. CV_8U - 8-bit unsigned integers ( 0..255 )
    #. CV_8S - 8-bit signed integers ( -128..127 )
    #. CV_16U - 16-bit unsigned integers ( 0..65535 )
    #. CV_16S - 16-bit signed integers ( -32768..32767 )
    #. CV_32S - 32-bit signed integers ( -2147483648..2147483647 )
    #. CV_32F - 32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
    #. CV_64F - 64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )

    The array type, use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_MAX_CN channels) matrices

.. code-block:: c++
    :linenos:

    // make 7x7 complex matrix having type of 2-channel floating point array.
    cv::Mat M(7,7,CV_32FC2);
    
If the user tries to type same rows and columns for multiplication then,

.. code-block:: c++
    :linenos:
    
    cv::Mat(int rows, int cols, int type)
    cv::Mat A = cv::Mat(3,3,CV_64FC1);
    cv::Mat B = cv::Mat(3,3,CV_64FC1);
    cv::Mat C = A.mul(B);
    
The above example's answer will show in 3*3 matrix according to the types.
But if the user types

.. code-block:: c++
    :linenos:
    
    cv::Mat(int rows, int cols, int type)
    cv::Mat A = cv::Mat(4,3,CV_64FC1);
    cv::Mat B = cv::Mat(4,3,CV_64FC1);
    cv::Mat C = A.mul(B);

Then the mul operator or '*' performs element-wise multiplication. Note that it is not a matrix multiplication.
The above example's answer will show in 4*3 matrix according to the types.


If the user tries to type different types like CV_64FC1 or CV_32FC1 in matrix then the value will be different. Also in these types values after C is kept for channels ie. 1,2,.....,n.

.. code-block:: c++
    :linenos:
    
    cv::Mat(int rows, int cols, int type)
    cv::Mat A = cv::Mat(4,3,CV_64FC1 or CV_32FC1);


If the user tries to type different rows and columns for multiplication then the error:sizes of input arguments do not match and the multiplied matrix will have the same rows and columns of any matrix.

.. code-block:: c++
    :linenos:
    
    cv::Mat(int rows, int cols, int type)
    cv::Mat A = cv::Mat(3,4,CV_64FC1);
    cv::Mat B = cv::Mat(4,3,CV_64FC1);
    cv::Mat C = A.mul(B);

The above example's answer will show in 4*3 matrix according to the types.
Also if the user tries to type different types while performing multiplication then the error:the input arrays in functions have different types, the output array type must be explicitly specified.


    
If the user uses ones command then all the elements of matrix will have value 1 and by multiplying it with any number the user will get the multiplied value.

.. code-block:: c++
    :linenos:
    
    cv::Mat F = cv::Mat::ones(3,4,CV_32FC1)*3;


If the user uses eye command then it will form identity matrix and by multiplying it with any number the user will get the multiplied value in identity matrix.

.. code-block:: c++
    :linenos:
    
    cv::Mat F = cv::Mat::eye(3,4,CV_32FC1)*6;


If the user uses zeros command then all the elements of matrix will have value 0.

.. code-block:: c++
    :linenos:
    
    cv::Mat F = cv::Mat::zeros(3,3,CV_32FC1)*3;
    
Adressing values in a matrix
----------------------------
Creating a big Matrix

.. code-block:: c++
    :linenos:
    
    cv::Mat W = cv::Mat(100,100,CV_32FC1);

Creating another header for the same matrix, this is an instant operation regardless of the matrix size.

.. code-block:: c++
    :linenos:
    
    cv::Mat X = W;

Create another header for the 3-rd row of W, no data is copied either and also created separate matrix

.. code-block:: c++
    :linenos:
    
    cv::Mat Y = X.row(3);
    cv::Mat Z = X.clone();
    
Copying the 2-nd row of X to Y, that is, copy the 2-nd row of to the 3-rd row of W.

.. code-block:: c++
    :linenos:
    
    X.row(2).copyTo(Y);

Now let W and Z share the data; after that the modified version of W is still referenced by X and Y. now make X an empty matrix (which references no memory buffers) but the modified version of W will still be referenced by Y, despite that Y is just a single row of the original W

.. code-block:: c++
    :linenos:
    
    W = Z;
    X.release();

Finally, make a full copy of Y. As a result, the big modified matrix will be deallocated, since it is not referenced by anyone

.. code-block:: c++
    :linenos:
    
    Y = Y.clone();

    
    
    
    
Shallow copy vs. deep copy
--------------------------


Creating Region Of Interest
---------------------------

#. cv::Mat::locateROI
Locates the matrix header within a parent matrix.

.. code-block:: c++
    :linenos:
    
    cv::Mat::locateROI(Size &wholeSize, cv::Point &ofs) const 

Parameters: * wholeSize - Output parameter that contains the size of the whole matrix containing *this as a part.
            * ofs - Output parameter that contains an offset of *this inside the whole matrix.
 
After you extracted a submatrix from a matrix using Mat::row(), Mat::col(), Mat::rowRange(), Mat::colRange() , and others, the resultant submatrix points just to the part of the original big matrix. However, each submatrix contains information (represented by datastart and dataend fields) that helps reconstruct the original matrix size and the position of the extracted submatrix within the original matrix. The method locateROI does exactly that.

#. cv::Mat::adjustROI
Adjusts a submatrix size and position within the parent matrix.

.. code-block:: c++
    :linenos:
    
    cv::Mat::adjustROI(int dtop, int dbottom, int dleft, int dright) 

Parameters: * dtop - Shift of the top submatrix boundary upwards.
            * dbottom - Shift of the bottom submatrix boundary downwards.
            * dleft - Shift of the left submatrix boundary to the left.
            * dright - Shift of the right submatrix boundary to the right.
 
The method is complimentary to Mat::locateROI() . The typical use of these functions is to determine the submatrix position within the parent matrix and then shift the position somehow. Typically, it can be required for filtering operations when pixels outside of the ROI should be taken into account. When all the method parameters are positive, the ROI needs to grow in all directions by the specified amount,
for example:

.. code-block:: c++
    :linenos:
    
    A.adjustROI(2, 2, 2, 2);

In this example, the matrix size is increased by 4 elements in each direction. The matrix is shifted by 2 elements to the left and 2 elements up, which brings in all the necessary pixels for the filtering with the 5x5 kernel.
adjustROI forces the adjusted ROI to be inside of the parent matrix that is boundaries of the adjusted ROI are constrained by boundaries of the parent matrix.
For example, if the submatrix A is located in the first row of a parent matrix and you called A.adjustROI(2, 2, 2, 2) then A will not be increased in the upward direction



A = cv::Mat(3,3,CV_32FC1)
B = A(cv::Range(0,1), cv::Range(0,2)) shallow copy

locateROI
adjustROI...

Simple operators
----------------

add, subtract, multiplication (element-wise, matrix-like), division, errors which can occur (different types, different sizes, wrong sizes...)

Advanced operations and functions
---------------------------------

filtering (low-pass filter), mean-value, max-value, min-value, median-filter, fourier-transform


