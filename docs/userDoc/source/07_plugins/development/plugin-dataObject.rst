.. include:: ../../include/global.inc

.. _plugin-dataObject:

DataObject
================================

The class **DataObject** (part of the library **dataObject**) provides a *n*-dimensional matrix that is used both in the core of |itom| as well as
in any plugins. The *n*-dimensional matrix can have different element types. These types and their often used enumeration value are defined
in the file *typeDefs.h* and are as follows.

================ ================ =========================================
Typedef          Enumeration      Description
================ ================ =========================================
ito::int8        ito::tInt8       8bit, signed, fixed point
ito::uint8       ito::tUint8      8bit, unsigned, fixed point
ito::int16       ito::tInt16      16bit, signed, fixed point
ito::uint16      ito::tUint16     16bit, unsigned, fixed point
ito::int32       ito::tInt32      32bit, signed, fixed point
ito::uint32      ito::tUint32     32bit, unsigned, fixed point
ito::float32     ito::tFloat32    32bit, single-precision floating point
ito::float64     ito::tFloat64    64bit, double-precision floating point
ito::complex64   ito::tComplex64  real and imaginary part is float32 each
ito::complex128  ito::tComplex128 real and imaginary part is float64 each
================ ================ =========================================

The last two dimensions of each dataObject are denoted *plane* and physically correspond to images. In order to also handle huge matrices in memory, 
The entire matrix is divided into planes, where each plane can be allocated at arbitrary positions and memory and is of type **cv::Mat_<type>**,
derived from **cv::Mat** of the **OpenCV** library. Therefore every plane can be used with every operator given by the **OpenCV**-framework (version 2.3.1
or higher). If available, the first *(n-2)* are allocated as one vector of pointers, each pointing to its corresponding plane. This kind of
dataObject and its way of allocating memory is called *unorganized*.

In order to make the *dataObject* compatible to matrices that are allocated in one huge memory block (like Numpy arrays), it is also possible to
make any *dataObject* continuous. Then, a huge data block is allocated, such that all planes lie consecutively in memory. Nevertheless, the pointer-tree
is still available, pointing to the starting points of all planes. This reallocation is implicitely done, when creating a Numpy-array from a dataObject.
	
DataObject can be declared in different possible ways with different dimensions and different data types.
Various possible implementations of declaring DataObject are listed below.
		
#.	DataObject();
#.	DataObject(const size_t size, const int type);
#.	DataObject(const size_t sizeY, const size_t sizeX, const int type);
#.	DataObject(const size_t sizeZ, const size_t sizeY, const size_t sizeX, const int type ,const unsigned char continuous = (unsigned char)'\000');
#.	DataObject(const unsigned char diimesions, const size_t *sizes, const int type, const unsigned char continuous = (unsigned char)'\000');
#.	DataObject(const unsigned char dimensions, const size_t *sizes, const int type, const uchar "continousouDataPtr, const size_t *steps = (const size_t *)0);
#.	DataObject(const size_t sizeZ, const size_t sizeY, const size_t sizeX, const int type, const uchar*continuousDataPtr, const size_t *steps = (const size_t *)0);
#.	DataObject(const ito::DataObject &copyConstr);
#.	DataObject(const unsigned char dimensions, const size_t *sizes, const int type, const cv::Mat *planes, const unsigned int nrOfPlanes);
#.	DataObject(const ito::DataObject dObj);

Following are some sample codes to get one quickly understand *basic programing structure of Data Objects*.

Example Codes
---------------

|itom|

The following code creates empty data object with no dimensions (0) and no type (0).

.. code-block:: c++
    :linenos:
	
    ito::DataObject d0;
    std::cout << "empty data object: \n";
    std::cout << " dimensions: " << d0.getDims() << "\n";
    std::cout << " type: " << d0.getType() << "\n" << std::endl;
	
.. note::

	Here **getDims()** method returns the dimensions of data object d0, whereas **getType()** method returns the type of d0.

Now, lets take an example of creating a two dimensional data object.

The following code creates such 2 dimensional data object of dimensions Y=2, X=5 and of type float32.
	
.. code-block:: c++
	:linenos:
	
	ito::DataObject d1(2,5, ito::tFloat32);
	std::cout << "2x5 data object, float32: \n";
	std::cout << " dimensions: " << d1.getDims() << "\n";
	std::cout << " type: " << d1.getType() << "\n";
	std::cout << " size: " << d1.getSize(0) << " x " << d1.getSize(1) << "\n";
	std::cout << " total: " << d1.getTotal() << "\n";
	std::cout << d1 << std::endl;

.. note::

	Here, **getSize()** method returns the size of each dimension of a data object d1.
	**getDims()** method returns the number of dimensions of d1.
	**getType()** method returns the type of d1.
	**getTotal()** method returns total number of elements in d1.
	
Addressing the elements of a data object
==========================================

Now, we know how to create a data object, so lets have a look how can one address the elements of a data object. 
Sometimes it is necessary to read or set single values in one matrix, sometimes one want to access all elements
in the matrix or a certain subregion. Therefore, the addressing can be done in one of the following ways:

method 1: direct access of one single value
--------------------------------------------------	

.. code-block:: c++
	:linenos:
	
	ito::DataObject d1(2,5, ito::tFloat32);
	d1.at<ito::float32>(0,1) = 5.2;
	std::cout << "d1(0,1) = " << d1.at<ito::float32>(0,1) << "\n" << std::endl;
	
Here, the addressing is done by the method **at** of the data object, which is pretty similar to the same method
of the **OpenCV** class **cv::Mat**. The **at** method can either be used to get the value at a certain position or
to set this value. There are special implementations of **at** for addressing values in a two- or three-dimensional
data object, where the first argument always is the **z-index** (3D), followed by the **y-index** and the **x-index**.
All indices are zero-based, hence the first element in this dimension is **0**.

.. note::
    
    The **at** method is templated where the template parameter must correspond to the type of the corresponding
    data object.

    
method 2: get line pointer for each line in matrix and work with line pointer to address elements of a data object
-----------------------------------------------------------------------------------------------------------------------------

.. code-block:: c++
	:linenos:
	
	ito::DataObject d1(3,5, ito::tInt16);
	size_t planeID = d1.seekMat(0);   //get internal plane number for the first plane
	ito::int16 *rowPtr = (ito::int16*)d1.rowPtr(planeID,0);
	size_t height = d1.getSize(0);
	size_t width = d1.getSize(1);	
	for(size_t m = 0 ; m< height ; m++)
	{
		rowPtr = (ito::int16*)d1.rowPtr(planeID,m);
		std::cout << "Row " << m << ":";
		for(size_t n=0 ; n < width; n++)
		{
			rowPtr[n] = m; 				//accessing each element of data object with line pointer
		}
	}
	std::cout << d1 << std::endl;
  
Here, **seekMat()** method gets the internal plane number of the 1st plane. 
In line #2, dynamic array rowPtr is defined as row pointer to the 0th plane of the data object d1. 
Now accessing each element of row pointer will access each element of the data object in that row. 

To use this row pointer method for data objects more than 2 dimensions, following code can be used. 

.. code-block:: c++
    :linenos:
    
    ito::int16 *rowPtr1= NULL;
    int dim1 = d1.getSize(0);
    int dim2 = d1.getSize(1);
    int dim3 = d1.getSize(2);
    int dim4 = d1.getSize(3);
    int dim5 = d1.getSize(4);
    size_t dataIdx = 0;
    for(int i=0; i<dim1; i++)
    {
        for(int j=0; j<dim2;j++)
        {
            for(int k=0; k<dim3;k++)
            {
                dataIdx = i*(dim2*dim3) + j*dim3 + k;
                for(int l=0; l<dim4;l++)
                {
                    rowPtr1= (TypeParam*)d1.rowPtr(dataIdx,l);
                    for(int m=0; m<dim5;m++)
                    {
                        //Assigning unique value to each element of d1.
                        rowPtr1[m] = cv::saturate_cast<TypeParam>(calcUniqueValue5D(i,j,k,l,m));
                    }
                }
            }
        }
    }
    
.. note::

    Here **dataIdx** represents the number of the plane in the matrix.
    The formula in line #15 assigns a non repeating increasing value to dataIdx such that each plane of the data object can be pointed out without any overlapping.

One can assign a single value to all elements of the data object using assignment operator "=" in the following way. 
Here we will also have a look on how to declare a 5 dimensional data object and assign a single floating point value to each element of the data object.

.. code-block:: c++
    :linenos:
    
    size_t *temp_size = new size_t[5];
    temp_size[0] = 10;
    temp_size[1] = 12;
    temp_size[2] = 16;
    temp_size[3] = 18;
    temp_size[4] = 10;
    ito::DataObject d1 = ito::DataObject(5,temp_size,ito::tFloat32);
    d1 = 3.7;
    std::cout << d1 << std::endl;
    delete[] temp_size;

Here line #7 uses implementation #4 for declaring 5 dimensional data object d1.

Working with Data Objects
=============================== 

Now, lets have a look on various methods to work with data objects.

Creating Eye Matrix
------------------------------------

Any square Data Object might need to be converted in eye matrix during many operations in matrix calculations. 
This can be quickly done using function **eye()** declared under *DataObject* class.
Syntax for the function **eye()** is shown below.

*dataObjectName.eye(noOfDimensions, dataType);*

*Return Type: void*

To understand the use of this function, following is an exemplary code given. Let's have a look at it.

.. code-block:: c++
	:linenos:
	
	ito::DataObject *d2 = new ito::DataObject();
	d2->eye(3, ito::tInt8);
	std::cout << "3x3-eye matrix (int8)" << *d2 << std::endl;
	delete d2;
	d2 = NULL;	
	
Here, the function eye() has been used with pointer variable d2 to convert the data object d2 into eye matrix. 

.. note::

	eye() function accepts only square matrices as inputs, otherwise it throws exception.
	
Creating Ones Matrix
-----------------------

Like Eye Matrix, Ones Matrix is equally important in matrix calculations.
So we have developed a function called **ones()** under *DataObject* class to quickly convert any *data object* into *ones matrix*.
Syntax for the function **ones()** is shown below.

*dataObjectName.ones(dim 1,dim 2,...,dim n, dataType);*

*Return Type: void*

.. code-block:: c++
	:linenos:
	
	ito::DataObject *d3 = new ito::DataObject();
	d3->ones(2,3,4,ito::tFloat64);
	std::cout << "2x3x4-ones matrix (double)" << *d3 << std::endl;
	delete d3;
	d3 = NULL;
	
Here, the function ones() has been used in Line #2 with pointer variable d3 to convert the data object d3 into ones matrix of dimension 2x3x4. 
	
Creating Zeros Matrix
-----------------------

We have developed a function called **zeros()** under *DataObject* class to quickly convert any *data object* into *zero matrix*.
Syntax for the function **zeros()** is shown below.

*dataObjectName.zeros(dataType);*      
                                                                                                                                                        
*dataObjectName.zeros(const size_t size, dataType);*

*dataObjectName.zeros(const size_t sizeY, const size_t sizeX, dataType);*

*dataObjectName.zeros(const size_t sizeZ, const size_t sizeY, const size_t sizeX, dataType);*

*dataObjectName.zeros(const unsigned char dimensions, const size_t *sizes, dataType);*

*Return Type: RetVal*

As zeros() function is overloaded, there are more than one syntax shown above.

.. note::
	
	The **RetVal** class is used for handling error management and return relative codes. More description on this class can be seen in :ref:`plugin-RetVal-Ref`.

Following code explains the usage of **zeros()** function.

.. code-block:: c++
	:linenos:
	
	ito::DataObject *dObjZeros = new ito::DataObject();
	dObjZeros->zeros(2,3,4,ito::tFloat64);
	std::cout << "3x4x5-zeros matrix (double)" << *dObjZeros << std::endl;
	delete dObjZeros;
	dObjZeros = NULL;
	
Here, line number 2 shows the way to use zeros() function to convert data object dObjZeros into a yzero matrix. 

Direct Access to the underlying cv::Mat
---------------------------------------

Following example shows one of the methods to access underlying planes in multidimensional matrices. 

.. code-block:: c++
	:linenos:
	
	// 4 x 5 x 3 DataObject, int16
	ito::DataObject d4(4,5,3,ito::tInt16);
	std::cout << "DataObject (4x5x3), int16 \n" << std::endl;
	d4 = 3;		//assign value 3 to all elements
	//access to the third plane (index 2)
	planeID = d4.seekMat(2);
	cv::Mat *plane3 = (cv::Mat*)d4.get_mdata()[planeID];
	std::cout << "OpenCV plane" << std::endl;
	std::cout << *plane3 << std::endl;
	//accessing second line in plane3
	ito::int16* rowPtr2 = (ito::int16*)plane3->ptr(1);
	//regions of interest
	//d5 = d4[1:3,0:2,:]
	ito::Range ranges[3] = { ito::Range(1,3), ito::Range(0,2), ito::Range::all() };
	ito::DataObject  d5 = d4.at(ranges);
	d5 = 7;
	
Let's try to analyse the code above. As we can see in line #6, we used **seekMat()** method to retrieve the plane id of 3th plane in 3 dimensional matrix d4. 

line #7 declares a pointer variable plane3 of type cv::Mat to hold the contents of plane 3 of data object d4. line #11 declares a row pointer to point a perticular row in plane 3 of data object d4.

line #14 defines the exemplary ranges to create a new data object d5 from a part of data object d4.

The other way to perform the same operation of line #14 is shown below.

.. note::
	
	**get_mdata()** is a function declared under *DataObject* class. It returns pointer to vector of *cv::_Mat-matrices*.
	
.. code-block:: c++
	:linenos:
	
	ito::Range *ranges = new ito::Range[3];
	ranges[0] = ito::Range(1,3);
	ranges[1] = ito::Range(0,2);
	ranges[2] = ito::Range::all();
	
This code shows the way to modify ranges individually, which can be very useful if one needs to modify this range later in this code to work on other data objects perhaps.

Adjusting ROI of a Data Object
------------------------------------

This section will teach you about how to adjust Region of Interest (ROI) in any data object. 

The following example code shows the way to adjust ROI with adjustROI() method and to locate ROI with locateROI() method.

.. code-block:: c++
	:linenos:
	
	//adjusting ROI of 6x7 data object.
	ito::DataObject d6(6,7,ito::tInt16);
	int roiLocate[]= {0,0,0,0}; //Empty Array to locate ROI of 2 dimensional data object d6.
	d6.adjustROI(-2,0,-1,-4);
	d6.locateROI(roiLocate);
	std::cout << d6 << std::endl;
	for(int i =0;i<4; i++)
	{
		std::cout << roiLocate[i] << std::endl;
	}
	
Here, line #4 shows the use of adjustROI() function where negative parameters indicate that the ROI is shrinking in perticular dimension. More detailed description of adjustROI() and locateROI() methods can be seen at `Document </Description document for adjustROI and locateROI methods>`_ .

One can also pass an array as a parameter to this adjustROI() function describing the offset details as shown in the following code.

.. code-block:: c++
	:linenos:
	
	int matLimits2d[] = {-2,0,-1,-4};
	d6.adjustROI(2,matLimits2d);
	std::cout << d6 << std::endl;

Here, adjustROI() function is called with 2 parameters and as can be seen in line #1, the array matLimits2d[] contains the same offset values as passed in adjustROI() method in the previous example.

One can use this example to adjust Region of Interest of data objects more than 2 dimensions as well as shown in later examples below. 
	
The following code shows such an example to modify Region of Interest of a 3 dimensional data object.
	
.. code-block:: c++
	:linenos:
	
	//adjusting ROI of 6x7x8 data object.
	ito::DataObject d7(6,7,8,ito::tFloat32);
	int matLimits3d[] = {-1,-2,0,-2,-3,-1};
	int lims3d[]= {0,0,0,0,0,0}; //Empty Array to locate ROI of 3 dimensional data object d7.
	d7.adjustROI(3,matLimits3d);
	d7.locateROI(lims3d);
	std::cout << d7 << std::endl;
	for(int i = 0; i<5; i++)
	{
		std::cout << lims3d[i] << std::endl;
	}
	
Here, a 3 dimensional data object d7 of dimensions 6x7x8 is declared in line #1 using the implementation #4 for data objects.

As can be seen in line #3, array matLimits3d[] of return type int contains required 6 offset values to adjust ROI of 3 dimensional data object d7. As shown in line #4, an empty array lims3d[] of int as return type is defined to locate the ROI of data object d7 using locateROI() function. 
Line #7 will print the resultant data object after being adjusted by adjustROI() method in line #5 and the for loop in line #8-11 will print these located offset values of the resultant ROI of d7.
		
Setting and Getting Axis Units
-------------------------------------------------------------

In this section, you will study about assigning and retrieving the axis units at each dimension of a data object.

.. code-block:: c++
	:linenos:
	
	ito::DataObject d8(6,7,ito::tFloat32);
	bool vop1, vop2 = 0;
	d8.setAxisUnit(0,"µm");
	d8.setAxisUnit(1,"cm");
	std::string AxisUnit1 =d8.getAxisUnit(0,vop1); //Getting axis unit of 1st dimension of data object d8
	std::string AxisUnit2 =d8.getAxisUnit(1,vop2); //Getting axis unit of 2nd dimension of data object d8
	std::cout << "Axis Unit of 1st Dimension:" << AxisUnit1 << std::endl;
	std::cout << "Axis Unit of 2nd Dimension:" << AxisUnit2 << std::endl;
	
Here, a 2 dimensional data object *d8* of dimensions *6x7* is declared in line #1. Line #2 declares boolean variables vop1 and vop2 to pass as parameters in getAxisUnit() method later. 
Line #3 and #4 sets the units for dimensions 1 and 2 of data object d8 respectively. 

These assigned axis units can be retrieved by getAxisUnit() method as shown in line #5 and #6. Line #7 and #8 prints these retrieved axis units of data object d8.

Setting and Getting Axis Scale
--------------------------------------------------

In this section, we will learn about setting and getting the scale values of perticular Axis (dimension) of a data object. 
This can be done using setAxisScale() and getAxisScale() functions as shown below. These both functions are declared under *DataObject* class.
Syntax for the setAxisScale() function is shown below:

*DataObjectName.setAxisScale(const unsigned int axisNum, const double scale);*

Return Type: int

.. code-block:: c++
	:linenos:
	
	ito::DataObject d9(6,5,3,ito::tInt16);
	d9.setAxisScale(0,5);
	d9.setAxisScale(1,-0.5);
	d9.setAxisScale(2,3.24);
	double AxisScale1 =d9.getAxisScale(0);
	double AxisScale2 =d9.getAxisScale(1);
	double AxisScale3 =d9.getAxisScale(2);
	std::cout << "Axis 1 Scale:" << AxisScale1 << "Axis 2 Scale:" << AxisScale2 << "Axis 3 Scale:" << AxisScale3 << std::endl; 

Documentation is to be written here.
	
	//adjugate the data object
	ito::DataObject d10(6,5,3,ito::tComplex128);
	d10.at<ito::complex128 >(0,1,2) = cv::saturate_cast<ito::complex128 >(ito::complex128 (23.2,0));
	d10.at<ito::complex128 >(1,0,1) = cv::saturate_cast<ito::complex128 >(ito::complex128 (0,3));
	d10.at<ito::complex128 >(2,2,1) = cv::saturate_cast<ito::complex128 >(ito::complex128 (1234,-23.34));
	ito::DataObject adjugatedDataObj = d10.adj();
	std::cout << "The Adjugated data object:" << std::endl;
	
	//Transpose data object
	ito::DataObject d11(2,2,ito::tInt16);
	int temp=0;
	for(int i=0;i<2;i++)
		for(int j=0;j<2;j++)
		{
		temp++;
		d11.at<ito::int16>(i,j)= cv::saturate_cast<ito::int16>(temp);
		}
	ito::DataObject transDObj = mat1_2d.trans();
	std::cout<< "The Transposed data object:" << std::endl;
	
.. note::
		
		Use of some basic arithmatic operators with data objects.(Same syntax can be used for other operators like '+','-','=+','=-', div, cross multiplication (!=), <<(shift left), >> ( shift right))
	
.. code-block:: c++
	:linenos:
	
	ito::DataObject d12(2,2,ito::tInt16);
	ito::DataObject d13(2,2,ito::tInt16);
	ito::DataObject d14(2,2,ito::tInt16);
	d12= cv::saturate_cast<ito::int16>(2);
	d13= cv::saturate_cast<ito::int16>(2);
	d14= d12 + d13;
	std::cout << "Addition of two matrix is:" << d14 << std::endl;

.. note::
		
			syntax for dot multiplication: d14 = d12.mul(d13);
			syntax for division operator: d14 = d12.div(d13);
			syntax for - minus operator: 
			syntax for * cross multiplication operator: mulCross_mat3_2d = mulCross_mat1_2d * mulCross_mat2_2d;  
			syntax for += operator: mat2_3d += mat1_3d;
			syntax for *= operator: mat1_2d *= mat2_2d;
			syntax for -= operator: mat1_2d -= mat2_2d;
			
			
			
.. note:: 

			Use of comparision operators.
		
	.. code-block:: c++
	:linenos:
	
	//Use of == (equal to) operator
	ito::DataObject d15(3,4,5,ito::tFloat32);
	ito::DataObject d16(3,4,5,ito::tFloat32);
	ito::DataObject d17(3,4,5,ito::tFloat32);
	d15= cv::saturate_cast<ito::int16>(2);
	d16= cv::saturate_cast<ito::int16>(2);
	d17 = d15 == d16		//d17 is supposed to get 0 value as d15 and d16 are equal matrices.
	std::cout << "If the result matrix is a zero matrix then d15 and d16 are equal matrices" << "result matrix" << d17;
	
.. note::
	
			syntax for != (not equa to) operator: mat3_1d = mat1_1d != mat2_1d;
			syntax for "<=" (Less Than or Equal to) operator:
			
normal text again. now without line numbering

.. code-block:: c++
    
    if(i > 2)
	{
		i=3;
	}

.. note::
    
    For a full reference of the class **DataObject** see :ref:`plugin-DataObject-Ref`.
    
.. toctree::
   :hidden:

   plugin-RetVal-Ref.rst

ito::DataObject :ref:`ito::DataObject`