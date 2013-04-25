.. include:: ../../include/global.inc

.. _plugin-dataObject:

DataObject
================================

dataObject is a class type which contains a n-dimensional matrix

    The n-dimensional matrix can have different element types. Recently the following types are supported:
    int8, uint8, int16, uint16, int32, uint32, float32, float64 (=> double), complex64 (2x float32), complex128 (2x float64)

    In order to handle huge matrices, the data object can divide one matrix into subparts in memory. Each subpart (called matrix-plane)
    is two-dimensional and covers data of the last two dimensions. Each of these matrix-planes is of type cv::Mat_<type> and can be used with every
    operator given by the openCV-framework (version 2.3.1 or higher).
	
DataObject can be declared in different possible ways with different dimensions and different data types.
Various possible implementations of declaring DataObject are listed below.
		
#.	DataObject();
#.	DataObject(const size_t size, const int type);
#.	DataObject(const size_t sizeY, const size_t sizeX, const int type);
#.	DataObject(const size_t sizeZ, const size_t sizeY, const size_t sizeX, const int type ,const unsigned char continuous = (unsigned char)'\000');
#.	DataObject(const unsigned char diimesions, const size_t *sizes, const int type, const unsigned char continuous = (unsigned char)'\000');
#.	DataObject(const unsigned char dimensions, const size_t '*sizes, const int type, const uchar "continousouDataPtr, const size_t *steps = (const size_t *)0);
#.	DataObject(const size_t sizeZ, const size_t sizeY, const size_t sizeX, const int type, const uchar*continuousDataPtr, const size_t *steps = (const size_t *)0);
#.	DataObject(const ito::DataObject &copyConstr);
#.	DataObject(const unsigned char dimensions, const size_t *sizes, const int type, const cv::Mat *planes, const unsigned int nrOfPlanes);
#.	DataObject(const ito::DataObject dObj);

Following are some sample codes to get one quickly understand *basic programing structure of Data Objects*.

Example Codes
-----------

|itom|

The following code creates empty data object with no dimensions (0) and no type (0).

.. code-block:: c++
    :linenos:
	
    ito::DataObject d0;
    std::cout << "empty data object: \n";
    std::cout << " dimensions: " << d0.getDims() << "\n";
    std::cout << " type: " << d0.getType() << "\n" << std::endl;
	
.. note::

	Here getDims() method returns the dimensions of data object d0, whereas getType() method returns the type of d0.

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
	
.. note::
	
	Addressing elements.
	
Elements of data objects can be addressed in one of the following ways.
method 1: direct access of one single value
	
.. code-block:: c++
	:linenos:
	
	ito::DataObject d1(2,5, ito::tFloat32);
    d1.at<ito::float32>(0,1) = 5.2;
    std::cout << "d1(0,1) = " << d1.at<ito::float32>(0,1) << "\n" << std::endl;
    
method 2: get line pointer for line in matrix and work with line pointer.

.. code-block:: c++
	:linenos:
	
    ito::DataObject d1(3,5, ito::tInt16);
	size_t planeID = d1.seekMat(0);	    //get internal plane number for the first plane
    ito::int16 *rowPtr = (ito::int16*)d1.rowPtr(planeID,0);
	size_t height = d1.getSize(0);
    size_t width = d1.getSize(1);

    for(size_t m = 0 ; m < height ; m++)
    {
        rowPtr = (ito::int16*)d1.rowPtr(planeID,m);
        std::cout << "Row " << m << ":";
        for(size_t n=0 ; n< width; n++)
        {
			rowPtr[n] = m;				//accessing each element of data object with line pointer
        } 
    }
	 std::cout << d1 << std::endl;   
   
.. note::

	Defining 5 dimensional data object d1 and assign one value to all elements of d1.
	
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

.. note::

	Defining 5 dimensional data object d1 and assign one value to all elements of d1.
	
.. code-block:: c++
	:linenos:
	
    //create 3x3 eye matrix: (Note: matrix should be square matrix)
    ito::DataObject *d2 = new ito::DataObject();
    d2->eye(3, ito::tInt8);
    std::cout << "3x3-eye matrix (int8)" << *d2 << std::endl;
    delete d2;
    d2 = NULL;

    //create 2x3x4 ones matrix:
    ito::DataObject *d3 = new ito::DataObject();
    d3->ones(2,3,4,ito::tFloat64);
    std::cout << "2x3x4-ones matrix (double)" << *d3 << std::endl;
    delete d3;
    d3 = NULL;

	//create 3x4x5 zeros matrix:
    ito::DataObject *dObjZeros = new ito::DataObject();
    dObjZeros->zeros(2,3,4,ito::tFloat64);
    std::cout << "3x4x5-zeros matrix (double)" << *dObjZeros << std::endl;
    delete dObjZeros;
    dObjZeros = NULL;
	
    // 4 x 5 x 3 DataObject, int16
    ito::DataObject d4(4,5,3,ito::tInt16);
    std::cout << "DataObject (4x5x3), int16 \n" << std::endl;

    //assign value 3 to all elements
    d4 = 3;

    //direct access to the underlying cv::Mat.
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
    //equivalent to:
    /*
    ito::Range *ranges = new ito::Range[3];
    ranges[0] = ito::Range(0,3);
    ranges[1] = ito::Range(0,2);
    ranges[2] = ito::Range::all();
    --> delete ranges later
    */
    ito::DataObject  d5 = d4.at(ranges);
    d5 = 7;
	
	//adjusting ROI of 6x7 data object.
	//Method:1
	ito::DataObject d6(6,7,ito::tInt16);
	int lims2d[]= {0,0,0,0}; //Empty Array to locate ROI of 2 dimensional data object d6.
	d6.adjustROI(-2,0,-1,-4);
	d6.locateROI(lims2d);
	std::cout << d6 << std::endl;
	for(int i =0;i<4; i++)
	{
		std::cout << lims2d[i] << std::endl;		
	}
	
	//Method:2
	int matLimits2d[] = {2,0,1,4};
	//adjusting back to original size of ROI
	d6.adjustROI(2,matLimits2d);
	std::cout << d6 << std::endl;
	
	//adjusting ROI of 6x7x8 data object.
	ito::DataObject d7(6,7,8,ito::tFloat32);
	int matLimists3d[] = {-1,-2,0,-2,-3,-1};
	int lims3d[]= {0,0,0,0,0,0}; //Empty Array to locate ROI of 3 dimensional data object d7.
	d7.adjustROI(3,matLimits3d);
	d7.locateROI(lims3d);
	std::cout << d7 << std::endl;
	for(int i = 0; i<5; i++)
	{
		std::cout << lims3d[i] << std::endl;
	}
	
	//setting and getting axis units.
	ito::DataObject d8(6,7,ito::tFloat32);
	bool vop1, vop2 = 0;
	d8.setAxisUnit(0,"µm");
	d8.setAxisUnit(1,"cm");
	std::string AxisUnit1 =d8.getAxisUnit(0,vop1); //Getting axis unit of 1st dimension of data object d8
	std::string AxisUnit2 =d8.getAxisUnit(1,vop2); //Getting axis unit of 1st dimension of data object d8
	std::cout << "Axis Unit of 1st Dimension:" << AxisUnit1 << std::endl;
	std::cout << "Axis Unit of 2nd Dimension:" << AxisUnit2 << std::endl;
	
	//setting and getting axis scale
	ito::DataObject d9(6,5,3,ito::tInt16);
	d9.setAxisScale(0,5);
	d9.setAxisScale(1,-0.5);
	d9.setAxisScale(2,3.24);
	double AxisScale1 =d9.getAxisScale(0);
	double AxisScale2 =d9.getAxisScale(1);
	double AxisScale3 =d9.getAxisScale(2);
	std::cout << "Axis 1 Scale:" << AxisScale1 << "Axis 2 Scale:" << AxisScale2 << "Axis 3 Scale:" << AxisScale3 << std::endl; 
	
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


