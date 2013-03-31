Glossary
========

.. glossary::
    
**(non-)continuous data object**

    per default, matrixes in the data object are stored as follows: the last two dimensions are stored 
    continuously in a two dimensional matrix by the help of openCV-Mat-structures. All these 2dim-matrixes are then
    stored in a one-dimensional vector, however the relation between the position in this vector and the first *(n-2)* dimensions, where
    *n* is the number of total dimensions, is given by a simple equation, which can for instance be found in the *openCV* documentation. Since
    the data blocks for each two-dimensional matrix must not be stored continuously in memory, this method is called *non-continuous data object*.

    In order to realize a compatible version with respect to *numpy*, *matlab*... the data can also be stored *continuously*. The basic structure for
    the data object is the same than in the *non-continuous* (default) version, but the data of each 2dim-matrix lies continuously in memory and each data-pointer
    of each matrix just points to the first element of the corresponding matrix in this big data block in memory. Data objects of class *ndDataObject* are always 
    organized as continuous data object, while ordinary data objects (class *dataObject*) can be continuous or non-continuous.

    The non-continuous representation has advantages especially with respect to huge data sets, since it is more difficult to get a big continuous block in memory without
    reorganizing it than multiple smaller blocks of memory, which can be distributed randomly in memory.

    Matrixes with only one or two dimension are automatically stored continuously.

    
    
**Deep Copies**

    Are copies


**Shallow Copies**

    are copied references
    

**Python Version**


