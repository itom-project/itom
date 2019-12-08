from itom import *


'''
This demo should show how to save and load dataObjects
to/from image formats as well as native itom formats
'''

def demo_loadSaveDataObjects():

    '''create a colored dataObject (typ rgba32)'''
    rgba32=dataObject([100,100],'rgba32')
    '''set all pixels to a gray value. Therefore red=green=blue with no transparency, what means that alpha has to be set to the maximal value''' 
    rgba32[0:100,0:100]=rgba(150,150,150,255)
    '''insert a red, green and blue bar in the picture wich are not complete intransparent'''
    rgba32[10:30,:]=rgba(255,0,0,150)
    rgba32[50:70,:]=rgba(0,255,0,150)
    rgba32[80:100,:]=rgba(0,0,255,150)
    '''show the image'''
    plot(rgba32)

    '''save the dataObject as a tiff file with a rgba color palette''' 
    filter("saveTiff",rgba32, 'pic_rgba.tiff', 'rgba') 

    '''reload the picture as it is->that means as a rgba32 dataObject''' 
    reload_tiff_rgba=dataObject()
    filter("loadAnyImage",reload_tiff_rgba, 'pic_rgba.tiff','asIs')

    '''save the dataObject as a tiff file with a rgb color palette, which causes 
    that the transparency of the bars will be ignored. 
    If gray or gray16 is choosen as color palette the colored Data Object will be converted to a gray Image '''
    filter("saveTiff",rgba32, 'pic_rgb.tiff', 'rgb')

    '''reload the picture as it is-> that means as a rgba32 dataObject with all alphas set to 255(no transparency)''' 
    reload_tiff_rgb=dataObject()
    filter("loadAnyImage",reload_tiff_rgb, 'pic_rgb.tiff','asIs')


    '''save the dataObject as a png file with a gray color palette(also gray16 and all colored palettes are supportted)'''
    filter("savePNG",rgba32, 'pic_gray.png', 'gray')

    '''reload the picture as it is -> that means as a "gray" dataObject (type uint8) ''' 
    reload_png_gray=dataObject()
    filter("loadAnyImage",reload_png_gray, 'pic_gray.png','asIs')


    '''save the dataObject as a pgm with a 16bit grayscale (only for gray images->only gray and gray16 supported)'''
    filter("savePGM",rgba32, 'pic_gray.pgm', 'gray16')
    '''load the pgm file as it is->that means as a "gray" dataObject (type uint16 due to the 16bit gray color palette) '''
    reload_pgm_gray16=dataObject()
    filter("loadAnyImage",reload_pgm_gray16, 'pic_gray.pgm','asIs')


    '''save the dataObject as IDC (itom data collection, saved using Python module 'pickle') therefore it must be wrapped into a dictionary'''
    dic={'data':rgba32}
    saveIDC('pic_idc.idc',dic)
    '''load the idc file'''
    loaded_dic = loadIDC('pic_idc.idc')
    reload_img = loaded_dic["data"]

    '''copy the dataObject'''
    rgba32_1=rgba32
    '''save both (also more possible) in one IDC file'''
    dic_1={'data_1':rgba32,'data_2':rgba32_1}
    loaded_dic_1=saveIDC('multi_pic_idc.idc',dic_1)


    ################################################################################################################
    ''' In this section a uint8 dataObject is created and saved in false colors'''

    '''create a gray image of type uint8'''
    uint8=dataObject([100,100],'uint8')
    '''insert blocks with values of 0.0, 1.0, 50 and 100'''
    uint8[0:25,:]=0
    uint8[25:50,:]=1
    uint8[50:75,:]=50
    uint8[75:100,:]=100

    '''save as tiff file colored in the ''hotIron'' color palette. Other palettes are for example "grayMarked" or "falseColor" '''
    filter('saveTiff',uint8,'pic_uint8.tiff','hotIron')

    ################################################################################################################
    '''This section shows how to save floating point dataObjects as a image''' 


    '''create a gray image of type float32'''
    float32=dataObject([100,100],'float32')
    '''insert blocks with values of 0.0, 1.0, 50 and 100'''
    float32[0:25,:]=0.0
    float32[25:50,:]=1.0
    float32[50:75,:]=50.0
    float32[75:100,:]=100.0



    '''save the float32 Object as a png file with a false color palette (here hotIron is used, others are for example "grayMarked" or "falseColor" )
    If you save a dataObject of type floate the color palette is spaced between [0,1] ->all values above 1.0 will be clipped to the maximum Value  '''
    filter("savePNG",float32, 'pic_falseColor.png', 'hotIron')

    '''reload the saved PNG as a uint8 dataObject->all steps with values above 1.0 have the same gray value'''
    reload_png_falseColor=dataObject()
    filter("loadAnyImage",reload_png_falseColor, 'pic_falseColor.png','GRAY')

    '''to get rid of the problem above you need to normalize your dataObject between 0.0 and 1.0 using the function ''normalize''
    '''
    normfloat32=float32.normalize(0.0,1.0,'float32')
    filter("savePNG",normfloat32, 'pic_normalized_falseColor.png', 'hotIron')

    '''reload the image as a uint8 dataObject->all steps are included''' 
    reload_normalized_falseColor=dataObject()
    filter("loadAnyImage", reload_normalized_falseColor, 'pic_normalized_falseColor.png','GRAY')

if __name__ == "__main__":
    demo_loadSaveDataObjects()