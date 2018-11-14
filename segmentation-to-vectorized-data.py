import numpy as np
import cv2
import locale
import datetime
import os
def getTimeString():
    #if platform.system()=='Linux':
    locale.setlocale( locale.LC_ALL , 'en_US.utf8' )
    #elif platform.system()=='Windows':
    #    locale.setlocale(locale.LC_ALL, 'eng_us')
    return datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")


def save_predictions(save_path, name, height, width, depth, approx):
        """Draw detected bounding boxes."""

        xml = '<annotation>\n'
        xml += '<folder>' + 'image' + '</folder>\n'
        xml += '<filename>' + name + '.jpg' + '</filename>\n'
        xml += '<source>\n'
        xml += '<database>' + 'The DLR Datensatz' + '</database>\n'
        xml += '<annotation>' + 'S_Majid_Azimi' + '</annotation>\n'
        xml += '</source>\n'
        xml += '<size>\n'
        xml += '<height>' + str(height) + '</height>\n'
        xml += '<width>' + str(width) + '</width>\n'
        xml += '<depth>' + str(depth) + '</depth>\n'
        xml += '</size>\n'

        for i, coord in enumerate(approx):
            xml += '<object>\n'
            xml += '<type>' + 'polygon' + '</type>\n'
            xml += '<name>' + 'lane' + '</name>\n'
            xml += '<truncated>' + str(0) + '</truncated>\n'
            xml += '<difficult>' + 'e' + '</difficult>\n'
            xml += '<verified>' + str(1) + '</verified>\n'
            xml += '<deleted>' + str(0) + '</deleted>\n'
            xml += '<date>' + getTimeString() + '</date>\n'
            xml += '<username>' + 'S_Majid_Azimi' + '</username>\n'
            xml += '<id>' + str(i) + '</id>\n'
            for c in coord:
                xml += '<pt>\n'
                xml += '<x>' + str(c[0]) + '</x>\n'
                xml += '<y>' + str(c[1]) + '</y>\n'
                xml += '</pt>\n'
            xml += '</object>\n'
        xml += '</annotation>'
        to_xml = open(save_path + '/' +name + '.xml', 'w')

        for line in xml:
            to_xml.write(line)
        to_xml.close()



# downscale and read image
# img = cv2.pyrDown(cv2.imread('hammer.jpg', cv2.IMREAD_UNCHANGED))

input_images_path = '/home/majid/Public/test_vectorized_lane_marking/images'
input_pixel_wise_labels_path = '/home/majid/Public/test_vectorized_lane_marking/pixel_wise_labels/pred'
input_vectorized_labels_path = '/home/majid/Public/test_vectorized_lane_marking/vectorized_labels'


pixel_wise_labels_name = [name for name in os.listdir(input_pixel_wise_labels_path) if
                   name.endswith(('.png'))]
for ind, pixel_wise_label_name in enumerate(pixel_wise_labels_name):
    _name = os.path.splitext(pixel_wise_label_name)[0]  # basename(name)  removing extension
    file_path = input_pixel_wise_labels_path + '/' + pixel_wise_label_name
    print('reading file  {}'.format(file_path))
    im = cv2.imread(file_path) #,-1)

    # threshold image
    #ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
    #                    127, 255, cv2.THRESH_BINARY)
    ret, thresh = cv2.threshold(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 127, 255, 0)

    # RETR_EXTERNAL 	 retrieves only the extreme outer contours. It sets hierarchy[i]    [2]=hierarchy[i][3]=-1 for all the contours.
    # RETR_LIST 	retrieves all of the contours without establishing any hierarchical relationships.
    # RETR_CCOMP 	retrieves all of the contours and organizes them into a two-level hierarchy. At the top level, there are external boundaries of the components. At the second level, there are boundaries of the holes. If there is another contour inside a hole of a connected component, it is still put at the top level.
    # RETR_TREE 	retrieves all of the contours and reconstructs a full hierarchy of nested contours.
    # RETR_FLOODFILL 	

    # CHAIN_APPROX_NONE 	stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.
    # CHAIN_APPROX_SIMPLE 	compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.
    # CHAIN_APPROX_TC89_L1 	applies one of the flavors of the Teh-Chin chain approximation algorithm [168]
    # CHAIN_APPROX_TC89_KCOS applies one of the flavors of the Teh-Chin chain approximation algorithm [168] 

    # get contours from image
    #image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL,
    #                    cv2.CHAIN_APPROX_SIMPLE)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    """
    mask = np.zeros((np.shape(im)))
    cv2.polylines(mask,contours,True,(0,255,255))
    cv2.imshow('mask',mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    mask = np.zeros((np.shape(im)))
    im = mask
    #cv2.imshow('mask',mask)
    approx_2c_total = []
    ARC_THRESH = 0.001
    im_height, im_width, im_ch = np.shape(im)
    # for each contour
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # print(i, area)
    
        # if not (100 <=area <= 100000): #
        if not (17 <=area <= 100): # dash-line
            continue
        #if not (0 <=area <= 0): # dash-line
        #    
        # calculate epsilon base on contour's perimeter
        # contour's perimeter is returned by cv2.arcLength
        epsilon = ARC_THRESH * cv2.arcLength(cnt, True)
        # get approx polygons
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # draw approx polygons
        cv2.drawContours(im, [approx], -1, (0, 255, 0), 1)
        ### Step #2 - Reshape to 2D matrices
        approx_2c = approx.reshape(-1,2)
        #print(approx_2c)
        approx_2c_total.append(approx_2c)
        # break
        # hull is convex shape as a polygon
        # get convex hull
        hull = cv2.convexHull(cnt)
        # draw it in red color
        # cv2.drawContours(im, [hull], -1, (0, 0, 255))



    save_predictions(input_vectorized_labels_path , _name, im_height, im_width, im_ch, approx_2c_total)


    #cv2.imshow("contours", im)
    cv2.imwrite(input_vectorized_labels_path + '/vis/' + _name + '.png' , im) 
ESC = 27
#while True:
#    keycode = cv2.waitKey(27)
#    if keycode != -1:
#        #keycode &amp;= 0xFF
#        if keycode == ESC:
#            break
 
#cv2.destroyAllWindows()
