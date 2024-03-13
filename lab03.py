import glob
import cv2 as cv
import numpy as np



def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, one_c):
    #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    
    pts = [((float(one_c[0])), float(one_c[1])),
            ((float(one_c[2])), float(one_c[3])),
            ((float(one_c[4])), float(one_c[5])),
            ((float(one_c[6])), float(one_c[7]))]
    
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
	    [0, 0],
	    [maxWidth - 1, 0],
	    [maxWidth - 1, maxHeight - 1],
	    [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def main(argv):
    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coords = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coords.append(sp_line)

    test_images = [img for img in glob.glob("test_images_zao/*.jpg")]
    test_results = [img for img in glob.glob("test_images_zao/*.txt")]

    template_images = [img for img in glob.glob("templates/*.jpg")]
    #template_images.sort()
    test_images.sort()
    test_results.sort()
    size = (100, 100)
    #print(test_images)
    cv.namedWindow('image_clone', 0)
    #cv.namedWindow('template', 0)
    #cv.namedWindow('one_palce_img', 0)
    total_parkings = 0
    total_correct_parkings = 0
    n_park = 0
    font = cv.FONT_HERSHEY_PLAIN
    for img_name, result in zip(test_images, test_results):
        img = cv.imread(img_name)
        res = open(result, 'r')
        res_lines = res.readlines()

        res_lines = [int(line.strip()) for line in res_lines]
        img_clone = img.copy()
        #cv.imshow('image', img)
        #print(res_lines)
        for coord, template in zip(pkm_coords, template_images):
            n_park += 1
            total_parkings += 1
            #print('coord', coord)
            #print('template', template)
            one_palce_img = four_point_transform(img, coord)
            one_palce_img = cv.resize(one_palce_img, size)

            temp = cv.imread(template)         
            temp_gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
            one_palce_img_gray = cv.cvtColor(one_palce_img, cv.COLOR_BGR2GRAY)


            #one_palce_img_gray = cv.equalizeHist(one_palce_img_gray)
            #temp_gray = cv.equalizeHist(temp_gray)
            #cv.imshow('one_palce_img', one_palce_img_gray)
            #cv.imshow('template', temp_gray)
            res = cv.matchTemplate(one_palce_img_gray, temp_gray, cv.TM_CCORR_NORMED  )
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            
            
            
            
            
            
            
            
            left_top = (int(coord[0]), int(coord[1]))
            right_bottom = (int(coord[4]), int(coord[5]))
            #print(left_top, right_bottom)
            center_x = (left_top[0] + right_bottom[0]) // 2
            center_y = (left_top[1] + right_bottom[1]) // 2
            cv.putText(img_clone, str(n_park), (center_x + 3 , center_y + 3), font, 1, (255, 0, 0), 2, cv.LINE_AA)
            print(min_val, max_val)
            
            
            if max_val > 0.97392:
                cv.circle(img_clone, (center_x, center_y), 10, (0, 255, 0), -1)
                if int(res_lines[n_park - 1]) == 0:
                    total_correct_parkings += 1
            else:
                cv.circle(img_clone, (center_x, center_y), 10, (0, 0, 255), -1)
                if int(res_lines[n_park - 1]) == 1:
                    total_correct_parkings += 1 

        print('\n')       
        cv.imshow('image_clone', img_clone)  
        n_park = 0
        cv.waitKey(0)

    print("Total parkings:", total_parkings)
    print("Total correct parkings:", total_correct_parkings)
    accuracy = (total_correct_parkings / total_parkings) * 100 if total_parkings != 0 else 0
    print("Accuracy:", accuracy)
    
        


        
    
            

if __name__ == "__main__":
    import sys
    main(sys.argv)