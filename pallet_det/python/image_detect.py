#!/usr/bin/env python3

import sys
import os
sys.path.append('../..')
import cv2
import darknet_new.darknet as darknet

COLORS_YELLOW = (255, 191, 0)
COLORS_PINK = (255, 20, 147)
COLORS = [COLORS_PINK, COLORS_YELLOW]

save_result_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)) + "/detected_image"
if not os.path.exists(save_result_dir):
    os.mkdir(save_result_dir) 

thre = 0.9
show_coordinates = False

data_file = []
cfg_file = []
weight_file = []
network = []
class_names = []
class_colors = []
num = sys.argv[1]
num = int(num)
argc = len(sys.argv)


# def draw_boxes(img, detection, colors): 
#     for (label, score, box) in detection:
#         text = label+ ":"+str(score)
#         left, top, right, bottom = darknet.bbox2points(box)
#         #for text background
#         fontFace = cv2.FONT_HERSHEY_COMPLEX
#         fontScale = 0.5
#         thickness = 1
#         labelSize = cv2.getTextSize(text, fontFace, fontScale, thickness)
#         _x1 = left # bottom_left x of text
#         _y1 = top # bottom_left y of text
#         _x2 = left+labelSize[0][0] # top_right x of text
#         _y2 = top-labelSize[0][1] # top_right y of text
#         cv2.rectangle(img, (left, top), (right, bottom), colors[label], 2)
#         cv2.rectangle(img, (_x1,_y1), (_x2,_y2), colors[label], cv2.FILLED) # text background
#         cv2.putText(img, text, (left, top - 4), cv2.FONT_HERSHEY_COMPLEX, fontScale, (0,0,0), thickness)

#     return img

def det_values(num, image, network, class_names, thre):
        detections = []
        darknet_image = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(darknet_image, image.tobytes()) 

        for i in range(num):
            detection = darknet.detect_image(network[i], class_names[i], darknet_image, thresh=thre)
            darknet.print_detections(detection, show_coordinates)
            detections.append(detection)
        darknet.free_image(darknet_image)
        return detections

def is_image(filename):
    return os.path.splitext(filename)[-1] in [".png", ".jpg"]

if __name__ == '__main__':
    cnt = 2
    for i in range(num):
        cfg_file.append(sys.argv[cnt+1])
        data_file.append(sys.argv[cnt])
        weight_file.append(sys.argv[cnt+2])
        if(i==num-1):
            imgdir = sys.argv[argc-1]
        cnt+=3

    color_cnt = 0
    for i in range(num):
        net, names, colors = darknet.load_network(
                cfg_file[i],
                data_file[i],
                weight_file[i],
                batch_size=1
            )
        network.append(net)
        class_names.append(names)
        class_colors.append(colors)
        if (color_cnt<2):
            for name in class_names[i]:
                class_colors[i][name] = COLORS[color_cnt] 
                color_cnt+=1

    global image_size, width, height
    for image_name in os.listdir(imgdir):
        if(is_image(image_name)==0):
            continue

        print("detect " + image_name + " ...")
        name = image_name.split('.jpg')[0]

        img = cv2.imread(os.path.join(imgdir, image_name))
        image_size = [img.shape[1],img.shape[0]]
        width = img.shape[1]
        height = img.shape[0]

        img_rgb = cv2.cvtColor( img, cv2.COLOR_BGR2RGB)
        
        detections = det_values(num, img_rgb, network, class_names, thre)
        for i in range(num):
            image = darknet.draw_boxes(detections[i], img_rgb, class_colors[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(save_result_dir, name+".jpg"), image)

