#!/usr/bin/env python3

# for rosrun
# sys.path.append(os.path.dirname(os.path.join(os.getcwd(), os.path.pardir))+ "/src")
#for roslaunch 
import sys
import os
Workspace_Path = sys.path[0].rsplit('/',2)[0]
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)) + "/yolo_detect_ws/src")
sys.path.append(Workspace_Path)
sys.path.insert(1, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
# print(sys.path)
# last_path = sys.path[0].rsplit('/',2)[0]
# print(last_path)
import cv2
import time
# import python.darknet as darknet
import darknet_new.darknet as darknet
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pallet_det.msg import bbox, bboxes

COLORS_YELLOW = (255, 191, 0)
COLORS_PINK = (255, 20, 147)
COLORS = [COLORS_PINK, COLORS_YELLOW]

win_title = 'YOLOv4 test'
# save_result_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)) + "/yolo_detect_ws/src/pallet_det/output_video"
save_result_dir = Workspace_Path + '/pallet_det/output_video'
out_video_name = "real_output.avi"
if not os.path.exists(save_result_dir):
    os.mkdir(save_result_dir)

show_coordinates = True
out_video_fps = 15

#字體與寬度設定
fontFace = cv2.FONT_HERSHEY_COMPLEX
fontScale = 0.5
font_thickness = 1
rect_thickness = 2

def load_files():
    global data_file, cfg_file, weights_file, num, thre, camera_path, width, height
    data_path = rospy.get_param('/data_path')
    cfg_path = rospy.get_param('/config_path')
    weights_path = rospy.get_param('/weights_path')

    data_file = rospy.get_param('/data_file/names') 
    cfg_file = rospy.get_param('/config_file/names')
    weights_file = rospy.get_param('/weights_file/names')
    num = rospy.get_param('/model_num/value')
    thre = rospy.get_param('/threshold/value')
    camera_path = rospy.get_param('/image')
    width = rospy.get_param('/color_width')
    height = rospy.get_param('/color_height')
    for i in range(num):
        data_file[i] = data_path+data_file[i]
        cfg_file[i] = cfg_path+cfg_file[i]
        weights_file[i] = weights_path+weights_file[i]

class Detect_node():
    def __init__(self, cfgfile, weightsfile, datafile):
        rospy.init_node("Detect_node")
        self.bridge = CvBridge()
        # 設定ROS的訂閱者，接收相機的影像資訊
        self.image_sub = rospy.Subscriber(camera_path, Image, self.imageCallback)
        self.pub_objs = rospy.Publisher('/darknet_ros/bounding_boxes', bboxes, queue_size=10)
        self.num = num
        self.out = cv2.VideoWriter(os.path.join(save_result_dir, out_video_name), cv2.VideoWriter_fourcc(*'XVID'), out_video_fps, (width, height))
        self.init_model(cfgfile, datafile, weightsfile)
        rospy.spin()     

    def init_model(self, cfg_file, data_file, weight_file):
        self.network = []
        self.class_names = []
        self.class_colors = []
        cnt = 0
        for i in range(self.num):
            net, names, colors = darknet.load_network(
                cfg_file[i],
                data_file[i],
                weight_file[i],
                batch_size=1
            )
            self.network.append(net)            
            self.class_names.append(names)      #type:str
            self.class_colors.append(colors)    #type:dict
            
            #將前兩種label的顏色改為粉色與黃色
            if (cnt<2):
                for name in self.class_names[i]:
                    self.class_colors[i][name] = COLORS[cnt] 
                    cnt+=1
    
    #讀取detections
    def det_values(self, image, thre):
        self.detections = []
        darknet_image = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(darknet_image, image.tobytes()) 

        for i in range(self.num):
            detection = darknet.detect_image(self.network[i], self.class_names[i], darknet_image, thresh=thre)
            darknet.print_detections(detection, show_coordinates)
            self.detections.append(detection)
        darknet.free_image(darknet_image)
    
    # #畫預測匡
    # def draw_boxes(self, img, det, colors): 
    #     for (label, score, box) in det:
    #         text = label+ ":"+str(score)
    #         xmin, ymin, xmax, ymax = darknet.bbox2points(box)

    #         #for text background
    #         labelSize = cv2.getTextSize(text, fontFace, fontScale, font_thickness)
    #         _x1 = xmin # bottom_left x of text
    #         _y1 = ymin # bottom_left y of text
    #         _x2 = xmin+labelSize[0][0] # top_right x of text
    #         _y2 = ymin-labelSize[0][1] # top_right y of text
    #         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[label], rect_thickness)
    #         cv2.rectangle(img, (_x1,_y1), (_x2,_y2), colors[label], cv2.FILLED) # text background
    #         cv2.putText(img, text, (xmin, ymin - 4), cv2.FONT_HERSHEY_COMPLEX, fontScale, (0,0,0), font_thickness)

    #     return img 
    
    #publish msg
    def msg_pub(self):
        bbs_objs = bboxes()
        for i in range(self.num):
            for (names, score, box) in self.detections[i]:
                bb = bbox()
                bb.xmin, bb.ymin, bb.xmax, bb.ymax = darknet.bbox2points(box)
                bb.score = float(score)
                bb.object_name = names
                bbs_objs.bboxes.append(bb)
        self.pub_objs.publish(bbs_objs)

    def imageCallback(self, img_msg):
        t_prev = time.time()
        try:
            # 將ROS的影像資訊轉換成OpenCV的影像格式
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(e)

        #Find detections(label_name, score, boxes)
        self.detections = []
        image_rgb = cv2.cvtColor( cv_image, cv2.COLOR_BGR2RGB)
        # image_rgb = cv2.flip(image_rgb, 1)
        image_resized = cv2.resize( image_rgb, (width, height))
        self.det_values(image_resized, thre)

        #將boxes的值publish到/darknet_ros/bounding_boxes上
        self.msg_pub() 

        #畫預測匡
        for i in range(self.num):
            final_image = darknet.draw_boxes(self.detections[i], image_resized, self.class_colors[i])
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

        #display fps
        fps = int(1/(time.time()-t_prev))
        cv2.rectangle(final_image, (5, 5), (75, 25), (0,0,0), -1)
        cv2.putText(final_image, f'FPS {fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 將影像保存
        self.out.write(final_image)

        # 將影像顯示在視窗中
        cv2.namedWindow(win_title,0)
        cv2.resizeWindow(win_title, 700, 500)
        cv2.moveWindow(win_title, 20, 20)
        cv2.imshow(win_title, final_image)

        # 按下q鍵退出程式
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("quit")

if __name__ == '__main__':
    load_files()
    node = Detect_node(cfg_file, weights_file, data_file)