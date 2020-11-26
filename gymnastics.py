import argparse as ap
import logging as log
import time
from pprint import pprint as pp
import cv2
import numpy as np
import sys
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math

logger = log.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(log.DEBUG)
ch = log.StreamHandler()
ch.setLevel(log.DEBUG)
formatter = log.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

show_fps = 0

def findPoint(pose, p):
    for point in pose:
        try:
            body_part = point.body_parts[p]
            return(int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
        except:
            return(0, 0)
        return(0, 0)

def euclidian(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def angleCalculation(p0, p1, p2):
    '''
        The center point where we measure the angle between p1, p0 and p2
    '''
    try:
        a = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
        b = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        c = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
        angle = math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi
    except:
        return 0
    return int(angle)

def handStandPose(a, b, c, d, e):
    '''
        Info
    '''
    if a in range(0, 180) and b in range(20, 200) and c in range(90, 150) and d in range(120, 175) and e in range(140, 195):
        return True
    return False

def spagatPose(a, b, c, d):
    # There are ranges of angle and distance to for spagat. 
    '''
        a and b are angles of hips 9 and 12
        c and d are the distance from the ankles because in the spagat the distance will be the maximum.
    '''
    if (a in range(50,250) or b in range(50,250)) and (c in range(50,250) or d in range(50,250)):
        return True
    return False

def poseTextInformation(dst, changes, s, color, scale):
    (x, y) = changes
    if (color[0] + color[1] + color[2] == 255 * 3):
        cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness = 1, lineType = 10)
    else:
        cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness = 1, lineType = 10)
    #cv2.line
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), lineType = 11)

if __name__ == '__main__':
    parser = ap.ArgumentParser(description = 'tf-pose-estimation realtime webcam')
    parser.add_argument('--gym',    type = str, default = 0)
    parser.add_argument('--resize', type = str, default = '432x368', help = 'Default = 432x368, Recommends: 432x368, 656x368 or 1312x736')
    parser.add_argument('--resize-out-ratio', type = float, default = 4.0, help = 'Default = 1.0')
    parser.add_argument('--model', type=str, default = 'cmu', help = 'cmu / mobilenet_thin')
    parser.add_argument('--show-process', type = bool, default = False, help = 'Hata ayiklama amaciyla etkinlestirildiginde, cikarim hizi dusurulur.')
    args = parser.parse_args()

    print("Mode 1: Handstand \nMode 2: Spagat \nMode 3: Bridge \nMode 4: Null \nMode 5: Null")
    mode = int(input("Enter a mode : "))

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size = (w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size = (432, 368))
    logger.debug('Camera or Video Reading... ')
    cam = cv2.VideoCapture(args.gym)
    ret, image = cam.read()
    logger.info('Camera Image = %dx%d' % (image.shape[1], image.shape[0]))

    count     = 0
    i         = 0
    frm       = 0
    y1        = [0, 0]
    red_color = (0, 0, 255)
    global height, width

    while True: # or while(1):
        ret, image = cam.read()
        i = 1
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size = args.resize_out_ratio)
        pose = humans
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy = False)
        height, width = image.shape[0], image.shape[1]

        if (mode == 1): # Handstand
            if len(pose) > 0:
                # distance calculations
                leftHeadHandDistence  = int(euclidian(findPoint(pose, 0), findPoint(pose, 7)))
                rightHeadHandDistence = int(euclidian(findPoint(pose, 0), findPoint(pose, 4)))
                distenceBetweenHands  = int(euclidian(findPoint(pose, 7), findPoint(pose, 4)))
                # angle calculations
                angle1 = angleCalculation(findPoint(pose, 6), findPoint(pose, 5), findPoint(pose, 1))
                angle2 = angleCalculation(findPoint(pose, 3), findPoint(pose, 2), findPoint(pose, 1))

                if (mode == 1) and handStandPose(distenceBetweenHands, angle1, angle2, leftHeadHandDistence, rightHeadHandDistence):
                    result = "Handstand"
                    is_gymnastic = True
                    poseTextInformation(image, (200, 200), result, red_color, 2)
                    logger.debug(" HANDSTAND POSE ")

        elif (mode == 2): # Spagat
            if len(pose) > 0:
                # distance calculations
                leftLegDistence       = int(euclidian(findPoint(pose, 8),  findPoint(pose, 14)))
                rightLegDistence      = int(euclidian(findPoint(pose, 8),  findPoint(pose, 11)))
                distenceBetweenAnkle  = int(euclidian(findPoint(pose, 14), findPoint(pose, 11)))
                # angle calculations
                angle3 = angleCalculation(findPoint(pose, 14), findPoint(pose, 13), findPoint(pose, 12))
                angle4 = angleCalculation(findPoint(pose, 11), findPoint(pose, 10), findPoint(pose, 9))
                # angle5 = angleCalculation(findPoint(pose, 4),  findPoint(pose, 3),  findPoint(pose, 2))
                # angle6 = angleCalculation(findPoint(pose, 8),  findPoint(pose, 9),  findPoint(pose, 10))

                if(mode == 4) and spagatPose(angle3, angle4, leftLegDistence, rightLegDistence, distenceBetweenAnkle):
                    result = "Spagat"
                    is_gymnastic = True
                    poseTextInformation(image, (200, 200), result, red_color, 2)
                    logger.debug(" SPAGAT POSE ")

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - show_fps)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if(frm == 0):
            out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (image.shape[1], image.shape[0]))
            print("Initializing...")
            frm+=1
        cv2.imshow('Gymnastic Movements Detection', image)
        if i != 0:
            out.write(image)
        show_fps = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()