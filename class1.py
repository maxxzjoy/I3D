import cv2
import numpy as np
from PIL import Image
from numba import autojit 


_CAMERA_WIDTH = 640
_CAMERA_HEIGH = 480
_LABEL_TEXT_size = 320
_SAMPLE_VIDEO_FRAMES = 16
_IMAGE_SIZE = 224


@autojit
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


@autojit
def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

def Take1Element(a):
    return a[0]

def Take3Element(n):
    return n[2]

@autojit
def NMS_1d(BBox, threshold = 0.5):
    '''
        BBox = [ start, end, confidence, label ]
    '''
    if len(BBox) == 0:
        return []
    
    BBox.sort( key = Take3Element )
    BBox_array = np.array( BBox )
    
    picked_BBox = []
    
    startT = BBox_array[:,0]
    endT = BBox_array[:,1]
    confidence = BBox_array[:,2]
    
    areas = endT - startT
    
    '''Confidence score sort'''
    order = np.argsort(confidence)
    
    while order.size > 0:
        '''The index of largest confidence score'''
        index = order[-1]
        
        picked_BBox.append(BBox[index])
        
        t1 = np.maximum(startT[index], startT[order[:-1]])
        t2 = np.minimum(endT[index], endT[order[:-1]])
        
        IoU_area = np.maximum(0.0, t2-t1)
        
        ratio = IoU_area/(areas[index] + areas[order[:-1]] - IoU_area)
        
        left = np.where(ratio < threshold)
        order = order[left]
    
    picked_BBox.sort( key = Take1Element )
    return picked_BBox

@autojit
def load_pre_trained_model(sess, Path, rgb_saver_savedata):
    rgb_saver_savedata.restore(sess, Path)
    print('RGB checkpoint restored')
    print('RGB data loaded')
      
@autojit   
def get_class_labels(file_name):
    answer_names = []
    classInd_lines = open(file_name, 'r')
    classInd_list = list(classInd_lines)
    for index in range(len(classInd_list)) :
        answer = classInd_list[index].strip('\n').split()
        answer_names.append(answer[0])
        
    return answer_names 

@autojit
def draw_label(classInd_list,color_base=(255,0,0),answer=-1,color_highlight=(0,0,255)):
    label_image = []
    label_image = np.zeros((_CAMERA_HEIGH, _LABEL_TEXT_size, 3), np.uint8)
    label_image.fill(255) #畫布顏色
    start_pixel = 50
    step = 27
    text = 'Classes'
    cv2.putText(label_image, text, (120, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1, cv2.LINE_AA)
    for i in range(0,len(classInd_list)):
        if i==answer:
            cv2.putText(label_image, classInd_list[answer], (10, start_pixel+step*i), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_highlight, 1, cv2.LINE_AA)
        else :
            cv2.putText(label_image, classInd_list[i], (10, start_pixel+step*i), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_base, 1, cv2.LINE_AA)

    return label_image


def window_display(frame, label_images, mode):
    merage_image = np.zeros((_CAMERA_HEIGH, _CAMERA_WIDTH+_LABEL_TEXT_size, 3), np.uint8)
    merage_image[0:_CAMERA_HEIGH,0:_CAMERA_WIDTH] = frame
    merage_image[0:_CAMERA_HEIGH, _CAMERA_WIDTH:_CAMERA_WIDTH+_LABEL_TEXT_size] = label_images
    
    return merage_image


def I3DVideoStackAppend(StackFun, image, ImageSize=224):
    img = Image.fromarray(image.astype(np.uint8))
    if(img.width>img.height):
        scale = float(ImageSize)/float(img.height)
        img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), ImageSize))).astype(np.float32)
    else:
        scale = float(ImageSize)/float(img.width)
        img = np.array(cv2.resize(np.array(img),(ImageSize, int(img.height * scale + 1)))).astype(np.float32)
    crop_x = int((img.shape[0] - ImageSize)/2)
    crop_y = int((img.shape[1] - ImageSize)/2)
    img = img[crop_x:crop_x+ImageSize, crop_y:crop_y+ImageSize,:]
    StackFun(img)

'''
    Using opencv to open a video
'''
class Openvideo(object):
    def __init__(self, filename = '', mode = 'video'):
        self.filename = filename
        self.mode = mode
        
    def __enter__(self):
        if self.mode == 'camera':
            self.cap = cv2.VideoCapture(0)
            # 設定擷取影像的尺寸大小
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAMERA_HEIGH)
            return self.cap
        elif self.mode == 'video':
            print('Read video: ' + self.filename)
            self.openfile = cv2.VideoCapture('./' + self.filename)
            return self.openfile
        else:
            print('Fail to open video or camera.')
            return None
    
    def __exit__(self, type, value, traceback):
        if self.mode == 'camera':
            print('Close camera')
            self.cap.release()
        else:
            print('Close video: ' + self.filename)
        cv2.destroyAllWindows()
        

'''
    Class for tensorflow to generate the segments of sliding window
    
    Example:
    |================| Video stream
    |              --| 1st segment (OutputStack[0]) x1
    |            ----| 2nd segment (OutputStack[1]) x2
    |        --------| 3rd segment (OutputStack[2]) x4
    |----------------| 4th segment (OutputStack[3]) x8
'''
class SlidingWindow(object):
    """
        Initialize all parameters from start
            __ForeStack : Local variable, store a sequence of video
            OutputStack : Return variable, segments of sliding window
            InputStack : Input video array
            MinFrame : Parameter of frame size
            LayersOfScale : Number of sliding window layers
            State : Number of segments
    """
    @autojit
    def __init__(self, ImageStackInput, MinFrame=16, LayersOfScale=3):
        self.__ForeStack = []
        self.OutputStack = []
        self.InputStack = ImageStackInput
        self.MinFrame = MinFrame
        self.LayersOfScale = LayersOfScale
        self.State = 0
        self.__ForeStackInsert = self.__ForeStack.insert
        self.__OutputStackAppend = self.OutputStack.append
        
    @autojit
    def __enter__(self):
        
        for layer in range(self.LayersOfScale ):
            if(len(self.InputStack) >= (2**layer * self.MinFrame) ):
                """(Change)Read the memory address"""
                for index in range(-1, (-self.MinFrame * 2**layer - 1), -2**layer):
                    self.__ForeStackInsert(0, self.InputStack[index])   
                
                self.__OutputStackAppend( np.array([self.__ForeStack]).astype(np.float32) )
                self.__ForeStack.clear()
                self.State = self.State + 1
            else:
                break
        return self
    
    def __exit__(self, type, value, traceback):
        pass
        



