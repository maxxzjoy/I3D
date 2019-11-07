from class1 import NMS_1d, Openvideo, SlidingWindow, load_pre_trained_model,\
                   get_class_labels, draw_label, window_display, I3DVideoStackAppend
import time
import cv2
import numpy as np
import tensorflow as tf
import i3d
import sys
sys.path.append('C:/Yolo/darknet-master/build/darknet/x64')
#import darknet

'''Initial parameter'''
SAMPLE_FRAME = 16
SLIDING_WINDOW_LAYER = 1
MAXIMUM_FRAME = ( SAMPLE_FRAME * 2**(SLIDING_WINDOW_LAYER-1) )
MODE = 'camera' # camera or video
_CAMERA_WIDTH = 640
_CAMERA_HEIGH = 480
_LABEL_TEXT_size = 320
_SAMPLE_VIDEO_FRAMES = 16
_IMAGE_SIZE = 224
NUM_CLASSES = 12
_BATCH_SIZE = 1
FPS = 0.0
STEP = 3
prelabel = 11

VideoStack=[]
VideoStackAppend = VideoStack.append
VideoStackPop = VideoStack.pop


classes_name = 'ClassDay.list' 
_CHECKPOINT_PATHS = {
    'rgb': 'models/i3d_ntuRGBD_model-124080',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'models3b/rgb_model-28044', #change model
    'rgb_Day': 'modelsTWCC20191030/rgb_model-302560',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

'''Definition of tensorflow'''
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('eval_type', 'rgb', 'rgb, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')
tf.logging.set_verbosity(tf.logging.INFO)
eval_type = FLAGS.eval_type
imagenet_pretrained = FLAGS.imagenet_pretrained


'''Placeholder of input images'''
labels_placeholder = tf.placeholder(tf.float32, [_BATCH_SIZE, NUM_CLASSES])
keep_prob = tf.placeholder(tf.float32)



'''Build the proposal network with i3d'''
# Proposal RGB input has 3 channels.
proposal_input = tf.placeholder(
    tf.float32,
    shape=(_BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

with tf.variable_scope('RGB'):
  proposal_model = i3d.InceptionI3d( NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits' )
  proposal_logits, _ = proposal_model( proposal_input, is_training=True, dropout_keep_prob=keep_prob )

proposal_variable_map = {}
proposal_saver_savedata = {}
for variable in tf.global_variables():
  if variable.name.split('/')[0] == 'RGB':
     proposal_variable_map[variable.name.replace(':0', '')] = variable        
proposal_saver_savedata =  tf.train.Saver(var_list=proposal_variable_map, reshape=True)


model_logits = proposal_logits

'''Tensorflow output definition''' 
model_predictions = tf.nn.softmax(model_logits)
output_class = tf.argmax(model_predictions, 1)

'''Initialize the tensorflow variavles'''
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

'''Initialize the display window image'''
merage_image = np.zeros((_CAMERA_HEIGH, _CAMERA_WIDTH+_LABEL_TEXT_size, 3), np.uint8)

'''Get the class labels'''
answer_names = get_class_labels(classes_name) 

f_list = open('act_outputDay.txt','w')
f_Pick = open('act_outPick.txt','w')

with tf.Session() as sess:
    
    sess.run(init)
    load_pre_trained_model(sess,  _CHECKPOINT_PATHS['rgb_Day'], proposal_saver_savedata)
    cou = 0
    
    '''BBox = [ start, end, confidence, label ]'''
    BBox = []
    
    with Openvideo(filename = '3.avi', mode = MODE) as vc:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        
        while(vc.isOpened()):
            rval , frame = vc.read()
            
            if rval == True:
                
                label_images = draw_label(answer_names,(200,200,200), prelabel)
                
                cou = cou + 1
                
                if( cou%STEP == 0):
                    '''Stack the video'''
                    I3DVideoStackAppend(VideoStackAppend, frame)
                    
                    if( len(VideoStack) > MAXIMUM_FRAME ):
                        VideoStackPop(0)
                    
                    '''Timer start'''
                    if(len(VideoStack) >= SAMPLE_FRAME):
                        ti1 = time.time()
                    
                    '''Sliding window segmentation'''
                    with SlidingWindow(VideoStack, MinFrame=SAMPLE_FRAME, LayersOfScale=SLIDING_WINDOW_LAYER) as SlidingSegment:
                        rgb_buffer = SlidingSegment.OutputStack
                    
                        if SlidingSegment.State:
                            
                            f_list.write( str(cou) + ',' )
                            for slidinglayer in range(SlidingSegment.State):
                                class_out = sess.run([model_predictions], feed_dict={proposal_input: rgb_buffer[slidinglayer], keep_prob: 1})
                                
                                a=list(class_out[0][0])
                                prelabel = int(a.index(max(a)))
                                print( '{}  0:{:.2f}  1:{:.2f}  2:{:.2f}  3:{:.2f}  4:{:.2f}  5:{:.2f}  6:{:.2f}  7:{:.2f}  8:{:.2f}  9:{:.2f}  10:{:.2f}  11:{:.2f}'.format(\
                                prelabel, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11] ))
                                f_list.write( str(prelabel) + ',' )
                                
                                label_images = draw_label(answer_names,(200,200,200), prelabel)
                                
                            BBox.append([ (cou-(STEP*SAMPLE_FRAME)), cou, max(a), prelabel])
                            
                        f_list.write( '\n' )
                        
                    '''Timer end'''
                    if(len(VideoStack) >= SAMPLE_FRAME):
                        FPS = 1/(time.time()-ti1)
                        
                if( len(BBox) > 100):
                    Picked_BBox = NMS_1d(BBox)
                    for n in Picked_BBox:
                        f_Pick.write('{}, {}, {}, {}\n'.format(n[0], n[1], n[2], n[3]))
                    BBox.clear()
                
                cv2.putText(frame, 'FPS:' + '{:5.3f}'.format(FPS), (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)
                Display = window_display(frame, label_images, 'camera')
                cv2.imshow('frame', Display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                break
    f_list.close()
    f_Pick.close()