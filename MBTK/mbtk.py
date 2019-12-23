#!/usr/bin/env python3

#
# Shi-Wei Lo, NCHC, 2018/10/20.
#


'''
    2018-03-09
    Testing pyc, pyo module.
    The module file.
    '''
'''
    2018-10-19
    Testing Cython module to create shared library file whcih
    calling by import and from.
    '''


# name it mtk should be ok!
'''
    # OK
    import MBTK.mbtk as mtk
    mtk.show_module_flag()
    
    # OK
    from MBTK.mbtk import show_module_flag
    show_module_flag()
    
    # OK
    from MBTK import mbtk as mtk
    mtk.show_module_flag()
    '''


import os
import sys
import time
import math
import json
import glob
import urllib
import random
import requests
import argparse
import platform

import cv2
import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

import mysql.connector
from mysql.connector import errorcode

from scipy.interpolate import griddata
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import maskoceans

from pyfiglet import Figlet
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from MBTK.help import bad_link, TNb_list





# Create a print() function
def ptk_print(words):
    print("ptk_print:", words)


# TF logging level output control
def tf_cpp_log_level():
    #os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    #0 = all messages are logged (default behavior)
    #1 = INFO messages are not printed
    #2 = INFO and WARNING messages are not printed
    #3 = INFO, WARNING, and ERROR messages are not printed
    #有些博客上的說明有誤須注意。
    os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
def tf_cpp_log_level_1():
    os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
def tf_cpp_log_level_2():
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
def tf_cpp_log_level_3():
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
def tf_log_silence():
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
def tf_log_noisily():
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# show steps
def show_steps_flag(N):
    figlet = Figlet(font='epic', width=100)
    # font:
    # alphabet doh big cyberlarge cybermedium
    # epic moscow letters standard  pebbles starwars
    print()
    print(figlet.renderText('              ' + 'step'))
    print(figlet.renderText('           --  {}  --'.format(N)))
    print(figlet.renderText('              ' + 'step'))
    print()
    
    #print(figlet.renderText('              ' + 'step'))
    #print(figlet.renderText('Classifier'))
    
    #ft = Figlet(font='letters', width=150)
    #print(ft.renderText('{} {}'.format(sys.platform, platform.machine())))
    time.sleep(2)
    
# show my flag
def show_module_flag():
    figlet = Figlet(font='epic', width=100)
    # font:
    # alphabet doh big cyberlarge cybermedium
    # epic moscow letters standard  pebbles starwars
    print()
    print(figlet.renderText('              ' + 'NCHC'))
    print(figlet.renderText('Classifier'))
    
    ft = Figlet(font='letters', width=150)
    print(ft.renderText('{} {}'.format(sys.platform, platform.machine())))
    time.sleep(2)


# show pkg flag
#def show_pkg_flag():
def ShowPyOcvOSVersion():
    #print("--Runing", os.name)
    #print("--OS is", sys.platform)
    #print("--OS platform is", platform.machine())
    print("--Python version = ", sys.version_info[0], ".", sys.version_info[1], ".", sys.version_info[2])#, sep='')
    print("--OpencV version =", cv2.__version__)
    print("--OS platform =", os.name, ",", sys.platform, ",", platform.machine())


# LSW 2017-07-11 #
# LSW 2018-10-09 #
# 依照數字大小排序 避免倒帶輸出 img_list.sort(key= lambda x:int(x[:-4]))
# 但是帶前綴文字的檔名需要分開處理！！
def load_dir(dirname):
    '''LSW load the dir path, get the image list of paths'''
    print("Infer. image Dir. :", dirname)
    imgNum=0
    countNum=0
    img_list = os.listdir(dirname)
                
    #img_list.sort(key= lambda x:int(x[:-4]))

    for imgname in img_list:
        imgNum = imgNum+1
        #print("# ", imgNum, "path:", imgname)
        '''
            #  1 path: 161006221556-100.bmp.jpg
            #  2 path: 161105054208-57.bmp.jpg
            #  3 path: 160121193737-436.bmp.jpg'''
            
    return img_list, imgNum


# how to send GLOBAL variable to other py
def plable_number_initial():
    print()

# LSW: 2018-01-08, Just for counting hits for each labels. Yes, it's a GLOBAL variable.

'''
    using import to use the blobal variable to other py file.
    # Magick number
    from MBTK.mbtk import P1,P2,P3,P4,P5,P6
    print(P1)
'''
'''
    L also found that man.py usisng "import mtk.PRCounter_CROSS" will not effects the P-n variables.
    place PRCounter_CROSS() in man.py works.
    
    '''
from MBTK.Pnum import P1,P2,P3#,P4,P5,P6
#P1=0
#P2=0
#P3=0
#P4=0
#P5=0
#P6=0
def PRCounter_CROSS(human_string):
    if human_string == 'floods':
        global P1
        P1 +=1
    if human_string == 'normal':
        global P2
        P2 +=1
    if human_string == 'unknow':
        global P3
        P3 +=1
#    if human_string == '4 ng spot':
#        global P4
#        P4 +=1
#    if human_string == '5 group ng spot':
#        global P5
#        P5 +=1
#    if human_string == '6 others':
#        global P6
#        P6 +=1

# Seems this not a regular way to share value with different module/function call as a GLOBAL.
"""
    Using https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
    to replace.
    Each module/call should import MBTK.Pnum once as a global name.
    
    # MBTK.Pnum.py
    flag_index = 0

    # Magic help
    import MBTK.Pnum
    MBTK.Pnum.flag_index +=1
    
    # run.py
    import MBTK.Pnum
    print(MBTK.Pnum.flag_index)
"""
from MBTK.Pnum import flag_index
def flag_add_one():
    global flag_index
    flag_index +=1


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
            
    return graph


# 修改自def load_graph(model_file):
def graph_parser(model_file,input_layer,output_layer):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    return graph,input_operation,output_operation



def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
#    print('Check inside [read_tensor..]', file_name)
    with tf.Graph().as_default(): # Add as_default() to initial graph at each time sess.run.
#        print('Check inside [read_tensor..] as_default()')
        input_name = "file_reader"
        output_name = "normalized"
#        print('Check inside [read_tensor..] output_name', output_name)
        file_reader = tf.read_file(file_name, input_name)
#        print('Check inside [read_tensor..] file_reader', input_name)
        if file_name.endswith(".png"):
            image_reader = tf.image.decode_png(file_reader, channels = 3, name='png_reader')
        elif file_name.endswith(".gif"):
            image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name='gif_reader'))
        elif file_name.endswith(".bmp"):
            image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
        else:
            image_reader = tf.image.decode_jpeg(file_reader, channels = 3, name='jpeg_reader')
#            print('Check inside [read_tensor..] jpeg_reader')
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0);
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
#        print('Check inside [read_tensor..] sess = tf.Session()')
        result = sess.run(normalized)
#        print('Check inside [read_tensor..] result=sess.run(normalized)')

    return result



#LSW 2018-10-09 add stream video inference realtime
def read_tensor_from_video_stream(frame, input_height=299, input_width=299,
                                  input_mean=0, input_std=255):
    with tf.Graph().as_default(): # Add as_default() to initial graph at each time sess.run.
        input_name = "file_reader"
        output_name = "normalized"
#        file_reader = tf.read_file(file_name, input_name)
#        if file_name.endswith(".png"):
#            image_reader = tf.image.decode_png(file_reader, channels = 3,
#                                               name='png_reader')
#        elif file_name.endswith(".gif"):
#            image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
#                                                          name='gif_reader'))
#        elif file_name.endswith(".bmp"):
#            image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
#        else:
#            image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
#                                                name='jpeg_reader')
        float_caster = tf.cast(frame, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0);
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)
    
    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label



# parser = argparse.ArgumentParser()

def arg_parser():

    file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
    model_file = "tf_files/retrained_graph.pb"
    label_file = "tf_files/retrained_labels.txt"
    #input_height = 224 #128 224
    #input_width = 224 #128 224
    #input_mean = 128
    #input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", default="224", type=int, help="input height")
    parser.add_argument("--input_width", default="224", type=int, help="input width")
    parser.add_argument("--input_mean", default="128", type=int, help="input mean")
    parser.add_argument("--input_std", default="128", type=int, help="input std")
#    parser.add_argument("--input_layer", help="name of input layer")
#    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--input_layer", default="input", help="name of input layer")
    parser.add_argument("--output_layer", default="final_result", help="name of output layer")

    
    #LSW: add args.
    parser.add_argument('--dir', default='/tmp', type=str, help='Absolute path to dir.')
    #parser.add_argument('--hit', required=True, type=str, default='all', help='A label name of hit performance or ALL for all labels.')
    parser.add_argument('--hit', required=True, type=str, default='all', help='A label name of hit performance or ALL for all labels.')
    
    parser.add_argument('--video_out', default='TT_STREAMTEST.mp4', type=str, help='A video name of testing output.')
    parser.add_argument('--imwritedir', default='/tmp/imwrite', type=str, help='Absolute path to save img to dir.')

        
        
    args = parser.parse_args()
    
    if args.graph:
        model_file = args.graph
    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer
    if args.dir:
        dir = args.dir
    if args.hit:
        hit = args.hit
    if args.imwritedir:
        imwritedir = args.imwritedir

    return args,model_file,file_name,label_file,input_height,input_width,input_mean,input_std,input_layer,output_layer,dir,hit, imwritedir


#def mbtk_wra_road_stream_multicam():



'''
    #NOTE: copy from: #label_image_MBNet_Images_DIRLoop_GraphDefault_PRcounting_HuWeiSP_LSW_slit_show_WRARoad_stream_multiCam_mbtk.py
    
    USAGE:
    
    # LSW 2018-10-19
    目前--video_out=會因會camera解析度不同而無法存檔，僅對單一camera或相同frame size者錄影。
    Host>
    $ vidsom ->> http://localhost:8097/
    Client>
    stream_multiCam_mbtk
    
    # 因為使用url因此取消掉input image dir的--dir參數，以及--video_out參數
    $ time py -m scripts.label_image_MBNet_Images_DIRLoop_GraphDefault_PRcounting_HuWeiSP_LSW_slit_show_WRARoad_stream_multiCam_mbtk --graph=tf_files/rt_wraroad_Tall_ims_128_mf_0.25_steps_500.pb   --labels=tf_files/rt_wraroad_Tall_labels.txt --hit=all
    
    # 原本所有參數版本
    $ time py -m scripts.label_image_MBNet_Images_DIRLoop_GraphDefault_PRcounting_HuWeiSP_LSW_slit_show_WRARoad_stream_multiCam_mbtk --graph=tf_files/rt_wraroad_Tall_ims_128_mf_0.25_steps_500.pb   --labels=tf_files/rt_wraroad_Tall_labels.txt  --dir=/Volumes/256gb/WRA_road_mac_tmp/N_F_U/T1_437K500_201707xx_evl_imgs/ --hit=all --video_out=TT_STREAMTEST.mp4
    
    stream_multiCam
    $ time py -m scripts.label_image_MBNet_Images_DIRLoop_GraphDefault_PRcounting_HuWeiSP_LSW_slit_show_WRARoad_stream_multiCam --graph=tf_files/rt_wraroad_Tall_ims_128_mf_0.25_steps_500.pb   --labels=tf_files/rt_wraroad_Tall_labels.txt  --dir=/Volumes/256gb/WRA_road_mac_tmp/N_F_U/T1_437K500_201707xx_evl_imgs/ --hit=all --video_out=TT_STREAMTEST.mp4
    
    $ time py -m scripts.label_image_MBNet_Images_DIRLoop_GraphDefault_PRcounting_HuWeiSP_LSW_slit_show --graph=tf_files/rt_slit_train_set_c2_allin_ims_224_mf_1.0_steps_4000.pb  --labels=tf_files/rt_slit_train_set_c2_allin_labels.txt --dir=/Volumes/256gb/sumika/sumika/simple_test/ --hit=all
    
    $ time py -m scripts.label_image_MBNet_Images_LSW     --graph=tf_files/retrained_graph.pb      --dir=tf_files/flower_photos/daisy/
    
    $ time py -m scripts.label_image_MBNet_Images_LSW     --graph=tf_files/retrained_graph.pb      --dir=tf_files/test_1000_imgs/
    
    for 1000 images for test the time!
    
    '''

# 公路總局 14xx THB
def get_cctv_info():
    df = pd.read_csv('MBTK/cctvinfo.csv', index_col=0)
    index = df.shape[0]
    return df, index


# 台北市_config_SW20190603 
# def get_cctv_info():
#     df = pd.read_csv('MBTK/台北市_config_SW20190603.csv', index_col=0)
#     index = df.shape[0]
#     return df, index


#台南市交通局_config_SW20190603
# def get_cctv_info():
#     df = pd.read_csv('MBTK/台南市交通局_config_SW20190603.csv', index_col=0)
#     index = df.shape[0]
#     return df, index

#台南市警察局_config_SW20190603
# def get_cctv_info():
#     df = pd.read_csv('MBTK/台南市警察局_config_SW20190603.csv', index_col=0)
#     index = df.shape[0]
#     return df, index

#桃園市_config_SW20190603
# def get_cctv_info():
#     df = pd.read_csv('MBTK/桃園市_config_SW20190603.csv', index_col=0)
#     index = df.shape[0]
#    return df, index


# test MJPEG source
# def get_cctv_info_switch():
#     df = pd.read_csv('MBTK/cv2_switch_test.csv', index_col=0)
#     index = df.shape[0]
#     return df, index
# sum_thb_TP_TY_TNT_TNP.csv
# def get_cctv_info_switch():
#     df = pd.read_csv('MBTK/sum_thb_TP_TY_TNT_TNP.csv', index_col=0)
#     index = df.shape[0]
#     return df, index
# sum_thb_C00_TY_TNb_TNb_new, sum_C00_TY_TNb_TNb_only for test
# sum_thb_C00_TY_TNb_TNb_new_only_TNab.csv  for test only TNb problem!

# 2019-05-05 First merge all Thb, TY, C000, TNa, TNb.
#MBTK/sum_thb_C00_TY_TNb_TNb_new.csv'

# 2019-08-23 補齊所有px py座標，另加上桃園智慧下水道，總計2012個鏡頭。
# sum_thb_C00_TY_TNb_TNb_new_refill_20190823.csv

#MBTK/sum_thb_C00_TY_TNb_TNb_new_refill_20191018.csv 智慧下水道 南投交通局 高雄交通局
#MBTK/sum_thb_C00_TY_TNb_TNb_new_refill_20191005.csv 智慧下水道

def get_cctv_info_switch():
    df = pd.read_csv('MBTK/sum_thb_C00_TY_TNb_TNb_new_refill_20191018.csv')#, index_col=0) # for parallel mode! Do not use first col (col=0 == id) as index, id numbering will repeat in different city.
    index = df.shape[0]
    return df, index
# sum_thb_C00_TY_TNb_TNb_new

#MBTK/sum_thb_C00_TY_TNb_TNb_new_refill_20190823_test_C1C2.csv'
def get_cctv_info_switch_C1C2():
    df = pd.read_csv('MBTK/sum_thb_C00_TY_TNb_TNb_new_refill_20190823_test_C1C2.csv')#, index_col=0) # for parallel mode! Do not use first col (col=0 == id) as index, id numbering will repeat in different city.
    index = df.shape[0]
    return df, index


def get_cctv_info_switch_south():
    df = pd.read_csv('MBTK/sum_thb_C00_TY_TNb_TNb_new_only_south.csv', index_col=0)
    index = df.shape[0]
    return df, index

def get_cctv_info_switch_only_one():
    df = pd.read_csv('MBTK/sum_thb_C00_TY_TNb_TNb_new_only_one.csv') # for parallel mode! Do not use first col (col=0 == id) as index, id numbering will repeat in different city.
    index = df.shape[0]
    return df, index

# only a few list
def get_cctv_info_1():
    df = pd.read_csv('MBTK/cctvinfo_1.csv', index_col=0)
    index = df.shape[0]
    return df, index


def get_frame_from_cv2_switch_cap(tvid, url):
    if tvid.startswith("TY"):
        #print("TY = ", tvid, url)
        
        try:
            stream=urlopen(url, timeout=10)
            BYTES=b''# MAKE IT BYTES
        except urllib.error.HTTPError as e:
            print('HTTPError: {} from {}'.format(e.code, tvid))
        except urllib.error.URLError as e:
            print('URLError: {}'.format(e.reason))
        else:
            while True:
                BYTES+=stream.read(10240)
                a = BYTES.find(b'\xff\xd8')
                b = BYTES.find(b'\xff\xd9')
                if a!=-1 and b!=-1:
                    jpg = BYTES[a:b+2]
                    BYTES= BYTES[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
                    # we now have frame stored in frame.
                    #cv2.imshow('cam2',frame)
                    if len(jpg) == 0:
                        break;
                    
                    return frame
# Press 'q' to quit
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break

    #            return frame

    if tvid.startswith("thbCCTV") or tvid.startswith("C000") or tvid.startswith("TNa") or tvid.startswith("TNb") or tvid.startswith("C1") or tvid.startswith("C2") or tvid.startswith("NT") or tvid.startswith("64000C"):
        #print("thb, TP, TN = ", tvid)
        
#        try:
#            print('Try url...', tvid)
#            stream=urlopen(url, timeout=5)
#        except urllib.error.HTTPError as e:
#            print('HTTPError: {} from {}'.format(e.code, tvid))
#        except urllib.error.URLError as e:
#            print('URLError: {}'.format(e.reason))
#        else:
#            print('cap url...', tvid)
        cap = cv2.VideoCapture(url)
            # Check stream
        if cap.isOpened(): #while(cap.isOpened()): is for continu read frame. TNb too slow will be problem.
            ret, frame = cap.read()
            return frame

        # Check stream (Old versijon)
#        if not cap.isOpened():
#            print("{} Cannot be Opened".format(url))
#            pass
#            #break # will break the for loop and outter while will run again.
#        else:
#            ##print("{} be Opened".format(url))
#            # Capture frame-by-frame
#            ret, frame = cap.read()
#            return frame



def req_post_joson_tvid(url, tvid, normal, floods, unknow, realtime):
    '''req_post_joson_tvid(url_2, tvid, normal, floods, unknow)'''
    #test = strftime("%Mm%Ss", gmtime())
    
    #id = random.randint(1,101)
    
    #url = url_2
    headers = {'Content-Type': 'application/json'}
    data = {"id":tvid, "normal":normal, "floods":floods, "unknow":unknow, "realtime":realtime}
    
    r = requests.post(url, headers=headers, json=data) #data=json.dumps(data)
    #r = requests.put('http://140.110.16.68/aimap/todo/api/v1.0/tasks', data = {"id":1, "normal":"1.99", "floods":"5.55", "unknow":"3.33"})
    print(r.status_code, r.reason, '--------------')
    #print(r.text, '--------------')
    #print(r.json)
    print(data)

def post_ELK(url, tvid, road, px, py, normal, floods, unknow, realtime, im_name, color_num):
    '''req_post_joson_tvid(url_2, tvid, normal, floods, unknow)'''
    #test = strftime("%Mm%Ss", gmtime())
    
    #id = random.randint(1,101)
    
    #url = url_2
    headers = {'Content-Type': 'application/json'}
    mesg = { "tvid":tvid, "road":road, "px":px, "py":py,"realtime":realtime, "im_name":im_name}
    data = { "normal":normal, "floods":floods, "unknow":unknow, "color_num":color_num}
    jj = {"mseg":mesg, "data":data}
    r = requests.post(url, headers=headers, json=jj) #data=json.dumps(data)

    print(r.status_code, r.reason, '--------------')
    #print(r.text, '--------------')
    #print(r.json)
    #print(mesg, data)
    print(jj)


def single_post_ELK(url, j_temp):
    ''' '''
    headers = {'Content-Type': 'application/json'}
#    mesg = { "tvid":tvid, "road":road, "px":px, "py":py,"realtime":realtime, "im_name":im_name}
#    data = { "normal":normal, "floods":floods, "unknow":unknow, "color_num":color_num}
#    jj = {"mseg":mesg, "data":data}
    r = requests.post(url, headers=headers, json=j_temp) #data=json.dumps(data)
    
    print(r.status_code, r.reason, '-------single_post_ELK JSON_TEMP-------')
    #print(r.text, '--------------')
    #print(r.json)
    #print(mesg, data)
    #print(jj)



def post_Linebot(tvid, road, realtime, im_name, im_full_name):
    # upload image to temp
    files = {'file': (im_name, open(im_full_name, 'rb'))}
    response = requests.post('https://mlab.nchc.org.tw/linebot/upload/', files=files)
    print(response.content)

    # send message to line
    content = 'imageUrl=https://mlab.nchc.org.tw/linebot/upload/' + im_name + '&thumbnailUrl=https://mlab.nchc.org.tw/linebot/upload/' + im_name + '&textContent=' + '[觀測時間] '+ realtime + '\n[觀測站] '+ tvid + '\n[位置] '+ road + '&groupId=C548424a26e1ca19af55daaad605aa9f6'
    response = requests.post('https://mlab.nchc.org.tw/linebot/message/', headers={'Content-type': 'application/x-www-form-urlencoded'}, data=content.encode('utf-8'))

def post_FLAG_Linebot(realtime, im_name, im_full_name):
    # upload image to temp
    files = {'file': (im_name, open(im_full_name, 'rb'))}
    response = requests.post('https://mlab.nchc.org.tw/linebot/upload/', files=files)
    print(response.content)

    # send message to line
    content = 'imageUrl=https://mlab.nchc.org.tw/linebot/upload/' + im_name + '&thumbnailUrl=https://mlab.nchc.org.tw/linebot/upload/' + im_name + '&textContent=' + '[觀測時間] '+ realtime + '\n[資訊頁面] ' + 'http://fmg.wra.gov.tw:8080/fmg/floodedAlarmList.php' + '&groupId=C548424a26e1ca19af55daaad605aa9f6'
    response = requests.post('https://mlab.nchc.org.tw/linebot/message/', headers={'Content-type': 'application/x-www-form-urlencoded'}, data=content.encode('utf-8'))




#def SESS_inf(t, input_operation, output_operation):
#    print('Cehck [SESS_inf]')
#    print('1 input_operation,output_operation', input_operation.outputs[0],output_operation.outputs[0])
#    with tf.Graph().as_default():
#        print('Cehck [SESS_inf] with tf.Graph().as_default()')
#        print('2 input_operation,output_operation', input_operation.outputs[0], output_operation.outputs[0])
#        sess = tf.Session(graph=graph)#, config=tf.ConfigProto(use_per_session_threads=True))
#        print('3 sess = tf.Session()')
#        result = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
#        print('4 result = sess.run()')
#    return result

""" 2019-08-18
    不能平行的原因在TF本身session，read_tensor這層可以，但外層esult = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})ㄗ則無作用。改變作法，只平行download跟POST階段，infer本身三分鐘跑完先暫時不做平行，。
    
    parall_excutor_if
    Still nor work for TF session, but some hit found here. the hit work for read_tensro... but not work result = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
    https://github.com/tensorflow/tensorflow/issues/14442
    
    """
#def parall_excutor_if(num_workers, img_list, df, imwritedir, saved_time_sec, JSON_TEMP, bad_link,
#                      input_operation, output_operation,
#                      input_height,
#                      input_width,
#                      input_mean,
#                      input_std):
#
#    processes = []
#    with ThreadPoolExecutor(max_workers=2) as executor:
#        for im_name in img_list:
#            processes.append(executor.submit(inf_im, im_name, df, imwritedir, saved_time_sec, JSON_TEMP, bad_link,
#                         input_operation, output_operation,
#                         input_height,
#                         input_width,
#                         input_mean,
#                         input_std))


"""
    lintbot_parall-dl-po.py
    """


# Parall download images then infer it once at time
def write_stream_frame(saved_time, imwritedir, tvid, url):
    #    if tvid not in TNb_list: # temporary skip TNb cctv, bcs it often hang for source.2019-08-20
#    print('start get frame', tvid)
    frame = get_frame_from_cv2_switch_cap(tvid, url)
#    print('end   get frame', tvid)
    if frame is not None:
        #im_name = tvid + '_' + time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time())) + '.jpg'
        im_name = tvid + '_' + saved_time + '.jpg'
        im_full_name = imwritedir + "/" + im_name
#        print('start save im_name', im_name)
        cv2.imwrite(im_full_name, frame)
#        print('\rSaved {}'.format(im_name), end="")# end="" not allow by Cython cimpile.
        print('Saved {}'.format(im_name))
    else:
#        print('URL not work:', tvid)
        print('Retry {}'.format(tvid))

def parall_excutor_dl(saved_time, imwritedir, df, index, num_workers):
    
    #processes = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for id_count in range(index):#for id_count in range(20):#for url in url_list:
            tvid = df.iloc[id_count]['cctvid']
            url = df.iloc[id_count]['url']
            #            processes.append(executor.submit(write_stream_frame, saved_time, imwritedir, tvid, url))
            executor.submit(write_stream_frame, saved_time, imwritedir, tvid, url)



def parall_excutor(saved_time, imwritedir, df, index, num_workers):
    
    processes = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for id_count in range(index):#for id_count in range(20):#for url in url_list:
            tvid = df.iloc[id_count]['cctvid']
            url = df.iloc[id_count]['url']
            processes.append(executor.submit(write_stream_frame, saved_time, imwritedir, tvid, url))


# 2019-08-16, post may take 3~5 sec. each time. Let us post all json at onece.
#                        mesg = { "tvid":tvid, "road":road, "px":px, "py":py,"realtime":saved_time_sec, "im_name":im_name}# realtime -> saved_time_sec
#                        data = { "normal":no_conf_len, "floods":ra_conf_len, "unknow":un_conf_len, "color_num":color_num}
#                        jj = {"mseg":mesg, "data":data}
#                        JSON_TEMP.append(jj)
#
# For older 'req_post_joson_tvid(url, tvid, normal, floods, unknow, realtime)'
# Parser j_temp to get tvid, normal, floods, unknow, realtime
#                for data in JSON_TEMP:
#                    tvid = str(data["mseg"]["tvid"])
#                    px = str(data["mseg"]["px"])
#                    py = str(data["mseg"]["py"])
#                    realtime = str(data["mseg"]["realtime"])
#                    im_name = str(data["mseg"]["im_name"])
#
#                    normal = str(data["data"]["normal"])
#                    floods = str(data["data"]["floods"])
#                    unknow = str(data["data"]["unknow"])
#                    address = str(data["mseg"]["road"])
#                    color_num = str(data["data"]["color_num"])
#                    print('tvid', tvid, address, im_name)

#                    start_req = time.time()
#                    req_post_joson_tvid(url_3, tvid, normal, floods, unknow, saved_time_sec)
#                    end_req = time.time()
#                    print('POST Time taken: [req]= {}\n'.format(end_req - start_req))
#
#                    start_ELK = time.time()
#                    #para_post_ELK(url_4, data)#not work, can not use j as post data. 20190820.
#                    #post_ELK(url, tvid, road, px, py, normal, floods, unknow, realtime, im_name, color_num):
#                    post_ELK(url_4, tvid, address, px, py, normal, floods, unknow, saved_time_sec, im_name, color_num) # OK GO
#                    end_ELK = time.time()
#                    print('POST Time taken: [req]= {}, [ELK]= {}\n'.format(end_req - start_req, end_ELK - start_ELK))
# Test JSON_TEMP
#                print('JSON_TEMP',  JSON_TEMP)
#                for j in JSON_TEMP:
#                    print(j)

# serial post
#                for j in JSON_TEMP:
#                    headers = {'Content-Type': 'application/json'}
#                    r = requests.post(url_4, headers=headers, json=j)
#                    print(r.status_code, r.reason, '--------------')
#                    print(j)
#or
#                for j in JSON_TEMP:
#                    para_post_ELK(url_4, j)# OK. 應該是server同時只允許一個post，serial都是正常。需注意lsdir順序不見得照字母數字。1586 cam serial post 85mins.




def print_j(j):
    print(j)
    print()


#def para_post_ELK(url, j):# not work, can not use j as post data. 20190820.要重新轉字串
#    ''' '''
#    #time.sleep(4 + (random.random() * 3))
#    headers = {'Content-Type': 'application/json'}
#    #    mesg = { "tvid":tvid, "road":road, "px":px, "py":py,"realtime":realtime, "im_name":im_name}
#    #    data = { "normal":normal, "floods":floods, "unknow":unknow, "color_num":color_num}
#    #    jj = {"mseg":mesg, "data":data}
#    #print(j)
#    start_ELK = time.time()
#    r = requests.post(url, headers=headers, json=j) #data=json.dumps(data)
#    end_ELK = time.time()
#    print(r.status_code, r.reason, '--------------', 'Spend ', end_ELK - start_ELK)
#    #print(r.text, '--------------')
#    #print(r.json)
#    #print(mesg, data)
#    print(j)

def parall_excutor_po_req(num_workers, url, JSON_TEMP, saved_time_sec):
    
    processes = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        #seem post only wrok with ThreadPool not ProcessPool,  ProcessPool will hang after laster post.
        # AWS lambada seems only taken 10 Tread post each time. max_workers=10 if >16 some tread fail, mayby it due to the Host physcial number of Treads of CPU!! ML-green 6 cpu cores may 12 Thread ==> try it 12
        #TWCC: Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz, 18 cores, 36 threads.
        #MLGreen:Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz, 6 cores, 12 threads. 10threads 1.6kCCTV about 28m28s.
        for data in JSON_TEMP:
            tvid = str(data["mseg"]["tvid"])
#            px = str(data["mseg"]["px"])
#            py = str(data["mseg"]["py"])
#            realtime = str(data["mseg"]["realtime"])
#            im_name = str(data["mseg"]["im_name"])

            normal = str(data["data"]["normal"])
            floods = str(data["data"]["floods"])
            unknow = str(data["data"]["unknow"])
#            address = str(data["mseg"]["road"])
#            color_num = str(data["data"]["color_num"])

            processes.append(executor.submit(req_post_joson_tvid, url, tvid, normal, floods, unknow, saved_time_sec))



def parall_excutor_po_ELK(num_workers, url, JSON_TEMP, saved_time_sec):
    
    processes = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        #seem post only wrok with ThreadPool not ProcessPool,  ProcessPool will hang after laster post.
        # AWS lambada seems only taken 10 Tread post each time. max_workers=10 if >16 some tread fail, mayby it due to the Host physcial number of Treads of CPU!! ML-green 6 cpu cores may 12 Thread ==> try it 12
        #TWCC: Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz, 18 cores, 36 threads.
        #MLGreen:Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz, 6 cores, 12 threads. 10threads 1.6kCCTV about 28m28s.
        for data in JSON_TEMP:
            tvid = str(data["mseg"]["tvid"])
            px = str(data["mseg"]["px"])
            py = str(data["mseg"]["py"])
            realtime = str(data["mseg"]["realtime"])
            im_name = str(data["mseg"]["im_name"])

            normal = str(data["data"]["normal"])
            floods = str(data["data"]["floods"])
            unknow = str(data["data"]["unknow"])
            address = str(data["mseg"]["road"])
            color_num = str(data["data"]["color_num"])
#            processes.append(executor.submit(print_j, j))
#            processes.append(executor.submit(para_post_ELK, url, data).result())##p.submit(task,i).result()即同步执行 == serial mode.
#            processes.append(executor.submit(para_post_ELK, url, data))
            processes.append(executor.submit(post_ELK, url, tvid, address, px, py, normal, floods, unknow, saved_time_sec, im_name, color_num))


def parall_excutor_po_ELK_serial(num_workers, url, JSON_TEMP, saved_time_sec):
    
    processes = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for data in JSON_TEMP:
            tvid = str(data["mseg"]["tvid"])
            px = str(data["mseg"]["px"])
            py = str(data["mseg"]["py"])
            realtime = str(data["mseg"]["realtime"])
            im_name = str(data["mseg"]["im_name"])
            
            normal = str(data["data"]["normal"])
            floods = str(data["data"]["floods"])
            unknow = str(data["data"]["unknow"])
            address = str(data["mseg"]["road"])
            color_num = str(data["data"]["color_num"])

            processes.append(executor.submit(post_ELK, url, tvid, address, px, py, normal, floods, unknow, saved_time_sec, im_name, color_num).result())

# aws.py: def lambda_handler(event, context):
# 2019-08-22 TWCC CAN NOT ACCESS $ping 140.110.27.197 not response!!!!!!請問TWCC開發型容器是否無法ping或連到中心140.110.27網段？目前需要從容器發送結果到27的ip上，目前看來是被擋住的。中心政策
'''
    1.現在中心vm已開放對外連線,替代140.110.27這台機器
    2.流程管理系統申請防火牆開啟
    3.從140.110.27 ssh tunnel到 container,這樣container就可以連到27機器
    '''
'''
    aws.py
    這個版本缺少C000 台北市交通局的parser  and C1 C2,addbylsw,currently the server side seems closed not works now.
    '''
def single_post_sql(event, context):
    start = time.time()
#    data = json.loads(event["body"])
    data=event
    cnx = mysql.connector.connect(
      user='wra_2019',
      password='wra_2019@NCHC',
      host='140.110.17.128'
        )
#    print(data)

    tvid = str(data["mseg"]["tvid"])
    normal = str(data["data"]["normal"])
    floods = str(data["data"]["floods"])
    unknow = str(data["data"]["unknow"])
    address = str(data["mseg"]["road"])
    color_num = str(data["data"]["color_num"])
    realtime = str(data["mseg"]["realtime"])
    im_name = str(data["mseg"]["im_name"])
    if(tvid[0:3] == "TNb"):
        cur = cnx.cursor()
        sql = "UPDATE `wra_2019`.`cctv_tainan_p` SET `normal` = '"+normal+"', `floods`='"+floods+"', `unknow`='"+unknow+"', `color_num`='"+color_num+"',   `realtime`='"+realtime+"', `im_name`='"+im_name+"' WHERE `id` = '"+tvid+"';\n"
        insertsql = "INSERT INTO `cctv_tainan_p_history`.`"+tvid+"` (`address`, `normal`, `floods`, `unknow`, `color_num`, `realtime`, `im_name`) VALUES ('"+address+"', '"+normal+"', '"+floods+"', '"+unknow+"', '"+color_num+"', '"+realtime+"', '"+im_name+"');\n"
        cur.execute(sql)
        timeP1 = time.time()
        cur.execute(insertsql)
        cnx.commit()
#        print(tvid+": 被修改")
        cur.close()
        cnx.close()
    elif(tvid[0:3] == "TNa"):
        cur = cnx.cursor()
        sql = "UPDATE `wra_2019`.`cctv_tainan_t` SET `normal` = '"+normal+"', `floods`='"+floods+"', `unknow`='"+unknow+"', `color_num`='"+color_num+"',   `realtime`='"+realtime+"', `im_name`='"+im_name+"' WHERE `id` = '"+tvid+"';\n"
        insertsql = "INSERT INTO `cctv_tainan_t_history`.`"+tvid+"` (`address`, `normal`, `floods`, `unknow`, `color_num`, `realtime`, `im_name`) VALUES ('"+address+"', '"+normal+"', '"+floods+"', '"+unknow+"', '"+color_num+"', '"+realtime+"', '"+im_name+"');\n"
        cur.execute(sql)
        cur.execute(insertsql)
        cnx.commit()
#        print(tvid+": 被修改")
        cur.close()
        cnx.close()
    elif(tvid[0:4] == "C000"):
        cur = cnx.cursor()
        sql = "UPDATE `wra_2019`.`cctv_taipei_t` SET `normal` = '"+normal+"', `floods`='"+floods+"', `unknow`='"+unknow+"', `color_num`='"+color_num+"',   `realtime`='"+realtime+"', `im_name`='"+im_name+"' WHERE `id` = '"+tvid+"';\n"
        insertsql = "INSERT INTO `cctv_taipei_t_history`.`"+tvid+"` (`address`, `normal`, `floods`, `unknow`, `color_num`, `realtime`, `im_name`) VALUES ('"+address+"', '"+normal+"', '"+floods+"', '"+unknow+"', '"+color_num+"', '"+realtime+"', '"+im_name+"');\n"
        cur.execute(sql)
        cur.execute(insertsql)
        cnx.commit()
        cnx.close()
#        print(tvid+": 被修改")
        cur.close()
    elif(tvid[0:2] == "TY"):
        cur = cnx.cursor()
        sql = "UPDATE `wra_2019`.`cctv_taoyuan` SET `normal` = '"+normal+"', `floods`='"+floods+"', `unknow`='"+unknow+"', `color_num`='"+color_num+"',   `realtime`='"+realtime+"', `im_name`='"+im_name+"' WHERE `id` = '"+tvid+"';\n"
        insertsql = "INSERT INTO `cctv_taoyuan_history`.`"+tvid+"` (`address`, `normal`, `floods`, `unknow`, `color_num`, `realtime`, `im_name`) VALUES ('"+address+"', '"+normal+"', '"+floods+"', '"+unknow+"', '"+color_num+"', '"+realtime+"', '"+im_name+"');\n"
        cur.execute(sql)
        cur.execute(insertsql)
        cnx.commit()
#        print(tvid+": 被修改")
        cur.close()
        cnx.close()
    elif(tvid[0:7] == "thbCCTV"):
        cur = cnx.cursor()
        sql = "UPDATE `wra_2019`.`cctv_gov` SET `normal` = '"+normal+"', `floods`='"+floods+"', `unknow`='"+unknow+"', `color_num`='"+color_num+"',   `realtime`='"+realtime+"', `im_name`='"+im_name+"' WHERE `id` = '"+tvid+"';\n"
        insertsql = "INSERT INTO `cctv_gov_history`.`"+tvid+"` (`address`, `normal`, `floods`, `unknow`, `color_num`, `realtime`, `im_name`) VALUES ('"+address+"', '"+normal+"', '"+floods+"', '"+unknow+"', '"+color_num+"', '"+realtime+"', '"+im_name+"');\n"
        cur.execute(sql)
        cur.execute(insertsql)
        cnx.commit()
#        print(tvid+": 被修改")
        cur.close()
        cnx.close()
    
    end = time.time()
#    print('\r[{}] SQL update time: {}'.format(tvid, end - start), end="")# not allow by Cython compile.
    print('[{}] SQL update time: {}'.format(tvid, end - start))

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }


def parall_excutor_po_sql(num_workers, JSON_TEMP, context):
    
    processes = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for jsn in JSON_TEMP:
            processes.append(executor.submit(single_post_sql, jsn, context))


# For lintbot_parall-dl-po, will turn off realimte post, then post it by paralle.
Para_if_ON = 'True'
def inf_im_json(sess, label_file, viz, im_name, df, imwritedir, saved_time_sec, JSON_TEMP, FOOD_TEMP, FLAG_TEMP, rainfall_data,
           input_operation, output_operation,
           input_height,
           input_width,
           input_mean,
           input_std):
    '''
        2019-08-22
        產出之json tvid:im_name有錯，tvid的TY, TN ,C00都沒有出現在json
        是tvname=TY, TN ,C00在找tvid時錯誤都回報成thbCCTV。因為index是取締第0行編號是五段重複的只會找前1400 thb範圍。
        '''
    
    # Magic help
    from MBTK.help import url_3, url_4, url_5, bad_link
    # Magic help
    import MBTK.Pnum

    # start
    start = time.time() # This ALL compute time include reading and convert the stream [2018-11-10@LSW]
    
    # Test from value to get index for parall image file name.
    tv_name =  im_name.split('_')[0]
#    print('Test tv_name: ', tv_name)
    im_idx = int(df[df['cctvid'].isin([tv_name])].index.to_native_types())# if df no index=0 then not use "-1" shift. #tolist()#full title and line
#        print('im_idx = {}'.format(im_idx))
#        print('{} road = {}'.format(tv_name, df.iloc[int(im_idx)]['roadsection']))
#        print('{} px == {}'.format(tv_name, round(df.iloc[int(im_idx)]['px'],5)))
#        print('{}\'s cctvid == {}'.format(tv_name, df.iloc[int(im_idx)]['cctvid']))
    
    # df layout of item's name
    #idc = df.iloc[id_count]['id']
    tvid = df.iloc[im_idx]['cctvid']
    url = df.iloc[im_idx]['url']
    road = df.iloc[im_idx]['roadsection']
    px = round(df.iloc[im_idx]['px'],5)
    py = round(df.iloc[im_idx]['py'],5)
    #print('start ID:{} {} ----------- {} ----------- {}'.format(id_count, tvid, road, url))
#    print('tvname=', tv_name)
#    print('tvid=  ', tvid)

    
    img_path = imwritedir + im_name
    #    print('img_path', img_path)
    if Para_if_ON is 'True':# Just lazy wouldn't change the block layout. This line should be remove.
#    if tvid not in bad_link:
#    if 1:
        #        print('GO to ', tvid)
        # load image by file
        t = read_tensor_from_image_file(img_path, # _from_video_stream OR file.
                                        input_height=input_height,
                                        input_width=input_width,
                                        input_mean=input_mean,
                                        input_std=input_std)

        ##LSW
        start_0 = time.time() # This only compute sess time not include reading and convert the image [2018-03-29@LSW]
        
        #        print('input_operation,output_operation', input_operation.outputs[0], output_operation.outputs[0])
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: t}) # not work with paralle!!!
        #        results = SESS_inf(t, input_operation, output_operation)
        
        end_0=time.time()
        
        results = np.squeeze(results)
        #print('\nEvaluation time (1-image): {:.3f}s FPS : {:.0f}\n'.format(end-start, 1/(end-start)))
        #end=time.time()
        #print('Inference time : {:.3f}s, FPS : {:.0f}, WallTime: {:.3f}s, FPS: {:.0f}'.format(end_0-start_0, 1/(end_0-start_0), end - start, 1/(end-start)))
        
        top_k = results.argsort()[-3:][::-1]#for top-1 #[-3:][::-1] for top-3
        labels = load_labels(label_file)
        
        """for i in top_k:
            #print('<',top_k[i],'>', labels[i], results[i])
            print('<{}> {} \t {:.3f}'.format(top_k[i], labels[i], results[i]))"""
        
        # PR Counting for --hit with 'all' or each label
        #print('fixed labels', labels, 'top k', top_k, 'top 1', labels[top_k[0]])
        #            if args.hit == 'all':
        #                hit_count = hit_count + 1
        #                #print('[Label]', 'all', "[hits]", "counting", "[progress]" , now_count, "/", imgNum)
        #                PRCounter_CROSS(labels[top_k[0]])
        
        #print('floods', 'normal', 'unknow')#'normal', 'raining','unclear')#, 'unclear') ['normal', 'rain', 'unclear']
        #print("{0}      {1}      {2}      {3}      {4}      {5} ".format(P1,P2,P3,P4,P5,P6))
        #print(P1,'\t', P2, '\t', P3)#,'\t',P3)
        
        
        ## LSW show image by CV2 and plot
        frame = cv2.imread(img_path)
        img = frame.copy() # dump a new image for plot keep frame org to saved.
        #height, width, channels = frame.shape # for ploting, should be remove to save time!
        
        blue=(255, 128, 0)#OpenCV, BGR
        white=(255, 255, 255)
        pink=(255, 100, 255)
        gray=(125,125,125)
        no_conf_len = 0
        ra_conf_len = 0
        un_conf_len = 0
        text_normal = 'normal' #'非實害' #CV2不支援中文putText(), but python ok for it.
        text_raining = 'floods' #'    實害'
        text_unclear = 'unknow'
        bar_ln = 100
        
        
        for i in top_k:
            #print('<',top_k[i],'>', labels[i], results[i])
            #print('<{}> {} \t {:.3f}'.format(top_k[i], labels[i], results[i]))
            if labels[i] == 'normal':
                no_conf_len = int(results[i] * bar_ln)
            if labels[i] == 'floods':
                ra_conf_len = int(results[i] * bar_ln)
            if labels[i] == 'unknow':
                un_conf_len = int(results[i] * bar_ln)
    
        # Only show positive list on visdom, bypass bad_link.
        if tvid not in bad_link: # some thbCCTV-23-1440-001-02 in bad_link still shown in page.
            #text_normal
            cv2.rectangle(img, (10, 10), (10+no_conf_len, 30), blue, -1)
            cv2.putText(img, text_normal,#文字
                        (10, 25),#左下角座標點,
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        white,
                        1,
                        cv2.LINE_AA
                        )
            #text_raining
            cv2.rectangle(img, (10, 10+40), (10+ra_conf_len, 30+40), pink, -1)
            cv2.putText(img, text_raining,#文字
                        (10, 25+40),#左下角座標點,
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        white,
                        1,
                        cv2.LINE_AA
                        )
                        
            #text_unclear
            cv2.rectangle(img, (10, 10+80), (10+un_conf_len, 30+80), gray, -1)
            cv2.putText(img, text_unclear,#文字
                        (10, 25+80),#左下角座標點,
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        white,
                        1,
                        cv2.LINE_AA
                        )
                        
            vis_img = img.copy()
            vis_img = vis_img[:,:,::-1]
            vis_img = vis_img.transpose(2,0,1)
            '''
                visdom:vis.image() by (3,w,h), vis.images() by (num,3,w,h), RGB
                cv2 imaeg is BGR with (w,h,3)
                so, the img.transpose is needed to fitting the visdom.image().
                '''
            viz.image(
                      vis_img, win=url,
                      opts=dict(title=str(im_idx) + ": " + str(tvid) + ',' + labels[top_k[0]], caption='THB ccTV stream',
                                width=320, height=240
                                )
                      )#width=320, height=240) resize the figure=img not the window!!! And the Win still need handly resize.
            
        # docker time to local time
#        epoch_add_8hrs =time.time()# + 28800
#        realtime = time.strftime("%Y-%m-%d:%H:%M:%S", time.localtime(epoch_add_8hrs))
#        realtime = time.strftime("%Y-%m-%d:%H:%M:%S", time.localtime(epoch_add_8hrs))
#
#        #we do not it in parallel download mode#
#        # LSW 2019-06-10, imwrite for backup.
#        im_name = tvid + '_' + time.strftime("%Y-%m-%d-%H-%M", time.localtime(epoch_add_8hrs)) + '.jpg'
#        im_full_name = imwritedir + "/" + im_name
#        #print('im_name', im_name)
#        cv2.imwrite(im_full_name, frame)
        
        # LSW 2019-06-10, set colors, normal flood unknow= 1 2 3, no stream = 0.
        # level: 1 一般(綠燈), 2 淹水(紅燈), 3 unknow(灰) 4.無資料(白)
        #labels[top_k[0]]
        color_num = 0
        if labels[top_k[0]] == 'normal':
            color_num = 1
        if labels[top_k[0]] == 'floods':
            color_num = 2
        if labels[top_k[0]] == 'unknow':
            color_num = 3
                                                
        # to WRA AI map
        #req_post_joson_tvid(url_3, tvid, no_conf_len, ra_conf_len, un_conf_len, saved_time_sec)# realtime -> saved_time_sec

        #print('End ID:{} {} ----------- {} -----------'.format(id_count, tvid, road))
        #print(realtime)

        # to ELK server, if flood is detected.
        #if labels[top_k[0]] == 'floods':
        #post_ELK(url_4, tvid, road, px, py, no_conf_len, ra_conf_len, un_conf_len, saved_time_sec, im_name, color_num)# realtime -> saved_time_sec
        #https://roamkgx409.execute-api.us-east-2.amazonaws.com/default/wra_2019_post
        # to Line Bot

        # FOOD_TEMP
        if tvid not in bad_link and labels[top_k[0]] == 'floods':
            cmd_line = '\'' + tvid + '\' \'' + road + '\' \'' + saved_time_sec + '\' \'' + im_name + '\' \'' + img_path + '\''
            FOOD_TEMP.append(cmd_line)

        # FLAG_TEMP
#        if 1:
        if tvid not in bad_link and labels[top_k[0]] == 'floods' and ifRain(rainfall_data, py, px, MBTK.Pnum.rain_hh, MBTK.Pnum.rain_mh):# and ra_conf_len > 99:
            """
            #labels[top_k[0]] == 'floods' and tvid not in bad_link:# and ra_conf_len >= 80:
            lon->px lat->py
            """
            
            print('IS Re and > V, Send to linebot -- >',  labels[top_k[0]], ra_conf_len, tvid, img_path)
#            cmd_line = 'post_Linebot(\'' + tvid + '\',\'' + road + '\',\'' + saved_time_sec + '\',\'' + im_name + '\',\'' + img_path + '\')'
#            cmd_line = '\'' + tvid + '\' \'' + road + '\' \'' + saved_time_sec + '\' \'' + im_name + '\' \'' + img_path + '\''
#            FOOD_TEMP.append(cmd_line)
            
            mesg = { "tvid":tvid, "road":road, "px":px, "py":py,"realtime":saved_time_sec, "im_name":im_name, "url":url}# realtime -> saved_time_sec
            data = { "index":MBTK.Pnum.flag_index, "normal":no_conf_len, "floods":ra_conf_len, "unknow":un_conf_len, "color_num":color_num}
            jj = {"mseg":mesg, "data":data}
            FLAG_TEMP.append(jj)

            MBTK.Pnum.flag_index +=1 #flag_add_one() no long used.
            
#            post_Linebot(tvid, road, saved_time_sec, im_name, img_path) #for parallel download.
            #post_Linebot(tvid, road, realtime, im_name, im_full_name)
                
        # Timer
        end=time.time()
#        print('\rInference time : {:.3f}s, FPS : {:.0f}, WallTime: {:.3f}s, FPS: {:.0f}'.format(end_0-start_0, 1/(end_0-start_0), end - start, 1/(end-start)), end="")
        print('Inference {} : {:.3f}s, FPS : {:.0f}, WallTime: {:.3f}s, FPS: {:.0f}'.format(tvid, end_0-start_0, 1/(end_0-start_0), end - start, 1/(end-start)))

        #mlGreen 2 post about FPS 4~6, if turn off 2 post about FPS 16~17, only req_post fps 5~6, only post_ELK fps ~15
#        print('')
        # 2019-08-16, post may take 3~5 sec. each time. Let us post all json at onece.
        if color_num is 3:# Re-filtering the unknown let it shown as normal(green).
            color_num = 1
        if tvid in bad_link:
            color_num = 1
        if not ifRain(rainfall_data, py, px, MBTK.Pnum.rain_hh, MBTK.Pnum.rain_mh):
            color_num = 1
        
        mesg = { "tvid":tvid, "road":road, "px":px, "py":py,"realtime":saved_time_sec, "im_name":im_name}# realtime -> saved_time_sec
        data = { "normal":no_conf_len, "floods":ra_conf_len, "unknow":un_conf_len, "color_num":color_num}
        jj = {"mseg":mesg, "data":data}
        JSON_TEMP.append(jj)






#***
#*** isDownpour
#***

"""
    isDownpour
    2019-10-05
    
    if __name__ == '__main__':
        lat = []
        lon = []
        f = open('coordinate.txt', 'r')
        for line in f.readlines():
            temp = line.strip().split(',')
            lon.append(temp[0])
            lat.append(temp[1])
       
        hh = 10
        mh = 5
        
        #下載雨量觀測資料
    #    get_rainfall_data()
        #畫出雨量圖
    #    mp = drawmap()
        
        st = time.time()
        #讀取雨量觀測資料
        data = read_rainfall_data()
        for i in range(len(lat)):
            #判斷雨量是否達標
            ifrain = ifRain(data,lat[i],lon[i],hh,mh)
            print("{}: {}".format(i, ifrain))
            #畫出指定座標點
    #        mp.plot(lon[i], lat[i], 'ok', markersize=5, color = 'black')
             
    #    plt.show()
        print('執行時間: ' + str(time.time() - st))
"""

#下載雨量觀測資料
def get_rainfall_data():
    attempts = 0
    success = False
    while attempts < 10 and not success:
        try:
            Authorization = 'CWB-9D1156C6-D4EC-491E-8629-2D73392C2D2D'
            elementName = 'RAIN,MIN_10'
            parameterName = 'CITY,CITY_SN,TOWN,TOWN_SN'
            r = requests.get("https://opendata.cwb.gov.tw/api/v1/rest/datastore/O-A0002-001?"
                             +"Authorization="+Authorization
                             +"&format=JSON&elementName="+elementName
                             +"&parameterName="+parameterName
                             )
            f = open('rainfall.json', 'w')
            f.write(r.text)
            f.close()
            success = True
        except:
            print('資料獲取失敗重新連線...')
            time.sleep(2)
            attempts += 1
            if attempts == 10:
                print('重新連線失敗...')
                break

#讀取雨量觀測資料
def read_rainfall_data():
    with open('rainfall.json', 'r') as f:
        data = json.load(f)
    list_of_dicts = data
    
    data = []
    for item in list_of_dicts["records"]["location"]:
        lat = float(item['lat'])
        lon = float(item['lon'])
        locationName = item['locationName']
        obsTime = item['time']['obsTime']
        hour_rain = 0
        min_10_rain = 0
              
        for rain_item  in item['weatherElement']:
            if rain_item['elementName'] == 'RAIN':
                hour_rain = float(rain_item['elementValue'])
            if rain_item['elementName'] == 'MIN_10':
                min_10_rain = float(rain_item['elementValue'])
                           
        if (hour_rain < 0) : hour_rain = 0.0
        if (min_10_rain < 0) : min_10_rain = 0.0
            
        data.append([lat,lon,hour_rain,min_10_rain,locationName,obsTime])
             
    return data

def drawmap():
    
    # 繪圖的經緯度範圍
    PLOTLATLON = [119.00, 122.50, 21.50, 25.50]
    # 插值時設置的格點數，需要依據實際情況動態調整
    PLOT_Interval = 1000j
    
    data = np.array(read_rainfall_data())
    
    lon = np.array(data[:,1], dtype='f')
    lat = np.array(data[:,0], dtype='f')
    rain = np.array(data[:,2], dtype='f')

    grid_x, grid_y = np.mgrid[PLOTLATLON[0]:PLOTLATLON[1]:PLOT_Interval, PLOTLATLON[2]:PLOTLATLON[3]:PLOT_Interval]
    # 插值方法：'nearest', 'linear', 'cubic'
    grid_z = griddata((lon, lat), rain, (grid_x, grid_y), method='nearest')
    
    # 畫圖
    fig = plt.figure(figsize=(16, 9))
    plt.rc('font', size=15, weight='bold')
        
    nws_precip_colors = [
    "#ccccb3",  #gray
    "#99ffff",  #blue
    "#66b3ff",
    "#0073e6",
    "#002699",
    "#009900",  #green
    "#1aff1a",
    "#ffff00",  #yellow
    "#ffcc00",
    "#ff9900",
    "#ff0000",  #rea
    "#cc0000",
    "#990000",
    "#800040",  #purple
    "#b30047",
    "#ff0066",
    "#ff80b3",
    "#FFB7DD"]

    precip_colormap = matplotlib.colors.ListedColormap(nws_precip_colors)
 
    # 創建底圖,等經緯度投影
    mp = Basemap(llcrnrlon=PLOTLATLON[0],llcrnrlat=PLOTLATLON[2],urcrnrlon=PLOTLATLON[1],
                 urcrnrlat=PLOTLATLON[3],projection='cyl',resolution='h',epsg=4326)
    
#    mp.drawcoastlines()
    mp.readshapefile('gadm36_TWN_shp/gadm36_TWN_1', name='Taiwan', linewidth=0.25, drawbounds=True)
    mp.readshapefile('gadm36_TWN_shp/gadm36_TWN_2', name='Taiwan', linewidth=0.25, drawbounds=True)
 

    levels = [0.01,1,2,6,10,15,20,30,40,50,70,90,110,130,150,200,300,400]
    norm = matplotlib.colors.BoundaryNorm(levels,len(levels))
    
    mask_rain  = maskoceans(grid_x, grid_y, grid_z, inlands=False, resolution='h')
    mp.contourf(grid_x, grid_y, mask_rain, levels=levels, cmap=precip_colormap, norm=norm , linestyles=None,alpha=0.9)
    mp.colorbar(ticks=levels[1:-1], label='rain (mm)')
    
    print('DataTime:'+data[0][-1])
    return mp

def haversine(lat1, lon1, lat2, lon2): # 緯度1，經度1，緯度2，經度2 （十進制度數
    # 將十進制度數轉化為弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 # 地球平均半徑，單位為公里
    return c * r * 1000

def ifRain(data,lat,lon,hh,mh):
    min_dist = float("inf")
    min_dist_index = None
    for i in range(len(data)):
        dist = haversine(float(lat),float(lon),data[i][0],data[i][1])
        if min_dist > dist:
            min_dist = dist
            min_dist_index = i
    
    if data[min_dist_index][2] >= hh or data[min_dist_index][3] >= mh:
        return True
    else:
        return False


#***
#*** isDownpour
#***



#***
#*** Reportor
#***
from PIL import Image,ImageDraw,ImageFont
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import json
import math
import time

black = (0,0,0)
white = (255,255,255)
freemonottf = 'MBTK/FreeMono.ttf'
wgymicrohei = '/usr/share/fonts/truetype/wgy/wqy-microhei.ttc'
class Reporter():
    def __init__(self, down_or_right, data):
        self.img_width = 320
        self.img_height = 240
        self.report_width = 0
        self.repot_height = 1
        self.url='http://140.110.17.128/aimap/stoca/color_lamp/nowred.aspx'#gis map
#        self.url='http://fmg.wra.gov.tw:8080/fmg/floodedAlarmList.php'#fmg web
        self.display = down_or_right # 0 => 小圖置下 ; 1 ＝> 小圖置右
        self.data = data
        print("init__", len(data))

    def get_Web(self,url,name,data):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
#        options.add_argument('-disable-javascript')
#        options.add_argument('blink-settings=imagesEnabled=false') # not load image

        options.add_argument("--window-size=800,960")#(W,H) "--window-size=800,960" for gis map,
#        options.add_argument("--window-size=1280x2560")#(W,H)  "--window-size=1200,3460" for fmg web.
#        options.add_argument("--start-maximized") #fmg web seems defult as phone size if use maximized.

        browser=webdriver.Chrome(chrome_options=options, executable_path='MBTK/chromedriver_64')
        browser.get(url)
        
        time.sleep(1)
#        img_path = './app/static/reports/'+name+'.png'
#        report_name = os.path.splitext(data[0]['mseg']['im_name'].split('_')[-1])[0]
#        img_path = '/tmp/' + report_name + '/' +name+'.png'
        img_path = '/tmp/' + name +'_web.png'
        browser.save_screenshot(img_path)

        browser.quit()
        return_img = Image.open(img_path)
        print("web img:", img_path)
        

        return return_img

    def draw_idx_circle(self,id,image):
        x=30
        y=30
        r=30
        draw = ImageDraw.Draw(image)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0,0))
        
        # You may need to change the path of fonts if you don't have the fonts
        if id<10:
            fnt = ImageFont.truetype(freemonottf, 70)
            draw.text((x-3*r/4,y-6*r/5), str(id), font=fnt, fill=black,align="left",)
        elif 9<id and id<100:
            fnt = ImageFont.truetype(freemonottf, 45)
            draw.text((x-4*r/5,y-4*r/5), str(id), font=fnt, fill=black,align="left",)
        else:
            fnt = ImageFont.truetype(freemonottf, 20)
            draw.text((x-r,y-4*r/5), str(id), font=fnt, fill=black,align="left",)
        
        return image

#    def text_flag_on_web_img(self, web_img, data):
#        img_path = os.path.join('/tmp', report_name, v['mseg']['im_name'])
#        img = Image.open(img_path)
    

    def make_report(self, imwritedir):
#        report_name = os.path.splitext(self.data[0]['mseg']['im_name'].split('_')[-1])[0]
        report_name = imwritedir
        print('report_name:\n', report_name)
        # screenshot of web
        web_img = self.get_Web(self.url, report_name, self.data)

        # text FLAG on web image [J-2019+10+09ß]
        draw = ImageDraw.Draw(web_img)
        fnt = ImageFont.truetype(wgymicrohei, 25,encoding="unic")# wqy-microhei.ttc wqy-zenhei.ttc
        draw.text((20, 150), str('[' +self.data[0]['mseg']['realtime'] + ']'), font=fnt, fill=(    0, 0, 0), lang='zh-TW',align="left")
        for k,v in enumerate(self.data):
#            draw.ellipse((10,150 + (30*k), 30, 150 + (k+1)*30), fill=(255,0,0))
#            draw.arc((10,150 + (25*k), 30, 150 + (k+1)*25),0,360,fill=(255,0,0,0))
#            draw.ellipse((10, 150 - (25*k), 20, 150 + (25*(k+1))), fill=(255,0,0,0))
            flag_text_full = str(self.data[k]['data']['index']) + '. ' + self.data[k]['mseg']['road'] #full name
#            flag_text = str(self.data[k]['data']['index']) + '. ' + self.data[k]['mseg']['road'].split('/')[0] #cut name
            if "/" in flag_text_full:
                flag_text = str(self.data[k]['data']['index']) + '. ' + self.data[k]['mseg']['road'].split('/')[1] #cut name
            else:
                flag_text =flag_text_full
            draw.text((10, 180 + (30*k) ), str(flag_text), font=fnt, fill=(0,0,0), lang='zh-TW',align="left")

        x_offset = 0
        y_offset = 0
        # setting the width & height of the report
        if self.display == 0:
            self.report_width = self.img_width * 3
            web_img = web_img.resize((self.report_width, int(self.report_width/web_img.size[0] * web_img.size[1])),Image.ANTIALIAS)
            self.report_height = math.ceil(len(self.data)/3) * self.img_height + web_img.size[1]
        else:
            self.report_height = self.img_height * 4
            self.report_width = math.ceil(len(self.data)/4)* self.img_width + web_img.size[0]

        new_im = Image.new('RGB', (self.report_width, self.report_height))
        
        #draw the small figures
        for k,v in enumerate(self.data):
            img_path = os.path.join('/tmp', report_name, v['mseg']['im_name'])
            print('report img_path: \n', img_path)
            img = Image.open(img_path)
            img = img.resize((self.img_width,self.img_height),Image.ANTIALIAS)
            igm = self.draw_idx_circle(v['data']['index'],img)
            new_im.paste(img, (x_offset, y_offset))
            if self.display==0:
                x_offset += self.img_width
                if (k+1) % 3 == 0:
                    x_offset=0
                    y_offset+=self.img_height
            else:
                y_offset += self.img_height
                if (k+1) % 4 == 0:
                    y_offset=0
                    x_offset+=self.img_width

       

        report_im = Image.new('RGB', (self.report_width, self.report_height))

        report_im.paste(web_img, (0, 0))
        if self.display == 0 :
            report_im.paste(new_im, (0, web_img.size[1]))
        else:
            report_im.paste(new_im, (web_img.size[0],0))

        report_path = os.path.join('/tmp',report_name+'_confmapflag.jpg')
        report_im.save(report_path)
        
        print("rpo image: ", report_path)
        
        return report_path
        

def send_report_to_linebot(imwritedir):
    list_of_flags = glob.glob('/tmp/*-FLAG.json')
    if list_of_flags!=[]:
        latest_flag = max(list_of_flags, key=os.path.getctime)
    if os.stat(latest_flag).st_size > 3:
        with open(latest_flag, 'r') as f:
            data = json.load(f)
            print('data len:', len(data))
            RP = Reporter(1, data)# 0 => 小圖置下 ; 1 ＝> 小圖置右
            #report_path = RP.make_report(data)
            print(RP.make_report(imwritedir))
            #send FLAGE picture to linebot
#            basename = data[0]['mseg']['im_name'].split('_')[1].split('.')[0]
#            print('basename', basename)
            basename = imwritedir
            print('basename', basename)
            im_name = basename +'_confmapflag.jpg'
            path_name = '/tmp/' + im_name
            print('path_name', path_name)
            post_FLAG_Linebot(data[0]['mseg']['realtime'], im_name, path_name)
    else:
        print("\n *-FLAG.json is empty! \n")
#***
#*** Reportor
#***


# Check http response
#        for task in as_completed(processes):
#            print(task.result())


    




#
#
#
#
#
#
#
#
#
# for back up....
#def inf_im(sess, label_file, viz, im_name, df, imwritedir, saved_time_sec, JSON_TEMP,
#           input_operation, output_operation,
#           input_height,
#           input_width,
#           input_mean,
#           input_std):
#    # Magic help
#    from MBTK.help import url_3, url_4, bad_link
#
#    # start
#    start = time.time() # This ALL compute time include reading and convert the stream [2018-11-10@LSW]
#
#    # Test from value to get index for papall image file name.
#    tv_name =  im_name.split('_')[0]
#    print('tvname=', tv_name)
#    print('im_name',im_name)
#    im_idx = int(df[df['cctvid'].isin([tv_name])].index.to_native_types()) -1 #tolist()#full title and line
#    #    print('im_idx = {}'.format(im_idx))
#    #    print('{} road = {}'.format(tv_name, df.iloc[int(im_idx)]['roadsection']))
#    #    print('{} px == {}'.format(tv_name, round(df.iloc[int(im_idx)]['px'],5)))
#    #    print('{}\'s cctvid == {}'.format(tv_name, df.iloc[int(im_idx)]['cctvid']))
#
#    # df layout of item's name
#    #idc = df.iloc[id_count]['id']
#    tvid = df.iloc[im_idx]['cctvid']
#    url = df.iloc[im_idx]['url']
#    road = df.iloc[im_idx]['roadsection']
#    px = round(df.iloc[im_idx]['px'],5)
#    py = round(df.iloc[im_idx]['py'],5)
#    #print('start ID:{} {} ----------- {} ----------- {}'.format(id_count, tvid, road, url))
#
#    img_path = imwritedir + im_name
#    #    print('img_path', img_path)
#    #if Para_ON is 'True':# Just lazy wouldn't change the block layout. This line should be remove.
#    if tvid not in bad_link:
#        #        print('GO to ', tvid)
#        # load image by file
#        t = read_tensor_from_image_file(img_path, # _from_video_stream OR file.
#                                        input_height=input_height,
#                                        input_width=input_width,
#                                        input_mean=input_mean,
#                                        input_std=input_std)
#                                        print('Check after [read_tensor_from_image_file]')
#                                        ##LSW
#                                        start_0 = time.time() # This only compute sess time not include reading and convert the image [2018-03-29@LSW]
#
#
#                                        print('Check [inf_im] start of results = sess.run')
#                                        #        print('input_operation,output_operation', input_operation.outputs[0], output_operation.outputs[0])
#                                        results = sess.run(output_operation.outputs[0],
#                                                           {input_operation.outputs[0]: t}) # not work with paralle!!!
#                                        #        results = SESS_inf(t, input_operation, output_operation)
#                                        print('Check [inf_im] end of results = sess.run')
#
#
#                                        end_0=time.time()
#
#                                        results = np.squeeze(results)
#                                        #print('\nEvaluation time (1-image): {:.3f}s FPS : {:.0f}\n'.format(end-start, 1/(end-start)))
#                                        #end=time.time()
#                                        #print('Inference time : {:.3f}s, FPS : {:.0f}, WallTime: {:.3f}s, FPS: {:.0f}'.format(end_0-start_0, 1/(end_0-start_0), end - start, 1/(end-start)))
#
#                                        top_k = results.argsort()[-3:][::-1]#for top-1 #[-3:][::-1] for top-3
#                                        labels = load_labels(label_file)
#
#                                        """for i in top_k:
#                                            #print('<',top_k[i],'>', labels[i], results[i])
#                                            print('<{}> {} \t {:.3f}'.format(top_k[i], labels[i], results[i]))"""
#
#                                        # PR Counting for --hit with 'all' or each label
#                                        #print('fixed labels', labels, 'top k', top_k, 'top 1', labels[top_k[0]])
#                                        #            if args.hit == 'all':
#                                        #                hit_count = hit_count + 1
#                                        #                #print('[Label]', 'all', "[hits]", "counting", "[progress]" , now_count, "/", imgNum)
#                                        #                PRCounter_CROSS(labels[top_k[0]])
#
#                                        #print('floods', 'normal', 'unknow')#'normal', 'raining','unclear')#, 'unclear') ['normal', 'rain', 'unclear']
#                                        #print("{0}      {1}      {2}      {3}      {4}      {5} ".format(P1,P2,P3,P4,P5,P6))
#                                        #print(P1,'\t', P2, '\t', P3)#,'\t',P3)
#
#
#                                        ## LSW show image by CV2 and plot
#                                        frame = cv2.imread(img_path)
#                                        img = frame.copy() # dump a new image for plot keep frame org to saved.
#                                        #height, width, channels = frame.shape # for ploting, should be remove to save time!
#
#                                        blue=(255, 128, 0)#OpenCV, BGR
#                                        white=(255, 255, 255)
#                                        pink=(255, 100, 255)
#                                        gray=(125,125,125)
#                                        no_conf_len = 0
#                                        ra_conf_len = 0
#                                        un_conf_len = 0
#                                        text_normal = 'normal' #'非實害' #CV2不支援中文putText(), but python ok for it.
#                                        text_raining = 'floods' #'    實害'
#                                        text_unclear = 'unknow'
#                                        bar_ln = 100
#
#
#                                        for i in top_k:
#                                            #print('<',top_k[i],'>', labels[i], results[i])
#                                            #print('<{}> {} \t {:.3f}'.format(top_k[i], labels[i], results[i]))
#                                            if labels[i] == 'normal':
#                                                no_conf_len = int(results[i] * bar_ln)
#                                                    if labels[i] == 'floods':
#                                                        ra_conf_len = int(results[i] * bar_ln)
#                                                            if labels[i] == 'unknow':
#                                                                un_conf_len = int(results[i] * bar_ln)
#
#
#                                                        #text_normal
#                                                        cv2.rectangle(img, (10, 10), (10+no_conf_len, 30), blue, -1)
#                                                                cv2.putText(img, text_normal,#文字
#                                                                            (10, 25),#左下角座標點,
#                                                                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                                                                            1,
#                                                                            white,
#                                                                            1,
#                                                                            cv2.LINE_AA
#                                                                            )
#                                                                    #text_raining
#                                                                    cv2.rectangle(img, (10, 10+40), (10+ra_conf_len, 30+40), pink, -1)
#                                                                    cv2.putText(img, text_raining,#文字
#                                                                                (10, 25+40),#左下角座標點,
#                                                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                                                                                1,
#                                                                                white,
#                                                                                1,
#                                                                                cv2.LINE_AA
#                                                                                )
#
#                                                                                #text_unclear
#                                                                                cv2.rectangle(img, (10, 10+80), (10+un_conf_len, 30+80), gray, -1)
#                                                                                cv2.putText(img, text_unclear,#文字
#                                                                                            (10, 25+80),#左下角座標點,
#                                                                                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                                                                                            1,
#                                                                                            white,
#                                                                                            1,
#                                                                                            cv2.LINE_AA
#                                                                                            )
#
#                                                                                            vis_img = img.copy()
#                                                                                            vis_img = vis_img[:,:,::-1]
#                                                                                            vis_img = vis_img.transpose(2,0,1)
#                                                                                            '''
#                                                                                                visdom:vis.image() by (3,w,h), vis.images() by (num,3,w,h), RGB
#                                                                                                cv2 imaeg is BGR with (w,h,3)
#                                                                                                so, the img.transpose is needed to fitting the visdom.image().
#                                                                                                '''
#                                                                                                    viz.image(
#                                                                                                              vis_img, win=url,
#                                                                                                              opts=dict(title=str(im_idx) + ": " + str(tvid) + ',' + labels[top_k[0]], caption='THB ccTV stream',
#                                                                                                                        width=320, height=240
#                                                                                                                        )
#                                                                                                              )#width=320, height=240) resize the figure=img not the window!!! And the Win still need handly resize.
#
#                                                                                                        # docker time to local time
#                                                                                                        #        epoch_add_8hrs =time.time()# + 28800
#                                                                                                        #        realtime = time.strftime("%Y-%m-%d:%H:%M:%S", time.localtime(epoch_add_8hrs))
#                                                                                                        #        realtime = time.strftime("%Y-%m-%d:%H:%M:%S", time.localtime(epoch_add_8hrs))
#                                                                                                        #
#                                                                                                        #        #we do not it in parallel download mode#
#                                                                                                        #        # LSW 2019-06-10, imwrite for backup.
#                                                                                                        #        im_name = tvid + '_' + time.strftime("%Y-%m-%d-%H-%M", time.localtime(epoch_add_8hrs)) + '.jpg'
#                                                                                                        #        im_full_name = imwritedir + "/" + im_name
#                                                                                                        #        #print('im_name', im_name)
#                                                                                                        #        cv2.imwrite(im_full_name, frame)
#
#                                                                                                        # LSW 2019-06-10, set colors, normal flood unknow= 1 2 3, no stream = 0.
#                                                                                                        # level: 1 一般(綠燈), 2 淹水(紅燈), 3 unknow(灰) 4.無資料(白)
#                                                                                                        #labels[top_k[0]]
#                                                                                                        color_num = 0
#                                                                                                            if labels[top_k[0]] == 'normal':
#                                                                                                                color_num = 1
#                                                                                                                    if labels[top_k[0]] == 'floods':
#                                                                                                                        color_num = 2
#                                                                                                                            if labels[top_k[0]] == 'unknow':
#                                                                                                                                color_num = 3
#
#                                                                                                                                    # to WRA AI map
#                                                                                                                                    req_post_joson_tvid(url_3, tvid, no_conf_len, ra_conf_len, un_conf_len, saved_time_sec)# realtime -> saved_time_sec
#
#                                                                                                                                    #print('End ID:{} {} ----------- {} -----------'.format(id_count, tvid, road))
#                                                                                                                                    #print(realtime)
#
#                                                                                                                                    # to ELK server, if flood is detected.
#                                                                                                                                    #if labels[top_k[0]] == 'floods':
#                                                                                                                                    #post_ELK(url, tvid, road, px, py, no_conf_len, ra_conf_len, un_conf_len, realtime, im_name, color_num)
#                                                                                                                                    #        post_ELK(url_4, tvid, road, px, py, no_conf_len, ra_conf_len, un_conf_len, saved_time_sec, im_name, color_num)# realtime -> saved_time_sec
#                                                                                                                                    #https://roamkgx409.execute-api.us-east-2.amazonaws.com/default/wra_2019_post
#                                                                                                                                    # to Line Bot
#                                                                                                                                    if labels[top_k[0]] == 'floods' and ra_conf_len >= 80:
#                                                                                                                                        print('IS Re and > 80, Send to linebot -- >',  labels[top_k[0]], ra_conf_len, tvid, img_path)
#                                                                                                                                            #post_Linebot(tvid, road, realtime, im_name, img_path) #for parallel download.
#                                                                                                                                            #post_Linebot(tvid, road, realtime, im_name, im_full_name)
#
#                                                                                                                                            # Timer
#                                                                                                                                            end=time.time()
#                                                                                                                                            print('Inference time : {:.3f}s, FPS : {:.0f}, WallTime: {:.3f}s, FPS: {:.0f}'.format(end_0-start_0, 1/(end_0-start_0), end - start, 1/(end-start)))
#                                                                                                                                            #mlGreen 2 post about FPS 4~6, if turn off 2 post about FPS 16~17, only req_post fps 5~6, only post_ELK fps ~15
#                                                                                                                                            print('')
#                                                                                                                                            # 2019-08-16, post may take 3~5 sec. each time. Let us post all json at onece.
#                                                                                                                                            mesg = { "tvid":tvid, "road":road, "px":px, "py":py,"realtime":saved_time_sec, "im_name":im_name}# realtime -> saved_time_sec
#                                                                                                                                                data = { "normal":no_conf_len, "floods":ra_conf_len, "unknow":un_conf_len, "color_num":color_num}
#                                                                                                                                                    jj = {"mseg":mesg, "data":data}
#                                                                                                                                                        JSON_TEMP.append(jj)
#
#                                                                                                                                                            print('Check after [inf_im]')







