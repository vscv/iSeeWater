

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# 2018-05-12
#
# Add PRCounter_ for HuWeiSP.
# sorce from: label_image_listdir_sumika_PRcounting_CROSS_LSW.py
# ==============================================================================
# 2018-05-12
#
# Add PRCounter_ for HuWeiSP.
# sorce from: label_image_listdir_sumika_PRcounting_CROSS_LSW.py
# ==============================================================================
# 2018-10-19
#
# TODO:
# Rewrite to lib MBTK API
# with Cython cimpiled shared library
# ==============================================================================
# 2019-08-14
#
# Rewrite to lib MBTK API
# with Cython cimpiled shared library
# 1. Add Parallel download mode.
# 2. TODO: also try to para-inference block. (this will not work on tf133 cause
#          the Session)
# ==============================================================================


# Basic modules
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function
import os
import sys
import cv2
import time
import json
import argparse
import datetime
import numpy as np
import tensorflow as tf

# MBTK modules
from MBTK.mbtk import *

# Magic help
from MBTK.help import *

# Magic number
from MBTK.Pnum import *

# Run py
'''
    Usage: Realtime service
    $time python -m scripts.label_image_MBNet_Images_DIRLoop_GraphDefault_PRcounting_HuWeiSP_LSW_slit_show_WRARoad_stream_multiCam_mbtk_linebot_parall-dl-po --graph=tf_files/rt_thb_im_224_mf_1.0_steps_8000_20190524fix702v3.pb  --labels=tf_files/rt_thb_im_224_mf_1.0_steps_8000_20190524fix702v3.txt --hit=all --imwritedir=mbimg_2019-08-13_1450_linebotP 2>&1 | tee  mbinf_2019-08-13_1450_linebotP.log
    
    Usage: Offline testing
    $ time python -m scripts.label_image_MBNet_Images_DIRLoop_GraphDefault_PRcounting_HuWeiSP_LSW_slit_show_WRARoad_stream_multiCam_mbtk_linebot_parall-dl-po --graph=tf_files/rt_thb_im_224_mf_1.0_steps_8000_20190524fix702v3.pb  --labels=tf_files/rt_thb_im_224_mf_1.0_steps_8000_20190524fix702v3.txt --hit=all --imwritedir=19700101-1200
    --imwritedir=19700101-1200: it's pre-download images, located at /tmp/.
    
'''

# vidsom module
"""
    x@xlab:~/x_temp_test$ visdom -port 8890 --hostname 140.110.xxx.xxx
    TWCC@J:~/x_temp_test$ visdom -port 5000 --hostname 203.145.xxx.xxx
    """
# Open preview page
from visdom import Visdom
##viz = Visdom()#port=FLAGS.port, server=FLAGS.server) # for GreenML
##viz = Visdom(port=8890)#port=FLAGS.port, server=FLAGS.server) # for mllab
viz = Visdom(port=5000)#port=FLAGS.port, server=FLAGS.server) # for TWCC




    
# How many parallel threads
num_workers = 256

# Trigger setting
Para_dl_ON = 'True'
Para_if_ON = 'True'
Para_po_ON = 'True'
Single_po_ON = 'True'
#DB_po_ON = 'True'

# Day time hours
day_h = ['6','7','8','9','10','11','12','13','14','15','16','17','18']


# OK GO
if __name__ == "__main__":
    
    # Print NCHC flag
    #show_module_flag()
    
    # Silence log date
    tf_log_silence()

    # Argument parser
    (args,model_file,file_name,label_file,input_height,input_width,
     input_mean,input_std,input_layer,output_layer,dir,hit,imwritedir) = arg_parser()
    
    # TF graph/model parser
    (graph,input_operation,output_operation)  = graph_parser(model_file,input_layer,output_layer)
    
    # Check dir for save images (redo code, no long use may remove)
#    if not os.path.exists(imwritedir):
#        os.makedirs(imwritedir)

    # LSW change inference single image to images in DIR #
    # Load Dir lsw@
    #img_list, imgNum = load_dir(args.dir)

    # LSW solve TX2 cuda error after random beboot.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # cv2 read once of video stream from URL to get the round size of frame.
    url = list_url[0]#"http://117.56.234.218:80/T1-437K+700"
    cap = cv2.VideoCapture(url)

    # Get HxW
    ret, frame = cap.read()
    height, width, channels = frame.shape
    print("stream shape:", height, width, channels)

        
    # Get ccTVInfo
    #df, index = get_cctv_info()
    # Get ccTVInfo for only one CCTV
    #df, index = get_cctv_info_1()
    df, index = get_cctv_info_switch()
    #df, index = get_cctv_info_switch_only_one()
    print('CCTV List: ', index)

    # TF Session (will remove after tf2.0)
    with tf.Session(graph=graph, config=config) as sess:

        # Initial count
        now_count = 0
        hit_count = 0


        # Save the Video [CV2]
        FPS= 5
        FrameSize=(width, height)#(int(width/2), int(height/2)) # MUST set or not thing happen !!!! vtest is 768,576.or 320,240 or 352,240
        isColor=1# flag for color(true or 1) or gray (0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') #X264
        out = cv2.VideoWriter(args.video_out, fourcc, FPS, FrameSize)

        # Limited looping
        n=1
#        while(n < 2):
#        while(True):
        while(n < 50):#while(N<2), while(True) # 60 = 10 minutes * 10 hours.
            hour = datetime.datetime.now().hour
            minu = datetime.datetime.now().minute
            
            # Timer to timing when to execute.
#            if 1:#Just run it as soon as possible.
            if str(hour) in day_h and (minu % 30 == 0):#for every 15 minutes run once.
                # Show steps.
                print("steps=", n)
                show_steps_flag(n)
                n += 1
                
                # Downlod rainfall data and grap it.
                print('== get_rainfall_data')
                get_rainfall_data()
                rainfall_data = read_rainfall_data()

                # Set current time stamp.
                #realtime = time.strftime("%Y-%m-%d:%H:%M:%S", time.localtime(time.time()))
                saved_time = time.strftime("%Y%m%d-%H%M", time.localtime(time.time()))
                saved_time_sec = time.strftime("%Y-%m-%d:%H:%M:%S", time.localtime(time.time()))

                ''' IF run as usual, BLOCK this block! '''
                if Para_dl_ON is 'True':
                    #             Parall streamimg getting              #

                    imwritedir = saved_time
                    imwritedir_path = '/tmp/' + saved_time + '/'
                    if not os.path.exists(imwritedir_path):
                        os.makedirs(imwritedir_path)
                    start = time.time()
                    parall_excutor_dl(saved_time, imwritedir_path, df, index, num_workers)
                    print('\nGet images time: ', time.time() - start)
                    time.sleep(1)
                    #             Parall streamimg getting             #
                    print('Take a rest...10')
                    print('Take a rest...10')
                    print('Take a rest...10')
                    print('Take a rest...10')
                    time.sleep(10)

                    #             Inference model interface             #
                    ''' If block above Parallel and change img_list to id_count that code came back to usual realtime inference method'''
                #imwritedir = '/tmp/20191017-1106/' # for hard test.
                
                
                if Para_if_ON is 'True':
                    # Opt-1. Run as Parallel url getting, then
                    start = time.time()
#                    path_to_imgs = '/tmp/' + imwritedir + '/'              #for Para_dl_ON not True
#                    img_list, imgNum = load_dir('/tmp/' + imwritedir + '/')#for Para_dl_ON not True
                    imwritedir_path = '/tmp/' + imwritedir + '/'
#                    path_to_imgs = imwritedir                #for Para_dl_ON is True
                    img_list, imgNum = load_dir(imwritedir_path)#for Para_dl_ON is True
                    JSON_TEMP = []
                    FOOD_TEMP = []
                    FLAG_TEMP = []
                    idx = 0
                    for im_name in img_list:
                        inf_im_json(sess, label_file, viz, im_name, df, imwritedir_path, saved_time, JSON_TEMP, FOOD_TEMP, FLAG_TEMP, rainfall_data,
                               input_operation, output_operation,
                               input_height,
                               input_width,
                               input_mean,
                               input_std)
                    print('\nInference [{}] images taken {}:\n'.format(len(img_list), time.time() - start))
                    time.sleep(2)

                    # Save mseg, data to Json file
                    s_j_path = '/tmp/' + saved_time + '.json'
                    with open(s_j_path, 'w', encoding='utf-8') as jsn:
                        json.dump(JSON_TEMP, jsn, ensure_ascii=False, indent=4) # indent=4 with nice json layout.
                    f_j_path = '/tmp/' + saved_time + '-FOOD' + '.json'
                    with open(f_j_path, 'w', encoding='utf-8') as jsn:
                        json.dump(FOOD_TEMP, jsn, ensure_ascii=False, indent=4) # indent=4 with nice json layout.
                    l_j_path = '/tmp/' + saved_time + '-FLAG' + '.json'
                    with open(l_j_path, 'w', encoding='utf-8') as jsn:
                        json.dump(FLAG_TEMP, jsn, ensure_ascii=False, indent=4) # indent=4 with nice json layout.

                    # copy mseg, data to a Json obj [JSON]
                    s_j_temp = json.dumps(JSON_TEMP, ensure_ascii=False, indent=4) # indent=4 with nice json layout.
    #                print(j_temp)
                    s_j = json.loads(s_j_temp)
                    print('Single JSON done with [{}] instance.\n'.format(len(s_j)))
                    time.sleep(1)
                    # copy mseg, data to a Json obj [FLAG]
                    l_j_temp = json.dumps(FLAG_TEMP, ensure_ascii=False, indent=4) # indent=4 with nice json layout.
    #                print(l_temp)
                    l_j = json.loads(l_j_temp)
                    print('FLAG JSON done with [{}] instance.\n'.format(len(l_j)))
                    time.sleep(1)

    #             Inference model interface             #


                #                 Post result to aws                 #
                # FLAG json post
                print('\n Start F\'L\'AG JSON Post....')
                start = time.time()
                if Single_po_ON is 'True':
                    single_post_ELK(url_nowred, l_j)
                    print('FLAG post done: ', time.time() - start)

                # Single json post
                print('\n Start Single JSON Post....')
                start = time.time()
                if Single_po_ON is 'True':
                    single_post_ELK(url_6, s_j)
                    print('Single post done: ', time.time() - start)



                #Paralle post req
#                print('Start req post....')
#                start = time.time()
#                if Para_po_ON is 'True':
#                    print('\n Parallel Post req...\n')
#                    num_workers = 36 #12 or 36, for MLG or TWCC.
#                    parall_excutor_po_req(num_workers, url_3, JSON_TEMP, saved_time_sec)
#                    print('REQ post done: ', time.time() - start)

                    # Paralle post ELK
#                    print('\n Parallel Post ELK...\n')
#                    num_workers = 12 #12 or 36, for MLG or TWCC.
#                    parall_excutor_po_ELK(num_workers, url_5, JSON_TEMP, saved_time_sec)

                    # Serial slow post ELK
#                    print('\n Serial slow Post ELK...\n')
#                    parall_excutor_po_ELK_serial(num_workers, url_5, JSON_TEMP, saved_time_sec)
                #                 Post result to aws                 #



                #                 Post result to DB                 #
#                print('Start SQL update....')
#                start = time.time()
#                if DB_po_ON is 'True':
#                    context = 'What is this!'
                    # parallel sql
#                    parall_excutor_po_sql(num_workers, JSON_TEMP, context) #parallel sql tool slow!!
                    # Serial sql
#                    for jsn in JSON_TEMP:
#                        single_post_sql(jsn, context)
#                print('\nSQL done: ', time.time() - start)

                print('Saved to ' ,imwritedir)
#                time.sleep(60)#for jump a gap to next 10 minutes
                #                 Post result to DB                 #

                
                # FLAG report to linebot #
                send_report_to_linebot(imwritedir)



# close all cv objects
cap.release()
out.release()
cv2.destroyAllWindows()




        
