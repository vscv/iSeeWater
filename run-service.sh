#!/bin/bash

cd ~/twcc_gpfs/RT_mbNetAll_so_test/
time python -m scripts.label_image_MBNet_Images_DIRLoop_GraphDefault_PRcounting_HuWeiSP_LSW_slit_show_WRARoad_stream_multiCam_mbtk_linebot_parall-dl-po --graph=tf_files/rt_thb_im_224_mf_1.0_steps_8000_20190524fix702v3.pb  --labels=tf_files/rt_thb_im_224_mf_1.0_steps_8000_20190524fix702v3.txt --hit=all --imwritedir=mbimg_2019-10-29-0910
