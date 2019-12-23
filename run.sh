


old vsersion:

time python -m scripts.label_image_MBNet_Images_DIRLoop_GraphDefault_PRcounting_HuWeiSP_LSW_slit_show_WRARoad_stream_multiCam_mbtk_linebot --graph=tf_files/rt_thb_im_224_mf_1.0_steps_8000_20190524fix702v3.pb  --labels=tf_files/rt_thb_im_224_mf_1.0_steps_8000_20190524fix702v3.txt --hit=all --imwritedir=mbimg_2019-08-16_linebotP



=====NEW=====
$pidof python
$kill all-py-pids


$ visdom -port 8888 --hostname 203.145.219.176
# TWCC Docker 目標埠: 8888 對外埠: 52211
# web open  http://203.145.219.176:52211

time python -m scripts.label_image_MBNet_Images_DIRLoop_GraphDefault_PRcounting_HuWeiSP_LSW_slit_show_WRARoad_stream_multiCam_mbtk_linebot_parall-dl-po --graph=tf_files/rt_thb_im_224_mf_1.0_steps_8000_20190524fix702v3.pb  --labels=tf_files/rt_thb_im_224_mf_1.0_steps_8000_20190524fix702v3.txt --hit=all --imwritedir=mbimg_2019-08-18_linebotP


time python -m scripts.label_image_MBNet_Images_DIRLoop_GraphDefault_PRcounting_HuWeiSP_LSW_slit_show_WRARoad_stream_multiCam_mbtk_linebot_parall --graph=tf_files/rt_thb_im_224_mf_1.0_steps_8000_20190524fix702v3.pb  --labels=tf_files/rt_thb_im_224_mf_1.0_steps_8000_20190524fix702v3.txt --hit=all --imwritedir=mbimg_2019-08-14_linebotP
