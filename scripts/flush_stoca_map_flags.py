# Basic modules
import json
import cv2
import datetime

# MBTK modules
from MBTK.mbtk import *

# Magic help module
from MBTK.help import *

# Magic number module
from MBTK.Pnum import *

# Trigger setting
Single_po_ON = 'True'

# OK GO
if __name__ == "__main__":
    df, index = get_cctv_info_switch()
    JSON_TEMP = []
    reset_time = '[Reset] ' + str(datetime.datetime.now())
    print(reset_time)
    
    for id_count in range(index):#for id_count in range(20):#for url in url_list:
        tvid = df.iloc[id_count]['cctvid']

        mesg = { "tvid":tvid, "road":'reset', "px":'f', "py":'f',"realtime":str(reset_time), "im_name":'[Reset]'}# realtime -> saved_time_sec
        data = { "normal":0, "floods":0, "unknow":0, "color_num":1}
        jj = {"mseg":mesg, "data":data}
        JSON_TEMP.append(jj)
    
        # add single post again for test db is ok or not
#        headers = {'Content-Type': 'application/json'}
#        r = requests.post(url_6, headers=headers, json=jj)
#        print(r.status_code, r.reason, '--------------')
#        print(jj)

    # Save mseg, data to Json file
    j_path = './' + 'flush_flag' + '.json'
    with open(j_path, 'w', encoding='utf-8') as jsn:
        json.dump(JSON_TEMP, jsn, ensure_ascii=False, indent=4) # indent=4 with nice json layout.

    # copy mseg, data to a Json obj
    j_temp = json.dumps(JSON_TEMP, ensure_ascii=False, indent=4) # indent=4 with nice json layout.
    j = json.loads(j_temp)

    # Onece post
    print('Start Single JSON Post....')
    start = time.time()
    if Single_po_ON is 'True':
        single_post_ELK(url_6, j)
        print('Single post done: ', time.time() - start)
