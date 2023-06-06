import os
import pandas as pd
import shutil

path = 'video'
label_dir = 'LA_mds_updrs_label_20230516update_finish.xls'

L_dataset = 'video_dataset_L'
R_dataset = 'video_dataset_R'

if not os.path.isdir(L_dataset):
    os.mkdir(L_dataset)

if not os.path.isdir(R_dataset):
    os.mkdir(R_dataset)


label_file = pd.read_excel(label_dir, index_col = 'VideoId')

for file in os.listdir(path):
    #print(file)
    temp_str = file.split('_')
    file_name = str(temp_str[0][2:] + '_' + temp_str[1] + '_' + temp_str[2] + '_' + temp_str[3] + '_' + temp_str[4][0])
    print(file_name)
    label_L, label_R = label_file.loc[file_name]['mds_updrs_left'], label_file.loc[file_name]['mds_updrs_right']
    label_L, label_R = int(label_L), int(label_R)
    if not os.path.isdir(L_dataset + '/' + str(label_L)):
        os.mkdir(L_dataset + '/' + str(label_L))

    if not os.path.isdir(R_dataset + '/' + str(label_R)):
        os.mkdir(R_dataset + '/' + str(label_R))

    shutil.copyfile(path + '/' + file, L_dataset + '/' + str(label_L) + '/' + file)
    shutil.copyfile(path + '/' + file, R_dataset + '/' + str(label_R) + '/' + file)
    



