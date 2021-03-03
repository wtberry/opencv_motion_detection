'''
Will analyze alpr log files
'''
import pandas as pd
import json
import os
#from parse import compile as pcompile, parse
import glob
import csv
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid




def prep_data(df):
    df["ナンバープレート"] = df["ナンバープレート"].str.strip()
    df['処理開始時間'] = pd.to_datetime(df['処理開始時間'])
    df['hour'] = df['処理開始時間'].dt.hour


    df['box_area'] = (df['box_x_max'] - df['box_x_min'])*(df['box_y_max'] - df['box_y_min'])
    df['overlap_area'] = (df['over_x_max'] - df['over_x_min'])*(df['over_y_max'] - df['over_y_min'])
    df['area_of_interest'] = (df['aoi_x_max'] - df['aoi_x_min'])*(df['aoi_y_max'] - df['aoi_y_min'])
    df['box_height_over_width'] = df['box_height'] / df['box_width']
    # checking this
    #df['overlap_right_buttom'] = df[(df['box_x_min'] < df['aoi_x_min']) & (df['aoi_x_min'] < df['aoi_x_max'])]
    df['frame_area'] = 1
    df.loc[df["ナンバープレート"] != "None", 'plate_present'] = 1
    df.loc[df["ナンバープレート"] == "None", 'plate_present'] = 0
    
    df['box_x_center'] = (df['box_x_min']+df['box_x_max'])/2
    df['box_y_center'] = (df['box_y_min']+df['box_y_max'])/2

    local_img_path = 'truck_test'
    df['local_img_path'] = df['img_url'].map(lambda x: url_to_local_img(x, local_img_path))
    print(df['local_img_path'])
    df['motion_score'] = ''
    df['blur_score'] = ''
    for ind, row in df.iterrows():
        dic = json.loads(row['物体詳細'])
        for k, v in dic.items():
            df.loc[ind, k] = v
    # drop value where gate_incoming == ?
    print('gate_incoming unique value:',df.gate_incoming.unique())
    print(df['gate_incoming'].value_counts())
    df = df[df['gate_incoming'] != "?"]
    df = df[df['gate_incoming'] != "-"]
    df = df.dropna(axis=0, subset=['gate_incoming'])

    df['gate_incoming'] = df['gate_incoming'].astype(float)
    df['gate_plate'] = 0
    df.loc[(df['plate_present'] == 1) & (df['gate_incoming'] ==1 ), 'gate_plate'] = 1


    feature_candidate_columns = ['hour','box_x_min','box_y_min', 'box_x_max', 'box_y_max',
            'over_x_min', 'over_y_min','over_x_max', 'over_y_max', 
            'aoi_x_min', 'aoi_y_min', 'aoi_x_max','aoi_y_max', 
            'box_height', 'box_width','box_area', 'overlap_area', 'area_of_interest', 'box_height_over_width',
            'frame_area', 'box_x_center', 'box_y_center',
            'motion_score', 'blur_score']

    image_columns = ['img_url', 'local_img_path']


    '''
    yolo_columns  = ['処理開始時間', 'ナンバープレート','タイムスタンプ','box_x_min',
           'box_y_min', 'box_x_max', 'box_y_max', 'over_x_min', 'over_y_min',
           'over_x_max', 'over_y_max', 'aoi_x_min', 'aoi_y_min', 'aoi_x_max',
           'aoi_y_max', 'box_height', 'box_width', 'img_url', 'gate_incoming',
           'box_area', 'overlap_area', 'area_of_interest', 'box_height_over_width',
           'frame_area', 'plate_present', 'box_x_center', 'box_y_center',
           'local_img_path', 'motion_score', 'blur_score', 'gate_plate',
           'yolo_x_min', 'yolo_x_max', 'yolo_y_min', 'yolo_y_max', 'yolo_time']
    '''
    label_columns = ['plate_present', 'gate_incoming', 'gate_plate']

    ## select columns that are relevant to data analysis purpose
    x = df[feature_candidate_columns]

    y = df[label_columns]
    #df.drop_duplicates(subset=['img_url'], inplace=True, keep=False)
    return x, y, df

pd.set_option("display.max_columns", 45)

def to_float(lp):
    try:
        return float(lp)
    except ValueError:
        return 0

def to_bool(num):
    return True if num == 1 else False

def to_link(string):
    '''
    given url string, return it in a tag
    '''
    fstring = '<a href={url}>img link</a>'.format(url=string)
    return fstring

def url_to_local_img(img_url, img_dir_path):
    '''convert url string into local image path'''
    img_file_name = img_url.split("?")[0].split('/')[-1]
    return os.path.join(img_dir_path, img_file_name)



#def parse_info(fname):
#    '''
#    given file name, open it, parse total # of frames per vehicle, only for ones with detected plate from the file and
#    put it into df
#    '''
#    data = {"car_id":[], "total_frame":[]}
#    with open(fname, 'r') as f:
#        csv_reader = csv.reader(f)
#        for row in csv_reader:
#            #params = p.parse(line)
#            print("Line:", len(row))
#            #print("params:", params)
#            #try:
#            #    print(params.named["detected"])
#            #except AttributeError:
#            #    print("No match found")
#
#file_dir_path = 'alpr_logs'
#file_name_pat  = 'alpr.log*'
#
#log_file_list = glob.glob(os.path.join(file_dir_path, file_name_pat))
#ha = 'vehicle_{car_num:d}.pickle'
#log_msg_fmt ='''{datetime:w} - cam_id: {cam_id:w}, v_id: {vehicle_id:d}, try_num / total_frames: {num_try:d}/{total_frames:d}, plate?: {detected:w}, credit_used: {credit_used}/{total_credit}, detected_vehicle_info: {vehicle_dic}, tot: {motion_score}, lp: {blur_score}, max_retry: {max_retry}, plate_visibility_check: {visibility_coord_dic}, instance: {instance_id}, frame_id: {frame_id}'''
#print(log_msg_fmt)
#p = pcompile(log_msg_fmt)
#
#for fname in log_file_list:
#    print(fname)
#    parse_info(fname)


# test the trained svm performance
#file = 'adjusted_video_box3.csv'
file = 'entry_2_16_9_yolo_hand_labeled.csv'
test_file = 'entry_2_22_hand_labeled.csv'
tf = pd.read_csv(test_file)
df = pd.read_csv(file)

xt, yt, tf = prep_data(tf)

mm =  preprocessing.MinMaxScaler()

xt['blur_score_norm'] = mm.fit_transform(xt['blur_score'].values.reshape(-1,1))
xt['motion_score_norm'] = mm.fit_transform(xt['motion_score'].values.reshape(-1,1))
xt['box_height_over_width_norm'] = mm.fit_transform(xt['box_height_over_width'].values.reshape(-1,1))

# drop features with 0 std
plt_feature_cols = xt.columns[xt.std() > 0.01]
plt_feature_cols = plt_feature_cols.drop(['motion_score', 'blur_score', 'box_height_over_width'])

data = xt[plt_feature_cols]
label = yt['gate_incoming']

#x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)
rfecv = joblib.load('rfecv.joblib')
clf = joblib.load('svc_model.joblib')

xtest = rfecv.transform(data)
pred = clf.predict(xtest)
ac = accuracy_score(label, pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(label,pred)
recall = cm[1][1]/(cm[1][0]+cm[1][1])
precision = cm[1][1]/(cm[0][1]+cm[1][1])
print("recall: ", recall)
print("precision:", precision)
fig = sns.heatmap(cm,annot=True,fmt="d")
plt.show()

# get image path of false negatives
x_test_data = xt.copy()
x_test_data.loc[:, 'pred'] = pred
test_data = pd.concat((x_test_data, label), axis=1)
FN = test_data.loc[(test_data['gate_incoming'] == 1) & (test_data['pred'] == 0), :]

FN_imgs = tf.loc[FN.index, 'local_img_path']

def plot_images(FN_imgs, nrows_ncols):
    fig = plt.figure(figsize=(15., 15.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=nrows_ncols,  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, FN_imgs.to_list()):
        img = plt.imread(im)
        ax.imshow(img)
    
    plt.show()
    
plot_images(FN_imgs, (2,6))
