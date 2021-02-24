'''
given frame by frame data for box2, 
mark each frame for car and good or not
by concatinating data with box1's data
'''
import pandas as pd 

v1 = 'video_analysis_box1.csv'
v2 = 'video_analysis_box2.csv'

df1 = pd.read_csv(v1)
df2 = pd.read_csv(v2)

df1 = df1.fillna(0)

df2['Car?'] = df1["Car?"]
df2['Good?'] = df1["Good?"]

#df2.to_csv(v2)

'''
Deploy and test the env
* create new instance for opencv version, alongside of rekognition
* delete the images instead of moving to new dir -> imageprocessor.json's param
'''


