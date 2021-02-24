'''
Given each frame's data and car, 
This will read and assign car numbers to frames, 
and visualize if wanted.
'''
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
import webbrowser


def to_float(lp):
    try:
        return float(lp)
    except ValueError:
        return 0

def to_bool(num):
    return True if num == 1 else False

#file = 'adjusted_video_box3.csv'
file = 'exit_table.csv'

df = pd.read_csv(file)
df["ナンバープレート"] = df["ナンバープレート"].str.strip()
df = df[['物体画像', 'ナンバープレート','box_x_min', 'box_y_min', 'box_x_max', 'box_y_max', 'over_x_min', 'over_y_min', 'over_x_max', 'over_y_max', 'aoi_x_min', 'aoi_y_min', 'aoi_x_max', 'aoi_y_max', 'box_height', 'box_width']]
df['box_area'] = (df['box_x_max'] - df['box_x_min'])*(df['box_y_max'] - df['box_y_min'])
df['overlap_area'] = (df['over_x_max'] - df['over_x_min'])*(df['over_y_max'] - df['over_y_min'])
df['area_of_interest'] = (df['aoi_x_max'] - df['aoi_x_min'])*(df['aoi_y_max'] - df['aoi_y_min'])
df['box_height_over_width'] = df['box_height'] / df['box_width']
# checking this
#df['overlap_right_buttom'] = df[(df['box_x_min'] < df['aoi_x_min']) & (df['aoi_x_min'] < df['aoi_x_max'])]
df['frame_area'] = 1
df.loc[df["ナンバープレート"] != "None", 'plate_present'] = 1
df.loc[df["ナンバープレート"] == "None", 'plate_present'] = 0

'''
df['plate'] = df['plate_visible?'].apply(to_bool)
df = df.dropna()

df['lp'] = df['lp'].apply(to_float)
df = df.fillna(0)


# group same vehcle into same group https://towardsdatascience.com/pandas-dataframe-group-by-consecutive-same-values-128913875dba
df['car_shift'] = df["car?"].shift()
df['not_equal'] = df['car?'] != df['car_shift']
df['csum'] = df['not_equal'].cumsum()

df.loc[(df.csum % 2 != 0), 'csum'] = -10
df.loc[(df.csum % 2 == 0), 'csum'] = df[df['csum']%2 ==0]/2

#df.to_csv('car_frames_box1.csv', index=False)
#df['car_minus_shift'] = df['Car?'] - df['car_shift']

#gf = df.groupby(['csum']).mean()
gf = df.groupby(['csum']).median()

fig = px.scatter(df, x='tot', y='lp', color='car?')
##hist = px.histogram(df, x=)
#
#fig = px.scatter_3d(df, x='tot', y='lp', z='car?')

'''

def do_click(trace, points, state):
    if points.point_inds:
        ind = points.point_inds[0]
        url = df["物体画像"].iloc[ind]
        webbrowser.open_new_tab(url)    

    
fig = go.FigureWidget(layout={'hovermode': 'closest'})
#fig = px.scatter_3d(df, x='box_height_over_width', y='plate_present', z='width', color='plate_visible?')
scatter = fig.add_scatter(x=df['box_height_over_width'], y=df['plate_present'], mode='markers', marker={"color":df['plate_present']})


fig.show()
#



