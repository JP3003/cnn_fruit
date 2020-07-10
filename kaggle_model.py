##
import tensorflow as tf

with open('./resnet_model.json', 'r') as json_file:
    json_savedModel = json_file.read()

model_j = tf.keras.models.model_from_json(json_savedModel)
model_j.load_weights('./resnet_weights.h5')
##
label_file = 'labels.txt'
with open(label_file, "r") as f:
    labels = [x.strip() for x in f.readlines()]

##
import os
from os import listdir, makedirs
from os.path import join, exists, expanduser

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf

##
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    'fruits-360/Training',
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical',
    shuffle=False)
##
validation_generator.reset()
##
with tf.device("/device:GPU:0"):
    y_pred = model_j.predict_generator(validation_generator, steps=(validation_generator.n // 20) + 1, verbose=True)
##
labels = os.listdir('fruits-360/Training')
y_true = validation_generator.labels

##
import numpy as np

pred = (np.argmax(y_pred, axis=1))
##
from sklearn.metrics import confusion_matrix, classification_report

report = classification_report(y_true, pred, target_names=labels, output_dict=True)
##
import pandas as pd
import io

df = pd.DataFrame(report).transpose()
# df.to_csv('resNetreport_final.csv')
##
out_resNet = zip(pred, y_true)
df2 = pd.DataFrame(out_resNet)
df2.to_csv('resNet_outputd_.csv')
##
dicti = validation_generator.class_indices
##
inv_map = {v: k for k, v in dicti.items()}
##
import pandas as pd

df2 = pd.read_csv('ResNet_imp_nowe_out.csv')
df2['pred'] = df2.iloc[:, 1].map(inv_map)
##
df2['true'] = df2.iloc[:, 2].map(inv_map)
##
df2.to_csv('resNet_outputd_final_1.csv')
##
import seaborn as sn
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df2 = pd.read_csv('resNet_outputd_final.csv')
Labels = np.unique(df2.iloc[:, 3])
print(Labels)
cm = confusion_matrix(df2.iloc[:, 3], df2.iloc[:, 4], normalize='all')

df_cm = pd.DataFrame(cm, index=[i for i in Labels], columns=[i for i in Labels])
# plt.figure(figsize=(40, 40))
# ax = sn.heatmap(df_cm, annot=True, square=True, fmt="d", linewidths=.2, cbar_kws={"shrink": 0.8})
# plot_confusion_matrix(df.iloc[:,0],df.iloc[:,1],labels=Labels)
##
# plot_confusion_matrix(df2.iloc[:,3],df2.iloc[:,4],labels=Labels)
plt.figure(figsize=(60, 60))
sn.heatmap(df_cm, linewidths=.2)
# sn.heatmap(cm, cmap="YlGnBu")
##
from pycm import *
import numpy as np

cm_new = ConfusionMatrix(actual_vector=np.array(df2.iloc[:, 4]), predict_vector=np.array(df2.iloc[:, 3])).save_csv(
    'Big_conf_nowe.csv')

##
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    'fruits-360/Test',
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical',
    shuffle=False)

train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = test_datagen.flow_from_directory(
    'fruits-360/Training',
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical',
    shuffle=False)
##
import pandas as pd

training_data = pd.DataFrame(train_generator.classes, columns=['classes'])
testing_data = pd.DataFrame(validation_generator.classes, columns=['classes'])
##
import pandas as pd
from plotly.offline import plot
import plotly.graph_objs as go


##
def create_stack_bar_data(col, df):
    aggregated = df[col].value_counts().sort_index()
    x_values = aggregated.index.tolist()
    y_values = aggregated.values.tolist()
    return x_values, y_values


##
x1, y1 = create_stack_bar_data('classes', training_data)
x1 = list(train_generator.class_indices.keys())
##
trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name="Class Count", marker=dict(color='Olive'))
layout = dict(height=400, width=1200, title='Class Distribution in Test Data', legend=dict(orientation="h"),
              yaxis=dict(title='Class Count'))
fig = go.Figure(data=[trace1], layout=layout);
plot(fig);
##
layer_name = 'conv5_block3_out'
# layer_outputs = [layer.output for layer in model_j.layers[80]]
# Extracts the outputs of the top 12 layers
activation_model = tf.keras.models.Model(inputs=model_j.input, outputs=model_j.get_layer(
    layer_name).output)  # Creates a model that will return these outputs, given the model input
##
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np

img_path = 'fruits-360/Test/Cantaloupe 2/156_100.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)
##
layer_name = 'conv2_block2_2_relu'
intermediate_layer_model = tf.keras.Model(inputs=model_j.input,
                                          outputs=model_j.get_layer(layer_name).output)
activations1 = intermediate_layer_model(img_data)
##
import matplotlib.pyplot as plt
plt.figure(frameon=False)
activations1 = np.array(activations1)
print(activations1.shape)
ix = 1
ax = [plt.subplot(8, 8, ix+1) for ix in range(64)]
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
plt.subplots_adjust(wspace=0, hspace=0)

for i in range(64):
    # specify subplot and turn of axis
    # plot filter channel in grayscale
    ax[i].imshow(activations1[0, :, :, i], cmap='viridis',aspect='auto')

# show the figure

plt.show()
##
for i in range(len(model_j.layers)):
    layer = model_j.layers[i]
    if 'conv' not in layer.name:
        continue
    print(i, layer.name, layer.output.shape)

##
from keras.preprocessing.image import ImageDataGenerator
pca_loader = ImageDataGenerator(rescale = 1/.255)
train_generator = pca_loader.flow_from_directory(
    'fruits-360/PCA',
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False)
##
list = range(200704)
for img,labels in iter(train_generator):
    output = intermediate_layer_model(img)
    output = np.array(output).flatten()
    list = np.vstack((list,output))
##
import numpy as np
from sklearn.decomposition import PCA
PCA_val = PCA(n_components=6)
pca_corn = PCA_val.fit_transform(list_corn)
pca_cantaloupe = PCA_val.fit_transform(list_cantaloupe)
pca_guava = PCA_val.fit_transform(list_guava)
pca_onion = PCA_val.fit_transform(list_onion)
##
import plotly.graph_objects as go
import numpy as np
from plotly.offline import plot
import  plotly.express as px
import pandas as pd

x1, y1, z1 = pca_corn[:,0],pca_corn[:,1],pca_corn[:,2]
x2, y2, z2 = pca_cantaloupe[:,0],pca_cantaloupe[:,1],pca_cantaloupe[:,2]
x3, y3, z3 = pca_guava[:,0],pca_guava[:,1],pca_guava[:,2]
x4, y4, z4 = pca_onion[:,0],pca_onion[:,1],pca_onion[:,2]
x11, y11, z11 = pca_corn[:,3],pca_corn[:,4],pca_corn[:,5]
x22, y22, z22 = pca_cantaloupe[:,3],pca_cantaloupe[:,4],pca_cantaloupe[:,5]
x33, y33, z33 = pca_guava[:,3],pca_guava[:,4],pca_guava[:,5]
x44, y44, z44 = pca_onion[:,3],pca_onion[:,4],pca_onion[:,5]
d1 = {'col1': x1, 'col2': y1, 'col3' : z1,'col4' : 'Corn','col5':1}
d2 = {'col1': x2, 'col2': y2, 'col3' : z2, 'col4' : 'Cantaloupe 2','col5':2}
d3 = {'col1': x3, 'col2': y3, 'col3' : z3, 'col4' : 'Guava','col5':3}
d4 = {'col1': x4, 'col2': y4, 'col3' : z4, 'col4' : 'White Onion','col5':4}
d11 = {'col1': x11, 'col2': y11, 'col3' : z11,'col4' : 'Corn','col5':5}
d22 = {'col1': x22, 'col2': y22, 'col3' : z22, 'col4' : 'Cantaloupe 2','col5':6}
d33 = {'col1': x33, 'col2': y33, 'col3' : z33, 'col4' : 'Guava','col5':7}
d44 = {'col1': x44, 'col2': y44, 'col3' : z44, 'col4' : 'White Onion','col5':8}
df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)
df3 = pd.DataFrame(d3)
df4 = pd.DataFrame(d4)
df11 = pd.DataFrame(d11)
df22 = pd.DataFrame(d22)
df33 = pd.DataFrame(d33)
df44 = pd.DataFrame(d44)
##
frames = [df1,df2,df3,df4,df11,df22,df33,df44]

df_f = pd.concat(frames)
##
fig = px.scatter_3d(df_f,x='col1',y='col2',z='col3',color='col4',size='col5',size_max=18)
plot(fig)
##
fig1 = px.scatter(df_f,x='col1',y='col2',color='col4')
plot(fig1)
##
list = []




##

