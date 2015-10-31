#ConvNets_2.py

import pandas
from pandas import DataFrame
from random import shuffle
import numpy as np 
from sklearn.utils import shuffle
from datetime import datetime
import cPickle as pickle
from skimage import exposure
from skimage import data


kpTemp = pandas.read_csv("training.csv")
col = kpTemp.columns[:-1].values

def load(test=False, cols=None):
    '''
    Loads the data from test.csv if test is True, 
    else loads the data from training.csv.
    Pass a list of cols, if interested in only a subset of columns. 
    '''

    if test == False:
        kp = pandas.read_csv("training.csv")
    else:
        kp = pandas.read_csv("test.csv")

    #print kp.describe()

    #kp = kp.dropna()
    kp = kp.fillna(kp.median())
    #kp.apply(lambda x: x.fillna(x.mean()),axis=0)
    #kp[kp.columns[:-1]].values = kp[kp.columns[:-1]].values.fillna(kp[kp.columns[:1]].values.median())
    #drop all rows with missing values in them

    #print kp.describe()

    ''' The image column has pixel values separated by a space.
    We are converting the values to numpy arrays. 
    '''

    kp['Image'] = kp['Image'].apply(lambda im: np.fromstring(im,sep=' '))

    print kp['Image']

    X= np.vstack(kp['Image'].values)
    #scaling pixel values to [0,1]

    #Implementing Contrast Stretching
    p2 = np.percentile(X, 5)
    p98 = np.percentile(X, 95)
    X = exposure.rescale_intensity(X, in_range=(p2, p98))
    

    X = X/255

    print X

    X= X.astype(np.float32)
    #used to cast the numpy array into float32 type

    print 'hello1',X

    print kp.columns.values  #outputs column headings

    print 'hello2', kp.columns[:-1].values
    #doesn't take the image column
    #means take all the columns but the last!

    print 'hello3', kp[kp.columns[:1]].values
    #fetches the values from the first column: left_eye_centre_x



    ''' 
    Hence verfified that we can safely use kp[kp.columns[:-1]].values,
    to fetch all the values of our features.
    Given X, which is the image pixel attributes, we need to learn to correctly 
    predict the values of features.'''

    if not test: #means we are dealing with training data
        y = kp[kp.columns[:-1]].values

        y = (y-48)/48
        #scale target coordinates to [-1,1]
        print 'helloY', y

        X,y = shuffle(X,y, random_state = 42)

        y = y.astype(np.float32)

    else:
        y = None


    return X,y


#X,y = load()



'''
print("X.shape == {}; X.min == {}; X.max == {}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {}; y.max == {}".format(
    y.shape, y.min(), y.max()))'''

def load2d(test=False, cols=None):
    X, y = load(test=test, cols= cols)
    X = X.reshape(-1,1,96,96)
    return X, y



#Now we will start using Lasagne


'''Initially creating a Neural network with only 
one hidden layer'''

from lasagne import layers

from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet

net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=400,
    verbose=1,
    )

X, y = load2d()  # load 2-d data
net2.fit(X, y)

'''
# Training for 1000 epochs will take a while.  We'll pickle the
# trained model so that we can load it back later:
import cPickle as pickle
with open('net2.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)
'''

X, _ = load(test=True)
y_pred = net2.predict(X)
y_pred = y_pred*48 + 48
y_pred = y_pred.clip(0,96)

#print "Final predictions:" , y_pred 

df = DataFrame(y_pred, columns = col)

print "Dataframe", df

lookup_table = pandas.read_csv("IdLookupTable.csv")
values = []

for index, row in lookup_table.iterrows():
    values.append((
        row['RowId'],
        df.ix[row.ImageId - 1][row.FeatureName],
        ))

now_str = datetime.now().isoformat().replace(':', '-')
submission = DataFrame(values, columns=('RowId', 'Location'))
filename = 'submission-{}.csv'.format(now_str)
submission.to_csv(filename, index=False)
print("Wrote {}".format(filename))

#print "first column:", y.values'''

from matplotlib import pyplot
'''
train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
pyplot.plot(train_loss, linewidth=3, label="train")
pyplot.plot(valid_loss, linewidth=3, label="valid")
pyplot.grid()
pyplot.legend()
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
pyplot.ylim(1e-3, 1e-2)
pyplot.yscale("log")
pyplot.show()
'''