#first_script.py

import pandas
from pandas import DataFrame
from random import shuffle
import numpy as np 
from sklearn.utils import shuffle
from datetime import datetime
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

    kp = kp.dropna()
    #kp = kp.fillna(kp.median())
    #kp.apply(lambda x: x.fillna(x.mean()),axis=0)
    #kp[kp.columns[:-1]].values = kp[kp.columns[:-1]].values.fillna(kp[kp.columns[:1]].values.median())
    #drop all rows with missing values in them

    #print kp.describe()

    ''' The image column has pixel values separated by a space.
    We are converting the values to numpy arrays. 
    '''

    kp['Image'] = kp['Image'].apply(lambda im: np.fromstring(im,sep=' '))

    #print kp['Image']

    X= np.vstack(kp['Image'].values)
    

    #Implementing Contrast Stretching
    
    p5 = np.percentile(X, 5)
    p95 = np.percentile(X, 95)
    X = exposure.rescale_intensity(X, in_range=(p5, p95))
    

    #Implementing histogram equalization

    #X = exposure.equalize_hist(X)
    

    #scaling pixel values to [0,1]
    X = X/255

    #print X
    
    X= X.astype(np.float32)
    #used to cast the numpy array into float32 type


    '''
    #Implementing Contrast Stretching
    p2 = np.percentile(X, 5)
    p98 = np.percentile(X, 95)
    X = exposure.rescale_intensity(X, in_range=(p2, p98))
    '''

    #print 'hello1',X

    print kp.columns.values  #outputs column headings

    #print 'hello2', kp.columns[:-1].values
    #doesn't take the image column
    #means take all the columns but the last!

    #print 'hello3', kp[kp.columns[:1]].values
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


X,y = load()


'''

print("X.shape == {}; X.min == {}; X.max == {}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {}; y.max == {}".format(
    y.shape, y.min(), y.max()))'''


#Now we will start using Lasagne


'''Initially creating a Neural network with only 
one hidden layer'''

from lasagne import layers

from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet

net1 = NeuralNet(             
        layers = [ #3 layers, including 1 hidden layer
                ('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),

                ],

        #layer parameters for every layer
        #refer to the layer using it's prefix

        #96X96 input pixels per batch
        #None -  gives us variable batch sizes
        input_shape = (None, 9216),

        #number of hidden units in a layer
        hidden_num_units = 100,

        #Output layer uses identity function
        #Using this, the output unit's activation becomes a linear combination of activations in the hidden layer.

        #Sine we haven't chosen anything for the hidden layer
        #the default non linearity is rectifier, which is chosen as the activation function of the hidden layer. 
        output_nonlinearity = None,

        #30 target values
        output_num_units = 30,


        #Optimization Method:
        #The following parameterize the update function
        #The update function updates the weight of our network after each batch has been processed.

        update = nesterov_momentum,
        update_learning_rate = 0.01, #step size of the gradient descent
        update_momentum = 0.9,

        #Regression flag set to true as this is a regression 
        #problem and not a classification problem
        regression=True, 

        max_epochs = 100,

        #speci fies that we wish to output information during training
        verbose = 1,


        #NOTE: Validation set is automatoically chosen as 20% (or 0.2) of the training samples for validation. 
        #Thus by default, eval_size=0.2 which user can change
        )


#X,y = load()
net1.fit(X,y)
#To train the neural nerwork

X, _ = load(test=True)
y_pred = net1.predict(X)
y_pred = y_pred*48 + 48
y_pred = y_pred.clip(0,96)

#print "Final predictions:" , y_pred 

df = DataFrame(y_pred, columns = col)

#print "Dataframe", df

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

#print "first column:", y.values

from matplotlib import pyplot

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


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

X, _ = load(test=True)
y_pred = net1.predict(X)

fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)

pyplot.show() 
