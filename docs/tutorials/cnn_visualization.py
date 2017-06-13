# -*- coding: utf-8 -*-

# start
try:
    import tensorflow as tf
    import numpy as np
    import pickle
    from tensorflow.python.platform import gfile
    from random import randint
    import os
    from scipy.misc import imsave
    from matplotlib import pyplot as plt
except ImportError:
    raise ValueError("Please install tensorflow and matplotlib.")


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict


def initWeight(shape):
    weights = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weights)


# start with 0.1 so reLu isnt always 0
def initBias(shape):
    bias = tf.constant(0.1, shape=shape)
    return tf.Variable(bias)


# the convolution with padding of 1 on each side, and moves by 1.
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# max pooling basically shrinks it by 2x, taking the highest value on each feature.
def maxPool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


batchsize = 50
imagesize = 32
colors = 3

sess = tf.InteractiveSession()

img = tf.placeholder("float", shape=[None, imagesize, imagesize, colors])
lbl = tf.placeholder("float", shape=[None, 10])
# for each 5x5 area, check for 32 features over 3 color channels
wConv1 = initWeight([5, 5, colors, 32])
bConv1 = initBias([32])
# move the conv filter over the picture
conv1 = conv2d(img, wConv1)
# adds bias
bias1 = conv1 + bConv1
# relu = max(0,x), adds nonlinearality
relu1 = tf.nn.relu(bias1)
# maxpool to 16x16
pool1 = maxPool2d(relu1)
# second conv layer, takes a 16x16 with 32 layers, turns to 8x8 with 64 layers
wConv2 = initWeight([5, 5, 32, 64])
bConv2 = initBias([64])
conv2 = conv2d(pool1, wConv2)
bias2 = conv2 + bConv2
relu2 = tf.nn.relu(bias2)
pool2 = maxPool2d(relu2)
# fully-connected is just a regular neural net: 8*8*64 for each training data
wFc1 = initWeight([(imagesize / 4) * (imagesize / 4) * 64, 1024])
bFc1 = initBias([1024])
# reduce dimensions to flatten
pool2flat = tf.reshape(pool2, [-1, (imagesize / 4) * (imagesize / 4) * 64])
# 128 training set by 2304 data points
fc1 = tf.matmul(pool2flat, wFc1) + bFc1
relu3 = tf.nn.relu(fc1)
# dropout removes duplicate weights
keepProb = tf.placeholder("float")
drop = tf.nn.dropout(relu3, keepProb)
wFc2 = initWeight([1024, 10])
bFc2 = initWeight([10])
# softmax converts individual probabilities to percentages
guesses = tf.nn.softmax(tf.matmul(drop, wFc2) + bFc2)
# how wrong it is
cross_entropy = -tf.reduce_sum(lbl * tf.log(guesses + 1e-9))
# theres a lot of tensorflow optimizers such as gradient descent
# adam is one of them
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# array of bools, checking if each guess was correct
correct_prediction = tf.equal(tf.argmax(guesses, 1), tf.argmax(lbl, 1))
# represent the correctness as a float [1,1,0,1] -> 0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_all_variables())

batch = unpickle("cifar-10-batches-py/data_batch_1")

validationData = batch["data"][555:batchsize + 555]
validationRawLabel = batch["labels"][555:batchsize + 555]
validationLabel = np.zeros((batchsize, 10))
validationLabel[np.arange(batchsize), validationRawLabel] = 1
validationData = validationData / 255.0
validationData = np.reshape(validationData, [-1, 3, 32, 32])
validationData = np.swapaxes(validationData, 1, 3)

saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint(os.getcwd() + "/training/"))


# train for 20000
# print mnistbatch[0].shape
def train():
    for i in range(20000):
        randomint = randint(0, 10000 - batchsize - 1)
        trainingData = batch["data"][randomint:batchsize + randomint]
        rawlabel = batch["labels"][randomint:batchsize + randomint]
        trainingLabel = np.zeros((batchsize, 10))
        trainingLabel[np.arange(batchsize), rawlabel] = 1
        trainingData = trainingData / 255.0
        trainingData = np.reshape(trainingData, [-1, 3, 32, 32])
        trainingData = np.swapaxes(trainingData, 1, 3)

        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                img: validationData, lbl: validationLabel, keepProb: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

            if i % 50 == 0:
                saver.save(sess, os.getcwd() + "/training/train", global_step=i)

        optimizer.run(feed_dict={img: trainingData, lbl: trainingLabel, keepProb: 0.5})
        print(i)


def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat(i, [out, out])
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def display():
    print("displaying")

    batchsizeFeatures = 50
    imageIndex = 56

    inputImage = batch["data"][imageIndex:imageIndex + batchsizeFeatures]
    inputImage = inputImage / 255.0
    inputImage = np.reshape(inputImage, [-1, 3, 32, 32])
    inputImage = np.swapaxes(inputImage, 1, 3)

    inputLabel = np.zeros((batchsize, 10))
    inputLabel[np.arange(1), batch["labels"][imageIndex:imageIndex + batchsizeFeatures]] = 1
    # inputLabel = batch["labels"][54]

    # prints a given image

    # saves pixel-representations of features from Conv layer 1
    featuresReLu1 = tf.placeholder("float", [None, 32, 32, 32])
    unReLu = tf.nn.relu(featuresReLu1)
    unBias = unReLu
    unConv = tf.nn.conv2d_transpose(unBias, wConv1, output_shape=[batchsizeFeatures, imagesize, imagesize, colors],
                                    strides=[1, 1, 1, 1], padding="SAME")
    activations1 = relu1.eval(feed_dict={img: inputImage, lbl: inputLabel, keepProb: 1.0})
    print(np.shape(activations1))

    # display features
    for i in range(32):
        isolated = activations1.copy()
        isolated[:, :, :, :i] = 0
        isolated[:, :, :, i + 1:] = 0
        print(np.shape(isolated))
        totals = np.sum(isolated, axis=(1, 2, 3))
        best = np.argmin(totals, axis=0)
        print(best)
        pixelactive = unConv.eval(feed_dict={featuresReLu1: isolated})
        # totals = np.sum(pixelactive,axis=(1,2,3))
        # best = np.argmax(totals,axis=0)
        # best = 0
        saveImage(pixelactive[best], "activ" + str(i) + ".png")
        saveImage(inputImage[best], "activ" + str(i) + "-base.png")

    # display same feature for many images
    # for i in xrange(batchsizeFeatures):
    #     isolated = activations1.copy()
    #     isolated[:,:,:,:6] = 0
    #     isolated[:,:,:,7:] = 0
    #     pixelactive = unConv.eval(feed_dict={featuresReLu1: isolated})
    #     totals = np.sum(pixelactive,axis=(1,2,3))
    #     best = np.argmax(totals,axis=0)
    #     saveImage(pixelactive[i],"activ"+str(i)+".png")
    #     saveImage(inputImage[i],"activ"+str(i)+"-base.png")



    # saves pixel-representations of features from Conv layer 2
    featuresReLu2 = tf.placeholder("float", [None, 16, 16, 64])
    unReLu2 = tf.nn.relu(featuresReLu2)
    unBias2 = unReLu2
    unConv2 = tf.nn.conv2d_transpose(unBias2, wConv2,
                                     output_shape=[batchsizeFeatures, imagesize / 2, imagesize / 2, 32],
                                     strides=[1, 1, 1, 1], padding="SAME")
    unPool = unpool(unConv2)
    unReLu = tf.nn.relu(unPool)
    unBias = unReLu
    unConv = tf.nn.conv2d_transpose(unBias, wConv1, output_shape=[batchsizeFeatures, imagesize, imagesize, colors],
                                    strides=[1, 1, 1, 1], padding="SAME")
    activations1 = relu2.eval(feed_dict={img: inputImage, lbl: inputLabel, keepProb: 1.0})
    print(np.shape(activations1))

    # display features
    # for i in xrange(64):
    #     isolated = activations1.copy()
    #     isolated[:,:,:,:i] = 0
    #     isolated[:,:,:,i+1:] = 0
    #     pixelactive = unConv.eval(feed_dict={featuresReLu2: isolated})
    #     # totals = np.sum(pixelactive,axis=(1,2,3))
    #     # best = np.argmax(totals,axis=0)
    #     best = 0
    #     saveImage(pixelactive[best],"activ"+str(i)+".png")
    #     saveImage(inputImage[best],"activ"+str(i)+"-base.png")


    # display same feature for many images
    # for i in xrange(batchsizeFeatures):
    #     isolated = activations1.copy()
    #     isolated[:,:,:,:8] = 0
    #     isolated[:,:,:,9:] = 0
    #     pixelactive = unConv.eval(feed_dict={featuresReLu2: isolated})
    #     totals = np.sum(pixelactive,axis=(1,2,3))
    #     # best = np.argmax(totals,axis=0)
    #     # best = 0
    #     saveImage(pixelactive[i],"activ"+str(i)+".png")
    #     saveImage(inputImage[i],"activ"+str(i)+"-base.png")


def saveImage(inputImage, name):
    # red = inputImage[:1024]
    # green = inputImage[1024:2048]
    # blue = inputImage[2048:]
    # formatted = np.zeros([3,32,32])
    # formatted[0] = np.reshape(red,[32,32])
    # formatted[1] = np.reshape(green,[32,32])
    # formatted[2] = np.reshape(blue,[32,32])
    # final = np.swapaxes(formatted,0,2)/255
    final = inputImage
    final = np.rot90(np.rot90(np.rot90(final)))
    imsave(name, final)


def main(argv=None):
    display()
    # train()


if __name__ == '__main__':
    tf.app.run()

    # end
