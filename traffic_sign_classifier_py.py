import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
from tempfile import TemporaryFile
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# Grayscales data.
def makegloomy(img):
    print (format(img.shape))
    return np.mean(img, axis=3)

# Zero centers data.
def herotozero(img):
    img -= np.mean(img)
    img /= np.std(img)
    return img

# Creates warped image.
def augment_image(img, *, angle=20, shear=10, translate=5, size=32):
    angle = 20
    shear = 10
    translate = 5
    size = 32

    rotate = np.random.uniform(angle)-angle/2  
    Rotation = cv2.getRotationMatrix2D((size/2,size/2),rotate,1)

    shift_x = translate*np.random.uniform()-translate/2
    shift_y = translate*np.random.uniform()-translate/2
    Final_Translate = np.float32([[1,0,shift_x],[0,1,shift_y]])

    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+shear*np.random.uniform()-shear/2
    pt2 = 20+shear*np.random.uniform()-shear/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    Final_Shear = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rotation,(size,size))
    img = cv2.warpAffine(img,Final_Translate,(size,size))
    img = cv2.warpAffine(img,Final_Shear,(size,size))

    return img

# Manages augmentation generation.
def gen_augment(X_train, y_train, *, transformed=5):
    count = 0
    img = np.zeros((1,len(X_train[1]),len(X_train[2])))
    yval = np.zeros((1,))
    while count < len(X_train[0]):
        track = 0
        yval[0] = y_train[count]
        while track < transformed:
            img[0] = augment_image(X_train[count])
            X_train = np.concatenate((X_train, img), axis=0)
            y_train = np.concatenate((y_train, yval), axis=0) #update arrays
            track += 1
        if count % 5000 == 0:
            print (count) #for tracking purposes
        count += 1

    return X_train, y_train

# Saves augmented images.
def create_new(X_train, y_train):
    X_train, y_train = gen_augment(X_train, y_train)

    np.save('augmentedx.npy', X_train)
    np.save('augmentedy.npy', y_train)
    
    return X_train, y_train

# Reshapes array for neural network.
def reshape_for_model(X):
    X = np.reshape(X, X.shape + (1,))
    
    return X

# Creates training split.
def prep_data(X_train, X_test, y_train, *, test_size=0.2):
    X_train = reshape_for_model(X_train)
    X_test = reshape_for_model(X_test)
    
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_size)
    
    return X_train, X_validation, X_test, y_train, y_validation

# Defines model.
def LeNetBased(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    #Layer 1: Convolutional. Input = 32x32x1. Output = 30x30x12.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 12), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(12))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    #Activation.
    conv1 = tf.nn.relu(conv1)

    #Layer 2: Convolutional. Output = 26x26x24.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 12, 24), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(24))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    #Activation. Output = 13x13x24
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    
    #Layer 3: Convolutional. Output = 8x8x32.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(6, 6, 24, 32), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(32))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    
    #Activation. Output = 4x4x32
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    
    fc0 = flatten(conv3)
    
    #Layer 4: Fully Connected. Input = 512. Output = 172.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(512, 172), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(172))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    
    #Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    #Layer 5: Fully Connected. Input = 172. Output = 86.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(172, 86), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(86))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    
    #Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    #Layer 6: Fully Connected. Input = 86. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(86, 43), mean = mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits

# Runs model evaluation.
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


training_file = "train.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Test Set:       {} samples".format(len(X_test)))

# TODO: Number of training examples
n_train = format(len(X_train))

# TODO: Number of testing examples.
n_test = format(len(X_test))

# TODO: What's the shape of an traffic sign image?
image_shape =format(X_train[0].shape)

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

%matplotlib inline

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.subplot(121),plt.imshow(X_train[index].squeeze()), plt.title("Before Preprocessing");
print(y_train[index])

plt.hist(y_train, bins=43);

X_train = herotozero(makegloomy(X_train))
X_test = herotozero(makegloomy(X_test))

print (format(X_train[index].shape))

plt.subplot(121),plt.imshow(X_train[index].squeeze(), cmap="gray"), plt.title("Gray'd and Zero'd");

try:
    X_train = np.load('augmentedx.npy')
    y_train = np.load('augmentedx.npy')
except IOError:
    print("Generating Training Data...")
    X_train, y_train = create_new(X_train, y_train)
    
plt.subplot(121),plt.imshow(X_train[n_train+index].squeeze(), cmap="gray"), plt.title("Augmented Image");

X_train, X_validation, X_test, y_train, y_validation = prep_data(X_train, X_test, y_train)

print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))


EPOCHS = 50
BATCH_SIZE = 128
keep_prob = 1 #disable dropout as the architecture is not wide enough (hardware is not strog enough with so many samples)

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

rate = 0.001

logits = LeNetBased(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        if i > 7:
            rate = .0001
            learning_rate = .0001
        if i > 15:
            rate = .0001
            learning_rate = .0001
        if i > 37:
            rate = .00001
            learning_rate = .00001
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenetbasedmodel')
    print("Model saved")
    
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    keep_prob = 1

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
