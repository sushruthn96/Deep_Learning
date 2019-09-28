import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pt
import struct as st
import math

dicttrain={"trainimgs":"/home/ranjan/Downloads/mnist/train-images-idx3-ubyte",
           "trainlabels":"/home/ranjan/Downloads/mnist/train-labels-idx1-ubyte",
           "testimgs": "/home/ranjan/Downloads/mnist/t10k-images-idx3-ubyte",
           "testlabels": "/home/ranjan/Downloads/mnist/t10k-labels-idx1-ubyte"
           }
learning_rate =0.015
epochs=50

def load_train_imgs(dicttrain,dictval):
    openimgs = open(dicttrain[dictval], 'rb')
    magic = st.unpack('>4B', openimgs.read(4))
    numimgs = st.unpack('>I', openimgs.read(4))[0]  # num of images
    nR = st.unpack('>I', openimgs.read(4))[0]  # num of rows
    nC = st.unpack('>I', openimgs.read(4))[0]  # num of column
    print(magic, numimgs, nR, nC)
    images_array = np.zeros((numimgs, nR, nC))
    nBytesTotal = numimgs * nR * nC * 1  # since each pixel data is 1 byte
    images_array = 255 - np.asarray(st.unpack('>' + 'B' * nBytesTotal, openimgs.read(nBytesTotal))).reshape(
        (numimgs, nR, nC))
    # tf.data()
    print(np.shape((images_array[0])))
    print(images_array[0])
    #i=pt.imshow(images_array[2])
    #print(i)
    #pt.show()
    return images_array,nR,nC,numimgs


def load_train_labels(dicttrain,dictval):
    labels = open(dicttrain[dictval], 'rb')
    magiclabels = st.unpack('>4B', labels.read(4))
    numlabels = st.unpack('>I', labels.read(4))[0]
    labels_array = np.zeros(numlabels)
    labels_array = np.asarray(st.unpack('>' + 'B' * numlabels, labels.read(numlabels))).reshape(numlabels)
    print(labels_array[0])
    # for i in range(numlabels):
    #     labels_array[i] = st.unpack('>I',openimgs.read(4))
    print(magiclabels, numlabels)
    print(labels_array[1], labels_array[2])
    return labels_array,numlabels

def create_placeholders(nh,nc,c,classes):
    X2= tf.placeholder(tf.float32,[None,nh,nc,c],name='X')
    Y2 = tf.placeholder(tf.float32,[None,classes],name='Y')
    #Y= tf.one_hot(indices=Y,depth=3)
    return X2,Y2

def initialize_filters():
    f1 = tf.get_variable('f1',[2,2,1,8], initializer=tf.contrib.layers.xavier_initializer())
    f2 = tf.get_variable('f2',[2,2,8,16], initializer=tf.contrib.layers.xavier_initializer())
    f3= tf.get_variable('f3',[2,2,16,16],initializer=tf.contrib.layers.xavier_initializer())
    dict = {'f1':f1,'f2':f2,'f3':f3}
    return dict

def model(inp,initial_filters):
    print("entering arch")
    z1 = tf.nn.conv2d(inp,initial_filters['f1'],strides=[1,1,1,1],padding='SAME')
    a1= tf.nn.relu(z1)
    p1= tf.nn.max_pool(a1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    z2 = tf.nn.conv2d(p1,initial_filters['f2'],strides=[1,1,1,1],padding='SAME')
    a2 = tf.nn.relu(z2)
    p2= tf.nn.max_pool(a2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    z3 = tf.nn.conv2d(p2,initial_filters['f3'],strides=[1,1,1,1],padding='SAME')
    a3 = tf.nn.relu(z3)
    p3= tf.nn.max_pool(a3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #print(p2)
    p4 = tf.contrib.layers.flatten(p3)
    p5 = tf.layers.dropout(p4,rate=0.1,training=True)
    l2_regularizer = tf.contrib.layers.l2_regularizer(0.01)
    p6 = tf.contrib.layers.fully_connected(p5, 10, activation_fn=None,weights_regularizer=l2_regularizer)
    return p6

def fully_connected_nn(p3):
    p4 = tf.contrib.layers.fully_connected(p3,10,activation_fn=None)
    return p4

def compute_cost(p4,lab):
    #regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    #weights = tf.get_variable(name="weights",regularizer=regularizer)
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = p4, labels = lab))
    reg_constant = 0.001  # Choose an appropriate one.
    loss = cost + reg_constant * sum(reg_variables)
    #cost+=reg_term
    return cost

def create_mini_batches(img,label,size,nos):
    imgssplitlist=np.split(img,1)
    #print(imgssplitlist[0].shape)
    labelsplit=np.split(label,1)
    #print(labelsplit[0].shape)
    #print(len(imgssplitlist))
    #print(imgssplitlist[0].shape[0])
    i=0
    mini_batches=[]
    for imgs in imgssplitlist:
        labels=labelsplit[i]
        #permutation = list(np.random.permutation(imgs))
        #print("permutation is")
        #print(permutation)
        #shuffled_X = imgs[:, permutation]
        #shuffled_Y = labels[:, permutation]
        shuffled_X=imgs
        shuffled_Y=labels
        #print(shuffled_X.shape)
        #print(shuffled_Y.shape)
        num_complete_minibatches = math.floor(imgs.shape[0] / size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * size:(k + 1) * size,:]
            mini_batch_Y = shuffled_Y[k * size:(k + 1) * size,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        if nos % size != 0:
            mini_batch_X = shuffled_X[(num_complete_minibatches) * size:imgs.shape[0],:]
            mini_batch_Y = shuffled_Y[(num_complete_minibatches) * size:imgs.shape[0],:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        i=i+1
    #print(mini_batches[1][0].shape)
    #print(mini_batches[1][1].shape)
    # pt.imshow(np.reshape(mini_batches[0][0][0],(28,28)))
    # pt.show()
    # pt.imshow(np.reshape(mini_batches[0][0][1],(28,28)))
    # pt.show()
    # pt.imshow(np.reshape(mini_batches[1][0][0],(28,28)))
    # pt.show()
    # pt.imshow(np.reshape(mini_batches[1][0][1],(28,28)))
    # pt.show()
    return mini_batches


if __name__ == '__main__':

    tf.reset_default_graph()
    #tf.set_random_seed(1)
    images,nr,nc,numings =load_train_imgs(dicttrain,"trainimgs")
    images=images/255
    testimgs,nrtest,nctest,numingstest=load_train_imgs(dicttrain,"testimgs")
    images=np.reshape(images,(numings,nr,nc,1))
    testimgs = np.reshape(testimgs, (numingstest, nrtest, nctest, 1))
    labels,numlabels=load_train_labels(dicttrain,"trainlabels")
    testlabels,numtestlabels=load_train_labels(dicttrain,"testlabels")
    costarray=[]
    #labels=np.reshape(labels,(numlabels,1))
    one_hot_targets = np.eye(10)[labels]
    one_hot_targets_test = np.eye(10)[testlabels]
    #print(one_hot_targets)
    #labels=tf.one_hot(labels,depth=9,axis=-1)
    X1,Y1 = create_placeholders(nc,nr,1,10)
    print(str(X1)+" Y is " + str(Y1))
    initial_filters=initialize_filters()
    print("entering model")
    p4 = model(X1,initial_filters)
    #p4 = fully_connected_nn(p3)
    print('computing cost')
    cost= compute_cost(p4,Y1)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        for i in range(epochs):
            minibatchcost = 0
            num_batches=int(numings/64)
            #print("creating mini batches")
            minibatches=create_mini_batches(images,one_hot_targets,64,numings)
            #print("minibatches created")
            for j in minibatches:
                #print("uo")
                (imgmini,labelmini)=j
                #labelmin=np.eye(10)[labelmini]
                _,costval=session.run([optimizer,cost],feed_dict={X1:imgmini,Y1:labelmini})
                minibatchcost+=costval/len(minibatches)
            print(str(minibatchcost))
            costarray.append(minibatchcost)
            #print("cost is "+str(costval)

        pt.plot(np.squeeze(costarray))
        pt.ylabel('cost')
        pt.xlabel('iterations (per tens)')
        pt.title("Learning rate =" + str(learning_rate))
        pt.show()
        predict_op = tf.argmax(p4, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y1, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X1: images, Y1: one_hot_targets})
        test_accuracy = accuracy.eval({X1: testimgs, Y1: one_hot_targets_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
