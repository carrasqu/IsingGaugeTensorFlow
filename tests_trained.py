import tensorflow as tf
import input_data
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# lattice definitions. Neighbours, plaquettes, and vertices.
def plqu(lx):
  k=0
  ly=lx
  nh=2*lx*ly
  neig=np.zeros((lx*ly,4),dtype=np.int)
  for j in range(ly):
    for i in range(lx):
       if i<lx-1:
         neig[k,0]=k+1
         if j<ly-1:
          neig[k,1]=k+lx
         elif j==ly-1:
          neig[k,1]=k-(ly-1)*lx
         if i==0:
           neig[k,2]=k+lx-1
         else:
           neig[k,2]=k-1
       elif i==lx-1:
         neig[k,0]=k-(lx-1)
         if j<ly-1:
          neig[k,1]=k+lx
         elif j==ly-1:
          neig[k,1]=k-(ly-1)*lx
         neig[k,2]=k-1
       if j==0:
         neig[k,3]=k+(ly-1)*lx
       else:
         neig[k,3]=k-lx
       k=k+1

  plaquette=np.zeros((lx*ly,4),dtype=np.int)
  vertex=np.zeros((lx*ly,4),dtype=np.int)
  for i in range(ly*lx):
    plaquette[i,0]=2*i
    plaquette[i,1]=2*i+1
    plaquette[i,2]=2*neig[i,0]+1
    plaquette[i,3]=2*neig[i,1]
    vertex[i,0]=2*i
    vertex[i,1]=2*i+1
    vertex[i,2]=2*neig[i,2]
    vertex[i,3]=2*neig[i,3]+1
    #print "p", i, plaquette[i,0],  plaquette[i,1], plaquette[i,2], plaquette[i,3]
    #print "v", i, vertex[i,0], vertex[i,1], vertex[i,2], vertex[i,3]
  return neig,plaquette,vertex

lx=16 # linear size of the system
numberlabels=2 # number of labels
mnist = input_data.read_data_sets(numberlabels,lx+1,'txt', one_hot=True) # importing the training and test sets. lx+1 to account for the boundary conditions (periodic)
print "reading sets ok"


# defining the convolutional and max pool layers
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

# defining the model

x = tf.placeholder("float", shape=[None, (lx+1)*(lx+1)*2]) # placeholder for the spin configurations
y_ = tf.placeholder("float", shape=[None, numberlabels]) # place holder for the labels

#first layer 
# convolutional layer # 2x2 patch size, 2 channel (2 sublattices), 16 feature maps computed. "Valid option will take care of the lx*lx plaquettes"

# convolutional filters defined analytically

                 #height(y)   # width(x)    # channels(sublattices)  # number of filters (16 in our model)
nmaps1=16
filt=np.zeros( ( 2,           2,            2,                       nmaps1       )    )
sg=np.array(           [ \
[  1,     1,     1,     1 ],\
[ -1,    -1,    -1,    -1 ],\
[  1,     1,    -1,    -1 ],\
[  1,    -1,    -1,     1 ],\
[ -1,    -1,     1,     1 ],\
[ -1,     1,     1,    -1 ],\
[  1,    -1,     1,    -1 ],\
[ -1,     1,    -1,     1 ],\
[  1,     1,     1,    -1 ],\
[  1,     1,    -1,     1 ],\
[  1,    -1,     1,     1 ],\
[ -1,     1,     1,     1 ],\
[ -1,    -1,    -1,     1 ],\
[ -1,    -1,     1,    -1 ],\
[ -1,     1,    -1,    -1 ],\
[  1,    -1,    -1,    -1 ]])
# spreading the plaquette filters around the two sublattice filters
for i in range(nmaps1):
 filt[0,0,0,i]=sg[i,0]
 filt[1,0,0,i]=sg[i,1]
 filt[0,0,1,i]=sg[i,2]
 filt[0,1,1,i]=sg[i,3]

# bias vector defined analitically 
bf=np.zeros(nmaps1)
eps=0.01
bf[:]=-(2+eps)

W_conv1 = tf.convert_to_tensor(filt, dtype=tf.float32) # transforming weights and bias vector to tensorflow variables 
b_conv1=tf.convert_to_tensor(bf, dtype=tf.float32)

# applying a reshape of the data to get the two dimensional structure back
x_image = tf.reshape(x, [-1,(lx+1),(lx+1),2])

#We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
beta=1000000.0 # beta makes the sigmoid behave like a perceptron
brrrr=conv2d(x_image, W_conv1) + b_conv1
h_conv1 = tf.nn.sigmoid(beta*(conv2d(x_image, W_conv1) + b_conv1))


# weights and bias of the fully connected (fc) layer. Ihn this case everything looks one dimensiona because it is fully connected
nmaps2=2

# This is the fully connected weight matrix
Wlast=np.ones(((lx) * (lx) * nmaps1,nmaps2)) # all ones at the begining
Wlast[8*lx*lx:,0]=-(lx*lx-lx+1) # # for the ground state neuron (0), if a single violated plaquette appears, then we get negative values before the perceptron
Wlast[0:8*lx*lx,1]=-1   # for the high temperature neuron(1), we take negative values of satisfied plaquettes 
Wlast[8*lx*lx:,1]=lx*lx-lx+1 # for the negative neuron (1), if a single vortex plaquette is present, we should consider the state high temperature  and the values should be positive before the precentrop

bf=np.zeros(nmaps2) # bias is chosen to be zero in this case
# transforming to tensorflow variables
b_fc1=tf.convert_to_tensor(bf, dtype=tf.float32) 
W_fc1=tf.convert_to_tensor(Wlast, dtype=tf.float32)

### reshaping the outout of the convolutional layer to a flat array to be fed to the fully-connected layer
h_conv1_flat=tf.transpose(h_conv1, perm=[0, 3, 2, 1]) # rearranges everything so that the first elements of h_conv1_flat correspond to images processed 
#by first filters first and last images processed by last filrters after the reshape
h_conv1_flat = tf.reshape(h_conv1_flat, [-1, (lx)*(lx)*nmaps1]) 

# then apply the perceptron with the fully connected weights and biases.
y_conv= tf.nn.sigmoid(beta*(tf.matmul(h_conv1_flat, W_fc1) + b_fc1))

#Evaluating the Model

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


#launch session
sess = tf.Session()
sess.run(tf.initialize_all_variables())



################## useful tests for debugging######
#ii=0
#batch=(mnist.test.images[ii,:].reshape((1,2*(lx+1)*(lx+1))),mnist.test.labels[ii,:].reshape((1,numberlabels)))
#ver=sess.run(h_conv1,feed_dict={x:batch[0]})
#iver=sess.run(brrrr,feed_dict={x:batch[0]})
#ver2=sess.run(h_conv1_flat,feed_dict={x:batch[0]})
#plt.plot(ver2[0,:])
#plt.savefig('check.pdf')
#sys.exit("bye")
#ver2=np.reshape(ver2,(1,16*lx*lx))
#batch=(np.reshape(np.arange(2*(lx+1)*(lx+1)),(1,2*(lx+1)*(lx+1))),mnist.test.labels[ii,:].reshape((1,numberlabels)))
#res=sess.run(x_image,feed_dict={x: batch[0]})
###########################################


# evaluating test accuracy
print "test accuracy %g"%sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels})


# output  neuron over the test set
f = open('nnout.dat', 'w')

Ntemp=2
samples_per_T_test=2500
ii=0
for i in range(Ntemp):
  av=0.0
  for j in range(samples_per_T_test):
        batch=(mnist.test.images[ii,:].reshape((1,2*(lx+1)*(lx+1))),mnist.test.labels[ii,:].reshape((1,numberlabels)))
        res=sess.run(y_conv,feed_dict={x: batch[0], y_: batch[1]})
        av=av+res
        #print ii, res
        ii=ii+1
  av=av/samples_per_T_test
  f.write(str(i)+' '+str(av[0,0])+' '+str(av[0,1])+"\n")
f.close()

# accuracy vs "temperature"
f = open('acc.dat', 'w')

# accuracy vs temperature
for ii in range(Ntemp):
  batch=(mnist.test.images[ii*samples_per_T_test:ii*samples_per_T_test+samples_per_T_test,:].reshape(samples_per_T_test,2*(lx+1)*(lx+1)), mnist.test.labels[ii*samples_per_T_test:ii*samples_per_T_test+samples_per_T_test,:].reshape((samples_per_T_test,numberlabels)) )
  train_accuracy = sess.run(accuracy,feed_dict={
        x:batch[0], y_: batch[1]})
  f.write(str(ii)+' '+str(train_accuracy)+"\n")
f.close()


#plotting a ground state configuration
#ii=0
#batch=(mnist.test.images[ii,:].reshape((1,2*(lx+1)*(lx+1))),mnist.test.labels[ii,:].reshape((1,numberlabels)))

#tccon=batch[0]
#Nx=lx+1
#Ny=Nx

#nh=Nx*Ny*2
#config=np.zeros(nh)

#k=0
#for i in np.arange(0,Ny,1):
#  for j in np.arange(0,Nx,1):

#    config[k]=(tccon[0,k])
#    k=k+1
#    config[k]=(tccon[0,k])
#    k=k+1

#k=0
#x=np.zeros(nh)
#y=np.zeros(nh)
#for i in np.arange(0,Ny,1):
# for j in np.arange(0,Nx,1):
#  x[k]=j+0.5
#  y[k]=i
#  k=k+1
#  x[k]=j
#  y[k]=i+0.5
#  k=k+1


#fig, ax = plt.subplots()
#im = ax.scatter(x, y, c=config,s=40, cmap=plt.cm.BuPu)
#plt.grid()
#plt.ylim([-0.5,Ny+0.5])
#plt.xlim([-0.5,Nx+0.5])
#fig.colorbar(im, ax=ax)
#plt.savefig('confMAX.pdf')

