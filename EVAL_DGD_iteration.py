# Script for evaluating one iteration of DGD for learned 3D photoacoustic
# imaging. This is meant to be called from Matlab, where evaluation of the
# gradient is done with the k-wave toolbox.
#
# This is accompanying code for: Hauptmann et al., Model based learning for 
# accelerated, limited-view 3D photoacoustic tomography, 
# https://arxiv.org/abs/1708.09832
#
# written by Andreas Hauptmann, January 2018
# ==============================================================================



import tensorflow as tf
import sys   
import EVAL_DGD_Load as loadPAT
import h5py
import numpy

FLAGS = None


def deepnn(x,grad):
  """Definition of the network structure for one iteration of DGD
  """
  x_image = tf.reshape(x, [-1, 240,240,80, 1])
  g_image = tf.reshape(grad, [-1, 240,240,80, 1])
  
  W_convPrio1 = weight_variable([5, 5, 5, 1, 16])
  b_convPrio1 = bias_variable([16])
  h_convPrio1 = tf.nn.relu(conv3d(x_image, W_convPrio1) + b_convPrio1)

  W_convPrio2 = weight_variable([5, 5, 5, 16, 32])
  b_convPrio2 = bias_variable([32])
  h_convPrio2 = tf.nn.relu(conv3d(h_convPrio1, W_convPrio2) + b_convPrio2)
  
  W_convGrad1 = weight_variable([5, 5, 5, 1, 16])
  b_convGrad1 = bias_variable([16])
  h_convGrad1 = tf.nn.relu(conv3d(g_image, W_convGrad1) + b_convGrad1)

  W_convGrad2 = weight_variable([5, 5, 5, 16, 32])
  b_convGrad2 = bias_variable([32])
  h_convGrad2 = tf.nn.relu(conv3d(h_convGrad1, W_convGrad2) + b_convGrad2)

  h_sum       = tf.add(h_convPrio2,h_convGrad2)
  
  W_convDown1 = weight_variable([5, 5, 5, 32, 16])
  b_convDown1 = bias_variable([16])
  h_convDown1 = tf.nn.relu(conv3d(h_sum, W_convDown1) + b_convDown1)
  
  W_convDown2 = weight_variable([5, 5, 5, 16, 1])
  b_convDown2 = bias_variable([1])
  
  lambdaFac = step_length()
  
  h_convDown2 = tf.scalar_mul(lambdaFac,conv3d(h_convDown1, W_convDown2) + b_convDown2)

  h_update = tf.nn.relu(tf.add(h_convDown2,x_image))  
  h_update = tf.reshape(h_update, [-1, 240,240,80])

  return h_update


def conv3d(x, W):
  """conv3d returns a 3d convolution layer with full stride."""
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def step_length():
  """Step length before update for a given shape."""
  initial = tf.constant(1.0)
  return tf.Variable(initial)



def main(filePath,fileOutName,testFileName):

  dataPAT = loadPAT.read_data_sets(testFileName)
  imSize=dataPAT.test.images.shape
  
  # Init placeholder variables
  imag = tf.placeholder(tf.float32, [None, 240,240,80])
  grad = tf.placeholder(tf.float32, [None, 240,240,80])
  output = tf.placeholder(tf.float32, [None, 240,240,80])
  
  # Build the graph for the deep net
  y_conv = deepnn(imag,grad)

  saver = tf.train.Saver()


  with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, filePath)
    print("Model restored.")

    start=0
    end=imSize[0]

    #evaluation and savinf for processing in Matlab      
    output = sess.run(y_conv,feed_dict={
         imag: dataPAT.test.images[start:end], grad: dataPAT.test.grad[start:end]})
      
    fData = h5py.File(fileOutName,'w')
    fData['result']= numpy.array(output)
    fData.close()


    print('--------------------> DONE <--------------------')
    
    return

#Call main function with arguements
main(sys.argv[1],sys.argv[2],sys.argv[3])