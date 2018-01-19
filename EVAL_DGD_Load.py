# Loading and shaping of data files for learned iterative reconstruction of
# 3D photoacoustic data.
#
# This is accompanying code for: Hauptmann et al., Model based learning for 
# accelerated, limited-view 3D photoacoustic tomography, 
# https://arxiv.org/abs/1708.09832
#
# written by Andreas Hauptmann, January 2018
# ==============================================================================

import numpy
import h5py


def extract_images(filename,imageName):
  """Extract the images into a 4D uint8 numpy array [index, y, x, z]."""
  fData = h5py.File(filename,'r')
  inData = fData.get(imageName)  
      
  if inData.shape[0] == inData.shape[1]:
  
    num_images = int(1)
    rows = inData.shape[0]
    cols = inData.shape[1]
    deps = inData.shape[2]
  
  else:
    num_images = inData.shape[0]
    rows = inData.shape[1]
    cols = inData.shape[2]
    deps = inData.shape[3]
  
  print(num_images, rows, cols,deps)
  data = numpy.array(inData)
    
  data = data.reshape(num_images, rows, cols, deps, 1)
  return data


class DataSet(object):

  def __init__(self, images, grad):
    """Construct a DataSet"""

    assert images.shape[0] == grad.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape,
                                                 grad.shape))
    self._num_examples = images.shape[0]

    assert images.shape[4] == 1
    images = images.reshape(images.shape[0],
                            images.shape[1],images.shape[2],images.shape[3])
    grad = grad.reshape(grad.shape[0],
                            grad.shape[1],grad.shape[2],grad.shape[3])
    self._images = images
    self._grad = grad

  @property
  def images(self):
    return self._images

  @property
  def grad(self):
    return self._grad

  @property
  def num_examples(self):
    return self._num_examples

def read_data_sets(testFileName):
  class DataSets(object):
    pass
  data_sets = DataSets()

  TEST_SET  = testFileName
  IMAGE_NAME = 'imag'
  GRAD_NAME  = 'grad'
  
  print('Start loading data')  
  test_images = extract_images(TEST_SET,IMAGE_NAME)
  test_grad   = extract_images(TEST_SET,GRAD_NAME)

  data_sets.test = DataSet(test_images, test_grad)

  return data_sets