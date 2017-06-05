# coding: utf-8

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
import os, sys
import argparse
import numpy as np
import cv2
from logging import getLogger, StreamHandler, INFO, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)

def main(args):
  try:
    logger.info("## main() start")
    logger.info("args = " + str(args))

    # default settings
    settings = {
        "image_x":160,
        "image_y":120,
        "nb_epoch":args.nb_epoch,
        "batch_size":args.batch_size,
        "learning_rate":args.learning_rate,
        "validation_split":0.1
    }

    # init
    train_path = args.train_path
    test_path = args.test_path
    weights_output_path = args.weights_output_path
    weights_input_path = args.weights_input_path
    weights_cache_path = args.weights_cache_path

    # create labels
    labels = create_labels(train_path)
    settings["label_num"] = len(labels)

    logger.info("## create_labels")
    logger.info(labels)

    logger.info("## settings")
    logger.info(settings)

    # create neural net 
    model = create_model(settings)

    # load weights
    if weights_input_path != None:
      logger.info("load weights = " + weights_input_path)
      model.load_weights(weights_input_path)
    
    if args.predict == False:
      # preprocess
      X_train, Y_train = preprocess(train_path, labels, settings)

      # training
      logger.info("## training exec")
      logger.info("X_train = " + str(X_train.shape))
      logger.info("Y_train = " + str(Y_train.shape))

      # callbacks
      checkpoint_template = os.path.join(weights_cache_path, "weights.{epoch:05d}.h5")
      callbacks= ModelCheckpoint(checkpoint_template)

      model = training(model, X_train, Y_train, callbacks, settings)
      
      # save weights
      if weights_output_path != None:
        logger.info("save weights = " + weights_output_path)
        model.save_weights(weights_output_path)
    else:
      logger.info("predict only")

    # accuracy
    X_test, Y_test = preprocess(test_path, labels, settings)
    score = predict(model, X_test, Y_test, settings)

    logger.info('Test loss     : ' + str(score[0]))
    logger.info('Test accuracy : ' + str(score[1]))
    
    logger.info("## main() end")
  except Exception as e:
    logger.error(str(type(e)))
    logger.error(str(e.args))
    logger.error(e.message)

def create_labels(images_path):
  labels = {}

  for dir in os.listdir(images_path):
    # create label
    if dir == "dog":
      label = 0
    elif dir == "cat":
      label = 1
    else:
      continue

    # add label
    logger.debug(dir)
    labels[dir] = label
  logger.debug("labels size = " + str(len(labels)))
  return labels

def preprocess(images_path, labels, settings):
  #nn_inputs = np.zeros((1, settings["image_x"] * settings["image_y"]))
  #nn_labels = np.zeros((1, 1))
  nn_inputs = []
  nn_labels = []
  i = 0

  for dir in os.listdir(images_path):
    base_path = images_path + "/" + dir

    if labels.has_key(dir) == False:
      continue

    for file in os.listdir(base_path):
      # get & update image
      logger.info("read image file .... " + base_path + file)
      image = cv2.imread(base_path + "/" + file)
      image = cv2.resize(image, (settings["image_x"], settings["image_y"]))
      logger.debug(image.shape)

      # add nn_input
      #nn_input = np.array(image, dtype=np.float32).flatten()
      nn_inputs.append(image)
      #logger.debug(nn_inputs.shape)

      # add nn_label
      #nn_labels.append(labels[dir])
      #label = np.array(labels, dtype=np.float32)
      nn_labels.append(labels[dir])

      i += 1

  x = np.array(nn_inputs)
  x = x.transpose((0, 3, 1, 2))
  y = np_utils.to_categorical(nn_labels,len(labels))

  logger.debug("i = " + str(i))
  logger.debug(x.shape)
  logger.debug(y.shape)
  return x, y

def create_model(settings):
  # init
  learning_rate = 0.001

  model = Sequential()
  model.add(Convolution2D(32, 3, 3, border_mode='same', activation='linear',
  input_shape=(3, settings["image_y"], settings["image_x"])))
  model.add(LeakyReLU(alpha=0.3))

  model.add(Convolution2D(32, 3, 3, border_mode='same', activation='linear'))
  model.add(LeakyReLU(alpha=0.3))

  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(Convolution2D(64, 3, 3, border_mode='same', activation='linear'))
  model.add(LeakyReLU(alpha=0.3))
  model.add(Convolution2D(64, 3, 3, border_mode='same', activation='linear'))
  model.add(LeakyReLU(alpha=0.3))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(Convolution2D(128, 3, 3, border_mode='same', activation='linear'))
  model.add(LeakyReLU(alpha=0.3))
  model.add(Convolution2D(128, 3, 3, border_mode='same', activation='linear'))
  model.add(LeakyReLU(alpha=0.3))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(Flatten())

  model.add(Dense(1024, activation='linear'))
  model.add(LeakyReLU(alpha=0.3))

  model.add(Dropout(0.5))
  model.add(Dense(1024, activation='linear'))
  model.add(LeakyReLU(alpha=0.3))
  model.add(Dropout(0.5))

  model.add(Dense(settings["label_num"]))
  model.add(Activation('softmax'))

  sgd = SGD(lr=settings['learning_rate'], decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd,
           loss='categorical_crossentropy', metrics=["accuracy"])
  return model

def training(model, nn_inputs, nn_labels, callbacks, settings):

  model.fit(nn_inputs, nn_labels, 
    nb_epoch=settings["nb_epoch"], 
    batch_size=settings["batch_size"],
    validation_split=settings["validation_split"],
    shuffle=True,
    callbacks=[callbacks])

  return model

def predict(model, nn_inputs, nn_labels, settings):
  score = model.evaluate(nn_inputs, nn_labels, verbose=0)

  return score

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument("-train_path", "-p", help="", default="./train_images")
  parser.add_argument("-test_path", "-t", help="", default="./test_images")
  parser.add_argument("-weights_input_path", "-w", help="", default=None)
  parser.add_argument("-weights_output_path", "-o", help="", default=None)
  parser.add_argument("-weights_cache_path", "-c", help="", default="./cache")
  parser.add_argument("-learning_rate", "-R", help="", type=float, default=.03)
  parser.add_argument("-nb_epoch", "-E", help="", type=int, default=10)
  parser.add_argument("-predict", "-P", help="", default=False, action="store_true")
  parser.add_argument("-batch_size", "-B", help="", type=int, default=16)

  args = parser.parse_args()
  main(args)