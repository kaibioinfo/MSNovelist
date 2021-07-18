# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:58:01 2018

@author: stravsmi
"""


from tqdm import tqdm

from fp_management import database as db
from fp_management import fingerprinting as fpr
from fp_management import fingerprint_map as fpm
from rdkit import Chem

import smiles_process as sp
import importlib
from importlib import reload
import smiles_config as sc

import infrastructure.generator as gen

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, \
    LambdaCallback, Callback

import numpy as np
import pandas as pd
import time
import math
import os
import pickle



# Setup logger
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("MSNovelist")
logger.setLevel(logging.INFO)
logger.info("training startup")

sampler_name = ""
if sampler_name != '':
    spl = importlib.import_module(sampler_name, 'fp_sampling')
#import models.quicktrain_fw_20190327 as sm


pipeline_x = sc.config['pipeline_x']
pipeline_y = sc.config['pipeline_y']
logger.info(f"pipeline_x: {pipeline_x}")
logger.info(f"pipeline_y: {pipeline_y}")

training_id = str(int(time.time()))
if sc.config['training_id'] != '':
    training_id = sc.config['training_id']

sc.config.setdefault('cv_fold', 0)
cv_fold = "X"#sc.config["cv_fold"]
training_set = f"fold[^{cv_fold}]"
validation_set = 'fold0'
if cv_fold != 'X':
    validation_set = f"fold{cv_fold}"


logger.info(f"Training model id {training_id}, fold {cv_fold}")

model_tag_id = "m-" + training_id + "-" + sc.config['model_tag']
logger.info(f"Tag: {model_tag_id}")

weights_path = os.path.join(
    sc.config["weights_folder"],
    model_tag_id,
    str(cv_fold))
log_path = os.path.join(
    sc.config['log_folder'],
    model_tag_id,
    str(cv_fold))

config_dump_path = os.path.join(
    weights_path,
    'config.yaml'
    )

os.makedirs(weights_path)
os.makedirs(log_path)

sc.config_dump(config_dump_path)



# Load mapping table for the CSI:FingerID predictors

logger.info(f"Datasets - loading database")
fp_train = gen.hdf5Generator(path.join(sc.config["db_path"], "msnovelist_data.h5"), sampling=1)
fp_val = gen.hdf5Generator(path.join(sc.config["db_path"], "train.h5"))
fp_indep = gen.hdf5Generator(path.join(sc.config["db_path"], "eval.h5"))
logger.info(f"Datasets - pipelines built")

fp_dataset_train = fp_train.batch(sc.config['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)

blueprints = gen.dataset_blueprint(fp_dataset_train)

fp_dataset_val = gen.dataset_zip(fp_val, pipeline_x, pipeline_y,
                                 **fp_db.get_pipeline_options())
fp_dataset_val = fp_dataset_val.prefetch(tf.data.experimental.AUTOTUNE)

fp_dataset_eval = gen.dataset_zip(fp_indep, pipeline_x, pipeline_y,
                                  **db_eval.get_pipeline_options())
fp_dataset_eval = fp_dataset_eval.prefetch(tf.data.experimental.AUTOTUNE)

training_total = len(fp_train)
validation_total= len(fp_val)
training_steps = math.floor(training_total /  sc.config['batch_size'])
if sc.config['steps_per_epoch'] > 0:
    training_steps = sc.config['steps_per_epoch']

validation_steps = math.floor(validation_total /  sc.config['batch_size'])
if sc.config['steps_per_epoch_validation'] > 0:
    validation_steps = sc.config['steps_per_epoch_validation']
    
batch_size = sc.config["batch_size"]
epochs=sc.config['epochs']

logger.info(f"Preparing training: {epochs} epochs, {training_steps} steps per epoch, batch size {batch_size}")


round_fingerprints = False
if sampler_name != '':
    round_fingerprints = sf.round_fingerprint_inference()

import model
transcoder_model = model.TranscoderModel(
    blueprints = blueprints,
    config = sc.config,
    round_fingerprints = round_fingerprints
    )

initial_epoch = 0


logger.info("Building model")

transcoder_model.compile()
#
# If set correspondingly: load weights and continue training
if 'continue_training_epoch' in sc.config: 
    if sc.config['continue_training_epoch'] > 0:
        transcoder_model.load_weights(os.path.join(
            sc.config['weights_folder'],
            sc.config['weights']))
        transcoder_model._make_train_function()
        with open(os.path.join(
                sc.config['weights_folder'],
                sc.config['weights_optimizer']), 'rb') as f:
            weight_values = pickle.load(f)
        transcoder_model.optimizer.set_weights(weight_values)
        initial_epoch = sc.config['continue_training_epoch']


logger.info("Model built")
# {eval_loss:.3f}
filepath= os.path.join(
    weights_path,
    "w-{epoch:02d}-{loss:.3f}-{val_loss:.3f}.hdf5"
    )


tensorflow_trace = sc.config["tensorflow_trace"]
if tensorflow_trace:
    tensorboard_profile_batch = 2
else:
    tensorboard_profile_batch = 0
verbose = sc.config["training_verbose"]

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                             save_best_only=True, mode='min', 
                             save_weights_only=True)
tensorboard = TensorBoard(log_dir=log_path, 
                          histogram_freq=1,  
                          profile_batch = tensorboard_profile_batch,
                          write_graph=tensorflow_trace,
                          write_images=tensorflow_trace)

save_optimizer = model.resources.SaveOptimizerCallback(weights_path)
evaluation = model.resources.AdditionalValidationSet(fp_dataset_eval, 
                                                     "eval", 
                                                     verbose = 0)

print_logs = LambdaCallback(
    on_epoch_end = lambda epoch, logs: print(logs)
    )


#

callbacks_list = [evaluation, 
                  tensorboard, 
                  print_logs, 
                  checkpoint, 
                  save_optimizer]

logger.info("Training - start")
transcoder_model.fit(x=fp_dataset_train, 
          epochs=epochs, 
          #batch_size=sc.config['batch_size'],
          steps_per_epoch=training_steps,
          callbacks = callbacks_list,
          validation_data = fp_dataset_val,
          validation_steps = validation_steps,
          initial_epoch = initial_epoch,
          verbose = verbose)
logger.info("Training - done")
fp_db.close()

logger.info("training end")

