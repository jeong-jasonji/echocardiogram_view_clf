import streamlit as st

### START OF MY CODE TO RUN VIEW CLASSIFICATION ###

import os
import cv2
import random
import pydicom
import numpy as np
from tqdm import tqdm
from PIL import Image
import SimpleITK as sitk
import matplotlib.pyplot as plt

import sys
import torch
import pickle
import argparse
import pandas as pd
from sklearn.metrics import classification_report
from src import model_base, training_base, dataloader_base

import copy
import time
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import shutil

class makeRGB(object):
    """
    make a grayscale image RGB
    """

    def __init__(self):
        self.initalized = True

    def __call__(self, img):
        return img.convert('RGB')

class classifyEchos(object):
    """
    classify all of the echocardiogram files in the input directory 
    with the pretrained view classifier model and save those with 
    high confidence in separate view folders
    
    *for now the input dir only has single patient dicoms
    """
    def __init__(self, input_dir, output_dir, min_confidence=0.8, model_path='resnext101_best.pth', gpu=False, cuda_id=0, verbose_pred=False):
        self.initalized = True
        self.input_dir = input_dir  # save_dir = '/media/Datacenter_storage/CardiacEcho_rawdata/LVH_Oh/DCM_external_pull_full_view_test/'
        self.output_dir = output_dir  # external_base_dir = '/media/Datacenter_storage/CardiacEcho_rawdata/LVH_Oh/DCM_external_pull/'
        self.model_path = model_path  # view_clf_model_dir = '/home/jason/simpleCode/echocardiogram_viewCLF/full_view_test/resnext101.pth'
        self.min_confidence = min_confidence
        self.gpu = gpu
        self.cuda_id = cuda_id
        self.verbose_pred = verbose_pred
        self.label_dict = {0: 'AP2', 1:'AP3', 2:'AP4', 3:'PLAX', 4:'PSAX_M', 5:'PSAX_V'}
        
        self.params = {'workers': 4, 'num_classes': 5, 'ngpu': 1, 'batch_size': 16, 'feature_extract': False, 'load_best': False, 
                  'save_every_epoch': True, 'is_inception': False, 'use_pretrained': True, 'weights': [], 'fusion_model': False,
                  'aug_mode': 'trad', 'num_epochs': 50, 'learning_rate': 0.000001, 'centerCrop': 0, 'optim_metric': 'f1',
                  'loss_fx': 'focal', 'weight_decay': 0.15, 'weighted_sample': False, 'model_name': 'resnext101',
                 }
        
        # initialize the model and load the weights
        print('Initializing the view classification model: {}'.format(self.model_path))
        model_ft, input_size = model_base.initialize_model(params)
        # use cpu or gpu and load the model
        print('Loading the model:{}'.format(view_clf_model_dir), file=logger)
        if self.gpu:
            device = torch.device("cuda:{}".format(self.cuda_id) if (torch.cuda.is_available() and params['ngpu'] > 0) else "cpu")
            model_ft = torch.load(view_clf_model_dir)
        else:
            device = torch.device("cpu")
            model_ft = torch.load(view_clf_model_dir, map_location=device)
        # set model to eval
        model_ft.eval()
        
        # make a quick data transformer for each frame to give as an input to the model
        print('Loading Transforms')
        frame_transform = transforms.Compose([
            makeRGB(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.softmax = nn.Softmax()
        self.model = model_ft
        self.transform = frame_transform
        
    def standard_crop(self, frame):
        # adding a normalization step in the middle so that the ultrasound isn't as dark
        frame_h = int(frame.shape[0] * 0.1)
        frame_w = int(frame.shape[1] * 0.25)
        frame_crop = frame[frame_h:-frame_h, frame_w:-frame_w]
        
        return frame_crop
        
    def classify_file(self, f):
        """
        get the file and classify the view
        """
        try:
            img = sitk.ReadImage(os.path.join(self.input_dir, f))
        except:
            print('{}: Image reading error'.format(i), file=self.logger)
            return None, None
        # start classifying the read image
        img_array = sitk.GetArrayFromImage(img)
        num_frames = img_array.shape[0]
        if num_frames == 1: # if there's only one frame, skip it since it's a screen shot
            print('{}: ScreenShot (one image)'.format(f), file=self.logger)
            return None, None
        # choose the whole image to predict from
        idxs = list(range(img_array.shape[0]))
        # run classification eval on the model

        ### This is where nifti output should be ###
        preds = []
        pred_score_dict = {}
        for idx in tqdm(idxs, desc='{} Clf Progress'.format(f), leave=False):            
            frame = img_array[idx]
            frame = np.array(Image.fromarray(frame).convert('L'))
            inputs = self.transform(Image.fromarray(frame))
            outputs = self.model(torch.unsqueeze(inputs, dim=0))
            score = self.softmax(outputs)
            pred_score, pred = torch.max(softmax(outputs), 1)
            # add something about the predictions sum
            preds.append(np.array(pred))
            if pred.item() not in pred_score_dict.keys():
                pred_score_dict[pred.item()] = [pred_score.item()]
            else:
                pred_score_dict[pred.item()].append(pred_score.item())
        # make the prediction on all the frames
        vals, counts = np.unique(preds, return_counts=True)
        # find mode/most frequent prediction
        mode_index = np.argwhere(counts == np.max(counts))
        prediction_score = np.max(counts)/num_frames
        if len(mode_index) != 1:
            print('{}: Predicted multiple'.format(f), file=self.logger)
            return None, None
        mode_value = vals[int(mode_index)]
        mode_pred = np.mean(pred_score_dict[int(mode_value)])
        predicted_view = self.label_dict[int(mode_value)]
        print('{}: Prediction: {} ({:.2f}% pred_slices, {:.2f}% avg_pred_score)'.format(f, predicted_view, prediction_score*100, mode_pred*100), file=self.logger)
        if self.verbose_pred:
            for i in pred_score_dict.keys():
                print('\tlabel:{}:{} count:{} score:{}'.format(self.label_dict[int(i)], i, len(pred_score_dict[int(i)]), np.mean(pred_score_dict[int(i)])), file=self.logger)
        self.logger.flush()
        
        return mode_pred, predicted_view
    
    def run_view_clf(self):
        # make the output directory
        os.makedirs(self.output_dir, exist_ok=True)
        # set up the logger
        self.logger = open(os.path.join(output_dir, 'view_clf_log.txt'), 'a')
        # get file list from input dir
        files = os.listdir(self.input_dir)
        # go through all dicom files and classify them
        for f in tdqm(files, desc='File Progress'):
            # classify file
            predcited_score, predicted_view = self.classify_file(f)
            # save file if above min confidence
            if predcited_score >= self.min_confidence and predcited_score is not None:
                # create a folder structure for the output view
                os.makedirs(os.path.join(self.output_dir, predicted_view), exist_ok=True)
                # copy file to output view dir
                shutil.copyfile(os.path.join(self.input_dir, f), os.path.join(os.path.join(self.output_dir, predicted_view), f))
        self.logger.close()

### END OF MY CODE TO RUN VIEW CLASSIFICATION ###

### Streamlit ###
# page title
st.markdown('# EchoCardiogram View Classification')
# inputs
input_dir = st.text_input('Input Directory')
output_dir = st.text_input('Output Directory')
