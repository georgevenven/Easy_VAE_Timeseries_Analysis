import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from PIL import Image
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import io
import librosa
import matplotlib
import gc
from matplotlib import pyplot as plt

# wav to numpy array
def wav_to_numpy(file):
    rate, data = wavfile.read(file)
    return rate, data

def createSonogram(songfile):
    data = wav_to_numpy(songfile)
    rate = data[0]
    data = data[1]

    f, t, Sxx = signal.spectrogram(data, rate, nfft=512, noverlap=25, scaling="density")

    fmin = 400 # Hz
    fmax = 8000 # Hz
    freq_slice = np.where((f >= fmin) & (f <= fmax))

    # keep only frequencies of interest
    f   = f[freq_slice]
    Sxx = Sxx[freq_slice,:][0]

    Sxx = np.log(Sxx + 1)

    np_sonogram = np.array(Sxx, dtype="float32")
    # figure size in inches 1,1
    # plt.figure(figsize=(300,5))

    # plt.pcolormesh(Sxx)
    # plt.axis('off')
    return plt, np_sonogram 

# window slides over vertical sum, takes the average and keeps it if it is above threshold
def sliding_window_average(vertical_sum, window_size, threshold):
    window = np.ones(int(window_size))/float(window_size)
    y = np.convolve(vertical_sum, window, 'same')
    y = np.where(y > threshold, y, 0)
    return y

def create_segments(sonogram, overlap, segment_length):
    # ~230 
    window_length = segment_length
    # 90 precent overlap 
    window_overlap = overlap

    # step size is equal to the amount of pixels the frame needs to move
    # this is equal to the frame size multiplied by the overlap
    # the frame size needs to be adjusted to the amount of pixels per ms
    step_size = int(window_length * (1 - window_overlap))
    start_frame = 0
    end_frame = int(window_length)
    
    positions = []

    sonogram = np.swapaxes(sonogram, 0, 1)

    while end_frame < sonogram.shape[0]:
        positions.append([start_frame, end_frame])
        start_frame += step_size
        end_frame += step_size
    return positions



class DataProcessor():
    def __init__(self, root_dir, dir, input_type, remove_empty_space, empty_window_size, empty_threshold, segment_length, segment_overlap, segment_size):
        self.root_dir = root_dir
        self.dir = dir
        self.input_type = input_type
        self.remove_empty_space = remove_empty_space
        self.empty_window_size = empty_window_size
        self.empty_threshold = empty_threshold
        self.segment_length = segment_length # 11 is 50 ms 
        self.segment_overlap = segment_overlap # .07 
        self.segment_size = segment_size

        self.prepare_data()

    def raw_audio_file_to_segments(self):
        if self.root_dir + 'segmentDirs' == None:
            # creates a parallel dir in the VAE web app dir 
            os.mkdir(self.root_dir + 'segmentDirs')
            for folder in os.listdir(self.dir):
                os.mkdir(self.root_dir + 'segmentDirs/' + folder)
                for subfolder in os.listdir(self.dir + '/' + folder):
                    os.mkdir(self.root_dir + 'segmentDirs/' + folder + '/' + subfolder)
        else:
            # delete all files and folders in segmentDirs
            # then create new folders just like done above
            for folder in os.listdir(self.rootDir + '/segmentDirs'):
                for subfolder in os.listdir(self.rootDir + '/segmentDirs/' + folder):
                    for file in os.listdir(self.rootDir + '/segmentDirs/' + folder + '/' + subfolder):
                        os.remove(self.rootDir + '/segmentDirs/' + folder + '/' + subfolder + '/' + file)
                    os.rmdir(self.rootDir + '/segmentDirs/' + folder + '/' + subfolder)
                os.rmdir(self.rootDir + '/segmentDirs/' + folder)

            for folder in os.listdir(self.dir):
                os.mkdir(self.root_dir + 'segmentDirs/' + folder)
                for subfolder in os.listdir(self.dir + '/' + folder):
                    os.mkdir(self.root_dir + 'segmentDirs/' + folder + '/' + subfolder)

        # # here we are goint to loop through each of the folders and create sonograms into the segmentDirs
        # for subject in os.listdir(self.dir):
        #     for condition in os.listdir(subject):
        #         for date_point in os.listdir(condition):
        #             list_of_np_sonograms = []
        #             for file in os.listdir(date_point):
        #                 try:
        #                     plt, np_sonogram = createSonogram(file)
        #                     np_sonogram = np_sonogram.T
        #                     vertical_sum = np.sum(np_sonogram, axis=1)
        #                     y = sliding_window_average(vertical_sum, self.empty_window_size, self.empty_threshold)
        #                     indices = np.where(y > 0)[0]
        #                     list_of_np_sonograms.append(np_sonogram[indices].T)

        #                 except:
        #                     print('error with file: ', file)


        #             list_of_positions = []
        #             for sonograms in list_of_np_sonograms:
        #                 list_of_positions.append(create_segments(sonograms, self.segment_overlap, self.segment_length))

        #             for i, sonogram in enumerate(list_of_np_sonograms):
        #                 print(f"sonogram number: {i}")
        #                 sonogram = np.swapaxes(sonogram, 0, 1)
        #                 for j, position in enumerate(list_of_positions[i]):
        #                     segment = sonogram[position[0]:position[1]]
        #                     segment = segment.T

        #                     size = self.segment_size / 100

        #                     plt.figure(figsize=(size,size))
        #                     plt.imshow(segment, aspect='auto')
        #                     plt.gca().set_axis_off()
        #                     plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
        #                                 hspace = 0, wspace = 0)
        #                     plt.margins(0,0)
        #                     plt.savefig('t_segments/test' + 'file:' + str(i) + '_segment:' + str(j) + '.png', dpi=100)
        #                     plt.clf()

    def sonogram_to_segments(self):
        pass
    
    def prepare_data(self): 
        if self.input_type == 'raw_audio_files':
            self.raw_audio_file_to_segments()
        elif self.input_type == 'sonograms':
            # extract (or not) and copy segments to segmentDir 
            pass
        elif self.input_type == 'segments':
            # copy segments to segmentDirs
            pass 