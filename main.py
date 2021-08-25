# from ppadb.client import Client

from ppadb.client import Client
import time

import PIL.Image
from PIL import ImageDraw, ImageTk, Image
import numpy as np

import cv2
from assets.viewer import AndroidViewer
import assets.keycodes as keycodes


# from tkinter import *

import matplotlib.pyplot as plt

# import tesserocr

# C:\Program Files\Tesseract-OCR\tessdata

import pytesseract

coords = []


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



def connect_device():
    adb = Client(host='127.0.0.1',port=5037)
    devices = adb.devices()
    if len(devices) == 0:
        print("No Devices Attached")
        quit()
    return devices[0]


def resizer(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    dsize = (width, height)
    imS = cv2.resize(image, dsize, interpolation=cv2.INTER_NEAREST )
    return imS


def masking(view, ra, name):
    lower = np.array(ra[0])
    upper = np.array(ra[1])
    hsv = cv2.cvtColor(view, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    view_height, view_width, _ = view.shape

    # cv2.imshow(name, mask)

    # cv2.waitKey(1)
    return(round(cv2.countNonZero(mask)/mask.size*100,0), mask)

def format_masks(tot_before, resize_size):
    tot = {}
    for key in tot_before:
        tot[key] = [[(np.multiply(tot_before[key][0][0], resize_size/100)).astype(int), (np.multiply(tot_before[key][0][1], resize_size/100)).astype(int)], tot_before[key][1]]


    return tot


def calcPercentage(msk):
	'''
	returns the percentage of white in a binary image
	'''
	height, width = msk.shape[:2]
	num_pixels = height * width
	count_white = cv2.countNonZero(msk)
	percent_white = (count_white/num_pixels) * 100
	percent_white = round(percent_white,2)
	return (percent_white)

def close_event():
    plt.close()

android = AndroidViewer()

ep = []

is_dead_list = []

with open('config_text.txt', 'r') as content_file:
    content = content_file.read()



def video_stream():

    resize_size = 10

    frames = android.get_next_frames()

    # print(android.resolution)

    all_color = [[0,0,0], [255,255,255]]

    purple = [[135,25,200], [150,50,255]]
    green = [[65,25,100], [70,35,255]]
    red = [[175,100,0], [180,255,255]]
    pink = [[150,25,200], [160,50,255]]
    yellow = [[20,0,200], [30,50,255]]
    cyan = [[80,0,100], [90,50,255]]
    yellow_2 = [[25,25,100], [40,50,255]]
    green_2 = [[50,0,0], [60,100,255]]
    white = [[0,0,253], [255,10,255]]

    wheels = [[0,0,225], [150,10,255]]

    # 1440, 3040

    ground_before = {
    "Ground":
                [[[0, 3040], [0, 1440]],
                purple],
    # "Ground 2":
    #             [[[0, 3040], [0, 1440]],
    #             pink],
    # "Ground 3":
    #             [[[0, 3040], [0, 1440]],
    #             cyan],
    # "Ground 4":
    #             [[[0, 3040], [0, 1440]],
    #             green_2],
                }

    gam_before = {
    "Gems and Movables":
                [[[0, 3040], [0, 1440]],
                green],
    # "Gems and Movables 2":
    #             [[[0, 3040], [0, 1440]],
    #             yellow],
    # "Gems and Movables 3":
    #             [[[0, 3040], [0, 1440]],
    #             yellow_2],
    }



    tot_before = {
    "Ground":
                [[[0, 3040], [0, 1440]],
                purple],
    "Gems and Movables":
                [[[0, 3040], [0, 1440]],
                green],
    "Wheels":
                [[[0, 3040], [0, 1440]],
                wheels],
    "Dangers":
                [[[0, 3040], [0, 1440]],
                red],
    # "Ground 2":
    #             [[[0, 3040], [0, 1440]],
    #             pink],
    "Gems and Movables 2":
                [[[0, 3040], [0, 1440]],
                yellow],
    # "Ground 3":
    #             [[[0, 3040], [0, 1440]],
    #             cyan],
    "Gems and Movables 3":
                [[[0, 3040], [0, 1440]],
                yellow_2],
    # "Ground 4":
    #             [[[0, 3040], [0, 1440]],
    #             green_2],
                }


    # ground = format_masks(ground_before, resize_size)



    # tot = {}
    # for key in tot_before:
    #     tot[key] = [[(np.multiply(tot_before[key][0][0], resize_size/100)).astype(int), (np.multiply(tot_before[key][0][1], resize_size/100)).astype(int)], tot_before[key][1]]

    if frames is None:
        pass

    else:

        for frame in frames:

            # 1440, 3040
            adds_removed = frame[0:(3040-200), 0:1440]

            hor = 400

            score = frame[600:(3040-2300), hor:(1440-hor)]

            # score = resizer(score, resize_size)




            imS = resizer(adds_removed, resize_size)
            # print(imS.shape[0], imS.shape[1])
            sc = cv2.cvtColor(imS, cv2.COLOR_BGR2RGB)
            percentages = []

            ground_mask_holder = []

            # for key in ground:
            #     x, y = ground[key][0]
            #     color_r = ground[key][1]
            #     selection = sc[ x[0]:x[1], y[0]:y[1] ]
            #
            #
            #     _ , du = masking(selection, color_r, key)
            #     ground_mask_holder.append(du)




            # print(type(mask_holder[0]))
            # g_1 = cv2.add(mask_holder[0], mask_holder[1])
            # # print(g_1)
            # cv2.imshow("G_1", g_1)
# mask_holder[0], mask_holder[1], mask_holder[4], mask_holder[5], mask_holder[6], mask_holder[7]

            g_2 = 0
            for i in ground_mask_holder:
                g_2 = cv2.add(g_2, i)

            edges = cv2.Canny(imS,100,1000)

            cv2.imshow("Edges", resizer(edges, 200))


            white_perc = calcPercentage(edges)



            if white_perc < 0.6:
                is_dead = 8
                print("DEAD")
            else:
                is_dead = 0

            edges_height, edges_width = edges.shape

            # 284 144

            # print(edges_height, edges_width)

            # score = imS[60:284-205, 30:144-30]

            # score = cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)

            _ , score = masking(score, white, "bloop")

            # PIL_version_score = Image.fromarray(np.uint8(score)).convert('RGB')

            # score




            # text = pytesseract.image_to_string(score, lang='eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789+')
            # print(text)

            # ep.append(white_perc)
            # is_dead_list.append(is_dead)


            # if len(ep)%500 == 0:
            #     # fig, (ax1, ax2) = plt.subplots(2)
            #     # fig.suptitle('Is Dead checks')
            #     plt.plot(ep)
            #     plt.plot(is_dead_list)
            #
            #     # plt.plot(ep)
            #
            #     plt.show()
            #     # ep = []
            #     # is_dead_list = []


            # g_2 = cv2.add(cv2.add((cv2.add(mask_holder[0], mask_holder[1])), (cv2.add(mask_holder[4], mask_holder[5]))), (cv2.add(mask_holder[6], mask_holder[7])))
            # # g_2 = g_2.clip(0,255).astype("uint8")

            # cv2.imshow("G_2", g_2)


            cv2.imshow('Phone Viewer', resizer(imS, 300))
            cv2.imshow("Score", resizer(score, 100))
            cv2.waitKey(1)


    # lmain.after(1, video_stream)


while True:
    video_stream()
