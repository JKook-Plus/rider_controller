# from ppadb.client import Client

from ppadb.client import Client
import time

import PIL.Image
from PIL import ImageDraw, ImageTk, Image
import numpy as np

import cv2
from assets.viewer import AndroidViewer
import assets.keycodes as keycodes

import asyncio
import threading

import os

# from tkinter import *

import matplotlib.pyplot as plt

# import tesserocr

# C:\Program Files\Tesseract-OCR\tessdata

import pytesseract

coords = []


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

android = AndroidViewer()


# Initial score will be 0
score = 0
score_img = None

public_frame = 0


# main loop constants
resize_size = 10
white = [[0,0,253], [255,10,255]]




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



# with open('config_text.txt', 'r') as content_file:
#     content = content_file.read()




def get_score(aaa):
    global score
    score = pytesseract.image_to_string(aaa, lang='1', config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789+').replace("\n", "").replace("\x0c", "")
    print("SCORE ----------------- ",score)
    # cv2.imwrite("attempts/1/attempt_1_-%s-_%s.png"%(text, ticker), score)
    with open("sample.txt", "a") as file_object:
        # Append 'hello' at the end of file
        file_object.write(score + "\n")


class score_getter(threading.Thread):

    def run(self):
        global score_img

        while_true_time = time.time()

        while True:

            if score_img is None:
                continue

            # print(score_img)
            _tmp = time.time()
            print("while_true_time =======", _tmp - while_true_time)
            while_true_time = _tmp
            get_score(score_img)

class Game:
    def game_loop(self, ticker):

        # ticker = 0

        # while True:
        frames = android.get_next_frames()

        if frames is None:
            return False

        else:
            global public_frame, score_img

            for frame in frames:
                ticker += 1
                public_frame = frame
                # 1440, 3040
                adds_removed = frame[0:(3040-200), 0:1440]

                hor = 400

                score = frame[580:(3040-2300), hor:(1440-hor)]

                # score = resizer(score, resize_size)




                imS = resizer(adds_removed, resize_size)
                # print(imS.shape[0], imS.shape[1])
                sc = cv2.cvtColor(imS, cv2.COLOR_BGR2RGB)
                percentages = []



                edges = cv2.Canny(imS,100,1000)

                cv2.imshow("Edges", resizer(edges, 200))


                white_perc = calcPercentage(edges)


                if white_perc < 0.6:
                    is_dead = 8
                    print("DEAD")
                else:
                    is_dead = 0

                edges_height, edges_width = edges.shape

                # print(edges_height, edges_width)

                # score = imS[60:284-205, 30:144-30]

                # score = cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)

                _ , score = masking(score, white, "bloop")

                # PIL_version_score = Image.fromarray(np.uint8(score)).convert('RGB')

                # score

                # score = (255-score)

                blur_amount = 11

                score_img = cv2.cvtColor(score, cv2.COLOR_BGR2RGB)

                score_img = cv2.GaussianBlur(score_img,(blur_amount,blur_amount),cv2.BORDER_DEFAULT)


                score_img = cv2.threshold(score_img,127,255,cv2.THRESH_BINARY)[1]



                # score = Image.frombytes('RGB', score.shape[:2], score, 'raw', 'BGR', 0, 0)




                # --psm 7, 8 and 9 work, 7 was picked at random

                # if ticker%10==0:
                #     get_score(score)

                # print("Text: %s\n      %s"%(text, pytesseract.image_to_string(score)))


                cv2.imshow('Phone Viewer', resizer(imS, 300))
                cv2.imshow("Score", score)
                cv2.waitKey(1)
                return True





    # print(android.resolution)




    # tot = {}
    # for key in tot_before:
    #     tot[key] = [[(np.multiply(tot_before[key][0][0], resize_size/100)).astype(int), (np.multiply(tot_before[key][0][1], resize_size/100)).astype(int)], tot_before[key][1]]


if __name__ == "__main__":



    # android.set_screen_power_mode(0)
    itt = 0
    fps_list = []

    project_dir = os.path.dirname(__file__)
    plot_style_filename = os.path.join(project_dir, 'assets\\mplstyles\\custom_1.mplstyle')

    score_getter().start()

    gme = Game()

    while itt <= 5000:



        s = time.perf_counter()




        suc = gme.game_loop(itt)
        if suc != False:
            itt = itt + 1
            elapsed = time.perf_counter() - s
            fps_list.append(1/elapsed)
            if itt%10 == 0:
                print(itt)
        # print(f"{__file__} executed in {elapsed:0.6f} seconds.")

    min_value = min(fps_list)
    max_value = max(fps_list)
    average = np.average(fps_list)
    median = np.median(fps_list)

    # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    plt.style.use(plot_style_filename)



    plt.axhline(y=min_value, linestyle="--", color="grey", label="Min: "+str(round(min_value)))
    plt.axhline(y=max_value, linestyle="-.", color="grey", label="Max: "+str(round(max_value)))
    plt.axhline(y=average, linestyle=":", color="blue", label="Mean: "+str(round(average)))
    plt.axhline(y=median, linestyle="--", color="blue", label="Median: "+str(round(median)))


    plt.plot(fps_list)


    plt.legend(loc=5, prop = {"size":10})
    plt.show()
