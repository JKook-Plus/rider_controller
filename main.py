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

from gym import Env
from gym.spaces import Discrete, Box

import random



coords = []


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

android = AndroidViewer()


# Initial score will be 0
score = 0
score_img = None

public_frame = 0

global_running = 1
global_is_dead = 80


# main loop constants
resize_size = 10
white = [[0,0,253], [255,10,255]]

plot_data = {
"FPS":[],
"img recog time":[],
"white percentage":[],
"score":[]}


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


def calcPercentage(img): # returns the percentage of white in a binary image
    dead_color = [224, 224, 244]

    diff = 100

    boundaries = [([dead_color[2], dead_color[1]-diff, dead_color[0]-diff],
           [dead_color[2]+diff, dead_color[1]+diff, dead_color[0]+diff])]


    # msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
    # hsv = cv2.cvtColor(msk, cv2.COLOR_BGR2HSV_FULL)
    # msk = cv2.inRange(hsv, np.array(dead_color), np.array(dead_color))


    ######################
    # height, width = msk.shape[:2]
    # num_pixels = height * width
    #
    # count_white = cv2.countNonZero(msk)
    # percent_white = (count_white/num_pixels) * 100
    # percent_white = round(percent_white,2)
    # return (percent_white)
    ######################
    # for (lower, upper) in boundaries:

    # You get the lower and upper part of the interval:
    # The HSV mask values, defined for the green color:
    lowerValues = np.array([dead_color[0]-diff, dead_color[1]-diff, dead_color[2]-diff])
    upperValues = np.array([255, 255, 255])

    # Convert the image to HSV:
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create the HSV mask
    hsvMask = cv2.inRange(hsvImage, lowerValues, upperValues)

    # AND mask & input image:
    hsvOutput = cv2.bitwise_and(img, img, mask=hsvMask)

    # Check out the ANDed mask:
    # cv2.imshow("ANDed mask", output)
    # cv2.waitKey(0)

    # You can use the mask to count the number of white pixels.
    # Remember that the white pixels in the mask are those that
    # fall in your defined range, that is, every white pixel corresponds
    # to a green pixel. Divide by the image size and you got the
    # percentage of green pixels in the original image:

    cv2.imshow("hsvMask", hsvMask)
    cv2.waitKey(1)


    ratio = cv2.countNonZero(hsvMask)/(img.size/3)

    # This is the color percent calculation, considering the resize I did earlier.
    colorPercent = (ratio * 100)

    # Print the color percent, use 2 figures past the decimal point
    # print('white pixel percentage:', np.round(colorPercent, 2))

    hue, sat, val = img[:,:,0], img[:,:,1], img[:,:,2]
    c = np.mean(img[:,:,2])



    # print("wackawacka: ", c)
    plot_data["white percentage"].append(c)

    # numpy's hstack is used to stack two images horizontally,
    # so you see the various images generated in one figure:
    # cv2.imshow("images", np.hstack([img, output]))
    # cv2.waitKey(1)
    return c






def close_event():
    plt.close()

def colorReduce(image):
    div = 64
    quantized = image // div * div + div // 2
    return quantized



# with open('config_text.txt', 'r') as content_file:
#     content = content_file.read()


def vectorize(img):
    # img = cv2.imread('loadtest.png', 0)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result_fill = np.ones(img.shape, np.uint8) * 255


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    edged = cv2.Canny(gray, 30, 200)
    # cv2.waitKey(0)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # cv2.imshow('Canny Edges After Contouring', edged)
    # cv2.waitKey(0)

    # print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    # -1 signifies drawing all contours
    out = cv2.drawContours(result_fill, contours, -1, (0, 0, 255), 1)
    return out



    result_fill = np.ones(img.shape, np.uint8) * 255
    result_borders = np.zeros(img.shape, np.uint8)

    # the '[:-1]' is used to skip the contour at the outer border of the image
    contours, hierarchy = cv2.findContours(img,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # fill spaces between contours by setting thickness to -1
    cv2.drawContours(result_fill, contours, -1, 0, -1)
    cv2.drawContours(result_borders, contours, -1, 255, 1)

    # xor the filled result and the borders to recreate the original image
    result = result_fill ^ result_borders
    return result


def drawCircle(event, x, y, flags, param):

    if event == cv2.EVENT_MOUSEMOVE:
        print('({}, {})'.format(x, y))

        # imgCopy = img.copy()
        # cv2.circle(imgCopy, (x, y), 10, (255, 0, 0), -1)
        #
        # cv2.imshow('image', imgCopy)


class graph:
    def create_graph(self, data):
        project_dir = os.path.dirname(__file__)
        plot_style_filename = os.path.join(project_dir, 'assets\\mplstyles\\custom_1.mplstyle')
        vals = {
        "FPS":[],
        "img recog time":[],
        "white percentage":[],
        "score":[]
        }
        for key, value in data.items():
            vals[key].append(min(value))
            vals[key].append(max(value))
            vals[key].append(np.average(value))
            vals[key].append(np.median(value))



        # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'



        plt.style.use(plot_style_filename)
        plt.figure(figsize=(1920/100, 1080/100))
        # fig, (ax1, ax3) = plt.subplots(1, 2)

        ax1 = plt.subplot(221)
        ax1.set_title("FPS of game")
        ax1.axhline(y=vals["FPS"][0], linestyle="--", color="grey", label="Min: "+str(round(vals["FPS"][0])))
        ax1.axhline(y=vals["FPS"][1], linestyle="-.", color="grey", label="Max: "+str(round(vals["FPS"][1])))
        ax1.axhline(y=vals["FPS"][2], linestyle=":", color="blue", label="Mean: "+str(round(vals["FPS"][2])))
        ax1.axhline(y=vals["FPS"][3], linestyle="--", color="blue", label="Median: "+str(round(vals["FPS"][3])))
        ax1.plot(data["FPS"])


        ax2 = plt.subplot(222)
        ax2.plot(data["score"])
        ax2.axhline(y=vals["score"][0], linestyle="--", color="grey", label="Min: "+str(round(vals["score"][0], 3)))
        ax2.axhline(y=vals["score"][1], linestyle="-.", color="grey", label="Max: "+str(round(vals["score"][1], 3)))
        ax2.axhline(y=vals["score"][2], linestyle=":", color="blue", label="Mean: "+str(round(vals["score"][2], 3)))
        ax2.axhline(y=vals["score"][3], linestyle="--", color="blue", label="Median: "+str(round(vals["score"][3], 3)))
        ax2.set_title("Score")

        ax3 = plt.subplot(223)
        ax3.plot(data["img recog time"])
        ax3.axhline(y=vals["img recog time"][0], linestyle="--", color="grey", label="Min: "+str(round(vals["img recog time"][0], 3)))
        ax3.axhline(y=vals["img recog time"][1], linestyle="-.", color="grey", label="Max: "+str(round(vals["img recog time"][1], 3)))
        ax3.axhline(y=vals["img recog time"][2], linestyle=":", color="blue", label="Mean: "+str(round(vals["img recog time"][2], 3)))
        ax3.axhline(y=vals["img recog time"][3], linestyle="--", color="blue", label="Median: "+str(round(vals["img recog time"][3], 3)))
        ax3.set_title("Score recognition time")


        ax4 = plt.subplot(224)
        ax4.plot(data["white percentage"])
        ax4.axhline(y=vals["white percentage"][0], linestyle="--", color="grey", label="Min: "+str(round(vals["white percentage"][0], 3)))
        ax4.axhline(y=vals["white percentage"][1], linestyle="-.", color="grey", label="Max: "+str(round(vals["white percentage"][1], 3)))
        ax4.axhline(y=vals["white percentage"][2], linestyle=":", color="blue", label="Mean: "+str(round(vals["white percentage"][2], 3)))
        ax4.axhline(y=vals["white percentage"][3], linestyle="--", color="blue", label="Median: "+str(round(vals["white percentage"][3], 3)))
        ax4.set_title("white percentage")






        ax1.legend(loc=5, prop = {"size":10})
        ax2.legend(loc=5, prop = {"size":10})
        ax3.legend(loc=5, prop = {"size":10})
        ax4.legend(loc=5, prop = {"size":10})
        # plt.rcParams["figure.figsize"] = (60,3)

        str_time = time.strftime("%d %b %Y %H-%M-%S", time.gmtime())

        plt.savefig("data/graphs/graph_{0}.svg".format(str_time), format="svg")
        plt.savefig("data/graphs/graph_{0}.png".format(str_time), format="png")

        plt.show()


class score_getter(threading.Thread):

    def get_score(self,aaa):
        global score

        # removes ♀ and newlines

        temp_score = pytesseract.image_to_string(aaa, lang='1', config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789+').replace("\n", "").replace("\x0c", "").replace(" ","")


        if temp_score != "":

            # print(temp_score)

            try:
                eval_score = int(eval(temp_score))
                print("{0}, {1} = {2}".format(score, eval_score, (eval_score-score)))
                if ((eval_score-score)<=30):
                    score = eval_score
                    plot_data["score"].append(eval_score)
                    # print("SCORE: ",score, eval_score)



            except Exception as e:
                print(e)
                eval_score = 0

        else:
            print("SCORE: Null")
        # cv2.imwrite("attempts/1/attempt_1_-%s-_%s.png"%(text, ticker), score)
        # with open("sample.txt", "a") as file_object:
        #     # Append 'hello' at the end of file
        #     file_object.write(score + "\n")

    def run(self):
        global score_img

        while_true_time = time.time()

        while True:
            if global_running == 0:
                break
            if score_img is None:
                continue

            # print(score_img)
            _tmp = time.time()

            plot_data["img recog time"].append(_tmp - while_true_time)

            # print("while_true_time =======", _tmp - while_true_time)
            while_true_time = _tmp
            self.get_score(score_img)

class Game:
    def __init__(self):
        self.action_space = Discrete(2)



    def game_loop(self, ticker):

        # ticker = 0

        # while True:
        frames = android.get_next_frames()

        if frames is None:
            return False

        else:
            global public_frame, score_img, global_is_dead

            for frame in frames:
                ticker += 1
                public_frame = frame
                # 1440, 3040   (1080, 2280)
                adds_removed = frame[0:(3040-200), 0:1440]

                hor = 200

                score = frame[580-150:740-150, hor:(1080-hor)]

                # score = resizer(score, resize_size)




                imS = resizer(frame, resize_size)
                # print(imS.shape[0], imS.shape[1])
                sc = cv2.cvtColor(imS, cv2.COLOR_BGR2RGB)
                percentages = []

                imS = colorReduce(imS)

                edges = vectorize(imS)

                # edges = cv2.Canny(imS,0,100,9)




                white_perc = calcPercentage(imS)
                #
                #
                # print(global_is_dead)
                if white_perc >= 150:
                    global_is_dead = 80
                    print("DEAD")
                if global_is_dead > 0:
                    # print("minused one")
                    global_is_dead = global_is_dead - 1

                if global_is_dead == 1:
                    android.tap(560, 1300)
                    print("tapped\n\n\n\n")
                    # (560, 1300)

                # else:
                #     is_dead = 0

                # edges_height, edges_width = edges.shape






                # print(edges_height, edges_width)

                # score = imS[60:284-205, 30:144-30]

                # score = cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)

                _ , score = masking(score, white, "white score mask")

                # PIL_version_score = Image.fromarray(np.uint8(score)).convert('RGB')

                # score

                # score = (255-score)

                blur_amount = 11

                score_img = cv2.cvtColor(score, cv2.COLOR_BGR2RGB)

                score_img = cv2.GaussianBlur(score_img,(blur_amount,blur_amount),cv2.BORDER_DEFAULT)


                score_img = cv2.threshold(score_img,127,255,cv2.THRESH_BINARY)[1]

                cv2.imwrite("attempts/1/attempt_1_%s.png"%(ticker), imS)


                # score = Image.frombytes('RGB', score.shape[:2], score, 'raw', 'BGR', 0, 0)




                # --psm 7, 8 and 9 work, 7 was picked at random

                # if ticker%10==0:
                #     get_score(score)

                # print("Text: %s\n      %s"%(text, pytesseract.image_to_string(score)))
                # , 200)
                # resizer(, 300)



                cv2.imshow("Edges", resizer(edges, 200))
                cv2.setMouseCallback("Edges", drawCircle)
                cv2.imshow('Phone Viewer', resizer(imS, 200))
                cv2.setMouseCallback('Phone Viewer', drawCircle)
                cv2.imshow("Score", score)
                cv2.waitKey(1)
                return True




if __name__ == "__main__":



    # android.set_screen_power_mode(0)

    print(android.resolution)

    itt = 0

    score_getter().start()

    gme = Game()

    while itt <= 1000:



        s = time.perf_counter()




        suc = gme.game_loop(itt)
        if suc != False:
            itt = itt + 1
            elapsed = time.perf_counter() - s
            plot_data["FPS"].append(1/elapsed)
            if itt%10 == 0:
                print(f"Frame numer: {itt}")
        # print(f"{__file__} executed in {elapsed:0.6f} seconds.")
    global_running = 0

    graph().create_graph(plot_data)
