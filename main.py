# from ppadb.client import Client

from ppadb.client import Client
import time

import PIL.Image
from PIL import ImageDraw, ImageTk, Image
import numpy as np

import cv2

# TO BE REPLACED

# from assets.viewer import AndroidViewer
# import assets.keycodes as keycodes

import scrcpy

import asyncio
import threading

import os

import matplotlib.pyplot as plt

import pytesseract

from gymnasium import Env

from gymnasium.spaces import Discrete, Box

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Flatten, Rescaling, Conv2D
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

import random

from datetime import datetime


public_frames = []



def get_first_device_connected():
    client = Client(host="127.0.0.1", port=5037)
    devices = client.devices()
    device = devices[0]
    return (device.serial)

def on_image(frame):
    if frame is not None:
        public_frames.append(frame)

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

def build_model(states, actions):
    model = Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(1, states)))
    # model.add(tf.keras.layers.Convolution2D(32, kernel_size=(6, 6), input_shape=(1, 228, 108, 1), activation="relu", data_format="channels_last"))
    # model.add(tf.keras.layers.Convolution2D(16, (4, 4), activation="relu", data_format="channels_last"))
    # model.add(tf.keras.layers.Convolution2D(64, (3, 3), activation="relu", data_format="channels_last"))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", data_format="channels_last"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    # model.add(tf.keras.layers.Dense(128, activation="relu"), data_format="channels_last")
    model.add(tf.keras.layers.Dense(actions, activation="linear"))




    # model = Sequential()
    # # model.add(tf.keras.layers.Flatten(input_shape=(1, states)))
    # model.add(tf.keras.layers.Convolution2D(32, kernel_size=(6, 6), input_shape=(1, 228, 108, 1), activation="relu", data_format="channels_last"))
    # model.add(tf.keras.layers.Convolution2D(16, (4, 4), activation="relu", data_format="channels_last"))
    # # model.add(tf.keras.layers.Convolution2D(64, (3, 3), activation="relu", data_format="channels_last"))
    # model.add(tf.keras.layers.Flatten())
    # # model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", data_format="channels_last"))
    # model.add(tf.keras.layers.Dense(512, activation="relu"))
    # model.add(tf.keras.layers.Dense(256, activation="relu"))
    # # model.add(tf.keras.layers.Dense(128, activation="relu"), data_format="channels_last")
    # model.add(tf.keras.layers.Dense(actions, activation="linear"))
    return model

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps", value_max=1., value_min=1., value_test=2., nb_steps=10000)
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)

    return dqn

coords = []

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

dev_name = get_first_device_connected()

client = scrcpy.Client(device=dev_name, max_fps=30)

client.add_listener(scrcpy.EVENT_FRAME, on_image)

client.start(threaded=True)

print("Connected to: {0}".format(dev_name))

# Initial score will be 0
score = 0
score_img = None

global_running = 1
global_is_dead = 80
globel_score_diff = 0


# main loop constants
resize_size = 10
white = [[0,0,253], [255,10,255]]

plot_data = {
"FPS":[],
"img recog time":[],
"white percentage":[],
"score":[]}

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

        str_time = datetime.today().strftime('%d-%m-%Y %H-%M-%S.%f')

        plt.savefig("data/graphs/graph_{0}.svg".format(str_time), format="svg")
        plt.savefig("data/graphs/graph_{0}.png".format(str_time), format="png")

        plt.show()

class score_getter(threading.Thread):

    def get_score(self,score_img):
        global score, globel_score_diff

        # removes ♀ and newlines

        temp_score = pytesseract.image_to_string(score_img, lang='1', config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789+').replace("\n", "").replace("\x0c", "").replace(" ","")

        if temp_score != "":

            # print(temp_score)

            try:
                eval_score = int(eval(temp_score))
                if ((eval_score-score)<=30):
                    # print("{0}, {1} = {2}".format(score, eval_score, (eval_score-score)))
                    score = eval_score
                    globel_score_diff = (eval_score-score)
                    # print(globel_score_diff)
                    plot_data["score"].append(eval_score)
                    # print("SCORE: ",score, eval_score)

            except Exception as e:
                print(e)
                eval_score = 0

    def run(self):
        global score_img

        while_true_time = time.time()

        while True:
            # print("THREAD SCORE: ", threading.current_thread().ident)
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

class Game(Env):
    def __init__(self):
        self.action_space = Discrete(2)
        self.state = 0
        self.clicked_state = 0
        self.not_increasing_score = 0
        self.reset_ticker = 0

        self.title_screen = cv2.imread("assets/title_screen.png")
        _temp = cv2.calcHist([cv2.cvtColor(self.title_screen, cv2.COLOR_BGR2RGB)], [0,1], None, [180,256], [0,180,0,256])
        self.title_screen_hist = cv2.normalize(_temp, _temp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        score_getter().start()
        # android.tap(560, 1300)


    def step(self, action=0):
        # print( "THREAD STEP: ", threading.current_thread().ident)
        # print("Action: {0}".format(action))

        global public_frames, score_img, global_is_dead, score, globel_score_diff

        while True:
            # frames = android.get_next_frames()

            if len(public_frames) == 0:
                continue


            else:
                print(len(public_frames))
                # If the amount of frames are starting to back stuff get the last 2 (to remove massive spikes)
                if len(public_frames) > 2:
                    public_frames = public_frames[-2:]
                
                for frame in public_frames:
                    # ticker += 1
                    # 1440, 3040   (1080, 2280)
                    adds_removed = frame[0:(3040-200), 0:1440]

                    hor = 200

                    score_cropped = frame[430-100:590-100, hor:(1080-hor)]

                    # score = resizer(score, resize_size)

                    imS = resizer(frame, resize_size)
                    # print(imS.shape[0], imS.shape[1])
                    sc = cv2.cvtColor(imS, cv2.COLOR_BGR2RGB)
                    percentages = []

                    imS = colorReduce(imS)

                    edges = vectorize(imS)

                    white_perc = calcPercentage(imS)

                    _ , score_cropped = masking(score_cropped, white, "white score mask")

                    blur_amount = 11

                    score_img = cv2.cvtColor(score_cropped, cv2.COLOR_BGR2RGB)

                    score_img = cv2.GaussianBlur(score_img,(blur_amount,blur_amount),cv2.BORDER_DEFAULT)

                    imS = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)


                    challanges = imS[180:228-5, 0:108]


                    challanges = cv2.merge([challanges, challanges, challanges])
                    _temp =  cv2.calcHist([challanges], [0,1], None, [180,256], [0,180,0,256])

                    normalized_hist_challanges = cv2.normalize(_temp, _temp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                    perc_diff = (cv2.compareHist(self.title_screen_hist, normalized_hist_challanges, cv2.HISTCMP_BHATTACHARYYA))

                    if (white_perc >= 150) or ((perc_diff<=0.2) and (self.reset_ticker >= 100)):
                        global_is_dead = 80
                        print("DEAD", self.reset_ticker)
                        # self.reward -= 20
                        # self.reset()
                        done = True
                    else:
                        done = False

                    if (self.clicked_state == 0) and (action == 1):
                        client.control.touch(560, 1300, scrcpy.ACTION_DOWN)

                    elif (self.clicked_state == 1) and (action == 0):
                        client.control.touch(560, 1300, scrcpy.ACTION_UP)

                    if global_is_dead > 0:
                        # print("minused one")
                        global_is_dead = global_is_dead - 1

                    score_img = cv2.threshold(score_img,127,255,cv2.THRESH_BINARY)[1]

                    str_time = datetime.today().strftime('%d-%m-%Y %H-%M-%S.%f')
                    # print(str_time)
                    # time.strftime("%d %b %Y %H-%M-%S", time.gmtime())

                    # cv2.imwrite("attempts/3/challanges_%s.png"%(str_time), challanges)
                    cv2.imshow("challanges", resizer(challanges, 200))
                    cv2.imshow("Edges", resizer(edges, 200))
                    cv2.setMouseCallback("Edges", drawCircle)
                    cv2.imshow('Phone Viewer', resizer(imS, 200))
                    cv2.setMouseCallback('Phone Viewer', drawCircle)
                    cv2.imshow("Score", score_cropped)
                    cv2.waitKey(1)

                    info = {}

                    self.reward = score

                    self.state = imS

                    if not (globel_score_diff >= 1):
                         self.not_increasing_score += 1

                    else:
                        self.not_increasing_score = 0

                    if (self.not_increasing_score >= 20):
                        # print(type(self.reward), self.reward)
                        self.reward -= 0.5

                    # print(self.reward)
                    # print("\n\n\n")
                    self.reset_ticker += 1
                    # print(np.expand_dims(np.expand_dims(self.state, axis=2), axis=0).shape)
                    # print(self.state.shape)

                    # print(imS.shape)
                    # print(imS)

                    # 228, 108
                    # print(self.state)
                    return (self.state.flatten(), self.reward, done, info)
                    # return ((np.expand_dims(self.state, axis=-1)), self.reward, done, info)

    def reset(self):

        # while True:
        #     frames = android.get_next_frames()
        #     if frames is None:
        #         pass
        #
        #     else:
        #         print(len(frames))


        global global_is_dead
        print("Beginning reset")
        self.reward = 0
        self.not_increasing_score = 0
        self.reset_ticker = 0
        num_to_complete = 120


        while num_to_complete >= 0:
            # print("THREAD RESET: ", threading.current_thread().ident)
            # print("num_to_complete: ", num_to_complete)
            # frames = android.get_next_frames()
            # print(frames.length())

            if len(public_frames) == 0:
                continue

            else:
                # print("111111111", len(frames))
            # if not (frames is None):
                # print(len(frames))
                for frame in public_frames:
                    # print("222222222222")
                    adds_removed = frame[0:(3040-200), 0:1440]

                    hor = 200

                    score = frame[580-150:740-150, hor:(1080-hor)]

                    # score = resizer(score, resize_size)

                    imS = resizer(frame, resize_size)
                    # print(imS.shape[0], imS.shape[1])
                    sc = cv2.cvtColor(imS, cv2.COLOR_BGR2RGB)

                    imS = colorReduce(imS)

                    edges = vectorize(imS)


                    white_perc = calcPercentage(imS)

                    _ , score = masking(score, white, "white score mask")

                    blur_amount = 11

                    imS = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)
                    challanges = imS[180:228-5, 0:108]

                    str_time = datetime.today().strftime('%d-%m-%Y %H-%M-%S.%f')

                    challanges = cv2.merge([challanges, challanges, challanges])
                    _temp =  cv2.calcHist([challanges], [0,1], None, [180,256], [0,180,0,256])

                    normalized_hist_challanges = cv2.normalize(_temp, _temp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                    perc_diff = (cv2.compareHist(self.title_screen_hist, normalized_hist_challanges, cv2.HISTCMP_BHATTACHARYYA))

                    if (perc_diff<=0.2) and (num_to_complete >= 4) and (num_to_complete%10 == 0):
                        print("HALVED")
                        num_to_complete = round(num_to_complete/2)

                        # android.tap(560, 1300)
                        # time.sleep(0.02)
                        # print("Finished reset")
                        # time.sleep(0.02)
                        # return(self.state)


                    # cv2.imwrite("attempts/3/challanges_%s.png"%(str_time), challanges)



                    cv2.imshow("challanges", resizer(challanges, 200))
                    cv2.imshow("Edges", resizer(edges, 200))
                    cv2.setMouseCallback("Edges", drawCircle)
                    cv2.imshow('Phone Viewer', resizer(imS, 200))
                    cv2.setMouseCallback('Phone Viewer', drawCircle)
                    cv2.imshow("Score", score)
                    cv2.waitKey(1)
                    num_to_complete -= 1
                    self.state = imS



        client.control.touch(560, 1300, scrcpy.ACTION_DOWN)
        client.control.touch(560, 1300, scrcpy.ACTION_UP)
        print("Finished reset")
        # print(self.state.shape)
        return(self.state.flatten())
        # return(np.expand_dims(self.state, axis=-1))


if __name__ == "__main__":


    # config = tf.ConfigProto(device_count = {'GPU': 1})
    # sess = tf.Session(config=config)
    # android.set_screen_power_mode(0)

    # print(android.resolution)

    itt = 0

    # score_getter().start()

    gme = Game()
    #
    # print(gme.action_space.n)

    episodes = 10
    global_running = 1


    print(threading.current_thread().ident)
    for episode in range(1, episodes+1):

        while True:
            # try:

            done = False
            score = 0

            itt = 0

            while not done:
                s = time.perf_counter()

                action = gme.action_space.sample()
                out = gme.step(action)
                itt += 1
                if out != None:
                    n_state, reward, done, info = out
                    # ai_visuals = n_state.reshape(8208,3)[:,0]
                    # reshape((3, 2))
                    elapsed = time.perf_counter() - s
                    # print("Elapsed FPS: {0}".format(1/elapsed))
                    plot_data["FPS"].append(1/elapsed)
            print("Episode: {0} Score:{1}".format(episode, reward))
            gme.reset()
            print("Reset Env")

            break

    global_running = 0

    graph().create_graph(plot_data)
