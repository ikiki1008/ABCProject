
# import pandas as pd
# Import required Libraries
import argparse
import os
import sys
import urllib.request
from pathlib import Path
import subprocess
import eel
import time
import multiprocessing as mp
from datetime import datetime as dt
from tkinter import *
from PIL import Image, ImageTk
import cv2
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn

win = Tk()
win.geometry("700x350")
label = Label(win)
label.grid(row=0, column=0)
cap = soruce
# cv2.moveWindow(win, 40, 30)

def show_frames():
    cv2image = cv2.cv2Color(cap.read()[1], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(20, show_frames)

win.mainloop(), show_frames()

# @eel.expose
# def camPage():
#     eel.init("www2")
#     eel.start("cameraPage.html", size=(2000, 1500), port=0)


# getData = pd.read_table('soldlist.txt')
# print(getData)
# f = open('soldlist.txt','r')
# for line in f:
#     box = (line)
#     print(box)
#         # eel.init("www2")
#         # eel.getData(box)
#         # eel.start("data.html", size=(1000, 800), port=0)


