# Rider Controller

> This requires:
> - An Android device with [USB debugging](https://developer.android.com/studio/debug/dev-options) enabled.
> - A computer with a USB cable to connect to the Android device.

# Info

A machine learning algorithm that controls an Android device running the mobile game [Rider](https://play.google.com/store/apps/details?id=com.ketchapp.rider) developed by [Ketchapp games](http://www.ketchappgames.com/).

A save editor for this program can be usefull. ([rider_save_editor](https://github.com/JKook-Plus/rider_save_editor))

# Setup

### PC side

> **Warning**
> Python 3.11 will not work

Install python ([3.10](https://www.python.org/downloads/release/python-3100/)). 

[Clone the repoository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) ***OR*** [Download it](https://github.com/JKook-Plus/rider_controller/archive/refs/heads/main.zip)

Create a virtual environment (recommended)

`py -3.10 -m venv venv`

Open the virtual environment

__Windows__
`venv\Scripts\activate.bat`

Install requirements.txt

`pip install -r requirements.txt`

Install [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) into `C:\Program Files\Tesseract-OCR\tesseract.exe`.




Add "screpy-server.jar" to assets/server/

Add "AdbWinUsbApi.dll", "AdbWinApi.dll", "adb.exe" to assets/adb/
