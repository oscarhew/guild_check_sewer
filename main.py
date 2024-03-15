import numpy as np
import pytesseract
import cv2
import pyautogui
import requests

from PIL import Image
from PIL import ImageFilter
from matplotlib import pyplot as plt

requests.packages.urllib3.disable_warnings()

image_path = 'guild_image.png'


opencvImage = cv2.imread(image_path)

opencvImage = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)

(thresh, blackAndWhiteImage) = cv2.threshold(opencvImage, 127, 255, cv2.THRESH_BINARY)


# cv2.imshow('image lai de', blackAndWhiteImage)
# cv2.waitKey(3000)
# cv2.destroyAllWindows()

# print(text)

# image = pyautogui.screenshot()
# image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
# cv2.imshow('screenshot hehe' , image)
# cv2.waitKey(3000)
# cv2.destroyAllWindows()

screenshot = cv2.imread('screenshot.png', cv2.IMREAD_GRAYSCALE)

template = cv2.imread('template_big.png', cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

img = screenshot.copy()
method = eval('cv2.TM_CCORR_NORMED')

# Apply template Matching
res = cv2.matchTemplate(img,template,method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# find guild list
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# adjust to fit guild list
top_left_whole = (max_loc[0], max_loc[1] + 30)
bottom_right_whole = (top_left[0] + w - 15, top_left[1] + 450)
width_ss = bottom_right_whole[0]
height_ss = bottom_right_whole[1]

# take screenshot of the guild list
cv2.rectangle(img,top_left_whole, bottom_right_whole, (255, 0, 0), 2)
guild_list_ss = pyautogui.screenshot(region=[top_left_whole[0], top_left_whole[1], width_ss, height_ss])
plt.imshow(guild_list_ss, cmap = 'gray')

# convert to black and white
guild_list_ss = np.array(guild_list_ss)
(thresh, blackAndWhiteImage) = cv2.threshold(guild_list_ss, 127, 255, cv2.THRESH_BINARY)

# extract text
# Adding custom options
custom_config = r'--psm 11'
text = pytesseract.image_to_string(guild_list_ss, config=custom_config)


# plt.subplot(121),plt.imshow(res,cmap = 'gray')
# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(img,cmap = 'gray')
# plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

# plt.show()