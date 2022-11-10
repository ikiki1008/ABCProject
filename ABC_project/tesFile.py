import urllib.request
import os

image_url = 'http://www.randalolson.com/wp-content/uploads/Frankenstein.jpg'
image_path = 'Frankenstein.jpg'

if not os.path.exists(image_path):
    urllib.request.urlretrieve(image_url, image_path)

#this algorithm works best for images w/ light background!


# from PIL import Image
#
# original image = image.open(image_path)
# bw_image = original_image.convert('1',dither=Image.NONE)
# bw_image