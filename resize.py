import os
from PIL import Image

path = './images/'
savedir ='./birds/'
nr = 0
for subdir in os.listdir(path):
    for img in os.listdir(path + subdir + '/'):
        im = Image.open(path + subdir + '/'+ img)
        im1 = im.resize((64, 64), Image.NEAREST)
        im1.save(savedir + str(nr) + '.png')
        nr +=1


