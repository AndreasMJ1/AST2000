import imageio
import numpy as np 
from PIL import Image
from tqdm import trange
images = []


for i in trange(360):
    img = Image.open(f"C:/Users/andym/Documents/GitHub/AST2000/Del4/360/{i}.png")
    pix = np.array(img)
    images.append(pix)

imageio.mimsave("movie.gif",images)
