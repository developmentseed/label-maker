"""Color values for creating visual labels"""
from PIL import ImageColor
# Taken from https://github.com/CartoDB/CartoColor/blob/master/cartocolor.js#L1633-L1733
colors = ["#DDCC77", "#CC6677", "#117733", "#332288", "#AA4499", "#88CCEE"]

def class_color(c):
    """Return 3-element tuple containing rgb values for a given class"""
    if c == 0:
        return (0, 0, 0) # background class
    return ImageColor.getrgb(colors[c % len(colors)])
