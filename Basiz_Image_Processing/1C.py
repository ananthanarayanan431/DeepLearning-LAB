
from PIL import Image,ImageFilter

image = Image.open("anantha.jpg")
image = image.convert("L")
image = image.filter(ImageFilter.FIND_EDGES)
image.save("output.png")
image.show("output.png")