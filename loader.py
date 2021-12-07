from PIL import Image
def MyLoader(path):
    im = Image.open(path)
    im_rgb = im.convert('RGB')
    region = im_rgb.resize((1280,384))
    return region