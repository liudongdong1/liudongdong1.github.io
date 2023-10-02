# PillowRecord


>  extensive file format support, an efficient internal representation, and fairly powerful image processing capabilities. The core image library is designed for fast access to data stored in a few basic pixel formats. `Image Arhives, image Dsplay, image processing`

#### 0. Concept

##### .1. Bands

> The Python Imaging Library allows you to `store several bands in a single image`, provided they all have the `same dimensions and depth`. For example, a PNG image might have ‘R’, ‘G’, ‘B’, and ‘A’ bands for the red, green, blue, and alpha transparency values. Many operations act on each band separately, e.g., histograms. It is often useful to think of each pixel as having one value per band.

##### .2.  Mode

The `mode` of an image is a string which defines the type and depth of a pixel in the image. Each pixel uses the full range of the bit depth. So a 1-bit pixel has a range of 0-1, an 8-bit pixel has a range of 0-255 and so on. The current release supports the following standard modes:

> - `1` (1-bit pixels, black and white, stored with one pixel per byte)
> - `L` (8-bit pixels, black and white)
> - `P` (8-bit pixels, mapped to any other mode using a color palette)
> - `RGB` (3x8-bit pixels, true color)
> - `RGBA` (4x8-bit pixels, true color with transparency mask)
> - `CMYK` (4x8-bit pixels, color separation)
> - `YCbCr` (3x8-bit pixels, color video format)
>   - Note that this refers to the JPEG, and not the ITU-R BT.2020, standard
> - `LAB` (3x8-bit pixels, the L*a*b color space)
> - `HSV` (3x8-bit pixels, Hue, Saturation, Value color space)
> - `I` (32-bit signed integer pixels)
> - `F` (32-bit floating point pixels)

Pillow also provides limited support for a few special modes, including:

> - `LA` (L with alpha)
> - `PA` (P with alpha)
> - `RGBX` (true color with padding)
> - `RGBa` (true color with premultiplied alpha)
> - `La` (L with premultiplied alpha)
> - `I;16` (16-bit unsigned integer pixels)
> - `I;16L` (16-bit little endian unsigned integer pixels)
> - `I;16B` (16-bit big endian unsigned integer pixels)
> - `I;16N` (16-bit native endian unsigned integer pixels)
> - `BGR;15` (15-bit reversed true colour)
> - `BGR;16` (16-bit reversed true colour)
> - `BGR;24` (24-bit reversed true colour)
> - `BGR;32` (32-bit reversed true colour)

##### .3. size&Coordinate

>  horizontal and vertical size in pixels. with (0,0) in the upper left corner.

##### .4. Filters

- `PIL.Image.``NEAREST`

  Pick one nearest pixel from the input image. Ignore all other input pixels.

- `PIL.Image.``BOX`

  Each pixel of source image contributes to one pixel of the destination image with identical weights. For upscaling is equivalent of [`NEAREST`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#PIL.Image.NEAREST). This filter can only be used with the [`resize()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize) and [`thumbnail()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail) methods.*New in version 3.4.0.*

- `PIL.Image.``BILINEAR`

  For resize calculate the output pixel value using linear interpolation on all pixels that may contribute to the output value. For other transformations linear interpolation over a 2x2 environment in the input image is used.

- `PIL.Image.``HAMMING`

  Produces a sharper image than [`BILINEAR`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#PIL.Image.BILINEAR), doesn’t have dislocations on local level like with [`BOX`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#PIL.Image.BOX). This filter can only be used with the [`resize()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize) and [`thumbnail()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail) methods.*New in version 3.4.0.*

- `PIL.Image.``BICUBIC`

  For resize calculate the output pixel value using cubic interpolation on all pixels that may contribute to the output value. For other transformations cubic interpolation over a 4x4 environment in the input image is used.

- `PIL.Image.``LANCZOS`

  Calculate the output pixel value using a high-quality Lanczos filter (a truncated sinc) on all pixels that may contribute to the output value. This filter can only be used with the [`resize()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize) and [`thumbnail()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail) methods.

#### 1. Read&write

```python
from PIL import Image
im = Image.open("hopper.ppm")
print(im.format, im.size, im.mode)
im.show()

#read from an open file
from PIL import Image
with open("hopper.ppm", "rb") as fp:
    im = Image.open(fp)
#read from binary data
from PIL import Image
import io
im = Image.open(io.BytesIO(buffer))
#read from tar archive
from PIL import Image, TarIO
fp = TarIO.TarIO("Tests/images/hopper.tar", "hopper.jpg")
im = Image.open(fp)
```

#### 2. cut&paste&merge

```python
box = (100, 100, 400, 400)
region = im.crop(box) #(left, upper, right, lower) uses a coordinate system with (0, 0) in the upper left corner

#rotate
region = region.transpose(Image.ROTATE_180)
im.paste(region, box)

def roll(image, delta):
    """Roll an image sideways."""
    xsize, ysize = image.size

    delta = delta % xsize
    if delta == 0: return image

    part1 = image.crop((0, 0, delta, ysize))
    part2 = image.crop((delta, 0, xsize, ysize))
    image.paste(part1, (xsize-delta, 0, xsize, ysize))
    image.paste(part2, (0, 0, xsize-delta, ysize))

    return image

#split&merge
r, g, b = im.split()
im = Image.merge("RGB", (b, g, r))
```

#### 3. Geometrical transforms

```python
out = im.resize((128, 128))
out = im.rotate(45) # degrees counter-clockwise

out = im.transpose(Image.FLIP_LEFT_RIGHT)
out = im.transpose(Image.FLIP_TOP_BOTTOM)
out = im.transpose(Image.ROTATE_90)
out = im.transpose(Image.ROTATE_180)
out = im.transpose(Image.ROTATE_270)
```

#### 4. Color transforms

```python
from PIL import Image
with Image.open("hopper.ppm") as im:
    im = im.convert("L")
```

#### 5. Enhancement

```
#filters
from PIL import ImageFilter
out = im.filter(ImageFilter.DETAIL)

# multiply each pixel by 1.2
out = im.point(lambda i: i * 1.2)

# split the image into individual bands
source = im.split()
R, G, B = 0, 1, 2
# select regions where red is less than 100
mask = source[R].point(lambda i: i < 100 and 255)
# process the green band
out = source[G].point(lambda i: i * 0.7)
# paste the processed band back, but only where red was < 100
source[G].paste(out, None, mask)
# build a new multiband image
im = Image.merge(im.mode, source)

from PIL import ImageEnhance
enh = ImageEnhance.Contrast(im)
enh.enhance(1.3).show("30% more contrast")#adjust contrast, brightness, color balance and sharpness
```

#### 6. Sequences

> Supported sequence formats include FLI/FLC, GIF, and a few experimental formats. TIFF files can also contain more than one frame.

```python
from PIL import ImageSequence
for frame in ImageSequence.Iterator(im):
    # ...do something to frame...
```

#### 7. Drawing PostAcript

```python
from PIL import Image
from PIL import PSDraw

with Image.open("hopper.ppm") as im:
    title = "hopper"
    box = (1*72, 2*72, 7*72, 10*72) # in points

    ps = PSDraw.PSDraw() # default is sys.stdout
    ps.begin_document(title)

    # draw the image (75 dpi)
    ps.image(box, im, 75)
    ps.rectangle(box)

    # draw title
    ps.setfont("HelveticaNarrow-Bold", 36)
    ps.text((3*72, 4*72), title)

    ps.end_document()
```

#### 8. Learning Resources

- https://pillow.readthedocs.io/en/stable/handbook/appendices.html

---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: liudongdong1.github.io/pillowrecord/  

