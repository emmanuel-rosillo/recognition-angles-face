from PIL import Image, ExifTags


def extract():
    img = Image.open("x.jpg")
    exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}
    print(exif)


if __name__ == '__main__':
    extract()
