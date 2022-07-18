from PIL import Image
from PIL.ExifTags import TAGS
import cv2
import json


def getImageName(path, nameOfPicture, extension):
    extensions = ['.jpg', '.png', '.gif']
    if path == '':
        pathName = nameOfPicture + extensions[extension]
    else:
        pathName = path + '/' + nameOfPicture + extensions[extension]
    return pathName


def capture(imagesName):

    cap = cv2.VideoCapture(0)
    flag = cap.isOpened()

    while flag:

        ret, frame = cap.read()
        cv2.imshow('Configure', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            cv2.imwrite(imagesName, frame)
            break
            cap.release()
    cv2.destroyAllWindows()


def getMetaData(nameOfImage):
    image = Image.open(nameOfImage)
    exifdata = image.getexif()
    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        # decode bytes
        if isinstance(data, bytes):
            data = data.decode()
        print(f"{tag:25}: {data}")
    # with open('Metadata.json', 'w') as f:
    #     json.dump(exif, f)


if __name__ == '__main__':
    imgName = getImageName(path='', nameOfPicture='x', extension=0)
    # capture(imgName)
    getMetaData(imgName)

