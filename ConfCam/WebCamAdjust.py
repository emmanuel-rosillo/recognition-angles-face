import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import json
from .camconfiguration import CamConfig

# names of files and directories
_nameConfigPath = 'ConfCam'
_nameFileConf = 'focalLength.json'

# object to confCam
findDistances = CamConfig()
detector = FaceMeshDetector(maxFaces=1)


def findFocalLength():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            face = faces[0]
            eyeRight = face[145]
            eyeLeft = face[374]
            # draws
            cv2.circle(img, eyeRight, 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, eyeLeft, 5, (255, 0, 255), cv2.FILLED)

            # *--* w *--*
            w, _ = detector.findDistance(eyeRight, eyeLeft)
            W = findDistances.distanceBetweenEyes
            d = findDistances.distanceEyes
            f = int((w * d) / W)
            cv2.putText(img, str(int(f)), (20, 20), 1, 1, (0, 255, 0), 2)

            confCam = {
                'focalLength': f
            }
        cv2.imshow("facial web", img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            fileName = getPathConfigCam()
            with open(fileName, 'w') as file:
                json.dump(confCam, file, indent=2)
            cap.release()
            cv2.destroyAllWindows()
            break


def getPathConfigCam():
    name = _nameConfigPath + '/' + _nameFileConf
    return name

# def _createConfigDir():
#     if not os.path.exists(_nameConfigPath):
#         os.mkdir(_nameConfigPath)
