# DeepFace-Recognition
Simply detect face information (age, gender, emotion) by using deepface module.


# import OpenCV module
import cv2
# import deepface module
from deepface import DeepFace


## Create init
    def __init__(self):
        # define paths
        self.pathA = 'you own path\\haar\\eye.xml'
        self.pathB = 'you own path\\haar\\face.xml'
        self.pathC = 'you own path\\haar\\nose.xml'
        self.pathF1 = 'you own path\\Kim.jpg'
        self.pathF2 = 'you own path\\Nam.jpg'

        result = DeepFace.verify(self.pathF1, self.pathF2)
        print("Is verified: ", result["verified"])


## Run codes
    def classifiers(self):
        # load OpenCV face detector
        eye_cascade = cv2.CascadeClassifier(self.pathA)
        face_cascade = cv2.CascadeClassifier(self.pathB)
        nose_cascade = cv2.CascadeClassifier(self.pathC)

        # list to hold all subject faces
        faces_holder = []

        # set images in a list & verify
        datasets = [self.pathF1, self.pathF2]

        # read image
        for image in datasets:
            image = cv2.imread(image)

            faces = face_cascade.detectMultiScale(image, 1.08, 5, minSize=(50, 50))
            for left, top, width, height in faces:
                # extract the faces area
                faces = image[top:top+height, left:left+width]
                cv2.rectangle(image, (left, top), (left+width, top+height),
                              (13, 230, 23), 2)

                # detect noses area from faces area
                noses = nose_cascade.detectMultiScale(faces, 1.15, 5,
                                                      flags=cv2.CASCADE_SCALE_IMAGE)
                for (nx, ny, nw, nh) in noses:
                    cv2.rectangle(faces, (nx, ny), (nx+nw, ny+nh), (255, 0, 255), 2)

                # detect eyes area from faces area
                eyes = eye_cascade.detectMultiScale(faces, 1.100, 4,
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
                for(ex, ey, ew, eh) in eyes:
                    cv2.rectangle(faces, (ex, ey), (ex+ew, ew+eh), (255, 255, 255), 2)

                if faces is not None:
                    # add face to list of faces_holder
                    faces_holder.append(image[top:top+height, left:left+width])

            # to analyze age, gender, emotion
            demography = DeepFace.analyze(image, ['emotion', 'age', 'gender'])

            # print total faces (text color:white)
            input_strings = "Age:{} Gender:{} Emotion:{} "
            cv2.putText(image, input_strings.format(str(demography["age"]),
                                                    demography["gender"],
                                                    demography["dominant_emotion"]),
                        (150, 300), cv2.FONT_HERSHEY_PLAIN, 2, (13, 230, 23), 2)

            # display an image window to show the image
            cv2.imshow('DeepFace Recognition ', image)
            cv2.waitKey()
            cv2.destroyAllWindows()

        # print the quantity of detected faces.
        print('Total detected faces:%d' % len(faces_holder))


if __name__ == "__main__":
    run = FaceRecognition()
    run.classifiers()
