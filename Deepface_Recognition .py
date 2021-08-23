import cv2
from deepface import DeepFace


class FaceRecognition(object):
    def __init__(self):
        self.pathA = 'your own path\\haar\\eye.xml'
        self.pathB = 'your own path\\haar\\face.xml'
        self.pathC = 'your own path\\haar\\nose.xml'
        self.pathF1 = 'your own path\\day07\\Kim.jpg'
        self.pathF2 = 'your own path\\day07\\Nam.jpg'

        result = DeepFace.verify(self.pathF1, self.pathF2)
        print("Is verified: ", result["verified"])

    def classifiers(self):
        eye_cascade = cv2.CascadeClassifier(self.pathA)
        face_cascade = cv2.CascadeClassifier(self.pathB)
        nose_cascade = cv2.CascadeClassifier(self.pathC)

        faces_holder = []
        datasets = [self.pathF1, self.pathF2]

        for image in datasets:
            image = cv2.imread(image)

            faces = face_cascade.detectMultiScale(image, 1.08, 5, minSize=(50, 50))
            for left, top, width, height in faces:
                faces = image[top:top+height, left:left+width]
                cv2.rectangle(image, (left, top), (left+width, top+height), (13, 230, 23), 2)

                noses = nose_cascade.detectMultiScale(faces, 1.15, 5,
                                                      flags=cv2.CASCADE_SCALE_IMAGE)
                for (nx, ny, nw, nh) in noses:
                    cv2.rectangle(faces, (nx, ny), (nx+nw, ny+nh), (255, 0, 255), 2)

                eyes = eye_cascade.detectMultiScale(faces, 1.100, 4, flags=cv2.CASCADE_SCALE_IMAGE)
                for(ex, ey, ew, eh) in eyes:
                    cv2.rectangle(faces, (ex, ey), (ex+ew, ew+eh), (255, 255, 255), 2)

                if faces is not None:
                    faces_holder.append(image[top:top+height, left:left+width])

            demography = DeepFace.analyze(image, ['emotion', 'age', 'gender'])
            input_strings = "Age:{} Gender:{} Emotion:{} "
            cv2.putText(image, input_strings.format(str(demography["age"]),
                                                    demography["gender"],
                                                    demography["dominant_emotion"]),
                        (150, 300), cv2.FONT_HERSHEY_PLAIN, 2, (13, 230, 23), 2)

            cv2.imshow('DeepFace Recognition ', image)
            cv2.waitKey()
            cv2.destroyAllWindows()

        print('Total detected faces:%d' % len(faces_holder))


if __name__ == "__main__":
    run = FaceRecognition()
    run.classifiers()
