# DeepFace-Recognition
Simply detect face information (age, gender, emotion) by using deepface module.

## Requirements
● Python 3.8    
● cv2  
● deepface


## Class
FaceRecognition


## Functions
● classifiers  


## Create init
#### define paths of all required harr xml :
    def __init__(self):
        # define paths
        self.pathA = 'you own path\\haar\\eye.xml'
        self.pathB = 'you own path\\haar\\face.xml'
        self.pathC = 'you own path\\haar\\nose.xml'
        self.pathF1 = 'you own path\\Kim.jpg'
        self.pathF2 = 'you own path\\Nam.jpg'
        
 
#### verify function under the DeepFace interface.
        result = DeepFace.verify(self.pathF1, self.pathF2)
        print("Is verified: ", result["verified"])


## Run codes
#### 1. Call the main finction to work, start.
    if __name__ == "__main__":
        run = FaceRecognition()
        run.classifiers()
        
#### 2. load OpenCV face detector :
    def classifiers(self):
        eye_cascade = cv2.CascadeClassifier(self.pathA)
        face_cascade = cv2.CascadeClassifier(self.pathB)
        nose_cascade = cv2.CascadeClassifier(self.pathC)

##### list to hold all subject faces
        faces_holder = []

##### set images in a list & verify
        datasets = [self.pathF1, self.pathF2]

##### read image
        for image in datasets:
            image = cv2.imread(image)
                        
##### extract the faces area
            faces = face_cascade.detectMultiScale(image, 1.08, 5, minSize=(50, 50))           
            for left, top, width, height in faces:
                faces = image[top:top+height, left:left+width]
                cv2.rectangle(image, (left, top), (left+width, top+height),
                              (13, 230, 23), 2)

##### detect noses area from faces area
                noses = nose_cascade.detectMultiScale(faces, 1.15, 5,
                                                      flags=cv2.CASCADE_SCALE_IMAGE)
                for (nx, ny, nw, nh) in noses:
                    cv2.rectangle(faces, (nx, ny), (nx+nw, ny+nh), (255, 0, 255), 2)

##### detect eyes area from faces area
                eyes = eye_cascade.detectMultiScale(faces, 1.100, 4,
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
                for(ex, ey, ew, eh) in eyes:
                    cv2.rectangle(faces, (ex, ey), (ex+ew, ew+eh), (255, 255, 255), 2)

                if faces is not None:
                    # add face to list of faces_holder
                    faces_holder.append(image[top:top+height, left:left+width])

##### to analyze age, gender, emotion
            demography = DeepFace.analyze(image, ['emotion', 'age', 'gender'])

##### print total faces (text color:green)
            input_strings = "Age:{} Gender:{} Emotion:{} "
            cv2.putText(image, input_strings.format(str(demography["age"]),
                                                    demography["gender"],
                                                    demography["dominant_emotion"]),
                        (150, 300), cv2.FONT_HERSHEY_PLAIN, 2, (13, 230, 23), 2)

##### display an image window to show the image
            cv2.imshow('DeepFace Recognition ', image)
            cv2.waitKey()
            cv2.destroyAllWindows()
![Kim](https://user-images.githubusercontent.com/70878758/130473152-d568a3cd-936a-48dd-96a3-9a1a7a732781.png)
![Nam](https://user-images.githubusercontent.com/70878758/130473195-f0f8ced6-c000-4f9e-9492-6502a6443b72.png)


##### print the quantity of detected faces.
        print('Total detected faces:%d' % len(faces_holder))
![Result](https://user-images.githubusercontent.com/70878758/130473130-3e9bcba3-4051-401f-9b9c-d5788d038e5b.png)
