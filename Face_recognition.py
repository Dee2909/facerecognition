import face_recognition
import os, sys
import cv2
import numpy as np
import math
from datetime import datetime
path = 'Training_images'
def face_confidence(face_distance, face_match_threshold=0.42):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        mylist=[]
        mylist=os.listdir(path)
        image=[]
        for cl in mylist:
            curImg = cv2.imread(f'{path}/{cl}')
            image.append(curImg)
            self.known_face_names.append(os.path.splitext(cl)[0])
        for img in image:    
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            self.known_face_encodings.append(encode)
            
        print(self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()

            
            if self.process_current_frame:
                
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                
                rgb_small_frame = small_frame[:, :, ::-1]

                
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'

                    
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    now = datetime.now()
                    dtString = now.strftime('%H%M')
            #cv2.imwrite('Name.jpg',img)            
                    img_name = "name"+str(dtString)+".jpg"
            #img_name="Name.jpg"
                    cv2.imwrite(img_name,frame)
                    import json
                    import requests
                    headers = {"Authorization":"Bearer ya29.a0AVvZVsrTqXm_qn3m47qSiXMlsswchTHXA_YsKz-aNAVCprqn-BrM6liENfMiFIr8JHXoyd0cViBaH_aWqj0O9Pnu4VWA5Nz4dJlpVyjrP-NsPd7kkeVb8ORbyungbQTIRbqIQbarhcswBOqCTjLktP81SW66aCgYKAeUSARASFQGbdwaIJaTMWjE84RId5I_ILmI8Bg0163"}
                    para = {"name":"unf"+str(dtString)+".jpeg","parents":["1OPbUigTxYWYfHk2rId0E2Ys9H0bvxGA4"]}
                    files = {'data':('metadata',json.dumps(para),'application/json;charset=UTF-8'),'file':open('./'+img_name,'rb')}
                    r = requests.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",headers=headers,files=files)
                    from twilio.rest import Client
                    account_sid = "AC52058c0fc48563cae065d5047a81af93"
                    auth_token ='a962a2aea77ab008b8690b2e4f09e467'
                    ac="AC0ac3a830f61004db6bf5f21d824fcaa0"
                    at='07d72c8e17d0d18e7170d889d9da70e9'
                    client = Client(account_sid, auth_token)
                    c=Client(ac,at)
                    for i in range(1):
                        message = client.messages.create(to='+919566972416',from_="+15802078712",body="https://drive.google.com/drive/folders/1OPbUigTxYWYfHk2rId0E2Ys9H0bvxGA4?usp=sharing")
                        #message = c.messages.create(to='+917200434425',from_="+16087361885",body="https://drive.google.com/drive/folders/1OPbUigTxYWYfHk2rId0E2Ys9H0bvxGA4?usp=sharing")
                        continue
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Create the frame with the name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
