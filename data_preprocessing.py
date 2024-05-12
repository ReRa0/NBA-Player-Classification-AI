from PIL import Image
import face_recognition
from instagram_crawling import player_list



for i in range(len(player_list)):
    player_list[i]['full_name']

'''
def 선수 얼굴만 따서 저장(선수 이름):
    for문을 돌려서 선수 이름을 받아서(player_list[i]['full_name']) ======>>>>>>> f'{path}\{tag}{i}.jpg
    
'''
#선수 이름 


#학습할 대가리
training_image_path = "test.image2.png"

#대가리 찾을 사진
target_image_path = "test.image7.png"

# 학습ㅋ
def train_face(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    
    if len(face_encodings) > 0:
        face_encoding = face_encodings[0]
        print("얼굴 학습완료다 븅아~")
        return face_encoding
    else:
        print("얼굴 학습실패 ㅋ ㅄ")
        return None

def recognize_and_extract_faces(training_face_encoding, target_image_path):
    target_image = face_recognition.load_image_file(target_image_path)
    face_locations = face_recognition.face_locations(target_image)
    face_encodings = face_recognition.face_encodings(target_image, face_locations)

    matched_faces = []

    for face_location, face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([training_face_encoding], face_encoding)
        if matches[0]:
            top, right, bottom, left = face_location
            face_image = target_image[top:bottom, left:right]
            matched_faces.append(face_image)

    return matched_faces


trained_face_encoding = train_face(training_image_path)

if trained_face_encoding is not None:

    matched_faces = recognize_and_extract_faces(trained_face_encoding, target_image_path)
    
    if matched_faces:
        # 매칭 대가리 있으면 출력
        for i, face_image in enumerate(matched_faces):
            pil_image = Image.fromarray(face_image)
            pil_image.show()
    else:
        print("매칭된 얼굴이 없어 ㅄ아")
else:
    print("얼굴 학습이 실패함 얼굴인식 불가능 ㅋ")