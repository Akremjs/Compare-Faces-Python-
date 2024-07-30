import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime, timedelta

# Répertoire contenant les images des ouvriers
path = 'persons'
images = []
classNames = []
personList = os.listdir(path)

# Charger les images et les noms
for cl in personList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img_rgb)
        if encodes:
            encodeList.append(encodes[0])
    return encodeList

encodeListKnown = findEncodings(images)
print('Encodings complete.')

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)

# Dictionnaire pour suivre les temps d'entrée/sortie
current_status = {}
# Dictionnaire pour suivre le dernier temps de capture
last_capture_time = {}
# Délai de temporisation en secondes pour ignorer les détections répétées
ignore_delay = 30

# Fonction pour sauvegarder dans le fichier Excel
def save_to_excel(name, entry_time=None, exit_time=None, duration=None):
    file_path = fr'C:/Users/Akrem/Desktop/Nouveau dossier2/{name}_log.xlsx'
    
    try:
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
        else:
            df = pd.DataFrame(columns=['Date', 'Entry Time', 'Exit Time', 'Total Duration'])
        
        if entry_time:
            entry_date, entry_time_only = entry_time.split()
            new_row = pd.DataFrame([{'Date': entry_date, 'Entry Time': entry_time_only, 'Exit Time': None, 'Total Duration': None}])
        elif exit_time:
            exit_date, exit_time_only = exit_time.split()
            new_row = pd.DataFrame([{'Date': exit_date, 'Entry Time': None, 'Exit Time': exit_time_only, 'Total Duration': duration}])
        
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(file_path, index=False)
    except PermissionError:
        print(f"Permission denied: {file_path}. Please check if the file is open or if you have write permissions.")
    except Exception as e:
        print(f"An error occurred: {e}")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    img_resized = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(img_rgb)
    encodeCurFrame = face_recognition.face_encodings(img_rgb, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            current_time_dt = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
            
            if name not in current_status:
                # Ouvrier détecté pour la première fois
                current_status[name] = {'Entry Time': current_time}
                last_capture_time[name] = current_time_dt
                save_to_excel(name, entry_time=current_time)
            else:
                last_time = last_capture_time.get(name, None)
                if last_time and (current_time_dt - last_time).total_seconds() >= ignore_delay:
                    entry_time = current_status[name].get('Entry Time', None)
                    if entry_time:
                        entry_time_dt = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
                        exit_time = current_time
                        duration = current_time_dt - entry_time_dt
                        current_status[name] = {'Entry Time': None, 'Exit Time': exit_time}
                        save_to_excel(name, exit_time=exit_time, duration=str(duration))
                    
                    last_capture_time[name] = current_time_dt

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
