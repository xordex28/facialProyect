import face_recognition
import click
from PIL import Image, ImageDraw
import pymongo
from bson.objectid import ObjectId
import numpy as np
import datetime
import math
import time
import shutil


cliente = pymongo.MongoClient("mongodb://localhost:27017/")
database = cliente["ibartiface"]

person_recognition = database["person_recognition"]
person_history = database["activity_history"]
status = database["status"]
category = database["category"]


image_file = './img/desconocidos/0001+0001+6+2019-11-08+15&14&59.jpg'
#image_file = './img/desconocidos/0001+0001+5+2019-11-08+15&14&59.jpg'
#image_file_2 = face_recognition.load_image_file('./img/conocidos/eliezer.jpg')
#image_file_3 = face_recognition.load_image_file('./img/conocidos/8668069.jpg')


def moveStandBy(filename):
    try:
        fileName = filename.split('/')[-1]
        shutil.move(filename, './img/standby/'+fileName)
    except:
        return False
    return './img/standby/'+fileName


def setHistory(data, metodo):
    try:
        if (metodo == 1):
            data["status"] = ObjectId("5dc4657cbf29c8d71fab3fce")
            data["create_date"] = str(datetime.datetime.now())
        elif (metodo == 2):
            data["status"] = ObjectId("5dcea3be058a6c5519c476d8")
            data["create_date"] = str(datetime.datetime.now())

        if(data["status"] and data["activity"]):
            if(status.find({"_id": data["status"]}).count() > 0):
                return person_history.insert(data)
            else:
                return False
        else:
            return False

    except pymongo.errors.PyMongoError as error:
        print(error)
        return error


def formatingFile(filename):
    partes = filename.split('/')[-1].split('.')
    if len(partes) > 1:
        partes = partes[0].split('+')
        if len(partes) >= 5:
            propiedades = {}
            try:
                propiedades['fecha'] = str(
                    datetime.datetime.strptime(partes[3], '%Y-%m-%d').date())
            except:
                return False
            try:
                partes[4] = partes[4].replace('&', ':')
                propiedades['hora'] = str(
                    datetime.datetime.strptime(partes[4], '%H:%M:%S').time())
            except:
                return False
            propiedades['cliente'] = partes[0]
            propiedades['dispositivo'] = partes[1]
            propiedades['nFoto'] = partes[2]
            return propiedades
        else:
            return False
    else:
        return False


def imagenRecognition(imgD):
    datos = formatingFile(imgD)

    if(datos):
        # your code
        datos['url'] = imgD
        template = getEncode(imgD)
        personas = searchPersons()
        reconocimiento(personas, template, datos)
    # print(datos)


def searchPersons():
    retornar = {
        "templates": [],
        "names": []
    }
    templates = list(person_recognition.find(
        {}, {'first_name', 'template_recognition'}))
    for elementos in templates:
        for (index, elemento) in enumerate(elementos['template_recognition']):
            retornar["templates"].append([float(i) for i in elemento])
            retornar["names"].append(elementos['first_name'])
    return (retornar)


def insertPerson(image):
    persona = {
        "cod_person": "0006",
        "doc_id": "2222222",
        "first_name": "pedro",
        "last_name": "perez",
        "status": ObjectId("5dc43db765d6db98e7ce67e1"),
        "client": ObjectId("5dc43ee965d6db98e7ce6860"),
        "template_recognition": image
    }
    resp = person_recognition.insert_one(persona)
    print(resp)


def getEncode(dir):
    # Cargo la imagen
    image_file = face_recognition.load_image_file(dir)
    face_locations = face_recognition.face_locations(image_file, model='hog')
    rostros = []
    # Obtenemos los puntos faciales
    if (len(face_locations) == 0):
        return rostros
    elif(len(face_locations) > 1):
        for (index, value) in enumerate(face_locations):
            insert = [str(i) for i in list(
                face_recognition.face_encodings(image_file, num_jitters=5)[index])]
            rostros.append(insert)
    else:
        insert = [str(i) for i in list(
            face_recognition.face_encodings(image_file, num_jitters=5)[0])]
        rostros.append(insert)
    return rostros


def face_multiple(matches, count):
    faces = []
    for (index, face) in enumerate(matches):
        if face:
            count += 1
            faces.append(index)
    return count, faces


def reconocimiento(search, templete, datos):
    templete = [float(i) for i in templete[0]]
    templete = np.asarray(templete)
    searchs = np.asarray(search['templates'])
    matches = face_recognition.compare_faces(searchs, templete, tolerance=0.5)
    face_distances = face_recognition.face_distance(searchs, templete)
    name = "NOT FOUND"
    count = 0
    print(face_distances)
    prueba, prueba1 = face_multiple(matches, count)

    print(prueba, prueba1)

    if True in matches:
        first_match_index = matches.index(True)
        name = search['names'][first_match_index]
        distance = face_distances[first_match_index]
        datos["name"] = name
        datos["distance"] = distance
        setHistory({
            "activity": "Reconocimiento ("+name+")",
            "json": datos
        }, 1)
    else:
        newLocalition = moveStandBy(datos['url'])
        datos['url'] = newLocalition
        setHistory({
            "activity": "Reconocimiento ("+name+")",
            "json": datos
        }, 2)


def face_distance_to_conf(face_distance, face_match_threshold=0.3):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        print(linear_val)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        print(linear_val + ((1.0 - linear_val) *
                            math.pow((linear_val - 0.5) * 2, 0.2)))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))


def face_distance_to_conf_1(face_distance, face_match_threshold, name):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        linear_val_f = linear_val + face_distance
        print(linear_val)
        print(face_distance)
        print(name)
        print(linear_val_f)
        print('if')
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        linear_val_f = linear_val + face_distance
        print(linear_val)
        print(face_distance)
        print(name)
        print(linear_val_f)
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def category():
    sta = list(category.find())
    print(sta)


#start_time = time.time()
#imagenRecognition(image_file)
#elapsed_time = time.time() - start_time
#print(str(elapsed_time))
# insertPerson(getEncode(image_file))
# face_distance_to_conf(0.5772279690431007,0.6,'nose')
category()
