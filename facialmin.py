import threading
import shutil
from itertools import groupby

import numpy as np
import face_recognition
import subprocess
from flask import Flask, request, redirect, jsonify
from PIL import Image
from flask_cors import CORS, cross_origin
import math
import os
import os.path
from util import train, predict, insertPerson, ALLOWED_EXTENSIONS, carpeta, carpeta_standby, carpeta_fotos, personas, \
    getEncode, formatingFile, moveToFotos, insertarasistencia, carpeta_reconocidos, carpeta_sin_rostro

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
cors = CORS(app, resources={r"/": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
servidor='192.168.33.74'
servidorlocal=servidor
codigoequipo=6666
lineacomando='curl -F "file=@1.jpg" http://192.168.33.74:5001'

app = Flask(__name__)

def puertolibre(numerouno):
    for puerto in range(int(numerouno)):
        try:
            socket.connect((host, puerto))
            return True
            socket.close()
        except :
            print("Puerto "+str(puerto)+" cerrado.")
            return False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def leerBandera():
    archivo = open('band.txt', 'r')
    band = archivo.read(1)
    archivo.close()
    return str(band) == 'T'

def cambiarBandera(val):
    archivo = open('band.txt', 'w')
    archivo.write(val)
    archivo.close()

def hayImagenesParaProcesar():
    imagenes = os.listdir(carpeta)
    imagenes_a_procesar = []
    if len(imagenes) > 0:
        imagenes_a_procesar = filterImagenesSinProcesar(imagenes)
    return len(imagenes_a_procesar) > 0

def getImagenesParaProcesar():
    nro_imagenes = 10 if len(threading.enumerate()) == 5 else 5
    imagenes = os.listdir(carpeta)
    imagenes_a_procesar = filterImagenesSinProcesar(imagenes)[:nro_imagenes]
    _imagenes_a_procesar = []
    for idx in range(len(imagenes_a_procesar)):
        new_nombre = imagenes_a_procesar[idx].replace('.jpg', '+P.jpg')
        os.rename(carpeta + imagenes_a_procesar[idx], carpeta + new_nombre)
        _imagenes_a_procesar.append(new_nombre)
    return _imagenes_a_procesar

def filterImagenesSinProcesar(path_imagenes):
    return list(filter(lambda img: '.jpg' in img and '+P.jpg' not in img, path_imagenes))

def _reconocimiento(imagenes):
    try:
        for img in imagenes:
            imagenRecognition(carpeta + img)
    except Exception as e:
        print('ERROR_RECONOCIMIENTO', e, imagenes)
        cambiarBandera('F')

def procesarImagenes():
    terminar = False
    while len(threading.enumerate()) >= 5 and not terminar:
        if len(threading.enumerate()) <= 6:
            if hayImagenesParaProcesar():
                begin = 0
                end = 5
                imagenes = getImagenesParaProcesar()
                while begin < len(imagenes):
                    hiloSecundario = threading.Thread(target=_reconocimiento, args=(imagenes,))
                    hiloSecundario.start()
                    begin += 5
                    end += 5
            else:
                terminar = True
    cambiarBandera('F')

def eliminarImagen(imagen):
    os.remove(imagen)

def moveStandBy(filename, cod_cliente):
    ruta = carpeta_standby+cod_cliente+'/'
    try:
        os.stat(carpeta_standby)
    except:
        os.mkdir(carpeta_standby)
    try:
        fileName = filename.split('/')[-1]
        try:
          os.stat(ruta)
        except:
          os.mkdir(ruta)

        shutil.move(filename, ruta)
    except:
        return False
    return ruta, fileName

def moveToReconocidos(filename, cod_cliente):
    ruta = carpeta_reconocidos+cod_cliente+'/'
    try:
        os.stat(carpeta_reconocidos)
    except:
        os.mkdir(carpeta_reconocidos)
    try:
        fileName = filename.split('/')[-1]
        try:
          os.stat(ruta)
        except:
          os.mkdir(ruta)

        shutil.move(filename, ruta)
    except:
        return False
    return ruta+fileName

def moveToSinRostro(filename, cod_cliente):
    ruta = carpeta_sin_rostro+cod_cliente+'/'
    try:
        os.stat(carpeta_sin_rostro)
    except:
        os.mkdir(carpeta_sin_rostro)
    try:
        fileName = filename.split('/')[-1]
        try:
          os.stat(ruta)
        except:
          os.mkdir(ruta)

        shutil.move(filename, ruta)
    except:
        return False
    return ruta+fileName

def imagenRecognition(imgD):
    datos = formatingFile(imgD)
    result = None
    if datos:
        datos['url'] = imgD
        template = getEncode(imgD)
        personas = searchPersons()
        if len(personas['doc_ids']) > 0:
            result = reconocimiento(personas, template, datos)
        else:
            datos['url'] = moveStandBy(datos['url'], datos['cliente'])
            print('movido a standby No Ids', datos['url'], datos['cliente'])
            result = {'descripcion': 'movido a standby No Ids', 'url': datos['url'], 'cliente': datos['cliente']}
    return result
def searchPersons():
    retornar = {
        "templates": [],
        "doc_ids": []
    }
    templates = list(personas.find(
        {}, {'doc_id', 'template_recognition'}))
    for elementos in templates:
        for (index, elemento) in enumerate(elementos['template_recognition']):
            retornar["templates"].append([float(i) for i in elemento])
            retornar["doc_ids"].append(elementos['doc_id'])
    return (retornar)

def face_multiple(matches, count):
    faces = []
    for (index, face) in enumerate(matches):
        if face:
            count += 1
            faces.append(index)
    return count, faces

def reconocimiento(search, templete, datos):
    if (len(templete) > 0 and len(templete[0]) > 0):
        templete = [float(i) for i in templete[0]]
        templete = np.asarray(templete)
        searchs = np.asarray(search['templates'])
        matches = face_recognition.compare_faces(searchs, templete, tolerance=0.40)
        face_distances = face_recognition.face_distance(searchs, templete)
        count = 0
        prueba, prueba1 = face_multiple(matches, count)

        # for pos in prueba1:
        #     print('reconocido', prueba, pos, search['doc_ids'][pos], face_distances[pos])

        documentos = []
        promedios = []
        for doc, data in groupby(list(search['doc_ids'])):
            m = [face_distances[i] for i, x in enumerate(list(search['doc_ids'])) if x == str(doc)
                 and face_distances[i] < 0.55]
            if len(m) > 0:
                documentos.append(doc)
                promedios.append((sum(m)/len(m)))
        print('CANTFD', len(list(face_distances)))
        first_match_index = np.where(min(list(face_distances)) == face_distances)[0][0]
        doc_id = search['doc_ids'][first_match_index]
        distance = face_distances[first_match_index]

        doc_id_min_prom = documentos[promedios.index(min(promedios))] if documentos else 'N/A'
        distance_min_prom = min(promedios) if promedios else 'N/A'
        # print('MIN', datos['url'], doc_id, distance)
        # print('MIN-PROM', datos['url'], documentos[promedios.index(min(promedios))] if documentos else 'N/A',
        #       min(promedios) if promedios else 'N/A')

        if True in matches:
            distance = distance
            datos["doc_id"] = doc_id
            datos["distance"] = distance
            print('CNN', datos['url'], doc_id, distance)
            # setHistory({
            #     "activity": "Reconocimiento ("+name+")",
            #     "json": datos
            # }, 1)
            insertarasistencia(datos['dispositivo'], doc_id, datos)
            datos['url'] = moveToReconocidos(datos['url'], datos['cliente'])
            return {
                'descripcion': 'reconocido por cnn', 'url': datos['url'], 'cliente': datos['cliente'],
                'distance': distance, 'doc_id': doc_id, 'distance_min_prom':  distance_min_prom,
                'doc_id_min_prom': doc_id_min_prom
            }
        else:
            if os.path.exists('modeloknn.clf'):
                predictions, distance_knn = predict(datos['url'], model_path="modeloknn.clf")
                print(distance_knn)
                for name, (top, right, bottom, left) in predictions:
                    if str(name) != 'unknown' and name == documentos[promedios.index(min(promedios))]:
                        datos["doc_id"] = str(name)
                        resp, template_recognition_lentgh = insertPerson(
                            getEncode(datos['url']),
                            str(name),
                            None,
                            None,
                            datos['cliente']
                        )
                        ruta_foto = datos['url']
                        new_nombre = ruta_foto.replace('.jpg', '-'+str(template_recognition_lentgh)+'.jpg')
                        os.rename(ruta_foto, new_nombre)
                        moveToFotos(new_nombre, str(name))
                        print('KNN', new_nombre, name, distance, distance_knn)
                        insertarasistencia(datos['dispositivo'], str(name), datos)
                        datos['url'] = moveToReconocidos(new_nombre, datos['cliente'])
                        return {
                            'descripcion': 'reconocido por knn', 'url': new_nombre, 'cliente': datos['cliente'],
                            'distance': distance, 'distance_knn': distance_knn, 'doc_id_knn': name, 'doc_id': doc_id,
                            'distance_min_prom':  distance_min_prom, 'doc_id_min_prom': doc_id_min_prom
                        }
                    else:
                        datos['url'] = moveStandBy(datos['url'], datos['cliente'])
                        print('movido a standby', datos['url'], datos['cliente'], distance, distance_knn, doc_id)
                        return {
                            'descripcion': 'movido a standby', 'url': datos['url'], 'cliente': datos['cliente'],
                            'distance': distance, 'distance_knn': distance_knn, 'doc_id': doc_id,
                            'doc_id_knn': name
                        }

            # setHistory({
            #     "activity": "Reconocimiento ("+name+")",
            #     "json": datos
            # }, 2)
    else:
        datos['url'] = moveToSinRostro(datos['url'], datos['cliente'])
        print('movido a Sin Rostro', datos['url'], datos['cliente'])
        return {'descripcion': 'movido a sin Rostro', 'url': datos['url'], 'cliente': datos['cliente']}

def face_distance_to_conf(face_distance, face_match_threshold=0.3):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def face_distance_to_conf_1(face_distance, face_match_threshold, name):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        linear_val_f = linear_val + face_distance
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        linear_val_f = linear_val + face_distance
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

@app.route('/', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def upload_image():
    # Chequea la imagen que llego

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file'] #file almacena la imagen

        if file.filename == '':
            return redirect(request.url)

        if allowed_file(file.filename):
            # valida la imagen y la envia a reconocimiento facial y la devuelve en result
            file.filename = str(file.filename).replace('&', ':')
            image = Image.open(file)
            if not os.path.exists(carpeta+str(file.filename)) and \
                    not os.path.exists(carpeta+str(file.filename).replace('.jpg', '+P.jpg')):
                # cv2.imwrite(carpeta + str(file.filename), image) #Guardar imagen en './cloud/nombreDeImagen'
                image.save(carpeta + str(file.filename))
            result = imagenRecognition(carpeta + str(file.filename))
            # band = leerBandera()
            # if not band or len(threading.enumerate()) < 5:
            #     cambiarBandera('T')
            #     hilo1 = threading.Thread(name='hilo1', target=procesarImagenes)
            #     hilo1.start()
            return jsonify(result)
    return 'T'

if __name__ == "__main__":
    print("Scaneando Puerto....")
    print("******Cargando Datos de Servidor y Puerto***************")

    configuracion=[]
    with open("ibartir.txt") as f:
        for linea in f:
            configuracion.append(linea)
    f.close()

    mi_puerto = int(configuracion[0])
    mi_server = configuracion[1]
    codigoimagenl = configuracion[2]
    # try:
    #     #verificar si el puerto esta abierto
    #     resultado =subprocess.check_output(lineacomando, shell=True)
    #     print(resultado)
    # except subprocess.CalledProcessError as e:
    #     print(e.output)
    app.run(host='192.168.33.76', port=mi_puerto, debug=False)
