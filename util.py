import datetime
import json
import math
import shutil
import time
from mysql.connector import Error, MySQLConnection

import pymongo
from bson import ObjectId
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

from python_mysql_dbconfig import read_db_config

carpeta='./cloud/'
carpeta_standby='./standby/'
carpeta_reconocidos='./reconocidos/'
carpeta_sin_rostro='./sin_rostro/'
carpeta_fotos='./fotos/'
ALLOWED_EXTENSIONS = {'jpg'}

# Conexion Database MONGODB
cliente = pymongo.MongoClient("mongodb://localhost:27017/")
database = cliente["ibartiface"]
person_recognition = database["person_recognition"]
personas = database["persons"]
person_history = database["activity_history"]
categorias = database["category"]
estatus = database["conditions"]

def run_query_ibarti(query='', args=''):
    data = []
    if query:
        try:
            db_config = read_db_config()
            conn = MySQLConnection(**db_config)
            cursor = conn.cursor()         # Crear un cursor
            if args:
                cursor.execute(query, args)          # Ejecutar una consulta
            else:
                cursor.execute(query)          # Ejecutar una consulta con argumentos
            if query.upper().startswith('SELECT'):
                _data = cursor.fetchall()   # Traer los resultados de un select
                row_headers = [x[0] for x in cursor.description]  # Extraer las cabeceras de las filas
                for result in _data:
                    data.append(dict(zip(row_headers, result)))
                # data = json.dumps(data)
            else:
                conn.commit()              # Hacer efectiva la escritura de datos
        except Error as error:
            print(error)
        finally:
            cursor.close()                 # Cerrar el cursor
            conn.close()                   # Cerrar la conexión

    return data

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='auto', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        âââ <person1>/
        â   âââ <somename1>.jpeg
        â   âââ <somename2>.jpeg
        â   âââ ...
        âââ <person2>/
        â   âââ <somename1>.jpeg
        â   âââ <somename2>.jpeg
        âââ ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.50):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)], closest_distances[0][0][0]

def insertPerson(image, cedula, category, status, client):
    _persona = list(personas.find({'doc_id': cedula}))

    if _persona:
        persona = _persona[0]
        for encoding in image:
            persona['template_recognition'].append(encoding)
        myquery = {"_id":  _persona[0]['_id']}
        newvalues = {"$set": {'template_recognition': _persona[0]['template_recognition']}}
        personas.update_one(myquery, newvalues)
    else:
        persona = {
            "cod_person": "",
            "doc_id": cedula,
            "category": ObjectId(category),
            "status": ObjectId(status),
            "client": client,
            "template_recognition": image,
            "created_date": datetime.datetime.now()
        }
        persona = personas.insert_one(persona)

    return persona, len(_persona[0]['template_recognition']) if _persona else 1

def setHistory(data, metodo):
    try:
        # print('setHistory', data, metodo)
        if (metodo == 1):
            data["status"] = ObjectId("5dc4657cbf29c8d71fab3fce")
            data["create_date"] = str(datetime.datetime.now())
        elif (metodo == 2):
            data["status"] = ObjectId("5dcea3be058a6c5519c476d8")
            data["create_date"] = str(datetime.datetime.now())

        if(data["status"] and data["activity"]):
            if(estatus.find({"_id": data["status"]}).count() > 0):
                return person_history.insert(data)
            else:
                return False
        else:
            return False

    except pymongo.errors.PyMongoError as error:
        print(error)
        return error

def getEncode(dir):
    try:
        image_file = face_recognition.load_image_file(dir)
        face_locations = face_recognition.face_locations(image_file, number_of_times_to_upsample=2, model="hog")
    except Exception as e:
        print(e)
    rostros = []
    # Obtenemos los puntos faciales
    if (len(face_locations) == 0):
        return rostros
    elif(len(face_locations) > 1):
        for encodings in list(face_recognition.face_encodings(image_file, num_jitters=100)):
            insert = [str(i) for i in encodings]
            rostros.append(insert)
    elif len(list(face_recognition.face_encodings(image_file, num_jitters=100))) > 0:
        insert = [str(i) for i in list(face_recognition.face_encodings(image_file, num_jitters=100))[0]]
        rostros.append(insert)
    return rostros

def formatingFile(filename):
    print(filename)
    partes = filename.split('/')[-1].split('.')
    if len(partes) > 1:
        partes = partes[0].split('+')
        if len(partes) >= 4:
            propiedades = {}
            try:
                propiedades['fecha'] = str(
                    datetime.datetime.strptime(partes[2], '%Y-%m-%d').date())
            except:
                return False
            try:
                partes[3] = partes[3].replace('&', ':')
                partes[3] = partes[3].replace('_', ':')
                propiedades['hora'] = str(
                    datetime.datetime.strptime(partes[3], '%H:%M:%S:%f').time())
            except:
                return False
            propiedades['cliente'] = partes[0]
            propiedades['dispositivo'] = partes[1]
            # propiedades['nFoto'] = partes[2]
            propiedades['url'] = filename
            return propiedades
        else:
            return False
    else:
        return False

def moveToFotos(filename, cedula):
    ruta = carpeta_fotos+cedula+'/'
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

def insertarasistencia(equipo, auxidentificador, datos):
    datos["hora"] = datos["hora"][:-7]
    auxequipo = equipo
    auxhora = datos["hora"]
    auxfecha = datos["fecha"] + ' ' + datos["hora"]
    auxfechaservidor=time.strftime("%Y/%m/%d %H:%M:%S")
    auxchecktipo="E"
    auxtrama="Trama"
    auxevento="IDENTIFY"
    auxeventodata="FACIAL"
    auxorigen=equipo
    auxchequeado="N"
    # activot=fichaactiva(auxidentificador)
    query = "INSERT INTO ch_inout(huella,cod_dispositivo,fechaserver,fecha,hora,checktipo,trama,evento,eventodata," \
            "origen,checks) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    args = (auxidentificador,auxequipo,auxfechaservidor,auxfecha,auxhora,auxchecktipo,auxtrama,auxevento,auxeventodata,
            auxorigen,auxchequeado)

    return run_query_ibarti(query, args)

def consultarasistencia(fecha_desde, fecha_hasta):
    query = "SELECT v_ch_identify.codigo, IFNULL(ficha.cedula, 'SIN CEDULA') cedula , " \
            "IFNULL(ficha.cod_ficha, 'SIN FICHA') cod_ficha, IFNULL(CONCAT(ficha.apellidos,' ',ficha.nombres), " \
            "v_ch_identify.huella) ap_nombre , v_ch_identify.cod_dispositivo,  clientes_ubicacion.codigo cod_ubicacion," \
            " clientes_ubicacion.descripcion ubicacion, clientes.codigo cod_cliente, clientes.nombre cliente, " \
            "DATE_FORMAT(v_ch_identify.fechaserver, '%Y-%m-%d %h:%i:%s') fechaserver, DATE_FORMAT(v_ch_identify.fecha, '%Y-%m-%d') fecha, " \
            "DATE_FORMAT(v_ch_identify.hora, '%h:%i:%s') hora, " \
            "'SI' vetado FROM v_ch_identify LEFT JOIN ficha ON v_ch_identify.cedula = ficha.cedula AND " \
            "ficha.cod_ficha_status = 'A', clientes_ub_ch, clientes, clientes_ubicacion " \
            "WHERE DATE_FORMAT(v_ch_identify.fecha, '%Y-%m-%d') BETWEEN '"+fecha_desde+"' AND '"+fecha_hasta+"' " \
            "AND v_ch_identify.cod_dispositivo = clientes_ub_ch.cod_capta_huella " \
            "AND clientes_ub_ch.cod_cl_ubicacion = clientes_ubicacion.codigo " \
            "AND clientes_ubicacion.cod_cliente = clientes.codigo " \
            "AND ficha.cod_ficha IN (SELECT clientes_vetados.cod_ficha FROM  clientes_vetados " \
            "WHERE clientes_vetados.cod_cliente = clientes_ubicacion.cod_cliente " \
            "AND clientes_vetados.cod_ubicacion = clientes_ubicacion.codigo " \
            "AND ficha.cod_ficha = clientes_vetados.cod_ficha) AND v_ch_identify.eventodata = 'FACIAL' " \
            "UNION ALL " \
            "SELECT v_ch_identify.codigo, IFNULL(ficha.cedula, 'SIN CEDULA') cedula , " \
            " IFNULL(ficha.cod_ficha, 'SIN FICHA') cod_ficha, IFNULL(CONCAT(ficha.apellidos,' ',ficha.nombres)," \
            " v_ch_identify.huella) ap_nombre , v_ch_identify.cod_dispositivo,  " \
            "clientes_ubicacion.codigo cod_ubicacion, clientes_ubicacion.descripcion ubicacion," \
            " clientes.codigo cod_cliente, clientes.nombre cliente, " \
            "DATE_FORMAT(v_ch_identify.fechaserver, '%Y-%m-%d %h:%i:%s') fechaserver, " \
            "DATE_FORMAT(v_ch_identify.fecha, '%Y-%m-%d') fecha, DATE_FORMAT(v_ch_identify.hora, '%h:%i:%s') hora, " \
            "'NO' vetado FROM v_ch_identify LEFT JOIN ficha ON v_ch_identify.cedula = ficha.cedula " \
            "AND ficha.cod_ficha_status = 'A', clientes_ub_ch, clientes, clientes_ubicacion " \
            "WHERE DATE_FORMAT(v_ch_identify.fecha, '%Y-%m-%d') BETWEEN '"+fecha_desde+"' AND '"+fecha_hasta+"' " \
            "AND v_ch_identify.cod_dispositivo = clientes_ub_ch.cod_capta_huella " \
            "AND clientes_ub_ch.cod_cl_ubicacion = clientes_ubicacion.codigo " \
            "AND clientes_ubicacion.cod_cliente = clientes.codigo " \
            "AND ficha.cod_ficha NOT IN (SELECT clientes_vetados.cod_ficha FROM  clientes_vetados " \
            "WHERE clientes_vetados.cod_cliente = clientes_ubicacion.cod_cliente " \
            "AND clientes_vetados.cod_ubicacion = clientes_ubicacion.codigo " \
            "AND ficha.cod_ficha = clientes_vetados.cod_ficha) AND v_ch_identify.eventodata = 'FACIAL' " \
            "GROUP BY cedula, DATE_FORMAT(fecha, '%Y-%m-%d') ORDER BY fecha ASC, cedula"

    return run_query_ibarti(query)
