import os
import shutil
import pymongo
from bson.objectid import ObjectId
import datetime
import face_recognition
import math
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS, cross_origin
from util import train, insertPerson, ALLOWED_EXTENSIONS, carpeta_standby, carpeta_fotos, personas, categorias, estatus, \
    formatingFile, moveToFotos, consultarasistencia

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
cors = CORS(app, resources={r"/": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
servidor='192.168.33.14'
servidorlocal=servidor
carpeta='./cloud/'
carpeta_standby='./standby/'
carpeta_fotos='./fotos/'
# carpeta_standby = '/opt/lampp/htdocs/facial/assets/standby/'
codigoequipo=6666

# Conexion Database MONGODB
cliente = pymongo.MongoClient("mongodb://localhost:27017/")
database = cliente["ibartiface"]
person_recognition = database["person_recognition"]
personas = database["persons"]
person_history = database["activity_history"]
categorias = database["category"]
estatus = database["conditions"]

app = Flask(__name__)

def puertolibre(numerouno):
    for puerto in range(int(numerouno)):
        try:
            socket.connect((host, puerto))
            print("Puerto "+str(puerto)+" abierto")
            return True
            socket.close()

        except :
            print("Puerto "+str(puerto)+" cerrado.")
            return False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def eliminarImagen(imagen):
    os.remove(imagen)

def searchPersons():
    retornar = {
        "templates": [],
        "doc_ids": []
    }
    # print('searchPersons', 'REtornar', retornar)
    templates = list(personas.find(
        {}, {'doc_id', 'template_recognition'}))
    # print('searchPersons', 'TEmplate', templates)
    for elementos in templates:
        for (index, elemento) in enumerate(elementos['template_recognition']):
            retornar["templates"].append([float(i) for i in elemento])
            retornar["doc_ids"].append(elementos['doc_id'])
    return (retornar)

def getEncode(dir):
    # print('getEncode', dir)
    # Cargo la imagen
    try:
        image_file = face_recognition.load_image_file(dir)
        face_locations = face_recognition.face_locations(image_file, number_of_times_to_upsample=2, model="cnn")
    except Exception as e:
        print(e)
    # print('face_locations-len', len(face_locations))
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
        insert = [str(i) for i in list(face_recognition.face_encodings(image_file, num_jitters=5))[0]]
        rostros.append(insert)
    return rostros

def face_multiple(matches, count):
    faces = []
    for (index, face) in enumerate(matches):
        if face:
            count += 1
            faces.append(index)
    return count, faces

def face_distance_to_conf(face_distance, face_match_threshold=0.3):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        # print('face_distance_to_conf', linear_val)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        # print('face_distance_to_conf', linear_val + ((1.0 - linear_val) *
        #                     math.pow((linear_val - 0.5) * 2, 0.2)))
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

@app.route('/insert-person/', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def insert():
    data = request.json

    resp, template_recognition_lentgh = insertPerson(
        getEncode(carpeta_standby+data['cliente']+'/'+data['foto']),
        data['cedula'],
        data['category'],
        data['status'],
        data['cliente']
    )
    ruta_foto = carpeta_standby+data['cliente']+'/'+data['foto']
    new_nombre = ruta_foto.replace('.jpg', '-'+str(template_recognition_lentgh-1)+'.jpg')
    os.rename(ruta_foto, new_nombre)
    moveToFotos(new_nombre, data['cedula'])
    train(carpeta_fotos, model_save_path="modeloknn.clf", n_neighbors=1)
    # eliminarImagen(carpeta_standby+data['cliente']+'/'+data['foto'])
    return data

@app.route('/category/', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def category():
    data = request.json
    if request.method == 'GET':
        ct = []
        for c in list(categorias.find()):
            c['_id'] = str(c['_id'])
            ct.append(c)
        return jsonify(ct)

    elif request.method == 'POST':
        categorias.insert_one({'descripcion': data['descripcion']})
        return jsonify(data)

@app.route('/category/<id>/', methods=['PUT', 'DELETE'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def categoryEdit(id):
    data = request.json
    if request.method == 'PUT':
        myquery = {"_id": ObjectId(id)}
        newvalues = {"$set": {'descripcion': data['descripcion']}}
        categorias.update_one(myquery, newvalues)
        return jsonify(data)

    elif request.method == 'DELETE':
        categorias.delete_one({"_id": ObjectId(id)})
        return jsonify(data)

@app.route('/status/', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def status():
    data = request.json
    if request.method == 'GET':
        ct = []
        filtros = {}
        if request.args.get('category'):
            filtros['category'] = ObjectId(request.args.get('category'))
        for c in list(estatus.find(filtros)):
            c['_id'] = str(c['_id'])
            categ = list(categorias.find({'_id': c['category']}))
            c['category'] = categ[0]['descripcion'] if len(categ) > 0 else str(c['category'])
            ct.append(c)
        return jsonify(ct)

    elif request.method == 'POST':
        estatus.insert_one({'category': ObjectId(data['category']), 'descripcion': data['descripcion']})
        return jsonify(data)

@app.route('/status/<id>/', methods=['PUT', 'DELETE'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def statusEdit(id):
    data = request.json
    if request.method == 'PUT':
        myquery = {"_id": ObjectId(id)}
        newvalues = {"$set": {'descripcion': data['descripcion']}}
        estatus.update_one(myquery, newvalues)
        return jsonify(data)

    elif request.method == 'DELETE':
        estatus.delete_one({"_id": ObjectId(id)})
        return jsonify(data)

@app.route('/status/', methods=['GET',])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def getstatus():
    data = request.json
    if request.method == 'GET':
        pers = []
        filtros = {}

        if request.args.get('category'):
            filtros['category'] = ObjectId(request.args.get('category'))
        if request.args.get('status'):
            filtros['status'] = ObjectId(request.args.get('status'))

        for c in list(personas.find(filtros)):
            c['_id'] = str(c['_id'])
            categ = list(categorias.find({'_id': c['category']}))
            c['category'] = categ[0]['descripcion'] if len(categ) > 0 else str(c['category'])
            stat = list(categorias.find({'_id': c['status']}))
            c['status'] = categ[0]['descripcion'] if len(stat) > 0 else str(c['status'])
            pers.append(c)

        return jsonify(pers)

@app.route('/standby/<cliente>/', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def standby(cliente):
    _imagenes = []
    if request.method == 'GET':
        imagenes = os.listdir(carpeta_standby+cliente+'/')
        for img in imagenes:
            _imagenes.append(formatingFile(img))
        return jsonify(_imagenes)

@app.route('/standby/', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def standby_any():
    _imagenes = []
    if request.method == 'GET':
        directorios = os.listdir(carpeta_standby)
        for cliente in directorios:
            imagenes = os.listdir(carpeta_standby+cliente+'/')
            if imagenes:
                for img in imagenes:
                    _imagenes.append(formatingFile(img))

        return jsonify(_imagenes)


@app.route('/standby/<cliente>/<image>', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def send_image(cliente, image):
    return send_from_directory('standby', cliente+'/'+image)

# @app.route('/persons/', methods=['GET'])
# @cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
# def get_personas():
#     pers = list(personas.aggregate([
#         {
#             "$project": {"_id": 1, "doc_id": 1, "client": 1, "category": 1, "status": 1}
#         },
#         {
#             "$lookup": {
#                 "from": 'category',
#                 # "pipeline": [
#                 #     { "$project": {"_id": "$_id", "descripcion": "$descripcion"} }
#                 # ],
#                 "localField": 'category',
#                 "foreignField": '_id',
#                 "as": 'category'
#             }
#         },
#         {
#             "$lookup": {
#                 "from": 'conditions',
#                 "localField": 'status',
#                 "foreignField": '_id',
#                 "as": 'status'
#             }
#         }
#     ]))
#
#     return jsonify(pers)

@app.route('/delete-standby/<cliente>/', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def delete_standby(cliente):
    data = request.json
    eliminarImagen(carpeta_standby+cliente+'/'+data['foto'])
    return jsonify(data)

@app.route('/persons/', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def get_personas():
    p = []
    for _p in list(personas.find({}, {'_id', 'doc_id', 'category', 'status', 'client'})):
        categ = list(categorias.find({'_id': _p['category']}))
        _p['category'] = categ[0]['descripcion'] if len(categ) > 0 else str(_p['category'])
        stat = list(estatus.find({'_id': _p['status']}))
        _p['status'] = stat[0]['descripcion'] if len(stat) > 0 else str(_p['status'])
        _p['_id'] = str(_p['_id'])
        p.append(_p)

    return jsonify(p)

@app.route('/persons/<id>/', methods=['PUT', 'DELETE'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def personsEdit(id):
    data = request.json
    if request.method == 'PUT':
        myquery = {"_id": ObjectId(id)}
        dataSet = {}

        if data['category'] and categorias.find({'_id': data['category']}):
            dataSet['category'] = ObjectId(data['category'])
        if data['status'] and estatus.find({'_id': data['status']}):
            dataSet['status'] = ObjectId(data['status'])

        newvalues = {"$set": dataSet}
        personas.update_one(myquery, newvalues)
        return jsonify(data)

    elif request.method == 'DELETE':
        personas.delete_one({"_id": ObjectId(id)})
        return jsonify(data)

@app.route('/reporte/asistencia-ibarti/<fechadesde>/<fechahasta>/', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def asistenciaIbarti(fechadesde, fechahasta):
    results = consultarasistencia(fechadesde, fechahasta)
    return jsonify(results)


if __name__ == "__main__":
    print("Scaneando Puerto....")
    print("******Cargando Datos de Servidor y Puerto***************")

    configuracion=[]
    with open("api.txt") as f:
        for linea in f:
            configuracion.append(linea)
    f.close()

    mi_puerto = int(configuracion[0])
    mi_server = configuracion[1]
    app.run(host='192.168.33.76', port=mi_puerto, debug=False)
    app.run(host='192.168.33.14', port=mi_puerto, debug=False)
