from datetime import datetime
from util import carpeta_fotos, getEncode, train, ObjectId
import pymongo
import os

# Conexion Database MONGODB
cliente = pymongo.MongoClient("mongodb://localhost:27017/")
database = cliente["ibartiface"]
person_recognition = database["person_recognition"]
personas = database["persons"]

personas.delete_many({})
if os.path.exists('modeloknn.clf'):
    os.remove('./modeloknn.clf')

modelos = os.listdir(carpeta_fotos)
for modelo in modelos:
    print(modelo)
    fotos = os.listdir(carpeta_fotos+modelo+'/')
    inicial = True
    _persona = None
    for foto in fotos:
        encoding = getEncode(carpeta_fotos+modelo+'/'+foto)
        if inicial:
            inicial = False
            person = {
                "cod_person": "",
                "doc_id": str(modelo),
                "category": ObjectId("5df95dac5870edbd55618de8"),
                "status": ObjectId("5df95dd05870edbd55618de9"),
                "client": '001',
                "template_recognition": encoding,
                "created_date": datetime.now()
            }
            personas.insert_one(person)
            ruta_foto = carpeta_fotos+modelo+'/'+foto
            new_nombre = ruta_foto.replace('.jpg', '-0.jpg')
            os.rename(ruta_foto, new_nombre)
            print(new_nombre)
        else:
            if not _persona:
                _persona = list(personas.find({'doc_id': str(modelo)}))
            for encode in encoding:
                _persona[0]['template_recognition'].append(encode)
            myquery = {"_id":  _persona[0]['_id']}
            newvalues = {"$set": {'template_recognition': _persona[0]['template_recognition']}}
            personas.update_one(myquery, newvalues)
            ruta_foto = carpeta_fotos+modelo+'/'+foto
            new_nombre = ruta_foto.replace('.jpg', '-'+str(len(_persona[0]['template_recognition'])-1)+'.jpg')
            os.rename(ruta_foto, new_nombre)
            print(new_nombre)

train(carpeta_fotos, model_save_path="modeloknn.clf", n_neighbors=1)



