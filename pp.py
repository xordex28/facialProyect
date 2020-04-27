import face_recognition
from multiprocessing import Pool,cpu_count
import time
f1 = '/home/luis/Documentos/proyectos/ibartiface/img/desconocidos/0001+0001+1+2019-11-08+15&14&59.jpg'
f2 = '/home/luis/Documentos/proyectos/ibartiface/img/desconocidos/0001+0001+3+2019-11-08+15&14&59.jpg'

def f(x,y,z):
    print(x,y,z)
    times = time.time()
    known_image = face_recognition.load_image_file(f1)
    unknown_image = face_recognition.load_image_file(f2)
    biden_encoding = face_recognition.face_encodings(known_image)[0]
    print("hola")
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    timeS = time.time()
    print(timeS-times)
    return face_recognition.compare_faces([biden_encoding], unknown_encoding)
# f()
# print ('cpu_count() = ' + str(cpu_count()))
# PROCESSES = 1
# print ('Creating pool with %d processes\n' % PROCESSES)
pool = Pool(1)
fs = pool.apply(f,(5,6,7))
print ('pool = %s' % pool)

# f(1)
# if __name__ == "__main__":
#     p = Pool(processes=4)
#     print(p.map(f,[]))