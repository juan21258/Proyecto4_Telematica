Integrantes:

Juan David Perez Perez

Andres Alejandro Atehortua Parra

Este proyecto es una aplicación con tecnología y modelo de programación
distribuida en Big Data, específicamente el cual permite agrupar
(clustering) un conjunto de documentos utilizando el algoritmo de k–means y una métrica de
similaridad entre documentos.

La logica de este proyecto fue: recolectar los arhcivos y almacenarlos en arreglos, los
cuales contienen 3 cosas:
-Identificador de documento, Nombre del documento y Contenido del documento

Luego se convierte en un RDD (Resilient Distributed Dataset), luego se procede a usar
tecnicas de hashing

HashingTF se usa para generar el vector de frecuencias, el cual sera utilizada para
la metrica de similaridad.

Tambien se utilizan transformaciones para obtener el tfidf, luego de esto se obtiene la
matriz de similaridad mediante la similaridad de coseno, luego se pasa a kmeans; el cual
determinara los grupos o clusters de documentos en base a la matriz de similaridad que se le
pasa y al numero k

Para el presente proyecto se hizo uso de las siguientes tecnologias:

Python
http://docs.python-guide.org/en/latest/starting/install/linux/

Pyspark
https://blog.sicara.com/get-started-pyspark-jupyter-guide-tutorial-ae2fe84f594f

Numpy
https://docs.scipy.org/doc/numpy-1.13.0/user/install.html

Scipy
https://www.scipy.org/install.html

Sklearn kit
http://scikit-learn.org/stable/install.html

La ejecucion en una maquina local es:
$ python main.py

En un cluster:
$ spark-submit  \
  --master yarn \
  --deploy-mode cluster \ 
  --executor-memory 2G \
  --num-executors 4 \
  main.py
