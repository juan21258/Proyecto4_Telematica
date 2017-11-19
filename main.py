from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from numpy import array
from math import sqrt
#from pyspark.mllib.clustering import KMeans, KMeansModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.ml.feature import Normalizer
import pyspark.sql.functions as psf
import os
import time
import numpy as np
from sklearn.cluster import SpectralClustering

conf = SparkConf()
conf.setAppName('appSpark')
conf.setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

start_time = time.time()

path ='./txt_p'
files = [f for f in os.listdir(path) if os.path.split(f)]
filecontent = len(files)
dataset = []
cont = 0
for f in files:
	j = os.path.join(path, f)
	with open(j, 'r') as myfile:
		data=myfile.read().replace('\n', '')
		cont = cont + 1
		dataset.append((cont,f,data))		

rdd = sc.parallelize(dataset)
shemaData = rdd.map(lambda x: Row(num=x[0], title=x[1], text=x[2]))
dataFrame = sqlContext.createDataFrame(shemaData)
#dataFrame.select("num", "title").show()
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(dataFrame)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
rescaledData.select("title", "features").show()
#Normalizacion y transformada de la matriz
normalizer = Normalizer(inputCol="features", outputCol="norm")
data = normalizer.transform(rescaledData)

#Proceso de similaridad hallando la norma y el producto punto
mat = IndexedRowMatrix(
    data.select("num", "norm")\
        .rdd.map(lambda row: IndexedRow(row.num, row.norm.toArray()))).toBlockMatrix()
dot = mat.multiply(mat.transpose())
dot.toLocalMatrix().toArray()

dot_udf = psf.udf(lambda x,y: float(x.dot(y)), DoubleType())
data.alias("i").join(data.alias("j"), psf.col("i.num") < psf.col("j.num"))\
    .select(
        psf.col("i.num").alias("i"), 
        psf.col("j.num").alias("j"), 
        dot_udf("i.norm", "j.norm").alias("dot"))\
    .sort("i", "j")\
    .show()

tempcosine = data.alias("i").join(data.alias("j"), psf.col("i.num") < psf.col("j.num"))\
    			.select(
        			psf.col("i.num").alias("i"), 
        			psf.col("j.num").alias("j"), 
        			dot_udf("i.norm", "j.norm").alias("dot"))\
    			.sort("i", "j")

#Cantidad de filas en las columnas del dataframe
sizeval = len(tempcosine.select("dot").collect())
valuesi = []
valuesj = []
valuepair = []

run = 0
'''Se obtienen los valores de la columna i e j y su respectiva similaridad
por ejemplo i = 1 j = 2 dot = 0, significa que la similaridad entre
los documentos 1 y 2 es nula'''
while run < sizeval:
	valuesi.append(tempcosine.select("i").collect()[run][0])
	valuesj.append(tempcosine.select("j").collect()[run][0])
	valuepair.append(tempcosine.select("dot").collect()[run][0])
	run = run + 1

#Se crea la matrix nxn, segun la cantidad de archivos
similmatrix = np.zeros((filecontent,filecontent))

pos = 0
#Se llena la triangular superior
while pos < sizeval:
	x = valuesi[pos]
	y = valuesj[pos]
	value = valuepair[pos]
	similmatrix[x-1][y-1] = value
	pos = pos + 1

i = 0
#se llena la diagonal de 1's debido aque la similaridad de un documento consigo mismo es 1
while i < filecontent:
	similmatrix[i][i] = 1
	i = i + 1

reverse = filecontent - 1
'''Se llena la triangular inferior con los mismos valores de la superior, ya
que la similaridad es igual'''
while  reverse >= 0:
	j = 0
	while j < reverse:
		similmatrix[reverse][j] =similmatrix[j][reverse]
		j = j + 1
	reverse = reverse - 1

print similmatrix

#numero de clusters y matriz de similaridad
SpectralClustering(2).fit_predict(similmatrix)

eigen_values, eigen_vectors = np.linalg.eigh(similmatrix)
#valor de k y cantidad de archivos
orderedclusters = KMeans(n_clusters=2, init='k-means++').fit_predict(eigen_vectors[:, 2:filecontent])

print orderedclusters

kvalue = 2 #numero k

index = 0 #posicion que se esta analizando, significa que es el documento
group = 0 #grupos definidos segun el valor de k, comenzando en cero

recorrido = 0 #variable para iniciar el while
temp = "Cluster "
while recorrido < kvalue:
    temp += str(group) + ":"
    for cluster in orderedclusters:
        ''' Se mira la primera posicion del arreglo, quje equivale al documento 1,
        si en esa posicion hay un cero significa que el k means asigno el documento 1 al
        cluster 0; y asi sucesivamente'''
        if cluster == group:
            temp += files[index]
            temp += ", "
        index += 1
    index = 0
    print temp
    temp = "Cluster "
    group += 1
    recorrido += 1

sc.stop()

print("The execution time was %s seconds" % (time.time() - start_time))