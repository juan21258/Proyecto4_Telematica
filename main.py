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
import numpy as np
from sklearn.cluster import SpectralClustering

conf = SparkConf()
conf.setAppName('appSpark')
conf.setMaster("local[32]")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

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

#rescaledData.printSchema()

normalizer = Normalizer(inputCol="features", outputCol="norm")
data = normalizer.transform(rescaledData)

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

sizeval = len(tempcosine.select("dot").collect())
valuesi = []
valuesj = []
valuepair = []
print sizeval
run = 0

tempval = tempcosine.select("dot")
valueobtained = (tempval.groupBy().mean()).collect()[5]
print valueobtained
'''
while run < sizeval:
	valuesi.append(tempcosine.select("i").collect()[run])
	valuesj.append(tempcosine.select("j").collect()[run])
	valuepair.append(tempcosine.select("dot").collect()[run])
	run = run + 1
'''
'''
similmatrix = np.zeros((filecontent,filecontent))
print trysome
pos = 0

while pos < sizeval:
	x = valuesi[pos]
	y = valuesj[pos]
	value = valuepair[pos]
	similmatrix[x][y] = value
	pos = pos + 1

print similmatrix

mat = np.matrix([[1.0,0.0,0.18662266787146495,0.0],[0.0,1,0.0,0.1810105743631082],
	[0.18662266787146495,0.0,1.0,0.0],[0.0,0.1810105743631082,0.0,1.0]])
SpectralClustering(2).fit_predict(mat)

eigen_values, eigen_vectors = np.linalg.eigh(mat)
s = KMeans(n_clusters=2, init='k-means++').fit_predict(eigen_vectors[:, 2:4])

print s
'''
sc.stop()