from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import os

conf = SparkConf()
conf.setAppName('appSpark')
conf.setMaster("local[32]")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


path ='./txt_p'
files = [f for f in os.listdir(path) if os.path.split(f)]
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


sc.stop()
