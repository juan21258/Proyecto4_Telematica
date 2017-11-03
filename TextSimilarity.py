from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import TextValueProtocol
import re
import os
import time
import unicodedata
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

start_time = time.time()
#Parte para ver todos los archivos tipo txt y guardarlos en una lista
dataset =[]
filecontent = []
path ='./txt'
files = [f for f in os.listdir(path) if os.path.split(f)]
for f in files:
	if f.endswith(".txt"):
		filecontent.append(f)
#Vamos a ver como pasar el contenido del txt para poder analizarlo

for f in filecontent:
	j = os.path.join(path, f)
	with open(j, 'r') as myfile:
		data=myfile.read().replace('\n', '')
		dataset.append(data)