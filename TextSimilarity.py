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