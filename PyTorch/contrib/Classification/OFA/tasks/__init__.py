import nltk
import os

# 设置 NLTK 数据路径
nltk_data_path = '/data/teco-data/ofa/dataset/nltk_data'
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

from .cv_tasks import *
from .mm_tasks import *
from .nlg_tasks import *
from .nlu_tasks import *
from .pretrain_tasks import *
from .speech_tasks import *
from .ofa_task import OFATask
