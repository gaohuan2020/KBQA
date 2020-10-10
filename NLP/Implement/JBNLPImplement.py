import os
import jieba.analyse as analyse
import NLP.Config.Setting as config
from NLP.Interface.INLPInterface import INLPInterface

class JBNLPImplement(INLPInterface):
    def __init__(self):
        self.topk = 10
        self.withWeight = False
    def KeyWordsExtraction(self, sentence):
        if sentence == '' or sentence == None:
            return []
        if config.KeyWordsExtractionConfig != None:
            if "stopWordsPath" in config.KeyWordsExtractionConfig:
                if config.KeyWordsExtractionConfig["stopWordsPath"] != "":
                    stopWordsPath = config.KeyWordsExtractionConfig["stopWordsPath"]
                    if os.path.exists(stopWordsPath):
                        analyse.set_stop_words(stopWordsPath)
            if "topK" in config.KeyWordsExtractionConfig:        
                if config.KeyWordsExtractionConfig["topK"] != "" and config.KeyWordsExtractionConfig["topK"] > 0:
                    self.topk = int(config.KeyWordsExtractionConfig["topK"])
            if "withWeight" in config.KeyWordsExtractionConfig:  
                if config.KeyWordsExtractionConfig["withWeight"] != "":
                    self.withWeight = int(config.KeyWordsExtractionConfig["withWeight"])
        return analyse.extract_tags(sentence, topK=self.topk, withWeight=self.withWeight)
#TODO Add word segment with dict