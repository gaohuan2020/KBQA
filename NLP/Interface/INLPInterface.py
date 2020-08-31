from abc import ABCMeta, abstractmethod


class INLPInterface():
    @abstractmethod
    def KeyWordsExtraction(self, sentence):
        pass