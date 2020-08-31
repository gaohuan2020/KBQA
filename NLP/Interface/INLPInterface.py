from abc import ABCMeta, abstractmethod


class INLPInterface(metaclass=ABCMeta):
    @abstractmethod
    def KeyWordsExtraction(self, sentence):
        pass