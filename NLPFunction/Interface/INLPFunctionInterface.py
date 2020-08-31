from abc import ABCMeta, abstractmethod


class INLPFunctionInterfaces(metaclass=ABCMeta):
    @abstractmethod
    def KeyWordsExtraction(self, sentence):
        pass