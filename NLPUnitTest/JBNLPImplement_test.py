import unittest
import NLP.Config.Setting as config
from NLP.Implement.JBNLPImplement import JBNLPImplement

class TestJBNLPImplement(unittest.TestCase):
    def test_keywordExtractionSuccess(self):
        jbNLP = JBNLPImplement()
        sentence = "北京天安门"
        keywords = jbNLP.KeyWordsExtraction(sentence=sentence)
        self.assertEqual(len(keywords), 2)
        sentence = ""
        keywords = jbNLP.KeyWordsExtraction(sentence=sentence)
        self.assertEqual(len(keywords), 0)

    def test_keywordExtractionWithSetting(self):
        config.KeyWordsExtractionConfig = None
        jbNLP = JBNLPImplement()
        sentence = "北京天安门"
        keywords = jbNLP.KeyWordsExtraction(sentence=sentence)
        self.assertEqual(len(keywords), 2)
        config.KeyWordsExtractionConfig = dict()
        keywords = jbNLP.KeyWordsExtraction(sentence=sentence)
        self.assertEqual(len(keywords), 2)
        config.KeyWordsExtractionConfig = dict( topK = 1)
        keywords = jbNLP.KeyWordsExtraction(sentence=sentence)
        self.assertEqual(len(keywords), 1)
        config.KeyWordsExtractionConfig = dict(topK = 1, withWeight = True)
        keywords = jbNLP.KeyWordsExtraction(sentence=sentence)
        self.assertEqual(len(keywords), 1)
        self.assertTrue(isinstance(keywords[0][1], float))
        sentence = "北京不是中国的首都"
        config.KeyWordsExtractionConfig = None
        jbNLP = JBNLPImplement()
        keywords = jbNLP.KeyWordsExtraction(sentence=sentence)
        self.assertEqual(len(keywords), 4)
        config.KeyWordsExtractionConfig = dict(stopWordsPath = "NLP\Data\StopWords.txt")
        keywords = jbNLP.KeyWordsExtraction(sentence=sentence)
        self.assertEqual(len(keywords), 3)