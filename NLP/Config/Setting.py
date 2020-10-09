from typing import Sequence


NLPToolName = "jieba"
KeyWordsExtractionConfig = dict(
                                topK = 10,
                                withWeight = False,
                                stopWordsPath = "",
)
LSTMCRFModelConfig = dict(
                                batchSize = 16,
                                embedDim = 100,
                                hiddenDim = 100,
                                useGPU = False,
)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"
Sequence_TAG = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4, PAD_TAG: 5}