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