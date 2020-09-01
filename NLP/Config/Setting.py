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