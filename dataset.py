class BERTEmbedder():
    def __init__(self, max_vocab_size, embedding_dim):
        self.max_vocab_size = max_vocab_size
        self.embedding_dim = embedding_dim

    def get_embedding(self, sentence):
        raise NotImplementedError

class DistillBert(BERTEmbedder):

    def __init__(self, max_vocab_size = 30522, embedding_dim = 768, from_pt=False):
        super().__init__(max_vocab_size, embedding_dim)
        config = transformers.DistilBertConfig()
        config.vocab_size = max_vocab_size
        config.dim = embedding_dim
        self.model = transformers.modeling_tf_distilbert.TFDistilBertModel.from_pretrained(self.model_path, config=config, from_pt=from_pt)
        from transformers import DistilBertTokenizerFast
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.tokenizer_path)

        self.pipeline = transformers.pipeline(task='feature-extraction', model=self.model, tokenizer=self.tokenizer)
    def get_embedding(self, sentence):
        features = self.pipeline(sentence)
        # features = np.array(features)[:, 0, :] [CLS] mapping
        features = np.squeeze(features) # [1:-1] -> no tokens for CLS and SEP
        return features

class Distillbert_base_uncased(DistillBert):
    def __init__(self, max_vocab_size=30522, embedding_dim=768):
        self.model_path = 'distilbert-base-uncased'
        self.tokenizer_path = 'distilbert-base-uncased'
        super().__init__(max_vocab_size, embedding_dim)

class Veterans(Dataset):
    def load_jsons(self, prefix = "."):
        import json, glob, os
        jsons = glob.glob(os.path.join(prefix, "*.json"))
        data = []
        classes = dict()
        for json_file_name in jsons:
            with open(json_file_name) as input:
                loaded = json.load(input)
                sentences = loaded["sentences"]
                for sentence in sentences:
                    text = sentence["text"]
                    roles = sentence["rhetRole"]
                    if len(roles) != 1:
                        raise Exception("More then one rhetorical role per sentence")
                    role = roles[0]
                    if role not in classes:
                        classes[role] = len(classes)
                        print (f"role = {role}, id = {classes[role]}")
                    data += [(classes[role], text)]
        return data, classes

    def __init__(self, embed_cls):
        super().__init__()
        self.embedder = embed_cls()
        path = pathlib.Path(__file__).parent / 'resources/VetClaims-JSON/BVA Decisions JSON Format'
        data, self.classes = self.load_jsons(path)
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        self.train_x = [ pair[1] for pair in train]
        self.train_y = [ pair[0] for pair in train]
        self.test_x = [ pair[1] for pair in test]
        self.test_y = [ pair[0] for pair in test]