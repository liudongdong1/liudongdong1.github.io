# AllenNLPIntroduce


> you can write your own script to construct the dataset reader and model and run the training loop, or you can write a configuration file and use the `allennlp train` command

## 1. Text Classification

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201022211754511.png)

| Spam filtering     | Detect and filter spam emails | Email                   | Spam / Not spam          |
| ------------------ | ----------------------------- | ----------------------- | ------------------------ |
| Sentiment analysis | Detect the polarity of text   | Tweet, review           | Positive / Negative      |
| Topic detection    | Detect the topic of text      | News article, blog post | Business / Tech / Sports |

- **Reading Data**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201022212005872.png)

- **Model**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201022212412613.png)

## 2. Train with script

```python
from typing import Dict, Iterable, List
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()  ##？？
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}  ##？？
        self.max_tokens = max_tokens

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                tokens = self.tokenizer.tokenize(text)   ##？？得到的是什么
                if self.max_tokens:
                    tokens = tokens[:self.max_tokens]
                text_field = TextField(tokens, self.token_indexers)##？？token_indexers 用来干什么
                label_field = LabelField(sentiment)
                fields = {'text': text_field, 'label': label_field}
                yield Instance(fields)

dataset_reader = ClassificationTsvReader(max_tokens=64)
instances = dataset_reader.read("quick_start/data/movie_review/train.tsv")

for instance in instances[:10]:
    print(instance)

class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)

    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}

def build_dataset_reader() -> DatasetReader:
    return ClassificationTsvReader()

def read_data(
    reader: DatasetReader
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    training_data = reader.read("quick_start/data/movie_review/train.tsv")
    validation_data = reader.read("quick_start/data/movie_review/dev.tsv")
    return training_data, validation_data

def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)  #？？ 

def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(   ##？？
        {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=10)  #？？
    return SimpleClassifier(vocab, embedder, encoder)
def run_training_loop():
    dataset_reader = build_dataset_reader()

    # These are a subclass of pytorch Datasets, with some allennlp-specific
    # functionality added.
    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)

    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.
    train_data.index_with(vocab)   #？？
    dev_data.index_with(vocab)

    # These are again a subclass of pytorch DataLoaders, with an
    # allennlp-specific collate function, that runs our indexing and
    # batching code.
    train_loader, dev_loader = build_data_loaders(train_data, dev_data)

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this guide, we need to do this.
    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(
            model,
            serialization_dir,
            train_loader,
            dev_loader
        )
        print("Starting training")
        trainer.train()
        print("Finished training")

# The other `build_*` methods are things we've seen before, so they are
# in the setup section above.
def build_data_loaders(
    train_data: torch.utils.data.Dataset,
    dev_data: torch.utils.data.Dataset,
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=8, shuffle=False)
    return train_loader, dev_loader

def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader
) -> Trainer:
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = AdamOptimizer(parameters)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
    )
    return trainer


run_training_loop()
```

## 3. Train with allennlp

```python
import tempfile
import json
from typing import Dict, Iterable, List

import torch
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@DatasetReader.register("classification-tsv")
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self,
                         tokens: List[Token],
                         label: str = None) -> Instance:
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                tokens = self.tokenizer.tokenize(text)
                yield self.text_to_instance(tokens, sentiment)


@Model.register("simple_classifier")
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        print("In model.forward(); printing here just because binder is so slow")
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {'probs': probs}
        if label is not None:
            self.accuracy(logits, label)
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

config = {
    "dataset_reader" : {
        "type": "classification-tsv",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "quick_start/data/movie_review/train.tsv",
    "validation_data_path": "quick_start/data/movie_review/dev.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10
                }
            }
        },
        "encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 10
        }
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": True
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5
    }
}


with tempfile.TemporaryDirectory() as serialization_dir:
    config_filename = serialization_dir + "/training_config.json"
    with open(config_filename, 'w') as config_file:
        json.dump(config, config_file)
    from allennlp.commands.train import train_model_from_file
    # Instead of this python code, you would typically just call
    # allennlp train [config_file] -s [serialization_dir]
    train_model_from_file(config_filename,
                          serialization_dir,
                          file_friendly_logging=True,
                          force=True)
#allennlp train my_text_classifier.jsonnet -s model --include-package my_text_classifier
```

## 4. Unlabeled Prediction

```python
import tempfile
from typing import Dict, Iterable, List, Tuple

import torch

import allennlp
from allennlp.common import JsonDict
from allennlp.data import DataLoader, DatasetReader, Instance
from allennlp.data import Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.predictors import Predictor
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer


class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                yield self.text_to_instance(text, sentiment)


class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        output = {'probs': probs}
        if label is not None:
            self.accuracy(logits, label)
            # Shape: (1,)
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


def build_dataset_reader() -> DatasetReader:
    return ClassificationTsvReader()

def read_data(
    reader: DatasetReader
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    training_data = reader.read("quick_start/data/movie_review/train.tsv")
    validation_data = reader.read("quick_start/data/movie_review/dev.tsv")
    return training_data, validation_data

def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)

def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=10)
    return SimpleClassifier(vocab, embedder, encoder)

def build_data_loaders(
    train_data: torch.utils.data.Dataset,
    dev_data: torch.utils.data.Dataset,
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=8, shuffle=False)
    return train_loader, dev_loader

def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader
) -> Trainer:
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = AdamOptimizer(parameters)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
    )
    return trainer

def run_training_loop():
    dataset_reader = build_dataset_reader()

    # These are a subclass of pytorch Datasets, with some allennlp-specific
    # functionality added.
    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)

    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    # These are again a subclass of pytorch DataLoaders, with an
    # allennlp-specific collate function, that runs our indexing and
    # batching code.
    train_loader, dev_loader = build_data_loaders(train_data, dev_data)

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this guide, we need to do this.
    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(
            model,
            serialization_dir,
            train_loader,
            dev_loader
        )
        trainer.train()

    return model, dataset_reader
class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)


# We've copied the training loop from an earlier example, with updated model
# code, above in the Setup section. We run the training loop to get a trained
# model.
model, dataset_reader = run_training_loop()
vocab = model.vocab
predictor = SentenceClassifierPredictor(model, dataset_reader)

output = predictor.predict('A good movie!')
print([(vocab.get_token_from_index(label_id, 'labels'), prob)
       for label_id, prob in enumerate(output['probs'])])
output = predictor.predict('This was a monstrous waste of time.')
print([(vocab.get_token_from_index(label_id, 'labels'), prob)
       for label_id, prob in enumerate(output['probs'])])
```

```python
#部署到web应用
python allennlp-server/server_simple.py \
    --archive-path model/model.tar.gz \
    --predictor sentence_classifier \
    --field-name sentence
    --include-package my_text_classifier
```

## 5. API

### 5.1. DataReading

- **Field&&Instance**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201022222221588.png)

> A `Field` contains one piece of data for one example that is passed through your model. `Fields` get converted to tensors in a model, either as an input or an output, after being converted to IDs and batched & padded.

- **Dataset readers**

> [Text classification](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/text_classification_json.py)
>
> [Sequence labeling](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/sequence_tagging.py)
>
> [Language modeling](https://github.com/allenai/allennlp-models/blob/master/allennlp_models/lm/dataset_readers/simple_language_modeling.py)
>
> [Natural language inference](https://github.com/allenai/allennlp-models/blob/master/allennlp_models/pair_classification/dataset_readers/snli.py)
>
> [Semantic role labeling](https://github.com/allenai/allennlp-models/blob/master/allennlp_models/structured_prediction/dataset_readers/srl.py)
>
> [Seq2Seq tasks](https://github.com/allenai/allennlp-models/blob/master/allennlp_models/generation/dataset_readers/seq2seq.py)
>
> [Constituency parsing](https://github.com/allenai/allennlp-models/blob/master/allennlp_models/structured_prediction/dataset_readers/penn_tree_bank.py) and [dependency parsing](https://github.com/allenai/allennlp-models/blob/master/allennlp_models/structured_prediction/dataset_readers/universal_dependencies.py)
>
> [Reading comprehension](https://github.com/allenai/allennlp-models/tree/master/allennlp_models/rc/dataset_readers)
>
> [Semantic parsing](https://github.com/allenai/allennlp-semparse)

- **Vocabulary**

> `Vocabulary` manages different mappings using a concept called *namespaces*. Each namespace is a distinct mapping from strings to integers, so strings in different namespaces are treated separately. 

>  create a `Vocabulary` object is to pass a collection of `Instances` to the `Vocabulary.from_instances()` method. This will count all strings in the `Instances` that need to be mapped to integers, then use those counts to decide what strings should be in the vocabulary. 

### 5.2. Text Representation

> - GloVe or word2vec embeddings
> - Character CNNs
> - POS tag embeddings
> - Combination of GloVe and character CNNs
> - wordpieces and BERT

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201022230601867.png)

- **GloVe**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201022230658192.png)

> 1. Tokenizer (Text → Tokens)
> 2. TextField, TokenIndexer, and Vocabulary (Tokens → Ids)
> 3. TextFieldEmbedder (Ids → Vectors)

- **Tokenizers**

> - Characters (“AllenNLP is great” → `["A", "l", "l", "e", "n", "N", "L", "P", " ", "i", "s", " ", "g", "r", "e", "a", "t"]`)
> - Wordpieces (“AllenNLP is great” → `["Allen", "##NL", "##P", "is", "great"]`)
> - Words (“AllenNLP is great” → `["AllenNLP", "is", "great"]`)

- **TokenIndexers**

> Each `TokenIndexer` knows how to convert a `Token` into a representation that can be encoded by a corresponding piece of the model. This could be just mapping the token to an index in some vocabulary, or it could be breaking up the token into characters or wordpieces and representing the token by a sequence of indexed characters



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/allennlpintroduce/  

