import argparse
import numpy as np
import os
import pandas as pd
import torch
import datasets
import transformers
from tqdm import tqdm
from accelerate import Accelerator
from datasets import Dataset, load_metric, Features, Sequence, Value, load_dataset
from transformers import RagTokenizer, AutoTokenizer, RagSequenceForGeneration, RagModel, DPRContextEncoder, DPRContextEncoderTokenizerFast, set_seed, AutoModel, get_linear_schedule_with_warmup, BartForConditionalGeneration
from transformers.models.rag.retrieval_rag import *
from typing import List, Optional
from functools import partial
import faiss
import pysbd
import random


class RagRetriever:
    """
    Retriever used to get documents from vector queries. It retrieves the documents embeddings as well as the documents
    contents, and it formats them to be used with a RagModel.

    Args:
        config (:class:`~transformers.RagConfig`):
            The configuration of the RAG model this Retriever is used with. Contains parameters indicating which
            ``Index`` to build. You can load your own custom dataset with ``config.index_name="custom"`` or use a
            canonical one (default) from the datasets library with ``config.index_name="wiki_dpr"`` for example.
        question_encoder_tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer that was used to tokenize the question. It is used to decode the question and then use the
            generator_tokenizer.
        generator_tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer used for the generator part of the RagModel.
        index (:class:`~transformers.models.rag.retrieval_rag.Index`, optional, defaults to the one defined by the configuration):
            If specified, use this index instead of the one built using the configuration
    """


    def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, index=None, init_retrieval=True):
        self._init_retrieval = init_retrieval
        # requires_backends(self, ["datasets", "faiss"])
        super().__init__()
        self.index = index  # or self._build_index(config)
        self.generator_tokenizer = generator_tokenizer
        self.question_encoder_tokenizer = question_encoder_tokenizer

        self.n_docs = config.n_docs
        self.batch_size = config.retrieval_batch_size

        self.config = config

        self.ctx_encoder_tokenizer = None
        self.return_tokenized_docs = False

    def postprocess_docs(self, docs, input_strings, prefix, n_docs, return_tensors=None):
        r"""
        Postprocessing retrieved ``docs`` and combining them with ``input_strings``.

        Args:
            docs  (:obj:`dict`):
                Retrieved documents.
            input_strings (:obj:`str`):
                Input strings decoded by ``preprocess_query``.
            prefix (:obj:`str`):
                Prefix added at the beginning of each input, typically used with T5-based models.

        Return:
            :obj:`tuple(tensors)`: a tuple consisting of two elements: contextualized ``input_ids`` and a compatible
            ``attention_mask``.
        """

        def cat_input_and_doc(doc_text, input_string, prefix):
            if prefix is None:
                prefix = ""
            out = (prefix + doc_text + self.config.doc_sep + input_string).replace(
                "  ", " "
            )
            return out

        rag_input_strings = [
            cat_input_and_doc(
                docs[i]["text"][j],
                input_strings[i],
                prefix
            )
            for i in range(len(docs))
            for j in range(n_docs)
        ]

        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            rag_input_strings,
            max_length=self.config.max_combined_length,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
        )

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]

    def _chunk_tensor(self, t, chunk_size):
        return [t[i: i + chunk_size] for i in range(0, len(t), chunk_size)]

    def _main_retrieve(self, question_hidden_states, n_docs):
        question_hidden_states_batched = self._chunk_tensor(question_hidden_states, self.batch_size)
        ids_batched = []
        vectors_batched = []
        for question_hidden_states in question_hidden_states_batched:
            ids, vectors = self.index.get_top_docs(question_hidden_states, n_docs)
            ids_batched.extend(ids)
            vectors_batched.extend(vectors)
        return (
            np.array(ids_batched),
            np.array(vectors_batched),
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, List[dict]]:
        """
        Retrieves documents for specified ``question_hidden_states``.

        Args:
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`):
                A batch of query vectors to retrieve with.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.

        Return:
            :obj:`Tuple[np.ndarray, np.ndarray, List[dict]]`: A tuple with the following objects:

            - **retrieved_doc_embeds** (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs, dim)`) -- The retrieval
              embeddings of the retrieved docs per query.
            - **doc_ids** (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs)`) -- The ids of the documents in the
              index
            - **doc_dicts** (:obj:`List[dict]`): The :obj:`retrieved_doc_embeds` examples per query.
        """

        self.index = CustomHFIndexSum.load_from_disk(
            vector_size=self.config.retrieval_vector_size,
            dataset_path=self.config.passages_path,
            index_path=self.config.index_path)

        doc_ids, retrieved_doc_embeds = self._main_retrieve(question_hidden_states, n_docs)
        return retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids)

    def set_ctx_encoder_tokenizer(self, ctx_encoder_tokenizer):
        # used in end2end retriever training
        self.ctx_encoder_tokenizer = ctx_encoder_tokenizer
        self.return_tokenized_docs = True

    def __call__(self, question_input_ids, question_hidden_states, prefix=None, n_docs=None, return_tensors=None):
        """
        Retrieves documents for specified :obj:`question_hidden_states`.

        Args:
            question_input_ids: (:obj:`List[List[int]]`) batch of input ids
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            prefix: (:obj:`str`, `optional`):
                The prefix used by the generator's tokenizer.
            n_docs (:obj:`int`, `optional`):
                The number of docs retrieved per query.
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.

        Returns: :class:`~transformers.BatchEncoding`: A :class:`~transformers.BatchEncoding` with the following
        fields:

            - **context_input_ids** -- List of token ids to be fed to a model.

              `What are input IDs? <../glossary.html#input-ids>`__

            - **context_attention_mask** -- List of indices specifying which tokens should be attended to by the model
            (when :obj:`return_attention_mask=True` or if `"attention_mask"` is in :obj:`self.model_input_names`).

              `What are attention masks? <../glossary.html#attention-mask>`__

            - **retrieved_doc_embeds** -- List of embeddings of the retrieved documents
            - **doc_ids** -- List of ids of the retrieved documents
        """

        n_docs = n_docs if n_docs is not None else self.n_docs
        prefix = prefix if prefix is not None else self.config.generator.prefix
        retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)

        input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
        context_input_ids, context_attention_mask = self.postprocess_docs(
            docs, input_strings, prefix, n_docs, return_tensors=return_tensors
        )

        if self.return_tokenized_docs:
            retrived_doc_text = []
            retrived_doc_title = []

            for b_idx in range(len(docs)):
                for doc_idx in range(n_docs):
                    retrived_doc_text.append(docs[b_idx]["text"][doc_idx])
                    retrived_doc_title.append(docs[b_idx]["title"][doc_idx])

            tokenized_docs = self.ctx_encoder_tokenizer(
                retrived_doc_title,
                retrived_doc_text,
                truncation=True,
                padding="longest",
                return_tensors=return_tensors,
            )

            return BatchEncoding(
                {
                    "context_input_ids": context_input_ids,
                    "context_attention_mask": context_attention_mask,
                    "retrieved_doc_embeds": retrieved_doc_embeds,
                    "doc_ids": doc_ids,
                    "tokenized_doc_ids": tokenized_docs["input_ids"],
                    "tokenized_doc_attention_mask": tokenized_docs["attention_mask"],
                },
                tensor_type=return_tensors,
            )

        else:
            return BatchEncoding(
                {
                    "context_input_ids": context_input_ids,
                    "context_attention_mask": context_attention_mask,
                    "retrieved_doc_embeds": retrieved_doc_embeds,
                    "doc_ids": doc_ids,
                },
                tensor_type=return_tensors,
            )


class CustomHFIndexSum(Index):
    def __init__(self, vector_size, dataset, index_path, index_initialized=False):
        self.vector_size = vector_size
        self.dataset = dataset
        self.index_path = index_path
        self.index_initialized = index_initialized
        self.init_index()
        self._check_dataset_format(with_index=self.index_initialized)
        dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True, dtype="float32")


    def _check_dataset_format(self, with_index: bool):
        if not isinstance(self.dataset, Dataset):
            raise ValueError(f"Dataset should be a datasets.Dataset object, but got {type(self.dataset)}")
        if len({"title", "text", "embeddings"} - set(self.dataset.column_names)) > 0:
            raise ValueError(
                "Dataset should be a dataset with the following columns: "
                "title (str), text (str) and embeddings (arrays of dimension vector_size), "
                f"but got columns {self.dataset.column_names}"
            )
        if with_index and "embeddings" not in self.dataset.list_indexes():
            raise ValueError(
                "Missing faiss index in the dataset. Make sure you called `dataset.add_faiss_index` to compute it "
                "or `dataset.load_faiss_index` to load one from the disk."
            )

    @classmethod
    def load_from_disk(cls, vector_size, dataset_path, index_path):
        if dataset_path is None or index_path is None:
            raise ValueError(
                "Please provide ``dataset_path`` and ``index_path`` after calling ``dataset.save_to_disk(dataset_path)`` "
                "and ``dataset.get_index('embeddings').save(index_path)``."
            )
        dataset = load_from_disk(dataset_path)
        return cls(vector_size=vector_size, dataset=dataset, index_path=index_path)

    def init_index(self):
        if not self.is_initialized():
            self.dataset.load_faiss_index("embeddings", file=self.index_path)
            self.index_initialized = True

    def is_initialized(self):
        return self.index_initialized

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        _, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        return np.array(ids), np.array(vectors)  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)


def create_and_set_indexed_dataset(chunks):
    # create a dataframe for the document, containing the text and the title for each chunk
    chunks_with_titles = []
    for i,chunk in enumerate(chunks):
        title = "title" + str(i)
        row = {"title":title, "text":chunk}
        chunks_with_titles.append(row)

    chunks_df = pd.DataFrame(chunks_with_titles)
    # create the dataset
    dataset = Dataset.from_pandas(chunks_df)
    # map dataset to new features
    dataset = dataset.map(
        partial(embed),
        batched=True,
        batch_size=16,
        features=new_features
    )

    dataset_path = os.path.join("my_knowledge_dataset")
    dataset.save_to_disk(dataset_path)
    # Let's use the Faiss implementation of HNSW for fast approximate nearest neighbor search
    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)

    try:
        # add faiss index to dataset
        dataset = dataset.add_faiss_index("embeddings", custom_index=index)
        index_path = os.path.join("my_knowledge_dataset_hnsw_index.faiss")
        dataset.get_index("embeddings").save(index_path)
        # set dataset and index into model
        model.config.passages_path = dataset_path
        model.config.index_path = index_path
    except ValueError:
        empty_chunks = []
        for i in range(len(chunks)):
            empty_chunks.extend(" ")
        create_and_set_indexed_dataset(empty_chunks)


def add_doc_index(dataframe_chunked, dataframe_chunked_idx):
    dataframe_chunked["doc_index"] = 0
    index = 0
    for i in range(len(dataframe_chunked_idx)):
        num_chunks = dataframe_chunked_idx.iloc[i]["idx"]
        dataframe_chunked.loc[index:index + num_chunks, "doc_index"] = i
        index += num_chunks
    return dataframe_chunked


def remove_chunks_without_target(dataframe_chunked, dataframe_chunked_no_nan, dataframe_chunked_idx, dataframe_chunked_idx_no_nan):
    index = 0
    for i in range(len(dataframe_chunked_idx)):
        num_chunks = dataframe_chunked_idx.iloc[i]["idx"]
        count_not_nan = dataframe_chunked[index:index + num_chunks]["summary"].count()
        dataframe_chunked_idx_no_nan.iloc[i]["idx"] = count_not_nan
        index += num_chunks
    dataframe_chunked_no_nan = dataframe_chunked_no_nan.dropna()
    return dataframe_chunked_no_nan, dataframe_chunked_idx_no_nan


def preprocess_function(batch):
    """Prepares the dataset to be process by the model.

    Args:
        batch: The batch to process.

    Returns:
        The batch processed.
    """
    inputs = tokenizer(batch["text"], padding="max_length", max_length=max_len_source, truncation=True)
    outputs = tokenizer(batch["summary"], padding="max_length", max_length=max_len_summary, truncation=True)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids
    batch["labels"] = [[-100 if token == tokenizer.generator.pad_token_id else token for token in labels]
                       for labels in batch["labels"]]
    return batch


def embed(documents):
    """Compute the DPR embeddings of document"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


def create_random_chunks(chunks, sentences_tokenizer, f=5):
    all_sentences = []
    for chunk in chunks:
        sentences = sentences_tokenizer.segment(chunk)
        all_sentences.extend(sentences)
    f = len(all_sentences) if len(all_sentences) < f else f
    random_chunks = [" ".join(random.sample(all_sentences, f)) for _ in range(len(all_sentences)//2)]
    return random_chunks


def create_dataset_retriever_list(dataset, dataframe, sentence_mode, random_chunks):
    if sentence_mode or random_chunks:
        # set up the tokenizer to split the sentences
        sentences_tokenizer = pysbd.Segmenter(language="en", clean=True)
    dataset_retriever_list = []
    # for each chunk to summarize
    for elem in dataset:
        # retrieve the other chunks of the same document
        chunks_filtered = dataframe[dataframe["doc_index"] == elem["doc_index"]]
        # if only 1 chunk is retrieved, we do not filter again
        if len(chunks_filtered) > 1:
            chunks_filtered = chunks_filtered[chunks_filtered["Unnamed: 0"] != elem["chunk_id"]]
        chunks_filtered = chunks_filtered["text"]
        if random_chunks:
            chunks_filtered = pd.Series(create_random_chunks(chunks_filtered, sentences_tokenizer))
        # for each chunk retrieved, we take its sentences
        elif sentence_mode:
            all_sentences = []
            for chunk in chunks_filtered:
                sentences = sentences_tokenizer.segment(chunk)
                all_sentences.extend(sentences)
            chunks_filtered = pd.Series(all_sentences)
        dataset_retriever_list.append(chunks_filtered)
    return dataset_retriever_list


def set_correct_chunks_to_retrieve(dataset_retriever, doc_index, chunk_id):
    chunks_filtered = dataset_retriever.filter(lambda x : x["doc_index"] == doc_index)
    chunks_filtered = chunks_filtered.filter(lambda x : x["chunk_id"] != chunk_id)
    create_and_set_indexed_dataset(chunks_filtered["text"])


def train(model):
    # Define the optimizer that will be used to tune network weights during the training session.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    model, optimizer, train_dataset_loader = accelerator.prepare(model, optimizer, train_dataset_model_input)
    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # may change its length.
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_dataset_loader)//10,
        num_training_steps=len(train_dataset_loader) * args.num_epochs
    )

    # Training loop.
    print("Initiating fine-tuning on our dataset")

    progress_bar = tqdm(range(args.num_epochs * len(train_dataset_loader)), disable=not accelerator.is_main_process)

    # Set the model in the training mode.
    model.train()
    for epoch in range(args.num_epochs):
        # The data loader passes data to the model based on the batch size.
        for i,data in enumerate(train_dataset_loader, 0):
            # Create the indexed dataset.
            create_and_set_indexed_dataset(train_dataset_retriever_list[i])
            # Create tensors to pass to the model
            input_ids = torch.tensor(data["input_ids"], device=device).unsqueeze(0)
            attention_mask = torch.tensor(data["attention_mask"], device=device).unsqueeze(0)
            labels = torch.tensor(data["labels"], device=device).unsqueeze(0)

            # The model outputs the first element that gives the loss for the forward pass.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]

            if i % len(train_dataset_loader) == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

            # Backpropagation.
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()  # the default action is to accumulate the gradients on every loss.backward() call.
            progress_bar.update(1)

        if not args.debug:
            # Save the model at the end of each epoch.
            model.save_pretrained(model_path)
            if args.eval_with_bart:
                model.generator.save_pretrained(model_generator_path)


def get_rouge_metrics(preds, refs):
    """Computes the rouge metrics.

    Args:
        preds: list. The model predictions.
        refs: list. The references.

    Returns:
        The rouge metrics.
    """

    rouge_output = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return {
        "r1": round(rouge_output["rouge1"].mid.fmeasure, 4),
        "r2": round(rouge_output["rouge2"].mid.fmeasure, 4),
        "rL": round(rouge_output["rougeL"].mid.fmeasure, 4)
    }


def generate_predictions(batch):
    """Generates the predictions on the test set with the fine-tuned model.
    Args:
        batch: The batch to process.
    Returns:
        The batch processed to obtain the predictions.
    """
    if not args.eval_with_bart:
        # Create dataset indexed
        create_and_set_indexed_dataset(eval_dataset_retriever_list[batch["chunk_id"]])

    inputs = tokenizer(batch["text"], padding="max_length", max_length=max_len_source, return_tensors="pt",
                      truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    predicted_summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                           max_length=max_len_summary, num_beams=2, repetition_penalty=2.5,
                                           length_penalty=1.0, early_stopping=True)
    batch["predicted_summary"] = tokenizer.batch_decode(predicted_summary_ids, skip_special_tokens=True)
    return batch


def concatenate_summaries(preds, dataset_idx):
    """Concatenates the predicted chunks' summaries to rebuild the final summary of each document.

    Args:
        preds: list. The predicted chunks' summaries.
        dataset_idx: DataFrame. The number of chunks per document.

    Returns:
        The final summaries.
    """
    final_summaries = []
    # For each chunk.
    for i in dataset_idx:
        # Build its final summary by concatenating the summaries of its chunks.
        summary = ""
        for j in range(i):
            summary = summary + preds[j][0] + " "
        final_summaries.append(summary)
        # Delete the first "i" predictions to compute the next document.
        del preds[:i]
    return final_summaries


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="The batch size for training")
    parser.add_argument("--batch_eval", type=int, default=4, help="The batch size for evaluating")
    parser.add_argument("--checkpoint_chunk_encoder", default="facebook/dpr-question_encoder-single-nq-base",
                        help="The model checkpoint to use")
    parser.add_argument("--checkpoint_generator", default="models/checkpoint_bart_se3_1024_256/",
                        help="The model checkpoint to use")
    parser.add_argument("--dataset", default="billsum", help="The dataset to use")
    parser.add_argument("--debug", default=False, action="store_true", help="If in debug mode")
    parser.add_argument("--eval_with_bart", default=False, action="store_true", help="If eval with just bart")
    parser.add_argument("--k", type=int, default=5, help="The k document to retrieve")
    parser.add_argument("--lr", type=int, default=1e-6, help="The learning rate")
    parser.add_argument("--max_len_source", type=int, default=512, help="The input max size")
    parser.add_argument("--max_len_summary", type=int, default=128, help="The output max size")
    parser.add_argument("--n_docs", type=int, default=0, help="The number of training examples")
    parser.add_argument("--n_docs_eval", type=int, default=0, help="The number of examples to evaluate")
    parser.add_argument("--no_train", default=False, action="store_true", help="If perform training")
    parser.add_argument("--num_epochs", type=int, default=1, help="The number of epochs for training")
    parser.add_argument("--random_chunks", default=False, action="store_true", help="If use random chunks in retrieval")
    parser.add_argument("--seed", type=int, default=1234, help="The seed to use")
    parser.add_argument("--sentence_mode", default=False, action="store_true", help="If use sentences in the retrieval")

    args = parser.parse_args()

    set_seed(args.seed)

    accelerator = Accelerator(fp16=True)

    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    device = accelerator.device

    max_len_source = args.max_len_source
    max_len_summary = args.max_len_summary

    # define ctx_encoder, ctx_tokenizer and new_features
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base").to(device)
    ctx_encoder.config.max_length = 512
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    # optional, save as float32 instead of float64 to save space
    new_features = Features(
        {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))}
    )

    question_encoder_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_chunk_encoder)
    generator_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_generator)
    tokenizer = RagTokenizer(question_encoder_tokenizer, generator_tokenizer)

    model = RagSequenceForGeneration.from_pretrained_question_encoder_generator(args.checkpoint_chunk_encoder,
                                                                                args.checkpoint_generator).to(device)
    model.config.n_docs = args.k
    retriever = RagRetriever(model.config, question_encoder_tokenizer, generator_tokenizer)

    #if args.sentence_mode:
        #retriever.config.max_combined_length = 768
    #else:
    retriever.config.max_combined_length = 1024
    model.set_retriever(retriever)

    rouge = load_metric("rouge")

    data_dir = "data/"

    path_experiment = args.dataset + "_" + str(max_len_source) + "_" + str(max_len_summary) + "_" + \
                      str(args.num_epochs) + "_epochs"
    predictions_path = "predictions/" + path_experiment

    model_path = "models/" + path_experiment

    if args.eval_with_bart:
        model_generator_path = "models/" + path_experiment + "_bart"
        if args.n_docs > 0:
            model_generator_path += "_" + str(args.n_docs)
        if args.sentence_mode:
            model_generator_path += "_sent"
        if args.random_chunks:
            model_generator_path += "_random"

    if args.checkpoint_generator == "facebook/bart-base":
        predictions_path += "_bart_base"

    if args.n_docs > 0:
        predictions_path += "_" + str(args.n_docs)
        model_path += "_" + str(args.n_docs)

    if args.sentence_mode:
        predictions_path += "_sent"
        model_path += "_sent"

    if args.random_chunks:
        predictions_path += "_random"
        model_path += "_random"

    if args.eval_with_bart:
        predictions_path += "_bart"

    train_file_path = data_dir + "billsum_training_set"
    train_file_chunked_path = data_dir + "bart_billsum_training_set_chunked_256_512_contrastive"
    train_file_chunked_idx_path = data_dir + "bart_billsum_training_set_chunked_idx_256_512_contrastive"

    test_file_path = data_dir + "billsum_test_set"
    test_file_chunked_path = data_dir + "bart_billsum_test_set_chunked_256_512_contrastive"
    test_file_chunked_idx_path = data_dir + "bart_billsum_test_set_chunked_idx_256_512_contrastive"

    train_dataset = pd.read_csv(train_file_path)
    train_dataset_chunked = pd.read_csv(train_file_chunked_path)
    train_dataset_chunked_idx = pd.read_csv(train_file_chunked_idx_path)

    test_dataset = pd.read_csv(test_file_path)
    test_dataset_chunked = pd.read_csv(test_file_chunked_path)
    test_dataset_chunked_idx = pd.read_csv(test_file_chunked_idx_path)

    train_dataset_chunked = add_doc_index(train_dataset_chunked, train_dataset_chunked_idx)
    test_dataset_chunked = add_doc_index(test_dataset_chunked, test_dataset_chunked_idx)

    train_dataset_chunked_no_nan = train_dataset_chunked.copy(deep=True)
    train_dataset_chunked_idx_no_nan = train_dataset_chunked_idx.copy(deep=True)

    train_dataset_chunked_no_nan, train_dataset_chunked_idx_no_nan = \
        remove_chunks_without_target(train_dataset_chunked, train_dataset_chunked_no_nan, train_dataset_chunked_idx,
                                     train_dataset_chunked_idx_no_nan)

    train_dataframe = train_dataset_chunked_no_nan[:np.sum(train_dataset_chunked_idx_no_nan[:args.n_docs]["idx"])]
    train_dataset = Dataset.from_pandas(train_dataframe)
    train_dataset_model_input = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1,
        remove_columns=["text", "summary", "__index_level_0__"]
    )
    train_dataset_model_input = train_dataset_model_input.rename_column("Unnamed: 0", "chunk_id")

    train_dataset_retriever_list = create_dataset_retriever_list(train_dataset_model_input, train_dataset_chunked,
                                                                 args.sentence_mode, args.random_chunks)

    train_dataset_model_input.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "doc_index", "chunk_id"]
    )
    print(f"len train_dataset_retriever_list -> {len(train_dataset_retriever_list)}")

    if not args.no_train:
        # fine-tuning
        train(model)

    if args.n_docs_eval > 0:
        test_dataset_chunked = test_dataset_chunked[:np.sum(test_dataset_chunked_idx[:args.n_docs_eval]["idx"])]
        eval_dataset_chunked = Dataset.from_pandas(test_dataset_chunked).rename_column("Unnamed: 0", "chunk_id")
    else:
        # evaluation
        eval_dataset_chunked = Dataset.from_pandas(test_dataset_chunked).rename_column("Unnamed: 0", "chunk_id")

    eval_dataset_retriever_list = create_dataset_retriever_list(eval_dataset_chunked, test_dataset_chunked,
                                                                args.sentence_mode, args.random_chunks)
    print(f"len eval_dataset_retriever_list -> {len(eval_dataset_retriever_list)}")

    if args.eval_with_bart:
        tokenizer = generator_tokenizer
        model = BartForConditionalGeneration.from_pretrained(model_generator_path).to(device)

    predictions = eval_dataset_chunked.map(generate_predictions)

    if args.n_docs_eval > 0:
        references = test_dataset[:args.n_docs_eval]["summary"]
        predictions_final = concatenate_summaries(predictions["predicted_summary"],
                                                  test_dataset_chunked_idx[:args.n_docs_eval]["idx"])
    else:
        references = test_dataset["summary"]
        predictions_final = concatenate_summaries(predictions["predicted_summary"], test_dataset_chunked_idx["idx"])

    # compute rouge scores
    rouge_result = get_rouge_metrics(preds=predictions_final, refs=references)
    print(f"\nROUGE-1: {rouge_result['r1']}\nROUGE-2: {rouge_result['r2']}\nROUGE-L: {rouge_result['rL']}")

    print(predictions_final[0])
    print("\n")
    print(references[0])

    if not args.debug:
        # save predictions
        pd.DataFrame(data=predictions_final, columns=["prediction"]).to_csv(predictions_path)