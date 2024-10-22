import random
import numpy as np
from datasets import Dataset
from database.sparse_retrieval import SparseRetrieval
from datasets import DatasetDict, Features, Sequence, Value
from utils.arguments_inference import ModelArguments, DataTrainingArguments, OurTrainingArguments


seed = 104
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정


def run_retrieval() -> DatasetDict:
    # Load model & tokenizer
    q_encoder = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pooler = Pooler(args.pooler)
    
    # Load valid dataset.
    test_dataset = BiEncoderDataset.load_valid_dataset(args.valid_data)

    # Load faiss index & context
    faiss_vector = VectorDatabase(args.faiss_path)
    
    faiss_index = faiss_vector.faiss_index
    text_index = faiss_vector.text_index
    text = faiss_vector.text

    # Load bm25 model.
    if args.bm25_path:
        bm25_model = BM25Reranker(tokenizer=tokenizer, bm25_pickle=args.bm25_path)
    else:
        bm25_model = None
    
    # Get top-k accuracy
    search_evaluation(q_encoder, tokenizer, test_dataset, faiss_index, text_index, text, search_k=args.search_k,
                               bm25_model=bm25_model, faiss_weight=args.faiss_weight, bm25_weight=args.bm25_weight,
                               max_length=args.max_length, pooler=pooler, padding=args.padding, truncation=args.truncation,
                               batch_size=args.batch_size, device=args.device)


def run_sparse_retrieval(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: OurTrainingArguments,
    datasets: DatasetDict,
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(model_args)
    retriever.get_sparse_embedding()
    
    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets