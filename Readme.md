For record, below document is not implemented in code.

### 1. Scaling
```text
- **Paper 1-1: Training Compute-Optimal Large Language Models**
    - The number of training tokens and model size should be scaled together.
    - Note: The paper used Huber loss for fitting the scaling-law model.
```

### 2. Data
```text
- **Paper 2-1: FineWeb** / (below is from Fineweb post.)
    - **2-1-1. Related resources**
        - [FineWeb blog post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
    - **2-1-2. Large-scale data processing library**
        - [Hugging Face DataTrove](https://github.com/huggingface/datatrove)
    - **2-1-3. Collecting high-quality data**
        - **Perplexity-based evaluation**
            - Low perplexity does not always lead to better downstream performance.
            - Medical, mathematical, and other specialized data may have distributions different from Wikipedia.
            - Even when such data has high perplexity under a Wikipedia-trained model, it may still be useful for domain-specific tasks.
        - **Small-model evaluation**
            - Train small models on representative subsets of candidate datasets.
            - Evaluate the models on multiple downstream benchmarks.
            - Using diverse benchmarks helps prevent overfitting the data-selection process to a single benchmark.
            - lighteval.(https://github.com/huggingface/lighteval/) / (Below is from lighteval repo.)
                  - General knowledge : MMLU , MMMU , BIG-Bench
                  - Question Answering : TriviaQA , Natural Questions, SimpleQA , Humanity's last question
                  - Math and Code Benchmark :
                  - Chat Model Evaluation :
                      - Instruction Following : IFEval , IFEval-fr
                      - Reasoning : MUSR, DROP(discrete reasoning)
        - **Pipeline**
        - commoncrawl     
            - extract text -> language filter -> url, document filter -> deduplication -> quality heuristic -> quality evaluation -> tokenization

        - **How to crwal**
            - preprocess CommonCrawl data(already in data format)
            - commoncrawl has two format : 1) WARC(raw data) 2) WET(text only version)
                  - can extract text data from WARC using **trafilatura library**
                  - WET was poor compared to text extracted from WARC 
                  - for tradeoff(budget and quality), it is good to use WET
                    
        - **Piepline**
            - text extraction
            - base filtering
                - removing lower quality data in pipeline.
                      - filtering basis
                          - blocklist(URL filtering)
                          - fastText language classifier
                          - MassiveText filters(quality and repetition)
                - Deduplicating
                      - speed up method
                          - hashing
                          - efficient data structures in indexing
                      - if searching is only for complete word, it's better to use hash table, inverted index, trie, etc in general rather than suffix array.
                      - RefinedWeb, using MinHash & LSH bounding with (for 8 hash values(MinHash values) in one bounding and 14 bounding. So,8*14=112)
                      - For common crawling, it periodically crawl data.
                          - e.g) 2024-04-10 , 2024-04-11..(snapshot/ each is called dump).
                              - If we deduplicate inside each snapshot, deplication amoong snapshots remain.
                                  - From recent dump, deduplicate 1) inside dump 2) among dumps
                      - deduplication does not show better performance.
                          - duplication does not always mean their quality is bad. It might be essential data
                              - Because it is improtant, it might show a lot.
                              - data in deduplication shows better performance than kept dataset.
                          - deduplication
                              - did MinHash deduplication independently within each web dump rather than deduplicating across all dumps together shows better performance.
                              - Did deduplication independently preserved recurring high-quality data while removing massive duplicate clusters, allowing the resulting dataset to match RefinedWeb’s performance.
                              - If filterting process did well, deduplication with common sense seems to do better.
                                  - filterning vs deduplication
                                      - filtering : eliminating low quality data
                                      - deduplication : eliminating same data
                              
                      
            -  Heuristic filtering
                  - eliminating low quality data in accordance with Human criteria
                        - if sentence length is too short, delete
                        - if special character/numbers are too much, delete
                        - if too much same sentence appear, delete
                        - if . does not show a lot in the end of sentence, delete
                        - if advertisement sentence shows a lot, delete
            - Heuristic evaluation
                  - fast, simple such as if sentence length is too short, bad score. So the problem is that it shows low quality.

            - Experiement
                  - Let global MinHash data is relatively low quality data compared to target data.
                  - If some traits in global MinHash shows a lot but not in target data, delete those datas in target data.
                      - Criteria is one more. do some small ablation test and if deleting these shows better performance, do delete.
```

- **Big Data process piepline -> DataTrove**
```text
    - Big text data process library from huggingface
    - It can do filtering, deduplication, process for big data.
    - It is also used for fine web data process.
```

- **Big Data process pipeline (2) -> NVIDIA NeMo Curator**
```text
    - Detect language and handle multi langauge
    - exact,fuzzy,semantic deduplication
    - Heuristic quality filtering
    - quality evaluation based on classifier
    - code data process
    - generating synthetic data
```

- **Dolma Toolkit**
```text
    - good toolkit for data procee pipeline tool
```

- **Label data preprocess**
```text

```

- **generating synthetic data**
```text

- **Distilabel**
  - famous framework to generate data for SFT, preference
        - generating multi answers
        - judge evaluate answer
        - score, reason save
        - chosen/rejected building
        - filtering and saving

- **NeMo Curator Synthetic Data**
        - generaint multilingual Q&A
        - paraphrasing
        - get knowldge
        - SFT data generating
        - preference data generating
```

### 2. GPU

- **gpu use**
```python
      allocate = torch.cuda.memory_allocated() / 1024**3 -> gb
      reserved = torch.cuda.memory_reserved() / 1024 **3
      peak = torch.cuda.max_memory_allocated() / 1024 ** 3
```

- **When memory usage faces limitation**
    - gradient checkpointing
    - optimizer state / parameter sharding
    -  torch.cuda.empty_cache() does not affect to live tensor.
    -  use bf16
    - 
      ```python
      if torch.cuda.is_available() and torch.cuda.is_bf16_supported() -> bf16 is possible to use.
      torch.bfloat16 if is_bf16 else else torch.float16
      ```

    - 

    - 





# Modular decoder-only pretraining

The original notebook modules were separated by function and connected with a
single immutable `argparse` configuration. The default public corpus is the
English `wikimedia/wikipedia` subset `20231101.en`, loaded in streaming mode and
bounded by `--max_samples`.

## Structure

```text
llm_pretraining_modular/
├── config.py               # the only argparse definition
├── main.py                 # wiring only
├── data/
│   ├── load_data.py        # Hugging Face / toy text loading
│   ├── tokenizer.py        # original regex tokenizer + vocab save/load
│   └── dataset.py          # packed next-token dataset and collator
├── model/
│   └── model.py            # RoPE, attention, blocks, decoder LM
└── train/
    └── trainer.py          # optimizer, CE loss, checkpoints, loop
```

## Install

```bash
pip install -r requirements.txt
```

## Wikipedia sample pretraining

```bash
python main.py \
  --dataset_name wikimedia/wikipedia \
  --dataset_config 20231101.en \
  --dataset_split train \
  --text_column text \
  --streaming \
  --max_samples 10000 \
  --max_vocab_size 30000 \
  --min_token_frequency 2 \
  --context_length 256 \
  --batch_size 8 \
  --block_num 8 \
  --embed_dim 512 \
  --num_heads 8 \
  --epochs 1 \
  --learning_rate 2e-4 \
  --output_dir outputs/wiki_pretrain
```

All runtime values come from `config.py`. There is no second dictionary or
`TrainingArguments` object that can silently replace a command-line value.
`vocab_size` is intentionally derived from the tokenizer dictionary instead of
being separately configurable.

## Test each file independently

Run from the project root:

```bash
python config.py
python -m data.load_data
python -m data.tokenizer
python -m data.dataset
python -m model.model
python -m train.trainer
```

Each test uses a tiny local sample. Only the real Wikipedia command requires a
network connection.

## Important scaling limit

This keeps the original simple design: selected documents are materialized in
memory, the vocabulary is built in one pass, and token IDs are packed into one
in-memory tensor. It is appropriate for debugging and sampled-corpus training,
but not yet for full multi-billion-token pretraining. A later scale-up should
replace only `data/dataset.py` with a sharded/streaming token dataset; the model,
trainer, CLI, and main wiring can remain.
