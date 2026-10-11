# Results

This document contains the results of the Model2Vec project. The results are presented in the following sections:
- [MTEB Results (English)](#mteb-results-english)
- [MMTEB Results (Multilingual)](#mmteb-results-multilingual)
- [Retrieval Results](#retrieval-results)
- [Training Results](#training-results)
- [Ablations](#ablations)

## MTEB Results (English)

Model2Vec is evaluated on MTEB, as well as two additional tasks: [PEARL](https://github.com/tigerchen52/PEARL) (a phrase representation task) and WordSim (a collection of _word_ similarity tasks). The results are shown in the table below.

Note: The `potion` and `M2V` models are our static models.

| Model                  |   Avg (All) |   Avg (MTEB) |   Class |   Clust |   PairClass |   Rank |    Ret |    STS |    Sum |   Pearl |   WordSim |
|:-----------------------|------------:|-------------:|--------:|--------:|------------:|-------:|-------:|-------:|-------:|--------:|----------:|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)        | 55.80     | 55.93      | 69.25  | 44.90  | 82.37     | 47.14  | 42.92  | 78.95  | 25.96  | 60.83  | 49.91   |
| [potion-base-32M](https://huggingface.co/minishlab/potion-base-32M)                     | 52.83     | 52.13      | 71.70  | 41.25  | 78.17     | 42.45  | 32.67  | 73.93  | 24.74  | 55.37  | 55.15   |
| [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M)                       | 51.32     | 51.08      | 70.34  | 39.74  | 76.62     | 41.79  | 31.11  | 72.91  | 25.06  | 53.54  | 50.75   |
| [M2V_base_output](https://huggingface.co/minishlab/M2V_base_output)                     | 48.77     | 47.96      | 66.84  | 33.96  | 74.90     | 39.31  | 25.36  | 68.76  | 26.61  | 54.02  | 49.18   |
| [GloVe_300d](https://huggingface.co/sentence-transformers/average_word_embeddings_glove.6B.300d)             | 45.49     | 45.82      | 62.73  | 37.10  | 72.48     | 38.28  | 21.80  | 61.52  | 26.81  | 45.65  | 43.05   |
| [BPEmb_50k_300d](https://github.com/bheinzerling/bpemb)                                  | 42.33     | 41.74      | 61.72  | 35.17  | 57.86     | 37.26  | 15.36  | 55.30  | 29.49  | 47.56  | 41.28   |


<details>
  <summary>  Task Abbreviations </summary>

For readability, the MTEB task names are abbreviated as follows:
- Class: Classification
- Clust: Clustering
- PairClass: PairClassification
- Rank: Reranking
- Ret: Retrieval
- STS: Semantic Textual Similarity
- Sum: Summarization
</details>

The results show that [potion-base-32M](https://huggingface.co/minishlab/potion-base-32M) is the most performant static embedding model. It reaches 93.21% of the performance of [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) with an average MTEB score of 52.13 while being orders of magnitude faster.

The figure below shows the relationship between the number of sentences per second and the average MTEB score. The circle sizes correspond to the number of parameters in the models (larger = more parameters).
This plot shows that the potion and M2V models are much faster than the other models, while still being competitive in terms of performance with the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model.
NOTE: for fairness of comparison, we disabled multiprocessing for Model2Vec for this benchmark. All sentence-transformers models are run with the [sentence-transformers](https://github.com/UKPLab/sentence-transformers) library's default settings for `encode`.

| ![Description](../assets/images/speed_vs_mteb_plot.png) |
|:--:|
|*Figure: The average MTEB score plotted against sentences per second. The circle size indicates model size.*|


## MMTEB Results (Multilingual)
The results for the multilingual models are shown in the table below. We compare against the [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) model, as well as other multilingual static embedding models.

Note: the MMTEB leaderboard ranks models using a [Borda count](https://en.wikipedia.org/wiki/Borda_count) over per-task ranks rather than a simple average. This rewards models that perform consistently well across all tasks, rather than those that excel on one task type while performing poorly on others.

| Model                                     | Mean (Task) | Mean (TaskType) | BitMining | Class | Clust | InstRet | MultiClass | PairClass | Rank | Ret | STS       |
| :---------------------------------------- | :---------- | :-------------- | :------------ | :------------- | :--------- | :-------------------- | :------------------------ | :------------------ | :-------- | :-------- | :-------- |
| [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) |       52.07 |           45.65 |         76.35 |          54.60 |      38.08 |                 −3.00 |                     20.12 |               75.97 |     50.20 |     33.17 | 65.35 |
| [potion-multilingual-128M](https://huggingface.co/minishlab/potion-multilingual-128M)              | 47.31   | 40.40           | 40.72         | 52.36      | 38.80  | −2.08                 | 15.95                 | 71.39               | 47.39     | 37.86     | 61.23     |
| [static-similarity-mrl-multilingual-v1](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) | 47.24       | 41.38       | 50.62     | 48.60          | 30.67      | −1.24                 | 14.74                     | 74.34           | 49.45 | 41.21 | 64.02 |
| [M2V_multilingual_output](https://huggingface.co/minishlab/M2V_multilingual_output)           | 42.13       | 35.89           | 36.88         | 49.75          | 30.09      | −0.07             | 14.34                     | 69.74               | 41.51     | 25.42     | 55.33     |

As can be seen, [potion-multilingual-128M](https://huggingface.co/minishlab/potion-multilingual-128M) is the most performant static multilingual model, reaching 90.86% of the performance of [LaBSE](https://huggingface.co/sentence-transformers/LaBSE). There are differences per task. The [static-similarity-mrl-multilingual-v1](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) model is better for retrieval and STS tasks (which can be explained by the fact that it's trained for STS), while the [potion-multilingual-128M](https://huggingface.co/minishlab/potion-multilingual-128M) model is better for classification and clustering tasks. It is important to note that the [potion-multilingual-128M](https://huggingface.co/minishlab/potion-multilingual-128M) model supports a  total of 101 languages, while [static-similarity-mrl-multilingual-v1](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) supports only 50 languages. It is also important to note that MMTEB does not include tasks for every language, and there may be a bias towards larger languages.


<details>
  <summary>  Task Abbreviations </summary>

For readability, the MMTEB task names are abbreviated as follows:

- BitMining: Bitext Mining
- Class: Classification
- Clust: Clustering
- InstRet: Instruction Retrieval
- MultiClass: Multilabel Classification
- PairClass: PairClassification
- Rank: Reranking
- Ret: Retrieval
- STS: Semantic Textual Similarity

</details>

## Retrieval Results

Some of our models are specifically designed for retrieval tasks. The results are shown in the table below, with [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) included as a transformer baseline and [potion-base-32M](https://huggingface.co/minishlab/potion-base-32M) as a general-purpose static baseline.

| Model                  |   Retrieval Score |
|:-----------------------|------------------:|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)        | 42.92              |
| [potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M)           | 35.06              |
| [static-retrieval-mrl-en-v1](https://huggingface.co/minishlab/static-retrieval-mrl-en-v1) | 34.95     |
| [potion-base-32M](https://huggingface.co/minishlab/potion-base-32M)                     | 32.67              |

As can be seen, [potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) is the most performant static retrieval model, reaching 81.69% of the performance of [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) with a retrieval score of 35.06.

## Training Results

The main results for Model2Vec training are outlined in this section.

We compare six different architectures for our main results:
- `tfidf`: A TF-IDF model with a scikit-learn `LogisticRegressionCV` on top.
- `fasttext`: A supervised [fastText](https://fasttext.cc/) classifier.
- `model2vec + logreg`: A model2vec model ([potion-base-32M](https://huggingface.co/minishlab/potion-base-32M)) with a scikit-learn `LogisticRegressionCV` on top.
- `model2vec full finetune`: A model2vec classifier with the full model finetuned, starting from [potion-base-32M](https://huggingface.co/minishlab/potion-base-32M). This uses our `StaticModelForClassification`.
- `setfit`: A [SetFit](https://github.com/huggingface/setfit/tree/main) model trained using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as a base model.
- `minilm full finetune`: A [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model, fully finetuned as a sequence classifier using the Hugging Face `Trainer`.

We use 26 classification datasets, using 1000 examples from the train set, and the full test set. The scores are weighted F1 scores on the test set. No parameters were tuned on any validation set. All datasets except `banking77` ([mteb/banking77](https://huggingface.co/datasets/mteb/banking77)) and `clinc_oos` ([clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos)) were taken from the [Setfit organization on Hugging Face](https://huggingface.co/datasets/SetFit).

| dataset                    | tfidf | fasttext | model2vec + logreg | model2vec full finetune | setfit | minilm full finetune |
|:---------------------------|------:|---------:|-------------------:|------------------------:|-------:|---------------------:|
| 20_newgroups               | 52.13 |    28.27 |              56.24 |                   58.14 |  61.28 |                58.49 |
| ade                        | 79.76 |    76.25 |              79.20 |                   75.65 |  80.23 |                82.81 |
| ag_news                    | 82.91 |    77.09 |              86.70 |                   87.00 |  87.69 |                87.95 |
| amazon_counterfactual      | 92.77 |    93.00 |              90.96 |                   91.31 |  90.81 |                94.85 |
| amazon_polarity            | 78.36 |    74.30 |              82.04 |                   83.25 |  78.87 |                87.77 |
| banking77                  | 69.60 |    53.23 |              79.95 |                   75.61 |  56.42 |                48.72 |
| bbc                        | 96.60 |    95.30 |              95.80 |                   96.41 |  94.59 |                96.70 |
| clinc_oos                  | 56.71 |    39.67 |              68.11 |                   68.89 |  41.82 |                38.53 |
| emotion                    | 61.88 |    56.48 |              65.57 |                   66.46 |  60.73 |                81.22 |
| enron_spam                 | 96.30 |    95.50 |              96.40 |                   96.95 |  96.05 |                96.85 |
| ethos                      | 61.82 |    58.31 |              69.38 |                   63.11 |  67.66 |                74.70 |
| hatespeech_offensive       | 80.59 |    77.78 |              83.54 |                   84.90 |  85.50 |                86.20 |
| imdb                       | 81.78 |    78.03 |              85.34 |                   86.08 |  82.92 |                81.30 |
| insincere_questions        | 90.79 |    91.89 |              93.50 |                   93.24 |  94.43 |                94.09 |
| massive_intent             | 67.54 |    63.99 |              73.83 |                   72.00 |  62.32 |                67.00 |
| massive_scenario           | 79.60 |    77.37 |              82.86 |                   83.91 |  81.24 |                86.53 |
| senteval_cr                | 74.89 |    73.78 |              77.03 |                   79.11 |  82.52 |                86.00 |
| sst2                       | 72.16 |    70.70 |              79.76 |                   81.33 |  82.81 |                82.26 |
| sst5                       | 31.12 |    33.02 |              32.34 |                   37.71 |  40.96 |                35.65 |
| student                    | 79.74 |    75.39 |              83.20 |                   85.60 |  87.27 |                89.22 |
| subj                       | 86.95 |    87.39 |              89.20 |                   89.79 |  89.90 |                92.65 |
| toxic_conversations        | 88.30 |    88.48 |              90.44 |                   88.30 |  91.45 |                89.93 |
| trec                       | 67.68 |    66.50 |              56.34 |                   55.97 |  57.79 |                73.58 |
| tweet_sentiment_extraction | 57.88 |    52.53 |              64.96 |                   64.45 |  72.07 |                72.79 |
| tweet_stance_abortion      | 63.58 |    64.11 |              71.99 |                   68.33 |  66.86 |                69.72 |
| yelp_review_full           | 45.21 |    41.77 |              48.42 |                   50.96 |  48.38 |                48.54 |


|         | tfidf | fasttext | model2vec + logreg | model2vec full finetune | setfit | minilm full finetune |
|:--------|------:|---------:|-------------------:|------------------------:|-------:|---------------------:|
| average |  72.9 |     68.9 |               76.3 |                    76.3 |   74.7 |                 77.1 |


The fully finetuned MiniLM model has the highest average score, followed by the two model2vec variants, which both outperform `setfit`. Full fine-tuning of model2vec scores higher than logistic regression on 17 of the 26 datasets, but loses heavily on a few, such as `ethos` and `banking77`, which leads to the same average score. Our advice is to test both if you can use `potion-base-32m`, and to use full fine-tuning if you are starting from another base model.

The table below shows inference speed. The model2vec full finetune is about 80x faster than `setfit` and about 295x faster than the `minilm full finetune`, and comes close to the speed of `tfidf` and `fasttext`.


|                  | tfidf | fasttext | model2vec + logreg | model2vec full finetune | setfit | minilm full finetune |
|:-----------------|------:|---------:|-------------------:|------------------------:|-------:|---------------------:|
| samples / second | 93718 |    96288 |              32869 |                   81018 |   1010 |                  274 |



## Ablations

To better understand the factors contributing to the performance of Model2Vec, we conducted a comprehensive set of ablation studies, covering various aspects of the model's architecture and preprocessing methods. In these studies, we examined the impact of key elements such as PCA, Zipf weighting, and the use of Sentence Transformers versus regular transformer models. We also compared the performance of input embeddings versus output embeddings, since it would seem plausible that these should also work well. The results are shown in the table below.


| Model                        |   Avg (All) |   Avg (MTEB) |   Class |   Clust |   PairClass |   Rank |   Ret |   STS |   Sum |   Pearl |   WordSim |
|:-----------------------------|------------:|-------------:|--------:|--------:|------------:|-------:|------:|------:|------:|--------:|----------:|
| M2V_base_output              |       46.79 |        45.34 |   61.25 |   25.58 |       74.9  |  47.63 | 26.14 | 68.58 | 29.2  |   54.02 |     49.18 |
| M2V_base_output_nopca        |       44.04 |        42.31 |   61.42 |   20.15 |       68.21 |  44.67 | 25.25 | 61.87 | 29.85 |   51.02 |     48.96 |
| M2V_base_output_nozipf       |       43.61 |        41.52 |   60.44 |   21.62 |       72.15 |  45.57 | 20.35 | 62.71 | 30.66 |   52.28 |     49.17 |
| M2V_base_input_nozipf_nopca  |       40.97 |        39.55 |   54.16 |   18.62 |       68.3  |  43.65 | 23.63 | 59.38 | 32.04 |   50.19 |     40.52 |
| M2V_base_output_nozipf_nopca |       40.8  |        38.44 |   59.78 |   19.31 |       62.39 |  42.26 | 19.01 | 55.16 | 30    |   49.09 |     48.97 |
| M2V_base_input               |       40.74 |        39.93 |   60.35 |   22.66 |       59.63 |  43.02 | 25.47 | 50.05 | 29.35 |   50.61 |     34.47 |
| M2V_bert_output_nozipf_nopca              |       35.54 |        34.82 |   55.69 |   15.42 |       58.68 |  39.87 | 12.92 | 55.24 | 30.15 |   46.9  |     26.72 |


There's four main findings in these results:
1. Non-Sentence Transformers do not work well. This can be seen by comparing `M2V_bert_output_nozipf_nopca` (which uses [BERT](https://huggingface.co/google-bert/bert-base-uncased), a non-Sentence Transformer) and `M2V_base_output_nozipf_nopca` (which uses [BGE-base](https://huggingface.co/BAAI/bge-base-en-v1.5), a Sentence Transformer). Using a Sentence Transformer gives a ~5.2% increase in performance.
2. PCA is crucial for performance. This can be seen by comparing `M2V_base_output_nozipf_nopca` and `M2V_base_output_nozipf` which gives a ~2.8% increase in performance. Furthermore, PCA improves performance on _all_ tasks.
3. Zipf weighting is crucial for performance. This can be seen by comparing `M2V_base_output_nozipf_nopca` and `M2V_base_output_nopca` which gives a ~3.1% increase in performance.
4. Output embeddings outperform input embeddings. This can be seen by comparing `M2V_base_input` and `M2V_base_output` which gives a ~6.1% increase in performance. Note that input embeddings do work well for some tasks. We hypothesize that this is because input embeddings are inherently normalized.
