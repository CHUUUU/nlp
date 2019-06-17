## SNLI - The Stanford Natural Language Processing Group
- https://nlp.stanford.edu/projects/snli/
- data : SNLI 1.0

## Task
- classification task
- inferential relationship between two or more given sentences (premise, hypothesis) 
- labels = ["neutral", "contradiction", "entailment"]  

## model evaluation
|                  | Train ACC | Validation ACC | Test ACC | Paper ACC |
| :--------------- | :-------: | :------------: | :------: | :------: |
| Baseline (Feed Forward)         |  -  |     -     |  -  |  -  |
| HBMP           |  -  |  84.62%  |  84.9%  |  86.6%  |

## python main.py 로 실행하면 됩니다.
- main.py → vocab.py → custom_dataset.py → main.py → model.py → main.py 
- https://arxiv.org/pdf/1808.08762.pdf
- 600D HBMP, paper accuracy : 86.6%
## pretain
- Glove (glove.840B.300d.zip, 840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB)
- https://nlp.stanford.edu/projects/glove/
