## python main.py 로 실행하면 됩니다.
- main.py → vocab.py → custom_dataset.py → main.py → model.py → main.py 
- https://arxiv.org/pdf/1808.08762.pdf
- 600D HBMP, paper accuracy : 86.6%
## 모델 성능
- test acc : 84.6%
- dev acc : 84.9%

## HBMP: Hierarchical BiLSTM with Max Pooling
+ intro
	+ iterative refinement strategy (hierarchy of BiLSTM and max pooling)
	+ model the inferential relationship between two or more given sentences
	+ In particular, given two sentences - the premise p and the hypothesis h
+ (refer) Natural Language Inference by Tree-Based Convolution and Heuristic Matching 
	+ previous reference
		+ linear offset of vectors can capture relationships between two words
		+ but it has not been exploited in sentence-pair relation recognition.
	+ Our study verifies that vector offset is useful in capturing generic sentence relationships
	+ sentence embeddings are combined using a heuristic
	+ m = [h1; h2; h1 − h2; h1 ◦ h2]  (concat, difference, product)
	+ vector representations of individual sentences are combined to capture the relation between the p and h
	+ As the dataset is large, we prefer O(1) matching operations because of efficiency concerns. 
	+ first one (concat) = 
		+ the most standard procedure of the “Siamese” architectures, 
		+ W[h1, h2], where W = [W1, W2]
	+ latter two (difference, product) = 
		+ certain measures of “similarity” or “closeness.”
		+ special cases of concatenation
		+ element-wise product, 용량관점에서 concat에 포함 (same)
		+ element-wise difference, W0(h1−h2) = [W0, −W0][h1, h2]T
	+ heuristic significantly improves the performance
+ (refer) Supervised Learning of Universal Sentence Representations from Natural Language Inference Data 
	+ hierarchical 장점 
		+ (attentive 세심하게, 주의 깊게) 이전 레이어의 가중치를 받아 같은 동작을 다시 반복함으로써 attentive 해짐
		+ concat 으로 계층마다 다른 관점들을 블랜딩 함으로써 표현이 주의깊은 추상화가 됨
		+ representations of the sentences at different level of abstractions 을 max-pooling 으로 추출
			+ 그러니까 max pooling 으로 인하여 level 마다 관점들이 더 두드러짐
	+ fully connected layer 가 한개
+ iterative refinement architecture
	+ 이전 LSTM 레이어의 info을 다음 레이어의 initialisation 함으로써 반복적 정제 아키텍쳐를 가진다고 함.
+ Max pooling is defined in the standard way of taking the highest value over each dimension of the hidden states = u
+ sentence encoder 에서의 u1, u2, u3 은 concat 하여 나오고, premise와 hypothesis, 2개의 encoder가 있음.
