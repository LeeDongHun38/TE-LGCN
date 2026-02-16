
# Topic-Enhanced LightGCN (TE-LGCN) Research Pipeline

## 1. 문제 정의 (Problem Formulation)

* 
**배경:** 추천 시스템에서의 정보 과부하 문제와 기존 협업 필터링(CF) 모델의 한계.


* **핵심 문제:**
* 
**데이터 희소성 (Data Sparsity):** 대다수의 사용자와 아이템이 상호작용이 매우 적음 (Long-tail 문제).


* 
**Cold-Start:** 상호작용 기록이 부족한 신규 사용자나 아이템에 대한 추천 품질 저하.


* 
**기존 LightGCN의 한계:** 상호작용 데이터에만 의존하기 때문에 희소 환경에서 성능이 급격히 저하됨.




* 
**목표:** 영화 줄거리(Plot Summary)와 같은 텍스트 정보를 활용하여 그래프 구조를 확장하고 추천 성능을 개선.



## 2. 데이터 구축 및 전처리 (Data Preparation)

### 2.1 데이터셋 구성

* 
**기본 데이터:** MovieLens 데이터셋 사용.


* 
**텍스트 데이터 수집:** 기존 데이터셋의 단순 키워드 한계를 극복하기 위해, IMDb에서 각 영화의 **상세 줄거리(Plot Summary)**를 웹 크롤링하여 수집.



### 2.2 데이터 필터링 (Preprocessing)

* 
**K-Core Filtering:** 그래프 연결성 보장을 위해 사용자와 아이템 모두 최소 10회 이상의 상호작용이 있는 데이터만 남김 ().


* 
**Implicit Feedback 변환:** 5점 만점의 평점 중 4점 이상()인 경우만 긍정적 상호작용(1)으로 간주하고 나머지는 무시.


* 
**텍스트 전처리:** 소문자 변환, 특수문자 제거, 불용어(Stopword) 제거, 표제어 추출(Lemmatization) 수행 .



## 3. 이중 향상 전략 (Dual Enhancement Strategy)

텍스트 정보를 활용하여 그래프와 임베딩을 강화하는 두 가지 병렬 프로세스입니다.

### 3.1 전략 1: 의미론적 초기화 (Semantic Initialization)

* 
**방법:** **Doc2Vec** 모델을 사용하여 전처리된 영화 줄거리를 고정된 크기의 벡터로 변환.


* 
**적용:** 생성된 Doc2Vec 벡터()를 아이템 노드의 **초기 임베딩()**으로 설정.


* 
**효과:** 무작위 초기화(Random Initialization) 대신 텍스트의 의미적 정보를 포함한 상태로 학습을 시작하여 Cold-start 문제 완화.



### 3.2 전략 2: 구조적 확장 (Structural Expansion)

* 
**방법:** **LDA (Latent Dirichlet Allocation)** 토픽 모델링을 수행하여 잠재된 개의 토픽을 추출 ().


* 
**연결 생성:** 특정 아이템이 특정 토픽에 속할 확률이 임계값()을 초과할 경우, 아이템과 토픽 사이에 엣지(Edge)를 생성.


* 
**효과:** 직접적인 상호작용이 없는 아이템들 사이를 토픽(Topic) 노드를 통해 의미적으로 연결하는 경로(Semantic Bridge) 제공.



## 4. 이종 그래프 구축 (Heterogeneous Graph Construction)

* 기존의 사용자-아이템(User-Item) 이분 그래프(Bipartite Graph)를 확장.


* **구성 요소:**
* 
**노드(Nodes):** 사용자(), 아이템(), **토픽()**.


* 
**엣지(Edges):** 사용자-아이템 상호작용() + **아이템-토픽 연결()**.




* 이 구조를 통해 사용자의 선호도가 아이템을 넘어 토픽으로, 다시 다른 아이템으로 전파될 수 있음.



## 5. 모델 학습 및 최적화 (Model Training)

### 5.1 메시지 패싱 (Message Passing)

* 
**User Update:** 연결된 아이템들로부터 정보를 집계.


* 
**Item Update (핵심):** 연결된 사용자(협업 신호)뿐만 아니라, 연결된 **토픽(의미적 신호)**들로부터도 정보를 함께 집계하여 임베딩 업데이트.


* 
**Layer Aggregation:** 개의 레이어를 통과한 임베딩들의 평균을 최종 임베딩으로 사용 ().



### 5.2 손실 함수 (Loss Function)

다음 세 가지 요소의 합으로 모델을 최적화함.

1. 
**BPR Loss:** 관측된 상호작용(Positive)이 관측되지 않은 것(Negative)보다 높은 점수를 갖도록 학습.


2. 
**L2 Regularization:** 사용자, 아이템, 토픽 임베딩의 과적합 방지.


3. 
**Content Consistency Loss:** 학습된 최종 아이템 임베딩이 초기 Doc2Vec 의미 벡터와 지나치게 달라지지 않도록 제약.



## 6. 평가 및 결론 (Evaluation & Conclusion)

* 
**평가 지표:** Precision@10, Recall@10, NDCG@10 사용 .


* 
**결과:** Vanilla LightGCN 대비 Recall@10 기준 **26.4% 성능 향상** 달성.


* 
**의의:** 협업 필터링과 콘텐츠 기반 필터링을 단일 그래프 프레임워크 내에서 효과적으로 결합하여 희소성 문제를 해결함.