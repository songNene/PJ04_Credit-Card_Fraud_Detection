
# 💳 Credit Card Fraud Detection

이 프로젝트는 **신용카드 거래 데이터를 분석하여 사기 거래를 탐지**하는 머신러닝 분류 모델을 구축하는것입니다.
데이터는 개인정보 보호를 위해 익명화되었으며, Class 변수가 0(정상 거래), 1(사기 거래)로 이진 분류 문제입니다.

## 📂 데이터 설명

- `id`: 각 거래의 고유 식별자
- `Time`: 첫 거래 이후 경과 시간 (초 단위)
- `V1` ~ `V28`: PCA로 변환된 익명화된 피처
- `Amount`: 거래 금액
- `Class`: 타겟 변수 (0 = 정상 거래, 1 = 사기 거래)

> 데이터는 비정상 거래(Class=1)가 매우 적은 **심각한 클래스 불균형 문제**를 갖고 있습니다.

---

## 🔍 EDA (탐색적 데이터 분석)

- Class 비율: 정상 거래가 대부분이며, 사기 거래는 극소수
- 사기 거래는 소액일 가능성이 높음
- Amount는 log 변환으로 분포 왜곡을 줄임
- 상관관계 분석: 일부 변수는 Class와 음의 상관이 강함 (예: V14, V17 등)

---

## ⚙️ 데이터 전처리

- `Amount` 피처 로그 변환 (`log1p`)
- `Time`, `id`, `Amount` 제거
- `SMOTE`를 이용해 Class 불균형 해결 (Synthetic Minority Oversampling)
- 스케일링: `StandardScaler`

---

## 🧠 모델 학습

### 사용한 모델
- **Logistic Regression**
- **LightGBM**
- **XGBoost**

### 평가 지표
- 정확도 (Accuracy)
- 정밀도 (Precision)
- 재현율 (Recall)
- F1-score
- ROC-AUC

### 하이퍼파라미터 튜닝
- `GridSearchCV`와 `RandomizedSearchCV`로 최적 파라미터 탐색
- 평가 지표로 `roc-auc`를 우선적으로 사용

## 📌 사용 라이브러리

- pandas, numpy
- scikit-learn
- imbalanced-learn
- lightgbm
- xgboost
- matplotlib, seaborn

---

## 🏆 결과 요약

- SMOTE를 통해 Recall이 향상되었으나 Precision이 낮아지는 trade-off 발생
- LightGBM이 전체적으로 가장 안정적이며 높은 ROC-AUC 성능을 보임
- Logistic Regression은 baseline으로 적합
- XGBoost는 튜닝 이후 성능 개선이 뚜렷함

---

## 🔁 회고

이번 메인퀘스트를 통해 **분류 문제에서의 전처리 중요성**을 깊이 느낄 수 있었습니다.  
특히, 단순히 스케일링이나 결측치 처리뿐 아니라 **도메인 지식과 데이터 자체에 대한 이해도**가 정확한 전처리에 큰 영향을 준다는 것을 깨달았습니다.  

아직 데이터의 의미를 정확히 해석하고 적용하는 능력은 부족하다고 느꼈고,  
이 프로젝트를 통해 **분류 모델의 설계, 데이터 불균형 처리, 전처리 기법의 중요성**에 대해 더 깊이 공부할 수 있는 계기가 되었습니다.

---

## 📁 파일 구성
- sample_submission.csv
- test.csv
- train.csv

