# 🚀 Dart-B 2024-1st MainProject <KBO team Hanhwa baseball analysis about player and issues>
<img width="810" alt="스크린샷 2024-07-03 오후 10 59 44" src="https://github.com/yousrchive/kbo_hanhwa/assets/147587058/ce3153b2-594f-4338-9cdd-a8452019303d">

## 📚 목차

1. [기획배경](#기획배경)
2. [프로젝트 기간](#프로젝트-기간)
3. [팀원 소개](#팀원-소개)
4. [기능 소개](#기능-소개)
5. [모델 설계서](#모델-설계서)
6. [API 설계서](#api-설계서)
7. [ERD](#erd)
8. [시스템 아키텍처](#시스템-아키텍처)
9. [데이터 명세서](#데이터-명세서)
10. [기술 스택](#기술-스택)
11. [결과물](#결과물)

## 🎯 기획배경

" 왜 중앙대학교 후문에는 포토부스가 하나도 없을까? 내가 세우면 사람들이 오려나? "

- “이따 사진 한 장 찍을까?”
당연시되고 있는 우리 주위의 포토부스.
셀프 포토부스가 성행하는 시점에서 어떤 요소가 포토부스의 입점에 관련되어 있고,
새롭게 포토부스를 설치하고자 한다면 어디에 입점시키는 것이 가장 좋을까?
  
## ⏰ 프로젝트 기간

24년 6월 20일 (수) ~ 24년 4월 3일 (수) (2주)

## 👥 팀원 소개

| 이름 | 역할 |
|---|---|
| 이은학(팀장) | 프론트엔드, 웹 서빙 |
| 심영보 | 데이터 사이언티스트(데이터 수집, 추천 모델링 개발) |
| 이유정 | 데이터 사이언티스트(데이터 수집, 정제, 시각화) |
| 김나현 | 데이터 애널리스트(데이터 시각화) |

## 📄 모델 설계서
  ### 데이터

1) 서울시 가구 특성정보 - 소득정보
2) 소상공인시장진흥공단_상가(상권)정보
3) 한국대학및전문대학정보표준데이터 & 전국초중등학교위치표준데이터
4) 상권별 소규모 상가 임대료
5) 서울시 상권분석서비스(추정매출-상권)

+ 서울시 내 랜덤 스팟의 포토부스 개수 및 상관 피쳐 개수 (300개)

<img width="620" alt="스크린샷 2024-05-30 오후 3 07 35" src="https://github.com/yousrchive/Ideal-location-for-Photobooth/assets/147587058/c77caa7c-33d0-4fcf-8256-71a62497acf7">

<img width="656" alt="스크린샷 2024-05-30 오후 3 07 56" src="https://github.com/yousrchive/Ideal-location-for-Photobooth/assets/147587058/eb10a521-89da-4386-b4ac-48ef02cc81b3">

```
import random
import csv

def get_random_location_within_seoul():
    # 서울 내의 경도, 위도 범위
    min_lat, max_lat = 37.4264, 37.6922
    min_lng, max_lng = 126.7645, 127.1833

    # 무작위 지점 좌표 생성
    random_locations = []
    for _ in range(1000):
        random_lat = random.uniform(min_lat, max_lat)
        random_lng = random.uniform(min_lng, max_lng)
        random_locations.append((random_lat, random_lng))

    return random_locations

def save_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['latitude', 'longitude'])
        writer.writerows(data)

# 서울 내의 무작위 지점 1000개의 위도와 경도 좌표 가져오기
random_locations_seoul = get_random_location_within_seoul()

# CSV 파일로 저장
save_to_csv(random_locations_seoul, 'sampling.csv')

#학교 술집 노래방 개수 검색

import requests
import csv

def get_total_count(keyword, longitude, latitude, radius, format_type):
    url = "https://dapi.kakao.com/v2/local/search/keyword.{}"
    headers = {"Authorization": "KakaoAK {}".format(REST_API_KEY)}
    params = {
        "query": keyword,
        "x": longitude,
        "y": latitude,
        "radius": radius
    }a
    response = requests.get(url.format(format_type), params=params, headers=headers)
    data = response.json()
    if 'meta' in data and data['meta']['total_count'] > 0:
        return data['meta']['total_count']
    else:
        return 0

def search_and_save_results(locations, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['latitude', 'longitude', 'school_count', 'alc_count', 'sing_count'])

        for latitude, longitude in locations:
            school_count = get_total_count("학교", longitude, latitude, 500, 'json')
            alc_count = get_total_count("술집", longitude, latitude, 500, 'json')
            sing_count = get_total_count("노래방", longitude, latitude, 500, 'json')

            writer.writerow([latitude, longitude, school_count, alc_count, sing_count])

# 저장된 무작위 좌표 읽기
def read_random_locations_from_csv(filename):
    random_locations = []
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 스킵
        for row in reader:
            latitude, longitude = float(row[0]), float(row[1])
            random_locations.append((latitude, longitude))
    return random_locations

# API 키
REST_API_KEY = "2f55e6eb59d65a3dbf95fa448046683d"

# CSV 파일에서 무작위 좌표 읽기
random_locations = read_random_locations_from_csv("sampling.csv")

# 검색하고 결과 저장
search_and_save_results(random_locations, "search_results.csv")

# search_results.csv: 검색 결과가 저장된 파일 읽기
search_results_df = pd.read_csv("search_results.csv")
photobooth_df = pd.read_csv(urls['photobooth_df'])

# 반경 500m 이내의 포토부스 개수 계산
photobooth_counts = []
for i, row in search_results_df.iterrows():
    latitude, longitude = row['latitude'], row['longitude']
    # 반경 500m 이내에 있는 포토부스 개수 계산
    count = photobooth_df[
        (photobooth_df['위도'] - latitude) ** 2 + (photobooth_df['경도'] - longitude) ** 2 <= 0.000005
    ].shape[0]
    photobooth_counts.append(count)

# 결과를 DataFrame에 추가
search_results_df['photobooth'] = photobooth_counts

# 결과 저장
search_results_df.to_csv("search_results_with_photobooth.csv", index=False)
```

### 모델 학습 및 추론 결과

#### 모델학습
```
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 데이터 불러오기
data = pd.read_csv('search_results_with_photobooth.csv')

# 입력 변수와 타겟 변수 분리
X = data[['school_count', 'alc_count', 'sing_count']]
y = data['photobooth']

# k-fold 교차 검증
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# 선형 회귀 모델 생성
model = LinearRegression()

# 교차 검증 수행 및 모델 학습
mse_scores = []
mae_scores = []
r2_scores = []
rmse_scores = []
corr_scores = []
for train_idx, test_idx in k_fold.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    mse_score = mean_squared_error(y_test, model.predict(X_test))
    mae_score = mean_absolute_error(y_test, model.predict(X_test))
    r2_score = model.score(X_test, y_test)
    mse_scores.append(mse_score)
    mae_scores.append(mae_score)
    r2_scores.append(r2_score)
    rmse_score = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    corr_score = np.corrcoef(y_test, model.predict(X_test))[0, 1]
    rmse_scores.append(rmse_score)
    corr_scores.append(corr_score)

# 교차 검증 결과 출력
print("Mean Squared Error:", np.mean(mse_scores))
print("RMSE:", np.mean(rmse_scores))
print("Mean Absolute Error:", np.mean(mae_scores))
print("R-squared:", np.mean(r2_scores))
print("Correlation Coefficient:", np.mean(corr_scores))

#다중공선성 확인

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data

# 입력 변수의 VIF 계산
vif_scores = calculate_vif(X)
print(vif_scores)
```
#### 추론 결과
<img width="220" alt="스크린샷 2024-05-30 오후 3 08 13" src="https://github.com/yousrchive/Ideal-location-for-Photobooth/assets/147587058/6045e4d0-7a9a-4a4f-b0ef-44211bb23560">

input을 정확한 주소, 혹은 키워드로 입력해도 kakao api 기준 피쳐(현재 술집, 노래방)를 통해 주요 상권을 분석, 예상 포토부스 개수와 실제 포토부스 개수가 나옵니다.

if ( 예상 포토부스 개수 < 실제 포토부스 개수) = 포화상태
if ( 예상 포토부스 개수 = 실제 포토부스 개수) = 정석 개수 수준
if ( 예상 포토부스 개수 > 실제 포토부스 개수) = 추가 포토부스 설치 고려 가능 지역


##### 모델 서빙 플로우 설계

##### 추천서비스 로직 구성

핵심 기능: 
1) 키워드를 입력하면 좌표값을 반환할 수 있다
2) 좌표, 범위 지정 후 키워드를 쿼리해 반경 내 요소를 검색할 수 있다

<img width="599" alt="스크린샷 2024-05-30 오후 3 11 42" src="https://github.com/yousrchive/Ideal-location-for-Photobooth/assets/147587058/973cb5e5-5a87-424d-9392-edf6919a0a27">

<img width="599" alt="스크린샷 2024-05-30 오후 3 11 55" src="https://github.com/yousrchive/Ideal-location-for-Photobooth/assets/147587058/02bb0236-f9d5-4902-bb63-b6c7117a6c75">

```
import requests
import pandas as pd
import joblib

REST_API_KEY = "secret"

def get_coordinates(keyword):
    url = "https://dapi.kakao.com/v2/local/search/keyword"
    headers = {"Authorization": "KakaoAK {}".format(REST_API_KEY)}
    params = {"query": keyword}
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    if 'documents' in data and data['documents']:
        # 검색 결과 중 첫 번째 항목의 좌표를 반환
        latitude = data['documents'][0]['y']
        longitude = data['documents'][0]['x']
        return latitude, longitude
    else:
        print("검색 결과를 찾을 수 없습니다.")
        return None, None

def get_nearby_place_count(latitude, longitude, keyword):
    url = "https://dapi.kakao.com/v2/local/search/keyword"
    headers = {"Authorization": "KakaoAK {}".format(REST_API_KEY)}
    params = {
        "query": keyword,
        "x": longitude,
        "y": latitude,
        "radius": 500
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    if 'meta' in data and data['meta']['total_count'] > 0:
        return data['meta']['total_count']
    else:
        return 0

# 사용자로부터 키워드 입력 받기
keyword = input("장소에 관한 키워드를 입력하세요: ")

# 키워드를 이용하여 좌표 검색
latitude, longitude = get_coordinates(keyword)
if latitude is not None and longitude is not None:
    # 좌표를 이용하여 반경 500m 이내의 "학교", "술집", "노래방"의 개수 검색
    places = ["학교", "술집", "노래방"]
    place_counts = {}
    for place in places:
        count = get_nearby_place_count(latitude, longitude, place)
        place_counts[place] = count

    # 검색 결과 출력
    print("검색한 장소:", keyword)
    print(f"좌표: (위도: {latitude}, 경도: {longitude})")
    print("반경 500m 이내의 장소 개수")
    for place, count in place_counts.items():
        print(f"{place}: {count}")
    search_results = {
        "school_count": [place_counts.get("학교", 0)],
        "alc_count": [place_counts.get("술집", 0)],
        "sing_count": [place_counts.get("노래방", 0)],
    }
    search_df = pd.DataFrame(search_results)

else:
    print("검색 결과를 찾을 수 없습니다.")

# 모델 다운로드 함수 정의
def download_model(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# 모델 다운로드
model_url = "https://github.com/DartB-2024-1st-Toy-Project/Photobooth/raw/main/server/save.pkl"
model_save_path = "save.pkl"  # 저장할 파일 이름
download_model(model_url, model_save_path)

# 모델 불러오기
model = joblib.load(model_save_path)

# 모델에 입력하여 결과 예측
predicted_photobooth_count = model.predict(search_df)

# 결과 출력
print("예상 포토부스 개수:", predicted_photobooth_count[0])

# 실제 포토부스 개수 검색
photobooth_df = pd.read_csv(urls['photobooth_df'])
latitude = float(latitude)
longitude = float(longitude)

count = photobooth_df[
    ((photobooth_df['위도'] - latitude) ** 2 + (photobooth_df['경도'] - longitude) ** 2) <= 0.000005
].shape[0]
print("실제 포토부스 개수:", count)
```

## 🌐 API 설계서

### 카카오 API
- **기능**: 키워드 검색을 통한 입력 위치 설정 
- **요청 URL**: `https://dapi.kakao.com/v2/maps/sdk.js?appkey=5b475d258fb5e345e3944cb9418a3f5b&libraries=services`
- **요청 방식**: GET
- **요청 파라미터**: `query` (검색어)
- **응답 형식**: JSON
- **에러 코드**: 401 (인증 실패), 404 (찾을 수 없음)

## 🏗️ 시스템 아키텍처

![웹아키택처]

## 📄 데이터 명세서

### 1. 가구_특성정보

- **항목**: 가구_특성정보_(+소득정보)_211203.csv
- **칼럼**: adstrd_nm, adstrd_cd, legaldong_nm, legaldong_cd, tot_po, tot_hshld_co, hshld_per_po, ave_income_amt
- **갯수**: 19577
- **데이터타입**:
  - astrd_nm, legaldong_nm : object
  - adstrd_cd, legaldong_cd, tot_po, tot_hshld_co, hshld_per_po, ave_income_amt: float

### 2. 대한민국 대학 및 전문대학 데이터

- **항목**: 대한민국 대학 및 전문대학 데이터
- **칼럼**: 학교명, 학교영문명, 본분교구분명, 학교구분명, 설립형태구분명, 시도코드, 시도명, 소재지도로명주소, 소재지지번주소, 도로명우편번호, 소재지우편번호, 홈페이지주소, 대표전화번호, 대표팩스번호, 설립일자, 기준연도, 데이터기준일자, 제공기관코드, 제공기관명, 지번주소
- **갯수**: 441
- **데이터타입**:
  - 시도코드, 도로명우편번호, 소재지우편번호, 대표전화번호, 대표팩스번호, 설립일자, 기준연도, 데이터기준일자, 제공기관코드: int
  - 학교명, 학교영문명, 본분교구분명, 학교구분명, 설립형태구분명, 시도명, 소재지도로명주소, 소재지지번주소, 홈페이지주소, 제공기관명 : object

### 3. 서울시 상권분석서비스(소득소비-상권배후지)

- **항목**: 서울시 상권분석서비스(추정매출-자치구)
- **칼럼**: 
- **갯수**: 
- **데이터타입**:


### 4. 서울시 상권분석서비스(추정매출-자치구)

- **항목**: 서울시 상권분석서비스(추정매출-자치구)
- **칼럼**: 
- **갯수**: 
- **데이터타입**:


### 5. 관광지별 추천 데이터

- **항목**: 관광지별 추천 데이터
- **칼럼**: ID, 이름, 태그, 조회, 임베딩, 태그임베딩, 유사관광지1, 유사관광지2, 유사관광지3, 유사관광지1의 유사도, 유사관광지2의 유사도, 유사관광지3의 유사도
- **갯수**: 1465
- **데이터타입**:
  - ID: object (웹페이지 구분자)
  - 조회: float
  - 임베딩(설명 핵심키워드의 임베딩 값), 태그 임베딩(태그 임베딩 값): array
  - 이름, 태그, 유사관광지1, 유사관광지2, 유사관광지3(각 관광지별 유사 관광지), 유사관광지1의 유사도, 유사관광지2의 유사도, 유사관광지3의 유사도(각 관광지별 유사 관광지 유사도): object

## 💻 기술 스택

| 분류 | 항목 |
|---|---|
| FRONT-END | HTML, CSS |
| BACK-END | Django, Python, Kakao API |
| DATA | Python |
| 협업 | Colab |

## 🎈 결과물


## 📎 이슈
1. 행정동, 법정동, 위치 정규화 -> 병합의 한계
2. 모든 구, 동에 대한 임대료 정보가 없음 -> 각 포토부스 내 거리 상 가까운 3곳을 뽑아 거리에 가중치를 두어 평균을 구함
3. 데이터 정규화: 스튜디오 구분
4. 웹의 input, output 결정 상 고민
5. 주위 {n}m 내의 {feature}, 반경 기수 설정
6. 머신러닝 학습 평가지표, 라벨의 부재 -> 3000개 샘플 데이터 수집
