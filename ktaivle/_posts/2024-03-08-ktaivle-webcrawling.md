---
layout: post
title: KT AIVLE SCHOOL 5기 3주차 | 웹 크롤링(Web crawling)(1)
description: KT AIVLE SCHOOL 5기 3주차에 진행한 웹 크롤링(Web crawling) 강의 내용 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}

## 0. Python

> cpu, ram, ssd(hdd) 활용

1. 변수 선언: ram 사용 -> 메모리에 데이터 저장 : 식별자
2. 데이터 타입: ram 효율적 사용: 동적 타이핑
3. 연산자: cpu 사용 : 산술, 비교, 논리, 할당, 멤버
4. 조건문, 반복문 : 코드 작성의 효율을 높이는 문법
5. 함수 : 반복되는 코드를 묶어서 코드 작성 실행 
6. 클래스 : 여러 개의 변수, 함수를 묶어서 코드 작성 실행 문법
7. 입출력 : ssd(hdd) 사용 : pickle

### 클래스

- 객체 지향 구현 문법 : 실제 세계를 모델링하여 개발하는 방법론 : 협업을 용이하게
- 함수 사용법 : 함수 선언 (코드 작성) > 함수 호출 (코드 실행)
- 클래스 사용법
    - 클래스 선언(코드 작성) > 객체 생성(메모리 사용) > 메서드(함수) 호출(코드 실행)
- 클래스 식별자 : PascalCase, UpperCamelCase - 클래스
    - snake_case - 함수
    - (PEP8 -> 맞춰서 사용하는게 좋음)
- 메서도 종류 : instance(90 ~ 95% 이상), class, static
- special method : 앞뒤로 __ 사용 : 특별한 기능을 하는 메서드
    - 생성자 메서드 : `__init__()` : 객체가 생성될 때 실행되는 메서드
        - 메서드에서 사용되는 변수의 초기값을 설정하거나 검사할 때 사용

```python
# 클래스 선언 : 코드 작성
# 은행 계좌 : Account : balance, deposit(), withdraw()

class Account:
    def __init__(self, balance): # 생성자
        self.balance = balance
        
    def deposit(self, amount): # 입금
        self.balance += amount
        
    def withdraw(self, amount): # 출금
        self.balance -= amount
```

```python
# 객체 생성 : 메모리 사용
acc1 = Account(10000)
acc2 = Account(10000)

# dir() : 객체에 들어있는 변수(함수) 출력
dir(acc1)[-3:]

# 변수 값 변경
acc2.balance = 6000

# 메서드 호출 : 코드 실행
acc1.deposit(2000)
acc2.withdraw(3000)

# help : docstring(함수설명글) 출력
# help(df.drop)
```

- 클래스는 사용자정의 데이터타입이다.

```python
# acc 객체 클래스 : Account
acc = Account(1000)

# acc 객체의 데이터 타입 : Account
type(acc)

# > 클래스는 데이터 타입이다

# Account 객체 만든 사람 : 우리가 직접 : 커스터 마이즈 : 사용자 정의
```

- 데이터 타입(=클래스)
    - 데이터 타입(클래스)가 다르다는 의미 : 사용 가능한 변수(함수)가 다름

## 1. Web

### Server & Client Architecture

- Client (요청하는 사람)
    - Request: Browser를 사용하여 Server에 데이터를 요청
- Server (요청에 응답하는 사람)
    - Response: Client의 Browser에서 데이터를 요청하면 요청에 따라 데이터를 Client로 전송

### URL

- Uniform Resource Locator
- ex) [http://news.naver.com:80/main/read.nhn?mode=LSD&mid=shm&sid1=105&oid=001&aid=0009847211#da_727145](http://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=105&oid=001&aid=0009847211#da_727145)
    - `http://` : Protocol (`https://` : 약간 보안이 강화된 형태 (요즘 많이 사용))
    - `news` : Sub Domain
    - `naver.com` : Primary Domain
        - Domain은 ip주소와 매핑 (서버 컴퓨터)
    - `80` : Port → 서버 컴퓨터의 app 접속에 사용
    - `/main/` : Path
    - `read.nhn` : Page (File)
    - **`? 앞은 file, 뒤는 Data`**
    - `mode=LSD` : Query
        - client가 server에 요청하는 data 값
    - `#da_727145` - Fragment (페이지의 위치값)
        - 브라우저 컨트롤에 사용
        - 웹크롤링에서는 잘 사용 안함
    

### HTTP Request Methods

- Get
    - URL에 Query 포함
    - Query(데이터) 노출, 전송 가능 데이터 작음
    - ex) 검색어 입력
- Post
    - Body에 Query 포함
    - Query(데이터) 비노출, 전송 가능 데이터 많음
    - ex) ID, PW 입력하는 경우

### HTTP Status Code

- Client와 Server가 데이터를 주고 받은 결과 정보
    - 2xx - Success
    - 3xx - Redirect
        - 이미 받은 정보를 다시 보여줌
        - 네트워크 트래픽 줄이고 페이지 빠르게 불러오기 위해
    - 4xx - Request Error
    - 5xx - Server Error
- [HTTP Status Code 관련 링크](http://bit.ly/2nlZM8L)

### cookie & session & cache

- Cookie
    - Client의 Browser에 저장하는 문자열 데이터
    - ssd에 저장했다가 불러오는 형식
    - 데이터 불러오는 게 느림, 가격 경쟁력은 높음(가격이 쌈)
    - 사용 예시 : 로그인 정보, 내가 봤던 상품 정보, 팝업 다시 보지 않음 등
- Session
    - Client의 Browser와 Server의 연결 정보
    - 사용 예시 : 자동 로그인
- Cache
    - Client, Server의 RAM(메모리)에 저장하는 데이터
    - RAM에 데이터를 저장하면 데이터 입출력이 빠름
    - 가격 경쟁력은 낮음 (비쌈)

### Web Language & Framework

- Framework
    - 빈칸 채우기와 비슷함
    - 기능적으로 존재하는 코드들이 이미 존재해서 남은 부분만 채워 넣으면 된다
- Client (Frontend)
    - HTML
    - CSS - Bootstrap, Semantic UI, Materialize, Material Design Lite
    - Javascript - react.js, vue.js, angular, jQuery
- Server (Backend)
    - Python - Django, Flask, FastAPI
    - Java - Spring
    - Ruby - Rails
    - Scala - Play
    - Javascript - Express(node.js)

### Scraping & Crawling

- Scraping
    - 특정 데이터를 수집하는 작업
- Crawling
    - 웹서비스의 여러 페이지를 이동하며 데이터를 수집하는 작업
    - spider, web crawler, bot 용어 사용
    - ex) google bot
    

### Internet

- 컴퓨터로 연결하여 TCP/IP 프로토콜을 이용하여 정보를 주고 받는 컴퓨터 네트워크
- 해저 케이블을 사용하여 전세계 컴퓨터에 접속
- 무선 인터넷은 매체(media)를 주파수 사용

## 2. Web Crawling

- 웹 페이지에서 데이터를 수집하는 방법

### 웹크롤링 방법

#### 웹페이지의 종류

- 정적인 페이지 : 웹 브라우져에 화면이 한번 뜨면 이벤트에 의한 화면의 변경이 없는 페이지
- 동적인 페이지 : 웹 브라우져에 화면이 뜨고 이벤트가 발생하면 서버에서 데이터를 가져와 화면을 변경하는 페이지

#### requests 이용

- 받아오는 문자열에 따라 두 가지 방법으로 구분
    - `json` 문자열로 받아서 파싱하는 방법 : 주로 **동적 페이지** 크롤링할 때 사용
    - `html` 문자열로 받아서 파싱하는 방법 : 주로 **정적 페이지** 크롤링할 때 사용

#### selenium 이용

- 브라우져를 직접 열어서 데이터를 받는 방법

#### 크롤링 방법에 따른 속도

- requests json > requests html >> selenium (브라우저가 메모리 많이 잡아먹는다)

## 3. 동적 페이지 크롤링

### Crawling Naver Stock Data

- 네이버 증권 사이트에서 주가 데이터 수집
- 수집할 데이터 : 일별 kospi, kosdaq 주가, 일별 환율(exchange rate) 데이터
- 과정
    
    ```python
    def stock_price(code = 'KOSPI', page_size = 60, page = 1): # 주가 데이터 크롤링
        
        # 1. 웹서비스 분석 > URL
        url = f'https://m.stock.naver.com/api/index/{code}/price?pageSize={page_Size}&page={page}'
        
        # 2. request(URL) > response(JSON) : JSON(str)
        response = requests.get(url)
        
        # 3. JSON(str) > list, dict > DataFrame : Data
        return pd.DataFrame(response.json())[['localTradedAt', 'closePrice']]
    
    def exchange_rate(code = 'FX_USDKRW', page_size = 60, page = 1): # 환율 데이터 크롤링
        
        # 1. 웹서비스 분석 > URL
        url = f'https://m.stock.naver.com/front-api/v1/marketIndex/prices?category=exchange&reutersCode={code}&page={page}&pageSize={page_Size}'
    
        # 2. request(URL) > response(JSON) : JSON(str)
        response = requests.get(url)
        
        # 3. JSON(str) > list, dict > DataFrame : Data
        return pd.DataFrame(response.json()['result'])[['localTradedAt', 'closePrice']]
    ```
    
    1. 웹서비스 분석 : url
        - pc 웹페이지가 복잡하면 mobile 웹페이지에서 수집
    2. 서버에 데이터 요청 : request(url) > response : json(str)
        - response의 status code가 200이 나오는지 확인
        - 403이나 500이 나오면 request가 잘못되거나 web server에서 수집이 안되도록 설정이 된것임
            - header 설정 또는 selenium 사용
        - 200이 나오더라도 response 안에 있는 내용을 확인 > 확인하는 방법 : `response.text`
    3. **서버에서 받은 데이터 파싱(데이터 형태를 변경) : json(str) > list, dict > DataFrame**
- 추가 작업
    - `page_size` = 60 으로 설정하여 KOSPI, KOSDAQ, USD 데이터 수집
    - 수집한 데이터 `min_max_scaling`하여 데이터 시각화하여 비교
    - 상관 관계 분석 진행
        1. kospi, kosdaq은 아주 강한 양의 상관 관계를 갖는다
        2. kospi, usd는 강한 음의 상관 관계를 갖는다

### Daum Exchange

- 어뷰징 방지를 위해 User-Agent가 python이면 request 방지하기도 함
    - header 설정하여 이를 회피

```python
def daum_exchange():
    # 1. URL
    url = 'https://finance.daum.net/api/exchanges/summaries'

    # header 설정 -> request 방지를 회피할 수 있다
    headers = {
        'User-Agent' : 'User-Agent 값' # 개발자 도구에서 확인 가능
        'Referer' : 'Referer 값', # 추가적으로 설정 필요한 값
    }

    # 2. request > response : json(str)
    response = requests.get(url, headers = headers)

    # 3. json(str) > list, dict : DataFrame 
    return pd.DataFrame(response.json()['data'])[['symbolCode', 'currencyCode' ,'basePrice']]
```

### REST API

- API 받아오는 것도 동적 페이지와 비슷 → json을 받아오기 때문
- 과정
    1. 어플리케이션 등록 : app_key
    2. api 문서 확인 : url, params, headers
    3. request(url, params, headers(app_key)) > response(data) : data(json(str))
    4. data(json(str)) > list, dict > DataFrame

#### Kakao API
- KoGPT
    - 다음 문장 만들기
        
        ```python
        def kogpt_api(prompt, command = '', max_tokens = 128, temperature = 1, n = 1):
            # 1. document : URL
            url = 'https://api.kakaobrain.com/v1/inference/kogpt/generation'
            
            # 2. request(URL : headers, params) > response : json(str)
            headers = {'Authorization' : f'KakaoAK {REST_API_KEY}', 'Content-Type' : 'application/json'}
            params = {'prompt' : prompt + command, 'max_tokens' : max_tokens, 'temperature' : temperature, 'n' : n}
            response = requests.post(url, json.dumps(params), headers = headers) # json.dumps : 문자열 인코딩 : 한글은 인터넷상에서 사용 X : 한글 > 영문, 특수문자로 인코딩
            
            # 3. json(str) > parsing : text
            results = response.json()['generations']
            return [result['text'] for result in results]
        ```
        
    - 추가 작업
        - 문장 분류, 한줄 요약, excel 데이터 내용들 요약하여 새로운 열 생성
            
#### Naver API
- 통합 검색어 트렌드 API
- 위의 Kakao API 받아오는 방식과 거의 동일
- sdk : API를 통해서 얻어온 정보를 데이터프레임으로 만들어 주는 패키지를 제공하는 서비스
    - 구글은 sdk 제공

```python
keywordGroups = [{'groupName' : '트위터', 'keywords' : ['트위터', '트윗']},
        {'groupName' : '페이스북', 'keywords' : ['페이스북', '페북']},
        {'groupName' : '인스타그램', 'keywords' : ['인스타그램', '인스타']},
]

def naver_api(startDate = '2018-01-01', endDate = '2024-03-01', timeUnit = 'month', keywordGroups = keywordGroups):
    url = 'https://openapi.naver.com/v1/datalab/search'
    params = {
        'startDate' : startDate,
        'endDate' : endDate,
        'timeUnit' : timeUnit,
        'keywordGroups' : keywordGroups,
    }

    headers = {
        'X-Naver-Client-Id' : CLIENT_ID, # App key
        'X-Naver-Client-Secret': CLIENT_SECRET, # App key
        'Content-Type' : 'application/json'
    }
    
    response = requests.post(url, json.dumps(params), headers = headers)
    data = response.json()['results']
    dfs = []
    for row in data:
        df = pd.DataFrame(row['data'])
        df['title'] = row['title']
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index = True)
```
    

### 직방 매물 정보 가져오기

- 과정
    
    ```python
    addr = '망원동'
    # 1. 동이름으로 위도 경도 구하기
    url = f'https://apis.zigbang.com/v2/search?leaseYn=N&q={addr}&serviceType=원룸'
    response = requests.get(url)
    data = response.json()['items'][0]
    lat, lng = data['lat'], data['lng']
    
    # 2. 위도 경도로 geohash 알아내기
    geohash = geohash2.encode(lat, lng, precision = 5)
    
    # 3. geohash로 매물 아이디 가져오기
    url = f'https://apis.zigbang.com/v2/items/oneroom?\
    geohash={geohash}&depositMin=0&rentMin=0&salesTypes[0]=전세&salesTypes[1]=월세\
    &domain=zigbang&checkAnyItemWithoutFilter=true'
    response = requests.get(url)
    items = response.json()['items']
    items = [item['itemId'] for item in items]
    
    # 4. 매물 아이디로 매물 정보 가져오기
    url = 'https://apis.zigbang.com/v2/items/list'
    params = {'domain' : 'zigbang', 'item_ids' : items}
    response = requests.post(url, params)
    columns = ['item_id', 'sales_title', 'deposit', 'rent', 'size_m2', 'title', 'address1']
    df = pd.DataFrame(response.json()['items'])[columns]
    df = df.loc[df['address1'].str.contains(addr)].reset_index(drop = True)
    df.tail(2)
    ```