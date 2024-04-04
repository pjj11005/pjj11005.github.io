---
layout: post
title: 3주차 | 웹 크롤링(Web crawling)(2)
description: KT AIVLE SCHOOL 5기 3주차에 진행한 웹 크롤링(Web crawling) 강의 내용 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}

## 정적 페이지 크롤링

### HTML

- 웹 문서를 작성하는 마크업 언어

#### 구성 요소

- Document : 한 페이지를 나타내는 단위
- Element : 하나의 레이아웃을 나타내는 단위 : 시작 태그, 끝 태그, 텍스트로 구성
- Tag : Element의 종류를 정의 : 시작 태그(속성 값), 끝 태그
- Attribute : 시작 태그에서 태그의 특정 기능을 하는 값
    - id : 웹 페이지에서 유일한 값
    - class : 동일한 여러 개의 값 사용 가능 : element들을 그룹핑 할 때 사용
    - attr : id와 class를 제외한 나머지 속성들
- Text : 시작 태그와 끝 태그 사이에 있는 문자열
- Element는 서로 계층적 구조를 가질 수 있다

#### 구조

- DOCTYPE
    - 문서의 종류를 선언하는 태그
- html
    - head
        - meta
            - 웹페이지에 대한 정보
        - title
            - 웹페이지의 제목 정보
    - body
        - 화면을 구성하는 Element
    
    ```html
    <!-- HTML 웹문서의 기본적인 구조 -->
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title></title>
    </head>
    <body>
    
    </body>
    </html>
    ```
    

#### 태그

- html에서 문자를 나타내는 태그들
- head
    - title을 나타낼 때 사용
    - head는 총 6가지 종류의 태그가 있다 (h1~h6)
    - 숫자가 커질수록 문자의 크기가 줄어든다
- p: 한 줄, span : 한 블럭, pre : 줄 바꿈이나 띄어쓰기
- code
    - 코드를 작성하는 태그
    - 들여쓰기나 두 칸 이상의 공백은 적용이 안됨

#### 문자 이외의 HTML 태그

- div : 레이아웃(가장 많이 사용됨), table : 테이블, (ul, li) : 리스트, image : 이미지
- a
    - 링크를 나타내는 태그
    - href 속성에 url을 넣습니다.
        - url과 상대 경로를 모두 사용 가능
        - `target="_blank"`는 링크를 열 때 새 탭에서 열도록 하는 기능이 있다
- iframe
    - 외부 url 링크 페이지를 보여주기 위한 element
    - 모든 웹 페이지를 보여줄 수 있는건 아니고 iframe으로만 출력이 되던가 안되거나 하는 등의 설정을 할 수 있다
- input
    - text: 문자열, password : 비밀번호, checkbox : 여러 개의 버튼, button : 버튼, textarea : 여러 줄, (select, option) : 옵션 선택
    - radio
        - 여러 개의 버튼 중에서 한 개의 버튼만 체크되는 버튼
        - radio 버튼은 name 속성 값으로 그룹핑
    

### CSS Selector

- CSS 스타일을 적용 시킬 HTML 엘리먼트를 찾기 위한 방법
- 하나의 엘리먼트 선택
    - Tag : `div`
    - id : `#data`
    - class : `.data`
    - attr : `[value='no1']`
- `.data:not(.data1)` : data 클래스 모두에서 data1 클래스 엘리먼트 제외
- `.data:nth-child(3)` : 3번째 하위 엘리먼트에서 .data 클래스 가진 엘리먼트 선택
- `공백` : #data .btn : data 아이디 하위의 모든 btn 클래스 엘리먼트를 선택
- `>` : #data > .btn : data 아이디 한 단계 아래 btn 클래스 엘리먼트를 선택
- `.d1, .d2` : d1, d2 클래스 둘 다 선택

- **위의 방법들을 잊었을 때 : chrome 개발자 도구에서 copy selector하면 selector 얻을 수 있음**

### 네이버 연관 검색어 수집

- 정적 웹페이지 데이터 수집
- BeautifulSoup을 이용하여 HTML 문자열 데이터 parsing
    - html 포멧의 데이터를 css-selector를 이용하여 필요한 데이터 추출
- 과정
    
    ```python
    def naver_relational_keyword(query = '삼성전자'):
        # 1. 웹페이지 분석 : URL
        url = f'https://search.naver.com/search.naver?query={query}'
        
        # 2. **request(url, params, headers)** > response : str(html)
        response = requests.get(url)
        
        # 3. **str(html) > bs object**
        dom = BeautifulSoup(response.text, 'html.parser')
        
        # 4. bs object > .select(css-selector), .select_one(css-selector) > str(text)
        elements = dom.select('.lst_related_srch > .item')
        keywords = [element.text.strip() for element in elements]
        
        # 5. str(text) > DataFrame
        df = pd.DataFrame(keywords, columns = ['keywords'])
        
        return df
    ```


### Gmarket

- 베스트 상품 200개 데이터 수집
- 상품의 이미지 200개 다운로드
- 과정
    
    ```python
    def gmarket():
        # 1. URL 찾기
        url = 'https://www.gmarket.co.kr/n/best'
        
        # 2. request > response : str(html)
        response = requests.get(url)
        
        # 3. bs > DataFrame
        dom = BeautifulSoup(response.text, 'html.parser')
        # select 200 items
        elements = dom.select('#gBestWrap > div.best-list > ul > li')
        items = []
        for element in elements:
            data ={
            'title' : element.select_one('.itemname').text,
            'link' : element.select_one('.itemname').get('href'),
            'image' : 'https:' + element.select_one('img.image__lazy').get('src'),
            's_price' : element.select_one('.s-price').text.split(' ')[0][3:-1].replace(',', ''),
            }
            items.append(data)
        df = pd.DataFrame(items)
        df.to_excel('g_items.xlsx', index = False, encoding = 'utf-8-sig') # excel 파일로 저장 (encoding 필요)
        
        # 4. download image (200 items)
        os.makedirs('data')
        for idx, data in df.iterrows():
            print(idx, data['s_price'], data['image'])
            response = requests.get(data['image'])
            filename = f'data/{idx}.png'
            with open(filename, 'wb') as file:
                file.write(response.content)
    ```
    

## selenium

- [https://www.selenium.dev](https://www.selenium.dev)
- 자동화를 목적으로 만들어진 다양한 브라우져와 언어를 지원하는 라이브러리
- 크롬 브라우져 설치
    - 크롬 브라우져 드라이버 다운로드 (크롬 브라우져와 같은 버전)
        - [https://googlechromelabs.github.io/chrome-for-testing/](https://googlechromelabs.github.io/chrome-for-testing/)
    - 다운로드한 드라이버 압축 해제
    - chromedriver, chromedriver.exe 생성
    - windows : 주피터 노트북 파일과 동일한 디렉토리에 chromedriver.exe 파일 업로드
    - mac : `sudo cp ~/Download/chromedirver /usr/local/bin`
    
- 텍스트 데이터 가져오기
    - TED 사이트 : [https://www.ted.com](https://www.ted.com)
    
    ```python
    # 브라우져를 실행하여 테드 사이트 열기
    driver = webdriver.Chrome()
    driver.get("https://www.ted.com/talks")
    
    # CSS Selector를 이용하여 HTML 태그와 태그 사이의 text 데이터 가져오기
    driver.find_element(By.CSS_SELECTOR, ".text-textPrimary-onLight").text
    'TED Talks: Discover ideas worth spreading'
    
    # 제목 데이터 가져오기
    selector = '.container > section > .relative > div:nth-child(2) > div > div'
    contents = driver.find_elements(By.CSS_SELECTOR, selector)
    
    # 가장 처음 텍스트 데이터 가져오기
    contents[0].find_element(By.CSS_SELECTOR, '.text-textPrimary-onLight').text
    
    # 전체 제목 데이터 가져오기
    titles = []
    for content in contents:
        title = content.find_element(By.CSS_SELECTOR, '.text-textPrimary-onLight').text
        titles.append(title)
    
    # 링크 데이터 크롤링 (속성(attribute)값 가져오는 방법)
    links = []
    selector = '[data-testid="TalkGrid Talk Item"]'
    for content in contents:
        link = content.find_element(By.CSS_SELECTOR, selector).get_attribute("href")
        links.append(link)
    links[-3:]
    
    # 종료
    driver.quit()
    ```
    
- Headless
    - 브라우져를 화면에 띄우지 않고 메모리상에서만 올려서 크롤링하는 방법
    - **window가 지원되지 않는 환경에서 사용이 가능**
    - `chrome version 60.0.0.0` 이상부터 지원
    
    ```python
    # 현재 사용중인 크롬 버전 확인
    driver = webdriver.Chrome()
    version = driver.capabilities["browserVersion"]
    print(version) # 122.0.6261.112
    driver.quit()
    
    # headless 사용
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome(options=options)
    driver.get("https://www.ted.com/talks")
    text = driver.find_element(By.CSS_SELECTOR, ".text-textPrimary-onLight").text
    driver.quit()
    ```
    

## 크롤링 정책

- `robots.txt` : 크롤링 정책을 설명한 페이지
- 과도한 크롤링으로 서비스에 영향을 주었을 때 법적 문제가 있을 수 있다
- api 사용 > robots.txt > 서비스에 피해가 가지 않는 선에서 수집
- 서비스 피해
    - 지적재산권
    - 서비스 과부화
    - 데이터 사용표준