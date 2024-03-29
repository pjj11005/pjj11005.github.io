---
layout: post
title: B 트리(B Tree)
categories: 
  - cs
  - algorithm
description: B 트리(B Tree) 관련 개념 정리글 입니다.
sitemap: false
---

B 트리는 **`균형 이진 탐색 트리`**의 일종으로, 자료 구조에서 널리 사용되는 트리이다. 

* this unordered seed list will be replaced by the toc
{:toc}

## B 트리의 개념과 특징

### 정의
- **`보통 B 트리라고 하면 B- 트리`**를 의미한다. B 트리는 트리 자료구조의 일종으로 이진트리를 확장해 하나의 노드가 가질 수 있는 **`자식 노드의 최대 숫자가 2보다 큰`** 트리 구조이다. 

### 특징
1. 노드에는 2개 이상의 데이터(key)가 들어갈 수 있으며, 항상 정렬된 상태로 저장된다.

2. 모든 리프 노드는 같은 레벨에 있으며, 리프 노드를 제외한 모든 노드는 **`m/2 ~ m`**개의 자식을 가질 수 있다. (m은 B 트리의 차수)

3. 모든 리프 노드는 같은 레벨에 존재한다.

4. 특정 노드의 데이터(key)가 **`K`**개라면, 자식 노드의 개수는 **`K+1`**개여야 한다.

5. 특정 노드의 왼쪽 서브 트리는 특정 노드의 key 보다 작은 값들로, 오른쪽 서브 트리는 큰 값들로 구성된다.

6. 노드 내에 데이터는 **`floor(M/2)-1`**개부터 최대 **`M-1`**개까지 포함될 수 있다 ( floor : 내림 함수, floor(3.7) = 3 )

![B 트리의 구조](/assets/img/blog/Btree.png){: width="800" height="400"} 
    
B 트리의 구조
{:.figcaption}


## B 트리의 연산

### 삽입
- B 트리에 새로운 키를 삽입하는 경우, 해당 위치를 찾아 삽입한다.
- 만약 노드에 공간이 충분하다면, 새로운 키를 삽입하고 키들의 정렬을 유지한다.
- 만약 노드에 공간이 없다면, 노드를 분할하여 키들을 재분배하고 중간값을 상위 노드로 올린다.

![Case 1: 분리가 일어나지 않는 경우](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F8ywf1%2FbtrBiZqkG4n%2FohjxMtloMZvzcgHIjRVSik%2Fimg.webp){: width="800" height="400"}

![Case 1: 분리가 일어나지 않는 경우](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbO1PgF%2FbtrBgEaCeip%2Faq9R488vE8xz0MUL8K14L1%2Fimg.webp){: width="800" height="400"}

Case 1: 분리가 일어나지 않는 경우
{:.figcaption}

![Case 2 : 분리가 일어나는 경우](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F5OD0u%2FbtrBiHpVDoe%2FFlK7Dw0fahejkwokWLqNo1%2Fimg.webp){: width="800" height="400"}

![Case 2 : 분리가 일어나는 경우](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbnYL3M%2FbtrBhuE97RC%2FSDP7trISsSUnZ1jKDc2RK0%2Fimg.webp){: width="800" height="400"}

Case 2 : 분리가 일어나는 경우
{:.figcaption}

### 삭제
- 설명의 편의를 위한 용어 정의
    - **`Lmax`** = 현재 노드의 왼쪽 자식들 중 가장 큰 key
    - **`Rmin`** = 현재 노드의 오른쪽 자식들 중 가장 작은 key
    - **`Parent`** : 현재 노드를 가리키는 부모 노드의 자식 포인터 오른쪽에 있는 key (단, 마지막 자식 노드의 경우는 부모의 마지막 key)
    - **`K`** : 삭제할 key

- 3차 트리를 예시로 볼 때, **`1~3개의 자식을 가질 수 있고, 0~2개의 key 값을 가질 수 있다.`** 이 조건을 **`최소 유지 개수`**라고 정한다. 

#### Case 1. 리프 노드에서 삭제
- Case 1-1: 리프 노드에서 값을 삭제하더라도 최소 유지 개수 조건을 만족하는 경우
    - **`이 경우 바로 노드를 삭제해주면 된다.`**

![Case 1-1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcBvoFw%2FbtrBqviK4bS%2FpCx4UwKQxDEsHysiFrfomk%2Fimg.webp){: width="800" height="400"}

- Case 1-2: 리프 노드에서 값을 삭제할 때, 최소 유지 개수를 만족하지 못하지만 바로 옆 형제 노드들에게 값을 빌려올 수 있는 경우
    - 이 경우 **`K`**를 **`Parent`**와 바꿔준다. 이후 왼쪽 형제 노드에게서 값을 빌려올 수 있다면 **`Lmax`**와 **`Parent`**를, 오른쪽 형제에게서 값을 빌려올 수 있다면 **`Rmin`**과 **`Parent`**를 바꿔주면 된다. 둘 다 가능하면 하나를 선택하면 된다.

![Case 1-2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc99xpS%2FbtrBo22JiR9%2F8VhjhycktJekG9ixkbAFqK%2Fimg.webp){: width="800" height="400"}

![Case 1-2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fv8pqL%2FbtrBqfUSHHl%2Fi47FsOcEzm2x1lK4Za3ySk%2Fimg.webp){: width="800" height="400"}

- Case 1-3: 리프 노드에서 값을 삭제할 때, 최소 유지 개수를 만족하지 못하고 형제 노드들에게 값을 빌려올 수 없지만, 부모 노드를 분할할 수 있을 때
    - **`K`**를 삭제하고, **`Parent`**를 부모 노드에서 분할하여 형제 노드에 합친다. 이렇게 하면 부모 노드의 **`key`**가 하나 줄고, 자식 노드의 수도 하나 줄어들어 **`최소 유지 개수`**를 만족한다.

![Case 1-3](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc1c4g1%2FbtrBnLHckLL%2FkFGlr1xvb5So3kF3vlDy91%2Fimg.webp){: width="800" height="400"}

![Case 1-3](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FuQCC5%2FbtrBmvSoNAh%2Fj1cv3kdadXAK20wf6jjBH1%2Fimg.webp){: width="800" height="400"}

- Case 1-4: 리프 노드에서 값을 삭제할 때, 최소 유지 개수를 만족하지 못하고, 형제 노드들에게 값을 빌려올 수 없고 부모 노드도 분할할 수 없을 때
    - 이 경우는 Case 2-2의 경우와 동일하므로 뒤의 설명 참고 

#### Case 2. 리프 노드가 아닌 내부 노드에서 삭제
- Case 2-1: 내부 노드에서 값을 삭제할 때, 현재 노드 혹은 자식 노드의 최소 유지 개수의 최소보다 큰 경우
    - 이 경우 **`K`**의 **`Lmax`** 혹은 **`Rmin`**과 자리를 바꿔준다. 이후 리프 노드에서의 **`K`** 삭제와 과정이 동일하다.

![Case 2-1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdKDhX4%2FbtrBq630pPK%2Ff7Gqu11LNK3HYQOmbxifh0%2Fimg.webp){: width="800" height="400"}

![Case 2-1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FNBTH7%2FbtrBnLUQ8DJ%2FKUoMLK564D94SWDeRlTT60%2Fimg.webp){: width="800" height="400"}

- Case 2-2: 내부 노드에서 값을 삭제할 때, 현재 노드와 자식 노드 모두 key 개수가 최소인 경우
    - 이 경우 다음과 같은 트리의 재구조화가 필요하다.
        1. K를 삭제하고 K의 자식을 하나로 합친다. 합쳐진 노드를 N1이라고 하자.

        2. K의 Parent를 K의 형제 노드에 합친다. 합쳐진 노드를 N2라고 하자.
        
        3. N1을 N2의 자식이 되도록 연결한다.

        4-1. 만약 N2의 key수가 최대보다 크다면 key 삽입 과정과 동일하게 분할한다.

        4-2. 만약 N2의 key수가 최소보다 작다면 2번 과정으로 돌아가서 동일한 과정을 반복한다. 

![Case 2-2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcgIlVi%2FbtrBpV3wajC%2F8ck3r9sDkMbs515Ux3kt10%2Fimg.webp){: width="800" height="400"}

![Case 2-2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbciKkW%2FbtrBpVh9F2a%2FckAzTAkAbWcgbXxXyDvovK%2Fimg.webp){: width="800" height="400"}

![Case 2-2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F419pT%2FbtrBm8Qm7lS%2FFF9K7ZYn1d42A1q3JF4AO1%2Fimg.webp){: width="800" height="400"}

이 후 key의 개수가 너무 많으므로 삽입 과정에서와 같이 분할한다.
{:.figcaption}

### 검색
- B 트리에서 특정 키를 검색하는 경우, 루트 노드에서 시작하여 키 값을 비교하며 자식 노드로 이동한다.
- 키 값을 비교하여 작은 값이면 왼쪽 자식으로, 큰 값이면 오른쪽 자식으로 이동한다.
- 검색 대상 키를 찾을 때까지 이동하고, 키를 찾지 못하면 검색 실패를 반환한다.

![B 트리의 검색](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fm2qHc%2FbtrBgo6Ux7Y%2FSIcDWcUE45PyOwC5tL4ekK%2Fimg.webp){: width="800" height="400"} 

![B 트리의 검색](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fdng4RK%2FbtrBgWIOYFc%2F6d6EleKTL8wqS4UQBoZMj0%2Fimg.webp){: width="800" height="400"} 

B 트리의 검색
{:.figcaption}


## B 트리의 시간 복잡도 분석
   - B 트리의 삽입, 삭제, 검색 연산의 시간 복잡도는 $$O(log n)$$ -> 이는 **`B 트리의 높이에 비례`**하며, 각 노드에서 키 값을 비교하고 자식 노드로 이동하는 과정을 반복하기 때문
   - B 트리의 높이는 log n 이하로 유지되므로, 이러한 시간 복잡도는 매우 효율적인 검색과 수정 연산을 제공한다.
   - 또한, B 트리는 균형을 유지하며 삽입과 삭제 시에 노드 분할과 합병 연산을 수행하여 균형을 조정하기 때문에, 트리의 불균형 현상을 최소화할 수 있다.

## B 트리의 장점과 활용

>B 트리는 자료 구조에서 다양한 장점과 활용 가능성을 가지고 있다.
{: .lead}

1. 빠른 검색 성능: B 트리는 높이가 낮고, 모든 노드가 최소한 m/2개의 자식을 가지므로 빠른 검색 속도 보장

2. 효율적인 삽입과 삭제: B 트리는 균형을 유지하며 노드 분할과 합병 연산을 통해 삽입과 삭제 시에도 최적의 성능을 유지한다.

3. 범위 탐색: B 트리의 구조는 키 값들이 정렬된 상태를 유지하므로, 특정 범위에 속하는 키들을 탐색하는 데에도 효과적이다.

## B 트리의 확장과 변형

### 1. B+ 트리
>B 트리는 탐색을 위해서 노드를 찾아서 이동해야 한다는 단점을 가지고 있다.
{: .lead}
- **`B+Tree`**는 **`같은 레벨의 모든 키값들이 정렬`**되어 있고, **`같은 레벨의 형제 노드들은 연결리스트 형태`**로 이어져 있다. (같은 레벨의 형제 노드들은 모두 연결되어 있어서 키값이 중복되지 않는다)

- 만약 특정 값을 찾아야 하는 상황이 된다면 리프 노드에 모든 자료들이 존재하고, 그 자료들이 연결리스트로 연결되어 있으므로 탐색에 있어서 매우매우 유리하다.

- **`리프 노드가 아닌 자료`**는 **`인덱스 노드`**, **`리프 노드`**는 **`데이터 노드`**라고 부른다.

- 인덱스 노드는 다음 노드를 가리킬 수 있는 포인터 주소가 존재하고, 데이터 노드의 Value값에 데이터가 존재한다.

- 키값은 중복될 수 있고 **`인덱스 노드와 데이터 노드에서 동시에 등장 가능하다.`** 따라서, 데이터 검색을 위해서는 반드시 리프 노드까지 내려가야 한다는 특징을 가지고 있다.

- 오늘날 데이터베이스에서 가장 중요한 것은 검색속도이기 때문에 대부분의 데이터베이스 시스템은 **`B+Tree구조`**를 채택하고 있다.

![B+ 트리](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FlmNml%2FbtrhSZR4BuI%2FKBL65KkAp6rNDJqk85MmZK%2Fimg.png){: width="800" height="400"} 

B+ 트리
{:.figcaption}

### 2. B* 트리

>B 트리는 구조를 유지하기 위해서 추가적인 연산이 수행되거나 새로운 노드가 생성된다는 단점이 있다.
{: .lead}

- 노드의 약 2/3이상이 채워지는 B트리
- 노드가 꽉 차면 분리하지 않고, 키와 포인터를 재배치하여 다른 형제 노드로 옮김
- 삽입/삭제 시 발생하는 노드 분리를 줄이려고 고안됨

> 특징 (차수가 m인 B* 트리)
1. 공집합 or 높이가 1 이상인 m원 탐색 트리
2. 루트 노드는 **`2`**이상 **`2(2m-2)/3 + 1`**개 이하의 자식노드를 가짐
3. 내부노드는 최소 **`(2m-1)/3`**개의 자식노드를 가짐
4. 모든 리프 노드 동일한 레벨
5. 포인터가 **`k`**개인 **`잎이 아닌 노드`**는 **`k-1`**개의 키값 가짐 **`(루트 노드 포함)`**

## 참고 문헌 및 사이트

- [https://code-lab1.tistory.com/217](https://code-lab1.tistory.com/217)

- [https://ssocoit.tistory.com/217](https://ssocoit.tistory.com/217)

- [https://3catpapa.tistory.com/201](https://3catpapa.tistory.com/201)

- chat gpt