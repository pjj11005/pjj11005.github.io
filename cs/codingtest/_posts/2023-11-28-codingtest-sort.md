---
layout: post
title: 4. 정렬 (Sort)
categories: 
  - cs
  - codingtest
description: 이것이 취업을 위한 코딩테스트다 with 파이썬 책을 참고하여 **정렬**을 공부한 내용입니다.
sitemap: false
---

프로그램 작성 시 가장 많이 사용

* this unordered seed list will be replaced by the toc
{:toc}


## 정렬 알고리즘 개요

- 정렬: 데이터를 특정한 기준에 따라 순서대로 나열하는 것
- 프로그램 작성 시 가장 많이 사용되는 알고리즘 중 하나
- 이진 탐색의 전처리 과정
- 선택 정렬, 삽입 정렬, 퀵 정렬, 계수 정렬
- 면접 단골 문제

## 선택 정렬

- 가장 원시적인 방법 → 매번 가장 작은 것을 선택하여 앞으로 보내서 정렬
    
    ```python
    array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]
    
    for i in range(len(array)):
        min_index = i # 가장 작은 원소의 인덱스
        for j in range(i + 1, len(array)):
            if array[min_index] > array[j]:
                min_index = j
        array[i], array[min_index] = array[min_index], array[i] # 스와프
    
    print(array)
    ```
    
- 시간 복잡도: $$O(N^2)$$
- 정렬 알고리즘 수행시간 비교

| 데이터의 개수(N) | 선택 정렬 | 퀵 정렬 | 기본 정렬 라이브러리 |
| --- | --- | --- | --- |
| N=100 | 0,0123초 | 0,00156초 | 0.00000753초 |
| N=1000 | 0.354초 | 0.00343초 | 0.0000365초 |
| N=10000 | 15.475초 | 0.0312초 | 0.000248초 |

- 선택 정렬은 다른 알고리즘에 비해 비효율적이지만, 특정 리스트에서 가장 작은 원소 찾는 일이 잦으므로 익숙해질 필요 있다.

## 삽입 정렬

- 특정한 데이터를 적절한 위치에 삽입하는 정렬
- 선택 정렬에 비해 구현 난이도가 높지만, 실행 시간이 더 효율적
- 데이터가 거의 정렬되어 있을 때 효율적
- 정렬되어 있는 리스트에서 위치를 찾은 후, 그 위치에 삽입
    
    ```python
    array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]
    
    for i in range(1, len(array)):
        for j in range(i, 0, -1): # 인덱스 i부터 1까지 1씩 감소하며 반복하는 문법
            if array[j] < array[j - 1]: # 한 칸씩 왼쪽으로 이동
                array[j], array[j - 1] = array[j - 1], array[j]
            else: # 자기보다 작은 데이터를 만나면 그 위치에서 멈춤
                break
    
    print(array)
    ```
    
- 정렬이 이루어진 원소는 항상 오름차순으로 유지
- 시간 복잡도: $$O(N^2)$$, 최선의 경우 $$O(N)$$
- 정렬이 거의 되어있을 때 → 삽입 정렬 유리

## 퀵 정렬

- 가장 많이 사용되는 정렬 알고리즘
- 기준 데이터(피벗)를 설정한 다음 큰 수와 작은 수를 교환한 후 리스트를 반으로 나누는 방식으로 동작
- 분할된 후, 각각의 리스트 별로 동일한 방식으로 정렬 진행
- 재귀 함수로 구현 시 매우 간결 → 종료 조건: 현재 리스트의 길이가 1(더 이상 분할 불가능)
    
    ```python
    array = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]
    
    def quick_sort(array, start, end):
        if start >= end: # 원소가 1개인 경우 종료
            return
        pivot = start # 피벗은 첫 번째 원소
        left = start + 1
        right = end
        while(left <= right):
            # 피벗보다 큰 데이터를 찾을 때까지 반복 
            while(left <= end and array[left] <= array[pivot]):
                left += 1
            # 피벗보다 작은 데이터를 찾을 때까지 반복
            while(right > start and array[right] >= array[pivot]):
                right -= 1
            if(left > right): # 엇갈렸다면 작은 데이터와 피벗을 교체
                array[right], array[pivot] = array[pivot], array[right]
            else: # 엇갈리지 않았다면 작은 데이터와 큰 데이터를 교체
                array[left], array[right] = array[right], array[left]
        # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행
        quick_sort(array, start, right - 1)
        quick_sort(array, right + 1, end)
    
    quick_sort(array, 0, len(array) - 1)
    print(array)
    ```
    
    가장 널리 사용되는 퀵 정렬 소스 코드
    {:.figcaption}
    
    ```python
    array = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]
    
    def quick_sort(array):
        # 리스트가 하나 이하의 원소만을 담고 있다면 종료
        if len(array) <= 1:
            return array
    
        pivot = array[0] # 피벗은 첫 번째 원소
        tail = array[1:] # 피벗을 제외한 리스트
    
        left_side = [x for x in tail if x <= pivot] # 분할된 왼쪽 부분
        right_side = [x for x in tail if x > pivot] # 분할된 오른쪽 부분
    
        # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬을 수행하고, 전체 리스트를 반환
        return quick_sort(left_side) + [pivot] + quick_sort(right_side)
    
    print(quick_sort(array))
    ```
    
    파이썬의 장점을 살린 퀵 정렬 소스 코드
    {:.figcaption}

    - 시간 면에서 비효율적 이지만 직관적이다.
    
- 시간 복잡도: 평균 → $$O(NlogN)$$, 최악 → $$O(N^2)$$
- 데이터가 이미 정렬되어 있는 경우 최악

## 계수 정렬

- 특정한 조건이 부합할 때 만 사용 가능 하지만, 매우 빠른 정렬 알고리즘
- 데이터의 크기 범위가 정수 형태이고, 가장 큰 수와 가장 작은 수의 차이가 1,000,000을 넘지 않을 때 사용 가능(모든 원소 0 이상)
- 동작 방식
    1. 데이터의 범위를 담을 수 있는 리스트 생성
    2. 데이터를 확인하며 데이터 값과 동일한 인덱스의 값을 1증가
    3. 그 후, 생성된 리스트의 인덱스 별 개수만큼 인덱스를 출력
    
    ```python
    # 모든 원소의 값이 0보다 크거나 같다고 가정
    array = [7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2]
    # 모든 범위를 포함하는 리스트 선언 (모든 값은 0으로 초기화)
    count = [0] * (max(array) + 1)
    
    for i in range(len(array)):
        count[array[i]] += 1 # 각 데이터에 해당하는 인덱스의 값 증가
    
    for i in range(len(count)): # 리스트에 기록된 정렬 정보 확인
        for j in range(count[i]):
            print(i, end=' ') # 띄어쓰기를 구분으로 등장한 횟수만큼 인덱스 출력
    ```
    
- 시간 복잡도: 데이터의 개수 $$N$$, 최대 값의 크기 $$K$$ → $$O(N+K)$$
- 공간 복잡도
    - 때에 따라 심각한 비효율성 초래
    - 동일한 데이터가 여러 개 등장할 때 적합 ex) 시험 성적 정렬
    - 일반적인 경우 → 퀵 정렬
    - 일반적인 코딩 테스트 환경 → 공간 복잡도 $$O(N+K)$$

## 파이썬의 정렬 라이브러리

- 정렬 알고리즘 문제: 정해진 답이 있는, 외워서 풀 수 있는 문제
- sorted(),sort()
    - 병합 정렬 기반
    - 퀵 정렬보단 느리지만 최악의 경우에도 $$O(NlogN)$$보장
    - key값을 통해 특정 원소 기준으로 정렬도 가능
        
        ```python
        array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]
        
        result = sorted(array)
        print(result)
        ```
        
        ```python
        array = [('바나나', 2), ('사과', 5), ('당근', 3)]
        
        def setting(data):
            return data[1]
        
        result = sorted(array, key=setting)
        print(result)
        ```
        
- 시간 복잡도
    - 최악의 경우에도 $$O(NlogN)$$보장
    - 직접 퀵 정렬 구현하는 것 보다 효과적
    - 정확히는 병합 + 삽입 정렬 의 하이브리드 방식
    - 문제 유형
        - 정렬 라이브러리로 푸는 문제: 단순 정렬 기법 문제
        - 정렬 알고리즘 원리 문제: 선택, 삽입, 퀵 정렬 등의 원리 알아야 해결 가능
        - 더 빠른 정렬 문제: 계수 정렬 혹은 기존 알고리즘의 구조적 개선으로 해결
        

## 예제 문제

- 예제 1: 위에서 아래로

    1. 내 풀이
        1. 선택 정렬을 이용해서 내림차순으로 정렬
        
        ```python
        n=int(input())
        array=[]
        for i in range(n):
          array.append(int(input()))
        
        for i in range(n):
          maximum=i
          for j in range(i+1,n):
            if array[j]>array[maximum]:
              maximum=j
          array[i],array[maximum]=array[maximum],array[i]
          print(array[i],end=" ")
        ```
        
    2. 풀이를 본 후
        
        데이터의 수가 많기 때문에 기본 정렬 알고리즘을 사용해도 된다.
        
        ```python
        # N 입력 받기
        n = int(input())
        
        # N개의 정수를 입력 받아 리스트에 저장
        array = []
        for i in range(n):
            array.append(int(input()))
        
        # 파이썬 정렬 라이브러리를 이용하여 내림차순 정렬 수행
        array = sorted(array, reverse=True)
        
        # 정렬이 수행된 결과를 출력
        for i in array:
            print(i, end=' ')
        ```
        
    3. 해결한 후
        
        기본적인 정렬 문제이다.
        
- 예제 2: 성적이 낮은 순서로 학생 출력하기

    1. 내 풀이
        1. sorted의 key를 이용하여 두번째 요소를 기준으로 오름차순 정렬
        2. 정렬 후, 첫번째 요소를 순서대로 출력
        
        ```python
        n=int(input())
        array=[]
        for i in range(n):
          array.append(list(input().split()))
          
        array=sorted(array,key=lambda x:x[1])
        
        for a in array:
          print(a[0],end=' ')
        ```
        
    2. 풀이를 본 후
        
        ```python
        # N 입력 받기
        n = int(input())
        
        # N명의 학생 정보를 입력 받아 리스트에 저장
        array = []
        for i in range(n):
            input_data = input().split()
            # 이름은 문자열 그대로, 점수는 정수형으로 변환하여 저장
            array.append((input_data[0], int(input_data[1])))
        
        # 키(Key)를 이용하여, 점수를 기준으로 정렬
        array = sorted(array, key=lambda student: student[1])
        
        # 정렬이 수행된 결과를 출력
        for student in array:
            print(student[0], end=' ')
        ```
        
    3. 해결한 후
        
        정확히는 점수 값을 정수형으로 변환을 해줘야 한다.
        
- 예제 3: 두 배열의 원소 교체

    1. 내 풀이
        1. K번의 반복 시행 시 매 시행마다 A의 최소, B의 최소 계산
        2. 그 후, 바꿔주고 K번 이 과정 반복
        3. A배열의 합 출력
        
        ```python
        n,k=map(int,input().split())
        A=list(map(int,input().split()))
        B=list(map(int,input().split()))
        
        for i in range(k):
          min_A,max_B=min(A),max(B)
          A.remove(min_A)
          B.remove(max_B)
          A.append(max_B)
          B.append(min_A)
        
        print(sum(A))
        ```
        
    2. 풀이를 본 후
        
        A를 오름차순, B를 내림차순으로 정렬 후, 앞에서부터 비교하여 바꿔주면 되었다…
        
        ```python
        n, k = map(int, input().split()) # N과 K를 입력 받기
        a = list(map(int, input().split())) # 배열 A의 모든 원소를 입력받기
        b = list(map(int, input().split())) # 배열 B의 모든 원소를 입력받기
        
        a.sort() # 배열 A는 오름차순 정렬 수행
        b.sort(reverse=True) # 배열 B는 내림차순 정렬 수행
        
        # 첫 번째 인덱스부터 확인하며, 두 배열의 원소를 최대 K번 비교
        for i in range(k):
            # A의 원소가 B의 원소보다 작은 경우
            if a[i] < b[i]:
                # 두 원소를 교체
                a[i], b[i] = b[i], a[i]
            else: # A의 원소가 B의 원소보다 크거나 같을 때, 반복문을 탈출
                break
        
        print(sum(a)) # 배열 A의 모든 원소의 합을 출력
        ```
        
    3. 해결한 후
        
        인덱스를 이용해서 변경하는 것이 더 좋다.
        

## 기출 문제

- 기출 문제 1: 국영수

    1. 내 풀이
        1. sorted의 key를 기준으로 정렬
        2. 우선순위 기준 별로 쉼표를 이용하여 나열
        
        시간이 엄청 오래 걸렸다…
        
        python: 4304ms, pypy3: 528ms
        
        ```python
        # 정렬 기준 함수
        def sorting_criteria(student):
            return (-student[1], student[2], -student[3], student[0])
        
        # 학생 수 입력
        N = int(input())
        
        # 학생 정보 입력
        students = []
        for _ in range(N):
            name, kor, eng, math = input().split()
            students.append((name, int(kor), int(eng), int(math)))
        
        # 정렬 후 출력
        sorted_students = sorted(students, key=sorting_criteria)
        
        for student in sorted_students:
            print(student[0])
        ```
        
    2. 풀이를 본 후
        
        나의 풀이와 같았다…
        
        ```python
        n = int(input())
        students = [] # 학생 정보를 담을 리스트
        
        # 모든 학생 정보를 입력 받기
        for _ in range(n):
            students.append(input().split())
        
        '''
        [ 정렬 기준 ]
        1) 두 번째 원소를 기준으로 내림차순 정렬
        2) 두 번째 원소가 같은 경우, 세 번째 원소를 기준으로 오름차순 정렬
        3) 세 번째 원소가 같은 경우, 네 번째 원소를 기준으로 내림차순 정렬
        4) 네 번째 원소가 같은 경우, 첫 번째 원소를 기준으로 오름차순 정렬
        '''
        students.sort(key=lambda x: (-int(x[1]), int(x[2]), -int(x[3]), x[0]))
        
        # 정렬된 학생 정보에서 이름만 출력
        for student in students:
            print(student[0])
        ```
        
    3. 해결한 후
        
        sort(), sorted()에서 key 속성에 값을 대입하여 정렬 가능했고, 우선순위와 -유무로 튜플을 정렬할 수 있다는 사실을 알았다.


- 기출 문제 2: 안테나

    1. 내 풀이
        1. 집의 위치 오름차순 정렬
        2. 중간 인덱스와 그 다음 인덱스 위치의 값을 선정
        3. 각각의 위치에서의 거리 측정
        4. 비교하여 최소의 거리 위치 선정
        
        ```python
        N = int(input())
        
        house = list(map(int, input().split()))
        house.sort()
        
        index1=(N//2)-1
        index2=index1+1
        num1,num2=house[index1],house[index2]
        sum1,sum2=0,0
        
        for h in house:
          sum1+=abs(h-num1)
          sum2+=abs(h-num2)
        
        if sum1<sum2:
          print(num1)
        elif sum1==sum2:
          print(min(num1,num2))
        else:
          print(num2)
        ```
        
    2. 풀이를 본 후
        
        단순히 중간 인덱스의 위치 값을 출력하면 해결되었다…
        
        ```python
        n = int(input())
        a = list(map(int, input().split()))
        a.sort()
        
        # 중간값(median)을 출력
        print(a[(n - 1) // 2])
        ```
        
    3. 해결한 후
        
        조금 더 세심하게 생각해보는 것도 좋을 것 같다.

- 기출 문제 3: 실패율

    1. 내 풀이
        1. 계수 정렬을 이용하여 N까지의 숫자는 따로 배열에 개수를 저장
        2. N+1이상의 수의 개수는 overcount 변수에 합쳐서 저장
        3. fail 배열에 stage수와 실패율 저장
        4. 내림차순 및 오름차순 정렬 후 stage 값 answer에 저장
        
        메모리 용량이 작아 정답까지 꽤 걸렸다…
        
        ```python
        def solution(N, stages):
            answer = []
            fail=[]
            array=[0]*501
            overcount=0
            for stage in stages:
                if stage>=N+1:
                    overcount+=1
                else:
                    array[stage]+=1
            
            s=sum(array)+overcount
            for i in range(1,N+1):
                if array[i]==0:
                    failure=0
                else:
                    failure=array[i]/s
                fail.append((i,failure))
                x=array[i]
                s-=x
            fail.sort(key=lambda x: (-x[1],x[0]))
            
            for f in fail:
                answer.append(f[0])
            return answer
        ```
        
    2. 풀이를 본 후
        
        방법은 거의 비슷했다. 다만, count 함수 사용, 조건 분기의 차이, 배열을 하나 더 만들지 않은 점이 차이였다.
        
        ```python
        def solution(N, stages):
            answer = []
            length = len(stages)
        
            # 스테이지 번호를 1부터 N까지 증가시키며
            for i in range(1, N + 1):
                # 해당 스테이지에 머물러 있는 사람의 수 계산
                count = stages.count(i)
                
                # 실패율 계산
                if length == 0:
                    fail = 0
                else:
                    fail = count / length
                
                # 리스트에 (스테이지 번호, 실패율) 원소 삽입
                answer.append((i, fail))
                length -= count
        
            # 실패율을 기준으로 각 스테이지를 내림차순 정렬
            answer = sorted(answer, key=lambda t: t[1], reverse=True)
            
            # 정렬된 스테이지 번호 반환
            answer = [i[0] for i in answer]
            return answer
        ```
        
    3. 해결한 후
        
        count함수와 배열에서 한 가지의 원소만 빼낼 때의 방법도 알게 되었다.

- 기출 문제 4: 카드 정렬하기

    1. 내 풀이
        1. 입력된 카드 수들을 오름차순으로 정렬
        2. 그리고 작은 수들을 최대한 많이 더하도록 하여 합을 구함
        
        하지만, (50,60,70,80)인 경우에는 다르게 나온다. 
        
        ```python
        N = int(input())
        num=[]
        for i in range(N):
          num.append(int(input()))
        num.sort()
        
        result=num[0]*(N-1)
        for i in range(1,N):
          result+=num[i]*(N-i)
        
        print(result)
        ```
        
    2. 풀이를 본 후
        
        카드를 두개 씩 합치면서 그 때 마다 가장 작은 두 카드를 합쳐야 한다. 따라서, 우선 순위 큐를 이용하여 해결했다.
        
        ```python
        import heapq
        
        n = int(input())
        
        # 힙(Heap)에 초기 카드 묶음을 모두 삽입
        heap = []
        for i in range(n):
            data = int(input())
            heapq.heappush(heap, data)
        
        result = 0
        
        # 힙(Heap)에 원소가 1개 남을 때까지
        while len(heap) != 1:
            # 가장 작은 2개의 카드 묶음 꺼내기
            one = heapq.heappop(heap)
            two = heapq.heappop(heap)
            # 카드 묶음을 합쳐서 다시 삽입
            sum_value = one + two
            result += sum_value
            heapq.heappush(heap, sum_value)
        
        print(result)
        ```
        
    3. 해결한 후
        
        문제의 상황을 자세히 보고 완벽히 이해해야 한다. 우선순위 큐를 이용하는 방법을 숙지하자.