---
layout: post
title: 이것이 취업을 위한 코딩테스트다 | 5. 이진 탐색 (Binary Search)
categories: 
  - cs
  - codingtest
description: 이것이 취업을 위한 코딩테스트다 with 파이썬 책을 참고하여 **이진 탐색**을 공부한 내용입니다.
sitemap: false
---

탐색 범위를 반으로 좁혀 빠르게 탐색하는 알고리즘

* this unordered seed list will be replaced by the toc
{:toc}

## 순차 탐색

- 리스트 안에서 특정한 데이터를 찾기 위해 앞에서부터 데이터를 하나씩 차례대로 확인하는 방법
- 정렬되지 않은 리스트에서 데이터를 찾을 때 사용
- 장점: 리스트 내에 데이터가 아무리 많아도 시간만 많으면 데이터를 찾을 수 있다.
- count 메서드: 리스트에서 특정한 값을 갖는 데이터의 수를 센다 → 순차 탐색으로 수행
- 시간 복잡도: 최악의 경우 $$O(N)$$
    
    ```python
    # 순차 탐색 소스코드 구현
    def sequential_search(n, target, array):
        # 각 원소를 하나씩 확인하며
        for i in range(n):
            # 현재의 원소가 찾고자 하는 원소와 동일한 경우
            if array[i] == target:
                return i + 1 # 현재의 위치 반환 (인덱스는 0부터 시작하므로 1 더하기)
        return -1 # 원소를 찾지 못한 경우 -1 반환
    
    print("생성할 원소 개수를 입력한 다음 한 칸 띄고 찾을 문자열을 입력하세요.")
    input_data = input().split()
    n = int(input_data[0]) # 원소의 개수
    target = input_data[1] # 찾고자 하는 문자열
    
    print("앞서 적은 원소 개수만큼 문자열을 입력하세요. 구분은 띄어쓰기 한 칸으로 합니다.")  
    array = input().split()
    
    # 순차 탐색 수행 결과 출력
    print(sequential_search(n, target, array))
    ```
    

## 이진 탐색: 반으로 쪼개면서 탐색하기

- 데이터가 정렬되어 있어야만 사용 가능
- 특징
    - 매우 빠르게 데이터를 찾을 수 있다
    - 탐색 범위를 절반씩 좁혀가며 탐색
- 찾으려는 데이터와 중간점 위치에 있는 데이터를 반복적으로 비교 (변수 3개 사용: 시작점, 끝점, 중간점)
- 시간 복잡도: $$O(logN)$$, 절반 씩 데이터를 줄이는 점이 퀵정렬과 공통
    
    ```python
    # 이진 탐색 소스코드 구현 (재귀 함수)
    def binary_search(array, target, start, end):
        if start > end:
            return None
        mid = (start + end) // 2
        # 찾은 경우 중간점 인덱스 반환
        if array[mid] == target:
            return mid
        # 중간점의 값보다 찾고자 하는 값이 작은 경우 왼쪽 확인
        elif array[mid] > target:
            return binary_search(array, target, start, mid - 1)
        # 중간점의 값보다 찾고자 하는 값이 큰 경우 오른쪽 확인
        else:
            return binary_search(array, target, mid + 1, end)
    
    # n(원소의 개수)과 target(찾고자 하는 값)을 입력 받기
    n, target = list(map(int, input().split()))
    # 전체 원소 입력 받기
    array = list(map(int, input().split()))
    
    # 이진 탐색 수행 결과 출력
    result = binary_search(array, target, 0, n - 1)
    if result == None:
        print("원소가 존재하지 않습니다.")
    else:
        print(result + 1)
    ```
    
    ```python
    # 이진 탐색 소스코드 구현 (반복문)
    def binary_search(array, target, start, end):
        while start <= end:
            mid = (start + end) // 2
            # 찾은 경우 중간점 인덱스 반환
            if array[mid] == target:
                return mid
            # 중간점의 값보다 찾고자 하는 값이 작은 경우 왼쪽 확인
            elif array[mid] > target:
                end = mid - 1
            # 중간점의 값보다 찾고자 하는 값이 큰 경우 오른쪽 확인
            else:
                start = mid + 1
        return None
    
    # n(원소의 개수)과 target(찾고자 하는 값)을 입력 받기
    n, target = list(map(int, input().split()))
    # 전체 원소 입력 받기
    array = list(map(int, input().split()))
    
    # 이진 탐색 수행 결과 출력
    result = binary_search(array, target, 0, n - 1)
    if result == None:
        print("원소가 존재하지 않습니다.")
    else:
        print(result + 1)
    ```
    

### 코딩 테스트에서의 이진 탐색

- 참고 소스 코드가 없는 상태에서 소스 코드 구현은 어려운 작업이 될 수 있음
- 코딩 테스트에 단골로 나오는 문제이니 구현 방법은 암기
- 다른 알고리즘에서 사용되는 원리와 유사해서 매우 중요
- 높은 난이도의 문제에서 이진 탐색 알고리즘이 다른 알고리즘과 같이 사용됨
    - ex) 그리디 알고리즘 + 이진 탐색 알고리즘
- 탐색 범위가 2000만을 넘어가면 이진 탐색으로 접근
- 처리해야 할 데이터의 수가 1000만 단위 이상으로 넘어가면 이진 탐색과 같은 $$O(logN)$$의 속도를 갖는 알고리즘 사용하기

## 트리 자료구조

- 노드
    - 정보의 단위
    - 어떠한 정보를 가진 개체
- 그래프 자료구조의 일종으로 데이터베이스 시스템 or 파일 시스템과 같은 곳에서 많은 양의 데이터를 관리하기 위한 목적으로 사용
- 특징
    - 부모와 자식 노드의 관계로 표현
    - 루트 노드: 최상단 노드, 자식 노드: 최하단 노드
    - 서브 트리: 트리에서 일부를 떼어낸 트리 구조
    - 계층적이고 정렬된 데이터를 다루기에 적합

## 이진 탐색 트리

- 이진 탐색이 동작할 수 있도록 고안된, 효율적인 탐색이 가능한 자료구조
- 특징
    - 왼쪽 자식 노드 < 부모 노드 < 오른쪽 자식 노드
- 방문한 노드와 크기 비교를 하여 타겟이 작으면 왼쪽, 크면 오른쪽 자식 노드의 서브 트리로 방문

### 빠르게 입력 받기

- 입력 데이터가 1000만개 이상, 탐색 범위가 1000억 이상 → sys 라이브러리의 readline() 함수 사용
    
    ```python
    import sys
    
    # 하나의 문자열 데이터 입력 받기
    input_data = sys.stdin.readline().rstrip()
    # 입력 받은 문자열 그대로 출력하기
    print(input_data)
    ```
    

## 예제 문제

- 예제 1: 부품 찾기

    1. 내 풀이
        1. 부품 배열을 오름차순으로 정렬
        2. 이진 탐색을 수행하여 찾으려는 부품의 번호가 있으면 yes, 아니면 no 출력
        
        ```python
        N = int(input())
        array=list(map(int,input().split()))
        M=int(input())
        num=list(map(int,input().split()))
        array.sort()
        
        for n in num:
          start,end=0,N-1
          result='no'
          while start<=end:
            mid=(start+end)//2
            if array[mid]==n:
              result='yes'
              break
            elif array[mid]<n:
              start=mid+1
            else:
              end=mid-1
          print(result,end=' ')
        ```
        
    2. 풀이를 본 후
        
        여러가지 방법으로 풀 수 있었다. 이진 탐색의 풀이는 거의 유사했다.
        
        ```python
        # 계수 정렬 이용
        
        # N(가게의 부품 개수) 입력
        n = int(input())
        array = [0] * 1000001
        
        # 가게에 있는 전체 부품 번호를 입력 받아서 기록
        for i in input().split():
            array[int(i)] = 1
        
        # M(손님이 확인 요청한 부품 개수) 입력
        m = int(input())
        # 손님이 확인 요청한 전체 부품 번호를 공백을 기준으로 구분하여 입력
        x = list(map(int, input().split()))
        
        # 손님이 확인 요청한 부품 번호를 하나씩 확인
        for i in x:
            # 해당 부품이 존재하는지 확인
            if array[i] == 1:
                print('yes', end=' ')
            else:
                print('no', end=' ')
        ```
        
        ```python
        # 집합 이용
        
        # N(가게의 부품 개수) 입력
        n = int(input())
        # 가게에 있는 전체 부품 번호를 입력 받아서 집합(Set) 자료형에 기록
        array = set(map(int, input().split()))
        
        # M(손님이 확인 요청한 부품 개수) 입력
        m = int(input())
        # 손님이 확인 요청한 전체 부품 번호를 공백을 기준으로 구분하여 입력
        x = list(map(int, input().split()))
        
        # 손님이 확인 요청한 부품 번호를 하나씩 확인
        for i in x:
            # 해당 부품이 존재하는지 확인
            if i in array:
                print('yes', end=' ')
            else:
                print('no', end=' ')
        ```
        
    3. 해결한 후
        
        여러가지 방법으로 해결할 수 있다는 점을 숙지
        
- 예제 2: 떡볶이 떡 만들기

    1. 내 풀이
        1. 떡의 최소 길이, 최대 길이를 시작점과 끝점으로 설정
        2. 이진 탐색으로 남은 떡의 길이가 M보다 크거나 같을 때마다 정답과 크기 비교
        3. 탐색 종료 후 정답 출력
        
        ```python
        N,M = map(int,input().split())
        array=list(map(int,input().split()))
        answer=0
        
        start,end=min(array),max(array)
        while start<=end:
          mid=(start+end)//2
          result=0
          for i in array:
            if i>mid:
              result+=i-mid
              
          if result>=M:
            start=mid+1
            answer=max(answer,mid)
          else:
            end=mid-1
            
        print(answer)
        ```
        
    2. 풀이를 본 후
        
        값을 기준으로 이진 탐색할 때는 시작점은 0으로, 끝점은 탐색할 수 있는 범위에서 가장 큰 값으로 한다. 그리고 굳이 최대 값을 저장 안하고 크거나 같을 때마다 그냥 저장만 해도 알아서 최대 길이가 저장된다.
        
        ```python
        # 떡의 개수(N)와 요청한 떡의 길이(M)을 입력
        n, m = list(map(int, input().split(' ')))
        # 각 떡의 개별 높이 정보를 입력
        array = list(map(int, input().split()))
        
        # 이진 탐색을 위한 시작점과 끝점 설정
        start = 0
        end = max(array)
        
        # 이진 탐색 수행 (반복적)
        result = 0
        while(start <= end):
            total = 0
            mid = (start + end) // 2
            for x in array:
                # 잘랐을 때의 떡볶이 양 계산
                if x > mid:
                    total += x - mid
            # 떡볶이 양이 부족한 경우 더 많이 자르기 (오른쪽 부분 탐색)
            if total < m:
                end = mid - 1
            # 떡볶이 양이 충분한 경우 덜 자르기 (왼쪽 부분 탐색)
            else:
                result = mid # 최대한 덜 잘랐을 때가 정답이므로, 여기에서 result에 기록
                start = mid + 1
        
        # 정답 출력
        print(result)
        ```
        
    3. 해결한 후
        
        이진 탐색의 기초 설정부터 다시 복습할 수 있는 좋은 문제였다.

## 기출 문제

- 기출 문제 1: 정렬된 배열에서 특정 수의 개수 구하기

    1. 내 풀이
        1. x를 찾을 때 가장 작은 인덱스, 가장 큰 인덱스를 찾는 함수를 구현
        2. 두개의 인덱스 차를 이용해서 개수를 구한다.
        
        처음에는 계수 정렬로 구현했으나, 시간 초과를 막기 위해 인덱스를 찾는 함수 구현
        
        ```python
        def binarysearch1(x):
          idx=-1
          start,end=0,N-1
          while start<=end:
            mid=(start+end)//2
        
            if array[mid]==x:
              idx=mid
              end=mid-1
            elif array[mid]>x:
              end=mid-1
            else:
              start=mid+1
          return idx
        
        def binarysearch2(x):
          idx=-1
          start,end=0,N-1
          while start<=end:
            mid=(start+end)//2
          
            if array[mid]==x:
              idx=mid
              start=mid+1
            elif array[mid]>x:
              end=mid-1
            else:
              start=mid+1
          return idx
        
        N,x = map(int,input().split())
        array=list(map(int,input().split()))
          
        idx1=binarysearch1(x)
        idx2=binarysearch2(x)
        
        if idx1==-1 and idx2==-1:
          print(-1)
        else:
          print(idx2-idx1+1)
        ```
        
    2. 풀이를 본 후
        
        재귀와 bisect라이브러리를 이용하여 해결했다. 방식은 내 풀이와 비슷했다.
        
        ```python
        from bisect import bisect_left, bisect_right
        
        # 값이 [left_value, right_value]인 데이터의 개수를 반환하는 함수
        def count_by_range(array, left_value, right_value):
            right_index = bisect_right(array, right_value)
            left_index = bisect_left(array, left_value)
            return right_index - left_index
        
        n, x = map(int, input().split()) # 데이터의 개수 N, 찾고자 하는 값 x 입력 받기
        array = list(map(int, input().split())) # 전체 데이터 입력 받기
        
        # 값이 [x, x] 범위에 있는 데이터의 개수 계산
        count = count_by_range(array, x, x)
        
        # 값이 x인 원소가 존재하지 않는다면
        if count == 0:
            print(-1)
        # 값이 x인 원소가 존재한다면
        else:
            print(count)
        ```
        
    3. 해결한 후
        
        시간 제한을 지키기 위해 항상 고려를 해야 한다. 그리고 bisect 라이브러리에 대해서 알게 되었다.

- 기출 문제 2: 고정점 찾기

    1. 내 풀이
        1. 중간 값과 중간 값이 가리키는 지점의 값을 기준으로 이진 탐색 진행
        2. 찾으면 중간 값을 못 찾으면 -1을 출력한다.
        
        ```python
        def binarysearch():
          answer=-1
          start,end=0,N-1
          while start<=end:
            mid=(start+end)//2
        
            if array[mid]==mid:
              answer=mid
              break
            elif array[mid]>mid:
              end=mid-1
            else:
              start=mid+1
          return answer
        
        N = int(input())
        array=list(map(int,input().split()))
          
        result=binarysearch()
        print(result)
        ```
        
    2. 풀이를 본 후
        
        나의 풀이를 재귀 형태로 구현했다.
        
        ```python
        # 이진 탐색 소스코드 구현(재귀 함수)
        def binary_search(array, start, end):
            if start > end:
                return None
            mid = (start + end) // 2
            # 고정점을 찾은 경우 인덱스 반환
            if array[mid] == mid:
                return mid
            # 중간점이 가리키는 값보다 중간점이 작은 경우 왼쪽 확인
            elif array[mid] > mid:
                return binary_search(array, start, mid - 1)
            # 중간점이 가리키는 값보다 중간점이 큰 경우 오른쪽 확인
            else:
                return binary_search(array, mid + 1, end)
        
        n = int(input())
        array = list(map(int, input().split()))
        
        # 이진 탐색(Binary Search) 수행
        index = binary_search(array, 0, n - 1)
        
        # 고정점이 없는 경우 -1 출력
        if index == None:
            print(-1)
        # 고정점이 있는 경우 해당 인덱스 출력
        else:
            print(index)
        ```
        
    3. 해결한 후
        
        이진 탐색에서는 비교 기준에 따라 탐색 범위를 지정하는 것이 중요하다.

- 기출 문제 3: 공유기 설치

    1. 내 풀이
        1. 집의 좌표 오름차순으로 정렬
        2. 이진 탐색으로 mid값 구한 후 첫번째 집의 위치부터 mid이상으로 떨어진 집의 개수 구함
        3. 집의 개수를 기준으로 최대거리 탐색 후 출력
        
        input()에서 시간 초과가 나와 PyPy3로 제출하니 해결했다. 푸는데 시간이 1시간 정도 걸렸다…
        
        ```python
        def binarysearch():
          answer=0
          start,end=0,array[N-1]
          while start<=end:
            mid=(start+end)//2
            count=1
            temp=array[0]
            for i in range(1,N):
              if array[i]-temp>=mid:
                count+=1
                temp=array[i]
        
            if count>=C:
              answer=mid
              start=mid+1
            else:
              end=mid-1
          return answer
        
        N,C= map(int,input().split())
        array=[]
        for i in range(N):
          array.append(int(input()))
        
        array.sort()
        result=0
        if N==2:
          result=array[1]-array[0]
        else:
          result=binarysearch()
        print(result)
        ```
        
    2. 풀이를 본 후
        
        나의 풀이와 거의 유사했다. start, end의 값을 인접 거리의 최소, 최대로 해줬다.
        
        ```python
        # 집의 개수(N)와 공유기의 개수(C)를 입력 받기
        n, c = list(map(int, input().split(' ')))
        
        # 전체 집의 좌표 정보를 입력 받기
        array = []
        for _ in range(n):
             array.append(int(input()))
        array.sort() # 이진 탐색 수행을 위해 정렬 수행
        
        start = 1 # 가능한 최소 거리(min gap)
        end = array[-1] - array[0] # 가능한 최대 거리(max gap)
        result = 0
        
        while(start <= end):
            mid = (start + end) // 2 # mid는 가장 인접한 두 공유기 사이의 거리(gap)을 의미
            # 첫째 집에는 무조건 공유기를 설치한다고 가정
            value = array[0]
            count = 1
            # 현재의 mid 값을 이용해 공유기를 설치하기
            for i in range(1, n): # 앞에서부터 차근차근 설치 
                if array[i] >= value + mid:
                    value = array[i]
                    count += 1
            if count >= c: # C개 이상의 공유기를 설치할 수 있는 경우, 거리를 증가시키기
                start = mid + 1
                result = mid # 최적의 결과를 저장
            else: # C개 이상의 공유기를 설치할 수 없는 경우, 거리를 감소시키기
                end = mid - 1
        
        print(result)
        ```
        
    3. 해결한 후
        
        예전에 틀렸던 문제였지만 다시 보고 탐색할 값을 잘 지정하니 해결했다. 이진 탐색에서는 구하고자 하는 것을 기준으로 이진 탐색을 수행하면 된다.