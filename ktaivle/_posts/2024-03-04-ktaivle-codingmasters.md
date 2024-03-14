---
layout: post
title: KT AIVLE SCHOOL 5기 코딩 테스트 연습 | 1차 코딩 마스터스
description: KT AIVLE SCHOOL 5기 1차 코딩 마스터스 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}

## 기초

### `문제 8469. 비밀번호 찾기` 

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8469.png)
    
2. 내 풀이
    1. 입력된 문자열을 공백을 두고 출력하다가 ‘c’를 발견하면 출력 후 종
    
    ```python
    # -*- coding: utf-8 -*-
    import sys
    input = sys.stdin.readline
    
    answer = list(input().split())
    
    for a in answer:
        if a == 'c':
            print(a)
            break
        print(a, end = ' ')
    ```
    
3. 해결한 후
    
    쉬운 문제이다
        
    
### `문제 8471: 8진수와 16진수` 

1.  문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8471.png)
    
2. 내 풀이
    1. 8진수, 16진수 계산을 진행할 때마다의 몫을 리스트에 저장
    2. 각 리스트를 뒤집은 후 출력
    
    ```python
    # -*- coding: utf-8 -*-
    import sys
    input = sys.stdin.readline
    
    n = int(input())
    answer8, answer16 = [], []
    
    temp = n
    
    while temp > 0:
        answer8.append(str(temp % 8))
        temp //= 8
        
    while n > 0:
        if (n % 16) >= 10:
            answer16.append(chr(65 + ((n % 16) - 10)))
        else:
            answer16.append(str(n % 16))
        n //= 16
    
        
    answer8.reverse()
    answer16.reverse()
    
    a = ''.join(answer8)
    b = ''.join(answer16)
    print(f'{a} {b}')
    ```
    
3. 해결한 후
    
    변수 사용에서 실수를 해서 시간이 조금 걸렸다
        
    
### `문제 8481. 우리반 아이큐왕은`

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8481.png)
    
2. 내 풀이
    1. IQ 높은 순으로 정렬 후, 같으면 인덱스가 낮은 순으로 정렬
    2. 그 후, 상위 3명의 학생 이름 출력
    
    ```python
    # -*- coding: utf-8 -*-
    import sys
    input = sys.stdin.readline
    
    n = int(input())
    students = []
    
    for i in range(n):
        name, iq = input().split()
        students.append((int(iq), name))
    
    sorted_students = sorted(students, key=lambda x: (-x[0], students.index(x)))
    
    for i in range(3):
        print(student[i][1])
    ```
    
3. 해결한 후
    
    IQ가 같을 떄 먼저 입력된 학생을 앞에 세우는 부분에서 조금 막혔다
        
    
### `문제 8487. 최대공약수`

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8487.png)
    
2. 내 풀이
    1. 유클리드 호제법을 이용하여 최대공약수 구함
    
    ```python
    # -*- coding: utf-8 -*-
    import sys
    input = sys.stdin.readline
    
    n, m = map(int, input().split())
    
    while m != 0:
        n, m = m, n % m
    print(n)
    ```
    
3. 해결한 후
    
    유클리드 호제법을 다시 한번 상기시킬 수 있었다

## 초급

## `문제 8508. 신입사원 채용`

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8508.png)
    
2. 내 풀이
    1. 우선 (서류 점수, 면접 점수, 초기 인덱스) 를 리스트에 저장
    2. 그 후, 내림차순으로 정렬 후, 조건에 따라서 등수를 매겨준다
    
    먼저 정렬을 하고 조건에 따라 등수를 매겨줘야 했다…
    
    ```python
    import sys
    input = sys.stdin.readline
    
    n = int(input())
    array = []
    for i in range(n):
        x, y = map(int, input().split())
        array.append((x, y, i))
    
    array.sort(reverse = True)
    
    compare = []
    answer = [1] * n
    for i in range(n - 1):
        a1, a2, index1 = array[i]
        for j in range(i + 1, n):
            b1, b2, index2 = array[j]
            if a2 > b2: # A > B
                answer[index2] += 1
            elif a2 == b2:
                if a1 == b1:
                    compare.append((index1, index2))
                else:
                    answer[index2] += 1
            else:
                if a1 == b1:
                    answer[index1] += 1
                else:
                    compare.append((index1, index2))
                
    for i, j in compare:
        if answer[i] != answer[j]:
            min_num = min(answer[i], answer[j])
            answer[i], answer[j] = min_num, min_num
        
    for a in answer:
        print(a, end = ' ')
    print()
    ```
    
3. 해결한 후
    
    항상 정렬과 관련된 문제는 초기에 정렬이 대부분 필요하다
    

## `문제 8509. 커트라인 정하기`

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8509.png)
    
2. 내 풀이
    1. 각 학생들의 점수들을 기준으로 합격 학생의 수를 구함
    2. 조건을 만족할 때마다 리스트에 넣는다
    3. 최종적으로 합격자, 점수 순으로 내림차순 정렬하여 맨 앞 요소의 점수 출력
    4. 리스트가 비어있으면 점수 최솟값 -1 출력
    
    ```python
    import sys
    input = sys.stdin.readline
    
    n, k = map(int, input().split())
    array = list(map(int, input().split()))
    answer = []
    
    for a in array:
        count = 0
        for i in range(n):
            if i == 0:
                if (array[i] >= a) or (array[i + 1] >= a):
                    count += 1
            elif i == n - 1:
                if (array[i] >= a) or (array[i - 1] >= a):
                    count += 1
            else:
                if (array[i] >= a) or (array[i + 1] >= a) or (array[i - 1] >= a):
                    count += 1
        
        if count <= k:
            answer.append((count, a))
            
    
    answer.sort(reverse = True)
    
    if not answer:
        print(min(array) - 1)
    else:
        print(answer[0][1])
    ```
    
3. 해결한 후
    
    너무 어렵게 접근했다
    

## `문제 8511. 가우스와 정다각형`

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8511.png)
    
2. 내 풀이
    1. 우선 페르마 소수로 최대한 나누어 준다
    2. 그 후, 나누어진 결과가 2의 거듭 제곱인지 확인해준다
    3. 최종 조건에 맞게 결과 출력
    
    ```python
    import sys
    input = sys.stdin.readline
    
    k = int(input())
    array = [3, 5, 17, 257, 65537]
    visited = [0] * 5
    
    while True:
        temp = False
        
        for i in range(5):
            if (k % array[i] == 0) and (visited[i] == 0):
                visited[i] += 1
                temp = True
                k //= array[i]
                break
                
        if not temp:
            break      
    
    if k == 1:
        print('YES')
    else:
        x = 4 
        array2 = []
        while x <= 10 ** 18:
            array2.append(x)
            x *= 2
        
        temp2 = False
        for a2 in array2:
            if k == a2:
                temp = True
                break
        
        if temp:
            print('YES')
        else:
            print('NO')
    ```
    
3. 해결한 후
    
    처음에 문제 이해를 잘못했다