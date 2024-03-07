---
layout: post
title: KT AIVLE SCHOOL 5기 코딩 테스트 연습 | 1차 코딩 마스터스
description: KT AIVLE SCHOOL 5기 1차 코딩 마스터스 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}

## 기초

1. `문제 8469. 비밀번호 찾기` 
    - 문제 내용
        
        ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8469.png)
        
    - 내 풀이
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
        
    - 해결한 후
        
        쉬운 문제이다
        
    
2. `문제 8471: 8진수와 16진수` 
    - 문제 내용
        
        ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8471.png)
        
    - 내 풀이
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
        
    - 해결한 후
        
        변수 사용에서 실수를 해서 시간이 조금 걸렸다
        
    
3. `문제 8481. 우리반 아이큐왕은`
    - 문제 내용
        
        ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8481.png)
        
    - 내 풀이
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
        
    - 해결한 후
        
        IQ가 같을 떄 먼저 입력된 학생을 앞에 세우는 부분에서 조금 막혔다
        
    
4. `문제 8487. 최대공약수`
    - 문제 내용
        
        ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8487.png)
        
    - 내 풀이
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
        
    - 해결한 후
        
        유클리드 호제법을 다시 한번 상기시킬 수 있었다

