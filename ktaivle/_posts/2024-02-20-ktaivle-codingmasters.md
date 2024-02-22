---
layout: post
title: KT AIVLE SCHOOL 5기 코딩 테스트 연습 | 코딩 마스터즈
description: KT AIVLE SCHOOL 5기 코딩 마스터즈  정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}


## 초급

1. `문제 8068: 압축된 수열`
    - 문제 내용
        
        ![Untitled](/assets/img/blog/codingmasters/8068.png)
        
    - 내 풀이
        1. 10 진법부터 시작하여 각 숫자별로 길이를 총 파일 길이에 더함
        2. 각 숫자들이 몇차항까지 가능한지만 알면 자릿수를 구하는 것은 쉬웠다
        
        ```python
        # -*- coding: utf-8 -*-
        import sys
        input = sys.stdin.readline
        
        n, m = map(int, input().split())
        nums = list(map(int, input().split()))
        answer = 10
        file = 0
        
        while answer <= 62:
            for num in nums:
                i = 0
                while True:
                    if (answer ** i) <= num < (answer ** (i + 1)):
                        break
                    i += 1
                    
                file += (i + 1) # 자릿수 더함
                
            file += (n - 1) # 공백의 길이 더함
            
            if file <= m:
                break
                
            answer += 1   
            file = 0
        
        if answer == 63:
            print(-1)
        else:
            print(answer)
        ```
        
    - 해결한 후
        
        각 숫자별 자릿수 구하는 부분에서 조금 헤맸다…
        

1. `문제 8065: 채터링`
    - 문제 내용
        
        ![Untitled](/assets/img/blog/codingmasters/8065.png)
        
    - 풀이
        1. 단순히 각 문자들을 k번 연속으로 answer 리스트에 저장
        2. 최종 문자열 출력
        
        ```python
        # -*- coding: utf-8 -*-
        import sys
        input = sys.stdin.readline
        
        n, k = map(int, input().split())
        S = input()
        answer = []
        
        for i in range(n):
            for j in range(k):
                answer.append(S[i]) # 각 문자 k번 저장
        
        print(''.join(answer))  # 총 문자열 출력력
        ```
        
    - 해결한 후
        
        단순한 문제였다