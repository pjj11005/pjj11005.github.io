---
layout: post
title: 이것이 취업을 위한 코딩테스트다 | 6. 다이나믹 프로그래밍 (Dynamic Programming)
categories: 
  - cs
  - codingtest
description: 이것이 취업을 위한 코딩테스트다 with 파이썬 책을 참고하여 **다이나믹 프로그래밍**을 공부한 내용입니다.
sitemap: false
---

한 번 계산한 문제는 다시 계산하지 않도록 하는 알고리즘

* this unordered seed list will be replaced by the toc
{:toc}

## 중복되는 연산을 피하자

- 메모리 공간을 약간 더 사용하면 연산 속도를 비약적으로 증가시킬 수 있는 기법
- 동적 할당의 다이나믹과는 다른 의미
- ex) 피보나치 수열
    - 점화식을 재귀함수 사용하여 구현
        
        ```python
        # 피보나치 함수(Fibonacci Function)을 재귀함수로 구현
        def fibo(x):
            if x == 1 or x == 2:
                return 1
            return fibo(x - 1) + fibo(x - 2)
        
        print(fibo(4))
        ```
        
        - n이 커질 수록 수행 시간 기하급수적으로 증가
        - 시간복잡도: $$O(2^N)$$
        - n이 커질수록 반복 호출하는 값이 많아진다
    - 다이나믹 프로그래밍 사용 조건
        - 큰 문제를 작은 문제로 나눌 수 있다
        - 작은 문제에서 구한 정답은 그것을 포함하는 큰 문제에서도 동일하다
    - 메모이제이션(Memoization) 기법 사용
        - 다이나믹 프로그래밍 기법 중 하나
        - 한번 구현한 결과를 메모리 공간에 메모해두고 같은 식이 호출될 때 메모한 결과를 사용하는 기법
        - 캐싱: 메모이제이션에서 값을 저장하는 방법
        - 한번 구한 정보 리스트에 저장 후 재귀적으로 수행하다가 같은 정보 필요할 때 리스트의 값 사용
            
            ```python
            # 한 번 계산된 결과를 메모이제이션(Memoization)하기 위한 리스트 초기화
            d = [0] * 100
            
            # 피보나치 함수(Fibonacci Function)를 재귀함수로 구현 (탑다운 다이나믹 프로그래밍)
            def fibo(x):
                # 종료 조건(1 혹은 2일 때 1을 반환)
                if x == 1 or x == 2:
                    return 1
                # 이미 계산한 적 있는 문제라면 그대로 반환
                if d[x] != 0:
                    return d[x]
                # 아직 계산하지 않은 문제라면 점화식에 따라서 피보나치 결과 반환
                d[x] = fibo(x - 1) + fibo(x - 2)
                return d[x]
            
            print(fibo(99))
            ```
            
            - 재귀함수 사용 방식은 오버헤드 발생할 수 있어 일반적으로 반복문 사용한 방식이 더 성능이 좋다
            - 시간 복잡도: $$O(N)$$
        - 탑 다운 방식(Top-down)이다 → 큰 문제를 해결하기 위해 작은 문제를 호출(하향식)
    - 반복문 사용
        - 보텀업 방식(Bottom-up)이다 → 작은 문제부터 차근차근 답을 도출(상향식), 전형적인 방식
            
            ```python
            # 앞서 계산된 결과를 저장하기 위한 DP 테이블 초기화
            d = [0] * 100
            
            # 첫 번째 피보나치 수와 두 번째 피보나치 수는 1
            d[1] = 1
            d[2] = 1
            n = 99
            
            # 피보나치 함수(Fibonacci Function) 반복문으로 구현(보텀업 다이나믹 프로그래밍)
            for i in range(3, n + 1):
                d[i] = d[i - 1] + d[i - 2]
            
            print(d[n])
            ```
            
        - DP 테이블: 보텀업 방식에서 사용되는 결과 저장용 리스트
- 코딩 테스트에서의 다이나믹 프로그래밍 → 대체로 간단한 형태 출제
- 문제 푸는 방식
    - 첫번째로 당연히 다이나믹 프로그래밍 유형인지 파악
        - 완전 탐색으로 접근했을 때 시간이 오래걸리면 다이나믹 프로그래밍으로 접근할 수 있는지 부분 문제들의 중복 여부 확인
    - 재귀함수 작성 후 메모이 제이션 적용가능하면 변경하는 것도 좋다
    - 보텀업 방식으로 구현 권장
        - 시스템상의 스택 크기가 한정되어 있을 수도 있기 때문이다
    

## 예제 문제

- 예제 1: 1로 만들기
    1. 내 풀이
        1. 우선 보텀업 방식으로 1부터 최소의 연산 수로 n에 도달하도록 한다.
        2. while문을 통해 5,3,2 순으로 곱할 수 있을 때 곱하면서 진행한다
        3. n에 도달하면 값을 출력
        
        내 풀이로 진행하면 시간복잡도가 $$O(N^2)$$이 나온다…
        
        ```python
        def solve(n):
          i=1
          while i<n:
            if (i*5)<n:
              dp[i*5]=dp[i]+1
              i*=5
            elif (i*3)<=n:
              dp[i*3]=dp[i]+1
              i*=3
            elif (i*2)<=n:
              dp[i*2]=dp[i]+1
              i*=2
            else:
              dp[i+1]=dp[i]+1
              i+=1
          return print(dp[n])
              
        n=int(input())
        dp=[0]*(n+1)
        solve(n)
        ```
        
        while문이 아닌 for문으로 개선하면 좋을 것 같다.
        
        ```python
        def solve(n):
          dp = [0] * (n + 1)
          for i in range(1, n + 1):
            for x in (5, 3, 2):
              if i * x <= n:
                dp[i * x] = min(dp[i * x], dp[i] + 1)
            dp[i + 1] = min(dp[i + 1], dp[i] + 1)
        
          print(dp[n])
        
        n = int(input())
        solve(n)
        ```
        
    2. 풀이를 본 후
        
        반복문과 보텀업 방식으로 수가 증가하면서 나누어떨어질 때 최소값을 넣어주면서 진행
        
        ```python
        # 정수 X를 입력 받기
        x = int(input())
        
        # 앞서 계산된 결과를 저장하기 위한 DP 테이블 초기화
        d = [0] * 1000001
        
        # 다이나믹 프로그래밍(Dynamic Programming) 진행(보텀업)
        for i in range(2, x + 1):
            # 현재의 수에서 1을 빼는 경우
            d[i] = d[i - 1] + 1
            # 현재의 수가 2로 나누어 떨어지는 경우
            if i % 2 == 0:
                d[i] = min(d[i], d[i // 2] + 1)
            # 현재의 수가 3으로 나누어 떨어지는 경우
            if i % 3 == 0:
                d[i] = min(d[i], d[i // 3] + 1)
            # 현재의 수가 5로 나누어 떨어지는 경우
            if i % 5 == 0:
                d[i] = min(d[i], d[i // 5] + 1)
        
        print(d[x])
        ```
        
    3. 해결한 후
        
        보텀업 방식을 다시 제대로 보고 이전의 저장된 값을 이용하여 큰 문제를 해결하도록 코드를 구현해야한다. 그리고 최대한 시간복잡도가 $$O(N)$$이 되도록 구현해야한다…

- 예제 2: 개미 전사
    1. 내 풀이
        1. n≥3일 때의 점화식 max(dp[n-2]+현재값, dp[n-1])을 구함
        2. n번째까지 반복문 수행하여 dp[n] 값 출력
        
        ```python
        def solve(n):
          for i in range(3,n+1):
            dp[i]=max(dp[i-2]+array[i-1],dp[i-1])
          return print(dp[n])
              
        n=int(input())
        array=list(map(int,input().split()))
        dp=[0]*(n+1)
        dp[1]=array[0]
        dp[2]=max(array[0],array[1])
        solve(n)
        ```
        
    2. 풀이를 본 후
        
        나와 같은 방식으로 구현
        
        ```python
        # 정수 N을 입력 받기
        n = int(input())
        # 모든 식량 정보 입력 받기
        array = list(map(int, input().split()))
        
        # 앞서 계산된 결과를 저장하기 위한 DP 테이블 초기화
        d = [0] * 100
        
        # 다이나믹 프로그래밍(Dynamic Programming) 진행 (보텀업)
        d[0] = array[0]
        d[1] = max(array[0], array[1]) 
        for i in range(2, n):
            d[i] = max(d[i - 1], d[i - 2] + array[i])
        
        # 계산된 결과 출력
        print(d[n - 1])
        ```
        
    3. 해결한 후
        
        다이나믹 프로그래밍 = 보텀업 방식 + 작은 문제들을 합쳐서 큰 문제 해결을 숙지
        

- 예제 3: 바닥 공사
    1. 내 풀이
        1. n≥3일 때, n-2까지의 dp가 2번, n-1까지의 dp가 1번 더해지는 규칙 발견
        2. 이를 이용하여 dp[n] 구함
        
        ```python
        def solve(n):
          for i in range(3,n+1):
            dp[i]=(2*dp[i-2])+dp[i-1]
          return print(dp[n]%796796)
              
        n=int(input())
        dp=[0]*1001
        dp[1]=1
        dp[2]=3
        solve(n)
        ```
        
    2. 풀이를 본 후
        
        내 풀이와 같다.
        
        ```python
        # 정수 N을 입력 받기
        n = int(input())
        
        # 앞서 계산된 결과를 저장하기 위한 DP 테이블 초기화
        d = [0] * 1001
        
        # 다이나믹 프로그래밍(Dynamic Programming) 진행 (보텀업)
        d[1] = 1
        d[2] = 3
        for i in range(3, n + 1):
            d[i] = (d[i - 1] + 2 * d[i - 2]) % 796796
        
        # 계산된 결과 출력
        print(d[n])
        ```
        
    3. 해결한 후
        
        타일 채우기 같은 그림 문제는 꼭 그림을 그려서 문제를 파악하자. 또한, 전체를 부분으로 나누어 관련성을 보면 빠르게 찾을 수 있다.

## **참고 문헌 및 사이트** 

- 이것이 취업을 위한 코딩테스트다 with 파이썬