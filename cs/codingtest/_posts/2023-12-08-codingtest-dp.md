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

- 예제 4: 효율적인 화폐 구하기
    1. 내 풀이
        1. 화폐단위를 큰 단위부터 내림차순으로 정렬
        2. 그 후, 화폐를 나누고 개수를 더해서 출력
        
        다이나믹 프로그래밍으로의 풀이를 구현 못해서 단순히 구현
        
        ```python
        def solve(m):
          count=0
          for a in array:
            if m==0:
              break
            else:
              count+=m//a
              m=(m%a)
        
          if m!=0:
            return print(-1)
          else:
            return print(count)
              
        n,m=map(int,input().split())
        array=[]
        for i in range(n):
          array.append(int(input()))
        array.sort(reverse=True)
        solve(m)
        ```
        
    2. 풀이를 본 후
        
        우선, m을 기준으로 dp를 진행하고 인덱스를 이용하여 전에 만들 수 있는 값이 있으면 비교를 해주는 방법으로 진행한다…
        
        항상 구하고자 하는 것을 기준으로 한다면 해결 가능하다…
        
        ```python
        # 정수 N, M을 입력 받기
        n, m = map(int, input().split())
        # N개의 화폐 단위 정보를 입력 받기
        array = []
        for i in range(n):
            array.append(int(input()))
        
        # 한 번 계산된 결과를 저장하기 위한 DP 테이블 초기화
        d = [10001] * (m + 1)
        
        # 다이나믹 프로그래밍(Dynamic Programming) 진행(보텀업)
        d[0] = 0
        for i in range(n):
            for j in range(array[i], m + 1):
                if d[j - array[i]] != 10001: # (i - k)원을 만드는 방법이 존재하는 경우
                    d[j] = min(d[j], d[j - array[i]] + 1)
        
        # 계산된 결과 출력
        if d[m] == 10001: # 최종적으로 M원을 만드는 방법이 없는 경우
            print(-1)
        else:
            print(d[m])
        ```

## 기출 문제

- 기출 문제 1: 금광
    1. 내 풀이
        1. 우선 입력된 값들을 array에 저장 후, 다시 dp에 저장한다.
        2. 그 후, 왼쪽 위, 왼쪽, 왼쪽 아래에서 오는 경우 중 큰 값을 다음 열의 값에 더해간다.
        3. 최종적으로 마지막 열의 값들 중에서 최대 값을 출력한다.
        
        ```python
        def solve(n,m):
          for i in range(1,m):
            for j in range(n):
              if j==0:
                dp[j][i]+=max(dp[j+1][i-1],dp[j][i-1])
              elif j==n-1:
                dp[j][i]+=max(dp[j-1][i-1],dp[j][i-1])
              else:
                dp[j][i]+=max(dp[j-1][i-1],dp[j+1][i-1],dp[j][i-1])
        
          column = [row[m-1] for row in dp]
          return print(max(column))
          
        T = int(input())
        for i in range(T):
          n, m = map(int, input().split())
          array = list(map(int, input().split()))
          dp = [[0]*m for _ in range(n)]
          for i in range(n):
            for j in range(m):
              a=array.pop(0)
              dp[i][j]=a
          solve(n,m)
        ```
        
    2. 풀이를 본 후
        
        dp에 저장할 때, 행을 기준으로 삽입하는 방법, 경우에 따라 값을 0으로 바꾸는 방법이 유용한것 같다.
        
        ```python
        # 테스트 케이스(Test Case) 입력
        for tc in range(int(input())):
            # 금광 정보 입력
            n, m = map(int, input().split())
            array = list(map(int, input().split()))
        
            # 다이나믹 프로그래밍을 위한 2차원 DP 테이블 초기화
            dp = []
            index = 0
            for i in range(n):
                dp.append(array[index:index + m])
                index += m
        
            # 다이나믹 프로그래밍 진행
            for j in range(1, m):
                for i in range(n):
                    # 왼쪽 위에서 오는 경우
                    if i == 0:
                        left_up = 0
                    else:
                        left_up = dp[i - 1][j - 1]
                    # 왼쪽 아래에서 오는 경우
                    if i == n - 1:
                        left_down = 0
                    else:
                        left_down = dp[i + 1][j - 1]
                    # 왼쪽에서 오는 경우
                    left = dp[i][j - 1]
                    dp[i][j] = dp[i][j] + max(left_up, left_down, left)
        
            result = 0
            for i in range(n):
                result = max(result, dp[i][m - 1])
        
            print(result)
        ```
        
    3. 해결한 후
        
        행을 기준으로 여러번 삽입하면 이차원 배열을 만들 수 있었다. 이 부분에서 조금 시간이 걸린 것 같다.
        

- 기출 문제 2: 정수 삼각형
    1. 내 풀이
        1. 양 끝의 지점 들은 각각 위쪽 줄의 바로 인접한 값들을 더해간다.
        2. 나머지는 왼쪽 대각선 위, 오른쪽 대각선 위의 값들 중에서 큰 값을 더해간다.
        3. 최종적으로 마지막 행의 값들 중에서 최대 값을 출력한다.
        
        ```python
        def solve(n):
          for i in range(1,n):
            for j in range(i+1):
              if j==0:
                dp[i][j]+=dp[i-1][0]
              elif j==i:
                dp[i][j]+=dp[i-1][i-1]
              else:
                dp[i][j]+=max(dp[i-1][j-1],dp[i-1][j])
          
          return print(max(dp[n-1]))
          
        n = int(input())
        dp=[]
        for i in range(n):
          array=list(map(int,input().split()))
          dp.append(array)
        solve(n)
        ```
        
    2. 풀이를 본 후
        
        인덱스를 기준으로 값을 0으로 지정하여 구했다.
        
        ```python
        n = int(input())
        dp = [] # 다이나믹 프로그래밍을 위한 DP 테이블 초기화
        
        for _ in range(n):
            dp.append(list(map(int, input().split())))
        
        # 다이나믹 프로그래밍으로 2번째 줄부터 내려가면서 확인
        for i in range(1, n):
            for j in range(i + 1):
                # 왼쪽 위에서 내려오는 경우
                if j == 0:
                    up_left = 0
                else:
                    up_left = dp[i - 1][j - 1]
                # 바로 위에서 내려오는 경우
                if j == i:
                    up = 0
                else:
                    up = dp[i - 1][j]
                # 최대 합을 저장
                dp[i][j] = dp[i][j] + max(up_left, up)
        
        print(max(dp[n - 1]))
        ```
        
    3. 해결한 후
        
        책의 풀이와 같은 방법도 알고 있어야겠다.

- 기출 문제 3: 퇴사
    1. 내 풀이
        1. 우선 임의의 날짜까지 얻을 수 있는 최대 수익을 갱신하면서 dp값으로 넣어준다.
        2. 최종적으로 마지막 부분의 dp값이 얻을 수 있는 최대 이익이 된다.
        
        일수를 더해나가는 방식으로 너무 단순하게 생각해서 오래걸렸다…
        
        ```python
        def solve(n):
          for i in range(n+1):
            for j in range(i):
              if j+T[j] <= i:
                dp[i]=max(dp[i],dp[j]+P[j])
            
          return print(dp[n])
          
        n = int(input())
        dp=[0]*16
        T,P=[],[]
        for i in range(n):
          t,p=map(int,input().split())
          T.append(t)
          P.append(p)
        
        solve(n)
        ```
        
    2. 풀이를 본 후
        
        풀이는 뒤쪽 부터 최대 값을 지정하면서 진행해왔다. 
        
        ```python
        n = int(input()) # 전체 상담 개수
        t = [] # 각 상담을 완료하는데 걸리는 기간
        p = [] # 각 상담을 완료했을 때 받을 수 있는 금액
        dp = [0] * (n + 1) # 다이나믹 프로그래밍을 위한 1차원 DP 테이블 초기화
        max_value = 0
        
        for _ in range(n):
            x, y = map(int, input().split())
            t.append(x)
            p.append(y)
        
        # 리스트를 뒤에서부터 거꾸로 확인
        for i in range(n - 1, -1, -1):
            time = t[i] + i
            # 상담이 기간 안에 끝나는 경우
            if time <= n:
                # 점화식에 맞게, 현재까지의 최고 이익 계산
                dp[i] = max(p[i] + dp[time], max_value)
                max_value = dp[i]
            # 상담이 기간을 벗어나는 경우
            else:
                dp[i] = max_value
        
        print(max_value)
        ```
        
    3. 해결한 후
        
        문제를 제대로 읽고 조건과 진행 상황을 제대로 파악해야한다.

- 기출 문제 4: 병사 배치하기
    1. 내 풀이
        1. 우선 전투력 배열을 거꾸로 뒤집는다
        2. 그 후, 증가하는 최대의 길이를 dp값으로 구한다
        3. n-dp[n]으로 값을 구한다
        
        그런데 틀렸다. 
        
        ```python
        def solve(n):
          for i in range(1, n):
            for j in range(i):
              if array[i] >= array[j]:
                dp[i + 1] = max(dp[i + 1], dp[j + 1] + 1)
              else:
                dp[i + 1] = max(dp[i + 1], dp[j + 1])
        
          return print(n - dp[n])
        
        n = int(input())
        dp = [0] * 2001
        dp[1] = 1
        array = list(map(int, input().split()))
        array.reverse()
        solve(n)
        ```
        
    2. 풀이를 본 후
        
        우선 풀이의 개념은 맞았지만 구현에서 문제가 있었다. 
        
        - 첫번째로 모든 dp테이블을 1로 두고 하는 것이 더 편했다.
        - 두번째는 증가하는 부분이 아니면 넘어가야 한다. 증가하는 경우만 처리해야 정확하게 개수가 세어지기 때문이다.
        - 마지막은 dp에서 최대 값을 빼줘야 한다. 항상 끝 부분에서 최대 길이가 나오지 않을 수 있기 때문이다.
        
        ```python
        n = int(input())
        array = list(map(int, input().split()))
        # 순서를 뒤집어 '최장 증가 부분 수열' 문제로 변환
        array.reverse()
        
        # 다이나믹 프로그래밍을 위한 1차원 DP 테이블 초기화
        dp = [1] * n
        
        # 가장 긴 증가하는 부분 수열(LIS) 알고리즘 수행
        for i in range(1, n):
            for j in range(0, i):
                if array[j] < array[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        # 열외해야 하는 병사의 최소 수를 출력
        print(n - max(dp))
        ```

- 기출 문제 5: 못생긴 수
    1. 내 풀이
        1. 2,3,5의 조합으로 만들 수 있는 수를 차례로 dp테이블에 넣는다.
        2. n번째 못생긴 수 출력
        
        하지만 어떻게 구현할지 모르겠어서 실패
        
    2. 풀이를 본 후
        
        2,3,5를 곱할 인덱스를 정해두고 계속해서 작은 수부터 곱해나가면서 dp테이블에 넣어준다. 최종적으로 n번째 수를 출력하면 된다.
        
        다음 2,3,5곱해진 수와, 계속해서 곱할 인덱스를 정하는 것이 어려웠다.
        
        ```python
        n = int(input())
        
        ugly = [0] * n # 못생긴 수를 담기 위한 테이블 (1차원 DP 테이블)
        ugly[0] = 1 # 첫 번째 못생긴 수는 1
        
        # 2배, 3배, 5배를 위한 인덱스
        i2 = i3 = i5 = 0
        # 처음에 곱셈 값을 초기화
        next2, next3, next5 = 2, 3, 5
        
        # 1부터 n까지의 못생긴 수들을 찾기
        for l in range(1, n):
            # 가능한 곱셈 결과 중에서 가장 작은 수를 선택
            ugly[l] = min(next2, next3, next5)
            # 인덱스에 따라서 곱셈 결과를 증가
            if ugly[l] == next2:
                i2 += 1
                next2 = ugly[i2] * 2
            if ugly[l] == next3:
                i3 += 1
                next3 = ugly[i3] * 3
            if ugly[l] == next5:
                i5 += 1
                next5 = ugly[i5] * 5
        
        # n번째 못생긴 수를 출력
        print(ugly[n - 1])
        ```

- 기출 문제 6: 편집 거리
    1. 내 풀이
        1. 문자열의 가장 긴 같은 부분을 찾아 해결하려 했지만 구현에 실패했다…
    2. 풀이를 본 후
        
        2차원 배열의 dp 테이블을 만들어서 점화식을 이용하여 계산한다… 
        
        dp 테이블을 2차원 형태로 만드는 방법도 알고있어야한다…
        
        ```python
        # 최소 편집 거리(Edit Distance) 계산을 위한 다이나믹 프로그래밍
        def edit_dist(str1, str2):
            n = len(str1)
            m = len(str2)
        
            # 다이나믹 프로그래밍을 위한 2차원 DP 테이블 초기화
            dp = [[0] * (m + 1) for _ in range(n + 1)]
        
            # DP 테이블 초기 설정
            for i in range(1, n + 1):
                dp[i][0] = i
            for j in range(1, m + 1):
                dp[0][j] = j
            
            # 최소 편집 거리 계산
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    # 문자가 같다면, 왼쪽 위에 해당하는 수를 그대로 대입
                    if str1[i - 1] == str2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    # 문자가 다르다면, 세 가지 경우 중에서 최솟값 찾기
                    else: # 삽입(왼쪽), 삭제(위쪽), 교체(왼쪽 위) 중에서 최소 비용을 찾아 대입
                        dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
        
            return dp[n][m]
        
        # 두 문자열을 입력 받기
        str1 = input()
        str2 = input()
        
        # 최소 편집 거리 출력
        print(edit_dist(str1, str2))
        ```

## 출처

- 이것이 취업을 위한 코딩테스트다 with 파이썬