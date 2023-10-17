---
layout: post
title: 구현 (Implementation)
description: 이것이 취업을 위한 코딩테스트다 with 파이썬 책을 참고하여 **구현**을 공부한 내용입니다.
sitemap: false
---

그리디와 비슷하게 코딩테스트 공부의 기초

* this unordered seed list will be replaced by the toc
{:toc}


## 피지컬로 승부 하기
- 머릿속에 있는 알고리즘을 소스 코드로 바꾸는 과정
- 모든 범위의 코딩 테스트 유형을 포함하는 개념
- 풀이를 떠올리기는 쉽지만, 소스 코드로 옮기기는 어렵다.

> - 완전 탐색: 모든 경우의 수를 주저 없이 다 계산하는 해결 방법
> - 시뮬레이션: 문제에서 제시한 알고리즘을 한 단계씩 차례대로 직접 수행


## 고려할 메모리 제약사항
    
> 파이썬
> 
> - 직접 자료형 저장 필요 X, 매우 큰 수의 연산 지원,
> - 실수형 변수는 유효숫자에 따라 연산 결과가 다를 수 있음을 기억
> - 리스트의 크기 꼭 고려
> - 메모리 사용량 보다 더 적은 양의 메모리를 사용해야 하는 것 기억
> 
>   | 데이터의 개수(리스트의 길이) | 메모리 사용량 |
>   | --- | --- |
>   | 1000 | 약 4KB |
>   | 1000000 | 약 4MB |
>   | 10000000 | 약 40 MB |

## 채점 환경
- 파이썬은 C/C++에 비해 수행 시간이 2배정도 됨
- 파이썬 - `1초에 2000만번의 연산 수행한다고 가정`하면 문제 없다.
- ex) 시간제한 1초, 데이터 100만개 → 시간복잡도 O(NlogN)이내로 해결해야함

## 접근 방법
- 언어 별 비교

    |  | 구현 난이도 | 프로그램 실행 시간 |
    | --- | --- | --- |
    | 파이썬 | 쉬운 편 | 긴 편 |
    | PyPy | 쉬운 편 | 다소 짧은 편 |
    | C++ | 어려운 편 | 짧은 편 |

- PyPy3 Python보다 연산 속도 빠르다. (`대략 1초에 2000만 번에서 1억번 정도의 연산 처리`)
- PyPy3의 환경을 지원하면 활용하도록 하자

## 예제 풀이
- 예제 1: 왕실의 나이트

    1. 내 풀이
        
        1. 우선 x,y방향으로 이동 좌표 배열 생성
        
        1. 그 후, 반복문을 통해 나이트가 이동 가능할 때만 카운트
        
        ```python
        def solve(start_x,start_y):
        global count
        
        for i in range(8):
            nx,ny=start_x+dx[i],start_y+dy[i]
            if 0<=nx<8 and 0<=ny<8:
            count+=1
        
        start=input()
        start_x,start_y=int(start[1])-1,ord(start[0])-ord('a')
        dx=[2,2,-2,-2,-1,1,-1,1]
        dy=[-1,1,-1,1,2,2,-2,-2]
        count=0
        solve(start_x,start_y)
        print(count)
        ```
        
    2. 풀이를 본 후
        
        이동 좌표를 한번에 정의할 수도 있었다…
        
    3. 해결한 후
        
        기본적으로 좌표를 이동 시키면서 조건에 맞을 때만 카운트 해주면 되었다.
    
        
- 예제 2: 게임 개발
    1. 내 풀이
        
        1. 현재 방향에 따라 네 방향으로 탐색을 진행한다.
        
        1. 그 후, 진행 할 수 있는 경우에 진행하면서 방문한 육지의 수를 세어준다.
        
        너무 의심을 하면서 문제를 풀었다…
        
    2. 풀이를 본 후
        
        ```python
        # N, M을 공백을 기준으로 구분하여 입력받기
        n, m = map(int, input().split())
        
        # 방문한 위치를 저장하기 위한 맵을 생성하여 0으로 초기화
        d = [[0] * m for _ in range(n)]
        # 현재 캐릭터의 X 좌표, Y 좌표, 방향을 입력받기
        x, y, direction = map(int, input().split())
        d[x][y] = 1 # 현재 좌표 방문 처리
        
        # 전체 맵 정보를 입력받기
        array = []
        for i in range(n):
            array.append(list(map(int, input().split())))
        
        # 북, 동, 남, 서 방향 정의
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]
        
        # 왼쪽으로 회전
        def turn_left():
            global direction
            direction -= 1
            if direction == -1:
                direction = 3
        
        # 시뮬레이션 시작
        count = 1
        turn_time = 0
        while True:
            # 왼쪽으로 회전
            turn_left()
            nx = x + dx[direction]
            ny = y + dy[direction]
            # 회전한 이후 정면에 가보지 않은 칸이 존재하는 경우 이동
            if d[nx][ny] == 0 and array[nx][ny] == 0:
                d[nx][ny] = 1
                x = nx
                y = ny
                count += 1
                turn_time = 0
                continue
            # 회전한 이후 정면에 가보지 않은 칸이 없거나 바다인 경우
            else:
                turn_time += 1
            # 네 방향 모두 갈 수 없는 경우
            if turn_time == 4:
                nx = x - dx[direction]
                ny = y - dy[direction]
                # 뒤로 갈 수 있다면 이동하기
                if array[nx][ny] == 0:
                    x = nx
                    y = ny
                # 뒤가 바다로 막혀있는 경우
                else:
                    break
                turn_time = 0
        
        # 정답 출력
        print(count)
        ```
        
    3. 해결한 후
        
        구현 문제는 문제의 주어진 조건을 기능으로 빠짐없이 구현하면 된다.
        

## 기출 문제

- 기출 문제 1: 럭키 스트레이트
    1. 내 풀이
        
        1. 입력받은 숫자를 문자열로 바꾼 후 리스트로 만들어준다.
        
        1. 반으로 나눈 후 왼쪽 리스트와 오른쪽 리스트의 합을 비교한다.
        
        ```python
        def solve(N):
          N_str = str(N)
          # 문자열의 각 문자를 리스트로 변환
          list_n = [int(digit) for digit in N_str]
          middle=len(list_n) //2
          left=list_n[:middle]
          right=list_n[middle:]
          
          if sum(left)==sum(right):
            return print("LUCKY")
          else:
            return print("READY")
        
        N=int(input())
        solve(N)
        ```
        
    2. 풀이를 본 후
        
        굳이 문자열로 바꾸고 하지 않고 그냥 처음부터 문자열로 입력 받는다. 그리고, 두 합의 차를 이용하여 분류한다…
        
        ```python
        n = input()
        length = len(n) # 점수 값의 총 자릿수
        summary = 0
        
        # 왼쪽 부분의 자릿수의 합 더하기
        for i in range(length // 2):
            summary += int(n[i])
        
        # 오른쪽 부분의 자릿수의 합 빼기
        for i in range(length // 2, length):
            summary -= int(n[i])
        
        # 왼쪽 부분과 오른쪽 부분의 자릿수 합이 동일한지 검사
        if summary == 0:
            print("LUCKY")
        else:
            print("READY")
        ```
        
    3. 해결한 후
        
        숫자를 리스트로 바꾸는 방법말고 그저 문자열로 입력받고 처리하면 된다…
    


    
## **참고 문헌 및 사이트** 

- 이것이 취업을 위한 코딩테스트다 with 파이썬