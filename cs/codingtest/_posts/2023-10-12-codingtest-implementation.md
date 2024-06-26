---
layout: post
title: 2. 구현 (Implementation)
categories: 
  - cs
  - codingtest
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
- ex) 시간제한 1초, 데이터 100만개 → 시간복잡도 O(NlogN)이내로 해결해야함 (**약 2000만번의 연산**)

## 접근 방법
- 언어 별 비교

    |  | 구현 난이도 | 프로그램 실행 시간 |
    | --- | --- | --- |
    | 파이썬 | 쉬운 편 | 긴 편 |
    | PyPy | 쉬운 편 | 다소 짧은 편 |
    | C++ | 어려운 편 | 짧은 편 |

- PyPy3: Python보다 연산 속도 빠르다. (`대략 1초에 2000만 번에서 1억번 정도의 연산 처리`)
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
        
        - 처음 풀이
            1. 입력 받은 숫자를 문자열로 바꾼 후 리스트로 만들어준다.
            2. 반으로 나눈 후 왼쪽 리스트와 오른쪽 리스트의 합을 비교한다.
            
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
            
        - 두 번째 풀이
            1. 입력 받은 점수 문자열 변환
            2. 앞 부분의 합과 뒷 부분의 합의 차를 이용하여 결과 출력
            
            ```python
            n=int(input()) # 점수
            n_str=str(n) # 점수 문자열 변환
            length=len(n_str)
            sum=0
            
            for i in range(length//2): # 앞부분의 합
                sum+=int(n_str[i])
            
            for i in range(length//2, length): # 앞부분의 합과 뒷부분의 합의 차
                sum-=int(n_str[i])
            
            if sum==0:
                print('LUCKY')
            else:
                print('READY')
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
    

- 기출 문제 2: 문자열 재정렬
    1. 내 풀이
        
        - 처음 풀이
            1. 우선 알파벳은 리스트에 담고 숫자는 변수를 정해서 더해줬다.
            2. 그 후, 알파벳 리스트를 오름차순으로 정렬하고 숫자의 합과 연결시켜서 문자열을 만들었다.
            
            ```python
            def solve(S):
                eng=[]
                num=0
                
                for i in S:
                    if 'A'<=i<='Z':
                        eng.append(i)
                    else:
                        num+=int(i)
                    
                eng.sort()
                eng_string = ''.join(eng)
                answer=eng_string+str(num)
                
                return print(answer)
            
            S=input()
            solve(S)
            ```
            
        - 두번째 풀이
            1. 대문자 알파벳과 숫자를 따로 추출하고 각각 정렬 or 합을 구함
            2. 문자열로 만들어 연결
            
            ```python
            S=input() # 문자열 입력
            
            str_string = ''.join(char for char in S if not char.isdigit()) # 대문자 알파벳
            num = sum(int(char) for char in S if char.isdigit()) # 숫자들의 합
            sorted_string=''.join(sorted(str_string)) # 알파벳 정렬
            
            if num>0: # 숫자 존재
                result=sorted_string + str(num) # 알파벳 + 숫자 합
            else: # 숫자 없을 때
                result=sorted_string # 알파벳
            
            print(result)
            ```
        
    2. 풀이를 본 후
        
        isalpha()라는 함수에 대해서 알 수 있었다… 또한, 알파벳을 리스트에 담아 ‘’.join해줄 수도 있었다..
        
        ```python
        data = input()
        result = []
        value = 0
        
        # 문자를 하나씩 확인하며
        for x in data:
            # 알파벳인 경우 결과 리스트에 삽입
            if x.isalpha():
                result.append(x)
            # 숫자는 따로 더하기
            else:
                value += int(x)
        
        # 알파벳을 오름차순으로 정렬
        result.sort()
        
        # 숫자가 하나라도 존재하는 경우 가장 뒤에 삽입
        if value != 0:
            result.append(str(value))
        
        # 최종 결과 출력(리스트를 문자열로 변환하여 출력)
        print(''.join(result))
        ```
        
    3. 해결한 후
        
        새로운 함수를 알 수 있었고 쉬운 문제였다…

- 기출 문제 3: 문자열 압축

    1. 내 풀이
        
        - 처음 풀이
            1. 단위의 수를 증가시키면서 압축할 수 있는 최소의 길이를 구하려고 했다.
            
            하지만 어떻게 구현할지 감이 오지 않았다… 
            
        - 두번째 풀이
            1. 단위 크기를 증가시키면서 문자열을 단위 크기로 나눈 토큰들을 비교하며 압축길이 구함
            2. 하지만, 마지막 토큰 부분 처리에서 실패했다…
            
            풀이를 보니 마지막 토큰 부분은 압축 길이 계산 반복문을 탈출한 후에 처리해주면 되었다… 신중하게 문제 풀이 시도할 것…
            
            ```python
            s=input() # 문자열 입력
            length=len(s) # 문자열 길이
            unit=1 # 단위
            minimum=length
            
            while unit*2<=length:
                token=[s[i:i+unit] for i in range(0, len(s), unit)]
                result=0
                index=0
                while index<len(token)-1:
                    t=token[index]
                    substring=token[index+1:]
                    count=1
                    for sub in substring:
                        if token==sub:
                            count+=1
                        else:
                            break
                        
                    if count==1:
                        result+=len(t)
                    else:
                        result+=len(str(count))+len(t)
                    
                    index+=count
                    
                if result<length:
                    minimum=min(minimum, result)
                
                unit+=1
                            
            print(minimum)
            ```
        
    2. 풀이를 본 후
        
        단위를 증가시키며 진행하는 것은 맞았다. 하지만, 단위 크기별로 인덱스를 이용하여 뒤의 문자와 비교하는 부분이 핵심이었다…
        
        ```python
        def solution(s):
            answer = len(s)
            # 1개 단위(step)부터 압축 단위를 늘려가며 확인
            for step in range(1, len(s) // 2 + 1):
                compressed = ""
                prev = s[0:step] # 앞에서부터 step만큼의 문자열 추출
                count = 1
                # 단위(step) 크기만큼 증가시키며 이전 문자열과 비교
                for j in range(step, len(s), step):
                    # 이전 상태와 동일하다면 압축 횟수(count) 증가
                    if prev == s[j:j + step]:
                        count += 1
                    # 다른 문자열이 나왔다면(더 이상 압축하지 못하는 경우라면)
                    else:
                        compressed += str(count) + prev if count >= 2 else prev
                        prev = s[j:j + step] # 다시 상태 초기화
                        count = 1
                # 남아있는 문자열에 대해서 처리
                compressed += str(count) + prev if count >= 2 else prev
                # 만들어지는 압축 문자열이 가장 짧은 것이 정답
                answer = min(answer, len(compressed))
            return answer
        ```

- 기출 문제 4: 자물쇠와 열쇠

    1. 내 풀이
        
        - 처음 풀이
            1. 열쇠의 돌기 부분들 사이의 좌표 차이를 알아낸다.
            2. 그 후, 자물쇠의 구멍 부분 좌표를 탐색하며 열쇠가 자물쇠의 구멍에 들어갈 수 있으면 해제 가능
            
            하지만, 구현을 하다가 막혔다…

        - 두 번째 풀이
            1. 열쇠로 자물쇠를 탐색하기 위해 자물쇠 배열의 크기를 3배하고 가운데에 자물쇠 정보 입력
            2. 네 방향 마다 탐색 범위를 이동하면서 자물쇠의 홈을 다 채울 수 있는지 확인
            
            하지만 일부만 맞았다…
            
            >- **자물쇠의 홈이 다 채워졌는지 확인하는 부분에서 합으로 체크하면 안된다. 홈이 다 채워진다는 뜻은 모든 요소가 1이라는 것이지 합이 $$N \times N$$ 인 것이 아니다 (요소가 0, 1, 2로 합이 $$N \times N$$ 이면 맞다고 판단하는 경우가 발생)**
            >- **그리고, 굳이 3배 자물쇠 배열을 계속 만들지 않고 열쇠로 더한 부분을 빼주기만하면 초기에 만들어 놓고 재사용 가능, 탐색 범위도 구체적인 것도 좋지만 범용적인 범위가 사용 가능하면 사용, 2차원 배열을 90도 회전하는 방법 숙지**


            ```
            채점 결과
            정확성: 26.0
            합계: 26.0 / 100.0
            ```
            
            ```python
            def rotate(key):
                # 행과 열의 길이
                rows = len(key)
                cols = len(key[0])
                
                # 결과 배열 초기화
                rotated_matrix = [[0] * rows for _ in range(cols)]
                
                # 회전된 배열 생성
                for i in range(rows):
                    for j in range(cols):
                        rotated_matrix[j][rows - 1 - i] = key[i][j]
                
                return rotated_matrix
            
            def solution(key, lock):
                answer = False
                m=len(key)
                n=len(lock)
                for i in range(4): # 네 방향
                    for j in range(n-m+1, (2*n)+m-1): # 3배 확장 자물쇠 탐색
                        for k in range(n-m+1, (2*n)+m-1):
                            lock_3=[[0]*(3*n) for _ in range(3*n)] # 자물쇠 3배
                            for r in range(n):
                                for c in range(n):
                                    lock_3[r+n][c+n]=lock[r][c]
                
                            for r in range(m): # 이동 위치에 따라 열쇠의 값 더함
                                for c in range(m):
                                    lock_3[j+r][k+c]+=key[r][c]
                
                            sum=0
                            for r in range(n, 2*n): # 자물쇠 위치의 합 구하기
                                for c in range(n, 2*n):
                                    sum+=lock_3[r][c]
                
                            if sum==n**2: # 자물쇠의 모든 홈이 채워지면
                                answer=True
                                return answer
                
                    key=rotate(key) # 시계방향으로 90도 회전
                
                return answer
            ```
                
    2. 풀이를 본 후
        
        우선 탐색의 효율성을 위해서 자물쇠의 크기를 3배로 증가 시킨 후, 기존 자물쇠의 값을 가운데에 넣는다. 그 후, 4가지 방향에 대해서 열쇠를 이동 시키면서 자물쇠의 겹치는 부분과의 합을 구한다. 최종으로 가운데 부분 요소들이 모두 1일 때가 True이다…
        
        구현 문제에서는 데이터의 크기를 탐색 후 방향을 정해나간다… 그리고 항상 머리 속에 있는 것을 최대한 구현해 보는 것이 중요하다.
        
        ```python
        # 2차원 리스트 90도 회전하기
        def rotate_a_matrix_by_90_degree(a):
            n = len(a) # 행 길이 계산
            m = len(a[0]) # 열 길이 계산
            result = [[0] * n for _ in range(m)] # 결과 리스트
            for i in range(n):
                for j in range(m):
                    result[j][n - i - 1] = a[i][j]
            return result
        
        # 자물쇠의 중간 부분이 모두 1인지 확인
        def check(new_lock):
            lock_length = len(new_lock) // 3
            for i in range(lock_length, lock_length * 2):
                for j in range(lock_length, lock_length * 2):
                    if new_lock[i][j] != 1:
                        return False
            return True
        
        def solution(key, lock):
            n = len(lock)
            m = len(key)
            # 자물쇠의 크기를 기존의 3배로 변환
            new_lock = [[0] * (n * 3) for _ in range(n * 3)]
            # 새로운 자물쇠의 중앙 부분에 기존의 자물쇠 넣기
            for i in range(n):
                for j in range(n):
                    new_lock[i + n][j + n] = lock[i][j]
        
            # 4가지 방향에 대해서 확인
            for rotation in range(4):
                key = rotate_a_matrix_by_90_degree(key) # 열쇠 회전
                for x in range(n * 2):
                    for y in range(n * 2):
                        # 자물쇠에 열쇠를 끼워 넣기
                        for i in range(m):
                            for j in range(m):
                                new_lock[x + i][y + j] += key[i][j]
                        # 새로운 자물쇠에 열쇠가 정확히 들어 맞는지 검사
                        if check(new_lock) == True:
                            return True
                        # 자물쇠에서 열쇠를 다시 빼기
                        for i in range(m):
                            for j in range(m):
                                new_lock[x + i][y + j] -= key[i][j]
            return False
        ```

- 기출 문제 5: 뱀

    1. 내 풀이
        
        - 처음 풀이
            1. 우선 뱀의 머리, 꼬리, 현재 이동 방향 좌표 변수들을 정의
            2. 그 후, 시간이 지날 때마다 방향을 바꾸며 진행
            3. 진행하면서 벽을 만나거나 자기 자신과 부딪히면 종료
            
            하지만 틀렸다…
            
            ```python
            def change_direction(now_d,C):
                if C=='D':
                    if now_d[0]==0:
                        now_d[0],now_d[1]=now_d[1],now_d[0]
                    else:
                        now_d[0],now_d[1]=now_d[1],-now_d[0]
                else:
                    if now_d[0]==0:
                        now_d[0],now_d[1]=-now_d[1],now_d[0]
                    else:
                        now_d[0],now_d[1]=now_d[1],now_d[0]
                return now_d
            
            def solve():
                count=0
                now_d=[0,1]
                head,tail=[0,0],[0,0]
                
                for d in direction:
                    while count<=d[0]:
                        x,y=head[0]+now_d[0],head[1]+now_d[1]
                        if 0<=x<N and 0<=y<N:
                            if board[x][y]==2:
                                count+=1
                                return print(count)
                            elif board[x][y]==0:
                                board[tail[0]][tail[1]]=0
                                board[x][y]=2
                                head=[x,y]
                                tail[0]+=now_d[0]
                                tail[1]+=now_d[1]
                                count+=1
                            else:
                                board[x][y]=2
                                head=[x,y]
                                count+=1
                        else:
                            count+=1
                            return print(count)
                    now_d=change_direction(now_d,d[1])
                        
            N=int(input())
            board=[[0]*N for _ in range(N)]
            K=int(input())
            for i in range(K):
                r,c=map(int,input().split())
                board[r-1][c-1]=1
            board[0][0]=2
            L=int(input())
            direction=[]
            for i in range(L):
                X,C=map(str,input().split())
                direction.append((int(X),C))
                
            solve()
            ```
            
        - 두 번째 풀이
            1. 빈칸, 사과, 뱀을 각 0, 1, 2로 표현하고 방향 전환 시간 및 정보를 배열에 저장
            2. 뱀의 좌표를 저장할 배열을 이용하여 방향 전환 시간인지 확인하며 게임 진행
            3. 진행하면서 벽을 만나거나 자기 자신과 부딪히면 종료
            
            신중하게 구현하면 풀 수 있다.
            
            - **회전할 시간을 찾을 때 인덱스를 이용하여 증가 시키면서 시간 정보만 확인하면 더 빠른 시간 안에 구현될 것 같다. 또한, 방향 정보도 인덱스를 이용하면 조금 더 깔끔하게 구현 가능하다.**
            
            ```python
            from collections import deque
            
            def change_direction(dx, dy, play_time):
                for time, direction in directions:
                    if play_time==time: # 방향 전환 시간
                        if direction=='L': # 왼쪽으로
                            if dy==0:
                                dx, dy= dy, dx
                            else:
                                dx, dy= -dy, dx
                            
                        else: # 오른쪽으로
                            if dy==0:
                                dx, dy= dy, -dx
                            else:
                                dx, dy= dy, dx
                            
                return  dx, dy, play_time
            
            def solution():
                snake=deque([(0,0)])
                play_time=0 # 총 게임 시간
                x, y= 0, 0 # 머리 위치
                dx, dy=0, 1 # 방향 정보
                while True:
                    dx, dy, play_time=change_direction(dx, dy, play_time)
                    rx, ry= x+dx, y+dy
                    play_time+=1
                    
                    if rx<0 or rx>=n or ry<0 or ry>=n or board[rx][ry]==2: # 벽 만나거나 자기 몸과 부딪힐 때
                        break
                
                    if board[rx][ry]==1: # 사과 먹을 때
                        snake.append((rx, ry))
                        board[rx][ry]=2
                        x, y=rx, ry
                    else: # 빈 칸일때
                        snake.append((rx, ry))
                        board[rx][ry]=2
                        x, y=rx, ry
                        a, b=snake.popleft()
                        board[a][b]=0
                    
                return print(play_time)
            
            n=int(input())
            board=[[0]*n for _ in range(n)]
            board[0][0]=2 # 뱀의 초기 위치
            
            k=int(input())
            for i in range(k):
                r, c=map(int, input().split()) # 사과의 위치
                board[r-1][c-1]=1
            
            directions=[]
            L=int(input())
            for i in range(L):
                time, direction=input().split() # 시간, 방향 전환 정보
                directions.append((int(time), direction))
            
            solution()
            ```
        
    2. 풀이를 본 후
        
        >보드에서 뱀, 사과, 빈 공간이 차지하는 부분의 숫자 처리, 조건 처리 등 거의 유사했지만 마지막 처리를 하지 못했다… 마지막 방향 전환 이후에는 그냥 종료 조건까지 탐색 해야 하는데 그 조건을 넣지 않아 마지막 방향 전환 시간 이후에는 계속 종료가 되어버렸다…
        >또한, 방향 전환의 처리를 조금 더 깔끔하게 할 수 있었다…
        >**항상 조건을 끝까지 구현하는 것이 구현 문제의 핵심이다…**

        ```python
        n = int(input())
        k = int(input())
        data = [[0] * (n + 1) for _ in range(n + 1)] # 맵 정보
        info = [] # 방향 회전 정보

        # 맵 정보(사과 있는 곳은 1로 표시)
        for _ in range(k):
            a, b = map(int, input().split())
            data[a][b] = 1

        # 방향 회전 정보 입력
        l = int(input())
        for _ in range(l):
            x, c = input().split()
            info.append((int(x), c))

        # 처음에는 오른쪽을 보고 있으므로(동, 남, 서, 북)
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]

        def turn(direction, c):
            if c == "L":
                direction = (direction - 1) % 4
            else:
                direction = (direction + 1) % 4
            return direction

        def simulate():
            x, y = 1, 1 # 뱀의 머리 위치
            data[x][y] = 2 # 뱀이 존재하는 위치는 2로 표시
            direction = 0 # 처음에는 동쪽을 보고 있음
            time = 0 # 시작한 뒤에 지난 '초' 시간
            index = 0 # 다음에 회전할 정보
            q = [(x, y)] # 뱀이 차지하고 있는 위치 정보(꼬리가 앞쪽)

            while True:
                nx = x + dx[direction]
                ny = y + dy[direction]
                # 맵 범위 안에 있고, 뱀의 몸통이 없는 위치라면
                if 1 <= nx and nx <= n and 1 <= ny and ny <= n and data[nx][ny] != 2:
                    # 사과가 없다면 이동 후에 꼬리 제거
                    if data[nx][ny] == 0:
                        data[nx][ny] = 2
                        q.append((nx, ny))
                        px, py = q.pop(0)
                        data[px][py] = 0
                    # 사과가 있다면 이동 후에 꼬리 그대로 두기
                    if data[nx][ny] == 1:
                        data[nx][ny] = 2
                        q.append((nx, ny))
                # 벽이나 뱀의 몸통과 부딪혔다면
                else:
                    time += 1
                    break
                x, y = nx, ny # 다음 위치로 머리를 이동
                time += 1
                if index < l and time == info[index][0]: # 회전할 시간인 경우 회전
                    direction = turn(direction, info[index][1])
                    index += 1
            return time

        print(simulate())
        ```

- 기출 문제 6: 기둥과 보 설치
    
    1. 내 풀이
        
        - 처음 풀이
            1. 우선 설치 및 삭제 가능을 확인하는 check함수 생성
            2. build_frame 별로 check하며 작업이 가능할 때마다 기둥과 보 배열에 각각의 정보 저장
            
            하지만 삭제를 어떻게 구현할지 모르겠어서 실패했다…
            
        - 두 번째 풀이
            1. answer 리스트에 구조물의 좌표와 종류 저장
            2. 작업을 진행 후 check 함수를 통해서 가능한 작업인지 확인
            3. 작업이 가능하지 않다면 다시 작업 전의 상태로 돌림
            4. answer 오름차순 정렬 후 출력
            
            ```python
            def build_operation(x, y, a, b, answer):
                if b==1: # 설치
                    answer.append([x, y, a])
                else: # 제거
                    answer.remove([x, y, a])
                    
                return answer
            
            def check(answer):
                for x, y, a in answer:
                    if a==0: # 기둥
                        if y==0 or ([x,y,1] in answer) or ([x-1,y,1] in answer) or ([x,y-1,0] in answer):
                            continue
                        else:
                            return False
                    else: # 보
                        if ([x,y-1,0] in answer) or ([x+1,y-1,0] in answer) or (([x-1,y,1] in answer) and ([x+1,y,1] in answer)):
                            continue
                        else:
                            return False
                return True
            
            def solution(n, build_frame):
                answer = [] # 최종 답
                
                for x, y, a, b in build_frame:
                    answer=build_operation(x, y, a, b, answer) # 작업 진행        
                    
                    if check(answer): # 작업 가능
                        continue
                    else: # 작업 취소
                        if b==1: # 설치 취소
                            answer.remove([x, y, a])
                        else: # 삭제 취소
                            answer.append([x, y, a])
                
                answer.sort()
                
                return answer
            ```
    
            > **check 함수 구성을 생각하는데 조금 오래 걸렸다. 리스트에서 특정 요소를 찾을 때는 이와 같은 방식을 떠올릴 수 있도록 해야겠다**

        
        
    2. 풀이를 본 후
        
        >우선 굳이 기둥, 보의 좌표를 따로 저장할 필요 없이 그냥 answer 배열에 저장하면 되었다…
        >
        >**그리고, 삭제의 경우 삭제할 구조물과 인접해있는 요소를 생각하려고만 했다. 그런데 그게아니라 작업 진행 후 작업마다 전체 구조물의 상태를 보고 가능하면 작업 진행, 아니면 해당 작업 진행을 취소하면 된다…**
        >
        >부분적인 것만 보지말고 전체를 봐야겠다…
        
        ```python
        # 현재 설치된 구조물이 '가능한' 구조물인지 확인하는 함수
        def possible(answer):
            for x, y, stuff in answer:
                if stuff == 0: # 설치된 것이 '기둥'인 경우
                    # '바닥 위' 혹은 '보의 한쪽 끝 부분 위' 혹은 '다른 기둥 위'라면 정상
                    if y == 0 or [x - 1, y, 1] in answer or [x, y, 1] in answer or [x, y - 1, 0] in answer:
                        continue
                    return False # 아니라면 거짓(False) 반환
                elif stuff == 1: # 설치된 것이 '보'인 경우
                    # '한쪽 끝부분이 기둥 위' 혹은 '양쪽 끝부분이 다른 보와 동시에 연결'이라면 정상
                    if [x, y - 1, 0] in answer or [x + 1, y - 1, 0] in answer or ([x - 1, y, 1] in answer and [x + 1, y, 1] in answer):
                        continue
                    return False # 아니라면 거짓(False) 반환
            return True
        
        def solution(n, build_frame):
            answer = []
            for frame in build_frame: # 작업(frame)의 개수는 최대 1,000개
                x, y, stuff, operate = frame
                if operate == 0: # 삭제하는 경우
                    answer.remove([x, y, stuff]) # 일단 삭제를 해본 뒤에
                    if not possible(answer): # 가능한 구조물인지 확인
                        answer.append([x, y, stuff]) # 가능한 구조물이 아니라면 다시 설치
                if operate == 1: # 설치하는 경우
                    answer.append([x, y, stuff]) # 일단 설치를 해본 뒤에
                    if not possible(answer): # 가능한 구조물인지 확인
                        answer.remove([x, y, stuff]) # 가능한 구조물이 아니라면 다시 제거
            return sorted(answer) # 정렬된 결과를 반환
        ```

- 기출 문제 7: 치킨 배달
    
    
    1. 내 풀이
        
        - 처음 풀이
            1. 우선 치킨집과 집 사이의 거리를 나타내는 이차원 배열 생성
            2. 그 후, 행의 합을 기준으로 오름차순 정렬 후, M개의 행을 선택
            3. 그리고 각 열의 값중 최소값을 다 더해서 조시의 치킨거리 최소값 도출
            
            그런데 틀렸다…
            
            ```python
            def solve():
                answer = 0
                sorted_distance = sorted(distance, key=lambda row: sum(row))
                selected_rows = sorted_distance[:M]
                for col in range(len(house)):
                    col_min = min(row[col] for row in selected_rows)
                    answer += col_min
                
                return print(answer)
            
            N, M = map(int, input().split())
            city = [list(map(int, input().split())) for _ in range(N)]
            house, chicken = [], []
            for i in range(N):
                for j in range(N):
                    if city[i][j] == 1:
                        house.append((i, j))
                    if city[i][j] == 2:
                        chicken.append((i, j))
            
            distance=[]
            for c in chicken:
                temp=[]
                for h in house:
                    temp.append(abs(c[0]-h[0])+abs(c[1]-h[1]))
                distance.append(temp)
            
            solve()
            ```
            
        - 두 번째 풀이
            1. 집, 치킨의 좌표들 저장
            2. 치킨 집 m개의 조합마다 집들의 도시의 치킨 거리를 구해서 도시의 치킨 거릐의 최솟값 구함
            
            ```python
            from itertools import combinations
            
            def solution():
                chicken_combination = list(combinations(chicken, m)) # 치킨집 m개 경우의 수
                result=1e9 # 최종 최소 도시치킨거리
                for chicken_comb in chicken_combination:
                    sum=0 # 각 경우의 도시치킨거리
                    for h in house:
                    temp=1e9 # 집마다의 치킨거리
                    for chicken_co in chicken_comb:
                        r1, c1 = h
                        r2, c2 = chicken_co
                        temp=min(temp,abs(r1-r2)+abs(c1-c2))
                        
                    sum+=temp  
                
                    result=min(result, sum)
                
                return print(result)
            
            n,m=map(int, input().split()) 
            city=[list(map(int, input().split())) for _ in range(n)] # 도시 정보
            house, chicken=[], [] # 집과 치킨집 좌표
            for i in range(n):
                for j in range(n):
                    if city[i][j]==1:
                        house.append((i,j))
                    elif city[i][j]==2:
                        chicken.append((i,j))
            
            solution()
            ```
        
    2. 풀이를 본 후
        
        >처음에 조합 라이브러리 사용을 생각했었지만 사용하지 않고 해결해보고 싶었다…
        >
        >**우선 나의 풀이에는 오류가 있었다… 행의 합이 작은 순서대로 나열하는 점이 오류였다…**
        >**행의 합이 작다고 열의 요소 하나하나의 최소 값들의 조합이 작지 않은 경우를 생각 못했다…**
        >
        >풀이는 조합 라이브러리를 사용해서 간단하게 해결했다…
        
        ```python
        from itertools import combinations
        
        n, m = map(int, input().split())
        chicken, house = [], []
        
        for r in range(n):
            data = list(map(int, input().split()))
            for c in range(n):
                if data[c] == 1:
                    house.append((r, c)) # 일반 집
                elif data[c] == 2:
                    chicken.append((r, c)) # 치킨집
        
        # 모든 치킨 집 중에서 m개의 치킨 집을 뽑는 조합 계산
        candidates = list(combinations(chicken, m))
        
        # 치킨 거리의 합을 계산하는 함수
        def get_sum(candidate):
            result = 0
            # 모든 집에 대하여
            for hx, hy in house:
                # 가장 가까운 치킨 집을 찾기
                temp = 1e9
                for cx, cy in candidate:
                    temp = min(temp, abs(hx - cx) + abs(hy - cy))
                # 가장 가까운 치킨 집까지의 거리를 더하기
                result += temp
            # 치킨 거리의 합 반환
            return result
        
        # 치킨 거리의 합의 최소를 찾아 출력
        result = 1e9
        for candidate in candidates:
            result = min(result, get_sum(candidate))
        
        print(result)
        ```
        
        백트래킹으로 해결하는 풀이도 있었다…
        [https://aigong.tistory.com/467](https://aigong.tistory.com/467)

- 기출 문제 8: 외벽 점검

1. 내 풀이
    
    - 처음 풀이
        1. 친구 수의 값을 증가 시키면서 weak값을 기준으로 탐색 범위를 정한다.
        2. 모든 weak값을 다 탐색할 수 있으면 최소의 친구 수이다.
        
        구현을 어떻게 할지 모르겠다…
        
    - 두 번째 풀이
        1. 원형 문제라 좌표를 두 배로 증가 시키고 일직선 상으로 생각
        2. 친구의 수를 증가 시키며 탐색
        
        하지만 친구의 배치와 구간 설정을 어떻게 할지 모르겠어서 실패했다.
        
        **순열로 친구들의 배치를 생각하고, 인덱스를 이용하여 구간을 확인하며 점검 가능 유무에 따라 친구의 수를 증가시키는게 중요했다.**
    
2. 풀이를 본 후
    - 원형 문제에서는 길이를 두 배로 늘려서 원형을 일자 형태로 바꿔서 해결
    - 순열을 이용해서 친구 나열 경우의 수를 완전 탐색하여 배치할 수 있는 최소 친구 수 구함
    
    **원형 문제 풀이 법을 알아 둬야 할 것 같다…**
    
    ```python
    from itertools import permutations
    
    def solution(n, weak, dist):
        # 길이를 2배로 늘려서 '원형'을 일자 형태로 변형
        length = len(weak)
        for i in range(length):
            weak.append(weak[i] + n)
        answer = len(dist) + 1 # 투입할 친구 수의 최솟값을 찾아야 하므로 len(dist) + 1로 초기화
        # 0부터 length - 1까지의 위치를 각각 시작점으로 설정
        for start in range(length):
            # 친구를 나열하는 모든 경우 각각에 대하여 확인
            for friends in list(permutations(dist, len(dist))):
                count = 1 # 투입할 친구의 수
                # 해당 친구가 점검할 수 있는 마지막 위치
                position = weak[start] + friends[count - 1]
                # 시작점부터 모든 취약한 지점을 확인
                for index in range(start, start + length):
                    # 점검할 수 있는 위치를 벗어나는 경우
                    if position < weak[index]:
                        count += 1 # 새로운 친구를 투입
                        if count > len(dist): # 더 투입이 불가능하다면 종료
                            break
                        position = weak[index] + friends[count - 1]
                answer = min(answer, count) # 최솟값 계산
        if answer > len(dist):
            return -1
        return answer
    ```

## 출처

- 이것이 취업을 위한 코딩테스트다 with 파이썬