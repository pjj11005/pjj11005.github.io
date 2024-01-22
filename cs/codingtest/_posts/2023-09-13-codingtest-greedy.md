---
layout: post
title: 이것이 취업을 위한 코딩테스트다 | 1. 그리디 알고리즘 (Greedy)
categories: 
  - cs
  - codingtest
description: 이것이 취업을 위한 코딩테스트다 with 파이썬 책을 참고하여 **그리디 알고리즘**을 공부한 내용입니다.
sitemap: false
---

그리디 알고리즘은 단순하지만 강력한 문제 해결 방법이다.

* this unordered seed list will be replaced by the toc
{:toc}

## 정의 
> **`현재의 상황에서 지금 당장 좋은 것만 고르는 방법`**
    
## 문제 유형

- 사전에 외우고 있지 않아도 해결할 가능성이 큰 문제 유형

- 정렬, 최단 거리 등의 알고리즘 유형과 반대 (정확히 알고 있어야 해결 가능)

- 문제 해결 시 최소한의 아이디어 요구 → 단순히 현재 상황에서 가장 좋아 보이는 것 만을 선택해도 문제를 풀 수 있는지 파악해야 한다.

- `‘가장 큰 순서대로’, ‘가장 작은 순서대로’` 와 같은 기준 알게 모르게 제시 → 대체로 정렬 알고리즘 사용 시 이 기준을 만족하여 정렬 알고리즘과 자주 짝을 이뤄 출제
    
## 대표 예제
    
거스름돈 → 거슬러 줘야 할 동전의 최소 개수 (500원, 100원, 50원, 10원)
    
> - 대표적인 그리디 문제
> - 해결 방법: ‘가장 큰 화폐 단위부터’ 돈을 거슬러 준다.
    
## 그리디 알고리즘의 정당성
- 대부분의 문제 그리디 알고리즘 이용 시 `‘최적의 해’`를 찾을 수 없음 → 탐욕적 접근시 정확한 답을 찾을 수 있다는 보장이 있으면 매우 효과적

- 따라서, `문제 풀이를 위한 최소한의 아이디어를 떠올리고 이것이 정당한지 검토할 수 있어야` 답 도출 가능

- 어떤 문제 해결 시 문제 유형 파악 어렵다면 `그리디로 먼저 접근` → 해결 방법 찾기 어려우면 `다이나믹 프로그래밍, 그래프 알고리즘` 등으로 문제 해결 고민

> 거스름돈 예제: `큰 단위가 작은 단위의 배수 형태` 이므로 `그리디 접근 시 항상 최적의 해를 보장` (작은 단위의 동전들을 종합해서 다른 해가 나올 수 없다)
>
> `동전의 단위가 무작위 형태` → `다이나믹 프로그래밍`을 이용하여 해결

## 예제 풀이
- 예제 1: 큰 수의 법칙

    1. 내 풀이
        
        1. 우선 주어진 수에서 가장 큰 수와 두번 째로 큰 수만 더해진다는 사실을 확인했다.
        
        1. 이후 K개만큼 가장 큰수를 더하고 그 후에 두번 째로 큰 수를 더해주며 번갈아가면서 행하도록 구현했다.
        2. 총 개수가 M개가 되면 종료한다. 
        
        ```python
        N,M,K=map(int,input().split())
        number=list(map(int,input().split()))
        number.sort(reverse=True)
        
        first,second=number[0],number[1]
        count,temp, sum=0,0,0
        
        while count<M:
            if temp<K:
                sum+=first
                temp+=1
                count+=1
            else:
                sum+=second
                temp=0
                count+=1
        
        print(sum)
        ```
        
    2. 풀이를 본 후
        
        반복되는 수열에 대해서 파악하는 것이 핵심이었다. 그 후 식을 이용해서 횟수를 구하면 쉽게 해결할 수 있다.
        
        ```python
        N, M, K를 공백을 기준으로 구분하여 입력 받기
        n, m, k = map(int, input().split())
        N개의 수를 공백을 기준으로 구분하여 입력 받기
        data = list(map(int, input().split()))
        
        data.sort() # 입력 받은 수들 정렬하기
        first = data[n - 1] # 가장 큰 수
        second = data[n - 2] # 두 번째로 큰 수
        
        #가장 큰 수가 더해지는 횟수 계산
        count = int(m / (k + 1)) * k
        count += m % (k + 1)
        
        result = 0
        result += (count) * first # 가장 큰 수 더하기
        result += (m - count) * second # 두 번째로 큰 수 더하기
        
        print(result) # 최종 답안 출력
        ```
        
    3. 해결한 후
        
        조금만 더 생각했으면 식을 유도했을 것 같다. 문제 해결 시 생각된 해결 방법이 있으면 끝까지 생각해서 적용 시키도록 노력해야겠다. 
    

- 예제 2: 숫자 카드 게임

    1. 내 풀이
        
        1. 우선 행마다 가장 작은수를 구해준다.
        
        1. 그 후, 계속해서 최종적으로 구할 가장 큰 수를 갱신하면서 답을 구한다.
        
        ```python
        N,M=map(int,input().split())
        card=[list(map(int,input().split())) for _ in range(N)]
        maximum=0
        
        for i in range(N):
            maximum=max(maximum,min(card[i]))
        
        print(maximum)
        ```
        
    2. 풀이를 본 후
        
        **각 행마다 가장 작은 수를 찾은 뒤에 그 수 중에서 가장 큰 수**를 구하면 되는 문제였다.
        
        ```python
        # N, M을 공백을 기준으로 구분하여 입력 받기
        n, m = map(int, input().split())
        
        result = 0
        # 한 줄씩 입력 받아 확인하기
        for i in range(n):
            data = list(map(int, input().split()))
            # 현재 줄에서 '가장 작은 수' 찾기
            min_value = 10001
            for a in data:
                min_value = min(min_value, a)
            # '가장 작은 수'들 중에서 가장 큰 수 찾기
            result = max(result, min_value)
        
        print(result) # 최종 답안 출력
        ```
        
    3. 해결한 후
        
        조금만 생각하면 바로 풀 수 있는 문제였다.
    

- 예제 3: 1이 될 때까지

    1. 내 풀이
        
        1. 우선 N을 K로 최대한 많이 나누는 것이 연산의 횟수를 최소로 만들 수 있다.
        
        1. 따라서, 우선적으로 K로 나누어 떨어질 때는 나누어주고 아니면 -1을 해주도록 했다.
        
        ```python
        N,K=map(int,input().split())
        count=0
        
        while N>1:
        if N%K==0:
            N/=K
            count+=1
        else:
            N-=1
            count+=1
            
        print(count)
        ```
        
    2. 풀이를 본 후
        
        내 풀이와 같은 아이디어로 해결하는 것이 맞았다. 하지만 더 명확하게 나누어 떨어지는 수가 될 때까지 1을 빼준다. 그 후, 나누기를 하고 N이 K보다 작아지면 나눌 수가 없기 때문에 1을 빼주는 연산을 한다. 이렇게 좀더 명확하게 나눠야 했다…
        
        ```python
        # N, K공백을 기준으로 구분하여 입력 받기
        n, k = map(int, input().split())
        
        result = 0
        
        while True:
            # N이 K로 나누어 떨어지는 수가 될 때까지만 1씩 빼기
            target = (n // k) * k
            result += (n - target)
            n = target
            # N이 K보다 작을 때 (더 이상 나눌 수 없을 때) 반복문 탈출
            if n < k:
                break
            # K로 나누기
            result += 1
            n //= k
        
        # 마지막으로 남은 수에 대하여 1씩 빼기
        result += (n - 1)
        print(result)
        ```
        
    3. 해결한 후
        
        아이디어는 거의 맞았지만 구현에서 너무 대충 구현하는 것 같다… 최대한 일반적으로 적용 가능하게 구현해야겠다…
    

## 기출문제
- 기출 문제 1: 모험가 길드

    1. 내 풀이
        
        1. 우선 내림차순으로 정렬을 한 후 팀을 구해간다.
        2. 그 후, 맨 앞의 수 다음부터 다시 시작하면서 구성 가능한 최대 팀수를 구한다.
        3. 최대 팀 수가 남은 모험가 수 이상이면 종료한다.
        
        하지만 구현을 하다가 실패했다…
        
        ```python
        N=int(input())
        gongpo=list(map(int,input().split()))
        gongpo.sort(reverse=True)
        maximum,left,right=0,0,0
        temp=0
        while right<N:
            if right+gongpo[right]<N:
                right+=gongpo[right]
            
        
        print(maximum)
        ```
        
    2. 풀이를 본 후
        
        오름차순으로 정렬한 후 현재의 팀을 계속해서 결성하면서 진행했다… **오름 차순으로 작은 수부터 진행해야 넣지 않아도 되는 모험가를 거를 수 있다…**
        
        너무 어렵게만 생각한것 같다… 항상 그리디 문제는 현재 상황에서 최적의 수를 생각하면서 진행해야한다…
        
        **스택을 이용한 풀이도 있다 - 혜원**
        
        >**내림차순으로 구현 시**
        >
        >N=6이고 공포도가 4 3 3 3 3 2 인 경우
        >
        >`내림차순 정렬 4 3 3 3 / 3 2 -> result는 1인데`
        >
        >`오름차순 정렬 2 3 / 3 3 3 / 4 -> result가 2여야 하므로 오류 ! → 오름차순 정렬로 풀어야 함 `

        ```python
        n = int(input())
        data = list(map(int, input().split()))
        data.sort()
        
        result = 0 # 총 그룹의 수
        count = 0 # 현재 그룹에 포함된 모험가의 수
        
        for i in data: # 공포도를 낮은 것부터 하나씩 확인하며
            count += 1 # 현재 그룹에 해당 모험가를 포함시키기
            if count >= i: # 현재 그룹에 포함된 모험가의 수가 현재의 공포도 이상이라면, 그룹 결성
                result += 1 # 총 그룹의 수 증가시키기
                count = 0 # 현재 그룹에 포함된 모험가의 수 초기화
        
        print(result) # 총 그룹의 수 출력
        ```
    

- 기출 문제 2: 곱하기 혹은 더하기

    1. 내 풀이
        
        - 처음 풀이
            1. 우선 maximum변수에 제일 앞에 있는 수를 저장한다.
            2. 그리고 다음 수를 차례대로 곱하기 혹은 더하기 연산 진행 후 더 큰 값을 maximum으로 넣어준다.
            
            ```python
            # 처음 풀이
            S=list(str(input()))
            maximum=int(S[0])
            
            for i in range(1,len(S)):
                maximum=max(maximum+int(S[i]),maximum*int(S[i]))
            
            print(maximum)
            ```
        
        - 두번째 풀이
            1. 연산 시 둘 중 하나라도 0이나 1이면 더하기, 나머지 경우는 곱하기 진행
        
            ```python
            # 두번째 풀이
            s=input()
            result=0
            first=int(s[0]) # 연산 앞부분 
            
            for i in range(1,len(s)):
                second=int(s[i]) # 연산 뒷부분
                if 0<=first<=1 or 0<=second<=1: # 둘 중 하나라도 0이나1일 떄 (더하기 연산)
                    first+=second
                else: # 곱하기 연산
                    first*=second
            
            print(first)
            ```

    2. 풀이를 본 후
        
        우선, 풀이를 보니 연산할 두 개의 숫자 중에서 하나라도 0 혹은 1이면 더하기를 진행하도록 했다. 또한, 굳이 문자열 형태로 받지 않아도 인덱스로 구분 가능했다.
        
        생각해보니 0혹은 1이면 더하기를 한 값이 더 크게 된다… 
        
        ```python
        data = input()
        
        # 첫 번째 문자를 숫자로 변경하여 대입
        result = int(data[0])
        
        for i in range(1, len(data)):
            # 두 수 중에서 하나라도 '0' 혹은 '1'인 경우, 곱하기보다는 더하기 수행
            num = int(data[i])
            if num <= 1 or result <= 1:
                result += num
            else:
                result *= num
        
        print(result)
        ```
        
    3. 해결한 후
        
        아마 나의 풀이로 풀면 시간 초과가 발생할 가능성이 크다… 문제 해결 시 더욱 최적의 해결방법을 찾도록 해야한다.
    

- 기출 문제 3: 문자열 뒤집기
    
    [https://www.acmicpc.net/problem/1439](https://www.acmicpc.net/problem/1439)
    
    1. 내 풀이
        
        - 처음 풀이
            1. 우선, 0의 덩어리 수, 1의 덩어리 수를 각각  zero, one으로 정의했다.
            2. 맨 처음 숫자부터 시작해서 끝자리까지 서로 비교하며 0과 1의 덩어리수를 구해줬다.
            3. 최종적으로 더 적은 덩어리 수를 가진 값을 출력하게했다.
            
            ```python
            S=input()
            zero,one=0,0
            
            temp=S[0]
            if temp=='0':
                zero+=1
            else:
                one+=1
            
            for i in range(1,len(S)):
                if S[i]!=temp:
                    if S[i]=='0':
                        temp=S[i]
                        one+=1
                    else:
                        temp=S[i]
                        zero+=1
            
            print(min(one,zero))
            ```
        
        - 두번째 풀이

            1. s의 0과 1 덩어리의 수를 구한 후, 최소 덩어리의 수를 출력
                
                더 작은 덩어리를 뒤집는게 최소 행동 횟수가 된다.
                

            ```python
            s=input()
            zero, one=0, 0 # s에서 0 or 1 덩어리 개수
            first=int(s[0]) # 처음 숫자

            if first==0:
                zero+=1
            else:
                one+=1

            for i in range(1,len(s)):
                second=int(s[i]) # 비교할 숫자
                if first==second: # 같을 때
                    continue
                else: # 다를 때
                    if second==0:
                        zero+=1
                    else:
                        one+=1
                        first=second

            print(min(zero, one))
            ```

    2. 풀이를 본 후
        
        풀이도 거의 유사했다. 풀이는 0으로 바꾸는 경우, 1로 바꾸는 경우로 변수를 설정하고 진행했다.
        
        그리고 인덱스 순서대로 비교하면서 수를 더해갔다.
        
        ```python
        data = input()
        count0 = 0 # 전부 0으로 바꾸는 경우
        count1 = 0 # 전부 1로 바꾸는 경우
        
        # 첫 번째 원소에 대해서 처리
        if data[0] == '1':
            count0 += 1
        else:
            count1 += 1
        
        # 두 번째 원소부터 모든 원소를 확인하며
        for i in range(len(data) - 1):
            if data[i] != data[i + 1]:
                # 다음 수에서 1로 바뀌는 경우
                if data[i + 1] == '1':
                    count0 += 1
                # 다음 수에서 0으로 바뀌는 경우
                else:
                    count1 += 1
        
        print(min(count0, count1))
        ```
        
    3. 해결한 후
        
        연속된 수를 처리해 나갈 때는 인덱스를 이용해서 구현하는 것이 더 좋다.
        

- 기출 문제 4: 만들 수 없는 금액

    1. 내 풀이 
        
        1. 우선 동전들 중에서 최소가 1원보다 크면 무조건 정답은 1이된다.
        
        1. 하지만 최소 동전이 1원이면 이제 만들 수 없는 동전 중 최소를 구해야한다.
        
        동전을 조합하는 것을 어떻게 구현할지 감을 못잡았다…
        
        ```python
        N=int(input())
        coin=list(map(int,input().split()))
        
        if min(coin)>1:
        print(1)
        else:
        coin.sort()
        minimum=1
        for i in coin:
            if i==minimum:
            minimum+=1
        ```
        
    2. 풀이를 본 후
        
        단순하게 타겟을 정해 놓고 작은 단위의 동전부터 차례로 더하면서 현재의 동전보다 작으면 그 값이 정답이 된다… 동전을 조합하는 부분은 작은 동전을 차례로 더하면 자동으로 점점 더해지므로 값을 탐색할 수 있게된다.. 
        
        이러한 해결방법을 만들 수 있게 노력해야겠다…
        
        ```python
        n = int(input())
        data = list(map(int, input().split()))
        data.sort()
        
        target = 1
        for x in data:
            # 만들 수 없는 금액을 찾았을 때 반복 종료
            if target < x:
                break
            target += x
        
        # 만들 수 없는 금액 출력
        print(target)
        ```
    

- 기출 문제 5: 볼링공 고르기

    1. 내 풀이
        
        1. 단순하게 배열의 첫번째 요소부터 다른 무게의 볼링공과 짝지을 때만 개수를 세주었다…
        
        하지만, 시간복잡도가 너무 높아 조금 더 쉽게 계산 하는법을 찾다가 실패했다.
        
        ```python
        N,M= map(int, input().split())
        ball = list(map(int, input().split()))
        result=0
        
        for i in range(N-1):
        for j in range(i+1,N):
            if ball[i]!=ball[j]:
            result+=1
        
        print(result)
        ```
        
    2. 풀이를 본 후
        
        우선 각 무게별 볼링 공의 개수를 세서 배열에 저장한다. 그 후, 가벼운 볼링공부터 짝지을 수 있는 개수를 세주면서 카운트 해준다.
        
        ex) A가 선택하는 무게 기준 B가 선택할 무게를 카운트 → A의 선택하는 무게를 높이며 반복문 진행
        
        이 방법이 훨씬 빠르게 진행할 수 있다… 시간 복잡도를 항상 좋게할 수 있게 노력해야한다…
        
        ```python
        n, m = map(int, input().split())
        data = list(map(int, input().split()))
        
        # 1부터 10까지의 무게를 담을 수 있는 리스트
        array = [0] * 11
        
        for x in data:
            # 각 무게에 해당하는 볼링공의 개수 카운트
            array[x] += 1
        
        result = 0
        # 1부터 m까지의 각 무게에 대하여 처리
        for i in range(1, m + 1):
            n -= array[i] # 무게가 i인 볼링공의 개수(A가 선택할 수 있는 개수) 제외
            result += array[i] * n # B가 선택하는 경우의 수와 곱해주기
        
        print(result)
        ```
    

- 기출 문제 6: 무지의 먹방 라이브

    1. 내 풀이
        
        1. 우선 k를 감소 및 인덱스를 증가 시키면서 음식을 먹게 진행 시켰다.
        
        1. 그리고 k=-1일 때, 중단 시키고 먹어야 할 음식의 번호를 출력했다…
        
        하지만 런타임 에러가 발생했다…
        
        ```python
        def solution(food_times, k):
        answer = 0
        n=len(food_times)
        
        while k>-1:
            if sum(food_times)==0:
            return -1
            
            if food_times[answer]==0:
            if answer==n-1:
                answer=0
            else:
                answer+=1
            else:
            if answer==n-1:
                k-=1
                food_times[answer]-=1
                answer=0
            else:
                k-=1
                food_times[answer]-=1
                answer+=1
        
        if answer==0:
            answer=n
            
        return answer
        
        food_times = list(map(int, input().split()))
        k=int(input())
        result=solution(food_times,k)
        print(result)
        ```
        
    2. 풀이를 본 후
        
        우선 먼저 먹을 음식을 처리하면서 우선순위 큐를 이용하는 방법을 이용했다. 이 후, 우선순위에 따라 남은 시간을 차감하면서 중단 시간 이후 먹을 음식의 번호를 출력한다…
        
        솔직히 이런 식의 구현은 생각지도 못했다… 방법을 숙지하고 다음에 활용할 수 있도록 해야겠다…
        
        ```python
        import heapq
        
        def solution(food_times, k):
            # 전체 음식을 먹는 시간보다 k가 크거나 같다면 -1
            if sum(food_times) <= k:
                return -1
        
            # 시간이 작은 음식부터 빼야 하므로 우선순위 큐를 이용
            q = []
            for i in range(len(food_times)):
                # (음식 시간, 음식 번호) 형태로 우선순위 큐에 삽입
                heapq.heappush(q, (food_times[i], i + 1))  
        
            sum_value = 0 # 먹기 위해 사용한 시간
            previous = 0 # 직전에 다 먹은 음식 시간
            length = len(food_times) # 남은 음식의 개수
        
            # sum_value + (현재의 음식 시간 - 이전 음식 시간) * 현재 음식 개수와 k 비교
            while sum_value + ((q[0][0] - previous) * length) <= k:
                now = heapq.heappop(q)[0]
                sum_value += (now - previous) * length
                length -= 1 # 다 먹은 음식 제외
                previous = now # 이전 음식 시간 재설정
        
            # 남은 음식 중에서 몇 번째 음식인지 확인하여 출력
            result = sorted(q, key=lambda x: x[1]) # 음식의 번호 기준으로 정렬
            return result[(k - sum_value) % length][1]
        ```

## 출처

- 이것이 취업을 위한 코딩테스트다 with 파이썬