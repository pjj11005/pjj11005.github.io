---
layout: post
title: 이것이 취업을 위한 코딩테스트다 | 7. 최단 경로 (Shortest Path)
categories: 
  - cs
  - codingtest
description: 이것이 취업을 위한 코딩테스트다 with 파이썬 책을 참고하여 **최단 경로**을 공부한 내용입니다.
sitemap: false
---

특정 지점까지 가장 빠르게 도달하는 방법을 찾는 알고리즘

* this unordered seed list will be replaced by the toc
{:toc}

## 가장 빠르게 도달하는 방법

- 단순히 최단 거리를 출력하는 문제가 많이 출제
- 다익스트라, 플로이드 워셜 알고리즘 → 코테 최단 경로 문제는 쉽게 해결
- 그리디, 다이나믹 프로그래밍 알고리즘들이 최단 경로에 그대로 적용 → 그리디, DP의 한 유형

## 다익스트라 최단 경로 알고리즘

- 특정 노드에서 출발하여 다른 노드로 가는 각각의 최단 경로를 구해주는 알고리즘
- 음의 간선이 없을 때 사용 → GPS의 기본 알고리즘으로 채택
- 그리디 알고리즘으로 분류 → 매번 ‘가장 적은 비용’의 간선 선택해서 임의의 과정 반복
- 동작 원리
    1. 출발 노드 설정
    2. 최단 거리 테이블 초기화 → 다른 모든 노드로 가는 거리 int(1e9)로 초기화
    3. 방문하지 않은 노드 중에서 최단거리가 가장 짧은 노드 선택
    4. 해당 노드를 거쳐 다른 노드로 가는 비용 계산하여 최단 거리 테이블 갱신
    5. 3, 4번 반복
- 구현 방법
    1. 구현은 쉽지만 동작은 느린 코드
    2. 구현은 까다롭지만 동작은 빠른 코드
    - 시험 준비를 위해서는 방법 2를 정확히 이해하고 구현할 수 있도록 연습 필요
- **한 단계당 하나의 노드에 대한 최단 거리를 확실히 찾는다**
- 방법 1: 간단한 다익스트라 알고리즘
    - 시간 복잡도: $$O(V^2)$$ → V는 노드의 개수
    - 동작 원리
        1. 각 노드에 대한 최단 거리를 담는 1차원 리스트 선언
        2. 단계마다 ‘방문하지 않은 노드 중에서 최단 거리가 가장 짧은 노드 선택’하기 위해 매 단계마다 1차원 리스트의 모든 원소 순차 탐색
    - 전체 노드의 개수가 5000개 이하이면 해결 가능 → 하지만 노드의 개수가 10000개를 넘어가는 문제이면 개선된 다익스트라 알고리즘 이용해야함

    ```python
    import sys
    input = sys.stdin.readline
    INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

    # 노드의 개수, 간선의 개수를 입력받기
    n, m = map(int, input().split())
    # 시작 노드 번호를 입력받기
    start = int(input())
    # 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
    graph = [[] for i in range(n + 1)]
    # 방문한 적이 있는지 체크하는 목적의 리스트를 만들기
    visited = [False] * (n + 1)
    # 최단 거리 테이블을 모두 무한으로 초기화
    distance = [INF] * (n + 1)

    # 모든 간선 정보를 입력받기
    for _ in range(m):
        a, b, c = map(int, input().split())
        # a번 노드에서 b번 노드로 가는 비용이 c라는 의미
        graph[a].append((b, c))

    # 방문하지 않은 노드 중에서, 가장 최단 거리가 짧은 노드의 번호를 반환
    def get_smallest_node():
        min_value = INF
        index = 0 # 가장 최단 거리가 짧은 노드(인덱스)
        for i in range(1, n + 1):
            if distance[i] < min_value and not visited[i]:
                min_value = distance[i]
                index = i
        return index

    def dijkstra(start):
        # 시작 노드에 대해서 초기화
        distance[start] = 0
        visited[start] = True
        for j in graph[start]:
            distance[j[0]] = j[1]
        # 시작 노드를 제외한 전체 n - 1개의 노드에 대해 반복
        for i in range(n - 1):
            # 현재 최단 거리가 가장 짧은 노드를 꺼내서, 방문 처리
            now = get_smallest_node()
            visited[now] = True
            # 현재 노드와 연결된 다른 노드를 확인
            for j in graph[now]:
                cost = distance[now] + j[1]
                # 현재 노드를 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우
                if cost < distance[j[0]]:
                    distance[j[0]] = cost

    # 다익스트라 알고리즘을 수행
    dijkstra(start)

    # 모든 노드로 가기 위한 최단 거리를 출력
    for i in range(1, n + 1):
        # 도달할 수 없는 경우, 무한(INFINITY)이라고 출력
        if distance[i] == INF:
            print("INFINITY")
        # 도달할 수 있는 경우 거리를 출력
        else:
            print(distance[i])
    ```
- 방법 2: 개선된 다익스트라 알고리즘
    - 시간 복잡도: $$O(ElogV)$$ → V는 노드의 개수, E는 간선의 개수
    - 특정 노드까지의 최단 거리 정보→ 힙 자료구조 이용하여 빠르게 탐색, 로그 시간이 걸림
    - 힙 설명
        - 우선순위 큐를 구현하기 위한 자료구조
        - 우선순위 큐: **우선순위가 가장 높은 데이터를 가장 먼저 삭제**
            
            
            | 자료구조 | 추출되는 데이터 |
            | --- | --- |
            | 스택(Stack) | 가장 나중에 삽입된 데이터 |
            | 큐(Queue) | 가장 먼저 삽입된 데이터 |
            | 우선순위 큐(Priority Queue) | 가장 우선순위가 높은 데이터 |

            - 데이터를 우선순위에 따라 처리하고 싶을 때 사용
            - 우선순위 큐 → heapq 라이브러리 사용 권장
            - 값은 일반적으로 정수형 변수 사용 → 일반적으로 첫번째 값을 기준으로 우선순위 설정
            - 최소 힙: 값이 낮은 데이터 먼저 삭제, 최대 힙: 값이 높은 데이터 먼저 삭제
                - 파이썬 우선순위 라이브러리 기본적으로 최소힙 사용 → 다익스트라에서는 그래도 사용하면 적합
                - 최소 힙 최대 힙처럼 사용: (-)부호를 붙여서 꺼낸 다음 다시 원래 부호로 돌려 놓음
            - 구현 방식
                
                
                | 우선순위 큐 구현 방식 | 삽입 시간 | 삭제 시간 |
                | --- | --- | --- |
                | 리스트 | $$O(1)$$ | $$O(N)$$ |
                | 힙(Heap) | $$O(logN)$$ | $$O(logN)$$ |

                - 데이터 모두 넣고 빼는 작업의 시간 복잡도
                    - 힙: $$O(NlogN)$$
                    - 리스트: $$O(N^2)$$
    - 동작 원리
        1. 방법 1과 같이 1차원 최단거리 테이블 사용
        2. 현재 가장 가까운 노드 저장을 위해 우선순위 큐만 추가로 이용하여 진행
        - heapq 라이브러리 첫번째 원소 기준으로 우선순위 큐 구성 → (거리, 노드 번호) 순서대로 튜플 데이터 넣으면 됨
        - 최단 거리 가장 짧은 노드 선택하는 과정 → 우선순위 큐 이용하여 대체

    ```python
    import heapq
    import sys
    input = sys.stdin.readline
    INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

    # 노드의 개수, 간선의 개수를 입력받기
    n, m = map(int, input().split())
    # 시작 노드 번호를 입력받기
    start = int(input())
    # 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
    graph = [[] for i in range(n + 1)]
    # 최단 거리 테이블을 모두 무한으로 초기화
    distance = [INF] * (n + 1)

    # 모든 간선 정보를 입력받기
    for _ in range(m):
        a, b, c = map(int, input().split())
        # a번 노드에서 b번 노드로 가는 비용이 c라는 의미
        graph[a].append((b, c))

    def dijkstra(start):
        q = []
        # 시작 노드로 가기 위한 최단 경로는 0으로 설정하여, 큐에 삽입
        heapq.heappush(q, (0, start))
        distance[start] = 0
        while q: # 큐가 비어있지 않다면
            # 가장 최단 거리가 짧은 노드에 대한 정보 꺼내기
            dist, now = heapq.heappop(q)
            # 현재 노드가 이미 처리된 적이 있는 노드라면 무시
            if distance[now] < dist:
                continue
            # 현재 노드와 연결된 다른 인접한 노드들을 확인
            for i in graph[now]:
                cost = dist + i[1]
                # 현재 노드를 거쳐서, 다른 노드로 이동하는 거리가 더 짧은 경우
                if cost < distance[i[0]]:
                    distance[i[0]] = cost
                    heapq.heappush(q, (cost, i[0]))

    # 다익스트라 알고리즘을 수행
    dijkstra(start)

    # 모든 노드로 가기 위한 최단 거리를 출력
    for i in range(1, n + 1):
        # 도달할 수 없는 경우, 무한(INFINITY)이라고 출력
        if distance[i] == INF:
            print("INFINITY")
        # 도달할 수 있는 경우 거리를 출력
        else:
            print(distance[i])
    ```

## 플로이드 워셜 알고리즘

- ‘모든 지점에서 다른 모든 지점까지의 최단 경로를 모두 구해야 하는 경우’에 사용할 수 있는 알고리즘
- 단계마다 ‘거쳐 가는 노드’를 기준으로 알고리즘 수행 → 매번 방문하지 않는 노드 중에서 최단 경로를 찾을 필요가 없다
- 시간 복잡도: $$O(N^3)$$
- 다이나믹 프로그래밍 → 2차원 리스트에 최단 거리 정보를 담음
- 점화식
    
    $$D_{ab}=min(D_{ab},D_{ak}+D_{kb})$$
    
- 초기 테이블에 연결된 간선 = 단순히 그 값, 연결되지 않은 간선 = 무한대 값을 넣고 시작

```python
INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

# 노드의 개수 및 간선의 개수를 입력받기
n = int(input())
m = int(input())
# 2차원 리스트(그래프 표현)를 만들고, 모든 값을 무한으로 초기화
graph = [[INF] * (n + 1) for _ in range(n + 1)]

# 자기 자신에서 자기 자신으로 가는 비용은 0으로 초기화
for a in range(1, n + 1):
    for b in range(1, n + 1):
        if a == b:
            graph[a][b] = 0

# 각 간선에 대한 정보를 입력 받아, 그 값으로 초기화
for _ in range(m):
    # A에서 B로 가는 비용은 C라고 설정
    a, b, c = map(int, input().split())
    graph[a][b] = c

# 점화식에 따라 플로이드 워셜 알고리즘을 수행
for k in range(1, n + 1):
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])

# 수행된 결과를 출력
for a in range(1, n + 1):
    for b in range(1, n + 1):
        # 도달할 수 없는 경우, 무한(INFINITY)이라고 출력
        if graph[a][b] == 1e9:
            print("INFINITY", end=" ")
        # 도달할 수 있는 경우 거리를 출력
        else:
            print(graph[a][b], end=" ")
    print()
```

## 예제 문제

- 예제 1: 미래 도시
    1. 내 풀이
        1. 플로이드 워셜 알고리즘을 이용하여 각 지점까지의 최단거리 구함
        2. 그 후, (1→K) + (K→X) 최단거리를 각각 더해서 값 출력
        
        ```python
        import sys
        input=sys.stdin.readline
        
        INF=int(1e9)
        
        N,M=map(int,input().split())
        graph=[[INF]*(N+1) for _ in range(N+1)]
        
        for i in range(N+1):
          for j in range(N+1):
            if i==j:
              graph[i][j]=0
        
        for i in range(M):
          a,b=map(int,input().split())
          graph[a][b]=1
          graph[b][a]=1
          
        X,K=map(int,input().split())
        
        for k in range(1,N+1):
          for i in range(1,N+1):
            for j in range(1,N+1):
              graph[i][j]=min(graph[i][j],graph[i][k]+graph[k][j])
        
        if graph[1][K]==INF or graph[K][X]==INF:
          print(-1)
        else:
          print(graph[1][K]+graph[K][X])
        ```
        
    2. 풀이를 본 후
        
        풀이도 같은 방법이다
        
        ```python
        INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정
        
        # 노드의 개수 및 간선의 개수를 입력받기
        n, m = map(int, input().split())
        # 2차원 리스트(그래프 표현)를 만들고, 모든 값을 무한으로 초기화
        graph = [[INF] * (n + 1) for _ in range(n + 1)]
        
        # 자기 자신에서 자기 자신으로 가는 비용은 0으로 초기화
        for a in range(1, n + 1):
            for b in range(1, n + 1):
                if a == b:
                    graph[a][b] = 0
        
        # 각 간선에 대한 정보를 입력 받아, 그 값으로 초기화
        for _ in range(m):
            # A와 B가 서로에게 가는 비용은 1이라고 설정
            a, b = map(int, input().split())
            graph[a][b] = 1
            graph[b][a] = 1
        
        # 거쳐 갈 노드 X와 최종 목적지 노드 K를 입력받기
        x, k = map(int, input().split())
        
        # 점화식에 따라 플로이드 워셜 알고리즘을 수행
        for k in range(1, n + 1):
            for a in range(1, n + 1):
                for b in range(1, n + 1):
                    graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])
        
        # 수행된 결과를 출력
        distance = graph[1][k] + graph[k][x]
        
        # 도달할 수 없는 경우, -1을 출력
        if distance >= 1e9:
            print("-1")
        # 도달할 수 있다면, 최단 거리를 출력
        else:
            print(distance)
        ```
        
    3. 해결한 후
        
        플로이드 워셜 알고리즘 구현방법 숙지 필수
        
    
- 예제 2: 전보
    1. 내 풀이
        1. 다익스트라 알고리즘을 이용하여 최단거리 테이블 구함
        2. 도달 할 수 있는 도시의 수와 최대 시간을 구해서 출력
        
        ```python
        import heapq
        import sys
        input=sys.stdin.readline
        INF=int(1e9)
        
        def solve(start):
          q=[]
          count, time=0,0
          heapq.heappush(q,(0,C))
          distance[start]=0
          while q:
            dist, now = heapq.heappop(q)
            if dist>distance[now]:
              continue
            for g in graph[now]:
              cost=dist+g[1]
              if cost<distance[g[0]]:
                distance[g[0]]=cost
                heapq.heappush(q,(cost,g[0]))
        
          for d in distance:
            if d==INF or d==0:
              continue
            else:
              count+=1
              time=max(time, d)
          return print(count, time)
        
        N,M,C=map(int,input().split())
        graph=[[] for _ in range(N+1)]
        distance=[INF]*(N+1)
        
        for i in range(M):
          a,b,c=map(int,input().split())
          graph[a].append((b,c))
        
        solve(C)
        ```
        
    2. 풀이를 본 후
        
        풀이도 같은 방법이다
        
        ```python
        import heapq
        import sys
        input = sys.stdin.readline
        INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정
        
        # 노드의 개수, 간선의 개수, 시작 노드를 입력받기
        n, m, start = map(int, input().split())
        # 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
        graph = [[] for i in range(n + 1)]
        # 최단 거리 테이블을 모두 무한으로 초기화
        distance = [INF] * (n + 1)
        
        # 모든 간선 정보를 입력받기
        for _ in range(m):
            x, y, z = map(int, input().split())
            # X번 노드에서 Y번 노드로 가는 비용이 Z라는 의미
            graph[x].append((y, z))
        
        def dijkstra(start):
           q = []
           # 시작 노드로 가기 위한 최단 경로는 0으로 설정하여, 큐에 삽입
           heapq.heappush(q, (0, start))
           distance[start] = 0
           while q: # 큐가 비어있지 않다면
                # 가장 최단 거리가 짧은 노드에 대한 정보를 꺼내기
                dist, now = heapq.heappop(q)
                if distance[now] < dist:
                    continue
                # 현재 노드와 연결된 다른 인접한 노드들을 확인
                for i in graph[now]:
                    cost = dist + i[1]
                    # 현재 노드를 거쳐서, 다른 노드로 이동하는 거리가 더 짧은 경우
                    if cost < distance[i[0]]:
                        distance[i[0]] = cost
                        heapq.heappush(q, (cost, i[0]))
        
        # 다익스트라 알고리즘을 수행
        dijkstra(start)
        
        # 도달할 수 있는 노드의 개수
        count = 0
        # 도달할 수 있는 노드 중에서, 가장 멀리 있는 노드와의 최단 거리
        max_distance = 0
        for d in distance:
            # 도달할 수 있는 노드인 경우
            if d != 1e9:
                count += 1
                max_distance = max(max_distance, d)
        
        # 시작 노드는 제외해야 하므로 count - 1을 출력
        print(count - 1, max_distance)
        ```
        
    3. 해결한 후
        
        우선순위 큐를 이용하여 다익스트라를 구현하는 방법을 반드시 외우고 있어야겠다.

## 기출 문제

- 기출 문제 1: 플로이드
    1. 내 풀이
        1. 플로이드 워셜 알고리즘을 이용하여 초기 테이블 생성
        2. 이 때 도시간 이동 비용는 최소값을 넣어준다.
        3. 최종적인 최소 이동 비용 출력 
        
        도시간 노선이 한개 이상, 도시간 이동 불가 처리를 추가적으로 해야했다.
        
        ```python
        import sys
        input=sys.stdin.readline
        INF=int(1e9)
        
        def solve():
          for k in range(1,N+1):
            for i in range(1,N+1):
              for j in range(1,N+1):
                graph[i][j]=min(graph[i][j],graph[i][k]+graph[k][j])
        
          for i in range(1,N+1):
            for j in range(1,N+1):
              if graph[i][j]==INF:
                print(0,end=" ")
              else:
                print(graph[i][j],end=" ")
            print()
        
        N=int(input())
        M=int(input())
        graph=[[INF]*(N+1) for _ in range(N+1)]
        
        for i in range(1,N+1):
          for j in range(1,N+1):
            if i==j:
              graph[i][j]=0
              
        for i in range(M):
          a,b,c=map(int,input().split())
          graph[a][b]=min(graph[a][b],c)
        
        solve()
        ```
        
    2. 풀이를 본 후
        
        풀이도 같은 방법이다
        
        ```python
        INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정
        
        # 노드의 개수 및 간선의 개수를 입력받기
        n = int(input())
        m = int(input())
        # 2차원 리스트(그래프 표현)를 만들고, 모든 값을 무한으로 초기화
        graph = [[INF] * (n + 1) for _ in range(n + 1)]
        
        # 자기 자신에서 자기 자신으로 가는 비용은 0으로 초기화
        for a in range(1, n + 1):
            for b in range(1, n + 1):
                if a == b:
                    graph[a][b] = 0
        
        # 각 간선에 대한 정보를 입력받아, 그 값으로 초기화
        for _ in range(m):
            # A에서 B로 가는 비용은 C라고 설정
            a, b, c = map(int, input().split())
            # 가장 짧은 간선 정보만 저장
            if c < graph[a][b]:
                graph[a][b] = c
        
        # 점화식에 따라 플로이드 워셜 알고리즘을 수행
        for k in range(1, n + 1):
            for a in range(1, n + 1):
                for b in range(1, n + 1):
                    graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])
        
        # 수행된 결과를 출력
        for a in range(1, n + 1):
            for b in range(1, n + 1):
                # 도달할 수 없는 경우, 0을 출력
                if graph[a][b] == INF:
                    print(0, end=" ")
                # 도달할 수 있는 경우 거리를 출력
                else:
                    print(graph[a][b], end=" ")
            print()
        ```
        
    3. 해결한 후
        
        플로이드 워셜 알고리즘 구현 방법은 알고있는게 당연하고 문제의 조건도 항상 유심히 봐야한다.
        
- 기출 문제 2: 정확한 순위
    1. 내 풀이
        1. 플로이드 워셜을 이용하여 최단거리의 모든 쌍을 구한다.
        2. 특정 학생을 기준으로 다른 모든 학생들과 들어가고 나가고가 한번이라도 가능하면 정확한 순위를 알 수 있다.
        3. 최종적인 정확한 순위 알 수 있는 학생의 수 출력
        
        ```python
        import sys
        input=sys.stdin.readline
        INF=int(1e9)
        
        def solve():
          count=0
          for k in range(1,N+1):
            for i in range(1,N+1):
              for j in range(1,N+1):
                graph[i][j]=min(graph[i][j],graph[i][k]+graph[k][j])
        
          for i in range(1,N+1):
            temp=True
            for j in range(1,N+1):
              if graph[i][j]==INF and graph[j][i]==INF:
                temp=False
                break
            if temp:
              count+=1
          return print(count)
        
        N,M=map(int,input().split())
        graph=[[INF]*(N+1) for _ in range(N+1)]
        
        for i in range(1,N+1):
          for j in range(1,N+1):
            if i==j:
              graph[i][j]=0
              
        for i in range(M):
          a,b=map(int,input().split())
          graph[a][b]=1
        
        solve()
        ```
        
    2. 풀이를 본 후
        
        풀이도 같은 방법이다
        
        ```python
        INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정
        
        # 노드의 개수, 간선의 개수를 입력받기
        n, m = map(int, input().split())
        # 2차원 리스트(그래프 표현)를 만들고, 모든 값을 무한으로 초기화
        graph = [[INF] * (n + 1) for _ in range(n + 1)]
         
        # 자기 자신에서 자기 자신으로 가는 비용은 0으로 초기화
        for a in range(1, n + 1):
            for b in range(1, n + 1):
                if a == b:
                    graph[a][b] = 0
         
        # 각 간선에 대한 정보를 입력 받아, 그 값으로 초기화
        for _ in range(m):
            # A에서 B로 가는 비용을 1로 설정
            a, b = map(int, input().split())
            graph[a][b] = 1
         
        # 점화식에 따라 플로이드 워셜 알고리즘을 수행
        for k in range(1, n + 1):
            for a in range(1, n + 1):
                for b in range(1, n + 1):
                    graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])
        
        result = 0
        # 각 학생을 번호에 따라 한 명씩 확인하며 도달 가능한지 체크
        for i in range(1, n + 1):
            count = 0
            for j in range(1, n + 1):
                if graph[i][j] != INF or graph[j][i] != INF:
                    count += 1
            if count == n:
                result += 1
        print(result)
        ```
        
    3. 해결한 후
        
        정확한 순위를 알 수 있는 조건을 판단하는 것이 중요한 문제였다.

## 출처

- 이것이 취업을 위한 코딩테스트다 with 파이썬