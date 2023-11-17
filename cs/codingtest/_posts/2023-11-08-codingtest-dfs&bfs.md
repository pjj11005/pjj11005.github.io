---
layout: post
title: 3. DFS/BFS
categories: 
  - cs
  - codingtest
description: 이것이 취업을 위한 코딩테스트다 with 파이썬 책을 참고하여 **DFS/BFS**을 공부한 내용입니다.
sitemap: false
---

대표적인 탐색 알고리즘

* this unordered seed list will be replaced by the toc
{:toc}

## 꼭 필요한 자료구조 기초

- 탐색: 많은 양의 데이터에서 원하는 데이터를 찾는 과정
    - 대표적인 탐색 알고리즘: DFS, BFS
- 자료구조: 데이터를 표현하고 관리하고 처리하기 위한 구조
    - 스택, 큐: pop(삭제), push(삽입)
        - 오버 플로(Overflow): 데이터의 크기가 가득 찬 상태로 삽입 시 발생
        - 언더 플로(Underflow): 데이터가 전혀 들어있지 않을 때 삭제 시 발생
        
- 스택
    - 박스 쌓기, 선입후출(FILO) 또는 후입선출(LIFO) 구조
    - 기본적으로 append(), pop() 연산 사용
- 큐
    - 대기 줄, 선입선출(FIFO) 구조
    - 파이썬에서 큐 구현 시 collections 모듈의 deque 자료구조 활용하기
    - 리스트에 비해 넣고 빼는 속도가 빠르다
- 재귀 함수
    - 자기 자신을 호출하는 함수
    - 프랙털 구조와 비슷 ex) 시에르핀스키의 삼각형
    - 재귀 함수 문제 풀이에 사용 시 종료 조건을 꼭 명시해야 한다.
    - 내부적으로 스택 자료구조와 동일
    - 대표적 예제: 팩토리얼 문제
    - 점화식을 이용하여 문제 해결(다이나믹 프로그래밍과 연결)
    

## 탐색 알고리즘 DFS/BFS

- DFS
    - 깊이 우선 탐색, 그래프에서 깊은 부분을 우선적으로 탐색하는 알고리즘
        - 그래프: 노드(정점), 간선으로 표현
        - 인접 행렬: 2차원 배열로 그래프의 연결 관계를 표현하는 방식 → 2차원 리스트로 구현
        - 인접 리스트: 리스트로 그래프의 연결 관계를 표현하는 방식 → 2차원 리스트로 구현
            
            
            |  | 인접 행렬 | 인접 리스트 |
            | --- | --- | --- |
            | 메모리 낭비 | 크다 | 작다 |
            | 정보 얻는 속도 | 빠르다 | 느리다 |
    - 동작 과정(스택 구조 이용)
        1. 탐색 시작 노드를 스택에 삽입하고 방문 처리를 한다
        2. 스택의 최상단 노드에 방문하지 않은 인접 노드가 있으면 그 인접 노드를 스택에 방문 처리를 한다. 방문하지 않은 인접 노드가 없으면 스택에서 최상단 노드를 꺼낸다.
        3. 2번의 과정을 더 이상 수행할 수 없을 때 까지 반복한다
    - O(N)의 시간복잡도
    - DFS 예제 코드
        
        ```python
        # DFS 함수 정의
        def dfs(graph, v, visited):
            # 현재 노드를 방문 처리
            visited[v] = True
            print(v, end=' ')
            # 현재 노드와 연결된 다른 노드를 재귀적으로 방문
            for i in graph[v]:
                if not visited[i]:
                    dfs(graph, i, visited)
        
        # 각 노드가 연결된 정보를 리스트 자료형으로 표현(2차원 리스트)
        graph = [
          [],
          [2, 3, 8],
          [1, 7],
          [1, 4, 5],
          [3, 5],
          [3, 4],
          [7],
          [2, 6, 8],
          [1, 7]
        ]
        
        # 각 노드가 방문된 정보를 리스트 자료형으로 표현(1차원 리스트)
        visited = [False] * 9
        
        # 정의된 DFS 함수 호출
        dfs(graph, 1, visited)
        ```
        
- BFS
    - 너비 우선 탐색, 가까운 노드부터 탐색하는 알고리즘
    - DFS 와는 반대로 동작
    - 동작 과정(큐 자료구조 이용)
        1. 탐색 시작 노드를 큐에 삽입하고 방문 처리를 한다
        2. 큐에서 노드를 꺼내 해당 노드의 인접 노드 중에서 방문하지 않은 노드를 모두 큐에 삽입하고 방문 처리를 한다
        3. 2번의 과정을 더 이상 수행 할 수 없을 때까지 반복한다
    - deque 라이브러리를 사용 하는게 좋고, 시간 복잡도는 O(N)
    - 일반적으로 DFS보다 수행 시간이 좋은 편
    - BFS 예제 코드
        
        ```python
        from collections import deque
        
        # BFS 함수 정의
        def bfs(graph, start, visited):
            # 큐(Queue) 구현을 위해 deque 라이브러리 사용
            queue = deque([start])
            # 현재 노드를 방문 처리
            visited[start] = True
            # 큐가 빌 때까지 반복
            while queue:
                # 큐에서 하나의 원소를 뽑아 출력
                v = queue.popleft()
                print(v, end=' ')
                # 해당 원소와 연결된, 아직 방문하지 않은 원소들을 큐에 삽입
                for i in graph[v]:
                    if not visited[i]:
                        queue.append(i)
                        visited[i] = True
        
        # 각 노드가 연결된 정보를 리스트 자료형으로 표현(2차원 리스트)
        graph = [
          [],
          [2, 3, 8],
          [1, 7],
          [1, 4, 5],
          [3, 5],
          [3, 4],
          [7],
          [2, 6, 8],
          [1, 7]
        ]
        
        # 각 노드가 방문된 정보를 리스트 자료형으로 표현(1차원 리스트)
        visited = [False] * 9
        
        # 정의된 BFS 함수 호출
        bfs(graph, 1, visited)
        ```
        
- 정리
    
    
    |  | DFS | BFS |
    | --- | --- | --- |
    | 동작 원리 | 스택 | 큐 |
    | 구현 방법 | 재귀 함수 이용 | 큐 자료구조 이용 |
    
    1,2차원 배열을 그래프의 형태로 생각하여 문제 해결


## 예제 문제

- 예제 1: 음료수 얼려 먹기

    1. 내 풀이
        
        1. 우선 DFS를 이용해서 얼음 틀이고 방문하지 않은 곳을 시작점으로 하여 얼음 탐색
        
        1. 반복하여 다른 얼음 들도 탐색하고 얼음의 개수 출력
        
        ```python
        def solve(x,y):
          for i in range(4):
            nx,ny=x+dx[i],y+dy[i]
            if 0<=nx<N and 0<=ny<M and ice[nx][ny]==0 and visited[nx][ny]==0:
              visited[nx][ny]=1
              solve(nx,ny)
                
        N,M=map(int,input().split())
        ice=[list(map(int,input())) for _ in range(N)]
        visited=[[0 for _ in range(M)] for _ in range(N)]
        dx=[-1,1,0,0]
        dy=[0,0,-1,1]
        answer=0
        
        for i in range(N):
          for j in range(M):
            if ice[i][j]==0 and visited[i][j]==0:
              visited[i][j]=1
              solve(i,j)
              answer+=1
              
        print(answer)
        ```
        
    2. 풀이를 본 후
        
        우선 방문 노드를 만들지 않고도 방문한 얼음 틀은 1로 바꿔주면 된다. 나머지 부분은 나의 풀이와 비슷했다.
        
        ```python
        # N, M을 공백을 기준으로 구분하여 입력 받기
        n, m = map(int, input().split())
        
        # 2차원 리스트의 맵 정보 입력 받기
        graph = []
        for i in range(n):
            graph.append(list(map(int, input())))
        
        # DFS로 특정한 노드를 방문한 뒤에 연결된 모든 노드들도 방문
        def dfs(x, y):
            # 주어진 범위를 벗어나는 경우에는 즉시 종료
            if x <= -1 or x >= n or y <= -1 or y >= m:
                return False
            # 현재 노드를 아직 방문하지 않았다면
            if graph[x][y] == 0:
                # 해당 노드 방문 처리
                graph[x][y] = 1
                # 상, 하, 좌, 우의 위치들도 모두 재귀적으로 호출
                dfs(x - 1, y)
                dfs(x, y - 1)
                dfs(x + 1, y)
                dfs(x, y + 1)
                return True
            return False
        
        # 모든 노드(위치)에 대하여 음료수 채우기
        result = 0
        for i in range(n):
            for j in range(m):
                # 현재 위치에서 DFS 수행
                if dfs(i, j) == True:
                    result += 1
        
        print(result) # 정답 출력
        ```
        
    3. 해결한 후
        
        DFS 문제는 종료 조건 설정과 종료 지점을 파악해서 구현하는 것이 중요한 것 같다.
        
- 예제 문제 2: 미로 탈출

    1. 내 풀이
        
        1. 처음 위치의 좌표와 이동 칸의 수를 큐에 넣으며 BFS탐색
        
        1. 큐에 있는 좌표들을 모두 탐색한 후, 목표지점에 도달할 때 마다 이동 칸의 수를 갱신해준다.
        2. 마지막에 answer값에 저장된 값을 출력
        
        ```python
        from collections import deque
        
        def solve():
          q=deque()
          q.append((0,0,1))
          visited[0][0]=1
          answer=1e9
          while q:
            x,y,count=q.popleft()
            for i in range(4):
              nx,ny=x+dx[i],y+dy[i]
        
              if 0<=nx<N and 0<=ny<M and maze[nx][ny]==1 and visited[nx][ny]==0:
                if nx==N-1 and ny==M-1:
                  answer=min(answer,count+1)          
                else:
                  visited[nx][ny]=1
                  q.append((nx,ny,count+1))
        
          return print(answer)
          
        N,M=map(int,input().split())
        maze=[list(map(int,input())) for _ in range(N)]
        visited=[[0 for _ in range(M)] for _ in range(N)]
        dx=[-1,1,0,0]
        dy=[0,0,-1,1]
              
        solve()
        ```
        
    2. 풀이를 본 후
        
        우선 방문 노드 없이 수를 더해가며 값을 바꿔 갔고, 목표 지점에 제일 먼저 도달 할 때가 최소 거리가 되어 그저 먼저 도달하면 목표 지점의 값을 출력하면 된다.
        
        ```python
        from collections import deque
        
        # N, M을 공백을 기준으로 구분하여 입력 받기
        n, m = map(int, input().split())
        # 2차원 리스트의 맵 정보 입력 받기
        graph = []
        for i in range(n):
            graph.append(list(map(int, input())))
        
        # 이동할 네 가지 방향 정의 (상, 하, 좌, 우)
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        
        # BFS 소스코드 구현
        def bfs(x, y):
            # 큐(Queue) 구현을 위해 deque 라이브러리 사용
            queue = deque()
            queue.append((x, y))
            # 큐가 빌 때까지 반복하기
            while queue:
                x, y = queue.popleft()
                # 현재 위치에서 4가지 방향으로의 위치 확인
                for i in range(4):
                    nx = x + dx[i]
                    ny = y + dy[i]
                    # 미로 찾기 공간을 벗어난 경우 무시
                    if nx < 0 or nx >= n or ny < 0 or ny >= m:
                        continue
                    # 벽인 경우 무시
                    if graph[nx][ny] == 0:
                        continue
                    # 해당 노드를 처음 방문하는 경우에만 최단 거리 기록
                    if graph[nx][ny] == 1:
                        graph[nx][ny] = graph[x][y] + 1
                        queue.append((nx, ny))
            # 가장 오른쪽 아래까지의 최단 거리 반환
            return graph[n - 1][m - 1]
        
        # BFS를 수행한 결과 출력
        print(bfs(0, 0))
        ```
        
    3. 해결한 후
        
        굳이 다른 변수들로 메모리를 할당하지 않고 효율적으로 할 수 있는 방법도 생각하면 좋을 것 같다.
        

## 기출 문제

- 기출 문제 1: 특정 거리의 도시 찾기

    1. 내 풀이
        
        1.  우선 간선의 정보를 저장한다.
        
        1. 그 후, BFS 탐색을 통해서 최단 거리가 K가 되는 도시들을 answer 배열에 저장한다.
        2. 아무것도 없으면 -1을 존재하면 오름차순으로 정렬 후 출력
        
        PyPy3로 제출하니 맞았다…
        
        ```python
        from collections import deque
        
        def solve():
          q=deque()
          q.append((X,0))
          visited[X]=1
          while q:
            x,count=q.popleft()
            for d in dist[x]:
              if visited[d]==0:
                if count+1==K:
                  visited[d]=1
                  answer.append(d)
                else:
                  visited[d]=1
                  q.append((d,count+1))
          
        N,M,K,X=map(int,input().split())
        dist=[[] for _ in range(N+1)]
        for i in range(M):
          a,b=map(int,input().split())
          dist[a].append(b)
          
        visited=[0]*(N+1)
        answer=[]   
        solve()
        
        if len(answer)==0:
          print(-1)
        else:
          answer.sort()
          for a in answer:
            print(a)
        ```
        
    2. 풀이를 본 후
        
        풀이는 최단 거리 배열을 따로 만들어서 각 도시 까지의 최단 거리를 저장하면서 진행했다.
        
        시간은 내 풀이가 조금 더 빨랐다^^
        
        ```python
        from collections import deque
        
        # 도시의 개수, 도로의 개수, 거리 정보, 출발 도시 번호
        n, m, k, x = map(int, input().split())
        graph = [[] for _ in range(n + 1)]
        
        # 모든 도로 정보 입력 받기
        for _ in range(m):
            a, b = map(int, input().split())
            graph[a].append(b)
        
        # 모든 도시에 대한 최단 거리 초기화
        distance = [-1] * (n + 1)
        distance[x] = 0 # 출발 도시까지의 거리는 0으로 설정
        
        # 너비 우선 탐색(BFS) 수행
        q = deque([x])
        while q:
            now = q.popleft()
            # 현재 도시에서 이동할 수 있는 모든 도시를 확인
            for next_node in graph[now]:
                # 아직 방문하지 않은 도시라면
                if distance[next_node] == -1:
                    # 최단 거리 갱신
                    distance[next_node] = distance[now] + 1
                    q.append(next_node)
        
        # 최단 거리가 K인 모든 도시의 번호를 오름차순으로 출력
        check = False
        for i in range(1, n + 1):
            if distance[i] == k:
                print(i)
                check = True
        
        # 만약 최단 거리가 K인 도시가 없다면, -1 출력
        if check == False:
            print(-1)
        ```
        
    3. 해결한 후
        
        최단 거리나 가까운 거리부터 탐색을 진행하면 BFS로 해결할 수 있다. 또한, 문제 해결 시 조건을 제대로 파악하고 해결하자

- 기출 문제 2: 연구소

    1. 내 풀이
        
        1. 우선 빈칸, 바이러스의 좌표를 저장해 놓는다.
        2. 빈칸의 좌표들 중 3개를 골라서 만들 수 있는 조합을 리스트에 저장한다.
        3. 조합 리스트를 돌면서 가장 작게 바이러스가 퍼질때의 칸의 수를 answer에 저장한다.
        4. 마지막으로 빈칸의 개수에서 answer를 빼준다.
        시간 초과가 발생했다.
        
        ```python
        from collections import deque
        from itertools import combinations
        
        def solve(x,y):
          global count
          for i in range(4):
            nx,ny=x+dx[i],y+dy[i]
            if 0<=nx<N and 0<=ny<M and visited[nx][ny]==0 and lab[nx][ny]==0:
              visited[nx][ny]=1
              lab[nx][ny]=2
              count+=1
              solve(nx,ny)
              visited[nx][ny]=0
              lab[nx][ny]=0
         
          
        N,M=map(int,input().split())
        lab=[list(map(int,input().split())) for _ in range(N)]
        point=[]
        virus=[]
        dx=[-1,1,0,0]
        dy=[0,0,-1,1]
        for i in range(N):
          for j in range(M):
            if lab[i][j]==0:
              point.append((i,j))
            if lab[i][j]==2:
              virus.append((i,j))
              
        comb_point=list(combinations(point, 3))
        
        visited=[[0]*M for _ in range(N)]
        answer=64
        
        for comb in comb_point:
          count=0
          for c in comb:
            lab[c[0]][c[1]]=1
        
          for v in virus:
            solve(v[0],v[1])
          
          for c in comb:
            lab[c[0]][c[1]]=0
          
          answer=min(answer,count)
        
        print(len(point)-answer)
        ```
        
    2. 풀이를 본 후
        
        아이디어는 거의 유사했다. 하지만, 조합 라이브러리를 사용하지 않고 DFS 탐색 함수 내에서 완전 탐색을 이용해서 벽을 세워주고 그때마다 탐색을 해서 값을 구하도록 했다. 이 때, 안전 영역의 값은 그냥 완전 탐색하여 0의 좌표 수를 세어 주는 것으로 했다. 또한, 벽을 세울 때마다 저장할 임의의 배열도 만들었다. 
        
        조금만 신중하게 생각했다면 해결했을 것 같다…

        