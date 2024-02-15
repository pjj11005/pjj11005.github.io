---
layout: post
title: 이것이 취업을 위한 코딩테스트다 | 3. DFS/BFS
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
        
        - 처음 풀이
            1. 우선 간선의 정보를 저장한다.
            2. 그 후, BFS 탐색을 통해서 최단 거리가 K가 되는 도시들을 answer 배열에 저장한다.
            3. 아무것도 없으면 -1을 존재하면 오름차순으로 정렬 후 출력
            
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
            
        - 두 번째 풀이
            1. 간선의 정보를 저장
            2. 그 후, BFS 탐색을 통해서 최단 거리가 K가 되는 도시들을 answer 배열에 저장
            3. 아무것도 없으면 -1, 존재하면 오름차순으로 정렬 후 도시들 출력
            
            입력정보가 많아 PyPy3로 제출하니 맞았다, 처음 풀이와 거의 유사했다 → 다음에는 최단거리 정보를 저장하며 풀이를 진행해봐야겠다.
            
            ```python
            from collections import deque
            
            def bfs():
                q=deque([(x,0)])
                visited[x]=1
                
                while q:
                    a, dist=q.popleft()
                    if dist==k:
                        answer.append(a)
                    
                    for i in graph[a]:
                        if not visited[i]:
                            q.append((i, dist+1))
                            visited[i]=1
            
            if len(answer)==0: # 존재 X
                print(-1)
            else:
                answer.sort()
                for a in answer:
                    print(a)
            
            n,m,k,x=map(int, input().split()) # 도시 개수, 도로 개수, 거리, 출발 도시
            visited=[0]*(n+1)
            graph=[[] for _ in range(n+1)] # 간선 정보
            for i in range(m):
                a, b = map(int, input().split())
                graph[a].append(b)
            
            answer=[]
            
            bfs()
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
        - 처음 풀이
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
        
        - 두 번째 풀이
            1. 빈칸, 바이러스 좌표 저장
            2. 빈칸에 벽 3개를 설치할 때마다 바이러스를 퍼지게 한 후 안전 영역 계산
            3. 최댓값을 모든 조합에서 계산한 후 출력
            
            >- **시간이 조금 오래 걸렸다 (배열 복사를 잘못하고 있었다…)**
            >- **풀이를 보면 벽의 설치도 DFS를 이용하여 진행했다 → 다음에는 풀이와 같이 진행해보도록 해야겠다**
            >    - **이렇게 하면 빈칸, 바이러스의 좌표를 저장할 필요가 없어진다**
            >- **백준에서는 PyPy3로 제출해야 한다고 함**
            

            ```python
            from itertools import combinations
            
            def dfs(x,y):
            
                for i in range(4):
                    nx, ny=x+dx[i], y+dy[i]
                
                    if 0<=nx<n and 0<=ny<m and temp_array[nx][ny]==0:
                        temp_array[nx][ny]=2
                        dfs(nx, ny)
            
            n,m=map(int,input().split())
            array=[list(map(int,input().split())) for _ in range(n)] # 지도 정보
            zero=[] # 빈칸의 좌표들
            second=[] # 바이러스의 좌표들
            
            for i in range(n):
                for j in range(m):
                    if array[i][j]==0:
                        zero.append((i,j))
                    elif array[i][j]==2:
                        second.append((i,j))
                
            dx=[-1,1,0,0]
            dy=[0,0,-1,1]
            answer=0
            temp_array=[[0]*m for _ in range(n)] # 복사할 지도
            
            for comb in list(combinations(zero,3)):
                for i in range(n):
                    for j in range(m):
                        temp_array[i][j]=array[i][j]
                
                for i, j in comb: # 벽 설치
                    temp_array[i][j]=1
                
                for x, y in second: # 바이러스 퍼짐
                    dfs(x,y)
                
                count=0
                for i in range(n):
                    for j in range(m):
                        if temp_array[i][j]==0:
                            count+=1
                
                answer=max(answer, count)
            
            print(answer)
            ```

    2. 풀이를 본 후
        
        아이디어는 거의 유사했다. 하지만, 조합 라이브러리를 사용하지 않고 DFS 탐색 함수 내에서 완전 탐색을 이용해서 벽을 세워주고 그때마다 탐색을 해서 값을 구하도록 했다. 이 때, 안전 영역의 값은 그냥 완전 탐색하여 0의 좌표 수를 세어 주는 것으로 했다. 또한, 벽을 세울 때마다 저장할 임의의 배열도 만들었다. 
        
        조금만 신중하게 생각했다면 해결했을 것 같다…
        
        ```python
        # BOJ에서는 [언어]를 PyPy3로 설정하여 제출해주세요.
        
        n, m = map(int, input().split())
        data = [] # 초기 맵 리스트
        temp = [[0] * m for _ in range(n)] # 벽을 설치한 뒤의 맵 리스트
        
        for _ in range(n):
            data.append(list(map(int, input().split())))
        
        # 4가지 이동 방향에 대한 리스트
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]
        
        result = 0
        
        # 깊이 우선 탐색(DFS)을 이용해 각 바이러스가 사방으로 퍼지도록 하기
        def virus(x, y):
            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]
                # 상, 하, 좌, 우 중에서 바이러스가 퍼질 수 있는 경우
                if nx >= 0 and nx < n and ny >= 0 and ny < m:
                    if temp[nx][ny] == 0:
                        # 해당 위치에 바이러스 배치하고, 다시 재귀적으로 수행
                        temp[nx][ny] = 2
                        virus(nx, ny)
        
        # 현재 맵에서 안전 영역의 크기 계산하는 메서드
        def get_score():
            score = 0
            for i in range(n):
                for j in range(m):
                    if temp[i][j] == 0:
                        score += 1
            return score
        
        # 깊이 우선 탐색(DFS)을 이용해 울타리를 설치하면서, 매 번 안전 영역의 크기 계산
        def dfs(count):
            global result
            # 울타리가 3개 설치된 경우
            if count == 3:
                for i in range(n):
                    for j in range(m):
                        temp[i][j] = data[i][j]
                # 각 바이러스의 위치에서 전파 진행
                for i in range(n):
                    for j in range(m):
                        if temp[i][j] == 2:
                            virus(i, j)
                # 안전 영역의 최대값 계산
                result = max(result, get_score())
                return
            # 빈 공간에 울타리를 설치
            for i in range(n):
                for j in range(m):
                    if data[i][j] == 0:
                        data[i][j] = 1
                        count += 1
                        dfs(count)
                        data[i][j] = 0
                        count -= 1
        
        dfs(0)
        print(result)
        ```
        

- 기출 문제 3: 경쟁적 전염

    1. 내 풀이
        - 처음 풀이
            1. 우선 초기 바이러스 위치, 바이러스 종류, 시간 값을 virus 리스트에 저장 후 작은 값부터 오름차순으로 정렬
            2. 그 후, BFS탐색을 시행하여 시간 S까지 탐색
            3. 최종적으로 X,Y 좌표 출력
            
            반복문 종료에서 어이없이 실수를 해서 조금 오래걸렸다.
            
            ```python
            from collections import deque
            
            def solve():
                q = deque(virus)
                
                while q:
                    x, y, v, time = q.popleft()
                    if time == S:
                        break
                    
                    for i in range(4):
                        nx, ny = x + dx[i], y + dy[i]
                        if 0 <= nx < N and 0 <= ny < N and test[nx][ny] == 0:
                            test[nx][ny] = v
                            q.append((nx, ny, v, time + 1))
            
            N, K = map(int, input().split())
            test = [list(map(int, input().split())) for _ in range(N)]
            S, X, Y = map(int, input().split())
            
            virus = []
            for i in range(N):
                for j in range(N):
                    if test[i][j] != 0:
                        virus.append((i, j, test[i][j],0))
            
            virus = sorted(virus, key=lambda x: x[2])
            dx = [-1, 1, 0, 0]
            dy = [0, 0, -1, 1]
            
            solve()
            print(test[X - 1][Y - 1])
            ```
        
        - 두 번째 풀이
            1. 바이러스의 번호, 좌표, 시간 정보를 번호 순으로 리스트에 저장
            2. 바이러스 번호 순서대로 BFS 탐색을 진행
            3. S초 후, (x, y) 위치의 바이러스 번호 출력
            
            ```python
            from collections import deque
            import sys
            input=sys.stdin.readline
            
            def bfs():
                q=deque(virus)
                while q:
                    num, a, b, time= q.popleft()
                    if time==s:
                        break
                    
                    for i in range(4):
                        na, nb=a+dx[i], b+dy[i]
                        if 0<=na<n and 0<=nb<n and array[na][nb]==0:
                            array[na][nb]=num
                            q.append((num, na, nb, time+1))
                
                return print(array[x-1][y-1])
            
            n,k=map(int,input().split())
            array=[list(map(int,input().split())) for _ in range(n)]
            virus=[]
            for i in range(n): # 바이러스들 저장
                for j in range(n):
                    if array[i][j]!=0:
                        virus.append((array[i][j],i,j,0))
            virus.sort() # 바이러스 번호 오름차순 정렬
            
            s,x,y=map(int,input().split())
            dx=[-1,1,0,0]
            dy=[0,0,-1,1]
            
            bfs()
            ```

    2. 풀이를 본 후
        
        풀이의 방법도 거의 유사했다.
        
        ```python
        from collections import deque
        
        n, k = map(int, input().split())
        
        graph = [] # 전체 보드 정보를 담는 리스트
        data = [] # 바이러스에 대한 정보를 담는 리스트
        
        for i in range(n):
            # 보드 정보를 한 줄 단위로 입력
            graph.append(list(map(int, input().split())))
            for j in range(n):
                # 해당 위치에 바이러스가 존재하는 경우
                if graph[i][j] != 0:
                    # (바이러스 종류, 시간, 위치 X, 위치 Y) 삽입
                    data.append((graph[i][j], 0, i, j))
        
        # 정렬 이후에 큐로 옮기기 (낮은 번호의 바이러스가 먼저 증식하므로)
        data.sort()
        q = deque(data)
         
        target_s, target_x, target_y = map(int, input().split())
         
        # 바이러스가 퍼져나갈 수 있는 4가지의 위치
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]
        
        # 너비 우선 탐색(BFS) 진행
        while q:
            virus, s, x, y = q.popleft()
            # 정확히 s초가 지나거나, 큐가 빌 때까지 반복
            if s == target_s:
                break
            # 현재 노드에서 주변 4가지 위치를 각각 확인
            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]
                # 해당 위치로 이동할 수 있는 경우
                if 0 <= nx and nx < n and 0 <= ny and ny < n:
                    # 아직 방문하지 않은 위치라면, 그 위치에 바이러스 넣기
                    if graph[nx][ny] == 0:
                        graph[nx][ny] = virus
                        q.append((virus, s + 1, nx, ny))
        
        print(graph[target_x - 1][target_y - 1])
        ```
        
    3. 해결한 후
        
        나의 풀이가 시간이 조금 더 빨랐다^^
        
        항상 반복문의 종료 조건, 탈출 조건을 조금 더 신중하게 고려하기 

- 기출 문제 4: 괄호 변환

    1. 내 풀이
        1. 우선 앞에서부터 여는괄호와 닫는 괄호의 수를 세서 같아질 때까지가 u, 그리고 나머지가 v
        2. 그 후, 조건에 맞게 함수 구현
        
        재귀적으로 탐색하면서 원본 문자열을 어떻게 처리할지 모르겠어서 실패
        
        ```python
         from collections import deque
        
        def solution(p):
          if p=='':
            return ''
            
          answer = ''
          u,v='',''
          open_count,close_count=0,0
          
          for i in p:
              if p[i]=='(':
                  open_count+=1
              else:
                  close_count+=1
          
              if open_count-close_count==0:
                  u=p[:i+1]
                  v=p[i+1:]
                  break
        
          if u[0]=='(':
            answer=answer+u
            solution(v)
          else:
            temp=
                
          return answer
        
        p=input()
        ```
        
    2. 풀이를 본 후
        
        >주어진 조건을 그대로 실수 없이 차근차근 구현하는 것이 핵심이었다. 
        >
        >균형 잡힌 괄호 문자열 인덱스 반환, 올바른 괄호 문자열 판단 함수들을 구현하는 것이 중요했다.
        >
        >그 후, 재귀 함수 solution을 이용해서 조건대로 구현하고 그저 더해주면 된다. 원본 문자열에 재귀 함수를 더해주면 재귀적으로 탐색하면서 최종적인 문자열이 구해진다…
        
        ```python
        # "균형잡힌 괄호 문자열"의 인덱스 반환
        def balanced_index(p):
            count = 0 # 왼쪽 괄호의 개수
            for i in range(len(p)):
                if p[i] == '(':
                    count += 1
                else:
                    count -= 1
                if count == 0:
                    return i
        
        # "올바른 괄호 문자열"인지 판단
        def check_proper(p):
            count = 0 # 왼쪽 괄호의 개수
            for i in p:
                if i == '(':
                    count += 1
                else:
                    if count == 0: # 쌍이 맞지 않는 경우에 False 반환
                        return False
                    count -= 1
            return True # 쌍이 맞는 경우에 True 반환
        
        def solution(p):
            answer = ''
            if p == '':
                return answer
            index = balanced_index(p)
            u = p[:index + 1]
            v = p[index + 1:]
            # "올바른 괄호 문자열"이면, v에 대해 함수를 수행한 결과를 붙여 반환
            if check_proper(u):
                answer = u + solution(v)
            # "올바른 괄호 문자열"이 아니라면 아래의 과정을 수행
            else:
                answer = '('
                answer += solution(v)
                answer += ')'
                u = list(u[1:-1]) # 첫 번째와 마지막 문자를 제거
                for i in range(len(u)):
                    if u[i] == '(':
                        u[i] = ')'
                    else:
                        u[i] = '('
                answer += "".join(u)
            return answer
        ```

- 기출 문제 5:  연산자 끼워넣기

    1. 내 풀이
        1. DFS로 탐색하면서 연산자의 개수가 남아있는 것들을 탐색하도록 했다.
        2. 연산을 N-1번 하면 최소, 최대값을 갱신시켜줬다.
        
        하지만 연산을 이어나가는 처리를 제대로 하지 못했다.
        
        ```python
        from collections import deque
        
        def solve(sum,plus,minus,multi,divide,idx):
          global minimum,maximum
          if idx==N:
            minimum=min(sum,minimum)
            maximum=max(sum,maximum)
            return
            
          for i in range(idx,N):
            if plus>0:
              solve(sum+A[i],plus-1,minus,multi,divide,idx+1)
            if minus>0:
              solve(sum-A[i],plus,minus-1,multi,divide,idx+1)
            if multi>0:
              solve(sum*A[i],plus,minus,multi-1,divide,idx+1)
            if divide>0:
              solve(sum/A[i],plus,minus,multi,divide-1,idx+1)
        
          return
        
        N=int(input())
        A=list(map(int,input().split()))
        op=list(map(int,input().split()))
        
        minimum=1e9
        maximum=-1e9
        solve(A[0],op[0],op[1],op[2],op[3],1)
        print(maximum)
        print(minimum)
        ```
        
    2. 풀이를 본 후
        
        거의 맞았지만 안쪽에서 반복문을 돌 필요가 없었다. idx값이 어차피 증가하면서 연산할 값을 지정해주기 때문이다. 조금 신중하지 못했다.
        
        ```python
        N = int(input())
        num = list(map(int, input().split()))
        op = list(map(int, input().split()))  # +, -, *, //
        
        maximum = -1e9
        minimum = 1e9
        
        def dfs(depth, total, plus, minus, multiply, divide):
            global maximum, minimum
            if depth == N:
                maximum = max(total, maximum)
                minimum = min(total, minimum)
                return
        
            if plus:
                dfs(depth + 1, total + num[depth], plus - 1, minus, multiply, divide)
            if minus:
                dfs(depth + 1, total - num[depth], plus, minus - 1, multiply, divide)
            if multiply:
                dfs(depth + 1, total * num[depth], plus, minus, multiply - 1, divide)
            if divide:
                dfs(depth + 1, int(total / num[depth]), plus, minus, multiply, divide - 1)
        
        dfs(1, num[0], op[0], op[1], op[2], op[3])
        print(maximum)
        print(minimum)
        ```
        
        내 예전 풀이
        {:.figcaption}
        
        ```python
        n = int(input())
        # 연산을 수행하고자 하는 수 리스트
        data = list(map(int, input().split()))
        # 더하기, 빼기, 곱하기, 나누기 연산자 개수
        add, sub, mul, div = map(int, input().split())
        
        # 최솟값과 최댓값 초기화
        min_value = 1e9
        max_value = -1e9
        
        # 깊이 우선 탐색 (DFS) 메서드
        def dfs(i, now):
            global min_value, max_value, add, sub, mul, div
            # 모든 연산자를 다 사용한 경우, 최솟값과 최댓값 업데이트
            if i == n:
                min_value = min(min_value, now)
                max_value = max(max_value, now)
            else:
                # 각 연산자에 대하여 재귀적으로 수행
                if add > 0:
                    add -= 1
                    dfs(i + 1, now + data[i])
                    add += 1
                if sub > 0:
                    sub -= 1
                    dfs(i + 1, now - data[i])
                    sub += 1
                if mul > 0:
                    mul -= 1
                    dfs(i + 1, now * data[i])
                    mul += 1
                if div > 0:
                    div -= 1
                    dfs(i + 1, int(now / data[i])) # 나눌 때는 나머지를 제거
                    div += 1
        
        # DFS 메서드 호출
        dfs(1, data[0])
        
        # 최댓값과 최솟값 차례대로 출력
        print(max_value)
        print(min_value)
        ```
        
        책의 풀이
        {:.figcaption}

        
        >책의 풀이도 내 예전 풀이와 유사했다. DFS 문제 해결시 종료조건, 탐색하는 기본 방식을 생각하면서 풀어야겠다.
        
- 기출 문제 6: 감시 피하기

    1. 내 풀이
        1. 전체 복도에서 완전 탐색을 하여 벽을 3개 설치
        2. 그 때마다 선생님을 상하좌우로 이동시켜서 벽 or 범위 밖 전까지 만나는 좌표들을 ‘T’로 변경
        3. 그 후, 학생의 수의 변화 유무로 판단
        
        푸는데 1시간 반 정도가 걸렸다…
        
        ```python
        def move(x,y,dx,dy):
          while True:
            nx,ny=x+dx,y+dy
            if nx<0 or nx>=N or ny<0 or ny>=N:
              break
            if 0<=nx<N and 0<=ny<N:
              if temp[nx][ny]=='O':
                break
              else:
                temp[nx][ny]='T'
                x,y=nx,ny
              
        def solve(count):
          global answer
          if count==3:
            s_count=0
            for i in range(N):
              for j in range(N):
                temp[i][j]=land[i][j]
        
            for i in range(4):
              for t in teacher:
                move(t[0],t[1],dx[i],dy[i])
        
            for i in range(N):
              for j in range(N):
                if temp[i][j]=='S':
                  s_count+=1
        
            if s_count==len(student):
              answer='YES'
            return
            
          for i in range(N):
            for j in range(N):
              if land[i][j]=='X':
                count+=1
                land[i][j]='O'
                solve(count)
                count-=1
                land[i][j]='X'
                
        N=int(input())
        land=[list(input().split()) for _ in range(N)]
        temp=[['']*N for _ in range(N)]
        student=[]
        teacher=[]
        answer='NO'
        
        dx=[-1,1,0,0]
        dy=[0,0,-1,1]
        
        for i in range(N):
          for j in range(N):
            if land[i][j]=='S':
              student.append((i,j))
            if land[i][j]=='T':
              teacher.append((i,j))
              
        solve(0)
        print(answer)
        ```
        
    2. 풀이를 본 후
        
        조합과 감시에 걸렸는지를 판단하는 방식으로 진행했다. 감시할 방향 선정과 조합을 이용해서 빈 공간을 바꿨다가 다시 돌려놓는 방법이다. 이런 방법도 알아 놔야 할 것 같다. 
        
        ```python
        from itertools import combinations
        
        n = int(input()) # 복도의 크기
        board = [] # 복도 정보 (N x N)
        teachers = [] # 모든 선생님 위치 정보
        spaces = [] # 모든 빈 공간 위치 정보
        
        for i in range(n):
            board.append(list(input().split()))
            for j in range(n):
                # 선생님이 존재하는 위치 저장
                if board[i][j] == 'T':
                    teachers.append((i, j))
                # 장애물을 설치할 수 있는 (빈 공간) 위치 저장
                if board[i][j] == 'X':
                    spaces.append((i, j))
        
        # 특정 방향으로 감시를 진행 (학생 발견: True, 학생 미발견: False)
        def watch(x, y, direction):
            # 왼쪽 방향으로 감시
            if direction == 0:
                while y >= 0:
                    if board[x][y] == 'S': # 학생이 있는 경우
                        return True
                    if board[x][y] == 'O': # 장애물이 있는 경우
                        return False
                    y -= 1
            # 오른쪽 방향으로 감시
            if direction == 1:
                while y < n:
                    if board[x][y] == 'S': # 학생이 있는 경우
                        return True
                    if board[x][y] == 'O': # 장애물이 있는 경우
                        return False
                    y += 1
            # 위쪽 방향으로 감시
            if direction == 2:
                while x >= 0:
                    if board[x][y] == 'S': # 학생이 있는 경우
                        return True
                    if board[x][y] == 'O': # 장애물이 있는 경우
                        return False
                    x -= 1
            # 아래쪽 방향으로 감시
            if direction == 3:
                while x < n:
                    if board[x][y] == 'S': # 학생이 있는 경우
                        return True
                    if board[x][y] == 'O': # 장애물이 있는 경우
                        return False
                    x += 1
            return False
        
        # 장애물 설치 이후에, 한 명이라도 학생이 감지되는지 검사
        def process():
            # 모든 선생의 위치를 하나씩 확인
            for x, y in teachers:
                # 4가지 방향으로 학생을 감지할 수 있는지 확인
                for i in range(4):
                    if watch(x, y, i):
                        return True
            return False
        
        find = False # 학생이 한 명도 감지되지 않도록 설치할 수 있는지의 여부
        
        # 빈 공간에서 3개를 뽑는 모든 조합을 확인
        for data in combinations(spaces, 3):
            # 장애물들을 설치해보기
            for x, y in data:
                board[x][y] = 'O'
            # 학생이 한 명도 감지되지 않는 경우
            if not process():
                # 원하는 경우를 발견한 것임
                find = True
                break
            # 설치된 장애물을 다시 없애기
            for x, y in data:
                board[x][y] = 'X'
        
        if find:
            print('YES')
        else:
            print('NO')
        ```
        
    3. 해결한 후
        
        사실 처음의 아이디어는 책의 풀이와 같았지만 구현에서 어려움이 있어 변경했다. 최대한 아이디어를 구현해낼 수 있도록 연습이 필요하다.
        
        책의 풀이가 시간이 훨씬 적게 걸린다…

- 기출 문제 7: 인구 이동

    1. 내 풀이
        1. BFS 탐색을 통해 0,0의 위치부터 조건에 해당하면 임의의 리스트 temp에 저장
        2. 그 후, temp의 저장된 위치들의 값을 평균 값으로 바꿈
        3. 그 후, 인구 이동이 없을 때까지 반복
        
        하지만 시간 초과 발생
        
        ```python
        from collections import deque
        
        def check(x, y, nx, ny):
          if L <= abs(land[x][y] - land[nx][ny]) <= R:
            return True
          else:
            return False
        
        def solve():
          global answer, flag
          q = deque()
          q.append((0, 0))
          while q:
            x, y = q.popleft()
            temp.append((x, y))
            for i in range(4):
              nx, ny = x + dx[i], y + dy[i]
              if 0 <= nx < N and 0 <= ny < N and ((nx, ny) not in temp):
                if check(x, y, nx, ny):
                  q.append((nx, ny))
        
          if len(temp) == 0:
            flag = False
          else:
            sum = 0
            for x, y in temp:
              sum += land[x][y]
            average = sum / len(temp)
            for x, y in temp:
              land[x][y] = average
            answer += 1
        
        N, L, R = map(int, input().split())
        land = [list(map(int, input().split())) for _ in range(N)]
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        temp = []
        answer = 0
        flag = True
        
        while True:
          if flag is False:
            print(answer)
            break
            
          solve()
        ```
        
    2. 풀이를 본 후
        
        연합의 구분, 처리 되지 않은 나라의 처리가 내 풀이에는 없었다. 이 두가지를 추가하여 진행했다면 해결 가능 했을 것 같다. 항상 필요한 조건을 구현하도록 해야겠다.
        
        ```python
         from collections import deque
        
        # 땅의 크기(N), L, R 값을 입력받기
        n, l, r = map(int, input().split())
        
        # 전체 나라의 정보(N x N)를 입력 받기
        graph = []
        for _ in range(n):
            graph.append(list(map(int, input().split())))
        
        dx = [-1, 0, 1, 0]
        dy = [0, -1, 0, 1]
        
        # 특정 위치에서 출발하여 모든 연합을 체크한 뒤에 데이터 갱신
        def process(x, y, index):
            # (x, y)의 위치와 연결된 나라(연합) 정보를 담는 리스트
            united = []
            united.append((x, y))
            # 너비 우선 탐색 (BFS)을 위한 큐 라이브러리 사용
            q = deque()
            q.append((x, y))
            union[x][y] = index # 현재 연합의 번호 할당
            summary = graph[x][y] # 현재 연합의 전체 인구 수
            count = 1 # 현재 연합의 국가 수
            # 큐가 빌 때까지 반복(BFS)
            while q:
                x, y = q.popleft()
                # 현재 위치에서 4가지 방향을 확인하며
                for i in range(4):
                    nx = x + dx[i]
                    ny = y + dy[i]
                    # 바로 옆에 있는 나라를 확인하여
                    if 0 <= nx < n and 0 <= ny < n and union[nx][ny] == -1:
                        # 옆에 있는 나라와 인구 차이가 L명 이상, R명 이하라면
                        if l <= abs(graph[nx][ny] - graph[x][y]) <= r:
                            q.append((nx, ny))
                            # 연합에 추가하기
                            union[nx][ny] = index
                            summary += graph[nx][ny]
                            count += 1
                            united.append((nx, ny))
            # 연합 국가끼리 인구를 분배
            for i, j in united:
                graph[i][j] = summary // count
        
        total_count = 0
        
        # 더 이상 인구 이동을 할 수 없을 때까지 반복
        while True:
            union = [[-1] * n for _ in range(n)]
            index = 0
            for i in range(n):
                for j in range(n):
                    if union[i][j] == -1: # 해당 나라가 아직 처리되지 않았다면
                        process(i, j, index)
                        index += 1
            # 모든 인구 이동이 끝난 경우
            if index == n * n:
                break
            total_count += 1
        
        # 인구 이동 횟수 출력
        print(total_count)
        ```

- 기출 문제 8: 블록 이동하기
    1. 내 풀이
        1. 상하좌우, 회전 이동을 나누어서 BFS로 시행
        2. N,N 좌표에 도달하면 시간 출력
        
        하지만 test case 4개만 정답, 나머지 10개 틀림
        
        ```python
        from collections import deque
        
        def solution(board):
            answer = 0
            N=len(board)
            visited1=[[0]*N for _ in range(N)]
            visited2=[[0]*N for _ in range(N)]
            dx1=[-1,1,0,0]
            dx2=[-1,-1,1,1]
            dy1=[0,0,-1,1]
            dy2=[-1,1,-1,1]
        
            q=deque([(0,0,0,1,0)])
            visited1[0][0],visited2[0][1]=1,1
            while q:
                x1,y1,x2,y2,time=q.popleft()
                if (x1==N-1 and y1==N-1) or (x2==N-1 and y2==N-1):
                    answer=time
                    break
                for i in range(4):
                    nx1,ny1,nx2,ny2=x1+dx1[i],y1+dy1[i],x2+dx1[i],y2+dy1[i]
                    if 0<=nx1<N and 0<=ny1<N and 0<=nx2<N and 0<=ny2<N:
                        if board[nx1][ny1]==0 and board[nx2][ny2]==0 and visited1[nx1][ny1]==0 and visited2[nx2][ny2]==0:
                            visited1[nx1][ny1],visited2[nx2][ny2]=1,1
                            q.append((nx1,ny1,nx2,ny2,time+1))
        
                for i in range(4):
                    nx1,ny1,nx2,ny2=x1+dx2[i],y1+dy2[i],x2+dx2[i],y2+dy2[i]
                    if 0<=nx1<N and 0<=ny1<N and abs(nx1-x2)+abs(ny1-y2)==1:
                        if board[nx1][ny1]==0 and (board[nx1][y1]==0 or board[x1][ny1]==0) and visited1[nx1][ny1]==0:
                            visited1[nx1][ny1]=1
                            q.append((nx1,ny1,x2,y2,time+1))
                    if 0<=nx2<N and 0<=ny2<N and abs(x1-nx2)+abs(y1-ny2)==1:
                        if board[nx2][ny2]==0 and (board[nx2][y2]==0 or board[x2][ny2]==0) and visited2[nx2][ny2]==0:
                            visited2[nx2][ny2]=1
                            q.append((x1,y1,nx2,ny2,time+1))  
        
            return answer
        ```
        
    2. 풀이를 본 후
    
    이동 방법에 따라 나누고, 방문 처리를 세트 형식으로 저장하여 처리했다. 조금 더 세밀하게 탐색 구현을 해야겠다.
    
    ```python
     from collections import deque
    
    def get_next_pos(pos, board):
        next_pos = [] # 반환 결과 (이동 가능한 위치들)
        pos = list(pos) # 현재 위치 정보를 리스트로 변환 (집합 → 리스트)
        pos1_x, pos1_y, pos2_x, pos2_y = pos[0][0], pos[0][1], pos[1][0], pos[1][1]
        # (상, 하, 좌, 우)로 이동하는 경우에 대해서 처리
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        for i in range(4):
            pos1_next_x, pos1_next_y, pos2_next_x, pos2_next_y = pos1_x + dx[i], pos1_y + dy[i], pos2_x + dx[i], pos2_y + dy[i]
            # 이동하고자 하는 두 칸이 모두 비어 있다면
            if board[pos1_next_x][pos1_next_y] == 0 and board[pos2_next_x][pos2_next_y] == 0:
                next_pos.append({(pos1_next_x, pos1_next_y), (pos2_next_x, pos2_next_y)})
        # 현재 로봇이 가로로 놓여 있는 경우
        if pos1_x == pos2_x:
            for i in [-1, 1]: # 위쪽으로 회전하거나, 아래쪽으로 회전
                if board[pos1_x + i][pos1_y] == 0 and board[pos2_x + i][pos2_y] == 0: # 위쪽 혹은 아래쪽 두 칸이 모두 비어 있다면
                    next_pos.append({(pos1_x, pos1_y), (pos1_x + i, pos1_y)})
                    next_pos.append({(pos2_x, pos2_y), (pos2_x + i, pos2_y)})
        # 현재 로봇이 세로로 놓여 있는 경우
        elif pos1_y == pos2_y:
            for i in [-1, 1]: # 왼쪽으로 회전하거나, 오른쪽으로 회전
                if board[pos1_x][pos1_y + i] == 0 and board[pos2_x][pos2_y + i] == 0: # 왼쪽 혹은 오른쪽 두 칸이 모두 비어 있다면
                    next_pos.append({(pos1_x, pos1_y), (pos1_x, pos1_y + i)})
                    next_pos.append({(pos2_x, pos2_y), (pos2_x, pos2_y + i)})
        # 현재 위치에서 이동할 수 있는 위치를 반환
        return next_pos
    
    def solution(board):
        # 맵의 외곽에 벽을 두는 형태로 맵 변형
        n = len(board)
        new_board = [[1] * (n + 2) for _ in range(n + 2)]
        for i in range(n):
            for j in range(n):
                new_board[i + 1][j + 1] = board[i][j]
        # 너비 우선 탐색(BFS) 수행
        q = deque()
        visited = []
        pos = {(1, 1), (1, 2)} # 시작 위치 설정
        q.append((pos, 0)) # 큐에 삽입한 뒤에
        visited.append(pos) # 방문 처리
        # 큐가 빌 때까지 반복
        while q:
            pos, cost = q.popleft()
            # (n, n) 위치에 로봇이 도달했다면, 최단 거리이므로 반환
            if (n, n) in pos:
                return cost
            # 현재 위치에서 이동할 수 있는 위치 확인
            for next_pos in get_next_pos(pos, new_board):
                # 아직 방문하지 않은 위치라면 큐에 삽입하고 방문 처리
                if next_pos not in visited:
                    q.append((next_pos, cost + 1))
                    visited.append(next_pos)
        return 0
    ```

## 출처

- 이것이 취업을 위한 코딩테스트다 with 파이썬