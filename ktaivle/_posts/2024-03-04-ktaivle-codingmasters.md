---
layout: post
title: 코딩 테스트 연습 | 1차 코딩 마스터스
description: KT AIVLE SCHOOL 5기 1차 코딩 마스터스 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}

## 기초

### 문제 8469. 비밀번호 찾기

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8469.png)
    
2. 내 풀이
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
    
3. 해결한 후
    
    쉬운 문제이다
        
    
### 문제 8471: 8진수와 16진수

1.  문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8471.png)
    
2. 내 풀이
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
    
3. 해결한 후
    
    변수 사용에서 실수를 해서 시간이 조금 걸렸다
        
    
### 문제 8481. 우리반 아이큐왕은

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8481.png)
    
2. 내 풀이
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
    
3. 해결한 후
    
    IQ가 같을 떄 먼저 입력된 학생을 앞에 세우는 부분에서 조금 막혔다
        
    
### 문제 8487. 최대공약수

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8487.png)
    
2. 내 풀이
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
    
3. 해결한 후
    
    유클리드 호제법을 다시 한번 상기시킬 수 있었다

## 초급

### 문제 8508. 신입사원 채용

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8508.png)
    
2. 내 풀이
    1. 우선 (서류 점수, 면접 점수, 초기 인덱스) 를 리스트에 저장
    2. 그 후, 내림차순으로 정렬 후, 조건에 따라서 등수를 매겨준다
    
    먼저 정렬을 하고 조건에 따라 등수를 매겨줘야 했다…
    
    ```python
    import sys
    input = sys.stdin.readline
    
    n = int(input())
    array = []
    for i in range(n):
        x, y = map(int, input().split())
        array.append((x, y, i))
    
    array.sort(reverse = True)
    
    compare = []
    answer = [1] * n
    for i in range(n - 1):
        a1, a2, index1 = array[i]
        for j in range(i + 1, n):
            b1, b2, index2 = array[j]
            if a2 > b2: # A > B
                answer[index2] += 1
            elif a2 == b2:
                if a1 == b1:
                    compare.append((index1, index2))
                else:
                    answer[index2] += 1
            else:
                if a1 == b1:
                    answer[index1] += 1
                else:
                    compare.append((index1, index2))
                
    for i, j in compare:
        if answer[i] != answer[j]:
            min_num = min(answer[i], answer[j])
            answer[i], answer[j] = min_num, min_num
        
    for a in answer:
        print(a, end = ' ')
    print()
    ```
    
3. 해결한 후
    
    항상 정렬과 관련된 문제는 초기에 정렬이 대부분 필요하다
    

### 문제 8509. 커트라인 정하기

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8509.png)
    
2. 내 풀이
    1. 각 학생들의 점수들을 기준으로 합격 학생의 수를 구함
    2. 조건을 만족할 때마다 리스트에 넣는다
    3. 최종적으로 합격자, 점수 순으로 내림차순 정렬하여 맨 앞 요소의 점수 출력
    4. 리스트가 비어있으면 점수 최솟값 -1 출력
    
    ```python
    import sys
    input = sys.stdin.readline
    
    n, k = map(int, input().split())
    array = list(map(int, input().split()))
    answer = []
    
    for a in array:
        count = 0
        for i in range(n):
            if i == 0:
                if (array[i] >= a) or (array[i + 1] >= a):
                    count += 1
            elif i == n - 1:
                if (array[i] >= a) or (array[i - 1] >= a):
                    count += 1
            else:
                if (array[i] >= a) or (array[i + 1] >= a) or (array[i - 1] >= a):
                    count += 1
        
        if count <= k:
            answer.append((count, a))
            
    
    answer.sort(reverse = True)
    
    if not answer:
        print(min(array) - 1)
    else:
        print(answer[0][1])
    ```
    
3. 해결한 후
    
    너무 어렵게 접근했다
    

### 문제 8511. 가우스와 정다각형

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8511.png)
    
2. 내 풀이
    1. 우선 페르마 소수로 최대한 나누어 준다
    2. 그 후, 나누어진 결과가 2의 거듭 제곱인지 확인해준다
    3. 최종 조건에 맞게 결과 출력
    
    ```python
    import sys
    input = sys.stdin.readline
    
    k = int(input())
    array = [3, 5, 17, 257, 65537]
    visited = [0] * 5
    
    while True:
        temp = False
        
        for i in range(5):
            if (k % array[i] == 0) and (visited[i] == 0):
                visited[i] += 1
                temp = True
                k //= array[i]
                break
                
        if not temp:
            break      
    
    if k == 1:
        print('YES')
    else:
        x = 4 
        array2 = []
        while x <= 10 ** 18:
            array2.append(x)
            x *= 2
        
        temp2 = False
        for a2 in array2:
            if k == a2:
                temp = True
                break
        
        if temp:
            print('YES')
        else:
            print('NO')
    ```
    
3. 해결한 후
    
    처음에 문제 이해를 잘못했다

## 중급

### 문제 8519. 태국 택시

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8519.png)
    
2. 내 풀이
    1. 크루스칼 알고리즘을 이용하여 모든 마을 방문하기 위한 최소 금액 구함
    
    ```python
    import sys
    input = sys.stdin.readline
    
    def find_parent(parent, x):
        if parent[x] != x:
            parent[x] = find_parent(parent, parent[x])
        return parent[x]
    
    def union_parent(parent, a, b):
        a = find_parent(parent, a)
        b = find_parent(parent, b)
        if a < b:
            parent[b] = a
        else:
            parent[a] = b 
    
    n, m = map(int, input().split())
    edges = []
    for i in range(m):
        a, b, c = map(int, input().split())
        edges.append((c, a, b))
    
    edges.sort()
    
    parent = [x for x in range(n + 1)]
    result = 0
    for c, a, b in edges:
        if find_parent(parent, a) != find_parent(parent, b):
            union_parent(parent, a, b)
            result += c
    
    print(result)
    ```
    
3. 해결한 후
    
    크루스칼 알고리즘을 복습할 좋은 문제이다.
    

### 문제 8521. 효율적인 화폐 구성

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8521.png)
    
2. 내 풀이
    1. 우선 구성된 화폐의 수를 1로 저장
    2. 그 후, m 이하의 수중에서 화폐로 만들 수 있는 수이면 비교하며 최솟값을 갱신
    3. 최종 값 출력
    
    ```python
    n, m = map(int, input().split())
    coin_values = []
    for _ in range(n):
        coin_values.append(int(input()))
    
    dp = [10001] * (m + 1)
    dp[0] = 0
    
    for value in coin_values:
        if value <= m:
            dp[value] = 1
    
    for i in range(1, m + 1):
        for value in coin_values:
            if i - value >= 0:
                dp[i] = min(dp[i], dp[i - value] + 1)
    
    if dp[m] == 10001:
        print(-1)
    else:
        print(dp[m])
    ```
    
3. 해결한 후
    
    다이나믹 프로그래밍을 이용하여 풀 때는 항상 값을 갱신시키는 조건을 생각해야한다

### 문제 8522. 먹보 수민이

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8522.png)
    
2. 내 풀이
    1. 편의점 정보를 거리순으로 오름차순 정렬
    2. 그 후, 현재 이동 가능한 경우에서 가장 멀리 갈 수 있는 편의점 선택
    3. 최종 지점에 도달하면 성공, 중간에 실패하면 도달 실패
    
    ```python
    import sys
    input = sys.stdin.readline
    
    n = int(input())
    array = [list(map(int, input().split())) for _ in range(n)]
    l, p = map(int, input().split())
    
    array.sort()
    start = 0
    count = 0
    idx = 0
    answer = True
    
    while p < l:
        maximum = 0
        for i in range(idx, n):
            if array[i][0] > p:
                break
            if (start < array[i][0] <= p) and (array[i][1] > maximum):
                idx = i + 1
                maximum = array[i][1]
                start = array[i][0]
                
        if maximum == 0:
            answer = False
            break
        
        p += maximum
        count += 1
        
    if answer:
        print(count)
    else:
        print(-1)       
    ```
    
3. 해결한 후
    
    중간에 `idx` 에 1을 더해 주지 않아줬었다…
    

### 문제 8523. 숫자 맞추기

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8523.png)
    
2. 내 풀이
    1. BFS를 통해서 각 연산 방법으로 진행 하다가 목표 숫자에 도달하면 연산 횟수 출력
    2. K ≥ N인 경우 처리를 따로 추가해줬다
    
    ```python
    import sys
    input = sys.stdin.readline
    from collections import deque
    
    def min_operations_to_target(N, K):
        if K >= N:  # 시작 숫자가 목표 숫자보다 크거나 같으면 바로 결과 반환
            return K - N
        
        visited = [0] * (2 * N + 1)  # 범위를 넓힌 visited 배열 생성
        queue = deque([(K, 0)])  # 시작 숫자와 횟수(0)를 큐에 삽입
        visited[K] = 1  # 시작 숫자를 방문했다고 표시
    
        while queue:
            current_num, count = queue.popleft()  # 현재 숫자와 횟수를 가져옴
    
            # 목표 숫자에 도달했을 때 횟수를 반환
            if current_num == N:
                return count
            
            # 현재 숫자를 조작하여 나올 수 있는 다음 숫자를 계산하고 큐에 삽입
            for next_num in (current_num - 1, current_num + 1, current_num * 2):
                # 숫자가 범위를 벗어나거나 이미 방문한 경우 건너뜀
                if 0 <= next_num <= 2 * N and not visited[next_num]:
                    visited[next_num] = 1  # 방문했다고 표시
                    queue.append((next_num, count + 1))  # 다음 숫자와 횟수(현재 횟수 + 1)를 큐에 삽입
        
        # 목표 숫자에 도달할 수 없는 경우 -1 반환
        return -1
    
    # 입력 받기
    N, K = map(int, input().split())
    
    # 결과 출력
    print(min_operations_to_target(N, K))
    ```
    
3. 해결한 후
    
    K ≥ N인 경우 처리를 제대로 하지 못했다, BFS를 이용한 풀이를 생각해 놓기
    

### 문제 8525. 타격왕 정우성

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8525.png)
    
2. 내 풀이
    1. 더 해야할 경기의 수를 이진 탐색으로 찾는다
    
    ```python
    import sys
    input = sys.stdin.readline
    import math
    MAXIMUM = 1000000000
    
    x, y = map(int, input().split())
    prev = math.floor((y / x) * 100) / 100
    
    start, end = 1, MAXIMUM 
    while start <= end:
        mid = (start + end) // 2
        hit = (y + mid) / (x + mid)
        present = math.floor(hit * 100) / 100
        
        if prev < present:
            end = mid - 1
        else:
            start = mid + 1
    
    if start <= MAXIMUM:
        print(start) 
    else:
        print(-1)    
    ```
    
3. 해결한 후
    
    반복 횟수가 지나치게 많거나 입력의 수가 많으면 이진 탐색 의심하기, 이진 탐색 마지막 조건 처리 신중하게 생각하기
    

### 문제 8527. 리버스 게임

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8527.png)
    
2. 내 풀이
    1. 행과 열을 뒤집는 조합의 계산을 수행한다
    2. 그 후, 뒤집은 결과마다 흰 바둑돌의 수를 계산한다
    
    ```python
    def min_white_stones(N, board):
        min_white = float('inf')
        for row_flip in range(1 << N):  # 모든 행 뒤집기 조합
            for col_flip in range(1 << N):  # 모든 열 뒤집기 조합
                white = 0
                for i in range(N):
                    for j in range(N):
                        stone = board[i][j]
                        if (row_flip & (1 << i)) != 0:  # 행 뒤집기
                            stone = 'B' if stone == 'W' else 'W'
                        if (col_flip & (1 << j)) != 0:  # 열 뒤집기
                            stone = 'B' if stone == 'W' else 'W'
                        if stone == 'W':
                            white += 1
                min_white = min(min_white, white)
        return min_white
    
    # 입력값 받기
    N = int(input())
    board = [input().strip() for _ in range(N)]
    
    # 흰색 바둑돌의 최소 개수 계산
    min_white_stones = min_white_stones(N, board)
    
    # 결과 출력
    print(min_white_stones)
    ```
    
3. 해결한 후
    
    행과 열을 뒤집는 모든 조합을 수행하는 방법밖에 생각이 나지 않았다… 다른 풀이도 필요할 것 같다…
    

### 문제 8529. 떡 하나 주면 안 잡아먹지

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8529.png)
    
2. 내 풀이
    1. 다익스트라 알고리즘을 이용하여 최소 떡의 양을 계산
    
    ```python
    import heapq
    
    def dijkstra(grid):
        INF = float('inf')
        distances = [[INF] * n for _ in range(n)]
        distances[0][0] = grid[0][0]  # 시작 지점
        
        # 우측, 하단 방향만 고려
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        
        heap = [(grid[0][0], 0, 0)]  # (떡의 양, 행, 열)
        
        while heap:
            d, row, col = heapq.heappop(heap)
            
            if distances[row][col] < d:
                continue
            
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < n and 0 <= nc < n:
                    nd = d + grid[nr][nc]
                    if nd < distances[nr][nc]:
                        distances[nr][nc] = nd
                        heapq.heappush(heap, (nd, nr, nc))
        
        return distances[n-1][n-1]
    
    # 입력 받기
    n = int(input())
    grid = [list(map(int, input().split())) for _ in range(n)]
    
    # 다익스트라 알고리즘으로 최소 떡의 양 계산
    result = dijkstra(grid)
    print(result)
    ```
    
3. 해결한 후
    
    최단거리 문제면 다익스트라 알고리즘, 플로이드 워샬 알고리즘 떠올리기
    

### 문제 8530. 사탕 꺼내기

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8530.png)
    
2. 내 풀이
    1. deque의 rotate를 이용하여 회전시키며 최소 회전 횟수 계산
    
    ```python
    import sys
    input = sys.stdin.readline
    
    from collections import deque
    
    def min_rotation(N, M, candies):
        queue = deque(range(1, N+1))
        idx = 0
        rotation = 0
    
        for candy in candies:
            while True:
                if queue[idx] == candy:
                    break
                idx += 1
                if idx == len(queue):
                    idx = 0
            rotation += min(idx, len(queue)-idx)
            queue.rotate(-idx)
            queue.popleft()
            idx = 0
    
        return rotation
    
    N, M = map(int, input().split())
    candies = list(map(int, input().split()))
    
    result = min_rotation(N, M, candies)
    print(result)
    
    ```
    
3. 해결한 후
    
    rotate 함수를 이용한 방법도 익혀 놔야겠다
    

### 문제 8532. 그림판

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8532.png)
    
2. 내 풀이
    1. BFS를 이용하여 하나의 영역을 구한 후 조건에 맞게 빨간색과 파란색의 수를 조절했다
    2. 그 후, 전체 개수 들을 출력
    
    ```python
    from collections import deque
    import sys
    input = sys.stdin.readline
    
    def bfs(i, j):
        r, b = 0, 0
        q = deque([(i, j)])
        if array[i][j] == 'A':
            r += 1
        if array[i][j] == 'B':
            b += 1
            
        while q:
            x, y = q.popleft()
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if 0<= nx < n and 0 <= ny < m and visited[nx][ny] == 0 and array[nx][ny] != 'X':
                    if array[nx][ny] == 'A':
                        r += 1
                    if array[nx][ny] == 'B':
                        b += 1
                        
                    visited[nx][ny] = 1
                    q.append((nx, ny))
        if b >= r:
            return 0, b
        else:
            return r, 0
    
    n, m = map(int, input().split())
    array = [input().rstrip() for _ in range(n)]
    visited = [[0] * m for _ in range(n)]
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    b_count, r_count = 0, 0
    
    for i in range(n):
        for j in range(m):
            if (array[i][j] != 'X') and (visited[i][j] == 0):
                visited[i][j] = 1
                r, b = bfs(i, j)
                r_count += r
                b_count += b
                
    print(r_count, b_count)
    ```
    
3. 해결한 후
    
    문제 조건을 제대로 읽자…
    

### 문제 8535. 같은 그래프

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8535.png)
    
2. 내 풀이
    1. 정점과 간선들을 집합으로 저장한다
    2. 그 후, 가능한 정점들의 교환을 수행 후 간선의 집합이 같은지 비교한
    
    ```python
    from itertools import permutations
    
    def parse_graph(n, m):
        vertices = set()
        edges = set()
        for _ in range(m):
            u, v = map(int, input().split())
            vertices.add(u)
            vertices.add(v)
            edges.add((u, v))
        return vertices, edges
    
    def are_graphs_equal(graph1, graph2):
        vertices1, edges1 = graph1
        vertices2, edges2 = graph2
        
        # 첫 번째 그래프의 정점 번호를 바꾸면서 두 번째 그래프와 간선 집합이 일치하는지 확인
        for perm in permutations(vertices1):
            perm_edges1 = set()
            for u, v in edges1:
                perm_edges1.add((perm[u - 1], perm[v - 1]))
                
            if {tuple(sorted(pair)) for pair in perm_edges1} == {tuple(sorted(pair)) for pair in edges2}:
                return "YES"
        
        return "NO"
    
    # 첫 번째 그래프 입력 받기
    n1, m1 = map(int, input().split())
    graph1 = parse_graph(n1, m1)
    
    # 두 번째 그래프 입력 받기
    n2, m2 = map(int, input().split())
    graph2 = parse_graph(n2, m2)
    
    # 두 그래프가 같은지 확인하여 출력
    print(are_graphs_equal(graph1, graph2))
    
    ```
    
3. 해결한 후
    
    순열과 집합 내에서 튜플의 순서가 다를 때 비교하는 방법을 배울 수 있었다
    

### 문제 8539. 아티스트 재승

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8539.png)
    
2. 내 풀이
    1. DFS를 이용하여 문자열의 첫번째 문자와 같은 부분을 찾으면 temp 값을 문자들로 채움
    2. 그 후, 같으면 개수를 추가한
    
    ```python
    import sys
    input = sys.stdin.readline
    
    def can_draw(x, y, idx):
        global flag
        if idx == k - 1:
            case = True
            for i in range(n):
                for j in range(m):
                    if temp[i][j] != grid[i][j]:
                        case = False
                        break
            if case:
                flag = True
            return
        
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < n and 0 <= ny < m and temp[nx][ny] == '.':
                temp[nx][ny] = word[idx + 1]
                can_draw(nx, ny, idx + 1)
                temp[nx][ny] = '.'
    
    word = input().rstrip()
    k = len(word)
    n, m = map(int, input().split())
    grid = [list(input().rstrip()) for _ in range(n)]
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    
    count = 0
    
    for i in range(n):
        for j in range(m):
            if grid[i][j] == word[0]:
                flag = False
                temp = [['.'] * m for _ in range(n)]
                temp[i][j] = word[0]
                can_draw(i, j, 0)
                if flag:
                    count += 1
    
    print(count)
    
    ```
    
3. 해결한 후
    
    DFS로의 풀이인 것을 알았지만 조금 오래 걸렸다… 또한, 조금 더 빠른 풀이를 시도했지만 자꾸 1개만 틀렸다…
    

### 문제 8542. 도어락

1. 문제 내용
    
    ![Untitled](/assets/img/blog/KT_AIVLE/codingmasters/1/8542.png)
    
2. 내 풀이
    1. dp를 이용하여 n번째 눌렀을 때의 경우의 수를 해당 열에 더해간다
    2. 최종 마지막 행의 합을 구한다
    
    ```python
    MOD = 1000000007
    
    def count_passwords(N, times):
        dp = [[0] * 9 for _ in range(N)]
        for j in range(9):
            dp[0][j] = 1
    
        for i in range(1, N):
            for j in range(9):
                if times[i - 1] == 1:
                    dp[i][j] += dp[i - 1][j]
                elif times[i - 1] == 2:
                    for k in range(9):
                        if j == k:
                            continue
                        x, y = abs(j // 3 - k // 3), abs(j % 3 - k % 3)
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            dp[i][j] += dp[i - 1][k]
                else:
                    for k in range(9):
                        if j == k:
                            continue
                        x, y = abs(j // 3 - k // 3), abs(j % 3 - k % 3)
                        if x >= 2 or y >= 2:
                            dp[i][j] += dp[i - 1][k]
    
        return sum(dp[N - 1]) % MOD
    
    N = int(input())
    times = list(map(int, input().split()))
    
    print(count_passwords(N, times))
    ```
    
3. 해결한 후
    
    dp의 풀이인 것을 알았지만 아이디어까지 오래 걸렸다
