---
layout: post
title: 이것이 취업을 위한 코딩테스트다 | 9. 2020년 상반기 삼성전자 기출문제 (Samsung 2020)
categories: 
  - cs
  - codingtest
description: 이것이 취업을 위한 코딩테스트다 with 파이썬 책을 참고하여 **2020년 상반기 삼성전자 기출문제**를 푼 내용입니다.
sitemap: false
---

2020년 상반기 삼성전자 기출문제

* this unordered seed list will be replaced by the toc
{:toc}

## 삼성 기출 문제

### 기출 문제 1: 아기 상어
1. 내 풀이
    1. 우선 BFS탐색으로 물고기의 위치를 찾으며 거리와 좌표를 저장
    2. 물고기를 정렬 후 가장 가까운 물고기로 이동
    3. 최종 시간 출력
    
    3시간 정도 해봤지만 실패…
    
    ```python
    import sys
    from collections import deque
    
    input = sys.stdin.readline
    
    def bfs(start_x, start_y):
        visited = [[0] * n for _ in range(n)]
        q = deque([(start_x, start_y, 0)])
        visited[start_x][start_y] = 1
        fishes = []
    
        while q:
            x, y, dist = q.popleft()
    
            if 0 < map[x][y] < size:
                fishes.append((dist, x, y))
    
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx < n and 0 <= ny < n and map[nx][ny] <= size and not visited[nx][ny]:
                    visited[nx][ny] = 1
                    q.append((nx, ny, dist + 1))
    
        return fishes
    
    def find_fish(x, y):
        nearest_fish = bfs(x, y)
        if nearest_fish:
            nearest_fish.sort()  # 거리 순으로 정렬
            return nearest_fish[0]
        else:
            return (1e9, 0, 0)
    
    n = int(input())
    map = [list(map(int, input().split())) for _ in range(n)]
    time, size, count = 0, 2, 0
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    x, y = 0, 0
    
    while True:
        dist, a, b = find_fish(x, y)
        if dist == 1e9:
            break
        else:
            map[x][y] = 0
            x, y = a, b
            map[x][y] = 9
            time += dist
            count += 1
    
            if count == size:
                count = 0
                size += 1
    
    print(time)
    ```
    
2. 풀이를 본 후
    
    우선, 거의 유사했지만 시작 위치를 0으로 바꾸고 시작하는 것과 현재 위치에서 BFS탐색을 통해 도달 가능한 모든 거리를 구하는 것이 달랐다. 도달 가능한 지점 중에서 먹을 수 있는 물고기들 중 가장 가까운 지점으로 이동하면 되었다…
    
    또한, 반복문으로 가장 가까운 지점을 찾을 때 가장 위와 왼쪽 부터 찾으므로 거리가 작을 때만 먹을 물고기 위치를 갱신하면 되었다…
    
    풀이의 흐름과 방식을 많이 봐둬야겠다…
    
    ```python
    from collections import deque
    INF = 1e9 # 무한을 의미하는 값으로 10억을 설정
    
    # 맵의 크기 N 입력
    n = int(input())
    
    # 전체 모든 칸에 대한 정보 입력
    array = []
    for i in range(n):
        array.append(list(map(int, input().split())))
    
    # 아기 상어의 현재 크기 변수와 현재 위치 변수
    now_size = 2
    now_x, now_y = 0, 0
    
    # 아기 상어의 시작 위치를 찾은 뒤에 그 위치엔 아무것도 없다고 처리
    for i in range(n):
        for j in range(n):
            if array[i][j] == 9:
                now_x, now_y = i, j
                array[now_x][now_y] = 0
    
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]
    
    # 모든 위치까지의 '최단 거리만' 계산하는 BFS 함수
    def bfs():
        # 값이 -1이라면 도달할 수 없다는 의미 (초기화)
        dist = [[-1] * n for _ in range(n)]
        # 시작 위치는 도달이 가능하다고 보며 거리는 0
        q = deque([(now_x, now_y)])
        dist[now_x][now_y] = 0
        while q:
            x, y = q.popleft()
            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]
                if 0 <= nx and nx < n and 0 <= ny and ny < n:
                    # 자신의 크기보다 작거나 같은 경우에 지나갈 수 있음
                    if dist[nx][ny] == -1 and array[nx][ny] <= now_size:
                        dist[nx][ny] = dist[x][y] + 1
                        q.append((nx, ny))
        # 모든 위치까지의 최단 거리 테이블 반환
        return dist
    
    # 최단 거리 테이블이 주어졌을 때, 먹을 물고기를 찾는 함수
    def find(dist):
        x, y = 0, 0
        min_dist = INF
        for i in range(n):
            for j in range(n):
                # 도달이 가능하면서 먹을 수 있는 물고기일 때
                if dist[i][j] != -1 and 1 <= array[i][j] and array[i][j] < now_size:
                    # 가장 가까운 물고기 한 마리만 선택
                    if dist[i][j] < min_dist:
                        x, y = i, j
                        min_dist = dist[i][j]
        if min_dist == INF: # 먹을 수 있는 물고기가 없는 경우
            return None
        else:
            return x, y, min_dist # 먹을 물고기의 위치와 최단 거리
    
    result = 0 # 최종 답안
    ate = 0 # 현재 크기에서 먹은 양
    
    while True:
        # 먹을 수 있는 물고기의 위치 찾기
        value = find(bfs())
        # 먹을 수 있는 물고기가 없는 경우, 현재까지 움직인 거리 출력
        if value == None:
            print(result)
            break
        else:
            # 현재 위치 갱신 및 이동 거리 변경
            now_x, now_y = value[0], value[1]
            result += value[2]
            # 먹은 위치에는 이제 아무것도 없도록 처리
            array[now_x][now_y] = 0
            ate += 1
            # 자신의 현재 크기 이상으로 먹은 경우, 크기 증가
            if ate >= now_size:
                now_size += 1
                ate = 0
    ```






## 출처

- 이것이 취업을 위한 코딩테스트다 with 파이썬