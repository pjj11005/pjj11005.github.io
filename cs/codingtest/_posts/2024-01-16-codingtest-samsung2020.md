---
layout: post
title: 9. 2020년 상반기 삼성전자 기출문제 (Samsung 2020)
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


### 기출 문제 2: 청소년 상어

1. 내 풀이
    1. array에 물고기 번호, 방향 저장
    2. fishes에 물고기 번호, 방향, 좌표 최소힙으로 저장
    3. 물고기 이동시키고 상어 이동으로 구현
    
    구현에 실패했다…
    
    ```python
    import heapq
    from collections import deque
    INF = 1e9 # 무한을 의미하는 값으로 10억을 설정
    
    array=[] # 좌표, 방향
    fishes=[] # 번호, 방향, 좌표
    
    dx = [-1, -1, 0, 1, 1, 1, 0, -1]
    dy = [0, -1, -1, -1, 0, 1, 1, 1, 1]
    
    for i in range(4):
        temp = list(map(int, input().split()))
        array.append([(temp[0],temp[1]),(temp[2],temp[3]),(temp[4],temp[5]),(temp[6],temp[7])]) # array에 (번호, 방향)
    
    x, y, direct = 0, 0, 0 # 초기 상어 좌표, 방향
    result=0 # 물고기 번호 최댓값
    result+=array[x][y][0]
    direct=array[x][y][1]
    array[x][y]=17
    
    for i in range(4):
      for j in range(4):
        fish=array[i][j]
        heapq.heappush(fishes, (fish[0], fish[1], i, j))
    
    while fishes:
      num, d, a, b=heapq.heappop(fishes) # 물고기 번호, 방향, x, y
      for i in range(8):
        nd=(d-1+i)%8
        na, nb = a+dx[nd], b+dy[nd]
        if array[na][nb]==17 or na<0 or na>=4 or nb<0 or nb>=4: # 이동 불가
          continue
        if 0<=array[na][nb]<=16: # 교환 가능
          fish2=array[na][nb]
          array[a][b], array[na][nb] = (fish2[0], fish2[1]), (num, nd)
          break
    
    maximum=0
    while True:
      ndirect=direct-1
      nx, ny = x+dx[ndirect], y+dy[ndirect]
    
      if nx<0 or nx>=4 or ny<0 or ny>=4:
        direct=array[nx][ny][1]
        result+=maximum
        break
      else:
        if maximum<array[nx][ny][0]:
          maximum=array[nx][ny][0]
          x, y=nx, ny
    ```
    
2. 풀이를 본 후
    
    물고기의 좌표와 방향을 저장해 놓고 완전 탐색으로 물고기와 상어의 위치 변경 까지는 맞았다… 하지만, 풀이는 상어가 이동하는 위치에 따라 경우들이 나뉘어지고 이 경우들을 처리해주기 위해 DFS를 이용했다. 이로써, 상어 이동이 불가할 때 마다 최대 값을 갱신하여 정답을 출력한다… 그리고 경우에 따라 배열의 값들이 달라지면 안되므로 copy를 이용했다.
    
     종료 조건과 문제의 조건을 바탕으로 차근차근 구현하도록 해야겠다…
    
    ```python
    import copy
    
    # 4 X 4 크기 격자에 존재하는 각 물고기의 번호(없으면 -1)와 방향 값을 담는 테이블
    array = [[None] * 4 for _ in range(4)]
    
    for i in range(4):
        data = list(map(int, input().split()))
        # 매 줄마다 4마리의 물고기를 하나씩 확인하며
        for j in range(4):
            # 각 위치마다 [물고기의 번호, 방향]을 저장
            array[i][j] = [data[j * 2], data[j * 2 + 1] - 1]
    
    # 8가지 방향에 대한 정의
    dx = [-1, -1, 0, 1, 1, 1, 0, -1]
    dy = [0, -1, -1, -1, 0, 1, 1, 1]
    
    # 현재 위치에서 왼쪽으로 회전된 결과 반환
    def turn_left(direction):
        return (direction + 1) % 8
    
    result = 0 # 최종 결과
    
    # 현재 배열에서 특정한 번호의 물고기 위치 찾기
    def find_fish(array, index):
        for i in range(4):
            for j in range(4):
                if array[i][j][0] == index:
                    return (i, j)
        return None
    
    # 모든 물고기를 회전 및 이동시키는 함수
    def move_all_fishes(array, now_x, now_y):
        # 1번부터 16번까지의 물고기를 차례대로 (낮은 번호부터) 확인
        for i in range(1, 17):
            # 해당 물고기의 위치를 찾기
            position = find_fish(array, i)
            if position != None:
                x, y = position[0], position[1]
                direction = array[x][y][1]
                # 해당 물고기의 방향을 왼쪽으로 계속 회전시키며 이동이 가능한지 확인
                for j in range(8):
                    nx = x + dx[direction]
                    ny = y + dy[direction]
                    # 해당 방향으로 이동이 가능하다면 이동 시키기
                    if 0 <= nx and nx < 4 and 0 <= ny and ny < 4:
                        if not (nx == now_x and ny == now_y):
                            array[x][y][1] = direction
                            array[x][y], array[nx][ny] = array[nx][ny], array[x][y]
                            break
                    direction = turn_left(direction)
            
    # 상어가 현재 위치에서 먹을 수 있는 모든 물고기의 위치 반환
    def get_possible_positions(array, now_x, now_y):
        positions = []
        direction = array[now_x][now_y][1]
        # 현재의 방향으로 쭉 이동하기
        for i in range(4):
            now_x += dx[direction]
            now_y += dy[direction]
            # 범위를 벗어나지 않는지 확인하며
            if 0 <= now_x and now_x < 4 and 0 <= now_y and now_y < 4:
                # 물고기가 존재하는 경우
                if array[now_x][now_y][0] != -1:
                    positions.append((now_x, now_y))
        return positions
    
    # 모든 경우를 탐색하기 위한 DFS 함수
    def dfs(array, now_x, now_y, total):
        global result
        array = copy.deepcopy(array) # 리스트를 통째로 복사
        
        total += array[now_x][now_y][0] # 현재 위치의 물고기 먹기
        array[now_x][now_y][0] = -1 # 물고기를 먹었으므로 번호 값을 -1로 변환
        
        move_all_fishes(array, now_x, now_y) # 전체 물고기 이동 시키기
    
        # 이제 다시 상어가 이동할 차례이므로, 이동 가능한 위치 찾기
        positions = get_possible_positions(array, now_x, now_y)
        # 이동할 수 있는 위치가 하나도 없다면 종료
        if len(positions) == 0:
            result = max(result, total) # 최댓값 저장
            return 
        # 모든 이동할 수 있는 위치로 재귀적으로 수행
        for next_x, next_y in positions:
            dfs(array, next_x, next_y, total)
    
    # 청소년 상어의 시작 위치(0, 0)에서부터 재귀적으로 모든 경우 탐색
    dfs(array, 0, 0, 0)
    print(result)
    ```

### 기출 문제 3: 어른 상어

1. 내 풀이
    1. array에 상어 번호, 남은 시간, 방향 저장
    2. 동시에 이동을 처리하기 위해 각 상어들의 이동 가능 좌표들 저장 후, 나중에 한번에 이동 처리
    3. 1번만 남거나, 1000초 넘을 때 종료
    
    구현에 실패했다… 2시간 정도 소요…
    
    ```python
    import sys
    
    input = sys.stdin.readline
    
    n, m, k = map(int, input().split())
    array = [[None] * n for _ in range(n)]
    for i in range(n):
      data = list(map(int, input().split()))
      for j in range(n):
        if data[j] == 0:
          array[i][j] = [0, 0, 0]
        else:
          array[i][j] = [data[j], k, 0]
    
    first_direction = list(map(int, input().split()))
    for i in range(1, m + 1):
      for a in range(n):
        for b in range(n):
          if i == array[a][b][0]:
            array[a][b][2] = first_direction[i - 1]
    
    rule = [[] for _ in range(m)]
    for i in range(m):
      for j in range(4):
        rule[i].append(list(map(int, input().split())))
    
    # 상, 하, 좌, 우
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    
    play_time = 0
    while True:
      # move
      temp=[]
      for i in range(1, m + 1):
        for a in range(n):
          for b in range(n):
            num, time, direct = array[a][b]  # 번호, 남은 시간, 방향
            if num != 0 and direct == 0:  # 냄새 칸
              time -= 1
              if time == 0:  # 사라짐
                array[a][b] = [0, 0, 0]
              else:  # 시간 줄음
                array[a][b] = [num, time, direct]
    
            if i == num and direct != 0:  # 해당 번호 상어 찾음
              d = direct - 1  # 상, 하, 좌, 우 인덱스
              rule[num - 1][d]  # 현재 방향에 대한 우선 순위 방향들
              for r in rule[num - 1][d]:  # 이동 가능할 때의 이동한 위치의 값들 저장
                na, nb = a + dx[r - 1], b + dy[r - 1]
                if na < 0 or na >= n or nb < 0 or nb >= n:  # 범위 밖
                  continue
                num2, time2, direct2 = array[na][nb]  # 이동할 곳의 번호, 남은 시간, 방향
                if 0 <= na < n and 0 <= nb < n:  # 격자 안쪽
                  if num2 == 0 or num == num2:  # 빈 칸 or 자기 냄새 칸
                    temp.append((a, b, na, nb))
                    break
    
      for t in temp:
        a, b, na, nb = t
        num, time, direct = array[a][b]
        num2, time2, direct2 = array[na][nb]
        if num2 == 0 or num2 == num:  # 바로 진입 가능 or 우선순위 높은 곳
          array[a][b], array[na][nb] = [num, time - 1, 0], [num, time, direct]
        else:  # 이미 강한 상어가 자리한 경우
          array[a][b] = [num, time - 1, 0]
          
      play_time += 1
    
      # check
      count = 0
      for i in range(1, m + 1):
        for a in range(n):
          for b in range(n):
            num, time, direct = array[a][b]
            if num == i and direct != 0:  # 상어
              count += 1
    
      if count == 1:  # 1번만 남았을 때
        print(play_time)
        break
    
      if play_time > 1000:  # 1000초 넘어갈 때
        print(-1)
        break
    ```
    
2. 풀이를 본 후
    
    풀이는 냄새 정보 만을 저장하는 배열을 따로 만들어서 혼동을 줄였다. 또한, 상어가 이동 가능할 때 방향을 바꿔주는 것을 까먹었다…
    
    시뮬레이션 및 구현 문제에서는 문제의 조건을 빠짐없이 차근차근 구현해 나가는 것이 핵심이다.
    
    ```python
    n, m, k = map(int, input().split())
    
    # 모든 상어의 위치와 방향 정보를 포함하는 2차원 리스트
    array = []
    for i in range(n):
        array.append(list(map(int, input().split())))
    
    # 모든 상어의 현재 방향 정보
    directions = list(map(int, input().split()))
    
    # 각 위치마다 [특정 냄새의 상어 번호, 특정 냄새의 남은 시간]을 저장하는 2차원 리스트
    smell = [[[0, 0]] * n for _ in range(n)]
    
    # 각 상어의 회전 우선순위 정보
    priorities = [[] for _ in range(m)]
    for i in range(m):
        for j in range(4):
            priorities[i].append(list(map(int, input().split())))
    
    # 특정 위치에서 이동 가능한 4가지 방향
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    
    # 모든 냄새 정보를 업데이트
    def update_smell():
        # 각 위치를 하나씩 확인하며
        for i in range(n):
            for j in range(n):
                # 냄새가 존재하는 경우, 시간을 1만큼 감소시키기
                if smell[i][j][1] > 0:
                    smell[i][j][1] -= 1
                # 상어가 존재하는 해당 위치의 냄새를 k로 설정
                if array[i][j] != 0:
                    smell[i][j] = [array[i][j], k]
    
    # 모든 상어를 이동시키는 함수
    def move():
        # 이동 결과를 담기 위한 임시 결과 테이블 초기화
        new_array = [[0] * n for _ in range(n)]
        # 각 위치를 하나씩 확인하며
        for x in range(n):
            for y in range(n):
                # 상어가 존재하는 경우
                if array[x][y] != 0:
                    direction = directions[array[x][y] - 1] # 현재 상어의 방향
                    found = False
                    # 일단 냄새가 존재하지 않는 곳이 있는지 확인
                    for index in range(4):
                        nx = x + dx[priorities[array[x][y] - 1][direction - 1][index] - 1]
                        ny = y + dy[priorities[array[x][y] - 1][direction - 1][index] - 1]
                        if 0 <= nx and nx < n and 0 <= ny and ny < n:
                            if smell[nx][ny][1] == 0: # 냄새가 존재하지 않는 곳이면
                                # 해당 상어의 방향 이동시키기
                                directions[array[x][y] - 1] = priorities[array[x][y] - 1][direction - 1][index]
                                # 상어 이동시키기 (만약 이미 다른 상어가 있다면 번호가 낮은 것이 들어가도록)
                                if new_array[nx][ny] == 0:
                                    new_array[nx][ny] = array[x][y]
                                else:
                                    new_array[nx][ny] = min(new_array[nx][ny], array[x][y])
                                found = True
                                break
                    if found:
                        continue
                    # 주변에 모두 냄새가 남아 있다면, 자신의 냄새가 있는 곳으로 이동
                    for index in range(4):
                        nx = x + dx[priorities[array[x][y] - 1][direction - 1][index] - 1]
                        ny = y + dy[priorities[array[x][y] - 1][direction - 1][index] - 1]
                        if 0 <= nx and nx < n and 0 <= ny and ny < n:
                            if smell[nx][ny][0] == array[x][y]: # 자신의 냄새가 있는 곳이면
                                # 해당 상어의 방향 이동시키기
                                directions[array[x][y] - 1] = priorities[array[x][y] - 1][direction - 1][index]
                                # 상어 이동시키기
                                new_array[nx][ny] = array[x][y]
                                break
        return new_array
    
    time = 0
    while True:
        update_smell() # 모든 위치의 냄새를 업데이트
        new_array = move() # 모든 상어를 이동시키기
        array = new_array # 맵 업데이트
        time += 1 # 시간 증가
    
        # 1번 상어만 남았는지 체크
        check = True
        for i in range(n):
            for j in range(n):
                if array[i][j] > 1:
                    check = False
        if check:
            print(time)
            break
    
        # 1000초가 지날 때까지 끝나지 않았다면
        if time >= 1000:
            print(-1)
            break
    ```

## 출처

- 이것이 취업을 위한 코딩테스트다 with 파이썬