---
layout: post
title: 이것이 취업을 위한 코딩테스트다 | 8. 그래프 이론 (Graph Theory)
categories: 
  - cs
  - codingtest
description: 이것이 취업을 위한 코딩테스트다 with 파이썬 책을 참고하여 **그래프 이론**을 공부한 내용입니다.
sitemap: false
---

코딩 테스트에서 자주 등장하는 기타 그래프 이론 공부

* this unordered seed list will be replaced by the toc
{:toc}

## 이미 배운 내용 훑어보자

- 5장, 9장 이외의 그래프 알고리즘 → 코딩 테스트 출제 비중이 낮은 편이지만, 다양한 응용 문제들을 해결하는데 도움이 될 것이다.
- ‘서로 다른 개체 연결’ → 그래프 알고리즘 / ex) 여러 개의 도시가 연결되어 있다.
- 트리 자료구조는 다양한 알고리즘에 사용되므로 꼭 숙지
    
    
    |  | 그래프 | 트리 |
    | --- | --- | --- |
    | 방향성 | 방향 그래프 혹은 무방향 그래프 | 방향 그래프 |
    | 순환성 | 순환 및 비순환 | 비순환 |
    | 루트 노드 존재 여부 | 루트 노드가 없음 | 루트 노드가 존재 |
    | 노드간 관계성 | 부모와 자식 관계 없음 | 부모와 자식 관계 있음 |
    | 모델의 종류 | 네트워크 모델 | 계층 모델 |

- 어떤 문제를 만나든 메모리와 시간을 염두하고 알고리즘 선택

## 서로소 집합

- 공통 원소가 없는 두 집합

### 서로소 집합 자료구조

- 서로소 부분 집합들로 나누어진 원소들의 데이터를 처리하기 위한 자료구조
- 연산
    - union(합집합), find(찾기)
- 트리 자료구조 이용하여 표현
    1. union(합집합) 연산을 확인하여, 서로 연결된 두 노드 A, B를 확인
        1. A와B의 루트 노드 A’, B’을 찾는다
        2. A’을 B’의 부모 노드로 설정(B’가 A’를 가리키도록 한다, 번호가 작은 노드를 부모 노드로 설정)
    2. 모든 union(합집합) 연산을 처리할 때까지 1번 과정 반복
- 루트는 재귀적으로 부모를 거슬러 올라가서 찾는다
- 기본적인 서로소 집합 알고리즘
    
    ```python
    # 특정 원소가 속한 집합을 찾기
    def find_parent(parent, x):
        # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
        if parent[x] != x:
            return find_parent(parent, parent[x])
        return x
    
    # 두 원소가 속한 집합을 합치기
    def union_parent(parent, a, b):
        a = find_parent(parent, a)
        b = find_parent(parent, b)
        if a < b:
            parent[b] = a
        else:
            parent[a] = b
    
    # 노드의 개수와 간선(Union 연산)의 개수 입력 받기
    v, e = map(int, input().split())
    parent = [0] * (v + 1) # 부모 테이블 초기화하기
    
    # 부모 테이블상에서, 부모를 자기 자신으로 초기화
    for i in range(1, v + 1):
        parent[i] = i
    
    # Union 연산을 각각 수행
    for i in range(e):
        a, b = map(int, input().split())
        union_parent(parent, a, b)
    
    # 각 원소가 속한 집합 출력하기
    print('각 원소가 속한 집합: ', end='')
    for i in range(1, v + 1):
        print(find_parent(parent, i), end=' ')
    
    print()
    
    # 부모 테이블 내용 출력하기
    print('부모 테이블: ', end='')
    for i in range(1, v + 1):
        print(parent[i], end=' ')
    ```
    
    - 노드의 개수: V, 연산 (find, union)의 개수: M → 전체 시간복잡도: $$O(VM)$$
    
- 개선된 서로소 집합 알고리즘 (경로 압축)
    - 경로 압축
        - find함수 재귀적으로 호출하여 부모 테이블 값 갱신
        - 루트 노드에 더 빠르게 접근하여 시간 복잡도 개선
    
    ```python
    # 특정 원소가 속한 집합을 찾기
    def find_parent(parent, x):
        # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
        if parent[x] != x:
            parent[x] = find_parent(parent, parent[x])
        return parent[x]
    
    # 두 원소가 속한 집합을 합치기
    def union_parent(parent, a, b):
        a = find_parent(parent, a)
        b = find_parent(parent, b)
        if a < b:
            parent[b] = a
        else:
            parent[a] = b
    
    # 노드의 개수와 간선(Union 연산)의 개수 입력 받기
    v, e = map(int, input().split())
    parent = [0] * (v + 1) # 부모 테이블 초기화하기
    
    # 부모 테이블상에서, 부모를 자기 자신으로 초기화
    for i in range(1, v + 1):
        parent[i] = i
    
    # Union 연산을 각각 수행
    for i in range(e):
        a, b = map(int, input().split())
        union_parent(parent, a, b)
    
    # 각 원소가 속한 집합 출력하기
    print('각 원소가 속한 집합: ', end='')
    for i in range(1, v + 1):
        print(find_parent(parent, i), end=' ')
    
    print()
    
    # 부모 테이블 내용 출력하기
    print('부모 테이블: ', end='')
    for i in range(1, v + 1):
        print(parent[i], end=' ')
    ```
    

### 서로소 집합 알고리즘의 시간 복잡도

- 경로 압축만 고려(V: 노드의 개수, 최대 V-1 개의 union연산, M개의 find 연산) → $$O(V+M(1+\log_{2-M/V}V ))$$
- ex) 노드의 개수 1000개, 연산(union, find)의 개수 100만번 → 약 1000만번 가량의 연산 필요

### 서로소 집합을 이용한 사이클 판별

- 무방향 그래프에서 사이클 판별에 사용 가능 (방향 그래프에서는 DFS 이용하여 가능)
- 알고리즘 과정
    1. 각 간선을 확인하며 두 노드의 루트 노드를 확인한다
        1. 루트 노드가 서로 다르면 두 노드에 대해 union 연산 수행
        2. 루트 노드가 같으면 사이클이 발생한 것
    2. 그래프에 포함되어 있는 모든 간선에 대하여 1번 과정 반복
- 매 간선에 대해 union 및 find 함수를 호출하는 방식으로 동작

```python
# 특정 원소가 속한 집합을 찾기
def find_parent(parent, x):
    # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
    if parent[x] != x:
        parent[x] = find_parent(parent, parent[x])
    return parent[x]

# 두 원소가 속한 집합을 합치기
def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b

# 노드의 개수와 간선(Union 연산)의 개수 입력 받기
v, e = map(int, input().split())
parent = [0] * (v + 1) # 부모 테이블 초기화하기

# 부모 테이블상에서, 부모를 자기 자신으로 초기화
for i in range(1, v + 1):
    parent[i] = i

cycle = False # 사이클 발생 여부

for i in range(e):
    a, b = map(int, input().split())
    # 사이클이 발생한 경우 종료
    if find_parent(parent, a) == find_parent(parent, b):
        cycle = True
        break
    # 사이클이 발생하지 않았다면 합집합(Union) 연산 수행
    else:
        union_parent(parent, a, b)

if cycle:
    print("사이클이 발생했습니다.")
else:
    print("사이클이 발생하지 않았습니다.")
```

## 신장 트리

- 하나의 그래프가 있을 때 모든 노드를 포함하면서 사이클이 존재하지 않는 부분 그래프 → 트리의 성립 조건이기도 하다

### 크루스칼 알고리즘

- 최소 신장 트리 알고리즘: 신장트리 중에서 최소 비용의 신장 트리를 찾는 알고리즘 / ex) 두 도시 A, B의 연결 통로 설치의 최소 비용 구하기
- 대표적인 최소 신장 트리 알고리즘
- 가장 적은 비용으로 모든 노드 연결 가능 (그리디 알고리즘으로 분류)
- 알고리즘 과정
    1. 간선 데이터를 비용에 따라 오름차순으로 정렬
    2. 간선을 하나씩 확인하며 현재 간선이 사이클을 발생시키는지 확인
        1. 발생 X → 최소 신장 트리에 포함
        2. 발생 O → 최소 신장 트리에 포함 X
    3. 모든 간선에 대해 2번의 과정 반복
- 일종의 트리 자료구조 → 최종적인 신장 트리에 포함되는 간선의 개수: 노드의 개수 - 1
- 핵심 : 가장 짧은 거리의 간선부터 차례로 집합에 포함 (사이클 발생 간선 제외)
    
    ```python
    # 특정 원소가 속한 집합을 찾기
    def find_parent(parent, x):
        # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
        if parent[x] != x:
            parent[x] = find_parent(parent, parent[x])
        return parent[x]
    
    # 두 원소가 속한 집합을 합치기
    def union_parent(parent, a, b):
        a = find_parent(parent, a)
        b = find_parent(parent, b)
        if a < b:
            parent[b] = a
        else:
            parent[a] = b
    
    # 노드의 개수와 간선(Union 연산)의 개수 입력 받기
    v, e = map(int, input().split())
    parent = [0] * (v + 1) # 부모 테이블 초기화하기
    
    # 모든 간선을 담을 리스트와, 최종 비용을 담을 변수
    edges = []
    result = 0
    
    # 부모 테이블상에서, 부모를 자기 자신으로 초기화
    for i in range(1, v + 1):
        parent[i] = i
    
    # 모든 간선에 대한 정보를 입력 받기
    for _ in range(e):
        a, b, cost = map(int, input().split())
        # 비용순으로 정렬하기 위해서 튜플의 첫 번째 원소를 비용으로 설정
        edges.append((cost, a, b))
    
    # 간선을 비용순으로 정렬
    edges.sort()
    
    # 간선을 하나씩 확인하며
    for edge in edges:
        cost, a, b = edge
        # 사이클이 발생하지 않는 경우에만 집합에 포함
        if find_parent(parent, a) != find_parent(parent, b):
            union_parent(parent, a, b)
            result += cost
    
    print(result)
    ```
    

### 크루스칼 알고리즘의 시간 복잡도

- 간선의 개수 E개 → $$O(ElogE)$$ (간선 정렬 시간 복잡도가 가장 크다)

## 위상 정렬

- 정렬 알고리즘의 일종, 순서가 정해져 있는 일련의 작업을 차례로 수행할 때 사용
- 방향 그래프의 모든 노드를 ‘방향성에 거스르지 않도록 순서대로 나열하는 것’ / ex) 선수 과목을 고려한 학습 순서 설정
- 그래프 상에서 선후 관계가 있다면, 위상 정렬을 수행하여 모든 선후 관계를 지키는 전체 순서 계산 가능
- 진입차수 : 특정한 노드로 ‘들어오는’ 간선의 개수
- 알고리즘 과정
    1. 진입차수가 0인 노드 큐에 넣음
    2. 큐가 빌 때까지 다음의 과정 반복
        1. 큐에서 원소를 꺼내 해당 노드에서 출발하는 간선을 그래프에서 제거
        2. 새롭게 진입차수가 0이 된 노드를 큐에 넣음
- 큐에 새롭게 들어가는 원소가 2개 이상인 경우 여러 개의 답이 될 수 있다

```python
from collections import deque

# 노드의 개수와 간선의 개수를 입력 받기
v, e = map(int, input().split())
# 모든 노드에 대한 진입차수는 0으로 초기화
indegree = [0] * (v + 1)
# 각 노드에 연결된 간선 정보를 담기 위한 연결 리스트 초기화
graph = [[] for i in range(v + 1)]

# 방향 그래프의 모든 간선 정보를 입력 받기
for _ in range(e):
    a, b = map(int, input().split())
    graph[a].append(b) # 정점 A에서 B로 이동 가능
    # 진입 차수를 1 증가
    indegree[b] += 1

# 위상 정렬 함수
def topology_sort():
    result = [] # 알고리즘 수행 결과를 담을 리스트
    q = deque() # 큐 기능을 위한 deque 라이브러리 사용

    # 처음 시작할 때는 진입차수가 0인 노드를 큐에 삽입
    for i in range(1, v + 1):
        if indegree[i] == 0:
            q.append(i)

    # 큐가 빌 때까지 반복
    while q:
        # 큐에서 원소 꺼내기
        now = q.popleft()
        result.append(now)
        # 해당 원소와 연결된 노드들의 진입차수에서 1 빼기
        for i in graph[now]:
            indegree[i] -= 1
            # 새롭게 진입차수가 0이 되는 노드를 큐에 삽입
            if indegree[i] == 0:
                q.append(i)

    # 위상 정렬을 수행한 결과 출력
    for i in result:
        print(i, end=' ')

topology_sort()
```

### 위상 정렬의 시간 복잡도

- 노드의 개수: V, 간선의 개수: E → $$O(V+E)$$
- 모든 노드를 확인하면서 해당 노드에서 출발하는 간선을 확인하기 때문


## 예제 문제

- 예제 1: 팀 결성
    1. 내 풀이
        1. 서로소 집합 알고리즘을 이용하여 find, union 연산 구현
        2. 그 후, 조건에 맞게 연산 수
        
        ```python
        import sys
        input=sys.stdin.readline
        
        def find_parent(parent, x):
          if parent[x] != x:
            parent[x] = find_parent(parent, parent[x])
          return parent[x]
          
        def union(a,b):
          a = find_parent(parent, a)
          b = find_parent(parent, b)
          if a<b:
            parent[b]=a
          else:
            parent[a]=b
        
        N,M=map(int,input().split())
        parent=[0]*(N+1)
        
        for i in range(N+1):
          parent[i]=i
          
        for i in range(M):
          op,a,b=map(int,input().split())
          if op==0:
            union(a,b)
          else:
            if find_parent(parent,a)==find_parent(parent,b):
              print("YES")
            else:
              print("NO")
        ```
        
    2. 풀이를 본 후
        
        풀이도 같다
        
        ```python
        # 특정 원소가 속한 집합을 찾기
        def find_parent(parent, x):
            # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
            if parent[x] != x:
                parent[x] = find_parent(parent, parent[x])
            return parent[x]
        
        # 두 원소가 속한 집합을 합치기
        def union_parent(parent, a, b):
            a = find_parent(parent, a)
            b = find_parent(parent, b)
            if a < b:
                parent[b] = a
            else:
                parent[a] = b
        
        n, m = map(int, input().split())
        parent = [0] * (n + 1) # 부모 테이블 초기화
        
        # 부모 테이블상에서, 부모를 자기 자신으로 초기화
        for i in range(0, n + 1):
            parent[i] = i
        
        # 각 연산을 하나씩 확인
        for i in range(m):
            oper, a, b = map(int, input().split())
            # 합치합(Union) 연산인 경우
            if oper == 0:
                union_parent(parent, a, b)
            # 찾기(Find) 연산인 경우
            elif oper == 1:
                if find_parent(parent, a) == find_parent(parent, b):
                    print('YES')
                else:
                    print('NO')
        ```
        
    3. 해결한 후
        
        서로소 집합 알고리즘 연산들 구현 숙지
        
- 예제 2: 도시 분할 계획
    1. 내 풀이
        1. 크루스칼 알고리즘을 통해 최소 신장 트리를 구한다.
        2. 그 후, 도시간의 간선 중에서 가장 큰 간선을 제거한다.
        3. 마지막으로 간선들의 총합을 구한다.
        
        ```python
        import sys
        input=sys.stdin.readline
        
        def find_parent(parent, x):
          if parent[x] != x:
            parent[x] = find_parent(parent, parent[x])
          return parent[x]
          
        def union(a,b):
          a = find_parent(parent, a)
          b = find_parent(parent, b)
          if a<b:
            parent[b]=a
          else:
            parent[a]=b
        
        N,M=map(int,input().split())
        edges=[]
        result=0
        
        parent=[0]*(N+1)
        for i in range(N+1):
          parent[i]=i
          
        for i in range(M):
          a,b,cost=map(int,input().split())
          edges.append((cost,a,b))
        
        edges.sort()
        max_cost=0
        for edge in edges:
          cost,a,b=edge
          if find_parent(parent, a) != find_parent(parent, b):
            union(a,b)
            max_cost=max(max_cost, cost)
            result+=cost
        
        print(result-max_cost)
        ```
        
    2. 풀이를 본 후
        
        풀이도 같다
        
        ```python
        # 특정 원소가 속한 집합을 찾기
        def find_parent(parent, x):
            # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
            if parent[x] != x:
                parent[x] = find_parent(parent, parent[x])
            return parent[x]
        
        # 두 원소가 속한 집합을 합치기
        def union_parent(parent, a, b):
            a = find_parent(parent, a)
            b = find_parent(parent, b)
            if a < b:
                parent[b] = a
            else:
                parent[a] = b
        
        # 노드의 개수와 간선(Union 연산)의 개수 입력받기
        v, e = map(int, input().split())
        parent = [0] * (v + 1) # 부모 테이블 초기화
        
        # 모든 간선을 담을 리스트와, 최종 비용을 담을 변수
        edges = []
        result = 0
        
        # 부모 테이블상에서, 부모를 자기 자신으로 초기화
        for i in range(1, v + 1):
            parent[i] = i
        
        # 모든 간선에 대한 정보를 입력받기
        for _ in range(e):
            a, b, cost = map(int, input().split())
            # 비용순으로 정렬하기 위해서 튜플의 첫 번째 원소를 비용으로 설정
            edges.append((cost, a, b))
        
        # 간선을 비용순으로 정렬
        edges.sort()
        last = 0 # 최소 신장 트리에 포함되는 간선 중에서 가장 비용이 큰 간선
        
        # 간선을 하나씩 확인하며
        for edge in edges:
            cost, a, b = edge
            # 사이클이 발생하지 않는 경우에만 집합에 포함
            if find_parent(parent, a) != find_parent(parent, b):
                union_parent(parent, a, b)
                result += cost
                last = cost
        
        print(result - last)
        ```
        
    3. 해결한 후
        
        도시간 거리의 최소 비용 등의 문제가 나오면 크루스칼 알고리즘 떠올리기
        
- 예제 3: 커리큘럼
    1. 내 풀이
        1. 위상 정렬과 강의 시간을 담을 리스트 구현
        2. 알고리즘을 진행하면서 선행 과목의 강의 시간을 더해주면서 강의 시간 갱신
        3. 최종적인 최소 수강 시간 출력
        
        ```python
        from collections import deque
        import sys
        input=sys.stdin.readline
        
        def topology_sort():
          q=deque()
        
          for i in range(1,N+1):
            if indegree[i]==0:
              q.append(i)
        
          while q:
            now=q.popleft()
            for i in graph[now]:
              indegree[i]-=1
              if indegree[i]==0:
                result[i]+=result[now]
                q.append(i)
        
          for i in range(1,N+1):
            print(result[i])
                
        N=int(input())
        indegree=[0]*(N+1)
        graph=[[] for _ in range(N+1)]
        result=[0]*(N+1)
        
        for i in range(N):
          temp=list(map(int,input().split()))
          result[i+1]=temp[0]
          num=temp[1:-1]
          for n in num:
            graph[n].append(i+1)
            indegree[i+1]+=1
            
        topology_sort()
        ```
        
    2. 풀이를 본 후
        
        인접한 노드에 대해서 현재 저장된 시간보다 강의 시간이 더 긴 경우가 있다면 갱신을 해줘야한다. 
        
        ```python
        from collections import deque
        import copy
        
        # 노드의 개수 입력받기
        v = int(input())
        # 모든 노드에 대한 진입차수는 0으로 초기화
        indegree = [0] * (v + 1)
        # 각 노드에 연결된 간선 정보를 담기 위한 연결 리스트(그래프) 초기화
        graph = [[] for i in range(v + 1)]
        # 각 강의 시간을 0으로 초기화
        time = [0] * (v + 1)
        
        # 방향 그래프의 모든 간선 정보를 입력받기
        for i in range(1, v + 1):
            data = list(map(int, input().split()))
            time[i] = data[0] # 첫 번째 수는 시간 정보를 담고 있음
            for x in data[1:-1]:
                indegree[i] += 1
                graph[x].append(i)
        
        # 위상 정렬 함수
        def topology_sort():
            result = copy.deepcopy(time) # 알고리즘 수행 결과를 담을 리스트
            q = deque() # 큐 기능을 위한 deque 라이브러리 사용
        
            # 처음 시작할 때는 진입차수가 0인 노드를 큐에 삽입
            for i in range(1, v + 1):
                if indegree[i] == 0:
                    q.append(i)
        
            # 큐가 빌 때까지 반복
            while q:
                # 큐에서 원소 꺼내기
                now = q.popleft()
                # 해당 원소와 연결된 노드들의 진입차수에서 1 빼기
                for i in graph[now]:
                    result[i] = max(result[i], result[now] + time[i])
                    indegree[i] -= 1
                    # 새롭게 진입차수가 0이 되는 노드를 큐에 삽입
                    if indegree[i] == 0:
                        q.append(i)
        
            # 위상 정렬을 수행한 결과 출력
            for i in range(1, v + 1):
                print(result[i])
        
        topology_sort()
        ```
        
    3. 해결한 후
        
        문제의 예시를 잘못 해석해서 틀렸다. 선행 과목을 다 수강 해야 하므로 최대의 시간으로 저장해야 한다.

## 기출 문제

- 기출 문제 1: 여행 계획
    1. 내 풀이
        1. 서로소 집합을 이용하여 연결된 도시끼리 union연산 진행
        2. 그 후, 인접한 여행 계획 도시들의 부모가 같으면 여행 가능, 아니면 여행 불가로 판별
        
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
        parent = [0] * (n + 1)
        
        for i in range(0, n + 1):
          parent[i] = i
        
        trip = []
        for i in range(n):
          trip.append(list(map(int, input().split())))
        
        plan = list(map(int, input().split()))
        
        for i in range(n):
          for j in range(n):
            if trip[i][j] == 1:
              union_parent(parent, i + 1, j + 1)
        
        ans = 'YES'
        for i in range(m - 1):
          if find_parent(parent, plan[i]) != find_parent(parent, plan[i + 1]):
            ans = 'NO'
            break
        print(ans)
        ```
        
    2. 풀이를 본 후
        
        풀이도 거의 유사한 방법이다.
        
        ```python
        # 특정 원소가 속한 집합을 찾기
        def find_parent(parent, x):
            # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
            if parent[x] != x:
                parent[x] = find_parent(parent, parent[x])
            return parent[x]
        
        # 두 원소가 속한 집합을 합치기
        def union_parent(parent, a, b):
            a = find_parent(parent, a)
            b = find_parent(parent, b)
            if a < b:
                parent[b] = a
            else:
                parent[a] = b
        
        # 여행지의 개수와 여행 계획에 속한 여행지의 개수 입력받기
        n, m = map(int, input().split())
        parent = [0] * (n + 1) # 부모 테이블 초기화
        
        # 부모 테이블상에서, 부모를 자기 자신으로 초기화
        for i in range(1, n + 1):
            parent[i] = i
        
        # Union 연산을 각각 수행
        for i in range(n):
            data = list(map(int, input().split()))
            for j in range(n):
                if data[j] == 1: # 연결된 경우 합집합(Union) 연산 수행
                    union_parent(parent, i + 1, j + 1)
        
        # 여행 계획 입력받기
        plan = list(map(int, input().split()))
        
        result = True
        # 여행 계획에 속하는 모든 노드의 루트가 동일한지 확인
        for i in range(m - 1):
            if find_parent(parent, plan[i]) != find_parent(parent, plan[i + 1]):
                result = False
        
        # 여행 계획에 속하는 모든 노드가 서로 연결되어 있는지(루트가 동일한지) 확인
        if result:
            print("YES")
        else:
            print("NO")
        ```
        
    3. 해결한 후
        
        행렬 정보를 입력 받으면서 union연산까지 진행 할 수 있었다…


## 출처

- 이것이 취업을 위한 코딩테스트다 with 파이썬