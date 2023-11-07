---
layout: post
title: 스택(Stack) & 큐(Queue)
categories: 
  - cs
  - algorithm
description: 스택(Stack) & 큐(Queue) 관련 개념 정리글 입니다.
sitemap: false
---

스택(Stack)과 큐(Queue)는 데이터를 저장하고 접근하는 데 사용되는 일반적인 추상 자료 구조이다.

* this unordered seed list will be replaced by the toc
{:toc}

## 스택(Stack)

### 정의
- 스택은 데이터를 저장하고 접근하는데 사용되는 추상 자료 구조이다. 
- 스택은 후입선출 (Last-In, First-Out) 방식으로 동작하는데 이는 가장 최근에 삽입된 데이터가 가장 먼저 제거되는 원리를 의미한다.

### 동작 원리
- 스택에 데이터를 삽입하는 연산을 "push"라고 한다. 이때 데이터는 스택의 맨 위에 위치하게 된다.
- 스택에서 데이터를 제거하는 연산을 "pop"이라고 한다. pop 연산은 스택의 맨 위에 있는 데이터를 제거하고 반환한다.
- 스택의 맨 위에 있는 데이터를 확인하는 연산을 "peek"라고 한다. peek 연산은 데이터를 제거하지 않고 반환한다.


![스택(Stack)](/assets/img/blog/Stack.png){: width="800" height="400"}

스택(Stack)의 동작 원리
{:.figure}


### 구현방법

#### 1. 배열을 이용한 구현
- 스택의 크기가 고정되어 있는 경우에 적합하다.
- 주요 연산인 push와 pop은 배열의 인덱스를 조정하여 구현할 수 있다. 

- 장점
    - 접근이 빠르고 $$O(1)$$ 내에 연산이 수행된다.
- 단점
    - 스택의 크기가 고정되어 있어, 크기를 동적으로 조절하기 어렵다.
    - 배열의 크기를 초과할 경우, 추가적인 메모리 할당과 복사 과정이 필요하다.

~~~python
class ArrayStack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()

    def peek(self):
        if not self.is_empty():
            return self.stack[-1]

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)

~~~

배열 기반 스택 코드 구현
{:.figure}

#### 2. 연결 리스트를 이용한 구현

- 스택을 구현할 때는 단순 연결 리스트 또는 이중 연결 리스트를 사용한다. 

- 장점
    - 동적으로 크기가 조절되기 때문에 크기에 제한이 없다.
    - 삽입과 삭제 연산이 간단하고 $$O(1)$$ 내에 수행된다.

- 단점
    - 포인터를 사용하기 때문에 추가적인 메모리 공간이 필요하다.
    - 인덱스로 직접 접근할 수 없기 때문에 원하는 위치로 바로 이동하는 것이 어렵다.

~~~python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedListStack:
    def __init__(self):
        self.head = None

    def push(self, item):
        new_node = Node(item)
        new_node.next = self.head
        self.head = new_node

    def pop(self):
        if not self.is_empty():
            popped_item = self.head.data
            self.head = self.head.next
            return popped_item

    def peek(self):
        if not self.is_empty():
            return self.head.data

    def is_empty(self):
        return self.head is None

    def size(self):
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count
~~~
연결리스트 기반 스택 코드 구현
{:.figure}

### 연산

- push (삽입): 스택의 맨 위에 요소를 추가 --> 시간 복잡도는 $$O(1)$$
- pop (삭제): 스택의 맨 위 요소를 제거하고 반환 --> 시간 복잡도는 $$O(1)$$
- peek (탑 요소 확인): 스택의 맨 위 요소를 제거하지 않고 반환 --> 시간 복잡도는 $$O(1)$$
- is_empty (비어있는지 확인): 스택이 비어있는지 여부를 반환 --> 시간 복잡도는 $$O(1)$$
- size (스택의 크기 확인): 스택의 요소 개수를 반환 --> 시간 복잡도는 $$O(1)$$

### 응용

- 함수 호출 스택: 함수 호출과 관련된 정보를 스택에 저장하여 함수 호출 순서를 추적하고, 함수 실행이 완료되면 역순으로 호출을 처리
- 괄호 검사: 여는 괄호는 스택에 push하고, 닫는 괄호를 만났을 때 스택에서 pop하여 쌍이 맞는지 확인
- 문자열 뒤집기: 원소를 스택에 push한 후, 스택에서 pop하여 역순으로 뒤집힌 결과를 얻을 수 있다.

### 시간 복잡도 분석

- push, pop, peek, is_empty, size 연산의 시간 복잡도는 모두 $$O(1)$$이고 이는 스택의 크기에 관계없이 일정한 시간이 소요됨을 의미한다.
- 스택의 전체 시간 복잡도는 주로 스택을 사용하는 알고리즘의 복잡도에 따라 결정된다. 스택 자체의 연산은 모두 $$O(1)$$이지만, 스택을 활용한 다른 알고리즘의 시간 복잡도에 따라 달라진다.


## 큐(Queue)

### 개념 및 동작 원리
- 큐는 선입선출 (First-In-First-Out, FIFO) 방식으로 동작하는 자료 구조이다. 새로운 요소는 큐의 뒤에 추가되고, 삭제는 큐의 앞에서 이루어진다.
- 큐는 실생활에서 줄을 서는 것과 유사한 동작을 한다. 처음으로 도착한 사람이 먼저 서고, 대기열에서는 먼저 온 사람이 먼저 나가는 것과 같다.

![큐(Queue)](/assets/img/blog/Queue.png){: width="800" height="400"}

큐(Queue)의 동작 원리
{:.figure}

### 구현 방법

#### 1. 배열을 이용한 구현
- 배열을 사용하여 큐를 구현하는 방법은 가장 간단한 방법 중 하나이다.
- 큐의 원소들을 배열에 저장하고, 인덱스를 이용하여 삽입과 삭제를 수행한다. 다만, 배열의 크기를 미리 정해야 하며, 큐의 크기가 제한되는 단점이 있다.

~~~python
class ArrayQueue:
    def __init__(self):
        self.queue = []
    
    def enqueue(self, item):
        self.queue.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)
    
    def front(self):
        if not self.is_empty():
            return self.queue[0]
    
    def is_empty(self):
        return len(self.queue) == 0
    
    def size(self):
        return len(self.queue)
~~~
배열 기반 큐 코드 구현
{:.figure}

#### 2. 연결 리스트를 이용한 구현
- 연결 리스트를 사용하여 큐를 구현하는 방법은 큐의 크기를 제한하지 않고 유동적으로 크기를 조절할 수 있다.
- 큐의 각 노드는 데이터와 다음 노드를 가리키는 포인터로 구성된다.

~~~python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedListQueue:
    def __init__(self):
        self.head = None
        self.tail = None

    def enqueue(self, item):
        new_node = Node(item)
        if self.is_empty():
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

    def dequeue(self):
        if not self.is_empty():
            item = self.head.data
            self.head = self.head.next
            if self.head is None:
                self.tail = None
            return item

    def front(self):
        if not self.is_empty():
            return self.head.data

    def is_empty(self):
        return self.head is None

    def size(self):
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count
~~~
연결리스트 기반 큐 코드 구현
{:.figure}

### 연산

- enqueue (삽입): 큐의 뒤(rear)에 요소를 추가
- dequeue (삭제): 큐의 앞(front)에서 요소를 제거하고 반환
- peek (프런트 요소 확인): 큐의 앞(front)에 있는 요소를 제거하지 않고 반환
- is_empty (비어있는지 확인): 큐가 비어있는지 여부를 반환
- size (큐의 크기 확인): 큐의 요소 개수를 반환

### 응용

- 우선순위 큐: 요소마다 우선순위를 가지고 있는 큐로, 우선순위에 따라 요소의 삽입과 삭제가 이루어진다.
- 버퍼 관리: 입출력 작업이나 데이터 전달 시 일시적으로 데이터를 저장하는 버퍼로 큐를 사용한다.
- 프로세스 스케줄링: 여러 프로세스의 실행 순서를 결정하는 스케줄링 알고리즘에서 큐를 활용하여 작업을 관리한다.

### 시간 복잡도 분석

- enqueue, dequeue, peek, is_empty, size 연산의 시간 복잡도는 모두 $$O(1$$)으로, 큐의 크기에 관계없이 일정한 시간이 소요됩니다.
- 큐의 전체 시간 복잡도는 주로 큐를 사용하는 알고리즘의 복잡도에 따라 결정된다. 큐 자체의 연산은 모두 $$O(1)$$이지만, 큐를 활용한 다른 알고리즘의 시간 복잡도에 따라 달라진다.

## 스택(Stack)과 큐(Queue) 비교

1. 동작 원리
    - 스택: 후입선출(LIFO, Last-In-First-Out) 방식으로 동작 --> 가장 최근에 삽입된 요소가 먼저 제거
    - 큐: 선입선출(FIFO, First-In-First-Out) 방식으로 동작 --> 가장 먼저 삽입된 요소가 먼저 제거

2. 구조
    - 스택: 단일 엔드(End)로 구성된다. 요소들은 스택의 맨 위에 쌓임
    - 큐: 두 개의 엔드(End)로 구성된다. 요소들은 큐의 한쪽 엔드에서 삽입되고, 다른 쪽 엔드에서 제거

3. 시간 복잡도
    - 스택과 큐의 기본 연산(push, pop, enqueue, dequeue)의 시간 복잡도는 모두 $$O(1)$$이다.
    - 스택과 큐의 전체 시간 복잡도는 구현 방식에 따라 달라진다. 배열을 사용한 구현의 경우, 데이터 이동이 필요한 경우가 있어 $$O(n)$$의 시간 복잡도를 가질 수 있다. 하지만 연결 리스트를 사용한 구현은 데이터 이동이 없으므로 일반적으로 $$O(1)$$의 시간 복잡도를 가진다.

4. 활용
    - 스택은 주로 요소의 추가와 제거가 스택의 맨 위에서 이루어지는 경우에 사용되고, 큐는 요소의 추가와 제거가 각각 한쪽 끝에서 이루어지는 경우에 사용된다.


## **참고 문헌 및 사이트** 

- [jud00.tistory.com / 자료구조-스택Stack과-큐Queue에-대해서-알아보자](https://jud00.tistory.com/entry/%EC%9E%90%EB%A3%8C%EA%B5%AC%EC%A1%B0-%EC%8A%A4%ED%83%9DStack%EA%B3%BC-%ED%81%90Queue%EC%97%90-%EB%8C%80%ED%95%B4%EC%84%9C-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90)

- chat gpt