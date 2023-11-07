---
layout: post
title: 데크(Deque)
categories: 
  - cs
  - algorithm
description: 데크(Deque) 관련 개념 정리글 입니다.
sitemap: false
---

보통 큐는 선입선출(FIFO) 방식으로 작동한다. 그러나 데크는 양방향 큐 이다.

* this unordered seed list will be replaced by the toc
{:toc}

## 데크(Deque)의 정의

- 데크는 양쪽 끝에서 원소의 삽입과 삭제가 가능한 선형 자료 구조이다.
- 이중 연결 리스트(Double Linked List)를 기반으로 구현되며, 양쪽 끝에서 상수 시간   내에 연산을 수행할 수 있다.

## 데크(Deque)의 기능
    
1. 삽입과 삭제: 데크는 양쪽 끝에서의 삽입과 삭제 연산을 지원한다. 
    - 앞쪽에서 원소를 삽입하는 연산 --> push_front(element)
    - 뒤쪽에서 원소를 삽입하는 연산 --> push_back(element)
    - 앞쪽에서 원소를 삭제하는 연산 --> pop_front()
    - 뒤쪽에서 원소를 삭제하는 연산은 --> pop_back()

2. 접근: 데크는 앞쪽과 뒤쪽에서 원소를 확인할 수 있는 연산을 제공한다.
    - 데크의 앞쪽에 위치한 원소를 반환하는 연산 --> front()
    - 데크의 뒤쪽에 위치한 원소를 반환 --> back()

![데크(Deque)](/assets/img/blog/Deque.png){: width="800" height="400"}

데크(Deque)의 동작 원리
{:.figure}

## 데크(Deque)의 구현방법

### 1. 배열(Array)을 이용한 데크 구현
1. 특징
    - 배열은 연속된 메모리 공간에 원소를 저장하므로 인덱스를 이용한 빠른 접근이 가능
    - 데크의 앞쪽과 뒤쪽에 대한 포인터(인덱스)를 유지하여 삽입과 삭제 연산을 수행
2. 장점
    - 메모리 상에 연속적으로 저장되어 있어 캐시 효율성이 높음
    - 인덱스를 이용한 빠른 접근이 가능
3. 단점
    - 삽입과 삭제 시에는 원소들을 이동시켜야 할 수도 있음
    - 크기가 고정되어 있어 배열의 크기를 동적으로 변경하는 데에는 추가적인 작업이 필요
4. 시간 복잡도
    - 삽입(push_front, push_back) 및 삭제(pop_front, pop_back) 연산의 시간 복잡도 --> $$O(1)$$
5. 코드 구현

~~~python
class ArrayDeque:
    def __init__(self):
        self.deque = []

    def is_empty(self):
        return len(self.deque) == 0

    def push_front(self, item):
        self.deque.insert(0, item)

    def push_back(self, item):
        self.deque.append(item)

    def pop_front(self):
        if not self.is_empty():
            return self.deque.pop(0)
        return None

    def pop_back(self):
        if not self.is_empty():
            return self.deque.pop()
        return None

    def front(self):
        if not self.is_empty():
            return self.deque[0]
        return None

    def back(self):
        if not self.is_empty():
            return self.deque[-1]
        return None
~~~

배열 기반 데크 코드 구현
{:.figure}

### 2. 연결 리스트(Linked List)를 이용한 데크 구현

1. 특징
    - 연결 리스트는 각 노드가 원소와 다음 노드를 가리키는 포인터를 가지는 자료 구조
    - 데크의 앞쪽과 뒤쪽을 가리키는 포인터를 유지하여 삽입과 삭제 연산을 수행
2. 장점
    - 크기가 동적으로 조정될 수 있으며, 삽입과 삭제 연산이 간단하고 빠름
3. 단점
    - 포인터를 유지해야 하므로 추가적인 메모리 공간이 필요하며, 포인터 간의 참조 오버헤드가 있을 수 있음
4. 시간 복잡도
    - 삽입(push_front, push_back) 및 삭제(pop_front, pop_back) 연산의 시간 복잡도 --> O(1)
5. 코드 구현
    ~~~python
    class Node:
        def __init__(self, value):
            self.value = value
            self.prev = None
            self.next = None

    class LinkedListDeque:
        def __init__(self):
            self.head = None
            self.tail = None

        def is_empty(self):
            return self.head is None

        def push_front(self, item):
            new_node = Node(item)
            if self.is_empty():
                self.head = new_node
                self.tail = new_node
            else:
                new_node.next = self.head
                self.head.prev = new_node
                self.head = new_node

        def push_back(self, item):
            new_node = Node(item)
            if self.is_empty():
                self.head = new_node
                self.tail = new_node
            else:
                new_node.prev = self.tail
                self.tail.next = new_node
                self.tail = new_node

        def pop_front(self):
            if not self.is_empty():
                front_value = self.head.value
                if self.head == self.tail:
                    self.head = None
                    self.tail = None
                else:
                    self.head = self.head.next
                    self.head.prev = None
                return front_value
            return None

        def pop_back(self):
            if not self.is_empty():
                back_value = self.tail.value
                if self.head == self.tail:
                    self.head = None
                    self.tail = None
                else:
                    self.tail = self.tail.prev
                    self.tail.next = None
                return back_value
            return None

        def front(self):
            if not self.is_empty():
                return self.head.value
            return None

        def back(self):
            if not self.is_empty():
                return self.tail.value
            return None
~~~

연결리스트 기반 데크 코드 구현
{:.figure}

### 3. 구현 방법 비교

- 배열을 사용한 구현은 메모리 캐시 효율성이 높고 인덱스를 이용한 빠른 접근이 필요한 경우에 적합하다. 그러나 크기 조정이 어려울 수 있다.
- 연결 리스트를 사용한 구현은 크기가 동적으로 조정될 수 있고, 삽입과 삭제가 간단하며 유연한 구조가 필요한 경우에 적합하다. 그러나 포인터 오버헤드가 발생할 수 있다.

## 데크(Deque)의 시간 복잡도 및 활용

### 1. 데크의 시간 복잡도 분석

> enqueue_front(): $$O(1)$$
>
>enqueue_back(): $$O(1)$$
>
>dequeue_front(): $$O(1)$$
>
>dequeue_back(): $$O(1)$$
>
>front(): $$O(1)$$
>
>back(): $$O(1)$$
>
>is_empty(): $$O(1)$$
>
>size(): $$O(1)$$

데크의 모든 연산은 상수 시간 내에 수행된다. 데크의 크기에 관계없이 연산에 소요되는 시간은 일정하므로, 데크는 매우 효율적인 자료 구조이다.

### 2. 데크의 활용

- 양쪽에서의 삽입과 삭제가 필요한 문제에 활용될 수 있다.
- 또한, 데크는 스택과 큐의 기능을 모두 갖춘 자료 구조로서 다양한 응용 분야에서 활용될 수 있다.
- 예시: 함수 호출 스택, 우선순위 큐, 버퍼 관리, 프로세스 스케줄링 등


## **참고 문헌 및 사이트** 

- [https://jee-young.tistory.com/31](https://jee-young.tistory.com/31)

- [https://velog.io/@rhdmstj17/자료구조-데크(Deque)-자료형이란](https://velog.io/@rhdmstj17/%EC%9E%90%EB%A3%8C%EA%B5%AC%EC%A1%B0-%EB%8D%B0%ED%81%ACDeque-%EC%9E%90%EB%A3%8C%ED%98%95%EC%9D%B4%EB%9E%80)

- chat gpt