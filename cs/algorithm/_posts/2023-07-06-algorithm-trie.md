---
layout: post
title: 트라이(Trie)
categories: 
  - cs
  - algorithm
description: 트라이(Trie) 관련 개념 정리글 입니다.
sitemap: false
---

트라이(Trie)는 문자열을 저장하고 검색하기 위한 트리 기반의 자료구조이다.

* this unordered seed list will be replaced by the toc
{:toc}

## 트라이의 개념과 동작 원리
### 개념
- 정의: 트라이는 문자열을 저장하는 효율적인 자료구조로, 노드와 엣지로 구성된 트리 형태를 가진다. 각 노드는 문자를 저장하고, 엣지는 다음 노드로의 연결을 나타낸다.
- 공통된 접두사: 트라이는 문자열의 공통된 접두사를 효율적으로 공유하는 특징이 있다. 이는 문자열의 중복을 줄이고 저장 공간을 절약하는데 도움을 준다.

### 동작 원리
- 동작 원리와 구조: 트라이는 항상 루트 노드에서 시작하며, 문자열의 각 문자를 노드와 엣지를 따라 내려가면서 저장하거나 검색한다. 각 노드는 해당 위치에서의 문자를 나타내며, 리프 노드는 문자열의 종료를 나타낸다.
- 자식 노드와 문자열 매칭 관계: 각 노드의 자식 노드는 가능한 문자의 집합을 나타낸다. 트라이의 경로를 따라가면서 문자열을 매칭하는 과정에서, 매칭되는 문자열이 없거나 중간에 중복된 경로가 발생하지 않는다.

![트라이의 동작원리](/assets/img/blog/Trie.png)
    
'abc','ab','car'을 순차적으로 삽입한 모습
{:.figcaption}

## 트라이의 구현 방법

### 1. 연결 리스트
- 각 노드를 연결하는 방식으로 구현할 수 있고 노드는 문자 및 자식 노드에 대한 포인터를 포함한다.
- 동적으로 크기를 조정해야 하거나 문자열의 길이가 다양한 경우에 적합하다.

~~~python
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False

    class Trie:
        def __init__(self):
            self.root = TrieNode()

        def insert(self, word):
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end_of_word = True

        def search(self, word):
            node = self.root
            for char in word:
                if char not in node.children:
                    return False
                node = node.children[char]
            return node.is_end_of_word

        def starts_with(self, prefix):
            node = self.root
            for char in prefix:
                if char not in node.children:
                    return False
                node = node.children[char]
            return True
~~~

연결 리스트를 이용한 트라이 구현 
{:.figure}

### 2. 배열
- 배열의 인덱스는 문자에 대응하며, 각 요소는 노드를 나타낸다. 
- 일반적으로 알파벳의 경우 26개의 요소를 가지는 배열을 사용한다.
- 고정된 크기를 가지거나 문자열 길이가 동일한 경우에 유용하다.

~~~python
    class TrieNode:
        def __init__(self):
            self.children = [None] * 26
            self.is_end_of_word = False

    class Trie:
        def __init__(self):
            self.root = TrieNode()

        def _char_to_index(self, char):
            return ord(char) - ord('a')

        def insert(self, word):
            node = self.root
            for char in word:
                index = self._char_to_index(char)
                if not node.children[index]:
                    node.children[index] = TrieNode()
                node = node.children[index]
            node.is_end_of_word = True

        def search(self, word):
            node = self.root
            for char in word:
                index = self._char_to_index(char)
                if not node.children[index]:
                    return False
                node = node.children[index]
            return node.is_end_of_word

        def starts_with(self, prefix):
            node = self.root
            for char in prefix:
                index = self._char_to_index(char)
                if not node.children[index]:
                    return False
                node = node.children[index]
            return True
~~~

배열을 이용한 트라이 구현 
{:.figure}

### 3. 딕셔너리
- 파이썬의 딕셔너리 자료구조를 사용하여 트라이를 구현할 수 있다.
- 각 노드는 문자를 키로 사용하고, 자식 노드를 값으로 가지는 형태이다.
- 간결한 구현과 동적 크기 조정이 필요한 경우에 사용할 수 있다.

~~~python
    class TrieNode:
        def __init__(self):
            self.children = {}

    class Trie:
        def __init__(self):
            self.root = TrieNode()

        def insert(self, word):
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end_of_word = True

        def search(self, word):
            node = self.root
            for char in word:
                if char not in node.children:
                    return False
                node = node.children[char]
            return node.is_end_of_word

        def starts_with(self, prefix):
            node = self.root
            for char in prefix:
                if char not in node.children:
                    return False
                node = node.children[char]
            return True
~~~

딕셔너리를 이용한 트라이 구현 
{:.figure}

## 트라이의 시간복잡도 분석
- 트라이의 시간 복잡도는 주로 트라이의 높이와 탐색 연산의 길이에 의해 결정된다. 높이를 $$h$$라고 하면 삽입, 검색, 삭제 3가지 연산 모두 각 문자마다 한번의 탐색이 필요하므로 시간복잡도는 $$O(h)$$이다. 하지만, 최악의 경우 문자열의 길이에 비례하여 $$O(n)$$의 시간복잡도가 나올 수 있다. 

- 일반적으로 트라이는 문자열 처리와 검색에 특화된 자료 구조이므로, 문자열의 길이에 따라 성능이 크게 영향을 받을 수 있다.

## 트라이의 장단점
1. 장점
    - 빠른 문자열 검색: 트라이는 문자열의 접두사에 해당하는 모든 가능한 단어를 빠르게 찾을 수 있다.
    - 효율적인 메모리 사용: 트라이는 문자열의 공통 접두어를 공유하여 메모리를 효율적으로 사용할 수 있다. 
    - 빠른 삽입과 삭제: 트라이는 문자열을 삽입하고 삭제하는 연산도 효율적으로 수행할 수 있다.

2. 단점
    - 공간 복잡도: 트라이는 문자열의 길이에 따라 노드의 수가 증가하므로 공간 복잡도가 상대적으로 높을 수 있다. 
    - 구현 복잡도: 트라이는 노드와 엣지를 이용한 구조로 복잡한 구현을 요구할 수 있다. 
    - 정렬된 데이터에 제한적: 트라이는 문자열의 접두사에 대한 검색에 최적화되어 있기 때문에 정렬된 데이터에는 적합하지 않을 수 있다. 

## 트라이의 활용 예시
1. 자동 완성: 트라이는 입력 중인 문자열의 접두사에 해당하는 모든 가능한 단어를 빠르게 찾을 수 있기 때문에 자동 완성 기능에 활용된다. 예를 들어, 검색 엔진의 검색어 자동 완성이나 텍스트 편집기에서의 단어 자동 완성에 사용될 수 있다.

2. 사전 구현: 트라이는 단어 사전을 구현하는 데에도 활용된다. 단어 추천, 철자 검사, 단어 빈도수 계산 등의 기능을 구현할 때 트라이를 사용하여 효율적인 데이터 구조를 구성할 수 있다.

3. 자연어 처리: 트라이는 자연어 처리 분야에서도 활용된다. 형태소 분석, 문장 구문 분석, 개체명 인식 등의 작업에서 단어들을 저장하고 검색하는 데에 트라이를 사용할 수 있다.

## **참고 문헌 및 사이트** 

- [https://velog.io/@kimdukbae/%EC%9E%90%EB%A3%8C%EA%B5%AC%EC%A1%B0-%ED%8A%B8%EB%9D%BC%EC%9D%B4-Trie](https://velog.io/@kimdukbae/%EC%9E%90%EB%A3%8C%EA%B5%AC%EC%A1%B0-%ED%8A%B8%EB%9D%BC%EC%9D%B4-Trie)

- chat gpt