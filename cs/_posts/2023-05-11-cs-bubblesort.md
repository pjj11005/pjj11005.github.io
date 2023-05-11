---
layout: post
title: 버블 정렬(Bubble Sort)
description: 버블 정렬(Bubble Sort) 관련 개념 정리글 입니다.
sitemap: false
---

버블 정렬은 간단하고 이해하기 쉬운 정렬 알고리즘 중 하나이다. 따라서 초보 개발자들이 공부할 때 많이 사용하는 알고리즘 중 하나이기도 하다.

## 1. 정의

버블 정렬은 인접한 두 요소를 비교하여 정렬하는 알고리즘이다. 인접한 두 요소를 비교하여 큰 값이 오른쪽에 위치하게 되면, 이를 왼쪽으로 교환하는 과정을 반복하여 정렬을 완료한다.

## 2. 동작 원리

1. 버블 정렬은 (배열의 길이 - 1) 만큼의 패스를 반복하며, 각 패스마다 배열의 첫 요소부터 인접한 두 요소를 비교한다.

2. 이 때, 두 요소의 크기에 따라 교환 여부를 결정하고, 해당 패스가 끝나면 가장 큰 요소가 맨 오른쪽에 위치하게 된다. 

3. 이 과정을 반복하여 정렬을 완료한다.

~~~python
# Python program for implementation of Bubble Sort

def bubbleSort(arr):
	n = len(arr)

	# Traverse through all array elements
	for i in range(n):

		# Last i elements are already in place
		for j in range(0, n-i-1):

			# traverse the array from 0 to n-i-1
			# Swap if the element found is greater
			# than the next element
			if arr[j] > arr[j+1]:
				arr[j], arr[j+1] = arr[j+1], arr[j]


# Driver code to test above
if __name__ == "__main__":
arr = [5, 1, 4, 2, 8]

bubbleSort(arr)

print("Sorted array is:")
for i in range(len(arr)):
	print("%d" % arr[i], end=" ")

#output
# Sorted array: 
# 1 2 4 5 8 
~~~

버블 정렬 코드 예시 (시간 복잡도 O(n<sup>2</sup>))
{:.figure}


## 3. 특성

- 장점

    - 알고리즘을 이해하고 구현하기 쉽다.

    - 추가 메모리 공간이 필요하지 않는다.

    - 다양한 유형의 데이터에 대한 적용이 가능하다.

    - 안정적인 정렬 알고리즘이다.

> 안정 정렬 - 원래 데이터에서 같은 값을 가진 원소들의 상대적인 위치가 정렬된 후에도 유지되는 정렬 알고리즘
{:.lead}

- 단점

    - 버블 정렬은 O(n<sup>2</sup>)의 시간 복잡도를 가지므로 대용량 데이터 세트에 대해 매우 느리다.

    - 데이터를 여러 번 통과해야 하므로 대규모 데이터 세트에는 효율적이지 않다.

    - 버블 정렬은 비교 기반 정렬 알고리즘으로, 입력 데이터 세트에서 요소의 상대적인 순서를 결정하기 위해 비교 연산자가 필요하다. 이것이 반드시 단점은 아니지만 경우에 따라 알고리즘의 효율성을 제한할 수 있다.

## 4. 최적화 방법

- 교환이 일어나지 않은 패스가 있다면, 정렬이 이미 완료된 것으로 보고 종료하여 최적화 할 수 있다.

~~~python
# Optimized Python program for implementation of Bubble Sort

def bubbleSort(arr):
	n = len(arr)
	# Traverse through all array elements
	for i in range(n):
		swapped = False

		# Last i elements are already in place
		for j in range(0, n-i-1):

			# traverse the array from 0 to n-i-1
			# Swap if the element found is greater
			# than the next element
			if arr[j] > arr[j+1]:
				arr[j], arr[j+1] = arr[j+1], arr[j]
				swapped = True
		if (swapped == False):
			break

# Driver code to test above
if __name__ == "__main__":
	arr = [64, 34, 25, 12, 22, 11, 90]

	bubbleSort(arr)

	print("Sorted array is:")
	for i in range(len(arr)):
		print("%d" % arr[i], end=" ")

#output
# Sorted array: 
# 11 12 22 25 34 64 90 
~~~

버블 정렬 최적화 코드 예시 (시간 복잡도 O(n<sup>2</sup>))
{:.figure}

## 5. 재귀적인 방법

- 가장 큰 요소를 해당 위치에 배치하고 다른 모든 요소에 대해 동일한 작업을 계속 수행하는 방법 (각각의 패스에서 비교 진행 후 재귀적으로 다음 작업 진행)

~~~python
def bubbleSort(a, n):

	sorted = True
	# we are assuming that array is sorted
	for i in range(0, n-1):
		if a[i] > a[i+1]:
			#swap
			a[i], a[i+1] = a[i+1], a[i]
			sorted = False
			# now array is not sorted
			
		# if there are no swaps then we can
		# say that array is sorted.

	if sorted == False:
		#recursively calling until it was sorted.
		bubbleSort(a, n)

# Driver code to test above
if __name__ == "__main__":
	arr = [5, 4, 8, 2, 9, 7, 3]
	n = len(arr)
	bubbleSort(arr, n)

	print("Sorted array : ", end=" ")
	for i in range(len(arr)):
		print("%d" % arr[i], end=" ")

#output
# Sorted array : 2 3 4 5 7 8 9 
~~~

버블 정렬 재귀 구현 코드 예시 (시간 복잡도 O(n<sup>2</sup>))
{:.figure}

## 6. 버블 정렬 Worst case 분석

- 최악의 경우 및 평균적인 시간 복잡도: O(n<sup>2</sup>). 
    - 최악의 경우는 배열이 내림차순으로 정렬되었을 때 발생한다.

> 총 스왑 수 = 총 비교 수\
총 비교 수(최악의 경우) = n(n-1)/2\
총 스왑 수(최악의 경우) = n(n-1)/2
{:.lead}


- 최상의 경우 시간 복잡도: O(N). 
    - 최상의 경우는 배열이 이미 정렬되어 있을 때 발생한다.


#### **참고 문헌 및 사이트** 

- [https://www.geeksforgeeks.org/bubble-sort/](https://www.geeksforgeeks.org/bubble-sort/)

- chat gpt