---
layout: post
title:  "Daily Coding Problems by Category"
date:   2019-04-01 22:22:32 -0700
categories: Python/Java
---
* This will become a table of contents (this text will be scraped).
{:toc}

<!--
<details>
<summary class="lc_h">

- [****]() -- ** [*\(Try ME\)*]()

</summary>
<div>


</div>
</details>
-->

{::options parse_block_html="true" /}


## Math
---


<details>
<summary class="lc_h">

- [**\[Hard\] LC 273. Integer to English Words**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-25-2020-lc-273-hard-integer-to-english-words) -- *Convert a non-negative integer to its English word representation.* [*\(Try ME\)*](https://repl.it/@trsong/Convert-Int-to-English-Words-1)

</summary>
<div>

**Question:** Convert a non-negative integer to its English word representation. 

**Example 1:**
```py
Input: 123
Output: "One Hundred Twenty Three"
```

**Example 2:**
```py
Input: 12345
Output: "Twelve Thousand Three Hundred Forty Five"
```

**Example 3:**
```py
Input: 1234567
Output: "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
```

**Example 4:**
```py
Input: 1234567891
Output: "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One"
```

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] LC 13. Convert Roman Numerals to Decimal**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-8-2020-lc-13-easy-convert-roman-numerals-to-decimal) -- *Given a Roman numeral, find the corresponding decimal value. Inputs will be between 1 and 3999.* [*\(Try ME\)*](https://repl.it/@trsong/Convert-Roman-Format-Number-1)

</summary>
<div>

**Question:** Given a Roman numeral, find the corresponding decimal value. Inputs will be between 1 and 3999.
 
**Note:** Numbers are strings of these symbols in descending order. In some cases, subtractive notation is used to avoid repeated characters. The rules are as follows:
1. I placed before V or X is one less, so 4 = IV (one less than 5), and 9 is IX (one less than 10)
2. X placed before L or C indicates ten less, so 40 is XL (10 less than 50) and 90 is XC (10 less than 100).
3. C placed before D or M indicates 100 less, so 400 is CD (100 less than 500), and 900 is CM (100 less than 1000).

**Example:**
```py
Input: IX
Output: 9

Input: VII
Output: 7

Input: MCMIV
Output: 1904

Roman numerals are based on the following symbols:
I     1
IV    4
V     5
IX    9 
X     10
XL    40
L     50
XC    90
C     100
CD    400
D     500
CM    900
M     1000
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 8. String to Integer (atoi)**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-6-2020-lc-8-medium-string-to-integer-atoi) -- *Given a string, convert it to an integer without using the builtin str function.* [*\(Try ME\)*](https://repl.it/@trsong/String-to-Integer-atoi-1)

</summary>
<div>

**Question:** Given a string, convert it to an integer without using the builtin str function. You are allowed to use ord to convert a character to ASCII code.

Consider all possible cases of an integer. In the case where the string is not a valid integer, return `0`.

**Example:**
```py
atoi('-105')  # -105
```

</div>
</details>


<details>
<summary class="lc_m">


- [**\[Medium\] LC 166. Fraction to Recurring Decimal**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-30-2020-lc-166-medium-fraction-to-recurring-decimal) -- *Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.* [*\(Try ME\)*](https://repl.it/@trsong/Convert-Fraction-to-Recurring-Decimal-1)

</summary>
<div>

**Question:** Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.

If the fractional part is repeating, enclose the repeating part in parentheses.

**Example 1:**
```py
Input: numerator = 1, denominator = 2
Output: "0.5"
```

**Example 2:**
```py
Input: numerator = 2, denominator = 1
Output: "2"
```

**Example 3:**
```py
Input: numerator = 2, denominator = 3
Output: "0.(6)"
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 29. Divide Two Integers**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-23-2020-lc-29-medium-divide-two-integers) -- *Implement integer division without using the division operator: take two numbers,  return a tuple of (dividend, remainder)* [*\(Try ME\)*](https://repl.it/@trsong/Divide-Two-Integers-1)

</summary>
<div>

**Question:** Implement integer division without using the division operator. Your function should return a tuple of `(dividend, remainder)` and it should take two numbers, the product and divisor.

For example, calling `divide(10, 3)` should return `(3, 1)` since the divisor is `3` and the remainder is `1`.

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Overlapping Rectangles**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-17-2020-medium-overlapping-rectangles) -- *Find the area of overlapping rectangles.* [*\(Try ME\)*](https://repl.it/@trsong/Overlapping-Rectangle-Areas-1)

</summary>
<div>

**Question:** You’re given 2 over-lapping rectangles on a plane. For each rectangle, you’re given its bottom-left and top-right points. How would you find the area of their overlap?

</div>
</details>

### Puzzle
---

<details>
<summary class="lc_m">

- [**\[Medium\] Count Attacking Bishop Pairs**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-12-2020-medium-count-attacking-bishop-pairs) -- *Given N bishops, represented as (row, column) tuples on a M by M chessboard, count the number of pairs of bishops attacking each other.* [*\(Try ME\)*](https://repl.it/@trsong/Count-Number-of-Attacking-Bishop-Pairs-1)

</summary>
<div>

**Question:** On our special chessboard, two bishops attack each other if they share the same diagonal. This includes bishops that have another bishop located between them, i.e. bishops can attack through pieces.
 
You are given N bishops, represented as (row, column) tuples on a M by M chessboard. Write a function to count the number of pairs of bishops that attack each other. The ordering of the pair doesn't matter: `(1, 2)` is considered the same as `(2, 1)`.

For example, given `M = 5` and the list of bishops:

```py
(0, 0)
(1, 2)
(2, 2)
(4, 0)
```
The board would look like this:

```py
[b 0 0 0 0]
[0 0 b 0 0]
[0 0 b 0 0]
[0 0 0 0 0]
[b 0 0 0 0]
```

You should return 2, since bishops 1 and 3 attack each other, as well as bishops 3 and 4.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Find Next Biggest Integer**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-1-2020-medium-find-next-biggest-integer) -- *Given an integer n, find the next biggest integer with the same number of 1-bits on.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Next-Biggest-Integer-1)

</summary>
<div>

**Question:** Given an integer `n`, find the next biggest integer with the same number of 1-bits on. For example, given the number `6 (0110 in binary)`, return `9 (1001)`.

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Find Next Greater Permutation**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-2-2020-hard-find-next-greater-permutation) -- *Given a number represented by a list of digits, find the next greater permutation of a number, in terms of lexicographic ordering.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Next-Greater-Permutation-1)

</summary>
<div>

**Question:** Given a number represented by a list of digits, find the next greater permutation of a number, in terms of lexicographic ordering. If there is not greater permutation possible, return the permutation with the lowest value/ordering.

For example, the list `[1,2,3]` should return `[1,3,2]`. The list `[1,3,2]` should return `[2,1,3]`. The list `[3,2,1]` should return `[1,2,3]`.

Can you perform the operation without allocating extra memory (disregarding the input memory)?

</div>
</details>


### XOR
---

<details>
<summary class="lc_m">

- [**\[Medium\] Find Two Elements Appear Once**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-27-2020-medium-find-two-elements-appear-once) -- *Given an array with two elements appear exactly once and all other elements appear exactly twice,find the two elements that appear only once* [*\(Try ME\)*](https://repl.it/@trsong/Find-Two-Elements-Appear-Once-1)

</summary>
<div>

**Question:** Given an array of integers in which two elements appear exactly once and all other elements appear exactly twice, find the two elements that appear only once. Can you do this in linear time and constant space?

**Example:**
```py
Input: [2, 4, 6, 8, 10, 2, 6, 10]
Output: [4, 8] order does not matter
 ```

</div>
</details>

### Hashing
---

<details>
<summary class="lc_e">

- [**\[Easy\] LC 796. Shift-Equivalent Strings**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-25-2020-lc-796-easy-shift-equivalent-strings) -- *Given two strings A and B, return whether or not A can be shifted some number of times to get B.* [*\(Try ME\)*](https://repl.it/@trsong/Check-if-Strings-are-Shift-Equivalent-1)

</summary>
<div>

**Question:** Given two strings A and B, return whether or not A can be shifted some number of times to get B.

For example, if A is `'abcde'` and B is `'cdeab'`, return `True`. If A is `'abc'` and B is `'acb'`, return `False`.

</div>
</details>

## Array
---


<details>
<summary class="lc_m">

- [**\[Medium\] LC 54. Spiral Matrix**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-5-2020-lc-54-medium-spiral-matrix) -- *Given a matrix of n x m elements (n rows, m columns), return all elements of the matrix in spiral order.* [*\(Try ME\)*](https://repl.it/@trsong/Spiral-Matrix-Traversal-1)

</summary>
<div>

**Question:** Given a matrix of n x m elements (n rows, m columns), return all elements of the matrix in spiral order.

**Example 1:**

```py
Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1, 2, 3, 6, 9, 8, 7, 4, 5]
```

**Example 2:**

```py
Input:
[
  [1,  2,  3,  4],
  [5,  6,  7,  8],
  [9, 10, 11, 12]
]
Output: [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
```

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Record the Last N Orders**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-15-2020-easy-record-the-last-n-orders) -- *You run an e-commerce website and want to record the last N order ids in a log.* [*\(Try ME\)*](https://repl.it/@trsong/Record-the-Last-N-Orders-1)

</summary>
<div>

**Question:** You run an e-commerce website and want to record the last `N` order ids in a log. Implement a data structure to accomplish this, with the following API:

- `record(order_id)`: adds the order_id to the log
- `get_last(i)`: gets the ith last element from the log. `i` is guaranteed to be smaller than or equal to `N`.
 
You should be as efficient with time and space as possible.


</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] In-place Array Rotation**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-3-2020-medium-in-place-array-rotation) -- *Write a function that rotates an array by k elements.* [*\(Try ME\)*](https://repl.it/@trsong/Rotate-Array-In-place-1)

</summary>
<div>

Write a function that rotates an array by `k` elements.

For example, `[1, 2, 3, 4, 5, 6]` rotated by two becomes [`3, 4, 5, 6, 1, 2]`.

Try solving this without creating a copy of the array. How many swap or move operations do you need?

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Longest Subarray with Sum Divisible by K**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-11-2020-medium-longest-subarray-with-sum-divisible-by-k) -- *Find the length of the longest subarray with sum of the elements divisible by the given value k.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Longest-Subarray-with-Sum-Divisible-by-K-1)

</summary>
<div>

**Question:** Given an arr[] containing n integers and a positive integer k. The problem is to find the length of the longest subarray with sum of the elements divisible by the given value k.

**Example:**
```py
Input : arr[] = {2, 7, 6, 1, 4, 5}, k = 3
Output : 4
The subarray is {7, 6, 1, 4} with sum 18, which is divisible by 3.
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 152. Maximum Product Subarray**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-14-2020-lc-152-medium-maximum-product-subarray) -- *Given an array that contains both positive and negative integers, find the product of the maximum product subarray.* [*\(Try ME\)*](https://repl.it/@trsong/Maximum-Product-Subarray-1)

</summary>
<div>

**Question:** Given an array that contains both positive and negative integers, find the product of the maximum product subarray. 

**Example 1:**
```py
Input: [6, -3, -10, 0, 2]
Output:  180
Explanation: The subarray is [6, -3, -10]
```

**Example 2:**
```py
Input: [-1, -3, -10, 0, 60]
Output:   60 
Explanation: The subarray is [60]
```

**Example 3:**
```py
Input: [-2, -3, 0, -2, -40]
Output: 80
Explanation: The subarray is [-2, -40]
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] Majority Element**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-28-2020-medium-majority-element) -- *A majority element is an element that appears more than half the time. Given a list with a majority element, find the majority element.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Majority-Element-1)

</summary>
<div>

**Question:** A majority element is an element that appears more than half the time. Given a list with a majority element, find the majority element.

**Example:**
```py
majority_element([3, 5, 3, 3, 2, 4, 3])  # gives 3
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Valid Mountain Array**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-6-2020-easy-valid-mountain-array) -- *Given an array of heights, determine whether the array forms a “mountain” pattern. A mountain pattern goes up and then down.* [*\(Try ME\)*](https://repl.it/@trsong/Valid-Mountain-Array-1)

</summary>
<div>

**Question:** Given an array of heights, determine whether the array forms a "mountain" pattern. A mountain pattern goes up and then down.

**Example:**
```py
validMountainArray([1, 2, 3, 2, 1])  # True
validMountainArray([1, 2, 3])  # False
```

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Matrix Rotation**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-7-2020-medium-matrix-rotation) -- *Given an N by N matrix, rotate it by 90 degrees clockwise.* [*\(Try ME\)*](https://repl.it/@trsong/Matrix-Rotation-In-place-with-flip-1)

</summary>
<div>

**Question:** Given an N by N matrix, rotate it by 90 degrees clockwise.

For example, given the following matrix:

```py
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
 ```

you should return:

```py
[[7, 4, 1],
 [8, 5, 2],
 [9, 6, 3]]
 ```

Follow-up: What if you couldn't use any extra space?
 
</div>
</details>

### Interval
---


<details>
<summary class="lc_e">

- [**\[Easy\] Merge Overlapping Intervals**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-18-2020-easy-merge-overlapping-intervals) -- *Given a list of possibly overlapping intervals, return a new list of intervals where all overlapping intervals have been merged.* [*\(Try ME\)*](https://repl.it/@trsong/Merge-All-Overlapping-Intervals-1)

</summary>
<div>

**Question:** Given a list of possibly overlapping intervals, return a new list of intervals where all overlapping intervals have been merged.

The input list is not necessarily ordered in any way.

For example, given `[(1, 3), (5, 8), (4, 10), (20, 25)]`, you should return `[(1, 3), (4, 10), (20, 25)]`.

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] LC 253. Minimum Lecture Rooms**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-27-2020-lc-253-easy-minimum-lecture-rooms) -- *Given an array of time intervals (start, end) for classroom lectures (possibly overlapping), find the minimum number of rooms required.* [*\(Try ME\)*](https://repl.it/@trsong/Minimum-Required-Lecture-Rooms-1)

</summary>
<div>

**Questions:** Given an array of time intervals `(start, end)` for classroom lectures (possibly overlapping), find the minimum number of rooms required.

For example, given `[(30, 75), (0, 50), (60, 150)]`, you should return `2`.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 986. Interval List Intersections**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-15-2020-lc-986-medium-interval-list-intersections) -- *Given two lists of closed intervals, each list of intervals is pairwise disjoint and in sorted order.* [*\(Try ME\)*](https://repl.it/@trsong/Interval-List-Intersections-1)

</summary>
<div>

**Question:** Given two lists of closed intervals, each list of intervals is pairwise disjoint and in sorted order.

Return the intersection of these two interval lists.

**Example:**
```py
Input: A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
```

</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] LC 57. Insert Interval**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-14-2020-lc-57-hard-insert-interval) -- *Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).* [*\(Try ME\)*](https://repl.it/@trsong/Insert-Interval-1)

</summary>
<div>

**Question:** Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

**Example 1:**
```py
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
```

**Example 2:**
```py
Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
```

</div>
</details>

### Binary Search
---

<details>
<summary class="lc_e">

- [**\[Easy\] First and Last Indices of an Element in a Sorted Array**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-6-2020-easy-first-and-last-indices-of-an-element-in-a-sorted-array) -- *Given a sorted array, A, with possibly duplicated elements, find the indices of the first and last occurrences of a target element.* [*\(Try ME\)*](https://repl.it/@trsong/Find-First-and-Last-Indices-of-an-Element-in-a-Sorted-Arr-1)

</summary>
<div>

**Question:** Given a sorted array, A, with possibly duplicated elements, find the indices of the first and last occurrences of a target element, x. Return -1 if the target is not found.

**Examples:**
```py
Input: A = [1, 3, 3, 5, 7, 8, 9, 9, 9, 15], target = 9
Output: [6, 8]

Input: A = [100, 150, 150, 153], target = 150
Output: [1, 2]

Input: A = [1, 2, 3, 4, 5, 6, 10], target = 9
Output: [-1, -1]
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] LC 300. The Longest Increasing Subsequence**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-19-2020-lc-300-hard-the-longest-increasing-subsequence) -- *Given an array of numbers, find the length of the longest increasing subsequence in the array.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Longest-Increasing-Subsequence-1)

</summary>
<div>

**Question:** Given an array of numbers, find the length of the longest increasing **subsequence** in the array. The subsequence does not necessarily have to be contiguous.

For example, given the array `[0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]`, the longest increasing subsequence has length `6` ie. `[0, 2, 6, 9, 11, 15]`.

</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] Increasing Subsequence of Length K**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-20-2020-hard-increasing-subsequence-of-length-k) -- *Given an int array nums of length n and an int k. Return an increasing subsequence of length k (KIS).* [*\(Try ME\)*](https://repl.it/@trsong/Increasing-Subsequence-of-Length-K-1)

</summary>
<div>

**Question:** Given an int array nums of length n and an int k. Return an increasing subsequence of length k (KIS). Expected time complexity `O(nlogk)`.

**Example 1:**
```py
Input: nums = [10, 1, 4, 8, 2, 9], k = 3
Output: [1, 4, 8] or [1, 4, 9] or [1, 8, 9]
```

**Example 2:**
```py
Input: nums = [10, 1, 4, 8, 2, 9], k = 4
Output: [1, 4, 8, 9]
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 162. Find a Peak Element**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-26-2020-lc-162-easy-find-a-peak-element) -- *Given an unsorted array, in which all elements are distinct, find a “peak” element in O(log N) time.* [*\(Try ME\)*](https://repl.it/@trsong/Find-a-Peak-Element-2)

</summary>
<div>

**Question:** Given an unsorted array, in which all elements are distinct, find a "peak" element in `O(log N)` time.

An element is considered a peak if it is greater than both its left and right neighbors. It is guaranteed that the first and last elements are lower than all others.

**Example 1:**
```py
Input: [5, 10, 20, 15]
Output: 20
The element 20 has neighbours 10 and 15,
both of them are less than 20.
```

**Example 2:**
```py
Input: [10, 20, 15, 2, 23, 90, 67]
Output: 20 or 90
The element 20 has neighbours 10 and 15, 
both of them are less than 20, similarly 90 has neighbous 23 and 67.
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 163. Missing Ranges**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-29-2020-lc-163-medium-missing-ranges) -- *Given a sorted list of numbers, and two integers low and high, return a list of (inclusive) ranges where the numbers are missing.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Missing-Ranges-1)

</summary>
<div>

**Question:** Given a sorted list of numbers, and two integers low and high representing the lower and upper bound of a range, return a list of (inclusive) ranges where the numbers are missing. A range should be represented by a tuple in the format of (lower, upper).

**Example:**
```py
missing_ranges(nums=[1, 3, 5, 10], lower=1, upper=10)
# returns [(2, 2), (4, 4), (6, 9)]
```

</div>
</details>


### Two Pointers
---

<details>
<summary class="lc_e">

- [**\[Easy\] Array of Equal Parts**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-5-2020-easy-array-of-equal-parts) -- *Given an array with only positive integers, determine if exist two numbers such that removing them can break array into 3 equal-sum sub-arrays.* [*\(Try ME\)*](https://repl.it/@trsong/Array-of-Equal-Parts-1)

</summary>
<div>

**Question:** Given an array containing only positive integers, return if you can pick two integers from the array which cuts the array into three pieces such that the sum of elements in all pieces is equal.

**Example 1:**
```py
Input: [2, 4, 5, 3, 3, 9, 2, 2, 2]
Output: True
Explanation: choosing the number 5 and 9 results in three pieces [2, 4], [3, 3] and [2, 2, 2]. Sum = 6.
```

**Example 2:**
```py
Input: [1, 1, 1, 1]
Output: False
```

</div>
</details>


### Sliding Window
---


<details>
<summary class="lc_m">

- [**\[Medium\] LC 438. Anagram Indices Problem**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-13-2020-lc-438-medium-anagram-indices-problem) -- *Given a word W and a string S, find all starting indices in S which are anagrams of W.* [*\(Try ME\)*](https://repl.it/@trsong/Find-All-Anagram-Indices-1)

</summary>
<div>

**Question:**  Given a word W and a string S, find all starting indices in S which are anagrams of W.

For example, given that W is `"ab"`, and S is `"abxaba"`, return `0`, `3`, and `4`.
 
</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 239. Sliding Window Maximum**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-14-2020-lc-239-medium-sliding-window-maximum) -- *Given an array nums, return the max sliding window.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Sliding-Window-Maximum-1)

</summary>
<div>

**Question:** Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.
 

**Example:**

```py
Input: nums = [1, 3, -1, -3, 5, 3, 6, 7], and k = 3
Output: [3, 3, 5, 5, 6, 7] 
```

**Explanation:**
```
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
 ```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LT 386. Longest Substring with At Most K Distinct Characters**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-16-2020-lt-386-medium-longest-substring-with-at-most-k-distinct-characters) -- *Given a string, find the longest substring that contains at most k unique characters.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Longest-Substring-with-At-Most-K-Distinct-Characters-1)

</summary>
<div>

**Question:** Given a string, find the longest substring that contains at most k unique characters. 
 
For example, given `"abcbbbbcccbdddadacb"`, the longest substring that contains 2 unique character is `"bcbbbbcccb"`.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Longest Substring without Repeating Characters**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-17-2020-medium-longest-substring-without-repeating-characters) -- *Given a string, find the length of the longest substring without repeating characters.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Longest-Substring-without-Repeating-Characters-1)

</summary>
<div>

**Question:** Given a string, find the length of the longest substring without repeating characters.

**Note:** Can you find a solution in linear time?
 
**Example:**
```py
lengthOfLongestSubstring("abrkaabcdefghijjxxx") # => 10 as len("abcdefghij") == 10
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 209. Minimum Size Subarray Sum**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-23-2020-lc-209-medium-minimum-size-subarray-sum) -- *Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum >= s.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Minimum-Size-Subarray-Sum-1)

</summary>
<div>

**Question:** Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum >= s. If there isn't one, return 0 instead.

**Example:**
```py
Input: s = 7, nums = [2, 3, 1, 2, 4, 3]
Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.
```

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] LC 383. Ransom Note**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-10-2020-lc-383-easy-ransom-note) -- *Given a magazine of letters and the note he wants to write, determine whether he can construct the word.* [*\(Try ME\)*](https://repl.it/@trsong/Ransom-Note-1)

</summary>
<div>

**Question:** A criminal is constructing a ransom note. In order to disguise his handwriting, he is cutting out letters from a magazine.

Given a magazine of letters and the note he wants to write, determine whether he can construct the word.

**Example 1:**
```py
Input: ['a', 'b', 'c', 'd', 'e', 'f'], 'bed'
Output: True
```

**Example 2:**
```py
Input: ['a', 'b', 'c', 'd', 'e', 'f'], 'cat'
Output: False
```

</div>
</details>

 
## List
---

### Singly Linked List
---

<details>
<summary class="lc_e">

- [**\[Easy\] Remove Duplicates From Linked List**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-10-2020-easy-remove-duplicates-from-linked-list) -- *Given a linked list, remove all duplicate values from the linked list.* [*\(Try ME\)*](https://repl.it/@trsong/Remove-Duplicates-From-Linked-List-1)

</summary>
<div>

**Question:** Given a linked list, remove all duplicate values from the linked list.

For instance, given `1 -> 2 -> 3 -> 3 -> 4`, then we wish to return the linked list `1 -> 2 -> 4`.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 86. Partitioning Linked List**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-28-2020--lc-86-medium-partitioning-linked-list) -- *Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.* [*\(Try ME\)*](https://repl.it/@trsong/Partitioning-Singly-Linked-List-1)

</summary>
<div>

**Question:** Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.

You should preserve the original relative order of the nodes in each of the two partitions.

**Example:**
```py
Input: head = 1->4->3->2->5->2, x = 3
Output: 1->2->2->4->3->5
```


</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Zig-Zag Distinct LinkedList**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-16-2020-easy-zig-zag-distinct-linkedlist) -- *Rearrange distinct linked-list such that they appear in alternating order.* [*\(Try ME\)*](https://repl.it/@trsong/Zig-Zag-Order-of-Distinct-LinkedList-1)

</summary>
<div>

**Question:** Given a linked list with DISTINCT value, rearrange the node values such that they appear in alternating `low -> high -> low -> high ...` form. For example, given `1 -> 2 -> 3 -> 4 -> 5`, you should return `1 -> 3 -> 2 -> 5 -> 4`.

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Remove K-th Last Element from Singly Linked-list**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#may-31-2020-medium-remove-k-th-last-element-from-singly-linked-list) -- *Given a singly linked list and an integer k, remove the kth last element from the list.* [*\(Try ME\)*](https://repl.it/@trsong/Remove-the-K-th-Last-Element-from-Singly-Linked-list-1)

</summary>
<div>

**Question:** Given a singly linked list and an integer k, remove the kth last element from the list. k is guaranteed to be smaller than the length of the list.

**Note:**
- The list is very long, so making more than one pass is prohibitively expensive.
- Do this in constant space and in one pass.

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Intersection of Linked Lists**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-19-2020-easy-intersection-of-linked-lists) -- *Given two singly linked lists. The lists intersect at some node. Find, and return the node. Note: the lists are non-cyclical.* [*\(Try ME\)*](https://repl.it/@trsong/Intersection-of-Linked-Lists-1)

</summary>
<div>

**Question:** You are given two singly linked lists. The lists intersect at some node. Find, and return the node. Note: the lists are non-cyclical.

**Example:**
```py
A = 1 -> 2 -> 3 -> 4
B = 6 -> 3 -> 4
# This should return 3 (you may assume that any nodes with the same value are the same node)
```

</div>
</details>

### Doubly Linked List
---

### Circular Linked List
---

### Fast-slow Pointers
---

## Sort
---


<details>
<summary class="lc_e">

- [**\[Easy\] Make the Largest Number**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-21-2020-easy-make-the-largest-number) -- *Given a number of integers, combine them so it would create the largest number.* [*\(Try ME\)*](https://repl.it/@trsong/Construct-Largest-Number-1)

</summary>
<div>

**Question:** Given a number of integers, combine them so it would create the largest number.

**Example:**
```py
Input: [17, 7, 2, 45, 72]
Output: 77245217
```


</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Sorting Window Range**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-21-2020-medium-sorting-window-range) -- *Given a list of numbers, find the smallest window to sort such that the whole list will be sorted.* [*\(Try ME\)*](https://repl.it/@trsong/Min-Window-Range-to-Sort-1)

</summary>
<div>

**Question:** Given a list of numbers, find the smallest window to sort such that the whole list will be sorted. If the list is already sorted return (0, 0). 

**Example:**
```py
Input: [2, 4, 7, 5, 6, 8, 9]
Output: (2, 4)
Explanation: Sorting the window (2, 4) which is [7, 5, 6] will also means that the whole list is sorted.
```

</div>
</details>

### Merge Sort
---

<details>
<summary class="lc_m">

- [**\[Medium\] Sort Linked List**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-14-2020-medium-sort-linked-list) -- *Given a linked list, sort it in O(n log n) time and constant space.* [*\(Try ME\)*](https://repl.it/@trsong/Sort-Linked-List-1)

</summary>
<div>

**Question:** Given a linked list, sort it in `O(n log n)` time and constant space.

For example, the linked list `4 -> 1 -> -3 -> 99` should become `-3 -> 1 -> 4 -> 99`.


</div>
</details>


### Quick Select & Quick Sort
---

<details>
<summary class="lc_e">

- [**\[Easy\] Find the K-th Largest Number**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-24-2020-easy-find-the-k-th-largest-number) -- *Find the k-th largest number in a sequence of unsorted numbers.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-K-th-Largest-Number-1)

</summary>
<div>

**Question:** Find the k-th largest number in a sequence of unsorted numbers. Can you do this in linear time?

**Example:**
```py
Input: 3, [8, 7, 2, 3, 4, 1, 5, 6, 9, 0]
Output: 7
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Sorting a List With 3 Unique Numbers**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-18-2020-medium-sorting-a-list-with-3-unique-numbers) -- *Given a list of numbers with only 3 unique numbers (1, 2, 3), sort the list in O(n) time.* [*\(Try ME\)*](https://repl.it/@trsong/Sorting-a-List-With-3-Unique-Numbers-1)

</summary>
<div>

**Question:** Given a list of numbers with only `3` unique numbers `(1, 2, 3)`, sort the list in `O(n)` time.

**Example:**
```py
Input: [3, 3, 2, 1, 3, 2, 1]
Output: [1, 1, 2, 2, 3, 3, 3]
```

</div>
</details>

### Counting Sort
---

<details>
<summary class="lc_e">

- [**\[Easy\] LC 451. Sort Characters By Frequency**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-22-2020-lc-451-easy-sort-characters-by-frequency) -- *Given a string, sort it in decreasing order based on the frequency of characters.* [*\(Try ME\)*](https://repl.it/@trsong/Sort-Characters-By-Frequency-1)

</summary>
<div>

**Question:** Given a string, sort it in decreasing order based on the frequency of characters. If there are multiple possible solutions, return any of them.

For example, given the string `tweet`, return `tteew`. `eettw` would also be acceptable.


</div>
</details>


## Tree
---

<details>
<summary class="lc_e">

- [**\[Easy\] Depth of Binary Tree in Peculiar String Representation**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-31-2020-easy-depth-of-binary-tree-in-peculiar-string-representation) -- *Given a binary tree in a peculiar string representation, determine the depth of the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Depth-of-Binary-Tree-in-Peculiar-String-Representation-1)

</summary>
<div>

**Question:** You are given a binary tree in a peculiar string representation. Each node is written in the form `(lr)`, where `l` corresponds to the left child and `r` corresponds to the right child.

If either `l` or `r` is null, it will be represented as a zero. Otherwise, it will be represented by a new `(lr)` pair.
 
Given this representation, determine the depth of the tree.

**Here are a few examples:**
```py
A root node with no children: (00)
A root node with two children: ((00)(00))
An unbalanced tree with three consecutive left children: ((((00)0)0)0)
```

</div>
</details>


### BST
---

<details>
<summary class="lc_m">

- [**\[Medium\] Number of Smaller Elements to the Right**](https://repl.it/@trsong/Find-Number-of-Smaller-Elements-to-the-Right) -- *Given number array,  return the number of smaller elements to the right of each element in the original input array.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Number-of-Smaller-Elements-to-the-Right-1)

</summary>
<div>

**Question:** Given an array of integers, return a new array where each element in the new array is the number of smaller elements to the right of that element in the original input array.

For example, given the array `[3, 4, 9, 6, 1]`, return `[1, 1, 2, 1, 0]`, since:

* There is 1 smaller element to the right of 3
* There is 1 smaller element to the right of 4
* There are 2 smaller elements to the right of 9
* There is 1 smaller element to the right of 6
* There are no smaller elements to the right of 1
 
</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Generate Binary Search Trees**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-11-2020-easy-generate-binary-search-trees) -- *Given an integer N, construct all possible binary search trees with N nodes.* [*\(Try ME\)*](https://repl.it/@trsong/Generate-Binary-Search-Trees-with-N-Nodes-1)

</summary>
<div>

**Question:** Given an integer N, construct all possible binary search trees with N nodes.

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Second Largest in BST**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-24-2020-medium-second-largest-in-bst) -- *Given the root to a binary search tree, find the second largest node in the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Second-Largest-in-BST-1)

</summary>
<div>

**Question:** Given the root to a binary search tree, find the second largest node in the tree.

</div>
</details>

### Recursion
---


<details>
<summary class="lc_m">

- [**\[Medium\] LC 236. Lowest Common Ancestor of a Binary Tree**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-23-2020-lc-236-medium-lowest-common-ancestor-of-a-binary-tree) -- *Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Lowest-Common-Ancestor-of-a-Given-Binary-Tree-1)

</summary>
<div>

**Question:** Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

**Example:**

```py
     1
   /   \
  2     3
 / \   / \
4   5 6   7

LCA(4, 5) = 2
LCA(4, 6) = 1
LCA(3, 4) = 1
LCA(2, 4) = 2
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Tree Isomorphism Problem**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-29-2020-easy-tree-isomorphism-problem) -- *Write a function to detect if two trees are isomorphic.* [*\(Try ME\)*](https://repl.it/@trsong/Is-Binary-Tree-Isomorphic-1)

</summary>
<div>

**Question:** Write a function to detect if two trees are isomorphic. Two trees are called isomorphic if one of them can be obtained from other by a series of flips, i.e. by swapping left and right children of a number of nodes. Any number of nodes at any level can have their children swapped. Two empty trees are isomorphic.


**Example:** 
```py
The following two trees are isomorphic with following sub-trees flipped: 2 and 3, NULL and 6, 7 and 8.

Tree1:
     1
   /   \
  2     3
 / \   /
4   5 6
   / \
  7   8

Tree2:
   1
 /   \
3     2
 \   / \
  6 4   5
       / \
      8   7
```


</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Full Binary Tree**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-11-2020-easy-full-binary-tree) -- *Given a binary tree, remove the nodes in which there is only 1 child, so that the binary tree is a full binary tree.* [*\(Try ME\)*](https://repl.it/@trsong/Prune-to-Full-Binary-Tree-1)

</summary>
<div>

**Question:** Given a binary tree, remove the nodes in which there is only 1 child, so that the binary tree is a full binary tree.

So leaf nodes with no children should be kept, and nodes with 2 children should be kept as well.

**Example:**
```py
Given this tree:
     1
    / \ 
   2   3
  /   / \
 0   9   4

We want a tree like:
     1
    / \ 
   0   3
      / \
     9   4
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 938. Range Sum of BST**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-22-2020-lc-938-easy-range-sum-of-bst) -- *Given a binary search tree and a range [a, b] (inclusive), return the sum of the elements of the binary search tree within the range.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Range-Sum-of-BST-1)

</summary>
<div>

**Question:** Given a binary search tree and a range `[a, b]` (inclusive), return the sum of the elements of the binary search tree within the range.

**Example:**
```py
Given the range [4, 9] and the following tree:

    5
   / \
  3   8
 / \ / \
2  4 6  10

return 23 (5 + 4 + 6 + 8).
```

</div>
</details>




### Traversal
---

<details>
<summary class="lc_e">

- [**\[Easy\] Find Corresponding Node in Cloned Tree**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-27-2020-easy-find-corresponding-node-in-cloned-tree) -- *Given two binary trees that are duplicates of one another, and given a node in one tree, find that corresponding node in the second tree.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Corresponding-Node-in-Cloned-Tree-1)

</summary>
<div>

**Question:** Given two binary trees that are duplicates of one another, and given a node in one tree, find that corresponding node in the second tree. 
 
There can be duplicate values in the tree (so comparing node1.value == node2.value isn't going to work).
 
</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 653. Two Sum in BST**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-19-2020-easy-two-sum-in-bst) -- *Given the root of a binary search tree, and a target K, return two nodes in the tree whose sum equals K.* [*\(Try ME\)*](https://repl.it/@trsong/Two-Sum-in-BST-1)

</summary>
<div>

**Question:** Given the root of a binary search tree, and a target K, return two nodes in the tree whose sum equals K.

**Example:**
```py
Given the following tree and K of 20:

    10
   /   \
 5      15
       /  \
     11    15

Return the nodes 5 and 15.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 230. Kth Smallest Element in a BST**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-1-2020-lc-230-medium-kth-smallest-element-in-a-bst) -- *Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Kth-Smallest-Element-in-a-BST-1)

</summary>
<div>

**Question:** Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

**Example 1:**

```py
Input: 
   3
  / \
 1   4
  \
   2

k = 1
Output: 1
```

**Example 2:**
```py
Input: 
       5
      / \
     3   6
    / \
   2   4
  /
 1

k = 3
Output: 3
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Tree Serialization**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-20-2020-medium-tree-serialization) -- *Given the root of a binary tree. You need to implement 2 functions: Serialize and Deserialize.* [*\(Try ME\)*](https://repl.it/@trsong/Serialize-and-Deserialize-the-Binary-Tree-1)

</summary>
<div>

**Question:** You are given the root of a binary tree. You need to implement 2 functions:

1. `serialize(root)` which serializes the tree into a string representation
2. `deserialize(s)` which deserializes the string back to the original tree that it represents

For this problem, often you will be asked to design your own serialization format. However, for simplicity, let's use the pre-order traversal of the tree.

**Example:**
```py
     1
    / \
   3   4
  / \   \
 2   5   7

serialize(tree)
# returns "1 3 2 # # 5 # # 4 # 7 # #"
```


</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] Largest BST in a Binary Tree**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-3-2020-medium-largest-bst-in-a-binary-tree) -- *You are given the root of a binary tree. Find and return the largest subtree of that tree, which is a valid binary search tree.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Largest-BST-in-a-Binary-Tree-1)

</summary>
<div>

**Question:** You are given the root of a binary tree. Find and return the largest subtree of that tree, which is a valid binary search tree.

**Example1:**
```py
Input: 
      5
    /  \
   2    4
 /  \
1    3

Output:
   2  
 /  \
1    3
```

**Example2:**
```py
Input: 
       50
     /    \
  30       60
 /  \     /  \ 
5   20   45    70
              /  \
            65    80
            
Output: 
      60
     /  \ 
   45    70
        /  \
      65    80
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Symmetric K-ary Tree**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-5-2020-easy-symmetric-k-ary-tree) -- *Given a k-ary tree, figure out if the tree is symmetrical.* [*\(Try ME\)*](https://repl.it/@trsong/Is-Symmetric-K-ary-Tree-1)

</summary>
<div>

**Question:** Given a k-ary tree, figure out if the tree is symmetrical.
 
A k-ary tree is a tree with k-children, and a tree is symmetrical if the data of the left side of the tree is the same as the right side of the tree. 

Here's an example of a symmetrical k-ary tree.

```py
        4
     /     \
    3        3
  / | \    / | \
9   4  1  1  4  9
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 872. Leaf-Similar Trees**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-8-2020-lc-872-easy-leaf-similar-trees) -- *Given two trees, whether they are "leaf similar". Two trees are considered "leaf-similar" if their leaf orderings are the same.* [*\(Try ME\)*](https://repl.it/@trsong/Leaf-Similar-Trees-1)

</summary>
<div>

**Question:** Given two trees, whether they are `"leaf similar"`. Two trees are considered `"leaf-similar"` if their leaf orderings are the same. 

For instance, the following two trees are considered leaf-similar because their leaves are `[2, 1]`:

```py
# Tree1
    3
   / \ 
  5   1
   \
    2 

# Tree2
    7
   / \ 
  2   1
   \
    2 
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Construct BST from Post-order Traversal**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-17-2020-medium-construct-bst-from-post-order-traversal) -- *Given the sequence of keys visited by a postorder traversal of a binary search tree, reconstruct the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Construct-Binary-Search-Tree-from-Post-order-Traversal-1)

</summary>
<div>

**Question:** Given the sequence of keys visited by a postorder traversal of a binary search tree, reconstruct the tree.

**Example:**
```py
Given the sequence 2, 4, 3, 8, 7, 5, you should construct the following tree:

    5
   / \
  3   7
 / \   \
2   4   8
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] In-order & Post-order Binary Tree Traversal**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-29-2020-medium-in-order--post-order-binary-tree-traversal) -- *Given Postorder and Inorder traversals, construct the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Test-Build-Binary-Tree-with-In-order-and-Post-order-Traversal)

</summary>
<div>

**Question:** Given Postorder and Inorder traversals, construct the tree.

**Examples 1:**
```py
Input: 
in_order = [2, 1, 3]
post_order = [2, 3, 1]

Output: 
      1
    /   \
   2     3 
```

**Example 2:**
```py
Input: 
in_order = [4, 8, 2, 5, 1, 6, 3, 7]
post_order = [8, 4, 5, 2, 6, 7, 3, 1]

Output:
          1
       /     \
     2        3
   /    \   /   \
  4     5   6    7
    \
      8
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Pre-order & In-order Binary Tree Traversal**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-28-2020-medium-pre-order--in-order-binary-tree-traversal) -- *Given pre-order and in-order traversals of a binary tree, write a function to reconstruct the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Test-Pre-order-and-In-order-Binary-Tree-Traversal)

</summary>
<div>

**Question:** Given pre-order and in-order traversals of a binary tree, write a function to reconstruct the tree.

For example, given the following preorder traversal:

```py
[a, b, d, e, c, f, g]
```

And the following inorder traversal:

```py
[d, b, e, a, f, c, g]
```

You should return the following tree:

```py
    a
   / \
  b   c
 / \ / \
d  e f  g
```

</div>
</details>

## Trie
---

<details>
<summary class="lc_m">

- [**\[Medium\] LC 421. Maximum XOR of Two Numbers in an Array**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-20-2020-lc-421-medium-maximum-xor-of-two-numbers-in-an-array) -- *Given an array of integers, find the maximum XOR of any two elements.* [*\(Try ME\)*](https://repl.it/@trsong/Maximum-XOR-of-Two-Numbers-in-an-Array-1)

</summary>
<div>

**Question:** Given an array of integers, find the maximum XOR of any two elements.

**Example:**
```py
Input: nums = [3,10,5,25,2,8]
Output: 28
Explanation: The maximum result is 5 XOR 25 = 28.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Group Words that are Anagrams**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-8-2020-medium-group-words-that-are-anagrams) -- *Given a list of words, group the words that are anagrams of each other. (An anagram are words made up of the same letters).* [*\(Try ME\)*](https://repl.it/@trsong/Group-Anagrams-1)

</summary>
<div>

**Question:** Given a list of words, group the words that are anagrams of each other. (An anagram are words made up of the same letters).

**Example:**
```py
Input: ['abc', 'bcd', 'cba', 'cbd', 'efg']
Output: [['abc', 'cba'], ['bcd', 'cbd'], ['efg']]
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Autocompletion**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-16-2020-medium-autocompletion) -- *Given a large set of words and then a single word prefix, find all words that it can complete to.* [*\(Try ME\)*](https://repl.it/@trsong/Autocompletion-1)

</summary>
<div>

**Question:**  Implement auto-completion. Given a large set of words (for instance 1,000,000 words) and then a single word prefix, find all words that it can complete to.

**Example:**
```py
class Solution:
  def build(self, words):
    # Fill this in.
    
  def autocomplete(self, word):
    # Fill this in.

s = Solution()
s.build(['dog', 'dark', 'cat', 'door', 'dodge'])
s.autocomplete('do')  # Return ['dog', 'door', 'dodge']
```

</div>
</details>

## Stack
---


<details>
<summary class="lc_m">

- [**\[Medium\] Nearest Larger Number**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-13-2020-medium-nearest-larger-number) -- *Given an array of numbers and an index i, return the index of the nearest larger number of the number at index i.* [*\(Try ME\)*](https://repl.it/@trsong/Nearest-Larger-Number-1)

</summary>
<div>

**Question:** Given an array of numbers and an index `i`, return the index of the nearest larger number of the number at index `i`, where distance is measured in array indices.

For example, given `[4, 1, 3, 5, 6]` and index `0`, you should return `3`.

If two distances to larger numbers are the equal, then return any one of them. If the array at i doesn't have a nearest larger integer, then return null.

**Follow-up:** If you can preprocess the array, can you do this in constant time?

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 1047. Remove Adjacent Duplicate Characters**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-15-2020-lc-1047-easy-remove-adjacent-duplicate-characters) -- *Given a string, we want to remove 2 adjacent characters that are the same, and repeat the process with the new string until we can no longer* [*\(Try ME\)*](https://repl.it/@trsong/Remove-2-Adjacent-Duplicate-Characters-1)

</summary>
<div>

**Question:** Given a string, we want to remove 2 adjacent characters that are the same, and repeat the process with the new string until we can no longer perform the operation.

**Example:**
```py
remove_adjacent_dup("cabba")
# Start with cabba
# After remove bb: caa
# After remove aa: c
# Returns c
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Index of Larger Next Number**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-17-2020-medium-index-of-larger-next-number) -- *Given a list of numbers, for each element find the next element that is larger than the current element.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Index-of-Larger-Next-Number-1)

</summary>
<div>

**Question:** Given a list of numbers, for each element find the next element that is larger than the current element. Return the answer as a list of indices. If there are no elements larger than the current element, then use -1 instead.

**Example:** 
```py
larger_number([3, 2, 5, 6, 9, 8])
# return [2, 2, 3, 4, -1, -1]
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] The Celebrity Problem**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-27-2020-medium-the-celebrity-problem) -- *Given a list of N people and the above operation, find a way to identify the celebrity in O(N) time.* [*\(Try ME\)*](https://repl.it/@trsong/Test-The-Celebrity-Problem)

</summary>
<div>

**Question:** At a party, there is a single person who everyone knows, but who does not know anyone in return (the "celebrity"). To help figure out who this is, you have access to an `O(1)` method called `knows(a, b)`, which returns `True` if person `a` knows person `b`, else `False`.

Given a list of `N` people and the above operation, find a way to identify the celebrity in `O(N)` time.
 
</div>
</details>


## Priority Queue
---

<details>
<summary class="lc_m">

- [**\[Medium\] M Smallest in K Sorted Lists**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-4-2020-medium-m-smallest-in-k-sorted-lists) -- *Given k sorted arrays of possibly different sizes, find m-th smallest value in the merged array.* [*\(Try ME\)*](https://repl.it/@trsong/Find-M-Smallest-in-K-Sorted-Lists-1)

</summary>
<div>

**Question:** Given k sorted arrays of possibly different sizes, find m-th smallest value in the merged array.

**Example 1:**
```py
Input: [[1, 3], [2, 4, 6], [0, 9, 10, 11]], m = 5
Output: 4
Explanation: The merged array would be [0, 1, 2, 3, 4, 6, 9, 10, 11].  
The 5-th smallest element in this merged array is 4.
```

**Example 2:**
```py
Input: [[1, 3, 20], [2, 4, 6]], m = 2
Output: 2
```

**Example 3:**
```py
Input: [[1, 3, 20], [2, 4, 6]], m = 6
Output: 20
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Largest Product of 3 Elements**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-25-2020-easy-largest-product-of-3-elements) -- *You are given an array of integers. Return the largest product that can be made by multiplying any 3 integers in the array.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Largest-Product-of-3-Elements-in-Array-1)

</summary>
<div>

**Question:** You are given an array of integers. Return the largest product that can be made by multiplying any 3 integers in the array.

**Example:**
```py
Input: [-4, -4, 2, 8]
Output: 128
Explanation: the largest product can be made by multiplying -4 * -4 * 8 = 128.
```

</div>
</details>

### Scheduling
---


<details>
<summary class="lc_m">

- [**\[Medium\] LC 621. Task Scheduler**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-10-2020--lc-621-medium-task-scheduler) -- *Given a char array representing tasks CPU need to do.You need to return the least number of intervals the CPU will take to finish all tasks.* [*\(Try ME\)*](https://repl.it/@trsong/LC-621-Task-Scheduler-1)

</summary>
<div>

**Question:** Given a char array representing tasks CPU need to do. It contains capital letters A to Z where different letters represent different tasks. Tasks could be done without original order. Each task could be done in one interval. For each interval, CPU could finish one task or just be idle.

However, there is a non-negative cooling interval n that means between two same tasks, there must be at least n intervals that CPU are doing different tasks or just be idle.

You need to return the least number of intervals the CPU will take to finish all the given tasks.

**Example:**
```py
Input: tasks = ["A", "A", "A", "B", "B", "B"], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B.
```


</div>
</details>



## Hashmap
---


<details>
<summary class="lc_m">

- [**\[Medium\] Fixed Order Task Scheduler with Cooldown**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-26-2020-medium-fixed-order-task-scheduler-with-cooldown) -- *Given a list of tasks to perform, with a cooldown period. Return execution order.* [*\(Try ME\)*](Fixed Order Task Scheduler with Cooldown-1)

</summary>
<div>

**Question:** We have a list of tasks to perform, with a cooldown period. We can do multiple of these at the same time, but we cannot run the same task simultaneously.

Given a list of tasks, find how long it will take to complete the tasks in the order they are input.

**Example:**
```py
tasks = [1, 1, 2, 1]
cooldown = 2
output: 7 (order is 1 _ _ 1 2 _ 1)
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 763. Partition Labels**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-7-2020-lc-763-medium-partition-labels) -- *Partition the given string into as many parts as possible so that each letter appears in at most one part. Return length for each part.* [*\(Try ME\)*](https://repl.it/@trsong/Partition-Labels-1)

</summary>
<div>

**Question:** A string S of lowercase English letters is given. We want to partition this string into as many parts as possible so that each letter appears in at most one part, and return a list of integers representing the size of these parts.

**Example:**
```py
Input: S = "ababcbacadefegdehijhklij"
Output: [9, 7, 8]
Explanation: The partition is "ababcbaca", "defegde", "hijhklij". 
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits S into less parts.
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 1171. Remove Consecutive Nodes that Sum to 0**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-26-2020-lc-1171-medium-remove-consecutive-nodes-that-sum-to-0) -- *Given a linked list of integers, remove all consecutive nodes that sum up to 0.* [*\(Try ME\)*](https://repl.it/@trsong/Remove-Consecutive-Nodes-that-Sum-to-Zero-1)

</summary>
<div>

**Question:** Given a linked list of integers, remove all consecutive nodes that sum up to 0.

**Example 1:**
```py
Input: 10 -> 5 -> -3 -> -3 -> 1 -> 4 -> -4
Output: 10
Explanation: The consecutive nodes 5 -> -3 -> -3 -> 1 sums up to 0 so that sequence should be removed. 4 -> -4 also sums up to 0 too so that sequence should also be removed.
```

**Example 2:**

```py
Input: 1 -> 2 -> -3 -> 3 -> 1
Output: 3 -> 1
Note: 1 -> 2 -> 1 would also be accepted.
```

**Example 3:**
```py
Input: 1 -> 2 -> 3 -> -3 -> 4
Output: 1 -> 2 -> 4
```

**Example 4:**
```py
Input: 1 -> 2 -> 3 -> -3 -> -2
Output: 1
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 560. Subarray Sum Equals K**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-24-2020-lc-560-medium-subarray-sum-equals-k) -- *Given a list of integers and a number K, return which contiguous elements of the list sum to K.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Number-of-Sub-array-Sum-Equals-K-1)

</summary>
<div>

**Question:** Given a list of integers and a number `K`, return which contiguous elements of the list sum to `K`.

For example, if the list is `[1, 2, 3, 4, 5]` and `K` is `9`, then it should return `[2, 3, 4]`, since `2 + 3 + 4 = 9`.

</div>
</details>


## Set
---

### Basic
---

### Advance
---


## Greedy
---

<details>
<summary class="lc_h">

- [**\[Hard\] LC 358. Rearrange String K Distance Apart**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-11-2020-lc-358-hard-rearrange-string-k-distance-apart) -- *Given a non-empty string str and an integer k, rearrange the string such that the same characters are at least distance k from each other.* [*\(Try ME\)*](https://repl.it/@trsong/Rearrange-Strings-K-Distance-Apart-1)

</summary>
<div>

**Question:** Given a non-empty string str and an integer k, rearrange the string such that the same characters are at least distance k from each other.

All input strings are given in lowercase letters. If it is not possible to rearrange the string, return an empty string "".

**Example 1:**
```py
str = "aabbcc", k = 3
Result: "abcabc"
The same letters are at least distance 3 from each other.
```

**Example 2:**
```py
str = "aaabc", k = 3 
Answer: ""
It is not possible to rearrange the string.
```

**Example 3:**
```py
str = "aaadbbcc", k = 2
Answer: "abacabcd"
Another possible answer is: "abcabcda"
The same letters are at least distance 2 from each other.
```

</div>
</details>


## Divide and Conquer
---


## Graph
---

### BFS
---

<details>
<summary class="lc_e">

- [**\[Easy\] Deepest Node in a Binary Tree**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-2-2020-easy-deepest-node-in-a-binary-tree) -- *You are given the root of a binary tree. Return the deepest node (the furthest node from the root).* [*\(Try ME\)*](https://repl.it/@trsong/Find-Deepest-Node-in-a-Binary-Tree-1)

</summary>
<div>

**Question:** You are given the root of a binary tree. Return the deepest node (the furthest node from the root).

**Example:**
```py
    a
   / \
  b   c
 /
d
The deepest node in this tree is d at depth 3.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Maximum Number of Connected Colors**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-30-2020-medium-maximum-number-of-connected-colors) -- *Given a grid with cells in different colors, find the maximum number of same color cells that are connected.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Maximum-Number-of-Connected-Colors-1)

</summary>
<div>

**Question:** Given a grid with cells in different colors, find the maximum number of same color  cells that are connected.

Note: two cells are connected if they are of the same color and adjacent to each other: left, right, top or bottom. To stay simple, we use integers to represent colors:

The following grid have max 4 connected colors. `[color 3: (1, 2), (1, 3), (2, 1), (2, 2)]`

```py
 [
    [1, 1, 2, 2, 3], 
    [1, 2, 3, 3, 1],
    [2, 3, 3, 1, 2]
 ]
 ```


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Find All Cousins in Binary Tree**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-21-2020-medium-find-all-cousins-in-binary-tree) -- *Given a binary tree and a particular node, find all cousins of that node.* [*\(Try ME\)*](https://repl.it/@trsong/All-Cousins-in-Binary-Tree-1)

</summary>
<div>

**Question:** Two nodes in a binary tree can be called cousins if they are on the same level of the tree but have different parents. 

Given a binary tree and a particular node, find all cousins of that node.


**Example:**
```py
In the following diagram 4 and 6 are cousins:

    1
   / \
  2   3
 / \   \
4   5   6
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Longest Path in Binary Tree**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-31-2020-hard-longest-path-in-binary-tree) -- *Given a binary tree, return any of the longest path.* [*\(Try ME\)*](https://repl.it/@trsong/Test-Longest-Path-in-Binary-Tree)

</summary>
<div>

**Question:** Given a binary tree, return any of the longest path.

**Example 1:**
```py
Input:      1
          /   \
        2      3
      /  \
    4     5

Output: [4, 2, 1, 3] or [5, 2, 1, 3]  
```

**Example 2:**
```py
Input:      1
          /   \
        2      3
      /  \      \
    4     5      6

Output: [4, 2, 1, 3, 6] or [5, 2, 1, 3, 6] 
```

</div>
</details>

### Level-based BFS
---

<details>
<summary class="lc_h">

- [**\[Hard\] LC 934. Shortest Bridge**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-1-2020-lc-934-medium-shortest-bridge) -- *In a given 2D binary array A, there are two islands.Return the smallest number of 0s that must be flipped.* [*\(Try ME\)*](https://repl.it/@trsong/Shortest-Bridge-1)

</summary>
<div>

**Question:** In a given 2D binary array A, there are two islands.  (An island is a 4-directionally connected group of 1s not connected to any other 1s.)

Now, we may change 0s to 1s so as to connect the two islands together to form 1 island.

Return the smallest number of 0s that must be flipped.  (It is guaranteed that the answer is at least 1.)

**Example 1:**
```py
Input: 
[
    [0, 1],
    [1, 0]
]
Output: 1
```

**Example 2:**
```py
Input: 
[
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 1]

]
Output: 2
```
**Example 3:**
```py
Input: 
[
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]
Output: 1
```


</div>
</details>


### DFS
---

<details>
<summary class="lc_h">

- [**\[Hard\] De Bruijn Sequence**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-24-2020-hard-de-bruijn-sequence) -- *Given a set of characters C and an integer k,  find a De Bruijn Sequence.* [*\(Try ME\)*](https://repl.it/@trsong/Find-De-Bruijn-Sequence-1)

</summary>
<div>

**Question:** Given a set of characters `C` and an integer `k`, a **De Bruijn** Sequence is a cyclic sequence in which every possible k-length string of characters in C occurs exactly once.
 
**Background:** De Bruijn Sequence can be used to shorten a brute-force attack on a PIN-like code lock that does not have an "enter" key and accepts the last n digits entered. For example, a digital door lock with a 4-digit code would have B (10, 4) solutions, with length 10000. Therefore, only at most 10000 + 3 = 10003 (as the solutions are cyclic) presses are needed to open the lock. Trying all codes separately would require 4 × 10000 = 40000 presses.

**Example1:**
```py
Input: C = [0, 1], k = 3
Output: 0011101000
All possible strings of length three (000, 001, 010, 011, 100, 101, 110 and 111) appear exactly once as sub-strings in C.
```

**Example2:**
```py
Input: C = [0, 1], k = 2
Output: 01100
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Level of tree with Maximum Sum**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-3-2020-easy-level-of-tree-with-maximum-sum) -- *Given a binary tree, find the level in the tree where the sum of all nodes on that level is the greatest.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Level-of-tree-with-Maximum-Sum-1)

</summary>
<div>

**Question:** Given a binary tree, find the level in the tree where the sum of all nodes on that level is the greatest.

**Example:**
```py
The following tree should return level 1:
    1          Level 0 - Sum: 1
   / \
  4   5        Level 1 - Sum: 9 
 / \ / \
3  2 4 -1      Level 2 - Sum: 8
```

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Flatten a Nested Dictionary**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-8-2020-easy-flatten-a-nested-dictionary) -- *Write a function to flatten a nested dictionary. Namespace the keys with a period.* [*\(Try ME\)*](https://repl.it/@trsong/Flatten-Nested-Dictionary-1)

</summary>
<div>

**Question:** Write a function to flatten a nested dictionary. Namespace the keys with a period.

**Example:**
```py
Given the following dictionary:
{
    "key": 3,
    "foo": {
        "a": 5,
        "bar": {
            "baz": 8
        }
    }
}

it should become:
{
    "key": 3,
    "foo.a": 5,
    "foo.bar.baz": 8
}

You can assume keys do not contain dots in them, i.e. no clobbering will occur.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Circle of Chained Words**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-9-2020-medium-circle-of-chained-words) -- *Given a list of words, determine if there is a way to ‘chain’ all the words in a circle.* [*\(Try ME\)*](https://repl.it/@trsong/Contains-Circle-of-Chained-Words-1)

</summary>
<div>

**Question:** Two words can be 'chained' if the last character of the first word is the same as the first character of the second word.

Given a list of words, determine if there is a way to 'chain' all the words in a circle.

**Example:**
```py
Input: ['eggs', 'karat', 'apple', 'snack', 'tuna']
Output: True
Explanation:
The words in the order of ['apple', 'eggs', 'snack', 'karat', 'tuna'] creates a circle of chained words.
```

</div>
</details>


### Topological Sort
---

<details>
<summary class="lc_h">

- [**\[Hard\] Order of Course Prerequisites**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-18-2020-hard-order-of-course-prerequisites) -- *Given a hashmap of courseId key to a list of courseIds values. Return a sorted ordering of courses such that we can finish all courses.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Order-of-Course-Prerequisites-1)

</summary>
<div>

We're given a hashmap associating each courseId key with a list of courseIds values, which represents that the prerequisites of courseId are courseIds. Return a sorted ordering of courses such that we can finish all courses.

Return null if there is no such ordering.

For example, given `{'CSC300': ['CSC100', 'CSC200'], 'CSC200': ['CSC100'], 'CSC100': []}`, should return `['CSC100', 'CSC200', 'CSC300']`.

</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] LC 329. Longest Increasing Path in a Matrix**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-16-2020-lc-329-hard-longest-increasing-path-in-a-matrix) -- *Given an integer matrix, find the length of the longest increasing path.* [*\(Try ME\)*](https://repl.it/@trsong/Longest-Increasing-Path-in-a-Matrix-1)

</summary>
<div>

**Question:** Given an integer matrix, find the length of the longest increasing path.

From each cell, you can either move to four directions: left, right, up or down. You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).

**Example 1:**
```py
Input: nums = 
[
  [9,9,4],
  [6,6,8],
  [2,1,1]
] 
Output: 4 
Explanation: The longest increasing path is [1, 2, 6, 9].
```

**Example 2:**
```py
Input: nums = 
[
  [3,4,5],
  [3,2,6],
  [2,2,1]
] 
Output: 4 
Explanation: The longest increasing path is [3, 4, 5, 6]. Moving diagonally is not allowed.
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Shortest Uphill and Downhill Route**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-7-2020-hard-shortest-uphill-and-downhill-route) -- *A runner decide the route goes entirely uphill at first, and then entirely downhill.  Find the length of the shortest of such route.* [*\(Try ME\)*](https://repl.it/@trsong/Shortest-Uphill-and-Downhill-Route-1)

</summary>
<div>

**Question:** A competitive runner would like to create a route that starts and ends at his house, with the condition that the route goes entirely uphill at first, and then entirely downhill.

Given a dictionary of places of the form `{location: elevation}`, and a dictionary mapping paths between some of these locations to their corresponding distances, find the length of the shortest route satisfying the condition above. Assume the runner's home is location `0`.

**Example:**
```py
Suppose you are given the following input:

elevations = {0: 5, 1: 25, 2: 15, 3: 20, 4: 10}
paths = {
    (0, 1): 10,
    (0, 2): 8,
    (0, 3): 15,
    (1, 3): 12,
    (2, 4): 10,
    (3, 4): 5,
    (3, 0): 17,
    (4, 0): 10
}

In this case, the shortest valid path would be 0 -> 2 -> 4 -> 0, with a distance of 28.
```


</div>
</details>


### Union Find
---

<details>
<summary class="lc_m">

- [**\[Medium\] Is Bipartite**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-22-2020-medium-is-bipartite) -- *Given an undirected graph G, check whether it is bipartite.* [*\(Try ME\)*](https://repl.it/@trsong/Is-a-Graph-Bipartite-1)

</summary>
<div>

**Question:** Given an undirected graph G, check whether it is bipartite. Recall that a graph is bipartite if its vertices can be divided into two independent sets, U and V, such that no edge connects vertices of the same set.

**Example:**
```py
is_bipartite(vertices=3, edges=[(0, 1), (1, 2), (2, 0)])  # returns False 
is_bipartite(vertices=2, edges=[(0, 1), (1, 0)])  # returns True. U = {0}. V = {1}. 
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 261. Graph Valid Tree**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-10-2020-lc-261-medium-graph-valid-tree) -- *Given n nodes and a list of undirected edges, write a function to check whether these edges make up a valid tree.* [*\(Try ME\)*](https://repl.it/@trsong/Graph-Valid-Tree-1)

</summary>
<div>

**Question:** Given n nodes labeled from 0 to n-1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.

**Example 1:**
```py
Input: n = 5, and edges = [[0,1], [0,2], [0,3], [1,4]]
Output: True
```

**Example 2:**
```py
Input: n = 5, and edges = [[0,1], [1,2], [2,3], [1,3], [1,4]]
Output: False
```

</div>
</details>


### A-Star
---


<details>
<summary class="lc_h">

- [**\[Hard\] Minimum Step to Reach One**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-26-2020-easy-minimum-step-to-reach-one) -- *Given a positive integer N, find the smallest number of steps it will take to reach 1.* [*\(Try ME\)*](https://repl.it/@trsong/Test-Find-Minimum-Step-to-Reach-One)

</summary>
<div>

**Question:** Given a positive integer `N`, find the smallest number of steps it will take to reach `1`.

There are two kinds of permitted steps:
- You may decrement `N` to `N - 1`.
- If `a * b = N`, you may decrement `N` to the larger of `a` and `b`.
 
For example, given `100`, you can reach `1` in five steps with the following route: `100 -> 10 -> 9 -> 3 -> 2 -> 1`.

</div>
</details>


## Backtracking
---

<details>
<summary class="lc_h">

- [**\[Hard\] LC 37. Sudoku Solver**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-13-2020-lc-37-hard-sudoku-solver) -- *Write a program to solve a Sudoku puzzle by filling the empty cells.* [*\(Try ME\)*](https://repl.it/@trsong/Sudoku-Solver-1)

</summary>
<div>

**Question:** Write a program to solve a Sudoku puzzle by filling the empty cells.

A sudoku solution must satisfy all of the following rules:
- Each of the digits `1-9` must occur exactly once in each row.
- Each of the digits `1-9` must occur exactly once in each column.
- Each of the the digits `1-9` must occur exactly once in each of the 9 `3x3` sub-boxes of the grid.
- Empty cells are indicated `0`.

**Example:**
```py
Input: [
    [3, 0, 6, 5, 0, 8, 4, 0, 0], 
    [5, 2, 0, 0, 0, 0, 0, 0, 0], 
    [0, 8, 7, 0, 0, 0, 0, 3, 1], 
    [0, 0, 3, 0, 1, 0, 0, 8, 0], 
    [9, 0, 0, 8, 6, 3, 0, 0, 5], 
    [0, 5, 0, 0, 9, 0, 6, 0, 0], 
    [1, 3, 0, 0, 0, 0, 2, 5, 0], 
    [0, 0, 0, 0, 0, 0, 0, 7, 4], 
    [0, 0, 5, 2, 0, 6, 3, 0, 0]
]

Possible output:
[
    [3, 1, 6, 5, 7, 8, 4, 9, 2],
    [5, 2, 9, 1, 3, 4, 7, 6, 8],
    [4, 8, 7, 6, 2, 9, 5, 3, 1],
    [2, 6, 3, 4, 1, 5, 9, 8, 7],
    [9, 7, 4, 8, 6, 3, 1, 2, 5],
    [8, 5, 1, 7, 9, 2, 6, 4, 3],
    [1, 3, 8, 9, 4, 7, 2, 5, 6],
    [6, 9, 2, 3, 5, 1, 8, 7, 4],
    [7, 4, 5, 2, 8, 6, 3, 1, 9]
]
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 78. Generate All Subsets**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-6-2020-lc-78-medium-generate-all-subsets) -- *Given a list of unique numbers, generate all possible subsets without duplicates. This includes the empty set as well.* [*\(Try ME\)*](https://repl.it/@trsong/Generate-All-the-Subsets-1)

</summary>
<div>

**Question:** Given a list of unique numbers, generate all possible subsets without duplicates. This includes the empty set as well.

**Example:**
```py
generate_all_subsets([1, 2, 3])
# [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Graph Coloring**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-3-2020-hard-graph-coloring) -- *Given an undirected graph and an integer k, determine whether color with k colors such that no two adjacent nodes share the same color.* [*\(Try ME\)*](https://repl.it/@trsong/k-Graph-Coloring-1)

</summary>
<div>

**Question:** Given an undirected graph represented as an adjacency matrix and an integer `k`, determine whether each node in the graph can be colored such that no two adjacent nodes share the same color using at most `k` colors.

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Generate All Possible Subsequences**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-7-2020-easy-generate-all-possible-subsequences) -- *Given a string, generate all possible subsequences of the string.* [*\(Try ME\)*](https://repl.it/@trsong/Generate-All-Possible-Subsequences-1)

</summary>
<div>

**Question:** Given a string, generate all possible subsequences of the string.

For example, given the string `xyz`, return an array or set with the following strings:

```py
x
y
z
xy
xz
yz
xyz
```

Note that `zx` is not a valid subsequence since it is not in the order of the given string.


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 93. All Possible Valid IP Address Combinations**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-23-2020-lc-93-medium-all-possible-valid-ip-address-combinations) -- *Given a string of digits, generate all possible valid IP address combinations.* [*\(Try ME\)*](https://repl.it/@trsong/Find-All-Possible-Valid-IP-Address-Combinations-1)

</summary>
<div>

**Question:** Given a string of digits, generate all possible valid IP address combinations.

IP addresses must follow the format A.B.C.D, where A, B, C, and D are numbers between `0` and `255`. Zero-prefixed numbers, such as `01` and `065`, are not allowed, except for `0` itself.

For example, given `"2542540123"`, you should return `['254.25.40.123', '254.254.0.123']`
 
</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] The N Queens Puzzle**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-2-2020-hard-the-n-queens-puzzle) -- *Given N, returns the number of possible arrangements of the board where N queens can be placed on the board without attacking each other.* [*\(Try ME\)*](https://repl.it/@trsong/Solve-the-N-Queen-Problem-1)

</summary>
<div>

**Question:** You have an N by N board. Write a function that, given N, returns the number of possible arrangements of the board where N queens can be placed on the board without threatening each other, i.e. no two queens share the same row, column, or diagonal.

</div>
</details>


## Dynamic Programming
---

### 1D DP
---

<details>
<summary class="lc_h">

- [**\[Hard\] Largest Divisible Pairs Subset**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-28-2020-hard-largest-divisible-pairs-subset) -- *Given a set of distinct positive integers, find the largest subset such that every pair of elements has divisible counterpart in the subset.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Largest-Divisible-Pairs-Subset-1)

</summary>
<div>

**Question:** Given a set of distinct positive integers, find the largest subset such that every pair of elements in the subset `(i, j)` satisfies either `i % j = 0` or `j % i = 0`.

For example, given the set `[3, 5, 10, 20, 21]`, you should return `[5, 10, 20]`. Given `[1, 3, 6, 24]`, return `[1, 3, 6, 24]`.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 139. Word Break**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-9-2020-lc-139-medium-word-break) -- *Given a non-empty string s and a dictionary with a list of non-empty words, determine if s can be segmented into a space-separated sequence.* [*\(Try ME\)*](https://repl.it/@trsong/Word-Break-Problem-1)

</summary>
<div>

**Question:** Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words.

**Note:**
* The same word in the dictionary may be reused multiple times in the segmentation.
* You may assume the dictionary does not contain duplicate words.

**Example 1:**
```py
Input: s = "Pseudocode", wordDict = ["Pseudo", "code"]
Output: True
Explanation: Return true because "Pseudocode" can be segmented as "Pseudo code".
```

**Example 2:**
```py
Input: s = "applepenapple", wordDict = ["apple", "pen"]
Output: True
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
             Note that you are allowed to reuse a dictionary word.
```

**Example 3:**
```py
Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
Output: False
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 91. Decode Ways**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-19-2020-lc-91-medium-decode-ways) -- *A message containing letters from A-Z is being encoded to numbers using the following mapping: 'A' -> 1 'B' -> 2 ... 'Z' -> 26.* [*\(Try ME\)*](https://repl.it/@trsong/Number-of-Decode-Ways-1)

</summary>
<div>

**Question:** A message containing letters from `A-Z` is being encoded to numbers using the following mapping:

 ```py
 'A' -> 1
 'B' -> 2
 ...
 'Z' -> 26
 ```
 Given an encoded message containing digits, determine the total number of ways to decode it.

**Example 1:**
```py
Input: "12"
Output: 2
Explanation: It could be decoded as AB (1 2) or L (12).
```

**Example 2:**
```py
Input: "10"
Output: 1
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] Making Changes**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-9-2020-medium-making-changes) -- *Given a list of possible coins in cents, and an amount (in cents) n, return the minimum number of coins needed to create the amount n.* [*\(Try ME\)*](https://repl.it/@trsong/Making-Changes-Problem-1)

</summary>
<div>

**Question:** Given a list of possible coins in cents, and an amount (in cents) n, return the minimum number of coins needed to create the amount n. If it is not possible to create the amount using the given coin denomination, return None.

**Example:**
```py
make_change([1, 5, 10, 25], 36)  # gives 3 coins (25 + 10 + 1) 
```

</div>
</details>


### 2D DP
---


<details>
<summary class="lc_h">

- [**\[Hard\] LC 76. Minimum Window Substring**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-12-2020-lc-76-hard-minimum-window-substring) -- *Given a string and a set of characters, return the shortest substring containing all the characters in the set.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Minimum-Window-Substring-1)

</summary>
<div>

**Question:** Given a string and a set of characters, return the shortest substring containing all the characters in the set.

For example, given the string `"figehaeci"` and the set of characters `{a, e, i}`, you should return `"aeci"`.

If there is no substring containing all the characters in the set, return null.
 
</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] LC 727. Minimum Window Subsequence**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-15-2020-lc-727-hard-minimum-window-subsequence) -- *Given strings S and T, find the minimum (contiguous) substring W of S, so that T is a subsequence of W.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Minimum-Window-Subsequence-1)

</summary>
<div>

**Question:** Given strings S and T, find the minimum (contiguous) substring W of S, so that T is a subsequence of W.

If there is no such window in S that covers all characters in T, return the empty string "". If there are multiple such minimum-length windows, return the one with the left-most starting index.

**Example:**
```py
Input: 
S = "abcdebdde", T = "bde"
Output: "bcde"

Explanation: 
"bcde" is the answer because it occurs before "bdde" which has the same length.
"deb" is not a smaller window because the elements of T in the window must occur in order.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Longest Common Subsequence**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-18-2020-medium-longest-common-subsequence) -- *Given two sequences, find the length of longest subsequence present in both of them.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Longest-Common-Subsequence-1)

</summary>
<div>

**Question:** Given two sequences, find the length of longest subsequence present in both of them. 

A subsequence is a sequence that appears in the same relative order, but not necessarily contiguous.

**Example 1:**

```py
Input:  "ABCD" and "EDCA"
Output:  1
	
Explanation:
LCS is 'A' or 'D' or 'C'
```

**Example 2:**

```py
Input: "ABCD" and "EACB"
Output:  2
	
Explanation: 
LCS is "AC"
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 446. Count Arithmetic Subsequences**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-22-2020-lc-446-medium-count-arithmetic-subsequences) -- *Given an array of n positive integers. The task is to count the number of Arithmetic Subsequence in the array.* [*\(Try ME\)*](https://repl.it/@trsong/Count-Number-of-Arithmetic-Subsequences-1)

</summary>
<div>

**Question:** Given an array of n positive integers. The task is to count the number of Arithmetic Subsequence in the array. Note: Empty sequence or single element sequence is also Arithmetic Sequence. 

**Example 1:**
```py
Input : arr[] = [1, 2, 3]
Output : 8
Arithmetic Subsequence from the given array are:
[], [1], [2], [3], [1, 2], [2, 3], [1, 3], [1, 2, 3].
```

**Example 2:**
```py
Input : arr[] = [10, 20, 30, 45]
Output : 12
```

**Example 3:**
```py
Input : arr[] = [1, 2, 3, 4, 5]
Output : 23
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 718. Longest Common Sequence of Browsing Histories**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-9-2020-lc-718-medium-longest-common-sequence-of-browsing-histories) -- *Write a function that takes two users’ browsing histories as input and returns the longest contiguous sequence of URLs that appear in both.* [*\(Try ME\)*](https://repl.it/@trsong/Longest-Common-Sequence-of-Browsing-Histories-1)

</summary>
<div>

**Question:** We have some historical clickstream data gathered from our site anonymously using cookies. The histories contain URLs that users have visited in chronological order.

Write a function that takes two users' browsing histories as input and returns the longest contiguous sequence of URLs that appear in both.

For example, given the following two users' histories:

```py
user1 = ['/home', '/register', '/login', '/user', '/one', '/two']
user2 = ['/home', '/red', '/login', '/user', '/one', '/pink']
```

You should return the following:

```py
['/login', '/user', '/one']
```

</div>
</details>


### DP+ 
---

<details>
<summary class="lc_m">

- [**\[Medium\] Maximum Circular Subarray Sum**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-2-2020-medium-maximum-circular-subarray-sum) -- *Given a circular array, compute its maximum subarray sum in O(n) time. A subarray can be empty, and in this case the sum is 0.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Maximum-Circular-Subarray-Sum-1)

</summary>
<div>

**Question:** Given a circular array, compute its maximum subarray sum in `O(n)` time. A subarray can be empty, and in this case the sum is 0.

**Example 1:**
```py
Input: [8, -1, 3, 4]
Output: 15 
Explanation: we choose the numbers 3, 4, and 8 where the 8 is obtained from wrapping around.
```

**Example 2:**
```py
Input: [-4, 5, 1, 0]
Output: 6 
Explanation: we choose the numbers 5 and 1.
```


</div>
</details>


## String
---


<details>
<summary class="lc_e">

- [**\[Easy\] Run-length String Encode and Decode**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-21-2020-easy-run-length-string-encode-and-decode) -- *Implement run-length encoding and decoding.* [*\(Try ME\)*](https://repl.it/@trsong/Run-length-String-Encode-and-Decode-1)

</summary>
<div>

**Question:** Run-length encoding is a fast and simple method of encoding strings. The basic idea is to represent repeated successive characters as a single count and character. For example, the string `"AAAABBBCCDAA"` would be encoded as `"4A3B2C1D2A"`.

Implement run-length encoding and decoding. You can assume the string to be encoded have no digits and consists solely of alphabetic characters. You can assume the string to be decoded is valid.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Zig-Zag String**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-4-2020-medium-zig-zag-string) -- *Given a string and a number of lines k, print the string in zigzag form.* [*\(Try ME\)*](https://repl.it/@trsong/Print-Zig-Zag-String-1)

</summary>
<div>

**Question:** Given a string and a number of lines k, print the string in zigzag form. In zigzag, characters are printed out diagonally from top left to bottom right until reaching the kth line, then back up to top right, and so on.

**Example:**
```py
Given the sentence "thisisazigzag" and k = 4, you should print:
t     a     g
 h   s z   a
  i i   i z
   s     g
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 1021. Remove One Layer of Parenthesis**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-12-2020-lc-1021-easy-remove-one-layer-of-parenthesis) -- *Given a valid parenthesis string, remove the outermost layers of the parenthesis string and return the new parenthesis string.* [*\(Try ME\)*](https://repl.it/@trsong/Remove-One-Layer-of-Parenthesis-1)

</summary>
<div>

**Question:** Given a valid parenthesis string (with only '(' and ')', an open parenthesis will always end with a close parenthesis, and a close parenthesis will never start first), remove the outermost layers of the parenthesis string and return the new parenthesis string.

If the string has multiple outer layer parenthesis (ie (())()), remove all outer layers and construct the new string. So in the example, the string can be broken down into (()) + (). By removing both components outer layer we are left with () + '' which is simply (), thus the answer for that input would be ().

**Example 1:**
```py
Input: '(())()'
Output: '()'
```

**Example 2:**
```py
Input: '(()())'
Output: '()()'
```

**Example 3:**
```py
Input: '()()()'
Output: ''
```


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Tokenization**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-20-2020-medium-tokenization) -- *Given a dictionary of words and a string made up of those words (no spaces), return the original sentence in a list.* [*\(Try ME\)*](https://repl.it/@trsong/String-Tokenization-1)

</summary>
<div>

**Questions:** Given a dictionary of words and a string made up of those words (no spaces), return the original sentence in a list. If there is more than one possible reconstruction, return any of them. If there is no possible reconstruction, then return null.

**Example 1:**
```py
Input: ['quick', 'brown', 'the', 'fox'], 'thequickbrownfox'
Output: ['the', 'quick', 'brown', 'fox']
```

**Example 2:**
```py
Input: ['bed', 'bath', 'bedbath', 'and', 'beyond'], 'bedbathandbeyond'
Output:  Either ['bed', 'bath', 'and', 'beyond'] or ['bedbath', 'and', 'beyond']
```

</div>
</details>

### Anagram
---

### Palindrome
---

<details>
<summary class="lc_e">

- [**\[Easy\] LC 680. Remove Character to Create Palindrome**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-1-2020-lc-680-easy-remove-character-to-create-palindrome) -- *Given a string, determine if you can remove any character to create a palindrome.* [*\(Try ME\)*](https://repl.it/@trsong/Remove-Character-to-Create-Palindrome-1)

</summary>
<div>

**Question:** Given a string, determine if you can remove any character to create a palindrome.

**Example 1:**
```py
Input: "abcdcbea"
Output: True 
Explanation: Remove 'e' gives "abcdcba"
```

**Example 2:**
```py
Input: "abccba"
Output: True
```

**Example 3:**
```py
Input: "abccaa"
Output: False
```

</div>
</details>

{::options parse_block_html="false" /}
