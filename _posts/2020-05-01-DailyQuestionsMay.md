---
layout: post
title:  "Daily Coding Problems May to Jul"
date:   2020-05-01 22:22:32 -0700
categories: Python/Java
---
* This will become a table of contents (this text will be scraped).
{:toc}

## Daily Coding Problems

### Enviroment Setup
---

**Python 2.7 Playground:** [https://repl.it/languages/python](https://repl.it/languages/python)

**Python 3 Playground:** [https://repl.it/languages/python3](https://repl.it/languages/python3) 

**Java Playground:** [https://repl.it/languages/java](https://repl.it/languages/java)

<!--
### Apr 6, 2020 \[Medium\] JSON Encoding
---
> **Question:** Write a function that takes in a number, string, list, or dictionary and returns its JSON encoding. It should also handle nulls.

**Example:**
```py
Given the following input:
[None, 123, ["a", "b"], {"c":"d"}]

You should return the following, as a string:
'[null, 123, ["a", "b"], {"c": "d"}]'
```
-->

### May 20, 2020  \[Medium\] Split a Binary Search Tree
---
> **Question:** Given a binary search tree (BST) and a value s, split the BST into 2 trees, where one tree has all values less than or equal to s, and the other tree has all values greater than s while maintaining the tree structure of the original BST. You can assume that s will be one of the node's value in the BST. Return both tree's root node as a tuple.

**Example:**
```py
Given the following tree, and s = 2
     3
   /   \
  1     4
   \     \
    2     5

Split into two trees:
 1    And   3
  \          \
   2          4
               \
                5
```


### May 19, 2020 \[Easy\] Max and Min with Limited Comparisons
---
> **Question:** Given a list of numbers of size `n`, where `n` is greater than `3`, find the maximum and minimum of the list using less than `2 * (n - 1)` comparisons.

**My thoughts:** The idea is to use Tournament Method. Think about each number as a team in the tournament. One team, zero matches. Two team, one match. N team, let's break them into half and use two matches to get best of best and worest of worest:

```
T(n) = 2 * T(n/2) + 2
T(1) = 0
T(2) = 1

=>
T(n) = 3n/2 - 2 
```


**Solution with Divide And Conquer:** [https://repl.it/@trsong/Find-Max-and-Min-with-Limited-Comparisons](https://repl.it/@trsong/Find-Max-and-Min-with-Limited-Comparisons)
```py
import unittest
from functools import reduce

class MinMaxPair(object):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val


def find_min_max(nums):
    return find_min_max_recur(nums, 0, len(nums) - 1)


def find_min_max_recur(nums, lo, hi):
    if lo > hi:
        return None
    elif lo == hi:
        return MinMaxPair(nums[lo], nums[lo])
    elif lo == hi - 1:
        return MinMaxPair(nums[lo], nums[hi]) if nums[lo] < nums[hi] else MinMaxPair(nums[hi], nums[lo])
    else:
        mid = lo + (hi - lo) // 2
        left_res = find_min_max_recur(nums, lo, mid)
        right_res = find_min_max_recur(nums, mid+1, hi)
        if left_res and right_res:
            return MinMaxPair(min(left_res.min_val, right_res.min_val), max(left_res.max_val, right_res.max_val))
        else:
            return left_res or right_res 
    

#######################################
# Testing Utilities
#######################################
class Number(int):
    def __new__(self, value):
        self.history = []
        return int.__new__(self, value)

    def __cmp__(self, other):
        self.history.append((self, other))
        return int.__cmp__(self, other)

    def count_comparison(self):
        return len(self.history)

    def get_history(self):
        return '\n'.join(self.history)


class GetMinMaxSpec(unittest.TestCase):
    def assert_find_min_max(self, nums):
        min_val = min(nums)
        max_val = max(nums)
        n = len(nums)
        mapped_nums = list(map(Number, nums))
        res = find_min_max(mapped_nums)
        self.assertEqual(min_val, res.min_val)
        self.assertEqual(max_val, res.max_val)
        total_num_comparisons = sum(map(lambda num: num.count_comparison(), mapped_nums))
        history = reduce(lambda accu, num: accu + '\n' + num.get_history(), mapped_nums, '')
        if total_num_comparisons > 0:
            msg = "Expect less than %d comparisons but gives %d\n%s" % (2 * (n - 1), total_num_comparisons, history) 
            self.assertLess(total_num_comparisons, 2 * (n - 1), msg)

    def test_empty_list(self):
        self.assertIsNone(find_min_max([]))

    def test_list_with_one_element(self):
        self.assert_find_min_max([-1])

    def test_list_with_two_elements(self):
        self.assert_find_min_max([1, 2])

    def test_increasing_list(self):
        self.assert_find_min_max([1, 2, 3, 4])

    def test_list_with_duplicated_element(self):
        self.assert_find_min_max([-1, 1, -1, 1])

    def test_long_list(self):
        self.assert_find_min_max([1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1])


if __name__ == '__main__':
    unittest.main(exit=False)
```


### May 18, 2020 LC 435 \[Medium\] Non-overlapping Intervals
---
> **Question:** Given a collection of intervals, find the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.
>
> Intervals can "touch", such as `[0, 1]` and `[1, 2]`, but they won't be considered overlapping.
>
> For example, given the intervals `(7, 9), (2, 4), (5, 8)`, return `1` as the last interval can be removed and the first two won't overlap.
>
> The intervals are not necessarily sorted in any order.

**My thoughts:** Think about the problem backwards: to remove the least number of intervals to make non-overlapping is equivalent to pick most number of non-overlapping intervals and remove the rest. Therefore we just need to pick most number of non-overlapping intervals that can be done with greedy algorithm by sorting on end time and pick as many intervals as possible.

**Greedy Solution:** [https://repl.it/@trsong/Non-overlapping-Intervals](https://repl.it/@trsong/Non-overlapping-Intervals)
```py
import unittest

def remove_overlapping_intervals(intervals):
    inteval_by_end_time = sorted(intervals, key=lambda start_end: start_end[-1])
    prev_end_time = None
    chosen = 0

    for start, end in inteval_by_end_time:
        if start >= prev_end_time:
            chosen += 1
            prev_end_time = end
    
    return len(intervals) - chosen


class RemoveOveralppingIntervalSpec(unittest.TestCase):
    def test_example(self):
        intervals = [(7, 9), (2, 4), (5, 8)]
        expected = 1  # remove (7, 9)
        self.assertEqual(expected, remove_overlapping_intervals(intervals))

    def test_no_intervals(self):
        self.assertEqual(0, remove_overlapping_intervals([]))

    def test_one_interval(self):
        intervals = [(0, 42)]
        expected = 0
        self.assertEqual(expected, remove_overlapping_intervals(intervals))

    def test_return_least_number_of_interval_to_remove(self):
        intervals = [(1, 2), (2, 3), (3, 4), (1, 3)]
        expected = 1  # remove (1, 3)
        self.assertEqual(expected, remove_overlapping_intervals(intervals))

    def test_duplicated_intervals(self):
        intervals = [(1, 2), (1, 2), (1, 2)]
        expected = 2  # remove (1, 2), (1, 2)
        self.assertEqual(expected, remove_overlapping_intervals(intervals))

    def test_non_overlapping_intervals(self):
        intervals = [(1, 2), (2, 3)]
        expected = 0
        self.assertEqual(expected, remove_overlapping_intervals(intervals))

    def test_share_end_points(self):
        intervals = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        expected = 3  # remove (1, 3), (1, 4), (2, 4)
        self.assertEqual(expected, remove_overlapping_intervals(intervals))

    def test_overlapping_intervals(self):
        intervals = [(1, 4), (2, 3), (4, 6), (8, 9)]
        expected = 1  # remove (2, 3)
        self.assertEqual(expected, remove_overlapping_intervals(intervals))

    def test_should_remove_first_interval(self):
        intervals = [(1, 9), (2, 3), (5, 7)]
        expected = 1 # remove (1, 9)
        self.assertEqual(expected, remove_overlapping_intervals(intervals))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### May 17, 2020 LC 480 \[Hard\] Sliding Window Median
---
> **Question:** Given an array of numbers arr and a window of size k, print out the median of each window of size k starting from the left and moving right by one position each time.
>
> For example, given that `k = 3` and array equals `[-1, 5, 13, 8, 2, 3, 3, 1]`, your function should return `[5, 8, 8, 3, 3, 3]`

**Explanation:**
```py
Recall that the median of an even-sized list is the average of the two middle numbers.
5 <- median of [-1, 5, 13]
8 <- median of [5, 13, 8]
8 <- median of [13, 8, 2]
3 <- median of [8, 2, 3]
3 <- median of [2, 3, 3]
3 <- median of [3, 3, 1]
```

**My thoughts:** Maintain a max heap and a min heap where all elem in max heap is smaller than min heap. Together, they can hold k elements. For each new elem incoming, there must be an element kickout. Determine if they are on the same side. If not, we will need to balance both heap by moving element to the other. As removed element, we can lazily remove them: after each iteration, we need to double-check the head of each heap in order to make sure those element are within the window.

**Solution with Priority Queue:** [https://repl.it/@trsong/Sliding-Window-Median](https://repl.it/@trsong/Sliding-Window-Median)
```py
import unittest
from Queue import PriorityQueue

def sliding_median(nums, k):
    max_heap = PriorityQueue()  # left heap
    min_heap = PriorityQueue()  # right heap

    for i in xrange(k):
        max_heap.put((-nums[i], i))

    while min_heap.qsize() < max_heap.qsize():
        move_head(max_heap, min_heap)

    res = [find_median(max_heap, min_heap, k)]

    for i in xrange(k, len(nums)):
        cur_num = nums[i]
        removed_num = nums[i - k]
        if cur_num >= min_heap.queue[0][0]:
            # insert new elem into the right heap
            min_heap.put((cur_num, i))
            if removed_num <= min_heap.queue[0][0]:
                # the removed elem create a hole on the left heap
                move_head(min_heap, max_heap)
        else:
            # insert new elem into left heap
            max_heap.put((-cur_num, i))
            if removed_num >= min_heap.queue[0][0]:
                # the removed elem create a hole on the right heap
                move_head(max_heap, min_heap)

        while not max_heap.empty() and max_heap.queue[0][1] <= i - k:
            max_heap.get()

        while not min_heap.empty() and min_heap.queue[0][1] <= i - k:
            min_heap.get()

        res.append(find_median(max_heap, min_heap, k))

    return res


def move_head(pq1, pq2):
    num, i = pq1.get()
    pq2.put((-num, i))


def find_median(max_heap, min_heap, k):
    if k % 2 == 0:
        neg_num1, _ = max_heap.queue[0]
        num2, _ = min_heap.queue[0]
        return (-neg_num1 + num2) * 0.5
    else:
        num, _ = min_heap.queue[0]
        return num


class SlidingMedianSpec(unittest.TestCase):
    def test_example(self):
        k, nums = 3, [-1, 5, 13, 8, 2, 3, 3, 1]
        expected = [5, 8, 8, 3, 3, 3]
        self.assertEqual(expected, sliding_median(nums, k))

    def test_example2(self):
        k, nums = 3, [1, 3, -1, -3, 5, 3, 6, 7]
        expected = [1, -1, -1, 3, 5, 6]
        self.assertEqual(expected, sliding_median(nums, k))

    def test_k_equals_array_size(self):
        k, nums = 5, [1, 2, 3, 4, 5]
        expected = [3]
        self.assertEqual(expected, sliding_median(nums, k))

    def test_k_equals_array_size2(self):
        k, nums = 4, [1, 2, 3, 4]
        expected = [2.5]
        self.assertEqual(expected, sliding_median(nums, k))

    def test_k_equals_one(self):
        k, nums = 1, [3, 1, 2, 4, 1, 2]
        expected = [3, 1, 2, 4, 1, 2]
        self.assertEqual(expected, sliding_median(nums, k))

    def test_even_sized_window(self):
        k, nums = 2, [1, 3, 3, 1, 1, 3, 3, 1]
        expected = [2, 3, 2, 1, 2, 3, 2]
        self.assertEqual(expected, sliding_median(nums, k))

    def test_array_with_duplicated_elements(self):
        k, nums = 5, [1, 1, 1, 1, 1, 1, 1, 1]
        expected = [1, 1, 1, 1]
        self.assertEqual(expected, sliding_median(nums, k))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 16, 2020 LC 394 \[Medium\] Decode String
---
> **Question:** Given a string with a certain rule: `k[string]` should be expanded to string `k` times. So for example, `3[abc]` should be expanded to `abcabcabc`. Nested expansions can happen, so `2[a2[b]c]` should be expanded to `abbcabbc`.


**Solution:** [https://repl.it/@trsong/LC-394-Decode-String](https://repl.it/@trsong/LC-394-Decode-String)
```py
import unittest

def decode_string(encoded_string):
    stack = []
    count = 0
    pattern = []
    for ch in encoded_string:
        if ch.isdigit():
            count = 10 * count + int(ch)
        elif ch == '[':
            prev_str = ''.join(pattern)
            prev_count = count
            stack.append((prev_str, prev_count))
            count = 0
            pattern = []
        elif ch == ']':
            pattern_str = ''.join(pattern)
            prev_str, count = stack.pop()
            pattern = [prev_str, pattern_str * count]
            count = 0
        else:
            pattern.append(ch)
    return ''.join(pattern)
        

class DecodeStringSpec(unittest.TestCase):
    def test_example1(self):
        input = "3[abc]"
        expected = 3 * "abc"
        self.assertEqual(expected, decode_string(input))

    def test_example2(self):
        input = "2[a2[b]c]"
        expected = 2 * ('a' + 2 * 'b' + 'c')
        self.assertEqual(expected, decode_string(input)) 

    def test_example3(self):
        input = "3[a]2[bc]"
        expected = "aaabcbc"
        self.assertEqual(expected, decode_string(input)) 
    
    def test_example4(self):
        input = "3[a2[c]]"
        expected = "accaccacc"
        self.assertEqual(expected, decode_string(input)) 

    def test_example5(self):
        input = "2[abc]3[cd]ef"
        expected = "abcabccdcdcdef"
        self.assertEqual(expected, decode_string(input)) 

    def test_empty_string(self):
        self.assertEqual("", decode_string(""))
        self.assertEqual("", decode_string("42[]"))

    def test_not_decode_negative_number_of_strings(self):
        input = "-3[abc]"
        expected = "-abcabcabc"
        self.assertEqual(expected, decode_string(input))

    def test_duplicate_more_than_10_times(self):
        input = "233[ab]"
        expected =  233 * "ab"
        self.assertEqual(expected, decode_string(input))

    def test_2Level_nested_encoded_string(self):
        input = "2[3[a]3[bc]2[d]]4[2[e]]"
        expected = 2 * (3*"a" + 3*"bc" + 2*"d") + 4*2*"e"
        self.assertEqual(expected, decode_string(input))

    def test_3Level_nested_encoded_string(self):
        input = "2[a2[b3[c]d4[ef]g]h]"
        expected = 2*('a' + 2*('b' + 3*'c' + 'd' + 4 * 'ef' + 'g' ) + 'h')
        self.assertEqual(expected, decode_string(input))


if __name__ == "__main__":
    unittest.main(exit=False)
```

### May 15, 2020 \[Medium\] Longest Consecutive Sequence in an Unsorted Array
---
> **Question:** Given an array of integers, return the largest range, inclusive, of integers that are all included in the array.
>
> For example, given the array `[9, 6, 1, 3, 8, 10, 12, 11]`, return `(8, 12)` since `8, 9, 10, 11, and 12` are all in the array.

**My thoughts:** We can use BFS to find max length of consecutive sequence. Two number are neighbor if they differ by 1. Thus for any number we can scan though its left-most and right-most neighbor recursively. 

**Solution with BFS:** [https://repl.it/@trsong/Longest-Consecutive-Sequence-in-an-Unsorted-Array](https://repl.it/@trsong/Longest-Consecutive-Sequence-in-an-Unsorted-Array)
```py
import unittest

def longest_consecutive_seq(nums):
    if not nums:
        return None

    num_set = set(nums)
    max_length = 0
    global_lo = global_hi = 0

    while num_set:
        pivot = next(iter(num_set))
        num_set.remove(pivot)
        lower_bound = pivot - 1
        upper_bound = pivot + 1

        while num_set and lower_bound in num_set:
            num_set.remove(lower_bound)
            lower_bound -= 1

        while num_set and upper_bound in num_set:
            num_set.remove(upper_bound)
            upper_bound += 1

        length = upper_bound - lower_bound - 1
        if length > max_length:
            max_length = length
            global_lo = lower_bound
            global_hi = upper_bound

    return (global_lo + 1, global_hi - 1)


class LongestConsecutiveSeqSpec(unittest.TestCase):
    def test_example(self):
        nums = [9, 6, 1, 3, 8, 10, 12, 11]
        expected = (8, 12)
        self.assertEqual(expected, longest_consecutive_seq(nums))

    def test_example2(self):
        nums = [2, 10, 3, 12, 5, 4, 11, 8, 7, 6, 15]
        expected = (2, 8)
        self.assertEqual(expected, longest_consecutive_seq(nums))

    def test_empty_array(self):
        self.assertIsNone(longest_consecutive_seq([]))

    def test_no_consecutive_sequence(self):
        nums = [1, 3, 5, 7]
        possible_solutions = [(1, 1), (3, 3), (5, 5), (7, 7)]
        self.assertIn(longest_consecutive_seq(nums), possible_solutions)

    def test_more_than_one_solution(self):
        nums = [0, 3, 4, 5, 9, 10, 13, 14, 15, 19, 20, 1]
        possible_solutions = [(3, 5), (13, 15)]
        self.assertIn(longest_consecutive_seq(nums), possible_solutions)

    def test_longer_array(self):
        nums = [10, 21, 45, 22, 7, 2, 67, 19, 13, 45, 12, 11, 18, 16, 17, 100, 201, 20, 101]
        expected = (16, 22)
        self.assertEqual(expected, longest_consecutive_seq(nums))

    def test_entire_array_is_continous(self):
        nums = [0, 1, 2, 3, 4, 5]
        expected = (0, 5)
        self.assertEqual(expected, longest_consecutive_seq(nums))

    def test_array_with_duplicated_numbers(self):
        nums = [0, 0, 3, 3, 2, 2, 1, 4, 7, 8, 10]
        expected = (0, 4)
        self.assertEqual(expected, longest_consecutive_seq(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### May 14, 2020 \[Medium\] Bitwise AND of a Range
---
> **Question:** Write a function that returns the bitwise AND of all integers between M and N, inclusive.

**Solution:** [https://repl.it/@trsong/Bitwise-AND-of-a-Range](https://repl.it/@trsong/Bitwise-AND-of-a-Range)
```py
import unittest

def bitwise_and_of_range(m, n):
    """
      0b10110001   -> m
    & 0b10110010
    & 0b10110011
    & 0b10110100   
    & 0b10110101   -> n
    = 0b10110000

    We need to find longest prefix between m and n. That means we can keep removing least significant bit of n until n < m. Then we can get the longest prefix. 
    """

    if m > n:
        m, n = n, m
    
    while m < n:
        # remove the least significant bit
        n &= n - 1  

    return n


class BitwiseAndOfRangeSpec(unittest.TestCase):
    def test_same_end_point(self):
        m, n, expected = 42, 42, 42
        self.assertEqual(expected, bitwise_and_of_range(m, n))

    def test_end_share_same_prefix(self):
        m = 0b110100
        n = 0b110011
        expected = 0b110000
        self.assertEqual(expected, bitwise_and_of_range(m, n))

    def test_one_end_has_zero(self):
        m = 0b1111111
        n = 0b0
        expected = 0b0
        self.assertEqual(expected, bitwise_and_of_range(m, n))

    def test_neither_end_has_same_digit(self):
        m = 0b100101
        n = 0b011010
        expected = 0b0
        self.assertEqual(expected, bitwise_and_of_range(m, n))

    def test_bitwise_all_number_within_range(self):
        """
          0b1100
        & 0b1101
        & 0b1110
        & 0b1111
        = 0b1100
        """
        m = 12  # 0b1100
        n = 15  # 0b1111
        expected = 0b1100
        self.assertEqual(expected, bitwise_and_of_range(m, n))

    def test_bitwise_all_number_within_range2(self):
        """
          0b10001
        & 0b10010
        & 0b10011
        = 0b10000
        """
        m = 17  # 0b10001
        n = 19  # 0b10011
        expected = 0b10000
        self.assertEqual(expected, bitwise_and_of_range(m, n))

    def test_both_end_share_some_digits(self):
        """
          0b01010
        & 0b01011
        ...
        & 0b10000
        ...
        & 0b10100
        = 0b00000
        """
        m = 10  # 0b01010
        n = 20  # 0b10100
        expected = 0
        self.assertEqual(expected, bitwise_and_of_range(m, n))
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 13, 2020 \[Hard\] Climb Staircase Problem
---
> **Question:** There exists a staircase with N steps, and you can climb up either 1 or 2 steps at a time. Given N, write a function that **PRINT** out all possible unique ways you can climb the staircase. The **ORDER** of the steps matters. 
>
> For example, if N is 4, X = {1, 2} then there are 5 unique ways. This time we print them out as the following:

```py
1, 1, 1, 1
2, 1, 1
1, 2, 1
1, 1, 2
2, 2
```

> What if, instead of being able to climb 1 or 2 steps at a time, you could climb any number from a set of positive integers X? 
>
> For example, if N is 6, and X = {2, 5}. You could climb 2 or 5 steps at a time. Then there is only 1 unique way, so we print the following:

```py
2, 2, 2
```
**My thoughts:** The only way to figure out each path is to manually test all outcomes. However, certain cases are invalid (like exceed the target value while climbing) so we try to modify certain step until its valid. Such technique is called ***Backtracking***.

We may use recursion to implement backtracking, each recursive step will create a separate branch which also represent different recursive call stacks. Once the branch is invalid, the call stack will bring us to a different branch, i.e. backtracking to a different solution space.

For example, if N is 4, and feasible steps are `[1, 2]`, then there are 5 different solution space/path. Each node represents a choice we made and each branch represents a recursive call.

Note we also keep track of the remaining steps while doing recursion. 
```py
├ 1 
│ ├ 1
│ │ ├ 1
│ │ │ └ 1 SUCCEED
│ │ └ 2 SUCCEED
│ └ 2 
│   └ 1 SUCCEED
└ 2 
  ├ 1
  │ └ 1 SUCCEED
  └ 2 SUCCEED
```

If N is 6 and fesible step is `[5, 2]`:
```py
├ 5 FAILURE
└ 2 
  └ 2
    └ 2 SUCCEED
```

**Solution with Backtracking:** [https://repl.it/@trsong/Climb-Staircase-Problem](https://repl.it/@trsong/Climb-Staircase-Problem)
```py
import unittest

def climb_stairs(n, feasible_steps):
    sorted_feasible_steps = sorted(list(set(feasible_steps)))
    res = []
    backtack(res, [], n, sorted_feasible_steps)
    if not res:
        return None
    else:
        return "\n".join(res)


def backtack(res, path, remain, feasible_steps):
    if remain == 0:
        path_str = ", ".join(path)
        res.append(path_str)
    else:
        for step in feasible_steps:
            if step > remain:
                break
            
            path.append(str(step))
            backtack(res, path, remain - step, feasible_steps)
            path.pop()


class ClimbStairSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        expected_lines = expected.splitlines()
        result_lines = result.splitlines()
        self.assertEqual(sorted(expected_lines), sorted(result_lines))
        
    def test_example(self):
        n, feasible_steps = 4, [1, 2]
        expected = "\n".join([
            "1, 1, 1, 1", 
            "2, 1, 1",
            "1, 2, 1",
            "1, 1, 2",
            "2, 2"])
        self.assert_result(expected, climb_stairs(n, feasible_steps))

    def test_example2(self):
        n, feasible_steps = 5, [1, 3, 5]
        expected = "\n".join([
            "1, 1, 1, 1, 1",
            "3, 1, 1",
            "1, 3, 1",
            "1, 1, 3",
            "5"])
        self.assert_result(expected, climb_stairs(n, feasible_steps))

    def test_no_feasible_solution(self):
        n, feasible_steps = 9, []
        self.assertIsNone(climb_stairs(n, feasible_steps))

    def test_no_feasible_solution2(self):
        n, feasible_steps = 99, [7]
        self.assertIsNone(climb_stairs(n, feasible_steps))

    def test_only_one_step_qualify(self):
        n, feasible_steps = 42, [41, 42, 43]
        expected = "42"
        self.assert_result(expected, climb_stairs(n, feasible_steps))

    def test_various_way_to_climb(self):
        n, feasible_steps = 4, [1, 2, 3, 4]
        expected = "\n".join([
            "1, 1, 1, 1",
            "1, 1, 2",
            "1, 2, 1",
            "1, 3",
            "2, 1, 1",
            "2, 2",
            "3, 1",
            "4"])
        self.assert_result(expected, climb_stairs(n, feasible_steps))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 12, 2020 \[Medium\] Look-and-Say Sequence
--- 
> **Question:** The "look and say" sequence is defined as follows: beginning with the term 1, each subsequent term visually describes the digits appearing in the previous term. The first few terms are as follows:

```py
1
11
21
1211
111221
```

> As an example, the fourth term is 1211, since the third term consists of one 2 and one 1.
>
> Given an integer N, print the Nth term of this sequence


**Solution:** [https://repl.it/@trsong/Find-the-n-th-Look-and-Say-Sequence](https://repl.it/@trsong/Find-the-n-th-Look-and-Say-Sequence)
```py
import unittest

def look_and_say(n):
    res = "1"
    for _ in xrange(n-1):
        prev = res[0]
        count = 0
        str_buffer = []
        for i in xrange(len(res)+1):
            char = res[i] if i < len(res) else None
            if prev == char:
                count += 1
            else:
                str_buffer.append(str(count))
                str_buffer.append(prev)
                prev = char
                count = 1
        res = "".join(str_buffer)
    
    return res


class LookAndSaySpec(unittest.TestCase):
    def test_1st_term(self):
        self.assertEqual("1", look_and_say(1))
        
    def test_2nd_term(self):
        self.assertEqual("11", look_and_say(2))
        
    def test_3rd_term(self):
        self.assertEqual("21", look_and_say(3))
        
    def test_4th_term(self):
        self.assertEqual("1211", look_and_say(4))
        
    def test_5th_term(self):
        self.assertEqual("111221", look_and_say(5))
        
    def test_6th_term(self):
        self.assertEqual("312211", look_and_say(6))
        
    def test_7th_term(self):
        self.assertEqual("13112221", look_and_say(7))

    def test_10th_term(self):
        self.assertEqual("13211311123113112211", look_and_say(10))

if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 11, 2020 \[Easy\] Compare Version Numbers
--- 
> **Question:** Version numbers are strings that are used to identify unique states of software products. A version number is in the format a.b.c.d. and so on where a, b, etc. are numeric strings separated by dots. These generally represent a hierarchy from major to minor changes. 
> 
> Given two version numbers version1 and version2, conclude which is the latest version number. Your code should do the following:
> - If version1 > version2 return 1.
> - If version1 < version2 return -1.
> - Otherwise return 0.
>
> Note that the numeric strings such as a, b, c, d, etc. may have leading zeroes, and that the version strings do not start or end with dots. Unspecified level revision numbers default to 0.

**Example 1:**
```py
Input: 
version1 = "1.0.33"
version2 = "1.0.27"
Output: 1 
#version1 > version2
```

**Example 2:**
```py
Input:
version1 = "0.1"
version2 = "1.1"
Output: -1
#version1 < version2
```

**Example 3:**
```py
Input: 
version1 = "1.01"
version2 = "1.001"
Output: 0
#ignore leading zeroes, 01 and 001 represent the same number. 
```

**Example 4:**
```py
Input:
version1 = "1.0"
version2 = "1.0.0"
Output: 0
#version1 does not have a 3rd level revision number, which
defaults to "0"
```

**Solution:** [https://repl.it/@trsong/Version-Number-Comparison](https://repl.it/@trsong/Version-Number-Comparison)
```py
import unittest

def compare(v1, v2):
    version_seq1 = map(int, v1.split('.') if v1 else [0])
    version_seq2 = map(int, v2.split('.') if v2 else [0])

    max_len = max(len(version_seq1), len(version_seq2))

    for _ in xrange(max_len - len(version_seq1)):
        version_seq1.append(0)

    for _ in xrange(max_len - len(version_seq2)):
        version_seq2.append(0)

    for num1, num2 in zip(version_seq1, version_seq2):
        if num1 == num2:
            continue
        elif num1 < num2:
            return -1
        else:
            return 1
    
    return 0


class VersionNumberCompareSpec(unittest.TestCase):
    def test_example1(self):
        version1 = "1.0.33"
        version2 = "1.0.27"
        self.assertEqual(1, compare(version1, version2))

    def test_example2(self):
        version1 = "0.1"
        version2 = "1.1"
        self.assertEqual(-1, compare(version1, version2))

    def test_example3(self):
        version1 = "1.01"
        version2 = "1.001"
        self.assertEqual(0, compare(version1, version2))

    def test_example4(self):
        version1 = "1.0"
        version2 = "1.0.0"
        self.assertEqual(0, compare(version1, version2))

    def test_unspecified_version_numbers(self):
        self.assertEqual(0, compare("", ""))
        self.assertEqual(-1, compare("", "1"))
        self.assertEqual(1, compare("2", ""))

    def test_unaligned_zeros(self):
        version1 = "00000.00000.00000.0"
        version2 = "0.00000.000.00.00000.000.000.0"
        self.assertEqual(0, compare(version1, version2))

    def test_same_version_yet_unaligned(self):
        version1 = "00001.001"
        version2 = "1.000001.0000000.0000"
        self.assertEqual(0, compare(version1, version2))

    def test_different_version_numbers(self):
        version1 = "1.2.3.4"
        version2 = "1.2.3.4.5"
        self.assertEqual(-1, compare(version1, version2))

    def test_different_version_numbers2(self):
        version1 = "3.2.1"
        version2 = "3.1.2.3"
        self.assertEqual(1, compare(version1, version2))

    def test_different_version_numbers3(self):
        version1 = "00001.001.0.1"
        version2 = "1.000001.0000000.0000"
        self.assertEqual(1, compare(version1, version2))

    def test_without_dots(self):
        version1 = "32123"
        version2 = "3144444"
        self.assertEqual(-1, compare(version1, version2))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### May 10, 2020 \[Medium\] All Root to Leaf Paths in Binary Tree
---
> **Question:** Given a binary tree, return all paths from the root to leaves.
>
> For example, given the tree:

```py
   1
  / \
 2   3
    / \
   4   5
```
> Return `[[1, 2], [1, 3, 4], [1, 3, 5]]`.


**Solution with Backtracking:** [https://repl.it/@trsong/Find-All-Root-to-Leaf-Paths-in-Binary-Tree](https://repl.it/@trsong/Find-All-Root-to-Leaf-Paths-in-Binary-Tree)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right= right


def path_to_leaves(tree):
    if not tree:
        return []
    res = []
    backtrack(res, [], tree)
    return res


def backtrack(res, path_to_node, cur_node):
    if cur_node.left is None and cur_node.right is None:
        res.append(path_to_node + [cur_node.val])
    else:
        for child in [cur_node.left, cur_node.right]:
            if child is not None:
                path_to_node.append(cur_node.val)
                backtrack(res, path_to_node, child)
                path_to_node.pop()


class PathToLeavesSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertEqual([], path_to_leaves(None))

    def test_one_level_tree(self):
        self.assertEqual([[1]], path_to_leaves(TreeNode(1)))

    def test_two_level_tree(self):
        tree = TreeNode(1, TreeNode(2), TreeNode(3))
        self.assertEqual([[1, 2], [1, 3]], path_to_leaves(tree))

    def test_example(self):
        """
          1
         / \
        2   3
           / \
          4   5
        """
        n3 = TreeNode(3, TreeNode(4), TreeNode(5))
        tree = TreeNode(1, TreeNode(2), n3)
        self.assertEqual([[1, 2], [1, 3, 4], [1, 3, 5]], path_to_leaves(tree))

    def test_complete_tree(self):
        """
             1
           /   \
          2     3
         / \   / \
        4   5 6   7
        """
        n2 = TreeNode(2, TreeNode(4), TreeNode(5))
        n3 = TreeNode(3, TreeNode(6), TreeNode(7))
        root = TreeNode(1, n2, n3)
        expected = [[1, 2, 4], [1, 2, 5], [1, 3, 6], [1, 3, 7]]
        self.assertEqual(expected, path_to_leaves(root))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 9, 2020  \[Easy\] String Compression
---
> **Question:** Given an array of characters with repeats, compress it in place. The length after compression should be less than or equal to the original array.

**Example:**
```py
Input: ['a', 'a', 'b', 'c', 'c', 'c']
Output: ['a', '2', 'b', 'c', '3']
```

**Solution:** [https://repl.it/@trsong/Compress-String](https://repl.it/@trsong/Compress-String)
```py
import unittest

def string_compression(msg):
    if not msg:
        return

    pos = 0
    count = 0
    prev_char = msg[0]
    n = len(msg)

    for i in xrange(n+1):
        char = msg[i] if i < n else None
        if char != prev_char:
            msg[pos] = prev_char
            if count > 1:
                for count_char in str(count):
                    pos += 1
                    msg[pos] = count_char
            pos += 1
            count = 1
            prev_char = char
        else:
            count += 1
    
    for _ in xrange(pos, n):
        msg.pop()


class StringCompressionSpec(unittest.TestCase):
    def test_example(self):
        msg = ['a', 'a', 'b', 'c', 'c', 'c']
        expected = ['a', '2', 'b', 'c', '3']
        string_compression(msg)
        self.assertEqual(expected, msg)

    def test_empty_msg(self):
        msg = []
        expected = []
        string_compression(msg)
        self.assertEqual(expected, msg)

    def test_msg_with_one_char(self):
        msg = ['a']
        expected = ['a']
        string_compression(msg)
        self.assertEqual(expected, msg)

    def test_msg_with_distinct_chars(self):
        msg = ['a', 'b', 'c', 'd']
        expected = ['a', 'b', 'c', 'd']
        string_compression(msg)
        self.assertEqual(expected, msg)

    def test_msg_with_repeated_chars(self):
        msg = ['a'] * 12
        expected = ['a', '1', '2']
        string_compression(msg)
        self.assertEqual(expected, msg)

    def test_msg_with_repeated_chars2(self):
        msg = ['a', 'b', 'b']
        expected = ['a', 'b', '2']
        string_compression(msg)
        self.assertEqual(expected, msg)

    def test_msg_with_repeated_chars3(self):
        msg = ['a'] * 10 + ['b'] * 21 + ['c'] * 198
        expected = ['a', '1', '0', 'b', '2', '1', 'c', '1', '9', '8']
        string_compression(msg)
        self.assertEqual(expected, msg)

    def test_msg_contains_digits(self):
        msg = ['a', '2', 'a', '3', '3']
        expected = ['a', '2', 'a', '3', '2']
        string_compression(msg)
        self.assertEqual(expected, msg)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 8, 2020  \[Easy\] Find Unique Element among Array of Duplicates
---
> **Question:** Given an array of integers, arr, where all numbers occur twice except one number which occurs once, find the number. Your solution should ideally be `O(n)` time and use constant extra space.

**Example:**
```py
Input: arr = [7, 3, 5, 5, 4, 3, 4, 8, 8]
Output: 7
```

**My thoughts:** XOR has many pretty useful properties:

* 0 ^ x = x
* x ^ x = 0
* x ^ y = y ^ x 

**For example:** 
```py
7 ^ 3 ^ 5 ^ 5 ^ 4 ^ 3 ^ 4 ^ 8 ^ 8
= 7 ^ 3 ^ (5 ^ 5) ^ 4 ^ 3 ^ 4 ^ (8 ^ 8)
= 7 ^ 3 ^ 4 ^ 3 ^ 4 
= 7 ^ 3 ^ 3 ^ 4 ^ 4
= 7 ^ (3 ^ 3) ^ (4 ^ 4)
= 7
```

**Solution with XOR:** [https://repl.it/@trsong/Find-Unique-Element-in-Array](https://repl.it/@trsong/Find-Unique-Element-in-Array)
```py
import unittest

def find_unique_element(nums):
    return reduce(lambda num, accu: num ^ accu, nums)

class FindUniqueElementSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(7, find_unique_element([7, 3, 5, 5, 4, 3, 4, 8, 8]))

    def test_array_with_one_element(self):
        self.assertEqual(42, find_unique_element([42]))

    def test_same_duplicated_number_not_consecutive(self):
        self.assertEqual(5, find_unique_element([1, 2, 1, 5, 3, 2, 3]))

    def test_array_with_negative_elements(self):
        self.assertEqual(-1, find_unique_element([-1, 1, 0, 0, 1]))

    def test_array_with_negative_elements2(self):
        self.assertEqual(0, find_unique_element([-1, 0, 1, -2, 2, -1, 1, -2, 2]))
    
if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 7, 2020  \[Easy\] Full Binary Tree
--- 
> **Question:** Given a binary tree, remove the nodes in which there is only 1 child, so that the binary tree is a full binary tree.
>
> So leaf nodes with no children should be kept, and nodes with 2 children should be kept as well.

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

**Solution:** [https://repl.it/@trsong/Full-Binary-Tree-Prune](https://repl.it/@trsong/Full-Binary-Tree-Prune)
```py
import unittest
import copy

def full_tree_prune(root):
    if not root:
        return None
    
    left_pruned_tree = full_tree_prune(root.left)
    right_pruned_tree = full_tree_prune(root.right)

    if left_pruned_tree and right_pruned_tree:
        root.left = left_pruned_tree
        root.right = right_pruned_tree
        return root
    elif left_pruned_tree:
        return left_pruned_tree
    elif right_pruned_tree:
        return right_pruned_tree
    else:
        return root


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __eq__(self, other):
        return other and other.val == self.val and other.left == self.left and other.right == self.right

class RemovePartialNodeSpec(unittest.TestCase):
    def test_example(self):
        """
             1
            / \ 
          *2   3
          /   / \
         0   9   4
        """
        n3 = TreeNode(3, TreeNode(9), TreeNode(4))
        n2 = TreeNode(2, TreeNode(0))
        original_tree = TreeNode(1, n2, n3)
        
        """
             1
            / \ 
           0   3
              / \
             9   4
        """
        t3 = TreeNode(3, TreeNode(9), TreeNode(4))
        expected_tree = TreeNode(1, TreeNode(0), t3)
        self.assertEqual(expected_tree, full_tree_prune(original_tree))

    def test_empty_tree(self):        
        self.assertIsNone(None, full_tree_prune(None))

    def test_both_parent_and_child_node_is_not_full(self):
        """
             2
           /   \
         *7    *5
           \     \
            6    *9
           / \   /
          1  11 4
        """
        n7 = TreeNode(7, right=TreeNode(6, TreeNode(1), TreeNode(11)))
        n5 = TreeNode(5, right=TreeNode(9, TreeNode(4)))
        original_tree = TreeNode(2, n7, n5)

        """
            2
           / \
          6   4
         / \
        1  11 
        """
        t6 = TreeNode(6, TreeNode(1), TreeNode(11))
        expected_tree = TreeNode(2, t6, TreeNode(4))
        self.assertEqual(expected_tree, full_tree_prune(original_tree))

    def test_root_is_partial(self):
        """
           *1
           /
         *2
         /
        3
        """
        original_tree = TreeNode(1, TreeNode(2, TreeNode(3)))
        expected_tree = TreeNode(3)
        self.assertEqual(expected_tree, full_tree_prune(original_tree))

    def test_root_is_partial2(self):
        """
           *1
             \
             *2
             /
            3
        """
        original_tree = TreeNode(1, right=TreeNode(2, TreeNode(3)))
        expected_tree = TreeNode(3)
        self.assertEqual(expected_tree, full_tree_prune(original_tree))

    def test_tree_is_full(self):
        """
              1
            /   \
           4     5
          / \   / \
         2   3 6   7 
        / \       / \
       8   9    10  11
        """
        n2 = TreeNode(2, TreeNode(8), TreeNode(9))
        n7 = TreeNode(7, TreeNode(10), TreeNode(11))
        n4 = TreeNode(4, n2, TreeNode(3))
        n5 = TreeNode(5, TreeNode(6), n7)
        original_tree = TreeNode(1, n4, n5)
        expected_tree = copy.deepcopy(original_tree)
        self.assertEqual(expected_tree, full_tree_prune(original_tree))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 6, 2020 LC 301 \[Hard\] Remove Invalid Parentheses
--- 
> **Question:** Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.
>
> **Note:** The input string may contain letters other than the parentheses ( and ).

**Example 1:**
```py
Input: "()())()"
Output: ["()()()", "(())()"]
```

**Example 2:**
```py
Input: "(a)())()"
Output: ["(a)()()", "(a())()"]
```

**Example 3:**
```py
Input: ")("
Output: [""]
```

**My thoughts:** What makes a string with parenthese invalid? There must be an index such that number of open parentheses is less than close parentheses or in the end all open and close parentheses are not equal. Now we can count how many parentheses are invalid so that we can remove those invalid ones during backtracking.

We can define:
Number of invalid open is equal to total open - total close.
Number of invalid close is equal to number of close exceed previous open. 

During backtracking, each open and close could be invalid one, so give a try to remove those and that will decrese the invalid count and we can hope all the best that our solution works. If it works, ie. the final string is valid, then add to result, else backtrack.


How to avoid duplicates? For each candidate of invalid ones, we only remove the first one and skip the duplicates.
eg. `((()((())`
We have `6` open and `3` close gives `6 - 3 = 3` invalid open and we have no invalid close.

```py
((()((())
^^^
Removing any two of above gives same result

To avoid duplicate, we only remove first two:
((()((())
^^
```


**Solution with Backtracking:** [https://repl.it/@trsong/LC301-Remove-Invalid-Parentheses](https://repl.it/@trsong/LC301-Remove-Invalid-Parentheses)
```py
import unittest

def remove_invalid_parenthese(s):
    invalid_open = invalid_close = 0
    for c in s:
        if c == ')' and invalid_open == 0:
            invalid_close += 1
        elif c == '(':
            invalid_open += 1
        elif c == ')':
            invalid_open -= 1
    
    res = []
    backtrack(s, res, 0, invalid_open, invalid_close)
    return filter(is_valid, res)


def backtrack(s, res, next_index, invalid_open, invalid_close):
    if invalid_open == 0 and invalid_close == 0:
        res.append(s)
    else:
        for i in xrange(next_index, len(s)):
            c = s[i]
            if c == '(' and invalid_open > 0 or c == ')' and invalid_close > 0:
                if i > next_index and s[i] == s[i-1]:
                    # skip same character
                    continue
                
                updated_s = s[:i] + s[i+1:]
                if c == '(':
                    backtrack(updated_s, res, i, invalid_open-1, invalid_close)
                else:
                    backtrack(updated_s, res, i, invalid_open, invalid_close-1)


def is_valid(s):
    count = 0
    for c in s:
        if count < 0:
            return False
        elif c == '(':
            count += 1
        elif c == ')':
            count -= 1
    return count == 0


class RemoveInvalidParentheseSpec(unittest.TestCase):
    def assert_result(self, expected, output):
        self.assertEqual(sorted(expected), sorted(output))

    def test_example1(self):
        input = "()())()"
        expected = ["()()()", "(())()"]
        self.assert_result(expected, remove_invalid_parenthese(input))

    def test_example2(self):
        input = "(a)())()"
        expected = ["(a)()()", "(a())()"]
        self.assert_result(expected, remove_invalid_parenthese(input))

    def test_example3(self):
        input = ")("
        expected = [""]
        self.assert_result(expected, remove_invalid_parenthese(input))

    def test_valid_string1(self):
        input = "(a)((b))(c)"
        expected = ["(a)((b))(c)"]
        self.assert_result(expected, remove_invalid_parenthese(input))

    def test_empty_string(self):
        input = ""
        expected = [""]
        self.assert_result(expected, remove_invalid_parenthese(input))

    def test_unique_result1(self):
        input = "(a)(((a)"
        expected = ["(a)(a)"]
        self.assert_result(expected, remove_invalid_parenthese(input))

    def test_unique_result2(self):
        input = "()))((()"
        expected = ["()()"]
        self.assert_result(expected, remove_invalid_parenthese(input))
    
    def test_unique_result3(self):
        input = "a))b))c)d"
        expected = ["abcd"]
        self.assert_result(expected, remove_invalid_parenthese(input))

    def test_multiple_results(self):
        input = "a(b(c(d)"
        expected = ["a(bcd)", "ab(cd)", "abc(d)"]
        self.assert_result(expected, remove_invalid_parenthese(input))

    def test_multiple_results2(self):
        input = "(a)b)c)d)"
        expected = ["(a)bcd", "(ab)cd", "(abc)d", "(abcd)"]
        self.assert_result(expected, remove_invalid_parenthese(input))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 5, 2020 \[Hard\] Maximum Spanning Tree
--- 
> **Question:** Recall that the minimum spanning tree is the subset of edges of a tree that connect all its vertices with the smallest possible total edge weight. 
> 
> Given an undirected graph with weighted edges, compute the maximum weight spanning tree.


**My thoughts:** Both Kruskal's and Prim's Algorithm works for this question. The idea is to flip all edge weight into negative and then apply either of previous algorithms until find a spanning tree. 

**Solution with Kruskal's Algorithm:** [https://repl.it/@trsong/Find-Maximum-Spanning-Tree](https://repl.it/@trsong/Find-Maximum-Spanning-Tree)
```py
import unittest
import sys

class DisjointSet(object):
    def __init__(self, size):
        self.parent = [-1] * size

    def find(self, p):
        while self.parent[p] != -1:
            p = self.parent[p]
        return p
    
    def union(self, p1, p2):
        parent1 = self.find(p1)
        parent2 = self.find(p2)
        if parent1 != parent2:
            self.parent[parent1] = parent2

    def is_connected(self, p1, p2):
        return self.find(p1) == self.find(p2)


def max_spanning_tree(vertices, edges):
    uf = DisjointSet(vertices)
    edges.sort(key=lambda uvw: uvw[-1], reverse=True)  # sort based on weight
    res = []

    for u, v, _ in edges:
        if not uf.is_connected(u, v):
            res.append((u, v))
            uf.union(u, v)

    return res


class MaxSpanningTreeSpec(unittest.TestCase):
    def assert_spanning_tree(self, vertices, edges, spanning_tree_edges, expected_weight):
        # check if it's a tree
        self.assertEqual(vertices-1, len(spanning_tree_edges))

        neighbor = [[sys.maxint for _ in xrange(vertices)] for _ in xrange(vertices)]
        for u, v, w in edges:
            neighbor[u][v] = w
            neighbor[v][u] = w

        visited = [0] * vertices
        total_weight = 0
        for u, v in spanning_tree_edges:
            total_weight += neighbor[u][v]
            visited[u] = 1
            visited[v] = 1
        
        # check if all vertices are visited
        self.assertEqual(vertices, sum(visited))
        
        # check if sum of all weight satisfied
        self.assertEqual(expected_weight, total_weight)

    
    def test_empty_graph(self):
        self.assertEqual([], max_spanning_tree(0, []))

    def test_simple_graph1(self):
        """
        Input:
            1
          / | \
         0  |  2
          \ | /
            3
        
        Output:
            1
            |  
         0  |  2
          \ | /
            3
        """
        vertices = 4
        edges = [(0, 1, 1), (1, 2, 2), (2, 3, 3), (3, 0, 4), (1, 3, 5)]
        expected_weight = 3 + 4 + 5
        res = max_spanning_tree(vertices, edges)
        self.assert_spanning_tree(vertices, edges, res, expected_weight)

    def test_simple_graph2(self):
        """
        Input:
            1
          / | \
         0  |  2
          \ | /
            3
        
        Output:
            1
            | \
         0  |  2
          \ |  
            3
        """
        vertices = 4
        edges = [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 1), (1, 3, 1)]
        expected_weight = 1 + 1 + 1
        res = max_spanning_tree(vertices, edges)
        self.assert_spanning_tree(vertices, edges, res, expected_weight)

    def test_k3_graph(self):
        """
        Input:
           1
          / \
         0---2

        Output:
           1
            \
         0---2         
        """
        vertices = 3
        edges = [(0, 1, 1), (1, 2, 1), (0, 2, 2)]
        expected_weight = 1 + 2
        res = max_spanning_tree(vertices, edges)
        self.assert_spanning_tree(vertices, edges, res, expected_weight)

    def test_k4_graph(self):
        """
        Input:
        0 - 1
        | x |
        3 - 2

        Output:
        0   1
        | / 
        3 - 2
        """
        vertices = 4
        edges = [(0, 1, 10), (0, 2, 11), (0, 3, 12), (1, 2, 13), (1, 3, 14), (2, 3, 15)]
        expected_weight = 12 + 14 + 15
        res = max_spanning_tree(vertices, edges)
        self.assert_spanning_tree(vertices, edges, res, expected_weight)

    def test_complicated_graph(self):
        """
        Input:
        0 - 1 - 2
        | / | / |
        3 - 4 - 5 

        Output:
        0   1 - 2
        | /   / 
        3   4 - 5 
        """
        vertices = 6
        edges = [(0, 1, 1), (1, 2, 6), (0, 5, 3), (5, 1, 5), (4, 1, 1), (4, 2, 5), (3, 2, 2), (5, 4, 1), (4, 3, 4)]
        expected_weight = 3 + 5 + 6 + 5 + 4
        res = max_spanning_tree(vertices, edges)
        self.assert_spanning_tree(vertices, edges, res, expected_weight)

    def test_graph_with_all_negative_weight(self):
        # Note only all negative weight can use greedy approach. Mixed postive and negative graph is NP-Hard
        """
        Input:
        0 - 1 - 2
        | / | / |
        3 - 4 - 5 

        Output:
        0 - 1   2
            |   |
        3 - 4 - 5 
        """
        vertices = 6
        edges = [(0, 1, -1), (1, 2, -6), (0, 5, -3), (5, 1, -5), (4, 1, -1), (4, 2, -5), (3, 2, -2), (5, 4, -1), (4, 3, -4)]
        expected_weight = -1 -1 -1 -4 -2
        res = max_spanning_tree(vertices, edges)
        self.assert_spanning_tree(vertices, edges, res, expected_weight)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 4, 2020 LC 138 \[Medium\] Deepcopy List with Random Pointer
--- 
> **Question:** A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.
>
> Return a deep copy of the list.

 **Example:**
```py
Input:
{"$id":"1","next":{"$id":"2","next":null,"random":{"$ref":"2"},"val":2},"random":{"$ref":"2"},"val":1}

Explanation:
Node 1's value is 1, both of its next and random pointer points to Node 2.
Node 2's value is 2, its next pointer points to null and its random pointer points to itself.
```

**Solution:** [https://repl.it/@trsong/Deepcopy-List-with-Random-Pointer](https://repl.it/@trsong/Deepcopy-List-with-Random-Pointer)
```py
import unittest
import copy

def deep_copy(head):
    if not head:
        return None

    # Step1: Duplicate node to every other position
    p = head
    while p:
        p.next = ListNode(p.val, p.next)
        p = p.next.next

    # Step2: Copy random attribute
    p = head
    while p:
        copy = p.next
        if p.random:
            copy.random = p.random.next
        p = p.next.next

    # Step3: Partition even and odd nodes to restore original list and get result
    copy_head = head.next
    p = head
    while p:
        copy = p.next
        if copy.next:
            p.next = copy.next
            copy.next = copy.next.next
        else:
            p.next = None
            copy.next = None
        p = p.next
        
    return copy_head


##################### 
# Testing Utiltities
#####################
class ListNode(object):
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

    def __eq__(self, other):
        if not other:
            return False
        is_random_none = self.random is None and other.random is None
        is_random_equal = self.random and other.random and self.random.val == other.random.val
        is_random_valid = is_random_none or is_random_equal
        return is_random_valid and self.val == other.val and self.next == other.next
        
    def __repr__(self):
        lst = []
        random = []
        p = self
        while p:
            lst.append(str(p.val))
            if p.random:
                random.append(str(p.random.val))
            else:
                random.append("N")
            p = p.next
        
        return "\nList: [{}].\nRandom: [{}]".format(','.join(lst), ','.join(random))
            

class DeepCopySpec(unittest.TestCase):
    def test_empty_list(self):
        self.assertIsNone(deep_copy(None))
    
    def test_list_with_random_point_to_itself(self):
        n = ListNode(1)
        n.random = n
        expected = copy.deepcopy(n)
        self.assertEqual(expected, deep_copy(n))
        self.assertEqual(expected, n)

    def test_random_pointer_is_None(self):
        # 1 -> 2 -> 3 -> 4
        n4 = ListNode(4)
        n3 = ListNode(3, n4)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)
        expected = copy.deepcopy(n1)
        self.assertEqual(expected, deep_copy(n1))
        self.assertEqual(expected, n1)

    def test_list_with_forward_random_pointers(self):
        # 1 -> 2 -> 3 -> 4
        n4 = ListNode(4)
        n3 = ListNode(3, n4)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)

        # random pointer:
        # 1 -> 3
        # 2 -> 3
        # 3 -> 4
        n1.random = n3
        n2.random = n3
        n3.random = n4
        expected = copy.deepcopy(n1)
        self.assertEqual(expected, deep_copy(n1))
        self.assertEqual(expected, n1)

    def test_list_with_backward_random_pointers(self):
        # 1 -> 2 -> 3 -> 4
        n4 = ListNode(4)
        n3 = ListNode(3, n4)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)

        # random pointer:
        # 1 -> 1
        # 2 -> 1
        # 3 -> 2
        # 4 -> 1
        n1.random = n1
        n2.random = n1
        n3.random = n2
        n4.random = n1
        expected = copy.deepcopy(n1)
        self.assertEqual(expected, deep_copy(n1))
        self.assertEqual(expected, n1)

    def test_list_with_both_forward_and_backward_pointers(self):
        # 1 -> 2 -> 3 -> 4
        n4 = ListNode(4)
        n3 = ListNode(3, n4)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)

        # random pointer:
        # 1 -> 3
        # 2 -> 1
        # 3 -> 4
        # 4 -> 3
        n1.random = n3
        n2.random = n2
        n3.random = n4
        n4.random = n3
        expected = copy.deepcopy(n1)
        self.assertEqual(expected, deep_copy(n1))
        self.assertEqual(expected, n1)

    def test_list_with_both_forward_and_backward_pointers2(self):
        # 1 -> 2 -> 3 -> 4 -> 5 -> 6
        n6 = ListNode(6)
        n5 = ListNode(5, n6)
        n4 = ListNode(4, n5)
        n3 = ListNode(3, n4)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)

        # random pointer:
        # 1 -> 1
        # 2 -> 1
        # 3 -> 4
        # 4 -> 5
        # 5 -> 3
        # 6 -> None
        n1.random = n1
        n2.random = n1
        n3.random = n4
        n4.random = n5
        n5.random = n3
        expected = copy.deepcopy(n1)
        self.assertEqual(expected, deep_copy(n1))
        self.assertEqual(expected, n1)
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 3, 2020 \[Hard\] Maximize Sum of the Minimum of K Subarrays
--- 
> **Question:** Given an array a of size N and an integer K, the task is to divide the array into K segments such that sum of the minimum of K segments is maximized.

**Example1:**
```py
Input: [5, 7, 4, 2, 8, 1, 6], K = 3
Output: 13
Divide the array at indexes 0 and 1. Then the segments are [5], [7], [4, 2, 8, 1, 6] Sum of the minimus is 5 + 7 + 1 = 13
```

**Example2:**
```py
Input: [6, 5, 3, 8, 9, 10, 4, 7, 10], K = 4
Output: 27
[6, 5, 3, 8, 9], [10], [4, 7], [10] => 3 + 10 + 4 + 10 = 27
```

**My thoughts:** Think about the problem in a recursive manner. Suppose we know the solution `dp[n][k]` that is the solution (max sum of all min) for all subarray number, ie. `num_subarray = 1, 2, ..., k`. Then with the introduction of the `n+1` element, what can we say about the solution? ie. `dp[n+1][k+1]`.

What we can do is to create the k+1 th subarray, and try to absorb all previous elements one by one and find the maximum.

Example for introduction of new element and the process of absorbing previous elements:
- `[1, 2, 3, 1, 2], [6]` => `f([1, 2, 3, 1, 2]) + min(6)`
- `[1, 2, 3, 1], [2, 6]` => `f([1, 2, 3, 1]) + min(6, 2)`
- `[1, 2], [1, 2, 6]`    => `f([1, 2]) + min(6, 2, 1)`
- `[1], [2, 1, 2, 6]`    => `f([1]) + min(6, 2, 1, 2)`

Of course, the min value of last array will change, but we can calculate that along the way when we absorb more elements, and we can use `dp[n-p][k] for all p <= n` to calculate the answer. Thus `dp[n][k] = max{dp[n-p][k-1] + min_value of last_subarray} for all p < n, ie. num[p] is in last subarray`.


**Solution with DP:** [https://repl.it/@trsong/Find-Maximize-Sum-of-the-Minimum-of-K-Subarrays](https://repl.it/@trsong/Find-Maximize-Sum-of-the-Minimum-of-K-Subarrays)
```py
import unittest

def max_aggregate_subarray_min(nums, k):
    n = len(nums)
    # dp[n][k] represents max of k min segments 
    # dp[n][k] = max(dp[n-p][k-1] + min(nums[n-p:n])) for p < n
    dp = [[float('-inf') for _ in xrange(k+1)] for _ in xrange(n+1)]
    dp[0][0] = 0

    for j in xrange(1, k+1):
        for i in xrange(j, n+1):
            last_seg_min = nums[i-1]
            for p in xrange(n):
                last_seg_min = min(last_seg_min, nums[i-1-p])
                dp[i][j] = max(dp[i][j], dp[i-1-p][j-1] + last_seg_min)

    return dp[n][k]


class MaxAggregateSubarrayMinSpec(unittest.TestCase):
    def test_example1(self):
        nums = [5, 7, 4, 2, 8, 1, 6]
        k = 3
        expected = 13  #  [5], [7], [4, 2, 8, 1, 6] => 5 + 7 + 1 = 13
        self.assertEqual(expected, max_aggregate_subarray_min(nums, k))

    def test_example2(self):
        nums =  [6, 5, 3, 8, 9, 10, 4, 7, 10]
        k = 4
        expected = 27  # [6, 5, 3, 8, 9], [10], [4, 7], [10] => 3 + 10 + 4 + 10 = 27
        self.assertEqual(expected, max_aggregate_subarray_min(nums, k))

    def test_empty_array(self):
        self.assertEqual(0, max_aggregate_subarray_min([], 0))

    def test_not_split_array(self):
        self.assertEqual(1, max_aggregate_subarray_min([1, 2, 3], 1))

    def test_not_allow_split_into_empty_subarray(self):
        self.assertEqual(-1, max_aggregate_subarray_min([5, -3, 0, 3, -6], 5))

    def test_local_max_vs_global_max(self):
        nums =  [1, 2, 3, 1, 2, 3, 1, 2, 3]
        k = 3
        expected = 6  # [1, 2, 3, 1, 2, 3, 1], [2], [3] => 1 + 2 + 3 = 6
        self.assertEqual(expected, max_aggregate_subarray_min(nums, k))

    def test_local_max_vs_global_max2(self):
        nums =  [3, 2, 1, 3, 2, 1, 3, 2, 1]
        k = 4
        expected = 8  # [3], [2, 1], [3], [2, 1, 3, 2, 1] => 3 + 1 + 3 + 1 = 8
        self.assertEqual(expected, max_aggregate_subarray_min(nums, k))

    def test_array_contains_negative_elements(self):
        nums =  [6, 3, -2, -4, 2, -1, 3, 2, 1, -5, 3, 5]
        k = 3
        expected = 6  # [6], [3, -2, -4, 2, -1, 3, 2, 1, -5, 3], [5] => 6 - 5 + 5 = 6
        self.assertEqual(expected, max_aggregate_subarray_min(nums, k))

    def test_array_contains_negative_elements2(self):
        nums =  [1, -2, 3, -3, 0]
        k = 3
        expected = -2  # [1, -2], [3], [-3, 0] => -2 + 3 - 3 = -2
        self.assertEqual(expected, max_aggregate_subarray_min(nums, k))

    def test_array_with_all_negative_numbers(self):
        nums =  [-1, -2, -3, -1, -2, -3]
        k = 2
        expected = -4  # [-1], [-2, -3, -1, -2, -3] => - 1 - 3 = -4
        self.assertEqual(expected, max_aggregate_subarray_min(nums, k))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### May 2, 2020  \[Easy\] Spreadsheet Columns
--- 
> **Question:** In many spreadsheet applications, the columns are marked with letters. From the 1st to the 26th column the letters are A to Z. Then starting from the 27th column it uses AA, AB, ..., ZZ, AAA, etc.
>
> Given a number n, find the n-th column name.


**Examples:**
```py
Input          Output
 26             Z
 51             AY
 52             AZ
 80             CB
 676            YZ
 702            ZZ
 705            AAC
```

**Solution:** [https://repl.it/@trsong/CandidFriendlyCalculators](https://repl.it/@trsong/CandidFriendlyCalculators)
```py
import unittest

def digit_to_letter(num):
    ord_A = ord('A')
    return chr(ord_A + num)


def spreadsheet_columns(n):
    base = 26
    q = n - 1 # convert to zero-based
    res = []
    
    while q >= 0:
        q, r = divmod(q, base)
        res.append(digit_to_letter(r))
        q -= 1  # A, B, C....Z, AA, AB, ... AZ, BA

    res.reverse()
    return "".join(res)


class SpreadsheetColumnSpec(unittest.TestCase):
    def test_trivial_example(self):
        self.assertEqual("A", spreadsheet_columns(1))
    
    def test_example1(self):
        self.assertEqual("Z", spreadsheet_columns(26))
    
    def test_example2(self):
        self.assertEqual("AY", spreadsheet_columns(51))
    
    def test_example3(self):
        self.assertEqual("AZ", spreadsheet_columns(52))
    
    def test_example4(self):
        self.assertEqual("CB", spreadsheet_columns(80))
    
    def test_example5(self):
        self.assertEqual("YZ", spreadsheet_columns(676))
    
    def test_example6(self):
        self.assertEqual("ZZ", spreadsheet_columns(702))
    
    def test_example7(self):
        self.assertEqual("AAC", spreadsheet_columns(705))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### May 1, 2020 \[Easy\] ZigZag Binary Tree
---
> **Questions:** In Ancient Greece, it was common to write text with the first line going left to right, the second line going right to left, and continuing to go back and forth. This style was called "boustrophedon".
>
> Given a binary tree, write an algorithm to print the nodes in boustrophedon order.

**Example:**
```py
Given the following tree:
       1
    /     \
  2         3
 / \       / \
4   5     6   7
You should return [1, 3, 2, 4, 5, 6, 7].
```

**Solution with BFS:** [https://repl.it/@trsong/Print-ZigZag-Traversal](https://repl.it/@trsong/Print-ZigZag-Traversal)
```py
import unittest

def zigzag_traversal(tree):
    if not tree:
        return []

    res = []
    queue = [tree]
    reverse_order = False
    while queue:
        if reverse_order:
            res.extend(queue[::-1])
        else:
            res.extend(queue)
        reverse_order = not reverse_order

        for _ in xrange(len(queue)):
            cur = queue.pop(0)
            if cur.left:
                queue.append(cur.left)

            if cur.right:
                queue.append(cur.right)

    return map(lambda cur: cur.val, res)


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ZigzagTraversalSpec(unittest.TestCase):
    def test_example(self):
        """
               1
            /     \
          2         3
         / \       / \
        4   5     6   7
        """
        left = TreeNode(2, TreeNode(4), TreeNode(5))
        right = TreeNode(3, TreeNode(6), TreeNode(7))
        root = TreeNode(1, left, right)
        expected_traversal = [1, 3, 2, 4, 5, 6, 7]
        self.assertEqual(expected_traversal, zigzag_traversal(root))

    def test_empty(self):
        self.assertEqual([], zigzag_traversal(None))

    def test_right_heavy_tree(self):
        """
            3
           / \
          9  20
            /  \
           15   7
        """
        n20 = TreeNode(20, TreeNode(15), TreeNode(7))
        n3 = TreeNode(3, TreeNode(9), n20)
        expected_traversal = [3, 20, 9, 15, 7]
        self.assertEqual(expected_traversal, zigzag_traversal(n3))
    
    def test_complete_tree(self):
        """
             1
           /   \
          3     2
         / \   /  
        4   5 6  
        """
        n3 = TreeNode(3, TreeNode(4), TreeNode(5))
        n2 = TreeNode(2, TreeNode(6))
        n1 = TreeNode(1, n3, n2)
        expected_traversal = [1, 2, 3, 4, 5, 6]
        self.assertEqual(expected_traversal, zigzag_traversal(n1))

    def test_sparse_tree(self):
        """
             1
           /   \
          3     2
           \   /  
            4 5
           /   \
          7     6
           \   /  
            8 9
        """
        n3 = TreeNode(3, right=TreeNode(4, TreeNode(7, right=TreeNode(8))))
        n2 = TreeNode(2, TreeNode(5, right=TreeNode(6, TreeNode(9))))
        n1 = TreeNode(1, n3, n2)
        expected_traversal = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(expected_traversal, zigzag_traversal(n1))


if __name__ == '__main__':
    unittest.main(exit=False)
```
