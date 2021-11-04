---
layout: post
title:  "Daily Coding Problems 2021 Nov to Jan"
date:   2021-11-01 22:22:32 -0700
categories: Python/Java
---
* This will become a table of contents (this text will be scraped).
{:toc}

## Daily Coding Problems

**Past Problems by Category:** [https://trsong.github.io/python/java/2019/04/02/DailyQuestionsByCategory.html](https://trsong.github.io/python/java/2019/04/02/DailyQuestionsByCategory.html)

### Enviroment Setup
---

**Python 2.7 Playground:** [https://repl.it/languages/python](https://repl.it/languages/python)

**Python 3 Playground:** [https://repl.it/languages/python3](https://repl.it/languages/python3) 

**Java Playground:** [https://repl.it/languages/java](https://repl.it/languages/java)


### Nov 3, 2021 \[Medium\] K-th Missing Number in Sorted Array
---
> **Question:** Given a sorted without any duplicate integer array, define the missing numbers to be the gap among numbers. Write a function to calculate K-th missing number. If such number does not exist, then return null.
> 
> For example, original array: `[2,4,7,8,9,15]`, within the range defined by original array, all missing numbers are: `[3,5,6,10,11,12,13,14]`
> - the 1st missing number is 3,
> - the 2nd missing number is 5,
> - the 3rd missing number is 6

**Solution with Binary Search:** [https://replit.com/@trsong/Find-K-th-Missing-Number-in-Sorted-Array-2](https://replit.com/@trsong/Find-K-th-Missing-Number-in-Sorted-Array-2)
```py
import unittest

def find_kth_missing_number(nums, k):
    if not nums or k <= 0 or count_left_missing(nums, len(nums) - 1) < k:
        return None

    lo = 0
    hi = len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if count_left_missing(nums, mid) < k:
            lo = mid + 1
        else:
            hi = mid
    
    offset = count_left_missing(nums, lo) - k + 1
    return nums[lo] - offset


def count_left_missing(nums, index):
    return nums[index] - nums[0] - index


class FindKthMissingNumberSpec(unittest.TestCase):
    def test_empty_source(self):
        self.assertIsNone(find_kth_missing_number([], 0))
    
    def test_empty_source2(self):
        self.assertIsNone(find_kth_missing_number([], 1))

    def test_missing_number_not_exists(self):
        self.assertIsNone(find_kth_missing_number([1, 2, 3], 0))
    
    def test_missing_number_not_exists2(self):
        self.assertIsNone(find_kth_missing_number([1, 2, 3], 1))

    def test_missing_number_not_exists3(self):
        self.assertIsNone(find_kth_missing_number([1, 3], 2))

    def test_one_gap_in_source(self):
        self.assertEqual(5, find_kth_missing_number([3, 4, 8, 9, 10, 11, 12], 1))
    
    def test_one_gap_in_source2(self):
        self.assertEqual(6, find_kth_missing_number([3, 4, 8, 9, 10, 11, 12], 2))

    def test_one_gap_in_source3(self):
        self.assertEqual(7, find_kth_missing_number([3, 4, 8, 9, 10, 11, 12], 3))

    def test_one_gap_in_source4(self):
        self.assertEqual(4, find_kth_missing_number([3, 6, 7], 1))

    def test_one_gap_in_source5(self):
        self.assertEqual(5, find_kth_missing_number([3, 6, 7], 2))
    
    def test_multiple_gap_in_source(self):
        self.assertEqual(3, find_kth_missing_number([2, 4, 7, 8, 9, 15], 1))
    
    def test_multiple_gap_in_source2(self):
        self.assertEqual(5, find_kth_missing_number([2, 4, 7, 8, 9, 15], 2))
    
    def test_multiple_gap_in_source3(self):
        self.assertEqual(6, find_kth_missing_number([2, 4, 7, 8, 9, 15], 3))

    def test_multiple_gap_in_source4(self):
        self.assertEqual(10, find_kth_missing_number([2, 4, 7, 8, 9, 15], 4))

    def test_multiple_gap_in_source5(self):
        self.assertEqual(11, find_kth_missing_number([2, 4, 7, 8, 9, 15], 5))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Nov 2, 2021 LC 525 \[Medium\] Largest Subarray with Equal Number of 0s and 1s
---
> **Question:** Given an array containing only 0s and 1s, find the largest subarray which contain equal number of 0s and 1s. Expected time complexity is O(n).


**Example 1:**
```py
Input: arr[] = [1, 0, 1, 1, 1, 0, 0]
Output: 1 to 6 (Starting and Ending indexes of output subarray)
```

**Example 2:**
```py
Input: arr[] = [1, 1, 1, 1]
Output: No such subarray
```

**Example 3:**
```py
Input: arr[] = [0, 0, 1, 1, 0]
Output: 0 to 3 Or 1 to 4
```

**Solution:** [https://replit.com/@trsong/Find-Largest-Subarray-with-Equal-Number-of-0s-and-1s](https://replit.com/@trsong/Find-Largest-Subarray-with-Equal-Number-of-0s-and-1s)
```py
import unittest

def largest_even_subarray(nums):
    balance = 0
    balance_occurance = {0: -1}
    max_window_size = -1
    max_window_start = -1
    
    for index, num in enumerate(nums):
        balance += 1 if num == 1 else -1

        if balance not in balance_occurance:
            balance_occurance[balance] = index
        window_size = index - balance_occurance[balance]

        if window_size > max_window_size:
            max_window_size = window_size
            max_window_start = balance_occurance[balance] + 1
    return (max_window_start, max_window_start + max_window_size - 1) if max_window_size > 0 else None


class LargestEvenSubarraySpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual((1, 6), largest_even_subarray([1, 0, 1, 1, 1, 0, 0]))

    def test_example2(self):
        self.assertTrue(largest_even_subarray([0, 0, 1, 1, 0]) in [(0, 3), (1, 4)])

    def test_entire_array_is_even(self):
        self.assertEqual((0, 1), largest_even_subarray([0, 1]))

    def test_no_even_subarray(self):
        self.assertIsNone(largest_even_subarray([0, 0, 0, 0, 0]))

    def test_no_even_subarray2(self):
        self.assertIsNone(largest_even_subarray([1]))

    def test_no_even_subarray3(self):
        self.assertIsNone(largest_even_subarray([]))

    def test_larger_array(self):
        self.assertEqual((0, 9), largest_even_subarray([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1]))

    def test_larger_array2(self):
        self.assertEqual((3, 8), largest_even_subarray([1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1]))

    def test_larger_array3(self):
        self.assertEqual((0, 13), largest_even_subarray([1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1])) 


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Nov 1, 2021 \[Medium\] Smallest Number of Perfect Squares
---
> **Question:** Write a program that determines the smallest number of perfect squares that sum up to N.
>
> Here are a few examples:
```py
Given N = 4, return 1 (4)
Given N = 17, return 2 (16 + 1)
Given N = 18, return 2 (9 + 9)
```

**Solution with DP:** [https://replit.com/@trsong/Minimum-Squares-Sum-to-N-2](https://replit.com/@trsong/Minimum-Squares-Sum-to-N-2)
```py
import unittest

def min_square_sum(n):
    # Let dp[n] represents min # of squares sum to n
    # dp[n] = 1 + min{ dp[n - i * i] }  for all i st. i * i <= n
    dp = [float('inf')] * (n + 1)
    dp[0] = 0

    for num in range(1, n + 1):
        for i in range(1, num):
            if i * i > num:
                continue
            dp[num] = min(dp[num], 1 + dp[num - i * i])
    return dp[n]


class MinSquareSumSpec(unittest.TestCase):
    def test_example(self):
        # 13 = 3^2 + 2^2
        self.assertEqual(2, min_square_sum(13))

    def test_example2(self):
        # 27 = 3^2 + 3^2 + 3^2
        self.assertEqual(3, min_square_sum(27))

    def test_perfect_square(self):
        # 100 = 10^2
        self.assertEqual(min_square_sum(100), 1) 

    def test_random_number(self):
        # 63 = 7^2+ 3^2 + 2^2 + 1^2
        self.assertEqual(min_square_sum(63), 4) 

    def test_random_number2(self):
        # 12 = 4 + 4 + 4
        self.assertEqual(min_square_sum(12), 3) 

    def test_random_number3(self):
        # 6 = 2 + 2 + 2
        self.assertEqual(3, min_square_sum(6)) 


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```