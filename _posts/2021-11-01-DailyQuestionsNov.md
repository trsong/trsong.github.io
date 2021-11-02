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