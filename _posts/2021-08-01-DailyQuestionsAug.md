---
layout: post
title:  "Daily Coding Problems 2021 Aug to Oct"
date:   2021-08-01 22:22:32 -0700
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


### Sep 2, 2021 \[Medium\] Sum of Squares
---
> **Question:** Given a number n, find the least number of squares needed to sum up to the number. For example, `13 = 3^2 + 2^2`, thus the least number of squares requried is 2. 

**My thoughts:** This problem is equivalent to given a list of sorted square number of 0 to n, find the minimum number of chosen elements st. their sum equal to target. Note that we only need to check 1 to square root of n.

**Solution with DP:** [https://replit.com/@trsong/Least-Number-of-Squares-Sum-to-Target](https://replit.com/@trsong/Least-Number-of-Squares-Sum-to-Target)
```py
import unittest
from math import sqrt

def sum_of_squares(target):
    if is_square(target):
        return 1

    # Let dp[s][i] represents min number of squares sum to s using num from 1 to i
    # dp[s][i] = min(dp[s][i - 1], dp[s - i * i] + 1)
    sqrt_target = int(sqrt(target))
    dp = [[target for _ in range(sqrt_target + 1)] for _ in range(target + 1)]
    for i in range(sqrt_target+1):
        dp[0][i] = 0

    for s in range(1, target + 1):
        for i in range(1, sqrt_target + 1):
            dp[s][i] = dp[s][i - 1]
            if s >= i * i:
                dp[s][i] = min(dp[s][i], dp[s - i * i][i] + 1)
    return dp[target][sqrt_target]


def is_square(num):
    sqrt_num = int(sqrt(num))
    return sqrt_num * sqrt_num == num


class SumOfSquareSpec(unittest.TestCase):        
    def test_example(self):
        self.assertEqual(2, sum_of_squares(13))  # return 2 since 13 = 3^2 + 2^2 = 9 + 4.

    def test_example2(self):
        self.assertEqual(3, sum_of_squares(27))  # return 3 since 27 = 3^2 + 3^2 + 3^2 = 9 + 9 + 9

    def test_zero(self):
        self.assertEqual(1, sum_of_squares(0)) # 0 = 0^2

    def test_perfect_square(self):
        self.assertEqual(1, sum_of_squares(100))  # 10^2 = 100

    def test_random_number(self):
        self.assertEqual(4, sum_of_squares(63))  # return 4 since 63 = 7^2+ 3^2 + 2^2 + 1^2

    def test_random_number2(self):
        self.assertEqual(3, sum_of_squares(12))  # return 3 since 12 = 4 + 4 + 4

    def test_random_number3(self):
        self.assertEqual(3, sum_of_squares(6))  # 6 = 2 + 2 + 2


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 1, 2021 \[Medium\] Numbers With Equal Digit Sum
---
> **Question:** Given an array containing integers, find two integers a, b such that sum of digits of a and b is equal. Return maximum sum of a and b. Return -1 if no such numbers exist.

**Example 1:**
```py
Input: [51, 71, 17, 42, 33, 44, 24, 62]
Output: 133
Explanation: Max sum can be formed by 71 + 62 which has same digit sum of 8
```


**Example 2:**
```py
Input: [51, 71, 17, 42]
Output: 93
Explanation: Max sum can be formed by 51 + 42 which has same digit sum of 6
```


**Example 3:**
```py
Input: [42, 33, 60]
Output: 102
Explanation: Max sum can be formed by 42 + 60 which has same digit sum of 6
```


**Example 4:**
```py
Input: [51, 32, 43]
Output: -1
Explanation: There are no 2 numbers with same digit sum
```


**Solution:** [https://replit.com/@trsong/Numbers-With-Max-Equal-Digit-Sum](https://replit.com/@trsong/Numbers-With-Max-Equal-Digit-Sum)
```py
import unittest

def find_max_digit_sum(nums):
    groupby_dsum = {}
    for num in nums:
        dsum = digit_sum(num)
        if dsum not in groupby_dsum:
            groupby_dsum[dsum] = []
        
        dsum_group = groupby_dsum[dsum]
        if len(dsum_group) > 1:
            larger, smaller = dsum_group
            dsum_group[0] = max(larger, num)
            dsum_group[1] = larger + smaller + num - dsum_group[0] - min(num, smaller)
        else:
            dsum_group.append(num)
    
    res = -1
    for dsum_group in groupby_dsum.values():
        if len(dsum_group) == 2:
            res = max(res, sum(dsum_group))
    return res


def digit_sum(num):
    res = 0
    while num > 0:
        res += num % 10
        num //= 10
    return res


class FindMaxDigitSumSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(133, find_max_digit_sum([51, 71, 17, 42, 33, 44, 24, 62]))  # 71 + 62

    def test_example2(self):
        self.assertEqual(93, find_max_digit_sum([51, 71, 17, 42]))  # 51 + 42

    def test_example3(self):
        self.assertEqual(102, find_max_digit_sum([42, 33, 60]))  # 63 + 60

    def test_example4(self):
        self.assertEqual(-1, find_max_digit_sum([51, 32, 43]))

    def test_empty_array(self):
        self.assertEqual(-1, find_max_digit_sum([]))

    def test_same_digit_sum_yet_different_digits(self):
        self.assertEqual(11000, find_max_digit_sum([0, 1, 10, 100, 1000, 10000]))  # 10000 + 1000

    def test_special_edge_case(self):
        self.assertEqual(22, find_max_digit_sum([11, 11, 22, 33]))  # 11 + 11


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 31, 2021 \[Easy\] Reconstruct a Jumbled Array
---
> **Question:** The sequence `[0, 1, ..., N]` has been jumbled, and the only clue you have for its order is an array representing whether each number is larger or smaller than the last. 
> 
> Given this information, reconstruct an array that is consistent with it. For example, given `[None, +, +, -, +]`, you could return `[1, 2, 3, 0, 4]`.

**My thoughts:** treat `+` as `+1`, `+2`, `+3` and `-` as `-1`, `-2` from `0` you can generate an array with satisfied condition yet incorrect range. Then all we need to do is to shift to range from `0` to `N` by minusing each element with global minimum value. 

**Solution:** [https://replit.com/@trsong/Reconstruct-a-Jumbled-Array-by-Given-Order](https://replit.com/@trsong/Reconstruct-a-Jumbled-Array-by-Given-Order)
```py
import unittest

def build_jumbled_array(clues):
    n = len(clues)
    res = [0] * n
    upper = lower = 0

    for i in range(n):
        if clues[i] == '+':
            res[i] = upper + 1
            upper += 1
        elif clues[i] == '-':
            res[i] = lower - 1
            lower -= 1
    
    for i in range(n):
        res[i] -= lower
    
    return res
    

class BuildJumbledArraySpec(unittest.TestCase):
    @staticmethod
    def generate_clues(nums):
        nums_signs = [None] * len(nums)
        for i in range(1, len(nums)):
            nums_signs[i] = '+' if nums[i] > nums[i-1] else '-'
        return nums_signs

    def validate_result(self, cludes, res):
        res_set = set(res)
        res_signs = BuildJumbledArraySpec.generate_clues(res)
        msg = "Incorrect result %s. Expect %s but gives %s." % (res, cludes, res_signs)
        self.assertEqual(len(cludes), len(res_set), msg)
        self.assertEqual(0, min(res), msg)
        self.assertEqual(len(cludes) - 1, max(res), msg)
        self.assertEqual(cludes, res_signs, msg)

    def test_example(self):
        clues = [None, '+', '+', '-', '+']
        # possible solution: [1, 2, 3, 0, 4]
        res = build_jumbled_array(clues)
        self.validate_result(clues, res)

    def test_empty_array(self):
        self.assertEqual([], build_jumbled_array([]))

    def test_one_element_array(self):
        self.assertEqual([0], build_jumbled_array([None]))

    def test_two_elements_array(self):
        self.assertEqual([1, 0], build_jumbled_array([None, '-']))

    def test_ascending_array(self):
        clues = [None, '+', '+', '+']
        expected = [0, 1, 2, 3]
        self.assertEqual(expected, build_jumbled_array(clues))

    def test_descending_array(self):
        clues = [None, '-', '-', '-', '-']
        expected = [4, 3, 2, 1, 0]
        self.assertEqual(expected, build_jumbled_array(clues))

    def test_random_array(self):
        clues = [None, '+', '-', '+', '-']
        # possible solution: [1, 4, 2, 3, 0]
        res = build_jumbled_array(clues)
        self.validate_result(clues, res)

    def test_random_array2(self):
        clues = [None, '-', '+', '-', '-', '+']
        # possible solution: [3, 1, 4, 2, 0, 5]
        res = build_jumbled_array(clues)
        self.validate_result(clues, res)

    def test_random_array3(self):
        clues = [None, '+', '-', '+', '-', '+', '+', '+']
        # possible solution: [1, 7, 0, 6, 2, 3, 4, 5]
        res = build_jumbled_array(clues)
        self.validate_result(clues, res)
    
    def test_random_array4(self):
        clues = [None, '+', '+', '-', '+']
        # possible solution: [1, 2, 3, 0, 4]
        res = build_jumbled_array(clues)
        self.validate_result(clues, res)

    def test_random_array5(self):
        clues = [None, '-', '-', '-', '+']
        # possible solution: [3, 2, 1, 0, 4]
        res = build_jumbled_array(clues)
        self.validate_result(clues, res)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 30, 2021  \[Easy\] Find Unique Element among Array of Duplicates
---
> **Question:** Given an array of integers, arr, where all numbers occur twice except one number which occurs once, find the number. Your solution should ideally be `O(n)` time and use constant extra space.

**Example:**
```py
Input: arr = [7, 3, 5, 5, 4, 3, 4, 8, 8]
Output: 7
```


**My thoughts:** XOR has the following properties:

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

**Solution:** [https://replit.com/@trsong/Find-the-Unique-Element-among-Array-of-Duplicates](https://replit.com/@trsong/Find-the-Unique-Element-among-Array-of-Duplicates)
```py
import unittest

def find_unique_element(nums):
    res = 0
    for num in nums:
        res ^= num
    return res


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
    unittest.main(exit=False, verbosity=2)
```


### Aug 29, 2021 LC 375 \[Medium\] Guess Number Higher or Lower II
---
> **Question:** We are playing the Guess Game. The game is as follows:
>
> I pick a number from 1 to n. You have to guess which number I picked.
> 
> Every time you guess wrong, I'll tell you whether the number I picked is higher or lower.
> 
> However, when you guess a particular number x, and you guess wrong, you pay $x. You win the game when you guess the number I picked.
>
> Given a particular n ≥ 1, find out how much money you need to have to guarantee a win.

**Example:**
```py
n = 10, I pick 8.

First round:  You guess 5, I tell you that it's higher. You pay $5.
Second round: You guess 7, I tell you that it's higher. You pay $7.
Third round:  You guess 9, I tell you that it's lower. You pay $9.

Game over. 8 is the number I picked.

You end up paying $5 + $7 + $9 = $21.
```

**My thoughts:** The guarantee amount is the maximum amount of money you have to pay no matter how unlucky you are. i.e. the strategy is to take the best move given the worest luck. 

Suppose `n = 4`. The best garentee to lose minimum strategy is to first guess 1, if not work, then guess 3. If you are just unlucky, the target number is 2 or 4, then you only need to pay at most `$1 + $3 = $4` in worest case scenario whereas other strategies like choosing 1 through 4 one by one will yield `$1 + $2 + $3 = $6` in worest case when the target is `$4`.

The game play strategy is called ***Minimax***, which is basically choose the maximum among the minimum gain.


**Solution with Minimax:** [https://replit.com/@trsong/Calculate-Guarantee-Money-to-Guess-Number-Higher-or-Lower-II](https://replit.com/@trsong/Calculate-Guarantee-Money-to-Guess-Number-Higher-or-Lower-II)
```py
import unittest

def guess_number_cost(n):
    cache = [[None for _ in range(n+1)] for _ in range(n+1)]
    return -guess_number_between(1, n, cache)


def guess_number_between(lo, hi, cache):
    if lo >= hi:
        return 0
    elif cache[lo][hi] is not None:
        return cache[lo][hi]
    
    res = float('-inf')
    for i in range(lo, hi+1):
        # Assuming we are just so unlucky.
        # The guarantee amount should always be the best among those worest choices
        res = max(res, -i + min(guess_number_between(lo, i-1, cache), guess_number_between(i+1, hi, cache)))
    cache[lo][hi] = res
    return res


class GuessNumberCostSpec(unittest.TestCase):
    def test_n_equals_one(self):
        self.assertEqual(0, guess_number_cost(1))

    def test_n_equals_two(self):
        # pick: 1
        self.assertEqual(1, guess_number_cost(2)) 

    def test_n_equals_three(self):
        # Worest case target=3
        # choose 2, target is higher, pay $2
        # total = $2
        self.assertEqual(2, guess_number_cost(3)) 

    def test_n_equals_four(self):
        # Worest case target=4
        # choose 1, target is higher, pay $1
        # choose 3, target is higher, pay $3
        # total = $4
        self.assertEqual(4, guess_number_cost(4)) 

    def test_n_equals_five(self):
        # pick: 2, 4
        self.assertEqual(6, guess_number_cost(5)) 

    def test_n_equals_six(self):
        # pick: 3, 5
        self.assertEqual(8, guess_number_cost(6)) 

    def test_n_equals_5(self):
        # Worest case target=5
        # choose 2, target is higher, pay $2
        # choose 4, target is higher, pay $4
        # total = $6
        self.assertEqual(6, guess_number_cost(5))

    def test_n_equals_10(self):
        # Worest case target=10
        # choose 7, target is higher, pay $7
        # choose 9, target is higher, pay $9
        # total = $16
        self.assertEqual(16, guess_number_cost(10))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 28, 2021 LC 312 \[Hard\] Burst Balloons
---
> **Question:** Given n balloons, indexed from 0 to n-1. Each balloon is painted with a number on it represented by array nums. You are asked to burst all the balloons. If the you burst balloon i you will get `nums[left] * nums[i] * nums[right]` coins. Here left and right are adjacent indices of i. After the burst, the left and right then becomes adjacent.
>
> Find the maximum coins you can collect by bursting the balloons wisely.
>
> Note:
>
> You may imagine nums[-1] = nums[n] = 1. They are not real therefore you can not burst them.

**Example:**
```py
Input: [3,1,5,8]
Output: 167 
Explanation: nums = [3,1,5,8] --> [3,5,8] -->   [3,8]   -->  [8]  --> []
             coins =  3*1*5      +  3*5*8    +  1*3*8      + 1*8*1   = 167
```


**My thoughts:** think about the problem backwards: the last balloon will have coins coins[-1] * coins[i] * coins[n] for some i. 
We can solve this problem recursively to figure out the index i at each step to give the maximum coins. That gives recursive formula:

```py
burst_in_range_recur(left, right) = max of (coins[left] * coins[i] * coins[right] + burst_in_range_recur(left, i) + burst_in_range_recur(i, right)) for all i between left and right.
```

The final result is by calling `burst_in_range_recur(-1, n)`.


**Solution with Top-down DP:** [https://replit.com/@trsong/Max-Profit-from-Bursting-Balloons](https://replit.com/@trsong/Max-Profit-from-Bursting-Balloons)
```py
def burst_balloons(coins):
    n = len(coins)
    cache = [[None for _ in range(n + 2)] for _ in range(n + 2)]
    return burst_balloons_recur(coins, -1, n, cache)


def burst_balloons_recur(coins, left, right, cache):
    if right - left <= 1:
        return 0
    elif cache[left][right] is None:
        res = 0
        left_coin = coins[left] if left >= 0 else 1
        right_coin = coins[right] if right < len(coins) else 1
        for i in range(left + 1, right):
            left_res = burst_balloons_recur(coins, left, i, cache)
            right_res = burst_balloons_recur(coins, i, right, cache)
            res = max(res, left_coin * coins[i] * right_coin + left_res + right_res)
        cache[left][right] = res
    return cache[left][right]
```

**Solution with Bottom-up DP:** [https://replit.com/@trsong/Max-Profit-from-Bursting-Balloons-Top-down-DP](https://replit.com/@trsong/Max-Profit-from-Bursting-Balloons-Top-down-DP)
```py
import unittest

def burst_balloons(coins):
    if not coins:
        return 0

    n = len(coins)
    # Let dp[left][right] represents max profit for coins[left:right + 1]
    # dp[left][right] = max {dp[left][i - 1] + dp[i + 1][right] + coins[i] * coins[left - 1] * coins[right + 1] } for all i btw left and right
    dp = [[None for _ in range(n)] for _ in range(n)]
    for delta in range(n):
        for left in range(n):
            right = left + delta
            if right >= n:
                break
            res = 0
            left_coin = coins[left - 1] if left > 0 else 1
            right_coin = coins[right + 1] if right < n - 1 else 1
            for i in range(left, right + 1):
                left_res = dp[left][i - 1] if i > left else 0
                right_res = dp[i + 1][right] if i < right else 0
                res = max(res, left_res + right_res + coins[i] * left_coin * right_coin)
            dp[left][right] = res
    return dp[0][n - 1]
                

class BurstBalloonSpec(unittest.TestCase):
    def test_example(self):
        # Burst 1, 5, 3, 8 in order gives:
        # 3*1*5 + 3*5*8 + 1*3*8 + 1*8*1 = 167
        self.assertEqual(167, burst_balloons([3, 1, 5, 8])) 

    def test_ascending_balloons(self):
        # Burst 3, 2, 1, 4 in order gives:
        # 2*3*4 + 1*2*4 + 1*1*4 + 1*4*1 = 40
        self.assertEqual(40, burst_balloons([1, 2, 3, 4]))

    def test_empty_array(self):
        self.assertEqual(0, burst_balloons([]))

    def test_one_elem_array(self):
        self.assertEqual(24, burst_balloons([24]))

    def test_two_elem_array(self):
        # 2*4 + 4 = 12
        self.assertEqual(12, burst_balloons([2, 4]))

    def test_two_elem_array2(self):
        # 1*5 + 5 = 10
        self.assertEqual(10, burst_balloons([1, 5]))

    def test_three_elem_array(self):
        # 3*6*2 + 3*2 + 3 = 45
        self.assertEqual(45, burst_balloons([3, 6, 2]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Aug 27, 2021 \[Medium\] Allocate Minimum Number of Pages
---
> **Question:** Given number of pages in n different books and m students. The books are arranged in ascending order of number of pages. Every student is assigned to read some consecutive books. The task is to assign books in such a way that the maximum number of pages assigned to a student is minimum.

**Example:**
```py
Input : pages[] = {12, 34, 67, 90}
        m = 2
Output : 113

Explanation:
There are 2 number of students. Books can be distributed 
in following fashion : 
  1) [12] and [34, 67, 90]
      Max number of pages is allocated to student 
      2 with 34 + 67 + 90 = 191 pages
  2) [12, 34] and [67, 90]
      Max number of pages is allocated to student
      2 with 67 + 90 = 157 pages 
  3) [12, 34, 67] and [90]
      Max number of pages is allocated to student 
      1 with 12 + 34 + 67 = 113 pages
Of the 3 cases, Option 3 has the minimum pages = 113.       
```

**Solution with Binary Search:** [https://replit.com/@trsong/Minimize-the-Maximum-Page-Assigned-to-Students](https://replit.com/@trsong/Minimize-the-Maximum-Page-Assigned-to-Students)
```py
import unittest

def allocate_min_num_books(pages, num_students):
    lo = max(pages)
    hi = sum(pages)

    # Binary search smallest group_capacity that can form groups
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if can_form_group(mid, pages, num_students):
            hi = mid
        else:
            lo = mid + 1 
    return lo

    
def can_form_group(group_capacity, pages, num_students):
    """
    Set max capacity per group and check if there are enough students
    """
    accu = 0
    group = 0
    for page in pages:
        accu += page
        if accu > group_capacity:
            group += 1
            accu = page
    
        if group >= num_students:
            return False
    return True


class AllocateMinNumBooks(unittest.TestCase):
    def test_two_students(self):
        pages = [12, 34, 67, 90]
        num_students = 2
        # max of book sum([12, 34, 67], [90]) = 12 + 34 + 67 = 113
        expected = 113
        self.assertEqual(expected, allocate_min_num_books(pages, num_students)) 

    def test_three_students(self):
        pages = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        num_students = 3
        # max of book sum([1, 1, 1], [1, 1, 1], [1, 1, 1]) = 1 + 1 + 1 = 3
        expected = 3
        self.assertEqual(expected, allocate_min_num_books(pages, num_students)) 

    def test_four_students(self):
        pages = [100, 101, 102, 103, 104]
        num_students = 4
        # max of book sum([100, 101], [102], [103], [104]) = 100 + 101 = 201
        expected = 201
        self.assertEqual(expected, allocate_min_num_books(pages, num_students)) 

    def test_five_students(self):
        pages = [8, 9, 8, 8, 6, 7, 8, 9, 10]
        num_students = 5
        # max of book sum([9, 8], [8, 8], [6, 7], [8, 9], [10]) = 9 + 8 = 17
        expected = 17
        self.assertEqual(expected, allocate_min_num_books(pages, num_students)) 


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 26, 2021 \[Medium\] Number of Moves on a Grid
---
> **Question:**  There is an N by M matrix of zeroes. Given N and M, write a function to count the number of ways of starting at the top-left corner and getting to the bottom-right corner. You can only move right or down.
>
> For example, given a 2 by 2 matrix, you should return 2, since there are two ways to get to the bottom-right:
>
> - Right, then down
> - Down, then right
>
> Given a 5 by 5 matrix, there are 70 ways to get to the bottom-right.

**Solution with DP:** [https://replit.com/@trsong/Count-Number-of-Moves-on-a-Grid](https://replit.com/@trsong/Count-Number-of-Moves-on-a-Grid)
```py
import unittest

def calc_num_moves(grid_height, grid_width):
    if grid_height <= 0 or grid_width <= 0:
        return 0

    dp_prev = [1] * grid_width
    dp_cur = [1] * grid_width

    for _ in range(1, grid_height):
        for col in range(1, grid_width):
            dp_cur[col] = dp_cur[col - 1] + dp_prev[col]
        dp_cur, dp_prev = dp_prev, dp_cur
    return dp_prev[-1]
        

class CalcNumMoveSpec(unittest.TestCase):
    def test_size_zero_grid(self):
        self.assertEqual(calc_num_moves(0, 0), 0)
    
    def test_size_zero_grid2(self):
        self.assertEqual(calc_num_moves(1, 0), 0)
    
    def test_size_zero_grid3(self):
        self.assertEqual(calc_num_moves(0, 1), 0)

    def test_square_grid(self):
        self.assertEqual(calc_num_moves(1, 1), 1)
    
    def test_square_grid2(self):
        self.assertEqual(calc_num_moves(3, 3), 6)
    
    def test_square_grid3(self):
        self.assertEqual(calc_num_moves(5, 5), 70)

    def test_rectangle_grid(self):
        self.assertEqual(calc_num_moves(1, 5), 1)

    def test_rectangle_grid2(self):
        self.assertEqual(calc_num_moves(5, 1), 1)

    def test_rectangle_grid3(self):
        self.assertEqual(calc_num_moves(2, 3), 3)
    
    def test_rectangle_grid4(self):
        self.assertEqual(calc_num_moves(3, 2), 3)

    def test_large_grid(self):
         self.assertEqual(calc_num_moves(10, 20), 6906900)
    
    def test_large_grid2(self):
         self.assertEqual(calc_num_moves(20, 10), 6906900)

    def test_large_grid3(self):
         self.assertEqual(calc_num_moves(20, 20), 35345263800)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 25, 2021 \[Medium\] Fixed Order Task Scheduler with Cooldown
--- 
> **Question:** We have a list of tasks to perform, with a cooldown period. We can do multiple of these at the same time, but we cannot run the same task simultaneously.
>
> Given a list of tasks, find how long it will take to complete the tasks in the order they are input.

**Example:**
```py
tasks = [1, 1, 2, 1]
cooldown = 2
output: 7 (order is 1 _ _ 1 2 _ 1)
```

**My thoughts:** Since we have to execute the task with specific order and each task has a cooldown time, we can use a map to record the last occurence of the same task and set up a threshold in order to make sure we will always wait at least the cooldown amount of time before proceed.

**Solution with Hashmap:** [https://replit.com/@trsong/Calculate-Fixed-Order-Task-Scheduler-with-Cooldown](https://replit.com/@trsong/Calculate-Fixed-Order-Task-Scheduler-with-Cooldown)
```py
import unittest

def multitasking_time(task_seq, cooldown):
    cur_time = 0
    last_occurrence = {}
    for task in task_seq:
        delta = cur_time - last_occurrence.get(task, float('-inf'))
        idle = max(cooldown - delta + 1, 0)
        cur_time += idle

        last_occurrence[task] = cur_time
        cur_time += 1
    return cur_time

    
class MultitaskingTimeSpec(unittest.TestCase):
    def test_example(self):
        tasks = [1, 1, 2, 1]
        cooldown = 2
        # order is 1 _ _ 1 2 _ 1
        self.assertEqual(7, multitasking_time(tasks, cooldown))

    def test_example2(self):
        tasks = [1, 1, 2, 1, 2]
        cooldown = 2
        # order is 1 _ _ 1 2 _ 1 2
        self.assertEqual(8, multitasking_time(tasks, cooldown))
    
    def test_zero_cool_down_time(self):
        tasks = [1, 1, 1]
        cooldown = 0
        # order is 1 1 1 
        self.assertEqual(3, multitasking_time(tasks, cooldown))
    
    def test_task_queue_is_empty(self):
        tasks = []
        cooldown = 100
        self.assertEqual(0, multitasking_time(tasks, cooldown))

    def test_cooldown_is_three(self):
        tasks = [1, 2, 1, 2, 1, 1, 2, 2]
        cooldown = 3
        # order is 1 2 _ _ 1 2 _ _ 1 _ _ _ 1 2 _ _ _ 2
        self.assertEqual(18, multitasking_time(tasks, cooldown))
    
    def test_multiple_takes(self):
        tasks = [1, 2, 3, 1, 3, 2, 1, 2]
        cooldown = 2
        # order is 1 2 3 1 _ 3 2 1 _ 2
        self.assertEqual(10, multitasking_time(tasks, cooldown))

    def test_when_cool_down_is_huge(self):
        tasks = [1, 2, 2, 1, 2]
        cooldown = 100
        # order is 1 2 [_ * 100] 2 1 [_ * 99] 2
        self.assertEqual(204, multitasking_time(tasks, cooldown))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Aug 24, 2021 \[Hard\] Longest Common Subsequence of Three Strings
--- 
> **Question:** Write a program that computes the length of the longest common subsequence of three given strings. For example, given `"epidemiologist"`, `"refrigeration"`, and `"supercalifragilisticexpialodocious"`, it should return `5`, since the longest common subsequence is `"eieio"`.

**Solution with DP:** [https://replit.com/@trsong/Longest-Common-Subsequence-of-3-Strings-2](https://replit.com/@trsong/Longest-Common-Subsequence-of-3-Strings-2)
```py
import unittest

def lcs(seq1, seq2, seq3):
    n, m, t = len(seq1), len(seq2), len(seq3)
    # Let dp[n][m][t] represents lcs for seq1[:n], seq2[:m], seq3[:t]
    # dp[n][m][t] = 1 + dp[n-1][m-1][t-1] if all of seq1[n - 1], seq2[m - 1], seq3[t - 1] match
    #             = max of subproblems: dp[i - 1][j][k], dp[i][j - 1][k], dp[i][j][k - 1] otherwise
    dp = [[[0 for _ in range(t + 1)] for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            for k in range(1, t + 1):
                if seq1[i - 1] == seq2[j - 1] == seq3[k - 1]:
                    dp[i][j][k] = 1 + dp[i - 1][j - 1][k - 1]
                else:
                    dp[i][j][k] = max(dp[i - 1][j][k], dp[i][j - 1][k], dp[i][j][k - 1])
    
    return dp[n][m][t]
    

class LCSSpec(unittest.TestCase):
    def test_empty_sequences(self):
        self.assertEqual(0, lcs("", "", ""))
    
    def test_example(self):
        self.assertEqual(5, lcs(
            "epidemiologist",
            "refrigeration",
            "supercalifragilisticexpialodocious"))  # "eieio"

    def test_match_last_position(self):
        self.assertEqual(1, lcs("abcdz", "efghijz", "123411111z"))  # z

    def test_match_first_position(self):
        self.assertEqual(1, lcs("aefgh", "aijklmnop", "a12314213"))  # a

    def test_off_by_one_position(self):
        self.assertEqual(4, lcs("10101", "01010", "0010101"))  # 0101

    def test_off_by_one_position2(self):
        self.assertEqual(3, lcs("12345", "1235", "2535"))  # 235

    def test_off_by_one_position3(self):
        self.assertEqual(2, lcs("1234", "1243", "2431"))  # 24

    def test_off_by_one_position4(self):
        self.assertEqual(4, lcs("12345", "12340", "102030400"))  # 1234

    def test_multiple_matching(self):
        self.assertEqual(5, lcs("afbgchdie",
                                "__a__b_c__de___f_g__h_i___", "/a/b/c/d/e"))  # abcde

    def test_ascending_vs_descending(self):
        self.assertEqual(1, lcs("01234", "_4__3___2_1_0__", "4_3_2_1_0"))  # 0

    def test_multiple_ascending(self):
        self.assertEqual(5, lcs("012312342345", "012345", "0123401234"))  # 01234

    def test_multiple_descending(self):
        self.assertEqual(5, lcs("54354354421", "5432432321", "54321"))  # 54321

    def test_same_length_strings(self):
        self.assertEqual(2, lcs("ABCD", "EACB", "AFBC"))  # AC


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Aug 23, 2021 \[Medium\] Jumping Numbers
---
> **Question:** Given a positive int n, print all jumping numbers smaller than or equal to n. A number is called a jumping number if all adjacent digits in it differ by 1. For example, 8987 and 4343456 are jumping numbers, but 796 and 89098 are not. All single digit numbers are considered as jumping numbers.

**Example:**

```py
Input: 105
Output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 21, 23, 32, 34, 43, 45, 54, 56, 65, 67, 76, 78, 87, 89, 98, 101]
```

**My thoughts:** We can use Brutal Force to search from 0 to given upperbound to find jumping numbers which might not be so efficient. Or we can take advantage of the property of jummping number: a jumping number is:
- either one of all single digit numbers 
- or 10 * some jumping number + last digit of that jumping number +/- by 1

For example, 

```
1 -> 10, 12.
2 -> 21, 23.
3 -> 32, 34.
...
10 -> 101
12 -> 121, 123
```

We can get all qualified jumping number by BFS searching for all candidates.

**Solution with BFS:** [https://replit.com/@trsong/Generate-Jumping-Numbers](https://replit.com/@trsong/Generate-Jumping-Numbers)
```py
import unittest

def generate_jumping_numbers(upper_bound):
    if upper_bound < 0:
        return []

    queue = [num for num in range(1, 10)]
    res = [0]
    while queue:
        cur = queue.pop(0)
        if cur > upper_bound:
            break

        res.append(cur)
        last_digit = cur % 10
        if last_digit > 0:
            queue.append(10 * cur + last_digit - 1)
        
        if last_digit < 9:
            queue.append(10 * cur + last_digit + 1)
    return res


class GenerateJumpingNumberSpec(unittest.TestCase):
    def test_zero_as_upperbound(self):
        self.assertEqual([0], generate_jumping_numbers(0))

    def test_single_digits_are_all_jummping_numbers(self):
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], generate_jumping_numbers(9))

    def test_five_as_upperbound(self):
        self.assertEqual([0, 1, 2, 3, 4, 5], generate_jumping_numbers(5))

    def test_not_always_contains_upperbound(self):
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], generate_jumping_numbers(13))

    def test_example(self):
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 21, 23, 32, 34, 43, 45, 54, 56, 65, 67, 76, 78, 87, 89, 98, 101]
        self.assertEqual(expected, generate_jumping_numbers(105))
    
    def test_negative_upperbound(self):
        self.assertEqual([], generate_jumping_numbers(-1))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 22, 2021 \[Easy\] Plus One
--- 
> **Question:** Given a non-empty array where each element represents a digit of a non-negative integer, add one to the integer. The most significant digit is at the front of the array and each element in the array contains only one digit. Furthermore, the integer does not have leading zeros, except in the case of the number '0'.

**Example:**
```py
Input: [2, 3, 4]
Output: [2, 3, 5]
```

**Solution:** [https://replit.com/@trsong/Integer-Plus-One](https://replit.com/@trsong/Integer-Plus-One)
```py
import unittest

def add_one(digits):
    carry = 1
    for i in range(len(digits) - 1, -1, -1):
        digits[i] += carry
        carry = digits[i] // 10
        digits[i] %= 10

        if carry == 0:
            break
    
    if carry:
        digits.insert(0, 1)
    return digits


class AddOneSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual([2, 3, 5], add_one([2, 3, 4]))

    def test_zero(self):
        self.assertEqual([1], add_one([0]))
    
    def test_two_digit_number(self):
        self.assertEqual([9, 0], add_one([8, 9]))
    
    def test_four_digit_number(self):
        self.assertEqual([3, 1, 0, 0], add_one([3, 0, 9, 9]))

    def test_carryover(self):
        self.assertEqual([1, 0], add_one([9]))

    def test_carryover_and_early_break(self):
        self.assertEqual([2, 8, 3, 0, 0], add_one([2, 8, 2, 9, 9]))

    def test_early_break(self):
        self.assertEqual([1, 0, 0, 1], add_one([1, 0, 0, 0]))

    def test_carryover2(self):
        self.assertEqual([1, 0, 0, 0, 0], add_one([9, 9, 9, 9]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 21, 2021 LC 1136 \[Hard\] Parallel Courses
---
> **Question:** There are N courses, labelled from 1 to N.
>
> We are given `relations[i] = [X, Y]`, representing a prerequisite relationship between course X and course Y: course X has to be studied before course Y.
>
> In one semester you can study any number of courses as long as you have studied all the prerequisites for the course you are studying.
>
> Return the minimum number of semesters needed to study all courses.  If there is no way to study all the courses, return -1.


**Example 1:**
```py
Input: N = 3, relations = [[1,3],[2,3]]
Output: 2
Explanation: 
In the first semester, courses 1 and 2 are studied. In the second semester, course 3 is studied.
```

**Example 2:**
```py
Input: N = 3, relations = [[1,2],[2,3],[3,1]]
Output: -1
Explanation: 
No course can be studied because they depend on each other.
```

**My thoughts:** Treat each course as vertex and prerequisite relation as edges. We will have a directed acyclic graph (DAG) with edge weight = 1. The problem then convert to find the longest path in a DAG. To find the answer, we need to find topological order of courses and then find longest path based on topological order.  

**Solution with Topological Search:** [https://replit.com/@trsong/Calculate-Parallel-Courses](https://replit.com/@trsong/Calculate-Parallel-Courses)
```py
import unittest

def min_semesters(total_courses, prerequisites):
    if total_courses == 0:
        return 0

    neighbors = [[] for _ in range(total_courses + 1)]
    inbound = [0] * (total_courses + 1)
    for cur, next in prerequisites:
        neighbors[cur].append(next)
        inbound[next] += 1
    
    semesters = [1] * (total_courses + 1)
    queue = [course for course in range(1, total_courses + 1) if inbound[course] == 0]
    
    while queue:
        cur = queue.pop(0)
        for next in neighbors[cur]:
            inbound[next] -= 1
            if inbound[next] == 0:
                queue.append(next)
            semesters[next] = semesters[cur] + 1

    if any(inbound):
        # exist circular dependencies
        return -1

    return max(semesters)


class min_semesterss(unittest.TestCase):
    def test_example(self):
        total_courses = 3
        prerequisites = [[1, 3], [2, 3]]
        expected = 2  # gradudation path: 1/2, 3
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_no_course_to_take(self):
        total_courses = 0
        prerequisites = []
        expected = 0
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_all_courses_are_independent(self):
        total_courses = 3
        prerequisites = []
        expected = 1  # gradudation path: 1/2/3
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_grap_with_cycle(self):
        total_courses = 3
        prerequisites = [[1, 2], [3, 1], [2, 3]]
        expected = -1
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_grap_with_cycle2(self):
        total_courses = 2
        prerequisites = [[1, 2], [2, 1]]
        expected = -1
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_disconnected_graph(self):
        total_courses = 5
        prerequisites = [[1, 2], [3, 4], [4, 5]]
        expected = 3  # gradudation path: 1/3, 2/4, 5
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_graph_with_two_paths(self):
        total_courses = 5
        prerequisites = [[1, 2], [2, 5], [1, 3], [3, 4], [4, 5]]
        expected = 4  # gradudation path: 1, 2/3, 4, 5
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_graph_with_two_paths2(self):
        total_courses = 5
        prerequisites = [[1, 3], [3, 4], [4, 5], [1, 2], [2, 5]]
        expected = 4  # gradudation path: 1, 2/3, 4, 5
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_connected_graph_with_paths_of_different_lenghths(self):
        total_courses = 7
        prerequisites = [[1, 3], [1, 4], [2, 3], [2, 4], [3, 4], [3, 6],
                         [4, 5], [4, 6], [4, 7], [3, 6], [6, 7]]
        expected = 5  # path: 1/2, 3, 4, 5/6, 7
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Aug 20, 2021 \[Medium\] Cutting a Rod
---
> **Question:** Given a rod of length n inches and an array of prices that contains prices of all pieces of size smaller than n. Determine the maximum value obtainable by cutting up the rod and selling the pieces. 
>
> For example, if length of the rod is 8 and the values of different pieces are given as following, then the maximum obtainable value is 22 (by cutting in two pieces of lengths 2 and 6) = 5 + 17

```
length   | 1   2   3   4   5   6   7   8  
--------------------------------------------
price    | 1   5   8   9  10  17  17  20
```

> And if the prices are as following, then the maximum obtainable value is 24 (by cutting in eight pieces of length 1) = 3 + 3 + 3 + 3 + 3 + 3 + 3 + 3

```
length   | 1   2   3   4   5   6   7   8  
--------------------------------------------
price    | 3   5   8   9  10  17  17  20
```

**My thoughts:** Think about the problem backwards: among all rot cutting spaces, at the very last step we have to choose one cut the length n rod into `{ all (first_rod, second_rod)} = {(0, n), (1, n-1), (2, n-2), ..., (n-1, 1)}` And suppose there is a magic function f(n) that can gives the max price we can get when rod length is n, then we can say that `f(n) = max of f(first_rod) + price of second_rod for all (first_rod, second_roc)`, that means, `f(n) = max(f(n-k) + price(k)) (index is 1-based) for all k` which can be solved using DP.

**Solution with DP:** [https://replit.com/@trsong/Max-Value-to-Cut-a-Rod](https://replit.com/@trsong/Max-Value-to-Cut-a-Rod)
```py
import unittest

def max_cut_rod_price(piece_prices):
    # Let dp[i] where 0 <= i <= n, represents max cut price with i pieces of rods
    # dp[i] = max(dp[i - k] + piece_prices[k - 1]) for all k <= i
    n = len(piece_prices)
    dp = [0] * (n + 1)
    for i in range(n + 1):
        for k in range(1, i + 1):
            dp[i] = max(dp[i], dp[i - k] + piece_prices[k - 1])
    return dp[n]


class MaxCutRodPriceSpec(unittest.TestCase):
    def test_all_cut_to_one(self):
        # 3 + 3 + 3 = 9
        self.assertEqual(9, max_cut_rod_price([3, 4, 5])) 

    def test_cut_to_one_and_two(self):
        # 3 + 7 = 10
        self.assertEqual(10, max_cut_rod_price([3, 7, 8])) 

    def test_when_cut_has_tie(self):
        # 4 or 1 + 3
        self.assertEqual(4, max_cut_rod_price([1, 3, 4])) 

    def test_no_need_to_cut(self):
        self.assertEqual(5, max_cut_rod_price([1, 2, 5]))

    def test_example1(self):
        # 5 + 17 = 22
        self.assertEqual(22, max_cut_rod_price([1, 5, 8, 9, 10, 17, 17, 20]))

    def test_example2(self):
        # 3 * 8 = 24
        self.assertEqual(24, max_cut_rod_price([3, 5, 8, 9, 10, 17, 17, 20]))
        

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Aug 19, 2021 \[Medium\] Forward DNS Look Up Cache
----
> **Question:** Forward DNS look up is getting IP address for a given domain name typed in the web browser. e.g. Given "www.samsung.com" should return "107.108.11.123"
> 
> The cache should do the following operations:
> 1. Add a mapping from URL to IP address
> 2. Find IP address for a given URL.
> 
> Note: 
> - You can assume all domain name contains only lowercase characters
> 
> Hint:
> - The idea is to store URLs in Trie nodes and store the corresponding IP address in last or leaf node.

**Solution with Trie:** [https://replit.com/@trsong/Design-Forward-DNS-Look-Up-Cache](https://replit.com/@trsong/Design-Forward-DNS-Look-Up-Cache)
```py
import unittest

class ForwardDNSCache(object):
    def __init__(self):
        self.trie = Trie()

    def insert(self, url, ip):
        self.trie.insert(url, ip)

    def search(self, url):
        return self.trie.search(url)


class Trie(object):
    def __init__(self):
        self.ip = None
        self.children = None
    
    def insert(self, url, ip):
        p = self
        for ch in url:
            p.children = p.children or {}
            p.children[ch] = p.children.get(ch, Trie())
            p = p.children[ch]
        p.ip = ip

    def search(self, url):
        p = self
        for ch in url:
            if not p or not p.children or ch not in p.children:
                return None
            p = p.children[ch]
        return p.ip


class ForwardDNSCacheSpec(unittest.TestCase):
    def setUp(self):
        self.cache = ForwardDNSCache()
    
    def test_fail_to_get_out_of_bound_result(self):
        self.assertIsNone(self.cache.search("www.google.ca"))
        self.cache.insert("www.google.com", "1.1.1.1")
        self.assertIsNone(self.cache.search("www.google.com.ca"))

    def test_result_not_found(self):
        self.cache.insert("www.cnn.com", "2.2.2.2")
        self.assertIsNone(self.cache.search("www.amazon.ca"))
        self.assertIsNone(self.cache.search("www.cnn.ca"))

    def test_overwrite_url(self):
        self.cache.insert("www.apple.com", "1.2.3.4")
        self.assertEqual( "1.2.3.4", self.cache.search("www.apple.com"))
        self.cache.insert("www.apple.com", "5.6.7.8")
        self.assertEqual("5.6.7.8", self.cache.search("www.apple.com"))
    
    def test_url_with_same_prefix(self):
        self.cache.insert("www.apple.com", "1.2.3.4")
        self.cache.insert("www.apple.com.ca", "5.6.7.8")
        self.cache.insert("www.apple.com.hk", "9.10.11.12")
        self.assertEqual("1.2.3.4", self.cache.search("www.apple.com"), )
        self.assertEqual("5.6.7.8", self.cache.search("www.apple.com.ca"))
        self.assertEqual("9.10.11.12", self.cache.search("www.apple.com.hk"))

    def test_non_overlapping_url(self):
        self.cache.insert("bilibili.tv", "11.22.33.44")
        self.cache.insert("taobao.com", "55.66.77.88")
        self.assertEqual("11.22.33.44", self.cache.search("bilibili.tv"))
        self.assertEqual("55.66.77.88", self.cache.search("taobao.com"))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 18, 2021 \[Easy\] Rearrange Array in Alternating Positive & Negative Order
---
> **Question:** Given an array of positive and negative numbers, arrange them in an alternate fashion such that every positive number is followed by negative and vice-versa maintaining the order of appearance.
> 
> Number of positive and negative numbers need not be equal. If there are more positive numbers they appear at the end of the array. If there are more negative numbers, they too appear in the end of the array.

**Example1:**
```py
Input:  arr[] = {1, 2, 3, -4, -1, 4}
Output: arr[] = {-4, 1, -1, 2, 3, 4}
```

**Example2:**
```py
Input:  arr[] = {-5, -2, 5, 2, 4, 7, 1, 8, 0, -8}
output: arr[] = {-5, 5, -2, 2, -8, 4, 7, 1, 8, 0} 
```

**Solution:** [https://replit.com/@trsong/Rearrange-the-Array-in-Alternating-Positive-and-Negative-Order](https://replit.com/@trsong/Rearrange-the-Array-in-Alternating-Positive-and-Negative-Order)
```py
import unittest

def rearrange_array(nums):
    positives = []
    negatives = []
    for num in nums:
        if num >= 0:
            positives.append(num)
        else:
            negatives.append(num)
    
    i = j = k = 0
    while k < len(nums):
        if i < len(negatives):
            nums[k] = negatives[i]
            i += 1
            k += 1
        
        if j < len(positives):
            nums[k] = positives[j]
            j += 1
            k += 1   
    return nums

    
class RearrangeArraySpec(unittest.TestCase):
    def test_example(self):
        nums = [1, 2, 3, -4, -1, 4]
        expected = [-4, 1, -1, 2, 3, 4]
        self.assertEqual(expected, rearrange_array(nums))

    def test_example2(self):
        nums = [-5, -2, 5, 2, 4, 7, 1, 8, 0, -8]
        expected = [-5, 5, -2, 2, -8, 4, 7, 1, 8, 0]
        self.assertEqual(expected, rearrange_array(nums))

    def test_more_negatives_than_positives(self):
        nums = [-1, -2, -3, -4, 1, 2, -5, -6]
        expected =  [-1, 1, -2, 2, -3, -4, -5, -6]
        self.assertEqual(expected, rearrange_array(nums))

    def test_more_positives_than_negatives(self):
        nums = [-1, 1, 2, 3, -2]
        expected =  [-1, 1, -2, 2, 3]
        self.assertEqual(expected, rearrange_array(nums))

    def test_empty_array(self):
        nums = []
        expected = []
        self.assertEqual(expected, rearrange_array(nums))

    def test_no_negatives(self):
        nums = [1, 1, 2, 3]
        expected = [1, 1, 2, 3]
        self.assertEqual(expected, rearrange_array(nums))

    def test_no_positive_array(self):
        nums = [-1]
        expected = [-1]
        self.assertEqual(expected, rearrange_array(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 17, 2021 \[Easy\] Height-balanced Binary Tree
---
> **Question:** Given a binary tree, determine whether or not it is height-balanced. A height-balanced binary tree can be defined as one in which the heights of the two subtrees of any node never differ by more than one.

**Solution with Recursion:** [https://replit.com/@trsong/Determine-If-Height-balanced-Binary-Tree](https://replit.com/@trsong/Determine-If-Height-balanced-Binary-Tree)
```py
import unittest

def is_balanced_tree(root):
    res, _ = check_height_recur(root)
    return res


def check_height_recur(node):
    if node is None:
        return True, 0

    left_res, left_height = check_height_recur(node.left)
    if not left_res:
        return False, -1

    right_res, right_height = check_height_recur(node.right)
    if not right_res or abs(left_height - right_height) > 1:
        return False, -1

    return True, max(left_height, right_height) + 1


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class IsBalancedTreeSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertTrue(is_balanced_tree(None))

    def test_tree_with_only_one_node(self):
        self.assertTrue(is_balanced_tree(TreeNode(1)))

    def test_depth_one_tree_is_always_balanced(self):
        """
        0
         \
          1
        """
        root = TreeNode(0, right=TreeNode(1))
        self.assertTrue(is_balanced_tree(root))

    def test_depth_two_unbalanced_tree(self):
        """
         0
          \
           1
          / \
         2   3
        """
        right_tree = TreeNode(1, TreeNode(2), TreeNode(3))
        root = TreeNode(0, right=right_tree)
        self.assertFalse(is_balanced_tree(root))

    def test_left_heavy_tree(self):
        """
          0
         / \
        1   4
         \
          2
         /
        3
        """
        left_tree = TreeNode(1, right=TreeNode(2, TreeNode(3)))
        root = TreeNode(0, left_tree, TreeNode(4))
        self.assertFalse(is_balanced_tree(root))

    def test_complete_tree(self):
        """
            0
           / \
          1   2
         / \ 
        3   4
        """
        left_tree = TreeNode(1, TreeNode(3), TreeNode(4))
        root = TreeNode(0, left_tree, TreeNode(2))
        self.assertTrue(is_balanced_tree(root))

    def test_unbalanced_internal_node(self):
        """
            0
           / \
          1   3
         /     \
        2       4
               /
              5 
        """
        left_tree = TreeNode(1, TreeNode(2))
        right_tree = TreeNode(3, right=TreeNode(4, TreeNode(5)))
        root = TreeNode(0, left_tree, right_tree)
        self.assertFalse(is_balanced_tree(root))

    def test_balanced_internal_node(self):
        """
            0
           / \
          1   3
         /   / \
        2   6   4
               /
              5 
        """
        left_tree = TreeNode(1, TreeNode(2))
        right_tree = TreeNode(3, TreeNode(6), TreeNode(4, TreeNode(5)))
        root = TreeNode(0, left_tree, right_tree)
        self.assertTrue(is_balanced_tree(root))

    def test_unbalanced_tree_with_all_children_filled(self):
        """
            0
           / \ 
          1   2
         / \
        3   4
           / \
          5   6
        """
        n4 = TreeNode(4, TreeNode(5), TreeNode(6))
        n1 = TreeNode(1, TreeNode(3), n4)
        root = TreeNode(0, n1, TreeNode(2))
        self.assertFalse(is_balanced_tree(root))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 16, 2021 \[Easy\] Zombie in Matrix
---
> **Question:** Given a 2D grid, each cell is either a zombie 1 or a human 0. Zombies can turn adjacent (up/down/left/right) human beings into zombies every hour. Find out how many hours does it take to infect all humans?

**Example:**
```py
Input:
[[0, 1, 1, 0, 1],
 [0, 1, 0, 1, 0],
 [0, 0, 0, 0, 1],
 [0, 1, 0, 0, 0]]

Output: 2

Explanation:
At the end of the 1st hour, the status of the grid:
[[1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1],
 [0, 1, 0, 1, 1],
 [1, 1, 1, 0, 1]]

At the end of the 2nd hour, the status of the grid:
[[1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1]]
 ```

**My thoughts:** The problem can be solved with BFS. Just imagining initially all zombines are virtually associated with a single root where after we perform a BFS, the result will be depth of the BFS search tree.

**Solution with BFS:** [https://replit.com/@trsong/Zombie-Infection-in-Matrix](https://replit.com/@trsong/Zombie-Infection-in-Matrix)
```py
import unittest

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def zombie_infection_time(grid):
    if not grid or not grid[0]:
        return -1
    
    n, m = len(grid), len(grid[0])
    queue = []
    for r in range(n):
        for c in range(m):
            if grid[r][c] == 1:
                queue.append((r, c))
    
    time = -1
    while queue:
        for _ in range(len(queue)):
            r, c = queue.pop(0)
            if time > 0 and grid[r][c] == 1:
                continue
            grid[r][c] = 1
            
            for dr, dc in DIRECTIONS:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < n and 0 <= new_c < m and grid[new_r][new_c] == 0:
                    queue.append((new_r, new_c))
        time += 1
    return time


class ZombieInfectionTimeSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(2, zombie_infection_time([
            [0, 1, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0]]))

    def test_empty_grid(self):
        # Assume grid with no zombine returns -1
        self.assertEqual(-1, zombie_infection_time([]))
    
    def test_grid_without_zombie(self):
        # Assume grid with no zombine returns -1
        self.assertEqual(-1, zombie_infection_time([
            [0, 0, 0],
            [0, 0, 0]
        ]))

    def test_1x1_grid(self):
        self.assertEqual(0, zombie_infection_time([[1]]))

    def test_when_all_human_are_infected(self):
        self.assertEqual(0, zombie_infection_time([
            [1, 1],
            [1, 1]]))
    
    def test_grid_with_one_zombie(self):
        self.assertEqual(4, zombie_infection_time([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]))
    
    def test_grid_with_one_zombie2(self):
        self.assertEqual(2, zombie_infection_time([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]))
    
    def test_grid_with_multiple_zombies(self):
        self.assertEqual(4, zombie_infection_time([
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0]]))
    
    def test_grid_with_multiple_zombies2(self):
        self.assertEqual(4, zombie_infection_time([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
 ```

### Aug 15, 2021 \[Medium\] Power of 4
---
> **Questions:** Given a 32-bit positive integer N, determine whether it is a power of four in faster than `O(log N)` time.

**Example1:**
```py
Input: 16
Output: 16 is a power of 4
```

**Example2:**
```py
Input: 20
Output: 20 is not a power of 4
```

**My thoughts:** A power-of-4 number must be power-of-2. Thus `n & (n-1) == 0` must hold. The only different between pow-4 and pow-2 is the number of trailing zeros. pow-4 must have even number of trailing zeros. So we can either check that through `n & 0xAAAAAAAA` (`0xA` in binary `0b1010`) or just use binary search to count zeros. 

**Solution with Binary Search:** [https://replit.com/@trsong/Determine-If-Number-Is-Power-of-4](https://replit.com/@trsong/Determine-If-Number-Is-Power-of-4)
```py
import unittest

def is_power_of_four(num):
    is_positive = num > 0
    is_power_two = num & (num - 1) == 0
    
    if is_positive and is_power_two:
        # binary search number of zeros
        lo = 0
        hi = 32
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if num == (1 << mid):
                return mid % 2 == 0
            elif num >= (1 << mid):
                lo = mid + 1
            else:
                hi = mid - 1
    return False


class IsPowerOfFourSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(is_power_of_four(16))

    def test_example2(self):
        self.assertFalse(is_power_of_four(20))

    def test_zero(self):
        self.assertFalse(is_power_of_four(0))
    
    def test_one(self):
        self.assertTrue(is_power_of_four(1))

    def test_number_smaller_than_four(self):
        self.assertFalse(is_power_of_four(3))

    def test_negative_number(self):
        self.assertFalse(is_power_of_four(-4))
    
    def test_all_bit_being_one(self):
        self.assertFalse(is_power_of_four(4**8 - 1))

    def test_power_of_two_not_four(self):
        self.assertFalse(is_power_of_four(2 ** 5))

    def test_all_power_4_bit_being_one(self):
        self.assertFalse(is_power_of_four(4**0 + 4**1 + 4**2 + 4**3 + 4**4))
    
    def test_larger_number(self):
        self.assertTrue(is_power_of_four(2 ** 32))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 14, 2021 \[Medium\] Add Subtract Currying
---
> **Question:** Write a function, add_subtract, which alternately adds and subtracts curried arguments. 

**Example:**
```py
add_subtract(7) -> 7

add_subtract(1)(2)(3) -> 1 + 2 - 3 -> 0

add_subtract(-5)(10)(3)(9) -> -5 + 10 - 3 + 9 -> 11
```

**Solution:** [https://replit.com/@trsong/Add-and-Subtract-Currying](https://replit.com/@trsong/Add-and-Subtract-Currying)
```py
import unittest

def add_subtract(first=None):
    class Number:
        def __init__(self, val, sign=None):
            self.val = val
            self.sign = sign if sign else 1

        def __call__(self, val):
            new_val = self.val + self.sign * val
            new_sign = -1 * self.sign 
            return Number(new_val, new_sign)

        def __str__(self):
            return str(self.val)

    return Number(first if first else 0)


class AddSubtractSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual('7', str(add_subtract(7)))

    def test_example2(self):
        self.assertEqual('0', str(add_subtract(1)(2)(3)))

    def test_example3(self):
        self.assertEqual('11', str(add_subtract(-5)(10)(3)(9)))

    def test_empty_argument(self):
        self.assertEqual('0', str(add_subtract()))

    def test_positive_arguments(self):
        self.assertEqual('4', str(add_subtract(1)(2)(3)(4)))

    def test_negative_arguments(self):
        self.assertEqual('9', str(add_subtract(-1)(-3)(-5)(-1)(-9)))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 13, 2021 \[Easy\] N-th Perfect Number
---
> **Question:** A number is considered perfect if its digits sum up to exactly `10`.
> 
> Given a positive integer `n`, return the n-th perfect number.
> 
> For example, given `1`, you should return `19`. Given `2`, you should return `28`.

**My thougths:** notice the pattern that consecutive terms differ by `9`. But there is exception: `100` is not perfect number.

**Solution:** [https://replit.com/@trsong/Find-N-th-Perfect-Number](https://replit.com/@trsong/Find-N-th-Perfect-Number)
```py
import unittest

def perfect_number_at(n):
    target = 10
    diff = 9
    res = 19
    for _ in range(n - 1):
        while True:
            res += diff
            if sum_digits(res) == target:
                break
    return res


def sum_digits(num):
    res = 0
    while num > 0:
        res += num % 10
        num //= 10
    return res


class PerfectNumberAtSpec(unittest.TestCase):
    """
    Perfect number from 1st to 50th term:
    
    [ 19,  28,  37,  46,  55,  64,  73,  82,  91, 109,
     118, 127, 136, 145, 154, 163, 172, 181, 190, 208,
     217, 226, 235, 244, 253, 262, 271, 280, 307, 316,
     325, 334, 343, 352, 361, 370, 406, 415, 424, 433,
     442, 451, 460, 505, 514, 523, 532, 541, 550, 604]
    """
    def test_1st_term(self):
        self.assertEqual(19, perfect_number_at(1))

    def test_2nd_term(self):
        self.assertEqual(28, perfect_number_at(2))

    def test_3rd_term(self):
        self.assertEqual(37, perfect_number_at(3))

    def test_4th_term(self):
        self.assertEqual(46, perfect_number_at(4))

    def test_5th_term(self):
        self.assertEqual(55, perfect_number_at(5))

    def test_6th_term(self):
        self.assertEqual(64, perfect_number_at(6))

    def test_10th_term(self):
        self.assertEqual(109, perfect_number_at(10))

    def test_42nd_term(self):
        self.assertEqual(451, perfect_number_at(42))

    def test_51th_term(self):
        self.assertEqual(604, perfect_number_at(50))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 12, 2021 \[Medium\] XOR Linked List
--- 
> **Question:** An XOR linked list is a more memory efficient doubly linked list. Instead of each node holding next and prev fields, it holds a field named both, which is an XOR of the next node and the previous node. Implement an XOR linked list; it has an `add(element)` which adds the element to the end, and a `get(index)` which returns the node at index.
>
> If using a language that has no pointers (such as Python), you can assume you have access to `get_pointer` and `dereference_pointer` functions that converts between nodes and memory addresses.

**My thoughts:** For each node, store XOR(prev pointer, next pointer) value. Upon retrival, take advantage of XOR's property `(prev XOR next) XOR prev = next` and  `(prev XOR next) XOR next = prev` and we shall be able to iterative through entire list.

**Solution:** [https://replit.com/@trsong/XOR-Linked-List](https://replit.com/@trsong/XOR-Linked-List)
```py
import ctypes
import unittest

class XORLinkedList(object):
    def __init__(self):
        self.head = None
        self.tail = None

    def add(self, node):
        if not self.head:
            self.head = node
            self.tail = node
        else:
            self.tail.both ^= get_pointer(node)
            node.both = get_pointer(self.tail)
            self.tail = node

    def get(self, index):
        prev_pointer = 0
        node = self.head
        for _ in range(index):
            next_pointer = prev_pointer ^ node.both
            prev_pointer = get_pointer(node)
            node = dereference_pointer(next_pointer)
        return node


#######################
# Testing Utilities
#######################
class Node(object):
    # Workaround to prevent GC
    all_nodes = []

    def __init__(self, val):
        self.val = val
        self.both = 0
        Node.all_nodes.append(self)


get_pointer = id
dereference_pointer = lambda ptr: ctypes.cast(ptr, ctypes.py_object).value


class XORLinkedListSpec(unittest.TestCase):
    def test_empty_list(self):
        self.assertIsNotNone(XORLinkedList())

    def test_one_element_list(self):
        lst = XORLinkedList()
        lst.add(Node(1))
        self.assertEqual(1, lst.get(0).val)

    def test_two_element_list(self):
        lst = XORLinkedList()
        lst.add(Node(1))
        lst.add(Node(2))
        self.assertEqual(2, lst.get(1).val)

    def test_list_with_duplicated_values(self):
        lst = XORLinkedList()
        lst.add(Node(1))
        lst.add(Node(2))
        lst.add(Node(1))
        lst.add(Node(2))
        self.assertEqual(1, lst.get(0).val)
        self.assertEqual(2, lst.get(1).val)
        self.assertEqual(1, lst.get(2).val)
        self.assertEqual(2, lst.get(3).val)

    def test_larger_example(self):
        lst = XORLinkedList()
        for i in range(1000):
            lst.add(Node(i))

        for i in range(100):
            self.assertEqual(i, lst.get(i).val)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Aug 11, 2021 \[Easy\] Permutation with Given Order
---
> **Question:** A permutation can be specified by an array `P`, where `P[i]` represents the location of the element at `i` in the permutation. For example, `[2, 1, 0]` represents the permutation where elements at the index `0` and `2` are swapped.
>
> Given an array and a permutation, apply the permutation to the array. 
>
> For example, given the array `["a", "b", "c"]` and the permutation `[2, 1, 0]`, return `["c", "b", "a"]`.

**My thoughts:** In-place solution requires swapping `i` with `j` if `j > i`. However, if `j < i`, then `j`'s position has been swapped, we backtrack recursively to find `j`'s new position.

**Solution:** [https://replit.com/@trsong/Find-Permutation-with-Given-Order](https://replit.com/@trsong/Find-Permutation-with-Given-Order)
```py
import unittest

def permute(arr, order):
    for i in range(len(order)):
        target_pos = order[i]

        # Check index if it has already been swapped before
        while target_pos < i:
            target_pos = order[target_pos]
            
        arr[i], arr[target_pos] = arr[target_pos], arr[i]
    return arr
        

class PermuteSpec(unittest.TestCase):
    def test_example(self):
        arr = ["a", "b", "c"]
        order = [2, 1, 0]
        expected = ["c", "b", "a"]
        self.assertEqual(expected, permute(arr, order))

    def test_example2(self):
        arr = ["11", "32", "3", "42"]
        order = [2, 3, 0, 1]
        expected = ["3", "42", "11", "32"]
        self.assertEqual(expected, permute(arr, order))

    def test_empty_array(self):
        self.assertEqual([], permute([], []))

    def test_order_form_different_cycles(self):
        arr = ["a", "b", "c", "d", "e"]
        order = [1, 0, 4, 2, 3]
        expected = ["b", "a", "e", "c", "d"]
        self.assertEqual(expected, permute(arr, order))
    
    def test_reverse_order(self):
        arr = ["a", "b", "c", "d"]
        order = [3, 2, 1, 0]
        expected = ["d", "c", "b", "a"]
        self.assertEqual(expected, permute(arr, order))

    def test_nums_array(self):
        arr = [50, 40, 70, 60, 90]
        order = [3,  0,  4,  1,  2]
        expected = [60, 50, 90, 40, 70]
        self.assertEqual(expected, permute(arr, order))

    def test_nums_array2(self):
        arr = [9, 3, 7, 6, 2]
        order = [4, 0, 3, 1, 2]
        expected = [2, 9, 6, 3, 7]
        self.assertEqual(expected, permute(arr, order))

    def test_array_with_duplicate_number(self):
        arr = ['a', 'a', 'b', 'c']
        order = [0, 2, 1, 3]
        expected = ['a', 'b', 'a', 'c']
        self.assertEqual(expected, permute(arr, order))

    def test_fail_to_swap(self):
        arr = [50, 30, 40, 70, 60, 90]
        order = [3, 5, 0, 4, 1, 2]
        expected = [70, 90, 50, 60, 30, 40]
        self.assertEqual(expected, permute(arr, order))

    def test_already_in_correct_order(self):
        arr = [0, 1, 2, 3]
        order = [0, 1, 2, 3]
        expected = [0, 1, 2, 3]
        self.assertEqual(expected, permute(arr, order))
        

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 10, 2021 LC 16 \[Medium\] Closest to 3 Sum
---
> **Question:** Given a list of numbers and a target number n, find 3 numbers in the list that sums closest to the target number n. There may be multiple ways of creating the sum closest to the target number, you can return any combination in any order.

**Example:**
```py
Input: [2, 1, -5, 4], -1
Output: [-5, 1, 2]
Explanation: Closest sum is -5+1+2 = -2 OR -5+1+4 = 0
```

**Solution with Sorting and Two-Pointers:** [https://replit.com/@trsong/Closest-to-3-Sum-2](https://replit.com/@trsong/Closest-to-3-Sum-2)
```py
import unittest

def closest_3_sum(nums, target):
    nums.sort()

    res = None
    min_delta = float('inf')
    n = len(nums)
    for i in range(n - 2):
        lo = i + 1
        hi = n - 1
        while lo < hi:
            total = nums[i] + nums[lo] + nums[hi]
            if abs(total - target) < min_delta:
                min_delta = abs(total - target)
                res = [nums[i], nums[lo], nums[hi]]

            if total < target:
                lo += 1
            elif  total > target:
                hi -= 1
    return res

    
class Closest3SumSpec(unittest.TestCase):
    def test_example(self):
        target, nums = -1, [2, 1, -5, 4]
        expected1 = [-5, 1, 2]
        expected2 = [-5, 1, 4]
        self.assertIn(sorted(closest_3_sum(nums, target)), [expected1, expected2])

    def test_example2(self):
        target, nums = 1, [-1, 2, 1, -4]
        expected = [-1, 2, 1]
        self.assertEqual(sorted(expected), sorted(closest_3_sum(nums, target)))

    def test_example3(self):
        target, nums = 10, [1, 2, 3, 4, -5]
        expected = [2, 3, 4]
        self.assertEqual(sorted(expected), sorted(closest_3_sum(nums, target)))

    def test_empty_array(self):
        target, nums = 0, []
        self.assertIsNone(closest_3_sum(nums, target))

    def test_array_without_enough_elements(self):
        target, nums = 3, [1, 2]
        self.assertIsNone(closest_3_sum(nums, target))

    def test_array_without_enough_elements2(self):
        target, nums = 3, [3]
        self.assertIsNone(closest_3_sum(nums, target))


    def test_all_negatives(self):
        target, nums = 10, [-2, -1, -5]
        expected = [-2, -1, -5]
        self.assertEqual(sorted(expected), sorted(closest_3_sum(nums, target)))

    def test_all_negatives2(self):
        target, nums = -10, [-2, -1, -5, -10]
        expected = [-2, -1, -5]
        self.assertEqual(sorted(expected), sorted(closest_3_sum(nums, target)))

    def test_negative_target(self):
        target, nums = -10, [0, 0, 0, 0, 0, 1, 1, 1, 1]
        expected = [0, 0, 0]
        self.assertEqual(sorted(expected), sorted(closest_3_sum(nums, target)))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 9, 2021 \[Medium\] Number of Flips to Make Binary String
---
> **Question:** You are given a string consisting of the letters `x` and `y`, such as `xyxxxyxyy`. In addition, you have an operation called flip, which changes a single `x` to `y` or vice versa.
>
> Determine how many times you would need to apply this operation to ensure that all x's come before all y's. In the preceding example, it suffices to flip the second and sixth characters, so you should return 2.

**My thoughts:** Basically, the question is about finding a sweet cutting spot so that # flip on left plus # flip on right is minimized. We can simply scan through the array from left and right to allow constant time query for number of flip need on the left and right for a given spot. And the final answer is just the min of sum of left and right flips.

**Solution with DP:** [https://replit.com/@trsong/Find-Number-of-Flips-to-Make-Binary-String-2](https://replit.com/@trsong/Find-Number-of-Flips-to-Make-Binary-String-2)
```py
import unittest

def min_flip_to_make_binary(s):
    n = len(s)
    left_y_count = [None] * n
    right_x_count = [None] * n
    left_accu = right_accu = 0

    for i in range(n):
        left_y_count[i] = left_accu
        left_accu += 1 if s[i] == 'y' else 0

        right_x_count[n - 1 - i] = right_accu
        right_accu += 1 if s[n - 1 - i] == 'x' else 0

    res = n
    for i in range(n):
        res = min(res, left_y_count[i] + right_x_count[i]) 
    return res


class MinFlipToMakeBinarySpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(2, min_flip_to_make_binary('xyxxxyxyy'))  # xxxxxxxyy

    def test_empty_string(self):
        self.assertEqual(0, min_flip_to_make_binary(''))
    
    def test_already_binary(self):
        self.assertEqual(0, min_flip_to_make_binary('xxxxxxxyy'))

    def test_flipped_string(self):
        self.assertEqual(3, min_flip_to_make_binary('yyyxxx'))  # yyyyyy

    def test_flip_all_x(self):
        self.assertEqual(1, min_flip_to_make_binary('yyx'))  # yyy

    def test_flip_all_y(self):
        self.assertEqual(2, min_flip_to_make_binary('yyxxx'))  # xxxxx

    def test_flip_all_y2(self):
        self.assertEqual(4, min_flip_to_make_binary('xyxxxyxyyxxx'))  # xxxxxxxxxxxx

    def test_flip_y(self):
        self.assertEqual(2, min_flip_to_make_binary('xyxxxyxyyy'))  # xxxxxxxyyy

    def test_flip_x_and_y(self):
        self.assertEqual(3, min_flip_to_make_binary('xyxxxyxyyx'))  # xxxxxyyyyy


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 8, 2021 \[Easy\] Sevenish Number
---
> **Question:** Let's define a "sevenish" number to be one which is either a power of 7, or the sum of unique powers of 7. The first few sevenish numbers are 1, 7, 8, 49, and so on. Create an algorithm to find the nth sevenish number.

**My thoughts:** The n-th "sevenish" number its just binary number based 7.
```py
# "0" => 7^0
# "10" => 7^1     
# "11" => 7^1 + 7^0
# "100" => 7^2 
# "101" => 7^2 + 7^0 = 50
```

**Solution:** [https://replit.com/@trsong/Sevenish-Numbe](https://replit.com/@trsong/Sevenish-Numbe)
```py
import unittest

def sevenish_number(n):
    res = 0
    shift_amt = 0
    while n: 
        if n & 1:
            res += 7 ** shift_amt
        shift_amt += 1
        n >>= 1
    return res


class SevennishNumberSpec(unittest.TestCase):
    def test_1st_term(self):
        # "0" => 7^0
        self.assertEqual(1, sevenish_number(1))

    def test_2nd_term(self):
        # "10" => 7^1
        self.assertEqual(7, sevenish_number(2))

    def test_3rd_term(self):
        # "11" => 7^1 + 7^0
        self.assertEqual(8, sevenish_number(3))

    def test_4th_term(self):
        # "100" => 7^2 
        self.assertEqual(49, sevenish_number(4))

    def test_5th_term(self):
        # "101" => 7^2 + 7^0 = 50
        self.assertEqual(50, sevenish_number(5))

    def test_6th_term(self):
        self.assertEqual(56, sevenish_number(6))

    def test_7th_term(self):
        self.assertEqual(57, sevenish_number(7))

    def test_8th_term(self):
        self.assertEqual(343, sevenish_number(8))

    def test_9th_term(self):
        self.assertEqual(344, sevenish_number(9))

    def test_10th_term(self):
        self.assertEqual(350, sevenish_number(10))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 7, 2021 LC 212 \[Hard\] Word Search II
---
> **Question:** Given an m x n board of characters and a list of strings words, return all words on the board.
>
> Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.
 
**Example 1:**
```py
Input: words = ["oath","pea","eat","rain"], board = [
    ["o","a","a","n"],
    ["e","t","a","e"],
    ["i","h","k","r"],
    ["i","f","l","v"]]
Output: ["eat", "oath"]
```

**Example 2:**
```py
Input: words = ["abcb"], board = [
    ["a","b"],
    ["c","d"]]
Output: []
```


**Solution with Trie and Backtracking:** [https://replit.com/@trsong/Word-Search-II-2](https://replit.com/@trsong/Word-Search-II-2)
```py
import unittest

def search_word(board, words):
    if not board or not board[0]:
        return []

    trie = Trie()
    for word in words:
        trie.insert(word)

    n, m = len(board), len(board[0])
    res = []
    for r in range(n):
        for c in range(m):
            backtrack(board, res, trie, (r, c))

    return res


class Trie(object):
    def __init__(self):
        self.children = {}
        self.word = None

    def insert(self, word):
        p = self
        for ch in word:
            p.children[ch] = p.children.get(ch, Trie())
            p = p.children[ch]
        p.word = word


DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]


def backtrack(board, res, trie, pos):
    r, c = pos
    ch = board[r][c]
    if not ch or ch not in trie.children:
        return
    trie = trie.children[ch]
    if trie.word:
        res.append(trie.word)
        trie.word = None

    board[r][c] = None
    n, m = len(board), len(board[0])
    for dr, dc in DIRECTIONS:
        new_r, new_c = r + dr, c + dc
        if 0 <= new_r < n and 0 <= new_c < m and board[new_r][new_c]:
            backtrack(board, res, trie, (new_r, new_c))
    board[r][c] = ch


class SearchWordSpec(unittest.TestCase):
    def assert_result(self, expected, res):
        self.assertEqual(sorted(expected), sorted(res))

    def test_example(self):
        words = ['oath', 'pea', 'eat', 'rain']
        board = [['o', 'a', 'a', 'n'], ['e', 't', 'a', 'e'],
                 ['i', 'h', 'k', 'r'], ['i', 'f', 'l', 'v']]
        expected = ['eat', 'oath']
        self.assert_result(expected, search_word(board, words))

    def test_example2(self):
        words = ['abcb']
        board = [['a', 'b'], ['c', 'd']]
        expected = []
        self.assert_result(expected, search_word(board, words))

    def test_unique_char(self):
        words = ['a', 'aa', 'aaa']
        board = [['a', 'a'], ['a', 'a']]
        expected = ['a', 'aa', 'aaa']
        self.assert_result(expected, search_word(board, words))

    def test_empty_grid(self):
        self.assertEqual([], search_word([], ['a']))

    def test_empty_empty_word(self):
        self.assertEqual([], search_word(['a'], []))

    def test_word_use_all_letters(self):
        words = ['abcdef']
        board = [['a', 'b'], ['f', 'c'], ['e', 'd']]
        expected = ['abcdef']
        self.assert_result(expected, search_word(board, words))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 6, 2021 \[Easy\] Move Zeros
---
> **Question:** Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.
> 
> You must do this in-place without making a copy of the array. Minimize the total number of operations.

**Example:**
```py
Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
```

**Solution:** [https://replit.com/@trsong/Move-Zeros](https://replit.com/@trsong/Move-Zeros)
```py
import unittest

def move_zeros(nums):
    slow = fast = 0
    while fast < len(nums):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1
        fast += 1
        
    while slow < len(nums):
        nums[slow] = 0
        slow += 1


class MoveZeroSpec(unittest.TestCase):
    def test_example(self):
        nums = [0, 1, 0, 3, 12]
        move_zeros(nums)
        expected = [1, 3, 12, 0, 0]
        self.assertEqual(expected, nums)

    def test_empty_array(self):
        nums = []
        move_zeros(nums)
        expected = []
        self.assertEqual(expected, nums)

    def test_adjacent_zeros(self):
        nums = [0, 0, 1, 2, 0, 0, 2, 1, 0, 0]
        move_zeros(nums)
        expected = [1, 2, 2, 1, 0, 0, 0, 0, 0, 0]
        self.assertEqual(expected, nums)

    def test_just_zeros(self):
        nums = [0, 0, 0, 0]
        move_zeros(nums)
        expected = [0, 0, 0, 0]
        self.assertEqual(expected, nums)

    def test_array_with_negative_numbers(self):
        nums = [0, -1, 1, -1, 1]
        move_zeros(nums)
        expected = [-1, 1, -1, 1, 0]
        self.assertEqual(expected, nums)

    def test_without_zeros(self):
        nums = [-1, 1, 2, 3, -4]
        move_zeros(nums)
        expected = [-1, 1, 2, 3, -4]
        self.assertEqual(expected, nums)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 5, 2021 \[Hard\] First Unique Character from a Stream
---
> **Question:** Given a stream of characters, find the first unique (non-repeating) character from stream. You need to tell the first unique character in O(1) time at any moment.

**Example:**
```py
Input: Stream.of('abracadabra')
Output: Stream.of('aaabbbbbrcc')
Explanation:
a => a
abr => a
abra => b
abracadab => r
abracadabra => c
```

**Solution with Hashmap and Double-Linked List:** [https://replit.com/@trsong/First-Unique-Character-from-a-Stream-2](https://replit.com/@trsong/First-Unique-Character-from-a-Stream-2)
```py
import unittest

def find_first_unique_letter(char_stream):
    duplicates = set()
    look_up = {}
    head = ListNode()
    tail = ListNode()
    head.next = tail
    tail.prev = head

    for ch in char_stream:
        if ch in look_up:
            node = look_up[ch]
            node.next.prev = node.prev
            node.prev.next = node.next
            del look_up[ch]
            duplicates.add(ch)
        elif ch not in duplicates:
            tail.data = ch
            look_up[ch] = tail
            tail.next = ListNode(prev=tail)
            tail = tail.next
        yield head.next.data if head.next != tail else None
    

class ListNode(object):
    def __init__(self, data=None, next=None, prev=None):
        self.data = data
        self.next = next
        self.prev = prev


class FindFirstUniqueLetter(unittest.TestCase):
    def assert_result(self, expected, output):
        self.assertEqual(list(expected), list(output))

    def test_example(self):
        char_stream = iter("abracadabra")
        expected = iter("aaabbbbbrcc")
        self.assert_result(expected, find_first_unique_letter(char_stream))

    def test_empty_stream(self):
        char_stream = iter("")
        expected = iter("")
        self.assert_result(expected, find_first_unique_letter(char_stream))

    def test_duplicated_letter_stream(self):
        char_stream = iter("aaa")
        expected = iter(["a", None, None])
        self.assert_result(expected, find_first_unique_letter(char_stream))

    def test_duplicated_letter_stream2(self):
        char_stream = iter("aaabbbccc")
        expected = iter(["a", None, None, "b", None, None, "c", None, None])
        self.assert_result(expected, find_first_unique_letter(char_stream))

    def test_palindrome_stream(self):
        char_stream = iter("abccba")
        expected = iter(["a", "a", "a", "a", "a", None])
        self.assert_result(expected, find_first_unique_letter(char_stream))

    def test_repeated_pattern(self):
        char_stream = iter("abcabc")
        expected = iter(["a", "a", "a", "b", "c", None])
        self.assert_result(expected, find_first_unique_letter(char_stream))

    def test_repeated_pattern2(self):
        char_stream = iter("aabbcc")
        expected = iter(["a", None, "b", None, "c", None])
        self.assert_result(expected, find_first_unique_letter(char_stream))

    def test_unique_characters(self):
        char_stream = iter("abcde")
        expected = iter("aaaaa")
        self.assert_result(expected, find_first_unique_letter(char_stream))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 4, 2021 \[Medium\] Invert a Binary Tree
---
> **Question:** You are given the root of a binary tree. Invert the binary tree in place. That is, all left children should become right children, and all right children should become left children.

**Example:**
```py
Given the following tree:

     a
   /   \
  b     c
 / \   /
d   e f

should become:
   a
 /   \
c     b
 \   / \
  f e   d
```

**Solution with Post-order Traversal:** [https://replit.com/@trsong/Invert-All-Nodes-in-Binary-Tree-2](https://replit.com/@trsong/Invert-All-Nodes-in-Binary-Tree-2)
```py
import unittest

def invert_tree(root):
    if not root:
        return None
    
    left_res = invert_tree(root.left)
    right_res = invert_tree(root.right)
    root.left = right_res
    root.right = left_res
    return root


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __eq__(self, other):
        return (other and
            self.val == other.val and 
            self.left == other.left and 
            self.right == other.right)

    def __repr__(self):
        stack = [(self, 0)]
        res = []
        while stack:
            node, depth = stack.pop()
            res.append("\n" + "\t" * depth)
            if not node:
                res.append("* None")
                continue

            res.append("* " + str(node.val))
            for child in [node.right, node.left]:
                stack.append((child, depth+1))
        return "\n" + "".join(res) + "\n"


class InvertTreeSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertIsNone(invert_tree(None))

    def test_heavy_left_tree(self):
        """
            1
           /
          2
         /
        3
        """
        tree = TreeNode(1, TreeNode(2, TreeNode(3)))
        """
        1
         \
          2
           \
            3
        """
        expected_tree = TreeNode(1, right=TreeNode(2, right=TreeNode(3)))
        self.assertEqual(invert_tree(tree), expected_tree)

    def test_heavy_right_tree(self):
        """
          1
         / \
        2   3
           /
          4 
        """
        tree = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4)))
        """
          1
         / \
        3   2
         \ 
          4         
        """
        expected_tree = TreeNode(1, TreeNode(3, right=TreeNode(4)), TreeNode(2))
        self.assertEqual(invert_tree(tree), expected_tree)

    def test_sample_tree(self):
        """
             1
           /   \
          2     3
         / \   /
        4   5 6
        """
        n2 = TreeNode(2, TreeNode(4), TreeNode(5))
        n3 = TreeNode(3, TreeNode(6))
        n1 = TreeNode(1, n2, n3)
        """
            1
          /   \
         3     2
          \   / \
           6 5   4
        """
        en2 = TreeNode(2, TreeNode(5), TreeNode(4))
        en3 = TreeNode(3, right=TreeNode(6))
        en1 = TreeNode(1, en3, en2)
        self.assertEqual(invert_tree(n1), en1)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 3, 2021 \[Medium\] Zig-Zag String
--- 
> **Question:** Given a string and a number of lines k, print the string in zigzag form. In zigzag, characters are printed out diagonally from top left to bottom right until reaching the kth line, then back up to top right, and so on.

**Example:**
```py
Given the sentence "thisisazigzag" and k = 4, you should print:
t     a     g
 h   s z   a
  i i   i z
   s     g
```

**Solution:** [https://replit.com/@trsong/Print-Zig-Zag-String-2](https://replit.com/@trsong/Print-Zig-Zag-String-2)
```py
import unittest

def zig_zag_format(sentence, k):
    if k == 1:
        return [sentence]

    res = [[] for _ in range(min(k, len(sentence)))]
    direction = -1
    row = 0

    for ch in sentence:
        if not res[row]:
            padding = row
        elif direction < 0:
            padding = 2 * (k - 1 - row) - 1
        else:
            padding = 2 * row - 1

        res[row].append(" " * padding + ch)
        if row == k - 1 or row == 0:
            direction *= -1
        row += direction
    return list(map(lambda line: "".join(line), res))


class ZigZagFormatSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(len(expected), len(result))
        for expected_line, result_line in zip(expected, result):
            self.assertEqual(expected_line.rstrip(), result_line.rstrip())

    def test_example(self):
        k, sentence = 4, "thisisazigzag"
        expected = [
            "t     a     g",
            " h   s z   a ",
            "  i i   i z  ",
            "   s     g   "
        ]
        self.assert_result(expected, zig_zag_format(sentence, k))

    def test_empty_string(self):
        k, sentence = 10, ""
        expected = []
        self.assert_result(expected, zig_zag_format(sentence, k))

    def test_trivial_case(self):
        k, sentence = 1, "lumberjack"
        expected = ["lumberjack"]
        self.assert_result(expected, zig_zag_format(sentence, k))

    def test_split_into_2_rows(self):
        k, sentence = 2, "cheese steak jimmy's"
        expected = [
            "c e s   t a   i m ' ",
            " h e e s e k j m y s"
        ]
        self.assert_result(expected, zig_zag_format(sentence, k))

    def test_k_large_than_sentence(self):
        k, sentence = 10, "rock on"
        expected = [
           "r",
           " o",
           "  c",
           "   k",
           "     ",
           "     o",
           "      n"
        ]
        self.assert_result(expected, zig_zag_format(sentence, k)) 

    def test_k_barely_make_two_folds(self):
        k, sentence = 6, "robin hood"
        expected = [
            "r         ",
            " o       d",
            "  b     o ",
            "   i   o  ",
            "    n h   ",
            "          "
        ]
        self.assert_result(expected, zig_zag_format(sentence, k))

    def test_k_barely_make_three_folds(self):
        k, sentence = 10, "how do you turn this on"
        expected = [
            "h                 i     ",
            " o               h s    ",
            "  w             t       ",
            "                     o  ",
            "    d         n       n ",
            "     o       r          ",
            "            u           ",
            "       y   t            ",
            "        o               ",
            "         u              "
        ]
        self.assert_result(expected, zig_zag_format(sentence, k))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Aug 2, 2021 \[Hard\] Exclusive Product
---
> **Question:**  Given an array of integers, return a new array such that each element at index i of the new array is the product of all the numbers in the original array except the one at i.
>
> For example, if our input was [1, 2, 3, 4, 5], the expected output would be [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output would be [2, 3, 6].
>
> Follow-up: what if you can't use division?

**Solution:** [https://replit.com/@trsong/Calculate-Exclusive-Product-2](https://replit.com/@trsong/Calculate-Exclusive-Product-2)
```py
import unittest

def exclusive_product(nums):
    n = len(nums)
    res = [1] * n
    left_accu = 1
    right_accu = 1
    for i in range(n):
        res[i] *= left_accu
        left_accu *= nums[i]

        res[n - 1 - i] *= right_accu
        right_accu *= nums[n - 1 - i]
    return res


class ExclusiveProductSpec(unittest.TestCase):
    def test_example(self):
        nums = [1, 2, 3, 4, 5]
        expected = [120, 60, 40, 30, 24]
        self.assertEqual(expected, exclusive_product(nums))

    def test_example2(self):
        nums = [3, 2, 1]
        expected = [2, 3, 6]
        self.assertEqual(expected, exclusive_product(nums))

    def test_empty_array(self):
        self.assertEqual([], exclusive_product([]))
    
    def test_one_element_array(self):
        nums = [2]
        expected = [1]
        self.assertEqual(expected, exclusive_product(nums))
    
    def test_two_elements_array(self):
        nums = [42, 98]
        expected = [98, 42]
        self.assertEqual(expected, exclusive_product(nums))
    
    def test_array_with_negative_elements(self):
        nums = [-2, 3, -5]
        expected = [-15, 10, -6]
        self.assertEqual(expected, exclusive_product(nums))
    
    def test_array_with_negative_elements2(self):
        nums = [-1, -3, -4, -5]
        expected = [-60, -20, -15, -12]
        self.assertEqual(expected, exclusive_product(nums))
    
    def test_array_with_zero(self):
        nums = [1, -1, 0, 3]
        expected = [0, 0, -3, 0]
        self.assertEqual(expected, exclusive_product(nums))
    
    def test_array_with_zero2(self):
        nums = [1, -1, 0, 3, 0, 1]
        expected = [0, 0, 0, 0, 0, 0]
        self.assertEqual(expected, exclusive_product(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Aug 1, 2021 \[Hard\] Order of Course Prerequisites
---
> **Question:** We're given a hashmap associating each courseId key with a list of courseIds values, which represents that the prerequisites of courseId are courseIds. Return a sorted ordering of courses such that we can finish all courses.
>
> Return null if there is no such ordering.
>
> For example, given `{'CSC300': ['CSC100', 'CSC200'], 'CSC200': ['CSC100'], 'CSC100': []}`, should return `['CSC100', 'CSC200', 'CSC300']`.


**My thoughts:** there are two ways to produce toplogical sort order: DFS or count inward edge as below. For DFS technique see this [post](https://trsong.github.io/python/java/2019/05/01/DailyQuestions.html#jul-5-2019-hard-order-of-course-prerequisites). For below method, we count number of inward edges for each node and recursively remove edges of each node that has no inward egdes. The node removal order is what we call topological order.

**Solution with Topological Sort:** [https://replit.com/@trsong/Find-Order-of-Course-Prerequisites-2](https://replit.com/@trsong/Find-Order-of-Course-Prerequisites-2)
```py
import unittest

def sort_courses(prereq_map):
    inward_edge = {course: len(prereqs) for course, prereqs in prereq_map.items()}
    neighbours = {course: [] for course in prereq_map}
    queue = []
    for course, prereqs in prereq_map.items():
        if not prereqs:
            queue.append(course)
        for prereq in prereqs:
            neighbours[prereq].append(course)

    top_order = []
    while queue:
        node = queue.pop(0)
        top_order.append(node)

        for nb in neighbours[node]:
            inward_edge[nb] -= 1
            if inward_edge[nb] == 0:
                queue.append(nb)

    return top_order if len(top_order) == len(prereq_map) else None


class SortCourseSpec(unittest.TestCase):
    def assert_course_order_with_prereq_map(self, prereq_map):
        # Test utility for validation of the following properties for each course:
        # 1. no courses can be taken before its prerequisites
        # 2. order covers all courses
        orders = sort_courses(prereq_map)
        self.assertEqual(len(prereq_map), len(orders))

        # build a quick courseId to index lookup 
        course_priority_map = dict(zip(orders, xrange(len(orders))))
        for course, prereq_list in prereq_map.iteritems():
            for prereq in prereq_list:
                # Any prereq course must be taken before the one that depends on it
                self.assertTrue(course_priority_map[prereq] < course_priority_map[course])

    def test_courses_with_mutual_dependencies(self):
        prereq_map = {
            'CS115': ['CS135'],
            'CS135': ['CS115']
        }
        self.assertIsNone(sort_courses(prereq_map))

    def test_courses_within_same_department(self):
        prereq_map = {
            'CS240': [],
            'CS241': [],
            'MATH239': [],
            'CS350': ['CS240', 'CS241', 'MATH239'],
            'CS341': ['CS240'],
            'CS445': ['CS350', 'CS341']
        }
        self.assert_course_order_with_prereq_map(prereq_map)

    def test_courses_in_different_departments(self):
        prereq_map = {
            'MATH137': [],
            'CS116': ['MATH137', 'CS115'],
            'JAPAN102': ['JAPAN101'],
            'JAPAN101': [],
            'MATH138': ['MATH137'],
            'ENGL119': [],
            'MATH237': ['MATH138'],
            'CS246': ['MATH138', 'CS116'],
            'CS115': []
        }
        self.assert_course_order_with_prereq_map(prereq_map)

    def test_courses_without_dependencies(self):
        prereq_map = {
            'ENGL119': [],
            'ECON101': [],
            'JAPAN101': [],
            'PHYS111': []
        }
        self.assert_course_order_with_prereq_map(prereq_map)
   

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```
