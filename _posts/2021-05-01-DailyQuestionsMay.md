---
layout: post
title:  "Daily Coding Problems 2021 May to Jul"
date:   2021-05-01 22:22:32 -0700
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


### May 9, 2021 LT 1859 \[Easy\] Minimum Amplitude
--- 
> **Question:** Given an array A consisting of N integers. In one move, we can choose any element in this array and replace it with any value. The amplitude of an array is the difference between the largest and the smallest values it contains.
> 
> Return the smallest amplitude of array A that we can achieve by performing at most three moves.

**Example 1:**
```py
Input: A = [-9, 8, -1]
Output: 0
Explanation: We can replace -9 and 8 with -1 so that all element are equal to -1, and then the amplitude is 0
```

**Example 2:**
```py
Input: A = [14, 10, 5, 1, 0]
Output: 1
Explanation: To achieve an amplitude of 1, we can replace 14, 10 and 5 with 1 or 0.
```

**Example 3:**
```py
Input: A = [11, 0, -6, -1, -3, 5]
Output: 3
Explanation: This can be achieved by replacing 11, -6 and 5 with three values of -2.
```


### May 8, 2021 LC 1525 \[Medium\] Number of Good Ways to Split a String
--- 
> **Question:** Given a string S, we can split S into 2 strings: S1 and S2. Return the number of ways S can be split such that the number of unique characters between S1 and S2 are the same.

**Example 1:**
```py
Input: "aaaa"
Output: 3
Explanation: we can get a - aaa, aa - aa, aaa- a
```

**Example 2:**
```py
Input: "bac"
Output: 0
```

**Example 3:**
```py
Input: "ababa"
Output: 2
Explanation: ab - aba, aba - ba
```

**Solution with Sliding Window:** [https://replit.com/@trsong/Number-of-Good-Ways-to-Split-a-String](https://replit.com/@trsong/Number-of-Good-Ways-to-Split-a-String)
```py
import unittest

def count_equal_splits(s):
    n = len(s)
    last_occurance = {}
    forward_bit_map = 0
    backward_bit_map = 0

    for i in range(n - 1, -1, -1):
        ch = s[i]
        mask = 1 << ord(ch)
        if ~backward_bit_map & mask:
            last_occurance[ch] = i
            backward_bit_map |= mask

    res = 0
    for i, ch in enumerate(s):
        mask = 1 << ord(ch)
        forward_bit_map |= mask
        if last_occurance[ch] == i:
            backward_bit_map ^= mask
        
        if forward_bit_map == backward_bit_map:
            res += 1

    return res
            

class CountEqualSplitSpec(unittest.TestCase):
    def test_example(self):
        s = "aaaa"
        # a - aaa, aa - aa, aaa- a
        expected = 3
        self.assertEqual(expected, count_equal_splits(s))

    def test_example2(self):
        s = "bac"
        expected = 0
        self.assertEqual(expected, count_equal_splits(s))

    def test_example3(self):
        s = "ababa"
        # ab - aba, aba - ba
        expected = 2
        self.assertEqual(expected, count_equal_splits(s))
    
    def test_empty_string(self):
        s = ""
        expected = 0
        self.assertEqual(expected, count_equal_splits(s))
    
    def test_string_with_unique_characters(self):
        s = "abcdef"
        expected = 0
        self.assertEqual(expected, count_equal_splits(s))

    def test_palindrome(self):
        s = "123454321"
        expected = 0
        self.assertEqual(expected, count_equal_splits(s))

    def test_palindrome2(self):
        s = "1234554321"
        expected = 1
        self.assertEqual(expected, count_equal_splits(s))

    def test_string_with_duplicates(self):
        s = "123123112233"
        # 123-123112233, 1231-23112233, 12312-3112233, 123123-112233, 1231231-12233
        expected = 5
        self.assertEqual(expected, count_equal_splits(s))
    

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 7, 2021 \[Medium\] Minimum Days to Bloom Roses
--- 
> **Question:** Given an array of roses. `roses[i]` means rose `i` will bloom on day `roses[i]`. Also given an int `k`, which is the minimum number of adjacent bloom roses required for a bouquet, and an int `n`, which is the number of bouquets we need. Return the earliest day that we can get `n` bouquets of roses.

**Example:**
```py
Input: roses = [1, 2, 4, 9, 3, 4, 1], k = 2, n = 2
Output: 4
Explanation:
day 1: [b, n, n, n, n, n, b]
The first and the last rose bloom.

day 2: [b, b, n, n, n, n, b]
The second rose blooms. Here the first two bloom roses make a bouquet.

day 3: [b, b, n, n, b, n, b]

day 4: [b, b, b, n, b, b, b]
Here the last three bloom roses make a bouquet, meeting the required n = 2 bouquets of bloom roses. So return day 4.
```

**My thoughts:** Unless rose field cannot produce `n * k` roses, the final answer lies between `1` and `max(roses)`. And if on `x` day we can get expected number of bloom roses then any `day > x` can also be. So we can use the binary search to guess the min day to get target number of rose bunquets. 

**Solution with Binary Search:** [https://replit.com/@trsong/Minimum-Days-to-Bloom-Roses](https://replit.com/@trsong/Minimum-Days-to-Bloom-Roses)
```py
import unittest

def min_day_rose_bloom(roses, n, k):
    if n * k > len(roses):
        return -1
    
    lo = 1
    hi = max(roses)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if can_bloom_rose(roses, mid, n, k):
            hi = mid
        else:
            lo = mid + 1
    return lo


def can_bloom_rose(roses, target, n, k):
    bonquet = 0
    count = 0

    for day in roses:
        if day > target:
            count = 0
            continue
        
        count += 1
        if count >= k:
            count = 0
            bonquet += 1
        
        if bonquet >= n:
            break

    return bonquet >= n


class MinDayRoseBloomSpec(unittest.TestCase):
    def test_example(self):
        n, k, roses = 2, 2, [1, 2, 4, 9, 3, 4, 1]
        # [b, n, n, n, n, n, b]
        # [b, b, n, n, n, n, b]
        # [b, b, n, n, b, n, b]
        # [b, b, b, n, b, b, b]
        expected = 4
        self.assertEqual(expected, min_day_rose_bloom(roses, n, k))

    def test_window_size_one(self):
        n, k, roses = 3, 1, [1, 10, 3, 10, 2]
        # [b, n, n, n, n]
        # [b, n, n, n, b]
        # [b, n, b, n, b]
        expected = 3
        self.assertEqual(expected, min_day_rose_bloom(roses, n, k))

    def test_required_size_greater_than_array(self):
        n, k, roses = 3, 2, [1, 1, 1, 1, 1]
        expected = -1
        self.assertEqual(expected, min_day_rose_bloom(roses, n, k))

    def test_just_meet_required_size(self):
        n, k, roses = 2, 3, [1, 2, 3, 1, 2, 3]
        # [b, n, n, b, n, n]
        # [b, b, n, b, b, n]
        # [b, b, b, b, b, b]
        expected = 3
        self.assertEqual(expected, min_day_rose_bloom(roses, n, k))

    def test_array_with_outlier_number(self):
        n, k, roses = 2, 3, [7, 7, 7, 7, 12, 7, 7]
        expected = 12
        self.assertEqual(expected, min_day_rose_bloom(roses, n, k))

    def test_array_with_extreme_large_number(self):
        n, k, roses = 1, 1, [10000, 9999999]
        expected = 10000
        self.assertEqual(expected, min_day_rose_bloom(roses, n, k))

    def test_continuous_bonquet(self):
        n, k, roses = 4, 2, [1, 10, 2, 9, 3, 8, 4, 7, 5, 6]
        expected = 9
        self.assertEqual(expected, min_day_rose_bloom(roses, n, k))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### May 6, 2021 \[Easy\] Find Corresponding Node in Cloned Tree
--- 
> **Question:** Given two binary trees that are duplicates of one another, and given a node in one tree, find that corresponding node in the second tree. 
> 
> There can be duplicate values in the tree (so comparing node1.value == node2.value isn't going to work).

**Solution with DFS Traversal:** [https://replit.com/@trsong/Find-Corresponding-Node-in-Cloned-Tree-2](https://replit.com/@trsong/Find-Corresponding-Node-in-Cloned-Tree-2)
```py
from copy import deepcopy
import unittest

def find_node(root1, root2, node1):
    if root1 is None or root2 is None or node1 is None:
        return None
    
    traversal1 = dfs_traversal(root1)
    traversal2 = dfs_traversal(root2)

    for n1, n2 in zip(traversal1, traversal2):
        if n1 == node1:
            return n2
    return None


def dfs_traversal(root):
    stack = [root]
    while stack:
        cur = stack.pop()
        yield cur
        for child in [cur.left, cur.right]:
            if child is None:
                continue
            stack.append(child)


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return "TreeNode(%d, %s, %s)" % (self.val, self.left, self.right)


class FindNodeSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertIsNone(find_node(None, None, None))

    def test_one_node_tree(self):
        root1 = TreeNode(1)
        root2 = deepcopy(root1)
        self.assertEqual(root2, find_node(root1, root2, root1))

    def test_leaf_node(self):
        """
          1
         / \
        2   3
             \
              4
        """
        root1 = TreeNode(1, TreeNode(2), TreeNode(3, right=TreeNode(4)))
        root2 = deepcopy(root1)
        f = lambda root: root.right.right
        self.assertEqual(f(root2), find_node(root1, root2, f(root1)))

    def test_internal_node(self):
        """
            1
           / \
          2   3
         /   /
        0   1
        """
        left_tree = TreeNode(2, TreeNode(0))
        right_tree = TreeNode(3, TreeNode(1))
        root1 = TreeNode(1, left_tree, right_tree)
        root2 = deepcopy(root1)
        f = lambda root: root.left
        self.assertEqual(f(root2), find_node(root1, root2, f(root1))) 

    def test_duplicated_value_in_tree(self):
        """
          1
           \
            1
           /
          1
         /
        1
        """
        root1 = TreeNode(1, right=TreeNode(1, TreeNode(1, TreeNode(1))))
        root2 = deepcopy(root1)
        f = lambda root: root.right.left
        self.assertEqual(f(root2), find_node(root1, root2, f(root1))) 
    
    def test_find_root_node(self):
        """
          1
         / \
        2   3
        """
        root1 = TreeNode(1, TreeNode(2), TreeNode(3))
        root2 = deepcopy(root1)
        self.assertEqual(root2, find_node(root1, root2, root1))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 5, 2021 LC 93 \[Medium\] All Possible Valid IP Address Combinations
---
> **Question:** Given a string of digits, generate all possible valid IP address combinations.
>
> IP addresses must follow the format A.B.C.D, where A, B, C, and D are numbers between `0` and `255`. Zero-prefixed numbers, such as `01` and `065`, are not allowed, except for `0` itself.
>
> For example, given `"2542540123"`, you should return `['254.25.40.123', '254.254.0.123']`

**Solution with Backtracking:** [https://replit.com/@trsong/Find-All-Possible-Valid-IP-Address-Combinations-3](https://replit.com/@trsong/Find-All-Possible-Valid-IP-Address-Combinations-3)
```py
import unittest

def all_ip_combinations(raw_str):
    res = []
    accu = []
    n = len(raw_str)

    def backtrack(i):
        if len(accu) > 4:
            return
    
        if i == n and len(accu) == 4:
            res.append('.'.join(accu))
        else:
            for num_digit in [1, 2, 3]:
                if i + num_digit > n: 
                    break
                num = int(raw_str[i: i + num_digit])
                
                # From 0 to 9
                case1 = num_digit == 1

                # From 10 to 99
                case2 = case1 or num_digit == 2 and num >= 10

                # From 100 to 255
                case3 = case2 or num_digit == 3 and 100 <= num <= 255
                if case3:
                    accu.append(str(num))
                    backtrack(i + num_digit)
                    accu.pop()
    
    backtrack(0)
    return res
                    

class AllIpCombinationSpec(unittest.TestCase):
    def test_example(self):
        raw_str = '2542540123'
        expected = ['254.25.40.123', '254.254.0.123']
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_empty_string(self):
        self.assertItemsEqual([], all_ip_combinations(''))

    def test_no_valid_ips(self):
        raw_str = '25505011535'
        expected = []
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_multiple_outcomes(self):
        raw_str = '25525511135'
        expected = ['255.255.11.135', '255.255.111.35']
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_multiple_outcomes2(self):
        raw_str = '25011255255'
        expected = ['250.112.55.255', '250.11.255.255']
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_multiple_outcomes3(self):
        raw_str = '10101010'
        expected = [
            '10.10.10.10', '10.10.101.0', '10.101.0.10', '101.0.10.10',
            '101.0.101.0'
        ]
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_multiple_outcomes4(self):
        raw_str = '01010101'
        expected = ['0.10.10.101', '0.101.0.101']
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_unique_outcome(self):
        raw_str = '111111111111'
        expected = ['111.111.111.111']
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_unique_outcome2(self):
        raw_str = '0000'
        expected = ['0.0.0.0']
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_missing_parts(self):
        raw_str = '000'
        expected = []
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 4, 2021 \[Medium\] In-place Array Rotation
---
> **Question:** Write a function that rotates an array by `k` elements.
>
> For example, `[1, 2, 3, 4, 5, 6]` rotated by two becomes `[3, 4, 5, 6, 1, 2]`.
>
> Try solving this without creating a copy of the array. How many swap or move operations do you need?

**Solution:** [https://replit.com/@trsong/Rotate-Array-In-place-2](https://replit.com/@trsong/Rotate-Array-In-place-2)
```py
import unittest

def rotate(nums, k):
    if not nums:
        return []
        
    n = len(nums)
    k %= n
    reverse(nums, 0, k - 1)
    reverse(nums, k , n - 1)
    reverse(nums, 0, n - 1)
    return nums


def reverse(nums, start, end):
    while start < end:
        nums[start], nums[end] = nums[end], nums[start]
        start += 1
        end -= 1


class RotateSpec(unittest.TestCase):
    def test_example(self):
        k, nums = 2, [1, 2, 3, 4, 5, 6]
        expected = [3, 4, 5, 6, 1, 2]
        self.assertEqual(expected, rotate(nums, k))

    def test_rotate_0_position(self):
        k, nums = 0, [0, 1, 2, 3]
        expected = [0, 1, 2, 3]
        self.assertEqual(expected, rotate(nums, k))

    def test_empty_array(self):
        self.assertEqual([], rotate([], k=10))

    def test_shift_negative_position(self):
        k, nums = -1, [0, 1, 2, 3]
        expected = [3, 0, 1, 2]
        self.assertEqual(expected, rotate(nums, k))

    def test_shift_more_than_array_size(self):
        k, nums = 8,  [1, 2, 3, 4, 5, 6]
        expected = [3, 4, 5, 6, 1, 2]
        self.assertEqual(expected, rotate(nums, k))

    def test_multiple_round_of_forward_and_backward_shift(self):
        k, nums = 5, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        expected = [5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4]
        self.assertEqual(expected, rotate(nums, k))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 3, 2021 \[Easy\] Find Duplicates
---
> **Question:** Given an array of size n, and all values in the array are in the range 1 to n, find all the duplicates.

**Example:**
```py
Input: [4, 3, 2, 7, 8, 2, 3, 1]
Output: [2, 3]
```

**Solution:** [https://replit.com/@trsong/Find-Duplicates](https://replit.com/@trsong/Find-Duplicates)
```py
import unittest

def find_duplicates(nums):
    res = []
    for num in nums:
        index = abs(num) - 1
        if nums[index] < 0:
            res.append(abs(num))
        else:
            nums[index] *= - 1

    for i in range(len(nums)):
        nums[i] = abs(nums[i])
    return res


class FindDuplicateSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(sorted(expected), sorted(result))

    def test_example1(self):
        input = [4, 3, 2, 7, 8, 2, 3, 1]
        expected = [2, 3]
        self.assert_result(expected, find_duplicates(input))

    def test_example2(self):
        input = [4, 5, 2, 6, 8, 2, 1, 5]
        expected = [2, 5]
        self.assert_result(expected, find_duplicates(input))

    def test_empty_array(self):
        self.assertEqual([], find_duplicates([]))

    def test_no_duplicated_numbers(self):
        input = [6, 1, 4, 3, 2, 5]
        expected = []
        self.assert_result(expected, find_duplicates(input))

    def test_duplicated_number(self):
        input = [1, 1, 2]
        expected = [1]
        self.assert_result(expected, find_duplicates(input))

    def test_duplicated_number2(self):
        input = [1, 1, 3, 5, 6, 8, 8, 1, 1]
        expected = [1, 8, 1, 1]
        self.assert_result(expected, find_duplicates(input))
  
    def test_duplicated_number3(self):
        input = [1, 3, 3]
        expected = [3]
        self.assert_result(expected, find_duplicates(input))
    
    def test_duplicated_number4(self):
        input = [3, 2, 3, 2, 3, 2, 7]
        expected = [3, 2, 3, 2]
        self.assert_result(expected, find_duplicates(input))

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 2, 2021 \[Hard\] Increasing Subsequence of Length K
---
> **Question:** Given an int array nums of length n and an int k. Return an increasing subsequence of length k (KIS). Expected time complexity `O(nlogk)`.

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

**Solution with DP and Binary Search:** [https://replit.com/@trsong/Increasing-Subsequence-of-Length-K-2](https://replit.com/@trsong/Increasing-Subsequence-of-Length-K-2)
```py
import unittest

def increasing_sequence(nums, k):
    if not nums:
        return []

    # Let dp[i] represents i-th element in a length i + 1 length subsequence
    dp = []
    prev_elem = {}

    for num in nums:
        insert_pos = binary_search(dp, num)
        if insert_pos == len(dp):
            dp.append(num)
        else:
            dp[insert_pos] = num
        prev_elem[num] = dp[insert_pos - 1] if insert_pos > 0 else None

        if len(dp) == k:
            break

    res = []
    num = dp[-1]
    while num is not None:
        res.append(num)
        num = prev_elem[num]
    return res[::-1]


def binary_search(dp, target):
    lo = 0
    hi = len(dp)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if dp[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


class IncreasingSequenceSpec(unittest.TestCase):
    def validate_result(self, nums, k):
        subseq = increasing_sequence(nums, k)
        self.assertEqual(k, len(subseq), str(subseq) + " Is not of length " + str(k))

        i = 0
        for num in nums:
            if i < len(subseq) and num == subseq[i]:
                i += 1
        self.assertEqual(len(subseq), i, str(subseq) + " Is not valid subsequence.")
        
        for i in xrange(1, len(subseq)):
            self.assertLessEqual(subseq[i-1], subseq[i], str(subseq) + " Is not increasing subsequene.")

    def test_example(self):
        k, nums = 3, [10, 1, 4, 8, 2, 9]
        self.validate_result(nums, k)  # possible result: [1, 4, 8]

    def test_example2(self):
        k, nums = 4, [10, 1, 4, 8, 2, 9]
        self.validate_result(nums, k)  # possible result: [1, 4, 8, 9]

    def test_empty_sequence(self):
        k, nums = 0, []
        self.validate_result(nums, k)

    def test_longest_increasing_subsequence(self):
        k, nums = 4, [10, 9, 2, 5, 3, 7, 101, 18]
        self.validate_result(nums, k)  # possible result: [2, 3, 7, 101]

    def test_longest_increasing_subsequence_in_second_half_sequence(self):
        k, nums = 4, [1, 2, 3, -2, -1, 0, 1]
        self.validate_result(nums, k)  # possible result: [-2, -1, 0, 1]

    def test_should_return_valid_subsequene(self):
        k, nums = 3, [8, 9, 7, 6, 10]
        self.validate_result(nums, k)  # possible result: [8, 9, 10]


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### May 1, 2021 \[Hard\] Decreasing Subsequences
---
> **Question:** Given an int array nums of length n. Split it into strictly decreasing subsequences. Output the min number of subsequences you can get by splitting.

**Example 1:**
```py
Input: [5, 2, 4, 3, 1, 6]
Output: 3
Explanation:
You can split this array into: [5, 2, 1], [4, 3], [6]. And there are 3 subsequences you get.
Or you can split it into [5, 4, 3], [2, 1], [6]. Also 3 subsequences.
But [5, 4, 3, 2, 1], [6] is not legal because [5, 4, 3, 2, 1] is not a subsuquence of the original array.
```

**Example 2:**
```py
Input: [2, 9, 12, 13, 4, 7, 6, 5, 10]
Output: 4
Explanation: [2], [9, 4], [12, 10], [13, 7, 6, 5]
```

**Example 3:**
```py
Input: [1, 1, 1]
Output: 3
Explanation: Because of the strictly descending order you have to split it into 3 subsequences: [1], [1], [1]
```

**My thoughts:** This question is equivalent to [Longest Increasing Subsequence](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-19-2020-lc-300-hard-the-longest-increasing-subsequence). Can be solved with greedy approach.

Imagine we are to create a list of stacks each in descending order (stack top is smallest). And those stacks are sorted by each stack's top element. 

Then for each element from input sequence, we just need to figure out (using binary search) the stack such that by pushing this element into stack, result won't affect the order of stacks and decending property of each stack. 

Finally, the total number of stacks equal to min number of subsequence we can get by splitting. Each stack represents a decreasing subsequence.

**Greedy Solution with Descending Stack and Binary Search:** [https://replit.com/@trsong/Decreasing-Subsequences](https://replit.com/@trsong/Decreasing-Subsequences)
```py
import unittest

def min_decreasing_subsequences(sequence):
    # maintain a list of descending stacks sorted by each stack top 
    stack_list = []
    for num in sequence:
        stack_index = binary_search_stack_top(stack_list, num)
        if stack_index == len(stack_list):
            stack_list.append([])
        stack_list[stack_index].append(num)
    return len(stack_list)


def binary_search_stack_top(stack_list, target):
    lo = 0
    hi = len(stack_list)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        stack_top = stack_list[mid][-1]
        if stack_top <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


class MinDecreasingSubsequnceSpec(unittest.TestCase):
    def test_example(self):
        sequence = [5, 2, 4, 3, 1, 6]
        # [5, 2, 1]
        # [4, 3]
        # [6]
        expected = 3
        self.assertEqual(expected, min_decreasing_subsequences(sequence))

    def test_example2(self):
        sequence = [2, 9, 12, 13, 4, 7, 6, 5, 10]
        # [2]
        # [9, 4]
        # [12, 7, 6, 5]
        # [13, 10]
        expected = 4
        self.assertEqual(expected, min_decreasing_subsequences(sequence))

    def test_example3(self):
        sequence = [1, 1, 1]
        # [1]
        # [1]
        # [1]
        expected = 3
        self.assertEqual(expected, min_decreasing_subsequences(sequence))

    def test_empty_sequence(self):
        self.assertEqual(0, min_decreasing_subsequences([]))

    def test_last_elem_is_local_max(self):
        seq = [1, 2, 3, 0, 2]
        # [1, 0]
        # [2]
        # [3, 2]
        expected = 3 
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_last_elem_is_global_max(self):
        seq = [1, 2, 3, 0, 6]
        # [1, 0]
        # [2]
        # [3]
        # [6]
        expected = 4
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_min_decreasing_subsequences_in_first_half_sequence(self):
        seq = [4, 5, 6, 7, 1, 2, 3]
        # [4, 1]
        # [5, 2]
        # [6, 3]
        # [7]
        expected = 4 
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_min_decreasing_subsequences_in_second_half_sequence(self):
        seq = [1, 2, 3, -2, -1, 0, 1]
        # [1, -2]
        # [2, -1]
        # [3, 0]
        # [1]
        expected = 4
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_sequence_in_up_down_up_pattern(self):
        seq = [1, 2, 3, 2, 4]
        # [1]
        # [2]
        # [3, 2]
        # [4]
        expected = 4 
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_sequence_in_up_down_up_pattern2(self):
        seq = [1, 2, 3, -1, 0]
        # [1, -1, 0]
        # [2]
        # [3]
        expected = 3 
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_sequence_in_down_up_down_pattern(self):
        seq = [4, 3, 5]
        # [4, 3]
        # [5]
        expected = 2
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_sequence_in_down_up_down_pattern2(self):
        seq = [4, 0, 1]
        # [4, 0]
        # [1]
        expected = 2
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_multiple_result(self):
        seq = [10, 9, 2, 5, 3, 7, 101, 18]
        # [10, 9, 2]
        # [5, 3]
        # [7]
        # [101, 18]
        expected = 4 
        self.assertEqual(expected, min_decreasing_subsequences(seq))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```