---
layout: post
title:  "Daily Coding Problems 2020 Nov to Jan"
date:   2020-11-01 22:22:32 -0700
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


### Jan 27, 2021 LC 89 \[Medium\] Generate Gray Code
---
> **Question:**  Gray code is a binary code where each successive value differ in only one bit, as well as when wrapping around. Gray code is common in hardware so that we don't see temporary spurious values during transitions.
>
> Given a number of bits n, generate a possible gray code for it.

**Example:**

```py
For n = 2, one gray code would be [00, 01, 11, 10].
```

**My thoughts:** Test Grey Code under different size. Try to find the pattern:
- For n = 0, [0]
- For n = 1, [00, 01]
- For n = 2, [00, 01, 11, 10]
- For n = 3, [000, 001, 011, 010, 110, 111, 101, 100]

Notice that `00 => 000, 001`.   `01 => 011, 010`.  `11 => 110, 111`. So the pattern is if original value is of even index, append 0 and then append 1. Otherwise if it's on odd index, append 1 then append 0. And we do that for all elements.

**Solution:** [https://repl.it/@trsong/Generate-the-Gray-Code](https://repl.it/@trsong/Generate-the-Gray-Code)
```py
import unittest

def gray_code(n):
    """
                      0
                  /       \
              0                 1
            /   \              /  \
         00       01        11       10
       /  \      /  \      /  \     /  \
     000  001  011  010  110  111  101  100
    """
    res = [0] * (2 ** n)
    for step in xrange(n):
        for i in xrange(2 ** step - 1, -1, -1):
            prev = res[i]
            prev_append_zero = prev << 1
            prev_append_one = prev_append_zero + 1 
            if i % 2 == 0:
                res[2 * i] = prev_append_zero
                res[2 * i + 1] = prev_append_one
            else:
                res[2 * i] = prev_append_one
                res[2 * i + 1] = prev_append_zero
    return res


class GrayCodeSpec(unittest.TestCase):
    def validate_grey_code(self, code_arr):
        for i in xrange(1, len(code_arr)):
            cur = code_arr[i]
            prev = code_arr[i-1]
            xor_val = cur ^ prev
            if xor_val & (xor_val - 1) != 0:
                # make sure the current value is 1 bit different from the previous value
                return False
        return True

    def test_zero_bit_grey_code(self):
        self.assertTrue(self.validate_grey_code(gray_code(0)))

    def test_one_bit_grey_code(self):
        self.assertTrue(self.validate_grey_code(gray_code(1)))

    def test_two_bits_grey_code(self):
        self.assertTrue(self.validate_grey_code(gray_code(2)))

    def test_three_bits_grey_code(self):
        self.assertTrue(self.validate_grey_code(gray_code(3)))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Jan 26, 2021 \[Easy\] Rotate Linked List
---
> **Question:** Given a linked list and a positive integer `k`, rotate the list to the right by k places.
>
> For example, given the linked list `7 -> 7 -> 3 -> 5` and `k = 2`, it should become `3 -> 5 -> 7 -> 7`.
>
> Given the linked list `1 -> 2 -> 3 -> 4 -> 5` and `k = 3`, it should become `3 -> 4 -> 5 -> 1 -> 2`.

**Solution with Fast-slow Pointers:** [https://repl.it/@trsong/Rotate-Singly-Linked-List](https://repl.it/@trsong/Rotate-Singly-Linked-List)
```py
import unittest

def rotate_list(head, k):
    if not head:
        return None

    n = list_len(head)
    k %= n
    if k == 0:
        return head

    fast = slow = head
    for _ in xrange(k):
        fast = fast.next

    while fast.next:
        fast = fast.next
        slow = slow.next
    
    res = slow.next
    fast.next = head
    slow.next = None
    return res


def list_len(lst):
    res = 0
    while lst:
        lst = lst.next
        res += 1
    return res 


class Node(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    def __eq__(self, other):
        return other and self.val == other.val and self.next == other.next

    def __repr__(self):
        return "%d -> %s" % (self.val, self.next)

    @staticmethod
    def List(*vals):
        p = dummy = Node(-1)
        for val in vals:
            p.next = Node(val)
            p = p.next
        return dummy.next


class RotateListSpec(unittest.TestCase):
    def test_example(self):
        k, lst = 2, Node.List(1, 2, 3, 4)
        expected = Node.List(3, 4, 1, 2)
        self.assertEqual(expected, rotate_list(lst, k))

    def test_empty_list(self):
        self.assertIsNone(rotate_list(None, 42))

    def test_single_elem_list(self):
        k, lst = 42, Node.List(1)
        expected = Node.List(1)
        self.assertEqual(expected, rotate_list(lst, k))

    def test_k_greater_than_length_of_list(self):
        k, lst = 5, Node.List(1, 2)
        expected = Node.List(2, 1)
        self.assertEqual(expected, rotate_list(lst, k))

    def test_k_equal_length_of_list(self):
        k, lst = 3, Node.List(1, 2, 3)
        expected = Node.List(1, 2, 3)
        self.assertEqual(expected, rotate_list(lst, k))

    def test_k_less_than_length_of_list(self):
        k, lst = 5, Node.List(0, 1, 2, 3, 4, 5)
        expected = Node.List(1, 2, 3, 4, 5, 0)
        self.assertEqual(expected, rotate_list(lst, k))

    def test_array_contains_duplicates(self):
        k, lst = 1, Node.List(1, 2, 2, 3, 3, 3, 4)
        expected = Node.List(4, 1, 2, 2, 3, 3, 3)
        self.assertEqual(expected, rotate_list(lst, k))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Jan 25, 2021 \[Hard\] Partition Array to Reach Mininum Difference
---
> **Question:** Given an array of positive integers, divide the array into two subsets such that the difference between the sum of the subsets is as small as possible.
>
> For example, given `[5, 10, 15, 20, 25]`, return the sets `[10, 25]` and `[5, 15, 20]`, which has a difference of `5`, which is the smallest possible difference.


**Solution with DP:** [https://repl.it/@trsong/Partition-Array-to-Reach-Min-Difference](https://repl.it/@trsong/Partition-Array-to-Reach-Min-Difference)
```py
import unittest

def min_partition_difference(nums):
    nums_sum = sum(nums)
    n = len(nums)

    # Let dp[i][sum] represents exists subset sum for nums[:i]
    # dp[i][sum] = dp[i-1][sum]              if exclude nums[i - 1]
    #            = dp[i-1][sum - nums[i-1]]  if include nums[i - 1]
    dp = [[False for _ in xrange(nums_sum // 2 + 1)] for _ in xrange(n + 1)]
    for i in xrange(n + 1):
        dp[i][0] = True

    for i in xrange(1, n + 1):
        elem = nums[i-1]
        for s in xrange(1, nums_sum // 2 + 1):
            # Exclude current element, as we can already reach sum s
            dp[i][s] = dp[i - 1][s]

            if s -  elem >= 0:
                # As all number are positive, include current elem cannot exceed current sum
                dp[i][s] = dp[i - 1][s - elem]
    
    """
     Let's do some math here:
     Let s1, s2 be the size to two subsets after partition and assume s1 >= s2
     We can have s1 + s2 = sum_nums and we want to get min{s1 - s2} where s1 >= s2:

     min{s1 - s2}
     = min{s1 + s2 - s2 - s2}
     = min{sum_nums - 2 * s2}  in this step sum_nums - 2 * s2 >=0, gives s2 <= sum_nums/ 2
     = sum_nums - 2 * max{s2}  where s2 <= sum_nums/2
    """
    for s in xrange(nums_sum // 2, -1, -1):
        if dp[n][s]:
            return nums_sum - 2 * s

    return None


class MinPartitionDifferenceSpec(unittest.TestCase):
    def test_example(self):
        # Partition: [10, 25] and [5, 15, 20]
        self.assertEqual(5, min_partition_difference([5, 10, 15, 20, 25]))

    def test_empty_array(self):
        self.assertEqual(0, min_partition_difference([]))

    def test_array_with_one_element(self):
        self.assertEqual(42, min_partition_difference([42]))

    def test_array_with_two_elements(self):
        self.assertEqual(0, min_partition_difference([42, 42]))

    def test_unsorted_array_with_duplicated_numbers(self):
        # Partition: [3, 4] and [1, 2, 2, 1]
        self.assertEqual(1, min_partition_difference([3, 1, 4, 2, 2, 1]))

    def test_unsorted_array_with_unique_numbers(self):
        # Partition: [11] and [1, 5, 6]
        self.assertEqual(1, min_partition_difference([1, 6, 11, 5]))

    def test_sorted_array_with_duplicated_numbers(self):
        # Partition: [1, 2, 2] and [4]
        self.assertEqual(1, min_partition_difference([1, 2, 2, 4]))

    def test_min_partition_difference_is_zero(self):
        # Partition: [1, 8, 2, 7] and [3, 6, 4, 5]
        self.assertEqual(0, min_partition_difference([1, 2, 3, 4, 5, 6, 7, 8]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 24, 2021 \[Hard\] Smallest Stab Set
--- 
> **Question:** Let `X` be a set of `n` intervals on the real line. We say that a set of points `P` "stabs" `X` if every interval in `X` contains at least one point in `P`. Compute the smallest set of points that stabs X.
>
> For example, given the intervals `[(1, 4), (4, 5), (7, 9), (9, 12)]`, you should return `[4, 9]`.
 

**My thoughts:** Sort intevals by end time. Greedily choose earlist end time of a interval. Skip all intervals that covers that end time unless not covered, then move onto earilst end time of next interval. 


**Solution with Greedy Approach:** [https://repl.it/@trsong/Smallest-Stab-Set](https://repl.it/@trsong/Smallest-Stab-Set)
```py
import unittest

def smallest_stub_set(intervals):
    intervals.sort(key=lambda interval: (interval[1], interval[0]))

    smallest_end = None
    res = []
    for start, end in intervals:
        if start > smallest_end:
            res.append(end)
            smallest_end = end
    return res


class SmallestStubSetSpec(unittest.TestCase):
    def assert_result(self, expected, intervals):
        res = smallest_stub_set(intervals)
        self.assertEqual(len(expected), len(res), "Expected result like {} but given {}".format(expected, res))
        for num in res:
            self.assertTrue(any(lo <= num <= hi for lo, hi in intervals))

    def test_example(self):
        intervals = [(1, 4), (4, 5), (7, 9), (9, 12)]
        expected = [4, 9]
        self.assert_result(expected, intervals)

    def test_empty_intervals(self):
        self.assertEqual([], smallest_stub_set([]))

    def test_optimal_result(self):
        intervals = [(1, 5), (5, 10), (2, 9)]
        expected = [5]
        self.assert_result(expected, intervals)

    def test_optimal_result2(self):
        intervals = [(2, 8), (8, 15), (1, 5), (5, 10), (10, 20)]
        expected = [5, 10]
        self.assert_result(expected, intervals)

    def test_have_to_include_all_ends(self):
        intervals = [(2, 2), (8, 8), (1, 1), (5, 5), (2, 2)]
        expected = [1, 2, 5, 8]
        self.assert_result(expected, intervals)

    def test_multiple_solutions(self):
        intervals = [(1, 2), (1, 3), (1, 4), (3, 5)]
        expected = [2, 5]  # other solution also work
        self.assert_result(expected, intervals)
    
    def test_multiple_solutions2(self):
        intervals = [(1, 5), (2, 5), (3, 5), (4, 10)]
        expected = [5]  # other solution also work
        self.assert_result(expected, intervals)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```



### Jan 23, 2021 \[Easy\] Valid UTF-8 Encoding
--- 
> **Question:** UTF-8 is a character encoding that maps each symbol to one, two, three, or four bytes.
>
> Write a program that takes in an array of integers representing byte values, and returns whether it is a valid UTF-8 encoding.
> 
> For example, the Euro sign, €, corresponds to the three bytes 11100010 10000010 10101100. The rules for mapping characters are as follows:
>
> - For a single-byte character, the first bit must be zero.
> - For an n-byte character, the first byte starts with n ones and a zero. The other n - 1 bytes all start with 10.
> Visually, this can be represented as follows.

```py
 Bytes   |           Byte format
-----------------------------------------------
   1     | 0xxxxxxx
   2     | 110xxxxx 10xxxxxx
   3     | 1110xxxx 10xxxxxx 10xxxxxx
   4     | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
```   


**Solution:** [https://repl.it/@trsong/Valid-UTF-8-Encoding](https://repl.it/@trsong/Valid-UTF-8-Encoding)
```py
import unittest

MAX_BYTE = 0b11111111
FOLLOWING_BYTE_PREFIX = 0b10


def validate_utf8(byte_arr):
    if not byte_arr:
        return False
    
    first_byte = byte_arr[0]
    if len(byte_arr) == 1:
        return first_byte >> 7 == 0
    
    if not validate_byte_value(byte_arr):
        return False

    num_bytes = count_prefix_ones(first_byte)
    if num_bytes != len(byte_arr):
        return False
    
    return validate_following_bytes(byte_arr[1:])


def validate_byte_value(byte_arr):
    return all(byte <= MAX_BYTE for byte in byte_arr)


def count_prefix_ones(byte):
    count = 0
    for i in xrange(7, -1, -1):
        if byte & 1 << i == 0:
            break
        count += 1
    return count


def validate_following_bytes(byte_arr):
    return all(FOLLOWING_BYTE_PREFIX == (byte >> 6) for byte in byte_arr)


class ValidateUtf8Spec(unittest.TestCase):
    def test_example(self):
        byte_arr = [0b11100010, 0b10000010, 0b10101100]
        self.assertTrue(validate_utf8(byte_arr))

    def test_empty_arr(self):
        byte_arr = []
        self.assertFalse(validate_utf8(byte_arr))

    def test_valid_one_byte(self):
        byte_arr = [0b0]
        self.assertTrue(validate_utf8(byte_arr))

    def test_valid_one_byte2(self):
        byte_arr = [0b01111111]
        self.assertTrue(validate_utf8(byte_arr))

    def test_invalid_one_byte(self):
        byte_arr = [0b10101010]
        self.assertFalse(validate_utf8(byte_arr))

    def test_invalid_one_byte2(self):
        byte_arr = [0b0, 0b10000010]
        self.assertFalse(validate_utf8(byte_arr))

    def test_valid_two_bytes(self):
        byte_arr = [0b11000000, 0b10000000]
        self.assertTrue(validate_utf8(byte_arr))

    def test_valid_two_bytes2(self):
        byte_arr = [0b11011111, 0b10111111]
        self.assertTrue(validate_utf8(byte_arr))

    def test_valid_three_bytes(self):
        byte_arr = [0b11100000, 0b10000000, 0b10000000]
        self.assertTrue(validate_utf8(byte_arr))

    def test_valid_three_bytes2(self):
        byte_arr = [0b11101111, 0b10111111, 0b10111111]
        self.assertTrue(validate_utf8(byte_arr))

    def test_valid_four_bytes(self):
        byte_arr = [0b11110000, 0b10000000, 0b10000000, 0b10000000]
        self.assertTrue(validate_utf8(byte_arr))

    def test_valid_four_bytes2(self):
        byte_arr = [0b11110111, 0b10111111, 0b10111111, 0b10111111]
        self.assertTrue(validate_utf8(byte_arr))

    def test_short_on_bytes(self):
        byte_arr = [0b11110111, 0b10111111, 0b10111111]
        self.assertFalse(validate_utf8(byte_arr))

    def test_more_than_necessary_bytes(self):
        byte_arr = [0b11000111, 0b10111111, 0b10111111]
        self.assertFalse(validate_utf8(byte_arr))
    
    def test_following_bytes_with_incorrect_prefix(self):
        byte_arr = [0b11100000, 0b11000000, 0b11000000]
        self.assertFalse(validate_utf8(byte_arr))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Jan 22, 2021 \[Easy\] Palindrome Integers
--- 
> **Question:** Write a program that checks whether an integer is a palindrome. For example, `121` is a palindrome, as well as `888`. But neither `678` nor `80` is a palindrome. Do not convert the integer into a string.

**Solution:** [https://repl.it/@trsong/Determine-Palindrome-Integer](https://repl.it/@trsong/Determine-Palindrome-Integer)
```py
import unittest

def is_palindrome_integer(num):
    return num >= 0 and num == reverse_number(num)


def reverse_number(num):
    res = 0
    while num > 0:
        res = 10 * res + num % 10
        num //= 10
    return res


class IsPalindromeIntegerSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(is_palindrome_integer(121))

    def test_example2(self):
        self.assertTrue(is_palindrome_integer(88))

    def test_example3(self):
        self.assertFalse(is_palindrome_integer(678))

    def test_example4(self):
        self.assertFalse(is_palindrome_integer(80))

    def test_zero(self):
        self.assertTrue(is_palindrome_integer(0))

    def test_single_digit_number(self):
        self.assertTrue(is_palindrome_integer(7))

    def test_non_palindrome1(self):
        self.assertFalse(is_palindrome_integer(123421))

    def test_non_palindrome2(self):
        self.assertFalse(is_palindrome_integer(21010112))

    def test_negative_number1(self):
        self.assertFalse(is_palindrome_integer(-1))

    def test_negative_number2(self):
        self.assertFalse(is_palindrome_integer(-222))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 21, 2021 \[Easy\] Power Set
---
> **Question:** The power set of a set is the set of all its subsets. Write a function that, given a set, generates its power set.
>
> For example, given a set represented by a list `[1, 2, 3]`, it should return `[[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]` representing the power set.

**My thoughts:** There are multiple ways to solve this problem. My solution recursively subsets incrementally. 

Let's calculate the first few terms and try to figure out the pattern
```py
power_set([]) => [[]]
power_set([1]) => [[], [1]]
power_set([1, 2]) => [[], [1], [2], [1, 2]]
power_set([1, 2, 3]) => [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]] which is power_set([1, 2]) + append powerSet([1, 2]) with new elem 3
...
power_set([1, 2, ..., n]) =>  power_set([1, 2, ..., n - 1]) + append powerSet([1, 2, ..., n - 1]) with new elem n
```

**Solution:** [https://repl.it/@trsong/Power-Set](https://repl.it/@trsong/Power-Set)
```py
import unittest

def generate_power_set(nums):
    res = [[]]
    for num in nums:
        include = []
        for existing_set in res:
            include.append(existing_set + [num])
        res.extend(include)
    return res
            

class GeneratePowerSetSpec(unittest.TestCase):
    def test_example(self):
        nums = [1, 2, 3]
        expected = [[], [1], [2], [3], [1, 2], [2, 3], [1, 3], [1, 2, 3]]
        self.assertItemsEqual(expected, generate_power_set(nums))

    def test_empty_set(self):
        nums = []
        expected = [[]]
        self.assertItemsEqual(expected, generate_power_set(nums))

    def test_one_elem_set(self):
        nums = [1]
        expected = [[], [1]]
        self.assertItemsEqual(expected, generate_power_set(nums))

    def test_two_elem_set(self):
        nums = [1, 2]
        expected = [[], [1], [2], [1, 2]]
        self.assertItemsEqual(expected, generate_power_set(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 20, 2021 \[Medium\] Most Frequent Subtree Sum
--- 
> **Question:** Given a binary tree, find the most frequent subtree sum.
> 
> If there is a tie between the most frequent sum, return the smaller one.

**Example:**
```
   3
  / \
 1   -3

The above tree has 3 subtrees.:
The root node with 3, and the 2 leaf nodes, which gives us a total of 3 subtree sums.
The root node has a sum of 1 (3 + 1 + -3).
The left leaf node has a sum of 1, and the right leaf node has a sum of -3. 
Therefore the most frequent subtree sum is 1.
```

**Solution:** [https://repl.it/@trsong/Find-Most-Frequent-Subtree-Sum](https://repl.it/@trsong/Find-Most-Frequent-Subtree-Sum)
```py
import unittest

def most_freq_tree_sum(tree):
    if not tree:
        return None

    freq = {}
    node_sum(tree, freq)
    return max(freq.keys(), key=lambda k: (freq[k], -k))


def node_sum(node, freq):
    if not node:
        return 0
    
    left_res = node_sum(node.left, freq)
    right_res = node_sum(node.right, freq)
    res = node.val + left_res + right_res
    freq[res] = freq.get(res, 0) + 1
    return res


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class MostFreqTreeSumSpec(unittest.TestCase):
    def test_example1(self):
        """
           3
          / \
         2  -3
        """
        t = TreeNode(3, TreeNode(2), TreeNode(-3))
        self.assertEqual(2, most_freq_tree_sum(t))

    def test_empty_tree(self):
        self.assertIsNone(most_freq_tree_sum(None))

    def test_tree_with_unique_value(self):
        """
          0
         / \
        0   0
         \
          0
        """
        l = TreeNode(0, right=TreeNode(0))
        t = TreeNode(0, l, TreeNode(0))
        self.assertEqual(0, most_freq_tree_sum(t))

    def test_depth_3_tree(self):
        """
           _0_ 
          /   \
         0     -3  
        / \   /  \   
       1  -1 3   -1  
        """
        l = TreeNode(0, TreeNode(1), TreeNode(-1))
        r = TreeNode(-3, TreeNode(3), TreeNode(-1))
        t = TreeNode(0, l, r)
        self.assertEqual(-1, most_freq_tree_sum(t))

    def test_return_smaller_freq_when_there_is_a_tie(self):
        """
            -2
           /   \
          0     1
         / \   / \
        0   0 1   1
        """
        l = TreeNode(0, TreeNode(0), TreeNode(0))
        r = TreeNode(1, TreeNode(1), TreeNode(1))
        t = TreeNode(-2, l, r)
        self.assertEqual(0, most_freq_tree_sum(t))

    def test_return_smaller_freq_when_there_is_a_tie2(self):
        """
           3
          / \
         2   -1
        """
        t = TreeNode(3, TreeNode(2), TreeNode(-1))
        self.assertEqual(-1, most_freq_tree_sum(t))
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Jan 19, 2021 \[Medium\] Friend Cycle Problem
--- 
> **Question:** A classroom consists of `N` students, whose friendships can be represented in an adjacency list. For example, the following descibes a situation where `0` is friends with `1` and `2`, `3` is friends with `6`, and so on.

```py
{0: [1, 2],
 1: [0, 5],
 2: [0],
 3: [6],
 4: [],
 5: [1],
 6: [3]} 
```
> Each student can be placed in a friend group, which can be defined as the transitive closure of that student's friendship relations. In other words, this is the smallest set such that no student in the group has any friends outside this group. For the example above, the friend groups would be `{0, 1, 2, 5}`, `{3, 6}`, `{4}`.
>
> Given a friendship list such as the one above, determine the number of friend groups in the class.


**Solution with DisjointSet(Union-Find):** [https://repl.it/@trsong/Friend-Cycle-Problem](https://repl.it/@trsong/Friend-Cycle-Problem)
```py
import unittest

def find_cycles(friendships):
    capacity = len(friendships)
    uf = DisjointSet(capacity)
    for p1, friends in friendships.items():
        for p2 in friends:
            uf.union(p1, p2)
    return uf.groups


class DisjointSet(object):
    def __init__(self, capacity):
        self.parent = range(capacity)
        self.groups = capacity

    def find(self, p):
        while p != self.parent[p]:
            self.parent[p] = self.find(self.parent[p])
            p = self.parent[p]
        return p

    def union(self, p1, p2):
        parent1 = self.find(p1)
        parent2 = self.find(p2)
        if parent1 != parent2:
            self.parent[parent1] = parent2
            self.groups -= 1


class FindCycleSpec(unittest.TestCase):
    def test_example(self):
        friendships = {
            0: [1, 2],
            1: [0, 5],
            2: [0],
            3: [6],
            4: [],
            5: [1],
            6: [3]
        }
        expected = 3  # [0, 1, 2, 5], [3, 6], [4]
        self.assertEqual(expected, find_cycles(friendships))

    def test_no_friends(self):
        friendships = {
            0: [],
            1: [],
            2: []
        }
        expected = 3  # [0], [1], [2]
        self.assertEqual(expected, find_cycles(friendships))

    def test_all_friends(self):
        friendships = {
            0: [1, 2],
            1: [0, 2],
            2: [0, 1]
        }
        expected = 1  # [0, 1, 2]
        self.assertEqual(expected, find_cycles(friendships))
    
    def test_common_friend(self):
        friendships = {
            0: [1],
            1: [0, 2, 3],
            2: [1],
            3: [1],
            4: []
        }
        expected = 2  # [0, 1, 2, 3], [4]
        self.assertEqual(expected, find_cycles(friendships))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Jan 18, 2021 \[Medium\] Regular Numbers
--- 
> **Question:**  A regular number in mathematics is defined as one which evenly divides some power of `60`. Equivalently, we can say that a regular number is one whose only prime divisors are `2`, `3`, and `5`.
>
> These numbers have had many applications, from helping ancient Babylonians keep time to tuning instruments according to the diatonic scale.
>
> Given an integer N, write a program that returns, in order, the first N regular numbers.
 

**Solution:** [https://repl.it/@trsong/Regular-Numbers](https://repl.it/@trsong/Regular-Numbers)
```py
import unittest

def generate_regular_numbers(n):
    if n == 0:
        return []

    res = [1]
    i, j, k = 0, 0, 0
    for _ in xrange(n - 1):
        num = min(2 * res[i], 3 * res[j], 5 * res[k])
        res.append(num)

        if num % 2 == 0:
            i += 1

        if num % 3 == 0:
            j += 1

        if num % 5 == 0:
            k += 1
    return res


# Source: https://en.wikipedia.org/wiki/Regular_number
REGULAR_NUMBERS = [
    1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40,
    45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128,
    135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270,
    288, 300, 320, 324, 360, 375, 384, 400, 405
]


class GenerateRegularNumberSpec(unittest.TestCase):
    def test_empty_result(self):
        self.assertEqual([], generate_regular_numbers(0))

    def test_5_elements(self):
        self.assertEqual(REGULAR_NUMBERS[:5], generate_regular_numbers(5))

    def test_10_elements(self):
        self.assertEqual(REGULAR_NUMBERS[:10], generate_regular_numbers(10))

    def test_15_elements(self):
        self.assertEqual(REGULAR_NUMBERS[:15], generate_regular_numbers(15))

    def test_20_elements(self):
        self.assertEqual(REGULAR_NUMBERS[:20], generate_regular_numbers(20))

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Jan 17, 2021 \[Hard\] Count Elements in Sorted Matrix
--- 
> **Question:** Let A be an `N` by `M` matrix in which every row and every column is sorted.
>
> Given `i1`, `j1`, `i2`, and `j2`, compute the number of elements smaller than `A[i1, j1]` and larger than `A[i2, j2]`.

**Example:**
```py
Given the following matrix:
[[ 1,  3,  6, 10, 15, 20],
 [ 2,  7,  9, 14, 22, 25],
 [ 3,  8, 10, 15, 25, 30],
 [10, 11, 12, 23, 30, 35],
 [20, 25, 30, 35, 40, 45]]
And i1 = 1, j1 = 1, i2 = 3, j2 = 3
 
return 15 as there are 15 numbers in the matrix smaller than 7 or greater than 23.
```

**My thoughts:** The trick is to start from top-right cell. Either go left or go down, we can quickly figure out smaller elements within linear time `O(N + M)`. Once know how to find smaller, finding greater is just using total elements minus smaller.

**Solution with Two-Pointers:** [https://repl.it/@trsong/Count-Elements-in-Sorted-Matrix](https://repl.it/@trsong/Count-Elements-in-Sorted-Matrix)
```py
import unittest

def count_elements(matrix, pos1, pos2):
    n, m = len(matrix), len(matrix[0])
    v1 = matrix[pos1[0]][pos1[1]]
    v2 = matrix[pos2[0]][pos2[1]]
    if v1 > v2:
        return n * m

    less_than_v1 = count_smaller_elements(matrix, v1)
    not_greater_than_v2 = count_smaller_elements(matrix, v2, exclusive=False)
    return less_than_v1 + n * m - not_greater_than_v2


def count_smaller_elements(matrix, target, exclusive=True):
    n, m = len(matrix), len(matrix[0])
    res = 0
    row = 0
    for col in xrange(m - 1, -1, -1):
        while row < n:
            if matrix[row][col] > target or exclusive and matrix[row][col] == target:
                break
            row += 1
        res += row
    return res


class CountElementSpec(unittest.TestCase):
    def test_example(self):
        matrix = [
            [1, 3, 6, 10, 15, 20], 
            [2, 7, 9, 14, 22, 25],
            [3, 8, 10, 15, 25, 30], 
            [10, 11, 12, 23, 30, 35],
            [20, 25, 30, 35, 40, 45]]
        pos1, pos2 = (1, 1), (3, 3)
        expected = 15
        self.assertEqual(expected, count_elements(matrix, pos1, pos2))

    def test_no_elem_found(self):
        matrix = [
            [1, 2],
            [3, 6]
        ]
        pos1, pos2 = (0, 0), (1, 1)
        expected = 0
        self.assertEqual(expected, count_elements(matrix, pos1, pos2))

    def test_top_right_bottom_left(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]
        pos1, pos2 = (0, 2), (2, 0)
        expected = 4
        self.assertEqual(expected, count_elements(matrix, pos1, pos2))

    def test_covers_entire_matrix(self):
        matrix = [
            [1, 2, 3],
            [2, 3, 4],
            [9, 10, 11]
        ]
        pos1, pos2 = (2, 2), (0, 0)
        expected = 9
        self.assertEqual(expected, count_elements(matrix, pos1, pos2))

    def test_identical_pos(self):
        matrix = [
            [1, 2, 3],
            [2, 3, 4],
            [9, 10, 11]
        ]
        pos1, pos2 = (1, 1), (1, 1)
        expected = 7
        self.assertEqual(expected, count_elements(matrix, pos1, pos2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 16, 2021 \[Easy\] Exists Overlap Rectangle
--- 
> **Question:** You are given a list of rectangles represented by min and max x- and y-coordinates. Compute whether or not a pair of rectangles overlap each other. If one rectangle completely covers another, it is considered overlapping.
>
> For example, given the following rectangles:
```py
{
    "top_left": (1, 4),
    "dimensions": (3, 3) # width, height
},
{
    "top_left": (-1, 3),
    "dimensions": (2, 1)
},
{
    "top_left": (0, 5),
    "dimensions": (4, 3)
}
```
> return true as the first and third rectangle overlap each other.

**Solution:** [https://repl.it/@trsong/Exists-Overlap-Rectangles](https://repl.it/@trsong/Exists-Overlap-Rectangles)
```py
import unittest

def exists_overlap_rectangle(rectangles):
    rects = map(Rectangle.from_object, rectangles)
    # sort by x_min, then y_min
    rects.sort(key=Rectangle.get_bottom_left)
    n = len(rects)
    for i, rect1 in enumerate(rects):
        for j in xrange(i + 1, n):
            rect2 = rects[j]
            if rect2.x_min >= rect1.x_max:
                break
            if rect1.is_overlapped(rect2):
                return True
    
    return False


class Rectangle(object):
    def __init__(self, top_left, dimension):
        self.x_min = top_left[0]
        self.x_max = top_left[0] + dimension[0]
        self.y_min = top_left[1] - dimension[1]
        self.y_max = top_left[1]

    def __repr__(self):
        return "Rectangle(x_min=%d, x_max=%d, y_min=%d, y_max=%d)" % (self.x_min, self.x_max, self.y_min, self.y_max)

    def is_overlapped(self, other):
        return (max(self.x_min, other.x_min) < min(self.x_max, other.x_max) and
                max(self.y_min, other.y_min) < min(self.y_max, other.y_max))

    @staticmethod
    def from_object(obj):
        return Rectangle(obj.get("top_left"), obj.get("dimensions"))
    
    @staticmethod
    def get_bottom_left(rect):
        return (rect.x_min, rect.y_min)


class ExistsOverlapRectangleSpec(unittest.TestCase):
    def test_example(self):
        rectangles = [
            {
                "top_left": (1, 4),
                "dimensions": (3, 3)  # width, height
            },
            {
                "top_left": (-1, 3),
                "dimensions": (2, 1)
            },
            {
                "top_left": (0, 5),
                "dimensions": (4, 3)
            }
        ]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_empty_rectangle_list(self):
        self.assertFalse(exists_overlap_rectangle([]))

    def test_two_overlap_rectangle(self):
        rectangles = [
            {
                "top_left": (0, 1),
                "dimensions": (1, 3)  # width, height
            },
            {
                "top_left": (-1, 0),
                "dimensions": (3, 1)
            }
        ]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_two_overlap_rectangle_form_a_cross(self):
        rectangles = [
            {
                "top_left": (-1, 1),
                "dimensions": (3, 2)  # width, height
            },
            {
                "top_left": (0, 0),
                "dimensions": (1, 1)
            }
        ]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_same_y_coord_not_overlap(self):
        rectangles = [
            {
                "top_left": (0, 0),
                "dimensions": (1, 1)  # width, height
            },
            {
                "top_left": (1, 0),
                "dimensions": (2, 2)
            },
            {
                "top_left": (3, 0),
                "dimensions": (5, 2)
            }
        ]
        self.assertFalse(exists_overlap_rectangle(rectangles))

    def test_same_y_coord_overlap(self):
        rectangles = [
            {
                "top_left": (0, 0),
                "dimensions": (1, 1)  # width, height
            },
            {
                "top_left": (1, 0),
                "dimensions": (2, 2)
            },
            {
                "top_left": (3, 0),
                "dimensions": (5, 2)
            }
        ]
        self.assertFalse(exists_overlap_rectangle(rectangles))

    def test_rectangles_in_different_quadrant(self):
        rectangles = [
            {
                "top_left": (1, 1),
                "dimensions": (2, 2)  # width, height
            },
            {
                "top_left": (-1, 1),
                "dimensions": (2, 2)
            },
            {
                "top_left": (1, -1),
                "dimensions": (2, 2)
            },
            {
                "top_left": (-1, -1),
                "dimensions": (2, 2)
            }
        ]
        self.assertFalse(exists_overlap_rectangle(rectangles))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Jan 15, 2021 \[Medium\] Satisfactory Playlist
---
> **Question:** You have access to ranked lists of songs for various users. Each song is represented as an integer, and more preferred songs appear earlier in each list. For example, the list `[4, 1, 7]` indicates that a user likes song `4` the best, followed by songs `1` and `7`.
>
> Given a set of these ranked lists, interleave them to create a playlist that satisfies everyone's priorities.
>
> For example, suppose your input is `[[1, 7, 3], [2, 1, 6, 7, 9], [3, 9, 5]]`. In this case a satisfactory playlist could be `[2, 1, 6, 7, 3, 9, 5]`.

**My thoughts:** create a graph with vertex being song and edge `(u, v)` representing that u is more preferred than v. A topological order will make sure that all more preferred song will go before less preferred ones. Thus gives a list that satisfies everyone's priorities, if there is one (no cycle).

**Solution with Topological Sort:** [https://repl.it/@trsong/Satisfactory-Playlist-for-Everyone](https://repl.it/@trsong/Satisfactory-Playlist-for-Everyone)
```py
import unittest

def calculate_satisfactory_playlist(preference):
    neighbors = {}
    inbound = {}
    for songs in preference:
        prev = None
        for cur_song in songs:
            inbound[cur_song] = inbound.get(cur_song, 0)
            if prev:
                neighbors[prev] = neighbors.get(prev, set())
                if cur_song not in neighbors[prev]:
                    neighbors[prev].add(cur_song)
                    inbound[cur_song] += 1
            prev = cur_song

    queue = []
    for song, count in inbound.items():
        if count == 0:
            queue.append(song)

    res = []
    while queue:
        cur = queue.pop(0)
        res.append(cur)

        if cur not in neighbors:
            continue

        for nb in neighbors[cur]:
            inbound[nb] -= 1
            if inbound[nb] == 0:
                queue.append(nb)
            
    return res if len(res) == len(inbound) else None
        

class CalculateSatisfactoryPlaylistSpec(unittest.TestCase):
    def validate_result(self, preference, suggested_order):
        song_set = set([song for songs in preference for song in songs])
        self.assertEqual(
            song_set,
            set(suggested_order),
            "Missing song: " + str(str(song_set - set(suggested_order))))

        for i in xrange(len(suggested_order)):
            for j in xrange(i+1, len(suggested_order)):
                for lst in preference:
                    song1, song2 = suggested_order[i], suggested_order[j]
                    if song1 in lst and song2 in lst:
                        self.assertLess(
                            lst.index(song1), 
                            lst.index(song2),
                            "Suggested order {} conflict: {} cannot be more popular than {}".format(suggested_order, song1, song2))

    def test_example(self):
        preference = [[1, 7, 3], [2, 1, 6, 7, 9], [3, 9, 5]]
        # possible order: 2, 1, 6, 7, 3, 9, 5
        suggested_order = calculate_satisfactory_playlist(preference) 
        self.validate_result(preference, suggested_order)
    
    def test_preference_contains_duplicate(self):
        preference = [[1, 2], [1, 2], [1, 2]]
        # possible order: 1, 2
        suggested_order = calculate_satisfactory_playlist(preference) 
        self.validate_result(preference, suggested_order)

    def test_empty_graph(self):
        self.assertEqual([], calculate_satisfactory_playlist([]))

    def test_cyclic_graph(self):
        preference = [[1, 2, 3], [1, 3, 2]]
        self.assertIsNone(calculate_satisfactory_playlist(preference))

    def test_acyclic_graph(self):
        preference = [[1, 2], [2, 3], [1, 3, 5], [2, 5], [2, 4]]
        # possible order: 1, 2, 3, 4, 5
        suggested_order = calculate_satisfactory_playlist(preference)
        self.validate_result(preference, suggested_order)

    def test_disconnected_graph(self):
        preference = [[0, 1], [2, 3], [3, 4]]
        # possible order: 0, 1, 2, 3, 4
        suggested_order = calculate_satisfactory_playlist(preference)
        self.validate_result(preference, suggested_order)

    def test_disconnected_graph2(self):
        preference = [[0, 1], [2], [3]]
        # possible order: 0, 1, 2, 3
        suggested_order = calculate_satisfactory_playlist(preference)
        self.validate_result(preference, suggested_order)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 14, 2021 LC 1344 \[Easy\] Angle between Clock Hands
---
> **Question:** Given a clock time in `hh:mm` format, determine, to the nearest degree, the angle between the hour and the minute hands.

**Example:**
```py
Input: "9:00"
Output: 90
```

**Solution:** [https://repl.it/@trsong/Calculate-Angle-between-Clock-Hands](https://repl.it/@trsong/Calculate-Angle-between-Clock-Hands)
```py
import unittest

def clock_angle(hhmm):
    h, m = map(int, hhmm.split(':'))
    h %= 12
    unit_hr = 360 / 12.0
    unit_min = 360 / 60.0

    hr_hand = h * unit_hr + m * unit_hr / 60.0 
    min_hand = m * unit_min
    angle = abs(min_hand - hr_hand)
    return min(angle, 360 - angle)


class ClockAngleSpec(unittest.TestCase):
    def test_minute_point_zero(self):
        hhmm = "12:00"
        angle = 0
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_minute_point_zero2(self):
        hhmm = "1:00"
        angle = 30
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_minute_point_zero3(self):
        hhmm = "9:00"
        angle = 90
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_minute_point_zero4(self):
        hhmm = "6:00"
        angle = 180
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_half_pass_hour(self):
        hhmm = "12:30"
        angle = 165
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_half_pass_hour2(self):
        hhmm = "3:30"
        angle = 75
        self.assertEqual(angle, clock_angle(hhmm))

    def test_irregular_time(self):
        hhmm = "16:20"
        angle = 10
        self.assertEqual(angle, clock_angle(hhmm))

    def test_irregular_time3(self):
        hhmm = "3:15"
        angle = 7.5
        self.assertEqual(angle, clock_angle(hhmm))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 13, 2021 \[Hard\] Longest Common Subsequence of 3 Strings
---
> **Question:** Write a program that computes the length of the longest common subsequence of three given strings. 
> 
> For example, given `"epidemiologist"`, `"refrigeration"`, and `"supercalifragilisticexpialodocious"`, it should return `5`, since the longest common subsequence is `"eieio"`.

**Solution with DP:** [https://repl.it/@trsong/Longest-Common-Subsequence-of-3-Strings](https://repl.it/@trsong/Longest-Common-Subsequence-of-3-Strings)
```py
import unittest

def lcs(seq1, seq2, seq3):
    n, m, k = len(seq1), len(seq2), len(seq3)
    # Let dp[n][m][k] represents lcs for seq1[:n], seq2[:m], seq3[:k]
    # dp[n][m][k] = dp[n-1][m-1][k-1] + 1  if all of following matches: seq[n-1], seq2[m-1], seq3[k-1]
    #          = max(dp[n-1][m][k], dp[n][m-1][k], dp[n][m][k-1]) otherwise
    dp = [[[0 for _ in xrange(k + 1)] for _ in xrange(m + 1)] for _ in xrange(n + 1)]
    for i in xrange(1, n + 1):
        for j in xrange(1, m + 1):
            for t in xrange(1, k + 1):
                if seq1[i-1] == seq2[j-1] == seq3[t-1]:
                    dp[i][j][t] = dp[i-1][j-1][t-1] + 1
                else:
                    dp[i][j][t] = max(dp[i-1][j][t], dp[i][j-1][t], dp[i][j][t-1])
    return dp[n][m][k]


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

### Jan 12, 2021 LC 138 \[Medium\] Deepcopy List with Random Pointer
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

**Solution:** [https://repl.it/@trsong/Create-Deepcopy-List-with-Random](https://repl.it/@trsong/Create-Deepcopy-List-with-Random)
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
        if p.random:
            cur_copy = p.next
            cur_copy.random = p.random.next
        p = p.next.next
    
    # Step3: Partition even and odd nodes to restore original list and get result
    copy_head = head.next
    p = head
    while p:
        if p.next.next:
            cur_copy = p.next
            cur_copy.next = p.next.next
        p.next = p.next.next
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
    unittest.main(exit=False, verbosity=2)
```

### Jan 11, 2021 \[Easy\] Spreadsheet Columns
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

**My thoughts:** Map each digit to letter with value of digit - 1 won't work. E.g. `1 - 1 = 0 -> 'A'`. But `27 - 1 = 26 -> "AB" != 'AA'`. The trick is to treat `'Z'` as zero while printing and treat it as 26 while doing calculations. e.g. 1 -> A. 2 -> B. 25 -> Y. 0 -> Z.  

**Solution:** [https://repl.it/@trsong/Find-Spreadsheet-Columns](https://repl.it/@trsong/Find-Spreadsheet-Columns)
```py
import unittest

def spreadsheet_columns(n):
    base = 26
    res = []
    while n > 0:
        remainder = n % base
        letter_index = (remainder - 1) % base
        letter = chr(ord('A') + letter_index)
        res.append(letter)
        n //= base
        if remainder == 0:
            n -= 1
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
    unittest.main(verbosity=2, exit=False)
```

### Jan 10, 2021 \[Medium\] Multiset Partition
---
> **Question:** Given a multiset of integers, return whether it can be partitioned into two subsets whose sums are the same.
>
> For example, given the multiset {15, 5, 20, 10, 35, 15, 10}, it would return true, since we can split it up into {15, 5, 10, 15, 10} and {20, 35}, which both add up to 55.
>
> Given the multiset {15, 5, 20, 10, 35}, it would return false, since we can't split it up into two subsets that add up to the same sum.

**Solution with DP:** [https://repl.it/@trsong/Multiset-Partition-into-Equal-Sum](https://repl.it/@trsong/Multiset-Partition-into-Equal-Sum)
```py
import unittest

def has_equal_sum_partition(multiset):
    num_sum = sum(multiset)
    return num_sum % 2 == 0 and has_subset_sum(multiset, num_sum // 2)


def has_subset_sum(nums, target):
    n = len(nums)
    # Let dp[target] represents if there exists a subset sum target
    dp = [False] * (target + 1)
    dp[0] = True

    for i in xrange(1, n + 1):
        cur_num = nums[i - 1]
        for sum in xrange(target, cur_num - 1, -1):
            dp[sum] |= dp[sum - cur_num]
    
    return dp[target]


class HasEqualSumPartitionSpec(unittest.TestCase):
    def test_example(self):
        nums = [15, 5, 20, 10, 35, 15, 10]
        # Partition into [15, 5, 10, 15, 10] and [20, 35]
        self.assertTrue(has_equal_sum_partition(nums))

    def test_example2(self):
        nums = [15, 5, 20, 10, 35]
        self.assertFalse(has_equal_sum_partition(nums))

    def test_empty_sets(self):
        nums = []
        self.assertTrue(has_equal_sum_partition(nums))

    def test_partition_set_has_same_elements(self):
        nums = [5, 1, 5, 1]
        self.assertTrue(has_equal_sum_partition(nums))

    def test_unable_to_partition(self):
        nums = [1, 2, 2]
        self.assertFalse(has_equal_sum_partition(nums))

    def test_has_distinct_elements(self):
        nums = [1, 2, 4, 3, 8]
        # Partition into [1, 8] and [2, 4, 3]
        self.assertTrue(has_equal_sum_partition(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 9, 2021 \[Easy\] Compare Version Numbers
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

**Solution:** [https://repl.it/@trsong/Compare-Two-Version-Numbers](https://repl.it/@trsong/Compare-Two-Version-Numbers)
```py
import unittest

def compare(v1, v2):
    version_seq1 = map(int, (v1 or '0').split('.'))
    version_seq2 = map(int, (v2 or '0').split('.'))
    len1, len2 = len(version_seq1), len(version_seq2)

    for i in xrange(max(len1, len2)):
        num1 = version_seq1[i] if i < len1 else 0
        num2 = version_seq2[i] if i < len2 else 0
        if num1 < num2:
            return -1
        elif num1 > num2:
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
    unittest.main(exit=False, verbosity=2)
```

### Jan 8, 2021 \[Easy\] Step Word Anagram
---
> **Question:** A step word is formed by taking a given word, adding a letter, and anagramming the result. For example, starting with the word `"APPLE"`, you can add an `"A"` and anagram to get `"APPEAL"`.
>
> Given a dictionary of words and an input word, create a function that returns all valid step words.

**Solution:** [https://repl.it/@trsong/Step-Word-Anagram](https://repl.it/@trsong/Step-Word-Anagram)
```py
import unittest

def find_step_anagrams(word, dictionary):
    word_histogram = generate_histogram(word)
    step_anagram_distance = 1
    is_step_anagram = lambda w: (
        step_anagram_distance == 
        len(w) - len(word) ==  
        histogram_distance_between(generate_histogram(w), word_histogram))
    return filter(is_step_anagram, dictionary)


def generate_histogram(word):
    histogram = {}
    for ch in word:
        histogram[ch] = histogram.get(ch, 0) + 1
    return histogram


def histogram_distance_between(hst1, hst2):
    dist = 0
    for key in hst1.keys():
        dist += max(0, hst1[key] - hst2.get(key, 0))
    return dist


class FindStepAnagramSpec(unittest.TestCase):
    def test_example(self):
        word = 'APPLE'
        dictionary = ['APPEAL', 'CAPPLE', 'PALPED']
        expected = ['APPEAL', 'CAPPLE', 'PALPED']
        self.assertEqual(sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_empty_word(self):
        word = ''
        dictionary = ['A', 'B', 'AB', 'ABC']
        expected = ['A', 'B']
        self.assertEqual(sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_empty_dictionary(self):
        word = 'ABC'
        dictionary = []
        expected = []
        self.assertEqual(sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_no_match(self):
        word = 'ABC'
        dictionary = ['BBB', 'ACCC']
        expected = []
        self.assertEqual(sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_no_match2(self):
        word = 'AA'
        dictionary = ['ABB']
        expected = []
        self.assertEqual(sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_repeated_chars(self):
        word = 'AAA'
        dictionary = ['A', 'AA', 'AAA', 'AAAA', 'AAAAB', 'AAB', 'AABA']
        expected = ['AAAA', 'AABA']
        self.assertEqual(sorted(expected), sorted(find_step_anagrams(word, dictionary)))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 7, 2021 \[Medium\] Normalize Pathname
---
> **Question:** Given an absolute pathname that may have `"."` or `".."` as part of it, return the shortest standardized path.
>
> For example, given `"/usr/bin/../bin/./scripts/../"`, return `"/usr/bin"`.

**Solution with Stack:** [https://repl.it/@trsong/Normalize-to-Absolute-Pathname](https://repl.it/@trsong/Normalize-to-Absolute-Pathname)
```py
import unittest

def normalize_pathname(path):
    stack = []
    for file in path.split('/'):
        if not file or file == '.':
            continue
        elif stack and file == '..':
            stack.pop()
        else:
            stack.append(file)
    return '/' + '/'.join(stack)


class NormalizePathnameSpec(unittest.TestCase):
    def test_example(self):
        original_path = 'usr/bin/../bin/./scripts/../'
        normalized_path = '/usr/bin'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_empty_path(self):
        original_path = ''
        normalized_path = '/'
        self.assertEqual(normalized_path, normalize_pathname(original_path))
    
    def test_file_in_root_directory(self):
        original_path = 'bin'
        normalized_path = '/bin'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_parent_of_root_dirctory(self):
        original_path = '/a/b/..'
        normalized_path = '/a'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_parent_of_root_dirctory2(self):
        original_path = '../../'
        normalized_path = '/'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_parent_of_root_dirctory3(self):
        original_path = '../../../../a/'
        normalized_path = '/a'
        self.assertEqual(normalized_path, normalize_pathname(original_path))
    
    def test_current_directory(self):
        original_path = '.'
        normalized_path = '/'
        self.assertEqual(normalized_path, normalize_pathname(original_path))
    
    def test_current_directory2(self):
        original_path = './a/./b/c/././d/e/.'
        normalized_path = '/a/b/c/d/e'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_parent_of_current_directory(self):
        original_path = 'a/b/c/././.././../'
        normalized_path = '/a'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_hidden_file(self):
        original_path = './home/.bashrc'
        normalized_path = '/home/.bashrc'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_file_with_extention(self):
        original_path = 'home/autorun.inf'
        normalized_path = '/home/autorun.inf'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_directory_with_dots(self):
        original_path = 'home/work/com.myPythonProj.ui/src/../.git/'
        normalized_path = '/home/work/com.myPythonProj.ui/.git'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_parent_of_directory_with_dots(self):
        original_path = 'home/work/com.myPythonProj.db.server.test/./.././com.myPythonProj.db.server/./src/./.git/../.git/.gitignore'
        normalized_path = '/home/work/com.myPythonProj.db.server/src/.git/.gitignore'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    """
    A unix file system support consecutive slashes. 
    Consecutive slashes is equivalent to single slash. 
    eg. 'a//////b////c//' is equivalent to '/a/b/c'
    """
    def test_consecutive_slashes(self):
        original_path = 'a/.//////b///..//c//'
        normalized_path = '/a/c'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_consecutive_slashes2(self):
        original_path = '///////'
        normalized_path = '/'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_consecutive_slashes3(self):
        original_path = '/../..////..//././//.//../a/////'
        normalized_path = '/a'
        self.assertEqual(normalized_path, normalize_pathname(original_path))
        
    def test_consecutive_slashes4(self):
        original_path = '//////.a//////'
        normalized_path = '/.a'
        self.assertEqual(normalized_path, normalize_pathname(original_path))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 6, 2021 LC 692 \[Medium\] Top K Frequent words
--- 
> **Question:** Given a non-empty list of words, return the k most frequent words. The output should be sorted from highest to lowest frequency, and if two words have the same frequency, the word with lower alphabetical order comes first. Input will contain only lower-case letters.

**Example 1:**
```py
Input: ["i", "love", "leapcode", "i", "love", "coding"], k = 2
Output: ["i", "love"]
Explanation: "i" and "love" are the two most frequent words.
    Note that "i" comes before "love" due to a lower alphabetical order.
```

**Example 2:**
```py
Input: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
Output: ["the", "is", "sunny", "day"]
Explanation: "the", "is", "sunny" and "day" are the four most frequent words,
    with the number of occurrence being 4, 3, 2 and 1 respectively.
```

**Solution with Priority Queue:** [https://repl.it/@trsong/Find-Top-K-Frequent-Elements](https://repl.it/@trsong/Find-Top-K-Frequent-Elements)
```py
import unittest
from Queue import PriorityQueue

def top_k_freq_words(words, k):
    histogram = {}
    for word in words:
        histogram[word] = histogram.get(word, 0) + 1
    
    max_heap = PriorityQueue()
    for word, count in histogram.items():
        max_heap.put((-count, word))

    res = []
    for _ in xrange(k):
        _, word = max_heap.get()
        res.append(word)
    return res


class TopKFreqWordSpec(unittest.TestCase):
    def test_example1(self):
        input = ["i", "love", "leapcode", "i", "love", "coding"]
        k = 2
        expected = ["i", "love"]
        self.assertEqual(expected, top_k_freq_words(input, k))

    def test_example2(self):
        input =  ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"]
        k = 4
        expected = ["the", "is", "sunny", "day"]
        self.assertEqual(expected, top_k_freq_words(input, k))

    def test_same_count_words(self):
        input =  ["c", "cb", "cba", "cdbaa"]
        k = 3
        expected = ["c", "cb", "cba"]
        self.assertEqual(expected, top_k_freq_words(input, k))

    def test_same_count_words2(self):
        input =  ["a", "c", "b", "d"]
        k = 3
        expected = ["a", "b", "c"]
        self.assertEqual(expected, top_k_freq_words(input, k))
    
if __name__ == '__main__':
    unittest.main(exit=False)
```

### Jan 5, 2021 \[Hard\] Anagram to Integer
---
> **Question:** You are given a string formed by concatenating several words corresponding to the integers `zero` through `nine` and then anagramming.
>
> For example, the input could be `'niesevehrtfeev'`, which is an anagram of `'threefiveseven'`. Note that there can be multiple instances of each integer.
>
> Given this string, return the original integers in sorted order. In the example above, this would be `357`.

**Solution with Backtracking:** [https://repl.it/@trsong/Convert-Anagram-to-Integer](https://repl.it/@trsong/Convert-Anagram-to-Integer)
```py
import unittest

DIGITS = ['zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine']


def anagram_to_integer(s):
    if not s:
        return 0

    input_freq = calculate_char_freq(s)
    digit_freq_map = map(calculate_char_freq, DIGITS)
    return int(backtrack([], 0, input_freq, digit_freq_map))


def calculate_char_freq(s):
    char_freq = {}
    for ch in s:
        char_freq[ch] = char_freq.get(ch, 0) + 1
    return char_freq


def backtrack(accu, digit_index, input_freq, digit_freq_map):
    if digit_index >= len(DIGITS):
        return None
    elif not input_freq:
        return "".join(accu)
    else:
        for digit in xrange(digit_index, len(DIGITS)):
            max_repeats = float('inf')
            for ch, ch_count in digit_freq_map[digit].items():
                max_repeats = min(max_repeats, input_freq.get(ch, 0) // ch_count)

            for repeats in xrange(max_repeats, 0, -1):
                for ch, ch_count in digit_freq_map[digit].items():
                    input_freq[ch] -= ch_count * repeats
                    if input_freq[ch] == 0:
                        del input_freq[ch]

                accu.append(str(digit) * repeats)

                res = backtrack(accu, digit_index + 1, input_freq, digit_freq_map)
                if res is not None:
                    return res

                accu.pop()

                for ch, ch_count in digit_freq_map[digit].items():
                    input_freq[ch] = input_freq.get(ch, 0) + ch_count * repeats
        return None
    
        
class AnagramToIntegerSpec(unittest.TestCase):
    def test_example(self):
        s = 'niesevehrtfeev'
        expected = 357
        self.assertEqual(expected, anagram_to_integer(s))

    def test_empty_string(self):
        self.assertEqual(0, anagram_to_integer(''))

    def test_contains_duplicate_characters(self):
        s = 'nininene'
        expected = 99
        self.assertEqual(expected, anagram_to_integer(s))

    def test_contains_duplicate_characters2(self):
        s = 'twoonefourfourtwoone'
        expected = 112244
        self.assertEqual(expected, anagram_to_integer(s))
    
    def test_char_in_sorted_order(self):
        s = 'eeeffhioorrttuvw'
        expected = 2345
        self.assertEqual(expected, anagram_to_integer(s))

    def test_zero(self):
        s = 'zero'
        expected = 0
        self.assertEqual(expected, anagram_to_integer(s))
    
    def test_should_omit_zero(self):
        s = 'onetwothreefourfivesixseveneightnine'
        expected = 123456789
        self.assertEqual(expected, anagram_to_integer(s))

    def test_unique_character(self):
        s = 'oneoneoneone'
        expected = 1111
        self.assertEqual(expected, anagram_to_integer(s))

    def test_one_not_exists(self):
        s = 'twonine'
        expected = 29
        self.assertEqual(expected, anagram_to_integer(s))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Jan 4, 2021 \[Medium\] Sort a K-Sorted Array
---
> **Question:** You are given a list of `N` numbers, in which each number is located at most `k` places away from its sorted position. For example, if `k = 1`, a given element at index `4` might end up at indices `3`, `4`, or `5`.
>
> Come up with an algorithm that sorts this list in O(N log k) time.

**Example:**
```py
Input: [3, 2, 6, 5, 4], k=2
Output: [2, 3, 4, 5, 6]
As seen above, every number is at most 2 indexes away from its proper sorted index.
``` 

**Solution with Priority Queue:** [https://repl.it/@trsong/Sort-a-K-Sorted-Array](https://repl.it/@trsong/Sort-a-K-Sorted-Array)
```py
import unittest
from Queue import PriorityQueue

def sort_partial_list(nums, k):
    pq = PriorityQueue()
    n = len(nums)
    for i in xrange(min(n, k + 1)):
        pq.put(nums[i])

    res = []
    for i in xrange(k + 1, n):
        res.append(pq.get())
        pq.put(nums[i])

    while not pq.empty():
        res.append(pq.get())

    return res
    

class SortPartialListSpec(unittest.TestCase):
    def test_example(self):
        k, nums = 2, [3, 2, 6, 5, 4]
        self.assertEqual(sorted(nums), sort_partial_list(nums, k))

    def test_empty_list(self):
        self.assertEqual([], sort_partial_list([], 0))
        self.assertEqual([], sort_partial_list([], 3))
    
    def test_unsorted_list(self):
        k, nums = 10, [8, 7, 6, 5, 3, 1, 2, 4]
        self.assertEqual(sorted(nums), sort_partial_list(nums, k))

    def test_k_is_3(self):
        k, nums = 3, [6, 5, 3, 2, 8, 10, 9]
        self.assertEqual(sorted(nums), sort_partial_list(nums, k))

    def test_another_example_k_is_3(self):
        k, nums = 3, [2, 6, 3, 12, 56, 8]
        self.assertEqual(sorted(nums), sort_partial_list(nums, k))

    def test_k_is_4(self):
        k, nums = 4, [10, 9, 8, 7, 4, 70, 60, 50]
        self.assertEqual(sorted(nums), sort_partial_list(nums, k))

    def test_list_with_duplicated_values(self):
        k, nums = 3, [3, 2, 2, 4, 4, 3]
        self.assertEqual(sorted(nums), sort_partial_list(nums, k))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 3, 2021 \[Medium\] Detect Linked List Cycle
---
> **Question:** Given a linked list, determine if the linked list has a cycle in it. 

**Example:**
```py
Input: 4 -> 3 -> 2 -> 1 -> 3 ... 
Output: True
```

**Solution with Fast and Slow Pointers:** [https://repl.it/@trsong/Detect-Linked-List-Cycle](https://repl.it/@trsong/Detect-Linked-List-Cycle)
```py
import unittest

def contains_cycle(lst):
    fast = slow = lst
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    @staticmethod
    def List(intersect_index, vals):
        dummy = p = ListNode(-1)
        intersect_node = None
        for val in vals:
            p.next = ListNode(val)
            p = p.next
            if intersect_index == 0:
                intersect_node = p
            intersect_index -= 1
        p.next = intersect_node
        return dummy.next


class ContainsCycleSpec(unittest.TestCase):
    def test_example(self):
        lst = ListNode.List(intersect_index = 1, vals=[4, 3, 2, 1])
        self.assertTrue(contains_cycle(lst))

    def test_empty_list(self):
        self.assertFalse(contains_cycle(None))

    def test_list_with_self_pointing_node(self):
        lst = ListNode.List(intersect_index = 0, vals=[1])
        self.assertTrue(contains_cycle(lst))

    def test_acyclic_list_with_duplicates(self):
        lst = ListNode.List(intersect_index = -1, vals=[1, 1, 1, 1, 1, 1])
        self.assertFalse(contains_cycle(lst))

    def test_acyclic_list_with_duplicates2(self):
        lst = ListNode.List(intersect_index = -1, vals=[1, 2, 3, 1, 2, 3])
        self.assertFalse(contains_cycle(lst))

    def test_cyclic_list_with_duplicates(self):
        lst = ListNode.List(intersect_index = 0, vals=[1, 2, 3, 1, 2, 3])
        self.assertTrue(contains_cycle(lst))

    def test_cyclic_list(self):
        lst = ListNode.List(intersect_index = 6, vals=[0, 1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(contains_cycle(lst))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Jan 2, 2021 \[Easy\] Determine If Linked List is Palindrome
---
> **Question:** You are given a doubly linked list. Determine if it is a palindrome. 

**Solution:** [https://repl.it/@trsong/Determine-If-Linked-List-is-Palindrome](https://repl.it/@trsong/Determine-If-Linked-List-is-Palindrome)
```py
import unittest

def is_palindrome(lst):
    if not lst:
        return True

    first = last = lst
    length = 1
    while last.next:
        last = last.next
        length += 1

    for _ in xrange(length // 2):
        if first.val != last.val:
            return False
        first = first.next
        last = last.prev

    return True


class ListNode(object):
    def __init__(self, val, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next
    
    @staticmethod
    def List(*vals):
        dummy = cur = ListNode(-1)
        for val in vals:
            cur.next = ListNode(val, cur)
            cur = cur.next
        dummy.next.prev = None
        return dummy.next


class IsPalindromeSpec(unittest.TestCase):
    def test_empty_list(self):
        self.assertTrue(is_palindrome(None))

    def test_one_element_list(self):
        self.assertTrue(is_palindrome(ListNode.List(42)))

    def test_two_element_list(self):
        self.assertFalse(is_palindrome(ListNode.List(1, 2)))

    def test_two_element_palindrome(self):
        self.assertTrue(is_palindrome(ListNode.List(6, 6)))

    def test_three_element_list(self):
        self.assertFalse(is_palindrome(ListNode.List(1, 2, 3)))

    def test_three_element_list2(self):
        self.assertFalse(is_palindrome(ListNode.List(1, 1, 2)))

    def test_three_element_list3(self):
        self.assertFalse(is_palindrome(ListNode.List(1, 2, 2)))

    def test_three_element_palindrome(self):
        self.assertTrue(is_palindrome(ListNode.List(1, 2, 1)))

    def test_three_element_palindrome2(self):
        self.assertTrue(is_palindrome(ListNode.List(1, 1, 1)))

    def test_even_element_list(self):
        self.assertFalse(is_palindrome(ListNode.List(1, 2, 3, 4, 2, 1)))

    def test_even_element_list2(self):
        self.assertTrue(is_palindrome(ListNode.List(1, 2, 3, 3, 2, 1)))

    def test_odd_element_list(self):
        self.assertTrue(is_palindrome(ListNode.List(1, 2, 3, 2, 1)))

    def test_odd_element_list2(self):
        self.assertFalse(is_palindrome(ListNode.List(1, 2, 3, 3, 1)))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Jan 1, 2021 \[Easy\] Map Digits to Letters
---
> **Question:** Given a mapping of digits to letters (as in a phone number), and a digit string, return all possible letters the number could represent. You can assume each valid number in the mapping is a single digit.

**Example:**
```py
Input: {'2': ['a', 'b', 'c'], '3': ['d', 'e', 'f']}, '23'
Output: ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
```

**Soltion with Backtracking:** [https://repl.it/@trsong/Map-Digits-to-All-Possible-Letters](https://repl.it/@trsong/Map-Digits-to-All-Possible-Letters)
```py
import unittest

def digits_to_letters(digits, dictionary):
    if not digits:
        return []

    res = []
    backtrack(res, [], digits, 0, dictionary)
    return res


def backtrack(res, accu, digits, digit_index, dictionary):
    if digit_index >= len(digits):
        res.append(''.join(accu))
    else:
        for ch in dictionary[digits[digit_index]]:
            accu.append(ch)
            backtrack(res, accu, digits, digit_index + 1, dictionary)
            accu.pop()


class DigitsToLetterSpec(unittest.TestCase):
    def assert_letters(self, res, expected):
        self.assertEqual(sorted(res), sorted(expected))

    def test_empty_digits(self):
        self.assert_letters(digits_to_letters("", {}), [])

    def test_one_digit(self):
        dictionary = {'2': ['a', 'b', 'c'], '3': ['d', 'e', 'f']}
        self.assert_letters(
            ['a', 'b', 'c'],
            digits_to_letters("2", dictionary))

    def test_repeated_digits(self):
        dictionary = {'2': ['a', 'b']}
        self.assert_letters(
            ['aa', 'ab', 'ba', 'bb'],
            digits_to_letters("22", dictionary))

    def test_example(self):
        dictionary = {'2': ['a', 'b', 'c'], '3': ['d', 'e', 'f']}
        self.assert_letters(
            ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf'],
            digits_to_letters("23", dictionary))

    def test_early_google_url(self):
        dictionary = {'2': ['a', 'b', 'c'], '3': ['d', 'e', 'f'], '4': ['g', 'h', 'i'], '5': ['j', 'k', 'l'], '6': ['m', 'n', 'o']}
        self.assertTrue('google' in digits_to_letters("466453", dictionary))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Dec 31, 2020 \[Medium\] Find Minimum Element in a Sorted and Rotated Array
---
> **Question:** Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand. Find the minimum element in `O(log N)` time. You may assume the array does not contain duplicates.
>
> For example, given `[5, 7, 10, 3, 4]`, return `3`.


**My thoughts:** the idea of binary search is all about how to shrink searching domain into half. When mid element is smaller than right, always go left. Or else, there must be a rotation pivot between. In this case, go right. 

**Solution with Binary Search:** [https://repl.it/@trsong/Find-Minimum-Element-in-a-Sorted-and-Rotated-Array](https://repl.it/@trsong/Find-Minimum-Element-in-a-Sorted-and-Rotated-Array)
```py
import unittest

def find_min_index(nums):
    if not nums:
        return None
        
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] > nums[hi]:
            lo = mid + 1
        else:
            hi = mid
    return lo


class FindMinIndexSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(3, find_min_index([5, 7, 10, 3, 4]))

    def test_array_without_rotation(self):
        self.assertIsNone(find_min_index([]))
    
    def test_array_without_rotation2(self):
        self.assertEqual(0, find_min_index([1, 2, 3]))
    
    def test_array_without_rotation3(self):
        self.assertEqual(0, find_min_index([1, 3]))

    def test_array_with_one_rotation(self):
        self.assertEqual(3, find_min_index([4, 5, 6, 1, 2, 3]))
    
    def test_array_with_one_rotation2(self):
        self.assertEqual(3, find_min_index([13, 18, 25, 2, 8, 10]))

    def test_array_with_one_rotation3(self):
        self.assertEqual(1, find_min_index([6, 1, 2, 3, 4, 5]))

    def test_array_with_one_rotation4(self):
        self.assertEqual(2, find_min_index([5, 6, 1, 2, 3, 4]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 30, 2020 LC 77 \[Medium\] Combinations
---
> **Question:** Given two integers `n` and `k`, return all possible combinations of `k` numbers out of `1 ... n`.

**Example:**
```py
Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```


**Solution with Backtracking:** [https://repl.it/@trsong/Combinations](https://repl.it/@trsong/Combinations)
```py
import unittest

def generate_combinations(n, k):
    if k > n or k <= 0:
        return []
    res = []
    nums = range(1, n + 1)
    backtrack(res, [], 0, k, nums)
    return res


def backtrack(res, chosen, current_index, k, nums):
    n = len(nums)
    if k == 0:
        res.append(chosen[:])
    elif n - current_index >= k:
        # Case 1: choose current num
        chosen.append(nums[current_index])
        backtrack(res, chosen, current_index + 1, k - 1, nums)
        chosen.pop()
        # Case 2: not choose current num
        backtrack(res, chosen, current_index + 1, k, nums)


class GenerateCombinationSpec(unittest.TestCase):
    def test_example(self):
        n, k = 4, 2
        expected = [[2, 4], [3, 4], [2, 3], [1, 2], [1, 3], [1, 4]]
        self.assertEqual(sorted(expected), generate_combinations(n, k))

    def test_choose_zero(self):
        n, k = 10, 0
        expected = []
        self.assertEqual(sorted(expected), generate_combinations(n, k))

    def test_choose_one(self):
        n, k = 6, 1
        expected = [[1], [2], [3], [4], [5], [6]]
        self.assertEqual(sorted(expected), generate_combinations(n, k))

    def test_choose_all(self):
        n, k = 6, 6
        expected = [[1, 2, 3, 4, 5, 6]]
        self.assertEqual(sorted(expected), generate_combinations(n, k))

    def test_choose_three(self):
        n, k = 4, 3
        expected = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
        self.assertEqual(sorted(expected), generate_combinations(n, k))

    def test_choose_more_than_supply(self):
        n, k = 4, 10
        expected = []
        self.assertEqual(sorted(expected), generate_combinations(n, k))


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```


### Dec 29, 2020 \[Easy\] Permutations
---
> **Question:** Given a number in the form of a list of digits, return all possible permutations.
>
> For example, given `[1,2,3]`, return `[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]`.

**My thoughts:** total number of permutations equals `n * n-1 * n-2 * ... * 2 * 1`. So in **Step 1**: swap all `n` number with index `0`. And in **Step 2**: swap the rest `n - 1` numbers with index `1` ... and so on.

For problem with size `k`, we swap the `n - k` th element with the result from problem with size `n - k`. 

**Example:**
```py
Suppose the input is [0, 1, 2]
├ swap(0, 0) 
│ ├ swap(1, 1) 
│ │ └ swap(2, 2)  gives [0, 1, 2]
│ └ swap(1, 2)  
│   └ swap(2, 2)  gives [0, 2, 1]
├ swap(0, 1) 
│ ├ swap(1, 1) 
│ │ └ swap(2, 2)  gives [1, 0, 2]
│ └ swap(1, 2)  
│   └ swap(2, 2)  gives [1, 2, 0]
└ swap(0, 2)  
  ├ swap(1, 1)
  │ └ swap(2, 2)  gives [2, 1, 0]
  └ swap(1, 2)
    └ swap(2, 2)  gives [2, 0, 1]
```

**Solution with Backtracking:** [https://repl.it/@trsong/Generate-Permutations](https://repl.it/@trsong/Generate-Permutations)
```py
import unittest

def generate_permutations(nums):
    res = []
    backtrack(res, 0, nums)
    return res


def backtrack(res, current_index, nums):
    n = len(nums)
    if current_index >= n :
        res.append(nums[:])
    else:
        for i in xrange(current_index, n):
            nums[current_index], nums[i] = nums[i], nums[current_index]
            backtrack(res, current_index + 1, nums)
            nums[current_index], nums[i] = nums[i], nums[current_index]


class CalculatePermutationSpec(unittest.TestCase):
    def test_permuation_of_empty_array(self):
        self.assertEqual( [[]], generate_permutations([]))

    def test_permuation_of_2(self):
        self.assertEqual(
            sorted([[0, 1], [1, 0]]),
            sorted(generate_permutations([0, 1])))

    def test_permuation_of_3(self):
        self.assertEqual(
            sorted([[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]),
            sorted(generate_permutations([1, 2, 3])))
    

if __name__ == '__main__':
    unittest.main(exit=False)
```


### Dec 28, 2020 \[Medium\] Index of Larger Next Number
---
> **Question:** Given a list of numbers, for each element find the next element that is larger than the current element. Return the answer as a list of indices. If there are no elements larger than the current element, then use `-1` instead.

**Example:** 
```py
larger_number([3, 2, 5, 6, 9, 8])
# return [2, 2, 3, 4, -1, -1]
```

**My thoughts:** The idea is to iterate backwards and only store large element along the way with a stack. Doing such will mantain the stack in ascending order. We can then treat the stack as a history of larger element on the right. The algorithm work in the following way:

For each element, we push current element in stack. And the the same time, we pop all elements that are smaller than current element until we find a larger element that is the next larger element in the list.

Note that in worst case scenario, each element can only be pushed and poped from stack once, leaves the time complexity being `O(n)`.


**Solution with Stack:** [https://repl.it/@trsong/Find-the-Index-of-Larger-Next-Number](https://repl.it/@trsong/Find-the-Index-of-Larger-Next-Number)
```py

import unittest

def larger_number(nums):
    stack = []
    n = len(nums)
    res = [-1] * n
    
    for i in xrange(n - 1, -1, -1):
        num = nums[i]
        while stack and nums[stack[-1]] <= num:
            stack.pop()
        
        if stack:
            res[i] = stack[-1]
        
        stack.append(i)
    return res


class LargerNumberSpec(unittest.TestCase):
    def test_example(self):
        nums = [3, 2, 5, 6, 9, 8]
        expected = [2, 2, 3, 4, -1, -1]
        self.assertEqual(expected, larger_number(nums))

    def test_empty_list(self):
        self.assertEqual([], larger_number([]))

    def test_asecending_list(self):
        nums = [0, 1, 2, 2, 3, 3, 3, 4, 5]
        expected = [1, 2, 4, 4, 7, 7, 7, 8, -1]
        self.assertEqual(expected, larger_number(nums))

    def test_descending_list(self):
        nums = [9, 8, 8, 7, 4, 3, 2, 1, 0, -1]
        expected = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.assertEqual(expected, larger_number(nums))

    def test_up_down_up(self):
        nums = [0, 1, 2, 1, 2, 3, 4, 5]
        expected = [1, 2, 5, 4, 5, 6, 7, -1]
        self.assertEqual(expected, larger_number(nums))

    def test_up_down_up2(self):
        nums = [0, 4, -1, 2]
        expected = [1, -1, 3, -1]
        self.assertEqual(expected, larger_number(nums))

    def test_down_up_down(self):
        nums = [9, 5, 6, 3]
        expected = [-1, 2, -1, -1]
        self.assertEqual(expected, larger_number(nums))
    
    def test_up_down(self):
        nums = [11, 21, 31, 3]
        expected = [1, 2, -1, -1]
        self.assertEqual(expected, larger_number(nums))

    def test_random_order(self):
        nums = [4, 3, 5, 2, 4, 7]
        expected = [2, 2, 5, 4, 5, -1]
        self.assertEqual(expected, larger_number(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 27, 2020  LC 218 \[Hard\] City Skyline
---
> **Question:** Given a list of building in the form of `(left, right, height)`, return what the skyline should look like. The skyline should be in the form of a list of `(x-axis, height)`, where x-axis is the point where there is a change in height starting from 0, and height is the new height starting from the x-axis.

**Example:**
```py
Input: [(2, 8, 3), (4, 6, 5)]
Output: [(2, 3), (4, 5), (7, 3), (9, 0)]
Explanation:
             2 2 2
             2   2
         1 1 2 1 2 1 1
         1   2   2   1
         1   2   2   1      

pos: 0 1 2 3 4 5 6 7 8 9
We have two buildings: one has height 3 and the other 5. The city skyline is just the outline of combined looking. 
The result represents the scanned height of city skyline from left to right.
```


**Solution with Priority Queue:** [https://repl.it/@trsong/Find-City-Skyline](https://repl.it/@trsong/Find-City-Skyline)
```py
import unittest
from Queue import PriorityQueue

def scan_city_skyline(buildings):
    bar_set = []
    for left, right, _ in buildings:
         # Critial positions are left end and right end + 1 of each building
        bar_set.append(left)
        bar_set.append(right + 1)

    i = 0
    res = []
    prev_height = 0
    max_heap = PriorityQueue()

    for bar in sorted(bar_set):
        while i < len(buildings):
            # Add all buildings whose starts before the bar
            left, right, height = buildings[i]
            if left > bar:
                break
            max_heap.put((-height, right))
            i += 1
        
        while not max_heap.empty():
            # Remove building that ends before the bar
            _, right = max_heap.queue[0]
            if right >= bar:
               break
            max_heap.get()

        # We want to make sure we get max height of building and bar is between both ends 
        height = abs(max_heap.queue[0][0]) if not max_heap.empty() else 0
        if height != prev_height:
            res.append((bar, height))
            prev_height = height

    return res
    

class ScanCitySkylineSpec(unittest.TestCase):
    def test_example(self):
        """
                     2 2 2
                     2   2
                 1 1 2   2 1 1
                 1   2   2   1
                 1   2   2   1      

        pos: 0 1 2 3 4 5 6 7 8 9
        """
        buildings = [(2, 8, 3), (4, 6, 5)]
        expected = [(2, 3), (4, 5), (7, 3), (9, 0)]
        self.assertEqual(expected, scan_city_skyline(buildings))

    def test_multiple_building_overlap(self):
        buildings = [(2, 9, 10), (3, 7, 15), (5, 12, 12), (15, 20, 10), (19, 24, 8)]
        expected = [(2, 10), (3, 15), (8, 12), (13, 0), (15, 10), (21, 8), (25, 0)]
        self.assertEqual(expected, scan_city_skyline(buildings))

    def test_empty_land(self):
        self.assertEqual([], scan_city_skyline([]))

    def test_length_one_building(self):
        buildings = [(0, 0, 1), (0, 0, 10)]
        self.assertEqual([(0, 10), (1, 0)], scan_city_skyline(buildings))

    def test_upward_staircase(self):
        """
                     4 4 4 4 4
                   3 3 3 3   4
                 2 2 2   3   4
               1 1 1 1 1 3   4
               1 2   2 1 3   4
               1 2   2 1 3   4

        pos: 0 1 2 3 4 5 6 7 8 9
        """
        buildings = [(1, 5, 3), (2, 4, 4), (3, 6, 5), (4, 8, 6)]
        expected = [(1, 3), (2, 4), (3, 5), (4, 6), (9, 0)]
        self.assertEqual(expected, scan_city_skyline(buildings))

    def test_downward_staircase(self):
        """
             1 1 1 1 1 
             1       1
             1 2 2 2 2 2 2
             1 2   3 3 3 3 3 3
             1 2   3 1   2   3

        pos: 0 1 2 3 4 5 6 7 8 9
        """
        buildings = [(0, 4, 5), (1, 6, 3), (3, 8, 2)]
        expected = [(0, 5), (5, 3), (7, 2), (9, 0)]
        self.assertEqual(expected, scan_city_skyline(buildings))

    def test_same_height_overlap_skyline(self):
        """
             1 1 1 2 2 2 2 2 3 3 2 2   4 4 4
             1     2 1       3 3   2   4   4 

        pos: 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6
        """
        buildings = [(0, 4, 2), (3, 11, 2), (8, 9, 2), (13, 15, 2)]
        expected = [(0, 2), (12, 0), (13, 2), (16, 0)]
        self.assertEqual(expected, scan_city_skyline(buildings))

    def test_non_overlap_sky_line(self):
        """    
                                         5 5 5  
               1 1 1                     5   5
               1   1 2 2 2 3 3           5   5
               1   1 2   2 3 3     4 4   5   5
        
        pos: 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7
        """
        buildings = [(1, 3, 3), (4, 6, 2), (7, 8, 2), (11, 12, 1), (14, 16, 4)]
        expected = [(1, 3), (4, 2), (9, 0), (11, 1), (13, 0), (14, 4), (17, 0)]
        self.assertEqual(expected, scan_city_skyline(buildings))

    def test_down_hill_and_up_hill(self):
        """
               1 1 1 1 1         4 4
               1       1         4 4
               1 2 2 2 2 2 2 3 3 4 4 3
               1 2     1   3 2   4 4 3

        pos: 0 1 2 3 4 5 6 7 8 9 0 1 2 3 
        """
        buildings = [(1, 5, 4), (2, 8, 2), (7, 12, 2), (10, 11, 4)]
        expected = [(1, 4), (6, 2), (10, 4), (12, 2), (13, 0)]
        self.assertEqual(expected, scan_city_skyline(buildings))

    def test_height_zero_buildings(self):
        buildings = [(0, 1, 0), (0, 2, 0), (2, 11, 0), (4, 8, 0), (8, 11, 0), (11, 200, 0), (300, 400, 0)]
        expected = []
        self.assertEqual(expected, scan_city_skyline(buildings))


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```


### Dec 26, 2020 \[Medium\] Largest Rectangular Area in a Histogram
---
> **Question:** You are given a histogram consisting of rectangles of different heights. Determine the area of the largest rectangle that can be formed only from the bars of the histogram.

**Example:**
```py
These heights are represented in an input list, such that [1, 3, 2, 5] corresponds to the following diagram:

      x
      x  
  x   x
  x x x
x x x x

For the diagram above, for example, this would be six, representing the 2 x 3 area at the bottom right.
```

**Solution with Stack:** [https://repl.it/@trsong/Find-Largest-Rectangular-Area-in-a-Histogram](https://repl.it/@trsong/Find-Largest-Rectangular-Area-in-a-Histogram)
```py
import unittest

def largest_rectangle_in_histogram(histogram):
    stack = []
    res = 0
    n = len(histogram)
    i = 0

    while i < n or stack:
        if not stack or i < n and histogram[stack[-1]] < histogram[i]:
            # mantain stack in non-descending order
            stack.append(i)
            i += 1
        else:
            # if stack starts decreasing,
            # then left boundary must be stack[-2] and right boundary must be i. Note both boundaries are exclusive
            # and height is stack[-1]
            height = histogram[stack.pop()]
            left_bound = stack[-1] if stack else -1  #
            width = i - left_bound - 1
            area = height * width
            res = max(res, area)
    return res


class LargestRectangleInHistogramSpec(unittest.TestCase):
    def test_example(self):
        """
              x
              x  
          x   x
          X X X
        x X X X
        """
        histogram = [1, 3, 2, 5]
        expected = 2 * 3
        self.assertEqual(expected, largest_rectangle_in_histogram(histogram))

    def test_empty_histogram(self):
        self.assertEqual(0, largest_rectangle_in_histogram([]))

    def test_width_one_rectangle(self):
        """
              X
              X
              X
              X 
          x   X
          x x X
        x x x X
        """
        histogram = [1, 3, 2, 7]
        expected = 7 * 1
        self.assertEqual(expected, largest_rectangle_in_histogram(histogram))

    def test_ascending_sequence(self):
        """
                x  
              x x
            X X X
          x X X X
        """
        histogram = [0, 1, 2, 3, 4]
        expected = 2 * 3
        self.assertEqual(expected, largest_rectangle_in_histogram(histogram))

    def test_descending_sequence(self):
        """
        x  
        X X
        X X x    
        X X x x
        """
        histogram = [4, 3, 2, 1, 0]
        expected = 3 * 2
        self.assertEqual(expected, largest_rectangle_in_histogram(histogram))

    def test_sequence3(self):
        """
            x
          x x x 
        X X X X X 
        X X X X X
        X X X X X
        """       
        histogram = [3, 4, 5, 4, 3]
        expected = 3 * 5
        self.assertEqual(expected, largest_rectangle_in_histogram(histogram))

    def test_sequence4(self):
        """
          x   x   x   x
          x   x   x   x
          x   x   x   x
          x   x   x   x
          x   x x x   x
          x   x x x   x
          X X X X X X X 
        x X X X X X X X x
        x X X X X X X X x
        x X X X X X X X x
        """      
        histogram = [3, 10, 4, 10, 5, 10, 4, 10, 3]
        expected = 4 * 7
        self.assertEqual(expected, largest_rectangle_in_histogram(histogram))

    def test_sequence5(self):
        """
        x           x
        x   x   x   x
        x   X X X   x
        x   X X X   x
        x x X X X   x
        x x X X X x x 
        """
        histogram = [6, 2, 5, 4, 5, 1, 6]
        expected = 4 * 3
        self.assertEqual(expected, largest_rectangle_in_histogram(histogram))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 25, 2020 \[Medium\] Longest Consecutive Sequence in an Unsorted Array
---
> **Question:** Given an array of integers, return the largest range, inclusive, of integers that are all included in the array.
>
> For example, given the array `[9, 6, 1, 3, 8, 10, 12, 11]`, return `(8, 12)` since `8, 9, 10, 11, and 12` are all in the array.


**My thoughts:** We can use DFS to find max length of consecutive sequence. Two number are neighbor if they differ by 1. Thus for any number we can scan though its left-most and right-most neighbor recursively. 

**Solution with DFS:** [https://repl.it/@trsong/Longest-Consecutive-Sequence](https://repl.it/@trsong/Longest-Consecutive-Sequence)
```py
import unittest

def longest_consecutive_seq(nums):
    if not nums:
        return None

    num_set = set(nums)
    max_len = 0
    res = None
    for center in nums:
        if center not in num_set:
            continue
        num_set.remove(center)
        lower = center - 1
        higher = center + 1
        while lower in num_set:
            lower -= 1

        while higher in num_set:
            higher += 1

        len = higher - lower - 1
        if len > max_len:
            res = (lower + 1, higher - 1)
            max_len = len
    return res
        

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
    unittest.main(exit=False, verbosity=2)
```

### Dec 24, 2020 \[Easy\] Smallest Sum Not Subset Sum
---
> **Question:** Given a sorted list of positive numbers, find the smallest positive number that cannot be a sum of any subset in the list.

**Example:**
```py
Input: [1, 2, 3, 8, 9, 10]
Output: 7
Numbers 1 to 6 can all be summed by a subset of the list of numbers, but 7 cannot.
```

**My thoughts:** Suppose the array is sorted and all elements are positive, then max positive subset sum is the `prefix_sum` of the array. Thus the min number subset sum cannot reach is `prefix_sum + 1`. 

We can prove above statement by Math Induction. 

**Case 1:** We want to show that if each elem is less than `prefix_sum`, the subset sum range from `0` to `prefix_sum` of array: 
- Base case: for empty array, subset sum max is `0` which equals prefix sum `0`. 
- Inductive Hypothesis: for the `i-th element`, if i-th element smaller than prefix sum, then subset sum range is `0` to `prefix_sum[i]` ie. sum(nums[0..i]).
- Induction Step: upon `i-th` step, the range is `0` to `prefix_sum[i]`. If the `(i + 1)-th` element `nums[i + 1]` is within that range, then smaller subset sum is still `0`. Largest subset sum is `prefix_sum[i] + nums[i + 1]` which equals `prefix_sum[i + 1]`

**Case 2:** If the i-th element is greater than `prefix_sum`, then we can omit the result of element as there is a hole in that range. 
Because the previous subset sum ranges from `[0, prefix_sum]`, then for `nums[i] > prefix_sum`, there is a hole that `prefix sum + 1` cannot be covered with the introduction of new element.

As the max positive sum we can reach is prefix sum, the min positive subset sum we cannot reach is prefix sum + 1. 


**Solution with DP:** [https://repl.it/@trsong/Smallest-Sum-Not-Subset-Sum](https://repl.it/@trsong/Smallest-Sum-Not-Subset-Sum)
```py
import unittest

def smallest_non_subset_sum(nums):
    # Initially sum has range [0, 1)
    upper_bound = 1
    for num in nums:
        if num <= upper_bound:
            # Previously sum has range [0, upper_bound), 
            # with introduction of num, that range becomes
            # new range [0, upper_bound + num)
            upper_bound += num
        else:
            return upper_bound
    return upper_bound


class SmallestNonSubsetSumSpec(unittest.TestCase):
    def test_example(self):
        nums = [1, 2, 3, 8, 9, 10]
        expected = 7
        self.assertEqual(expected, smallest_non_subset_sum(nums))

    def test_empty_array(self):
        nums = []
        expected = 1
        self.assertEqual(expected, smallest_non_subset_sum(nums))

    def test_first_num_not_one(self):
        nums = [2]
        expected = 1
        self.assertEqual(expected, smallest_non_subset_sum(nums))

    def test_array_with_duplicated_numbers(self):
        nums = [1, 1, 1, 1, 1]
        expected = 6
        self.assertEqual(expected, smallest_non_subset_sum(nums))

    def test_result_larger_than_sum_of_all(self):
        nums = [1, 1, 3, 4]
        expected = 10
        self.assertEqual(expected, smallest_non_subset_sum(nums))

    def test_result_larger_than_sum_of_all2(self):
        nums = [1, 2, 3, 4, 5, 6]
        expected = 22
        self.assertEqual(expected, smallest_non_subset_sum(nums))

    def test_result_smaller_than_max(self):
        nums = [1, 3, 6, 10, 11, 15]
        expected = 2
        self.assertEqual(expected, smallest_non_subset_sum(nums))

    def test_result_smaller_than_max2(self):
        nums = [1, 2, 5, 10, 20, 40]
        expected = 4
        self.assertEqual(expected, smallest_non_subset_sum(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 23, 2020 LT 879 \[Medium\] NBA Playoff Matches
---
> **Question:** During the NBA playoffs, we always arrange the rather strong team to play with the rather weak team, like make the rank 1 team play with the rank nth team, which is a good strategy to make the contest more interesting. Now, you're given n teams, and you need to output their final contest matches in the form of a string.
>
> The n teams are given in the form of positive integers from 1 to n, which represents their initial rank. (Rank 1 is the strongest team and Rank n is the weakest team.) We'll use parentheses () and commas , to represent the contest team pairing - parentheses () for pairing and commas , for partition. During the pairing process in each round, you always need to follow the strategy of making the rather strong one pair with the rather weak one.
>
> We ensure that the input n can be converted into the form `2^k`, where k is a positive integer.

**Example 1:**

```py
Input: 2
Output: "(1,2)"
```

**Example 2:**
```py
Input: 4
Output: "((1,4),(2,3))"
Explanation: 
  In the first round, we pair the team 1 and 4, the team 2 and 3 together, as we need to make the strong team and weak team together.
  And we got (1,4),(2,3).
  In the second round, the winners of (1,4) and (2,3) need to play again to generate the final winner, so you need to add the paratheses outside them.
  And we got the final answer ((1,4),(2,3)).
```

**Example 3:**
```py
Input: 8
Output: "(((1,8),(4,5)),((2,7),(3,6)))"
Explanation:
  First round: (1,8),(2,7),(3,6),(4,5)
  Second round: ((1,8),(4,5)),((2,7),(3,6))
  Third round: (((1,8),(4,5)),((2,7),(3,6)))
```

**Solution:** [https://repl.it/@trsong/Print-NBA-Playoff-Matches](https://repl.it/@trsong/Print-NBA-Playoff-Matches)
```py
import unittest

def NBA_Playoff_Matches(n):
    res = map(str, range(1, n + 1))
    while n > 1:
        for i in xrange(n):
            res[i] = "(%s,%s)" % (res[i], res[n - i - 1])
        n //= 2
    return res[0]


class NBAPlayoffMatcheSpec(unittest.TestCase):
    def test_2_teams(self):
        expected = "(1,2)"
        self.assertEqual(expected, NBA_Playoff_Matches(2))

    def test_4_teams(self):
        expected = "((1,4),(2,3))"
        self.assertEqual(expected, NBA_Playoff_Matches(4))

    def test_8_teams(self):
        expected = "(((1,8),(4,5)),((2,7),(3,6)))"
        self.assertEqual(expected, NBA_Playoff_Matches(8))

    def test_16_teams(self):
        # round1: (1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)
        # round2: ((1, 16), (8, 9)), ((2, 15), (7, 10)), ((3, 14), (6, 11)), ((4, 13), (5, 12))
        # round3: ((((1,16),(8,9)),((4,13),(5,12))),(((2,15),(7,10)),((3,14),(6,11))))
        expected = "((((1,16),(8,9)),((4,13),(5,12))),(((2,15),(7,10)),((3,14),(6,11))))"
        self.assertEqual(expected, NBA_Playoff_Matches(16))


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```

### Dec 22, 2020 LC 301 \[Hard\] Remove Invalid Parentheses
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

**Solution with Backtracking:** [https://repl.it/@trsong/Ways-to-Remove-Invalid-Parentheses](https://repl.it/@trsong/Ways-to-Remove-Invalid-Parentheses)
```py
import unittest

def remove_invalid_parenthese(s):
    invalid_open = invalid_close = 0
    for ch in s:
        if invalid_open == 0 and ch == ')':
            invalid_close += 1
        elif ch == '(':
            invalid_open += 1
        elif ch == ')':
            invalid_open -= 1

    res = []
    backtrack(s, res, 0, invalid_open, invalid_close)
    return res


def backtrack(s, res, next_index, invalid_open, invalid_close):
    if invalid_open == invalid_close == 0:
        if is_valid(s):
            res.append(s)
    else:
        for i in xrange(next_index, len(s)):
            if i > next_index and s[i] == s[i - 1]:
                continue
            elif s[i] == '(' and invalid_open > 0:
                backtrack(s[:i] + s[i + 1:], res, i, invalid_open - 1,
                          invalid_close)
            elif s[i] == ')' and invalid_close > 0:
                backtrack(s[:i] + s[i + 1:], res, i, invalid_open,
                          invalid_close - 1)


def is_valid(s):
    count = 0
    for ch in s:
        if count < 0:
            return False
        elif ch == '(':
            count += 1
        elif ch == ')':
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


### Dec 21, 2020 \[Easy\] Invalid Parentheses to Remove 
---
> **Question:** Given a string of parentheses, write a function to compute the minimum number of parentheses to be removed to make the string valid (i.e. each open parenthesis is eventually closed).
>
> For example, given the string `"()())()"`, you should return `1`. Given the string `")("`, you should return `2`, since we must remove all of them.

**Solution:** [https://repl.it/@trsong/Count-Invalid-Parentheses-to-Remove](https://repl.it/@trsong/Count-Invalid-Parentheses-to-Remove)
```py

import unittest

def count_invalid_parentheses(input_str):
    balance = 0
    invalid = 0
    for ch in input_str:
        if ch == '(':
            balance += 1
        elif balance > 0:
            balance -= 1
        else:
            invalid += 1
    return balance + invalid


class CountInvalidParentheseSpec(unittest.TestCase):
    def test_incomplete_parentheses(self):
        self.assertEqual(1, count_invalid_parentheses("(()"))
    
    def test_incomplete_parentheses2(self):
        self.assertEqual(2, count_invalid_parentheses("(()("))
    
    def test_overflown_close_parentheses(self):
        self.assertEqual(2, count_invalid_parentheses("()))"))
    
    def test_overflown_close_parentheses2(self):
        self.assertEqual(2, count_invalid_parentheses(")()("))

    def test_valid_parentheses(self):
        self.assertEqual(0, count_invalid_parentheses("((()))"))
    
    def test_valid_parentheses2(self):
        self.assertEqual(0, count_invalid_parentheses("()()()"))
    
    def test_valid_parentheses3(self):
        self.assertEqual(0, count_invalid_parentheses("((())(()))"))

    def test_valid_parentheses4(self):
        self.assertEqual(0, count_invalid_parentheses(""))


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False) 
```


### Dec 20, 2020 \[Easy\] Longest Consecutive 1s in Binary Representation
---
> **Question:**  Given an integer n, return the length of the longest consecutive run of 1s in its binary representation.

**Example:**
```py
Input: 156
Output: 3
Exaplanation: 156 in binary is 10011100
```

**My thoughts:** For any number `x`, bitwise AND itself with `x << 1` eliminates last 1 for each consecutive 1's:

```py
   10011100 & 
  10011100
= 000011000

And 

   11000 &
  1100
= 010000 
``` 

We can keep doing this until all 1's are exhausted. 

**Solution:** [https://repl.it/@trsong/Longest-Consecutive-1s-in-Binary-Representation](https://repl.it/@trsong/Longest-Consecutive-1s-in-Binary-Representation)
```py
import unittest

def count_longest_consecutive_ones(num):
    count = 0
    while num > 0:
        num &= num << 1
        count += 1
    return count


class CountLongestConsecutiveOneSpec(unittest.TestCase):
    def test_example(self):
        expected, num = 3, 0b10011100
        self.assertEqual(expected, count_longest_consecutive_ones(num))

    def test_zero(self):
        expected, num = 0, 0b0
        self.assertEqual(expected, count_longest_consecutive_ones(num))

    def test_one(self):
        expected, num = 1, 0b1
        self.assertEqual(expected, count_longest_consecutive_ones(num))

    def test_every_other_one(self):
        expected, num = 1, 0b1010101
        self.assertEqual(expected, count_longest_consecutive_ones(num))

    def test_all_ones(self):
        expected, num = 5, 0b11111
        self.assertEqual(expected, count_longest_consecutive_ones(num))

    def test_should_return_longest(self):
        expected, num = 4, 0b10110111011110111010101
        self.assertEqual(expected, count_longest_consecutive_ones(num))

    def test_consecutive_zeros(self):
        expected, num = 2, 0b100010001100010010001001
        self.assertEqual(expected, count_longest_consecutive_ones(num))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 19, 2020 \[Medium\] Craft Sentence
---
> **Question:** Write an algorithm to justify text. Given a sequence of words and an integer line length k, return a list of strings which represents each line, fully justified.
> 
> More specifically, you should have as many words as possible in each line. There should be at least one space between each word. Pad extra spaces when necessary so that each line has exactly length k. Spaces should be distributed as equally as possible, with the extra spaces, if any, distributed starting from the left.
> 
> If you can only fit one word on a line, then you should pad the right-hand side with spaces.
> 
> Each word is guaranteed not to be longer than k.
>
> For example, given the list of words `["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]` and `k = 16`, you should return the following:

```py
["the  quick brown",
 "fox  jumps  over",
 "the   lazy   dog"]
```

**Solution:** [https://repl.it/@trsong/Craft-Sentence-and-Adjust-Text-Width](https://repl.it/@trsong/Craft-Sentence-and-Adjust-Text-Width)
```py

import unittest

def craft_sentence(words, k):
    buffer = []
    res = []
    sentence_size = 0
    for word in words:
        if sentence_size + len(word) > k:
            sentence = craft_one_sentence(buffer, k)
            res.append(sentence)
            buffer = []
            sentence_size = 0
        
        buffer.append(word)
        sentence_size += len(word) + 1
    
    if buffer:
        res.append(craft_one_sentence(buffer, k))

    return res


def craft_one_sentence(words, k):
    n = len(words)
    if n == 1:
        return words[0] + " " * (k - len(words[0]))

    num_char = sum(map(len, words))
    white_spaces = k - num_char
    padding = white_spaces // (n - 1) 
    extra_padding = white_spaces % (n - 1)

    res = [words[0]]
    for i in xrange(1, n):
        res.append(" " * padding)
        if extra_padding > 0:
            res.append(" ")
            extra_padding -= 1
        res.append(words[i])
    return ''.join(res)


class CraftSentenceSpec(unittest.TestCase):
    def test_fit_only_one_word(self):
        k, words = 7, ["test", "same", "length", "string"]
        expected = [
            "test   ",
            "same   ",
            "length ",
            "string "]
        self.assertEqual(expected, craft_sentence(words, k))
    
    def test_fit_only_one_word2(self):
        k, words = 6, ["test", "same", "length", "string"]
        expected = [
            "test  ",
            "same  ",
            "length",
            "string"]
        self.assertEqual(expected, craft_sentence(words, k))

    def test_no_padding(self):
        k, words = 2, ["to", "be"]
        expected = ["to", "be"]
        self.assertEqual(expected, craft_sentence(words, k))

    def test_fit_two_words(self):
        k, words = 6, ["To", "be", "or", "not", "to", "be"]
        expected = [
            "To  be",
            "or not",
            "to  be"]
        self.assertEqual(expected, craft_sentence(words, k))

    def test_fit_two_words2(self):
        k, words = 11, ["Greed", "is", "not", "good"]
        expected = [
            "Greed    is",
            "not    good"]
        self.assertEqual(expected, craft_sentence(words, k))

    def test_fit_more_words(self):
        k, words = 16, ["the", "quick", "brown", "fox", "jumps",
        "over", "the", "lazy", "dog"]
        expected = [
            "the  quick brown",
            "fox  jumps  over",
            "the   lazy   dog"]
        self.assertEqual(expected, craft_sentence(words, k))

    def test_longer_sentence(self):
        k, words = 16, ["This", "is", "an", "example", "of", "text", "justification."]
        expected = [
            "This    is    an",
            "example  of text",
            "justification.  "]
        self.assertEqual(expected, craft_sentence(words, k))

    def test_longer_sentence2(self):
        k, words = 16, ["What", "must", "be", "acknowledgment",
         "shall", "be"]
        expected = [
            "What   must   be",
            "acknowledgment  ",
            "shall         be"]
        self.assertEqual(expected, craft_sentence(words, k))

    def test_longer_sentence3(self):
        k, words = 20, ["Science", "is", "what", "we", "understand", 
        "well", "enough", "to", "explain", "to", "a", "computer.", 
        "Art", "is", "everything", "else", "we", "do"]
        expected = [
            "Science  is  what we",
            "understand      well",
            "enough to explain to",
            "a  computer.  Art is",
            "everything  else  we",
            "do                  "]
        self.assertEqual(expected, craft_sentence(words, k))


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```

### Dec 18, 2020 \[Medium\] Reverse Coin Change
---
> **Question:** You are given an array of length `N`, where each element `i` represents the number of ways we can produce `i` units of change. For example, `[1, 0, 1, 1, 2]` would indicate that there is only one way to make `0, 2, or 3` units, and two ways of making `4` units.
>
> Given such an array, determine the denominations that must be in use. In the case above, for example, there must be coins with value `2, 3, and 4`.


**Thoughts:** Thinking backwards. Given a base, we use `dp[num] += dp[num - base_num] for all base_num in base`. Likewise, whenver we discover a base_num, `dp[num] -= dp[num - base_num]` to eliminate base's effect.

**Solution with DP:** [https://repl.it/@trsong/Reverse-Coin-Change](https://repl.it/@trsong/Reverse-Coin-Change)
```py
import unittest

def reverse_coin_change(coin_ways):
    if not coin_ways:
        return []

    n = len(coin_ways)
    base = []
    for base_num in xrange(1, n):
        if coin_ways[base_num] == 0:
            continue
        base.append(base_num)

        for num in xrange(n - 1, base_num - 1, -1):
            coin_ways[num] -= coin_ways[num - base_num]
    return base


class ReverseCoinChangeSpec(unittest.TestCase):
    @staticmethod
    def generate_coin_ways(base, size=None):
        max_num = size if size is not None else max(base)
        coin_ways = [0] * (max_num + 1)
        coin_ways[0] = 1
        for base_num in base:
            for num in xrange(base_num, max_num + 1):
                coin_ways[num] += coin_ways[num - base_num]
        return coin_ways  

    def test_example(self):
        coin_ways = [1, 0, 1, 1, 2]
        # 0: 0
        # 1: 
        # 2: one 2
        # 3: one 3
        # 4: two 2's or one 4
        # Therefore: [2, 3, 4] as base produces above coin ways
        expected = [2, 3, 4]
        self.assertEqual(expected, reverse_coin_change(coin_ways))

    def test_empty_input(self):
        self.assertEqual([], reverse_coin_change([]))

    def test_empty_base(self):
        self.assertEqual([], reverse_coin_change([1, 0, 0, 0, 0, 0]))

    def test_one_number_base(self):
        coin_ways = [1, 1, 1, 1, 1, 1]
        expected = [1]
        self.assertEqual(expected, reverse_coin_change(coin_ways))

    def test_prime_number_base(self):
        # ReverseCoinChangeSpec.generate_coin_ways([2, 3, 5, 7], 10)
        coin_ways = [1, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5]
        expected = [2, 3, 5, 7]
        self.assertEqual(expected, reverse_coin_change(coin_ways))

    def test_composite_base(self):
        # ReverseCoinChangeSpec.generate_coin_ways([2, 4, 6], 10)
        coin_ways = [1, 0, 1, 0, 2, 0, 3, 0, 5, 0, 6]
        expected = [2, 4, 6, 8]
        self.assertEqual(expected, reverse_coin_change(coin_ways))

    def test_all_number_bases(self):
        # ReverseCoinChangeSpec.generate_coin_ways([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        coin_ways = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42]
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(expected, reverse_coin_change(coin_ways))


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```


### Dec 17, 2020 LC 352 \[Hard\] Data Stream as Disjoint Intervals
---
> **Question:** Given a data stream input of non-negative integers `a1, a2, ..., an, ...`, summarize the numbers seen so far as a list of disjoint intervals.
>
> For example, suppose the integers from the data stream are `1, 3, 7, 2, 6, ...`, then the summary will be:

```py
[1, 1]
[1, 1], [3, 3]
[1, 1], [3, 3], [7, 7]
[1, 3], [7, 7]
[1, 3], [6, 7]
```

**My thoughts:** Using heap to store all intervals. When get intervals, pop element with smallest start time one by one, there are only 3 cases:
- Element Overlapping w/ previous interval, then we do nothing
- Element will update existing interval's start or end time, then we update the interval
- Element will cause two intervals to merge.

**Solution with Priority Queue:** [https://repl.it/@trsong/Print-Data-Stream-as-Disjoint-Intervals](https://repl.it/@trsong/Print-Data-Stream-as-Disjoint-Intervals)
```py
import unittest
from Queue import PriorityQueue

class SummaryRanges(object):
    def __init__(self):
      self.min_heap = PriorityQueue()  
      self.num_set = set()

    def add_num(self, val):
        if val not in self.num_set:
            self.num_set.add(val)
            self.min_heap.put((val, [val, val]))

    def get_intervals(self):
        res = []
        while not self.min_heap.empty():
            _, interval = self.min_heap.get()
            prev = res[-1] if res else None
            if prev is None or prev[1] + 1 < interval[0]:
                res.append(interval)
            else:
                prev[1] = max(prev[1], interval[1])

        for interval in res:
            self.min_heap.put((interval[0], interval))
        return res


class SummaryRangesSpec(unittest.TestCase):
    def test_sample(self):
        sr = SummaryRanges()
        sr.add_num(1)
        self.assertEqual([[1, 1]], sr.get_intervals())
        sr.add_num(3)
        self.assertEqual([[1, 1], [3, 3]], sr.get_intervals())
        sr.add_num(7)
        self.assertEqual([[1, 1], [3, 3], [7, 7]], sr.get_intervals())
        sr.add_num(2)
        self.assertEqual([[1, 3], [7, 7]], sr.get_intervals())
        sr.add_num(6)
        self.assertEqual([[1, 3], [6, 7]], sr.get_intervals())

    def test_none_overlapping(self):
        sr = SummaryRanges()
        sr.add_num(3)
        sr.add_num(1)
        sr.add_num(5)
        self.assertEqual([[1, 1], [3, 3], [5, 5]], sr.get_intervals())

    def test_val_in_existing_intervals(self):
        sr = SummaryRanges()
        sr.add_num(3)
        sr.add_num(2)
        sr.add_num(1)
        sr.add_num(5)
        sr.add_num(6)
        sr.add_num(7)
        self.assertEqual([[1, 3], [5, 7]], sr.get_intervals())
        sr.add_num(6)
        self.assertEqual([[1, 3], [5, 7]], sr.get_intervals())

    def test_val_join_two_intervals(self):
        sr = SummaryRanges()
        sr.add_num(3)
        sr.add_num(2)
        sr.add_num(1)
        sr.add_num(5)
        sr.add_num(6)
        self.assertEqual([[1, 3], [5, 6]], sr.get_intervals())
        sr.add_num(4)
        self.assertEqual([[1, 6]], sr.get_intervals())


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```

### Dec 16, 2020 \[Easy\] Max and Min with Limited Comparisons
---
> **Question:** Given a list of numbers of size `n`, where `n` is greater than `3`, find the maximum and minimum of the list using less than `2 * (n - 1)` comparisons.

**Example:**
```py
Input: [3, 5, 1, 2, 4, 8]
Output: (1, 8)
```

**My thoughts:** The idea is to use Tournament Method. Think about each number as a team in the tournament. One team, zero matches. Two team, one match. N team, let's break them into half and use two matches to get best of best and worest of worest:

```
T(n) = 2 * T(n/2) + 2
T(1) = 0
T(2) = 1

=>
T(n) = 3n/2 - 2 
```

**Solution with Divide And Conquer:** [https://repl.it/@trsong/Find-the-Max-and-Min-with-Limited-Comparisons](https://repl.it/@trsong/Find-the-Max-and-Min-with-Limited-Comparisons)
```py
import unittest


def find_min_max(nums):
    if not nums:
        return None
    return find_min_max_recur(nums, 0, len(nums) - 1)


def find_min_max_recur(nums, i, j):
    if j - i <= 1:
        return (nums[i], nums[j]) if nums[i] < nums[j] else (nums[j], nums[i])

    mid = i + (j - i) // 2
    left_min, left_max = find_min_max_recur(nums, i, mid)
    right_min, right_max = find_min_max_recur(nums, mid + 1, j)

    return min(left_min, right_min), max(left_max, right_max)


class GetMinMaxSpec(unittest.TestCase):
    def assert_find_min_max(self, nums):
        min_val, max_val = min(nums), max(nums)
        self.assertEqual((min_val, max_val), find_min_max(nums))

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


### Dec 15, 2020 LC 307 \[Medium\] Range Sum Query - Mutable
---
> **Question:** Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.
>
> The update(i, val) function modifies nums by updating the element at index i to val.

**Example:**
```py
Given nums = [1, 3, 5]

sumRange(0, 2) -> 9
update(1, 2)
sumRange(0, 2) -> 8
```

**Solution with Segment Tree:** [https://repl.it/@trsong/Mutable-Range-Sum-Query](https://repl.it/@trsong/Mutable-Range-Sum-Query)
```py
import unittest

class RangeSumQuery(object):
    def __init__(self, nums):
        self.size = len(nums)
        self.tree = RangeSumQuery.build_segment_tree(nums)

    @staticmethod
    def build_segment_tree(nums):
        if not nums:
            return []
        
        n = len(nums)
        tree = [0] * n + nums
        for i in xrange(n-1, -1, -1):
            tree[i] = tree[2 * i] + tree[2 * i + 1]

        return tree

    def update(self, i, val):
        pos = i + self.size
        self.tree[pos] = val
        while pos > 0:
            left = pos
            right = pos
            if pos % 2 == 0:
                # pos is left child
                right = pos + 1
            else:
                # pos is right child
                left = pos - 1
            
            # parent is updated after child is updated
            self.tree[pos // 2] = self.tree[left] + self.tree[right]
            pos //= 2

    def range_sum(self, i, j):
        #              1~7
        #           /       \  
        #          /         \
        #         1~3        4~7
        #        /   \      /   \
        #       1    2~3  4~5   6~7
        #           /  |  | |  /   \
        #          2   3  4 5 6     7
        n = self.size
        l, r = i + n, j + n
        sum = 0

        while l <= r:
            # case1: suppose l is right child that represents 2~3, we sum value(2~3) then move l right to 4~5
            # case2: suppose l is left child do nothing
            if l % 2 == 1:
                # l bound is right child
                sum += self.tree[l]
                l += 1
            
            # case3: suppose r is right child that represents 6~7, do nothing
            # case4: suppose r is left child that represents 4~5, we sum value(4~5) then move r left to 2~3
            if r % 2 == 0:
                # r bound is left child
                sum += self.tree[r]
                r -= 1
            
            # Mov l and r to parent and continue
            l //=2
            r //=2
        return sum


class RangeSumQuerySpec(unittest.TestCase):
    def test_example(self):
        rsq = RangeSumQuery([1, 3, 5])
        self.assertEqual(rsq.range_sum(0, 2), 9)
        rsq.update(1, 2)
        self.assertEqual(rsq.range_sum(0, 2), 8)

    def test_one_elem_array(self):
        rsq = RangeSumQuery([8])
        rsq.update(0, 2)
        self.assertEqual(rsq.range_sum(0, 0), 2)

    def test_update_all_elements(self):
        req = RangeSumQuery([1, 4, 2, 3])
        self.assertEqual(req.range_sum(0, 3), 10)
        req.update(0, 0)
        req.update(2, 0)
        req.update(1, 0)
        req.update(3, 0)
        self.assertEqual(req.range_sum(0, 3), 0)
        req.update(2, 1)
        self.assertEqual(req.range_sum(0, 1), 0)
        self.assertEqual(req.range_sum(1, 2), 1)
        self.assertEqual(req.range_sum(2, 3), 1)
        self.assertEqual(req.range_sum(3, 3), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```

### Dec 14, 2020 LC 554 \[Medium\] Brick Wall
---
> **Question:** A wall consists of several rows of bricks of various integer lengths and uniform height. Your goal is to find a vertical line going from the top to the bottom of the wall that cuts through the fewest number of bricks. If the line goes through the edge between two bricks, this does not count as a cut.
>
> For example, suppose the input is as follows, where values in each row represent the lengths of bricks in that row:

```py
[[3, 5, 1, 1],
 [2, 3, 3, 2],
 [5, 5],
 [4, 4, 2],
 [1, 3, 3, 3],
 [1, 1, 6, 1, 1]]
```
>
> The best we can we do here is to draw a line after the eighth brick, which will only require cutting through the bricks in the third and fifth row.
>
> Given an input consisting of brick lengths for each row such as the one above, return the fewest number of bricks that must be cut to create a vertical line.


**Solution:** [https://repl.it/@trsong/Min-Cut-of-Wall-Bricks](https://repl.it/@trsong/Min-Cut-of-Wall-Bricks)
```py
import unittest

def least_cut_bricks(wall):
    histogram = {}
    for row in wall:
        pos = 0
        for i in xrange(len(row) - 1):
            pos += row[i]
            histogram[pos] = histogram.get(pos, 0) + 1

    max_pos = max(histogram.values()) if histogram else 0
    return len(wall) - max_pos
    

class LeastCutBrickSpec(unittest.TestCase):
    def test_example(self):
        wall = [
            [3, 5, 1, 1],
            [2, 3, 3, 2],
            [5, 5],
            [4, 4, 2],
            [1, 3, 3, 3],
            [1, 1, 6, 1, 1]]
        self.assertEqual(2, least_cut_bricks(wall))  # cut at col 8
    
    def test_empty_wall(self):
        self.assertEqual(0, least_cut_bricks([]))
    
    def test_properly_align_all_bricks(self):
        wall = [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]
        ]
        self.assertEqual(0, least_cut_bricks(wall))

    def test_one_column_properly_aligned(self):
        wall = [
            [1, 1, 1, 1],
            [2, 2],
            [1, 1, 2],
            [2, 1, 1]
        ]
        self.assertEqual(0, least_cut_bricks(wall))  # cut at col 2

    def test_local_answer_is_not_optimal(self):
        wall = [
            [1, 1, 2, 1],
            [3, 2],
            [2, 3]
        ]
        self.assertEqual(1, least_cut_bricks(wall))  # cut at col 2


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```

### Dec 13, 2020 LC 240 \[Medium\] Search a 2D Matrix II
---
> **Question:** Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

- Integers in each row are sorted in ascending from left to right.
- Integers in each column are sorted in ascending from top to bottom.

**Example:**

```py
Consider the following matrix:

[
  [ 1,  4,  7, 11, 15],
  [ 2,  5,  8, 12, 19],
  [ 3,  6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
Given target = 5, return True.
Given target = 20, return False.
```

**My thoughts:** This problem can be solved using Divide and Conquer. First we find a mid-point (mid of row and column point). We break the matrix into 4 sub-matrices: top-left, top-right, bottom-left, bottom-right. And notice the following properties:
1. number in top-left matrix is strictly is **less** than mid-point 
2. number in bottom-right matrix is strictly **greater** than mid-point
3. number in the other two could be **greater** or **smaller** than mid-point, we cannot say until find out

So each time when we find a mid-point in recursion, if target number is greater than mid-point then we can say that it cannot be in top-left matrix (property 1). So we check all other 3 matrices except top-left.

Or if the target number is smaller than mid-point then we can say it cannot be in bottom-right matrix (property 2). We check all other 3 matrices except bottom-right.

Therefore we have `T(mn) = 3/4 * (mn/4) + O(1)`. By Master Theorem, the time complexity is `O(log(mn)) = O(log(m) + log(n))`

**Solution with Divide and Conquer:** [https://repl.it/@trsong/Search-in-a-Sorted-2D-Matrix](https://repl.it/@trsong/Search-in-a-Sorted-2D-Matrix)
```py
import unittest

class Square(object):
    def __init__(self, rlo, rhi, clo, chi):
        self.rlo = rlo
        self.rhi = rhi
        self.clo = clo
        self.chi = chi

def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False

    n, m = len(matrix), len(matrix[0])
    stack = [Square(0, n - 1, 0, m - 1)]
    while stack:
        sqr = stack.pop()
        if sqr.rlo > sqr.rhi or sqr.clo > sqr.chi:
            continue

        rmid = sqr.rlo + (sqr.rhi - sqr.rlo) // 2
        cmid = sqr.clo + (sqr.chi - sqr.clo) // 2

        if matrix[rmid][cmid] == target:
            return True
        elif matrix[rmid][cmid] < target:
            # target not in top left
            bottom_left = Square(rmid + 1, sqr.rhi, sqr.clo, cmid)
            right = Square(sqr.rlo, sqr.rhi, cmid + 1, sqr.chi)
            stack.extend([bottom_left, right])
        else:
            # target not in bottom right
            top_right = Square(sqr.rlo, rmid - 1, cmid, sqr.chi)
            left = Square(sqr.rlo, sqr.rhi, sqr.clo, cmid - 1)
            stack.extend([top_right, left])

    return False


class SearchMatrixSpec(unittest.TestCase):
    def test_empty_matrix(self):
        self.assertFalse(search_matrix([], target=0))
        self.assertFalse(search_matrix([[]], target=0))

    def test_example(self):
        matrix = [
            [ 1, 4, 7,11,15],
            [ 2, 5, 8,12,19],
            [ 3, 6, 9,16,22],
            [10,13,14,17,24],
            [18,21,23,26,30]
        ]
        self.assertTrue(search_matrix(matrix, target=5))
        self.assertFalse(search_matrix(matrix, target=20))

    def test_mid_less_than_top_right(self):
        matrix = [
            [ 1, 2, 3, 4, 5],
            [ 6, 7, 8, 9,10],
            [11,12,13,14,15],
            [16,17,18,19,20],
            [21,22,23,24,25]
        ]
        self.assertTrue(search_matrix(matrix, target=5))

    def test_mid_greater_than_top_right(self):
        matrix = [
            [5 , 6,10,14],
            [6 ,10,13,18],
            [10,13,18,19]
        ]
        self.assertTrue(search_matrix(matrix, target=14))

    def test_mid_less_than_bottom_right(self):
        matrix = [
            [1,4],
            [2,5]
        ]
        self.assertTrue(search_matrix(matrix, target=5))

    def test_element_out_of_matrix_range(self):
        matrix = [
            [ 1, 4, 7,11,15],
            [ 2, 5, 8,12,19],
            [ 3, 6, 9,16,22],
            [10,13,14,17,24],
            [18,21,23,26,30]
        ]
        self.assertFalse(search_matrix(matrix, target=-1))
        self.assertFalse(search_matrix(matrix, target=31))

if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```

### Dec 12, 2020 \[Hard\] Construct Cartesian Tree from Inorder Traversal
---
> **Question:** A Cartesian tree with sequence S is a binary tree defined by the following two properties:
>
> - It is heap-ordered, so that each parent value is strictly less than that of its children.
> - An in-order traversal of the tree produces nodes with values that correspond exactly to S.
>
> Given a sequence S, construct the corresponding Cartesian tree.

**Example:**
```py
Given the sequence [3, 2, 6, 1, 9], the resulting Cartesian tree would be:
      1
    /   \   
  2       9
 / \
3   6
```

**My thoughts:** The root of min heap is always the smallest element. In order to maintain the given inorder traversal order: we can find the min element and recursively build the tree based on subarray on the left and right.

**Solution with Inorder Traversal:** [https://repl.it/@trsong/Construct-Cartesian-Tree-from-Inorder-Traversal](https://repl.it/@trsong/Construct-Cartesian-Tree-from-Inorder-Traversal)
```py
import unittest

def construct_cartesian_tree(nums):
    return construct_cartesian_tree_recur(nums, 0, len(nums) - 1)


def construct_cartesian_tree_recur(nums, start, end):
    if start > end:
        return None

    local_min, local_min_index = find_local_min_and_index(nums, start, end)
    left_res = construct_cartesian_tree_recur(nums, start, local_min_index - 1)
    right_res = construct_cartesian_tree_recur(nums, local_min_index + 1, end)
    return TreeNode(local_min, left_res, right_res)


def find_local_min_and_index(nums, start, end):
    min_val, min_index = nums[start], start
    for i in xrange(start + 1, end + 1):
        if nums[i] < min_val:
            min_val = nums[i]
            min_index = i
    return min_val, min_index


#####################
# Testing Utilities
#####################
class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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
                stack.append((child, depth + 1))
        return "\n" + "".join(res) + "\n"

    def is_heap(self):
        for child in [self.left, self.right]:
            if child and (self.val > child.val or not child.is_heap()):
                return False
        return True

    def traversal(self):
        res = []
        if self.left:
            res.extend(self.left.traversal())
        res.append(self.val)
        if self.right:
            res.extend(self.right.traversal())
        return res


class ConstructCartesianTree(unittest.TestCase):
    def assert_result(self, res, nums):
        self.assertEqual(nums, res.traversal())
        self.assertTrue(res.is_heap(), res)

    def test_example(self):
        """
             1
            / \
           2   9
          / \
         3   6
        """
        nums = [3, 2, 6, 1, 9]
        res = construct_cartesian_tree(nums)
        self.assert_result(res, nums)

    def test_example2(self):
        """
             1
           /   \
          3     5
         / \   /
        9   7 8
               \
                10
               /  \
              12  15
                  / \
                20  18
        """
        nums = [9, 3, 7, 1, 8, 12, 10, 20, 15, 18, 5]
        res = construct_cartesian_tree(nums)
        self.assert_result(res, nums)

    def test_empty_array(self):
        self.assertEqual(None, construct_cartesian_tree([]))

    def test_ascending_array(self):
        """
        1
         \
          2
           \
            3
        """
        nums = [1, 2, 3]
        res = construct_cartesian_tree(nums)
        self.assert_result(res, nums)

    def test_descending_array(self):
        """
              1
             /
            2
           /
          3
         / 
        4
        """
        nums = [1, 2, 3, 4]
        res = construct_cartesian_tree(nums)
        self.assert_result(res, nums)

    def test_ascending_descending_array(self):
        """
          1
         / 
        1   
         \
          2
           \
            2
           /
          3  
        """
        nums = [1, 2, 3, 2, 1]
        res = construct_cartesian_tree(nums)
        self.assert_result(res, nums)

    def test_descending_ascending_array(self):
        """
           1
          / \
         2   2
        /     \
       3       3
        """
        nums = [3, 2, 1, 2, 3]
        res = construct_cartesian_tree(nums)
        self.assert_result(res, nums)   


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Dec 11, 2020 \[Hard\] Find Next Sparse Number
---
> **Question:** We say a number is sparse if there are no adjacent ones in its binary representation. For example, `21 (10101)` is sparse, but `22 (10110)` is not. 
> 
> For a given input `N`, find the smallest sparse number greater than or equal to `N`.
>
> Do this in faster than `O(N log N)` time.

**My thoughts:** Whenever we see sub-binary string `011` mark it as `100` and set all bit on the right to 0. eg. `100110` => `101000`, `101101` => `1000000`

**Solution:** [https://repl.it/@trsong/Find-the-Next-Spare-Number](https://repl.it/@trsong/Find-the-Next-Spare-Number)
```py
import unittest

def next_sparse_number(num):
    target_pattern = 0b011
    window = 0b111
    i = 0
    last_adj = 0
    while num >= (target_pattern << i):
        if num & (window << i) == (target_pattern << i):
            # Change window from 011 to 100
            num ^= (window << i)
            last_adj = i
        i += 1

    # Set all bits on the right to 0
    num &= ~0 << last_adj
    return num


class NextSparseNumberSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(0b10101, next_sparse_number(0b10101))

    def test_no_bit_is_set(self):
        self.assertEqual(0, next_sparse_number(0))

    def test_next_sparse_is_itself(self):
        self.assertEqual(0b100, next_sparse_number(0b100))

    def test_adjacent_bit_is_set(self):
        self.assertEqual(0b1000, next_sparse_number(0b110))
    
    def test_adjacent_bit_is_set2(self):
        self.assertEqual(0b101000, next_sparse_number(0b100110))

    def test_bit_shift_cause_another_bit_shift(self):
        self.assertEqual(0b1000000, next_sparse_number(0b101101))
    
    def test_complicated_number(self):
        self.assertEqual(0b1010010000000, next_sparse_number(0b1010001011101))

    def test_all_bit_is_one(self):
        self.assertEqual(0b1000, next_sparse_number(0b111))


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```

### Dec 10, 2020 \[Easy\] Distribute Bonuses
---
> **Question:** MegaCorp wants to give bonuses to its employees based on how many lines of codes they have written. They would like to give the smallest positive amount to each worker consistent with the constraint that if a developer has written more lines of code than their neighbor, they should receive more money.
>
> Given an array representing a line of seats of employees at MegaCorp, determine how much each one should get paid.

**Example:**
```py
Input: [10, 40, 200, 1000, 60, 30]
Output: [1, 2, 3, 4, 2, 1].
```

**Solution:** [https://repl.it/@trsong/Distribute-Bonuses](https://repl.it/@trsong/Distribute-Bonuses)
```py
import unittest

def distribute_bonus(line_of_codes):
    n = len(line_of_codes)
    res = [1] * n

    for i in xrange(1, n):
        if line_of_codes[i] == line_of_codes[i-1]:
            res[i] = max(res[i],  res[i-1])
        elif line_of_codes[i] > line_of_codes[i-1]:
            res[i] = max(res[i], 1 + res[i-1])

        j = n - 1 - i
        if line_of_codes[j] == line_of_codes[j+1]:
            res[j] = max(res[j], res[j+1])
        elif line_of_codes[j] > line_of_codes[j+1]:
            res[j] = max(res[j], 1 + res[j+1])
    return res


class DistributeBonusSpec(unittest.TestCase):
    def test_example(self):
        line_of_codes = [10, 40, 200, 1000, 60, 30]
        expected = [1, 2, 3, 4, 2, 1]
        self.assertEqual(expected, distribute_bonus(line_of_codes))

    def test_example2(self):
        line_of_codes = [10, 40, 200, 1000, 900, 800, 30]
        expected = [1, 2, 3, 4, 3, 2, 1]
        self.assertEqual(expected, distribute_bonus(line_of_codes))

    def test_empty_array(self):
        line_of_codes = []
        expected = []
        self.assertEqual(expected, distribute_bonus(line_of_codes))

    def test_one_employee(self):
        line_of_codes = [42]
        expected = [1]
        self.assertEqual(expected, distribute_bonus(line_of_codes))

    def test_two_employees(self):
        line_of_codes = [99, 42]
        expected = [2, 1]
        self.assertEqual(expected, distribute_bonus(line_of_codes))

    def test_reach_tie_on_lines_of_code(self):
        line_of_codes = [3, 3, 3, 4, 4, 4, 2, 2, 2]
        expected = [1, 1, 1, 2, 2, 2, 1, 1, 1]
        self.assertEqual(expected, distribute_bonus(line_of_codes))

    def test_reach_tie_on_lines_of_code2(self):
        line_of_codes = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        expected = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        self.assertEqual(expected, distribute_bonus(line_of_codes))

    def test_reach_tie_on_lines_of_code4(self):
        line_of_codes = [4, 4, 3, 3, 2, 2, 1, 1]
        expected = [4, 4, 3, 3, 2, 2, 1, 1]
        self.assertEqual(expected, distribute_bonus(line_of_codes))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Dec 9, 2020 \[Hard\] Power Supply to All Cities
---
> **Question:** Given a graph of possible electricity connections (each with their own cost) between cities in an area, find the cheapest way to supply power to all cities in the area. 

**Example 1:**
```py
Input: cities = ['Vancouver', 'Richmond', 'Burnaby']
       cost_btw_cities = [
           ('Vancouver', 'Richmond', 1),
           ('Vancouver', 'Burnaby', 1),
           ('Richmond', 'Burnaby', 2)]
Output: 2  
Explanation: 
Min cost to supply all cities is to connect the following cities with total cost 1 + 1 = 2: 
(Vancouver, Burnaby), (Vancouver, Richmond)
```

**Example 2:**
```py
Input: cities = ['Toronto', 'Mississauga', 'Waterloo', 'Hamilton']
       cost_btw_cities = [
           ('Mississauga', 'Toronto', 1),
           ('Toronto', 'Waterloo', 2),
           ('Waterloo', 'Hamilton', 3),
           ('Toronto', 'Hamilton', 2),
           ('Mississauga', 'Hamilton', 1),
           ('Mississauga', 'Waterloo', 2)]
Output: 4
Explanation: Min cost to connect to all cities is 4:
(Toronto, Mississauga), (Toronto, Waterloo), (Mississauga, Hamilton)
```

**My thoughts:** This question is an undirected graph problem asking for total cost of minimum spanning tree. Both of the follwing algorithms can solve this problem: Kruskal’s and Prim’s MST Algorithm. First one keeps choosing edges whereas second one starts from connecting vertices. Either one will work. 

**Solution with Kruskal’s MST Algorithm:** [https://repl.it/@trsong/Design-Power-Supply-to-All-Cities](https://repl.it/@trsong/Design-Power-Supply-to-All-Cities)
```py
import unittest

class DisjointSet(object):
    def __init__(self, size):
        self.parent = range(size)

    def find(self, p):
        while self.parent[p] != p:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p
    
    def union(self, p1, p2):
        root1 = self.find(p1)
        root2 = self.find(p2)
        if root1 != root2:
            self.parent[root1] = root2

    def is_connected(self, p1, p2):
        return self.find(p1) == self.find(p2)


def min_cost_power_supply(cities, cost_btw_cities):
    city_id_lookup = {city: index for index, city in enumerate(cities)}
    uf = DisjointSet(len(cities))

    cost_btw_cities.sort(key=lambda uvw: uvw[-1])
    res = 0
    for city1, city2, cost in cost_btw_cities:
        id1, id2 = city_id_lookup[city1], city_id_lookup[city2]
        if uf.is_connected(id1, id2):
            continue

        uf.union(id1, id2)
        res += cost
    return res


class MinCostPowerSupplySpec(unittest.TestCase):
    def test_k3_graph(self):
        cities = ['Vancouver', 'Richmond', 'Burnaby']
        cost_btw_cities = [
            ('Vancouver', 'Richmond', 1),
            ('Vancouver', 'Burnaby', 1),
            ('Richmond', 'Burnaby', 2)
        ]
        # (Vancouver, Burnaby), (Vancouver, Richmond)
        self.assertEqual(2, min_cost_power_supply(cities, cost_btw_cities))  

    def test_k4_graph(self):
        cities = ['Toronto', 'Mississauga', 'Waterloo', 'Hamilton']
        cost_btw_cities = [
            ('Mississauga', 'Toronto', 1),
            ('Toronto', 'Waterloo', 2),
            ('Waterloo', 'Hamilton', 3),
            ('Toronto', 'Hamilton', 2),
            ('Mississauga', 'Hamilton', 1),
            ('Mississauga', 'Waterloo', 2)
        ]
        # (Toronto, Mississauga), (Toronto, Waterloo), (Mississauga, Hamilton)
        self.assertEqual(4, min_cost_power_supply(cities, cost_btw_cities)) 

    def test_connected_graph(self):
        cities = ['Shanghai', 'Nantong', 'Suzhou', 'Hangzhou', 'Ningbo']
        cost_btw_cities = [
            ('Shanghai', 'Nantong', 1),
            ('Nantong', 'Suzhou', 1),
            ('Suzhou', 'Shanghai', 1),
            ('Suzhou', 'Hangzhou', 3),
            ('Hangzhou', 'Ningbo', 2),
            ('Hangzhou', 'Shanghai', 2),
            ('Ningbo', 'Shanghai', 2)
        ]
        # (Shanghai, Nantong), (Shanghai, Suzhou), (Shanghai, Hangzhou), (Shanghai, Nantong)
        self.assertEqual(6, min_cost_power_supply(cities, cost_btw_cities)) 


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 8, 2020 LC 743 \[Medium\] Network Delay Time
---
> **Question:** A network consists of nodes labeled 0 to N. You are given a list of edges `(a, b, t)`, describing the time `t` it takes for a message to be sent from node `a` to node `b`. Whenever a node receives a message, it immediately passes the message on to a neighboring node, if possible.
>
> Assuming all nodes are connected, determine how long it will take for every node to receive a message that begins at node 0.

**Example:** 
```py
given N = 5, and the following edges:

edges = [
    (0, 1, 5),
    (0, 2, 3),
    (0, 5, 4),
    (1, 3, 8),
    (2, 3, 1),
    (3, 5, 10),
    (3, 4, 5)
]

You should return 9, because propagating the message from 0 -> 2 -> 3 -> 4 will take that much time.
```

**Solution with Dijkstra’s Algorithm:** [https://repl.it/@trsong/Find-Network-Delay-Time](https://repl.it/@trsong/Find-Network-Delay-Time)
```py
import unittest
import sys
from Queue import PriorityQueue

def max_network_delay(times, nodes):
    neighbors = [None] * (1 + nodes)
    for u, v, t in times:
        neighbors[u] = neighbors[u] or []
        neighbors[u].append((v, t))
    
    distance = [sys.maxint] * (1 + nodes)
    pq = PriorityQueue()
    pq.put((0, 0))

    while not pq.empty():
        cur_time, cur = pq.get()
        if distance[cur] != sys.maxint:
            continue
        distance[cur] = cur_time
        
        if neighbors[cur] is None:
            continue

        for nb, t in neighbors[cur]:
            alt_time = distance[cur] + t
            if distance[nb] > alt_time:
                pq.put((alt_time, nb))

    max_delay = max(distance)
    return max_delay if max_delay != sys.maxint else -1


class MaxNetworkDelay(unittest.TestCase):
    def test_example(self):
        times = [
            (0, 1, 5), (0, 2, 3), (0, 5, 4), (1, 3, 8), 
            (2, 3, 1), (3, 5, 10), (3, 4, 5)
        ]
        self.assertEqual(9, max_network_delay(times, nodes=5))  # max path: 0 - 2 - 3 - 4

    def test_discounted_graph(self):
        self.assertEqual(-1, max_network_delay([], nodes=2))

    def test_disconnected_graph2(self):
        """
        0(start)    3
        |           |
        v           v
        2           1
        """
        times = [(0, 2, 1), (3, 1, 2)]
        self.assertEqual(-1, max_network_delay(times, nodes=3))

    def test_unreachable_node(self):
        """
        1
        |
        v
        2 
        |
        v
        0 (start)
        |
        v
        3
        """
        times = [(1, 2, 1), (2, 0, 2), (0, 3, 3)]
        self.assertEqual(-1, max_network_delay(times, nodes=3))

    def test_given_example(self):
        """
    (start)
        0 --> 3
        |     |
        v     v
        1     2
        """
        times = [(0, 1, 1), (0, 3, 1), (3, 2, 1)]
        self.assertEqual(2, max_network_delay(times, nodes=3))

    def test_exist_alternative_path(self):
        """
    (start)  1
        0 ---> 3
      1 | \ 4  | 2
        v  \   v
        2   -> 1
        """
        times = [(0, 2, 1), (0, 3, 1), (0, 1, 4), (3, 1, 2)]
        self.assertEqual(3, max_network_delay(times, nodes=3))  # max path: 0 - 3 - 1

    def test_graph_with_cycle(self):
        """
    (start) 
        0 --> 2
        ^     |
        |     v
        1 <-- 3
        """
        times = [(0, 2, 1), (2, 3, 1), (3, 1, 1), (1, 0, 1)]
        self.assertEqual(3, max_network_delay(times, nodes=3))  # max path: 0 - 2 - 3

    def test_multiple_paths(self):
        """
            0 (start)
           /|\
          / | \
        1| 2| 3|
         v  v  v
         2  3  4
        2| 3| 1|
         v  v  v
         5  6  1
        """
        times = [(0, 2, 1), (0, 3, 2), (0, 4, 3), (2, 5, 2), (3, 6, 3), (4, 1, 1)]
        self.assertEqual(5, max_network_delay(times, nodes=6))  # max path: 0 - 3 - 6

    
if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 7, 2020 \[Medium\] Rearrange String with Repeated Characters
---
> **Question:** Given a string with repeated characters, rearrange the string so that no two adjacent characters are the same. If this is not possible, return None.
>
> For example, given `"aaabbc"`, you could return `"ababac"`. Given `"aaab"`, return `None`.

**My thoughts:** This prblem is a special case of [Rearrange String K Distance Apart](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-11-2020-lc-358-hard-rearrange-string-k-distance-apart). Just Greedily choose the character with max remaining number for each window size 2. If no such character satisfy return None instead.

**Solution with Greedy Algorithm:** [https://repl.it/@trsong/Rearrange-String-with-Repeated-Characters](https://repl.it/@trsong/Rearrange-String-with-Repeated-Characters)
```py
import unittest
from Queue import PriorityQueue

def rearrange_string(s):
    histogram = {}
    for ch in s:
        histogram[ch] = histogram.get(ch, 0) + 1

    max_heap = PriorityQueue()
    for ch, count in histogram.items():
        max_heap.put((-count, ch))

    res = []
    while not max_heap.empty():
        remainings = []
        for _ in xrange(2):
            if max_heap.empty() and not remainings:
                break
            if max_heap.empty():
                return None
            
            neg_count, ch = max_heap.get()
            res.append(ch)
            if abs(neg_count) > 1:
                remainings.append((ch, abs(neg_count) - 1))

        for ch, count in remainings:
            max_heap.put((-count, ch))

    return "".join(res)


class RearrangeStringSpec(unittest.TestCase):
    def assert_result(self, original):
        res = rearrange_string(original)
        self.assertEqual(sorted(original), sorted(res))
        for i in xrange(1, len(res)):
            self.assertNotEqual(res[i], res[i-1])

    def test_example1(self):
        # possible solution: ababac
        self.assert_result("aaabbc")
    
    def test_example2(self):
        self.assertIsNone(rearrange_string("aaab"))
    
    def test_example3(self):
        # possible solution: ababacdc
        self.assert_result("aaadbbcc")
    
    def test_unable_to_rearrange(self):
        self.assertIsNone(rearrange_string("aaaaaaaaa"))
    
    def test_unable_to_rearrange2(self):
        self.assertIsNone(rearrange_string("121211"))

    def test_empty_input_string(self):
        self.assert_result("")
    
    def test_possible_to_arrange(self):
        # possible solution: ababababab
        self.assert_result("aaaaabbbbb")
    
    def test_possible_to_arrange2(self):
        # possible solution: 1213141
        self.assert_result("1111234")
    
    def test_possible_to_arrange3(self):
        # possible solution: 1212
        self.assert_result("1122")
    

if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```

### Dec 6, 2020 \[Easy\] Implement Prefix Map Sum
---
> **Question:** Implement a PrefixMapSum class with the following methods:
>
> - `insert(key: str, value: int)`: Set a given key's value in the map. If the key already exists, overwrite the value.
> - `sum(prefix: str)`: Return the sum of all values of keys that begin with a given prefix.

**Example:**
```py
mapsum.insert("columnar", 3)
assert mapsum.sum("col") == 3

mapsum.insert("column", 2)
assert mapsum.sum("col") == 5
```

**Solution with Trie:** [https://repl.it/@trsong/Implement-Prefix-Map-Sum](https://repl.it/@trsong/Implement-Prefix-Map-Sum)
```py
import unittest

class Trie(object):
    def __init__(self):
        self.count = 0
        self.children = None
    

class PrefixMap(object):
    def __init__(self):
        self.root = Trie()
        self.record = {}

    def insert(self, word, val):
        updated_val = val - self.record.get(word, 0)
        self.record[word] = val

        p = self.root
        for ch in word:
            p.count += updated_val
            p.children = p.children or {}
            if ch not in p.children:
                p.children[ch] = Trie()
            p = p.children[ch]
        p.count += updated_val

    def sum(self, word):
        p = self.root
        for ch in word:
            if not p or not p.children:
                return 0
            p = p.children.get(ch, None)
        return p.count if p else 0
            

class PrefixMapSpec(unittest.TestCase):
    def test_example(self):
        prefix_map = PrefixMap()
        prefix_map.insert("columnar", 3)
        self.assertEqual(3, prefix_map.sum("col"))
        prefix_map.insert("column", 2)
        self.assertEqual(5, prefix_map.sum("col"))

    def test_empty_map(self):
        prefix_map = PrefixMap()
        self.assertEqual(0, prefix_map.sum(""))
        self.assertEqual(0, prefix_map.sum("unknown"))

    def test_same_prefix(self):
        prefix_map = PrefixMap()
        prefix_map.insert("a", 1)
        prefix_map.insert("aa", 2)
        prefix_map.insert("aaa", 3)
        self.assertEqual(0, prefix_map.sum("aaaa"))
        self.assertEqual(3, prefix_map.sum("aaa"))
        self.assertEqual(5, prefix_map.sum("aa"))
        self.assertEqual(6, prefix_map.sum("a"))
        self.assertEqual(6, prefix_map.sum(""))

    def test_same_prefix2(self):
        prefix_map = PrefixMap()
        prefix_map.insert("aa", 1)
        prefix_map.insert("a", 2)
        prefix_map.insert("b", 1)
        self.assertEqual(0, prefix_map.sum("aaa"))
        self.assertEqual(1, prefix_map.sum("aa"))
        self.assertEqual(3, prefix_map.sum("a"))
        self.assertEqual(4, prefix_map.sum(""))

    def test_double_prefix(self):
        prefix_map = PrefixMap()
        prefix_map.insert("abc", 1)
        prefix_map.insert("abd", 2)
        prefix_map.insert("abzz", 1)
        prefix_map.insert("bazz", 1)
        self.assertEqual(4, prefix_map.sum("ab"))
        self.assertEqual(0, prefix_map.sum("abq"))
        self.assertEqual(4, prefix_map.sum("a"))
        self.assertEqual(1, prefix_map.sum("b"))

    def test_update_value(self):
        prefix_map = PrefixMap()
        prefix_map.insert('a', 1)
        prefix_map.insert('ab', 1)
        prefix_map.insert('abc', 1)
        prefix_map.insert('ab', 100)
        self.assertEqual(102, prefix_map.sum('a'))


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```


### Dec 5, 2020 LC 274 \[Medium\] H-Index
---
> **Question:** The h-index is a metric that attempts to measure the productivity and citation impact of the publication of a scholar. The definition of the h-index is if a scholar has at least h of their papers cited h times.
>
> Given a list of publications of the number of citations a scholar has, find their h-index.

**Example:**
```py
Input: [3, 5, 0, 1, 3]
Output: 3
Explanation:
There are 3 publications with 3 or more citations, hence the h-index is 3.
```

**Solution:** [https://repl.it/@trsong/Calculate-H-Index](https://repl.it/@trsong/Calculate-H-Index)
```py
import unittest

def calculate_h_index(citations):
    citations.sort(reverse=True)
    count = 0
    for num in citations:
        if num <= count:
            return count
        count += 1
    return count


class CalculateHIndexSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(3, calculate_h_index([3, 5, 0, 1, 3]))

    def test_another_citation_array(self):
        self.assertEqual(3, calculate_h_index([3, 0, 6, 1, 5]))

    def test_empty_citations(self):
        self.assertEqual(0, calculate_h_index([]))

    def test_only_one_publications(self):
        self.assertEqual(1, calculate_h_index([42]))
    
    def test_h_index_appear_once(self):
        self.assertEqual(5, calculate_h_index([5, 10, 10, 10, 10]))

    def test_balanced_citation_counts(self):
        self.assertEqual(5, calculate_h_index([9, 8, 7, 6, 5, 4, 3, 2, 1]))

    def test_duplicated_citations(self):
        self.assertEqual(3, calculate_h_index([3, 3, 3, 2, 2, 2, 2, 2]))
    
    def test_zero_citations_not_count(self):
        self.assertEqual(2, calculate_h_index([10, 0, 0, 0, 0, 10]))
    
    def test_citations_number_greater_than_publications(self):
        self.assertEqual(4, calculate_h_index([9, 8, 7, 6]))

    def test_citations_number_greater_than_publications2(self):
        self.assertEqual(3, calculate_h_index([1, 7, 9, 4]))   


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Dec 4, 2020 \[Hard\] Sliding Puzzle 
---
> **Question:**  An 8-puzzle is a game played on a 3 x 3 board of tiles, with the ninth tile missing. The remaining tiles are labeled 1 through 8 but shuffled randomly. Tiles may slide horizontally or vertically into an empty space, but may not be removed from the board.
>
> Design a class to represent the board, and find a series of steps to bring the board to the state `[[1, 2, 3], [4, 5, 6], [7, 8, None]]`.

**Solution with A-Star Search:** [https://repl.it/@trsong/Solve-Sliding-Puzzle](https://repl.it/@trsong/Solve-Sliding-Puzzle)
```py
import unittest
from copy import deepcopy
from Queue import PriorityQueue

class SlidingPuzzle(object):
    BLANK_VALUE = 9
    GOAL_STATE = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, None]
    ]
    DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    @staticmethod
    def solve(grid):
        """
        Given a grid and returns a sequence of moves to reach to goal state
        """
        pq = PriorityQueue()
        end_hash = SlidingPuzzle.hash(SlidingPuzzle.GOAL_STATE)
        start_hash = SlidingPuzzle.hash(grid)

        visited = set()
        pq.put((0, start_hash))
        prev_states = {start_hash: (None, None)}
        actual_cost = {start_hash: 0}

        while not pq.empty():
            _, cur_hash = pq.get()
            if cur_hash == end_hash:
                break
            if cur_hash in visited:
                continue
            visited.add(cur_hash)

            for neihbor_hash, neighbor_heuristic, move in SlidingPuzzle.neighbor_costs(cur_hash):
                if neihbor_hash in visited:
                    continue
                actual_cost[neihbor_hash] = actual_cost[cur_hash] + 1
                pq.put((actual_cost[neihbor_hash] + neighbor_heuristic, neihbor_hash))
                prev_states[neihbor_hash] = (cur_hash, move)

        moves = []
        while end_hash:
            prev_hash, move = prev_states[end_hash]
            if move:
                moves.append(move)
            end_hash = prev_hash
        moves.reverse()
        return moves

    @staticmethod
    def neighbor_costs(grid_hash):
        """
        Given a grid hash, returns next grid's hash value, cost and move 
        """
        blank_row, blank_col = 0, 0
        grid = [[None for _ in xrange(3)] for _ in xrange(3)]
        for r in xrange(2, -1, -1):
            for c in xrange(2, -1, -1):
                grid[r][c] = grid_hash % 10
                grid_hash //= 10
                if grid[r][c] == SlidingPuzzle.BLANK_VALUE:
                    grid[r][c] = None
                    blank_row, blank_col = r, c

        for dr, dc in SlidingPuzzle.DIRECTIONS:
            new_r, new_c = blank_row + dr, blank_col + dc
            if 0 <= new_r < 3 and 0 <= new_c < 3:
                move = grid[new_r][new_c]
                grid[new_r][new_c] = None
                grid[blank_row][blank_col] = move
                yield SlidingPuzzle.hash(grid), SlidingPuzzle.heuristc(grid), move
                grid[new_r][new_c] = move
                grid[blank_row][blank_col] = None

    @staticmethod
    def hash(grid):
        """
        Flatten grid and then convert to an integer
        """
        res = 0
        for row in grid:
            for num in row:
                # treat None as 9
                res = res * 10 + (num or SlidingPuzzle.BLANK_VALUE)
        return res

    @staticmethod
    def heuristc(grid):
        """
        Estimation of remaining cost based on current grid
        """
        cost = 0
        for r in xrange(3):
            for c in xrange(3):
                # treat None as 9
                num = grid[r][c] or SlidingPuzzle.BLANK_VALUE
                expected_r = (num - 1) // 3
                expected_c = (num - 1) % 3
                cost += abs(r - expected_r) + abs(c - expected_c)
        return cost
        

class SlidingPuzzleSpec(unittest.TestCase):
    ###################
    # Testing Utility
    ###################
    @staticmethod
    def validate(grid, steps):
        blank_row = map(lambda row: None in row, grid).index(True)
        blank_col = grid[blank_row].index(None)

        for num in steps:
            for dr, dc in SlidingPuzzle.DIRECTIONS:
                new_r, new_c = blank_row + dr, blank_col + dc
                if 0 <= new_r < 3 and 0 <= new_c < 3 and grid[new_r][new_c] == num:
                    grid[blank_row][blank_col] = grid[new_r][new_c]
                    grid[new_r][new_c] = None
                    blank_row = new_r
                    blank_col = new_c
                    break
        
        return grid == SlidingPuzzle.GOAL_STATE

    def assert_result(self, grid):
        user_grid = deepcopy(grid)
        steps = SlidingPuzzle.solve(user_grid)
        self.assertTrue(SlidingPuzzleSpec.validate(grid, steps), user_grid)

    def test_heuristic_function_should_not_overestimate(self):
        # Optimial solution: [1, 2, 3, 6, 5, 4, 7, 8]
        self.assert_result([
            [None, 1, 2],
            [5, 6, 3],
            [4, 7, 8],
        ])

    def test_random_grid(self):
        # Optimial solution: [7, 4, 5, 6, 2, 5, 6, 1, 4, 7, 8, 6, 5, 2, 1, 4, 7, 8]
        self.assert_result([
            [6, 2, 3],
            [5, 4, 8],
            [1, 7, None]
        ])

    def test_random_grid2(self):
        # Optimial solution: [3, 6, 8, 4, 7, 5, 2, 1, 4, 7, 5, 3, 6, 8, 7, 4, 1, 2, 3, 6]
        self.assert_result([
            [1, 2, 5],
            [8, 4, 7],
            [6, 3, None]
        ])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 3, 2020 \[Hard\] Knight's Tour Problem
---
> **Question:** A knight's tour is a sequence of moves by a knight on a chessboard such that all squares are visited once.
>
> Given N, write a function to return the number of knight's tours on an N by N chessboard.


**Solution with Backtracking:** [https://repl.it/@trsong/Knights-Tour-Problem](https://repl.it/@trsong/Knights-Tour-Problem)
```py
import unittest

KNIGHT_MOVES = [
    (1, 2), (2, 1), 
    (-1, 2), (-2, 1), 
    (-1, -2), (-2, -1),
    (1, -2), (2, -1)
]


def knights_tours(n):
    if n == 0:
        return 0

    grid = [[False for _ in xrange(n)] for _ in xrange(n)]
    count = 0
    for r in xrange(n):
        for c in xrange(n):
            grid[r][c] = True
            count += backtrack(1, grid, (r, c))
            grid[r][c] = False
    return count


def backtrack(step, grid, pos):
    n = len(grid)
    if step == n * n:
        return 1
    else:
        count = 0
        r, c = pos
        for dr, dc in KNIGHT_MOVES:
            new_r, new_c = r + dr, c + dc
            if 0 <= new_r < n and 0 <= new_c < n and not grid[new_r][new_c]:
                grid[new_r][new_c] = True
                count += backtrack(step + 1, grid, (new_r, new_c))
                grid[new_r][new_c] = False
        return count
                

class KnightsTourSpec(unittest.TestCase):
    """
    Kngiths Tours answer adapts from wiki: https://en.wikipedia.org/wiki/Knight%27s_tour
    """
    def test_size_zero_grid(self):
        self.assertEqual(0, knights_tours(0))

    def test_size_one_grid(self):
        self.assertEqual(1, knights_tours(1))

    def test_size_two_grid(self):
        self.assertEqual(0, knights_tours(2))

    def test_size_three_grid(self):
        self.assertEqual(0, knights_tours(3))

    def test_size_four_grid(self):
        self.assertEqual(0, knights_tours(4))
        
    # Long running execution: took 235 sec
    def test_size_five_grid(self):
        self.assertEqual(1728, knights_tours(5))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 2, 2020 LC 127 \[Medium\] Word Ladder
---
> **Question:** Given a `start` word, an `end` word, and a dictionary of valid words, find the shortest transformation sequence from `start` to `end` such that only one letter is changed at each step of the sequence, and each transformed word exists in the dictionary. If there is no possible transformation, return null. Each word in the dictionary have the same length as start and end and is lowercase.
>
> For example, given `start = "dog"`, `end = "cat"`, and `dictionary = {"dot", "dop", "dat", "cat"}`, return `["dog", "dot", "dat", "cat"]`.
>
> Given `start = "dog"`, `end = "cat"`, and `dictionary = {"dot", "tod", "dat", "dar"}`, return `null` as there is no possible transformation from `dog` to `cat`.

**Solution with BFS:** [https://repl.it/@trsong/Word-Ladder](https://repl.it/@trsong/Word-Ladder)
```py
import unittest
from Queue import Queue

def word_ladder(start, end, word_set):
    parents = {}
    visited = set()
    queue = Queue()
    queue.put((start, None))

    while not queue.empty():
        for _ in xrange(queue.qsize()):
            cur, prev = queue.get()
            if cur in visited:
                continue
            visited.add(cur)

            parents[cur] = prev
            if cur == end:
                return find_path(parents, end)

            for word in word_set:
                if word in visited or not is_neighbor(cur, word):
                    continue
                queue.put((word, cur))
    
    return None


def find_path(parents, start):
    res = []
    while start:
        res.append(start)
        start = parents.get(start, None)
    res.reverse()
    return res


def is_neighbor(word1, word2):
    count = 0
    for c1, c2 in zip(word1, word2):
        if c1 != c2:
            count += 1
        if count > 1:
            return False
    return count == 1


class WordLadderSpec(unittest.TestCase):
    def test_example(self):
        start = 'dog'
        end = 'cat'
        word_set = {'dot', 'dop', 'dat', 'cat'}
        expected = ['dog', 'dot', 'dat', 'cat']
        self.assertEqual(expected, word_ladder(start, end, word_set))

    def test_example2(self):
        start = 'dog'
        end = 'cat'
        word_set = {'dot', 'tod', 'dat', 'dar'}
        self.assertIsNone(word_ladder(start, end, word_set))

    def test_empty_dict(self):
        self.assertIsNone(word_ladder('start', 'end', {}))

    def test_example3(self):
        start = 'hit'
        end = 'cog'
        word_set = {'hot', 'dot', 'dog', 'lit', 'log', 'cog'}
        expected = ['hit', 'hot', 'dot', 'dog', 'cog']
        self.assertEqual(expected, word_ladder(start, end, word_set))

    def test_end_word_not_in_dictionary(self):
        start = 'hit'
        end = 'cog'
        word_set = ['hot', 'dot', 'dog', 'lot', 'log']
        self.assertIsNone(word_ladder(start, end, word_set))

    def test_long_example(self):
        start = 'coder'
        end = 'goner'
        word_set = {
            'lover', 'coder', 'comer', 'toner', 'cover', 'tower', 'coyer',
            'bower', 'honer', 'poles', 'hover', 'lower', 'homer', 'boyer',
            'goner', 'loner', 'boner', 'cower', 'never', 'sower', 'asian'
        }
        expected = ['coder', 'cower', 'lower', 'loner', 'goner']
        self.assertEqual(expected, word_ladder(start, end, word_set))

    def test_long_example2(self):
        start = 'coder'
        end = 'goner'
        word_set = {
            'lover', 'coder', 'comer', 'toner', 'cover', 'tower', 'coyer',
            'bower', 'honer', 'poles', 'hover', 'lower', 'homer', 'boyer',
            'goner', 'loner', 'boner', 'cower', 'never', 'sower', 'asian'
        }
        expected = ['coder', 'cower', 'lower', 'loner', 'goner']
        self.assertEqual(expected, word_ladder(start, end, word_set))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 1, 2020 \[Easy\] Intersection of N Arrays
---
> **Question:** Given n arrays, find the intersection of them.

**Example:**
```py
intersection([1, 2, 3, 4], [2, 4, 6, 8], [3, 4, 5])  # returns [4]
```

**Solution:** [https://repl.it/@trsong/Intersection-of-N-Arrays](https://repl.it/@trsong/Intersection-of-N-Arrays)
```py
import unittest

def intersection(*num_lsts):
    if not num_lsts:
        return []

    n = len(num_lsts)
    sorted_lsts = map(sorted, num_lsts)
    positions = [0] * n

    res = []
    while True:
        min_val = float('inf')
        for i, pos in enumerate(positions):
            if pos >= len(sorted_lsts[i]):
                return res
            min_val = min(min_val, sorted_lsts[i][pos])

        count = 0
        for i, pos in enumerate(positions):
            if sorted_lsts[i][pos] == min_val:
                count += 1
                positions[i] += 1
        
        if count == n:
            res.append(min_val)

    return None


class IntersectionSpec(unittest.TestCase):
    def test_example(self):
        list1 = [1, 2, 3, 4]
        list2 = [2, 4, 6, 8]
        list3 = [3, 4, 5]
        expected = [4]
        self.assertEqual(expected, intersection(list1, list2, list3))

    def test_example2(self):
        list1 = [1, 5, 10, 20, 40, 80]
        list2 = [6, 7, 20, 80, 100]
        list3 = [3, 4, 15, 20, 30, 70, 80, 120]
        expected = [20, 80]
        self.assertEqual(expected, intersection(list1, list2, list3))

    def test_example3(self):
        list1 = [1, 5, 6, 7, 10, 20]
        list2 = [6, 7, 20, 80]
        list3 = [3, 4, 5, 7, 15, 20]
        expected = [7, 20]
        self.assertEqual(expected, intersection(list1, list2, list3))
    
    def test_empty_array(self):
        self.assertEqual([], intersection())

    def test_one_array(self):
        self.assertEqual([1, 2], intersection([1, 2]))

    def test_two_arrays(self):
        list1 = [1, 2, 3]
        list2 = [5, 3, 1]
        expected = [1, 3]
        self.assertEqual(expected, intersection(list1, list2))

    def test_reverse_order(self):
        list1 = [4, 3, 2, 1]
        list2 = [2, 4, 6, 8]
        list3 = [5, 4, 3]
        expected = [4]
        self.assertEqual(expected, intersection(list1, list2, list3))

    def test_contains_duplicate(self):
        list1 = [1, 5, 5]
        list2 = [3, 4, 5, 5, 10]
        list3 = [5, 5, 10, 20]
        expected = [5, 5]
        self.assertEqual(expected, intersection(list1, list2, list3))

    def test_different_length_lists(self):
        list1 = [1, 5, 10, 20, 30]
        list2 = [5, 13, 15, 20]
        list3 = [5, 20]
        expected = [5, 20]
        self.assertEqual(expected, intersection(list1, list2, list3))
    
    def test_empty_list(self):
        list1 = [1, 2, 3, 4, 5]
        list2 = [4, 5, 6, 7]
        list3 = []
        expected = []
        self.assertEqual(expected, intersection(list1, list2, list3))

    def test_empty_list2(self):
        self.assertEqual([], intersection([], [], []))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 30, 2020 LC 228 \[Easy\] Extract Range
---
> **Question:** Given a sorted list of numbers, return a list of strings that represent all of the consecutive numbers.

**Example:**
```py
Input: [0, 1, 2, 5, 7, 8, 9, 9, 10, 11, 15]
Output: ['0->2', '5', '7->11', '15']
```

**Solution:** [https://repl.it/@trsong/Extract-Range](https://repl.it/@trsong/Extract-Range)
```py
import unittest

def extract_range(nums):
    if not nums:
        return []

    res = []
    base = nums[0]
    n = len(nums)

    for i in xrange(1, n+1):
        if i < n and (nums[i] - nums[i-1]) <= 1:
            continue

        if base == nums[i-1]:
            res.append(str(base))
        else:
            res.append("%d->%d" % (base, nums[i-1]))

        if i < n:
            base = nums[i]

    return res


class ExtractRangeSpec(unittest.TestCase):
    def test_example(self):
        nums = [0, 1, 2, 5, 7, 8, 9, 9, 10, 11, 15]
        expected = ['0->2', '5', '7->11', '15']
        self.assertEqual(expected, extract_range(nums))

    def test_empty_array(self):
        self.assertEqual([], extract_range([]))

    def test_one_elem_array(self):
        self.assertEqual(['42'], extract_range([42]))

    def test_duplicates(self):
        nums = [1, 1, 1, 1]
        expected = ['1']
        self.assertEqual(expected, extract_range(nums))

    def test_duplicates2(self):
        nums = [1, 1, 2, 2]
        expected = ['1->2']
        self.assertEqual(expected, extract_range(nums))

    def test_duplicates3(self):
        nums = [1, 1, 3, 3, 5, 5, 5]
        expected = ['1', '3', '5']
        self.assertEqual(expected, extract_range(nums))

    def test_first_elem_in_range(self):
        nums = [1, 2, 3, 10, 11]
        expected = ['1->3', '10->11']
        self.assertEqual(expected, extract_range(nums))

    def test_first_elem_not_in_range(self):
        nums = [-5, -3, -2]
        expected = ['-5', '-3->-2']
        self.assertEqual(expected, extract_range(nums))

    def test_last_elem_in_range(self):
        nums = [0, 15, 16, 17]
        expected = ['0', '15->17']
        self.assertEqual(expected, extract_range(nums))

    def test_last_elem_not_in_range(self):
        nums = [-42, -1, 0, 1, 2, 15]
        expected = ['-42', '-1->2', '15']
        self.assertEqual(expected, extract_range(nums))

    def test_entire_array_in_range(self):
        nums = list(range(-10, 10))
        expected = ['-10->9']
        self.assertEqual(expected, extract_range(nums))

    def test_no_range_at_all(self):
        nums = [1, 3, 5]
        expected = ['1', '3', '5']
        self.assertEqual(expected, extract_range(nums))

    def test_range_and_not_range(self):
        nums = [0, 1, 3, 5, 7, 8, 9, 11, 13, 14, 15]
        expected = ['0->1', '3', '5', '7->9', '11', '13->15']
        self.assertEqual(expected, extract_range(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 29, 2020 \[Medium\] Implement Soundex
---
> **Question:** **Soundex** is an algorithm used to categorize phonetically, such that two names that sound alike but are spelled differently have the same representation.
>
> **Soundex** maps every name to a string consisting of one letter and three numbers, like M460.
>
> One version of the algorithm is as follows:
>
> 1. Remove consecutive consonants with the same sound (for example, change ck -> c).
> 2. Keep the first letter. The remaining steps only apply to the rest of the string.
> 3. Remove all vowels, including y, w, and h.
> 4. Replace all consonants with the following digits:
>    - b, f, p, v → 1
>    - c, g, j, k, q, s, x, z → 2
>    - d, t → 3
>    - l → 4
>    - m, n → 5
>    - r → 6
> 5. If you don't have three numbers yet, append zeros until you do. Keep the first three numbers.
>
> Using this scheme, Jackson and Jaxen both map to J250.

**Solution:** [https://repl.it/@trsong/Implement-Soundex](https://repl.it/@trsong/Implement-Soundex)
```py
import unittest

def soundex_code(s):
    skip_letters = {'y', 'w', 'h'}
    score_to_letter = {
        '1': 'bfpv',
        '2': 'cgjkqsxz',
        '3': 'dt',
        '4': 'l',
        '5': 'mn',
        '6': 'r'
    }
    letter_to_score = {ch: score for score, chs in score_to_letter.items() for ch in chs}
    
    initial = s[0].upper()
    res = [initial]
    prev = None
    for ch in s:
        lower_case = ch.lower()
        if lower_case in skip_letters:
            continue

        code = letter_to_score.get(lower_case, '0')
        if prev is not None and code != '0' and code != prev:
            res.append(code)
        prev = code

        if len(res) == 4:
            break

    res.extend('0000')
    return ''.join(res[:4])


class SoundexCodeSpec(unittest.TestCase):
    def test_example(self):
        # J, 2 for the C, K ignored, S ignored, 5 for the N, 0 added
        self.assertEqual('J250', soundex_code('Jackson'))

    def test_example2(self):
        self.assertEqual('J250', soundex_code('Jaxen'))

    def test_name_with_double_letters(self):
        # G, 3 for the T, 6 for the first R, second R ignored, 2 for the Z
        self.assertEqual('G362', soundex_code('Gutierrez'))

    def test_side_by_side_same_code(self):
        # P, F ignored, 2 for the S, 3 for the T, 6 for the R
        self.assertEqual('P236', soundex_code('Pfister'))

    def test_side_by_side_same_code2(self):
        # T, 5 for the M, 2 for the C, Z ignored, 2 for the K
        self.assertEqual('T522', soundex_code('Tymczak'))

    def test_append_zero_to_end(self):
        self.assertEqual('L000', soundex_code('Lee'))

    def test_discard_extra_letters(self):
        # W, 2 for the S, 5 for the N, 2 for the G, remaining letters disregarded
        self.assertEqual('W252', soundex_code('Washington')) 

    def test_separate_consonant_with_same_code(self):
        self.assertEqual('A261', soundex_code('Ashcraft'))

    def test_more_example(self):
        self.assertEqual('K530', soundex_code('Knuth'))

    def test_more_example2(self):
        self.assertEqual('K530', soundex_code('Kant'))

    def test_more_example3(self):
        self.assertEqual('J612', soundex_code('Jarovski'))

    def test_more_example4(self):
        self.assertEqual('R252', soundex_code('Resnik'))

    def test_more_example5(self):
        self.assertEqual('R252', soundex_code('Reznick'))

    def test_more_example6(self):
        self.assertEqual('E460', soundex_code('Euler'))

    def test_more_example7(self):
        self.assertEqual('P362', soundex_code('Peterson'))

    def test_more_example8(self):
        self.assertEqual('J162', soundex_code('Jefferson'))

    def test_more_example9(self):
        self.assertEqual('T526', soundex_code('Tangrui'))

    def test_more_example10(self):
        self.assertEqual('S520', soundex_code('Song'))

    def test_more_example11(self):
        self.assertEqual('J520', soundex_code('Jing'))

    def test_more_example12(self):
        self.assertEqual('Z520', soundex_code('Zhang'))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 28, 2020 LC 151 \[Medium\] Reverse Words in a String
---
> **Question:** Given an input string, reverse the string word by word.
>
> Note:
>
> - A word is defined as a sequence of non-space characters.
> - Input string may contain leading or trailing spaces. However, your reversed string should not contain leading or trailing spaces.
> - You need to reduce multiple spaces between two words to a single space in the reversed string.

**Example 1:**
```py
Input: "the sky is blue"
Output: "blue is sky the"
```

**Example 2:**
```py
Input: "  hello world!  "
Output: "world! hello"
Explanation: Your reversed string should not contain leading or trailing spaces.
```

**Example 3:**
```py
Input: "a good   example"
Output: "example good a"
Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.
```


**My thoughts:** Suppose after strip out extra whitespaces the string is `"the sky is blue"`. We just need to do the following steps:
1. reverse entire sentence. `"eulb si yks eht"`
2. reverse each word in that sentence. `"blue is sky the"`

**Solution:** [https://repl.it/@trsong/Reverse-words-in-a-string](https://repl.it/@trsong/Reverse-words-in-a-string)
```py
import unittest

def reverse_word(s):
    char_arr = remove_white_spaces(s)
    reverse_section(char_arr, 0, len(char_arr) - 1)
    reverse_each_word(char_arr)
    return ''.join(char_arr)


def remove_white_spaces(s):
    prev = ' '
    res = []
    
    for ch in s:
        if prev == ch == ' ':
            continue
        res.append(ch)
        prev = ch
    
    if res and res[-1] == ' ':
        res.pop()
    
    return res


def reverse_each_word(arr):
    i = 0
    n = len(arr)

    while i < n:
        if arr[i] == ' ':
            i += 1
            continue
        
        start = i
        while i < n and arr[i] != ' ':
            i += 1
        
        reverse_section(arr, start, i-1)


def reverse_section(arr, start, end):
    while start < end:
        arr[start], arr[end] = arr[end], arr[start]
        start += 1
        end -= 1


class ReverseWordSpec(unittest.TestCase):
    def test_example1(self):
        s = "the sky is blue"
        expected = "blue is sky the"
        self.assertEqual(expected, reverse_word(s))

    def test_example2(self):
        s = "  hello world!  "
        expected = "world! hello"
        self.assertEqual(expected, reverse_word(s))

    def test_example3(self):
        s = "a good   example"
        expected = "example good a"
        self.assertEqual(expected, reverse_word(s))

    def test_mutliple_whitespaces(self):
        s = "      "
        expected = ""
        self.assertEqual(expected, reverse_word(s))
    
    def test_mutliple_whitespaces2(self):
        s = "the sky is blue"
        expected = "blue is sky the"
        self.assertEqual(expected, reverse_word(s))

    def test_even_number_of_words(self):
        s = " car cat"
        expected = "cat car"
        self.assertEqual(expected, reverse_word(s))

    def test_even_number_of_words2(self):
        s = "car cat "
        expected = "cat car"
        self.assertEqual(expected, reverse_word(s))

    def test_no_whitespaces(self):
        s = "asparagus"
        expected = "asparagus"
        self.assertEqual(expected, reverse_word(s))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 27, 2020 \[Medium\] Direction and Position Rule Verification
---
> **Question:** A rule looks like this:
>
> ```py
> A NE B
> ``` 
> This means this means point A is located northeast of point B.
> 
> ```
> A SW C
> ```
> means that point A is southwest of C.
>
>Given a list of rules, check if the sum of the rules validate. For example:
> 
> ```
> A N B
> B NE C
> C N A
> ```
> 
> does not validate, since A cannot be both north and south of C.
> 
> ```
> A NW B
> A N B
> ```
> 
> is considered valid.

**My thoughts:** A rule is considered invalid if there exists an opposite rule either implicitly or explicitly. eg. `'A N B'`, `'B N C'` and `'C N A'`. But that doesn't mean we need to scan through the rules over and over again to check if any pair of two rules are conflicting with each other. We can simply scan through the list of rules to build a directional graph for each direction. i.e. E, S, W, N

Then what about diagonal directions. i.e. NE, NW, SE, SW.? As the nature of those rules are 'AND' relation, we can break one diagonal rules into two normal rules. eg, `'A NE B' => 'A N B' and 'A E B'`.

For each direction we build a directional graph, with nodes being the points and edge represents a rule. And each rule will be added to two graphs with opposite directions. e.g. `'A N B'` added to both 'N' graph and 'S' graph. If doing this rule forms a cycle, we simply return False. And if otherwise for all rules, then we return True in the end. 

**Solution with DFS:** [https://repl.it/@trsong/Verify-List-of-Direction-and-Position-Rules](https://repl.it/@trsong/Verify-List-of-Direction-and-Position-Rules)
```py
import unittest

def direction_rule_validate(rules):
    neighbors_by_direction = { 
        direction: {} for direction in [Direction.E, Direction.W, Direction.S, Direction.N]
    }

    for rule in rules:
        start, directions, end = rule.split()
        for direction in directions:
            neighbors = neighbors_by_direction[direction]

            if check_connectivity(neighbors, end, start):
                return False
            
            neighbors[start] = neighbors.get(start, set())
            neighbors[start].add(end)

            opposite_neighbors = neighbors_by_direction[Direction.get_opposite_direction(direction)]
            opposite_neighbors[end] = neighbors.get(end, set())
            opposite_neighbors[end].add(start)

    return True


def check_connectivity(neighbors, start, end):
    stack = [start]
    visited = set()

    while stack:
        cur = stack.pop()
        if cur == end:
            return True
        if cur in visited:
            continue
        visited.add(cur)

        if cur not in neighbors:
            continue

        for child in neighbors[cur]:
            if child not in visited:
                stack.append(child)

    return False


class Direction:
    E = 'E'
    S = 'S'
    W = 'W'
    N = 'N'
    
    @staticmethod
    def get_opposite_direction(d):
        if d == Direction.E: return Direction.W
        if d == Direction.W: return Direction.E
        if d == Direction.S: return Direction.N
        if d == Direction.N: return Direction.S
        return None


class DirectionRuleValidationSpec(unittest.TestCase):
    def test_example1(self):
        self.assertFalse(direction_rule_validate([
            "A N B",
            "B NE C",
            "C N A"
        ]))

    def test_example2(self):
        self.assertTrue(direction_rule_validate([
            "A NW B",
            "A N B"
        ]))

    def test_ambigious_rules(self):
        self.assertTrue(direction_rule_validate([
            "A SE B",
            "C SE B",
            "C SE A",
            "A N C"
        ]))

    def test_conflict_diagonal_directions(self):
        self.assertFalse(direction_rule_validate([
            "B NW A",
            "C SE A",
            "C NE B"
        ]))

    def test_paralllel_rules(self):
        self.assertTrue(direction_rule_validate([
            "A N B",
            "C N D",
            "C E B",
            "B W D",
            "B S D",
            "A N C",
            "D N B",
            "C E A"
        ]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 26, 2020 \[Hard\] Inversion Pairs
---
> **Question:**  We can determine how "out of order" an array A is by counting the number of inversions it has. Two elements `A[i]` and `A[j]` form an inversion if `A[i] > A[j]` but `i < j`. That is, a smaller element appears after a larger element. Given an array, count the number of inversions it has. Do this faster than `O(N^2)` time. You may assume each element in the array is distinct.
>
> For example, a sorted list has zero inversions. The array `[2, 4, 1, 3, 5]` has three inversions: `(2, 1)`, `(4, 1)`, and `(4, 3)`. The array `[5, 4, 3, 2, 1]` has ten inversions: every distinct pair forms an inversion.


**Trivial Solution:** 
```py
def count_inversion_pairs_naive(nums):
    inversions = 0
    n = len(nums)
    for i in xrange(n):
        for j in xrange(i+1, n):
            inversions += 1 if nums[i] > nums[j] else 0
    return inversions
```

**My thoughts:** We can start from trivial solution and perform optimization after. In trivial solution basically, we try to count the total number of larger number on the left of smaller number. However, while solving this question, you need to ask yourself, is it necessary to iterate through all combination of different pairs and calculate result?

e.g. For input `[5, 4, 3, 2, 1]` each pair is an inversion pair, once I know `(5, 4)` will be a pair, then do I need to go over the remain `(5, 3)`, `(5, 2)`, `(5, 1)` as 3,2,1 are all less than 4? 

So there probably exits some tricks to allow us save some effort to not go over all possible combinations. 

Did you notice the following properties? 

1. `count_inversion_pairs([5, 4, 3, 2, 1]) = count_inversion_pairs([5, 4]) + count_inversion_pairs([3, 2, 1]) + inversion_pairs_between([5, 4], [3, 2, 1])`
2. `inversion_pairs_between([5, 4], [3, 2, 1]) = inversion_pairs_between(sorted([5, 4]), sorted([3, 2, 1])) = inversion_pairs_between([4, 5], [1, 2, 3])`

This is bascially modified version of merge sort. Consider we break `[5, 4, 3, 2, 1]` into two almost equal parts: `[5, 4]` and `[3, 2, 1]`. Notice such break won't affect inversion pairs, as whatever on the left remains on the left. However, inversion pairs between `[5, 4]` and `[3, 2, 1]` can be hard to count without doing it one-by-one. 

If only we could sort them separately as sort won't affect the inversion order between two lists. i.e. `[4, 5]` and `[1, 2, 3]`. Now let's see if we can find the pattern, if `4 < 1`, then `5 should < 1`. And we also have `4 < 2` and `4 < 3`. We can simply skip all `elem > than 4` on each iteration, i.e. we just need to calculate how many elem > 4 on each iteration. This gives us **property 2**.

And we can futher break `[5, 4]` into `[5]` and `[4]` recursively. This gives us **property 1**.

Combine property 1 and 2 gives us the modified version of ***Merge-Sort***.

**Solution with Merge Sort:** [https://repl.it/@trsong/Count-Inversion-Pairs](https://repl.it/@trsong/Count-Inversion-Pairs)
```py
import unittest

def count_inversion_pairs(nums):
    count, _ = count_and_sort(nums)
    return count


def count_and_sort(nums):
    if len(nums) < 2:
        return 0, nums
    
    mid = len(nums) // 2
    sub_res1, sorted1 = count_and_sort(nums[:mid])
    sub_res2, sorted2 = count_and_sort(nums[mid:])
    count_res, merged = count_and_merge(sorted1, sorted2)
    return sub_res1 + sub_res2 + count_res, merged


def count_and_merge(nums1, nums2):
    inversions = 0
    merged = []

    i = j = 0
    len1, len2 = len(nums1), len(nums2)
    while i < len1 and j < len2:
        if nums1[i] <= nums2[j]:
            merged.append(nums1[i])
            i += 1
        else:
            # there are len1 - i numbers one the left array greater than right
            inversions += len1 - i 
            merged.append(nums2[j])
            j += 1

    while i < len1:
        merged.append(nums1[i])
        i += 1

    while j < len2:
        merged.append(nums2[j])
        j += 1

    return inversions, merged


class CountInversionPairSpec(unittest.TestCase):
    def test_example(self):
        nums = [2, 4, 1, 3, 5]
        expected = 3  # (2, 1), (4, 1), (4, 3)
        self.assertEqual(expected, count_inversion_pairs(nums))

    def test_example2(self):
        nums = [5, 4, 3, 2, 1]
        expected = 10  # (5, 4), (5, 3), ... (2, 1) = 4 + 3 + 2 + 1 = 10 
        self.assertEqual(expected, count_inversion_pairs(nums))

    def test_empty_array(self):
        self.assertEqual(0, count_inversion_pairs([]))

    def test_one_elem_array(self):
        self.assertEqual(0, count_inversion_pairs([42]))

    def test_ascending_array(self):
        nums = [1, 4, 6, 8, 9]
        expected = 0
        self.assertEqual(expected, count_inversion_pairs(nums))

    def test_ascending_array2(self):
        nums = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        expected = 0
        self.assertEqual(expected, count_inversion_pairs(nums))

    def test_increasing_decreasing_array(self):
        nums = [1, 2, 3, 2, 5]
        expected = 1  # (3, 2)
        self.assertEqual(expected, count_inversion_pairs(nums))

    def test_decreasing_increasing_array(self):
        nums = [0, -1, -2, -2, 2, 3]
        expected = 5  # (0, -1), (0, -2), (0, -2), (-1, -2), (-1, -2)
        self.assertEqual(expected, count_inversion_pairs(nums))

    def test_unique_value_array(self):
        nums = [0, 0, 0]
        expected = 0
        self.assertEqual(expected, count_inversion_pairs(nums))

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 25, 2020 \[Hard\] Minimum Cost to Construct Pyramid with Stones
---
> **Question:** You have `N` stones in a row, and would like to create from them a pyramid. This pyramid should be constructed such that the height of each stone increases by one until reaching the tallest stone, after which the heights decrease by one. In addition, the start and end stones of the pyramid should each be one stone high.
>
> You can change the height of any stone by paying a cost of `1` unit to lower its height by `1`, as many times as necessary. Given this information, determine the lowest cost method to produce this pyramid.
>
> For example, given the stones `[1, 1, 3, 3, 2, 1]`, the optimal solution is to pay `2` to create `[0, 1, 2, 3, 2, 1]`.

**My thoughts:** To find min cost is equivalent to find max pyramid we can construct. With that being said, all we need is to figure out each position's max possible height. 

A position's height can only be 1 greater than prvious one on the left and right position. We can scan from left and right to figure out the max height for each position. 

After that, we need to find center location of pyramid. That is just the max height. And the total number of stones equals `1 + 2 + .. + n + ... + 1 = n * n`.

Therefore the `min cost` is just `sum of all stones - total stones of max pyramid`. 

**Solution:** [https://repl.it/@trsong/Minimum-Cost-to-Construct-Pyramid-with-Stones](https://repl.it/@trsong/Minimum-Cost-to-Construct-Pyramid-with-Stones)
```py
import unittest

def min_cost_pyramid(stones):
    n = len(stones)
    left_trim_heights = [None] * n
    right_trim_heights = [None] * n

    left_trim_heights[0] = min(1, stones[0])
    right_trim_heights[n - 1] = min(1, stones[n - 1])

    for i in xrange(1, n):
        left_trim_heights[i] = min(left_trim_heights[i - 1] + 1, stones[i])
        right_trim_heights[n - i - 1] = min(right_trim_heights[n - i] + 1, stones[n - i - 1])

    trim_heights = map(min, zip(left_trim_heights, right_trim_heights))
    pyramid_height = max(trim_heights)

    total_pyramid_stones = pyramid_height * pyramid_height  # 1 + 2 + ... + 2 + 1
    return sum(stones) - total_pyramid_stones


class MinCostPyramidSpec(unittest.TestCase):
    def test_example(self):
        stones = [1, 1, 3, 3, 2, 1]
        expected = 2  # [0, 1, 2, 3, 2, 1]
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_small_pyramid(self):
        stones = [1, 2, 1]
        expected = 0
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_small_pyramid2(self):
        stones = [1, 1, 1]
        expected = 2  # [0, 1, 0]
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_almost_pyramid(self):
        stones = [1, 2, 3, 4, 2, 1]
        expected = 4  # [1, 2, 3, 2, 1, 0]
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_choice_between_different_pyramid(self):
        stones = [1, 2, 1, 0, 0, 1, 2, 3, 2, 1, 0, 1, 0]
        expected = 5  # [0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0]
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_build_from_flat_plane(self):
        stones = [5, 5, 5, 5, 5]
        expected = 16  # [1, 2, 3, 2, 1]
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_concave_array(self):
        stones = [0, 0, 3, 2, 3, 0]
        expected = 4  # [0, 0, 1, 2, 1, 0]
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_multiple_layer_platforms(self):
        stones = [2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 5, 5, 5, 5, 5, 5, 2, 2]
        #        [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0]
        expected = 61
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_multiple_layer_platforms2(self):
        stones = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 1, 1, 1, 0]
        expected = 16  # [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0]
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_choose_between_two_pyramids(self):
        stones = [1, 2, 3, 2, 1, 0, 0, 1, 6, 1, 0]
        expected = 8  # [1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0]
        self.assertEqual(expected, min_cost_pyramid(stones))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 24, 2020 \[Easy\] Reconstruct a Jumbled Array
---
> **Question:** The sequence `[0, 1, ..., N]` has been jumbled, and the only clue you have for its order is an array representing whether each number is larger or smaller than the last. 
> 
> Given this information, reconstruct an array that is consistent with it. For example, given `[None, +, +, -, +]`, you could return `[1, 2, 3, 0, 4]`.

**My thoughts:** treat `+` as `+1`, `+2`, `+3` and `-` as `-1`, `-2` from `0` you can generate an array with satisfied condition yet incorrect range. Then all we need to do is to shift to range from `0` to `N` by minusing each element with global minimum value. 

**Solution:** [https://repl.it/@trsong/Reconstruct-a-Jumbled-Array](https://repl.it/@trsong/Reconstruct-a-Jumbled-Array)
```py
import unittest

def build_jumbled_array(clues):
    res = [0] * len(clues)
    upper = lower = 0

    for i in xrange(1, len(clues)):
        if clues[i] == '+':
            res[i] = upper + 1
            upper += 1
        elif clues[i] == '-':
            res[i] = lower - 1
            lower -= 1

    return map(lambda num: num - lower, res)


class BuildJumbledArraySpec(unittest.TestCase):
    @staticmethod
    def generate_clues(nums):
        nums_signs = [None] * len(nums)
        for i in xrange(1, len(nums)):
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
    unittest.main(exit=False)
```

### Nov 23, 2020 LC 286 \[Medium\] Walls and Gates
---
> **Question:** You are given a m x n 2D grid initialized with these three possible values.
> * -1 - A wall or an obstacle.
> * 0 - A gate.
> * INF - Infinity means an empty room. We use the value `2^31 - 1 = 2147483647` to represent INF as you may assume that the distance to a gate is less than 2147483647.
> 
> Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.

**Example:**

```py
Given the 2D grid:

INF  -1  0  INF
INF INF INF  -1
INF  -1 INF  -1
  0  -1 INF INF

After running your function, the 2D grid should be:

  3  -1   0   1
  2   2   1  -1
  1  -1   2  -1
  0  -1   3   4
  ```

**My thoughts:** Most of time, the BFS you are familiar with has only one starting point and searching from that point onward will produce the shortest path from start to visited points. For multi-starting points, it works exactly as single starting point. All you need to do is to imagine a single vitual starting point connecting to all starting points. Moreover, the way we achieve that is to put all starting point into the queue before doing BFS. 


**Solution with BFS:** [https://repl.it/@trsong/Identify-Walls-and-Gates](https://repl.it/@trsong/Identify-Walls-and-Gates)
```py
import unittest
import sys
from Queue import Queue

INF = sys.maxint

def nearest_gate(grid):
    if not grid or not grid[0]:
        return

    n, m = len(grid), len(grid[0])
    queue = Queue()
    for r in xrange(n):
        for c in xrange(m):
            if grid[r][c] == 0:
                queue.put((r, c))

    depth = 0
    DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    while not queue.empty():
        for _ in xrange(queue.qsize()):
            r, c = queue.get()
            if 0 < grid[r][c] < INF:
                continue
            grid[r][c] = depth

            for dr, dc in DIRECTIONS:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < n and 0 <= new_c < m and grid[new_r][new_c] == INF:
                    queue.put((new_r, new_c))

        depth += 1


class NearestGateSpec(unittest.TestCase):
    def test_example(self):
        grid = [
            [INF,  -1,   0, INF],
            [INF, INF, INF,  -1],
            [INF,  -1, INF,  -1],
            [  0,  -1, INF, INF]
        ]
        expected_grid = [
            [  3,  -1,   0,   1],
            [  2,   2,   1,  -1],
            [  1,  -1,   2,  -1],
            [  0,  -1,   3,   4]
        ]
        nearest_gate(grid)
        self.assertEqual(expected_grid, grid)

    def test_unreachable_room(self):
        grid = [
            [INF, -1],
            [ -1,  0]
        ]
        expected_grid = [
            [INF, -1],
            [ -1,  0]
        ]
        nearest_gate(grid)
        self.assertEqual(expected_grid, grid)

    def test_no_gate_exists(self):
        grid = [
            [-1,   -1],
            [INF, INF]
        ]
        expected_grid = [
            [-1,   -1],
            [INF, INF]
        ]
        nearest_gate(grid)
        self.assertEqual(expected_grid, grid)

    def test_all_gates_no_room(self):
        grid = [
            [0, 0, 0],
            [0, 0, 0]
        ]
        expected_grid = [
            [0, 0, 0],
            [0, 0, 0]
        ]
        nearest_gate(grid)
        self.assertEqual(expected_grid, grid)

    def test_empty_grid(self):
        grid = []
        nearest_gate(grid)
        self.assertEqual([], grid)

    def test_1D_grid(self):
        grid = [[INF, 0, INF, INF, INF, 0, INF, 0, 0, -1, INF]]
        expected_grid = [[1, 0, 1, 2, 1, 0, 1, 0, 0, -1, INF]]
        nearest_gate(grid)
        self.assertEqual(expected_grid, grid)

    def test_multi_gates(self):
        grid = [
            [INF, INF,  -1,   0, INF],
            [INF, INF, INF, INF, INF],
            [  0, INF, INF, INF,   0],
            [INF, INF,  -1, INF, INF]
        ]
        expected_grid = [
            [  2,   3,  -1,   0,   1],
            [  1,   2,   2,   1,   1],
            [  0,   1,   2,   1,   0],
            [  1,   2,  -1,   2,   1]
        ]
        nearest_gate(grid)
        self.assertEqual(expected_grid, grid)

    def test_at_center(self):
        grid = [
            [INF, INF, INF, INF, INF],
            [INF, INF, INF, INF, INF],
            [INF, INF,   0, INF, INF],
            [INF, INF, INF, INF, INF]
        ]
        expected_grid = [
            [  4,   3,   2,   3,   4],
            [  3,   2,   1,   2,   3],
            [  2,   1,   0,   1,   2],
            [  3,   2,   1,   2,   3]
        ]
        nearest_gate(grid)
        self.assertEqual(expected_grid, grid)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 22, 2020 LC 212 \[Hard\] Word Search II
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


**Solution with Backtracking and Trie:** [https://repl.it/@trsong/Word-Search-II](https://repl.it/@trsong/Word-Search-II)
```py
import unittest

def search_word(board, words):
    if not board or not board[0] or not words:
        return []

    trie = Trie()
    for word in words:
        trie.insert(word)

    res = []
    n, m = len(board), len(board[0])
    for r in xrange(n):
        for c in xrange(m):
            backtrack(board, trie, (r, c), res)

    return res


class Trie(object):
    def __init__(self):
        self.word = None
        self.children = None
    
    def insert(self, word):
        p = self
        for ch in word:
            if not p.children:
                p.children =  {}
            
            if ch not in p.children:
                p.children[ch] = Trie()

            p = p.children[ch]
        p.word = word

DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]          

def backtrack(board, parent_node, pos, res):
    r, c = pos
    ch = board[r][c]
    if not ch or not parent_node.children or not parent_node.children.get(ch, None):
        return
    
    node = parent_node.children[ch]
    if node.word:
        res.append(node.word)
        node.word = None
    
    board[r][c] = None
    n, m = len(board), len(board[0])
    for dr, dc in DIRECTIONS:
        new_r, new_c = r + dr, c + dc
        if 0 <= new_r < n and 0 <= new_c < m and board[new_r][new_c]:
            backtrack(board, node, (new_r, new_c), res)
    board[r][c] = ch


class SearchWordSpec(unittest.TestCase):
    def assert_result(self, expected, res):
        self.assertEqual(sorted(expected), sorted(res))

    def test_example(self):
        words = ['oath','pea','eat','rain']
        board = [
            ['o','a','a','n'],
            ['e','t','a','e'],
            ['i','h','k','r'],
            ['i','f','l','v']]
        expected = ['eat', 'oath']
        self.assert_result(expected, search_word(board, words))

    def test_example2(self):
        words = ['abcb']
        board = [
            ['a','b'],
            ['c','d']]
        expected = []
        self.assert_result(expected, search_word(board, words))

    def test_unique_char(self):
        words = ['a', 'aa', 'aaa']
        board = [
            ['a','a'],
            ['a','a']]
        expected = ['a', 'aa', 'aaa']
        self.assert_result(expected, search_word(board, words))

    def test_empty_grid(self):
        self.assertEqual([], search_word([], ['a']))

    def test_empty_empty_word(self):
        self.assertEqual([], search_word(['a'], []))

    def test_word_use_all_letters(self):
        words = ['abcdef']
        board = [
            ['a','b'],
            ['f','c'],
            ['e','d']]
        expected = ['abcdef']
        self.assert_result(expected, search_word(board, words))

    
if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 21, 2020 \[Hard\] Find the Element That Appears Once While Others Occur 3 Times
---
> **Question:** Given an array of integers where every integer occurs three times except for one integer, which only occurs once, find and return the non-duplicated integer.
>
> For example, given `[6, 1, 3, 3, 3, 6, 6]`, return `1`. Given `[13, 19, 13, 13]`, return `19`.
>
> Do this in `O(N)` time and `O(1)` space.

**My thoughts:** An interger repeats 3 time, then each of its digit will repeat 3 times. If a digit repeat 1 more time on top of that, then that digit must be contributed by the unique number. 

**Solution:** [https://repl.it/@trsong/Find-the-Element-That-Appears-Once-While-Others-Occur-3-Time](https://repl.it/@trsong/Find-the-Element-That-Appears-Once-While-Others-Occur-3-Time)
```py
import unittest

INT_SIZE = 32

def find_uniq_elem(nums):
    res = 0
    count = 0
    for i in xrange(INT_SIZE):
        count = 0
        for num in nums:
            if num & 1 << i:
                count += 1

        if count % 3 == 1:
            res |= 1 << i 
    return res


class FindUniqElemSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(1, find_uniq_elem([6, 1, 3, 3, 3, 6, 6]))

    def test_example2(self):
        self.assertEqual(19, find_uniq_elem([13, 19, 13, 13]))

    def test_example3(self):
        self.assertEqual(2, find_uniq_elem([12, 1, 12, 3, 12, 1, 1, 2, 3, 3]))

    def test_example4(self):
        self.assertEqual(20, find_uniq_elem([10, 20, 10, 30, 10, 30, 30]))

    def test_ascending_array(self):
        self.assertEqual(4, find_uniq_elem([1, 1, 1, 2, 2, 2, 3, 3, 3, 4]))

    def test_descending_array(self):
        self.assertEqual(2, find_uniq_elem([2, 1, 1, 1, 0, 0, 0]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 20, 2020 \[Easy\] Word Ordering in a Different Alphabetical Order
---
> **Question:** Given a list of words, and an arbitrary alphabetical order, verify that the words are in order of the alphabetical order.

**Example 1:**
```py
Input: 
words = ["abcd", "efgh"]
order="zyxwvutsrqponmlkjihgfedcba"

Output: False
Explanation: 'e' comes before 'a' so 'efgh' should come before 'abcd'
```

**Example 2:**
```py
Input:
words = ["zyx", "zyxw", "zyxwy"]
order="zyxwvutsrqponmlkjihgfedcba"

Output: True
Explanation: The words are in increasing alphabetical order
```

**Solution:** [https://repl.it/@trsong/Determine-Word-Ordering-in-a-Different-Alphabetical-Order](https://repl.it/@trsong/Determine-Word-Ordering-in-a-Different-Alphabetical-Order)
```py
import unittest

def is_sorted(words, order):
    char_rank = {ch: rank for rank, ch in enumerate(order)}
    prev = ""
    for word in words:
        is_smaller = False
        for ch1, ch2 in zip(prev, word):
            if char_rank[ch1] > char_rank[ch2]:
                return False

            if char_rank[ch1] < char_rank[ch2]:
                is_smaller = True
                break

        if not is_smaller and len(prev) > len(word):
            return False

        prev = word
    return True


class IsSortedSpec(unittest.TestCase):
    def test_example1(self):
        words = ["abcd", "efgh"]
        order = "zyxwvutsrqponmlkjihgfedcba"
        self.assertFalse(is_sorted(words, order))

    def test_example2(self):
        words = ["zyx", "zyxw", "zyxwy"]
        order = "zyxwvutsrqponmlkjihgfedcba"
        self.assertTrue(is_sorted(words, order))

    def test_empty_list(self):
        self.assertTrue(is_sorted([], ""))
        self.assertTrue(is_sorted([], "abc"))

    def test_one_elem_list(self):
        self.assertTrue(is_sorted(["z"], "xyz"))

    def test_empty_words(self):
        self.assertTrue(is_sorted(["", "", ""], ""))

    def test_word_of_different_length(self):
        words = ["", "1", "11", "111", "1111"]
        order = "4321"
        self.assertTrue(is_sorted(words, order))

    def test_word_of_different_length2(self):
        words = ["", "11", "", "111", "1111"]
        order = "1"
        self.assertFalse(is_sorted(words, order))

    def test_large_word_dictionary(self):
        words = ["123", "1a1b1A2ca", "ABC", "Aaa", "aaa", "bbb", "c11", "cCa"]
        order = "".join(map(chr, range(256)))
        self.assertTrue(is_sorted(words, order))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 19, 2020 \[Easy\] Count Visible Nodes in Binary Tree
---
> **Question:** In a binary tree, if in the path from root to the node A, there is no node with greater value than A’s, this node A is visible. We need to count the number of visible nodes in a binary tree.

**Example 1:**
```py
Input:
        5
     /     \
   3        10
  /  \     /
20   21   1

Output: 4
Explanation: There are 4 visible nodes: 5, 20, 21, and 10.
```

**Example 2:**
```py
Input:
  -10
    \
    -15
      \
      -1

Output: 2
Explanation: Visible nodes are -10 and -1.
```

**Solution with DFS:** [https://repl.it/@trsong/Count-Visible-Nodes-in-Binary-Tree](https://repl.it/@trsong/Count-Visible-Nodes-in-Binary-Tree)
```py
import unittest

def count_visible_nodes(root):
    if not root:
        return 0

    stack = [(root, float('-inf'))]
    res = 0
    while stack:
        cur, prev_max = stack.pop()
        if cur.val > prev_max:
            res += 1
        
        for child in [cur.left, cur.right]:
            if not child:
                continue
            stack.append((child, max(prev_max, cur.val)))
    
    return res


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class CountVisibleNodeSpec(unittest.TestCase):
    def test_example(self):
        """
                5*
             /     \
           3        10*
          /  \     /
        20*  21*  1
        """
        left_tree = TreeNode(3, TreeNode(20), TreeNode(21))
        right_tree = TreeNode(10, TreeNode(1))
        root = TreeNode(5, left_tree, right_tree)
        self.assertEqual(4, count_visible_nodes(root))

    def test_example2(self):
        """
         -10*
           \
           -15
             \
             -1*
        """
        root = TreeNode(-10, right=TreeNode(-15, TreeNode(-1)))
        self.assertEqual(2, count_visible_nodes(root))

    def test_empty_tree(self):
        self.assertEqual(0, count_visible_nodes(None))

    def test_full_tree(self):
        """
             1*
           /    \
          2*     3*
         / \    / \
        4*  5* 6*  7*
        """
        left_tree = TreeNode(2, TreeNode(4), TreeNode(5))
        right_tree = TreeNode(3, TreeNode(6), TreeNode(7))
        root = TreeNode(1, left_tree, right_tree)
        self.assertEqual(7, count_visible_nodes(root))

    def test_one_node_tree(self):
        self.assertEqual(1, count_visible_nodes(TreeNode(42)))

    def test_complete_tree(self):
        """
            10*
           /   \
          9     8
         / \   /
        7   6 5
        """
        left_tree = TreeNode(9, TreeNode(7), TreeNode(6))
        right_tree = TreeNode(8, TreeNode(5))
        root = TreeNode(10, left_tree, right_tree)
        self.assertEqual(1, count_visible_nodes(root))
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 18, 2020 \[Hard\] Non-adjacent Subset Sum
---
> **Question:** Given an array of size n with unique positive integers and a positive integer K,
check if there exists a combination of elements in the array satisfying both of below constraints:
> - The sum of all such elements is K
> - None of those elements are adjacent in the original array

**Example:**
```py
Input: K = 14, arr = [1, 9, 8, 3, 6, 7, 5, 11, 12, 4]
Output: [3, 7, 4]
```

**Solution with DP:** [https://repl.it/@trsong/Non-adjacent-Subset-Sum](https://repl.it/@trsong/Non-adjacent-Subset-Sum)
```py
import unittest

def non_adj_subset_sum(nums, k):
    if not nums:
        return None

    n = len(nums)
    # Let dp[k][n] represents whether exists subset sum k for nums[:n]
    # dp[k][n] = dp[k][n-1] or dp[k - nums[n-1]][n-2]
    dp = [[False for _ in xrange(n + 1)] for _ in xrange(k + 1)]
    for s in xrange(1, k+1):
        for i in xrange(1, n+1):
            if s == nums[i-1] or dp[s][i-1]:
                dp[s][i] = True
                continue
            if i > 2 and s >= nums[i-1]:
                dp[s][i] = dp[s - nums[i-1]][i-2]

    if not dp[k][n]:
        return None
    res = []
    for i in xrange(n, 0, -1):
        if dp[k][i-1]:
            continue

        num = nums[i-1]
        if k == num or i > 2 and dp[k - num][i - 2]:
            res.append(num)
            k -= num
    return res[::-1]
            

class NonAdjSubsetSumSpec(unittest.TestCase):
    def assert_result(self, k, nums, res):
        self.assertEqual(set(), set(res) - set(nums))
        self.assertEqual(k, sum(res))

    def test_example(self):
        k, nums = 14, [1, 9, 8, 3, 6, 7, 5, 11, 12, 4]
        # Possible solution [3, 7, 4]
        res = non_adj_subset_sum(nums, k) 
        self.assert_result(k, nums, res)

    def test_multiple_solution(self):
        k, nums = 12, [1, 2, 3, 4, 5, 6, 7]
        # Possible solution [2, 4, 6]
        res = non_adj_subset_sum(nums, k)
        self.assert_result(k, nums, res)

    def test_no_subset_satisfied(self):
        k, nums = 100, [1, 2]
        self.assertIsNone(non_adj_subset_sum(nums, k))

    def test_no_subset_satisfied2(self):
        k, nums = 3, [1, 2]
        self.assertIsNone(non_adj_subset_sum(nums, k))

    def test_should_not_pick_adjacent_elements(self):
        k, nums = 3, [1, 2, 3]
        expected = [3]
        self.assertEqual(expected, non_adj_subset_sum(nums, k))

    def test_should_not_pick_adjacent_elements2(self):
        k, nums = 4, [1, 2, 3]
        expected = [1, 3]
        self.assertEqual(expected, non_adj_subset_sum(nums, k))

    def test_pick_every_other_elements(self):
        k, nums = 11, [1, 90, 2, 80, 3, 100, 5]
        expected = [1, 2, 3, 5]
        self.assertEqual(expected, non_adj_subset_sum(nums, k))

    def test_pick_first_and_last(self):
        k, nums = 3, [1, 10, 11, 7, 4, 12, 2]
        expected = [1, 2]
        self.assertEqual(expected, non_adj_subset_sum(nums, k))

    def test_pick_every_three_elements(self):
        k, nums = 6, [1, 100, 109, 2, 101, 110, 3]
        expected = [1, 2, 3]
        self.assertEqual(expected, non_adj_subset_sum(nums, k))
  

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 17, 2020 \[Easy\] Count Number of Unival Subtrees
---
> **Question:** A unival tree is a tree where all the nodes have the same value. Given a binary tree, return the number of unival subtrees in the tree.

**Example 1:**
```py
The following tree should return 5:

   0
  / \
 1   0
    / \
   1   0
  / \
 1   1

The 5 trees are:
- The three single '1' leaf nodes. (+3)
- The single '0' leaf node. (+1)
- The [1, 1, 1] tree at the bottom. (+1)
```

**Example 2:**
```py
Input: root of below tree
              5
             / \
            1   5
           / \   \
          5   5   5
Output: 4
There are 4 subtrees with single values.
```

**Example 3:**
```py
Input: root of below tree
              5
             / \
            4   5
           / \   \
          4   4   5                
Output: 5
There are five subtrees with single values.
```

**Solution with Postorder Traversal:** [https://repl.it/@trsong/Count-Total-Number-of-Uni-val-Subtrees](https://repl.it/@trsong/Count-Total-Number-of-Uni-val-Subtrees)
```py
import unittest

class Result(object):
    def __init__(self, res, is_unival):
        self.res = res
        self.is_unival = is_unival


def count_unival_subtrees(tree):
    return count_unival_subtrees_recur(tree).res


def count_unival_subtrees_recur(tree):
    if not tree:
        return Result(0, True)

    left_res = count_unival_subtrees_recur(tree.left)
    right_res = count_unival_subtrees_recur(tree.right)
    is_current_unival = left_res.is_unival and left_res.is_unival
    if is_current_unival and tree.left and tree.left.val != tree.val:
        is_current_unival = False
    if is_current_unival and tree.right and tree.right.val != tree.val:
        is_current_unival = False

    current_count = left_res.res + right_res.res + (1 if is_current_unival else 0)
    return Result(current_count, is_current_unival)


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class CountUnivalSubTreeSpec(unittest.TestCase):
    def test_example1(self):
        """
           0
          / \
         1   0
            / \
           1   0
          / \
         1   1
        """
        rl = TreeNode(1, TreeNode(1), TreeNode(1))
        r = TreeNode(0, rl, TreeNode(0))
        root = TreeNode(0, TreeNode(1), r)
        self.assertEqual(5, count_unival_subtrees(root))

    def test_example2(self):
        """
              5
             / \
            1   5
           / \   \
          5   5   5
        """
        l = TreeNode(1, TreeNode(5), TreeNode(5))
        r = TreeNode(5, right=TreeNode(5))
        root = TreeNode(5, l, r)
        self.assertEqual(4, count_unival_subtrees(root))

    def test_example3(self):
        """
              5
             / \
            4   5
           / \   \
          4   4   5  
        """
        l = TreeNode(4, TreeNode(4), TreeNode(4))
        r = TreeNode(5, right=TreeNode(5))
        root = TreeNode(5, l, r)
        self.assertEqual(5, count_unival_subtrees(root))

    def test_empty_tree(self):
        self.assertEqual(0, count_unival_subtrees(None))

    def test_left_heavy_tree(self):
        """
            1
           /
          1
         / \ 
        1   0
        """
        root = TreeNode(1, TreeNode(1, TreeNode(1), TreeNode(0)))
        self.assertEqual(2, count_unival_subtrees(root))

    def test_right_heavy_tree(self):
        """
          0
         / \
        1   0
             \
              0
               \
                0
        """
        rr = TreeNode(0, right=TreeNode(0))
        r = TreeNode(0, right=rr)
        root = TreeNode(0, TreeNode(1), r)
        self.assertEqual(4, count_unival_subtrees(root))

    def test_unival_tree(self):
        """
            0
           / \
          0   0
         /   /
        0   0          
        """
        l = TreeNode(0, TreeNode(0))
        r = TreeNode(0, TreeNode(0))
        root = TreeNode(0, l, r)
        self.assertEqual(5, count_unival_subtrees(root))

    def test_distinct_value_trees(self):
        """
               _0_
              /   \
             1     2
            / \   / \
           3   4 5   6
          /
         7
        """
        n1 = TreeNode(1, TreeNode(3, TreeNode(7)), TreeNode(4))
        n2 = TreeNode(2, TreeNode(5), TreeNode(6))
        n0 = TreeNode(0, n1, n2)
        self.assertEqual(4, count_unival_subtrees(n0))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 16, 2020 LC 1647 \[Medium\] Minimum Deletions to Make Character Frequencies Unique
---
> **Question:** A string s is called good if there are no two different characters in s that have the same frequency.
>
> Given a string s, return the minimum number of characters you need to delete to make s good.
>
> The frequency of a character in a string is the number of times it appears in the string. For example, in the string "aab", the frequency of 'a' is 2, while the frequency of 'b' is 1.


**Example 1:**
```py
Input: s = "aab"
Output: 0
Explanation: s is already good.
```
**Example 2:**
```py
Input: s = "aaabbbcc"
Output: 2
Explanation: You can delete two 'b's resulting in the good string "aaabcc".
Another way it to delete one 'b' and one 'c' resulting in the good string "aaabbc".
```
**Example 3:**
```py
Input: s = "ceabaacb"
Output: 2
Explanation: You can delete both 'c's resulting in the good string "eabaab".
Note that we only care about characters that are still in the string at the end (i.e. frequency of 0 is ignored).
```

**My thoughts:** sort frequency in descending order, while iterate through all frequencies, keep track of biggest next frequency we can take. Then the min deletion for that letter is `freq - biggestNextFreq`. Remember to reduce the biggest next freq by 1 for each step.  

**Greedy Solution**: [https://repl.it/@trsong/Minimum-Deletions-to-Make-Character-Frequencies-Unique](https://repl.it/@trsong/Minimum-Deletions-to-Make-Character-Frequencies-Unique)
```py
import unittest

CHAR_SIZE = 26

def min_deletions(s):
    histogram = [0] * CHAR_SIZE
    ord_a = ord('a')
    for ch in s:
        histogram[ord(ch) - ord_a] += 1

    histogram.sort(reverse=True)
    next_count = histogram[0]
    res = 0

    for count in histogram:
        if count <= next_count:
            next_count = count - 1
        else:
            # reduce count to next_count
            res += count - next_count
            next_count -= 1    
        next_count = max(0, next_count)
    return res


class MinDeletionSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(0, min_deletions("aab"))
        
    def test_example2(self):
        # remove 2b's
        self.assertEqual(2, min_deletions("aaabbbcc"))
        
    def test_example3(self):
        # remove 2b's
        self.assertEqual(2, min_deletions("ceabaacb"))
        
    def test_empty_string(self):
        self.assertEqual(0, min_deletions(""))
        
    def test_string_with_same_char_freq(self):
        s = 'a' * 100 + 'b' * 100 + 'c' * 2 + 'd' * 1
        self.assertEqual(1, min_deletions(s))
        
    def test_remove_all_other_string(self):
        self.assertEqual(4, min_deletions("abcde"))
        
    def test_collision_after_removing(self):
        # remove 1b, 1c, 2d, 2e, 1f 
        s = 'a' * 10 + 'b' * 10 + 'c' * 9 + 'd' * 9 + 'e' * 8 + 'f' * 6
        self.assertEqual(7, min_deletions(s))

    def test_remove_all_of_certain_letters(self):
        # remove 3b, 1f
        s = 'a' * 3 + 'b' * 3 + 'c' * 2 + 'd' + 'f' 
        self.assertEqual(4, min_deletions(s))
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 15, 2020 \[Medium\] Number of Flips to Make Binary String
---
> **Question:** You are given a string consisting of the letters `x` and `y`, such as `xyxxxyxyy`. In addition, you have an operation called flip, which changes a single `x` to `y` or vice versa.
>
> Determine how many times you would need to apply this operation to ensure that all x's come before all y's. In the preceding example, it suffices to flip the second and sixth characters, so you should return 2.

**My thoughts:** Basically, the question is about finding a sweet cutting spot so that # flip on left plus # flip on right is minimized. We can simply scan through the array from left and right to allow constant time query for number of flip need on the left and right for a given spot. And the final answer is just the min of sum of left and right flips.


**Solution with DP:** [https://repl.it/@trsong/Find-Number-of-Flips-to-Make-Binary-String](https://repl.it/@trsong/Find-Number-of-Flips-to-Make-Binary-String)
```py
import unittest

def min_flip_to_make_binary(s):
    if not s:
        return 0

    n = len(s)
    left_y_count = 0 
    right_x_count = 0
    left_accu = [0] * n
    right_accu = [0] * n
    for i in xrange(n):
        left_accu[i] = left_y_count
        left_y_count += 1 if s[i] == 'y' else 0

        right_accu[n - 1 - i] = right_x_count
        right_x_count += 1 if s[n - 1 - i] == 'x' else 0
    
    res = float('inf')
    for left_y, right_x in zip(left_accu, right_accu):
        res = min(res, left_y + right_x)

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
    unittest.main(exit=False)
```

### Nov 14, 2020 \[Medium\] Isolated Islands
---
> **Question:** Given a matrix of 1s and 0s, return the number of "islands" in the matrix. A 1 represents land and 0 represents water, so an island is a group of 1s that are neighboring whose perimeter is surrounded by water.
>
> For example, this matrix has 4 islands.

```py
1 0 0 0 0
0 0 1 1 0
0 1 1 0 0
0 0 0 0 0
1 1 0 0 1
1 1 0 0 1
```

**Solution with DFS:** [https://repl.it/@trsong/Count-Number-of-Isolated-Islands](https://repl.it/@trsong/Count-Number-of-Isolated-Islands)
```py
import unittest

DIRECTIONS = [-1, 0, 1]

def calc_islands(area_map):
    if not area_map or not area_map[0]:
        return 0

    n, m = len(area_map), len(area_map[0])
    visited = set()
    res = 0
    for r in xrange(n):
        for c in xrange(m):
            if area_map[r][c] == 0 or (r, c) in visited:
                continue
            res += 1
            dfs_island(area_map, (r, c), visited)
    return res


def dfs_island(area_map, pos, visited):
    n, m = len(area_map), len(area_map[0])
    stack = [pos]
    while stack:
        cur_r, cur_c = stack.pop()
        if (cur_r, cur_c) in visited:
            continue
        visited.add((cur_r, cur_c))
            
        for dr in DIRECTIONS:
            for dc in DIRECTIONS:
                new_r, new_c = cur_r + dr, cur_c + dc
                if (0 <= new_r < n and 0 <= new_c < m and 
                    area_map[new_r][new_c] == 1 and 
                    (new_r, new_c) not in visited):
                    stack.append((new_r, new_c))


class CalcIslandSpec(unittest.TestCase):
    def test_sample_area_map(self):
        self.assertEqual(4, calc_islands([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1]
        ]))
    
    def test_some_random_area_map(self):
        self.assertEqual(5, calc_islands([
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1] 
        ]))

    def test_island_edge_of_map(self):
        self.assertEqual(5, calc_islands([
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1] 
        ]))

    def test_huge_water(self):
        self.assertEqual(0, calc_islands([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]))

    def test_huge_island(self):
        self.assertEqual(1, calc_islands([
            [1, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
            [1, 1, 1, 0, 1]
        ]))

    def test_non_square_island(self):
        self.assertEqual(1, calc_islands([
            [1],
            [1],
            [1]
        ]))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 13, 2020 \[Medium\] Find Missing Positive
---
> **Question:** Given an unsorted integer array, find the first missing positive integer.

**Example 1:**
```py
Input: [1, 2, 0]
Output: 3
```

**Example 2:**
```py
Input: [3, 4, -1, 1]
Output: 2
```

**My thougths:** Ideally each positive number should map to the same index as its `value - 1`. So all we need to do is for each postion, use its value as index and swap with that element until we find correct number. Keep doing this and each postive should store in postion of its `value - 1`.  Now we just scan through the entire array until find the first missing number by checking each element's value against index. 


**Soluion:** [https://repl.it/@trsong/Find-First-Missing-Positive](https://repl.it/@trsong/Find-First-Missing-Positive)
```py
import unittest

def find_missing_positive(nums):
    n = len(nums)
    value_to_index = lambda value: value - 1

    for i in xrange(n):
        while 1 <= nums[i] <= n and value_to_index(nums[i]) != i:
            target_index = value_to_index(nums[i])
            if nums[i] == nums[target_index]:
                break
            nums[i], nums[target_index] = nums[target_index], nums[i]

    for i, num in enumerate(nums):
        if value_to_index(num) != i:
            return i + 1

    return n + 1
    

class FindMissingPositiveSpec(unittest.TestCase):
    def test_example(self):
        nums = [1, 2, 0]
        expected = 3
        self.assertEqual(expected, find_missing_positive(nums))

    def test_example2(self):
        nums = [3, 4, -1, 1]
        expected = 2
        self.assertEqual(expected, find_missing_positive(nums))
    
    def test_empty_array(self):
        nums = []
        expected = 1
        self.assertEqual(expected, find_missing_positive(nums))

    def test_all_non_positives(self):
        nums = [-1, 0, -1, -2, -1, -3, -4]
        expected = 1
        self.assertEqual(expected, find_missing_positive(nums))

    def test_number_out_of_range(self):
        nums = [101, 102, 103]
        expected = 1
        self.assertEqual(expected, find_missing_positive(nums))

    def test_duplicated_numbers(self):
        nums = [1, 1, 3, 3, 2, 2, 5]
        expected = 4
        self.assertEqual(expected, find_missing_positive(nums))

    def test_missing_positive_falls_out_of_range(self):
        nums = [5, 4, 3, 2, 1]
        expected = 6
        self.assertEqual(expected, find_missing_positive(nums))

    def test_number_off_by_one_position(self):
        nums = [0, 2, 3, 4, 7, 6, 1]
        expected = 5
        self.assertEqual(expected, find_missing_positive(nums))

    def test_positive_and_negative_numbers(self):
        nums = [-1, -3, -2, 0, 1, 2, 4, -4, 5, -6, 7]
        expected = 3
        self.assertEqual(expected, find_missing_positive(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 12, 2020 \[Medium\] Max Value of Coins to Collect in a Matrix
---
> **Question:** You are given a 2-d matrix where each cell represents number of coins in that cell. Assuming we start at `matrix[0][0]`, and can only move right or down, find the maximum number of coins you can collect by the bottom right corner.

**Example:**

```py
Given below matrix:

0 3 1 1
2 0 0 4
1 5 3 1

The most we can collect is 0 + 2 + 1 + 5 + 3 + 1 = 12 coins.
```

**My thoughts:** This problem gives you a strong feeling that this must be a DP question. 'coz for each step you can either move right or down, that is, max number of coins you can collect so far at current cell depends on top and left solution gives the following recurrence formula: 

```py
Let dp[i][j] be the max coin value collect when reach cell (i, j) in grid.
dp[i][j] = grid[i][j] + max(dp[i-1][j], dp[i][j-1])
```

You can also do it in-place using original grid. However, mutating input params in general is a bad habit as those parameters may be used in other place and might be immutable.


**Solution with DP:** [https://repl.it/@trsong/Find-Max-Value-of-Coins-to-Collect-in-a-Matrix](https://repl.it/@trsong/Find-Max-Value-of-Coins-to-Collect-in-a-Matrix)
```py
import unittest

def max_coins(grid):
    if not grid or not grid[0]:
        return 0

    n, m = len(grid), len(grid[0])
    dp = [[0 for _ in xrange(m)] for _ in xrange(n)]

    for r in xrange(n):
        for c in xrange(m):
            left_max = dp[r-1][c] if r > 0 else 0
            top_max = dp[r][c-1] if c > 0 else 0
            dp[r][c] = grid[r][c] + max(left_max, top_max)

    return dp[n-1][m-1]


class MaxCoinSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(max_coins([
            [0, 3, 1, 1],
            [2, 0, 0, 4],
            [1, 5, 3, 1]
        ]), 12)

    def test_empty_grid(self):
        self.assertEqual(max_coins([]), 0)
        self.assertEqual(max_coins([[]]), 0)

    def test_one_way_or_the_other(self):
        self.assertEqual(max_coins([
            [0, 3],
            [2, 0]
        ]), 3)

    def test_until_the_last_moment_knows(self):
        self.assertEqual(max_coins([
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [2, 0, 3, 0]
        ]), 5)

    def test_try_to_get_most_coins(self):
        self.assertEqual(max_coins([
            [1, 1, 1],
            [2, 3, 1],
            [1, 4, 5]
        ]), 15)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 11, 2020 LC 859 \[Easy\] Buddy Strings
---
> **Question:** Given two strings A and B of lowercase letters, return true if and only if we can swap two letters in A so that the result equals B.

**Example 1:**
```py
Input: A = "ab", B = "ba"
Output: true
```

**Example 2:**
```py
Input: A = "ab", B = "ab"
Output: false
```

**Example 3:**
```py
Input: A = "aa", B = "aa"
Output: true
```

**Example 4:**
```py
Input: A = "aaaaaaabc", B = "aaaaaaacb"
Output: true
```

**Example 5:**
```py
Input: A = "", B = "aa"
Output: false
```

**Solution:** [https://repl.it/@trsong/Buddy-Strings](https://repl.it/@trsong/Buddy-Strings)
```py
import unittest

def is_buddy_string(s1, s2):
    if len(s1) != len(s2):
        return False

    unmatches = []
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            unmatches.append((c1, c2))
        if len(unmatches) > 2:
            return False

    if len(unmatches) == 2:
        return unmatches[0] == unmatches[1][::-1]
    elif len(unmatches) == 0:
        return has_duplicates(s1)
    else:
        return False 


def has_duplicates(s):
    visited = set()
    for c in s:
        if c in visited:
            return True
        visited.add(c)
    return False


class IsBuddyString(unittest.TestCase):
    def test_example(self):
        self.assertTrue(is_buddy_string('ab', 'ba'))

    def test_example2(self):
        self.assertFalse(is_buddy_string('ab', 'ab'))

    def test_example3(self):
        self.assertTrue(is_buddy_string('aa', 'aa'))

    def test_example4(self):
        self.assertTrue(is_buddy_string('aaaaaaabc', 'aaaaaaacb'))

    def test_example5(self):
        self.assertFalse(is_buddy_string('ab', 'aa'))

    def test_empty_string(self):
        self.assertFalse(is_buddy_string('', ''))

    def test_single_char_string(self):
        self.assertFalse(is_buddy_string('a', 'b'))

    def test_same_string_without_duplicates(self):
        self.assertFalse(is_buddy_string('abc', 'abc'))

    def test_string_with_duplicates(self):
        self.assertFalse(is_buddy_string('aba', 'abc'))

    def test_different_length_string(self):
        self.assertFalse(is_buddy_string('aa', 'aaa'))

    def test_different_length_string2(self):
        self.assertFalse(is_buddy_string('ab', 'baa'))

    def test_different_length_string3(self):
        self.assertFalse(is_buddy_string('ab', 'abba'))

    def test_swap_failure(self):
        self.assertFalse(is_buddy_string('abcaa', 'abcbb'))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 10, 2020 \[Medium\] All Root to Leaf Paths in Binary Tree
---
> **Question:** Given a binary tree, return all paths from the root to leaves.

**Example:** 
```py
Given the tree:
   1
  / \
 2   3
    / \
   4   5

Return [[1, 2], [1, 3, 4], [1, 3, 5]]
```

**Solution with Backtracking:** [https://repl.it/@trsong/Print-All-Root-to-Leaf-Paths-in-Binary-Tree](https://repl.it/@trsong/Print-All-Root-to-Leaf-Paths-in-Binary-Tree)
```py
import unittest

def path_to_leaves(tree):
    if not tree:
        return []
    res = []
    backtrack(res, tree, [])
    return res


def backtrack(res, current_node, path):
    if not current_node.left and not current_node.right:
        res.append(path + [current_node.val])
    else:
        for child in [current_node.left, current_node.right]:
            if not child:
                continue
            path.append(current_node.val)
            backtrack(res, child, path)
            path.pop()

    
class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right= right


class PathToLeavesSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertEqual([], path_to_leaves(None))

    def test_one_level_tree(self):
        self.assertEqual([[1]], path_to_leaves(TreeNode(1)))

    def test_two_level_tree(self):
        """
          1
         / \
        2   3
        """
        tree = TreeNode(1, TreeNode(2), TreeNode(3))
        expected = [[1, 2], [1, 3]]
        self.assertEqual(expected, path_to_leaves(tree))

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
        expected = [[1, 2], [1, 3, 4], [1, 3, 5]]
        self.assertEqual(expected, path_to_leaves(tree))

    def test_complete_tree(self):
        """
               1
             /   \
            2     3
           / \   /
          4   5 6
        """
        left_tree = TreeNode(2, TreeNode(4), TreeNode(5))
        right_tree = TreeNode(3, TreeNode(6))
        tree = TreeNode(1, left_tree, right_tree)
        expected = [[1, 2, 4], [1, 2, 5], [1, 3, 6]]
        self.assertEqual(expected, path_to_leaves(tree))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 9, 2020 \[Medium\] Invert a Binary Tree
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

**Solution:** [https://repl.it/@trsong/Invert-All-Nodes-in-Binary-Tree](https://repl.it/@trsong/Invert-All-Nodes-in-Binary-Tree)
```py
import unittest

def invert_tree(root):
    if root:
        root.left, root.right = invert_tree(root.right), invert_tree(root.left)
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
    unittest.main(exit=False)
```

### Nov 8, 2020 \[Medium\] Similar Websites
---
> **Question:** You are given a list of (website, user) pairs that represent users visiting websites. Come up with a program that identifies the top k pairs of websites with the greatest similarity.
>
> **Note:** The similarity metric bewtween two sets equals intersection / union. 

**Example:**
```py
Suppose k = 1, and the list of tuples is:

[('a', 1), ('a', 3), ('a', 5),
 ('b', 2), ('b', 6),
 ('c', 1), ('c', 2), ('c', 3), ('c', 4), ('c', 5),
 ('d', 4), ('d', 5), ('d', 6), ('d', 7),
 ('e', 1), ('e', 3), ('e', 5), ('e', 6)]
 
Then a reasonable similarity metric would most likely conclude that a and e are the most similar, so your program should return [('a', 'e')].
```

**My thoughts:** The similarity metric bewtween two sets equals intersection / union. So, the way to get top k similar website is first calculate the similarity score between any two websites and after that use a priority queue to mantain top k similarity pairs.

However, as duplicate entry might occur, we have to treat normal set to multiset: treat `3, 3, 3` as `3(first), 3(second), 3(third)`. 
```py
a: 1 2 3(first) 3(second) 3(third)
b: 1 2 3(first)
The similarty between (a, b) is 3/5
```


**Solution with Priority Queue:** [https://repl.it/@trsong/Find-Similar-Websites](https://repl.it/@trsong/Find-Similar-Websites)
```py
import unittest
from collections import defaultdict
from Queue import PriorityQueue

def top_similar_websites(website_log, k):
    user_site_hits = defaultdict(lambda: defaultdict(int))
    site_hits = defaultdict(int)
    for site, user in website_log:
        user_site_hits[user][site] += 1
        site_hits[site] += 1

    cross_site_hits = defaultdict(lambda: defaultdict(int))
    for site_hits_per_user in user_site_hits.values():
        for site1 in site_hits_per_user:
            for site2 in site_hits_per_user:
                cross_site_hits[site1][site2] += min(site_hits_per_user[site1], site_hits_per_user[site2])

    min_heap = PriorityQueue()
    sites = sorted(site_hits.keys())
    for site1 in sites:
        for site2 in sites:
            if site2 == site1:
                break
            intersection = cross_site_hits[site1][site2]
            total = site_hits[site1] + site_hits[site2]
            union = total - intersection
            similarity = float(intersection) / union

            if min_heap.qsize() >= k and min_heap.queue[0][0] < similarity:
                min_heap.get()
            
            if min_heap.qsize() < k:
                min_heap.put((similarity, (site1, site2)))

    ascending_sites = [min_heap.get()[1] for _ in xrange(k)]
    return ascending_sites[::-1]


class TopSimilarWebsiteSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        # same length
        self.assertEqual(len(expected), len(result))
        for e, r in zip(expected, result):
            # pair must be the same, order doesn't matter
            self.assertEqual(set(e), set(r), "Expected %s but get %s" % (expected, result))

    def test_example(self):
        website_log = [
            ('a', 1), ('a', 3), ('a', 5),
            ('b', 2), ('b', 6),
            ('c', 1), ('c', 2), ('c', 3), ('c', 4), ('c', 5),
            ('d', 4), ('d', 5), ('d', 6), ('d', 7),
            ('e', 1), ('e', 3), ('e', 5), ('e', 6)]
        # Similarity: (a,e)=3/4, (a,c)=3/5, (c, e)=1/2
        expected = [('a', 'e'), ('a', 'c'), ('c', 'e')]
        self.assert_result(expected, top_similar_websites(website_log, len(expected)))

    def test_no_overlapping(self):
        website_log = [('a', 1), ('b', 2)]
        expected = [('a', 'b')]
        self.assert_result(expected, top_similar_websites(website_log, len(expected)))
    
    def test_should_return_correct_order(self):
        website_log = [
            ('a', 1),
            ('b', 1), ('b', 2),
            ('c', 1), ('c', 2), ('c', 3), 
            ('d', 1), ('d', 2), ('d', 3), ('d', 4),
            ('e', 1), ('e', 2), ('e', 3), ('e', 4), ('e', 5)]
        # Similarity: (d,e)=4/5, (c,d)=3/4, (b,c)=2/3, (c,e)=3/5
        expected = [('d', 'e'), ('c', 'd'), ('b', 'c'), ('c', 'e')]
        self.assert_result(expected, top_similar_websites(website_log, len(expected)))
        
    def test_duplicated_entries(self):
        website_log = [
            ('a', 1), ('a', 1),
            ('b', 1),
            ('c', 1), ('c', 1), ('c', 2),
            ('d', 1), ('d', 3), ('d', 3), ('d', 4),
            ('e', 1), ('e', 1), ('e', 5), ('e', 6),
            ('f', 1), ('f', 7), ('f', 8), ('f', 8)
        ]
        # Similarity: (a,c)=2/3
        expected = [('a', 'c')]
        self.assert_result(expected, top_similar_websites(website_log, len(expected)))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 7, 2020 \[Medium\] Largest Square
---
> **Question:** Given an N by M matrix consisting only of 1's and 0's, find the largest square matrix containing only 1's and return its dimension size.

**Example:**
```py
Given the following matrix:

[[1, 0, 0, 0],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [0, 1, 0, 0]]

Return 2. As the following 1s form the largest square matrix containing only 1s:
 [1, 1],
 [1, 1]
```

**Solution with DP:** [https://repl.it/@trsong/Largest-Square](https://repl.it/@trsong/Largest-Square)
```py
import unittest

def largest_square_dimension(grid):
    if not grid or not grid[0]:
        return 0
    n, m = len(grid), len(grid[0])

    # Let dp[r][c] represents max dimension of square matrix with bottom right corner at (r, c)
    # dp[r][c] = min(dp[r-1][c], dp[r][c-1], dp[r-1][c-1]) + 1 if (r, c) is 1
    #          = 0 otherwise
    dp = [[grid[r][c] for c in xrange(m)] for r in xrange(n)]

    for r in xrange(1, n):
        for c in xrange(1, m):
            if grid[r][c] == 0:
                continue
            dp[r][c] = 1 + min(dp[r-1][c], dp[r][c-1], dp[r-1][c-1])
    
    return max(map(max, dp))


class LargestSquareDimensionSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(2, largest_square_dimension([
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 0, 0]
        ]))

    def test_empty_grid(self):
        self.assertEqual(0, largest_square_dimension([]))

    def test_1d_grid(self):
        self.assertEqual(1, largest_square_dimension([
            [0, 0, 0, 0, 1, 0, 0]
        ]))

    def test_1d_grid2(self):
        self.assertEqual(0, largest_square_dimension([
            [0],
            [0],
            [0]
        ]))

    def test_dimond_shape(self):
        self.assertEqual(3, largest_square_dimension([
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0]
        ]))

    def test_dimond_shape2(self):
        self.assertEqual(3, largest_square_dimension([
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0]
        ]))

    def test_square_on_edge(self):
        self.assertEqual(2, largest_square_dimension([
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 0]
        ]))

    def test_dense_matrix(self):
        self.assertEqual(3, largest_square_dimension([
            [0, 1, 1, 0, 1], 
            [1, 1, 0, 1, 0], 
            [0, 1, 1, 1, 0], 
            [1, 1, 1, 1, 0], 
            [1, 1, 1, 1, 1], 
            [0, 0, 0, 0, 0]
        ]))



if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 6, 2020 \[Medium\] M Smallest in K Sorted Lists
---
> **Question:** Given k sorted arrays of possibly different sizes, find m-th smallest value in the merged array.

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

**Solution:** [https://repl.it/@trsong/Find-the-M-thSmallest-in-K-Sorted-Lists](https://repl.it/@trsong/Find-the-M-thSmallest-in-K-Sorted-Lists)
```py
import unittest
from Queue import PriorityQueue

def find_m_smallest(ksorted_list, m):
    pq = PriorityQueue()
    for lst in ksorted_list:
        if not lst:
            continue
        it = iter(lst)
        pq.put((it.next(), it))

    while not pq.empty():
        num, it = pq.get()
        if m == 1:
            return num
        m -= 1

        next_num = next(it, None)
        if next_num is not None:
            pq.put((next_num, it))
    
    return None
        

class FindMSmallestSpec(unittest.TestCase):
    def test_example1(self):
        m, ksorted_list = 5, [
            [1, 3],
            [2, 4, 6],
            [0, 9, 10, 11]
        ]
        expected = 4
        self.assertEqual(expected, find_m_smallest(ksorted_list, m))

    def test_example2(self):
        m, ksorted_list = 2, [
            [1, 3, 20],
            [2, 4, 6]
        ]
        expected = 2
        self.assertEqual(expected, find_m_smallest(ksorted_list, m))

    def test_example3(self):
        m, ksorted_list = 6, [
            [1, 3, 20],
            [2, 4, 6]
        ]
        expected = 20
        self.assertEqual(expected, find_m_smallest(ksorted_list, m))

    def test_empty_sublist(self):
        m, ksorted_list = 2, [
            [1],
            [],
            [0, 2]
        ]
        expected = 1
        self.assertEqual(expected, find_m_smallest(ksorted_list, m))

    def test_one_sublist(self):
        m, ksorted_list = 5, [
            [1, 2, 3, 4, 5],
        ]
        expected = 5
        self.assertEqual(expected, find_m_smallest(ksorted_list, m))

    def test_target_out_of_boundary(self):
        m, ksorted_list = 7, [
            [1, 2, 3],
            [4, 5, 6]
        ]
        self.assertIsNone(find_m_smallest(ksorted_list, m))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 5, 2020 \[Easy\] Maximum Subarray Sum 
---
> **Question:** You are given a one dimensional array that may contain both positive and negative integers, find the sum of contiguous subarray of numbers which has the largest sum.
>
> For example, if the given array is `[-2, -5, 6, -2, -3, 1, 5, -6]`, then the maximum subarray sum is 7 as sum of `[6, -2, -3, 1, 5]` equals 7
>
> Solve this problem with Divide and Conquer as well as DP separately.


**Solution with Divide and Conquer:** [https://repl.it/@trsong/Maximum-Subarray-Sum-Divide-and-Conquer](https://repl.it/@trsong/Maximum-Subarray-Sum-Divide-and-Conquer)
```py
def max_sub_array_sum(nums):
    if not nums:
        return 0
    res = max_sub_array_sum_recur(nums, 0, len(nums) - 1)
    return res.max


def max_sub_array_sum_recur(nums, lo, hi):
    if lo == hi:
        return Result(nums[lo])
    
    mid = lo + (hi - lo) // 2
    left_res = max_sub_array_sum_recur(nums, lo, mid)
    right_res = max_sub_array_sum_recur(nums, mid+1, hi)

    res = Result(0)
    # From left to right, expand range
    res.prefix = max(0, left_res.prefix, left_res.sum + right_res.prefix, left_res.sum + right_res.sum)
    # From right to left, expand range
    res.suffix = max(0, right_res.suffix, right_res.sum + left_res.suffix, right_res.sum + left_res.sum)
    res.sum = left_res.sum + right_res.sum
    res.max = max(res.prefix, res.suffix, left_res.suffix + right_res.prefix, left_res.max, right_res.max)
    return res


class Result(object):
    def __init__(self, res):
        self.prefix = res
        self.suffix = res
        self.sum = res
        self.max = res
```

**Solution with DP:** [https://repl.it/@trsong/Maximum-Subarray-Sum-DP](https://repl.it/@trsong/Maximum-Subarray-Sum-DP)
```py
import unittest

def max_sub_array_sum(nums):
    n = len(nums)
    # Let dp[i] represents max sub array sum ends at nums[i-1]
    # dp[i] = max(0, dp[i-1] + nums[i-1])
    dp = [0] * (n + 1)
    res = 0
    for i in xrange(1, n+1):
        dp[i] = max(0, dp[i-1] + nums[i-1])
        res = max(res, dp[i])
    return res


class MaxSubArraySum(unittest.TestCase):
    def test_empty_array(self):
        self.assertEqual(0, max_sub_array_sum([]))

    def test_ascending_array(self):
        self.assertEqual(6, max_sub_array_sum([-3, -2, -1, 0, 1, 2, 3]))
        
    def test_descending_array(self):
        self.assertEqual(6, max_sub_array_sum([3, 2, 1, 0, -1]))

    def test_example_array(self):
        self.assertEqual(7, max_sub_array_sum([-2, -5, 6, -2, -3, 1, 5, -6]))

    def test_negative_array(self):
        self.assertEqual(0, max_sub_array_sum([-2, -1]))

    def test_positive_array(self):
        self.assertEqual(3, max_sub_array_sum([1, 2]))

    def test_swing_array(self):
        self.assertEqual(5, max_sub_array_sum([-3, 3, -2, 2, -5, 5]))
        self.assertEqual(1, max_sub_array_sum([-1, 1, -1, 1, -1]))
        self.assertEqual(2, max_sub_array_sum([-100, 1, -100, 2, -100]))

    def test_converging_array(self):
        self.assertEqual(4, max_sub_array_sum([-3, 3, -2, 2, 1, 0]))

    def test_positive_negative_positive_array(self):
        self.assertEqual(8, max_sub_array_sum([7, -1, -2, 3, 1]))
        self.assertEqual(7, max_sub_array_sum([7, -1, -2, 0, 1, 1]))

    def test_negative_positive_array(self):
        self.assertEqual(3, max_sub_array_sum([-100, 1, 0, 2, -100]))
  

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 4, 2020 \[Easy\] Largest Path Sum from Root To Leaf
---
> **Question:** Given a binary tree, find and return the largest path from root to leaf.

**Example:**
```py
Input:
    1
  /   \
 3     2
  \   /
   5 4
Output: [1, 3, 5]
```

**Solution with DFS:** [https://repl.it/@trsong/Find-Largest-Path-Sum-from-Root-To-Leaf](https://repl.it/@trsong/Find-Largest-Path-Sum-from-Root-To-Leaf)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def largest_sum_path(root):
    if not root:
        return []
    parent_lookup = {}
    stack = [(root, 0)]
    max_sum = float('-inf')
    max_sum_node = root

    while stack:
        cur, prev_sum = stack.pop()
        cur_sum = prev_sum + cur.val
        if not cur.left and not cur.right and cur_sum > max_sum:
            max_sum = cur_sum
            max_sum_node = cur
        else:
            for child in [cur.left, cur.right]:
                if not child:
                    continue
                parent_lookup[child] = cur
                stack.append((child, cur_sum))
    
    res = []
    node = max_sum_node
    while node:
        res.append(node.val)
        node = parent_lookup.get(node, None)
    return res[::-1]


class LargestSumPathSpec(unittest.TestCase):
    def test_example(self):
        """
            1
          /   \
         3     2
          \   /
           5 4
        """
        left_tree = TreeNode(3, right=TreeNode(5))
        right_tree = TreeNode(2, TreeNode(4))
        root = TreeNode(1, left_tree, right_tree)
        expected_path = [1, 3, 5]
        self.assertEqual(expected_path, largest_sum_path(root))

    def test_negative_nodes(self):
        """
             10
            /  \
          -2    7
         /  \     
        8   -4    
        """
        left_tree = TreeNode(-2, TreeNode(8), TreeNode(-4))
        root = TreeNode(10, left_tree, TreeNode(7))
        expected_path = [10, 7]
        self.assertEqual(expected_path, largest_sum_path(root))

    def test_empty_tree(self):
        self.assertEqual([], largest_sum_path(None))

    def test_heavy_right_tree(self):
        """
          1
         / \
        2   3
       /   / \
      8   4   5
         / \   \
        6   7   9
        """
        n5 = TreeNode(5, right=TreeNode(9))
        n4 = TreeNode(4, TreeNode(6), TreeNode(7))
        n3 = TreeNode(3, n4, n5)
        n2 = TreeNode(2, TreeNode(8))
        n1 = TreeNode(1, n2, n3)
        expected_path = [1, 3, 5, 9]
        self.assertEqual(expected_path, largest_sum_path(n1))

    def test_all_paths_are_negative(self):
        """
              -1
            /     \
          -2      -3
          / \    /  \
        -4  -5 -6   -7
        """
        left_tree = TreeNode(-2, TreeNode(-4), TreeNode(-5))
        right_tree = TreeNode(-3, TreeNode(-6), TreeNode(-7))
        root = TreeNode(-1, left_tree, right_tree)
        expected_path = [-1, -2, -4]
        self.assertEqual(expected_path, largest_sum_path(root))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 3, 2020 LC 121 \[Easy\] Best Time to Buy and Sell Stock
---
> **Question:** You are given an array. Each element represents the price of a stock on that particular day. Calculate and return the maximum profit you can make from buying and selling that stock only once.

**Example:**
```py
Input: [9, 11, 8, 5, 7, 10]
Output: 5
Explanation: Here, the optimal trade is to buy when the price is 5, and sell when it is 10, so the return value should be 5 (profit = 10 - 5 = 5).
```

**Solution:** [https://repl.it/@trsong/Best-Time-to-Buy-and-Sell-Stock](https://repl.it/@trsong/Best-Time-to-Buy-and-Sell-Stock)
```py
import unittest

def max_profit(stock_data):
    if not stock_data:
        return 0
    local_min = stock_data[0]
    res = 0
    for price in stock_data:
        if price < local_min:
            local_min = price
        else:
            res = max(res, price - local_min)
    return res


class MaxProfitSpec(unittest.TestCase):
    def test_blank_data(self):
        self.assertEqual(max_profit([]), 0)
    
    def test_1_day_data(self):
        self.assertEqual(max_profit([9]), 0)
        self.assertEqual(max_profit([-1]), 0)

    def test_monotonically_increase(self):
        self.assertEqual(max_profit([1, 2, 3]), 2)
        self.assertEqual(max_profit([1, 1, 1, 2, 2, 3, 3, 3]), 2)
    
    def test_monotonically_decrease(self):
        self.assertEqual(max_profit([3, 2, 1]), 0)
        self.assertEqual(max_profit([3, 3, 3, 2, 2, 1, 1, 1]), 0)

    def test_raise_suddenly(self):
        self.assertEqual(max_profit([3, 2, 1, 1, 2]), 1)
        self.assertEqual(max_profit([3, 2, 1, 1, 9]), 8)

    def test_drop_sharply(self):
        self.assertEqual(max_profit([1, 3, 0]), 2)
        self.assertEqual(max_profit([1, 3, -1]), 2)

    def test_bear_market(self):
        self.assertEqual(max_profit([10, 11, 5, 7, 1, 2]), 2)
        self.assertEqual(max_profit([10, 11, 1, 4, 2, 7, 5]), 6)

    def test_bull_market(self):
        self.assertEqual(max_profit([1, 5, 3, 7, 2, 14, 10]), 13)
        self.assertEqual(max_profit([5, 1, 11, 10, 12]), 11)


if __name__ == '__main__':
    unittest.main(exit=False)
```
 
### Nov 2, 2020 LC 403 \[Hard\] Frog Jump
---
> **Question:** A frog is crossing a river. The river is divided into x units and at each unit there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.
>
> Given a list of stones' positions (in units) in sorted ascending order, determine if the frog is able to cross the river by landing on the last stone. Initially, the frog is on the first stone and assume the first jump must be 1 unit.
> 
> If the frog's last jump was k units, then its next jump must be either k - 1, k, or k + 1 units. Note that the frog can only jump in the forward direction.

**Example 1:**

```py
[0, 1, 3, 5, 6, 8, 12, 17]

There are a total of 8 stones.
The first stone at the 0th unit, second stone at the 1st unit,
third stone at the 3rd unit, and so on...
The last stone at the 17th unit.

Return true. The frog can jump to the last stone by jumping 
1 unit to the 2nd stone, then 2 units to the 3rd stone, then 
2 units to the 4th stone, then 3 units to the 6th stone, 
4 units to the 7th stone, and 5 units to the 8th stone.
```

**Example 2:**

```py
[0, 1, 2, 3, 4, 8, 9, 11]

Return false. There is no way to jump to the last stone as 
the gap between the 5th and 6th stone is too large.
```

**Solution with DFS:** [https://repl.it/@trsong/Solve-Frog-Jump-Problem](https://repl.it/@trsong/Solve-Frog-Jump-Problem)
```py
import unittest

def can_cross(stones):
    stone_set = set(stones)
    visited = set()
    stack = [(0, 0)]
    goal = stones[-1]

    while stack:
        stone, step = stack.pop()
        if stone == goal:
            return True
        visited.add((stone, step))
        for delta in [-1, 0, 1]:
            next_step = step + delta
            next_stone = stone + next_step
            if next_stone >= stone and next_stone in stone_set and (next_stone, next_step) not in visited:
                stack.append((next_stone, next_step))

    return False
        

class CanCrossSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(can_cross([0, 1, 3, 5, 6, 8, 12, 17])) # step: 1(1), 2(3), 2(5), 3(8), 4(12), 5(17)

    def test_example2(self):
        self.assertFalse(can_cross([0, 1, 2, 3, 4, 8, 9, 11]))

    def test_fast_then_slow(self):
        self.assertTrue(can_cross([0, 1, 3, 6, 10, 13, 15, 16, 16]))

    def test_fast_then_cooldown(self):
        self.assertFalse(can_cross([0, 1, 3, 6, 10, 11]))

    def test_unreachable_last_stone(self):
        self.assertFalse(can_cross([0, 1, 3, 6, 11]))

    def test_reachable_last_stone(self):
        self.assertTrue(can_cross([0, 1, 3, 6, 10]))

    def test_fall_into_water_in_the_middle(self):
        self.assertFalse(can_cross([0, 1, 10, 1000, 1000]))
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 1, 2020 \[Medium\] The Tower of Hanoi
---
> **Question:** The Tower of Hanoi is a puzzle game with three rods and n disks, each a different size.
>
> All the disks start off on the first rod in a stack. They are ordered by size, with the largest disk on the bottom and the smallest one at the top.
>
> The goal of this puzzle is to move all the disks from the first rod to the last rod while following these rules:
>
> - You can only move one disk at a time.
> - A move consists of taking the uppermost disk from one of the stacks and placing it on top of another stack.
> - You cannot place a larger disk on top of a smaller disk.
>
> Write a function that prints out all the steps necessary to complete the Tower of Hanoi. 
> - You should assume that the rods are numbered, with the first rod being 1, the second (auxiliary) rod being 2, and the last (goal) rod being 3.

**Example:** 
```py
with n = 3, we can do this in 7 moves:

Move 1 to 3
Move 1 to 2
Move 3 to 2
Move 1 to 3
Move 2 to 1
Move 2 to 3
Move 1 to 3
```

**My thoughts:** Think about the problem backwards, like what is the most significant states to reach the final state. There are three states coming into my mind: 

- First state, we move all disks except for last one from rod 1 to rod 2. i.e. `[[3], [1, 2], []]`.
- Second state, we move the last disk from rod 1 to rod 3. i.e. `[[], [1, 2], [3]]`
- Third state, we move all disks from rod 2 to rod 3. i.e. `[[], [], [1, 2, 3]]`

There is a clear recrusive relationship between game with size n and size n - 1. So we can perform above stategy recursively for game with size n - 1 which gives the following implementation.

**Solution with Divide-and-Conquer:** [https://repl.it/@trsong/Solve-the-Tower-of-Hanoi-Problem](https://repl.it/@trsong/Solve-the-Tower-of-Hanoi-Problem)
```py
import unittest

def hanoi_moves(n):
    res = []

    def hanoi_moves_recur(n, src, dst):
        if n <= 0:
            return 

        bridge = 3 - src - dst
        # use the unused rod as bridge rod
        # Step1: move n - 1 disks from src to bridge to allow last disk move to dst
        hanoi_moves_recur(n - 1, src, bridge)

        # Step2: move last disk from src to dst
        res.append((src, dst))

        # Step3: move n - 1 disks from bridge to dst
        hanoi_moves_recur(n - 1, bridge, dst)

    hanoi_moves_recur(n, 0, 2)
    return res


class HanoiMoveSpec(unittest.TestCase):
    def assert_hanoi_moves(self, n, moves):
        game = HanoiGame(n)
        # Turn on verbose for debugging
        self.assertTrue(game.can_moves_finish_game(moves, verbose=False))

    def test_three_disks(self):
        moves = hanoi_moves(3)
        self.assert_hanoi_moves(3, moves)

    def test_one_disk(self):
        moves = hanoi_moves(1)
        self.assert_hanoi_moves(1, moves)

    def test_ten_disks(self):
        moves = hanoi_moves(10)
        self.assert_hanoi_moves(10, moves)


class HanoiGame(object):
    def __init__(self, num_disks):
        self.num_disks = num_disks
        self.reset()
        
    def reset(self):
        self.rods = [[disk for disk in xrange(self.num_disks, 0, -1)], [], []]

    def move(self, src, dst):
        disk = self.rods[src].pop()
        self.rods[dst].append(disk)

    def is_feasible_move(self, src, dst):
        return 0 <= src <= 2 and 0 <= dst <= 2 and self.rods[src] and (not self.rods[dst] or self.rods[src][-1] < self.rods[dst][-1])

    def is_game_finished(self):
        return len(self.rods[-1]) == self.num_disks

    def can_moves_finish_game(self, actions, verbose=False):
        self.reset()
        for step, action in enumerate(actions):
            src, dst = action
            if verbose:
                self.display()
                print "Step %d: %d -> %d" % (step, src, dst)
            if not self.is_feasible_move(src, dst):
                return False
            else:
                self.move(src, dst)
        if verbose:
            self.display()
            
        return self.is_game_finished()

    def display(self):
        for plates in self.rods:
            print "- %s" % str(plates)
    

class HanoiGameSpec(unittest.TestCase):
    def test_example_moves(self):
        game = HanoiGame(3)
        moves = [(0, 2), (0, 1), (2, 1), (0, 2), (1, 0), (1, 2), (0, 2)]
        self.assertTrue(game.can_moves_finish_game(moves))

    def test_invalid_moves(self):
        game = HanoiGame(3)
        moves = [(0, 1), (0, 1)]
        self.assertFalse(game.can_moves_finish_game(moves))

    def test_unfinished_moves(self):
        game = HanoiGame(3)
        moves = [(0, 1)]
        self.assertFalse(game.can_moves_finish_game(moves))


if __name__ == '__main__':
    unittest.main(exit=False)
```
