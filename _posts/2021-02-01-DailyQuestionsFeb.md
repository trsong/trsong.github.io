---
layout: post
title:  "Daily Coding Problems 2021 Feb to Apr"
date:   2021-02-01 22:22:32 -0700
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


### Mar 7, 2021 \[Medium\] Distance Between 2 Nodes in BST
---
> **Question:**  Write a function that given a BST, it will return the distance (number of edges) between 2 nodes.

**Example:**
```py
Given the following tree:

         5
        / \
       3   6
      / \   \
     2   4   7
    /         \
   1           8
The distance between 1 and 4 is 3: [1 -> 2 -> 3 -> 4]
The distance between 1 and 8 is 6: [1 -> 2 -> 3 -> 5 -> 6 -> 7 -> 8]
```

**Solution:** [https://repl.it/@trsong/Distance-Between-2-Nodes-in-BST](https://repl.it/@trsong/Distance-Between-2-Nodes-in-BST)
```py
import unittest

def find_distance(tree, v1, v2):    
    path1 = find_path(tree, v1)
    path2 = find_path(tree, v2)

    common_nodes = 0
    for n1, n2 in zip(path1, path2):
        if n1 != n2:
            break
        common_nodes += 1

    return len(path1) + len(path2) - 2 * common_nodes
    

def find_path(tree, v):
    res = []
    while True:
        res.append(tree)
        if tree.val == v:
            break
        elif tree.val < v:
            tree = tree.right
        else:
            tree = tree.left
    return res


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class FindLCASpec(unittest.TestCase):
    def setUp(self):
        """
             4
           /   \
          2     6
         / \   / \
        1   3 5   7
        """
        self.n1 = TreeNode(1)
        self.n3 = TreeNode(3)
        self.n5 = TreeNode(5)
        self.n7 = TreeNode(7)
        self.n2 = TreeNode(2, self.n1, self.n3)
        self.n6 = TreeNode(6, self.n5, self.n7)
        self.root = TreeNode(4, self.n2, self.n6)

    def test_both_nodes_on_leaves(self):
        self.assertEqual(2, find_distance(self.root, 1, 3))

    def test_both_nodes_on_leaves2(self):
        self.assertEqual(2, find_distance(self.root, 5, 7))
    
    def test_both_nodes_on_leaves3(self):
        self.assertEqual(4, find_distance(self.root, 1, 5))

    def test_nodes_on_different_levels(self):
        self.assertEqual(1, find_distance(self.root, 1, 2))
    
    def test_nodes_on_different_levels2(self):
        self.assertEqual(3, find_distance(self.root, 1, 6))
    
    def test_nodes_on_different_levels3(self):
        self.assertEqual(2, find_distance(self.root, 1, 4))

    def test_same_nodes(self):
        self.assertEqual(0, find_distance(self.root, 2, 2))
    
    def test_same_nodes2(self):
        self.assertEqual(0, find_distance(self.root, 5, 5))

    def test_example(self):
        """
                 5
                / \
               3   6
              / \   \
             2   4   7
            /         \
           1           8
        """
        left_tree = TreeNode(3, TreeNode(2, TreeNode(1)), TreeNode(4))
        right_tree = TreeNode(6, right=TreeNode(7, right=TreeNode(8)))
        root = TreeNode(5, left_tree, right_tree)
        self.assertEqual(3, find_distance(root, 1, 4))
        self.assertEqual(6, find_distance(root, 1, 8))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Mar 6, 2021 \[Hard\] Efficiently Manipulate a Very Long String
---
> **Question:** Design a tree-based data structure to efficiently manipulate a very long string that supports the following operations:
>
> - `char char_at(int index)`, return char at index
> - `LongString substring_at(int start_index, int end_index)`, return substring based on start and end index
> - `void delete(int start_index, int end_index)`, deletes the substring 


**My thoughts:** Rope data structure is just a balanced binary tree where leaf stores substring and inner node stores length of all substring of left children recursively. Rope is widely used in text editor program and supports log time insertion, deletion and appending. And is super memory efficent. 

**Solution with Rope:** [https://repl.it/@trsong/Efficiently-Manipulate-a-Very-Long-String](https://repl.it/@trsong/Efficiently-Manipulate-a-Very-Long-String)
```py
import unittest

class LongString(object):
    def __init__(self, s):
        self.rope = RopeNode(s)

    def char_at(self, index):
        return self.rope[index]

    def substring_at(self, start_index, end_index):
        return LongString(self.rope[start_index: end_index + 1])

    def delete(self, start_index, end_index):
        self.rope = self.rope.delete(start_index, end_index)

    
class LongStringSpec(unittest.TestCase):
    def test_empty_string(self):
        self.assertIsNotNone(LongString(''))

    def test_char_at(self):
        s = LongString('01234567')
        self.assertEqual('0', s.char_at(0))
        self.assertEqual('1', s.char_at(1))
        self.assertEqual('3', s.char_at(3))

    def test_chart_at_substring(self):
        s = LongString('012345678')
        self.assertEqual('0', s.substring_at(0, 3).char_at(0))
        self.assertEqual('8', s.substring_at(0, 8).char_at(8))
        self.assertEqual('5', s.substring_at(5, 8).char_at(0))

    def test_delete_string(self):
        s = LongString('012345678')
        s.delete(1, 7)
        self.assertEqual('0', s.char_at(0))
        self.assertEqual('8', s.char_at(1))

        s = LongString('012345678')
        s.delete(0, 3)
        self.assertEqual('4', s.char_at(0))
        self.assertEqual('7', s.char_at(3))

        s = LongString('012345678')
        s.delete(7, 8)
        self.assertEqual('4', s.char_at(4))
        self.assertEqual('6', s.char_at(6))

    def test_char_at_deleted_substring(self):
        s = LongString('012345678')
        s.delete(2, 7)  # gives 018 
        self.assertEqual('1', s.substring_at(1, 2).char_at(0))
        self.assertEqual('8', s.substring_at(1, 2).char_at(1))

    def test_char_at_substring_of_deleted_string(self):
        s = LongString('e012345678eee')
        sub = s.substring_at(1, 8)  # 01234567  
        sub.delete(0, 6)
        self.assertEqual('7', sub.char_at(0))
        

class RopeNode(object):
    def __init__(self, s=None):
        self.weight = len(s) if s else 0
        self.left = None
        self.right = None
        self.data = s

    def delete(self, start_index, end_index):
        if start_index <= 0:
            return self[end_index + 1:]
        elif end_index >= len(self) - 1:
            return self[:start_index]
        else:
            return self[:start_index] + self[end_index + 1:] 

    def __len__(self):
        if self.data is not None:
            return self.weight
        right_len = len(self.right) if self.right else 0
        return self.weight + right_len

    def __add__(self, other):
        # omit tree re-balance 
        node = RopeNode()
        node.weight = len(self)
        node.left = self
        node.right = other
        return node

    def __getitem__(self, key):
        if self.data is not None:
            return self.data[key]
        elif key < self.weight:
            return self.left[key]
        else:
            return self.right[key - self.weight]

    def __getslice__(self, i, j):
        if i >= j:
            return RopeNode('')
        elif self.data:
            return RopeNode(self.data[i:j])
        elif j <= self.weight:
            return self.left[i:j]
        elif i >= self.weight:
            return self.right[i - self.weight:j - self.weight]
        else:
            left_res = self.left[i:self.weight]
            right_res = self.right[0:j - self.weight]
            return left_res + right_res            

    ####################
    # Testing Utilities
    ####################

    def __repr__(self):
        if self.data:
            return self.data
        else:
            return str(self.left or '') + str(self.right or '')

    def print_tree(self):
        stack = [(self, 0)]
        res = ['\n']
        while stack:
            cur, depth = stack.pop()
            res.append('\t' * depth)
            if cur:
                if cur.data:
                    res.append('* data=' + cur.data)
                else:
                    res.append('* weight=' + str(cur.weight))
                stack.append((cur.right, depth + 1))
                stack.append((cur.left, depth + 1))
            else:
                res.append('* None')
            res.append('\n')            
        print ''.join(res)


class RopeNodeSpec(unittest.TestCase):
    def test_string_slice(self):
        """ 
            x 
           / \
          x   2
         / \       
        0   1
        """
        s = RopeNode('0') + RopeNode('1') + RopeNode('2')
        self.assertEqual(1, s.left.weight)
        self.assertEqual(2, s.weight)
        self.assertEqual("0", str(s[0:1]))
        self.assertEqual("01", str(s[0:2]))
        self.assertEqual("012", str(s[0:3]))
        self.assertEqual("12", str(s[1:3]))
        self.assertEqual("2", str(s[2:3]))
        self.assertEqual("1", str(s[1:2]))

    def test_string_slice2(self):
        """ 
              x 
           /     \
          x       x
         / \     / \     
        01 23   4  567
        """
        s = (RopeNode('01') + RopeNode('23')) + (RopeNode('4') + RopeNode('567'))
        self.assertEqual(2, s.left.weight)
        self.assertEqual(4, s.weight)
        self.assertEqual(1, s.right.weight)
        self.assertEqual("012", str(s[0:3]))
        self.assertEqual("3456", str(s[3:7]))
        self.assertEqual("1234567", str(s[1:8]))
        self.assertEqual("7", str(s[7:8]))

    def test_delete(self):
        """ 
              x 
           /     \
          x       x
         / \     / \     
        01 23   4  567
        """
        s = (RopeNode('01') + RopeNode('23')) + (RopeNode('4') + RopeNode('567'))
        self.assertEqual("012", str(s.delete(3, 7)))
        self.assertEqual("4567", str(s.delete(0, 3)))
        self.assertEqual("01237", str(s.delete(4, 6)))

    def test_get_item(self):
        """ 
              x 
           /     \
          x       x
         / \     / \     
        01 23   4  567
        """
        s = (RopeNode('01') + RopeNode('23')) + (RopeNode('4') + RopeNode('567'))
        self.assertEqual("0", s[0])
        self.assertEqual("4", s[4])


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 5, 2021 \[Hard\] Reverse Words Keep Delimiters
---
> **Question:** Given a string and a set of delimiters, reverse the words in the string while maintaining the relative order of the delimiters. For example, given "hello/world:here", return "here/world:hello"
>
> Follow-up: Does your solution work for the following cases: "hello/world:here/", "hello//world:here"

**Solution:** [https://repl.it/@trsong/Reverse-Words-and-Keep-Delimiters](https://repl.it/@trsong/Reverse-Words-and-Keep-Delimiters)
```py
import unittest

def reverse_words_and_keep_delimiters(s, delimiters):
    tokens = filter(len, tokenize(s, delimiters))
    i, j = 0, len(tokens) - 1
    while i < j:
        if tokens[i] in delimiters:
            i += 1
        elif tokens[j] in delimiters:
            j -= 1
        else:
            tokens[i], tokens[j] = tokens[j], tokens[i]
            i += 1
            j -= 1
    return ''.join(tokens)  
     

def tokenize(s, delimiters):
    res = []
    prev_index = -1
    for i, ch in enumerate(s):
        if ch in delimiters:
            res.append(s[prev_index + 1: i])
            res.append(s[i])
            prev_index = i
    res.append(s[prev_index + 1: len(s)])
    return res


class ReverseWordsKeepDelimiterSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(reverse_words_and_keep_delimiters("hello/world:here", ['/', ':']), "here/world:hello")
    
    def test_example2(self):
        self.assertEqual(reverse_words_and_keep_delimiters("hello/world:here/", ['/', ':']), "here/world:hello/")

    def test_example3(self):
        self.assertEqual(reverse_words_and_keep_delimiters("hello//world:here", ['/', ':']), "here//world:hello")

    def test_only_has_delimiters(self):
        self.assertEqual(reverse_words_and_keep_delimiters("--++--+++", ['-', '+']), "--++--+++")

    def test_without_delimiters(self):
        self.assertEqual(reverse_words_and_keep_delimiters("--++--+++", []), "--++--+++")

    def test_without_delimiters2(self):
        self.assertEqual(reverse_words_and_keep_delimiters("--++--+++", ['a', 'b']), "--++--+++")

    def test_first_delimiter_then_word(self):
        self.assertEqual(reverse_words_and_keep_delimiters("///a/b", ['/']), "///b/a")
    
    def test_first_word_then_delimiter(self):
        self.assertEqual(reverse_words_and_keep_delimiters("a///b///", ['/']), "b///a///")


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 4, 2021 \[Hard\] Maximize Sum of the Minimum of K Subarrays
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

Of course, the min value of last array will change, but we can calculate that along the way when we absorb more elements, and we can use `dp[n-p][k] for all p <= n` to calculate the answer. Thus `dp[n][k] = max{dp[n-p][k-1] + min_value of last_subarray} for all p <= n, ie. num[p] is in last subarray`.


**Solution with DP:** [https://repl.it/@trsong/Find-the-Maximize-Sum-of-the-Minimum-of-K-Subarrays](https://repl.it/@trsong/Find-the-Maximize-Sum-of-the-Minimum-of-K-Subarrays)
```py
import unittest

def max_aggregate_subarray_min(nums, k):
    n = len(nums)

    # Let dp[n][k] represents max aggregate subarray of min for nums[:n] 
    # and target number of partition is k
    # dp[n][k] = max(dp[n - p][k - 1] + min(nums[n - p: n])) for p <= n
    dp = [[float('-inf') for _ in xrange(k + 1)] for _ in xrange(n + 1)]
    dp[0][0] = 0
    
    for j in xrange(1, k + 1):
        for i in xrange(j, n + 1):
            last_window_min = nums[i - 1]
            for p in xrange(1, i + 1):
                last_window_min = min(last_window_min, nums[i - p])
                dp[i][j] = max(dp[i][j], dp[i - p][j - 1] + last_window_min)
    
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
    unittest.main(exit=False, verbosity=2)
```

### Mar 3, 2021 \[Hard\] Maximize the Minimum of Subarray Sum
---
> **Question:** Given an array of numbers `N` and an integer `k`, your task is to split `N` into `k` partitions such that the maximum sum of any partition is minimized. Return this sum.
>
> For example, given `N = [5, 1, 2, 7, 3, 4]` and `k = 3`, you should return `8`, since the optimal partition is `[5, 1, 2], [7], [3, 4]`.


**My thoughts:** The method to solve this problem is through guessing the result. We have the following observations: 

- The lower bound of guessing number is `max(nums)`, as max element has to present in some subarray. 
- The upper bound of guessing number is `sum(nums)` as sum of subarray cannot exceed sum of entire array. 

We can use the guessing number to cut the array greedily so that each parts cannot exceed the guessing number:
- If the guessing number is lower than expected, then we over-cut the array.  (cut more than k parts)
- If the guessing number is higher than expected, then we under-cut the array.  (cut less than k parts)

We can use binary search to get result such that it just-cut the array: under-cut and just-cut goes left and over-cut goes right. 

But how can we make sure that result we get from binary search is indeed sum of subarray?

The reason is simply, if it is just-cut, it won't stop until it's almost over-cut that gives smallest just-cut. Maximum of the Minimum Sum among all subarray sum is actually the smallest just-cut. And it will stop at that number. 


**Solution with Binary Search:** [https://repl.it/@trsong/Maximize-the-Minimum-of-Subarray-Sum](https://repl.it/@trsong/Maximize-the-Minimum-of-Subarray-Sum)
```py
import unittest


def max_of_min_sum_subarray(nums, k):
    lo = max(nums)
    hi = sum(nums)

    while lo < hi:
        mid = lo + (hi - lo) // 2
        if within_partition_constraint(nums, k, mid):
            hi = mid
        else:
            lo = mid + 1
    
    return lo


def within_partition_constraint(nums, k, subarray_limit):
    accu = 0
    for num in nums:
        if accu + num > subarray_limit:
            accu = num
            k -= 1
        else:
            accu += num
    return k >= 1



class MaxOfMinSumSubarraySpec(unittest.TestCase):
    def test_example(self):
        k, nums = 3, [5, 1, 2, 7, 3, 4]
        expected = 8  # [5, 1, 2], [2, 7], [3, 4]
        self.assertEqual(expected, max_of_min_sum_subarray(nums, k))

    def test_ascending_array(self):
        k, nums = 3, [1, 2, 3, 4]
        expected = 4  # [1, 2], [3], [4]
        self.assertEqual(expected, max_of_min_sum_subarray(nums, k))

    def test_k_is_one(self):
        k, nums = 1, [1, 1, 1, 1, 4]
        expected = 8  # [1, 1, 1, 1, 4]
        self.assertEqual(expected, max_of_min_sum_subarray(nums, k))

    def test_return_larger_half(self):
        k, nums = 2,  [1, 2, 3, 4, 5, 10, 11, 3, 6, 16]
        expected = 36  # [1, 2, 3, 4, 5, 10, 11], [3, 6, 16]
        self.assertEqual(expected, max_of_min_sum_subarray(nums, k))

    def test_evenly_distributed(self):
        k, nums = 4, [1, 1, 1, 1]
        expected = 1
        self.assertEqual(expected, max_of_min_sum_subarray(nums, k))

    def test_evenly_distributed2(self):
        k, nums = 3, [1, 1, 1, 1, 1, 1, 1, 1, 1]
        expected = 3
        self.assertEqual(expected, max_of_min_sum_subarray(nums, k))

    def test_outlier_element(self):
        k, nums = 3, [1, 1, 1, 100, 1]
        expected = 100  # [1, 1, 1], [100], [1]
        self.assertEqual(expected, max_of_min_sum_subarray(nums, k))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 2, 2021 \[Medium\] Bitwise AND of a Range
---
> **Question:** Write a function that returns the bitwise AND of all integers between M and N, inclusive.

**Solution:** [https://repl.it/@trsong/Calculate-Bitwise-AND-of-a-Range](https://repl.it/@trsong/Calculate-Bitwise-AND-of-a-Range)
```py
import unittest

def bitwise_and_of_range(m, n):
    """"
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
        # remove last set bit
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
    unittest.main(exit=False, verbosity=2)
```

### Mar 1, 2021 LC 336 \[Hard\] Palindrome Pairs
---
> **Question:** Given a list of words, find all pairs of unique indices such that the concatenation of the two words is a palindrome.
>
> For example, given the list `["code", "edoc", "da", "d"]`, return `[(0, 1), (1, 0), (2, 3)]`.


**My thoughts:** any word in the list can be partition into `prefix` and `suffix`. If there exists another word such that its reverse equals either prefix or suffix, then we can combine them and craft a new palindrome: 
1. `reverse_suffix + prefix + suffix` where prefix is a palindrome or 
2. `prefix + suffix + reverse_prefix` where suffix is a palindrome

**Solution:** [https://repl.it/@trsong/Find-All-Palindrome-Pairs](https://repl.it/@trsong/Find-All-Palindrome-Pairs)
```py

import unittest
from collections import defaultdict

def find_all_palindrome_pairs(words):
    reverse_words = defaultdict(list)
    for i, word in enumerate(words):
        reverse_words[word[::-1]].append(i)
    
    res = []

    if "" in reverse_words:
        palindrome_indices = filter(lambda j: is_palindrome(words[j]), xrange(len(words)))
        res.extend((i, j) for i in reverse_words[""] for j in palindrome_indices if i != j)

    for i, word in enumerate(words):
        for pos in xrange(len(word)):
            prefix = word[:pos]
            suffix = word[pos:]
            if prefix in reverse_words and is_palindrome(suffix):
                res.extend((i, j) for j in reverse_words[prefix] if i != j)

            if suffix in reverse_words and is_palindrome(prefix):
                res.extend((j, i) for j in reverse_words[suffix] if i != j)

    return res


def is_palindrome(s):
    i = 0
    j = len(s) - 1
    while i < j:
        if s[i] != s[j]:
            return False
        i += 1
        j -= 1
    return True


class FindAllPalindromePairSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertSetEqual(set(expected), set(result))
        self.assertEqual(len(expected), len(result))

    def test_example(self):
        words = ["code", "edoc", "da", "d"]
        expected = [(0, 1), (1, 0), (2, 3)]
        self.assert_result(expected, find_all_palindrome_pairs(words))

    def test_example2(self):
        words = ["bat", "tab", "cat"]
        expected = [(0, 1), (1, 0)]
        self.assert_result(expected, find_all_palindrome_pairs(words))

    def test_example3(self):
        words = ["abcd", "dcba", "lls", "s", "sssll"]
        expected = [(0, 1), (1, 0), (3, 2), (2, 4)]
        self.assert_result(expected, find_all_palindrome_pairs(words))
    
    def test_single_word_string(self):
        words = ["a", "ab", "b", ""]
        expected = [(1, 0), (2, 1), (0, 3), (3, 0), (2, 3), (3, 2)]
        self.assert_result(expected, find_all_palindrome_pairs(words))

    def test_empty_lists(self):
        self.assert_result([], find_all_palindrome_pairs([]))
    
    def test_contains_empty_word(self):
        words = [""]
        expected = []
        self.assert_result(expected, find_all_palindrome_pairs(words))

    def test_contains_empty_word2(self):
        words = ["", ""]
        expected = [(0, 1), (1, 0)]
        self.assert_result(expected, find_all_palindrome_pairs(words))
    
    def test_contains_empty_word3(self):
        words = ["", "", "a"]
        expected = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        self.assert_result(expected, find_all_palindrome_pairs(words))
    
    def test_contains_empty_word4(self):
        words = ["", "a"]
        expected = [(0, 1), (1, 0)]
        self.assert_result(expected, find_all_palindrome_pairs(words))


    def test_contains_duplicate_word(self):
        words = ["a", "a", "aa"]
        expected = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]
        self.assert_result(expected, find_all_palindrome_pairs(words))

    def test_no_pairs(self):
        words = ["abc", "gaba", "abcg"]
        expected = []
        self.assert_result(expected, find_all_palindrome_pairs(words))

    def test_avoid_hash_collision(self):
        words = ["a", "jfdjfhgidffedfecbfh"]
        expected = []
        self.assert_result(expected, find_all_palindrome_pairs(words))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Feb 28, 2021 \[Medium\] 24-Hour Hit Counter
---
> **Question:** You are given an array of length 24, where each element represents the number of new subscribers during the corresponding hour. Implement a data structure that efficiently supports the following:
>
> - `update(hour: int, value: int)`: Increment the element at index hour by value.
> - `query(start: int, end: int)`: Retrieve the number of subscribers that have signed up between start and end (inclusive). You can assume that all values get cleared at the end of the day, and that you will not be asked for start and end values that wrap around midnight.

**My thoughts:** Binary-indexed Tree a.k.a BIT or Fenwick Tree is used for dynamic range query. Basically, it allows efficiently query for sum of a continous interval of values.

BIT Usage:
```py
bit = BIT(4)
bit.query(3) # => 0
bit.update(index=0, delta=1) # => [1, 0, 0, 0]
bit.update(index=0, delta=1) # => [2, 0, 0, 0]
bit.update(index=2, delta=3) # => [2, 0, 3, 0]
bit.query(2) # => 2 + 0 + 3 = 5
bit.update(index=1, delta=-1) # => [2, -1, 3, 0]
bit.query(1) # => 2 + -1 = 1
```


**Solution with BIT:** [https://repl.it/@trsong/24-Hour-Hit-Counter](https://repl.it/@trsong/24-Hour-Hit-Counter)
```py
import unittest

class HitCounter(object):
    HOUR_OF_DAY = 24

    def __init__(self, hours=HOUR_OF_DAY):
        self.tree = [0] * (hours + 1)

    def update(self, hour, value):
        old_val = self.query(hour, hour)
        delta = value - old_val

        index = hour + 1
        while index < len(self.tree):
            self.tree[index] += delta
            index += index & -index

    def query(self, start_time, end_time):
        return self.query_from_begin(end_time) - self.query_from_begin(start_time - 1)
    
    def query_from_begin(self, end_time):
        index = end_time + 1
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res


class HitCounterSpec(unittest.TestCase):
    def test_query_without_update(self):
        hc = HitCounter()
        self.assertEqual(0, hc.query(0, 23))
        self.assertEqual(0, hc.query(3, 5))
    
    def test_update_should_affect_query_value(self):
        hc = HitCounter()
        hc.update(5, 10)
        hc.update(10, 15)
        hc.update(12, 20)
        self.assertEqual(0, hc.query(0, 4))
        self.assertEqual(10, hc.query(0, 5))
        self.assertEqual(10, hc.query(0, 9))
        self.assertEqual(25, hc.query(0, 10))
        self.assertEqual(45, hc.query(0, 13))
        hc.update(3, 2)
        self.assertEqual(2, hc.query(0, 4))
        self.assertEqual(12, hc.query(0, 5))
        self.assertEqual(12, hc.query(0, 9))
        self.assertEqual(27, hc.query(0, 10))
        self.assertEqual(47, hc.query(0, 13))

    def test_number_of_subscribers_can_decrease(self):
        hc = HitCounter()
        hc.update(10, 5)
        hc.update(20, 10)
        self.assertEqual(10, hc.query(15, 23))
        hc.update(12, -3)
        self.assertEqual(10, hc.query(15, 23))
        hc.update(17, -7)
        self.assertEqual(3, hc.query(15, 23))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 27, 2021 \[Easy\] Flatten Nested List Iterator
---
> **Question:** Implement a 2D iterator class. It will be initialized with an array of arrays, and should implement the following methods:
>
> - `next()`: returns the next element in the array of arrays. If there are no more elements, raise an exception.
> - `has_next()`: returns whether or not the iterator still has elements left.
>
> For example, given the input `[[1, 2], [3], [], [4, 5, 6]]`, calling `next()` repeatedly should output `1, 2, 3, 4, 5, 6`.
>
> Do not use flatten or otherwise clone the arrays. Some of the arrays can be empty.

**Solution:** [https://repl.it/@trsong/Flatten-Nested-List-Iterator](https://repl.it/@trsong/Flatten-Nested-List-Iterator)
```py
import unittest

class NestedIterator(object):
    def __init__(self, nested_list):
        self.row = 0
        self.col = 0
        self.nested_list = [[None]] + nested_list
        self.next()
        
    def next(self):
        res = self.nested_list[self.row][self.col]

        self.col += 1
        while self.row < len(self.nested_list) and self.col >= len(self.nested_list[self.row]):
            self.col = 0
            self.row += 1

        return res       

    def has_next(self):
        return self.row < len(self.nested_list)
        

class NestedIteratorSpec(unittest.TestCase):
    def assert_result(self, nested_list):
        expected = []
        for lst in nested_list:
            if not lst:
                continue
            expected.extend(lst)

        res = []
        it = NestedIterator(nested_list)
        while it.has_next():
            res.append(it.next())
        
        self.assertEqual(expected, res)       
        
    def test_example(self):
        self.assert_result([[1, 2], [3], [], [4, 5, 6]])

    def test_empty_list(self):
        it = NestedIterator([])
        self.assertFalse(it.has_next())

    def test_empty_list2(self):
        it = NestedIterator([[], [], []])
        self.assertFalse(it.has_next())

    def test_non_empty_list(self):
        self.assert_result([[1], [2], [3], [4]])

    def test_non_empty_list2(self):
        self.assert_result([[1, 1, 1], [4], [1, 2, 3], [5]])

    def test_has_empty_list(self):
        self.assert_result([[], [1, 2, 3], []])


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 26, 2021 LC 227 \[Medium\] Basic Calculator II
---
> **Question:** Implement a basic calculator to evaluate a simple expression string.
>
> The expression string contains only non-negative integers, +, -, *, / operators and empty spaces. The integer division should truncate toward zero.

**Example 1:**
```py
Input: "3+2*2"
Output: 7
```

**Example 2:**
```py
Input: " 3/2 "
Output: 1
```

**Example 3:**
```py
Input: " 3+5 / 2 "
Output: 5
```


**My thoughts:** A complicated _expression_ can be broken into multiple normal _terms_. `Expr = term1 + term2 - term3 ...`. Between each consecutive term we only allow `+` and `-`. Whereas within each term we only allow `*` and `/`. So we will have the following definition of an expression. e.g. `1 + 2 - 1*2*1 - 3/4*4 + 5*6 - 7*8 + 9/10 = (1) + (2) - (1*2*1) - (3/4*4) + (5*6) - (7*8) + (9/10)` 

_Expression_ is one of the following:
- Empty or 0
- Term - Expression
- Term + Expression

_Term_ is one of the following:
- 1
- A number * Term
- A number / Term


Thus, we can comupte each term value and sum them together.


**Solution:** [https://repl.it/@trsong/Implement-Basic-Calculator-II](https://repl.it/@trsong/Implement-Basic-Calculator-II)
```py
import unittest

OP_SET = {'+', '-', '*', '/', 'EOF'}

def calculate(s):
    tokens = tokenize(s)
    expr_value = 0
    term_value = 0
    num = 0
    prev_op = '+'

    for token in tokens:
        if not token or token.isspace():
            continue
            
        if token not in OP_SET:
            num = int(token)
            continue
        
        if prev_op == '+':
            expr_value += term_value
            term_value = num
        elif prev_op == '-':
            expr_value += term_value
            term_value = -num
        elif prev_op == '*':
            term_value *= num
        elif prev_op == '/':
            sign = 1 if term_value > 0 else -1
            term_value = abs(term_value) / num * sign
            
        num = 0
        prev_op = token

    expr_value += term_value
    return expr_value
    

def tokenize(s):
    res = []
    prev_pos = -1
    for pos, ch in enumerate(s):
        if ch in OP_SET:
            res.append(s[prev_pos + 1: pos])
            res.append(ch)
            prev_pos = pos
    res.append(s[prev_pos + 1: ])
    res.append('EOF')
    return res


class CalculateSpec(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(0, calculate(""))

    def test_example1(self):
        self.assertEqual(7, calculate("3+2*2"))

    def test_example2(self):
        self.assertEqual(1, calculate(" 3/2 "))

    def test_example3(self):
        self.assertEqual(5, calculate(" 3+5 / 2 "))

    def test_negative1(self):
        self.assertEqual(-1, calculate("-1"))

    def test_negative2(self):
        self.assertEqual(0, calculate(" -1/2 "))

    def test_negative3(self):
        self.assertEqual(-1, calculate(" -7 / 4 "))

    def test_minus(self):
        self.assertEqual(-5, calculate("-2-3"))
    
    def test_positive1(self):
        self.assertEqual(10, calculate("100/ 10"))
    
    def test_positive2(self):
        self.assertEqual(4, calculate("9 /2"))

    def test_complicated_operations(self):
        self.assertEqual(-24, calculate("1*2-3/4+5*6-7*8+9/10"))

    def test_complicated_operations2(self):
        self.assertEqual(10000, calculate("10000-1000/10+100*1"))

    def test_complicated_operations3(self):
        self.assertEqual(13, calculate("14-3/2"))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 25, 2021 \[Medium\] Evaluate Expression in Reverse Polish Notation
---
> **Question:** Given an arithmetic expression in **Reverse Polish Notation**, write a program to evaluate it.
>
> The expression is given as a list of numbers and operands. 

**Example 1:** 
```py
[5, 3, '+'] should return 5 + 3 = 8.
```

**Example 2:**
```py
 [15, 7, 1, 1, '+', '-', '/', 3, '*', 2, 1, 1, '+', '+', '-'] should return 5, 
 since it is equivalent to ((15 / (7 - (1 + 1))) * 3) - (2 + (1 + 1)) = 5.
 ```


**Solution with Stack:** [https://repl.it/@trsong/Evaluate-Expression-Represented-in-Reverse-Polish-Notation](https://repl.it/@trsong/Evaluate-Expression-Represented-in-Reverse-Polish-Notation)
```py
import unittest

class RPNExprEvaluator(object):
    @staticmethod
    def run(tokens):
        stack = []
        for token in tokens:
            if type(token) is int:
                stack.append(token)
            else:
                num2 = stack.pop()
                num1 = stack.pop()
                res = RPNExprEvaluator.apply(token, num1, num2)
                stack.append(res)
        return stack[-1] if stack else 0

    @staticmethod
    def apply(operation, num1, num2):
        if operation == '+':
            return num1 + num2
        elif operation == '-':
            return num1 - num2
        elif operation == '*':
            return num1 * num2
        elif operation == '/':
            return num1 / num2
        else:
            raise NotImplementedError
        

class RPNExprEvaluatorSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(8, RPNExprEvaluator.run([5, 3, '+'])) # 5 + 3 = 8

    def test_example2(self):
        tokens = [15, 7, 1, 1, '+', '-', '/', 3, '*', 2, 1, 1, '+', '+', '-']
        self.assertEqual(5, RPNExprEvaluator.run(tokens))

    def test_empty_tokens(self):
        self.assertEqual(0, RPNExprEvaluator.run([]))

    def test_expression_contains_just_number(self):
        self.assertEqual(42, RPNExprEvaluator.run([42]))
    
    def test_balanced_expression_tree(self):
        tokens = [7, 2, '-', 4, 1, '+', '*'] 
        self.assertEqual(25, RPNExprEvaluator.run(tokens))  # (7 - 2) * (4 + 1) = 25
    
    def test_left_heavy_expression_tree(self):
        tokens = [6, 4, '-', 2, '/']  
        self.assertEqual(1, RPNExprEvaluator.run(tokens)) # (6 - 4) / 2 = 1

    def test_right_heavy_expression_tree(self):
        tokens = [2, 8, 2, '/', '*']
        self.assertEqual(8, RPNExprEvaluator.run(tokens)) # 2 * (8 / 2) = 8


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 24, 2021 \[Easy\] Minimum Distance between Two Words
---
> **Question:** Find an efficient algorithm to find the smallest distance (measured in number of words) between any two given words in a string.
>
> For example, given words `"hello"`, and `"world"` and a text content of `"dog cat hello cat dog dog hello cat world"`, return `1` because there's only one word `"cat"` in between the two words.

**Solution with Two Pointers:** [https://repl.it/@trsong/Minimum-Distance-between-Two-Words](https://repl.it/@trsong/Minimum-Distance-between-Two-Words)
```py
import unittest
import sys

def word_distance(s, word1, word2):
    i = sys.maxint
    j = -sys.maxint
    res = sys.maxint

    for index, word in enumerate(s.split()):
        if word == word1:
            i = index
        
        if word == word2:
            j = index

        res = min(res, abs(j - i))

    return res if res < sys.maxint else -1


class WordDistanceSpec(unittest.TestCase):
    def test_example(self):
        s = 'dog cat hello cat dog dog hello cat world'
        word1 = 'hello'
        word2 = 'world'
        self.assertEqual(2, word_distance(s, word1, word2))

    def test_word_not_exists_in_sentence(self):
        self.assertEqual(-1, word_distance("", "a", "b"))

    def test_word_not_exists_in_sentence2(self):
        self.assertEqual(-1, word_distance("b", "a", "a"))

    def test_word_not_exists_in_sentence3(self):
        self.assertEqual(-1, word_distance("ab", "a", "b"))
    
    def test_only_one_word_exists(self):
        s = 'a b c d'
        word1 = 'a'
        word2 = 'e'
        self.assertEqual(-1, word_distance(s, word1, word2))
        self.assertEqual(-1, word_distance(s, word2, word1))
        
    def test_search_for_same_word_in_sentence(self):
        s = 'cat dog cat cat dog dog dog cat'
        word = 'cat'
        self.assertEqual(0, word_distance(s, word, word))

    def test_second_word_comes_first(self):
        s = 'air water water earth water water water'
        word1 = "earth"
        word2 = "air"
        self.assertEqual(3, word_distance(s, word1, word2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 23, 2021 \[Hard\] K-Palindrome
---
> **Question:** Given a string which we can delete at most k, return whether you can make a palindrome.
>
> For example, given `'waterrfetawx'` and a k of 2, you could delete f and x to get `'waterretaw'`.

**My thoughts:** We can either solve this problem by modifying the edit distance function or taking advantage of longest common subsequnce. Here we choose the later method.

We can first calculate the minimum deletion needed to make a palindrome and the way we do it is to compare the original string vs the reversed string in order to calculate the LCS - longest common subsequence. Thus the minimum deletion equals length of original string minus LCS.

To calculate LCS, we use DP and will encounter the following situations:
1. If the last digit of each string matches each other, i.e. `lcs(seq1 + s, seq2 + s)` then `result = 1 + lcs(seq1, seq2)`.
2. If the last digit not matches, i.e. `lcs(seq1 + s, seq2 + p)`, then res is either ignore s or ignore q. Just like insert a whitespace or remove a letter from edit distance, which gives `max(lcs(seq1, seq2 + p), lcs(seq1 + s, seq2))`


**Solution with DP:** [https://repl.it/@trsong/Find-K-Palindrome](https://repl.it/@trsong/Find-K-Palindrome)
```py
import unittest

def is_k_palindrome(s, k):
    lcs = longest_common_subsequence(s, s[::-1])
    min_letter_to_remove = len(s) - lcs
    return min_letter_to_remove <= k


def longest_common_subsequence(seq1, seq2):
    n, m = len(seq1), len(seq2)

    # Let dp[n][m] represents lcs of seq1[:n] and seq2[:m]
    # dp[n][m] = 1 + dp[n-1][m-1]             if seq1[n-1] == seq2[m-1]
    #          = max(dp[n-1][m], dp[n][m-1])  otherwise
    dp = [[0 for _ in xrange(m + 1)] for _ in xrange(n + 1)]
    for i in xrange(1, n + 1):
        for j in xrange(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[n][m]


class IsKPalindromeSpec(unittest.TestCase):
    def test_example(self):
        # waterrfetawx => waterretaw
        self.assertTrue(is_k_palindrome('waterrfetawx', k=2))

    def test_empty_string(self):
        self.assertTrue(is_k_palindrome('', k=2))
    
    def test_palindrome_string(self):
        self.assertTrue(is_k_palindrome('abcddcba', k=1))

    def test_removing_exact_k_characters(self):
        # abcdecba => abcecba
        self.assertTrue(is_k_palindrome('abcdecba', k=1))

    def test_removing_exact_k_characters2(self):
        # abcdeca => acdca
        self.assertTrue(is_k_palindrome('abcdeca', k=2))

    def test_removing_exact_k_characters3(self):
        # acdcb => cdc
        self.assertTrue(is_k_palindrome('acdcb', k=2))

    def test_not_k_palindrome(self):
        self.assertFalse(is_k_palindrome('acdcb', k=1))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 22, 2021 \[Medium\] Lazy Binary Tree Generation
---
> **Question:** Generate a finite, but an arbitrarily large binary tree quickly in `O(1)`.
>
> That is, `generate()` should return a tree whose size is unbounded but finite.

**Solution:** [https://repl.it/@trsong/Lazy-Binary-Tree-Generation](https://repl.it/@trsong/Lazy-Binary-Tree-Generation)
```py
import random

def generate():
    return TreeNode(0)


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self._left = left
        self._right = right
        self._is_left_init = False
        self._is_right_init = False

    @property
    def left(self):
        if not self._is_left_init:
            if random.randint(0, 1):
                self._left = TreeNode(0)
            self._is_left_init = True
        return self._left

    @property
    def right(self):
        if not self._is_right_init:
            if random.randint(0, 1):
                self._right = TreeNode(0)
            self._is_right_init = True
        return self._right

    def __repr__(self):
        stack = [(self, 0)]
        res = ['\n']
        while stack:
            cur, depth = stack.pop()
            res.append('\t' * depth)
            if cur:
                res.append('* ' + str(cur.val))
                stack.append((cur.right, depth + 1))
                stack.append((cur.left, depth + 1))
            else:
                res.append('* None')
            res.append('\n')            
        return ''.join(res)


if __name__ == '__main__':
    print generate()
```

### Feb 21, 2021 LC 131 \[Medium\] Palindrome Partitioning
---
> **Question:** Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.
>
> A palindrome string is a string that reads the same backward as forward.

**Example 1:**
```py
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]
```

**Example 2:**
```py
Input: s = "a"
Output: [["a"]]
```

**Solution with Backtracking:** [https://repl.it/@trsong/Palindrome-Partitioning](https://repl.it/@trsong/Palindrome-Partitioning)
```py
import unittest

def palindrome_partition(s):
    n = len(s)
    cache = [[None for _ in xrange(n)] for _ in xrange(n)]
    res = []
    backtrack(s, res, [], 0, cache)
    return res


def backtrack(s, res, accu, start, cache):
    n = len(s)
    if start >= n:
        res.append(accu[:])
    
    for end in xrange(start, n):
        if is_palindrome(s, start, end, cache):
            accu.append(s[start: end + 1])
            backtrack(s, res, accu, end + 1, cache)
            accu.pop()


def is_palindrome(s, start, end, cache):
    if end - start < 1:
        return True
    
    if cache[start][end] is None:
        cache[start][end] = s[start] == s[end] and is_palindrome(s, start + 1, end - 1, cache)
    
    return cache[start][end]


class PalindromePartitionSpec(unittest.TestCase):
    def assert_result(self, expected, res):
        expected.sort()
        res.sort()
        self.assertEqual(expected, res)
    
    def test_example(self):
        s = "aab"
        expected = [["a","a","b"],["aa","b"]]
        self.assert_result(expected, palindrome_partition(s))
    
    def test_example2(self):
        s = "a"
        expected = [["a"]]
        self.assert_result(expected, palindrome_partition(s))
    
    def test_multiple_results(self):
        s = "12321"
        expected = [
            ['1', '2', '3', '2', '1'], 
            ['1', '232', '1'], 
            ['12321']]
        self.assert_result(expected, palindrome_partition(s))
    
    def test_multiple_results2(self):
        s = "112321"
        expected = [
            ['1', '1', '2', '3', '2', '1'], 
            ['1', '1', '232', '1'], 
            ['1', '12321'], 
            ['11', '2', '3', '2', '1'], 
            ['11', '232', '1']]
        self.assert_result(expected, palindrome_partition(s))
    
    def test_multiple_results3(self):
        s = "aaa"
        expected = [["a", "aa"], ["aa", "a"], ["aaa"], ['a', 'a', 'a']]
        self.assert_result(expected, palindrome_partition(s))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 20, 2021 \[Hard\] Minimum Palindrome Substring
---
> **Question:** Given a string, split it into as few strings as possible such that each string is a palindrome.
>
> For example, given the input string `"racecarannakayak"`, return `["racecar", "anna", "kayak"]`.
>
> Given the input string `"abc"`, return `["a", "b", "c"]`.


**Solution with DP:** [https://repl.it/@trsong/Minimum-Palindrome-Substring](https://repl.it/@trsong/Minimum-Palindrome-Substring)
```py
import unittest

def min_palindrome_substring(s):
    n = len(s)
    # Let cache[start][end] represents s[start: end + 1] is palindrome or not
    cache = [[None for _ in xrange(n)] for _ in xrange(n)]

    # Let dp[size] represents the min palindrome for string s[:size]
    # dp[size] = dp[start] + [s[start: size]] if s[start: size] is palindrome and start = argmin(len(dp[x])) 
    dp = [[] for _ in xrange(n + 1)]
    for size in xrange(1, n + 1):
        start = size - 1
        min_sub_size = float('inf')
        for x in xrange(size):
            if not is_palindrome(s, x, size - 1, cache):
                continue
            if len(dp[x]) < min_sub_size:
                min_sub_size = len(dp[x])
                start = x
        dp[size] = dp[start] + [s[start: size]]
    return dp[-1]


def is_palindrome(s, start, end, cache):
    if end - start < 1:
        return True
    
    if cache[start][end] is None:
        cache[start][end] = s[start] == s[end] and is_palindrome(s, start + 1, end - 1, cache)
    
    return cache[start][end]
    

class MinPalindromeSubstringSpec(unittest.TestCase):
    def test_example(self):
        s = 'racecarannakayak'
        expected = ['racecar', 'anna', 'kayak']
        self.assertEqual(expected, min_palindrome_substring(s))

    def test_example2(self):
        s = 'abc'
        expected = ['a', 'b', 'c']
        self.assertEqual(expected, min_palindrome_substring(s))

    def test_empty_string(self):
        self.assertEqual([], min_palindrome_substring(''))

    def test_one_char_string(self):
        self.assertEqual(['a'], min_palindrome_substring('a'))

    def test_already_palindrome(self):
        s = 'abbacadabraarbadacabba'
        expected = ['abbacadabraarbadacabba']
        self.assertEqual(expected, min_palindrome_substring(s))

    def test_long_and_short_palindrome_substrings(self):
        s1 = 'aba'
        s2 = 'abbacadabraarbadacabba'
        s3 = 'c'
        expected = [s1, s2, s3]
        self.assertEqual(expected, min_palindrome_substring(s1 + s2 + s3))

    def test_should_return_optimal_solution(self):
        s = 'xabaay'
        expected = ['x', 'aba', 'a', 'y']
        self.assertEqual(expected, min_palindrome_substring(s))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 19, 2021 \[Medium\] Smallest Number of Perfect Squares
---
> **Question:** Write a program that determines the smallest number of perfect squares that sum up to N.
>
> Here are a few examples:
```py
Given N = 4, return 1 (4)
Given N = 17, return 2 (16 + 1)
Given N = 18, return 2 (9 + 9)
```

**Solution with DP:** [https://repl.it/@trsong/Calculate-the-Smallest-Number-of-Perfect-Squares](https://repl.it/@trsong/Calculate-the-Smallest-Number-of-Perfect-Squares)
```py
import unittest
import math

def min_perfect_squares(target):
    if target == 0:
        return 1
    elif target < 0:
        return -1

    # Let dp[num] represents min num of squares sum to num
    # dp[num] = dp[num - i * i] + 1 where i * i <= num
    dp = [float("inf")] * (target + 1)
    dp[0] = 0
    for num in xrange(1, target + 1):
        for i in xrange(1, int(math.sqrt(num) + 1)):
            dp[num] = min(dp[num], 1 + dp[num - i * i])
    
    return dp[target]


class MinPerfectSquareSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(1, min_perfect_squares(4))  # 4 = 4

    def test_example2(self):
        self.assertEqual(2, min_perfect_squares(17))  # 17 = 16 + 1

    def test_example3(self):
        self.assertEqual(2, min_perfect_squares(18))  # 18 = 9 + 9

    def test_greedy_not_work(self):
        self.assertEqual(2, min_perfect_squares(32))  # 32 = 16 + 16
    
    def test_zero(self):
        self.assertEqual(1, min_perfect_squares(0))  # 0 = 0

    def test_negative_number(self):
        self.assertEqual(-1, min_perfect_squares(-3))  # negatives cannot be a sum of perfect squares
    
    def test_perfect_numbers(self):
        self.assertEqual(1, min_perfect_squares(169))  # 169 = 169

    def test_odd_number(self):
        self.assertEqual(3, min_perfect_squares(3))  # 3 = 1 + 1 + 1

    def test_even_number(self):
        self.assertEqual(3, min_perfect_squares(6))  # 6 = 4 + 1 + 1


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Feb 18, 2021 LC 1155 \[Medium\] Number of Dice Rolls With Target Sum
---
> **Question:** You have `d` dice, and each die has `f` faces numbered `1, 2, ..., f`.
>
> Return the number of possible ways (out of `f^d` total ways) modulo `10^9 + 7` to roll the dice so the sum of the face up numbers equals target.


**Example 1:**
```py
Input: d = 1, f = 6, target = 3
Output: 1
Explanation: 
You throw one die with 6 faces.  There is only one way to get a sum of 3.
```

**Example 2:**
```py
Input: d = 2, f = 6, target = 7
Output: 6
Explanation: 
You throw two dice, each with 6 faces.  There are 6 ways to get a sum of 7:
1+6, 2+5, 3+4, 4+3, 5+2, 6+1.
```

**Example 3:**
```py
Input: d = 2, f = 5, target = 10
Output: 1
Explanation: 
You throw two dice, each with 5 faces.  There is only one way to get a sum of 10: 5+5.
```

**Example 4:**
```py
Input: d = 1, f = 2, target = 3
Output: 0
Explanation: 
You throw one die with 2 faces.  There is no way to get a sum of 3.
```

**Example 5:**
```py
Input: d = 30, f = 30, target = 500
Output: 222616187
Explanation: 
The answer must be returned modulo 10^9 + 7.
```

**Solution with DP:** [https://repl.it/@trsong/Number-of-Dice-Rolls-With-Target-Sum](https://repl.it/@trsong/Number-of-Dice-Rolls-With-Target-Sum)
```py
import unittest

MODULE_NUM = 1000000007

def throw_dice(d, f, target):
    return throw_dice_with_cache(d, f, target, {})


def throw_dice_with_cache(d, f, target, cache):
    if target == d == 0:
        return 1
    elif target == 0 or d == 0:
        return 0
    elif (d, target) in cache:
        return cache[(d, target)]
    
    res = 0
    for num in xrange(1, f + 1):
        res += throw_dice_with_cache(d - 1, f, target - num, cache)
    
    cache[(d, target)] = res % MODULE_NUM
    return cache[(d, target)]


class ThrowDiceSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(15, throw_dice(3, 6, 7))

    def test_example1(self):
        self.assertEqual(1, throw_dice(1, 6, 3))

    def test_example2(self):
        self.assertEqual(6, throw_dice(2, 6, 7))

    def test_example3(self):
        self.assertEqual(1, throw_dice(2, 5, 10))

    def test_example4(self):
        self.assertEqual(0, throw_dice(1, 2, 3))

    def test_example5(self):
        self.assertEqual(222616187, throw_dice(30, 30, 500))

    def test_target_total_too_large(self):
        self.assertEqual(0, throw_dice(1, 6, 12))
    
    def test_target_total_too_small(self):
        self.assertEqual(0, throw_dice(4, 2, 1))
    
    def test_throw_dice1(self):
        self.assertEqual(2, throw_dice(2, 2, 3))
    
    def test_throw_dice2(self):
        self.assertEqual(21, throw_dice(6, 3, 8))
    
    def test_throw_dice3(self):
        self.assertEqual(4, throw_dice(4, 2, 5))
    
    def test_throw_dice4(self):
        self.assertEqual(6, throw_dice(3, 4, 5))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 17, 2021 \[Medium\] Substrings with Exactly K Distinct Characters
---
> **Question:** Given a string `s` and an int `k`, return an int representing the number of substrings (not unique) of `s` with exactly `k` distinct characters. 
>
> If the given string doesn't have `k` distinct characters, return `0`.

**Example 1:**
```py
Input: s = "pqpqs", k = 2
Output: 7
Explanation: ["pq", "pqp", "pqpq", "qp", "qpq", "pq", "qs"]
```

**Example 2:**
```py
Input: s = "aabab", k = 3
Output: 0
```

**My thoughts:** Counting number of substr with exact k distinct char is hard. However, counting substr with at most k distinct char is easier: just mantain a sliding window of (start, end) and in each iteration count number of substring ended at end, that is, `end - start + 1`. 

`Number of exact k-distinct substring = Number of k-most substr - Number of (k-1)most substr`.

**Solution with Sliding Window:** [https://repl.it/@trsong/Substrings-with-Exactly-K-Distinct-Characters](https://repl.it/@trsong/Substrings-with-Exactly-K-Distinct-Characters)
```py
import unittest

def count_k_distinct_substring(s, k):
    if k <= 0 or k > len(s):
        return 0

    return count_k_most_substring(s, k) - count_k_most_substring(s, k - 1)


def count_k_most_substring(s, k):
    char_freq = {}
    start = 0
    res = 0

    for end, incoming_char in enumerate(s):
        char_freq[incoming_char] = char_freq.get(incoming_char, 0) + 1
        while len(char_freq) > k:
            outgoing_char = s[start]
            char_freq[outgoing_char] -= 1
            if char_freq[outgoing_char] == 0:
                del char_freq[outgoing_char]
            start += 1
        res += end - start + 1
         
    return res


class CountKDistinctSubstringSpec(unittest.TestCase):
    def test_example(self):
        k, s = 2, 'pqpqs'
        # pq, pqp, pqpq, qp, qpq, pq, qs
        expected = 7
        self.assertEqual(expected, count_k_distinct_substring(s, k))

    def test_example2(self):
        k, s = 3, 'aabab'
        expected = 0
        self.assertEqual(expected, count_k_distinct_substring(s, k))
    
    def test_k_is_zero(self):
        k, s = 0, 'abc'
        expected = 0
        self.assertEqual(expected, count_k_distinct_substring(s, k))

    def test_substring_does_not_need_to_be_unique(self):
        k, s = 2, 'aba'
        # ab, ba, aba
        expected = 3
        self.assertEqual(expected, count_k_distinct_substring(s, k))

    def test_substring_does_not_need_to_be_unique2(self):
        k, s = 3, 'abcdbacba'
        # abc, bcd, cdb, dba, bac, acb, cba, bcdb, acba, bacb, bacba
        expected = 11
        self.assertEqual(expected, count_k_distinct_substring(s, k))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 16, 2021 \[Medium\] Shortest Unique Prefix
---
> **Question:** Given an array of words, find all shortest unique prefixes to represent each word in the given array. Assume that no word is prefix of another.

**Example:**
```py
Input: ['zebra', 'dog', 'duck', 'dove']
Output: ['z', 'dog', 'du', 'dov']
Explanation: dog => dog
             dove = dov 
             duck = du
             z   => zebra 
```

**My thoughts:** Most string prefix searching problem can be solved using Trie (Prefix Tree). A trie is a N-nary tree with each edge represent a char. Each node will have two attribues: is_end and count, representing if a word is end at this node and how many words share same prefix so far separately. 

The given example will generate the following trie:
```py
The number inside each parenthsis represents how many words share the same prefix underneath.

             ""(4)
         /  |    |   \  
      d(1) c(1) a(2) f(1)
     /      |    |      \
   o(1)    a(1) p(2)   i(1)
  /         |    |  \     \
g(1)       t(1) p(1) r(1) s(1)
                 |    |    |
                l(1) i(1) h(1)
                 |    |
                e(1) c(1)
                      |
                     o(1)
                      |
                     t(1)
```
Our goal is to find a path for each word from root to the first node that has count equals 1, above example gives: `d, c, app, apr, f`

**Solution with Trie:** [https://repl.it/@trsong/Find-All-Shortest-Unique-Prefix](https://repl.it/@trsong/Find-All-Shortest-Unique-Prefix)
```py
import unittest

class Trie(object):
    def __init__(self):
        self.children = {}
        self.count = 0

    def insert(self, word):
        p = self
        for ch in word:
            if ch not in p.children:
                p.children[ch] = Trie()
            p = p.children[ch]
            p.count += 1

    def find_prefix(self, word):
        p = self
        for i, ch in enumerate(word):
            if p.count == 1:
                return word[:i]
            p = p.children[ch]
        return word


def shortest_unique_prefix(words):
    trie = Trie()
    for word in words:
        trie.insert(word)
    
    return map(lambda word: trie.find_prefix(word), words)


class UniquePrefixSpec(unittest.TestCase):
    def test_example(self):
        words = ['zebra', 'dog', 'duck', 'dove']
        expected = ['z', 'dog', 'du', 'dov']
        self.assertEqual(expected, shortest_unique_prefix(words))
    
    def test_example2(self):
        words = ['dog', 'cat', 'apple', 'apricot', 'fish']
        expected = ['d', 'c', 'app', 'apr', 'f']
        self.assertEqual(expected, shortest_unique_prefix(words))
    
    def test_empty_word(self):
        words = ['', 'alpha', 'aztec']
        expected = ['', 'al', 'az']
        self.assertEqual(expected, shortest_unique_prefix(words))

    def test_prefix_overlapp_with_each_other(self):
        words = ['abc', 'abd', 'abe', 'abf', 'abg']
        expected = ['abc', 'abd', 'abe', 'abf', 'abg']
        self.assertEqual(expected, shortest_unique_prefix(words))
    
    def test_only_entire_word_is_shortest_unique_prefix(self):
        words = ['greek', 'greedisbad', 'greedisgood', 'greeting']
        expected = ['greek', 'greedisb', 'greedisg', 'greet']
        self.assertEqual(expected, shortest_unique_prefix(words))

    def test_unique_prefix_is_not_empty_string(self):
        words = ['naturalwonders']
        expected = ['n']
        self.assertEqual(expected, shortest_unique_prefix(words))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 15, 2021 \[Easy\] Fixed Point
---
> **Questions:** A fixed point in an array is an element whose value is equal to its index. Given a sorted array of distinct elements, return a fixed point, if one exists. Otherwise, return `False`.
>
> For example, given `[-6, 0, 2, 40]`, you should return `2`. Given `[1, 5, 7, 8]`, you should return `False`.

**Solution with Binary Search:** [https://repl.it/@trsong/Find-the-Fixed-Point](https://repl.it/@trsong/Find-the-Fixed-Point)
```py
import unittest

def fixed_point(nums):
    if not nums:
        return False

    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == mid:
            return mid
        if nums[mid] < mid:
            lo = mid + 1
        else:
            hi = mid - 1
    return False
            

class FixedPointSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(2, fixed_point([-6, 0, 2, 40]))

    def test_example2(self):
        self.assertFalse(fixed_point([1, 5, 7, 8]))

    def test_empty_array(self):
        self.assertFalse(fixed_point([]))
    
    def test_array_without_fixed_point(self):
        self.assertFalse(fixed_point([10, 10, 10, 10]))
    
    def test_array_without_fixed_point2(self):
        self.assertFalse(fixed_point([1, 2, 3, 4]))
    
    def test_array_without_fixed_point3(self):
        self.assertFalse(fixed_point([-1, 0, 1]))

    def test_sorted_array_with_duplicate_elements1(self):
        self.assertEqual(4, fixed_point([-10, 0, 1, 2, 4, 10, 10, 10, 20, 30]))

    def test_sorted_array_with_duplicate_elements2(self):
        self.assertEqual(3, fixed_point([-1, 3, 3, 3, 3, 3, 4]))

    def test_sorted_array_with_duplicate_elements3(self):
        self.assertEqual(6, fixed_point([-3, 0, 0, 1, 3, 4, 6, 8, 10]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 14, 2021 \[Easy\] Filter Binary Tree Leaves
---
> **Questions:** Given a binary tree and an integer k, filter the binary tree such that its leaves don't contain the value k. Here are the rules:
>
> - If a leaf node has a value of k, remove it.
> - If a parent node has a value of k, and all of its children are removed, remove it.

**Example:**
```py
Given the tree to be the following and k = 1:
     1
    / \
   1   1
  /   /
 2   1

After filtering the result should be:
     1
    /
   1
  /
 2
```

**Solution:** [https://repl.it/@trsong/Filter-Binary-Tree-Leaves-of-Certain-Value](https://repl.it/@trsong/Filter-Binary-Tree-Leaves-of-Certain-Value)
```py
import unittest

def filter_tree_leaves(tree, k):
    if not tree:
        return None

    left_res = filter_tree_leaves(tree.left, k)
    right_res = filter_tree_leaves(tree.right, k)
    if not left_res and not right_res and tree.val == k:
        return None
    else:
        tree.left = left_res
        tree.right = right_res
        return tree


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __eq__(self, other):
        return other and other.val == self.val and self.left == other.left and self.right == other.right

    def __repr__(self):
        stack = [(self, 0)]
        res = ['\n']
        while stack:
            cur, depth = stack.pop()
            res.append('\t' * depth)
            if cur:
                res.append('* ' + str(cur.val))
                stack.append((cur.right, depth + 1))
                stack.append((cur.left, depth + 1))
            else:
                res.append('* None')
            res.append('\n')            
        return ''.join(res)


class FilterTreeLeaveSpec(unittest.TestCase):
    def test_example(self):
        """
        Input:
             1
            / \
           1   1
          /   /
         2   1

        Result:
             1
            /
           1
          /
         2
        """
        left_tree = TreeNode(1, TreeNode(2))
        right_tree = TreeNode(1, TreeNode(1))
        root = TreeNode(1, left_tree, right_tree)
        expected_tree = TreeNode(1, TreeNode(1, TreeNode(2)))
        self.assertEqual(expected_tree, filter_tree_leaves(root, k=1))

    def test_remove_empty_tree(self):
        self.assertIsNone(filter_tree_leaves(None, 1))

    def test_remove_the_last_node(self):
        self.assertIsNone(filter_tree_leaves(TreeNode(2), 2))

    def test_filter_cause_all_nodes_to_be_removed(self):
        k = 42
        left_tree = TreeNode(k, right=TreeNode(k))
        right_tree = TreeNode(k, TreeNode(k), TreeNode(k))
        tree = TreeNode(k, left_tree, right_tree)
        self.assertIsNone(filter_tree_leaves(tree, k))

    def test_filter_not_internal_nodes(self):
        from copy import deepcopy
        """
             1
           /   \
          1     1
         / \   /
        2   2 2
        """
        left_tree = TreeNode(1, TreeNode(2), TreeNode(2))
        right_tree = TreeNode(1, TreeNode(2))
        root = TreeNode(1, left_tree, right_tree)
        expected = deepcopy(root)
        self.assertEqual(expected, filter_tree_leaves(root, k=1))
    
    def test_filter_only_leaves(self):
        """
        Input :    
               4
            /     \
           5       5
         /  \    /
        3    1  5 
      
        Output :  
            4
           /     
          5       
         /  \    
        3    1  
        """
        left_tree = TreeNode(5, TreeNode(3), TreeNode(1))
        right_tree = TreeNode(5, TreeNode(5))
        root = TreeNode(4, left_tree, right_tree)
        expected = TreeNode(4, TreeNode(5, TreeNode(3), TreeNode(1)))
        self.assertEqual(expected, filter_tree_leaves(root, 5))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 13, 2021 \[Hard\] Max Path Value in Directed Graph
---
> **Question:** In a directed graph, each node is assigned an uppercase letter. We define a path's value as the number of most frequently-occurring letter along that path. For example, if a path in the graph goes through "ABACA", the value of the path is 3, since there are 3 occurrences of 'A' on the path.
>
> Given a graph with n nodes and m directed edges, return the largest value path of the graph. If the largest value is infinite, then return null.
>
> The graph is represented with a string and an edge list. The i-th character represents the uppercase letter of the i-th node. Each tuple in the edge list (i, j) means there is a directed edge from the i-th node to the j-th node. Self-edges are possible, as well as multi-edges.

**Example1:** 
```py
Input: "ABACA", [(0, 1), (0, 2), (2, 3), (3, 4)]
Output: 3
Explanation: maximum value 3 using the path of vertices [0, 2, 3, 4], ie. A -> A -> C -> A.
```

**Example2:**
```py
Input: "A", [(0, 0)]
Output: None
Explanation: we have an infinite loop.
```


**My thoughts:** This question is a perfect example illustrates how to apply different teachniques, such as Toplogical Sort and DP, to solve a graph problem.

The brute force solution is to iterate through all possible vertices and start from where we can search neighbors recursively and find the maximum path value. Which takes `O(V * (V + E))`.

However, certain nodes will be calculated over and over again. e.g. "AAB", [(0, 1), (2, 1)] both share same neighbor second A.

Thus, in order to speed up, we can use DP to cache the intermediate result. Let `dp[v][letter]` represents the path value starts from v with the letter. `dp[v][non_current_letter] = max(dp[nb][non_current_letter]) for all neighbour nb ` or `dp[v][current_letter] = max(dp[nb][current_letter]) + 1 for all neighbour nb`.  The final solution is `max{ dp[v][current_letter_v] } for all v`.

With DP solution, the time complexity drop to `O(V + E)`, 'cause each vertix and edge only needs to be visited once.


**Solution with Toplogical Sort and DP:** [https://repl.it/@trsong/Find-Max-Letter-Path-Value-in-Directed-Graph](https://repl.it/@trsong/Find-Max-Letter-Path-Value-in-Directed-Graph)
```py
import unittest
from collections import defaultdict


def find_max_path_value(letters, edges):
    if not letters:
        return 0

    n = len(letters)
    neighbors = [[] for _ in xrange(n)]
    inbound = [0] * n

    for src, dst in edges:
        neighbors[src].append(dst)
        inbound[dst] += 1
    
    top_order = find_top_order(neighbors, inbound)
    if not top_order:
        return None

    # Let dp[node][letter] represents path value of letter ended ended at node
    # dp[node][letter] = max(dp[prev][letter] for all prev)  if current node letter is not letter
    #                  = 1 + max(dp[prev][letter] for all prev)  if current node letter is letter
    dp = [defaultdict(int) for _ in xrange(n)]
    for node in top_order:
        dp[node][letters[node]] += 1

        for letter, count in dp[node].items():
            for nb in neighbors[node]:
                dp[nb][letter] = max(dp[nb][letter], count)
    return max(dp[top_order[-1]].values())


def find_top_order(neighbors, inbound):
    n = len(neighbors)
    queue = filter(lambda node: inbound[node] == 0, xrange(n))
    res = []

    while queue:
        node = queue.pop(0)
        res.append(node)

        for nb in neighbors[node]:
            inbound[nb] -= 1
            if inbound[nb] == 0:
                queue.append(nb)

    return res if len(res) == n else None


class FindMaxPathValueSpec(unittest.TestCase):
    def test_graph_with_self_edge(self):
        letters ='A'
        edges = [(0, 0)]
        self.assertIsNone(find_max_path_value(letters, edges))

    def test_example_graph(self):
        letters ='ABACA'
        edges = [(0, 1), (0, 2), (2, 3), (3, 4)]
        self.assertEqual(3, find_max_path_value(letters, edges))
    
    def test_empty_graph(self):
        self.assertEqual(0, find_max_path_value('', []))

    def test_diconnected_graph(self):
        self.assertEqual(1, find_max_path_value('AABBCCDD', []))
    
    def test_graph_with_cycle(self):
        letters ='XZYABC'
        edges = [(0, 1), (1, 2), (2, 0), (3, 2), (4, 3), (5, 3)]
        self.assertIsNone(find_max_path_value(letters, edges))

    def test_graph_with_disconnected_components(self):
        letters ='AABBB'
        edges = [(0, 1), (2, 3), (3, 4)]
        self.assertEqual(3, find_max_path_value(letters, edges))

    def test_complicated_graph(self):
        letters ='XZYZYZYZQX'
        edges = [(0, 1), (0, 9), (1, 9), (1, 3), (1, 5), (3, 5), 
            (3, 4), (5, 4), (5, 7), (1, 7), (2, 4), (2, 6), (2, 8), (9, 8)]
        self.assertEqual(4, find_max_path_value(letters, edges))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 12, 2021 LC 332 \[Medium\] Reconstruct Itinerary
---
> **Questions:** Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK. Thus, the itinerary must begin with JFK.

Note:

> 1. If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string. For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].
> 2. All airports are represented by three capital letters (IATA code).
> 3. You may assume all tickets form at least one valid itinerary.
   
**Example 1:**
```java
Input: [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
Output: ["JFK", "MUC", "LHR", "SFO", "SJC"]
```

**Example 2:**
```java
Input: [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
Explanation: Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"].
             But it is larger in lexical order.
```

**My thoughts:** Forget about lexical requirement for now, consider all airports as vertices and each itinerary as an edge. Then all we need to do is to find a path from "JFK" that consumes all edges. By using DFS to iterate all potential solution space, once we find a solution we will return immediately.

Now let's consider the lexical requirement, when we search from one node to its neighbor, we can go from smaller lexical order first and by keep doing that will lead us to the result. 

**Solution with DFS:** [https://repl.it/@trsong/Reconstruct-Flight-Itinerary](https://repl.it/@trsong/Reconstruct-Flight-Itinerary)
```py
import unittest

def reconstruct_itinerary(tickets):
    neighbors = {}
    for src, dst in sorted(tickets):
        if src not in neighbors:
            neighbors[src] = []
        neighbors[src].append(dst)
    res = []
    dfs_route(res, 'JFK', neighbors)
    return res[::-1]


def dfs_route(res, src, neighbors):
    while src in neighbors and neighbors[src]:
        dst = neighbors[src].pop(0)
        dfs_route(res, dst, neighbors)
    res.append(src)


class ReconstructItinerarySpec(unittest.TestCase):
    def test_example(self):
        tickets = [['MUC', 'LHR'], ['JFK', 'MUC'], ['SFO', 'SJC'],
                   ['LHR', 'SFO']]
        expected = ['JFK', 'MUC', 'LHR', 'SFO', 'SJC']
        self.assertEqual(expected, reconstruct_itinerary(tickets))

    def test_example2(self):
        tickets = [['JFK', 'SFO'], ['JFK', 'ATL'], ['SFO', 'ATL'],
                   ['ATL', 'JFK'], ['ATL', 'SFO']]
        expected = ['JFK', 'ATL', 'JFK', 'SFO', 'ATL', 'SFO']
        expected2 = ['JFK', 'SFO', 'ATL', 'JFK', 'ATL', 'SFO']
        self.assertIn(reconstruct_itinerary(tickets), [expected, expected2])

    def test_not_run_into_loop(self):
        tickets = [['JFK', 'YVR'], ['LAX', 'LAX'], ['YVR', 'YVR'],
                   ['YVR', 'YVR'], ['YVR', 'LAX'], ['LAX', 'LAX'],
                   ['LAX', 'YVR'], ['YVR', 'JFK']]
        expected = [
            'JFK', 'YVR', 'LAX', 'LAX', 'LAX', 'YVR', 'YVR', 'YVR', 'JFK'
        ]
        self.assertEqual(expected, reconstruct_itinerary(tickets))

    def test_not_run_into_form_loop2(self):
        tickets = [['JFK', 'YVR'], ['JFK', 'LAX'], ['LAX', 'JFK']]
        expected = ['JFK', 'LAX', 'JFK', 'YVR']
        self.assertEqual(expected, reconstruct_itinerary(tickets))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 11, 2021 \[Medium\] Generate Binary Search Trees
--- 
> **Question:** Given a number n, generate all binary search trees that can be constructed with nodes 1 to n.
 
**Example:**
```py
Pre-order traversals of binary trees from 1 to n:
- 123
- 132
- 213
- 312
- 321

   1      1      2      3      3
    \      \    / \    /      /
     2      3  1   3  1      2
      \    /           \    /
       3  2             2  1
``` 

**Solution:** [https://repl.it/@trsong/Generate-All-Binary-Search-Trees](https://repl.it/@trsong/Generate-All-Binary-Search-Trees)
```py

import unittest

def generate_bst(n):
    if n < 1:
        return []
    return generate_bst_recur(1, n)


def generate_bst_recur(lo, hi):
    if lo > hi:
        return [None]

    res = []
    for val in xrange(lo, hi + 1):
        for left_child in generate_bst_recur(lo, val - 1):
            for right_child in generate_bst_recur(val + 1, hi):
                res.append(TreeNode(val, left_child, right_child))
    return res


###################
# Testing Utilities
###################
class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def preorder_traversal(self):
        res = [self.val]
        if self.left:
            res += self.left.preorder_traversal()
        if self.right:
            res += self.right.preorder_traversal()
        return res


class GenerateBSTSpec(unittest.TestCase):
    def assert_result(self, expected_preorder_traversal, bst_seq):
        self.assertEqual(len(expected_preorder_traversal), len(bst_seq))
        result_traversal = map(lambda t: t.preorder_traversal(), bst_seq)
        self.assertEqual(sorted(expected_preorder_traversal), sorted(result_traversal))

    def test_example(self):
        expected_preorder_traversal = [
            [1, 2, 3],
            [1, 3, 2],
            [2, 1, 3],
            [3, 1, 2],
            [3, 2, 1]
        ]
        self.assert_result(expected_preorder_traversal, generate_bst(3))
    
    def test_empty_tree(self):
        self.assertEqual([], generate_bst(0))

    def test_base_case(self):
        expected_preorder_traversal = [[1]]
        self.assert_result(expected_preorder_traversal, generate_bst(1))

    def test_generate_4_nodes(self):
        expected_preorder_traversal = [
            [1, 2, 3, 4],
            [1, 2, 4, 3],
            [1, 3, 2, 4],
            [1, 4, 2, 3],
            [1, 4, 3, 2],
            [2, 1, 3, 4],
            [2, 1, 4, 3],
            [3, 1, 2, 4],
            [3, 2, 1, 4],
            [4, 1, 2, 3],
            [4, 1, 3, 2],
            [4, 2, 1, 3],
            [4, 3, 1, 2],
            [4, 3, 2, 1]
        ]
        self.assert_result(expected_preorder_traversal, generate_bst(4))
    

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 10, 2021 \[Easy\] Making a Height Balanced Binary Search Tree
---
> **Question:** Given a sorted list, create a height balanced binary search tree, meaning the height differences of each node can only differ by at most 1.

**Solution:** [https://repl.it/@trsong/Making-a-Height-Balanced-Binary-Search-Tree](https://repl.it/@trsong/Making-a-Height-Balanced-Binary-Search-Tree)
```py
import unittest

def build_balanced_bst(sorted_list):
    def build_balanced_bst_recur(left, right):
        if left > right:
            return None
        mid = left + (right - left) // 2
        left_res = build_balanced_bst_recur(left, mid - 1)
        right_res = build_balanced_bst_recur(mid + 1, right)
        return TreeNode(sorted_list[mid], left_res, right_res)

    return build_balanced_bst_recur(0, len(sorted_list) - 1)


#####################
# Testing Utilities
#####################
class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def flatten(self):
        res = []
        stack = []
        p = self
        while p or stack:
            if p:
                stack.append(p)
                p = p.left
            elif stack:
                p = stack.pop()
                res.append(p.val)
                p = p.right
        return res

    def is_balanced(self):
        res, _ = self._is_balanced_helper()
        return res

    def _is_balanced_helper(self):
        left_res, left_height = True, 0
        right_res, right_height = True, 0
        if self.left:
            left_res, left_height = self.left._is_balanced_helper()
        if self.right:
            right_res, right_height = self.right._is_balanced_helper()
        res = left_res and right_res and abs(left_height - right_height) <= 1
        if left_res and right_res and not res:
            print "Violate balance requirement"
            print self
        return res, max(left_height, right_height) + 1

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


class BuildBalancdBSTSpec(unittest.TestCase):
    def test_empty_list(self):
        self.assertIsNone(build_balanced_bst([]))

    def test_one_elem_list(self):
        lst = [42]
        root = build_balanced_bst(lst)
        self.assertTrue(root.is_balanced())
        self.assertEqual(lst, root.flatten())

    def test_list1(self):
        lst = [1, 2, 3]
        root = build_balanced_bst(lst)
        self.assertTrue(root.is_balanced())
        self.assertEqual(lst, root.flatten())

    def test_list2(self):
        lst = [1, 5, 6, 9, 10, 11, 13]
        root = build_balanced_bst(lst)
        self.assertTrue(root.is_balanced())
        self.assertEqual(lst, root.flatten())

    def test_list_with_duplicated_elem(self):
        lst = [1, 1, 2, 2, 3, 3, 3, 3, 3]
        root = build_balanced_bst(lst)
        self.assertTrue(root.is_balanced())
        self.assertEqual(lst, root.flatten())

    def test_list_with_duplicated_elem2(self):
        lst = [1, 1, 1, 1, 1, 1]
        root = build_balanced_bst(lst)
        self.assertTrue(root.is_balanced())
        self.assertEqual(lst, root.flatten())


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 9, 2021 \[Medium\] LRU Cache
---
> **Question:** Implement an LRU (Least Recently Used) cache. It should be able to be initialized with a cache size n, and contain the following methods:
>
> - `put(key, value)`: sets key to value. If there are already n items in the cache and we are adding a new item, then it should also remove the least recently used item.
> - `get(key)`: gets the value at key. If no such key exists, return null.
>  
> Each operation should run in O(1) time.

**Example:**
```py
cache = LRUCache(2)
cache.put(3, 3)
cache.put(4, 4)
cache.get(3)  # returns 3
cache.get(2)  # returns None

cache.put(2, 2)
cache.get(4)  # returns None (pre-empted by 2)
cache.get(3)  # returns 3
```

**Solution:** [https://repl.it/@trsong/Implement-the-LRU-Cache](https://repl.it/@trsong/Implement-the-LRU-Cache)
```py
import unittest

class Node(object):
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = None
        self.prev = None


class LinkedList(object):
    def __init__(self):
        self.head = Node(-1, -1)
        self.tail = Node(-1, -1)
        self.head.next = self.tail
        self.tail.prev = self.head

    def add(self, node):
        prev = self.tail.prev
        prev.next = node
        node.prev = prev
        node.next = self.tail
        self.tail.prev = node

    def remove(self, node):
        prev = node.prev
        prev.next = node.next
        node.next.prev = prev
        node.prev = None
        node.next = None
    

class LRUCache(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.key_node_map = {}
        self.list = LinkedList()

    def get(self, key):
        if key not in self.key_node_map:
            return None
        
        node = self.key_node_map[key]
        self.list.remove(node)
        self.list.add(node)
        return node.val

    def put(self, key, val):
        if key in self.key_node_map:
            self.list.remove(self.key_node_map[key])

        self.key_node_map[key] = Node(key, val)
        self.list.add(self.key_node_map[key])
        
        if len(self.key_node_map) > self.capacity:
            evit_node = self.list.head.next
            self.list.remove(evit_node)
            del self.key_node_map[evit_node.key]


class LRUCacheSpec(unittest.TestCase):
    def test_example(self):
        cache = LRUCache(2)
        cache.put(3, 3)
        cache.put(4, 4)
        self.assertEqual(3, cache.get(3))
        self.assertIsNone(cache.get(2))

        cache.put(2, 2)
        self.assertIsNone(cache.get(4))  # returns None (pre-empted by 2)
        self.assertEqual(3, cache.get(3))

    def test_get_element_from_empty_cache(self):
        cache = LRUCache(1)
        self.assertIsNone(cache.get(-1))

    def test_cachecapacity_is_one(self):
        cache = LRUCache(1)
        cache.put(-1, 42)
        self.assertEqual(42, cache.get(-1))

        cache.put(-1, 10)
        self.assertEqual(10, cache.get(-1))

        cache.put(2, 0)
        self.assertIsNone(cache.get(-1))
        self.assertEqual(0, cache.get(2))

    def test_evict_most_inactive_element_when_cache_is_full(self):
        cache = LRUCache(3)
        cache.put(1, 1)
        cache.put(1, 1)
        cache.put(2, 1)
        cache.put(3, 1)

        cache.put(4, 1)
        self.assertIsNone(cache.get(1))

        cache.put(2, 1)
        cache.put(5, 1)
        self.assertIsNone(cache.get(3))

    def test_update_element_should_get_latest_value(self):
        cache = LRUCache(2)
        cache.put(3, 10)
        cache.put(3, 42)
        cache.put(1, 1)
        cache.put(1, 2)
        self.assertEqual(42, cache.get(3))

    def test_end_to_end_workflow(self):
        cache = LRUCache(3)
        cache.put(0, 0)  # Least Recent -> 0 -> Most Recent
        cache.put(1, 1)  # Least Recent -> 0, 1 -> Most Recent
        cache.put(2, 2)  # Least Recent -> 0, 1, 2 -> Most Recent
        cache.put(3, 3)  # Least Recent -> 1, 2, 3 -> Most Recent. Evict 0
        self.assertIsNone(cache.get(0))  
        self.assertEqual(2, cache.get(2))  # Least Recent -> 1, 3, 2 -> Most Recent
        cache.put(4, 4)  # Least Recent -> 3, 2, 4 -> Most Recent. Evict 1 
        self.assertIsNone(cache.get(1))
        self.assertEqual(2, cache.get(2))  # Least Recent -> 3, 4, 2 -> Most Recent 
        self.assertEqual(3, cache.get(3))  # Least Recent -> 4, 2, 3 -> Most Recent
        self.assertEqual(2, cache.get(2))  # Least Recent -> 4, 3, 2 -> Most Recent
        cache.put(5, 5)  # Least Recent -> 3, 2, 5 -> Most Recent. Evict 4
        cache.put(6, 6)  # Least Recent -> 2, 5, 6 -> Most Recent. Evict 3
        self.assertIsNone(cache.get(4))
        self.assertIsNone(cache.get(3))
        cache.put(7, 7)  # Least Recent -> 5, 6, 7 -> Most Recent. Evict 2
        self.assertIsNone(cache.get(2))

    def test_end_to_end_workflow2(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        self.assertEqual(cache.get(1), 1)
        cache.put(3, 3)  # evicts key 2
        self.assertIsNone(cache.get(2))
        cache.put(4, 4)  # evicts key 1
        self.assertIsNone(cache.get(1))
        self.assertEqual(3, cache.get(3))
        self.assertEqual(4, cache.get(4))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 8, 2021 \[Hard\] LFU Cache
---
> **Question:** Implement an LFU (Least Frequently Used) cache. It should be able to be initialized with a cache size n, and contain the following methods:
>
> - `put(key, value)`: sets key to value. If there are already n items in the cache and we are adding a new item, then it should also remove the least frequently used item. If there is a tie, then the least recently used key should be removed.
> - `get(key)`: gets the value at key. If no such key exists, return null.
Each operation should run in O(1) time.

**My thoughts:** Create two maps:

- `Key - (Value, Freq)` Map
- `Freq - Deque<Key>` Map: new elem goes from right and old elem stores on the left. 

We also use a variable to point to minimum frequency, so when it meets capacity, elem with min freq and on the left of deque will first be evicted.

**Solution:** [https://repl.it/@trsong/Design-LFU-Cache](https://repl.it/@trsong/Design-LFU-Cache)
```py
import unittest
from collections import defaultdict
from collections import deque

class LFUCache(object):
    def __init__(self, capacity):
        self.kvf_map = {}
        self.freq_order_map = defaultdict(deque)
        self.min_freq = 1
        self.capacity = capacity

    def get(self, key):
        if key not in self.kvf_map:
            return None

        val, freq = self.kvf_map[key]
        self.kvf_map[key] = (val, freq + 1)

        self.freq_order_map[freq].remove(key)
        self.freq_order_map[freq + 1].append(key)
        
        if not self.freq_order_map[freq]:
            del self.freq_order_map[freq]
            if freq == self.min_freq:
                self.min_freq += 1
        return val

    def put(self, key, val):
        if self.capacity <= 0:
            return 

        if key in self.kvf_map:
            _, freq = self.kvf_map[key]
            self.kvf_map[key] = (val, freq)
            self.get(key)
        else:
            if len(self.kvf_map) >= self.capacity:
                evict_key = self.freq_order_map[self.min_freq].popleft()
                if not self.freq_order_map[self.min_freq]:
                    del self.freq_order_map[self.min_freq]
                del self.kvf_map[evict_key]
            
            self.kvf_map[key] = (val, 1)
            self.freq_order_map[1].append(key)
            self.min_freq = 1


class LFUCacheSpec(unittest.TestCase):
    def test_empty_cache(self):
        cache = LFUCache(0)
        self.assertIsNone(cache.get(0))
        cache.put(0, 0)

    def test_end_to_end_workflow(self):
        cache = LFUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        self.assertEqual(1, cache.get(1))
        cache.put(3, 3)  # remove key 2
        self.assertIsNone(cache.get(2))  # key 2 not found
        self.assertEqual(3, cache.get(3))
        cache.put(4, 4)  # remove key 1
        self.assertIsNone(cache.get(1))  # key 1 not found
        self.assertEqual(3, cache.get(3))
        self.assertEqual(4, cache.get(4))

    def test_end_to_end_workflow2(self):
        cache = LFUCache(3)
        cache.put(2, 2)
        cache.put(1, 1)
        self.assertEqual(2, cache.get(2))
        self.assertEqual(1, cache.get(1))
        self.assertEqual(2, cache.get(2))
        cache.put(3, 3)
        cache.put(4, 4)  # remove key 3
        self.assertIsNone(cache.get(3))
        self.assertEqual(2, cache.get(2))
        self.assertEqual(1, cache.get(1))
        self.assertEqual(4, cache.get(4))

    def test_end_to_end_workflow3(self):
        cache = LFUCache(2)
        cache.put(3, 1)
        cache.put(2, 1)
        cache.put(2, 2)
        cache.put(4, 4)
        self.assertEqual(2, cache.get(2))

    def test_remove_least_freq_elements_when_evict(self):
        cache = LFUCache(3)
        cache.put(1, 'a')
        cache.put(1, 'aa')
        cache.put(1, 'aaa')
        cache.put(2, 'b')
        cache.put(2, 'bb')
        cache.put(3, 'c')
        cache.put(4, 'd')
        self.assertIsNone(cache.get(3))
        self.assertEqual('d', cache.get(4))
        cache.get(4)
        cache.get(4)
        cache.get(2)
        cache.get(2)
        cache.put(3, 'cc')
        self.assertIsNone(cache.get(1))
        self.assertEqual('cc', cache.get(3))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 7, 2021 \[Hard\] Regular Expression: Period and Asterisk
---
> **Question:** Implement regular expression matching with the following special characters:
>
> - `.` (period) which matches any single character
> 
> - `*` (asterisk) which matches zero or more of the preceding element
> 
> That is, implement a function that takes in a string and a valid regular expression and returns whether or not the string matches the regular expression.
>
> For example, given the regular expression "ra." and the string "ray", your function should return true. The same regular expression on the string "raymond" should return false.
>
> Given the regular expression ".*at" and the string "chat", your function should return true. The same regular expression on the string "chats" should return false.

**My thoughts:** First consider the solution without `*` (asterisk), then we just need to match letters one by one while doing some special handling for `.` (period) so that it can match all letters. 

Then we consider the solution with `*`, we will need to perform ONE look-ahead, so `*` can represent 0 or more matching. And we consider those two situations separately:
- If we have 1 matching, we just need to check if the rest of text match the current pattern.
- Or if we do not have matching, then we need to advance both text and pattern.

**Solution:** [https://repl.it/@trsong/Regular-Expression-Period-and-Asterisk](https://repl.it/@trsong/Regular-Expression-Period-and-Asterisk)
```py
import unittest

def pattern_match(text, pattern):
    if not pattern:
        return not text
    
    match_first = text and (pattern[0] == text[0] or pattern[0] == '.')

    if len(pattern) > 1 and pattern[1] == '*':
        return pattern_match(text, pattern[2:]) or (match_first and pattern_match(text[1:], pattern))
    else:
        return match_first and pattern_match(text[1:], pattern[1:])
    

class PatternMatchSpec(unittest.TestCase):
    def test_empty_pattern(self):
        text = "a"
        pattern = ""
        self.assertFalse(pattern_match(text, pattern))

    def test_empty_text(self):
        text = ""
        pattern = "a"
        self.assertFalse(pattern_match(text, pattern))

    def test_empty_text_and_pattern(self):
        text = ""
        pattern = ""
        self.assertTrue(pattern_match(text, pattern))

    def test_asterisk(self):
        text = "aa"
        pattern = "a*"
        self.assertTrue(pattern_match(text, pattern))

    def test_asterisk2(self):
        text = "aaa"
        pattern = "ab*ac*a"
        self.assertTrue(pattern_match(text, pattern))

    def test_asterisk3(self):
        text = "aab"
        pattern = "c*a*b"
        self.assertTrue(pattern_match(text, pattern))

    def test_period(self):
        text = "ray"
        pattern = "ra."
        self.assertTrue(pattern_match(text, pattern))

    def test_period2(self):
        text = "raymond"
        pattern = "ra."
        self.assertFalse(pattern_match(text, pattern))

    def test_period_and_asterisk(self):
        text = "chat"
        pattern = ".*at"
        self.assertTrue(pattern_match(text, pattern))

    def test_period_and_asterisk2(self):
        text = "chats"
        pattern = ".*at"
        self.assertFalse(pattern_match(text, pattern))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 6, 2021 LC 388 \[Medium\] Longest Absolute File Path
---
> **Question:** Suppose we represent our file system by a string in the following manner:
>
> The string `"dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"` represents:

```py
dir
    subdir1
    subdir2
        file.ext
```

> The directory dir contains an empty sub-directory subdir1 and a sub-directory subdir2 containing a file file.ext.
>
> The string `"dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"` represents:

```py
dir
    subdir1
        file1.ext
        subsubdir1
    subdir2
        subsubdir2
            file2.ext
```

> The directory dir contains two sub-directories subdir1 and subdir2. subdir1 contains a file file1.ext and an empty second-level sub-directory subsubdir1. subdir2 contains a second-level sub-directory subsubdir2 containing a file file2.ext.
>
> We are interested in finding the longest (number of characters) absolute path to a file within our file system. For example, in the second example above, the longest absolute path is `"dir/subdir2/subsubdir2/file2.ext"`, and its length is `32` (not including the double quotes).
> 
> Given a string representing the file system in the above format, return the length of the longest absolute path to a file in the abstracted file system. If there is no file in the system, return `0`.

**Note:**
- The name of a file contains at least a period and an extension.
- The name of a directory or sub-directory will not contain a period.


**Solution:** [https://repl.it/@trsong/Longest-Absolute-File-Path](https://repl.it/@trsong/Longest-Absolute-File-Path)
```py
import unittest

def longest_abs_file_path(fs):
    lines = fs.split('\n')
    res = 0
    prev_indent_len = [0] * len(lines)
    for line in lines:
        indent = line.count('\t')
        parent_line_size = prev_indent_len[indent - 1] if indent > 0 else -1
        line_size = parent_line_size + 1 + len(line) - indent
        prev_indent_len[indent] = line_size

        if '.' in line:
            res = max(res, line_size)

    return res
        

class LongestAbsFilePathSpec(unittest.TestCase):
    def test_example(self):
        """
        dir
            subdir1
            subdir2
                file.ext
        """
        fs = "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"
        expected = len("dir/subdir2/file.ext")
        self.assertEqual(expected, longest_abs_file_path(fs))

    def test_example2(self):
        """
        dir
            subdir1
                file1.ext
                subsubdir1
            subdir2
                subsubdir2
                    file2.ext
        """
        fs = "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"
        expected = len("dir/subdir2/subsubdir2/file2.ext")
        self.assertEqual(expected, longest_abs_file_path(fs))

    def test_empty_fs(self):
        self.assertEqual(0, longest_abs_file_path(""))

    def test_empty_folder_with_no_files(self):
        self.assertEqual(0, longest_abs_file_path("folder"))

    def test_nested_folder_with_no_files(self):
        """
        a
            b
                c
            d
                e
        """
        fs = "a\n\tb\n\t\tc\n\td\n\t\te"
        self.assertEqual(0, longest_abs_file_path(fs))

    def test_shoft_file_path_vs_long_folder_with_no_file_path(self):
        """
        dir
            subdir1
                subdir2
            a.txt
        """
        fs = "dir\n\tsubdir1\n\t\tsubdir2\n\ta.txt"
        expected = len("dir/a.txt")
        self.assertEqual(expected, longest_abs_file_path(fs))

    def test_nested_fs(self):
        """
        dir
            sub1
                sub2
                    sub3
                        file1.txt
                    sub4
                        file2.in
                sub5
                    file3.in
            sub6
                file4.in
                sub7
                    file5.in
                    sub8
                        file6.output
            file7.txt            
        """
        fs = 'dir\n\tsub1\n\t\tsub2\n\t\t\tsub3\n\t\t\t\tfile1.txt\n\t\t\tsub4\n\t\t\t\tfile2.in\n\t\tsub5\n\t\t\tfile3.in\n\tsub6\n\t\tfile4.in\n\t\tsub7\n\t\t\tfile5.in\n\t\t\tsub8\n\t\t\t\tfile6.output\n\tfile7.txt'
        expected = len('dir/sub6/sub7/sub8/file6.output')
        self.assertEqual(expected, longest_abs_file_path(fs))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 5, 2021 LC 120 \[Easy\] Max Path Sum in Triangle
---
> **Question:** You are given an array of arrays of integers, where each array corresponds to a row in a triangle of numbers. For example, `[[1], [2, 3], [1, 5, 1]]` represents the triangle:


```py
  1
 2 3
1 5 1
```

> We define a path in the triangle to start at the top and go down one row at a time to an adjacent value, eventually ending with an entry on the bottom row. For example, `1 -> 3 -> 5`. The weight of the path is the sum of the entries.
>
> Write a program that returns the weight of the maximum weight path.


**Solution with DP:** [https://repl.it/@trsong/Max-Path-Sum-in-Triangle](https://repl.it/@trsong/Max-Path-Sum-in-Triangle)
```py
import unittest

def max_path_sum(triangle):
    if not triangle or not triangle[0]:
        return 0

    for r in xrange(len(triangle) - 2, -1, -1):
        for c in xrange(r + 1):
            left_child = triangle[r + 1][c]
            right_child = triangle[r + 1][c + 1]
            triangle[r][c] += max(left_child, right_child)

    return triangle[0][0]
        

class MathPathSumSpec(unittest.TestCase):
    def test_example(self):
        triangle = [
            [1], 
            [2, 3],
            [1, 5, 1]]
        expected = 9  # 1 -> 3 -> 5
        self.assertEqual(expected, max_path_sum(triangle))

    def test_empty_triangle(self):
        self.assertEqual(0, max_path_sum([]))

    def test_one_elem_triangle(self):
        triangle = [
            [-1]
        ]
        expected = -1
        self.assertEqual(expected, max_path_sum(triangle))
    
    def test_two_level_trianlge(self):
        triangle = [
            [1],
            [2, -1]
        ]
        expected = 3
        self.assertEqual(expected, max_path_sum(triangle))
    
    def test_all_negative_trianlge(self):
        triangle = [
            [-1],
            [-2, -3],
            [-4, -5, -6]
        ]
        expected = -7  # -1 -> -2 -> -4
        self.assertEqual(expected, max_path_sum(triangle))
    
    def test_greedy_solution_not_work(self):
        triangle = [
            [0],
            [2, 0],
            [3, 0, 0],
            [-10, -100, -100, 30]
        ]
        expected = 30
        self.assertEqual(expected, max_path_sum(triangle))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 4, 2021 LC 279 \[Medium\] Minimum Number of Squares Sum to N
---
> **Question:** Given a positive integer n, find the smallest number of squared integers which sum to n.
>
> For example, given `n = 13`, return `2` since `13 = 3^2 + 2^2 = 9 + 4`.
> 
> Given `n = 27`, return `3` since `27 = 3^2 + 3^2 + 3^2 = 9 + 9 + 9`.


**Solution with DP:** [https://repl.it/@trsong/Minimum-Squares-Sum-to-N](https://repl.it/@trsong/Minimum-Squares-Sum-to-N)
```py
import unittest
import math

def min_square_sum(n):
    # Let dp[n] represents min num sqr sum to n
    # dp[n] = min(dp[n - i * i]) + 1 for all i such that i * i <= n
    dp = [float('inf')] * (n + 1)
    dp[0] = 0

    for num in xrange(1, n + 1):
        for i in xrange(1, int(math.sqrt(num) + 1)):
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

### Feb 3, 2021 \[Medium\] Count Occurrence in Multiplication Table
---
> **Question:**  Suppose you have a multiplication table that is `N` by `N`. That is, a 2D array where the value at the `i-th` row and `j-th` column is `(i + 1) * (j + 1)` (if 0-indexed) or `i * j` (if 1-indexed).
>
> Given integers `N` and `X`, write a function that returns the number of times `X` appears as a value in an `N` by `N` multiplication table.
>
> For example, given `N = 6` and `X = 12`, you should return `4`, since the multiplication table looks like this:

```py
| 1 |  2 |  3 |  4 |  5 |  6 |

| 2 |  4 |  6 |  8 | 10 | 12 |

| 3 |  6 |  9 | 12 | 15 | 18 |

| 4 |  8 | 12 | 16 | 20 | 24 |

| 5 | 10 | 15 | 20 | 25 | 30 |

| 6 | 12 | 18 | 24 | 30 | 36 |
```
> And there are `4` 12's in the table.
 
**My thoughts:** Sometimes, it is against intuitive to solve a grid searching question without using grid searching stategies. But it could happen. As today's question is just a math problem features integer factorization.

**Solution:** [https://repl.it/@trsong/Count-Occurrence-in-Multiplication-Table](https://repl.it/@trsong/Count-Occurrence-in-Multiplication-Table)
 ```py
import unittest
import math

def count_number_in_table(N, X):
    if X <= 0:
        return 0

    sqrt_x = int(math.sqrt(X))
    res = 0
    for factor in xrange(1, min(sqrt_x, N) + 1):
        if X % factor == 0 and X / factor <= N:
            res += 2

    if sqrt_x * sqrt_x == X and sqrt_x <= N:
        # When candidate and its coefficient are the same, we double-count the result. Therefore take it off.
        res -= 1

    return res


class CountNumberInTableSpec(unittest.TestCase):
    def test_target_out_of_boundary(self):
        self.assertEqual(0, count_number_in_table(1, 100))

    def test_target_out_of_boundary2(self):
        self.assertEqual(0, count_number_in_table(2, -100))
    
    def test_target_range_from_N_to_N_Square(self):
        self.assertEqual(0, count_number_in_table(3, 7))
    
    def test_target_range_from_N_to_N_Square2(self):
        self.assertEqual(1, count_number_in_table(3, 4))
    
    def test_target_range_from_N_to_N_Square3(self):
        self.assertEqual(2, count_number_in_table(3, 6))

    def test_target_range_from_Zero_to_N(self):
        self.assertEqual(0, count_number_in_table(4, 0))

    def test_target_range_from_Zero_to_N2(self):
        self.assertEqual(2, count_number_in_table(4, 2))

    def test_target_range_from_Zero_to_N3(self):
        self.assertEqual(2, count_number_in_table(4, 3))
    
    def test_target_range_from_Zero_to_N4(self):
        self.assertEqual(3, count_number_in_table(4, 4))
    
    def test_target_range_from_Zero_to_N5(self):
        self.assertEqual(6, count_number_in_table(12, 12))
    
    def test_target_range_from_Zero_to_N6(self):
        self.assertEqual(3, count_number_in_table(27, 25))
    
    def test_target_range_from_Zero_to_N7(self):
        self.assertEqual(1, count_number_in_table(4, 1))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
 ```

### Feb 2, 2021 \[Easy\] ZigZag Binary Tree
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

**Solution with BFS:** [https://repl.it/@trsong/ZigZag-Order-of-Binary-Tree](https://repl.it/@trsong/ZigZag-Order-of-Binary-Tree)
```py

import unittest
from Queue import deque as Deque

def zigzag_traversal(tree):
    if not tree:
        return []

    queue = Deque()
    queue.append(tree)
    reverse_order = False
    res = []

    while queue:
        if reverse_order:
            res.extend(reversed(queue))
        else:
            res.extend(queue)
        
        reverse_order = not reverse_order
        for _ in xrange(len(queue)):
            cur = queue.popleft()
            for child in [cur.left, cur.right]:
                if not child:
                    continue
                queue.append(child)
    
    return map(lambda node: node.val, res)


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


### Feb 1, 2021 \[Medium\] Maximum Path Sum in Binary Tree
--- 
> **Question:** You are given the root of a binary tree. Find the path between 2 nodes that maximizes the sum of all the nodes in the path, and return the sum. The path does not necessarily need to go through the root.

**Example:**
```py
Given the following binary tree, the result should be 42 = 20 + 2 + 10 + 10.
       *10
       /  \
     *2   *10
     / \     \
   *20  1    -25
             /  \
            3    4
(* denotes the max path)
```

**My thoughts:** The maximum path sum can either roll up from maximum of recursive children value or calculate based on maximum left path sum and right path sum.

Example1: Final result rolls up from children
```py
     0
   /   \
  2     0
 / \   /
4   5 0
```

Example2: Final result is calculated based on max left path sum and right path sum

```py
    1
   / \
  2   3
 /   / \
8   0   5
   / \   \
  0   0   9
```

**Solution:** [https://repl.it/@trsong/Find-Maximum-Path-Sum-in-Binary-Tree](https://repl.it/@trsong/Find-Maximum-Path-Sum-in-Binary-Tree)
```py
import unittest

def max_path_sum(tree):
    return max_path_sum_recur(tree)[0]


def max_path_sum_recur(node):
    if not node:
        return 0, 0
    
    left_max_sum, left_max_path = max_path_sum_recur(node.left)
    right_max_sum, right_max_path = max_path_sum_recur(node.right)
    max_path = node.val + max(left_max_path, right_max_path)
    max_sum = node.val + left_max_path + right_max_path 
    return max(left_max_sum, right_max_sum, max_sum), max_path


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class MaxPathSumSpec(unittest.TestCase):
    def test_example(self):
        """
                10
               /  \
              2    10
             / \     \
            20  1    -25
                     /  \
                    3    4
        """
        n2 = TreeNode(2, TreeNode(20), TreeNode(1))
        n25 = TreeNode(-25, TreeNode(3), TreeNode(4))
        n10 = TreeNode(10, right=n25)
        root = TreeNode(10, n2, n10)
        self.assertEqual(42, max_path_sum(root))  # Path: 20, 2, 10, 10

    def test_empty_tree(self):
        self.assertEqual(0, max_path_sum(None))

    def test_result_rolling_up_from_children(self):
        """
             0
           /   \
          2     0
         / \     \
        4   5     0
        """
        n2 = TreeNode(2, TreeNode(4), TreeNode(5))
        n0 = TreeNode(0, right=TreeNode(0))
        root = TreeNode(0, n2, n0)
        self.assertEqual(11, max_path_sum(root))  # Path: 4 - 2 - 5
    
    def test_result_calculated_based_on_max_left_path_sum_and_right_path_sum(self):
        """
            1
           / \
          2   3
         /   / \
        8   0   5
           / \   \
          0   0   9
        """
        n0 = TreeNode(0, TreeNode(0), TreeNode(0))
        n5 = TreeNode(5, right=TreeNode(9))
        n3 = TreeNode(3, n0, n5)
        n2 = TreeNode(2, TreeNode(8))
        root = TreeNode(1, n2, n3)
        self.assertEqual(28, max_path_sum(root))  # Path: 8 - 2 - 1 - 3 - 5 - 9

    def test_max_path_sum_not_pass_root(self):
        """
             1
           /   \
          2     0
         / \   /
        4   5 1
        """
        n2 = TreeNode(2, TreeNode(4), TreeNode(5))
        n0 = TreeNode(0, TreeNode(1))
        root = TreeNode(1, n2, n0)
        self.assertEqual(11, max_path_sum(root)) # Path: 4 - 2 - 5

    def test_max_path_sum_pass_root(self):
        """
              1
             /
            2
           /
          3
         /
        4
        """
        n3 = TreeNode(3, TreeNode(4))
        n2 = TreeNode(2, n3)
        n1 = TreeNode(1, n2)
        self.assertEqual(10, max_path_sum(n1))  # Path: 1 - 2 - 3 - 4

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
        self.assertEqual(28, max_path_sum(n1))  # Path: 8 - 2 - 1 - 3 - 5 - 9 

    def test_tree_with_negative_nodes(self):
        """
            -1
           /  \
         -2   -3
         /    / \
       -8   -4  -5
            / \   \
          -6  -7  -9
        """
        n5 = TreeNode(-5, right=TreeNode(-9))
        n4 = TreeNode(-4, TreeNode(-6), TreeNode(-7))
        n3 = TreeNode(-3, n4, n5)
        n2 = TreeNode(-2, TreeNode(-8))
        n1 = TreeNode(-1, n2, n3)
        self.assertEqual(0, max_path_sum(n1))  # Path: 8 - 2 - 1 - 3 - 5 - 9 


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```