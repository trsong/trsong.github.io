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

### May 5, 2019 \[Hard\] Maximum Spanning Tree
--- 
> **Question:** Recall that the minimum spanning tree is the subset of edges of a tree that connect all its vertices with the smallest possible total edge weight. 
> 
> Given an undirected graph with weighted edges, compute the maximum weight spanning tree.


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