---
layout: post
title:  "Daily Coding Problems Aug to Oct"
date:   2019-08-01 22:22:32 -0700
categories: Python/Java
---
* This will become a table of contents (this text will be scraped).
{:toc}

## Daily Coding Problems

### Enviroment Setup
---

**Python 2.7 Playground:** [https://repl.it/languages/python](https://repl.it/languages/python)


<!-- 

### Sep 11, 2019 \[Hard\] Longest Path in the Tree
---
> **Question:**  Given a tree where each edge has a weight, compute the length of the longest path in the tree.

**Example:**

```py
Given the following tree:

   a
  /|\
 b c d
    / \
   e   f
  / \
 g   h
and the weights: a-b: 3, a-c: 5, a-d: 8, d-e: 2, d-f: 4, e-g: 1, e-h: 1, the longest path would be c -> a -> d -> f, with a length of 17.

The path does not have to pass through the root, and each node can have any amount of children.
``` 

--->

### Oct 16, 2019 \[Medium\] Count Arithmetic Subsequences
---
> **Question:** Given an array of n positive integers. The task is to count the number of Arithmetic Subsequence in the array. Note: Empty sequence or single element sequence is also Arithmetic Sequence. 

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

**Similar Question:** LC 413 Arithmetic Slices

### Oct 15, 2019 \[Medium\] Insert into Sorted Circular Linked List
---
> **Question:** Insert a new value into a sorted circular linked list (last element points to head). Return the node with smallest value.  

**My thoughts:** This question isn't hard. It's just it has so many edge case we need to cover:

* original list is empty
* original list contains duplicate numbers
* the first element of original list is not the smallest
* the insert elem is the smallest
* the insert elem is the largest
* etc.

**Solution:** [https://repl.it/@trsong/Insert-into-Sorted-Circular-Linked-List](https://repl.it/@trsong/Insert-into-Sorted-Circular-Linked-List)
```py
import unittest

def insert(lst, val):
    target_node = Node(val)
    if lst is None:
        target_node.next = target_node
        return target_node
    
    root = lst
    while root.next != lst and root.val <= root.next.val:
        root = root.next

    # root is max, root.next is min
    if val < root.next.val:
        # the inserted node is the new min
        target_node.next = root.next
        root.next = target_node
        return target_node
    elif val > root.val:
        # the inserted node is the new max
        target_node.next = root.next
        root.next = target_node
        return target_node.next

    last = root
    # proceed max, now root is the min
    root = root.next
    lst = root

    while lst != last and lst.next.val < val:
        lst = lst.next
  
    target_node.next = lst.next
    lst.next = target_node
    return root
    

##############################
# Below are testing utilities
##############################
class Node(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

 
    def __eq__(self, other):
        l1 = Node.flatten(self)
        l2 = Node.flatten(other)
        if l1 != l2:
            print str(l1), '!=', str(l2)
        return l1 == l2

    def __str__(self):
        return str(Node.flatten(self))

    @staticmethod
    def flatten(lst):
        if not lst:
            return []
        root = lst
        res = [root.val]
        lst = lst.next
        while lst != root:
            res.append(lst.val)
            lst = lst.next
        return res

    @staticmethod
    def create(*vals):
        dummy = Node(-1)
        t = dummy
        for v in vals:
            t.next = Node(v)
            t = t.next
        t.next = dummy.next
        return dummy.next

class InsertSpec(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(Node.create(1), insert(None, 1))

    def test_prepend_list(self):
        self.assertEqual(Node.create(0, 1), insert(Node.create(1), 0))

    def test_append_list(self):
        self.assertEqual(Node.create(1, 2, 3), insert(Node.create(1, 2), 3))

    def test_insert_into_correct_position(self):
        self.assertEqual(Node.create(1, 2, 3, 4, 5), insert(Node.create(1, 2, 4, 5), 3))

    def test_duplicated_elements(self):
        self.assertEqual(Node.create(0, 0, 1, 2), insert(Node.create(0, 0, 2), 1))
    
    def test_duplicated_elements2(self):
        self.assertEqual(Node.create(0, 0, 1), insert(Node.create(0, 0), 1))

    def test_duplicated_elements3(self):
        self.assertEqual(Node.create(0, 0, 0, 0), insert(Node.create(0, 0, 0), 0))

    def test_first_element_is_not_smallest(self):
        self.assertEqual(Node.create(0, 1, 2, 3), insert(Node.create(2, 3, 0), 1))

    def test_first_element_is_not_smallest2(self):
        self.assertEqual(Node.create(0, 1, 2, 3), insert(Node.create(3, 0, 2), 1))

    def test_first_element_is_not_smallest3(self):
        self.assertEqual(Node.create(0, 1, 2, 3), insert(Node.create(2, 0, 1), 3))

    def test_first_element_is_not_smallest4(self):
        self.assertEqual(Node.create(0, 1, 2, 3), insert(Node.create(2, 3, 1), 0))

    def test_first_element_is_not_smallest5(self):
        self.assertEqual(Node.create(0, 0, 1, 2, 2), insert(Node.create(2, 0, 0, 2), 1))

    def test_first_element_is_not_smallest6(self):
        self.assertEqual(Node.create(-1, 0, 0, 2, 2), insert(Node.create(2, 0, 0, 2), -1))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 14, 2019 \[Easy\] Word Ordering in a Different Alphabetical Order
---
> **Question:** Given a list of words, and an arbitrary alphabetical order, verify that the words are in order of the alphabetical order.

**Example:**
```py
Input:
words = ["abcd", "efgh"],
order="zyxwvutsrqponmlkjihgfedcba"

Output: False
Explanation: 'e' comes before 'a' so 'efgh' should come before 'abcd'
```

**Example 2:**
```py
Input:
words = ["zyx", "zyxw", "zyxwy"],
order="zyxwvutsrqponmlkjihgfedcba"

Output: True
Explanation: The words are in increasing alphabetical order
```

**Solution:** [https://repl.it/@trsong/Word-Ordering-in-a-Different-Alphabetical-Order](https://repl.it/@trsong/Word-Ordering-in-a-Different-Alphabetical-Order)
```py
import unittest


def is_sorted(words, order):
    if not words or len(words) <= 1:
        return True

    score = dict(zip(order, xrange(len(order))))
    for i in xrange(1, len(words)):
        cur_word = words[i]
        prev_word = words[i-1]
        is_short_circuit = False
        for prev_char, cur_char in zip(prev_word, cur_word):
            if score[prev_char] < score[cur_char]:
                is_short_circuit = True
                break
            if score[prev_char] > score[cur_char]:
                return False
        if not is_short_circuit and len(prev_word) > len(cur_word):
            return False
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

### Oct 13, 2019 LC 525 \[Medium\] Largest Subarray with Equal Number of 0s and 1s
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

**Solution:** [https://repl.it/@trsong/Largest-Subarray-with-Equal-Number-of-0s-and-1s](https://repl.it/@trsong/Largest-Subarray-with-Equal-Number-of-0s-and-1s)
```py
import unittest

def largest_even_subarray(nums):
    if not nums:
        return None

    count_map = {0:-1}
    count = 0
    max_length = 0
    max_start_location = None
    for i, num in enumerate(nums):
        if num == 1:
            count += 1
        else:
            count -= 1

        if count not in count_map:
            count_map[count] = i
        elif i - count_map[count] > max_length:
            max_length = i - count_map[count]
            max_start_location = count_map[count]

    if max_start_location is not None:
        return max_start_location + 1, max_start_location + max_length
    else:
        None


class LargestEvenSubarraySpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual((1, 6), largest_even_subarray([1, 0, 1, 1, 1, 0, 0]))

    def test_example2(self):
        self.assertTrue(largest_even_subarray([0, 0, 1, 1, 0]) in [(0, 3), (1, 4)])

    def test_entire_array_is_even(self):
        self.assertEqual((0, 1), largest_even_subarray([0, 1]))

    def test_no_even_subarray(self):
        self.assertIsNone(largest_even_subarray([0, 0, 0, 0, 0]))
        self.assertIsNone(largest_even_subarray([1]))
        self.assertIsNone(largest_even_subarray([]))

    def test_larger_array(self):
        self.assertEqual((0, 9), largest_even_subarray([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1]))

    def test_larger_array2(self):
        self.assertEqual((3, 8), largest_even_subarray([1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1]))

    def test_larger_array3(self):
        self.assertEqual((0, 13), largest_even_subarray([1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1])) 


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 12, 2019 \[Easy\] Most Frequent Subtree Sum
---
> **Question:** Given the root of a binary tree, find the most frequent subtree sum. The subtree sum of a node is the sum of all values under a node, including the node itself.

**Example:**
```py
Given the following tree:

  5
 / \
2  -5

Return 2 as it occurs twice: once as the left leaf, and once as the sum of 2 + 5 - 5.
```

**Solution:** [https://repl.it/@trsong/Most-Frequent-Subtree-Sum](https://repl.it/@trsong/Most-Frequent-Subtree-Sum)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def calculate_tree_sum_recur(t, tree_sum_histogram):
    if t is None:
        return 0
    tree_sum = t.val + calculate_tree_sum_recur(t.left, tree_sum_histogram) + calculate_tree_sum_recur(t.right, tree_sum_histogram)
    tree_sum_histogram[tree_sum] = tree_sum_histogram.get(tree_sum, 0) + 1
    return tree_sum


def most_freq_tree_sum(tree):
    if tree is None:
        return None

    tree_sum_histogram = {}
    calculate_tree_sum_recur(tree, tree_sum_histogram)
    max_sum_count = 0
    max_sum = None
    for tree_sum, freq in tree_sum_histogram.items():
        if freq > max_sum_count:
            max_sum = tree_sum
            max_sum_count = freq

    return max_sum


class MostFreqTreeSumSpec(unittest.TestCase):
    def test_example1(self):
        """
           5
          / \
         2  -5
        """
        t = TreeNode(5, TreeNode(2), TreeNode(-5))
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
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 11, 2019 \[Hard\] Largest Divisible Pairs Subset
---
> **Question:** Given a set of distinct positive integers, find the largest subset such that every pair of elements in the subset `(i, j)` satisfies either `i % j = 0` or `j % i = 0`.
>
> For example, given the set `[3, 5, 10, 20, 21]`, you should return `[5, 10, 20]`. Given `[1, 3, 6, 24]`, return `[1, 3, 6, 24]`.


**My thoughts:** Notice that smaller number mod large number can never be zero, but the other way around might be True. e.g.  `2 % 4 == 2`.  `9 % 3 == 0`. Thus we can eliminate the condition `i % j = 0 or j % i = 0` to just `j % i = 0 where j is larger`. This can be done by sorting into descending order.

After that we can use dp to solve this problem. Define `dp[i]` to be the largest number of divisible pairs ends at `i`. And `max(dp)` will give the largest number of division pairs among all i. By backtracking from the index, we can find what are these numbers.

**Solution with DP:** [https://repl.it/@trsong/Largest-Divisible-Pairs-Subset](https://repl.it/@trsong/Largest-Divisible-Pairs-Subset)
```py
import unittest

def largest_divisible_pairs_subset(nums):
    n = len(nums)
    if n < 2:
        return []
    descending_nums = sorted(nums, reverse=True)

    # Let dp[i] be the largest number of divisible pairs ends at i
    dp = [0] * n
    dp[0] = 1
    parent = [None] * n
    for i in xrange(1, n):
        max_num_pairs = 0
        cur_num = descending_nums[i]
        for j in xrange(i):
            # Check among all multiple of cur_num, find max number of pairs among them
            if descending_nums[j] % cur_num == 0 and dp[j] > max_num_pairs:
                max_num_pairs = dp[j]
                parent[i] = j
        dp[i] = max_num_pairs + 1
    
    # Backtracking from the start position of largest number of divisible pairs
    i = dp.index(max(dp))
    res = []
    while i is not None:
        res.append(descending_nums[i])
        i = parent[i]
    return res if len(res) > 1 else []

class LargestDivisiblePairsSubsetSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(set([5, 10, 20]), set(largest_divisible_pairs_subset([3, 5, 10, 20, 21])))

    def test_example2(self):
        self.assertEqual(set([1, 3, 6, 24]), set(largest_divisible_pairs_subset([1, 3, 6, 24])))

    def test_multiple_of_3_and_5(self):
        self.assertEqual(set([10, 5, 20]), set(largest_divisible_pairs_subset([10, 5, 3, 15, 20])))

    def test_prime_and_multiple_of_3(self):
        self.assertEqual(set([18, 1, 3, 6]), set(largest_divisible_pairs_subset([18, 1, 3, 6, 13, 17])))

    def test_decrease_array(self):
        self.assertEqual(set([8, 4, 2, 1]), set(largest_divisible_pairs_subset([8, 7, 6, 5, 4, 2, 1])))

    def test_array_with_duplicated_values(self):
        self.assertEqual(set([3, 3, 3, 1]), set(largest_divisible_pairs_subset([2, 2, 3, 3, 3, 1])))

    def test_no_divisible_pairs(self):
        self.assertEqual([], largest_divisible_pairs_subset([2, 3, 5, 7, 11, 13, 17, 19]))

    def test_no_divisible_pairs2(self):
        self.assertEqual([], largest_divisible_pairs_subset([1]))
        self.assertEqual([], largest_divisible_pairs_subset([]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 10, 2019 LC 134 \[Medium\] Gas Station
---
> **Question:** There are `N` gas stations along a circular route, where the amount of gas at station `i` is `gas[i]`.
> 
> You have a car with an unlimited gas tank and it costs `cost[i]` of gas to travel from station `i` to its next station `(i+1)`. You begin the journey with an empty tank at one of the gas stations.
>
> Return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return `-1`.
>
> **Note:**
> - If there exists a solution, it is guaranteed to be unique.
> - Both input arrays are non-empty and have the same length.
> - Each element in the input arrays is a non-negative integer.

**Example 1:**
```py
Input: 
gas  = [1,2,3,4,5]
cost = [3,4,5,1,2]

Output: 3

Explanation:
Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 4. Your tank = 4 - 1 + 5 = 8
Travel to station 0. Your tank = 8 - 2 + 1 = 7
Travel to station 1. Your tank = 7 - 3 + 2 = 6
Travel to station 2. Your tank = 6 - 4 + 3 = 5
Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
Therefore, return 3 as the starting index.
```

**Example 2:**
```py
Input: 
gas  = [2,3,4]
cost = [3,4,3]

Output: -1

Explanation:
You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 0. Your tank = 4 - 3 + 2 = 3
Travel to station 1. Your tank = 3 - 3 + 3 = 3
You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.
Therefore, you can't travel around the circuit once no matter where you start.
```

**My thougths:** If there exists such a gas station at index `i` then for all stop we will have `gas_level >= 0` when we reach `k` for `all k`. However if starts `i`, `i+1`, ..., `k`  and at `k`,  `gas_level < 0`. Then instead of starting again from `i+1`, we starts from `k+1` as without `i` sum from `i+1` to `k` can only go lower. 

**Solution:** [https://repl.it/@trsong/Gas-Station](https://repl.it/@trsong/Gas-Station)
```py
import unittest

def valid_starting_station(gas, cost):
    gas_level = 0
    # gross_net_gas_gain is the sum of all gas gain or loss from index 0
    # actually it is also the total gas gain or less when starts from any indices = all gas gain - all gas loss. 
    gross_net_gas_gain = 0
    i = 0
    for increase, decrease, k in zip(gas, cost, xrange(len(gas))):
        net_gas_gain = increase - decrease
        gross_net_gas_gain += net_gas_gain
        gas_level += net_gas_gain
        if gas_level < 0:
            # if sum of net gas gain from last i all the way to k < 0
            # instead of start a new i form i + 1, we jump to k + 1
            # Even with gas gain at i i, sum to k < 0, then without gas gain at i, it will be even smaller. The same case happens for all index from i to k, so we jump to k + 1
            gas_level = 0
            i = (k + 1) % len(gas)
    return i if gross_net_gas_gain >= 0 else -1


class ValidStartingStationSpec(unittest.TestCase):
    def assert_valid_starting_station(self, gas, cost, start):
        gas_level = 0
        n = len(gas)
        for i in range(n):
            shifted_index = (start + i) % n
            gas_level += gas[shifted_index] - cost[shifted_index]
            self.assertTrue(gas_level >= 0)

    def test_example1(self):
        gas  = [1, 2, 3, 4, 5]
        cost = [3, 4, 5, 1, 2]
        start = valid_starting_station(gas, cost)
        self.assert_valid_starting_station(gas, cost, start)

    def test_example2(self):
        gas  = [2, 3, 4]
        cost = [3, 4, 3]
        self.assertEqual(-1, valid_starting_station(gas, cost))

    def test_decreasing_gas_level(self):
        gas = [1, 1, 1]
        cost = [2, 2, 2]
        self.assertEqual(-1, valid_starting_station(gas, cost))

    def test_increasing_gas_level(self):
        gas = [2, 1, 0]
        cost = [1, 1, 0]
        start = valid_starting_station(gas, cost)
        self.assert_valid_starting_station(gas, cost, start)

    def test_decreasing_increasing_decreasing_increasing(self):
        gas = [0, 1, 0, 13]
        cost = [3, 0, 10, 1]
        start = valid_starting_station(gas, cost)
        print start
        self.assert_valid_starting_station(gas, cost, start)

    def test_total_gas_level_decrease(self):
        gas = [3, 2, 1, 0, 0]
        cost = [2, 3, 1, 1, 0]
        self.assertEqual(-1, valid_starting_station(gas, cost))

    def test_total_gas_level_increase(self):
        gas = [1, 1, 0, 2]
        cost = [0, 0, -3, 0]
        start = valid_starting_station(gas, cost)
        self.assert_valid_starting_station(gas, cost, start)


if __name__ == '__main__':
    unittest.main(exit=False)
```
### Oct 9, 2019 \[Hard\] Number of Ways to Divide an Array into K Equal Sum Sub-arrays
---
> **Question:** Given an integer K and an array arr[] of N integers, the task is to find the number of ways to split the array into K equal sum sub-arrays of non-zero lengths.

**Example 1:**

```py
Input: arr[] = [0, 0, 0, 0], K = 3
Output: 3
All possible ways are:
[[0], [0], [0, 0]]
[[0], [0, 0], [0]]
[[0, 0], [0], [0]]
```

**Example 2:**
```py
Input: arr[] = [1, -1, 1, -1], K = 2
Output: 1
```

**My thoughts:** This problem is so hard to think and even harder to deal with bugs related to index. I ended up w/ wasting lots of time figuring out the correct index. 

The idea is to first figure out what is the target subarray sum. This can be achieved by sum of all element divide by K. But what if K is zero? So we need special handling for that.

After we figure out the target subarray sum, it's still hard to come up w/ dp solution because one of the coordinate acctually represents spot to insert splitter. 

Imagining the following: in order to break an array in to K parts, we can have K - 1 splitters fit into spots between each element. Like `0 | 0 0 | 0`. And we can pre-compute the prefix_sum at index i. Therefore we can quickly tell if the i-th spot can put a split or not. e.g. `0 | 1 0 1` cannot put split here as the prefix_sum does not satisfy the target subarray sum.

One dp solution is to let `dp[i][k]` represents problem size i and k remaining spliters. We know in the end `dp[n][k+1]` represents array size `n` and `k+1` spliter will be the goal. So we compute backward and figure out `dp[0][1]` as the prefix_sum for index 0 is 0 and always qualify target subarray sum so we put 1 splitter there.

The dp recursive relation is like the following: `dp[i][k] = dp[i+1][k]` when we cannot split at i due to the `prefix_sum[i]` not qualified. Or  `dp[i][k] = dp[i-1][k] + dp[i+1][k+1]` if `prefix_sum[i]` qualified.

 Qualified means if we split at i, then the prefix_sum[i] can be broken into remaining k subarrays and each of them has target subarray sum. e.g.  `0 1 0 1 | 0 1` If K == 3 and we know that `0 1 0 1` sum to 2 and is `(k-1) * target_sub_array_sum = 2` which means it can be break into k-1 subarray with suitable subarray sum. And we can do that again for `0 1 | 0 1`. 


**Solution with DP:** [https://repl.it/@trsong/Number-of-Ways-to-Divide-an-Array-into-K-Equal-Sum-Sub-array](https://repl.it/@trsong/Number-of-Ways-to-Divide-an-Array-into-K-Equal-Sum-Sub-array)
```py
import unittest

def num_k_subarray(nums, K):
    if K <= 0 or not nums or K > len(nums):
        return 0
    nums_sum = sum(nums)
    if nums_sum % K > 0:
        return 0
    n = len(nums)
    subarray_sum = nums_sum // K
    prefix_sum = nums[:]
    for i in xrange(1, n):
        prefix_sum[i] += prefix_sum[i-1]
    
    # Let dp[i][k] represents number of qualified subarray for remaining array size i and remaining splitter k
    # Remember in order to break into K parts, we will need K+1 splitters
    # dp[i][k] = dp[i+1][k]                if we can cannot split at i as prefix_sum not qualified 
    #          = dp[i+1][k] + dp[i+1][k+1] if we can split as the prefix_sum[i] is qualified
    dp = [[0 for _ in xrange(K+2)] for _ in xrange(n+1)]
    dp[n][K+1] = 1
    for i in xrange(n-1, -1, -1):
        for k in xrange(K, 0, -1):
            dp[i][k] = dp[i+1][k]
            if subarray_sum == 0 and prefix_sum[i] == 0 or subarray_sum != 0 and prefix_sum[i] == subarray_sum * k:
                dp[i][k] += dp[i+1][k+1]
    
    # If remaining array has size 0 and there is one splitter
    return dp[0][1]


class NumKSubarraySpec(unittest.TestCase):
    def test_example1(self):
        """
        0 0|0|0
        0|0 0|0
        0|0|0 0
        """
        self.assertEqual(3, num_k_subarray([0, 0, 0, 0], 3))

    def test_exampl2(self):
        """
        1 -1|1 -1
        """
        self.assertEqual(1, num_k_subarray([1, -1, 1, -1], 2))

    def test_contains_negative_numbers(self):
        """
        1 1|-2 2 1 1
        1 1 -2 2|1 1
        """
        self.assertEqual(2, num_k_subarray([1, 1, -2, 2, 1, 1], 2))

    def test_array_with_unique_number(self):
        """
        1|1|1|1|1|1
        """
        self.assertEqual(1, num_k_subarray([1, 1, 1, 1, 1, 1], 6))

    def test_K_equals_4(self):
        """
        0 0 1|0 1 0|0 0 1|0 1 0
        0 0 1 0|1 0|0 0 1|0 1 0
        0 0 1 0|1|0 0 0 1|0 1 0
        0 0 1 0|1 0 0|0 1|0 1 0
        0 0 1 0|1 0 0 0|1|0 1 0
        ...
        0 0 1|0 1 0|0 0 1|0 1 0
        Subarray Sum: 4 / 4 = 1
        Number of ways to place Splitters: 2 * 4 * 2 = 16
        """
        self.assertEqual(16, num_k_subarray([0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0], 4))

    def test_K_equals_0(self):
        self.assertEqual(0, num_k_subarray([1, 2], 0))
        self.assertEqual(0, num_k_subarray([], 0))

    def test_empty_array(self):
        self.assertEqual(0, num_k_subarray([], 3))

    def test_K_too_big(self):
        self.assertEqual(0, num_k_subarray([1, 1, 1], 4))

    def test_K_equals_1(self):
        self.assertEqual(0, num_k_subarray([], 1)) # subarray must be non-zero length
        self.assertEqual(1, num_k_subarray([1, 2, 3], 1))

    def test_sum_not_multiple_or_K(self):
        self.assertEqual(0, num_k_subarray([1, 1, 1], 2))

        
if __name__ == '__main__':
    unittest.main(exit=False)
```
### Oct 8, 2019 \[Easy\] Count Number of Unival Subtrees
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

**Solution:** [https://repl.it/@trsong/Count-Number-of-Unival-Subtrees](https://repl.it/@trsong/Count-Number-of-Unival-Subtrees)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def count_subtree_helper(tree):
    if not tree:
        return 0, True
    left_count, is_left_unival = count_subtree_helper(tree.left)
    right_count, is_right_unival = count_subtree_helper(tree.right)
    is_current_unival = is_left_unival and is_right_unival
    if is_current_unival and tree.left and tree.left.val != tree.val:
        is_current_unival = False
    if is_current_unival and tree.right and tree.right.val != tree.val:
        is_current_unival = False
    count = left_count + right_count + (1 if is_current_unival else 0)
    return count, is_current_unival


def count_unival_subtrees(tree):
    return count_subtree_helper(tree)[0]


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

### Oct 7, 2019 \[Easy\] One-to-one Character Mapping
---
> **Question:** Determine whether there exists a one-to-one character mapping from one string s1 to another s2.
>
> For example, given s1 = 'abc' and s2 = 'bcd', return True since we can map a to b, b to c, and c to d.
>
> Given s1 = 'foo' and s2 = 'bar', return False since the o cannot map to two characters.

**Solution:** [https://repl.it/@trsong/One-to-one-Character-Mapping](https://repl.it/@trsong/One-to-one-Character-Mapping)
```py
import unittest

CHAR_SPACE_SIZE = 256

def is_one_to_one_mapping(s1, s2):
    n1, n2 = len(s1), len(s2)

    if n1 != n2:
        return False

    char_mapping = [None] * CHAR_SPACE_SIZE
    for c1, c2 in zip(s1, s2):
        ord_c1, ord_c2 = ord(c1), ord(c2)
        if char_mapping[ord_c1] is None:
            char_mapping[ord_c1] = ord_c2
        elif char_mapping[ord_c1] != ord_c2:
            return False

    return True


class IsOneToOneMappingSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(is_one_to_one_mapping('abc', 'bcd'))

    def test_example2(self):
        self.assertFalse(is_one_to_one_mapping('foo', 'bar'))

    def test_empty_mapping(self):
        self.assertTrue(is_one_to_one_mapping('', ''))
        self.assertFalse(is_one_to_one_mapping('', ' '))
        self.assertFalse(is_one_to_one_mapping(' ', ''))

    def test_map_strings_with_different_lengths(self):
        self.assertFalse(is_one_to_one_mapping('abc', 'abcd'))
        self.assertFalse(is_one_to_one_mapping('abcd', 'abc'))
    
    def test_duplicated_chars(self):
        self.assertTrue(is_one_to_one_mapping('aabbcc', '112233'))
        self.assertTrue(is_one_to_one_mapping('abccba', '123321'))

    def test_same_domain(self):
        self.assertTrue(is_one_to_one_mapping('abca', 'bcab'))
        self.assertFalse(is_one_to_one_mapping('abca', 'bcaa'))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 6, 2019 \[Medium\] Number of Smaller Elements to the Right
---
> **Question:** Given an array of integers, return a new array where each element in the new array is the number of smaller elements to the right of that element in the original input array.
>
> For example, given the array `[3, 4, 9, 6, 1]`, return `[1, 1, 2, 1, 0]`, since:
>
> * There is 1 smaller element to the right of 3
> * There is 1 smaller element to the right of 4
> * There are 2 smaller elements to the right of 9
> * There is 1 smaller element to the right of 6
> * There are no smaller elements to the right of 1

**My thoughts:** Binary-indexed Tree a.k.a BIT or Fenwick Tree is used for dynamic range query. Basically, it allows efficiently query for sum of a continous interval of values. 

**BIT Usage:**
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

So the way we can solve this problem is to treat each number in array as an index. By BIT querying for sum, we can quickly count how many numbers are less than current number.


**Solution with BIT:** [https://repl.it/@trsong/Number-of-Smaller-Elements-to-the-Right](https://repl.it/@trsong/Number-of-Smaller-Elements-to-the-Right)
```py
import unittest

class BIT(object):
    @staticmethod
    def LSB(num):
        return num & -num

    def __init__(self, n):
        self.arr = [0] * (n+1)

    def query(self, k):
        k = k + 1
        sum = 0 
        while k > 0:
            sum += self.arr[k]
            k -= BIT.LSB(k)
        return sum

    def increment(self, i):
        i = i + 1
        while i < len(self.arr):
            self.arr[i] += 1
            i += BIT.LSB(i)


def count_right_smaller_numbers(nums):
    if not nums:
        return []
    max_val = max(nums)
    min_val = min(nums)
    bit = BIT(max_val - min_val + 1)
    n = len(nums)
    nums_count = [0] * n
    for i in xrange(n-1, -1, -1):
        shifted_value = nums[i] - min_val
        # query for number of elements smaller than nums[i]
        nums_count[i] = bit.query(shifted_value - 1)
        bit.increment(shifted_value)
    return nums_count 
 

class CountRightSmallerNuberSpec(unittest.TestCase):
    def test_example(self):
        nums = [3, 4, 9, 6, 1]
        expected_result = [1, 1, 2, 1, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))

    def test_empty_array(self):
        self.assertEqual([], count_right_smaller_numbers([]))

    def test_ascending_array(self):
        nums = [0, 1, 2, 3, 4]
        expected_result = [0, 0, 0, 0, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))

    def test_array_with_unique_value(self):
        nums = [1, 1, 1]
        expected_result = [0, 0, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))

    def test_descending_array(self):
        nums = [4, 3, 2, 1]
        expected_result = [3, 2, 1, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))

    def test_increase_decrease_increase(self):
        nums = [1, 4, 2, 5, 0, 4]
        expected_result = [1, 2, 1, 2, 0, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))

    def test_decrease_increase_decrease(self):
        nums = [3, 1, 2, 0]
        expected_result = [3, 1, 1, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))

    def test_decrease_increase_decrease2(self):
        nums = [12, 1, 2, 3, 0, 11, 4]
        expected_result = [6, 1, 1, 1, 0, 1, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))

    def test_negative_values(self):
        nums = [8, -2, -1, -2, -1, 3]
        expected_result = [5, 0, 1, 0, 0, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Oct 5, 2019 LC 375 \[Medium\] Guess Number Higher or Lower II
---
> **Question:**  We are playing the Guess Game. The game is as follows:
>
> * I pick a number from 1 to n. You have to guess which number I picked.
> * Every time you guess wrong, I'll tell you whether the number I picked is higher or lower.
> * However, when you guess a particular number x, and you guess wrong, you pay $x. You win the game when you guess the number I picked.
>
> Given a particular n ≥ 1, find out how much money you need to have to guarantee a win.

**Example:**

```
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

**Solution with Minimax Algorithm:** [https://repl.it/@trsong/Guess-Number-Game-2](https://repl.it/@trsong/Guess-Number-Game-2)
```py
import unittest

def guess_number_between(lo, hi, cache):
    if lo >= hi:
        return 0
    elif cache[lo][hi] is not None:
        return cache[lo][hi]
    
    res = float('-inf')
    for i in xrange(lo, hi+1):
        # Assuming we are just so unlucky.
        # The guarantee amount should always be the best among those worest choices
        res = max(res, -i + min(guess_number_between(lo, i-1, cache), guess_number_between(i+1, hi, cache)))
    cache[lo][hi] = res
    return res
    

def guess_number(n):
    cache = [[None for _ in range(n+1)] for _ in range(n+1)]
    return -guess_number_between(1, n, cache)


class GuessNumberSpec(unittest.TestCase):
    def test_n_equals_3(self):
        # Worest case target=3
        # choose 2, target is higher, pay $2
        # total = $2
        self.assertEqual(2, guess_number(3)) # choose 2, pay $2

    def test_n_equals_4(self):
        # Worest case target=4
        # choose 1, target is higher, pay $1
        # choose 3, target is higher, pay $3
        # total = $4
        self.assertEqual(4, guess_number(4)) 

    def test_n_equals_5(self):
        # Worest case target=5
        # choose 2, target is higher, pay $2
        # choose 4, target is higher, pay $4
        # total = $6
        self.assertEqual(6, guess_number(5))

    def test_n_equals_10(self):
        # Worest case target=10
        # choose 7, target is higher, pay $7
        # choose 9, target is higher, pay $9
        # total = $16
        self.assertEqual(16, guess_number(10))


if __name__ == '__main__':
    unittest.main(exit=False)
```

**Note:** Minimax can be further optimized through ***Alpha-Beta Pruning***: a technique to eliminate choices that cannot make current situation better. 

**Solution with Minimax Algorithm with Alpha-Beta Pruning:** [https://repl.it/@trsong/Guess-Number-Game-2-with-Alpha-Beta-Pruning](https://repl.it/@trsong/Guess-Number-Game-2-with-Alpha-Beta-Pruning)
```py
def best_max_move(lo, hi, cache, alpha, beta):
    if lo >= hi:
        return 0
    elif cache[lo][hi] is not None:
        return cache[lo][hi]

    res = float('-inf')
    for i in xrange(lo, hi+1):
        res = max(res, best_min_move(lo, hi, i, cache, alpha, beta))
        if beta <= res:
            break
        alpha = max(alpha, res)    
    cache[lo][hi] = res
    return res

def best_min_move(lo, hi, i, cache, alpha, beta):
    res = -i + best_max_move(lo, i-1, cache, alpha, beta)
    if res <= alpha:
        return res
    res = min(res, -i + best_max_move(i+1, hi, cache, alpha, beta))
    return res

def guess_number(n):
    cache = [[None for _ in range(n+1)] for _ in range(n+1)]
    return -best_max_move(1, n, cache, float('-inf'), float('inf'))

```

### Oct 4, 2019 \[Medium\] Maximum Circular Subarray Sum
---
> **Question:** Given a circular array, compute its maximum subarray sum in O(n) time. A subarray can be empty, and in this case the sum is 0.

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

**My thoughts:** The max circular subarray sum can be divied into sub-problems: max non-circular subarray sum and max circular-only subarray sum. 

For max non-circular subarray sum problem, we can use `dp[i]` to represent max subarray sum end at index `i` and `max(dp)` will be the answer.

For max circular-only subarray sum problem, we want to find `i`, `j` where `i < j` such that `nums[0] + nums[1] + ... + nums[i] + nums[j] + .... + nums[n-1]` reaches maximum. The way we can handle it is to calculate prefix sum and suffix sum array and find max accumulated sum on the left and on the right. The max circular-only subarray sum equals the sum of those two accumulated sum. 

Finally, the answer to the original problem is the larger one between answers to above two sub-problems. And one thing worth to notice is that if all elements are negative, then the answer should be `0`.

**Solution with DP and Prefix Sum:** [https://repl.it/@trsong/Maximum-Circular-Subarray-Sum](https://repl.it/@trsong/Maximum-Circular-Subarray-Sum)
```py
import unittest

def max_subarray_sum_helper(nums):
    n = len(nums)
    # let dp[i] represents max subarray sum ends at index i - 1
    dp = [0] * (n + 1)
    for i in xrange(1, n+1):
        dp[i] = max(dp[i-1] + nums[i-1], nums[i-1])
    return max(dp)


def max_circular_subarray_sum_helper(nums):
    n = len(nums)
    left_accu_sum = 0
    max_left_accu_sum = float('-inf')
    left_max_sum = [0] * n

    right_accu_sum = 0
    max_right_accu_sum = float('-inf')
    right_max_sum = [0] * n

    for i in xrange(n):
        left_accu_sum += nums[i]
        max_left_accu_sum = max(max_left_accu_sum, left_accu_sum)
        left_max_sum[i] = max_left_accu_sum

    for i in xrange(n-1, -1, -1):
        right_accu_sum += nums[i]
        max_right_accu_sum = max(max_right_accu_sum, right_accu_sum)
        right_max_sum[i] = max_right_accu_sum

    max_sum = 0
    for i in xrange(n):
        l_sum = left_max_sum[i-1] if i > 0 else 0
        r_sum = right_max_sum[i+1] if i < n - 1 else 0
        max_sum = max(max_sum, l_sum + r_sum)

    return max_sum


def max_circular_sum(nums):
    if not nums:
        return 0
    return max(0, max_subarray_sum_helper(nums), max_circular_subarray_sum_helper(nums))


class MaxCircularSumSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(15, max_circular_sum([8, -1, 3, 4]))  # 3 + 4 + 8
    
    def test_example2(self):
        self.assertEqual(6, max_circular_sum([-4, 5, 1, 0]))  # 5 + 1

    def test_empty_array(self):
        self.assertEqual(0, max_circular_sum([]))

    def test_negative_array(self):
        self.assertEqual(0, max_circular_sum([-1, -2, -3]))

    def test_circular_array1(self):
        self.assertEqual(22, max_circular_sum([8, -8, 9, -9, 10, -11, 12]))  # 12 + 8 - 8 + 9 - 9 + 10

    def test_circular_array2(self):
        self.assertEqual(23, max_circular_sum([10, -3, -4, 7, 6, 5, -4, -1]))  # 7 + 6 + 5 - 4 -1 + 10

    def test_circular_array3(self):
        self.assertEqual(52, max_circular_sum([-1, 40, -14, 7, 6, 5, -4, -1]))  # 7 + 6 + 5 - 4 - 1 - 1 + 40

    def test_all_positive_array(self):
        self.assertEqual(10, max_circular_sum([1, 2, 3, 4]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 3, 2019 \[Easy\] Shift-Equivalent Strings
---
> **Question:** Given two strings A and B, return whether or not A can be shifted some number of times to get B.
>
> For example, if A is `'abcde'` and B is `'cdeab'`, return `True`. If A is `'abc'` and B is `'acb'`, return `False`.

**My thoughts:** It should be pretty easy to come up with non-linear time complexity solution. But for linear, I can only come up w/ rolling hash solution. The idea is to treat each digit as a number. For example, `"1234"` is really `1234`, each time we move the most significant bit to right by `(1234 - 1 * 10^3) * 10 + 1 = 2341`. In general, we can treat `'abc'` as numeric value of `abc` base `p0` ie. `a * p0^2 + b * p0^1 + c * p0^0` and in order to prevent overflow, we use a larger prime number which I personally prefer 666667 (easy to remember), `'abc' =>  (a * p0^2 + b * p0^1 + c * p0^0) % p1 where p0 and p1 are both prime and p0 is much smaller than p1`.

**Solution with Rolling Hash:** [https://repl.it/@trsong/Shift-Equivalent-Strings](https://repl.it/@trsong/Shift-Equivalent-Strings)
```py
import unittest

P0 = 23  # small prime number
P1 = 666667 # larger prime number

def hash(s):
    res = 0
    for char in s:
        ord_char = ord(char)
        res = (res * P0 % P1 + ord_char) % P1
    return res


def base_at(n):
    res = 1
    for _ in xrange(n-1):
        res = (res * P0) % P1
    return res


def is_shift_eq(source, target):
    if len(source) != len(target):
        return False    
    if source == target:
        return True

    n = len(target)
    most_significant_base = base_at(n)
    target_hash = hash(target)
    source_hash = hash(source)

    for shift_char in source:
        ord_shift_char = ord(shift_char)
        most_significant_value = (ord_shift_char * most_significant_base) % P1
        source_hash = ((source_hash - most_significant_value) * P0 + ord_shift_char) % P1
        if source_hash == target_hash:
            return True
    
    return False


class IsShiftEqSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(is_shift_eq('abcde', 'cdeab'))

    def test_example2(self):
        self.assertFalse(is_shift_eq('abc', 'acb'))

    def test_different_length_strings(self):
        self.assertFalse(is_shift_eq(' a ', ' a'))

    def test_empty_strings(self):
        self.assertTrue(is_shift_eq('', ''))

    def test_string_with_unique_word(self):
        self.assertTrue(is_shift_eq('aaaaa', 'aaaaa'))

    def test_string_with_multiple_spaces(self):
        self.assertFalse(is_shift_eq('aa aa aa', 'aaaa  aa'))

    def test_number_strins(self):
        self.assertTrue(is_shift_eq("567890", "890567"))

    def test_large_string_performance_test(self):
        N = 100000
        source = str(range(N))
        target = source[:N//2] + source[N//2:]
        self.assertTrue(is_shift_eq(source, target))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Oct 2, 2019 \[Medium\] Numbers With Equal Digit Sum
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

**Solution:** [https://repl.it/@trsong/Numbers-With-Equal-Digit-Sum](https://repl.it/@trsong/Numbers-With-Equal-Digit-Sum)

```py
import unittest

def calc_digit_sum(num):
    res = 0
    while num > 0:
        res += num % 10
        num //= 10
    return res


def find_max_digit_sum(nums):
    nums_groupby_digit_sum = {}
    for num in nums:
        digit_sum = calc_digit_sum(num)
        if digit_sum not in nums_groupby_digit_sum:
            nums_groupby_digit_sum[digit_sum] = []
        
        same_digit_sum_nums = nums_groupby_digit_sum[digit_sum]
        if len(same_digit_sum_nums) > 1:
            max_num, min_num = same_digit_sum_nums
            same_digit_sum_nums[0] = max(max_num, num)
            same_digit_sum_nums[1] = max_num + min_num + num - same_digit_sum_nums[0] - min(min_num, num)
        else:
            same_digit_sum_nums.append(num)

    max_digit_sum = -1
    for same_digit_sum_nums in nums_groupby_digit_sum.values():
        if len(same_digit_sum_nums) < 2:
            continue
        max_digit_sum = max(max_digit_sum, sum(same_digit_sum_nums))
    
    return max_digit_sum


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
    unittest.main(exit=False)
```

### Oct 1, 2019 LC 1171 \[Medium\] Remove Consecutive Nodes that Sum to 0
---
> **Question:** Given a linked list of integers, remove all consecutive nodes that sum up to 0.

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

**My thoughts:** This question is just the list version of [Contiguous Sum to K](https://trsong.github.io/python/java/2019/05/01/DailyQuestions/#jul-24-2019-medium-contiguous-sum-to-k). The idea is exactly the same, in previous question: `sum[i:j]` can be achieved use `prefix[j] - prefix[i-1] where i <= j`, whereas for this question, we can use map to store the "prefix" sum: the sum from the head node all the way to current node. And by checking the prefix so far, we can easily tell if there is a node we should have seen before that has "prefix" sum with same value. i.e. There are consecutive nodes that sum to 0 between these two nodes.

**Solution:** [https://repl.it/@trsong/Remove-Consecutive-Nodes-that-Sum-to-0](https://repl.it/@trsong/Remove-Consecutive-Nodes-that-Sum-to-0)
```py
import unittest

class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    def __eq__(self, other):
        res = str(self) == str(other)
        if not res:
            print str(self), '!=' , str(other)
        return res

    def __str__(self):
        current_node = self
        result = []
        while current_node:
            result.append(current_node.val)
            current_node = current_node.next
        return str(result)

    @staticmethod  
    def List(*vals):
        dummy = ListNode(-1)
        p = dummy
        for elem in vals:
            p.next = ListNode(elem)
            p = p.next
        return dummy.next  


def remove_zero_sum_sublists(head):
    if not head:
        return None
    
    dummy = ListNode(-1, head)
    node_with_prefix = {0: dummy}
    p = head
    sum_so_far = 0
    while p:
        sum_so_far += p.val
        if sum_so_far in node_with_prefix:
            sum_to_remove = sum_so_far
            t = node_with_prefix[sum_so_far].next
            while t != p:
                sum_to_remove += t.val
                del node_with_prefix[sum_to_remove]
                t = t.next
                    
            node_with_prefix[sum_so_far].next = p.next
        else:
            node_with_prefix[sum_so_far] = p
        p = p.next
    return dummy.next


class RemoveZeroSumSublistSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(ListNode.List(10), remove_zero_sum_sublists(ListNode.List(10, 5, -3, -3, 1, 4, -4)))

    def test_example2(self):
        self.assertEqual(ListNode.List(3, 1), remove_zero_sum_sublists(ListNode.List(1, 2, -3, 3, 1)))

    def test_example3(self):
        self.assertEqual(ListNode.List(1, 2, 4), remove_zero_sum_sublists(ListNode.List(1, 2, 3, -3, 4)))

    def test_example4(self):
        self.assertEqual(ListNode.List(1), remove_zero_sum_sublists(ListNode.List(1, 2, 3, -3, -2)))

    def test_empty_list(self):
        self.assertEqual(ListNode.List(), remove_zero_sum_sublists(ListNode.List()))

    def test_all_zero_list(self):
        self.assertEqual(ListNode.List(), remove_zero_sum_sublists(ListNode.List(0, 0, 0)))

    def test_add_up_to_zero_list(self):
        self.assertEqual(ListNode.List(), remove_zero_sum_sublists(ListNode.List(1, -1, 0, -1, 1)))

    def test_overlap_section_add_to_zero(self):
        self.assertEqual(ListNode.List(1, 5, -1, -2, 99), remove_zero_sum_sublists(ListNode.List(1, -1, 2, 3, 0, -3, -2, 1, 5, -1, -2, 99)))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 30, 2019 LC 139 \[Medium\] Word Break
---
> **Question:** Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words.
>
> **Note:**
> * The same word in the dictionary may be reused multiple times in the segmentation.
> * You may assume the dictionary does not contain duplicate words.

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

**My thoughts:** This question feels almost the same as [LC 279 Minimum Number of Squares Sum to N](https://trsong.github.io/python/java/2019/08/02/DailyQuestionsAug/#sep-22-2019-lc-279-medium-minimum-number-of-squares-sum-to-n). The idea is to think about the problem backwards and you may want to ask yourself: what makes `s[:n]` to be `True`? There must exist some word with length `m` where `m < n` such that `s[:n-m]` is `True` and string `s[n-m:n]` is in the dictionary. Therefore, the problem size shrinks from `n` to  `m` and it will go all the way to empty string which definitely is `True`.

**Solution with DP:** [https://repl.it/@trsong/Word-Break](https://repl.it/@trsong/Word-Break)
```py
import unittest

def word_break(s, word_dict):
    if not s:
        return True
    
    n = len(s)
    # dp[n] represents where s[:n] can be word break
    # dp[k] = True only if dp[i] = True and s[i:k] is in the dictionary
    dp = [False] * (n + 1)
    dp[0] = True
    for k in xrange(1, n+1):
        for word in word_dict:
            word_len = len(word)
            if word_len <= k and dp[k-word_len] and word == s[k-word_len:k]:
                # Once make sure s[:k] satisfied, just short circuit and move to next k
                dp[k] = True
                break
    return dp[n]


class WordBreakSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(word_break("Pseudocode", ["Pseudo", "code"]))

    def test_example2(self):
        self.assertTrue(word_break("applepenapple", ["apple", "pen"]))

    def test_example3(self):
        self.assertFalse(word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]))

    def test_word_in_dict_is_a_prefix(self):
        self.assertTrue(word_break("123456", ["12", "123", "456"]))
    
    def test_word_in_dict_is_a_prefix2(self):
        self.assertTrue(word_break("123456", ["12", "123", "456"]))

    def test_word_in_dict_is_a_prefix3(self):
        self.assertFalse(word_break("123456", ["12", "12345", "456"]))

    def test_empty_word(self):
        self.assertTrue(word_break("", ['a']))
        self.assertTrue(word_break("", []))
        self.assertFalse(word_break("a", []))

    def test_use_same_word_twice(self):
        self.assertTrue(word_break("aaabaaa", ["aaa", "b"]))

    def test_impossible_word_combination(self):
        self.assertFalse(word_break("aaaaa", ["aaa", "aaaa"]))
        

if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 29, 2019 LC 773 \[Hard\] Sliding Puzzle
---
> **Question:** On a 2x3 board, there are 5 tiles represented by the integers 1 through 5, and an empty square represented by 0. Given a puzzle board, return the least number of moves required so that the state of the board is solved. If it is impossible for the state of the board to be solved, return -1.
>
> **Note:**
> * A move consists of choosing 0 and a 4-directionally adjacent number and swapping it.
> * The state of the board is solved if and only if the board is `[[1,2,3],[4,5,0]]`.

**Example 1:**
```py
Input: board = [
    [1, 2, 3],
    [4, 0, 5]
]
Output: 1
Explanation: Swap the 0 and the 5 in one move.
```

**Example 2:**
```py
Input: board = [
    [1, 2, 3],
    [5, 4, 0]
]
Output: -1
Explanation: No number of moves will make the board solved.
```

**Example 3:**
```py
Input: board = [
    [4, 1, 2],
    [5, 0, 3]
]
Output: 5
Explanation: 5 is the smallest number of moves that solves the board.
An example path:
After move 0: [
    [4, 1, 2],
    [5, 0, 3]
]
After move 1: [
    [4, 1, 2],
    [0, 5, 3]
]
After move 2: [
    [0, 1, 2],
    [4, 5, 3]
]
After move 3: [
    [1, 0, 2],
    [4, 5, 3]
]
After move 4: [
    [1, 2, 0],
    [4, 5, 3]
]
After move 5: [
    [1, 2, 3],
    [4, 5, 0]
]
```

**Example 4:**
```py
Input: board = [
    [3, 2, 4],
    [1, 5, 0]
]
Output: 14
```

**My thoughts:** This is a typical solution searching problem. BFS/DFS/A* will work. Probably BFS easier to implement than others: just figure out how to encode intial board state, goal state and state transition function. However this reason why I choose this problem is to demonstrate A* search. 

**Approach 1: 2 x 3 Puzzle BFS Solution** [https://repl.it/@trsong/Sliding-Puzzle](https://repl.it/@trsong/Sliding-Puzzle)
```py
import unittest
from Queue import Queue


def hash_state(board):
    res = 0
    for row in board:
        for cell in row:
            res = res * 10 + cell
    return res


def state_to_board_and_start(state, n, m):
    grid = [[0 for _ in range(m)] for _ in range(n)]
    start = None
    for r in range(n - 1, -1, -1):
        for c in range(m - 1, -1, -1):
            grid[r][c] = state % 10
            if grid[r][c] == 0:
                start = (r, c)
            state //= 10
    return grid, start


def next_move_states(state, n, m):
    grid, start = state_to_board_and_start(state, n, m)
    r, c = start
    directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    res = []
    for dr, dc in directions:
        new_r, new_c = r + dr, c + dc
        if 0 <= new_r < n and 0 <= new_c < m:
            grid[r][c], grid[new_r][new_c] = grid[new_r][new_c], grid[r][c]
            res.append(hash_state(grid))
            grid[r][c], grid[new_r][new_c] = grid[new_r][new_c], grid[r][c]
    return res


def solve_puzzle(board):
    if not board or not board[0]:
        return -1

    n, m = len(board), len(board[0])
    goal_state = hash_state([list(range(1, n*m)) + [0]])
    queue = Queue()
    visited = set()
    initial_state = hash_state(board)
    queue.put(initial_state)
    depth = 0
    while not queue.empty():
        for _ in range(queue.qsize()):
            current_state = queue.get()
            if current_state in visited:
                continue
            elif current_state == goal_state:
                return depth

            visited.add(current_state)
            for next_state in next_move_states(current_state, n, m):
                if next_state in visited:
                    continue
                queue.put(next_state)
        depth += 1
    return -1


class SolvePuzzleSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(1, solve_puzzle([
            [1, 2, 3],
            [4, 0, 5]
        ]))  # 0 -> 5

    def test_example2(self):
        self.assertEqual(-1, solve_puzzle([
            [1, 2, 3],
            [5, 4, 0]
        ]))

    def test_example3(self):
        self.assertEqual(5, solve_puzzle([
            [4, 1, 2],
            [5, 0, 3]
        ]))  # 0 -> 5 -> 4 -> 1 -> 2 -> 3

    def test_example4(self):
        self.assertEqual(14, solve_puzzle([
            [3, 2, 4],
            [1, 5, 0]
        ]))

    def test_back_tracking_3(self):
        self.assertEqual(3, solve_puzzle([
            [0, 1, 3],
            [4, 2, 5]
        ]))  # 0 -> 1 -> 2 -> 5

    def test_back_tacking_4(self):
        self.assertEqual(4, solve_puzzle([
            [1, 5, 2],
            [0, 4, 3]
        ]))  # 0 -> 4 -> 5 -> 2 -> 3

   def test_finish_state(self):
        self.assertEqual(0, solve_puzzle([
            [1, 2, 3],
            [4, 5, 0]
        ]))

if __name__ == '__main__':
    unittest.main(exit=False)
```

**Approach 2: n x m Puzzle AStar Search Solution** [https://repl.it/@trsong/Sliding-Puzzle-AStar-Search](https://repl.it/@trsong/Sliding-Puzzle-AStar-Search)
```py
import unittest
from Queue import PriorityQueue

# Suppose puzzle size < 23
p = 23


def hash_state(board):
    res = 0
    for row in board:
        for cell in row:
            res = res * p + cell
    return res


def state_to_board_and_start(state, n, m):
    grid = [[0 for _ in range(m)] for _ in range(n)]
    start = None
    for r in range(n - 1, -1, -1):
        for c in range(m - 1, -1, -1):
            grid[r][c] = state % p
            if grid[r][c] == 0:
                start = (r, c)
            state //= p
    return grid, start


def heuristic_cost(board):
    # Define the heuristic cost to be the total manhattan distance of each misplaced puzzle piece
    res = 0
    n, m = len(board), len(board[0])
    for r in range(n):
        for c in range(m):
            val = board[r][c]
            if val == 0:
                continue
            target_r, target_c = val // m, val % m

            # The original co-ordinates start from 0, shift left by 1
            if target_c == 0:
                target_c = m - 1
                target_r -= 1
            else:
                target_r - 1
            distance = abs(r - target_r) + abs(c - target_c)
            res += distance
    return res


def next_move_states_and_cost(state, n, m):
    grid, start = state_to_board_and_start(state, n, m)
    r, c = start
    directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    res = []
    edge_cost = 1
    for dr, dc in directions:
        new_r, new_c = r + dr, c + dc
        if 0 <= new_r < n and 0 <= new_c < m:
            # apply move to grid
            grid[r][c], grid[new_r][new_c] = grid[new_r][new_c], grid[r][c]

            # calculate next move
            remaining_cost_estimate = heuristic_cost(grid)
            res.append((hash_state(grid), edge_cost + remaining_cost_estimate))

            # restore previous state
            grid[r][c], grid[new_r][new_c] = grid[new_r][new_c], grid[r][c]
    return res


def solve_puzzle(board):
    if not board or not board[0]:
        return -1

    n, m = len(board), len(board[0])
    goal_state = hash_state([list(range(1, n * m)) + [0]])
    pq = PriorityQueue()
    initial_state = hash_state(board)
    pq.put((0, initial_state))
    visited = set()
    state_cost = {initial_state: 0}
    edge_cost = 1

    while not pq.empty():
        _, current_state = pq.get()
        if current_state in visited:
            continue

        visited.add(current_state)
        cost_so_far = state_cost[current_state]
        for next_state, remaining_cost in next_move_states_and_cost(current_state, n, m):
            is_visited = next_state in visited
            is_a_larger_path = next_state in state_cost and cost_so_far + edge_cost > state_cost[next_state]
            if is_visited or is_a_larger_path:
                continue
            state_cost[next_state] = cost_so_far + edge_cost
            pq.put((cost_so_far + remaining_cost, next_state))

    return state_cost[goal_state] if goal_state in state_cost else -1


class SolvePuzzleSpec(unittest.TestCase):
    def test_heuristic_function_should_not_overestimate(self):
        self.assertEqual(8, solve_puzzle([
            [0, 1, 2],
            [5, 6, 3],
            [4, 7, 8],
        ]))  # 0 -> 1 -> 2 -> 3 -> 6 -> 5 -> 4 -> 7 -> 8

    ## Time Consuming Test!
    # def test_not_solvable_puzzle(self):
    #     self.assertEqual(-1, solve_puzzle([
    #         [8, 1, 2],
    #         [0, 4, 3],
    #         [7, 6, 5],
    #     ]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 28, 2019 LC 286 \[Medium\] Walls and Gates
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

**Solution with BFS:** [https://repl.it/@trsong/Walls-and-Gates](https://repl.it/@trsong/Walls-and-Gates)
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
    direction = [[1, 0], [0, -1], [-1, 0], [0, 1]]
    while not queue.empty():
        for _ in xrange(queue.qsize()):
            r, c = queue.get()
            if 0 < grid[r][c] < INF:
                continue
            grid[r][c] = depth
            for dr, dc in direction:
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

### Sep 27, 2019 \[Medium\] Construct BST from Post-order Traversal
---
> **Question:** Given the sequence of keys visited by a postorder traversal of a binary search tree, reconstruct the tree.

**Example:**
```py
Given the sequence 2, 4, 3, 8, 7, 5, you should construct the following tree:

    5
   / \
  3   7
 / \   \
2   4   8
```

**Solution:** [https://repl.it/@trsong/Construct-BST-from-Post-order-Traversal](https://repl.it/@trsong/Construct-BST-from-Post-order-Traversal)
```py
import unittest
import sys

class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __eq__(self, other):
        return other and other.val == self.val and other.left == self.left and other.right == self.right


def construct_bst(post_order_traversal):
    if not post_order_traversal:
        return None
    reverse_order = post_order_traversal[::-1]
    class Context:
        index = 0

    def construct_bst_recur(min_val, max_val):
        if Context.index >= len(reverse_order):
            return None
            
        current_val = reverse_order[Context.index]
        if current_val < min_val or current_val > max_val:
            return None
        Context.index += 1
        current = Node(current_val)
        current.right = construct_bst_recur(current_val, max_val)
        current.left = construct_bst_recur(min_val, current_val)
        return current

    return construct_bst_recur(-sys.maxint, sys.maxint)


class ConstructBSTSpec(unittest.TestCase):
    def test_example(self):
        """
            5
           / \
          3   7
         / \   \
        2   4   8
        """
        post_order_traversal = [2, 4, 3, 8, 7, 5]
        n3 = Node(3, Node(2), Node(4))
        n7 = Node(7, right=Node(8))
        n5 = Node(5, n3, n7)
        self.assertEqual(n5, construct_bst(post_order_traversal))

    def test_empty_bst(self):
        self.assertIsNone(construct_bst([]))

    def test_left_heavy_bst(self):
        """
            3
           /
          2
         /
        1
        """
        self.assertEqual(Node(3, Node(2, Node(1))), construct_bst([1, 2, 3]))

    def test_right_heavy_bst(self):
        """
          1
         / \
        0   3
           / \
          2   4
               \
                5
        """
        post_order_traversal = [0, 2, 5, 4, 3, 1]
        n3 = Node(3, Node(2), Node(4, right=Node(5)))
        n1 = Node(1, Node(0), n3)
        self.assertEqual(n1, construct_bst(post_order_traversal))

    def test_complete_binary_tree(self):
        """
             3
           /   \
          1     5
         / \   / 
        0   2 4
        """
        post_order_traversal = [0, 2, 1, 4, 5, 3]
        n1 = Node(1, Node(0), Node(2))
        n5 = Node(5, Node(4))
        n3 = Node(3, n1, n5)
        self.assertEqual(n3, construct_bst(post_order_traversal))

    def test_right_left_left(self):
        """
          1
         / \
        0   4
           /
          3
         /
        2
        """
        post_order_traversal = [0, 2, 3, 4, 1]
        n4 = Node(4, Node(3, Node(2)))
        n1 = Node(1, Node(0), n4)
        self.assertEqual(n1, construct_bst(post_order_traversal))

    def test_left_right_right(self):
        """
          4
         / \
        1   5
         \
          2
           \
            3
        """
        post_order_traversal = [3, 2, 1, 5, 4]
        n1 = Node(1, right=Node(2, right=Node(3)))
        n4 = Node(4, n1, Node(5))
        self.assertEqual(n4, construct_bst(post_order_traversal))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 26, 2019 \[Hard\] Ordered Minimum Window Subsequence
---
> **Question:** Given an array nums and a subsequence sub, find the shortest subarray of nums that contains sub.
> 
> * If such subarray does not exist, return -1, -1.
> * Note that the subarray must contain the elements of sub in the correct order.

**Example:**

```py
Input: nums = [1, 2, 3, 5, 8, 7, 6, 9, 5, 7, 3, 0, 5, 2, 3, 4, 4, 7], sub = [5, 7]
Output: start = 8, size = 2
```

**Solution with DP:** [https://repl.it/@trsong/Ordered-Minimum-Window-Subsequence](https://repl.it/@trsong/Ordered-Minimum-Window-Subsequence)
```py
import unittest
import sys

def min_window(nums, sub):
    if not nums:
        return -1, -1
    n, m = len(nums), len(sub)
    # dp[i][j] represents the largest index between [0, i) such that nums[:i] contains subsequence sub[:j]
    # then dp[i][0] = i.
    # and dp[i][j] = dp[i-1][j-1] if nums[i-1] matches sub[j-1]
    #              = dp[i-1][j]   otherwise
    # dp[i][m] will be the largest index that contains subsequence sub
    dp = [[-1 for _ in xrange(m+1)] for _ in xrange(n+1)]
    min_size = sys.maxint
    for i in xrange(n+1):
        dp[i][0] = i
    
    for i in xrange(1, n+1):
        # we cannot have j > i as we want to check if  nums[:i] contains sub[:j] or not
        for j in xrange(1, min(m, i)+1):
            if nums[i-1] == sub[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = dp[i-1][j]

    for i in xrange(n+1):
        if dp[i][m] != -1:
            size = i - dp[i][m]
            if size < min_size:
                start = dp[i][m]
                min_size = size

    if min_size == sys.maxint or min_size < m:
        return -1, -1
    else:
        return start, min_size


class MinWindowSpec(unittest.TestCase):
    def test_example(self):
        nums = [1, 2, 3, 5, 8, 7, 6, 9, 5, 7, 3, 0, 5, 2, 3, 4, 4, 7]
        sub = [5, 7]
        self.assertEqual((8, 2), min_window(nums, sub))

    def test_sub_not_exits(self):
        nums = [0, 1, 0, 2, 0, 0, 3, 4]
        sub = [2, 1, 3]
        self.assertEqual((-1, -1), min_window(nums, sub))

    def test_nums_is_empty(self):
        self.assertEqual((-1, -1), min_window([], [42]))
    
    def test_sub_is_empty(self):
        self.assertEqual((0, 0), min_window([1, 4, 3, 2], []))

    def test_both_nums_and_sub_are_empty(self):
        self.assertEqual((-1, -1), min_window([], []))

    def test_duplicated_numbers(self):
        nums = [1, 1, 1, 1]
        sub = [1, 1, 1, 1]
        self.assertEqual((0, 4), min_window(nums, sub))

    def test_duplicated_numbers2(self):
        nums = [1, 1, 1]
        sub = [1, 1, 1, 1]
        self.assertEqual((-1, -1), min_window(nums, sub))

    def test_duplicated_numbers3(self):
        nums = [1, 1]
        sub = [1, 0]
        self.assertEqual((-1, -1), min_window(nums, sub))

    def test_min_window(self):
        nums = [1, 0, 2, 0, 0, 1, 0, 2, 1, 1, 2]
        sub = [1, 2, 1]
        self.assertEqual((5, 4), min_window(nums, sub))
        sub2 = [1, 2, 2, 2, 1]
        self.assertEqual((-1, -1), min_window(nums, sub2))

    def test_moving_window(self):
        nums = [1, 1, 2, 1, 2, 3, 1, 2]
        sub = [1, 2, 3]
        self.assertEqual((3, 3), min_window(nums, sub))

    def test_min_window2(self):
        nums = [1, 1, 1, 0, 2, 2, 1, 1, 2, 2, 2, 2]
        sub = [1, 2]
        self.assertEqual((7, 2), min_window(nums, sub))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 25, 2019 \[Easy\] Flatten a Nested Dictionary
---
> **Question:** Write a function to flatten a nested dictionary. Namespace the keys with a period.

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

**Solution with Iterator:** [https://repl.it/@trsong/Flatten-a-Nested-Dictionary](https://repl.it/@trsong/Flatten-a-Nested-Dictionary)
```py
import unittest

def create_dictionary_iterator(dictionary):
    for k, v in dictionary.items():
        if type(v) is dict:
            for sub_k, sub_v in create_dictionary_iterator(v):
                yield '{}.{}'.format(k, sub_k), sub_v
        else:
            yield k, v


def flatten_dictionary(dictionary):
    iterator = create_dictionary_iterator(dictionary)
    return {k: v for k,v in iterator}


class FlattenDictionarySpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual({
            "key": 3,
            "foo.a": 5,
            "foo.bar.baz": 8
        }, flatten_dictionary({
            "key": 3,
            "foo": {
                "a": 5,
                "bar": {
                    "baz": 8}}}))

    def test_empty_dictionary(self):
        self.assertEqual({}, flatten_dictionary({}))

    def test_simple_dictionary(self):
        d = {
            'a': 1,
            'b': 2
        }
        self.assertEqual(d, flatten_dictionary(d))
    
    def test_multi_level_dictionary(self):
        d_e = {'e': 0}
        d_d = {'d': d_e}
        d_c = {'c': d_d}
        d_b = {'b': d_c}
        d_a = {'a': d_b}
        self.assertEqual({
            'a.b.c.d.e': 0
        }, flatten_dictionary(d_a))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 24, 2019 \[Medium\] Interleave Stacks
---
> **Question:** Given a stack of N elements, interleave the first half of the stack with the second half reversed using only one other queue. This should be done in-place.
>
> Recall that you can only push or pop from a stack, and enqueue or dequeue from a queue.
>
> For example, if the stack is `[1, 2, 3, 4, 5]`, it should become `[1, 5, 2, 4, 3]`. If the stack is `[1, 2, 3, 4]`, it should become `[1, 4, 2, 3]`.
>
> Hint: Try working backwards from the end state.

**Solution:** [https://repl.it/@trsong/Interleave-Stacks](https://repl.it/@trsong/Interleave-Stacks)
```py
import unittest

def interleave_stack(stack):
    if not stack:
        return 

    # stack = [1, 2, 3, 4, 5, 6, 7], queue = []
    queue = []
    size = len(stack)
    half_size = size / 2
    for _ in xrange(half_size):
        queue.append(stack.pop())

    # stack = [1, 2, 3, 4], queue = [7, 6, 5]
    for _ in xrange(half_size):
        stack.append(queue.pop(0))

    # stack = [1, 2, 3, 4, 7, 6, 5], queue = []
    for _ in xrange(half_size):
        queue.append(stack.pop())

    # stack = [1, 2, 3, 4], queue = [5, 6, 7]
    last = None
    if len(stack) != len(queue):
        last = stack.pop()

    # stack = [1, 2, 3], queue = [5, 6, 7]
    for _ in xrange(half_size):
        queue.append(queue.pop(0))
        queue.append(stack.pop())

    # stack = [], queue = [5, 3, 6, 2, 7, 1]
    while queue:
        stack.append(queue.pop(0))

    # stack = [5, 3, 6, 2, 7, 1], queue = []
    while stack:
         queue.append(stack.pop())

    # stack = [], queue = [1, 7, 2, 6, 3, 5]
    while queue:
        stack.append(queue.pop(0))

    # stack = [1, 7, 2, 6, 3, 5], queue = []
    if last is not None:
        stack.append(last)
    # stack = [1, 7, 2, 6, 3, 5, 4], queue = []


class InterleaveStackSpec(unittest.TestCase):
    def test_example1(self):
        stack = [1, 2, 3, 4, 5]
        expected = [1, 5, 2, 4, 3]
        interleave_stack(stack)
        self.assertEqual(stack, expected)

    def test_example2(self):
        stack = [1, 2, 3, 4]
        expected = [1, 4, 2, 3]
        interleave_stack(stack)
        self.assertEqual(stack, expected)

    def test_empty_stack(self):
        stack = []
        expected = []
        interleave_stack(stack)
        self.assertEqual(stack, expected)

    def test_size_one_stack(self):
        stack = [1]
        expected = [1]
        interleave_stack(stack)
        self.assertEqual(stack, expected)

    def test_size_two_stack(self):
        stack = [1, 2]
        expected = [1, 2]
        interleave_stack(stack)
        self.assertEqual(stack, expected)

    def test_size_three_stack(self):
        stack = [1, 2, 3]
        expected = [1, 3, 2]
        interleave_stack(stack)
        self.assertEqual(stack, expected)

    def test_size_seven_stack(self):
        stack = [1, 2, 3, 4, 5, 6, 7]
        expected = [1, 7, 2, 6, 3, 5, 4]
        interleave_stack(stack)
        self.assertEqual(stack, expected)
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 23, 2019 \[Easy\] BST Nodes Sum up to K
---
> **Question:** Given the root of a binary search tree, and a target K, return two nodes in the tree whose sum equals K.

**Example:** 
```py
Given the following tree and K of 20

    10
   /   \
 5      15
       /  \
     11    15
Return the nodes 5 and 15.
```

**My thoughts:** BST in-order traversal is equivalent to sorted list. Therefore the question can be converted to 2-sum with sorted input. 

**Solution with In-order Traversal:** [https://repl.it/@trsong/BST-Nodes-Sum-up-to-K](https://repl.it/@trsong/BST-Nodes-Sum-up-to-K)
```py
import unittest

class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def in_order_traversal(tree):
    if tree:
        for left_tree in in_order_traversal(tree.left):
            yield left_tree
        yield tree
        for right_tree in in_order_traversal(tree.right):
            yield right_tree


def reverse_in_order_traversal(tree):
    if tree:
        for right_tree in reverse_in_order_traversal(tree.right):
            yield right_tree
        yield tree
        for left_tree in reverse_in_order_traversal(tree.left):
            yield left_tree


def find_pair(tree, k):
    if not tree:
        return None
    
    traversal = in_order_traversal(tree)
    reverse_traversal = reverse_in_order_traversal(tree)

    left = next(traversal)
    right = next(reverse_traversal)
    while left != right:
        sum = left.val + right.val
        if sum == k:
            return [left.val, right.val]
        elif sum < k:
            left = next(traversal)
        else:
            # sum > k
            right = next(reverse_traversal)
    return None


class FindPairSpec(unittest.TestCase):
    def test_example(self):
        """
            10
           /   \
         5      15
               /  \
             11    15
        """
        n15 = Node(15, Node(11), Node(15))
        n10 = Node(10, Node(5), n15)
        self.assertEqual(find_pair(n10, 20), [5, 15])

    def test_empty_tree(self):
        self.assertIsNone(find_pair(None, 0), None)

    def test_full_tree(self):
        """
             7
           /   \
          3     13
         / \   /  \
        2   5 11   17
        """
        n3 = Node(3, Node(2), Node(5))
        n13 = Node(13, Node(11), Node(17))
        n7 = Node(7, n3, n13)
        self.assertEqual(find_pair(n7, 7), [2, 5])
        self.assertEqual(find_pair(n7, 18), [5, 13])
        self.assertEqual(find_pair(n7, 24), [7, 17])
        self.assertEqual(find_pair(n7, 28), [11, 17])
        self.assertIsNone(find_pair(n7, 4))

    def test_tree_with_same_value(self):
        """
        42
          \
           42
        """
        tree = Node(42, right=Node(42))
        self.assertEqual(find_pair(tree, 84), [42, 42])
        self.assertIsNone(find_pair(tree, 42))

    def test_sparse_tree(self):
        """
           7
         /   \
        2     17
         \   /
          5 11
         /   \
        3     13
        """
        n2 = Node(2, right=Node(5, Node(3)))
        n17 = Node(17, Node(11, right=Node(13)))
        n7 = Node(7, n2, n17)
        self.assertEqual(find_pair(n7, 7), [2, 5])
        self.assertEqual(find_pair(n7, 18), [5, 13])
        self.assertEqual(find_pair(n7, 24), [7, 17])
        self.assertEqual(find_pair(n7, 28), [11, 17])
        self.assertIsNone(find_pair(n7, 4))


if __name__ == '__main__':
   unittest.main(exit=False)
```

### Sep 22, 2019 LC 279 \[Medium\] Minimum Number of Squares Sum to N
---
> **Question:** Given a positive integer n, find the smallest number of squared integers which sum to n.
>
> For example, given `n = 13`, return `2` since `13 = 3^2 + 2^2 = 9 + 4`.
> 
> Given `n = 27`, return `3` since `27 = 3^2 + 3^2 + 3^2 = 9 + 9 + 9`.

**Solution with DP:** [https://repl.it/@trsong/Minimum-Number-of-Squares-Sum-to-N](https://repl.it/@trsong/Minimum-Number-of-Squares-Sum-to-N)
```py
import unittest
import math
import sys

def is_perfect_square(num):
    sqr_root = math.sqrt(num)
    return sqr_root - math.floor(sqr_root) == 0


def perfect_squres(N):
    if is_perfect_square(N):
        return 1
    
    # Let dp[n] represents smallest number of perfect squares that add up to n
    # Then dp[n] = 1 + min(dp[n - perfect_squares]) for all suitable perfect_squares
    dp = [sys.maxint] * (N + 1)
    dp[0] = 0

    for num in xrange(1, N + 1):
        for i in xrange(1, int(math.sqrt(num) + 1)):
            dp[num] = min(dp[num - i * i] + 1, dp[num])
    return dp[N]


class PerfectSqureSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(perfect_squres(13), 2) # n = 13, return 2 since 13 = 3^2 + 2^2 = 9 + 4.

    def test_example2(self):
        self.assertEqual(perfect_squres(27), 3) # n = 27, return 3 since 27 = 3^2 + 3^2 + 3^2 = 9 + 9 + 9

    def test_perfect_square(self):
        self.assertEqual(perfect_squres(100), 1) # 10^2 = 100

    def test_random_number(self):
        self.assertEqual(perfect_squres(63), 4) # n = 63, return 4 since 63 = 7^2+ 3^2 + 2^2 + 1^2

    def test_random_number2(self):
        self.assertEqual(perfect_squres(12), 3) # n = 12, return 3, 12 = 4 + 4 + 4

    def test_random_number3(self):
        self.assertEqual(perfect_squres(6), 3) # 6 = 2 + 2 + 2


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 21, 2019 \[Medium\] Subtree with Maximum Average
---
> **Question:** Given an N-ary tree, find the subtree with the maximum average. Return the root of the subtree.
> 
> A subtree of a tree is the node which have at least 1 child plus all its descendants. The average value of a subtree is the sum of its values, divided by the number of nodes.

**Example:**

```py
Input:
     _20_
    /    \
   12    18
 / | \   / \
11 2  3 15  8

Output: 18
Explanation:
There are 3 nodes which have children in this tree:
12 => (11 + 2 + 3 + 12) / 4 = 7
18 => (18 + 15 + 8) / 3 = 13.67
20 => (12 + 11 + 2 + 3 + 18 + 15 + 8 + 20) / 8 = 11.125

18 has the maximum average so output 18.
```

**Solution:** [https://repl.it/@trsong/Subtree-with-Maximum-Average](https://repl.it/@trsong/Subtree-with-Maximum-Average)

```py
import unittest

class Node(object):
    def __init__(self, val, *children):
        self.val = val
        self.children = children

def max_avg_subtree(tree):
    def max_avg_and_n(tree):
        if not tree.children:
            # (child_max_avg, child_node), sum, size
            return (float('-inf'), tree), tree.val, 1

        child_res = map(max_avg_and_n, tree.children)
        child_max_avg_and_nodes = map(lambda x: x[0], child_res)
        child_max_avgs = map(lambda x: x[0], child_max_avg_and_nodes)
        child_max_avg = max(child_max_avgs)
        sum_children = sum(map(lambda x: x[1], child_res))
        num_children = sum(map(lambda x: x[2], child_res))
        node_sum = sum_children + tree.val
        current_max_avg = float(node_sum) / (1 + num_children)
        res = (current_max_avg, tree)
        if child_max_avg > current_max_avg:
            index = child_max_avgs.index(child_max_avg)
            target_child = child_max_avg_and_nodes[index][1]
            res = (child_max_avg, target_child)
        return res, node_sum, num_children + 1

    return max_avg_and_n(tree)[0][1].val


class MaxAvgSubtreeSpec(unittest.TestCase):
    def test_example(self):
        """
             _20_
            /    \
           12    18
         / | \   / \
        11 2  3 15  8

        12 => (11 + 2 + 3 + 12) / 4 = 7
        18 => (18 + 15 + 8) / 3 = 13.67
        20 => (12 + 11 + 2 + 3 + 18 + 15 + 8 + 20) / 8 = 11.125
        """
        n12 = Node(12, Node(12), Node(2), Node(3))
        n18 = Node(18, Node(15), Node(8))
        n20 = Node(20, n12, n18)
        self.assertEqual(max_avg_subtree(n20), 18) 

    def test_tree_with_negative_node(self):
        """
             1
           /   \
         -5     11
         / \   /  \
        1   2 4   -2

        -5 => (-5 + 1 + 2) / 3 = -0.67
        11 => (11 + 4 - 2) / 3 = 4.333
        1 => (1 -5 + 11 + 1 + 2 + 4 - 2) / 7 = 1
        """
        n5 = Node(-5, Node(1), Node(2))
        n11 = Node(11, Node(4), Node(-2))
        n1 = Node(1, n5, n11)
        self.assertEqual(max_avg_subtree(n1), 11)

    def test_right_heavy_tree(self):
        """
          0
         / \
        10  1
          / | \
         0  4  3
         1 => (0 + 4 + 3 + 1) / 4 = 2
         0 => (10 + 1 + 4 + 3) / 6 = 3
        """
        n1 = Node(1, Node(0), Node(4), Node(3))
        n0 = Node(0, Node(10), n1)
        self.assertEqual(max_avg_subtree(n0), 0)

    def test_multiple_level_tree(self):
        """
         0
          \ 
           0
            \
             2
              \
               4

        0 => (0 + 2 + 4) / 4 = 1.5
        2 => (2 + 4) / 2 = 3       
        """
        t = Node(2, Node(0, Node(2, Node(4))))
        self.assertEqual(max_avg_subtree(t), 2)

    def test_all_negative_value(self):
        """
        -4
          \
          -2
            \
             0

        -2 => -2 / 2 = -1
        -4 => (-4 + -2) / 2 = -3
        """
        t = Node(-4, Node(-2, Node(0)))
        self.assertEqual(max_avg_subtree(t), -2)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 20, 2019 \[Medium\] Smallest Missing Positive Number from an Unsorted Array
---
> **Question:** You are given an unsorted array with both positive and negative elements. You have to find the smallest positive number missing from the array in O(n) time using constant extra space. You can modify the original array.

**Example 1:**
```py
Input: {2, 3, 7, 6, 8, -1, -10, 15}
Output: 1
```

**Example 2:**
```py
Input: { 2, 3, -7, 6, 8, 1, -10, 15 }
Output: 4
```

**Example 3:**
```py
Input: {1, 1, 0, -1, -2}
Output: 2 
```

**My thoughts:** The main idea is to focus on all postive numbers and use its value as index. By marking the corresponding value as negative will allow us to find which number is covered already. Then the first positive number not being covered is what we are looking for.

For example, 

- Step 1: `[2, 3, 7, 6, 8, 1, -10, 15]` will be partitioned into `[2, 3, 7, 6, 8, 15, 1, -10]` and we ignore the negative part: `[2, 3, 7, 6, 8, 15, 1]`. 
- Step 2: ignore number that is too large, and mark corresponding value as negative `[2, 3, 7, 6, 8, 15, 1]`. Suitable values are `[1, 2, 3, 6, 7]` will map to index `[0, 1, 2, 5, 6]`. After that we mark corresponding val as negative: `[-2, -3, -7, 6, 8, -15, -1]`
- Step 3: the first positive's index is 3, which means 4 is missing.


**Solution:** [https://repl.it/@trsong/Smallest-Missing-Positive-Number-from-an-Unsorted-Array](https://repl.it/@trsong/Smallest-Missing-Positive-Number-from-an-Unsorted-Array)
```py
import unittest

def find_missing_positive(nums):
    if not nums:
        return 1

    # Step 1: Partition the array into positive part + non-positive part
    n = len(nums)
    i = 0 
    j = n - 1
    while i <= j:
        if nums[i] > 0:
            i += 1
        elif nums[j] <= 0:
            j -= 1
        else:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1

    # Step 2: Treat the first positive part as an array, and use val as index
    # val is between 0 and new_len - 1 which represents 1 to new_len, we turn corresponding number into negative
    new_len = i
    for i in xrange(new_len):
        val_as_index = abs(nums[i])
        if val_as_index - 1 < new_len and nums[val_as_index - 1] > 0:
            nums[val_as_index - 1] *= -1

    # Step 3: Find first index that is positive. Then + 1 will give the missing number
    for i in xrange(new_len):
        if nums[i] > 0:
            return i+1
    
    # if numbers are consecutive, meaning all number between 1 and new_len is covered, 
    # then new_len + 1 should be the missing number
    return new_len + 1


class FindMissingPositiveSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(find_missing_positive([2, 3, 7, 6, 8, -1, -10, 15]), 1)
    
    def test_example2(self):
        self.assertEqual(find_missing_positive([2, 3, -7, 6, 8, 1, -10, 15]), 4)

    def test_example3(self):
        self.assertEqual(find_missing_positive([1, 1, 0, -1, -2]), 2)

    def test_consecutive_array1(self):
        self.assertEqual(find_missing_positive([1, 2, 3]), 4)

    def test_consecutive_array2(self):
        self.assertEqual(find_missing_positive([-1, 0, 1]), 2)

    def test_non_positive_array(self):
        self.assertEqual(find_missing_positive([-5, -3, -1]), 1)

    def test_missing_multiple_positive_numbers(self):
        self.assertEqual(find_missing_positive([ 7, 8, 9, 10,-4, 1, 2, 3, 5,]), 4)

    def test_empty_array(self):
        self.assertEqual(find_missing_positive([]), 1)

    def test_negative_array(self):
        self.assertEqual(find_missing_positive([-1, -2, -3]), 1)

    def test_array_with_duplicated_number(self):
        self.assertEqual(find_missing_positive([1, 1, 2, 2, 3, 3, 5, 5, 6, -1, -1]), 4)

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 19, 2019 LC 987 \[Medium\] Vertical Order Traversal of a Binary Tree
---
> **Question:** Given a binary tree, return the vertical order traversal of its nodes' values. (ie, from top to bottom, column by column).
>
> If two nodes are in the same row and column, the order should be from left to right.

**Example 1:**
```py
Given binary tree:

    3
   / \
  9  20
    /  \
   15   7

return its vertical order traversal as:

[
  [9],
  [3,15],
  [20],
  [7]
]
```

**Example 2:**
```py
Given binary tree:

    _3_
   /   \
  9    20
 / \   / \
4   5 2   7

return its vertical order traversal as:

[
  [4],
  [9],
  [3,5,2],
  [20],
  [7]
]
```

**My thoughts:** Treat root node as postion 0, when move to left child positon - 1, or position + 1 for right child. Then use BFS to scan node from top to bottom to enforce vertical order of traversal.

**Solution with BFS:** [https://repl.it/@trsong/Vertical-Order-Traversal-of-a-Binary-Tree](https://repl.it/@trsong/Vertical-Order-Traversal-of-a-Binary-Tree)
```py
import unittest

class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def vertical_traversal(tree):
    if not tree:
        return []
    lo = hi = 0
    position_map = {}
    queue = [(tree, 0)]
    while queue:
        for _ in xrange(len(queue)):
            node, pos = queue.pop(0)
            lo = min(lo, pos)
            hi = max(hi, pos)

            if pos not in position_map:
                position_map[pos] = []

            position_map[pos].append(node.val)

            if node.left:
                queue.append((node.left, pos - 1))

            if node.right:
                queue.append((node.right, pos + 1))

    res = []
    for pos in xrange(lo, hi + 1):
        res.append(position_map[pos])

    return res


class VerticalTraversalSpec(unittest.TestCase):
    def test_example1(self):
        """
         3
        / \
       9  20
         /  \
        15   7
        """
        t = Node(3, Node(9), Node(20, Node(15), Node(7)))
        self.assertEqual(vertical_traversal(t), [
            [9],
            [3,15],
            [20],
            [7]
        ])
    
    def test_example2(self):
        """
            _3_
           /   \
          9    20
         / \   / \
        4   5 2   7
        """
        t9 = Node(9, Node(4), Node(5))
        t20 = Node(20, Node(2), Node(7))
        t = Node(3, t9, t20)

        self.assertEqual(vertical_traversal(t), [
            [4],
            [9],
            [3,5,2],
            [20],
            [7]
        ])

    def test_empty_tree(self):
        self.assertEqual(vertical_traversal(None), [])

    def test_left_heavy_tree(self):
        """
            1
           / \
          2   3
         / \   \
        4   5   6
        """
        t2 = Node(2, Node(4), Node(5))
        t3 = Node(3, right=Node(6))
        t = Node(1, t2, t3)
        self.assertEqual(vertical_traversal(t), [
            [4],
            [2],
            [1,5],
            [3],
            [6]
        ])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 18, 2019 \[Medium\] Max Value of Coins to Collect in a Matrix
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

**Solution with DP:** [https://repl.it/@trsong/Max-Value-of-Coins-to-Collect-in-a-Matrix](https://repl.it/@trsong/Max-Value-of-Coins-to-Collect-in-a-Matrix)
```py
import unittest

def max_coins(grid):
    if not grid or not grid[0]:
        return 0
    n, m = len(grid), len(grid[0])

    for r in xrange(n):
        for c in xrange(m):
            left = grid[r][c-1] if c > 0 else 0
            top = grid[r-1][c] if r > 0 else 0
            grid[r][c] += max(left, top)

    return grid[n-1][m-1]


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

### Sep 17, 2019 LC 296 \[Hard\] Best Meeting Point
---
> **Question:** A group of two or more people wants to meet and minimize the total travel distance. You are given a 2D grid of values 0 or 1, where each 1 marks the home of someone in the group. The distance is calculated using *Manhattan Distance*, where `distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|`.
> 
> Hint: Try to solve it in one dimension first. How can this solution apply to the two dimension case?

**Example:**

```py
Input: 

1 - 0 - 0 - 0 - 1
|   |   |   |   |
0 - 0 - 0 - 0 - 0
|   |   |   |   |
0 - 0 - 1 - 0 - 0

Output: 6 

Explanation: Given three people living at (0,0), (0,4), and (2,2):
             The point (0,2) is an ideal meeting point, as the total travel distance 
             of 2+2+2=6 is minimal. So return 6.
```

**My thoughts:** Exactly as the hint mentioned, let's first check the 1D case. 

Example 1:
If the array look like `[0, 1, 0, 0, 0, 1, 0, 0]` then by the definition of Manhattan Distance, any location between two 1s should be optimal. ie. all x spot in `[0, 1, x, x, x, 1, 0, 0]`

Example2:
If the array look like `[0, 1, 0, 1, 0, 1, 0, 0, 1]`, then we can reduce the result to  `[0, x, x, x, x, 1, 0, 0, 1]` and `[0, 0, 0, 1, x, 1, 0, 0, 0]` where x is the target spots. We shrink the optimal to the x spot in `[0, 0, 0, 1, x, 1, 0, 0, 0]`. 

So if we have 2D array, then notice that the target position's (x, y) co-ordinates are indepent of each other, ie, x and y won't affect each other. Why? Because by definition of *Manhattan Distance*, `distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|`. Suppose we have an optimal positon x, y. Then `total distance = all projected x-distance + all projected y-distance`. 

Example 3:
Suppose we use the example from the question body, our projected_row is `[1, 0, 1, 0, 1]` which gives best meeting distance 4, and projected_col is `[1, 0, 1]` which gives best meeting distance 2. Therefore the total best meeting distance equals `4 + 2 = 6`

**Solution:** [https://repl.it/@trsong/Best-Meeting-Point](https://repl.it/@trsong/Best-Meeting-Point)
```py
import unittest

def best_meeting_distance_1D(arr):
    i, j = 0, len(arr) - 1
    res = 0
    while True:
        if i >= j:
            break
        elif arr[i] > 0 and arr[j] > 0:
            res += j - i
            arr[i] -= 1
            arr[j] -= 1
        elif arr[i] == 0:
            i += 1
        elif arr[j] == 0:
            j -= 1
    return res
       

def best_meeting_distance(grid):
    if not grid or not grid[0]:
        return 0

    n, m = len(grid), len(grid[0])
    projected_row = [0] * m
    projected_col = [0] * n
    
    for r in xrange(n):
        for c in xrange(m):
            if grid[r][c]:
                projected_row[c] += 1
                projected_col[r] += 1

    return best_meeting_distance_1D(projected_row) + best_meeting_distance_1D(projected_col)


class BestMeetingPointSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(best_meeting_distance([
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ]), 6) # best position at [0, 2]
    
    def test_1D_array(self):
        self.assertEqual(best_meeting_distance([
            [1, 0, 1, 0, 0, 1]
        ]), 5) # best position at index 2
    
    def test_a_city_with_no_one(self):
        self.assertEqual(best_meeting_distance([
            [0, 0, 0],
            [0, 0, 0],
        ]), 0)

    def test_empty_grid(self):
       self.assertEqual(best_meeting_distance([
            []
        ]), 0)

    def test_even_number_of_points(self):
        self.assertEqual(best_meeting_distance([
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ]), 17) # best distance = x-distance + y-distance = 8 + 9 = 17. Best position at [3, 2]

    def test_odd_number_of_points(self):
        self.assertEqual(best_meeting_distance([
            [1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0]
        ]), 24) # best distance = x-distance + y-distance = 10 + 14 = 24. 


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 16, 2019 LC 317 \[Hard\] Shortest Distance from All Buildings
---
> **Question:** You want to build a house on an empty land which reaches all buildings in the shortest amount of distance. You can only move up, down, left and right. You are given a 2D grid of values 0, 1 or 2, where:
>
> - Each 0 marks an empty land which you can pass by freely.
> - Each 1 marks a building which you cannot pass through.
> - Each 2 marks an obstacle which you cannot pass through.
> 
> **Note:**
There will be at least one building. If it is not possible to build such house according to the above rules, return -1.

**Example:**

```py
Input: [[1,0,2,0,1],[0,0,0,0,0],[0,0,1,0,0]]

1 - 0 - 2 - 0 - 1
|   |   |   |   |
0 - 0 - 0 - 0 - 0
|   |   |   |   |
0 - 0 - 1 - 0 - 0

Output: 7 

Explanation: Given three buildings at (0,0), (0,4), (2,2), and an obstacle at (0,2),
             the point (1,2) is an ideal empty land to build a house, as the total 
             travel distance of 3+3+1=7 is minimal. So return 7.
```

**My thoughts:** There is no easy way to find the shortest distance to all building. Due to the fact that obstacle is unpredictable. There could be exponentially many different situations that obstacle can affect our shortest path. And sometimes it might simply just block the way to some building which cause the problem to short-circuit and return -1. 

Thus, what we can do for this problem is to run BFS on each building and calculate the accumulated/aggregated distance to current building for EACH empty land. And once done that, simple iterate through the aggregated distance array to find mimimum distance which will be the answer.

**Solution with BFS:** [https://repl.it/@trsong/Shortest-Distance-from-All-Buildings](https://repl.it/@trsong/Shortest-Distance-from-All-Buildings)
```py
import unittest
import sys


def shortest_distance(grid):
    if not grid or not grid[0]:
        return -1

    n, m = len(grid), len(grid[0])
    buildings = [(r, c) for r in xrange(n) for c in xrange(m) if grid[r][c] == 1]

    if not buildings:
        return -1

    num_building = len(buildings)
    accu_distance = [[0 for _ in xrange(m)] for _ in xrange(n)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for building in buildings:
        reached_building = 0
        distance = 0
        queue = [building]
        visited = [[False for _ in xrange(m)] for _ in xrange(n)]
        while queue:
            for _ in xrange(len(queue)):
                r, c = queue.pop(0)
                if visited[r][c]:
                    continue

                accu_distance[r][c] += distance
                visited[r][c] = True
                if grid[r][c] == 1:
                    reached_building += 1

                for d in directions:
                    nbr_r, nbr_c = r + d[0], c + d[1]
                    if 0 <= nbr_r < n and 0 <= nbr_c < m and not visited[nbr_r][nbr_c] and grid[nbr_r][nbr_c] != 2:
                        queue.append((nbr_r, nbr_c))
            distance += 1

        if reached_building != num_building:
            # Check if all building are connected
            return -1


    min_dist = sys.maxint
    for r in xrange(n):
        for c in xrange(m):
            if grid[r][c] == 0:
                min_dist = min(min_dist, accu_distance[r][c])

    return min_dist


class ShortestDistanceSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(shortest_distance([
            [1, 0, 2, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ]), 7)  # target location is [1, 2] which has distance 3 + 3 + 1 = 7

    def test_inaccessible_buildings(self):
        self.assertEqual(shortest_distance([
            [1, 0, 0],
            [1, 2, 2],
            [2, 1, 1]
        ]), -1)

    def test_no_building_at_all(self):
        self.assertEqual(shortest_distance([
            [0, 2, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]), -1)

    def test_empty_grid(self):
        self.assertEqual(shortest_distance([]), -1)

    def test_building_on_same_line(self):
        self.assertEqual(shortest_distance([
            [2, 1, 0, 1, 0, 0, 1]  # target is at index 2, which has distance = 1 + 1 + 4
        ]), 6)

    def test_multiple_road_same_distance(self):
        self.assertEqual(shortest_distance([
            [0, 1, 0],
            [1, 2, 0],
            [2, 1, 0]
        ]), 7)  # either top-left or top-right will give 7


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 15, 2019 LC 1014 \[Medium\] Best Sightseeing Pair
---
> **Question:** Given an array `A` of positive integers, `A[i]` represents the value of the i-th sightseeing spot, and two sightseeing spots `i` and `j` have distance `j - i` between them.
>
> The score of a pair (`i < j`) of sightseeing spots is (`A[i] + A[j] + i - j`) : the sum of the values of the sightseeing spots, minus the distance between them.
>
> Return the maximum score of a pair of sightseeing spots.

**Example:**

```py
Input: [8,1,5,2,6]
Output: 11
Explanation: i = 0, j = 2, A[i] + A[j] + i - j = 8 + 5 + 0 - 2 = 11
```

**My thoughts:** You probably don't even notice, but the problem already presents a hint in the question. So to say, `A[i] + A[j] - (j - i) = A[i] + A[j] + i - j`, if we re-arrange the terms even further, we can get `(A[i] + i) + (A[j] - j)`. Remember we want to maximize `(A[i] + i) + (A[j] - j)`. Notice that when we iterate throught the list along the way, before process `A[j]` we should've seen `A[i]` and `i` already. Thus we can store the max of `(A[i] + i)` so far and plus `(A[j] - j)` to get max along the way. This question is just a variant of selling stock problem. [https://trsong.github.io/python/java/2019/05/01/DailyQuestions/#june-4-2019-easy-sell-stock](https://trsong.github.io/python/java/2019/05/01/DailyQuestions/#june-4-2019-easy-sell-stock)

**Solution:** [https://repl.it/@trsong/Best-Sightseeing-Pair](https://repl.it/@trsong/Best-Sightseeing-Pair)
```py
import unittest

def max_score_sightseeing_pair(A):
    max_so_far = A[0] + 0
    res = 0
    for j in xrange(1, len(A)):
        res = max(res, max_so_far + A[j] - j)
        max_so_far = max(max_so_far, A[j] + j)
    return res


class MaxScoreSightseeingPairSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(max_score_sightseeing_pair([8, 1, 5, 2, 6]), 11) # i = 0, j = 2, A[i] + A[j] + i - j = 8 + 5 + 0 - 2 = 11

    def test_two_high_value_spots(self):
        self.assertEqual(max_score_sightseeing_pair([1, 9, 1, 10]), 17) # i = 1, j = 3, 9 + 10 + 1 - 3 = 17

    def test_decreasing_value(self):
        self.assertEqual(max_score_sightseeing_pair([3, 1, 1, 1]), 3) # i = 0, j = 1, 3 + 1 + 0 - 1 = 3

    def test_increasing_value(self):
        self.assertEqual(max_score_sightseeing_pair([1, 2, 4, 8]), 11) # i = 2, j = 3, 4 + 8 + 2 - 3 = 11

    def test_tie(self):
        self.assertEqual(max_score_sightseeing_pair([2, 2, 2, 2]), 3) # i = 0, j = 1, 2 + 2 + 0 - 1 = 3

    def test_two_elements(self):
        self.assertEqual(max_score_sightseeing_pair([5, 4]), 8) # i = 0, j = 1, 5 + 4 + 0 - 1 = 8


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 14, 2019 \[Medium\] Unique Prefix
---
> **Question:** Given a list of words, return the shortest unique prefix of each word. 

**Example:**
```py
Given the list:
dog
cat
apple
apricot
fish

Return the list:
d
c
app
apr
f
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

**Solution with Trie:** [https://repl.it/@trsong/Unique-Prefix](https://repl.it/@trsong/Unique-Prefix)
```py
import unittest

class Trie(object):
    TRIE_SIZE = 26
    def __init__(self):
        self.is_end = False
        self.count = 0
        self.children = None

    def has_word(self, word):
        t = self
        for char in word:
            ord_char = ord(char) - ord('a')
            if not t or not t.children:
                return False
            t = t.children[ord_char]
        return t and t.is_end
    
    def insert(self, word):
        if self.has_word(word): return
        t = self
        for char in word:
            ord_char = ord(char) - ord('a')
            if not t.children:
                t.children = [None] * Trie.TRIE_SIZE
            
            if not t.children[ord_char]:
                t.children[ord_char] = Trie()
            t.count += 1 # <-- Make sure to set the last node's count to be 1
            t = t.children[ord_char]
        t.count = 1
        t.is_end = True

    def find_prefix(self, word):
        t = self
        end = 0
        for char in word:
            ord_char = ord(char) - ord('a')
            if t.count == 1:
                break
            t = t.children[ord_char]
            end += 1
        return word[:end]


def unique_prefix(words):
    trie = Trie()
    for word in words:
        trie.insert(word)
    return map(lambda w: trie.find_prefix(w), words)


class UniquePrefixSpec(unittest.TestCase):
    """
    Assumption I made for this question:
    1. It's possible to have words being prefix of another; If that's the case, then the prefix is itself
    2. Empty word's prefix is empty word itself
    3. Duplicated words are treated as one and should have same unique prefix
    4. All letters in a word are in lower case with no whitespaces
    """
    def test_example(self):
        words = ['dog', 'cat', 'apple', 'apricot', 'fish']
        expected = ['d', 'c', 'app', 'apr', 'f']
        self.assertEqual(unique_prefix(words), expected)
    
    def test_prefix_word_of_another(self):
        words = ['greed', 'greedis', 'greedisgood', 'greeting']
        expected = ['greed', 'greedis', 'greedisg', 'greet']
        self.assertEqual(unique_prefix(words), expected)

    def test_empty_word(self):
        words = ['', 'a', 'alpha', 'aztec']
        expected = ['', 'a', 'al', 'az']
        self.assertEqual(unique_prefix(words), expected)

    def test_duplicated_words(self):
        words = ['a', 'a', 'applies', 'apt', 'apt', 'applies', 'apple', 'apple']
        expected = ['a', 'a', 'appli', 'apt', 'apt', 'appli', 'apple', 'apple']
        self.assertEqual(unique_prefix(words), expected)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 13, 2019 \[Easy\] Tree Isomorphism Problem
---
> **Question:** Write a function to detect if two trees are isomorphic. Two trees are called isomorphic if one of them can be obtained from other by a series of flips, i.e. by swapping left and right children of a number of nodes. Any number of nodes at any level can have their children swapped. Two empty trees are isomorphic.


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

**My thoughts:** If current tree value match  , only two situations would occur:
1. T1.left match T2.left and T1.right match T2.right
2. T1.left match T2.right and T1.right match T2.left 

**Solution with Recursion:** [https://repl.it/@trsong/Tree-Isomorphism-Problem](https://repl.it/@trsong/Tree-Isomorphism-Problem)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_isomorphic(t1, t2):
    if not t1 and not t2: return True
    if not t1 or not t2: return False
    if t1.val != t2.val: return False
    if is_isomorphic(t1.left, t2.left) and is_isomorphic(t1.right, t2.right): return True
    if is_isomorphic(t1.left, t2.right) and is_isomorphic(t1.right, t2.left): return True
    return False

class IsIsomorphicSpec(unittest.TestCase):
    def test_example(self):
        """
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
        """
        p5 = TreeNode(5, TreeNode(7), TreeNode(8))
        p2 = TreeNode(2, TreeNode(4), p5)
        p3 = TreeNode(3, TreeNode(6))
        p1 = TreeNode(1, p2, p3)

        q5 = TreeNode(5, TreeNode(8), TreeNode(7))
        q2 = TreeNode(2, TreeNode(4), q5)
        q3 = TreeNode(3, right=TreeNode(6))
        q1 = TreeNode(1, q3, q2)

        self.assertTrue(is_isomorphic(p1, q1))

    def test_empty_trees(self):
        self.assertTrue(is_isomorphic(None, None))

    def test_empty_vs_nonempty_trees(self):
        self.assertFalse(is_isomorphic(None, TreeNode(1)))

    def test_same_tree_val(self):
        """
        Tree1:
        1
         \
          1
         /
        1 

        Tree2:
            1
           /
          1
           \
            1
        """
        t1 = TreeNode(1, right=TreeNode(1, TreeNode(1)))
        t2 = TreeNode(1, TreeNode(1, right=TreeNode(1)))
        self.assertTrue(is_isomorphic(t1, t2))


    def test_same_val_yet_not_isomorphic(self):
        """
        Tree1:
          1
         / \
        1   1
           / \
          1   1

        Tree2:
            1
           / \
          1   1
         /     \
        1       1
        """
        t1 = TreeNode(1, TreeNode(1, TreeNode(1), TreeNode(1)))
        t2 = TreeNode(1, TreeNode(1, TreeNode(1)), TreeNode(1, right=TreeNode(1)))
        self.assertFalse(is_isomorphic(t1, t2))       


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 12, 2019 \[Medium\] Favorite Genres
---
> **Question:** Given a map Map<String, List<String>> userMap, where the key is a username and the value is a list of user's songs.
Also given a map Map<String, List<String>> genreMap, where the key is a genre and the value is a list of songs belonging to this genre.
The task is to return a map Map<String, List<String>>, where the key is a username and the value is a list of the user's favorite genres. Favorite genre is a genre with the most song.

**Example 1:**
```py
Input:
userMap = {  
   "David": ["song1", "song2", "song3", "song4", "song8"],
   "Emma":  ["song5", "song6", "song7"]
},
genreMap = {  
   "Rock":    ["song1", "song3"],
   "Dubstep": ["song7"],
   "Techno":  ["song2", "song4"],
   "Pop":     ["song5", "song6"],
   "Jazz":    ["song8", "song9"]
}

Output: {  
   "David": ["Rock", "Techno"],
   "Emma":  ["Pop"]
}

Explanation:
David has 2 Rock, 2 Techno and 1 Jazz song. So he has 2 favorite genres.
Emma has 2 Pop and 1 Dubstep song. Pop is Emma's favorite genre.
```

**Example 2:**
```py
Input:
userMap = {  
   "David": ["song1", "song2"],
   "Emma":  ["song3", "song4"]
},
genreMap = {}

Output: {  
   "David": [],
   "Emma":  []
}
```

**Solution:** [https://repl.it/@trsong/Favorite-Genres](https://repl.it/@trsong/Favorite-Genres)
```py
import unittest

def favorite_genre(user_map, genre_map):
    song_to_genre_map = {}
    for genre, songs in genre_map.items():
        for song in songs:
            if song not in song_to_genre_map:
                song_to_genre_map[song] = []
            song_to_genre_map[song].append(genre)

    res = {}
    for user, songs in user_map.items():
        genere_histogram = {}
        for song in songs:
            if song in song_to_genre_map:
                for genre in song_to_genre_map[song]:
                    genere_histogram[genre] = genere_histogram.get(genre, 0) + 1
        fav_genres = []
        max_count = 0
        for genre, count in genere_histogram.items():
            if count > max_count:
                fav_genres = [genre]
                max_count = count
            elif count == max_count:
                fav_genres.append(genre)
        res[user] = fav_genres

    return res


class FavoriteGenreSpec(unittest.TestCase):
    def assert_map(self, map1, map2):
        for _, values in map1.items():
            values.sort()
        for _, values in map2.items():
            values.sort()
        self.assertEqual(map1, map2)


    def test_example1(self):
        user_map = {
            "David": ["song1", "song2", "song3", "song4", "song8"],
            "Emma": ["song5", "song6", "song7"]
            }
        genre_map = {
            "Rock": ["song1", "song3"],
            "Dubstep": ["song7"],
            "Techno": ["song2", "song4"],
            "Pop": ["song5", "song6"],
            "Jazz": ["song8", "song9"]
        }
        expected = {
            "David": ["Rock", "Techno"],
            "Emma": ["Pop"]
        }
        self.assert_map(favorite_genre(user_map, genre_map), expected)

    def test_example2(self):
        user_map = {
            "David": ["song1", "song2"],
            "Emma": ["song3", "song4"]
        }
        genre_map = {}
        expected = {
            "David": [],
            "Emma": []
        }
        self.assert_map(favorite_genre(user_map, genre_map), expected)

    def test_same_song_with_multiple_genres(self):
        user_map = {
            "David": ["song1", "song2"],
            "Emma": ["song3", "song4"]
        }
        genre_map = {
            "Rock": ["song1", "song3"],
            "Dubstep": ["song1"],
        }
        expected = {
            "David": ["Rock", "Dubstep"],
            "Emma": ["Rock"]
        }
        self.assert_map(favorite_genre(user_map, genre_map), expected)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 11, 2019 LC 89 \[Medium\] Generate Gray Code
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

**Solution:** [https://repl.it/@trsong/Generate-Gray-Code](https://repl.it/@trsong/Generate-Gray-Code)
```py
import unittest

def gray_code(n):
    res = [0] * (2 ** n)
    for depth in xrange(n):
        for i in xrange(2**depth - 1, -1, -1):
            append_zero = res[i] << 1
            append_one = append_zero + 1
            if i % 2 == 0:
                res[2*i] = append_zero
                res[2*i + 1] = append_one
            else:
                res[2*i] = append_one
                res[2*i + 1] = append_zero
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
    unittest.main(exit=False)
```

### Sep 10, 2019 \[Medium\] Matrix Rotation
---
> **Question:** Given an N by N matrix, rotate it by 90 degrees clockwise.
>
> For example, given the following matrix:

```py
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
 ```

> you should return:

```py
[[7, 4, 1],
 [8, 5, 2],
 [9, 6, 3]]
 ```

> Follow-up: What if you couldn't use any extra space?

**My thoughts:** There are two ways to solve this problem without using any extra space. First one is to flip matrix diagonally and vertically. Second one is to move element one by one in spiral order: `left->top, bottom->left, right->bottom, top->right`. 

**Solution with Matrix Flip:** [https://repl.it/@trsong/Matrix-Rotation](https://repl.it/@trsong/Matrix-Rotation)
```py
import unittest

def matrix_rotation(matrix):
    """
    Step1: Flip Diagonally: (r, c) -> (c, r)
    [[1, 2],
     [3, 4]]
    =>
    [[1, 3],
     [2, 4]]
     
    Step2: Flip Vertically: (r, c) -> (r, n-1-c)
    [[1, 3],
     [2, 4]]
    =>
    [[3, 1],
     [4, 2]]
    """
    n = len(matrix)
    for r in xrange(n):
        for c in xrange(r):
            matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]

    for row in matrix:
        for c in xrange(n/2):
            row[c], row[n-1-c] = row[n-1-c], row[c]

    return matrix

class MatrixRotationSpec(unittest.TestCase):
    def test_empty_matrix(self):
        self.assertEqual(matrix_rotation([]), [])

    def test_size_one_matrix(self):
        self.assertEqual(matrix_rotation([[1]]), [[1]])

    def test_size_two_matrix(self):
        input_matrix = [
            [1, 2],
            [3, 4]
        ]
        expected_matrix = [
            [3, 1],
            [4, 2]
        ]
        self.assertEqual(matrix_rotation(input_matrix), expected_matrix)

    def test_size_three_matrix(self):
        input_matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        expected_matrix = [
            [7, 4, 1],
            [8, 5, 2],
            [9, 6, 3]
        ]
        self.assertEqual(matrix_rotation(input_matrix), expected_matrix)

    def test_size_four_matrix(self):
        input_matrix = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]
        expected_matrix = [
            [13, 9, 5, 1],
            [14, 10, 6, 2],
            [15, 11, 7, 3],
            [16, 12, 8, 4]
        ]
        self.assertEqual(matrix_rotation(input_matrix), expected_matrix)


if __name__ == '__main__':
    unittest.main(exit=False)
```

**Solution with Spiral Element Swap:** [https://repl.it/@trsong/Matrix-Rotation-Spiral](https://repl.it/@trsong/Matrix-Rotation-Spiral)
```py
def matrix_rotation(matrix):
    n = len(matrix)
    lo = 0
    hi = n - 1
    while lo < hi:
        for i in xrange(hi - lo):
            tmp = matrix[lo][lo+i]
            # left -> top
            matrix[lo][lo+i] = matrix[hi-i][lo]
            # bottom -> left
            matrix[hi-i][lo] = matrix[hi][hi-i]
            # right -> bottom
            matrix[hi][hi-i] = matrix[lo+i][hi]
            # top -> right
            matrix[lo+i][hi] = tmp
        lo += 1
        hi -= 1
    return matrix
```

### Sep 9, 2019 LC 403 \[Hard\] Frog Jump
---
> **Question:** A frog is crossing a river. The river is divided into x units and at each unit there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.
>
> Given a list of stones' positions (in units) in sorted ascending order, determine if the frog is able to cross the river by landing on the last stone. Initially, the frog is on the first stone and assume the first jump must be 1 unit.
> 
> If the frog's last jump was k units, then its next jump must be either k - 1, k, or k + 1 units. Note that the frog can only jump in the forward direction.

**Example 1:**

```py
[0,1,3,5,6,8,12,17]

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
[0,1,2,3,4,8,9,11]

Return false. There is no way to jump to the last stone as 
the gap between the 5th and 6th stone is too large.
```

**My thoughts:** This is a typical graph seaching problem (DAG to be specific), except that we not only need to store current stone, but the previous step as well. Besides that, so as not to exceed the time limit, we have to do tree pruning, i.e. do not visit same node with same step twice. e.g. `[0, 1, 2, 3, 4]` has path `0, 1, 2, 3, 4` and `0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4`.

**Solution with BFS and Pruning:** [https://repl.it/@trsong/Frog-Jump](https://repl.it/@trsong/Frog-Jump)
```py
import unittest

def can_cross(stones):
    stone_set = set(stones)
    stack = [(0, 0)]
    goal = stones[-1]
    visited = set()
    while stack:
        stone, step = stack.pop()
        if stone == goal:
            return True
        visited.add((stone, step))
        
        for next_jump_delta in [-1, 0, 1]:
            next_jump = step + next_jump_delta
            next_stone = stone + next_jump
            if next_stone >= stone and next_stone in stone_set and (next_stone, next_jump) not in visited:
                # check if next step is always forward, accessible and unvisited
                stack.append((next_stone, next_jump))
    return False

class CanCrossSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(can_cross([0, 1, 3, 5, 6, 8, 12, 17])) # step: 1(1), 2(3), 2(5), 3(8), 4(12), 5(17)

    def test_example2(self):
        self.assertFalse(can_cross([0, 1, 2, 3, 4, 8, 9, 11]))

    def test_unreachable_last_stone(self):
        self.assertFalse(can_cross([0, 1, 3, 6, 11]))

    def test_reachable_last_stone(self):
        self.assertTrue(can_cross([0, 1, 3, 6, 10]))

    def test_fall_into_water_in_the_middle(self):
        self.assertFalse(can_cross([0, 1, 10, 1000, 1000]))
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 8, 2019 \[Medium\] Evaluate Expression in Reverse Polish Notation
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

**Solution with Stack:** [https://repl.it/@trsong/Evaluate-Expression-in-Reverse-Polish-Notation](https://repl.it/@trsong/Evaluate-Expression-in-Reverse-Polish-Notation)
 ```py
 import unittest

class RPNExprEvaluator(object):
    op = {
        '+': lambda a,b: a + b,
        '-': lambda a,b: a - b,
        '*': lambda a,b: a * b,
        '/': lambda a,b: a / b
    }

    @staticmethod
    def run(tokens):
        stack = []
        for token in tokens:
            if type(token) == int:
                stack.append(token)
            else:
                operand2 = stack.pop()
                operand1 = stack.pop()
                op = RPNExprEvaluator.op[token]
                stack.append(op(operand1, operand2))
        return stack[-1] if stack else 0


class RPNExprEvaluatorSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(RPNExprEvaluator.run([5, 3, '+']), 8) # 5 + 3 = 8

    def test_example2(self):
        tokens = [15, 7, 1, 1, '+', '-', '/', 3, '*', 2, 1, 1, '+', '+', '-']
        self.assertEqual(RPNExprEvaluator.run(tokens), 5)

    def test_empty_tokens(self):
        self.assertEqual(RPNExprEvaluator.run([]), 0)

    def test_expression_contains_just_number(self):
        self.assertEqual(RPNExprEvaluator.run([42]), 42)
    
    def test_balanced_expression_tree(self):
        tokens = [7, 2, '-', 4, 1, '+', '*'] 
        self.assertEqual(RPNExprEvaluator.run(tokens), 25)  # (7 - 2) * (4 + 1) = 25
    
    def test_left_heavy_expression_tree(self):
        tokens = [6, 4, '-', 2, '/']  
        self.assertEqual(RPNExprEvaluator.run(tokens), 1) # (6 - 4) / 2 = 1

    def test_right_heavy_expression_tree(self):
        tokens = [2, 8, 2, '/', '*']
        self.assertEqual(RPNExprEvaluator.run(tokens), 8) # 2 * (8 / 2) = 8


if __name__ == '__main__':
    unittest.main(exit=False)
 ```

### Sep 7, 2019 LC 838 \[Medium\] Push Dominoes
---
> **Question:** Given a string with the initial condition of dominoes, where:
>
> - . represents that the domino is standing still
> - L represents that the domino is falling to the left side
> - R represents that the domino is falling to the right side
>
> Figure out the final position of the dominoes. If there are dominoes that get pushed on both ends, the force cancels out and that domino remains upright.

**Example 1:**
```py
Input:  "..R...L..R."
Output: "..RR.LL..RR"
```

**Example 2:**
```py
Input: "RR.L"
Output: "RR.L"
Explanation: The first domino expends no additional force on the second domino.
```

**Example 3:**
```py
Input: ".L.R...LR..L.."
Output: "LL.RR.LLRRLL.."
```

**My thoughts:** After some observation, you will find that 'R' and 'L' will always stay the same but '.' is depended on state of first non-dot on left and right:

- `.....L => LLLLLL`
- `R..... => RRRRRR`
- `L....L => LLLLLL`
- `R....R => RRRRRR`
- `R....L => RRRLLL`
- `R...L => RR.LL`

So, we just need to store the last non-dot domino state and compare w/ current domino will give the current state of domino.

**Solution:** [https://repl.it/@trsong/Push-Dominoes](https://repl.it/@trsong/Push-Dominoes)
```py
import unittest

def push_dominoes(dominoes):
    if len(dominoes) <= 1: return dominoes
    n = len(dominoes)
    last_falling_position = 0
    res = ['.'] * n
    for i, domino in enumerate(dominoes):
        if domino == '.':
            continue
       
        if domino == 'L' and dominoes[last_falling_position] == 'R':
            # R....L => RRRLLL
            # R...L => RR.LL
            j = last_falling_position
            k = i
            while j < k:
                res[j] = 'R'
                res[k] = 'L'
                j += 1
                k -= 1
        elif domino == dominoes[last_falling_position] or last_falling_position == 0 and domino == 'L':
            # .....L => LLLLLL
            # L....L => LLLLLL
            # R....R => RRRRRR
            for j in xrange(last_falling_position, i+1):
                res[j] = domino
        last_falling_position = i
            
    if dominoes[last_falling_position] == 'R':
        # R.... => RRRR
        for j in xrange(last_falling_position, n):
            res[j] = 'R'
    return ''.join(res)


class PushDominoeSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(push_dominoes("..R...L..R."), "..RR.LL..RR")

    def test_example2(self):
        self.assertEqual(push_dominoes("RR.L"), "RR.L")

    def test_example3(self):
        self.assertEqual(push_dominoes(".L.R...LR..L.."), "LL.RR.LLRRLL..")

    def test_empty_dominoes(self):
        self.assertEqual(push_dominoes(""), "")
    
    def test_one_domino(self):
        self.assertEqual(push_dominoes("."), ".")
        self.assertEqual(push_dominoes("L"), "L")
        self.assertEqual(push_dominoes("R"), "R")
    
    def test_all_fall_to_left(self):
        self.assertEqual(push_dominoes("...L"), "LLLL")
    
    def test_all_fall_to_right(self):
        self.assertEqual(push_dominoes("R..."), "RRRR")
    
    def test_left_right_and_right_left(self):
        self.assertEqual(push_dominoes(".L.RR..L."), "LL.RRRLL.")

    def test_right_left_and_left_right(self):
        self.assertEqual(push_dominoes(".R.LL..R."), ".R.LL..RR")
    
    def test_right_right_left_left(self):
        self.assertEqual(push_dominoes("R.R...LL"), "RRRR.LLL")


if __name__ == '__main__':
    unittest.main(exit=False)
```
### Sep 6, 2019 LC 287 \[Medium\] Find the Duplicate Number
---
> **Question:** You are given an array of length `n + 1` whose elements belong to the set `{1, 2, ..., n}`. By the pigeonhole principle, there must be a duplicate. Find it in linear time and space.

**My thoughts:** Use value as the 'next' element index which will form a loop evently. 

Why? Because the following scenarios will happen:

**Scenario 1:** If `a[i] != i for all i`, then since a[1] ... a[n] contains elements 1 to n, each time when interate to next index, one of the element within range 1 to n will be removed until no element is available and/or hit a previous used element and form a loop.  

**Scenario 2:** If `a[i] == i for all i > 0`, then as `a[0] != 0`, we will have a loop 0 -> a[0] -> a[0]

**Scenario 3:** If `a[i] == i for some i > 0`, then like scenario 2 we either we hit i evently or like scenario 1, for each iteration, we consume one element between 1 to n until all elements are used up and form a cycle. 

So we can use a fast and slow pointer to find the element when loop begins. 

**Solution with Fast and Slow Pointers:** [https://repl.it/@trsong/Find-the-Duplicate-Number](https://repl.it/@trsong/Find-the-Duplicate-Number)
```py
import unittest

def find_duplicate(nums):
    slow = fast = 0
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    
    p = 0
    while p != slow:
        p = nums[p]
        slow = nums[slow]
    return p


class FindDuplicateSpec(unittest.TestCase):
    def test_all_numbers_are_same(self):
        self.assertEqual(find_duplicate([2, 2, 2, 2, 2]), 2)

    def test_number_duplicate_twice(self):
        # index: 0 1 2 3 4 5 6
        # value: 2 6 4 1 3 1 5
        # chain: 0 -> 2 -> 4 -> 3 -> 1 -> 6 -> 5 -> 1
        #                            ^              ^
        self.assertEqual(find_duplicate([2, 6, 4, 1, 3, 1, 5]), 1)

    def test_rest_of_element_form_a_loop(self):
        # index: 0 1 2 3 4
        # value: 3 1 3 4 2
        # chain: 0 -> 3 -> 4 -> 2 -> 3
        #             ^              ^
        self.assertEqual(find_duplicate([3, 1, 3, 4, 2]), 3)

    def test_rest_of_element_are_sorted(self):
        # index: 0 1 2 3 4
        # value: 4 1 2 3 4
        # chain: 0 -> 4 -> 4
        #             ^    ^
        self.assertEqual(find_duplicate([4, 1, 2, 3, 4]), 4)
    
    def test_number_duplicate_more_than_twice(self):
        # index: 0 1 2 3 4 5 6 7 8 9
        # value: 2 5 9 6 9 3 8 9 7 1
        # chain: 0 -> 2 -> 9 -> 1 -> 5 -> 3 -> 6 -> 8 -> 7 -> 9
        #                  ^                                  ^
        self.assertEqual(find_duplicate([2, 5, 9, 6, 9, 3, 8, 9, 7, 1]), 9)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 5, 2019 \[Medium\] In-place Array Rotation
---
> **Question:** Write a function that rotates a list by k elements. 
> 
> For example, `[1, 2, 3, 4, 5, 6]` rotated by two becomes `[3, 4, 5, 6, 1, 2]`.
> 
> Try solving this without creating a copy of the list. How many swap or move operations do you need?

**My thoughts:** Test different examples until find pattern.

**Experiment 1:** How do you shift one element all the way till the end? You might want to do something like the following:

```py
[0, 1, 2, 3] => [1, 0, 2, 3] => [1, 2, 0, 3] => [1, 2, 3, 0]
 ^                  ^                  ^                  ^
```

**Experiment 2:** What about two elements?

```py
[0, 1, 2, 3] => [2, 3, 0, 1]
 ^  ^                  ^  ^
```

**Experiment 3:** If the array is able to fit in two windows with size k, we can shrink the problem into smaller one:

Suppose k = 3
```py
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] => [3, 4, 5, 0, 1, 2, 6, 7, 8, 9]
 ^  ^  ^                                    ^  ^  ^

The problem become swap the following array with window size 3
[0, 1, 2, 6, 7, 8, 9] => [6, 7, 8, 0, 1, 2, 9]
 ^  ^  ^                           ^  ^  ^

Until we are not able to fit two windows of size k
[0, 1, 2, 9]
 ^  ^  ^
```

**Experiment 4:** If we are not able to fit two windows of size k, we can shift the element backwards

Suppose k = 3:
```py
Since we are not able to fit two windows of size k
[0, 1, 2, 4, 5]
 ^  ^  ^

We instead shift the remaining elements backwards:
[0, 1, 2, 4, 5] => [0, 4, 5, 1, 2] 
          ^  ^         ^  ^

The problem size shrink again:
[0, 4, 5]
    ^  ^

As we cannot longer backward shift, we shift the remaining element forward
[0, 4, 5] => [4, 0, 5] => [4, 5, 0] 
 ^               ^               ^

that is:
[4, 5, 0, 1, 2] 
```

**Experiment 5:** Let's shift forwards and backwards multiple times with the following example

Suppose k = 5:
```py
k = 5, Forwards >>>
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] => [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10, 11, 12] 
 ^  ^  ^  ^  ^  ^                                             ^  ^  ^  ^  ^

k = 3, Backwards <<<
[0, 1, 2, 3, 4, 10, 11, 12] => [0, 1, 10, 11, 12, 2, 3, 4]
                ^^  ^^  ^^            ^^  ^^  ^^
that is [5, 6, 7, 8, 9, 0, 1, 10, 11, 12, 2, 3, 4]

k = 2, Forwards >>>
[0, 1, 10, 11, 12] => [10, 11, 0, 1, 12]
 ^  ^                          ^  ^
that is [5, 6, 7, 8, 9, 10, 11, 0, 1, 12, 2, 3, 4]

k = 1, Backwards <<<
[0, 1, 12] => [12, 0, 1] 
       ^^      ^^
that is [5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4]
```


 
**Solution with Two Pointers:** [https://repl.it/@trsong/In-place-Array-Rotation](https://repl.it/@trsong/In-place-Array-Rotation)
```py
import unittest

def swap_array(nums, pos1, pos2, size):
    for i in xrange(size):
        nums[pos1 + i], nums[pos2 + i] = nums[pos2 + i], nums[pos1 + i]
 

def rotate(nums, k):
    if not nums: return []
    k = k % len(nums)
    left = 0
    right = len(nums) - 1
    while left < right and k > 0:
        # Forward swap
        while left + 2*k - 1 <= right:
            swap_array(nums, left, left + k, k)
            left += k
        k = right - left + 1 - k
        if k == 0: break

        # Backward swap
        while right + 1 - 2*k >= left:
            swap_array(nums, right + 1 - k, right + 1 - 2*k, k)
            right -= k
        k = right - left + 1 - k
    return nums


class RotateSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(rotate([1, 2, 3, 4, 5, 6], k=2), [3, 4, 5, 6, 1, 2])

    def test_rotate_0_position(self):
        self.assertEqual(rotate([0, 1, 2, 3], k=0), [0, 1, 2, 3])

    def test_empty_array(self):
        self.assertEqual(rotate([], k=10), [])

    def test_shift_negative_position(self):
        self.assertEqual(rotate([0, 1, 2, 3], k=-1), [3, 0, 1, 2])

    def test_shift_more_than_array_size(self):
        self.assertEqual(rotate([1, 2, 3, 4, 5, 6], k=8), [3, 4, 5, 6, 1, 2])

    def test_multiple_round_of_forward_and_backward_shift(self):
        nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        k = 5
        expected = [5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4]
        self.assertEqual(rotate(nums, k), expected)
        

if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 4, 2019 \[Hard\] Reverse Words Keep Delimiters
---
> **Question:** Given a string and a set of delimiters, reverse the words in the string while maintaining the relative order of the delimiters. For example, given "hello/world:here", return "here/world:hello"
>
> Follow-up: Does your solution work for the following cases: "hello/world:here/", "hello//world:here"

**My thoughts:** Tokenize the word into multiple smaller words with each of them either being a normal word or delimiter word. While performing tokenization, memorize the index of normal words. After that just reverse order of normal words using that index of normal words. Finally combine all tokens to form a new word. 

**Solution:** [https://repl.it/@trsong/Reverse-Words-Keep-Delimiters](https://repl.it/@trsong/Reverse-Words-Keep-Delimiters)
```py
import unittest

def reverse_list_wit_indices(lst, indices):
    i = 0
    j = len(indices) - 1
    while i < j:
        index_i = indices[i]
        index_j = indices[j]
        lst[index_i], lst[index_j] = lst[index_j], lst[index_i]
        i += 1
        j -= 1

def reverse_words_keep_delimiters(s, delimiters):
    if not s: return ""
    delimiter_set = set(delimiters)
    res = []
    word_indices = []
    i = 0
    while i < len(s):
        tmp = []
        while i < len(s) and s[i] not in delimiter_set:
            tmp.append(s[i])
            i += 1
        if tmp:
            word_indices.append(len(res))
            res.append(tmp)
                
        tmp = []
        while i < len(s) and s[i] in delimiter_set:
            tmp.append(s[i])
            i += 1
        if tmp:
            res.append(tmp)

    reverse_list_wit_indices(res, word_indices)
    return ''.join(map(lambda lst: ''.join(lst), res))
     

class ReverseWordsKeepDelimiterSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(reverse_words_keep_delimiters("hello/world:here", ['/', ':']), "here/world:hello")
    
    def test_example2(self):
        self.assertEqual(reverse_words_keep_delimiters("hello/world:here/", ['/', ':']), "here/world:hello/")

    def test_example3(self):
        self.assertEqual(reverse_words_keep_delimiters("hello//world:here", ['/', ':']), "here//world:hello")

    def test_only_has_delimiters(self):
        self.assertEqual(reverse_words_keep_delimiters("--++--+++", ['-', '+']), "--++--+++")

    def test_without_delimiters(self):
        self.assertEqual(reverse_words_keep_delimiters("--++--+++", []), "--++--+++")
        self.assertEqual(reverse_words_keep_delimiters("--++--+++", ['a', 'b']), "--++--+++")

    def test_first_delimiter_then_word(self):
        self.assertEqual(reverse_words_keep_delimiters("///a/b", ['/']), "///b/a")
    
    def test_first_word_then_delimiter(self):
        self.assertEqual(reverse_words_keep_delimiters("a///b///", ['/']), "b///a///")


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Similar Question: LC 151 \[Medium\] Reverse Words in a String
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

**My thoughts:** Solution with space complexity O(n) can be easily achieved through two pointers iterate from the back of string and a buffer to store each word along the way. However in-place solution with O(1) space complexity using C++ requires some trick:

Suppose the original string is `"the sky is blue"`
1. reverse entire sentence. `"eulb si yks eht"`
2. reverse each word in that sentence. `"blue is sky the"`

**Solution with Two Pointers:** [https://repl.it/@trsong/Reverse-Words-in-a-String](https://repl.it/@trsong/Reverse-Words-in-a-String)
```py
import unittest

def reverse_word(s):
    if not s: return ""
    res = []
    i = j = len(s) - 1
    while j >= 0:
        if j >= 0 and s[j] == ' ':
            j -= 1
            i -= 1
        elif i >= 0 and s[i] != ' ':
            i -= 1
        else:
            for k in xrange(i+1, j+1):
                res.append(s[k])
            res.append(' ')
            j = i
    if res:
        # pop the last whitespace
        res.pop()
    return ''.join(res)


class ReverseWordSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(reverse_word("the sky is blue"), "blue is sky the")

    def test_example2(self):
        self.assertEqual(reverse_word("  hello world!  "), "world! hello")

    def test_example3(self):
        self.assertEqual(reverse_word("a good   example"), "example good a")

    def test_mutliple_whitespaces(self):
        self.assertEqual(reverse_word("   "), "")
        self.assertEqual(reverse_word(""), "")

    def test_even_number_of_words(self):
        self.assertEqual(reverse_word(" car cat"), "cat car")
        self.assertEqual(reverse_word("car cat "), "cat car")

    def test_no_whitespaces(self):
        self.assertEqual(reverse_word("asparagus"), "asparagus")


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 3, 2019 \[Medium\] Direction and Position Rule Verification
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

**Solution with DFS:** [https://repl.it/@trsong/Direction-and-Position-Rule-Verification](https://repl.it/@trsong/Direction-and-Position-Rule-Verification)
```py
import unittest

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
        

def DFS_check_connection(neighbors, start, end):
    visited = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur == end:
            return True

        if cur not in visited:
            if cur not in neighbors: continue
            stack.extend(neighbors[cur])
            visited.add(cur)
    return False
        

def direction_rule_validate(rules):
    direction_neighbor = { d:{} for d in [Direction.E, Direction.S, Direction.W, Direction.N] }
    for rule in rules:
        p1, directions, p2 = rule.split(' ')
        for d in directions:
            neighbors = direction_neighbor[d]
            # Before add edge (p1, p2), check connection between (p2, p1) to detect cycle
            if DFS_check_connection(neighbors, p2, p1):
                return False
            
            if p1 not in neighbors:
                neighbors[p1] = []
            neighbors[p1].append(p2)

            opposite_neighbor = direction_neighbor[Direction.get_opposite_direction(d)]
            if p2 not in opposite_neighbor:
                opposite_neighbor[p2] = []
            opposite_neighbor[p2].append(p1)
    return True


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

### Sep 2, 2019 LC 937 \[Easy\] Reorder Log Files
---
> **Question:** You have an array of logs. Each log is a space delimited string of words.
>
> For each log, the first word in each log is an alphanumeric identifier.  Then, either:
>
> - Each word after the identifier will consist only of lowercase letters, or;
> - Each word after the identifier will consist only of digits.
We will call these two varieties of logs letter-logs and digit-logs.  It is guaranteed that each log has at least one word after its identifier.
>
> Reorder the logs so that all of the letter-logs come before any digit-log.  The letter-logs are ordered lexicographically ignoring identifier, with the identifier used in case of ties.  The digit-logs should be put in their original order.
>
> Return the final order of the logs.

**Example:**

```py
Input: ["a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo"]
Output: ["g1 act car","a8 act zoo","ab1 off key dog","a1 9 2 3 1","zo4 4 7"]
```

**Solution with Customized Sort Function:** [https://repl.it/@trsong/Reorder-Log-Files](https://repl.it/@trsong/Reorder-Log-Files)
```py
import unittest

def is_number(char):
    return '0' <= char <= '9'

def compare_string(s1, begin1, end1, s2, begin2, end2):
    i = 0
    while begin1 + i < end1 and begin2 + i < end2:
        char1 = s1[begin1 + i]
        char2 = s2[begin2 + i]
        if char1 < char2:
            return -1
        elif char1 > char2:
            return 1
        i += 1
    
    len1 = end1 - begin1
    len2 = end2 - begin2
    if len1 < len2:
        return -1
    elif len1 > len2:
        return 1
    else:
        return 0

def compare(log1_with_index, log2_with_index):
    log1, pos1 = log1_with_index
    log2, pos2 = log2_with_index
    is_log1_digit = is_number(log1[pos1])
    is_log2_digit = is_number(log2[pos2])
    if is_log1_digit and is_log2_digit:
        return 0
    elif is_log1_digit:
        return 1
    elif is_log2_digit:
        return -1
    else:
        log_res = compare_string(log1, pos1, len(log1), log2, pos2, len(log2))
        if log_res != 0:
            return log_res
        else:
            return compare_string(log1, 0, pos1, log2, 0, pos2)

def reorder_log_files(logs):
    logs_with_index = map(lambda log: (log, log.index(' ') + 1), logs)
    logs_with_index.sort(cmp = compare)
    return map(lambda x: x[0], logs_with_index)


class ReorderLogFileSpec(unittest.TestCase):
    dlog1 = "a1 9 2 3 1"
    dlog2 = "zo4 4 7"
    llog1 = "g1 act car"
    llog2 = "ab1 off key dog"
    llog3 = "a8 act zoo"
    llog4 = "g1 act car jet jet jet"
    llog5 = "hhh1 act car"
    llog6 = "g1 act car jet"

    def test_example(self):
        self.assertEqual(
            reorder_log_files([self.dlog1, self.llog1, self.dlog2, self.llog2, self.llog3]), 
            [self.llog1,self.llog3, self.llog2, self.dlog1, self.dlog2])

    def test_empty_logs(self):
        self.assertEqual(reorder_log_files([]), [])

    def test_digit_logs_maintaining_same_order(self):
        self.assertEqual(reorder_log_files([self.dlog1, self.dlog2]), [self.dlog1, self.dlog2])
        self.assertEqual(reorder_log_files([self.dlog2, self.dlog1, self.dlog2]), [self.dlog2, self.dlog1, self.dlog2])

    def test_when_letter_logs_have_a_tie(self):
        self.assertEqual(
            reorder_log_files([self.llog6, self.llog4, self.llog1, self.llog5]), 
            [self.llog1, self.llog5, self.llog6, self.llog4])
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 1, 2019 LT 892 \[Medium\] Alien Dictionary
---
> **Question:** There is a new alien language which uses the latin alphabet. However, the order among letters are unknown to you. You receive a list of non-empty words from the dictionary, where words are sorted lexicographically by the rules of this new language. Derive the order of letters in this language.
>
> - You may assume all letters are in lowercase.
> 
> - You may assume that if a is a prefix of b, then a must appear before b in the given dictionary.
> 
> - If the order is invalid, return an empty string.
There may be multiple valid order of letters, return the smallest in normal lexicographical order

**Example 1:**
```py
Input: ["wrt", "wrf", "er", "ett", "rftt"]
Output: "wertf"
Explanation：
from "wrt" and "wrf", we can get 't'<'f'
from "wrt" and "er", we can get 'w'<'e'
from "er" and "ett", we can get 'r'<'t'
from "ett" and "rtff", we can get 'e'<'r'
So return "wertf"
```

**Example 2:**
```py
Input: ["z", "x"]
Output: "zx"
Explanation：
from "z" and "x"，we can get 'z' < 'x'
So return "zx"
```

**My thoughts:** As the alien letters are topologically sorted, we can just mimic what topological sort with numbers and try to find pattern.

Suppose the dictionary contains: `01234`. Then the words can be `023, 024, 12, 133, 2433`. Notice that we can only find the relative order by finding first unequal letters between consecutive words. eg.  `023, 024 => 3 < 4`.  `024, 12 => 0 < 1`.  `12, 133 => 2 < 3`

With relative relation, we can build a graph with each occurring letters being veteces and edge `(u, v)` represents `u < v`. If there exists a loop that means we have something like `a < b < c < a` and total order not exists. Otherwise we preform a topological sort to generate the total order which reveals the alien dictionary. 


**Solution with Topological Sort:** [https://repl.it/@trsong/Alien-Dictionary](https://repl.it/@trsong/Alien-Dictionary)
```py
import unittest

class NodeState(object):
    UNVISITED = 0
    VISITING = 1
    VISITED = 2

def alien_dict_order(words):
    neigbor = {}
    for i in xrange(1, len(words)):
        prev_word, cur_word = words[i-1], words[i]
        for j in xrange(min(len(prev_word), len(cur_word))):
            prev_word_char, cur_word_char = prev_word[j], cur_word[j]
            if prev_word_char != cur_word_char:
                if prev_word_char not in neigbor:
                    neigbor[prev_word_char] = []
                neigbor[prev_word_char].append(cur_word_char)
                break
    
    for key in neigbor:
        neigbor[key].sort()

    node_states = {}
    stack = []
    tsort_stack = []
    for word in words:
        for char in word:
            if char not in neigbor:
                neigbor[char] = []
            node_states[char] = NodeState.UNVISITED


    for node in sorted(neigbor.keys(), reverse=True):
        if node_states[node] != NodeState.VISITED:
            stack.append(node)

        while stack:
            cur = stack[-1]
            if node_states[cur] == NodeState.VISITING:
                node_states[cur] = NodeState.VISITED
            elif node_states[cur] == NodeState.UNVISITED: 
                node_states[cur] = NodeState.VISITING
                for nei in neigbor[cur]:
                    if node_states[nei] == NodeState.VISITING:
                        return ""
                    elif node_states[nei] == NodeState.UNVISITED:
                        stack.append(nei)
            else:
                tsort_stack.append(stack.pop())

    top_order = []
    while tsort_stack:
        top_order.append(tsort_stack.pop())
        
    return "".join(top_order)
            

class AlienDictOrderSpec(unittest.TestCase):
    def test_example1(self):
        # 01234
        # wertf
        # decode array result become 023, 024, 12, 133, 2433
        self.assertEqual(alien_dict_order(["wrt", "wrf", "er", "ett", "rftt"]), "wertf")

    def test_example2(self):
        self.assertEqual(alien_dict_order(["z", "x"]), "zx")

    def test_invalid_order(self):
        self.assertEqual(alien_dict_order(["a", "b", "a"]), "")

    def test_invalid_order2(self):
        # 012
        # abc
        # decode array result become 210, 211, 212, 012
        self.assertEqual(alien_dict_order(["cba", "cbb", "cbc", "abc"]), "")

    def test_invalid_order3(self):
        # 012
        # abc
        # decode array result become 10, 11, 211, 22, 20 
        self.assertEqual(alien_dict_order(["ba", "bb", "cbb", "cc", "ca"]), "")

    def test_valid_order(self):
        # 012
        # abc
        # decode array result become 01111, 122, 20
        self.assertEqual(alien_dict_order(["abbbb", "bcc", "ca"]), "abc")

    def test_valid_order2(self):
        # 0123
        # bdac
        # decode array result become 022, 2031, 2032, 320, 321
        self.assertEqual(alien_dict_order(["baa", "abcd", "abca", "cab", "cad"]), "bdac")

    def test_multiple_valid_result(self):
        self.assertEqual(alien_dict_order(["edcba"]), "abcde")

    def test_multiple_valid_result2(self):
        # 01
        # ab
        # cd
        self.assertEqual(alien_dict_order(["aa", "ab", "cc", "cd"]), "abcd")

    def test_multiple_valid_result3(self):
        # 01
        # ab
        #  c
        #  d
        self.assertEqual(alien_dict_order(["aaaaa", "aaad", "aab", "ac"]), "abcd")


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 31, 2019 \[Hard\] Encode and Decode Array of Strings
---
> **Question:** Given an array of string, write function "encode" to convert an array of strings into a single string and function "decode" to restore the original array.
> 
> Hint: string can be encoded as `<length>:<contents>`
> 
> Follow-up: what about array of integers, strings, and dictionaries?

**My thoughts:** There are many ways to encode/decode. Our solution use BEncode, the encoding used by BitTorrent. 

According to BEncode Wiki: [https://en.wikipedia.org/wiki/Bencode](https://en.wikipedia.org/wiki/Bencode) and specification of BitTorrent: [http://www.bittorrent.org/beps/bep_0003.html](http://www.bittorrent.org/beps/bep_0003.html). BEncode works as following:

- Integer num is encoded as `"i{num}e"`. eg. `42 => "i42e"`, `-32 => "i-32e"`, `0 => "i0e"`
- String s is encoded as `"{len(s)}:{s}"`. eg. `"abc" => "3:abc"`, `"" => "0:"`, `"s" => "1:s"`, `"doge" => "4:doge"`
- List lst is encoded as `"l{encode(lst[0])}{encode(lst[1])}....{encode(lst[n-1])}e"`.  e.g. `[] => "le"`, `[42, "cat"]` => `"li42e3:cate"`, `[[11], [22], [33]] => lli11eeli22eeli33eee`
- Dictionary d is encoded as `"d{encode(key1)}{encode(val1)}...{encode(key_n)}{encode(val_n)}e"` e.g. `{'bar': 'spam','foo': 42} => "d3:bar4:spam3:fooi42ee"`


**Solution with BEncode:** [https://repl.it/@trsong/Encode-and-Decode-Array-of-Strings](https://repl.it/@trsong/Encode-and-Decode-Array-of-Strings)
```py
import unittest

"""
Encode Utilities
"""
def encode_int(num):
    return ["i", str(num), "e"]

def encode_str(string):
    return [str(len(string)), ":", string]

def encode_list(lst):
    res = ["l"]
    for e in lst:
        res.extend(encode_func[type(e)](e))
    res.append("e")
    return res

def encode_dict(dicionary):
    res = ["d"]
    for k, v in sorted(dicionary.items()):
        res.extend(encode_str(k))
        res.extend(encode_func[type(v)](v))
    res.append("e")
    return res

encode_func = {
    int: encode_int,
    str: encode_str,
    list: encode_list,
    dict: encode_dict
}

"""
Decode Utilities
"""
def decode_int(encoded, pos):
    pos += 1
    new_pos = encoded.index('e', pos)
    num = int(encoded[pos:new_pos])
    return (num, new_pos + 1)

def decode_str(encoded, pos):
    colon = encoded.index(':', pos)
    n = int(encoded[pos:colon])
    str_start = colon + 1
    return (encoded[str_start:str_start + n], str_start + n)

def decode_list(encoded, pos):
    res = []
    pos += 1
    while encoded[pos] != 'e':
        val, new_pos = decode_func[encoded[pos]](encoded, pos)
        pos = new_pos
        res.append(val)
    return (res, pos + 1)

def decode_dict(encoded, pos):
    res = {}
    pos += 1
    while encoded[pos] != 'e':
        key, key_end_pos = decode_str(encoded, pos)
        val, val_end_pos = decode_func[encoded[key_end_pos]](encoded, key_end_pos)
        pos = val_end_pos
        res[key] = val
    return (res, pos + 1)

decode_func = {
    'l': decode_list,
    'd': decode_dict,
    'i': decode_int
}
decode_func.update({
    chr(ord('0') + i): decode_str for i in xrange(10) # 0-9 all use decode_str
})


class BEncode(object):
    @staticmethod
    def encode(obj):
        return "".join(encode_func[type(obj)](obj))

    
    @staticmethod
    def decode(encoded):
        obj, _ = decode_func[encoded[0]](encoded, 0)
        return obj


class BEncodeSpec(unittest.TestCase):
    def assert_BEncode_result(self, obj):
        encoded_str = BEncode.encode(obj)
        decoded_obj = BEncode.decode(encoded_str)
        self.assertEqual(decoded_obj, obj)

    def test_positive_integer(self):
        self.assert_BEncode_result(42)  # encode as "i42e"
    
    def test_negative_integer(self):
        self.assert_BEncode_result(-32)  # encode as "i-32e"

    def test_zero(self):
        self.assert_BEncode_result(0)  # encode as "i0e"

    def test_empty_string(self):
        self.assert_BEncode_result("")  # encode as "0:""

    def test_string_with_whitespaces(self):
        self.assert_BEncode_result(" ")  # encode as "1: "
        self.assert_BEncode_result(" a ")  # encode as "3: a "
        self.assert_BEncode_result("a b  c   1 2 3 ")  # encode as "15:a b  c   1 2 3 "

    def test_string_with_delimiter_characters(self):
        self.assert_BEncode_result("i42ei42e")  # encode as "8:i42ei42e"
        self.assert_BEncode_result("4:spam")  # encode as "6:4:spam"
        self.assert_BEncode_result("d3:bar4:spam3:fooi42ee")  # encode as "22:d3:bar4:spam3:fooi42ee"
    
    def test_string_with_special_characters(self):
        self.assert_BEncode_result("!@#$%^&*(){}[]|\;:'',.?/`~") # encode as "26:!@#$%^&*(){}[]|\;:'',.?/`~"

    def test_empty_list(self):
        self.assert_BEncode_result([])  # encode as "le"

    def test_list_of_empty_strings(self):
        self.assert_BEncode_result(["", "", ""])  # encode as "l0:0:0:e"

    def test_nested_empty_lists(self):
        self.assert_BEncode_result([[], [[]], [[[]]]]) # encoded as "llelleellleee"

    def test_list_of_strings(self):
        self.assert_BEncode_result(['a', '', 'abc']) # encode as "l1:a0:3:abce"

    def test_nested_lists(self):
        self.assert_BEncode_result([0, ["a", 1], "ab", [[2], "c"]]) # encode as "li0el1:ai1ee2:ablli2ee1:cee"

    def test_empty_dictionary(self):
        self.assert_BEncode_result({}) # encode as "de"

    def test_dictionary(self):
        self.assert_BEncode_result({
            'bar': 'spam',
            'foo': 42
        })  # encode as "d3:bar4:spam3:fooi42ee"

    def test_nested_dictionary(self):
        self.assert_BEncode_result([
            "s",
            42, 
            { 
                'a': 12,
                'list': [
                    'b', 
                    {
                        'c': [[1], 2, [3, [4]]],
                        'd': 12
                    }]
            }]) # encode as 'l1:si42ed1:ai12e4:listl1:bd1:clli1eei2eli3eli4eeee1:di12eeeee'
 

if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 30, 2019 LT 623 \[Hard\] K Edit Distance
---
> **Question:** Given a set of strings which just has lower case letters and a target string, output all the strings for each the edit distance with the target no greater than k.
You have the following 3 operations permitted on a word:
> - Insert a character
> - Delete a character
> - Replace a character

**Example 1:**
```py
Given words = ["abc", "abd", "abcd", "adc"] and target = "ac", k = 1
Return ["abc", "adc"]
Explanation:
- "abc" remove "b"
- "adc" remove "d"
```

**Example 2:**
```py
Given words = ["acc","abcd","ade","abbcd"] and target = "abc", k = 2
Return ["acc","abcd","ade","abbcd"]
Explanation:
- "acc" turns "c" into "b"
- "abcd" remove "d"
- "ade" turns "d" into "b" turns "e" into "c"
- "abbcd" gets rid of "b" and "d"
```

**My thoughts:** The brutal force way is to calculate the distance between each word and target and filter those qualified words. However, notice that word might have exactly the same prefix and that share the same DP array. So we can build a prefix tree that contains all words and calculate the DP array along the way.

**Solution with Trie and DFS:** [https://repl.it/@trsong/K-Edit-Distance](https://repl.it/@trsong/K-Edit-Distance)
```py
import unittest

class Trie(object):
    def __init__(self):
        self.count = 0
        self.word = None
        self.edit_distance_dp = None
        self.children = None

    def insert(self, word): 
        t = self
        for char in word:
            if not t.children:
                t.children = {}
            if char not in t.children:
                t.children[char] = Trie()
            t = t.children[char]
        t.count += 1
        t.word = word


def filter_k_edit_distance(words, target, k):
    trie = Trie()
    n = len(target)
    filtered_word = filter(lambda word: n - k <= len(word) <= n + k, words)
    for word in filtered_word:
        trie.insert(word)
        
    trie.edit_distance_dp = [i for i in xrange(n+1)] # edit distance between "" and target[:i] equals i (insert i letters)
    stack = [trie]
    res = []
    while stack:
        parent = stack.pop()

        parent_dp = parent.edit_distance_dp
        if parent.word is not None and parent_dp[n] <= k:
            res.extend([parent.word] * parent.count)

        if not parent.children:
            continue

        for char, child in parent.children.items():
            dp = [0] * (n+1)
            dp[0] = parent_dp[0] + 1
            for j in xrange(1, n+1):
                if char == target[j-1]:
                    dp[j] = parent_dp[j-1]
                else:
                    dp[j] = min(1 + parent_dp[j-1], 1 + dp[j-1], 1 + parent_dp[j])
            child.edit_distance_dp = dp
            stack.append(child)
    
    return res


class FilterKEditDistanceSpec(unittest.TestCase):
    def assert_k_distance_array(self, res, expected):
        self.assertEqual(sorted(res), sorted(expected))

    def test_example1(self):
        words =["abc", "abd", "abcd", "adc"] 
        target = "ac"
        k = 1
        expected = ["abc", "adc"]
        self.assert_k_distance_array(filter_k_edit_distance(words, target, k), expected)
    
    def test_example2(self):
        words = ["acc","abcd","ade","abbcd"]
        target = "abc"
        k = 2
        expected = ["acc","abcd","ade","abbcd"]
        self.assert_k_distance_array(filter_k_edit_distance(words, target, k), expected)

    def test_duplicated_words(self):
        words = ["a","b","a","c", "bb", "cc"]
        target = ""
        k = 1
        expected = ["a","b","a","c"]
        self.assert_k_distance_array(filter_k_edit_distance(words, target, k), expected)

    def test_empty_words(self):
        words = ["", "", "", "c", "bbbbb", "cccc"]
        target = "ab"
        k = 2
        expected = ["", "", "", "c"]
        self.assert_k_distance_array(filter_k_edit_distance(words, target, k), expected)

    def test_same_word(self):
        words = ["ab", "ab", "ab"]
        target = "ab"
        k = 1000
        expected = ["ab", "ab", "ab"]
        self.assert_k_distance_array(filter_k_edit_distance(words, target, k), expected)

    def test_unqualified_words(self):
        words = ["", "a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa"]
        target = "aaaaa"
        k = 2
        expected = ["aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa"]
        self.assert_k_distance_array(filter_k_edit_distance(words, target, k), expected)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 29, 2019 \[Easy\] Flip Bit to Get Longest Sequence of 1s
---
> **Question:** Given an integer, can you flip exactly one bit from a 0 to a 1 to get the longest sequence of 1s? Return the longest possible length of 1s after flip.

**Example:**
```py
Input: 183 (or binary: 10110111)
Output: 6
Explanation: 10110111 => 10111111. The longest sequence of 1s is of length 6.
```

**Solution:** [https://repl.it/@trsong/Flip-Bit-to-Get-Longest-Sequence-of-1s](https://repl.it/@trsong/Flip-Bit-to-Get-Longest-Sequence-of-1s)
```py
import unittest

def flip_bits(num):
    prev = 0
    cur = 0
    max_len = 0
    while num > 0:
        last_digit = num & 1
        if last_digit == 1:
            cur += 1
        else:
            max_len = max(max_len, cur + prev + 1)
            prev = cur
            cur = 0
        num >>= 1
    return max(max_len, cur + prev + 1)


class FlipBitSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(flip_bits(0b10110111), 6)  # 10110111 => 10111111

    def test_not_exist_ones(self):
        self.assertEqual(flip_bits(0), 1)  # 0 => 1

    def test_flip_last_digit(self):
        self.assertEqual(flip_bits(0b100110), 3)  # 100110 => 100111

    def test_three_zeros(self):
        self.assertEqual(flip_bits(0b1011110110111), 7)  # 1011110110111 => 1011111110111

    def test_one(self):
        self.assertEqual(flip_bits(1), 2)  # 01 => 11


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 28, 2019 LC 103 \[Medium\] Binary Tree Zigzag Level Order Traversal
---
> **Question:** Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

**For example:**
```py
Given following binary tree:
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]
```

**Solution with BFS and Two Stacks:** [https://repl.it/@trsong/Binary-Tree-Zigzag-Level-Order-Traversal](https://repl.it/@trsong/Binary-Tree-Zigzag-Level-Order-Traversal)
```py
import unittest

class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def zig_zag_level_order(root):
    if not root:
        return []
    stack = [root]
    res = []
    is_left_first = True
    while stack:
        level = []
        next_stack = []
        for _ in xrange(len(stack)):
            cur = stack.pop()
            if cur:
                level.append(cur.val)
                if is_left_first:
                    next_stack.append(cur.left)
                    next_stack.append(cur.right)
                else:
                    next_stack.append(cur.right)
                    next_stack.append(cur.left)
        stack = next_stack
        is_left_first ^= True # flip the boolean flag
        if level:
            res.append(level)
    return res


class ZigZagLevelOrderSpec(unittest.TestCase):
    def test_example(self):
        """
            3
           / \
          9  20
            /  \
           15   7
        """
        n20 = Node(20, Node(15), Node(7))
        n3 = Node(3, Node(9), n20)
        self.assertEqual(zig_zag_level_order(n3), [
            [3],
            [20, 9],
            [15, 7]
        ])
    
    def test_complete_tree(self):
        """
             1
           /   \
          3     2
         / \   /  
        4   5 6  
        """
        n3 = Node(3, Node(4), Node(5))
        n2 = Node(2, Node(6))
        n1 = Node(1, n3, n2)
        self.assertEqual(zig_zag_level_order(n1), [
            [1],
            [2, 3],
            [4, 5, 6]
        ])

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
        n3 = Node(3, right=Node(4, Node(7, right=Node(8))))
        n2 = Node(2, Node(5, right=Node(6, Node(9))))
        n1 = Node(1, n3, n2)
        self.assertEqual(zig_zag_level_order(n1), [
            [1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]
        ])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 27, 2019 \[Hard\] Minimum Appends to Craft a Palindrome
---
> **Question:** Given a string s we need to append (insertion at end) minimum characters to make a string palindrome.
>
> Follow-up: Don't use Manacher’s Algorithm, even though Longest Palindromic Substring can be efficiently solved with that algorithm.  

**Example 1:**

```
Input : s = "abede"
Output : "abedeba"
We can make string palindrome as "abedeba" by adding ba at the end of the string.
```

**Example 2:**
```
Input : s = "aabb"
Output : "aabbaa"
We can make string palindrome as"aabbaa" by adding aa at the end of the string.
```

**My thoughts:** An efficient way to solve this problem is to find the max len of suffix that is a palindrome. We can use a rolling hash function to quickly convert string into a number and by comparing the forward and backward hash value we can easily tell if a string is a palidrome or not. 
Example 1:
```py
Hash("123") = 123, Hash("321") = 321. Not Palindrome 
```
Example 2:
```py
Hash("101") = 101, Hash("101") = 101. Palindrome.
```
Rolling hashes are amazing, they provide you an ability to calculate the hash values without rehashing the whole string. eg. Hash("123") = Hash("12") ~ 3.  ~ is some function that can efficient using previous hashing value to build new caching value. 

However, we should not use Hash("123") = 123 as when the number become too big, the hash value be come arbitrarily big. Thus we use the following formula for rolling hash:

```py
hash("1234") = (1*p0^3 + 2*p0^2 + 3*p0^1 + 4*p0^0) % p1. where p0 is a much smaller prime and p1 is relatively large prime. 
```

There might be some hashing collision. However by choosing a much smaller p0 and relatively large p1, such collison is highly unlikely. Here I choose to remember a special large prime number `666667` and smaller number you can just use any smaller prime number, it shouldn't matter.


**Solution with Rolling Hash:** [https://repl.it/@trsong/Minimum-Appends-to-Craft-a-Palindrome](https://repl.it/@trsong/Minimum-Appends-to-Craft-a-Palindrome)
```py
import unittest

def craft_palindrome_with_min_appends(input_string):
    p0 = 17
    p1 = 666667  # large prime number worth to remember 
    reversed_string = input_string[::-1]
    forward_hash = 0  # right-most is most significant digit
    backward_hash = 0  # left-most is most significant digit
    max_len_palindrome_suffix = 0
    for i, char in enumerate(reversed_string):
        ord_char = ord(char)
        forward_hash = (forward_hash + ord_char * pow(p0, i)) % p1
        backward_hash = (p0 * backward_hash + ord_char) % p1
        if forward_hash == backward_hash:
            max_len_palindrome_suffix = i + 1

    return input_string + reversed_string[max_len_palindrome_suffix:]
  

class CraftPalindromeWithMinAppendSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(craft_palindrome_with_min_appends('abede'), 'abedeba')

    def test_example2(self):
        self.assertEqual(craft_palindrome_with_min_appends('aabb'), 'aabbaa')

    def test_empty_string(self):
        self.assertEqual(craft_palindrome_with_min_appends(''), '')
    
    def test_already_palindrome(self):
        self.assertEqual(craft_palindrome_with_min_appends('147313741'), '147313741')
        self.assertEqual(craft_palindrome_with_min_appends('328823'), '328823')

    def test_ascending_sequence(self):
        self.assertEqual(craft_palindrome_with_min_appends('12345'), '123454321')

    def test_binary_sequence(self):
        self.assertEqual(craft_palindrome_with_min_appends('10001001'), '100010010001')
        self.assertEqual(craft_palindrome_with_min_appends('100101'), '100101001')
        self.assertEqual(craft_palindrome_with_min_appends('010101'), '0101010')


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 26, 2019 \[Hard\] Find Next Greater Permutation
---
> **Question:** Given a number represented by a list of digits, find the next greater permutation of a number, in terms of lexicographic ordering. If there is not greater permutation possible, return the permutation with the lowest value/ordering.
>
> For example, the list `[1,2,3]` should return `[1,3,2]`. The list `[1,3,2]` should return `[2,1,3]`. The list `[3,2,1]` should return `[1,2,3]`.
>
> Can you perform the operation without allocating extra memory (disregarding the input memory)?

**My thoughts:** Imagine the list as a number, if it's in descending order, then there will be no number greater than that and we have to return the number in ascending order, that is, the smallest number. e.g. 321 will become 123. 

Leave first part untouched. If the later part of array are first increasing then decreasing, like 1321, then based on previous observation, we know the descending part will change from largest to smallest, we want the last increasing digit to increase as little as possible, i.e. slightly larger number on the right. e.g. 2113

Here are all the steps:
1. Find last increase number
2. Find the slightly larger number. i.e. the smallest one among all number greater than the last increase number on the right
3. Swap the slightly larger number with last increase number
4. Turn the descending array on right to be ascending array 

**Solution:** [https://repl.it/@trsong/Find-Next-Greater-Permutation](https://repl.it/@trsong/Find-Next-Greater-Permutation)
```py
import unittest

def next_greater_permutation(num_lst):
    n = len(num_lst)
    last_increase_index = n - 2

    # Step1: Find last increase number
    while last_increase_index >= 0:
        if num_lst[last_increase_index] >= num_lst[last_increase_index + 1]:
            last_increase_index -= 1
        else:
            break

    if last_increase_index >= 0:
        # Step2: Find the slightly larger number. i.e. the smallest one among all number greater than the last increase number on the right
        larger_num_index = n - 1
        while num_lst[larger_num_index] <= num_lst[last_increase_index]:
            larger_num_index -= 1

        # Step3: Swap the slightly larger number with last increase number
        num_lst[larger_num_index], num_lst[last_increase_index] = num_lst[last_increase_index], num_lst[larger_num_index]

    # Step4: Turn the descending array on right to be ascending array 
    i, j = last_increase_index + 1, n - 1
    while i < j:
        num_lst[i], num_lst[j] = num_lst[j], num_lst[i]
        i += 1
        j -= 1
    return num_lst


class NextGreaterPermutationSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(next_greater_permutation([1, 2, 3]), [1, 3, 2])
    
    def test_example2(self):
        self.assertEqual(next_greater_permutation([1, 3, 2]), [2, 1, 3])

    def test_example3(self):
        self.assertEqual(next_greater_permutation([3, 2, 1]), [1, 2, 3])

    def test_empty_array(self):
        self.assertEqual(next_greater_permutation([]), [])

    def test_one_elem_array(self):
        self.assertEqual(next_greater_permutation([1]), [1])

    def test_decrease_increase_decrease_array(self):
        self.assertEqual(next_greater_permutation([3, 2, 1, 6, 5, 4]), [3, 2, 4, 1, 5, 6])
        self.assertEqual(next_greater_permutation([3, 2, 4, 6, 5, 4]), [3, 2, 5, 4, 4, 6])

    def test_increasing_decreasing_increasing_array(self):
        self.assertEqual(next_greater_permutation([4, 5, 6, 1, 2, 3]), [4, 5, 6, 1, 3, 2])

    def test_multiple_decreasing_and_increasing_array(self):
        self.assertEqual(next_greater_permutation([5, 3, 4, 9, 7, 6]), [5, 3, 6, 4, 7, 9])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 25, 2019 \[Medium\] Longest Subarray with Sum Divisible by K
---
> **Question:** Given an arr[] containing n integers and a positive integer k. The problem is to find the length of the longest subarray with sum of the elements divisible by the given value k.

**Example:**
```py
Input : arr[] = {2, 7, 6, 1, 4, 5}, k = 3
Output : 4
The subarray is {7, 6, 1, 4} with sum 18, which is divisible by 3.
```

**My thoughts:** Recall the way to efficiently calculate subarray sum is to calculate prefix sum, `prefix_sum[i] = arr[0] + arr[1] + ... + arr[i] and prefix_sum[i] = prefix_sum[i-1] + arr[i]`. The subarray sum between index i and j, `arr[i] + arr[i+1] + ... + arr[j] = prefix_sum[j] - prefix_sum[i-1]`.

But this question is asking to find subarray whose sum is divisible by 3, that is,  `(prefix_sum[j] - prefix_sum[i-1]) mod k == 0 ` which implies `prefix_sum[j] % k == prefix_sum[j-1] % k`. So we just need to generate prefix_modulo array and find i,j such that `j - i reaches max` and `prefix_modulo[j] == prefix_modulo[i-1]`. As `j > i` and we must have value of `prefix_modulo[i-1]` already when we reach j. We can use a map to store the first occurance of certain prefix_modulo. This feels similar to Two-Sum question in a sense that we use map to store previous reached element and is able to quickly tell if current element satisfies or not.

**Solution:** [https://repl.it/@trsong/Longest-Subarray-with-Sum-Divisible-by-K](https://repl.it/@trsong/Longest-Subarray-with-Sum-Divisible-by-K)
```py
import unittest

def longest_subarray(nums, k):
    if not nums: return 0
    n = len(nums)
    prefix_modulo = [0] * n
    mod_so_far = 0
    for i in xrange(n):
        mod_so_far = (mod_so_far + nums[i] % k) % k
        prefix_modulo[i] = mod_so_far
    
    mod_first_occur_map = {0: -1}
    max_len = 0
    for i, prefix_mod in enumerate(prefix_modulo):
        if prefix_mod not in mod_first_occur_map:
            mod_first_occur_map[prefix_mod] = i
        else:
            max_len = max(max_len, i - mod_first_occur_map[prefix_mod])
    return max_len


class LongestSubarraySpec(unittest.TestCase):
    def test_example(self):
        # Modulo 3, prefix array = [2, 0, 0, 1, 2, 1]. max (end - start) = 4 such that prefix[end] - prefix[start-1] = 0
        self.assertEqual(longest_subarray([2, 7, 6, 1, 4, 5], 3), 4)  # sum([7, 6, 1, 4]) = 18 (18 % 3 = 0)

    def test_empty_array(self):
        self.assertEqual(longest_subarray([], 10), 0)
    
    def test_no_existance_of_such_subarray(self):
        self.assertEqual(longest_subarray([1, 2, 3, 4], 11), 0)
    
    def test_entire_array_qualify(self):
        # Modulo 4, prefix array: [0, 1, 2, 3, 3, 2, 1, 0]. max (end - start) = 8 such that prefix[end] - prefix[start-1] = 0
        self.assertEqual(longest_subarray([0, 1, 1, 1, 0, -1, -1, -1], 4), 8)  # entire array sum = 0
        self.assertEqual(longest_subarray([4, 5, 9, 17, 8, 3, 7, -1], 4), 8)  # entire array sum = 52 (52 % 4 = 0)

    def test_unique_subarray(self):
        # Modulo 6, prefix array: [0, 1, 1, 2, 3, 2, 4, 4, 5]. max (end - start) = 2 such that prefix[end] - prefix[start-1] = 0
        self.assertEqual(longest_subarray([0, 1, 0, 1, 1, -1, 2, 0, 1], 6), 2)  #  sum([1, -1]) = 0
        self.assertEqual(longest_subarray([6, 7, 12, 7, 13, 5, 8, 36, 19], 6), 2)  #  sum([13, 5]) = 18 (18 % 6 = 0)
    

if __name__ == '__main__':
    unittest.main(exit=False)

```

### Additional Question: LC 560 \[Medium\] Subarray Sum Equals K
---
> **Question:** Given an array of integers and an integer k, you need to find the total number of continuous subarrays whose sum equals to k.

**Example:**
```py
Input: nums = [1, 1, 1], k = 2
Output: 2
```

**My thoughts:** Just like how we efficiently calculate prefix_sum in previous question. We want to find how many index i exists such that `prefix[j] - prefix[i] = k`. As `j > i`, when we reach j, we pass i already, so we can store `prefix[i]` in a map and put value as occurance of `prefix[i]`, that is why this question feels similar to Two Sum question.

**Solution:** [https://repl.it/@trsong/Subarray-Sum-Equals-K](https://repl.it/@trsong/Subarray-Sum-Equals-K)
```py
import unittest

def subarray_sum(nums, k):
    sum_so_far = 0
    prefix_sum_occur_map = {0: 1}
    res = 0
    for num in nums:
        sum_so_far += num
        target_sum = sum_so_far - k
        res += prefix_sum_occur_map.get(target_sum, 0)
        prefix_sum_occur_map[sum_so_far] = prefix_sum_occur_map.get(sum_so_far, 0) + 1
    return res


class SubarraySumSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(subarray_sum([1, 1, 1], 2), 2)  # [1, 1] and [1, 1]
    
    def test_empty_array(self):
        self.assertEqual(subarray_sum([], 2), 0) 

    def test_target_is_zero(self):
        self.assertEqual(subarray_sum([0, 0], 0), 3) # [0], [0], [0, 0]

    def test_array_with_one_elem(self):
        self.assertEqual(subarray_sum([1], 0), 0)
        self.assertEqual(subarray_sum([1], 1), 1) # [1]

    def test_array_with_unique_target_prefix(self):
        # suppose the prefix_sum = [1, 2, 3, 3, 2, 1]
        self.assertEqual(subarray_sum([1, 1, 1, 0, -1, -1], 2), 4)  # [1, 1], [1, ,1], [1, 1, 0], [1, 1, 1, 0, -1]


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 24, 2019 LC 358 \[Hard\] Rearrange String K Distance Apart
---
> **Question:** Given a non-empty string str and an integer k, rearrange the string such that the same characters are at least distance k from each other.
>
> All input strings are given in lowercase letters. If it is not possible to rearrange the string, return an empty string "".

**Example 1:**
```
str = "aabbcc", k = 3
Result: "abcabc"
The same letters are at least distance 3 from each other.
```

**Example 2:**
```
str = "aaabc", k = 3 
Answer: ""
It is not possible to rearrange the string.
```

**Example 3:**
```
str = "aaadbbcc", k = 2
Answer: "abacabcd"
Another possible answer is: "abcabcda"
The same letters are at least distance 2 from each other.
```

**My thoughts:** The problem is just a variant of yesterday's Task Scheduler problem. The idea is to greedily choose the character with max remaining number for each window k. If no such character satisfy return empty string directly.

**Solution with Greedy Algorithm:** [https://repl.it/@trsong/Rearrange-String-K-Distance-Apart](https://repl.it/@trsong/Rearrange-String-K-Distance-Apart)
```py
import unittest
from Queue import PriorityQueue

def rearrange_string(input_string, k):
    if not input_string or k <= 0: return ""
    histogram = {}
    for c in input_string:
        # use negative key with min-heap to achieve max heap
        histogram[c] = histogram.get(c, 0) + 1
    
    max_heap = PriorityQueue()
    for c, count in histogram.items():
        max_heap.put((-count, c))

    res = []
    while not max_heap.empty():
        remaining_char = []
        for _ in xrange(k):
            # Greedily choose the char with max remaining count
            if max_heap.empty() and not remaining_char:
                break
            elif max_heap.empty():
                return ""
            neg_count, char = max_heap.get()
            count = -neg_count - 1
            res.append(char)
            if count > 0:
                remaining_char.append((-count, char))
        for count_char in remaining_char:
            max_heap.put(count_char)
    return ''.join(res)


class RearrangeStringSpec(unittest.TestCase):
    def assert_k_distance_apart(self, rearranged_string, original_string, k):
        # Test same length
        self.assertTrue(len(original_string) == len(rearranged_string))

        # Test containing all characters
        self.assertTrue(sorted(original_string) == sorted(rearranged_string))

        # Test K distance apart
        last_occur_map = {}
        for i, c in enumerate(rearranged_string):
            last_occur = last_occur_map.get(c, float('-inf'))
            self.assertTrue(i - last_occur >= k)
            last_occur_map[c] = i
    
    def test_utility_function_is_correct(self):
        original_string = "aaadbbcc"
        k = 2
        ans1 = "abacabcd"
        ans2 = "abcabcda"
        self.assert_k_distance_apart(ans1, original_string, k)
        self.assert_k_distance_apart(ans2, original_string, k)
        self.assertRaises(AssertionError, self.assert_k_distance_apart, original_string, original_string, k)

    def test_example1(self):
        original_string = "aabbcc"
        k = 3
        target_string = rearrange_string(original_string, k)
        self.assert_k_distance_apart(target_string, original_string, k)
    
    def test_example2(self):
        original_string = "aaabc"
        self.assertEqual(rearrange_string(original_string, 3),"")
    
    def test_example3(self):
        original_string = "aaadbbcc"
        k = 2
        target_string = rearrange_string(original_string, k)
        self.assert_k_distance_apart(target_string, original_string, k)
    
    def test_large_distance(self):
        original_string = "abcd"
        k = 10
        rearranged_string = rearrange_string(original_string, k)
        self.assert_k_distance_apart(rearranged_string, original_string, k)

    def test_empty_input_string(self):
        self.assertEqual(rearrange_string("", 1),"")
    
    def test_impossible_to_rearrange(self):
        self.assertEqual(rearrange_string("aaaabbbcc", 3), "")

    def test_k_too_small(self):
        self.assertEqual(rearrange_string("a", 0), "")
        self.assertEqual(rearrange_string("a", -1), "")
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 23, 2019 LC 621 \[Medium\] Task Scheduler
---
> **Question:** Given a char array representing tasks CPU need to do. It contains capital letters A to Z where different letters represent different tasks. Tasks could be done without original order. Each task could be done in one interval. For each interval, CPU could finish one task or just be idle.
>
> However, there is a non-negative cooling interval n that means between two same tasks, there must be at least n intervals that CPU are doing different tasks or just be idle.
>
> You need to return the least number of intervals the CPU will take to finish all the given tasks.

**Example:**
```py
Input: tasks = ["A", "A", "A", "B", "B", "B"], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B.
```

**My thoughts:** Treat n+1 as the size of each window. For each window, we try to fit as many tasks as possible following the max number of remaining tasks. If all tasks are chosen, we instead use idle. 

**Solution with Greedy Algorithm:** [https://repl.it/@trsong/Task-Scheduler](https://repl.it/@trsong/Task-Scheduler)
```py
import unittest
from Queue import PriorityQueue

def least_interval(tasks, n):
    if not tasks: return 0
    occurrence = {}
    for task in tasks:
        occurrence[task] = occurrence.get(task, 0) + 1
    
    max_heap = PriorityQueue()
    for task, occur in occurrence.items():
        # use negative key with min-heap to achieve max heap
        max_heap.put((-occur, task))
    
    res = 0
    while not max_heap.empty():
        remaining_tasks = []
        for _ in xrange(n+1):
            if max_heap.empty() and not remaining_tasks:
                break
            elif not max_heap.empty():
                # Greedily choose the task with max occurrence 
                negative_occur, task = max_heap.get()
                occur = -negative_occur - 1
                if occur > 0:
                    remaining_tasks.append((-occur, task))
            res += 1
        for task_occur in remaining_tasks:
            max_heap.put(task_occur)
    
    return res


class LeastIntervalSpec(unittest.TestCase):
    def test_example(self):
        tasks = ["A", "A", "A", "B", "B", "B"]
        n = 2
        self.assertEqual(least_interval(tasks, n), 8) # A -> B -> idle -> A -> B -> idle -> A -> B

    def test_no_tasks(self):
        self.assertEqual(least_interval([], 0), 0)
        self.assertEqual(least_interval([], 2), 0)
    
    def test_same_task_and_idle(self):
        tasks = ["A", "A", "A"]
        n = 1
        self.assertEqual(least_interval(tasks, n), 5)  # A -> idle -> A -> idle -> A

    def test_three_kind_tasks_no_idle(self):
        tasks = ["A", "B", "A", "C"]
        n = 1
        self.assertEqual(least_interval(tasks, n), 4)  # A -> B -> A -> C
    
    def test_three_kind_tasks_with_one_idle(self):
        tasks = ["A", "A", "A", "B", "C", "C"]
        n = 2
        self.assertEqual(least_interval(tasks, n), 7)  # A -> C -> B -> A -> C -> idle -> A

    def test_each_kind_one_task(self):
        tasks = ["A", "B", "C", "D"]
        n = 10
        self.assertEqual(least_interval(tasks, n), 4)  # A -> B -> C -> D


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 22, 2019 \[Medium\] Amazing Number
---
> **Question:** Define amazing number as: its value is less than or equal to its index. Given a circular array, find the starting position, such that the total number of amazing numbers in the array is maximized. 
> 
> Follow-up: Should get a solution with time complexity less than O(N^2)

**Example 1:**
```py
Input: [0, 1, 2, 3]
Ouptut: 0. When starting point at position 0, all the elements in the array are equal to its index. So all the numbers are amazing number.
```

**Example 2:** 
```py
Input: [1, 0, 0]
Output: 1. When starting point at position 1, the array becomes 0, 0, 1. All the elements are amazing number.
If there are multiple positions, return the smallest one.
```

**My thoughts:** We can use Brute-force Solution to get answer in O(n^2). But can we do better? Well, some smart observation is needed.

- First, we know that 0 and all negative numbers are amazing numbers. 
- Second, if a number is too big, i.e. greater than the length of array, then there is no way for that number to be an amazing number. 
- Finally, if a number is neither too small nor too big, i.e. between (0, n-1), then we can define "dangerous" range as [i - nums[i] + 1, i] and its complement: "safe" range [i + 1, i - nums[i]] should be safe. So we store all safe intervals to an array.

We accumlate those intervals by using interval counting technique: define interval_accu array, for each interval (start, end), interval_accu[start] += 1 and interval_accu[end+1] -= 1 so that when we can make interval accumulation by interval_accu[i] += interval_accu[i-1] for all i. 

Find max safe interval along the interval accumulation, i.e. the index that has maximum safe interval overlapping. 

**Brute-force Solution:** [https://repl.it/@trsong/Amazing-Number-Brute-force](https://repl.it/@trsong/Amazing-Number-Brute-force)
```py
def max_amazing_number_index(nums):
    n = len(nums)
    max_count = 0
    max_count_index = 0
    for i in xrange(n):
        count = 0
        for j in xrange(i, i + n):
            index = (j - i) % n
            if nums[j % n] <= index:
                count += 1
        
        if count > max_count:
            max_count = count
            max_count_index = i
    return max_count_index
```

**Efficient Solution with Interval Count:** [https://repl.it/@trsong/Amazing-Number](https://repl.it/@trsong/Amazing-Number)
```py
import unittest

def max_amazing_number_index(nums):
    n = len(nums)
    valid_intervals = []
    for i in xrange(n):
        # invalid zone starts from i - nums[i] + 1 and ends at i
        # 0 0 0 0 0 3 0 0 0 0 0 
        #       ^ ^ ^
        #       invalid
        # thus the valid zone is the complement [i + 1, i - nums[i]]
        if nums[i] > n:
            continue
        elif nums[i] < 0:
            valid_intervals.append([0, n-1])
        else:
            valid_intervals.append([(i + 1) % n, (i - nums[i]) % n])

    interval_accumulation = [0] * n
    for start, end in valid_intervals:
        # valid interval [start, end] is circular, i.e. end < start
        # thus can be broken into [0, end] and [start, n-1]
        interval_accumulation[start] += 1

        # with one exception: end > start, when the number is too small, like smaller than 0,
        # if that's the case, we don't count 
        if start > end:
            interval_accumulation[0] += 1

        if end + 1 < n:
            interval_accumulation[end + 1] -= 1

    max_count = interval_accumulation[0]
    max_count_index = 0
    for i in xrange(1, n):
        interval_accumulation[i] += interval_accumulation[i-1]
        if interval_accumulation[i] > max_count:
            max_count = interval_accumulation[i]
            max_count_index = i
    return max_count_index 


class MaxAmazingNumberIndexSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(max_amazing_number_index([0, 1, 2, 3]), 0)  # max # amazing number = 4 at [0, 1, 2, 3]

    def test_example2(self):
        self.assertEqual(max_amazing_number_index([1, 0, 0]), 1)  # max # amazing number = 3 at [0, 0, 1]

    def test_non_descending_array(self):
        self.assertEqual(max_amazing_number_index([0, 0, 0, 1, 2, 3]), 0)  # max # amazing number = 0 at [0, 0, 0, 1, 2, 3]

    def test_random_array(self):
        self.assertEqual(max_amazing_number_index([1, 4, 3, 2]), 1)  # max # amazing number = 2 at [4, 3, 2, 1]

    def test_non_ascending_array(self):
        self.assertEqual(max_amazing_number_index([3, 3, 2, 1, 0]), 2)  # max # amazing number = 4 at [2, 1, 0, 3, 3]

    def test_return_smallest_index_when_no_amazing_number(self):
        self.assertEqual(max_amazing_number_index([99, 99, 99, 99]), 0)  # max # amazing number = 0 thus return smallest possible index

    def test_negative_number(self):
        self.assertEqual(max_amazing_number_index([3, -99, -99, -99]), 1)  # max # amazing number = 4 at [-1, -1, -1, 3])

if __name__ == '__main__':
    unittest.main(exit=False)
```




### Aug 21, 2019 LC 273 \[Hard\] Integer to English Words
---
> **Question:** Convert a non-negative integer to its English word representation. 

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

**My thoughts:** The main difficulty of this problem comes from edge case scenarios from breaking large number into smaller ones and conquer them separately. Includes but not limit to "Zero", "Ten", "Twenty One" and other edge cases from missing millions, thousands and hundreds, like "Thirty Billion Two Million" and "Fifty Billion Two Hundred".


**Solution:** [https://repl.it/@trsong/Integer-to-English-Words](https://repl.it/@trsong/Integer-to-English-Words)
```py
import unittest

word_lookup = {
    0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five',
    6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten',
    11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', 15: 'Fifteen',
    16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', 19: 'Nineteen', 20: 'Twenty',
    30: 'Thirty', 40: 'Forty', 50: 'Fifty', 60: 'Sixty', 
    70: 'Seventy', 80: 'Eighty', 90: 'Ninety',
    100: 'Hundred', 1000: 'Thousand', 1000000: 'Million', 1000000000: 'Billion'
}

def read_hundreds(num):
    # Helper function to read num between 1 and 999
    global word_lookup
    if num == 0: return []
    res = []
    if num >= 100:
        res.append(word_lookup[num / 100])
        res.append(word_lookup[100])
    
    num %= 100
    if 21 <= num <= 99:
        res.append(word_lookup[num - num % 10])
        if num % 10 > 0:
            res.append(word_lookup[num % 10])
    elif num > 0:
        res.append(word_lookup[num])
        
    return res


def number_to_words(num):
    global word_lookup
    if num == 0: return word_lookup[0]
    res = []
    separators = [1000000000, 1000000, 1000] # 'Billion', 'Million', 'Thousand'
    for sep in separators:
        if num >= sep:
            res += read_hundreds(num/sep)
            res.append(word_lookup[sep])
            num %= sep
    if num > 0:
        res += read_hundreds(num)
    return ' '.join(res)


class NumberToWordSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(number_to_words(123), "One Hundred Twenty Three")

    def test_example2(self):
        self.assertEqual(number_to_words(12345), "Twelve Thousand Three Hundred Forty Five")

    def test_example3(self):
        self.assertEqual(number_to_words(1234567), "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven")

    def test_example4(self):
        self.assertEqual(number_to_words(1234567891), "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One")

    def test_zero(self):
        self.assertEqual(number_to_words(0), "Zero")

    def test_one_digit(self):
        self.assertEqual(number_to_words(8), "Eight")

    def test_two_digits(self):
        self.assertEqual(number_to_words(21), "Twenty One")
        self.assertEqual(number_to_words(10), "Ten")
        self.assertEqual(number_to_words(20), "Twenty")
        self.assertEqual(number_to_words(16), "Sixteen")
        self.assertEqual(number_to_words(32), "Thirty Two")
        self.assertEqual(number_to_words(30), "Thirty")

    def test_ignore_thousand_part(self):
        self.assertEqual(number_to_words(30002000000), "Thirty Billion Two Million")

    def test_ignore_million_part(self):
        self.assertEqual(number_to_words(50000000200), "Fifty Billion Two Hundred")


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 20, 2019 LC 297 \[Hard\] Serialize and Deserialize Binary Tree
---
> **Question:** Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.
>
> Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

**Example 1:**

```py
You may serialize the following tree:

    1
   / \
  2   3
     / \
    4   5

as "[1,2,3,null,null,4,5]"
```

**Example 2:**

```py
You may serialize the following tree:

       5
      / \ 
     4   7
    /   /
   3   2
  /   /
-1   9

as "[5,4,7,3,null,2,null,-1,null,9]"
```

**Solution with BFS:** [https://repl.it/@trsong/Serialize-and-Deserialize-Binary-Tree](https://repl.it/@trsong/Serialize-and-Deserialize-Binary-Tree)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __eq__(self, other):
        return other and other.val == self.val and other.left == self.left and other.right == self.right

class BinaryTreeSerializer(object):
    @staticmethod
    def serialize(tree):
        if not tree: return "[]"
        res = []
        queue = [tree]
        is_done = False
        while queue and not is_done:
            level_size = len(queue)
            is_done = True
            for _ in xrange(level_size):
                cur = queue.pop(0)
                if not cur:
                    res.append("null")
                else:
                    res.append(str(cur.val))
                    if cur.left or cur.right:
                        is_done = False
                    queue.append(cur.left)
                    queue.append(cur.right)
        return "[" + ",".join(res) + "]"

    @staticmethod
    def deserialize(encoded_string):
        if encoded_string == "[]": return None
        nums = encoded_string[1:-1].split(',')
        tree = TreeNode(int(nums[0]))
        i = 1
        queue = [tree]
        while queue:
            level_size = len(queue)
            for _ in xrange(level_size):
                cur = queue.pop(0)
                if i < len(nums) and nums[i] != "null":
                    cur.left = TreeNode(int(nums[i]))
                    queue.append(cur.left)
                i += 1

                if i < len(nums) and nums[i] != "null":
                    cur.right = TreeNode(int(nums[i]))
                    queue.append(cur.right)
                i += 1

                if i > len(nums):
                    break
        return tree
   

class BinaryTreeSerializerSpec(unittest.TestCase):
    def test_example1(self):
        """
            1
           / \
          2   3
             / \
            4   5
        """
        n3 = TreeNode(3, TreeNode(4), TreeNode(5))
        n1 = TreeNode(1, TreeNode(2), n3)
        encoded = BinaryTreeSerializer.serialize(n1)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(decoded, n1)

    def test_example2(self):
        """
               5
              / \ 
             4   7
            /   /
           3   2
          /   /
        -1   9
        """
        n4 = TreeNode(4, TreeNode(3, TreeNode(-1)))
        n7 = TreeNode(7, TreeNode(2, TreeNode(9)))
        n5 = TreeNode(5, n4, n7)
        encoded = BinaryTreeSerializer.serialize(n5)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(decoded, n5)

    def test_serialize_empty_tree(self):
        encoded = BinaryTreeSerializer.serialize(None)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertIsNone(decoded)

    def test_serialize_left_heavy_tree(self):
        """
            1
           /
          2
         /
        3
        """
        tree = TreeNode(1, TreeNode(2, TreeNode(3)))
        encoded = BinaryTreeSerializer.serialize(tree)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(decoded, tree)

    def test_serialize_right_heavy_tree(self):
        """
        1
         \
          2
         /
        3
        """
        tree = TreeNode(1, right=TreeNode(2, TreeNode(3)))
        encoded = BinaryTreeSerializer.serialize(tree)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(decoded, tree) 
        

        
if __name__ == '__main__':
    unittest.main(exit=False)
```

### Additional Question: \[Medium\] M Smallest in K Sorted Lists
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

**My thoughts:** This problem is almost the same as merge k sorted list. The idea is to leverage priority queue to keep track of minimum element among all k sorted list.

**Solution:** [https://repl.it/@trsong/M-Smallest-in-K-Sorted-Lists](https://repl.it/@trsong/M-Smallest-in-K-Sorted-Lists)
```py
import unittest
from Queue import PriorityQueue

def find_m_smallest(ksorted_list, m):
    pq = PriorityQueue()
    for lst in ksorted_list:
        if lst:
            pq.put((lst[0], [0, lst]))
    
    res = None
    while not pq.empty() and m >= 1:
        index, lst = pq.get()[1]
        if m == 1:
            res = lst[index]
            break
        m -= 1
        if index + 1 < len(lst):
            pq.put((lst[index + 1], [index + 1, lst]))
    return res


class FindMSmallestSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(find_m_smallest([[1, 3], [2, 4, 6], [0, 9, 10, 11]], m=5), 4) 

    def test_example2(self):
        self.assertEqual(find_m_smallest([[1, 3, 20], [2, 4, 6]], m=2), 2) 

    def test_example3(self):
        self.assertEqual(find_m_smallest([[1, 3, 20], [2, 4, 6]], m=6), 20)

    def test_empty_sublist(self):
        self.assertEqual(find_m_smallest([[1], [], [0, 2]], m=2), 1)

    def test_one_sublist(self):
        self.assertEqual(find_m_smallest([[1, 2, 3, 4, 5]], m=5), 5)

    def test_target_out_of_boundary(self):
        self.assertIsNone(find_m_smallest([[1, 2, 3], [4, 5, 6]], 7))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 19, 2019 \[Medium\] Jumping Numbers
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

**Solution with BFS:** [https://repl.it/@trsong/Jumping-Numbers](https://repl.it/@trsong/Jumping-Numbers)
```py
import unittest

def generate_jumping_numbers(upper_bound):
    if upper_bound < 0: return []
    queue = [x for x in xrange(1, 10)]
    res = [0]
    while queue:
        # Apply BFS to search for jumping numbers
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
        self.assertEqual(generate_jumping_numbers(0), [0])

    def test_single_digits_are_all_jummping_numbers(self):
        self.assertEqual(generate_jumping_numbers(9), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_five_as_upperbound(self):
        self.assertEqual(generate_jumping_numbers(5), [0, 1, 2, 3, 4, 5])

    def test_not_always_contains_upperbound(self):
        self.assertEqual(generate_jumping_numbers(13), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12])

    def test_example(self):
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 21, 23, 32, 34, 43, 45, 54, 56, 65, 67, 76, 78, 87, 89, 98, 101]
        self.assertEqual(generate_jumping_numbers(105), expected)
    
    def test_negative_upperbound(self):
        self.assertEqual(generate_jumping_numbers(-1), [])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Additional Question: \[Easy\] Swap Even and Odd Nodes
---
> **Question:** Given the head of a singly linked list, swap every two nodes and return its head.
> 
> Note: Make sure it's acutally nodes that get swapped not value. 


**Example:**
```py
given 1 -> 2 -> 3 -> 4, return 2 -> 1 -> 4 -> 3.
```

**Solution with Recursion:** [https://repl.it/@trsong/Swap-Even-and-Odd-Nodes](https://repl.it/@trsong/Swap-Even-and-Odd-Nodes)
```py
import unittest

class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


def swap_list(lst):
    if not lst or not lst.next:
        return lst
    first = lst
    second = first.next
    third = second.next

    second.next = first
    first.next = swap_list(third)
    return second
  

class SwapListSpec(unittest.TestCase):
    def assert_lists(self, lst, node_seq):
        p = lst
        for node in node_seq:
            if p != node: print (p.data if p else "None"), (node.data if node else "None")
            self.assertTrue(p == node)
            p = p.next
        self.assertTrue(p is None)

    def test_empty(self):
        self.assert_lists(swap_list(None), [])

    def test_one_elem_list(self):
        n1 = ListNode(1)
        self.assert_lists(swap_list(n1), [n1])

    def test_two_elems_list(self):
        # 1 -> 2
        n2 = ListNode(2)
        n1 = ListNode(1, n2)
        self.assert_lists(swap_list(n1), [n2, n1])

    def test_three_elems_list(self):
        # 1 -> 2 -> 3
        n3 = ListNode(3)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)
        self.assert_lists(swap_list(n1), [n2, n1, n3])

    def test_four_elems_list(self):
        # 1 -> 2 -> 3 -> 4
        n4 = ListNode(4)
        n3 = ListNode(3, n4)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)
        self.assert_lists(swap_list(n1), [n2, n1, n4, n3])


if __name__ == '__main__':
    unittest.main(exit=False)
```

**Solution2 with Iteration:** [https://repl.it/@trsong/Swap-Even-and-Odd-Nodes-Iterative](https://repl.it/@trsong/Swap-Even-and-Odd-Nodes-Iterative)

```py
def swap_list(lst):
    dummy = ListNode(-1, lst)
    prev = dummy
    p = lst
    while p and p.next:
        first = p
        second = first.next

        first.next = second.next
        second.next = first
        prev.next = second
        prev = first
        p = prev.next
    return dummy.next
```


### Aug 18, 2019 LT 612 \[Medium\] K Closest Points
--- 
> **Question:** Given some points and a point origin in two dimensional space, find k points out of the some points which are nearest to origin.
> 
> Return these points sorted by distance, if they are same with distance, sorted by x-axis, otherwise sorted by y-axis.


**Example:**

```py
Given points = [[4, 6], [4, 7], [4, 4], [2, 5], [1, 1]], origin = [0, 0], k = 3
return [[1, 1], [2, 5], [4, 4]]
```

**My thoguhts:** This problem can be easily solved with k Max-heap with key being the distance and value being the point. First heapify first k elements to form a k max-heap. Then for the remaining n - k element, replace top of heap with smaller-distance point.

**Solution with k Max-Heap:** [https://repl.it/@trsong/K-Closest-Points](https://repl.it/@trsong/K-Closest-Points)
```py
import unittest
from Queue import PriorityQueue

def distance2(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy 

def k_closest_points(points, origin, k):
    if not points and k == 0: return []
    elif len(points) < k: return None

    max_heap = PriorityQueue()
    for i in xrange(k):
        max_heap.put((-distance2(points[i], origin), points[i]))

    for i in xrange(k, len(points)):
        dist = distance2(points[i], origin)
        top = max_heap.queue[0]
        if -top[0] > dist:   
            max_heap.get()
            max_heap.put((-dist, points[i]))
            
    res = [None] * k
    for i in xrange(k-1, -1, -1):
        res[i] = max_heap.get()[1]
    return res


class KClosestPointSpec(unittest.TestCase):
    def assert_points(self, result, expected):
        self.assertEqual(sorted(result), sorted(expected))

    def test_example(self):
        points = [[4, 6], [4, 7], [4, 4], [2, 5], [1, 1]]
        origin = [0, 0]
        k = 3
        expected = [[1, 1], [2, 5], [4, 4]]
        self.assert_points(k_closest_points(points, origin, k), expected)

    def test_empty_points(self):
        self.assert_points(k_closest_points([], [0, 0], 0), [])
        self.assertIsNone(k_closest_points([], [0, 0], 1))

    def test_descending_distance(self):
        points = [[1, 6], [1, 5], [1, 4], [1, 3], [1, 2], [1, 1]]
        origin = [1, 1]
        k = 2
        expected = [[1, 2], [1, 1]]
        self.assert_points(k_closest_points(points, origin, k), expected)

    def test_ascending_distance(self):
        points = [[-1, -1], [-2, -1], [-3, -1], [-4, -1], [-5, -1], [-6, -1]]
        origin = [-1, -1]
        k = 1
        expected = [[-1, -1]]
        self.assert_points(k_closest_points(points, origin, k), expected)

    def test_duplicated_distance(self):
        points = [[1, 0], [0, 1], [-1, -1], [1, 1], [2, 1], [-2, 0]]
        origin = [0, 0]
        k = 5
        expected = [[1, 0], [0, 1], [-1, -1], [1, 1], [-2, 0]]
        self.assert_points(k_closest_points(points, origin, k), expected)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Additional Question: \[Medium\] Swap Even and Odd Bits
---
> **Question:** Given an unsigned 8-bit integer, swap its even and odd bits. The 1st and 2nd bit should be swapped, the 3rd and 4th bit should be swapped, and so on.

**Example:**

```py
10101010 should be 01010101. 11100010 should be 11010001.
```
> Bonus: Can you do this in one line?

**Solution:** [https://repl.it/@trsong/Swap-Even-and-Odd-Bits](https://repl.it/@trsong/Swap-Even-and-Odd-Bits)
```py
import unittest

def swap_bits(num):
    # 1010 is 0xa and  0101 is 0x5
    # 32 bit has 8 bits (4 * 8 = 32)
    return (num & 0xaaaaaaaa) >> 1 | (num & 0x55555555) << 1

class SwapBitSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(swap_bits(0b10101010), 0b01010101)

    def test_example2(self):
        self.assertEqual(swap_bits(0b11100010), 0b11010001)

    def test_zero(self):
        self.assertEqual(swap_bits(0), 0)
    
    def test_one(self):
        self.assertEqual(swap_bits(0b1), 0b10)

    def test_odd_digits(self):
        self.assertEqual(swap_bits(0b111), 0b1011)

    def test_large_number(self):
        self.assertEqual(swap_bits(0xffffffff), 0xffffffff)
        self.assertEqual(swap_bits(0x55555555), 0xaaaaaaaa)
        self.assertEqual(swap_bits(0xaaaaaaaa), 0x55555555)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 17, 2019 \[Medium\] Deep Copy Linked List with Pointer to Random Node
---
> **Question:** Make a deep copy of a linked list that has a random link pointer and a next pointer.

**My thoughts:** The way we solve this problem is to mingle old nodes and cloned nodes so that every odd node is original node and every even node is clone node which will allow us to access both nodes through `node.next` and `node.next.next`. And we then build random pointer and finally connect every other node to build cloned node's next as well as restore original node's next. 

**Solution:** [https://repl.it/@trsong/Deep-Copy-Linked-List-with-Pointer-to-Random-Node](https://repl.it/@trsong/Deep-Copy-Linked-List-with-Pointer-to-Random-Node)
```py
import unittest

class ListNode(object):
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random
    
    def __eq__(self, other):
        if other is not None:
            is_random_valid = self.random is None and other.random is None or self.random is not None and other.random is not None and self.random.val == other.random.val
            return is_random_valid and self.val == other.val and self.next == other.next
        else:
            return False

def deep_copy(lst):
    if not lst: return None
    
    # Insert cloned node into original list
    # Now every odd pointer is old node and every even pointer is cloned node
    node = lst
    while node:
        node.next = ListNode(node.val, node.next)
        node = node.next.next

    # Build cloned node's random pointer
    node = lst
    while node:
        node.next.random = node.random.next if node.random else None
        node = node.next.next

    # Build cloned node's next and restore old node's next
    node = lst
    res = lst.next
    while node:
        cloned_node = node.next
        old_next = cloned_node.next
        if old_next:
            cloned_node.next = old_next.next
        node.next = old_next
        node = old_next
    
    return res


class DeepCopySpec(unittest.TestCase):
    def test_empty_list(self):
        self.assertIsNone(deep_copy(None))
    
    def test_list_with_random_point_to_itself(self):
        n = ListNode(1)
        n.random = n
        self.assertEqual(deep_copy(n), n)

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
        self.assertEqual(deep_copy(n1), n1)

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
        self.assertEqual(deep_copy(n1), n1)

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
        self.assertEqual(deep_copy(n1), n1)
        

if __name__ == '__main__':
    unittest.main(exit=False)
``` 

### Aug 16, 2019 \[Medium\] Longest Substring without Repeating Characters
---
> **Question:** Given a string, find the length of the longest substring without repeating characters.
>
> **Note:** Can you find a solution in linear time?
 
**Example:**
```py
lengthOfLongestSubstring("abrkaabcdefghijjxxx") # => 10 as len("abcdefghij") == 10
```

**My thoughts:** This is a typical sliding window problem. The idea is to mantain a last occurance map while proceeding the sliding window. Such window is bounded by indices `(i, j)`, whenever we process next character j, we check the last occurance map to see if the current character `a[j]` is duplicated within the window `(i, j)`, ie. `i <= k < j`, if that's the case, we move `i` to `k + 1` so that `a[j]` no longer exists in window. And we mantain the largest window size `j - i + 1` as the longest substring without repeating characters.

**Solution with Sliding Window:** [https://repl.it/@trsong/Longest-Substring-without-Repeating-Characters](https://repl.it/@trsong/Longest-Substring-without-Repeating-Characters)
```py
import unittest

def longest_nonrepeated_substring(input_string):
    if not input_string: return 0
    last_occur = {}
    max_len = 0
    i = 0
    for j in xrange(len(input_string)):
        cur = input_string[j]
        last_index = last_occur.get(cur, -1)
        if last_index >= i:
            i = last_index + 1
        last_occur[cur] = j
        max_len = max(max_len, j - i + 1)
    return max_len
    

class LongestNonrepeatedSubstringSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(longest_nonrepeated_substring("abrkaabcdefghijjxxx"), 10) # "abcdefghij"

    def test_empty_string(self):
        self.assertEqual(longest_nonrepeated_substring(""), 0)

    def test_string_with_repeated_characters(self):
        self.assertEqual(longest_nonrepeated_substring("aabbafacbbcacbfa"), 4) # "facb"

    def test_some_random_string(self):
        self.assertEqual(longest_nonrepeated_substring("ABDEFGABEF"), 6) # "ABDEFG"

    def test_all_repated_characters(self):
        self.assertEqual(longest_nonrepeated_substring("aaa"), 1) # "a"
    
    def test_non_repated_characters(self):
        self.assertEqual(longest_nonrepeated_substring("abcde"), 5) # "abcde"

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 15, 2019 \[Hard\] Largest Rectangle
---
> **Question:** Given an N by M matrix consisting only of 1's and 0's, find the largest rectangle containing only 1's and return its area.

**For Example:**

```py
Given the following matrix:

[[1, 0, 0, 0],
 [1, 0, 1, 1],
 [1, 0, 1, 1],
 [0, 1, 0, 0]]

Return 4. As the following 1s form the largest rectangle containing only 1s:
 [1, 1],
 [1, 1]
```

**My thoughts:** This problem is an application of finding largest rectangle in histogram. That question gives you an array of height of bar in histogram and find the largest area of rectangle bounded by the bar. (consider bar width as 1)

Example:
```py
largest_rectangle_in_histogram([3, 4, 5, 4, 3]) # return 15 as max at height 3 * width 5
```

Now the way we take advantage of largest_rectangle_in_histogram is that, we can calculate the histogram of each row with each cell value being the accumulated value since last saw 1. 

Example:

```py
Suppose the table looks like the following:

[
    [0, 1, 0, 1],
    [1, 1, 1, 0],
    [0, 1, 1, 0]
]

The histogram of first row is just itself:
    [0, 1, 0, 1]  # largest_rectangle_in_histogram => 1 as max at height 1 * width 1
The histogram of second row is:
    [0, 1, 0, 1]
    +
    [1, 1, 1, 0]
    =
    [1, 2, 1, 0]  # largest_rectangle_in_histogram => 3 as max at height 1 * width 3
              ^ will not accumulate 0
The histogram of third row is:
    [1, 2, 1, 0]       
    +
    [0, 1, 1, 0]
    =
    [0, 3, 2, 0]  # largest_rectangle_in_histogram => 4 as max at height 2 * width 2
              ^ will not accumulate 0
     ^ will not accumulate 0

Therefore, the largest rectangle has 4 1s in it.
```

**Solution with DP:** [https://repl.it/@trsong/Largest-Rectangle](https://repl.it/@trsong/Largest-Rectangle)
```py
import unittest

def largest_rectangle_in_histogram(histogram):
    stack = []
    i = 0
    max_area = 0
    while i < len(histogram) or stack:
        if not stack or i < len(histogram) and histogram[stack[-1]] <= histogram[i]:
            # maintain an ascending stack
            stack.append(i)
            i += 1
        else:
            # if stack starts decreasing,
            # then left boundary must be stack[-2] and right boundary must be i. Note both boundaries are exclusive
            # and height is stack[-1]
            height = histogram[stack.pop()]
            left_boundary = stack[-1] if stack else -1
            right_boundary = i
            current_area = height * (right_boundary - left_boundary - 1)
            max_area = max(max_area, current_area)
    return max_area


class LargestRectangleInHistogramSpec(unittest.TestCase):
    def test_ascending_sequence(self):
        self.assertEqual(largest_rectangle_in_histogram([0, 1, 2, 3, 4]), 6) # max at height 2 * width 3

    def test_descending_sequence(self):
        self.assertEqual(largest_rectangle_in_histogram([4, 3, 2, 1, 0]), 6) # max at height 3 * width 2

    def test_sequence3(self):
        self.assertEqual(largest_rectangle_in_histogram([3, 4, 5, 4, 3]), 15) # max at height 3 * width 5

    def test_sequence4(self):
        self.assertEqual(largest_rectangle_in_histogram([3, 10, 4, 10, 5, 10, 4, 10, 3]), 28)  # max at height 4 * width 7

    def test_sequence5(self):
        self.assertEqual(largest_rectangle_in_histogram([6, 2, 5, 4, 5, 1, 6]), 12)  # max at height 4 * width 3


def largest_rectangle(table):
    if not table or not table[0]: return 0
    n, m = len(table), len(table[0])
    max_area = largest_rectangle_in_histogram(table[0])
    for r in xrange(1, n):
        for c in xrange(m):
            # calculate the histogram of each row since last saw 1 at same column
            if table[r][c] == 1:
                table[r][c] = table[r-1][c] + 1
        max_area = max(max_area, largest_rectangle_in_histogram(table[r]))
    return max_area
    

class LargestRectangleSpec(unittest.TestCase):
    def test_empty_table(self):
        self.assertEqual(largest_rectangle([]), 0)
        self.assertEqual(largest_rectangle([[]]), 0)

    def test_example(self):
        self.assertEqual(largest_rectangle([
            [1, 0, 0, 0],
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [0, 1, 0, 0]
        ]), 4)

    def test_table2(self):
        self.assertEqual(largest_rectangle([
            [0, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 1, 1, 0]
        ]), 4)

    def test_table3(self):
        self.assertEqual(largest_rectangle([
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0],
        ]), 3)

    def test_table4(self):
        self.assertEqual(largest_rectangle([
            [0, 0, 1, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 0, 1],
        ]), 4)

    def test_table5(self):
        self.assertEqual(largest_rectangle([
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 0]
        ]), 8)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 14, 2019 LC 375 \[Medium\] Guess Number Higher or Lower II
---
> **Question:** We are playing the Guess Game. The game is as follows:
>
> I pick a number from 1 to n. You have to guess which number I picked.
> 
> Every time you guess wrong, I'll tell you whether the number I picked is higher or lower.
> 
> However, when you guess a particular number x, and you guess wrong, you pay $x. You win the game when you guess the number I picked.

**Example:**

```
n = 10, I pick 8.

First round:  You guess 5, I tell you that it's higher. You pay $5.
Second round: You guess 7, I tell you that it's higher. You pay $7.
Third round:  You guess 9, I tell you that it's lower. You pay $9.

Game over. 8 is the number I picked.

You end up paying $5 + $7 + $9 = $21.
```

Given a particular n ≥ 1, find out how much money you need to have to guarantee a win.

**My thoughts:** This question looks really similar to *LC 312 [Hard] Burst Balloons* in a sense that it choose the best candidate at each step and deligate the subproblem to recursive calls. eg. `guess_number_cost_between(i, j) = max(k + guess_number_cost_between(i, k-1), guess_number_cost_between(k+1, j)) for all k between i+1 and j-1 inclusive`.

Take a look at base case:
- when 1 number to pick, the min $ to secure a win is $0
- when 2 numbers to pick, the min $ to secure a win is to choose the smaller one, 
- when 3 numbers to pick, the min $ to secure a win is to choose the middle one

**Solution with Top-down DP(Recursion with Cache):** [https://repl.it/@trsong/Guess-Number-Higher-or-Lower-II](https://repl.it/@trsong/Guess-Number-Higher-or-Lower-II)
```py
import unittest
import sys

def guess_number_cost(n):
    cache = [[None for _ in xrange(n+1)] for _ in xrange(n+1)]
    return guess_number_cost_with_cache(1, n, cache)


def guess_number_cost_between(i, j, cache):
    if i == j: return 0
    elif i + 1 == j: return i
    elif i + 2 == j: return i + 1
    cost = sys.maxint
    for k in xrange(i+1, j):
        pick_k_cost = k + guess_number_cost_with_cache(i, k-1, cache) + guess_number_cost_with_cache(k+1, j, cache)
        cost = min(cost, pick_k_cost)
    return cost


def guess_number_cost_with_cache(i, j, cache):
    if cache[i][j] is None:
        cache[i][j] = guess_number_cost_between(i, j, cache)
    return cache[i][j]


class GuessNumberCostSpec(unittest.TestCase):
    def test_n_equals_one(self):
        self.assertEqual(guess_number_cost(1), 0)

    def test_n_equals_two(self):
        self.assertEqual(guess_number_cost(2), 1) # pick: 1

    def test_n_equals_three(self):
        self.assertEqual(guess_number_cost(3), 2) # pick: 2

    def test_n_equals_four(self):
        self.assertEqual(guess_number_cost(4), 4) # pick: 1, 3

    def test_n_equals_five(self):
        self.assertEqual(guess_number_cost(5), 6) # pick: 2, 4

    def test_n_equals_six(self):
        self.assertEqual(guess_number_cost(6), 9) # pick 1, 3, 5


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 13, 2019 LC 727 \[Hard\] Minimum Window Subsequence
---
> **Question:** Given strings S and T, find the minimum (contiguous) substring W of S, so that T is a subsequence of W.
>
> If there is no such window in S that covers all characters in T, return the empty string "". If there are multiple such minimum-length windows, return the one with the left-most starting index.

**Example:**

```
Input: 
S = "abcdebdde", T = "bde"
Output: "bcde"

Explanation: 
"bcde" is the answer because it occurs before "bdde" which has the same length.
"deb" is not a smaller window because the elements of T in the window must occur in order.
```

**My thoughts:** Have you noticed the pattern that substring of s always has the same first char as t. i.e. `s = "abcdebdde", t = "bde", substring = "bcde", substring[0] == t[0]`,  we can take advantage of that to keep track of previous index such that t[0] == s[index] and we can do that recursively for the rest of t and s. We get the following recursive definition

```py
Let dp[i][j] = index where index represents index such that s[index:i] has subsequence t[0:j].

dp[i][j] = dp[i-1][j-1] if there s[i-1] matches t[j-1] 
         = dp[i-1][j]   otherwise
```

And the final solution is to find index where `len of t <= index <= len of s` such that `index - dp[index][len of t]` i.e. the length of substring, reaches minimum. 

**Solution with DP:** [https://repl.it/@trsong/Minimum-Window-Subsequence](https://repl.it/@trsong/Minimum-Window-Subsequence)
```py
import unittest
import sys

def min_window_subsequence(s, t):
    if len(s) < len(t): return ""
    n, m = len(s), len(t)

    # let dp[i][j] = index where index represents index such that s[index:i] has subsequence t[0:j]
    # Then s[start: end] where start = dp[i][m], end = i such that len = i - dp[i][m] reaches minimum
    dp = [[-1 for _ in xrange(m+1)] for _ in xrange(n+1)]
    for i in xrange(n):
        dp[i][0] = i

    for i in xrange(1, n+1):
        for j in xrange(1, m+1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = dp[i-1][j]

    start = -1
    min_len = sys.maxint
    for i in xrange(m, n+1):
        if dp[i][m] != -1:
            cur_len = i - dp[i][m]
            if cur_len < min_len:
                start = dp[i][m]
                min_len = cur_len
    return s[start: start + min_len] if start != -1 else ""


class MinWindowSubsequenceSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(min_window_subsequence("abcdebdde", "bde"), "bcde")

    def test_target_too_long(self):
        self.assertEqual(min_window_subsequence("a", "aaa"), "")

    def test_duplicated_char_in_target(self):
        self.assertEqual(min_window_subsequence("abbbbabbbabbababbbb", "aa"), "aba")

    def test_duplicated_char_but_no_matching(self):
        self.assertEqual(min_window_subsequence("ccccabbbbabbbabbababbbbcccc", "aca"), "")

    def test_match_last_char(self):
        self.assertEqual(min_window_subsequence("abcdef", "f"), "f")

    def test_match_first_char(self):
        self.assertEqual(min_window_subsequence("abcdef", "a"), "a")

    def test_equal_length_string(self):
        self.assertEqual(min_window_subsequence("abc", "abc"), "abc")
        self.assertEqual(min_window_subsequence("abc", "bca"), "")


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 12, 2019 LC 230 \[Medium\] Kth Smallest Element in a BST
---
> **Question:** Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

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

**My thoughts:** In BST, the in-order traversal presents orders from smallest to largest. Thus you can use both recursion with global variable k or the following template for iterative in-order traversal.

**Template for Iterative In-order Traversal:**

```py
while True:
    if t is not None:
        stack.append(t)
        t = t.left
    elif stack:
        t = stack.pop()
        print t.val # this will give node value follows in-order traversal    
        t = t.right
    else:
        break
return None
```

**Solution with Iterative In-order Traversal:** [https://repl.it/@trsong/Kth-Smallest-Element-in-a-BST](https://repl.it/@trsong/Kth-Smallest-Element-in-a-BST)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def kth_smallest(tree, k):
    t = tree
    stack = []

    while True:
        if t is not None:
            stack.append(t)
            t = t.left
        elif stack:
            t = stack.pop()
            if k == 1:
                return t.val
            else:
                k -= 1
            t = t.right
        else:
            break
    return None


class KthSmallestSpec(unittest.TestCase):
    def test_example1(self):
        """
           3
          / \
         1   4
          \
           2
        """
        n1 = TreeNode(1, right=TreeNode(2))
        tree = TreeNode(3, n1, TreeNode(4))
        check_expected = [1, 2, 3, 4]
        for e in check_expected:
            self.assertEqual(kth_smallest(tree, e), e)

    def test_example2(self):
        """
              5
             / \
            3   6
           / \
          2   4
         /
        1
        """
        n2 = TreeNode(2, TreeNode(1))
        n3 = TreeNode(3, n2, TreeNode(4))
        tree = TreeNode(5, n3, TreeNode(6))
        check_expected = [1, 2, 3, 4, 5, 6]
        for e in check_expected:
            self.assertEqual(kth_smallest(tree, e), e)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 11, 2019 LC 684 \[Medium\] Redundant Connection
---
> **Question:** In this problem, a tree is an **undirected** graph that is connected and has no cycles.
>
> The given input is a graph that started as a tree with N nodes (with distinct values 1, 2, ..., N), with one additional edge added. The added edge has two different vertices chosen from 1 to N, and was not an edge that already existed.
>
> The resulting graph is given as a 2D-array of edges. Each element of edges is a pair [u, v] with u < v, that represents an undirected edge connecting nodes u and v.
>
> Return an edge that can be removed so that the resulting graph is a tree of N nodes. If there are multiple answers, return the answer that occurs last in the given 2D-array. The answer edge [u, v] should be in the same format, with u < v.

**Example 1:**

```
Input: [[1,2], [1,3], [2,3]]
Output: [2,3]
Explanation: The given undirected graph will be like this:
  1
 / \
2 - 3
```

**Example 2:**

```
Input: [[1,2], [2,3], [3,4], [1,4], [1,5]]
Output: [1,4]
Explanation: The given undirected graph will be like this:
5 - 1 - 2
    |   |
    4 - 3
```

**My thoughts:** Process edges one by one, when we encouter an edge that has two ends already connected then adding current edge will form a cycle, then we return such edge. 

In order to efficiently checking connection between any two nodes. The idea is to keep track of all nodes that are already connected. Disjoint-set(Union Find) is what we are looking for. Initially UF makes all nodes disconnected, and whenever we encounter an edge, connect both ends. And we do that for all edges. 

**Solution with Disjoint-Set(Union-Find):** [https://repl.it/@trsong/CharmingCluelessApplets](https://repl.it/@trsong/CharmingCluelessApplets)
```py
import unittest

class UnionFind(object):
    def __init__(self, size):
        self.parent = range(size)
        self.rank = [0] * size
    
    def find(self, v):
        p = v
        while self.parent[p] != p:
            p = self.parent[p]
        self.parent[v] = p
        return p
    
    def union(self, v1, v2):
        p1 = self.find(v1)
        p2 = self.find(v2)
        if p1 != p2:
            if self.rank[p1] > self.rank[p2]:
                self.parent[p2] = p1
                self.rank[p1] += 1
            else:
                self.parent[p1] = p2
                self.rank[p2] += 1            
            
    def is_connected(self, v1, v2):
        return self.find(v1) == self.find(v2)

def find_redundant_connection(edges):
    uf = UnionFind(len(edges) + 1)
    for u, v in edges:
        if uf.is_connected(u, v):
            return [u, v]
        else:
            uf.union(u, v)
    return None


class FindRedundantConnectionSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(find_redundant_connection([[1,2], [1,3], [2,3]]), [2, 3])

    def test_example2(self):
        self.assertEqual(find_redundant_connection([[1,2], [2,3], [3,4], [1,4], [1,5]]), [1,4])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 10, 2019 LC 308 \[Hard\] Range Sum Query 2D - Mutable
---
> **Question:** Given a 2D matrix matrix, find the sum of the elements inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).

**Example:**

```py
Given matrix = [
  [3, 0, 1, 4, 2],
  [5, 6, 3, 2, 1],
  [1, 2, 0, 1, 5],
  [4, 1, 0, 1, 7],
  [1, 0, 3, 0, 5]
]

sumRegion(2, 1, 4, 3) -> 8
update(3, 2, 2)
sumRegion(2, 1, 4, 3) -> 10
```

**Solution with 2D Binary Indexed Tree:** [https://repl.it/@trsong/Range-Sum-Query-2D-Mutable](https://repl.it/@trsong/Range-Sum-Query-2D-Mutable)
```py
import unittest

class RangeSumQuery(object):
    def __init__(self, matrix):
        n, m = len(matrix), len(matrix[0])
        self.bit_matrix = [[0 for _ in xrange(m+1)] for _ in xrange(n+1)]
        for r in xrange(n):
            for c in xrange(m):
                self.update(r, c, matrix[r][c])

    def sumOriginToPosition(self, position):
        row, col = position
        res = 0
        rIdx = row + 1
        while rIdx > 0:
            cIdx = col + 1
            while cIdx > 0:
                res += self.bit_matrix[rIdx][cIdx]
                cIdx -= cIdx & -cIdx
            rIdx -= rIdx & -rIdx
        return res

    def sumRegion(self, row1, col1, row2, col2):
        top_left = (row1 - 1, col1 - 1)
        top_right = (row1 - 1, col2)
        bottom_left = (row2, col1 - 1)
        bottom_right = (row2, col2)
        top_left_sum = self.sumOriginToPosition(top_left)
        top_right_sum = self.sumOriginToPosition(top_right)
        bottom_left_sum = self.sumOriginToPosition(bottom_left)
        bottom_right_sum = self.sumOriginToPosition(bottom_right)
        return bottom_right_sum - bottom_left_sum - top_right_sum + top_left_sum

    def update(self, row, col, val):
        diff = val - self.sumRegion(row, col, row, col)
        n, m = len(self.bit_matrix), len(self.bit_matrix[0])
        rIdx = row + 1
        while rIdx < n:
            cIdx = col + 1
            while cIdx < m:
                self.bit_matrix[rIdx][cIdx] += diff
                cIdx += cIdx & -cIdx
            rIdx += rIdx & -rIdx


class RangeSumQuerySpec(unittest.TestCase):
    def test_example(self):
        matrix = [
            [3, 0, 1, 4, 2],
            [5, 6, 3, 2, 1],
            [1, 2, 0, 1, 5],
            [4, 1, 0, 1, 7],
            [1, 0, 3, 0, 5]
        ]
        rsq = RangeSumQuery(matrix)
        self.assertEqual(rsq.sumRegion(2, 1, 4, 3), 8)
        rsq.update(3, 2, 2)
        self.assertEqual(rsq.sumRegion(2, 1, 4, 3), 10)

    def test_non_square_matrix(self):
        matrix = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]
        rsq = RangeSumQuery(matrix)
        self.assertEqual(rsq.sumRegion(0, 1, 1, 3), 6)
        rsq.update(0, 2, 2)
        self.assertEqual(rsq.sumRegion(0, 1, 1, 3), 7)


if __name__ == '__main__':
    unittest.main(exit=False)
```
### Additional Question: LC 54 \[Medium\] Spiral Matrix 
---
> **Question:** Given a matrix of n x m elements (n rows, m columns), return all elements of the matrix in spiral order.

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

**Solution:** [https://repl.it/@trsong/Spiral-Matrix](https://repl.it/@trsong/Spiral-Matrix)
```py
import unittest

def spiral_order(matrix):
    if not matrix or not matrix[0]: return []
    n, m = len(matrix), len(matrix[0])
    row_lower_bound = 0
    row_upper_bound = n - 1
    col_lower_bound = 0
    col_upper_bound = m - 1
    res = []
        
    while row_lower_bound < row_upper_bound and col_lower_bound < col_upper_bound:
        r, c = row_lower_bound, col_lower_bound
        # top
        while c < col_upper_bound:
            res.append(matrix[r][c])
            c += 1
        
        # right
        while r < row_upper_bound:
            res.append(matrix[r][c])
            r += 1
        
        # bottom
        while c > col_lower_bound:
            res.append(matrix[r][c])
            c -= 1
        
        # left 
        while r > row_lower_bound:
            res.append(matrix[r][c])
            r -= 1
    
        col_upper_bound -= 1
        col_lower_bound += 1
        row_upper_bound -= 1
        row_lower_bound += 1
        
    r, c = row_lower_bound, col_lower_bound
    if row_lower_bound == row_upper_bound: 
        # Edge Case 1: when remaining block is 1xk
        for col in xrange(col_lower_bound, col_upper_bound + 1):
            res.append(matrix[r][col])
    elif col_lower_bound == col_upper_bound:
        # Edge Case 2: when remaining block is kx1
        for row in xrange(row_lower_bound, row_upper_bound + 1):
            res.append(matrix[row][c])
            
    return res


class SpiralOrderSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(spiral_order([
            [ 1, 2, 3 ],
            [ 4, 5, 6 ],
            [ 7, 8, 9 ]
        ]), [1, 2, 3, 6, 9, 8, 7, 4, 5])

    def test_example2(self):
        self.assertEqual(spiral_order([
            [1,  2,  3,  4],
            [5,  6,  7,  8],
            [9, 10, 11, 12]
        ]), [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7])

    def test_empty_table(self):
        self.assertEqual(spiral_order([]), [])
        self.assertEqual(spiral_order([[]]), [])

    def test_two_by_two_table(self):
        self.assertEqual(spiral_order([
            [1, 2],
            [4, 3]
        ]), [1, 2, 3, 4])

    def test_one_element_table(self):
        self.assertEqual(spiral_order([[1]]), [1])

    def test_one_by_k_table(self):
        self.assertEqual(spiral_order([
            [1, 2, 3, 4]
        ]), [1, 2, 3, 4])

    def test_k_by_one_table(self):
        self.assertEqual(spiral_order([
            [1],
            [2],
            [3]
        ]), [1, 2, 3])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 9, 2019 LC 307 \[Medium\] Range Sum Query - Mutable
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

**Solution with Binary Indexed Tree:** [https://repl.it/@trsong/Range-Sum-Query-Mutable](https://repl.it/@trsong/Range-Sum-Query-Mutable)
```py
import unittest

class RangeSumQuery(object):
    @staticmethod
    def last_bit(num):
        return num & -num

    def __init__(self, nums):
        n = len(nums)
        self._BITree = [0] * (n + 1)
        for i in xrange(n):
            self.update(i, nums[i])

    def prefixSum(self, i):
        """Get sum of value from index 0 to i """
        # BITree starts from index 1
        index = i + 1
        res = 0
        while index > 0:
            res += self._BITree[index]
            index -= RangeSumQuery.last_bit(index)
        return res

    def rangeSum(self, i, j):
        return self.prefixSum(j) - self.prefixSum(i-1)

    def update(self, i, val):
        """Update the sum by add delta on top of result"""
        # BITree starts from index 1
        delta = val - self.rangeSum(i, i)
        index = i + 1
        while index < len(self._BITree):
            self._BITree[index] += delta
            index += RangeSumQuery.last_bit(index)


class RangeSumQuerySpec(unittest.TestCase):
    def test_example(self):
        rsq = RangeSumQuery([1, 3, 5])
        self.assertEqual(rsq.rangeSum(0, 2), 9)
        rsq.update(1, 2)
        self.assertEqual(rsq.rangeSum(0, 2), 8)

    def test_one_elem_array(self):
        rsq = RangeSumQuery([8])
        rsq.update(0, 2)
        self.assertEqual(rsq.rangeSum(0, 0), 2)

    def test_update_all_elements(self):
        req = RangeSumQuery([1, 4, 2, 3])
        self.assertEqual(req.rangeSum(0, 3), 10)
        req.update(0, 0)
        req.update(2, 0)
        req.update(1, 0)
        req.update(3, 0)
        self.assertEqual(req.rangeSum(0, 3), 0)
        req.update(2, 1)
        self.assertEqual(req.rangeSum(0, 1), 0)
        self.assertEqual(req.rangeSum(1, 2), 1)
        self.assertEqual(req.rangeSum(2, 3), 1)
        self.assertEqual(req.rangeSum(3, 3), 0)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Additional Question: LC 114 \[Medium\] Flatten Binary Tree to Linked List
---
> **Question:** Given a binary tree, flatten it to a linked list in-place.
>
> For example, given the following tree:

```py
    1
   / \
  2   5
 / \   \
3   4   6
```
>
> The flattened tree should look like:

```py
1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
```

**Solution with Recursion:** [https://repl.it/@trsong/Flatten-Binary-Tree-to-Linked-List](https://repl.it/@trsong/Flatten-Binary-Tree-to-Linked-List)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __eq__(self, other):
        return other and other.val == self.val and other.left == self.left and other.right == self.right


def flatten(tree):
    flatten_and_get_leaf(tree)


def flatten_and_get_leaf(tree):
    # flatten the tree and return the leaf node of that tree
    if not tree or not tree.left and not tree.right: return tree
    right_leaf = flatten_and_get_leaf(tree.right)
    left_leaf = flatten_and_get_leaf(tree.left)

    if left_leaf:
        # append right tree to the end of left tree leaf 
        left_leaf.right = tree.right
    if tree.left:
        # move left tree to right tree
        tree.right = tree.left
    tree.left = None

    # return tree right leaf if exits otherwise return left one 
    return right_leaf if right_leaf else left_leaf


class FlattenSpec(unittest.TestCase):
    def list_to_tree(self, lst):
        p = dummy = TreeNode(-1)
        for num in lst:
            p.right = TreeNode(num)
            p = p.right
        return dummy.right

    def test_example(self):
        """
            1
           / \
          2   5
         / \   \
        3   4   6
        """
        n2 = TreeNode(2, TreeNode(3), TreeNode(4))
        n5 = TreeNode(5, right = TreeNode(6))
        tree = TreeNode(1, n2, n5)
        flatten_list = [1, 2, 3, 4, 5, 6]
        flatten(tree)
        self.assertEqual(tree, self.list_to_tree(flatten_list))

    def test_empty_tree(self):
        tree = None
        flatten(tree)
        self.assertIsNone(tree)

    def test_only_right_child(self):
        """
        1
         \
          2
           \
            3
        """
        n2 = TreeNode(2, right=TreeNode(3))
        tree = TreeNode(1, right = n2)
        flatten_list = [1, 2, 3]
        flatten(tree)
        self.assertEqual(tree, self.list_to_tree(flatten_list))

    def test_only_left_child(self):
        """
            1
           /
          2
         /
        3
        """
        n2 = TreeNode(2, TreeNode(3))
        tree = TreeNode(1, n2)
        flatten_list = [1, 2, 3]
        flatten(tree)
        self.assertEqual(tree, self.list_to_tree(flatten_list))  


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 8, 2019 \[Medium\] Delete Columns to Make Sorted II
---
> **Question:** You are given an N by M 2D matrix of lowercase letters. The task is to count the number of columns to be deleted so that all the rows are lexicographically sorted.

**Example 1:**
```
Given the following table:
hello
geeks

Your function should return 1 as deleting column 1 (index 0)
Now both strings are sorted in lexicographical order:
ello
eeks
```

**Example 2:**
```
Given the following table:
xyz
lmn
pqr

Your function should return 0. All rows are already sorted lexicographically.
```

**My thoughts:** This problem feels like 2D version of ***The Longest Increasing Subsequence Problem*** (LIP) (check Jul 2, 2019 problem for details). The LIP says find longest increasing subsequence. e.g. `01212342345` gives `01[2]1234[23]45`, with `2,2,3` removed. So if we only have 1 row, we can simply find longest increasing subsequence and use that to calculate how many columns to remove i.e. `# of columns to remove = m - LIP`. Similarly, for n by m table, we can first find longest increasing sub-columns and use that to calculate which columns to remove. Which can be done using DP:

let `dp[i]` represents max number of columns to keep at ends at column i. 
- `dp[i] = max(dp[j]) + 1 where j < i` if all characters in column `i` have lexicographical order larger than column `j`
- `dp[0] = 1`


**Solution with DP:** [https://repl.it/@trsong/Delete-Columns-to-Make-Sorted-II](https://repl.it/@trsong/Delete-Columns-to-Make-Sorted-II)
```py
import unittest

def delete_column(table):
    # let dp[i] represents max number of columns to keep at ends at column i
    # i.e. column i is the last column to keep
    # dp[i] = max(dp[j]) + 1 where j < i  if all e in column i have lexicographical order larger than column j
    m = len(table[0])
    dp = [1] * m
    for i in xrange(1, m):
        for j in xrange(i):
            if all(r[j] <= r[i] for r in table):
                dp[i] = max(dp[i], dp[j] + 1)
    return m - max(dp)


class DeleteColumnSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(delete_column([
            'hello',
            'geeks'
        ]), 1)

    def test_example2(self):
        self.assertEqual(delete_column([
            'xyz',
            'lmn',
            'pqr'
        ]), 0)

    def test_table_with_one_row(self):
        self.assertEqual(delete_column([
            '01212342345' # 01[2]1234[23]45   
        ]), 3)  

    def test_table_with_two_rows(self):
        self.assertEqual(delete_column([
            '01012',  # [0] 1 [0] 1 2
            '20101'   # [2] 0 [1] 0 1
        ]), 2)  


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Additional Question: LT 640 \[Medium\] One Edit Distance
---
> **Question:** Given two strings S and T, determine if they are both one edit distance apart.

**Example 1:**

```
Input: s = "aDb", t = "adb" 
Output: True
```

**Example 2:**
```
Input: s = "ab", t = "ab" 
Output: False
Explanation:
s=t, so they aren't one edit distance apart
```

**My thougths:** The trick for this problem is that one insertion of shorter string is equivalent to one removal of longer string. And if both string of same length, then we only allow one replacement. The other cases are treated as more than one edit distance between two strings.

**Solution:** [https://repl.it/@trsong/One-Edit-Distance](https://repl.it/@trsong/One-Edit-Distance)
```py
import unittest

def is_one_edit_distance_between(s1, s2):
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > 1: return False
    i = distance = 0
    (shorter_str, longer_str) = (s1, s2) if len1 < len2 else (s2, s1)
    len_shorter_str = len(shorter_str)
    for c in longer_str:
        if distance > 1: return False
        if i >= len_shorter_str or c != shorter_str[i]:
            distance += 1
            if len1 != len2:
                # when two strings of different length and there is mismatch, only proceed the pointer to longer string
                i -= 1
        i += 1
    return distance == 1


class IsOneEditDistanceBetweenSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(is_one_edit_distance_between('aDb', 'adb'))

    def test_example2(self):
        self.assertFalse(is_one_edit_distance_between('ab', 'ab'))

    def test_empty_string(self):
        self.assertFalse(is_one_edit_distance_between('', ''))
        self.assertTrue(is_one_edit_distance_between('', 'a'))
        self.assertFalse(is_one_edit_distance_between('', 'ab'))

    def test_one_insert_between_two_strings(self):
        self.assertTrue(is_one_edit_distance_between('abc', 'ac'))
        self.assertFalse(is_one_edit_distance_between('abcd', 'ad'))

    def test_one_remove_between_two_strings(self):
        self.assertTrue(is_one_edit_distance_between('abcd', 'abd'))
        self.assertFalse(is_one_edit_distance_between('abcd', 'cd'))

    def test_one_replace_between_two_string(self):
        self.assertTrue(is_one_edit_distance_between('abc', 'abd'))
        self.assertFalse(is_one_edit_distance_between('abc', 'ddc'))

    def test_length_difference_greater_than_one(self):
        self.assertFalse(is_one_edit_distance_between('abcd', 'abcdef'))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 7, 2019 \[Easy\] Delete Columns to Make Sorted I
---
> **Question:** You are given an N by M 2D matrix of lowercase letters. Determine the minimum number of columns that can be removed to ensure that each row is ordered from top to bottom lexicographically. That is, the letter at each column is lexicographically later as you go down each row. It does not matter whether each row itself is ordered lexicographically.

**Example 1:**

```
Given the following table:
cba
daf
ghi

This is not ordered because of the a in the center. We can remove the second column to make it ordered:
ca
df
gi

So your function should return 1, since we only needed to remove 1 column.
```

**Example 2:**


```
Given the following table:
abcdef

Your function should return 0, since the rows are already ordered (there's only one row).
```

**Example 3:**

```
Given the following table:
zyx
wvu
tsr

Your function should return 3, since we would need to remove all the columns to order it.
```

**My thoughts:** Wordy this problem may be, but extremely easy to be solved. We will take a look at *Delete Columns to Make Sorted II* tomorrow.

**Solution:** [https://repl.it/@trsong/Delete-Columns-to-Make-Sorted-I](https://repl.it/@trsong/Delete-Columns-to-Make-Sorted-I)
```py
import unittest

def columns_to_delete(table):
    if not table or not table[0]: return 0
    n, m = len(table), len(table[0])
    count = 0
    for c in xrange(m):
        for r in xrange(1, n):
            if table[r][c] < table[r-1][c]:
                count += 1
                break
    return count

class ColumnToDeleteSpec(unittest.TestCase):
    def test_empty_table(self):
        self.assertEqual(columns_to_delete([]), 0)
        self.assertEqual(columns_to_delete([""]), 0)

    def test_example1(self):
        self.assertEqual(columns_to_delete([
            'cba',
            'daf',
            'ghi'
        ]), 1)

    def test_example2(self):
        self.assertEqual(columns_to_delete([
            'abcdef'
        ]), 0)

    def test_example3(self):
        self.assertEqual(columns_to_delete([
            'zyx',
            'wvu',
            'tsr'
        ]), 3)


if __name__ == '__main__':
    unittest.main(exit=False)     
```

### Aug 6, 2019 LC 236 \[Medium\] Lowest Common Ancestor of a Binary Tree
---
> **Question:** Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

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

**My thoughts:** Notice that only the nodes at the same level can find common ancestor with same number of tracking backward. e.g. Consider 3 and 4 in above example: the common ancestor is 1, 3 needs 1 tracking backward, but 4 need 2 tracking backward. So the idea is to move those two nodes to the same level and then tacking backward until hit the common ancestor. The algorithm works as below:

We can use BFS to find target nodes and their depth. And by tracking backward the parent of the deeper node, we can make sure both of nodes are on the same level. Finally, we can tracking backwards until hit a common ancestor. 

**Solution with BFS and Backward Tracking Ancestor:** [https://repl.it/@trsong/Lowest-Common-Ancestor-of-a-Binary-Tree](https://repl.it/@trsong/Lowest-Common-Ancestor-of-a-Binary-Tree)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def LCA(tree, v1, v2):
    if v1 == v2: return v1
    parent = {}
    n1 = n2 = None
    lv1 = lv2 = lv = 0
    queue = [tree]
    
    # Run BFS to find node with value v1 and v2 and its depth
    while queue and (n1 is None or n2 is None):
        level_size = len(queue)
        for _ in xrange(level_size):
            node = queue.pop(0)
            if node.val == v1:
                n1 = node
                lv1 = lv
            elif node.val == v2:
                n2 = node
                lv2 = lv
            
            if node.left:
                parent[node.left] = node
                queue.append(node.left)
            if node.right:
                parent[node.right] = node
                queue.append(node.right)
        lv += 1
    
    # Backtrack the parent of deeper node up until at the same level as the other node
    (deeper_node, other_node) = (n1, n2) if lv1 > lv2 else (n2, n1)
    for _ in xrange(abs(lv1 - lv2)):
        deeper_node = parent[deeper_node]

    # Find the ancestor of both nodes recursively until find the common ancestor
    while deeper_node != other_node:
        deeper_node = parent[deeper_node]
        other_node = parent[other_node]

    return deeper_node.val


class LCASpec(unittest.TestCase):
    def setUp(self):
        """
             1
           /   \
          2     3
         / \   / \
        4   5 6   7
        """
        n2 = TreeNode(2, TreeNode(4), TreeNode(5))
        n3 = TreeNode(3, TreeNode(6), TreeNode(7))
        self.tree = TreeNode(1, n2, n3)

    def test_both_nodes_on_leaves(self):
        self.assertEqual(LCA(self.tree, 4, 5), 2)
        self.assertEqual(LCA(self.tree, 6, 7), 3)
        self.assertEqual(LCA(self.tree, 4, 6), 1)

    def test_nodes_on_different_levels(self):
        self.assertEqual(LCA(self.tree, 4, 2), 2)
        self.assertEqual(LCA(self.tree, 4, 3), 1)
        self.assertEqual(LCA(self.tree, 4, 1), 1)

    def test_same_nodes(self):
        self.assertEqual(LCA(self.tree, 2, 2), 2)
        self.assertEqual(LCA(self.tree, 6, 6), 6)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 5, 2019 \[Easy\] Single Bit Switch
---
> **Question:** Given three 32-bit integers x, y, and b, return x if b is 1 and y if b is 0, using only mathematical or bit operations. You can assume b can only be 1 or 0.

**Solution:** [https://repl.it/@trsong/Single-Bit-Switch](https://repl.it/@trsong/Single-Bit-Switch)
```py
import unittest

def single_bit_switch_1(b, x, y):
    return b * x + (1 - b) * y


def single_bit_switch_2(b, x, y):
    # When b = 0001b,
    # -b = 1111b, ~-b = 0000b
    # When b = 0000b
    # -b = 0000b, ~-b = 1111b
    return (x & -b) | (y & ~-b) 


class SingleBitSwitchSpec(unittest.TestCase):
    def test_b_is_zero(self):
        b, x, y = 0, 8, 16
        self.assertEqual(single_bit_switch_1(b, x, y), y)
        self.assertEqual(single_bit_switch_2(b, x, y), y)

    def test_b_is_one(self):
        b, x, y = 1, 8, 16
        self.assertEqual(single_bit_switch_1(b, x, y), x)
        self.assertEqual(single_bit_switch_2(b, x, y), x)

    def test_negative_numbers(self):
        b0, b1, x, y = 0, 1, -1, -2
        self.assertEqual(single_bit_switch_1(b0, x, y), y)
        self.assertEqual(single_bit_switch_2(b1, x, y), x)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 4, 2019 LC 392 \[Medium\] Is Subsequence
---
> **Question:** Given a string s and a string t, check if s is subsequence of t.
>
> A subsequence of a string is a new string which is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (ie, "ace" is a subsequence of "abcde" while "aec" is not).

**Example 1:**
```
s = "abc", t = "ahbgdc"
Return true.
```

**Example 2:**
```
s = "axc", t = "ahbgdc"
Return false.
```

**My thoughts:** Just base on the definition of subsequence, mantaining an pointer of s to the next letter we are about to checking and check it against all letter of t.

**Solution:** [https://repl.it/@trsong/Is-Subsequence](https://repl.it/@trsong/Is-Subsequence)

```py
import unittest

def isSubsequence(s, t):
    if len(s) > len(t): return False
    if not len(s): return True
    i = 0
    for c in t:
        if i >= len(s):
            break
        if c == s[i]:
            i += 1
    return i >= len(s)


class IsSubsequenceSpec(unittest.TestCase):
    def test_empty_s(self):
        self.assertTrue(isSubsequence("", ""))
        self.assertTrue(isSubsequence("", "a"))

    def test_empty_t(self):
        self.assertFalse(isSubsequence("a", ""))

    def test_s_longer_than_t(self):
        self.assertFalse(isSubsequence("ab", "a"))

    def test_size_one_input(self):
        self.assertTrue(isSubsequence("a", "a"))
        self.assertFalse(isSubsequence("a", "b"))

    def test_end_with_same_letter(self):
        self.assertTrue(isSubsequence("ab", "aaaaccb"))

    def test_example(self):
        self.assertTrue(isSubsequence("abc", "ahbgdc"))

    def test_example2(self):
        self.assertFalse(isSubsequence("axc", "ahbgdc"))


if __name__ == '__main__':
    unittest.main(exit=False)
```
### Aug 3, 2019 \[Medium\] Toss Biased Coin
---
> **Question:** Assume you have access to a function toss_biased() which returns 0 or 1 with a probability that's not 50-50 (but also not 0-100 or 100-0). You do not know the bias of the coin. Write a function to simulate an unbiased coin toss.

**My thoughts:** Suppose the biased toss has probablilty p to return 0 and (1-p) to get 1. Then the probability to get:

- 0, 0 is `p * p`
- 1, 1 is `(1-p) * (1-p)`
- 1, 0 is `(1-p) * p`
- 0, 1 is `p * (1-p)`
  
Thus we can take advantage that 1, 0 and 0, 1 has same probility to get unbiased toss. Of course, above logic works only if p is neither 0% nor 100%. 


**Solution:** [https://repl.it/@trsong/Toss-Biased-Coin](https://repl.it/@trsong/Toss-Biased-Coin)
```py
from random import randint

def toss_biased():
    # suppose the toss has 1/4 chance to get 0 and 3/4 to get 1
    return 0 if randint(0, 3) == 0 else 1


def toss_unbiased():
    while True:
        t1 = toss_biased()
        t2 = toss_biased()
        if t1 != t2:
            return t1


def print_distribution(repeat):
    histogram = {}
    for _ in xrange(repeat):
        res = toss_unbiased()
        if res not in histogram:
            histogram[res] = 0
        histogram[res] += 1
    print histogram


def main():
    # Distribution looks like {0: 99931, 1: 100069}
    print_distribution(repeat=200000)


if __name__ == '__main__':
    main()
```

### Aug 2, 2019 \[Medium\] The Tower of Hanoi
---
> **Question:**  The Tower of Hanoi is a puzzle game with three rods and n disks, each a different size.
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

```
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

**Solution with Recursion:** [https://repl.it/@trsong/The-Tower-of-Hanoi](https://repl.it/@trsong/The-Tower-of-Hanoi)
```py
import unittest

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

    def can_moves_finish_game(self, actions):
        self.reset()
        for src, dst in actions:
            if not self.is_feasible_move(src, dst):
                return False
            else:
                self.move(src, dst)
        return self.is_game_finished()
    

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


def hanoi_moves(n):
    moves = []

    def hanoi_moves_recur(n, src, dst):
        if n <= 0: return
        other = 3 - src - dst

        # Step1: move n - 1 disks from src to 'other' to allow last disk move to dst
        hanoi_moves_recur(n-1, src, other)

        # Step2: move last disk from src to dst
        moves.append((src, dst))

        # Step3: move n - 1 disks from 'other' to dst
        hanoi_moves_recur(n-1, other, dst)

    hanoi_moves_recur(n, 0, 2)
    return moves


class HanoiMoveSpec(unittest.TestCase):
    def assert_hanoi_moves(self, n, moves):
        game = HanoiGame(n)
        self.assertTrue(game.can_moves_finish_game(moves))

    def test_three_disks(self):
        moves = hanoi_moves(3)
        self.assert_hanoi_moves(3, moves)

    def test_one_disk(self):
        moves = hanoi_moves(1)
        self.assert_hanoi_moves(1, moves)

    def test_ten_disks(self):
        moves = hanoi_moves(10)
        self.assert_hanoi_moves(10, moves)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 1, 2019 \[Medium\] All Root to Leaf Paths in Binary Tree
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

**Solution with Recursion:** [https://repl.it/@trsong/All-Root-to-Leaf-Paths-in-Binary-Tree](https://repl.it/@trsong/All-Root-to-Leaf-Paths-in-Binary-Tree)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right= right


def path_to_leaves(tree):
    if not tree: return []
    res = []
    def path_to_leaves_recur(tree, root_to_parent):
        root_to_current = root_to_parent + [tree.val]
        if not tree.left and not tree.right: 
            res.append(root_to_current)
        else:
            if tree.left:
                path_to_leaves_recur(tree.left, root_to_current)
            if tree.right:
                path_to_leaves_recur(tree.right, root_to_current)
    
    path_to_leaves_recur(tree, [])
    return res


class PathToLeavesSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertEqual(path_to_leaves(None), [])

    def test_one_level_tree(self):
        self.assertEqual(path_to_leaves(TreeNode(1)), [[1]])

    def test_two_level_tree(self):
        tree = TreeNode(1, TreeNode(2), TreeNode(3))
        self.assertEqual(path_to_leaves(tree), [[1, 2], [1, 3]])

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
        self.assertEqual(path_to_leaves(tree), [[1, 2], [1, 3, 4], [1, 3, 5]])


if __name__ == '__main__':
    unittest.main(exit=False)
```
