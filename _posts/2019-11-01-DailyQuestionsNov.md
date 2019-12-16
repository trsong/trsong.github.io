---
layout: post
title:  "Daily Coding Problems Nov to Jan"
date:   2019-11-01 22:22:32 -0700
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

### Dec 16, 2019 \[Medium\] Bridges in a Graph
--- 
> **Question:** A bridge in a connected (undirected) graph is an edge that, if removed, causes the graph to become disconnected. Find all the bridges in a graph.

### Dec 15, 2019 \[Hard\] De Bruijn Sequence 
--- 
> **Question:** Given a set of characters C and an integer k, a De Bruijn Sequence is a cyclic sequence in which every possible k-length string of characters in C occurs exactly once.
> 
> **Background:** De Bruijn Sequence can be used to shorten a brute-force attack on a PIN-like code lock that does not have an "enter" key and accepts the last n digits entered. For example, a digital door lock with a 4-digit code would have B (10, 4) solutions, with length 10000. Therefore, only at most 10000 + 3 = 10003 (as the solutions are cyclic) presses are needed to open the lock. Trying all codes separately would require 4 × 10000 = 40000 presses.

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

**My thoughts:** Treat all substring as nodes, substr1 connect to substr2 if substr1 shift 1 become substr2. Eg.  `123 -> 234 -> 345`. In order to only include each substring only once, we traverse entire graph using DFS and mark visited nodes and avoid visit same node over and over again.  

**Solution with DFS:** [https://repl.it/@trsong/De-Bruijn-Sequence](https://repl.it/@trsong/De-Bruijn-Sequence)
```py
import unittest

def de_bruijn_sequence(char_set, k):
    char_set = map(str, char_set)
    begin_state = char_set[0] * k
    visited = set()
    stack = [(begin_state, begin_state)]
    res = []
    
    while stack:
        cur_state, appended_char = stack.pop()
        if cur_state in visited:
            continue
        visited.add(cur_state)
        res.append(appended_char)

        for char in char_set:
            next_state = cur_state[1:] + char
            if next_state not in visited:
                stack.append((next_state, char))
    
    return "".join(res)


class DeBruijnSequenceSpec(unittest.TestCase):
    @staticmethod
    def cartesian_product(char_set, k):
        def cartesian_product_recur(char_set, k):
            if k == 1:
                return [[str(c)] for c in char_set]
            res = []
            for accu_list in cartesian_product_recur(char_set, k-1):
                for char in char_set:
                    res.append(accu_list + [str(char)])
            return res
        
        return map(lambda lst: ''.join(lst), cartesian_product_recur(char_set, k))

    def validate_de_bruijn_seq(self, char_set, k, seq_res):
        n = len(char_set)
        expected_substr_set = set(DeBruijnSequenceSpec.cartesian_product(char_set, k))
        result_substr_set = set()
        for i in xrange(n**k):
            result_substr_set.add(seq_res[i:i+k])
        # Check if all substr are covered
        self.assertEqual(expected_substr_set, result_substr_set)
        
    def test_example1(self):
        k, char_set = 3, [0, 1]
        res = de_bruijn_sequence(char_set, k)
        # Possible Solution: "0011101000"
        self.validate_de_bruijn_seq(char_set, k, res)

    def test_example2(self):
        k, char_set = 2, [0, 1]
        res = de_bruijn_sequence(char_set, k)
        # Possible Solution: "01100"
        self.validate_de_bruijn_seq(char_set, k, res)

    def test_multi_charset(self):
        k, char_set = 2, [0, 1, 2]
        res = de_bruijn_sequence(char_set, k)
        # Possible Solution : "0022120110"
        self.validate_de_bruijn_seq(char_set, k, res)

    def test_multi_charset2(self):
        k, char_set = 3, [0, 1, 2]
        res = de_bruijn_sequence(char_set, k)
        # Possible Solution : "00022212202112102012001110100"
        self.validate_de_bruijn_seq(char_set, k, res)
        
    def test_larger_k(self):
        k, char_set = 5, [0, 1, 2]
        res = de_bruijn_sequence(char_set, k)
        self.validate_de_bruijn_seq(char_set, k, res)

        
if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 14, 2019 \[Medium\] Generate Binary Search Trees
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

**Solution with Recursion:** [https://repl.it/@trsong/Generate-Binary-Search-Trees](https://repl.it/@trsong/Generate-Binary-Search-Trees)
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
    for i in xrange(lo, hi+1):
        left_trees = generate_bst_recur(lo, i-1)
        right_trees = generate_bst_recur(i+1, hi)
        for left in left_trees:
            for right in right_trees:
                res.append(TreeNode(i, left, right))
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
    unittest.main(exit=False)
```

### Dec 13, 2019 \[Hard\] The Most Efficient Way to Sort a Million 32-bit Integers
--- 
> **Question:** Given an array of a million integers between zero and a billion, out of order, how can you efficiently sort it?

**My thoughts:** Inspired by this article [What is the most efficient way to sort a million 32-bit integers?](https://www.quora.com/What-is-the-most-efficient-way-to-sort-a-million-32-bit-integers).

We all know that comparison-based takes `O(n*logn)` time is worse than radix sort `O(n)`. But sometime we cannot neglect the size of data. The complexity of sorting 32bit number is `O(lg(domainSize)∗n)` in this case `O(32∗n)` or `o(32n)`. But `O(n∗lg(n))` is better than `O(n)`. `O(n∗lg(n))=O(n∗lg(1M)) = o(20n)`. 

Can we do better for radix sort to achieve better than `o(20n)`? Yes, we can! It turns out with 1 digit bucket we can achieve `o(32(n+2)) = o(32n) + merge overhead`. 2 digit bucket, `o(16n) + merge overhead`. 3 digit, `o(8n) + merge overhead`. We cannot go beyond that as the merge cost is increasing dramatically. After some experiment, it seems 3 digit bucket is optimal.

**Solution by Radix Sort:** [https://repl.it/@trsong/Sort-a-Million-32-bit-Integers](https://repl.it/@trsong/Sort-a-Million-32-bit-Integers)
```py
import unittest
import random

MAX_32_BIT_INT = 2 ** 33 - 1

def radix_sort_by_digits(nums, num_digit):
    if not nums:
        return []

    min_val = float('inf')
    max_val = float('-inf')
    for num in nums:
        if num < min_val:
            min_val = num
        elif num > max_val:
            max_val = num
    
    bucket = [0] * (1 << num_digit)
    radix = 1 << num_digit
    sorted_nums = [None] * len(nums)
    iteration = 0
    while (max_val - min_val) >> (num_digit * iteration) > 0:
        shift_amount = num_digit * iteration

        for i in xrange(len(bucket)):
            bucket[i] = 0

        # Count occurance
        for num in nums:
            shifted_num = (num - min_val) >> shift_amount
            bucket_index = shifted_num % radix
            bucket[bucket_index] += 1

        # Accumulate occurance to get actual index mappping
        for i in xrange(1, len(bucket)):
            bucket[i] += bucket[i-1]

        # Copy back record
        for i in xrange(len(nums)-1, -1, -1):
            shifted_num = (nums[i] - min_val) >> shift_amount
            bucket_index = shifted_num % radix
            bucket[bucket_index] -= 1
            sorted_nums[bucket[bucket_index]] = nums[i]
        
        iteration += 1
        nums, sorted_nums = sorted_nums, nums
    return nums


def sort_32bit(nums):
    # 3 digit bucket is optimal for 1M 32bit integers
    return radix_sort_by_digits(nums, 3)


class Sort32BitSpec(unittest.TestCase):
    random.seed(42)

    def test_sort_empty_array(self):
        self.assertEqual([], sort_32bit([]))

    def test_small_array(self):
        self.assertEqual([1, 1, 2, 3, 4, 5, 6], sort_32bit([6, 2, 3, 1, 4, 1, 5]))

    def test_larger_array(self):
        nums = [random.randint(0, MAX_32_BIT_INT) for _ in xrange(1000)]
        sorted_result = sorted(nums)
        self.assertEqual(sorted_result, sort_32bit(nums))

    def test_sort_1m_numbers(self):
        nums = [random.randint(0, MAX_32_BIT_INT) for _ in xrange(1000000)]
        sorted_result = sorted(nums)
        self.assertEqual(sorted_result, sort_32bit(nums))


if __name__ == '__main__':    
    unittest.main(exit=False)
```

### Dec 12, 2019 \[Medium\] Sorting Window Range
--- 
> **Question:** Given a list of numbers, find the smallest window to sort such that the whole list will be sorted. If the list is already sorted return (0, 0). 

**Example:**
```py
Input: [2, 4, 7, 5, 6, 8, 9]
Output: (2, 4)
Explanation: Sorting the window (2, 4) which is [7, 5, 6] will also means that the whole list is sorted.
```

**My thoughts:** A sorted array has no min range to sort. So we want first identity the range `(i, j)` that goes wrong, that is, we want to identify first `i` and last `j` that makes array not sorted. ie. smallest `i` such that `nums[i] > nums[i+1]`, largest `j` such that `nums[j] < nums[j-1]`. 

Secondly, range `(i, j)` inclusive is where we should start. And there could be number smaller than `nums[i+1]` and bigger than `nums[j-1]`, therefore we need to figure out how we can release the boundary of `(i, j)` to get `(i', j')` where `i' <= i` and `j' <= j` so that `i'`, `j'` covers those smallest and largest number within `(i, j)`. 

After doing that, we will get smallest range to make original array sorted, the range is `i'` through `j'` inclusive.

**Solution:** [https://repl.it/@trsong/Sorting-Window-Range](https://repl.it/@trsong/Sorting-Window-Range)

```py
import unittest

def sort_window_range(nums):
    if len(nums) <= 1:
        return 0, 0

    n = len(nums)
    left = 0
    right = n - 1
    while left < n - 1 and nums[left] <= nums[left+1]:
        left += 1

    
    if left == n - 1:
        # nums is already sorted
        return 0, 0

    while right > left and nums[right-1] <= nums[right]:
        right -= 1
    
    min_val = float('inf')
    max_val = float('-inf')
    for i in xrange(left, right+1):
        num = nums[i]
        if num < min_val:
            min_val = num
        if num > max_val:
            max_val = num
    
    while left > 0 and nums[left-1] > min_val:
        left -= 1
    
    while right < n - 1 and nums[right+1] < max_val:
        right += 1

    return left, right


class SortWindowRangeSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual((2, 4), sort_window_range([2, 4, 7, 5, 6, 8, 9]))

    def test_example1(self):
        self.assertEqual((1, 5), sort_window_range([1, 7, 9, 5, 7, 8, 10]))
        
    def test_example2(self):
        self.assertEqual((3, 8), sort_window_range([10, 12, 20, 30, 25, 40, 32, 31, 35, 50, 60]))

    def test_example3(self):
        self.assertEqual((2, 5), sort_window_range([0, 1, 15, 25, 6, 7, 30, 40, 50]))

    def test_empty_array(self):
        self.assertEqual((0, 0), sort_window_range([]))

    def test_already_sorted_array(self):
        self.assertEqual((0, 0), sort_window_range([1, 2, 3, 4]))

    def test_array_contains_one_elem(self):
        self.assertEqual((0, 0), sort_window_range([42]))

    def test_reverse_sorted_array(self):
        self.assertEqual((0, 3), sort_window_range([4, 3, 2, 1]))

    def test_table_shape_array(self):
        self.assertEqual((2, 5), sort_window_range([1, 2, 3, 3, 3, 2]))

    def test_increase_decrease_then_increase(self):
        self.assertEqual((2, 6), sort_window_range([1, 2, 3, 4, 3, 2, 3, 4, 5, 6]))

    def test_increase_decrease_then_increase2(self):
        self.assertEqual((0, 4), sort_window_range([0, 1, 2, -1, 1, 2]))

    def test_increase_decrease_then_increase3(self):
        self.assertEqual((0, 6), sort_window_range([0, 1, 2, 99, -99, 1, 2]))
        self.assertEqual((0, 6), sort_window_range([0, 1, 2, -99, 99, 1, 2]))
    
    def test_array_contains_duplicated_numbers(self):
        self.assertEqual((0, 5), sort_window_range([1, 1, 1, 0, -1, -1, 1, 1, 1]))

    def test_array_contains_one_outlier(self):
        self.assertEqual((3, 6), sort_window_range([0, 0, 0, 1, 0, 0, 0]))
        self.assertEqual((0, 3), sort_window_range([0, 0, 0, -1, 0, 0, 0]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 11, 2019 \[Easy\] Jumbled Sequence
--- 
> **Question:** The sequence [0, 1, ..., N] has been jumbled, and the only clue you have for its order is an array representing whether each number is larger or smaller than the last. Given this information, reconstruct an array that is consistent with it. For example, given [None, +, +, -, +], you could return [1, 2, 3, 0, 4]

**Solution:** [https://repl.it/@trsong/Jumbled-Sequence](https://repl.it/@trsong/Jumbled-Sequence)
```py
import unittest

def generate_sequence(jumbled_seq):
    if not jumbled_seq:
        return []
    num_increment = jumbled_seq.count('+')
    n = len(jumbled_seq)
    pivot = n - 1 - num_increment
    larger = smaller = pivot
    
    res = [None] * n
    res[0] = pivot
    for i in xrange(1, n):
        if jumbled_seq[i] == '+':
            larger += 1
            res[i] = larger
        else:
            smaller -= 1
            res[i] = smaller
    return res


class GenerateSequenceSpec(unittest.TestCase):
    def validate_result(self, jumbled_seq, result_seq):
        self.assertEqual(len(jumbled_seq), len(result_seq))
        self.assertEqual(sorted(result_seq), range(len(jumbled_seq)))
        for i in xrange(1, len(result_seq)):
            if jumbled_seq[i] == '+':
                self.assertLess(result_seq[i-1], result_seq[i])
            else:
                self.assertGreater(result_seq[i-1], result_seq[i])

    def test_example(self):
        # possible result: [1, 2, 3, 0, 4]
        input = [None, '+', '+', '-', '+']
        self.validate_result(input, generate_sequence(input))
    
    def test_exampty_array(self):
        self.validate_result([], generate_sequence([]))

    def test_array_with_one_elem(self):
        # possible result: [0]
        input = [None]
        self.validate_result(input, generate_sequence(input))

    def test_descending_array(self):
        # possible result: [3, 2, 1, 0]
        input = [None, '-', '-', '-']
        self.validate_result(input, generate_sequence(input))

    def test_ascending_array(self):
        # possible result: [0, 1, 2, 3, 4]
        input = [None, '+', '+', '+', '+']
        self.validate_result(input, generate_sequence(input))

    def test_zigzag_array(self):
        # possible result: [0, 5, 1, 4, 2, 3]
        input = [None, '+', '-', '+', '-', '+']
        self.validate_result(input, generate_sequence(input))

    def test_zigzag_array2(self):
        # possible result: [5, 0, 4, 1, 3, 2]
        input = [None,  '-', '+', '-', '+', '-']
        self.validate_result(input, generate_sequence(input))

    def test_decrease_then_increase(self):
        # possible result: [5, 4, 0, 1, 2, 3]
        input = [None,  '-', '-', '+', '+', '+']
        self.validate_result(input, generate_sequence(input))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 10, 2019 \[Medium\] Point in Polygon
--- 
> **Question:** You are given a list of N points (x1, y1), (x2, y2), ..., (xN, yN) representing a polygon. You can assume these points are given in order; that is, you can construct the polygon by connecting point 1 to point 2, point 2 to point 3, and so on, finally looping around to connect point N to point 1.
>
> Determine if a new point p lies inside this polygon. (If p is on the boundary of the polygon, you should return False).


**Hint:** Cast a ray from left to right and count intersections. Odd number means inside. Even means outside.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/RecursiveEvenPolygon.svg/2560px-RecursiveEvenPolygon.svg.png" width="220" height="99">


**Solution:** [https://repl.it/@trsong/Point-in-Polygon](https://repl.it/@trsong/Point-in-Polygon)
```py
import unittest
import math

def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def is_on_segment(line, p):
    # A point is on line segment iff d1 + d2 = d3 
    d1 = distance(p, line[0])
    d2 = distance(p, line[1])
    d3 = distance(line[0], line[1])
    return abs(d1 + d2 - d3) <= 10 ** -6


def cross(p1, p2, p3):
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    return x1 * y2 - x2 * y1     


def has_intersection(line1, line2):
    p1, p2 = line1
    p3, p4 = line2
    if max(p1[0], p2[0]) <= min(p3[0], p4[0]): return False  # line1 left line2
    if min(p1[0], p2[0]) >= max(p3[0], p4[0]): return False  # line1 right line2
    if max(p1[1], p2[1]) <= min(p3[1], p4[1]): return False  # line1 below line2
    if min(p1[1], p2[1]) >= max(p3[1], p4[1]): return False  # line1 above line2
    if cross(p1, p2, p3) * cross(p1, p2, p4) > 0: return False  # both line2 ends are on same side of line1
    if cross(p3, p4, p1) * cross(p3, p4, p2) > 0: return False  # both line1 ends are on same side of line2
    return True


def is_point_in_polygon(polygon, point):
    right_end = (float('inf'), point[1])
    ray = (point, right_end)
    num_intersection = 0
    start = polygon[-1]
    for end in polygon:
        segment = (start, end)
        if is_on_segment(segment, point):
            return False
        if has_intersection(segment, ray):
            num_intersection += 1
        start = end

    # Cast a ray from a point inside polygon should have odd number of intersection with boundary
    return num_intersection % 2 > 0


class IsPointInPolygonSpec(unittest.TestCase):
    def test_square(self):
        polygon = [(-1, 1), (-1, -1), (1, -1), (1, 1)]
        self.assertFalse(is_point_in_polygon(polygon, (0, 2))) # Above
        self.assertFalse(is_point_in_polygon(polygon, (0, -2))) # Below
        self.assertFalse(is_point_in_polygon(polygon, (-2, 0))) # Left
        self.assertFalse(is_point_in_polygon(polygon, (2, 0))) # Right
        self.assertTrue(is_point_in_polygon(polygon, (0, 0))) # Inside
        self.assertFalse(is_point_in_polygon(polygon, (-1, 0))) # Boundary1 
        self.assertFalse(is_point_in_polygon(polygon, (1, 1))) # Boundary2 
        self.assertFalse(is_point_in_polygon(polygon, (0, -1))) # Boundary3 

    def test_triangle(self):
        polygon = [(0, 1), (-1, 0), (1, 0)]
        self.assertFalse(is_point_in_polygon(polygon, (0, 2))) # Above
        self.assertFalse(is_point_in_polygon(polygon, (0, -1))) # Below
        self.assertFalse(is_point_in_polygon(polygon, (-0.5, 1))) # Left
        self.assertFalse(is_point_in_polygon(polygon, (2, 0))) # Right
        self.assertTrue(is_point_in_polygon(polygon, (0, 0.5))) # Inside
        self.assertFalse(is_point_in_polygon(polygon, (-0.5, 0.5))) # Boundary1 
        self.assertFalse(is_point_in_polygon(polygon, (-1, 0))) # Boundary2 
        self.assertFalse(is_point_in_polygon(polygon, (1, 0))) # Boundary3 

    def test_convex_polygon(self):
        polygon = [(0, 2), (-2, 0), (-1, -2), (1, -2), (2, 0)]
        self.assertFalse(is_point_in_polygon(polygon, (0, 3))) # Above
        self.assertFalse(is_point_in_polygon(polygon, (0, -3))) # Below
        self.assertFalse(is_point_in_polygon(polygon, (-0.5, 2))) # Left
        self.assertFalse(is_point_in_polygon(polygon, (2, 0.5))) # Right
        self.assertTrue(is_point_in_polygon(polygon, (0.5, 0.5))) # Inside
        self.assertFalse(is_point_in_polygon(polygon, (-2, 0))) # Boundary1 
        self.assertFalse(is_point_in_polygon(polygon, (2, 0))) # Boundary2 
        self.assertFalse(is_point_in_polygon(polygon, (0, -2))) # Boundary3 

    def test_concave_polygon(self):
        polygon = [(1, 3), (-3, 0), (0, -3), (3, 0), (-1, -1)]
        self.assertFalse(is_point_in_polygon(polygon, (0, 4))) # Above
        self.assertFalse(is_point_in_polygon(polygon, (0, -4))) # Below
        self.assertFalse(is_point_in_polygon(polygon, (-2, 2))) # Left
        self.assertFalse(is_point_in_polygon(polygon, (1, 0))) # Right
        self.assertTrue(is_point_in_polygon(polygon, (-0.5, 0.5))) # Inside1
        self.assertTrue(is_point_in_polygon(polygon, (-1, -0.5))) # Inside2
        self.assertTrue(is_point_in_polygon(polygon, (1, -1))) # Inside3
        self.assertFalse(is_point_in_polygon(polygon, (-2, -1))) # Boundary1 
        self.assertFalse(is_point_in_polygon(polygon, (-1, -1))) # Boundary2 
        self.assertFalse(is_point_in_polygon(polygon, (1, -2))) # Boundary3 


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 9, 2019 LC 448 \[Easy\] Find Missing Numbers in an Array
--- 
> **Question:** Given an array of integers of size n, where all elements are between 1 and n inclusive, find all of the elements of [1, n] that do not appear in the array. Some numbers may appear more than once.
> 
> Follow-up: Could you do it without extra space and in O(n) runtime?

**Example1:**
```py
Input: [4, 3, 2, 7, 8, 2, 3, 1]
Output: [5, 6]
```

**Example2:**
```py
Input: [4, 5, 2, 6, 8, 2, 1, 5]
Output: [3, 7]
```

**Solution:** [https://repl.it/@trsong/Find-Missing-Numbers-in-an-Array](https://repl.it/@trsong/Find-Missing-Numbers-in-an-Array)
```py
import unittest

def find_missing_numbers(nums):
    # Mark existing value as negative
    for num in nums:
        index = abs(num) - 1
        if nums[index] > 0:
            nums[index] *= -1
    
    # Find missing numbers as well as restore original values
    res = []
    for i in xrange(len(nums)):
        if nums[i] > 0:
            res.append(i+1)
        else:
            nums[i] *= -1
    return res
    

class FindMissingNumberSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(sorted(expected), sorted(result))

    def test_example1(self):
        input = [4, 3, 2, 7, 8, 2, 3, 1]
        expected = [5, 6]
        self.assert_result(expected, find_missing_numbers(input))

    def test_example2(self):
        input = [4, 5, 2, 6, 8, 2, 1, 5]
        expected = [3, 7]
        self.assert_result(expected, find_missing_numbers(input))

    def test_empty_array(self):
        self.assertEqual([], find_missing_numbers([]))

    def test_no_missing_numbers(self):
        input = [6, 1, 4, 3, 2, 5]
        expected = []
        self.assert_result(expected, find_missing_numbers(input))

    def test_duplicated_number1(self):
        input = [1, 1, 2]
        expected = [3]
        self.assert_result(expected, find_missing_numbers(input))

    def test_duplicated_number2(self):
        input = [1, 1, 3, 5, 6, 8, 8, 1, 1]
        expected = [2, 4, 7, 9]
        self.assert_result(expected, find_missing_numbers(input))
    
    def test_missing_first_number(self):
        input = [2, 2]
        expected = [1]
        self.assert_result(expected, find_missing_numbers(input))
    
    def test_missing_multiple_numbers1(self):
        input = [1, 3, 3]
        expected = [2]
        self.assert_result(expected, find_missing_numbers(input))
    
    def test_missing_multiple_numbers2(self):
        input = [3, 2, 3, 2, 3, 2, 7]
        expected = [1, 4, 5, 6]
        self.assert_result(expected, find_missing_numbers(input))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 8, 2019 LC 393 \[Medium\] UTF-8 Validator
--- 
> **Question:** Given a list of integers where each integer represents 1 byte, return whether or not the list of integers is a valid UTF-8 encoding.
> 
> A UTF-8 character encoding is a variable width character encoding that can vary from 1 to 4 bytes depending on the character. The structure of the encoding is as follows:

```py
1 byte:  0xxxxxxx
2 bytes: 110xxxxx 10xxxxxx
3 bytes: 1110xxxx 10xxxxxx 10xxxxxx
4 bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
```

**My thoughts:** The trick is to break the array into multiple groups and tackle them one by one. For each group, count the leading 1's of initial byte and use that to validate the remaining bytes. A helper function to count leading 1's will be super useful. Make sure to watch out for the edge case test that consumes more than 4 bytes. Remember that a valid utf-8 string has at most 4 bytes. 

**Solution:** [https://repl.it/@trsong/UTF-8-Validator](https://repl.it/@trsong/UTF-8-Validator)
```py
import unittest

def count_leading_ones(byte):
    num_ones = 0
    for shift_amout in xrange(7, -1, -1):
        if byte & (1 << shift_amout) == 0:
            break
        num_ones += 1
    return num_ones


def utf8_validate(data):
    if not data:
        return True
    
    n = len(data)
    i = 0
    while i < n:
        initial_byte = data[i]
        i += 1
        num_bytes = count_leading_ones(initial_byte)
        if num_bytes == 1 or num_bytes > 4:
            return False
        for remaining_num_bytes in xrange(num_bytes - 1, 0, -1):
            if i >= n or count_leading_ones(data[i]) != 1:
                return False
            i += 1
    return True


class UTF8ValidateSpec(unittest.TestCase):
    def test_example1(self):
        data = [0b11000101, 0b10000010, 0b00000001]
        self.assertTrue(utf8_validate(data))

    def test_example2(self):
        data = [0b11101011, 0b10001100, 0b00000100]
        self.assertFalse(utf8_validate(data))

    def test_empty_data(self):
        self.assertTrue(utf8_validate([]))
    
    def test_sequence_of_one_byte_string(self):
        data = [0b00000000, 0b01000000, 0b00100000, 0b00010000, 0b01111111]
        self.assertTrue(utf8_validate(data))

    def test_should_be_no_more_than_4_byte(self):
        data = [0b11111010, 0b10010001, 0b10010001, 0b10010001, 0b10010001]
        self.assertFalse(utf8_validate(data))

    def test_can_not_start_with_10(self):
        data = [0b10010001]
        self.assertFalse(utf8_validate(data))

    def test_various_length_strings(self):
        one_byte = [0b00010000]
        two_byte = [0b11010000, 0b10010000]
        three_byte = [0b11101000, 0b10010000, 0b10010000]
        four_byte = [0b11110100, 0b10010000, 0b10010000, 0b10010000]
        data = one_byte + two_byte + one_byte + three_byte + two_byte + four_byte + one_byte
        self.assertTrue(utf8_validate(data))

    def test_mix_various_length(self):
        data = [0b11110100, 0b10010000, 0b00010000, 0b10010000, 0b10010000]
        self.assertFalse(utf8_validate(data))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 7, 2019 \[Easy\] Zig-Zag Distinct LinkedList
--- 
> **Question:** Given a linked list with DISTINCT value, rearrange the node values such that they appear in alternating `low -> high -> low -> high ...` form. For example, given `1 -> 2 -> 3 -> 4 -> 5`, you should return `1 -> 3 -> 2 -> 5 -> 4`.


**Solution:** [https://repl.it/@trsong/Zig-Zag-LinkedList](https://repl.it/@trsong/Zig-Zag-LinkedList)
```py
import unittest

def zig_zag_order(lst):
    if not lst:
        return
        
    p = lst
    isLessThan = True
    while p.next:
        if isLessThan and p.val > p.next.val or not isLessThan and p.val < p.next.val:
            p.val, p.next.val = p.next.val, p.val
        p = p.next
        isLessThan = not isLessThan


###################
# Testing Utilities
###################
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
    
    @staticmethod  
    def List(*vals):
        dummy = ListNode(-1)
        p = dummy
        for elem in vals:
            p.next = ListNode(elem)
            p = p.next
        return dummy.next  


class ZigZagOrderSpec(unittest.TestCase):
    def verify_order(self, lst):
        isLessThanPrevious = False
        if not lst:
            return
        p = lst.next
        prev = lst
        while p:
            if isLessThanPrevious:
                self.assertLess(p.val, prev.val)
            else:
                self.assertGreater(p.val, prev.val)

            isLessThanPrevious = not isLessThanPrevious
            prev = p
            p = p.next

    def test_example(self):
        lst = ListNode.List(1, 2, 3, 4, 5)
        zig_zag_order(lst)
        self.verify_order(lst)

    def test_empty_array(self):
        lst = ListNode.List()
        zig_zag_order(lst)
        self.verify_order(lst)

    def test_unsorted_list1(self):
        lst = ListNode.List(10, 5, 6, 3, 2, 20, 100, 80)
        zig_zag_order(lst)
        self.verify_order(lst)

    def test_unsorted_list2(self):
        lst = ListNode.List(2, 4, 6, 8, 10, 20)
        zig_zag_order(lst)
        self.verify_order(lst)

    def test_unsorted_list3(self):
        lst = ListNode.List(3, 6, 5, 10, 7, 20)
        zig_zag_order(lst)
        self.verify_order(lst)

    def test_unsorted_list4(self):
        lst = ListNode.List(20, 10, 8, 6, 4, 2)
        zig_zag_order(lst)
        self.verify_order(lst)

    def test_unsorted_list5(self):
        lst = ListNode.List(6, 4, 2, 1, 8, 3)
        zig_zag_order(lst)
        self.verify_order(lst)

    def test_sorted_list(self):
        lst = ListNode.List(6, 5, 4, 3, 2, 1)
        zig_zag_order(lst)
        self.verify_order(lst)
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 6, 2019 \[Easy\] Convert Roman Numerals to Decimal
--- 
> **Question:** Given a Roman numeral, find the corresponding decimal value. Inputs will be between 1 and 3999.
> 
> **Note:** Numbers are strings of these symbols in descending order. In some cases, subtractive notation is used to avoid repeated characters. The rules are as follows:
> 1. I placed before V or X is one less, so 4 = IV (one less than 5), and 9 is IX (one less than 10)
> 2. X placed before L or C indicates ten less, so 40 is XL (10 less than 50) and 90 is XC (10 less than 100).
> 3. C placed before D or M indicates 100 less, so 400 is CD (100 less than 500), and 900 is CM (100 less than 1000).

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

**Solution:** [https://repl.it/@trsong/Convert-Roman-Numerals-to-Decimal](https://repl.it/@trsong/Convert-Roman-Numerals-to-Decimal)
```py
import unittest

roman_unit = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500,'M': 1000}

def roman_to_decimal(roman_str):
    prev = None
    decimal = 0
    for letter in roman_str:
        if prev and roman_unit[prev] < roman_unit[letter]:
            decimal -= 2 * roman_unit[prev]
        decimal += roman_unit[letter]
        prev = letter
    return decimal


class RomanToDecimalSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(9, roman_to_decimal("IX"))

    def test_example2(self):
        self.assertEqual(7, roman_to_decimal("VII"))

    def test_example3(self):
        self.assertEqual(1904, roman_to_decimal("MCMIV"))

    def test_boundary(self):
        self.assertEqual(3999, roman_to_decimal("MMMCMXCIX"))

    def test_descending_order_rule1(self):
        self.assertEqual(34, roman_to_decimal("XXXIV"))

    def test_descending_order_rule2(self):
        self.assertEqual(640, roman_to_decimal("DCXL"))

    def test_descending_order_rule3(self):
        self.assertEqual(912, roman_to_decimal("CMXII"))

    def test_all_decending_rules_applied(self):
        self.assertEqual(3949, roman_to_decimal("MMMCMXLIX"))

    def test_all_decending_rules_applied2(self):
        self.assertEqual(2994, roman_to_decimal("MMCMXCIV"))

    def test_all_in_normal_order(self):
        self.assertEqual(1666, roman_to_decimal("MDCLXVI"))

    def test_all_in_normal_order2(self):
        self.assertEqual(3888, roman_to_decimal("MMMDCCCLXXXVIII"))
        

if __name__ == '__main__':
    unittest.main(exit=False)
```
### Dec 5, 2019 LC 222 \[Medium\] Count Complete Tree Nodes
--- 
> **Question:** Given a complete binary tree, count the number of nodes in faster than O(n) time. Recall that a complete binary tree has every level filled except the last, and the nodes in the last level are filled starting from the left.

**My thoughts:** For any complete binary tree, the max height of left tree vs the max height of right tree differ at most by one. We can take advantage of this property to quickly identify either left or right is full binary tree.

The trick is to check the left max height of left tree vs the left max height of right tree. If they are equal, that means left tree is full binaray tree. ie 2^height - 1 number of nodes. Otherwise, we can say the right tree must be full.

**Solution:** [https://repl.it/@trsong/Count-Complete-Tree-Nodes](https://repl.it/@trsong/Count-Complete-Tree-Nodes)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def left_height(tree):
    height = 0
    while tree:
        height += 1
        tree = tree.left
    return height


def count_complete_tree(root):
    if not root:
        return 0

    left_height1 = left_height(root.left)
    left_height2 = left_height(root.right)

    if left_height1 == left_height2:
        # left child guarantee to be full
        return 2 ** left_height1 + count_complete_tree(root.right)
    else:
        # right child guarantee to be full
        return 2 ** left_height2 + count_complete_tree(root.left)


class CountCompleteTreeSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertEqual(0, count_complete_tree(None))

    def test_full_tree_with_depth_1(self):
        """
          1
         / \
        2   3
        """
        root = TreeNode(1, TreeNode(2), TreeNode(3))
        self.assertEqual(3, count_complete_tree(root))

    def test_depth_2_complete_tree(self):
        """
             1
           /   \
          2     3
         / \   /
        4   5 6
        """
        left_tree = TreeNode(2, TreeNode(4), TreeNode(5))
        right_tree = TreeNode(3, TreeNode(6))
        root = TreeNode(1, left_tree, right_tree)
        self.assertEqual(6, count_complete_tree(root))

    def test_last_level_with_one_element(self):
        """
             1
           /   \
          2     3
         /
        4
        """
        root = TreeNode(1, TreeNode(2, TreeNode(4)), TreeNode(3))
        self.assertEqual(4, count_complete_tree(root))

    def test_last_level_missing_2_elements(self):
        """
            __ 1 __
           /       \
          2         3
         / \       / \
        4    5    6   7
       / \  / \  / \
      8  9 10 11 12 13
        """
        n4 = TreeNode(4, TreeNode(8), TreeNode(9))
        n5 = TreeNode(5, TreeNode(10), TreeNode(11))
        n2 = TreeNode(2, n4, n5)
        n6 = TreeNode(6, TreeNode(12), TreeNode(13))
        n3 = TreeNode(3, n6, TreeNode(7))
        root = TreeNode(1, n2, n3)
        self.assertEqual(13, count_complete_tree(root))
        

if __name__ == '__main__':
    unittest.main(exit=False)
```
### Dec 4, 2019 \[Medium\] Zig-Zag String
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

**Solution:** [https://repl.it/@trsong/NarrowUnfitInterfacestandard](https://repl.it/@trsong/NarrowUnfitInterfacestandard)
```py
import unittest

def zig_zag_format(sentence, k):
    if not sentence:
        return []
    elif k == 1:
        return [sentence]

    result = [[] for _ in xrange(min(k, len(sentence)))]
    row = 0
    direction = -1
    for i, char in enumerate(sentence):
        if i < k:
            padding = i
        elif direction == 1:
            padding = 2 * row - 1
        else:
            padding = 2 * (k - 1 - row) - 1
        result[row].append(" " * padding + char)

        if row == 0 or row == k - 1:
            direction *= -1
        row += direction
    return map(lambda line: "".join(line), result)


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
    unittest.main(exit=False)
```
### Dec 3, 2019 \[Medium\] Multitasking
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

**Solution:**  [https://repl.it/@trsong/Multitasking](https://repl.it/@trsong/Multitasking)
```py
import unittest

def multitasking_time(task_seq, cooldown):
    n = len(task_seq)
    if cooldown <= 0:
        return n

    last_occur_log = {}
    total_time = 0
    for task in task_seq:
        current_time = total_time + 1
        last_occur = last_occur_log.get(task, -1)
        delta = current_time - last_occur
        if last_occur < 0 or delta > cooldown:
            idle_time = 0 
        else:
            idle_time = cooldown - delta + 1
        total_time += 1 + idle_time
        last_occur_log[task] = total_time
    return total_time


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
    unittest.main(exit=False)
```

### Dec 2, 2019 \[Medium\] Sort a Partially Sorted List
--- 
> **Question:** You are given a list of n numbers, where every number is at most k indexes away from its properly sorted index. Write a sorting algorithm (that will be given the number k) for this list that can solve this in O(n log k)

**Example:**
```py
Input: [3, 2, 6, 5, 4], k=2
Output: [2, 3, 4, 5, 6]
As seen above, every number is at most 2 indexes away from its proper sorted index.
``` 

**Solution with PriorityQueue:** [https://repl.it/@trsong/Sort-a-Partially-Sorted-List](https://repl.it/@trsong/Sort-a-Partially-Sorted-List)
```py
import unittest
from Queue import PriorityQueue


def sort_partial_list(nums, k):
    n = len(nums)
    pq = PriorityQueue()
    res = []
    for i in xrange(min(n, k+1)):
        pq.put(nums[i])

    for i in xrange(k+1, n):
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
    unittest.main(exit=False)
```
### Dec 1, 2019 LC 508 \[Medium\] Most Frequent Subtree Sum
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

**Solution:** [https://repl.it/@trsong/Most-Frequent-SubTree-Sum](https://repl.it/@trsong/Most-Frequent-SubTree-Sum)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def most_freq_tree_sum(tree):
    freq_map = {}
    node_sum(tree, freq_map)

    max_sum = None
    max_sum_freq = float("-inf")
    for sum, freq in freq_map.items():
        if freq > max_sum_freq or freq == max_sum_freq and sum < max_sum:
            max_sum_freq = freq
            max_sum = sum
    return max_sum 

def node_sum(node, freq_map):
    if not node:
        return 0
    left_sum = node_sum(node.left, freq_map)
    right_sum = node_sum(node.right, freq_map)
    current_sum = node.val + left_sum + right_sum
    freq_map[current_sum] = freq_map.get(current_sum, 0) + 1
    return current_sum


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

### Nov 30, 2019 \[Medium\] Longest Increasing Subsequence
--- 
> **Question:** You are given an array of integers. Return the length of the longest increasing subsequence (not necessarily contiguous) in the array.

**Example:**
```py
Input: [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
Output: 6
Explanation: since the longest increasing subsequence is 0, 2, 6, 9 , 11, 15.
```

**My thoughts:** Try different examples until find a pattern to break problem into smaller subproblems. I tried `[1, 2, 3, 0, 2]`, `[1, 2, 3, 0, 6]`, `[4, 5, 6, 7, 1, 2, 3]` and notice the pattern that the longest increasing subsequence ends at index i equals 1 + max of all previous longest increasing subsequence ends before i if that elem is smaller than sequence[i].

Example, suppose f represents the longest increasing subsequence ends at last element 

```py
f([1, 2, 3, 0, 2]) = 1 + max(f([1]), f([1, 2, 3, 0])) # as 2 > 1 and 2 > 0, gives [1, 2], [0, 2] and max len is 2
f([1, 2, 3, 0, 6]) = 1 + max(f[1], f([1, 2]), f([1, 2, 3]), f([1, 2, 3, 0])) # as 6 > all. gives [1, 6], [1, 2, 6] or [1, 2, 3, 6] and max len is 4
```

And finally once we get an array of all the longest increasing subsequence ends at i. We can take the maximum to find the global longest increasing subsequence among all i.

**Solution with DP:** [https://repl.it/@trsong/Longest-Increasing-Subsequence](https://repl.it/@trsong/Longest-Increasing-Subsequence)
```py
import unittest

def longest_increasing_subsequence(sequence):
    if not sequence:
        return 0
    
    # Let dp[i] represents the length of longest sequence ends at index i
    #     dp[i] = max{dp[j]} + 1 for all j < i where sequnece[j] < sequence[i]
    n = len(sequence)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if sequence[j] < sequence[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    res = max(dp)
    return res if res > 1 else 0


class LongestIncreasingSubsequnceSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(6, longest_increasing_subsequence([0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]))  # 0, 2, 6, 9 , 11, 15

    def test_empty_sequence(self):
        self.assertEqual(0, longest_increasing_subsequence([]))

    def test_last_elem_is_local_max(self):
        self.assertEqual(3, longest_increasing_subsequence([1, 2, 3, 0, 2]))  # 1, 2, 3

    def test_last_elem_is_global_max(self):
        self.assertEqual(4, longest_increasing_subsequence([1, 2, 3, 0, 6]))  # 1, 2, 3, 6

    def test_longest_increasing_subsequence_in_first_half_sequence(self):
        self.assertEqual(4, longest_increasing_subsequence([4, 5, 6, 7, 1, 2, 3]))  # 4, 5, 6, 7

    def test_longest_increasing_subsequence_in_second_half_sequence(self):
        self.assertEqual(4, longest_increasing_subsequence([1, 2, 3, -2, -1, 0, 1]))  # -2, -1, 0, 1

    def test_sequence_in_up_down_up_pattern(self):
        self.assertEqual(4, longest_increasing_subsequence([1, 2, 3, 2, 4]))  # 1, 2, 3, 4

    def test_sequence_in_up_down_up_pattern2(self):
        self.assertEqual(3, longest_increasing_subsequence([1, 2, 3, -1, 0]))  # 1, 2, 3

    def test_sequence_in_down_up_down_pattern(self):
        self.assertEqual(2, longest_increasing_subsequence([4, 3, 5])) # 4, 5
    
    def test_sequence_in_down_up_down_pattern2(self):
        self.assertEqual(2, longest_increasing_subsequence([4, 0, 1]))  # 0, 1

    def test_array_with_unique_value(self):
        self.assertEqual(0, longest_increasing_subsequence([1, 1, 1, 1, 1]))

    def test_decreasing_array(self):
        self.assertEqual(0, longest_increasing_subsequence([3, 2, 1, 0]))


if __name__ == '__main__':
    unittest.main(exit=False)
```
### Nov 29, 2019 \[Easy\] Spreadsheet Columns
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

**Solution:** [https://repl.it/@trsong/Spreadsheet-Columns](https://repl.it/@trsong/Spreadsheet-Columns)
```py
import unittest

def digit_to_letter(num):
    ord_A = ord('A')
    return chr(ord_A + num)


def spreadsheet_columns(n):
    base = 26
    res = []
    while n > 0:
        remainder = n % base
        res.append(digit_to_letter((remainder - 1) % base))
        n //= base
        if remainder == 0:
            n -= 1
    return ''.join(reversed(res))


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


### Nov 28, 2019 \[Hard\] Maximize Sum of the Minimum of K Subarrays
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

**Solution with DP:** [https://repl.it/@trsong/Maximize-Sum-of-the-Minimum-of-K-Subarrays](https://repl.it/@trsong/Maximize-Sum-of-the-Minimum-of-K-Subarrays)
```py
import unittest
import sys

def max_aggregate_subarray_min(nums, k):
    n = len(nums)
    dp = [[-sys.maxint for _ in xrange(k+1)] for _ in xrange(n+1)]

    # Let dp[n][k] represents the max aggregate subarray sum with problem size n and number of splits k
    #     dp[n][k] = max{dp[n-p][k-1] + min_value of last_subarray} for all p < n, ie. num[p] is in last subarray
    dp[0][0] = 0
    for num_subarray in xrange(1, k+1):
        for problem_size in xrange(num_subarray, n+1):
            # last_subarray_min is the smallest element in last subarray
            last_subarray_min = nums[problem_size-1]
            for s in xrange(problem_size, 0, -1):
                # With the introduction of the current element to last subarray, the min value of last subarray might change 
                last_subarray_min = min(last_subarray_min, nums[s-1])
                dp[problem_size][num_subarray] = max(dp[problem_size][num_subarray], dp[s-1][num_subarray-1] + last_subarray_min)

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

### Nov 27, 2019 \[Easy\] Palindrome Integers
--- 
> **Question:** Write a program that checks whether an integer is a palindrome. For example, `121` is a palindrome, as well as `888`. But neither `678` nor `80` is a palindrome. Do not convert the integer into a string.

**Solution:** [https://repl.it/@trsong/Palindrome-Integers](https://repl.it/@trsong/Palindrome-Integers)
```py
import unittest
import math

def is_palindrome_integer(num):
    if num < 0:
        return False
    if num == 0:
        return True
    n = int(math.log10(num)) + 1
    i = 0
    j = n - 1
    while i < j:
        first_digit = num / 10 ** j
        last_digit = num % 10
        if first_digit != last_digit:
            return False
        num /= 10
        i += 1
        j -= 1
    return True


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
    unittest.main(exit=False)
```

### Nov 26, 2019 LC 189 \[Easy\] Rotate Array to Right K Elements In-place
--- 
> **Question:** Given an array and a number k that's smaller than the length of the array, rotate the array to the right k elements in-place.

**Solution:** [https://repl.it/@trsong/Rotate-Array-to-Right-K-Elements-In-place](https://repl.it/@trsong/Rotate-Array-to-Right-K-Elements-In-place)
```py
import unittest

def reverse_array_in_range(array, start, end):
    while start < end:
        array[start], array[end] = array[end], array[start]
        start += 1
        end -= 1


def rotate_array(nums, k):
    if not nums:
        return nums

    n = len(nums)
    k = k % n
    reverse_array_in_range(nums, 0, n-1)
    reverse_array_in_range(nums, 0, n-k-1)
    reverse_array_in_range(nums, n-k, n-1)
    return nums


class RotateArraySpec(unittest.TestCase):
    def test_simple_array(self):
        self.assertEqual([3, 4, 5, 6, 1, 2], rotate_array([1, 2, 3, 4, 5, 6], k=2))

    def test_rotate_0_position(self):
        self.assertEqual([0, 1, 2, 3], rotate_array([0, 1, 2, 3], k=0))

    def test_empty_array(self):
        self.assertEqual([], rotate_array([], k=10))

    def test_shift_negative_position(self):
        self.assertEqual([3, 0, 1, 2], rotate_array([0, 1, 2, 3], k=-1))

    def test_shift_more_than_array_size(self):
        self.assertEqual([3, 4, 5, 6, 1, 2], rotate_array([1, 2, 3, 4, 5, 6], k=8))

    def test_multiple_round_of_forward_and_backward_shift(self):
        nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        k = 5
        expected = [5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4]
        self.assertEqual(rotate_array(nums, k), expected)
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 25, 2019 \[Medium\] Maximum Amount of Money from a Game
--- 
> **Question:** In front of you is a row of N coins, with values v1, v1, ..., vn.
>
> You are asked to play the following game. You and an opponent take turns choosing either the first or last coin from the row, removing it from the row, and receiving the value of the coin.
>
> Write a program that returns the maximum amount of money you can win with certainty, if you move first, assuming your opponent plays optimally.

**My thoughts:** In a zero-sum game, one's lose is the other's gain. Therefore, the move you make is the best move among the worest situations. 

There are 2 scenarios at each step, assume the current coin array ranges from i's position to j's position of original array:

- Scenario 1: if I choose the i-th coin, opponent must choose either i+1 or j
- Scenario 2: if I choose the j-th coin, opponent must choose either i or j-1

In any of the two scenarios, assume the opponent always makes best move for him which leaves the worst situation for me. 

Note that the greedy solution might not work. Because the best move in the short-run (local maximum) might not be the best move in the long-run (global maximum).

Example of greedy approach not produces optimal solution:
```
Coins: [8, 15, 3, 7]

Greedy approach:
Round1 - If I choose 8, then Opponent will choose 15
Round2 - If I choose 7, then Opponent will choose 3
Leaves 15 for me and 18 for opponent in total

Optimal approach:
Round1 - If I choose 7, then Opponent will choose 8
Round2 - If I choose 15, then Opponent will choose 3
Leaves 22 for me and 11 for opponent in total

In round1, if I choose 8 the greedy approach I can only get 15 assuming the opponent is always smart. But the opitimal is to choose 7.
```

**MiniMax Solution:** [https://repl.it/@trsong/Maximum-Amount-of-Money-from-a-Game](https://repl.it/@trsong/Maximum-Amount-of-Money-from-a-Game)
```py
import unittest

def best_strategy(coins):
    n = len(coins)
    cache = [[None for _ in xrange(n)] for _ in xrange(n)]
    return max_money_within_range(coins, 0, n - 1, cache)


def max_money_within_range(coins, i, j, cache):
    if i == j:
        return coins[i]
    elif i + 1 == j:
        return coins[i] if coins[1] > coins[i+1] else coins[i+1]
    elif i > j:
        return 0
    elif cache[i][j] is not None:
        return cache[i][j]
    
    # Scenario 1: User choose first coin within range i to j
    # Opponent choose either i + 1 or j whichever works best for him
    res1 = coins[i] + min(max_money_within_range(coins, i+2, j, cache), max_money_within_range(coins, i+1, j-1, cache))

    # Scenario 2: User choose last coin within range i to j
    # Oppopnet choose either i or j-1 whichever works best for him
    res2 = coins[j] + min(max_money_within_range(coins, i+1, j-1, cache), max_money_within_range(coins, i, j-2, cache))

    cache[i][j] = max(res1, res2)
    return cache[i][j]


class BestStrategySpec(unittest.TestCase):
    def test_trival_game(self):
        self.assertEqual(0, best_strategy([]))
        self.assertEqual(1, best_strategy([1]))
        self.assertEqual(2, best_strategy([1, 2]))
        self.assertEqual(1 + 3, best_strategy([1, 2, 3]))

    def test_simple_game(self):
        self.assertEqual(3 + 9, best_strategy([1, 9, 1, 3]))

    def test_greedy_is_not_optimal(self):
        self.assertEqual(7 + 15, best_strategy([8, 15, 3, 7]))

    def test_same_value_of_coins(self):
        self.assertEqual(2 + 2, best_strategy([2, 2, 2, 2]))

    def test_greedy_is_not_optimal2(self):
        self.assertEqual(10 + 30 + 2, best_strategy([20, 30, 2, 2, 2, 10]))

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 24, 2019 \[Easy\] Markov Chain
--- 
> **Question:**You are given a starting state start, a list of transition probabilities for a Markov chain, and a number of steps num_steps. Run the Markov chain starting from start for num_steps and compute the number of times we visited each state.
>
> For example, given the starting state a, number of steps 5000, and the following transition probabilities:

```py
[
  ('a', 'a', 0.9),
  ('a', 'b', 0.075),
  ('a', 'c', 0.025),
  ('b', 'a', 0.15),
  ('b', 'b', 0.8),
  ('b', 'c', 0.05),
  ('c', 'a', 0.25),
  ('c', 'b', 0.25),
  ('c', 'c', 0.5)
]
One instance of running this Markov chain might produce { 'a': 3012, 'b': 1656, 'c': 332 }.
```

**My thoughts:** In order to uniformly choose among the probablity of next state, we can align the probably in a row and calculate accumulated probability. eg. `[a: .1, b: .2, c: .7]` => `[a: 0 ~ 0.1, b: 0.1 ~ 0.3, c: 0.3 ~ 1.0]` Thus, by generate a random number we can tell what's the next state based on probablity.

**Solution:** [https://repl.it/@trsong/Markov-Chain](https://repl.it/@trsong/Markov-Chain)
```py
import random
import unittest

def binary_search(dst_probs, target_prob):
    lo = 0
    hi = len(dst_probs) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if dst_probs[mid] < target_prob:
            lo = mid + 1
        else:
            hi = mid
    return lo


def markov_chain(start_state, num_steps, transition_probabilities):
    transition_map = {}
    for src, dst, prob in transition_probabilities:
        if src not in transition_map:
            transition_map[src] = [[], []]
        transition_map[src][0].append(dst)
        transition_map[src][1].append(prob)
    
    for src in transition_map:
        dst_probs = transition_map[src][1]
        for i in xrange(1, len(dst_probs)):
            # accumulate probability
            dst_probs[i] += dst_probs[i-1]
    
    state_counter = {}
    current_state = start_state
    for _ in xrange(num_steps):
        state_counter[current_state] = state_counter.get(current_state, 0) + 1
        dst_list, des_prob = transition_map[current_state]
        random_number = random.random()
        next_state_index = binary_search(des_prob, random_number)
        next_state = dst_list[next_state_index]
        current_state = next_state

    return state_counter


class MarkovChainSpec(unittest.TestCase):
    def test_example(self):
        start_state = 'a'
        num_steps = 5000
        transition_probabilities = [
            ('a', 'a', 0.9),
            ('a', 'b', 0.075),
            ('a', 'c', 0.025),
            ('b', 'a', 0.15),
            ('b', 'b', 0.8),
            ('b', 'c', 0.05),
            ('c', 'a', 0.25),
            ('c', 'b', 0.25),
            ('c', 'c', 0.5)
        ]
        random.seed(42)
        expected = {'a': 3205, 'c': 330, 'b': 1465}
        self.assertEqual(expected, markov_chain(start_state, num_steps, transition_probabilities))
    
    def test_transition_matrix2(self):
        start_state = '4'
        num_steps = 100
        transition_probabilities = [
            ('1', '2', 0.23),
            ('2', '1', 0.09),
            ('1', '4', 0.77),
            ('2', '3', 0.06),
            ('2', '6', 0.85),
            ('6', '2', 0.62),
            ('3', '4', 0.63),
            ('3', '6', 0.37),
            ('6', '5', 0.38),
            ('5', '6', 1),
            ('4', '5', 0.65),
            ('4', '6', 0.35),

        ]
        random.seed(42)
        expected = {'1': 3, '3': 2, '2': 27, '5': 21, '4': 4, '6': 43}
        self.assertEqual(expected, markov_chain(start_state, num_steps, transition_probabilities))

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 23, 2019 \[Hard\] Number of Subscribers during a Given Interval
--- 
> **Question:** You are given an array of length 24, where each element represents the number of new subscribers during the corresponding hour. Implement a data structure that efficiently supports the following:
>
- `update(hour: int, value: int)`: Increment the element at index hour by value.
- `query(start: int, end: int)`: Retrieve the number of subscribers that have signed up between start and end (inclusive).
>
> You can assume that all values get cleared at the end of the day, and that you will not be asked for start and end values that wrap around midnight.

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

Here all we need to do is to refactor the interface to adapt to SubscriberTracker.

**Solution with BIT:** [https://repl.it/@trsong/Number-of-Subscribers-during-a-Given-Interval](https://repl.it/@trsong/Number-of-Subscribers-during-a-Given-Interval)
```py
import unittest

class SubscriberTracker(object):
    def __init__(self, n=24):
        self.bit_tree = [0] * (n+1)

    @staticmethod
    def least_significant_bit(num):
        return num & -num

    def update(self, hour, value):
        bit_index = hour + 1
        while bit_index < len(self.bit_tree):
            self.bit_tree[bit_index] += value
            bit_index += SubscriberTracker.least_significant_bit(bit_index)

    def query_from_begining(self, hour):
        bit_index = hour + 1
        res = 0
        while bit_index > 0:
            res += self.bit_tree[bit_index]
            bit_index -= SubscriberTracker.least_significant_bit(bit_index)
        return res

    def query(self, start, end):
        return self.query_from_begining(end) - self.query_from_begining(start-1)


class SubscriberTrackerSpec(unittest.TestCase):
    def test_query_without_update(self):
        st = SubscriberTracker()
        self.assertEqual(0, st.query(0, 23))
        self.assertEqual(0, st.query(3, 5))
    
    def test_update_should_affect_query_value(self):
        st = SubscriberTracker()
        st.update(5, 10)
        st.update(10, 15)
        st.update(12, 20)
        self.assertEqual(0, st.query(0, 4))
        self.assertEqual(10, st.query(0, 5))
        self.assertEqual(10, st.query(0, 9))
        self.assertEqual(25, st.query(0, 10))
        self.assertEqual(45, st.query(0, 13))
        st.update(3, 2)
        self.assertEqual(2, st.query(0, 4))
        self.assertEqual(12, st.query(0, 5))
        self.assertEqual(12, st.query(0, 9))
        self.assertEqual(27, st.query(0, 10))
        self.assertEqual(47, st.query(0, 13))

    def test_number_of_subscribers_can_decrease(self):
        st = SubscriberTracker()
        st.update(10, 5)
        st.update(20, 10)
        self.assertEqual(10, st.query(15, 23))
        st.update(12, -3)
        self.assertEqual(10, st.query(15, 23))
        st.update(17, -7)
        self.assertEqual(3, st.query(15, 23))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 22, 2019 \[Easy\] Compare Version Numbers
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

**Solution with Two Pointers:** [https://repl.it/@trsong/Compare-Version-Numbers](https://repl.it/@trsong/Compare-Version-Numbers)
```py
import unittest

def version_number_compare(v1, v2):
    i = j = 0
    n, m = len(v1), len(v2)

    while i < n or j < m:
        sub_v1 = 0
        sub_v2 = 0
        while i < n and v1[i].isdigit():
            sub_v1 = 10 * sub_v1 + int(v1[i])
            i += 1

        while j < m and v2[j].isdigit():
            sub_v2 = 10 * sub_v2 + int(v2[j])
            j += 1

        i += 1
        j += 1
        if sub_v1 < sub_v2:
            return -1
        elif sub_v1 > sub_v2:
            return 1
    return 0


class VersionNumberCompareSpec(unittest.TestCase):
    def test_example1(self):
        version1 = "1.0.33"
        version2 = "1.0.27"
        self.assertEqual(1, version_number_compare(version1, version2))

    def test_example2(self):
        version1 = "0.1"
        version2 = "1.1"
        self.assertEqual(-1, version_number_compare(version1, version2))

    def test_example3(self):
        version1 = "1.01"
        version2 = "1.001"
        self.assertEqual(0, version_number_compare(version1, version2))

    def test_example4(self):
        version1 = "1.0"
        version2 = "1.0.0"
        self.assertEqual(0, version_number_compare(version1, version2))

    def test_unspecified_version_numbers(self):
        self.assertEqual(0, version_number_compare("", ""))
        self.assertEqual(-1, version_number_compare("", "1"))
        self.assertEqual(1, version_number_compare("2", ""))

    def test_unaligned_zeros(self):
        version1 = "00000.00000.00000.0"
        version2 = "0.00000.000.00.00000.000.000.0"
        self.assertEqual(0, version_number_compare(version1, version2))

    def test_same_version_yet_unaligned(self):
        version1 = "00001.001"
        version2 = "1.000001.0000000.0000"
        self.assertEqual(0, version_number_compare(version1, version2))

    def test_different_version_numbers(self):
        version1 = "1.2.3.4"
        version2 = "1.2.3.4.5"
        self.assertEqual(-1, version_number_compare(version1, version2))

    def test_different_version_numbers2(self):
        version1 = "3.2.1"
        version2 = "3.1.2.3"
        self.assertEqual(1, version_number_compare(version1, version2))

    def test_different_version_numbers3(self):
        version1 = "00001.001.0.1"
        version2 = "1.000001.0000000.0000"
        self.assertEqual(1, version_number_compare(version1, version2))

    def test_without_dots(self):
        version1 = "32123"
        version2 = "3144444"
        self.assertEqual(-1, version_number_compare(version1, version2))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 21, 2019 \[Easy\] GCD of N Numbers
--- 
> **Question:** Given `n` numbers, find the greatest common denominator between them.
>
> For example, given the numbers `[42, 56, 14]`, return `14`.


**Solution:** [https://repl.it/@trsong/GCD-of-N-Numbers](https://repl.it/@trsong/GCD-of-N-Numbers)
```py
import unittest

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a 


def gcd_n_numbers(nums):
    res = nums[0]
    for num in nums:
        # gcd(a, b, c) = gcd(a, gcd(b, c))
        res = gcd(res, num)
    return res


class GCDNNumberSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(14, gcd_n_numbers([42, 56, 14]))

    def test_co_prime_numbers(self):
        self.assertEqual(1, gcd_n_numbers([6, 5, 7, 11]))

    def test_composite_numbers(self):
        self.assertEqual(11, gcd_n_numbers([11 * 3, 11 * 5, 11 * 7]))

    def test_even_numbers(self):
        self.assertEqual(2, gcd_n_numbers([2, 8, 6, 4]))
    
    def test_odd_numbers(self):
        self.assertEqual(5, gcd_n_numbers([3 * 5, 3 * 2 * 5, 5 * 2 * 2]))


if __name__ == '__main__':
    unittest.main(exit=False)

```

### Nov 20, 2019 LC 138 \[Medium\] Copy List with Random Pointer
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

**Solution:** [https://repl.it/@trsong/Copy-List-with-Random-Pointer](https://repl.it/@trsong/Copy-List-with-Random-Pointer)
```py
import unittest


def deep_copy(head):
    if not head:
        return None

    # Step1: Duplicate each node and insert them in every other place: n1 -> copy_n1 -> n2 -> copy_n2 ... -> nk -> copy_nk
    p = head
    while p:
        p.next = ListNode(p.val, p.next)
        p = p.next.next
    
    # Step2: Connect random pointer of duplicated node to the duplicated node of random pointer. ie.  copy_node(n1).random = copy_node(n1.random). Note in Step1: copy_node(n1) == n1.next
    p = head
    while p:
        if p.random:
            p.next.random = p.random.next
        p = p.next.next
    
    # Step3: Partition list into even sub-list and odd sub-list. And the odd sub-list is the deep-copy list
    p = head
    new_head = p.next
    while p:
        new_p = p.next
        p.next = p.next.next
        if p.next:
            new_p.next = p.next.next
        else:
            new_p.next = None
        p = p.next
    
    return new_head
    

### Testing Utiltities
class ListNode(object):
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random
    
    def __eq__(self, other):
        if not self.equal(other):
            print "List1 != List2 where"
            print "List1:"
            print str(self)
            print "List2:"
            print str(other)
            print "\n"
            return False
        else:
            return True

    def equal(self, other):
        if other is not None:
            is_random_valid = self.random is None and other.random is None or self.random is not None and other.random is not None and self.random.val == other.random.val
            return is_random_valid and self.val == other.val and self.next == other.next
        else:
            return False
        

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
        
        return "List: [{}].\nRandom[{}]".format(','.join(lst), ','.join(random))
            


class DeepCopySpec(unittest.TestCase):
    def test_empty_list(self):
        self.assertIsNone(deep_copy(None))
    
    def test_list_with_random_point_to_itself(self):
        n = ListNode(1)
        n.random = n
        self.assertEqual(deep_copy(n), n)

    def test_random_pointer_is_None(self):
        # 1 -> 2 -> 3 -> 4
        n4 = ListNode(4)
        n3 = ListNode(3, n4)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)
        self.assertEqual(deep_copy(n1), n1)

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
        self.assertEqual(deep_copy(n1), n1)
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 19, 2019 LC 692 \[Medium\] Top K Frequent words
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

**My thoughts:** Record the frequence of each word, put them into a max heap, finally the final result is the top k element from the max heap. 

**Solution with PriorityQueue:** [https://repl.it/@trsong/Top-K-Frequent-words](https://repl.it/@trsong/Top-K-Frequent-words)
```py
import unittest
from Queue import PriorityQueue

def top_k_freq_words(words, k):
    histogram = {}
    for word in words:
        histogram[word] = histogram.get(word, 0) + 1
        
    pq = PriorityQueue()
    for word, count in histogram.items():
        pq.put((-count, word))
    
    res = []
    for _ in xrange(k):
        _, word = pq.get()
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

### Nov 18, 2019 \[Medium\] Intersection of Two Unsorted Arrays
--- 
> **Question:** Given two arrays, write a function to compute their intersection - the intersection means the numbers that are in both arrays.
> 
> **Note:**
> - Each element in the result must be unique.
> - The result can be in any order.

**Example 1:**
```py
Input: nums1 = [1, 2, 2, 1], nums2 = [2, 2]
Output: [2]
```

**Example 2:**
```py
Input: nums1 = [4, 9, 5], nums2 = [9, 4, 9, 8, 4]
Output: [9, 4]
```

**Solution:** [https://repl.it/@trsong/Intersection-of-Two-Unsorted-Arrays](https://repl.it/@trsong/Intersection-of-Two-Unsorted-Arrays)
```py
import unittest

def binary_search(nums, target):
    lo = 0
    hi = len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return True
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False


def remove_duplicates(nums):
    nums.sort()
    j = 0
    for i in xrange(len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        nums[j] = nums[i]
        j += 1
    
    for _ in xrange(j, len(nums)):
        nums.pop()
    
    return nums


def intersection(nums1, nums2):
    if not nums1 or not nums2:
        return []

    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    nums1.sort()
    res = []
    for i, num in enumerate(nums2):
        if binary_search(nums1, num):
            res.append(num)
    return remove_duplicates(res)


class IntersectionSpec(unittest.TestCase):
    def assert_array(self, arr1, arr2):
        self.assertEqual(sorted(arr1), sorted(arr2))

    def test_example1(self):
        nums1 = [1, 2, 2, 1]
        nums2 = [2, 2]
        expected = [2]
        self.assert_array(expected, intersection(nums1, nums2))
        self.assert_array(expected, intersection(nums2, nums1))

    def test_example2(self):
        nums1 = [4, 9, 5]
        nums2 = [9, 4, 9, 8, 4]
        expected = [9, 4]
        self.assert_array(expected, intersection(nums1, nums2))

    def test_empty_array(self):
        self.assertEqual([], intersection([], []))

    def test_empty_array2(self):
        self.assertEqual([], intersection([1, 1, 1], []))
        self.assertEqual([], intersection([], [1, 2, 3]))

    def test_array_with_duplicated_elements(self):
        nums1 = [1, 1, 1]
        nums2 = [1, 2, 3, 4]
        expected = [1]
        self.assert_array(expected, intersection(nums1, nums2))

    def test_array_with_duplicated_elements2(self):
        nums1 = [1, 2, 3]
        nums2 = [1, 1, 1, 2, 2, 2, 4, 4, 4, 4]
        expected = [1, 2]
        self.assert_array(expected, intersection(nums1, nums2))

    def test_array_with_not_overlapping_elements(self):
        nums1 = [1, 2, 3]
        nums2 = [4, 5, 6]
        expected = []
        self.assert_array(expected, intersection(nums1, nums2))

    
if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 17, 2019 \[Easy\] Plus One
--- 
> **Question:** Given a non-empty array where each element represents a digit of a non-negative integer, add one to the integer. The most significant digit is at the front of the array and each element in the array contains only one digit. Furthermore, the integer does not have leading zeros, except in the case of the number '0'.

**Example:**
```py
Input: [2,3,4]
Output: [2,3,5]
```

**Solution:** [https://repl.it/@trsong/Plus-One](https://repl.it/@trsong/Plus-One)
```py
import unittest

def add1(digits):
    carryover = 1
    for i in xrange(len(digits)-1, -1, -1):
        digits[i] += carryover
        if digits[i] >= 10:
            carryover = 1
            digits[i] %= 10
        else:
            break
    
    if digits[0] == 0:
        digits.insert(0, 1)
    
    return digits


class Add1Spec(unittest.TestCase):
    def test_example(self):
        self.assertEqual([2, 3, 5], add1([2, 3, 4]))

    def test_zero(self):
        self.assertEqual([1], add1([0]))

    def test_carryover(self):
        self.assertEqual([1, 0], add1([9]))

    def test_carryover_and_early_break(self):
        self.assertEqual([2, 8, 3, 0, 0], add1([2, 8, 2, 9, 9]))

    def test_early_break(self):
        self.assertEqual([1, 0, 0, 1], add1([1, 0, 0, 0]))

    def test_carryover2(self):
        self.assertEqual([1, 0, 0, 0, 0], add1([9, 9, 9, 9]))

if __name__ == '__main__':
    unittest.main(exit=False)

```

### Nov 16, 2019 \[Easy\] Exists Overlap Rectangle
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


**Solution:** [https://repl.it/@trsong/Exists-Overlap-Rectangle](https://repl.it/@trsong/Exists-Overlap-Rectangle)
```py
import unittest

def get_bottom_right(rect):
    x, y = rect.get("top_left")
    width, height = rect.get("dimensions")
    return x + width, y - height


def is_overlapped_rectange(rect1, rect2):
    x1, y1 = rect1.get("top_left")
    x2, y2 = get_bottom_right(rect1)
    x3, y3 = rect2.get("top_left")
    x4, y4 = get_bottom_right(rect2)
    x_not_overlap = x2 <= x3 or x1 >= x4
    y_not_overlap = y1 <= y4 or y3 <= y2
    return not x_not_overlap and not y_not_overlap


def exists_overlap_rectangle(rectangles):
    if not rectangles:
        return False
    n = len(rectangles)
    sorted_rectangles = sorted(rectangles, key=lambda rect: rect.get("top_left"))
    for i, rect in enumerate(sorted_rectangles):
        x, _ = rect.get("top_left")
        width, _ = rect.get("dimensions")
        j = i + 1
        while j < n and sorted_rectangles[j].get("top_left")[0] <= x + width:
            if is_overlapped_rectange(sorted_rectangles[j], rect):
                return True
            j += 1 
    return False


class ExistsOverlapRectangleSpec(unittest.TestCase):
    def test_example(self):
        rectangles = [
            {
                "top_left": (1, 4),
                "dimensions": (3, 3) # width, height
            }, {
                "top_left": (-1, 3),
                "dimensions": (2, 1)
            },{
                "top_left": (0, 5),
                "dimensions": (4, 3)
            }]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_empty_rectangle_list(self):
        self.assertFalse(exists_overlap_rectangle([]))
    
    def test_two_overlap_rectangle(self):
        rectangles = [
            {
                "top_left": (0, 1),
                "dimensions": (1, 3) # width, height
            }, {
                "top_left": (-1, 0),
                "dimensions": (3, 1)
            }]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_two_overlap_rectangle_form_a_cross(self):
        rectangles = [
            {
                "top_left": (-1, 1),
                "dimensions": (3, 2) # width, height
            }, {
                "top_left": (0, 0),
                "dimensions": (1, 1)
            }]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_same_y_coord_not_overlap(self):
        rectangles = [
            {
                "top_left": (0, 0),
                "dimensions": (1, 1) # width, height
            }, {
                "top_left": (1, 0),
                "dimensions": (2, 2)
            },{
                "top_left": (3, 0),
                "dimensions": (5, 2)
            }]
        self.assertFalse(exists_overlap_rectangle(rectangles))

    def test_same_y_coord_overlap(self):
        rectangles = [
            {
                "top_left": (0, 0),
                "dimensions": (1, 1) # width, height
            }, {
                "top_left": (1, 0),
                "dimensions": (2, 2)
            },{
                "top_left": (3, 0),
                "dimensions": (5, 2)
            }]
        self.assertFalse(exists_overlap_rectangle(rectangles))
    
    def test_rectangles_in_different_quadrant(self):
        rectangles = [
            {
                "top_left": (1, 1),
                "dimensions": (2, 2) # width, height
            }, {
                "top_left": (-1, 1),
                "dimensions": (2, 2)
            },{
                "top_left": (1, -1),
                "dimensions": (2, 2)
            },{
                "top_left": (-1, -1),
                "dimensions": (2, 2)
            }]
        self.assertFalse(exists_overlap_rectangle(rectangles))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 15, 2019 \[Hard\] Maximum Spanning Tree
--- 
> **Question:** Recall that the minimum spanning tree is the subset of edges of a tree that connect all its vertices with the smallest possible total edge weight. 
> 
> Given an undirected graph with weighted edges, compute the maximum weight spanning tree.

**My thoughts:** Both Kruskal's and Prim's Algorithm works for this question. The idea is to flip all edge weight into negative and then apply either of previous algorithm until find a spanning tree. Check [this question](https://trsong.github.io/python/java/2019/05/01/DailyQuestions/#jul-6-2019-hard-power-supply-to-all-cities) for solution with Kruskal's Algorithm.

**Solution with Prim's Algorithm:** [https://repl.it/@trsong/Maximum-Spanning-Tree](https://repl.it/@trsong/Maximum-Spanning-Tree)
```py
import unittest
from Queue import PriorityQueue
import sys


def max_spanning_tree(vertices, edges):
    if not edges:
        return []

    neighbor = [None] * vertices
    for u, v, w in edges:
        if not neighbor[u]:
            neighbor[u] = []
        neighbor[u].append((v, w))

        if not neighbor[v]:
            neighbor[v] = []
        neighbor[v].append((u, w))
    
    visited = [False] * vertices
    pq = PriorityQueue()
    pq.put((0, 0, None))
    res = []

    while not pq.empty():
        _, dst, src = pq.get()
        if visited[dst]:
            continue

        visited[dst] = True
        if src is not None:
            res.append((src, dst))
        
        if len(res) == vertices - 1:
            break
        
        u = dst
        for v, w in neighbor[u]:
            if visited[v]:
                continue
            pq.put((-w, v, u))

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

### Nov 14, 2019 \[Easy\] Smallest Number that is not a Sum of a Subset of List
--- 
> **Question:** Given a sorted array, find the smallest positive integer that is not the sum of a subset of the array.
>
> For example, for the input `[1, 2, 3, 10]`, you should return `7`.

**My thoughts:** The idea of solving this problem comes from mathematical induction. As the number list is sorted in ascending order, we take element from list one by one. Can consider with the help of the such element, how does it help to push the max possible sum? eg, Let's walk through `[1, 2, 3, 10]`

- `n = 0`. our goal is to check if there is any subset added up to `1`. That's easy, check `nums[0]` to see if it equals `1`.
- `n = 1`. Now we move to next element say `2`. Currently we can from `1`. Now with the help of `2`, additionally we can form `2` and `2 + 1`. It seems we can form all number from 1 to 3.
- `n = 2`. As we can form any number from `1` to `3`. The current element is `3`. With help of this element, additionally we can form `3`, `3 + 1`, `3 + 2`, `3 + 3`. It seems we can form all number from `1` to `6`.
- `n = 3`. As we can form any number from `1` to `6`. The current element is `10`. With the help of this element, additionally we can only get `10`, `10 + 1`, ..., `10 + 6`. However, last round we expand the possible sum to `6`. Now the smallest we can get is `10`, gives a gap. Therefore we return `6 + 1`. 


**Solution by Math Induction:**[https://repl.it/@trsong/Smallest-Number-that-is-not-a-Sum-of-a-Subset-of-List](https://repl.it/@trsong/Smallest-Number-that-is-not-a-Sum-of-a-Subset-of-List)
```py
import unittest

def smallest_subset_sum(nums):
    if not nums:
        return 1

    max_sum_ = 0
    for num in nums:
        if num > max_sum + 1:
            break
        max_sum += num
    return max_sum + 1


class SmallestSubsetSumSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(7, smallest_subset_sum([1, 2, 3, 10]))

    def test_empty_array(self):
        self.assertEqual(1, smallest_subset_sum([]))

    def test_subset_sum_has_gap1(self):
        self.assertEqual(2, smallest_subset_sum([1, 3, 6, 10, 11, 15]))

    def test_subset_sum_has_gap2(self):
        self.assertEqual(4, smallest_subset_sum([1, 2, 5, 10, 20, 40]))

    def test_subset_sum_has_gap3(self):
        self.assertEqual(1, smallest_subset_sum([101, 102, 104]))

    def test_subset_sum_covers_all1(self):
        self.assertEqual(5, smallest_subset_sum([1, 1, 1, 1]))

    def test_subset_sum_covers_all2(self):
        self.assertEqual(26, smallest_subset_sum([1, 2, 4, 8, 10]))

    def test_subset_sum_covers_all3(self):
        self.assertEqual(16, smallest_subset_sum([1, 2, 4, 8]))
        
    def test_subset_sum_covers_all4(self):
        self.assertEqual(10, smallest_subset_sum([1, 1, 3, 4]))

    def test_subset_sum_covers_all5(self):
        self.assertEqual(22, smallest_subset_sum([1, 2, 3, 4, 5, 6]))
 

if __name__ == '__main__':
    unittest.main(exit=False)
```





### Nov 13, 2019 LC 301 \[Hard\] Remove Invalid Parentheses
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


**Solution with Backtracking:** [https://repl.it/@trsong/Remove-Invalid-Parentheses](https://repl.it/@trsong/Remove-Invalid-Parentheses)
```py
import unittest

def remove_invalid_parenthese(s):
    invalid_open = 0
    invalid_close = 0
    for c in s:
        if c == ')' and invalid_open == 0:
            invalid_close += 1
        elif c == '(':
            invalid_open += 1
        elif c == ')':
            invalid_open -= 1
    
    res = []
    backtrack(s, 0, res, invalid_open, invalid_close)
    return res


def backtrack(s, next_index, res, invalid_open, invalid_close):
    if invalid_open == 0 and invalid_close == 0:
        if is_valid(s):
            res.append(s)
    else:
        for i in xrange(next_index, len(s)):
            c = s[i]
            if c == '(' and invalid_open > 0 or c == ')' and invalid_close > 0:
                if i > next_index and s[i] == s[i-1]:
                    # skip consecutive same letters
                    continue

                # update s with c removed
                updated_s = s[:i] + s[i+1:]
                if c == '(':
                    backtrack(updated_s, i, res, invalid_open-1, invalid_close)
                elif c == ')':
                    backtrack(updated_s, i, res, invalid_open, invalid_close-1)


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

### Nov 12, 2019 \[Easy\] Busiest Period in the Building
--- 
> **Question:** You are given a list of data entries that represent entries and exits of groups of people into a building. 
> 
> Find the busiest period in the building, that is, the time with the most people in the building. Return it as a pair of (start, end) timestamps. 
> 
> You can assume the building always starts off and ends up empty, i.e. with 0 people inside.
>
> An entry looks like this:

```py
{"timestamp": 1526579928, count: 3, "type": "enter"}

This means 3 people entered the building. An exit looks like this:

{"timestamp": 1526580382, count: 2, "type": "exit"}

This means that 2 people exited the building. timestamp is in Unix time.
```


**Solution:** [https://repl.it/@trsong/Busiest-Period-in-the-Building](https://repl.it/@trsong/Busiest-Period-in-the-Building) 
```py
import unittest

def busiest_period(event_entries):
    sorted_events = sorted(event_entries, key=lambda e: e.get("timestamp"))
    accu = 0
    max_index = 0
    max_accu = 0
    for i, event in enumerate(sorted_events):
        count = event.get("count")
        accu += count if event.get("type") == "enter" else -count
        if accu > max_accu:
            max_accu = accu
            max_index = i
            
    start = sorted_events[max_index].get("timestamp")
    end = sorted_events[max_index+1].get("timestamp")
    return start, end


class BusiestPeriodSpec(unittest.TestCase):
    def test_example(self):
        # Number of people: 0, 3, 1, 0
        events = [
            {"timestamp": 1526579928, "count": 3, "type": "enter"},
            {"timestamp": 1526580382, "count": 2, "type": "exit"},
            {"timestamp": 1526600382, "count": 1, "type": "exit"}
        ]
        self.assertEqual((1526579928, 1526580382), busiest_period(events))

    def test_multiple_entering_and_exiting(self):
        # Number of people: 0, 3, 1, 7, 8, 3, 2
        events = [
            {"timestamp": 1526579928, "count": 3, "type": "enter"},
            {"timestamp": 1526580382, "count": 2, "type": "exit"},
            {"timestamp": 1526579938, "count": 6, "type": "enter"},
            {"timestamp": 1526579943, "count": 1, "type": "enter"},
            {"timestamp": 1526579944, "count": 0, "type": "enter"},
            {"timestamp": 1526580345, "count": 5, "type": "exit"},
            {"timestamp": 1526580351, "count": 3, "type": "exit"}
        ]
        self.assertEqual((1526579943, 1526579944), busiest_period(events))

    def test_timestamp_not_sorted(self):
        # Number of people: 0, 1, 3, 0
        events = [
            {"timestamp": 2, "count": 2, "type": "enter"},
            {"timestamp": 3, "count": 3, "type": "exit"},
            {"timestamp": 1, "count": 1, "type": "enter"}
        ]
        self.assertEqual((2, 3), busiest_period(events))

    def test_max_period_reach_a_tie(self):
        # Number of people: 0, 1, 10, 1, 10, 1, 0
        events = [
            {"timestamp": 5, "count": 9, "type": "exit"},
            {"timestamp": 3, "count": 9, "type": "exit"},
            {"timestamp": 6, "count": 1, "type": "exit"},
            {"timestamp": 1, "count": 1, "type": "enter"},
            {"timestamp": 4, "count": 9, "type": "enter"},
            {"timestamp": 2, "count": 9, "type": "enter"}
        ]
        self.assertEqual((2, 3), busiest_period(events))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 11, 2019 \[Easy\] Full Binary Tree
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

**Solution:** [https://repl.it/@trsong/Full-Binary-Tree](https://repl.it/@trsong/Full-Binary-Tree)
```py
import unittest
import copy

def remove_partial_nodes(root):
    if root is None:
        return None

    root.left = remove_partial_nodes(root.left)
    root.right = remove_partial_nodes(root.right)
    
    if root.left is None and root.right is not None:
        right_child = root.right
        root.right = None
        return right_child
    elif root.right is None and root.left is not None:
        left_child = root.left
        root.left = None
        return left_child
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
        self.assertEqual(expected_tree, remove_partial_nodes(original_tree))

    def test_empty_tree(self):        
        self.assertIsNone(None, remove_partial_nodes(None))

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
        self.assertEqual(expected_tree, remove_partial_nodes(original_tree))

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
        self.assertEqual(expected_tree, remove_partial_nodes(original_tree))

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
        self.assertEqual(expected_tree, remove_partial_nodes(original_tree))

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
        self.assertEqual(expected_tree, remove_partial_nodes(original_tree))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 10, 2019 \[Medium\] Maximum Path Sum in Binary Tree
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

**Solution with Recursion:** [https://repl.it/@trsong/Maximum-Path-Sum-in-Binary-Tree](https://repl.it/@trsong/Maximum-Path-Sum-in-Binary-Tree)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_path_sum_helper(tree):
    if not tree:
        return 0, 0
    
    left_max_sum, left_max_path = max_path_sum_helper(tree.left)
    right_max_sum, right_max_path = max_path_sum_helper(tree.right)

    max_path = max(left_max_path, right_max_path) + tree.val
    path_sum = left_max_path + tree.val + right_max_path
    max_sum = max(left_max_sum, right_max_sum, path_sum)
    return max_sum, max_path


def max_path_sum(tree):
    return max_path_sum_helper(tree)[0]


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
    unittest.main(exit=False)
```

### Nov 9, 2019 \[Hard\] Order of Alien Dictionary
--- 
> **Question:** You come across a dictionary of sorted words in a language you've never seen before. Write a program that returns the correct order of letters in this language.
>
> For example, given `['xww', 'wxyz', 'wxyw', 'ywx', 'ywz']`, you should return `['x', 'z', 'w', 'y']`.

**My thoughts:** As the alien letters are topologically sorted, we can just mimic what topological sort with numbers and try to find pattern.

Suppose the dictionary contains: `01234`. Then the words can be `023, 024, 12, 133, 2433`. Notice that we can only find the relative order by finding first unequal letters between consecutive words. eg.  `023, 024 => 3 < 4`.  `024, 12 => 0 < 1`.  `12, 133 => 2 < 3`

With relative relation, we can build a graph with each occurring letters being veteces and edge `(u, v)` represents `u < v`. If there exists a loop that means we have something like `a < b < c < a` and total order not exists. Otherwise we preform a topological sort to generate the total order which reveals the alien dictionary. 

**Solution with Topological Sort:** [https://repl.it/@trsong/Order-of-Alien-Dictionary](https://repl.it/@trsong/Order-of-Alien-Dictionary)
```py
import unittest

class NodeState:
    UNVISITED = 0
    VISITING = 1
    VISITED = 2

def dictionary_order(sorted_words):
    if not sorted_words:
        return []

    char_order = {}
    for i in xrange(1, len(sorted_words)):
        prev_word = sorted_words[i-1]
        cur_word = sorted_words[i]
        for prev_char, cur_char in zip(prev_word, cur_word):
            if prev_char == cur_char:
                continue
            if prev_char not in char_order:
                char_order[prev_char] = set()
            char_order[prev_char].add(cur_char)
            break

    char_states = {}
    for word in sorted_words:
        for c in word:
            char_states[c] = NodeState.UNVISITED

    # Performe DFS on all unvisited nodes
    topo_order = []
    for char in char_states:
        if char_states[char] == NodeState.VISITED:
            continue
        
        stack = [char]
        while stack:
            cur_char = stack[-1]
            if char_states[cur_char] == NodeState.VISITED:
                topo_order.append(stack.pop())
                continue
            elif char_states[cur_char] == NodeState.VISITING:
                char_states[cur_char] = NodeState.VISITED
            else:
                char_states[cur_char] = NodeState.VISITING

            if cur_char not in char_order:
                continue
            
            for next_char in char_order[cur_char]:
                if char_states[next_char] == NodeState.UNVISITED:
                    stack.append(next_char)
                elif char_states[next_char] == NodeState.VISITING:
                    return None

    topo_order.reverse()
    return topo_order


class DictionaryOrderSpec(unittest.TestCase):
    def test_example(self):
        # 0123
        # xzwy
        # Decode Array: 022, 2031, 2032, 320, 321
        self.assertEqual(['x', 'z', 'w', 'y'], dictionary_order(['xww', 'wxyz', 'wxyw', 'ywx', 'ywz']))

    def test_empty_dictionary(self):
        self.assertEqual([], dictionary_order([]))

    def test_unique_characters(self):
        self.assertEqual(['z', 'x'], dictionary_order(["z", "x"]), )

    def test_invalid_order(self):
        self.assertIsNone(dictionary_order(["a", "b", "a"]))

    def test_invalid_order2(self):
        # 012
        # abc
        # decode array result become 210, 211, 212, 012
        self.assertIsNone(dictionary_order(["cba", "cbb", "cbc", "abc"]))

    def test_invalid_order3(self):
        # 012
        # abc
        # decode array result become 10, 11, 211, 22, 20 
        self.assertIsNone(dictionary_order(["ba", "bb", "cbb", "cc", "ca"]))
    
    def test_valid_order(self):
        # 01234
        # wertf
        # decode array result become 023, 024, 12, 133, 2433
        self.assertEqual(['w', 'e', 'r', 't', 'f'], dictionary_order(["wrt", "wrf", "er", "ett", "rftt"]))

    def test_valid_order2(self):
        # 012
        # abc
        # decode array result become 01111, 122, 20
        self.assertEqual(['a', 'b', 'c'], dictionary_order(["abbbb", "bcc", "ca"]))

    def test_valid_order3(self):
        # 0123
        # bdac
        # decode array result become 022, 2031, 2032, 320, 321
        self.assertEqual(['b', 'd', 'a', 'c'], dictionary_order(["baa", "abcd", "abca", "cab", "cad"]))

    def test_multiple_valid_result(self):
        self.assertEqual(['a', 'b', 'c', 'd', 'e'], sorted(dictionary_order(["edcba"])))

    def test_multiple_valid_result2(self):
        # 01
        # ab
        # cd
        res = dictionary_order(["aa", "ab", "cc", "cd"])
        expected_set = [['a', 'b', 'c', 'd'], ['a', 'c', 'b', 'd'], ['a', 'd', 'c', 'b'], ['a', 'c', 'd', 'b']]
        self.assertTrue(res in expected_set)

    def test_multiple_valid_result3(self):
        # 01
        # ab
        #  c
        #  d
        res = dictionary_order(["aaaaa", "aaad", "aab", "ac"])
        one_res = ['a', 'b', 'c', 'd']
        self.assertTrue(len(res) == len(one_res) and res[0] == one_res[0] and sorted(res) == one_res)
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 8, 2019 \[Hard\] Longest Common Subsequence of Three Strings
--- 
> **Question:** Write a program that computes the length of the longest common subsequence of three given strings. For example, given `"epidemiologist"`, `"refrigeration"`, and `"supercalifragilisticexpialodocious"`, it should return `5`, since the longest common subsequence is `"eieio"`.

**My thoughts:** The way to tackle 3 lcs of 3 strings is exactly the same as 2 strings. 

For 2 string case, we start from empty strings `s1 = ""` and `s2 = ""` and for each char we append we will need to decide whether or not we want to keep that char that gives the following 2 situations:

- `lcs(s1 + c, s2 + c)` if the new char to append in each string matches each other, then `lcs(s1 + c, s2 + c) = 1 + lcs(s1, s2)`
- `lcs(s1 + c1, s2 + c2)` if not match, we either append to s1 or s2, `lcs(s1 + c, s2 + c) = max(lcs(s1, s2 + c), lcs(s1 + c, s2))`

If we generalize it to 3 string case, we will have 
- `lcs(s1 + c, s2 + c, s3 + c) = 1 + lcs(s1, s2, s3)`
- `lcs(s1 + c, s2 + c, s3 + c) = max(lcs(s1 + c, s2, s3), lcs(s1, s2 + c, s3), lcs(s1, s2, s3 + c))`

One thing worth mentioning: 

`lcs(s1, s2, s2) != lcs(find_lcs(s1, s2), s3)` where find_cls returns string and lcs returns number. The reason we may run into this is that the local max of two strings isn't global max of three strings. 
eg. `s1 = 'abcde'`, `s2 = 'cdeabcd'`, `s3 = 'e'`. `find_lcs(s1, s2) = 'abcd'`. `lcs('abcd', 'e') = 0`. However `lcs(s1, s2, s3) = 1`



**Solution with DP:** [https://repl.it/@trsong/Longest-Common-Subsequence-of-Three-Strings](https://repl.it/@trsong/Longest-Common-Subsequence-of-Three-Strings)
```py
import unittest

def longest_common_subsequence(seq1, seq2, seq3):
    if not seq1 or not seq2 or not seq3:
        return 0
    
    m, n, p = len(seq1), len(seq2), len(seq3)
    # Let dp[m][n][p] represents lcs for seq with len m, n, p
    # dp[m][n][p] = dp[m-1][n-1][p-1] if last char is a match
    # dp[m][n][p] = max(dp[m-1][n][p], dp[m][n-1][p], dp[m][n][p-1])
    dp = [[[0 for _ in xrange(p+1)] for _ in xrange(n+1)] for _ in xrange(m+1)]
    for i in xrange(1, m+1):
        for j in xrange(1, n+1):
            for k in xrange(1, p+1):
                if seq1[i-1] == seq2[j-1] == seq3[k-1]:
                    dp[i][j][k] = dp[i-1][j-1][k-1] + 1
                else:
                    dp[i][j][k] = max(dp[i-1][j][k], dp[i][j-1][k], dp[i][j][k-1])
    return dp[m][n][p]
                    
                    
class LongestCommonSubsequenceSpec(unittest.TestCase):
    def test_example(self):
        seq1 = "epidemiologist"
        seq2 = "refrigeration"
        seq3 = "supercalifragilisticexpialodocious"
        self.assertEqual(5, longest_common_subsequence(seq1, seq2, seq3)) # eieio

    def test_empty_inputs(self):
        self.assertEqual(0, longest_common_subsequence("", "", ""))
        self.assertEqual(0, longest_common_subsequence("", "", "a"))
        self.assertEqual(0, longest_common_subsequence("", "a", "a"))
        self.assertEqual(0, longest_common_subsequence("a", "a", ""))

    def test_contains_common_subsequences(self):
        seq1 = "abcd1e2"  
        seq2 = "bc12ea"  
        seq3 = "bd1ea"
        self.assertEqual(3, longest_common_subsequence(seq1, seq2, seq3)) # b1e

    def test_contains_same_prefix(self):
        seq1 = "abc"  
        seq2 = "abcd"  
        seq3 = "abcde"
        self.assertEqual(3, longest_common_subsequence(seq1, seq2, seq3)) # abc

    def test_contains_same_subfix(self):
        seq1 = "5678"  
        seq2 = "345678"  
        seq3 = "12345678"
        self.assertEqual(4, longest_common_subsequence(seq1, seq2, seq3)) # 5678

    def test_lcs3_not_subset_of_lcs2(self):
        seq1 = "abedcf"
        seq2 = "ibdcfe"
        seq3 = "beklm"
        """
        LCS(seq1, seq2) = bdcf
        LCS(LCS(seq1, seq2), seq3) = b
        LCS(seq1, seq2, seq3) = be
        """
        self.assertEqual(2, longest_common_subsequence(seq1, seq2, seq3)) # be

    def test_lcs3_not_subset_of_lcs2_example2(self):
        seq1 = "abedcf"
        seq2 = "ibdcfe"
        seq3 = "eklm"
        """
        LCS(seq1, seq2) = bdcf
        LCS(LCS(seq1, seq2), seq3) = None
        LCS(seq1, seq2, seq3) = e
        """
        self.assertEqual(1, longest_common_subsequence(seq1, seq2, seq3)) # e

    def test_multiple_candidates_of_lcs(self):
        s1 = "abcd"
        s2 = "123"
        s3 = "456"
        s4 = "efgh"
        seq1 = s1 + s3 + s4 + s2
        seq2 = s3 + s4 + s2 + s1
        seq3 = s4 + s3 + s2 + s1
        self.assertEqual(len(s4) + len(s2), longest_common_subsequence(seq1, seq2, seq3))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 7, 2019 \[Medium\] No Adjacent Repeating Characters
--- 
> **Question:** Given a string, rearrange the string so that no character next to each other are the same. If no such arrangement is possible, then return None.

**Example:**
```py
Input: abbccc
Output: cbcbca
```

**My thougths:** This problem is basically ["LC 358 Rearrange String K Distance Apart"](https://trsong.github.io/python/java/2019/08/02/DailyQuestionsAug/#aug-24-2019-lc-358-hard-rearrange-string-k-distance-apart) with `k = 2`. The idea is to apply greedy approach, with a window of 2, choose the safest two remaining charactors until either all characters are picked, or just mulitple copy of one character left.

For example, for input string: `"aaaabc"`
1. Pick `a`, `b`. Remaining `"aaac"`. Result: `"ab"`
2. Pick `a`, `c`. Remaining `"aa"`. Result: `"abac"`
3. We can no longer proceed as we cannot pick same character as it violate adjacency requirement

Another Example, for input string: `"abbccc"`
1. Pick `c`, `b`. Remaining `"abcc"`. Result: `"ab"`
2. Pick `c`, `a`. Remaining `"bc"`. Result: `"abca"`
3. Pick `b`, `c`. Result: `"abcabc"`


**Solution with Greedy Algorithm:** [https://repl.it/@trsong/No-Adjacent-Repeating-Characters](https://repl.it/@trsong/No-Adjacent-Repeating-Characters)
```py
import unittest
from Queue import PriorityQueue

def rearrange_string(original_string):
    histogram = {}
    for c in original_string:
        histogram[c] = histogram.get(c, 0) + 1
        
    max_heap = PriorityQueue()
    for c, count in histogram.items():
        # Max heap is implemented with PriorityQueue (small element goes first), therefore use negative key to achieve max heap
        max_heap.put((-count, c))

    res = []
    while not max_heap.empty():
        neg_first_count, first_char = max_heap.get()
        remaining_first_count = -neg_first_count - 1
        res.append(first_char)

        if remaining_first_count > 0 and max_heap.empty():
            return None
        elif max_heap.empty():
            break

        neg_second_count, second_char = max_heap.get()
        remainging_second_count = -neg_second_count - 1
        res.append(second_char)

        if remaining_first_count > 0:
            max_heap.put((-remaining_first_count, first_char))
        if remainging_second_count > 0:
            max_heap.put((-remainging_second_count, second_char))

    return ''.join(res)


class RearrangeStringSpec(unittest.TestCase):
    def assert_not_adjacent(self, rearranged_string, original_string):
        # Test same length
        self.assertTrue(len(original_string) == len(rearranged_string))

        # Test containing all characters
        self.assertTrue(sorted(original_string) == sorted(rearranged_string))

        # Test not adjacent
        last_occur_map = {}
        for i, c in enumerate(rearranged_string):
            last_occur = last_occur_map.get(c, float('-inf'))
            self.assertTrue(i - last_occur >= 2)
            last_occur_map[c] = i
    
    def test_empty_string(self):
        self.assertEqual("", rearrange_string(""))

    def test_example(self):
        original_string = "abbccc"
        rearranged_string = rearrange_string(original_string)
        self.assert_not_adjacent(rearranged_string, original_string)

    def test_original_string_contains_duplicated_characters(self):
        original_string = "aaabb"
        rearranged_string = rearrange_string(original_string)
        self.assert_not_adjacent(rearranged_string, original_string)

    def test_original_string_contains_duplicated_characters2(self):
        original_string = "aaabc"
        rearranged_string = rearrange_string(original_string)
        self.assert_not_adjacent(rearranged_string, original_string)

    def test_impossible_to_rearrange_string(self):
        original_string = "aa"
        self.assertIsNone(rearrange_string(original_string))

    def test_impossible_to_rearrange_string2(self):
        original_string = "aaaabc"
        self.assertIsNone(rearrange_string(original_string))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 6, 2019 \[Easy\] Zombie in Matrix
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

**Solution with BFS:** [https://repl.it/@trsong/Zombie-in-Matrix](https://repl.it/@trsong/Zombie-in-Matrix) 
```py
import unittest
from Queue import Queue

def zombie_infection_time(grid):
    if not grid or not grid[0]:
        return -1
    n, m = len(grid), len(grid[0])
    
    queue = Queue()
    for r in xrange(n):
        for c in xrange(m):
            if grid[r][c] == 1:
                queue.put((r, c))

    time = -1
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    while not queue.empty():
        for _ in xrange(queue.qsize()):
            r, c = queue.get()
            if time > 0 and grid[r][c] == 1:
                continue
                
            grid[r][c] = 1
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < m and grid[nr][nc] == 0:
                    queue.put((nr, nc))
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
    unittest.main(exit=False)
```

### Nov 5, 2019 \[Medium\] Normalize Pathname
---
> **Question:** Given an absolute pathname that may have `"."` or `".."` as part of it, return the shortest standardized path.
>
> For example, given `"/usr/bin/../bin/./scripts/../"`, return `"/usr/bin"`.

**Solution with Stack:** [https://repl.it/@trsong/Normalize-Pathname](https://repl.it/@trsong/Normalize-Pathname)
```py
import unittest

def normalize_pathname(path):
    stack = []
    pathname = []
    prev = None
    for c in path:
        if c == '/' and prev != c and pathname:
            stack.append(''.join(pathname))
            pathname = []
        elif c == '.' and prev == c and stack:
            stack.pop()
        elif c.isalnum():
            if prev == '.':
                pathname.append('.')
            pathname.append(c)
        prev = c
    
    if pathname:
        stack.append(''.join(pathname))

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
    unittest.main(exit=False)
```

### Nov 4, 2019 \[Easy\] Make the Largest Number
---
> **Question:** Given a number of integers, combine them so it would create the largest number.

**Example:**
```py
Input:  [17, 7, 2, 45, 72]
Output:  77245217
```

**Solution with Customized Sort:** [https://repl.it/@trsong/Make-the-Largest-Number](https://repl.it/@trsong/Make-the-Largest-Number)
```py
import unittest

def construct_largest_number(nums):
    if not nums:
        return 0

    def string_cmp(s1, s2):
        s12 = s1 + s2
        s21 = s2 + s1
        if s12 < s21:
            return -1
        elif s12 == s21:
            return 0
        else:
            return 1
    
    negative_nums = filter(lambda x: x < 0, nums)
    filtered_nums = filter(lambda x: x >= 0, nums)
    str_nums = map(str, filtered_nums)

    if negative_nums:
        str_nums.sort(string_cmp)
        return int(str(negative_nums[0]) + "".join(str_nums))
    else:
        str_nums.sort(string_cmp, reverse=True)
        return int("".join(str_nums))


class ConstructLargestNumberSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(77245217, construct_largest_number([17, 7, 2, 45, 72]))

    def test_empty_array(self):
        self.assertEqual(0, construct_largest_number([]))

    def test_array_with_one_element(self):
        self.assertEqual(-42, construct_largest_number([-42]))

    def test_array_with_duplicated_zeros(self):
        self.assertEqual(4321000, construct_largest_number([4, 1, 3, 2, 0, 0, 0]))

    def test_array_with_unaligned_digits(self):
        self.assertEqual(123451234123121, construct_largest_number([1, 12, 123, 1234, 12345]))

    def test_array_with_unaligned_digits2(self):
        self.assertEqual(4434324321, construct_largest_number([4321, 432, 43, 4]))
    
    def test_array_with_unaligned_digits3(self):
        self.assertEqual(6054854654, construct_largest_number([54, 546, 548, 60]))
    
    def test_array_with_unaligned_digits4(self):
        self.assertEqual(998764543431, construct_largest_number([1, 34, 3, 98, 9, 76, 45, 4]))

    def test_array_with_negative_numbers(self):
        self.assertEqual(-101234, construct_largest_number([-1, 0, 1, 2, 3, 4]))
    
    def test_array_with_negative_numbers2(self):
        self.assertEqual(-99990101202123442, construct_largest_number([0, 1, 10, 2, 21, 20, 34, 42, -9999]))
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 3, 2019 \[Easy\] Find Unique Element among Array of Duplicates
---
> **Question:** Given an array of integers, arr, where all numbers occur twice except one number which occurs once, find the number. Your solution should ideally be O(n) time and use constant extra space.
> 
**Example:**
```py
Input: arr = [7, 3, 5, 5, 4, 3, 4, 8, 8]
Output: 7
```

**My thoughts:** XOR has many pretty useful properties:

* 0 ^ x = x
* x ^ x = 0
* x ^ y = y ^ x 

For example: 
```py
7 ^ 3 ^ 5 ^ 5 ^ 4 ^ 3 ^ 4 ^ 8 ^ 8
= 7 ^ 3 ^ (5 ^ 5) ^ 4 ^ 3 ^ 4 ^ (8 ^ 8)
= 7 ^ 3 ^ 4 ^ 3 ^ 4 
= 7 ^ 3 ^ 3 ^ 4 ^ 4
= 7 ^ (3 ^ 3) ^ (4 ^ 4)
= 7
```

**Solution with XOR:** [https://repl.it/@trsong/Find-Unique-Element-among-Array-of-Duplicates](https://repl.it/@trsong/Find-Unique-Element-among-Array-of-Duplicates)
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
    unittest.main(exit=False)
```


### Nov 2, 2019 \[Easy\] String Compression
---
> **Question:** Given an array of characters with repeats, compress it in place. The length after compression should be less than or equal to the original array.

**Example:**
```py
Input: ['a', 'a', 'b', 'c', 'c', 'c']
Output: ['a', '2', 'b', 'c', '3']
```

**Solution with Two-Pointers:** [https://repl.it/@trsong/String-Compression](https://repl.it/@trsong/String-Compression)
```py
import unittest

def string_compression(msg):
    if not msg:
        return

    n = len(msg)
    count = 1
    j = 0
    for i in xrange(1, len(msg)+1):
        if i < n and msg[i] == msg[i-1]:
            count += 1
        else:
            msg[j] = msg[i-1]
            j += 1
            if count > 1:    
                for char in str(count):
                    msg[j] = char
                    j += 1
            count = 1
    
    for _ in xrange(j, n):
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

### Nov 1, 2019 \[Hard\] Partition Array to Reach Mininum Difference
---
> **Question:** Given an array of positive integers, divide the array into two subsets such that the difference between the sum of the subsets is as small as possible.
>
> For example, given `[5, 10, 15, 20, 25]`, return the sets `[10, 25]` and `[5, 15, 20]`, which has a difference of `5`, which is the smallest possible difference.


**Solution with DP:** [https://repl.it/@trsong/Partition-Array-to-Reach-Mininum-Difference](https://repl.it/@trsong/Partition-Array-to-Reach-Mininum-Difference)
```py
import unittest

def min_partition_difference(nums):
    if not nums:
        return 0

    sum_nums = sum(nums)
    n = len(nums)

    # Let dp[i][s] represents whether subset of nums[0:i] can sum up to s
    # dp[i][s] = dp[i-1][s]               if exclude current element
    # dp[i][s] = dp[i-1][s - nums[i-1]]   if incldue current element
    dp = [[False for _ in xrange(sum_nums + 1)] for _ in xrange(n+1)]

    for i in xrange(n+1):
        # Any subsets can sum up to 0
        dp[i][0] = True

    for i in xrange(1, n+1):
        for s in xrange(1, sum_nums+1):
            if dp[i-1][s]:
                # Exclude current element, as we can already reach sum s
                dp[i][s] = True
            elif nums[i-1] <= s:
                # As all number are positive, include current elem cannot exceed current sum
                dp[i][s] = dp[i-1][s - nums[i-1]]

    """
     Let's do some math here:
     Let s1, s2 be the size to two subsets after partition and assume s1 >= s2
     We can have s1 + s2 = sum_nums and we want to get min{s1 - s2} where s1 >= s2:

     min{s1 - s2}
     = min{s1 + s2 - s2 - s2}
     = min{sum_nums - 2 * s2}  in this step sum_nums - 2 * s2 >=0, gives s2 <= sum_nums/ 2
     = sum_nums - 2 * max{s2}  where s2 <= sum_nums/2
    """
    s2 = 0
    for s in xrange(sum_nums/2, 0, -1):
        if dp[n][s]:
            s2 = s
            break

    return sum_nums - 2 * s2


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
    unittest.main(exit=False)
```

