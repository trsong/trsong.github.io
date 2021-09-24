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


### Sep 23, 2021 \[Medium\] Longest Alternating Subsequence Problem
---
> **Question:** Finding the length of a subsequence of a given sequence in which the elements are in alternating order, and in which the sequence is as long as possible. 
>
> For example, consider array `A[] = [8, 9, 6, 4, 5, 7, 3, 2, 4]` The length of longest subsequence is `6` and the subsequence is `[8, 9, 6, 7, 3, 4]` as `(8 < 9 > 6 < 7 > 3 < 4)`.


**Solution:** [https://replit.com/@trsong/Solve-Longest-Alternating-Subsequence-Problem](https://replit.com/@trsong/Solve-Longest-Alternating-Subsequence-Problem)
```py
import unittest

def find_len_of_longest_alt_seq(nums):
    if not nums:
        return 0

    prev_sign = None
    res = 1

    for i in range(1, len(nums)):
        sign = nums[i] - nums[i - 1]
        if sign == 0:
            continue

        if prev_sign is None or sign * prev_sign < 0:
            res += 1
        prev_sign = sign
    return res


class FindLenOfLongestAltSeqSpec(unittest.TestCase):
    def test_example(self):
        nums = [8, 9, 6, 4, 5, 7, 3, 2, 4]
        expected = 6  # [8, 9, 6, 7, 3, 4]
        self.assertEqual(expected, find_len_of_longest_alt_seq(nums))

    def test_empty_array(self):
        nums = []
        expected = 0
        self.assertEqual(expected, find_len_of_longest_alt_seq(nums))

    def test_entire_array_is_alternating(self):
        nums = [1, 7, 4, 9, 2, 5]
        expected = 6  # [1, 7, 4, 9, 2, 5]
        self.assertEqual(expected, find_len_of_longest_alt_seq(nums))

    def test_multiple_results(self):
        nums = [1, 17, 5, 10, 13, 15, 10, 5, 16, 8]
        expected = 7  # One solution: [1, 17, 10, 13, 10, 16, 8]
        self.assertEqual(expected, find_len_of_longest_alt_seq(nums))

    def test_increasing_array(self):
        nums = [1, 3, 8, 9]
        expected = 2  # One solution: [1, 3]
        self.assertEqual(expected, find_len_of_longest_alt_seq(nums))

    def test_local_solution_is_not_optimal(self):
        nums = [4, 8, 2, 5, 8, 6]
        expected = 5  # [4, 8, 2, 8, 6]
        self.assertEqual(expected, find_len_of_longest_alt_seq(nums))

    def test_same_element(self):
        nums = [0, 0, 0, 1, -1, 1, -1]
        expected = 5  # [0, 1, -1, 1, -1]
        self.assertEqual(expected, find_len_of_longest_alt_seq(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 22, 2021 \[Medium\] Egg Dropping Puzzle
---
> **Questions:** You are given N identical eggs and access to a building with k floors. Your task is to find the lowest floor that will cause an egg to break, if dropped from that floor. Once an egg breaks, it cannot be dropped again. If an egg breaks when dropped from the xth floor, you can assume it will also break when dropped from any floor greater than x.
>
> Write an algorithm that finds the minimum number of trial drops it will take, in the worst case, to identify this floor.

**Example1:**
```py
Input: N = 1, k = 5, 
Output: 5
we will need to try dropping the egg at every floor, beginning with the first, until we reach the fifth floor, so our solution will be 5.
```

**Example2:**
```py
Input: N = 2, k = 36
Minimum number of trials in worst case with 2 eggs and 36 floors is 8
```

**My thoughts:** You might be confused about what this question is asking, take a look at the following walkthrough with 2 eggs and 10 floors:
```py
# Test Plan for 2 eggs and 10 floors:
# 1.Test floor 4, on failure floor 1 to 3, worst case total = 1 + 3 = 4. 
# 2.Test floor 4+3=7, on failure floor 5 to 6, worst case total = 2 + 2 = 4
# 3.Test floor 4+3+2=9, on failure floor 8, worst case total = 1 + 3 = 4
# 4.Test floor 4+3+2=10. Total = 4
# So, if we try floor, 4, 7, 9, 10 with 1st egg and on failure use 2nd egg to test floors in the middle, we can use as few as 2 eggs and mimium 4 trials.
# Thus the answer should be 4. 

# Note: we cannot skip floor 10 given that we have only 10 floors and floor 1 to 9 are already tested because we cannot make sure egg won't break at floor 10 until we actually drop the egg at that floor
```

The idea is to use dp to iterate through all possible scenarios. Notice the recursion relationship: 
```py
solve_egg_drop_puzzle(eggs, floors) = 1 + min(max(solve_egg_drop_puzzle(eggs-1, i-1), solve_egg_drop_puzzle(eggs, floors-i))) for all i ranges from 1 to floors inclusive.
```
There are two different outcomes if we drop egg at level i: break or not break. Either way we waste 1 trail. 

- If egg break, we end up solving `solve_egg_drop_puzzle(eggs-1, i-1)`, as we waste 1 egg to test floor i, the remaining egg reduce and floors reduce.
- Otherwise, we try to solve `solve_egg_drop_puzzle(eggs, floors-i)`, as we keep the last egg and test remainging floors upstairs.

In above two cases, floor `i` can be any floor, and we want to find the result of the best i such that no matter which siutation is the worse, we end up with minimum trails.


**Solution with Minimax:** [https://replit.com/@trsong/Egg-Dropping-Puzzle-Problem](https://replit.com/@trsong/Egg-Dropping-Puzzle-Problem)
```py
import unittest

def solve_egg_drop_puzzle(eggs, floors):
    cache = [[None for _ in range(floors + 1)] for _ in range(eggs + 1)]
    return solve_egg_drop_puzzle_with_cache(eggs, floors, cache)


def solve_egg_drop_puzzle_with_cache(eggs, floors, cache):
    if eggs == 1 or floors == 0 or floors == 1:
        return floors
    
    if cache[eggs][floors] is None:
        res = float('inf')
        for f in range(1, floors + 1):
            egg_break = solve_egg_drop_puzzle_with_cache(eggs - 1, f - 1, cache)
            egg_non_break = solve_egg_drop_puzzle_with_cache(eggs, floors - f, cache)
            worse_case = 1 + max(egg_break, egg_non_break)
            res = min(res, worse_case)
        cache[eggs][floors] = res

    return cache[eggs][floors]


class SolveEggDropPuzzleSpec(unittest.TestCase):
    def test_example1(self):
        # worst case floor = 5
        # test from floor 1 to 5
        self.assertEqual(5, solve_egg_drop_puzzle(eggs=1, floors=5))

    def test_example2(self):
        # Possible Testing Plan with minimum trials(solution is not unique):
        # 1. Test floor 8, on failure test floor 1 to 7, worst case total = 8
        # 2. Test floor 8+7=15, on failure test floor 9 to 14, worst case total = 2 + 6 = 8
        # 3. Test floor 8+7+6=21, on failure test floor 16 to 20, worst case total = 3 + 5 = 8
        # 4. Test floor 8+7+6+5=26, on failure test floor 22 to 25, worst case total = 4 + 4 = 8
        # 5. Test floor 8+7+6+5+4=30, on failure test floor 27 to 29, worst case total = 5 + 3 = 8
        # 6. Test floor 8+7+6+5+4+3=33, on failure test floor 31 to 32, worst case total = 6 + 2 = 8
        # 7. Test floor 8+7+6+5+4+3+2=35, on failure test floor 34, worst case total = 7 + 1 = 8
        # 8. Test floor 36. total = 8
        self.assertEqual(8, solve_egg_drop_puzzle(eggs=2, floors=36))

    def test_num_eggs_greater_than_floors(self):
        self.assertEqual(3, solve_egg_drop_puzzle(eggs=3, floors=4))

    def test_num_eggs_greater_than_floors2(self):
        self.assertEqual(4, solve_egg_drop_puzzle(eggs=20, floors=10))

    def test_zero_floors(self):
        self.assertEqual(0, solve_egg_drop_puzzle(eggs=10, floors=0))
    
    def test_one_floor(self):
        self.assertEqual(1, solve_egg_drop_puzzle(eggs=1, floors=1))
    
    def test_unique_solution(self):
        # 1.Test floor 4, on failure floor 1 to 3, worst case total = 1 + 3 = 4. 
        # 2.Test floor 4+3=7, on failure floor 5 to 6, worst case total = 2 + 2 = 4
        # 3.Test floor 4+3+2=9, on failure floor 8, worst case total = 1 + 3 = 4
        # 4.Test floor 4+3+2=10. Total = 4
        self.assertEqual(4, solve_egg_drop_puzzle(eggs=2, floors=10))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 21, 2021 \[Medium\] 4 Sum
---
> **Question:** Given a list of numbers, and a target number n, find all unique combinations of a, b, c, d, such that a + b + c + d = n.

**Example 1:**
```py
fourSum([1, 1, -1, 0, -2, 1, -1], 0)
# returns [[-1, -1, 1, 1], [-2, 0, 1, 1]]
```

**Example 2:**
```py
fourSum([3, 0, 1, -5, 4, 0, -1], 1)
# returns [[-5, -1, 3, 4]]
```

**Example 3:**
```py
fourSum([0, 0, 0, 0, 0], 0)
# returns [[0, 0, 0, 0]]
```

**Solution:** [https://replit.com/@trsong/Solve-4-Sum-Problem](https://replit.com/@trsong/Solve-4-Sum-Problem)
```py
import unittest

def four_sum(nums, target):
    nums.sort()
    n = len(nums)
    res = []
    for i, first in enumerate(nums):
        if i > 0 and nums[i - 1] == nums[i]:
            continue

        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j - 1] == nums[j]:
                continue
            second = nums[j]
            sub_target = target - first - second
            
            for third, fourth in two_sum(nums, sub_target, j + 1, n - 1):
                res.append([first, second, third, fourth])
    return res


def two_sum(nums, target, start, end):
    while start < end:
        total = nums[start] + nums[end]
        if total < target:
            start += 1
        elif total > target:
            end -= 1
        else:
            yield nums[start], nums[end]
            start += 1
            end -= 1
            
            while start < end and nums[start - 1] == nums[start]:
                start += 1

            while start < end and nums[end] == nums[end + 1]:
                end -= 1
     
            
class FourSumSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(len(expected), len(result))
        for lst in expected:
            lst.sort()
        for lst2 in result:
            lst2.sort()
        expected.sort()
        result.sort()
        self.assertEqual(expected, result)

    def test_example1(self):
        target, nums = 0, [1, 1, -1, 0, -2, 1, -1]
        expected = [[-1, -1, 1, 1], [-2, 0, 1, 1]]
        self.assert_result(expected, four_sum(nums, target))

    def test_example2(self):
        target, nums = 1, [3, 0, 1, -5, 4, 0, -1]
        expected = [[-5, -1, 3, 4]]
        self.assert_result(expected, four_sum(nums, target))

    def test_example3(self):
        target, nums = 0, [0, 0, 0, 0, 0]
        expected = [[0, 0, 0, 0]]
        self.assert_result(expected, four_sum(nums, target))

    def test_not_enough_elements(self):
        target, nums = 0, [0, 0, 0]
        expected = []
        self.assert_result(expected, four_sum(nums, target))

    def test_unable_to_find_target_sum(self):
        target, nums = 0, [-1, -2, -3, -1, 0, 0, 0]
        expected = []
        self.assert_result(expected, four_sum(nums, target))

    def test_all_positives(self):
        target, nums = 23, [10, 2, 3, 4, 5, 9, 7, 8]
        expected = [
            [2, 3, 8, 10], 
            [2, 4, 7, 10], 
            [2, 4, 8, 9], 
            [2, 5, 7, 9], 
            [3, 4, 7, 9], 
            [3, 5, 7, 8]
        ]
        self.assert_result(expected, four_sum(nums, target))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 20, 2021 \[Easy\] Fancy Number
---
> **Question:** Check if a given number is Fancy. A fancy number is one which when rotated 180 degrees is the same. Given a number, find whether it is fancy or not.
>
> 180 degree rotations of 6, 9, 1, 0 and 8 are 9, 6, 1, 0 and 8 respectively

```py
import unittest

ROTATION_MAP = {
    0: 0,
    1: 1,
    6: 9,
    8: 8,
    9: 6
}


def is_fancy_number(num):
    return num == rotate(num)


def rotate(num):
    res = 0
    while num > 0:
        digit = num % 10
        if digit not in ROTATION_MAP:
            return None
        res = 10 * res + ROTATION_MAP[digit]
        num //= 10
    return res


class IsFancyNumberSpec(unittest.TestCase):
    def test_fancy_number(self):
        self.assertTrue(is_fancy_number(69))
    
    def test_fancy_number2(self):
        self.assertTrue(is_fancy_number(916))

    def test_fancy_number3(self):
        self.assertTrue(is_fancy_number(0))

    def test_not_fancy_number(self):
        self.assertFalse(is_fancy_number(996))
    
    def test_not_fancy_number2(self):
        self.assertFalse(is_fancy_number(121))

    def test_not_fancy_number3(self):
        self.assertFalse(is_fancy_number(110))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 19, 2021 LC 133 \[Medium\] Deep Copy Graph
---
> **Question:** Given a node in a connected directional graph, create a deep copy of it.
>
> Each node in the graph contains a val (int) and a list (List[Node]) of its neighbors:

```py
class Node(object):
    def __init__(self, val, neighbors=None):
        self.val = val
        self.neighbors = neighbors
```

**Solution with DFS:** [https://replit.com/@trsong/Deep-Copy-A-Connected-Directional-Graph](https://replit.com/@trsong/Deep-Copy-A-Connected-Directional-Graph)
```py
import unittest

def deep_copy(root):
    if not root:
        return None

    node_lookup = { node: Node(node.val) for node in dfs_traversal(root) }
    for node, copy in node_lookup.items():
        if not node.neighbors:
            continue
        copy.neighbors = list(map(lambda child: node_lookup[child], node.neighbors))
    return node_lookup[root]


def dfs_traversal(root):
    stack = [root]
    visited = set()
    res = []
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        res.append(cur)

        if not cur.neighbors:
            continue
        for child in cur.neighbors:
            stack.append(child)
    return res


###################
# Testing Utilities
###################
class Node(object):
    def __init__(self, val, neighbors=None):
        self.val = val
        self.neighbors = neighbors

    def __eq__(self, other):
        if not other:
            return False

        nodes = dfs_traversal(self)
        other_nodes = dfs_traversal(other)
        if len(nodes) != len(other_nodes):
            return False
        
        for n1, n2 in zip(nodes, other_nodes):
            if n1.val != n2.val:
                return False

            num_neighbor1 = len(n1.neighbors) if n1.neighbors else 0
            num_neighbor2 = len(n2.neighbors) if n2.neighbors else 0
            if num_neighbor1 != num_neighbor2:
                return False

        return True

    def __repr__(self):
        res = []
        for node in dfs_traversal(self):
            res.append(str(node.val))
        return "DFS: " + ",".join(res)     
    
    def __hash__(self):
        return id(self)


class DeepCopySpec(unittest.TestCase):
    def assert_result(self, root1, root2):
        # check against each node if two graph have same value yet different memory addresses
        self.assertEqual(root1, root2)
        node_set1 = set(dfs_traversal(root1))
        node_set2 = set(dfs_traversal(root2))
        self.assertEqual(set(), node_set1 & node_set2)

    def test_empty_graph(self):
        self.assertIsNone(deep_copy(None))

    def test_graph_with_one_node(self):
        root = Node(1)
        self.assert_result(root, deep_copy(root))

    def test_k3(self):
        n = [Node(i) for i in range(3)]
        n[0].neighbors = [n[1], n[2]]
        n[1].neighbors = [n[0], n[2]]
        n[2].neighbors = [n[1], n[0]]
        self.assert_result(n[0], deep_copy(n[0]))
    
    def test_DAG(self):
        n = [Node(i) for i in range(4)]
        n[0].neighbors = [n[1], n[2]]
        n[1].neighbors = [n[2], n[3]]
        n[2].neighbors = [n[3]]
        n[3].neighbors = []
        self.assert_result(n[0], deep_copy(n[0]))

    def test_graph_with_cycle(self):
        n = [Node(i) for i in range(6)]
        n[0].neighbors = [n[1]]
        n[1].neighbors = [n[2]]
        n[2].neighbors = [n[3]]
        n[3].neighbors = [n[4]]
        n[4].neighbors = [n[5]]
        n[5].neighbors = [n[2]]
        self.assert_result(n[0], deep_copy(n[0]))

    def test_k10(self):
        n = [Node(i) for i in range(10)]
        for i in range(10):
            n[i].neighbors = n[:i] + n[i+1:]
        self.assert_result(n[0], deep_copy(n[0]))

    def test_tree(self):
        n = [Node(i) for i in range(5)]
        n[0].neighbors = [n[1], n[2]]
        n[2].neighbors = [n[3], n[4]]
        self.assert_result(n[0], deep_copy(n[0]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 18, 2021 LC 308 \[Hard\] Range Sum Query 2D - Mutable
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

sum_region(2, 1, 4, 3)   # returns 8
update(3, 2, 2)
sum_region(2, 1, 4, 3)   # returns 10
```


**Solution with 2D Binary Indexed Tree:** [https://replit.com/@trsong/Solve-Range-Sum-Query-2D-Mutable](https://replit.com/@trsong/Solve-Range-Sum-Query-2D-Mutable)
```py
import unittest

class RangeSumQuery2D(object):
    def __init__(self, matrix):
        n, m = len(matrix), len(matrix[0])
        self.bit_matrix = [[0 for _ in range(m+1)] for _ in range(n+1)]
        for r in range(n):
            for c in range(m):
                self.update(r, c, matrix[r][c])

    def sum_top_left_reigion(self, row, col):
        res = 0
        bit_row_index = row + 1
        while bit_row_index > 0:
            bit_col_index = col + 1
            while bit_col_index > 0:
                res += self.bit_matrix[bit_row_index][bit_col_index]
                bit_col_index -= bit_col_index & -bit_col_index
            bit_row_index -= bit_row_index & -bit_row_index
        return res

    def sum_region(self, top_left_row, top_left_col, bottom_right_row, bottom_right_col):
        sum_bottom_right = self.sum_top_left_reigion(bottom_right_row, bottom_right_col)
        sum_bottom_left = self.sum_top_left_reigion(bottom_right_row, top_left_col-1)
        sum_top_right = self.sum_top_left_reigion(top_left_row-1, bottom_right_col)
        sum_top_left = self.sum_top_left_reigion(top_left_row-1, top_left_col-1)
        return sum_bottom_right - sum_bottom_left - sum_top_right + sum_top_left

    def update(self, row, col, val):
        n, m = len(self.bit_matrix), len(self.bit_matrix[0])
        diff = val - self.sum_region(row, col, row, col)
        bit_row_index = row + 1
        while bit_row_index < n:
            bit_col_index = col + 1
            while bit_col_index < m:
                self.bit_matrix[bit_row_index][bit_col_index] += diff
                bit_col_index += bit_col_index & -bit_col_index
            bit_row_index += bit_row_index & -bit_row_index


class RangeSumQuery2DSpec(unittest.TestCase):
    def test_example(self):
        matrix = [
            [3, 0, 1, 4, 2],
            [5, 6, 3, 2, 1],
            [1, 2, 0, 1, 5],
            [4, 1, 0, 1, 7],
            [1, 0, 3, 0, 5]
        ]
        rsq = RangeSumQuery2D(matrix)
        self.assertEqual(8, rsq.sum_region(2, 1, 4, 3))
        rsq.update(3, 2, 2)
        self.assertEqual(10, rsq.sum_region(2, 1, 4, 3))

    def test_non_square_matrix(self):
        matrix = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]
        rsq = RangeSumQuery2D(matrix)
        self.assertEqual(6, rsq.sum_region(0, 1, 1, 3))
        rsq.update(0, 2, 2)
        self.assertEqual(7, rsq.sum_region(0, 1, 1, 3))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 17, 2021 \[Easy\] Markov Chain
--- 
> **Question:** You are given a starting state start, a list of transition probabilities for a Markov chain, and a number of steps num_steps. Run the Markov chain starting from start for num_steps and compute the number of times we visited each state.
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

**Solution with Binary Search:** [https://replit.com/@trsong/Run-the-Markov-Chain](https://replit.com/@trsong/Run-the-Markov-Chain)
```py
import random
import unittest

def markov_chain(start_state, num_steps, transition_probabilities):
    transition_map = {}
    for src, dst, prob in transition_probabilities:
        if src not in transition_map:
            transition_map[src] = [[], []]
        transition_map[src][0].append(dst)
        transition_map[src][1].append(prob)
    
    for src in transition_map:
        dst_probs = transition_map[src][1]
        for i in range(1, len(dst_probs)):
            # accumulate probability
            dst_probs[i] += dst_probs[i-1]
    
    state_counter = {}
    current_state = start_state
    for _ in range(num_steps):
        state_counter[current_state] = state_counter.get(current_state, 0) + 1
        dst_list, des_prob = transition_map[current_state]
        random_number = random.random()
        next_state_index = binary_search(des_prob, random_number)
        next_state = dst_list[next_state_index]
        current_state = next_state

    return state_counter


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
    unittest.main(exit=False, verbosity=2)
```


### Sep 16, 2021 \[Medium\] Strongly Connected Directed Graph
--- 
> **Question:** Given a directed graph, find out whether the graph is strongly connected or not. A directed graph is strongly connected if there is a path between any two pair of vertices.

**Example:**
```py
is_SCDG(vertices=5, edges=[(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 2)])  # returns True
is_SCDG(vertices=4, edges=[(0, 1), (1, 2), (2, 3)])  # returns False
```

**My thoughts:** A directed graph being strongly connected indicates that for any vertex v, there exists a path to all other vertices and for all other vertices there exists a path to v. Here we can just pick any vertex from which we run DFS and see if it covers all vertices. And once done that, we can show v can go to all other vertices (excellent departure). Then we reverse the edges and run DFS from v again to test if v can be reach by all other vertices (excellent destination). 

Therefore, as v can be connected from any other vertices as well as can be reach by any other vertices. For any vertices u, w, we can simply connect them to v to reach each other. Thus, we can show such algorithm can test if a directed graph is strongly connected or not.

**Solution with DFS:** [https://replit.com/@trsong/Determine-if-Strongly-Connected-Directed-Graph](https://replit.com/@trsong/Determine-if-Strongly-Connected-Directed-Graph)
```py
import unittest

def is_SCDG(vertices, edges):
    # Neigbors relation in original graph
    neighbors = [None] * vertices

    # Neigbors in reverse graph
    reverse_neighbors = [None] * vertices

    for u, v in edges:
        neighbors[u] = neighbors[u] or []
        neighbors[u].append(v)

        reverse_neighbors[v] = reverse_neighbors[v] or []
        reverse_neighbors[v].append(u)

    start = 0
    return vertices == dfs_count_nodes(neighbors, start) == dfs_count_nodes(reverse_neighbors, start)


def dfs_count_nodes(neighbors, start):
    visited = set()
    stack = [start]
    
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)

        if not neighbors[cur]:
            continue
        for child in neighbors[cur]:
            if child not in visited:
                stack.append(child)
    return len(visited)


class IsSCDGSpec(unittest.TestCase):
    def test_unconnected_graph(self):
        self.assertFalse(is_SCDG(3, [(0, 1), (1, 0)]))
    
    def test_strongly_connected_graph(self):
        self.assertTrue(is_SCDG(5, [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 2)]))

    def test_not_strongly_connected_graph(self):
        self.assertFalse(is_SCDG(4, [(0, 1), (1, 2), (2, 3)]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 15, 2021 LC 392 \[Medium\] Is Subsequence
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

**Solution:** [https://replit.com/@trsong/Check-if-Subsequence](https://replit.com/@trsong/Check-if-Subsequence)
```py
import unittest

def is_subsequence(s, t):
    if not s:
        return True
    
    i = 0
    for ch in t:
        if i >= len(s):
            break
        if ch == s[i]:
            i += 1
    
    return i >= len(s)


class isSubsequenceSpec(unittest.TestCase):
    def test_empty_s(self):
        self.assertTrue(is_subsequence("", ""))

    def test_empty_s2(self):
        self.assertTrue(is_subsequence("", "a"))

    def test_empty_t(self):
        self.assertFalse(is_subsequence("a", ""))

    def test_s_longer_than_t(self):
        self.assertFalse(is_subsequence("ab", "a"))

    def test_size_one_input(self):
        self.assertTrue(is_subsequence("a", "a"))
    
    def test_size_one_input2(self):
        self.assertFalse(is_subsequence("a", "b"))

    def test_end_with_same_letter(self):
        self.assertTrue(is_subsequence("ab", "aaaaccb"))

    def test_example(self):
        self.assertTrue(is_subsequence("abc", "ahbgdc"))

    def test_example2(self):
        self.assertFalse(is_subsequence("axc", "ahbgdc"))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 14, 2021 \[Medium\] Subtree with Maximum Average
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

**Solution with Recursion:** [https://replit.com/@trsong/Find-Subtree-with-Maximum-Average](https://replit.com/@trsong/Find-Subtree-with-Maximum-Average)
```py
import unittest

def max_avg_subtree(tree):
    _, _, (_, accu_max_avg_node) = max_avg_subtree_recur(tree)
    return accu_max_avg_node


def max_avg_subtree_recur(tree):
    accu_sum, accu_count, accu_max_avg_and_node = tree.val, 1, (float('-inf'), tree.val)
    if not tree.children:
        return accu_sum, accu_count, accu_max_avg_and_node
    
    child_results = map(max_avg_subtree_recur, tree.children)
    for child_sum, child_count, child_max_avg_and_node in child_results:
        accu_sum += child_sum
        accu_count += child_count
        accu_max_avg_and_node = max(accu_max_avg_and_node, child_max_avg_and_node)
    return accu_sum, accu_count, max(accu_max_avg_and_node, (accu_sum / accu_count, tree.val))
        

class Node(object):
    def __init__(self, val, *children):
        self.val = val
        self.children = children


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
        self.assertEqual(18, max_avg_subtree(n20)) 

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
        self.assertEqual(11, max_avg_subtree(n1))

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
        self.assertEqual(0, max_avg_subtree(n0))

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
        self.assertEqual(2, max_avg_subtree(t))

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
        self.assertEqual(-2, max_avg_subtree(t))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 13, 2021 LC 722 \[Medium\] Remove Comments
---
> **Question:** Given a C/C++ program, remove comments from it.

**Example 1:**
```py
Input: 
source = ["/*Test program */", "int main()", "{ ", "  // variable declaration ", "int a, b, c;", "/* This is a test", "   multiline  ", "   comment for ", "   testing */", "a = b + c;", "}"]

Output: ["int main()","{ ","  ","int a, b, c;","a = b + c;","}"]
```

**Example 2:**
```py
Input: 
source = ["a/*comment", "line", "more_comment*/b"]

Output: ["ab"]
```

**Solution:** [https://replit.com/@trsong/Remove-C-Comments](https://replit.com/@trsong/Remove-C-Comments)
```py
import unittest

def remove_comments(source):
    res = []
    in_block = False
    for line in source:
        i = 0
        n = len(line)
        if not in_block:
            new_line = []
        while i < n:
            if not in_block and i < n - 1 and line[i] == '/' and line[i + 1] == '*':
                in_block = True
                i += 1
            elif in_block and i < n - 1 and line[i] == '*' and line[i + 1] == '/':
                in_block = False
                i += 1
            elif not in_block and i < n - 1 and line[i] == line[i + 1] == '/':
                break
            elif not in_block:
                new_line.append(line[i])
            i += 1
        
        if not in_block and new_line:
            res.append(''.join(new_line))
    return res


class RemoveCommentSpec(unittest.TestCase):
    def test_example1(self):
        source = ["/*Test program */", "int main()", "{ ", "  // variable declaration ", "int a, b, c;", "/* This is a test", "   multiline  ", "   comment for ", "   testing */", "a = b + c;", "}"]
        output = ["int main()","{ ","  ","int a, b, c;","a = b + c;","}"]
        self.assertEqual(output, remove_comments(source))

    def test_example2(self):
        source = ["a/*comment", "line", "more_comment*/b"]
        output = ["ab"]
        self.assertEqual(output, remove_comments(source))

    def test_empty_source(self):
        self.assertEqual([], remove_comments([]))
    
    def test_empty_source2(self):
        self.assertEqual([], remove_comments(["//"]))

    def test_empty_source3(self):
        self.assertEqual([], remove_comments(["/*","","*/"]))

    def test_block_comments_has_line_comment(self):
        source = ["return 1;", "/*", "function // ", "*/"]
        output = ["return 1;"]
        self.assertEqual(output, remove_comments(source))

    def test_multiple_block_comment_on_same_line(self):
        source = ["return 1 /*don't*/+ 2/*don't*//*don't*/ - 1; "]
        output = ["return 1 + 2 - 1; "]
        self.assertEqual(output, remove_comments(source))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 12, 2021 \[Easy\] Grid Path
---
> **Question:** You are given an M by N matrix consisting of booleans that represents a board. Each True boolean represents a wall. Each False boolean represents a tile you can walk on.
>
> Given this matrix, a start coordinate, and an end coordinate, return the minimum number of steps required to reach the end coordinate from the start. If there is no possible path, then return null. You can move up, left, down, and right. You cannot move through walls. You cannot wrap around the edges of the board.

**Example:**
```py
Given the following grid: 

[
  [F, F, F, F],
  [T, T, F, T],
  [F, F, F, F],
  [F, F, F, F]
]
and start = (3, 0) (bottom left) and end = (0, 0) (top left),

The minimum number of steps required to reach the end is 7, since we would need to go through (1, 2) because there is a wall everywhere else on the second row.
```

**Solution with BFS:** [https://replit.com/@trsong/Find-the-Min-Grid-Path-Distance](https://replit.com/@trsong/Find-the-Min-Grid-Path-Distance)
```py
import unittest

F = False
T = True
DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

def grid_path(grid_wall, start, end):
    if not grid_wall or not grid_wall[0]:
        return -1
    
    n, m = len(grid_wall), len(grid_wall[0])
    queue = [start]
    distance = 0
    visited = [[False for _ in range(m)] for _ in range(n)]

    while queue:
        for _ in range(len(queue)):
            r, c = queue.pop(0)
            if (r, c) == end:
                return distance

            if visited[r][c]:
                continue
            visited[r][c] = True

            for dr, dc in DIRECTIONS:
                new_r, new_c = r + dr, c + dc
                if (0 <= new_r < n and 
                    0 <= new_c < m and
                    not visited[new_r][new_c] and
                    not grid_wall[new_r][new_c]):
                    queue.append((new_r, new_c))
        distance += 1
    return -1
            

class GridPathSpec(unittest.TestCase):
    def test_one_row(self):
        grid_wall= [
            [F, F]
        ]
        start = (0, 0)
        end = (0, 1)
        self.assertEqual(1, grid_path(grid_wall, start, end))

    def test_example(self):
        grid_wall= [
            [F, F, F, F],
            [T, T, F, T],
            [F, F, F, F],
            [F, F, F, F]
        ]
        start = (0, 0)
        end = (3, 0)
        self.assertEqual(7, grid_path(grid_wall, start, end))

    def test_not_valid_path(self):
        grid_wall= [
            [F, F, F, F],
            [T, T, T, T],
            [F, F, F, F],
            [F, F, F, F]
        ]
        start = (0, 0)
        end = (3, 0)
        self.assertEqual(-1, grid_path(grid_wall, start, end))

    def test_large_grid(self):
        grid_wall= [
            [F, F, F, F, T, F, F, F],
            [T, T, T, F, T, F, T, T],
            [F, F, F, F, T, F, F, F],
            [F, T, T, T, T, T, T, F] ,
            [F, F, F, F, F, F, F, F]
        ]
        start = (0, 0)
        end = (0, 7)
        self.assertEqual(25, grid_path(grid_wall, start, end))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 11, 2021 LC 296 \[Hard\] Best Meeting Point
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


**Solution:** [https://replit.com/@trsong/Find-Best-Meeting-Point](https://replit.com/@trsong/Find-Best-Meeting-Point)
```py
import unittest

def best_meeting_distance(grid):
    if not grid or not grid[0]:
        return 0
    
    n, m = len(grid), len(grid[0])
    projected_row = [0] * m
    projected_col = [0] * n

    for r in range(n):
        for c in range(m):
            if grid[r][c]:
                projected_row[c] += 1
                projected_col[r] += 1
    return best_meeting_distance_1D(projected_row) + best_meeting_distance_1D(projected_col)


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


class BestMeetingPointSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(6, best_meeting_distance([
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ])) # best position at [0, 2]
    
    def test_1D_array(self):
        self.assertEqual(5, best_meeting_distance([
            [1, 0, 1, 0, 0, 1]
        ])) # best position at index 2
    
    def test_a_city_with_no_one(self):
        self.assertEqual(0, best_meeting_distance([
            [0, 0, 0],
            [0, 0, 0],
        ]))

    def test_empty_grid(self):
       self.assertEqual(0, best_meeting_distance([
            []
        ]))

    def test_even_number_of_points(self):
        self.assertEqual(17, best_meeting_distance([
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ])) # best distance = x-distance + y-distance = 8 + 9 = 17. Best position at [3, 2]

    def test_odd_number_of_points(self):
        self.assertEqual(24, best_meeting_distance([
            [1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0]
        ])) # best distance = x-distance + y-distance = 10 + 14 = 24. 


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 10, 2021 LC 317 \[Hard\] Shortest Distance from All Buildings
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


**Solution with DFS:** [https://replit.com/@trsong/Calculate-Shortest-Distance-from-All-Buildings](https://replit.com/@trsong/Calculate-Shortest-Distance-from-All-Buildings)
```py
import unittest

class CellType:
    LAND = 0
    BUILDING = 1
    OBSTACLE = 2


def shortest_distance(grid):
    if not grid or not grid[0]:
        return -1
    
    n, m = len(grid), len(grid[0])
    buildings = [(r, c) for r in range(n) for c in range(m) if grid[r][c] == CellType.BUILDING]

    if not buildings:
        return -1

    accu_distance = [[0 for _ in range(m)] for _ in range(n)]
    for building in buildings:
        num_building = dfs_building(grid, building, accu_distance)
        if num_building != len(buildings):
            return -1

    return min(accu_distance[r][c] for r in range(n) for c in range(m) if grid[r][c] == CellType.LAND)


DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

def dfs_building(grid, start_pos, accu_distance):
    n, m = len(grid), len(grid[0])
    num_building = 0
    distance = 0
    visited = [[False for _ in range(m)] for _ in range(n)]
    queue = [start_pos]
    
    while queue:
        for _ in range(len(queue)):
            r, c = queue.pop(0)
            if visited[r][c]:
                continue
            visited[r][c] = True
            accu_distance[r][c] += distance
            if grid[r][c] == CellType.BUILDING:
                num_building += 1

            for dr, dc in DIRECTIONS:
                new_r, new_c = r + dr, c + dc
                if (0 <= new_r < n and 
                    0 <= new_c < m and 
                    not visited[new_r][new_c] and 
                    grid[new_r][new_c] != CellType.OBSTACLE):
                    queue.append((new_r, new_c))
        distance += 1
    return num_building


class ShortestDistanceSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(7, shortest_distance([
            [1, 0, 2, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ]))  # target location is [1, 2] which has distance 3 + 3 + 1 = 7

    def test_inaccessible_buildings(self):
        self.assertEqual(-1, shortest_distance([
            [1, 0, 0],
            [1, 2, 2],
            [2, 1, 1]
        ]))

    def test_no_building_at_all(self):
        self.assertEqual(-1, shortest_distance([
            [0, 2, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]))

    def test_empty_grid(self):
        self.assertEqual(-1, shortest_distance([]))

    def test_building_on_same_line(self):
        self.assertEqual(6, shortest_distance([
            [2, 1, 0, 1, 0, 0, 1]  # target is at index 2, which has distance = 1 + 1 + 4
        ]))

    def test_multiple_road_same_distance(self):
        self.assertEqual(7, shortest_distance([
            [0, 1, 0],
            [1, 2, 0],
            [2, 1, 0]
        ]))  # either top-left or top-right will give 7


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 9, 2021 \[Easy\] Flip Bit to Get Longest Sequence of 1s
---
> **Question:** Given an integer, can you flip exactly one bit from a 0 to a 1 to get the longest sequence of 1s? Return the longest possible length of 1s after flip.

**Example:**
```py
Input: 183 (or binary: 10110111)
Output: 6
Explanation: 10110111 => 10111111. The longest sequence of 1s is of length 6.
```

**Solution:** [https://replit.com/@trsong/Flip-Bit-in-a-Binary-Number-to-Get-Longest-Sequence-of-1s](https://replit.com/@trsong/Flip-Bit-in-a-Binary-Number-to-Get-Longest-Sequence-of-1s)
```py
import unittest

def flip_bits(num):
    prev_bits = 0
    cur_bits = 0
    max_bits = 0

    while num > 0:
        if num & 1:
            cur_bits += 1
        else:
            max_bits = max(max_bits, cur_bits + prev_bits + 1)
            prev_bits = cur_bits
            cur_bits = 0
        num >>= 1
    return max(max_bits, cur_bits + prev_bits + 1)
            

class FlipBitSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(6, flip_bits(0b10110111))  # 10110111 => 10111111

    def test_not_exist_ones(self):
        self.assertEqual(1, flip_bits(0))  # 0 => 1

    def test_flip_last_digit(self):
        self.assertEqual(3, flip_bits(0b100110))  # 100110 => 100111

    def test_three_zeros(self):
        self.assertEqual(7, flip_bits(0b1011110110111))  # 1011110110111 => 1011111110111

    def test_one(self):
        self.assertEqual(2, flip_bits(1))  # 01 => 11


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 8, 2021 \[Hard\] Edit Distance
---
> **Question:**  The edit distance between two strings refers to the minimum number of character insertions, deletions, and substitutions required to change one string to the other. For example, the edit distance between “kitten” and “sitting” is three: substitute the “k” for “s”, substitute the “e” for “i”, and append a “g”.
> 
> Given two strings, compute the edit distance between them.


**Solution with Bottom-up DP:** [https://replit.com/@trsong/Calculate-Edit-Distance-with-Bottom-up-DP](https://replit.com/@trsong/Calculate-Edit-Distance-with-Bottom-up-DP)
```py
def edit_distance(source, target):
    n = len(source)
    m = len(target)

    # Let dp[n][m] represents edit distance between substring of source[:n] and target[:m]
    # dp[n][m] = dp[n - 1][m - 1]                                       if last char of both word match
    # or       = 1 + min(dp[n - 1][m], dp[n][m - 1], dp[n - 1][m - 1])  otherwise 
    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if source[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # insert, delete, and update
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
    return dp[n][m]
```

 **A little bit background information:** I first learnt this question in my 3rd year algorithem class. At that moment, this question was introduced to illustrate how dynamic programming works. And even nowadays, I can still recall the formula to be something like `dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + (0 if source[i] == target[j] else 1)`. However, when I ask my friends what dp represents in this question and how above formula works, few can give me convincing explanation. So the same thing could happen to readers like you: if you just know the formula without understanding which part represents insertion, removal or updating, then probably you just memorize the solution and pretend you understand the answer. And what could happen in the near future is that the next time when a similar question, like May 19 Regular expression, shows up during interview, you end up spending 20 min, got stuck trying to come up w/ dp formula.

**My thoughts:** My suggestion is that let's forget about dp at first, and we should just focus on recursion. (The idea between those two is kinda the same.) Just use the following template you learnt in first year of university to figure out the solution:

```py
def recursion(input):
    if ...base case...:
        return ...base case solution...
    else:
        do something to first of input 
        return recursion(rest of input)

def recursion2(input1, input2):
    if ...base case...:
        return ...base case solution...
    else:
        do something to first of input1
        do something to first of input2
        res1 = recursion2(rest of input1, input2)
        res2 = recursion2(input1, rest of input2)
        res3 = recursion2(rest of input1, rest of input2)
        return do something to res1, res2, res3

def recursion3(input1, input2):
    if ...base case...:
        return ...base case solution...
    else:
        do something to last of input1
        do something to last of input2
        res1 = recursion3(drop last of input1, input2)
        res2 = recursion3(input1, drop last of input2)
        res3 = recursion3(drop last of input1, drop last of input2)
        return do something to res1, res2, res3
```

Since we have two inputs for this question, we can either use recursion2 or recursion3 in above template, it doesn't really matter in this question. `def edit_distance(source, target)`. And now let's think about what is the base case. Base case is usually something trivial, like empty list. Then it's easy to know that if either source or target is empty, the edit distance is the other one's length. e.g. `edit_distance("", "kitten") == 6`

Then let's think about how we can shrink the size of source and target. In above template, there are 3 different ways to shrink the input size. Check the line res1, res2 and res3. 

1. Shrink source size: 

    I came up w/ some example like `edit_distance("ab", "a")` and `edit_distance("a", "a")`. Now think about the relationship between them. It turns out `edit_distance("ab", "a") == 1 + edit_distance("a", "a")`. As the minimum edit we need is just remove "b" from "ab" in source: only 1 extra edit.

2. Shrink target size:
   
    I came up w/ some example like `edit_distance("a", "ab")` and `edit_distance("a", "a")`. It turns out the mimum edit is to append "b" to the source "a" that gives "ab". So `edit_distance("a", "ab") == 1 + edit_distance("a", "a")`.

3. Shrink both source and target size: 

    I came up w/ some example like `edit_distance("aa", "ab")` and `edit_distance("a", "a")`. It seems we just need to replace "a" with "b" or "b" with "a". So only 1 more edit should be sufficient. However if we have `edit_distance("ab", "ab")` and `edit_distance("a", "a")`. Then that means we don't need to do anything when there is a match.
    That gives `edit_distance("aa", "ab") = 1 + edit_distance("a", "a")` and `edit_distance("ab", "ab") == edit_distance("a", "a")`

Note that any of res1, res2 and res3 might give the mimum edit. So we need to apply min to get the smallest among them.

**Solution with Recursion:** [https://replit.com/@trsong/Calculate-Edit-Distance](https://replit.com/@trsong/Calculate-Edit-Distance)
```py
import unittest

def edit_distance(source, target):
    if not source or not target:
        return max(len(source), len(target))
    # The edit_distance when insert/append an elem to source to match last elem in target
    insert_res = edit_distance(source, target[:-1]) + 1

    # The edit_distance when update last elem in source to match last elem in target
    update_res = edit_distance(source[:-1], target[:-1]) + (0 if source[-1] == target[-1] else 1)

    # The edit_distance when remove the last elem from source
    remove_res = edit_distance(source[:-1], target) + 1

    return min(insert_res, update_res, remove_res)


# If we modify the algorithem to insert/update/remove upon the first elem, it also works
def edit_distance2(source, target):
    if not source or not target:
        return max(len(source), len(target))
    # The edit_distance when insert/prepend an elem to source to match the first elem in target
    insert_res = edit_distance2(source, target[1:]) + 1

    # The edit_distance when update the first elem in source to match the first elem in target
    update_res = edit_distance2(source[1:], target[1:]) + (0 if source[0] == target[0] else 1)

    # The edit_distance when remove the first elem from source
    remove_res = edit_distance2(source[1:], target) + 1

    return min(insert_res, update_res, remove_res)
```

> **Note:** Above solution can be optimized using a cache. Or based on recursive formula, generate DP array. However, I feel too lazy for DP solution, you can probably google ***Levenshtein Distance*** or "edit distance". Here, I just give the optimized solution w/ cache. You can see how similar the dp solution vs optimziation w/ cache. And the benefit of using cache is that, you don't need to figure out the order to fill dp array as well as the initial value for dp array which is quite helpful. If you are curious about what question can give a weird dp filling order, just check May 19 question: Regular Expression and try to solve that w/ dp. 

**Solution with Top-down DP:** [https://replit.com/@trsong/Calculate-Edit-Distance-with-Cache](https://replit.com/@trsong/Calculate-Edit-Distance-with-Cache)
```py
def edit_distance(source, target):
    n, m = len(source), len(target)
    cache = [[None for _ in range(m)] for _ in range(n)]
    return edit_distance_helper_with_cache(source, target, 0, 0, cache)


# edit_distance(source[i:], target[j:])
def edit_distance_helper_with_cache(source, target, i, j, cache):
    n, m = len(source), len(target)
    if i > n - 1 or j > m - 1:
        return max(n - i, m - j)

    if cache[i][j] is None:
        # The edit_distance when insert/prepend an elem to source to match the first elem in target
        insert_res = edit_distance_helper_with_cache(source, target, i, j + 1, cache) + 1

        # The edit_distance when update the first elem in source to match the first elem in target
        update_res = edit_distance_helper_with_cache(source, target, i + 1, j + 1, cache) + (0 if source[i] == target[j] else 1)

        # The edit_distance when remove the first elem from source
        remove_res = edit_distance_helper_with_cache(source, target, i + 1, j, cache) + 1

        cache[i][j] = min(insert_res, update_res, remove_res)
    return cache[i][j]


class EditDistanceSpec(unittest.TestCase):
    def test_empty_strings(self):
        self.assertEqual(0, edit_distance("", ""))

    def test_empty_strings2(self):
        self.assertEqual(6, edit_distance("", "kitten"))

    def test_empty_strings3(self):
        self.assertEqual(7, edit_distance("sitting", ""))

    def test_non_empty_string(self):
        self.assertEqual(3, edit_distance("kitten", "sitting"))
    
    def test_non_empty_string2(self):
        self.assertEqual(3, edit_distance("sitting", "kitten"))


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
```


### Sep 7, 2021 LC 937 \[Easy\] Reorder Log Files
---
> **Question:** You have an array of logs. Each log is a space delimited string of words.
>
> For each log, the first word in each log is an alphanumeric identifier.  Then, either:
>
> - Each word after the identifier will consist only of lowercase letters, or;
> - Each word after the identifier will consist only of digits.
>
> We will call these two varieties of logs letter-logs and digit-logs.  It is guaranteed that each log has at least one word after its identifier.
>
> Reorder the logs so that all of the letter-logs come before any digit-log.  The letter-logs are ordered lexicographically ignoring identifier, with the identifier used in case of ties.  The digit-logs should be put in their original order.
>
> Return the final order of the logs.

**Example:**
```py
Input: ["a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo"]
Output: ["g1 act car","a8 act zoo","ab1 off key dog","a1 9 2 3 1","zo4 4 7"]
```

**Solution:** [https://replit.com/@trsong/Reorder-the-Log-Files](https://replit.com/@trsong/Reorder-the-Log-Files)
```py
import unittest

def reorder_log_files(logs):
    logs.sort(key=compute_log_key)
    return logs


def compute_log_key(log):
    tokens = log.split()
    is_digit_log = len(tokens) > 1 and tokens[1].isdigit()
    order_key1 = None if is_digit_log else tokens[1:]
    order_key2 = None if is_digit_log else tokens[0]
    return is_digit_log, order_key1, order_key2


class ReorderLogFileSpec(unittest.TestCase):
    dlog1 = "a1 9 2 3 1"
    dlog2 = "zo4 4 7"
    dlog3 = "a2 act car"
    llog1 = "g1 act car"
    llog2 = "ab1 off key dog"
    llog3 = "a8 act zoo"
    llog4 = "g1 act car jet jet jet"
    llog5 = "hhh1 act car"
    llog6 = "g1 act car jet"

    def assert_result(self, expected_order, original_order):
        self.assertEqual(str(expected_order), str(reorder_log_files(original_order)))

    def test_example(self):
        original_order = [self.dlog1, self.llog1, self.dlog2, self.llog2, self.llog3]
        expected_order = [self.llog1,self.llog3, self.llog2, self.dlog1, self.dlog2]
        self.assert_result(expected_order, reorder_log_files(original_order))

    def test_empty_logs(self):
        self.assert_result(reorder_log_files([]), [])

    def test_digit_logs_maintaining_same_order(self):
        original_order = [self.dlog1, self.dlog2]
        expected_order = [self.dlog1, self.dlog2]
        self.assert_result(expected_order, reorder_log_files(original_order))

    def test_digit_logs_maintaining_same_order2(self):
        original_order = [self.dlog2, self.dlog1, self.dlog2]
        expected_order = [self.dlog2, self.dlog1, self.dlog2]
        self.assert_result(expected_order, reorder_log_files(original_order))

    def test_when_letter_logs_have_a_tie(self):
        original_order = [self.llog6, self.llog4, self.llog1, self.llog5]
        expected_order = [self.llog1, self.llog5, self.llog6, self.llog4]
        self.assert_result(expected_order, reorder_log_files(original_order))
    
    def test_when_letter_logs_have_a_tie2(self):
        original_order = [self.dlog1, self.llog1, self.dlog2, self.llog2, self.llog3, self.dlog3]
        expected_order = [self.dlog3, self.llog1, self.llog3, self.llog2, self.dlog1, self.dlog2]
        self.assert_result(expected_order, reorder_log_files(original_order))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 6, 2021 LT 623 \[Hard\] K Edit Distance
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

**Solution with DP, Trie and DFS:** [https://replit.com/@trsong/All-Strings-within-K-Edit-Distance](https://replit.com/@trsong/All-Strings-within-K-Edit-Distance)
```py
import unittest

def filter_k_edit_distance(words, target, k):
    n = len(target)
    trie = Trie()
    filtered_words = filter(lambda word: n - k <= len(word) <= n + k, words)
    for word in filtered_words:
        trie.insert(word)
    
    # Let trie node dp[i] represents edit distance between prefix trie to target[:i]
    trie.dp = [i for i in range(n + 1)]
    stack = [trie]
    res = []
    while stack:
        cur = stack.pop()
        prev_dp = cur.dp
        if cur.count and prev_dp[-1] <= k:
            res.extend([cur.word] * cur.count)
        if not cur.children:
            continue
        
        for ch, child in cur.children.items():
            dp = [0] * (n + 1)
            dp[0] = prev_dp[0] + 1
            for i in range(1, n + 1):
                if ch == target[i - 1]:
                    dp[i] = prev_dp[i - 1]
                else:
                    dp[i] = 1 + min(dp[i - 1], prev_dp[i], prev_dp[i - 1])
            child.dp = dp
            stack.append(child)
    return res


class Trie(object):
    def __init__(self):
        self.count = 0
        self.dp = None
        self.children = None
        self.word = None

    def insert(self, word):
        p = self
        for ch in word:
            p.children = p.children or {}
            p.children[ch] = p.children.get(ch, Trie())
            p = p.children[ch]
        p.word = word
        p.count += 1


class FilterKEditDistanceSpec(unittest.TestCase):
    def assert_k_distance_array(self, res, expected):
        self.assertEqual(sorted(res), sorted(expected))

    def test_example1(self):
        words =["abc", "abd", "abcd", "adc"] 
        target = "ac"
        k = 1
        expected = ["abc", "adc"]
        self.assert_k_distance_array(expected, filter_k_edit_distance(words, target, k))
    
    def test_example2(self):
        words = ["acc","abcd","ade","abbcd"]
        target = "abc"
        k = 2
        expected = ["acc","abcd","ade","abbcd"]
        self.assert_k_distance_array(expected, filter_k_edit_distance(words, target, k))

    def test_duplicated_words(self):
        words = ["a","b","a","c", "bb", "cc"]
        target = ""
        k = 1
        expected = ["a","b","a","c"]
        self.assert_k_distance_array(expected, filter_k_edit_distance(words, target, k))

    def test_empty_words(self):
        words = ["", "", "", "c", "bbbbb", "cccc"]
        target = "ab"
        k = 2
        expected = ["", "", "", "c"]
        self.assert_k_distance_array(expected, filter_k_edit_distance(words, target, k))

    def test_same_word(self):
        words = ["ab", "ab", "ab"]
        target = "ab"
        k = 1000
        expected = ["ab", "ab", "ab"]
        self.assert_k_distance_array(expected, filter_k_edit_distance(words, target, k))

    def test_unqualified_words(self):
        words = ["", "a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa"]
        target = "aaaaa"
        k = 2
        expected = ["aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa"]
        self.assert_k_distance_array(expected, filter_k_edit_distance(words, target, k))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Sep 5, 2021 \[Easy\] GCD of N Numbers
---
> **Question:** Given `n` numbers, find the greatest common denominator between them.
>
> For example, given the numbers `[42, 56, 14]`, return `14`.

**Solution:** [https://replit.com/@trsong/Calculate-GCD-of-N-Numbers](https://replit.com/@trsong/Calculate-GCD-of-N-Numbers)
```py
import unittest

def gcd_n_numbers(nums):
    # gcd(a, b, c) = gcd(gcd(a, b), c)
    res = nums[0]
    for num in nums:
        res = gcd(res, num)
    return res


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


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
    unittest.main(exit=False, verbosity=2)
```

### Sep 4, 2021 \[Easy\] Sum Binary Numbers
---
> **Question:** Given two binary numbers represented as strings, return the sum of the two binary numbers as a new binary represented as a string. Do this without converting the whole binary string into an integer.

**Example:**
```py
sum_binary("11101", "1011")
# returns "101000"
```

**Solution:** [https://replit.com/@trsong/Add-Two-Binary-Numbers](https://replit.com/@trsong/Add-Two-Binary-Numbers)
```py
import unittest

def binary_sum(bin1, bin2):
    i = len(bin1) - 1
    j = len(bin2) - 1
    carry = 0
    res = []

    while i >= 0 or j >= 0 or carry:
        carry += 1 if i >= 0 and bin1[i] == '1' else 0
        carry += 1 if j >= 0 and bin2[j] == '1' else 0
        res.append(str(carry % 2))
        carry //= 2
        i -= 1
        j -= 1
    
    res.reverse()
    return ''.join(res)


class BinarySumSpec(unittest.TestCase):
    def test_example(self):
        bin1 = "11101"
        bin2 = "1011"
        expected = "101000"
        self.assertEqual(expected, binary_sum(bin1, bin2))

    def test_add_zero(self):
        bin1 = "0"
        bin2 = "0"
        expected = "0"
        self.assertEqual(expected, binary_sum(bin1, bin2))

    def test_should_not_overflow(self):
        bin1 = "1111111"
        bin2 = "1111111"
        expected = "11111110"
        self.assertEqual(expected, binary_sum(bin1, bin2))

    def test_calculate_carry_correctly(self):
        bin1 = "1111111"
        bin2 = "100"
        expected = "10000011"
        self.assertEqual(expected, binary_sum(bin1, bin2))
        
    def test_no_overlapping_digits(self):
        bin1 = "10100101"
        bin2 =  "1011010"
        expected = "11111111"
        self.assertEqual(expected, binary_sum(bin1, bin2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Sep 3, 2021 LC 134 \[Medium\] Gas Station
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

**Solution with Greedy Approach:** [https://replit.com/@trsong/Gas-Station-Problem](https://replit.com/@trsong/Gas-Station-Problem)
```py
import unittest

def valid_starting_station(gas, cost):
    if sum(gas) - sum(cost) < 0:
        return -1

    gas_level = 0
    start = 0
    for i in range(len(gas)):
        gas_level += gas[i] - cost[i]
        if gas_level < 0:
            start = (i + 1) % len(gas)
            gas_level = 0
    return start


class ValidStartingStationSpec(unittest.TestCase):
    def assert_valid_starting_station(self, gas, cost, start):
        gas_level = 0
        n = len(gas)
        for i in range(n):
            shifted_index = (start + i) % n
            gas_level += gas[shifted_index] - cost[shifted_index]
            self.assertTrue(gas_level >= 0)

    def test_example(self):
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
    unittest.main(exit=False, verbosity=2)
```


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
