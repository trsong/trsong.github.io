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




### Jan 9, 2022 LC 138 \[Medium\] Deepcopy List with Random Pointer
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


### Jan 8, 2022 \[Medium\] All Root to Leaf Paths in Binary Tree
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

**Solution with Backtracking:** [https://replit.com/@trsong/Print-All-Root-to-Leaf-Paths-in-Binary-Tree-2](https://replit.com/@trsong/Print-All-Root-to-Leaf-Paths-in-Binary-Tree-2)
```py
import unittest

def path_to_leaves(tree):
    if not tree:
        return []

    res = []
    backtrack(res, [], tree)
    return res


def backtrack(res, accu_path, node):
    if not node.left and not node.right:
        res.append(accu_path + [node.val])
    else:
        accu_path.append(node.val)
        for child in [node.left, node.right]:
            if not child:
                continue
            backtrack(res, accu_path, child)
        accu_path.pop()
    
    
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
    unittest.main(exit=False, verbosity=2)
```


### Jan 7, 2022 \[Medium\] Second Largest in BST
---
> **Question:** Given the root to a binary search tree, find the second largest node in the tree.

**My thoughts:**Recall the way we figure out the largest element in BST: we go all the way to the right until not possible. So the second largest element must be on the left of largest element. We have two possibilities here:
- Either it's the parent of rightmost element, if there is no child underneath
- Or it's the rightmost element in left subtree of the rightmost element. ie. 2nd rightmost

```py
Case 1: Parent
1
 \ 
  2*
   \
    3

Case 2: 2nd rightmost 
1
 \
  4
 /
2
 \ 
  3* 
```


**Solution:** [https://replit.com/@trsong/Find-Second-Largest-in-BST-2](https://replit.com/@trsong/Find-Second-Largest-in-BST-2)
```py
import unittest

def bst_2nd_max(node):
    max_node, max_node_parent = find_right_most_and_parent(node)
    if not max_node or not max_node.left:
        return max_node_parent
    
    second_max_node, _ = find_right_most_and_parent(max_node.left)
    return second_max_node


def find_right_most_and_parent(root):
    prev = None
    p = root
    while p and p.right:
        prev = p
        p = p.right
    return p, prev


###################
# Testing Utilities
###################
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
                stack.append((child, depth+1))
        return "\n" + "".join(res) + "\n"


class Bst2ndMaxSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertIsNone(bst_2nd_max(None))

    def test_one_element_tree(self):
        root = TreeNode(1)
        self.assertIsNone(bst_2nd_max(root))

    def test_left_heavy_tree(self):
        """
          2
         /
        1*
        """
        left_tree = TreeNode(1)
        root = TreeNode(2, left_tree)
        self.assertEqual(left_tree, bst_2nd_max(root))

    def test_left_heavy_tree2(self):
        """
          3
         /
        1
         \
          2*
        """
        leaf = TreeNode(2)
        root = TreeNode(3, TreeNode(1, right=leaf))
        self.assertEqual(leaf, bst_2nd_max(root))

    def test_balanced_tree(self):
        """
          2*
         / \
        1   3
        """
        root = TreeNode(2, TreeNode(1), TreeNode(3))
        self.assertEqual(root, bst_2nd_max(root))

    def test_balanced_tree2(self):
        """
             4
           /   \
          2     6*
         / \   / \ 
        1   3 5   7 
        """
        left_tree = TreeNode(2, TreeNode(1), TreeNode(3))
        right_tree = TreeNode(6, TreeNode(5), TreeNode(7))
        root = TreeNode(4, left_tree, right_tree)
        self.assertEqual(right_tree, bst_2nd_max(root))

    def test_right_heavy_tree(self):
        """
        1
         \
          2*
           \
            3
        """
        right_tree = TreeNode(2, right=TreeNode(3))
        root = TreeNode(1, right=right_tree)
        self.assertEqual(right_tree, bst_2nd_max(root))

    def test_unbalanced_tree(self):
        """
        1
         \
          4
         /
        2
         \
          3*
        """
        leaf = TreeNode(3)
        root = TreeNode(1, right=TreeNode(4, TreeNode(2, right=leaf)))
        self.assertEqual(leaf, bst_2nd_max(root))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 6, 2022 LC 279 \[Medium\] Minimum Number of Squares Sum to N
---
> **Question:** Given a positive integer n, find the smallest number of squared integers which sum to n.
>
> For example, given `n = 13`, return `2` since `13 = 3^2 + 2^2 = 9 + 4`.
> 
> Given `n = 27`, return `3` since `27 = 3^2 + 3^2 + 3^2 = 9 + 9 + 9`.

**Solution with DP:** [https://replit.com/@trsong/Minimum-Squares-Sum-to-N-3](https://replit.com/@trsong/Minimum-Squares-Sum-to-N-3)
```py
import unittest

def min_square_sum(n):
    # Let dp[n] represents min number of square sum to n
    # dp[n] = 1 + min{ dp[n - x * x] for all x st. n >= x * x }
    
    dp = [float('inf')] * (n + 1)
    dp[0] = 0

    for i in range(1, n + 1):
        for x in range(i):
            sqr_x = x * x
            if sqr_x > i:
                continue
            dp[i] = min(dp[i], 1 + dp[i - sqr_x])
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


### Jan 5, 2022 LC 668 \[Hard\] Kth Smallest Number in Multiplication Table
---
> **Question:** Find out the k-th smallest number quickly from the multiplication table.
>
> Given the height m and the length n of a m * n Multiplication Table, and a positive integer k, you need to return the k-th smallest number in this table.

**Example 1:**
```py
Input: m = 3, n = 3, k = 5
Output: 
Explanation: 
The Multiplication Table:
1	2	3
2	4	6
3	6	9
The 5-th smallest number is 3 (1, 2, 2, 3, 3).
```

**Example 2:**
```py
Input: m = 2, n = 3, k = 6
Output: 
Explanation: 
The Multiplication Table:
1	2	3
2	4	6
The 6-th smallest number is 6 (1, 2, 2, 3, 4, 6).
```


**Solution with Binary Search:** [https://replit.com/@trsong/Kth-Smallest-Number-in-Multiplication-Table-2](https://replit.com/@trsong/Kth-Smallest-Number-in-Multiplication-Table-2)
```py
import unittest

def find_kth_num(m, n, k):
    lo = 1
    hi = m * n
    while lo < hi:
        mid = lo + (hi - lo) // 2
        num_smaller = count_smaller(m, n, mid)
        if num_smaller < k:
            lo = mid + 1
        else:
            hi = mid
    return lo


def count_smaller(n, m, target):
    m, n = min(n, m), max(n, m)
    res = 0
    for r in range(1, min(m, target) + 1):
        # The r-th row cannot have more than n elements
        res += min(n, target // r)
    return res


class FindKthNumSpec(unittest.TestCase):
    def test_example(self):
        """
        1	2	3
        2	4	6
        3	6	9
        """
        m, n, k = 3, 3, 5
        self.assertEqual(3, find_kth_num(m, n, k))

    def test_example2(self):
        """
        1	2	3
        2	4	6
        """
        m, n, k = 2, 3, 6
        self.assertEqual(6, find_kth_num(m, n, k))

    def test_single_row(self):
        """
        1	2	3   4   5
        """
        m, n, k = 1, 5, 5
        self.assertEqual(5, find_kth_num(m, n, k))

    def test_single_column(self):
        """
        1
        2
        3
        """
        m, n, k = 3, 1, 2
        self.assertEqual(2, find_kth_num(m, n, k))

    def test_single_cell(self):
        """
        1
        """
        m, n, k = 1, 1, 1
        self.assertEqual(1, find_kth_num(m, n, k))

    def test_large_table(self):
        m, n, k = 42, 34, 401
        self.assertEqual(126, find_kth_num(m, n, k))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Jan 4, 2022 \[Medium\] Reverse Coin Change
---
> **Question:** You are given an array of length `N`, where each element `i` represents the number of ways we can produce `i` units of change. For example, `[1, 0, 1, 1, 2]` would indicate that there is only one way to make `0, 2, or 3` units, and two ways of making `4` units.
>
> Given such an array, determine the denominations that must be in use. In the case above, for example, there must be coins with value `2, 3, and 4`.

**Solution with DP:** [https://replit.com/@trsong/Reverse-Coin-Change-2](https://replit.com/@trsong/Reverse-Coin-Change-2)
```py
import unittest

def reverse_coin_change(coin_ways):
    # coin_ways is originally generated like the following
    # coin_ways[n] += coin_ways[n - b] where b <= n for b in base 
    base = []
    n = len(coin_ways)
    for coin in range(1, n):
        if coin_ways[coin] == 0:
            continue
        base.append(coin)

        for i in range(n - 1, coin - 1, -1):
            # work backwards to cancel base's effect
            coin_ways[i] -= coin_ways[i - coin]
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


### Jan 3, 2022  \[Medium\] Majority Element
---
> **Question:** A majority element is an element that appears more than half the time. Given a list with a majority element, find the majority element.

**Example:**
```py
majority_element([3, 5, 3, 3, 2, 4, 3])  # gives 3
```

**Solution:** [https://replit.com/@trsong/Find-Majority-Element-2](https://replit.com/@trsong/Find-Majority-Element-2)
```py
import unittest

def majority_element(nums):
    count = 0
    candidate = 0

    for num in nums:
        if count == 0:
            candidate = num
        
        if candidate == num:
            count += 1
        else:
            count -= 1
    return candidate if validate(nums, candidate) else None
    
    
def validate(nums, major_elem):
    count = 0
    for num in nums:
        count += 1 if num == major_elem else 0
    return count + count > len(nums)


class MajorityElementSpec(unittest.TestCase):
    def test_no_majority_element_exists(self):
        self.assertIsNone(majority_element([1, 2, 3, 4]))

    def test_example(self):
        self.assertEqual(3, majority_element([3, 5, 3, 3, 2, 4, 3]))

    def test_example2(self):
        self.assertEqual(3, majority_element([3, 2, 3]))

    def test_example3(self):
        self.assertEqual(2, majority_element([2, 2, 1, 1, 1, 2, 2]))

    def test_there_is_a_tie(self):
        self.assertIsNone(majority_element([1, 2, 1, 2, 1, 2]))

    def test_majority_on_second_half_of_list(self):
        self.assertEqual(1, majority_element([2, 2, 1, 2, 1, 1, 1]))
    
    def test_more_than_two_kinds(self):
        self.assertEqual(1, majority_element([1, 2, 1, 1, 2, 2, 1, 3, 1, 1, 1]))

    def test_zero_is_the_majority_element(self):
        self.assertEqual(0, majority_element([0, 1, 0, 1, 0, 1, 0]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 2, 2022 LC 685 \[Hard\] Redundant Connection II
---
> **Questions:** In this problem, a rooted tree is a directed graph such that, there is exactly one node (the root) for which all other nodes are descendants of this node, plus every node has exactly one parent, except for the root node which has no parents.
>
> The given input is a **directed** graph that started as a rooted tree with N nodes (with distinct values `1, 2, ..., N`), with one additional directed edge added. The added edge has two different vertices chosen from 1 to N, and was not an edge that already existed.
>
> The resulting graph is given as a 2D-array of edges. Each element of edges is a pair `[u, v]` that represents a directed edge connecting nodes `u` and `v`, where `u` is a parent of child `v`.
>
> Return an edge that can be removed so that the resulting graph is a rooted tree of `N` nodes. If there are multiple answers, return the answer that occurs last in the given 2D-array.

**Example 1:**
```py
Input: [[1,2], [1,3], [2,3]]
Output: [2,3]
Explanation: The given directed graph will be like this:
  1
 / \
v   v
2-->3
```

**Example 2:**
```py
Input: [[1,2], [2,3], [3,4], [4,1], [1,5]]
Output: [4,1]
Explanation: The given directed graph will be like this:
5 <- 1 -> 2
     ^    |
     |    v
     4 <- 3
```

**My thoughts:** Based on definition of rooted tree: all node have one parent except root that has no parent. 

The redundant edge must either cause some node to have 2 parents or connect to root and make it no longer a root. 

For the first case, we can assume either edge is redundant and remove it. Yet that creates two separate outcomes: assumption is correct (no cycle after) or assumption is incorrect (cycle still exists), then we choose the other as answer. 

For the second case, we need to figure out which edge that firsts introduces a cycle. 


**Solution with UnionFind:** [https://replit.com/@trsong/Redundant-Connection-II](https://replit.com/@trsong/Redundant-Connection-II)
```py
import unittest

def find_redundant_connection(edges):
    candidate1, candidate2 = find_conflict_edges(edges)
    parent = {}

    for u, v in edges:
        # assume candidate2 is redundant
        if [u, v] == candidate2:
            continue
        
        # test if (u, v) create a cycle, if so they assumption of candidate 2 is incorrect
        parent[v] = find_and_update_root(parent, u)
        if parent[v] == v:
            return candidate1 or [u, v]
        
    # assumption is correct
    return candidate2


def find_conflict_edges(edges):
    parent = {}
    for u, v in edges:
        if v in parent:
            return [parent[v], v], [u, v]
        parent[v] = u
    return None, None


def find_and_update_root(parent, v):
    parent[v] = parent.get(v, v)
    if parent[v] != v:
        parent[v] = find_and_update_root(parent, parent[v])
        v = parent[v]
    return v


class FindRedundantConnectionSpec(unittest.TestCase):
    def test_example(self):
        """
          1
         / \
        v   v
        2-->3
        """
        edges = [[1, 2], [1, 3], [2, 3]]
        expected = [2, 3]
        self.assertEqual(expected, find_redundant_connection(edges))

    def test_example2(self):
        """
        5 <- 1 -> 2
             ^    |
             |    v
             4 <- 3
        """
        edges = [[1, 2], [2, 3], [3, 4], [4, 1], [1, 5]]
        expected = [4, 1]
        self.assertEqual(expected, find_redundant_connection(edges))

    def test_graph_with_cycle(self):
        """
        1 <-> 2
        """
        edges = [[1, 2], [2, 1]]
        expected = [2, 1]
        self.assertEqual(expected, find_redundant_connection(edges))

    def test_graph_with_cycle2(self):
        """
        1 -> 3 
        | ^
        v   \
        2 -> 4
        """
        edges = [[1, 2], [1, 3], [4, 1], [2, 4]]
        expected = [2, 4]
        self.assertEqual(expected, find_redundant_connection(edges))

    def test_graph_with_cycle3(self):
        """
        1 -> 6
        |
        v
        2 <- 5
        |    ^
        v    |
        3 -> 4
        """
        edges = [[1, 2], [1, 6], [5, 2], [2, 3], [3, 4], [4, 5]]
        expected = [5, 2]
        self.assertEqual(expected, find_redundant_connection(edges))

    def test_graph_with_cycle4(self):
        """
        5 -> 1 -> 2
             ^    |
             |    v
             4 <- 3
        """
        edges = [[3, 4], [4, 1], [2, 3], [1, 2], [5, 1]]
        expected = [4, 1]
        self.assertEqual(expected, find_redundant_connection(edges))

    def test_graph_with_cycle5(self):
        """
        1  -> 5 -> 3
              |  
              v   
        4 <-> 2
        """
        edges = [[4, 2], [1, 5], [5, 2], [5, 3], [2, 4]]
        expected = [4, 2]
        self.assertEqual(expected, find_redundant_connection(edges))

    def test_graph_without_cycle(self):
        """
        1 -> 3 
        |    ^
        v    |
        2 -> 4
        """
        edges = [[1, 2], [1, 3], [2, 4], [4, 3]]
        expected = [4, 3]
        self.assertEqual(expected, find_redundant_connection(edges))

    def test_large_graph(self):
        edges = [[37, 30], [21, 34], [10, 40], [8, 36], [18, 10], [50, 11],
                 [13, 6], [40, 7], [14, 38], [41, 24], [32, 17], [31, 15],
                 [6, 27], [45, 3], [30, 42], [43, 26], [9, 4], [4, 31],
                 [1, 29], [5, 23], [44, 19], [15, 44], [49, 20], [26, 5],
                 [23, 50], [48, 41], [47, 22], [3, 46], [11, 16], [12, 35],
                 [33, 50], [34, 45], [38, 2], [2, 32], [24, 49], [35, 37],
                 [29, 13], [46, 48], [28, 12], [7, 21], [27, 18], [17, 39],
                 [42, 14], [20, 47], [36, 1], [22, 9], [25, 8], [39, 25],
                 [16, 28], [19, 43]]
        expected = [23, 50]
        self.assertEqual(expected, find_redundant_connection(edges))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Jan 1, 2022 LC 772 \[Hard\] Basic Calculator III
---
> **Questions:** Implement a basic calculator to evaluate a simple expression string.
>
> The expression string contains integers, `+`, `-`, `*`, `/` operators , open `(` and closing parentheses `)` and empty spaces. The integer division should truncate toward zero.

**Example 1:**
```py
Input: "1 + 1"
Output: 2
```

**Example 2:**
```py
Input: " 6-4 / 2 " 
Output: 4
```

**Example 3:**
```py
Input: "2*(5+5*2)/3+(6/2+8)" 
Output: 21
```

**Example 4:**
```py
Input: "(2+6* 3+5- (3*14/7+2)*5)+3"
Output: -12
```

**My thoughts:** First, think about how to deal with parentheses. 

- We can treat parentheses as function call. 
- `(` starts recursive call. 
- `)` ends call and returns. 
- And finally treat the result of parentheses expression as a single number. 

Once parentheses are properly handled, we can think about priority of `+-*/`:

we can use the following definiton as a guidance:

```py
An Expr is one of
* Term
* Term + Expr
* Term - Expr   # You can think about it as Term + (-Expr), or in another way Term + (-1) * Expr

A Term is one of 
* Number
* Number * Term
* Number / Term 
```

We can break an Expr into a stack of Term's. `1 + 2 * 3 - 4 * 5 / 6` => `[1, 2 * 3,   (-4) * 5 / 6]`.
And evaluation of expression is like `eval(Expr) => sum(stack)`

Keep in mind that `+-` creates a new term, yet `*/` updates last term. 
`Term1 - Term2` is exactly the same as `Term1 + (-Term2)` or `Term1 + (-1) * Term2`


**Solution with Stack:** [https://replit.com/@trsong/Basic-Calculator-III](https://replit.com/@trsong/Basic-Calculator-III)
```py
import unittest

def evaluate_expression(expr):
    return eval_stream(iter(expr))


def eval_stream(ch_stream):
    stack = []
    last_sign = '+'
    last_num = 0
        
    for ch in ch_stream:
        if '0' <= ch <= '9':
            last_num = 10 * last_num + int(ch)
        elif ch in '+-*/':
            udpate_stack(stack, last_sign, last_num)
            last_sign = ch
            last_num = 0
        elif ch == '(':
            last_num = eval_stream(ch_stream)
        elif ch == ')':
            break
    
    # do not forget EOF
    udpate_stack(stack, last_sign, last_num)
    return sum(stack)


def udpate_stack(stack, op, num):
    if op == '+':
        stack.append(num)
    elif op == '-':
        stack.append(-num)
    elif op == '*':
        stack.append(stack.pop() * num)
    elif op == '/':
        stack.append(int(stack.pop() / num))


class EvaluateExpressionSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(2, evaluate_expression("1 + 1"))

    def test_example2(self):
        self.assertEqual(4, evaluate_expression(" 6-4 / 2 "))

    def test_example3(self):
        self.assertEqual(21, evaluate_expression("2*(5+5*2)/3+(6/2+8)"))

    def test_example4(self):
        self.assertEqual(-12,
                         evaluate_expression("(2+6* 3+5- (3*14/7+2)*5)+3"))

    def test_simple_expression_with_parentheses(self):
        self.assertEqual(4, evaluate_expression("-1 + (2 + 3)"))

    def test_empty_string(self):
        self.assertEqual(0, evaluate_expression(""))

    def test_basic_expression1(self):
        self.assertEqual(7, evaluate_expression("3+2*2"))

    def test_basic_expression2(self):
        self.assertEqual(1, evaluate_expression(" 3/2 "))

    def test_basic_expression3(self):
        self.assertEqual(5, evaluate_expression(" 3+5 / 2 "))

    def test_basic_expression4(self):
        self.assertEqual(-24, evaluate_expression("1*2-3/4+5*6-7*8+9/10"))

    def test_basic_expression5(self):
        self.assertEqual(10000, evaluate_expression("10000-1000/10+100*1"))

    def test_basic_expression6(self):
        self.assertEqual(13, evaluate_expression("14-3/2"))

    def test_negative(self):
        self.assertEqual(-1, evaluate_expression(" -7 / 4 "))

    def test_minus(self):
        self.assertEqual(-5, evaluate_expression("-2-3"))

    def test_expression_with_parentheses(self):
        self.assertEqual(42, evaluate_expression("  -(-42)"))

    def test_expression_with_parentheses2(self):
        self.assertEqual(3, evaluate_expression("(-1 + 2) * 3"))

    def test_complicated_operations(self):
        self.assertEqual(
            2,
            evaluate_expression("-2 - (-2) * ( ((-2) - 3) * 2 + (-3) * (-4))"))

    def test_complicated_operations2(self):
        self.assertEqual(-2600,
                         evaluate_expression("-3*(10000-1000)/10-100*(-1)"))

    def test_complicated_operations3(self):
        self.assertEqual(100, evaluate_expression("100 * ( 2 + 12 ) / 14"))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 31, 2021 \[Easy\] Longest Consecutive 1s in Binary Representation
---
> **Question:**  Given an integer n, return the length of the longest consecutive run of 1s in its binary representation.

**Example:**
```py
Input: 156
Output: 3
Exaplanation: 156 in binary is 10011100
```

**Solution:** [https://replit.com/@trsong/Longest-Consecutive-1s-in-Binary-Representation-2](https://replit.com/@trsong/Longest-Consecutive-1s-in-Binary-Representation-2)
```py
import unittest

def count_longest_consecutive_ones(num):
    res = 0
    while num:
        res += 1
        # in each consecutive block of 1s, remove trailing zero in parallel
        #    001100111001
        # => 001000110000
        num &= (num << 1)
    return res


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
    unittest.main(exit=False, verbosity=2)
```


### Dec 30, 2021 \[Medium\] Generate Brackets
---
> **Question:** Given a number n, generate all possible combinations of n well-formed brackets.

**Example 1:**
```py
generate_brackets(1)  # returns ['()']
```

**Example 2:**
```py
generate_brackets(3)  # returns ['((()))', '(()())', '()(())', '()()()', '(())()']
```


**Solution with Backtracking:** [https://replit.com/@trsong/Generate-Well-formed-Brackets-2](https://replit.com/@trsong/Generate-Well-formed-Brackets-2)
```py
import unittest

def generate_brackets(n):
    if n == 0:
        return []
        
    res = []
    backtrack(res, [], n, n)
    return res


def backtrack(res, accu, num_open, num_close):
    if num_open == num_close == 0:
        res.append(''.join(accu))
    else:
        if num_open > 0:
            accu.append('(')
            backtrack(res, accu, num_open - 1, num_close)
            accu.pop()

        if num_open < num_close:
            accu.append(')')
            backtrack(res, accu, num_open, num_close - 1)
            accu.pop()


class GenerateBracketSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(sorted(expected), sorted(result))

    def test_example1(self):
        self.assert_result(['()'], generate_brackets(1))

    def test_example2(self):
        n, expected = 3, ['((()))', '(()())', '()(())', '()()()', '(())()']
        self.assert_result(expected, generate_brackets(n))

    def test_input_size_is_two(self):
        n, expected = 2, ['()()', '(())']
        self.assert_result(expected, generate_brackets(n))

    def test_input_size_is_zero(self):
        self.assert_result([], generate_brackets(0))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 29, 2021 \[Medium\] Number of Android Lock Patterns
---
> **Question:** One way to unlock an Android phone is through a pattern of swipes across a 1-9 keypad.
>
> For a pattern to be valid, it must satisfy the following:
>
> - All of its keys must be distinct.
> - It must not connect two keys by jumping over a third key, unless that key has already been used.
>
> For example, 4 - 2 - 1 - 7 is a valid pattern, whereas 2 - 1 - 7 is not.
>
> Find the total number of valid unlock patterns of length N, where 1 <= N <= 9.

**My thoughts:** By symmetricity, code starts with 1, 3, 7, 9 has same number of total combination and same as 2, 4, 6, 8. Thus we only need to figure out total combinations starts from 1, 2 and 5. We can count that number through DFS with Backtracking and make sure to check if there is no illegal jump between recursive calls.

**Solution with Backtracking:** [https://replit.com/@trsong/Count-Number-of-Android-Lock-Patterns-2](https://replit.com/@trsong/Count-Number-of-Android-Lock-Patterns-2)
```py
import unittest

def android_lock_combinations(code_len):
    """
    1 2 3
    4 5 6
    7 8 9
    """
    overlap = {
        2: [(1, 3)],
        4: [(1, 7)],
        8: [(7, 9)],
        6: [(3, 9)],
        5: [(1, 9), (2, 8), (3, 7), (4, 6)]
    }
    overlap_lookup = {(start, end): k
                      for k in overlap
                      for start, end in overlap[k] + map(reversed, overlap[k])}

    visited = [False] * 10
    start_from_one = backtrack(1, code_len - 1, visited, overlap_lookup)
    start_from_two = backtrack(2, code_len - 1, visited, overlap_lookup)
    start_from_five = backtrack(5, code_len - 1, visited, overlap_lookup)
    return 4 * (start_from_one + start_from_two) + start_from_five


def backtrack(start, code_len, visited, overlap_lookup):
    if code_len == 0:
        return 1
    else:
        visited[start] = True
        res = 0
        for next in range(1, 10):
            overlap = overlap_lookup.get((start, next))
            overlap_visited = overlap is None or visited[overlap]
            if not visited[next] and overlap_visited:
                res += backtrack(next, code_len - 1, visited, overlap_lookup)
        visited[start] = False
        return res


class AndroidLockCombinationSpec(unittest.TestCase):
    def test_length_1_code(self):
        self.assertEqual(9, android_lock_combinations(1))

    def test_length_2_code(self):
        # 1-2, 1-4, 1-5, 1-6, 1-8
        # 2-1, 2-3, 2-4, 2-5, 2-6, 2-7, 2-9
        # 5-1, 5-2, 5-3, 5-4, 5-6, 5-7, 5-8, 5-9
        # Due to symmetricity, code starts with 3, 7, 9 has same number as 1
        #                      code starts with 4, 6, 8 has same number as 2
        # Total = 5*4 + 7*4 + 8 = 56
        self.assertEqual(56, android_lock_combinations(2))

    def test_length_3_code(self):
        self.assertEqual(320, android_lock_combinations(3))

    def test_length_4_code(self):
        self.assertEqual(1624, android_lock_combinations(4))

    def test_length_5_code(self):
        self.assertEqual(7152, android_lock_combinations(5))

    def test_length_6_code(self):
        self.assertEqual(26016, android_lock_combinations(6))

    def test_length_7_code(self):
        self.assertEqual(72912, android_lock_combinations(7))

    def test_length_8_code(self):
        self.assertEqual(140704, android_lock_combinations(8))

    def test_length_9_code(self):
        self.assertEqual(140704, android_lock_combinations(9))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 28, 2021 LC 684 \[Medium\] Redundant Connection
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

**Solution with DisjointSet(Union-Find):** [https://replit.com/@trsong/Redundant-Connection](https://replit.com/@trsong/Redundant-Connection)
```py
import unittest

def find_redundant_connection(edges):
    uf = DisjointSet()
    for u, v in edges:
        if uf.is_connected(u, v):
            return [u, v]
        else:
            uf.union(u, v)
    return None


class DisjointSet(object):
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, p):
        self.parent[p] = self.parent.get(p, p)
        while self.parent[p] != p:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def union(self, p1, p2):
        r1 = self.find(p1)
        r2 = self.find(p2)
        if r1 != r2:
            if self.rank.get(r1, 0) < self.rank.get(r2, 0):
                self.parent[r1] = r2
                self.rank[r2] = self.rank.get(r2, 0) + 1
            else:
                self.parent[r2] = r1
                self.rank[r1] = self.rank.get(r1, 0) + 1

    def is_connected(self, p1, p2):
        return self.find(p1) == self.find(p2)


class FindRedundantConnectionSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual([2, 3],
                         find_redundant_connection([[1, 2], [1, 3], [2, 3]]))

    def test_example2(self):
        self.assertEqual([1, 4],
                         find_redundant_connection([[1, 2], [2, 3], [3, 4],
                                                    [1, 4], [1, 5]]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 27, 2021 LT 434 \[Medium\] Number of Islands II
---
> **Question:** Given a `n,m` which means the row and column of the 2D matrix and an array of pair `A( size k)`. Originally, the 2D matrix is all 0 which means there is only sea in the matrix. The list pair has `k` operator and each operator has two integer `A[i].x, A[i].y` means that you can change the grid matrix`[A[i].x][A[i].y]` from sea to island. 
> 
> Return how many island are there in the matrix after each operator. You need to return an array of size `K`.

**Example 1:**
```py
Input: n = 4, m = 5, A = [[1,1],[0,1],[3,3],[3,4]]
Output: [1,1,2,2]
Explanation:
0.  00000
    00000
    00000
    00000
1.  00000
    01000
    00000
    00000
2.  01000
    01000
    00000
    00000
3.  01000
    01000
    00000
    00010
4.  01000
    01000
    00000
    00011
```

**Example 2:**
```py
Input: n = 3, m = 3, A = [[0,0],[0,1],[2,2],[2,1]]
Output: [1,1,2,2]
```

**Solution with DisjointSet(Union-Find):** [https://replit.com/@trsong/Number-of-Islands-II](https://replit.com/@trsong/Number-of-Islands-II)
```py
import unittest

DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

def calculate_islands(n, m, island_positions):
    if n <= 0 or m <= 0 or not island_positions:
        return []
    island_ds = DisjointSet()
    res = []
    for r, c in island_positions:
        island_ds.find((r, c))
        for dr, dc in DIRECTIONS:
            new_r, new_c = r + dr, c + dc
            if (0 <= new_r < n and 
                0 <= new_c < m and 
                (new_r, new_c) in island_ds):
                island_ds.union((r, c), (new_r, new_c))
        res.append(island_ds.cardinal())
    return res
        

class DisjointSet(object):
    def __init__(self):
        self.parent = {}
    
    def __contains__(self, p):
        return p in self.parent

    def find(self, p):
        self.parent[p] = self.parent.get(p, p)
        while self.parent.get(p, p) != p:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def union(self, p1, p2):
        r1 = self.find(p1)
        r2 = self.find(p2)
        if r1 != r2:
            self.parent[r1] = r2
    
    def cardinal(self):
        return len(set(map(self.find, self.parent)))


class CalculateIslandSpec(unittest.TestCase):
    def test_example(self):
        """
        0.  00000
            00000
            00000
            00000
        1.  00000
            01000
            00000
            00000
        2.  01000
            01000
            00000
            00000
        3.  01000
            01000
            00000
            00010
        4.  01000
            01000
            00000
            00011
        """
        n, m = 4, 5
        island_positions = [[1, 1], [0, 1], [3, 3], [3, 4]]
        expected = [1, 1, 2, 2]
        self.assertEqual(expected, calculate_islands(n, m, island_positions))

    def test_example2(self):
        n, m = 3, 3
        island_positions = [[0, 0], [0, 1], [2, 2], [2, 1]]
        expected = [1, 1, 2, 2]
        self.assertEqual(expected, calculate_islands(n, m, island_positions))
    
    def test_empty_grid(self):
        self.assertEqual([], calculate_islands(0, 0, None))
        self.assertEqual([], calculate_islands(0, 0, []))
        self.assertEqual([], calculate_islands(0, 0, [[]]))
    
    def test_duplicated_island_positions(self):
        n, m = 3, 7
        islands_positions = [[1, 1], [1, 2], [1, 1], [3, 3], [1, 1]]
        expected = [1, 1, 1, 2, 2]
        self.assertEqual(expected, calculate_islands(n, m, islands_positions))
    
    def test_one_dimension_grid(self):
        n, m = 1, 3
        island_positions = [[0, 2], [0, 1], [0, 0]]
        expected = [1, 1, 1]
        self.assertEqual(expected, calculate_islands(n, m, island_positions))
    
    def test_one_dimension_grid2(self):
        n, m = 3, 1
        island_positions = [[0, 1], [0, 3]]
        expected = [1, 2]
        self.assertEqual(expected, calculate_islands(n, m, island_positions))
    
    def test_one_dimension_grid3(self):
        n, m = 1, 1
        island_positions = [[0, 0]]
        expected = [1]
        self.assertEqual(expected, calculate_islands(n, m, island_positions))

    def test_diagonal_positions(self):
        n, m = 3, 3
        island_positions = [[0, 0], [1, 1], [2, 2], [0, 2], [2, 0]]
        expected = [1, 2, 3, 4, 5]
        self.assertEqual(expected, calculate_islands(n, m, island_positions))

    def test_connect_different_areas(self):
        n, m = 3, 3
        island_positions = [[1, 0], [0, 1], [1, 2], [2, 1], [1, 1]]
        expected = [1, 2, 3, 4, 1]
        self.assertEqual(expected, calculate_islands(n, m, island_positions))

    def test_expand_entire_grid(self):
        n, m = 2, 3
        island_positions = [[1, 2], [0, 2], [0, 1], [1, 1], [1, 0], [0, 0]]
        expected = [1, 1, 1, 1, 1, 1]
        self.assertEqual(expected, calculate_islands(n, m, island_positions))

    def test_performance_test(self):
        n = m = 1 << 64
        island_positions = [[x, x] for x in range(128)]
        expected = list(range(1, 129))
        self.assertEqual(expected, calculate_islands(n, m, island_positions))

    def test_performance_test2(self):
        n = m = 1 << 64
        offset = 1 << 32
        island_positions = [[offset + x, offset + 0] for x in range(128)]
        expected = [1] * 128
        self.assertEqual(expected, calculate_islands(n, m, island_positions))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 26, 2021  LC 239 \[Medium\] Sliding Window Maximum
---
> **Question:** Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.
> 

**Example:**

```py
Input: nums = [1, 3, -1, -3, 5, 3, 6, 7], and k = 3
Output: [3, 3, 5, 5, 6, 7] 
```

**Explanation:**
```
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
 ```


**My thoughts:** The idea is to efficiently keep track of **INDEX** of 1st max, 2nd max, 3rd max and potentially k-th max elem. The reason for storing index is for the sake of avoiding index out of window. We can achieve that by using ***Double-Ended Queue*** which allow us to efficiently push and pop from both ends of the queue. 

The queue looks like `[index of 1st max, index of 2nd max, ...., index of k-th max]`

We might run into the following case as we progress:
- index of 1st max is out of bound of window: we pop left and index of 2nd max because 1st max within window
- the next elem become j-th max: evict old j-th max all the way to index of k-th max on the right of dequeue, i.e. pop right: `[index of 1st max, index of 2nd max, ..., index of j-1-th max, index of new elem]`
- 
**Solution with Sliding Window:** [https://replit.com/@trsong/Find-Sliding-Window-Maximum-3](https://replit.com/@trsong/Find-Sliding-Window-Maximum-3)
 ```py
 import unittest
from queue import deque

def max_sliding_window(nums, k):
    dq = deque()
    res = []

    for i, num in enumerate(nums):
        if dq and dq[0] <= i - k:
            dq.popleft()

        while dq and nums[dq[-1]] <= num:
            # mantain an increasing double-ended queue
            dq.pop()
        dq.append(i)

        if i >= k - 1:
            res.append(nums[dq[0]])
    return res


class MaxSlidingWindowSpec(unittest.TestCase):
    def test_example_array(self):
        k, nums = 3, [1, 3, -1, -3, 5, 3, 6, 7]
        expected = [3, 3, 5, 5, 6, 7]
        self.assertEqual(expected, max_sliding_window(nums, k))

    def test_empty_array(self):
        self.assertEqual([], max_sliding_window([], 1))

    def test_window_has_same_size_as_array(self):
        self.assertEqual([3], max_sliding_window([3, 2, 1], 3))

    def test_window_has_same_size_as_array2(self):
        self.assertEqual([2], max_sliding_window([1, 2], 2))

    def test_window_has_same_size_as_array3(self):
        self.assertEqual([-1], max_sliding_window([-1], 1))

    def test_non_ascending_array(self):
        k, nums = 2, [4, 3, 3, 2, 2, 1]
        expected = [4, 3, 3, 2, 2]
        self.assertEqual(expected, max_sliding_window(nums, k))

    def test_non_ascending_array2(self):
        k, nums = 2, [1, 1, 1]
        expected = [1, 1]
        self.assertEqual(expected, max_sliding_window(nums, k))

    def test_non_descending_array(self):
        k, nums = 3, [1, 1, 2, 2, 2, 3]
        expected = [2, 2, 2, 3]
        self.assertEqual(expected, max_sliding_window(nums, k))
    
    def test_non_descending_array2(self):
        self.assertEqual(max_sliding_window([1, 1, 2, 3], 1), [1, 1, 2 ,3])

    def test_first_decreasing_then_increasing_array(self):
        k, nums = 3, [5, 4, 1, 1, 1, 2, 2, 2]
        expected = [5, 4, 1, 2, 2, 2]
        self.assertEqual(expected, max_sliding_window(nums, k))
    
    def test_first_decreasing_then_increasing_array2(self):
        k, nums = 2, [3, 2, 1, 2, 3]
        expected = [3, 2, 2, 3]
        self.assertEqual(expected, max_sliding_window(nums, k))

    def test_first_decreasing_then_increasing_array3(self):
        k, nums = 3, [3, 2, 1, 2, 3]
        expected = [3, 2, 3]
        self.assertEqual(expected, max_sliding_window(nums, k))
    
    def test_first_increasing_then_decreasing_array(self):
        k, nums = 2, [1, 2, 3, 2, 1]
        expected = [2, 3, 3, 2]
        self.assertEqual(expected, max_sliding_window(nums, k))
    
    def test_first_increasing_then_decreasing_array2(self):
        k, nums = 3, [1, 2, 3, 2, 1]
        expected = [3, 3, 3]
        self.assertEqual(expected, max_sliding_window(nums, k))

    def test_oscillation_array(self):
        k, nums = 2, [1, -1, 1, -1, -1, 1, 1]
        expected = [1, 1, 1, -1, 1, 1]
        self.assertEqual(expected, max_sliding_window(nums, k))
    
    def test_oscillation_array2(self):
        k, nums = 3, [1, 3, 1, 2, 0, 5]
        expected = [3, 3, 2, 5]
        self.assertEqual(expected, max_sliding_window(nums, k))
 

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
 ```


### Dec 25, 2021 \[Medium\] Isolated Islands
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

**Solution with DFS:** [https://replit.com/@trsong/Count-Number-of-Isolated-Islands-2](https://replit.com/@trsong/Count-Number-of-Isolated-Islands-2)
```py
import unittest

def calc_islands(area_map):
    if not area_map or not area_map[0]:
        return 0
    n, m = len(area_map), len(area_map[0])

    visited = set()
    res = 0
    for r in range(n):
        for c in range(m):
            if area_map[r][c] == 1 and (r, c) not in visited:
                res += 1
                dfs_mark_island(area_map, (r, c), visited)
    return res


DIRECTIONS = [-1, 0, 1]

def dfs_mark_island(area_map, pos, visited):
    stack = [pos]
    n, m = len(area_map), len(area_map[0])
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))

        for dr in DIRECTIONS:
            for dc in DIRECTIONS:
                if dr == dc == 0:
                    continue
                new_r, new_c = r + dr, c + dc
                if (0 <= new_r < n and 
                    0 <= new_c < m and
                    (new_r, new_c) not in visited and
                    area_map[new_r][new_c] == 1):
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
    unittest.main(exit=False, verbosity=2)
```


### Dec 24, 2021 LC 65 \[Hard\] Valid Number
---
> **Question:** Given a string, return whether it represents a number. Here are the different kinds of numbers:
>
> - "10", a positive integer
> - "-10", a negative integer
> - "10.1", a positive real number
> - "-10.1", a negative real number
> - "1e5", a number in scientific notation
>
> And here are examples of non-numbers:
>
> - "a"
> - "x 1"
> - "a -2"
> - "-"

**Solution:** [https://replit.com/@trsong/Determine-valid-number-2](https://replit.com/@trsong/Determine-valid-number-2)
```py
import unittest

def is_valid_number(raw_num):
    """
    (WHITE_SPACE) (SIGN) DIGITS (DOT DIGITS) (e (SIGN) DIGITS) (WHITE_SPACE)
    or
     (WHITE_SPACE) (SIGN) DOT DIGITS (e (SIGN) DIGITS) (WHITE_SPACE)
    
    States:
    - START
    - SIGN
    - DIGITS
    - DOT
    - E
    - END
    
    State Transformation:
    START -> SIGN
          -> DOT
          -> DIGITS
    SIGN -> DIGITS
         -> DOT
    DIGITS -> DOT
           -> E
           -> END
    DOT -> DIGITS
    E -> SIGN
      -> DIGITS  

    Prev States:
    START: []
    SIGN: [START, E]
    DIGITS: [START, SIGN, DOT]
    DOT: [START, SIGN, DIGITS]
    E: [DIGITS]
    END: [DIGITS]
    """
    raw_num = raw_num.strip()
    dot_seen = False
    e_seen = False
    number_seen = False
    prev_ch = None
    for ch in raw_num:
        if '0' <= ch <= '9':
            number_seen = True
        elif ch == '.':
            if e_seen or dot_seen:
                return False
            dot_seen = True
        elif ch == 'e':
            if e_seen or not number_seen:
                return False
            number_seen = False
            e_seen = True
        elif ch in ['-', '+']:
            if prev_ch is not None and prev_ch != 'e':
                return False
        else:
            return False
        prev_ch = ch
    return number_seen


class IsValidNumberSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(is_valid_number("123"))  # Integer

    def test_example2(self):
        self.assertTrue(is_valid_number("12.3"))  # Floating point
    
    def test_example3(self):
        self.assertTrue(is_valid_number("-123"))  # Negative numbers
    
    def test_example4(self):
        self.assertTrue(is_valid_number("-.3"))  # Negative floating point
    
    def test_example5(self):
        self.assertTrue(is_valid_number("1.5e5")) # Scientific notation
    
    def test_example6(self):
        self.assertFalse(is_valid_number("12a"))  # No letters
    
    def test_example7(self):
        self.assertFalse(is_valid_number("1 2")) # No space between numbers
    
    def test_example8(self):
        self.assertFalse(is_valid_number("1e1.2")) # Exponent can only be an integer (positive or negative or 0)

    def test_empty_string(self):
        self.assertFalse(is_valid_number(""))

    def test_blank_string(self):
        self.assertFalse(is_valid_number("   "))

    def test_just_signs(self):
        self.assertFalse(is_valid_number("+"))

    def test_zero(self):
        self.assertTrue(is_valid_number("0"))

    def test_contains_no_number(self):
        self.assertFalse(is_valid_number("e"))

    def test_contains_white_spaces(self):
        self.assertTrue(is_valid_number(" -123.456  "))

    def test_scientific_notation(self):
        self.assertTrue(is_valid_number("2e10"))

    def test_scientific_notation2(self):
        self.assertFalse(is_valid_number("10e5.4"))

    def test_scientific_notation3(self):
        self.assertTrue(is_valid_number("-24.35e-10"))

    def test_scientific_notation4(self):
        self.assertFalse(is_valid_number("1e1e1"))

    def test_scientific_notation5(self):
        self.assertTrue(is_valid_number("+.5e-23"))

    def test_scientific_notation6(self):
        self.assertFalse(is_valid_number("+e-23"))

    def test_scientific_notation7(self):
        self.assertFalse(is_valid_number("0e"))

    def test_multiple_signs(self):
        self.assertFalse(is_valid_number("+-2"))

    def test_multiple_signs2(self):
        self.assertFalse(is_valid_number("-2-2-2-2"))

    def test_multiple_signs3(self):
        self.assertFalse(is_valid_number("6+1"))

    def test_multiple_dots(self):
        self.assertFalse(is_valid_number("10.24.25"))

    def test_sign_and_dot(self):
        self.assertFalse(is_valid_number(".-4"))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 23, 2021  LC 679 \[Hard\] 24 Game
---
> **Question:** The 24 game is played as follows. You are given a list of four integers, each between 1 and 9, in a fixed order. By placing the operators +, -, *, and / between the numbers, and grouping them with parentheses, determine whether it is possible to reach the value 24.
>
> For example, given the input `[5, 2, 7, 8]`, you should return `True`, since `(5 * 2 - 7) * 8 = 24`.
>
> Write a function that plays the 24 game.

**Solution with Backtracking:** [https://replit.com/@trsong/24-Game-2](https://replit.com/@trsong/24-Game-2)
```py
import unittest

def play_24_game(cards):
    if len(cards) == 1:
        return abs(cards[0] - 24) < 1e-3
    else:
        n = len(cards)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for new_card in apply_ops(cards[i], cards[j]):
                    remaining_cards = [new_card] + [cards[x] for x in range(n) if x != i and x != j]
                    if play_24_game(remaining_cards):
                        return True
        return False


def apply_ops(num1, num2):
    yield num1 + num2
    yield num1 - num2
    yield num1 * num2
    if num2 != 0:
        yield num1 / num2 


class Play24GameSpec(unittest.TestCase):
    def test_example(self):
        cards = [5, 2, 7, 8]  # (5 * 2 - 7) * 8 = 24
        self.assertTrue(play_24_game(cards))

    def test_example2(self):
        cards = [4, 1, 8, 7]  # (8 - 4) * (7 - 1) = 24
        self.assertTrue(play_24_game(cards))

    def test_example3(self):
        cards = [1, 2, 1, 2] 
        self.assertFalse(play_24_game(cards))

    def test_sum_to_24(self):
        cards = [6, 6, 6, 6]  # 6 + 6 + 6 + 6 = 24
        self.assertTrue(play_24_game(cards))

    def test_require_division(self):
        cards = [4, 7, 8, 8]  # 4 * (7 - 8 / 8) = 24
        self.assertTrue(play_24_game(cards))
    
    def test_has_fraction(self):
        cards = [1, 3, 4, 6]  # 6 / (1 - 3/ 4) = 24
        self.assertTrue(play_24_game(cards))

    def test_unable_to_solve(self):
        cards = [1, 1, 1, 1] 
        self.assertFalse(play_24_game(cards))

    def test_unable_to_solve2(self):
        cards = [1, 5, 5, 8] 
        self.assertFalse(play_24_game(cards))

    def test_unable_to_solve3(self):
        cards = [2, 9, 9, 9] 
        self.assertFalse(play_24_game(cards))

    def test_unable_to_solve4(self):
        cards = [2, 2, 7, 9] 
        self.assertFalse(play_24_game(cards))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 22, 2021 \[Easy\] Max of Min Pairs
---
> **Question:** Given an array of length `2 * n (even length)` that consists of random integers, divide the array into pairs such that the sum of smaller integers in each pair is maximized. Return such sum.

**Example:**
```py
Input: [3, 4, 2, 5]
Ouput: 6
Explanation: The maximum sum of pairs is 6 = min(2, 3) + min(4, 5)
```

**My thoughts:** To get max sum as possible, we want smaller of each pair as large as possible. 

Well, that's one way of thinking. Another way is think about how not to waste number. We can start from max number which is guarentee to be wasted: We can pair 1st max with 2nd max and earn 2nd max. 

After pick those two numbers, we end up with a sub-problem. To solve that sub-problem, we pick the largest two numbers, ie. 3rd max and 4th max from original lists. 

Such process continues until we exhaust the entire lists. Then you start realize, we can sort the list and just sum all even postions.


**Greedy Solution:** [https://replit.com/@trsong/Max-of-Min-Pairs](https://replit.com/@trsong/Max-of-Min-Pairs)
```py
import unittest

def sum_of_min_pairs(nums):
    nums.sort()
    res = 0
    for i in range(0, len(nums), 2):
        res += nums[i]
    return res
        

class SumOfMinPairSpec(unittest.TestCase):
    def test_example(self):
        nums = [3, 4, 2, 5]
        expected = 6  # min(5, 4) + min(3, 2)
        self.assertEqual(expected, sum_of_min_pairs(nums))

    def test_empty_array(self):
        self.assertEqual(0, sum_of_min_pairs([]))

    def test_array_with_two_elements(self):
        nums = [1, 2]
        expected = 1  # min(1, 2) 
        self.assertEqual(expected, sum_of_min_pairs(nums))

    def test_array_with_unique_value(self):
        nums = [2, 2, 2, 2, 2, 2]
        expected = 6  # min(2, 2) + min(2, 2) + min(2, 2)  
        self.assertEqual(expected, sum_of_min_pairs(nums))

    def test_array_with_duplicated_elements(self):
        nums = [1, 2, 2, 1, 3, 3]
        expected = 6  # min(3, 3) + min(2, 2) + min(1, 1) 
        self.assertEqual(expected, sum_of_min_pairs(nums))

    def test_array_with_negative_numbers(self):
        nums = [1, -1, -2, -3, 2, 3]
        expected = -2  # min(3, 2) + min(1, -1) + min(-2, -3)
        self.assertEqual(expected, sum_of_min_pairs(nums))

    def test_array_with_outliers(self):
        nums = [1, 2, 3, 101, 102, 103]
        expected = 106  # min(103, 102) + min(101, 3) + min(2, 1)
        self.assertEqual(expected, sum_of_min_pairs(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 21, 2021 \[Medium\] Max Number of Equal Sum Pairs
---
> **Question:** You are given an array of integers. Your task is to create pairs of them, such that every created pair has the same sum. This sum is NOT specified, but the number of created pairs should be the maximum possible. Each array element may belong to one pair only. 
> 
> Write a function: `public int solution(int[] A)` that given an array A of N integers, returns the maximum possible number of pairs with the same sum.

**Example 1:**
```py
Input: A = [1, 9, 8, 100, 2]
Return: 2
Explanation: the pairs are [1, 9] and [8, 2] for a sum of 10
```

**Example 2:**
```py
Input: A = [2, 2, 2, 3]
Return: 1
Explanation: [2, 2] (sum 4) OR [2, 3] (sum 5). 
Notice we can only form sum 4 once since there is overlap between the elements
```

**Example 3:**
```py
Input: A = [2, 2, 2, 2, 2]
Return: 2 
Explanation: [2, 2] and [2, 2] for a sum for 4. 
The fifth 2 is not used, while the first four 2's are used.
```

**Solution:** [https://replit.com/@trsong/Max-Number-of-Equal-Sum-Pairs](https://replit.com/@trsong/Max-Number-of-Equal-Sum-Pairs)
```py
import unittest

def max_equal_sum_pairs(nums):
    histogram = {}
    for num in nums:
        histogram[num] = histogram.get(num, 0) + 1

    sum_histogram = {}
    uniq_nums = list(histogram.keys())
    for i, num1 in enumerate(uniq_nums):
        for j in range(i + 1):
            num2 = uniq_nums[j]
            pair_sum = num1 + num2
            count = histogram[num1] // 2 if i == j else min(
                histogram[num1], histogram[num2])
            sum_histogram[pair_sum] = sum_histogram.get(pair_sum, 0) + count
    return max(sum_histogram.values())


class MaxEqualSumPairSpec(unittest.TestCase):
    def test_example(self):
        nums = [1, 9, 8, 100, 2]
        expected = 2  # [1, 9] and [8, 2]
        self.assertEqual(expected, max_equal_sum_pairs(nums))

    def test_example2(self):
        nums = [2, 2, 2, 3]
        expected = 1  # [2, 2]
        self.assertEqual(expected, max_equal_sum_pairs(nums))

    def test_example3(self):
        nums = [2, 2, 2, 2, 2]
        expected = 2  # [2, 2] and [2, 2]
        self.assertEqual(expected, max_equal_sum_pairs(nums))

    def test_element_equals_target(self):
        nums = [1, 2, 4, 3, 3, 5, 6]
        expected = 3  # [1, 5], [2, 4] and [3, 3]
        self.assertEqual(expected, max_equal_sum_pairs(nums))

    def test_unique_values(self):
        nums = [1, 1]
        expected = 1  # [1, 1]
        self.assertEqual(expected, max_equal_sum_pairs(nums))

    def test_unique_values2(self):
        nums = [1, 1, 1, 1, 1, 1, 1, 1]
        expected = 4  # [1, 1], [1, 1], [1, 1] and [1, 1]
        self.assertEqual(expected, max_equal_sum_pairs(nums))

    def test_different_values(self):
        nums = [1, 2]
        expected = 1  # [1, 2]
        self.assertEqual(expected, max_equal_sum_pairs(nums))

    def test_element_can_only_be_picked_once(self):
        nums = [1, 2, 3, 4, 5]
        expected = 2  # [1, 5] and [2, 4]
        self.assertEqual(expected, max_equal_sum_pairs(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 20, 2021 \[Medium\] Rearrange String with Repeated Characters
---
> **Question:** Given a string with repeated characters, rearrange the string so that no two adjacent characters are the same. If this is not possible, return None.
>
> For example, given `"aaabbc"`, you could return `"ababac"`. Given `"aaab"`, return `None`.

**My thoughts:** This prblem is a special case of [Rearrange String K Distance Apart](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug.html#sep-11-2020-lc-358-hard-rearrange-string-k-distance-apart). Just Greedily choose the character with max remaining number for each window size 2. If no such character satisfy return None instead.

**Solution with Max-Heap:** [https://replit.com/@trsong/Rearrange-String-with-Repeated-Characters-3](https://replit.com/@trsong/Rearrange-String-with-Repeated-Characters-3)
```py
import unittest
from Queue import PriorityQueue

def rearrange_string(s):
    char_freq = {}
    for ch in s:
        char_freq[ch] = char_freq.get(ch, 0) + 1

    max_heap = PriorityQueue()
    for ch, freq in char_freq.items():
        max_heap.put((-freq, ch))

    res = []
    while not max_heap.empty():
        remaining = []
        for _ in range(2):
            if max_heap.empty() and not remaining:
                break
            if max_heap.empty():
                return None
            
            neg_freq, ch = max_heap.get()
            if abs(neg_freq) > 1:
                remaining.append((abs(neg_freq) - 1, ch))
            res.append(ch)
        
        for freq, ch in remaining:
            max_heap.put((-freq, ch))
    return ''.join(res)


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

### Dec 19, 2021  \[Medium\] Satisfactory Playlist
---
> **Question:** You have access to ranked lists of songs for various users. Each song is represented as an integer, and more preferred songs appear earlier in each list. For example, the list `[4, 1, 7]` indicates that a user likes song `4` the best, followed by songs `1` and `7`.
>
> Given a set of these ranked lists, interleave them to create a playlist that satisfies everyone's priorities.
>
> For example, suppose your input is `[[1, 7, 3], [2, 1, 6, 7, 9], [3, 9, 5]]`. In this case a satisfactory playlist could be `[2, 1, 6, 7, 3, 9, 5]`.


**My thoughts:** create a graph with vertex being song and edge `(u, v)` representing that u is more preferred than v. A topological order will make sure that all more preferred song will go before less preferred ones. Thus gives a list that satisfies everyone's priorities, if there is one (no cycle).

**Solution with Topological Sort:** [https://replit.com/@trsong/Satisfactory-Playlist-for-Everyone-3](https://replit.com/@trsong/Satisfactory-Playlist-for-Everyone-3)
```py
import unittest

def calculate_satisfactory_playlist(preference):
    inward_counts = {}
    node_set = set(node for order in preference for node in order)
    neighbors = {}
    
    for order in preference:
        for i in range(1, len(order)):
            prev, cur = order[i - 1], order[i]
            neighbors[prev] = neighbors.get(prev, [])
            neighbors[prev].append(cur)
            inward_counts[cur] = inward_counts.get(cur, 0) + 1
            
    res = []
    queue = [node for node in node_set if node not in inward_counts]
    while queue:
        cur = queue.pop(0)
        res.append(cur)
        for nb in neighbors.get(cur, []):
            inward_counts[nb] -= 1
            if inward_counts[nb] == 0:
                del inward_counts[nb]
                queue.append(nb)
    
    # check if cycle exists
    return res if len(res) == len(node_set) else None


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

### Dec 18, 2021 LC 91 \[Medium\] Decode Ways
---
> **Question:** A message containing letters from `A-Z` is being encoded to numbers using the following mapping:
>
> ```py
> 'A' -> 1
> 'B' -> 2
> ...
> 'Z' -> 26
> ```
> Given an encoded message containing digits, determine the total number of ways to decode it.

**Example 1:**
```py
Input: "12"
Output: 2
Explanation: It could be decoded as AB (1 2) or L (12).
```

**Example 2:**
```py
Input: "10"
Output: 1
```

**Solution with DP:** [https://replit.com/@trsong/Number-of-Decode-Ways-2](https://replit.com/@trsong/Number-of-Decode-Ways-2)
```py
import unittest

def decode_ways(encoded_string):
    if not encoded_string or encoded_string == '0':
        return 0

    n = len(encoded_string)
    # Let dp[n] represents number of decode ways ended at length n i.e. s[:n]
    # dp[n] = dp[n - 1] + dp[n - 2] if both last and last 2 digits are valid
    # or    = dp[n - 1]             if only 1 digit is valid (1 ~ 9)
    # or    = dp[n - 2]             if only 2 digits are valid (10 ~ 26)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1

    ord_zero = ord('0')
    for i in range(2, n + 1):
        last_digit = ord(encoded_string[i - 1]) - ord_zero
        second_last_digit = ord(encoded_string[i - 2]) - ord_zero
        if last_digit > 0:
            dp[i] += dp[i - 1]
        
        if 10 <= 10 * second_last_digit + last_digit <= 26:
            dp[i] += dp[i - 2]
    return dp[n]


class DecodeWaySpec(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(0, decode_ways(""))

    def test_invalid_string(self):
        self.assertEqual(0, decode_ways("0"))

    def test_length_one_string(self):
        self.assertEqual(1, decode_ways("2"))

    def test_length_one_string2(self):
        self.assertEqual(1, decode_ways("9"))

    def test_length_two_string(self):
        self.assertEqual(1, decode_ways("20"))  # 20

    def test_length_two_string2(self):
        self.assertEqual(2, decode_ways("19"))  # 1,9 and 19

    def test_length_three_string(self):
        self.assertEqual(3, decode_ways("121"))  # 1, 20 and 12, 0

    def test_length_three_string2(self):
        self.assertEqual(1, decode_ways("120"))  # 1, 20

    def test_length_three_string3(self):
        self.assertEqual(1, decode_ways("209"))  # 20, 9

    def test_length_three_string4(self):
        self.assertEqual(2, decode_ways("912"))  # 9,1,2 and 9,12

    def test_length_three_string5(self):
        self.assertEqual(2, decode_ways("231"))  # 2,3,1 and 23, 1

    def test_length_three_string6(self):
        self.assertEqual(3, decode_ways("123"))  # 1,2,3, and 1, 23 and 12, 3

    def test_length_four_string(self):
        self.assertEqual(3, decode_ways("1234"))

    def test_length_four_string2(self):
        self.assertEqual(5, decode_ways("1111"))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 17, 2021 \[Hard\] Longest Path in Binary Tree
---
> **Question:** Given a binary tree, return any of the longest path.

**Example 1:**
```py
Input:      1
          /   \
        2      3
      /  \
    4     5

Output: [4, 2, 1, 3] or [5, 2, 1, 3]  
```

**Example 2:**
```py
Input:      1
          /   \
        2      3
      /  \      \
    4     5      6

Output: [4, 2, 1, 3, 6] or [5, 2, 1, 3, 6] 
```

**My thoughts:** We can BFS from any node `x` to find the farthest node `y`. Such `y` must be an end of the longest path. Then we can perform another BFS from `y` to find farthest node of `y` say `v`. That path `y-v` will be the longest path in a tree.

**Solution with BFS:** [https://replit.com/@trsong/Longest-Path-in-Binary-Tree-1](https://replit.com/@trsong/Longest-Path-in-Binary-Tree-1)
```py
import unittest

def find_longest_path(root):
    if not root:
        return []

    parents = {}
    farest_node = bfs_longest_path(root, parents)[0]
    longest_path = bfs_longest_path(farest_node, parents)
    return list(map(lambda node: node.val, longest_path))


def bfs_longest_path(start, parents):
    prev_nodes = {}
    queue = [(start, None)]
    last_node = start
    while queue:
        for _ in range(len(queue)):
            cur, prev = queue.pop(0)
            last_node = cur
            parents[cur] = parents.get(cur, prev)
                        
            for next_node in [cur.right, cur.left, parents.get(cur)]:
                if not next_node or next_node == prev:
                    continue
                queue.append((next_node, cur))
                prev_nodes[next_node] = cur

    res = []
    while last_node:
        res.append(last_node)
        last_node = prev_nodes.get(last_node)
    return res


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class FindLongetPathSpec(unittest.TestCase):
    def assert_result(self, possible_solutions, result):
        reversed_solution = list(map(lambda path: path[::-1], possible_solutions))
        solutions = possible_solutions + reversed_solution
        self.assertIn(result, solutions, "\nIncorrect result: {}.\nPossible solutions:\n{}".format(str(result), "\n".join(map(str, solutions))))

    def test_example(self):
        """ 
           1
          / \
         2   3
        / \
       4   5
        """
        root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
        possible_solutions = [
            [4, 2, 1, 3],
            [5, 2, 1, 3]
        ]
        self.assert_result(possible_solutions, find_longest_path(root))

    def test_example2(self):
        """
            1
           / \
          2   3
         / \   \
        4   5   6
        """
        left_tree = TreeNode(2, TreeNode(4), TreeNode(5))
        right_tree = TreeNode(3, right=TreeNode(6))
        root = TreeNode(1, left_tree, right_tree)
        possible_solutions = [
            [4, 2, 1, 3, 6],
            [5, 2, 1, 3, 6]
        ]
        self.assert_result(possible_solutions, find_longest_path(root))

    def test_empty_tree(self):
        self.assertEqual([], find_longest_path(None))

    def test_longest_path_start_from_root(self):
        """
        1
         \
          2
         / 
        3  
       / \
      5   4
         /
        6
        """
        n3 = TreeNode(3, TreeNode(5), TreeNode(4, TreeNode(6)))
        n2 = TreeNode(2, n3)
        root = TreeNode(1, right=n2)
        possible_solutions = [
            [1, 2, 3, 4, 6]
        ]
        self.assert_result(possible_solutions, find_longest_path(root))

    def test_longest_path_goes_through_root(self):
        """
            1
           / \
          2   3
         /     \
        4       5
        """
        left_tree = TreeNode(2, TreeNode(4))
        right_tree = TreeNode(3, right=TreeNode(5))
        root = TreeNode(1, left_tree, right_tree)
        possible_solutions = [
            [4, 2, 1, 3, 5]
        ]
        self.assert_result(possible_solutions, find_longest_path(root))

    def test_longest_path_not_through_root(self):
        """
         1
        / \
       2   3
          / \
         4   5
        /   / \
       6   7   8
      /    \
     9     10
        """
        right_left_tree = TreeNode(4, TreeNode(6, TreeNode(9)))
        right_right_tree = TreeNode(5, TreeNode(7, right=TreeNode(10)), TreeNode(8))
        right_tree = TreeNode(3, right_left_tree, right_right_tree)
        root = TreeNode(1, TreeNode(2), right_tree)
        possible_solutions = [
            [10, 7, 5, 3, 4, 6, 9]
        ]
        self.assert_result(possible_solutions, find_longest_path(root))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 16, 2021 \[Medium\] All Max-size Subarrays with Distinct Elements
---
> **Question:** Given an array of integers, print all maximum size sub-arrays having all distinct elements in them.

**Example:**
```py
Input: [5, 2, 3, 5, 4, 3]
Output: [[5, 2, 3], [2, 3, 5, 4], [5, 4, 3]]
```

**Solution with Sliding Window:** [https://replit.com/@trsong/Find-All-Max-size-Subarrays-with-Distinct-Elements-2](https://replit.com/@trsong/Find-All-Max-size-Subarrays-with-Distinct-Elements-2)
```py
import unittest

def all_max_distinct_subarray(nums):
    if not nums:
        return []
        
    last_occur = {}
    start = 0
    res = []

    for end, num in enumerate(nums):
        if start <= last_occur.get(num, -1):
            res.append(nums[start: end])
            start = last_occur[num] + 1
        last_occur[num] = end
    
    res.append(nums[start:])
    return res


class AllMaxDistinctSubarraySpec(unittest.TestCase):
    def test_example(self):
        nums = [5, 2, 3, 5, 4, 3]
        expected = [[5, 2, 3], [2, 3, 5, 4], [5, 4, 3]]
        self.assertCountEqual(expected, all_max_distinct_subarray(nums))

    def test_empty_array(self):
        self.assertCountEqual([], all_max_distinct_subarray([]))

    def test_array_with_no_duplicates(self):
        nums = [1, 2, 3, 4, 5, 6]
        expected = [[1, 2, 3, 4, 5, 6]]
        self.assertCountEqual(expected, all_max_distinct_subarray(nums))

    def test_array_with_unique_numbers(self):
        nums = [0, 0, 0]
        expected = [[0], [0], [0]]
        self.assertCountEqual(expected, all_max_distinct_subarray(nums))
    
    def test_should_give_max_size_disctinct_array(self):
        nums = [0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3]
        expected = [[0, 1], [1, 0], [0, 1, 2], [1, 2, 0], [0, 1, 2, 3]]
        self.assertCountEqual(expected, all_max_distinct_subarray(nums))

    def test_should_give_max_size_disctinct_array2(self):
        nums = [0, 1, 2, 3, 2, 3, 1, 3, 0]
        expected = [[0, 1, 2, 3], [3, 2], [2, 3, 1], [1, 3, 0]]
        self.assertCountEqual(expected, all_max_distinct_subarray(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Dec 15, 2021 LC 543 \[Easy\] Diameter of Binary Tree
---
> **Question:** Given the root of a binary tree, return the length of the diameter of the tree.
>
> The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.
>
> The length of a path between two nodes is represented by the number of edges between them.

**Example 1:**
```py
Input: 
    1
   / \
  2   3
 / \
4   5
Output: 3
Explanation: 3 is the length of the path [4,2,1,3] or [5,2,1,3].
```

**Example 2:**
```py
Input: 
  1
 /
2
Output: 1
```

**Solution with Recursion:** [https://replit.com/@trsong/Diameter-of-Binary-Tree](https://replit.com/@trsong/Diameter-of-Binary-Tree)
```py
import unittest

def binary_tree_diameter(root):
    if not root:
        return None

    _, max_path = binary_tree_max_path_recur(root)
    return max_path - 1


def binary_tree_max_path_recur(root):
    if not root:
        return 0, 0

    left_max_height, left_res = binary_tree_max_path_recur(root.left)
    right_max_height, right_res = binary_tree_max_path_recur(root.right)
    return 1 + max(left_max_height, right_max_height), max(left_res, right_res, 1 + left_max_height + right_max_height)


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class BinaryTreeDiameterSpec(unittest.TestCase):
    def test_example(self):
        """
            1
           / \
          2   3
         / \
        4   5
        """
        n2 = TreeNode(2, TreeNode(4), TreeNode(5))
        root = TreeNode(1, n2, TreeNode(3))
        # 4, 2, 1, 3
        self.assertEqual(3, binary_tree_diameter(root))  

    def test_example2(self):
        """
          1
         /
        2
        """
        root = TreeNode(1, TreeNode(2))
        self.assertEqual(1, binary_tree_diameter(root))

    def test_example3(self):
        """
            1
           / \
          2   3
         / \   \
        4   5   6
        """
        left_tree = TreeNode(2, TreeNode(4), TreeNode(5))
        right_tree = TreeNode(3, right=TreeNode(6))
        root = TreeNode(1, left_tree, right_tree)
        # 4, 2, 1, 3, 6
        self.assertEqual(4, binary_tree_diameter(root))

    def test_empty_tree(self):
        self.assertIsNone(binary_tree_diameter(None))

    def test_longest_path_start_from_root(self):
        """
        1
         \
          2
         / 
        3  
       / \
      5   4
         /
        6
        """
        n3 = TreeNode(3, TreeNode(5), TreeNode(4, TreeNode(6)))
        n2 = TreeNode(2, n3)
        root = TreeNode(1, right=n2)
        # 1, 2, 3, 4, 6
        self.assertEqual(4, binary_tree_diameter(root))

    def test_longest_path_goes_through_root(self):
        """
            1
           / \
          2   3
         /     \
        4       5
        """
        left_tree = TreeNode(2, TreeNode(4))
        right_tree = TreeNode(3, right=TreeNode(5))
        root = TreeNode(1, left_tree, right_tree)
        # 4, 2, 1, 3, 5
        self.assertEqual(4, binary_tree_diameter(root))

    def test_longest_path_not_through_root(self):
        """
         1
        / \
       2   3
          / \
         4   5
        /   / \
       6   7   8
      /    \
     9     10
        """
        right_left_tree = TreeNode(4, TreeNode(6, TreeNode(9)))
        right_right_tree = TreeNode(5, TreeNode(7, right=TreeNode(10)), TreeNode(8))
        right_tree = TreeNode(3, right_left_tree, right_right_tree)
        root = TreeNode(1, TreeNode(2), right_tree)
        # 10, 7, 5, 3, 4, 6, 9
        self.assertEqual(6, binary_tree_diameter(root))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 14, 2021  \[Medium\] Power of 4
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

**Solution:** [https://replit.com/@trsong/Determine-If-Number-Is-Power-of-4-2](https://replit.com/@trsong/Determine-If-Number-Is-Power-of-4-2)
```py
import unittest

def is_power_of_four(num):
    is_positive = num > 0
    is_power_of_two = num & (num - 1) == 0
    none_even_bits = num & 0xAAAAAAAA == 0  #  A is 0b1010
    return is_positive and is_power_of_two and none_even_bits


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

### Dec 13, 2021 LC 300 \[Hard\] The Longest Increasing Subsequence
---
> **Question:** Given an array of numbers, find the length of the longest increasing **subsequence** in the array. The subsequence does not necessarily have to be contiguous.
>
> For example, given the array `[0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]`, the longest increasing subsequence has length `6` ie. `[0, 2, 6, 9, 11, 15]`.

**Solution:** [https://replit.com/@trsong/Find-the-Longest-Increasing-Subsequence-2](https://replit.com/@trsong/Find-the-Longest-Increasing-Subsequence-2)
```py
import unittest

def longest_increasing_subsequence(sequence):
    ascending_seq = []
    for num in sequence:
        # insert_pos is the min index such that ascending_seq[i] >= num
        # replace ascending_seq[i] with num won't affect ascending order
        insert_pos = binary_search(ascending_seq, num)
        if insert_pos == len(ascending_seq):
            ascending_seq.append(num)
        else:
            ascending_seq[insert_pos] = num
    return len(ascending_seq)


def binary_search(nums, target):
    lo = 0
    hi = len(nums)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


class LongestIncreasingSubsequnceSpec(unittest.TestCase):
    def test_empty_sequence(self):
        self.assertEqual(0, longest_increasing_subsequence([]))

    def test_last_elem_is_local_max(self):
        seq = [1, 2, 3, 0, 2]
        expected = 3  # [1, 2, 3]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_last_elem_is_global_max(self):
        seq = [1, 2, 3, 0, 6]
        expected = 4  # [1, 2, 3, 6]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_longest_increasing_subsequence_in_first_half_sequence(self):
        seq = [4, 5, 6, 7, 1, 2, 3]
        expected = 4  # [4, 5, 6, 7]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_longest_increasing_subsequence_in_second_half_sequence(self):
        seq = [1, 2, 3, -2, -1, 0, 1]
        expected = 4  # [-2, -1, 0, 1]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_sequence_in_up_down_up_pattern(self):
        seq = [1, 2, 3, 2, 4]
        expected = 4  # [1, 2, 2, 4]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_sequence_in_up_down_up_pattern2(self):
        seq = [1, 2, 3, -1, 0]
        expected = 3  # [1, 2, 3]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_sequence_in_down_up_down_pattern(self):
        seq = [4, 3, 5]
        expected = 2  # [3, 5]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_sequence_in_down_up_down_pattern2(self):
        seq = [4, 0, 1]
        expected = 2  # [0, 1]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_multiple_result(self):
        seq = [10, 9, 2, 5, 3, 7, 101, 18]
        expected = 4  # [2, 3, 7, 101]
        self.assertEqual(expected, longest_increasing_subsequence(seq))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Dec 12, 2021 \[Hard\] Decreasing Subsequences
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

**My thoughts:** This question is equivalent to [Longest Increasing Subsequence](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay.html#june-19-2020-lc-300-hard-the-longest-increasing-subsequence). Can be solved with greedy approach.

Imagine we are to create a list of stacks each in descending order (stack top is smallest). And those stacks are sorted by each stack's top element. 

Then for each element from input sequence, we just need to figure out (using binary search) the stack such that by pushing this element into stack, result won't affect the order of stacks and decending property of each stack. 

Finally, the total number of stacks equal to min number of subsequence we can get by splitting. Each stack represents a decreasing subsequence.

**Greedy Solution with Descending Stack and Binary Search:** [https://replit.com/@trsong/Decreasing-Subsequences-2](https://replit.com/@trsong/Decreasing-Subsequences-2)
```py
import unittest

def min_decreasing_subsequences(sequence):
    stack_list = []
    for num in sequence:
        pos = binary_search_stack_tops(stack_list, num)
        if pos == len(stack_list):
            stack_list.append([])
        stack_list[pos].append(num)
    return len(stack_list)


def binary_search_stack_tops(stack_list, target):
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

### Dec 11, 2021 \[Medium\] Longest Consecutive Sequence in an Unsorted Array
---
> **Question:** Given an array of integers, return the largest range, inclusive, of integers that are all included in the array.
>
> For example, given the array `[9, 6, 1, 3, 8, 10, 12, 11]`, return `(8, 12)` since `8, 9, 10, 11, and 12` are all in the array.


**Solution:** [https://replit.com/@trsong/Longest-Consecutive-Sequence-3](https://replit.com/@trsong/Longest-Consecutive-Sequence-3)
```py
import unittest

def longest_consecutive_seq(nums):
    num_set = set(nums)
    max_window = None
    max_window_size = 0

    for num in nums:
        if num not in num_set:
            continue
        num_set.remove(num)

        lo = num - 1
        while lo in num_set:
            num_set.remove(lo)
            lo -= 1
        
        hi = num + 1
        while hi in num_set:
            num_set.remove(hi)
            hi += 1 
        
        window_size = hi - lo - 1
        if window_size > max_window_size:
            max_window_size = window_size
            max_window = (lo + 1, hi - 1)
    return max_window


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


### Dec 10, 2021 \[Medium\] Count Attacking Bishop Pairs
---
> **Question:** On our special chessboard, two bishops attack each other if they share the same diagonal. This includes bishops that have another bishop located between them, i.e. bishops can attack through pieces.
> 
> You are given N bishops, represented as (row, column) tuples on a M by M chessboard. Write a function to count the number of pairs of bishops that attack each other. The ordering of the pair doesn't matter: `(1, 2)` is considered the same as `(2, 1)`.
>
> For example, given `M = 5` and the list of bishops:

```py
(0, 0)
(1, 2)
(2, 2)
(4, 0)
```
> The board would look like this:

```py
[b 0 0 0 0]
[0 0 b 0 0]
[0 0 b 0 0]
[0 0 0 0 0]
[b 0 0 0 0]
```

> You should return 2, since bishops 1 and 3 attack each other, as well as bishops 3 and 4.


**My thoughts:** Cell on same diagonal has the following properties:

- Major diagonal: col - row = constant
- Minor diagonal: col + row = constant
  

**Example:**
```py
>>> [[r-c for c in xrange(5)] for r in xrange(5)]
[
    [0, -1, -2, -3, -4],
    [1, 0, -1, -2, -3],
    [2, 1, 0, -1, -2],
    [3, 2, 1, 0, -1],
    [4, 3, 2, 1, 0]
]

>>> [[r+c for c in xrange(5)] for r in xrange(5)]
[
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8]
]
```
Thus, we can store the number of bishop on the same diagonal and use the formula to calculate n-choose-2: `n(n-1)/2`. Or we think about for each bishop added to a specific diagonal, the total number of pairs contributed by such bishop equal to number of bishop on the existing diagonal.


**Solution:** [https://replit.com/@trsong/Count-Number-of-Attacking-Bishop-Pairs-2](https://replit.com/@trsong/Count-Number-of-Attacking-Bishop-Pairs-2)
```py
import unittest

def count_attacking_pairs(bishop_positions):
    if not bishop_positions or not bishop_positions[0]:
        return 0

    res = 0
    major_diagonal_count = {}
    minor_diagonal_count = {}
    for r, c in bishop_positions:
        major_diagonal = r - c
        minor_diagonal = r + c

        major_count = major_diagonal_count.get(major_diagonal, 0) 
        minor_count = minor_diagonal_count.get(minor_diagonal, 0) 
        res += major_count + minor_count

        major_diagonal_count[major_diagonal] = major_count + 1
        minor_diagonal_count[minor_diagonal] = minor_count + 1
    return res


class CountAttackingPairSpec(unittest.TestCase):
    def test_zero_bishops(self):
        self.assertEqual(0, count_attacking_pairs([]))

    def test_bishops_everywhere(self):
        """
        b b
        b b
        """
        self.assertEqual(2, count_attacking_pairs([(0, 0), (0, 1), (1, 0), (1, 1)]))

    def test_zero_attacking_pairs(self):
        """
        0 b 0 0
        0 b 0 0
        0 b 0 0
        0 b 0 0
        """
        self.assertEqual(0, count_attacking_pairs([(0, 1), (1, 1), (2, 1), (3, 1)]))
        """
        0 0 0 b
        b 0 0 0
        b 0 b 0
        0 0 0 0
        """
        self.assertEqual(0, count_attacking_pairs([(0, 3), (1, 0), (2, 0), (2, 2)]))

    def test_no_bishop_between_attacking_pairs(self):
        """
        b 0 b
        b 0 b
        b 0 b
        """
        self.assertEqual(2, count_attacking_pairs([(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2)]))
    
    def test_no_bishop_between_attacking_pairs2(self):
        """
        b 0 0 0 0
        0 0 b 0 0
        0 0 b 0 0
        0 0 0 0 0
        b 0 0 0 0
        """
        self.assertEqual(2, count_attacking_pairs([(0, 0), (1, 2), (2, 2), (4, 0)]))

    def test_has_bishop_between_attacking_pairs(self):
        """
        b 0 b
        0 b 0
        b 0 b
        """
        self.assertEqual(6, count_attacking_pairs([(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Dec 9, 2021  \[Easy\] Swap Even and Odd Nodes
---
> **Question:** Given the head of a singly linked list, swap every two nodes and return its head.
>
> **Note:** Make sure it’s acutally nodes that get swapped not value.

**Example:**
```py
Given 1 -> 2 -> 3 -> 4, return 2 -> 1 -> 4 -> 3.
```

**Solution:** [https://replit.com/@trsong/Swap-Every-Even-and-Odd-Nodes-in-Linked-List-3](https://replit.com/@trsong/Swap-Every-Even-and-Odd-Nodes-in-Linked-List-3)
```py
import unittest

def swap_list(lst):
    prev = dummy = ListNode(-1, lst)
    while prev and prev.next and prev.next.next:
        first = prev.next
        second = first.next
        third = second.next

        prev.next = second
        second.next = first
        first.next = third
        prev = first
    return dummy.next


class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


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

    def test_five_elems_list(self):
        # 1 -> 2 -> 3 -> 4 -> 5
        n5 = ListNode(5)
        n4 = ListNode(4, n5)
        n3 = ListNode(3, n4)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)
        self.assert_lists(swap_list(n1), [n2, n1, n4, n3, n5])


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Dec 8, 2021 \[Medium\] Lazy Bartender
---
> **Question:** At a popular bar, each customer has a set of favorite drinks, and will happily accept any drink among this set. 
>
> For example, in the following situation, customer 0 will be satisfied with drinks 0, 1, 3, or 6.

```py
preferences = {
    0: [0, 1, 3, 6],
    1: [1, 4, 7],
    2: [2, 4, 7, 5],
    3: [3, 2, 5],
    4: [5, 8]
}
```

> A lazy bartender working at this bar is trying to reduce his effort by limiting the drink recipes he must memorize. 
>
> Given a dictionary input such as the one above, return the fewest number of drinks he must learn in order to satisfy all customers.
>
> For the input above, the answer would be 2, as drinks 1 and 5 will satisfy everyone.

**My thoughts:** This problem is a famous NP-Complete problem: SET-COVER. Therefore no better solution except brutal-force can be applied. Although there exists a log-n approximation algorithm (sort and pick drinks loved by minority), still that is not optimal.

**Solution with Backtracking:** [https://replit.com/@trsong/Lazy-Bartender-Problem-3](https://replit.com/@trsong/Lazy-Bartender-Problem-3)
```py
import unittest

def solve_lazy_bartender(preferences):
    drink_map = {}
    for customer, drinks in preferences.items():
        if not drinks:
            continue
        for drink in drinks:
            drink_map[drink] = drink_map.get(drink, set())
            drink_map[drink].add(customer)
    
    class Context:
        min_cover = len(drink_map)
    
    all_alcoholics = set(customer for customer, drinks in preferences.items() if drinks)
    
    def backtrack(selected_drinks, remaining_drinks):
        customers_by_drink = map(lambda drink: drink_map[drink], selected_drinks)
        covered_alcoholics = set.union(*customers_by_drink) if customers_by_drink else set()

        if covered_alcoholics == all_alcoholics:
            Context.min_cover = min(Context.min_cover, len(selected_drinks))
        else:
            for i, drink in enumerate(remaining_drinks):
                if drink_map[drink] in covered_alcoholics:
                    continue
                selected_drinks.append(drink)
                updated_remaining_drinks = remaining_drinks[:i] + remaining_drinks[i + 1:]
                backtrack(selected_drinks, updated_remaining_drinks)
                selected_drinks.pop()

    
    backtrack([], drink_map.keys())
    return Context.min_cover


class SolveLazyBartenderSpec(unittest.TestCase):
    def test_example(self):
        preferences = {
            0: [0, 1, 3, 6],
            1: [1, 4, 7],
            2: [2, 4, 7, 5],
            3: [3, 2, 5],
            4: [5, 8]
        }
        self.assertEqual(2, solve_lazy_bartender(preferences))  # drink 1 and 5 

    def test_empty_preference(self):
        self.assertEqual(0, solve_lazy_bartender({}))
    
    def test_non_alcoholic(self):
        preferences = {
            2: [],
            5: [],
            7: [10, 100]
        }
        self.assertEqual(1, solve_lazy_bartender(preferences))  # 10

    def test_has_duplicated_drinks_in_preference(self):
        preferences = {
            0: [3, 7, 5, 2, 9],
            1: [5],
            2: [2, 3],
            3: [4],
            4: [3, 4, 3, 5, 7, 9]
        }
        self.assertEqual(3, solve_lazy_bartender(preferences))  # drink 3, 4 and 5

    def test_should_return_optimal_solution(self):
        preferences = {
            1: [1, 3],
            2: [2, 3],
            3: [1, 3],
            4: [1, 3],
            5: [2]
        }
        self.assertEqual(2, solve_lazy_bartender(preferences))  # drink 2, 3

    def test_greedy_solution_not_work(self):
        preferences = {
            1: [1, 4],
            2: [1, 2, 5],
            3: [2, 4],
            4: [2, 5],
            5: [2, 4],
            6: [3, 5],
            7: [3, 4],
            8: [3, 5],
            9: [3, 4],
            10: [3, 5],
            11: [3, 4],
            12: [3, 5],
            13: [3, 4, 5]
        }
        self.assertEqual(2, solve_lazy_bartender(preferences))  # drink 4, 5

    def test_greedy_solution_not_work2(self):
        preferences = {
            0: [0, 3],
            1: [1, 4],
            2: [5, 6],
            3: [4, 5],
            4: [3, 5],
            5: [2, 6]
        }
        self.assertEqual(3, solve_lazy_bartender(preferences))  # drink 3, 4, 6


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 7, 2021 LC 287 \[Medium\] Find the Duplicate Number
---
> **Question:** You are given an array of length `n + 1` whose elements belong to the set `{1, 2, ..., n}`. By the pigeonhole principle, there must be a duplicate. Find it in linear time and space.


**My thoughts:** Use value as the 'next' element index which will form a loop evently. 

Why? Because the following scenarios will happen:

**Scenario 1:** If `a[i] != i for all i`, then since a[1] ... a[n] contains elements 1 to n, each time when interate to next index, one of the element within range 1 to n will be removed until no element is available and/or hit a previous used element and form a loop.  

**Scenario 2:** If `a[i] == i for all i > 0`, then as `a[0] != 0`, we will have a loop 0 -> a[0] -> a[0]

**Scenario 3:** If `a[i] == i for some i > 0`, then like scenario 2 we either we hit i evently or like scenario 1, for each iteration, we consume one element between 1 to n until all elements are used up and form a cycle. 

So we can use a fast and slow pointer to find the element when loop begins. 

**Solution with Fast-Slow Pointers:** [https://replit.com/@trsong/Find-the-Duplicate-Number-from-Array-2](https://replit.com/@trsong/Find-the-Duplicate-Number-from-Array-2)
```py
import unittest

def find_duplicate(nums):
    slow = fast = 0
    while True:
        fast = nums[nums[fast]]
        slow = nums[slow]
        if slow == fast:
            break

    p = 0
    while p != slow:
        p = nums[p]
        slow = nums[slow]
    return p
     

class FindDuplicateSpec(unittest.TestCase):
    def test_all_numbers_are_same(self):
        self.assertEqual(2, find_duplicate([2, 2, 2, 2, 2]))

    def test_number_duplicate_twice(self):
        # index: 0 1 2 3 4 5 6
        # value: 2 6 4 1 3 1 5
        # chain: 0 -> 2 -> 4 -> 3 -> 1 -> 6 -> 5 -> 1
        #                            ^              ^
        self.assertEqual(1, find_duplicate([2, 6, 4, 1, 3, 1, 5]))

    def test_rest_of_element_form_a_loop(self):
        # index: 0 1 2 3 4
        # value: 3 1 3 4 2
        # chain: 0 -> 3 -> 4 -> 2 -> 3
        #             ^              ^
        self.assertEqual(3, find_duplicate([3, 1, 3, 4, 2]))

    def test_rest_of_element_are_sorted(self):
        # index: 0 1 2 3 4
        # value: 4 1 2 3 4
        # chain: 0 -> 4 -> 4
        #             ^    ^
        self.assertEqual(4, find_duplicate([4, 1, 2, 3, 4]))
    
    def test_number_duplicate_more_than_twice(self):
        # index: 0 1 2 3 4 5 6 7 8 9
        # value: 2 5 9 6 9 3 8 9 7 1
        # chain: 0 -> 2 -> 9 -> 1 -> 5 -> 3 -> 6 -> 8 -> 7 -> 9
        #                  ^                                  ^
        self.assertEqual(9, find_duplicate([2, 5, 9, 6, 9, 3, 8, 9, 7, 1]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Dec 6, 2021 \[Easy\] Valid UTF-8 Encoding
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


**Solution:** [https://replit.com/@trsong/Valid-UTF-8-Encoding-2](https://replit.com/@trsong/Valid-UTF-8-Encoding-2)
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


### Dec 5, 2021 \[Easy\] Intersection of Linked Lists
---
> **Question:** You are given two singly linked lists. The lists intersect at some node. Find, and return the node. Note: the lists are non-cyclical.

**Example:**
```py
A = 1 -> 2 -> 3 -> 4
B = 6 -> 3 -> 4
# This should return 3 (you may assume that any nodes with the same value are the same node)
```

**Solution:** [https://replit.com/@trsong/Intersection-of-Linked-Lists-2](https://replit.com/@trsong/Intersection-of-Linked-Lists-2)
```py
import unittest

def intersection(l1, l2):
    len1, len2 = count_length(l1), count_length(l2)
    if len1 < len2:
        l1, l2 = l2, l1
        len1, len2 = len2, len1

    for _ in range(len1 - len2):
        l1 = l1.next

    while l1.val != l2.val:
        l1 = l1.next
        l2 = l2.next
    return l1.val


def count_length(lst):
    res = 0
    while lst:
        res += 1
        lst = lst.next
    return res


class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return "%d -> %s" % (self.val, str(self.next))

    @staticmethod
    def List(*vals):
        p = dummy = ListNode(-1)
        for v in vals:
            p.next = ListNode(v)
            p = p.next
        return dummy.next


class IntersectionSpec(unittest.TestCase):
    def test_example(self):
        l1 = ListNode.List(1, 2, 3, 4)
        l2 = ListNode.List(6, 3, 4)
        self.assertEqual(3, intersection(l1, l2))

    def test_intersect_at_last_elem(self):
        l1 = ListNode.List(1, 2, 3, 4)
        l2 = ListNode.List(4)
        self.assertEqual(4, intersection(l1, l2))

    def test_intersect_at_first_elem(self):
        l1 = ListNode.List(1, 2, 3, 4)
        l2 = ListNode.List(0, 1, 2, 3, 4)
        self.assertEqual(1, intersection(l1, l2))

    def test_same_list(self):
        l1 = ListNode.List(1, 2, 3, 4)
        l2 = ListNode.List(1, 2, 3, 4)
        self.assertEqual(1, intersection(l1, l2))

    def test_same_list2(self):
        l1 = ListNode.List(1)
        l2 = ListNode.List(1)
        self.assertEqual(1, intersection(l1, l2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 4, 2021 \[Easy\] Record the Last N Orders
--- 
> **Question:** You run an e-commerce website and want to record the last `N` order ids in a log. Implement a data structure to accomplish this, with the following API:
>
> - `record(order_id)`: adds the order_id to the log
> - `get_last(i)`: gets the ith last element from the log. `i` is guaranteed to be smaller than or equal to `N`.
> 
> You should be as efficient with time and space as possible.

**Solution:** [https://replit.com/@trsong/Record-the-Last-N-Orders-2](https://replit.com/@trsong/Record-the-Last-N-Orders-2)
```py
import unittest

class ECommerceLogService(object):
    def __init__(self, n):
        self.logs = [None] * n
        self.next_index = 0

    def record(self, order_id):
        self.logs[self.next_index] = order_id
        self.next_index = (self.next_index + 1) % len(self.logs)

    def get_last(self, i):
        index = (self.next_index - i) % len(self.logs)
        if self.logs[index] is None:
            raise ValueError('Log %d does not exists' % i)
        else:
            return self.logs[index]


class ECommerceLogServiceSpec(unittest.TestCase):
    def test_throw_exception_when_acessing_log_not_exists(self):
        service = ECommerceLogService(10)
        self.assertRaises(ValueError, service.get_last, 1)

    def test_access_last_element(self):
        service = ECommerceLogService(1)
        service.record(42)
        self.assertEqual(42, service.get_last(1))

    def test_acess_fist_element(self):
        service = ECommerceLogService(3)
        service.record(1)
        service.record(2)
        service.record(3)
        self.assertEqual(1, service.get_last(3))

    def test_overwrite_previous_record(self):
        service = ECommerceLogService(1)
        service.record(1)
        service.record(2)
        service.record(3)
        self.assertEqual(3, service.get_last(1))

    def test_overwrite_previous_record2(self):
        service = ECommerceLogService(2)
        service.record(1)
        service.record(2)
        service.record(3)
        service.record(4)
        self.assertEqual(3, service.get_last(2))

    def test_overwrite_previous_record3(self):
        service = ECommerceLogService(3)
        service.record(1)
        service.record(2)
        service.record(3)
        service.record(4)
        self.assertEqual(3, service.get_last(2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 3, 2021 LC 140 \[Hard\] Word Break II
---
> **Question:**  Given a string s and a dictionary of strings wordDict, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in any order.
>
> Note that the same word in the dictionary may be reused multiple times in the segmentation.

**Example 1:**
```py
Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
Output: ["cats and dog","cat sand dog"]
```

**Example 2:**
```py
Input: s = "pineapplepenapple", wordDict = ["apple","pen","applepen","pine","pineapple"]
Output: ["pine apple pen apple","pineapple pen apple","pine applepen apple"]
Explanation: Note that you are allowed to reuse a dictionary word.
```

**Example 3:**
```py
Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: []
```


**Solution with Backtracking:** [https://replit.com/@trsong/Word-Break-II](https://replit.com/@trsong/Word-Break-II)
```py
import unittest

def word_break(s, word_dict):
    cache = {}
    backtrack(s, word_dict, cache)
    return cache[s]


def backtrack(s, word_dict, cache):
    if s in cache:
        return cache[s]

    if not s:
        return ['']
    
    res = []
    for size in range(len(s) + 1):
        prefix = s[:size]
        if prefix in word_dict:
            sufficies = backtrack(s[size:], word_dict, cache)
            for suffix in sufficies:
                if suffix:
                    res.append(prefix + ' ' + suffix)
                else:
                    res.append(prefix)
    cache[s] = res
    return res


class WordBreakSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(sorted(expected), sorted(result))

    def test_example(self):
        s = 'catsanddog'
        word_dict = ['cat', 'cats', 'and', 'sand', 'dog']
        expected = ['cats and dog', 'cat sand dog']
        self.assert_result(expected, word_break(s, word_dict))

    def test_example2(self):
        s = 'pineapplepenapple'
        word_dict = ['apple', 'pen', 'applepen', 'pine', 'pineapple']
        expected = [
            'pine apple pen apple',
            'pineapple pen apple',
            'pine applepen apple'
        ]
        self.assert_result(expected, word_break(s, word_dict))

    def test_example3(self):
        s = 'catsandog'
        word_dict = ['cats', 'dog', 'sand', 'and', 'cat']
        expected = []
        self.assert_result(expected, word_break(s, word_dict))

    def test_dictionary_with_prefix_words(self):
        s = 'aaa'
        word_dict = ['a', 'aa']
        expected = ['a a a', 'a aa', 'aa a']
        self.assert_result(expected, word_break(s, word_dict))

    def test_dictionary_with_prefix_words2(self):
        s = 'aaaaa'
        word_dict = ['aa', 'aaa']
        expected = ['aa aaa', 'aaa aa']
        self.assert_result(expected, word_break(s, word_dict))

    def test_dictionary_with_prefix_words3(self):
        s = 'aaaaaa'
        word_dict = ['aa', 'aaa']
        expected = ['aaa aaa', 'aa aa aa']
        self.assert_result(expected, word_break(s, word_dict))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Dec 2, 2021 \[Hard\] Word Concatenation
---
> **Question:** Given a set of words, find all words that are concatenations of other words in the set.

**Example:**
```py
Input: ['rat', 'cat', 'cats', 'dog', 'catsdog', 'dogcat', 'dogcatrat']
Output: ['catsdog', 'dogcat', 'dogcatrat']
```

**My thoughts:** Imagine we have a function `word_break(s, word_dict)` which can tell if `s` can be broken into smaller combination of words from `word_dict`. 

Also note that longer words can only be any combinations of smaller words. So if we sort the words in ascending order based on word length, we can easily tell if a word is breakable by calling `word_break(s, all_smaller_word_dict)`.

**Solution with DP:** [https://replit.com/@trsong/Word-Concatenation](https://replit.com/@trsong/Word-Concatenation)
```py
import unittest

def find_all_concatenated_words(words):
    words.sort(key=len)
    word_dict = set()
    res = []
    for word in words:
        if word_break(word, word_dict):
            res.append(word)
        word_dict.add(word)
    return res


def word_break(s, word_dict):
    if not word_dict:
        return False

    n = len(s)
    # Let dp[i] indicates whether s[:i] is breakable
    # dp[i] = True if exists j < i st. dp[j] is True and s[i:j] is a word_dict word
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j: i] in word_dict:
                dp[i] = True
                break
    return dp[n]
    

class FindAllConcatenatedWordSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(sorted(expected), sorted(result))

    def test_example(self):
        words = ['rat', 'cat', 'cats', 'dog', 'catsdog', 'dogcat', 'dogcatrat']
        expected = ['catsdog', 'dogcat', 'dogcatrat']
        self.assert_result(expected, find_all_concatenated_words(words))

    def test_example2(self):
        words = ['cat','cats','catsdogcats','dog','dogcatsdog','hippopotamuses','rat','ratcatdogcat']
        expected = ['catsdogcats','dogcatsdog','ratcatdogcat']
        self.assert_result(expected, find_all_concatenated_words(words))
        self.assert_result(expected, find_all_concatenated_words(words))

    def test_example3(self):
        words = ["cat","dog","catdog"]
        expected = ['catdog']
        self.assert_result(expected, find_all_concatenated_words(words))

    def test_empty_array(self):
        self.assert_result([], find_all_concatenated_words([]))

    def test_one_word_array(self):
        self.assert_result([], find_all_concatenated_words(['abc']))

    def test_array_with_substrings(self):
        self.assert_result([], find_all_concatenated_words(['a', 'ab', 'abc']))

    def test_word_reuse(self):
        words = ['a', 'aa', 'aaa']
        expected = ['aa', 'aaa']
        self.assert_result(expected, find_all_concatenated_words(words))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Dec 1, 2021 \[Medium\] Minimum Number of Jumps to Reach End
---
> **Question:** You are given an array of integers, where each element represents the maximum number of steps that can be jumped going forward from that element. 
> 
> Write a function to return the minimum number of jumps you must take in order to get from the start to the end of the array.
>
> For example, given `[6, 2, 4, 0, 5, 1, 1, 4, 2, 9]`, you should return `2`, as the optimal solution involves jumping from `6 to 5`, and then from `5 to 9`.

**My thoughts:** Instead of using DP to calculate min step required to reach current index, we can treat this problem as climbing floor with ladders. For each floor you reach, you will get a new ladder with length `i + step[i]`. Now all you need to do is to greedily use the max length ladder you have seen so far and swap to the next one when the current one reaches end. The answer will be the total number of max length ladder you have used. 


**Greedy Solution:** [https://replit.com/@trsong/Calculate-Minimum-Number-of-Jumps-to-Reach-End-2](https://replit.com/@trsong/Calculate-Minimum-Number-of-Jumps-to-Reach-End-2)
```py
import unittest

def min_jump_to_reach_end(steps):
    if not steps:
        return None

    max_ladder = 0
    current_ladder = 0
    res = 0
    for i, step in enumerate(steps):
        if max_ladder < i:
            return None

        if current_ladder < i:
            current_ladder = max_ladder
            res += 1
        next_ladder = i + step
        max_ladder = max(max_ladder, next_ladder)
    return res


class MinJumpToReachEndSpec(unittest.TestCase):
    def test_example(self):
        steps = [6, 2, 4, 0, 5, 1, 1, 4, 2, 9]
        expected = 2  # 6 -> 5 -> 9
        self.assertEqual(expected, min_jump_to_reach_end(steps))

    def test_empty_steps(self):
        self.assertIsNone(min_jump_to_reach_end([]))
    
    def test_trivial_case(self):
        self.assertEqual(0, min_jump_to_reach_end([0]))

    def test_multiple_ways_to_reach_end(self):
        steps = [1, 3, 5, 6, 8, 12, 17]
        expected = 3  # 1 -> 3 -> 5 -> 17
        self.assertEqual(expected, min_jump_to_reach_end(steps)) 

    def test_should_return_min_step_to_reach_end(self):
        steps = [1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9]
        expected = 3  # 1 -> 3 -> 9 -> 9
        self.assertEqual(expected, min_jump_to_reach_end(steps))

    def test_should_return_min_step_to_reach_end2(self):
        steps = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        expected = 4
        self.assertEqual(expected, min_jump_to_reach_end(steps))

    def test_should_return_min_step_to_reach_end3(self):
        steps = [1, 3, 6, 3, 2, 3, 6, 8, 9, 5]
        expected = 4  # 1 -> 3 -> 6 -> 9 -> 5
        self.assertEqual(expected, min_jump_to_reach_end(steps))

    def test_should_return_min_step_to_reach_end4(self):
        steps = [1, 3, 6, 1, 0, 9]
        expected = 3  # 1 -> 3 -> 6 -> 9
        self.assertEqual(expected, min_jump_to_reach_end(steps))

    def test_unreachable_end(self):
        steps = [1, -2, -3, 4, 8, 9, 11]
        self.assertIsNone(min_jump_to_reach_end(steps))

    def test_unreachable_end2(self):
        steps = [1, 3, 2, -11, 0, 1, 0, 0, -1]
        self.assertIsNone(min_jump_to_reach_end(steps))

    def test_reachable_end(self):
        steps = [1, 3, 6, 10]
        expected = 2  # 1 -> 3 -> 10
        self.assertEqual(expected, min_jump_to_reach_end(steps))

    def test_stop_in_the_middle(self):
        steps = [1, 2, 0, 0, 0, 1000, 1000]
        self.assertIsNone(min_jump_to_reach_end(steps))

    def test_stop_in_the_middle2(self):
        steps = [2, 1, 0, 9]
        self.assertIsNone(min_jump_to_reach_end(steps))

    def test_greedy_solution_fails(self):
        steps = [5, 3, 3, 3, 4, 2, 1, 1, 1]
        expected = 2  # 5 -> 4 -> 1
        self.assertEqual(expected, min_jump_to_reach_end(steps))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Nov 30, 2021  LC 236 \[Medium\] Lowest Common Ancestor of a Binary Tree
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

**Solution:** [https://replit.com/@trsong/Find-the-Lowest-Common-Ancestor-of-a-Given-Binary-Tree-2](https://replit.com/@trsong/Find-the-Lowest-Common-Ancestor-of-a-Given-Binary-Tree-2)
```py
import unittest

def find_lca(tree, n1, n2):    
    if not tree or tree == n1 or tree == n2:
        return tree

    left_res = find_lca(tree.left, n1, n2)
    right_res = find_lca(tree.right, n1, n2)

    if left_res and right_res:
        return tree
    else:
        return left_res or right_res


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return "NodeValue: " + str(self.val)


class FindLCASpec(unittest.TestCase):
    def setUp(self):
        """
             1
           /   \
          2     3
         / \   / \
        4   5 6   7
        """
        self.n4 = TreeNode(4)
        self.n5 = TreeNode(5)
        self.n6 = TreeNode(6)
        self.n7 = TreeNode(7)
        self.n2 = TreeNode(2, self.n4, self.n5)
        self.n3 = TreeNode(3, self.n6, self.n7)
        self.n1 = TreeNode(1, self.n2, self.n3)

    def test_both_nodes_on_leaves(self):
        self.assertEqual(self.n2, find_lca(self.n1, self.n4, self.n5))

    def test_both_nodes_on_leaves2(self):
        self.assertEqual(self.n3, find_lca(self.n1, self.n6, self.n7))
    
    def test_both_nodes_on_leaves3(self):
        self.assertEqual(self.n1, find_lca(self.n1, self.n4, self.n6))

    def test_nodes_on_different_levels(self):
        self.assertEqual(self.n2, find_lca(self.n1, self.n4, self.n2))
    
    def test_nodes_on_different_levels2(self):
        self.assertEqual(self.n2, find_lca(self.n1, self.n4, self.n2))
    
    def test_nodes_on_different_levels3(self):
        self.assertEqual(self.n1, find_lca(self.n1, self.n4, self.n1))

    def test_same_nodes(self):
        self.assertEqual(self.n2, find_lca(self.n1, self.n2, self.n2))
    
    def test_same_nodes2(self):
        self.assertEqual(self.n6, find_lca(self.n1, self.n6, self.n6))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Nov 29, 2021 LC 938 \[Easy\] Range Sum of BST
---
> **Question:** Given a binary search tree and a range `[a, b]` (inclusive), return the sum of the elements of the binary search tree within the range.

**Example:**
```py
Given the range [4, 9] and the following tree:

    5
   / \
  3   8
 / \ / \
2  4 6  10

return 23 (5 + 4 + 6 + 8).
```

**Solution with DFS:** [https://replit.com/@trsong/Find-Range-Sum-of-BST-2](https://replit.com/@trsong/Find-Range-Sum-of-BST-2)
```py
import unittest

def bst_range_sum(root, low, hi):
    stack = [root]
    res = 0
    while stack:
        cur = stack.pop()
        if not cur:
            continue
        if low <= cur.val <= hi:
            res += cur.val

        if cur.val <= hi:
            stack.append(cur.right)
        
        if low <= cur.val:
            stack.append(cur.left)
    return res
    

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class BSTRangeSumSpec(unittest.TestCase):
    def setUp(self):
        """
            5
           / \
          3   8
         / \ / \
        2  4 6  10
        """
        left = TreeNode(3, TreeNode(2), TreeNode(4))
        right = TreeNode(8, TreeNode(6), TreeNode(10))
        self.full_root = TreeNode(5, left, right)

        """
          1
         / \ 
        1   1
           / \
          1   1
        """
        right = TreeNode(1, TreeNode(1), TreeNode(1))
        self.ragged_root = TreeNode(1, TreeNode(1), right)

        """
                 15
             /        \
            7          23
           / \        /   \
          3   11     19   27
         / \    \   /    
        1   5   13 17  
        """
        ll = TreeNode(3, TreeNode(1), TreeNode(5))
        lr = TreeNode(11, right=TreeNode(13))
        l = TreeNode(7, ll, lr)
        rl = TreeNode(19, TreeNode(17))
        r = TreeNode(23, rl, TreeNode(27))
        self.large_root = TreeNode(15, l, r)

    def test_example(self):
        self.assertEqual(23, bst_range_sum(self.full_root, 4, 9))

    def test_empty_tree(self):
        self.assertEqual(0, bst_range_sum(None, 0, 1))

    def test_tree_with_unique_value(self):
        self.assertEqual(5, bst_range_sum(self.ragged_root, 1, 1))

    def test_no_elem_in_range(self):
        self.assertEqual(0, bst_range_sum(self.full_root, 7, 7))

    def test_no_elem_in_range2(self):
        self.assertEqual(0, bst_range_sum(self.full_root, 11, 20))

    def test_end_points_are_inclusive(self):
        self.assertEqual(36, bst_range_sum(self.full_root, 3, 10))

    def test_just_cover_root(self):
        self.assertEqual(5, bst_range_sum(self.full_root, 5, 5))

    def test_large_tree(self):
        # 71 = 3 + 5 + 7 + 11 + 13 + 15 + 17
        self.assertEqual(71, bst_range_sum(self.large_root, 2, 18)) 

    def test_large_tree2(self):
        self.assertEqual(55, bst_range_sum(self.large_root, 0, 16)) 


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Nov 28, 2021 LC 678 \[Medium\] Balanced Parentheses with Wildcard
---
> **Question:** You're given a string consisting solely of `(`, `)`, and `*`. `*` can represent either a `(`, `)`, or an empty string. Determine whether the parentheses are balanced.
>
> For example, `(()*` and `(*)` are balanced. `)*(` is not balanced.

**My thoughts:** The wildcard `*` can represents `-1`, `0`, `1`, thus `x` number of `"*"`s can represents range from `-x` to `x`. Just like how we check balance without wildcard, but this time balance is a range: the wildcard just make any balance number within the range become possible. While keep the balance range in mind, we need to make sure each time the range can never go below 0 to become unbalanced, ie. number of open parentheses less than close ones.  


**Solution:** [https://replit.com/@trsong/Determine-Balanced-Parentheses-with-Wildcardi-2](https://replit.com/@trsong/Determine-Balanced-Parentheses-with-Wildcard-2)
```py
import unittest

def balanced_parentheses(s):
    lower_balance = higher_balance = 0
    for ch in s:
        if ch == '(':
            lower_balance += 1
            higher_balance += 1
        elif ch == ')':
            lower_balance -= 1
            higher_balance -= 1
        else:
            lower_balance -= 1
            higher_balance += 1

        if higher_balance < 0:
            return False
        lower_balance = max(lower_balance, 0)
    return lower_balance == 0

    
class BalancedParentheseSpec(unittest.TestCase):
    def test_example(self):
        self.assertTrue(balanced_parentheses("(()*"))

    def test_example2(self):
        self.assertTrue(balanced_parentheses("(*)"))

    def test_example3(self):
        self.assertFalse(balanced_parentheses(")*("))

    def test_empty_string(self):
        self.assertTrue(balanced_parentheses(""))

    def test_contains_only_wildcard(self):
        self.assertTrue(balanced_parentheses("*"))

    def test_contains_only_wildcard2(self):
        self.assertTrue(balanced_parentheses("**"))

    def test_contains_only_wildcard3(self):
        self.assertTrue(balanced_parentheses("***"))

    def test_without_wildcard(self):
        self.assertTrue(balanced_parentheses("()(()()())"))

    def test_unbalanced_string(self):
        self.assertFalse(balanced_parentheses("()()())()"))

    def test_unbalanced_string2(self):
        self.assertFalse(balanced_parentheses("*(***))))))))****"))

    def test_unbalanced_string3(self):
        self.assertFalse(balanced_parentheses("()((((*"))

    def test_unbalanced_string4(self):
        self.assertFalse(balanced_parentheses("((**)))))*"))
    
    def test_without_open_parentheses(self):
        self.assertTrue(balanced_parentheses("*)**)*)*))"))
    
    def test_without_close_parentheses(self):
        self.assertTrue(balanced_parentheses("(*((*(*(**"))

    def test_wildcard_can_only_be_empty(self):
        self.assertFalse(balanced_parentheses("((*)(*))((*"))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Nov 27, 2021 \[Medium\] Merge K Sorted Lists
---
> **Question:** Given k sorted singly linked lists, write a function to merge all the lists into one sorted singly linked list.


**Solution with PriorityQueue:** [https://replit.com/@trsong/Merge-K-Sorted-Linked-Lists-2](https://replit.com/@trsong/Merge-K-Sorted-Linked-Lists-2)
```py
import unittest
from queue import PriorityQueue

def merge_k_sorted_lists(lists):
    pq = PriorityQueue()
    lst_ptr = lists
    while lst_ptr:
        sub_ptr = lst_ptr.val
        if sub_ptr:
            pq.put((sub_ptr.val, sub_ptr))
        lst_ptr = lst_ptr.next

    p = dummy = ListNode(-1)
    while not pq.empty():
        _, sub_ptr = pq.get()
        p.next = ListNode(sub_ptr.val)
        p = p.next
        sub_ptr = sub_ptr.next
        if sub_ptr:
            pq.put((sub_ptr.val, sub_ptr))
    return dummy.next
        

##################
# Testing Utility
##################
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    def __eq__(self, other):
        return other and self.val == other.val and self.next == other.next
    
    def __lt__(self, other):
        return other and self.val < other.val

    def __repr__(self):
        return "{} -> {}".format(str(self.val), str(self.next))

    @staticmethod
    def List(*vals):
        p = dummy = ListNode(-1)
        for v in vals:
            p.next = ListNode(v)
            p = p.next
        return dummy.next


class MergeKSortedListSpec(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(ListNode.List(), merge_k_sorted_lists(ListNode.List()))

    def test_list_contains_empty_sub_lists(self):
        lists = ListNode.List(
            ListNode.List(),
            ListNode.List(),
            ListNode.List(1), 
            ListNode.List(), 
            ListNode.List(2), 
            ListNode.List(0, 4))
        expected = ListNode.List(0, 1, 2, 4)
        self.assertEqual(expected, merge_k_sorted_lists(lists))

    def test_sub_lists_with_duplicated_values(self):
        lists = ListNode.List(
            ListNode.List(1, 1, 3), 
            ListNode.List(1), 
            ListNode.List(3), 
            ListNode.List(1, 2), 
            ListNode.List(2, 3), 
            ListNode.List(2, 2), 
            ListNode.List(3))
        expected = ListNode.List(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)
        self.assertEqual(expected, merge_k_sorted_lists(lists))

    def test_general_lists(self):
        lists = ListNode.List(
            ListNode.List(),
            ListNode.List(1, 4, 7, 15),
            ListNode.List(),
            ListNode.List(2),
            ListNode.List(0, 3, 9, 10),
            ListNode.List(8, 13),
            ListNode.List(),
            ListNode.List(11, 12, 14),
            ListNode.List(5, 6)
        )
        expected = ListNode.List(*range(16))
        self.assertEqual(expected, merge_k_sorted_lists(lists))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Nov 26, 2021 \[Medium\] Autocompletion
---
> **Question:**  Implement auto-completion. Given a large set of words (for instance 1,000,000 words) and then a single word prefix, find all words that it can complete to.

**Example:**
```py
class Solution:
  def build(self, words):
    # Fill this in.
    
  def autocomplete(self, word):
    # Fill this in.

s = Solution()
s.build(['dog', 'dark', 'cat', 'door', 'dodge'])
s.autocomplete('do')  # Return ['dog', 'door', 'dodge']
```

**Solution with Trie:** [https://replit.com/@trsong/Autocompletion-Problem](https://replit.com/@trsong/Autocompletion-Problem)
```py
import unittest

class Autocomplete:
    def __init__(self):
        self.trie = Trie()

    def build(self, words):
        for word in words:
            self.trie.insert(word)

    def run(self, word):
        return self.trie.search(word)


class Trie(object):
    def __init__(self):
        self.children = {}
        self.words = set()

    def insert(self, word):
        p = self
        for ch in word:
            p.words.add(word)
            p.children[ch] = p.children.get(ch, Trie())
            p = p.children[ch]
        p.words.add(word)

    def search(self, word):
        p = self
        for ch in word:
            if p is None or ch not in p.children:
                return []
            p = p.children[ch]
        return list(p.words)


class AutocompleteSpec(unittest.TestCase):
    def test_example(self):
        auto = Autocomplete()
        auto.build(['dog', 'dark', 'cat', 'door', 'dodge'])
        expected = ['dog', 'door', 'dodge']
        self.assertCountEqual(expected, auto.run('do'))

    def test_empty_prefix(self):
        auto = Autocomplete()
        auto.build(['dog', 'dark', 'cat', 'door', 'dodge'])
        expected = ['dog', 'dark', 'cat', 'door', 'dodge']
        self.assertCountEqual(expected, auto.run(''))

    def test_search_exact_word(self):
        auto = Autocomplete()
        auto.build(['a', 'aa', 'aaa'])
        expected = ['aaa']
        self.assertCountEqual(expected, auto.run('aaa'))

    def test_prefix_not_exist(self):
        auto = Autocomplete()
        auto.build(['a', 'aa', 'aaa'])
        expected = []
        self.assertCountEqual(expected, auto.run('aaabc'))

    def test_prefix_not_exist2(self):
        auto = Autocomplete()
        auto.build(['a', 'aa', 'aaa'])
        expected = []
        self.assertCountEqual(expected, auto.run('c'))

    def test_sentence_with_duplicates(self):
        auto = Autocomplete()
        auto.build(['a', 'aa', 'aa', 'aaa', 'aab', 'abb', 'aaa'])
        expected = ['aa', 'aaa', 'aab']
        self.assertCountEqual(expected, auto.run('aa'))

    def test_word_with_same_prefix(self):
        auto = Autocomplete()
        auto.build(['aaa', 'aa', 'a'])
        expected = ['a', 'aa', 'aaa']
        self.assertCountEqual(expected, auto.run('a'))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Nov 25, 2021 \[Easy\] Count Number of Unival Subtrees
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


**Solution:** [https://replit.com/@trsong/Count-Total-Number-of-Uni-val-Subtrees-2](https://replit.com/@trsong/Count-Total-Number-of-Uni-val-Subtrees-2)
```py
import unittest

def count_unival_subtrees(tree):
    return count_unival_subtrees_recur(tree)[0]


def count_unival_subtrees_recur(tree):
    if not tree:
        return 0, True

    left_res, is_left_unival = count_unival_subtrees_recur(tree.left)
    right_res, is_right_unival = count_unival_subtrees_recur(tree.right)
    is_current_unival = is_left_unival and is_right_unival
    if is_current_unival and tree.left and tree.left.val != tree.val:
        is_current_unival = False
    if is_current_unival and tree.right and tree.right.val != tree.val:
        is_current_unival = False
    current_count = left_res + right_res + (1 if is_current_unival else 0)
    return current_count, is_current_unival


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
    unittest.main(exit=False, verbosity=2)
```

### Nov 24, 2021 \[Hard\] Largest Sum of Non-adjacent Numbers
---

> **Question:** Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0 or negative.
>
> For example, `[2, 4, 6, 2, 5]` should return 13, since we pick 2, 6, and 5. `[5, 1, 1, 5]` should return 10, since we pick 5 and 5.
>
> Follow-up: Can you do this in O(N) time and constant space?

**Solution with DP:** [https://replit.com/@trsong/Find-Largest-Sum-of-Non-adjacent-Numbers-2](https://replit.com/@trsong/Find-Largest-Sum-of-Non-adjacent-Numbers-2)
```py
import unittest

def max_non_adj_sum(nums):
    # Let dp[i] represents max non-adj sum for num[:i]
    # dp[i] = max(dp[i-1], nums[i-1] + dp[i-2])
    # That is max_so_far = max(prev_max, prev_prev_max + cur)
    prev_prev_max = prev_max = 0
    for num in nums:
        max_so_far = max(prev_max, prev_prev_max + num)
        prev_prev_max = prev_max
        prev_max = max_so_far
    return prev_max


class MaxNonAdjSumSpec(unittest.TestCase):
    def test_example(self):
        nums = [2, 4, 6, 2, 5]
        expected = 13  # 2, 6, 5
        self.assertEqual(expected, max_non_adj_sum(nums))

    def test_example2(self):
        nums = [5, 1, 1, 5]
        expected = 10  # 5, 5
        self.assertEqual(expected, max_non_adj_sum(nums))

    def test_empty_array(self):
        self.assertEqual(0, max_non_adj_sum([]))

    def test_one_elem_array(self):
        nums = [42]
        expected = 42
        self.assertEqual(expected, max_non_adj_sum(nums))

    def test_one_elem_array2(self):
        nums = [-42]
        expected = 0
        self.assertEqual(expected, max_non_adj_sum(nums))

    def test_two_elem_array(self):
        nums = [2, 4]
        expected = 4
        self.assertEqual(expected, max_non_adj_sum(nums))

    def test_unique_element(self):
        nums = [1, 1, 1]
        expected = 2
        self.assertEqual(expected, max_non_adj_sum(nums))

    def test_unique_element2(self):
        nums = [1, 1, 1, 1, 1, 1]
        expected = 3
        self.assertEqual(expected, max_non_adj_sum(nums))

    def test_array_with_non_positive_elem(self):
        nums = [10, 0, 0, 0, 0, 10, 0, 0, 10]
        expected = 30  # 10, 10, 10
        self.assertEqual(expected, max_non_adj_sum(nums))

    def test_array_with_non_positive_elem2(self):
        nums = [2, 4, -1]
        expected = 4
        self.assertEqual(expected, max_non_adj_sum(nums))

    def test_array_with_non_positive_elem3(self):
        nums = [2, 4, -1, 0]
        expected = 4
        self.assertEqual(expected, max_non_adj_sum(nums))

    def test_array_with_non_positive_elem4(self):
        nums = [-1, -2, -3, -4]
        expected = 0
        self.assertEqual(expected, max_non_adj_sum(nums))

    def test_array_with_non_positive_elem5(self):
        nums = [1, -1, 1, -1]
        expected = 2
        self.assertEqual(expected, max_non_adj_sum(nums))

    def test_array_with_non_positive_elem6(self):
        nums = [1, -1, -1, -1]
        expected = 1  
        self.assertEqual(expected, max_non_adj_sum(nums))

    def test_array_with_non_positive_elem7(self):
        nums = [1, -1, -1, 1, 2]
        expected = 3  # 1, 2
        self.assertEqual(expected, max_non_adj_sum(nums))

    def test_array_with_non_positive_elem8(self):
        nums = [1, -1, -1, 2, 1]
        expected = 3  # 1, 2
        self.assertEqual(expected, max_non_adj_sum(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Nov 23, 2021 LC 696 \[Easy\] Count Binary Substrings
---
> **Question:** Give a binary string s, return the number of non-empty substrings that have the same number of 0's and 1's, and all the 0's and all the 1's in these substrings are grouped consecutively.
>
> Substrings that occur multiple times are counted the number of times they occur.

**Example 1:**
```py
Input: s = "00110011"
Output: 6
Explanation: There are 6 substrings that have equal number of consecutive 1's and 0's: "0011", "01", "1100", "10", "0011", and "01".
Notice that some of these substrings repeat and are counted the number of times they occur.
Also, "00110011" is not a valid substring because all the 0's (and 1's) are not grouped together.
```

**Example 2:**
```py
Input: s = "10101"
Output: 4
Explanation: There are 4 substrings: "10", "01", "10", "01" that have equal number of consecutive 1's and 0's.
```

**My thoughts:** Group by consecutive one's and zero's, like `110001111000000` becomes `[2, 3, 4, 5]`. Notice that each consecutive group `g[i]` and `g[i - 1]` can at most form `min(g[i], g[i - 1])` substrings, because we can only match equal number of zero's and one's like `01`, `0011`, `000111`, etc. and the smaller group becomes a bottleneck. Finally, we can scan through groups of zero's and one's and we will get final answer. 

**Solution:** [https://replit.com/@trsong/Count-Binary-Substrings](https://replit.com/@trsong/Count-Binary-Substrings)
```py
import unittest

def count_bin_substring(s):
    previous_group = current_group = 0
    res = 0
    n = len(s)
    for i in range(n + 1):
        if 0 < i < n and s[i] == s[i - 1]:
            current_group += 1
        else:
            res += min(previous_group, current_group)
            previous_group, current_group = current_group, 1
    return res


class CountBinSubstringSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(6, count_bin_substring('00110011'))

    def test_example2(self):
        self.assertEqual(4, count_bin_substring('10101'))

    def test_ascending_group_numbers(self):
        self.assertEqual(9, count_bin_substring('110001111000000'))

    def test_descending_group_numbers(self):
        self.assertEqual(6, count_bin_substring('0000111001'))

    def test_empty_input(self):
        self.assertEqual(0, count_bin_substring(''))

    def test_unique_number(self):
        self.assertEqual(0, count_bin_substring('0000000000'))

    def test_even_number_of_ones_and_zeros(self):
        self.assertEqual(3, count_bin_substring('000111'))

    def test_edge_case(self):
        self.assertEqual(1, count_bin_substring('0011'))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Nov 22, 2021 LC 1647 \[Medium\] Minimum Deletions to Make Character Frequencies Unique
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

**Greedy Solution:** [https://replit.com/@trsong/Minimum-Deletions-to-Make-Character-Frequencies-Unique-2](https://replit.com/@trsong/Minimum-Deletions-to-Make-Character-Frequencies-Unique-2)
```py
import unittest

def min_deletions(s):
    histogram = {}
    for ch in s:
        histogram[ch] = histogram.get(ch, 0) + 1

    next_count = float('inf')
    res = 0
    for count in sorted(histogram.values(), reverse=True):
        if count <= next_count:
            next_count = count - 1
        else:
            res += count - next_count
            next_count -= 1
        next_count = max(next_count, 0)
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
    unittest.main(exit=False, verbosity=2)
```

### Nov 21, 2021 \[Easy\] Min Steps to Make Piles Equal Height
---
> **Question:** Alexa is given n piles of equal or unequal heights. In one step, Alexa can remove any number of boxes from the pile which has the maximum height and try to make it equal to the one which is just lower than the maximum height of the stack. Determine the minimum number of steps required to make all of the piles equal in height.

**Example:**
```py
Input: piles = [5, 2, 1]
Output: 3
Explanation:
Step 1: reducing 5 -> 2 [2, 2, 1]
Step 2: reducing 2 -> 1 [2, 1, 1]
Step 3: reducing 2 -> 1 [1, 1, 1]
So final number of steps required is 3.
```

**Solution with Max Heap:** [https://replit.com/@trsong/Min-Steps-to-Make-Piles-Equal-Height](https://replit.com/@trsong/Min-Steps-to-Make-Piles-Equal-Height)
```py
import unittest
from queue import PriorityQueue

def min_step_remove_piles(piles):
    histogram = {}
    for height in piles:
        histogram[height] = histogram.get(height, 0) + 1

    max_heap = PriorityQueue()
    for height, count in histogram.items():
        max_heap.put((-height, count))

    prev_count = 0
    res = 0
    while not max_heap.empty():
        res += prev_count
        _, count = max_heap.get()
        prev_count += count
    return res


class MinStepRemovePileSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(3, min_step_remove_piles([5, 2, 1]))

    def test_one_pile(self):
        self.assertEqual(0, min_step_remove_piles([42]))

    def test_same_height_piles(self):
        self.assertEqual(0, min_step_remove_piles([42, 42]))

    def test_pile_with_duplicated_heights(self):
        self.assertEqual(2, min_step_remove_piles([4, 4, 8, 8]))

    def test_different_height_piles(self):
        self.assertEqual(6, min_step_remove_piles([4, 5, 5, 4, 2]))

    def test_different_height_piles2(self):
        self.assertEqual(6, min_step_remove_piles([4, 8, 16, 32]))

    def test_different_height_piles3(self):
        self.assertEqual(2, min_step_remove_piles([4, 8, 8]))

    def test_sorted_heights(self):
        self.assertEqual(9, min_step_remove_piles([1, 2, 2, 3, 3, 4]))

    def test_sorted_heights2(self):
        self.assertEqual(15, min_step_remove_piles([1, 1, 2, 2, 2, 3, 3, 3, 4, 4]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Nov 20, 2021 \[Easy\] Pascal's triangle
---
> **Question:** Pascal's triangle is a triangular array of integers constructed with the following formula:
>
> - The first row consists of the number 1.
> - For each subsequent row, each element is the sum of the numbers directly above it, on either side.
>
> For example, here are the first few rows:
```py
    1
   1 1
  1 2 1
 1 3 3 1
1 4 6 4 1
```
> Given an input k, return the kth row of Pascal's triangle.
>
> Bonus: Can you do this using only O(k) space?

**Solution:** [https://replit.com/@trsong/Pascals-triangle](https://replit.com/@trsong/Pascals-triangle)
```py
import unittest

def pascal_triangle(k):
    res = [0] * k
    res[0] = 1
    for r in range(2, k + 1):
        prev_num = 0
        for i in range(r):
            prev_num, res[i] = res[i], res[i] + prev_num
    return res


class PascalTriangleSpec(unittest.TestCase):
    def test_1st_row(self):
        self.assertEqual([1], pascal_triangle(1))

    def test_2nd_row(self):
        self.assertEqual([1, 1], pascal_triangle(2))

    def test_3rd_row(self):
        self.assertEqual([1, 2, 1], pascal_triangle(3))

    def test_4th_row(self):
        self.assertEqual([1, 3, 3, 1], pascal_triangle(4))

    def test_5th_row(self):
        self.assertEqual([1, 4, 6, 4, 1], pascal_triangle(5))

    def test_6th_row(self):
        self.assertEqual([1, 5, 10, 10, 5, 1], pascal_triangle(6))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Nov 19, 2021 LC 240 \[Medium\] Search a 2D Matrix II
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

**Divide and Conquer Solution**: [https://replit.com/@trsong/Search-in-a-Sorted-2D-Matrix-2](https://replit.com/@trsong/Search-in-a-Sorted-2D-Matrix-2)
```py
import unittest

def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False

    stack = [(0, len(matrix) - 1, 0, len(matrix[0]) - 1)]
    while stack:
        rlo, rhi, clo, chi = stack.pop()
        if rlo > rhi or clo > chi:
            continue

        rmid = rlo + (rhi - rlo) // 2
        cmid = clo + (chi - clo) // 2
        if matrix[rmid][cmid] == target:
            return True
        elif matrix[rmid][cmid] < target:
            # taget cannot exist in top-left
            stack.append((rlo, rmid, cmid + 1, chi))  # top-right
            stack.append((rmid + 1, rhi, clo, chi))   # buttom
        else:
            # target cannot exist in bottom-right
            stack.append((rmid, rhi, clo, cmid - 1))  # bottom-left
            stack.append((rlo, rmid - 1, clo, chi))  # top
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


### Nov 18, 2021 LC 54 \[Medium\] Spiral Matrix 
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

**Solution:** [https://replit.com/@trsong/Spiral-Matrix-Traversal-3](https://replit.com/@trsong/Spiral-Matrix-Traversal-3)
```py
import unittest

def spiral_order(matrix):
    if not matrix or not matrix[0]:
        return []

    rlo = 0
    rhi = len(matrix) - 1
    clo = 0
    chi = len(matrix[0]) - 1

    res = []
    while rlo <= rhi and clo <= chi:
        r = rlo
        c = clo
        while c < chi:
            res.append(matrix[r][c])
            c += 1
        chi -= 1

        while r < rhi:
            res.append(matrix[r][c])
            r += 1
        rhi -= 1

        if rlo > rhi or clo > chi:
            res.append(matrix[r][c])
            break

        while c > clo:
            res.append(matrix[r][c])
            c -= 1
        clo += 1

        while r > rlo:
            res.append(matrix[r][c])
            r -= 1
        rlo += 1
    return res


class SpiralOrderSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual([1, 2, 3, 6, 9, 8, 7, 4, 5], spiral_order([
            [ 1, 2, 3 ],
            [ 4, 5, 6 ],
            [ 7, 8, 9 ]
        ]))

    def test_example2(self):
        self.assertEqual([1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7], spiral_order([
            [1,  2,  3,  4],
            [5,  6,  7,  8],
            [9, 10, 11, 12]
        ]))

    def test_empty_table(self):
        self.assertEqual([], spiral_order([]))
        self.assertEqual([], spiral_order([[]]))

    def test_two_by_two_table(self):
        self.assertEqual([1, 2, 3, 4], spiral_order([
            [1, 2],
            [4, 3]
        ]))

    def test_one_element_table(self):
        self.assertEqual([1], spiral_order([[1]]))

    def test_one_by_k_table(self):
        self.assertEqual([1, 2, 3, 4], spiral_order([
            [1, 2, 3, 4]
        ]))

    def test_k_by_one_table(self):
        self.assertEqual([1, 2, 3], spiral_order([
            [1],
            [2],
            [3]
        ]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Nov 17, 2021 LC 743 \[Medium\] Network Delay Time
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

**Solution with Uniform Cost Search (Dijkstra’s Algorithm):** [https://replit.com/@trsong/Find-Network-Delay-Time-3](https://replit.com/@trsong/Find-Network-Delay-Time-3)
```py
import unittest
import sys
from Queue import PriorityQueue

def max_network_delay(times, nodes):
    neighbors = [None] * (nodes + 1)
    for u, v, w in times:
        neighbors[u] = neighbors[u] or []
        neighbors[u].append((v, w))
    
    distance = [sys.maxint] * (nodes + 1)
    pq = PriorityQueue()
    pq.put((0, 0))

    while not pq.empty():
        cur_dist, cur = pq.get()
        if distance[cur] != sys.maxint:
            continue
        distance[cur] = cur_dist

        for v, w in neighbors[cur] or []:
            alt_dist = cur_dist + w
            if alt_dist < distance[v]:
                pq.put((alt_dist, v))

    res = max(distance)
    return res if res < sys.maxint else -1


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
    unittest.main(exit=False, verbosity=2)
```

### Nov 16, 2021 \[Medium\] Toss Biased Coin
---
> **Question:** Assume you have access to a function toss_biased() which returns 0 or 1 with a probability that's not 50-50 (but also not 0-100 or 100-0). You do not know the bias of the coin. Write a function to simulate an unbiased coin toss.

**Solution:** [https://replit.com/@trsong/Toss-Biased-Coins](https://replit.com/@trsong/Toss-Biased-Coins)
```py
from random import randint

def toss_unbiased():
    # Let P(T1, T2) represents probability to get T1, T2 in first and second toss: 
    # P(0, 0) = p * p
    # P(1, 1) = (1 - p) * (1 - p)
    # P(1, 0) = (1 - p) * p
    # P(0, 1) = p * (1 - p)
    # Notice that P(1, 0) = P(0, 1)
    while True:
        t1 = toss_biased()
        t2 = toss_biased()
        if t1 != t2:
            return t1
    

def toss_biased():
    # suppose the toss has 1/4 chance to get 0 and 3/4 to get 1
    return 0 if randint(0, 3) == 0 else 1


def print_distribution(repeat):
    histogram = {}
    for _ in range(repeat):
        res = toss_unbiased()
        if res not in histogram:
            histogram[res] = 0
        histogram[res] += 1
    print(histogram)


if __name__ == '__main__':
     # Distribution looks like {0: 99931, 1: 100069}
    print_distribution(repeat=200000)
```


### Nov 15, 2021 \[Easy\] Single Bit Switch
---
> **Question:** Given three 32-bit integers x, y, and b, return x if b is 1 and y if b is 0, using only mathematical or bit operations. You can assume b can only be 1 or 0.

**Solution:** [https://replit.com/@trsong/Create-a-single-bit-switch](https://replit.com/@trsong/Create-a-single-bit-switch)
```py
import unittest

def single_bit_switch(b, x, y):
    return b * x + (1 - b) * y


class SingleBitSwitchSpec(unittest.TestCase):
    def test_b_is_zero(self):
        x, y = 8, 16
        self.assertEqual(y, single_bit_switch(0, x, y))
        self.assertEqual(x, single_bit_switch(1, x, y))

    def test_b_is_one(self):
        x, y = 8, 16
        self.assertEqual(y, single_bit_switch(0, x, y))
        self.assertEqual(x, single_bit_switch(1, x, y))

    def test_negative_numbers(self):
        x, y = -1, -2
        self.assertEqual(y, single_bit_switch(0, x, y))
        self.assertEqual(x, single_bit_switch(1, x, y))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Nov 14, 2021 LC 790 \[Medium\] Domino and Tromino Tiling
---
> **Question:**  You are given a 2 x N board, and instructed to completely cover the board with the following shapes:
>
> - Dominoes, or 2 x 1 rectangles.
> - Trominoes, or L-shapes.
>
> Given an integer N, determine in how many ways this task is possible.
 
**Example:**
```py
if N = 4, here is one possible configuration, where A is a domino, and B and C are trominoes.

A B B C
A B C C
```

**Solution with DP:** [https://replit.com/@trsong/Domino-and-Tromino-Tiling-Problem](https://replit.com/@trsong/Domino-and-Tromino-Tiling-Problem)
```py
import unittest

def domino_tiling(n):
    if n <= 2:
        return n
    # Let f[n] represents # ways for 2 * n pieces:
    # f[1]: x 
    #       x
    #
    # f[2]: x x
    #       x x
    f = [0] * (n + 1)
    f[1] = 1 
    f[2] = 2

    # Let g[n] represents # ways for 2*n + 1 pieces:
    # g[1]: x      or   x x
    #       x x         x
    #
    # g[2]: x x    or   x x x  
    #       x x x       x x
    g = [0] * (n + 1)
    g[1] = 1
    g[2] = 2  # domino + tromino or tromino + domino

    # Pattern:
    # f[n]: x x x x = f[n-1]: x x x y  +  f[n-2]: x x y y  + g[n-2]: x x x y + g[n-2]: x x y y
    #       x x x x           x x x y             x x z z            x x y y           x x x y
    #
    # g[n]: x x x x x = f[n-1]: x x x y y + g[n-1]: x x y y 
    #       x x x x             x x x y             x x x
    for n in range(3, n + 1):
        g[n] = f[n-1] + g[n-1]
        f[n] = f[n-1] + f[n-2] + 2 * g[n-2]
    return f[n]


class DominoTilingSpec(unittest.TestCase):
    def test_empty_grid(self):
        self.assertEqual(0, domino_tiling(0))
        
    def test_size_one(self):
        """
        A
        A
        """
        self.assertEqual(1, domino_tiling(1))

    def test_size_two(self):
        """
        A B or A A
        A B    B B
        """
        self.assertEqual(2, domino_tiling(2))

    def test_size_three(self):
        """
        x x C 
        x x C    

        x C C  
        x B B

        x C C or x x C
        x x C    x C C
        """
        self.assertEqual(5, domino_tiling(3))

    def test_size_four(self):
        self.assertEqual(11, domino_tiling(4))

    def test_size_five(self):
        self.assertEqual(24, domino_tiling(5))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Nov 13, 2021 \[Medium\] Max Number of Edges Added to Tree to Stay Bipartite
---
> **Question:** Maximum number of edges to be added to a tree so that it stays a Bipartite graph
>
> A tree is always a Bipartite Graph as we can always break into two disjoint sets with alternate levels. In other words we always color it with two colors such that alternate levels have same color. The task is to compute the maximum no. of edges that can be added to the tree so that it remains Bipartite Graph.

**Example 1:**
```py
Input : Tree edges as vertex pairs 
        1 2
        1 3
Output : 0
Explanation :
The only edge we can add is from node 2 to 3.
But edge 2, 3 will result in odd cycle, hence 
violation of Bipartite Graph property.
```

**Example 2:**
```py
Input : Tree edges as vertex pairs 
        1 2
        1 3
        2 4
        3 5
Output : 2
Explanation : On colouring the graph, {1, 4, 5} 
and {2, 3} form two different sets. Since, 1 is 
connected from both 2 and 3, we are left with 
edges 4 and 5. Since, 4 is already connected to
2 and 5 to 3, only options remain {4, 3} and 
{5, 2}.
```

**My thoughts:** The maximum number of edges between set Black and set White equals `size(Black) * size(White)`. And we know the total number of edges in a tree. Thus, we can run BFS to color each node and then the remaining edges is just `size(Black) * size(White) - #TreeEdges`.

**Solution with BFS:** [https://replit.com/@trsong/Find-the-Max-Number-of-Edges-Added-to-Tree-to-Stay-Bipartite](https://replit.com/@trsong/Find-the-Max-Number-of-Edges-Added-to-Tree-to-Stay-Bipartite)
```py
import unittest

def max_edges_to_add(edges):
    if not edges:
        return 0

    neighbors = {}
    for u, v in edges:
        neighbors[u] = neighbors.get(u, set())
        neighbors[v] = neighbors.get(v, set())
        neighbors[u].add(v)
        neighbors[v].add(u)

    num_color = [0, 0]
    color = 0
    queue = [edges[0][0]]
    visited = set()

    while queue:
        for _ in range(len(queue)):
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            for nb in neighbors[cur]:
                queue.append(nb)
            num_color[color] += 1
        color = 1 - color
    return num_color[0] * num_color[1] - len(edges)


class MaxEdgesToAddSpec(unittest.TestCase):
    def test_example(self):
        """
          1
         / \
        2   3
        """
        edges = [(1, 2), (1, 3)]
        self.assertEqual(0, max_edges_to_add(edges))

    def test_example2(self):
        """
            1
           / \
          2   3
         /     \
        4       5
        """
        edges = [(1, 2), (1, 3), (2, 4), (3, 5)]
        self.assertEqual(2, max_edges_to_add(edges)) # (3, 4), (2, 5)

    def test_empty_tree(self):
        self.assertEqual(0, max_edges_to_add([]))

    def test_right_heavy_tree(self):
        """
         1
          \ 
           2
            \
             3
              \
               4
                \
                 5
        """
        edges = [(1, 2), (4, 5), (2, 3), (3, 4)]
        # White=[1, 3, 5]. Black=[2, 4]. #TreeEdge = 4. Max = #W * #B - #T = 3 * 2 - 4 = 2
        self.assertEqual(2, max_edges_to_add(edges))  # (1, 4), (2, 5)

    def test_general_tree(self):
        """
             1
           / | \ 
          2  3  4
         / \   /|\
        5   6 7 8 9 
         \     /   \
         10   11    12
        """
        edges = [(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (4, 7), (4, 8), (4, 9), (5, 10), (8, 11), (9, 12)]
        # White=[1, 5, 6, 7, 8, 9]. Black=[2, 3, 4, 10, 11, 12]. #TreeEdge=11. Max = #W * #B - #T = 6 * 6 - 11 = 25
        self.assertEqual(25, max_edges_to_add(edges))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Nov 12, 2021 LC 301 \[Hard\] Remove Invalid Parentheses
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

**Solution with Backtracking:** [https://replit.com/@trsong/Ways-to-Remove-Invalid-Parentheses-2](https://replit.com/@trsong/Ways-to-Remove-Invalid-Parentheses-2)
```py
import unittest

def remove_invalid_parenthese(s):
    invalid_open = invalid_close = 0
    for ch in s:
        if ch == ')' and invalid_open == 0:
            invalid_close += 1
        elif ch == '(':
            invalid_open += 1
        elif ch == ')':
            invalid_open -= 1
    res = []
    backtrack(res, s, 0, invalid_open, invalid_close)
    return res


def backtrack(res, s, next_index, invalid_open, invalid_close):
    if invalid_open == invalid_close == 0:
        if is_valid(s):
            res.append(s)
    else:
        for i in range(next_index, len(s)):
            if i > next_index and s[i] == s[i - 1]:
                continue
            elif s[i] == '(' and invalid_open > 0:
                backtrack(res, s[:i] + s[i + 1:], i, invalid_open - 1, invalid_close)
            elif s[i] == ')' and invalid_close > 0:
                backtrack(res, s[:i] + s[i + 1:], i, invalid_open, invalid_close - 1)


def is_valid(s):
    balance = 0
    for ch in s:
        if balance < 0:
            return False
        elif ch == '(':
            balance += 1
        elif ch == ')':
            balance -= 1
    return balance == 0



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
    unittest.main(exit=False, verbosity=2)
```

### Nov 11, 2021 \[Easy\] Smallest Sum Not Subset Sum
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


**Solution with Induction:** [https://replit.com/@trsong/Smallest-Sum-Not-Subset-Sum-2](https://replit.com/@trsong/Smallest-Sum-Not-Subset-Sum-2)
```py
import unittest

def smallest_non_subset_sum(nums):
    # Initially sum has range [0, 1)
    upper_bound = 1
    for num in nums:
        if num > upper_bound:
            break
        # Previously sum has range [0, upper_bound), 
        # with introduction of num, that range becomes
        # new range [0, upper_bound + num)
        upper_bound += num
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

### Nov 10, 2021 \[Medium\] Amazing Number
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
- Finally, if a number is neither too small nor too big, i.e. between `(0, n-1)`, then we can define "invalid" range as `[i - nums[i] + 1, i]`.

We accumlate those intervals by using interval counting technique: define interval_accu array, for each interval `(start, end)`, `interval_accu[start] += 1` and `interval_accu[end+1] -= 1` so that when we can make interval accumulation by `interval_accu[i] += interval_accu[i-1]` for all `i`. 

Find max amazing number is equivalent to find min overllaping of invalid intervals. We can find min number of overllaping intervals along the interval accumulation.



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

**Efficient Solution with Interval Count:** [https://replit.com/@trsong/Calculate-Amazing-Number](https://replit.com/@trsong/Calculate-Amazing-Number)
```py
import unittest

def max_amazing_number_index(nums):
    n = len(nums)
    interval_accumulation = [0] * n
    for i in range(n):
        for invalid_start, invalid_end in invalid_intervals_at(nums, i):
            interval_accumulation[invalid_start] += 1
            if invalid_end + 1 < n:
                interval_accumulation[invalid_end + 1] -= 1
    
    min_cut = interval_accumulation[0]
    min_cut_index = 0
    for i in range(1, n):
        interval_accumulation[i] += interval_accumulation[i-1]
        if interval_accumulation[i] < min_cut:
            min_cut = interval_accumulation[i]
            min_cut_index = i
    return min_cut_index 


def invalid_intervals_at(nums, i):
    # invalid zone starts from i - nums[i] + 1 and ends at i
    # 0 0 0 0 0 3 0 0 0 0 0 
    #       ^ ^ ^
    #      invalid
    n = len(nums)
    if 0 < nums[i] < n:
        start = (i - nums[i] + 1) % n
        end = i
        if start <= end:
            yield start, end
        else:
            yield 0, end
            yield start, n - 1


class MaxAmazingNumberIndexSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(0, max_amazing_number_index([0, 1, 2, 3]))  # max # amazing number = 4 at [0, 1, 2, 3]

    def test_example2(self):
        self.assertEqual(1, max_amazing_number_index([1, 0, 0]))  # max # amazing number = 3 at [0, 0, 1]

    def test_non_descending_array(self):
        self.assertEqual(0, max_amazing_number_index([0, 0, 0, 1, 2, 3]))  # max # amazing number = 0 at [0, 0, 0, 1, 2, 3]

    def test_random_array(self):
        self.assertEqual(1, max_amazing_number_index([1, 4, 3, 2]))  # max # amazing number = 2 at [4, 3, 2, 1]

    def test_non_ascending_array(self):
        self.assertEqual(2, max_amazing_number_index([3, 3, 2, 1, 0]))  # max # amazing number = 4 at [2, 1, 0, 3, 3]

    def test_return_smallest_index_when_no_amazing_number(self):
        self.assertEqual(0, max_amazing_number_index([99, 99, 99, 99]))  # max # amazing number = 0 thus return smallest possible index

    def test_negative_number(self):
        self.assertEqual(1, max_amazing_number_index([3, -99, -99, -99]))  # max # amazing number = 4 at [-1, -1, -1, 3])


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Nov 9, 2021 \[Medium\] Integer Division
---
> **Question:** Implement division of two positive integers without using the division, multiplication, or modulus operators. Return the quotient as an integer, ignoring the remainder.


**Solution:** [https://replit.com/@trsong/Divide-Two-Integers-2](https://replit.com/@trsong/Divide-Two-Integers-2)
```py
import unittest

def divide(dividend, divisor):
    INT_DIGITS = 32
    divisor_sign = -1 if divisor < 0 else 1
    sign = -1 if (dividend > 0) ^ (divisor > 0) else 1
    dividend, divisor = abs(dividend), abs(divisor)
    quotient = 0

    for i in xrange(INT_DIGITS, -1, -1):
        if dividend >= (divisor << i):
            quotient |= 1 << i
            dividend -= (divisor << i)

    if dividend == 0:
        return sign * quotient, 0
    elif sign > 0:
        return quotient, divisor_sign * dividend
    else:
        return -quotient-1, divisor_sign * (divisor-dividend)


class DivideSpec(unittest.TestCase):
    def test_example(self):
        dividend, divisor = 10, 3
        expected = 3, 1
        self.assertEqual(expected, divide(dividend, divisor))

    def test_product_is_zero(self):
        dividend, divisor = 0, 1
        expected = 0, 0
        self.assertEqual(expected, divide(dividend, divisor))

    def test_divisor_is_one(self):
        dividend, divisor = 42, 1
        expected = 42, 0
        self.assertEqual(expected, divide(dividend, divisor))

    def test_product_is_negative(self):
        dividend, divisor = -17, 3
        expected = -6, 1
        self.assertEqual(expected, divide(dividend, divisor))

    def test_divisor_is_negative(self):
        dividend, divisor = 42, -5
        expected = -9, -3
        self.assertEqual(expected, divide(dividend, divisor))

    def test_both_num_are_negative(self):
        dividend, divisor = -42, -5
        expected = 8, -2
        self.assertEqual(expected, divide(dividend, divisor))

    def test_product_is_divisible(self):
        dividend, divisor = 42, 3
        expected = 14, 0
        self.assertEqual(expected, divide(dividend, divisor))

    def test_product_is_divisible2(self):
        dividend, divisor = 42, -3
        expected = -14, 0
        self.assertEqual(expected, divide(dividend, divisor))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Nov 8, 2021 \[Hard\] Power Supply to All Cities
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

**My thoughts**: This question is an undirected graph problem asking for total cost of minimum spanning tree. Both of the follwing algorithms can solve this problem: Kruskal’s and Prim’s MST Algorithm. First one keeps choosing edges whereas second one starts from connecting vertices. Either one will work.

**Solution with Kruskal Algorithm:** [https://replit.com/@trsong/Design-Power-Supply-to-All-Cities-2#main.py](https://replit.com/@trsong/Design-Power-Supply-to-All-Cities-2)
```py
import unittest

def min_cost_power_supply(cities, cost_btw_cities):
    city_lookup = { city: index for index, city in enumerate(cities) }
    uf = DisjointSet(len(cities))
    res = 0

    cost_btw_cities.sort(key=lambda uvw: uvw[-1])
    for u, v, w in cost_btw_cities:
        uid, vid = city_lookup[u], city_lookup[v]
        if not uf.is_connected(uid, vid):
            uf.union(uid, vid)
            res += w
    return res


class DisjointSet(object):
    def __init__(self, size):
        self.parent = range(size)

    def find(self, p):
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def union(self, p1, p2):
        r1 = self.find(p1)
        r2 = self.find(p2)
        if r1 != r2:
            self.parent[r1] = r2
    
    def is_connected(self, p1, p2):
        return self.find(p1) == self.find(p2)


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
    unittest.main(exit=False, verbosity=2)
```

### Nov 7, 2021 LC 130 \[Medium\] Surrounded Regions
---

> **Question:**  Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.
> A region is captured by flipping all 'O's into 'X's in that surrounded region.
 
**Example:**
```py
X X X X
X O O X
X X O X
X O X X

After running your function, the board should be:

X X X X
X X X X
X X X X
X O X X

Explanation:
Surrounded regions shouldn’t be on the border, which means that any 'O' on the border of the board are not flipped to 'X'. Any 'O' that is not on the border and it is not connected to an 'O' on the border will be flipped to 'X'. Two cells are connected if they are adjacent cells connected horizontally or vertically.
```

**Solution with DisjointSet(Union-Find):** [https://replit.com/@trsong/Flip-Surrounded-Regions-2](https://replit.com/@trsong/Flip-Surrounded-Regions-2)
```py
import unittest

X, O = 'X', 'O'

def flip_region(grid):
    if not grid or not grid[0]:
        return grid
    
    n, m = len(grid), len(grid[0])
    uf = DisjointSet(n * m + 1)
    rc_to_pos = lambda r, c: r * m + c
    for r in range(n):
        for c in range(m):
            if grid[r][c] == 'X':
                continue
            if r > 0 and grid[r - 1][c] == 'O':
                uf.union(rc_to_pos(r - 1, c), rc_to_pos(r, c))
            if c > 0 and grid[r][c - 1] == 'O':
                uf.union(rc_to_pos(r, c - 1), rc_to_pos(r, c))

    edge_root = n * m
    for r in range(n):
        uf.union(rc_to_pos(r, 0), edge_root)
        uf.union(rc_to_pos(r, m - 1), edge_root)
    
    for c in range(m):
        uf.union(rc_to_pos(0, c), edge_root)
        uf.union(rc_to_pos(n - 1, c), edge_root)

    for r in range(n):
        for c in range(m):
            if (grid[r][c] == 'O' and 
                not uf.is_connected(rc_to_pos(r, c), edge_root)):
                grid[r][c] = 'X'
    return grid


class DisjointSet(object):
    def __init__(self, size):
        self.parent = [-1] * size

    def union(self, p1, p2):
        r1 = self.find(p1)
        r2 = self.find(p2)
        if r1 != r2:
            self.parent[r1] = r2

    def find(self, p):
        while self.parent[p] != -1:
            p = self.parent[p]
        return p

    def is_connected(self, p1, p2):
        return self.find(p1) == self.find(p2)


class FlipRegionSpec(unittest.TestCase):
    def test_example(self):
        grid = [
            [X, X, X, X], 
            [X, O, O, X], 
            [X, X, O, X], 
            [X, O, X, X]]
        expected = [
            [X, X, X, X], 
            [X, X, X, X], 
            [X, X, X, X], 
            [X, O, X, X]]
        self.assertEqual(expected, flip_region(grid))
    
    def test_empty_grid(self):
        self.assertEqual([], flip_region([]))
        self.assertEqual([[]], flip_region([[]]))

    def test_non_surrounded_region(self):
        grid = [
            [O, O, O, O, O], 
            [O, O, O, O, O],
            [O, O, O, O, O],
            [O, O, O, O, O]]
        expected = [
            [O, O, O, O, O],
            [O, O, O, O, O],
            [O, O, O, O, O],
            [O, O, O, O, O]]
        self.assertEqual(expected, flip_region(grid))

    def test_all_surrounded_region(self):
        grid = [
            [X, X, X], 
            [X, X, X]]
        expected = [
            [X, X, X], 
            [X, X, X]]
        self.assertEqual(expected, flip_region(grid))

    def test_region_touching_boundary(self):
        grid = [
            [X, O, X, X, O],
            [X, X, O, O, X],
            [X, O, X, X, O],
            [O, O, O, X, O]]
        expected = [
            [X, O, X, X, O],
            [X, X, X, X, X],
            [X, O, X, X, O],
            [O, O, O, X, O]]
        self.assertEqual(expected, flip_region(grid))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Nov 6, 2021 \[Hard\] Order of Alien Dictionary
--- 
> **Question:** You come across a dictionary of sorted words in a language you've never seen before. Write a program that returns the correct order of letters in this language.
>
> For example, given `['xww', 'wxyz', 'wxyw', 'ywx', 'ywz']`, you should return `['x', 'z', 'w', 'y']`.

**My thoughts:** As the alien letters are topologically sorted, we can just mimic what topological sort with numbers and try to find pattern.

Suppose the dictionary contains: `01234`. Then the words can be `023, 024, 12, 133, 2433`. Notice that we can only find the relative order by finding first unequal letters between consecutive words. eg.  `023, 024 => 3 < 4`.  `024, 12 => 0 < 1`.  `12, 133 => 2 < 3`

With relative relation, we can build a graph with each occurring letters being veteces and edge `(u, v)` represents `u < v`. If there exists a loop that means we have something like `a < b < c < a` and total order not exists. Otherwise we preform a topological sort to generate the total order which reveals the alien dictionary. 

As for implementation of topological sort, there are two ways, one is the following by constantly removing edges from visited nodes. The other is to [first DFS to find the reverse topological order then reverse again to find the result](https://trsong.github.io/python/java/2019/11/02/DailyQuestionsNov.html#nov-9-2019-hard-order-of-alien-dictionary). 


**Solution with Topological Sort:** [https://replit.com/@trsong/Alien-Dictionary-Order-2](https://replit.com/@trsong/Alien-Dictionary-Order-2)
```py
import unittest

def dictionary_order(sorted_words):
    neighbors = {}
    inward_degree = {}
    for i in range(1, len(sorted_words)):
        cur_word = sorted_words[i]
        prev_word = sorted_words[i - 1]
        for prev_ch, cur_ch in zip(prev_word, cur_word):
            if prev_ch != cur_ch:
                neighbors[prev_ch] = neighbors.get(prev_ch, [])
                neighbors[prev_ch].append(cur_ch)
                inward_degree[cur_ch] = inward_degree.get(cur_ch, 0) + 1
                break

    char_set = { ch for word in sorted_words for ch in word }
    queue = [ch for ch in char_set if ch not in inward_degree]
    top_order = []
    while queue:
        cur = queue.pop(0)
        top_order.append(cur)
        for nb in neighbors.get(cur, []):
            inward_degree[nb] -= 1
            if inward_degree[nb] == 0:
                del inward_degree[nb]
                queue.append(nb)
    return top_order if len(top_order) == len(char_set) else None


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
    unittest.main(exit=False, verbosity=2)
```

### Nov 5, 2021 \[Medium\] Tree Serialization
---
> **Question:** You are given the root of a binary tree. You need to implement 2 functions:
>
> 1. `serialize(root)` which serializes the tree into a string representation
> 2. `deserialize(s)` which deserializes the string back to the original tree that it represents
>
> For this problem, often you will be asked to design your own serialization format. However, for simplicity, let's use the pre-order traversal of the tree.

**Example:**
```py
     1
    / \
   3   4
  / \   \
 2   5   7

serialize(tree)
# returns "1 3 2 # # 5 # # 4 # 7 # #"
```

**Solution with Preorder Traversal:** [https://replit.com/@trsong/Serialize-and-Deserialize-the-Binary-Tree-3](https://replit.com/@trsong/Serialize-and-Deserialize-the-Binary-Tree-3)
```py
import unittest

class BinaryTreeSerializer(object):
    @staticmethod
    def serialize(root):
        stack = [root]
        res = []
        while stack:
            cur = stack.pop()
            if cur == None:
                res.append("#")
            else:
                res.append(str(cur.val))
                stack.extend([cur.right, cur.left])
        return ' '.join(res)

    @staticmethod
    def deserialize(s):
        tokens = iter(s.split())
        return BinaryTreeSerializer.deserialize_tokens(tokens)

    @staticmethod
    def deserialize_tokens(tokens):
        raw_val = next(tokens)
        if raw_val == '#':
            return None
        
        left_child = BinaryTreeSerializer.deserialize_tokens(tokens)
        right_child = BinaryTreeSerializer.deserialize_tokens(tokens)
        return TreeNode(int(raw_val), left_child, right_child)


###################
# Testing Utilities
###################
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


class BinaryTreeSerializerSpec(unittest.TestCase):
    def test_example(self):
        """
             1
            / \
           3   4
          / \   \
         2   5   7
        """
        n3 = TreeNode(3, TreeNode(2), TreeNode(5))
        n4 = TreeNode(4, right=TreeNode(7))
        root = TreeNode(1, n3, n4)
        encoded = BinaryTreeSerializer.serialize(root)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(root, decoded)

    def test_serialize_right_heavy_tree(self):
        """
            1
           / \
          2   3
             / \
            4   5
        """
        n3 = TreeNode(3, TreeNode(4), TreeNode(5))
        root = TreeNode(1, TreeNode(2), n3)
        encoded = BinaryTreeSerializer.serialize(root)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(root, decoded)

    def test_balanced_tree(self):
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
        root = TreeNode(5, n4, n7)
        encoded = BinaryTreeSerializer.serialize(root)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(root, decoded)

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
        root = TreeNode(1, TreeNode(2, TreeNode(3)))
        encoded = BinaryTreeSerializer.serialize(root)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(root, decoded)

    def test_serialize_right_heavy_tree2(self):
        """
        1
         \
          2
         /
        3
        """
        root = TreeNode(1, right=TreeNode(2, TreeNode(3)))
        encoded = BinaryTreeSerializer.serialize(root)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(root, decoded)

    def test_zig_zag_tree(self):
        """
            1
           /
          2
         /
        3
         \
          4
           \
            5
           /
          6
        """
        n5 = TreeNode(5, TreeNode(6))
        n3 = TreeNode(3, right=TreeNode(4, right=n5))
        root = TreeNode(1, TreeNode(2, n3))
        encoded = BinaryTreeSerializer.serialize(root)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(root, decoded)

    def test_full_tree(self):
        """
             1
           /   \
          2     3
         / \   / \
        4   5 6   7
        """
        n2 = TreeNode(2, TreeNode(4), TreeNode(5))
        n3 = TreeNode(3, TreeNode(6), TreeNode(7))
        root = TreeNode(1, n2, n3)
        encoded = BinaryTreeSerializer.serialize(root)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(root, decoded)

    def test_all_node_has_value_zero(self):
        """
          0
         / \
        0   0
        """
        root = TreeNode(0, TreeNode(0), TreeNode(0))
        encoded = BinaryTreeSerializer.serialize(root)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(root, decoded)
        

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Nov 4, 2021 LC 448 \[Easy\] Find Missing Numbers in an Array
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

**Solution:** [https://replit.com/@trsong/Find-Missing-Numbers-in-an-Array-of-Range-n](https://replit.com/@trsong/Find-Missing-Numbers-in-an-Array-of-Range-n)
```py
import unittest

def find_missing_numbers(nums):
    n = len(nums)
    for num in nums:
        index = abs(num) - 1
        if 0 <= index < n:
            nums[index] = -abs(nums[index])
    
    res = []
    for i in range(n):
        if nums[i] > 0:
            res.append(i + 1)
        nums[i] = abs(nums[i])
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
    unittest.main(exit=False, verbosity=2)
```

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
