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

### May 11, 2019 \[Easy\] Compare Version Numbers
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


### May 10, 2019 \[Medium\] All Root to Leaf Paths in Binary Tree
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


**Solution with Backtracking:** [https://repl.it/@trsong/Find-All-Root-to-Leaf-Paths-in-Binary-Tree](https://repl.it/@trsong/Find-All-Root-to-Leaf-Paths-in-Binary-Tree)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right= right


def path_to_leaves(tree):
    if not tree:
        return []
    res = []
    backtrack(res, [], tree)
    return res


def backtrack(res, path_to_node, cur_node):
    if cur_node.left is None and cur_node.right is None:
        res.append(path_to_node + [cur_node.val])
    else:
        for child in [cur_node.left, cur_node.right]:
            if child is not None:
                path_to_node.append(cur_node.val)
                backtrack(res, path_to_node, child)
                path_to_node.pop()


class PathToLeavesSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertEqual([], path_to_leaves(None))

    def test_one_level_tree(self):
        self.assertEqual([[1]], path_to_leaves(TreeNode(1)))

    def test_two_level_tree(self):
        tree = TreeNode(1, TreeNode(2), TreeNode(3))
        self.assertEqual([[1, 2], [1, 3]], path_to_leaves(tree))

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
        self.assertEqual([[1, 2], [1, 3, 4], [1, 3, 5]], path_to_leaves(tree))

    def test_complete_tree(self):
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
        expected = [[1, 2, 4], [1, 2, 5], [1, 3, 6], [1, 3, 7]]
        self.assertEqual(expected, path_to_leaves(root))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 9, 2019  \[Easy\] String Compression
---
> **Question:** Given an array of characters with repeats, compress it in place. The length after compression should be less than or equal to the original array.

**Example:**
```py
Input: ['a', 'a', 'b', 'c', 'c', 'c']
Output: ['a', '2', 'b', 'c', '3']
```

**Solution:** [https://repl.it/@trsong/Compress-String](https://repl.it/@trsong/Compress-String)
```py
import unittest

def string_compression(msg):
    if not msg:
        return

    pos = 0
    count = 0
    prev_char = msg[0]
    n = len(msg)

    for i in xrange(n+1):
        char = msg[i] if i < n else None
        if char != prev_char:
            msg[pos] = prev_char
            if count > 1:
                for count_char in str(count):
                    pos += 1
                    msg[pos] = count_char
            pos += 1
            count = 1
            prev_char = char
        else:
            count += 1
    
    for _ in xrange(pos, n):
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

### May 8, 2019  \[Easy\] Find Unique Element among Array of Duplicates
---
> **Question:** Given an array of integers, arr, where all numbers occur twice except one number which occurs once, find the number. Your solution should ideally be `O(n)` time and use constant extra space.

**Example:**
```py
Input: arr = [7, 3, 5, 5, 4, 3, 4, 8, 8]
Output: 7
```

**My thoughts:** XOR has many pretty useful properties:

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

**Solution with XOR:** [https://repl.it/@trsong/Find-Unique-Element-in-Array](https://repl.it/@trsong/Find-Unique-Element-in-Array)
```py
import unittest

def find_unique_element(nums):
    return reduce(lambda num, accu: num ^ accu, nums)

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

### May 7, 2019  \[Easy\] Full Binary Tree
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

**Solution:** [https://repl.it/@trsong/Full-Binary-Tree-Prune](https://repl.it/@trsong/Full-Binary-Tree-Prune)
```py
import unittest
import copy

def full_tree_prune(root):
    if not root:
        return None
    
    left_pruned_tree = full_tree_prune(root.left)
    right_pruned_tree = full_tree_prune(root.right)

    if left_pruned_tree and right_pruned_tree:
        root.left = left_pruned_tree
        root.right = right_pruned_tree
        return root
    elif left_pruned_tree:
        return left_pruned_tree
    elif right_pruned_tree:
        return right_pruned_tree
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
        self.assertEqual(expected_tree, full_tree_prune(original_tree))

    def test_empty_tree(self):        
        self.assertIsNone(None, full_tree_prune(None))

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
        self.assertEqual(expected_tree, full_tree_prune(original_tree))

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
        self.assertEqual(expected_tree, full_tree_prune(original_tree))

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
        self.assertEqual(expected_tree, full_tree_prune(original_tree))

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
        self.assertEqual(expected_tree, full_tree_prune(original_tree))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 6, 2019 LC 301 \[Hard\] Remove Invalid Parentheses
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


**Solution with Backtracking:** [https://repl.it/@trsong/LC301-Remove-Invalid-Parentheses](https://repl.it/@trsong/LC301-Remove-Invalid-Parentheses)
```py
import unittest

def remove_invalid_parenthese(s):
    invalid_open = invalid_close = 0
    for c in s:
        if c == ')' and invalid_open == 0:
            invalid_close += 1
        elif c == '(':
            invalid_open += 1
        elif c == ')':
            invalid_open -= 1
    
    res = []
    backtrack(s, res, 0, invalid_open, invalid_close)
    return filter(is_valid, res)


def backtrack(s, res, next_index, invalid_open, invalid_close):
    if invalid_open == 0 and invalid_close == 0:
        res.append(s)
    else:
        for i in xrange(next_index, len(s)):
            c = s[i]
            if c == '(' and invalid_open > 0 or c == ')' and invalid_close > 0:
                if i > next_index and s[i] == s[i-1]:
                    # skip same character
                    continue
                
                updated_s = s[:i] + s[i+1:]
                if c == '(':
                    backtrack(updated_s, res, i, invalid_open-1, invalid_close)
                else:
                    backtrack(updated_s, res, i, invalid_open, invalid_close-1)


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

### May 5, 2019 \[Hard\] Maximum Spanning Tree
--- 
> **Question:** Recall that the minimum spanning tree is the subset of edges of a tree that connect all its vertices with the smallest possible total edge weight. 
> 
> Given an undirected graph with weighted edges, compute the maximum weight spanning tree.


**My thoughts:** Both Kruskal's and Prim's Algorithm works for this question. The idea is to flip all edge weight into negative and then apply either of previous algorithms until find a spanning tree. 

**Solution with Kruskal's Algorithm:** [https://repl.it/@trsong/Find-Maximum-Spanning-Tree](https://repl.it/@trsong/Find-Maximum-Spanning-Tree)
```py
import unittest
import sys

class DisjointSet(object):
    def __init__(self, size):
        self.parent = [-1] * size

    def find(self, p):
        while self.parent[p] != -1:
            p = self.parent[p]
        return p
    
    def union(self, p1, p2):
        parent1 = self.find(p1)
        parent2 = self.find(p2)
        if parent1 != parent2:
            self.parent[parent1] = parent2

    def is_connected(self, p1, p2):
        return self.find(p1) == self.find(p2)


def max_spanning_tree(vertices, edges):
    uf = DisjointSet(vertices)
    edges.sort(key=lambda uvw: uvw[-1], reverse=True)  # sort based on weight
    res = []

    for u, v, _ in edges:
        if not uf.is_connected(u, v):
            res.append((u, v))
            uf.union(u, v)

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
