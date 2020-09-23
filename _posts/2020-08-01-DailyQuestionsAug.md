---
layout: post
title:  "Daily Coding Problems 2020 Aug to Oct"
date:   2020-08-01 22:22:32 -0700
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


### Sep 23, 2020 LC 236 \[Medium\] Lowest Common Ancestor of a Binary Tree
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


### Sep 22, 2020 \[Medium\] Is Bipartite
---
> **Question:** Given an undirected graph G, check whether it is bipartite. Recall that a graph is bipartite if its vertices can be divided into two independent sets, U and V, such that no edge connects vertices of the same set.

**Example:**
```py
is_bipartite(vertices=3, edges=[(0, 1), (1, 2), (2, 0)])  # returns False 
is_bipartite(vertices=2, edges=[(0, 1), (1, 0)])  # returns True. U = {0}. V = {1}. 
```

**My thoughts:** A graph is a bipartite if we can just use 2 colors to cover entire graph so that every other node have same color. Like this [solution with DFS](https://trsong.github.io/python/java/2020/02/02/DailyQuestionsFeb/#apr-15-2020-medium-is-bipartite). Or the following way with union-find. For each node `u`, just union all of its neihbors. And at the same time, `u` should not be connectted with any of its neihbor.

**Solution with Union-Find (DisjointSet):** [https://repl.it/@trsong/Is-a-Graph-Bipartite](https://repl.it/@trsong/Is-a-Graph-Bipartite)
```py
import unittest

class DisjointSet(object):
    def __init__(self, n):
        self.parent = range(n)

    def find(self, v):
        p = self.parent
        while p[v] != v:
            p[v] = p[p[v]]
            v = p[v]
        return v

    def is_connected(self, u, v):
        return self.find(u) == self.find(v)

    def union(self, u, v):
        p1 = self.find(u)
        p2 = self.find(v)
        if p1 != p2:
            self.parent[p1] = p2


def is_bipartite(vertices, edges):
    neighbors = [None] * vertices
    for u, v in edges:
        neighbors[u] = neighbors[u] or []
        neighbors[v] = neighbors[v] or []
        neighbors[u].append(v)
        neighbors[v].append(u)
    
    uf = DisjointSet(vertices)
    for u in xrange(vertices):
        if neighbors[u] is None:
            continue
        for v in neighbors[u]:
            if uf.is_connected(u, v):
                return False
            uf.union(v, neighbors[u][0])
    return True
            

class IsBipartiteSpec(unittest.TestCase):
    def test_example1(self):
        self.assertFalse(is_bipartite(vertices=3, edges=[(0, 1), (1, 2), (2, 0)]))

    def test_example2(self):
        self.assertTrue(is_bipartite(vertices=2, edges=[(0, 1), (1, 0)]))

    def test_empty_graph(self):
        self.assertTrue(is_bipartite(vertices=0, edges=[]))

    def test_one_node_graph(self):
        self.assertTrue(is_bipartite(vertices=1, edges=[]))
    
    def test_disconnect_graph1(self):
        self.assertTrue(is_bipartite(vertices=10, edges=[(0, 1), (1, 0)]))

    def test_disconnect_graph2(self):
        self.assertTrue(is_bipartite(vertices=10, edges=[(0, 1), (1, 0), (2, 3), (3, 4), (4, 5), (5, 2)]))

    def test_disconnect_graph3(self):
        self.assertFalse(is_bipartite(vertices=10, edges=[(0, 1), (1, 0), (2, 3), (3, 4), (4, 2)])) 

    def test_square(self):
        self.assertTrue(is_bipartite(vertices=4, edges=[(0, 1), (1, 2), (2, 3), (3, 0)]))

    def test_k5(self):
        vertices = 5
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4), 
            (3, 4)
        ]
        self.assertFalse(is_bipartite(vertices, edges))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 21, 2020 \[Easy\] Run-length String Encode and Decode
---
> **Question:** Run-length encoding is a fast and simple method of encoding strings. The basic idea is to represent repeated successive characters as a single count and character. For example, the string `"AAAABBBCCDAA"` would be encoded as `"4A3B2C1D2A"`.
>
> Implement run-length encoding and decoding. You can assume the string to be encoded have no digits and consists solely of alphabetic characters. You can assume the string to be decoded is valid.


**Solution:** [https://repl.it/@trsong/Run-length-String-Encode-and-Decode](https://repl.it/@trsong/Run-length-String-Encode-and-Decode)
```py
import unittest
from collections import Counter

class RunLengthProcessor(object):
    @staticmethod
    def encode(s):
        if not s:
            return ""

        count = 0
        prev = s[0]
        res = []
        for i, ch in enumerate(s):
            if ch == prev:
                count += 1
            else:
                res.extend([str(count), prev])
                prev = ch
                count = 1
        res.extend([str(count), prev])
        return ''.join(res)

    @staticmethod
    def decode(s):
        i = 0
        res = []
        while i < len(s):
            count = 0
            while '0' <= s[i] <= '9':
                count = 10 * count + int(s[i])
                i += 1

            res.append(s[i] * count)
            i += 1
        return ''.join(res)

            
class RunLengthProcessorSpec(unittest.TestCase):
    def assert_encode_decode(self, original, encoded):
        self.assertEqual(encoded, RunLengthProcessor.encode(original))
        self.assertEqual(Counter(original), Counter(RunLengthProcessor.decode(encoded)))
        self.assertEqual(original, RunLengthProcessor.decode(encoded))

    def test_encode_example(self):
        original = "AAAABBBCCDAA"
        encoded = "4A3B2C1D2A"
        self.assert_encode_decode(original, encoded)

    def test_empty_string(self):
        self.assert_encode_decode("", "")

    def test_single_digit_chars(self):
        original = "ABCD"
        encoded = "1A1B1C1D"
        self.assert_encode_decode(original, encoded)

    def test_two_digit_chars(self):
        original = 'a' * 10 + 'b' * 11 + 'c' * 21
        encoded = "10a11b21c"
        self.assert_encode_decode(original, encoded)

    def test_multiple_digit_chars(self):
        original = 'a' + 'b' * 100 + 'c' + 'd' * 2 + 'e' * 32
        encoded = "1a100b1c2d32e"
        self.assert_encode_decode(original, encoded)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 20, 2020 LC 421 \[Medium\] Maximum XOR of Two Numbers in an Array
---
> **Question:** Given an array of integers, find the maximum XOR of any two elements.

**Example:**
```py
Input: nums = [3,10,5,25,2,8]
Output: 28
Explanation: The maximum result is 5 XOR 25 = 28.
```

**My thoughts:** The idea is to efficiently get the other number for each number such that xor is max. We can parellel this process with Trie: insert all number into trie, and for each number greedily choose the largest number that has different digit. 

**Solution with Trie:** [https://repl.it/@trsong/Maximum-XOR-of-Two-Numbers-in-an-Array](https://repl.it/@trsong/Maximum-XOR-of-Two-Numbers-in-an-Array)
```py
import unittest

class Trie(object):
    def __init__(self):
        self.children = [None, None]

    def insert(self, num):
        p = self
        for i in xrange(32, -1, -1):
            bit = 1 & (num >> i)
            if p.children[bit] is None:
                p.children[bit] = Trie()
            p = p.children[bit]

    def find_max_xor_with(self, num):
        p = self
        accu = 0
        for i in xrange(32, -1, -1):
            bit = 1 & (num >> i)
            if p.children[1 - bit]:
                accu ^= 1 << i
                p = p.children[1 - bit]
            else:
                p = p.children[bit]
        return accu


def find_max_xor(nums):
    trie = Trie()
    for num in nums:
        trie.insert(num)

    return max(map(trie.find_max_xor_with, nums))


class FindMaxXORSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(bin(expected), bin(result))

    def test_example(self):
        nums = [  0b11,
                0b1010,
                 0b101,
               0b11001,
                  0b10,
                0b1000]
        expected = 0b101 ^ 0b11001
        self.assert_result(expected, find_max_xor(nums))

    def test_only_one_element(self):
        nums = [0b11]
        expected = 0b11 ^ 0b11
        self.assert_result(expected, find_max_xor(nums))

    def test_two_elements(self):
        nums = [0b10, 0b100]
        expected = 0b10 ^ 0b100
        self.assert_result(expected, find_max_xor(nums))

    def test_example2(self):
        nums = [ 0b1110,
              0b1000110,
               0b110101,
              0b1010011,
               0b110001,
              0b1011011,
               0b100100,
              0b1010000,
              0b1011100,
               0b110011,
              0b1000010,
              0b1000110]
        expected = 0b1011011 ^ 0b100100
        self.assert_result(expected, find_max_xor(nums))

    def test_return_max_number_of_set_bit(self):
        nums = [ 0b111,
                  0b11,
                   0b1,
                     0]
        expected = 0b111 ^ 0
        self.assert_result(expected, find_max_xor(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 19, 2020 LC 653 \[Easy\] Two Sum in BST
---
> **Question:** Given the root of a binary search tree, and a target K, return two nodes in the tree whose sum equals K.

**Example:**
```py
Given the following tree and K of 20:

    10
   /   \
 5      15
       /  \
     11    15

Return the nodes 5 and 15.
```

**Solution with In-order Traversal:** [https://repl.it/@trsong/Two-Sum-in-BST](https://repl.it/@trsong/Two-Sum-in-BST)
```py
import unittest

def solve_two_sum_bst(root, k):
    if not root:
        return None
    
    it1 = find_inorder_traversal(root)
    it2 = find_inorder_traversal(root, reverse=True)
    p1 = next(it1)
    p2 = next(it2)
    while p1 != p2:
        pair_sum = p1.val + p2.val
        if pair_sum < k:
            p1 = next(it1)
        elif pair_sum > k:
            p2 = next(it2)
        else:
            return [p1.val, p2.val]
    return None


def find_inorder_traversal(root, reverse=False):
    p = root
    stack = []
    while p or stack:
        if p:
            stack.append(p)
            p = p.right if reverse else p.left
        else:
            p = stack.pop()
            yield p
            p = p.left if reverse else p.right


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class SolveTwoSumBSTSepc(unittest.TestCase):
    def test_example(self):
        """
             10
            /   \
           5    15
               /  \
             11    15
        """
        root = TreeNode(10, TreeNode(5), TreeNode(15, TreeNode(11), TreeNode(15)))
        k = 20
        expected = [5, 15]
        self.assertItemsEqual(expected, solve_two_sum_bst(root, k))

    def test_empty_tree(self):
        self.assertIsNone(solve_two_sum_bst(None, 0))

    def test_left_heavy_tree(self):
        """
            5
           / \
          3   6
         / \   \
        2   4   7
        """
        left_tree = TreeNode(3, TreeNode(2), TreeNode(4))
        right_tree = TreeNode(6, right=TreeNode(7))
        root = TreeNode(5, left_tree, right_tree)
        k = 9
        expected = [2, 7]
        self.assertItemsEqual(expected, solve_two_sum_bst(root, k))

    def test_no_pair_exists(self):
        """
          5
         / \
        1   8
        """
        root = TreeNode(5, TreeNode(1), TreeNode(8))
        k = 10
        self.assertIsNone(solve_two_sum_bst(root, k))

    def test_balanced_tree(self):
        """
             5
           /   \
          3     11
         / \   /  \
        0   4 10  25
        """
        left_tree = TreeNode(3, TreeNode(0), TreeNode(4))
        right_tree = TreeNode(11, TreeNode(10), TreeNode(25))
        root = TreeNode(5, left_tree, right_tree)
        k = 13
        expected = [3, 10]
        self.assertItemsEqual(expected, solve_two_sum_bst(root, k))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 18, 2020 \[Hard\] Order of Course Prerequisites
---
> **Question:** We're given a hashmap associating each courseId key with a list of courseIds values, which represents that the prerequisites of courseId are courseIds. Return a sorted ordering of courses such that we can finish all courses.
>
> Return null if there is no such ordering.
>
> For example, given `{'CSC300': ['CSC100', 'CSC200'], 'CSC200': ['CSC100'], 'CSC100': []}`, should return `['CSC100', 'CSC200', 'CSC300']`.


**My thoughts:** there are two ways to produce toplogical sort order: DFS or count inward edge as below. For DFS technique see this [post](http://trsong.github.io/python/java/2019/05/01/DailyQuestions/#jul-5-2019-hard-order-of-course-prerequisites). For below method, we count number of inward edges for each node and recursively remove edges of each node that has no inward egdes. The node removal order is what we call topological order.

**Solution with Topological Sort:** [https://repl.it/@trsong/Find-Order-of-Course-Prerequisites](https://repl.it/@trsong/Find-Order-of-Course-Prerequisites)
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
    unittest.main(exit=False)
```


### Sep 17, 2020 \[Medium\] Construct BST from Post-order Traversal
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

**Solution:** [https://repl.it/@trsong/Construct-Binary-Search-Tree-from-Post-order-Traversal](https://repl.it/@trsong/Construct-Binary-Search-Tree-from-Post-order-Traversal)
```py
import unittest

def construct_bst(post_order_traversal):
    return construct_bst_recur(post_order_traversal, float('-inf'), float('inf'))


def construct_bst_recur(stack, lo, hi):
    if not (stack and lo <= stack[-1] <= hi):
        return None
    val = stack.pop()
    right_child = construct_bst_recur(stack, val, hi)
    left_child = construct_bst_recur(stack, lo, val)
    return Node(val, left_child, right_child)


class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __eq__(self, other):
        return other and other.val == self.val and other.left == self.left and other.right == self.right

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

### Sep 16, 2020 LC 329 \[Hard\] Longest Increasing Path in a Matrix
---
> **Question:** Given an integer matrix, find the length of the longest increasing path.
>
> From each cell, you can either move to four directions: left, right, up or down. You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).

**Example 1:**
```py
Input: nums = 
[
  [9,9,4],
  [6,6,8],
  [2,1,1]
] 
Output: 4 
Explanation: The longest increasing path is [1, 2, 6, 9].
```

**Example 2:**
```py
Input: nums = 
[
  [3,4,5],
  [3,2,6],
  [2,2,1]
] 
Output: 4 
Explanation: The longest increasing path is [3, 4, 5, 6]. Moving diagonally is not allowed.
```

**My thoughts:** From any node, a valid path can never decrease and won't form a cycle; that create a DAG. The longest path in DAG can be calculated based on topological order within linear time.

**Solution 1 with Topological Sort (Remove Edge):** [https://repl.it/@trsong/Longest-Increasing-Path-in-a-Matrix-Remove-Edge-Top-Sort](https://repl.it/@trsong/Longest-Increasing-Path-in-a-Matrix-Remove-Edge-Top-Sort)
```py
def longest_path_length(grid):
    if not grid or not grid[0]:
        return 0
    
    DIRECTIONS = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    n, m = len(grid), len(grid[0])
    inward_edges = [[0 for _ in xrange(m)] for _ in xrange(n)]
    queue = []
    
    for r in xrange(n):
        for c in xrange(m):
            cur_val = grid[r][c]
            for dr, dc in DIRECTIONS:
                from_r, from_c = r + dr, c + dc
                if 0 <= from_r < n and 0 <=  from_c < m and grid[from_r][from_c] < cur_val:
                    inward_edges[r][c] += 1
            if inward_edges[r][c] == 0:
                queue.append((r, c))
                
    max_distance = 0
    distance_grid = [[1 for _ in xrange(m)] for _ in xrange(n)]
    while queue:
        r, c = queue.pop(0)
        max_distance = max(max_distance, distance_grid[r][c])
        cur_val = grid[r][c]
        for dr, dc in DIRECTIONS:
            to_r, to_c = r + dr, c + dc
            if 0 <= to_r < n and 0 <= to_c < m and cur_val < grid[to_r][to_c]:
                distance_grid[to_r][to_c] = max(distance_grid[to_r][to_c], 1 + distance_grid[r][c])
                inward_edges[to_r][to_c] -= 1
                if inward_edges[to_r][to_c] == 0:
                    queue.append((to_r, to_c))
                    
    return max_distance
```

**Solution 2 with Topological Sort (DFS):** [https://repl.it/@trsong/Longest-Increasing-Path-in-a-Matrix](https://repl.it/@trsong/Longest-Increasing-Path-in-a-Matrix)
```py
import unittest

DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]


def longest_path_length(grid):
    if not grid or not grid[0]:
        return 0

    top_order = find_top_order(grid)
    return get_longest_distance(grid, top_order)


def find_top_order(grid):
    class NodeState:
        UNVISITED = 0
        VISITING = 1
        VISITED = 2

    n, m = len(grid), len(grid[0])
    node_states = [[NodeState.UNVISITED for _ in xrange(m)] for _ in xrange(n)]
    reverse_top_order = []

    for c in xrange(m):
        for r in xrange(n):
            if node_states[r][c] is not NodeState.UNVISITED:
                continue

            stack = [(r, c)]
            while stack:
                cur_r, cur_c = stack[-1]
                cur_val = grid[cur_r][cur_c]
                if node_states[cur_r][cur_c] is NodeState.VISITED:
                    stack.pop()
                elif node_states[cur_r][cur_c] is NodeState.VISITING:
                    reverse_top_order.append((cur_r, cur_c))
                    node_states[cur_r][cur_c] = NodeState.VISITED
                else:
                    node_states[cur_r][cur_c] = NodeState.VISITING
                    for dr, dc in DIRECTIONS:
                        new_r, new_c = cur_r + dr, cur_c + dc
                        if (0 <= new_r < n and 0 <= new_c < m
                                and node_states[new_r][new_c] is
                                NodeState.UNVISITED and grid[new_r][new_c] > cur_val):
                            stack.append((new_r, new_c))
    return reverse_top_order[::-1]


def get_longest_distance(grid, top_order):
    n, m = len(grid), len(grid[0])
    distances = [[1 for _ in xrange(m)] for _ in xrange(n)]
    max_distance = 1
    for r, c in top_order:
        cur_val = grid[r][c]
        for dr, dc in DIRECTIONS:
            from_r, from_c = r + dr, c + dc
            if 0 <= from_r < n and 0 <= from_c < m and grid[from_r][from_c] < cur_val:
                distances[r][c] = max(distances[r][c], 1 + distances[from_r][from_c])
                max_distance = max(max_distance, distances[r][c])
    return max_distance


class LongestPathLengthSpec(unittest.TestCase):
    def test_example(self):
        grid = [
            [9, 9, 4],
            [6, 6, 8],
            [2, 1, 1]
        ]
        expected = 4  # 1, 2, 6, 9
        self.assertEqual(expected, longest_path_length(grid))

    def test_example2(self):
        grid = [
            [3, 4, 5],
            [3, 2, 6],
            [2, 2, 1]
        ]
        expected = 4  # 3, 4, 5, 6
        self.assertEqual(expected, longest_path_length(grid))

    def test_empty_grid(self):
        self.assertEqual(0, longest_path_length([[]]))

    def test_sping_around_entire_grid(self):
        grid = [
            [5, 6, 7],
            [4, 1, 8],
            [3, 2, 9]
        ]
        expected = 9  # 1, 2, 3, 4, 5, 6, 7, 8, 9
        self.assertEqual(expected, longest_path_length(grid))

    def test_no_path(self):
        grid = [
            [1, 1],
            [1, 1],
            [1, 1]
        ]
        expected = 1 
        self.assertEqual(expected, longest_path_length(grid))

    def test_two_paths(self):
        grid = [
            [4, 3, 2],
            [5, 0, 1],
            [3, 2, 3]
        ]
        expected = 6  # 0, 1, 2, 3, 4, 5 
        self.assertEqual(expected, longest_path_length(grid))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 15, 2020 LC 986 \[Medium\] Interval List Intersections
---
> **Question:** Given two lists of closed intervals, each list of intervals is pairwise disjoint and in sorted order.
>
> Return the intersection of these two interval lists.

**Example:**
```py
Input: A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
```

**Solution:** [https://repl.it/@trsong/Interval-List-Intersections](https://repl.it/@trsong/Interval-List-Intersections)
```py
import unittest

def interval_intersection(seq1, seq2):
    stream1, stream2 = iter(seq1), iter(seq2)
    res = []
    interval1 = next(stream1, None)
    interval2 = next(stream2, None)
    while interval1 and interval2:
        start1, end1 = interval1
        start2, end2 = interval2
        if end1 < start2:
            interval1 = next(stream1, None)
        elif end2 < start1:
            interval2 = next(stream2, None)
        else:
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)  
            res.append([overlap_start, overlap_end])
            # cut off overlap and reuse
            interval1 = [overlap_end + 1, end1]  
            interval2 = [overlap_end + 1, end2]
    return res


class IntervalIntersectionSpec(unittest.TestCase):
    def test_example(self):
        seq1 = [[0, 2], [5, 10], [13, 23], [24, 25]]
        seq2 = [[1, 5], [8, 12], [15, 24], [25, 26]]
        expected = [[1, 2], [5, 5], [8, 10], [15, 23], [24, 24], [25, 25]]
        self.assertEqual(expected, interval_intersection(seq1, seq2))

    def test_empty_interval(self):
        seq1 = []
        seq2 = [[1, 2]]
        expected = []
        self.assertEqual(expected, interval_intersection(seq1, seq2))

    def test_overlap_entire_interval(self):
        seq1 = [[1, 1], [2, 2], [3, 3], [8, 8]]
        seq2 = [[0, 5]]
        expected = [[1, 1], [2, 2], [3, 3]]
        self.assertEqual(expected, interval_intersection(seq1, seq2))

    def test_overlap_entire_interval2(self):
        seq1 = [[0, 5]]
        seq2 = [[1, 2], [4, 7]]
        expected = [[1, 2], [4, 5]]
        self.assertEqual(expected, interval_intersection(seq1, seq2))

    def test_overlap_one_interval(self):
        seq1 = [[0, 2], [5, 7]]
        seq2 = [[1, 3], [4, 6]]
        expected = [[1, 2], [5, 6]]
        self.assertEqual(expected, interval_intersection(seq1, seq2))

    def test_same_start_time(self):
        seq1 = [[0, 2], [3, 7]]
        seq2 = [[0, 5]]
        expected = [[0, 2], [3, 5]]
        self.assertEqual(expected, interval_intersection(seq1, seq2))

    def test_same_start_time2(self):
        seq1 = [[0, 2], [5, 7]]
        seq2 = [[5, 7], [8, 9]]
        expected = [[5, 7]]
        self.assertEqual(expected, interval_intersection(seq1, seq2))
    
    def test_no_overlapping(self):
        seq1 = [[1, 2], [5, 7]]
        seq2 = [[0, 0], [3, 4], [8, 9]]
        expected = []
        self.assertEqual(expected, interval_intersection(seq1, seq2))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 14, 2020 LC 57 \[Hard\] Insert Interval
---
> **Question:** Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).
>
> You may assume that the intervals were initially sorted according to their start times.

**Example 1:**
```py
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
```

**Example 2:**
```py
Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
```

**Solution:** [https://repl.it/@trsong/Insert-Interval](https://repl.it/@trsong/Insert-Interval)
```py
import unittest

def insert_interval(intervals, newInterval):
    res = []
    cur_start, cur_end = newInterval
    for start, end in intervals:
        if end < cur_start:
            res.append([start, end])
        elif cur_end < start:
            res.append([cur_start, cur_end])
            cur_start, cur_end = start, end
        else:
            cur_start = min(cur_start, start)
            cur_end = max(cur_end, end)
    res.append([cur_start, cur_end])
    return res


class InsertIntervalSpec(unittest.TestCase):
    def test_example(self):
        intervals = [[1, 3], [6, 9]]
        newInterval = [2, 5]
        expected = [[1, 5], [6, 9]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_example2(self):
        intervals = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
        newInterval = [4, 8]
        expected = [[1, 2], [3, 10], [12, 16]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_insert_into_empty_intervals(self):
        intervals = []
        newInterval = [4, 8]
        expected = [[4, 8]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_new_interval_overlaps_all(self):
        intervals = [[1, 2], [3, 4], [5, 6]]
        newInterval = [2, 10]
        expected = [[1, 10]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_new_interval_does_not_expand_original(self):
        intervals = [[1, 7], [9, 12]]
        newInterval = [10, 11]
        expected = [[1, 7], [9, 12]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_new_interval_before_all(self):
        intervals = [[1, 2], [3, 4]]
        newInterval = [0, 1]
        expected = [[0, 2], [3, 4]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_new_interval_before_all2(self):
        intervals = [[1, 2], [3, 4]]
        newInterval = [0, 0]
        expected = [[0, 0], [1, 2], [3, 4]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_new_interval_after_all(self):
        intervals = [[1, 2], [3, 4]]
        newInterval = [5, 6]
        expected = [[1, 2], [3, 4], [5, 6]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_new_interval_after_all2(self):
        intervals = [[1, 2], [3, 4]]
        newInterval = [3, 8]
        expected = [[1, 2], [3, 8]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 13, 2020 LC 37 \[Hard\] Sudoku Solver
---
> **Question:** Write a program to solve a Sudoku puzzle by filling the empty cells.
>
> A sudoku solution must satisfy all of the following rules:
> - Each of the digits `1-9` must occur exactly once in each row.
> - Each of the digits `1-9` must occur exactly once in each column.
> - Each of the the digits `1-9` must occur exactly once in each of the 9 `3x3` sub-boxes of the grid.
> - Empty cells are indicated `0`.

**Example:**
```py
Input: [
    [3, 0, 6, 5, 0, 8, 4, 0, 0], 
    [5, 2, 0, 0, 0, 0, 0, 0, 0], 
    [0, 8, 7, 0, 0, 0, 0, 3, 1], 
    [0, 0, 3, 0, 1, 0, 0, 8, 0], 
    [9, 0, 0, 8, 6, 3, 0, 0, 5], 
    [0, 5, 0, 0, 9, 0, 6, 0, 0], 
    [1, 3, 0, 0, 0, 0, 2, 5, 0], 
    [0, 0, 0, 0, 0, 0, 0, 7, 4], 
    [0, 0, 5, 2, 0, 6, 3, 0, 0]
]

Possible output:
[
    [3, 1, 6, 5, 7, 8, 4, 9, 2],
    [5, 2, 9, 1, 3, 4, 7, 6, 8],
    [4, 8, 7, 6, 2, 9, 5, 3, 1],
    [2, 6, 3, 4, 1, 5, 9, 8, 7],
    [9, 7, 4, 8, 6, 3, 1, 2, 5],
    [8, 5, 1, 7, 9, 2, 6, 4, 3],
    [1, 3, 8, 9, 4, 7, 2, 5, 6],
    [6, 9, 2, 3, 5, 1, 8, 7, 4],
    [7, 4, 5, 2, 8, 6, 3, 1, 9]
]
```

**Solution with Backtracking:** [https://repl.it/@trsong/Sudoku-Solver](https://repl.it/@trsong/Sudoku-Solver)
```py
import unittest
from functools import reduce
from copy import deepcopy

class SampleSudoku(object):
    UNFINISHED1 = [
        [3, 0, 6, 5, 0, 8, 4, 0, 0],
        [5, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 7, 0, 0, 0, 0, 3, 1], 
        [0, 0, 3, 0, 1, 0, 0, 8, 0],
        [9, 0, 0, 8, 6, 3, 0, 0, 5], 
        [0, 5, 0, 0, 9, 0, 6, 0, 0],
        [1, 3, 0, 0, 0, 0, 2, 5, 0],
        [0, 0, 0, 0, 0, 0, 0, 7, 4],
        [0, 0, 5, 2, 0, 6, 3, 0, 0]
    ]
    
    FINISHED1 = [
        [3, 1, 6, 5, 7, 8, 4, 9, 2],
        [5, 2, 9, 1, 3, 4, 7, 6, 8],
        [4, 8, 7, 6, 2, 9, 5, 3, 1],
        [2, 6, 3, 4, 1, 5, 9, 8, 7],
        [9, 7, 4, 8, 6, 3, 1, 2, 5],
        [8, 5, 1, 7, 9, 2, 6, 4, 3],
        [1, 3, 8, 9, 4, 7, 2, 5, 6],
        [6, 9, 2, 3, 5, 1, 8, 7, 4],
        [7, 4, 5, 2, 8, 6, 3, 1, 9]
    ]

    UNFINISHED2 = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    FINISHED2 = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2], 
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1], 
        [7, 1, 3, 9, 2, 4, 8, 5, 6], 
        [9, 6, 1, 5, 3, 7, 2, 8, 4], 
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ]



class SudokuSolver(object):
    def __init__(self, grid):
        self.grid = grid
        self.row_occupied = [[False for _ in xrange(9)] for _ in xrange(9)]
        self.col_occupied = [[False for _ in xrange(9)] for _ in xrange(9)]
        self.reigon_occupied = [[[False for _ in xrange(9)] for _ in xrange(3)] for _ in xrange(3)]

    def mark_position(self, r, c, num):
        self.grid[r][c] = num
        num_index = num - 1
        self.row_occupied[r][num_index] = True
        self.col_occupied[c][num_index] = True
        self.reigon_occupied[r//3][c//3][num_index] = True
    
    def unmark_position(self, r, c):
        num = self.grid[r][c]
        self.grid[r][c] = 0
        num_index = num - 1
        self.row_occupied[r][num_index] = False
        self.col_occupied[c][num_index] = False
        self.reigon_occupied[r//3][c//3][num_index] = False

    def is_valid_number(self, r, c, num):
        num_index = num - 1
        return not(self.row_occupied[r][num_index] 
                or self.col_occupied[c][num_index]
                or self.reigon_occupied[r//3][c//3][num_index])

    def backtrack(self, index, blank_positions):
        if index >= len(blank_positions):
            return True
            
        r, c = blank_positions[index]
        valid_numbers = filter(lambda num: self.is_valid_number(r, c, num), xrange(1, 10))
        for num in valid_numbers:
            self.mark_position(r, c, num)
            if self.backtrack(index + 1, blank_positions):
               return True
            self.unmark_position(r, c)
        return False

    def solve(self):
        blank_positions = []
        for r in xrange(9):
            for c in xrange(9):
                num = self.grid[r][c]
                if num == 0:
                    blank_positions.append((r, c))
                else:
                    self.mark_position(r, c, num)
        
        self.backtrack(0, blank_positions)
        return self.grid

        
class SudokuSolverSpec(unittest.TestCase):
    def assert_result(self, input_grid, expected_grid):
        grid = deepcopy(input_grid)
        solver = SudokuSolver(grid)
        finished_grid = solver.solve()
        validation_res = SudokuValidator.validate(finished_grid)
        msg = '\n' + '\n'.join(map(str, finished_grid)) + '\nPossible Solution: \n' + '\n'.join(map(str, expected_grid))
        self.assertTrue(validation_res, msg)

    def test_example(self):
        self.assert_result(SampleSudoku.UNFINISHED1, SampleSudoku.FINISHED1)

    def test_example2(self):
        self.assert_result(SampleSudoku.UNFINISHED2, SampleSudoku.FINISHED2)


##################
# Testing Utility
##################
class SudokuValidator(object):
    @staticmethod
    def validate(grid):
        return (SudokuValidator.validate_row(grid)
                and SudokuValidator.validate_col(grid)
                and SudokuValidator.validate_region(grid))

    @staticmethod
    def validate_row(grid):
        rows = [[(r, c) for c in xrange(9)] for r in xrange(9)]
        return all(map(lambda row: SudokuValidator.validate_uniq(grid, row), rows))

    @staticmethod
    def validate_col(grid):
        cols = [[(r, c) for r in xrange(9)] for c in xrange(9)]
        return all(map(lambda col: SudokuValidator.validate_uniq(grid, col), cols))

    @staticmethod
    def validate_region(grid):
        indices = [0, 3, 6]
        width = 3
        reigons = [[(r + dr, c + dc) for dr in xrange(width)
                    for dc in xrange(width)] for r in indices for c in indices]
        return all(map(lambda reigon: SudokuValidator.validate_uniq(grid, reigon), reigons))

    @staticmethod
    def validate_uniq(grid, postions):
        values = map(lambda pos: grid[pos[0]][pos[1]], postions)
        count = sum(1 for _ in values)
        bits = reduce(lambda accu, v: accu | (1 << v), values, 0)
        return count == 9 and bits == 0b1111111110


class SudokuValidatorSpec(unittest.TestCase):
    def test_valid_grid(self):
        self.assertTrue(SudokuValidator.validate(SampleSudoku.FINISHED1))

    def test_invalid_grid(self):
        self.assertFalse(SudokuValidator.validate(SampleSudoku.UNFINISHED1))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 12, 2020 \[Medium\] Count Attacking Bishop Pairs
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
Thus, we can store the number of bishop on the same diagonal and use the formula to calculate n-choose-2: `n(n-1)/2`


**Solution:** [https://repl.it/@trsong/Count-Number-of-Attacking-Bishop-Pairs](https://repl.it/@trsong/Count-Number-of-Attacking-Bishop-Pairs)
```py
import unittest

def count_attacking_pairs(bishop_positions):
    diagonal_key1 = lambda pos: pos[0] - pos[1]
    diagonal_key2 = lambda pos: pos[0] + pos[1]
    return count_pairs_by(bishop_positions, diagonal_key1) + count_pairs_by(bishop_positions, diagonal_key2)


def count_pairs_by(positions, key):
    histogram = {}
    for pos in positions:
        pos_key = key(pos)
        histogram[pos_key] = histogram.get(pos_key, 0) + 1
    # n choos 2 = n * (n-1) / 2
    return sum(map(lambda count: count * (count - 1) // 2, histogram.values()))


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
    unittest.main(exit=False)
```

### Sep 11, 2020 LC 358 \[Hard\] Rearrange String K Distance Apart
---
> **Question:** Given a non-empty string str and an integer k, rearrange the string such that the same characters are at least distance k from each other.
>
> All input strings are given in lowercase letters. If it is not possible to rearrange the string, return an empty string "".

**Example 1:**
```py
str = "aabbcc", k = 3
Result: "abcabc"
The same letters are at least distance 3 from each other.
```

**Example 2:**
```py
str = "aaabc", k = 3 
Answer: ""
It is not possible to rearrange the string.
```

**Example 3:**
```py
str = "aaadbbcc", k = 2
Answer: "abacabcd"
Another possible answer is: "abcabcda"
The same letters are at least distance 2 from each other.
```

**My thoughts:** Greedily choose the character with max remaining number for each window k. If no such character satisfy return empty string directly. 

**Solution with Greedy Approach:** [https://repl.it/@trsong/Rearrange-Strings-K-Distance-Apart](https://repl.it/@trsong/Rearrange-Strings-K-Distance-Apart)
```py
import unittest
from Queue import PriorityQueue

def rearrange_string(input_string, k):
    if k <= 0:
        return ""

    histogram = {}
    for ch in input_string:
        histogram[ch] = histogram.get(ch, 0) + 1

    pq = PriorityQueue()
    for ch, count in histogram.items():
        pq.put((-count, ch))

    res = []
    while not pq.empty():
        remaining_chars = []
        for _ in xrange(k):
            if pq.empty() and not remaining_chars:
                break
            elif pq.empty():
                return ""
            
            neg_count, ch = pq.get()
            res.append(ch)
            if abs(neg_count) > 1:
                remaining_chars.append((abs(neg_count)-1, ch))
        
        for count, ch in remaining_chars:
            pq.put((-count, ch))

    return "".join(res)
                

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

### Sep 10, 2020 \[Easy\] Remove Duplicates From Linked List
---
> **Question:** Given a linked list, remove all duplicate values from the linked list.
>
> For instance, given `1 -> 2 -> 3 -> 3 -> 4`, then we wish to return the linked list `1 -> 2 -> 4`.

**Solution:** [https://repl.it/@trsong/Remove-Duplicates-From-Linked-List](https://repl.it/@trsong/Remove-Duplicates-From-Linked-List)
```py
import unittest

def remove_duplicates(lst):
    prev = dummy = ListNode(-1, lst)
    p = lst
    visited = set()
    while p:
        if p.val in visited:
            prev.next = p.next
        else:
            visited.add(p.val)
            prev = p
        p = p.next
    return dummy.next


###################
# Testing Utilities
###################
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    def __eq__(self, other):
        return other and self.val == other.val and self.next == other.next

    def __repr__(self):
        return "%d -> %s" % (self.val, self.next)

    @staticmethod  
    def List(*vals):
        dummy = ListNode(-1)
        p = dummy
        for elem in vals:
            p.next = ListNode(elem)
            p = p.next
        return dummy.next 


class RemoveDuplicateSpec(unittest.TestCase):
    def test_example(self):
        source_list = ListNode.List(1, 2, 3, 3, 4)
        expected = ListNode.List(1, 2, 3, 4)
        self.assertEqual(expected, remove_duplicates(source_list))

    def test_empty_list(self):
        self.assertIsNone(remove_duplicates(None))
    
    def test_one_element_list(self):
        source_list = ListNode.List(-1)
        expected = ListNode.List(-1)
        self.assertEqual(expected, remove_duplicates(source_list))

    def test_list_with_unique_value(self):
        source_list = ListNode.List(1, 1, 1, 1)
        expected = ListNode.List(1)
        self.assertEqual(expected, remove_duplicates(source_list))
    
    def test_list_with_duplicate_elements(self):
        source_list = ListNode.List(11, 11, 11, 21, 43, 43, 60)
        expected = ListNode.List(11, 21, 43, 60)
        self.assertEqual(expected, remove_duplicates(source_list))
    
    def test_list_with_duplicate_elements2(self):
        source_list = ListNode.List(1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5)
        expected = ListNode.List(1, 2, 3, 4, 5)
        self.assertEqual(expected, remove_duplicates(source_list))

    def test_list_without_duplicate_elements(self):
        source_list = ListNode.List(1, 2)
        expected = ListNode.List(1, 2)
        self.assertEqual(expected, remove_duplicates(source_list))

    def test_unsorted_list(self):
        source_list = ListNode.List(2, 1, 1, 2, 1)
        expected = ListNode.List(2, 1)
        self.assertEqual(expected, remove_duplicates(source_list))

    def test_unsorted_list2(self):
        source_list = ListNode.List(3, 2, 1, 3, 2, 1)
        expected = ListNode.List(3, 2, 1)
        self.assertEqual(expected, remove_duplicates(source_list))

    def test_unsorted_list3(self):
        source_list = ListNode.List(1, 1, 1)
        expected = ListNode.List(1)
        self.assertEqual(expected, remove_duplicates(source_list))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 9, 2020 LC 139 \[Medium\] Word Break
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

**Solution with DP:** [https://repl.it/@trsong/Word-Break-Problem](https://repl.it/@trsong/Word-Break-Problem)
```py
import unittest

def word_break(s, word_dict):
    n = len(s)
    # Let dp[i] indicates whether s[:i] is breakable
    # dp[i] = dp[i-k] for s[i-k:i] in dictinary with length k
    dp = [False] * (n + 1)
    dp[0] = True
    for i in xrange(1, n + 1):
        for word in word_dict:
            if dp[i]:
                break
            k = len(word)
            dp[i] = i >= k and dp[i - k] and s[i - k:i] == word
    return dp[n]


class WordBreakSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(word_break("Pseudocode", ["Pseudo", "code"]))

    def test_example2(self):
        self.assertTrue(word_break("applepenapple", ["apple", "pen"]))

    def test_example3(self):
        self.assertFalse(
            word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]))

    def test_word_in_dict_is_a_prefix(self):
        self.assertTrue(word_break("123456", ["12", "123", "456"]))

    def test_word_in_dict_is_a_prefix2(self):
        self.assertTrue(word_break("123456", ["12", "123", "456"]))

    def test_word_in_dict_is_a_prefix3(self):
        self.assertFalse(word_break("123456", ["12", "12345", "456"]))

    def test_empty_word(self):
        self.assertTrue(word_break("", ['a']))

    def test_empty_word2(self):
        self.assertTrue(word_break("", []))

    def test_empty_word3(self):
        self.assertFalse(word_break("a", []))

    def test_use_same_word_twice(self):
        self.assertTrue(word_break("aaabaaa", ["aaa", "b"]))

    def test_impossible_word_combination(self):
        self.assertFalse(word_break("aaaaa", ["aaa", "aaaa"]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 8, 2020 LC 13 \[Easy\] Convert Roman Numerals to Decimal
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

**Solution:** [https://repl.it/@trsong/Convert-Roman-Format-Number](https://repl.it/@trsong/Convert-Roman-Format-Number)
```py
import unittest

roman_unit = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

def roman_to_decimal(roman_str):
    prev = 0
    res = 0
    for ch in roman_str:
        cur = roman_unit[ch]
        if prev < cur:
            res -= 2 * prev
        res += cur
        prev = cur
    return res 


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

### Sep 7, 2020 LC 763 \[Medium\] Partition Labels
---
> **Question:** A string S of lowercase English letters is given. We want to partition this string into as many parts as possible so that each letter appears in at most one part, and return a list of integers representing the size of these parts.

**Example:**
```py
Input: S = "ababcbacadefegdehijhklij"
Output: [9, 7, 8]
Explanation: The partition is "ababcbaca", "defegde", "hijhklij". 
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits S into less parts.
```

**Solution:** [https://repl.it/@trsong/Partition-Labels](https://repl.it/@trsong/Partition-Labels)
```py
import unittest

def partion_labels(s):
    last_occur = {ch: pos for pos, ch in enumerate(s)}
    res = []
    cut_pos = -1
    last_cut_pos = -1
    for i, ch in enumerate(s):
        cut_pos = max(cut_pos, last_occur[ch])
        if i == cut_pos:
            window_size = cut_pos - last_cut_pos
            res.append(window_size)
            last_cut_pos = cut_pos
    return res


class PartionLabelSpec(unittest.TestCase):
    def test_example(self):
        s = "ababcbacadefegdehijhklij"
        expected = [9, 7, 8]  # "ababcbaca", "defegde", "hijhklij"
        self.assertEqual(expected, partion_labels(s))

    def test_empty_string(self):
        self.assertEqual([], partion_labels(""))

    def test_one_whole_string(self):
        s = "aaaaa"
        expected = [5]
        self.assertEqual(expected, partion_labels(s))

    def test_string_with_distinct_characters(self):
        s = "abcde"
        expected = [1, 1, 1, 1, 1]
        self.assertEqual(expected, partion_labels(s))

    def test_string_that_cannot_break(self):
        s = "abca"
        expected = [4]
        self.assertEqual(expected, partion_labels(s))

    def test_string_that_cannot_break2(self):
        s = "abaca"
        expected = [5]
        self.assertEqual(expected, partion_labels(s))

    def test_break_as_many_parts(self):
        s = "abacddc"
        expected = [3, 4]
        self.assertEqual(expected, partion_labels(s))

    def test_break_as_many_parts2(self):
        s = "abacdde"
        expected = [3, 1, 2, 1]
        self.assertEqual(expected, partion_labels(s))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 6, 2020 LC 8 \[Medium\] String to Integer (atoi)
---
> **Question:** Given a string, convert it to an integer without using the builtin str function. You are allowed to use ord to convert a character to ASCII code.
>
> Consider all possible cases of an integer. In the case where the string is not a valid integer, return `0`.

**Example:**
```py
atoi('-105')  # -105
```

**Solution:** [https://repl.it/@trsong/String-to-Integer-atoi](https://repl.it/@trsong/String-to-Integer-atoi)
```py
import unittest

INT_MIN = -2 ** 31
INT_MAX = 2 ** 31 - 1

def atoi(s):
    n = len(s)
    pos = 0
    while pos < n and s[pos] == ' ':
        pos += 1
    
    sign = 1
    if pos < n and (s[pos] == '+' or s[pos] == '-'):
        sign = -1 if s[pos] == '-' else 1
        pos += 1

    res = 0
    while pos < n and '0' <= s[pos] <= '9':
        digit = ord(s[pos]) - ord('0')
        if sign > 0 and res > (INT_MAX - digit) // 10:
            return INT_MAX
        if sign < 0 and res <= (INT_MIN + digit) // 10:
            return INT_MIN
        res = 10 * res + sign * digit
        pos += 1

    return res


class AtoiSpec(unittest.TestCase):
    def test_example(self):
        s = '-105'
        expected = -105
        self.assertEqual(expected, atoi(s))

    def test_empty_string(self):
        s = ''
        expected = 0
        self.assertEqual(expected, atoi(s))

    def test_positive_number(self):
        s = '42'
        expected = 42
        self.assertEqual(expected, atoi(s))

    def test_negative_number_with_space(self):
        s = '    -42'
        expected = -42
        self.assertEqual(expected, atoi(s))

    def test_invalid_string_as_suffix(self):
        s = '4123 some words 2'
        expected = 4123
        self.assertEqual(expected, atoi(s))

    def test_invalid_string_as_prefix(self):
        s = 'some prefix and 1'
        expected = 0
        self.assertEqual(expected, atoi(s))

    def test_number_out_of_range(self):
        s = '-91283472332'
        expected = INT_MIN
        self.assertEqual(expected, atoi(s))

    def test_number_out_of_range2(self):
        s = '-2147483649'
        expected = INT_MIN
        self.assertEqual(expected, atoi(s))

    def test_number_out_of_range3(self):
        s = '-2147483647'
        expected = -2147483647
        self.assertEqual(expected, atoi(s))

    def test_number_out_of_range4(self):
        s = '2147483647'
        expected = INT_MAX
        self.assertEqual(expected, atoi(s))

    def test_number_out_of_range5(self):
        s = '2147483648'
        expected = INT_MAX
        self.assertEqual(expected, atoi(s))

    def test_number_out_of_range6(self):
        s = '2147483646'
        expected = 2147483646
        self.assertEqual(expected, atoi(s))

    def test_prefix_with_zero(self):
        s = '   00012345'
        expected = 12345
        self.assertEqual(expected, atoi(s))

    def test_invalid_prefix(self):
        s = '+-2'
        expected = 0
        self.assertEqual(expected, atoi(s))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 5, 2020 LC 54 \[Medium\] Spiral Matrix 
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

**Solution:** [https://repl.it/@trsong/Spiral-Matrix-Traversal](https://repl.it/@trsong/Spiral-Matrix-Traversal)
```py
import unittest

def spiral_order(matrix):
    if not matrix or not matrix[0]:
        return []
    r_min, r_max = 0, len(matrix)-1
    c_min, c_max = 0, len(matrix[0])-1
    res = []

    while r_min <= r_max and c_min <= c_max:
        r, c = r_min, c_min

        # Move right
        while c < c_max:
            res.append(matrix[r][c])
            c += 1
        c_max -= 1

        # Move down
        while r < r_max:
            res.append(matrix[r][c])
            r += 1
        r_max -= 1

        # Is stuck and can neither move left nor up
        if c_min > c_max or r_min > r_max:
            res.append(matrix[r][c])
            break

        # Move left
        while c > c_min:
            res.append(matrix[r][c])
            c -= 1
        c_min += 1

        # Move up
        while r > r_min:
            res.append(matrix[r][c])
            r -= 1
        r_min += 1

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
        self.assertEqual(spiral_order([]), [])
        self.assertEqual(spiral_order([[]]), [])

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
    unittest.main(exit=False)
```

### Sep 4, 2020 \[Medium\] Zig-Zag String
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

**Solution:** [https://repl.it/@trsong/Print-Zig-Zag-String](https://repl.it/@trsong/Print-Zig-Zag-String)
```py
import unittest

def zig_zag_format(sentence, k):
    if not sentence:
        return []
    elif k == 1:
        return [sentence]

    res = [[] for _ in xrange(min(k, len(sentence)))]
    direction = -1
    row = 0

    for i, ch in enumerate(sentence):
        if i < k:
            space = i
        elif direction > 0:
            space = 2 * row - 1
        else:
            space = 2 * (k - row - 1) - 1
        res[row].append(space * " " + ch)

        if row == 0 or row == k-1:
            direction *= -1
        row += direction

    return map(lambda line: "".join(line), res)


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


### Sep 3, 2020 \[Easy\] Level of tree with Maximum Sum
---
> **Question:** Given a binary tree, find the level in the tree where the sum of all nodes on that level is the greatest.

**Example:**
```py
The following tree should return level 1:
    1          Level 0 - Sum: 1
   / \
  4   5        Level 1 - Sum: 9 
 / \ / \
3  2 4 -1      Level 2 - Sum: 8
```

**Solution with DFS:** [https://repl.it/@trsong/Find-Level-of-tree-with-Maximum-Sum](https://repl.it/@trsong/Find-Level-of-tree-with-Maximum-Sum)
```py
import unittest

def max_sum_tree_level(tree):
    if not tree:
        return -1
        
    sum_by_row = []
    stack = [(tree, 0)]

    while stack:
        cur, depth = stack.pop()
        if not cur:
            continue
        if depth >= len(sum_by_row):
            sum_by_row.append(0)
        sum_by_row[depth] += cur.val
        stack.extend([(cur.right, depth+1), (cur.left, depth+1)])

    max_index, _ = max(enumerate(sum_by_row), key=lambda kv: kv[1])
    return max_index


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class MaxSumTreeLevelSpec(unittest.TestCase):
    def test_example(self):
        """
            1          Level 0 - Sum: 1
           / \
          4   5        Level 1 - Sum: 9 
         / \ / \
        3  2 4 -1      Level 2 - Sum: 8
        """
        n4 = TreeNode(4, TreeNode(3), TreeNode(2))
        n5 = TreeNode(5, TreeNode(4), TreeNode(-1))
        root = TreeNode(1, n4, n5)
        self.assertEqual(1, max_sum_tree_level(root))

    def test_empty_tree(self):
        self.assertEqual(-1, max_sum_tree_level(None))

    def test_tree_with_one_node(self):
        root = TreeNode(42)
        self.assertEqual(0, max_sum_tree_level(root))
    
    def test_unbalanced_tree(self):
        """
            20
           / \
          8   22
         / \
        4  12
           / \
         10  14
        """
        n4 = TreeNode(4)
        n10 = TreeNode(10)
        n14 = TreeNode(14)
        n22 = TreeNode(22)
        n12 = TreeNode(12, n10, n14)
        n8 = TreeNode(8, n4, n12)
        n20 = TreeNode(20, n8, n22)
        self.assertEqual(1, max_sum_tree_level(n20))
    
    def test_zigzag_tree(self):
        """
        1
         \
          5
         /
        2 
         \
          3
        """
        n3 = TreeNode(3)
        n2 = TreeNode(2, right=n3)
        n5 = TreeNode(5, n2)
        n1 = TreeNode(1, right=n5)
        self.assertEqual(1, max_sum_tree_level(n1))

    def test_tree_with_negative_values(self):
        """
             -1
            /  \
          -2   -3
          /    /
        -1   -6
        """
        left_tree = TreeNode(-2, TreeNode(-1))
        right_tree = TreeNode(-3, TreeNode(-6))
        root = TreeNode(-1, left_tree, right_tree)
        self.assertEqual(0, max_sum_tree_level(root))

    def test_tree_with_negative_values_and_zeros(self):
        """
        -1
          \
           0
            \
            -2
        """
        tree = TreeNode(-1, right=TreeNode(0, right=TreeNode(-2)))
        self.assertEqual(1, max_sum_tree_level(tree))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 2, 2020 \[Easy\] Deepest Node in a Binary Tree
---
> **Question:** You are given the root of a binary tree. Return the deepest node (the furthest node from the root).

**Example:**
```py
    a
   / \
  b   c
 /
d
The deepest node in this tree is d at depth 3.
```

**Solution with BFS:** [https://repl.it/@trsong/Find-Deepest-Node-in-a-Binary-Tree](https://repl.it/@trsong/Find-Deepest-Node-in-a-Binary-Tree)
```py
import unittest
from functools import reduce

def find_deepest_node(root):
    it = bfs_traversal(root)
    return reduce(lambda _, x: x, it, None)


def bfs_traversal(root):
    queue = [root]
    while queue:
        cur = queue.pop(0)
        if not cur:
            continue
        yield cur
        queue.extend([cur.left, cur.right])
        

class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return "Node(%d)" % self.val


class FindDeepestNodeSpec(unittest.TestCase):
    def test_example(self):
        """
            1
           / \
          2   3
         /
        4
        """
        deepest_node = Node(4)
        root = Node(1, Node(2, deepest_node), Node(3))
        self.assertEqual(deepest_node, find_deepest_node(root))

    def test_empty_tree(self):
        self.assertIsNone(find_deepest_node(None))

    def test_root_only(self):
        root = Node(1)
        self.assertEqual(root, find_deepest_node(root))

    def test_complete_tree(self):
        """
               1
             /   \
            2     3
           / \   / \
          4   5 6   7
         /
        8
        """
        deepest_node = Node(8)
        left_tree = Node(2, Node(4, deepest_node), Node(5))
        right_tree = Node(3, Node(6), Node(7))
        root = Node(1, left_tree, right_tree)
        self.assertEqual(deepest_node, find_deepest_node(root))

    def test_has_more_than_one_answer(self):
        """
           1
          / \
         2   3
        / \   \
       4   5   6
           /    \
          7      8 
        """
        deepest_node1 = Node(7)
        deepest_node2 = Node(8)
        left_tree = Node(2, Node(4), Node(5, deepest_node1))
        right_tree = Node(3, right=Node(6, right=deepest_node2))
        root = Node(1, left_tree, right_tree)
        self.assertIn(find_deepest_node(root), [deepest_node1, deepest_node2])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 1, 2020 LC 230 \[Medium\] Kth Smallest Element in a BST
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

**Solution with In-order Traversal:** [https://repl.it/@trsong/Find-Kth-Smallest-Element-in-a-BST](https://repl.it/@trsong/Find-Kth-Smallest-Element-in-a-BST)
```py
import unittest

def kth_smallest(root, k):
    it = in_order_traversal(root)
    for _ in xrange(k):
        v = next(it)
    return v


def in_order_traversal(root):
    p = root
    stack = []
    while p or stack:
        if p:
            stack.append(p)
            p = p.left
        else:
            p = stack.pop()
            yield p.val
            p = p.right
    

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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
            self.assertEqual(e, kth_smallest(tree, e))

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
            self.assertEqual(e, kth_smallest(tree, e))

    def test_full_BST(self):
        """
             4
           /   \
          2     6
         / \   / \
        1   3 5   7
        """
        n2 = TreeNode(2, TreeNode(1), TreeNode(3))
        n6 = TreeNode(6, TreeNode(5), TreeNode(7))
        tree = TreeNode(4, n2, n6)
        check_expected = [1, 2, 3, 4, 5, 6, 7]
        for e in check_expected:
            self.assertEqual(e, kth_smallest(tree, e))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 31, 2020 \[Easy\] Depth of Binary Tree in Peculiar String Representation
---
> **Question:** You are given a binary tree in a peculiar string representation. Each node is written in the form `(lr)`, where `l` corresponds to the left child and `r` corresponds to the right child.
>
> If either `l` or `r` is null, it will be represented as a zero. Otherwise, it will be represented by a new `(lr)` pair.
> 
> Given this representation, determine the depth of the tree.

**Here are a few examples:**
```py
A root node with no children: (00)
A root node with two children: ((00)(00))
An unbalanced tree with three consecutive left children: ((((00)0)0)0)
```


**Solution:** [https://repl.it/@trsong/Depth-of-Binary-Tree-in-Peculiar-String-Representation](https://repl.it/@trsong/Depth-of-Binary-Tree-in-Peculiar-String-Representation)
```py
import unittest

def depth(tree_str):
    depth_delta = {
        '(': 1,
        ')': -1,
        '0': 0
    }
    max_accu = 0
    accu = 0
    for c in tree_str:
        accu += depth_delta[c]
        max_accu = max(max_accu, accu)
    return max_accu


class DepthSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertEqual(0, depth("0"))
    
    def test_one_node_without_child(self):
        self.assertEqual(1, depth("(00)"))
    
    def test_one_node_with_two_children(self):
        """
          .
         / \
        .   .
        """
        self.assertEqual(2, depth("((00)(00))"))

    def test_left_heavy_tree(self):
        """
              .
             / \
            .   .
           / \
          .   .
         /
        . 
        """
        node = "(00)"
        ll = "(%s0)" % node
        l = "(%s%s)" % (ll, node)
        root = "(%s0)" % l
        self.assertEqual(4, depth(root))

    def test_right_heavy_tree(self):
        """
            .
           / \
          .   .
         / \   \
        .   .   .
               / \
              .   .
        """ 
        node2 = "((00)(00))"
        r = "(0%s)" % node2
        root = "(%s%s)" % (node2, r)
        self.assertEqual(4, depth(root))

    def test_zig_zag_tree(self):
        """
          .
         /
        .
         \
          .
         /
        .
         \
          .
        """
        node = "(00)"
        make_left = lambda n: "(%s0)" % n
        make_right = lambda n: "(0%s)" % n
        lrl = make_right(node)
        lr = make_left(lrl)
        l = make_right(lr)
        root = make_left(l)
        self.assertEqual(5, depth(root))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 30, 2020 LC 166 \[Medium\] Fraction to Recurring Decimal
---
> **Question:** Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.
>
> If the fractional part is repeating, enclose the repeating part in parentheses.

**Example 1:**
```py
Input: numerator = 1, denominator = 2
Output: "0.5"
```

**Example 2:**
```py
Input: numerator = 2, denominator = 1
Output: "2"
```

**Example 3:**
```py
Input: numerator = 2, denominator = 3
Output: "0.(6)"
```

**Solution:** [https://repl.it/@trsong/Convert-Fraction-to-Recurring-Decimal](https://repl.it/@trsong/Convert-Fraction-to-Recurring-Decimal)
```py
import unittest

def fraction_to_decimal(numerator, denominator):
    if numerator == 0:
        return "0"

    sign = "-" if (numerator > 0) ^ (denominator > 0) else ""
    numerator, denominator = abs(numerator), abs(denominator)
    quotient, remainder = numerator // denominator, numerator % denominator
    if remainder == 0:
        return "%s%d" % (sign, quotient)
    
    decimals = []
    remainder_history = {}
    i = 0

    while remainder != 0 and remainder not in remainder_history:
        remainder_history[remainder] = i
        i += 1

        remainder *= 10
        decimals.append(remainder // denominator)
        remainder %= denominator


    if remainder == 0:
        return "%s%d.%s" % (sign, quotient, "".join(map(str, decimals)))
    else:
        repeated_start = remainder_history[remainder]
        non_repeat_decimals = "".join(map(str, decimals[:repeated_start]))
        repeated_decimals = "".join(map(str, decimals[repeated_start:]))
        return "%s%d.%s(%s)" % (sign, quotient, non_repeat_decimals, repeated_decimals)


class FractionToDecimalSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual("0.5", fraction_to_decimal(1, 2))

    def test_example2(self):
        self.assertEqual("2", fraction_to_decimal(2, 1))

    def test_example3(self):
        self.assertEqual("0.(6)", fraction_to_decimal(2, 3))
    
    def test_decimal_has_duplicate_digits(self):
        self.assertEqual("1011.(1011)", fraction_to_decimal(3370000, 3333))

    def test_result_is_zero(self):
        self.assertEqual("0", fraction_to_decimal(0, -42))

    def test_negative_numerator_and_denominator(self):
        self.assertEqual("1.75", fraction_to_decimal(-7, -4))

    def test_negative_numerator(self):
        self.assertEqual("-1.7(5)", fraction_to_decimal(-79, 45))

    def test_negative_denominator(self):
        self.assertEqual("-3", fraction_to_decimal(3, -1))

    def test_non_recurring_decimal(self):
        self.assertEqual("0.1234123", fraction_to_decimal(1234123, 10000000))

    def test_recurring_decimal(self):
        self.assertEqual("-0.03(571428)", fraction_to_decimal(-1, 28))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 29, 2020 \[Easy\] Tree Isomorphism Problem
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

**Solution:** [https://repl.it/@trsong/Is-Binary-Tree-Isomorphic](https://repl.it/@trsong/Is-Binary-Tree-Isomorphic)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_isomorphic(t1, t2):
    if not t1 and not t2:
        return True
    
    if not t1 or not t2:
        return False
    
    if t1.val != t2.val:
        return False

    return (is_isomorphic(t1.left, t2.left)
            and is_isomorphic(t1.right, t2.right)
            or is_isomorphic(t1.right, t2.left)
            and is_isomorphic(t1.left, t2.right))


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
        t2 = TreeNode(1, TreeNode(1, TreeNode(1)),
                      TreeNode(1, right=TreeNode(1)))
        self.assertFalse(is_isomorphic(t1, t2))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 28, 2020  LC 86 \[Medium\] Partitioning Linked List
---
> **Question:** Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
>
> You should preserve the original relative order of the nodes in each of the two partitions.

**Example:**
```py
Input: head = 1->4->3->2->5->2, x = 3
Output: 1->2->2->4->3->5
```

**Solution:** [https://repl.it/@trsong/Partitioning-Singly-Linked-List](https://repl.it/@trsong/Partitioning-Singly-Linked-List)
```py
import unittest

def partition(lst, target):
    p1 = dummy1 = Node(-1)
    p2 = dummy2 = Node(-1)

    while lst:
        if lst.val < target:
            p1.next = lst
            p1 = p1.next
        else:
            p2.next = lst
            p2 = p2.next
        lst = lst.next
    
    p2.next = None
    p1.next = dummy2.next
    return dummy1.next
            

##############################
# Below are testing utilities
##############################
class Node(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    @staticmethod
    def flatten(lst):
        res = []
        while lst:
            res.append(lst.val)
            lst = lst.next
        return res

    @staticmethod
    def create(*vals):
        t = dummy = Node(-1)
        for v in vals:
            t.next = Node(v)
            t = t.next
        return dummy.next

class PartitionSpec(unittest.TestCase):
    def assert_result(self, expected_list, result_list, target):
        expected_arr = Node.flatten(expected_list)
        result_arr = Node.flatten(result_list)
        
        self.assertEqual(len(expected_arr), len(result_arr))
        split_index = 0
        for i in xrange(len(expected_arr)):
            if expected_arr[i] >= target:
                split_index = i
                break
        e1, e2 = set(expected_arr[:split_index]), set(expected_arr[split_index:]) 
        r1, r2 = set(result_arr[:split_index]), set(result_arr[split_index:]) 
    
        self.assertEqual(e1, r1)
        self.assertEqual(e2, r2)

    def test_example(self):
        original  = Node.create(1, 4, 3, 2, 5, 2)
        expected = Node.create(1, 2, 2, 4, 3, 5)
        target = 3
        self.assert_result(expected, partition(original, target), target)

    def test_empty_list(self):
        self.assertIsNone(partition(None, 42))

    def test_one_element_list(self):
        original  = Node.create(1)
        expected = Node.create(1)
        target = 0
        self.assert_result(expected, partition(original, target), target)
        target = 1
        self.assert_result(expected, partition(original, target), target)

    def test_list_with_duplicated_elements(self):
        original  = Node.create(1, 1, 0, 1, 1)
        expected = Node.create(0, 1, 1, 1, 1)
        target = 1
        self.assert_result(expected, partition(original, target), target)

    def test_list_with_duplicated_elements2(self):
        original  = Node.create(0, 2, 0, 2, 0)
        expected = Node.create(0, 0, 0, 2, 2)
        target = 1
        self.assert_result(expected, partition(original, target), target)

    def test_list_with_duplicated_elements3(self):
        original  = Node.create(1, 1, 1, 1)
        expected = Node.create(1, 1, 1, 1)
        target = 2
        self.assert_result(expected, partition(original, target), target)

    def test_unsorted_array(self):
        original  = Node.create(10, 4, 20, 10, 3)
        expected = Node.create(3, 10, 4, 20, 10)
        target = 3
        self.assert_result(expected, partition(original, target), target)

    def test_unsorted_array2(self):
        original  = Node.create(1, 4, 3, 2, 5, 2, 3)
        expected = Node.create(1, 2, 2, 3, 3, 4, 5)
        target = 3
        self.assert_result(expected, partition(original, target), target)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 27, 2020 LC 253 \[Easy\] Minimum Lecture Rooms
---
> **Questions:** Given an array of time intervals `(start, end)` for classroom lectures (possibly overlapping), find the minimum number of rooms required.
>
> For example, given `[(30, 75), (0, 50), (60, 150)]`, you should return `2`.

**My thoughts:** whenever we enter an interval at time t, the total number of room at time t increment by 1 and whenever we leave an interval the total number of required room decrement by 1. For `(1, 10)`, `(5, 15)` and `(6, 15)`, `+1` at `t = 1, 5, 6` and `-1` at `t=10, 15, 15`. And at peak hour, the total room equals 3. 

**Solution:** [https://repl.it/@trsong/Minimum-Required-Lecture-Rooms](https://repl.it/@trsong/Minimum-Required-Lecture-Rooms)
```py
import unittest

def min_lecture_rooms(intervals):
    start_times = map(lambda t: (t[0], 1), intervals)
    end_times = map(lambda t: (t[1], -1), intervals)
    times = sorted(start_times + end_times)

    accu_rooms = 0
    max_rooms = 0
    for _, room_diff in times:
        accu_rooms += room_diff
        max_rooms = max(max_rooms, accu_rooms)
    return max_rooms


class MinLectureRoomSpec(unittest.TestCase):
    def setUp(self):
        self.t1 = (-10, 0)
        self.t2 = (-5, 5)
        self.t3 = (0, 10)
        self.t4 = (5, 15)
    
    def test_overlapping_end_points(self):
        intervals = [self.t1] * 3
        expected = 3
        self.assertEqual(expected, min_lecture_rooms(intervals))
    
    def test_overlapping_end_points2(self):
        intervals = [self.t3, self.t1]
        expected = 1
        self.assertEqual(expected, min_lecture_rooms(intervals))

    def test_not_all_overlapping_intervals(self):
        intervals = [(30, 75), (0, 50), (60, 150)]
        expected = 2
        self.assertEqual(expected, min_lecture_rooms(intervals))

    def test_not_all_overlapping_intervals2(self):
        intervals = [self.t1, self.t3, self.t2, self.t4]
        expected = 2
        self.assertEqual(expected, min_lecture_rooms(intervals))

    def test_not_all_overlapping_intervals3(self):
        intervals = [self.t1, self.t3, self.t2, self.t4] * 2
        expected = 4
        self.assertEqual(expected, min_lecture_rooms(intervals))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 26, 2020 LC 1171 \[Medium\] Remove Consecutive Nodes that Sum to 0
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


**Solution:** [https://repl.it/@trsong/Remove-Consecutive-Nodes-that-Sum-to-Zero](https://repl.it/@trsong/Remove-Consecutive-Nodes-that-Sum-to-Zero)
```py
import unittest

def remove_zero_sum_sublists(head):
    dummy = ListNode(-1, head)
    prefix_sum_lookup = {0: dummy}
    p = dummy.next
    accu_sum = 0
    while p:
        accu_sum += p.val
        if accu_sum not in prefix_sum_lookup:
            prefix_sum_lookup[accu_sum] = p
        else:
            loop_start = prefix_sum_lookup[accu_sum]
            q = loop_start.next
            accu_sum_to_remove = accu_sum
            while q != p:
                accu_sum_to_remove += q.val
                del prefix_sum_lookup[accu_sum_to_remove]
                q = q.next
            loop_start.next = p.next
        p = p.next
    return dummy.next


###################
# Testing Utility
###################
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    def __eq__(self, other):
        return other and self.val == other.val and self.next == other.next

    def __repr__(self):
        return "%s -> %s" % (str(self.val), str(self.next))

    @staticmethod  
    def List(*vals):
        p = dummy = ListNode(-1)
        for elem in vals:
            p.next = ListNode(elem)
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


### Aug 25, 2020 LC 796 \[Easy\] Shift-Equivalent Strings
---
> **Question:** Given two strings A and B, return whether or not A can be shifted some number of times to get B.
>
> For example, if A is `'abcde'` and B is `'cdeab'`, return `True`. If A is `'abc'` and B is `'acb'`, return `False`.

**My thoughts:** It should be pretty easy to come up with non-linear time complexity solution. But for linear, I can only come up w/ rolling hash solution. The idea is to treat each digit as a number. For example, `"1234"` is really `1234`, each time we move the most significant bit to right by `(1234 - 1 * 10^3) * 10 + 1 = 2341`. In general, we can treat `'abc'` as numeric value of `abc` base `p0` ie. `a * p0^2 + b * p0^1 + c * p0^0` and in order to prevent overflow, we use a larger prime number which I personally prefer 666667 (easy to remember), `'abc' =>  (a * p0^2 + b * p0^1 + c * p0^0) % p1 where p0 and p1 are both prime and p0 is much smaller than p1`.

**Solution with Roling Hash:** [https://repl.it/@trsong/Check-if-Strings-are-Shift-Equivalent](https://repl.it/@trsong/Check-if-Strings-are-Shift-Equivalent)
```py
import unittest
from functools import reduce

P0 = 23  # small prime number
P1 = 666667 # larger prime number

def hash(s):
    rolling_hash = lambda accu, ch: (accu * P0 % P1 + ord(ch)) % P1
    return reduce(rolling_hash, s, 0)


def is_shift_eq(source, target):
    if len(source) != len(target):
        return False
    elif source == target:
        return True
    
    source_hash = hash(source)
    target_hash = hash(target)
    n = len(source)
    leftmost_digit_base = reduce(lambda accu, _: (accu * P0) % P1, xrange(n-1), 1)

    for ch in source:
        ord_ch = ord(ch)
        leftmost_digit = (ord_ch * leftmost_digit_base) % P1
        source_hash = ((source_hash - leftmost_digit) * P0 + ord_ch) % P1
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

### Aug 24, 2020 LC 560 \[Medium\] Subarray Sum Equals K
---
> **Question:** Given a list of integers and a number `K`, return which contiguous elements of the list sum to `K`.
>
> For example, if the list is `[1, 2, 3, 4, 5]` and `K` is `9`, then it should return `[2, 3, 4]`, since `2 + 3 + 4 = 9`.


**My thoughts:** Store prefix sum along the way to find how many index i exists such that `prefix[j] - prefix[i] = k`. As `j > i`, when we reach j, we pass i already, so we can store `prefix[i]` in a map and put value as occurance of `prefix[i]`, that is why this question feels similar to Two Sum question.

**Solution:** [https://repl.it/@trsong/Find-Number-of-Sub-array-Sum-Equals-K](https://repl.it/@trsong/Find-Number-of-Sub-array-Sum-Equals-K)
```py
import unittest

def subarray_sum(nums, k):
    prefix_sum = 0
    prefix_count = {0: 1}
    res = 0
    for num in nums:
        prefix_sum += num
        target = prefix_sum - k
        res += prefix_count.get(target, 0)
        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1
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
    
    def test_array_with_one_elem2(self):
        self.assertEqual(subarray_sum([1], 1), 1) # [1]

    def test_array_with_unique_target_prefix(self):
        # suppose the prefix_sum = [1, 2, 3, 3, 2, 1]
        self.assertEqual(subarray_sum([1, 1, 1, 0, -1, -1], 2), 4)  # [1, 1], [1, ,1], [1, 1, 0], [1, 1, 1, 0, -1]


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 23, 2020 LC 29 \[Medium\] Divide Two Integers
---
> **Question:** Implement integer division without using the division operator. Your function should return a tuple of `(dividend, remainder)` and it should take two numbers, the product and divisor.
>
> For example, calling `divide(10, 3)` should return `(3, 1)` since the divisor is `3` and the remainder is `1`.

**My thoughts:** Left shift `x << i` is just `x * (2 ** i)`, we can take advantage of that to figure out each digit of the quotient. And find pattern from following to handle special case when operands are negative.
```py
divide(1, 3) => (0, 1)    # 0*3+1 = 1
divide(-1, 3) => (-1, 2)  # -1*3+2 = -1
divide(1, -3) => (-1, -2) # -1*-3-2= 1
divide(-1, -3) => (0, -1) # 0*-3-1= -1
```

**Solution:** [https://repl.it/@trsong/Divide-Two-Integers](https://repl.it/@trsong/Divide-Two-Integers)
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
    unittest.main(exit=False)
```


### Aug 22, 2020 LC 451 \[Easy\] Sort Characters By Frequency
---
> **Question:** Given a string, sort it in decreasing order based on the frequency of characters. If there are multiple possible solutions, return any of them.
>
> For example, given the string `tweet`, return `tteew`. `eettw` would also be acceptable.


**Solution with Counting Sort:** [https://repl.it/@trsong/Sort-Characters-By-Frequency](https://repl.it/@trsong/Sort-Characters-By-Frequency)
```py
import unittest

def string_ordered_by_frequency(s):
    char_freq = {}
    for c in s:
        char_freq[c] = char_freq.get(c, 0) + 1

    n = len(s)
    counting = [None] * (n + 1)
    for c, freq in char_freq.items():
        if counting[freq] is None:
            counting[freq] = []
        counting[freq].append(c)
    
    res = []
    for freq in xrange(n, 0, -1):
        if counting[freq] is None:
            continue
        
        for c in counting[freq]:
            res.append(c * freq)

    return ''.join(res)
           

class StringOrderedByFrequencySpec(unittest.TestCase):
    def test_example(self):
        s = 'tweet'
        expected = ['tteew', 'eettw']
        self.assertIn(string_ordered_by_frequency(s), expected)

    def test_empty_string(self):
        self.assertEqual('', string_ordered_by_frequency(''))

    def test_contains_upper_and_lower_letters(self):
        s = 'aAbb'
        expected = ['bbaA', 'bbAa']
        self.assertIn(string_ordered_by_frequency(s), expected)

    def test_letter_with_differnt_frequency(self):
        s = '241452345534535'
        expected = '555554444333221'
        self.assertEqual(expected, string_ordered_by_frequency(s))

    def test_string_with_unique_letter(self):
        s = 'aaaaaaa'
        self.assertEqual(s, string_ordered_by_frequency(s))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 21, 2020 \[Easy\] Make the Largest Number
---
> **Question:** Given a number of integers, combine them so it would create the largest number.

**Example:**
```py
Input: [17, 7, 2, 45, 72]
Output: 77245217
```

**Solution with Custom Sort:** [https://repl.it/@trsong/Construct-Largest-Number](https://repl.it/@trsong/Construct-Largest-Number)
```py
import unittest

def construct_largest_number(nums):
    if not nums:
        return 0

    negative_nums = filter(lambda x: x < 0, nums)
    filtered_nums = filter(lambda x: x >= 0, nums)
    str_nums = map(str, filtered_nums)
    combined_num_cmp = lambda s1, s2: cmp(s1 + s2, s2 + s1)

    if negative_nums:
        str_nums.sort(combined_num_cmp)
        return int(str(negative_nums[0]) + "".join(str_nums))
    else:
        str_nums.sort(combined_num_cmp, reverse=True)
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

### Aug 20, 2020 \[Medium\] Tree Serialization
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

**Solution with Pre-order Traversal**: [https://repl.it/@trsong/Serialize-and-Deserialize-the-Binary-Tree](https://repl.it/@trsong/Serialize-and-Deserialize-the-Binary-Tree)
```py
import unittest

class BinaryTreeSerializer(object):
    @staticmethod
    def serialize(root):
        res = []
        stack = [root]
        while stack:
            cur = stack.pop()
            if cur is None:
                res.append('#')
            else:
                res.append(str(cur.val))
                stack.extend([cur.right, cur.left])
        return ' '.join(res)

    @staticmethod
    def deserialize(s):
        tokens = iter(s.split())
        return BinaryTreeSerializer.build_tree_recur(tokens)

    @staticmethod
    def build_tree_recur(stream):
        raw_value = next(stream)
        if raw_value == '#':
            return None
        
        left_child = BinaryTreeSerializer.build_tree_recur(stream)
        right_child = BinaryTreeSerializer.build_tree_recur(stream)
        return TreeNode(int(raw_value), left_child, right_child)


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
    unittest.main(exit=False)
```

### Aug 19, 2020 LC 91 \[Medium\] Decode Ways
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

**My thoughts:** This question can be solved w/ DP. Similar to the climb stair problem, `DP[n] = DP[n-1] + DP[n-2]` under certain conditions. If last digits can form a number, i.e. `1 ~ 9` then `DP[n] = DP[n-1]`. And if last two digits can form a number, i.e. `10 ~ 26`, then `DP[n] = DP[n-2]`. If we consider both digits, we will have 

```py
dp[k] = dp[k-1] if str[k-1] can form a number. i.e not zero, 1-9
       + dp[k-2] if str[k-2] and str[k-1] can form a number. 10-26
```

**Solution with DP:** [https://repl.it/@trsong/Number-of-Decode-Ways](https://repl.it/@trsong/Number-of-Decode-Ways)
```py
import unittest

def decode_ways(encoded_string):
    if not encoded_string or encoded_string[0] == '0':
        return 0

    n = len(encoded_string)
    # Let dp[i] represents # of decode ways for encoded_string[:i]
    # dp[i] = dp[i-1]               if digit at i-1 is not zero
    # or    = dp[i-1] + dp[i-2]     if digits at i-1,i-2 is valid 26 char
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    ord_zero = ord('0')
    for i in xrange(2, n + 1):
        first_digit = ord(encoded_string[i - 2]) - ord_zero
        second_digit = ord(encoded_string[i - 1]) - ord_zero
        if second_digit > 0:
            dp[i] += dp[i - 1]
        if 10 <= 10 * first_digit + second_digit <= 26:
            dp[i] += dp[i-2]
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
    unittest.main(exit=False)
```

### Aug 18, 2020 \[Easy\] Merge Overlapping Intervals
---
> **Question:** Given a list of possibly overlapping intervals, return a new list of intervals where all overlapping intervals have been merged.
>
> The input list is not necessarily ordered in any way.
>
> For example, given `[(1, 3), (5, 8), (4, 10), (20, 25)]`, you should return `[(1, 3), (4, 10), (20, 25)]`.

**My thoughts:** Sort intervals based on start time, then for each consecutive intervals s1, s2 the following could occur:
- `s1.end < s2.start`, we append s2 to result
- `s2.start <= s1.end < s2.end`, we merge s1 and s2
- `s2.end <= s1.end`, s1 overlaps all of s2, we do nothing

Note: as all intervals are sorted based on start time, `s1.start <= s2.start`

**Solution with Stack:** [https://repl.it/@trsong/Merge-All-Overlapping-Intervals](https://repl.it/@trsong/Merge-All-Overlapping-Intervals)
```py
import unittest

def merge_intervals(interval_seq):
    interval_seq.sort(key=lambda interval: interval[0])
    stack = []
    for start, end in interval_seq:
        if not stack or stack[-1][1] < start:
            stack.append((start, end))
        elif stack[-1][1] < end:
            prev_start, _ = stack.pop()
            stack.append((prev_start, end))
    return stack


class MergeIntervalSpec(unittest.TestCase):
    def test_interval_with_zero_mergings(self):
        self.assertItemsEqual(merge_intervals([]), [])

    def test_interval_with_zero_mergings2(self):
        interval_seq = [(1, 2), (3, 4), (5, 6)]
        expected = [(1, 2), (3, 4), (5, 6)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))

    def test_interval_with_zero_mergings3(self):
        interval_seq = [(-3, -2), (5, 6), (1, 4)]
        expected = [(-3, -2), (1, 4), (5, 6)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))

    def test_interval_with_one_merging(self):
        interval_seq = [(1, 3), (5, 7), (7, 11), (2, 4)]
        expected = [(1, 4), (5, 11)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))

    def test_interval_with_one_merging2(self):
        interval_seq = [(1, 4), (0, 8)]
        expected = [(0, 8)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))

    def test_interval_with_two_mergings(self):
        interval_seq = [(1, 3), (3, 5), (5, 8)]
        expected = [(1, 8)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))

    def test_interval_with_two_mergings2(self):
        interval_seq = [(5, 8), (1, 6), (0, 2)]
        expected = [(0, 8)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))

    def test_interval_with_multiple_mergings(self):
        interval_seq = [(-5, 0), (1, 4), (1, 4), (1, 4), (5, 7), (6, 10), (0, 1)]
        expected = [(-5, 4), (5, 10)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))

    def test_interval_with_multiple_mergings2(self):
        interval_seq = [(1, 3), (5, 8), (4, 10), (20, 25)]
        expected = [(1, 3), (4, 10), (20, 25)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 17, 2020 \[Medium\] Overlapping Rectangles
--- 
> **Question:** You’re given 2 over-lapping rectangles on a plane. For each rectangle, you’re given its bottom-left and top-right points. How would you find the area of their overlap?


**Solution:** [https://repl.it/@trsong/Overlapping-Rectangle-Areas](https://repl.it/@trsong/Overlapping-Rectangle-Areas)
```py
import unittest

class Rectangle(object):
    def __init__(self, x1, y1, x2, y2):
        self.x_min = min(x1, x2)
        self.x_max = max(x1, x2)
        self.y_min = min(y1, y2)
        self.y_max = max(y1, y2)


def overlapping_areas(rect1, rect2):
    has_no_x_overlap = rect1.x_max < rect2.x_min or rect2.x_max < rect1.x_min
    has_no_y_overlap = rect1.y_max < rect2.y_min or rect2.y_max < rect1.y_min
    if has_no_x_overlap or has_no_y_overlap:
        return 0
    x_proj = min(rect1.x_max, rect2.x_max) - max(rect1.x_min, rect2.x_min)
    y_proj = min(rect1.y_max, rect2.y_max) - max(rect1.y_min, rect2.y_min)
    return x_proj * y_proj


class OverlappingAreaSpec(unittest.TestCase):
    def assert_result(self, res, rect1, rect2):
        self.assertEqual(res, overlapping_areas(rect1, rect2))
        self.assertEqual(res, overlapping_areas(rect2, rect1))

    def test_no_overlapping(self):
        rect1 = Rectangle(-1, 1, 0, 0)
        rect2 = Rectangle(0, 0, 1, -1)
        self.assert_result(0, rect1, rect2)
        
    def test_no_overlapping2(self):
        rect1 = Rectangle(-1, -1, 0, 2)
        rect2 = Rectangle(0, 2, 1, -1)
        self.assert_result(0, rect1, rect2)

    def test_no_overlapping3(self):
        rect1 = Rectangle(0, 0, 1, 1)
        rect2 = Rectangle(0, -1, 1, -2)
        self.assert_result(0, rect1, rect2)

    def test_rectanlge_contains_the_other(self):
        rect1 = Rectangle(-2, 2, 2, -2)
        rect2 = Rectangle(-1, -1, 1, 1)
        self.assert_result(4, rect1, rect2)

    def test_rectanlge_contains_the_other2(self):
        rect1 = Rectangle(0, 0, 2, 2)
        rect2 = Rectangle(0, 0, 2, 1)
        self.assert_result(2, rect1, rect2)

    def test_overlapping_top_bottom(self):
        rect1 = Rectangle(-2, 0, 2, -2)
        rect2 = Rectangle(-1, 1, 1, -1)
        self.assert_result(2, rect1, rect2)
    
    def test_overlapping_left_right(self):
        rect1 = Rectangle(-1, -1, 1, 1)
        rect2 = Rectangle(0, -2, 2, 2)
        self.assert_result(2, rect1, rect2)
    
    def test_overlapping_top_left_bottom_right(self):
        rect1 = Rectangle(-2, 2, 1, -1)
        rect2 = Rectangle(-1, 1, 2, -2)
        self.assert_result(4, rect1, rect2)
    
    def test_overlapping_top_right_bottom_left(self):
        rect1 = Rectangle(-1, -1, 2, 2)
        rect2 = Rectangle(-2, -2, 1, 1)
        self.assert_result(4, rect1, rect2)

    def test_entire_overlapping(self):
        rect1 = Rectangle(0, 0, 1, 1)
        self.assert_result(1, rect1, rect1)

    
if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 16, 2020 \[Easy\] Zig-Zag Distinct LinkedList
--- 
> **Question:** Given a linked list with DISTINCT value, rearrange the node values such that they appear in alternating `low -> high -> low -> high ...` form. For example, given `1 -> 2 -> 3 -> 4 -> 5`, you should return `1 -> 3 -> 2 -> 5 -> 4`.


**Solution:** [https://repl.it/@trsong/Zig-Zag-Order-of-Distinct-LinkedList](https://repl.it/@trsong/Zig-Zag-Order-of-Distinct-LinkedList)
```py
import unittest
import copy

def zig_zag_order(lst):
    should_increase = True
    prev = dummy = ListNode(-1, lst)
    p = prev.next
    while p and p.next:
        does_increase = p.next.val > p.val
        if should_increase != does_increase:
            second = p.next

            p.next = second.next
            second.next = p

            prev.next = second
            prev = second
        else:
            prev = p
            p = p.next
        should_increase = not should_increase
    return dummy.next
        

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

    def __repr__(self):
        return "{} -> {}".format(str(self.val), str(self.next))

    def to_list(self):
        res = []
        p = self
        while p:
            res.append(p.val)
            p = p.next
        return res


class ZigZagOrderSpec(unittest.TestCase):
    def verify_order(self, original_lst):
        lst = zig_zag_order(copy.deepcopy(original_lst))
        self.assertIsNotNone(lst)
        self.assertEqual(set(original_lst.to_list()), set(lst.to_list()))

        isLessThanPrevious = False
        p = lst.next
        prev = lst
        while p:
            if isLessThanPrevious:
                self.assertLess(p.val, prev.val, "%d in %s" % (p.val, lst))
            else:
                self.assertGreater(p.val, prev.val, "%d in %s" % (p.val, lst))

            isLessThanPrevious = not isLessThanPrevious
            prev = p
            p = p.next

    def test_example(self):
        lst = ListNode.List(1, 2, 3, 4, 5)
        self.verify_order(lst)

    def test_empty_array(self):
        self.assertIsNone(zig_zag_order(None))

    def test_unsorted_list1(self):
        lst = ListNode.List(10, 5, 6, 3, 2, 20, 100, 80)
        self.verify_order(lst)

    def test_unsorted_list2(self):
        lst = ListNode.List(2, 4, 6, 8, 10, 20)
        self.verify_order(lst)

    def test_unsorted_list3(self):
        lst = ListNode.List(3, 6, 5, 10, 7, 20)
        self.verify_order(lst)

    def test_unsorted_list4(self):
        lst = ListNode.List(20, 10, 8, 6, 4, 2)
        self.verify_order(lst)

    def test_unsorted_list5(self):
        lst = ListNode.List(6, 4, 2, 1, 8, 3)
        self.verify_order(lst)

    def test_sorted_list(self):
        lst = ListNode.List(6, 5, 4, 3, 2, 1)
        self.verify_order(lst)
    

if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 15, 2020 \[Easy\] Record the Last N Orders
--- 
> **Question:** You run an e-commerce website and want to record the last `N` order ids in a log. Implement a data structure to accomplish this, with the following API:
>
> - `record(order_id)`: adds the order_id to the log
> - `get_last(i)`: gets the ith last element from the log. `i` is guaranteed to be smaller than or equal to `N`.
> 
> You should be as efficient with time and space as possible.

**Solution with Circular Buffer:** [https://repl.it/@trsong/Record-the-Last-N-Orders](https://repl.it/@trsong/Record-the-Last-N-Orders)
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
        physical_index = (self.next_index - i) % len(self.logs)
        log = self.logs[physical_index]
        if log is None:
            raise ValueError('Log %d does not exists' % i)
        else:
            return log


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
    unittest.main(exit=False)
```

### Aug 14, 2020 \[Medium\] Sort Linked List
--- 
> **Question:** Given a linked list, sort it in `O(n log n)` time and constant space.
>
> For example, the linked list `4 -> 1 -> -3 -> 99` should become `-3 -> 1 -> 4 -> 99`.

**Solution with Merge Sort:** [https://repl.it/@trsong/Sort-Linked-List](https://repl.it/@trsong/Sort-Linked-List)
```py
import unittest

def sort_linked_list(head):
    return MergeSort.sort(head)


class MergeSort(object):
    @staticmethod
    def sort(lst):
        if not lst or not lst.next:
            return lst

        l1, l2 = MergeSort.partition(lst)
        sorted_l1 = MergeSort.sort(l1)
        sorted_l2 = MergeSort.sort(l2)
        return MergeSort.merge(sorted_l1, sorted_l2)

    @staticmethod
    def partition(lst):
        slow = fast = ListNode(-1, lst)
        fast = fast.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        half_lst = slow.next
        slow.next = None
        return lst, half_lst

    @staticmethod
    def merge(lst1, lst2):
        p = dummy = ListNode(-1)
        while lst1 and lst2:
            if lst1.val < lst2.val:
                p.next = lst1
                lst1 = lst1.next
            else:
                p.next = lst2
                lst2 = lst2.next
            p = p.next 
        
        if lst1:
            p.next = lst1
        elif lst2:
            p.next = lst2
        
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

    def __repr__(self):
        return "{} -> {}".format(str(self.val), str(self.next))

    @staticmethod
    def List(*vals):
        p = dummy = ListNode(-1)
        for v in vals:
            p.next = ListNode(v)
            p = p.next
        return dummy.next


class SortLinkedListSpec(unittest.TestCase):
    def test_example(self):
        lst = ListNode.List(4, 1, -3, 99)
        expected = ListNode.List(-3, 1, 4, 99)
        self.assertEqual(expected, sort_linked_list(lst))

    def test_empty_list(self):
        self.assertIsNone(sort_linked_list(None))

    def test_list_with_one_element(self):
        lst = ListNode.List(1)
        expected = ListNode.List(1)
        self.assertEqual(expected, sort_linked_list(lst))

    def test_already_sorted_list(self):
        lst = ListNode.List(1, 2, 3, 4, 5)
        expected = ListNode.List(1, 2, 3, 4, 5)
        self.assertEqual(expected, sort_linked_list(lst))

    def test_list_in_descending_order(self):
        lst = ListNode.List(5, 4, 3, 2, 1)
        expected = ListNode.List(1, 2, 3, 4, 5)
        self.assertEqual(expected, sort_linked_list(lst))

    def test_list_with_duplicated_elements(self):
        lst = ListNode.List(1, 1, 3, 2, 1, 2, 1, 1, 3)
        expected = ListNode.List(1, 1, 1, 1, 1, 2, 2, 3, 3)
        self.assertEqual(expected, sort_linked_list(lst))

    def test_binary_list(self):
        lst = ListNode.List(0, 1, 0, 1, 0, 1)
        expected = ListNode.List(0, 0, 0, 1, 1, 1)
        self.assertEqual(expected, sort_linked_list(lst))

    def test_odd_length_list(self):
        lst = ListNode.List(4, 1, 2)
        expected = ListNode.List(1, 2, 4)
        self.assertEqual(expected, sort_linked_list(lst))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 13, 2020 \[Medium\] Nearest Larger Number
--- 
> **Question:** Given an array of numbers and an index `i`, return the index of the nearest larger number of the number at index `i`, where distance is measured in array indices.
>
> For example, given `[4, 1, 3, 5, 6]` and index `0`, you should return `3`.
>
> If two distances to larger numbers are the equal, then return any one of them. If the array at i doesn't have a nearest larger integer, then return null.
>
> **Follow-up:** If you can preprocess the array, can you do this in constant time?

**Thoughts:** The logical is exactly the same as [**find next largest number on the right**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-17-2020-medium-index-of-larger-next-number) except we have to compare next largest number from both left and right for this question.

**Solution with Stack:** [https://repl.it/@trsong/Nearest-Larger-Number](https://repl.it/@trsong/Nearest-Larger-Number)
```py
import unittest

class NearestLargerNumberFinder(object):
    def __init__(self, nums):
        right_larger_indices = NearestLargerNumberFinder.previous_larger_number(nums)
        reverse_left_larger_indices = NearestLargerNumberFinder.previous_larger_number(nums[::-1])
        n = len(nums)

        self.nearest_indiecs = [None] * n
        for i in xrange(n):
            j = n-1-i
            left_index = n-1-reverse_left_larger_indices[j] if reverse_left_larger_indices[j] is not None else None
            right_index = right_larger_indices[i]
            if left_index is not None and right_index is not None:
                self.nearest_indiecs[i] = left_index if nums[left_index] < nums[right_index] else right_index
            elif left_index is None:
                self.nearest_indiecs[i] = right_index
            elif right_index is None:
                self.nearest_indiecs[i] = left_index

    def search_index(self, i):
        return self.nearest_indiecs[i]

    @staticmethod
    def previous_larger_number(nums):
        if not nums:
            return []

        n = len(nums)
        stack = []
        res = [None] * n
        for i in xrange(n-1, -1, -1):
            while stack and nums[i] >= nums[stack[-1]]:
                stack.pop()
            if stack:
                res[i] = stack[-1]
            stack.append(i)
        return res


class LargerNumberSpec(unittest.TestCase):
    def assert_all_result(self, nums, expected):
        finder = NearestLargerNumberFinder(nums)
        self.assertEqual(expected, map(finder.search_index, range(len(nums))))

    def test_random_list(self):
        nums = [3, 2, 5, 6, 9, 8]
        expected = [2, 0, 3, 4, None, 4]
        self.assert_all_result(nums, expected)

    def test_empty_list(self):
        self.assert_all_result([], [])

    def test_asecending_list(self):
        nums = [0, 1, 2, 2, 3, 3, 3, 4, 5]
        expected = [1, 2, 4, 4, 7, 7, 7, 8, None]
        self.assert_all_result(nums, expected)

    def test_descending_list(self):
        nums = [9, 8, 8, 7, 4, 3, 2, 1, 0, -1]
        expected = [None, 0, 0, 2, 3, 4, 5, 6, 7, 8]
        self.assert_all_result(nums, expected)

    def test_up_down_up(self):
        nums = [0, 1, 2, 1, 2, 3, 4, 5]
        expected = [1, 2, 5, 4, 5, 6, 7, None]
        self.assert_all_result(nums, expected)

    def test_up_down_up2(self):
        nums = [0, 4, -1, 2]
        expected = [1, None, 3, 1]
        self.assert_all_result(nums, expected)

    def test_down_up_down(self):
        nums = [9, 5, 6, 3]
        expected = [None, 2, 0, 2]
        self.assert_all_result(nums, expected)
    
    def test_up_down(self):
        nums = [11, 21, 31, 3]
        expected = [1, 2, None, 2]
        self.assert_all_result(nums, expected)

    def test_random_order(self):
        nums = [4, 3, 5, 2, 4, 7]
        expected = [2, 0, 5, 4, 2, None]
        self.assert_all_result(nums, expected)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 12, 2020 LC 1021 \[Easy\] Remove One Layer of Parenthesis
--- 
> **Question:** Given a valid parenthesis string (with only '(' and ')', an open parenthesis will always end with a close parenthesis, and a close parenthesis will never start first), remove the outermost layers of the parenthesis string and return the new parenthesis string.
>
> If the string has multiple outer layer parenthesis (ie (())()), remove all outer layers and construct the new string. So in the example, the string can be broken down into (()) + (). By removing both components outer layer we are left with () + '' which is simply (), thus the answer for that input would be ().

**Example 1:**
```py
Input: '(())()'
Output: '()'
```

**Example 2:**
```py
Input: '(()())'
Output: '()()'
```

**Example 3:**
```py
Input: '()()()'
Output: ''
```

**Solution:** [https://repl.it/@trsong/Remove-One-Layer-of-Parenthesis](https://repl.it/@trsong/Remove-One-Layer-of-Parenthesis)
```py
import unittest

class ValidCharFilter(object):
    def __init__(self):
        self.balance = 0
    
    def is_valid_char(self, c):
        if c == '(':
            self.balance += 1
            return self.balance > 1
        else:
            self.balance -= 1
            return self.balance > 0
    

def remove_one_layer_parenthesis(s):
    f = ValidCharFilter()
    return ''.join(filter(f.is_valid_char, s))


class RemoveOneLayerParenthesisSpec(unittest.TestCase):
    def test_example(self):
        s = '(())()'
        expected = '()'
        self.assertEqual(expected, remove_one_layer_parenthesis(s))

    def test_example2(self):
        s = '(()())'
        expected = '()()'
        self.assertEqual(expected, remove_one_layer_parenthesis(s))

    def test_example3(self):
        s = '()()()'
        expected = ''
        self.assertEqual(expected, remove_one_layer_parenthesis(s))

    def test_empty_string(self):
        self.assertEqual('', remove_one_layer_parenthesis(''))

    def test_nested_parenthesis(self):
        s = '(()())(())'
        expected = '()()()'
        self.assertEqual(expected, remove_one_layer_parenthesis(s))

    def test_complicated_parenthesis(self):
        s = '(()())(())(()(()))'
        expected = '()()()()(())'
        self.assertEqual(expected, remove_one_layer_parenthesis(s))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 11, 2020 \[Easy\] Full Binary Tree
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

**Solution with Recursion:** [https://repl.it/@trsong/Prune-to-Full-Binary-Tree](https://repl.it/@trsong/Prune-to-Full-Binary-Tree)
```py
import unittest

def full_tree_prune(root):
    if not root:
        return None
    
    updated_left = full_tree_prune(root.left)
    updated_right = full_tree_prune(root.right)

    if updated_left and updated_right:
        root.left = updated_left
        root.right = updated_right
        return root
    elif updated_left:
        return updated_left
    elif updated_right:
        return updated_right
    else:
        return root


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __eq__(self, other):
        return other and other.val == self.val and other.left == self.left and other.right == self.right

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
        import copy
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


### Aug 10, 2020 LC 261 \[Medium\] Graph Valid Tree
---
> **Question:** Given n nodes labeled from 0 to n-1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.

**Example 1:**
```py
Input: n = 5, and edges = [[0,1], [0,2], [0,3], [1,4]]
Output: True
```

**Example 2:**
```py
Input: n = 5, and edges = [[0,1], [1,2], [2,3], [1,3], [1,4]]
Output: False
```

**My thoughts:** A tree is a connected graph with `n-1` edges. As it is undirected graph and a tree must not have cycle, we can use Disjoint Set (Union-Find) to detect cycle: start from empty graph, for each edge we add to the graph, check both ends of that edge and see if it is already connected. 

**Solution with Union-Find:** [https://repl.it/@trsong/Graph-Valid-Tree](https://repl.it/@trsong/Graph-Valid-Tree)
```py
import unittest

class UnionFind(object):
    def __init__(self, n):
        self.parent = [-1] * n

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


def is_valid_tree(n, edges):
    if n == 0:
        return True
    if len(edges) != n - 1:
        return False
    
    uf = UnionFind(n)
    for u, v in edges:
        if uf.is_connected(u, v):
            return False
        uf.union(u, v)
    return True


class IsValidTreeSpec(unittest.TestCase):
    def test_example(self):
        n, edges = 5, [[0, 1], [0, 2], [0, 3], [1, 4]]
        self.assertTrue(is_valid_tree(n, edges))

    def test_example2(self):
        n, edges = 5, [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]
        self.assertFalse(is_valid_tree(n, edges))

    def test_empty_graph(self):
        n, edges = 0, []
        self.assertTrue(is_valid_tree(n, edges))

    def test_one_node_graph(self):
        n, edges = 1, []
        self.assertTrue(is_valid_tree(n, edges))

    def test_disconnected_graph(self):
        n, edges = 3, []
        self.assertFalse(is_valid_tree(n, edges))

    def test_disconnected_graph2(self):
        n, edges = 5, [[0, 1], [2, 3], [3, 4], [4, 2]]
        self.assertFalse(is_valid_tree(n, edges))

    def test_disconnected_graph3(self):
        n, edges = 4, [[0, 1], [2, 3]]
        self.assertFalse(is_valid_tree(n, edges))

    def test_tree_with_cycle(self):
        n, edges = 4, [[0, 1], [0, 2], [0, 3], [3, 2]]
        self.assertFalse(is_valid_tree(n, edges))

    def test_binary_tree(self):
        n, edges = 7, [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]]
        self.assertTrue(is_valid_tree(n, edges))


if __name__ == "__main__":
    unittest.main(exit=False)
```

### Aug 9, 2020 \[Medium\] Making Changes
---
> **Question:** Given a list of possible coins in cents, and an amount (in cents) n, return the minimum number of coins needed to create the amount n. If it is not possible to create the amount using the given coin denomination, return None.

**Example:**
```py
make_change([1, 5, 10, 25], 36)  # gives 3 coins (25 + 10 + 1) 
```

**Solution with DP:** [https://repl.it/@trsong/Making-Changes-Problem](https://repl.it/@trsong/Making-Changes-Problem)
```py
import unittest
import sys

def make_change(coins, target):
    if target == 0:
        return 0

    # Let dp[v] represents smallest # of coin for value v
    # dp[v] = 1 if v is one of coins
    #       = 1 + min(dp[v-coin]) for all valid coin
    dp = [sys.maxint] * (target+1)
    valid_coins = filter(lambda coin: 0 < coin <= target, coins)
    if not valid_coins:
        return None

    dp[0] = 0
    for coin in valid_coins:
        dp[coin] = 1

    for v in xrange(1, target+1):
        for coin in valid_coins:
            if v > coin:
                dp[v] = min(dp[v], 1 + dp[v-coin])
    
    return dp[target] if dp[target] != sys.maxint else None


class MakeChangeSpec(unittest.TestCase):
    def test_example(self):
        target, coins = 36, [1, 5, 10, 25]
        expected = 3  # 25 + 10 + 1
        self.assertEqual(expected, make_change(coins, target))
    
    def test_target_is_zero(self):
        self.assertEqual(0, make_change([1, 2, 3], 0))
        self.assertEqual(0, make_change([], 0))

    def test_unable_to_reach_target(self):
        target, coins = -1, [1, 2, 3]
        self.assertIsNone(make_change(coins, target))

    def test_unable_to_reach_target2(self):
        target, coins = 10, []
        self.assertIsNone(make_change(coins, target))

    def test_greedy_approach_fails(self):
        target, coins = 11, [9, 6, 5, 1]
        expected = 2  # 5 + 6
        self.assertEqual(expected, make_change(coins, target))

    def test_use_same_coin_multiple_times(self):
        target, coins = 12, [1, 2, 3]
        expected = 4  # 3 + 3 + 3 + 3
        self.assertEqual(expected, make_change(coins, target))

    def test_should_produce_minimum_number_of_changes(self):
        target, coins = 30, [25, 10, 5]
        expected = 2  # 25 + 5
        self.assertEqual(expected, make_change(coins, target))
    
    def test_impossible_get_answer(self):
        target, coins = 4, [3, 5, 7]
        self.assertIsNone(make_change(coins, target))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 8, 2020 \[Easy\] Flatten a Nested Dictionary
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

**Solution with DFS:** [https://repl.it/@trsong/Flatten-Nested-Dictionary](https://repl.it/@trsong/Flatten-Nested-Dictionary)
```py
import unittest

def flatten_dictionary(dictionary):
    stack = [(dictionary, None)]
    res = {}
    while stack:
        val, prefix = stack.pop()
        if type(val) is dict:
            for k, v in val.items():
                updated_prefix = k if prefix is None else "{}.{}".format(prefix, k)
                stack.append((v, updated_prefix))
        else:
            res[prefix] = val
    return res


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

### Aug 7, 2020 \[Hard\] Shortest Uphill and Downhill Route
---
> **Question:** A competitive runner would like to create a route that starts and ends at his house, with the condition that the route goes entirely uphill at first, and then entirely downhill.
>
> Given a dictionary of places of the form `{location: elevation}`, and a dictionary mapping paths between some of these locations to their corresponding distances, find the length of the shortest route satisfying the condition above. Assume the runner's home is location `0`.

**Example:**
```py
Suppose you are given the following input:

elevations = {0: 5, 1: 25, 2: 15, 3: 20, 4: 10}
paths = {
    (0, 1): 10,
    (0, 2): 8,
    (0, 3): 15,
    (1, 3): 12,
    (2, 4): 10,
    (3, 4): 5,
    (3, 0): 17,
    (4, 0): 10
}

In this case, the shortest valid path would be 0 -> 2 -> 4 -> 0, with a distance of 28.
```

**My thoughts:** Break the graph in to uphill and downhill graph separately. For uphill graph, we can calculate shortest distance from 0 to target node. For downhill graph, we want to calculate shortest distance from node to 0. But how? One way is that we can reverse the edge of graph and convert the problem to find shortest distance from 0 to node. Once get shorest distance from and to node, we can sum them up and find the shorest combined distance which will give the final solution.

Then how to do we find shortest distance in directed graph? We can use Dijkstra's algorithm that will be `O(V + E log E)` due to priority queue used in the algorithm. However, notice that in this problem neither uphill graph nor downhil graph can form a cycle. Then we will have a DAG. The shortest distance in DAG can be found in `O(V + E)`. Therefore our solution will have `O(V + E)` in time complexity.

**Solution with Topological Sort:** [https://repl.it/@trsong/Shortest-Uphill-and-Downhill-Route](https://repl.it/@trsong/Shortest-Uphill-and-Downhill-Route)
```py
import unittest
from collections import defaultdict

def shortest_uphill_downhill_cycle(elevations, paths):
    uphill_neighbors = defaultdict(list)
    reverse_downhill_neighbors = defaultdict(list)
    for u, v in paths:
        if elevations[u] < elevations[v]:
            uphill_neighbors[u].append(v)
        else:
            reverse_downhill_neighbors[v].append(u)
    
    uphill_order = topological_sort(uphill_neighbors, elevations)
    shorest_distance_to = shortest_distance(uphill_order, uphill_neighbors, paths)

    reverse_downhill_order = topological_sort(reverse_downhill_neighbors, elevations)
    shortest_distance_from = shortest_distance(reverse_downhill_order, reverse_downhill_neighbors, paths, reversed=True)

    del shorest_distance_to[0]
    del shortest_distance_from[0]
    shortest_cycle_through = map(lambda u: shorest_distance_to[u] + shortest_distance_from[u], elevations.keys())
    return min(shortest_cycle_through) 

    
def topological_sort(neighbors, elevations):
    class Vertex_State:
        UNVISITED = 0
        VISITING = 1
        VISITED = 2

    vertext_states = {node: Vertex_State.UNVISITED for node in elevations}
    reverse_top_order = []

    for node, node_state in vertext_states.items():
        if node_state is not Vertex_State.UNVISITED:
            continue
        
        stack = [node]
        while stack:
            cur = stack[-1]
            if vertext_states[cur] is Vertex_State.VISITED:
                stack.pop()
            elif vertext_states[cur] is Vertex_State.VISITING:
                vertext_states[cur] = Vertex_State.VISITED
                reverse_top_order.append(cur)
            else:
                # vertext_states[cur] is Vertex_State.UNVISITED
                vertext_states[cur] = Vertex_State.VISITING
                for nb in neighbors[cur]:
                    if vertext_states[nb] is Vertex_State.UNVISITED:
                        stack.append(nb)
    
    return reverse_top_order[::-1]


def shortest_distance(top_order, neighbors, paths, reversed=False):
    node_distance = defaultdict(lambda: float('inf'))
    node_distance[0] = 0
    for u in top_order:
        for v in neighbors[u]:
            w = float('inf')
            if reversed:
                w = paths.get((v, u), float('inf'))
            else:
                w = paths.get((u, v), float('inf'))
            node_distance[v] = min(node_distance[v], node_distance[u] + w)
    return node_distance


class ShortestUphillDownhillCycleSpec(unittest.TestCase):
    def test_example(self):
        elevations = {0: 5, 1: 25, 2: 15, 3: 20, 4: 10}
        paths = {
            (0, 1): 10,
            (0, 2): 8,
            (0, 3): 15,
            (1, 3): 12,
            (2, 4): 10,
            (3, 4): 5,
            (3, 0): 17,
            (4, 0): 10
        }
        expected = 28  # 0 -> 2 -> 4 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))

    def test_choose_between_downhill_routes(self):
        """
         1
       / | \
      2  |  3
       \ | /
         0
        """
        elevations = {0: 0, 1: 10, 2: 5, 3: 7}
        paths = {
            (0, 1): 10,
            (1, 2): 20,
            (2, 0): 30,
            (1, 3): 5,
            (3, 0): 6
        }
        expected = 21  # 0 -> 1 -> 3 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))
    
    def test_star_graph(self):
        """
          3
          |
          0
         / \
        1   2
        """
        elevations = {0: 0, 1: 1, 2: 2, 3: 3}
        paths = {
            (0, 1): 1,
            (1, 0): 10,
            (0, 2): 8,
            (2, 0): 5,
            (0, 3): 4,
            (3, 0): 6
        }
        expected = 10  # 0 -> 3 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))
    
    def test_clockwise_vs_conterclockwise(self):
        """
        0 - 1
        |   |
        3 - 2 
        """
        elevations = {0: 0, 1: 1, 2: 2, 3: 3}
        paths = {
            (0, 1): 10,
            (1, 2): 20,
            (2, 3): 30,
            (3, 0): 99,
            (0, 3): 1,
            (3, 2): 2,
            (2, 1): 3,
            (1, 0): 4
        }
        expected = 10  # 0 -> 3 -> 2 -> 1 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))
    
    def test_choose_downhill_or_uphill(self):
        """
         1
       / | \
      2  |  3
       \ | /
         0
        """
        elevations = {0: 0, 1: 50, 2: 25, 3: 100}
        paths = {
            (0, 1): 10,
            (1, 0): 9999,
            (1, 2): 5,
            (2, 0): 3,
            (1, 3): 1,
            (3, 0): 0,
            (0, 3): 999,
            (0, 2): 999
        }
        expected = 11  # 0 -> 1 -> 3 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 6, 2020 LC 78 \[Medium\] Generate All Subsets
---
> **Question:** Given a list of unique numbers, generate all possible subsets without duplicates. This includes the empty set as well.

**Example:**
```py
generate_all_subsets([1, 2, 3])
# [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]
```

**Solution with Recursion:** [https://repl.it/@trsong/Generate-All-Subsets-with-recursion](https://repl.it/@trsong/Generate-All-Subsets-with-recursion)
```py
from functools import reduce

def generate_all_subsets(nums):
    return reduce(
        lambda accu_subsets, elem: accu_subsets + map(lambda subset: subset + [elem], accu_subsets),
        nums,
        [[]])
```

**Solution with Backtracking:** [https://repl.it/@trsong/Generate-All-the-Subsets](https://repl.it/@trsong/Generate-All-the-Subsets)
```py
import unittest

def generate_all_subsets(nums):
    res = []
    backtrack(0, res, [], nums)
    return res


def backtrack(next_index, res, accu, nums):
    res.append(accu[:])
    for i in xrange(next_index, len(nums)):
        accu.append(nums[i])
        backtrack(i+1, res, accu, nums)
        accu.pop()


class GenerateAllSubsetSpec(unittest.TestCase):
    def test_example(self):
        nums = [1, 2, 3]
        expected = [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]
        self.assertItemsEqual(expected, generate_all_subsets(nums))

    def test_empty_list(self):
        nums = []
        expected = [[]]
        self.assertItemsEqual(expected, generate_all_subsets(nums))

    def test_one_elem_list(self):
        nums = [1]
        expected = [[], [1]]
        self.assertItemsEqual(expected, generate_all_subsets(nums))

    def test_two_elem_list(self):
        nums = [1, 2]
        expected = [[1], [2], [1, 2], []]
        self.assertItemsEqual(expected, generate_all_subsets(nums))

    def test_four_elem_list(self):
        nums = [1, 2, 3, 4]
        expected = [
            [], 
            [1], [2], [3],  [4],
            [1, 2], [1, 3], [2, 3], [1, 4], [2, 4], [3, 4],
            [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4], [1, 2, 3, 4]
        ]
        self.assertItemsEqual(expected, generate_all_subsets(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 5, 2020 \[Easy\] Array of Equal Parts
---
> **Question:** Given an array containing only positive integers, return if you can pick two integers from the array which cuts the array into three pieces such that the sum of elements in all pieces is equal.

**Example 1:**
```py
Input: [2, 4, 5, 3, 3, 9, 2, 2, 2]
Output: True
Explanation: choosing the number 5 and 9 results in three pieces [2, 4], [3, 3] and [2, 2, 2]. Sum = 6.
```

**Example 2:**
```py
Input: [1, 1, 1, 1]
Output: False
```

**Solution with Prefix Sum and Two Pointers:** [https://repl.it/@trsong/Array-of-Equal-Parts](https://repl.it/@trsong/Array-of-Equal-Parts)
```py
import unittest

def contains_3_equal_parts(nums):
    if not nums:
        return False

    prefix_sum = nums[:]
    n = len(nums)
    for i in xrange(1, n):
        prefix_sum[i] += prefix_sum[i-1]

    i, j = 0, n-1
    total = prefix_sum[n-1]
    while i < j:
        left_sum = prefix_sum[i] - nums[i]
        right_sum = total - prefix_sum[j]
        if left_sum < right_sum:
            i += 1
        elif left_sum > right_sum:
            j -= 1
        else:
            # left_sum == right_sum
            mid_sum = prefix_sum[j] - nums[j] - prefix_sum[i]
            if mid_sum == left_sum:
                return True
            i += 1
            j -= 1

    return False
        

class Contains3EqualPartSpec(unittest.TestCase):
    def test_example(self):
        nums = [2, 4, 5, 3, 3, 9, 2, 2, 2]
        # Remove 5, 9 to break array into [2, 4], [3, 3] and [2, 2, 2]
        self.assertTrue(contains_3_equal_parts(nums))

    def test_example2(self):
        nums = [1, 1, 1, 1]
        self.assertFalse(contains_3_equal_parts(nums))

    def test_empty_array(self):
        self.assertFalse(contains_3_equal_parts([]))

    def test_two_element_array(self):
        nums = [1, 2]
        # [], [], []
        self.assertTrue(contains_3_equal_parts(nums))

    def test_three_element_array(self):
        nums = [1, 2, 3]
        self.assertFalse(contains_3_equal_parts(nums))

    def test_symmetic_array(self):
        nums = [1, 2, 4, 3, 5, 2, 1]
        # remove 4, 5 gives [1, 2], [3], [2, 1]
        self.assertTrue(contains_3_equal_parts(nums))

    def test_sum_not_divisiable_by_3(self):
        nums = [2, 2, 2, 2, 2, 2]
        self.assertFalse(contains_3_equal_parts(nums))

    def test_ascending_array(self):
        nums = [1, 2, 3, 3, 3, 3, 4, 6]
        # remove 3, 4 gives [1, 2, 3], [3, 3], [6]
        self.assertTrue(contains_3_equal_parts(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```



### Aug 4, 2020 \[Medium\] M Smallest in K Sorted Lists
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

**Solution with Priority Queue:** [https://repl.it/@trsong/Find-M-Smallest-in-K-Sorted-Lists](https://repl.it/@trsong/Find-M-Smallest-in-K-Sorted-Lists)
```py
import unittest
from Queue import PriorityQueue

def find_m_smallest(ksorted_list, m):
    sorted_iterators = map(iter, ksorted_list)
    pq = PriorityQueue()

    for it in sorted_iterators:
        num = next(it, None)
        if num is not None:
            pq.put((num, it))

    while m > 1 and not pq.empty():
        _, it = pq.get()
        next_num = next(it, None)
        if next_num is not None:
            pq.put((next_num, it))
        m -= 1

    return None if pq.empty() else pq.get()[0]


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


### Aug 3, 2020 \[Hard\] Graph Coloring
---
> **Question:** Given an undirected graph represented as an adjacency matrix and an integer `k`, determine whether each node in the graph can be colored such that no two adjacent nodes share the same color using at most `k` colors.

**My thoughts:** Solve this problem with backtracking. For each node, testing all colors one-by-one; if it turns out there is something wrong with current color, we will backtrack to test other colors.

**Solution with Backtracking:** [https://repl.it/@trsong/k-Graph-Coloring](https://repl.it/@trsong/k-Graph-Coloring)
```py
import unittest

def solve_graph_coloring(neighbor_matrix, k):
    n = len(neighbor_matrix)
    node_color = [None] * n

    def backtrack(next_node):
        if next_node >= n:
            return True
        
        processed_neighbors = [i for i in xrange(next_node) if neighbor_matrix[next_node][i]]
        for color in xrange(k):
            if any(color == node_color[nb] for nb in processed_neighbors):
                continue
            node_color[next_node] = color
            if backtrack(next_node+1):
                return True
            node_color[next_node] = None
        return False

    return backtrack(0)
        

class SolveGraphColoringSpec(unittest.TestCase):
    @staticmethod
    def generateCompleteGraph(n):
        return [[1 if i != j else 0 for i in xrange(n)] for j in xrange(n)] 

    def test_k2_graph(self):
        k2 = SolveGraphColoringSpec.generateCompleteGraph(2)
        self.assertFalse(solve_graph_coloring(k2, 1))
        self.assertTrue(solve_graph_coloring(k2, 2))
        self.assertTrue(solve_graph_coloring(k2, 3))

    def test_k3_graph(self):
        k3 = SolveGraphColoringSpec.generateCompleteGraph(3)
        self.assertFalse(solve_graph_coloring(k3, 2))
        self.assertTrue(solve_graph_coloring(k3, 3))
        self.assertTrue(solve_graph_coloring(k3, 4))

    def test_k4_graph(self):
        k4 = SolveGraphColoringSpec.generateCompleteGraph(4)
        self.assertFalse(solve_graph_coloring(k4, 3))
        self.assertTrue(solve_graph_coloring(k4, 4))
        self.assertTrue(solve_graph_coloring(k4, 5))

    def test_square_graph(self):
        square = [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ]
        self.assertFalse(solve_graph_coloring(square, 1))
        self.assertTrue(solve_graph_coloring(square, 2))
        self.assertTrue(solve_graph_coloring(square, 3))

    def test_star_graph(self):
        star = [
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0]
        ]
        self.assertFalse(solve_graph_coloring(star, 2))
        self.assertTrue(solve_graph_coloring(star, 3))
        self.assertTrue(solve_graph_coloring(star, 4))

    def test_disconnected_graph(self):
        disconnected = [[0 for _ in xrange(10)] for _ in xrange(10)]
        self.assertTrue(solve_graph_coloring(disconnected, 1))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 2, 2020 \[Medium\] Maximum Circular Subarray Sum
---
> **Question:** Given a circular array, compute its maximum subarray sum in `O(n)` time. A subarray can be empty, and in this case the sum is 0.

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

**Solution with DP and Prefix-Sum:** [https://repl.it/@trsong/Find-Maximum-Circular-Subarray-Sum](https://repl.it/@trsong/Find-Maximum-Circular-Subarray-Sum)
```py
import unittest

def max_circular_sum(nums):
    return max(max_subarray_sum(nums), max_circular_subarray_sum(nums))


def max_subarray_sum(nums):
    n = len(nums)
    # Let dp[n] represents max subarray max ends at index n-1
    dp = [0] * (n+1)
    for i in xrange(1, n+1):
        dp[i] = nums[i-1] + max(dp[i-1], 0)
    return max(dp)


def max_circular_subarray_sum(nums):
    if not nums:
        return 0

    left_max_sums = max_prefix_sums(nums)
    right_max_sums = reversed(max_prefix_sums(reversed(nums)))
    combined_sum = map(sum, zip(left_max_sums, right_max_sums))
    return max(combined_sum)


def max_prefix_sums(stream):
    res = []
    prefix_sum = 0
    max_prefix_sum = 0

    for num in stream:
        max_prefix_sum = max(max_prefix_sum, prefix_sum)
        res.append(max_prefix_sum)
        prefix_sum += num

    return res


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


### Aug 1, 2020 LC 934 \[Medium\] Shortest Bridge
---
> **Question:** In a given 2D binary array A, there are two islands.  (An island is a 4-directionally connected group of 1s not connected to any other 1s.)
>
> Now, we may change 0s to 1s so as to connect the two islands together to form 1 island.
>
> Return the smallest number of 0s that must be flipped.  (It is guaranteed that the answer is at least 1.)

**Example 1:**
```py
Input: 
[
    [0, 1],
    [1, 0]
]
Output: 1
```

**Example 2:**
```py
Input: 
[
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 1]
]
Output: 2
```

**Example 3:**
```py
Input: 
[
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]
Output: 1
```


**My thoughts:** First color each island with 2 and -2. Then search from each island by bi-directional bfs (2-way bfs). Like below

```py
Color Each Island, mark coastline1 and coastline2 to be start queue and end queue

  .   .   .   .   .   .   .   .
  .   .   .   2   .   2   2   .
  .   .   2   2   2   2   .   .
  .   2   2   2   2   2   .   .
  .   2   .   .   2   2   .   .
  .   .   .   .   2   2   .   .
  .   .   .   .   2   .   .   .
  .   .   .   .   2   .   .   .
  .   .   .   .   .   .   .   .
  .  -2   .   .   .   .   .   .
  .  -2  -2   .   .   .   .   .
  .   .  -2   .   .   .   .   .
  .   .  -2  -2  -2   .   .   .


======= Iteration 1 ======
=> Start Queue Move 

  .   .   .   3   .   3   3   .
  .   .   3   2   3   2   2   3
  .   3   2   2   2   2   3   .
  3   2   2   2   2   2   3   .
  3   2   3   3   2   2   3   .
  .   3   .   3   2   2   3   .
  .   .   .   3   2   3   .   .
  .   .   .   3   2   3   .   .
  .   .   .   .   3   .   .   .
  .  -2   .   .   .   .   .   .
  .  -2  -2   .   .   .   .   .
  .   .  -2   .   .   .   .   .
  .   .  -2  -2  -2   .   .   .

=> End Queue Move 

  .   .   .   3   .   3   3   .
  .   .   3   2   3   2   2   3
  .   3   2   2   2   2   3   .
  3   2   2   2   2   2   3   .
  3   2   3   3   2   2   3   .
  .   3   .   3   2   2   3   .
  .   .   .   3   2   3   .   .
  .   .   .   3   2   3   .   .
  .  -3   .   .   3   .   .   .
 -3  -2  -3   .   .   .   .   .
 -3  -2  -2  -3   .   .   .   .
  .  -3  -2  -3  -3   .   .   .
  .  -3  -2  -2  -2  -3   .   .


======= Iteration 2 ======
=> Start Queue Move 

  .   .   4   3   4   3   3   4
  .   4   3   2   3   2   2   3
  4   3   2   2   2   2   3   4
  3   2   2   2   2   2   3   4
  3   2   3   3   2   2   3   4
  4   3   4   3   2   2   3   4
  .   4   4   3   2   3   4   .
  .   .   4   3   2   3   4   .
  .  -3   .   4   3   4   .   .
 -3  -2  -3   .   4   .   .   .
 -3  -2  -2  -3   .   .   .   .
  .  -3  -2  -3  -3   .   .   .
  .  -3  -2  -2  -2  -3   .   .

=> End Queue Move 

  .   .   4   3   4   3   3   4
  .   4   3   2   3   2   2   3
  4   3   2   2   2   2   3   4
  3   2   2   2   2   2   3   4
  3   2   3   3   2   2   3   4
  4   3   4   3   2   2   3   4
  .   4   4   3   2   3   4   .
  .  -4   4   3   2   3   4   .
 -4  -3  -4   4   3   4   .   .
 -3  -2  -3  -4   4   .   .   .
 -3  -2  -2  -3  -4   .   .   .
 -4  -3  -2  -3  -3  -4   .   .
 -4  -3  -2  -2  -2  -3  -4   .

```

**Solution with 2-way BFS:** [https://repl.it/@trsong/Shortest-Bridge](https://repl.it/@trsong/Shortest-Bridge)
```py
import unittest
from Queue import deque


SEARCH_DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

def shortest_bridge(grid):
    coastline1, coastline2 = dfs_partition(grid)
    distance = bi_directional_bfs(grid, coastline1, coastline2)
    return distance


def dfs_partition(grid):
    """
    Color first island with 2 and second island with -2. Return coastline of each.
    """
    n, m = len(grid), len(grid[0])
    sea_color = 0
    original_island_color = 1
    current_color = 2
    max_num_island = 2
    res = []

    for r in xrange(n):
        for c in xrange(m):
            if grid[r][c] != original_island_color:
                continue
            
            island_costline = []
            stack = [(r, c)]
            while stack:
                cur_r, cur_c = stack.pop()

                # visited pos has same color
                if grid[cur_r][cur_c] is current_color:
                    continue
                grid[cur_r][cur_c] = current_color

                is_coast = False
                for dr, dc in SEARCH_DIRECTIONS:
                    new_r = cur_r + dr
                    new_c = cur_c + dc
                    if not (0 <= new_r < n and 0 <= new_c < m):
                        continue
                    
                    if grid[new_r][new_c] is sea_color:
                        is_coast = True
                    elif grid[new_r][new_c] is original_island_color:
                        # unvisited pos has original color
                        stack.append((new_r, new_c))

                if is_coast:
                    island_costline.append((cur_r, cur_c))
            
            # the other island has negative color
            current_color = -2
            res.append(island_costline)
            if len(res) == max_num_island:
                return res

    return res


def bi_directional_bfs(grid, coastline1, coastline2):
    iteration = 0
    color = 2
    start_queue, end_queue = deque(coastline1), deque(coastline2)
    while start_queue and end_queue:
        if search_by_level(grid, start_queue, color):
            return 2 * iteration - 2

        if search_by_level(grid, end_queue, -color):
            return 2 * iteration - 1

        iteration += 1
        color += 1

    return None


def search_by_level(grid, queue, current_color):
    n, m = len(grid), len(grid[0])
    sign = 1 if current_color > 0 else -1
    sea_color = 0

    for _ in xrange(len(queue)):
        cur_r, cur_c = queue.popleft()

        if grid[cur_r][cur_c] * sign < 0:
            # found the other island
            return True

        if grid[cur_r][cur_c] != sea_color and abs(grid[cur_r][cur_c]) < abs(current_color):
            # visited pos has smaller color 
            continue
        
        grid[cur_r][cur_c] = current_color
        for dr, dc in SEARCH_DIRECTIONS:
            new_r = cur_r + dr
            new_c = cur_c + dc
            if not (0 <= new_r < n and 0 <= new_c < m):
                continue 
            
            if grid[new_r][new_c] is sea_color or grid[new_r][new_c] * sign < 0:
                queue.append((new_r, new_c))
    
    return False
    

class ShortestBridgeSpec(unittest.TestCase):
    @staticmethod
    def print_grid(grid):
        print
        for row in grid:
            print ' '.join('{:3}'.format(v if v != 0 else '.') for v in row)
        print

    def test_example(self):
        grid = [
            [0, 1], 
            [1, 0]
        ]
        self.assertEqual(1, shortest_bridge(grid))  # flip (1, 1)

    def test_example2(self):
        grid = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1]
        ]
        self.assertEqual(2, shortest_bridge(grid))  # flip (0, 2) and (1, 2)

    def test_example3(self):
        grid = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
        self.assertEqual(1, shortest_bridge(grid))  # flip (1, 1)

    def test_1D_grid(self):
        grid = [[1, 0, 1]]
        self.assertEqual(1, shortest_bridge(grid))

    def test_1D_grid2(self):
        grid = [[1, 0, 0, 1]]
        self.assertEqual(2, shortest_bridge(grid))
    
    def test_not_all_0_are_connected(self):
        grid = [
            [0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 0],
        ]
        self.assertEqual(2, shortest_bridge(grid))  # flip (1, 2) and (1, 3)

    def test_complicated_islands(self):
        grid = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
        ]
        self.assertEqual(4, shortest_bridge(grid))  # flip (5, 1), (6, 1), (7, 1) and (8, 1)

    def test_find_shortest_bridge(self):
        grid = [
            [1, 1, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
        self.assertEqual(1, shortest_bridge(grid))  # flip (3, 0)

    def test_performance(self):
        n, m = 15, 15
        grid = [[0 for _ in xrange(m)] for _ in xrange(n)]
        grid[0][0] = 1
        grid[n-1][m-1] = 1
        self.assertEqual(n+m-3, shortest_bridge(grid))


if __name__ == '__main__':
    unittest.main(exit=False)
```