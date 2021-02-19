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