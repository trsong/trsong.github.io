---
layout: post
title:  "Daily Coding Problems Feb to Apr"
date:   2020-02-01 22:22:32 -0700
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

### Mar 31, 2020 \[Medium\] Root to Leaf Numbers Summed
---
> **Question:** A number can be constructed by a path from the root to a leaf. Given a binary tree, sum all the numbers that can be constructed from the root to all leaves.

**Example:**
```py
Input:
       1
     /   \
    2     3
   / \
  4   5
Output: 262
Explanation: 124 + 125 + 13 = 262
```

### Mar 30, 2020 \[Easy\] Permutation with Given Order
---
> **Question:** A permutation can be specified by an array `P`, where `P[i]` represents the location of the element at `i` in the permutation. For example, `[2, 1, 0]` represents the permutation where elements at the index `0` and `2` are swapped.
>
> Given an array and a permutation, apply the permutation to the array. 
>
> For example, given the array `["a", "b", "c"]` and the permutation `[2, 1, 0]`, return `["c", "b", "a"]`.

**My thoughts:** In-place solution requires swapping `i` with `j` if `j > i`. However, if `j < i`, then `j`'s position has been swapped, we backtrack recursively to find `j`'s new position.

**Solution:** [https://repl.it/@trsong/Permutation-with-Given-Order](https://repl.it/@trsong/Permutation-with-Given-Order)
```py
import unittest

def permute(arr, order):
    for i in xrange(len(order)):
        index_to_swap = order[i]
        
        # Check index if it has already been swapped before
        while index_to_swap < i:
            index_to_swap = order[index_to_swap]

        arr[i], arr[index_to_swap] = arr[index_to_swap], arr[i]
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
    unittest.main(exit=False)
```

### Mar 29, 2020 \[Medium\] Rearrange String to Have Different Adjacent Characters
---
> **Question:** Given a string s, rearrange the characters so that any two adjacent characters are not the same. If this is not possible, return `None`.
>
> For example, if `s = yyz` then return `yzy`. If `s = yyy` then return `None`.

**My thougths:** This problem is basically ["LC 358 Rearrange String K Distance Apart"](https://trsong.github.io/python/java/2019/08/02/DailyQuestionsAug/#aug-24-2019-lc-358-hard-rearrange-string-k-distance-apart) with `k = 2`. The idea is to apply greedy approach, with a window of 2, choose the safest two remaining charactors until either all characters are picked, or just mulitple copy of one character left.

For example, for input string: `"aaaabc"`
1. Pick `a`, `b`. Remaining `"aaac"`. Result: `"ab"`
2. Pick `a`, `c`. Remaining `"aa"`. Result: `"abac"`
3. We can no longer proceed as we cannot pick same character as it violate adjacency requirement

Another Example, for input string: `"abbccc"`
1. Pick `c`, `b`. Remaining `"abcc"`. Result: `"ab"`
2. Pick `c`, `a`. Remaining `"bc"`. Result: `"abca"`
3. Pick `b`, `c`. Result: `"abcabc"`


**Solution with Priority Queue:** [https://repl.it/@trsong/Rearrange-String-to-Have-Different-Adjacent-Characters](https://repl.it/@trsong/Rearrange-String-to-Have-Different-Adjacent-Characters)
```py
import unittest
from Queue import PriorityQueue

def rearrange_string(original_string):
    freq_map = {}
    for ch in original_string:
        freq_map[ch] = freq_map.get(ch, 0) + 1
    
    max_heap = PriorityQueue()
    for ch, count in freq_map.items():
        max_heap.put((-count, ch))

    res = []
    while not max_heap.empty():
        neg_count1, first_ch = max_heap.get()
        remain_first_count = -neg_count1 - 1
        res.append(first_ch)

        if max_heap.empty() and remain_first_count > 0:
            return None
        elif max_heap.empty():
            break

        neg_count2, second_ch = max_heap.get()
        remain_second_count = -neg_count2 - 1
        res.append(second_ch)
        
        if remain_first_count > 0:
            max_heap.put((-remain_first_count, first_ch))
        if remain_second_count > 0:
            max_heap.put((-remain_second_count, second_ch))
    
    return "".join(res)


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
    
    def test_example2(self):
        original_string = "yyz"
        rearranged_string = rearrange_string(original_string)
        self.assert_not_adjacent(rearranged_string, original_string)
    
    def test_example3(self):
        original_string = "yyy"
        self.assertIsNone(rearrange_string(original_string))

    def test_original_string_contains_duplicated_characters(self):
        original_string = "aaabb"
        rearranged_string = rearrange_string(original_string)
        self.assert_not_adjacent(rearranged_string, original_string)

    def test_original_string_contains_duplicated_characters2(self):
        original_string = "aaaaabbbc"
        rearranged_string = rearrange_string(original_string)
        self.assert_not_adjacent(rearranged_string, original_string)
    
    def test_original_string_contains_duplicated_characters3(self):
        original_string = "aaabc"
        rearranged_string = rearrange_string(original_string)
        self.assert_not_adjacent(rearranged_string, original_string)

    def test_already_rearranged(self):
        original_string = "abcdefg"
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

### Mar 28, 2020 LC 743 \[Medium\] Network Delay Time
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

**Solution with Dijkstra’s Algorithm:** [https://repl.it/@trsong/Calculate-Network-Delay-Time](https://repl.it/@trsong/Calculate-Network-Delay-Time)
```py
import unittest
import sys
from Queue import PriorityQueue

def max_network_delay(times, nodes):
    neighbors = [None] * (nodes+1)
    for u, v, w in times:
        if not neighbors[u]:
            neighbors[u] = []
        neighbors[u].append((v, w))

    distance = [sys.maxint] * (nodes+1)
    pq = PriorityQueue()
    pq.put((0, 0))
    while not pq.empty():
        dist, cur = pq.get()
        if distance[cur] != sys.maxint:
            # current node has been visited
            continue
        distance[cur] = dist
        if not neighbors[cur]:
            continue
        for v, w in neighbors[cur]:
            alt_dist = dist + w
            if distance[v] == sys.maxint:
                pq.put((alt_dist, v))

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

### Mar 27, 2020 \[Medium\] Shortest Unique Prefix
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

**My thoughts:** Most string prefix searching problem can be solved using Trie (Prefix Tree). Trie Solution can be found here: [https://trsong.github.io/python/java/2019/11/02/DailyQuestionsNov/#jan-25-2020-medium-shortest-unique-prefix](https://trsong.github.io/python/java/2019/11/02/DailyQuestionsNov/#jan-25-2020-medium-shortest-unique-prefix). 

However, trie can use a lot of memory. A more memory efficient data structure is Ternary Search Tree. 

For example, the TST for given example looks like the following, note by natural of TST only middle child count as prefix. Check previous day's question for TST: [https://trsong.github.io/python/java/2020/02/02/DailyQuestionsFeb/#mar-26-2020-easy-ternary-search-tree](https://trsong.github.io/python/java/2020/02/02/DailyQuestionsFeb/#mar-26-2020-easy-ternary-search-tree)
```py
         z 
 /       | 
d        e
|        |
o        b
| \      |
g  u     r
   | \   |
   c  v  a
   |  |
   k  e
```

While building each node, we also include the count of words that share the same prefix: 
```py
                 z (1)
 /               |  
d (3)            e (1)
|                |
o (2)            b (1)
|     \          |
g (1)  u (1)     r  (1)
       | \       |
   (1) c  v (1)  a (1)
       |  |
   (1) k  e (1) 

Under each node, if the count is 1 then we have a unique prefix: z, dog, du, dov
```


**Solution with Ternary Search Tree:** [https://repl.it/@trsong/Find-Shortest-Unique-Prefix](https://repl.it/@trsong/Find-Shortest-Unique-Prefix)
```py
import unittest

class TSTNode(object):
    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.middle = None
        self.right = None
        self.count = 0

    def insert(self, word):
        self.insert_recur(self, word, 0)

    def insert_recur(self, node, word, i):
        if i >= len(word):
            return None
        
        ch = word[i]
        if not node:
            node = TSTNode(ch)
        if node.val < ch:
            node.right = self.insert_recur(node.right, word, i)
        elif node.val > ch:
            node.left = self.insert_recur(node.left, word, i)
        else:
            node.count += 1
            node.middle = self.insert_recur(node.middle, word, i+1)
        return node

    def find_prefix(self, word):
        if not word:
            return ""
        unique_prefix_index = self.find_prefix_index_recur(self, word, 0)
        return word[:unique_prefix_index+1]

    def find_prefix_index_recur(self, node, word, i):
        ch = word[i]
        if node.val < ch:
            return self.find_prefix_index_recur(node.right, word, i)
        elif node.val > ch:
            return self.find_prefix_index_recur(node.left, word, i)
        elif node.count > 1:
            return self.find_prefix_index_recur(node.middle, word, i+1)
        else:
            return i


def shortest_unique_prefix(words):
    t = TSTNode()
    for word in words:
        t.insert(word)

    return map(t.find_prefix, words)


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
    unittest.main(exit=False)
```

### Mar 26, 2020 \[Easy\] Ternary Search Tree
---
> **Question:** A ternary search tree is a trie-like data structure where each node may have up to three children:
>
> - left child nodes link to words lexicographically earlier than the parent prefix
> - right child nodes link to words lexicographically later than the parent prefix
> - middle child nodes continue the current word
>
> Implement insertion and search functions for a ternary search tree. 

**Example:**
```py
Input: code, cob, be, ax, war, and we.
Output:
       c
    /  |  \
   b   o   w
 / |   |   |
a  e   d   a
|    / |   | \ 
x   b  e   r  e  

since code is the first word inserted in the tree, and cob lexicographically precedes cod, cob is represented as a left child extending from cod.
```

**Solution:** [https://repl.it/@trsong/Ternary-Search-Tree](https://repl.it/@trsong/Ternary-Search-Tree)
```py
import unittest

class TernarySearchTree(object):
    def __init__(self):
        self.root = None
        self.has_empty_word = False

    def search(self, word):
        if not word:
            return self.has_empty_word
        else:
            return self.search_recur(self.root, word, 0)

    def search_recur(self, node, word, i):
        if not node:
            return False
        ch = word[i]
        if ch < node.val:
            return self.search_recur(node.left, word, i)
        elif ch > node.val:
            return self.search_recur(node.right, word, i)
        elif i < len(word) - 1:
            return self.search_recur(node.middle, word, i+1)
        else:
            return node.is_end

    def insert(self, word):
        if not word:
            self.has_empty_word = True
        else:
            self.root = self.insert_recur(self.root, word, 0)
    
    def insert_recur(self, node, word, i):
        ch = word[i]
        if not node:
            node = TSTNode(ch)
        
        if ch < node.val:
            node.left = self.insert_recur(node.left, word, i)
        elif ch > node.val:
            node.right = self.insert_recur(node.right, word, i)
        elif i < len(word) - 1:
            node.middle = self.insert_recur(node.middle, word, i+1)
        else:
            node.is_end = True
        return node

    def __repr__(self):
        return str(self.root)
        

class TSTNode(object):
    def __init__(self, val=None, left=None, middle=None, right=None):
        self.val = val
        self.left = left
        self.middle = middle
        self.right = right
        self.is_end = False

    ###################
    # Testing Utilities
    ###################
    def __eq__(self, other):
        return (other and
            other.val == self.val and
            other.left == self.left and 
            other.right == self.right)

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
            if node.is_end:
                res.append(" [END]")
            for child in [node.left, node.middle, node.right]:
                stack.append((child, depth+1))
        return "\n" + "".join(res) + "\n"


class TSTNodeSpec(unittest.TestCase):
    def test_example(self):
        t = TernarySearchTree()
        words = ["code", "cob", "be", "ax", "war", "we"]
        for w in words:
            t.insert(w)
        for w in words:
            self.assertTrue(t.search(w), msg=str(t))

    def test_insert_empty_word(self):
        t = TernarySearchTree()
        t.insert("")
        self.assertTrue(t.search(""), msg=str(t))

    def test_search_unvisited_word(self):
        t = TernarySearchTree()
        self.assertFalse(t.search("a"), msg=str(t))
        t.insert("a")
        self.assertTrue(t.search("a"), msg=str(t))

    def test_insert_word_with_same_prefix(self):
        t = TernarySearchTree()
        t.insert("a")
        t.insert("aa")
        self.assertTrue(t.search("a"), msg=str(t))
        t.insert("aaa")
        self.assertFalse(t.search("aaaa"), msg=str(t))

    def test_insert_word_with_same_prefix2(self):
        t = TernarySearchTree()
        t.insert("bbb")
        t.insert("aaa")
        self.assertFalse(t.search("a"), msg=str(t))
        t.insert("aa")
        self.assertFalse(t.search("a"), msg=str(t))
        self.assertFalse(t.search("b"), msg=str(t))
        self.assertTrue("aa", msg=str(t))
        self.assertTrue("aaa", msg=str(t))

    def test_tst_should_follow_specification(self):
        """
                  c
                / | \
               a  u  h
               |  |  | \
               t  t  e  u
             /  / |   / |
            s  p  e  i  s
        """
        t = TernarySearchTree()
        words = ["cute", "cup","at","as","he", "us", "i"]
        for w in words:
            t.insert(w)


        left_tree = TSTNode('a', middle=TSTNode('t', TSTNode('s')))
        middle_tree = TSTNode('u', middle=TSTNode('t', TSTNode('p'), TSTNode('e')))

        right_right_tree = TSTNode('u', TSTNode('i'), TSTNode('s'))
        right_tree = TSTNode('h', middle=TSTNode('e'), right=right_right_tree)
        root = TSTNode('c', left_tree, middle_tree, right_tree)
        self.assertEqual(root, t.root)
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 25, 2020 LC 336 \[Hard\] Palindrome Pairs
---
> **Question:** Given a list of words, find all pairs of unique indices such that the concatenation of the two words is a palindrome.
>
> For example, given the list `["code", "edoc", "da", "d"]`, return `[(0, 1), (1, 0), (2, 3)]`.

**My thoughts:** any word in the list can be partition into `prefix` and `suffix`. If there exists another word such that its reverse equals either prefix or suffix, then we can combine them and craft a new palindrome: 
1. `reverse_suffix + prefix + suffix` where prefix is a palindrome or 
2. `prefix + suffix + reverse_prefix` where suffix is a palindrome

**Solution:** [https://repl.it/@trsong/Palindrome-Pairs](https://repl.it/@trsong/Palindrome-Pairs)
```py
import unittest


def find_all_palindrome_pairs(words):
    reverse_word_lookup = {}
    for i, word in enumerate(words):
        reverse_word = word[::-1]
        if reverse_word not in reverse_word_lookup:
            reverse_word_lookup[reverse_word] = []
        reverse_word_lookup[reverse_word].append(i)

    res = []
    for i, word in enumerate(words):
        if "" in reverse_word_lookup and is_palindrome(word):
            for j in reverse_word_lookup[""]:
                if i != j:
                    res.append((j, i))

        for pos in xrange(len(word)):
            prefix = word[:pos]
            suffix = word[pos:]
            if is_palindrome(prefix) and suffix in reverse_word_lookup:
                # reverse_suffix + prefix + suffix where prefix itself is palindrome
                for j in reverse_word_lookup[suffix]:
                    if i != j:
                        res.append((j, i))

            if is_palindrome(suffix) and prefix in reverse_word_lookup:
                # prefix + suffix + reverse_prefix where suffix itself is palindrome
                for j in reverse_word_lookup[prefix]:
                    if i != j:
                        res.append((i, j))
    return res


def is_palindrome(s):
    i, j = 0, len(s)-1
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

**Optimization tips:** Frequently checking reverse string and calculate substring can be expensive. We can use a rolling hash function to quickly convert string into a number and by comparing the forward and backward hash value we can easily tell if a string is a palidrome or not. 
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


**Optimization with Rolling Hash:** [https://repl.it/@trsong/Palindrome-Pairs-with-RollingHash](https://repl.it/@trsong/Palindrome-Pairs-with-RollingHash)
```py
def find_all_palindrome_pairs(words):
    backward_hash_words = map(RollingHash.backward_hash, words)
    forward_hash_words = map(RollingHash.forward_hash, words)

    forward_hash_map = {}
    for i, hash_val in enumerate(forward_hash_words):
        if hash_val not in forward_hash_map:
            forward_hash_map[hash_val] = []
        forward_hash_map[hash_val].append(i)

    res = []
    empty_word_hash = RollingHash.forward_hash('')
    for i, word in enumerate(words):
        prefix_fwd_hash = 0
        prefix_bwd_hash = 0
        suffix_fwd_hash = forward_hash_words[i]
        suffix_bwd_hash = backward_hash_words[i]
        if suffix_fwd_hash == suffix_bwd_hash and empty_word_hash in forward_hash_map:
            for j in forward_hash_map[empty_word_hash]:
                if j != i:
                    res.append((i, j))

        for pos, ch in enumerate(word):
            word_len = len(word)
            prefix_fwd_hash = RollingHash.rolling_forward(prefix_fwd_hash, ch)
            prefix_bwd_hash = RollingHash.rolling_backward(prefix_bwd_hash, pos, ch)
            suffix_fwd_hash = RollingHash.unapply_forward_front(suffix_fwd_hash, word_len-1-pos, ch)
            suffix_bwd_hash = RollingHash.unapply_backward_front(suffix_bwd_hash, ch)
            if prefix_fwd_hash == prefix_bwd_hash and suffix_bwd_hash in forward_hash_map:
                # if prefix is a palindrome and reversed_suffix exists, 
                # then reversed_suffix + prefix + suffix must be a palindrome
                for j in forward_hash_map[suffix_bwd_hash]:
                    if j != i:
                        res.append((j, i))
            if suffix_fwd_hash == suffix_bwd_hash and prefix_bwd_hash in forward_hash_map:
                # if suffix is a palindrome and reverse_prefix exists,
                # then prefix + suffix + reversed_prefix must be a palindrome
                for j in forward_hash_map[prefix_bwd_hash]:
                    if j != i:
                        res.append((i, j))
    
    # avoid hash collison
    return filter(lambda pair: is_palindrome(words, pair[0], pair[1]), res)


def is_palindrome(words, w1_idx, w2_idx):
    w1, w2 = words[w1_idx], words[w2_idx]
    n, m = len(w1), len(w2)
    for i in xrange((n+m)//2):
        ch1 = w1[i] if i < n else w2[i-n]
        j = n+m-1-i
        ch2 = w2[j-n] if j >= n else w1[j]
        if ch1 != ch2:
            return False
    return True 


class RollingHash(object):
    p0 = 17
    p1 = 666667
    
    @staticmethod
    def forward_hash(s):
        # eg. "123" => 123
        res = 0
        for ch in s:
            res = RollingHash.rolling_forward(res, ch)
        return res
    
    @staticmethod
    def rolling_forward(prev_hash, ch):
        # eg. 12 + "3" = 123
        return (prev_hash * RollingHash.p0 + ord(ch)) % RollingHash.p1

    @staticmethod
    def unapply_forward_front(prev_hash, pos, ch):
        # eg. 123 => 23
        return (prev_hash - ord(ch) * pow(RollingHash.p0, pos)) % RollingHash.p1

    @staticmethod
    def backward_hash(s):
        # eg. "123" => 321
        res = 0
        for i, ch in enumerate(s):
            res = RollingHash.rolling_backward(res, i, ch)
        return res

    @staticmethod
    def rolling_backward(prev_hash, pos, ch):
        # eg. 21 + "3" => 321
        return (prev_hash + ord(ch) * pow(RollingHash.p0, pos)) % RollingHash.p1

    @staticmethod
    def unapply_backward_front(prev_hash, ch):
        # eg. 321 => 32
        return ((prev_hash - ord(ch)) % RollingHash.p1 * Modulo.modinv(RollingHash.p0, RollingHash.p1)) % RollingHash.p1


class Modulo(object):
    """
    In python3.8, use pow(b, -1, mod=m) to calculate (1/b % m)
    """
    @staticmethod
    def egcd(a, b):
        if a == 0:
            return (b, 0, 1)
        else:
            g, y, x = Modulo.egcd(b % a, a)
            return (g, x - (b // a) * y, y)
    
    @staticmethod
    def modinv(a, m):
        g, x, y = Modulo.egcd(a, m)
        if g != 1:
            raise Exception('modular inverse does not exist')
        else:
            return x % m
```

### Mar 24, 2020 \[Medium\] Remove Adjacent Duplicate Characters
---
> **Question:** Given a string, we want to remove 2 adjacent characters that are the same, and repeat the process with the new string until we can no longer perform the operation.

**Example:**
```py
remove_adjacent_dup("cabba")
# Start with cabba
# After remove bb: caa
# After remove aa: c
# Returns c
```

**Solution with Stack:** [https://repl.it/@trsong/Remove-Adjacent-Duplicate-Characters](https://repl.it/@trsong/Remove-Adjacent-Duplicate-Characters)
```py
import unittest

def remove_adjacent_dup(s):
    stack = []
    is_top_dup = False
    for ch in s:
        if stack and stack[-1] != ch and is_top_dup:
            stack.pop()
            is_top_dup = False

        if stack and stack[-1] == ch:
            is_top_dup = True
        else:
            stack.append(ch)

    if is_top_dup:
        stack.pop()

    return ''.join(stack) 


class RemoveAdjacentDupSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual("c", remove_adjacent_dup("cabba"))

    def test_empty_string(self):
        self.assertEqual("", remove_adjacent_dup(""))

    def test_no_adj_duplicates(self):
        self.assertEqual("abc123abc", remove_adjacent_dup("abc123abc"))

    def test_no_adj_duplicates2(self):
        self.assertEqual("1010101010", remove_adjacent_dup("1010101010"))

    def test_reduce_to_empty_string(self):
        self.assertEqual("", remove_adjacent_dup("caaabbbaacdddd"))

    def test_should_reduce_to_simplist(self):
        self.assertEqual("acac", remove_adjacent_dup("acaaabbbacdddd"))

    def test_should_reduce_to_simplist2(self):
        self.assertEqual("", remove_adjacent_dup("43211234"))

    def test_should_reduce_to_simplist3(self):
        self.assertEqual("3", remove_adjacent_dup("333111221113"))

    def test_one_char_string(self):
        self.assertEqual("a", remove_adjacent_dup("a"))

    def test_contains_uniq_char(self):
        self.assertEqual("", remove_adjacent_dup("aaaa"))

    def test_contains_uniq_char2(self):
        self.assertEqual("", remove_adjacent_dup("11111"))

    def test_duplicates_occur_in_the_middle(self):
        self.assertEqual("mipie", remove_adjacent_dup("mississipie"))

    def test_shrink_to_one_char(self):
        self.assertEqual("g", remove_adjacent_dup("gghhgghhg"))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 23, 2020 \[Medium\] Satisfactory Playlist
---
> **Question:** You have access to ranked lists of songs for various users. Each song is represented as an integer, and more preferred songs appear earlier in each list. For example, the list `[4, 1, 7]` indicates that a user likes song `4` the best, followed by songs `1` and `7`.
>
> Given a set of these ranked lists, interleave them to create a playlist that satisfies everyone's priorities.
>
> For example, suppose your input is `[[1, 7, 3], [2, 1, 6, 7, 9], [3, 9, 5]]`. In this case a satisfactory playlist could be `[2, 1, 6, 7, 3, 9, 5]`.

**My thoughts:** create a graph with vertex being song and edge `(u, v)` representing that u is more preferred than v. A topological order will make sure that all more preferred song will go before less preferred ones. Thus gives a list that satisfies everyone's priorities, if there is one (no cycle).

**Solution with Topological Sort:** [https://repl.it/@trsong/Satisfactory-Playlist](https://repl.it/@trsong/Satisfactory-Playlist)
```py
import unittest

def calculate_satisfactory_playlist(preference):
    class NodeState:
        UNVISITED = 0
        VISITED = 1
        VISITING = 2

    neighbors = {}
    node_set = set()
    for lst in preference:
        for i, node1 in enumerate(lst):
            node2 = lst[i+1] if i+1 < len(lst) else None
            node_set.add(node1)
            if node1 not in neighbors:
                neighbors[node1] = []
            if node2 is not None:
                neighbors[node1].append(node2)
                node_set.add(node2)

    reverse_top_order = []
    node_states = {node: NodeState.UNVISITED for node in node_set}
    for node in node_set:
        if node_states[node] == NodeState.VISITED:
            continue
        stack = [node]
        while stack:
            cur = stack[-1]
            if node_states[cur] == NodeState.VISITED:
                stack.pop()
            elif node_states[cur] == NodeState.VISITING:
                node_states[cur] = NodeState.VISITED
                reverse_top_order.append(cur)
            else:
                # node_state is UNVISITED
                node_states[cur] = NodeState.VISITING
                for nbr in neighbors[cur]:
                    if node_states[nbr] == NodeState.VISITING:
                        # cycle detected
                        return None
                    elif node_states[nbr] == NodeState.UNVISITED:
                        stack.append(nbr)
    
    reverse_top_order.reverse()
    return reverse_top_order
        

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
    unittest.main(exit=False)
```

### Mar 22, 2020 \[Medium\] Add Subtract Currying
---
> **Question:** Write a function, add_subtract, which alternately adds and subtracts curried arguments. 

**Example:**
```py
add_subtract(7) -> 7

add_subtract(1)(2)(3) -> 1 + 2 - 3 -> 0

add_subtract(-5)(10)(3)(9) -> -5 + 10 - 3 + 9 -> 11
```

**Solution:** [https://repl.it/@trsong/Add-Subtract-Currying](https://repl.it/@trsong/Add-Subtract-Currying)
```py
import unittest

def add_subtract(first=None):
    if first is None:
        return 0

    class Context:
        sign = 1
        res = first

    def f(next=None):
        if next is None:
            return Context.res
        else:
            Context.res += Context.sign * next
            Context.sign *= -1
            return f
    return f


class AddSubtractSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(7, add_subtract(7)())

    def test_example2(self):
        self.assertEqual(0, add_subtract(1)(2)(3)())

    def test_example3(self):
        self.assertEqual(11, add_subtract(-5)(10)(3)(9)())

    def test_empty_argument(self):
        self.assertEqual(0, add_subtract())

    def test_positive_arguments(self):
        self.assertEqual(4, add_subtract(1)(2)(3)(4)())

    def test_negative_arguments(self):
        self.assertEqual(9, add_subtract(-1)(-3)(-5)(-1)(-9)())


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 21, 2020 \[Medium\] Ways to Form Heap with Distinct Integers
---
> **Question:** Write a program to determine how many distinct ways there are to create a max heap from a list of `N` given integers.

**Example:**
```py
If N = 3, and our integers are [1, 2, 3], there are two ways, shown below.

  3      3
 / \    / \
1   2  2   1
```

**My thoughts:** First make observation that the root of heap is always maximum element. Then we can reduce the problem by choosing max then decide which elem goes to left so that other elem can go to right. 

However, as heap has a height constraint, we cannot allow random number of elem go to left, that is, we have to keep the tree height balanced. Suppose we have l elem go to left and r elem to right, then we will have `l + r + 1 = n` (don't forget to include root). 

Finally, let `dp[n]` represents solution for size n, then we will have `dp[n] = (n-1 choose l) * dp[l] * dp[r]` where `l + r + 1 = n`

As for how to calculuate l, as last level of heap tree might not be full, we have two situations:
1. last level is less than half full, then all last level goes to left tree
2. last level is more than half full, then only half last level goes to left tree

**Solution with DP:** [https://repl.it/@trsong/Ways-to-Form-Heap-with-Distinct-Integers](https://repl.it/@trsong/Ways-to-Form-Heap-with-Distinct-Integers)
```py
import unittest
import math

def form_heap_ways(n):
    if n <= 1:
        return 1

    dp = [None] * (n+1)
    choose_dp = [[None for _ in xrange(n+1)] for _ in xrange(n+1)]

    dp[0] = dp[1] = 1
    for i in xrange(2, n+1):
        h = int(math.log(i, 2))
        leaf_level_max = 2 ** h     # max number of possible elem at leaf level
        full_tree = 2 ** (h+1) - 1  # max number of possible elem of full tree
        unfilled_leaf_level = full_tree - i
        filled_leaf_level = leaf_level_max - unfilled_leaf_level

        internal_nodes = i - filled_leaf_level
        left_tree_size = (internal_nodes - 1) // 2   # exclude root
        if filled_leaf_level < leaf_level_max // 2:
            left_tree_size += filled_leaf_level
        else:
            left_tree_size += leaf_level_max // 2

        right_tree_size = i - 1 - left_tree_size
        dp[i] = choose(i-1, left_tree_size, choose_dp) * dp[left_tree_size] * dp[right_tree_size]

    return dp[n]


def choose(n, k, dp=None):
    if 2 * k > n:
        return choose(n, n-k, dp)
    elif k > n:
        return 0
    elif k == 0 or n <= 1:
        return 1
    
    if not dp[n][k]:
        # way to choose last elem + # ways not to choose last elem
        dp[n][k] = choose(n-1, k-1, dp) + choose(n-1, k, dp)  

    return dp[n][k]
    

class FormHeapWaySpec(unittest.TestCase):
    def test_example(self):
        """
          3      3
         / \    / \
        1   2  2   1
        """
        self.assertEqual(2, form_heap_ways(3))

    def test_empty_heap(self):
        self.assertEqual(1, form_heap_ways(0))

    def test_size_one_heap(self):
        self.assertEqual(1, form_heap_ways(1))

    def test_size_four_heap(self):
        """
            4      4      4
           / \    / \    / \
          3   1  3   2  2   3
         /      /      /
        2      1      1
        """
        self.assertEqual(3, form_heap_ways(4))

    def test_size_five_heap(self):
        self.assertEqual(8, form_heap_ways(5))

    def test_size_ten_heap(self):
        self.assertEqual(3360, form_heap_ways(10))

    def test_size_twenty_heap(self):
        self.assertEqual(319258368000, form_heap_ways(20))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 20, 2020 \[Medium\] Next Higher Number
---
> **Question:** Given an integer n, find the next biggest integer with the same number of 1-bits on. For example, given the number `6 (0110 in binary)`, return `9 (1001)`.

**My thoughts:** The idea is to find the leftmost of rightmost ones, swap it with left zero and push remaining rightmost ones all the way till the end. 

**Example:**
```py
   10011100
      ^      swap with left zero
=> 10101100 
       ^^    push till the end
=> 10100011 
```

**Solution:** [https://repl.it/@trsong/Next-Higher-Number](https://repl.it/@trsong/Next-Higher-Number)
```py
import unittest

def next_higher_number(num):
    if num <= 0:
        return None
    
    # Step1: Find the rightmost 1
    # eg. 10011100
    #          ^
    rightmost_one = num - (num & (num - 1))
    num_rightmost_ones = 0

    # Step2: Find the leftmost of rightmost 1's and count number
    # eg. 10011100
    #        ^
    #  => 10000000, count = 3
    #        ^
    while rightmost_one & num:
        num &= ~rightmost_one  
        num_rightmost_ones += 1
        rightmost_one <<= 1
    
    # Step3: Swap the leftmost of rightmost 1's with its left zero
    # eg. 1001000
    #       ^^ 
    #  => 1010000   
    num |= rightmost_one

    # Step4: Futher shift remaining right ones all the way till the end
    # eg. 10011100
    #        ^
    #  => 10010011
    if num_rightmost_ones > 0:
        num |= ((1 << num_rightmost_ones - 1) - 1)
    return num


class NextHigherNumberSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(bin(expected), bin(result))

    def test_example(self):
        self.assert_result(0b1001, next_higher_number(0b0110))

    def test_example2(self):
        self.assert_result(0b110, next_higher_number(0b101))

    def test_example3(self):
        self.assert_result(0b1101, next_higher_number(0b1011))

    def test_zero(self):
        self.assertIsNone(next_higher_number(0))

    def test_end_in_one(self):
        self.assert_result(0b10, next_higher_number(0b01))

    def test_end_in_one2(self):
        self.assert_result(0b1011, next_higher_number(0b111))

    def test_end_in_one3(self):
        self.assert_result(0b110001101101, next_higher_number(0b110001101011))

    def test_end_in_zero(self):
        self.assert_result(0b100, next_higher_number(0b010))

    def test_end_in_zero2(self):
        self.assert_result(0b1000011, next_higher_number(0b0111000))

    def test_end_in_zero3(self):
        self.assert_result(0b1101110001, next_higher_number(0b1101101100))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 19, 2020 \[Medium\] The Celebrity Problem
---
> **Question:** At a party, there is a single person who everyone knows, but who does not know anyone in return (the "celebrity"). To help figure out who this is, you have access to an `O(1)` method called `knows(a, b)`, which returns `True` if person a knows person b, else `False`.
>
> Given a list of N people and the above operation, find a way to identify the celebrity in O(N) time.


**Solution with Stack:** [https://repl.it/@trsong/The-Celebrity-Problem](https://repl.it/@trsong/The-Celebrity-Problem)
```py
import unittest

def find_celebrity(knows, people):
    if not people:
        return None
    
    stack = people
    while len(stack) > 1:
        p1 = stack.pop()
        p2 = stack.pop()
        
        if knows(p1, p2):
            stack.append(p2)
        else:
            stack.append(p1)
            
    return stack.pop()
    
   
class FindCelebritySpec(unittest.TestCase):
    def knows_factory(self, neighbors):
        return lambda me, you: you in neighbors[me]
        
    def test_no_one_exist(self):
        knows = lambda x, y: False
        self.assertIsNone(find_celebrity(knows, []))
        
    def test_only_one_person(self):
        neighbors = {
            0: []
        }
        knows = self.knows_factory(neighbors)
        self.assertEqual(0, find_celebrity(knows, neighbors.keys()))
        
    def test_no_one_knows_others_except_celebrity(self):
        neighbors = {
            0: [4],
            1: [4],
            2: [4],
            3: [4],
            4: []
        }
        knows = self.knows_factory(neighbors)
        self.assertEqual(4, find_celebrity(knows, neighbors.keys()))
        
    def test_no_one_knows_others_except_celebrity2(self):
        neighbors = {
            0: [],
            1: [0],
            2: [0],
            3: [0],
            4: [0]
        }
        knows = self.knows_factory(neighbors)
        self.assertEqual(0, find_celebrity(knows, neighbors.keys()))
        
    def test_every_one_not_know_someone2(self):
        neighbors = {
            0: [1, 2, 3, 4],
            1: [0, 2, 3, 4],
            2: [4],
            3: [2, 4],
            4: []
        }
        knows = self.knows_factory(neighbors)
        self.assertEqual(4, find_celebrity(knows, neighbors.keys()))
        
    def test_every_one_not_know_someone3(self):
        neighbors = {
            0: [1, 4, 5],
            1: [0, 4, 5],
            
            2: [3, 4, 5],
            3: [2, 4, 5],
            
            4: [5, 0],
            5: [],
        }
        knows = self.knows_factory(neighbors)
        self.assertEqual(5, find_celebrity(knows, neighbors.keys()))
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 18, 2020 LC 938 \[Easy\] Range Sum of BST
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

**Solution:** [https://repl.it/@trsong/Range-Sum-of-BST](https://repl.it/@trsong/Range-Sum-of-BST)
```py
import unittest


def bst_range_sum(root, low, hi):
    if not root:
        return 0
    if hi < root.val:
        return bst_range_sum(root.left, low, hi)
    elif root.val < low:
        return bst_range_sum(root.right, low, hi)
    else:
        left_sum = bst_range_sum(root.left, low, root.val)
        right_sum = bst_range_sum(root.right, root.val, hi)
        return left_sum + root.val + right_sum


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
    unittest.main(exit=False)
```

### Mar 17, 2020 \[Hard\] Largest Rectangle
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

**My thoughts:** Think in a DP way: scan through row by row to accumulate each cell above. We will have a historgram for each row. Then all we need to do is to find the ara for the largest rectangle in histogram on each row. 

For example,
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

**Solution with DP:** [https://repl.it/@trsong/Find-Largest-Rectangle](https://repl.it/@trsong/Find-Largest-Rectangle)
```py
import unittest

def largest_rectangle(grid):
    if not grid or not grid[0]:
        return 0

    n, m = len(grid), len(grid[0])
    max_area = largest_rectangle_in_histogram(grid[0])

    for r in xrange(1, n):
        for c in xrange(m):
            if grid[r][c] == 1:
                grid[r][c] = grid[r-1][c] + 1

        max_area_on_row = largest_rectangle_in_histogram(grid[r])
        max_area = max(max_area, max_area_on_row)
    
    return max_area


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


class LargestRectangleSpec(unittest.TestCase):
    def test_empty_table(self):
        self.assertEqual(0, largest_rectangle([]))
        self.assertEqual(0, largest_rectangle([[]]))

    def test_example(self):
        self.assertEqual(4, largest_rectangle([
            [1, 0, 0, 0],
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [0, 1, 0, 0]
        ]))

    def test_table2(self):
        self.assertEqual(4, largest_rectangle([
            [0, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 1, 1, 0]
        ]))

    def test_table3(self):
        self.assertEqual(3, largest_rectangle([
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0],
        ]))

    def test_table4(self):
        self.assertEqual(4, largest_rectangle([
            [0, 0, 1, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 0, 1],
        ]))

    def test_table5(self):
        self.assertEqual(8, largest_rectangle([
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 0]
        ]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 16, 2020 \[Easy\] Triplet Sum to K
---
> **Question:** Given an array of numbers and a number `k`, determine if there are three entries in the array which add up to the specified number `k`. For example, given `[20, 303, 3, 4, 25]` and `k = 49`, return `true` as `20 + 4 + 25 = 49`.


**Solution:** [https://repl.it/@trsong/Triplet-Sum-to-K-3SUM](https://repl.it/@trsong/Triplet-Sum-to-K-3SUM)
```py
import unittest

def two_sum(sorted_nums, i, j, target):
    while i < j:
        sum = sorted_nums[i] + sorted_nums[j]
        if sum == target:
            return True
        elif sum < target:
            i += 1
        else:
            j -= 1
    return False


def three_sum(nums, target):
    nums.sort()
    n = len(nums)

    for i in xrange(n-2):
        if i > 0 and nums[i-1] == nums[i]:
            # skip duplicates
            continue

        sub_target = target - nums[i]
        if two_sum(nums, i+1, n-1, sub_target):
            return True
    
    return False


class ThreeSumSpec(unittest.TestCase):
    def test_example(self):
        target, nums = 49, [20, 303, 3, 4, 25]
        self.assertTrue(three_sum(nums, target))  # 20 + 4 + 25 = 49

    def test_empty_list(self):
        target, nums = 0, []
        self.assertFalse(three_sum(nums, target))

    def test_unsort_list(self):
        target, nums = 24, [12, 3, 4, 1, 6, 9]
        self.assertTrue(three_sum(nums, target))  # 12 + 3 + 9 = 24

    def test_list_with_duplicates(self):
        target, nums = 4, [1, 1, 2, 3, 3]  
        self.assertTrue(three_sum(nums, target))  # 1 + 1 + 2 = 4

    def test_list_with_duplicates2(self):
        target, nums = 3, [1, 1, 2, 2, 3, 3]  
        self.assertFalse(three_sum(nums, target))

    def test_list_with_duplicates3(self):
        target, nums = 0, [0, 0, 0, 0]  
        self.assertTrue(three_sum(nums, target))

    def test_list_with_negative_elements(self):
        target, nums = 4, [4, 8, -1, 2, -2, 10]  
        self.assertTrue(three_sum(nums, target))  # 2 - 2 + 4 = 4
  

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 15, 2020 \[Medium\] Largest Rectangular Area in a Histogram
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

**Solution with Stack:** [https://repl.it/@trsong/Largest-Rectangular-Area-in-a-Histogram](https://repl.it/@trsong/Largest-Rectangular-Area-in-a-Histogram)
```py
import unittest

def largest_rectangle_in_histogram(histogram):
    stack = []
    index = 0
    n = len(histogram)
    max_area = 0

    while index < n or stack:
        if not stack or index < n and histogram[stack[-1]] <= histogram[index]:
            # mantain stack in non-descending order
            stack.append(index)
            index += 1
        else:
            # if stack starts decreasing,
            # then left boundary must be stack[-2] and right boundary must be i. Note both boundaries are exclusive
            # and height is stack[-1]
            height = histogram[stack.pop()]         # height is current local max
            right_bound = index                     # right bound < height
            left_bound = stack[-1] if stack else -1 # left bound < height
            area = height * (right_bound - left_bound - 1)  # this is the max area we can achieve given the current height
            max_area = max(max_area, area)
    
    return max_area


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
    unittest.main(exit=False)
```

### Mar 14, 2020 \[Hard\] Graph Coloring
---
> **Question:** Given an undirected graph represented as an adjacency matrix and an integer k, determine whether each node in the graph can be colored such that no two adjacent nodes share the same color using at most k colors.

**My thoughts:** Solve this problem with backtracking. For each node, testing all colors one-by-one; if it turns out there is something wrong with current color, we will backtrack to test other colors.

**Solution with Backtracking:** [https://repl.it/@trsong/Graph-Coloring](https://repl.it/@trsong/Graph-Coloring)
```py
import unittest

def solve_graph_coloring(neighbor_matrix, k):
    n = len(neighbor_matrix)
    node_colors = [None] * n

    def backtrack(current_node):
        if current_node >= n:
            return True
        
        processed_neighbors = [i for i in xrange(current_node) if neighbor_matrix[current_node][i]]

        for color in xrange(k):
            if any(color == node_colors[i] for i in processed_neighbors):
                continue
            node_colors[current_node] = color
            if backtrack(current_node+1):
                return True
            node_colors[current_node] = None    
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

### Mar 13, 2020 LC 438 \[Medium\] Anagram Indices Problem
---
> **Question:**  Given a word W and a string S, find all starting indices in S which are anagrams of W.
>
> For example, given that W is `"ab"`, and S is `"abxaba"`, return `0`, `3`, and `4`.


**Solution with Sliding Window:** [https://repl.it/@trsong/Anagram-Indices-Problem](https://repl.it/@trsong/Anagram-Indices-Problem)
```py
import unittest

def find_anagrams(word, s):
    if not s or len(word) < len(s):
        return []

    freq_balance = {}
    for c in s:
        freq_balance[c] = freq_balance.get(c, 0) + 1

    res = []
    for j, end in enumerate(word):
        i = j - len(s)
        if i >= 0:
            start = word[i]
            freq_balance[start] = freq_balance.get(start, 0) + 1
            if freq_balance[start] == 0:
                del freq_balance[start]
        
        freq_balance[end] = freq_balance.get(end, 0) - 1
        if freq_balance[end] == 0:
            del freq_balance[end]

        if not freq_balance:
            res.append(i+1)

    return res


class FindAnagramSpec(unittest.TestCase):
    def test_example(self):
        word = 'abxaba'
        s = 'ab'
        self.assertEqual([0, 3, 4], find_anagrams(word, s))

    def test_example2(self):
        word = 'acdbacdacb'
        s = 'abc'
        self.assertEqual([3, 7], find_anagrams(word, s))

    def test_empty_source(self):
        self.assertEqual([], find_anagrams('', 'a'))
    
    def test_empty_pattern(self):
        self.assertEqual([], find_anagrams('a', ''))

    def test_pattern_contains_unseen_characters_in_source(self):
        word = "abcdef"
        s = "123"
        self.assertEqual([], find_anagrams(word, s))
    
    def test_pattern_not_in_source(self):
        word = 'ab9cd9abc9d'
        s = 'abcd'
        self.assertEqual([], find_anagrams(word, s))
    
    def test_matching_strings_have_overlapping_positions_in_source(self):
        word = 'abab'
        s = 'ab'
        self.assertEqual([0, 1, 2], find_anagrams(word, s))
    
    def test_find_all_matching_positions(self):
        word = 'cbaebabacd'
        s = 'abc'
        self.assertEqual([0, 6], find_anagrams(word, s))
    
    def test_find_all_matching_positions2(self):
        word = 'BACDGABCDA'
        s = 'ABCD'
        self.assertEqual([0, 5, 6], find_anagrams(word, s))
    
    def test_find_all_matching_positions3(self):
        word = 'AAABABAA'
        s = 'AABA'
        self.assertEqual([0, 1, 4], find_anagrams(word, s))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 12, 2020 \[Easy\] Add Two Numbers as a Linked List
---
> **Question:** You are given two linked-lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

**Example:**
```py
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```

**Solution:** [https://repl.it/@trsong/Add-Two-Numbers-as-a-Linked-List](https://repl.it/@trsong/Add-Two-Numbers-as-a-Linked-List)
```py
import unittest

def lists_addition(l1, l2):
    if not l1:
        return l2
    elif not l2:
        return l1
    p1 = l1
    p2 = l2
    p = dummy = ListNode(-1)
    carry = 0
    while p1 or p2:
        v1 = p1.val if p1 else 0
        v2 = p2.val if p2 else 0
        v = v1 + v2 + carry
        carry = v // 10
        v %= 10

        p.next = ListNode(v)
        p = p.next
        p1 = p1.next if p1 else None
        p2 = p2.next if p2 else None

    if carry:
        p.next = ListNode(carry)

    return dummy.next


###################
# Testing Utilities
###################
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return "{} -> {}".format(str(self.val), str(self.next))

    def __eq__(self, other):
        return other and self.val == other.val and self.next == other.next

    @staticmethod
    def build_list(*nums):
        node = dummy = ListNode(-1)
        for num in nums:
            node.next = ListNode(num)
            node = node.next
        return dummy.next


class ListsAdditionSpec(unittest.TestCase):
    def test_example(self):
        l1 = ListNode.build_list(2, 4, 3)
        l2 = ListNode.build_list(5, 6, 4)
        expected = ListNode.build_list(7, 0, 8)
        self.assertEqual(expected, lists_addition(l1, l2))

    def test_add_empty_list(self):
        self.assertEqual(None, lists_addition(None, None))

    def test_add_nonempty_to_empty_list(self):
        l1 = None
        l2 = ListNode.build_list(1, 2, 3)
        expected = ListNode.build_list(1, 2, 3)
        self.assertEqual(expected, lists_addition(l1, l2))

    def test_add_empty_to_nonempty_list(self):
        l1 = ListNode.build_list(1)
        l2 = None
        expected = ListNode.build_list(1)
        self.assertEqual(expected, lists_addition(l1, l2))

    def test_addition_with_carryover(self):
        l1 = ListNode.build_list(1, 1)
        l2 = ListNode.build_list(9, 9, 9, 9)
        expected = ListNode.build_list(0, 1, 0, 0, 1)
        self.assertEqual(expected, lists_addition(l1, l2))

    def test_addition_with_carryover2(self):
        l1 = ListNode.build_list(7, 5, 9, 4, 6)
        l2 = ListNode.build_list(8, 4)
        expected = ListNode.build_list(5, 0, 0, 5, 6)
        self.assertEqual(expected, lists_addition(l1, l2))

    def test_add_zero_to_number(self):
        l1 = ListNode.build_list(4, 2)
        l2 = ListNode.build_list(0)
        expected = ListNode.build_list(4, 2)
        self.assertEqual(expected, lists_addition(l1, l2))
    
    def test_same_length_lists(self):
        l1 = ListNode.build_list(1, 2, 3)
        l2 = ListNode.build_list(9, 8, 7)
        expected = ListNode.build_list(0, 1, 1, 1)
        self.assertEqual(expected, lists_addition(l1, l2))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 11, 2020 \[Medium\] LRU Cache
---
> **Question:** LRU cache is a cache data structure that has limited space, and once there are more items in the cache than available space, it will preempt the least recently used item. What counts as recently used is any item a key has `'get'` or `'put'` called on it.
>
> Implement an LRU cache class with the 2 functions `'put'` and `'get'`. `'put'` should place a value mapped to a certain key, and preempt items if needed. `'get'` should return the value for a given key if it exists in the cache, and return None if it doesn't exist.

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

**Solution:** [https://repl.it/@trsong/Implement-LRU-Cache](https://repl.it/@trsong/Implement-LRU-Cache)
```py
import unittest

class LRUCache(object):
    class ListNode(object):
        """
        Linkedlist to maintain LRU, order is from most inactive to most active
        """
        def __init__(self, key, val, next=None):
            self.update(key, val, next)
        
        def update(self, new_key, new_val, new_next):
            self.key = new_key
            self.val = new_val
            self.next = new_next

        def __repr__(self):
            return "({}, {}) -> {}".format(str(self.key), str(self.val), str(self.next))

    def __init__(self, capacity):
        self._capacity = capacity  # assume capacity is always positive
        self._node_lookup = {}
        self._dummy_tail = LRUCache.ListNode(None, None)
        self._dummy_head = LRUCache.ListNode(None, None, self._dummy_tail)

    def get(self, key):
        if key not in self._node_lookup:
            return None
        
        # Duplicate the node and move to tail
        node = self._node_lookup[key]
        key, val = node.key, node.val
        self._insert_node(key, val)
        
        # Remove original node
        if node.next.key:
            self._node_lookup[node.next.key] = node
  
        node.update(node.next.key, node.next.val, node.next.next)

        return val

    def put(self, key, val):
        if key in self._node_lookup:
            node = self._node_lookup[key]
            node.val = val
            self.get(key)  # populate the entry
        else:
            if len(self._node_lookup) >= self._capacity:
                most_inactive_node = self._dummy_head.next
                del self._node_lookup[most_inactive_node.key]
                self._dummy_head.next = most_inactive_node.next
            self._insert_node(key, val)
            
    def _insert_node(self, key, val):
        old_dummy_tail = self._dummy_tail
        self._dummy_tail = LRUCache.ListNode(None, None)
        
        old_dummy_tail.update(key, val, self._dummy_tail)
        self._node_lookup[key] = old_dummy_tail


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

    def test_cache_capacity_is_one(self):
        cache = LRUCache(1)
        cache.put(-1, 42)
        self.assertEqual(42, cache.get(-1))

        cache.put(-1, 10)
        self.assertEqual(10, cache.get(-1))

        cache.put(2, 0)  # evicts key -1
        self.assertIsNone(cache.get(-1))
        self.assertEqual(0, cache.get(2))

    def test_evict_most_inactive_element_when_cache_is_full(self):
        cache = LRUCache(3)
        cache.put(1, 1)
        cache.put(1, 1)
        cache.put(2, 1)
        cache.put(3, 1)
        cache.put(4, 1)  # evicts key 1
        self.assertIsNone(cache.get(1))

        cache.put(2, 1)
        cache.put(5, 1)  # evicts key 3
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
    unittest.main(exit=False)
```

### Mar 10, 2020 \[Medium\] Smallest Number of Perfect Squares
---
> **Question:** Write a program that determines the smallest number of perfect squares that sum up to N.
>
> Here are a few examples:
```py
Given N = 4, return 1 (4)
Given N = 17, return 2 (16 + 1)
Given N = 18, return 2 (9 + 9)
```

**Solution with DP:** [https://repl.it/@trsong/Smallest-Number-of-Perfect-Squares](https://repl.it/@trsong/Smallest-Number-of-Perfect-Squares)
```py
import unittest
import math

def min_perfect_squares(target):
    if target < 0:
        return -1
    elif is_perfect_square(target):
        return 1
    
    # Let dp[num] represents min perfect square to sum to target
    #     dp[num] = 1 + min(dp[num - ps]) for all perfect square ps < num
    # or  dp[num] = 1 if num is a perfect square 
    dp = [float('inf')] * (target + 1)
    dp[0] = 0

    for num in xrange(1, target+1):
        if is_perfect_square(num):
            dp[num] = 1
        else:
            for sqrt_num in xrange(1, int(math.sqrt(num))+1):
                ps = sqrt_num * sqrt_num
                dp[num] = min(dp[num], 1 + dp[num - ps])

    return dp[target]


def is_perfect_square(num):
    round_sqrt = int(math.sqrt(num))
    return round_sqrt * round_sqrt == num


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

### Mar 9, 2020 LC 308 \[Hard\] Range Sum Query 2D - Mutable
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

**Solution with 2D Binary Indexed Tree:** [https://repl.it/@trsong/Range-Sum-Query-2D-Table-Mutable](https://repl.it/@trsong/Range-Sum-Query-2D-Table-Mutable)
```py
import unittest

class RangeSumQuery2D(object):
    def __init__(self, matrix):
        n, m = len(matrix), len(matrix[0])
        self.bit_matrix = [[0 for _ in xrange(m+1)] for _ in xrange(n+1)]
        for r in xrange(n):
            for c in xrange(m):
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
    unittest.main(exit=False)
```

### Mar 8, 2020 \[Easy\] Number of 1 bits
---
> **Question:** Given an integer, find the number of 1 bits it has.

**Example:**
```py
one_bits(23)  # Returns 4 as 23 equals 0b10111
```

**My thoughts:** ***Brian Kernighan’s Algorithm*** is an efficient way to count number of set bits. In binary representation, a number num & with num - 1 always remove rightmost set bit. 

```py
1st iteration: 0b10111 -> 0b10110
  0b10111
& 0b10110  
= 0b10110

2nd iteration: 0b10110 -> 0b10100
  0b10110
& 0b10101
= 0b10100

3rd iteration: 0b10100 -> 0b10000
  0b10100
& 0b10011
= 0b10000

4th iteration: 0b10000 -> 0b00000
  0b10000
& 0b00000
= 0b00000
```

**Solution with Brian Kernighan’s Algorithm:** [https://repl.it/@trsong/Number-of-1-bits](https://repl.it/@trsong/Number-of-1-bits)
```py
import unittest

def count_bits(num):
    count = 0
    while num:
        count += 1
        num &= num - 1  # remove last set bit
    return count


class CountBitSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(4, count_bits(0b10111))

    def test_zero(self):
        self.assertEqual(0, count_bits(0b0))

    def test_all_bits_set(self):
        self.assertEqual(5, count_bits(0b11111))

    def test_power_of_two(self):
        self.assertEqual(1, count_bits(0b1000000000000))

    def test_every_other_bit_set(self):
        self.assertEqual(3, count_bits(0b0101010))

    def test_random_one_and_zeros(self):
        self.assertEqual(7, count_bits(0b1010010001000010000010000001))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 7, 2020 LC 114 \[Medium\] Flatten Binary Tree to Linked List
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

**Solution:** [https://repl.it/@trsong/Flatten-the-Binary-Tree-to-Linked-List-In-Place](https://repl.it/@trsong/Flatten-the-Binary-Tree-to-Linked-List-In-Place)
```py
import unittest

def flatten(tree):
    class Context:
        # inorder: cur, left, right
        prev_node = TreeNode(-1, right=tree)

    def inorder_flatten(root):
        if not root:
            return
        
        left = root.left
        right = root.right

        Context.prev_node.right = root
        Context.prev_node.left = None
        Context.prev_node = root

        inorder_flatten(left)
        inorder_flatten(right)

    inorder_flatten(tree)
    return tree


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __eq__(self, other):
        return other and other.val == self.val and other.left == self.left and other.right == self.right

    def __repr__(self):
        return "TreeNode({}, {}, {})".format(self.val, str(self.left), str(self.right))


class FlattenSpec(unittest.TestCase):
    @staticmethod
    def list_to_tree(lst):
        p = dummy = TreeNode(-1)
        for num in lst:
            p.right = TreeNode(num)
            p = p.right
        return dummy.right

    def assert_result(self, lst, tree):
        self.assertEqual(FlattenSpec.list_to_tree(lst), tree)

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
        self.assert_result(flatten_list, flatten(tree))

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
        self.assert_result(flatten_list, flatten(tree))

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
        self.assert_result(flatten_list, flatten(tree))  

    def test_right_heavy_tree(self):
        """
        1
       / \
      3   4
         /
        2
         \
          5
        """
        n4 = TreeNode(4, TreeNode(2, right=TreeNode(5)))
        tree = TreeNode(1, TreeNode(3), n4)
        flatten_list = [1, 3, 4, 2, 5]
        self.assert_result(flatten_list, flatten(tree)) 


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 6, 2020 LC 133 [Medium] Deep Copy Graph
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

**Solution with DFS:** [https://repl.it/@trsong/Deep-Copy-Graph](https://repl.it/@trsong/Deep-Copy-Graph)
```py
import unittest

def deep_copy(root):
    if not root:
        return None

    node_lookup = {}
    source_nodes = get_dfs_traversal(root)
    
    for source in source_nodes:
        node_lookup[source] = Node(source.val)


    for source, cloned_source in node_lookup.items():
        if not source.neighbors:
            continue

        cloned_source.neighbors = []
        for neighbor in source.neighbors:
            cloned_neighbor = node_lookup[neighbor]
            cloned_source.neighbors.append(cloned_neighbor)

    return node_lookup[source_nodes[0]]


def get_dfs_traversal(root):
    stack = [root]
    visited = set()
    res = []
    while stack:
        cur = stack.pop()
        if cur not in visited:
            res.append(cur)
        visited.add(cur)
            
        if not cur.neighbors:
            continue

        for n in cur.neighbors:
            if n not in visited:
                stack.append(n)
    
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

        nodes = get_dfs_traversal(self)
        other_nodes = get_dfs_traversal(other)
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
        for node in get_dfs_traversal(self):
            res.append(str(node.val))
        return "DFS: " + ",".join(res)     


class DeepCopySpec(unittest.TestCase):
    def assert_result(self, root1, root2):
        # check against each node if two graph have same value yet different memory addresses
        self.assertEqual(root1, root2)
        node_set1 = set(get_dfs_traversal(root1))
        node_set2 = set(get_dfs_traversal(root2))
        self.assertEqual(set(), node_set1 & node_set2)

    def test_empty_graph(self):
        self.assertIsNone(deep_copy(None))

    def test_graph_with_one_node(self):
        root = Node(1)
        self.assert_result(root, deep_copy(root))

    def test_k3(self):
        n = [Node(i) for i in xrange(3)]
        n[0].neighbors = [n[1], n[2]]
        n[1].neighbors = [n[0], n[2]]
        n[2].neighbors = [n[1], n[0]]
        self.assert_result(n[0], deep_copy(n[0]))
    
    def test_DAG(self):
        n = [Node(i) for i in xrange(4)]
        n[0].neighbors = [n[1], n[2]]
        n[1].neighbors = [n[2], n[3]]
        n[2].neighbors = [n[3]]
        n[3].neighbors = []
        self.assert_result(n[0], deep_copy(n[0]))

    def test_graph_with_cycle(self):
        n = [Node(i) for i in xrange(6)]
        n[0].neighbors = [n[1]]
        n[1].neighbors = [n[2]]
        n[2].neighbors = [n[3]]
        n[3].neighbors = [n[4]]
        n[4].neighbors = [n[5]]
        n[5].neighbors = [n[2]]
        self.assert_result(n[0], deep_copy(n[0]))

    def test_k10(self):
        n = [Node(i) for i in xrange(10)]
        for i in xrange(10):
            n[i].neighbors = n[:i] + n[i+1:]
        self.assert_result(n[0], deep_copy(n[0]))

    def test_tree(self):
        n = [Node(i) for i in xrange(5)]
        n[0].neighbors = [n[1], n[2]]
        n[2].neighbors = [n[3], n[4]]
        self.assert_result(n[0], deep_copy(n[0]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 5, 2020 LC 678 [Medium] Balanced Parentheses with Wildcard
---
> **Question:** You're given a string consisting solely of `(`, `)`, and `*`. `*` can represent either a `(`, `)`, or an empty string. Determine whether the parentheses are balanced.
>
> For example, `(()*` and `(*)` are balanced. `)*(` is not balanced.

**My thoughts:** The wildcard `*` can represents `-1`, `0`, `1`, thus `x` number of `"*"`s can represents range from `-x` to `x`. Just like how we check balance without wildcard, but this time balance is a range: the wildcard just make any balance number within the range become possible. While keep the balance range in mind, we need to make sure each time the range can never go below 0 to become unbalanced, ie. number of open parentheses less than close ones.  

**Solution:** [https://repl.it/@trsong/Balanced-Parentheses-with-Wildcard](https://repl.it/@trsong/Balanced-Parentheses-with-Wildcard)
```py
import unittest

def balanced_parentheses(s):
    balance_low = balance_high = 0

    for ch in s:
        if balance_high < 0:
            return False
        elif ch == '(':
            balance_low += 1
            balance_high += 1
        elif ch == ')':
            balance_low -= 1
            balance_high -= 1
        elif ch == '*':
            balance_high += 1
            balance_low -= 1

        # balance should never below 0
        balance_low = max(balance_low, 0)
    
    return balance_low <= 0 <= balance_high


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
    unittest.main(exit=False)
```

### Mar 4, 2020 [Easy] Longest Common Prefix
---
> **Question:** Given a list of strings, find the longest common prefix between all strings.

**Example:**
```py 
longest_common_prefix(['helloworld', 'hellokitty', 'helly'])
# returns 'hell'
```


**Solution:** [https://repl.it/@trsong/Longest-Common-Prefix](https://repl.it/@trsong/Longest-Common-Prefix)
```py
import unittest

def longest_common_prefix(words):
    if not words:
        return ''

    for i, ch in enumerate(words[0]):
        for word in words:
            if i >= len(word) or word[i] != ch:
                return words[0][:i]
    
    return None


class LongestCommonPrefixSpec(unittest.TestCase):
    def test_example(self):
        words = ['helloworld', 'hellokitty', 'helly']
        expected = 'hell'
        self.assertEqual(expected, longest_common_prefix(words))

    def test_words_share_prefix(self):
        words = ['flower', 'flow', 'flight']
        expected = 'fl'
        self.assertEqual(expected, longest_common_prefix(words))

    def test_words_not_share_prefix(self):
        words = ['dog', 'racecar', 'car']
        expected = ''
        self.assertEqual(expected, longest_common_prefix(words))

    def test_list_with_empty_word(self):
        words = ['abc', 'abc', '']
        expected = ''
        self.assertEqual(expected, longest_common_prefix(words))

    def test_empty_array(self):
        words = []
        expected = ''
        self.assertEqual(expected, longest_common_prefix(words))

    def test_prefix_words(self):
        words = ['abcdefgh', 'abcdefg', 'abcd', 'abcde']
        expected = 'abcd'
        self.assertEqual(expected, longest_common_prefix(words))

    def test_nothing_in_common(self):
        words = ['abc', 'def', 'ghi', '123']
        expected = ''
        self.assertEqual(expected, longest_common_prefix(words))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 3, 2020 [Hard] Unique Sum Combinations
---
> **Question:** Given a list of numbers and a target number, find all possible unique subsets of the list of numbers that sum up to the target number. The numbers will all be positive numbers.

**Example:**
```py
sum_combinations([10, 1, 2, 7, 6, 1, 5], 8)
# returns [(2, 6), (1, 1, 6), (1, 2, 5), (1, 7)]
# order doesn't matter
```


**Solution with Backtracking:** [https://repl.it/@trsong/Unique-Sum-Combinations](https://repl.it/@trsong/Unique-Sum-Combinations)
```py
import unittest

def find_uniq_sum_combinations(nums, target):
    res = []
    nums.sort()
    backtack_sum(res, [], nums, 0, target)
    return res


def backtack_sum(res, accu_list, nums, index, remain_target):
    if remain_target == 0:
        res.append(accu_list[:])
    else:
        n = len(nums)
        for i in xrange(index, n):
            cur_num = nums[i]

            if i > index and nums[i] == nums[i-1]:
                # skip duplicates
                continue

            if cur_num > remain_target:
                # remaining numbers are even larger
                break
                
            accu_list.append(cur_num)
            backtack_sum(res, accu_list, nums, i+1, remain_target - cur_num)
            accu_list.pop()


class FindUniqSumCombinationSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(len(expected), len(result))
        for l1, l2 in zip(expected, result):
            l1.sort()
            l2.sort()
        expected.sort()
        result.sort()
        self.assertEqual(expected, result)

    def test_example(self):
        target, nums = 8, [10, 1, 2, 7, 6, 1, 5]
        expected = [[2, 6], [1, 1, 6], [1, 2, 5], [1, 7]]
        self.assert_result(expected, find_uniq_sum_combinations(nums, target))

    def test_ascending_list(self):
        target, nums = 10, [2, 3, 5, 6, 8, 10]
        expected = [[5, 2, 3], [2, 8], [10]]
        self.assert_result(expected, find_uniq_sum_combinations(nums, target))

    def test_ascending_list2(self):
        target, nums = 10, [1, 2, 3, 4, 5]
        expected = [[4, 3, 2, 1], [5, 3, 2], [5, 4, 1]]
        self.assert_result(expected, find_uniq_sum_combinations(nums, target))

    def test_empty_array(self):
        target, nums = 0, []
        expected = [[]]
        self.assert_result(expected, find_uniq_sum_combinations(nums, target))

    def test_empty_array2(self):
        target, nums = 1, []
        expected = []
        self.assert_result(expected, find_uniq_sum_combinations(nums, target))

    def test_unable_to_find_target_sum(self):
        target, nums = -1, [1, 2, 3]
        expected = []
        self.assert_result(expected, find_uniq_sum_combinations(nums, target))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 2, 2020 LC 78 [Medium] Generate All Subsets
---
> **Question:** Given a list of unique numbers, generate all possible subsets without duplicates. This includes the empty set as well.

**Example:**
```py
generate_all_subsets([1, 2, 3])
# [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]
```

**My thoughts:** Solution 1 use reduce: initially the set of all subsets is `[[]]`. For each element e, create two copy of the all subsets, one remain the same, the other insert e to each subsets. e.g. `[[]]` => `[[]] + [[1]]` => `[[], [1]] + [[2], [1, 2]]`

Solution 2 use binary representation of `2^n`. Each set digit represents chosen elements. e.g. Binary representation `0101` in `[1, 2, 3, 4]` represents `[2, 4]`

**Solution with Recursion:** [https://repl.it/@trsong/Generate-All-Subsets-with-recursion](https://repl.it/@trsong/Generate-All-Subsets-with-recursion)
```py
from functools import reduce

def generate_all_subsets(nums):
    return reduce(
        lambda accu_subsets, elem: accu_subsets + map(lambda subset: subset + [elem], accu_subsets),
        nums,
        [[]])
```

**Solution with Binary Representation:** [https://repl.it/@trsong/Generate-All-Subsets](https://repl.it/@trsong/Generate-All-Subsets)
```py
import unittest
import math

def last_significant_bit_index(num):
    without_last_bit = num & (num - 1)  # clear last significant bit
    last_bit_number = num - without_last_bit
    base = 2
    index = int(math.log(last_bit_number, base)) if last_bit_number > 0 else -1
    return index


def all_set_indexes(num):
    while num > 0:
        i = last_significant_bit_index(num)
        yield i
        num &= ~(1 << i)


def subset_generator(nums):
    n = len(nums)
    for chosen_binary in xrange(2 ** n):
        yield [nums[i] for i in all_set_indexes(chosen_binary)]
        

def generate_all_subsets(nums):
    return [subset for subset in subset_generator(nums)]


class GenerateAllSubsetSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(len(expected), len(result))
        for l1, l2 in zip(expected, result):
            l1.sort()
            l2.sort()
        expected.sort()
        result.sort()
        self.assertEqual(expected, result)

    def test_example(self):
        nums = [1, 2, 3]
        expected = [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]
        self.assert_result(expected, generate_all_subsets(nums))

    def test_empty_list(self):
        nums = []
        expected = [[]]
        self.assert_result(expected, generate_all_subsets(nums))

    def test_one_elem_list(self):
        nums = [1]
        expected = [[], [1]]
        self.assert_result(expected, generate_all_subsets(nums))

    def test_two_elem_list(self):
        nums = [1, 2]
        expected = [[1], [2], [1, 2], []]
        self.assert_result(expected, generate_all_subsets(nums))

    def test_four_elem_list(self):
        nums = [1, 2, 3, 4]
        expected = [
            [], 
            [1], [2], [3],  [4],
            [1, 2], [1, 3], [2, 3], [1, 4], [2, 4], [3, 4],
            [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4], [1, 2, 3, 4]
        ]
        self.assert_result(expected, generate_all_subsets(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Mar 1, 2020 LC 34 [Medium] Range Searching in a Sorted List
---
> **Question:** Given a sorted list with duplicates, and a target number n, find the range in which the number exists (represented as a tuple `(low, high)`, both inclusive. If the number does not exist in the list, return `(-1, -1)`). 

**Example 1:**
```py
search_range([1, 1, 3, 5, 7], 1)  # returns (0, 1)
```

**Example 2:**
```py
search_range([1, 2, 3, 4], 5)  # returns (-1, -1)
```

**Solution with Binary Search:** [https://repl.it/@trsong/Range-Searching-in-a-Sorted-List](https://repl.it/@trsong/Range-Searching-in-a-Sorted-List)
```py
import unittest

def binary_search(nums, target, exclusive=False):
    # by default exclusive is off: return the smallest index of number >= target
    # if exclusive is turned on: return the smallest index of number > target
    lo = 0
    hi = len(nums)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if exclusive and nums[mid] == target:
            lo = mid + 1
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def search_range(nums, target):
    if not nums or target < nums[0] or target > nums[-1]:
        return (-1, -1)

    lo = binary_search(nums, target)
    if nums[lo] != target:
        return (-1, -1)
        
    hi = binary_search(nums, target, exclusive=True) - 1
    return (lo, hi)


class SearchRangeSpec(unittest.TestCase):
    def test_example1(self):
        target, nums = 1, [1, 1, 3, 5, 7]
        expected = (0, 1)
        self.assertEqual(expected, search_range(nums, target))

    def test_example2(self):
        target, nums = 5, [1, 2, 3, 4]
        expected = (-1, -1)
        self.assertEqual(expected, search_range(nums, target))

    def test_empty_list(self):
        target, nums = 0, []
        expected = (-1, -1)
        self.assertEqual(expected, search_range(nums, target))

    def test_list_with_unique_value(self):
        target, nums = 1, [1, 1, 1, 1]
        expected = (0, 3)
        self.assertEqual(expected, search_range(nums, target))

    def test_list_with_duplicate_elements(self):
        target, nums = 0, [0, 0, 0, 1, 1, 1, 1]
        expected = (0, 2)
        self.assertEqual(expected, search_range(nums, target))

    def test_list_with_duplicate_elements2(self):
        target, nums = 1, [0, 1, 1, 1, 1, 2, 2, 2, 2]
        expected = (1, 4)
        self.assertEqual(expected, search_range(nums, target))

    def test_target_element_fall_into_a_specific_range(self):
        target, nums = 10, [1, 3, 5, 7, 9, 11, 11, 12]
        expected = (-1, -1)
        self.assertEqual(expected, search_range(nums, target))

    def test_smaller_than_min_element(self):
        target, nums = -10, [0]
        expected = (-1, -1)
        self.assertEqual(expected, search_range(nums, target))

    def test_larger_than_max_element(self):
        target, nums = 10, [0]
        expected = (-1, -1)
        self.assertEqual(expected, search_range(nums, target))

    def test_target_is_the_max_element(self):
        target, nums = 1, [0, 1]
        expected = (1, 1)
        self.assertEqual(expected, search_range(nums, target))

    def test_target_is_the_min_element(self):
        target, nums = 0, [0, 1]
        expected = (0, 0)
        self.assertEqual(expected, search_range(nums, target))
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Feb 29, 2020 [Easy] Deepest Node in a Binary Tree
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

**Solution with DFS:** [https://repl.it/@trsong/Deepest-Node-in-a-Binary-Tree](https://repl.it/@trsong/Deepest-Node-in-a-Binary-Tree)
```py
import unittest

class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def find_deepest_node(root):
    if not root:
        return None
    
    max_depth = 0
    deepest_node = root
    stack = [(root, 0)]
    
    while stack:
        cur, depth = stack.pop()
        if depth > max_depth:
            deepest_node = cur
            max_depth = depth
        
        if cur.left:
            stack.append((cur.left, depth+1))
        if cur.right:
            stack.append((cur.right, depth+1))
    
    return deepest_node


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


### Feb 28, 2020 [Medium] Index of Larger Next Number
---
> **Question:** Given a list of numbers, for each element find the next element that is larger than the current element. Return the answer as a list of indices. If there are no elements larger than the current element, then use -1 instead.

**Example:** 
```py
larger_number([3, 2, 5, 6, 9, 8])
# return [2, 2, 3, 4, -1, -1]
```

**My thoughts:** The idea is to iterate backwards and only store large element along the way with a stack. Doing such will mantain the stack in ascending order. We can then treat the stack as a history of larger element on the right. The algorithm work in the following way:

For each element, we push current element in stack. And the the same time, we pop all elements that are smaller than current element until we find a larger element that is the next larger element in the list. 

Note that in worst case scenario, each element can only be pushed and poped from stack once, leaves the time complexity being `O(n)`.

**Solution with Stack:** [https://repl.it/@trsong/Index-of-Larger-Next-Number](https://repl.it/@trsong/Index-of-Larger-Next-Number)
```py
import unittest

def larger_number(nums):
    if not nums:
        return []

    n = len(nums)
    res = [None] * n
    stack = []

    # iterate backwards and store largest element along the way
    for j in xrange(n-1, -1, -1):
        while stack and nums[stack[-1]] <= nums[j]:
            # keep poping smaller element until find larger element
            stack.pop()

        if not stack:
            res[j] = -1
        else:
            res[j] = stack[-1]

        stack.append(j)
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

### Feb 27, 2020 [Medium] Maximum Non Adjacent Sum
---
> **Question:** Given a list of positive numbers, find the largest possible set such that no elements are adjacent numbers of each other.

**Example 1:**
```py
max_non_adjacent_sum([3, 4, 1, 1])
# returns 5
# max sum is 4 (index 1) + 1 (index 3)
```

**Example 2:**
```py
max_non_adjacent_sum([2, 1, 2, 7, 3])
# returns 9
# max sum is 2 (index 0) + 7 (index 3)
```

**Solution with DP:** [https://repl.it/@trsong/Maximum-Non-Adjacent-Sum](https://repl.it/@trsong/Maximum-Non-Adjacent-Sum)
```py
import unittest

def max_non_adjacent_sum(nums):
    if not nums:
        return 0

    n = len(nums)
    # Let dp[i] represents max non adj sum after consume i numbers from nums
    # dp[i] = max{dp[i-1], dp[i-2] + nums[i-1]}
    dp = [0] * (n+1)
    dp[1] = max(nums[0], 0)  # nums[0] can be negative

    for i in xrange(2, n+1):
        # choose max between include and exclude current number
        dp[i] = max(dp[i-1], dp[i-2] + nums[i-1])

    return dp[n]


class MaxNonAdjacentSumSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(5, max_non_adjacent_sum([3, 4, 1, 1]))  # 4 + 1 

    def test_example2(self):
        self.assertEqual(9, max_non_adjacent_sum([2, 1, 2, 7, 3]))  # 7 + 2

    def test_example3(self):
        self.assertEqual(110, max_non_adjacent_sum([5, 5, 10, 100, 10, 5])) 

    def test_empty_array(self):
        self.assertEqual(0, max_non_adjacent_sum([]))

    def test_length_one_array(self):
        self.assertEqual(42, max_non_adjacent_sum([42]))

    def test_length_one_array2(self):
        self.assertEqual(0, max_non_adjacent_sum([-10]))

    def test_length_two_array(self):
        self.assertEqual(0, max_non_adjacent_sum([-20, -10]))

    def test_length_three_array(self):
        self.assertEqual(1, max_non_adjacent_sum([1, -1, -1]))

    def test_length_three_array2(self):
        self.assertEqual(0, max_non_adjacent_sum([-1, -1, -1]))

    def test_length_three_array3(self):
        self.assertEqual(3, max_non_adjacent_sum([1, 3, 1]))

    def test_length_three_array4(self):
        self.assertEqual(4, max_non_adjacent_sum([2, 3, 2]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Feb 26, 2020 [Medium] Majority Element
---
> **Question:** A majority element is an element that appears more than half the time. Given a list with a majority element, find the majority element.

**Example:**
```py
majority_element([3, 5, 3, 3, 2, 4, 3])  # gives 3
```

**My thoughts:** ***Boyce-Moore Voting Algorithm*** is the one we should take a close look at. As it can gives `O(n)` time complexity and `O(1)` space complexity: here is how it works, the idea is to shrink the array so that the majority result is equivalent between the original array as well as the shrinked array.

The way we shrink the array is to treat the very first element as majority candidate and shrink the array recursively.

- If the candidate is not majority, there exists an even point `p > 0` such that the number of "majority" vs "minority" is the same. And we chop out the array before and equal to the even point p. And the real majority of the rest of array should be the same as the shrinked array
- If the candidate is indeed majority however there is still an even point q such that the number of majority vs minority is the same. And we the same thing to chop out the array before and equal to the even point q. And a majority should still be a majority of the rest of array as we eliminate same number of majority and minority that leaves the majority unchange.
- If the candidate is indeed majority and there is no even point such that the number of majority vs minority is the same. Thus the candidate can be safely returned as majority.


**Solution with Boyce-Moore Voting Algorithm:** [https://repl.it/@trsong/Find-the-Majority-Element](https://repl.it/@trsong/Find-the-Majority-Element)
```py
import unittest

def majority_element(nums):
    if not nums:
        return None
    
    major_elem = nums[0]
    count = 0

    for elem in nums:
        if count == 0:
            major_elem = elem
            count += 1
        elif elem == major_elem:
            count +=1
        else:
            count -=1
    
    if count_target(nums, major_elem) > len(nums) / 2:
        return major_elem
    else:
        return None


def count_target(nums, target):
    count = 0
    for elem in nums:
        if elem == target:
            count += 1
    return count


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
    unittest.main(exit=False)
```

### Feb 25, 2020 [Easy] Mouse Holes
---
> **Question:** Consider the following scenario: there are N mice and N holes placed at integer points along a line. Given this, find a method that maps mice to holes such that the largest number of steps any mouse takes is minimized.
>
> Each move consists of moving one mouse one unit to the left or right, and only one mouse can fit inside each hole.
>
> For example, suppose the mice are positioned at `[1, 4, 9, 15]`, and the holes are located at `[10, -5, 0, 16]`. In this case, the best pairing would require us to send the mouse at 1 to the hole at -5, so our function should return 6.


**Greedy Solution:** [https://repl.it/@trsong/Mouse-Holes](https://repl.it/@trsong/Mouse-Holes)
```py
import unittest

def min_last_mouse_steps(mouse_positions, hole_positions):
    if not mouse_positions or not hole_positions:
        return 0
    
    mouse_positions.sort()
    hole_positions.sort()

    last_mouse_steps = abs(mouse_positions[0] - hole_positions[0])
    for mouse, hole in zip(mouse_positions, hole_positions):
        last_mouse_steps = max(last_mouse_steps, abs(mouse - hole))
    
    return last_mouse_steps


class MisLastMouseStepSpec(unittest.TestCase):
    def test_example(self):
        mouse_positions = [1, 4, 9, 15]
        hole_positions = [10, -5, 0, 16]
        # sorted mouse: 1 4 9 15
        # sorted hole: -5 0 10 16
        # distance: 6 4 1 1
        expected = 6
        self.assertEqual(expected, min_last_mouse_steps(mouse_positions, hole_positions))

    def test_no_mice_nor_holes(self):
        self.assertEqual(0, min_last_mouse_steps([], []))

    def test_simple_case(self):
        mouse_positions = [0, 1, 2]
        hole_positions = [0, 1, 2]
        # sorted mouse: 0 1 2
        # sorted hole: 0 1 2
        # distance: 0 0 0
        expected = 0
        self.assertEqual(expected, min_last_mouse_steps(mouse_positions, hole_positions))

    def test_position_in_reverse_order(self):
        mouse_positions = [0, 1, 2]
        hole_positions = [2, 1, 0]
        # sorted mouse: 0 1 2
        # sorted hole: 0 1 2
        # distance: 0 0 0
        expected = 0
        self.assertEqual(expected, min_last_mouse_steps(mouse_positions, hole_positions))

    def test_unorded_positions(self):
        mouse_positions = [4, -4, 2]
        hole_positions = [4, 0, 5]
        # sorted mouse: -4 2 4
        # sorted hole:   0 4 5
        # distance: 4 2 1
        expected = 4
        self.assertEqual(expected, min_last_mouse_steps(mouse_positions, hole_positions))

    def test_large_example(self):
        mouse_positions = [-10, -79, -79, 67, 93, -85, -28, -94]
        hole_positions = [-2, 9, 69, 25, -31, 23, 50, 78]
        # sorted mouse: -94 -85 -79 -79 -28 -10 67 93
        # sorted hole: -31 -2 9 23 25 50 69 78
        # distance: 63 83 88 102 53 60 2 15
        expected = 102
        self.assertEqual(expected, min_last_mouse_steps(mouse_positions, hole_positions))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Feb 24, 2020 [Medium] Number of Flips to Make Binary String
---
> **Question:** You are given a string consisting of the letters x and y, such as xyxxxyxyy. In addition, you have an operation called flip, which changes a single x to y or vice versa.
>
> Determine how many times you would need to apply this operation to ensure that all x's come before all y's. In the preceding example, it suffices to flip the second and sixth characters, so you should return 2.

**My thoughts:** Basically, the question is about finding a sweet cutting spot so that # flip on left plus # flip on right is minimized. We can simply scan through the array from left and right to allow constant time query for number of flip need on the left and right for a given spot. And the final answer is just the min of sum of left and right flips.

**Solution:** [https://repl.it/@trsong/Number-of-Flips-to-Make-Binary-String](https://repl.it/@trsong/Number-of-Flips-to-Make-Binary-String)
```py
import unittest

def min_flip_to_make_binary(s):
    if not s:
        return 0
    
    n = len(s)
    y_on_left = [0] * n
    x_on_right = [0] * n
    prev_y = 0
    prev_x = 0
    
    for i in xrange(n):
        y_on_left[i] = prev_y
        prev_y += 1 if s[i] == 'y' else 0

    for j in xrange(n-1, -1, -1):
        x_on_right[j] = prev_x
        prev_x += 1 if s[j] == 'x' else 0

    num_flip = n
    for y, x in zip(y_on_left, x_on_right):
        num_flip = min(num_flip, y + x)

    return num_flip


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

### Feb 23, 2020 LC 163 [Medium] Missing Ranges
---
> **Question:** Given a sorted list of numbers, and two integers low and high representing the lower and upper bound of a range, return a list of (inclusive) ranges where the numbers are missing. A range should be represented by a tuple in the format of (lower, upper).

**Example:**
```py
missing_ranges(nums=[1, 3, 5, 10], lower=1, upper=10)
# returns [(2, 2), (4, 4), (6, 9)]
```

**Solution:** [https://repl.it/@trsong/Missing-Ranges](https://repl.it/@trsong/Missing-Ranges)
```py
import unittest
import sys

def missing_ranges(nums, lower, upper):
    if not nums:
        return [(lower, upper)]

    res = []
    for i, num in enumerate(nums):
        if lower > upper:
            break

        if i > 0 and nums[i-1] == num:
            # skip duplicates
            continue

        if lower < num:
            res.append((lower, min(num-1, upper)))
        
        lower = num + 1

    if lower <= upper:
        res.append((lower, upper))
    
    return res
        

class MissingRangeSpec(unittest.TestCase):
    def test_example(self):
        lower, upper, nums = 1, 10, [1, 3, 5, 10] 
        expected = [(2, 2), (4, 4), (6, 9)]
        self.assertEqual(expected, missing_ranges(nums, lower, upper))

    def test_example2(self):
        lower, upper, nums = 0, 99, [0, 1, 3, 50, 75] 
        expected = [(2, 2), (4, 49), (51, 74), (76, 99)]
        self.assertEqual(expected, missing_ranges(nums, lower, upper))

    def test_empty_array(self):
        lower, upper, nums = 1, 42, []
        expected = [(1, 42)]
        self.assertEqual(expected, missing_ranges(nums, lower, upper))

    def test_target_range_greater_than_array_range(self):
        lower, upper, nums = 1, 5, [2, 3]
        expected = [(1, 1), (4, 5)]
        self.assertEqual(expected, missing_ranges(nums, lower, upper))

    def test_lower_bound_equals_upper_bound(self):
        lower, upper, nums = 1, 1, [0, 1, 4]
        expected = []
        self.assertEqual(expected, missing_ranges(nums, lower, upper))

    def test_lower_bound_equals_upper_bound2(self):
        lower, upper, nums = 1, 1, [0, 2, 3, 4]
        expected = [(1, 1)]
        self.assertEqual(expected, missing_ranges(nums, lower, upper))

    def test_lower_larger_than_high(self):
        self.assertEqual([], missing_ranges([1, 2], 10, 1))

    def test_missing_range_of_different_length(self):
        lower, upper, nums = 1, 11, [0, 1, 3,  6, 10, 11]
        expected = [(2, 2), (4, 5), (7, 9)]
        self.assertEqual(expected, missing_ranges(nums, lower, upper))

    def test_range_not_overflow(self):
        lower, upper, nums = -sys.maxint - 1, sys.maxint, [0]
        expected = [(-sys.maxint - 1, -1), (1, sys.maxint)]
        self.assertEqual(expected, missing_ranges(nums, lower, upper))

    def test_right_bound(self):
        lower, upper, nums = 0, 1, [1]
        expected = [(0, 0)]
        self.assertEqual(expected, missing_ranges(nums, lower, upper))

    def test_left_bound(self):
        lower, upper, nums = 0, 1, [0]
        expected = [(1, 1)]
        self.assertEqual(expected, missing_ranges(nums, lower, upper))

    def test_no_missing_range(self):
        lower, upper, nums = 4, 6, [0, 1, 2, 3, 4, 5, 6, 7, 8]
        expected = []
        self.assertEqual(expected, missing_ranges(nums, lower, upper))
    
    def test_duplicate_numbers(self):
        lower, upper, nums = 3, 14, [4, 4, 4, 5, 5, 7, 7, 7, 9, 9, 9, 11, 11, 16]
        expected = [(3, 3), (6, 6), (8, 8), (10, 10), (12, 14)]
        self.assertEqual(expected, missing_ranges(nums, lower, upper))
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Feb 22, 2020 LC 79 [Medium] Word Search
---
> **Question:** Given a 2D board and a word, find if the word exists in the grid.
>
> The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

**Example:**
```py
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
```


**Solution with Backtracking:** [https://repl.it/@trsong/Word-Search](https://repl.it/@trsong/Word-Search)
```py
import unittest

def search_word(grid, word):
    if not word or not grid or not grid[0]:
        return False
    
    n, m = len(grid), len(grid[0])
    visited = [[False for _ in xrange(m)] for _ in xrange(n)]
        
    for r in xrange(n):
        for c in xrange(m):
            if grid[r][c] == word[0] and backtrack_word(grid, word, visited, r, c, 0):
                return True
    return False


def backtrack_word(grid, word, visited, row, col, index):
    n, m = len(grid), len(grid[0])
    if index == len(word):
        return True
    elif row >= n or row < 0 or col >= m or col < 0:
        return False
    elif grid[row][col] != word[index]:
        return False
    elif visited[row][col]:
        return False
    else:
        visited[row][col] = True
        is_valid = False
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            new_r, new_c = row + dr, col + dc
            if backtrack_word(grid, word, visited, new_r, new_c, index+1):
                is_valid = True
                break
        visited[row][col] = False
    return is_valid


class SearchWordSpec(unittest.TestCase):
    def test_example(self):
        grid = [
            ['A','B','C','E'],
            ['S','F','C','S'],
            ['A','D','E','E']
        ]
        self.assertTrue(search_word(grid, "ABCCED"))
        self.assertTrue(search_word(grid, "SEE"))
        self.assertFalse(search_word(grid, "ABCB"))

    def test_empty_word(self):
        self.assertFalse(search_word([['A']], ''))

    def test_empty_grid(self):
        self.assertFalse(search_word([[]], 'a'))

    def test_exhaust_all_characters(self):
        grid = [
            ["a","b"],
            ["c","d"]
        ]
        self.assertTrue(search_word(grid, "acdb"))

    def test_jump_is_not_allowed(self):
        grid = [
            ["a","b"],
            ["c","d"]
        ]
        self.assertFalse(search_word(grid, "adbc"))
    
    def test_wrap_around(self):
        grid = [
            ["A","B","C","E"],
            ["S","F","E","S"],
            ["A","D","E","E"]
        ]
        self.assertTrue(search_word(grid, "ABCESEEEFS"))
    
    def test_wrap_around2(self):
        grid = [
            ["A","A","A","A"],
            ["A","B","A","A"]
        ]
        self.assertTrue(search_word(grid, "AAAAAAAB"))
        self.assertTrue(search_word(grid, "AAABAAAA"))
        self.assertFalse(search_word(grid, "AAAAAAAA"))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Feb 21, 2020 LC 240 [Medium] Search a 2D Matrix II
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

**Solution with DFS and Binary Search:** [https://repl.it/@trsong/Search-a-Row-and-Column-Sorted-2D-Matrix](https://repl.it/@trsong/Search-a-Row-and-Column-Sorted-2D-Matrix)
```py
import unittest

def search_matrix(matrix, target):
    if not matrix:
        return False

    n, m = len(matrix), len(matrix[0])
    stack = [(0, n-1, 0, m-1)]
    while stack:
        row_lo, row_hi, col_lo, col_hi = stack.pop()
        if row_lo > row_hi or col_lo > col_hi:
            continue

        row_mid = row_lo + (row_hi - row_lo) // 2
        col_mid = col_lo + (col_hi - col_lo) // 2
        if matrix[row_mid][col_mid] == target:
            return True
        elif matrix[row_mid][col_mid] < target:
            # target cannot be strictly smaller than current ie. cannot go top_left
            bottom_left = (row_mid+1, row_hi, col_lo, col_mid)
            right = (row_lo, row_hi, col_mid+1, col_hi)
            stack.append(bottom_left)
            stack.append(right)
        else:
            # target cannot be strictly larger than current ie. cannot go bottom_right
            top_right = (row_lo, row_mid-1, col_mid, col_hi)
            left = (row_lo, row_hi, col_lo, col_mid-1)
            stack.append(top_right)
            stack.append(left)
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
    unittest.main(exit=False)
```

### Feb 20, 2020 \[Medium\]  Generate Brackets
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

**Solution with Backtracking:** [https://repl.it/@trsong/Generate-Brackets](https://repl.it/@trsong/Generate-Brackets)
```py
import unittest

def generate_brackets(n):
    if n <= 0:
        return []
    res = []
    string_buffer = []
    backtrack_brackets(res, string_buffer, n, n)
    return res

def backtrack_brackets(res, string_buffer, remain_left, remain_right):
    if remain_left == 0 and remain_right == 0:
        res.append("".join(string_buffer))
    else:
        if remain_left > 0:
            string_buffer.append('(')
            backtrack_brackets(res, string_buffer, remain_left-1, remain_right)
            string_buffer.pop()
        if remain_right > remain_left:
            string_buffer.append(')')
            backtrack_brackets(res, string_buffer, remain_left, remain_right-1)
            string_buffer.pop()


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
    unittest.main(exit=False)
```



### Feb 19, 2020 \[Easy\] Sum Binary Numbers
---
> **Question:** Given two binary numbers represented as strings, return the sum of the two binary numbers as a new binary represented as a string. Do this without converting the whole binary string into an integer.

**Example:**
```py
sum_binary("11101", "1011")
# returns "101000"
```

**Solution:** [https://repl.it/@trsong/Sum-Binary-Numbers](https://repl.it/@trsong/Sum-Binary-Numbers)
```py
import unittest

def binary_sum(bin1, bin2):
    reverse_result = []
    i = len(bin1) - 1
    j = len(bin2) - 1
    has_carry = False
    
    while i >= 0 or j >= 0:
        digit_sum = 1 if has_carry else 0
        if i >= 0:
            digit_sum += int(bin1[i])
            i -= 1
        if j >= 0:
            digit_sum += int(bin2[j])
            j -=1
        
        has_carry = digit_sum > 1
        digit_sum %= 2
        reverse_result.append(str(digit_sum))
    
    if has_carry:
        reverse_result.append("1")
    reverse_result.reverse()

    return "".join(reverse_result)


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
    unittest.main(exit=False)
```

### Feb 18, 2020 \[Medium\] 4 Sum
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

**Solution:** [https://repl.it/@trsong/4-Sum](https://repl.it/@trsong/4-Sum)
```py
import unittest

def two_sum(nums, target, start, end):
    while start < end:
        sum = nums[start] + nums[end]
        if sum > target:
            end -= 1
        elif sum < target:
            start += 1
        else:
            yield nums[start], nums[end]
            start += 1
            end -= 1
            while start < end and nums[start-1] == nums[start]:
                start += 1
            while start < end and nums[end+1] == nums[end]:
                end -= 1


def four_sum(nums, target):
    nums.sort()
    n = len(nums)
    res = []
    for i, first in enumerate(nums):
        if i > 0 and nums[i-1] == nums[i]:
            continue

        for j in xrange(i+1, n-2):
            if j > i+1 and nums[j-1] == nums[j]:
                continue

            second = nums[j]
            two_sum_target = target - first - second

            for third, fourth in two_sum(nums, two_sum_target, j+1, n-1):
                res.append([first, second, third, fourth])

    return res

            
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
    unittest.main(exit=False)
```

### Feb 17, 2020 \[Medium\] Paint House
---
> **Question:** A builder is looking to build a row of N houses that can be of K different colors. He has a goal of minimizing cost while ensuring that no two neighboring houses are of the same color.
>
> Given an N by K matrix where the n-th row and k-th column represents the cost to build the n-th house with k-th color, return the minimum cost which achieves this goal.


**Solution with DP:** [https://repl.it/@trsong/Paint-House](https://repl.it/@trsong/Paint-House)
```py
import unittest

def min_paint_houses_cost(paint_cost):
    if not paint_cost or not paint_cost[0]:
        return 0

    num_houses, num_colors = len(paint_cost), len(paint_cost[0])
    # Let dp[n][c] represents the min cost for first n houses with last house end at color c
    # dp[n][c] = min(dp[n-1][k]) + paint_cost[n-1][c] where color k != c
    dp = [[float('inf') for _ in xrange(num_colors)] for _ in xrange(num_houses+1)]

    for c in xrange(num_colors):
        dp[0][c] = 0

    for n in xrange(1, num_houses+1):
        first_min_cost = float('inf')
        first_min_color = -1
        second_min_cost = float('inf')

        for end_color, prev_cost in enumerate(dp[n-1]):
            if prev_cost < first_min_cost:
                second_min_cost = first_min_cost
                first_min_cost = prev_cost
                first_min_color = end_color
            elif prev_cost < second_min_cost:
                second_min_cost = prev_cost

        for color in xrange(num_colors):
            if color == first_min_color:
                dp[n][color] = second_min_cost + paint_cost[n-1][color]
            else:
                dp[n][color] = first_min_cost + paint_cost[n-1][color]
    
    # The global min cost is the min cost among n houses end in any color
    min_cost = min(dp[num_houses])
    return min_cost


class MinPaintHousesCostSpec(unittest.TestCase):
    def test_three_houses(self):
        paint_cost = [
            [7, 3, 8, 6, 1, 2],
            [5, 6, 7, 2, 4, 3],
            [10, 1, 4, 9, 7, 6]
        ]
        # min_cost: 1, 2, 1
        self.assertEqual(4, min_paint_houses_cost(paint_cost))

    def test_four_houses(self):
        paint_cost = [
            [7, 3, 8, 6, 1, 2],
            [5, 6, 7, 2, 4, 3],
            [10, 1, 4, 9, 7, 6],
            [10, 1, 4, 9, 7, 6]
        ] 
        # min_cost: 1, 2, 4, 1
        self.assertEqual(8, min_paint_houses_cost(paint_cost))

    def test_long_term_or_short_term_cost_tradeoff(self):
        paint_cost = [
            [0, 1],
            [1, 0],
            [0, 1],
            [0, 5]
        ]
        # min_cost: 1, 1, 1, 0
        self.assertEqual(3, min_paint_houses_cost(paint_cost))

    def test_long_term_or_short_term_cost_tradeoff2(self):
        paint_cost = [
            [1, 2, 3],
            [3, 2, 1],
            [1, 3, 2],
            [1, 1, 1],
            [5, 2, 1]
        ]
        # min_cost: 1, 1, 1, 1, 1
        self.assertEqual(5, min_paint_houses_cost(paint_cost))

    def test_no_houses(self):
        self.assertEqual(0, min_paint_houses_cost([]))

    def test_one_house(self):
        paint_cost = [
            [3, 2, 1, 2, 3, 4, 5]
        ]
        self.assertEqual(1, min_paint_houses_cost(paint_cost))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Feb 16, 2020 \[Medium\] Find All Cousins in Binary Tree
---
> **Question:** Two nodes in a binary tree can be called cousins if they are on the same level of the tree but have different parents. 
>
> Given a binary tree and a particular node, find all cousins of that node.


**Example:**
```py
In the following diagram 4 and 6 are cousins:

    1
   / \
  2   3
 / \   \
4   5   6
```

**My thoughts:** Scan all nodes in layer-by-layer order with BFS. Once find the target node, simply return all nodes on that layer except for the sibling.

**Solution with BFS:** [https://repl.it/@trsong/Find-All-Cousins-in-Binary-Tree](https://repl.it/@trsong/Find-All-Cousins-in-Binary-Tree)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def find_cousions(root, target):
    if not root or not target:
        return []

    has_found_target = False
    queue = [root]
    while queue and not has_found_target:
        for _ in xrange(len(queue)):
            cur = queue.pop(0)
            if cur.left == target or cur.right == target:
                has_found_target = True
                continue
            
            if cur.left:
                queue.append(cur.left)
            if cur.right:
                queue.append(cur.right)

    return queue


class FindCousionSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(set(expected), set(result))

    def test_example(self):
        """
            1
           / \
          2   3
         / \   \
        4   5   6
        """
        n4 = TreeNode(4)
        n6 = TreeNode(6)
        left_tree = TreeNode(2, n4, TreeNode(5))
        right_tree = TreeNode(3, right=n6)
        root = TreeNode(1, left_tree, right_tree)
        self.assert_result([n6], find_cousions(root, n4))
        

    def test_empty_tree(self):
        self.assert_result([], find_cousions(None, None))

    def test_target_node_not_in_tree(self):
        """
            1
           / \
          2   3
         /     \
        4       5
        """
        not_exist_target = TreeNode(-1)
        left_tree = TreeNode(2, TreeNode(4))
        right_tree = TreeNode(3, right=TreeNode(5))
        root = TreeNode(1, left_tree, right_tree)
        self.assertEqual([], find_cousions(root, not_exist_target))
        self.assertEqual([], find_cousions(root, None))

    def test_root_has_no_cousions(self):
        """
            1
           / \
          2   3
         /   / 
        4   5    
        """
        left_tree = TreeNode(2, TreeNode(4))
        right_tree = TreeNode(3, TreeNode(5))
        root = TreeNode(1, left_tree, right_tree)
        self.assert_result([], find_cousions(root, root))

    def test_first_level_node_has_no_cousions(self):
        """
          1
         / \
        2   3
        """
        n2 = TreeNode(2)
        root = TreeNode(1, n2, TreeNode(3))
        self.assert_result([], find_cousions(root, n2))

    def test_get_all_cousions_of_internal_node(self):
        """
              1
             / \
            2   3
           / \   \
          4   5   6
         /   /   / \ 
        7   8   9  10
        """
        n4 = TreeNode(4, TreeNode(7))
        n5 = TreeNode(5, TreeNode(8))
        n2 = TreeNode(2, n4, n5)
        n6 = TreeNode(6, TreeNode(9), TreeNode(10))
        n3 = TreeNode(3, right=n6)
        root = TreeNode(1, n2, n3)
        self.assert_result([n4, n5], find_cousions(root, n6))

    def test_tree_has_unique_value(self):
        """
             1
           /   \
          1     1
         / \   / \
        1   1 1   1
        """
        ll, lr, rl, rr = TreeNode(1), TreeNode(1), TreeNode(1), TreeNode(1)
        l = TreeNode(1, ll, lr)
        r = TreeNode(1, rl, rr)
        root = TreeNode(1, l, r)
        self.assertEqual([ll, lr], find_cousions(root, rr))

    def test_internal_node_without_cousion(self):
        """
          1
         / \
        2   3
           /
          4
           \ 
            5
        """
        n4 = TreeNode(4, right=TreeNode(5))
        n3 = TreeNode(3, n4)
        root = TreeNode(1, TreeNode(2), n3)
        self.assertEqual([], find_cousions(root, n4))

    def test_get_all_cousions_of_leaf_node(self):
        """
              ____ 1 ___
             /          \
            2            3
           /   \       /    \
          4     5     6      7
         / \   / \   / \    /  \
        8   9 10 11 12 13  14  15
        """
        nodes = [TreeNode(i) for i in xrange(16)]
        nodes[4].left, nodes[4].right = nodes[8], nodes[9]
        nodes[5].left, nodes[5].right = nodes[10], nodes[11]
        nodes[6].left, nodes[6].right = nodes[12], nodes[13]
        nodes[7].left, nodes[7].right = nodes[14], nodes[15]
        nodes[2].left, nodes[2].right = nodes[4], nodes[5]
        nodes[3].left, nodes[3].right = nodes[6], nodes[7]
        nodes[1].left, nodes[1].right = nodes[2], nodes[3]
        target, expected = nodes[14], nodes[8:14]
        self.assert_result(expected, find_cousions(nodes[1], target))
        

if __name__ == '__main__':
    unittest.main(exit=False)
```


### Feb 15, 2020 \[Easy\] Common Characters
---
> **Question:** Given n strings, find the common characters in all the strings. In simple words, find characters that appear in all the strings and display them in alphabetical order or lexicographical order.

**Example:**
```py
common_characters(['google', 'facebook', 'youtube'])
# ['e', 'o']
```

**Solution with Counting Sort:** [https://repl.it/@trsong/Common-Characters](https://repl.it/@trsong/Common-Characters)
```py
import unittest

CHAR_SET_SIZE = 128

def find_common_characters(words):
    if not words:
        return []

    n = len(words)   
    char_count = [0] * CHAR_SET_SIZE
    for index, word in enumerate(words):
        expected_char_occurance = index
        for ch in word:
            ord_ch = ord(ch)
            # for each char in current word we only accept same char in previous word
            if char_count[ord_ch] == expected_char_occurance:
                char_count[ord_ch] += 1

    res = []
    for i, count in enumerate(char_count):
        if count == n:
            ch = chr(i)
            res.append(ch)
    return res


class FindCommonCharacterSpec(unittest.TestCase):
    def test_example(self):
        words = ['google', 'facebook', 'youtube']
        expected = ['e', 'o']
        self.assertEqual(expected, find_common_characters(words))

    def test_empty_array(self):
        self.assertEqual([], find_common_characters([]))

    def test_contains_empty_word(self):
        words = ['a', 'a', 'aa', '']
        expected = []
        self.assertEqual(expected, find_common_characters(words))

    def test_different_intersections(self):
        words = ['aab', 'bbc', 'acc']
        expected = []
        self.assertEqual(expected, find_common_characters(words))

    def test_contains_duplicate_characters(self):
        words = ['zbbccaa', 'fcca', 'gaaacaaac', 'tccaccc']
        expected = ['a', 'c']
        self.assertEqual(expected, find_common_characters(words))

    def test_captical_letters(self):
        words = ['aAbB', 'aAb', 'AaB']
        expected = ['A', 'a']
        self.assertEqual(expected, find_common_characters(words))
    
    def test_numbers(self):
        words = ['123321312', '3321', '1123']
        expected = ['1', '2', '3']
        self.assertEqual(expected, find_common_characters(words))

    def test_output_in_alphanumeric_orders(self):
        words = ['123a!  bcABC', '3abc  !ACB12?', 'A B abC! c1c2 b 3']
        expected = [' ', '!', '1', '2', '3', 'A', 'B', 'C', 'a', 'b', 'c']
        self.assertEqual(expected, find_common_characters(words))

    def test_no_overlapping_letters(self):
        words = ['aabbcc', '112233', 'AABBCC']
        expected = []
        self.assertEqual(expected, find_common_characters(words))  

if __name__ == '__main__':
    unittest.main(exit=False)
```


### Feb 14, 2020 \[Easy\] Minimum Number of Operations
---
> **Question:** You are only allowed to perform 2 operations, multiply a number by 2, or subtract a number by 1. Given a number x and a number y, find the minimum number of operations needed to go from x to y.

**Example:**
```py
min_operations(6, 20) # returns 3  
# (((6 - 1) * 2) * 2) = 20 : 3 operations needed only
```

**Solution with BFS:** [https://repl.it/@trsong/Minimum-Number-of-Operations](https://repl.it/@trsong/Minimum-Number-of-Operations)
```py
import unittest

def min_operations(start, end):
    queue = [start]
    level = 0
    visited = set()

    while queue:
        for _ in xrange(len(queue)):
            cur = queue.pop(0)
            if cur == end:
                return level
            if cur in visited:
                continue
            visited.add(cur)
            
            is_same_sign = cur * queue > 0
            double = 2 * cur
            if is_same_sign and abs(cur) < abs(end) and double not in visited:
                queue.append(double)
            
            decrement = cur - 1
            if decrement not in visited:
                queue.append(decrement)
        level += 1
    
    return None


class MinOperationSpec(unittest.TestCase):
    def test_example(self):
        # (((6 - 1) * 2) * 2) = 20 
        self.assertEqual(3, min_operations(6, 20))

    def test_first_double_then_decrement(self):
        # 4 * 2 - 1 = 7
        self.assertEqual(2, min_operations(4, 7))

    def test_first_decrement_then_double(self):
        # (4 - 1) * 2 = 6
        self.assertEqual(2, min_operations(4, 6))

    def test_first_decrement_then_double2(self):
        # (2 * 2 - 1) * 2 - 1 = 5
        self.assertEqual(4, min_operations(2, 5))

    def test_first_decrement_then_double3(self):
        # (((10 * 2) - 1 ) * 2 - 1) * 2
        self.assertEqual(5, min_operations(10, 74))

    def test_no_need_to_apply_operations(self):
        self.assertEqual(0, min_operations(2, 2))

    def test_avoid_inifite_loop(self):
        # ((0 - 1 - 1) * 2  - 1) * 2  = -10
        self.assertEqual(5, min_operations(0, -10))

    def test_end_is_smaller(self):
        # 10 - 1 -1 ... - 1 = 0
        self.assertEqual(10, min_operations(10, 0))

    def test_end_is_smaller2(self):
        # (10 - 1 -1 ... - 1) * 2 * 2 = 0
        self.assertEqual(13, min_operations(10, -4))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Feb 13, 2020 \[Easy\] Minimum Step to Reach One
---
> **Question:** Given a positive integer N, find the smallest number of steps it will take to reach 1.
>
> There are two kinds of permitted steps:
> - You may decrement N to N - 1.
> - If a * b = N, you may decrement N to the larger of a and b.
> 
> For example, given 100, you can reach 1 in five steps with the following route: 100 -> 10 -> 9 -> 3 -> 2 -> 1.


**DP Solution:** [https://repl.it/@trsong/Minimum-Step-to-Reach-One-DP-Solution](https://repl.it/@trsong/Minimum-Step-to-Reach-One-DP-Solution)
```py
def min_step(target):
    dp = [float('inf')] * (target + 1)
    dp[1] = 0

    for i in xrange(1, target+1):
        if i < target:
            dp[i+1] = min(dp[i+1], dp[i] + 1)
        for j in xrange(2, min(i, target) + 1):
            if i * j > target:
                break
            product = i * j
            dp[product] = min(dp[product], dp[i] + 1)
            
    return dp[target]
```

**Solution with BFS:** [https://repl.it/@trsong/Minimum-Step-to-Reach-One-BFS](https://repl.it/@trsong/Minimum-Step-to-Reach-One-BFS)
```py
import unittest
import math

def min_step(target):
    visited = set()
    queue = [target]
    level = 0

    while queue:
        for _ in xrange(len(queue)):
            cur = queue.pop(0)
            if cur in visited:
                continue
            if cur == 1:
                return level
            visited.add(cur)

            # neighbors: number that are 1 distance away
            sqrt_cur = int(math.sqrt(cur))
            for neighbor in xrange(sqrt_cur, cur):
                if neighbor * neighbor < cur or neighbor in visited or cur % neighbor != 0:
                    continue
                queue.append(neighbor)

            if cur > 1 and cur - 1 not in visited:
                queue.append(cur - 1)
        level += 1

    return None
            

class MinStepSpec(unittest.TestCase):
    def test_example(self):
        # 100 -> 10 -> 9 -> 3 -> 2 -> 1
        self.assertEqual(5, min_step(100))

    def test_one(self):
        self.assertEqual(0, min_step(1))

    def test_prime_number(self):
        # 17 -> 16 -> 4 -> 2 -> 1
        self.assertEqual(4, min_step(17))

    def test_even_number(self):
        # 6 -> 3 -> 2 -> 1
        self.assertEqual(3, min_step(6))

    def test_even_number2(self):
        # 50 -> 10 -> 5 -> 4 -> 2 -> 1
        self.assertEqual(5, min_step(50))

    def test_power_of_2(self):
        # 1024 -> 32 -> 8 -> 4 -> 2 -> 1
        self.assertEqual(5, min_step(1024))

    def test_square_number(self):
        # 16 -> 4 -> 2 -> 1
        self.assertEqual(3, min_step(16))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Feb 12, 2020 LC 218 \[Hard\] City Skyline
---
> **Question:** Given a list of building in the form of `(left, right, height)`, return what the skyline should look like. The skyline should be in the form of a list of `(x-axis, height)`, where x-axis is the next point where there is a change in height starting from 0, and height is the new height starting from the x-axis.

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

**Solution with PriorityQueue:** [https://repl.it/@trsong/City-Skyline](https://repl.it/@trsong/City-Skyline)
```py
import unittest
from Queue import PriorityQueue

def scan_city_skyline(buildings):
    unique_building_ends = set()
    for left_pos, right_pos, _ in buildings:
        # Critial positions are left end and right end + 1 of each building
        unique_building_ends.add(left_pos)
        unique_building_ends.add(right_pos+1)

    res = []
    max_heap = PriorityQueue()
    prev_height = 0
    index = 0
    
    for bar in sorted(unique_building_ends):
        # Add all buildings whose starts before the bar
        while index < len(buildings):
            left_pos, right_pos, height = buildings[index]
            if left_pos > bar:
                break
            max_heap.put((-height, right_pos))
            index += 1
        
        # Remove building that ends before the bar
        while not max_heap.empty():
            _, right_pos = max_heap.queue[0]
            if right_pos < bar:
                max_heap.get()
            else:
                break

        # We want to make sure we get max height of building and bar is between both ends 
        height = -max_heap.queue[0][0] if not max_heap.empty() else 0
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
    unittest.main(exit=False)
```

### Feb 11, 2020 LC 821 \[Medium\] Shortest Distance to Character
---
> **Question:**  Given a string s and a character c, find the distance for all characters in the string to the character c in the string s. 
>
> You can assume that the character c will appear at least once in the string.

**Example:**
```py
shortest_dist('helloworld', 'l') 
# returns [2, 1, 0, 0, 1, 2, 2, 1, 0, 1]
```

**My thoughts:** The idea is similar to Problem ["LC 42 Trap Rain Water"](https://trsong.github.io/python/java/2019/05/01/DailyQuestions/#may-11-2019-lc-42-hard-trapping-rain-water): we can simply scan from left to know the shortest distance from nearest character on the left and vice versa when we can from right to left.  

**Solution:** [https://repl.it/@trsong/Shortest-Distance-to-Character](https://repl.it/@trsong/Shortest-Distance-to-Character)
```py
import unittest

def shortest_dist_to_char(s, ch):
    n = len(s)
    res = [float('inf')] * n

    left_distance = n
    for i in xrange(n):
        if s[i] == ch:
            left_distance = 0
        res[i] = min(res[i], left_distance)
        left_distance += 1
    
    right_distance = n
    for i in xrange(n-1, -1, -1):
        if s[i] == ch:
            right_distance = 0
        res[i] = min(res[i], right_distance)
        right_distance += 1

    return res


class ShortestDistToCharSpec(unittest.TestCase):
    def test_example(self):
        ch, s = 'l', 'helloworld'
        expected = [2, 1, 0, 0, 1, 2, 2, 1, 0, 1]
        self.assertEqual(expected, shortest_dist_to_char(s, ch))

    def test_example2(self):
        ch, s = 'o', 'helloworld'
        expected = [4, 3, 2, 1, 0, 1, 0, 1, 2, 3]
        self.assertEqual(expected, shortest_dist_to_char(s, ch))

    def test_one_letter_string(self):
        self.assertEqual([0], shortest_dist_to_char('a', 'a'))

    def test_target_char_as_head(self):
        ch, s = 'a', 'abcde'
        expected = [0, 1, 2, 3, 4]
        self.assertEqual(expected, shortest_dist_to_char(s, ch))

    def test_target_char_as_last(self):
        ch, s = 'a', 'eeeeeeea'
        expected = [7, 6, 5, 4, 3, 2, 1, 0]
        self.assertEqual(expected, shortest_dist_to_char(s, ch))

    def test_unique_letter_string(self):
        ch, s = 'a', 'aaaaa'
        expected = [0, 0, 0, 0, 0]
        self.assertEqual(expected, shortest_dist_to_char(s, ch))

    def test_multiple_occurance_of_target(self):
        ch, s = 'a', 'babbabbbaabbbbb'
        expected = [1, 0, 1, 1, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 5]
        self.assertEqual(expected, shortest_dist_to_char(s, ch))

    def test_no_duplicate_letters(self):
        ch, s = 'a', 'bcadefgh'
        expected = [2, 1, 0, 1, 2, 3, 4, 5]
        self.assertEqual(expected, shortest_dist_to_char(s, ch))
    
    def test_long_string(self):
        ch, s = 'a', 'a' + 'b' * 9999
        expected = range(10000)
        self.assertEqual(expected, shortest_dist_to_char(s, ch))
    
    def test_long_string2(self):
        ch, s = 'a', 'b' * 999999 + 'a'
        expected = range(1000000)[::-1]
        self.assertEqual(expected, shortest_dist_to_char(s, ch))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Feb 10, 2020 LC 790 \[Medium\] Domino and Tromino Tiling
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

**Solution with DP:** [https://repl.it/@trsong/Domino-and-Tromino-Tiling](https://repl.it/@trsong/Domino-and-Tromino-Tiling)
```py
import unittest

# Inspired by yuweiming70's solution for LC 790
# https://bit.ly/3bwLz0y
def domino_tiling(N):
    if N <= 2:
        return N

    # Let f[n] represents # ways for 2 * n pieces:
    # f[1]: x 
    #       x
    #
    # f[2]: x x
    #       x x
    f = [0] * (N+1)
    f[1] = 1 
    f[2] = 2

    # Let g[n] represents # ways for 2*n + 1 pieces:
    # g[1]: x      or   x x
    #       x x         x
    #
    # g[2]: x x    or   x x x  
    #       x x x       x x
    g = [0] * (N+1)
    g[1] = 1
    g[2] = 2  # domino + tromino or tromino + domino

    # Pattern:
    # f[n]: x x x x = f[n-1]: x x x y  +  f[n-2]: x x y y  + g[n-2]: x x x y + g[n-2]: x x y y
    #       x x x x           x x x y             x x z z            x x y y           x x x y
    #
    # g[n]: x x x x x = f[n-1]: x x x y y + g[n-1]: x x y y 
    #       x x x x             x x x y             x x x
    for n in xrange(3, N+1):
        g[n] = f[n-1] + g[n-1]
        f[n] = f[n-1] + f[n-2] + 2 * g[n-2]

    return f[N]


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
    unittest.main(exit=False)
```

### Feb 9, 2020 LC 43 \[Medium\] Multiply Strings
---
> **Question:** Given two strings which represent non-negative integers, multiply the two numbers and return the product as a string as well. You should assume that the numbers may be sufficiently large such that the built-in integer type will not be able to store the input (Python does have BigNum, but assume it does not exist).

**Example:**
```py
multiply("11", "13")  # returns "143"
```

**My thoughts:** Just try some example, `"abc" * "de" = (100a + 10b + c) * (10d + e) = 1000ad + 10be + ... + ce`. So it seems digit multiplication `a*d` is shifted left 3 (ie. 1000 * ad) and be is shifted left by 1 (ie. 10 * be). Thus the pattern is that the shift amount has something to do with each digit's position. Therefore, we can accumulate product of each pair of digits and push forward the carry.  

**Solution:** [https://repl.it/@trsong/Multiply-Strings](https://repl.it/@trsong/Multiply-Strings)
```py
import unittest

def multiply(num1, num2):
    if num1 == "0" or num2 == "0":
        return "0"

    len1, len2 = len(num1), len(num2)
    reverse_result = [0] * (len1 + len2)
    reverse_num1 = num1[::-1]
    reverse_num2 = num2[::-1]

    # Shift and accumlate each digit 
    for i, digit1 in enumerate(reverse_num1):
        for j, digit2 in enumerate(reverse_num2):
            product = int(digit1) * int(digit2)
            shift_amount = i + j
            product_lo, product_hi = product % 10, product // 10
            reverse_result[shift_amount] += product_lo
            reverse_result[shift_amount+1] += product_hi
    
    # Push forward carries
    carry = 0
    for i in xrange(len(reverse_result)):
        reverse_result[i] += carry
        carry, digit = reverse_result[i] // 10, reverse_result[i] % 10
        reverse_result[i] = str(digit)

    
    # Remove leading zeros
    raw_result = reverse_result[::-1]
    leading_zero_ends = 0
    while leading_zero_ends < len(raw_result):
        if raw_result[leading_zero_ends] != '0':
            break
        leading_zero_ends += 1

    result = raw_result[leading_zero_ends:]
    return "".join(result)


class MultiplySpec(unittest.TestCase):
    def test_example(self):
        num1, num2 = "11", "13"
        expected = "143"
        self.assertEqual(expected, multiply(num1, num2))
    
    def test_example2(self):
        num1, num2 = "4154", "51454"
        expected = "213739916"
        self.assertEqual(expected, multiply(num1, num2))

    def test_zero(self):
        self.assertEqual("0", multiply("42", "0"))

    def test_trivial_case(self):
        num1, num2 = "1", "1"
        expected = "1"
        self.assertEqual(expected, multiply(num1, num2))

    def test_operand_contains_zero(self):
        num1, num2 = "9012", "2077"
        expected = "18717924"
        self.assertEqual(expected, multiply(num1, num2))

    def test_should_omit_leading_zeros(self):
        num1, num2 = "10", "10"
        expected = "100"
        self.assertEqual(expected, multiply(num1, num2))

    def test_should_not_overflow(self):
        num1, num2 = "99", "999999"
        expected = "98999901"
        self.assertEqual(expected, multiply(num1, num2))

    def test_result_has_lot_of_zeros(self):
        num1, num2 = "33667003667", "3"
        expected = "101001011001"
        self.assertEqual(expected, multiply(num1, num2))

    def test_handle_super_large_numbers(self):
        num1 = "654154154151454545415415454" 
        num2 = "63516561563156316545145146514654"
        expected = "41549622603955309777243716069997997007620439937711509062916"
        self.assertEqual(expected, multiply(num1, num2))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Feb 8, 2020 \[Easy\] Intersection of Lists
---
> **Question:** Given 3 sorted lists, find the intersection of those 3 lists.

**Example:**
```py
intersection([1, 2, 3, 4], [2, 4, 6, 8], [3, 4, 5])  # returns [4]
```

**Solution:** [https://repl.it/@trsong/Intersection-of-Lists](https://repl.it/@trsong/Intersection-of-Lists)
```py
import unittest

def intersection(list1, list2, list3):
    len1, len2, len3 = len(list1), len(list2), len(list3)
    i = j = k = 0
    res = []

    while i < len1 and j < len2 and k < len3:
        min_val = min(list1[i], list2[j], list3[k])
        if list1[i] == list2[j] == list3[k]:
            res.append(min_val)
        
        if list1[i] == min_val:
            i += 1
        
        if list2[j] == min_val:
            j += 1

        if list3[k] == min_val:
            k += 1

    return res


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

### Feb 7, 2020 \[Medium\] Similar Websites
---
> **Question:** You are given a list of (website, user) pairs that represent users visiting websites. Come up with a program that identifies the top k pairs of websites with the greatest similarity.

**Example:**
```py
Suppose k = 1, and the list of tuples is:

[('a', 1), ('a', 3), ('a', 5),
 ('b', 2), ('b', 6),
 ('c', 1), ('c', 2), ('c', 3), ('c', 4), ('c', 5),
 ('d', 4), ('d', 5), ('d', 6), ('d', 7),
 ('e', 1), ('e', 3), ('e': 5), ('e', 6)]
 
Then a reasonable similarity metric would most likely conclude that a and e are the most similar, so your program should return [('a', 'e')].
```

**My thoughts:** The similarity metric bewtween two sets equals intersection / union. However, as duplicate entry might occur, we have to convert normal set to multiset. So, the way to get top k similar website is first calculate the similarity score between any two websites and after that use a priority queue to mantain top k similarity pairs.

**Solution:** [https://repl.it/@trsong/Similar-Websites](https://repl.it/@trsong/Similar-Websites)
```py
import unittest
from Queue import PriorityQueue

def to_multiset(lst):
    frequency_map = {}
    for num in lst:
        frequency_map[num] = frequency_map.get(num, 0) + 1
    
    res = set()
    for k, count in frequency_map.items():
        for i in xrange(count):
            res.add((k, i))
    return res


def set_similarity(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a) + len(set_b) - intersection
    return float(intersection) / union           


def top_similar_websites(website_log, k):
    website_userlist_map = {}
    for website, user in website_log:
        if website not in website_userlist_map:
            website_userlist_map[website] = []
        website_userlist_map[website].append(user)
    
    website_userset_map = {}
    for website, userlist in website_userlist_map.items():
        website_userset_map[website] = to_multiset(userlist)

    websites = website_userset_map.keys()
    n = len(websites)
    min_heap = PriorityQueue()
    for i in xrange(n):
        w1 = websites[i]
        w1_userset = website_userset_map[w1]

        for j in xrange(i+1, n):
            w2 = websites[j]
            w2_userset = website_userset_map[w2]

            score = set_similarity(w1_userset, w2_userset)
            if min_heap.qsize() >= k and min_heap.queue[0][0] < score:
                min_heap.get()
            if min_heap.qsize() < k:
                min_heap.put((score, (w1, w2)))

    reverse_ranking = []
    while not min_heap.empty():
        score, website = min_heap.get()
        reverse_ranking.append(website)

    ranking = reverse_ranking[::-1]
    return ranking


class TopSimilarWebsiteSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        # same length
        self.assertEqual(len(expected), len(result))
        for e, r in zip(expected, result):
            # pair must be the same, order doesn't matter
            self.assertEqual(set(e), set(r))

    def test_example(self):
        website_log = [
            ('a', 1), ('a', 3), ('a', 5),
            ('b', 2), ('b', 6),
            ('c', 1), ('c', 2), ('c', 3), ('c', 4), ('c', 5),
            ('d', 4), ('d', 5), ('d', 6), ('d', 7),
            ('e', 1), ('e', 3), ('e', 5), ('e', 6)]
        # Similarity: (a,e)=3/4, (a,c)=3/5
        expected = [('a', 'e'), ('a', 'c')]
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


### Feb 6, 2020 \[Easy\] Implement a Bit Array
---
> **Question:** A bit array is a space efficient array that holds a value of 1 or 0 at each index.
> - `init(size)`: initialize the array with size
> - `set(i, val)`: updates index at i with val where val is either 1 or 0.
> - `get(i)`: gets the value at index i.


**Solution:** [https://repl.it/@trsong/Implement-a-Bit-Array](https://repl.it/@trsong/Implement-a-Bit-Array)
```py
import unittest

class BitArray(object):
    BYTE_SIZE = 8

    def __init__(self, size): 
        # initialize the array with size
        self.size = size
        self.data = [0] * (size // BitArray.BYTE_SIZE + 1)

    def set(self, i, val):
        # updates index at i with val where val is either 1 or 0.
        bucket, index = i // BitArray.BYTE_SIZE, i % BitArray.BYTE_SIZE
        mask = 1 << index
        if val:
            self.data[bucket] |= mask
        else:
            self.data[bucket] &= ~mask
    
    def get(self, i):
        # gets the value at index i.
        bucket, index = i // BitArray.BYTE_SIZE, i % BitArray.BYTE_SIZE
        mask = 1 << index
        res = self.data[bucket] & mask > 0
        return 1 if res > 0 else 0 


class BitArraySpec(unittest.TestCase):
    def test_init_an_empty_array(self):
        bit_array = BitArray(0)
        self.assertIsNotNone(bit_array)

    def test_get_unset_value(self):
        bit_array = BitArray(1)
        self.assertEqual(0, bit_array.get(0))

    def test_get_set_value(self):
        bit_array = BitArray(2)
        bit_array.set(0, 1)
        self.assertEqual(1, bit_array.get(0))

    def test_get_latest_set_value(self):
        bit_array = BitArray(3)
        bit_array.set(1, 1)
        self.assertEqual(1, bit_array.get(1))
        bit_array.set(1, 0)
        self.assertEqual(0, bit_array.get(1))
    
    def test_double_set_value(self):
        bit_array = BitArray(3)
        bit_array.set(1, 1)
        bit_array.set(1, 1)
        self.assertEqual(1, bit_array.get(1))

    def test_check_set_the_correct_bits(self):
        indices = set([0, 1, 4, 6, 7])
        bit_array = BitArray(8)
        for i in indices:
            bit_array.set(i, 1)

        for i in xrange(8):
            if i in indices:
                self.assertEqual(1, bit_array.get(i))
            else:
                self.assertEqual(0, bit_array.get(i))

    def test_check_set_the_correct_bits2(self):
        indices = set([5, 10, 15, 20, 25, 30, 35])
        bit_array = BitArray(100)
        for i in indices:
            bit_array.set(i, 1)

        for i in xrange(100):
            if i in indices:
                self.assertEqual(1, bit_array.get(i))
            else:
                self.assertEqual(0, bit_array.get(i))

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Feb 5, 2020 \[Easy\] Largest Path Sum from Root To Leaf
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

**Solution with DFS:** [https://repl.it/@trsong/Largest-Path-Sum-from-Root-To-Leaf](https://repl.it/@trsong/Largest-Path-Sum-from-Root-To-Leaf)
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

    max_path_sum = float('-inf')
    max_path_leaf = root
    parent_map = {root: None}
    stack = [(root, root.val)]

    while stack:
        current, path_sum = stack.pop()

        if not current.left and not current.right and path_sum > max_path_sum:
            max_path_sum = path_sum
            max_path_leaf = current
        else:
            for child in [current.left, current.right]:
                if child is not None:
                    parent_map[child] = current
                    stack.append((child, path_sum + child.val))

    node = max_path_leaf
    leaf_to_root_path = []
    while node is not None:
        leaf_to_root_path.append(node.val)
        node = parent_map[node]
    
    root_to_leaf_path = leaf_to_root_path[::-1]
    return root_to_leaf_path


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

### Feb 4, 2020 \[Hard\] Teams without Enemies
---
> **Question:** A teacher must divide a class of students into two teams to play dodgeball. Unfortunately, not all the kids get along, and several refuse to be put on the same team as that of their enemies.
>
> Given an adjacency list of students and their enemies, write an algorithm that finds a satisfactory pair of teams, or returns False if none exists.

**Example 1:**
```py
Given the following enemy graph you should return the teams {0, 1, 4, 5} and {2, 3}.

students = {
    0: [3],
    1: [2],
    2: [1, 4],
    3: [0, 4, 5],
    4: [2, 3],
    5: [3]
}
```

**Example 2:**
```py
On the other hand, given the input below, you should return False.

students = {
    0: [3],
    1: [2],
    2: [1, 3, 4],
    3: [0, 2, 4, 5],
    4: [2, 3],
    5: [3]
}
```

**My thoughts:** This question is basically checking if the graph is a bipartite. In previous question I used BFS by color every other nodes. [https://trsong.github.io/python/java/2019/08/02/DailyQuestionsAug/#oct-21-2019-medium-is-bipartite](https://trsong.github.io/python/java/2019/08/02/DailyQuestionsAug/#oct-21-2019-medium-is-bipartite). 

But in this question, I'd like to use something different like use union-find to check if a graph is a bipartite. The idea is based on "The enemy of my enemy is my friend". So for any student, its enemies should be friends ie. connected by union-find. If for any reason, one become a friend of enemy, then we cannot have a bipartite.

**Solution:** [https://repl.it/@trsong/Teams-without-Enemies](https://repl.it/@trsong/Teams-without-Enemies)
```py
import unittest
from collections import defaultdict

class DisjointSet(object):
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, elem):
        p = elem
        if self.parent[p] == p:
            return p
        else:
            root = self.find(self.parent[p])
            self.parent[p] = root
            return root

    def union(self, elem1, elem2):
        p1 = self.find(elem1)
        p2 = self.find(elem2)
        if p1 != p2:
            self.parent[p1] = p2

    def is_connected(self, elem1, elem2):
        return self.find(elem1) == self.find(elem2)

    def union_all(self, elems):
        if len(elems) <= 1:
            return
        root = elems[0]
        for i in xrange(1, len(elems)):
            self.union(root, elems[i])

    def find_roots(self):
        return set(self.parent)

    def find_bipartite(self):
        root_set = self.find_roots()
        if len(root_set) <= 1:
            return False
        pivot_root = next(iter(root_set))  # find first elem from set as pivot
        group1 = []
        group2 = []
        for i in xrange(len(self.parent)):
            if self.is_connected(i, pivot_root):
                group1.append(i)
            else:
                group2.append(i)
        return group1, group2


def team_without_enemies(enemy_map):
    if not enemy_map:
        return [], []
    if len(enemy_map) == 1:
        return list(enemy_map.keys()), []

    incompatible_map = defaultdict(set)
    for student, enemies in enemy_map.items():
        for enemy in enemies:
            incompatible_map[student].add(enemy)
            incompatible_map[enemy].add(student)

    uf = DisjointSet(len(incompatible_map))
    for student, enemies in incompatible_map.items():
        for enemy in enemies:
            if uf.is_connected(student, enemy):
                # One cannot be friend with enemy
                return False

        # All enemies are friends
        uf.union_all(list(enemies))

    return uf.find_bipartite()


class TeamWithoutEnemiesSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        expected_group1_set, expected_group2_set = set(expected[0]), set(expected[1])
        result_group1_set, result_group2_set = set(result[0]), set(result[1])
        outcome1 = (expected_group1_set == result_group1_set) and (expected_group2_set == result_group2_set)
        outcome2 = (expected_group2_set == result_group1_set) and (expected_group1_set == result_group2_set)
        self.assertTrue(outcome1 or outcome2)

    def test_example(self):
        enemy_map = {
            0: [3],
            1: [2],
            2: [1, 4],
            3: [0, 4, 5],
            4: [2, 3],
            5: [3]
        }
        expected = ([0, 1, 4, 5], [2, 3])
        self.assert_result(expected, team_without_enemies(enemy_map))

    def test_example2(self):
        enemy_map = {
            0: [3],
            1: [2],
            2: [1, 3, 4],
            3: [0, 2, 4, 5],
            4: [2, 3],
            5: [3]
        }
        self.assertFalse(team_without_enemies(enemy_map))

    def test_empty_graph(self):
        enemy_map = {}
        expected = ([], [])
        self.assert_result(expected, team_without_enemies(enemy_map))

    def test_one_node_graph(self):
        enemy_map = {0: []}
        expected = ([0], [])
        self.assert_result(expected, team_without_enemies(enemy_map))

    def test_disconnect_graph(self):
        enemy_map = {
            0: [],
            1: [0],
            2: [3],
            3: [4],
            4: [2]
        }
        self.assertFalse(team_without_enemies(enemy_map))

    def test_square(self):
        enemy_map = {
            0: [1],
            1: [2],
            2: [3],
            3: [0]
        }
        expected = ([0, 2], [1, 3])
        self.assert_result(expected, team_without_enemies(enemy_map))

    def test_k5(self):
        enemy_map = {
            0: [1, 2, 3, 4],
            1: [2, 3, 4],
            2: [3, 4],
            3: [3],
            4: []
        }
        self.assertFalse(team_without_enemies(enemy_map))

    def test_square2(self):
        enemy_map = {
            0: [3],
            1: [2],
            2: [1],
            3: [0, 2]
        }
        expected = ([0, 2], [1, 3])
        self.assert_result(expected, team_without_enemies(enemy_map))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Feb 3, 2020 \[Hard\] Design a Hit Counter
---
> **Question:**  Design and implement a HitCounter class that keeps track of requests (or hits). It should support the following operations:
>
> - `record(timestamp)`: records a hit that happened at timestamp
> - `total()`: returns the total number of hits recorded
> - `range(lower, upper)`: returns the number of hits that occurred between timestamps lower and upper (inclusive)
>
> **Follow-up:** What if our system has limited memory?

**My thoughts:** Based on the nature of timestamp, it will only increase, so we should append timestamp to an flexible-length array and perform binary search to query for elements. However, as the total number of timestamp might be arbitrarily large, re-allocate space once array is full is not so memory efficient. And also another concern is that keeping so many entries in memory without any persistence logic is dangerous and hard to scale in the future. 

A common way to tackle this problem is to create a fixed bucket of record and gradually add more buckets based on demand. And inactive bucket can be persistent and evict from memory, which makes it so easy to scale in the future.

**Solution:** [https://repl.it/@trsong/Design-a-Hit-Counter](https://repl.it/@trsong/Design-a-Hit-Counter)
```py
import unittest

def binary_search(sequence, low_bound):
    """
    Return index of first element that is greater than or equal low_bound
    """
    lo = 0
    hi = len(sequence)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if sequence[mid] < low_bound:
            lo = mid + 1
        else:
            hi = mid
    return lo


class PersistentHitRecord(object):
    RECORD_CAPACITY = 100

    def __init__(self):
        self.start_timestamp = None
        self.end_timestamp = None
        self.timestamp_records = []

    def size(self):
        return len(self.timestamp_records)

    def is_full(self):
        return self.size() == PersistentHitRecord.RECORD_CAPACITY
    
    def add(self, timestamp):
        if self.start_timestamp is None:
            self.start_timestamp = timestamp
        self.end_timestamp = timestamp
        self.timestamp_records.append(timestamp)


class HitCounter(object):
    def __init__(self):
        self.current_record = PersistentHitRecord()
        self.history_records = []

    def record(self, timestamp):
        """
        Records a hit that happened at timestamp
        """
        if self.current_record.is_full():
            self.history_records.append(self.current_record)
            self.current_record = PersistentHitRecord()
        self.current_record.add(timestamp)

    def total(self):
        """
        Returns the total number of hits recorded
        """
        num_current_record_entries = self.current_record.size()
        num_history_record_entries = len(self.history_records) * PersistentHitRecord.RECORD_CAPACITY
        return num_current_record_entries + num_history_record_entries

    def range(self, lower, upper):
        """
        Returns the number of hits that occurred between timestamps lower and upper (inclusive)
        """
        if lower > upper: 
            return 0
        if self.current_record.size() > 0:
            all_records = self.history_records + [self.current_record]
        else:
            all_records = self.history_records
        if not all_records:
            return 0

        start_timestamps = map(lambda entry: entry.start_timestamp, all_records)
        end_timestamps = map(lambda entry: entry.end_timestamp, all_records)

        first_bucket_index = binary_search(end_timestamps, lower)
        first_bucket = all_records[first_bucket_index].timestamp_records
        last_bucket_index = binary_search(start_timestamps, upper+1) - 1
        last_bucket = all_records[last_bucket_index].timestamp_records
        
        first_entry_index = binary_search(first_bucket, lower)
        last_entry_index = binary_search(last_bucket, upper+1)
        if first_bucket_index == last_bucket_index:
            return last_entry_index - first_entry_index
        else:
            capacity = PersistentHitRecord.RECORD_CAPACITY
            num_full_bucket_entires = (last_bucket_index - first_bucket_index - 1) * capacity
            num_first_bucket_entries = capacity - first_entry_index
            num_last_bucket_entires = last_entry_index
            return num_first_bucket_entries + num_full_bucket_entires + num_last_bucket_entires


class HitCounterSpec(unittest.TestCase):
    def test_no_record_exists(self):
        hc = HitCounter()
        self.assertEqual(0, hc.total())
        self.assertEqual(0, hc.range(float('-inf'), float('inf')))
    
    def test_return_correct_number_of_records(self):
        hc = HitCounter()
        query_number = 10000
        for i in xrange(10000):
            hc.record(i)
        self.assertEqual(query_number, hc.total())
        self.assertEqual(5000, hc.range(1, 5000))
        self.assertEqual(query_number, hc.range(float('-inf'), float('inf')))
    
    def test_return_correct_number_of_records2(self):
        hc = HitCounter()
        hc.record(1)
        self.assertEqual(1, hc.total())
        self.assertEqual(1, hc.range(-10, 10))
        hc.record(2)
        hc.record(5)
        hc.record(8)
        self.assertEqual(4, hc.total())
        self.assertEqual(3, hc.range(0, 6))
    
    def test_query_range_is_inclusive(self):
        hc = HitCounter()
        hc.record(1)
        hc.record(3)
        hc.record(5)
        hc.record(5)
        self.assertEqual(3, hc.range(3, 5))

    def test_invalid_range(self):
        hc = HitCounter()
        hc.record(1)
        self.assertEqual(0, hc.range(1, 0))

    def test_duplicated_timestamps(self):
        hc = HitCounter()
        hc.record(1)
        hc.record(1)
        hc.record(2)
        hc.record(5)
        hc.record(5)
        self.assertEqual(5, hc.total())
        self.assertEqual(3, hc.range(0, 4))

    def test_duplicated_timestamps2(self):
        hc = HitCounter()
        for i in xrange(5):
            for _ in xrange(200):
                hc.record(i)
        self.assertEqual(1000, hc.total())
        self.assertEqual(600, hc.range(2, 4))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Feb 2, 2020 LC 166 \[Medium\] Fraction to Recurring Decimal
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

**My thoughts:** There are three different outcomes when convert fraction to recurring decimals mentioned exactly in above examples: integers, repeat decimals, non-repeat decimals. For integer, it's simple, just make sure numerator is divisible by denominator (ie. remainder is 0). However, for repeat vs non-repeat, in each iteration we time remainder by 10 and perform division again, if the quotient repeats then we have a repeat decimal, otherwise we have a non-repeat decimal. 

**Solution:** [https://repl.it/@trsong/Fraction-to-Recurring-Decimal](https://repl.it/@trsong/Fraction-to-Recurring-Decimal)
```py
import unittest

def fraction_to_decimal(numerator, denominator):
    if numerator == 0:
        return "0"
    
    sign = ""
    if numerator * denominator < 0:
        sign = "-"
    
    numerator = abs(numerator)
    denominator = abs(denominator)

    quotient, remainder = numerator // denominator, numerator % denominator
    if remainder == 0:
        return sign + str(quotient)

    decimals = []
    index = 0
    seen_remainder_position = {}
    while remainder > 0:
        if remainder in seen_remainder_position:
            break
        
        seen_remainder_position[remainder] = index
        remainder *= 10
        decimals.append(str(remainder / denominator))
        remainder %= denominator
        index += 1

    if remainder > 0:
        pivot_index = seen_remainder_position[remainder]
        non_repeat_part, repeat_part = "".join(decimals[:pivot_index]), "".join(decimals[pivot_index:])
        return "{}{}.{}({})".format(sign, str(quotient), non_repeat_part, repeat_part)
    else:
        return "{}{}.{}".format(sign, str(quotient), "".join(decimals))
    

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

### Feb 1, 2020 \[Medium\] Largest BST in a Binary Tree
---
> **Question:** You are given the root of a binary tree. Find and return the largest subtree of that tree, which is a valid binary search tree.

**Example1:**
```py
Input: 
      5
    /  \
   2    4
 /  \
1    3

Output:
   2  
 /  \
1    3
```

**Example2:**
```py
Input: 
       50
     /    \
  30       60
 /  \     /  \ 
5   20   45    70
              /  \
            65    80
            
Output: 
      60
     /  \ 
   45    70
        /  \
      65    80
```

**My thoughts:** This problem is similar to finding height of binary tree where post-order traversal is used. The idea is to gather infomation from left and right tree to determine if current node forms a valid BST or not through checking if the value fit into the range. And the infomation from children should contain if children are valid BST, the min & max of subtree and accumulated largest sub BST size.

**Solution with Recursion:** [https://repl.it/@trsong/Largest-BST-in-a-Binary-Tree](https://repl.it/@trsong/Largest-BST-in-a-Binary-Tree)
```py
import unittest

class BSTResult(object):
    def __init__(self):
        self.min_val = None
        self.max_val = None
        self.max_bst_size = 0
        self.max_bst = None
        self.is_valid_bst = True


def largest_bst_recur(root):
    if not root:
        return BSTResult()

    left_res = largest_bst_recur(root.left)
    right_res = largest_bst_recur(root.right)
    has_valid_subtrees = left_res.is_valid_bst and right_res.is_valid_bst
    is_root_valid = (not left_res.max_val or left_res.max_val <= root.val) and (not right_res.min_val or root.val <= right_res.min_val)

    result = BSTResult()
    if has_valid_subtrees and is_root_valid:
        result.min_val = root.val if left_res.min_val is None else left_res.min_val
        result.max_val = root.val if right_res.max_val is None else right_res.max_val
        result.max_bst = root
        result.max_bst_size = left_res.max_bst_size + right_res.max_bst_size + 1
    else:
        result.is_valid_bst = False
        result.max_bst = left_res.max_bst
        result.max_bst_size = left_res.max_bst_size
        if left_res.max_bst_size < right_res.max_bst_size:
            result.max_bst = right_res.max_bst
            result.max_bst_size = right_res.max_bst_size
    return result

           
def largest_bst(root):
    res = largest_bst_recur(root)
    return res.max_bst


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __eq__(self, other):
        return (other and 
         other.val == self.val and 
         other.left == self.left and 
         other.right == self.right)


class LargestBSTSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertIsNone(largest_bst(None))
    
    def test_right_heavy_tree(self):
        """
           1
            \
             10
            /  \
           11  28
        """
        n11, n28 = TreeNode(11), TreeNode(28)
        n10 = TreeNode(10, n11, n28)
        n1 = TreeNode(1, right=n10)
        result = largest_bst(n1)
        self.assertTrue(result == n11 or result == n28)

    def test_left_heavy_tree(self):
        """  
              0
             / 
            3
           /
          2
         /
        1
        """
        n1 = TreeNode(1)
        n2 = TreeNode(2, n1)
        n3 = TreeNode(3, n2)
        n0 = TreeNode(0, n3)
        self.assertEqual(n3, largest_bst(n0))

    def test_largest_BST_on_left_subtree(self):
        """ 
            0
           / \
          2   -2
         / \   \
        1   3   -1
        """
        n2 = TreeNode(2, TreeNode(1), TreeNode(3))
        n2m = TreeNode(2, right=TreeNode(-1))
        n0 = TreeNode(0, n2, n2m)
        self.assertEqual(n2, largest_bst(n0))

    def test_largest_BST_on_right_subtree(self):
        """
               50
             /    \
           30      60
          /  \    /  \ 
         5   20  45   70
                     /  \
                    65   80
        """
        n30 = TreeNode(30, TreeNode(5), TreeNode(20))
        n70 = TreeNode(70, TreeNode(65), TreeNode(80))
        n60 = TreeNode(60, TreeNode(45), n70)
        n50 = TreeNode(50, n30, n60)
        self.assertEqual(n60, largest_bst(n50))

    def test_entire_tree_is_bst(self):
        """ 
            4
           / \
          2   5
         / \   \
        1   3   6
        """
        left_tree = TreeNode(2, TreeNode(1), TreeNode(3))
        right_tree = TreeNode(5, right=TreeNode(6))
        root = TreeNode(4, left_tree, right_tree)
        self.assertEqual(root, largest_bst(root))


if __name__ == '__main__':
    unittest.main(exit=False)
```