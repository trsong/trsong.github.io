---
layout: post
title:  "Daily Coding Problems 2022 Feb to Apr"
date:   2022-02-01 02:22:32 -0700
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


### Mar 20, 2022 \[Medium\] Pre-order & In-order Binary Tree Traversal
---
> **Question:** Given pre-order and in-order traversals of a binary tree, write a function to reconstruct the tree.
>
> For example, given the following preorder traversal:

```py
[a, b, d, e, c, f, g]
```

> And the following inorder traversal:

```py
[d, b, e, a, f, c, g]
```

> You should return the following tree:

```py
    a
   / \
  b   c
 / \ / \
d  e f  g
```

### Mar 19, 2022 \[Medium\] In-order & Post-order Binary Tree Traversal
---
> **Question:** Given Postorder and Inorder traversals, construct the tree.

**Examples 1:**
```py
Input: 
in_order = [2, 1, 3]
post_order = [2, 3, 1]

Output: 
      1
    /   \
   2     3 
```

**Example 2:**
```py
Input: 
in_order = [4, 8, 2, 5, 1, 6, 3, 7]
post_order = [8, 4, 5, 2, 6, 7, 3, 1]

Output:
          1
       /     \
     2        3
   /    \   /   \
  4     5   6    7
    \
      8
```

### Mar 18, 2022 \[Medium\] Construct BST from Post-order Traversal
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

**Solution:** [https://replit.com/@trsong/Construct-Binary-Search-Tree-from-Post-order-Traversal-2](https://replit.com/@trsong/Construct-Binary-Search-Tree-from-Post-order-Traversal-2)
```py
import unittest

def construct_bst(post_order_traversal):
    return construct_bst_recur(post_order_traversal)


def construct_bst_recur(stack, lo=float('-inf'), hi=float('inf')):
    if not (stack and lo <= stack[-1] <= hi):
        return None

    cur_val = stack.pop()
    right_child = construct_bst_recur(stack, cur_val, hi)
    left_child = construct_bst_recur(stack, lo, cur_val)
    return Node(cur_val, left_child, right_child)
    

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
    unittest.main(exit=False, verbosity=2)
```

### Mar 17, 2022 LC 981 \[Medium\] Time Based Key-Value Store
---
> **Question:** Write a map implementation with a get function that lets you retrieve the value of a key at a particular time.
>
> It should contain the following methods:
>
> - `set(key, value, time)`: sets key to value for t = time.
> - `get(key, time)`: gets the key at t = time.
>
> The map should work like this. If we set a key at a particular time, it will maintain that value forever or until it gets set at a later time. In other words, when we get a key at a time, it should return the value that was set for that key set at the most recent time.

**Solution with BST:** [https://replit.com/@trsong/Time-Based-Key-Value-Store-2](https://replit.com/@trsong/Time-Based-Key-Value-Store-2)
```py
import unittest
from collections import defaultdict

class TimeMap(object):
    def __init__(self):
        self.lookup = defaultdict(BST)
    
    def set(self, key, value, time):
        self.lookup[key].set(time, value)
        
    def get(self, key, time):
        return self.lookup[key].get(time)


class BSTNode(object):
    def __init__(self, key, val, left=None, right=None):
        self.key = key
        self.val = val
        self.left = left
        self.right = right

    
class BST(object):
    def __init__(self):
        self.root = None

    def get(self, key):
        p = self.root
        res = None
        while p:
            if p.key == key:
                return p.val
            elif p.key < key:
                res = p.val
                p = p.right
            else:
                p = p.left
        return res

    def set(self, key, val):
        # Omit tree re-balancing
        node = BSTNode(key, val)
        if self.root is None:
            self.root = node

        p = self.root
        while True:
            if p.key == key:
                p.val = val
                break
            elif p.key < key:
                if p.right is None:
                    p.right = node
                    break
                p = p.right
            else:
                if p.left is None:
                    p.left = node
                    break
                p = p.left


class TimeMapSpec(unittest.TestCase):
    def test_example(self):
        d = TimeMap()
        d.set(1, 1, time=0)
        d.set(1, 2, time=2)
        self.assertEqual(1, d.get(1, time=1)) 
        self.assertEqual(2, d.get(1, time=3))
    
    def test_example2(self):
        d = TimeMap()
        d.set(1, 1, time=5)
        self.assertIsNone(d.get(1, time=0))
        self.assertEqual(1, d.get(1, time=10))
    
    def test_example3(self):
        d = TimeMap()
        d.set(1, 1, time=0)
        d.set(1, 2, time=0)
        self.assertEqual(2, d.get(1, time=0))

    def test_set_then_get(self):
        d = TimeMap()
        d.set(1, 100, time=10)
        d.set(1, 99, time=20)
        self.assertIsNone(d.get(1, time=5))
        self.assertEqual(100, d.get(1, time=10))
        self.assertEqual(100, d.get(1, time=15))
        self.assertEqual(99, d.get(1, time=20))
        self.assertEqual(99, d.get(1, time=25))

    def test_get_no_exist_key(self):
        d = TimeMap()
        self.assertIsNone(d.get(1, time=0))
        d.set(1, 100, time=0)
        self.assertIsNone(d.get(42, time=0))
        self.assertEqual(100, d.get(1, time=0))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Mar 16, 2022 \[Medium\] Paint House
---
> **Question:** A builder is looking to build a row of N houses that can be of K different colors. He has a goal of minimizing cost while ensuring that no two neighboring houses are of the same color.
>
> Given an N by K matrix where the n-th row and k-th column represents the cost to build the n-th house with k-th color, return the minimum cost which achieves this goal.


**Solution with DP:** [https://replit.com/@trsong/Min-Cost-to-Paint-House-2](https://replit.com/@trsong/Min-Cost-to-Paint-House-2)
```py
import unittest

def min_paint_houses_cost(paint_cost):
    if not paint_cost or not paint_cost[0]:
        return 0

    num_house, num_color = len(paint_cost), len(paint_cost[0])
    # Let dp[n][c] represents min cost for n houses and k colors
    # dp[n][c] = paint_cost[n-1][c] + min{ dp[n-1][k] } for all k != c
    dp = [[float('inf') for _ in range(num_color)] for _ in range(num_house + 1)]
    for c in range(num_color):
        dp[0][c] = 0

    for i in range(1, num_house + 1):
        min1 = min2 = float('inf')

        for prev_accu in dp[i-1]:
            if prev_accu < min2:
                min2 = prev_accu

            if min2 < min1:
                min1, min2 = min2, min1

        for c in range(num_color):
            prev_min = min2 if dp[i-1][c] == min1 else min1
            dp[i][c] = paint_cost[i-1][c] + prev_min

    return min(dp[num_house])


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
    unittest.main(exit=False, verbosity=2)
```

### Mar 15, 2022 \[Medium\] Tokenization
---
> **Questions:** Given a dictionary of words and a string made up of those words (no spaces), return the original sentence in a list. If there is more than one possible reconstruction, return any of them. If there is no possible reconstruction, then return null.

**Example 1:**
```py
Input: ['quick', 'brown', 'the', 'fox'], 'thequickbrownfox'
Output: ['the', 'quick', 'brown', 'fox']
```

**Example 2:**
```py
Input: ['bed', 'bath', 'bedbath', 'and', 'beyond'], 'bedbathandbeyond'
Output:  Either ['bed', 'bath', 'and', 'beyond'] or ['bedbath', 'and', 'beyond']
```

**Solution with Backtracking:** [https://replit.com/@trsong/String-Tokenization-3](https://replit.com/@trsong/String-Tokenization-3)
```py
import unittest

def tokenize(dictionary, sentence):
    word_set = set(dictionary)
    return tokenize_recur(sentence, word_set)


def tokenize_recur(sentence, word_set):
    if not sentence:
        return []

    n = len(sentence)
    for token_size in map(len, word_set):
        if token_size > n or sentence[:token_size] not in word_set:
            continue

        sub_res = tokenize_recur(sentence[token_size:], word_set)
        if sub_res is not None:
            return [sentence[:token_size]] + sub_res
    return None
    

class TokenizeSpec(unittest.TestCase):
    def test_example(self):
        dictionary = ['quick', 'brown', 'the', 'fox']
        sentence = 'thequickbrownfox'
        expected = ['the', 'quick', 'brown', 'fox']
        self.assertEqual(expected, tokenize(dictionary, sentence))

    def test_example2(self):
        dictionary = ['bed', 'bath', 'bedbath', 'and', 'beyond']
        sentence = 'bedbathandbeyond'
        expected1 = ['bed', 'bath', 'and', 'beyond']
        expected2 = ['bedbath', 'and', 'beyond']
        res = tokenize(dictionary, sentence)
        self.assertIn(res, [expected1, expected2])

    def test_match_entire_sentence(self):
        dictionary = ['thequickbrownfox']
        sentence = 'thequickbrownfox'
        expected = ['thequickbrownfox']
        self.assertEqual(expected, tokenize(dictionary, sentence))

    def test_cannot_tokenize(self):
        dictionary = ['thequickbrownfox']
        sentence = 'thefox'
        self.assertIsNone(tokenize(dictionary, sentence))

    def test_longer_sentence(self):
        dictionary = ['i', 'and', 'like', 'sam', 'sung', 'samsung', 'mobile', 'ice', 'cream', 'icecream', 'man', 'go', 'mango']
        sentence = 'ilikesamsungmobile'
        expected1 = ['i', 'like', 'samsung', 'mobile']
        expected2 = ['i', 'like', 'sam', 'sung', 'mobile']
        res = tokenize(dictionary, sentence)
        self.assertIn(res, [expected1, expected2])

    def test_longer_sentence2(self):
        dictionary = ['i', 'and', 'like', 'sam', 'sung', 'samsung', 'mobile', 'ice', 'cream', 'icecream', 'go', 'mango']
        sentence = 'ilikeicecreamandmango'
        expected1 = ['i', 'like', 'icecream', 'and', 'mango']
        expected2 = ['i', 'like', 'ice', 'cream', 'and', 'mango']
        res = tokenize(dictionary, sentence)
        self.assertIn(res, [expected1, expected2])

    def test_greedy_approach_will_fail(self):
        dictionary = ['ice', 'icecream', 'coffee']
        sentence = 'icecream'
        expected = ['icecream']
        self.assertEqual(expected, tokenize(dictionary, sentence))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Mar 14, 2022 LC 392 \[Medium\] Is Subsequence
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

**Solution:** [https://replit.com/@trsong/Check-if-Subsequence-2](https://replit.com/@trsong/Check-if-Subsequence-2)
```py
import unittest

def is_subsequence(s, t):
    if not s:
        return True
        
    i = 0
    for ch in t:
        if i >= len(s):
            return True

        if s[i] == ch:
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

### Mar 13, 2022 LC 253 \[Easy\] Minimum Lecture Rooms
---
> **Questions:** Given an array of time intervals `(start, end)` for classroom lectures (possibly overlapping), find the minimum number of rooms required.
>
> For example, given `[(30, 75), (0, 50), (60, 150)]`, you should return `2`.

**My thoughts:** whenever we enter an interval at time t, the total number of room at time t increment by 1 and whenever we leave an interval the total number of required room decrement by 1. For `(1, 10)`, `(5, 15)` and `(6, 15)`, `+1` at `t = 1, 5, 6` and `-1` at `t=10, 15, 15`. And at peak hour, the total room equals 3. 

**Solution:** [https://replit.com/@trsong/Minimum-Required-Lecture-Rooms-2](https://replit.com/@trsong/Minimum-Required-Lecture-Rooms-2)
```py
import unittest

def min_lecture_rooms(intervals):
    starts = map(lambda x: (x[0], 1), intervals)
    ends = map(lambda x: (x[1], -1), intervals)
    all_times = sorted(starts + ends)

    max_rooms = 0
    cur_rooms = 0
    for _, diff in all_times:
        cur_rooms += diff
        max_rooms = max(max_rooms, cur_rooms)
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
    unittest.main(exit=False, verbosity=2)
```

### Mar 12, 2022 \[Hard\] Reverse Words Keep Delimiters
---
> **Question:** Given a string and a set of delimiters, reverse the words in the string while maintaining the relative order of the delimiters. For example, given "hello/world:here", return "here/world:hello"
>
> Follow-up: Does your solution work for the following cases: "hello/world:here/", "hello//world:here"

**Solution:** [https://replit.com/@trsong/Reverse-Words-and-Keep-Delimiters-3](https://replit.com/@trsong/Reverse-Words-and-Keep-Delimiters-3)
```py
import unittest

def reverse_words_and_keep_delimiters(s, delimiters):
    d_set = set(delimiters)
    tokens = tokenize(s, d_set)

    lo = 0
    hi = len(tokens) - 1
    while lo < hi:
        if tokens[lo] in d_set:
            lo += 1
        elif tokens[hi] in d_set:
            hi -= 1
        else:
            tokens[lo], tokens[hi] = tokens[hi], tokens[lo]
            lo += 1
            hi -= 1
    return ''.join(tokens)


def tokenize(s, delimiters):
    prev_dpos = -1
    res = []

    for i, ch in enumerate(s):
        if ch not in delimiters:
            continue
        res.append(s[prev_dpos + 1: i])
        res.append(ch)
        prev_dpos = i
    res.append(s[prev_dpos + 1:])
    return filter(len, res)
            

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

### Mar 11, 2022  LC 239 \[Medium\] Sliding Window Maximum
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

**Solution with Sliding Window:** [https://replit.com/@trsong/Find-Sliding-Window-Maximum-4](https://replit.com/@trsong/Find-Sliding-Window-Maximum-4)
```py
 import unittest
from queue import deque

def max_sliding_window(nums, k):
    dq = deque()
    res = []

    for i, num in enumerate(nums):
        if i >= k and dq[0] <= i - k:
            dq.popleft()

        while dq and nums[dq[-1]] <= num:
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

### Mar 10, 2022  LC 821 \[Medium\] Shortest Distance to Character
---
> **Question:**  Given a string s and a character c, find the distance for all characters in the string to the character c in the string s. 
>
> You can assume that the character c will appear at least once in the string.

**Example:**
```py
shortest_dist('helloworld', 'l') 
# returns [2, 1, 0, 0, 1, 2, 2, 1, 0, 1]
```

**My thoughts:** The idea is similar to Problem ["LC 42 Trap Rain Water"](https://trsong.github.io/python/java/2019/05/01/DailyQuestions.html#may-11-2019-lc-42-hard-trapping-rain-water): we can simply scan from left to know the shortest distance from nearest character on the left and vice versa when we can from right to left. 

**Solution:** [https://replit.com/@trsong/Find-Shortest-Distance-to-Characters-2](https://replit.com/@trsong/Find-Shortest-Distance-to-Characters-2)
```py
import unittest

def shortest_dist_to_char(s, ch):
    n = len(s)
    res = [n] * n
    left_dist = n
    right_dist = n

    for i in range(n):
        if s[i] == ch:
            left_dist = 0
        res[i] = min(res[i], left_dist)
        left_dist += 1

        if s[n - 1 - i] == ch:
            right_dist = 0
        res[n - 1 - i] = min(res[n - 1 - i], right_dist)
        right_dist += 1
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
    unittest.main(exit=False, verbosity=2)
```

### Mar 9, 2022 \[Medium\] Shortest Unique Prefix
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


**Solution with Trie:** [https://replit.com/@trsong/Find-All-Shortest-Unique-Prefix-2](https://replit.com/@trsong/Find-All-Shortest-Unique-Prefix-2)
```py
import unittest

class Trie(object):
    def __init__(self):
        self.count = 0
        self.children = {}

    def insert(self, word):
        p = self
        for ch in word:
            p.children[ch] = p.children.get(ch, Trie())
            p = p.children[ch]
            p.count += 1
        
    def find_prefix(self, word):
        p = self
        depth = 0
        for ch in word:
            if p.count == 1:
                break
            p = p.children[ch]
            depth += 1
        return word[:depth]


def shortest_unique_prefix(words):
    trie = Trie()
    for ch in words:
        trie.insert(ch)
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

### Mar 8, 2022 \[Medium\] Friend Cycle Problem
--- 
> **Question:** A classroom consists of `N` students, whose friendships can be represented in an adjacency list. For example, the following descibes a situation where `0` is friends with `1` and `2`, `3` is friends with `6`, and so on.

```py
{0: [1, 2],
 1: [0, 5],
 2: [0],
 3: [6],
 4: [],
 5: [1],
 6: [3]} 
```
> Each student can be placed in a friend group, which can be defined as the transitive closure of that student's friendship relations. In other words, this is the smallest set such that no student in the group has any friends outside this group. For the example above, the friend groups would be `{0, 1, 2, 5}`, `{3, 6}`, `{4}`.
>
> Given a friendship list such as the one above, determine the number of friend groups in the class.

**Solution with DFS:** [https://replit.com/@trsong/Friend-Cycle-Problem-2](https://replit.com/@trsong/Friend-Cycle-Problem-2)
```py
import unittest

def find_cycles(friendships):
    visited = set()
    res = 0

    for cur in friendships:
        if cur in visited:
            continue
        res += 1

        stack = [cur]
        while stack:
            p = stack.pop()
            if p in visited:
                continue
            visited.add(p)

            for nb in friendships[p]:
                if nb not in visited:
                    stack.append(nb)
    return res
                
                
class FindCycleSpec(unittest.TestCase):
    def test_example(self):
        friendships = {
            0: [1, 2],
            1: [0, 5],
            2: [0],
            3: [6],
            4: [],
            5: [1],
            6: [3]
        }
        expected = 3  # [0, 1, 2, 5], [3, 6], [4]
        self.assertEqual(expected, find_cycles(friendships))

    def test_no_friends(self):
        friendships = {
            0: [],
            1: [],
            2: []
        }
        expected = 3  # [0], [1], [2]
        self.assertEqual(expected, find_cycles(friendships))

    def test_all_friends(self):
        friendships = {
            0: [1, 2],
            1: [0, 2],
            2: [0, 1]
        }
        expected = 1  # [0, 1, 2]
        self.assertEqual(expected, find_cycles(friendships))
    
    def test_common_friend(self):
        friendships = {
            0: [1],
            1: [0, 2, 3],
            2: [1],
            3: [1],
            4: []
        }
        expected = 2  # [0, 1, 2, 3], [4]
        self.assertEqual(expected, find_cycles(friendships))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 7, 2022 \[Easy\] Zombie in Matrix
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

**Solution with BFS:** [https://replit.com/@trsong/Zombie-Infection-in-Matrix-2](https://replit.com/@trsong/Zombie-Infection-in-Matrix-2)
 ```py
 import unittest

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def zombie_infection_time(grid):
    if not grid or not grid[0]:
        return -1

    n, m = len(grid), len(grid[0])
    queue = [(r, c) for r in range(n) 
                    for c in range(m) 
                    if grid[r][c]]
    round = -1
    while queue:
        for _ in range(len(queue)):
            cur_r, cur_c = queue.pop(0)
            if round > 0 and grid[cur_r][cur_c]:
                continue
            grid[cur_r][cur_c] = 1

            for dr, dc in DIRECTIONS:
                new_r, new_c = cur_r + dr, cur_c + dc
                if (0 <= new_r < n and 
                    0 <= new_c < m and 
                    grid[new_r][new_c] == 0):
                    queue.append((new_r, new_c))
        round += 1
    return round
    

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

### Mar 6, 2022 \[Medium\] Running Median of a Number Stream
---
> **Question:** Compute the running median of a sequence of numbers. That is, given a stream of numbers, print out the median of the list so far on each new element.
> 
> Recall that the median of an even-numbered list is the average of the two middle numbers.
> 
> For example, given the sequence `[2, 1, 5, 7, 2, 0, 5]`, your algorithm should print out:

```py
2
1.5
2
3.5
2
2
2
```

**Solution with PriorityQueue:** [https://replit.com/@trsong/Running-Median-of-a-Number-Stream-2](https://replit.com/@trsong/Running-Median-of-a-Number-Stream-2)
```py
import unittest
from queue import PriorityQueue

def generate_running_median(num_stream):
    # max_heap for lower half, min_heap for upper half
    max_heap = PriorityQueue()
    min_heap = PriorityQueue()
    res = []

    for num in num_stream:
        min_heap.put(num)
        if min_heap.qsize() - max_heap.qsize() > 1:
            max_heap.put(-min_heap.get())

        if min_heap.qsize() == max_heap.qsize():
            res.append((-max_heap.queue[0] + min_heap.queue[0]) / 2)
        else:
            res.append(min_heap.queue[0])
    return res


class GenerateRunningMedian(unittest.TestCase):
    def test_example(self):
        num_stream = iter([2, 1, 5, 7, 2, 0, 5])
        expected = [2, 1.5, 2, 3.5, 2, 2, 2]
        self.assertEqual(expected, list(generate_running_median(num_stream)))

    def test_empty_stream(self):
        self.assertEqual([], list(generate_running_median(iter([]))))

    def test_unique_value(self):
        num_stream = iter([1, 1, 1, 1, 1])
        expected = [1, 1, 1, 1, 1]
        self.assertEqual(expected, list(generate_running_median(num_stream)))

    def test_contains_zero(self):
        num_stream = iter([0, 1, 1, 0, 0])
        expected = [0, 0.5, 1, 0.5, 0]
        self.assertEqual(expected, list(generate_running_median(num_stream)))

    def test_contains_zero2(self):
        num_stream = iter([2, 0, 1])
        expected = [2, 1, 1]
        self.assertEqual(expected, list(generate_running_median(num_stream)))

    def test_even_iteration_gives_average(self):
        num_stream = iter([3, 0, 1, 2])
        expected = [3, 1.5, 1, 1.5]
        self.assertEqual(expected, list(generate_running_median(num_stream)))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 5, 2022 \[Easy\] Run-length String Encode and Decode
---
> **Question:** Run-length encoding is a fast and simple method of encoding strings. The basic idea is to represent repeated successive characters as a single count and character. For example, the string `"AAAABBBCCDAA"` would be encoded as `"4A3B2C1D2A"`.
>
> Implement run-length encoding and decoding. You can assume the string to be encoded have no digits and consists solely of alphabetic characters. You can assume the string to be decoded is valid.


**Solution:** [https://replit.com/@trsong/Run-length-String-Encode-and-Decode-2](https://replit.com/@trsong/Run-length-String-Encode-and-Decode-2)
```py
import unittest
from collections import Counter

class RunLengthProcessor(object):
    @staticmethod
    def encode(s):
        if not s:
            return ""

        prev = s[0]
        count = 0
        res = []

        for ch in s:
            if prev == ch:
                count += 1
            else:
                res.append(str(count))
                res.append(prev)
                prev = ch
                count = 1
                
        res.append(str(count))
        res.append(prev)
        return ''.join(res)
                
    @staticmethod
    def decode(s):
        if not s:
            return ""
            
        res = []
        count = 0

        for ch in s:
            if '0' <= ch <= '9':
                count = 10 * count + int(ch)
            else:
                res.append(ch * count)
                count = 0
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
    unittest.main(exit=False, verbosity=2)
```

### Mar 4, 2022 \[Easy\] Remove Duplicates From Sorted Linked List
---
> **Question:** Given a sorted linked list, remove all duplicate values from the linked list.

**Example 1:**
```py
Input: 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> 4 -> 4 -> 4 -> 5 -> 5 -> 6 -> 7 -> 9
Output: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 9
```

**Example 2:**
```py
Input: 1 -> 1 -> 1 -> 1
Output: 1
```

**Solution:** [https://replit.com/@trsong/Remove-Duplicates-From-Sorted-Linked-List-2](https://replit.com/@trsong/Remove-Duplicates-From-Sorted-Linked-List-2)
```py
import unittest

def remove_duplicates(lst):
    prev = dummy = ListNode(None, lst)
    p = lst
    while p:
        if p.val == prev.val:
            prev.next = p.next
        else:
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

    def __repr__(self):
        return "%s -> %s" % (self.val, self.next)

    def __eq__(self, other):
        return other and self.val == other.val and self.next == other.next

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
        source_list = ListNode.List(1, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 6, 7, 9)
        expected = ListNode.List(1, 2, 3, 4, 5, 6, 7, 9)
        self.assertEqual(expected, remove_duplicates(source_list))
        
    def test_example2(self):
        source_list = ListNode.List(1, 1, 1, 1)
        expected = ListNode.List(1)
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


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 3, 2022 \[Easy\] Generate All Possible Subsequences
---
> **Question:** Given a string, generate all possible subsequences of the string.
>
> For example, given the string `xyz`, return an array or set with the following strings:

```py
x
y
z
xy
xz
yz
xyz
```

**Solution:** [https://replit.com/@trsong/Generate-All-Possible-Subsequences-2](https://replit.com/@trsong/Generate-All-Possible-Subsequences-2)
```py
import unittest

def generate_subsequences(s):
    if not s:
        return ['']
    res = []
    generate_subsequences_recur(res, '', s, 0)
    return res


def generate_subsequences_recur(res, accu, s, current_index):
    if accu:
        res.append(accu)

    for i in range(current_index, len(s)):
        generate_subsequences_recur(res, accu + s[i], s, i + 1)


class GenerateSubsequenceSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(set(expected), set(result))

    def test_example(self):
        s = 'xyz'
        expected = ['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz']
        self.assert_result(expected, generate_subsequences(s))

    def test_empty_string(self):
        s = ''
        expected = ['']
        self.assert_result(expected, generate_subsequences(s))

    def test_binary_string(self):
        s = '01'
        expected = ['0', '1', '01']
        self.assert_result(expected, generate_subsequences(s))

    def test_length_four_string(self):
        s = '0123'
        expected = [
            '0', '1', '2', '3', '01', '02', '03', '12', '13', '23', '123',
            '023', '013', '012', '0123'
        ]
        self.assert_result(expected, generate_subsequences(s))

    def test_duplicated_characters(self):
        s = 'aaa'
        expected = ['a', 'aa', 'aaa']
        self.assert_result(expected, generate_subsequences(s))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

> Note that `zx` is not a valid subsequence since it is not in the order of the given string.

### Mar 2, 2022 \[Hard\] Largest Sum of Non-adjacent Numbers
---
> **Question:** Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0 or negative.
>
> For example, `[2, 4, 6, 2, 5]` should return 13, since we pick 2, 6, and 5. `[5, 1, 1, 5]` should return 10, since we pick 5 and 5.
>
> Follow-up: Can you do this in O(N) time and constant space?


**Solution with DP:** [https://replit.com/@trsong/Find-Largest-Sum-of-Non-adjacent-Numbers-3](https://replit.com/@trsong/Find-Largest-Sum-of-Non-adjacent-Numbers-3)
```py
import unittest

def max_non_adj_sum(nums):
    # let dp[n] represents max non adjacent sum for subarray nums[:n]
    # dp[n] = max(dp[n-2] + nums[n-1], dp[n-1])
    pre_max = pre_pre_max = 0
    for num in nums:
        cur_max = max(pre_pre_max + num, pre_max)
        pre_pre_max, pre_max = pre_max, cur_max
    return pre_max


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

### Mar 1, 2022 \[Medium\] Longest Consecutive Sequence in an Unsorted Array
---
> **Question:** Given an array of integers, return the largest range, inclusive, of integers that are all included in the array.
>
> For example, given the array `[9, 6, 1, 3, 8, 10, 12, 11]`, return `(8, 12)` since `8, 9, 10, 11, and 12` are all in the array.

**Solution:** [https://replit.com/@trsong/Longest-Consecutive-Sequence-4](https://replit.com/@trsong/Longest-Consecutive-Sequence-4)
```py
import unittest

def longest_consecutive_seq(nums):        
    num_set = set(nums)
    res = None
    max_window = 0

    for num in nums:
        if num - 1 in num_set:
            continue

        window_size = 0
        while num + window_size in num_set:
            window_size += 1

        if window_size > max_window:
            res = (num, num + window_size - 1)
            max_window = window_size
    return res


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

### Feb 28, 2022 \[Medium\] Tree Serialization
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

**Solution:** [https://replit.com/@trsong/Serialize-and-Deserialize-the-Binary-Tree-4](https://replit.com/@trsong/Serialize-and-Deserialize-the-Binary-Tree-4)
```py
import unittest

class BinaryTreeSerializer(object):
    @staticmethod
    def serialize(root):
        stack = [root]
        res = []
        while stack:
            cur = stack.pop()
            if cur is None:
                res.append('#')
                continue
            
            res.append(str(cur.val))
            stack.append(cur.right)
            stack.append(cur.left)
        return ' '.join(res)

    @staticmethod
    def deserialize(s):
        tokens = iter(s.split())
        return BinaryTreeSerializer.construct_tree(tokens)

    @staticmethod
    def construct_tree(token_stream):
        raw_val = next(token_stream, None)
        if raw_val == '#':
            return None
            
        left_child = BinaryTreeSerializer.construct_tree(token_stream)
        right_child = BinaryTreeSerializer.construct_tree(token_stream)
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

### Feb 27, 2022 \[Medium\] Matrix Rotation
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

**Solution:** [https://replit.com/@trsong/Matrix-Rotation-In-place-with-flip-2](https://replit.com/@trsong/Matrix-Rotation-In-place-with-flip-2)
```py
import unittest

def matrix_rotation(matrix):
    if not matrix or not matrix[0]:
        return matrix

    n = len(matrix)
    # flip diagonaly
    for r in range(n):
        for c in range(r + 1, n):
            matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]
            
    # flip vertically
    for r in range(n):
        for c in range(n // 2):
            matrix[r][c], matrix[r][n - 1 - c] = matrix[r][n - 1 - c], matrix[r][c]
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
    unittest.main(exit=False, verbosity=2)
```

### Feb 26, 2022 \[Easy\] Symmetric K-ary Tree
---
> **Question:** Given a k-ary tree, figure out if the tree is symmetrical.
> 
> A k-ary tree is a tree with k-children, and a tree is symmetrical if the data of the left side of the tree is the same as the right side of the tree. 
>
> Here's an example of a symmetrical k-ary tree.

```py
        4
     /     \
    3        3
  / | \    / | \
9   4  1  1  4  9
```

**Solution with DFS:** [https://replit.com/@trsong/Is-Symmetric-K-ary-Tree-3](https://replit.com/@trsong/Is-Symmetric-K-ary-Tree-3)
```py
import unittest

def is_symmetric(root):
    if not root:
        return True

    dfs_order = dfs_traversal(root)
    reverse_dfs_order = dfs_traversal(root, reverse=True)
    for n1, n2 in zip(dfs_order, reverse_dfs_order):
        num_child1 = len(n1.children) if n1.children else 0
        num_child2 = len(n2.children) if n2.children else 0
        if n1.val != n2.val or num_child1 != num_child2:
            return False
    return True


def dfs_traversal(root, reverse=False):
    stack = [root]
    while stack:
        cur = stack.pop()
        yield cur
        if cur.children:
            children = reversed(cur.children) if reverse else cur.children
            stack.extend(children)


class TreeNode(object):
    def __init__(self, val, children=None):
        self.val = val
        self.children = children


class IsSymmetricSpec(unittest.TestCase):
    def test_example(self):
        """
                4
             /     \
            3        3
          / | \    / | \
        9   4  1  1  4  9
        """
        left_tree = TreeNode(3, [TreeNode(9), TreeNode(4), TreeNode(1)])
        right_tree = TreeNode(3, [TreeNode(1), TreeNode(4), TreeNode(9)])
        root = TreeNode(4, [left_tree, right_tree])
        self.assertTrue(is_symmetric(root))

    def test_empty_tree(self):
        self.assertTrue(is_symmetric(None))

    def test_node_with_odd_number_of_children(self):
        """
                8
            /   |   \
          4     5     4
         / \   / \   / \
        1   2 3   3 2   1
        """
        left_tree = TreeNode(4, [TreeNode(1), TreeNode(2)])
        mid_tree = TreeNode(5, [TreeNode(3), TreeNode(3)])
        right_tree= TreeNode(4, [TreeNode(2), TreeNode(1)])
        root = TreeNode(8, [left_tree, mid_tree, right_tree])
        self.assertTrue(is_symmetric(root))

    def test_binary_tree(self):
        """
             6
           /   \
          4     4 
         / \   / \
        1   2 2   1
         \       / 
          3     3 
        """
        left_tree = TreeNode(4, [TreeNode(1, [TreeNode(3)]), TreeNode(2)])
        right_tree = TreeNode(4, [TreeNode(2), TreeNode(1, [TreeNode(3)])])
        root = TreeNode(6, [left_tree, right_tree])
        self.assertTrue(is_symmetric(root))

    def test_unsymmetric_tree(self):
        """
             6
           / | \
          4  5  4 
         /  /  / \
        1  2  2   1
        """
        left_tree = TreeNode(4, [TreeNode(1)])
        mid_tree = TreeNode(5, [TreeNode(2)])
        right_tree = TreeNode(4, [TreeNode(2), TreeNode(1)])
        root = TreeNode(6, [left_tree, mid_tree, right_tree])
        self.assertFalse(is_symmetric(root))

    def test_unsymmetric_tree2(self):
        """
             6
           / | \
          4  5  4 
           / | \
          2  2  1
        """
        left_tree = TreeNode(4)
        mid_tree = TreeNode(5, [TreeNode(2), TreeNode(2), TreeNode(1)])
        right_tree = TreeNode(4)
        root = TreeNode(6, [left_tree, mid_tree, right_tree])
        self.assertFalse(is_symmetric(root))

    def test_unsymmetric_tree3(self):
        """
              6
           / | | \
          4  5 5  4 
          |  | |  |
          2  2 2  3
        """
        left_tree = TreeNode(4, [TreeNode(2)])
        mid_left_tree = TreeNode(5, [TreeNode(2)])
        mid_right_tree = TreeNode(5, [TreeNode(2)])
        right_tree = TreeNode(4, [TreeNode(3)])
        root = TreeNode(6, [left_tree, mid_left_tree, mid_right_tree, right_tree])
        self.assertFalse(is_symmetric(root))

    def test_unsymmetric_tree4(self):
        """
              1
            /    \
           2      2
          / \   / | \
         4   5 6  5  4
             |
             6
        """
        left_tree = TreeNode(2, [TreeNode(4), TreeNode(5, [TreeNode(6)])])
        right_tree = TreeNode(2, [TreeNode(6), TreeNode(5), TreeNode(4)])
        root = TreeNode(1, [left_tree, right_tree])
        self.assertFalse(is_symmetric(root))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 25, 2022  \[Hard\] LFU Cache
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


**Solution:** [https://replit.com/@trsong/Design-LFU-Cache-2](https://replit.com/@trsong/Design-LFU-Cache-2)
```py
import unittest
from collections import defaultdict, deque

class LFUCache(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.min_freq = 1
        self.kvf_map = {}
        self.freq_order_map = defaultdict(deque)

    def get(self, key):
        if key not in self.kvf_map:
            return None

        val, freq = self.kvf_map[key]
        self.kvf_map[key] = (val, freq + 1)
        self.freq_order_map[freq].remove(key)
        self.freq_order_map[freq + 1].append(key)
        
        if not self.freq_order_map[freq]:
            del self.freq_order_map[freq]
            if self.min_freq == freq:
                self.min_freq = freq + 1
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

### Feb 24, 2022 \[Medium\] Find Missing Positive
---
> **Question:** Given an unsorted integer array, find the first missing positive integer.

**Example 1:**
```py
Input: [1, 2, 0]
Output: 3
```

**Example 2:**
```py
Input: [3, 4, -1, 1]
Output: 2
```

**My thougths:** Ideally each positive number should map to the same index as its `value - 1`. So all we need to do is for each postion, use its value as index and swap with that element until we find correct number. Keep doing this and each postive should store in postion of its `value - 1`.  Now we just scan through the entire array until find the first missing number by checking each element's value against index. 


**Solution:** [https://replit.com/@trsong/Find-First-Missing-Positive-3](https://replit.com/@trsong/Find-First-Missing-Positive-3)
```py
import unittest

def find_missing_positive(nums):
    n = len(nums)
    val_to_index = lambda v: v - 1
    
    for i in range(n):
        while val_to_index(nums[i]) != i and 1 <= nums[i] <= n:
            target_index = val_to_index(nums[i])
            if target_index == val_to_index(nums[target_index]):
                break
            nums[i], nums[target_index] = nums[target_index], nums[i]

    for i in range(n):
        if val_to_index(nums[i]) != i:
            return i + 1

    return n + 1
    

class FindMissingPositiveSpec(unittest.TestCase):
    def test_example(self):
        nums = [1, 2, 0]
        expected = 3
        self.assertEqual(expected, find_missing_positive(nums))

    def test_example2(self):
        nums = [3, 4, -1, 1]
        expected = 2
        self.assertEqual(expected, find_missing_positive(nums))
    
    def test_empty_array(self):
        nums = []
        expected = 1
        self.assertEqual(expected, find_missing_positive(nums))

    def test_all_non_positives(self):
        nums = [-1, 0, -1, -2, -1, -3, -4]
        expected = 1
        self.assertEqual(expected, find_missing_positive(nums))

    def test_number_out_of_range(self):
        nums = [101, 102, 103]
        expected = 1
        self.assertEqual(expected, find_missing_positive(nums))

    def test_duplicated_numbers(self):
        nums = [1, 1, 3, 3, 2, 2, 5]
        expected = 4
        self.assertEqual(expected, find_missing_positive(nums))

    def test_missing_positive_falls_out_of_range(self):
        nums = [5, 4, 3, 2, 1]
        expected = 6
        self.assertEqual(expected, find_missing_positive(nums))

    def test_number_off_by_one_position(self):
        nums = [0, 2, 3, 4, 7, 6, 1]
        expected = 5
        self.assertEqual(expected, find_missing_positive(nums))

    def test_positive_and_negative_numbers(self):
        nums = [-1, -3, -2, 0, 1, 2, 4, -4, 5, -6, 7]
        expected = 3
        self.assertEqual(expected, find_missing_positive(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 23, 2022 \[Easy\] First and Last Indices of an Element in a Sorted Array
---
> **Question:** Given a sorted array, A, with possibly duplicated elements, find the indices of the first and last occurrences of a target element, x. Return -1 if the target is not found.

**Examples:**
```py
Input: A = [1, 3, 3, 5, 7, 8, 9, 9, 9, 15], target = 9
Output: [6, 8]

Input: A = [100, 150, 150, 153], target = 150
Output: [1, 2]

Input: A = [1, 2, 3, 4, 5, 6, 10], target = 9
Output: [-1, -1]
```


**Solution with Binary Search:** [https://repl.it/@trsong/Get-First-and-Last-Indices-of-an-Element-in-a-Sorted-Array](https://repl.it/@trsong/Get-First-and-Last-Indices-of-an-Element-in-a-Sorted-Array)
```py
import unittest

def search_range(nums, target):
    if not nums or target < nums[0] or nums[-1] < target:
        return -1, -1

    left_index = binary_search(nums, target)
    if nums[left_index] != target:
        return -1, -1

    right_index = binary_search(nums, target, exclusive=True) - 1
    return left_index, right_index


def binary_search(nums, target, exclusive=False):
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
```

**Solution with Divide And Conquer:** [https://replit.com/@trsong/Find-First-and-Last-Indices-of-an-Element-in-a-Sorted-Arr-2](https://replit.com/@trsong/Find-First-and-Last-Indices-of-an-Element-in-a-Sorted-Arr-2)
```py
import unittest

def search_range(nums, target):
    return search_range_recur(nums, target, 0, len(nums) - 1) or (-1, -1)


def search_range_recur(nums, target, lo, hi):
    if lo > hi or nums[hi] < target or nums[lo] > target:
        return None
    
    if nums[lo] == nums[hi] == target:
        return (lo, hi)

    mid = lo + (hi - lo) // 2
    left_res = search_range_recur(nums, target, lo, mid)
    right_res = search_range_recur(nums, target, mid + 1, hi)
    if left_res and right_res:
        return (left_res[0], right_res[1])
    else:
        return left_res or right_res
    

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
    unittest.main(exit=False, verbosity=2)
```


### Feb 22, 2022 \[Hard\] Find the Element That Appears Once While Others Occur 3 Times
---
> **Question:** Given an array of integers where every integer occurs three times except for one integer, which only occurs once, find and return the non-duplicated integer.
>
> For example, given `[6, 1, 3, 3, 3, 6, 6]`, return `1`. Given `[13, 19, 13, 13]`, return `19`.
>
> Do this in `O(N)` time and `O(1)` space.


**My thoughts:** An interger repeats 3 time, then each of its digit will repeat 3 times. If a digit repeat 1 more time on top of that, then that digit must be contributed by the unique number. 

**Trivial Solution:** [https://repl.it/@trsong/Find-the-Element-That-Appears-Once-While-Others-Occur-3-Time](https://repl.it/@trsong/Find-the-Element-That-Appears-Once-While-Others-Occur-3-Time)
```py
import unittest

INT_SIZE = 32

def find_uniq_elem(nums):
    res = 0
    count = 0
    for i in xrange(INT_SIZE):
        count = 0
        for num in nums:
            if num & 1 << i:
                count += 1

        if count % 3 == 1:
            res |= 1 << i 
    return res
```

**Solution:** [https://replit.com/@trsong/Find-the-Element-That-Appears-Once-While-Others-Occur-3-Ti-2](https://replit.com/@trsong/Find-the-Element-That-Appears-Once-While-Others-Occur-3-Ti-2)
```py
import unittest

def find_uniq_elem(nums):
    once = twice = 0
    for num in nums:
        twice |= once & num
        once ^= num
        not_thrice = ~(once & twice)
        once &= not_thrice
        twice &= not_thrice
    return once


class FindUniqElemSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(1, find_uniq_elem([6, 1, 3, 3, 3, 6, 6]))

    def test_example2(self):
        self.assertEqual(19, find_uniq_elem([13, 19, 13, 13]))

    def test_example3(self):
        self.assertEqual(2, find_uniq_elem([12, 1, 12, 3, 12, 1, 1, 2, 3, 3]))

    def test_example4(self):
        self.assertEqual(20, find_uniq_elem([10, 20, 10, 30, 10, 30, 30]))

    def test_ascending_array(self):
        self.assertEqual(4, find_uniq_elem([1, 1, 1, 2, 2, 2, 3, 3, 3, 4]))

    def test_descending_array(self):
        self.assertEqual(2, find_uniq_elem([2, 1, 1, 1, 0, 0, 0]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 21, 2022 \[Hard\] Order of Course Prerequisites
---
> **Question:** We're given a hashmap associating each courseId key with a list of courseIds values, which represents that the prerequisites of courseId are courseIds. Return a sorted ordering of courses such that we can finish all courses.
>
> Return null if there is no such ordering.
>
> For example, given `{'CSC300': ['CSC100', 'CSC200'], 'CSC200': ['CSC100'], 'CSC100': []}`, should return `['CSC100', 'CSC200', 'CSC300']`.


**Solution with Topological Sort:** [https://replit.com/@trsong/Find-Order-of-Course-Prerequisites-3](https://replit.com/@trsong/Find-Order-of-Course-Prerequisites-3)
```py
import unittest

def sort_courses(prereq_map):
    neighbors = {}
    inward_node = {}
    queue = []
    for course, dependencies in prereq_map.items():
        if not dependencies:
            queue.append(course)
            
        for pre_course in dependencies:
            neighbors[pre_course] = neighbors.get(pre_course, [])
            neighbors[pre_course].append(course)
            inward_node[course] = inward_node.get(course, 0) + 1

    top_order = []
    while queue:
        cur = queue.pop(0)
        top_order.append(cur)

        for nb in neighbors.get(cur, []):
            inward_node[nb] -= 1
            if inward_node[nb] == 0:
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

### Feb 20, 2022  \[Medium\] Index of Larger Next Number
---
> **Question:** Given a list of numbers, for each element find the next element that is larger than the current element. Return the answer as a list of indices. If there are no elements larger than the current element, then use `-1` instead.

**Example:** 
```py
larger_number([3, 2, 5, 6, 9, 8])
# return [2, 2, 3, 4, -1, -1]
```

**My thoughts:** The idea is to iterate backwards and only store large element along the way with a stack. Doing such will mantain the stack in ascending order. We can then treat the stack as a history of larger element on the right. The algorithm work in the following way:

For each element, we push current element in stack. And the the same time, we pop all elements that are smaller than current element until we find a larger element that is the next larger element in the list.

Note that in worst case scenario, each element can only be pushed and poped from stack once, leaves the time complexity being `O(n)`.

**Solution with Stack:** [https://replit.com/@trsong/Find-Index-of-Larger-Next-Number-2](https://replit.com/@trsong/Find-Index-of-Larger-Next-Number-2)
```py
import unittest

def larger_number(nums):
    n = len(nums)
    stack = []
    res = [-1] * n
    for i in range(n - 1, -1, -1):
        while stack and nums[i] >= nums[stack[-1]]:
            stack.pop()
        
        if stack:
            res[i] = stack[-1]
        stack.append(i)
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
    unittest.main(exit=False, verbosity=2)
```

### Feb 19, 2022  \[Easy\] Determine If Singly Linked List is Palindrome
---
> **Question:** Determine whether a singly linked list is a palindrome. 
>
> For example, `1 -> 4 -> 3 -> 4 -> 1` returns `True` while `1 -> 4` returns `False`. 

**Solution:** [https://replit.com/@trsong/Determine-If-Singly-Linked-List-is-Palindrome-2](https://replit.com/@trsong/Determine-If-Singly-Linked-List-is-Palindrome-2)
```py
import unittest

def is_palindrome(lst):
    reverse_lst = list_reverse(lst)
    return is_list_equal(lst, reverse_lst)


def list_reverse(lst):
    res = None
    p = lst
    while p:
        res = ListNode(p.val, res)
        p = p.next
    return res


def is_list_equal(l1, l2):
    p, q = l1, l2
    while p or q:
        if p.val != q.val:
            return False
        p = p.next
        q = q.next
    return p is None and q is None


class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
    
    @staticmethod
    def List(*vals):
        dummy = cur = ListNode(-1)
        for val in vals:
            cur.next = ListNode(val)
            cur = cur.next
        return dummy.next

    def __repr__(self):
        return "%s -> %s" % (self.val, self.next)


class IsPalindromeSpec(unittest.TestCase):
    def test_empty_list(self):
        self.assertTrue(is_palindrome(None))

    def test_one_element_list(self):
        self.assertTrue(is_palindrome(ListNode.List(42)))

    def test_two_element_list(self):
        self.assertFalse(is_palindrome(ListNode.List(1, 2)))

    def test_two_element_palindrome(self):
        self.assertTrue(is_palindrome(ListNode.List(6, 6)))

    def test_three_element_list(self):
        self.assertFalse(is_palindrome(ListNode.List(1, 2, 3)))

    def test_three_element_list2(self):
        self.assertFalse(is_palindrome(ListNode.List(1, 1, 2)))

    def test_three_element_list3(self):
        self.assertFalse(is_palindrome(ListNode.List(1, 2, 2)))

    def test_three_element_palindrome(self):
        self.assertTrue(is_palindrome(ListNode.List(1, 2, 1)))

    def test_three_element_palindrome2(self):
        self.assertTrue(is_palindrome(ListNode.List(1, 1, 1)))

    def test_even_element_list(self):
        self.assertFalse(is_palindrome(ListNode.List(1, 2, 3, 4, 2, 1)))

    def test_even_element_list2(self):
        self.assertTrue(is_palindrome(ListNode.List(1, 2, 3, 3, 2, 1)))

    def test_odd_element_list(self):
        self.assertTrue(is_palindrome(ListNode.List(1, 2, 3, 2, 1)))

    def test_odd_element_list2(self):
        self.assertFalse(is_palindrome(ListNode.List(1, 2, 3, 3, 1)))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 18, 2022 \[Medium\] Insert into Sorted Circular Linked List
---
> **Question:** Insert a new value into a sorted circular linked list (last element points to head). Return the node with smallest value.  

**Solution:** [https://replit.com/@trsong/Insert-into-Already-Sorted-Circular-Linked-List-2](https://replit.com/@trsong/Insert-into-Already-Sorted-Circular-Linked-List-2)
```py
import unittest

def insert(root, val):
    node = Node(val)

    if not root:
        node.next = node
        return node
    
    # it does not matter whether the to-be-insert value is min or max
    search_val = val if val >= root.val else float('inf')
    p = root
    while p.next != root:
        if p.next.val >= search_val:
            node.next = p.next
            p.next = node
            return root
        p = p.next
    
    p.next = node
    node.next = root
    return root if val >= root.val else node
    

##############################
# Testing utilities
##############################
class Node(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    def __str__(self):
        return str(Node.flatten(self))

    @staticmethod
    def flatten(root):
        if not root:
            return []
        p = root
        res = []
        while True:
            res.append(p.val)
            p = p.next
            if p == root:
                break
        return res

    @staticmethod
    def list(*vals):
        dummy = Node(-1)
        t = dummy
        for v in vals:
            t.next = Node(v)
            t = t.next
        t.next = dummy.next
        return dummy.next


class InsertSpec(unittest.TestCase):
    def assert_result(self, expected, res):
        self.assertEqual(str(expected), str(res))

    def test_empty_list(self):
        self.assert_result(Node.list(1), insert(None, 1))

    def test_prepend_list(self):
        self.assert_result(Node.list(0, 1), insert(Node.list(1), 0))

    def test_append_list(self):
        self.assert_result(Node.list(1, 2, 3), insert(Node.list(1, 2), 3))

    def test_insert_into_correct_position(self):
        self.assert_result(Node.list(1, 2, 3, 4, 5),
                           insert(Node.list(1, 2, 4, 5), 3))

    def test_duplicated_elements(self):
        self.assert_result(Node.list(0, 0, 1, 2),
                           insert(Node.list(0, 0, 2), 1))

    def test_duplicated_elements2(self):
        self.assert_result(Node.list(0, 0, 1), insert(Node.list(0, 0), 1))

    def test_duplicated_elements3(self):
        self.assert_result(Node.list(0, 0, 0, 0),
                           insert(Node.list(0, 0, 0), 0))

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 17, 2022 \[Easy\] Maximum Subarray Sum 
---
> **Question:** You are given a one dimensional array that may contain both positive and negative integers, find the sum of contiguous subarray of numbers which has the largest sum.
>
> For example, if the given array is `[-2, -5, 6, -2, -3, 1, 5, -6]`, then the maximum subarray sum is 7 as sum of `[6, -2, -3, 1, 5]` equals 7
>
> Solve this problem with Divide and Conquer as well as DP separately.

**DP Solution:** [https://replit.com/@trsong/Maximum-Subarray-Sum-1-1](https://replit.com/@trsong/Maximum-Subarray-Sum-1-1)
```py
def max_sub_array_sum(nums):
    n = len(nums)
    # Let dp[i] represents max sub array sum with window ended at index i - 1
    # dp[i] = 0                          if dp[i - 1] < 0
    #       = nums[i - 1] + dp[i - 1]    otherwise
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = nums[i - 1] + max(dp[i - 1], 0)
    return max(dp)
```

**Divide and Conquer Solution:** [https://replit.com/@trsong/Maximum-Subarray-Sum-1-2](https://replit.com/@trsong/Maximum-Subarray-Sum-1-2)
```py
import unittest

def max_sub_array_sum(nums):
    if not nums:
        return 0
    return max_sub_array_sum_recur(nums, 0, len(nums) - 1).result


def max_sub_array_sum_recur(nums, start, end):
    if start == end:
        num = nums[start]
        return Result(num, num, num, num)
    
    mid = start + (end - start) // 2
    left_res = max_sub_array_sum_recur(nums, start, mid)
    right_res = max_sub_array_sum_recur(nums, mid + 1, end)
    prefix = max(0, left_res.total + right_res.prefix, left_res.prefix)
    suffix = max(0, right_res.suffix, right_res.total + left_res.suffix)
    total = left_res.total + right_res.total
    result = max(prefix, suffix, left_res.result, right_res.result,  left_res.suffix + right_res.prefix)
    return Result(prefix, suffix, total, result)


class Result(object):
    def __init__(self, prefix=0, suffix=0, total=0, result=0):
        self.prefix = prefix
        self.suffix = suffix
        self.total = total
        self.result = result
    

class MaxSubArraySum(unittest.TestCase):
    def test_empty_array(self):
        self.assertEqual(0, max_sub_array_sum([]))

    def test_ascending_array(self):
        self.assertEqual(6, max_sub_array_sum([-3, -2, -1, 0, 1, 2, 3]))
        
    def test_descending_array(self):
        self.assertEqual(6, max_sub_array_sum([3, 2, 1, 0, -1]))

    def test_example_array(self):
        self.assertEqual(7, max_sub_array_sum([-2, -5, 6, -2, -3, 1, 5, -6]))

    def test_negative_array(self):
        self.assertEqual(0, max_sub_array_sum([-2, -1]))

    def test_positive_array(self):
        self.assertEqual(3, max_sub_array_sum([1, 2]))

    def test_swing_array(self):
        self.assertEqual(5, max_sub_array_sum([-3, 3, -2, 2, -5, 5]))
        self.assertEqual(1, max_sub_array_sum([-1, 1, -1, 1, -1]))
        self.assertEqual(2, max_sub_array_sum([-100, 1, -100, 2, -100]))

    def test_converging_array(self):
        self.assertEqual(4, max_sub_array_sum([-3, 3, -2, 2, 1, 0]))

    def test_positive_negative_positive_array(self):
        self.assertEqual(8, max_sub_array_sum([7, -1, -2, 3, 1]))
        self.assertEqual(7, max_sub_array_sum([7, -1, -2, 0, 1, 1]))

    def test_negative_positive_array(self):
        self.assertEqual(3, max_sub_array_sum([-100, 1, 0, 2, -100]))
  

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 16, 2022 \[Medium\] LRU Cache
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


**Solution:** [https://replit.com/@trsong/Design-LRU-Cache-3](https://replit.com/@trsong/Design-LRU-Cache-3)
```py
import unittest

class LRUCache(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.node_lookup = {}
        self.head = ListNode()
        self.tail = ListNode(prev=self.head)
        self.head.next = self.tail

    def get(self, key):
        if key not in self.node_lookup:
            return None
        
        node = self.node_lookup[key]
        self.populate(node)
        return node.item.val

    def put(self, key, val):
        # put value & populate
        node = self.node_lookup.get(key, ListNode())
        node.item = CacheEntry(key, val)
        self.populate(node)

        # evit at capacity
        if len(self.node_lookup) > self.capacity:
            evited_node = self.head.next
            self.head.next = evited_node.next
            evited_node.next.prev = self.head
            del self.node_lookup[evited_node.item.key]

    def populate(self, node):
        key = node.item.key
        self.node_lookup[key] = node

        # detech node
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        
        # reset node
        node.prev = self.tail.prev
        node.next = self.tail
        
        # reset neighbors of node 
        node.prev.next = node
        self.tail.prev = node


class CacheEntry(object):
    def __init__(self, key, val):
        self.key = key
        self.val = val


class ListNode(object):
    def __init__(self, item=None, next=None, prev=None):
        self.item = item
        self.next = next
        self.prev = prev


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

### Feb 15, 2022 LC 1171 \[Medium\] Remove Consecutive Nodes that Sum to 0
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

**My thoughts:** This question is just the list version of [Contiguous Sum to K](https://trsong.github.io/python/java/2019/05/01/DailyQuestions.html#jul-24-2019-medium-contiguous-sum-to-k). The idea is exactly the same, in previous question: `sum[i:j]` can be achieved use `prefix[j] - prefix[i-1] where i <= j`, whereas for this question, we can use map to store the "prefix" sum: the sum from the head node all the way to current node. And by checking the prefix so far, we can easily tell if there is a node we should have seen before that has "prefix" sum with same value. i.e. There are consecutive nodes that sum to 0 between these two nodes.

**Solution:** [https://replit.com/@trsong/Remove-Consecutive-Nodes-that-Sum-to-Zero-3](https://replit.com/@trsong/Remove-Consecutive-Nodes-that-Sum-to-Zero-3)
```py
import unittest

def remove_zero_sum_sublists(head):
    p = dummy = ListNode(0, head)
    p = head
    prefix_node = {0: dummy}
    prefix_sum = 0

    while p:
        prefix_sum += p.val
        if prefix_sum not in prefix_node:
            prefix_node[prefix_sum] = p
        else:
            loop_start = prefix_node[prefix_sum]
            q = loop_start.next
            prefix_sum_to_remove = prefix_sum
            while q != p:
                prefix_sum_to_remove += q.val
                del prefix_node[prefix_sum_to_remove]
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
    unittest.main(exit=False, verbosity=2)
```

### Feb 14, 2022 \[Easy\] URL Shortener
---
> **Question:** Implement a URL shortener with the following methods:
>
> - `shorten(url)`, which shortens the url into a six-character alphanumeric string, such as `zLg6wl`.
> - `restore(short)`, which expands the shortened string into the original url. If no such shortened string exists, return `null`.
>
> **Follow-up:** What if we enter the same URL twice?

**Solution:** [https://replit.com/@trsong/URL-Shortener-2](https://replit.com/@trsong/URL-Shortener-2)
```py
import unittest
import random

class URLShortener(object):
    CHAR_SET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    CODE_LEN = 6

    @staticmethod
    def encode(s):
        return "".join(
            random.choice(URLShortener.CHAR_SET)
            for _ in range(URLShortener.CODE_LEN))

    def __init__(self):
        self.long_to_short = {}
        self.short_to_long = {}

    def shorten(self, url):
        if url not in self.long_to_short:
            code = URLShortener.encode(url)
            while code in self.short_to_long:
                # avoid duplicated short url
                code = URLShortener.encode(url)
            self.long_to_short[url] = code
            self.short_to_long[code] = url
        return self.long_to_short[url]

    def restore(self, short):
        return self.short_to_long.get(short, None)


class URLShortenerSpec(unittest.TestCase):
    def test_should_be_able_to_init(self):
        URLShortener()

    def test_restore_should_not_fail_when_url_not_exists(self):
        url_shortener = URLShortener()
        self.assertIsNone(url_shortener.restore("oKImts"))

    def test_shorten_result_into_six_letters(self):
        url_shortener = URLShortener()
        res = url_shortener.shorten("http://magic_url")
        self.assertEqual(6, len(res))

    def test_restore_short_url_gives_original(self):
        url_shortener = URLShortener()
        original_url = "http://magic_url"
        short_url = url_shortener.shorten(original_url)
        self.assertEqual(original_url, url_shortener.restore(short_url))

    def test_shorten_different_url_gives_different_results(self):
        url_shortener = URLShortener()
        url1 = "http://magic_url_1"
        res1 = url_shortener.shorten(url1)
        url2 = "http://magic_url_2"
        res2 = url_shortener.shorten(url2)
        self.assertNotEqual(res1, res2)

    def test_shorten_same_url_gives_same_result(self):
        url_shortener = URLShortener()
        url = "http://magic_url_1"
        res1 = url_shortener.shorten(url)
        res2 = url_shortener.shorten(url)
        self.assertEqual(res1, res2)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```
### Feb 13, 2022 \[Hard\] Order of Alien Dictionary
--- 
> **Question:** You come across a dictionary of sorted words in a language you've never seen before. Write a program that returns the correct order of letters in this language.
>
> For example, given `['xww', 'wxyz', 'wxyw', 'ywx', 'ywz']`, you should return `['x', 'z', 'w', 'y']`.

**My thoughts:** As the alien letters are topologically sorted, we can just mimic what topological sort with numbers and try to find pattern.

Suppose the dictionary contains: `01234`. Then the words can be `023, 024, 12, 133, 2433`. Notice that we can only find the relative order by finding first unequal letters between consecutive words. eg.  `023, 024 => 3 < 4`.  `024, 12 => 0 < 1`.  `12, 133 => 2 < 3`

With relative relation, we can build a graph with each occurring letters being veteces and edge `(u, v)` represents `u < v`. If there exists a loop that means we have something like `a < b < c < a` and total order not exists. Otherwise we preform a topological sort to generate the total order which reveals the alien dictionary. 

As for implementation of topological sort, there are two ways, one is the following by constantly removing edges from visited nodes. The other is to [first DFS to find the reverse topological order then reverse again to find the result](https://trsong.github.io/python/java/2019/11/02/DailyQuestionsNov.html#nov-9-2019-hard-order-of-alien-dictionary). 


**Solution with Toplogical Sort:** [https://replit.com/@trsong/Alien-Dictionary-Order-3](https://replit.com/@trsong/Alien-Dictionary-Order-3)
```py
import unittest

def dictionary_order(sorted_words):
    neighbors = {}
    inward_count = {}
    for i in range(1, len(sorted_words)):
        prev_word = sorted_words[i - 1]
        cur_word = sorted_words[i]
        for prev_ch, cur_ch in zip(prev_word, cur_word):
            if prev_ch != cur_ch:
                neighbors[prev_ch] = neighbors.get(prev_ch, [])
                neighbors[prev_ch].append(cur_ch)
                inward_count[cur_ch] = inward_count.get(cur_ch, 0) + 1
                break
            
    
    char_set = { ch for word in sorted_words for ch in word }
    queue = [ch for ch in char_set if ch not in inward_count]
    top_order = []

    while queue:
        cur = queue.pop(0)
        top_order.append(cur)

        for child in neighbors.get(cur, []):
            inward_count[child] -= 1
            if inward_count[child] == 0:
                del inward_count[child]
                queue.append(child)
    return top_order if len(char_set) == len(top_order) else None


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

### Feb 12, 2022 \[Medium\] Unit Converter
---
> **Question:** The United States uses the imperial system of weights and measures, which means that there are many different, seemingly arbitrary units to measure distance. There are 12 inches in a foot, 3 feet in a yard, 22 yards in a chain, and so on.
>
> Create a data structure that can efficiently convert a certain quantity of one unit to the correct amount of any other unit. You should also allow for additional units to be added to the system.

**Solution with DFS:** [https://replit.com/@trsong/Unit-converter](https://replit.com/@trsong/Unit-converter)
```py
import unittest

class UnitConverter(object):
    def __init__(self):
        self.neighbor = {}

    def add_rule(self, src, dst, unit):
        """
        Assumptions: 
        1) a rule set up bi-directional link
        2) new rule override old rule
        3) all rules are compatiable with one another
        """
        self.neighbor[src] = self.neighbor.get(src, {})
        self.neighbor[src][dst] = unit
        self.neighbor[dst] = self.neighbor.get(dst, {})
        self.neighbor[dst][src] = 1.0 / unit

    def convert(self, src, dst, amount):
        stack = [(src, amount)]
        visited = set()

        while stack:
            cur, accu = stack.pop()
            if cur == dst:
                return accu
            
            if cur in visited:
                continue
            visited.add(cur)
            
            for child, unit in self.neighbor.get(cur, {}).items():
                if child in visited:
                    continue
                stack.append((child, unit * accu))
        return None


class Units:
    INCH = 'inch'
    FOOT = 'foot'
    YARD = 'yard'
    CHAIN = 'chain'
    FURLONG = 'furlong'
    MILE = 'mile'
    CAD = 'CAD'
    USD = 'USD'


class UnitConverterSpec(unittest.TestCase):
    def test_base_unit(self):
        unitConverter = UnitConverter()
        unitConverter.add_rule(Units.FOOT, Units.INCH, 12)
        self.assertEqual(12, unitConverter.convert(Units.FOOT, Units.INCH, 1))
        self.assertEqual(24, unitConverter.convert(Units.FOOT, Units.INCH, 2))

    def test_base_unit_chaining(self):
        unitConverter = UnitConverter()
        unitConverter.add_rule(Units.FOOT, Units.INCH, 12)
        unitConverter.add_rule(Units.YARD, Units.FOOT, 3)
        unitConverter.add_rule(Units.CHAIN, Units.YARD, 22)
        self.assertEqual(36, unitConverter.convert(Units.YARD, Units.INCH, 1))
        self.assertEqual(66, unitConverter.convert(Units.CHAIN, Units.FOOT, 1))
    
    def test_alternative_path(self):
        unitConverter = UnitConverter()
        unitConverter.add_rule(Units.MILE, Units.FOOT, 5280)
        unitConverter.add_rule(Units.FOOT, Units.INCH, 12)
        unitConverter.add_rule(Units.YARD, Units.FOOT, 3)
        unitConverter.add_rule(Units.CHAIN, Units.YARD, 22)
        unitConverter.add_rule(Units.FURLONG, Units.CHAIN, 10)
        unitConverter.add_rule(Units.MILE, Units.FURLONG, 8)
        self.assertEqual(5280, unitConverter.convert(Units.MILE, Units.FOOT, 1))
        self.assertEqual(220, unitConverter.convert(Units.FURLONG, Units.YARD, 1))

    def test_no_rule_exist(self):
        unitConverter = UnitConverter()
        self.assertIsNone(unitConverter.convert(Units.FOOT, Units.INCH, 1))

    def test_no_rule_exist2(self):
        unitConverter = UnitConverter()
        unitConverter.add_rule(Units.FOOT, Units.INCH, 12)
        unitConverter.add_rule(Units.CHAIN, Units.YARD, 22)
        unitConverter.add_rule(Units.FURLONG, Units.CHAIN, 10)
        self.assertIsNone(unitConverter.convert(Units.FURLONG, Units.FOOT, 1))

    def test_assumption_one(self):
        unitConverter = UnitConverter()
        unitConverter.add_rule(Units.FOOT, Units.INCH, 12)
        self.assertAlmostEqual(1/12, unitConverter.convert(Units.INCH, Units.FOOT, 1))
        unitConverter.add_rule(Units.YARD, Units.FOOT, 3)
        self.assertAlmostEqual(1/36, unitConverter.convert(Units.INCH, Units.YARD, 1))

    def test_assumption_two(self):
        unitConverter = UnitConverter()
        unitConverter.add_rule(Units.USD, Units.CAD, 1)
        self.assertEqual(2, unitConverter.convert(Units.USD, Units.CAD, 2))
        unitConverter.add_rule(Units.USD, Units.CAD, 1.2)
        self.assertEqual(2.4, unitConverter.convert(Units.USD, Units.CAD, 2))

    def test_assumption_one_and_two(self):
        unitConverter = UnitConverter()
        unitConverter.add_rule(Units.USD, Units.CAD, 1)
        self.assertEqual(2, unitConverter.convert(Units.USD, Units.CAD, 2))
        unitConverter.add_rule(Units.CAD, Units.USD, 0.8)
        self.assertEqual(2.5, unitConverter.convert(Units.USD, Units.CAD, 2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 11, 2022 \[Medium\] Swap Even and Odd Bits
---
> **Question:** Given an unsigned 8-bit integer, swap its even and odd bits. The 1st and 2nd bit should be swapped, the 3rd and 4th bit should be swapped, and so on.

**Example:**

```py
10101010 should be 01010101. 11100010 should be 11010001.
```
> Bonus: Can you do this in one line?

**Solution:** [https://replit.com/@trsong/Swap-Even-and-Odd-Bits-of-Binary-Number](https://replit.com/@trsong/Swap-Even-and-Odd-Bits-of-Binary-Number)
```py
import unittest

def swap_bits(num):
    # 1010 is 0xa, 0101 is 0x5
    # 32 bit has 8 bits (4 * 8 = 32)
    return (num & 0xaaaaaaaa) >> 1 | (num & 0x55555555 ) << 1


class SwapBitSpec(unittest.TestCase):
    def assert_result(self, expected, res):
        self.assertEqual(bin(expected), bin(res))

    def test_example1(self):
        self.assert_result(0b01010101, swap_bits(0b10101010))

    def test_example2(self):
        self.assert_result(0b11010001, swap_bits(0b11100010))

    def test_zero(self):
        self.assert_result(0, swap_bits(0))
    
    def test_one(self):
        self.assert_result(0b10, swap_bits(0b1))

    def test_odd_digits(self):
        self.assert_result(0b1011, swap_bits(0b111))

    def test_large_number(self):
        self.assert_result(0xffffffff, swap_bits(0xffffffff))
    
    def test_large_number2(self):
        self.assert_result(0xaaaaaaaa, swap_bits(0x55555555))

    def test_large_number3(self):
        self.assert_result(0x55555555, swap_bits(0xaaaaaaaa))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 10, 2022 \[Easy\] BST Nodes Sum up to K
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

**Solution:** [https://replit.com/@trsong/Find-BST-Nodes-Sum-up-to-K-2](https://replit.com/@trsong/Find-BST-Nodes-Sum-up-to-K-2)
```py
import unittest

def find_pair(tree, k):
    left_stream = in_order_iteration(tree)
    right_stream = reverse_in_order_iteration(tree)
    left = next(left_stream, None)
    right = next(right_stream, None)

    while left != right:
        pair_sum = left.val + right.val
        if pair_sum == k:
            return [left.val, right.val]
        elif pair_sum < k:
            left = next(left_stream, None)
        else:
            right = next(right_stream, None)
    return None


def in_order_iteration(root):
    if root is not None:
        for node in in_order_iteration(root.left):
            yield node
        yield root
        for node in in_order_iteration(root.right):
            yield node


def reverse_in_order_iteration(root):
    if root is not None:
        for node in reverse_in_order_iteration(root.right):
            yield node
        yield root
        for node in reverse_in_order_iteration(root.left):
            yield node


class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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
        self.assertEqual([5, 15], find_pair(n10, 20))

    def test_empty_tree(self):
        self.assertIsNone(find_pair(None, 0))

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
        self.assertEqual([2, 5], find_pair(n7, 7))
        self.assertEqual([5, 13], find_pair(n7, 18))
        self.assertEqual([7, 17], find_pair(n7, 24))
        self.assertEqual([11, 17], find_pair(n7, 28))
        self.assertIsNone(find_pair(n7, 4))

    def test_tree_with_same_value(self):
        """
        42
          \
           42
        """
        tree = Node(42, right=Node(42))
        self.assertEqual([42, 42], find_pair(tree, 84))
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
        self.assertEqual([2, 5], find_pair(n7, 7))
        self.assertEqual([5, 13], find_pair(n7, 18))
        self.assertEqual([7, 17], find_pair(n7, 24))
        self.assertEqual([11, 17], find_pair(n7, 28))
        self.assertIsNone(find_pair(n7, 4))


if __name__ == '__main__':
   unittest.main(exit=False, verbosity=2)
```

### Feb 9, 2022 LC 1344 \[Easy\] Angle between Clock Hands
---
> **Question:** Given a clock time in `hh:mm` format, determine, to the nearest degree, the angle between the hour and the minute hands.

**Example:**
```py
Input: "9:00"
Output: 90
```

**Solution:** [https://replit.com/@trsong/Calculate-Angle-between-Clock-Hands-2](https://replit.com/@trsong/Calculate-Angle-between-Clock-Hands-2)
```py
import unittest

def clock_angle(hhmm):
    hh, mm = hhmm.split(':')
    h = int(hh) % 12
    m = int(mm)

    m_unit = 360 / 60.0
    h_unit = 360 / 12.0

    m_angle = m_unit * m
    h_angle = h_unit * h + h_unit * m / 60.0
    delta = abs(m_angle - h_angle)
    
    return min(360 - delta, delta)


class ClockAngleSpec(unittest.TestCase):
    def test_minute_point_zero(self):
        hhmm = "12:00"
        angle = 0
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_minute_point_zero2(self):
        hhmm = "1:00"
        angle = 30
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_minute_point_zero3(self):
        hhmm = "9:00"
        angle = 90
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_minute_point_zero4(self):
        hhmm = "6:00"
        angle = 180
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_half_pass_hour(self):
        hhmm = "12:30"
        angle = 165
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_half_pass_hour2(self):
        hhmm = "3:30"
        angle = 75
        self.assertEqual(angle, clock_angle(hhmm))

    def test_irregular_time(self):
        hhmm = "16:20"
        angle = 10
        self.assertEqual(angle, clock_angle(hhmm))

    def test_irregular_time3(self):
        hhmm = "3:15"
        angle = 7.5
        self.assertEqual(angle, clock_angle(hhmm))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```
### Feb 8, 2022 \[Easy\] Exists Overlap Rectangle
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

**Solution:** [https://replit.com/@trsong/Exists-Overlap-Rectangles-2](https://replit.com/@trsong/Exists-Overlap-Rectangles-2)
```py
import unittest

def exists_overlap_rectangle(rectangles):
    rects = list(map(Rectangle.from_json, rectangles))
    rects.sort(key=lambda rect: (rect.xmin, rect.ymin))

    n = len(rects)
    for i in range(n):
        rect1 = rects[i]
        for j in range(i + 1, n):
            rect2 = rects[j]
            if rect2.xmin > rect1.xmax or rect2.ymin > rect1.ymax:
                continue
            if rect1.has_overlap(rect2):
                return True
    return False


class Rectangle(object):
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    @staticmethod
    def from_json(json):
        top_left = json.get("top_left")
        dimensions = json.get("dimensions")
        xmin = top_left[0]
        xmax = xmin + dimensions[0]
        ymax = top_left[1]
        ymin = ymax - dimensions[1]
        return Rectangle(xmin, xmax, ymin, ymax)

    def has_overlap(self, other):
        has_x_overlap = min(self.xmax, other.xmax) > max(self.xmin, other.xmin)
        has_y_overlap = min(self.ymax, other.ymax) > max(self.ymin, other.ymin)
        return has_x_overlap and has_y_overlap
    

class ExistsOverlapRectangleSpec(unittest.TestCase):
    def test_example(self):
        rectangles = [
            {
                "top_left": (1, 4),
                "dimensions": (3, 3)  # width, height
            },
            {
                "top_left": (-1, 3),
                "dimensions": (2, 1)
            },
            {
                "top_left": (0, 5),
                "dimensions": (4, 3)
            }
        ]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_empty_rectangle_list(self):
        self.assertFalse(exists_overlap_rectangle([]))

    def test_two_overlap_rectangle(self):
        rectangles = [
            {
                "top_left": (0, 1),
                "dimensions": (1, 3)  # width, height
            },
            {
                "top_left": (-1, 0),
                "dimensions": (3, 1)
            }
        ]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_two_overlap_rectangle_form_a_cross(self):
        rectangles = [
            {
                "top_left": (-1, 1),
                "dimensions": (3, 2)  # width, height
            },
            {
                "top_left": (0, 0),
                "dimensions": (1, 1)
            }
        ]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_same_y_coord_not_overlap(self):
        rectangles = [
            {
                "top_left": (0, 0),
                "dimensions": (1, 1)  # width, height
            },
            {
                "top_left": (1, 0),
                "dimensions": (2, 2)
            },
            {
                "top_left": (3, 0),
                "dimensions": (5, 2)
            }
        ]
        self.assertFalse(exists_overlap_rectangle(rectangles))

    def test_same_y_coord_overlap(self):
        rectangles = [
            {
                "top_left": (0, 0),
                "dimensions": (1, 1)  # width, height
            },
            {
                "top_left": (1, 0),
                "dimensions": (2, 2)
            },
            {
                "top_left": (3, 0),
                "dimensions": (5, 2)
            }
        ]
        self.assertFalse(exists_overlap_rectangle(rectangles))

    def test_rectangles_in_different_quadrant(self):
        rectangles = [
            {
                "top_left": (1, 1),
                "dimensions": (2, 2)  # width, height
            },
            {
                "top_left": (-1, 1),
                "dimensions": (2, 2)
            },
            {
                "top_left": (1, -1),
                "dimensions": (2, 2)
            },
            {
                "top_left": (-1, -1),
                "dimensions": (2, 2)
            }
        ]
        self.assertFalse(exists_overlap_rectangle(rectangles))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 7, 2022 \[Easy\] Permutation with Given Order
---
> **Question:** A permutation can be specified by an array `P`, where `P[i]` represents the location of the element at `i` in the permutation. For example, `[2, 1, 0]` represents the permutation where elements at the index `0` and `2` are swapped.
>
> Given an array and a permutation, apply the permutation to the array. 
>
> For example, given the array `["a", "b", "c"]` and the permutation `[2, 1, 0]`, return `["c", "b", "a"]`.

**My thoughts:** In-place solution requires swapping current position `i` with target postion `j` if `j > i` (out of already processed window). However, if `j < i`, then `j`'s position has been swapped, we backtrack recursively to find `j`'s new position.

**Solution:** [https://replit.com/@trsong/Find-Permutation-with-Given-Order-2](https://replit.com/@trsong/Find-Permutation-with-Given-Order-2)
```py
import unittest

def permute(arr, order):
    for from_pos in range(len(order)):
        to_pos = order[from_pos]

        # Only swap current elem out of already processed window
        # If target position is within window, then that position has been swapped 
        while to_pos < from_pos:
            to_pos = order[to_pos]
        
        arr[to_pos], arr[from_pos] = arr[from_pos], arr[to_pos]
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

### Feb 6, 2022  \[Medium\] Maximum Distance among Binary Strings
---
> **Question:** The distance between 2 binary strings is the sum of their lengths after removing the common prefix. For example: the common prefix of `1011000` and `1011110` is `1011` so the distance is `len("000") + len("110") = 3 + 3 = 6`.
>
> Given a list of binary strings, pick a pair that gives you maximum distance among all possible pair and return that distance.

**My thoughts:** The idea is to build a trie to keep track of common characters as well as remaining characters to allow quickly calculate max path length on the left child or right child. 

There are three situations:

- A node has two children: max distance = max distannce of left node + max distance of right node
- A node has one child and is terminal node: max distance = max distance of that child
- A node has one child and is not terminal node: do nothing


**Solution with Trie:** [https://replit.com/@trsong/Maximum-Distance-among-Binary-Strings-2](https://replit.com/@trsong/Maximum-Distance-among-Binary-Strings-2)
```py
import unittest

def max_distance(bins):
    if len(bins) < 2:
        return -1

    t = Trie()
    for num_str in bins:
        t.insert(num_str)
    return t.max_distance()


class Trie(object):
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.max_path = 0

    def insert(self, num_str):
        p = self
        n = len(num_str)
        for index, digit in enumerate(num_str):
            remaining_char = n - index
            p.max_path = max(p.max_path, remaining_char)
            p.children[digit] = p.children.get(digit, Trie())
            p = p.children[digit]
        p.is_end = True

    def max_distance(self):
        stack = [self]
        res = 0

        while stack:
            cur = stack.pop()
            if cur.is_end:
                res = max(res, cur.max_path)

            if len(cur.children) == 2:
                left_max_path = cur.children['0'].max_path
                right_max_path = cur.children['1'].max_path
                res = max(res, 2 + left_max_path + right_max_path)

            stack.extend(cur.children.values())
        return res


class MaxDistanceSpec(unittest.TestCase):
    def test_example(self):
        bins = ['1011000', '1011110']
        expected = len('000') + len('110')
        self.assertEqual(expected, max_distance(bins))

    def test_less_than_two_strings(self):
        self.assertEqual(-1, max_distance([]))
        self.assertEqual(-1, max_distance(['']))
        self.assertEqual(-1, max_distance(['0010']))

    def test_empty_string(self):
        self.assertEqual(0, max_distance(['', '']))
        self.assertEqual(6, max_distance(['', '010101']))

    def test_string_with_same_prefix(self):
        bins = ['000', '0001', '0001001']
        expected = len('1001')
        self.assertEqual(expected, max_distance(bins))

    def test_string_with_same_prefix2(self):
        bins = ['000', '0001', '0001001']
        expected = len('1001')
        self.assertEqual(expected, max_distance(bins))

    def test_return_max_distance_through_root(self):
        """
          0
         / \
        0   1
           / \
          0   1
           \
            1  
        """
        bins = ['00', '0101', '011']
        expected = len('0') + len('101')
        self.assertEqual(expected, max_distance(bins))

    def test_return_max_distance_not_through_root(self):
        """
        0
         \
          1
         / \
        0   1
       /   / \ 
      0   0   1
               \
                1
        """
        bins = ['0100', '0110', '01111']
        expected = len('00') + len('111')
        self.assertEqual(expected, max_distance(bins))

    def test_return_max_distance_when_there_is_prefix(self):
        """
        0
         \
          1 *
           \
            1
             \
              1
             / \
            0   1
        """
        bins = ['01', '01110', '01111']
        expected = len('111')
        self.assertEqual(expected, max_distance(bins))

    def test_return_max_distance_when_there_is_prefix2(self):
        """
        0
       / \
      0   1 *
           \
            1
             \
              1
             / \
            0   1
               /
              0 
        """
        bins = ['01', '01110', '011110', '00']
        expected = len('0') + len('11110')
        self.assertEqual(expected, max_distance(bins))

    def test_return_max_distance_when_there_is_prefix_and_empty_string(self):
        bins = ['', '0', '00', '000', '1']
        expected = len('1') + len('000')
        self.assertEqual(expected, max_distance(bins))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 5, 2022 \[Easy\] Step Word Anagram
---
> **Question:** A step word is formed by taking a given word, adding a letter, and anagramming the result. For example, starting with the word `"APPLE"`, you can add an `"A"` and anagram to get `"APPEAL"`.
>
> Given a dictionary of words and an input word, create a function that returns all valid step words.


**Solution:** [https://replit.com/@trsong/Step-Word-Anagram-3](https://replit.com/@trsong/Step-Word-Anagram-3)
```py
import unittest

def find_step_anagrams(word, dictionary):
    word_histogram = generate_historgram(word)

    return list(filter(
        lambda candidate: 1 ==
            len(candidate) - len(word) ==
            frequency_distance(generate_historgram(candidate), word_histogram),
        dictionary))


def generate_historgram(word):
    histogram = {}
    for ch in word:
        histogram[ch] = histogram.get(ch, 0) + 1
    return histogram


def frequency_distance(histogram1, histogram2):
    res = 0
    for ch, count in histogram1.items():
        res += max(0, count - histogram2.get(ch, 0))
    return res


class FindStepAnagramSpec(unittest.TestCase):
    def test_example(self):
        word = 'APPLE'
        dictionary = ['APPEAL', 'CAPPLE', 'PALPED']
        expected = ['APPEAL', 'CAPPLE', 'PALPED']
        self.assertEqual(
            sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_empty_word(self):
        word = ''
        dictionary = ['A', 'B', 'AB', 'ABC']
        expected = ['A', 'B']
        self.assertEqual(
            sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_empty_dictionary(self):
        word = 'ABC'
        dictionary = []
        expected = []
        self.assertEqual(
            sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_no_match(self):
        word = 'ABC'
        dictionary = ['BBB', 'ACCC']
        expected = []
        self.assertEqual(
            sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_no_match2(self):
        word = 'AA'
        dictionary = ['ABB']
        expected = []
        self.assertEqual(
            sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_repeated_chars(self):
        word = 'AAA'
        dictionary = ['A', 'AA', 'AAA', 'AAAA', 'AAAAB', 'AAB', 'AABA']
        expected = ['AAAA', 'AABA']
        self.assertEqual(
            sorted(expected), sorted(find_step_anagrams(word, dictionary)))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 4, 2022 \[Medium\] Distance Between 2 Nodes in BST
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

**Solution:** [https://replit.com/@trsong/Distance-Between-2-Nodes-in-BST-2](https://replit.com/@trsong/Distance-Between-2-Nodes-in-BST-2)
```py
import unittest

def find_distance(tree, v1, v2):    
    path1 = find_path(tree, v1)
    path2 = find_path(tree, v2)
    num_common_ancestor = 0

    for p1, p2 in zip(path1, path2):
        if p1 != p2:
            break
        num_common_ancestor += 1
    return len(path1) + len(path2) - 2 * num_common_ancestor


def find_path(root, value):
    p = root
    res = []
    while p:
        res.append(p)
        if p.val == value:
            return res
        elif p.val < value:
            p = p.right
        else:
            p = p.left
    return None


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class FindDistanceSpec(unittest.TestCase):
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
    unittest.main(exit=False, verbosity=2)
```
### Feb 3, 2022 LC 872 \[Easy\] Leaf-Similar Trees
---
> **Question:** Given two trees, whether they are `"leaf similar"`. Two trees are considered `"leaf-similar"` if their leaf orderings are the same. 
>
> For instance, the following two trees are considered leaf-similar because their leaves are `[2, 1]`:

```py
# Tree1
    3
   / \ 
  5   1
   \
    2 

# Tree2
    7
   / \ 
  2   1
   \
    2 
```

**Solution with DFS:** [https://replit.com/@trsong/Leaf-Similar-Trees-3](https://replit.com/@trsong/Leaf-Similar-Trees-3)
```py
import unittest

def is_leaf_similar(t1, t2):
    return dfs_path(t1) == dfs_path(t2)


def dfs_path(root):
    if not root:
        return []

    res = []
    stack = [root]
    while stack:
        cur = stack.pop()
        if cur.left is None and cur.right is None:
            res.append(cur.val)
            continue

        for child in [cur.right, cur.left]:
            if child is None:
                continue
            stack.append(child)
    return res
            

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class IsLeafSimilarSpec(unittest.TestCase):
    def test_example(self):
        """
            3
           / \ 
          5   1
           \
            2 

            7
           / \ 
          2   1
           \
            2 
        """
        t1 = TreeNode(3, TreeNode(5, right=TreeNode(2)), TreeNode(1))
        t2 = TreeNode(7, TreeNode(2, right=TreeNode(2)), TreeNode(1))
        self.assertTrue(is_leaf_similar(t1, t2))

    def test_both_empty(self):
        self.assertTrue(is_leaf_similar(None, None))

    def test_one_tree_empty(self):
        self.assertFalse(is_leaf_similar(TreeNode(0), None))

    def test_tree_of_different_depths(self):
        """
          1
         / \
        2   3

           1
         /   \
        5     4
         \   /
          2 3
        """
        t1 = TreeNode(1, TreeNode(2), TreeNode(3))
        t2l = TreeNode(5, right=TreeNode(2))
        t2r = TreeNode(4, TreeNode(3))
        t2 = TreeNode(1, t2l, t2r)
        self.assertTrue(is_leaf_similar(t1, t2))

    def test_tree_with_different_number_of_leaves(self):
        """
          1
         / \
        2   3

           1
         /   
        2     
        """
        t1 = TreeNode(1, TreeNode(2), TreeNode(3))
        t2 = TreeNode(1, TreeNode(2))
        self.assertFalse(is_leaf_similar(t1, t2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 2, 2022 \[Medium\] Generate Binary Search Trees
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

**Solution:** [https://replit.com/@trsong/Generate-Binary-Search-Trees-with-N-Nodes-2](https://replit.com/@trsong/Generate-Binary-Search-Trees-with-N-Nodes-2)
```py
import unittest

def generate_bst(n):
    if n < 1:
        return []

    return list(generate_bst_between(1, n))

def generate_bst_between(lo, hi):
    if lo > hi:
        yield None
    
    for val in range(lo, hi + 1):
        for left_child in generate_bst_between(lo, val - 1):
            for right_child in generate_bst_between(val + 1, hi):
                yield TreeNode(val, left_child, right_child)


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

### Feb 1, 2022 \[Medium\] Maximum Path Sum in Binary Tree
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

**Solution with Recursion:** [https://replit.com/@trsong/Find-Maximum-Path-Sum-in-Binary-Tree-2](https://replit.com/@trsong/Find-Maximum-Path-Sum-in-Binary-Tree-2)
```py
import unittest

def max_path_sum(tree):
    return max_path_sum_recur(tree)[0]


def max_path_sum_recur(tree):
    if not tree:
        return 0, 0

    left_res, left_max_path = max_path_sum_recur(tree.left)
    right_res, right_max_path = max_path_sum_recur(tree.right)
    max_path = tree.val + max(left_max_path, right_max_path)
    max_sum = tree.val + left_max_path + right_max_path
    res = max(left_res, right_res, max_sum)
    return res, max_path
    

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