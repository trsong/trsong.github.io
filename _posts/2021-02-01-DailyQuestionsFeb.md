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