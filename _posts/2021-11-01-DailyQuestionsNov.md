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


### Nov 24, 2021 \[Hard\] Largest Sum of Non-adjacent Numbers
---

> **Question:** Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0 or negative.
>
> For example, `[2, 4, 6, 2, 5]` should return 13, since we pick 2, 6, and 5. `[5, 1, 1, 5]` should return 10, since we pick 5 and 5.
>
> Follow-up: Can you do this in O(N) time and constant space?


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
