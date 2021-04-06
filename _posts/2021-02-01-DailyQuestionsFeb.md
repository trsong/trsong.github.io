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


### Apr 6, 2021 \[Medium\] Minimum Number of Boats to Save Population
---
> **Question:** An imminent hurricane threatens the coastal town of Codeville. If at most two people can fit in a rescue boat, and the maximum weight limit for a given boat is k, determine how many boats will be needed to save everyone.
>
> For example, given a population with weights `[100, 200, 150, 80]` and a boat limit of `200`, the smallest number of boats required will be three.


### Apr 5, 2021 \[Medium\] Merge K Sorted Lists
---
> **Question:** Given k sorted singly linked lists, write a function to merge all the lists into one sorted singly linked list.

**Solution with Priority Queue:** [https://replit.com/@trsong/Merge-K-Sorted-Linked-Lists](https://replit.com/@trsong/Merge-K-Sorted-Linked-Lists)
```py
import unittest
from queue import PriorityQueue

def merge_k_sorted_lists(lists):
    if not lists: 
        return None

    pq = PriorityQueue()
    list_ptr = lists
    while list_ptr:
        sub_list_ptr = list_ptr.val
        if sub_list_ptr:
            pq.put((sub_list_ptr.val, sub_list_ptr))
        list_ptr = list_ptr.next
    
    dummy = p = ListNode(-1)
    while not pq.empty():
        _, lst = pq.get()
        p.next = ListNode(lst.val)
        p = p.next
        if lst.next:
            lst = lst.next
            pq.put((lst.val, lst))
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

### Apr 4, 2021 \[Easy\] Count Line Segment Intersections
---
> **Question:** Suppose you are given two lists of n points, one list `p1, p2, ..., pn` on the line `y = 0` and the other list `q1, q2, ..., qn` on the line `y = 1`. 
>
> Imagine a set of n line segments connecting each point pi to qi. 
>
> Write an algorithm to determine how many pairs of the line segments intersect.

**Solution:** [https://replit.com/@trsong/Count-Line-Segment-Intersections](https://replit.com/@trsong/Count-Line-Segment-Intersections)
```py
import unittest

def num_intersection(p, q):
    res = 0
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if intercept(p[i], q[i], p[j], q[j]):
                res += 1
    return res


def intercept(p1, q1, p2, q2):
    return p1 < p2 and q1 > q2 or p1 > p2 and q1 < q2


class NumIntersectionSpec(unittest.TestCase):
    def test_zero_lines(self):
        p = []
        q = []
        self.assertEqual(0, num_intersection(p, q))

    def test_one_line(self):
        p = [0]
        q = [0]
        self.assertEqual(0, num_intersection(p, q))

    def test_two_lines(self):
        p = [0, 1]
        q = [0, 1]
        self.assertEqual(0, num_intersection(p, q))

    def test_two_lines2(self):
        p = [0, 1]
        q = [1, 0]
        self.assertEqual(1, num_intersection(p, q))

    def test_three_lines(self):
        p = [0, 1, 2]
        q = [1, 2, 0]
        self.assertEqual(2, num_intersection(p, q))

    def test_three_lines2(self):
        p = [0, 1, 2]
        q = [1, 0, 2]
        self.assertEqual(1, num_intersection(p, q))

    def test_three_lines3(self):
        p = [0, 1, 2]
        q = [0, 1, 2]
        self.assertEqual(0, num_intersection(p, q))

    def test_three_lines4(self):
        p = [0, 1, 2]
        q = [2, 1, 0]
        self.assertEqual(3, num_intersection(p, q))

    def test_four_lines(self):
        p = [0, 1, 2, 3]
        q = [3, 2, 1, 0]
        self.assertEqual(6, num_intersection(p, q))

    def test_four_lines2(self):
        p = [0, 1, 2, 3]
        q = [3, 1, 2, 0]
        self.assertEqual(5, num_intersection(p, q))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Apr 3, 2021 \[Medium\] Add Bold Tag in String
---
> **Question:** Implement the function `embolden(s, lst)` which takes in a string s and list of substrings lst, and wraps all substrings in `s` with an HTML bold tag `<b>` and `</b>`.
>
> If two bold tags overlap or are contiguous, they should be merged.

**Example 1:**
```py
Input: s = 'abcdefg', lst = ['bc', 'ef'] 
Output: 'a<b>bc</b>d<b>ef</b>g'
```

**Example 2:**
```py
Input: s = 'abcdefg', lst = ['bcd', 'def']
Output: 'a<b>bcdef</b>g'
```

**Example 3:**
```py
Input: s = 'abcxyz123', lst = ['abc', '123']
Output:
'<b>abc</b>xyz<b>123</b>'
```

**Example 4:**
```py
Input: s = 'aaabbcc', lst = ['aaa','aab','bc']
Output: "<b>aaabbc</b>c"
```

**Solution with Trie:** [https://replit.com/@trsong/Add-Bold-Tag-in-String](https://replit.com/@trsong/Add-Bold-Tag-in-String)
```py
import unittest

def embolden(s, lst):
    trie = Trie()
    for word in lst:
        trie.insert(word)

    bold_position = {}
    for i, ch in enumerate(s):
        num_match_chars = trie.match(s, i)
        if num_match_chars > 0:
            bold_position[i] = i + num_match_chars - 1

    end_position = -1
    res = []
    for i in range(len(s)):
        if end_position < i and i in bold_position:
            res.append('<b>')
        res.append(s[i])

        end_position = max(end_position, bold_position.get(i, -1))
        end_position = max(end_position, bold_position.get(end_position + 1, -1))
        if i == end_position:
            res.append('</b>')
    return ''.join(res)


class Trie(object):
    def __init__(self):
        self.children = None
        self.terminated = False

    def insert(self, word):
        p = self
        for ch in word:
            p.children = p.children or {}
            if ch not in p.children:
                p.children[ch] = Trie()
            p = p.children[ch]
        p.terminated = True

    def match(self, word, start):
        p = self
        end = start
        for i in range(start, len(word)):
            ch = word[i]
            if not p or not p.children or ch not in p.children:
                break
            p = p.children[ch]
            if p and p.terminated:
                end = i + 1
        return end - start
            

class EmboldenSpec(unittest.TestCase):
    def test_example(self):
        s = 'abcdefg'
        lst = ['bc', 'ef'] 
        expected = 'a<b>bc</b>d<b>ef</b>g'
        self.assertEqual(expected, embolden(s, lst))

    def test_example2(self):
        s = 'abcdefg'
        lst = ['bcd', 'def']
        expected = 'a<b>bcdef</b>g'
        self.assertEqual(expected, embolden(s, lst))

    def test_example3(self):
        s = 'abcxyz123'
        lst = ['abc', '123']
        expected = '<b>abc</b>xyz<b>123</b>'
        self.assertEqual(expected, embolden(s, lst))

    def test_example4(self):
        s = 'aaabbcc'
        lst = ['aaa','aab','bc']
        expected = '<b>aaabbc</b>c'
        self.assertEqual(expected, embolden(s, lst))

    def test_non_pattern_match(self):
        s = 'abc123'
        lst = ['z']
        self.assertEqual(s, embolden(s, lst))

    def test_match_prefix(self):
        s = 'abc123'
        lst = ['a']
        expected = '<b>a</b>bc123'
        self.assertEqual(expected, embolden(s, lst))

    def test_match_suffix(self):
        s = 'abc123'
        lst = ['3']
        expected = 'abc12<b>3</b>' 
        self.assertEqual(expected, embolden(s, lst))

    def test_match_partial(self):
        s = 'abc123'
        lst = ['c133']
        expected = 'abc123' 
        self.assertEqual(expected, embolden(s, lst))

    def test_match_max_pattern(self):
        s = '000abc000'
        lst = ['a', 'ab', 'abc', 'abcd', 'abcde']
        expected = '000<b>abc</b>000' 
        self.assertEqual(expected, embolden(s, lst))

    def test_overlapping_patterns(self):
        s = '321abc123'
        lst = ['abc', '1abc1', '11abc11']
        expected = '32<b>1abc1</b>23'
        self.assertEqual(expected, embolden(s, lst))

    def test_overlapping_patterns2(self):
        s = '321abc123'
        lst = ['321abc', 'abc', 'abc123']
        expected = '<b>321abc123</b>'
        self.assertEqual(expected, embolden(s, lst))

    def test_non_overlapping_patterns(self):
        s = '321abc123'
        lst = ['3', '2', '1', 'ab', 'c']
        expected = '<b>321abc123</b>'
        self.assertEqual(expected, embolden(s, lst))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Apr 2, 2021 \[Hard\] Teams without Enemies
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

**Solution with Disjoint-Set(Union Find):** [https://replit.com/@trsong/Build-Teams-without-Enemies](https://replit.com/@trsong/Build-Teams-without-Enemies)
```py
import unittest
from collections import defaultdict

def team_without_enemies(students):
    enemy_map = defaultdict(defaultdict)
    for student, enemies in students.items():
        for enemy in enemies:
            enemy_map[student][enemy] = True
            enemy_map[enemy][student] = True

    uf = DisjointSet()
    for student, enemies in enemy_map.items():
        enemy0 = next(iter(enemies), None)
        for enemy in enemies:
            if uf.is_connected(student, enemy):
                return False
            uf.union(enemy, enemy0)
    
    team1_leader = next(iter(enemy_map), None)
    team1 = [student for student in students if uf.is_connected(student, team1_leader)]
    team2 = [student for student in students if not uf.is_connected(student, team1_leader)]
    return team1, team2
            

class DisjointSet(object):
    def __init__(self):
        self.parent = {}

    def find(self, p):
        self.parent[p] = self.parent.get(p, p)
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def union(self, p1, p2):
        root1 = self.find(p1)
        root2 = self.find(p2)
        if root1 != root2:
            self.parent[root1] = root2

    def is_connected(self, p1, p2):
        return self.find(p1) == self.find(p2)


class TeamWithoutEnemiesSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        expected_group1_set, expected_group2_set = set(expected[0]), set(expected[1])
        result_group1_set, result_group2_set = set(result[0]), set(result[1])
        outcome1 = (expected_group1_set == result_group1_set) and (expected_group2_set == result_group2_set)
        outcome2 = (expected_group2_set == result_group1_set) and (expected_group1_set == result_group2_set)
        self.assertTrue(outcome1 or outcome2)

    def test_example(self):
        students = {0: [3], 1: [2], 2: [1, 4], 3: [0, 4, 5], 4: [2, 3], 5: [3]}
        expected = ([0, 1, 4, 5], [2, 3])
        self.assert_result(expected, team_without_enemies(students))

    def test_example2(self):
        students = {
            0: [3],
            1: [2],
            2: [1, 3, 4],
            3: [0, 2, 4, 5],
            4: [2, 3],
            5: [3]
        }
        self.assertFalse(team_without_enemies(students))

    def test_empty_graph(self):
        students = {}
        expected = ([], [])
        self.assert_result(expected, team_without_enemies(students))

    def test_one_node_graph(self):
        students = {0: []}
        expected = ([0], [])
        self.assert_result(expected, team_without_enemies(students))

    def test_disconnect_graph(self):
        students = {0: [], 1: [0], 2: [3], 3: [4], 4: [2]}
        self.assertFalse(team_without_enemies(students))

    def test_square(self):
        students = {0: [1], 1: [2], 2: [3], 3: [0]}
        expected = ([0, 2], [1, 3])
        self.assert_result(expected, team_without_enemies(students))

    def test_k5(self):
        students = {0: [1, 2, 3, 4], 1: [2, 3, 4], 2: [3, 4], 3: [3], 4: []}
        self.assertFalse(team_without_enemies(students))

    def test_square2(self):
        students = {0: [3], 1: [2], 2: [1], 3: [0, 2]}
        expected = ([0, 2], [1, 3])
        self.assert_result(expected, team_without_enemies(students))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Apr 1, 2021 LC 16 \[Medium\] Closest to 3 Sum
---
> **Question:** Given a list of numbers and a target number n, find 3 numbers in the list that sums closest to the target number n. There may be multiple ways of creating the sum closest to the target number, you can return any combination in any order.

**Example:**
```py
Input: [2, 1, -5, 4], -1
Output: [-5, 1, 2]
Explanation: Closest sum is -5+1+2 = -2 OR -5+1+4 = 0
```

**Solution with Two-Pointers:** [https://replit.com/@trsong/Closest-to-3-Sum](https://replit.com/@trsong/Closest-to-3-Sum)
```py
import unittest

def closest_3_sum(nums, target):
    res = None
    n = len(nums)
    min_diff = float('inf')
    nums.sort()
    for i in range(n - 2):
        lo, hi = i + 1, n - 1
        while lo < hi:
            total = nums[i] + nums[lo] + nums[hi]
            distance = abs(total - target) 
            if distance < min_diff:
                min_diff = distance
                res = [nums[i], nums[lo], nums[hi]]

            if total > target:
                hi -= 1
            else:
                lo += 1
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


### Mar 31, 2021 \[Medium\] Ways to Form Heap with Distinct Integers
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


**Solution with DP:** [https://replit.com/@trsong/Total-Number-of-Ways-to-Form-Heap-with-Distinct-Integers](https://replit.com/@trsong/Total-Number-of-Ways-to-Form-Heap-with-Distinct-Integers)
```py
import unittest
import math

def form_heap_ways(n, cache=None):
    if n <= 2:
        return 1

    cache = cache or {}

    if n not in cache:
        height = int(math.log(n)) + 1
        bottom = n - (2 ** height - 1)

        left = 2 ** (height - 1) - 1 + min(2 ** (height - 1), bottom)
        right = n - left - 1
        cache[n] = choose(n - 1, left) * form_heap_ways(left, cache) * form_heap_ways(right, cache)

    return cache[n]
        

def choose(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)
    

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
    unittest.main(exit=False, verbosity=2)
```

### Mar 30, 2021 \[Easy\] Mouse Holes
---
> **Question:** Consider the following scenario: there are N mice and N holes placed at integer points along a line. Given this, find a method that maps mice to holes such that the largest number of steps any mouse takes is minimized.
>
> Each move consists of moving one mouse one unit to the left or right, and only one mouse can fit inside each hole.
>
> For example, suppose the mice are positioned at `[1, 4, 9, 15]`, and the holes are located at `[10, -5, 0, 16]`. In this case, the best pairing would require us to send the mouse at `1` to the hole at `-5`, so our function should return `6`.
 
**Solution with Greedy Algorithm:** [https://replit.com/@trsong/Min-Max-Step-Mouse-to-Holes](https://replit.com/@trsong/Min-Max-Step-Mouse-to-Holes)
```py
import unittest

def min_last_mouse_steps(mouse_positions, hole_positions):
    mouse_positions.sort()
    hole_positions.sort()
    res = 0
    for mouse, hole in zip(mouse_positions, hole_positions):
        res = max(res, abs(mouse - hole))
    return res


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
    unittest.main(exit=False, verbosity=2)
```

### Mar 29, 2021 LC 405 \[Easy\] Convert a Number to Hexadecimal
---
> **Question:** Given an integer, write an algorithm to convert it to hexadecimal. For negative integer, two’s complement method is used.

**Example 1:**
```py
Input: 26
Output: "1a"
```

**Example 2:**
```py
Input: -1
Output: "ffffffff"
```

**Solution:** [https://replit.com/@trsong/Convert-a-Number-to-Hexadecimal](https://replit.com/@trsong/Convert-a-Number-to-Hexadecimal)
```py
import unittest

def to_hex(num):
    if num == 0:
        return '0'

    DIGITS = "0123456789abcdef"
    if num < 0:
        num += 1 << 32
    
    res = []
    while num > 0:
        res.append(DIGITS[num % 16])
        num //= 16
    
    return ''.join(res[::-1])


class ToHexSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual('1a', to_hex(26))

    def test_example2(self):
        self.assertEqual('ffffffff', to_hex(-1))

    def test_zero(self):
        self.assertEqual('0', to_hex(0))

    def test_double_digits(self):
        self.assertEqual('cf', to_hex(0xcf))

    def test_double_digits2(self):
        self.assertEqual('43', to_hex(0x43))

    def test_double_digits3(self):
        self.assertEqual('a1', to_hex(0xa1))

    def test_double_digits4(self):
        self.assertEqual('4d', to_hex(0x4d))

    def test_one_digit(self):
        self.assertEqual('a', to_hex(0xa))

    def test_positive_boundary(self):
        self.assertEqual('7fffffff', to_hex(2147483647))

    def test_nagative_boundary(self):
        self.assertEqual('80000000', to_hex(-2147483648))

    def test_negative_number(self):
        self.assertEqual('fffffffe', to_hex(-2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Mar 28, 2021 \[Medium\] Minimum Number of Operations
---
> **Question:** You are only allowed to perform 2 operations:
> - either multiply a number by 2;
> - or subtract a number by 1. 
>
> Given a number `x` and a number `y`, find the minimum number of operations needed to go from `x` to `y`.

**Solution with BFS:** [https://replit.com/@trsong/Find-Min-Number-of-Operations](https://replit.com/@trsong/Find-Min-Number-of-Operations)
```py
import unittest

def min_operations(start, end):
    queue = [start]
    visited = set()
    step = 0

    while queue:
        for _ in xrange(len(queue)):
            cur = queue.pop(0)
            if end == cur:
                return step
            elif cur in visited:
                continue
            else:
                visited.add(cur)

            for child in [cur - 1, cur * 2]:
                if child not in visited:
                    queue.append(child)
        step += 1
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
    unittest.main(exit=False, verbosity=2)
```

### Mar 27, 2021 \[Medium\] Smallest Window Contains Every Distinct Character
---
> **Question:** Given a string, find the length of the smallest window that contains every distinct character. Characters may appear more than once in the window.
>
> For example, given `"jiujitsu"`, you should return `5`, corresponding to the final five letters.


**Solution with Sliding Window:** [https://replit.com/@trsong/Smallest-Window-Contains-Every-Distinct-Character](https://replit.com/@trsong/Smallest-Window-Contains-Every-Distinct-Character)
```py
import unittest

def find_min_window_all_char(s):
    char_set_size = len(set(s))
    char_freq = {}
    res = len(s)

    start = 0
    for end, incoming_char in enumerate(s):
        char_freq[incoming_char] = char_freq.get(incoming_char, 0) + 1
        while len(char_freq) == char_set_size:
            res = min(res, end - start + 1)
            outgoing_char = s[start]
            char_freq[outgoing_char] -= 1
            if char_freq[outgoing_char] == 0:
                del char_freq[outgoing_char]
            start += 1
    return res


class FindMinWindowAllCharSpec(unittest.TestCase):
    def test_example(self):
        s = 'jiujitsu'
        expected = len('jitsu')
        self.assertEqual(expected, find_min_window_all_char(s))

    def test_suffix_has_result(self):
        s = "ADOBECODEBANC"
        expected = len('ODEBANC')
        self.assertEqual(expected, find_min_window_all_char(s))

    def test_prefix_has_result(self):
        s = "CANADA"
        expected = len("CANAD")
        self.assertEqual(expected, find_min_window_all_char(s))

    def test_string_has_no_duplicates(self):
        s = 'USD'
        expected = len(s)
        self.assertEqual(expected, find_min_window_all_char(s))
    
    def test_has_to_include_entire_string(self):
        s = "BANANAS"
        expected = len(s)
        self.assertEqual(expected, find_min_window_all_char(s))

    def test_non_letter_char(self):
        s = "AB_AABB_AB_BAB_ABB_BB_BACAB"
        expected = len('_BAC')
        self.assertEqual(expected, find_min_window_all_char(s))

    def test_result_in_middle(self):
        s = "AAAACCCCCBADDBBAADBBAAADDDCCBBA"
        expected = len('CBAD')
        self.assertEqual(expected, find_min_window_all_char(s))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 26, 2021 \[Easy\] Common Characters
---
> **Question:** Given n strings, find the common characters in all the strings. In simple words, find characters that appear in all the strings and display them in alphabetical order or lexicographical order.

**Example:**
```py
common_characters(['google', 'facebook', 'youtube'])
# ['e', 'o']
```

**Solution:** [https://replit.com/@trsong/Find-All-Common-Characters](https://replit.com/@trsong/Find-All-Common-Characters)
```py
import unittest

def find_common_characters(words):
    char_freq = {}
    for index, word in enumerate(words):
        for ch in word:
            if char_freq.get(ch, 0) == index:
                char_freq[ch] = index + 1
    
    return sorted(filter(lambda ch: char_freq[ch] == len(words), char_freq.keys()))


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
    unittest.main(exit=False, verbosity=2)
```

### Mar 25, 2021 \[Hard\] Maximum Spanning Tree
--- 
> **Question:** Recall that the minimum spanning tree is the subset of edges of a tree that connect all its vertices with the smallest possible total edge weight.
>	
> Given an undirected graph with weighted edges, compute the maximum weight spanning tree.

**My thoughts:** Both Kruskal's and Prim's Algorithm works for this question. The idea is to flip all edge weight into negative and then apply either of previous algorithms until find a spanning tree

**Solution with Kruskal's Algorithm:** [https://replit.com/@trsong/Calculate-Maximum-Spanning-Tree](https://replit.com/@trsong/Calculate-Maximum-Spanning-Tree)
```py
import unittest
import sys


class DisjointSet(object):
    def __init__(self):
        self.parent = {}
        
    def find(self, p):
        self.parent[p] = self.parent.get(p, p)
        while self.parent[p] != p:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p
        
    def union(self, p1, p2):
        root1 = self.find(p1)
        root2 = self.find(p2)
        if root1 != root2:
            self.parent[root1] = root2
            
    def is_connected(self, p1, p2):
        return self.find(p1) == self.find(p2)
    

def max_spanning_tree(vertices, edges):
    res = []
    edges.sort(reverse=True, key=lambda uvw: uvw[-1])
    
    uf = DisjointSet()
    for u, v, _ in edges:
        if uf.is_connected(u, v):
            continue
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
    unittest.main(exit=False, verbosity=2)
``` 


### Mar 24, 2021 \[Hard\] Optimal Strategy For Coin Game
---
> **Question:** In front of you is a row of `N` coins, with values `v_1, v_2, ..., v_n`.
>
> You are asked to play the following game. You and an opponent take turns choosing either the first or last coin from the row, removing it from the row, and receiving the value of the coin.
>
> Write a program that returns the maximum amount of money you can win with certainty, if you move first, assuming your opponent plays optimally.

**Solution with Minimax:** [https://replit.com/@trsong/Optimal-Strategy-For-Coin-Game](https://replit.com/@trsong/Optimal-Strategy-For-Coin-Game)
```py
import unittest

def max_coin_game_profit(coins):
    if not coins:
        return 0
    
    n = len(coins)
    # Let dp[i][j] represents result for coins[i:j+1]
    # dp[i][j] = max {
    #                  coins[i] + min(dp[i+2][j], dp[i+1][j-1]),
    #                  coins[j] + min(dp[i+1][j-1], dp[i][j-2])
    #                }
    # Remember your opponent always make optimal move that leaves you min of profit for subproblem
    dp = [[None for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        dp[i][i] = coins[i]
        
    for i in range(1, n):
        dp[i-1][i] = max(coins[i-1], coins[i])
        
    for offset in range(2, n):
        for i in range(n - offset):
            j = i + offset
            dp[i][j] = max(
                coins[i] + min(dp[i+2][j], dp[i+1][j-1]), 
                coins[j] + min(dp[i+1][j-1], dp[i][j-2]))
    
    return dp[0][n-1]


class MaxCoinGameProfitSpec(unittest.TestCase):
    def test_greedy_not_work(self):
        coins = [10, 24, 5, 9]
        expected = 9 + 24  # 9 vs 10, 24 vs 5
        self.assertEqual(expected, max_coin_game_profit(coins))
        
    def test_empty_coins(self):
        self.assertEqual(0, max_coin_game_profit([]))
        
    def test_one_coin(self):
        self.assertEqual(42, max_coin_game_profit([42]))
        
    def test_two_coins(self):
        coins = [100, 1]
        expected = 100
        self.assertEqual(expected, max_coin_game_profit(coins))
        
    def test_local_max_profit_is_global_max(self):
        coins = [100, 0, 100, 0]
        expected = 200
        self.assertEqual(expected, max_coin_game_profit(coins))
        
    def test_choose_larger_first_coin(self):
        coins = [5, 3, 7, 10]
        expected = 10 + 5  # 10 vs 7, 5 vs 3
        self.assertEqual(expected, max_coin_game_profit(coins))
        
    def test_choose_smaller_first_coin(self):
        coins = [8, 15, 3, 7]
        expected = 7 + 15  # 7 vs 8, 15 vs 3
        self.assertEqual(expected, max_coin_game_profit(coins))
    
    def test_opponent_always_make_optimal_move(self):
        coins = [20, 30, 2, 1, 3, 10]
        expected = 10 + 1 + 30  # 10 vs 3, 1 vs 20, 30 vs 2
        self.assertEqual(expected, max_coin_game_profit(coins))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Mar 23, 2021 \[Medium\] Multiply Large Numbers Represented as Strings
---
> **Question:** Given two strings which represent non-negative integers, multiply the two numbers and return the product as a string as well. You should assume that the numbers may be sufficiently large such that the built-in integer type will not be able to store the input.

**Example 1:**
```py
Input: num1 = "2", num2 = "3"
Output: "6"
```

**Example 2:**
```py
Input: num1 = "123", num2 = "456"
Output: "56088"
```

**Solution:** [https://replit.com/@trsong/Multiply-Large-Numbers-Represented-as-Strings](https://replit.com/@trsong/Multiply-Large-Numbers-Represented-as-Strings)
```py
import unittest

def string_multiply(s1, s2):
    rev1, rev2 = map(int, reversed(s1)), map(int, reversed(s2))
    
    set1, set2 = set(rev1), set(rev2)
    if len(set1) < len(set2):
        rev1, rev2 = rev2, rev1
        set1, set2 = set2, set1

    cache = { digit: multiply_digit(rev1, digit) for digit in set2 }
    res = [0] * (len(rev1) + len(rev2))
    for offset, digit in enumerate(rev2):
        index = offset
        carry = 0
        partial_res = iter(cache[digit])
        num = next(partial_res, None)

        while carry > 0 or num is not None:
            res[index] += carry + (num if num is not None else 0)
            carry = res[index] // 10
            res[index] %= 10
            index += 1
            num = next(partial_res, None)
    
    raw_result = ''.join(map(str, reversed(res))).lstrip('0')
    return raw_result or '0'


def multiply_digit(s, digit):
    res = []
    carry = 0
    for num in s:
        digit_res = digit * num + carry
        res.append(digit_res % 10)
        carry = digit_res // 10

    if carry > 0:
        res.append(carry)

    return res


class StringMultiplySpec(unittest.TestCase):
    def test_example(self):
        s1 = '2'
        s2 = '3'
        expected = '6'
        self.assertEqual(expected, string_multiply(s1, s2))

    def test_example2(self):
        s1 = '123'
        s2 = '456'
        expected = '56088'
        self.assertEqual(expected, string_multiply(s1, s2))

    def test_multiply_single_digit(self):
        s1 = '9'
        s2 = '9'
        expected = '81'
        self.assertEqual(expected, string_multiply(s1, s2))

    def test_multiply_by_zero(self):
        s1 = '0'
        s2 = '0'
        expected = '0'
        self.assertEqual(expected, string_multiply(s1, s2))

    def test_multiply_by_zero2(self):
        s1 = '0'
        s2 = '9999'
        expected = '0'
        self.assertEqual(expected, string_multiply(s1, s2))
    
    def test_multiply_by_one(self):
        s1 = '1'
        s2 = '9999'
        expected = '9999'
        self.assertEqual(expected, string_multiply(s1, s2))

    def test_input_with_different_lengths(self):
        s1 = '1024'
        s2 = '999'
        expected = '1022976'
        self.assertEqual(expected, string_multiply(s1, s2))

    def test_large_numbers(self):
        s1 = '1235421415454545454545454544'
        s2 = '1714546546546545454544548544544545'
        expected = '2118187521397235888154583183918321221520083884298838480662480'
        self.assertEqual(expected, string_multiply(s1, s2))
    

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 22, 2021 \[Medium\] Paint House
---
> **Question:** A builder is looking to build a row of N houses that can be of K different colors. He has a goal of minimizing cost while ensuring that no two neighboring houses are of the same color.
>
> Given an N by K matrix where the n-th row and k-th column represents the cost to build the n-th house with k-th color, return the minimum cost which achieves this goal.

**Solution with DP:** [https://replit.com/@trsong/Min-Cost-to-Paint-House](https://replit.com/@trsong/Min-Cost-to-Paint-House)
```py
import unittest

def min_paint_houses_cost(paint_cost):
    if not paint_cost or not paint_cost[0]:
        return 0

    num_house, num_color = len(paint_cost), len(paint_cost[0])
    # Let dp[n][c] represents min paint house for n houses and k colors
    # dp[n][c] = min(dp[n-1][k]) + paint_cost[n-1][c] where k != c
    dp = [[float('inf') for _ in xrange(num_color)] for _ in xrange(num_house + 1)]
    for c in xrange(num_color):
        dp[0][c] = 0

    for i in xrange(1, num_house + 1):
        first_min = second_min = float('inf')
        first_min_color = None

        for color, prev_cost in enumerate(dp[i - 1]):
            if prev_cost < first_min:
                second_min = first_min
                first_min = prev_cost
                first_min_color = color
            elif prev_cost < second_min:
                second_min = prev_cost

        for color in xrange(num_color):
            if color == first_min_color:
                dp[i][color] = second_min + paint_cost[i - 1][color]
            else:
                dp[i][color] = first_min + paint_cost[i - 1][color]
    
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

### Mar 21, 2021 \[Easy\] Determine If Singly Linked List is Palindrome
---
> **Question:** Determine whether a singly linked list is a palindrome. 
>
> For example, `1 -> 4 -> 3 -> 4 -> 1` returns `True` while `1 -> 4` returns `False`. 

**My thoughts:** An easy way to solve this problem is to use stack. But that is not memory efficient. An alternative way is to flip the second half of list and check each element against fist half. 

**Solution with Fast-Slow Pointers:** [https://replit.com/@trsong/Determine-If-Singly-Linked-List-is-Palindrome](https://replit.com/@trsong/Determine-If-Singly-Linked-List-is-Palindrome)
```py
import unittest

def is_palindrome(lst):
    if not lst:
        return True

    slow = fast = lst
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    if fast.next:
        fast = fast.next
        reverse(slow.next)
    else:
        reverse(slow)

    slow.next = None
    return is_equal(lst, fast)


def reverse(lst):
    prev = None
    cur = lst
    while cur:
        next = cur.next
        cur.next = prev
        prev = cur
        cur = next
    return prev


def is_equal(l1, l2):
    while l1 and l2:
        if l1.val != l2.val:
            return False
        l1 = l1.next
        l2 = l2.next
    return l1 == l2 == None


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

### Mar 20, 2021 LC 679 \[Hard\] 24 Game
---
> **Question:** The 24 game is played as follows. You are given a list of four integers, each between 1 and 9, in a fixed order. By placing the operators +, -, *, and / between the numbers, and grouping them with parentheses, determine whether it is possible to reach the value 24.
>
> For example, given the input `[5, 2, 7, 8]`, you should return `True`, since `(5 * 2 - 7) * 8 = 24`.
>
> Write a function that plays the 24 game.


**Solution with Backtracking:** [https://replit.com/@trsong/24-Game](https://replit.com/@trsong/24-Game)
```py
import unittest

def play_24_game(cards):
    if len(cards) == 1:
        return abs(cards[0] - 24) < 1e-3
    elif len(cards) == 2:
        combined1 = apply_ops(cards[0], cards[1]) 
        combined2 = apply_ops(cards[1], cards[0])
        return any(play_24_game([card]) for card in combined1 + combined2)
    else:
        for i in range(len(cards)):
            for j in range(len(cards)):
                if i == j:
                    continue

                for combine_cards in apply_ops(cards[i], cards[j]):
                    remaining_cards = [combine_cards] + [cards[k] for k in range(len(cards)) if k != i and k != j]
                    if play_24_game(remaining_cards):
                        return True
        return False


def apply_ops(num1, num2):
    return [num1 + num2, num1 - num2, num1 * num2, num1 / num2 if num2 != 0 else float('inf')]


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


### Mar 19, 2021 \[Easy\] Swap Even and Odd Nodes
---
> **Question:** Given the head of a singly linked list, swap every two nodes and return its head.
>
> **Note:** Make sure it’s acutally nodes that get swapped not value.

**Example:**
```py
Given 1 -> 2 -> 3 -> 4, return 2 -> 1 -> 4 -> 3.
```

**Solution:** [https://replit.com/@trsong/Swap-Every-Even-and-Odd-Nodes-in-Linked-List](https://replit.com/@trsong/Swap-Every-Even-and-Odd-Nodes-in-Linked-List)
```py
import unittest

def swap_list(lst):
    prev = dummy = ListNode(-1, lst)
    while prev and prev.next and prev.next.next:
        first = prev.next
        second = prev.next.next

        first.next = second.next
        second.next = first
        prev.next = second

        prev = prev.next.next
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


### Mar 18, 2021 \[Medium\] K-th Missing Number in Sorted Array
---
> **Question:** Given a sorted without any duplicate integer array, define the missing numbers to be the gap among numbers. Write a function to calculate K-th missing number. If such number does not exist, then return null.
> 
> For example, original array: `[2,4,7,8,9,15]`, within the range defined by original array, all missing numbers are: `[3,5,6,10,11,12,13,14]`
> - the 1st missing number is 3,
> - the 2nd missing number is 5,
> - the 3rd missing number is 6


**My thoughts:** An array without any gap must be continuous and should look something like the following:

```py
[0, 1, 2, 3, 4, 5, 6, 7]
[8, 9, 10, 11]
...
```

Which can be easily verified by using `last - first` element and check the result against length of array.

Likewise, given any index, treat the element the index refer to as the last element, we can easily the number of missing elements by using the following:

```py
def count_missing(index):
    return nums[i] - nums[0] - i
```

Thus, ever since it's `O(1)` to verify the total missing numbers on the left against the k-th missing number, we can always use binary search to shrink the searching space into half. 

> Tips: do you know the template for binary searching the array with lots of duplicates?
> 
> eg. Find the index of first 1 in [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3].

```py
lo = 0
hi = n - 1
while lo < hi:
    mid = lo + (hi - lo) / 2  # avoid overflow in some other language like Java
    if arr[mid] < target:     # this is the condition. When exit the loop, lo will stop at first element that not statify the condition
        lo = mid + 1          # why + 1?  lo = (lo + hi) / 2 if hi = lo + 1, we will have lo = (2 * lo + 1) / 2 == lo
    else:
        hi = mid
return lo
```

Note: above template will return the index of first element that not statify the condition. e.g. If `target == 1`, the first elem that >= 1 is 1 at index 3. If `target == 2`, the first elem that >= 2 is 3 at position 10.

> Note, why we need to know above template? Like how does it help to solve this question?

Let's consider the following case:  `find_kth_missing_number([3, 4, 8, 9, 10, 11, 12], 2)`.

```py
[3, 4, 8, 9, 10, 11, 12]
```

Above source array can be converted to the following count_missing array:

```py
[0, 0, 3, 3, 3, 3, 3]
```

Above array represents how many missing numbers are on the left of current element. 

Since we are looking at the 2-nd missing number. The first element that not less than target is 3 at position 2. Then using that position we can backtrack to get first element that has 3 missing number on the left is 8. Finally, since 8 has 3 missing number on the left, then we imply that 6 must has 2 missing number on the left which is what we are looking for.


**Solution with Binary Search:** [https://replit.com/@trsong/Find-K-th-Missing-Number-in-Sorted-Array](https://replit.com/@trsong/Find-K-th-Missing-Number-in-Sorted-Array)
```py
import unittest

def find_kth_missing_number(nums, k):
    n = len(nums)
    if not nums or k <= 0 or count_missing(nums, n - 1) < k :
        return None
        
    lo = 0
    hi = len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if count_missing(nums, mid) < k:
            lo = mid + 1
        else:
            hi = mid

    delta = count_missing(nums, lo) - k
    return nums[lo] - 1 - delta


def count_missing(nums, i):
    return nums[i] - nums[0] - i


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

### Mar 17, 2021 \[Medium\] K Closest Elements
---
> **Question:** Given a list of sorted numbers, and two integers `k` and `x`, find `k` closest numbers to the pivot `x`.

**Example:**
```py
closest_nums([1, 3, 7, 8, 9], 3, 5)  # gives [7, 3, 8]
```

**Solution with Binary Search:** [https://replit.com/@trsong/Find-K-Closest-Elements-in-a-Sorted-Array](https://replit.com/@trsong/Find-K-Closest-Elements-in-a-Sorted-Array)
```py
import unittest

def closest_nums(nums, k, target):
    right = binary_search(nums, target)
    left = right - 1
    res = []
    for _ in xrange(k):
        if right >= len(nums) or left >= 0 and target - nums[left] <  nums[right] - target:
            res.append(nums[left])
            left -= 1
        else:
            res.append(nums[right])
            right += 1
    return res


def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


class ClosestNumSpec(unittest.TestCase):
    def test_example(self):
        k, x, nums = 3, 5, [1, 3, 7, 8, 9]
        expected = [7, 3, 8]
        self.assertEqual(sorted(expected), sorted(closest_nums(nums, k, x)))

    def test_example2(self):
        k, x, nums = 5, 35, [12, 16, 22, 30, 35, 39, 42, 45, 48, 50, 53, 55, 56]
        expected = [30, 39, 35, 42, 45]
        self.assertEqual(sorted(expected), sorted(closest_nums(nums, k, x)))

    def test_empty_list(self):
        self.assertEqual([], closest_nums([], 0, 42))
    
    def test_entire_list_qualify(self):
        k, x, nums = 6, -1000, [0, 1, 2, 3, 4, 5]
        expected = [0, 1, 2, 3, 4, 5]
        self.assertEqual(sorted(expected), sorted(closest_nums(nums, k, x)))
    
    def test_entire_list_qualify2(self):
        k, x, nums = 2, 1000, [0, 1]
        expected = [0, 1]
        self.assertEqual(sorted(expected), sorted(closest_nums(nums, k, x)))

    def test_closest_number_on_both_sides(self):
        k, x, nums = 3, 5, [1, 5, 6, 10, 20]
        expected = [1, 5, 6]
        self.assertEqual(sorted(expected), sorted(closest_nums(nums, k, x)))

    def test_closest_number_from_head_of_list(self):
        k, x, nums = 2, -1, [0, 1, 2, 3]
        expected = [0, 1]
        self.assertEqual(sorted(expected), sorted(closest_nums(nums, k, x)))

    def test_closest_number_from_tail_of_list(self):
        k, x, nums = 4, 999, [0, 1, 2, 3]
        expected = [0, 1, 2, 3]
        self.assertEqual(sorted(expected), sorted(closest_nums(nums, k, x)))

    def test_contains_duplicate_numbers(self):
        k, x, nums = 5, 3, [1, 1, 1, 1, 3, 3, 3, 4, 4]
        expected = [3, 3, 3, 4, 4]
        self.assertEqual(sorted(expected), sorted(closest_nums(nums, k, x)))

    def test_(self):
        k, x, nums = 2, 99,[1, 2, 100]
        expected = [2, 100]
        self.assertEqual(sorted(expected), sorted(closest_nums(nums, k, x)))
   

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 16, 2021 LC 30 \[Hard\] Substring with Concatenation of All Words
---
> **Question:**  Given a string `s` and a list of words words, where each word is the same length, find all starting indices of substrings in s that is a concatenation of every word in words exactly once. The order of the indices does not matter.

**Example 1:**
```py
Input: s = "dogcatcatcodecatdog", words = ["cat", "dog"]
Output: [0, 13]
Explanation: "dogcat" starts at index 0 and "catdog" starts at index 13.
```

**Example 2:**
```py
Input: s = "barfoobazbitbyte", words = ["dog", "cat"]
Output: []
Explanation: there are no substrings composed of "dog" and "cat" in s.
```

**Solution with Trie:** [https://replit.com/@trsong/Substring-with-Concatenation-of-All-Words](https://replit.com/@trsong/Substring-with-Concatenation-of-All-Words)
```py
import unittest

def find_permutation_as_substring(s, words):
    if not s or not words:
        return []

    word_count = len(words)
    word_length = len(words[0])
    trie = Trie()
    for word in words:
        trie.insert(word)

    res = []
    for i in xrange(len(s) - word_count * word_length + 1):
        if is_substring(s, trie, i, word_count, word_length):
            res.append(i)
    return res


def is_substring(s, trie, start, word_count, word_length):
    nodes = []
    for step in xrange(word_count):
        offset = step * word_length
        word = s[offset + start: offset + start + word_length]
        node = trie.find(word)
        if node is None or node.count <= 0:
            break
        node.count -= 1
        nodes.append(node)

    for node in nodes:
        node.count += 1
    
    return len(nodes) == word_count


class Trie(object):
    def __init__(self):
        self.children = None
        self.count = 0
    
    def insert(self, word):
        p = self
        for ch in word:
            p.children = p.children or {}
            if ch not in p.children:
                p.children[ch] = Trie()
            p = p.children[ch]
        p.count += 1

    def find(self, word):
        p = self
        for ch in word:
            if not p or not p.children or ch not in p.children:
                return None
            p = p.children[ch]
        return p


class FindPermutationAsSubstring(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(sorted(expected), sorted(result))

    def test_example(self):
        s = "dogcatcatcodecatdog"
        words = ["cat", "dog"]
        # catdog, dogcat
        expected = [0, 13]
        self.assert_result(expected, find_permutation_as_substring(s, words))

    def test_example2(self):
        s = "barfoobazbitbyte"
        words = ["dog", "cat"]
        expected = []
        self.assert_result(expected, find_permutation_as_substring(s, words))

    def test_words_with_two_elements(self):
        s = "barfoothefoobarman"
        words = ["foo", "bar"]
        expected = [0, 9]
        self.assert_result(expected, find_permutation_as_substring(s, words))
    
    def test_does_not_exist_solution(self):
        s = "wordgoodgoodgoodbestword"
        words = ["word","good","best","word"]
        expected = []
        self.assert_result(expected, find_permutation_as_substring(s, words))

    def test_words_with_three_elements(self):
        s = "barfoofoobarthefoobarman"
        words = ["bar","foo","the"]
        expected = [6,9,12]
        self.assert_result(expected, find_permutation_as_substring(s, words))
    
    def test_empty_string(self):
        self.assert_result([], find_permutation_as_substring("", ["a"]))

    def test_empty_words(self):
        self.assert_result([], find_permutation_as_substring("abc", []))

    def test_words_with_one_element(self):
        s = "01020304"
        words = ["0"]
        expected = [0, 2, 4, 6]
        self.assert_result(expected, find_permutation_as_substring(s, words))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 15, 2021 \[Medium\] Maze Paths
---
> **Question:**  A maze is a matrix where each cell can either be a 0 or 1. A 0 represents that the cell is empty, and a 1 represents a wall that cannot be walked through. You can also only travel either right or down.
>
> Given a nxm matrix, find the number of ways someone can go from the top left corner to the bottom right corner. You can assume the two corners will always be 0.

**Example:**
```py
Input: [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
# 0 1 0
# 0 0 1
# 0 0 0
Output: 2
The two paths that can only be taken in the above example are: down -> right -> down -> right, and down -> down -> right -> right.
```

**Solution with DP:** [https://replit.com/@trsong/Count-Number-of-Maze-Paths](https://replit.com/@trsong/Count-Number-of-Maze-Paths)
```py
import unittest

def count_maze_path(grid):
    if not grid or not grid[0]:
        return 0

    n, m = len(grid), len(grid[0])
    # Let grid[r][c] represents # of ways from top left to cell (r, c)
    # grid[r][c] = grid[r-1][c] + grid[r][c-1] 
    grid[0][0] = -1
    for r in xrange(n):
        for c in xrange(m):
            if grid[r][c] != 0:
                continue

            left_val = grid[r - 1][c] if r > 0 and grid[r - 1][c] < 0 else 0
            top_val = grid[r][c - 1] if c > 0 and grid[r][c - 1] < 0 else 0
            grid[r][c] += left_val + top_val

    return abs(grid[-1][-1]) 


class CountMazePathSpec(unittest.TestCase):
    def test_example(self):
        grid = [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ]
        self.assertEqual(2, count_maze_path(grid))
    
    def test_one_cell_grid(self):
        self.assertEqual(1, count_maze_path([[0]]))

    def test_4x4_grid(self):
        grid = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        self.assertEqual(4, count_maze_path(grid))

    def test_4x4_grid2(self):
        grid = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ]
        self.assertEqual(16, count_maze_path(grid))
    
    def test_non_square_grid(self):
        grid = [
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 0]
        ]
        self.assertEqual(1, count_maze_path(grid))

    def test_alternative_path_exists(self):
        grid = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        self.assertEqual(18, count_maze_path(grid))
    
    def test_all_paths_are_blocked(self):
        grid = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0]
        ]
        self.assertEqual(0, count_maze_path(grid))

    def test_no_obstacles(self):
        grid = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        self.assertEqual(6, count_maze_path(grid))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Mar 14, 2021  \[Medium\] Split a Binary Search Tree
---
> **Question:** Given a binary search tree (BST) and a value s, split the BST into 2 trees, where one tree has all values less than or equal to s, and the other tree has all values greater than s while maintaining the tree structure of the original BST. You can assume that s will be one of the node's value in the BST. Return both tree's root node as a tuple.

**Example:**
```py
Given the following tree, and s = 2
     3
   /   \
  1     4
   \     \
    2     5

Split into two trees:
 1    And   3
  \          \
   2          4
               \
                5
```


**Solution:** [https://replit.com/@trsong/Split-BST-into-Two-BSTs](https://replit.com/@trsong/Split-BST-into-Two-BSTs)
```py
import unittest

def split_bst(root, s):
    if root is None:
        return None, None
    elif root.val <= s:
        root.right, right_res = split_bst(root.right, s)
        return root, right_res
    else:
        left_res, root.left = split_bst(root.left, s)
        return left_res, root


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
         other.val == self.val and
         other.left == self.left and
         other.right == self.right)

    def __repr__(self):
        return "TreeNode(%s, %s, %s)" % (self.val, self.left, self.right)


class SplitTreeSpec(unittest.TestCase):
    def test_example(self):
        """
             3
           /   \
          1     4
           \     \
            2     5

        Split into:
         1    And   3
          \          \
           2          4
                       \
                        5
        """
        original_left = TreeNode(1, right=TreeNode(2))
        original_right = TreeNode(4, right=TreeNode(5))
        original_tree = TreeNode(3, original_left, original_right)

        split_tree1 = TreeNode(1, right=TreeNode(2))
        split_tree2 = TreeNode(3, right=TreeNode(4, right=TreeNode(5)))
        expected = (split_tree1, split_tree2)
        self.assertEqual(expected, split_bst(original_tree, s=2)) 

    def test_empty_tree(self):
        self.assertEqual((None, None), split_bst(None, 42))

    def test_split_one_tree_into_empty(self):
        """
          2
         / \
        1   3 
        """
        
        original_tree = TreeNode(2, TreeNode(1), TreeNode(3))
        split_tree = TreeNode(2, TreeNode(1), TreeNode(3))
        self.assertEqual((split_tree, None), split_bst(original_tree, s=3)) 
        self.assertEqual((None, split_tree), split_bst(original_tree, s=0))

    def test_split_tree_change_original_tree_layout(self):
        """
             4
           /  \
          2     6
         / \   / \
        1   3 5   7
        
        Split into:
                 4
                /  \
          2    3    6
         /         / \
        1         5   7      
        """

        original_left = TreeNode(2, TreeNode(1), TreeNode(3))
        original_right = TreeNode(6, TreeNode(5), TreeNode(7))
        original_tree = TreeNode(4, original_left, original_right)

        split_tree1 = TreeNode(2, TreeNode(1))
        split_tree2_right = TreeNode(6, TreeNode(5), TreeNode(7))
        split_tree2 = TreeNode(4, TreeNode(3), split_tree2_right)
        expected = (split_tree1, split_tree2)
        self.assertEqual(expected, split_bst(original_tree, s=2)) 

    def test_split_tree_change_original_tree_layout2(self):
        """
             4
           /  \
          2     6
         / \   / \
        1   3 5   7
        
        Split into:
             4
           /  \
          2    5    6
         / \         \
        1   3         7     
        """

        original_left = TreeNode(2, TreeNode(1), TreeNode(3))
        original_right = TreeNode(6, TreeNode(5), TreeNode(7))
        original_tree = TreeNode(4, original_left, original_right)

        split_tree1_left = TreeNode(2, TreeNode(1), TreeNode(3))
        split_tree1 = TreeNode(4, split_tree1_left, TreeNode(5))
        split_tree2 = TreeNode(6, right=TreeNode(7))
        expected = (split_tree1, split_tree2)
        self.assertEqual(expected, split_bst(original_tree, s=5)) 
        

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 13, 2021 \[Hard\] Critical Routers (Articulation Point)
--- 
> **Question:** You are given an undirected connected graph. An articulation point (or cut vertex) is defined as a vertex which, when removed along with associated edges, makes the graph disconnected (or more precisely, increases the number of connected components in the graph). The task is to find all articulation points in the given graph.

**Example1:**
```py
Input: vertices = 4, edges = [[0, 1], [1, 2], [2, 3]]
Output: [1, 2]
Explanation: 
Removing either vertex 0 or 3 along with edges [0, 1] or [2, 3] does not increase number of connected components. 
But removing 1, 2 breaks graph into two components.
```

**Example2:**
```py
Input: vertices = 5, edges = [[0, 1], [0, 2], [1, 2], [0, 3], [3, 4]]
Output: [0, 3]
```

**My thoughts:** An articulation point also known as cut vertex must be one end of a bridge. A bridge is an edge without which number of connected component increased, i.e. break the connected graph into multiple components. So basically, a bridge is an edge without which the graph will cease to be connected. Recall that the way to detect if an edge (u, v) is a bridge is to find if there is alternative path from v to u without going through (u, v). Record time stamp of discover time as well as earliest discover time among all ancestor will do the trick.

As we already know how to find a bridge, an articulation pointer (cut vertex) is just one end (or both ends) of such bridge that is not a leaf (has more than 1 child). Why is that? A leaf can never be an articulation point as without that point, the total connected component in a graph won’t change.

For example, in graph 0 - 1 - 2 - 3, all edges are bridge edges. Yet, only vertex 1 and 2 qualify for articulation point, beacuse neither 1 and 2 is leaf node.

How to find bridge?

In order to tell if an edge u, v is a bridge, after finishing processed all childen of u, we check if v ever connect to any ancestor of u. That can be done by compare the discover time of u against the earliest ancestor discover time of v (while propagate to v, v might connect to some ancestor, we just store the ealiest discover of among those ancestors). If v’s earliest ancestor discover time is greater than discover time of u, ie. v doesn’t have access to u’s ancestor, then edge u,v must be a bridge, ‘coz it seems there is not way for v to connect to u other than edge u,v. By definition that edge is a bridge.

**Solution with DFS:** [https://replit.com/@trsong/Find-the-Critical-Routers-Articulation-Point](https://replit.com/@trsong/Find-the-Critical-Routers-Articulation-Point)
```py
import unittest

class NodeState:
    UNVISITED = 0
    VISITING = 1
    VISITED = 2


def critial_rounters(vertices, edges):
    if vertices <= 1 or not edges:
        return []

    neighbors = [[] for _ in xrange(vertices)]
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)

    node_states = [NodeState.UNVISITED] * vertices
    discover_time = [float('inf')] * vertices
    ancesor_time = [float('inf')] * vertices  # min discover time of non-parent ancesor
    time = 0
    res = set()
    stack = [(0, None)]

    while stack:
        u, parent_u = stack[-1]
        if node_states[u] is NodeState.VISITED:
            stack.pop()
        elif node_states[u] is NodeState.VISITING:
            node_states[u] = NodeState.VISITED
            for v in neighbors[u]:
                if node_states[v] is not NodeState.VISITED:
                    continue
                
                if discover_time[u] < ancesor_time[v]:
                    # edge u-v is a bridge and both side could be articulation points. And articulation point is non-leaf
                    if len(neighbors[u]) > 1:
                        res.add(u)

                    if len(neighbors[v]) > 1:
                        res.add(v)
                ancesor_time[u] = min(ancesor_time[u], ancesor_time[v])
        else:
            # When node_states[u] is NodeState.UNVISITED
            node_states[u] = NodeState.VISITING
            ancesor_time[u] = discover_time[u] = time
            time += 1
            for v in neighbors[u]:
                if node_states[v] is NodeState.UNVISITED:
                    stack.append((v, u))
                elif v != parent_u:
                    # edge u-v is a non-parent back-edge, v is a visiting ancestor
                    ancesor_time[u] = min(ancesor_time[u], discover_time[v])


    return list(res)


class CritialRouterSpec(unittest.TestCase):
    def validate_routers(self, expected, result):
        self.assertEqual(sorted(expected), sorted(result))

    def test_example1(self):
        vertices, edges = 4, [[0, 1], [1, 2], [2, 3]]
        expected = [1, 2]
        self.validate_routers(expected, critial_rounters(vertices, edges))

    def test_example2(self):
        vertices, edges = 5, [[0, 1], [0, 2], [1, 2], [0, 3], [3, 4]]
        expected = [0, 3]
        self.validate_routers(expected, critial_rounters(vertices, edges))

    def test_single_point_of_failure(self):
        vertices, edges = 6, [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
        expected = [0]
        self.validate_routers(expected, critial_rounters(vertices, edges))       
    
    def test_k3(self):
        vertices, edges = 3, [[0, 1], [1, 2], [2, 0]]
        expected = []
        self.validate_routers(expected, critial_rounters(vertices, edges))  
    
    def test_empty_graph(self):
        vertices, edges = 0, []
        expected = []
        self.validate_routers(expected, critial_rounters(vertices, edges))  

    def test_connected_graph1(self):
        vertices, edges = 4, [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]
        expected = []
        self.validate_routers(expected, critial_rounters(vertices, edges))  

    def test_connected_graph2(self):
        vertices, edges = 7, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [5, 0]]
        expected = [0, 5]
        self.validate_routers(expected, critial_rounters(vertices, edges))  
    
    def test_connected_graph3(self):
        vertices, edges = 5, [[0, 1], [1, 2], [2, 0], [0, 3], [3, 4]]
        expected = [0, 3]
        self.validate_routers(expected, critial_rounters(vertices, edges))  


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 12, 2021 \[Medium\] Number of Connected Components
---
> **Question:** Given a list of undirected edges which represents a graph, find out the number of connected components.

**Example:**
```py
Input: [(1, 2), (2, 3), (4, 1), (5, 6)]
Output: 2
Explanation: In the above example, vertices 1, 2, 3, 4 are all connected, and 5, 6 are connected, and thus there are 2 connected components in the graph above.
```

**Soluiton with DisjointSet(Union-Find):** [https://replit.com/@trsong/Number-of-Connected-Components](https://replit.com/@trsong/Number-of-Connected-Components)
```py
import unittest

def count_connected_components(edges):
    uf = DisjointSet()
    for u, v in edges:
        uf.union(u, v)
    return uf.count_roots()


class DisjointSet(object):
    def __init__(self):
        self.parent = {}
    
    def find(self, p):
        self.parent[p] = self.parent.get(p, p)
        while self.parent[p] != p:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def union(self, p1, p2):
        root1 = self.find(p1)
        root2 = self.find(p2)
        if root1 != root2:
            self.parent[root1] = root2

    def count_roots(self):
        root_set = { self.find(root) for root in self.parent.keys() }
        return len(root_set)


class CountConnectedComponentSpec(unittest.TestCase):
    def test_example(self):
        edges = [(1, 2), (2, 3), (4, 1), (5, 6)]
        expected = 2
        self.assertEqual(expected, count_connected_components(edges))

    def test_disconnected_graph(self):
        edges = [(1, 5), (0, 2), (2, 4), (3, 7)]
        expected = 3
        self.assertEqual(expected, count_connected_components(edges))
    
    def test_k3(self):
        edges = [(0, 1), (1, 2), (2, 0)]
        expected = 1
        self.assertEqual(expected, count_connected_components(edges))  
    
    def test_empty_graph(self):
        edges = []
        expected = 0
        self.assertEqual(expected, count_connected_components(edges))  

    def test_connected_graph1(self):
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
        expected = 1
        self.assertEqual(expected, count_connected_components(edges))  

    def test_connected_graph2(self):
        edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (5, 0)]
        expected = 1
        self.assertEqual(expected, count_connected_components(edges))  
    
    def test_connected_graph3(self):
        edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4)]
        expected = 1
        self.assertEqual(expected, count_connected_components(edges))  

    def test_connected_graph4(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        expected = 1
        self.assertEqual(expected, count_connected_components(edges))

    def test_connected_graph5(self):
        edges = [(0, 1), (0, 2), (1, 2), (0, 3), (3, 4)]
        expected = 1
        self.assertEqual(expected, count_connected_components(edges))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Mar 11, 2021 \[Easy\] Make the Largest Number
---
> **Question:** Given a number of integers, combine them so it would create the largest number.

**Example:**
```py
Input: [17, 7, 2, 45, 72]
Output: 77245217
```

**Solution with Customized Sort:** [https://repl.it/@trsong/Construct-the-Largest-Number](https://repl.it/@trsong/Construct-the-Largest-Number)
```py
import unittest

def construct_largest_number(nums):
    if not nums:
        return 0

    negatives = filter(lambda x: x < 0, nums)
    positives = filter(lambda x: x >= 0, nums)
    positive_strings = map(str, positives)
    custom_cmp = lambda x, y: cmp(x + y, y + x)
    if negatives:
        positive_strings.sort(cmp=custom_cmp)
        return int(str(negatives[0]) + "".join(positive_strings))
    else:
        positive_strings.sort(cmp=custom_cmp, reverse=True)
        return int("".join(positive_strings))


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
    unittest.main(exit=False, verbosity=2)
```

### Mar 10, 2021 LC 821 \[Medium\] Shortest Distance to Character
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

**Solution:** [https://repl.it/@trsong/Find-Shortest-Distance-to-Characters](https://repl.it/@trsong/Find-Shortest-Distance-to-Characters)
```py
import unittest

def shortest_dist_to_char(s, ch):
    n = len(s)
    res = [float('inf')] * n
    left_pos = float('-inf')
    for index, c in enumerate(s):
        if ch == c:
            left_pos = index
        res[index] = index - left_pos

    right_pos = float('inf')
    for index in xrange(n - 1, -1, -1):
        c = s[index]
        if ch == c:
            right_pos = index
        res[index] = min(res[index], right_pos - index)
        
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

### Mar 9, 2021 \[Medium\] Number of Android Lock Patterns
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

**Solution with Backtracking:** [https://repl.it/@trsong/Count-Number-of-Android-Lock-Patterns](https://repl.it/@trsong/Count-Number-of-Android-Lock-Patterns)
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
        5: [(1, 9), (3, 7), (2, 8), (4, 6)],
        6: [(3, 9)],
        8: [(7, 9)]
    }

    bypass = { (start, end): cut for cut in overlap for start, end in overlap[cut] + map(reversed, overlap[cut])}

    visited = [False] * 10
    def backtrack(start, length):
        if length == 0:
            return 1
        else:
            visited[start] = True
            res = 0
            for next in xrange(1, 10):
                if visited[next] or (start, next) in bypass and not visited[bypass[(start, next)]]:
                    continue
                res += backtrack(next, length - 1)
            visited[start] = False
            return res

    start_from_1 = backtrack(1, code_len - 1)  # same count as from 3, 7, 9
    start_from_2 = backtrack(2, code_len - 1)  # same count as from 4, 6, 8
    start_from_5 = backtrack(5, code_len - 1)
    return start_from_5 + 4 * (start_from_1 + start_from_2)
                    

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

### Mar 8, 2021 \[Medium\] Lazy Bartender
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

**Solution with Backtracking:** [https://repl.it/@trsong/Lazy-Bartender-Problem](https://repl.it/@trsong/Lazy-Bartender-Problem)
```py
import unittest

def solve_lazy_bartender(preferences):
    drink_map = {}
    alchoholic_customers = set()
    for customer, drinks in preferences.items():
        if not drinks:
            continue

        alchoholic_customers.add(customer)
        for drink in drinks:
            if drink not in drink_map:
                drink_map[drink] = set()
            drink_map[drink].add(customer)
    
    class Context:
        drink_covers = len(drink_map)

    def backtrack(memorized_drinks, remaining_drinks):
        covered_customers = set()
        for drink in memorized_drinks:
            for customer in drink_map[drink]:
                covered_customers.add(customer)

        if covered_customers == alchoholic_customers:
            Context.drink_covers = min(Context.drink_covers, len(memorized_drinks))
        else:
            for i, drink in enumerate(remaining_drinks):
                new_customers = drink_map[drink]
                if new_customers.issubset(covered_customers):
                    continue
                updated_drinks = remaining_drinks[:i] + remaining_drinks[i+1:]
                memorized_drinks.append(drink)
                backtrack(memorized_drinks, updated_drinks)
                memorized_drinks.pop()

    backtrack([], drink_map.keys())
    return Context.drink_covers


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

### Mar 7, 2021 \[Medium\] Distance Between 2 Nodes in BST
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

**Solution:** [https://repl.it/@trsong/Distance-Between-2-Nodes-in-BST](https://repl.it/@trsong/Distance-Between-2-Nodes-in-BST)
```py
import unittest

def find_distance(tree, v1, v2):    
    path1 = find_path(tree, v1)
    path2 = find_path(tree, v2)

    common_nodes = 0
    for n1, n2 in zip(path1, path2):
        if n1 != n2:
            break
        common_nodes += 1

    return len(path1) + len(path2) - 2 * common_nodes
    

def find_path(tree, v):
    res = []
    while True:
        res.append(tree)
        if tree.val == v:
            break
        elif tree.val < v:
            tree = tree.right
        else:
            tree = tree.left
    return res


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class FindLCASpec(unittest.TestCase):
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
    unittest.main(exit=False)
```

### Mar 6, 2021 \[Hard\] Efficiently Manipulate a Very Long String
---
> **Question:** Design a tree-based data structure to efficiently manipulate a very long string that supports the following operations:
>
> - `char char_at(int index)`, return char at index
> - `LongString substring_at(int start_index, int end_index)`, return substring based on start and end index
> - `void delete(int start_index, int end_index)`, deletes the substring 


**My thoughts:** Rope data structure is just a balanced binary tree where leaf stores substring and inner node stores length of all substring of left children recursively. Rope is widely used in text editor program and supports log time insertion, deletion and appending. And is super memory efficent. 

**Solution with Rope:** [https://repl.it/@trsong/Efficiently-Manipulate-a-Very-Long-String](https://repl.it/@trsong/Efficiently-Manipulate-a-Very-Long-String)
```py
import unittest

class LongString(object):
    def __init__(self, s):
        self.rope = RopeNode(s)

    def char_at(self, index):
        return self.rope[index]

    def substring_at(self, start_index, end_index):
        return LongString(self.rope[start_index: end_index + 1])

    def delete(self, start_index, end_index):
        self.rope = self.rope.delete(start_index, end_index)

    
class LongStringSpec(unittest.TestCase):
    def test_empty_string(self):
        self.assertIsNotNone(LongString(''))

    def test_char_at(self):
        s = LongString('01234567')
        self.assertEqual('0', s.char_at(0))
        self.assertEqual('1', s.char_at(1))
        self.assertEqual('3', s.char_at(3))

    def test_chart_at_substring(self):
        s = LongString('012345678')
        self.assertEqual('0', s.substring_at(0, 3).char_at(0))
        self.assertEqual('8', s.substring_at(0, 8).char_at(8))
        self.assertEqual('5', s.substring_at(5, 8).char_at(0))

    def test_delete_string(self):
        s = LongString('012345678')
        s.delete(1, 7)
        self.assertEqual('0', s.char_at(0))
        self.assertEqual('8', s.char_at(1))

        s = LongString('012345678')
        s.delete(0, 3)
        self.assertEqual('4', s.char_at(0))
        self.assertEqual('7', s.char_at(3))

        s = LongString('012345678')
        s.delete(7, 8)
        self.assertEqual('4', s.char_at(4))
        self.assertEqual('6', s.char_at(6))

    def test_char_at_deleted_substring(self):
        s = LongString('012345678')
        s.delete(2, 7)  # gives 018 
        self.assertEqual('1', s.substring_at(1, 2).char_at(0))
        self.assertEqual('8', s.substring_at(1, 2).char_at(1))

    def test_char_at_substring_of_deleted_string(self):
        s = LongString('e012345678eee')
        sub = s.substring_at(1, 8)  # 01234567  
        sub.delete(0, 6)
        self.assertEqual('7', sub.char_at(0))
        

class RopeNode(object):
    def __init__(self, s=None):
        self.weight = len(s) if s else 0
        self.left = None
        self.right = None
        self.data = s

    def delete(self, start_index, end_index):
        if start_index <= 0:
            return self[end_index + 1:]
        elif end_index >= len(self) - 1:
            return self[:start_index]
        else:
            return self[:start_index] + self[end_index + 1:] 

    def __len__(self):
        if self.data is not None:
            return self.weight
        right_len = len(self.right) if self.right else 0
        return self.weight + right_len

    def __add__(self, other):
        # omit tree re-balance 
        node = RopeNode()
        node.weight = len(self)
        node.left = self
        node.right = other
        return node

    def __getitem__(self, key):
        if self.data is not None:
            return self.data[key]
        elif key < self.weight:
            return self.left[key]
        else:
            return self.right[key - self.weight]

    def __getslice__(self, i, j):
        if i >= j:
            return RopeNode('')
        elif self.data:
            return RopeNode(self.data[i:j])
        elif j <= self.weight:
            return self.left[i:j]
        elif i >= self.weight:
            return self.right[i - self.weight:j - self.weight]
        else:
            left_res = self.left[i:self.weight]
            right_res = self.right[0:j - self.weight]
            return left_res + right_res            

    ####################
    # Testing Utilities
    ####################

    def __repr__(self):
        if self.data:
            return self.data
        else:
            return str(self.left or '') + str(self.right or '')

    def print_tree(self):
        stack = [(self, 0)]
        res = ['\n']
        while stack:
            cur, depth = stack.pop()
            res.append('\t' * depth)
            if cur:
                if cur.data:
                    res.append('* data=' + cur.data)
                else:
                    res.append('* weight=' + str(cur.weight))
                stack.append((cur.right, depth + 1))
                stack.append((cur.left, depth + 1))
            else:
                res.append('* None')
            res.append('\n')            
        print ''.join(res)


class RopeNodeSpec(unittest.TestCase):
    def test_string_slice(self):
        """ 
            x 
           / \
          x   2
         / \       
        0   1
        """
        s = RopeNode('0') + RopeNode('1') + RopeNode('2')
        self.assertEqual(1, s.left.weight)
        self.assertEqual(2, s.weight)
        self.assertEqual("0", str(s[0:1]))
        self.assertEqual("01", str(s[0:2]))
        self.assertEqual("012", str(s[0:3]))
        self.assertEqual("12", str(s[1:3]))
        self.assertEqual("2", str(s[2:3]))
        self.assertEqual("1", str(s[1:2]))

    def test_string_slice2(self):
        """ 
              x 
           /     \
          x       x
         / \     / \     
        01 23   4  567
        """
        s = (RopeNode('01') + RopeNode('23')) + (RopeNode('4') + RopeNode('567'))
        self.assertEqual(2, s.left.weight)
        self.assertEqual(4, s.weight)
        self.assertEqual(1, s.right.weight)
        self.assertEqual("012", str(s[0:3]))
        self.assertEqual("3456", str(s[3:7]))
        self.assertEqual("1234567", str(s[1:8]))
        self.assertEqual("7", str(s[7:8]))

    def test_delete(self):
        """ 
              x 
           /     \
          x       x
         / \     / \     
        01 23   4  567
        """
        s = (RopeNode('01') + RopeNode('23')) + (RopeNode('4') + RopeNode('567'))
        self.assertEqual("012", str(s.delete(3, 7)))
        self.assertEqual("4567", str(s.delete(0, 3)))
        self.assertEqual("01237", str(s.delete(4, 6)))

    def test_get_item(self):
        """ 
              x 
           /     \
          x       x
         / \     / \     
        01 23   4  567
        """
        s = (RopeNode('01') + RopeNode('23')) + (RopeNode('4') + RopeNode('567'))
        self.assertEqual("0", s[0])
        self.assertEqual("4", s[4])


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 5, 2021 \[Hard\] Reverse Words Keep Delimiters
---
> **Question:** Given a string and a set of delimiters, reverse the words in the string while maintaining the relative order of the delimiters. For example, given "hello/world:here", return "here/world:hello"
>
> Follow-up: Does your solution work for the following cases: "hello/world:here/", "hello//world:here"

**Solution:** [https://repl.it/@trsong/Reverse-Words-and-Keep-Delimiters](https://repl.it/@trsong/Reverse-Words-and-Keep-Delimiters)
```py
import unittest

def reverse_words_and_keep_delimiters(s, delimiters):
    tokens = filter(len, tokenize(s, delimiters))
    i, j = 0, len(tokens) - 1
    while i < j:
        if tokens[i] in delimiters:
            i += 1
        elif tokens[j] in delimiters:
            j -= 1
        else:
            tokens[i], tokens[j] = tokens[j], tokens[i]
            i += 1
            j -= 1
    return ''.join(tokens)  
     

def tokenize(s, delimiters):
    res = []
    prev_index = -1
    for i, ch in enumerate(s):
        if ch in delimiters:
            res.append(s[prev_index + 1: i])
            res.append(s[i])
            prev_index = i
    res.append(s[prev_index + 1: len(s)])
    return res


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

### Mar 4, 2021 \[Hard\] Maximize Sum of the Minimum of K Subarrays
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

Of course, the min value of last array will change, but we can calculate that along the way when we absorb more elements, and we can use `dp[n-p][k] for all p <= n` to calculate the answer. Thus `dp[n][k] = max{dp[n-p][k-1] + min_value of last_subarray} for all p <= n, ie. num[p] is in last subarray`.


**Solution with DP:** [https://repl.it/@trsong/Find-the-Maximize-Sum-of-the-Minimum-of-K-Subarrays](https://repl.it/@trsong/Find-the-Maximize-Sum-of-the-Minimum-of-K-Subarrays)
```py
import unittest

def max_aggregate_subarray_min(nums, k):
    n = len(nums)

    # Let dp[n][k] represents max aggregate subarray of min for nums[:n] 
    # and target number of partition is k
    # dp[n][k] = max(dp[n - p][k - 1] + min(nums[n - p: n])) for p <= n
    dp = [[float('-inf') for _ in xrange(k + 1)] for _ in xrange(n + 1)]
    dp[0][0] = 0
    
    for j in xrange(1, k + 1):
        for i in xrange(j, n + 1):
            last_window_min = nums[i - 1]
            for p in xrange(1, i + 1):
                last_window_min = min(last_window_min, nums[i - p])
                dp[i][j] = max(dp[i][j], dp[i - p][j - 1] + last_window_min)
    
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
    unittest.main(exit=False, verbosity=2)
```

### Mar 3, 2021 \[Hard\] Maximize the Minimum of Subarray Sum
---
> **Question:** Given an array of numbers `N` and an integer `k`, your task is to split `N` into `k` partitions such that the maximum sum of any partition is minimized. Return this sum.
>
> For example, given `N = [5, 1, 2, 7, 3, 4]` and `k = 3`, you should return `8`, since the optimal partition is `[5, 1, 2], [7], [3, 4]`.


**My thoughts:** The method to solve this problem is through guessing the result. We have the following observations: 

- The lower bound of guessing number is `max(nums)`, as max element has to present in some subarray. 
- The upper bound of guessing number is `sum(nums)` as sum of subarray cannot exceed sum of entire array. 

We can use the guessing number to cut the array greedily so that each parts cannot exceed the guessing number:
- If the guessing number is lower than expected, then we over-cut the array.  (cut more than k parts)
- If the guessing number is higher than expected, then we under-cut the array.  (cut less than k parts)

We can use binary search to get result such that it just-cut the array: under-cut and just-cut goes left and over-cut goes right. 

But how can we make sure that result we get from binary search is indeed sum of subarray?

The reason is simply, if it is just-cut, it won't stop until it's almost over-cut that gives smallest just-cut. Maximum of the Minimum Sum among all subarray sum is actually the smallest just-cut. And it will stop at that number. 


**Solution with Binary Search:** [https://repl.it/@trsong/Maximize-the-Minimum-of-Subarray-Sum](https://repl.it/@trsong/Maximize-the-Minimum-of-Subarray-Sum)
```py
import unittest


def max_of_min_sum_subarray(nums, k):
    lo = max(nums)
    hi = sum(nums)

    while lo < hi:
        mid = lo + (hi - lo) // 2
        if within_partition_constraint(nums, k, mid):
            hi = mid
        else:
            lo = mid + 1
    
    return lo


def within_partition_constraint(nums, k, subarray_limit):
    accu = 0
    for num in nums:
        if accu + num > subarray_limit:
            accu = num
            k -= 1
        else:
            accu += num
    return k >= 1



class MaxOfMinSumSubarraySpec(unittest.TestCase):
    def test_example(self):
        k, nums = 3, [5, 1, 2, 7, 3, 4]
        expected = 8  # [5, 1, 2], [2, 7], [3, 4]
        self.assertEqual(expected, max_of_min_sum_subarray(nums, k))

    def test_ascending_array(self):
        k, nums = 3, [1, 2, 3, 4]
        expected = 4  # [1, 2], [3], [4]
        self.assertEqual(expected, max_of_min_sum_subarray(nums, k))

    def test_k_is_one(self):
        k, nums = 1, [1, 1, 1, 1, 4]
        expected = 8  # [1, 1, 1, 1, 4]
        self.assertEqual(expected, max_of_min_sum_subarray(nums, k))

    def test_return_larger_half(self):
        k, nums = 2,  [1, 2, 3, 4, 5, 10, 11, 3, 6, 16]
        expected = 36  # [1, 2, 3, 4, 5, 10, 11], [3, 6, 16]
        self.assertEqual(expected, max_of_min_sum_subarray(nums, k))

    def test_evenly_distributed(self):
        k, nums = 4, [1, 1, 1, 1]
        expected = 1
        self.assertEqual(expected, max_of_min_sum_subarray(nums, k))

    def test_evenly_distributed2(self):
        k, nums = 3, [1, 1, 1, 1, 1, 1, 1, 1, 1]
        expected = 3
        self.assertEqual(expected, max_of_min_sum_subarray(nums, k))

    def test_outlier_element(self):
        k, nums = 3, [1, 1, 1, 100, 1]
        expected = 100  # [1, 1, 1], [100], [1]
        self.assertEqual(expected, max_of_min_sum_subarray(nums, k))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 2, 2021 \[Medium\] Bitwise AND of a Range
---
> **Question:** Write a function that returns the bitwise AND of all integers between M and N, inclusive.

**Solution:** [https://repl.it/@trsong/Calculate-Bitwise-AND-of-a-Range](https://repl.it/@trsong/Calculate-Bitwise-AND-of-a-Range)
```py
import unittest

def bitwise_and_of_range(m, n):
    """"
      0b10110001   -> m
    & 0b10110010
    & 0b10110011
    & 0b10110100   
    & 0b10110101   -> n
    = 0b10110000

    We need to find longest prefix between m and n. That means we can keep removing least significant bit of n until n < m. Then we can get the longest prefix. 
    """
    if m > n:
        m, n = n, m

    while m < n:
        # remove last set bit
        n &= n - 1

    return n


class BitwiseAndOfRangeSpec(unittest.TestCase):
    def test_same_end_point(self):
        m, n, expected = 42, 42, 42
        self.assertEqual(expected, bitwise_and_of_range(m, n))

    def test_end_share_same_prefix(self):
        m = 0b110100
        n = 0b110011
        expected = 0b110000
        self.assertEqual(expected, bitwise_and_of_range(m, n))

    def test_one_end_has_zero(self):
        m = 0b1111111
        n = 0b0
        expected = 0b0
        self.assertEqual(expected, bitwise_and_of_range(m, n))

    def test_neither_end_has_same_digit(self):
        m = 0b100101
        n = 0b011010
        expected = 0b0
        self.assertEqual(expected, bitwise_and_of_range(m, n))

    def test_bitwise_all_number_within_range(self):
        """
          0b1100
        & 0b1101
        & 0b1110
        & 0b1111
        = 0b1100
        """
        m = 12  # 0b1100
        n = 15  # 0b1111
        expected = 0b1100
        self.assertEqual(expected, bitwise_and_of_range(m, n))

    def test_bitwise_all_number_within_range2(self):
        """
          0b10001
        & 0b10010
        & 0b10011
        = 0b10000
        """
        m = 17  # 0b10001
        n = 19  # 0b10011
        expected = 0b10000
        self.assertEqual(expected, bitwise_and_of_range(m, n))

    def test_both_end_share_some_digits(self):
        """
          0b01010
        & 0b01011
        ...
        & 0b10000
        ...
        & 0b10100
        = 0b00000
        """
        m = 10  # 0b01010
        n = 20  # 0b10100
        expected = 0
        self.assertEqual(expected, bitwise_and_of_range(m, n))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Mar 1, 2021 LC 336 \[Hard\] Palindrome Pairs
---
> **Question:** Given a list of words, find all pairs of unique indices such that the concatenation of the two words is a palindrome.
>
> For example, given the list `["code", "edoc", "da", "d"]`, return `[(0, 1), (1, 0), (2, 3)]`.


**My thoughts:** any word in the list can be partition into `prefix` and `suffix`. If there exists another word such that its reverse equals either prefix or suffix, then we can combine them and craft a new palindrome: 
1. `reverse_suffix + prefix + suffix` where prefix is a palindrome or 
2. `prefix + suffix + reverse_prefix` where suffix is a palindrome

**Solution:** [https://repl.it/@trsong/Find-All-Palindrome-Pairs](https://repl.it/@trsong/Find-All-Palindrome-Pairs)
```py

import unittest
from collections import defaultdict

def find_all_palindrome_pairs(words):
    reverse_words = defaultdict(list)
    for i, word in enumerate(words):
        reverse_words[word[::-1]].append(i)
    
    res = []

    if "" in reverse_words:
        palindrome_indices = filter(lambda j: is_palindrome(words[j]), xrange(len(words)))
        res.extend((i, j) for i in reverse_words[""] for j in palindrome_indices if i != j)

    for i, word in enumerate(words):
        for pos in xrange(len(word)):
            prefix = word[:pos]
            suffix = word[pos:]
            if prefix in reverse_words and is_palindrome(suffix):
                res.extend((i, j) for j in reverse_words[prefix] if i != j)

            if suffix in reverse_words and is_palindrome(prefix):
                res.extend((j, i) for j in reverse_words[suffix] if i != j)

    return res


def is_palindrome(s):
    i = 0
    j = len(s) - 1
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


### Feb 28, 2021 \[Medium\] 24-Hour Hit Counter
---
> **Question:** You are given an array of length 24, where each element represents the number of new subscribers during the corresponding hour. Implement a data structure that efficiently supports the following:
>
> - `update(hour: int, value: int)`: Increment the element at index hour by value.
> - `query(start: int, end: int)`: Retrieve the number of subscribers that have signed up between start and end (inclusive). You can assume that all values get cleared at the end of the day, and that you will not be asked for start and end values that wrap around midnight.

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


**Solution with BIT:** [https://repl.it/@trsong/24-Hour-Hit-Counter](https://repl.it/@trsong/24-Hour-Hit-Counter)
```py
import unittest

class HitCounter(object):
    HOUR_OF_DAY = 24

    def __init__(self, hours=HOUR_OF_DAY):
        self.tree = [0] * (hours + 1)

    def update(self, hour, value):
        old_val = self.query(hour, hour)
        delta = value - old_val

        index = hour + 1
        while index < len(self.tree):
            self.tree[index] += delta
            index += index & -index

    def query(self, start_time, end_time):
        return self.query_from_begin(end_time) - self.query_from_begin(start_time - 1)
    
    def query_from_begin(self, end_time):
        index = end_time + 1
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res


class HitCounterSpec(unittest.TestCase):
    def test_query_without_update(self):
        hc = HitCounter()
        self.assertEqual(0, hc.query(0, 23))
        self.assertEqual(0, hc.query(3, 5))
    
    def test_update_should_affect_query_value(self):
        hc = HitCounter()
        hc.update(5, 10)
        hc.update(10, 15)
        hc.update(12, 20)
        self.assertEqual(0, hc.query(0, 4))
        self.assertEqual(10, hc.query(0, 5))
        self.assertEqual(10, hc.query(0, 9))
        self.assertEqual(25, hc.query(0, 10))
        self.assertEqual(45, hc.query(0, 13))
        hc.update(3, 2)
        self.assertEqual(2, hc.query(0, 4))
        self.assertEqual(12, hc.query(0, 5))
        self.assertEqual(12, hc.query(0, 9))
        self.assertEqual(27, hc.query(0, 10))
        self.assertEqual(47, hc.query(0, 13))

    def test_number_of_subscribers_can_decrease(self):
        hc = HitCounter()
        hc.update(10, 5)
        hc.update(20, 10)
        self.assertEqual(10, hc.query(15, 23))
        hc.update(12, -3)
        self.assertEqual(10, hc.query(15, 23))
        hc.update(17, -7)
        self.assertEqual(3, hc.query(15, 23))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 27, 2021 \[Easy\] Flatten Nested List Iterator
---
> **Question:** Implement a 2D iterator class. It will be initialized with an array of arrays, and should implement the following methods:
>
> - `next()`: returns the next element in the array of arrays. If there are no more elements, raise an exception.
> - `has_next()`: returns whether or not the iterator still has elements left.
>
> For example, given the input `[[1, 2], [3], [], [4, 5, 6]]`, calling `next()` repeatedly should output `1, 2, 3, 4, 5, 6`.
>
> Do not use flatten or otherwise clone the arrays. Some of the arrays can be empty.

**Solution:** [https://repl.it/@trsong/Flatten-Nested-List-Iterator](https://repl.it/@trsong/Flatten-Nested-List-Iterator)
```py
import unittest

class NestedIterator(object):
    def __init__(self, nested_list):
        self.row = 0
        self.col = 0
        self.nested_list = [[None]] + nested_list
        self.next()
        
    def next(self):
        res = self.nested_list[self.row][self.col]

        self.col += 1
        while self.row < len(self.nested_list) and self.col >= len(self.nested_list[self.row]):
            self.col = 0
            self.row += 1

        return res       

    def has_next(self):
        return self.row < len(self.nested_list)
        

class NestedIteratorSpec(unittest.TestCase):
    def assert_result(self, nested_list):
        expected = []
        for lst in nested_list:
            if not lst:
                continue
            expected.extend(lst)

        res = []
        it = NestedIterator(nested_list)
        while it.has_next():
            res.append(it.next())
        
        self.assertEqual(expected, res)       
        
    def test_example(self):
        self.assert_result([[1, 2], [3], [], [4, 5, 6]])

    def test_empty_list(self):
        it = NestedIterator([])
        self.assertFalse(it.has_next())

    def test_empty_list2(self):
        it = NestedIterator([[], [], []])
        self.assertFalse(it.has_next())

    def test_non_empty_list(self):
        self.assert_result([[1], [2], [3], [4]])

    def test_non_empty_list2(self):
        self.assert_result([[1, 1, 1], [4], [1, 2, 3], [5]])

    def test_has_empty_list(self):
        self.assert_result([[], [1, 2, 3], []])


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 26, 2021 LC 227 \[Medium\] Basic Calculator II
---
> **Question:** Implement a basic calculator to evaluate a simple expression string.
>
> The expression string contains only non-negative integers, +, -, *, / operators and empty spaces. The integer division should truncate toward zero.

**Example 1:**
```py
Input: "3+2*2"
Output: 7
```

**Example 2:**
```py
Input: " 3/2 "
Output: 1
```

**Example 3:**
```py
Input: " 3+5 / 2 "
Output: 5
```


**My thoughts:** A complicated _expression_ can be broken into multiple normal _terms_. `Expr = term1 + term2 - term3 ...`. Between each consecutive term we only allow `+` and `-`. Whereas within each term we only allow `*` and `/`. So we will have the following definition of an expression. e.g. `1 + 2 - 1*2*1 - 3/4*4 + 5*6 - 7*8 + 9/10 = (1) + (2) - (1*2*1) - (3/4*4) + (5*6) - (7*8) + (9/10)` 

_Expression_ is one of the following:
- Empty or 0
- Term - Expression
- Term + Expression

_Term_ is one of the following:
- 1
- A number * Term
- A number / Term


Thus, we can comupte each term value and sum them together.


**Solution:** [https://repl.it/@trsong/Implement-Basic-Calculator-II](https://repl.it/@trsong/Implement-Basic-Calculator-II)
```py
import unittest

OP_SET = {'+', '-', '*', '/', 'EOF'}

def calculate(s):
    tokens = tokenize(s)
    expr_value = 0
    term_value = 0
    num = 0
    prev_op = '+'

    for token in tokens:
        if not token or token.isspace():
            continue
            
        if token not in OP_SET:
            num = int(token)
            continue
        
        if prev_op == '+':
            expr_value += term_value
            term_value = num
        elif prev_op == '-':
            expr_value += term_value
            term_value = -num
        elif prev_op == '*':
            term_value *= num
        elif prev_op == '/':
            sign = 1 if term_value > 0 else -1
            term_value = abs(term_value) / num * sign
            
        num = 0
        prev_op = token

    expr_value += term_value
    return expr_value
    

def tokenize(s):
    res = []
    prev_pos = -1
    for pos, ch in enumerate(s):
        if ch in OP_SET:
            res.append(s[prev_pos + 1: pos])
            res.append(ch)
            prev_pos = pos
    res.append(s[prev_pos + 1: ])
    res.append('EOF')
    return res


class CalculateSpec(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(0, calculate(""))

    def test_example1(self):
        self.assertEqual(7, calculate("3+2*2"))

    def test_example2(self):
        self.assertEqual(1, calculate(" 3/2 "))

    def test_example3(self):
        self.assertEqual(5, calculate(" 3+5 / 2 "))

    def test_negative1(self):
        self.assertEqual(-1, calculate("-1"))

    def test_negative2(self):
        self.assertEqual(0, calculate(" -1/2 "))

    def test_negative3(self):
        self.assertEqual(-1, calculate(" -7 / 4 "))

    def test_minus(self):
        self.assertEqual(-5, calculate("-2-3"))
    
    def test_positive1(self):
        self.assertEqual(10, calculate("100/ 10"))
    
    def test_positive2(self):
        self.assertEqual(4, calculate("9 /2"))

    def test_complicated_operations(self):
        self.assertEqual(-24, calculate("1*2-3/4+5*6-7*8+9/10"))

    def test_complicated_operations2(self):
        self.assertEqual(10000, calculate("10000-1000/10+100*1"))

    def test_complicated_operations3(self):
        self.assertEqual(13, calculate("14-3/2"))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 25, 2021 \[Medium\] Evaluate Expression in Reverse Polish Notation
---
> **Question:** Given an arithmetic expression in **Reverse Polish Notation**, write a program to evaluate it.
>
> The expression is given as a list of numbers and operands. 

**Example 1:** 
```py
[5, 3, '+'] should return 5 + 3 = 8.
```

**Example 2:**
```py
 [15, 7, 1, 1, '+', '-', '/', 3, '*', 2, 1, 1, '+', '+', '-'] should return 5, 
 since it is equivalent to ((15 / (7 - (1 + 1))) * 3) - (2 + (1 + 1)) = 5.
 ```


**Solution with Stack:** [https://repl.it/@trsong/Evaluate-Expression-Represented-in-Reverse-Polish-Notation](https://repl.it/@trsong/Evaluate-Expression-Represented-in-Reverse-Polish-Notation)
```py
import unittest

class RPNExprEvaluator(object):
    @staticmethod
    def run(tokens):
        stack = []
        for token in tokens:
            if type(token) is int:
                stack.append(token)
            else:
                num2 = stack.pop()
                num1 = stack.pop()
                res = RPNExprEvaluator.apply(token, num1, num2)
                stack.append(res)
        return stack[-1] if stack else 0

    @staticmethod
    def apply(operation, num1, num2):
        if operation == '+':
            return num1 + num2
        elif operation == '-':
            return num1 - num2
        elif operation == '*':
            return num1 * num2
        elif operation == '/':
            return num1 / num2
        else:
            raise NotImplementedError
        

class RPNExprEvaluatorSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(8, RPNExprEvaluator.run([5, 3, '+'])) # 5 + 3 = 8

    def test_example2(self):
        tokens = [15, 7, 1, 1, '+', '-', '/', 3, '*', 2, 1, 1, '+', '+', '-']
        self.assertEqual(5, RPNExprEvaluator.run(tokens))

    def test_empty_tokens(self):
        self.assertEqual(0, RPNExprEvaluator.run([]))

    def test_expression_contains_just_number(self):
        self.assertEqual(42, RPNExprEvaluator.run([42]))
    
    def test_balanced_expression_tree(self):
        tokens = [7, 2, '-', 4, 1, '+', '*'] 
        self.assertEqual(25, RPNExprEvaluator.run(tokens))  # (7 - 2) * (4 + 1) = 25
    
    def test_left_heavy_expression_tree(self):
        tokens = [6, 4, '-', 2, '/']  
        self.assertEqual(1, RPNExprEvaluator.run(tokens)) # (6 - 4) / 2 = 1

    def test_right_heavy_expression_tree(self):
        tokens = [2, 8, 2, '/', '*']
        self.assertEqual(8, RPNExprEvaluator.run(tokens)) # 2 * (8 / 2) = 8


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 24, 2021 \[Easy\] Minimum Distance between Two Words
---
> **Question:** Find an efficient algorithm to find the smallest distance (measured in number of words) between any two given words in a string.
>
> For example, given words `"hello"`, and `"world"` and a text content of `"dog cat hello cat dog dog hello cat world"`, return `1` because there's only one word `"cat"` in between the two words.

**Solution with Two Pointers:** [https://repl.it/@trsong/Minimum-Distance-between-Two-Words](https://repl.it/@trsong/Minimum-Distance-between-Two-Words)
```py
import unittest
import sys

def word_distance(s, word1, word2):
    i = sys.maxint
    j = -sys.maxint
    res = sys.maxint

    for index, word in enumerate(s.split()):
        if word == word1:
            i = index
        
        if word == word2:
            j = index

        res = min(res, abs(j - i))

    return res if res < sys.maxint else -1


class WordDistanceSpec(unittest.TestCase):
    def test_example(self):
        s = 'dog cat hello cat dog dog hello cat world'
        word1 = 'hello'
        word2 = 'world'
        self.assertEqual(2, word_distance(s, word1, word2))

    def test_word_not_exists_in_sentence(self):
        self.assertEqual(-1, word_distance("", "a", "b"))

    def test_word_not_exists_in_sentence2(self):
        self.assertEqual(-1, word_distance("b", "a", "a"))

    def test_word_not_exists_in_sentence3(self):
        self.assertEqual(-1, word_distance("ab", "a", "b"))
    
    def test_only_one_word_exists(self):
        s = 'a b c d'
        word1 = 'a'
        word2 = 'e'
        self.assertEqual(-1, word_distance(s, word1, word2))
        self.assertEqual(-1, word_distance(s, word2, word1))
        
    def test_search_for_same_word_in_sentence(self):
        s = 'cat dog cat cat dog dog dog cat'
        word = 'cat'
        self.assertEqual(0, word_distance(s, word, word))

    def test_second_word_comes_first(self):
        s = 'air water water earth water water water'
        word1 = "earth"
        word2 = "air"
        self.assertEqual(3, word_distance(s, word1, word2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 23, 2021 \[Hard\] K-Palindrome
---
> **Question:** Given a string which we can delete at most k, return whether you can make a palindrome.
>
> For example, given `'waterrfetawx'` and a k of 2, you could delete f and x to get `'waterretaw'`.

**My thoughts:** We can either solve this problem by modifying the edit distance function or taking advantage of longest common subsequnce. Here we choose the later method.

We can first calculate the minimum deletion needed to make a palindrome and the way we do it is to compare the original string vs the reversed string in order to calculate the LCS - longest common subsequence. Thus the minimum deletion equals length of original string minus LCS.

To calculate LCS, we use DP and will encounter the following situations:
1. If the last digit of each string matches each other, i.e. `lcs(seq1 + s, seq2 + s)` then `result = 1 + lcs(seq1, seq2)`.
2. If the last digit not matches, i.e. `lcs(seq1 + s, seq2 + p)`, then res is either ignore s or ignore q. Just like insert a whitespace or remove a letter from edit distance, which gives `max(lcs(seq1, seq2 + p), lcs(seq1 + s, seq2))`


**Solution with DP:** [https://repl.it/@trsong/Find-K-Palindrome](https://repl.it/@trsong/Find-K-Palindrome)
```py
import unittest

def is_k_palindrome(s, k):
    lcs = longest_common_subsequence(s, s[::-1])
    min_letter_to_remove = len(s) - lcs
    return min_letter_to_remove <= k


def longest_common_subsequence(seq1, seq2):
    n, m = len(seq1), len(seq2)

    # Let dp[n][m] represents lcs of seq1[:n] and seq2[:m]
    # dp[n][m] = 1 + dp[n-1][m-1]             if seq1[n-1] == seq2[m-1]
    #          = max(dp[n-1][m], dp[n][m-1])  otherwise
    dp = [[0 for _ in xrange(m + 1)] for _ in xrange(n + 1)]
    for i in xrange(1, n + 1):
        for j in xrange(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[n][m]


class IsKPalindromeSpec(unittest.TestCase):
    def test_example(self):
        # waterrfetawx => waterretaw
        self.assertTrue(is_k_palindrome('waterrfetawx', k=2))

    def test_empty_string(self):
        self.assertTrue(is_k_palindrome('', k=2))
    
    def test_palindrome_string(self):
        self.assertTrue(is_k_palindrome('abcddcba', k=1))

    def test_removing_exact_k_characters(self):
        # abcdecba => abcecba
        self.assertTrue(is_k_palindrome('abcdecba', k=1))

    def test_removing_exact_k_characters2(self):
        # abcdeca => acdca
        self.assertTrue(is_k_palindrome('abcdeca', k=2))

    def test_removing_exact_k_characters3(self):
        # acdcb => cdc
        self.assertTrue(is_k_palindrome('acdcb', k=2))

    def test_not_k_palindrome(self):
        self.assertFalse(is_k_palindrome('acdcb', k=1))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 22, 2021 \[Medium\] Lazy Binary Tree Generation
---
> **Question:** Generate a finite, but an arbitrarily large binary tree quickly in `O(1)`.
>
> That is, `generate()` should return a tree whose size is unbounded but finite.

**Solution:** [https://repl.it/@trsong/Lazy-Binary-Tree-Generation](https://repl.it/@trsong/Lazy-Binary-Tree-Generation)
```py
import random

def generate():
    return TreeNode(0)


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self._left = left
        self._right = right
        self._is_left_init = False
        self._is_right_init = False

    @property
    def left(self):
        if not self._is_left_init:
            if random.randint(0, 1):
                self._left = TreeNode(0)
            self._is_left_init = True
        return self._left

    @property
    def right(self):
        if not self._is_right_init:
            if random.randint(0, 1):
                self._right = TreeNode(0)
            self._is_right_init = True
        return self._right

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


if __name__ == '__main__':
    print generate()
```

### Feb 21, 2021 LC 131 \[Medium\] Palindrome Partitioning
---
> **Question:** Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.
>
> A palindrome string is a string that reads the same backward as forward.

**Example 1:**
```py
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]
```

**Example 2:**
```py
Input: s = "a"
Output: [["a"]]
```

**Solution with Backtracking:** [https://repl.it/@trsong/Palindrome-Partitioning](https://repl.it/@trsong/Palindrome-Partitioning)
```py
import unittest

def palindrome_partition(s):
    n = len(s)
    cache = [[None for _ in xrange(n)] for _ in xrange(n)]
    res = []
    backtrack(s, res, [], 0, cache)
    return res


def backtrack(s, res, accu, start, cache):
    n = len(s)
    if start >= n:
        res.append(accu[:])
    
    for end in xrange(start, n):
        if is_palindrome(s, start, end, cache):
            accu.append(s[start: end + 1])
            backtrack(s, res, accu, end + 1, cache)
            accu.pop()


def is_palindrome(s, start, end, cache):
    if end - start < 1:
        return True
    
    if cache[start][end] is None:
        cache[start][end] = s[start] == s[end] and is_palindrome(s, start + 1, end - 1, cache)
    
    return cache[start][end]


class PalindromePartitionSpec(unittest.TestCase):
    def assert_result(self, expected, res):
        expected.sort()
        res.sort()
        self.assertEqual(expected, res)
    
    def test_example(self):
        s = "aab"
        expected = [["a","a","b"],["aa","b"]]
        self.assert_result(expected, palindrome_partition(s))
    
    def test_example2(self):
        s = "a"
        expected = [["a"]]
        self.assert_result(expected, palindrome_partition(s))
    
    def test_multiple_results(self):
        s = "12321"
        expected = [
            ['1', '2', '3', '2', '1'], 
            ['1', '232', '1'], 
            ['12321']]
        self.assert_result(expected, palindrome_partition(s))
    
    def test_multiple_results2(self):
        s = "112321"
        expected = [
            ['1', '1', '2', '3', '2', '1'], 
            ['1', '1', '232', '1'], 
            ['1', '12321'], 
            ['11', '2', '3', '2', '1'], 
            ['11', '232', '1']]
        self.assert_result(expected, palindrome_partition(s))
    
    def test_multiple_results3(self):
        s = "aaa"
        expected = [["a", "aa"], ["aa", "a"], ["aaa"], ['a', 'a', 'a']]
        self.assert_result(expected, palindrome_partition(s))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 20, 2021 \[Hard\] Minimum Palindrome Substring
---
> **Question:** Given a string, split it into as few strings as possible such that each string is a palindrome.
>
> For example, given the input string `"racecarannakayak"`, return `["racecar", "anna", "kayak"]`.
>
> Given the input string `"abc"`, return `["a", "b", "c"]`.


**Solution with DP:** [https://repl.it/@trsong/Minimum-Palindrome-Substring](https://repl.it/@trsong/Minimum-Palindrome-Substring)
```py
import unittest

def min_palindrome_substring(s):
    n = len(s)
    # Let cache[start][end] represents s[start: end + 1] is palindrome or not
    cache = [[None for _ in xrange(n)] for _ in xrange(n)]

    # Let dp[size] represents the min palindrome for string s[:size]
    # dp[size] = dp[start] + [s[start: size]] if s[start: size] is palindrome and start = argmin(len(dp[x])) 
    dp = [[] for _ in xrange(n + 1)]
    for size in xrange(1, n + 1):
        start = size - 1
        min_sub_size = float('inf')
        for x in xrange(size):
            if not is_palindrome(s, x, size - 1, cache):
                continue
            if len(dp[x]) < min_sub_size:
                min_sub_size = len(dp[x])
                start = x
        dp[size] = dp[start] + [s[start: size]]
    return dp[-1]


def is_palindrome(s, start, end, cache):
    if end - start < 1:
        return True
    
    if cache[start][end] is None:
        cache[start][end] = s[start] == s[end] and is_palindrome(s, start + 1, end - 1, cache)
    
    return cache[start][end]
    

class MinPalindromeSubstringSpec(unittest.TestCase):
    def test_example(self):
        s = 'racecarannakayak'
        expected = ['racecar', 'anna', 'kayak']
        self.assertEqual(expected, min_palindrome_substring(s))

    def test_example2(self):
        s = 'abc'
        expected = ['a', 'b', 'c']
        self.assertEqual(expected, min_palindrome_substring(s))

    def test_empty_string(self):
        self.assertEqual([], min_palindrome_substring(''))

    def test_one_char_string(self):
        self.assertEqual(['a'], min_palindrome_substring('a'))

    def test_already_palindrome(self):
        s = 'abbacadabraarbadacabba'
        expected = ['abbacadabraarbadacabba']
        self.assertEqual(expected, min_palindrome_substring(s))

    def test_long_and_short_palindrome_substrings(self):
        s1 = 'aba'
        s2 = 'abbacadabraarbadacabba'
        s3 = 'c'
        expected = [s1, s2, s3]
        self.assertEqual(expected, min_palindrome_substring(s1 + s2 + s3))

    def test_should_return_optimal_solution(self):
        s = 'xabaay'
        expected = ['x', 'aba', 'a', 'y']
        self.assertEqual(expected, min_palindrome_substring(s))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

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

**Solution with DP:** [https://repl.it/@trsong/Calculate-the-Smallest-Number-of-Perfect-Squares](https://repl.it/@trsong/Calculate-the-Smallest-Number-of-Perfect-Squares)
```py
import unittest
import math

def min_perfect_squares(target):
    if target == 0:
        return 1
    elif target < 0:
        return -1

    # Let dp[num] represents min num of squares sum to num
    # dp[num] = dp[num - i * i] + 1 where i * i <= num
    dp = [float("inf")] * (target + 1)
    dp[0] = 0
    for num in xrange(1, target + 1):
        for i in xrange(1, int(math.sqrt(num) + 1)):
            dp[num] = min(dp[num], 1 + dp[num - i * i])
    
    return dp[target]


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
