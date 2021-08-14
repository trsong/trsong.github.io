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