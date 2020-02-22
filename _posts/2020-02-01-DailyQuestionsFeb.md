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


### Feb 22, 2020 LC 79 [Medium] Word Search
---
> **Question:** You are given a 2D array of characters, and a target string. Return whether or not the word target word exists in the matrix. Unlike a standard word search, the word must be either going left-to-right, or top-to-bottom in the matrix.

**Example:**
```py
[['F', 'A', 'C', 'I'],
 ['O', 'B', 'Q', 'P'],
 ['A', 'N', 'O', 'B'],
 ['M', 'A', 'S', 'S']]

Given this matrix, and the target word FOAM, you should return true, as it can be found going up-to-down in the first column.
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