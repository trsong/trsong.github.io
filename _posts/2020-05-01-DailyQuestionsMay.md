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

### Jul 7, 2020 \[Medium\] Matrix Rotation
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

### Jul 6, 2020 \[Easy\] Valid Mountain Array
---
> **Question:** Given an array of heights, determine whether the array forms a "mountain" pattern. A mountain pattern goes up and then down.

**Example:**
```py
validMountainArray([1, 2, 3, 2, 1])  # True
validMountainArray([1, 2, 3])  # False
```

**Solution:** [https://repl.it/@trsong/Valid-Mountain-Array](https://repl.it/@trsong/Valid-Mountain-Array)
```py
import unittest

def is_valid_mountain(arr):
    if len(arr) < 3:
        return False
    
    sign = 1
    num_sign_change = 0
    has_increase = False
    for i in xrange(1, len(arr)):
        diff = arr[i] - arr[i-1]
        if diff > 0:
            has_increase = True
        if diff * sign < 0:
            sign *= -1
            num_sign_change += 1

    return has_increase and num_sign_change == 1
        

class IsValidMountainSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(is_valid_mountain([1, 2, 3, 2, 1]))

    def test_example2(self):
        self.assertFalse(is_valid_mountain([1, 2, 3]))

    def test_empty_array(self):
        self.assertFalse(is_valid_mountain([]))

    def test_one_element_array(self):
        self.assertFalse(is_valid_mountain([1]))

    def test_two_elements_array(self):
        self.assertFalse(is_valid_mountain([1, 2]))

    def test_three_elements_array(self):
        self.assertFalse(is_valid_mountain([1, 2, 3]))

    def test_three_elements_array2(self):
        self.assertTrue(is_valid_mountain([1, 2, 1]))

    def test_duplicted_elements(self):
        self.assertFalse(is_valid_mountain([1, 2, 2]))
    
    def test_duplicted_element2(self):
        self.assertFalse(is_valid_mountain([1, 1, 2]))

    def test_duplicted_elements2(self):
        self.assertFalse(is_valid_mountain([0, 0, 0]))

    def test_mutiple_mountains(self):
        self.assertFalse(is_valid_mountain([1, 2, 1, 2]))

    def test_mutiple_mountains2(self):
        self.assertFalse(is_valid_mountain([1, 2, 1, 2, 1]))

    def test_concave_array(self):
        self.assertFalse(is_valid_mountain([0, -1, 1]))

    def test_no_ascending(self):
        self.assertFalse(is_valid_mountain([0, 0, -1]))

    def test_no_ascending2(self):
        self.assertFalse(is_valid_mountain([0, -1, -1]))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Jul 5, 2020 \[Easy\] Symmetric K-ary Tree
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

**Solution with DFS:** [https://repl.it/@trsong/Is-Symmetric-K-ary-Tree](https://repl.it/@trsong/Is-Symmetric-K-ary-Tree)
```py
import unittest

def is_symmetric(tree):
    if not tree:
        return True

    traversal = dfs_traversal(tree)
    reverse_traversal = dfs_traversal(tree, reverse=True)

    for node1, node2 in zip(traversal, reverse_traversal):
        equal_val = node1.val == node2.val
        equal_children_num = len(node1.children or []) == len(node2.children or [])
        if not equal_val or not equal_children_num:
            return False
        
    return True
            

def dfs_traversal(tree, reverse=False):
    stack = [tree]
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
    unittest.main(exit=False)
```

### Jul 4, 2020 \[Easy\] One-to-one Character Mapping
---
> **Question:** Given two strings, find if there is a one-to-one mapping of characters between the two strings.

**Example 1:**
```py
Input: 'abc', 'def'
Output: True # a -> d, b -> e, c -> f
```

**Example 2:**
```py
Input: 'aab', 'def'
Ouput: False # a can't map to d and e 
```

**Solution:** [https://repl.it/@trsong/Find-One-to-one-Character-Mapping](https://repl.it/@trsong/Find-One-to-one-Character-Mapping)
```py
import unittest

CHAR_SPACE_SIZE = 256

def is_one_to_one_mapping(s1, s2):
    if len(s1) != len(s2):
        return False
    
    char_table = [None] * CHAR_SPACE_SIZE
    for u, v in zip(s1, s2):
        ord_u, ord_v = ord(u), ord(v)
        if char_table[ord_u] is None:
            char_table[ord_u] = ord_v
        elif char_table[ord_u] != ord_v:
            return False

    return True


class IsOneToOneMappingSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(is_one_to_one_mapping('abc', 'bcd'))

    def test_example2(self):
        self.assertFalse(is_one_to_one_mapping('foo', 'bar'))

    def test_empty_mapping(self):
        self.assertTrue(is_one_to_one_mapping('', ''))
        self.assertFalse(is_one_to_one_mapping('', ' '))
        self.assertFalse(is_one_to_one_mapping(' ', ''))

    def test_map_strings_with_different_lengths(self):
        self.assertFalse(is_one_to_one_mapping('abc', 'abcd'))
        self.assertFalse(is_one_to_one_mapping('abcd', 'abc'))
    
    def test_duplicated_chars(self):
        self.assertTrue(is_one_to_one_mapping('aabbcc', '112233'))
        self.assertTrue(is_one_to_one_mapping('abccba', '123321'))

    def test_same_domain(self):
        self.assertTrue(is_one_to_one_mapping('abca', 'bcab'))
        self.assertFalse(is_one_to_one_mapping('abca', 'bcaa'))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Jul 3, 2020 \[Medium\] Largest BST in a Binary Tree
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

**Solution with Recursion:** [https://repl.it/@trsong/Find-Largest-BST-in-a-Binary-Tree](https://repl.it/@trsong/Find-Largest-BST-in-a-Binary-Tree)
```py
import unittest


def largest_bst(root):
    res = largest_bst_recur(root)
    return res.max_bst


def largest_bst_recur(root):
    if not root:
        return BSTResult()

    left_res = largest_bst_recur(root.left)
    right_res = largest_bst_recur(root.right)
    
    left_max = left_res.max_val if root.left else root.val
    right_min = right_res.min_val if root.right else root.val
    is_root_valid = left_max <= root.val <= right_min

    res = BSTResult()
    if is_root_valid and left_res.is_valid and right_res.is_valid:
        res.min_val = left_res.min_val if root.left else root.val
        res.max_val = right_res.max_val if root.right else root.val
        res.max_bst_size = 1 + left_res.max_bst_size + right_res.max_bst_size
        res.max_bst = root
    else:
        res.is_valid = False
        if left_res.max_bst_size > right_res.max_bst_size:
            res.max_bst = left_res.max_bst
        else:
            res.max_bst = right_res.max_bst
    return res


class BSTResult(object):
    def __init__(self):
        self.min_val = None
        self.max_val = None
        self.max_bst_size = 0
        self.max_bst = None
        self.is_valid = True


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

### Jul 2, 2020 \[Hard\] The N Queens Puzzle
---
> **Question:** You have an N by N board. Write a function that, given N, returns the number of possible arrangements of the board where N queens can be placed on the board without threatening each other, i.e. no two queens share the same row, column, or diagonal.


**My thoughts:** Solve the N Queen Problem with Backtracking: place each queen on different columns one by one and test different rows. Mark previous chosen rows and diagonals.

**Solution with Backtracking:** [https://repl.it/@trsong/Solve-the-N-Queen-Problem](https://repl.it/@trsong/Solve-the-N-Queen-Problem)
```py
import unittest

def solve_n_queen(n):
    visited_column = [False] * n
    visited_diagonal1 = [False] * (2 * n)
    visited_diagonal2 = [False] * (2 * n)

    class Context:
        res = 0
        
    def backtrack(r):
        if r >= n:
            Context.res += 1
        else:
            for c in xrange(n):
                d1 = r + c
                d2 = n + r - c
                if visited_column[c] or visited_diagonal1[d1] or visited_diagonal2[d2]:
                    continue
                
                visited_column[c] = True
                visited_diagonal1[d1] = True
                visited_diagonal2[d2] = True
                backtrack(r+1)
                visited_column[c] = False
                visited_diagonal1[d1] = False
                visited_diagonal2[d2] = False

    backtrack(0)
    return Context.res
                    
    
class SolveNQueenSpec(unittest.TestCase):
    def test_one_queen(self):
        self.assertEqual(1, solve_n_queen(1))
    
    def test_two_three_queen(self):
        self.assertEqual(0, solve_n_queen(2))

    def test_three_queens(self):
        self.assertEqual(0, solve_n_queen(3))
        
    def test_four_queen(self):
        self.assertEqual(2, solve_n_queen(4))
    
    def test_eight_queen(self):
        self.assertEqual(92, solve_n_queen(8))


if __name__ == "__main__":
	unittest.main(exit=False)
```

### Jul 1, 2020 LC 680 \[Easy\] Remove Character to Create Palindrome
---
> **Question:** Given a string, determine if you can remove any character to create a palindrome.

**Example 1:**
```py
Input: "abcdcbea"
Output: True 
Explanation: Remove 'e' gives "abcdcba"
```

**Example 2:**
```py
Input: "abccba"
Output: True
```

**Example 3:**
```py
Input: "abccaa"
Output: False
```

**Solution:** [https://repl.it/@trsong/Remove-Character-to-Create-Palindrome](https://repl.it/@trsong/Remove-Character-to-Create-Palindrome)
```py
import unittest

def is_one_palindrome(s):
    return is_palindrome_in_range(s, 0, len(s)-1, 0)
    

def is_palindrome_in_range(s, lo, hi, num_error):
    if num_error > 1:
        return False

    while lo < hi:
        if s[lo] != s[hi]:
            return is_palindrome_in_range(s, lo+1, hi, num_error+1) or is_palindrome_in_range(s, lo, hi-1, num_error+1)
        lo += 1
        hi -= 1
    return True


class IsOnePalindromeSpec(unittest.TestCase):
    def test_example(self):
        # remove e gives abcdcba
        self.assertTrue(is_one_palindrome('abcdcbea'))

    def test_example2(self):
        self.assertTrue(is_one_palindrome('abccba'))

    def test_example3(self):
        self.assertFalse(is_one_palindrome('abccaa'))

    def test_empty_string(self):
        self.assertTrue(is_one_palindrome(''))

    def test_one_char_string(self):
        self.assertTrue(is_one_palindrome('a'))

    def test_palindrome_string(self):
        # remove 4 gives 012343210
        self.assertTrue(is_one_palindrome('0123443210'))

    def test_remove_first_letter(self):
        # remove 9 gives 0123443210
        self.assertTrue(is_one_palindrome('90123443210'))

    def test_remove_last_letter(self):
        # remove 9 gives 012343210
        self.assertTrue(is_one_palindrome('0123432109'))

    def test_impossible_string(self):
        self.assertFalse(is_one_palindrome('abecbea'))

    def test_string_with_duplicated_chars(self):
        # remove second d gives eedee
        self.assertTrue(is_one_palindrome('eedede'))

    def test_longer_example(self):
        # remove second e gives ebcbbccabbacecbbcbe
        self.assertTrue(is_one_palindrome('ebcbbcecabbacecbbcbe'))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 30, 2020 \[Medium\] Maximum Number of Connected Colors
---
> **Question:** Given a grid with cells in different colors, find the maximum number of same color  cells that are connected.
>
> Note: two cells are connected if they are of the same color and adjacent to each other: left, right, top or bottom. To stay simple, we use integers to represent colors:
>
> The following grid have max 4 connected colors. [color 3: (1, 2), (1, 3), (2, 1), (2, 2)]

```py
 [
    [1, 1, 2, 2, 3], 
    [1, 2, 3, 3, 1],
    [2, 3, 3, 1, 2]
 ]
 ```

**My thoughts:** Perform BFS/DFS or Union-Find on all unvisited cells, count its neighbors of same color and mark them as visited.

**Solution with DFS:** [https://repl.it/@trsong/Find-Maximum-Number-of-Connected-Colors](https://repl.it/@trsong/Find-Maximum-Number-of-Connected-Colors)
 ```py
 import unittest

def max_connected_colors(grid):
    n, m = len(grid), len(grid[0])
    visited = [[False for _ in xrange(m)] for _ in xrange(n)]
    max_size = 0
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    for i in xrange(n):
        for j in xrange(m):
            if visited[i][j]:
                continue

            size = 0
            stack = [(i, j)]
            color = grid[i][j]

            while stack:
                r, c = stack.pop()
                if visited[r][c]:
                    continue
                visited[r][c] = True
                size += 1
                for dr, dc in directions:
                    new_r, new_c = r + dr, c + dc
                    if 0 <= new_r < n and 0 <= new_c < m and not visited[new_r][new_c] and grid[new_r][new_c] == color:
                        stack.append((new_r, new_c))
                    
            max_size = max(max_size, size)

    return max_size


class MaxConnectedColorSpec(unittest.TestCase):
    def test_empty_graph(self):
        self.assertEqual(max_connected_colors([[]]), 0)   

    def test_example(self):
        self.assertEqual(max_connected_colors([
            [1, 1, 2, 2, 3],
            [1, 2, 3, 3, 1],
            [2, 3, 3, 1, 2]
        ]), 4)

    def test_disconnected_colors(self):
        self.assertEqual(max_connected_colors([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]), 1)

    def test_cross_shap(self):
        self.assertEqual(max_connected_colors([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ]), 5)

    def test_boundary(self):
        self.assertEqual(max_connected_colors([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]), 8)


if __name__ == '__main__':
    unittest.main(exit=False)
 ```


### June 29, 2020 LC 163 [Medium] Missing Ranges
---
> **Question:** Given a sorted list of numbers, and two integers low and high representing the lower and upper bound of a range, return a list of (inclusive) ranges where the numbers are missing. A range should be represented by a tuple in the format of (lower, upper).

**Example:**
```py
missing_ranges(nums=[1, 3, 5, 10], lower=1, upper=10)
# returns [(2, 2), (4, 4), (6, 9)]
```

**Solution with Binary Search:** [https://repl.it/@trsong/Find-Missing-Ranges](https://repl.it/@trsong/Find-Missing-Ranges)
```py
import unittest
import sys

def missing_ranges(nums, lower, upper):
    if lower > upper:
        return []
    elif not nums or upper < nums[0] or lower > nums[-1]:
        return [(lower, upper)]
    
    res = []
    start = binary_search(nums, lower)
    if start > 0 and nums[start-1] == lower:
        lower += 1

    for i in xrange(start, len(nums)):
        if i > 0 and nums[i-1] == nums[i]:
            continue
        elif nums[i] > upper:
            break
        elif lower < nums[i]:
            res.append((lower, min(nums[i]-1, upper)))
        lower = nums[i] + 1

    if lower <= upper:
        res.append((lower, upper))

    return res


def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo
        

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

### June 28, 2020 \[Medium\] Majority Element
---
> **Question:** A majority element is an element that appears more than half the time. Given a list with a majority element, find the majority element.

**Example:**
```py
majority_element([3, 5, 3, 3, 2, 4, 3])  # gives 3
```

**Althernative Solution with** [***Boyce-Moore Voting Algorithm***](https://trsong.github.io/python/java/2020/02/02/DailyQuestionsFeb/#feb-26-2020-medium-majority-element)

**Solution:** [https://repl.it/@trsong/Find-Majority-Element](https://repl.it/@trsong/Find-Majority-Element)
```py
import unittest
import math

def majority_element(nums):
    if not nums:
        return None

    max_num = max(nums)
    num_bits = int(math.log(max_num, 2)) + 1
    n = len(nums)
    res = 0

    for i in xrange(num_bits):
        count = 0
        for num in nums:
            if num & 1 << i:
                count += 1
        if count > n // 2:
            res |= 1 << i

    # Majority must have set bit > n/2, but converse is not necessarily true
    if nums.count(res) > n // 2:
        return res
    else:
        return None


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


### June 27, 2020 \[Medium\] Find Two Elements Appear Once
---
> **Question:** Given an array of integers in which two elements appear exactly once and all other elements appear exactly twice, find the two elements that appear only once. Can you do this in linear time and constant space?

**Example:**
```py
Input: [2, 4, 6, 8, 10, 2, 6, 10]
Output: [4, 8] order does not matter
 ```

**My thoughts:** XOR has many pretty useful properties:

```py
0 ^ x = x
x ^ x = 0
x ^ y = y ^ x
```

If there is only one unique element, we can simply xor all elements. For example,

```py
7 ^ 3 ^ 5 ^ 5 ^ 4 ^ 3 ^ 4 ^ 8 ^ 8
= 7 ^ 3 ^ (5 ^ 5) ^ 4 ^ 3 ^ 4 ^ (8 ^ 8)
= 7 ^ 3 ^ 4 ^ 3 ^ 4 
= 7 ^ 3 ^ 3 ^ 4 ^ 4
= 7 ^ (3 ^ 3) ^ (4 ^ 4)
= 7
```

But for two unique elements. The idea is to partition the array so that both unique numbers cannot present in the same array. We will need to xor through array twice. 
- The first time we xor all element, we will get `(unique number 1) xor (unique number 2)`. Notice that all bit set by that xor number can only belong to one number not the other. eg. `0b110 xor 0b011 = 0b101`. We can just chose any bit as a flag. Say the least significant bit. `0b001.` Just simply `num & -num` will give the LSB. 
- Then we can partition the original array such that one contians the flag, the other not contains. xor them separately shall yield the result.


**Solution with XOR:** [https://repl.it/@trsong/Find-Two-Elements-Appear-Once](https://repl.it/@trsong/Find-Two-Elements-Appear-Once)
```py
import unittest

def find_two_unique_elements(nums):
    xor_sum = xor_reduce(nums)
    one_set_bit = xor_sum & -xor_sum
    sub_nums1 = filter(lambda num: num & one_set_bit, nums)
    sub_nums2 = filter(lambda num: not (num & one_set_bit), nums)
    return [xor_reduce(sub_nums1), xor_reduce(sub_nums2)]


def xor_reduce(nums):
    return reduce(lambda accu, num: accu ^ num, nums, 0)


class FindTwoUniqueElementSpec(unittest.TestCase):
    def test_example(self):
        self.assertItemsEqual([4, 8], find_two_unique_elements([2, 4, 6, 8, 10, 2, 6, 10]))

    def test_array_with_positive_numbers(self):
        self.assertItemsEqual([7, 1], find_two_unique_elements([7, 3, 5, 5, 4, 3, 4, 8, 8, 1]))

    def test_one_of_elem_is_zero(self):
        self.assertItemsEqual([0, 1], find_two_unique_elements([0, 3, 2, 2, 1, 3]))

    def test_array_with_two_element(self):
        self.assertItemsEqual([42, 41], find_two_unique_elements([42, 41]))

    def test_same_duplicated_number_not_consecutive(self):
        self.assertItemsEqual([5, 7], find_two_unique_elements([1, 2, 1, 5, 3, 2, 3, 7]))

    def test_array_with_negative_elements(self):
        self.assertItemsEqual([-1, -2], find_two_unique_elements([-1, 1, 0, 0, 1, -2]))

    def test_array_with_negative_elements2(self):
        self.assertItemsEqual([3, 0], find_two_unique_elements([-1, 0, 1, -2, 2, -1, 1, -2, 2, 3]))
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 26, 2020 LC 162 \[Easy\] Find a Peak Element
---
> **Question:** Given an unsorted array, in which all elements are distinct, find a "peak" element in `O(log N)` time.
>
> An element is considered a peak if it is greater than both its left and right neighbors. It is guaranteed that the first and last elements are lower than all others.

**Example 1:**
```py
Input: [5, 10, 20, 15]
Output: 20
The element 20 has neighbours 10 and 15,
both of them are less than 20.
```

**Example 2:**
```py
Input: [10, 20, 15, 2, 23, 90, 67]
Output: 20 or 90
The element 20 has neighbours 10 and 15, 
both of them are less than 20, similarly 90 has neighbous 23 and 67.
```

**Solution with Binary Search:** [https://repl.it/@trsong/Find-a-Peak-Element](https://repl.it/@trsong/Find-a-Peak-Element)
```py
import unittest

def find_peak_value(nums):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        # uphill
        if mid == 0 or nums[mid-1] < nums[mid]:
            lo = mid + 1
        else:
            hi = mid
    
    # lo stops at downhill point
    return nums[lo-1]
            

class FindPeakValueSpec(unittest.TestCase):
    def validate_result(self, nums):
        peak = find_peak_value(nums)
        self.assertIn(peak, nums)
        peak_index = nums.index(peak)
        if peak_index > 0:
            self.assertLess(nums[peak_index - 1], peak)
        if peak_index < len(nums) - 1:
            self.assertGreater(peak, nums[peak_index + 1])

    def test_example(self):
        nums = [5, 10, 20, 15]  # solution: 20
        self.validate_result(nums)

    def test_example2(self):
        nums = [10, 20, 15, 2, 23, 90, 67]  # possible solution: 20, 90
        self.validate_result(nums)

    def test_unordered_array(self):
        nums = [0, 24, 23, 22, 17, 15, 35, 26, -1]  # possible solution: 24, 35
        self.validate_result(nums)
        
    def test_unordered_array2(self):
        nums = [3, 2, 8, 7, 19, 27, 5]  # solution: 3
        self.validate_result(nums)

    def test_up_and_down_array(self):
        nums = [-10, 5, -4, 4, -3, 2, -2, 1, -1, 0]  # possible solution: 5
        self.validate_result(nums)

    def test_increasing_array(self):
        nums = [1, 2, 3, 4, 5]  # solution: 5
        self.validate_result(nums)

    def test_decreasing_array(self):
        nums = [3, 2, 1, 0, -1]  # solution: 3
        self.validate_result(nums)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### June 25, 2020 \[Easy\] Largest Product of 3 Elements
---
> **Question:** You are given an array of integers. Return the largest product that can be made by multiplying any 3 integers in the array.

**Example:**
```py
Input: [-4, -4, 2, 8]
Output: 128
Explanation: the largest product can be made by multiplying -4 * -4 * 8 = 128.
```

**My thoughts:** The largest product of three comes from either `max1 * max2 * max3` or `min1 * min2 * max1` where `min1` is 1st min, `max1` is 1st max, vice versa for `max2`, `max3` and `min2`.

**Solution with Priority Queue:** [https://repl.it/@trsong/Find-Largest-Product-of-3-Elements-in-Array](https://repl.it/@trsong/Find-Largest-Product-of-3-Elements-in-Array)
```py
import unittest
from Queue import PriorityQueue

def max_3product(nums):
    max_heap = PriorityQueue()
    min_heap = PriorityQueue()

    for i in xrange(3):
        max_heap.put(-nums[i])
        min_heap.put(nums[i])

    for i in xrange(3, len(nums)):
        if nums[i] > min_heap.queue[0]:
            min_heap.get()
            min_heap.put(nums[i])
        elif nums[i] < -max_heap.queue[0]:
            max_heap.get()
            max_heap.put(-nums[i])
            
    max_heap.get()
    min2 = max_heap.get()
    min1 = max_heap.get()
    max3 = min_heap.get()
    max2 = min_heap.get()
    max1 = min_heap.get()
    return max(max1 * max2 * max3, max1 * min1 * min2)


class Max3ProductSpec(unittest.TestCase):
    def test_all_positive(self):
        # Max1 * Max2 * Max3 = 5 * 4 * 3 = 60
        self.assertEqual(60, max_3product([1, 2, 3, 4, 5]))
    
    def test_all_positive2(self):
        # Max1 * Max2 * Max3 = 6 * 6 * 6 = 216
        self.assertEqual(216, max_3product([2, 3, 6, 1, 1, 6, 3, 2, 1, 6]))

    def test_all_negative(self):
        # Max1 * Max2 * Max3 = -1 * -2 * -3 = -6
        self.assertEqual(-6, max_3product([-5, -4, -3, -2, -1]))
    
    def test_all_negative2(self):
        # Max1 * Max2 * Max3 = -1 * -2 * -3 = -6
        self.assertEqual(-6, max_3product([-1, -5, -2, -4, -3]))
    
    def test_all_negative3(self):
        # Max1 * Max2 * Max3 = -3 * -5 * -6 = -90
        self.assertEqual(-90, max_3product([-10, -3, -5, -6, -20]))

    def test_mixed(self):
        # Min1 * Min2 * Max1 =  -1 * -1 * 3 = 3
        self.assertEqual(3, max_3product([-1, -1, -1, 0, 2, 3]))
    
    def test_mixed2(self):
        # Max1 * Max2 * Max3 = 0 * 0 * -1 = 0
        self.assertEqual(0, max_3product([0, -1, -2, -3, 0]))
    
    def test_mixed3(self):
        # Min1 * Min2 * Max1 =  -6 * -4 * 7 = 168
        self.assertEqual(168, max_3product([1, -4, 3, -6, 7, 0]))
    
    def test_mixed4(self):
        # Max1 * Max2 * Max3 = 3 * 2 * 1 = 6
        self.assertEqual(6, max_3product([-3, 1, 2, 3]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 24, 2020 \[Easy\] Find the K-th Largest Number
---
> **Question:** Find the k-th largest number in a sequence of unsorted numbers. Can you do this in linear time?

**Example:**
```py
Input: 3, [8, 7, 2, 3, 4, 1, 5, 6, 9, 0]
Output: 7
```

**Solution with Quick Select:** [https://repl.it/@trsong/Find-the-K-th-Largest-Number](https://repl.it/@trsong/Find-the-K-th-Largest-Number)
```py
import unittest
import random

def find_kth_max(nums, k):
    n = len(nums)
    if k > n:
        return None

    lo, hi = 0, n - 1
    while True:
        pos = quick_select(nums, lo, hi)
        if pos == n - k:
            return nums[pos]
        elif pos < n - k:
            lo = pos + 1
        else:
            hi = pos - 1
        
    return None


def quick_select(nums, lo, hi):
    pivot_index = random.randint(lo, hi)
    # swap pivot with hi
    nums[hi], nums[pivot_index] = nums[pivot_index], nums[hi]

    i = lo
    for j in xrange(lo, hi):
        if nums[j] < nums[hi]:
            nums[j], nums[i] = nums[i], nums[j]
            i += 1
    
    # swap back pivot to correct position
    nums[i], nums[hi] = nums[hi], nums[i]
    return i

    
class FindKthMaxSpec(unittest.TestCase):
    def test_example(self):
        k, nums = 3, [8, 7, 2, 3, 4, 1, 5, 6, 9, 0]
        expected = 7
        self.assertEqual(expected, find_kth_max(nums, k))

    def test_find_max(self):
        k, nums = 1, [1, 2, 3]
        expected = 3
        self.assertEqual(expected, find_kth_max(nums, k))

    def test_find_min(self):
        k, nums = 5, [1, 2, 3, 4, 5]
        expected = 1
        self.assertEqual(expected, find_kth_max(nums, k))

    def test_array_with_duplicated_elements(self):
        k, nums = 3, [1, 1, 3, 5, 5]
        expected = 3
        self.assertEqual(expected, find_kth_max(nums, k))

    def test_array_with_duplicated_elements2(self):
        k, nums = 4, [1, 1, 1, 1, 1, 1, 1, 1]
        expected = 1
        self.assertEqual(expected, find_kth_max(nums, k)) 

    def test_array_with_duplicated_elements3(self):
        k, nums = 2, [1, 2, 3, 1, 2, 3, 1, 2, 3]
        expected = 3
        self.assertEqual(expected, find_kth_max(nums, k)) 


if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 23, 2020 LC 209 \[Medium\] Minimum Size Subarray Sum
---
> **Question:** Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum >= s. If there isn't one, return 0 instead.

**Example:**
```py
Input: s = 7, nums = [2, 3, 1, 2, 4, 3]
Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.
```

**My thoughts:** Naive solution is to iterate through all possible `(start, end)` intervals to calculate min size of qualified subarrays. However, there is a way to optimize such process. Notice that if `(start, end1)` already has `sum > s`, there is not need to go to another interval `(start, end2)` where `end2 > end1`. That is iterating through the rest of list won't improve the result. Therefore we shortcut start once figure out the qualified end index. 

During iteration, with s fixed, once we figure out the minimum interval `(s, e)` that has `sum < s`. Since `(s, e)` is minimum for all e and some s. If we proceed s, interval `(s+1, e)` won't have `sum > s`.

Thus we can move start and end index during the iteration that will form a sliding window.

**Solution with Sliding Window:** [https://repl.it/@trsong/Find-the-Minimum-Size-Subarray-Sum](https://repl.it/@trsong/Find-the-Minimum-Size-Subarray-Sum)
```py
import unittest
import sys

def min_size_subarray_sum(s, nums):
    accu_sum = 0
    j = 0
    res = sys.maxint
    n = len(nums)

    for i in xrange(n):
        while j < n and accu_sum < s:
            accu_sum += nums[j]
            j += 1

        if accu_sum >= s:
            res = min(res, j - i)
        accu_sum -= nums[i]

    return res if res < sys.maxint else 0


class MinSizeSubarraySumSpec(unittest.TestCase):
    def test_example(self):
        s, nums = 7, [2, 3, 1, 2, 4, 3]
        expected = 2  # [4, 3]
        self.assertEqual(expected, min_size_subarray_sum(s, nums))

    def test_empty_array(self):
        self.assertEqual(0,  min_size_subarray_sum(0, []))

    def test_no_such_subarray_exists(self):
        s, nums = 3, [1, 1]
        expected = 0
        self.assertEqual(expected, min_size_subarray_sum(s, nums))

    def test_no_such_subarray_exists2(self):
        s, nums = 8, [1, 2, 4]
        expected = 0
        self.assertEqual(expected, min_size_subarray_sum(s, nums))
    
    def test_target_subarray_size_greater_than_one(self):
        s, nums = 51, [1, 4, 45, 6, 0, 19]
        expected = 2  # [45, 6]
        self.assertEqual(expected, min_size_subarray_sum(s, nums))

    def test_target_subarray_size_one(self):
        s, nums = 9, [1, 10, 5, 2, 7]
        expected = 1  # [10]
        self.assertEqual(expected, min_size_subarray_sum(s, nums))

    def test_return_min_size_of_such_subarray(self):
        s, nums = 200, [1, 11, 100, 1, 0, 200, 3, 2, 1, 250]
        expected = 1   # [200]
        self.assertEqual(expected, min_size_subarray_sum(s, nums))
   

if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 22, 2020 LC 446 \[Medium\] Count Arithmetic Subsequences
---
> **Question:** Given an array of n positive integers. The task is to count the number of Arithmetic Subsequence in the array. Note: Empty sequence or single element sequence is also Arithmetic Sequence. 

**Example 1:**
```py
Input : arr[] = [1, 2, 3]
Output : 8
Arithmetic Subsequence from the given array are:
[], [1], [2], [3], [1, 2], [2, 3], [1, 3], [1, 2, 3].
```

**Example 2:**
```py
Input : arr[] = [10, 20, 30, 45]
Output : 12
```

**Example 3:**
```py
Input : arr[] = [1, 2, 3, 4, 5]
Output : 23
```

**My thoughts:** this problem can be solved with DP: defined as `dp[i][d]` represents number of arithemtic subsequence end at index `i` with common difference `d`. `dp[i][d] = dp[j][d] + 1 where d = nums[i] - nums[j] for all j < i`. Thus the total number of arithemtic subsequence = sum of `dp[i][d]` for all `i`, `d`. 

**Solution with DP:** [https://repl.it/@trsong/Count-Number-of-Arithmetic-Subsequences](https://repl.it/@trsong/Count-Number-of-Arithmetic-Subsequences)
```py
import unittest
from collections import defaultdict

def count_arithmetic_subsequence(nums):
    n = len(nums)
    # let dp[i][d] represents number of arithemtic subsequence end at index i with common difference d
    #     dp[i][d] = dp[j][d] + 1 where d = nums[i] - nums[j]
    # The total number of arithemtic subsequence = sum of dp[i][d] for all i, d
    dp = [defaultdict(int) for _ in xrange(n)]
    res = n + 1
    for i in xrange(n):
        for j in xrange(i):
            d = nums[i] - nums[j]
            dp[i][d] += dp[j][d] + 1  # (seq ends at j) append nums[i] and (nums[j], nums[i])
        res += sum(dp[i].values())
    return res


class CountArithmeticSubsequenceSpec(unittest.TestCase):
    def test_example1(self):
        # All arithemtic subsequence: [], [1], [2], [3], [1, 2], [2, 3], [1, 3], [1, 2, 3].
        self.assertEqual(8, count_arithmetic_subsequence([1, 2, 3]))

    def test_example2(self):
        self.assertEqual(12, count_arithmetic_subsequence([10, 20, 30, 45]))

    def test_example3(self):
        self.assertEqual(23, count_arithmetic_subsequence([1, 2, 3, 4, 5]))

    def test_empty_array(self):
        self.assertEqual(1, count_arithmetic_subsequence([]))

    def test_array_with_one_element(self):
        self.assertEqual(2, count_arithmetic_subsequence([1]))

    def test_array_with_two_element(self):
        # All arithemtic subsequence: [], [1], [2], [1, 2]
        self.assertEqual(4, count_arithmetic_subsequence([1, 2]))

    def test_array_with_unique_number(self):
        self.assertEqual(8, count_arithmetic_subsequence([1, 1, 1]))

    def test_contains_duplicate_number(self):
        self.assertEqual(12, count_arithmetic_subsequence([2, 1, 1, 1]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 21, 2020 \[Medium\] Sorting Window Range
--- 
> **Question:** Given a list of numbers, find the smallest window to sort such that the whole list will be sorted. If the list is already sorted return (0, 0). 

**Example:**
```py
Input: [2, 4, 7, 5, 6, 8, 9]
Output: (2, 4)
Explanation: Sorting the window (2, 4) which is [7, 5, 6] will also means that the whole list is sorted.
```

**My thoughts:** A sorted array has no min range to sort. So we want first identity the range `(i, j)` that goes wrong, that is, we want to identify first `i` and last `j` that makes array not sorted. ie. smallest `i` such that `nums[i] > nums[i+1]`, largest `j` such that `nums[j] < nums[j-1]`. 

Secondly, range `(i, j)` inclusive is where we should start. And there could be number smaller than `nums[i+1]` and bigger than `nums[j-1]`, therefore we need to figure out how we can release the boundary of `(i, j)` to get `(i', j')` where `i' <= i` and `j' <= j` so that `i'`, `j'` covers those smallest and largest number within `(i, j)`. 

After doing that, we will get smallest range to make original array sorted, the range is `i'` through `j'` inclusive.

**Solution:** [https://repl.it/@trsong/Min-Window-Range-to-Sort](https://repl.it/@trsong/Min-Window-Range-to-Sort)
```py
import unittest

def sort_window_range(nums):
    if not nums:
        return 0, 0
    n = len(nums)
    i = 0
    j = n - 1

    while i < n - 1 and nums[i] <= nums[i+1]:
        i += 1

    if i == n - 1:
        return 0, 0
    
    while j > i and nums[j-1] <= nums[j]:
        j -= 1

    window_min = float('inf')
    window_max = float('-inf')
    for k in xrange(i, j+1):
        if nums[k] < window_min:
            window_min = nums[k]
        if nums[k] > window_max:
            window_max = nums[k]

    while i > 0 and nums[i-1] > window_min:
        i -= 1

    while j < n - 1 and nums[j+1] < window_max:
        j += 1

    return i, j


class SortWindowRangeSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual((2, 4), sort_window_range([2, 4, 7, 5, 6, 8, 9]))

    def test_example1(self):
        self.assertEqual((1, 5), sort_window_range([1, 7, 9, 5, 7, 8, 10]))
        
    def test_example2(self):
        self.assertEqual((3, 8), sort_window_range([10, 12, 20, 30, 25, 40, 32, 31, 35, 50, 60]))

    def test_example3(self):
        self.assertEqual((2, 5), sort_window_range([0, 1, 15, 25, 6, 7, 30, 40, 50]))

    def test_empty_array(self):
        self.assertEqual((0, 0), sort_window_range([]))

    def test_already_sorted_array(self):
        self.assertEqual((0, 0), sort_window_range([1, 2, 3, 4]))

    def test_array_contains_one_elem(self):
        self.assertEqual((0, 0), sort_window_range([42]))

    def test_reverse_sorted_array(self):
        self.assertEqual((0, 3), sort_window_range([4, 3, 2, 1]))

    def test_table_shape_array(self):
        self.assertEqual((2, 5), sort_window_range([1, 2, 3, 3, 3, 2]))

    def test_increase_decrease_then_increase(self):
        self.assertEqual((2, 6), sort_window_range([1, 2, 3, 4, 3, 2, 3, 4, 5, 6]))

    def test_increase_decrease_then_increase2(self):
        self.assertEqual((0, 4), sort_window_range([0, 1, 2, -1, 1, 2]))

    def test_increase_decrease_then_increase3(self):
        self.assertEqual((0, 6), sort_window_range([0, 1, 2, 99, -99, 1, 2]))
        self.assertEqual((0, 6), sort_window_range([0, 1, 2, -99, 99, 1, 2]))
    
    def test_array_contains_duplicated_numbers(self):
        self.assertEqual((0, 5), sort_window_range([1, 1, 1, 0, -1, -1, 1, 1, 1]))

    def test_array_contains_one_outlier(self):
        self.assertEqual((3, 6), sort_window_range([0, 0, 0, 1, 0, 0, 0]))

    def test_array_contains_one_outlier2(self):
        self.assertEqual((0, 3), sort_window_range([0, 0, 0, -1, 0, 0, 0]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 20, 2020 \[Hard\] Increasing Subsequence of Length K
---
> **Question:** Given an int array nums of length n and an int k. Return an increasing subsequence of length k (KIS). Expected time complexity `O(nlogk)`.

**Example 1:**
```py
Input: nums = [10, 1, 4, 8, 2, 9], k = 3
Output: [1, 4, 8] or [1, 4, 9] or [1, 8, 9]
```

**Example 2:**
```py
Input: nums = [10, 1, 4, 8, 2, 9], k = 4
Output: [1, 4, 8, 9]
```

**Solution with DP and Binary Search:** [https://repl.it/@trsong/Increasing-Subsequence-of-Length-K](https://repl.it/@trsong/Increasing-Subsequence-of-Length-K)
```py
import unittest

def increasing_sequence(nums, k):
    if k <= 0 or not nums:
        return []
        
    # The index i in in dp[i] represents exits an increasing subseq of size i+1
    dp = []
    prev_num = {}
    for num in nums:
        i = binary_search(dp, 0, len(dp), num)
        # For any elem append to res, that means there exists a subseq of the same size as res
        if i == len(dp):
            dp.append(num)
        else:
            dp[i] = num
        prev_num[num] = dp[i-1] if i > 0 else None
        if len(dp) == k:
            break

    res = []
    num = dp[-1]
    while num in prev_num:
        res.append(num)
        num = prev_num[num]
    return res[::-1]


def binary_search(nums, lo, hi, target):
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


class IncreasingSequenceSpec(unittest.TestCase):
    def validate_result(self, nums, k):
        subseq = increasing_sequence(nums, k)
        self.assertEqual(k, len(subseq), str(subseq) + " Is not of length " + str(k))

        i = 0
        for num in nums:
            if i < len(subseq) and num == subseq[i]:
                i += 1
        self.assertEqual(len(subseq), i, str(subseq) + " Is not valid subsequence.")
        
        for i in xrange(1, len(subseq)):
            self.assertLessEqual(subseq[i-1], subseq[i], str(subseq) + " Is not increasing subsequene.")

    def test_example(self):
        k, nums = 3, [10, 1, 4, 8, 2, 9]
        self.validate_result(nums, k)  # possible result: [1, 4, 8]

    def test_example2(self):
        k, nums = 4, [10, 1, 4, 8, 2, 9]
        self.validate_result(nums, k)  # possible result: [1, 4, 8, 9]

    def test_empty_sequence(self):
        k, nums = 0, []
        self.validate_result(nums, k)

    def test_longest_increasing_subsequence(self):
        k, nums = 4, [10, 9, 2, 5, 3, 7, 101, 18]
        self.validate_result(nums, k)  # possible result: [2, 3, 7, 101]

    def test_longest_increasing_subsequence_in_second_half_sequence(self):
        k, nums = 4, [1, 2, 3, -2, -1, 0, 1]
        self.validate_result(nums, k)  # possible result: [-2, -1, 0, 1]

    def test_should_return_valid_subsequene(self):
        k, nums = 3, [8, 9, 7, 6, 10]
        self.validate_result(nums, k)  # possible result: [8, 9, 10]


if __name__ == '__main__':
    unittest.main(exit=False)
```


### June 19, 2020 LC 300 \[Hard\] The Longest Increasing Subsequence
---
> **Question:** Given an array of numbers, find the length of the longest increasing **subsequence** in the array. The subsequence does not necessarily have to be contiguous.
>
> For example, given the array `[0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]`, the longest increasing subsequence has length `6` ie. `[0, 2, 6, 9, 11, 15]`.

**Solution with DP and Binary Search:** [https://repl.it/@trsong/Find-the-Longest-Increasing-Subsequence](https://repl.it/@trsong/Find-the-Longest-Increasing-Subsequence)
```py
import unittest

def longest_increasing_subsequence(sequence):
    # The index i in in dp[i] represents exits an increasing subseq of size i+1
    dp = []
    for num in sequence:
        i = binary_search(dp, 0, len(dp), num)
        # For any elem append to res, that means there exists a subseq of the same size as res
        if i == len(dp):
            dp.append(num)
        else:
            dp[i] = num
    return len(dp)


def binary_search(nums, lo, hi, target):
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


class LongestIncreasingSubsequnceSpec(unittest.TestCase):
    def test_empty_sequence(self):
        self.assertEqual(0, longest_increasing_subsequence([]))

    def test_last_elem_is_local_max(self):
        seq = [1, 2, 3, 0, 2]
        expected = 3  # [1, 2, 3]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_last_elem_is_global_max(self):
        seq = [1, 2, 3, 0, 6]
        expected = 4  # [1, 2, 3, 6]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_longest_increasing_subsequence_in_first_half_sequence(self):
        seq = [4, 5, 6, 7, 1, 2, 3]
        expected = 4  # [4, 5, 6, 7]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_longest_increasing_subsequence_in_second_half_sequence(self):
        seq = [1, 2, 3, -2, -1, 0, 1]
        expected = 4  # [-2, -1, 0, 1]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_sequence_in_up_down_up_pattern(self):
        seq = [1, 2, 3, 2, 4]
        expected = 4  # [1, 2, 2, 4]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_sequence_in_up_down_up_pattern2(self):
        seq = [1, 2, 3, -1, 0]
        expected = 3  # [1, 2, 3]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_sequence_in_down_up_down_pattern(self):
        seq = [4, 3, 5]
        expected = 2  # [3, 5]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_sequence_in_down_up_down_pattern2(self):
        seq = [4, 0, 1]
        expected = 2  # [0, 1]
        self.assertEqual(expected, longest_increasing_subsequence(seq))

    def test_multiple_result(self):
        seq = [10, 9, 2, 5, 3, 7, 101, 18]
        expected = 4  # [2, 3, 7, 101]
        self.assertEqual(expected, longest_increasing_subsequence(seq))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### June 18, 2020 \[Medium\] Longest Common Subsequence
---
> **Question:** Given two sequences, find the length of longest subsequence present in both of them. 
>
> A subsequence is a sequence that appears in the same relative order, but not necessarily contiguous.

**Example 1:**

```
Input:  "ABCD" and "EDCA"
Output:  1
	
Explanation:
LCS is 'A' or 'D' or 'C'
```

**Example 2:**

```
Input: "ABCD" and "EACB"
Output:  2
	
Explanation: 
LCS is "AC"
```

**My thoughts:** This problem is similar to Levenshtein Edit Distance in multiple ways:

1. If the last digit of each string matches each other, i.e. lcs(seq1 + s, seq2 + s) then result = 1 + lcs(seq1, seq2).
2. If the last digit not matches,  i.e. lcs(seq1 + s, seq2 + p), then res is either ignore s or ignore q. Just like insert a whitespace or remove a letter from edit distance, which gives max(lcs(seq1, seq2 + p), lcs(seq1 + s, seq2))

The difference between this question and edit distance is that each subsequence does not allow switching to different letters.
 

**Solution with DP:** [https://repl.it/@trsong/Find-the-Longest-Common-Subsequence](https://repl.it/@trsong/Find-the-Longest-Common-Subsequence)
```py
import unittest

def lcs(seq1, seq2):
    n, m = len(seq1), len(seq2)
    # Let dp[n][m] represents lcs for seq1[:n] and seq[:m]
    # dp[n][m] = dp[n-1][m-1] + 1  if seq[n-1] matches seq2[m-1]
    #          = max(dp[n-1][m], dp[n][m-1]) otherwise
    dp = [[0 for _ in xrange(m+1)] for _ in xrange(n+1)]
    for i in xrange(1, n+1):
        for j in xrange(1, m+1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]


class LCSSpec(unittest.TestCase):
    def test_empty_sequences(self):
        self.assertEqual(0, lcs("", ""))

    def test_match_last_position(self):
        self.assertEqual(1, lcs("abcdz", "efghijz"))  # a

    def test_match_first_position(self):
        self.assertEqual(1, lcs("aefgh", "aijklmnop"))  # a

    def test_off_by_one_position(self):
        self.assertEqual(4, lcs("10101", "01010"))  # 0101

    def test_off_by_one_position2(self):
        self.assertEqual(4, lcs("12345", "1235"))  # 1235

    def test_off_by_one_position3(self):
        self.assertEqual(3, lcs("1234", "1243"))  # 124

    def test_off_by_one_position4(self):
        self.assertEqual(4, lcs("12345", "12340"))  # 1234

    def test_multiple_matching(self):
        self.assertEqual(5, lcs("afbgchdie",
                                "__a__b_c__de___f_g__h_i___"))  # abcde

    def test_ascending_vs_descending(self):
        self.assertEqual(1, lcs("01234", "_4__3___2_1_0__"))  # 0

    def test_multiple_ascending(self):
        self.assertEqual(6, lcs("012312342345", "012345"))  # 012345

    def test_multiple_descending(self):
        self.assertEqual(5, lcs("5432432321", "54321"))  # 54321

    def test_example(self):
        self.assertEqual(2, lcs("ABCD", "EACB"))  # AC


if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 17, 2020 \[Medium\] Longest Substring without Repeating Characters
---
> **Question:** Given a string, find the length of the longest substring without repeating characters.
>
> **Note:** Can you find a solution in linear time?
 
**Example:**
```py
lengthOfLongestSubstring("abrkaabcdefghijjxxx") # => 10 as len("abcdefghij") == 10
```

**My thoughts:** This is a typical sliding window problem. The idea is to mantain a last occurance map while proceeding the sliding window. Such window is bounded by indices `(i, j)`, whenever we process next character j, we check the last occurance map to see if the current character `a[j]` is duplicated within the window `(i, j)`, ie. `i <= k < j`, if that's the case, we move `i` to `k + 1` so that `a[j]` no longer exists in window. And we mantain the largest window size `j - i + 1` as the longest substring without repeating characters.

**Solution with Sliding Window:** [https://repl.it/@trsong/Find-Longest-Substring-without-Repeating-Characters](https://repl.it/@trsong/Find-Longest-Substring-without-Repeating-Characters)
```py
import unittest

def longest_nonrepeated_substring(s):
    max_window = 0
    last_occur = {}
    i = -1

    for j, in_char in enumerate(s):
        i = max(i, last_occur.get(in_char, -1))
        window = j - i
        max_window = max(max_window, window)
        last_occur[in_char] = j

    return max_window


class LongestNonrepeatedSubstringSpec(unittest.TestCase):
    def test_example(self):
        s = "abrkaabcdefghijjxxx"
        expected = 10  # "abcdefghij"
        self.assertEqual(expected, longest_nonrepeated_substring(s))

    def test_empty_string(self):
        self.assertEqual(0, longest_nonrepeated_substring(""))

    def test_string_with_repeated_characters(self):
        s = "aabbafacbbcacbfa"
        expected = 4  # "facb"
        self.assertEqual(expected, longest_nonrepeated_substring(s))

    def test_some_random_string(self):
        s = "ABDEFGABEF"
        expected = 6  # "ABDEFG"
        self.assertEqual(expected, longest_nonrepeated_substring(s))

    def test_all_repated_characters(self):
        s = "aaa"
        expected = 1  # "a"
        self.assertEqual(expected, longest_nonrepeated_substring(s))

    def test_non_repated_characters(self):
        s = "abcde"
        expected = 5  # "abcde"
        self.assertEqual(expected, longest_nonrepeated_substring(s))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 16, 2020 LT 386 \[Medium\] Longest Substring with At Most K Distinct Characters
---
> **Question:** Given a string, find the longest substring that contains at most k unique characters. 
> 
> For example, given `"abcbbbbcccbdddadacb"`, the longest substring that contains 2 unique character is `"bcbbbbcccb"`.


**Solution with Sliding Window:** [https://repl.it/@trsong/Find-Longest-Substring-with-At-Most-K-Distinct-Characters#main.py](https://repl.it/@trsong/Find-Longest-Substring-with-At-Most-K-Distinct-Characters#main.py)
```py
import unittest

def longest_substr_with_k_distinct_chars(s, k):
    max_window = 0
    max_window_start = 0
    window_char_freq = {}
    i = 0
    for j, in_char in enumerate(s):
        window_char_freq[in_char] = window_char_freq.get(in_char, 0) + 1

        while len(window_char_freq) > k:
            out_char = s[i]
            window_char_freq[out_char] -= 1
            if window_char_freq[out_char] == 0:
                del window_char_freq[out_char]
            i += 1

        new_window = j - i + 1
        if new_window > max_window:
            max_window = new_window
            max_window_start = i

    return s[max_window_start: max_window_start+max_window]


class LongestSubstrWithKDistinctCharSpec(unittest.TestCase):
    def test_example(self):
        k, s = 2, "abcbbbbcccbdddadacb"
        expected = "bcbbbbcccb"
        self.assertEqual(expected, longest_substr_with_k_distinct_chars(s, k))

    def test_empty_string(self):
        self.assertEqual("", longest_substr_with_k_distinct_chars("", 3))

    def test_substr_with_3_distinct_chars(self):
        k, s = 3, "abcadcacacaca"
        expected = "cadcacacaca"
        self.assertEqual(expected, longest_substr_with_k_distinct_chars(s, k))

    def test_substr_with_3_distinct_chars2(self):
        k, s = 3, "eceba"
        expected = "eceb"
        self.assertEqual(expected, longest_substr_with_k_distinct_chars(s, k))

    def test_multiple_solutions(self):
        k, s = 4, "WORLD" 
        res = longest_substr_with_k_distinct_chars(s, k)
        sol1 = "WORL"
        sol2 = "ORLD"
        self.assertTrue(sol1 == res or sol2 == res)

    def test_complicated_input(self):
        s = "abcbdbdbbdcdabd"
        k2, sol2 = 2, "bdbdbbd"
        k3, sol3 = 3, "bcbdbdbbdcd"
        k5, sol5 = 5, "abcbdbdbbdcdabd"
        self.assertEqual(sol2, longest_substr_with_k_distinct_chars(s, k2))
        self.assertEqual(sol3, longest_substr_with_k_distinct_chars(s, k3))
        self.assertEqual(sol5, longest_substr_with_k_distinct_chars(s, k5))
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 15, 2020 LC 727 \[Hard\] Minimum Window Subsequence
---
> **Question:** Given strings S and T, find the minimum (contiguous) substring W of S, so that T is a subsequence of W.
>
> If there is no such window in S that covers all characters in T, return the empty string "". If there are multiple such minimum-length windows, return the one with the left-most starting index.

**Example:**
```py
Input: 
S = "abcdebdde", T = "bde"
Output: "bcde"

Explanation: 
"bcde" is the answer because it occurs before "bdde" which has the same length.
"deb" is not a smaller window because the elements of T in the window must occur in order.
```

**My thoughts:** Have you noticed the pattern that substring of s always has the same first char as t. i.e. `s = "abcdebdde", t = "bde", substring = "bcde", substring[0] == t[0]`,  we can take advantage of that to keep track of previous index such that t[0] == s[index] and we can do that recursively for the rest of t and s. We get the following recursive definition

```py
Let dp[i][j] = index where index represents index such that s[index:i] has subsequence t[0:j].

dp[i][j] = dp[i-1][j-1] if there s[i-1] matches t[j-1] 
         = dp[i-1][j]   otherwise
```

And the final solution is to find index where `len of t <= index <= len of s` such that `index - dp[index][len of t]` i.e. the length of substring, reaches minimum. 

**Solution with DP:** [https://repl.it/@trsong/Find-Minimum-Window-Subsequence](https://repl.it/@trsong/Find-Minimum-Window-Subsequence)
```py
import unittest

def min_window_subsequence(s, t):
    if len(s) < len(t):
        return ""

    n, m = len(s), len(t)
    # Let dp[n][m] represents min index such that s[index:n] contains subsequence t[:m]
    # dp[n][m] = dp[n-1][m-1] if s[n-1] == t[m-1]
    #          = dp[n][m-1]   otherwise
    dp = [[-1 for _ in xrange(m+1)] for _ in xrange(n+1)]
    for i in xrange(n+1):
        dp[i][0] = i

    for i in xrange(1, n+1):
        for j in xrange(1, m+1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = dp[i-1][j]

    start = -1
    min_window = float('inf')
    for i in xrange(m, n+1):
        if dp[i][m] == -1:
            continue
        
        index = dp[i][m]
        window = i - index
        if window < min_window:
            min_window = window
            start = index

    return s[start: start + min_window] if start != -1 else ""
        

class MinWindowSubsequenceSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(min_window_subsequence("abcdebdde", "bde"), "bcde")

    def test_target_too_long(self):
        self.assertEqual(min_window_subsequence("a", "aaa"), "")

    def test_duplicated_char_in_target(self):
        self.assertEqual(min_window_subsequence("abbbbabbbabbababbbb", "aa"), "aba")

    def test_duplicated_char_but_no_matching(self):
        self.assertEqual(min_window_subsequence("ccccabbbbabbbabbababbbbcccc", "aca"), "")

    def test_match_last_char(self):
        self.assertEqual(min_window_subsequence("abcdef", "f"), "f")

    def test_match_first_char(self):
        self.assertEqual(min_window_subsequence("abcdef", "a"), "a")

    def test_equal_length_string(self):
        self.assertEqual(min_window_subsequence("abc", "abc"), "abc")
        self.assertEqual(min_window_subsequence("abc", "bca"), "")


if __name__ == '__main__':
    unittest.main(exit=False)
```


### June 14, 2020 LC 239 \[Medium\] Sliding Window Maximum
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

**Solution with Double-Ended Queue:** [https://repl.it/@trsong/Find-Sliding-Window-Maximum](https://repl.it/@trsong/Find-Sliding-Window-Maximum)
```py
from collections import deque
import unittest

def max_sliding_window(nums, k):
    dq = deque()
    res = []
    for j, in_num in enumerate(nums):
        while dq and nums[dq[-1]] <= in_num:
            # Maintain ascending order in dq
            dq.pop()
        dq.append(j)

        if dq[0] <= j - k:
            dq.popleft()

        if j >= k - 1:
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
    unittest.main(exit=False)
```

### June 13, 2020 LC 438 \[Medium\] Anagram Indices Problem
---
> **Question:**  Given a word W and a string S, find all starting indices in S which are anagrams of W.
>
> For example, given that W is `"ab"`, and S is `"abxaba"`, return `0`, `3`, and `4`.


**Solution with Sliding Window of Fixed Size:** [https://repl.it/@trsong/Find-All-Anagram-Indices](https://repl.it/@trsong/Find-All-Anagram-Indices)

```py
import unittest

def find_anagrams(word, s):
    if not s or len(word) < len(s):
        return []

    # Initially owing a lot of chars
    char_debt = {}
    for c in s:
        char_debt[c] = char_debt.get(c, 0) + 1
    
    res = []
    for j, in_char in enumerate(word):
        # Accumulate incomming chars
        char_debt[in_char] = char_debt.get(in_char, 0) - 1
        if char_debt[in_char] == 0:
            del char_debt[in_char]

        # Removing outgoing chars
        i = j - len(s)
        if i >= 0:
            out_char = word[i]
            char_debt[out_char] = char_debt.get(out_char, 0) + 1
            if char_debt[out_char] == 0:
                del char_debt[out_char]

        if not char_debt:
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

### June 12, 2020 LC 76 \[Hard\] Minimum Window Substring
---
> **Question:** Given a string and a set of characters, return the shortest substring containing all the characters in the set.
>
> For example, given the string `"figehaeci"` and the set of characters `{a, e, i}`, you should return `"aeci"`.
>
> If there is no substring containing all the characters in the set, return null.

**My thoughts:** Most substring problem can be solved with Sliding Window method which requires two pointers represent boundaries of a window as well as a map storing certain properties associated w/ letter in substring (in this problem, the count of letter).

In this problem, we first find the count of letter requirement of each letter in target. And we define two pointers: `start`, `end`. For each incoming letters, we proceed end and decrease the letter requirement of that letter; once all letter requirement satisfies, we proceed start that will eliminate unnecessary letters to shrink the window size for sure; however it might also introduces new letter requirement and then we proceed end and wait for all letter requirement satisfies again.

We do that over and over and record min window along the way gives the final result.

**Solution with Sliding Window:** [https://repl.it/@trsong/Find-Minimum-Window-Substring](https://repl.it/@trsong/Find-Minimum-Window-Substring)
```py
import unittest
import sys

def min_window_substring(source, target):
    target_histogram = {}
    for c in target:
        target_histogram[c] = target_histogram.get(c, 0) + 1

    min_window_start = 0
    min_window = sys.maxint
    balance = len(target)
    start = 0

    for end, end_char in enumerate(source):
        if end_char in target_histogram:
            if target_histogram[end_char] > 0:
                balance -= 1
            target_histogram[end_char] -= 1

        while balance == 0:
            window = end - start + 1
            if window < min_window:
                min_window = window
                min_window_start = start

            start_char = source[start]
            if start_char in target_histogram:
                if target_histogram[start_char] == 0:
                    balance += 1
                target_histogram[start_char] += 1

            start += 1

    return "" if min_window == sys.maxint else source[min_window_start: min_window_start + min_window]
                

class MinWindowSubstringSpec(unittest.TestCase):
    def test_example(self):
        source, target, expected = "ADOBECODEBANC", "ABC", "BANC"
        self.assertEqual(expected, min_window_substring(source, target))

    def test_no_matching_due_to_missing_letters(self):
        source, target, expected = "CANADA", "CAB", ""
        self.assertEqual(expected, min_window_substring(source, target))

    def test_no_matching_due_to_target_too_short(self):
        source, target, expected = "USD", "UUSD", ""
        self.assertEqual(expected, min_window_substring(source, target))
    
    def test_target_string_with_duplicated_letters(self):
        source, target, expected = "BANANAS", "ANANS", "NANAS"
        self.assertEqual(expected, min_window_substring(source, target))

    def test_matching_window_in_the_middle_of_source(self):
        source, target, expected = "AB_AABB_AB_BAB_ABB_BB_BACAB", "ABB", "ABB"
        self.assertEqual(expected, min_window_substring(source, target))

    def test_matching_window_in_different_order(self):
        source, target, expected = "CBADDBBAADBBAAADDDCCBBA", "AAACCCBBBBDDD", "CBADDBBAADBBAAADDDCC"
        self.assertEqual(expected, min_window_substring(source, target))


if __name__ == '__main__':
    unittest.main(exit=False)   
```


### June 11, 2020 \[Medium\] Longest Subarray with Sum Divisible by K
---
> **Question:** Given an arr[] containing n integers and a positive integer k. The problem is to find the length of the longest subarray with sum of the elements divisible by the given value k.

**Example:**
```py
Input : arr[] = {2, 7, 6, 1, 4, 5}, k = 3
Output : 4
The subarray is {7, 6, 1, 4} with sum 18, which is divisible by 3.
```

**My thoughts:** Recall the way to efficiently calculate subarray sum is to calculate prefix sum, `prefix_sum[i] = arr[0] + arr[1] + ... + arr[i] and prefix_sum[i] = prefix_sum[i-1] + arr[i]`. The subarray sum between index i and j, `arr[i] + arr[i+1] + ... + arr[j] = prefix_sum[j] - prefix_sum[i-1]`.

But this question is asking to find subarray whose sum is divisible by 3, that is,  `(prefix_sum[j] - prefix_sum[i-1]) mod k == 0 ` which implies `prefix_sum[j] % k == prefix_sum[j-1] % k`. So we just need to generate prefix_modulo array and find i,j such that `j - i reaches max` and `prefix_modulo[j] == prefix_modulo[i-1]`. As `j > i` and we must have value of `prefix_modulo[i-1]` already when we reach j. We can use a map to store the first occurance of certain prefix_modulo. This feels similar to Two-Sum question in a sense that we use map to store previous reached element and is able to quickly tell if current element satisfies or not.


**Solution:** [https://repl.it/@trsong/Find-Longest-Subarray-with-Sum-Divisible-by-K](https://repl.it/@trsong/Find-Longest-Subarray-with-Sum-Divisible-by-K)
```py
import unittest

def longest_subarray(nums, k):
    prefix_sum = {0: -1}
    res = 0
    accu = 0
    for i, num in enumerate(nums):
        accu += num
        accu %= k
        if accu not in prefix_sum:
            prefix_sum[accu] = i
        else:
            res = max(res, i - prefix_sum[accu])
    return res


class LongestSubarraySpec(unittest.TestCase):
    def test_example(self):
        # Modulo 3, prefix array = [2, 0, 0, 1, 2, 1]. max (end - start) = 4 such that prefix[end] - prefix[start-1] = 0
        self.assertEqual(longest_subarray([2, 7, 6, 1, 4, 5], 3), 4)  # sum([7, 6, 1, 4]) = 18 (18 % 3 = 0)

    def test_empty_array(self):
        self.assertEqual(longest_subarray([], 10), 0)
    
    def test_no_existance_of_such_subarray(self):
        self.assertEqual(longest_subarray([1, 2, 3, 4], 11), 0)
    
    def test_entire_array_qualify(self):
        # Modulo 4, prefix array: [0, 1, 2, 3, 3, 2, 1, 0]. max (end - start) = 8 such that prefix[end] - prefix[start-1] = 0
        self.assertEqual(longest_subarray([0, 1, 1, 1, 0, -1, -1, -1], 4), 8)  # entire array sum = 0
        self.assertEqual(longest_subarray([4, 5, 9, 17, 8, 3, 7, -1], 4), 8)  # entire array sum = 52 (52 % 4 = 0)

    def test_unique_subarray(self):
        # Modulo 6, prefix array: [0, 1, 1, 2, 3, 2, 4, 4, 5]. max (end - start) = 2 such that prefix[end] - prefix[start-1] = 0
        self.assertEqual(longest_subarray([0, 1, 0, 1, 1, -1, 2, 0, 1], 6), 2)  #  sum([1, -1]) = 0
        self.assertEqual(longest_subarray([6, 7, 12, 7, 13, 5, 8, 36, 19], 6), 2)  #  sum([13, 5]) = 18 (18 % 6 = 0)
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 10, 2020  LC 621 \[Medium\] Task Scheduler
---
> **Question:** Given a char array representing tasks CPU need to do. It contains capital letters A to Z where different letters represent different tasks. Tasks could be done without original order. Each task could be done in one interval. For each interval, CPU could finish one task or just be idle.
>
> However, there is a non-negative cooling interval n that means between two same tasks, there must be at least n intervals that CPU are doing different tasks or just be idle.
>
> You need to return the least number of intervals the CPU will take to finish all the given tasks.

**Example:**
```py
Input: tasks = ["A", "A", "A", "B", "B", "B"], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B.
```

**My thoughts:** Treat n+1 as the size of each window. For each window, we try to fit as many tasks as possible following the max number of remaining tasks. If all tasks are chosen, we instead use idle.

**Greedy Solution:** [https://repl.it/@trsong/LC-621-Task-Scheduler](https://repl.it/@trsong/LC-621-Task-Scheduler)
```py
import unittest
from Queue import PriorityQueue

def least_interval(tasks, n):
    task_histogram = {}
    for task in tasks:
        task_histogram[task] = task_histogram.get(task, 0) + 1

    max_heap = PriorityQueue()
    for task, count in task_histogram.items():
        max_heap.put((-count, task))

    res = 0
    while not max_heap.empty():
        next_round_tasks = []
        for _ in xrange(n+1):
            if max_heap.empty() and not next_round_tasks:
                break
            elif not max_heap.empty():
                neg_count, task = max_heap.get()
                remain_task = -neg_count - 1

                if remain_task > 0:
                    next_round_tasks.append((task, remain_task))
            res += 1

        for task, count in next_round_tasks:
            max_heap.put((-count, task))

    return res


class LeastIntervalSpec(unittest.TestCase):
    def test_example(self):
        tasks = ["A", "A", "A", "B", "B", "B"]
        n = 2
        self.assertEqual(least_interval(tasks, n), 8) # A -> B -> idle -> A -> B -> idle -> A -> B

    def test_no_tasks(self):
        self.assertEqual(least_interval([], 0), 0)
        self.assertEqual(least_interval([], 2), 0)
    
    def test_same_task_and_idle(self):
        tasks = ["A", "A", "A"]
        n = 1
        self.assertEqual(least_interval(tasks, n), 5)  # A -> idle -> A -> idle -> A

    def test_three_kind_tasks_no_idle(self):
        tasks = ["A", "B", "A", "C"]
        n = 1
        self.assertEqual(least_interval(tasks, n), 4)  # A -> B -> A -> C
    
    def test_three_kind_tasks_with_one_idle(self):
        tasks = ["A", "A", "A", "B", "C", "C"]
        n = 2
        self.assertEqual(least_interval(tasks, n), 7)  # A -> C -> B -> A -> C -> idle -> A

    def test_each_kind_one_task(self):
        tasks = ["A", "B", "C", "D"]
        n = 10
        self.assertEqual(least_interval(tasks, n), 4)  # A -> B -> C -> D


if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 9, 2020 \[Medium\] Circle of Chained Words
---
> **Question:** Two words can be 'chained' if the last character of the first word is the same as the first character of the second word.
>
> Given a list of words, determine if there is a way to 'chain' all the words in a circle.

**Example:**
```py
Input: ['eggs', 'karat', 'apple', 'snack', 'tuna']
Output: True
Explanation:
The words in the order of ['apple', 'eggs', 'snack', 'karat', 'tuna'] creates a circle of chained words.
```

**My thoughts:** Treat each non-empty word as an edge in a directed graph with vertices being the first and last letter of the word. Now, pick up any letter as a starting point. Perform DFS and remove any edge we visited from the graph. Check if all edges are used. And make sure the vertex we stop at is indeed the starting point. If all above statisfied, then there exists a cycle that chains all words. 

**Solution with DFS:** [https://repl.it/@trsong/Contains-Circle-of-Chained-Words](https://repl.it/@trsong/Contains-Circle-of-Chained-Words)
```py
import unittest

def exists_cycle(words):
    if not words:
        return False

    neighbors = {}
    for word in words:
        if not word: continue
        u, v = word[0], word[-1]
        if u not in neighbors:
            neighbors[u] = {}
        neighbors[u][v] = neighbors[u].get(v, 0) + 1

    if not neighbors:
        return False
        
    start = next(neighbors.iterkeys())
    stack = [start]
    while stack and neighbors:
        cur = stack.pop()
        if cur not in neighbors:
            return False
            
        nb = next(neighbors[cur].iterkeys())
        neighbors[cur][nb] -= 1
        if neighbors[cur][nb] == 0:
            del neighbors[cur][nb]
            if not neighbors[cur]:
                del neighbors[cur]
        
        stack.append(nb)

    return not neighbors and len(stack) == 1 and stack[0] == start
            

class ExistsCycleSpec(unittest.TestCase):
    def test_example(self):
        words = ['eggs', 'karat', 'apple', 'snack', 'tuna']
        self.assertTrue(exists_cycle(words)) # ['apple', 'eggs', 'snack', 'karat', 'tuna']

    def test_empty_words(self):
        words = []
        self.assertFalse(exists_cycle(words))
    
    def test_not_contains_cycle(self):
        words = ['ab']
        self.assertFalse(exists_cycle(words))

    def test_not_contains_cycle2(self):
        words = ['']
        self.assertFalse(exists_cycle(words))

    def test_not_exist_cycle(self):
        words = ['ab', 'c', 'c', 'def', 'gh']
        self.assertFalse(exists_cycle(words))

    def test_exist_cycle_but_not_chaining_all_words(self):
        words = ['ab', 'be', 'bf', 'bc', 'ca']
        self.assertFalse(exists_cycle(words))
    
    def test_exist_cycle_but_not_chaining_all_words2(self):
        words = ['ab', 'ba', 'bc', 'ca']
        self.assertFalse(exists_cycle(words))

    def test_duplicate_words_with_cycle(self):
        words = ['ab', 'bc', 'ca', 'ab', 'bd', 'da' ]
        self.assertTrue(exists_cycle(words))

    def test_contains_mutiple_cycles(self):
        words = ['ab', 'ba', 'ac', 'ca']
        self.assertTrue(exists_cycle(words))

    def test_disconnect_graph(self):
        words = ['ab', 'ba', 'cd', 'de', 'ec']
        self.assertFalse(exists_cycle(words))

    def test_conains_empty_string(self):
        words2 = ['', 'a', '', '', 'a']
        self.assertTrue(exists_cycle(words2))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 8, 2020 \[Medium\] Group Words that are Anagrams
---
> **Question:** Given a list of words, group the words that are anagrams of each other. (An anagram are words made up of the same letters).

**Example:**
```py
Input: ['abc', 'bcd', 'cba', 'cbd', 'efg']
Output: [['abc', 'cba'], ['bcd', 'cbd'], ['efg']]
```

**My thoughts:** Notice that two words are anagrams of each other if both of them be equal after sort and should have same prefix. eg. `'bcda' => 'abcd'` and `'cadb' => 'abcd'`. We can sort each word in the list and then insert those sorted words into a trie. Finally, we can perform a tree traversal to get all words with same prefix and those words will be words that are anagrams of each other.

**Solution with Trie:** [https://repl.it/@trsong/Group-Anagrams](https://repl.it/@trsong/Group-Anagrams)
```py
import unittest

class Trie(object):
    def __init__(self):
        self.children = None
        self.word_indices = None

    def insert_word(self, word, index):
        p = self
        for c in sorted(word):
            p.children = p.children or {}
            p.children[c] = p.children[c] if c in p.children else Trie()
            p = p.children[c]
        p.word_indices = p.word_indices or []
        p.word_indices.append(index)

    def words_groupby_anagram(self, words):
        stack = [self]
        res = []
        while stack:
            cur = stack.pop()

            if cur.word_indices:
                anagrams = map(lambda i: words[i], cur.word_indices)
                res.append(anagrams)
            
            if cur.children:
                stack.extend(cur.children.values())
        
        return res


def group_by_anagram(words):
    t = Trie()
    for i, word in enumerate(words):
        t.insert_word(word, i)

    return t.words_groupby_anagram(words)


class GroupyByAnagramSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        for l in result:
            l.sort()
        result.sort()
        for l in expected:
            l.sort()
        expected.sort()
        self.assertEqual(expected, result)

    def test_example(self):
        input = ['abc', 'bcd', 'cba', 'cbd', 'efg']
        output = [['abc', 'cba'], ['bcd', 'cbd'], ['efg']]
        self.assert_result(output, group_by_anagram(input))

    def test_empty_word_list(self):
        self.assert_result([], group_by_anagram([]))

    def test_contains_duplicated_words(self):
        input = ['a', 'aa', 'aaa', 'a', 'aaa', 'aa', 'aaa']
        output = [['a', 'a'], ['aa', 'aa'], ['aaa', 'aaa', 'aaa']]
        self.assert_result(output, group_by_anagram(input))

    def test_contains_duplicated_words2(self):
        input = ['abc', 'acb', 'abcd', 'dcba', 'abc', 'abcd', 'a']
        output = [['a'], ['abc', 'acb', 'abc'], ['abcd', 'dcba', 'abcd']]
        self.assert_result(output, group_by_anagram(input))

    def test_contains_empty_word(self):
        input = ['', 'a', 'b', 'c', '', 'bc', 'ca', '', 'ab']
        output = [['', '', ''], ['a'], ['b'], ['c'], ['ab'], ['ca'], ['bc']]
        self.assert_result(output, group_by_anagram(input))

    def test_word_with_duplicated_letters(self):
        input = ['aabcde', 'abbcde', 'abccde', 'abcdde', 'abcdee', 'abcdea']
        output = [['aabcde', 'abcdea'], ['abbcde'], ['abccde'], ['abcdde'], ['abcdee']]
        self.assert_result(output, group_by_anagram(input))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 7, 2020 \[Easy\] Generate All Possible Subsequences
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

> Note that `zx` is not a valid subsequence since it is not in the order of the given string.


**Solution with Backtracking:** [https://repl.it/@trsong/Generate-All-Possible-Subsequences](https://repl.it/@trsong/Generate-All-Possible-Subsequences)
```py
import unittest


def generate_subsequences(s):
    if not s:
        return ['']

    res = set()
    backtrack(s, res, "", -1, len(s))
    return res


def backtrack(s, res, accu_str, prev_index, remain):
    if accu_str:
        res.add(accu_str)

    if remain == 0:
        return

    n = len(s)
    for i in xrange(prev_index + 1, n):
        updated_str = accu_str + s[i]
        backtrack(s, res, updated_str, i, remain - 1)


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
    unittest.main(exit=False)
```

### June 6, 2020 \[Easy\] First and Last Indices of an Element in a Sorted Array
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

**Solution with Binary Search:** [https://repl.it/@trsong/Find-First-and-Last-Indices-of-an-Element-in-a-Sorted-Array](https://repl.it/@trsong/Find-First-and-Last-Indices-of-an-Element-in-a-Sorted-Array)
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
    
    hi_plus = binary_search(nums, target, exclusive=True)

    return (lo, hi_plus - 1)


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


### June 5, 2020 \[Medium\] Minimum Number of Operations
---
> **Question:** You are only allowed to perform 2 operations:
> - either multiply a number by 2;
> - or subtract a number by 1. 
>
> Given a number `x` and a number `y`, find the minimum number of operations needed to go from `x` to `y`.

**Solution with BFS:** [https://repl.it/@trsong/Find-the-min-number-of-operations](https://repl.it/@trsong/Find-the-min-number-of-operations)
```py
import unittest

def min_operations(start, end):
    queue = [start]
    visited = set()
    num_ops = 0

    while True:
        for _ in xrange(len(queue)):
            cur = queue.pop(0)
            if cur == end:
                return num_ops
            if cur in visited:
                continue
            else:
                visited.add(cur)
            
            for nb in [2*cur, cur-1]:
                if nb not in visited:
                    queue.append(nb)
        num_ops += 1

            
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

**Example:**
```py
min_operations(6, 20)  # Gives 3
# Since (((6 - 1) * 2) * 2) = 20 : 3 operations needed only
```

### June 4, 2020 LC 289 \[Medium\] Conway's Game of Life
---
> **Question:** Conway's Game of Life takes place on an infinite two-dimensional board of square cells. Each cell is either dead or alive, and at each tick, the following rules apply:
>
> - Any live cell with less than two live neighbours dies.
> - Any live cell with two or three live neighbours remains living.
> - Any live cell with more than three live neighbours dies.
> - Any dead cell with exactly three live neighbours becomes a live cell.
> - A cell neighbours another cell if it is horizontally, vertically, or diagonally adjacent.
>
> Implement Conway's Game of Life. It should be able to be initialized with a starting list of live cell coordinates and the number of steps it should run for. Once initialized, it should print out the board state at each step. Since it's an infinite board, print out only the relevant coordinates, i.e. from the top-leftmost live cell to bottom-rightmost live cell.
>
> You can represent a live cell with an asterisk (*) and a dead cell with a dot (.).


**Solution:** [https://repl.it/@trsong/Solve-Conways-Game-of-Life](https://repl.it/@trsong/Solve-Conways-Game-of-Life)
```py
import unittest

class ConwaysGameOfLife(object):
    def __init__(self, initial_life_coordinates):
        self._grid = {}
        self._reset_boundary()
        for coord in initial_life_coordinates:
            if coord[0] not in self._grid:
                self._grid[coord[0]] = set()
            self._grid[coord[0]].add(coord[1])
            self._update_boundary(coord)

    def _reset_boundary(self):
        self._left_boundary = None
        self._right_boundary = None
        self._top_boundary = None
        self._bottom_boundary = None

    def _update_boundary(self, coord):
        self._left_boundary = coord[1] if self._left_boundary is None else  min(self._left_boundary, coord[1])
        self._right_boundary = coord[1] if self._right_boundary is None else max(self._right_boundary, coord[1])
        self._top_boundary =  coord[0] if self._top_boundary is None else min(self._top_boundary, coord[0])
        self._bottom_boundary = coord[0] if self._bottom_boundary is None else max(self._bottom_boundary, coord[0])

    def _check_alive(self, coord):
        r, c = coord[0], coord[1]
        delta = [-1, 0, 1]
        num_neighbor = 0
        for dr in delta:
            for dc in delta:
                if dr == dc == 0:
                    continue
                if r+dr in self._grid and c+dc in self._grid[r+dr]:
                    num_neighbor += 1
        is_prev_alive = r in self._grid and c in self._grid[r]
        return is_prev_alive and 2 <= num_neighbor <= 3 or not is_prev_alive and num_neighbor == 3

    def _next_round(self):
        added_changeset = []
        removed_changeset = []
        for r in xrange(self._top_boundary-1, self._bottom_boundary + 2):
            for c in xrange(self._left_boundary-1, self._right_boundary + 2):
                is_prev_alive = r in self._grid and c in self._grid[r]
                is_now_alive = self._check_alive([r, c])
                if not is_prev_alive and is_now_alive:
                    added_changeset.append([r, c])
                elif is_prev_alive and not is_now_alive:
                    removed_changeset.append([r, c])

        for coord in removed_changeset:
            self._grid[coord[0]].remove(coord[1])
            if not len(self._grid[coord[0]]):
                del self._grid[coord[0]]

        for coord in added_changeset:
            if coord[0] not in self._grid:
                self._grid[coord[0]] = set()
            self._grid[coord[0]].add(coord[1])
        
        self._reset_boundary()
        for r in self._grid:
            for c in self._grid[r]:
                self._update_boundary([r, c])

    def proceed(self, n_round):
        for _ in xrange(n_round):
            self._next_round()

    def display_grid(self):
        if not self._grid: return []
        res = []
        for r in xrange(self._top_boundary, self._bottom_boundary + 1):
            if r in self._grid:
                line = []
                for c in xrange(self._left_boundary, self._right_boundary + 1):
                    if c in self._grid[r]:
                        line.append("*")
                    else:
                        line.append(".")
                res.append("".join(line))
            else:
                res.append("." * (self._right_boundary - self._left_boundary + 1))
        return res


class ConwaysGameOfLifeSpec(unittest.TestCase):
    def test_still_lives_scenario(self):
        game = ConwaysGameOfLife([[1, 1], [1, 2], [2, 1], [2, 2]])
        self.assertEqual(game.display_grid(), [
            "**",
            "**"
        ])
        game.proceed(1)
        self.assertEqual(game.display_grid(), [
            "**",
            "**"
        ])

        game2 = ConwaysGameOfLife([[0, 1], [0, 2], [1, 0], [1, 3], [2, 1], [2, 2]])
        self.assertEqual(game2.display_grid(), [
            ".**.",
            "*..*",
            ".**."
        ])
        game2.proceed(2)
        self.assertEqual(game2.display_grid(), [
            ".**.",
            "*..*",
            ".**."
        ])

    def test_oscillators_scenario(self):
        game = ConwaysGameOfLife([[-100, 0], [-100, 1], [-100, 2]])
        self.assertEqual(game.display_grid(), [
            "***",
        ])
        game.proceed(1)
        self.assertEqual(game.display_grid(), [
            "*",
            "*",
            "*"
        ])
        game.proceed(3)
        self.assertEqual(game.display_grid(), [
           "***",
        ])

        game2 = ConwaysGameOfLife([[0, 0], [0, 1], [1, 0], [2, 3], [3, 2], [3, 3]])
        self.assertEqual(game2.display_grid(), [
            "**..",
            "*...",
            "...*",
            "..**"
        ])
        game2.proceed(1)
        self.assertEqual(game2.display_grid(), [
            "**..",
            "**..",
            "..**",
            "..**",
        ])
        game2.proceed(3)
        self.assertEqual(game2.display_grid(), [
            "**..",
            "*...",
            "...*",
            "..**"
        ])


    def test_spaceships_scenario(self):
        game = ConwaysGameOfLife([[0, 2], [1, 0], [1, 2], [2, 1], [2, 2]])
        self.assertEqual(game.display_grid(), [
            "..*",
            "*.*",
            ".**"
        ])
        game.proceed(1)
        self.assertEqual(game.display_grid(), [
           "*..",
           ".**",
           "**."
        ])
        game.proceed(1)
        self.assertEqual(game.display_grid(), [
            ".*.",
            "..*",
            "***"
        ])
        game.proceed(1)
        self.assertEqual(game.display_grid(), [
            "*.*",
            ".**",
            ".*."
        ])
        game.proceed(1)
        self.assertEqual(game.display_grid(), [
            "..*",
            "*.*",
            ".**"
        ])
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 3, 2020 \[Medium\] In-place Array Rotation
---
> **Question:** Write a function that rotates an array by `k` elements.
>
> For example, `[1, 2, 3, 4, 5, 6]` rotated by two becomes [`3, 4, 5, 6, 1, 2]`.
>
> Try solving this without creating a copy of the array. How many swap or move operations do you need?


**Solution:** [https://repl.it/@trsong/Rotate-Array-In-place](https://repl.it/@trsong/Rotate-Array-In-place)
```py
import unittest

def rotate(nums, k):
    if not nums:
        return nums
    
    n = len(nums)
    k %= n
    reverse(nums, 0, k-1)
    reverse(nums, k, n-1)
    reverse(nums, 0, n-1)
    return nums


def reverse(nums, start, end):
    while start < end:
        nums[start], nums[end] = nums[end], nums[start]
        start += 1
        end -= 1


class RotateSpec(unittest.TestCase):
    def test_example(self):
        k, nums = 2, [1, 2, 3, 4, 5, 6]
        expected = [3, 4, 5, 6, 1, 2]
        self.assertEqual(expected, rotate(nums, k))

    def test_rotate_0_position(self):
        k, nums = 0, [0, 1, 2, 3]
        expected = [0, 1, 2, 3]
        self.assertEqual(expected, rotate(nums, k))

    def test_empty_array(self):
        self.assertEqual([], rotate([], k=10))

    def test_shift_negative_position(self):
        k, nums = -1, [0, 1, 2, 3]
        expected = [3, 0, 1, 2]
        self.assertEqual(expected, rotate(nums, k))

    def test_shift_more_than_array_size(self):
        k, nums = 8,  [1, 2, 3, 4, 5, 6]
        expected = [3, 4, 5, 6, 1, 2]
        self.assertEqual(expected, rotate(nums, k))

    def test_multiple_round_of_forward_and_backward_shift(self):
        k, nums = 5, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        expected = [5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4]
        self.assertEqual(expected, rotate(nums, k))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 2, 2020 \[Hard\] Find Next Greater Permutation
---
> **Question:** Given a number represented by a list of digits, find the next greater permutation of a number, in terms of lexicographic ordering. If there is not greater permutation possible, return the permutation with the lowest value/ordering.
>
> For example, the list `[1,2,3]` should return `[1,3,2]`. The list `[1,3,2]` should return `[2,1,3]`. The list `[3,2,1]` should return `[1,2,3]`.
>
> Can you perform the operation without allocating extra memory (disregarding the input memory)?

**My thoughts:** Imagine the list as a number, if it’s in descending order, then there will be no number greater than that and we have to return the number in ascending order, that is, the smallest number. e.g. `321` will become `123`.

Leave first part untouched. If the later part of array are first increasing then decreasing, like `1321`, then based on previous observation, we know the descending part will change from largest to smallest, we want the last increasing digit to increase as little as possible, i.e. slightly larger number on the right. e.g. `2113`

Here are all the steps:

1. Find last increase number, name it _pivot_
2. Find the slightly larger number _pivot_plus_. i.e. the smallest one among all number greater than the last increase number on the right
3. Swap the slightly larger number _pivot_plus_ with last increase number _pivot_
4. Turn the descending array on right to be ascending array

**Solution:** [https://repl.it/@trsong/Find-the-Next-Greater-Permutation](https://repl.it/@trsong/Find-the-Next-Greater-Permutation)
```py
import unittest

def find_next_permutation(nums):
    if not nums:
        return nums

    n = len(nums)
    pivot = n - 2
    while pivot >= 0 and nums[pivot] >= nums[pivot+1]:
        pivot -= 1

    if pivot >= 0:
        pivot_plus = pivot
        for i in xrange(pivot+1, n):
            if nums[i] <= nums[pivot]:
                break
            pivot_plus = i
        nums[pivot], nums[pivot_plus] = nums[pivot_plus], nums[pivot]
    
    i, j = pivot+1, n-1
    while i < j:
        nums[i], nums[j] = nums[j], nums[i]
        i += 1
        j -= 1
    
    return nums


class FindNextPermutationSpec(unittest.TestCase):
    def test_example1(self):
        nums = [1, 2, 3]
        expected = [1, 3, 2]
        self.assertEqual(expected, find_next_permutation(nums))
    
    def test_example2(self):
        nums = [1, 3, 2]
        expected = [2, 1, 3]
        self.assertEqual(expected, find_next_permutation(nums))

    def test_example3(self):
        nums = [3, 2, 1]
        expected = [1, 2, 3]
        self.assertEqual(expected, find_next_permutation(nums))

    def test_empty_array(self):
        self.assertEqual([], find_next_permutation([]))

    def test_one_elem_array(self):
        self.assertEqual([1], find_next_permutation([1]))

    def test_decrease_increase_decrease_array(self):
        nums = [3, 2, 1, 6, 5, 4]
        expected = [3, 2, 4, 1, 5, 6]
        self.assertEqual(expected, find_next_permutation(nums))       

    def test_decrease_increase_decrease_array2(self):    
        nums = [3, 2, 4, 6, 5, 4]
        expected = [3, 2, 5, 4, 4, 6]
        self.assertEqual(expected, find_next_permutation(nums))

    def test_increasing_decreasing_increasing_array(self):
        nums = [4, 5, 6, 1, 2, 3]
        expected = [4, 5, 6, 1, 3, 2]
        self.assertEqual(expected, find_next_permutation(nums))

    def test_increasing_decreasing_increasing_array2(self):
        nums = [1, 1, 2, 3, 4, 4, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 3, 2, 1]
        expected = [1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10]
        self.assertEqual(expected, find_next_permutation(nums))

    def test_multiple_decreasing_and_increasing_array(self):
        nums = [5, 3, 4, 9, 7, 6]
        expected = [5, 3, 6, 4, 7, 9]
        self.assertEqual(expected, find_next_permutation(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### June 1, 2020 \[Medium\] Find Next Biggest Integer
---
> **Question:** Given an integer `n`, find the next biggest integer with the same number of 1-bits on. For example, given the number `6 (0110 in binary)`, return `9 (1001)`.

**My thoughts:** The idea is to find the leftmost of rightmost ones, swap it with left zero and push remaining rightmost ones all the way till the end.

**Example:**
```py
   10011100
      ^      swap with left zero
=> 10101100 
       ^^    push till the end
=> 10100011 
```

**Solution:** [https://repl.it/@trsong/Find-the-Next-Biggest-Integer](https://repl.it/@trsong/Find-the-Next-Biggest-Integer)
```py
import unittest

def next_higher_number(num):
    if num == 0:
        return None

    last_one_mask = num & -num
    num_last_group_ones = 0
    while num & last_one_mask:
        num &= ~last_one_mask
        last_one_mask <<= 1
        num_last_group_ones += 1

    num |= last_one_mask
    if num_last_group_ones > 1:
        num |= (1 << num_last_group_ones - 1) - 1
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

### May 31, 2020 \[Medium\] Remove K-th Last Element from Singly Linked-list
---
> **Question:** Given a singly linked list and an integer k, remove the kth last element from the list. k is guaranteed to be smaller than the length of the list.
>
> **Note:**
> - The list is very long, so making more than one pass is prohibitively expensive.
> - Do this in constant space and in one pass.

**Solution with Fast and Slow Pointer:** [https://repl.it/@trsong/Remove-the-K-th-Last-Element-from-Singly-Linked-list](https://repl.it/@trsong/Remove-the-K-th-Last-Element-from-Singly-Linked-list)
```py

import unittest

def remove_last_kth_elem(k, lst):
    fast = slow = dummy = ListNode(-1, lst)
    for _ in xrange(k):
        fast = fast.next

    while fast.next:
        fast = fast.next
        slow = slow.next

    if slow.next:
        slow.next = slow.next.next
    return dummy.next
    

###################
# Testing Utilities
###################
class ListNode(object):
    def __init__(self, x, next=None):
        self.val = x
        self.next = next

    def __eq__(self, other):
        return other and self.val == other.val and self.next == other.next

    def __repr__(self):
        return "%d -> %s" % (self.val, self.next)

    @staticmethod
    def seq(*vals):
        p = dummy = ListNode(-1)
        for elem in vals:
            p.next = ListNode(elem)
            p = p.next
        return dummy.next


class RemoveLastKthElementSpec(unittest.TestCase):
    def test_empty_list(self):
        self.assertIsNone(remove_last_kth_elem(0, None))

    def test_remove_the_only_element(self):
        k, lst = 1, ListNode.seq(42)
        self.assertIsNone(remove_last_kth_elem(k, lst))
    
    def test_remove_the_last_element(self):
        k, lst = 1, ListNode.seq(1, 2, 3)
        expected = ListNode.seq(1, 2)
        self.assertEqual(expected, remove_last_kth_elem(k, lst))

    def test_remove_the_first_element(self):
        k, lst = 4, ListNode.seq(1, 2, 3, 4)
        expected = ListNode.seq(2, 3, 4)
        self.assertEqual(expected, remove_last_kth_elem(k, lst))

    def test_remove_element_in_the_middle(self):
        k, lst = 3, ListNode.seq(5, 4, 3, 2, 1)
        expected = ListNode.seq(5, 4, 2, 1)
        self.assertEqual(expected, remove_last_kth_elem(k, lst))

    def test_remove_the_second_last_element(self):
        k, lst = 2, ListNode.seq(4, 3, 2, 1)
        expected = ListNode.seq(4, 3, 1)
        self.assertEqual(expected, remove_last_kth_elem(k, lst))

    
if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 30, 2020 \[Easy\] Rotate Linked List
---
> **Question:** Given a linked list and a number k, rotate the linked list by k places.

**Example:**
```py
Input: 
Node(1, Node(2, Node(3, Node(4))))
2

Output:
Node(3, Node(4, Node(1, Node(2))))
```

**Solution:** [https://repl.it/@trsong/Rotate-Linked-List](https://repl.it/@trsong/Rotate-Linked-List)
```py
import unittest

def rotate_list(head, k):
    if not head:
        return head

    p = dummy = Node(-1, head)
    lst_len = 0
    while p.next:
        p = p.next
        lst_len += 1
    k %= lst_len
    if k == 0:
        return head

    p.next = dummy.next
    p = dummy
    for _ in xrange(k):
        p = p.next

    res = p.next
    p.next = None
    return res


class Node(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    def __eq__(self, other):
        return other and self.val == other.val and self.next == other.next

    def __repr__(self):
        return "%d -> %s" % (self.val, self.next)

    @staticmethod
    def List(*vals):
        p = dummy = Node(-1)
        for val in vals:
            p.next = Node(val)
            p = p.next
        return dummy.next


class RotateListSpec(unittest.TestCase):
    def test_example(self):
        k, lst = 2, Node.List(1, 2, 3, 4)
        expected = Node.List(3, 4, 1, 2)
        self.assertEqual(expected, rotate_list(lst, k))

    def test_empty_list(self):
        self.assertIsNone(rotate_list(None, 42))

    def test_single_elem_list(self):
        k, lst = 42, Node.List(1)
        expected = Node.List(1)
        self.assertEqual(expected, rotate_list(lst, k))

    def test_k_greater_than_length_of_list(self):
        k, lst = 5, Node.List(1, 2)
        expected = Node.List(2, 1)
        self.assertEqual(expected, rotate_list(lst, k))

    def test_k_equal_length_of_list(self):
        k, lst = 3, Node.List(1, 2, 3)
        expected = Node.List(1, 2, 3)
        self.assertEqual(expected, rotate_list(lst, k))

    def test_k_is_negative(self):
        k, lst = -1, Node.List(1, 2, 3)
        expected = Node.List(3, 1, 2)
        self.assertEqual(expected, rotate_list(lst, k))

    def test_k_less_than_length_of_list(self):
        k, lst = 5, Node.List(0, 1, 2, 3, 4, 5)
        expected = Node.List(5, 0, 1, 2, 3, 4)
        self.assertEqual(expected, rotate_list(lst, k))

    def test_array_contains_duplicates(self):
        k, lst = 1, Node.List(1, 1, 2, 2, 3, 3, 3, 4)
        expected = Node.List(1, 2, 2, 3, 3, 3, 4, 1)
        self.assertEqual(expected, rotate_list(lst, k))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### May 29, 2020 \[Easy\] Longest Consecutive Subsequence
---
> **Question:** Given an array of integers, find the length of the longest sub-sequence such that elements in the subsequence are consecutive integers, the consecutive numbers can be in any order.

**Example1:**
```py
Input: [1, 9, 3, 10, 4, 20, 2]
Output: 4
The subsequence 1, 3, 4, 2 is the longest subsequence
of consecutive elements
```

**Example2:**
```py
Input: [36, 41, 56, 35, 44, 33, 34, 92, 43, 32, 42]
Output: 5
The subsequence 36, 35, 33, 34, 32 is the longest subsequence
of consecutive elements. 
```

**Solution:** [https://repl.it/@trsong/Longest-Consecutive-Subsequence](https://repl.it/@trsong/Longest-Consecutive-Subsequence)
```py
import unittest

def find_longest_consecutive_seq(nums):
    num_set = set(nums)

    res = 0
    for pivot in nums:
        if pivot not in num_set:
            continue
        
        seq_len = 1
        p = pivot - 1
        while p in num_set:
            seq_len += 1
            num_set.remove(p)
            p -= 1

        p = pivot + 1
        while p in num_set:
            seq_len += 1
            num_set.remove(p)
            p += 1

        res = max(res, seq_len)
    return res   


class FindLongestConsecutiveSeqSpec(unittest.TestCase):
    def test_example(self):
        nums = [1, 9, 3, 10, 4, 20, 2]
        expected = 4  # [1, 3, 4, 2]
        self.assertEqual(expected, find_longest_consecutive_seq(nums))

    def test_example2(self):
        nums = [36, 41, 56, 35, 44, 33, 34, 92, 43, 32, 42]
        expected = 5  # [36, 35, 33, 34, 32]
        self.assertEqual(expected, find_longest_consecutive_seq(nums))

    def test_example3(self):
        nums = [9, 6, 1, 3, 8, 10, 12, 11]
        expected = 5  # [9, 8, 10, 12, 11]
        self.assertEqual(expected, find_longest_consecutive_seq(nums))

    def test_example4(self):
        nums = [2, 10, 3, 12, 5, 4, 11, 8, 7, 6, 15]
        expected = 7  # [2, 3, 5, 4, 8, 7, 6]
        self.assertEqual(expected, find_longest_consecutive_seq(nums))

    def test_empty_array(self):
        self.assertEqual(0, find_longest_consecutive_seq([]))

    def test_no_consecutive_sequence(self):
        nums = [1, 3, 5, 7]
        expected = 1
        self.assertEqual(expected, find_longest_consecutive_seq(nums))

    def test_more_than_one_solution(self):
        nums = [0, 3, 4, 5, 9, 10, 13, 14, 15, 19, 20, 1]
        expected = 3  # [3, 4, 5]
        self.assertEqual(expected, find_longest_consecutive_seq(nums))

    def test_longer_array(self):
        nums = [10, 21, 45, 22, 7, 2, 67, 19, 13, 45, 12, 11, 18, 16, 17, 100, 201, 20, 101]
        expected = 7  # [21, 22, 19, 18, 16, 17, 20]
        self.assertEqual(expected, find_longest_consecutive_seq(nums))

    def test_entire_array_is_continous(self):
        nums = [0, 1, 2, 3, 4, 5]
        expected = 6
        self.assertEqual(expected, find_longest_consecutive_seq(nums))

    def test_array_with_duplicated_numbers(self):
        nums = [0, 0, 3, 3, 2, 2, 1, 4, 7, 8, 10]
        expected = 5  # [0, 3, 2, 1, 4] 
        self.assertEqual(expected, find_longest_consecutive_seq(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 28, 2020 \[Easy\]  Rearrange Array in Alternating Positive & Negative Order
---
> **Question:** Given an array of positive and negative numbers, arrange them in an alternate fashion such that every positive number is followed by negative and vice-versa maintaining the order of appearance.
> 
> Number of positive and negative numbers need not be equal. If there are more positive numbers they appear at the end of the array. If there are more negative numbers, they too appear in the end of the array.

**Example1:**
```py
Input:  arr[] = {1, 2, 3, -4, -1, 4}
Output: arr[] = {-4, 1, -1, 2, 3, 4}
```

**Example2:**
```py
Input:  arr[] = {-5, -2, 5, 2, 4, 7, 1, 8, 0, -8}
output: arr[] = {-5, 5, -2, 2, -8, 4, 7, 1, 8, 0} 
```

**Solution:** [https://repl.it/@trsong/Rearrange-Array-in-Alternating-Positive-and-Negative-Order](https://repl.it/@trsong/Rearrange-Array-in-Alternating-Positive-and-Negative-Order)
```py
import unittest

def rearrange_array(nums):
    positives = []
    negatives = []
    for num in nums:
        if num >= 0:
            positives.append(num)
        else:
            negatives.append(num)

    i = j = k = 0
    while k < len(nums):
        if i < len(negatives):
            nums[k] = negatives[i]
            i += 1
            k += 1
        
        if j < len(positives):
            nums[k] = positives[j]
            j += 1
            k += 1   
    
    return nums

    
class RearrangeArraySpec(unittest.TestCase):
    def test_example(self):
        nums = [1, 2, 3, -4, -1, 4]
        expected = [-4, 1, -1, 2, 3, 4]
        self.assertEqual(expected, rearrange_array(nums))

    def test_example2(self):
        nums = [-5, -2, 5, 2, 4, 7, 1, 8, 0, -8]
        expected = [-5, 5, -2, 2, -8, 4, 7, 1, 8, 0]
        self.assertEqual(expected, rearrange_array(nums))

    def test_more_negatives_than_positives(self):
        nums = [-1, -2, -3, -4, 1, 2, -5, -6]
        expected =  [-1, 1, -2, 2, -3, -4, -5, -6]
        self.assertEqual(expected, rearrange_array(nums))

    def test_more_positives_than_negatives(self):
        nums = [-1, 1, 2, 3, -2]
        expected =  [-1, 1, -2, 2, 3]
        self.assertEqual(expected, rearrange_array(nums))

    def test_empty_array(self):
        nums = []
        expected = []
        self.assertEqual(expected, rearrange_array(nums))

    def test_no_negatives(self):
        nums = [1, 1, 2, 3]
        expected = [1, 1, 2, 3]
        self.assertEqual(expected, rearrange_array(nums))

    def test_no_positive_array(self):
        nums = [-1]
        expected = [-1]
        self.assertEqual(expected, rearrange_array(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 27, 2020 LC 665 \[Medium\] Off-by-One Non-Decreasing Array
---
> **Question:** Given an array of integers, write a function to determine whether the array could become non-decreasing by modifying at most 1 element.
>
> For example, given the array `[10, 5, 7]`, you should return true, since we can modify the `10` into a `1` to make the array non-decreasing.
>
> Given the array `[10, 5, 1]`, you should return false, since we can't modify any one element to get a non-decreasing array.

**My thoughts:** try to identify the down postition. The problematic position prevents array from non-decreasing is either the down position or its previous position. Just remove either position and test array again, if it works then it's off-by-one array otherwise it's not since more positions need to be removed.

**Solution:** [https://repl.it/@trsong/Determine-Off-by-One-Non-Decreasing-Array](https://repl.it/@trsong/Determine-Off-by-One-Non-Decreasing-Array)
```py
import unittest

def is_off_by_one_array(nums):
    if len(nums) <= 2:
        return True

    n = len(nums)
    down_pos = None
    for i in xrange(1, n):
        if nums[i-1] > nums[i]:
            if down_pos is not None:
                return False
            down_pos = i

    # Either prev_down_pos or down_pos has some issue, we try to fix that
    if down_pos is None or down_pos == 1 or down_pos == n - 1:
        return True
    else:
        prev_down_pos = down_pos - 1
        without_prev = nums[down_pos-1] <= nums[down_pos+1]
        without_cur = nums[prev_down_pos-1] <= nums[prev_down_pos+1]
        return without_prev or without_cur


class IsOffByOneArraySpec(unittest.TestCase):
    def test_example(self):
        self.assertTrue(is_off_by_one_array([10, 5, 7]))

    def test_example2(self):
        self.assertFalse(is_off_by_one_array([10, 5, 1]))

    def test_empty_array(self):
        self.assertTrue(is_off_by_one_array([]))

    def test_one_element_array(self):
        self.assertTrue(is_off_by_one_array([1]))

    def test_two_elements_array(self):
        self.assertTrue(is_off_by_one_array([1, 1]))
        self.assertTrue(is_off_by_one_array([1, 0]))
        self.assertTrue(is_off_by_one_array([0, 1]))

    def test_decreasing_array(self):
        self.assertFalse(is_off_by_one_array([8, 2, 0]))

    def test_non_decreasing_array(self):
        self.assertTrue(is_off_by_one_array([0, 0, 1, 2, 2]))
        self.assertTrue(is_off_by_one_array([0, 1, 2]))
        self.assertTrue(is_off_by_one_array([0, 0, 0, 0]))

    def test_off_by_one_array(self):
        self.assertTrue(is_off_by_one_array([2, 10, 0]))
        self.assertTrue(is_off_by_one_array([5, 2, 10]))
        self.assertTrue(is_off_by_one_array([0, 1, 0, 0]))
        self.assertTrue(is_off_by_one_array([-1, 4, 2, 3]))
        self.assertTrue(is_off_by_one_array([0, 1, 1, 0]))

    def test_off_by_two_array(self):
        self.assertFalse(is_off_by_one_array([5, 2, 10, 3, 4]))
        self.assertTrue(is_off_by_one_array([0, 1, 0, 0, 0, 1]))
        self.assertFalse(is_off_by_one_array([1, 1, 0, 0]))
        self.assertFalse(is_off_by_one_array([0, 1, 1, 0, 0, 1]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 26, 2020 \[Medium\] Longest Alternating Subsequence Problem
---
> **Question:** Finding the length of a subsequence of a given sequence in which the elements are in alternating order, and in which the sequence is as long as possible. 
>
> For example, consider array `A[] = [8, 9, 6, 4, 5, 7, 3, 2, 4]` The length of longest subsequence is `6` and the subsequence is `[8, 9, 6, 7, 3, 4]` as `(8 < 9 > 6 < 7 > 3 < 4)`.

**My thoughts:** Consider the value of array goes up and down. The length of longest alternating subsequence cannot exceed `1 + number of max and min points`. 

**Solution:** [https://repl.it/@trsong/Longest-Alternating-Subsequence](https://repl.it/@trsong/Longest-Alternating-Subsequence)
```py
import unittest

def find_len_of_longest_alt_seq(nums):
    if len(nums) <= 1:
        return len(nums)

    prev_sign = None
    res = 1
    for i in xrange(1, len(nums)):
        current_sign = cmp(nums[i - 1], nums[i])
        if current_sign != 0 and prev_sign != current_sign:
            res += 1
            prev_sign = current_sign

    return res


class FindLenOfLongestAltSeqSpec(unittest.TestCase):
    def test_example(self):
        nums = [8, 9, 6, 4, 5, 7, 3, 2, 4]
        expected = 6  # [8, 9, 6, 7, 3, 4]
        self.assertEqual(expected, find_len_of_longest_alt_seq(nums))

    def test_empty_array(self):
        nums = []
        expected = 0
        self.assertEqual(expected, find_len_of_longest_alt_seq(nums))

    def test_entire_array_is_alternating(self):
        nums = [1, 7, 4, 9, 2, 5]
        expected = 6  # [1, 7, 4, 9, 2, 5]
        self.assertEqual(expected, find_len_of_longest_alt_seq(nums))

    def test_multiple_results(self):
        nums = [1, 17, 5, 10, 13, 15, 10, 5, 16, 8]
        expected = 7  # One solution: [1, 17, 10, 13, 10, 16, 8]
        self.assertEqual(expected, find_len_of_longest_alt_seq(nums))

    def test_increasing_array(self):
        nums = [1, 3, 8, 9]
        expected = 2  # One solution: [1, 3]
        self.assertEqual(expected, find_len_of_longest_alt_seq(nums))

    def test_local_solution_is_not_optimal(self):
        nums = [4, 8, 2, 5, 8, 6]
        expected = 5  # [4, 8, 2, 8, 6]
        self.assertEqual(expected, find_len_of_longest_alt_seq(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 25, 2020 \[Medium\] Craft Sentence
---
> **Question:** Write an algorithm to justify text. Given a sequence of words and an integer line length k, return a list of strings which represents each line, fully justified.
> 
> More specifically, you should have as many words as possible in each line. There should be at least one space between each word. Pad extra spaces when necessary so that each line has exactly length k. Spaces should be distributed as equally as possible, with the extra spaces, if any, distributed starting from the left.
> 
> If you can only fit one word on a line, then you should pad the right-hand side with spaces.
> 
> Each word is guaranteed not to be longer than k.
>
> For example, given the list of words `["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]` and `k = 16`, you should return the following:

```py
["the  quick brown",
 "fox  jumps  over",
 "the   lazy   dog"]
```


**Solution:** [https://repl.it/@trsong/Craft-Sentence-Problem#main.py](https://repl.it/@trsong/Craft-Sentence-Problem#main.py)
```py
import unittest

def craft_sentence(words, k):
    buff = []
    res = []
    sentence_size = 0
    for word in words:
        if sentence_size + len(word) > k:
            sentence = craft_one_sentence(buff, k)
            res.append(sentence)
            buff = []
            sentence_size = 0
        
        buff.append(word)
        sentence_size += len(word) + 1
    
    if buff:
        res.append(craft_one_sentence(buff, k))

    return res


def craft_one_sentence(words, k):
    n = len(words)
    num_char = sum(map(len, words))
    white_spaces = k - num_char
    padding = white_spaces // (n - 1) if n > 1 else white_spaces
    extra_padding = white_spaces % (n - 1) if n > 1 else 0

    res = []
    for word in words:
        res.append(word)
        if extra_padding > 0:
            res.append(" " * (padding + 1))
            white_spaces -= (padding + 1)
            extra_padding -= 1
        elif white_spaces > 0:
            res.append(" " * padding)
            white_spaces -= padding
    return ''.join(res)


class CraftSentenceSpec(unittest.TestCase):
    def test_fit_only_one_word(self):
        k, words = 7, ["test", "same", "length", "string"]
        expected = ["test   ", "same   ", "length ", "string "]
        self.assertEqual(expected, craft_sentence(words, k))
    
    def test_fit_only_one_word2(self):
        k, words = 6, ["test", "same", "length", "string"]
        expected = ["test  ", "same  ", "length", "string"]
        self.assertEqual(expected, craft_sentence(words, k))

    def test_no_padding(self):
        k, words = 2, ["to", "be"]
        expected = ["to", "be"]
        self.assertEqual(expected, craft_sentence(words, k))

    def test_fit_two_words(self):
        k, words = 6, ["To", "be", "or", "not", "to", "be"]
        expected = ["To  be", "or not", "to  be"]
        self.assertEqual(expected, craft_sentence(words, k))

    def test_fit_two_words2(self):
        k, words = 11, ["Greed", "is", "not", "good"]
        expected = ["Greed    is", "not    good"]
        self.assertEqual(expected, craft_sentence(words, k))

    def test_fit_more_words(self):
        k, words = 16, ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        expected = ["the  quick brown", "fox  jumps  over", "the   lazy   dog"]
        self.assertEqual(expected, craft_sentence(words, k))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 24, 2020 \[Hard\] Longest Palindromic Substring
---
> **Question:** Given a string, find the longest palindromic contiguous substring. If there are more than one with the maximum length, return any one.
>
> For example, the longest palindromic substring of `"aabcdcb"` is `"bcdcb"`. The longest palindromic substring of `"bananas"` is `"anana"`.

**Solution with DP:** [https://repl.it/@trsong/Find-Longest-Palindromic-Substring](https://repl.it/@trsong/Find-Longest-Palindromic-Substring)
```py
import unittest

def find_longest_palindromic_substring(s):
    if not s:
        return ""

    n = len(s)
    # Let dp[i][j] represents whether substring[i:j+1] is palindromic or not
    #     dp[i][j] = True if dp[i+1][j-1] and s[i]==s[j] 
    dp = [[False for _ in xrange(n)] for _ in xrange(n)]
    
    max_window_size = 1
    max_window_start = 0
    for window_size in xrange(1, n+1):
        for start in xrange(n):
            end = start + window_size - 1
            if end >= n: 
                break
            
            if s[start] == s[end] and (start+1 >= end-1 or dp[start+1][end-1]):
                dp[start][end] = True
                if window_size > max_window_size:
                    max_window_size = window_size
                    max_window_start = start
    
    return s[max_window_start:max_window_start+max_window_size]
            

class FindLongestPalindromicSubstringSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual("bcdcb", find_longest_palindromic_substring("aabcdcb"))

    def test_example2(self):
        self.assertEqual("anana", find_longest_palindromic_substring("bananas"))

    def test_one_letter_palindrome(self):
        word = "abcdef"
        result = find_longest_palindromic_substring(word)
        self.assertTrue(result in word and len(result) == 1)
    
    def test_multiple_length_2_palindrome(self):
        result = find_longest_palindromic_substring("zaaqrebbqreccqreddz")
        self.assertIn(result, ["aa", "bb", "cc", "dd"])

    def test_multiple_length_3_palindrome(self):
        result = find_longest_palindromic_substring("xxaza1xttv1xpqp1x")
        self.assertIn(result, ["aza", "ttt", "pqp"])


if __name__ == '__main__':
    unittest.main(exit=False)
```


### May 23, 2020 LC 394 \[Medium\] Decode String (Invariant)
---
> **Question:** Given an encoded string in form of `"ab[cd]{2}def"`. You have to return decoded string `"abcdcddef"`
>
> Notice that if there is a number inside curly braces, then it means preceding string in square brackets has to be repeated the same number of times. It becomes tricky where you have nested braces.

**Example 1:**
```py
Input: "ab[cd]{2}"
Output: "abcdcd"
```

**Example 2:**
```py
Input: "def[ab[cd]{2}]{3}ghi"
Output: "defabcdcdabcdcdabcdcdghi"
```

**Solution:** [https://repl.it/@trsong/Decode-String-Invariant](https://repl.it/@trsong/Decode-String-Invariant)
```py
import unittest

def decode_string(encoded_string):
    buff = []
    stack = []
    count = 0
    is_in_counting_section = False
    buff_str = ''
    for ch in encoded_string:
        if ch == '{':
            is_in_counting_section = True
            count = 0
        elif ch == '}':
            is_in_counting_section = False
            prev_str = stack.pop()
            combined_str = prev_str + buff_str * count
            buff = [combined_str]
        elif is_in_counting_section:
            count = 10 * count + int(ch)
        elif ch == '[':
            buff_str = ''.join(buff)
            buff = []
            stack.append(buff_str)
        elif ch == ']':
            buff_str = ''.join(buff)
        else:
            buff.append(ch)
    
    return ''.join(buff)
        

class DecodeStringSpec(unittest.TestCase):
    def test_example(self):
        input = "ab[cd]{2}"
        expected = "abcdcd"
        self.assertEqual(expected, decode_string(input))
    
    def test_example2(self):
        input = "def[ab[cd]{2}]{3}ghi"
        expected = "defabcdcdabcdcdabcdcdghi"
        self.assertEqual(expected, decode_string(input))

    def test_empty_string(self):
        self.assertEqual("", decode_string(""))
    
    def test_empty_string2(self):
        self.assertEqual("", decode_string("[]{42}"))

    def test_two_pattern_back_to_back(self):
        input = "[a]{3}[bc]{2}"
        expected = "aaabcbc"
        self.assertEqual(expected, decode_string(input)) 
    
    def test_nested_pattern(self):
        input = "[a[c]{2}]{3}"
        expected = "accaccacc"
        self.assertEqual(expected, decode_string(input)) 
    
    def test_nested_pattern2(self):
        input = "[a[b]{2}c]{2}"
        expected = 2 * ('a' + 2 * 'b' + 'c')
        self.assertEqual(expected, decode_string(input)) 

    def test_back_to_back_pattern_with_extra_appending(self):
        input = "[abc]{2}[cd]{3}ef"
        expected = "abcabccdcdcdef"
        self.assertEqual(expected, decode_string(input)) 
    
    def test_simple_pattern(self):
        input = "[abc]{3}"
        expected = 3 * "abc"
        self.assertEqual(expected, decode_string(input))
    def test_duplicate_more_than_10_times(self):
        input = "[ab]{233}"
        expected =  233 * "ab"
        self.assertEqual(expected, decode_string(input))

    def test_2Level_nested_encoded_string(self):
        input = "[[a]{3}[bc]{3}[d]{2}]{2}[[e]{2}]{4}"
        expected = 2 * (3*"a" + 3*"bc" + 2*"d") + 4*2*"e"
        self.assertEqual(expected, decode_string(input))

    def test_3Level_nested_encoded_string(self):
        input = "[a[b[c]{3}d[ef]{4}g]{2}h]{2}"
        expected = 2*('a' + 2*('b' + 3*'c' + 'd' + 4 * 'ef' + 'g' ) + 'h')
        self.assertEqual(expected, decode_string(input))


if __name__ == "__main__":
    unittest.main(exit=False)
```

### May 22, 2020 LC 1136 \[Hard\] Parallel Courses
---
> **Question:** There are N courses, labelled from 1 to N.
>
> We are given `relations[i] = [X, Y]`, representing a prerequisite relationship between course X and course Y: course X has to be studied before course Y.
>
> In one semester you can study any number of courses as long as you have studied all the prerequisites for the course you are studying.
>
> Return the minimum number of semesters needed to study all courses.  If there is no way to study all the courses, return -1.


**Example 1:**
```py
Input: N = 3, relations = [[1,3],[2,3]]
Output: 2
Explanation: 
In the first semester, courses 1 and 2 are studied. In the second semester, course 3 is studied.
```

**Example 2:**
```py
Input: N = 3, relations = [[1,2],[2,3],[3,1]]
Output: -1
Explanation: 
No course can be studied because they depend on each other.
```

**My thoughts:** Treat each course as vertex and prerequisite relation as edges. We will have a directed acyclic graph (DAG) with edge weight = 1. The problem then convert to find the longest path in a DAG. To find the answer, we need to find topological order of courses and then find longest path based on topological order.  

**Solution with Topological Sort:** [https://repl.it/@trsong/Parallel-Courses](https://repl.it/@trsong/Parallel-Courses)
```py
import unittest


class VertexState(object):
    UNVISITED = 0
    VISITING = 1
    VISITED = 2


def min_semesters(total_courses, prerequisites):
    if total_courses <= 0:
        return 0

    neighbors = [None] * (total_courses + 1)
    for u, v in prerequisites:
        if neighbors[u] is None:
            neighbors[u] = []
        neighbors[u].append(v)

    top_order = find_topological_order(neighbors)
    if top_order is None:
        return -1
    max_path_length = find_max_path_length(neighbors, top_order)
    return max_path_length + 1


def find_topological_order(neighbors):
    n = len(neighbors)
    node_states = [VertexState.UNVISITED] * n
    reverse_top_order = []

    for node in xrange(n):
        if node_states[node] is not VertexState.UNVISITED:
            continue

        stack = [node]
        while stack:
            cur = stack[-1]
            if node_states[cur] is VertexState.VISITED:
                stack.pop()
            elif node_states[cur] is VertexState.VISITING:
                reverse_top_order.append(cur)
                node_states[cur] = VertexState.VISITED
            else:
                # node_states is UNVISITED
                node_states[cur] = VertexState.VISITING
                if neighbors[cur] is None:
                    continue
                for nb in neighbors[cur]:
                    if node_states[nb] is VertexState.VISITING:
                        return None
                    elif node_states[nb] is VertexState.UNVISITED:
                        stack.append(nb)

    return reverse_top_order[::-1]


def find_max_path_length(neighbors, node_traversal):
    n = len(neighbors)
    distance = [0] * n
    for node in node_traversal:
        if neighbors[node] is None:
            continue
        for nb in neighbors[node]:
            distance[nb] = max(distance[nb], 1 + distance[node])

    return max(distance)


class min_semesterss(unittest.TestCase):
    def test_example(self):
        total_courses = 3
        prerequisites = [[1, 3], [2, 3]]
        expected = 2  # gradudation path: 1/2, 3
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_no_course_to_take(self):
        total_courses = 0
        prerequisites = []
        expected = 0
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_all_courses_are_independent(self):
        total_courses = 3
        prerequisites = []
        expected = 1  # gradudation path: 1/2/3
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_grap_with_cycle(self):
        total_courses = 3
        prerequisites = [[1, 2], [3, 1], [2, 3]]
        expected = -1
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_grap_with_cycle2(self):
        total_courses = 2
        prerequisites = [[1, 2], [2, 1]]
        expected = -1
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_disconnected_graph(self):
        total_courses = 5
        prerequisites = [[1, 2], [3, 4], [4, 5]]
        expected = 3  # gradudation path: 1/3, 2/4, 5
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_graph_with_two_paths(self):
        total_courses = 5
        prerequisites = [[1, 2], [2, 5], [1, 3], [3, 4], [4, 5]]
        expected = 4  # gradudation path: 1, 2/3, 4, 5
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_graph_with_two_paths2(self):
        total_courses = 5
        prerequisites = [[1, 3], [3, 4], [4, 5], [1, 2], [2, 5]]
        expected = 4  # gradudation path: 1, 2/3, 4, 5
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))

    def test_connected_graph_with_paths_of_different_lenghths(self):
        total_courses = 7
        prerequisites = [[1, 3], [1, 4], [2, 3], [2, 4], [3, 4], [3, 6],
                         [4, 5], [4, 6], [4, 7], [3, 6], [6, 7]]
        expected = 5  # path: 1/2, 3, 4, 5/6, 7
        self.assertEqual(expected, min_semesters(total_courses, prerequisites))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 21, 2020 LT 189 \[Medium\] Find Missing Positive
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

**Solution:** [https://repl.it/@trsong/Find-Missing-Positive](https://repl.it/@trsong/Find-Missing-Positive)
```py
import unittest

def val_to_index(num):
    # the index num map to 
    return num - 1


def find_missing_positive(nums):
    n = len(nums)
    for i in xrange(n):
        while 1 <= nums[i] <= n and val_to_index(nums[i]) != i:
            swap_index = val_to_index(nums[i])
            if nums[i] == nums[swap_index]:
                # eliminate loop
                break
            nums[i], nums[swap_index] = nums[swap_index], nums[i]

    for i in xrange(n):
        if val_to_index(nums[i]) != i:
            return i+1
    
    return n+1


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
    unittest.main(exit=False)
```


### May 20, 2020  \[Medium\] Split a Binary Search Tree
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

**Solution:** [https://repl.it/@trsong/Split-BST](https://repl.it/@trsong/Split-BST)
```py
import unittest

def split_bst(root, s):
    if not root:
        return None, None

    if s < root.val:
        left_tree, right_tree = split_bst(root.left, s)
        root.left = right_tree
        return left_tree, root
    else:
        left_tree, right_tree = split_bst(root.right, s)
        root.right = left_tree
        return root, right_tree


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
    unittest.main(exit=False)
```

### May 19, 2020 \[Easy\] Max and Min with Limited Comparisons
---
> **Question:** Given a list of numbers of size `n`, where `n` is greater than `3`, find the maximum and minimum of the list using less than `2 * (n - 1)` comparisons.

**My thoughts:** The idea is to use Tournament Method. Think about each number as a team in the tournament. One team, zero matches. Two team, one match. N team, let's break them into half and use two matches to get best of best and worest of worest:

```
T(n) = 2 * T(n/2) + 2
T(1) = 0
T(2) = 1

=>
T(n) = 3n/2 - 2 
```


**Solution with Divide And Conquer:** [https://repl.it/@trsong/Find-Max-and-Min-with-Limited-Comparisons](https://repl.it/@trsong/Find-Max-and-Min-with-Limited-Comparisons)
```py
import unittest
from functools import reduce

class MinMaxPair(object):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val


def find_min_max(nums):
    return find_min_max_recur(nums, 0, len(nums) - 1)


def find_min_max_recur(nums, lo, hi):
    if lo > hi:
        return None
    elif lo == hi:
        return MinMaxPair(nums[lo], nums[lo])
    elif lo == hi - 1:
        return MinMaxPair(nums[lo], nums[hi]) if nums[lo] < nums[hi] else MinMaxPair(nums[hi], nums[lo])
    else:
        mid = lo + (hi - lo) // 2
        left_res = find_min_max_recur(nums, lo, mid)
        right_res = find_min_max_recur(nums, mid+1, hi)
        if left_res and right_res:
            return MinMaxPair(min(left_res.min_val, right_res.min_val), max(left_res.max_val, right_res.max_val))
        else:
            return left_res or right_res 
    

#######################################
# Testing Utilities
#######################################
class Number(int):
    def __new__(self, value):
        self.history = []
        return int.__new__(self, value)

    def __cmp__(self, other):
        self.history.append((self, other))
        return int.__cmp__(self, other)

    def count_comparison(self):
        return len(self.history)

    def get_history(self):
        return '\n'.join(self.history)


class GetMinMaxSpec(unittest.TestCase):
    def assert_find_min_max(self, nums):
        min_val = min(nums)
        max_val = max(nums)
        n = len(nums)
        mapped_nums = list(map(Number, nums))
        res = find_min_max(mapped_nums)
        self.assertEqual(min_val, res.min_val)
        self.assertEqual(max_val, res.max_val)
        total_num_comparisons = sum(map(lambda num: num.count_comparison(), mapped_nums))
        history = reduce(lambda accu, num: accu + '\n' + num.get_history(), mapped_nums, '')
        if total_num_comparisons > 0:
            msg = "Expect less than %d comparisons but gives %d\n%s" % (2 * (n - 1), total_num_comparisons, history) 
            self.assertLess(total_num_comparisons, 2 * (n - 1), msg)

    def test_empty_list(self):
        self.assertIsNone(find_min_max([]))

    def test_list_with_one_element(self):
        self.assert_find_min_max([-1])

    def test_list_with_two_elements(self):
        self.assert_find_min_max([1, 2])

    def test_increasing_list(self):
        self.assert_find_min_max([1, 2, 3, 4])

    def test_list_with_duplicated_element(self):
        self.assert_find_min_max([-1, 1, -1, 1])

    def test_long_list(self):
        self.assert_find_min_max([1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1])


if __name__ == '__main__':
    unittest.main(exit=False)
```


### May 18, 2020 LC 435 \[Medium\] Non-overlapping Intervals
---
> **Question:** Given a collection of intervals, find the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.
>
> Intervals can "touch", such as `[0, 1]` and `[1, 2]`, but they won't be considered overlapping.
>
> For example, given the intervals `(7, 9), (2, 4), (5, 8)`, return `1` as the last interval can be removed and the first two won't overlap.
>
> The intervals are not necessarily sorted in any order.

**My thoughts:** Think about the problem backwards: to remove the least number of intervals to make non-overlapping is equivalent to pick most number of non-overlapping intervals and remove the rest. Therefore we just need to pick most number of non-overlapping intervals that can be done with greedy algorithm by sorting on end time and pick as many intervals as possible.

**Greedy Solution:** [https://repl.it/@trsong/Non-overlapping-Intervals](https://repl.it/@trsong/Non-overlapping-Intervals)
```py
import unittest

def remove_overlapping_intervals(intervals):
    inteval_by_end_time = sorted(intervals, key=lambda start_end: start_end[-1])
    prev_end_time = None
    chosen = 0

    for start, end in inteval_by_end_time:
        if start >= prev_end_time:
            chosen += 1
            prev_end_time = end
    
    return len(intervals) - chosen


class RemoveOveralppingIntervalSpec(unittest.TestCase):
    def test_example(self):
        intervals = [(7, 9), (2, 4), (5, 8)]
        expected = 1  # remove (7, 9)
        self.assertEqual(expected, remove_overlapping_intervals(intervals))

    def test_no_intervals(self):
        self.assertEqual(0, remove_overlapping_intervals([]))

    def test_one_interval(self):
        intervals = [(0, 42)]
        expected = 0
        self.assertEqual(expected, remove_overlapping_intervals(intervals))

    def test_return_least_number_of_interval_to_remove(self):
        intervals = [(1, 2), (2, 3), (3, 4), (1, 3)]
        expected = 1  # remove (1, 3)
        self.assertEqual(expected, remove_overlapping_intervals(intervals))

    def test_duplicated_intervals(self):
        intervals = [(1, 2), (1, 2), (1, 2)]
        expected = 2  # remove (1, 2), (1, 2)
        self.assertEqual(expected, remove_overlapping_intervals(intervals))

    def test_non_overlapping_intervals(self):
        intervals = [(1, 2), (2, 3)]
        expected = 0
        self.assertEqual(expected, remove_overlapping_intervals(intervals))

    def test_share_end_points(self):
        intervals = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        expected = 3  # remove (1, 3), (1, 4), (2, 4)
        self.assertEqual(expected, remove_overlapping_intervals(intervals))

    def test_overlapping_intervals(self):
        intervals = [(1, 4), (2, 3), (4, 6), (8, 9)]
        expected = 1  # remove (2, 3)
        self.assertEqual(expected, remove_overlapping_intervals(intervals))

    def test_should_remove_first_interval(self):
        intervals = [(1, 9), (2, 3), (5, 7)]
        expected = 1 # remove (1, 9)
        self.assertEqual(expected, remove_overlapping_intervals(intervals))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### May 17, 2020 LC 480 \[Hard\] Sliding Window Median
---
> **Question:** Given an array of numbers arr and a window of size k, print out the median of each window of size k starting from the left and moving right by one position each time.
>
> For example, given that `k = 3` and array equals `[-1, 5, 13, 8, 2, 3, 3, 1]`, your function should return `[5, 8, 8, 3, 3, 3]`

**Explanation:**
```py
Recall that the median of an even-sized list is the average of the two middle numbers.
5 <- median of [-1, 5, 13]
8 <- median of [5, 13, 8]
8 <- median of [13, 8, 2]
3 <- median of [8, 2, 3]
3 <- median of [2, 3, 3]
3 <- median of [3, 3, 1]
```

**My thoughts:** Maintain a max heap and a min heap where all elem in max heap is smaller than min heap. Together, they can hold k elements. For each new elem incoming, there must be an element kickout. Determine if they are on the same side. If not, we will need to balance both heap by moving element to the other. As removed element, we can lazily remove them: after each iteration, we need to double-check the head of each heap in order to make sure those element are within the window.

**Solution with Priority Queue:** [https://repl.it/@trsong/Sliding-Window-Median](https://repl.it/@trsong/Sliding-Window-Median)
```py
import unittest
from Queue import PriorityQueue

def sliding_median(nums, k):
    max_heap = PriorityQueue()  # left heap
    min_heap = PriorityQueue()  # right heap

    for i in xrange(k):
        max_heap.put((-nums[i], i))

    while min_heap.qsize() < max_heap.qsize():
        move_head(max_heap, min_heap)

    res = [find_median(max_heap, min_heap, k)]

    for i in xrange(k, len(nums)):
        cur_num = nums[i]
        removed_num = nums[i - k]
        if cur_num >= min_heap.queue[0][0]:
            # insert new elem into the right heap
            min_heap.put((cur_num, i))
            if removed_num <= min_heap.queue[0][0]:
                # the removed elem create a hole on the left heap
                move_head(min_heap, max_heap)
        else:
            # insert new elem into left heap
            max_heap.put((-cur_num, i))
            if removed_num >= min_heap.queue[0][0]:
                # the removed elem create a hole on the right heap
                move_head(max_heap, min_heap)

        while not max_heap.empty() and max_heap.queue[0][1] <= i - k:
            max_heap.get()

        while not min_heap.empty() and min_heap.queue[0][1] <= i - k:
            min_heap.get()

        res.append(find_median(max_heap, min_heap, k))

    return res


def move_head(pq1, pq2):
    num, i = pq1.get()
    pq2.put((-num, i))


def find_median(max_heap, min_heap, k):
    if k % 2 == 0:
        neg_num1, _ = max_heap.queue[0]
        num2, _ = min_heap.queue[0]
        return (-neg_num1 + num2) * 0.5
    else:
        num, _ = min_heap.queue[0]
        return num


class SlidingMedianSpec(unittest.TestCase):
    def test_example(self):
        k, nums = 3, [-1, 5, 13, 8, 2, 3, 3, 1]
        expected = [5, 8, 8, 3, 3, 3]
        self.assertEqual(expected, sliding_median(nums, k))

    def test_example2(self):
        k, nums = 3, [1, 3, -1, -3, 5, 3, 6, 7]
        expected = [1, -1, -1, 3, 5, 6]
        self.assertEqual(expected, sliding_median(nums, k))

    def test_k_equals_array_size(self):
        k, nums = 5, [1, 2, 3, 4, 5]
        expected = [3]
        self.assertEqual(expected, sliding_median(nums, k))

    def test_k_equals_array_size2(self):
        k, nums = 4, [1, 2, 3, 4]
        expected = [2.5]
        self.assertEqual(expected, sliding_median(nums, k))

    def test_k_equals_one(self):
        k, nums = 1, [3, 1, 2, 4, 1, 2]
        expected = [3, 1, 2, 4, 1, 2]
        self.assertEqual(expected, sliding_median(nums, k))

    def test_even_sized_window(self):
        k, nums = 2, [1, 3, 3, 1, 1, 3, 3, 1]
        expected = [2, 3, 2, 1, 2, 3, 2]
        self.assertEqual(expected, sliding_median(nums, k))

    def test_array_with_duplicated_elements(self):
        k, nums = 5, [1, 1, 1, 1, 1, 1, 1, 1]
        expected = [1, 1, 1, 1]
        self.assertEqual(expected, sliding_median(nums, k))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 16, 2020 LC 394 \[Medium\] Decode String
---
> **Question:** Given a string with a certain rule: `k[string]` should be expanded to string `k` times. So for example, `3[abc]` should be expanded to `abcabcabc`. Nested expansions can happen, so `2[a2[b]c]` should be expanded to `abbcabbc`.


**Solution:** [https://repl.it/@trsong/LC-394-Decode-String](https://repl.it/@trsong/LC-394-Decode-String)
```py
import unittest

def decode_string(encoded_string):
    stack = []
    count = 0
    pattern = []
    for ch in encoded_string:
        if ch.isdigit():
            count = 10 * count + int(ch)
        elif ch == '[':
            prev_str = ''.join(pattern)
            prev_count = count
            stack.append((prev_str, prev_count))
            count = 0
            pattern = []
        elif ch == ']':
            pattern_str = ''.join(pattern)
            prev_str, count = stack.pop()
            pattern = [prev_str, pattern_str * count]
            count = 0
        else:
            pattern.append(ch)
    return ''.join(pattern)
        

class DecodeStringSpec(unittest.TestCase):
    def test_example1(self):
        input = "3[abc]"
        expected = 3 * "abc"
        self.assertEqual(expected, decode_string(input))

    def test_example2(self):
        input = "2[a2[b]c]"
        expected = 2 * ('a' + 2 * 'b' + 'c')
        self.assertEqual(expected, decode_string(input)) 

    def test_example3(self):
        input = "3[a]2[bc]"
        expected = "aaabcbc"
        self.assertEqual(expected, decode_string(input)) 
    
    def test_example4(self):
        input = "3[a2[c]]"
        expected = "accaccacc"
        self.assertEqual(expected, decode_string(input)) 

    def test_example5(self):
        input = "2[abc]3[cd]ef"
        expected = "abcabccdcdcdef"
        self.assertEqual(expected, decode_string(input)) 

    def test_empty_string(self):
        self.assertEqual("", decode_string(""))
        self.assertEqual("", decode_string("42[]"))

    def test_not_decode_negative_number_of_strings(self):
        input = "-3[abc]"
        expected = "-abcabcabc"
        self.assertEqual(expected, decode_string(input))

    def test_duplicate_more_than_10_times(self):
        input = "233[ab]"
        expected =  233 * "ab"
        self.assertEqual(expected, decode_string(input))

    def test_2Level_nested_encoded_string(self):
        input = "2[3[a]3[bc]2[d]]4[2[e]]"
        expected = 2 * (3*"a" + 3*"bc" + 2*"d") + 4*2*"e"
        self.assertEqual(expected, decode_string(input))

    def test_3Level_nested_encoded_string(self):
        input = "2[a2[b3[c]d4[ef]g]h]"
        expected = 2*('a' + 2*('b' + 3*'c' + 'd' + 4 * 'ef' + 'g' ) + 'h')
        self.assertEqual(expected, decode_string(input))


if __name__ == "__main__":
    unittest.main(exit=False)
```

### May 15, 2020 \[Medium\] Longest Consecutive Sequence in an Unsorted Array
---
> **Question:** Given an array of integers, return the largest range, inclusive, of integers that are all included in the array.
>
> For example, given the array `[9, 6, 1, 3, 8, 10, 12, 11]`, return `(8, 12)` since `8, 9, 10, 11, and 12` are all in the array.

**My thoughts:** We can use BFS to find max length of consecutive sequence. Two number are neighbor if they differ by 1. Thus for any number we can scan though its left-most and right-most neighbor recursively. 

**Solution with BFS:** [https://repl.it/@trsong/Longest-Consecutive-Sequence-in-an-Unsorted-Array](https://repl.it/@trsong/Longest-Consecutive-Sequence-in-an-Unsorted-Array)
```py
import unittest

def longest_consecutive_seq(nums):
    if not nums:
        return None

    num_set = set(nums)
    max_length = 0
    global_lo = global_hi = 0

    while num_set:
        pivot = next(iter(num_set))
        num_set.remove(pivot)
        lower_bound = pivot - 1
        upper_bound = pivot + 1

        while num_set and lower_bound in num_set:
            num_set.remove(lower_bound)
            lower_bound -= 1

        while num_set and upper_bound in num_set:
            num_set.remove(upper_bound)
            upper_bound += 1

        length = upper_bound - lower_bound - 1
        if length > max_length:
            max_length = length
            global_lo = lower_bound
            global_hi = upper_bound

    return (global_lo + 1, global_hi - 1)


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
    unittest.main(exit=False)
```


### May 14, 2020 \[Medium\] Bitwise AND of a Range
---
> **Question:** Write a function that returns the bitwise AND of all integers between M and N, inclusive.

**Solution:** [https://repl.it/@trsong/Bitwise-AND-of-a-Range](https://repl.it/@trsong/Bitwise-AND-of-a-Range)
```py
import unittest

def bitwise_and_of_range(m, n):
    """
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
        # remove the least significant bit
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
    unittest.main(exit=False)
```

### May 13, 2020 \[Hard\] Climb Staircase Problem
---
> **Question:** There exists a staircase with N steps, and you can climb up either 1 or 2 steps at a time. Given N, write a function that **PRINT** out all possible unique ways you can climb the staircase. The **ORDER** of the steps matters. 
>
> For example, if N is 4, X = {1, 2} then there are 5 unique ways. This time we print them out as the following:

```py
1, 1, 1, 1
2, 1, 1
1, 2, 1
1, 1, 2
2, 2
```

> What if, instead of being able to climb 1 or 2 steps at a time, you could climb any number from a set of positive integers X? 
>
> For example, if N is 6, and X = {2, 5}. You could climb 2 or 5 steps at a time. Then there is only 1 unique way, so we print the following:

```py
2, 2, 2
```
**My thoughts:** The only way to figure out each path is to manually test all outcomes. However, certain cases are invalid (like exceed the target value while climbing) so we try to modify certain step until its valid. Such technique is called ***Backtracking***.

We may use recursion to implement backtracking, each recursive step will create a separate branch which also represent different recursive call stacks. Once the branch is invalid, the call stack will bring us to a different branch, i.e. backtracking to a different solution space.

For example, if N is 4, and feasible steps are `[1, 2]`, then there are 5 different solution space/path. Each node represents a choice we made and each branch represents a recursive call.

Note we also keep track of the remaining steps while doing recursion. 
```py
├ 1 
│ ├ 1
│ │ ├ 1
│ │ │ └ 1 SUCCEED
│ │ └ 2 SUCCEED
│ └ 2 
│   └ 1 SUCCEED
└ 2 
  ├ 1
  │ └ 1 SUCCEED
  └ 2 SUCCEED
```

If N is 6 and fesible step is `[5, 2]`:
```py
├ 5 FAILURE
└ 2 
  └ 2
    └ 2 SUCCEED
```

**Solution with Backtracking:** [https://repl.it/@trsong/Climb-Staircase-Problem](https://repl.it/@trsong/Climb-Staircase-Problem)
```py
import unittest

def climb_stairs(n, feasible_steps):
    sorted_feasible_steps = sorted(list(set(feasible_steps)))
    res = []
    backtack(res, [], n, sorted_feasible_steps)
    if not res:
        return None
    else:
        return "\n".join(res)


def backtack(res, path, remain, feasible_steps):
    if remain == 0:
        path_str = ", ".join(path)
        res.append(path_str)
    else:
        for step in feasible_steps:
            if step > remain:
                break
            
            path.append(str(step))
            backtack(res, path, remain - step, feasible_steps)
            path.pop()


class ClimbStairSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        expected_lines = expected.splitlines()
        result_lines = result.splitlines()
        self.assertEqual(sorted(expected_lines), sorted(result_lines))
        
    def test_example(self):
        n, feasible_steps = 4, [1, 2]
        expected = "\n".join([
            "1, 1, 1, 1", 
            "2, 1, 1",
            "1, 2, 1",
            "1, 1, 2",
            "2, 2"])
        self.assert_result(expected, climb_stairs(n, feasible_steps))

    def test_example2(self):
        n, feasible_steps = 5, [1, 3, 5]
        expected = "\n".join([
            "1, 1, 1, 1, 1",
            "3, 1, 1",
            "1, 3, 1",
            "1, 1, 3",
            "5"])
        self.assert_result(expected, climb_stairs(n, feasible_steps))

    def test_no_feasible_solution(self):
        n, feasible_steps = 9, []
        self.assertIsNone(climb_stairs(n, feasible_steps))

    def test_no_feasible_solution2(self):
        n, feasible_steps = 99, [7]
        self.assertIsNone(climb_stairs(n, feasible_steps))

    def test_only_one_step_qualify(self):
        n, feasible_steps = 42, [41, 42, 43]
        expected = "42"
        self.assert_result(expected, climb_stairs(n, feasible_steps))

    def test_various_way_to_climb(self):
        n, feasible_steps = 4, [1, 2, 3, 4]
        expected = "\n".join([
            "1, 1, 1, 1",
            "1, 1, 2",
            "1, 2, 1",
            "1, 3",
            "2, 1, 1",
            "2, 2",
            "3, 1",
            "4"])
        self.assert_result(expected, climb_stairs(n, feasible_steps))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 12, 2020 \[Medium\] Look-and-Say Sequence
--- 
> **Question:** The "look and say" sequence is defined as follows: beginning with the term 1, each subsequent term visually describes the digits appearing in the previous term. The first few terms are as follows:

```py
1
11
21
1211
111221
```

> As an example, the fourth term is 1211, since the third term consists of one 2 and one 1.
>
> Given an integer N, print the Nth term of this sequence


**Solution:** [https://repl.it/@trsong/Find-the-n-th-Look-and-Say-Sequence](https://repl.it/@trsong/Find-the-n-th-Look-and-Say-Sequence)
```py
import unittest

def look_and_say(n):
    res = "1"
    for _ in xrange(n-1):
        prev = res[0]
        count = 0
        str_buffer = []
        for i in xrange(len(res)+1):
            char = res[i] if i < len(res) else None
            if prev == char:
                count += 1
            else:
                str_buffer.append(str(count))
                str_buffer.append(prev)
                prev = char
                count = 1
        res = "".join(str_buffer)
    
    return res


class LookAndSaySpec(unittest.TestCase):
    def test_1st_term(self):
        self.assertEqual("1", look_and_say(1))
        
    def test_2nd_term(self):
        self.assertEqual("11", look_and_say(2))
        
    def test_3rd_term(self):
        self.assertEqual("21", look_and_say(3))
        
    def test_4th_term(self):
        self.assertEqual("1211", look_and_say(4))
        
    def test_5th_term(self):
        self.assertEqual("111221", look_and_say(5))
        
    def test_6th_term(self):
        self.assertEqual("312211", look_and_say(6))
        
    def test_7th_term(self):
        self.assertEqual("13112221", look_and_say(7))

    def test_10th_term(self):
        self.assertEqual("13211311123113112211", look_and_say(10))

if __name__ == '__main__':
    unittest.main(exit=False)
```

### May 11, 2020 \[Easy\] Compare Version Numbers
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

**Solution:** [https://repl.it/@trsong/Version-Number-Comparison](https://repl.it/@trsong/Version-Number-Comparison)
```py
import unittest

def compare(v1, v2):
    version_seq1 = map(int, v1.split('.') if v1 else [0])
    version_seq2 = map(int, v2.split('.') if v2 else [0])

    max_len = max(len(version_seq1), len(version_seq2))

    for _ in xrange(max_len - len(version_seq1)):
        version_seq1.append(0)

    for _ in xrange(max_len - len(version_seq2)):
        version_seq2.append(0)

    for num1, num2 in zip(version_seq1, version_seq2):
        if num1 == num2:
            continue
        elif num1 < num2:
            return -1
        else:
            return 1
    
    return 0


class VersionNumberCompareSpec(unittest.TestCase):
    def test_example1(self):
        version1 = "1.0.33"
        version2 = "1.0.27"
        self.assertEqual(1, compare(version1, version2))

    def test_example2(self):
        version1 = "0.1"
        version2 = "1.1"
        self.assertEqual(-1, compare(version1, version2))

    def test_example3(self):
        version1 = "1.01"
        version2 = "1.001"
        self.assertEqual(0, compare(version1, version2))

    def test_example4(self):
        version1 = "1.0"
        version2 = "1.0.0"
        self.assertEqual(0, compare(version1, version2))

    def test_unspecified_version_numbers(self):
        self.assertEqual(0, compare("", ""))
        self.assertEqual(-1, compare("", "1"))
        self.assertEqual(1, compare("2", ""))

    def test_unaligned_zeros(self):
        version1 = "00000.00000.00000.0"
        version2 = "0.00000.000.00.00000.000.000.0"
        self.assertEqual(0, compare(version1, version2))

    def test_same_version_yet_unaligned(self):
        version1 = "00001.001"
        version2 = "1.000001.0000000.0000"
        self.assertEqual(0, compare(version1, version2))

    def test_different_version_numbers(self):
        version1 = "1.2.3.4"
        version2 = "1.2.3.4.5"
        self.assertEqual(-1, compare(version1, version2))

    def test_different_version_numbers2(self):
        version1 = "3.2.1"
        version2 = "3.1.2.3"
        self.assertEqual(1, compare(version1, version2))

    def test_different_version_numbers3(self):
        version1 = "00001.001.0.1"
        version2 = "1.000001.0000000.0000"
        self.assertEqual(1, compare(version1, version2))

    def test_without_dots(self):
        version1 = "32123"
        version2 = "3144444"
        self.assertEqual(-1, compare(version1, version2))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### May 10, 2020 \[Medium\] All Root to Leaf Paths in Binary Tree
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

### May 9, 2020  \[Easy\] String Compression
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

### May 8, 2020  \[Easy\] Find Unique Element among Array of Duplicates
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

### May 7, 2020  \[Easy\] Full Binary Tree
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

### May 6, 2020 LC 301 \[Hard\] Remove Invalid Parentheses
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

### May 5, 2020 \[Hard\] Maximum Spanning Tree
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
