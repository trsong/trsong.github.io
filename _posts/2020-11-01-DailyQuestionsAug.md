---
layout: post
title:  "Daily Coding Problems 2020 Nov to Jan"
date:   2020-11-01 22:22:32 -0700
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


### Nov 15, 2020 \[Medium\] Number of Flips to Make Binary String
---
> **Question:** You are given a string consisting of the letters `x` and `y`, such as `xyxxxyxyy`. In addition, you have an operation called flip, which changes a single `x` to `y` or vice versa.
>
> Determine how many times you would need to apply this operation to ensure that all x's come before all y's. In the preceding example, it suffices to flip the second and sixth characters, so you should return 2.


### Nov 14, 2020 \[Medium\] Isolated Islands
---
> **Question:** Given a matrix of 1s and 0s, return the number of "islands" in the matrix. A 1 represents land and 0 represents water, so an island is a group of 1s that are neighboring whose perimeter is surrounded by water.
>
> For example, this matrix has 4 islands.

```py
1 0 0 0 0
0 0 1 1 0
0 1 1 0 0
0 0 0 0 0
1 1 0 0 1
1 1 0 0 1
```


### Nov 13, 2020 \[Medium\] Find Missing Positive
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


**Soluion:** [https://repl.it/@trsong/Find-First-Missing-Positive](https://repl.it/@trsong/Find-First-Missing-Positive)
```py
import unittest

def find_missing_positive(nums):
    n = len(nums)
    value_to_index = lambda value: value - 1

    for i in xrange(n):
        while 1 <= nums[i] <= n and value_to_index(nums[i]) != i:
            target_index = value_to_index(nums[i])
            if nums[i] == nums[target_index]:
                break
            nums[i], nums[target_index] = nums[target_index], nums[i]

    for i, num in enumerate(nums):
        if value_to_index(num) != i:
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
    unittest.main(exit=False)
```


### Nov 12, 2020 \[Medium\] Max Value of Coins to Collect in a Matrix
---
> **Question:** You are given a 2-d matrix where each cell represents number of coins in that cell. Assuming we start at `matrix[0][0]`, and can only move right or down, find the maximum number of coins you can collect by the bottom right corner.

**Example:**

```py
Given below matrix:

0 3 1 1
2 0 0 4
1 5 3 1

The most we can collect is 0 + 2 + 1 + 5 + 3 + 1 = 12 coins.
```

**My thoughts:** This problem gives you a strong feeling that this must be a DP question. 'coz for each step you can either move right or down, that is, max number of coins you can collect so far at current cell depends on top and left solution gives the following recurrence formula: 

```py
Let dp[i][j] be the max coin value collect when reach cell (i, j) in grid.
dp[i][j] = grid[i][j] + max(dp[i-1][j], dp[i][j-1])
```

You can also do it in-place using original grid. However, mutating input params in general is a bad habit as those parameters may be used in other place and might be immutable.


**Solution with DP:** [https://repl.it/@trsong/Find-Max-Value-of-Coins-to-Collect-in-a-Matrix](https://repl.it/@trsong/Find-Max-Value-of-Coins-to-Collect-in-a-Matrix)
```py
import unittest

def max_coins(grid):
    if not grid or not grid[0]:
        return 0

    n, m = len(grid), len(grid[0])
    dp = [[0 for _ in xrange(m)] for _ in xrange(n)]

    for r in xrange(n):
        for c in xrange(m):
            left_max = dp[r-1][c] if r > 0 else 0
            top_max = dp[r][c-1] if c > 0 else 0
            dp[r][c] = grid[r][c] + max(left_max, top_max)

    return dp[n-1][m-1]


class MaxCoinSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(max_coins([
            [0, 3, 1, 1],
            [2, 0, 0, 4],
            [1, 5, 3, 1]
        ]), 12)

    def test_empty_grid(self):
        self.assertEqual(max_coins([]), 0)
        self.assertEqual(max_coins([[]]), 0)

    def test_one_way_or_the_other(self):
        self.assertEqual(max_coins([
            [0, 3],
            [2, 0]
        ]), 3)

    def test_until_the_last_moment_knows(self):
        self.assertEqual(max_coins([
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [2, 0, 3, 0]
        ]), 5)

    def test_try_to_get_most_coins(self):
        self.assertEqual(max_coins([
            [1, 1, 1],
            [2, 3, 1],
            [1, 4, 5]
        ]), 15)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 11, 2020 LC 859 \[Easy\] Buddy Strings
---
> **Question:** Given two strings A and B of lowercase letters, return true if and only if we can swap two letters in A so that the result equals B.

**Example 1:**
```py
Input: A = "ab", B = "ba"
Output: true
```

**Example 2:**
```py
Input: A = "ab", B = "ab"
Output: false
```

**Example 3:**
```py
Input: A = "aa", B = "aa"
Output: true
```

**Example 4:**
```py
Input: A = "aaaaaaabc", B = "aaaaaaacb"
Output: true
```

**Example 5:**
```py
Input: A = "", B = "aa"
Output: false
```

**Solution:** [https://repl.it/@trsong/Buddy-Strings](https://repl.it/@trsong/Buddy-Strings)
```py
import unittest

def is_buddy_string(s1, s2):
    if len(s1) != len(s2):
        return False

    unmatches = []
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            unmatches.append((c1, c2))
        if len(unmatches) > 2:
            return False

    if len(unmatches) == 2:
        return unmatches[0] == unmatches[1][::-1]
    elif len(unmatches) == 0:
        return has_duplicates(s1)
    else:
        return False 


def has_duplicates(s):
    visited = set()
    for c in s:
        if c in visited:
            return True
        visited.add(c)
    return False


class IsBuddyString(unittest.TestCase):
    def test_example(self):
        self.assertTrue(is_buddy_string('ab', 'ba'))

    def test_example2(self):
        self.assertFalse(is_buddy_string('ab', 'ab'))

    def test_example3(self):
        self.assertTrue(is_buddy_string('aa', 'aa'))

    def test_example4(self):
        self.assertTrue(is_buddy_string('aaaaaaabc', 'aaaaaaacb'))

    def test_example5(self):
        self.assertFalse(is_buddy_string('ab', 'aa'))

    def test_empty_string(self):
        self.assertFalse(is_buddy_string('', ''))

    def test_single_char_string(self):
        self.assertFalse(is_buddy_string('a', 'b'))

    def test_same_string_without_duplicates(self):
        self.assertFalse(is_buddy_string('abc', 'abc'))

    def test_string_with_duplicates(self):
        self.assertFalse(is_buddy_string('aba', 'abc'))

    def test_different_length_string(self):
        self.assertFalse(is_buddy_string('aa', 'aaa'))

    def test_different_length_string2(self):
        self.assertFalse(is_buddy_string('ab', 'baa'))

    def test_different_length_string3(self):
        self.assertFalse(is_buddy_string('ab', 'abba'))

    def test_swap_failure(self):
        self.assertFalse(is_buddy_string('abcaa', 'abcbb'))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 10, 2020 \[Medium\] All Root to Leaf Paths in Binary Tree
---
> **Question:** Given a binary tree, return all paths from the root to leaves.

**Example:** 
```py
Given the tree:
   1
  / \
 2   3
    / \
   4   5

Return [[1, 2], [1, 3, 4], [1, 3, 5]]
```

**Solution with Backtracking:** [https://repl.it/@trsong/Print-All-Root-to-Leaf-Paths-in-Binary-Tree](https://repl.it/@trsong/Print-All-Root-to-Leaf-Paths-in-Binary-Tree)
```py
import unittest

def path_to_leaves(tree):
    if not tree:
        return []
    res = []
    backtrack(res, tree, [])
    return res


def backtrack(res, current_node, path):
    if not current_node.left and not current_node.right:
        res.append(path + [current_node.val])
    else:
        for child in [current_node.left, current_node.right]:
            if not child:
                continue
            path.append(current_node.val)
            backtrack(res, child, path)
            path.pop()

    
class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right= right


class PathToLeavesSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertEqual([], path_to_leaves(None))

    def test_one_level_tree(self):
        self.assertEqual([[1]], path_to_leaves(TreeNode(1)))

    def test_two_level_tree(self):
        """
          1
         / \
        2   3
        """
        tree = TreeNode(1, TreeNode(2), TreeNode(3))
        expected = [[1, 2], [1, 3]]
        self.assertEqual(expected, path_to_leaves(tree))

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
        expected = [[1, 2], [1, 3, 4], [1, 3, 5]]
        self.assertEqual(expected, path_to_leaves(tree))

    def test_complete_tree(self):
        """
               1
             /   \
            2     3
           / \   /
          4   5 6
        """
        left_tree = TreeNode(2, TreeNode(4), TreeNode(5))
        right_tree = TreeNode(3, TreeNode(6))
        tree = TreeNode(1, left_tree, right_tree)
        expected = [[1, 2, 4], [1, 2, 5], [1, 3, 6]]
        self.assertEqual(expected, path_to_leaves(tree))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 9, 2020 \[Medium\] Invert a Binary Tree
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

**Solution:** [https://repl.it/@trsong/Invert-All-Nodes-in-Binary-Tree](https://repl.it/@trsong/Invert-All-Nodes-in-Binary-Tree)
```py
import unittest

def invert_tree(root):
    if root:
        root.left, root.right = invert_tree(root.right), invert_tree(root.left)
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
    unittest.main(exit=False)
```

### Nov 8, 2020 \[Medium\] Similar Websites
---
> **Question:** You are given a list of (website, user) pairs that represent users visiting websites. Come up with a program that identifies the top k pairs of websites with the greatest similarity.
>
> **Note:** The similarity metric bewtween two sets equals intersection / union. 

**Example:**
```py
Suppose k = 1, and the list of tuples is:

[('a', 1), ('a', 3), ('a', 5),
 ('b', 2), ('b', 6),
 ('c', 1), ('c', 2), ('c', 3), ('c', 4), ('c', 5),
 ('d', 4), ('d', 5), ('d', 6), ('d', 7),
 ('e', 1), ('e', 3), ('e', 5), ('e', 6)]
 
Then a reasonable similarity metric would most likely conclude that a and e are the most similar, so your program should return [('a', 'e')].
```

**My thoughts:** The similarity metric bewtween two sets equals intersection / union. So, the way to get top k similar website is first calculate the similarity score between any two websites and after that use a priority queue to mantain top k similarity pairs.

However, as duplicate entry might occur, we have to treat normal set to multiset: treat `3, 3, 3` as `3(first), 3(second), 3(third)`. 
```py
a: 1 2 3(first) 3(second) 3(third)
b: 1 2 3(first)
The similarty between (a, b) is 3/5
```


**Solution with Priority Queue:** [https://repl.it/@trsong/Find-Similar-Websites](https://repl.it/@trsong/Find-Similar-Websites)
```py
import unittest
from collections import defaultdict
from Queue import PriorityQueue

def top_similar_websites(website_log, k):
    user_site_hits = defaultdict(lambda: defaultdict(int))
    site_hits = defaultdict(int)
    for site, user in website_log:
        user_site_hits[user][site] += 1
        site_hits[site] += 1

    cross_site_hits = defaultdict(lambda: defaultdict(int))
    for site_hits_per_user in user_site_hits.values():
        for site1 in site_hits_per_user:
            for site2 in site_hits_per_user:
                cross_site_hits[site1][site2] += min(site_hits_per_user[site1], site_hits_per_user[site2])

    min_heap = PriorityQueue()
    sites = sorted(site_hits.keys())
    for site1 in sites:
        for site2 in sites:
            if site2 == site1:
                break
            intersection = cross_site_hits[site1][site2]
            total = site_hits[site1] + site_hits[site2]
            union = total - intersection
            similarity = float(intersection) / union

            if min_heap.qsize() >= k and min_heap.queue[0][0] < similarity:
                min_heap.get()
            
            if min_heap.qsize() < k:
                min_heap.put((similarity, (site1, site2)))

    ascending_sites = [min_heap.get()[1] for _ in xrange(k)]
    return ascending_sites[::-1]


class TopSimilarWebsiteSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        # same length
        self.assertEqual(len(expected), len(result))
        for e, r in zip(expected, result):
            # pair must be the same, order doesn't matter
            self.assertEqual(set(e), set(r), "Expected %s but get %s" % (expected, result))

    def test_example(self):
        website_log = [
            ('a', 1), ('a', 3), ('a', 5),
            ('b', 2), ('b', 6),
            ('c', 1), ('c', 2), ('c', 3), ('c', 4), ('c', 5),
            ('d', 4), ('d', 5), ('d', 6), ('d', 7),
            ('e', 1), ('e', 3), ('e', 5), ('e', 6)]
        # Similarity: (a,e)=3/4, (a,c)=3/5, (c, e)=1/2
        expected = [('a', 'e'), ('a', 'c'), ('c', 'e')]
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


### Nov 7, 2020 \[Medium\] Largest Square
---
> **Question:** Given an N by M matrix consisting only of 1's and 0's, find the largest square matrix containing only 1's and return its dimension size.

**Example:**
```py
Given the following matrix:

[[1, 0, 0, 0],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [0, 1, 0, 0]]

Return 2. As the following 1s form the largest square matrix containing only 1s:
 [1, 1],
 [1, 1]
```

**Solution with DP:** [https://repl.it/@trsong/Largest-Square](https://repl.it/@trsong/Largest-Square)
```py
import unittest

def largest_square_dimension(grid):
    if not grid or not grid[0]:
        return 0
    n, m = len(grid), len(grid[0])

    # Let dp[r][c] represents max dimension of square matrix with bottom right corner at (r, c)
    # dp[r][c] = min(dp[r-1][c], dp[r][c-1], dp[r-1][c-1]) + 1 if (r, c) is 1
    #          = 0 otherwise
    dp = [[grid[r][c] for c in xrange(m)] for r in xrange(n)]

    for r in xrange(1, n):
        for c in xrange(1, m):
            if grid[r][c] == 0:
                continue
            dp[r][c] = 1 + min(dp[r-1][c], dp[r][c-1], dp[r-1][c-1])
    
    return max(map(max, dp))


class LargestSquareDimensionSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(2, largest_square_dimension([
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 0, 0]
        ]))

    def test_empty_grid(self):
        self.assertEqual(0, largest_square_dimension([]))

    def test_1d_grid(self):
        self.assertEqual(1, largest_square_dimension([
            [0, 0, 0, 0, 1, 0, 0]
        ]))

    def test_1d_grid2(self):
        self.assertEqual(0, largest_square_dimension([
            [0],
            [0],
            [0]
        ]))

    def test_dimond_shape(self):
        self.assertEqual(3, largest_square_dimension([
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0]
        ]))

    def test_dimond_shape2(self):
        self.assertEqual(3, largest_square_dimension([
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0]
        ]))

    def test_square_on_edge(self):
        self.assertEqual(2, largest_square_dimension([
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 0]
        ]))

    def test_dense_matrix(self):
        self.assertEqual(3, largest_square_dimension([
            [0, 1, 1, 0, 1], 
            [1, 1, 0, 1, 0], 
            [0, 1, 1, 1, 0], 
            [1, 1, 1, 1, 0], 
            [1, 1, 1, 1, 1], 
            [0, 0, 0, 0, 0]
        ]))



if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 6, 2020 \[Medium\] M Smallest in K Sorted Lists
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

**Solution:** [https://repl.it/@trsong/Find-the-M-thSmallest-in-K-Sorted-Lists](https://repl.it/@trsong/Find-the-M-thSmallest-in-K-Sorted-Lists)
```py
import unittest
from Queue import PriorityQueue

def find_m_smallest(ksorted_list, m):
    pq = PriorityQueue()
    for lst in ksorted_list:
        if not lst:
            continue
        it = iter(lst)
        pq.put((it.next(), it))

    while not pq.empty():
        num, it = pq.get()
        if m == 1:
            return num
        m -= 1

        next_num = next(it, None)
        if next_num is not None:
            pq.put((next_num, it))
    
    return None
        

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

### Nov 5, 2020 \[Easy\] Maximum Subarray Sum 
---
> **Question:** You are given a one dimensional array that may contain both positive and negative integers, find the sum of contiguous subarray of numbers which has the largest sum.
>
> For example, if the given array is `[-2, -5, 6, -2, -3, 1, 5, -6]`, then the maximum subarray sum is 7 as sum of `[6, -2, -3, 1, 5]` equals 7
>
> Solve this problem with Divide and Conquer as well as DP separately.


**Solution with Divide and Conquer:** [https://repl.it/@trsong/Maximum-Subarray-Sum-Divide-and-Conquer](https://repl.it/@trsong/Maximum-Subarray-Sum-Divide-and-Conquer)
```py
def max_sub_array_sum(nums):
    if not nums:
        return 0
    res = max_sub_array_sum_recur(nums, 0, len(nums) - 1)
    return res.max


def max_sub_array_sum_recur(nums, lo, hi):
    if lo == hi:
        return Result(nums[lo])
    
    mid = lo + (hi - lo) // 2
    left_res = max_sub_array_sum_recur(nums, lo, mid)
    right_res = max_sub_array_sum_recur(nums, mid+1, hi)

    res = Result(0)
    # From left to right, expand range
    res.prefix = max(0, left_res.prefix, left_res.sum + right_res.prefix, left_res.sum + right_res.sum)
    # From right to left, expand range
    res.suffix = max(0, right_res.suffix, right_res.sum + left_res.suffix, right_res.sum + left_res.sum)
    res.sum = left_res.sum + right_res.sum
    res.max = max(res.prefix, res.suffix, left_res.suffix + right_res.prefix, left_res.max, right_res.max)
    return res


class Result(object):
    def __init__(self, res):
        self.prefix = res
        self.suffix = res
        self.sum = res
        self.max = res
```

**Solution with DP:** [https://repl.it/@trsong/Maximum-Subarray-Sum-DP](https://repl.it/@trsong/Maximum-Subarray-Sum-DP)
```py
import unittest

def max_sub_array_sum(nums):
    n = len(nums)
    # Let dp[i] represents max sub array sum ends at nums[i-1]
    # dp[i] = max(0, dp[i-1] + nums[i-1])
    dp = [0] * (n + 1)
    res = 0
    for i in xrange(1, n+1):
        dp[i] = max(0, dp[i-1] + nums[i-1])
        res = max(res, dp[i])
    return res


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
    unittest.main(exit=False)
```

### Nov 4, 2020 \[Easy\] Largest Path Sum from Root To Leaf
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

**Solution with DFS:** [https://repl.it/@trsong/Find-Largest-Path-Sum-from-Root-To-Leaf](https://repl.it/@trsong/Find-Largest-Path-Sum-from-Root-To-Leaf)
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
    parent_lookup = {}
    stack = [(root, 0)]
    max_sum = float('-inf')
    max_sum_node = root

    while stack:
        cur, prev_sum = stack.pop()
        cur_sum = prev_sum + cur.val
        if not cur.left and not cur.right and cur_sum > max_sum:
            max_sum = cur_sum
            max_sum_node = cur
        else:
            for child in [cur.left, cur.right]:
                if not child:
                    continue
                parent_lookup[child] = cur
                stack.append((child, cur_sum))
    
    res = []
    node = max_sum_node
    while node:
        res.append(node.val)
        node = parent_lookup.get(node, None)
    return res[::-1]


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

### Nov 3, 2020 LC 121 \[Easy\] Best Time to Buy and Sell Stock
---
> **Question:** You are given an array. Each element represents the price of a stock on that particular day. Calculate and return the maximum profit you can make from buying and selling that stock only once.

**Example:**
```py
Input: [9, 11, 8, 5, 7, 10]
Output: 5
Explanation: Here, the optimal trade is to buy when the price is 5, and sell when it is 10, so the return value should be 5 (profit = 10 - 5 = 5).
```

**Solution:** [https://repl.it/@trsong/Best-Time-to-Buy-and-Sell-Stock](https://repl.it/@trsong/Best-Time-to-Buy-and-Sell-Stock)
```py
import unittest

def max_profit(stock_data):
    if not stock_data:
        return 0
    local_min = stock_data[0]
    res = 0
    for price in stock_data:
        if price < local_min:
            local_min = price
        else:
            res = max(res, price - local_min)
    return res


class MaxProfitSpec(unittest.TestCase):
    def test_blank_data(self):
        self.assertEqual(max_profit([]), 0)
    
    def test_1_day_data(self):
        self.assertEqual(max_profit([9]), 0)
        self.assertEqual(max_profit([-1]), 0)

    def test_monotonically_increase(self):
        self.assertEqual(max_profit([1, 2, 3]), 2)
        self.assertEqual(max_profit([1, 1, 1, 2, 2, 3, 3, 3]), 2)
    
    def test_monotonically_decrease(self):
        self.assertEqual(max_profit([3, 2, 1]), 0)
        self.assertEqual(max_profit([3, 3, 3, 2, 2, 1, 1, 1]), 0)

    def test_raise_suddenly(self):
        self.assertEqual(max_profit([3, 2, 1, 1, 2]), 1)
        self.assertEqual(max_profit([3, 2, 1, 1, 9]), 8)

    def test_drop_sharply(self):
        self.assertEqual(max_profit([1, 3, 0]), 2)
        self.assertEqual(max_profit([1, 3, -1]), 2)

    def test_bear_market(self):
        self.assertEqual(max_profit([10, 11, 5, 7, 1, 2]), 2)
        self.assertEqual(max_profit([10, 11, 1, 4, 2, 7, 5]), 6)

    def test_bull_market(self):
        self.assertEqual(max_profit([1, 5, 3, 7, 2, 14, 10]), 13)
        self.assertEqual(max_profit([5, 1, 11, 10, 12]), 11)


if __name__ == '__main__':
    unittest.main(exit=False)
```
 
### Nov 2, 2020 LC 403 \[Hard\] Frog Jump
---
> **Question:** A frog is crossing a river. The river is divided into x units and at each unit there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.
>
> Given a list of stones' positions (in units) in sorted ascending order, determine if the frog is able to cross the river by landing on the last stone. Initially, the frog is on the first stone and assume the first jump must be 1 unit.
> 
> If the frog's last jump was k units, then its next jump must be either k - 1, k, or k + 1 units. Note that the frog can only jump in the forward direction.

**Example 1:**

```py
[0, 1, 3, 5, 6, 8, 12, 17]

There are a total of 8 stones.
The first stone at the 0th unit, second stone at the 1st unit,
third stone at the 3rd unit, and so on...
The last stone at the 17th unit.

Return true. The frog can jump to the last stone by jumping 
1 unit to the 2nd stone, then 2 units to the 3rd stone, then 
2 units to the 4th stone, then 3 units to the 6th stone, 
4 units to the 7th stone, and 5 units to the 8th stone.
```

**Example 2:**

```py
[0, 1, 2, 3, 4, 8, 9, 11]

Return false. There is no way to jump to the last stone as 
the gap between the 5th and 6th stone is too large.
```

**Solution with DFS:** [https://repl.it/@trsong/Solve-Frog-Jump-Problem](https://repl.it/@trsong/Solve-Frog-Jump-Problem)
```py
import unittest

def can_cross(stones):
    stone_set = set(stones)
    visited = set()
    stack = [(0, 0)]
    goal = stones[-1]

    while stack:
        stone, step = stack.pop()
        if stone == goal:
            return True
        visited.add((stone, step))
        for delta in [-1, 0, 1]:
            next_step = step + delta
            next_stone = stone + next_step
            if next_stone >= stone and next_stone in stone_set and (next_stone, next_step) not in visited:
                stack.append((next_stone, next_step))

    return False
        

class CanCrossSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(can_cross([0, 1, 3, 5, 6, 8, 12, 17])) # step: 1(1), 2(3), 2(5), 3(8), 4(12), 5(17)

    def test_example2(self):
        self.assertFalse(can_cross([0, 1, 2, 3, 4, 8, 9, 11]))

    def test_fast_then_slow(self):
        self.assertTrue(can_cross([0, 1, 3, 6, 10, 13, 15, 16, 16]))

    def test_fast_then_cooldown(self):
        self.assertFalse(can_cross([0, 1, 3, 6, 10, 11]))

    def test_unreachable_last_stone(self):
        self.assertFalse(can_cross([0, 1, 3, 6, 11]))

    def test_reachable_last_stone(self):
        self.assertTrue(can_cross([0, 1, 3, 6, 10]))

    def test_fall_into_water_in_the_middle(self):
        self.assertFalse(can_cross([0, 1, 10, 1000, 1000]))
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 1, 2020 \[Medium\] The Tower of Hanoi
---
> **Question:** The Tower of Hanoi is a puzzle game with three rods and n disks, each a different size.
>
> All the disks start off on the first rod in a stack. They are ordered by size, with the largest disk on the bottom and the smallest one at the top.
>
> The goal of this puzzle is to move all the disks from the first rod to the last rod while following these rules:
>
> - You can only move one disk at a time.
> - A move consists of taking the uppermost disk from one of the stacks and placing it on top of another stack.
> - You cannot place a larger disk on top of a smaller disk.
>
> Write a function that prints out all the steps necessary to complete the Tower of Hanoi. 
> - You should assume that the rods are numbered, with the first rod being 1, the second (auxiliary) rod being 2, and the last (goal) rod being 3.

**Example:** 
```py
with n = 3, we can do this in 7 moves:

Move 1 to 3
Move 1 to 2
Move 3 to 2
Move 1 to 3
Move 2 to 1
Move 2 to 3
Move 1 to 3
```

**My thoughts:** Think about the problem backwards, like what is the most significant states to reach the final state. There are three states coming into my mind: 

- First state, we move all disks except for last one from rod 1 to rod 2. i.e. `[[3], [1, 2], []]`.
- Second state, we move the last disk from rod 1 to rod 3. i.e. `[[], [1, 2], [3]]`
- Third state, we move all disks from rod 2 to rod 3. i.e. `[[], [], [1, 2, 3]]`

There is a clear recrusive relationship between game with size n and size n - 1. So we can perform above stategy recursively for game with size n - 1 which gives the following implementation.

**Solution with Divide-and-Conquer:** [https://repl.it/@trsong/Solve-the-Tower-of-Hanoi-Problem](https://repl.it/@trsong/Solve-the-Tower-of-Hanoi-Problem)
```py
import unittest

def hanoi_moves(n):
    res = []

    def hanoi_moves_recur(n, src, dst):
        if n <= 0:
            return 

        bridge = 3 - src - dst
        # use the unused rod as bridge rod
        # Step1: move n - 1 disks from src to bridge to allow last disk move to dst
        hanoi_moves_recur(n - 1, src, bridge)

        # Step2: move last disk from src to dst
        res.append((src, dst))

        # Step3: move n - 1 disks from bridge to dst
        hanoi_moves_recur(n - 1, bridge, dst)

    hanoi_moves_recur(n, 0, 2)
    return res


class HanoiMoveSpec(unittest.TestCase):
    def assert_hanoi_moves(self, n, moves):
        game = HanoiGame(n)
        # Turn on verbose for debugging
        self.assertTrue(game.can_moves_finish_game(moves, verbose=False))

    def test_three_disks(self):
        moves = hanoi_moves(3)
        self.assert_hanoi_moves(3, moves)

    def test_one_disk(self):
        moves = hanoi_moves(1)
        self.assert_hanoi_moves(1, moves)

    def test_ten_disks(self):
        moves = hanoi_moves(10)
        self.assert_hanoi_moves(10, moves)


class HanoiGame(object):
    def __init__(self, num_disks):
        self.num_disks = num_disks
        self.reset()
        
    def reset(self):
        self.rods = [[disk for disk in xrange(self.num_disks, 0, -1)], [], []]

    def move(self, src, dst):
        disk = self.rods[src].pop()
        self.rods[dst].append(disk)

    def is_feasible_move(self, src, dst):
        return 0 <= src <= 2 and 0 <= dst <= 2 and self.rods[src] and (not self.rods[dst] or self.rods[src][-1] < self.rods[dst][-1])

    def is_game_finished(self):
        return len(self.rods[-1]) == self.num_disks

    def can_moves_finish_game(self, actions, verbose=False):
        self.reset()
        for step, action in enumerate(actions):
            src, dst = action
            if verbose:
                self.display()
                print "Step %d: %d -> %d" % (step, src, dst)
            if not self.is_feasible_move(src, dst):
                return False
            else:
                self.move(src, dst)
        if verbose:
            self.display()
            
        return self.is_game_finished()

    def display(self):
        for plates in self.rods:
            print "- %s" % str(plates)
    

class HanoiGameSpec(unittest.TestCase):
    def test_example_moves(self):
        game = HanoiGame(3)
        moves = [(0, 2), (0, 1), (2, 1), (0, 2), (1, 0), (1, 2), (0, 2)]
        self.assertTrue(game.can_moves_finish_game(moves))

    def test_invalid_moves(self):
        game = HanoiGame(3)
        moves = [(0, 1), (0, 1)]
        self.assertFalse(game.can_moves_finish_game(moves))

    def test_unfinished_moves(self):
        game = HanoiGame(3)
        moves = [(0, 1)]
        self.assertFalse(game.can_moves_finish_game(moves))


if __name__ == '__main__':
    unittest.main(exit=False)
```
