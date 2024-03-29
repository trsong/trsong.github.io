---
layout: post
title:  "Daily Coding Problems 2020 Aug to Oct"
date:   2020-08-01 22:22:32 -0700
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


### Oct 31, 2020 LC 287 \[Medium\] Find the Duplicate Number
---
> **Question:** You are given an array of length `n + 1` whose elements belong to the set `{1, 2, ..., n}`. By the pigeonhole principle, there must be a duplicate. Find it in linear time and space.
 
**My thoughts:** Use value as the 'next' element index which will form a loop evently. 

Why? Because the following scenarios will happen:

**Scenario 1:** If `a[i] != i for all i`, then since a[1] ... a[n] contains elements 1 to n, each time when interate to next index, one of the element within range 1 to n will be removed until no element is available and/or hit a previous used element and form a loop.  

**Scenario 2:** If `a[i] == i for all i > 0`, then as `a[0] != 0`, we will have a loop 0 -> a[0] -> a[0]

**Scenario 3:** If `a[i] == i for some i > 0`, then like scenario 2 we either we hit i evently or like scenario 1, for each iteration, we consume one element between 1 to n until all elements are used up and form a cycle. 

So we can use a fast and slow pointer to find the element when loop begins. 

**Solution with Fast-Slow Pointers:** [https://repl.it/@trsong/Find-the-Duplicate-Number-from-Array](https://repl.it/@trsong/Find-the-Duplicate-Number-from-Array)
```py
import unittest

def find_duplicate(nums):
    fast = slow = 0
    while True:
        fast = nums[nums[fast]]
        slow = nums[slow]
        if fast == slow:
            break
    
    p = 0 
    while p != slow:
        p = nums[p]
        slow = nums[slow]
    return p


class FindDuplicateSpec(unittest.TestCase):
    def test_all_numbers_are_same(self):
        self.assertEqual(2, find_duplicate([2, 2, 2, 2, 2]))

    def test_number_duplicate_twice(self):
        # index: 0 1 2 3 4 5 6
        # value: 2 6 4 1 3 1 5
        # chain: 0 -> 2 -> 4 -> 3 -> 1 -> 6 -> 5 -> 1
        #                            ^              ^
        self.assertEqual(1, find_duplicate([2, 6, 4, 1, 3, 1, 5]))

    def test_rest_of_element_form_a_loop(self):
        # index: 0 1 2 3 4
        # value: 3 1 3 4 2
        # chain: 0 -> 3 -> 4 -> 2 -> 3
        #             ^              ^
        self.assertEqual(3, find_duplicate([3, 1, 3, 4, 2]))

    def test_rest_of_element_are_sorted(self):
        # index: 0 1 2 3 4
        # value: 4 1 2 3 4
        # chain: 0 -> 4 -> 4
        #             ^    ^
        self.assertEqual(4, find_duplicate([4, 1, 2, 3, 4]))
    
    def test_number_duplicate_more_than_twice(self):
        # index: 0 1 2 3 4 5 6 7 8 9
        # value: 2 5 9 6 9 3 8 9 7 1
        # chain: 0 -> 2 -> 9 -> 1 -> 5 -> 3 -> 6 -> 8 -> 7 -> 9
        #                  ^                                  ^
        self.assertEqual(9, find_duplicate([2, 5, 9, 6, 9, 3, 8, 9, 7, 1]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 30, 2020 LC 652 \[Medium\] Find Duplicate Subtrees
---
> **Question:** Given a binary tree, find all duplicate subtrees (subtrees with the same value and same structure) and return them as a list of list `[subtree1, subtree2, ...]` where `subtree1` is a duplicate of `subtree2` etc.

**Example1:**
```py
Given the following tree:
     1
    / \
   2   2
  /   /
 3   3

The duplicate subtrees are 
  2
 /    And  3
3
```

**Example2:**
```py
Given the following tree:
        1
       / \
      2   3
     /   / \
    4   2   4
       /
      4
      
The duplicate subtrees are 
      2
     /  And  4
    4
```

**Solution with Hash Post-order Traversal:** [https://repl.it/@trsong/Find-Duplicate-Subtree-Nodes](https://repl.it/@trsong/Find-Duplicate-Subtree-Nodes)
```py
import unittest
import hashlib 

def find_duplicate_subtree(tree):
    histogram = {}
    res = []
    postorder_traverse(tree, histogram, res)
    return res


def postorder_traverse(root, histogram, duplicated_nodes):
    if not root:
        return '#'
    
    left_hash = postorder_traverse(root.left, histogram, duplicated_nodes)
    right_hash = postorder_traverse(root.right, histogram, duplicated_nodes)
    current_hash = hash(left_hash + str(root.val) + right_hash)
    histogram[current_hash] = histogram.get(current_hash, 0) + 1
    if histogram[current_hash] == 2:
        duplicated_nodes.append(root)
    return current_hash 
    

def hash(msg):
    result = hashlib.sha256(msg.encode())
    return result.hexdigest()


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


class FindDuplicateSubtreeSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(len(expected), len(result))
        for item in expected:
            self.assertIn(item, result)

    def test_example(self):
        """
        Given the following tree:
             1
            / \
           2   2
          /   /
         3   3

        The duplicate subtrees are 
          2
         /    And  3
        3
        """
        left_tree = TreeNode(2, TreeNode(3))
        right_tree = TreeNode(2, TreeNode(3))
        tree = TreeNode(1, left_tree, right_tree)

        duplicate_tree1 = TreeNode(2, TreeNode(3))
        duplicate_tree2 = TreeNode(3)
        expected = [duplicate_tree1, duplicate_tree2]
        self.assert_result(expected, find_duplicate_subtree(tree))

    def test_example2(self):
        """
        Given the following tree:
                1
               / \
              2   3
             /   / \
            4   2   4
               /
              4
        The duplicate subtrees are 
          2
         /    And  4
        4
        """
        left_tree = TreeNode(2, TreeNode(4))
        right_tree = TreeNode(3, TreeNode(2, TreeNode(4)), TreeNode(4))
        tree = TreeNode(1, left_tree, right_tree)

        duplicate_tree1 = TreeNode(2, TreeNode(4))
        duplicate_tree2 = TreeNode(4)
        expected = [duplicate_tree1, duplicate_tree2]
        self.assert_result(expected, find_duplicate_subtree(tree))

    def test_empty_tree(self):
        self.assertEqual([], find_duplicate_subtree(None))

    def test_all_value_are_equal1(self):
        """
             1
           /   \
          1     1
         / \   / \
        1   1 1   1
        """
        left_tree = TreeNode(1, TreeNode(1), TreeNode(1))
        right_tree = TreeNode(1, TreeNode(1), TreeNode(1))
        tree = TreeNode(1, left_tree, right_tree)

        duplicate_tree1 = TreeNode(1, TreeNode(1), TreeNode(1))
        duplicate_tree2 = TreeNode(1)
        expected = [duplicate_tree1, duplicate_tree2]
        self.assert_result(expected, find_duplicate_subtree(tree))

    def test_all_value_are_equal2(self):
        """
           1
          / \
         1   1
              \
               1
              /
             1
        """
        right_tree = TreeNode(1, right=TreeNode(1, TreeNode(1)))
        tree = TreeNode(1, TreeNode(1), right_tree)

        duplicate_tree = TreeNode(1)
        expected = [duplicate_tree]
        self.assert_result(expected, find_duplicate_subtree(tree))
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 29, 2020 \[Hard\] Largest Rectangle
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

**Solution with DP:** [https://repl.it/@trsong/Largest-Rectangle-in-a-Grid](https://repl.it/@trsong/Largest-Rectangle-in-a-Grid)
```py
import unittest

def largest_rectangle(table):
    if not table or not table[0]: return 0
    heights = [0] * len(table[0])
    max_area = 0
    for row in table:
        for i, cell in enumerate(row):
            # calculate the histogram of each row since last saw 1 at same column
            if cell == 1:
                heights[i] += 1
            else:
                heights[i] = 0
        max_area = max(max_area, largest_rectangle_in_histogram(heights))
    return max_area


def largest_rectangle_in_histogram(histogram):
    stack = []
    i = 0
    max_area = 0
    while i < len(histogram) or stack:
        if not stack or i < len(histogram) and histogram[
                stack[-1]] <= histogram[i]:
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

### Oct 28, 2020 \[Hard\] Exclusive Product
---
> **Question:**  Given an array of integers, return a new array such that each element at index i of the new array is the product of all the numbers in the original array except the one at i.
>
> For example, if our input was [1, 2, 3, 4, 5], the expected output would be [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output would be [2, 3, 6].
>
> Follow-up: what if you can't use division?

**Solution:** [https://repl.it/@trsong/Calculate-Exclusive-Product](https://repl.it/@trsong/Calculate-Exclusive-Product)
```py
import unittest

def exclusive_product(nums):
    n = len(nums)
    left_product = right_product = 1
    res = [1] * n
    for i in xrange(n):
        res[i] *= left_product
        res[n-1-i] *= right_product
        left_product *= nums[i]
        right_product *= nums[n-1-i]
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
    unittest.main(exit=False)
```

### Oct 27, 2020 \[Medium\] Interleave Stacks
---
> **Question:** Given a stack of N elements, interleave the first half of the stack with the second half reversed using only one other queue. This should be done in-place.
>
> Recall that you can only push or pop from a stack, and enqueue or dequeue from a queue.
>
> For example, if the stack is `[1, 2, 3, 4, 5]`, it should become `[1, 5, 2, 4, 3]`. If the stack is `[1, 2, 3, 4]`, it should become `[1, 4, 2, 3]`.
>
> Hint: Try working backwards from the end state.

**Solution:** [https://repl.it/@trsong/Interleave-First-and-Second-Half-of-Stacks](https://repl.it/@trsong/Interleave-First-and-Second-Half-of-Stacks)
```py
import unittest
from Queue import deque

def interleave_stack(stack):
    # Case 1: when stack size is even: 1 2 3 4 5 6.
    # Work backwards:
    # [5] stack: 1 6 2 5 3 4  queue: 
    # [4] stack:              queue: 1 6 2 5 3 4
    # [3] stack: 3 2 1        queue: 6 5 4
    # [2] stack:              queue: 3 2 1 6 5 4
    # [1] stack:              queue: 6 5 4 3 2 1
    # [0] stack: 1 2 3 4 5 6  queue: 
    #
    # Case 2: when stack size is odd: 1 2 3 4 5
    # Work backwards:
    # [5] stack: 1 5 2 4 3  queue:
    # [4] stack:            queue: 1 5 2 4    Last: 3
    # [3] stack: 2 1        queue: 5 4        Last: 3
    # [2] stack:            queue: 3 2 1 5 4
    # [1] stack:            queue: 5 4 3 2 1
    # [0] stack: 1 2 3 4 5  queue: 

    n = len(stack)
    half_n = n // 2
    queue = deque()
    # Step 1:
    while stack:
        queue.append(stack.pop())
    
    # Step 2:
    for _ in xrange(half_n):
        queue.append(queue.popleft())
    
    last = queue.popleft() if n % 2 == 1 else None

    # Step 3:
    for _ in xrange(half_n):
        stack.append(queue.popleft())

    # Step 4
    for _ in xrange(half_n):
        queue.append(stack.pop())
        queue.append(queue.popleft())

    # Step 5
    while queue:
        stack.append(queue.popleft())

    if last is not None:
        stack.append(last)
    

class InterleaveStackSpec(unittest.TestCase):
    def test_example1(self):
        stack = [1, 2, 3, 4, 5]
        expected = [1, 5, 2, 4, 3]
        interleave_stack(stack)
        self.assertEqual(stack, expected)

    def test_example2(self):
        stack = [1, 2, 3, 4]
        expected = [1, 4, 2, 3]
        interleave_stack(stack)
        self.assertEqual(stack, expected)

    def test_empty_stack(self):
        stack = []
        expected = []
        interleave_stack(stack)
        self.assertEqual(stack, expected)

    def test_size_one_stack(self):
        stack = [1]
        expected = [1]
        interleave_stack(stack)
        self.assertEqual(stack, expected)

    def test_size_two_stack(self):
        stack = [1, 2]
        expected = [1, 2]
        interleave_stack(stack)
        self.assertEqual(stack, expected)

    def test_size_three_stack(self):
        stack = [1, 2, 3]
        expected = [1, 3, 2]
        interleave_stack(stack)
        self.assertEqual(stack, expected)

    def test_size_seven_stack(self):
        stack = [1, 2, 3, 4, 5, 6, 7]
        expected = [1, 7, 2, 6, 3, 5, 4]
        interleave_stack(stack)
        self.assertEqual(stack, expected)
    

if __name__ == '__main__':
    unittest.main(exit=False)
```


### Oct 26, 2020 \[Easy\] Add Digits
--- 
> **Question:** Given a number, add the digits repeatedly until you get a single number.

**Example:**
```py
Input: 159
Output: 6
Explanation:
1 + 5 + 9 = 15.
1 + 5 = 6.
So the answer is 6.
```

**Solution:** [https://repl.it/@trsong/Add-Digits](https://repl.it/@trsong/Add-Digits)
```py
import unittest

def add_digits(num):
    accu = 0
    while num + accu >= 10:
        if num == 0:
            num = accu
            accu = 0
        else:
            accu += num % 10
            num //= 10
    return num + accu


class AddDigitSpec(unittest.TestCase):
    def test_example(self):
        num = 159
        expected = 6  # 159 -> 15 -> 6
        self.assertEqual(expected, add_digits(num))

    def test_zero(self):
        num = 0
        expected = 0
        self.assertEqual(expected, add_digits(num))

    def test_no_carryover_during_calculation(self):
        num = 1024
        expected = 7
        self.assertEqual(expected, add_digits(num))

    def test_with_carryover_during_calculation(self):
        num = 199199
        expected = 2  # 199199 -> 38 -> 11 -> 2
        self.assertEqual(expected, add_digits(num))

    def test_with_carryover_during_calculation2(self):
        num = 10987654321
        expected = 1  # 10987654321 -> 55 -> 10 -> 1
        self.assertEqual(expected, add_digits(num))

    def test_with_carryover_during_calculation3(self):
        num = 1234
        expected = 1  # 1234 -> 10 -> 1
        self.assertEqual(expected, add_digits(num))

    def test_with_carryover_during_calculation4(self):
        num = 5674
        expected = 4  # 5674 -> 22 -> 4
        self.assertEqual(expected, add_digits(num))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 25, 2020 \[Hard\] Order of Alien Dictionary
--- 
> **Question:** You come across a dictionary of sorted words in a language you've never seen before. Write a program that returns the correct order of letters in this language.
>
> For example, given `['xww', 'wxyz', 'wxyw', 'ywx', 'ywz']`, you should return `['x', 'z', 'w', 'y']`.

**My thoughts:** As the alien letters are topologically sorted, we can just mimic what topological sort with numbers and try to find pattern.

Suppose the dictionary contains: `01234`. Then the words can be `023, 024, 12, 133, 2433`. Notice that we can only find the relative order by finding first unequal letters between consecutive words. eg.  `023, 024 => 3 < 4`.  `024, 12 => 0 < 1`.  `12, 133 => 2 < 3`

With relative relation, we can build a graph with each occurring letters being veteces and edge `(u, v)` represents `u < v`. If there exists a loop that means we have something like `a < b < c < a` and total order not exists. Otherwise we preform a topological sort to generate the total order which reveals the alien dictionary. 

As for implementation of topological sort, there are two ways, one is the following by constantly removing edges from visited nodes. The other is to [first DFS to find the reverse topological order then reverse again to find the result](https://trsong.github.io/python/java/2019/11/02/DailyQuestionsNov.html#nov-9-2019-hard-order-of-alien-dictionary). 


**Solution with Topological Sort (Remove Edge)::** [https://repl.it/@trsong/Alien-Dictionary-Order](https://repl.it/@trsong/Alien-Dictionary-Order)
```py
import unittest
from Queue import deque
from collections import defaultdict

def dictionary_order(sorted_words):
    neighbors = defaultdict(list)
    in_degree = defaultdict(int)
    for i in xrange(1, len(sorted_words)):
        small_word = sorted_words[i-1]
        large_word = sorted_words[i]
        for small_ch, large_ch in zip(small_word, large_word):
            if small_ch != large_ch:
                neighbors[small_ch].append(large_ch)
                in_degree[large_ch] += 1
                break
                    
    char_set = set()
    queue = deque()
    for word in sorted_words:
        for ch in word:
            if ch in char_set:
                continue
            char_set.add(ch)
            if ch not in in_degree:
                queue.append(ch)
    
    top_order = []
    while queue:
        small_ch = queue.popleft()
        top_order.append(small_ch)
        for large_ch in neighbors[small_ch]:
            in_degree[large_ch] -= 1
            if in_degree[large_ch] == 0:
                queue.append(large_ch)

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
    unittest.main(exit=False)
```


### Oct 24, 2020 \[Hard\] First Unique Character from a Stream
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

**Solution** [https://repl.it/@trsong/First-Unique-Character-from-a-Stream](https://repl.it/@trsong/First-Unique-Character-from-a-Stream)
```py
import unittest

def find_first_unique_letter(char_stream):
    head = Node()
    tail = Node(prev=head)
    head.next = tail
    duplicated = set()
    lookup = {}

    for ch in char_stream:
        if ch in lookup:
            node = lookup[ch]
            node.prev.next, node.next.prev = node.next, node.prev
            duplicated.add(node.val)
            del lookup[ch]
        elif ch not in duplicated:
            tail.val = ch
            tail.next = Node(prev=tail)
            lookup[ch] = tail
            tail = tail.next
        yield head.next.val if head.next != tail else None


class Node(object):
    def __init__(self, val=None, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next


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
    unittest.main(exit=False)
```

### Oct 23, 2020 \[Easy\] Floor and Ceiling of BST
---
> **Question:** Given an integer `k` and a binary search tree, find the `floor` (less than or equal to) of `k`, and the `ceiling` (larger than or equal to) of `k`. If either does not exist, then print them as None.

**Example:**
```py
          8
        /   \    
      4      12
    /  \    /  \
   2    6  10   14

k: 11  Floor: 10  Ceil: 12
k: 1   Floor: None  Ceil: 2
k: 6   Floor: 6   Ceil: 6
k: 15  Floor: 14  Ceil: None
```

**Solution:** [https://repl.it/@trsong/Floor-and-Ceiling-of-BST](https://repl.it/@trsong/Floor-and-Ceiling-of-BST)
```py
import unittest

def find_floor_ceiling(k, bst):
    floor = ceiling = None
    while bst:
        if bst.val == k:
            return [k, k]
        elif bst.val < k:
            floor = bst.val
            bst = bst.right
        else:
            ceiling = bst.val
            bst = bst.left

    return [floor, ceiling]
   

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class FindFloorCeilingSpec(unittest.TestCase):
    def test_example(self):
        """
             8
           /   \
          4     12
         / \   /  \
        2   6 10  14
        """
        left_tree = TreeNode(4, TreeNode(2), TreeNode(6))
        right_tree = TreeNode(12, TreeNode(10), TreeNode(14))
        root = TreeNode(8, left_tree, right_tree)
        self.assertEqual([10, 12], find_floor_ceiling(11, root))
        self.assertEqual([None, 2], find_floor_ceiling(1, root))
        self.assertEqual([6, 6], find_floor_ceiling(6, root))
        self.assertEqual([14, None], find_floor_ceiling(15, root))

    def test_empty_tree(self):
        self.assertEqual([None, None], find_floor_ceiling(42, None))

    def test_one_node_tree(self):
        self.assertEqual([None, 42], find_floor_ceiling(41, TreeNode(42)))
        self.assertEqual([42, 42], find_floor_ceiling(42, TreeNode(42)))
        self.assertEqual([42, None], find_floor_ceiling(43, TreeNode(42)))

    def test_floor_ceiling_between_root_and_leaf(self):
        """
           1
            \
             5
            /
           4
          /
         3
        """
        right_tree = TreeNode(5, TreeNode(4, TreeNode(3)))
        root = TreeNode(1, right=right_tree)
        self.assertEqual([1, 3], find_floor_ceiling(2, root))

    def test_floor_ceiling_between_root_and_leaf2(self):
        """
          4
         /
        1
         \
          2
        """
        root = TreeNode(4, TreeNode(1, right=TreeNode(2)))
        self.assertEqual([2, 4], find_floor_ceiling(3, root))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 22, 2020 \[Medium\] Find Pythagorean Triplets
---
> **Question:** Given a list of numbers, find if there exists a pythagorean triplet in that list. A pythagorean triplet is `3` variables `a`, `b`, `c` where `a * a + b * b = c * c`.

**Example:**
```py
Input: [3, 5, 12, 5, 13]
Output: True
Here, 5 * 5 + 12 * 12 = 13 * 13.
```

**Solution with Two Pointers:** [https://repl.it/@trsong/Find-Pythagorean-Triplets](https://repl.it/@trsong/Find-Pythagorean-Triplets)
```py
import unittest

def contains_triplet(nums):
    sqr_nums = map(lambda x: x * x, nums)
    sqr_nums.sort()

    for index in xrange(len(sqr_nums) - 1, 1, -1):
        target = sqr_nums[index]
        if exists_two_sums(sqr_nums, 0, index-1, target):
            return True

    return False


def exists_two_sums(nums, start, end, target):
    while start < end:
        total = nums[start] + nums[end]
        if total == target:
            return True
        elif total < target:
            start += 1
        else:
            end -= 1

    return False


class ContainsTripletSpec(unittest.TestCase):
    def test_empty_array(self):
        self.assertFalse(contains_triplet([]))

    def test_simple_array_with_triplet(self):
        # 3, 4, 5
        self.assertTrue(contains_triplet([3, 1, 4, 6, 5]))

    def test_simple_array_with_triplet2(self):
        # 5, 12, 13
        self.assertTrue(contains_triplet([5, 7, 8, 12, 13]))

    def test_simple_array_with_triplet3(self):
        # 9, 12, 15
        self.assertTrue(contains_triplet([9, 12, 15, 4, 5]))

    def test_complicated_array_with_triplet(self):
        # 28, 45, 53
        self.assertTrue(contains_triplet([25, 28, 32, 45, 47, 48, 50, 53, 55, 60])) 

    def test_array_without_triplet(self):
        self.assertFalse(contains_triplet([10, 4, 6, 12, 5]))

    def test_array_with_duplicated_numbers(self):
        self.assertFalse(contains_triplet([0, 0]))
    
    def test_array_with_duplicated_numbers2(self):
        self.assertTrue(contains_triplet([0, 0, 0]))
    
    def test_array_with_duplicated_numbers3(self):
        self.assertTrue(contains_triplet([1, 1, 0]))
    
    def test_array_with_negative_numbers(self):
        self.assertTrue(contains_triplet([-3, -5, -4]))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Oct 21, 2020 LT 640 \[Medium\] One Edit Distance
---
> **Question:** Given two strings S and T, determine if they are both one edit distance apart.

**Example 1:**
```
Input: s = "aDb", t = "adb" 
Output: True
```

**Example 2:**
```
Input: s = "ab", t = "ab" 
Output: False
Explanation:
s=t, so they aren't one edit distance apart
```

**My thougths:** The trick for this problem is that one insertion of shorter string is equivalent to one removal of longer string. And if both string of same length, then we only allow one replacement. The other cases are treated as more than one edit distance between two strings.

**Solution:** [https://repl.it/@trsong/Is-One-Edit-Distance](https://repl.it/@trsong/Is-One-Edit-Distance)
```py
import unittest

def is_one_edit_distance_between(s, t):
    if len(s) > len(t):
        s, t = t, s

    len1, len2 = len(s), len(t)
    if len2 - len1 > 1:
        return False
    
    edit_dist = 0
    i = 0
    for j in xrange(len2):
        if edit_dist > 1:
            break
        
        if i >= len1 or s[i] != t[j]:
            edit_dist += 1
            if len1 < len2:
                continue
        i += 1
    return edit_dist == 1


class IsOneEditDistanceBetweenSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(is_one_edit_distance_between('aDb', 'adb'))

    def test_example2(self):
        self.assertFalse(is_one_edit_distance_between('ab', 'ab'))

    def test_empty_string(self):
        self.assertFalse(is_one_edit_distance_between('', ''))
        self.assertTrue(is_one_edit_distance_between('', 'a'))
        self.assertFalse(is_one_edit_distance_between('', 'ab'))

    def test_one_insert_between_two_strings(self):
        self.assertTrue(is_one_edit_distance_between('abc', 'ac'))
        self.assertFalse(is_one_edit_distance_between('abcd', 'ad'))

    def test_one_remove_between_two_strings(self):
        self.assertTrue(is_one_edit_distance_between('abcd', 'abd'))
        self.assertFalse(is_one_edit_distance_between('abcd', 'cd'))

    def test_one_replace_between_two_string(self):
        self.assertTrue(is_one_edit_distance_between('abc', 'abd'))
        self.assertFalse(is_one_edit_distance_between('abc', 'ddc'))

    def test_length_difference_greater_than_one(self):
        self.assertFalse(is_one_edit_distance_between('abcd', 'abcdef'))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 20, 2020 \[Easy\] Sorted Square of Integers
---
> **Question:** Given a sorted list of integers, square the elements and give the output in sorted order.
>
> For example, given `[-9, -2, 0, 2, 3]`, return `[0, 4, 4, 9, 81]`.
>
> Additonal Requirement: Do it in-place. i.e. Space Complexity O(1).  


**Solution:** [https://repl.it/@trsong/Calculate-Sorted-Square-of-Integers](https://repl.it/@trsong/Calculate-Sorted-Square-of-Integers)
```py
import unittest

def sorted_square(nums):
    non_neg_pos = find_non_neg_pos(nums)
    swap_between(nums, 0, non_neg_pos - 1)
    map_square(nums)
    merge_inplace(nums, 0, non_neg_pos)
    return nums


def find_non_neg_pos(nums):
    lo = 0
    hi = len(nums)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < 0:
            lo = mid + 1
        else:
            hi = mid
    return lo


def swap_between(nums, i, j):
    while i < j:
        nums[i], nums[j] = nums[j], nums[i]
        i += 1
        j -= 1


def map_square(nums):
    for i in xrange(len(nums)):
        nums[i] *= nums[i] 


def merge_inplace(nums, i, j):
    n = len(nums)
    while i < j < n:
        if nums[i] < nums[j]:
            i += 1
        else:
            nums[i], nums[i+1:j+1] = nums[j], nums[i:j]
            i += 1
            j += 1


class SortedSquareSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual([0, 4, 4, 9, 81], sorted_square([-9, -2, 0, 2, 3]))

    def test_array_with_duplicate_elements(self):
        self.assertEqual([0, 0, 0, 1, 1, 1, 1, 1, 1], sorted_square([-1, -1, -1, 0, 0, 0, 1, 1, 1]))

    def test_array_with_all_negative_elements(self):
        self.assertEqual([1, 4, 9], sorted_square([-3, -2, -1]))

    def test_array_with_positive_elements(self):
        self.assertEqual([1, 4, 9], sorted_square([1, 2, 3]), )

    def test_array_with_positive_elements2(self):
        self.assertEqual([1, 4, 9, 36, 49, 81], sorted_square([-7, -6, 1, 2, 3, 9]))    


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Oct 19, 2020 LC 224 \[Hard\] Expression Evaluation
---
> **Questions:** Given a string consisting of parentheses, single digits, and positive and negative signs, convert the string into a mathematical expression to obtain the answer.
>
> Don't use eval or a similar built-in parser.

**Example 1:**
```py
Input: - ( 3 + ( 2 - 1 ) )
Output: -4
```

**Example 2:**
```py
Input: -1 + (2 + 3)
Output: 4
```

**My thoughts:** Treat `(expr)` as a single number which later on can be solved by recursion. 

Now we can deal with expresion without parentheses:

A complicated expression can be broken into multiple normal terms. `Expr = term1 + term2 - term3 ...`. Between each consecutive term we only allow `+` and `-`. Whereas within each term we only allow `*` and `/`. So we will have the following definition of an expression. e.g. `1 + 2 - 1*2*1 - 3/4*4 + 5*6 - 7*8 + 9/10 = (1) + (2) - (1*2*1) - (3/4*4) + (5*6) - (7*8) + (9/10)` 

```py
Expression is one of the following:
- Empty or 0
- Term - Expression
- Term + Expression

Term is one of the following:
- 1
- A number * Term
- A number / Term
```

Thus, we can comupte each term value and sum them together.

**Solution:** [https://repl.it/@trsong/Math-Expression-Evaluation](https://repl.it/@trsong/Math-Expression-Evaluation)
```py
import unittest

def evaluate_expression(expr):
    if not expr:
        return 0

    total_sum = term_sum = num = 0
    op = '+'
    op_set = {'+', '-', '*', '/'}
    index = 0
    n = len(expr)
    
    while index < n:
        char = expr[index]
        if char == ' ' and index < n - 1:
            index += 1
            continue
        elif char.isdigit():
            num = num * 10 + int(char)
        elif char == '(':
            left_parenthsis_index = index
            balance = 0
            while index < n:
                if expr[index] == '(':
                    balance += 1
                elif expr[index] == ')':
                    balance -= 1
                if balance == 0:
                    break
                index += 1
            num = evaluate_expression(expr[left_parenthsis_index+1:index])
        if char in op_set or index == n - 1:
            if op == '+':
                total_sum += term_sum
                term_sum = num
            elif op == '-':
                total_sum += term_sum
                term_sum = -num
            elif op == '*':
                term_sum *= num
            elif op == '/':
                sign = 1 if term_sum > 0 else -1
                term_sum = abs(term_sum) / num * sign
            op = char
            num = 0
        index += 1
    total_sum += term_sum
    return total_sum
                

class EvaluateExpressionSpec(unittest.TestCase):
    def example(self):
        self.assertEqual(4, evaluate_expression("-1 + (2 + 3)"))

    def test_empty_string(self):
        self.assertEqual(0, evaluate_expression(""))

    def test_basic_expression1(self):
        self.assertEqual(7, evaluate_expression("3+2*2"))

    def test_basic_expression2(self):
        self.assertEqual(1, evaluate_expression(" 3/2 "))

    def test_basic_expression3(self):
        self.assertEqual(5, evaluate_expression(" 3+5 / 2 "))
    
    def test_basic_expression4(self):
        self.assertEqual(-24, evaluate_expression("1*2-3/4+5*6-7*8+9/10"))

    def test_basic_expression5(self):
        self.assertEqual(10000, evaluate_expression("10000-1000/10+100*1"))

    def test_basic_expression6(self):
        self.assertEqual(13, evaluate_expression("14-3/2"))

    def test_negative(self):
        self.assertEqual(-1, evaluate_expression(" -7 / 4 "))

    def test_minus(self):
        self.assertEqual(-5, evaluate_expression("-2-3"))

    def test_expression_with_parentheses(self):
        self.assertEqual(42, evaluate_expression("  -(-42)"))
    
    def test_expression_with_parentheses2(self):
        self.assertEqual(3, evaluate_expression("(-1 + 2) * 3"))

    def test_complicated_operations(self):
        self.assertEqual(2, evaluate_expression("-2 - (-2) * ( ((-2) - 3) * 2 + (-3) * (-4))"))

    def test_complicated_operations2(self):
        self.assertEqual(-2600, evaluate_expression("-3*(10000-1000)/10-100*(-1)"))

    def test_complicated_operations3(self):
        self.assertEqual(100, evaluate_expression("100 * ( 2 + 12 ) / 14"))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 18, 2020 LC 130 \[Medium\] Surrounded Regions
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

**My thoughts** The way we use Union-Find to conquer this question looks like the following:

* Step 1: Init the parent arry:
Orginal Grid:
  ```py
  [
      ['X', 'O', 'X', 'X', 'O'],
      ['X', 'X', 'O', 'O', 'X'],
      ['X', 'O', 'X', 'X', 'O'],
      ['O', 'O', 'O', 'X', 'O']
  ]
  ```

  parent array
  ```py
  [
      [-1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1]
      [-1]  <----------------- we also create a secret cell for future use
  ]
  ```
* Step 2: Now connect all connected 'O' use Union-Find
  
  parent array: instead of using parent index, I use letter to represent different connected region
  ```py
  [
      [-1,  A, -1, -1,  B],
      [-1, -1,  C,  C, -1],
      [-1,  E, -1, -1,  D],
      [ E,  E,  E, -1,  D]
      [-1]  <----------------- In the next step we will take advantage of this cell
  ]
  ```

* Step 3: Let's connect the secret cell to all edge-connected 'O' cells 

 parent array: instead of using parent index, I use letter to represent different connected region
  ```py
  [
      [-1,  O, -1, -1,  O],
      [-1, -1,  C,  C, -1],
      [-1,  O, -1, -1,  O],
      [ O,  O,  O, -1,  O]
      [ O]  <----------------- this cell is used to connect to all 'O' on the edge of the grid
  ]
  ```

* Step 4: Replace all 'O' with 'X', except for connected-to-secret-spot ones

  ```py
  [
      ['X', 'O', 'X', 'X', 'O'],
      ['X', 'X', 'X', 'X', 'X'],
      ['X', 'O', 'X', 'X', 'O'],
      ['O', 'O', 'O', 'X', 'O']
  ]
  ``` 
  
Note for my implementation of union-find, I flatten the parent 2D array into 1D array.


**Solution with DisjointSet(Union-Find):** [https://repl.it/@trsong/Find-Surrounded-Regions](https://repl.it/@trsong/Find-Surrounded-Regions)
```py
import unittest

X, O = 'X', 'O'

def flip_region(grid):
    if not grid or not grid[0]:
        return grid
    n, m = len(grid), len(grid[0])
    uf = DisjointSet(n * m + 1)
    cell_to_pos = lambda r, c: r * m + c
    boundary_cell = n * m

    for r in xrange(n):
        for c in xrange(m):
            if r > 0 and grid[r-1][c] == grid[r][c]:
                uf.union(cell_to_pos(r-1, c), cell_to_pos(r, c))
            if c > 0 and grid[r][c-1] == grid[r][c]:
                uf.union(cell_to_pos(r, c-1), cell_to_pos(r, c))

    for r in xrange(n):
        uf.union(cell_to_pos(r, 0), boundary_cell)
        uf.union(cell_to_pos(r, m-1), boundary_cell)
    
    for c in xrange(m):
        uf.union(cell_to_pos(0, c), boundary_cell)
        uf.union(cell_to_pos(n-1, c), boundary_cell)

    for r in xrange(n):
        for c in xrange(m):
            if grid[r][c] == O and not uf.is_connected(cell_to_pos(r, c), boundary_cell):
                grid[r][c] = X

    return grid


class DisjointSet(object):
    def __init__(self, size):
        self.parent = range(size)
    
    def find(self, pos):
        while self.parent[pos] != pos:
            self.parent[pos] = self.parent[self.parent[pos]]
            pos = self.parent[pos]
        return pos

    def union(self, pos1, pos2):
        root1 = self.find(pos1)
        root2 = self.find(pos2)
        if root1 != root2:
            self.parent[root1] = root2

    def is_connected(self, pos1, pos2):
        return self.find(pos1) == self.find(pos2)


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
    unittest.main(exit=False)
```


### Oct 17, 2020 \[Hard\] Largest Sum of Non-adjacent Numbers
---

> **Question:** Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0 or negative.
>
> For example, `[2, 4, 6, 2, 5]` should return 13, since we pick 2, 6, and 5. `[5, 1, 1, 5]` should return 10, since we pick 5 and 5.
>
> Follow-up: Can you do this in O(N) time and constant space?

**My thoughts:** Try to find the pattern by examples:

Let keep it simple first, just assume all elements are positive:
```py
[] => 0  # Sum of none equals 0
[2] => 2  # the only elem is 2
[2, 4] => 4  # 2 and 4 we will choose 4
[2, 4, 6] => 2+6 => 8  # The pattern is eithe choose all odd index numbers or all even index numbers. eg.  [_, 4, _] vs [2, _, 6]. We choose 2,6 which gives 8. If it's [1, 99, 1], we will choose 99 instead.
[2, 4, 6, 2] => [_, 4, _, 2] vs [2, _, 6, _] => 6 vs 8 => 8

# If we let f[lst] represents the max non-adjacent sum when input array is lst
f[2, 4, 6] = max { f[2] + 6, f[4] } = max{6, 8} = 8
f[2, 4, 6, 2] = max { f[2, 4] + 2, f[2, 4, 6] } = max{4, 8} = 8
f[2, 4, 6, 2, 5] = max {f[2, 4, 6] + 5, f[2, 4, 6, 2]} = max{8 + 5, 8} = 13

f[a1, a2, ...., an] = max {f[a1, ..., an-2] + an, f[a1, a2, ..., an-1]}
```
What if one of the element is negative? i.e. `[-1, -1, 1]`

```py
# if we continue apply the formula above, we will end up minus a number from max sum
f[-1, -1, 1] = max{f[-1] + 1, f[-1, -1]} = max{-1+1, -1} = 0

# we can choose to not include that number
#f[-1, -1, 1] should equal to 1
f[-1, -1, 1] = max{f[-1] + 1, f[-1, -1], 1} = max{-1+1, -1, 1} = 1

# As all previous sum chould be a negative number, if that's the case, we can just include the positive number.
f[a1, a2, ...., an] = max {f[a1, ..., an-2] + an, f[a1, a2, ..., an-1], an}

# If we let dp[n] represents the max non-adjacent sum when input size is n, then 
dp[i] = max(dp[i-1], dp[i-2] + numbers[i-1], numbers[i-1])

# dp[n] will be the answer to f[a0, a2, ...., an-1]
```

**Solution with DP:** [https://repl.it/@trsong/Find-Largest-Sum-of-Non-adjacent-Numbers](https://repl.it/@trsong/Find-Largest-Sum-of-Non-adjacent-Numbers)
```py
import unittest

def max_non_adj_sum(nums):
    # Let dp[i] represents max non-adj sum for num[:i]
    # dp[i] = max(dp[i-1], nums[i-1] + dp[i-2])
    # That is max_so_far = max(prev_max, prev_prev_max + cur)
    max_so_far = prev_max = prev_prev_max = 0
    for num in nums:
        max_so_far = max(prev_max, prev_prev_max + num)
        prev_prev_max = prev_max
        prev_max = max_so_far
    return max_so_far


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
    unittest.main(exit=False)
```

### Oct 16, 2020 \[Medium\] Maximum In A Stack
---
> **Question:** Implement a class for a stack that supports all the regular functions (`push`, `pop`) and an additional function of `max()` which returns the maximum element in the stack (return None if the stack is empty). Each method should run in constant time.

**Example**:
```py
s = MaxStack()
s.push(1)
s.push(2)
s.push(3)
s.push(2)
print s.max()  # 3
s.pop()
s.pop()
print s.max()  # 2
```

**Solution:** [https://repl.it/@trsong/Maximum-In-A-Stack](https://repl.it/@trsong/Maximum-In-A-Stack)
```py
import unittest


class MaxStack(object):
    def __init__(self):
        self.val_stack = []
        self.max_stack = []

    def push(self, v):
        self.val_stack.append(v)
        local_max = max(v, self.max_stack[-1]) if self.max_stack else v
        self.max_stack.append(local_max)

    def pop(self):
        self.max_stack.pop()
        return self.val_stack.pop()

    def max(self):
        return self.max_stack[-1]


class MaxStackSpec(unittest.TestCase):
    def test_example(self):
        s = MaxStack()
        s.push(1)
        s.push(2)
        s.push(3)
        s.push(2)
        self.assertEqual(3, s.max())
        self.assertEqual(2, s.pop())
        self.assertEqual(3, s.pop())
        self.assertEqual(2, s.max())

    def test_ascending_stack(self):
        s = MaxStack()
        s.push(1)
        s.push(2)
        self.assertEqual(2, s.max())
        s.push(3)
        self.assertEqual(3, s.max())
        self.assertEqual(3, s.pop())
        self.assertEqual(2, s.max())
        s.push(4)
        self.assertEqual(4, s.pop())

    def test_descending_stack(self):
        s = MaxStack()
        s.push(4)
        self.assertEqual(4, s.pop())
        s.push(3)
        s.push(2)
        s.push(1)
        self.assertEqual(3, s.max())

    def test_up_down_up_stack(self):
        s = MaxStack()
        s.push(1)
        s.push(3)
        s.push(5)
        s.push(2)
        s.push(6)
        self.assertEqual(6, s.max())
        self.assertEqual(6, s.pop())
        self.assertEqual(5, s.max())
        self.assertEqual(2, s.pop())
        self.assertEqual(5, s.max())
        self.assertEqual(5, s.pop())
        self.assertEqual(3, s.max())
        self.assertEqual(3, s.pop())


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 15, 2020 \[Medium\] Delete Columns to Make Sorted II
---
> **Question:** You are given an N by M 2D matrix of lowercase letters. The task is to count the number of columns to be deleted so that all the rows are lexicographically sorted.

**Example 1:**
```
Given the following table:
hello
geeks

Your function should return 1 as deleting column 1 (index 0)
Now both strings are sorted in lexicographical order:
ello
eeks
```

**Example 2:**
```
Given the following table:
xyz
lmn
pqr

Your function should return 0. All rows are already sorted lexicographically.
```

**My thoughts:** This problem feels like 2D version of ***The Longest Increasing Subsequence Problem*** (LIP). The LIP says find longest increasing subsequence. e.g. `01212342345` gives `01[2]1234[23]45`, with `2,2,3` removed. So if we only have 1 row, we can simply find longest increasing subsequence and use that to calculate how many columns to remove i.e. `# of columns to remove = m - LIP`. Similarly, for n by m table, we can first find longest increasing sub-columns and use that to calculate which columns to remove. Which can be done using DP:

let `dp[i]` represents max number of columns to keep at ends at column i. 
- `dp[i] = max(dp[j]) + 1 where j < i` if all characters in column `i` have lexicographical order larger than column `j`
- `dp[0] = 1`

**Solution with DP:** [https://repl.it/@trsong/Delete-Columns-to-Make-Row-Sorted-II](https://repl.it/@trsong/Delete-Columns-to-Make-Row-Sorted-II)
```py
import unittest

def delete_column(table):
    """
    let dp[i] represents max number of columns to keep at ends at column i
    i.e. column i is the last column to keep
    dp[i] = max(dp[j]) + 1 where j < i  if all e in column i have lexicographical order larger than column j
    """
    if not table:
        return 0
    m = len(table[0])
    dp = [1] * m
    for i in xrange(1, m):
        for j in xrange(i):
            if all(row[j] <= row[i] for row in table):
                dp[i] = max(dp[i], dp[j] + 1)
    return m - max(dp)


class DeleteColumnSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(1, delete_column([
            'hello',
            'geeks'
        ]))

    def test_example2(self):
        self.assertEqual(0, delete_column([
            'xyz',
            'lmn',
            'pqr'
        ]))

    def test_table_with_no_row(self):
        self.assertEqual(0, delete_column([]))

    def test_table_with_one_row(self):
        self.assertEqual(3, delete_column([
            '01212342345' # 01[2]1234[23]45   
        ]))  

    def test_table_with_two_rows(self):
        self.assertEqual(2, delete_column([
            '01012',  # [0] 1 [0] 1 2
            '20101'   # [2] 0 [1] 0 1
        ]))  


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Oct 14, 2020 \[Easy\] Delete Columns to Make Sorted I
---
> **Question:** You are given an N by M 2D matrix of lowercase letters. Determine the minimum number of columns that can be removed to ensure that each row is ordered from top to bottom lexicographically. That is, the letter at each column is lexicographically later as you go down each row. It does not matter whether each row itself is ordered lexicographically.

**Example 1:**
```py
Given the following table:
cba
daf
ghi

This is not ordered because of the a in the center. We can remove the second column to make it ordered:
ca
df
gi

So your function should return 1, since we only needed to remove 1 column.
```

**Example 2:**
```py
Given the following table:
abcdef

Your function should return 0, since the rows are already ordered (there's only one row).
```

**Example 3:**
```py
Given the following table:
zyx
wvu
tsr

Your function should return 3, since we would need to remove all the columns to order it.
```

**Solution:** [https://repl.it/@trsong/Delete-Columns-From-Table-to-Make-Sorted-I](https://repl.it/@trsong/Delete-Columns-From-Table-to-Make-Sorted-I)
```py
import unittest

def columns_to_delete(table):
    if not table or not table[0]:
        return 0

    n, m = len(table), len(table[0])
    count = 0
    for c in xrange(m):
        for r in xrange(1, n):
            if table[r-1][c] > table[r][c]:
                count += 1
                break
    return count


class ColumnToDeleteSpec(unittest.TestCase):
    def test_empty_table(self):
        self.assertEqual(columns_to_delete([]), 0)
        self.assertEqual(columns_to_delete([""]), 0)

    def test_example1(self):
        self.assertEqual(columns_to_delete([
            'cba',
            'daf',
            'ghi'
        ]), 1)

    def test_example2(self):
        self.assertEqual(columns_to_delete([
            'abcdef'
        ]), 0)

    def test_example3(self):
        self.assertEqual(columns_to_delete([
            'zyx',
            'wvu',
            'tsr'
        ]), 3)


if __name__ == '__main__':
    unittest.main(exit=False) 
```


### Oct 13, 2020 \[Medium\] Locking in Binary Tree
---
> **Question:** Implement locking in a binary tree. A binary tree node can be locked or unlocked only if all of its descendants or ancestors are not locked.
> 
> Design a binary tree node class with the following methods:
>
> - `is_locked`, which returns whether the node is locked
> - `lock`, which attempts to lock the node. If it cannot be locked, then it should return false. Otherwise, it should lock it and return true.
> - `unlock`, which unlocks the node. If it cannot be unlocked, then it should return false. Otherwise, it should unlock it and return true.
>
> You may augment the node to add parent pointers or any other property you would like. You may assume the class is used in a single-threaded program, so there is no need for actual locks or mutexes. Each method should run in O(h), where h is the height of the tree.

**My thoughts:** Whether we can successfully lock or unlock a binary tree node depends on if there exist a locked node above or below. So for each node there is a reference to parent and a counter which stores the number of locked node below. Doing such can allow running time of `lock()` and `unlock()` to be `O(log n)`.

**Solution:** [https://repl.it/@trsong/Locking-and-Unlocking-in-Binary-Tree](https://repl.it/@trsong/Locking-and-Unlocking-in-Binary-Tree)
```py
import unittest

class TreeNode(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
        self.parent = None
        self.locked = False
        self.num_lock_below = 0

        if self.left:
            self.left.parent = self
        if self.right:
            self.right.parent = self

    def lock(self):
        ancesters = self.get_ancesters()
        has_lock_below = self.num_lock_below > 0
        has_lock_above = any(map(lambda p: p.locked, ancesters))
        if self.locked or has_lock_below or has_lock_above:
            return False

        self.locked = True
        for p in ancesters:
            p.num_lock_below += 1
        return True
        
    def unlock(self):
        if not self.locked:
            return False
        
        self.locked = False
        for p in self.get_ancesters():
            p.num_lock_below -= 1
        return True

    def is_locked(self):
        return self.locked

    def get_ancesters(self):
        p = self.parent
        res = []
        while p:
            res.append(p)
            p = p.parent
        return res


class TreeNodeSpec(unittest.TestCase):
    def setUp(self):
        """
            a
           / \
          b   c
         / \ / \
        d  e f  g
        """
        self.d = TreeNode()
        self.e = TreeNode()
        self.f = TreeNode()
        self.g = TreeNode()
        self.b = TreeNode(self.d, self.e)
        self.c = TreeNode(self.f, self.g)
        self.a = TreeNode(self.b, self.c)
    
    def assert_lock_node(self, node):
        self.assertIsNotNone(node)
        self.assertFalse(node.is_locked())
        self.assertTrue(node.lock())
        self.assertTrue(node.is_locked())

    def assert_unlock_node(self, node):
        self.assertIsNotNone(node)
        self.assertTrue(node.is_locked())
        self.assertTrue(node.unlock())
        self.assertFalse(node.is_locked())

    def test_non_overlapping_lock(self):
        self.assert_lock_node(self.b)
        self.assert_lock_node(self.f)
        self.assert_lock_node(self.g)
        self.assert_unlock_node(self.b)
        self.assert_unlock_node(self.f)
        self.assert_unlock_node(self.g)

    def test_has_lock_above(self):
        self.assert_lock_node(self.a)
        self.assertFalse(self.b.is_locked())
        self.assertFalse(self.b.lock())
        self.assert_unlock_node(self.a)
        self.assert_lock_node(self.b)

    def test_has_lock_below(self):
        self.assert_lock_node(self.e)
        self.assertFalse(self.a.lock())
        self.assertFalse(self.b.lock())
        self.assert_unlock_node(self.e)
        self.assert_lock_node(self.b)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Oct 12, 2020 \[Medium\] Searching in Rotated Array
---
> **Question:** A sorted array of integers was rotated an unknown number of times. Given such an array, find the index of the element in the array in faster than linear time. If the element doesn't exist in the array, return null.
> 
> For example, given the array `[13, 18, 25, 2, 8, 10]` and the element 8, return 4 (the index of 8 in the array).
> 
> You can assume all the integers in the array are unique.

**My thoughts:** In order to solve this problem, we will need the following properties:

1. Multiple rotation can at most break array into two sorted subarrays. 
2. All numbers on left sorted subarray are always larger than numbers on the right sorted subarray.

The idea of binary search is about how to breaking the problem size into half instead of checking `arr[mid]` against target number. Two edge cases could happen while doing binary search:

1. mid element is on the same part as the left-most element
2. mid element is on different part


**Solution with Binary Search:** [https://repl.it/@trsong/Searching-Elem-in-Rotated-Array](https://repl.it/@trsong/Searching-Elem-in-Rotated-Array)
```py
import unittest

def rotated_array_search(nums, target):
    if not nums:
        return None
    
    lo = 0
    hi = len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] < nums[mid] < target or nums[lo] > target:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


class RotatedArraySearchSpec(unittest.TestCase):
    def test_array_without_rotation(self):
        self.assertIsNone(rotated_array_search([], 0))
    
    def test_array_without_rotation2(self):
        self.assertEqual(2, rotated_array_search([1, 2, 3], 3))
    
    def test_array_without_rotation3(self):
        self.assertIsNone(rotated_array_search([1, 3], 2))

    def test_array_with_one_rotation(self):
        self.assertIsNone(rotated_array_search([4, 5, 6, 1, 2, 3], 0))
        
    def test_array_with_one_rotation2(self):
        self.assertEqual(2, rotated_array_search([4, 5, 6, 1, 2, 3], 6))
    
    def test_array_with_one_rotation3(self):
        self.assertEqual(4, rotated_array_search([13, 18, 25, 2, 8, 10], 8))

    def test_array_with_two_rotations(self):
        self.assertEqual(0, rotated_array_search([6, 1, 2, 3, 4, 5], 6))

    def test_array_with_two_rotations2(self):
        self.assertEqual(4, rotated_array_search([5, 6, 1, 2, 3, 4], 3))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 11, 2020 LC 668 \[Hard\] Kth Smallest Number in Multiplication Table
---
> **Question:** Find out the k-th smallest number quickly from the multiplication table.
>
> Given the height m and the length n of a m * n Multiplication Table, and a positive integer k, you need to return the k-th smallest number in this table.

**Example 1:**
```py
Input: m = 3, n = 3, k = 5
Output: 
Explanation: 
The Multiplication Table:
1	2	3
2	4	6
3	6	9
The 5-th smallest number is 3 (1, 2, 2, 3, 3).
```

**Example 2:**
```py
Input: m = 2, n = 3, k = 6
Output: 
Explanation: 
The Multiplication Table:
1	2	3
2	4	6
The 6-th smallest number is 6 (1, 2, 2, 3, 4, 6).
```

**Solution with Binary Search:** [https://repl.it/@trsong/Kth-Smallest-Number-in-Multiplication-Table](https://repl.it/@trsong/Kth-Smallest-Number-in-Multiplication-Table)
```py
import unittest

def find_kth_num(m, n, k):
    num_row, num_col = min(m, n), max(m, n)
    # count smaller number row by row
    count = lambda num: sum(map(lambda r: min(num // r, num_col), xrange(1, min(num, num_row)+1)))
    
    lo = 1
    hi = m * n
    while lo < hi:
        mid = lo + (hi - lo) // 2
        smaller_num = count(mid)
        if smaller_num < k:
            lo = mid + 1
        else:
            hi = mid
    
    return lo


class FindKthNumSpec(unittest.TestCase):
    def test_example(self):
        """
        1	2	3
        2	4	6
        3	6	9
        """
        m, n, k = 3, 3, 5
        self.assertEqual(3, find_kth_num(m, n, k))

    def test_example2(self):
        """
        1	2	3
        2	4	6
        """
        m, n, k = 2, 3, 6
        self.assertEqual(6, find_kth_num(m, n, k))

    def test_single_row(self):
        """
        1	2	3   4   5
        """
        m, n, k = 1, 5, 5
        self.assertEqual(5, find_kth_num(m, n, k))

    def test_single_column(self):
        """
        1
        2
        3
        """
        m, n, k = 3, 1, 2
        self.assertEqual(2, find_kth_num(m, n, k))

    def test_single_cell(self):
        """
        1
        """
        m, n, k = 1, 1, 1
        self.assertEqual(1, find_kth_num(m, n, k))

    def test_large_table(self):
        m, n, k = 42, 34, 401
        self.assertEqual(126, find_kth_num(m, n, k))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 10, 2020 \[Hard\] Longest Palindromic Substring
---
> **Question:** Given a string, find the longest palindromic contiguous substring. If there are more than one with the maximum length, return any one.
>
> For example, the longest palindromic substring of `"aabcdcb"` is `"bcdcb"`. The longest palindromic substring of `"bananas"` is `"anana"`.

**Solution with DP:** [https://repl.it/@trsong/Find-the-Longest-Palindromic-Substring](https://repl.it/@trsong/Find-the-Longest-Palindromic-Substring)
```py
import unittest

def find_longest_palindromic_substring(s):
    if not s:
        return ""

    n = len(s)
    # Let dp[i][j] represents whether substring[i:j+1] is palindromic or not
    #     dp[i][j] = True if dp[i+1][j-1] and s[i]==s[j] 
    dp = [[False for _ in xrange(n)] for _ in xrange(n)]

    max_window = 1
    max_window_start = 0

    for window in xrange(1, n+1):
        for start in xrange(n):
            end = start + window - 1
            if end >= n:
                break
                
            sub_result = window <= 2 or dp[start+1][end-1]
            dp[start][end] = sub_result and s[start] == s[end]

            if dp[start][end]:
                max_window = window
                max_window_start = start
    
    return s[max_window_start: max_window_start + max_window]
            

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

### Oct 9, 2020 LC 665 \[Medium\] Off-by-One Non-Decreasing Array
---
> **Question:** Given an array of integers, write a function to determine whether the array could become non-decreasing by modifying at most 1 element.
>
> For example, given the array `[10, 5, 7]`, you should return true, since we can modify the `10` into a `1` to make the array non-decreasing.
>
> Given the array `[10, 5, 1]`, you should return false, since we can't modify any one element to get a non-decreasing array.

**My thoughts:** try to identify the down postition. The problematic position prevents array from non-decreasing is either the down position or its previous position. Just remove either position and test array again, if it works then it's off-by-one array otherwise it's not since more positions need to be removed.

**Solution:** [https://repl.it/@trsong/Determine-if-Off-by-One-Non-Decreasing-Array](https://repl.it/@trsong/Determine-if-Off-by-One-Non-Decreasing-Array)
```py
import unittest

def is_off_by_one_array(nums):
    n = len(nums)
    if n <= 2:
        return True

    down_pos = None
    for i in xrange(1, n):
        if nums[i-1] <= nums[i]:
            continue

        if down_pos is not None:
            return False
        down_pos = i

    if down_pos is None or down_pos == 1 or down_pos == n-1:
        return True
    else:
        return nums[down_pos-1] <= nums[down_pos+1] or nums[down_pos-2] <= nums[down_pos] 


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


### Oct 8, 2020 \[Easy\] Reverse a Linked List
---
> **Question:** Given a singly-linked list, reverse the list. This can be done iteratively or recursively. Can you get both solutions?

**Example:**
```py
Input: 4 -> 3 -> 2 -> 1 -> 0 -> NULL
Output: 0 -> 1 -> 2 -> 3 -> 4 -> NULL
```

**Solution:** [https://repl.it/@trsong/Reverse-a-Linked-List](https://repl.it/@trsong/Reverse-a-Linked-List)
```py
import unittest

class ListUtil(object):
    @staticmethod
    def iterative_reverse(lst):
        prev = None
        while lst:
            next = lst.next
            lst.next = prev
            prev = lst
            lst = next
        return prev

    @staticmethod
    def recursive_reverse(lst):
        if not lst or not lst.next:
            return lst
        root = ListUtil.recursive_reverse(lst.next)
        lst.next.next = lst
        lst.next = None
        return root
    

class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    def __eq__(self, other):
        return other and self.val == other.val and self.next == other.next

    def __repr__(self):
        return "%d -> %s" % (self.val, self.next)
    
    @staticmethod
    def List(*vals):
        dummy = p = ListNode(-1)
        for v in vals:
            p.next = ListNode(v)
            p = p.next
        return dummy.next


class ListUtilSpec(unittest.TestCase):
    def assert_result(self, nums):
        lst = ListNode.List(*nums)
        lst2 = ListNode.List(*nums)
        expected = ListNode.List(*reversed(nums))
        self.assertEqual(expected, ListUtil.recursive_reverse(lst))
        self.assertEqual(expected, ListUtil.iterative_reverse(lst2))

    def test_empty_list(self):
        self.assert_result([])

    def test_one_elem_list(self):
        self.assert_result([1])

    def test_two_elem_list(self):
        self.assert_result([1, 2])

    def test_list_with_duplicate_elem(self):
        self.assert_result([1, 2, 1, 1])

    def test_list_with_duplicate_elem2(self):
        self.assert_result([1, 1, 1, 1, 1, 1])

    def test_unique_list(self):
        self.assert_result([1, 2, 3, 4, 5, 6])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 7, 2020 \[Easy\] First and Last Indices of an Element in a Sorted Array
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

### Oct 6, 2020 \[Hard\] LFU Cache
---
> **Question:** Implement an LFU (Least Frequently Used) cache. It should be able to be initialized with a cache size n, and contain the following methods:
>
> - `set(key, value)`: sets key to value. If there are already n items in the cache and we are adding a new item, then it should also remove the least frequently used item. If there is a tie, then the least recently used key should be removed.
> - `get(key)`: gets the value at key. If no such key exists, return null.
Each operation should run in O(1) time.

**Solution:** [https://repl.it/@trsong/LFU-Cache](https://repl.it/@trsong/LFU-Cache)
```py
import unittest

class FreqNode(object):
    def __init__(self, count=0, lru=None, next=None):
        self.update(count, lru, next)

    def update(self, new_count, new_lru, new_next):
        self.count = new_count
        self.lru = new_lru
        self.next = new_next


class LFUCache(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.lookup = {}
        self.tail = FreqNode()
        self.head = FreqNode(next=self.tail)
        self.size = 0

    def get(self, key):
        if key not in self.lookup:
            return None

        freq_node = self.lookup[key]
        val = freq_node.lru.get(key)
        self.next_or_insert_node(freq_node, key, val)

        if freq_node.lru.empty():
            self.delete_node(freq_node)
        return val

    def put(self, key, val):
        if self.capacity <= 0:
            return
        if key in self.lookup:
            freq_node = self.lookup[key]
            freq_node.lru.put(key, val)
            self.get(key)
        else:
            if self.size >= self.capacity:
                self.size -= 1
                least_freq_node = self.head.next
                least_freq_key = least_freq_node.lru.pop()
                del self.lookup[least_freq_key]
                if least_freq_node.lru.empty():
                    self.delete_node(least_freq_node)
            self.next_or_insert_node(self.head, key, val)
            self.size += 1
            
    def delete_node(self, freq_node):
        next_freq_node = freq_node.next
        freq_node.update(next_freq_node.count, next_freq_node.lru, next_freq_node.next)
        if next_freq_node == self.tail:
            return
        for key in next_freq_node.lru.keys():
            self.lookup[key] = freq_node
        
    def next_or_insert_node(self, freq_node, key, val):
        next_freq_node = freq_node.next
        if next_freq_node == self.tail:
            self.tail.update(freq_node.count+1, LRUCache(), FreqNode())
            self.tail = self.tail.next
        elif next_freq_node.count != freq_node.count + 1:
            next_freq_node = FreqNode(freq_node.count + 1, LRUCache(), next_freq_node)
            freq_node.next = next_freq_node

        next_freq_node.lru.put(key, val)
        self.lookup[key] = next_freq_node


class ListNode(object):
    def __init__(self, next=None):
        self.update(None, None, next)
        
    def update(self, new_key, new_val, new_next):
        self.key = new_key
        self.val = new_val
        self.next = new_next


class LRUCache(object):
    def __init__(self):
        self.lookup = {}
        self.tail = ListNode()
        self.head = ListNode(self.tail)

    def empty(self):
        return not self.lookup

    def keys(self):
        return self.lookup.keys()

    def get(self, key):
        if key not in self.lookup:
            return None
        
        node = self.lookup[key]
        val = node.val
        del self.lookup[key]
        
        # Remove original node
        if node.next.key:
            self.lookup[node.next.key] = node
        node.update(node.next.key, node.next.val, node.next.next)
        return val

    def put(self, key, val):
        self.get(key)
        self.insert_node(key, val)

    def pop(self):
        most_inactive_node = self.head.next
        del self.lookup[most_inactive_node.key]
        self.head.next = most_inactive_node.next
        return most_inactive_node.key
            
    def insert_node(self, key, val):
        self.lookup[key] = self.tail
        self.tail.update(key, val, ListNode())
        self.tail = self.tail.next


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


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Oct 5, 2020 \[Medium\] LRU Cache
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

**Solution:** [https://repl.it/@trsong/Design-LRU-Cache](https://repl.it/@trsong/Design-LRU-Cache)
```py
import unittest

class ListNode(object):
    def __init__(self, next=None):
        self.update(None, None, next)
        
    def update(self, new_key, new_val, new_next):
        self.key = new_key
        self.val = new_val
        self.next = new_next

class LRUCache(object):
    def __init__(self, capacity):
        self.capacity = capacity  # assume capacity is always positive
        self.lookup = {}
        self.tail = ListNode()
        self.head = ListNode(self.tail)

    def get(self, key):
        if key not in self.lookup:
            return None
        
        # Duplicate the node and move to tail
        node = self.lookup[key]
        key, val = node.key, node.val
        self.insert_node(key, val)
        
        # Remove original node
        if node.next.key:
            self.lookup[node.next.key] = node
        node.update(node.next.key, node.next.val, node.next.next)
        return val

    def put(self, key, val):
        if key not in self.lookup:
            self.insert_node(key, val)
        else:
            node = self.lookup[key]
            node.val = val
            self.get(key)  # populate the entry
            
        if len(self.lookup) > self.capacity:
            most_inactive_node = self.head.next
            del self.lookup[most_inactive_node.key]
            self.head.next = most_inactive_node.next
            
    def insert_node(self, key, val):
        self.lookup[key] = self.tail
        self.tail.update(key, val, ListNode())
        self.tail = self.tail.next


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
    unittest.main(exit=False)
```


### Oct 4, 2020  \[Easy\] Add Two Numbers as a Linked List
---
> **Question:** You are given two linked-lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

**Example:**
```py
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```

**Solution:** [https://repl.it/@trsong/Add-Two-Numbers-and-Return-as-a-Linked-List](https://repl.it/@trsong/Add-Two-Numbers-and-Return-as-a-Linked-List)
```py
import unittest

def lists_addition(l1, l2):
    carry = 0
    p = dummy = ListNode(-1)
    while l1 or l2 or carry:
        if l1:
            carry += l1.val
            l1 = l1.next
            
        if l2:
            carry += l2.val
            l2 = l2.next
        p.next = ListNode(carry % 10)
        carry //= 10
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


### Oct 3, 2020 \[Hard\] Find Arbitrage Opportunities
---
> **Question:** Suppose you are given a table of currency exchange rates, represented as a 2D array. Determine whether there is a possible arbitrage: that is, whether there is some sequence of trades you can make, starting with some amount A of any currency, so that you can end up with some amount greater than A of that currency.
>
> There are no transaction costs and you can trade fractional quantities.

**Example:**
```py
Given the following matrix:
#       RMB,   USD,  CAD
# RMB     1, 0.14, 0.19
# USD  6.97,    1,  1.3
# CAD  5.37, 0.77,    1

# Since RMB -> CAD -> RMB:  1 Yuan * 0.19 * 5.37 = 1.02 Yuan
# If we keep exchange RMB to CAD and exchange back, we can make a profit eventually.
```

**My thoughts:** The question ask if there exists a cycle that has edge weight multiplication > 1. 

But how to efficiently find such a cycle? 

The trick is to take advantage of the property of log function: `log(a*b) = log(a) + log(b)`. We can convert edge weights into negative log of original edge weights. As `a * b * c > 1 <=> -log(a*b*c) < 0 <=> -log(a) * -log(b) * -log(c) < 0`. We can use Bellman-ford Algorithm to detect if negative cycle exists or not. If so, there must be a cycle whose weight multiplication > 1.


**Solution with Bellman-Ford Algorithm(DP):** [https://repl.it/@trsong/Find-Arbitrage-Opportunities-for-Currency-Exchange](https://repl.it/@trsong/Find-Arbitrage-Opportunities-for-Currency-Exchange)
```py
import unittest
import math

def has_arbitrage_opportunities(currency_exchange_matrix):
    if not currency_exchange_matrix:
        return False

    n = len(currency_exchange_matrix)
    transform = lambda rate: -math.log(rate) if rate > 0 else float('inf')
    transformed_matrix = map(lambda row: map(transform, row), currency_exchange_matrix)

    distance = [float('inf')] * n
    distance[0] = 0
    for _ in xrange(n-1):
        # In each iteration, distance[v] is shortest distance from 0 to v
        for u in xrange(n):
            for v in xrange(n):
                distance[v] = min(distance[v], distance[u] + transformed_matrix[u][v])
    
    for u in xrange(n):
        for v in xrange(n):
            if distance[v] > distance[u] + transformed_matrix[u][v]:
                # Exists negative cycle that keeps shrinking shortest path
                return True
    return False


class HasArbitrageOpportunitiesSpec(unittest.TestCase):
    def test_empty_matrix(self):
        self.assertFalse(has_arbitrage_opportunities([]))

    def test_cannot_exchange_currencies(self):
        currency_exchange_matrix = [
        #    A, B
            [1, 0], # A 
            [0, 1]  # B
        ]
        self.assertFalse(has_arbitrage_opportunities(currency_exchange_matrix))

    def test_benefit_from_any_exchange_action(self):
        currency_exchange_matrix = [
        #    A, B
            [1, 2], # A 
            [2, 1]  # B
        ]
        # A -> B -> A:  $1 * 2 * 2 = $4
        self.assertTrue(has_arbitrage_opportunities(currency_exchange_matrix))

    def test_benefit_from_one_exchange_action(self):
        currency_exchange_matrix = [
        #    A,   B
            [1, 0.5], # A 
            [4,   1]  # B
        ]
        # A -> B -> A:  $1 * 0.5 * 4 = $2
        self.assertTrue(has_arbitrage_opportunities(currency_exchange_matrix))
    
    def test_system_glitch(self):
        currency_exchange_matrix = [
        #    A
            [2]
        ]
        # A -> A':  $1 * 2 = $2
        self.assertTrue(has_arbitrage_opportunities(currency_exchange_matrix))

    def test_multi_currency_system(self):
        currency_exchange_matrix = [
        #    RMB,   USD,  CAD
            [   1, 0.14, 0.19],  # RMB
            [6.97,    1,  1.3],  # USD
            [5.37, 0.77,    1]   # CAD
        ]
        # RMB -> CAD -> RMB:  1 Yuan * 0.19 * 5.37 = 1.02 Yuan
        self.assertTrue(has_arbitrage_opportunities(currency_exchange_matrix))
    
    def test_multi_currency_system2(self):
        currency_exchange_matrix = [
        #     RMB,   USD,    JPY
            [1   ,  0.14,  15.49],  # RMB
            [6.97,     1, 108.02],  # USD
            [0.06, 0.009,      1]   # JPY
        ]
        self.assertFalse(has_arbitrage_opportunities(currency_exchange_matrix))

    def test_exists_a_glitch_path_involves_all_currencies(self):
        currency_exchange_matrix = [
            #  A, B,   C,   D
            [  1, 1,   0,   0], # A
            [0.9, 1, 0.7,   0], # B
            [1.1, 0,   1, 0.2], # C
            [10,  0,   0,   1]  # D
        ]
        # A -> B -> C -> D -> A:  $1 * 1 * 0.7 * 0.2 * 10 = $1.4 
        # A -> B -> A: $1 * 1 * 0.9 = $0.9
        # A -> B -> C -> A: $1 * 1 * 0.7 * 1.1 = $0.77
        self.assertTrue(has_arbitrage_opportunities(currency_exchange_matrix))

    def test_compliated_example(self):
        currency_exchange_matrix = [
        #      PLN,   EUR,   USD,   RUB,   INR,   MXN
            [    1,  0.23,  0.25, 16.43, 18.21,  4.94],  # PLN
            [ 4.34,     1,  1.11, 71.40, 79.09, 21.44],  # EUR
            [ 3.93,  0.90,     1, 64.52, 71.48, 19.37],  # USD
            [0.061, 0.014, 0.015,     1,  1.11,  0.30],  # RUB
            [0.055, 0.013, 0.014,  0.90,     1,  0.27],  # INR
            [ 0.20, 0.047, 0.052,  3.33,  3.69,     1]   # MXN   
        ]
        # RUB --> INR --> PLN --> RUB: 1 * 1.11 * 0.055 * 16.43 = 1.0031
        # USD --> MXN --> USD --> RUB --> INR --> EUR --> PLN: 1 * 19.37 * 0.052 * 64.52 * 1.11 * 0.013 * 4.34 = 4.07
        # USD --> MXN --> USD --> PLN: 1 * 19.37 * 0.052 * 3.93 = 3.96
        self.assertTrue(has_arbitrage_opportunities(currency_exchange_matrix))
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Oct 2, 2020 LC 678 [Medium] Balanced Parentheses with Wildcard
---
> **Question:** You're given a string consisting solely of `(`, `)`, and `*`. `*` can represent either a `(`, `)`, or an empty string. Determine whether the parentheses are balanced.
>
> For example, `(()*` and `(*)` are balanced. `)*(` is not balanced.


**My thoughts:** The wildcard `*` can represents `-1`, `0`, `1`, thus `x` number of `"*"`s can represents range from `-x` to `x`. Just like how we check balance without wildcard, but this time balance is a range: the wildcard just make any balance number within the range become possible. While keep the balance range in mind, we need to make sure each time the range can never go below 0 to become unbalanced, ie. number of open parentheses less than close ones.  


**Solution:** [https://repl.it/@trsong/Determine-Balanced-Parentheses-with-Wildcard](https://repl.it/@trsong/Determine-Balanced-Parentheses-with-Wildcard)
```py
import unittest

def balanced_parentheses(s):
    lower_bound = higher_bound = 0
    for ch in s:
        lower_bound += 1 if ch == '(' else -1
        higher_bound -= 1 if ch == ')' else -1
                
        if higher_bound < 0:
            return False
            
        lower_bound = max(lower_bound, 0)
        
    return lower_bound == 0


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

### Oct 1, 2020 \[Easy\] Is Anagram of Palindrome
---
> **Question:** Given a string, determine whether any permutation of it is a palindrome.
>
> For example, `'carrace'` should return True, since it can be rearranged to form `'racecar'`, which is a palindrome. `'daily'` should return False, since there's no rearrangement that can form a palindrome.

**My thoughts:** An anagram of palindrome can have at most 1 character with odd counts. We can use xor to set each bit. And check if the final result is a power of 2. Mearning that, either there is only single digit of 1 in bit map or no 1 at all. 

**Solution:** [https://repl.it/@trsong/EnchantingAnnualArchitecture](https://repl.it/@trsong/EnchantingAnnualArchitecture)
```py
import unittest
from functools import reduce

def is_palindrome_anagram(s):
    set_bit = lambda accu, ch: accu ^ 1 << ord(ch)
    bit_map = reduce(set_bit, s, 0)
    return isPowerOfTwo(bit_map)


def isPowerOfTwo(num):
    return num & num - 1 == 0


class IsPalindromeAnagramSpec(unittest.TestCase):
    def test_example(self):
        s = 'carrace'
        self.assertTrue(is_palindrome_anagram(s))

    def test_empty_string(self):
        self.assertTrue(is_palindrome_anagram(''))

    def test_even_number_of_string(self):
        s = 'aa'
        self.assertTrue(is_palindrome_anagram(s))

    def test_even_number_of_string2(self):
        s = 'aabbcc'
        self.assertTrue(is_palindrome_anagram(s))

    def test_odd_number_of_string(self):
        s = 'a'
        self.assertTrue(is_palindrome_anagram(s))

    def test_odd_number_of_string2(self):
        s = 'abb'
        self.assertTrue(is_palindrome_anagram(s))

    def test_odd_number_of_string3(self):
        s = 'aaabbb'
        self.assertFalse(is_palindrome_anagram(s))

    def test_odd_number_of_string4(self):
        s = 'abc'
        self.assertFalse(is_palindrome_anagram(s))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 30, 2020 \[Easy\] Witness of The Tall People
---
> **Question:** There are `n` people lined up, and each have a height represented as an integer. A murder has happened right in front of them, and only people who are taller than everyone in front of them are able to see what has happened. How many witnesses are there?

**Example:**
```py
Input: [3, 6, 3, 4, 1]  
Output: 3
Explanation: Only [6, 4, 1] were able to see in front of them.
 #
 #
 # #
####
####
#####
36341  
```

**Solution:** [https://repl.it/@trsong/Witness-of-The-Tall-People](https://repl.it/@trsong/Witness-of-The-Tall-People)
```py
import unittest

def num_witness(heights):
    local_max = 0 
    res = 0
    for height in reversed(heights):
        if height > local_max:
            local_max = height
            res += 1
    return res


class NumWitnessSpec(unittest.TestCase):
    def test_example(self):
        heights = [3, 6, 3, 4, 1]
        expected = 3  # [1, 4, 6]
        self.assertEqual(expected, num_witness(heights))

    def test_no_witness(self):
        heights = []
        expected = 0
        self.assertEqual(expected, num_witness(heights))

    def test_one_witness(self):
        heights = [10]
        expected = 1
        self.assertEqual(expected, num_witness(heights))

    def test_two_witnesses(self):
        heights = [2, 1]
        expected = 2
        self.assertEqual(expected, num_witness(heights))

    def test_height_up_down_up(self):
        heights = [1, 10, 2, 20]
        expected = 1
        self.assertEqual(expected, num_witness(heights))

    def test_height_down_up_down(self):
        heights = [20, 2, 10, 1]
        expected = 3  # [1, 10, 20]
        self.assertEqual(expected, num_witness(heights))

    def test_same_height(self):
        heights = [1, 1, 1, 1]
        expected = 1
        self.assertEqual(expected, num_witness(heights))

    def test_same_height2(self):
        heights = [3, 3, 3, 2, 2, 1, 1]
        expected = 3
        self.assertEqual(expected, num_witness(heights))

    def test_same_height3(self):
        heights = [1, 1, 2, 2, 3, 3, 3, 3]
        expected = 1
        self.assertEqual(expected, num_witness(heights))

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 29, 2020 \[Medium\] Number of Smaller Elements to the Right
---
> **Question:** Given an array of integers, return a new array where each element in the new array is the number of smaller elements to the right of that element in the original input array.
>
> For example, given the array `[3, 4, 9, 6, 1]`, return `[1, 1, 2, 1, 0]`, since:
>
> * There is 1 smaller element to the right of 3
> * There is 1 smaller element to the right of 4
> * There are 2 smaller elements to the right of 9
> * There is 1 smaller element to the right of 6
> * There are no smaller elements to the right of 1

**My thoughts:** we want a data structure to allow quick insert and query for number of value smaller. BST fit into our requirement. We just need to add two fields to each tree node: number of left child and frequency of current value. 

While iterating through tree node, if we go right, then we accumulate all number on the left. If we go left, then we just need to accumulate the left number of last node. For example, given the follow BST, if we want to know number of elem less than 6, we just use node4, node6. `3 + 1 + 1 = 5`.

```py
     4 (3, 1)
   /   \
  2     6 (1, 1)
 / \   / \
1   3 5   7 (0, 1)

Each node looks like node (a, b) where 
- a is number in parenthese is the number of children on the left 
- b is frequency
```

**Solution with BST:** [https://repl.it/@trsong/Find-Number-of-Smaller-Elements-to-the-Right](https://repl.it/@trsong/Find-Number-of-Smaller-Elements-to-the-Right)
```py
import unittest

def count_right_smaller_numbers(nums):
    root = BSTNode(float('inf'))
    res = []
    for num in reversed(nums):
        res.append(root.insert(num))
    res.reverse()
    return res

 
class BSTNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.left_count = 0
        self.freq = 0

    def insert(self, v):
        res = 0
        cur = self
        while True:
            if cur.val == v:
                res += cur.left_count
                cur.freq += 1
                break
            elif cur.val < v:
                res += cur.left_count + cur.freq
                cur.right = cur.right or BSTNode(v)
                cur = cur.right
            else:
                cur.left_count += 1
                cur.left = cur.left or BSTNode(v)
                cur = cur.left
        return res
 

class CountRightSmallerNuberSpec(unittest.TestCase):
    def test_example(self):
        nums = [3, 4, 9, 6, 1]
        expected_result = [1, 1, 2, 1, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))

    def test_empty_array(self):
        self.assertEqual([], count_right_smaller_numbers([]))

    def test_ascending_array(self):
        nums = [0, 1, 2, 3, 4]
        expected_result = [0, 0, 0, 0, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))

    def test_array_with_unique_value(self):
        nums = [1, 1, 1]
        expected_result = [0, 0, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))

    def test_descending_array(self):
        nums = [4, 3, 2, 1]
        expected_result = [3, 2, 1, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))

    def test_increase_decrease_increase(self):
        nums = [1, 4, 2, 5, 0, 4]
        expected_result = [1, 2, 1, 2, 0, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))

    def test_decrease_increase_decrease(self):
        nums = [3, 1, 2, 0]
        expected_result = [3, 1, 1, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))

    def test_decrease_increase_decrease2(self):
        nums = [12, 1, 2, 3, 0, 11, 4]
        expected_result = [6, 1, 1, 1, 0, 1, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))

    def test_negative_values(self):
        nums = [8, -2, -1, -2, -1, 3]
        expected_result = [5, 0, 1, 0, 0, 0]
        self.assertEqual(expected_result, count_right_smaller_numbers(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 28, 2020 \[Hard\] Largest Divisible Pairs Subset
---
> **Question:** Given a set of distinct positive integers, find the largest subset such that every pair of elements in the subset `(i, j)` satisfies either `i % j = 0` or `j % i = 0`.
>
> For example, given the set `[3, 5, 10, 20, 21]`, you should return `[5, 10, 20]`. Given `[1, 3, 6, 24]`, return `[1, 3, 6, 24]`.

**My thoughts:** Notice that smaller number mod large number can never be zero, but the other way around might be True. e.g.  `2 % 4 == 2`.  `9 % 3 == 0`. Thus we can eliminate the condition `i % j = 0 or j % i = 0` to just `j % i = 0 where j is larger`. This can be done by sorting into descending order.

After that we can use dp to solve this problem. Define `dp[i]` to be the largest number of divisible pairs ends at `i`. And `max(dp)` will give the largest number of division pairs among all i. By backtracking from the index, we can find what are these numbers.

**Solution with DP:** [https://repl.it/@trsong/Find-Largest-Divisible-Pairs-Subset](https://repl.it/@trsong/Find-Largest-Divisible-Pairs-Subset)
```py
import unittest

def largest_divisible_pairs_subset(nums):
    if not nums:
        return []
        
    nums.sort()
    # Let dp[i] represents largest divisible parts end at nums[i]
    #     dp[i] = max(dp[j]) + 1 for all j < i and nums[i] divisible by nums[j]
    n = len(nums)
    dp = [0] * n
    parent = [-1] * n
    for i in xrange(n):
        max_pairs = -1
        for j in xrange(i):
            if nums[i] % nums[j] == 0 and dp[j] > max_pairs: 
                max_pairs = dp[j]
                parent[i] = j
        dp[i] = max_pairs + 1

    index = max(xrange(n), key=lambda i: dp[i])
    res = []
    while index >= 0:
        res.append(nums[index])
        index = parent[index]
    return res if len(res) > 1 else []


class LargestDivisiblePairsSubsetSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(sorted([5, 10, 20]), sorted(largest_divisible_pairs_subset([3, 5, 10, 20, 21])))

    def test_example2(self):
        self.assertEqual(sorted([1, 3, 6, 24]), sorted(largest_divisible_pairs_subset([1, 3, 6, 24])))

    def test_multiple_of_3_and_5(self):
        self.assertEqual(sorted([10, 5, 20]), sorted(largest_divisible_pairs_subset([10, 5, 3, 15, 20])))

    def test_prime_and_multiple_of_3(self):
        self.assertEqual(sorted([18, 1, 3, 6]), sorted(largest_divisible_pairs_subset([18, 1, 3, 6, 13, 17])))

    def test_decrease_array(self):
        self.assertEqual(sorted([8, 4, 2, 1]), sorted(largest_divisible_pairs_subset([8, 7, 6, 5, 4, 2, 1])))

    def test_array_with_duplicated_values(self):
        self.assertEqual(sorted([3, 3, 3, 1]), sorted(largest_divisible_pairs_subset([2, 2, 3, 3, 3, 1])))

    def test_no_divisible_pairs(self):
        self.assertEqual([], largest_divisible_pairs_subset([2, 3, 5, 7, 11, 13, 17, 19]))

    def test_no_divisible_pairs2(self):
        self.assertEqual([], largest_divisible_pairs_subset([1]))
        self.assertEqual([], largest_divisible_pairs_subset([]))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 27, 2020 \[Easy\] Find Corresponding Node in Cloned Tree
--- 
> **Question:** Given two binary trees that are duplicates of one another, and given a node in one tree, find that corresponding node in the second tree. 
> 
> There can be duplicate values in the tree (so comparing node1.value == node2.value isn't going to work).


**Solution with DFS:** [https://repl.it/@trsong/Find-Corresponding-Node-in-Cloned-Tree](https://repl.it/@trsong/Find-Corresponding-Node-in-Cloned-Tree)
```py
from copy import deepcopy
import unittest

def find_node(root1, root2, node1):
    if not root1 or not root2 or not node1:
        return None
    
    traversal1 = dfs_traversal(root1)
    traversal2 = dfs_traversal(root2)
    for n1, n2 in zip(traversal1, traversal2):
        if n1 == node1:
            return n2
    return None


def dfs_traversal(root):
    stack = [root]
    while stack:
        cur = stack.pop()
        yield cur
        if cur.right:
            stack.append(cur.right)
        if cur.left:
            stack.append(cur.left)


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return "TreeNode(%d, %s, %s)" % (self.val, self.left, self.right)


class FindNodeSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertIsNone(find_node(None, None, None))

    def test_one_node_tree(self):
        root1 = TreeNode(1)
        root2 = deepcopy(root1)
        self.assertEqual(root2, find_node(root1, root2, root1))

    def test_leaf_node(self):
        """
          1
         / \
        2   3
             \
              4
        """
        root1 = TreeNode(1, TreeNode(2), TreeNode(3, right=TreeNode(4)))
        root2 = deepcopy(root1)
        f = lambda root: root.right.right
        self.assertEqual(f(root2), find_node(root1, root2, f(root1)))

    def test_internal_node(self):
        """
            1
           / \
          2   3
         /   /
        0   1
        """
        left_tree = TreeNode(2, TreeNode(0))
        right_tree = TreeNode(3, TreeNode(1))
        root1 = TreeNode(1, left_tree, right_tree)
        root2 = deepcopy(root1)
        f = lambda root: root.left
        self.assertEqual(f(root2), find_node(root1, root2, f(root1))) 

    def test_duplicated_value_in_tree(self):
        """
          1
           \
            1
           /
          1
         /
        1
        """
        root1 = TreeNode(1, right=TreeNode(1, TreeNode(1, TreeNode(1))))
        root2 = deepcopy(root1)
        f = lambda root: root.right.left
        self.assertEqual(f(root2), find_node(root1, root2, f(root1))) 
    
    def test_find_root_node(self):
        """
          1
         / \
        2   3
        """
        root1 = TreeNode(1, TreeNode(2), TreeNode(3))
        root2 = deepcopy(root1)
        self.assertEqual(root2, find_node(root1, root2, root1))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 26, 2020 \[Medium\] Fixed Order Task Scheduler with Cooldown
--- 
> **Question:** We have a list of tasks to perform, with a cooldown period. We can do multiple of these at the same time, but we cannot run the same task simultaneously.
>
> Given a list of tasks, find how long it will take to complete the tasks in the order they are input.

**Example:**
```py
tasks = [1, 1, 2, 1]
cooldown = 2
output: 7 (order is 1 _ _ 1 2 _ 1)
```

**My thoughts:** Since we have to execute the task with specific order and each task has a cooldown time, we can use a map to record the last occurence of the same task and set up a threshold in order to make sure we will always wait at least the cooldown amount of time before proceed.

**Solution:** [https://repl.it/@trsong/Fixed-Order-Task-Scheduler-with-Cooldown](https://repl.it/@trsong/Fixed-Order-Task-Scheduler-with-Cooldown)
```py
import unittest

def multitasking_time(task_seq, cooldown):
    task_log = {}
    cur_time = 0
    for task in task_seq:
        # idle
        delta = cur_time - task_log.get(task, float('-inf'))
        idle = max(cooldown - delta + 1, 0)
        cur_time += idle

        # log time
        task_log[task] = cur_time

         # execuiton
        cur_time += 1
    
    return cur_time

    
class MultitaskingTimeSpec(unittest.TestCase):
    def test_example(self):
        tasks = [1, 1, 2, 1]
        cooldown = 2
        # order is 1 _ _ 1 2 _ 1
        self.assertEqual(7, multitasking_time(tasks, cooldown))

    def test_example2(self):
        tasks = [1, 1, 2, 1, 2]
        cooldown = 2
        # order is 1 _ _ 1 2 _ 1 2
        self.assertEqual(8, multitasking_time(tasks, cooldown))
    
    def test_zero_cool_down_time(self):
        tasks = [1, 1, 1]
        cooldown = 0
        # order is 1 1 1 
        self.assertEqual(3, multitasking_time(tasks, cooldown))
    
    def test_task_queue_is_empty(self):
        tasks = []
        cooldown = 100
        self.assertEqual(0, multitasking_time(tasks, cooldown))

    def test_cooldown_is_three(self):
        tasks = [1, 2, 1, 2, 1, 1, 2, 2]
        cooldown = 3
        # order is 1 2 _ _ 1 2 _ _ 1 _ _ _ 1 2 _ _ _ 2
        self.assertEqual(18, multitasking_time(tasks, cooldown))
    
    def test_multiple_takes(self):
        tasks = [1, 2, 3, 1, 3, 2, 1, 2]
        cooldown = 2
        # order is 1 2 3 1 _ 3 2 1 _ 2
        self.assertEqual(10, multitasking_time(tasks, cooldown))

    def test_when_cool_down_is_huge(self):
        tasks = [1, 2, 2, 1, 2]
        cooldown = 100
        # order is 1 2 [_ * 100] 2 1 [_ * 99] 2
        self.assertEqual(204, multitasking_time(tasks, cooldown))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 25, 2020 LC 273 \[Hard\] Integer to English Words
---
> **Question:** Convert a non-negative integer to its English word representation. 

**Example 1:**
```py
Input: 123
Output: "One Hundred Twenty Three"
```

**Example 2:**
```py
Input: 12345
Output: "Twelve Thousand Three Hundred Forty Five"
```

**Example 3:**
```py
Input: 1234567
Output: "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
```

**Example 4:**
```py
Input: 1234567891
Output: "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One"
```

**Solution:** [https://repl.it/@trsong/Convert-Int-to-English-Words](https://repl.it/@trsong/Convert-Int-to-English-Words)
```py
import unittest

word_lookup = {
    0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five',
    6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten',
    11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', 15: 'Fifteen',
    16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', 19: 'Nineteen', 20: 'Twenty',
    30: 'Thirty', 40: 'Forty', 50: 'Fifty', 60: 'Sixty', 
    70: 'Seventy', 80: 'Eighty', 90: 'Ninety',
    100: 'Hundred', 1000: 'Thousand', 1000000: 'Million', 1000000000: 'Billion'
}


def number_to_words(num):
    global word_lookup
    if num == 0:
        return word_lookup[0]

    res = []
    for threshold in [1000000000, 1000000, 1000]:
        if num >= threshold:
            res += read_three_digits(num // threshold)
            res.append(word_lookup[threshold])
            num %= threshold

    if num > 0:
        res += read_three_digits(num)
    
    return ' '.join(res)


def read_three_digits(num):
    global word_lookup
    res = []
    if num >= 100:
        res.append(word_lookup[num // 100])
        res.append(word_lookup[100])
        num %= 100
    
    if num > 20:
        res.append(word_lookup[num - num % 10])
        num %= 10

    if num > 0:
        res.append(word_lookup[num])

    return res


class NumberToWordSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(number_to_words(123), "One Hundred Twenty Three")

    def test_example2(self):
        self.assertEqual(number_to_words(12345), "Twelve Thousand Three Hundred Forty Five")

    def test_example3(self):
        self.assertEqual(number_to_words(1234567), "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven")

    def test_example4(self):
        self.assertEqual(number_to_words(1234567891), "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One")

    def test_zero(self):
        self.assertEqual(number_to_words(0), "Zero")

    def test_one_digit(self):
        self.assertEqual(number_to_words(8), "Eight")

    def test_two_digits(self):
        self.assertEqual(number_to_words(21), "Twenty One")
        self.assertEqual(number_to_words(10), "Ten")
        self.assertEqual(number_to_words(20), "Twenty")
        self.assertEqual(number_to_words(16), "Sixteen")
        self.assertEqual(number_to_words(32), "Thirty Two")
        self.assertEqual(number_to_words(30), "Thirty")

    def test_ignore_thousand_part(self):
        self.assertEqual(number_to_words(30002000000), "Thirty Billion Two Million")

    def test_ignore_million_part(self):
        self.assertEqual(number_to_words(50000000200), "Fifty Billion Two Hundred")


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 24, 2020 \[Hard\] De Bruijn Sequence 
--- 
> **Question:** Given a set of characters `C` and an integer `k`, a **De Bruijn** Sequence is a cyclic sequence in which every possible k-length string of characters in C occurs exactly once.
> 
> **Background:** De Bruijn Sequence can be used to shorten a brute-force attack on a PIN-like code lock that does not have an "enter" key and accepts the last n digits entered. For example, a digital door lock with a 4-digit code would have B (10, 4) solutions, with length 10000. Therefore, only at most 10000 + 3 = 10003 (as the solutions are cyclic) presses are needed to open the lock. Trying all codes separately would require 4 × 10000 = 40000 presses.

**Example1:**
```py
Input: C = [0, 1], k = 3
Output: 0011101000
All possible strings of length three (000, 001, 010, 011, 100, 101, 110 and 111) appear exactly once as sub-strings in C.
```

**Example2:**
```py
Input: C = [0, 1], k = 2
Output: 01100
```

**My thoughts:** Treat all substring as nodes, substr1 connect to substr2 if substr1 shift 1 become substr2. Eg.  `123 -> 234 -> 345`. In order to only include each substring only once, we traverse entire graph using DFS and mark visited nodes and avoid visit same node over and over again.  

**Solution with DFS:** [https://repl.it/@trsong/Find-De-Bruijn-Sequence](https://repl.it/@trsong/Find-De-Bruijn-Sequence)
```py
import unittest

def de_bruijn_sequence(char_set, k):
    char_set = map(str, char_set)
    begin_state = char_set[0] * k
    visited = set()
    stack = [(begin_state, begin_state)]
    res = []
    
    while stack:
        cur_state, appended_char = stack.pop()
        if cur_state in visited:
            continue
        visited.add(cur_state)
        res.append(appended_char)

        for char in char_set:
            next_state = cur_state[1:] + char
            if next_state not in visited:
                stack.append((next_state, char))
    
    return "".join(res)


class DeBruijnSequenceSpec(unittest.TestCase):
    @staticmethod
    def cartesian_product(char_set, k):
        def cartesian_product_recur(char_set, k):
            if k == 1:
                return [[str(c)] for c in char_set]
            res = []
            for accu_list in cartesian_product_recur(char_set, k-1):
                for char in char_set:
                    res.append(accu_list + [str(char)])
            return res
        
        return map(lambda lst: ''.join(lst), cartesian_product_recur(char_set, k))

    def validate_de_bruijn_seq(self, char_set, k, seq_res):
        n = len(char_set)
        expected_substr_set = set(DeBruijnSequenceSpec.cartesian_product(char_set, k))
        result_substr_set = set()
        for i in xrange(n**k):
            result_substr_set.add(seq_res[i:i+k])
        # Check if all substr are covered
        self.assertEqual(expected_substr_set, result_substr_set)
        
    def test_example1(self):
        k, char_set = 3, [0, 1]
        res = de_bruijn_sequence(char_set, k)
        # Possible Solution: "0011101000"
        self.validate_de_bruijn_seq(char_set, k, res)

    def test_example2(self):
        k, char_set = 2, [0, 1]
        res = de_bruijn_sequence(char_set, k)
        # Possible Solution: "01100"
        self.validate_de_bruijn_seq(char_set, k, res)

    def test_multi_charset(self):
        k, char_set = 2, [0, 1, 2]
        res = de_bruijn_sequence(char_set, k)
        # Possible Solution : "0022120110"
        self.validate_de_bruijn_seq(char_set, k, res)

    def test_multi_charset2(self):
        k, char_set = 3, [0, 1, 2]
        res = de_bruijn_sequence(char_set, k)
        # Possible Solution : "00022212202112102012001110100"
        self.validate_de_bruijn_seq(char_set, k, res)
        
    def test_larger_k(self):
        k, char_set = 5, [0, 1, 2]
        res = de_bruijn_sequence(char_set, k)
        self.validate_de_bruijn_seq(char_set, k, res)

        
if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 23, 2020 LC 236 \[Medium\] Lowest Common Ancestor of a Binary Tree
---
> **Question:** Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

**Example:**

```py
     1
   /   \
  2     3
 / \   / \
4   5 6   7

LCA(4, 5) = 2
LCA(4, 6) = 1
LCA(3, 4) = 1
LCA(2, 4) = 2
```

**Solution:** [https://repl.it/@trsong/Find-the-Lowest-Common-Ancestor-of-a-Given-Binary-Tree](https://repl.it/@trsong/Find-the-Lowest-Common-Ancestor-of-a-Given-Binary-Tree)
```py
import unittest

def find_lca(tree, n1, n2):    
    if not tree or tree == n1 or tree == n2:
        return tree
    
    left_res = find_lca(tree.left, n1, n2)
    right_res = find_lca(tree.right, n1, n2)

    if left_res and right_res:
        return tree
    else:
        return left_res or right_res


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return "NodeValue: " + str(self.val)


class FindLCASpec(unittest.TestCase):
    def setUp(self):
        """
             1
           /   \
          2     3
         / \   / \
        4   5 6   7
        """
        self.n4 = TreeNode(4)
        self.n5 = TreeNode(5)
        self.n6 = TreeNode(6)
        self.n7 = TreeNode(7)
        self.n2 = TreeNode(2, self.n4, self.n5)
        self.n3 = TreeNode(3, self.n6, self.n7)
        self.n1 = TreeNode(1, self.n2, self.n3)

    def test_both_nodes_on_leaves(self):
        self.assertEqual(self.n2, find_lca(self.n1, self.n4, self.n5))

    def test_both_nodes_on_leaves2(self):
        self.assertEqual(self.n3, find_lca(self.n1, self.n6, self.n7))
    
    def test_both_nodes_on_leaves3(self):
        self.assertEqual(self.n1, find_lca(self.n1, self.n4, self.n6))

    def test_nodes_on_different_levels(self):
        self.assertEqual(self.n2, find_lca(self.n1, self.n4, self.n2))
    
    def test_nodes_on_different_levels2(self):
        self.assertEqual(self.n2, find_lca(self.n1, self.n4, self.n2))
    
    def test_nodes_on_different_levels3(self):
        self.assertEqual(self.n1, find_lca(self.n1, self.n4, self.n1))

    def test_same_nodes(self):
        self.assertEqual(self.n2, find_lca(self.n1, self.n2, self.n2))
    
    def test_same_nodes2(self):
        self.assertEqual(self.n6, find_lca(self.n1, self.n6, self.n6))


if __name__ == '__main__':
    unittest.main(exit=False)
```
### Sep 22, 2020 \[Medium\] Is Bipartite
---
> **Question:** Given an undirected graph G, check whether it is bipartite. Recall that a graph is bipartite if its vertices can be divided into two independent sets, U and V, such that no edge connects vertices of the same set.

**Example:**
```py
is_bipartite(vertices=3, edges=[(0, 1), (1, 2), (2, 0)])  # returns False 
is_bipartite(vertices=2, edges=[(0, 1), (1, 0)])  # returns True. U = {0}. V = {1}. 
```

**My thoughts:** A graph is a bipartite if we can just use 2 colors to cover entire graph so that every other node have same color. Like this [solution with DFS](https://trsong.github.io/python/java/2020/02/02/DailyQuestionsFeb.html#apr-15-2020-medium-is-bipartite). Or the following way with union-find. For each node `u`, just union all of its neihbors. And at the same time, `u` should not be connectted with any of its neihbor.

**Solution with Union-Find (DisjointSet):** [https://repl.it/@trsong/Is-a-Graph-Bipartite](https://repl.it/@trsong/Is-a-Graph-Bipartite)
```py
import unittest

class DisjointSet(object):
    def __init__(self, n):
        self.parent = range(n)

    def find(self, v):
        p = self.parent
        while p[v] != v:
            p[v] = p[p[v]]
            v = p[v]
        return v

    def is_connected(self, u, v):
        return self.find(u) == self.find(v)

    def union(self, u, v):
        p1 = self.find(u)
        p2 = self.find(v)
        if p1 != p2:
            self.parent[p1] = p2


def is_bipartite(vertices, edges):
    neighbors = [None] * vertices
    for u, v in edges:
        neighbors[u] = neighbors[u] or []
        neighbors[v] = neighbors[v] or []
        neighbors[u].append(v)
        neighbors[v].append(u)
    
    uf = DisjointSet(vertices)
    for u in xrange(vertices):
        if neighbors[u] is None:
            continue
        for v in neighbors[u]:
            if uf.is_connected(u, v):
                return False
            uf.union(v, neighbors[u][0])
    return True
            

class IsBipartiteSpec(unittest.TestCase):
    def test_example1(self):
        self.assertFalse(is_bipartite(vertices=3, edges=[(0, 1), (1, 2), (2, 0)]))

    def test_example2(self):
        self.assertTrue(is_bipartite(vertices=2, edges=[(0, 1), (1, 0)]))

    def test_empty_graph(self):
        self.assertTrue(is_bipartite(vertices=0, edges=[]))

    def test_one_node_graph(self):
        self.assertTrue(is_bipartite(vertices=1, edges=[]))
    
    def test_disconnect_graph1(self):
        self.assertTrue(is_bipartite(vertices=10, edges=[(0, 1), (1, 0)]))

    def test_disconnect_graph2(self):
        self.assertTrue(is_bipartite(vertices=10, edges=[(0, 1), (1, 0), (2, 3), (3, 4), (4, 5), (5, 2)]))

    def test_disconnect_graph3(self):
        self.assertFalse(is_bipartite(vertices=10, edges=[(0, 1), (1, 0), (2, 3), (3, 4), (4, 2)])) 

    def test_square(self):
        self.assertTrue(is_bipartite(vertices=4, edges=[(0, 1), (1, 2), (2, 3), (3, 0)]))

    def test_k5(self):
        vertices = 5
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4), 
            (3, 4)
        ]
        self.assertFalse(is_bipartite(vertices, edges))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 21, 2020 \[Easy\] Run-length String Encode and Decode
---
> **Question:** Run-length encoding is a fast and simple method of encoding strings. The basic idea is to represent repeated successive characters as a single count and character. For example, the string `"AAAABBBCCDAA"` would be encoded as `"4A3B2C1D2A"`.
>
> Implement run-length encoding and decoding. You can assume the string to be encoded have no digits and consists solely of alphabetic characters. You can assume the string to be decoded is valid.


**Solution:** [https://repl.it/@trsong/Run-length-String-Encode-and-Decode](https://repl.it/@trsong/Run-length-String-Encode-and-Decode)
```py
import unittest
from collections import Counter

class RunLengthProcessor(object):
    @staticmethod
    def encode(s):
        if not s:
            return ""

        count = 0
        prev = s[0]
        res = []
        for i, ch in enumerate(s):
            if ch == prev:
                count += 1
            else:
                res.extend([str(count), prev])
                prev = ch
                count = 1
        res.extend([str(count), prev])
        return ''.join(res)

    @staticmethod
    def decode(s):
        i = 0
        res = []
        while i < len(s):
            count = 0
            while '0' <= s[i] <= '9':
                count = 10 * count + int(s[i])
                i += 1

            res.append(s[i] * count)
            i += 1
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
    unittest.main(exit=False)
```

### Sep 20, 2020 LC 421 \[Medium\] Maximum XOR of Two Numbers in an Array
---
> **Question:** Given an array of integers, find the maximum XOR of any two elements.

**Example:**
```py
Input: nums = [3,10,5,25,2,8]
Output: 28
Explanation: The maximum result is 5 XOR 25 = 28.
```

**My thoughts:** The idea is to efficiently get the other number for each number such that xor is max. We can parellel this process with Trie: insert all number into trie, and for each number greedily choose the largest number that has different digit. 

**Solution with Trie:** [https://repl.it/@trsong/Maximum-XOR-of-Two-Numbers-in-an-Array](https://repl.it/@trsong/Maximum-XOR-of-Two-Numbers-in-an-Array)
```py
import unittest

class Trie(object):
    def __init__(self):
        self.children = [None, None]

    def insert(self, num):
        p = self
        for i in xrange(32, -1, -1):
            bit = 1 & (num >> i)
            if p.children[bit] is None:
                p.children[bit] = Trie()
            p = p.children[bit]

    def find_max_xor_with(self, num):
        p = self
        accu = 0
        for i in xrange(32, -1, -1):
            bit = 1 & (num >> i)
            if p.children[1 - bit]:
                accu ^= 1 << i
                p = p.children[1 - bit]
            else:
                p = p.children[bit]
        return accu


def find_max_xor(nums):
    trie = Trie()
    for num in nums:
        trie.insert(num)

    return max(map(trie.find_max_xor_with, nums))


class FindMaxXORSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(bin(expected), bin(result))

    def test_example(self):
        nums = [  0b11,
                0b1010,
                 0b101,
               0b11001,
                  0b10,
                0b1000]
        expected = 0b101 ^ 0b11001
        self.assert_result(expected, find_max_xor(nums))

    def test_only_one_element(self):
        nums = [0b11]
        expected = 0b11 ^ 0b11
        self.assert_result(expected, find_max_xor(nums))

    def test_two_elements(self):
        nums = [0b10, 0b100]
        expected = 0b10 ^ 0b100
        self.assert_result(expected, find_max_xor(nums))

    def test_example2(self):
        nums = [ 0b1110,
              0b1000110,
               0b110101,
              0b1010011,
               0b110001,
              0b1011011,
               0b100100,
              0b1010000,
              0b1011100,
               0b110011,
              0b1000010,
              0b1000110]
        expected = 0b1011011 ^ 0b100100
        self.assert_result(expected, find_max_xor(nums))

    def test_return_max_number_of_set_bit(self):
        nums = [ 0b111,
                  0b11,
                   0b1,
                     0]
        expected = 0b111 ^ 0
        self.assert_result(expected, find_max_xor(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 19, 2020 LC 653 \[Easy\] Two Sum in BST
---
> **Question:** Given the root of a binary search tree, and a target K, return two nodes in the tree whose sum equals K.

**Example:**
```py
Given the following tree and K of 20:

    10
   /   \
 5      15
       /  \
     11    15

Return the nodes 5 and 15.
```

**Solution with In-order Traversal:** [https://repl.it/@trsong/Two-Sum-in-BST](https://repl.it/@trsong/Two-Sum-in-BST)
```py
import unittest

def solve_two_sum_bst(root, k):
    if not root:
        return None
    
    it1 = find_inorder_traversal(root)
    it2 = find_inorder_traversal(root, reverse=True)
    p1 = next(it1)
    p2 = next(it2)
    while p1 != p2:
        pair_sum = p1.val + p2.val
        if pair_sum < k:
            p1 = next(it1)
        elif pair_sum > k:
            p2 = next(it2)
        else:
            return [p1.val, p2.val]
    return None


def find_inorder_traversal(root, reverse=False):
    p = root
    stack = []
    while p or stack:
        if p:
            stack.append(p)
            p = p.right if reverse else p.left
        else:
            p = stack.pop()
            yield p
            p = p.left if reverse else p.right


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class SolveTwoSumBSTSepc(unittest.TestCase):
    def test_example(self):
        """
             10
            /   \
           5    15
               /  \
             11    15
        """
        root = TreeNode(10, TreeNode(5), TreeNode(15, TreeNode(11), TreeNode(15)))
        k = 20
        expected = [5, 15]
        self.assertItemsEqual(expected, solve_two_sum_bst(root, k))

    def test_empty_tree(self):
        self.assertIsNone(solve_two_sum_bst(None, 0))

    def test_left_heavy_tree(self):
        """
            5
           / \
          3   6
         / \   \
        2   4   7
        """
        left_tree = TreeNode(3, TreeNode(2), TreeNode(4))
        right_tree = TreeNode(6, right=TreeNode(7))
        root = TreeNode(5, left_tree, right_tree)
        k = 9
        expected = [2, 7]
        self.assertItemsEqual(expected, solve_two_sum_bst(root, k))

    def test_no_pair_exists(self):
        """
          5
         / \
        1   8
        """
        root = TreeNode(5, TreeNode(1), TreeNode(8))
        k = 10
        self.assertIsNone(solve_two_sum_bst(root, k))

    def test_balanced_tree(self):
        """
             5
           /   \
          3     11
         / \   /  \
        0   4 10  25
        """
        left_tree = TreeNode(3, TreeNode(0), TreeNode(4))
        right_tree = TreeNode(11, TreeNode(10), TreeNode(25))
        root = TreeNode(5, left_tree, right_tree)
        k = 13
        expected = [3, 10]
        self.assertItemsEqual(expected, solve_two_sum_bst(root, k))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 18, 2020 \[Hard\] Order of Course Prerequisites
---
> **Question:** We're given a hashmap associating each courseId key with a list of courseIds values, which represents that the prerequisites of courseId are courseIds. Return a sorted ordering of courses such that we can finish all courses.
>
> Return null if there is no such ordering.
>
> For example, given `{'CSC300': ['CSC100', 'CSC200'], 'CSC200': ['CSC100'], 'CSC100': []}`, should return `['CSC100', 'CSC200', 'CSC300']`.


**My thoughts:** there are two ways to produce toplogical sort order: DFS or count inward edge as below. For DFS technique see this [post](https://trsong.github.io/python/java/2019/05/01/DailyQuestions.html#jul-5-2019-hard-order-of-course-prerequisites). For below method, we count number of inward edges for each node and recursively remove edges of each node that has no inward egdes. The node removal order is what we call topological order.

**Solution with Topological Sort:** [https://repl.it/@trsong/Find-Order-of-Course-Prerequisites](https://repl.it/@trsong/Find-Order-of-Course-Prerequisites)
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
    unittest.main(exit=False)
```


### Sep 17, 2020 \[Medium\] Construct BST from Post-order Traversal
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

**Solution:** [https://repl.it/@trsong/Construct-Binary-Search-Tree-from-Post-order-Traversal](https://repl.it/@trsong/Construct-Binary-Search-Tree-from-Post-order-Traversal)
```py
import unittest

def construct_bst(post_order_traversal):
    return construct_bst_recur(post_order_traversal, float('-inf'), float('inf'))


def construct_bst_recur(stack, lo, hi):
    if not (stack and lo <= stack[-1] <= hi):
        return None
    val = stack.pop()
    right_child = construct_bst_recur(stack, val, hi)
    left_child = construct_bst_recur(stack, lo, val)
    return Node(val, left_child, right_child)


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
    unittest.main(exit=False)
```

### Sep 16, 2020 LC 329 \[Hard\] Longest Increasing Path in a Matrix
---
> **Question:** Given an integer matrix, find the length of the longest increasing path.
>
> From each cell, you can either move to four directions: left, right, up or down. You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).

**Example 1:**
```py
Input: nums = 
[
  [9,9,4],
  [6,6,8],
  [2,1,1]
] 
Output: 4 
Explanation: The longest increasing path is [1, 2, 6, 9].
```

**Example 2:**
```py
Input: nums = 
[
  [3,4,5],
  [3,2,6],
  [2,2,1]
] 
Output: 4 
Explanation: The longest increasing path is [3, 4, 5, 6]. Moving diagonally is not allowed.
```

**My thoughts:** From any node, a valid path can never decrease and won't form a cycle; that create a DAG. The longest path in DAG can be calculated based on topological order within linear time.

**Solution 1 with Topological Sort (Remove Edge):** [https://repl.it/@trsong/Longest-Increasing-Path-in-a-Matrix-Remove-Edge-Top-Sort](https://repl.it/@trsong/Longest-Increasing-Path-in-a-Matrix-Remove-Edge-Top-Sort)
```py
def longest_path_length(grid):
    if not grid or not grid[0]:
        return 0
    
    DIRECTIONS = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    n, m = len(grid), len(grid[0])
    inward_edges = [[0 for _ in xrange(m)] for _ in xrange(n)]
    queue = []
    
    for r in xrange(n):
        for c in xrange(m):
            cur_val = grid[r][c]
            for dr, dc in DIRECTIONS:
                from_r, from_c = r + dr, c + dc
                if 0 <= from_r < n and 0 <=  from_c < m and grid[from_r][from_c] < cur_val:
                    inward_edges[r][c] += 1
            if inward_edges[r][c] == 0:
                queue.append((r, c))
                
    max_distance = 0
    distance_grid = [[1 for _ in xrange(m)] for _ in xrange(n)]
    while queue:
        r, c = queue.pop(0)
        max_distance = max(max_distance, distance_grid[r][c])
        cur_val = grid[r][c]
        for dr, dc in DIRECTIONS:
            to_r, to_c = r + dr, c + dc
            if 0 <= to_r < n and 0 <= to_c < m and cur_val < grid[to_r][to_c]:
                distance_grid[to_r][to_c] = max(distance_grid[to_r][to_c], 1 + distance_grid[r][c])
                inward_edges[to_r][to_c] -= 1
                if inward_edges[to_r][to_c] == 0:
                    queue.append((to_r, to_c))
                    
    return max_distance
```

**Solution 2 with Topological Sort (DFS):** [https://repl.it/@trsong/Longest-Increasing-Path-in-a-Matrix](https://repl.it/@trsong/Longest-Increasing-Path-in-a-Matrix)
```py
import unittest

DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]


def longest_path_length(grid):
    if not grid or not grid[0]:
        return 0

    top_order = find_top_order(grid)
    return get_longest_distance(grid, top_order)


def find_top_order(grid):
    class NodeState:
        UNVISITED = 0
        VISITING = 1
        VISITED = 2

    n, m = len(grid), len(grid[0])
    node_states = [[NodeState.UNVISITED for _ in xrange(m)] for _ in xrange(n)]
    reverse_top_order = []

    for c in xrange(m):
        for r in xrange(n):
            if node_states[r][c] is not NodeState.UNVISITED:
                continue

            stack = [(r, c)]
            while stack:
                cur_r, cur_c = stack[-1]
                cur_val = grid[cur_r][cur_c]
                if node_states[cur_r][cur_c] is NodeState.VISITED:
                    stack.pop()
                elif node_states[cur_r][cur_c] is NodeState.VISITING:
                    reverse_top_order.append((cur_r, cur_c))
                    node_states[cur_r][cur_c] = NodeState.VISITED
                else:
                    node_states[cur_r][cur_c] = NodeState.VISITING
                    for dr, dc in DIRECTIONS:
                        new_r, new_c = cur_r + dr, cur_c + dc
                        if (0 <= new_r < n and 0 <= new_c < m
                                and node_states[new_r][new_c] is
                                NodeState.UNVISITED and grid[new_r][new_c] > cur_val):
                            stack.append((new_r, new_c))
    return reverse_top_order[::-1]


def get_longest_distance(grid, top_order):
    n, m = len(grid), len(grid[0])
    distances = [[1 for _ in xrange(m)] for _ in xrange(n)]
    max_distance = 1
    for r, c in top_order:
        cur_val = grid[r][c]
        for dr, dc in DIRECTIONS:
            from_r, from_c = r + dr, c + dc
            if 0 <= from_r < n and 0 <= from_c < m and grid[from_r][from_c] < cur_val:
                distances[r][c] = max(distances[r][c], 1 + distances[from_r][from_c])
                max_distance = max(max_distance, distances[r][c])
    return max_distance


class LongestPathLengthSpec(unittest.TestCase):
    def test_example(self):
        grid = [
            [9, 9, 4],
            [6, 6, 8],
            [2, 1, 1]
        ]
        expected = 4  # 1, 2, 6, 9
        self.assertEqual(expected, longest_path_length(grid))

    def test_example2(self):
        grid = [
            [3, 4, 5],
            [3, 2, 6],
            [2, 2, 1]
        ]
        expected = 4  # 3, 4, 5, 6
        self.assertEqual(expected, longest_path_length(grid))

    def test_empty_grid(self):
        self.assertEqual(0, longest_path_length([[]]))

    def test_sping_around_entire_grid(self):
        grid = [
            [5, 6, 7],
            [4, 1, 8],
            [3, 2, 9]
        ]
        expected = 9  # 1, 2, 3, 4, 5, 6, 7, 8, 9
        self.assertEqual(expected, longest_path_length(grid))

    def test_no_path(self):
        grid = [
            [1, 1],
            [1, 1],
            [1, 1]
        ]
        expected = 1 
        self.assertEqual(expected, longest_path_length(grid))

    def test_two_paths(self):
        grid = [
            [4, 3, 2],
            [5, 0, 1],
            [3, 2, 3]
        ]
        expected = 6  # 0, 1, 2, 3, 4, 5 
        self.assertEqual(expected, longest_path_length(grid))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 15, 2020 LC 986 \[Medium\] Interval List Intersections
---
> **Question:** Given two lists of closed intervals, each list of intervals is pairwise disjoint and in sorted order.
>
> Return the intersection of these two interval lists.

**Example:**
```py
Input: A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
```

**Solution:** [https://repl.it/@trsong/Interval-List-Intersections](https://repl.it/@trsong/Interval-List-Intersections)
```py
import unittest

def interval_intersection(seq1, seq2):
    stream1, stream2 = iter(seq1), iter(seq2)
    res = []
    interval1 = next(stream1, None)
    interval2 = next(stream2, None)
    while interval1 and interval2:
        start1, end1 = interval1
        start2, end2 = interval2
        if end1 < start2:
            interval1 = next(stream1, None)
        elif end2 < start1:
            interval2 = next(stream2, None)
        else:
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)  
            res.append([overlap_start, overlap_end])
            # cut off overlap and reuse
            interval1 = [overlap_end + 1, end1]  
            interval2 = [overlap_end + 1, end2]
    return res


class IntervalIntersectionSpec(unittest.TestCase):
    def test_example(self):
        seq1 = [[0, 2], [5, 10], [13, 23], [24, 25]]
        seq2 = [[1, 5], [8, 12], [15, 24], [25, 26]]
        expected = [[1, 2], [5, 5], [8, 10], [15, 23], [24, 24], [25, 25]]
        self.assertEqual(expected, interval_intersection(seq1, seq2))

    def test_empty_interval(self):
        seq1 = []
        seq2 = [[1, 2]]
        expected = []
        self.assertEqual(expected, interval_intersection(seq1, seq2))

    def test_overlap_entire_interval(self):
        seq1 = [[1, 1], [2, 2], [3, 3], [8, 8]]
        seq2 = [[0, 5]]
        expected = [[1, 1], [2, 2], [3, 3]]
        self.assertEqual(expected, interval_intersection(seq1, seq2))

    def test_overlap_entire_interval2(self):
        seq1 = [[0, 5]]
        seq2 = [[1, 2], [4, 7]]
        expected = [[1, 2], [4, 5]]
        self.assertEqual(expected, interval_intersection(seq1, seq2))

    def test_overlap_one_interval(self):
        seq1 = [[0, 2], [5, 7]]
        seq2 = [[1, 3], [4, 6]]
        expected = [[1, 2], [5, 6]]
        self.assertEqual(expected, interval_intersection(seq1, seq2))

    def test_same_start_time(self):
        seq1 = [[0, 2], [3, 7]]
        seq2 = [[0, 5]]
        expected = [[0, 2], [3, 5]]
        self.assertEqual(expected, interval_intersection(seq1, seq2))

    def test_same_start_time2(self):
        seq1 = [[0, 2], [5, 7]]
        seq2 = [[5, 7], [8, 9]]
        expected = [[5, 7]]
        self.assertEqual(expected, interval_intersection(seq1, seq2))
    
    def test_no_overlapping(self):
        seq1 = [[1, 2], [5, 7]]
        seq2 = [[0, 0], [3, 4], [8, 9]]
        expected = []
        self.assertEqual(expected, interval_intersection(seq1, seq2))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 14, 2020 LC 57 \[Hard\] Insert Interval
---
> **Question:** Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).
>
> You may assume that the intervals were initially sorted according to their start times.

**Example 1:**
```py
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
```

**Example 2:**
```py
Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
```

**Solution:** [https://repl.it/@trsong/Insert-Interval](https://repl.it/@trsong/Insert-Interval)
```py
import unittest

def insert_interval(intervals, newInterval):
    res = []
    cur_start, cur_end = newInterval
    for start, end in intervals:
        if end < cur_start:
            res.append([start, end])
        elif cur_end < start:
            res.append([cur_start, cur_end])
            cur_start, cur_end = start, end
        else:
            cur_start = min(cur_start, start)
            cur_end = max(cur_end, end)
    res.append([cur_start, cur_end])
    return res


class InsertIntervalSpec(unittest.TestCase):
    def test_example(self):
        intervals = [[1, 3], [6, 9]]
        newInterval = [2, 5]
        expected = [[1, 5], [6, 9]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_example2(self):
        intervals = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
        newInterval = [4, 8]
        expected = [[1, 2], [3, 10], [12, 16]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_insert_into_empty_intervals(self):
        intervals = []
        newInterval = [4, 8]
        expected = [[4, 8]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_new_interval_overlaps_all(self):
        intervals = [[1, 2], [3, 4], [5, 6]]
        newInterval = [2, 10]
        expected = [[1, 10]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_new_interval_does_not_expand_original(self):
        intervals = [[1, 7], [9, 12]]
        newInterval = [10, 11]
        expected = [[1, 7], [9, 12]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_new_interval_before_all(self):
        intervals = [[1, 2], [3, 4]]
        newInterval = [0, 1]
        expected = [[0, 2], [3, 4]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_new_interval_before_all2(self):
        intervals = [[1, 2], [3, 4]]
        newInterval = [0, 0]
        expected = [[0, 0], [1, 2], [3, 4]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_new_interval_after_all(self):
        intervals = [[1, 2], [3, 4]]
        newInterval = [5, 6]
        expected = [[1, 2], [3, 4], [5, 6]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))

    def test_new_interval_after_all2(self):
        intervals = [[1, 2], [3, 4]]
        newInterval = [3, 8]
        expected = [[1, 2], [3, 8]]
        self.assertEqual(expected, insert_interval(intervals, newInterval))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 13, 2020 LC 37 \[Hard\] Sudoku Solver
---
> **Question:** Write a program to solve a Sudoku puzzle by filling the empty cells.
>
> A sudoku solution must satisfy all of the following rules:
> - Each of the digits `1-9` must occur exactly once in each row.
> - Each of the digits `1-9` must occur exactly once in each column.
> - Each of the the digits `1-9` must occur exactly once in each of the 9 `3x3` sub-boxes of the grid.
> - Empty cells are indicated `0`.

**Example:**
```py
Input: [
    [3, 0, 6, 5, 0, 8, 4, 0, 0], 
    [5, 2, 0, 0, 0, 0, 0, 0, 0], 
    [0, 8, 7, 0, 0, 0, 0, 3, 1], 
    [0, 0, 3, 0, 1, 0, 0, 8, 0], 
    [9, 0, 0, 8, 6, 3, 0, 0, 5], 
    [0, 5, 0, 0, 9, 0, 6, 0, 0], 
    [1, 3, 0, 0, 0, 0, 2, 5, 0], 
    [0, 0, 0, 0, 0, 0, 0, 7, 4], 
    [0, 0, 5, 2, 0, 6, 3, 0, 0]
]

Possible output:
[
    [3, 1, 6, 5, 7, 8, 4, 9, 2],
    [5, 2, 9, 1, 3, 4, 7, 6, 8],
    [4, 8, 7, 6, 2, 9, 5, 3, 1],
    [2, 6, 3, 4, 1, 5, 9, 8, 7],
    [9, 7, 4, 8, 6, 3, 1, 2, 5],
    [8, 5, 1, 7, 9, 2, 6, 4, 3],
    [1, 3, 8, 9, 4, 7, 2, 5, 6],
    [6, 9, 2, 3, 5, 1, 8, 7, 4],
    [7, 4, 5, 2, 8, 6, 3, 1, 9]
]
```

**Solution with Backtracking:** [https://repl.it/@trsong/Sudoku-Solver](https://repl.it/@trsong/Sudoku-Solver)
```py
import unittest
from functools import reduce
from copy import deepcopy

class SampleSudoku(object):
    UNFINISHED1 = [
        [3, 0, 6, 5, 0, 8, 4, 0, 0],
        [5, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 7, 0, 0, 0, 0, 3, 1], 
        [0, 0, 3, 0, 1, 0, 0, 8, 0],
        [9, 0, 0, 8, 6, 3, 0, 0, 5], 
        [0, 5, 0, 0, 9, 0, 6, 0, 0],
        [1, 3, 0, 0, 0, 0, 2, 5, 0],
        [0, 0, 0, 0, 0, 0, 0, 7, 4],
        [0, 0, 5, 2, 0, 6, 3, 0, 0]
    ]
    
    FINISHED1 = [
        [3, 1, 6, 5, 7, 8, 4, 9, 2],
        [5, 2, 9, 1, 3, 4, 7, 6, 8],
        [4, 8, 7, 6, 2, 9, 5, 3, 1],
        [2, 6, 3, 4, 1, 5, 9, 8, 7],
        [9, 7, 4, 8, 6, 3, 1, 2, 5],
        [8, 5, 1, 7, 9, 2, 6, 4, 3],
        [1, 3, 8, 9, 4, 7, 2, 5, 6],
        [6, 9, 2, 3, 5, 1, 8, 7, 4],
        [7, 4, 5, 2, 8, 6, 3, 1, 9]
    ]

    UNFINISHED2 = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    FINISHED2 = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2], 
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1], 
        [7, 1, 3, 9, 2, 4, 8, 5, 6], 
        [9, 6, 1, 5, 3, 7, 2, 8, 4], 
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ]



class SudokuSolver(object):
    def __init__(self, grid):
        self.grid = grid
        self.row_occupied = [[False for _ in xrange(9)] for _ in xrange(9)]
        self.col_occupied = [[False for _ in xrange(9)] for _ in xrange(9)]
        self.reigon_occupied = [[[False for _ in xrange(9)] for _ in xrange(3)] for _ in xrange(3)]

    def mark_position(self, r, c, num):
        self.grid[r][c] = num
        num_index = num - 1
        self.row_occupied[r][num_index] = True
        self.col_occupied[c][num_index] = True
        self.reigon_occupied[r//3][c//3][num_index] = True
    
    def unmark_position(self, r, c):
        num = self.grid[r][c]
        self.grid[r][c] = 0
        num_index = num - 1
        self.row_occupied[r][num_index] = False
        self.col_occupied[c][num_index] = False
        self.reigon_occupied[r//3][c//3][num_index] = False

    def is_valid_number(self, r, c, num):
        num_index = num - 1
        return not(self.row_occupied[r][num_index] 
                or self.col_occupied[c][num_index]
                or self.reigon_occupied[r//3][c//3][num_index])

    def backtrack(self, index, blank_positions):
        if index >= len(blank_positions):
            return True
            
        r, c = blank_positions[index]
        valid_numbers = filter(lambda num: self.is_valid_number(r, c, num), xrange(1, 10))
        for num in valid_numbers:
            self.mark_position(r, c, num)
            if self.backtrack(index + 1, blank_positions):
               return True
            self.unmark_position(r, c)
        return False

    def solve(self):
        blank_positions = []
        for r in xrange(9):
            for c in xrange(9):
                num = self.grid[r][c]
                if num == 0:
                    blank_positions.append((r, c))
                else:
                    self.mark_position(r, c, num)
        
        self.backtrack(0, blank_positions)
        return self.grid

        
class SudokuSolverSpec(unittest.TestCase):
    def assert_result(self, input_grid, expected_grid):
        grid = deepcopy(input_grid)
        solver = SudokuSolver(grid)
        finished_grid = solver.solve()
        validation_res = SudokuValidator.validate(finished_grid)
        msg = '\n' + '\n'.join(map(str, finished_grid)) + '\nPossible Solution: \n' + '\n'.join(map(str, expected_grid))
        self.assertTrue(validation_res, msg)

    def test_example(self):
        self.assert_result(SampleSudoku.UNFINISHED1, SampleSudoku.FINISHED1)

    def test_example2(self):
        self.assert_result(SampleSudoku.UNFINISHED2, SampleSudoku.FINISHED2)


##################
# Testing Utility
##################
class SudokuValidator(object):
    @staticmethod
    def validate(grid):
        return (SudokuValidator.validate_row(grid)
                and SudokuValidator.validate_col(grid)
                and SudokuValidator.validate_region(grid))

    @staticmethod
    def validate_row(grid):
        rows = [[(r, c) for c in xrange(9)] for r in xrange(9)]
        return all(map(lambda row: SudokuValidator.validate_uniq(grid, row), rows))

    @staticmethod
    def validate_col(grid):
        cols = [[(r, c) for r in xrange(9)] for c in xrange(9)]
        return all(map(lambda col: SudokuValidator.validate_uniq(grid, col), cols))

    @staticmethod
    def validate_region(grid):
        indices = [0, 3, 6]
        width = 3
        reigons = [[(r + dr, c + dc) for dr in xrange(width)
                    for dc in xrange(width)] for r in indices for c in indices]
        return all(map(lambda reigon: SudokuValidator.validate_uniq(grid, reigon), reigons))

    @staticmethod
    def validate_uniq(grid, postions):
        values = map(lambda pos: grid[pos[0]][pos[1]], postions)
        count = sum(1 for _ in values)
        bits = reduce(lambda accu, v: accu | (1 << v), values, 0)
        return count == 9 and bits == 0b1111111110


class SudokuValidatorSpec(unittest.TestCase):
    def test_valid_grid(self):
        self.assertTrue(SudokuValidator.validate(SampleSudoku.FINISHED1))

    def test_invalid_grid(self):
        self.assertFalse(SudokuValidator.validate(SampleSudoku.UNFINISHED1))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 12, 2020 \[Medium\] Count Attacking Bishop Pairs
---
> **Question:** On our special chessboard, two bishops attack each other if they share the same diagonal. This includes bishops that have another bishop located between them, i.e. bishops can attack through pieces.
> 
> You are given N bishops, represented as (row, column) tuples on a M by M chessboard. Write a function to count the number of pairs of bishops that attack each other. The ordering of the pair doesn't matter: `(1, 2)` is considered the same as `(2, 1)`.
>
> For example, given `M = 5` and the list of bishops:

```py
(0, 0)
(1, 2)
(2, 2)
(4, 0)
```
> The board would look like this:

```py
[b 0 0 0 0]
[0 0 b 0 0]
[0 0 b 0 0]
[0 0 0 0 0]
[b 0 0 0 0]
```

> You should return 2, since bishops 1 and 3 attack each other, as well as bishops 3 and 4.


**My thoughts:** Cell on same diagonal has the following properties:

- Major diagonal: col - row = constant
- Minor diagonal: col + row = constant
  

**Example:**
```py
>>> [[r-c for c in xrange(5)] for r in xrange(5)]
[
    [0, -1, -2, -3, -4],
    [1, 0, -1, -2, -3],
    [2, 1, 0, -1, -2],
    [3, 2, 1, 0, -1],
    [4, 3, 2, 1, 0]
]

>>> [[r+c for c in xrange(5)] for r in xrange(5)]
[
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8]
]
```
Thus, we can store the number of bishop on the same diagonal and use the formula to calculate n-choose-2: `n(n-1)/2`


**Solution:** [https://repl.it/@trsong/Count-Number-of-Attacking-Bishop-Pairs](https://repl.it/@trsong/Count-Number-of-Attacking-Bishop-Pairs)
```py
import unittest

def count_attacking_pairs(bishop_positions):
    diagonal_key1 = lambda pos: pos[0] - pos[1]
    diagonal_key2 = lambda pos: pos[0] + pos[1]
    return count_pairs_by(bishop_positions, diagonal_key1) + count_pairs_by(bishop_positions, diagonal_key2)


def count_pairs_by(positions, key):
    histogram = {}
    for pos in positions:
        pos_key = key(pos)
        histogram[pos_key] = histogram.get(pos_key, 0) + 1
    # n choos 2 = n * (n-1) / 2
    return sum(map(lambda count: count * (count - 1) // 2, histogram.values()))


class CountAttackingPairSpec(unittest.TestCase):
    def test_zero_bishops(self):
        self.assertEqual(0, count_attacking_pairs([]))

    def test_bishops_everywhere(self):
        """
        b b
        b b
        """
        self.assertEqual(2, count_attacking_pairs([(0, 0), (0, 1), (1, 0), (1, 1)]))

    def test_zero_attacking_pairs(self):
        """
        0 b 0 0
        0 b 0 0
        0 b 0 0
        0 b 0 0
        """
        self.assertEqual(0, count_attacking_pairs([(0, 1), (1, 1), (2, 1), (3, 1)]))
        """
        0 0 0 b
        b 0 0 0
        b 0 b 0
        0 0 0 0
        """
        self.assertEqual(0, count_attacking_pairs([(0, 3), (1, 0), (2, 0), (2, 2)]))

    def test_no_bishop_between_attacking_pairs(self):
        """
        b 0 b
        b 0 b
        b 0 b
        """
        self.assertEqual(2, count_attacking_pairs([(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2)]))
    
    def test_no_bishop_between_attacking_pairs2(self):
        """
        b 0 0 0 0
        0 0 b 0 0
        0 0 b 0 0
        0 0 0 0 0
        b 0 0 0 0
        """
        self.assertEqual(2, count_attacking_pairs([(0, 0), (1, 2), (2, 2), (4, 0)]))

    def test_has_bishop_between_attacking_pairs(self):
        """
        b 0 b
        0 b 0
        b 0 b
        """
        self.assertEqual(6, count_attacking_pairs([(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 11, 2020 LC 358 \[Hard\] Rearrange String K Distance Apart
---
> **Question:** Given a non-empty string str and an integer k, rearrange the string such that the same characters are at least distance k from each other.
>
> All input strings are given in lowercase letters. If it is not possible to rearrange the string, return an empty string "".

**Example 1:**
```py
str = "aabbcc", k = 3
Result: "abcabc"
The same letters are at least distance 3 from each other.
```

**Example 2:**
```py
str = "aaabc", k = 3 
Answer: ""
It is not possible to rearrange the string.
```

**Example 3:**
```py
str = "aaadbbcc", k = 2
Answer: "abacabcd"
Another possible answer is: "abcabcda"
The same letters are at least distance 2 from each other.
```

**My thoughts:** Greedily choose the character with max remaining number for each window k. If no such character satisfy return empty string directly. 

**Solution with Greedy Approach:** [https://repl.it/@trsong/Rearrange-Strings-K-Distance-Apart](https://repl.it/@trsong/Rearrange-Strings-K-Distance-Apart)
```py
import unittest
from Queue import PriorityQueue

def rearrange_string(input_string, k):
    if k <= 0:
        return ""

    histogram = {}
    for ch in input_string:
        histogram[ch] = histogram.get(ch, 0) + 1

    pq = PriorityQueue()
    for ch, count in histogram.items():
        pq.put((-count, ch))

    res = []
    while not pq.empty():
        remaining_chars = []
        for _ in xrange(k):
            if pq.empty() and not remaining_chars:
                break
            elif pq.empty():
                return ""
            
            neg_count, ch = pq.get()
            res.append(ch)
            if abs(neg_count) > 1:
                remaining_chars.append((abs(neg_count)-1, ch))
        
        for count, ch in remaining_chars:
            pq.put((-count, ch))

    return "".join(res)
                

class RearrangeStringSpec(unittest.TestCase):
    def assert_k_distance_apart(self, rearranged_string, original_string, k):
        # Test same length
        self.assertTrue(len(original_string) == len(rearranged_string))

        # Test containing all characters
        self.assertTrue(sorted(original_string) == sorted(rearranged_string))

        # Test K distance apart
        last_occur_map = {}
        for i, c in enumerate(rearranged_string):
            last_occur = last_occur_map.get(c, float('-inf'))
            self.assertTrue(i - last_occur >= k)
            last_occur_map[c] = i
    
    def test_utility_function_is_correct(self):
        original_string = "aaadbbcc"
        k = 2
        ans1 = "abacabcd"
        ans2 = "abcabcda"
        self.assert_k_distance_apart(ans1, original_string, k)
        self.assert_k_distance_apart(ans2, original_string, k)
        self.assertRaises(AssertionError, self.assert_k_distance_apart, original_string, original_string, k)

    def test_example1(self):
        original_string = "aabbcc"
        k = 3
        target_string = rearrange_string(original_string, k)
        self.assert_k_distance_apart(target_string, original_string, k)
    
    def test_example2(self):
        original_string = "aaabc"
        self.assertEqual(rearrange_string(original_string, 3),"")
    
    def test_example3(self):
        original_string = "aaadbbcc"
        k = 2
        target_string = rearrange_string(original_string, k)
        self.assert_k_distance_apart(target_string, original_string, k)
    
    def test_large_distance(self):
        original_string = "abcd"
        k = 10
        rearranged_string = rearrange_string(original_string, k)
        self.assert_k_distance_apart(rearranged_string, original_string, k)

    def test_empty_input_string(self):
        self.assertEqual(rearrange_string("", 1),"")
    
    def test_impossible_to_rearrange(self):
        self.assertEqual(rearrange_string("aaaabbbcc", 3), "")

    def test_k_too_small(self):
        self.assertEqual(rearrange_string("a", 0), "")
        self.assertEqual(rearrange_string("a", -1), "")
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 10, 2020 \[Easy\] Remove Duplicates From Linked List
---
> **Question:** Given a linked list, remove all duplicate values from the linked list.
>
> For instance, given `1 -> 2 -> 3 -> 3 -> 4`, then we wish to return the linked list `1 -> 2 -> 4`.

**Solution:** [https://repl.it/@trsong/Remove-Duplicates-From-Linked-List](https://repl.it/@trsong/Remove-Duplicates-From-Linked-List)
```py
import unittest

def remove_duplicates(lst):
    prev = dummy = ListNode(-1, lst)
    p = lst
    visited = set()
    while p:
        if p.val in visited:
            prev.next = p.next
        else:
            visited.add(p.val)
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

    def __eq__(self, other):
        return other and self.val == other.val and self.next == other.next

    def __repr__(self):
        return "%d -> %s" % (self.val, self.next)

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
        source_list = ListNode.List(1, 2, 3, 3, 4)
        expected = ListNode.List(1, 2, 3, 4)
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

    def test_unsorted_list(self):
        source_list = ListNode.List(2, 1, 1, 2, 1)
        expected = ListNode.List(2, 1)
        self.assertEqual(expected, remove_duplicates(source_list))

    def test_unsorted_list2(self):
        source_list = ListNode.List(3, 2, 1, 3, 2, 1)
        expected = ListNode.List(3, 2, 1)
        self.assertEqual(expected, remove_duplicates(source_list))

    def test_unsorted_list3(self):
        source_list = ListNode.List(1, 1, 1)
        expected = ListNode.List(1)
        self.assertEqual(expected, remove_duplicates(source_list))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 9, 2020 LC 139 \[Medium\] Word Break
---
> **Question:** Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words.
>
> **Note:**
> * The same word in the dictionary may be reused multiple times in the segmentation.
> * You may assume the dictionary does not contain duplicate words.

**Example 1:**
```py
Input: s = "Pseudocode", wordDict = ["Pseudo", "code"]
Output: True
Explanation: Return true because "Pseudocode" can be segmented as "Pseudo code".
```

**Example 2:**
```py
Input: s = "applepenapple", wordDict = ["apple", "pen"]
Output: True
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
             Note that you are allowed to reuse a dictionary word.
```

**Example 3:**
```py
Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
Output: False
```

**My thoughts:** This question feels almost the same as [LC 279 Minimum Number of Squares Sum to N](https://trsong.github.io/python/java/2019/08/02/DailyQuestionsAug.html#sep-22-2019-lc-279-medium-minimum-number-of-squares-sum-to-n). The idea is to think about the problem backwards and you may want to ask yourself: what makes `s[:n]` to be `True`? There must exist some word with length `m` where `m < n` such that `s[:n-m]` is `True` and string `s[n-m:n]` is in the dictionary. Therefore, the problem size shrinks from `n` to  `m` and it will go all the way to empty string which definitely is `True`.

**Solution with DP:** [https://repl.it/@trsong/Word-Break-Problem](https://repl.it/@trsong/Word-Break-Problem)
```py
import unittest

def word_break(s, word_dict):
    n = len(s)
    # Let dp[i] indicates whether s[:i] is breakable
    # dp[i] = dp[i-k] for s[i-k:i] in dictinary with length k
    dp = [False] * (n + 1)
    dp[0] = True
    for i in xrange(1, n + 1):
        for word in word_dict:
            if dp[i]:
                break
            k = len(word)
            dp[i] = i >= k and dp[i - k] and s[i - k:i] == word
    return dp[n]


class WordBreakSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(word_break("Pseudocode", ["Pseudo", "code"]))

    def test_example2(self):
        self.assertTrue(word_break("applepenapple", ["apple", "pen"]))

    def test_example3(self):
        self.assertFalse(
            word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]))

    def test_word_in_dict_is_a_prefix(self):
        self.assertTrue(word_break("123456", ["12", "123", "456"]))

    def test_word_in_dict_is_a_prefix2(self):
        self.assertTrue(word_break("123456", ["12", "123", "456"]))

    def test_word_in_dict_is_a_prefix3(self):
        self.assertFalse(word_break("123456", ["12", "12345", "456"]))

    def test_empty_word(self):
        self.assertTrue(word_break("", ['a']))

    def test_empty_word2(self):
        self.assertTrue(word_break("", []))

    def test_empty_word3(self):
        self.assertFalse(word_break("a", []))

    def test_use_same_word_twice(self):
        self.assertTrue(word_break("aaabaaa", ["aaa", "b"]))

    def test_impossible_word_combination(self):
        self.assertFalse(word_break("aaaaa", ["aaa", "aaaa"]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 8, 2020 LC 13 \[Easy\] Convert Roman Numerals to Decimal
--- 
> **Question:** Given a Roman numeral, find the corresponding decimal value. Inputs will be between 1 and 3999.
> 
> **Note:** Numbers are strings of these symbols in descending order. In some cases, subtractive notation is used to avoid repeated characters. The rules are as follows:
> 1. I placed before V or X is one less, so 4 = IV (one less than 5), and 9 is IX (one less than 10)
> 2. X placed before L or C indicates ten less, so 40 is XL (10 less than 50) and 90 is XC (10 less than 100).
> 3. C placed before D or M indicates 100 less, so 400 is CD (100 less than 500), and 900 is CM (100 less than 1000).

**Example:**
```py
Input: IX
Output: 9

Input: VII
Output: 7

Input: MCMIV
Output: 1904

Roman numerals are based on the following symbols:
I     1
IV    4
V     5
IX    9 
X     10
XL    40
L     50
XC    90
C     100
CD    400
D     500
CM    900
M     1000
```

**Solution:** [https://repl.it/@trsong/Convert-Roman-Format-Number](https://repl.it/@trsong/Convert-Roman-Format-Number)
```py
import unittest

roman_unit = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

def roman_to_decimal(roman_str):
    prev = 0
    res = 0
    for ch in roman_str:
        cur = roman_unit[ch]
        if prev < cur:
            res -= 2 * prev
        res += cur
        prev = cur
    return res 


class RomanToDecimalSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(9, roman_to_decimal("IX"))

    def test_example2(self):
        self.assertEqual(7, roman_to_decimal("VII"))

    def test_example3(self):
        self.assertEqual(1904, roman_to_decimal("MCMIV"))

    def test_boundary(self):
        self.assertEqual(3999, roman_to_decimal("MMMCMXCIX"))

    def test_descending_order_rule1(self):
        self.assertEqual(34, roman_to_decimal("XXXIV"))

    def test_descending_order_rule2(self):
        self.assertEqual(640, roman_to_decimal("DCXL"))

    def test_descending_order_rule3(self):
        self.assertEqual(912, roman_to_decimal("CMXII"))

    def test_all_decending_rules_applied(self):
        self.assertEqual(3949, roman_to_decimal("MMMCMXLIX"))

    def test_all_decending_rules_applied2(self):
        self.assertEqual(2994, roman_to_decimal("MMCMXCIV"))

    def test_all_in_normal_order(self):
        self.assertEqual(1666, roman_to_decimal("MDCLXVI"))

    def test_all_in_normal_order2(self):
        self.assertEqual(3888, roman_to_decimal("MMMDCCCLXXXVIII"))
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 7, 2020 LC 763 \[Medium\] Partition Labels
---
> **Question:** A string S of lowercase English letters is given. We want to partition this string into as many parts as possible so that each letter appears in at most one part, and return a list of integers representing the size of these parts.

**Example:**
```py
Input: S = "ababcbacadefegdehijhklij"
Output: [9, 7, 8]
Explanation: The partition is "ababcbaca", "defegde", "hijhklij". 
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits S into less parts.
```

**Solution:** [https://repl.it/@trsong/Partition-Labels](https://repl.it/@trsong/Partition-Labels)
```py
import unittest

def partion_labels(s):
    last_occur = {ch: pos for pos, ch in enumerate(s)}
    res = []
    cut_pos = -1
    last_cut_pos = -1
    for i, ch in enumerate(s):
        cut_pos = max(cut_pos, last_occur[ch])
        if i == cut_pos:
            window_size = cut_pos - last_cut_pos
            res.append(window_size)
            last_cut_pos = cut_pos
    return res


class PartionLabelSpec(unittest.TestCase):
    def test_example(self):
        s = "ababcbacadefegdehijhklij"
        expected = [9, 7, 8]  # "ababcbaca", "defegde", "hijhklij"
        self.assertEqual(expected, partion_labels(s))

    def test_empty_string(self):
        self.assertEqual([], partion_labels(""))

    def test_one_whole_string(self):
        s = "aaaaa"
        expected = [5]
        self.assertEqual(expected, partion_labels(s))

    def test_string_with_distinct_characters(self):
        s = "abcde"
        expected = [1, 1, 1, 1, 1]
        self.assertEqual(expected, partion_labels(s))

    def test_string_that_cannot_break(self):
        s = "abca"
        expected = [4]
        self.assertEqual(expected, partion_labels(s))

    def test_string_that_cannot_break2(self):
        s = "abaca"
        expected = [5]
        self.assertEqual(expected, partion_labels(s))

    def test_break_as_many_parts(self):
        s = "abacddc"
        expected = [3, 4]
        self.assertEqual(expected, partion_labels(s))

    def test_break_as_many_parts2(self):
        s = "abacdde"
        expected = [3, 1, 2, 1]
        self.assertEqual(expected, partion_labels(s))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 6, 2020 LC 8 \[Medium\] String to Integer (atoi)
---
> **Question:** Given a string, convert it to an integer without using the builtin str function. You are allowed to use ord to convert a character to ASCII code.
>
> Consider all possible cases of an integer. In the case where the string is not a valid integer, return `0`.

**Example:**
```py
atoi('-105')  # -105
```

**Solution:** [https://repl.it/@trsong/String-to-Integer-atoi](https://repl.it/@trsong/String-to-Integer-atoi)
```py
import unittest

INT_MIN = -2 ** 31
INT_MAX = 2 ** 31 - 1

def atoi(s):
    n = len(s)
    pos = 0
    while pos < n and s[pos] == ' ':
        pos += 1
    
    sign = 1
    if pos < n and (s[pos] == '+' or s[pos] == '-'):
        sign = -1 if s[pos] == '-' else 1
        pos += 1

    res = 0
    while pos < n and '0' <= s[pos] <= '9':
        digit = ord(s[pos]) - ord('0')
        if sign > 0 and res > (INT_MAX - digit) // 10:
            return INT_MAX
        if sign < 0 and res <= (INT_MIN + digit) // 10:
            return INT_MIN
        res = 10 * res + sign * digit
        pos += 1

    return res


class AtoiSpec(unittest.TestCase):
    def test_example(self):
        s = '-105'
        expected = -105
        self.assertEqual(expected, atoi(s))

    def test_empty_string(self):
        s = ''
        expected = 0
        self.assertEqual(expected, atoi(s))

    def test_positive_number(self):
        s = '42'
        expected = 42
        self.assertEqual(expected, atoi(s))

    def test_negative_number_with_space(self):
        s = '    -42'
        expected = -42
        self.assertEqual(expected, atoi(s))

    def test_invalid_string_as_suffix(self):
        s = '4123 some words 2'
        expected = 4123
        self.assertEqual(expected, atoi(s))

    def test_invalid_string_as_prefix(self):
        s = 'some prefix and 1'
        expected = 0
        self.assertEqual(expected, atoi(s))

    def test_number_out_of_range(self):
        s = '-91283472332'
        expected = INT_MIN
        self.assertEqual(expected, atoi(s))

    def test_number_out_of_range2(self):
        s = '-2147483649'
        expected = INT_MIN
        self.assertEqual(expected, atoi(s))

    def test_number_out_of_range3(self):
        s = '-2147483647'
        expected = -2147483647
        self.assertEqual(expected, atoi(s))

    def test_number_out_of_range4(self):
        s = '2147483647'
        expected = INT_MAX
        self.assertEqual(expected, atoi(s))

    def test_number_out_of_range5(self):
        s = '2147483648'
        expected = INT_MAX
        self.assertEqual(expected, atoi(s))

    def test_number_out_of_range6(self):
        s = '2147483646'
        expected = 2147483646
        self.assertEqual(expected, atoi(s))

    def test_prefix_with_zero(self):
        s = '   00012345'
        expected = 12345
        self.assertEqual(expected, atoi(s))

    def test_invalid_prefix(self):
        s = '+-2'
        expected = 0
        self.assertEqual(expected, atoi(s))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Sep 5, 2020 LC 54 \[Medium\] Spiral Matrix 
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

**Solution:** [https://repl.it/@trsong/Spiral-Matrix-Traversal](https://repl.it/@trsong/Spiral-Matrix-Traversal)
```py
import unittest

def spiral_order(matrix):
    if not matrix or not matrix[0]:
        return []
    r_min, r_max = 0, len(matrix)-1
    c_min, c_max = 0, len(matrix[0])-1
    res = []

    while r_min <= r_max and c_min <= c_max:
        r, c = r_min, c_min

        # Move right
        while c < c_max:
            res.append(matrix[r][c])
            c += 1
        c_max -= 1

        # Move down
        while r < r_max:
            res.append(matrix[r][c])
            r += 1
        r_max -= 1

        # Is stuck and can neither move left nor up
        if c_min > c_max or r_min > r_max:
            res.append(matrix[r][c])
            break

        # Move left
        while c > c_min:
            res.append(matrix[r][c])
            c -= 1
        c_min += 1

        # Move up
        while r > r_min:
            res.append(matrix[r][c])
            r -= 1
        r_min += 1

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
        self.assertEqual(spiral_order([]), [])
        self.assertEqual(spiral_order([[]]), [])

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
    unittest.main(exit=False)
```

### Sep 4, 2020 \[Medium\] Zig-Zag String
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

**Solution:** [https://repl.it/@trsong/Print-Zig-Zag-String](https://repl.it/@trsong/Print-Zig-Zag-String)
```py
import unittest

def zig_zag_format(sentence, k):
    if not sentence:
        return []
    elif k == 1:
        return [sentence]

    res = [[] for _ in xrange(min(k, len(sentence)))]
    direction = -1
    row = 0

    for i, ch in enumerate(sentence):
        if i < k:
            space = i
        elif direction > 0:
            space = 2 * row - 1
        else:
            space = 2 * (k - row - 1) - 1
        res[row].append(space * " " + ch)

        if row == 0 or row == k-1:
            direction *= -1
        row += direction

    return map(lambda line: "".join(line), res)


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
    unittest.main(exit=False)
```


### Sep 3, 2020 \[Easy\] Level of tree with Maximum Sum
---
> **Question:** Given a binary tree, find the level in the tree where the sum of all nodes on that level is the greatest.

**Example:**
```py
The following tree should return level 1:
    1          Level 0 - Sum: 1
   / \
  4   5        Level 1 - Sum: 9 
 / \ / \
3  2 4 -1      Level 2 - Sum: 8
```

**Solution with DFS:** [https://repl.it/@trsong/Find-Level-of-tree-with-Maximum-Sum](https://repl.it/@trsong/Find-Level-of-tree-with-Maximum-Sum)
```py
import unittest

def max_sum_tree_level(tree):
    if not tree:
        return -1
        
    sum_by_row = []
    stack = [(tree, 0)]

    while stack:
        cur, depth = stack.pop()
        if not cur:
            continue
        if depth >= len(sum_by_row):
            sum_by_row.append(0)
        sum_by_row[depth] += cur.val
        stack.extend([(cur.right, depth+1), (cur.left, depth+1)])

    max_index, _ = max(enumerate(sum_by_row), key=lambda kv: kv[1])
    return max_index


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class MaxSumTreeLevelSpec(unittest.TestCase):
    def test_example(self):
        """
            1          Level 0 - Sum: 1
           / \
          4   5        Level 1 - Sum: 9 
         / \ / \
        3  2 4 -1      Level 2 - Sum: 8
        """
        n4 = TreeNode(4, TreeNode(3), TreeNode(2))
        n5 = TreeNode(5, TreeNode(4), TreeNode(-1))
        root = TreeNode(1, n4, n5)
        self.assertEqual(1, max_sum_tree_level(root))

    def test_empty_tree(self):
        self.assertEqual(-1, max_sum_tree_level(None))

    def test_tree_with_one_node(self):
        root = TreeNode(42)
        self.assertEqual(0, max_sum_tree_level(root))
    
    def test_unbalanced_tree(self):
        """
            20
           / \
          8   22
         / \
        4  12
           / \
         10  14
        """
        n4 = TreeNode(4)
        n10 = TreeNode(10)
        n14 = TreeNode(14)
        n22 = TreeNode(22)
        n12 = TreeNode(12, n10, n14)
        n8 = TreeNode(8, n4, n12)
        n20 = TreeNode(20, n8, n22)
        self.assertEqual(1, max_sum_tree_level(n20))
    
    def test_zigzag_tree(self):
        """
        1
         \
          5
         /
        2 
         \
          3
        """
        n3 = TreeNode(3)
        n2 = TreeNode(2, right=n3)
        n5 = TreeNode(5, n2)
        n1 = TreeNode(1, right=n5)
        self.assertEqual(1, max_sum_tree_level(n1))

    def test_tree_with_negative_values(self):
        """
             -1
            /  \
          -2   -3
          /    /
        -1   -6
        """
        left_tree = TreeNode(-2, TreeNode(-1))
        right_tree = TreeNode(-3, TreeNode(-6))
        root = TreeNode(-1, left_tree, right_tree)
        self.assertEqual(0, max_sum_tree_level(root))

    def test_tree_with_negative_values_and_zeros(self):
        """
        -1
          \
           0
            \
            -2
        """
        tree = TreeNode(-1, right=TreeNode(0, right=TreeNode(-2)))
        self.assertEqual(1, max_sum_tree_level(tree))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Sep 2, 2020 \[Easy\] Deepest Node in a Binary Tree
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

**Solution with BFS:** [https://repl.it/@trsong/Find-Deepest-Node-in-a-Binary-Tree](https://repl.it/@trsong/Find-Deepest-Node-in-a-Binary-Tree)
```py
import unittest
from functools import reduce

def find_deepest_node(root):
    it = bfs_traversal(root)
    return reduce(lambda _, x: x, it, None)


def bfs_traversal(root):
    queue = [root]
    while queue:
        cur = queue.pop(0)
        if not cur:
            continue
        yield cur
        queue.extend([cur.left, cur.right])
        

class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return "Node(%d)" % self.val


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

### Sep 1, 2020 LC 230 \[Medium\] Kth Smallest Element in a BST
---
> **Question:** Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

**Example 1:**

```py
Input: 
   3
  / \
 1   4
  \
   2

k = 1
Output: 1
```

**Example 2:**
```py
Input: 
       5
      / \
     3   6
    / \
   2   4
  /
 1

k = 3
Output: 3
```

**Solution with In-order Traversal:** [https://repl.it/@trsong/Find-Kth-Smallest-Element-in-a-BST](https://repl.it/@trsong/Find-Kth-Smallest-Element-in-a-BST)
```py
import unittest

def kth_smallest(root, k):
    it = in_order_traversal(root)
    for _ in xrange(k):
        v = next(it)
    return v


def in_order_traversal(root):
    p = root
    stack = []
    while p or stack:
        if p:
            stack.append(p)
            p = p.left
        else:
            p = stack.pop()
            yield p.val
            p = p.right
    

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class KthSmallestSpec(unittest.TestCase):
    def test_example1(self):
        """
           3
          / \
         1   4
          \
           2
        """
        n1 = TreeNode(1, right=TreeNode(2))
        tree = TreeNode(3, n1, TreeNode(4))
        check_expected = [1, 2, 3, 4]
        for e in check_expected:
            self.assertEqual(e, kth_smallest(tree, e))

    def test_example2(self):
        """
              5
             / \
            3   6
           / \
          2   4
         /
        1
        """
        n2 = TreeNode(2, TreeNode(1))
        n3 = TreeNode(3, n2, TreeNode(4))
        tree = TreeNode(5, n3, TreeNode(6))
        check_expected = [1, 2, 3, 4, 5, 6]
        for e in check_expected:
            self.assertEqual(e, kth_smallest(tree, e))

    def test_full_BST(self):
        """
             4
           /   \
          2     6
         / \   / \
        1   3 5   7
        """
        n2 = TreeNode(2, TreeNode(1), TreeNode(3))
        n6 = TreeNode(6, TreeNode(5), TreeNode(7))
        tree = TreeNode(4, n2, n6)
        check_expected = [1, 2, 3, 4, 5, 6, 7]
        for e in check_expected:
            self.assertEqual(e, kth_smallest(tree, e))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 31, 2020 \[Easy\] Depth of Binary Tree in Peculiar String Representation
---
> **Question:** You are given a binary tree in a peculiar string representation. Each node is written in the form `(lr)`, where `l` corresponds to the left child and `r` corresponds to the right child.
>
> If either `l` or `r` is null, it will be represented as a zero. Otherwise, it will be represented by a new `(lr)` pair.
> 
> Given this representation, determine the depth of the tree.

**Here are a few examples:**
```py
A root node with no children: (00)
A root node with two children: ((00)(00))
An unbalanced tree with three consecutive left children: ((((00)0)0)0)
```


**Solution:** [https://repl.it/@trsong/Depth-of-Binary-Tree-in-Peculiar-String-Representation](https://repl.it/@trsong/Depth-of-Binary-Tree-in-Peculiar-String-Representation)
```py
import unittest

def depth(tree_str):
    depth_delta = {
        '(': 1,
        ')': -1,
        '0': 0
    }
    max_accu = 0
    accu = 0
    for c in tree_str:
        accu += depth_delta[c]
        max_accu = max(max_accu, accu)
    return max_accu


class DepthSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertEqual(0, depth("0"))
    
    def test_one_node_without_child(self):
        self.assertEqual(1, depth("(00)"))
    
    def test_one_node_with_two_children(self):
        """
          .
         / \
        .   .
        """
        self.assertEqual(2, depth("((00)(00))"))

    def test_left_heavy_tree(self):
        """
              .
             / \
            .   .
           / \
          .   .
         /
        . 
        """
        node = "(00)"
        ll = "(%s0)" % node
        l = "(%s%s)" % (ll, node)
        root = "(%s0)" % l
        self.assertEqual(4, depth(root))

    def test_right_heavy_tree(self):
        """
            .
           / \
          .   .
         / \   \
        .   .   .
               / \
              .   .
        """ 
        node2 = "((00)(00))"
        r = "(0%s)" % node2
        root = "(%s%s)" % (node2, r)
        self.assertEqual(4, depth(root))

    def test_zig_zag_tree(self):
        """
          .
         /
        .
         \
          .
         /
        .
         \
          .
        """
        node = "(00)"
        make_left = lambda n: "(%s0)" % n
        make_right = lambda n: "(0%s)" % n
        lrl = make_right(node)
        lr = make_left(lrl)
        l = make_right(lr)
        root = make_left(l)
        self.assertEqual(5, depth(root))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 30, 2020 LC 166 \[Medium\] Fraction to Recurring Decimal
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

**Solution:** [https://repl.it/@trsong/Convert-Fraction-to-Recurring-Decimal](https://repl.it/@trsong/Convert-Fraction-to-Recurring-Decimal)
```py
import unittest

def fraction_to_decimal(numerator, denominator):
    if numerator == 0:
        return "0"

    sign = "-" if (numerator > 0) ^ (denominator > 0) else ""
    numerator, denominator = abs(numerator), abs(denominator)
    quotient, remainder = numerator // denominator, numerator % denominator
    if remainder == 0:
        return "%s%d" % (sign, quotient)
    
    decimals = []
    remainder_history = {}
    i = 0

    while remainder != 0 and remainder not in remainder_history:
        remainder_history[remainder] = i
        i += 1

        remainder *= 10
        decimals.append(remainder // denominator)
        remainder %= denominator


    if remainder == 0:
        return "%s%d.%s" % (sign, quotient, "".join(map(str, decimals)))
    else:
        repeated_start = remainder_history[remainder]
        non_repeat_decimals = "".join(map(str, decimals[:repeated_start]))
        repeated_decimals = "".join(map(str, decimals[repeated_start:]))
        return "%s%d.%s(%s)" % (sign, quotient, non_repeat_decimals, repeated_decimals)


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

### Aug 29, 2020 \[Easy\] Tree Isomorphism Problem
---
> **Question:** Write a function to detect if two trees are isomorphic. Two trees are called isomorphic if one of them can be obtained from other by a series of flips, i.e. by swapping left and right children of a number of nodes. Any number of nodes at any level can have their children swapped. Two empty trees are isomorphic.


**Example:** 
```py
The following two trees are isomorphic with following sub-trees flipped: 2 and 3, NULL and 6, 7 and 8.

Tree1:
     1
   /   \
  2     3
 / \   /
4   5 6
   / \
  7   8

Tree2:
   1
 /   \
3     2
 \   / \
  6 4   5
       / \
      8   7
```

**Solution:** [https://repl.it/@trsong/Is-Binary-Tree-Isomorphic](https://repl.it/@trsong/Is-Binary-Tree-Isomorphic)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_isomorphic(t1, t2):
    if not t1 and not t2:
        return True
    
    if not t1 or not t2:
        return False
    
    if t1.val != t2.val:
        return False

    return (is_isomorphic(t1.left, t2.left)
            and is_isomorphic(t1.right, t2.right)
            or is_isomorphic(t1.right, t2.left)
            and is_isomorphic(t1.left, t2.right))


class IsIsomorphicSpec(unittest.TestCase):
    def test_example(self):
        """
        Tree1:
             1
           /   \
          2     3
         / \   /
        4   5 6
           / \
          7   8

        Tree2:
           1
         /   \
        3     2
         \   / \
          6 4   5
               / \
              8   7
        """
        p5 = TreeNode(5, TreeNode(7), TreeNode(8))
        p2 = TreeNode(2, TreeNode(4), p5)
        p3 = TreeNode(3, TreeNode(6))
        p1 = TreeNode(1, p2, p3)

        q5 = TreeNode(5, TreeNode(8), TreeNode(7))
        q2 = TreeNode(2, TreeNode(4), q5)
        q3 = TreeNode(3, right=TreeNode(6))
        q1 = TreeNode(1, q3, q2)

        self.assertTrue(is_isomorphic(p1, q1))

    def test_empty_trees(self):
        self.assertTrue(is_isomorphic(None, None))

    def test_empty_vs_nonempty_trees(self):
        self.assertFalse(is_isomorphic(None, TreeNode(1)))

    def test_same_tree_val(self):
        """
        Tree1:
        1
         \
          1
         /
        1 

        Tree2:
            1
           /
          1
           \
            1
        """
        t1 = TreeNode(1, right=TreeNode(1, TreeNode(1)))
        t2 = TreeNode(1, TreeNode(1, right=TreeNode(1)))
        self.assertTrue(is_isomorphic(t1, t2))

    def test_same_val_yet_not_isomorphic(self):
        """
        Tree1:
          1
         / \
        1   1
           / \
          1   1

        Tree2:
            1
           / \
          1   1
         /     \
        1       1
        """
        t1 = TreeNode(1, TreeNode(1, TreeNode(1), TreeNode(1)))
        t2 = TreeNode(1, TreeNode(1, TreeNode(1)),
                      TreeNode(1, right=TreeNode(1)))
        self.assertFalse(is_isomorphic(t1, t2))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 28, 2020  LC 86 \[Medium\] Partitioning Linked List
---
> **Question:** Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
>
> You should preserve the original relative order of the nodes in each of the two partitions.

**Example:**
```py
Input: head = 1->4->3->2->5->2, x = 3
Output: 1->2->2->4->3->5
```

**Solution:** [https://repl.it/@trsong/Partitioning-Singly-Linked-List](https://repl.it/@trsong/Partitioning-Singly-Linked-List)
```py
import unittest

def partition(lst, target):
    p1 = dummy1 = Node(-1)
    p2 = dummy2 = Node(-1)

    while lst:
        if lst.val < target:
            p1.next = lst
            p1 = p1.next
        else:
            p2.next = lst
            p2 = p2.next
        lst = lst.next
    
    p2.next = None
    p1.next = dummy2.next
    return dummy1.next
            

##############################
# Below are testing utilities
##############################
class Node(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    @staticmethod
    def flatten(lst):
        res = []
        while lst:
            res.append(lst.val)
            lst = lst.next
        return res

    @staticmethod
    def create(*vals):
        t = dummy = Node(-1)
        for v in vals:
            t.next = Node(v)
            t = t.next
        return dummy.next

class PartitionSpec(unittest.TestCase):
    def assert_result(self, expected_list, result_list, target):
        expected_arr = Node.flatten(expected_list)
        result_arr = Node.flatten(result_list)
        
        self.assertEqual(len(expected_arr), len(result_arr))
        split_index = 0
        for i in xrange(len(expected_arr)):
            if expected_arr[i] >= target:
                split_index = i
                break
        e1, e2 = set(expected_arr[:split_index]), set(expected_arr[split_index:]) 
        r1, r2 = set(result_arr[:split_index]), set(result_arr[split_index:]) 
    
        self.assertEqual(e1, r1)
        self.assertEqual(e2, r2)

    def test_example(self):
        original  = Node.create(1, 4, 3, 2, 5, 2)
        expected = Node.create(1, 2, 2, 4, 3, 5)
        target = 3
        self.assert_result(expected, partition(original, target), target)

    def test_empty_list(self):
        self.assertIsNone(partition(None, 42))

    def test_one_element_list(self):
        original  = Node.create(1)
        expected = Node.create(1)
        target = 0
        self.assert_result(expected, partition(original, target), target)
        target = 1
        self.assert_result(expected, partition(original, target), target)

    def test_list_with_duplicated_elements(self):
        original  = Node.create(1, 1, 0, 1, 1)
        expected = Node.create(0, 1, 1, 1, 1)
        target = 1
        self.assert_result(expected, partition(original, target), target)

    def test_list_with_duplicated_elements2(self):
        original  = Node.create(0, 2, 0, 2, 0)
        expected = Node.create(0, 0, 0, 2, 2)
        target = 1
        self.assert_result(expected, partition(original, target), target)

    def test_list_with_duplicated_elements3(self):
        original  = Node.create(1, 1, 1, 1)
        expected = Node.create(1, 1, 1, 1)
        target = 2
        self.assert_result(expected, partition(original, target), target)

    def test_unsorted_array(self):
        original  = Node.create(10, 4, 20, 10, 3)
        expected = Node.create(3, 10, 4, 20, 10)
        target = 3
        self.assert_result(expected, partition(original, target), target)

    def test_unsorted_array2(self):
        original  = Node.create(1, 4, 3, 2, 5, 2, 3)
        expected = Node.create(1, 2, 2, 3, 3, 4, 5)
        target = 3
        self.assert_result(expected, partition(original, target), target)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 27, 2020 LC 253 \[Easy\] Minimum Lecture Rooms
---
> **Questions:** Given an array of time intervals `(start, end)` for classroom lectures (possibly overlapping), find the minimum number of rooms required.
>
> For example, given `[(30, 75), (0, 50), (60, 150)]`, you should return `2`.

**My thoughts:** whenever we enter an interval at time t, the total number of room at time t increment by 1 and whenever we leave an interval the total number of required room decrement by 1. For `(1, 10)`, `(5, 15)` and `(6, 15)`, `+1` at `t = 1, 5, 6` and `-1` at `t=10, 15, 15`. And at peak hour, the total room equals 3. 

**Solution:** [https://repl.it/@trsong/Minimum-Required-Lecture-Rooms](https://repl.it/@trsong/Minimum-Required-Lecture-Rooms)
```py
import unittest

def min_lecture_rooms(intervals):
    start_times = map(lambda t: (t[0], 1), intervals)
    end_times = map(lambda t: (t[1], -1), intervals)
    times = sorted(start_times + end_times)

    accu_rooms = 0
    max_rooms = 0
    for _, room_diff in times:
        accu_rooms += room_diff
        max_rooms = max(max_rooms, accu_rooms)
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
    unittest.main(exit=False)
```

### Aug 26, 2020 LC 1171 \[Medium\] Remove Consecutive Nodes that Sum to 0
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


**Solution:** [https://repl.it/@trsong/Remove-Consecutive-Nodes-that-Sum-to-Zero](https://repl.it/@trsong/Remove-Consecutive-Nodes-that-Sum-to-Zero)
```py
import unittest

def remove_zero_sum_sublists(head):
    dummy = ListNode(-1, head)
    prefix_sum_lookup = {0: dummy}
    p = dummy.next
    accu_sum = 0
    while p:
        accu_sum += p.val
        if accu_sum not in prefix_sum_lookup:
            prefix_sum_lookup[accu_sum] = p
        else:
            loop_start = prefix_sum_lookup[accu_sum]
            q = loop_start.next
            accu_sum_to_remove = accu_sum
            while q != p:
                accu_sum_to_remove += q.val
                del prefix_sum_lookup[accu_sum_to_remove]
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
    unittest.main(exit=False)
```


### Aug 25, 2020 LC 796 \[Easy\] Shift-Equivalent Strings
---
> **Question:** Given two strings A and B, return whether or not A can be shifted some number of times to get B.
>
> For example, if A is `'abcde'` and B is `'cdeab'`, return `True`. If A is `'abc'` and B is `'acb'`, return `False`.

**My thoughts:** It should be pretty easy to come up with non-linear time complexity solution. But for linear, I can only come up w/ rolling hash solution. The idea is to treat each digit as a number. For example, `"1234"` is really `1234`, each time we move the most significant bit to right by `(1234 - 1 * 10^3) * 10 + 1 = 2341`. In general, we can treat `'abc'` as numeric value of `abc` base `p0` ie. `a * p0^2 + b * p0^1 + c * p0^0` and in order to prevent overflow, we use a larger prime number which I personally prefer 666667 (easy to remember), `'abc' =>  (a * p0^2 + b * p0^1 + c * p0^0) % p1 where p0 and p1 are both prime and p0 is much smaller than p1`.

**Solution with Roling Hash:** [https://repl.it/@trsong/Check-if-Strings-are-Shift-Equivalent](https://repl.it/@trsong/Check-if-Strings-are-Shift-Equivalent)
```py
import unittest
from functools import reduce

P0 = 23  # small prime number
P1 = 666667 # larger prime number

def hash(s):
    rolling_hash = lambda accu, ch: (accu * P0 % P1 + ord(ch)) % P1
    return reduce(rolling_hash, s, 0)


def is_shift_eq(source, target):
    if len(source) != len(target):
        return False
    elif source == target:
        return True
    
    source_hash = hash(source)
    target_hash = hash(target)
    n = len(source)
    leftmost_digit_base = reduce(lambda accu, _: (accu * P0) % P1, xrange(n-1), 1)

    for ch in source:
        ord_ch = ord(ch)
        leftmost_digit = (ord_ch * leftmost_digit_base) % P1
        source_hash = ((source_hash - leftmost_digit) * P0 + ord_ch) % P1
        if source_hash == target_hash:
            return True
    
    return False


class IsShiftEqSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(is_shift_eq('abcde', 'cdeab'))

    def test_example2(self):
        self.assertFalse(is_shift_eq('abc', 'acb'))

    def test_different_length_strings(self):
        self.assertFalse(is_shift_eq(' a ', ' a'))

    def test_empty_strings(self):
        self.assertTrue(is_shift_eq('', ''))

    def test_string_with_unique_word(self):
        self.assertTrue(is_shift_eq('aaaaa', 'aaaaa'))

    def test_string_with_multiple_spaces(self):
        self.assertFalse(is_shift_eq('aa aa aa', 'aaaa  aa'))

    def test_number_strins(self):
        self.assertTrue(is_shift_eq("567890", "890567"))

    def test_large_string_performance_test(self):
        N = 100000
        source = str(range(N))
        target = source[:N//2] + source[N//2:]
        self.assertTrue(is_shift_eq(source, target))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 24, 2020 LC 560 \[Medium\] Subarray Sum Equals K
---
> **Question:** Given a list of integers and a number `K`, return which contiguous elements of the list sum to `K`.
>
> For example, if the list is `[1, 2, 3, 4, 5]` and `K` is `9`, then it should return `[2, 3, 4]`, since `2 + 3 + 4 = 9`.


**My thoughts:** Store prefix sum along the way to find how many index i exists such that `prefix[j] - prefix[i] = k`. As `j > i`, when we reach j, we pass i already, so we can store `prefix[i]` in a map and put value as occurance of `prefix[i]`, that is why this question feels similar to Two Sum question.

**Solution:** [https://repl.it/@trsong/Find-Number-of-Sub-array-Sum-Equals-K](https://repl.it/@trsong/Find-Number-of-Sub-array-Sum-Equals-K)
```py
import unittest

def subarray_sum(nums, k):
    prefix_sum = 0
    prefix_count = {0: 1}
    res = 0
    for num in nums:
        prefix_sum += num
        target = prefix_sum - k
        res += prefix_count.get(target, 0)
        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1
    return res


class SubarraySumSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(subarray_sum([1, 1, 1], 2), 2)  # [1, 1] and [1, 1]
    
    def test_empty_array(self):
        self.assertEqual(subarray_sum([], 2), 0) 

    def test_target_is_zero(self):
        self.assertEqual(subarray_sum([0, 0], 0), 3) # [0], [0], [0, 0]

    def test_array_with_one_elem(self):
        self.assertEqual(subarray_sum([1], 0), 0)
    
    def test_array_with_one_elem2(self):
        self.assertEqual(subarray_sum([1], 1), 1) # [1]

    def test_array_with_unique_target_prefix(self):
        # suppose the prefix_sum = [1, 2, 3, 3, 2, 1]
        self.assertEqual(subarray_sum([1, 1, 1, 0, -1, -1], 2), 4)  # [1, 1], [1, ,1], [1, 1, 0], [1, 1, 1, 0, -1]


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 23, 2020 LC 29 \[Medium\] Divide Two Integers
---
> **Question:** Implement integer division without using the division operator. Your function should return a tuple of `(dividend, remainder)` and it should take two numbers, the product and divisor.
>
> For example, calling `divide(10, 3)` should return `(3, 1)` since the divisor is `3` and the remainder is `1`.

**My thoughts:** Left shift `x << i` is just `x * (2 ** i)`, we can take advantage of that to figure out each digit of the quotient. And find pattern from following to handle special case when operands are negative.
```py
divide(1, 3) => (0, 1)    # 0*3+1 = 1
divide(-1, 3) => (-1, 2)  # -1*3+2 = -1
divide(1, -3) => (-1, -2) # -1*-3-2= 1
divide(-1, -3) => (0, -1) # 0*-3-1= -1
```

**Solution:** [https://repl.it/@trsong/Divide-Two-Integers](https://repl.it/@trsong/Divide-Two-Integers)
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
    unittest.main(exit=False)
```


### Aug 22, 2020 LC 451 \[Easy\] Sort Characters By Frequency
---
> **Question:** Given a string, sort it in decreasing order based on the frequency of characters. If there are multiple possible solutions, return any of them.
>
> For example, given the string `tweet`, return `tteew`. `eettw` would also be acceptable.


**Solution with Counting Sort:** [https://repl.it/@trsong/Sort-Characters-By-Frequency](https://repl.it/@trsong/Sort-Characters-By-Frequency)
```py
import unittest

def string_ordered_by_frequency(s):
    char_freq = {}
    for c in s:
        char_freq[c] = char_freq.get(c, 0) + 1

    n = len(s)
    counting = [None] * (n + 1)
    for c, freq in char_freq.items():
        if counting[freq] is None:
            counting[freq] = []
        counting[freq].append(c)
    
    res = []
    for freq in xrange(n, 0, -1):
        if counting[freq] is None:
            continue
        
        for c in counting[freq]:
            res.append(c * freq)

    return ''.join(res)
           

class StringOrderedByFrequencySpec(unittest.TestCase):
    def test_example(self):
        s = 'tweet'
        expected = ['tteew', 'eettw']
        self.assertIn(string_ordered_by_frequency(s), expected)

    def test_empty_string(self):
        self.assertEqual('', string_ordered_by_frequency(''))

    def test_contains_upper_and_lower_letters(self):
        s = 'aAbb'
        expected = ['bbaA', 'bbAa']
        self.assertIn(string_ordered_by_frequency(s), expected)

    def test_letter_with_differnt_frequency(self):
        s = '241452345534535'
        expected = '555554444333221'
        self.assertEqual(expected, string_ordered_by_frequency(s))

    def test_string_with_unique_letter(self):
        s = 'aaaaaaa'
        self.assertEqual(s, string_ordered_by_frequency(s))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 21, 2020 \[Easy\] Make the Largest Number
---
> **Question:** Given a number of integers, combine them so it would create the largest number.

**Example:**
```py
Input: [17, 7, 2, 45, 72]
Output: 77245217
```

**Solution with Custom Sort:** [https://repl.it/@trsong/Construct-Largest-Number](https://repl.it/@trsong/Construct-Largest-Number)
```py
import unittest

def construct_largest_number(nums):
    if not nums:
        return 0

    negative_nums = filter(lambda x: x < 0, nums)
    filtered_nums = filter(lambda x: x >= 0, nums)
    str_nums = map(str, filtered_nums)
    combined_num_cmp = lambda s1, s2: cmp(s1 + s2, s2 + s1)

    if negative_nums:
        str_nums.sort(combined_num_cmp)
        return int(str(negative_nums[0]) + "".join(str_nums))
    else:
        str_nums.sort(combined_num_cmp, reverse=True)
        return int("".join(str_nums))


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
    unittest.main(exit=False)
```

### Aug 20, 2020 \[Medium\] Tree Serialization
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

**Solution with Pre-order Traversal**: [https://repl.it/@trsong/Serialize-and-Deserialize-the-Binary-Tree](https://repl.it/@trsong/Serialize-and-Deserialize-the-Binary-Tree)
```py
import unittest

class BinaryTreeSerializer(object):
    @staticmethod
    def serialize(root):
        res = []
        stack = [root]
        while stack:
            cur = stack.pop()
            if cur is None:
                res.append('#')
            else:
                res.append(str(cur.val))
                stack.extend([cur.right, cur.left])
        return ' '.join(res)

    @staticmethod
    def deserialize(s):
        tokens = iter(s.split())
        return BinaryTreeSerializer.build_tree_recur(tokens)

    @staticmethod
    def build_tree_recur(stream):
        raw_value = next(stream)
        if raw_value == '#':
            return None
        
        left_child = BinaryTreeSerializer.build_tree_recur(stream)
        right_child = BinaryTreeSerializer.build_tree_recur(stream)
        return TreeNode(int(raw_value), left_child, right_child)


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
    unittest.main(exit=False)
```

### Aug 19, 2020 LC 91 \[Medium\] Decode Ways
---
> **Question:** A message containing letters from `A-Z` is being encoded to numbers using the following mapping:
>
> ```py
> 'A' -> 1
> 'B' -> 2
> ...
> 'Z' -> 26
> ```
> Given an encoded message containing digits, determine the total number of ways to decode it.

**Example 1:**
```py
Input: "12"
Output: 2
Explanation: It could be decoded as AB (1 2) or L (12).
```

**Example 2:**
```py
Input: "10"
Output: 1
```

**My thoughts:** This question can be solved w/ DP. Similar to the climb stair problem, `DP[n] = DP[n-1] + DP[n-2]` under certain conditions. If last digits can form a number, i.e. `1 ~ 9` then `DP[n] = DP[n-1]`. And if last two digits can form a number, i.e. `10 ~ 26`, then `DP[n] = DP[n-2]`. If we consider both digits, we will have 

```py
dp[k] = dp[k-1] if str[k-1] can form a number. i.e not zero, 1-9
       + dp[k-2] if str[k-2] and str[k-1] can form a number. 10-26
```

**Solution with DP:** [https://repl.it/@trsong/Number-of-Decode-Ways](https://repl.it/@trsong/Number-of-Decode-Ways)
```py
import unittest

def decode_ways(encoded_string):
    if not encoded_string or encoded_string[0] == '0':
        return 0

    n = len(encoded_string)
    # Let dp[i] represents # of decode ways for encoded_string[:i]
    # dp[i] = dp[i-1]               if digit at i-1 is not zero
    # or    = dp[i-1] + dp[i-2]     if digits at i-1,i-2 is valid 26 char
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    ord_zero = ord('0')
    for i in xrange(2, n + 1):
        first_digit = ord(encoded_string[i - 2]) - ord_zero
        second_digit = ord(encoded_string[i - 1]) - ord_zero
        if second_digit > 0:
            dp[i] += dp[i - 1]
        if 10 <= 10 * first_digit + second_digit <= 26:
            dp[i] += dp[i-2]
    return dp[n]


class DecodeWaySpec(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(0, decode_ways(""))

    def test_invalid_string(self):
        self.assertEqual(0, decode_ways("0"))

    def test_length_one_string(self):
        self.assertEqual(1, decode_ways("2"))

    def test_length_one_string2(self):
        self.assertEqual(1, decode_ways("9"))

    def test_length_two_string(self):
        self.assertEqual(1, decode_ways("20"))  # 20

    def test_length_two_string2(self):
        self.assertEqual(2, decode_ways("19"))  # 1,9 and 19

    def test_length_three_string(self):
        self.assertEqual(3, decode_ways("121"))  # 1, 20 and 12, 0

    def test_length_three_string2(self):
        self.assertEqual(1, decode_ways("120"))  # 1, 20

    def test_length_three_string3(self):
        self.assertEqual(1, decode_ways("209"))  # 20, 9

    def test_length_three_string4(self):
        self.assertEqual(2, decode_ways("912"))  # 9,1,2 and 9,12

    def test_length_three_string5(self):
        self.assertEqual(2, decode_ways("231"))  # 2,3,1 and 23, 1

    def test_length_three_string6(self):
        self.assertEqual(3, decode_ways("123"))  # 1,2,3, and 1, 23 and 12, 3

    def test_length_four_string(self):
        self.assertEqual(3, decode_ways("1234"))

    def test_length_four_string2(self):
        self.assertEqual(5, decode_ways("1111"))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 18, 2020 \[Easy\] Merge Overlapping Intervals
---
> **Question:** Given a list of possibly overlapping intervals, return a new list of intervals where all overlapping intervals have been merged.
>
> The input list is not necessarily ordered in any way.
>
> For example, given `[(1, 3), (5, 8), (4, 10), (20, 25)]`, you should return `[(1, 3), (4, 10), (20, 25)]`.

**My thoughts:** Sort intervals based on start time, then for each consecutive intervals s1, s2 the following could occur:
- `s1.end < s2.start`, we append s2 to result
- `s2.start <= s1.end < s2.end`, we merge s1 and s2
- `s2.end <= s1.end`, s1 overlaps all of s2, we do nothing

Note: as all intervals are sorted based on start time, `s1.start <= s2.start`

**Solution with Stack:** [https://repl.it/@trsong/Merge-All-Overlapping-Intervals](https://repl.it/@trsong/Merge-All-Overlapping-Intervals)
```py
import unittest

def merge_intervals(interval_seq):
    interval_seq.sort(key=lambda interval: interval[0])
    stack = []
    for start, end in interval_seq:
        if not stack or stack[-1][1] < start:
            stack.append((start, end))
        elif stack[-1][1] < end:
            prev_start, _ = stack.pop()
            stack.append((prev_start, end))
    return stack


class MergeIntervalSpec(unittest.TestCase):
    def test_interval_with_zero_mergings(self):
        self.assertItemsEqual(merge_intervals([]), [])

    def test_interval_with_zero_mergings2(self):
        interval_seq = [(1, 2), (3, 4), (5, 6)]
        expected = [(1, 2), (3, 4), (5, 6)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))

    def test_interval_with_zero_mergings3(self):
        interval_seq = [(-3, -2), (5, 6), (1, 4)]
        expected = [(-3, -2), (1, 4), (5, 6)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))

    def test_interval_with_one_merging(self):
        interval_seq = [(1, 3), (5, 7), (7, 11), (2, 4)]
        expected = [(1, 4), (5, 11)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))

    def test_interval_with_one_merging2(self):
        interval_seq = [(1, 4), (0, 8)]
        expected = [(0, 8)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))

    def test_interval_with_two_mergings(self):
        interval_seq = [(1, 3), (3, 5), (5, 8)]
        expected = [(1, 8)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))

    def test_interval_with_two_mergings2(self):
        interval_seq = [(5, 8), (1, 6), (0, 2)]
        expected = [(0, 8)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))

    def test_interval_with_multiple_mergings(self):
        interval_seq = [(-5, 0), (1, 4), (1, 4), (1, 4), (5, 7), (6, 10), (0, 1)]
        expected = [(-5, 4), (5, 10)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))

    def test_interval_with_multiple_mergings2(self):
        interval_seq = [(1, 3), (5, 8), (4, 10), (20, 25)]
        expected = [(1, 3), (4, 10), (20, 25)]
        self.assertItemsEqual(expected, merge_intervals(interval_seq))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 17, 2020 \[Medium\] Overlapping Rectangles
--- 
> **Question:** You’re given 2 over-lapping rectangles on a plane. For each rectangle, you’re given its bottom-left and top-right points. How would you find the area of their overlap?


**Solution:** [https://repl.it/@trsong/Overlapping-Rectangle-Areas](https://repl.it/@trsong/Overlapping-Rectangle-Areas)
```py
import unittest

class Rectangle(object):
    def __init__(self, x1, y1, x2, y2):
        self.x_min = min(x1, x2)
        self.x_max = max(x1, x2)
        self.y_min = min(y1, y2)
        self.y_max = max(y1, y2)


def overlapping_areas(rect1, rect2):
    has_no_x_overlap = rect1.x_max < rect2.x_min or rect2.x_max < rect1.x_min
    has_no_y_overlap = rect1.y_max < rect2.y_min or rect2.y_max < rect1.y_min
    if has_no_x_overlap or has_no_y_overlap:
        return 0
    x_proj = min(rect1.x_max, rect2.x_max) - max(rect1.x_min, rect2.x_min)
    y_proj = min(rect1.y_max, rect2.y_max) - max(rect1.y_min, rect2.y_min)
    return x_proj * y_proj


class OverlappingAreaSpec(unittest.TestCase):
    def assert_result(self, res, rect1, rect2):
        self.assertEqual(res, overlapping_areas(rect1, rect2))
        self.assertEqual(res, overlapping_areas(rect2, rect1))

    def test_no_overlapping(self):
        rect1 = Rectangle(-1, 1, 0, 0)
        rect2 = Rectangle(0, 0, 1, -1)
        self.assert_result(0, rect1, rect2)
        
    def test_no_overlapping2(self):
        rect1 = Rectangle(-1, -1, 0, 2)
        rect2 = Rectangle(0, 2, 1, -1)
        self.assert_result(0, rect1, rect2)

    def test_no_overlapping3(self):
        rect1 = Rectangle(0, 0, 1, 1)
        rect2 = Rectangle(0, -1, 1, -2)
        self.assert_result(0, rect1, rect2)

    def test_rectanlge_contains_the_other(self):
        rect1 = Rectangle(-2, 2, 2, -2)
        rect2 = Rectangle(-1, -1, 1, 1)
        self.assert_result(4, rect1, rect2)

    def test_rectanlge_contains_the_other2(self):
        rect1 = Rectangle(0, 0, 2, 2)
        rect2 = Rectangle(0, 0, 2, 1)
        self.assert_result(2, rect1, rect2)

    def test_overlapping_top_bottom(self):
        rect1 = Rectangle(-2, 0, 2, -2)
        rect2 = Rectangle(-1, 1, 1, -1)
        self.assert_result(2, rect1, rect2)
    
    def test_overlapping_left_right(self):
        rect1 = Rectangle(-1, -1, 1, 1)
        rect2 = Rectangle(0, -2, 2, 2)
        self.assert_result(2, rect1, rect2)
    
    def test_overlapping_top_left_bottom_right(self):
        rect1 = Rectangle(-2, 2, 1, -1)
        rect2 = Rectangle(-1, 1, 2, -2)
        self.assert_result(4, rect1, rect2)
    
    def test_overlapping_top_right_bottom_left(self):
        rect1 = Rectangle(-1, -1, 2, 2)
        rect2 = Rectangle(-2, -2, 1, 1)
        self.assert_result(4, rect1, rect2)

    def test_entire_overlapping(self):
        rect1 = Rectangle(0, 0, 1, 1)
        self.assert_result(1, rect1, rect1)

    
if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 16, 2020 \[Easy\] Zig-Zag Distinct LinkedList
--- 
> **Question:** Given a linked list with DISTINCT value, rearrange the node values such that they appear in alternating `low -> high -> low -> high ...` form. For example, given `1 -> 2 -> 3 -> 4 -> 5`, you should return `1 -> 3 -> 2 -> 5 -> 4`.


**Solution:** [https://repl.it/@trsong/Zig-Zag-Order-of-Distinct-LinkedList](https://repl.it/@trsong/Zig-Zag-Order-of-Distinct-LinkedList)
```py
import unittest
import copy

def zig_zag_order(lst):
    should_increase = True
    prev = dummy = ListNode(-1, lst)
    p = prev.next
    while p and p.next:
        does_increase = p.next.val > p.val
        if should_increase != does_increase:
            second = p.next

            p.next = second.next
            second.next = p

            prev.next = second
            prev = second
        else:
            prev = p
            p = p.next
        should_increase = not should_increase
    return dummy.next
        

###################
# Testing Utilities
###################
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
    
    @staticmethod  
    def List(*vals):
        dummy = ListNode(-1)
        p = dummy
        for elem in vals:
            p.next = ListNode(elem)
            p = p.next
        return dummy.next  

    def __repr__(self):
        return "{} -> {}".format(str(self.val), str(self.next))

    def to_list(self):
        res = []
        p = self
        while p:
            res.append(p.val)
            p = p.next
        return res


class ZigZagOrderSpec(unittest.TestCase):
    def verify_order(self, original_lst):
        lst = zig_zag_order(copy.deepcopy(original_lst))
        self.assertIsNotNone(lst)
        self.assertEqual(set(original_lst.to_list()), set(lst.to_list()))

        isLessThanPrevious = False
        p = lst.next
        prev = lst
        while p:
            if isLessThanPrevious:
                self.assertLess(p.val, prev.val, "%d in %s" % (p.val, lst))
            else:
                self.assertGreater(p.val, prev.val, "%d in %s" % (p.val, lst))

            isLessThanPrevious = not isLessThanPrevious
            prev = p
            p = p.next

    def test_example(self):
        lst = ListNode.List(1, 2, 3, 4, 5)
        self.verify_order(lst)

    def test_empty_array(self):
        self.assertIsNone(zig_zag_order(None))

    def test_unsorted_list1(self):
        lst = ListNode.List(10, 5, 6, 3, 2, 20, 100, 80)
        self.verify_order(lst)

    def test_unsorted_list2(self):
        lst = ListNode.List(2, 4, 6, 8, 10, 20)
        self.verify_order(lst)

    def test_unsorted_list3(self):
        lst = ListNode.List(3, 6, 5, 10, 7, 20)
        self.verify_order(lst)

    def test_unsorted_list4(self):
        lst = ListNode.List(20, 10, 8, 6, 4, 2)
        self.verify_order(lst)

    def test_unsorted_list5(self):
        lst = ListNode.List(6, 4, 2, 1, 8, 3)
        self.verify_order(lst)

    def test_sorted_list(self):
        lst = ListNode.List(6, 5, 4, 3, 2, 1)
        self.verify_order(lst)
    

if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 15, 2020 \[Easy\] Record the Last N Orders
--- 
> **Question:** You run an e-commerce website and want to record the last `N` order ids in a log. Implement a data structure to accomplish this, with the following API:
>
> - `record(order_id)`: adds the order_id to the log
> - `get_last(i)`: gets the ith last element from the log. `i` is guaranteed to be smaller than or equal to `N`.
> 
> You should be as efficient with time and space as possible.

**Solution with Circular Buffer:** [https://repl.it/@trsong/Record-the-Last-N-Orders](https://repl.it/@trsong/Record-the-Last-N-Orders)
```py
import unittest

class ECommerceLogService(object):
    def __init__(self, n):
        self.logs = [None] * n
        self.next_index = 0

    def record(self, order_id):
        self.logs[self.next_index] = order_id
        self.next_index = (self.next_index + 1) % len(self.logs)

    def get_last(self, i):
        physical_index = (self.next_index - i) % len(self.logs)
        log = self.logs[physical_index]
        if log is None:
            raise ValueError('Log %d does not exists' % i)
        else:
            return log


class ECommerceLogServiceSpec(unittest.TestCase):
    def test_throw_exception_when_acessing_log_not_exists(self):
        service = ECommerceLogService(10)
        self.assertRaises(ValueError, service.get_last, 1)

    def test_access_last_element(self):
        service = ECommerceLogService(1)
        service.record(42)
        self.assertEqual(42, service.get_last(1))

    def test_acess_fist_element(self):
        service = ECommerceLogService(3)
        service.record(1)
        service.record(2)
        service.record(3)
        self.assertEqual(1, service.get_last(3))

    def test_overwrite_previous_record(self):
        service = ECommerceLogService(1)
        service.record(1)
        service.record(2)
        service.record(3)
        self.assertEqual(3, service.get_last(1))

    def test_overwrite_previous_record2(self):
        service = ECommerceLogService(2)
        service.record(1)
        service.record(2)
        service.record(3)
        service.record(4)
        self.assertEqual(3, service.get_last(2))

    def test_overwrite_previous_record3(self):
        service = ECommerceLogService(3)
        service.record(1)
        service.record(2)
        service.record(3)
        service.record(4)
        self.assertEqual(3, service.get_last(2))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 14, 2020 \[Medium\] Sort Linked List
--- 
> **Question:** Given a linked list, sort it in `O(n log n)` time and constant space.
>
> For example, the linked list `4 -> 1 -> -3 -> 99` should become `-3 -> 1 -> 4 -> 99`.

**Solution with Merge Sort:** [https://repl.it/@trsong/Sort-Linked-List](https://repl.it/@trsong/Sort-Linked-List)
```py
import unittest

def sort_linked_list(head):
    return MergeSort.sort(head)


class MergeSort(object):
    @staticmethod
    def sort(lst):
        if not lst or not lst.next:
            return lst

        l1, l2 = MergeSort.partition(lst)
        sorted_l1 = MergeSort.sort(l1)
        sorted_l2 = MergeSort.sort(l2)
        return MergeSort.merge(sorted_l1, sorted_l2)

    @staticmethod
    def partition(lst):
        slow = fast = ListNode(-1, lst)
        fast = fast.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        half_lst = slow.next
        slow.next = None
        return lst, half_lst

    @staticmethod
    def merge(lst1, lst2):
        p = dummy = ListNode(-1)
        while lst1 and lst2:
            if lst1.val < lst2.val:
                p.next = lst1
                lst1 = lst1.next
            else:
                p.next = lst2
                lst2 = lst2.next
            p = p.next 
        
        if lst1:
            p.next = lst1
        elif lst2:
            p.next = lst2
        
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

    def __repr__(self):
        return "{} -> {}".format(str(self.val), str(self.next))

    @staticmethod
    def List(*vals):
        p = dummy = ListNode(-1)
        for v in vals:
            p.next = ListNode(v)
            p = p.next
        return dummy.next


class SortLinkedListSpec(unittest.TestCase):
    def test_example(self):
        lst = ListNode.List(4, 1, -3, 99)
        expected = ListNode.List(-3, 1, 4, 99)
        self.assertEqual(expected, sort_linked_list(lst))

    def test_empty_list(self):
        self.assertIsNone(sort_linked_list(None))

    def test_list_with_one_element(self):
        lst = ListNode.List(1)
        expected = ListNode.List(1)
        self.assertEqual(expected, sort_linked_list(lst))

    def test_already_sorted_list(self):
        lst = ListNode.List(1, 2, 3, 4, 5)
        expected = ListNode.List(1, 2, 3, 4, 5)
        self.assertEqual(expected, sort_linked_list(lst))

    def test_list_in_descending_order(self):
        lst = ListNode.List(5, 4, 3, 2, 1)
        expected = ListNode.List(1, 2, 3, 4, 5)
        self.assertEqual(expected, sort_linked_list(lst))

    def test_list_with_duplicated_elements(self):
        lst = ListNode.List(1, 1, 3, 2, 1, 2, 1, 1, 3)
        expected = ListNode.List(1, 1, 1, 1, 1, 2, 2, 3, 3)
        self.assertEqual(expected, sort_linked_list(lst))

    def test_binary_list(self):
        lst = ListNode.List(0, 1, 0, 1, 0, 1)
        expected = ListNode.List(0, 0, 0, 1, 1, 1)
        self.assertEqual(expected, sort_linked_list(lst))

    def test_odd_length_list(self):
        lst = ListNode.List(4, 1, 2)
        expected = ListNode.List(1, 2, 4)
        self.assertEqual(expected, sort_linked_list(lst))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 13, 2020 \[Medium\] Nearest Larger Number
--- 
> **Question:** Given an array of numbers and an index `i`, return the index of the nearest larger number of the number at index `i`, where distance is measured in array indices.
>
> For example, given `[4, 1, 3, 5, 6]` and index `0`, you should return `3`.
>
> If two distances to larger numbers are the equal, then return any one of them. If the array at i doesn't have a nearest larger integer, then return null.
>
> **Follow-up:** If you can preprocess the array, can you do this in constant time?

**Thoughts:** The logical is exactly the same as [**find next largest number on the right**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay.html#jul-17-2020-medium-index-of-larger-next-number) except we have to compare next largest number from both left and right for this question.

**Solution with Stack:** [https://repl.it/@trsong/Nearest-Larger-Number](https://repl.it/@trsong/Nearest-Larger-Number)
```py
import unittest

class NearestLargerNumberFinder(object):
    def __init__(self, nums):
        right_larger_indices = NearestLargerNumberFinder.previous_larger_number(nums)
        reverse_left_larger_indices = NearestLargerNumberFinder.previous_larger_number(nums[::-1])
        n = len(nums)

        self.nearest_indiecs = [None] * n
        for i in xrange(n):
            j = n-1-i
            left_index = n-1-reverse_left_larger_indices[j] if reverse_left_larger_indices[j] is not None else None
            right_index = right_larger_indices[i]
            if left_index is not None and right_index is not None:
                self.nearest_indiecs[i] = left_index if nums[left_index] < nums[right_index] else right_index
            elif left_index is None:
                self.nearest_indiecs[i] = right_index
            elif right_index is None:
                self.nearest_indiecs[i] = left_index

    def search_index(self, i):
        return self.nearest_indiecs[i]

    @staticmethod
    def previous_larger_number(nums):
        if not nums:
            return []

        n = len(nums)
        stack = []
        res = [None] * n
        for i in xrange(n-1, -1, -1):
            while stack and nums[i] >= nums[stack[-1]]:
                stack.pop()
            if stack:
                res[i] = stack[-1]
            stack.append(i)
        return res


class LargerNumberSpec(unittest.TestCase):
    def assert_all_result(self, nums, expected):
        finder = NearestLargerNumberFinder(nums)
        self.assertEqual(expected, map(finder.search_index, range(len(nums))))

    def test_random_list(self):
        nums = [3, 2, 5, 6, 9, 8]
        expected = [2, 0, 3, 4, None, 4]
        self.assert_all_result(nums, expected)

    def test_empty_list(self):
        self.assert_all_result([], [])

    def test_asecending_list(self):
        nums = [0, 1, 2, 2, 3, 3, 3, 4, 5]
        expected = [1, 2, 4, 4, 7, 7, 7, 8, None]
        self.assert_all_result(nums, expected)

    def test_descending_list(self):
        nums = [9, 8, 8, 7, 4, 3, 2, 1, 0, -1]
        expected = [None, 0, 0, 2, 3, 4, 5, 6, 7, 8]
        self.assert_all_result(nums, expected)

    def test_up_down_up(self):
        nums = [0, 1, 2, 1, 2, 3, 4, 5]
        expected = [1, 2, 5, 4, 5, 6, 7, None]
        self.assert_all_result(nums, expected)

    def test_up_down_up2(self):
        nums = [0, 4, -1, 2]
        expected = [1, None, 3, 1]
        self.assert_all_result(nums, expected)

    def test_down_up_down(self):
        nums = [9, 5, 6, 3]
        expected = [None, 2, 0, 2]
        self.assert_all_result(nums, expected)
    
    def test_up_down(self):
        nums = [11, 21, 31, 3]
        expected = [1, 2, None, 2]
        self.assert_all_result(nums, expected)

    def test_random_order(self):
        nums = [4, 3, 5, 2, 4, 7]
        expected = [2, 0, 5, 4, 2, None]
        self.assert_all_result(nums, expected)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 12, 2020 LC 1021 \[Easy\] Remove One Layer of Parenthesis
--- 
> **Question:** Given a valid parenthesis string (with only '(' and ')', an open parenthesis will always end with a close parenthesis, and a close parenthesis will never start first), remove the outermost layers of the parenthesis string and return the new parenthesis string.
>
> If the string has multiple outer layer parenthesis (ie (())()), remove all outer layers and construct the new string. So in the example, the string can be broken down into (()) + (). By removing both components outer layer we are left with () + '' which is simply (), thus the answer for that input would be ().

**Example 1:**
```py
Input: '(())()'
Output: '()'
```

**Example 2:**
```py
Input: '(()())'
Output: '()()'
```

**Example 3:**
```py
Input: '()()()'
Output: ''
```

**Solution:** [https://repl.it/@trsong/Remove-One-Layer-of-Parenthesis](https://repl.it/@trsong/Remove-One-Layer-of-Parenthesis)
```py
import unittest

class ValidCharFilter(object):
    def __init__(self):
        self.balance = 0
    
    def is_valid_char(self, c):
        if c == '(':
            self.balance += 1
            return self.balance > 1
        else:
            self.balance -= 1
            return self.balance > 0
    

def remove_one_layer_parenthesis(s):
    f = ValidCharFilter()
    return ''.join(filter(f.is_valid_char, s))


class RemoveOneLayerParenthesisSpec(unittest.TestCase):
    def test_example(self):
        s = '(())()'
        expected = '()'
        self.assertEqual(expected, remove_one_layer_parenthesis(s))

    def test_example2(self):
        s = '(()())'
        expected = '()()'
        self.assertEqual(expected, remove_one_layer_parenthesis(s))

    def test_example3(self):
        s = '()()()'
        expected = ''
        self.assertEqual(expected, remove_one_layer_parenthesis(s))

    def test_empty_string(self):
        self.assertEqual('', remove_one_layer_parenthesis(''))

    def test_nested_parenthesis(self):
        s = '(()())(())'
        expected = '()()()'
        self.assertEqual(expected, remove_one_layer_parenthesis(s))

    def test_complicated_parenthesis(self):
        s = '(()())(())(()(()))'
        expected = '()()()()(())'
        self.assertEqual(expected, remove_one_layer_parenthesis(s))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 11, 2020 \[Easy\] Full Binary Tree
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

**Solution with Recursion:** [https://repl.it/@trsong/Prune-to-Full-Binary-Tree](https://repl.it/@trsong/Prune-to-Full-Binary-Tree)
```py
import unittest

def full_tree_prune(root):
    if not root:
        return None
    
    updated_left = full_tree_prune(root.left)
    updated_right = full_tree_prune(root.right)

    if updated_left and updated_right:
        root.left = updated_left
        root.right = updated_right
        return root
    elif updated_left:
        return updated_left
    elif updated_right:
        return updated_right
    else:
        return root


class TreeNode(object):
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
        import copy
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


### Aug 10, 2020 LC 261 \[Medium\] Graph Valid Tree
---
> **Question:** Given n nodes labeled from 0 to n-1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.

**Example 1:**
```py
Input: n = 5, and edges = [[0,1], [0,2], [0,3], [1,4]]
Output: True
```

**Example 2:**
```py
Input: n = 5, and edges = [[0,1], [1,2], [2,3], [1,3], [1,4]]
Output: False
```

**My thoughts:** A tree is a connected graph with `n-1` edges. As it is undirected graph and a tree must not have cycle, we can use Disjoint Set (Union-Find) to detect cycle: start from empty graph, for each edge we add to the graph, check both ends of that edge and see if it is already connected. 

**Solution with Union-Find:** [https://repl.it/@trsong/Graph-Valid-Tree](https://repl.it/@trsong/Graph-Valid-Tree)
```py
import unittest

class UnionFind(object):
    def __init__(self, n):
        self.parent = [-1] * n

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


def is_valid_tree(n, edges):
    if n == 0:
        return True
    if len(edges) != n - 1:
        return False
    
    uf = UnionFind(n)
    for u, v in edges:
        if uf.is_connected(u, v):
            return False
        uf.union(u, v)
    return True


class IsValidTreeSpec(unittest.TestCase):
    def test_example(self):
        n, edges = 5, [[0, 1], [0, 2], [0, 3], [1, 4]]
        self.assertTrue(is_valid_tree(n, edges))

    def test_example2(self):
        n, edges = 5, [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]
        self.assertFalse(is_valid_tree(n, edges))

    def test_empty_graph(self):
        n, edges = 0, []
        self.assertTrue(is_valid_tree(n, edges))

    def test_one_node_graph(self):
        n, edges = 1, []
        self.assertTrue(is_valid_tree(n, edges))

    def test_disconnected_graph(self):
        n, edges = 3, []
        self.assertFalse(is_valid_tree(n, edges))

    def test_disconnected_graph2(self):
        n, edges = 5, [[0, 1], [2, 3], [3, 4], [4, 2]]
        self.assertFalse(is_valid_tree(n, edges))

    def test_disconnected_graph3(self):
        n, edges = 4, [[0, 1], [2, 3]]
        self.assertFalse(is_valid_tree(n, edges))

    def test_tree_with_cycle(self):
        n, edges = 4, [[0, 1], [0, 2], [0, 3], [3, 2]]
        self.assertFalse(is_valid_tree(n, edges))

    def test_binary_tree(self):
        n, edges = 7, [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]]
        self.assertTrue(is_valid_tree(n, edges))


if __name__ == "__main__":
    unittest.main(exit=False)
```

### Aug 9, 2020 \[Medium\] Making Changes
---
> **Question:** Given a list of possible coins in cents, and an amount (in cents) n, return the minimum number of coins needed to create the amount n. If it is not possible to create the amount using the given coin denomination, return None.

**Example:**
```py
make_change([1, 5, 10, 25], 36)  # gives 3 coins (25 + 10 + 1) 
```

**Solution with DP:** [https://repl.it/@trsong/Making-Changes-Problem](https://repl.it/@trsong/Making-Changes-Problem)
```py
import unittest
import sys

def make_change(coins, target):
    if target == 0:
        return 0

    # Let dp[v] represents smallest # of coin for value v
    # dp[v] = 1 if v is one of coins
    #       = 1 + min(dp[v-coin]) for all valid coin
    dp = [sys.maxint] * (target+1)
    valid_coins = filter(lambda coin: 0 < coin <= target, coins)
    if not valid_coins:
        return None

    dp[0] = 0
    for coin in valid_coins:
        dp[coin] = 1

    for v in xrange(1, target+1):
        for coin in valid_coins:
            if v > coin:
                dp[v] = min(dp[v], 1 + dp[v-coin])
    
    return dp[target] if dp[target] != sys.maxint else None


class MakeChangeSpec(unittest.TestCase):
    def test_example(self):
        target, coins = 36, [1, 5, 10, 25]
        expected = 3  # 25 + 10 + 1
        self.assertEqual(expected, make_change(coins, target))
    
    def test_target_is_zero(self):
        self.assertEqual(0, make_change([1, 2, 3], 0))
        self.assertEqual(0, make_change([], 0))

    def test_unable_to_reach_target(self):
        target, coins = -1, [1, 2, 3]
        self.assertIsNone(make_change(coins, target))

    def test_unable_to_reach_target2(self):
        target, coins = 10, []
        self.assertIsNone(make_change(coins, target))

    def test_greedy_approach_fails(self):
        target, coins = 11, [9, 6, 5, 1]
        expected = 2  # 5 + 6
        self.assertEqual(expected, make_change(coins, target))

    def test_use_same_coin_multiple_times(self):
        target, coins = 12, [1, 2, 3]
        expected = 4  # 3 + 3 + 3 + 3
        self.assertEqual(expected, make_change(coins, target))

    def test_should_produce_minimum_number_of_changes(self):
        target, coins = 30, [25, 10, 5]
        expected = 2  # 25 + 5
        self.assertEqual(expected, make_change(coins, target))
    
    def test_impossible_get_answer(self):
        target, coins = 4, [3, 5, 7]
        self.assertIsNone(make_change(coins, target))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 8, 2020 \[Easy\] Flatten a Nested Dictionary
---
> **Question:** Write a function to flatten a nested dictionary. Namespace the keys with a period.

**Example:**

```py
Given the following dictionary:
{
    "key": 3,
    "foo": {
        "a": 5,
        "bar": {
            "baz": 8
        }
    }
}

it should become:
{
    "key": 3,
    "foo.a": 5,
    "foo.bar.baz": 8
}

You can assume keys do not contain dots in them, i.e. no clobbering will occur.
```

**Solution with DFS:** [https://repl.it/@trsong/Flatten-Nested-Dictionary](https://repl.it/@trsong/Flatten-Nested-Dictionary)
```py
import unittest

def flatten_dictionary(dictionary):
    stack = [(dictionary, None)]
    res = {}
    while stack:
        val, prefix = stack.pop()
        if type(val) is dict:
            for k, v in val.items():
                updated_prefix = k if prefix is None else "{}.{}".format(prefix, k)
                stack.append((v, updated_prefix))
        else:
            res[prefix] = val
    return res


class FlattenDictionarySpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual({
            "key": 3,
            "foo.a": 5,
            "foo.bar.baz": 8
        }, flatten_dictionary({
            "key": 3,
            "foo": {
                "a": 5,
                "bar": {
                    "baz": 8}}}))

    def test_empty_dictionary(self):
        self.assertEqual({}, flatten_dictionary({}))

    def test_simple_dictionary(self):
        d = {
            'a': 1,
            'b': 2
        }
        self.assertEqual(d, flatten_dictionary(d))
    
    def test_multi_level_dictionary(self):
        d_e = {'e': 0}
        d_d = {'d': d_e}
        d_c = {'c': d_d}
        d_b = {'b': d_c}
        d_a = {'a': d_b}
        self.assertEqual({
            'a.b.c.d.e': 0
        }, flatten_dictionary(d_a))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 7, 2020 \[Hard\] Shortest Uphill and Downhill Route
---
> **Question:** A competitive runner would like to create a route that starts and ends at his house, with the condition that the route goes entirely uphill at first, and then entirely downhill.
>
> Given a dictionary of places of the form `{location: elevation}`, and a dictionary mapping paths between some of these locations to their corresponding distances, find the length of the shortest route satisfying the condition above. Assume the runner's home is location `0`.

**Example:**
```py
Suppose you are given the following input:

elevations = {0: 5, 1: 25, 2: 15, 3: 20, 4: 10}
paths = {
    (0, 1): 10,
    (0, 2): 8,
    (0, 3): 15,
    (1, 3): 12,
    (2, 4): 10,
    (3, 4): 5,
    (3, 0): 17,
    (4, 0): 10
}

In this case, the shortest valid path would be 0 -> 2 -> 4 -> 0, with a distance of 28.
```

**My thoughts:** Break the graph in to uphill and downhill graph separately. For uphill graph, we can calculate shortest distance from 0 to target node. For downhill graph, we want to calculate shortest distance from node to 0. But how? One way is that we can reverse the edge of graph and convert the problem to find shortest distance from 0 to node. Once get shorest distance from and to node, we can sum them up and find the shorest combined distance which will give the final solution.

Then how to do we find shortest distance in directed graph? We can use Dijkstra's algorithm that will be `O(V + E log E)` due to priority queue used in the algorithm. However, notice that in this problem neither uphill graph nor downhil graph can form a cycle. Then we will have a DAG. The shortest distance in DAG can be found in `O(V + E)`. Therefore our solution will have `O(V + E)` in time complexity.

**Solution with Topological Sort:** [https://repl.it/@trsong/Shortest-Uphill-and-Downhill-Route](https://repl.it/@trsong/Shortest-Uphill-and-Downhill-Route)
```py
import unittest
from collections import defaultdict

def shortest_uphill_downhill_cycle(elevations, paths):
    uphill_neighbors = defaultdict(list)
    reverse_downhill_neighbors = defaultdict(list)
    for u, v in paths:
        if elevations[u] < elevations[v]:
            uphill_neighbors[u].append(v)
        else:
            reverse_downhill_neighbors[v].append(u)
    
    uphill_order = topological_sort(uphill_neighbors, elevations)
    shorest_distance_to = shortest_distance(uphill_order, uphill_neighbors, paths)

    reverse_downhill_order = topological_sort(reverse_downhill_neighbors, elevations)
    shortest_distance_from = shortest_distance(reverse_downhill_order, reverse_downhill_neighbors, paths, reversed=True)

    del shorest_distance_to[0]
    del shortest_distance_from[0]
    shortest_cycle_through = map(lambda u: shorest_distance_to[u] + shortest_distance_from[u], elevations.keys())
    return min(shortest_cycle_through) 

    
def topological_sort(neighbors, elevations):
    class Vertex_State:
        UNVISITED = 0
        VISITING = 1
        VISITED = 2

    vertext_states = {node: Vertex_State.UNVISITED for node in elevations}
    reverse_top_order = []

    for node, node_state in vertext_states.items():
        if node_state is not Vertex_State.UNVISITED:
            continue
        
        stack = [node]
        while stack:
            cur = stack[-1]
            if vertext_states[cur] is Vertex_State.VISITED:
                stack.pop()
            elif vertext_states[cur] is Vertex_State.VISITING:
                vertext_states[cur] = Vertex_State.VISITED
                reverse_top_order.append(cur)
            else:
                # vertext_states[cur] is Vertex_State.UNVISITED
                vertext_states[cur] = Vertex_State.VISITING
                for nb in neighbors[cur]:
                    if vertext_states[nb] is Vertex_State.UNVISITED:
                        stack.append(nb)
    
    return reverse_top_order[::-1]


def shortest_distance(top_order, neighbors, paths, reversed=False):
    node_distance = defaultdict(lambda: float('inf'))
    node_distance[0] = 0
    for u in top_order:
        for v in neighbors[u]:
            w = float('inf')
            if reversed:
                w = paths.get((v, u), float('inf'))
            else:
                w = paths.get((u, v), float('inf'))
            node_distance[v] = min(node_distance[v], node_distance[u] + w)
    return node_distance


class ShortestUphillDownhillCycleSpec(unittest.TestCase):
    def test_example(self):
        elevations = {0: 5, 1: 25, 2: 15, 3: 20, 4: 10}
        paths = {
            (0, 1): 10,
            (0, 2): 8,
            (0, 3): 15,
            (1, 3): 12,
            (2, 4): 10,
            (3, 4): 5,
            (3, 0): 17,
            (4, 0): 10
        }
        expected = 28  # 0 -> 2 -> 4 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))

    def test_choose_between_downhill_routes(self):
        """
         1
       / | \
      2  |  3
       \ | /
         0
        """
        elevations = {0: 0, 1: 10, 2: 5, 3: 7}
        paths = {
            (0, 1): 10,
            (1, 2): 20,
            (2, 0): 30,
            (1, 3): 5,
            (3, 0): 6
        }
        expected = 21  # 0 -> 1 -> 3 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))
    
    def test_star_graph(self):
        """
          3
          |
          0
         / \
        1   2
        """
        elevations = {0: 0, 1: 1, 2: 2, 3: 3}
        paths = {
            (0, 1): 1,
            (1, 0): 10,
            (0, 2): 8,
            (2, 0): 5,
            (0, 3): 4,
            (3, 0): 6
        }
        expected = 10  # 0 -> 3 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))
    
    def test_clockwise_vs_conterclockwise(self):
        """
        0 - 1
        |   |
        3 - 2 
        """
        elevations = {0: 0, 1: 1, 2: 2, 3: 3}
        paths = {
            (0, 1): 10,
            (1, 2): 20,
            (2, 3): 30,
            (3, 0): 99,
            (0, 3): 1,
            (3, 2): 2,
            (2, 1): 3,
            (1, 0): 4
        }
        expected = 10  # 0 -> 3 -> 2 -> 1 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))
    
    def test_choose_downhill_or_uphill(self):
        """
         1
       / | \
      2  |  3
       \ | /
         0
        """
        elevations = {0: 0, 1: 50, 2: 25, 3: 100}
        paths = {
            (0, 1): 10,
            (1, 0): 9999,
            (1, 2): 5,
            (2, 0): 3,
            (1, 3): 1,
            (3, 0): 0,
            (0, 3): 999,
            (0, 2): 999
        }
        expected = 11  # 0 -> 1 -> 3 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 6, 2020 LC 78 \[Medium\] Generate All Subsets
---
> **Question:** Given a list of unique numbers, generate all possible subsets without duplicates. This includes the empty set as well.

**Example:**
```py
generate_all_subsets([1, 2, 3])
# [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]
```

**Solution with Recursion:** [https://repl.it/@trsong/Generate-All-Subsets-with-recursion](https://repl.it/@trsong/Generate-All-Subsets-with-recursion)
```py
from functools import reduce

def generate_all_subsets(nums):
    return reduce(
        lambda accu_subsets, elem: accu_subsets + map(lambda subset: subset + [elem], accu_subsets),
        nums,
        [[]])
```

**Solution with Backtracking:** [https://repl.it/@trsong/Generate-All-the-Subsets](https://repl.it/@trsong/Generate-All-the-Subsets)
```py
import unittest

def generate_all_subsets(nums):
    res = []
    backtrack(0, res, [], nums)
    return res


def backtrack(next_index, res, accu, nums):
    res.append(accu[:])
    for i in xrange(next_index, len(nums)):
        accu.append(nums[i])
        backtrack(i+1, res, accu, nums)
        accu.pop()


class GenerateAllSubsetSpec(unittest.TestCase):
    def test_example(self):
        nums = [1, 2, 3]
        expected = [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]
        self.assertItemsEqual(expected, generate_all_subsets(nums))

    def test_empty_list(self):
        nums = []
        expected = [[]]
        self.assertItemsEqual(expected, generate_all_subsets(nums))

    def test_one_elem_list(self):
        nums = [1]
        expected = [[], [1]]
        self.assertItemsEqual(expected, generate_all_subsets(nums))

    def test_two_elem_list(self):
        nums = [1, 2]
        expected = [[1], [2], [1, 2], []]
        self.assertItemsEqual(expected, generate_all_subsets(nums))

    def test_four_elem_list(self):
        nums = [1, 2, 3, 4]
        expected = [
            [], 
            [1], [2], [3],  [4],
            [1, 2], [1, 3], [2, 3], [1, 4], [2, 4], [3, 4],
            [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4], [1, 2, 3, 4]
        ]
        self.assertItemsEqual(expected, generate_all_subsets(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 5, 2020 \[Easy\] Array of Equal Parts
---
> **Question:** Given an array containing only positive integers, return if you can pick two integers from the array which cuts the array into three pieces such that the sum of elements in all pieces is equal.

**Example 1:**
```py
Input: [2, 4, 5, 3, 3, 9, 2, 2, 2]
Output: True
Explanation: choosing the number 5 and 9 results in three pieces [2, 4], [3, 3] and [2, 2, 2]. Sum = 6.
```

**Example 2:**
```py
Input: [1, 1, 1, 1]
Output: False
```

**Solution with Prefix Sum and Two Pointers:** [https://repl.it/@trsong/Array-of-Equal-Parts](https://repl.it/@trsong/Array-of-Equal-Parts)
```py
import unittest

def contains_3_equal_parts(nums):
    if not nums:
        return False

    prefix_sum = nums[:]
    n = len(nums)
    for i in xrange(1, n):
        prefix_sum[i] += prefix_sum[i-1]

    i, j = 0, n-1
    total = prefix_sum[n-1]
    while i < j:
        left_sum = prefix_sum[i] - nums[i]
        right_sum = total - prefix_sum[j]
        if left_sum < right_sum:
            i += 1
        elif left_sum > right_sum:
            j -= 1
        else:
            # left_sum == right_sum
            mid_sum = prefix_sum[j] - nums[j] - prefix_sum[i]
            if mid_sum == left_sum:
                return True
            i += 1
            j -= 1

    return False
        

class Contains3EqualPartSpec(unittest.TestCase):
    def test_example(self):
        nums = [2, 4, 5, 3, 3, 9, 2, 2, 2]
        # Remove 5, 9 to break array into [2, 4], [3, 3] and [2, 2, 2]
        self.assertTrue(contains_3_equal_parts(nums))

    def test_example2(self):
        nums = [1, 1, 1, 1]
        self.assertFalse(contains_3_equal_parts(nums))

    def test_empty_array(self):
        self.assertFalse(contains_3_equal_parts([]))

    def test_two_element_array(self):
        nums = [1, 2]
        # [], [], []
        self.assertTrue(contains_3_equal_parts(nums))

    def test_three_element_array(self):
        nums = [1, 2, 3]
        self.assertFalse(contains_3_equal_parts(nums))

    def test_symmetic_array(self):
        nums = [1, 2, 4, 3, 5, 2, 1]
        # remove 4, 5 gives [1, 2], [3], [2, 1]
        self.assertTrue(contains_3_equal_parts(nums))

    def test_sum_not_divisiable_by_3(self):
        nums = [2, 2, 2, 2, 2, 2]
        self.assertFalse(contains_3_equal_parts(nums))

    def test_ascending_array(self):
        nums = [1, 2, 3, 3, 3, 3, 4, 6]
        # remove 3, 4 gives [1, 2, 3], [3, 3], [6]
        self.assertTrue(contains_3_equal_parts(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```



### Aug 4, 2020 \[Medium\] M Smallest in K Sorted Lists
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

**My thoughts:** This problem is almost the same as merge k sorted list. The idea is to leverage priority queue to keep track of minimum element among all k sorted list.

**Solution with Priority Queue:** [https://repl.it/@trsong/Find-M-Smallest-in-K-Sorted-Lists](https://repl.it/@trsong/Find-M-Smallest-in-K-Sorted-Lists)
```py
import unittest
from Queue import PriorityQueue

def find_m_smallest(ksorted_list, m):
    sorted_iterators = map(iter, ksorted_list)
    pq = PriorityQueue()

    for it in sorted_iterators:
        num = next(it, None)
        if num is not None:
            pq.put((num, it))

    while m > 1 and not pq.empty():
        _, it = pq.get()
        next_num = next(it, None)
        if next_num is not None:
            pq.put((next_num, it))
        m -= 1

    return None if pq.empty() else pq.get()[0]


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


### Aug 3, 2020 \[Hard\] Graph Coloring
---
> **Question:** Given an undirected graph represented as an adjacency matrix and an integer `k`, determine whether each node in the graph can be colored such that no two adjacent nodes share the same color using at most `k` colors.

**My thoughts:** Solve this problem with backtracking. For each node, testing all colors one-by-one; if it turns out there is something wrong with current color, we will backtrack to test other colors.

**Solution with Backtracking:** [https://repl.it/@trsong/k-Graph-Coloring](https://repl.it/@trsong/k-Graph-Coloring)
```py
import unittest

def solve_graph_coloring(neighbor_matrix, k):
    n = len(neighbor_matrix)
    node_color = [None] * n

    def backtrack(next_node):
        if next_node >= n:
            return True
        
        processed_neighbors = [i for i in xrange(next_node) if neighbor_matrix[next_node][i]]
        for color in xrange(k):
            if any(color == node_color[nb] for nb in processed_neighbors):
                continue
            node_color[next_node] = color
            if backtrack(next_node+1):
                return True
            node_color[next_node] = None
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

### Aug 2, 2020 \[Medium\] Maximum Circular Subarray Sum
---
> **Question:** Given a circular array, compute its maximum subarray sum in `O(n)` time. A subarray can be empty, and in this case the sum is 0.

**Example 1:**
```py
Input: [8, -1, 3, 4]
Output: 15 
Explanation: we choose the numbers 3, 4, and 8 where the 8 is obtained from wrapping around.
```

**Example 2:**
```py
Input: [-4, 5, 1, 0]
Output: 6 
Explanation: we choose the numbers 5 and 1.
```

**My thoughts:** The max circular subarray sum can be divied into sub-problems: max non-circular subarray sum and max circular-only subarray sum. 

For max non-circular subarray sum problem, we can use `dp[i]` to represent max subarray sum end at index `i` and `max(dp)` will be the answer.

For max circular-only subarray sum problem, we want to find `i`, `j` where `i < j` such that `nums[0] + nums[1] + ... + nums[i] + nums[j] + .... + nums[n-1]` reaches maximum. The way we can handle it is to calculate prefix sum and suffix sum array and find max accumulated sum on the left and on the right. The max circular-only subarray sum equals the sum of those two accumulated sum. 

Finally, the answer to the original problem is the larger one between answers to above two sub-problems. And one thing worth to notice is that if all elements are negative, then the answer should be `0`.

**Solution with DP and Prefix-Sum:** [https://repl.it/@trsong/Find-Maximum-Circular-Subarray-Sum](https://repl.it/@trsong/Find-Maximum-Circular-Subarray-Sum)
```py
import unittest

def max_circular_sum(nums):
    return max(max_subarray_sum(nums), max_circular_subarray_sum(nums))


def max_subarray_sum(nums):
    n = len(nums)
    # Let dp[n] represents max subarray max ends at index n-1
    dp = [0] * (n+1)
    for i in xrange(1, n+1):
        dp[i] = nums[i-1] + max(dp[i-1], 0)
    return max(dp)


def max_circular_subarray_sum(nums):
    if not nums:
        return 0

    left_max_sums = max_prefix_sums(nums)
    right_max_sums = reversed(max_prefix_sums(reversed(nums)))
    combined_sum = map(sum, zip(left_max_sums, right_max_sums))
    return max(combined_sum)


def max_prefix_sums(stream):
    res = []
    prefix_sum = 0
    max_prefix_sum = 0

    for num in stream:
        max_prefix_sum = max(max_prefix_sum, prefix_sum)
        res.append(max_prefix_sum)
        prefix_sum += num

    return res


class MaxCircularSumSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(15, max_circular_sum([8, -1, 3, 4]))  # 3 + 4 + 8
    
    def test_example2(self):
        self.assertEqual(6, max_circular_sum([-4, 5, 1, 0]))  # 5 + 1

    def test_empty_array(self):
        self.assertEqual(0, max_circular_sum([]))

    def test_negative_array(self):
        self.assertEqual(0, max_circular_sum([-1, -2, -3]))

    def test_circular_array1(self):
        self.assertEqual(22, max_circular_sum([8, -8, 9, -9, 10, -11, 12]))  # 12 + 8 - 8 + 9 - 9 + 10

    def test_circular_array2(self):
        self.assertEqual(23, max_circular_sum([10, -3, -4, 7, 6, 5, -4, -1]))  # 7 + 6 + 5 - 4 -1 + 10

    def test_circular_array3(self):
        self.assertEqual(52, max_circular_sum([-1, 40, -14, 7, 6, 5, -4, -1]))  # 7 + 6 + 5 - 4 - 1 - 1 + 40

    def test_all_positive_array(self):
        self.assertEqual(10, max_circular_sum([1, 2, 3, 4]))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 1, 2020 LC 934 \[Medium\] Shortest Bridge
---
> **Question:** In a given 2D binary array A, there are two islands.  (An island is a 4-directionally connected group of 1s not connected to any other 1s.)
>
> Now, we may change 0s to 1s so as to connect the two islands together to form 1 island.
>
> Return the smallest number of 0s that must be flipped.  (It is guaranteed that the answer is at least 1.)

**Example 1:**
```py
Input: 
[
    [0, 1],
    [1, 0]
]
Output: 1
```

**Example 2:**
```py
Input: 
[
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 1]
]
Output: 2
```

**Example 3:**
```py
Input: 
[
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]
Output: 1
```


**My thoughts:** First color each island with 2 and -2. Then search from each island by bi-directional bfs (2-way bfs). Like below

```py
Color Each Island, mark coastline1 and coastline2 to be start queue and end queue

  .   .   .   .   .   .   .   .
  .   .   .   2   .   2   2   .
  .   .   2   2   2   2   .   .
  .   2   2   2   2   2   .   .
  .   2   .   .   2   2   .   .
  .   .   .   .   2   2   .   .
  .   .   .   .   2   .   .   .
  .   .   .   .   2   .   .   .
  .   .   .   .   .   .   .   .
  .  -2   .   .   .   .   .   .
  .  -2  -2   .   .   .   .   .
  .   .  -2   .   .   .   .   .
  .   .  -2  -2  -2   .   .   .


======= Iteration 1 ======
=> Start Queue Move 

  .   .   .   3   .   3   3   .
  .   .   3   2   3   2   2   3
  .   3   2   2   2   2   3   .
  3   2   2   2   2   2   3   .
  3   2   3   3   2   2   3   .
  .   3   .   3   2   2   3   .
  .   .   .   3   2   3   .   .
  .   .   .   3   2   3   .   .
  .   .   .   .   3   .   .   .
  .  -2   .   .   .   .   .   .
  .  -2  -2   .   .   .   .   .
  .   .  -2   .   .   .   .   .
  .   .  -2  -2  -2   .   .   .

=> End Queue Move 

  .   .   .   3   .   3   3   .
  .   .   3   2   3   2   2   3
  .   3   2   2   2   2   3   .
  3   2   2   2   2   2   3   .
  3   2   3   3   2   2   3   .
  .   3   .   3   2   2   3   .
  .   .   .   3   2   3   .   .
  .   .   .   3   2   3   .   .
  .  -3   .   .   3   .   .   .
 -3  -2  -3   .   .   .   .   .
 -3  -2  -2  -3   .   .   .   .
  .  -3  -2  -3  -3   .   .   .
  .  -3  -2  -2  -2  -3   .   .


======= Iteration 2 ======
=> Start Queue Move 

  .   .   4   3   4   3   3   4
  .   4   3   2   3   2   2   3
  4   3   2   2   2   2   3   4
  3   2   2   2   2   2   3   4
  3   2   3   3   2   2   3   4
  4   3   4   3   2   2   3   4
  .   4   4   3   2   3   4   .
  .   .   4   3   2   3   4   .
  .  -3   .   4   3   4   .   .
 -3  -2  -3   .   4   .   .   .
 -3  -2  -2  -3   .   .   .   .
  .  -3  -2  -3  -3   .   .   .
  .  -3  -2  -2  -2  -3   .   .

=> End Queue Move 

  .   .   4   3   4   3   3   4
  .   4   3   2   3   2   2   3
  4   3   2   2   2   2   3   4
  3   2   2   2   2   2   3   4
  3   2   3   3   2   2   3   4
  4   3   4   3   2   2   3   4
  .   4   4   3   2   3   4   .
  .  -4   4   3   2   3   4   .
 -4  -3  -4   4   3   4   .   .
 -3  -2  -3  -4   4   .   .   .
 -3  -2  -2  -3  -4   .   .   .
 -4  -3  -2  -3  -3  -4   .   .
 -4  -3  -2  -2  -2  -3  -4   .

```

**Solution with 2-way BFS:** [https://repl.it/@trsong/Shortest-Bridge](https://repl.it/@trsong/Shortest-Bridge)
```py
import unittest
from Queue import deque


SEARCH_DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

def shortest_bridge(grid):
    coastline1, coastline2 = dfs_partition(grid)
    distance = bi_directional_bfs(grid, coastline1, coastline2)
    return distance


def dfs_partition(grid):
    """
    Color first island with 2 and second island with -2. Return coastline of each.
    """
    n, m = len(grid), len(grid[0])
    sea_color = 0
    original_island_color = 1
    current_color = 2
    max_num_island = 2
    res = []

    for r in xrange(n):
        for c in xrange(m):
            if grid[r][c] != original_island_color:
                continue
            
            island_costline = []
            stack = [(r, c)]
            while stack:
                cur_r, cur_c = stack.pop()

                # visited pos has same color
                if grid[cur_r][cur_c] is current_color:
                    continue
                grid[cur_r][cur_c] = current_color

                is_coast = False
                for dr, dc in SEARCH_DIRECTIONS:
                    new_r = cur_r + dr
                    new_c = cur_c + dc
                    if not (0 <= new_r < n and 0 <= new_c < m):
                        continue
                    
                    if grid[new_r][new_c] is sea_color:
                        is_coast = True
                    elif grid[new_r][new_c] is original_island_color:
                        # unvisited pos has original color
                        stack.append((new_r, new_c))

                if is_coast:
                    island_costline.append((cur_r, cur_c))
            
            # the other island has negative color
            current_color = -2
            res.append(island_costline)
            if len(res) == max_num_island:
                return res

    return res


def bi_directional_bfs(grid, coastline1, coastline2):
    iteration = 0
    color = 2
    start_queue, end_queue = deque(coastline1), deque(coastline2)
    while start_queue and end_queue:
        if search_by_level(grid, start_queue, color):
            return 2 * iteration - 2

        if search_by_level(grid, end_queue, -color):
            return 2 * iteration - 1

        iteration += 1
        color += 1

    return None


def search_by_level(grid, queue, current_color):
    n, m = len(grid), len(grid[0])
    sign = 1 if current_color > 0 else -1
    sea_color = 0

    for _ in xrange(len(queue)):
        cur_r, cur_c = queue.popleft()

        if grid[cur_r][cur_c] * sign < 0:
            # found the other island
            return True

        if grid[cur_r][cur_c] != sea_color and abs(grid[cur_r][cur_c]) < abs(current_color):
            # visited pos has smaller color 
            continue
        
        grid[cur_r][cur_c] = current_color
        for dr, dc in SEARCH_DIRECTIONS:
            new_r = cur_r + dr
            new_c = cur_c + dc
            if not (0 <= new_r < n and 0 <= new_c < m):
                continue 
            
            if grid[new_r][new_c] is sea_color or grid[new_r][new_c] * sign < 0:
                queue.append((new_r, new_c))
    
    return False
    

class ShortestBridgeSpec(unittest.TestCase):
    @staticmethod
    def print_grid(grid):
        print
        for row in grid:
            print ' '.join('{:3}'.format(v if v != 0 else '.') for v in row)
        print

    def test_example(self):
        grid = [
            [0, 1], 
            [1, 0]
        ]
        self.assertEqual(1, shortest_bridge(grid))  # flip (1, 1)

    def test_example2(self):
        grid = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1]
        ]
        self.assertEqual(2, shortest_bridge(grid))  # flip (0, 2) and (1, 2)

    def test_example3(self):
        grid = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
        self.assertEqual(1, shortest_bridge(grid))  # flip (1, 1)

    def test_1D_grid(self):
        grid = [[1, 0, 1]]
        self.assertEqual(1, shortest_bridge(grid))

    def test_1D_grid2(self):
        grid = [[1, 0, 0, 1]]
        self.assertEqual(2, shortest_bridge(grid))
    
    def test_not_all_0_are_connected(self):
        grid = [
            [0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 0],
        ]
        self.assertEqual(2, shortest_bridge(grid))  # flip (1, 2) and (1, 3)

    def test_complicated_islands(self):
        grid = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
        ]
        self.assertEqual(4, shortest_bridge(grid))  # flip (5, 1), (6, 1), (7, 1) and (8, 1)

    def test_find_shortest_bridge(self):
        grid = [
            [1, 1, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
        self.assertEqual(1, shortest_bridge(grid))  # flip (3, 0)

    def test_performance(self):
        n, m = 15, 15
        grid = [[0 for _ in xrange(m)] for _ in xrange(n)]
        grid[0][0] = 1
        grid[n-1][m-1] = 1
        self.assertEqual(n+m-3, shortest_bridge(grid))


if __name__ == '__main__':
    unittest.main(exit=False)
```