---
layout: post
title:  "Daily Coding Problems 2023 Jan to Mar"
date:   2023-01-01 22:22:22 -0700
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


### Jan 1, 2023 LC 3 \[Medium\] Longest Substring Without Repeating Characters
---
> **Question:** Given a string s, find the length of the longest 
substring without repeating characters.

**Example 1:**
```py
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
```

**Example 2:**
```py
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
```

**Example 3:**
```py
Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
```

**Solution:** [https://replit.com/@trsong/LC3-Longest-Substring-Without-Repeating-Characters#main.py](https://replit.com/@trsong/LC3-Longest-Substring-Without-Repeating-Characters#main.py)

```py
import unittest

def longest_uniq_substr(s):
    last_occur_record = {}
    res = 0
    start = -1

    for end, ch in enumerate(s):
        last_occur = last_occur_record.get(ch, -1)
        start = max(start, last_occur)
        res = max(res, end - start)
        last_occur_record[ch] = end
        
    return res


class LongestUniqSubstrSpec(unittest.TestCase):
    def testExample1(self):
        s = "abcabcbb"
        ans = 3  # "abc"
        self.assertEqual(ans, longest_uniq_substr(s))

    def testExample2(self):
        s = "bbbbb"
        ans = 1  # "b"
        self.assertEqual(ans, longest_uniq_substr(s))    
        
    def testExample3(self):
        s = "pwwkew"
        ans = 3  # "wke"
        self.assertEqual(ans, longest_uniq_substr(s))

    def testEmptyString(self):
        s = ""
        ans = 0
        self.assertEqual(ans, longest_uniq_substr(s))

    def testLongestAtFront(self):
        s = "abcdabcaba"
        ans = 4  # "abcd"
        self.assertEqual(ans, longest_uniq_substr(s))

    def testLongestAtBack(self):
        s = "aababcabcd"
        ans = 4  # "abcd"
        self.assertEqual(ans, longest_uniq_substr(s))

    def testLongestInTheMiddle(self):
        s = "aababcabcdabcaba"
        ans = 4  # "abcd"
        self.assertEqual(ans, longest_uniq_substr(s))

    def testOneLetterString(self):
        s = "a"
        ans = 1
        self.assertEqual(ans, longest_uniq_substr(s))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


**Failed Attempts:**

> Rev 1: not pruning invalid case. When tracking last occurrance, fail to consider an invalid case: "a" in "abcba" where "bcba" is not a valid result.

```py
def longest_uniq_substr(s):
    last_occur_record = {}
    res = 0

    for end, ch in enumerate(s):
        last_occur = last_occur_record.get(ch, -1)
        res = max(res, end - last_occur)
        last_occur_record[ch] = end
        
    return res
```

> Rev 2: while updating map entry, removing an element is not necessary. 

```py
def longest_uniq_substr(s):
    last_occur_record = {}
    res = 0
    start = 0

    for end, ch in enumerate(s):
        last_occur = last_occur_record.get(ch, -1)
        while start < last_occur:
            if s[start] in last_occur_record:
                del last_occur_record[s[start]]
            start += 1

        res = max(res, end - start)
        last_occur_record[ch] = end
        
    return res
```

> Rev 3: window start position is incorrect. Not consider 1 letter edge case

```py
def longest_uniq_substr(s):
    last_occur_record = {}
    res = 0
    start = 0

    for end, ch in enumerate(s):
        last_occur = last_occur_record.get(ch, -1)
        start = max(start, last_occur)
        res = max(res, end - start)
        last_occur_record[ch] = end
        
    return res
```


### Jan 2, 2023 LC 23 \[Hard\] Merge k Sorted Lists
---
> **Question:** You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
>
> Merge all the linked-lists into one sorted linked-list and return it.

**Example 1:**

```py
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6
```

**Example 2:**
```py
Input: lists = []
Output: []
```

**Example 3:**

```py
Input: lists = [[]]
Output: []
```

**Solution:** [https://replit.com/@trsong/LC-23-Merge-k-Sorted-Lists#main.py](https://replit.com/@trsong/LC-23-Merge-k-Sorted-Lists#main.py)
```py
import unittest
from queue import PriorityQueue

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __eq__(self, other):
        return str(self) == str(other)

    def __str__(self):
        return f"{self.val} -> {str(self.next)}"

    def __repr__(self):
        return str(self)

    @classmethod
    def fromList(cls, lst):
        p = dummy = cls(-1)
        for num in lst:
            p.next = cls(num)
            p = p.next
        return dummy.next


def merge_k_sorted(num_lists):
    pq = PriorityQueue()
    for i, p in enumerate(num_lists):
        if p is None:
            continue
        pq.put((p.val, i, p))

    q = dummy = ListNode(-1)
    while not pq.empty():
        _, i, p = pq.get()
        q.next = ListNode(p.val)
        q = q.next

        if p.next is not None:
            pq.put((p.next.val, i, p.next))

    return dummy.next
        

class MergeKSortedSpec(unittest.TestCase):
    def testExample1(self):
        num_lists = list(map(ListNode.fromList,
                             [[1, 4, 5], [1, 3, 4], [2, 6]]))
        expected = ListNode.fromList([1, 1, 2, 3, 4, 4, 5, 6])
        self.assertEqual(expected, merge_k_sorted(num_lists))
        
    def testExample2(self):
        num_lists = list(map(ListNode.fromList,[[]]))
        expected = ListNode.fromList([])
        self.assertEqual(expected, merge_k_sorted(num_lists))
        
    def testExample3(self):
        num_lists = list(map(ListNode.fromList,[]))
        expected = ListNode.fromList([])
        self.assertEqual(expected, merge_k_sorted(num_lists))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 3, 2023 LC 11 \[Medium\] Container With Most Water
---
> **Question:** You are given an integer array height of length `n`. There are `n` vertical lines drawn such that the two endpoints of the `i`th line are `(i, 0)` and `(i, height[i])`.
>
> Find two lines that together with the x-axis form a container, such that the container contains the most water.
>
> Return the maximum amount of water a container can store.
>
> Notice that you may not slant the container.

**Example 1:**

```py
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
```

**Example 2:**

```py
Input: height = [1,1]
Output: 1
```

**Solution:** [https://replit.com/@trsong/LC-11-Container-With-Most-Water#main.py](https://replit.com/@trsong/LC-11-Container-With-Most-Water#main.py)

```py
import unittest

def max_area(height):
    lo = 0
    hi = len(height) - 1
    res = 0

    while lo < hi:
        area = min(height[lo], height[hi]) * (hi - lo)
        res = max(res, area)
        if height[lo] < height[hi]:
            lo += 1
        elif height[lo] > height[hi]:
            hi -= 1
        else:
            lo += 1
            hi -= 1
    return res


class MaxAreaSpec(unittest.TestCase):

    def testExample1(self):
        self.assertEqual(49, max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]))

    def testExample2(self):
        self.assertEqual(1, max_area([1, 1]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

**Failed Attempts:**

> Rev 1: Do not be confused with LC 42 Trapping Rain Water. This question is about pruning impossible choices 


### Jan 4, 2023 LC 22 \[Medium\] Generate Parentheses
---
> **Question:** Given `n` pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

**Example 1:**

```py
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
```

**Example 2:**

```py
Input: n = 1
Output: ["()"]
```

**Solution:** [https://replit.com/@trsong/LC-22-Generate-Parentheses#main.py](https://replit.com/@trsong/LC-22-Generate-Parentheses#main.py)
```py
import unittest

def backtrack(res, accu, open_bal, close_bal):
    if 0 == open_bal == close_bal:
        res.append("".join(accu))
    else: 
        if open_bal > 0:
            accu.append("(")
            backtrack(res, accu, open_bal - 1, close_bal)
            accu.pop()

        if close_bal > open_bal:
            accu.append(")")
            backtrack(res, accu, open_bal, close_bal - 1)
            accu.pop()


def generate_parentheses(n):
    if n <= 0:
        return []
    res = []
    backtrack(res, [], n, n)
    return res


class GenerateParentheseSpec(unittest.TestCase):
    def testExample1(self):
        expected = ["((()))","(()())","(())()","()(())","()()()"]
        self.assertCountEqual(expected, generate_parentheses(3))
        
    def testExample2(self):
        expected = ["()"]
        self.assertCountEqual(expected, generate_parentheses(1))

    def testEmpty(self):
        self.assertCountEqual([], generate_parentheses(0))

    def testSize2(self):
        expected = ["()()", "(())"]
        self.assertCountEqual(expected, generate_parentheses(2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

**Failed Attempts:**

> Rev 1: Did not handle edge case when `n` is `0` which gives `['']` instead of empty array.


### Jan 5, 2023 LC 19 \[Medium\] Remove Nth Node From End of List
---
> **Question:** Given the head of a linked list, remove the nth node from the end of the list and return its head.

 
**Example 1:**

```py
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
```

**Example 2:**

```py
Input: head = [1], n = 1
Output: []
```


**Example 3:**

```py
Input: head = [1,2], n = 1
Output: [1]
```


**Solution:** [https://replit.com/@trsong/LC-19-Remove-Nth-Node-From-End-of-List#main.py](https://replit.com/@trsong/LC-19-Remove-Nth-Node-From-End-of-List#main.py)

```py
import unittest

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __str__(self):
        return f"{self.val} -> {str(self.next)}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    @classmethod
    def fromList(cls, lst):
        p = dummy = cls(-1)
        for num in lst:
            p.next = cls(num)
            p = p.next
        return dummy.next


def remove_nth_from_end(head, n):
    dummy = slow = fast = ListNode(-1, head)
    for _ in range(n):
        fast = fast.next

    while fast and fast.next:
        slow = slow.next
        fast = fast.next

    if slow.next:
        slow.next = slow.next.next
    return dummy.next
    

class RemoveNthFromEndSpec(unittest.TestCase):
    def testExample1(self):
        n, head = 2, ListNode.fromList([1, 2, 3, 4, 5])
        expected = ListNode.fromList([1, 2, 3, 5])
        self.assertEqual(expected, remove_nth_from_end(head, n))

    def testExample2(self):
        n, head = 1, ListNode.fromList([1])
        expected = ListNode.fromList([])
        self.assertEqual(expected, remove_nth_from_end(head, n))

    def testExample3(self):
        n, head = 1, ListNode.fromList([1, 2])
        expected = ListNode.fromList([1])
        self.assertEqual(expected, remove_nth_from_end(head, n))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 6, 2023 LC 24 \[Medium\] Swap Nodes in Pairs
---
> **Question:** Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

**Example 1:**

```py
Input: head = [1,2,3,4]
Output: [2,1,4,3]
```

**Example 2:**

```py
Input: head = []
Output: []
```

**Example 3:**

```py
Input: head = [1]
Output: [1]
```

**Solution:** [https://replit.com/@trsong/LC-24-Swap-Nodes-in-Pairs#main.py](https://replit.com/@trsong/LC-24-Swap-Nodes-in-Pairs#main.py)
```py
import unittest

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __str__(self):
        return f"{self.val} -> {str(self.next)}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    @classmethod
    def fromList(cls, lst):
        p = dummy = cls(-1)
        for num in lst:
            p.next = cls(num)
            p = p.next
        return dummy.next


def swap_pairs(head):
    p = dummy = ListNode(-1, head)
    while p.next and p.next.next:
        first = p.next
        second = p.next.next
        
        first.next = second.next
        second.next = first
        p.next = second
        
        p = first
    return dummy.next
    

class SwapPairsSpec(unittest.TestCase):
    def testExample1(self):
        head = ListNode.fromList([1, 2, 3, 4])
        expected = ListNode.fromList([2, 1, 4, 3])
        self.assertEqual(expected, swap_pairs(head))

    def testExample2(self):
        head = ListNode.fromList([1])
        expected = ListNode.fromList([1])
        self.assertEqual(expected, swap_pairs(head))

    def testExample3(self):
        head = ListNode.fromList([])
        expected = ListNode.fromList([])
        self.assertEqual(expected, swap_pairs(head))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Jan 7, 2023 LC 31 \[Medium\] Next Permutation
---
> **Question:** A permutation of an array of integers is an arrangement of its members into a sequence or linear order.
>
> For example, for `arr = [1,2,3]`, the following are all the permutations of arr: `[1,2,3]`, `[1,3,2]`, `[2, 1, 3]`, `[2, 3, 1]`, `[3,1,2]`, `[3,2,1]`.
>
> The next permutation of an array of integers is the next lexicographically greater permutation of its integer. More formally, if all the permutations of the array are sorted in one container according to their lexicographical order, then the next permutation of that array is the permutation that follows it in the sorted container. If such arrangement is not possible, the array must be rearranged as the lowest possible order (i.e., sorted in ascending order).
>
> For example, the next permutation of `arr = [1,2,3]` is `[1,3,2]`.
>
> Similarly, the next permutation of `arr = [2,3,1]` is `[3,1,2]`.
While the next permutation of `arr = [3,2,1]` is `[1,2,3]` because `[3,2,1]` does not have a lexicographical larger rearrangement.
>
> Given an array of integers nums, find the next permutation of nums.
>
> The replacement must be in place and use only constant extra memory.

**Example 1:**

```py
Input: nums = [1,2,3]
Output: [1,3,2]
```

**Example 2:**

```py
Input: nums = [3,2,1]
Output: [1,2,3]
```

**Example 3:**

```py
Input: nums = [1,1,5]
Output: [1,5,1]
```

**Solution:** [https://replit.com/@trsong/LC-31-Next-Permutation#main.py](https://replit.com/@trsong/LC-31-Next-Permutation#main.py)
```py
import unittest

def next_permutation(nums):
    prev_peak_pos = find_prev_peak_from_right(nums)
    if prev_peak_pos < 0:
        nums.reverse()
        return

    swap_pos = find_target_plus_from_right(nums, nums[prev_peak_pos])
    nums[prev_peak_pos], nums[swap_pos] = nums[swap_pos], nums[prev_peak_pos]
    reverse(nums, prev_peak_pos + 1, len(nums) - 1)


def find_prev_peak_from_right(nums):
    for i in range(len(nums) - 2, -1, -1):
        if nums[i] < nums[i + 1]:
            return i
    return -1


def find_target_plus_from_right(nums, target):
    for i in range(len(nums) - 1, -1, -1):
        if nums[i] > target:
            return i
    return -1


def reverse(nums, lo, hi):
    while lo < hi:
        nums[lo], nums[hi] = nums[hi], nums[lo]
        lo += 1
        hi -= 1


class NextPermutationSpec(unittest.TestCase):
    def testExample1(self):
        nums = [1, 2, 3]
        expected = [1, 3, 2]
        next_permutation(nums)
        self.assertEqual(expected, nums)
        
    def testExample2(self):
        nums = [3, 2, 1]
        expected = [1, 2, 3]
        next_permutation(nums)
        self.assertEqual(expected, nums)
        
    def testExample3(self):
        nums = [1, 1, 5]
        expected = [1, 5, 1]
        next_permutation(nums)
        self.assertEqual(expected, nums)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 8, 2023 LC 33 \[Medium\] Search in Rotated Sorted Array
---
> **Question:** There is an integer array nums sorted in ascending order (with distinct values).
>
> Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0]`, `nums[1], ..., nums[k-1]]` (0-indexed). For example, `[0,1,2,4,5,6,7]` might be rotated at pivot index 3 and become `[4,5,6,7,0,1,2]`.
>
> Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or `-1` if it is not in nums.
>
> You must write an algorithm with `O(log n)` runtime complexity.

**Example 1:**

```py
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```


**Example 2:**

```py
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
```

**Example 3:**

```py
Input: nums = [1], target = 0
Output: -1
```

**Solution:** [https://replit.com/@trsong/LC-33-Search-in-Rotated-Sorted-Array#main.py](https://replit.com/@trsong/LC-33-Search-in-Rotated-Sorted-Array#main.py)
```py
import unittest

def search(nums, target):
    lo = 0
    hi = len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        
        # case 1 & 2: value of mid < target while on same side
        # case 3: value of mid > target while on differnt sides
        if (nums[lo] <= nums[mid] < target or
            nums[mid] < target <= nums[hi] or
            target <= nums[hi] < nums[mid]):
            lo = mid + 1
        else:
            hi = mid

    return lo if nums[lo] == target else -1


class SearchSpec(unittest.TestCase):
    def testExample1(self):
        nums = [4, 5, 6, 7, 0, 1, 2]
        target = 0
        expected = 4
        self.assertEqual(expected, search(nums, target))

    def testExample2(self):
        nums = [4, 5, 6, 7, 0, 1, 2]
        target = 3
        expected = -1
        self.assertEqual(expected, search(nums, target))

    def testExample3(self):
        nums = [1]
        target = 0
        expected = -1
        self.assertEqual(expected, search(nums, target))

    def testFailed1(self):
        nums = [1, 3]
        target = 3
        expected = 1
        self.assertEqual(expected, search(nums, target))

    def testFailed2(self):
        nums = [5, 1, 3]
        target = 3
        expected = 2
        self.assertEqual(expected, search(nums, target))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


**Failed Attempts:**

> Rev 1: Did not handle edge case when value of lo is equal to mid

```py
def search(nums, target):
    lo = 0
    hi = len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if (nums[lo] < nums[mid] < target or 
            nums[lo] < nums[mid] and nums[lo] > target):
            lo = mid + 1
        else:
            hi = mid

    return lo if nums[lo] == target else -1
```

> Rev 2: did not handle case when mid and value on different sides

```py
def search(nums, target):
    lo = 0
    hi = len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if (nums[lo] <= nums[mid] < target or 
            nums[lo] <= nums[mid] and nums[lo] > target):
            lo = mid + 1
        else:
            hi = mid

    return lo if nums[lo] == target else -1
```

### Jan 9, 2023 LC 45 \[Medium\] Jump Game II
---
> **Question:**  You are given a 0-indexed array of integers nums of length `n`. You are initially positioned at `nums[0]`.
>
> Each element `nums[i]` represents the maximum length of a forward jump from index `i`. In other words, if you are at `nums[i]`, you can jump to any `nums[i + j]` where:

* `0 <= j <= nums[i]` and
* `i + j < n`

> Return the minimum number of jumps to reach `nums[n - 1]`. The test cases are generated such that you can reach `nums[n - 1]`. 

**Example 1:**

```py
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

**Example 2:**

```py
Input: nums = [2,3,0,1,4]
Output: 2
```

**Solution:** [https://replit.com/@trsong/LC-45-Jump-Game-II#main.py](https://replit.com/@trsong/LC-45-Jump-Game-II#main.py)

```py
import unittest

def jump(nums):
    res = -1
    max_pos = -1
    potential_max_pos = -1

    for pos, step in enumerate(nums):
        if pos > max_pos:
            res += 1
            max_pos = potential_max_pos
            
        local_max_pos = pos + step
        potential_max_pos = max(potential_max_pos, local_max_pos)
    return res
    

class JumpSpec(unittest.TestCase):
    def testExample1(self):
        self.assertEqual(2, jump([2, 3, 1, 1, 4]))

    def testExample2(self):
        self.assertEqual(2, jump([2, 3, 0, 1, 4]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

**Failed Attempts:**

> Rev 1: violate the question setup: only increment when switch to new max position.

```py
def jump(nums):
    res = -1
    max_pos = -1

    for pos, step in enumerate(nums):
        if pos > max_pos:
            res += 1
            
        local_max_pos = pos + step
        max_pos = max(max_pos, local_max_pos)
    return res
```

### Jan 10, 2023 LC 36 \[Medium\] Valid Sudoku
---
> **Question:**  Determine if a `9 x 9` Sudoku board is valid. Only the filled cells need to be validated according to the following rules:
>
> * Each row must contain the digits `1-9` without repetition.
> * Each column must contain the digits `1-9` without repetition.
> * Each of the nine `3 x 3` sub-boxes of the grid must contain the digits `1-9` without repetition.
>
> Note:
> * A Sudoku board (partially filled) could be valid but is not necessarily solvable.
> * Only the filled cells need to be validated according to the mentioned rules.
 

**Example 1:**

```py
Input: board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true
```

**Example 2:**

```py
Input: board = 
[["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: false
Explanation: Same as Example 1, except with the 5 in the top left corner being modified to 8. Since there are two 8's in the top left 3x3 sub-box, it is invalid.
```

**Solution:** [https://replit.com/@trsong/LC-36-Valid-Sudoku#main.py](https://replit.com/@trsong/LC-36-Valid-Sudoku#main.py)

```py
import unittest

N = 9
M = 3

def isValidSudoku(board):
    for r in range(N):
        row_it = (board[r][c] for c in range(N))
        if not isValidSequence(row_it):
            return False

    for c in range(N):
        col_it = (board[r][c] for r in range(N))
        if not isValidSequence(col_it):
            return False

    for sub_r in range(M):
        for sub_c in range(M):
            sub_it = (board[r + sub_r * M][c + sub_c * M] 
                      for r in range(M)
                      for c in range(M))
            if not isValidSequence(sub_it):
                return False

    return True


def isValidSequence(it):
    vec = 0
    for ch in it:
        if ch == '.':
            continue

        if not '0' <= ch <= '9':
            return False

        num = int(ch)
        if vec & 1 << num:
            return False
        vec |= 1 << num
    return True


class IsValidSudokuSpec(unittest.TestCase):
    def testExample1(self):
        board = [["5", "3", ".", ".", "7", ".", ".", ".", "."],
                 ["6", ".", ".", "1", "9", "5", ".", ".", "."],
                 [".", "9", "8", ".", ".", ".", ".", "6", "."],
                 ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
                 ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
                 ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
                 [".", "6", ".", ".", ".", ".", "2", "8", "."],
                 [".", ".", ".", "4", "1", "9", ".", ".", "5"],
                 [".", ".", ".", ".", "8", ".", ".", "7", "9"]]
        self.assertTrue(isValidSudoku(board))

    def testExample2(self):
        board = [["8", "3", ".", ".", "7", ".", ".", ".", "."],
                 ["6", ".", ".", "1", "9", "5", ".", ".", "."],
                 [".", "9", "8", ".", ".", ".", ".", "6", "."],
                 ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
                 ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
                 ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
                 [".", "6", ".", ".", ".", ".", "2", "8", "."],
                 [".", ".", ".", "4", "1", "9", ".", ".", "5"],
                 [".", ".", ".", ".", "8", ".", ".", "7", "9"]]
        self.assertFalse(isValidSudoku(board))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 11, 2023 LC 43 \[Medium\] Multiply Strings
---
> **Question:** Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.
>
> Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.

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

**Solution:** [https://replit.com/@trsong/LC-43-Multiply-Strings#main.py](https://replit.com/@trsong/LC-43-Multiply-Strings#main.py)

```py
import unittest

def multiply(num1, num2):
    if len(num1) < len(num2):
        num1, num2 = num2, num1

    rev_num1 = list(map(int, num1))
    rev_num1.reverse()
    rev_num2 = list(map(int, num2))
    rev_num2.reverse()
    reverse_res = multiply_reverse_arr(rev_num1, rev_num2)
    return formate_reverse_arr(reverse_res)


def multiply_reverse_arr(nums1, nums2):
    n, m = len(nums1), len(nums2)
    res = [0] * (n + m)

    for i in range(n):
        for j in range(m):
            res[j + i] += nums1[i] * nums2[j]

    carry = 0
    for i in range(n + m):
        res[i] += carry
        carry = res[i] // 10
        res[i] %= 10
    return res


def formate_reverse_arr(nums):
    res = []
    skipped = True
    for i in range(len(nums) - 1, -1, -1):
        if skipped and nums[i] == 0:
            continue
        skipped = False
        res.append(str(nums[i]))
    return ''.join(res) or "0"


class MultiplySpec(unittest.TestCase):
    def testExample1(self):
        num1 = "2"
        num2 = "3"
        expected = "6"
        self.assertEqual(expected, multiply(num1, num2))

    def testExample2(self):
        num1 = "123"
        num2 = "456"
        expected = "56088"
        self.assertEqual(expected, multiply(num1, num2))

    def testFailedRev1(self):
        num1 = num2 = expected = "0"
        self.assertEqual(expected, multiply(num1, num2))

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

**Failed Attempts:**

> Rev 1: Did not consider when result is "0"

```py
"""
Omit the other parts
"""
def formate_reverse_arr(nums):
    res = []
    skipped = True
    for i in range(len(nums) - 1, -1, -1):
        if skipped and nums[i] == 0:
            continue
        skipped = False
        res.append(str(nums[i]))
    return ''.join(res)  # <----- what if res is empty 
```

### Jan 12, 2023 LC 46 \[Medium\] Permutations
---
> **Question:** Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

**Example 1:**

```py
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**Example 2:**

```py
Input: nums = [0,1]
Output: [[0,1],[1,0]]
```

**Example 3:**

```py
Input: nums = [1]
Output: [[1]]
```


**Solution:** [https://replit.com/@trsong/LC-46-Permutations#main.py](https://replit.com/@trsong/LC-46-Permutations#main.py)

```py
import unittest

def permute(nums):
    res = []
    backtrack(nums, res, 0)
    return res


def backtrack(nums, res, cur_pos):
    n = len(nums)
    if cur_pos >= n:
        res.append(nums[:])
    else:
        for swap_pos in range(cur_pos, n):
            nums[swap_pos], nums[cur_pos] = nums[cur_pos], nums[swap_pos]
            backtrack(nums, res, cur_pos + 1)
            nums[swap_pos], nums[cur_pos] = nums[cur_pos], nums[swap_pos]
    


class PermuteSpec(unittest.TestCase):
    def testExample1(self):
        nums = [1, 2, 3]
        expected = [
            [1, 2, 3], 
            [1, 3, 2], 
            [2, 1, 3], 
            [2, 3, 1], 
            [3, 1, 2],
            [3, 2, 1]
        ]
        self.assertCountEqual(expected, permute(nums))

    def testExample2(self):
        nums = [0, 1]
        expected = [
            [0, 1], 
            [1, 0]
        ]
        self.assertCountEqual(expected, permute(nums))
        
    def testExample3(self):
        nums = [1]
        expected = [
            [1]
        ]
        self.assertCountEqual(expected, permute(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 13, 2023 LC 47 \[Medium\] Permutations II
---
> **Question:** Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.

**Example 1:**

```py
Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]
```

**Example 2:**

```py
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**Less Optimal Yet Acceptable Solution:** [https://replit.com/@trsong/LC-47-Permutations-II#main.py](https://replit.com/@trsong/LC-47-Permutations-II#main.py)

```py
def permute_unique(nums):
    res = []
    backtrack(nums, res, 0)
    return res


def backtrack(nums, res, cur_pos):
    n = len(nums)
    if cur_pos >= n:
        res.append(nums[:])
    else:
        seen = set()
        for swap_pos in range(cur_pos, n):
            if nums[swap_pos] in seen:
                continue
            seen.add(nums[swap_pos])
            
            nums[swap_pos], nums[cur_pos] = nums[cur_pos], nums[swap_pos]
            backtrack(nums, res, cur_pos + 1)
            nums[swap_pos], nums[cur_pos] = nums[cur_pos], nums[swap_pos]
```

**Optimal Solution:** [https://replit.com/@trsong/LC-47-Permutations-II-optimal-solution#main.py](https://replit.com/@trsong/LC-47-Permutations-II-optimal-solution#main.py)

```py
import unittest

def permute_unique(nums):
    histogram = generate_histogram(nums)
    res = []
    backtrack(len(nums), [], res, histogram)
    return res


def backtrack(remain_step, accu, res, histogram):
    if remain_step == 0:
        res.append(accu[:])
    else:
        for num, count in histogram.items():
            if count <= 0:
                continue
                
            accu.append(num)
            histogram[num] -= 1
            backtrack(remain_step - 1, accu, res, histogram)
            accu.pop()
            histogram[num] += 1


def generate_histogram(nums):
    histogram = {}
    for num in nums:
        histogram[num] = histogram.get(num, 0) + 1
    return histogram


class PermuteUniqueSpec(unittest.TestCase):
    def testExample1(self):
        nums = [1, 1, 2]
        expected = [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
        self.assertCountEqual(expected, permute_unique(nums))

    def testExample2(self):
        nums = [1, 2, 3]
        expected = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2],
                    [3, 2, 1]]
        self.assertCountEqual(expected, permute_unique(nums))

    def testFailRev2(self):
        nums = [0, 1, 0, 0, 9]
        expected = [[0, 0, 0, 1, 9], [0, 0, 0, 9, 1], [0, 0, 1, 0, 9],
                    [0, 0, 1, 9, 0], [0, 0, 9, 0, 1], [0, 0, 9, 1, 0],
                    [0, 1, 0, 0, 9], [0, 1, 0, 9, 0], [0, 1, 9, 0, 0],
                    [0, 9, 0, 0, 1], [0, 9, 0, 1, 0], [0, 9, 1, 0, 0],
                    [1, 0, 0, 0, 9], [1, 0, 0, 9, 0], [1, 0, 9, 0, 0],
                    [1, 9, 0, 0, 0], [9, 0, 0, 0, 1], [9, 0, 0, 1, 0],
                    [9, 0, 1, 0, 0], [9, 1, 0, 0, 0]]
        self.assertCountEqual(expected, permute_unique(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


**Failed Attempts:**

> Rev 1: Neither space nor time complexity is ideal due to set add and lookup

```py
def permute_unique(nums):
    res = []
    backtrack(nums, res, 0)
    return res


def backtrack(nums, res, cur_pos):
    n = len(nums)
    if cur_pos >= n:
        res.append(nums[:])
    else:
        seen = set()
        for swap_pos in range(cur_pos, n):
            if nums[swap_pos] in seen:
                continue
            seen.add(nums[swap_pos])
            
            nums[swap_pos], nums[cur_pos] = nums[cur_pos], nums[swap_pos]
            backtrack(nums, res, cur_pos + 1)
            nums[swap_pos], nums[cur_pos] = nums[cur_pos], nums[swap_pos]
```

> Rev 2: Sort and skip duplicate number won't work either. Because swap position will change relative order. Below failed the test case: `[0, 1, 0, 0, 9]`


```py
def permute_unique(nums):
    res = []
    nums.sort()
    backtrack(nums, res, 0)
    return res


def backtrack(nums, res, cur_pos):
    n = len(nums)
    if cur_pos >= n:
        res.append(nums[:])
    else:
        for swap_pos in range(cur_pos, n):
            if swap_pos > cur_pos and nums[swap_pos] == nums[swap_pos - 1]:
                continue
            
            nums[swap_pos], nums[cur_pos] = nums[cur_pos], nums[swap_pos]
            backtrack(nums, res, cur_pos + 1)
            nums[swap_pos], nums[cur_pos] = nums[cur_pos], nums[swap_pos]
```


### Jan 14, 2023 LC 49 \[Medium\] Group Anagrams
---
> **Question:** Given an array of strings strs, group the anagrams together. You can return the answer in any order.
>
> An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

**Example 1:**

```py
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

**Example 2:**

```py
Input: strs = [""]
Output: [[""]]
```

**Example 3:**

```py
Input: strs = ["a"]
Output: [["a"]]
```

**Solution:** [https://replit.com/@trsong/LC-49-Group-Anagrams#main.py](https://replit.com/@trsong/LC-49-Group-Anagrams#main.py)

```py
import unittest

P = 101
CHAR_WEIGHT = {}

def group_anagram(strs):
    grouby_hash = {}
    for s in strs:
        hash_code = hash_anagram(s)
        grouby_hash[hash_code] = grouby_hash.get(hash_code, [])
        grouby_hash[hash_code].append(s)
    return list(grouby_hash.values())


def hash_anagram(s):
    global CHAR_WEIGHT
    histogram = generate_histogram(s)
    res = 0

    for ch, count in histogram.items():
        if ch not in CHAR_WEIGHT:
            CHAR_WEIGHT[ch] = P ** (ord(ch) - ord('a'))
        res += CHAR_WEIGHT[ch] * count
    return res


def generate_histogram(s):
    histogram = {}
    for ch in s:
        histogram[ch] = histogram.get(ch, 0) + 1
    return histogram


class GroupAnagram(unittest.TestCase):
    def assertResult(self, expected, res):
        format = lambda strs: list(map(lambda words: repr(sorted(words)), strs))
        self.assertCountEqual(format(expected), format(res))
    
    def testExample1(self):
        strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
        expected = [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]
        self.assertResult(expected, group_anagram(strs))

    def testExample2(self):
        strs = [""]
        expected = [[""]]
        self.assertResult(expected, group_anagram(strs))

    def testExample3(self):
        strs = ["a"]
        expected = [["a"]]
        self.assertResult(expected, group_anagram(strs))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jan 15, 2023 LC 39 \[Medium\] Combination Sum
---
> **Question:** Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.
>
> The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.
>
> The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.


**Example 1:**

```py
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
```


**Example 2:**

```py
Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]
```

**Example 3:**

```py
Input: candidates = [2], target = 1
Output: []
```

**Solution:** [https://replit.com/@trsong/LC-39-Combination-Sum#main.py](https://replit.com/@trsong/LC-39-Combination-Sum#main.py)

```py
import unittest


def combination_sum(candidates, target):
    sorted_desc_candidates = list(sorted(set(candidates), reverse=True))
    res = []
    backtrack(target, res, [], sorted_desc_candidates, 0)
    return res


def backtrack(balance, res, accu, sorted_desc_candidates, cur_candidate_index):
    if balance == 0:
        res.append(accu[:])
    else:
        for i in range(cur_candidate_index, len(sorted_desc_candidates)):
            num = sorted_desc_candidates[i]
            if num > balance:
                continue

            accu.append(num)
            backtrack(balance - num, res, accu, sorted_desc_candidates, i)
            accu.pop()


class CombinationSumSpec(unittest.TestCase):
    def assertResult(self, expected, res):
        format = lambda lon: list(
            map(lambda nums: repr(list(sorted(nums))), lon))
        self.assertCountEqual(format(expected), format(res))

    def testExample1(self):
        candidates = [2, 3, 6, 7]
        target = 7
        expected = [[2, 2, 3], [7]]
        self.assertResult(expected, combination_sum(candidates, target))

    def testExample2(self):
        candidates = [2, 3, 5]
        target = 8
        expected = [[2, 2, 2, 2], [2, 3, 3], [3, 5]]
        self.assertResult(expected, combination_sum(candidates, target))

    def testExample3(self):
        candidates = [2]
        target = 1
        expected = []
        self.assertResult(expected, combination_sum(candidates, target))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

**Failed Attempts:**

> Rev 1: space complexity is not optimal as recursive step involving slicing the candiate into sub arrays

```py
def combination_sum(candidates, target):
    sorted_desc_candidates = list(sorted(set(candidates), reverse=True))
    res = []
    backtrack(target, res, [], sorted_desc_candidates)
    return res


def backtrack(balance, res, accu, sorted_desc_candidates):
    if balance == 0:
        res.append(accu[:])
    else:
        for i, num in enumerate(sorted_desc_candidates):
            if num > balance:
                continue

            accu.append(num)
            backtrack(balance - num, res, accu, sorted_desc_candidates[i:])
            accu.pop()
```

### Jan 16, 2023 LC 40 \[Medium\] Combination Sum II
---
> **Question:** Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.
>
> Each number in candidates may only be used once in the combination.
>
> Note: The solution set must not contain duplicate combinations.

**Example 1:**

```py
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
```


**Example 2:**

```py
Input: candidates = [2,5,2,1,2], target = 5
Output: 
[
[1,2,2],
[5]
]
```

**Solution:** [https://replit.com/@trsong/LC-40-Combination-Sum-II#main.py](https://replit.com/@trsong/LC-40-Combination-Sum-II#main.py)

```py
import unittest

def combination_sum(candidates, target):
    candidate_value_rank = list(sorted(set(candidates), reverse=True))
    candidate_histogram = generate_histogram(candidates)
    res = []
    backtrack(target, res, [], candidate_value_rank, 0, candidate_histogram)
    return res


def backtrack(balance, res, accu, candidate_value_rank, candidate_value_index, candidate_histogram):
    if balance == 0:
        res.append(accu[:])
    else:
        for i in range(candidate_value_index, len(candidate_value_rank)):
            num = candidate_value_rank[i]
            if num > balance or candidate_histogram[num] <= 0:
                continue

            accu.append(num)
            candidate_histogram[num] -= 1
            backtrack(balance - num, res, accu, candidate_value_rank, i, candidate_histogram)
            candidate_histogram[num] += 1    
            accu.pop()

        
def generate_histogram(nums):
    histogram = {}
    for num in nums:
        histogram[num] = histogram.get(num, 0) + 1
    return histogram


class CombinationSumSpec(unittest.TestCase):
    def assertResult(self, expected, res):
        format = lambda lon: list(
            map(lambda nums: repr(list(sorted(nums))), lon))
        self.assertCountEqual(format(expected), format(res))

    def testExample1(self):
        candidates = [10, 1, 2, 7, 6, 1, 5]
        target = 8
        expected = [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]
        self.assertResult(expected, combination_sum(candidates, target))

    def testExample2(self):
        candidates = [2, 5, 2, 1, 2]
        target = 5
        expected = [[1, 2, 2], [5]]
        self.assertResult(expected, combination_sum(candidates, target))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

**Failed Attempts:**

> Rev 1: Failed the following case due to unable to differentiate which `1` to choose
>
> `target, candidate = 3, [2, 1, 1]`
> 
> `res = [[2, 1], [2, 1]]` 

```py
def combination_sum(candidates, target):
    sorted_desc_candidates = list(sorted(candidates, reverse=True))
    res = []
    backtrack(target, res, [], sorted_desc_candidates, 0)
    return res


def backtrack(balance, res, accu, sorted_desc_candidates, cur_candidate_index):
    if balance == 0:
        res.append(accu[:])
    else:
        for i in range(cur_candidate_index, len(sorted_desc_candidates)):
            num = sorted_desc_candidates[i]
            if num > balance:
                continue

            accu.append(num)
            backtrack(balance - num, res, accu, sorted_desc_candidates, i + 1)
            accu.pop()
```


### Jan 17, 2023 LC 53 \[Medium\] Maximum Subarray
---
> **Question:** Given an integer array nums, find the 
subarray with the largest sum, and return its sum.

**Example 1:**

```py
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.
```

**Example 2:**

```py
Input: nums = [1]
Output: 1
Explanation: The subarray [1] has the largest sum 1.
```

**Example 3:**

```py
Input: nums = [5,4,-1,7,8]
Output: 23
Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.
```

**Solution:** [https://replit.com/@trsong/LC-53-Maximum-Subarray#main.py](https://replit.com/@trsong/LC-53-Maximum-Subarray#main.py)

```py
import unittest

def cal_max_sub_arr_sum(nums):
    max_sum = 0
    max_sum_so_far = 0

    for num in nums:
        max_sum_so_far = max(0, max_sum_so_far) + num
        max_sum = max(max_sum_so_far, max_sum)

    return max_sum


class CalMaxSubArrSumSpec(unittest.TestCase):
    def testExample1(self):
        nums = [-2,1,-3,4,-1,2,1,-5,4]
        expected = 6
        self.assertEqual(expected, cal_max_sub_arr_sum(nums))
        
    def testExample2(self):
        nums = [1]
        expected = 1
        self.assertEqual(expected, cal_max_sub_arr_sum(nums))
        
    def testExample3(self):
        nums = [5,4,-1,7,8]
        expected = 23
        self.assertEqual(expected, cal_max_sub_arr_sum(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)

```

**Failed Attempts:**

> Rev 1: max_sum_so_far represents max dp ended at position thus should always include `num`

```py
def cal_max_sub_arr_sum(nums):
    max_sum = 0
    max_sum_so_far = 0

    for num in nums:
        max_sum_so_far = max(0, max_sum_so_far + num)
        max_sum = max(max_sum_so_far, max_sum)

    return max_sum
```

> Rev 2: Failed edge case when `nums = [-1]` should return `-1`

```py
def cal_max_sub_arr_sum(nums):
    max_sum = 0
    max_sum_so_far = 0

    for num in nums:
        max_sum_so_far = num + max(0, max_sum_so_far)
        max_sum = max(max_sum_so_far, max_sum)

    return max_sum
```
