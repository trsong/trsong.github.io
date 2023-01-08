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