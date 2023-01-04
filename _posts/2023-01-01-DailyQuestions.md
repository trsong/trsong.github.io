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
