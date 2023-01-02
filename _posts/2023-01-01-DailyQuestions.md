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
