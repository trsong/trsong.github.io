---
layout: post
title:  "Daily Coding Problems 2022 Feb to Apr"
date:   2022-02-01 02:22:32 -0700
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



### Feb 15, 2022 LC 1171 \[Medium\] Remove Consecutive Nodes that Sum to 0
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

**Solution:** [https://replit.com/@trsong/Remove-Consecutive-Nodes-that-Sum-to-Zero-3](https://replit.com/@trsong/Remove-Consecutive-Nodes-that-Sum-to-Zero-3)
```py
import unittest

def remove_zero_sum_sublists(head):
    p = dummy = ListNode(0, head)
    p = head
    prefix_node = {0: dummy}
    prefix_sum = 0

    while p:
        prefix_sum += p.val
        if prefix_sum not in prefix_node:
            prefix_node[prefix_sum] = p
        else:
            loop_start = prefix_node[prefix_sum]
            q = loop_start.next
            prefix_sum_to_remove = prefix_sum
            while q != p:
                prefix_sum_to_remove += q.val
                del prefix_node[prefix_sum_to_remove]
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
    unittest.main(exit=False, verbosity=2)
```

### Feb 14, 2022 \[Easy\] URL Shortener
---
> **Question:** Implement a URL shortener with the following methods:
>
> - `shorten(url)`, which shortens the url into a six-character alphanumeric string, such as `zLg6wl`.
> - `restore(short)`, which expands the shortened string into the original url. If no such shortened string exists, return `null`.
>
> **Follow-up:** What if we enter the same URL twice?

**Solution:** [https://replit.com/@trsong/URL-Shortener-2](https://replit.com/@trsong/URL-Shortener-2)
```py
import unittest
import random

class URLShortener(object):
    CHAR_SET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    CODE_LEN = 6

    @staticmethod
    def encode(s):
        return "".join(
            random.choice(URLShortener.CHAR_SET)
            for _ in range(URLShortener.CODE_LEN))

    def __init__(self):
        self.long_to_short = {}
        self.short_to_long = {}

    def shorten(self, url):
        if url not in self.long_to_short:
            code = URLShortener.encode(url)
            while code in self.short_to_long:
                # avoid duplicated short url
                code = URLShortener.encode(url)
            self.long_to_short[url] = code
            self.short_to_long[code] = url
        return self.long_to_short[url]

    def restore(self, short):
        return self.short_to_long.get(short, None)


class URLShortenerSpec(unittest.TestCase):
    def test_should_be_able_to_init(self):
        URLShortener()

    def test_restore_should_not_fail_when_url_not_exists(self):
        url_shortener = URLShortener()
        self.assertIsNone(url_shortener.restore("oKImts"))

    def test_shorten_result_into_six_letters(self):
        url_shortener = URLShortener()
        res = url_shortener.shorten("http://magic_url")
        self.assertEqual(6, len(res))

    def test_restore_short_url_gives_original(self):
        url_shortener = URLShortener()
        original_url = "http://magic_url"
        short_url = url_shortener.shorten(original_url)
        self.assertEqual(original_url, url_shortener.restore(short_url))

    def test_shorten_different_url_gives_different_results(self):
        url_shortener = URLShortener()
        url1 = "http://magic_url_1"
        res1 = url_shortener.shorten(url1)
        url2 = "http://magic_url_2"
        res2 = url_shortener.shorten(url2)
        self.assertNotEqual(res1, res2)

    def test_shorten_same_url_gives_same_result(self):
        url_shortener = URLShortener()
        url = "http://magic_url_1"
        res1 = url_shortener.shorten(url)
        res2 = url_shortener.shorten(url)
        self.assertEqual(res1, res2)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```
### Feb 13, 2022 \[Hard\] Order of Alien Dictionary
--- 
> **Question:** You come across a dictionary of sorted words in a language you've never seen before. Write a program that returns the correct order of letters in this language.
>
> For example, given `['xww', 'wxyz', 'wxyw', 'ywx', 'ywz']`, you should return `['x', 'z', 'w', 'y']`.

**My thoughts:** As the alien letters are topologically sorted, we can just mimic what topological sort with numbers and try to find pattern.

Suppose the dictionary contains: `01234`. Then the words can be `023, 024, 12, 133, 2433`. Notice that we can only find the relative order by finding first unequal letters between consecutive words. eg.  `023, 024 => 3 < 4`.  `024, 12 => 0 < 1`.  `12, 133 => 2 < 3`

With relative relation, we can build a graph with each occurring letters being veteces and edge `(u, v)` represents `u < v`. If there exists a loop that means we have something like `a < b < c < a` and total order not exists. Otherwise we preform a topological sort to generate the total order which reveals the alien dictionary. 

As for implementation of topological sort, there are two ways, one is the following by constantly removing edges from visited nodes. The other is to [first DFS to find the reverse topological order then reverse again to find the result](https://trsong.github.io/python/java/2019/11/02/DailyQuestionsNov.html#nov-9-2019-hard-order-of-alien-dictionary). 


**Solution with Toplogical Sort:** [https://replit.com/@trsong/Alien-Dictionary-Order-3](https://replit.com/@trsong/Alien-Dictionary-Order-3)
```py
import unittest

def dictionary_order(sorted_words):
    neighbors = {}
    inward_count = {}
    for i in range(1, len(sorted_words)):
        prev_word = sorted_words[i - 1]
        cur_word = sorted_words[i]
        for prev_ch, cur_ch in zip(prev_word, cur_word):
            if prev_ch != cur_ch:
                neighbors[prev_ch] = neighbors.get(prev_ch, [])
                neighbors[prev_ch].append(cur_ch)
                inward_count[cur_ch] = inward_count.get(cur_ch, 0) + 1
                break
            
    
    char_set = { ch for word in sorted_words for ch in word }
    queue = [ch for ch in char_set if ch not in inward_count]
    top_order = []

    while queue:
        cur = queue.pop(0)
        top_order.append(cur)

        for child in neighbors.get(cur, []):
            inward_count[child] -= 1
            if inward_count[child] == 0:
                del inward_count[child]
                queue.append(child)
    return top_order if len(char_set) == len(top_order) else None


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
    unittest.main(exit=False, verbosity=2)
```

### Feb 12, 2022 \[Medium\] Unit Converter
---
> **Question:** The United States uses the imperial system of weights and measures, which means that there are many different, seemingly arbitrary units to measure distance. There are 12 inches in a foot, 3 feet in a yard, 22 yards in a chain, and so on.
>
> Create a data structure that can efficiently convert a certain quantity of one unit to the correct amount of any other unit. You should also allow for additional units to be added to the system.

**Solution with DFS:** [https://replit.com/@trsong/Unit-converter](https://replit.com/@trsong/Unit-converter)
```py
import unittest

class UnitConverter(object):
    def __init__(self):
        self.neighbor = {}

    def add_rule(self, src, dst, unit):
        """
        Assumptions: 
        1) a rule set up bi-directional link
        2) new rule override old rule
        3) all rules are compatiable with one another
        """
        self.neighbor[src] = self.neighbor.get(src, {})
        self.neighbor[src][dst] = unit
        self.neighbor[dst] = self.neighbor.get(dst, {})
        self.neighbor[dst][src] = 1.0 / unit

    def convert(self, src, dst, amount):
        stack = [(src, amount)]
        visited = set()

        while stack:
            cur, accu = stack.pop()
            if cur == dst:
                return accu
            
            if cur in visited:
                continue
            visited.add(cur)
            
            for child, unit in self.neighbor.get(cur, {}).items():
                if child in visited:
                    continue
                stack.append((child, unit * accu))
        return None


class Units:
    INCH = 'inch'
    FOOT = 'foot'
    YARD = 'yard'
    CHAIN = 'chain'
    FURLONG = 'furlong'
    MILE = 'mile'
    CAD = 'CAD'
    USD = 'USD'


class UnitConverterSpec(unittest.TestCase):
    def test_base_unit(self):
        unitConverter = UnitConverter()
        unitConverter.add_rule(Units.FOOT, Units.INCH, 12)
        self.assertEqual(12, unitConverter.convert(Units.FOOT, Units.INCH, 1))
        self.assertEqual(24, unitConverter.convert(Units.FOOT, Units.INCH, 2))

    def test_base_unit_chaining(self):
        unitConverter = UnitConverter()
        unitConverter.add_rule(Units.FOOT, Units.INCH, 12)
        unitConverter.add_rule(Units.YARD, Units.FOOT, 3)
        unitConverter.add_rule(Units.CHAIN, Units.YARD, 22)
        self.assertEqual(36, unitConverter.convert(Units.YARD, Units.INCH, 1))
        self.assertEqual(66, unitConverter.convert(Units.CHAIN, Units.FOOT, 1))
    
    def test_alternative_path(self):
        unitConverter = UnitConverter()
        unitConverter.add_rule(Units.MILE, Units.FOOT, 5280)
        unitConverter.add_rule(Units.FOOT, Units.INCH, 12)
        unitConverter.add_rule(Units.YARD, Units.FOOT, 3)
        unitConverter.add_rule(Units.CHAIN, Units.YARD, 22)
        unitConverter.add_rule(Units.FURLONG, Units.CHAIN, 10)
        unitConverter.add_rule(Units.MILE, Units.FURLONG, 8)
        self.assertEqual(5280, unitConverter.convert(Units.MILE, Units.FOOT, 1))
        self.assertEqual(220, unitConverter.convert(Units.FURLONG, Units.YARD, 1))

    def test_no_rule_exist(self):
        unitConverter = UnitConverter()
        self.assertIsNone(unitConverter.convert(Units.FOOT, Units.INCH, 1))

    def test_no_rule_exist2(self):
        unitConverter = UnitConverter()
        unitConverter.add_rule(Units.FOOT, Units.INCH, 12)
        unitConverter.add_rule(Units.CHAIN, Units.YARD, 22)
        unitConverter.add_rule(Units.FURLONG, Units.CHAIN, 10)
        self.assertIsNone(unitConverter.convert(Units.FURLONG, Units.FOOT, 1))

    def test_assumption_one(self):
        unitConverter = UnitConverter()
        unitConverter.add_rule(Units.FOOT, Units.INCH, 12)
        self.assertAlmostEqual(1/12, unitConverter.convert(Units.INCH, Units.FOOT, 1))
        unitConverter.add_rule(Units.YARD, Units.FOOT, 3)
        self.assertAlmostEqual(1/36, unitConverter.convert(Units.INCH, Units.YARD, 1))

    def test_assumption_two(self):
        unitConverter = UnitConverter()
        unitConverter.add_rule(Units.USD, Units.CAD, 1)
        self.assertEqual(2, unitConverter.convert(Units.USD, Units.CAD, 2))
        unitConverter.add_rule(Units.USD, Units.CAD, 1.2)
        self.assertEqual(2.4, unitConverter.convert(Units.USD, Units.CAD, 2))

    def test_assumption_one_and_two(self):
        unitConverter = UnitConverter()
        unitConverter.add_rule(Units.USD, Units.CAD, 1)
        self.assertEqual(2, unitConverter.convert(Units.USD, Units.CAD, 2))
        unitConverter.add_rule(Units.CAD, Units.USD, 0.8)
        self.assertEqual(2.5, unitConverter.convert(Units.USD, Units.CAD, 2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 11, 2022 \[Medium\] Swap Even and Odd Bits
---
> **Question:** Given an unsigned 8-bit integer, swap its even and odd bits. The 1st and 2nd bit should be swapped, the 3rd and 4th bit should be swapped, and so on.

**Example:**

```py
10101010 should be 01010101. 11100010 should be 11010001.
```
> Bonus: Can you do this in one line?

**Solution:** [https://replit.com/@trsong/Swap-Even-and-Odd-Bits-of-Binary-Number](https://replit.com/@trsong/Swap-Even-and-Odd-Bits-of-Binary-Number)
```py
import unittest

def swap_bits(num):
    # 1010 is 0xa, 0101 is 0x5
    # 32 bit has 8 bits (4 * 8 = 32)
    return (num & 0xaaaaaaaa) >> 1 | (num & 0x55555555 ) << 1


class SwapBitSpec(unittest.TestCase):
    def assert_result(self, expected, res):
        self.assertEqual(bin(expected), bin(res))

    def test_example1(self):
        self.assert_result(0b01010101, swap_bits(0b10101010))

    def test_example2(self):
        self.assert_result(0b11010001, swap_bits(0b11100010))

    def test_zero(self):
        self.assert_result(0, swap_bits(0))
    
    def test_one(self):
        self.assert_result(0b10, swap_bits(0b1))

    def test_odd_digits(self):
        self.assert_result(0b1011, swap_bits(0b111))

    def test_large_number(self):
        self.assert_result(0xffffffff, swap_bits(0xffffffff))
    
    def test_large_number2(self):
        self.assert_result(0xaaaaaaaa, swap_bits(0x55555555))

    def test_large_number3(self):
        self.assert_result(0x55555555, swap_bits(0xaaaaaaaa))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 10, 2022 \[Easy\] BST Nodes Sum up to K
---
> **Question:** Given the root of a binary search tree, and a target K, return two nodes in the tree whose sum equals K.

**Example:** 
```py
Given the following tree and K of 20

    10
   /   \
 5      15
       /  \
     11    15
Return the nodes 5 and 15.
```

**Solution:** [https://replit.com/@trsong/Find-BST-Nodes-Sum-up-to-K-2](https://replit.com/@trsong/Find-BST-Nodes-Sum-up-to-K-2)
```py
import unittest

def find_pair(tree, k):
    left_stream = in_order_iteration(tree)
    right_stream = reverse_in_order_iteration(tree)
    left = next(left_stream, None)
    right = next(right_stream, None)

    while left != right:
        pair_sum = left.val + right.val
        if pair_sum == k:
            return [left.val, right.val]
        elif pair_sum < k:
            left = next(left_stream, None)
        else:
            right = next(right_stream, None)
    return None


def in_order_iteration(root):
    if root is not None:
        for node in in_order_iteration(root.left):
            yield node
        yield root
        for node in in_order_iteration(root.right):
            yield node


def reverse_in_order_iteration(root):
    if root is not None:
        for node in reverse_in_order_iteration(root.right):
            yield node
        yield root
        for node in reverse_in_order_iteration(root.left):
            yield node


class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class FindPairSpec(unittest.TestCase):
    def test_example(self):
        """
            10
           /   \
         5      15
               /  \
             11    15
        """
        n15 = Node(15, Node(11), Node(15))
        n10 = Node(10, Node(5), n15)
        self.assertEqual([5, 15], find_pair(n10, 20))

    def test_empty_tree(self):
        self.assertIsNone(find_pair(None, 0))

    def test_full_tree(self):
        """
             7
           /   \
          3     13
         / \   /  \
        2   5 11   17
        """
        n3 = Node(3, Node(2), Node(5))
        n13 = Node(13, Node(11), Node(17))
        n7 = Node(7, n3, n13)
        self.assertEqual([2, 5], find_pair(n7, 7))
        self.assertEqual([5, 13], find_pair(n7, 18))
        self.assertEqual([7, 17], find_pair(n7, 24))
        self.assertEqual([11, 17], find_pair(n7, 28))
        self.assertIsNone(find_pair(n7, 4))

    def test_tree_with_same_value(self):
        """
        42
          \
           42
        """
        tree = Node(42, right=Node(42))
        self.assertEqual([42, 42], find_pair(tree, 84))
        self.assertIsNone(find_pair(tree, 42))

    def test_sparse_tree(self):
        """
           7
         /   \
        2     17
         \   /
          5 11
         /   \
        3     13
        """
        n2 = Node(2, right=Node(5, Node(3)))
        n17 = Node(17, Node(11, right=Node(13)))
        n7 = Node(7, n2, n17)
        self.assertEqual([2, 5], find_pair(n7, 7))
        self.assertEqual([5, 13], find_pair(n7, 18))
        self.assertEqual([7, 17], find_pair(n7, 24))
        self.assertEqual([11, 17], find_pair(n7, 28))
        self.assertIsNone(find_pair(n7, 4))


if __name__ == '__main__':
   unittest.main(exit=False, verbosity=2)
```

### Feb 9, 2022 LC 1344 \[Easy\] Angle between Clock Hands
---
> **Question:** Given a clock time in `hh:mm` format, determine, to the nearest degree, the angle between the hour and the minute hands.

**Example:**
```py
Input: "9:00"
Output: 90
```

**Solution:** [https://replit.com/@trsong/Calculate-Angle-between-Clock-Hands-2](https://replit.com/@trsong/Calculate-Angle-between-Clock-Hands-2)
```py
import unittest

def clock_angle(hhmm):
    hh, mm = hhmm.split(':')
    h = int(hh) % 12
    m = int(mm)

    m_unit = 360 / 60.0
    h_unit = 360 / 12.0

    m_angle = m_unit * m
    h_angle = h_unit * h + h_unit * m / 60.0
    delta = abs(m_angle - h_angle)
    
    return min(360 - delta, delta)


class ClockAngleSpec(unittest.TestCase):
    def test_minute_point_zero(self):
        hhmm = "12:00"
        angle = 0
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_minute_point_zero2(self):
        hhmm = "1:00"
        angle = 30
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_minute_point_zero3(self):
        hhmm = "9:00"
        angle = 90
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_minute_point_zero4(self):
        hhmm = "6:00"
        angle = 180
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_half_pass_hour(self):
        hhmm = "12:30"
        angle = 165
        self.assertEqual(angle, clock_angle(hhmm))
        
    def test_half_pass_hour2(self):
        hhmm = "3:30"
        angle = 75
        self.assertEqual(angle, clock_angle(hhmm))

    def test_irregular_time(self):
        hhmm = "16:20"
        angle = 10
        self.assertEqual(angle, clock_angle(hhmm))

    def test_irregular_time3(self):
        hhmm = "3:15"
        angle = 7.5
        self.assertEqual(angle, clock_angle(hhmm))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```
### Feb 8, 2022 \[Easy\] Exists Overlap Rectangle
--- 
> **Question:** You are given a list of rectangles represented by min and max x- and y-coordinates. Compute whether or not a pair of rectangles overlap each other. If one rectangle completely covers another, it is considered overlapping.
>
> For example, given the following rectangles:
```py
{
    "top_left": (1, 4),
    "dimensions": (3, 3) # width, height
},
{
    "top_left": (-1, 3),
    "dimensions": (2, 1)
},
{
    "top_left": (0, 5),
    "dimensions": (4, 3)
}
```
> return true as the first and third rectangle overlap each other.

**Solution:** [https://replit.com/@trsong/Exists-Overlap-Rectangles-2](https://replit.com/@trsong/Exists-Overlap-Rectangles-2)
```py
import unittest

def exists_overlap_rectangle(rectangles):
    rects = list(map(Rectangle.from_json, rectangles))
    rects.sort(key=lambda rect: (rect.xmin, rect.ymin))

    n = len(rects)
    for i in range(n):
        rect1 = rects[i]
        for j in range(i + 1, n):
            rect2 = rects[j]
            if rect2.xmin > rect1.xmax or rect2.ymin > rect1.ymax:
                continue
            if rect1.has_overlap(rect2):
                return True
    return False


class Rectangle(object):
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    @staticmethod
    def from_json(json):
        top_left = json.get("top_left")
        dimensions = json.get("dimensions")
        xmin = top_left[0]
        xmax = xmin + dimensions[0]
        ymax = top_left[1]
        ymin = ymax - dimensions[1]
        return Rectangle(xmin, xmax, ymin, ymax)

    def has_overlap(self, other):
        has_x_overlap = min(self.xmax, other.xmax) > max(self.xmin, other.xmin)
        has_y_overlap = min(self.ymax, other.ymax) > max(self.ymin, other.ymin)
        return has_x_overlap and has_y_overlap
    

class ExistsOverlapRectangleSpec(unittest.TestCase):
    def test_example(self):
        rectangles = [
            {
                "top_left": (1, 4),
                "dimensions": (3, 3)  # width, height
            },
            {
                "top_left": (-1, 3),
                "dimensions": (2, 1)
            },
            {
                "top_left": (0, 5),
                "dimensions": (4, 3)
            }
        ]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_empty_rectangle_list(self):
        self.assertFalse(exists_overlap_rectangle([]))

    def test_two_overlap_rectangle(self):
        rectangles = [
            {
                "top_left": (0, 1),
                "dimensions": (1, 3)  # width, height
            },
            {
                "top_left": (-1, 0),
                "dimensions": (3, 1)
            }
        ]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_two_overlap_rectangle_form_a_cross(self):
        rectangles = [
            {
                "top_left": (-1, 1),
                "dimensions": (3, 2)  # width, height
            },
            {
                "top_left": (0, 0),
                "dimensions": (1, 1)
            }
        ]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_same_y_coord_not_overlap(self):
        rectangles = [
            {
                "top_left": (0, 0),
                "dimensions": (1, 1)  # width, height
            },
            {
                "top_left": (1, 0),
                "dimensions": (2, 2)
            },
            {
                "top_left": (3, 0),
                "dimensions": (5, 2)
            }
        ]
        self.assertFalse(exists_overlap_rectangle(rectangles))

    def test_same_y_coord_overlap(self):
        rectangles = [
            {
                "top_left": (0, 0),
                "dimensions": (1, 1)  # width, height
            },
            {
                "top_left": (1, 0),
                "dimensions": (2, 2)
            },
            {
                "top_left": (3, 0),
                "dimensions": (5, 2)
            }
        ]
        self.assertFalse(exists_overlap_rectangle(rectangles))

    def test_rectangles_in_different_quadrant(self):
        rectangles = [
            {
                "top_left": (1, 1),
                "dimensions": (2, 2)  # width, height
            },
            {
                "top_left": (-1, 1),
                "dimensions": (2, 2)
            },
            {
                "top_left": (1, -1),
                "dimensions": (2, 2)
            },
            {
                "top_left": (-1, -1),
                "dimensions": (2, 2)
            }
        ]
        self.assertFalse(exists_overlap_rectangle(rectangles))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 7, 2022 \[Easy\] Permutation with Given Order
---
> **Question:** A permutation can be specified by an array `P`, where `P[i]` represents the location of the element at `i` in the permutation. For example, `[2, 1, 0]` represents the permutation where elements at the index `0` and `2` are swapped.
>
> Given an array and a permutation, apply the permutation to the array. 
>
> For example, given the array `["a", "b", "c"]` and the permutation `[2, 1, 0]`, return `["c", "b", "a"]`.

**My thoughts:** In-place solution requires swapping current position `i` with target postion `j` if `j > i` (out of already processed window). However, if `j < i`, then `j`'s position has been swapped, we backtrack recursively to find `j`'s new position.

**Solution:** [https://replit.com/@trsong/Find-Permutation-with-Given-Order-2](https://replit.com/@trsong/Find-Permutation-with-Given-Order-2)
```py
import unittest

def permute(arr, order):
    for from_pos in range(len(order)):
        to_pos = order[from_pos]

        # Only swap current elem out of already processed window
        # If target position is within window, then that position has been swapped 
        while to_pos < from_pos:
            to_pos = order[to_pos]
        
        arr[to_pos], arr[from_pos] = arr[from_pos], arr[to_pos]
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

### Feb 6, 2022  \[Medium\] Maximum Distance among Binary Strings
---
> **Question:** The distance between 2 binary strings is the sum of their lengths after removing the common prefix. For example: the common prefix of `1011000` and `1011110` is `1011` so the distance is `len("000") + len("110") = 3 + 3 = 6`.
>
> Given a list of binary strings, pick a pair that gives you maximum distance among all possible pair and return that distance.

**My thoughts:** The idea is to build a trie to keep track of common characters as well as remaining characters to allow quickly calculate max path length on the left child or right child. 

There are three situations:

- A node has two children: max distance = max distannce of left node + max distance of right node
- A node has one child and is terminal node: max distance = max distance of that child
- A node has one child and is not terminal node: do nothing


**Solution with Trie:** [https://replit.com/@trsong/Maximum-Distance-among-Binary-Strings-2](https://replit.com/@trsong/Maximum-Distance-among-Binary-Strings-2)
```py
import unittest

def max_distance(bins):
    if len(bins) < 2:
        return -1

    t = Trie()
    for num_str in bins:
        t.insert(num_str)
    return t.max_distance()


class Trie(object):
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.max_path = 0

    def insert(self, num_str):
        p = self
        n = len(num_str)
        for index, digit in enumerate(num_str):
            remaining_char = n - index
            p.max_path = max(p.max_path, remaining_char)
            p.children[digit] = p.children.get(digit, Trie())
            p = p.children[digit]
        p.is_end = True

    def max_distance(self):
        stack = [self]
        res = 0

        while stack:
            cur = stack.pop()
            if cur.is_end:
                res = max(res, cur.max_path)

            if len(cur.children) == 2:
                left_max_path = cur.children['0'].max_path
                right_max_path = cur.children['1'].max_path
                res = max(res, 2 + left_max_path + right_max_path)

            stack.extend(cur.children.values())
        return res


class MaxDistanceSpec(unittest.TestCase):
    def test_example(self):
        bins = ['1011000', '1011110']
        expected = len('000') + len('110')
        self.assertEqual(expected, max_distance(bins))

    def test_less_than_two_strings(self):
        self.assertEqual(-1, max_distance([]))
        self.assertEqual(-1, max_distance(['']))
        self.assertEqual(-1, max_distance(['0010']))

    def test_empty_string(self):
        self.assertEqual(0, max_distance(['', '']))
        self.assertEqual(6, max_distance(['', '010101']))

    def test_string_with_same_prefix(self):
        bins = ['000', '0001', '0001001']
        expected = len('1001')
        self.assertEqual(expected, max_distance(bins))

    def test_string_with_same_prefix2(self):
        bins = ['000', '0001', '0001001']
        expected = len('1001')
        self.assertEqual(expected, max_distance(bins))

    def test_return_max_distance_through_root(self):
        """
          0
         / \
        0   1
           / \
          0   1
           \
            1  
        """
        bins = ['00', '0101', '011']
        expected = len('0') + len('101')
        self.assertEqual(expected, max_distance(bins))

    def test_return_max_distance_not_through_root(self):
        """
        0
         \
          1
         / \
        0   1
       /   / \ 
      0   0   1
               \
                1
        """
        bins = ['0100', '0110', '01111']
        expected = len('00') + len('111')
        self.assertEqual(expected, max_distance(bins))

    def test_return_max_distance_when_there_is_prefix(self):
        """
        0
         \
          1 *
           \
            1
             \
              1
             / \
            0   1
        """
        bins = ['01', '01110', '01111']
        expected = len('111')
        self.assertEqual(expected, max_distance(bins))

    def test_return_max_distance_when_there_is_prefix2(self):
        """
        0
       / \
      0   1 *
           \
            1
             \
              1
             / \
            0   1
               /
              0 
        """
        bins = ['01', '01110', '011110', '00']
        expected = len('0') + len('11110')
        self.assertEqual(expected, max_distance(bins))

    def test_return_max_distance_when_there_is_prefix_and_empty_string(self):
        bins = ['', '0', '00', '000', '1']
        expected = len('1') + len('000')
        self.assertEqual(expected, max_distance(bins))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Feb 5, 2022 \[Easy\] Step Word Anagram
---
> **Question:** A step word is formed by taking a given word, adding a letter, and anagramming the result. For example, starting with the word `"APPLE"`, you can add an `"A"` and anagram to get `"APPEAL"`.
>
> Given a dictionary of words and an input word, create a function that returns all valid step words.


**Solution:** [https://replit.com/@trsong/Step-Word-Anagram-3](https://replit.com/@trsong/Step-Word-Anagram-3)
```py
import unittest

def find_step_anagrams(word, dictionary):
    word_histogram = generate_historgram(word)

    return list(filter(
        lambda candidate: 1 ==
            len(candidate) - len(word) ==
            frequency_distance(generate_historgram(candidate), word_histogram),
        dictionary))


def generate_historgram(word):
    histogram = {}
    for ch in word:
        histogram[ch] = histogram.get(ch, 0) + 1
    return histogram


def frequency_distance(histogram1, histogram2):
    res = 0
    for ch, count in histogram1.items():
        res += max(0, count - histogram2.get(ch, 0))
    return res


class FindStepAnagramSpec(unittest.TestCase):
    def test_example(self):
        word = 'APPLE'
        dictionary = ['APPEAL', 'CAPPLE', 'PALPED']
        expected = ['APPEAL', 'CAPPLE', 'PALPED']
        self.assertEqual(
            sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_empty_word(self):
        word = ''
        dictionary = ['A', 'B', 'AB', 'ABC']
        expected = ['A', 'B']
        self.assertEqual(
            sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_empty_dictionary(self):
        word = 'ABC'
        dictionary = []
        expected = []
        self.assertEqual(
            sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_no_match(self):
        word = 'ABC'
        dictionary = ['BBB', 'ACCC']
        expected = []
        self.assertEqual(
            sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_no_match2(self):
        word = 'AA'
        dictionary = ['ABB']
        expected = []
        self.assertEqual(
            sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_repeated_chars(self):
        word = 'AAA'
        dictionary = ['A', 'AA', 'AAA', 'AAAA', 'AAAAB', 'AAB', 'AABA']
        expected = ['AAAA', 'AABA']
        self.assertEqual(
            sorted(expected), sorted(find_step_anagrams(word, dictionary)))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 4, 2022 \[Medium\] Distance Between 2 Nodes in BST
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

**Solution:** [https://replit.com/@trsong/Distance-Between-2-Nodes-in-BST-2](https://replit.com/@trsong/Distance-Between-2-Nodes-in-BST-2)
```py
import unittest

def find_distance(tree, v1, v2):    
    path1 = find_path(tree, v1)
    path2 = find_path(tree, v2)
    num_common_ancestor = 0

    for p1, p2 in zip(path1, path2):
        if p1 != p2:
            break
        num_common_ancestor += 1
    return len(path1) + len(path2) - 2 * num_common_ancestor


def find_path(root, value):
    p = root
    res = []
    while p:
        res.append(p)
        if p.val == value:
            return res
        elif p.val < value:
            p = p.right
        else:
            p = p.left
    return None


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class FindDistanceSpec(unittest.TestCase):
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
    unittest.main(exit=False, verbosity=2)
```
### Feb 3, 2022 LC 872 \[Easy\] Leaf-Similar Trees
---
> **Question:** Given two trees, whether they are `"leaf similar"`. Two trees are considered `"leaf-similar"` if their leaf orderings are the same. 
>
> For instance, the following two trees are considered leaf-similar because their leaves are `[2, 1]`:

```py
# Tree1
    3
   / \ 
  5   1
   \
    2 

# Tree2
    7
   / \ 
  2   1
   \
    2 
```

**Solution with DFS:** [https://replit.com/@trsong/Leaf-Similar-Trees-3](https://replit.com/@trsong/Leaf-Similar-Trees-3)
```py
import unittest

def is_leaf_similar(t1, t2):
    return dfs_path(t1) == dfs_path(t2)


def dfs_path(root):
    if not root:
        return []

    res = []
    stack = [root]
    while stack:
        cur = stack.pop()
        if cur.left is None and cur.right is None:
            res.append(cur.val)
            continue

        for child in [cur.right, cur.left]:
            if child is None:
                continue
            stack.append(child)
    return res
            

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class IsLeafSimilarSpec(unittest.TestCase):
    def test_example(self):
        """
            3
           / \ 
          5   1
           \
            2 

            7
           / \ 
          2   1
           \
            2 
        """
        t1 = TreeNode(3, TreeNode(5, right=TreeNode(2)), TreeNode(1))
        t2 = TreeNode(7, TreeNode(2, right=TreeNode(2)), TreeNode(1))
        self.assertTrue(is_leaf_similar(t1, t2))

    def test_both_empty(self):
        self.assertTrue(is_leaf_similar(None, None))

    def test_one_tree_empty(self):
        self.assertFalse(is_leaf_similar(TreeNode(0), None))

    def test_tree_of_different_depths(self):
        """
          1
         / \
        2   3

           1
         /   \
        5     4
         \   /
          2 3
        """
        t1 = TreeNode(1, TreeNode(2), TreeNode(3))
        t2l = TreeNode(5, right=TreeNode(2))
        t2r = TreeNode(4, TreeNode(3))
        t2 = TreeNode(1, t2l, t2r)
        self.assertTrue(is_leaf_similar(t1, t2))

    def test_tree_with_different_number_of_leaves(self):
        """
          1
         / \
        2   3

           1
         /   
        2     
        """
        t1 = TreeNode(1, TreeNode(2), TreeNode(3))
        t2 = TreeNode(1, TreeNode(2))
        self.assertFalse(is_leaf_similar(t1, t2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Feb 2, 2022 \[Medium\] Generate Binary Search Trees
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

**Solution:** [https://replit.com/@trsong/Generate-Binary-Search-Trees-with-N-Nodes-2](https://replit.com/@trsong/Generate-Binary-Search-Trees-with-N-Nodes-2)
```py
import unittest

def generate_bst(n):
    if n < 1:
        return []

    return list(generate_bst_between(1, n))

def generate_bst_between(lo, hi):
    if lo > hi:
        yield None
    
    for val in range(lo, hi + 1):
        for left_child in generate_bst_between(lo, val - 1):
            for right_child in generate_bst_between(val + 1, hi):
                yield TreeNode(val, left_child, right_child)


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

### Feb 1, 2022 \[Medium\] Maximum Path Sum in Binary Tree
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

**Solution with Recursion:** [https://replit.com/@trsong/Find-Maximum-Path-Sum-in-Binary-Tree-2](https://replit.com/@trsong/Find-Maximum-Path-Sum-in-Binary-Tree-2)
```py
import unittest

def max_path_sum(tree):
    return max_path_sum_recur(tree)[0]


def max_path_sum_recur(tree):
    if not tree:
        return 0, 0

    left_res, left_max_path = max_path_sum_recur(tree.left)
    right_res, right_max_path = max_path_sum_recur(tree.right)
    max_path = tree.val + max(left_max_path, right_max_path)
    max_sum = tree.val + left_max_path + right_max_path
    res = max(left_res, right_res, max_sum)
    return res, max_path
    

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