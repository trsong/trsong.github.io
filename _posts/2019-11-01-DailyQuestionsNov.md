---
layout: post
title:  "Daily Coding Problems Nov to Jan"
date:   2019-11-01 22:22:32 -0700
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


### Nov 25, 2019 \[Medium\] Maximum Amount of Money from a Game
--- 
> **Question:** In front of you is a row of N coins, with values v1, v1, ..., vn.
>
> You are asked to play the following game. You and an opponent take turns choosing either the first or last coin from the row, removing it from the row, and receiving the value of the coin.
>
> Write a program that returns the maximum amount of money you can win with certainty, if you move first, assuming your opponent plays optimally.

### Nov 24, 2019 \[Easy\] Markov Chain
--- 
> **Question:**You are given a starting state start, a list of transition probabilities for a Markov chain, and a number of steps num_steps. Run the Markov chain starting from start for num_steps and compute the number of times we visited each state.
>
> For example, given the starting state a, number of steps 5000, and the following transition probabilities:

```py
[
  ('a', 'a', 0.9),
  ('a', 'b', 0.075),
  ('a', 'c', 0.025),
  ('b', 'a', 0.15),
  ('b', 'b', 0.8),
  ('b', 'c', 0.05),
  ('c', 'a', 0.25),
  ('c', 'b', 0.25),
  ('c', 'c', 0.5)
]
One instance of running this Markov chain might produce { 'a': 3012, 'b': 1656, 'c': 332 }.
```

**My thoughts:** In order to uniformly choose among the probablity of next state, we can align the probably in a row and calculate accumulated probability. eg. `[a: .1, b: .2, c: .7]` => `[a: 0 ~ 0.1, b: 0.1 ~ 0.3, c: 0.3 ~ 1.0]` Thus, by generate a random number we can tell what's the next state based on probablity.

**Solution:** [https://repl.it/@trsong/Markov-Chain](https://repl.it/@trsong/Markov-Chain)
```py
import random
import unittest

def binary_search(dst_probs, target_prob):
    lo = 0
    hi = len(dst_probs) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if dst_probs[mid] < target_prob:
            lo = mid + 1
        else:
            hi = mid
    return lo


def markov_chain(start_state, num_steps, transition_probabilities):
    transition_map = {}
    for src, dst, prob in transition_probabilities:
        if src not in transition_map:
            transition_map[src] = [[], []]
        transition_map[src][0].append(dst)
        transition_map[src][1].append(prob)
    
    for src in transition_map:
        dst_probs = transition_map[src][1]
        for i in xrange(1, len(dst_probs)):
            # accumulate probability
            dst_probs[i] += dst_probs[i-1]
    
    state_counter = {}
    current_state = start_state
    for _ in xrange(num_steps):
        state_counter[current_state] = state_counter.get(current_state, 0) + 1
        dst_list, des_prob = transition_map[current_state]
        random_number = random.random()
        next_state_index = binary_search(des_prob, random_number)
        next_state = dst_list[next_state_index]
        current_state = next_state

    return state_counter


class MarkovChainSpec(unittest.TestCase):
    def test_example(self):
        start_state = 'a'
        num_steps = 5000
        transition_probabilities = [
            ('a', 'a', 0.9),
            ('a', 'b', 0.075),
            ('a', 'c', 0.025),
            ('b', 'a', 0.15),
            ('b', 'b', 0.8),
            ('b', 'c', 0.05),
            ('c', 'a', 0.25),
            ('c', 'b', 0.25),
            ('c', 'c', 0.5)
        ]
        random.seed(42)
        expected = {'a': 3205, 'c': 330, 'b': 1465}
        self.assertEqual(expected, markov_chain(start_state, num_steps, transition_probabilities))
    
    def test_transition_matrix2(self):
        start_state = '4'
        num_steps = 100
        transition_probabilities = [
            ('1', '2', 0.23),
            ('2', '1', 0.09),
            ('1', '4', 0.77),
            ('2', '3', 0.06),
            ('2', '6', 0.85),
            ('6', '2', 0.62),
            ('3', '4', 0.63),
            ('3', '6', 0.37),
            ('6', '5', 0.38),
            ('5', '6', 1),
            ('4', '5', 0.65),
            ('4', '6', 0.35),

        ]
        random.seed(42)
        expected = {'1': 3, '3': 2, '2': 27, '5': 21, '4': 4, '6': 43}
        self.assertEqual(expected, markov_chain(start_state, num_steps, transition_probabilities))

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 23, 2019 \[Hard\] Number of Subscribers during a Given Interval
--- 
> **Question:** You are given an array of length 24, where each element represents the number of new subscribers during the corresponding hour. Implement a data structure that efficiently supports the following:
>
- `update(hour: int, value: int)`: Increment the element at index hour by value.
- `query(start: int, end: int)`: Retrieve the number of subscribers that have signed up between start and end (inclusive).
>
> You can assume that all values get cleared at the end of the day, and that you will not be asked for start and end values that wrap around midnight.

**My thoughts:** Binary-indexed Tree a.k.a BIT or Fenwick Tree is used for dynamic range query. Basically, it allows efficiently query for sum of a continous interval of values.

BIT Usage:
```py
bit = BIT(4)
bit.query(3) # => 0
bit.update(index=0, delta=1) # => [1, 0, 0, 0]
bit.update(index=0, delta=1) # => [2, 0, 0, 0]
bit.update(index=2, delta=3) # => [2, 0, 3, 0]
bit.query(2) # => 2 + 0 + 3 = 5
bit.update(index=1, delta=-1) # => [2, -1, 3, 0]
bit.query(1) # => 2 + -1 = 1
```

Here all we need to do is to refactor the interface to adapt to SubscriberTracker.

**Solution with BIT:** [https://repl.it/@trsong/Number-of-Subscribers-during-a-Given-Interval](https://repl.it/@trsong/Number-of-Subscribers-during-a-Given-Interval)
```py
import unittest

class SubscriberTracker(object):
    def __init__(self, n=24):
        self.bit_tree = [0] * (n+1)

    @staticmethod
    def least_significant_bit(num):
        return num & -num

    def update(self, hour, value):
        bit_index = hour + 1
        while bit_index < len(self.bit_tree):
            self.bit_tree[bit_index] += value
            bit_index += SubscriberTracker.least_significant_bit(bit_index)

    def query_from_begining(self, hour):
        bit_index = hour + 1
        res = 0
        while bit_index > 0:
            res += self.bit_tree[bit_index]
            bit_index -= SubscriberTracker.least_significant_bit(bit_index)
        return res

    def query(self, start, end):
        return self.query_from_begining(end) - self.query_from_begining(start-1)


class SubscriberTrackerSpec(unittest.TestCase):
    def test_query_without_update(self):
        st = SubscriberTracker()
        self.assertEqual(0, st.query(0, 23))
        self.assertEqual(0, st.query(3, 5))
    
    def test_update_should_affect_query_value(self):
        st = SubscriberTracker()
        st.update(5, 10)
        st.update(10, 15)
        st.update(12, 20)
        self.assertEqual(0, st.query(0, 4))
        self.assertEqual(10, st.query(0, 5))
        self.assertEqual(10, st.query(0, 9))
        self.assertEqual(25, st.query(0, 10))
        self.assertEqual(45, st.query(0, 13))
        st.update(3, 2)
        self.assertEqual(2, st.query(0, 4))
        self.assertEqual(12, st.query(0, 5))
        self.assertEqual(12, st.query(0, 9))
        self.assertEqual(27, st.query(0, 10))
        self.assertEqual(47, st.query(0, 13))

    def test_number_of_subscribers_can_decrease(self):
        st = SubscriberTracker()
        st.update(10, 5)
        st.update(20, 10)
        self.assertEqual(10, st.query(15, 23))
        st.update(12, -3)
        self.assertEqual(10, st.query(15, 23))
        st.update(17, -7)
        self.assertEqual(3, st.query(15, 23))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 22, 2019 \[Easy\] Compare Version Numbers
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

**Solution with Two Pointers:** [https://repl.it/@trsong/Compare-Version-Numbers](https://repl.it/@trsong/Compare-Version-Numbers)
```py
import unittest

def version_number_compare(v1, v2):
    i = j = 0
    n, m = len(v1), len(v2)

    while i < n or j < m:
        sub_v1 = 0
        sub_v2 = 0
        while i < n and v1[i].isdigit():
            sub_v1 = 10 * sub_v1 + int(v1[i])
            i += 1

        while j < m and v2[j].isdigit():
            sub_v2 = 10 * sub_v2 + int(v2[j])
            j += 1

        i += 1
        j += 1
        if sub_v1 < sub_v2:
            return -1
        elif sub_v1 > sub_v2:
            return 1
    return 0


class VersionNumberCompareSpec(unittest.TestCase):
    def test_example1(self):
        version1 = "1.0.33"
        version2 = "1.0.27"
        self.assertEqual(1, version_number_compare(version1, version2))

    def test_example2(self):
        version1 = "0.1"
        version2 = "1.1"
        self.assertEqual(-1, version_number_compare(version1, version2))

    def test_example3(self):
        version1 = "1.01"
        version2 = "1.001"
        self.assertEqual(0, version_number_compare(version1, version2))

    def test_example4(self):
        version1 = "1.0"
        version2 = "1.0.0"
        self.assertEqual(0, version_number_compare(version1, version2))

    def test_unspecified_version_numbers(self):
        self.assertEqual(0, version_number_compare("", ""))
        self.assertEqual(-1, version_number_compare("", "1"))
        self.assertEqual(1, version_number_compare("2", ""))

    def test_unaligned_zeros(self):
        version1 = "00000.00000.00000.0"
        version2 = "0.00000.000.00.00000.000.000.0"
        self.assertEqual(0, version_number_compare(version1, version2))

    def test_same_version_yet_unaligned(self):
        version1 = "00001.001"
        version2 = "1.000001.0000000.0000"
        self.assertEqual(0, version_number_compare(version1, version2))

    def test_different_version_numbers(self):
        version1 = "1.2.3.4"
        version2 = "1.2.3.4.5"
        self.assertEqual(-1, version_number_compare(version1, version2))

    def test_different_version_numbers2(self):
        version1 = "3.2.1"
        version2 = "3.1.2.3"
        self.assertEqual(1, version_number_compare(version1, version2))

    def test_different_version_numbers3(self):
        version1 = "00001.001.0.1"
        version2 = "1.000001.0000000.0000"
        self.assertEqual(1, version_number_compare(version1, version2))

    def test_without_dots(self):
        version1 = "32123"
        version2 = "3144444"
        self.assertEqual(-1, version_number_compare(version1, version2))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 21, 2019 \[Easy\] GCD of N Numbers
--- 
> **Question:** Given `n` numbers, find the greatest common denominator between them.
>
> For example, given the numbers `[42, 56, 14]`, return `14`.


**Solution:** [https://repl.it/@trsong/GCD-of-N-Numbers](https://repl.it/@trsong/GCD-of-N-Numbers)
```py
import unittest

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a 


def gcd_n_numbers(nums):
    res = nums[0]
    for num in nums:
        # gcd(a, b, c) = gcd(a, gcd(b, c))
        res = gcd(res, num)
    return res


class GCDNNumberSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(14, gcd_n_numbers([42, 56, 14]))

    def test_co_prime_numbers(self):
        self.assertEqual(1, gcd_n_numbers([6, 5, 7, 11]))

    def test_composite_numbers(self):
        self.assertEqual(11, gcd_n_numbers([11 * 3, 11 * 5, 11 * 7]))

    def test_even_numbers(self):
        self.assertEqual(2, gcd_n_numbers([2, 8, 6, 4]))
    
    def test_odd_numbers(self):
        self.assertEqual(5, gcd_n_numbers([3 * 5, 3 * 2 * 5, 5 * 2 * 2]))


if __name__ == '__main__':
    unittest.main(exit=False)

```

### Nov 20, 2019 LC 138 \[Medium\] Copy List with Random Pointer
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

**Solution:** [https://repl.it/@trsong/Copy-List-with-Random-Pointer](https://repl.it/@trsong/Copy-List-with-Random-Pointer)
```py
import unittest


def deep_copy(head):
    if not head:
        return None

    # Step1: Duplicate each node and insert them in every other place: n1 -> copy_n1 -> n2 -> copy_n2 ... -> nk -> copy_nk
    p = head
    while p:
        p.next = ListNode(p.val, p.next)
        p = p.next.next
    
    # Step2: Connect random pointer of duplicated node to the duplicated node of random pointer. ie.  copy_node(n1).random = copy_node(n1.random). Note in Step1: copy_node(n1) == n1.next
    p = head
    while p:
        if p.random:
            p.next.random = p.random.next
        p = p.next.next
    
    # Step3: Partition list into even sub-list and odd sub-list. And the odd sub-list is the deep-copy list
    p = head
    new_head = p.next
    while p:
        new_p = p.next
        p.next = p.next.next
        if p.next:
            new_p.next = p.next.next
        else:
            new_p.next = None
        p = p.next
    
    return new_head
    

### Testing Utiltities
class ListNode(object):
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random
    
    def __eq__(self, other):
        if not self.equal(other):
            print "List1 != List2 where"
            print "List1:"
            print str(self)
            print "List2:"
            print str(other)
            print "\n"
            return False
        else:
            return True

    def equal(self, other):
        if other is not None:
            is_random_valid = self.random is None and other.random is None or self.random is not None and other.random is not None and self.random.val == other.random.val
            return is_random_valid and self.val == other.val and self.next == other.next
        else:
            return False
        

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
        
        return "List: [{}].\nRandom[{}]".format(','.join(lst), ','.join(random))
            


class DeepCopySpec(unittest.TestCase):
    def test_empty_list(self):
        self.assertIsNone(deep_copy(None))
    
    def test_list_with_random_point_to_itself(self):
        n = ListNode(1)
        n.random = n
        self.assertEqual(deep_copy(n), n)

    def test_random_pointer_is_None(self):
        # 1 -> 2 -> 3 -> 4
        n4 = ListNode(4)
        n3 = ListNode(3, n4)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)
        self.assertEqual(deep_copy(n1), n1)

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
        self.assertEqual(deep_copy(n1), n1)

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
        self.assertEqual(deep_copy(n1), n1)

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
        self.assertEqual(deep_copy(n1), n1)

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
        self.assertEqual(deep_copy(n1), n1)
        

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 19, 2019 LC 692 \[Medium\] Top K Frequent words
--- 
> **Question:** Given a non-empty list of words, return the k most frequent words. The output should be sorted from highest to lowest frequency, and if two words have the same frequency, the word with lower alphabetical order comes first. Input will contain only lower-case letters.

**Example 1:**
```py
Input: ["i", "love", "leapcode", "i", "love", "coding"], k = 2
Output: ["i", "love"]
Explanation: "i" and "love" are the two most frequent words.
    Note that "i" comes before "love" due to a lower alphabetical order.
```

**Example 2:**
```py
Input: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
Output: ["the", "is", "sunny", "day"]
Explanation: "the", "is", "sunny" and "day" are the four most frequent words,
    with the number of occurrence being 4, 3, 2 and 1 respectively.
```

**My thoughts:** Record the frequence of each word, put them into a max heap, finally the final result is the top k element from the max heap. 

**Solution with PriorityQueue:** [https://repl.it/@trsong/Top-K-Frequent-words](https://repl.it/@trsong/Top-K-Frequent-words)
```py
import unittest
from Queue import PriorityQueue

def top_k_freq_words(words, k):
    histogram = {}
    for word in words:
        histogram[word] = histogram.get(word, 0) + 1
        
    pq = PriorityQueue()
    for word, count in histogram.items():
        pq.put((-count, word))
    
    res = []
    for _ in xrange(k):
        _, word = pq.get()
        res.append(word)
    
    return res


class TopKFreqWordSpec(unittest.TestCase):
    def test_example1(self):
        input = ["i", "love", "leapcode", "i", "love", "coding"]
        k = 2
        expected = ["i", "love"]
        self.assertEqual(expected, top_k_freq_words(input, k))

    def test_example2(self):
        input =  ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"]
        k = 4
        expected = ["the", "is", "sunny", "day"]
        self.assertEqual(expected, top_k_freq_words(input, k))

    def test_same_count_words(self):
        input =  ["c", "cb", "cba", "cdbaa"]
        k = 3
        expected = ["c", "cb", "cba"]
        self.assertEqual(expected, top_k_freq_words(input, k))

    def test_same_count_words2(self):
        input =  ["a", "c", "b", "d"]
        k = 3
        expected = ["a", "b", "c"]
        self.assertEqual(expected, top_k_freq_words(input, k))
    
if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 18, 2019 \[Medium\] Intersection of Two Unsorted Arrays
--- 
> **Question:** Given two arrays, write a function to compute their intersection - the intersection means the numbers that are in both arrays.
> 
> **Note:**
> - Each element in the result must be unique.
> - The result can be in any order.

**Example 1:**
```py
Input: nums1 = [1, 2, 2, 1], nums2 = [2, 2]
Output: [2]
```

**Example 2:**
```py
Input: nums1 = [4, 9, 5], nums2 = [9, 4, 9, 8, 4]
Output: [9, 4]
```

**Solution:** [https://repl.it/@trsong/Intersection-of-Two-Unsorted-Arrays](https://repl.it/@trsong/Intersection-of-Two-Unsorted-Arrays)
```py
import unittest

def binary_search(nums, target):
    lo = 0
    hi = len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return True
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False


def remove_duplicates(nums):
    nums.sort()
    j = 0
    for i in xrange(len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        nums[j] = nums[i]
        j += 1
    
    for _ in xrange(j, len(nums)):
        nums.pop()
    
    return nums


def intersection(nums1, nums2):
    if not nums1 or not nums2:
        return []

    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    nums1.sort()
    res = []
    for i, num in enumerate(nums2):
        if binary_search(nums1, num):
            res.append(num)
    return remove_duplicates(res)


class IntersectionSpec(unittest.TestCase):
    def assert_array(self, arr1, arr2):
        self.assertEqual(sorted(arr1), sorted(arr2))

    def test_example1(self):
        nums1 = [1, 2, 2, 1]
        nums2 = [2, 2]
        expected = [2]
        self.assert_array(expected, intersection(nums1, nums2))
        self.assert_array(expected, intersection(nums2, nums1))

    def test_example2(self):
        nums1 = [4, 9, 5]
        nums2 = [9, 4, 9, 8, 4]
        expected = [9, 4]
        self.assert_array(expected, intersection(nums1, nums2))

    def test_empty_array(self):
        self.assertEqual([], intersection([], []))

    def test_empty_array2(self):
        self.assertEqual([], intersection([1, 1, 1], []))
        self.assertEqual([], intersection([], [1, 2, 3]))

    def test_array_with_duplicated_elements(self):
        nums1 = [1, 1, 1]
        nums2 = [1, 2, 3, 4]
        expected = [1]
        self.assert_array(expected, intersection(nums1, nums2))

    def test_array_with_duplicated_elements2(self):
        nums1 = [1, 2, 3]
        nums2 = [1, 1, 1, 2, 2, 2, 4, 4, 4, 4]
        expected = [1, 2]
        self.assert_array(expected, intersection(nums1, nums2))

    def test_array_with_not_overlapping_elements(self):
        nums1 = [1, 2, 3]
        nums2 = [4, 5, 6]
        expected = []
        self.assert_array(expected, intersection(nums1, nums2))

    
if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 17, 2019 \[Easy\] Plus One
--- 
> **Question:** Given a non-empty array where each element represents a digit of a non-negative integer, add one to the integer. The most significant digit is at the front of the array and each element in the array contains only one digit. Furthermore, the integer does not have leading zeros, except in the case of the number '0'.

**Example:**
```py
Input: [2,3,4]
Output: [2,3,5]
```

**Solution:** [https://repl.it/@trsong/Plus-One](https://repl.it/@trsong/Plus-One)
```py
import unittest

def add1(digits):
    carryover = 1
    for i in xrange(len(digits)-1, -1, -1):
        digits[i] += carryover
        if digits[i] >= 10:
            carryover = 1
            digits[i] %= 10
        else:
            break
    
    if digits[0] == 0:
        digits.insert(0, 1)
    
    return digits


class Add1Spec(unittest.TestCase):
    def test_example(self):
        self.assertEqual([2, 3, 5], add1([2, 3, 4]))

    def test_zero(self):
        self.assertEqual([1], add1([0]))

    def test_carryover(self):
        self.assertEqual([1, 0], add1([9]))

    def test_carryover_and_early_break(self):
        self.assertEqual([2, 8, 3, 0, 0], add1([2, 8, 2, 9, 9]))

    def test_early_break(self):
        self.assertEqual([1, 0, 0, 1], add1([1, 0, 0, 0]))

    def test_carryover2(self):
        self.assertEqual([1, 0, 0, 0, 0], add1([9, 9, 9, 9]))

if __name__ == '__main__':
    unittest.main(exit=False)

```

### Nov 16, 2019 \[Easy\] Exists Overlap Rectangle
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


**Solution:** [https://repl.it/@trsong/Exists-Overlap-Rectangle](https://repl.it/@trsong/Exists-Overlap-Rectangle)
```py
import unittest

def get_bottom_right(rect):
    x, y = rect.get("top_left")
    width, height = rect.get("dimensions")
    return x + width, y - height


def is_overlapped_rectange(rect1, rect2):
    x1, y1 = rect1.get("top_left")
    x2, y2 = get_bottom_right(rect1)
    x3, y3 = rect2.get("top_left")
    x4, y4 = get_bottom_right(rect2)
    x_not_overlap = x2 <= x3 or x1 >= x4
    y_not_overlap = y1 <= y4 or y3 <= y2
    return not x_not_overlap and not y_not_overlap


def exists_overlap_rectangle(rectangles):
    if not rectangles:
        return False
    n = len(rectangles)
    sorted_rectangles = sorted(rectangles, key=lambda rect: rect.get("top_left"))
    for i, rect in enumerate(sorted_rectangles):
        x, _ = rect.get("top_left")
        width, _ = rect.get("dimensions")
        j = i + 1
        while j < n and sorted_rectangles[j].get("top_left")[0] <= x + width:
            if is_overlapped_rectange(sorted_rectangles[j], rect):
                return True
            j += 1 
    return False


class ExistsOverlapRectangleSpec(unittest.TestCase):
    def test_example(self):
        rectangles = [
            {
                "top_left": (1, 4),
                "dimensions": (3, 3) # width, height
            }, {
                "top_left": (-1, 3),
                "dimensions": (2, 1)
            },{
                "top_left": (0, 5),
                "dimensions": (4, 3)
            }]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_empty_rectangle_list(self):
        self.assertFalse(exists_overlap_rectangle([]))
    
    def test_two_overlap_rectangle(self):
        rectangles = [
            {
                "top_left": (0, 1),
                "dimensions": (1, 3) # width, height
            }, {
                "top_left": (-1, 0),
                "dimensions": (3, 1)
            }]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_two_overlap_rectangle_form_a_cross(self):
        rectangles = [
            {
                "top_left": (-1, 1),
                "dimensions": (3, 2) # width, height
            }, {
                "top_left": (0, 0),
                "dimensions": (1, 1)
            }]
        self.assertTrue(exists_overlap_rectangle(rectangles))

    def test_same_y_coord_not_overlap(self):
        rectangles = [
            {
                "top_left": (0, 0),
                "dimensions": (1, 1) # width, height
            }, {
                "top_left": (1, 0),
                "dimensions": (2, 2)
            },{
                "top_left": (3, 0),
                "dimensions": (5, 2)
            }]
        self.assertFalse(exists_overlap_rectangle(rectangles))

    def test_same_y_coord_overlap(self):
        rectangles = [
            {
                "top_left": (0, 0),
                "dimensions": (1, 1) # width, height
            }, {
                "top_left": (1, 0),
                "dimensions": (2, 2)
            },{
                "top_left": (3, 0),
                "dimensions": (5, 2)
            }]
        self.assertFalse(exists_overlap_rectangle(rectangles))
    
    def test_rectangles_in_different_quadrant(self):
        rectangles = [
            {
                "top_left": (1, 1),
                "dimensions": (2, 2) # width, height
            }, {
                "top_left": (-1, 1),
                "dimensions": (2, 2)
            },{
                "top_left": (1, -1),
                "dimensions": (2, 2)
            },{
                "top_left": (-1, -1),
                "dimensions": (2, 2)
            }]
        self.assertFalse(exists_overlap_rectangle(rectangles))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 15, 2019 \[Hard\] Maximum Spanning Tree
--- 
> **Question:** Recall that the minimum spanning tree is the subset of edges of a tree that connect all its vertices with the smallest possible total edge weight. 
> 
> Given an undirected graph with weighted edges, compute the maximum weight spanning tree.

**My thoughts:** Both Kruskal's and Prim's Algorithm works for this question. The idea is to flip all edge weight into negative and then apply either of previous algorithm until find a spanning tree. Check [this question](https://trsong.github.io/python/java/2019/05/01/DailyQuestions/#jul-6-2019-hard-power-supply-to-all-cities) for solution with Kruskal's Algorithm.

**Solution with Prim's Algorithm:** [https://repl.it/@trsong/Maximum-Spanning-Tree](https://repl.it/@trsong/Maximum-Spanning-Tree)
```py
import unittest
from Queue import PriorityQueue
import sys


def max_spanning_tree(vertices, edges):
    if not edges:
        return []

    neighbor = [None] * vertices
    for u, v, w in edges:
        if not neighbor[u]:
            neighbor[u] = []
        neighbor[u].append((v, w))

        if not neighbor[v]:
            neighbor[v] = []
        neighbor[v].append((u, w))
    
    visited = [False] * vertices
    pq = PriorityQueue()
    pq.put((0, 0, None))
    res = []

    while not pq.empty():
        _, dst, src = pq.get()
        if visited[dst]:
            continue

        visited[dst] = True
        if src is not None:
            res.append((src, dst))
        
        if len(res) == vertices - 1:
            break
        
        u = dst
        for v, w in neighbor[u]:
            if visited[v]:
                continue
            pq.put((-w, v, u))

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

### Nov 14, 2019 \[Easy\] Smallest Number that is not a Sum of a Subset of List
--- 
> **Question:** Given a sorted array, find the smallest positive integer that is not the sum of a subset of the array.
>
> For example, for the input `[1, 2, 3, 10]`, you should return `7`.

**My thoughts:** The idea of solving this problem comes from mathematical induction. As the number list is sorted in ascending order, we take element from list one by one. Can consider with the help of the such element, how does it help to push the max possible sum? eg, Let's walk through `[1, 2, 3, 10]`

- `n = 0`. our goal is to check if there is any subset added up to `1`. That's easy, check `nums[0]` to see if it equals `1`.
- `n = 1`. Now we move to next element say `2`. Currently we can from `1`. Now with the help of `2`, additionally we can form `2` and `2 + 1`. It seems we can form all number from 1 to 3.
- `n = 2`. As we can form any number from `1` to `3`. The current element is `3`. With help of this element, additionally we can form `3`, `3 + 1`, `3 + 2`, `3 + 3`. It seems we can form all number from `1` to `6`.
- `n = 3`. As we can form any number from `1` to `6`. The current element is `10`. With the help of this element, additionally we can only get `10`, `10 + 1`, ..., `10 + 6`. However, last round we expand the possible sum to `6`. Now the smallest we can get is `10`, gives a gap. Therefore we return `6 + 1`. 


**Solution by Math Induction:**[https://repl.it/@trsong/Smallest-Number-that-is-not-a-Sum-of-a-Subset-of-List](https://repl.it/@trsong/Smallest-Number-that-is-not-a-Sum-of-a-Subset-of-List)
```py
import unittest

def smallest_subset_sum(nums):
    if not nums:
        return 1

    max_sum_ = 0
    for num in nums:
        if num > max_sum + 1:
            break
        max_sum += num
    return max_sum + 1


class SmallestSubsetSumSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(7, smallest_subset_sum([1, 2, 3, 10]))

    def test_empty_array(self):
        self.assertEqual(1, smallest_subset_sum([]))

    def test_subset_sum_has_gap1(self):
        self.assertEqual(2, smallest_subset_sum([1, 3, 6, 10, 11, 15]))

    def test_subset_sum_has_gap2(self):
        self.assertEqual(4, smallest_subset_sum([1, 2, 5, 10, 20, 40]))

    def test_subset_sum_has_gap3(self):
        self.assertEqual(1, smallest_subset_sum([101, 102, 104]))

    def test_subset_sum_covers_all1(self):
        self.assertEqual(5, smallest_subset_sum([1, 1, 1, 1]))

    def test_subset_sum_covers_all2(self):
        self.assertEqual(26, smallest_subset_sum([1, 2, 4, 8, 10]))

    def test_subset_sum_covers_all3(self):
        self.assertEqual(16, smallest_subset_sum([1, 2, 4, 8]))
        
    def test_subset_sum_covers_all4(self):
        self.assertEqual(10, smallest_subset_sum([1, 1, 3, 4]))

    def test_subset_sum_covers_all5(self):
        self.assertEqual(22, smallest_subset_sum([1, 2, 3, 4, 5, 6]))
 

if __name__ == '__main__':
    unittest.main(exit=False)
```





### Nov 13, 2019 LC 301 \[Hard\] Remove Invalid Parentheses
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


**Solution with Backtracking:** [https://repl.it/@trsong/Remove-Invalid-Parentheses](https://repl.it/@trsong/Remove-Invalid-Parentheses)
```py
import unittest

def remove_invalid_parenthese(s):
    invalid_open = 0
    invalid_close = 0
    for c in s:
        if c == ')' and invalid_open == 0:
            invalid_close += 1
        elif c == '(':
            invalid_open += 1
        elif c == ')':
            invalid_open -= 1
    
    res = []
    backtrack(s, 0, res, invalid_open, invalid_close)
    return res


def backtrack(s, next_index, res, invalid_open, invalid_close):
    if invalid_open == 0 and invalid_close == 0:
        if is_valid(s):
            res.append(s)
    else:
        for i in xrange(next_index, len(s)):
            c = s[i]
            if c == '(' and invalid_open > 0 or c == ')' and invalid_close > 0:
                if i > next_index and s[i] == s[i-1]:
                    # skip consecutive same letters
                    continue

                # update s with c removed
                updated_s = s[:i] + s[i+1:]
                if c == '(':
                    backtrack(updated_s, i, res, invalid_open-1, invalid_close)
                elif c == ')':
                    backtrack(updated_s, i, res, invalid_open, invalid_close-1)


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

### Nov 12, 2019 \[Easy\] Busiest Period in the Building
--- 
> **Question:** You are given a list of data entries that represent entries and exits of groups of people into a building. 
> 
> Find the busiest period in the building, that is, the time with the most people in the building. Return it as a pair of (start, end) timestamps. 
> 
> You can assume the building always starts off and ends up empty, i.e. with 0 people inside.
>
> An entry looks like this:

```py
{"timestamp": 1526579928, count: 3, "type": "enter"}

This means 3 people entered the building. An exit looks like this:

{"timestamp": 1526580382, count: 2, "type": "exit"}

This means that 2 people exited the building. timestamp is in Unix time.
```


**Solution:** [https://repl.it/@trsong/Busiest-Period-in-the-Building](https://repl.it/@trsong/Busiest-Period-in-the-Building) 
```py
import unittest

def busiest_period(event_entries):
    sorted_events = sorted(event_entries, key=lambda e: e.get("timestamp"))
    accu = 0
    max_index = 0
    max_accu = 0
    for i, event in enumerate(sorted_events):
        count = event.get("count")
        accu += count if event.get("type") == "enter" else -count
        if accu > max_accu:
            max_accu = accu
            max_index = i
            
    start = sorted_events[max_index].get("timestamp")
    end = sorted_events[max_index+1].get("timestamp")
    return start, end


class BusiestPeriodSpec(unittest.TestCase):
    def test_example(self):
        # Number of people: 0, 3, 1, 0
        events = [
            {"timestamp": 1526579928, "count": 3, "type": "enter"},
            {"timestamp": 1526580382, "count": 2, "type": "exit"},
            {"timestamp": 1526600382, "count": 1, "type": "exit"}
        ]
        self.assertEqual((1526579928, 1526580382), busiest_period(events))

    def test_multiple_entering_and_exiting(self):
        # Number of people: 0, 3, 1, 7, 8, 3, 2
        events = [
            {"timestamp": 1526579928, "count": 3, "type": "enter"},
            {"timestamp": 1526580382, "count": 2, "type": "exit"},
            {"timestamp": 1526579938, "count": 6, "type": "enter"},
            {"timestamp": 1526579943, "count": 1, "type": "enter"},
            {"timestamp": 1526579944, "count": 0, "type": "enter"},
            {"timestamp": 1526580345, "count": 5, "type": "exit"},
            {"timestamp": 1526580351, "count": 3, "type": "exit"}
        ]
        self.assertEqual((1526579943, 1526579944), busiest_period(events))

    def test_timestamp_not_sorted(self):
        # Number of people: 0, 1, 3, 0
        events = [
            {"timestamp": 2, "count": 2, "type": "enter"},
            {"timestamp": 3, "count": 3, "type": "exit"},
            {"timestamp": 1, "count": 1, "type": "enter"}
        ]
        self.assertEqual((2, 3), busiest_period(events))

    def test_max_period_reach_a_tie(self):
        # Number of people: 0, 1, 10, 1, 10, 1, 0
        events = [
            {"timestamp": 5, "count": 9, "type": "exit"},
            {"timestamp": 3, "count": 9, "type": "exit"},
            {"timestamp": 6, "count": 1, "type": "exit"},
            {"timestamp": 1, "count": 1, "type": "enter"},
            {"timestamp": 4, "count": 9, "type": "enter"},
            {"timestamp": 2, "count": 9, "type": "enter"}
        ]
        self.assertEqual((2, 3), busiest_period(events))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 11, 2019 \[Easy\] Full Binary Tree
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

**Solution:** [https://repl.it/@trsong/Full-Binary-Tree](https://repl.it/@trsong/Full-Binary-Tree)
```py
import unittest
import copy

def remove_partial_nodes(root):
    if root is None:
        return None

    root.left = remove_partial_nodes(root.left)
    root.right = remove_partial_nodes(root.right)
    
    if root.left is None and root.right is not None:
        right_child = root.right
        root.right = None
        return right_child
    elif root.right is None and root.left is not None:
        left_child = root.left
        root.left = None
        return left_child
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
        self.assertEqual(expected_tree, remove_partial_nodes(original_tree))

    def test_empty_tree(self):        
        self.assertIsNone(None, remove_partial_nodes(None))

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
        self.assertEqual(expected_tree, remove_partial_nodes(original_tree))

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
        self.assertEqual(expected_tree, remove_partial_nodes(original_tree))

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
        self.assertEqual(expected_tree, remove_partial_nodes(original_tree))

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
        self.assertEqual(expected_tree, remove_partial_nodes(original_tree))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 10, 2019 \[Medium\] Maximum Path Sum in Binary Tree
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

**My thoughts:** The maximum path sum can either roll up from maximum of recursive children value or calculate based on maximum left path sum and right path sum.

Example1: Final result rolls up from children
```py
     0
   /   \
  2     0
 / \   /
4   5 0
```

Example2: Final result is calculated based on max left path sum and right path sum

```py
    1
   / \
  2   3
 /   / \
8   0   5
   / \   \
  0   0   9
```

**Solution with Recursion:** [https://repl.it/@trsong/Maximum-Path-Sum-in-Binary-Tree](https://repl.it/@trsong/Maximum-Path-Sum-in-Binary-Tree)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_path_sum_helper(tree):
    if not tree:
        return 0, 0
    
    left_max_sum, left_max_path = max_path_sum_helper(tree.left)
    right_max_sum, right_max_path = max_path_sum_helper(tree.right)

    max_path = max(left_max_path, right_max_path) + tree.val
    path_sum = left_max_path + tree.val + right_max_path
    max_sum = max(left_max_sum, right_max_sum, path_sum)
    return max_sum, max_path


def max_path_sum(tree):
    return max_path_sum_helper(tree)[0]


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
    unittest.main(exit=False)
```

### Nov 9, 2019 \[Hard\] Order of Alien Dictionary
--- 
> **Question:** You come across a dictionary of sorted words in a language you've never seen before. Write a program that returns the correct order of letters in this language.
>
> For example, given `['xww', 'wxyz', 'wxyw', 'ywx', 'ywz']`, you should return `['x', 'z', 'w', 'y']`.

**My thoughts:** As the alien letters are topologically sorted, we can just mimic what topological sort with numbers and try to find pattern.

Suppose the dictionary contains: `01234`. Then the words can be `023, 024, 12, 133, 2433`. Notice that we can only find the relative order by finding first unequal letters between consecutive words. eg.  `023, 024 => 3 < 4`.  `024, 12 => 0 < 1`.  `12, 133 => 2 < 3`

With relative relation, we can build a graph with each occurring letters being veteces and edge `(u, v)` represents `u < v`. If there exists a loop that means we have something like `a < b < c < a` and total order not exists. Otherwise we preform a topological sort to generate the total order which reveals the alien dictionary. 

**Solution with Topological Sort:** [https://repl.it/@trsong/Order-of-Alien-Dictionary](https://repl.it/@trsong/Order-of-Alien-Dictionary)
```py
import unittest

class NodeState:
    UNVISITED = 0
    VISITING = 1
    VISITED = 2

def dictionary_order(sorted_words):
    if not sorted_words:
        return []

    char_order = {}
    for i in xrange(1, len(sorted_words)):
        prev_word = sorted_words[i-1]
        cur_word = sorted_words[i]
        for prev_char, cur_char in zip(prev_word, cur_word):
            if prev_char == cur_char:
                continue
            if prev_char not in char_order:
                char_order[prev_char] = set()
            char_order[prev_char].add(cur_char)
            break

    char_states = {}
    for word in sorted_words:
        for c in word:
            char_states[c] = NodeState.UNVISITED

    # Performe DFS on all unvisited nodes
    topo_order = []
    for char in char_states:
        if char_states[char] == NodeState.VISITED:
            continue
        
        stack = [char]
        while stack:
            cur_char = stack[-1]
            if char_states[cur_char] == NodeState.VISITED:
                topo_order.append(stack.pop())
                continue
            elif char_states[cur_char] == NodeState.VISITING:
                char_states[cur_char] = NodeState.VISITED
            else:
                char_states[cur_char] = NodeState.VISITING

            if cur_char not in char_order:
                continue
            
            for next_char in char_order[cur_char]:
                if char_states[next_char] == NodeState.UNVISITED:
                    stack.append(next_char)
                elif char_states[next_char] == NodeState.VISITING:
                    return None

    topo_order.reverse()
    return topo_order


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

### Nov 8, 2019 \[Hard\] Longest Common Subsequence of Three Strings
--- 
> **Question:** Write a program that computes the length of the longest common subsequence of three given strings. For example, given `"epidemiologist"`, `"refrigeration"`, and `"supercalifragilisticexpialodocious"`, it should return `5`, since the longest common subsequence is `"eieio"`.

**My thoughts:** The way to tackle 3 lcs of 3 strings is exactly the same as 2 strings. 

For 2 string case, we start from empty strings `s1 = ""` and `s2 = ""` and for each char we append we will need to decide whether or not we want to keep that char that gives the following 2 situations:

- `lcs(s1 + c, s2 + c)` if the new char to append in each string matches each other, then `lcs(s1 + c, s2 + c) = 1 + lcs(s1, s2)`
- `lcs(s1 + c1, s2 + c2)` if not match, we either append to s1 or s2, `lcs(s1 + c, s2 + c) = max(lcs(s1, s2 + c), lcs(s1 + c, s2))`

If we generalize it to 3 string case, we will have 
- `lcs(s1 + c, s2 + c, s3 + c) = 1 + lcs(s1, s2, s3)`
- `lcs(s1 + c, s2 + c, s3 + c) = max(lcs(s1 + c, s2, s3), lcs(s1, s2 + c, s3), lcs(s1, s2, s3 + c))`

One thing worth mentioning: 

`lcs(s1, s2, s2) != lcs(find_lcs(s1, s2), s3)` where find_cls returns string and lcs returns number. The reason we may run into this is that the local max of two strings isn't global max of three strings. 
eg. `s1 = 'abcde'`, `s2 = 'cdeabcd'`, `s3 = 'e'`. `find_lcs(s1, s2) = 'abcd'`. `lcs('abcd', 'e') = 0`. However `lcs(s1, s2, s3) = 1`



**Solution with DP:** [https://repl.it/@trsong/Longest-Common-Subsequence-of-Three-Strings](https://repl.it/@trsong/Longest-Common-Subsequence-of-Three-Strings)
```py
import unittest

def longest_common_subsequence(seq1, seq2, seq3):
    if not seq1 or not seq2 or not seq3:
        return 0
    
    m, n, p = len(seq1), len(seq2), len(seq3)
    # Let dp[m][n][p] represents lcs for seq with len m, n, p
    # dp[m][n][p] = dp[m-1][n-1][p-1] if last char is a match
    # dp[m][n][p] = max(dp[m-1][n][p], dp[m][n-1][p], dp[m][n][p-1])
    dp = [[[0 for _ in xrange(p+1)] for _ in xrange(n+1)] for _ in xrange(m+1)]
    for i in xrange(1, m+1):
        for j in xrange(1, n+1):
            for k in xrange(1, p+1):
                if seq1[i-1] == seq2[j-1] == seq3[k-1]:
                    dp[i][j][k] = dp[i-1][j-1][k-1] + 1
                else:
                    dp[i][j][k] = max(dp[i-1][j][k], dp[i][j-1][k], dp[i][j][k-1])
    return dp[m][n][p]
                    
                    
class LongestCommonSubsequenceSpec(unittest.TestCase):
    def test_example(self):
        seq1 = "epidemiologist"
        seq2 = "refrigeration"
        seq3 = "supercalifragilisticexpialodocious"
        self.assertEqual(5, longest_common_subsequence(seq1, seq2, seq3)) # eieio

    def test_empty_inputs(self):
        self.assertEqual(0, longest_common_subsequence("", "", ""))
        self.assertEqual(0, longest_common_subsequence("", "", "a"))
        self.assertEqual(0, longest_common_subsequence("", "a", "a"))
        self.assertEqual(0, longest_common_subsequence("a", "a", ""))

    def test_contains_common_subsequences(self):
        seq1 = "abcd1e2"  
        seq2 = "bc12ea"  
        seq3 = "bd1ea"
        self.assertEqual(3, longest_common_subsequence(seq1, seq2, seq3)) # b1e

    def test_contains_same_prefix(self):
        seq1 = "abc"  
        seq2 = "abcd"  
        seq3 = "abcde"
        self.assertEqual(3, longest_common_subsequence(seq1, seq2, seq3)) # abc

    def test_contains_same_subfix(self):
        seq1 = "5678"  
        seq2 = "345678"  
        seq3 = "12345678"
        self.assertEqual(4, longest_common_subsequence(seq1, seq2, seq3)) # 5678

    def test_lcs3_not_subset_of_lcs2(self):
        seq1 = "abedcf"
        seq2 = "ibdcfe"
        seq3 = "beklm"
        """
        LCS(seq1, seq2) = bdcf
        LCS(LCS(seq1, seq2), seq3) = b
        LCS(seq1, seq2, seq3) = be
        """
        self.assertEqual(2, longest_common_subsequence(seq1, seq2, seq3)) # be

    def test_lcs3_not_subset_of_lcs2_example2(self):
        seq1 = "abedcf"
        seq2 = "ibdcfe"
        seq3 = "eklm"
        """
        LCS(seq1, seq2) = bdcf
        LCS(LCS(seq1, seq2), seq3) = None
        LCS(seq1, seq2, seq3) = e
        """
        self.assertEqual(1, longest_common_subsequence(seq1, seq2, seq3)) # e

    def test_multiple_candidates_of_lcs(self):
        s1 = "abcd"
        s2 = "123"
        s3 = "456"
        s4 = "efgh"
        seq1 = s1 + s3 + s4 + s2
        seq2 = s3 + s4 + s2 + s1
        seq3 = s4 + s3 + s2 + s1
        self.assertEqual(len(s4) + len(s2), longest_common_subsequence(seq1, seq2, seq3))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 7, 2019 \[Medium\] No Adjacent Repeating Characters
--- 
> **Question:** Given a string, rearrange the string so that no character next to each other are the same. If no such arrangement is possible, then return None.

**Example:**
```py
Input: abbccc
Output: cbcbca
```

**My thougths:** This problem is basically ["LC 358 Rearrange String K Distance Apart"](https://trsong.github.io/python/java/2019/08/02/DailyQuestionsAug/#aug-24-2019-lc-358-hard-rearrange-string-k-distance-apart) with `k = 2`. The idea is to apply greedy approach, with a window of 2, choose the safest two remaining charactors until either all characters are picked, or just mulitple copy of one character left.

For example, for input string: `"aaaabc"`
1. Pick `a`, `b`. Remaining `"aaac"`. Result: `"ab"`
2. Pick `a`, `c`. Remaining `"aa"`. Result: `"abac"`
3. We can no longer proceed as we cannot pick same character as it violate adjacency requirement

Another Example, for input string: `"abbccc"`
1. Pick `c`, `b`. Remaining `"abcc"`. Result: `"ab"`
2. Pick `c`, `a`. Remaining `"bc"`. Result: `"abca"`
3. Pick `b`, `c`. Result: `"abcabc"`


**Solution with Greedy Algorithm:** [https://repl.it/@trsong/No-Adjacent-Repeating-Characters](https://repl.it/@trsong/No-Adjacent-Repeating-Characters)
```py
import unittest
from Queue import PriorityQueue

def rearrange_string(original_string):
    histogram = {}
    for c in original_string:
        histogram[c] = histogram.get(c, 0) + 1
        
    max_heap = PriorityQueue()
    for c, count in histogram.items():
        # Max heap is implemented with PriorityQueue (small element goes first), therefore use negative key to achieve max heap
        max_heap.put((-count, c))

    res = []
    while not max_heap.empty():
        neg_first_count, first_char = max_heap.get()
        remaining_first_count = -neg_first_count - 1
        res.append(first_char)

        if remaining_first_count > 0 and max_heap.empty():
            return None
        elif max_heap.empty():
            break

        neg_second_count, second_char = max_heap.get()
        remainging_second_count = -neg_second_count - 1
        res.append(second_char)

        if remaining_first_count > 0:
            max_heap.put((-remaining_first_count, first_char))
        if remainging_second_count > 0:
            max_heap.put((-remainging_second_count, second_char))

    return ''.join(res)


class RearrangeStringSpec(unittest.TestCase):
    def assert_not_adjacent(self, rearranged_string, original_string):
        # Test same length
        self.assertTrue(len(original_string) == len(rearranged_string))

        # Test containing all characters
        self.assertTrue(sorted(original_string) == sorted(rearranged_string))

        # Test not adjacent
        last_occur_map = {}
        for i, c in enumerate(rearranged_string):
            last_occur = last_occur_map.get(c, float('-inf'))
            self.assertTrue(i - last_occur >= 2)
            last_occur_map[c] = i
    
    def test_empty_string(self):
        self.assertEqual("", rearrange_string(""))

    def test_example(self):
        original_string = "abbccc"
        rearranged_string = rearrange_string(original_string)
        self.assert_not_adjacent(rearranged_string, original_string)

    def test_original_string_contains_duplicated_characters(self):
        original_string = "aaabb"
        rearranged_string = rearrange_string(original_string)
        self.assert_not_adjacent(rearranged_string, original_string)

    def test_original_string_contains_duplicated_characters2(self):
        original_string = "aaabc"
        rearranged_string = rearrange_string(original_string)
        self.assert_not_adjacent(rearranged_string, original_string)

    def test_impossible_to_rearrange_string(self):
        original_string = "aa"
        self.assertIsNone(rearrange_string(original_string))

    def test_impossible_to_rearrange_string2(self):
        original_string = "aaaabc"
        self.assertIsNone(rearrange_string(original_string))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 6, 2019 \[Easy\] Zombie in Matrix
---
> **Question:** Given a 2D grid, each cell is either a zombie 1 or a human 0. Zombies can turn adjacent (up/down/left/right) human beings into zombies every hour. Find out how many hours does it take to infect all humans?

**Example:**
```py
Input:
[[0, 1, 1, 0, 1],
 [0, 1, 0, 1, 0],
 [0, 0, 0, 0, 1],
 [0, 1, 0, 0, 0]]

Output: 2

Explanation:
At the end of the 1st hour, the status of the grid:
[[1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1],
 [0, 1, 0, 1, 1],
 [1, 1, 1, 0, 1]]

At the end of the 2nd hour, the status of the grid:
[[1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1]]
 ```

**My thoughts:** The problem can be solved with BFS. Just imagining initially all zombines are virtually associated with a single root where after we perform a BFS, the result will be depth of the BFS search tree.

**Solution with BFS:** [https://repl.it/@trsong/Zombie-in-Matrix](https://repl.it/@trsong/Zombie-in-Matrix) 
```py
import unittest
from Queue import Queue

def zombie_infection_time(grid):
    if not grid or not grid[0]:
        return -1
    n, m = len(grid), len(grid[0])
    
    queue = Queue()
    for r in xrange(n):
        for c in xrange(m):
            if grid[r][c] == 1:
                queue.put((r, c))

    time = -1
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    while not queue.empty():
        for _ in xrange(queue.qsize()):
            r, c = queue.get()
            if time > 0 and grid[r][c] == 1:
                continue
                
            grid[r][c] = 1
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < m and grid[nr][nc] == 0:
                    queue.put((nr, nc))
        time += 1
    return time


class ZombieInfectionTimeSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(2, zombie_infection_time([
            [0, 1, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0]]))

    def test_empty_grid(self):
        # Assume grid with no zombine returns -1
        self.assertEqual(-1, zombie_infection_time([]))
    
    def test_grid_without_zombie(self):
        # Assume grid with no zombine returns -1
        self.assertEqual(-1, zombie_infection_time([
            [0, 0, 0],
            [0, 0, 0]
        ]))

    def test_1x1_grid(self):
        self.assertEqual(0, zombie_infection_time([[1]]))

    def test_when_all_human_are_infected(self):
        self.assertEqual(0, zombie_infection_time([
            [1, 1],
            [1, 1]]))
    
    def test_grid_with_one_zombie(self):
        self.assertEqual(4, zombie_infection_time([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]))
    
    def test_grid_with_one_zombie2(self):
        self.assertEqual(2, zombie_infection_time([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]))
    
    def test_grid_with_multiple_zombies(self):
        self.assertEqual(4, zombie_infection_time([
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0]]))
    
    def test_grid_with_multiple_zombies2(self):
        self.assertEqual(4, zombie_infection_time([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 5, 2019 \[Medium\] Normalize Pathname
---
> **Question:** Given an absolute pathname that may have `"."` or `".."` as part of it, return the shortest standardized path.
>
> For example, given `"/usr/bin/../bin/./scripts/../"`, return `"/usr/bin"`.

**Solution with Stack:** [https://repl.it/@trsong/Normalize-Pathname](https://repl.it/@trsong/Normalize-Pathname)
```py
import unittest

def normalize_pathname(path):
    stack = []
    pathname = []
    prev = None
    for c in path:
        if c == '/' and prev != c and pathname:
            stack.append(''.join(pathname))
            pathname = []
        elif c == '.' and prev == c and stack:
            stack.pop()
        elif c.isalnum():
            if prev == '.':
                pathname.append('.')
            pathname.append(c)
        prev = c
    
    if pathname:
        stack.append(''.join(pathname))

    return '/' + '/'.join(stack)


class NormalizePathnameSpec(unittest.TestCase):
    def test_example(self):
        original_path = 'usr/bin/../bin/./scripts/../'
        normalized_path = '/usr/bin'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_empty_path(self):
        original_path = ''
        normalized_path = '/'
        self.assertEqual(normalized_path, normalize_pathname(original_path))
    
    def test_file_in_root_directory(self):
        original_path = 'bin'
        normalized_path = '/bin'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_parent_of_root_dirctory(self):
        original_path = '/a/b/..'
        normalized_path = '/a'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_parent_of_root_dirctory2(self):
        original_path = '../../'
        normalized_path = '/'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_parent_of_root_dirctory3(self):
        original_path = '../../../../a/'
        normalized_path = '/a'
        self.assertEqual(normalized_path, normalize_pathname(original_path))
    
    def test_current_directory(self):
        original_path = '.'
        normalized_path = '/'
        self.assertEqual(normalized_path, normalize_pathname(original_path))
    
    def test_current_directory2(self):
        original_path = './a/./b/c/././d/e/.'
        normalized_path = '/a/b/c/d/e'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_parent_of_current_directory(self):
        original_path = 'a/b/c/././.././../'
        normalized_path = '/a'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_hidden_file(self):
        original_path = './home/.bashrc'
        normalized_path = '/home/.bashrc'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_file_with_extention(self):
        original_path = 'home/autorun.inf'
        normalized_path = '/home/autorun.inf'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_directory_with_dots(self):
        original_path = 'home/work/com.myPythonProj.ui/src/../.git/'
        normalized_path = '/home/work/com.myPythonProj.ui/.git'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_parent_of_directory_with_dots(self):
        original_path = 'home/work/com.myPythonProj.db.server.test/./.././com.myPythonProj.db.server/./src/./.git/../.git/.gitignore'
        normalized_path = '/home/work/com.myPythonProj.db.server/src/.git/.gitignore'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    """
    A unix file system support consecutive slashes. 
    Consecutive slashes is equivalent to single slash. 
    eg. 'a//////b////c//' is equivalent to '/a/b/c'
    """
    def test_consecutive_slashes(self):
        original_path = 'a/.//////b///..//c//'
        normalized_path = '/a/c'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_consecutive_slashes2(self):
        original_path = '///////'
        normalized_path = '/'
        self.assertEqual(normalized_path, normalize_pathname(original_path))

    def test_consecutive_slashes3(self):
        original_path = '/../..////..//././//.//../a/////'
        normalized_path = '/a'
        self.assertEqual(normalized_path, normalize_pathname(original_path))
        
    def test_consecutive_slashes4(self):
        original_path = '//////.a//////'
        normalized_path = '/.a'
        self.assertEqual(normalized_path, normalize_pathname(original_path))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 4, 2019 \[Easy\] Make the Largest Number
---
> **Question:** Given a number of integers, combine them so it would create the largest number.

**Example:**
```py
Input:  [17, 7, 2, 45, 72]
Output:  77245217
```

**Solution with Customized Sort:** [https://repl.it/@trsong/Make-the-Largest-Number](https://repl.it/@trsong/Make-the-Largest-Number)
```py
import unittest

def construct_largest_number(nums):
    if not nums:
        return 0

    def string_cmp(s1, s2):
        s12 = s1 + s2
        s21 = s2 + s1
        if s12 < s21:
            return -1
        elif s12 == s21:
            return 0
        else:
            return 1
    
    negative_nums = filter(lambda x: x < 0, nums)
    filtered_nums = filter(lambda x: x >= 0, nums)
    str_nums = map(str, filtered_nums)

    if negative_nums:
        str_nums.sort(string_cmp)
        return int(str(negative_nums[0]) + "".join(str_nums))
    else:
        str_nums.sort(string_cmp, reverse=True)
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

### Nov 3, 2019 \[Easy\] Find Unique Element among Array of Duplicates
---
> **Question:** Given an array of integers, arr, where all numbers occur twice except one number which occurs once, find the number. Your solution should ideally be O(n) time and use constant extra space.
> 
**Example:**
```py
Input: arr = [7, 3, 5, 5, 4, 3, 4, 8, 8]
Output: 7
```

**My thoughts:** XOR has many pretty useful properties:

* 0 ^ x = x
* x ^ x = 0
* x ^ y = y ^ x 

For example: 
```py
7 ^ 3 ^ 5 ^ 5 ^ 4 ^ 3 ^ 4 ^ 8 ^ 8
= 7 ^ 3 ^ (5 ^ 5) ^ 4 ^ 3 ^ 4 ^ (8 ^ 8)
= 7 ^ 3 ^ 4 ^ 3 ^ 4 
= 7 ^ 3 ^ 3 ^ 4 ^ 4
= 7 ^ (3 ^ 3) ^ (4 ^ 4)
= 7
```

**Solution with XOR:** [https://repl.it/@trsong/Find-Unique-Element-among-Array-of-Duplicates](https://repl.it/@trsong/Find-Unique-Element-among-Array-of-Duplicates)
```py
import unittest

def find_unique_element(nums):
    res = 0
    for num in nums:
        res ^= num
    return res


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


### Nov 2, 2019 \[Easy\] String Compression
---
> **Question:** Given an array of characters with repeats, compress it in place. The length after compression should be less than or equal to the original array.

**Example:**
```py
Input: ['a', 'a', 'b', 'c', 'c', 'c']
Output: ['a', '2', 'b', 'c', '3']
```

**Solution with Two-Pointers:** [https://repl.it/@trsong/String-Compression](https://repl.it/@trsong/String-Compression)
```py
import unittest

def string_compression(msg):
    if not msg:
        return

    n = len(msg)
    count = 1
    j = 0
    for i in xrange(1, len(msg)+1):
        if i < n and msg[i] == msg[i-1]:
            count += 1
        else:
            msg[j] = msg[i-1]
            j += 1
            if count > 1:    
                for char in str(count):
                    msg[j] = char
                    j += 1
            count = 1
    
    for _ in xrange(j, n):
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

### Nov 1, 2019 \[Hard\] Partition Array to Reach Mininum Difference
---
> **Question:** Given an array of positive integers, divide the array into two subsets such that the difference between the sum of the subsets is as small as possible.
>
> For example, given `[5, 10, 15, 20, 25]`, return the sets `[10, 25]` and `[5, 15, 20]`, which has a difference of `5`, which is the smallest possible difference.


**Solution with DP:** [https://repl.it/@trsong/Partition-Array-to-Reach-Mininum-Difference](https://repl.it/@trsong/Partition-Array-to-Reach-Mininum-Difference)
```py
import unittest

def min_partition_difference(nums):
    if not nums:
        return 0

    sum_nums = sum(nums)
    n = len(nums)

    # Let dp[i][s] represents whether subset of nums[0:i] can sum up to s
    # dp[i][s] = dp[i-1][s]               if exclude current element
    # dp[i][s] = dp[i-1][s - nums[i-1]]   if incldue current element
    dp = [[False for _ in xrange(sum_nums + 1)] for _ in xrange(n+1)]

    for i in xrange(n+1):
        # Any subsets can sum up to 0
        dp[i][0] = True

    for i in xrange(1, n+1):
        for s in xrange(1, sum_nums+1):
            if dp[i-1][s]:
                # Exclude current element, as we can already reach sum s
                dp[i][s] = True
            elif nums[i-1] <= s:
                # As all number are positive, include current elem cannot exceed current sum
                dp[i][s] = dp[i-1][s - nums[i-1]]

    """
     Let's do some math here:
     Let s1, s2 be the size to two subsets after partition and assume s1 >= s2
     We can have s1 + s2 = sum_nums and we want to get min{s1 - s2} where s1 >= s2:

     min{s1 - s2}
     = min{s1 + s2 - s2 - s2}
     = min{sum_nums - 2 * s2}  in this step sum_nums - 2 * s2 >=0, gives s2 <= sum_nums/ 2
     = sum_nums - 2 * max{s2}  where s2 <= sum_nums/2
    """
    s2 = 0
    for s in xrange(sum_nums/2, 0, -1):
        if dp[n][s]:
            s2 = s
            break

    return sum_nums - 2 * s2


class MinPartitionDifferenceSpec(unittest.TestCase):
    def test_example(self):
        # Partition: [10, 25] and [5, 15, 20]
        self.assertEqual(5, min_partition_difference([5, 10, 15, 20, 25]))

    def test_empty_array(self):
        self.assertEqual(0, min_partition_difference([]))

    def test_array_with_one_element(self):
        self.assertEqual(42, min_partition_difference([42]))

    def test_array_with_two_elements(self):
        self.assertEqual(0, min_partition_difference([42, 42]))

    def test_unsorted_array_with_duplicated_numbers(self):
        # Partition: [3, 4] and [1, 2, 2, 1]
        self.assertEqual(1, min_partition_difference([3, 1, 4, 2, 2, 1]))

    def test_unsorted_array_with_unique_numbers(self):
        # Partition: [11] and [1, 5, 6]
        self.assertEqual(1, min_partition_difference([1, 6, 11, 5]))

    def test_sorted_array_with_duplicated_numbers(self):
        # Partition: [1, 2, 2] and [4]
        self.assertEqual(1, min_partition_difference([1, 2, 2, 4]))

    def test_min_partition_difference_is_zero(self):
        # Partition: [1, 8, 2, 7] and [3, 6, 4, 5]
        self.assertEqual(0, min_partition_difference([1, 2, 3, 4, 5, 6, 7, 8]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

