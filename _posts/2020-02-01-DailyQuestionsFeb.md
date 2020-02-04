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