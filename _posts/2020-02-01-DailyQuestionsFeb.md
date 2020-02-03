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


### Feb 3, 2020 \[Easy\] Design a Hit Counter
---
> **Question:**  Design and implement a HitCounter class that keeps track of requests (or hits). It should support the following operations:
>
> - `record(timestamp)`: records a hit that happened at timestamp
> - `total()`: returns the total number of hits recorded
> - `range(lower, upper)`: returns the number of hits that occurred between timestamps lower and upper (inclusive)
>
> **Follow-up:** What if our system has limited memory?


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