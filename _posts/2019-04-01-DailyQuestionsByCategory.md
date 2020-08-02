---
layout: post
title:  "Daily Coding Problems by Category"
date:   2019-04-01 22:22:32 -0700
categories: Python/Java
---
* This will become a table of contents (this text will be scraped).
{:toc}

<!--
<details>
<summary class="lc_h">

- [****]() -- ** [*\(Try ME\)*]()

</summary>
<div>


</div>
</details>
-->

{::options parse_block_html="true" /}


## Math
---

### Puzzle
---

### XOR
---

### Hashing
---

## Array
---

### Basic
---

### Two Pointers
---

### Sliding Window
---

<details>
<summary class="lc_e">

- [**\[Easy\] LC 383. Ransom Note**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-10-2020-lc-383-easy-ransom-note) -- *Given a magazine of letters and the note he wants to write, determine whether he can construct the word.* [*\(Try ME\)*](https://repl.it/@trsong/Ransom-Note-1)

</summary>
<div>

**Question:** A criminal is constructing a ransom note. In order to disguise his handwriting, he is cutting out letters from a magazine.

Given a magazine of letters and the note he wants to write, determine whether he can construct the word.

**Example 1:**
```py
Input: ['a', 'b', 'c', 'd', 'e', 'f'], 'bed'
Output: True
```

**Example 2:**
```py
Input: ['a', 'b', 'c', 'd', 'e', 'f'], 'cat'
Output: False
```

</div>
</details>

 
## List
---

### Singly Linked List
---

<details>
<summary class="lc_e">

- [**\[Easy\] Intersection of Linked Lists**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-19-2020-easy-intersection-of-linked-lists) -- *Given two singly linked lists. The lists intersect at some node. Find, and return the node. Note: the lists are non-cyclical.* [*\(Try ME\)*](https://repl.it/@trsong/Intersection-of-Linked-Lists-1)

</summary>
<div>

**Question:** You are given two singly linked lists. The lists intersect at some node. Find, and return the node. Note: the lists are non-cyclical.

**Example:**
```py
A = 1 -> 2 -> 3 -> 4
B = 6 -> 3 -> 4
# This should return 3 (you may assume that any nodes with the same value are the same node)
```

</div>
</details>

### Doubly Linked List
---

### Circular Linked List
---

### Fast-slow Pointers
---

## Sort
---

### Merge Sort
---

### Quick Select & Quick Sort
---

<details>
<summary class="lc_m">

- [**\[Medium\] Sorting a List With 3 Unique Numbers**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-18-2020-medium-sorting-a-list-with-3-unique-numbers) -- *Given a list of numbers with only 3 unique numbers (1, 2, 3), sort the list in O(n) time.* [*\(Try ME\)*](https://repl.it/@trsong/Sorting-a-List-With-3-Unique-Numbers-1)

</summary>
<div>

**Question:** Given a list of numbers with only `3` unique numbers `(1, 2, 3)`, sort the list in `O(n)` time.

**Example:**
```py
Input: [3, 3, 2, 1, 3, 2, 1]
Output: [1, 1, 2, 2, 3, 3, 3]
```

</div>
</details>

### Counting Sort
---


## Tree
---

### BST
---

<details>
<summary class="lc_e">

- [**\[Easy\] Generate Binary Search Trees**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-11-2020-easy-generate-binary-search-trees) -- *Given an integer N, construct all possible binary search trees with N nodes.* [*\(Try ME\)*](https://repl.it/@trsong/Generate-Binary-Search-Trees-with-N-Nodes-1)

</summary>
<div>

**Question:** Given an integer N, construct all possible binary search trees with N nodes.

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Second Largest in BST**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-24-2020-medium-second-largest-in-bst) -- *Given the root to a binary search tree, find the second largest node in the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Second-Largest-in-BST-1)

</summary>
<div>

**Question:** Given the root to a binary search tree, find the second largest node in the tree.

</div>
</details>

### Recursion
---

<details>
<summary class="lc_e">

- [**\[Easy\] LC 938. Range Sum of BST**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-22-2020-lc-938-easy-range-sum-of-bst) -- *Given a binary search tree and a range [a, b] (inclusive), return the sum of the elements of the binary search tree within the range.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Range-Sum-of-BST-1)

</summary>
<div>

**Question:** Given a binary search tree and a range `[a, b]` (inclusive), return the sum of the elements of the binary search tree within the range.

**Example:**
```py
Given the range [4, 9] and the following tree:

    5
   / \
  3   8
 / \ / \
2  4 6  10

return 23 (5 + 4 + 6 + 8).
```

</div>
</details>




### Traversal
---

<details>
<summary class="lc_e">

- [**\[Easy\] LC 872. Leaf-Similar Trees**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-8-2020-lc-872-easy-leaf-similar-trees) -- *Given two trees, whether they are "leaf similar". Two trees are considered "leaf-similar" if their leaf orderings are the same.* [*\(Try ME\)*](https://repl.it/@trsong/Leaf-Similar-Trees-1)

</summary>
<div>

**Question:** Given two trees, whether they are `"leaf similar"`. Two trees are considered `"leaf-similar"` if their leaf orderings are the same. 

For instance, the following two trees are considered leaf-similar because their leaves are `[2, 1]`:

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

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] In-order & Post-order Binary Tree Traversal**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-29-2020-medium-in-order--post-order-binary-tree-traversal) -- *Given Postorder and Inorder traversals, construct the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Test-Build-Binary-Tree-with-In-order-and-Post-order-Traversal)

</summary>
<div>

**Question:** Given Postorder and Inorder traversals, construct the tree.

**Examples 1:**
```py
Input: 
in_order = [2, 1, 3]
post_order = [2, 3, 1]

Output: 
      1
    /   \
   2     3 
```

**Example 2:**
```py
Input: 
in_order = [4, 8, 2, 5, 1, 6, 3, 7]
post_order = [8, 4, 5, 2, 6, 7, 3, 1]

Output:
          1
       /     \
     2        3
   /    \   /   \
  4     5   6    7
    \
      8
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Pre-order & In-order Binary Tree Traversal**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-28-2020-medium-pre-order--in-order-binary-tree-traversal) -- *Given pre-order and in-order traversals of a binary tree, write a function to reconstruct the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Test-Pre-order-and-In-order-Binary-Tree-Traversal)

</summary>
<div>

**Question:** Given pre-order and in-order traversals of a binary tree, write a function to reconstruct the tree.

For example, given the following preorder traversal:

```py
[a, b, d, e, c, f, g]
```

And the following inorder traversal:

```py
[d, b, e, a, f, c, g]
```

You should return the following tree:

```py
    a
   / \
  b   c
 / \ / \
d  e f  g
```

</div>
</details>

## Trie
---


<details>
<summary class="lc_m">

- [**\[Medium\] Autocompletion**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-16-2020-medium-autocompletion) -- *Given a large set of words and then a single word prefix, find all words that it can complete to.* [*\(Try ME\)*](https://repl.it/@trsong/Autocompletion-1)

</summary>
<div>

**Question:**  Implement auto-completion. Given a large set of words (for instance 1,000,000 words) and then a single word prefix, find all words that it can complete to.

**Example:**
```py
class Solution:
  def build(self, words):
    # Fill this in.
    
  def autocomplete(self, word):
    # Fill this in.

s = Solution()
s.build(['dog', 'dark', 'cat', 'door', 'dodge'])
s.autocomplete('do')  # Return ['dog', 'door', 'dodge']
```

</div>
</details>

## Stack
---


<details>
<summary class="lc_e">

- [**\[Easy\] LC 1047. Remove Adjacent Duplicate Characters**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-15-2020-lc-1047-easy-remove-adjacent-duplicate-characters) -- *Given a string, we want to remove 2 adjacent characters that are the same, and repeat the process with the new string until we can no longer* [*\(Try ME\)*](https://repl.it/@trsong/Remove-2-Adjacent-Duplicate-Characters-1)

</summary>
<div>

**Question:** Given a string, we want to remove 2 adjacent characters that are the same, and repeat the process with the new string until we can no longer perform the operation.

**Example:**
```py
remove_adjacent_dup("cabba")
# Start with cabba
# After remove bb: caa
# After remove aa: c
# Returns c
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Index of Larger Next Number**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-17-2020-medium-index-of-larger-next-number) -- *Given a list of numbers, for each element find the next element that is larger than the current element.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Index-of-Larger-Next-Number-1)

</summary>
<div>

**Question:** Given a list of numbers, for each element find the next element that is larger than the current element. Return the answer as a list of indices. If there are no elements larger than the current element, then use -1 instead.

**Example:** 
```py
larger_number([3, 2, 5, 6, 9, 8])
# return [2, 2, 3, 4, -1, -1]
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] The Celebrity Problem**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-27-2020-medium-the-celebrity-problem) -- *Given a list of N people and the above operation, find a way to identify the celebrity in O(N) time.* [*\(Try ME\)*](https://repl.it/@trsong/Test-The-Celebrity-Problem)

</summary>
<div>

**Question:** At a party, there is a single person who everyone knows, but who does not know anyone in return (the "celebrity"). To help figure out who this is, you have access to an `O(1)` method called `knows(a, b)`, which returns `True` if person `a` knows person `b`, else `False`.

Given a list of `N` people and the above operation, find a way to identify the celebrity in `O(N)` time.
 
</div>
</details>


## Priority Queue
---

### Scheduling
---


## Hashmap
---

### Basic
---

### Advance
---


## Set
---

### Basic
---

### Advance
---


## Greedy
---


## Divide and Conquer
---


## Graph
---

### BFS
---

<details>
<summary class="lc_m">

- [**\[Medium\] Find All Cousins in Binary Tree**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-21-2020-medium-find-all-cousins-in-binary-tree) -- *Given a binary tree and a particular node, find all cousins of that node.* [*\(Try ME\)*](https://repl.it/@trsong/All-Cousins-in-Binary-Tree-1)

</summary>
<div>

**Question:** Two nodes in a binary tree can be called cousins if they are on the same level of the tree but have different parents. 

Given a binary tree and a particular node, find all cousins of that node.


**Example:**
```py
In the following diagram 4 and 6 are cousins:

    1
   / \
  2   3
 / \   \
4   5   6
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Longest Path in Binary Tree**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-31-2020-hard-longest-path-in-binary-tree) -- *Given a binary tree, return any of the longest path.* [*\(Try ME\)*](https://repl.it/@trsong/Test-Longest-Path-in-Binary-Tree)

</summary>
<div>

**Question:** Given a binary tree, return any of the longest path.

**Example 1:**
```py
Input:      1
          /   \
        2      3
      /  \
    4     5

Output: [4, 2, 1, 3] or [5, 2, 1, 3]  
```

**Example 2:**
```py
Input:      1
          /   \
        2      3
      /  \      \
    4     5      6

Output: [4, 2, 1, 3, 6] or [5, 2, 1, 3, 6] 
```

</div>
</details>

### Level-based BFS
---

### DFS
---

### Topological Sort
---

### Union Find
---

### A-Star
---


<details>
<summary class="lc_h">

- [**\[Hard\] Minimum Step to Reach One**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-26-2020-easy-minimum-step-to-reach-one) -- *Given a positive integer N, find the smallest number of steps it will take to reach 1.* [*\(Try ME\)*](https://repl.it/@trsong/Test-Find-Minimum-Step-to-Reach-One)

</summary>
<div>

**Question:** Given a positive integer `N`, find the smallest number of steps it will take to reach `1`.

There are two kinds of permitted steps:
- You may decrement `N` to `N - 1`.
- If `a * b = N`, you may decrement `N` to the larger of `a` and `b`.
 
For example, given `100`, you can reach `1` in five steps with the following route: `100 -> 10 -> 9 -> 3 -> 2 -> 1`.

</div>
</details>


## Backtracking
---


<details>
<summary class="lc_m">

- [**\[Medium\] LC 93. All Possible Valid IP Address Combinations**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-23-2020-lc-93-medium-all-possible-valid-ip-address-combinations) -- *Given a string of digits, generate all possible valid IP address combinations.* [*\(Try ME\)*](https://repl.it/@trsong/Find-All-Possible-Valid-IP-Address-Combinations-1)

</summary>
<div>

**Question:** Given a string of digits, generate all possible valid IP address combinations.

IP addresses must follow the format A.B.C.D, where A, B, C, and D are numbers between `0` and `255`. Zero-prefixed numbers, such as `01` and `065`, are not allowed, except for `0` itself.

For example, given `"2542540123"`, you should return `['254.25.40.123', '254.254.0.123']`
 
</div>
</details>


## Dynamic Programming
---

### 1D DP
---

### 2D DP
---

<details>
<summary class="lc_m">

- [**\[Medium\] LC 718. Longest Common Sequence of Browsing Histories**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-9-2020-lc-718-medium-longest-common-sequence-of-browsing-histories) -- *Write a function that takes two users’ browsing histories as input and returns the longest contiguous sequence of URLs that appear in both.* [*\(Try ME\)*](https://repl.it/@trsong/Longest-Common-Sequence-of-Browsing-Histories-1)

</summary>
<div>

**Question:** We have some historical clickstream data gathered from our site anonymously using cookies. The histories contain URLs that users have visited in chronological order.

Write a function that takes two users' browsing histories as input and returns the longest contiguous sequence of URLs that appear in both.

For example, given the following two users' histories:

```py
user1 = ['/home', '/register', '/login', '/user', '/one', '/two']
user2 = ['/home', '/red', '/login', '/user', '/one', '/pink']
```

You should return the following:

```py
['/login', '/user', '/one']
```

</div>
</details>


### DP+ 
---


## String
---

<details>
<summary class="lc_m">

- [**\[Medium\] Tokenization**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-20-2020-medium-tokenization) -- *Given a dictionary of words and a string made up of those words (no spaces), return the original sentence in a list.* [*\(Try ME\)*](https://repl.it/@trsong/String-Tokenization-1)

</summary>
<div>

**Questions:** Given a dictionary of words and a string made up of those words (no spaces), return the original sentence in a list. If there is more than one possible reconstruction, return any of them. If there is no possible reconstruction, then return null.

**Example 1:**
```py
Input: ['quick', 'brown', 'the', 'fox'], 'thequickbrownfox'
Output: ['the', 'quick', 'brown', 'fox']
```

**Example 2:**
```py
Input: ['bed', 'bath', 'bedbath', 'and', 'beyond'], 'bedbathandbeyond'
Output:  Either ['bed', 'bath', 'and', 'beyond'] or ['bedbath', 'and', 'beyond']
```

</div>
</details>

### Anagram
---

### Palindrome
---


{::options parse_block_html="false" /}
