---
layout: post
title:  "Daily Coding Problems"
date:   2019-04-30 22:22:32 -0700
categories: Python/Java
---
* This will become a table of contents (this text will be scraped).
{:toc}

## Daily Coding Problems

### Enviroment Setup
---

**Python 2.7:** [https://repl.it/languages/python](https://repl.it/languages/python)

**Java 1.8:**  TBD

<!--
### May 12, 2019 \[Medium\] Craft Palindrome
---
> **Question:** Given a string, find the palindrome that can be made by inserting the fewest number of characters as possible anywhere in the word. If there is more than one palindrome of minimum length that can be made, return the lexicographically earliest one (the first one alphabetically).
>
> For example, given the string "race", you should return "ecarace", since we can add three letters to it (which is the smallest amount to make a palindrome). There are seven other palindromes that can be made from "race" by adding three letters, but "ecarace" comes first alphabetically.
>
> As another example, given the string "google", you should return "elgoogle".

### May 11, 2019 LC 42 \[Hard\] Trapping Rain Water
---
> Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
 
![rainwatertrap](https://assets.leetcode.com/uploads/2018/10/22/rainwatertrap.png)

> The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
> 
> Example:
> 
> Input: [0,1,0,2,1,0,1,3,2,1,2,1]
> 
> Output: 6

### May 10, 2019 \[Hard\] Execlusive Product
---
> **Question:**  Given an array of integers, return a new array such that each element at index i of the new array is the product of all the numbers in the original array except the one at i.
>
> For example, if our input was [1, 2, 3, 4, 5], the expected output would be [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output would be [2, 3, 6].
>
> Follow-up: what if you can't use division?

### May 9, 2019 \[Easy\] Grid Path
---
> **Question:**  You are given an M by N matrix consisting of booleans that represents a board. Each True boolean represents a wall. Each False boolean represents a tile you can walk on.
>
> Given this matrix, a start coordinate, and an end coordinate, return the minimum number of steps required to reach the end coordinate from the start. If there is no possible path, then return null. You can move up, left, down, and right. You cannot move through walls. You cannot wrap around the edges of the board.
>
> For example, given the following board:

```
[
  [F, F, F, F],
  [T, T, F, T],
  [F, F, F, F],
  [F, F, F, F]
]
```
> and start = (3, 0) (bottom left) and end = (0, 0) (top left), the minimum number of steps required to reach the end is 7, since we would need to go through (1, 2) because there is a wall everywhere else on the second row.


### May 8, 2019 LC 130 \[Medium\] Surrounded Regions
---

> **Question:**  Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.
> A region is captured by flipping all 'O's into 'X's in that surrounded region.
> 
> Example:

```
X X X X
X O O X
X X O X
X O X X
```
> After running your function, the board should be:

```
X X X X
X X X X
X X X X
X O X X
```
> Explanation:
>
> Surrounded regions shouldn’t be on the border, which means that any 'O' on the border of the board are not flipped to 'X'. Any 'O' that is not on the border and it is not connected to an 'O' on the border will be flipped to 'X'. Two cells are connected if they are adjacent cells connected horizontally or vertically.
-->

### May 7, 2019 \[Hard\] Largest Sum of Non-adjacent Numbers
---

> **Question:** Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0 or negative.
>
> For example, `[2, 4, 6, 2, 5]` should return 13, since we pick 2, 6, and 5. `[5, 1, 1, 5]` should return 10, since we pick 5 and 5.
>
> Follow-up: Can you do this in O(N) time and constant space?


### May 6, 2019 \[Hard\] Climb Staircase (Continued)
---

> **Question:** There exists a staircase with N steps, and you can climb up either 1 or 2 steps at a time. Given N, write a function that **PRINT** out all possible unique ways you can climb the staircase. The **ORDER** of the steps matters. 
>
> For example, if N is 4, then there are 5 unique ways (accoding to May 5's question). This time we print them out as the following:

```py
1, 1, 1, 1
2, 1, 1
1, 2, 1
1, 1, 2
2, 2
```

> What if, instead of being able to climb 1 or 2 steps at a time, you could climb any number from a set of positive integers X? 
>
> For example, if N is 6, and X = {2, 5}. You could climb 2 or 5 steps at a time. Then there is only 1 unique way, so we print the following:

```py
2, 2, 2
```

**My thoughts:** The only way to figure out each path is to manually test all outcomes. However, certain cases are invalid (like exceed the target value while climbing) so we try to modify certain step until its valid. Such technique is called ***Backtracking***.

We may use recursion to implement backtracking, each recursive step will create a separate branch which also represent different recursive call stacks. Once the branch is invalid, the call stack will bring us to a different branch, i.e. backtracking to a different solution space.

For example, if N is 4, and feasible steps are `[1, 2]`, then there are 5 different solution space/path. Each node represents a choice we made and each branch represents a recursive call.

Note we also keep track of the remaining steps while doing recursion. 
```
├ 1 
│ ├ 1
│ │ ├ 1
│ │ │ └ 1 SUCCEED
│ │ └ 2 SUCCEED
│ └ 2 
│   └ 1 SUCCEED
└ 2 
  ├ 1
  │ └ 1 SUCCEED
  └ 2 SUCCEED
```

If N is 6 and fesible step is `[5, 2]`:
```
├ 5 FAILURE
└ 2 
  └ 2
    └ 2 SUCCEED
```



**Python Solution:** [https://repl.it/@trsong/climbStairs2](https://repl.it/@trsong/climbStairs2)

```py
SEPARATOR = ", "

def climbStairsRecur(feasible, remain, path, res):
  global SEPARATOR
  # Once climbed all staircases, include this path
  if remain == 0:
    res.append(SEPARATOR.join(map(str, path)))
    return res
  else:
    # Test all feasible steps and if it works then proceed
    for step in feasible:
      if remain - step >= 0:
        newPath = path + [step]
        # What if we choose this path, do recursion on remaining staircases
        climbStairsRecur(feasible, remain - step, newPath, res)

def climbStairs2(n, feasibleStairs):
  res = []
  climbStairsRecur(feasibleStairs, n, [], res)
  return res

def testResult(res, expected):
  return sorted(res) == sorted(expected)

def main():
  assert testResult(climbStairs2(5, [1, 3, 5]), [
    "1, 1, 1, 1, 1",
    "3, 1, 1",
    "1, 3, 1",
    "1, 1, 3",
    "5"])
  assert testResult(climbStairs2(4, [1, 2]), [
    "1, 1, 1, 1", 
    "2, 1, 1",
    "1, 2, 1",
    "1, 1, 2",
    "2, 2"
  ])
  assert testResult(climbStairs2(9, []), [])
  assert testResult(climbStairs2(42, [42]), ["42"])
  assert testResult(climbStairs2(4, [1, 2, 3, 4]), [
    "1, 1, 1, 1",
    "1, 1, 2",
    "1, 2, 1",
    "1, 3",
    "2, 1, 1",
    "2, 2",
    "3, 1",
    "4"
    ])
  assert testResult(climbStairs2(99, [7]), [])

if __name__ == '__main__':
  main()
```  

### May 5, 2019 \[Medium\] Climb Staircase
---

> **Question:** There exists a staircase with N steps, and you can climb up either 1 or 2 steps at a time. Given N, write a function that returns a number represents the total number of unique ways you can climb the staircase.
>
> For example, if N is 4, then there are 5 unique ways:
* 1, 1, 1, 1
* 2, 1, 1
* 1, 2, 1
* 1, 1, 2
* 2, 2

**My thoughts:** Imagine when you step on the nth staircase, what could be the previous move you made to get to the nth staircase. You only have two ways:
* Either, you were on (n-1)th staircase and you climbed 1 staircase next
* Or, you were on (n-2)th staircase and you climbed 2 staircase next
So the total number of way to climb n staircase depends on how many ways to climb n-1 staircases + how many ways to climb n-2 staircases.

**Note:** Did you notice the solution forms a ***Fibonacci Sequence***? Well same optimization tricks also work for this question.

**Python Solutions:** [https://repl.it/@trsong/climbStairs](https://repl.it/@trsong/climbStairs)
```py
def climbStairs(n):
  # Let dp[n] represent the number of ways to climb n stairs, then we will have
  # dp[n] = dp[n-1] + dp[n-2]
  # Because in order to get to n-th stair case, you can only reach from the (n-1)th or (n-2)th.
  # Which have dp[n-1] and dp[n-2] of ways separately.
  dp = [0] * max(n + 1, 3)
  dp[1] = 1 # climb 1
  dp[2] = 2 # climb 1, 1 or 2
  for i in xrange(3, n+1):
    dp[i] = dp[i-1] + dp[i-2]
  return dp[n]

def main():
  assert climbStairs(1) == 1
  assert climbStairs(2) == 2
  assert climbStairs(3) == 3
  assert climbStairs(4) == 5
  assert climbStairs(5) == 8
  assert climbStairs(6) == 13

if __name__ == '__main__':
  main()
```

### May 4, 2019 \[Easy\] Power Set
---
> **Question:** The power set of a set is the set of all its subsets. Write a function that, given a set, generates its power set.
>
> For example, given a set represented by a list `[1, 2, 3]`, it should return `[[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]` representing the power set.

**My thoughts:** There are multiple ways to solve this problem. Solution 1 & 2 use recursion that build subsets incrementally. While solution 3, only cherry pick elem based on binary representation.

**Solution 1 & 2:** Let's calculate the first few terms and try to figure out the pattern
```
powerSet([]) => [[]]
powerSet([1]) => [[], [1]]
powerSet([1, 2]) => [[], [1], [2], [1, 2]]
powerSet([1, 2, 3]) => [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]] which is powerSet([1, 2]) + append powerSet([1, 2]) with new elem 3
...
powerSet([1, 2, ..., n]) =>  powerSet([1, 2, ..., n - 1]) + append powerSet([1, 2, ..., n - 1]) with new elem n
```

**Solution 3:** the total number of subsets eqauls $2^n$. Notice that we can use binary representation to represent which elem to cherry pick. e.g. `powerSet([1, 2, 3])`.  000 represent `[]`, 010 represents `[2]` and 101 represents `[1, 3]`.

Binary representation of selection has $|\{1, 2, 3\}| = 2^3 = 8$ outcomes
```
000 => []
001 => [3]
010 => [2]
011 => [2, 3]
100 => [1]
101 => [1, 3]
110 => [1, 2]
111 => [1, 2, 3]
```

**Python Solution:** Link: [https://repl.it/@trsong/powerSet](https://repl.it/@trsong/powerSet)
```py
def powerSet(inputSet):
  if not inputSet:
    return [[]]
  else:
    subsetRes = powerSet(inputSet[1:])
    return subsetRes + map(lambda lst: [inputSet[0]] + lst, subsetRes)

def powerSet2(inputSet):
  return reduce(lambda subRes, num: subRes + [[num] + subset for subset in subRes],
  inputSet,
  [[]])

def powerSet3(inputSet):
  size = len(inputSet)
  return [[inputSet[i] for i in xrange(size) if (0x1 << i) & num] for num in xrange(2 ** size)]

def main():
  assert powerSet([]) == [[]]
  assert sorted(powerSet([1])) == [[], [1]]
  assert sorted(powerSet([1, 2])) == [[], [1], [1, 2], [2]]
  assert sorted(powerSet([1, 2, 3])) == [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]

if __name__ == '__main__':
  main()

```

### May 3, 2019 \[Easy\] Running median of a number stream
---
> **Question:** Compute the running median of a sequence of numbers. That is, given a stream of numbers, print out the median of the list so far on each new element.
> 
> Recall that the median of an even-numbered list is the average of the two middle numbers.
> 
> For example, given the sequence `[2, 1, 5, 7, 2, 0, 5]`, your algorithm should print out:

```py
2
1.5
2
3.5
2
2
2
```

**My thoughts:** Given a sorted list, the median of a list is either the element in the middle of the list or average of left-max and right-min if we break the original list into left and right part:

```
* 1, [2], 3
* 1, [2], [3], 4
```

Notice that, as we get elem from stream 1-by-1, we don't need to keep the list sorted. If only we could partition the list into two equally large list and mantain the max of the left part and min of right part. We should be good to go.

i,e,

```
[5, 1, 7]            [8 , 20, 10]
       ^ left-max     ^ right-min
```

A max-heap plus a min-heap will make it a lot easier to mantain the value we are looking for: A max-heap on the left mantaining the largest on left half and a min-heap on the right holding the smallest on right half. And most importantly, we mantain the size of max-heap and min-heap while reading data. (Left and right can at most off by 1).

**Python Solution:** Link: [https://repl.it/@trsong/runningMedianOfStream](https://repl.it/@trsong/runningMedianOfStream)

```py
from queue import PriorityQueue

# Python doesn't have Max Heap, so we just implement one ourselves
class MaxPriorityQueue(PriorityQueue):
  def __init__(self):
    PriorityQueue.__init__(self)

  def get(self):
    return -PriorityQueue.get(self)
  
  def put(self, val):
    PriorityQueue.put(self, -val)

def head(queue):
  if isinstance(queue, MaxPriorityQueue):
    return -queue.queue[0]
  else:
    return queue.queue[0]

def runningMedianOfStream(stream):
  leftMaxHeap = MaxPriorityQueue()
  rightMinHeap = PriorityQueue()
  result = []

  for elem in stream:
    # Add elem to the left, when given a smaller-than-left-max number
    if leftMaxHeap.empty() or elem < head(leftMaxHeap):
      leftMaxHeap.put(elem)
    # Add elem to the right, when given a larger-than-right-min number   
    elif rightMinHeap.empty() or elem > head(rightMinHeap):
      rightMinHeap.put(elem)
    # Add elem to the left, when given a between-left-max-and-right-min number,  
    else:
      leftMaxHeap.put(elem)

    # Re-balance both heaps by transfer elem from larger heap to smaller heap
    if len(leftMaxHeap.queue) > len(rightMinHeap.queue) + 1:
      rightMinHeap.put(leftMaxHeap.get())
    elif len(leftMaxHeap.queue) < len(rightMinHeap.queue) - 1:
      leftMaxHeap.put(rightMinHeap.get())

    # Calcualte the median base on different situations
    leftMaxHeapSize = len(leftMaxHeap.queue) 
    rightMinHeapSize = len(rightMinHeap.queue)
    medium = 0
    # Median eqauls (left-max + right-min) / 2  
    # Be careful for value overflow!
    if leftMaxHeapSize == rightMinHeapSize:
      midLeft = head(leftMaxHeap)
      midRight = head(rightMinHeap)
      diff = midRight - midLeft
      if diff % 2 == 0:
        medium = midLeft + diff / 2
      else:
        medium = midLeft + diff / 2.0
    # Median is left-max   
    elif leftMaxHeapSize > rightMinHeapSize:
      medium = head(leftMaxHeap)
    # Median is right-min
    else:
      medium = head(rightMinHeap)
    result.append(medium)
  return result

def main():
  assert runningMedianOfStream([2, 1, 5, 7, 2, 0, 5]) == [2, 1.5, 2, 3.5, 2, 2, 2]
  assert runningMedianOfStream([1, 1, 1, 1, 1, 1]) == [1, 1, 1, 1, 1, 1]
  assert runningMedianOfStream([0, 1, 1, 0, 0]) == [0, 0.5, 1, 0.5, 0]
  assert runningMedianOfStream([2, 0, 1]) == [2, 1, 1]
  assert runningMedianOfStream([3, 0, 1, 2]) == [3, 1.5, 1, 1.5]

if __name__ == '__main__':
  main()
```

### May 2, 2019 \[Medium\] Remove K-th Last Element from Singly Linked-list
---
> **Question:** Given a singly linked list and an integer k, remove the kth last element from the list. k is guaranteed to be smaller than the length of the list.
> 
> Note:
> * The list is very long, so making more than one pass is prohibitively expensive.
> * Do this in constant space and in one pass.

**My thoughts:** Use two pointers, faster one are k position ahead of slower one. Once faster reaches last elem, the 1st from the right. Then slow one should be (k + 1) th to the right. i.e. k + 1 - 1 = k. (fast is k position ahead of slow). 

```
k + 1 (from the right) -> k (from the right) -> k - 1 (from the right) -> .... -> 1 (from the right)
^^^^^ slow is here                                                                ^ Fast is here
```

So in order to remove the k-th last element from the list, we just need to point `slow.next` to the next of that node: `slow.next = slow.next.next`


**Python Solution:** Link: [https://repl.it/@trsong/removeKthFromEnd](https://repl.it/@trsong/removeKthFromEnd)
```py

class ListNode(object):
    def __init__(self, x, next=None):
        self.val = x
        self.next = next
    def __eq__(self, other):
      return not self and not other or self and other and self.next == other.next

def List(*vals):
  dummy = ListNode(-1)
  p = dummy
  for elem in vals:
    p.next = ListNode(elem)
    p = p.next
  return dummy.next  

def removeKthFromEnd(head, k):
  fast = head
  slow = head
  # Having two pointers, faster one are k posisition ahead of slower one
  for i in xrange(k):
    fast = fast.next
  # Faster pointer initially starts at position 0, and after k shift, it's null, that means, k is the length of the list.
  # So we should remove the first element from the list  
  if not fast:
    return head.next
  # Now move fast all the way until the last element in the list
  # If fast == last elem, then slow should be last k + 1 element.
  # Becasue fast is always k position ahead of slow. (k+1 - 1 == k)
  while fast and fast.next:
    fast = fast.next
    slow = slow.next

  # Remove the k-th last elem from the list
  slow.next = slow.next.next
  return head

def main():
  assert removeKthFromEnd(List(1), 1) == List()
  assert removeKthFromEnd(List(1, 2), 1) == List(1)
  assert removeKthFromEnd(List(1, 2, 3), 1) == List(1, 2)
  assert removeKthFromEnd(List(1, 2, 3), 2) == List(1, 3)
  assert removeKthFromEnd(List(1, 2, 3), 3) == List(2, 3)

if __name__ == '__main__':
  main()
```

### May 1, 2019 \[Easy\] Balanced Brackets
---
> **Question:** Given a string of round, curly, and square open and closing brackets, return whether the brackets are balanced (well-formed).
> For example, given the string "([])[]({})", you should return true.
> Given the string "([)]" or "((()", you should return false.

**My thoughts:** Whenever there is an open bracket, there should be a corresponding close bracket. Likewise, whenver we encounter a close bracket that does not match corresponding open braket, such string is not valid.

So the idea is to iterate through the string, store all the open bracket, and whenever we see a close bracket we check and see if it matches the most rent open breaket we stored early. The data structure, **Stack**, staisfies all of our requirements.

**Python Solution:** Link: [https://repl.it/@trsong/isBalancedBrackets](https://repl.it/@trsong/isBalancedBrackets)
```py
def isBalancedBrackets(input):
  if not input:
    return True
  stack = []
  lookup = {'(':')', '[':']', '{':'}'}
  for char in input:
    # Push open bracket to stack
    if char in lookup:
      stack.append(char)

    # Pop open bracket and check if it matches corresponding close bracket  
    elif stack and char == lookup[stack[-1]]:
      stack.pop()

    # Short-circuit if open bracket not matches the counterpart
    else:
      return False

  # All char should be processed and nothing should remain in the stack    
  return not stack

def main():
  assert isBalancedBrackets("")
  assert not isBalancedBrackets(")")
  assert not isBalancedBrackets("(")
  assert not isBalancedBrackets("(]")
  assert not isBalancedBrackets("[}")
  assert not isBalancedBrackets("((]]")
  assert not isBalancedBrackets("(][)")
  assert isBalancedBrackets("(([]))")
  assert isBalancedBrackets("[]{}()")
  assert not isBalancedBrackets("())))")
  assert isBalancedBrackets("[](())")

if __name__ == '__main__':
  main()
```


### Apr 30, 2019 \[Medium\] Second Largest in BST
---

> **Question：**  Given the root to a binary search tree, find the second largest node in the tree.

**My thoughts:**
Recall the way we figure out the largest element in BST: we go all the way to the right until not possible. So the second largest element must be on the left of largest element. 
We have two possibilities here:
- Either it's the parent of rightmost element, if there is no child underneath
- Or it's the rightmost element in left subtree of the rightmost element. 

```
case 1: parent
1
 \ 
 [2]
   \
    3

case 2: rightmost of left subtree of rightmost
1
 \
  4
 /
2
 \ 
 [3] 
``` 


**Python solution:** 
Link: [https://repl.it/@trsong/secondLargestInBST](https://repl.it/@trsong/secondLargestInBST)
```py
class TreeNode(object):
  def __init__(self, x, left=None, right=None):
    self.val = x
    self.left = left
    self.right = right

def secondLargestInBST(node):
    if node is None:
      return None

    # Go all the way to the right until not possible
    # Store the second largest on the way
    secondLargest = None
    current = node
    while current.right:
      secondLargest = current
      current = current.right

    # At the rightmost element now and there exists no element between secondLargest and rightmost element
    # We have second largest already
    if current.left is None:
      return secondLargest

    # Go left and all the way to the right and the rightmost element shall be the second largest
    secondLargest = current.left
    current = current.left
    while current:
      secondLargest = current
      current = current.right
    return secondLargest

def main():
  emptyTree = None
  oneElementTree = TreeNode(1)
  leftHeavyTree1 = TreeNode(2, TreeNode(1))
  leftHeavyTree2 = TreeNode(3, TreeNode(1, right=TreeNode(2)))
  rightHeavyTree1 = TreeNode(1, right=TreeNode(2))
  rightHeavyTree2 = TreeNode(1, right=TreeNode(3, TreeNode(2)))
  balanceTree1 = TreeNode(2, TreeNode(1), TreeNode(3))
  balanceTree2 = TreeNode(6, TreeNode(5), TreeNode(7))
  balanceTree3 = TreeNode(4, balanceTree1, balanceTree2)
  sampleCase1 = TreeNode(1, right=TreeNode(2, right=TreeNode(3)))
  sampleCase2 = TreeNode(1, right=TreeNode(4, left=TreeNode(2, right=TreeNode(3))))
  assert secondLargestInBST(emptyTree) is None
  assert secondLargestInBST(oneElementTree) is None
  assert secondLargestInBST(leftHeavyTree1).val == 1
  assert secondLargestInBST(leftHeavyTree2).val == 2
  assert secondLargestInBST(rightHeavyTree1).val == 1
  assert secondLargestInBST(rightHeavyTree2).val == 2
  assert secondLargestInBST(balanceTree1).val == 2
  assert secondLargestInBST(balanceTree3).val == 6
  assert secondLargestInBST(sampleCase1).val == 2
  assert secondLargestInBST(sampleCase2).val == 3

if __name__ == '__main__':
    main()
```
 
