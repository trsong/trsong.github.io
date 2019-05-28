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

### June 9, 2019 \[Medium\]
---
> **Question:** Implement locking in a binary tree. A binary tree node can be locked or unlocked only if all of its descendants or ancestors are not locked.
> 
> Design a binary tree node class with the following methods:
>
> - `is_locked`, which returns whether the node is locked
lock, which attempts to lock the node. If it cannot be locked, then it should return false. Otherwise, it should lock it and return true.
> - `unlock`, which unlocks the node. If it cannot be unlocked, then it should return false. Otherwise, it should unlock it and return true.
>
> You may augment the node to add parent pointers or any other property you would like. You may assume the class is used in a single-threaded program, so there is no need for actual locks or mutexes. Each method should run in O(h), where h is the height of the tree.

### June 8, 2019 \[Easy\]
---
> **Question:** Given a array of numbers representing the stock prices of a company in chronological order, write a function that calculates the maximum profit you could have made from buying and selling that stock once. You must buy before you can sell it.
>
> For example, given [9, 11, 8, 5, 7, 10], you should return 5, since you could buy the stock at 5 dollars and sell it at 10 dollars.

### June 7, 2019 \[Hard\] Array Shuffle
---
> **Question:** Given an array, write a program to generate a random permutation of array elements. This question is also asked as “shuffle a deck of cards” or “randomize a given array”. Here shuffle means that every permutation of array element should equally likely.

```
Input: [1, 2, 3, 4, 5, 6]
Output: [3, 4, 1, 5, 6, 2]
The output can be any random permutation of the input such that all permutation are equally likely.
```
> **Hint:** Given a function that generates perfectly random numbers between 1 and k (inclusive) where k is an input, write a function that shuffles the input array using only swaps.

### June 6, 2019 \[Medium\]
---
> **Question:** An sorted array of integers was rotated an unknown number of times.
> 
> Given such an array, find the index of the element in the array in faster than linear time. If the element doesn't exist in the array, return null.
> 
> For example, given the array [13, 18, 25, 2, 8, 10] and the element 8, return 4 (the index of 8 in the array).
> 
> You can assume all the integers in the array are unique.


### June 5, 20119 \[Easy\]
---
> **Question:** Given a function `rand5()`, use that function to implement a function `rand7()` where rand5() returns an integer from 1 to 5 (inclusive) with uniform probability and `rand7()` is from 1 to 7 (inclusive). Also, use of any other library function and floating point arithmetic are not allowed.

### June 4, 20119 \[Hard\]
---
> **Question:** Given a string, find the longest palindromic contiguous substring. If there are more than one with the maximum length, return any one.
>
> For example, the longest palindromic substring of "aabcdcb" is "bcdcb". The longest palindromic substring of "bananas" is "anana".

### June 3, 20119 \[Medium\]
---
> **Question:** Given an undirected graph represented as an adjacency matrix and an integer k, write a function to determine whether each vertex in the graph can be colored such that no two adjacent vertices share the same color using at most k colors.

### June 2, 2019 \[Medium\] 
---
> **Question:** Write an algorithm to justify text. Given a sequence of words and an integer line length k, return a list of strings which represents each line, fully justified.
> 
>  More specifically, you should have as many words as possible in each line. There should be at least one space between each word. Pad extra spaces when necessary so that each line has exactly length k. Spaces should be distributed as equally as possible, with the extra spaces, if any, distributed starting from the left.
> 
> If you can only fit one word on a line, then you should pad the right-hand side with spaces.
> 
> Each word is guaranteed not to be longer than k.
 
 For example, given the list of words ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"] and k = 16, you should return the following:

```py
["the  quick brown",
 "fox  jumps  over",
 "the   lazy   dog"]
```

### June 1, 2019 \[Medium\] 
---
> **Question:** Given a string s and an integer k, break up the string into multiple lines such that each line has a length of k or less. You must break it up so that words don't break across lines. Each line has to have the maximum possible amount of words. If there's no way to break the text up, then return null.
>
> You can assume that there are no spaces at the ends of the string and that there is exactly one space between each word.
>
> For example, given the string "the quick brown fox jumps over the lazy dog" and k = 10, you should return: ["the quick", "brown fox", "jumps over", "the lazy", "dog"]. No string in the list has a length of more than 10.

### May 31, 2019 \[Easy\] 
---
> **Question:** Generate 0 and 1 with 25% and 75% probability
> Given a function rand50() that returns 0 or 1 with equal probability, write a function that returns 1 with 75% probability and 0 with 25% probability using rand50() only. Minimize the number of calls to rand50() method. Also, use of any other library function and floating point arithmetic are not allowed.

### May 30, 2019 \[Hard\] 
---
> **Question:** Given an array of strictly the characters 'R', 'G', and 'B', segregate the values of the array so that all the Rs come first, the Gs come second, and the Bs come last. You can only swap elements of the array.
>
> Do this in linear time and in-place.
>
> For example, given the array ['G', 'B', 'R', 'R', 'B', 'R', 'G'], it should become ['R', 'R', 'R', 'G', 'G', 'B', 'B']

### 🎂 May 29, 2019 \[Medium\]
---
> **Question:** Given pre-order and in-order traversals of a binary tree, write a function to reconstruct the tree.
>
> For example, given the following preorder traversal:
>
```py
[a, b, d, e, c, f, g]
```

> And the following inorder traversal:

```py
[d, b, e, a, f, c, g]
```

> You should return the following tree:

```
    a
   / \
  b   c
 / \ / \
d  e f  g
```

-->

### May 28, 2019 \[Hard\] Subset Sum
---
> **Question:** Given a list of integers S and a target number k, write a function that returns a subset of S that adds up to k. If such a subset cannot be made, then return null.
>
> Integers can appear more than once in the list. You may assume all numbers in the list are positive.
> 
> For example, given S = [12, 1, 61, 5, 9, 2] and k = 24, return [12, 9, 2, 1] since it sums up to 24.

**My thoughts:** It's too expensive to check every subsets one by one. What we can do is to divide and conquer this problem.

For each element, we either include it to pursuing for target or not include it:
```py
subset_sum(numbers, target) = subset_sum(numbers[1:], target) or subset_sum(numbers[1:], target - numbers[0])
```

or derive dp formula from above recursive relation:

```py
dp[target][n] = dp[target][n-1] or dp[target-numbers[n-1]][n-1]
```

**DP Solution inspired by yesterday's problem:** [https://repl.it/@trsong/Multiset-Partition](https://repl.it/@trsong/Multiset-Partition)
```py
def subset_sum(numbers, target):
    n = len(numbers)
    # Let dp[sum][n] represent exiting subset of numbers[:n] with sum as sum:
    # * dp[sum][n] = dp[sum][n-1] or dp[sum-a[n-1]][n-1]
    # * dp[0][0..n] = True
    # The final result = dp[target][n]
    dp = [[False for _ in xrange(n+1)] for _ in xrange(target+1)]
    for i in xrange(n+1):
        dp[0][i] = True
    for s in xrange(1, target+1):
        for i in xrange(1, n+1):
            if s - numbers[i-1] < 0:
                dp[s][i] = dp[s][i-1]
            else:
                dp[s][i] = dp[s][i-1] or dp[s - numbers[i-1]][i-1]
    return dp[target][n] 
```

### May 27, 2019 \[Medium\] Multiset Partition
---
> **Question:** Given a multiset of integers, return whether it can be partitioned into two subsets whose sums are the same.
>
> For example, given the multiset {15, 5, 20, 10, 35, 15, 10}, it would return true, since we can split it up into {15, 5, 10, 15, 10} and {20, 35}, which both add up to 55.
>
> Given the multiset {15, 5, 20, 10, 35}, it would return false, since we can't split it up into two subsets that add up to the same sum.

**Incorrect Solution:**
```py
def incorrect_solution(multiset):
    # When input = [5, 1, 5, 1], expect result to be True but given False
    if not multiset or len(multiset) <= 1: return False
    sorted_multiset = sorted(multiset)
    balance = 0
    lo = 0
    hi = len(multiset) - 1
    while lo < hi:
        if balance <= 0:
            balance += sorted_multiset[lo]
            lo += 1
        else:
            balance -= sorted_multiset[hi]
            hi -= 1
    return balance == 0
```

**My thoughts:** Initially I thought the problem is trivial as above, I simply sort the array and find a pivot position that can happen to break the array into two equal sum sub-arrays. However, I sooner realize that I cannot make assumption that one array will always have elements less than the other. eg. `[4, 2, 1, 1]` can break into `[1, 2, 2]` and `[4]` where all of `1, 2, 2 < 4`. But `[5, 1, 5, 1]` does not work if break into `[1, 1, 5]` and `[5]`.

> Note: watch over for case `[5, 1, 5, 1]` and `[1, 2, 4, 3, 8]`

After spend some time thinking more deeply about this issue, I find that this problem is actually equivalent to find target sum in subset. eg. Find subset in `[5, 1, 5, 1]` such that the sum equals 6. 

> Check out tomorrow's question for subset sum.


**DP Solution:** [https://repl.it/@trsong/Multiset-Partition](https://repl.it/@trsong/Multiset-Partition)
```py
def has_equal_sum_partition(multiset):
    if not multiset: return False
    total = sum(multiset)
    if total % 2 == 1: return False
    target = total / 2
    return subset_sum(multiset, target)

def subset_sum(numbers, target):
    n = len(numbers)
    # Let dp[sum][n] represent exiting subset of numbers[:n] with sum as sum:
    # * dp[sum][n] = dp[sum][n-1] or dp[sum-a[n-1]][n-1]
    # * dp[0][0..n] = True
    # The final result = dp[target][n]
    dp = [[False for _ in xrange(n+1)] for _ in xrange(target+1)]
    for i in xrange(n+1):
        dp[0][i] = True
    for s in xrange(1, target+1):
        for i in xrange(1, n+1):
            if s - numbers[i-1] < 0:
                dp[s][i] = dp[s][i-1]
            else:
                dp[s][i] = dp[s][i-1] or dp[s - numbers[i-1]][i-1]
    return dp[target][n] 


def main():
    assert has_equal_sum_partition([5, 1, 5, 1])
    assert not has_equal_sum_partition([1, 2, 2])
    assert not has_equal_sum_partition([])
    assert has_equal_sum_partition([15, 5, 20, 10, 35, 15, 10])
    assert not has_equal_sum_partition([15, 5, 20, 10, 35])
    assert has_equal_sum_partition([1, 2, 4, 3, 8])


if __name__ == '__main__':
    main()
```


### May 26, 2019 \[Medium\] Tic-Tac-Toe Game
---
> **Questions:** Implementation of Tic-Tac-Toe game. Rules of the Game:
> 
> - The game is to be played between two people.
One of the player chooses ‘O’ and the other ‘X’ to mark their respective cells.
> - The game starts with one of the players and the game ends when one of the players has one whole row/ column/ diagonal filled with his/her respective character (‘O’ or ‘X’).
> - If no one wins, then the game is said to be draw.
>
> Follow-up: create a method that makes the next optimal move such that the person should never lose if make that move.

**My thoughts:** There are multiple ways to implement Tic-Tac-Toe. I choose to use Observer Pattern which can be used to solve all grid-like game. 

> From Wiki, ***Observer Pattern*** is used when there is one-to-many relationship between objects such as if one object is modified, its depenedent objects are to be notified automatically.

So basically, the way this pattern works is that each cell (the observable) has at most 4 observers: row obser, col observer, and two cross observers. Each observer is responsible for check if all cells underneth have the same value. 

eg. If grid size is 3 and for cell at `(1,1)`,  row_observers contains all cells at row 1: `(1,0)`, `(1,1)` and `(1,2)`. col_observers has all cells at col 1: `(1, 0)`, `(1, 1)` and `(2, 1)`. Two cross observers check the diagonal cells `(0, 0)`, `(1, 1)` and `(2, 2)` as well as `(0, 2)`, `(1, 1)` and `(2, 0)`. When we make a move at cell `(1, 1)`, then all of 4 observers will be notified to check if all cell underneth have the same value, either 'o' or 'x'. If there exists 1 observer satisfies, then the game ends. Otherwise, it continues.

**Implementation Using Observer Pattern:** [https://repl.it/@trsong/Tic-Tac-Toe-Game](https://repl.it/@trsong/Tic-Tac-Toe-Game)
```py
class CellObserver(object):
    def __init__(self, cell, neighbors):
        self._cell = cell
        self._neighbors = neighbors
    
    def notify(self):
        return all(neighbor.val == self._cell.val for neighbor in self._neighbors)


class CellSubject(object):
    def __init__(self, val='_'):
        self.val = val
        self._observers = []

    def add_observer(self, cell_observer):
        self._observers.append(cell_observer)

    def notify_observers(self):
        return any(observer.notify() for observer in self._observers)


class Grid(object):
    def __init__(self, size, first_player='x'):
        self._next_move = first_player
        self._size = size
        self._reset()

    def _reset(self):
        n = self._size
        self._turn = 0
        self._game_finished = False
        self._table = [[ CellSubject() for _ in xrange(n)] for _ in xrange(n)]

        row_neighbors = [None] * n
        col_neighbors = [None] * n
        for i in xrange(n):
            row_neighbors[i] = [self._table[i][col] for col in xrange(n)]
            col_neighbors[i] = [self._table[row][i] for row in xrange(n)]

        for i in xrange(n):
            for j in xrange(n):
                cell = self._table[i][j]
                cell.add_observer(CellObserver(cell, row_neighbors[i]))
                cell.add_observer(CellObserver(cell, col_neighbors[j]))

        top_left_to_bottom_right = [self._table[i][i] for i in xrange(n)]
        for i in xrange(n):
            cell = self._table[i][i]
            cell.add_observer(CellObserver(cell,  top_left_to_bottom_right))

        top_right_to_bottom_left = [self._table[i][n-1-i] for i in xrange(n)]
        for i in xrange(n):
            cell = self._table[i][n-1-i]
            cell.add_observer(CellObserver(cell, top_right_to_bottom_left))

    def _switch_player(self):
        self._next_move = 'o' if self._next_move == 'x' else 'x'

    def get_next_move(self):
        return self._next_move

    def display(self):
        for row in xrange(self._size):
            print ','.join(cell.val for cell in self._table[row])

    def move(self, row, col):
        if not (0 <= row < self._size and 0 <= col < self._size):
            print "row or col out of range."
            return 
        cell = self._table[row][col]
        if cell.val != '_':
            print "Cell has been occupied. Choose a different cell"
        else:
            self._turn += 1
            cell.val = self._next_move
            self._game_finished = cell.notify_observers()
            self._switch_player()

    def is_game_finished(self):
        return self._game_finished or self._turn > self._size * self._size

    def result(self):
        if self._turn <= self._size * self._size and self._game_finished:
            loser = self.get_next_move()
            winner = 'o' if loser == 'x' else 'x'
            return 'Game over. %s wins.' % winner
        else:
            return "No one wins. There is a tie"


class Game(object):
    def start(self): 
        print '====== Welcome to Tic-Tac-Toe Game ======'
        while not raw_input("Press Enter to start..."):
            self.game_init()

    def game_init(self):
        size = int(input('Please choose grid size: '))
        self._grid = Grid(size)
        while not self._grid.is_game_finished():
            r, c = [int(x) for x in raw_input("Next player place %s in row, col: " % self._grid.get_next_move()).split(',')]
            self._grid.move(r, c)
            self._grid.display()
        print self._grid.result()

    def test(self):
        self._grid = Grid(3)
        self._grid.move(0, 0)
        self._grid.move(1, 1)
        self._grid.move(2, 2)
        self._grid.move(0, 1)
        self._grid.move(2, 1)
        self._grid.move(2, 0)
        self._grid.move(0, 2)
        self._grid.move(1, 2)
        self._grid.move(1, 0)
        assert self._grid.result() == "No one wins. There is a tie"
        self._grid = Grid(1)
        self._grid.move(0, 0)
        assert self._grid.result() == "Game over. x wins."


class main():
    game = Game()
    game.test()
    game.start()


if __name__ == '__main__':
    main()
```

**Game demo**
```
====== Welcome to Tic-Tac-Toe Game ======
Press Enter to start...
Please choose grid size: 3
Next player place x in row, col: 1,1
_,_,_
_,x,_
_,_,_
Next player place o in row, col: 0,1
_,o,_
_,x,_
_,_,_
Next player place x in row, col: 0,0
x,o,_
_,x,_
_,_,_
Next player place o in row, col: 2,2
x,o,_
_,x,_
_,_,o
Next player place x in row, col: 3,0
row or col out of range.
x,o,_
_,x,_
_,_,o
Next player place x in row, col: 0,0
Cell has been occupied. Choose a different cell
x,o,_
_,x,_
_,_,o
Next player place x in row, col: 2,0
x,o,_
_,x,_
x,_,o
Next player place o in row, col: 0,2
x,o,o
_,x,_
x,_,o
Next player place x in row, col: 1,0
x,o,o
x,x,_
x,_,o
Game over. x wins.
Press Enter to start...
```

> Follow-up Question. TODO tsong: It's 3:30 am, and I'm just too tired. I was planning to implement that using Backtracking algorithem. However, my mind is kinda dizzy and it's time to get some rest. I will probably go back to this question once get a chance in the future.

### May 25, 2019 \[Easy\] Run-length Encoding
---
> **Question:** Run-length encoding is a fast and simple method of encoding strings. The basic idea is to represent repeated successive characters as a single count and character. For example, the string "AAAABBBCCDAA" would be encoded as "4A3B2C1D2A".
>
> Implement run-length encoding and decoding. You can assume the string to be encoded have no digits and consists solely of alphabetic characters. You can assume the string to be decoded is valid.

**Python Solution:** [https://repl.it/@trsong/Run-length-Encoding](https://repl.it/@trsong/Run-length-Encoding)
```py
class Run_Length_Encoding(object):
    @staticmethod
    def decode(text):
        if not text: return ""
        i = 0
        res = []
        while i < len(text):
            repeat = 0
            while '0' <= text[i] <= '9':
                repeat = 10 * repeat + int(text[i])
                i += 1
            res.append(text[i] * repeat)
            i += 1
        return ''.join(res)
        
    @staticmethod
    def encode(text):
        if not text: return ""
        i = 0
        res = []
        while i < len(text):
            cur = text[i]
            repeat = 1
            i += 1
            while i < len(text) and text[i] == cur:
                repeat += 1
                i += 1
            res.append("{}{}".format(repeat, cur))
        return ''.join(res)


def test_result(raw_text, encoded_text):
    assert Run_Length_Encoding.encode(raw_text) == encoded_text
    assert Run_Length_Encoding.decode(encoded_text) == raw_text


def main():
    test_result("", "")
    test_result("ABC", "1A1B1C")
    test_result("A"*10, "10A")
    test_result("A" * 100 + "B" + "C" * 99, "100A1B99C")
    test_result("ABBCCCDDDDCCCBBA", "1A2B3C4D3C2B1A")


if __name__ == '__main__':
    main()
```

### May 24, 2019 \[Medium\] Maximum Subarray Sum
---
> **Question:** Given an array of numbers, find the maximum sum of any contiguous subarray of the array.
>
> For example, given the array [34, -50, 42, 14, -5, 86], the maximum sum would be 137, since we would take elements 42, 14, -5, and 86.
>
> Given the array [-5, -1, -8, -9], the maximum sum would be 0, since we would not take any elements.
>
> Do this in O(N) time.

**My thoughts:** Start from trival examples and then generalize our solution. 

Properties can be derived from trivial example:
1. if all nums are positive, then the result equals sum of all of them
2. if all nums are negative, then the result equals 0.

Consider we have answer when the input size equals i - 1. Then when we encouter the ith element:
- if it's positive, then we use property 1
- if it's negative, but when include the current num, the result is still positive, then we still use property 1. 'coz the list is equivalent to some all positive list.
- if it's negative, but even we include the current, the result is negative, then we use property 2. 'coz the list is equivalent to some all negative list.

**Python Solution:** [https://repl.it/@trsong/Maximum-Subarray-Sum](https://repl.it/@trsong/Maximum-Subarray-Sum)
```py
def max_sub_array(nums):
    if not nums: return 0
    max_sum = 0
    accu_sum = 0
    for num in nums:
        if num > 0:
            accu_sum += num
            max_sum = max(max_sum, accu_sum)
        elif accu_sum + num > 0:
            accu_sum += num
        else:
            # when accu_sum + num <= 0
            accu_sum = 0
    return max_sum


def main():
    assert max_sub_array([]) == 0
    assert max_sub_array([1]) == 1
    assert max_sub_array([-1]) == 0
    assert max_sub_array([34, -50, 42, 14, -5, 86])  == 137
    assert max_sub_array([-5, -1, -8, -9]) == 0
    assert max_sub_array([-1, -1, -1, 5, -1 , -1]) == 5
    assert max_sub_array([1, 1, -5, 1, 1]) == 2
    assert max_sub_array([1, 1, -1, 1, 1]) == 3
    assert max_sub_array([1, 1, -1, 1, 1, 1]) == 4


if __name__ == '__main__':
    main()
```

### May 23, 2019 \[Hard\] LRU Cache
---
> **Question:** Implement an LRU (Least Recently Used) cache. It should be able to be initialized with a cache size n, and contain the following methods:
>
> - `set(key, value)`: sets key to value. If there are already n items in the cache and we are adding a new item, then it should also remove the least recently used item.
>
> - `get(key)`: gets the value at key. If no such key exists, return null.
>  
> Each operation should run in O(1) time.

**My thoughts:** The implementation of LRU cache can be similar to ***Linked Hash Map***: a map that also preserve the insertion order. It has a lookup table for fast value retrival and a doubly linked list to keep track of least recent used item that can be evicted when the cache is full. 

However, during interview, usually the candidate will be asked to use a singly linked list to implement (once during SAP on-site). The trick is to figure out a way to remove an element in a singly linked list which can be achived use the following:

```py
node.val = node.next.val
node.next = node.next.next
```

Note that above method not work if attempt to remove the very last element which will cause null pointer exception. A way to overcome this issue is to create a dummy node at the end of list.

P.S: For doubly linked list solution check LC 146.

**Singly Linked-List Solution:** [https://repl.it/@trsong/LRU-Cache](https://repl.it/@trsong/LRU-Cache)
```py
class ListNode(object):
    def __init__(self, key=None, val=None, next=None):
        self.key = key
        self.val = val
        self.next = next


class LRUCache(object):
    def __init__(self, capacity):
        self._capacity = capacity
        self._size = 0
        self._start = ListNode()
        self._end = self._start
        self._lookup = {}

    def _populate(self, key):
        node = self._lookup[key]
        # Duplicate node and append it to the end of list
        self._end.key, self._end.val, self._end.next  = node.key, node.val, ListNode()
        self._lookup[node.key] = self._end
        self._end = self._end.next
        # Remove node
        node.key, node.val, node.next = node.next.key, node.next.val, node.next.next
        self._lookup[node.key] = node

    def get(self, key):
        if key not in self._lookup: return None
        self._populate(key)
        return self._lookup[key].val

    def set(self, key, val):
        if key in self._lookup:
            self._lookup[key].val = val
            self._populate(key)
        else:
            if self._size >= self._capacity:
                del self._lookup[self._start.key]
                self._start = self._start.next
                self._size -= 1
            self._start = ListNode(key, val, self._start)
            self._lookup[key] = self._start
            self._populate(key)
            self._size += 1


def main():
    cache = LRUCache(3)
    cache.set(0, 0)  # Least Recent -> 0 -> Most Recent
    cache.set(1, 1)  # Least Recent -> 0, 1 -> Most Recent
    cache.set(2, 2)  # Least Recent -> 0, 1, 2 -> Most Recent
    cache.set(3, 3)  # Least Recent -> 1, 2, 3 -> Most Recent. Evict 0
    assert cache.get(0) is None  
    assert cache.get(2) == 2  # Least Recent -> 1, 3, 2 -> Most Recent
    cache.set(4, 4)  # Least Recent -> 3, 2, 4 -> Most Recent. Evict 1 
    assert cache.get(1) is None
    assert cache.get(2) == 2  # Least Recent -> 3, 4, 2 -> Most Recent 
    assert cache.get(3) == 3  # Least Recent -> 4, 2, 3 -> Most Recent
    assert cache.get(2) == 2  # Least Recent -> 4, 3, 2 -> Most Recent
    cache.set(5, 5)  # Least Recent -> 3, 2, 5 -> Most Recent. Evict 4
    cache.set(6, 6)  # Least Recent -> 2, 5, 6 -> Most Recent. Evict 3
    assert cache.get(4) is None
    assert cache.get(3) is None
    cache.set(7, 7)  # Least Recent -> 5, 6, 7 -> Most Recent. Evict 2
    assert cache.get(2) is None


if __name__ == '__main__':
    main()
```

### May 22, 2019 \[Easy\] Special Stack
---
> **Question:** Implement a special stack that has the following methods:
>
> - `push(val)`, which pushes an element onto the stack
> - `pop()`, which pops off and returns the topmost element of the stack. If there are no elements in the stack, then it should throw an error or return null.
> - `max()`, which returns the maximum value in the stack currently. If there are no elements in the stack, then it should throw an error or return null.
> 
> Each method should run in constant time.

**Python Solution:** [https://repl.it/@trsong/Special-Stack](https://repl.it/@trsong/Special-Stack)
```py
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class SpecialStack(object):
    def __init__(self):
        self._stack = None
        self._max_stack = None
        self._size = 0

    def push(self, val):
        max_val = max(val, self._max_stack.val) if self._size > 0 else val
        self._max_stack = ListNode(max_val, self._max_stack)
        self._stack = ListNode(val, self._stack)
        self._size += 1

    def pop(self):
        if self._size <= 0:
            return None
        else:
            val = self._stack.val
            self._stack = self._stack.next
            self._max_stack = self._max_stack.next
            self._size -= 1
            return val

    def max(self):
        if self._size <= 0:
            return None
        else:
            return self._max_stack.val


def main():
    stack = SpecialStack()
    assert stack.pop() is None
    assert stack.max() is None
    stack.push(1)
    stack.push(2)
    stack.push(3)
    assert stack.max() == 3
    assert stack.pop() == 3
    assert stack.max() == 2
    assert stack.pop() == 2
    assert stack.pop() == 1
    assert stack.pop() is None
    stack.push(3)
    stack.push(2)
    stack.push(1)
    assert stack.max() == 3
    assert stack.pop() == 1
    assert stack.pop() == 2


if __name__ == '__main__':
    main()
```


### May 21, 2019 \[Hard\] Random Elements from Infinite Stream
---
> **Question:** Randomly choosing a sample of k items from a list S containing n items, where n is either a very large or unknown number. Typically, n is too large to fit the whole list into main memory.

**My thoughts:** Choose k elems randomly from an infitite stream can be tricky. We should simplify this question, find the pattern and then generalize the solution. So, let's first think about how to randomly get one element from the stream:

As the input stream can be arbitrarily large, our solution must be able to calculate the chosen element on the fly. So here comes the strategy:

```
When consume the i-th element:
- Either choose the i-th element with 1/(i+1) chance. 
- Or, keep the last chosen element with 1 - 1/(i+1) chance.
```

**Proof by Induction:**
- Base case: when there is 1 element, then the 0th element is chosen by `1/(0+1) = 100%` chance.
- Inductive Hypothesis: Suppose for above strategy works for all elemements between 0th and i-1th, which means all elem has `1/i` chance to be chosen.
- Inductive Step: when consume the i-th element:
  - If the i-th element is selected with `1/(i+1)` chance, then it works for the i-th element 
  - As for other elements to be choosen, we will have `1-1/(i+1)` chance not to choose the i-th element. And also based on the Inductive Hypothesis: all elemements between 0th and i-1th has 1/i chance to be chosen. Thus, the chance of any elem between 0-th to i-th element to be chosen in this round eqauls `1/i * (1-(1/(i+1))) = 1/(i+1)`, which means you cannot choose the i-th element and at the same time you choose any element between 0-th and i-th element. Therefore it works for all previous elements as each element has `1/(i+1)` chance to be chosen.

**Random One Element Solution:** [https://repl.it/@trsong/Random-One-Elem](https://repl.it/@trsong/Random-One-Elem)

```py
from random import randint

def random_one_elem(stream):
    pick = None
    for i, elem in enumerate(stream):
        if i == 0 or randint(0, i) == 0:
            # Has 1/(i+1) chance to pick i-th elem
            pick = elem
    return pick


def print_distribution(stream, repeat):
    histogram = {}
    for _ in xrange(repeat):
        res = random_one_elem(stream)
        if res not in histogram:
            histogram[res] = 0
        histogram[res] += 1
    print histogram


def main():
    # Distribution looks like {0: 1003, 1: 1004, 2: 943, 3: 994, 4: 1023, 5: 1019, 6: 1013, 7: 1025, 8: 1005, 9: 971}
    print_distribution(xrange(10), repeat=10000)


if __name__ == '__main__':
    main()
```

Ever since for randomly choose 1 element, we keep 1 chosen element during execution, k elements will be kept and we will use the following strategy to keep and kick out elements:

```
When consume the i-th element:
- Either choose the i-th element with k/(i+1) chance. (And kick out any of the chosen k element)
- Or, keep the last chosen elements with 1 - k/(i+1) chance.
```

Similar to previous proof to show that each element has `1/(i+1)` chance. It can be easily shown that each element has `k/(i+1)` to be chosen. Just google ***Reservoir Sampling*** if you are curious about other strategies.

**Random K Elements Solution:** [https://repl.it/@trsong/Random-K-Elem](https://repl.it/@trsong/Random-K-Elem)
```py
from random import randint

def random_k_elem(stream, k):
    pick = [None] * k 
    for i, elem in enumerate(stream):
        if i < k:
            pick[i] = elem
        elif randint(1, i+1) <= k:
            # The i-th elem has k/(n+1) chance to be picked
            pick[randint(0, k-1)] = elem
    return pick


def print_distribution(stream, k, repeat):
    histogram = {}
    for _ in xrange(repeat):
        res = random_k_elem(stream, k)
        for e in res:
            if e not in histogram:
                histogram[e] = 0
            histogram[e] += 1
    print histogram


def main():
    # Distribution looks like:
    # {0: 2984, 1: 3008, 2: 3085, 3: 3045, 4: 3075, 5: 3001, 6: 2923, 7: 2913, 8: 2954, 9: 3012}
    print_distribution(xrange(10), k=3, repeat=10000)


if __name__ == '__main__':
    main()
```

### May 20, 2019 \[Hard\] Edit Distance
---
> **Question:**  The edit distance between two strings refers to the minimum number of character insertions, deletions, and substitutions required to change one string to the other. For example, the edit distance between “kitten” and “sitting” is three: substitute the “k” for “s”, substitute the “e” for “i”, and append a “g”.
> 
> Given two strings, compute the edit distance between them.

 **A little bit background information:** I first learnt this question in my 3rd year algorithem class. At that moment, this question was introduced to illustrate how dynamic programming works. And even nowadays, I can still recall the formula to be something like `dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + (0 if source[i] == target[j] else 1)`. However, when I ask my friends what dp represents in this question and how above formula works, few can give me convincing explanation. So the same thing could happen to readers like you: if you just know the formula without understanding which part represents insertion, removal or updating, then probably you just memorize the solution and pretend you understand the answer. And what could happen in the near future is that the next time when a similar question, like May 19 Regular expression, shows up during interview, you end up spending 20 min, got stuck trying to come up w/ dp formula.

**My thoughts:** My suggestion is that let's forget about dp at first, and we should just focus on recursion. (The idea between those two is kinda the same.) Just use the following template you learnt in first year of university to figure out the solution:

```py
def recursion(input):
    if ...base case...:
        return ...base case solution...
    else:
        do something to first of input 
        return recursion(rest of input)

def recursion2(input1, input2):
    if ...base case...:
        return ...base case solution...
    else:
        do something to first of input1
        do something to first of input2
        res1 = recursion2(rest of input1, input2)
        res2 = recursion2(input1, rest of input2)
        res3 = recursion2(rest of input1, rest of input2)
        return do something to res1, res2, res3

def recursion3(input1, input2):
    if ...base case...:
        return ...base case solution...
    else:
        do something to last of input1
        do something to last of input2
        res1 = recursion3(drop last of input1, input2)
        res2 = recursion3(input1, drop last of input2)
        res3 = recursion3(drop last of input1, drop last of input2)
        return do something to res1, res2, res3
```

Since we have two inputs for this question, we can either use recursion2 or recursion3 in above template, it doesn't really matter in this question. `def edit_distance(source, target)`. And now let's think about what is the base case. Base case is usually something trivial, like empty list. Then it's easy to know that if either source or target is empty, the edit distance is the other one's length. e.g. `edit_distance("", "kitten") == 6`

Then let's think about how we can shrink the size of source and target. In above template, there are 3 different ways to shrink the input size. Check the line res1, res2 and res3. 

1. Shrink source size: 

    I came up w/ some example like `edit_distance("ab", "a")` and `edit_distance("a", "a")`. Now think about the relationship between them. It turns out `edit_distance("ab", "a") == 1 + edit_distance("a", "a")`. As the minimum edit we need is just remove "b" from "ab" in source: only 1 extra edit.

2. Shrink target size:
   
    I came up w/ some example like `edit_distance("a", "ab")` and `edit_distance("a", "a")`. It turns out the mimum edit is to append "b" to the source "a" that gives "ab". So `edit_distance("a", "ab") == 1 + edit_distance("a", "a")`.

3. Shrink both source and target size: 

    I came up w/ some example like `edit_distance("aa", "ab")` and `edit_distance("a", "a")`. It seems we just need to replace "a" with "b" or "b" with "a". So only 1 more edit should be sufficient. However if we have `edit_distance("ab", "ab")` and `edit_distance("a", "a")`. Then that means we don't need to do anything when there is a match.
    That gives `edit_distance("aa", "ab") = 1 + edit_distance("a", "a")` and `edit_distance("ab", "ab") == edit_distance("a", "a")`

Note that any of res1, res2 and res3 might give the mimum edit. So we need to apply min to get the smallest among them.

**Python Solution:** [https://repl.it/@trsong/Edit-Distance](https://repl.it/@trsong/Edit-Distance)
```py
def edit_distance(source, target):
    if not source or not target:
        return max(len(source), len(target))
    # The edit_distance when insert/append an elem to source to match last elem in target
    insert_res = edit_distance(source, target[:-1]) + 1

    # The edit_distance when update last elem in source to match last elem in target
    update_res = edit_distance(source[:-1], target[:-1]) + (0 if source[-1] == target[-1] else 1)

    # The edit_distance when remove the last elem from source
    remove_res = edit_distance(source[:-1], target) + 1

    return min(insert_res, update_res, remove_res)


# If we modify the algorithem to insert/update/remove upon the first elem, it also works
def edit_distance2(source, target):
    if not source or not target:
        return max(len(source), len(target))
    # The edit_distance when insert/prepend an elem to source to match the first elem in target
    insert_res = edit_distance2(source, target[1:]) + 1

    # The edit_distance when update the first elem in source to match the first elem in target
    update_res = edit_distance2(source[1:], target[1:]) + (0 if source[0] == target[0] else 1)

    # The edit_distance when remove the first elem from source
    remove_res = edit_distance2(source[1:], target) + 1

    return min(insert_res, update_res, remove_res)


def main():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("sitting", "kitten") == 3
    assert edit_distance("sitting", "") == 7
    assert edit_distance("", "kitten") == 6
    assert edit_distance("", "") == 0


if __name__ == "__main__":
    main()
```

> **Note:** Above solution can be optimized using a cache. Or based on recursive formula, generate DP array. However, I feel too lazy for DP solution, you can probably google ***Levenshtein Distance*** or "edit distance". Here, I just give the optimized solution w/ cache. You can see how similar the dp solution vs optimziation w/ cache. And the benefit of using cache is that, you don't need to figure out the order to fill dp array as well as the initial value for dp array which is quite helpful. If you are curious about what question can give a weird dp filling order, just check May 19 question: Regular Expression and try to solve that w/ dp. 

**Python Solution w/ Cache:** [https://repl.it/@trsong/Edit-Distance-with-Cache](https://repl.it/@trsong/Edit-Distance-with-Cache)

```py
def edit_distance_helper(source, target, i, j, cache):
    # The edit_distance when insert/prepend an elem to source to match the first elem in target
    insert_res = edit_distance_helper_with_cache(source, target, i, j + 1, cache) + 1

    # The edit_distance when update the first elem in source to match the first elem in target
    update_res = edit_distance_helper_with_cache(source, target, i + 1, j + 1, cache) + (0 if source[i] == target[j] else 1)

    # The edit_distance when remove the first elem from source
    remove_res = edit_distance_helper_with_cache(source, target, i + 1, j, cache) + 1

    return min(insert_res, update_res, remove_res)


# edit_distance(source[i:], target[j:])
def edit_distance_helper_with_cache(source, target, i, j, cache):
    n, m = len(source), len(target)
    if i > n - 1 or j > m - 1:
        return max(n - i, m - j)

    if cache[i][j] is None:
        cache[i][j] = edit_distance_helper(source, target, i, j, cache)
    return cache[i][j]


def edit_distance(source, target):
    n, m = len(source), len(target)
    cache = [[None for _ in range(m)] for _ in range(n)]
    return edit_distance_helper_with_cache(source, target, 0, 0, cache)
```

### May 19, 2019 \[Hard\] Regular Expression: Period and Asterisk
---
> **Question:** Implement regular expression matching with the following special characters:
>
> - `.` (period) which matches any single character
> 
> - `*` (asterisk) which matches zero or more of the preceding element
> 
> That is, implement a function that takes in a string and a valid regular expression and returns whether or not the string matches the regular expression.
>
> For example, given the regular expression "ra." and the string "ray", your function should return true. The same regular expression on the string "raymond" should return false.
>
> Given the regular expression ".*at" and the string "chat", your function should return true. The same regular expression on the string "chats" should return false.

**My thoughts:** First consider the solution without `*` (asterisk), then we just need to match letters one by one while doing some special handling for `.` (period) so that it can match all letters. 

Then we consider the solution with `*`, we will need to perform ONE look-ahead, so `*` can represent 0 or more matching. And we consider those two situations separately:
- If we have 1 matching, we just need to check if the rest of text match the current pattern.
- Or if we do not have matching, then we need to advance both text and pattern.

**Python Solution:** [https://repl.it/@trsong/Pattern-Match](https://repl.it/@trsong/Pattern-Match)
```py
def pattern_match(text, pattern):
    if not pattern: return not text
    match_first = text and (text[0] == pattern[0] or pattern[0] == '.')
    if len(pattern) > 1 and pattern[1] == '*':
        # Either match first letter and continue check on rest; Or not match at all
        return match_first and pattern_match(text[1:], pattern) or pattern_match(text, pattern[2:])
    else:
        # Without "*", we will proceed both text and pattern
        return match_first and pattern_match(text[1:], pattern[1:])

def main():
    assert not pattern_match("a", "")
    assert not pattern_match("", "a")
    assert pattern_match("", "")
    assert pattern_match("aa", "a*")
    assert pattern_match("aaa", "ab*ac*a")
    assert pattern_match("aab", "c*a*b")
    assert pattern_match("ray", "ra.")
    assert not pattern_match("raymond", "ra.")
    assert pattern_match("chat", ".*at")
    assert not pattern_match("chats", ".*at")

if __name__ == '__main__':
    main()
```

> Notice that as we keep calling `text[1:]`, `pattern[1:]` and `pattern[2:]` in recursive calls, there are certain results that can be cached so that none of the same input will be executed twice.
> Also, keep generating substring is expensive, we should replace it w/ in-place checking by passing index.

**Python Solution with Cache:** [https://repl.it/@trsong/Pattern-Match2](https://repl.it/@trsong/Pattern-Match2)
```py
def pattern_match_helper(text, pattern, i, j, cache):
    n, m = len(text), len(pattern)
    match_first = i < n and (text[i] == pattern[j] or pattern[j] == '.')
    if j < m - 1 and pattern[j+1] == '*':
        # Either match first letter and continue check on rest; Or not match at all
        return match_first and pattern_match_helper_with_cache(text, pattern, i+1, j, cache) or pattern_match_helper_with_cache(text, pattern, i, j+2, cache)
    else:
        # Without "*", we will proceed both text and pattern
        return match_first and pattern_match_helper_with_cache(text, pattern, i+1, j+1, cache)

def pattern_match_helper_with_cache(text, pattern, i, j, cache):
    n, m = len(text), len(pattern)
    if j >= m: return i >= n

    if cache[i][j] is None:
        cache[i][j] = pattern_match_helper(text, pattern, i, j, cache)
    return cache[i][j]

def pattern_match(text, pattern):
    if not pattern: return not text
    cache = [[None for _ in xrange(len(pattern)+1)] for _ in xrange(len(text)+1)]
    return pattern_match_helper(text, pattern, 0, 0, cache)
```

### May 18, 2019 \[Easy\] Intersecting Node
---
> **Question:** Given two singly linked lists that intersect at some point, find the intersecting node. The lists are non-cyclical.
>
> For example, given A = 3 -> 7 -> 8 -> 10 and B = 99 -> 1 -> 8 -> 10, return the node with value 8.
>
> In this example, assume nodes with the same value are the exact same node objects.
>
> Do this in O(M + N) time (where M and N are the lengths of the lists) and constant space.

**My thoughts:** For two lists w/ same length, say `[1, 2, 3]` and `[4, 2, 3]`, the intersecting node is the first element shared by two list when iterate through both lists at the same time. In above example, 2 is the elem we are looking for. 

For two lists w/ different length, say `[1, 1, 1, 2 3]` and `[4, 2, 3]`. We can convert this case into above case by first calculating the length difference between those two lists and let larger list proceed that difference number of nodes ahead of time. Then we will end up w/ 2 lists of the same length.

eg. The difference between `[1, 1, 1, 2 3]` and `[4, 2, 3]` is 2. We proceed the pointer of larger list by 2, gives `[1, 2, 3]`. Thus we have 2 lists of the same length.

**Python Soluion:** [https://repl.it/@trsong/Intersecting-Node](https://repl.it/@trsong/Intersecting-Node)
```py
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

def get_intersecting_node(lst1, lst2):
    l1, l2 = lst1, lst2
    len1, len2 = 0, 0
    while l1:
        l1 = l1.next
        len1 += 1
    while l2:
        l2 = l2.next
        len2 += 1
        
    long_list, short_list = lst1, lst2
    diff = len1 - len2
    if diff < 0:
        long_list, short_list = lst2, lst1
        diff = -diff
    for _ in xrange(diff):
        long_list = long_list.next
    while long_list != short_list:
        long_list = long_list.next
        short_list = short_list.next
    return long_list

def main():
    shared = ListNode(8, ListNode(0))
    l1 = ListNode(3, ListNode(7, shared))
    l2 = ListNode(99, ListNode(1, shared))
    assert get_intersecting_node(l1, l2).val == 8
    assert get_intersecting_node(l2, l1).val == 8

    l3 = ListNode(10, ListNode(11, l1))
    assert get_intersecting_node(l3, l2).val == 8
    assert get_intersecting_node(l2, l3).val == 8

    assert get_intersecting_node(None, None) is None
    assert get_intersecting_node(l1, None) is None
    assert get_intersecting_node(None, l1) is None

if __name__ == '__main__':
    main()
```

### May 17, 2019 LC 332 \[Medium\] Reconstruct Itinerary
---
> **Questions:** Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK. Thus, the itinerary must begin with JFK.

Note:

> 1. If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string. For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].
> 2. All airports are represented by three capital letters (IATA code).
> 3. You may assume all tickets form at least one valid itinerary.
   
Example 1:

```java
Input: [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
Output: ["JFK", "MUC", "LHR", "SFO", "SJC"]
```
Example 2:

```java
Input: [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
Explanation: Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"].
             But it is larger in lexical order.
```

**My thoughts:** Forget about lexical requirement for now, consider all airports as vertices and each itinerary as an edge. Then all we need to do is to find a path from "JFK" that consumes all edges. By using DFS to iterate all potential solution space, once we find a solution we will return immediately.

Now let's consider the lexical requirement, when we search from one node to its neighbor, we can go from smaller lexical order first and by keep doing that will lead us to the result. 

**Python Solution:** [https://repl.it/@trsong/Reconstruct-Itinerary](https://repl.it/@trsong/Reconstruct-Itinerary)
```py
def search_itinerary_DFS(src, itinerary_lookup, route_res):
    # Using DFS to find a solution that can consume all edges
    while src in itinerary_lookup and itinerary_lookup[src]:
        search_itinerary_DFS(itinerary_lookup[src].pop(0), itinerary_lookup, route_res)
    # Once find a route, build the routes in reverse order so that source will go before destination 
    route_res.insert(0, src)
    
def reconstruct_itinerary(tickets):
    itinerary_lookup = {}
    # Build up neighbor lookup table
    for it in tickets:
        src, dst = it
        if src not in itinerary_lookup:
            itinerary_lookup[src] = [dst]
        else:
            itinerary_lookup[src].append(dst)
        
    # Sort all destinations in alphabet order so that smaller lexical order comes first
    for key in itinerary_lookup.keys():
        itinerary_lookup[key] = sorted(itinerary_lookup[key])
            
    res = []
    search_itinerary_DFS("JFK", itinerary_lookup, res)
    return res

def main():
    assert reconstruct_Itinerary([["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]) == ["JFK", "MUC", "LHR", "SFO", "SJC"]
    assert reconstruct_Itinerary([["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]) == ["JFK","ATL","JFK","SFO","ATL","SFO"]
    assert reconstruct_Itinerary([["JFK", "YVR"], ["LAX", "LAX"], ["YVR", "YVR"], ["YVR", "YVR"], ["YVR", "LAX"], ["LAX", "LAX"], ["LAX", "YVR"], ["YVR", "JFK"]]) == ['JFK', 'YVR', 'LAX', 'LAX', 'LAX', 'YVR', 'YVR', 'YVR', 'JFK']

if __name__ == '__main__':
    main()
```

### May 16, 2019 \[Easy\] Minimum Lecture Rooms
---
> **Questions:** Given an array of time intervals (start, end) for classroom lectures (possibly overlapping), find the minimum number of rooms required.
>
> For example, given [(30, 75), (0, 50), (60, 150)], you should return 2.

**My thoughts:** In order to solve this problem, do NOT worry about **minimum** at first. Think about how many rooms we might need by testing different examples:

Exreme Case
1. If there are 10 intervals that all of them are overlapping w/ each other, then we need 10 rooms
   at peak hours
2. If there are 10 intervals that none of them are overlapping w/ each other, then 1 room is sufficient to hold all 10 intervals one by one

Normal Case
1. If we just have 1 interval, if we need 1 room
   - If we have 2 intervals that might overlapping at some point, then we need to add 1 more room, which gives 2 in total. eg. (1, 10) and (5, 15)
   - If we have 3 intervals that 
     - all of them overlapping at some point, then we need to add 2 more rooms, which gives 3 in total eg. (1, 10), (5, 15) and (6, 15)
     - 2 of them overlapping at some point, then we only ned to add 1 more rooms, which gives 2 in total eg. (1, 10), (5, 15) and (10, 15)

For `(1, 10), (5, 15) and (6, 15)` we can use the folling time table to find the pattern

|t|                0| 1| 2| 3| 4| 5| 6| 7| 8| 9| 10| 11| 12| 13| 14| 15 | 16 |
|:-|               -:|-:|-:|-:|-:|-:|-:|-:|-:|-:| -:| -:| -:| -:| -:|  -:|  -:|
|total room at t | 0| 1| 1| 1| 1| 2| 3| 3| 3| 3|  2|  2|  2|  2|  2|   0|   0|
|difference      | 0|+1| 0| 0| 0|+1|+1| 0| 0| 0| -1|  0|  0|  0|  0|  -2|   0|

> Note: Did you notice whenever we enter an interval at time t, the total number of room at time t `+1` and whenever we leave an interval the total number `-1`? e.g. `+1` at `t = 1, 5, 6` and `-1` at `t=10, 15, 15`. And at peak hour, the total room equals 3. 

**Counting Sort Python Solution:** [https://repl.it/@trsong/Minimum-Lecture-Rooms](https://repl.it/@trsong/Minimum-Lecture-Rooms)
```py
def min_lec_room(intervals):
    min_start = 0
    max_end = 0
    for elem in intervals:
        min_start = min(min_start, elem[0])
        max_end = max(max_end, elem[1])
    
    # To save space, we only start time at min_start instead of time = 0
    room_at = [0] * (max_end - min_start + 1)
    for elem in intervals:
        room_at[elem[0] - min_start] += 1
        room_at[elem[1] - min_start] -= 1

    max_accumulated_room = 0
    accumulated_room = 0
    for t in range(max_end - min_start + 1):
        accumulated_room += room_at[t]
        max_accumulated_room = max(max_accumulated_room, accumulated_room)
    return max_accumulated_room

def main():
    t1 = (-10, 0)
    t2 = (-5, 5)
    t3 = (0, 10)
    t4 = (5, 15)
    assert min_lec_room([t1, t1, t1]) == 3
    assert min_lec_room([(30, 75), (0, 50), (60, 150)]) == 2
    assert min_lec_room([t3, t1]) == 1
    assert min_lec_room([t1, t3, t2, t4]) == 2
    assert min_lec_room([t4, t3, t2, t1, t1, t2, t3, t4]) == 4

if __name__ == '__main__':
    main()
```


> Note: in order to save space, can we just calculate at either start time or end time?

For `(1, 10), (5, 15) and (6, 15)`, in order to save space, we just consider end point

|t|                1| 5| 6| 10| 15|
|:-|              -:|-:|-:| -:| -:|
|total room at t | 1| 2| 3|  2|  0|
|difference      |+1|+1|+1| -1| -2|

**Python Solution:** [https://repl.it/@trsong/Minimum-Lecture-Rooms2](https://repl.it/@trsong/Minimum-Lecture-Rooms2)
```py
def min_lec_room(intervals):
    begins = [(e[0], 1) for e in intervals]
    ends = [(e[1], -1) for e in intervals]
    all_points = begins + ends
    sorted_all_points = sorted(all_points, key=lambda x: x[0])

    max_accumulated_room = 0
    accumulated_room = 0
    i = 0
    while i < len(sorted_all_points):
        accu = sorted_all_points[i][1]
        # Combine all the rooms for the same time
        while i + 1 < len(sorted_all_points) and sorted_all_points[i+1][0] == sorted_all_points[i][0]:
            accu += sorted_all_points[i+1][1]
            i += 1
        accumulated_room += accu
        max_accumulated_room = max(max_accumulated_room, accumulated_room)
        i += 1
    return max_accumulated_room

def main():
    t1 = (-10, 0)
    t2 = (-5, 5)
    t3 = (0, 10)
    t4 = (5, 15)
    assert min_lec_room([t1, t1, t1]) == 3
    assert min_lec_room([(30, 75), (0, 50), (60, 150)]) == 2
    assert min_lec_room([t3, t1]) == 1
    assert min_lec_room([t1, t3, t2, t4]) == 2
    assert min_lec_room([t4, t3, t2, t1, t1, t2, t3, t4]) == 4

if __name__ == '__main__':
    main()
```



### May 15, 2019 \[Medium\] Tokenization
---
> **Questions:** Given a dictionary of words and a string made up of those words (no spaces), return the original sentence in a list. If there is more than one possible reconstruction, return any of them. If there is no possible reconstruction, then return null.
>
> For example, given the set of words 'quick', 'brown', 'the', 'fox', and the string "thequickbrownfox", you should return ['the', 'quick', 'brown', 'fox'].
>
> Given the set of words ['bed', 'bath', 'bedbath', 'and', 'beyond'], and the string "bedbathandbeyond", return either ['bed', 'bath', 'and', 'beyond] or ['bedbath', 'and', 'beyond'].

**My thoughts:** Divide and Conquer 
Break the word into smaller chunks: prefix and suffix. Process either of them and doing recursion on the other one.

Suppose `process(word) = [word] if word in dictionary`

Then `tokenization("thequickbrownfox") =` Any of the following result
```py
tokenization("thequickbrownfox") + process("") => None
tokenization("thequickbrownfo") + process("x") => None
tokenization("thequickbrownf") + process("ox") => None
tokenization("thequickbrown") + process("fox") => tokenization("thequickbrown") + ["fox"]
tokenization("thequickbrow") + process("nfox") => None
tokenization("thequickbro") + process("wnfox") => None
tokenization("thequickbr") + process("ownfox") => None
tokenization("thequickb") + process("rownfox") => None
tokenization("thequick") + process("brownfox") => None
tokenization("thequic") + process("kbrownfox") => None
tokenization("thequ") + process("ickbrownfox") => None
tokenization("theq") + process("uickbrownfox") => None
tokenization("the") + process("quickbrownfox") => None
tokenization("th") + process("equickbrownfox") => None
tokenization("t") + process("hequickbrownfox") => None
tokenization("") + process("thequickbrownfox") => None
```

**Python Solution:** [https://repl.it/@trsong/Tokenization](https://repl.it/@trsong/Tokenization)

```py
def tokenization_recur(word, dictionary):
    if not word:
        return []
    for i in xrange(len(word)):
        suffix = word[i:]
        processed_suffix = [suffix] if not suffix or suffix in dictionary else None
        if not processed_suffix: continue

        processed_prefix = tokenization_recur(word[:i], dictionary)
        if processed_prefix is not None:
            return processed_prefix + processed_suffix
    return None

def tokenization(word, dictionary):
    if not word or not dictionary:
        return None
    else:
        res = tokenization_recur(word, set(dictionary))
        return ' '.join(res) if res else None

def main():
    assert tokenization("thequickbrownfox", ['the', 'quick', 'brown', 'fox']) == "the quick brown fox"
    assert tokenization("bedbathandbeyond", ['bed', 'bath', 'bedbath', 'and', 'beyond']) == 'bedbath and beyond'
    assert tokenization("thequickbrownfox", ['thequickbrownfox']) == "thequickbrownfox"
    assert tokenization("thefox", ['thequickbrownfox']) is None
    d1 = ['i', 'and', 'like', 'sam', 'sung', 'samsung', 'mobile', 'ice', 'cream', 'icecream', 'man', 'go', 'mango']
    assert tokenization("ilikesamsungmobile", d1) == "i like samsung mobile"
    assert tokenization("ilikeicecreamandmango", d1) == "i like icecream and mango"

if __name__ == '__main__':
    main()
```

> Note that in `tokenization("fox")` will end up w/ `tokenization("o")` and `tokenization("brown")` will also end up w/ `tokenization("o")`. Can we do some optimization there?

**Python Solution:** [https://repl.it/@trsong/Tokenization-Optimziation](https://repl.it/@trsong/Tokenization-Optimziation)

```py
def tokenization_recur_with_cache(word, dictionary, cache):
    if word not in cache:
        cache[word] = tokenization_recur(word, dictionary, cache)
    return cache[word]

def tokenization_recur(word, dictionary, cache):
    if not word:
        return []
    for i in xrange(len(word)):
        suffix = word[i:]
        processed_suffix = [suffix] if not suffix or suffix in dictionary else None
        if not processed_suffix: continue

        processed_prefix = tokenization_recur_with_cache(word[:i], dictionary, cache)
        if processed_prefix is not None:
            return processed_prefix + processed_suffix
    return None

def tokenization(word, dictionary):
    if not word or not dictionary:
        return None
    else:
        res = tokenization_recur_with_cache(word, set(dictionary), {})
        return ' '.join(res) if res else None
```

### May 14, 2019 \[Medium\] Overlapping Rectangles
---
> **Questions:** You're given 2 over-lapping rectangles on a plane. For each rectangle, you're given its bottom-left and top-right points. How would you find the area of their overlap?

**My thoughts:** If you haven't seen this question before, don't worry. You probably should have seen the 1D version of this question. i.e. Find the overlapping length for two intervals.
Now think about 2D version of that question, in order to solve the overlapping region, we can project the shape of the graph onto x-axis and y-axis to get x-intersection and y-intersection. The overlapping ara is just product of those two intersections.

**Python Solution:** [https://repl.it/@trsong/Overlapping-Rectangles](https://repl.it/@trsong/Overlapping-Rectangles)
```py
 # -*- coding: utf-8 -*
class Point(object):
    @staticmethod
    def x_projection(p1, p2):
        return Interval(p1.x, p2.x)
    
    @staticmethod
    def y_projection(p1, p2):
        return Interval(p1.y, p2.y)

    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

class Interval(object):
    @staticmethod
    def intersection(interval1, interval2):
        if interval1.end <= interval2.begin or interval2.end <= interval1.begin:
            return 0
        else:
            return min(interval1.end, interval2.end) - max(interval1.begin, interval2.begin)

    def __init__(self, begin, end):
        self.begin = min(begin, end)
        self.end = max(begin, end)

class Rectangle(object):
    @staticmethod
    def overlapping(r1, r2):
        x_projection1 = r1.get_x_projection()
        x_projection2 = r2.get_x_projection()
        y_projection1 = r1.get_y_projection()
        y_projection2 = r2.get_y_projection()

        x_intersection = Interval.intersection(x_projection1, x_projection2)
        y_intersection = Interval.intersection(y_projection1, y_projection2)
        return x_intersection * y_intersection

    def __init__(self, bottom_left, top_right):
        self.bottom_left = bottom_left
        self.top_right = top_right

    def get_x_projection(self):
        return Point.x_projection(self.bottom_left, self.top_right)

    def get_y_projection(self):
        return Point.y_projection(self.bottom_left, self.top_right)

    def __eq__(self, other):
        return self.bottom_left == other.bottom_left and self.top_right == other.top_right
```

```
 Test strategy: use the following graph to cover all different edge cases
 ┌───┬───┐ 2,2
 │ ┌─┼─┐ │
 ├─┼─┼─┼─┤
 │ └─┼─┘ │
 └───┴───┘
 -2,-2
```

```py
def main():
    tl = Rectangle(Point(-2,0), Point(0,2))
    tr = Rectangle(Point(0,0), Point(2,2))
    bl = Rectangle(Point(-2,-2), Point(0,0))
    br = Rectangle(Point(0,-2), Point(2,0))
    inner = Rectangle(Point(-1,-1), Point(1,1))
    outer = Rectangle(Point(-2,-2), Point(2,2))
    lst = [tl, tr, bl, br]
    origin = Rectangle(Point(0, 0), Point(0, 0))

    # Verify all combinations with no overlapping
    assert Rectangle.overlapping(origin, origin) == 0
    for r in lst + [inner, outer]:
        assert Rectangle.overlapping(origin, r) == 0
        assert Rectangle.overlapping(r, origin) == 0
    for r1 in lst:
        for r2 in lst:
            if r1 != r2:
                assert Rectangle.overlapping(r1, r2) == 0
    
    # Verify all combinations with overlapping = 1
    for r in lst:
        assert Rectangle.overlapping(r, inner) == 1
        assert Rectangle.overlapping(inner, r) == 1

    # Verify all combinations with overlapping = 4
    assert Rectangle.overlapping(inner, outer) == 4
    assert Rectangle.overlapping(outer, inner) == 4
    for r in lst:
        assert Rectangle.overlapping(r, outer) == 4
        assert Rectangle.overlapping(outer, r) == 4
    for e in lst + [inner]:
        assert Rectangle.overlapping(e, e) == 4
    
    # Verify all combinations with overlapping = 16
    assert Rectangle.overlapping(outer, outer) == 16

if __name__ == '__main__':
    main()
```

### May 13, 2019 \[Medium\] Craft Palindrome
---
> **Question:** Given a string, find the palindrome that can be made by inserting the fewest number of characters as possible anywhere in the word. If there is more than one palindrome of minimum length that can be made, return the lexicographically earliest one (the first one alphabetically).
>
> For example, given the string "race", you should return "ecarace", since we can add three letters to it (which is the smallest amount to make a palindrome). There are seven other palindromes that can be made from "race" by adding three letters, but "ecarace" comes first alphabetically.
>
> As another example, given the string "google", you should return "elgoogle".

**My thoughts:** There are two ways to solving this problem. Brute force way VS Dynamic Programming + Tracking Backwards. I was too lazy to post the brute force solution but the idea is to think about each possible piovt positions: there are `2n - 1` of them. And find the smallest   among all posible incertions.

e.g. For input = "abcde", all position of pivot are marked:
```py
| a | b | c | d | e
0 1 2 3 4 5 6 7 8 9
```
Suppose we choose 1 as pivot, we will need to insert all char in parentheses like the following:
```py
(edcb)abcde
```

Suppose we choose 2 as pivot, we will need to insert all char in parentheses like the following:
```py
a(cde)bcde(a)
```

But the brute force solution is NOT efficient in terms of both space and time. As the seach space is huge and a lot of cases are duplicated.

**Dynamic Programming + Tracking Backwards Solution**

Before think about how to insert character, let's do that step by step. First consider what's the minimum number of solutions needed to form a palindrome. 

> **STEP 1:** Find Minimum Number of Insertion

Let `word[l:r]` represents substring of word from index `l` to index `r` inclusive. We can break the `find_min_insertion(word[l:r])` into:
- `find_min_insertion(word[l:r-1])`
- `find_min_insertion(word[l-1:r])`
- `find_min_insertion(word[l-1:r-1])` 

And we will have the following recursive formula:

If `word[l] = word[r]` which means the first and last character matches, then we solve the sub-problem: 

`find_min_insertion(word[l:r]) = find_min_insertion(word[l-1:r-1])`

else,  we will need to insert 1 charater either to front or back, depends on which sub-problem is smaller

`find_min_insertion(word[l:r]) = 1 + min(find_min_insertion(word[l-1:r]), find_min_insertion(word[l:r-1]))`

The following code snippet illustrates above recursion

```py
def find_min_insertion(word, left, right):
    if left == right:
        return 0
    elif left == right - 1:
        return 0 if word[left] == word[right] else 1
    elif word[left] == word[right]:
        return find_min_insertion(word, left + 1, right - 1)
    else:
        drop_left_res = find_min_insertion(word, left + 1, right)
        drop_right_res = find_min_insertion(word, left, right - 1)
        return 1 + min(drop_left_res, drop_right_res)
```

> **STEP 2:** Optimize Solution and Store Each Decision We Made

In the recusive solution mentioned above, our solution is not efficient in both time and space. Plus there is no way for us to know what decision we made to get the final value. 

So what about using DP to solve this problem?

Well it seems to be a good idea. We define `dp[l][r]` to be the solution for `find_min_insertion(word[l:r])`. Then we can use the following formula:

```py
if word[left] == word[right]:
    dp[left][right] = dp[left+1][right-1]
else:
    dp[left][right] = 1 + min(dp[left+1][right], dp[left][right-1])
```

But another problem pops up: how do we interate through the 2D array? As `dp[left][right] = dp[left+1][right-1]`. We cannot do left to right as index `l` depend on `l+1`.

A good idea to tackle this problem is the think about how does the recursion tree looks like for previous recursive solution. 

e.g. What does recursion looks like for `word[0,3]`
```        
                  f(0, 3)
            /        |       \
           /         |        \
       f(0, 2) f    (1, 2)   f(1, 3)
   /      |      \         /      |      \            
f(0, 1) f(1, 1) f(1, 2)  f(1, 2) f(2, 2) f(2, 3)             
```

Note that the base case (leaf positon) is either  `l = r` or `l = r - 1`. We define `gap = r - l`

Then we find that `gap = 2` depends on `gap = 1`. And `gap = 3` depends on `gap = 2`. So that means we have to solve `gap=1 gap=2 gap=3 ... gap=n-2` one-by-one.

Ok, now let's go back to the issue regarding how we iterate through the dp array. 

When `gap = 1`, we have
```py
dp[0][1], dp[1][2], dp[2][3], dp[3][4]
```

When `gap = 2`, we have
```py
dp[0][2], dp[1][3], dp[2][4]
```

When `gap = 3`, we have
```py
dp[0][3], dp[1][4]
```

When `gap = 4`, we have
```py
dp[0][4]
```

So you find that `r` ranges from `gap` to `n-1` and `l = r - gap`. And here is the DP way to solve this problem:

```py
def find_min_insertion_DP(word):
    n = len(word)
    dp = [[0 for r in xrange(n)] for c in xrange(n)]

    for gap in xrange(1, n):
        for right in xrange(gap, n):
            left = right - gap
            if word[left] == word[right]:
                dp[left][right] = dp[left+1][right-1]
            else:
                dp[left][right] = 1 + min(dp[left+1][right], dp[left][right-1])
    
    return dp[0][n-1]
```

> **STEP 3:** Tracking Backwards Using the DP Decision Array

Right now we know what is the minimum insertion we have to take. Now we can go backwards and backtracking the decision we made to get such result. 

So first of all, let's take a look at the recursive relation:

```py
if word[left] == word[right]:
    dp[left][right] = dp[left+1][right-1]
else:
    dp[left][right] = 1 + min(dp[left+1][right], dp[left][right-1])
```

So if it's the first case, the we down and left. And if it's the second case, base on which one is smaller we can determine if we insert `word[right]` or insert `word[left]`. And remember the question ask to return the **lexicographically earliest** one, thus if there is a tie between
`dp[left+1][right]` and `dp[left][right-1]`. We will use the lexicographical order of `word[left]` and `word[right]` to determine who wins. 

You might ask what if we have a tie again between `word[left]` and `word[right]`. Then the answer is, we will fall into the first case, `dp[left][right] = dp[left+1][right-1]`.

**Python Solution:** [https://repl.it/@trsong/Craft-Palindrome](https://repl.it/@trsong/Craft-Palindrome)
```py
def find_min_insertion_DP_helper(word):
    n = len(word)
    dp = [[0 for r in xrange(n)] for c in xrange(n)]

    for gap in xrange(1, n):
        for right in xrange(gap, n):
            left = right - gap
            if word[left] == word[right]:
                dp[left][right] = dp[left+1][right-1]
            else:
                dp[left][right] = 1 + min(dp[left+1][right], dp[left][right-1])
    return dp

def craft_palindrome(word):
    if not word: return ""
    n = len(word)
    dp = find_min_insertion_DP_helper(word)
    min_insertion = dp[0][n-1]
    res = [None] * (n + min_insertion)
    left, right = 0, n - 1
    updated_left, updated_right = 0, len(res) - 1
    
    while left <= right:
        chosen = None
        if word[left] == word[right]:
            chosen = word[left]
            left += 1
            right -= 1
        else: 
            drop_left_res = dp[left+1][right]
            drop_right_res = dp[left][right-1]
            if drop_left_res < drop_right_res or drop_left_res == drop_right_res and word[left] < word[right]:
                chosen = word[left]
                left += 1
            else:
                # Either drop_left_res > drop_right_res; Or drop_left_res == drop_right_res and word[left] > word[right]
                chosen = word[right]
                right -= 1

        res[updated_left] = chosen
        res[updated_right] = chosen
        updated_left += 1
        updated_right -= 1
    
    return ''.join(res)

def main():
    assert craft_palindrome("") == ""
    assert craft_palindrome("a") == "a"
    assert craft_palindrome("aa") == "aa"
    assert craft_palindrome("race") == "ecarace"
    assert craft_palindrome("abcd") == "abcdcba"
    assert craft_palindrome("abcda") == "abcdcba"
    assert craft_palindrome("abcde") == "abcdedcba"
    assert craft_palindrome("google") == "elgoogle"

if __name__ == "__main__":
    main()
```


### May 12, 2019 \[Medium\] Inversion Pairs
---
> **Question:**  We can determine how "out of order" an array A is by counting the number of inversions it has. Two elements `A[i]` and `A[j]` form an inversion if `A[i] > A[j]` but `i < j`. That is, a smaller element appears after a larger element. Given an array, count the number of inversions it has. Do this faster than `O(N^2)` time. You may assume each element in the array is distinct.
>
> For example, a sorted list has zero inversions. The array `[2, 4, 1, 3, 5]` has three inversions: `(2, 1)`, `(4, 1)`, and `(4, 3)`. The array `[5, 4, 3, 2, 1]` has ten inversions: every distinct pair forms an inversion.

**Python Trivial Solution:** 

```py
def count_inversion_pairs_naive(nums):
    inversions = 0
    n = len(nums)
    for i in xrange(n):
        for j in xrange(i+1, n):
            inversions += 1 if nums[i] > nums[j] else 0
    return inversions
```

**My thoughts:** We can start from trivial solution and perform optimization after. In trivial solution basically, we try to count the total number of larger number on the left of smaller number. However, while solving this question, you need to ask yourself, is it necessary to iterate through all combination of different pairs and calculate result?

e.g. For input `[5, 4, 3, 2, 1]` each pair is an inversion pair, once I know `(5, 4)` will be a pair, then do I need to go over the remain `(5, 3)`, `(5, 2)`, `(5, 1)` as 3,2,1 are all less than 4? 

So there probably exits some tricks to allow us save some effort to not go over all possible combinations. 

**Solution 1**: Divide and Conquer

Did you notice the following properties? 

1. `count_inversion_pairs([5, 4, 3, 2, 1]) = count_inversion_pairs([5, 4]) + count_inversion_pairs([3, 2, 1]) + inversion_pairs_between([5, 4], [3, 2, 1])`
2. `inversion_pairs_between([5, 4], [3, 2, 1]) = inversion_pairs_between(sorted([5, 4]), sorted([3, 2, 1])) = inversion_pairs_between([4, 5], [1, 2, 3])`

This is bascially modified version of merge sort. Consider we break `[5, 4, 3, 2, 1]` into two almost equal parts: `[5, 4]` and `[3, 2, 1]`. Notice such break won't affect inversion pairs, as whatever on the left remains on the left. However, inversion pairs between `[5, 4]` and `[3, 2, 1]` can be hard to count without doing it one-by-one. 

If only we could sort them separately as sort won't affect the inversion order between two lists. i.e. `[4, 5]` and `[1, 2, 3]`. Now let's see if we can find the pattern, if `4 < 1`, then `5 should < 1`. And we also have `4 < 2` and `4 < 3`. We can simply skip all `elem > than 4` on each iteration, i.e. we just need to calculate how many elem > 4 on each iteration. This gives us **property 2**.

And we can futher break `[5, 4]` into `[5]` and `[4]` recursively. This gives us **property 1**.

Combine property 1 and 2 gives us the modified version of ***Merge-Sort***.


**Python Solution1:** [https://repl.it/@trsong/Inversion-Pairs](https://repl.it/@trsong/Inversion-Pairs)

```py
def merge(arr, begin, middle, end):
    p1 = begin
    p2 = middle + 1
    mid = middle
    inversions = 0
    if p2 <= end and arr[mid] <= arr[p2]: return 0
    while p1 <= mid and p2 <= end:
        if arr[p1] <= arr[p2]:
            p1 += 1
        else:
            inversions += mid - p1 + 1
                
            # shift value by 1 to make room to merge p2 value
            value = arr[p2]
            for i in xrange(p2, p1, -1):
                arr[i] = arr[i-1]
            arr[p1] = value

            p1 += 1
            p2 += 1
            mid += 1
    return inversions

def merge_sort(arr, begin, end):
    if begin >= end: return 0
    mid = begin + (end - begin) / 2
    left_sub_inversions = merge_sort(arr, begin, mid)
    right_sub_inversions = merge_sort(arr, mid + 1, end)
    current_inversions = merge(arr, begin, mid, end)
    return left_sub_inversions + right_sub_inversions + current_inversions

def count_inversion_pairs(nums):
    return merge_sort(nums, 0, len(nums) - 1)

def main():
    # For list in descending order, each pair forms an inversion pair, so we have n(n-1)/2 = 5 * 4 / 2 = 10 
    assert count_inversion_pairs([5, 4, 3, 2, 1]) == 10
    assert count_inversion_pairs([2, 4, 1, 3, 5]) == 3
    assert count_inversion_pairs([1, 2, 3]) == 0
    assert count_inversion_pairs([1, 1, 1, 1]) == 0
    assert count_inversion_pairs([1]) == 0
    assert count_inversion_pairs([]) == 0

if __name__ == '__main__':
    main()
```

**Solution 2**: Counting number of smaller elements efficiently

In the trivial solution above, we scan through all different combination of elements. However, is there any way we can sort of **cache** the previous counting result. i.e. Going backwards from the list and keep track of number of smaller element on the left. 

e.g. Process from right to left of `[1, 2, 3, 4, 5]` 
1. Process 5, no smaller elem found
2. Process 4, no smaller elem found
3. Process 3, no smaller elem found
4. Process 2, no smaller elem found
5. Process 1, no smaller elem found
Total Number of inversions: 0

e.g. Process from right to left of `[5, 4, 3, 2, 1]`
1. Process 1, no smaller elem found
2. Process 2, found 1 candidate which is 1, possible inversion (2, 1)
3. Process 3, found 2 candidates: 1 and 2. possible inversion (3, 2), (3, 1)
4. Process 4, found 3 candidates: 1, 2 and 3. possible inversion (4, 3), (4, 2), (4, 1)
5. Process 5, found 4 candidates: 1, 2, 3, 4. possible inversion (5, 4), (5, 3), (5, 2), (5,1)
Total Number of inversions: 0 + 1 + 2 + 3 + 4 = 10
 
Notice that we don't need to keep track of exact smaller number, just storing those numbers are sufficient. And we happen to have a powerful data structure that by giving a value, it can quickly returns number of smaller elements. Such data structure is called ***Binary Indexed Tree (BIT)*** and it can do well in range queries. i.e. calculate range sum.

How? 

Think about an array A where `A[i]` represents number of elem smaller than `i`. Then counting the number of smaller-than-target elem is sum of `A[0]`, `A[1]`, ...,  `A[target-1]`. And BIT can do well in range queries. 

**Python Solution2:** [https://repl.it/@trsong/Inversion-Pairs2](https://repl.it/@trsong/Inversion-Pairs2)

```py
class PrefixSum(object):
    """PrefixSum provide an efficient way to update and calculate range sum on the fly"""

    @staticmethod
    def last_bit(num):
        return num & -num

    def __init__(self, size):
        self._BITree = [0] * (size + 1)

    def get_sum(self, i):
        """Get sum of value from index 0 to i """
        # BITree starts from index 1
        index = i + 1
        res = 0
        while index > 0:
            res += self._BITree[index]
            index -= PrefixSum.last_bit(index)
        return res

    def update(self, i, delta):
        """Update the sum by add delta on top of result"""
        # BITree starts from index 1
        index = i + 1
        while index < len(self._BITree):
            self._BITree[index] += delta
            index += PrefixSum.last_bit(index)

def count_inversion_pairs2(nums):
    if not nums: return 0
    max_value = max(nums)
    # Store number of elem less than current value in prefix_sum
    prefix_sum = PrefixSum(max_value)
    inversions = 0

    for i in xrange(len(nums) - 1, -1, -1):
        # We count inversions backwards. For each elem, we use prefix sum to calculate number of elem less than current value
        current_value = nums[i]
        inversions += prefix_sum.get_sum(current_value - 1)
        prefix_sum.update(current_value, 1)
    return inversions
```

### May 11, 2019 LC 42 \[Hard\] Trapping Rain Water
---
> **Question:** Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
 
![rainwatertrap](https://assets.leetcode.com/uploads/2018/10/22/rainwatertrap.png)

> The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
> 
> Example:
> 
> Input: [0,1,0,2,1,0,1,3,2,1,2,1]
> 
> Output: 6

**My thoughts:** This question can be solved w/ pre-record left & right boundary as well as with 2 pointers. Both solution take `O(n)` time. However, first solution requires more memory than second one.

**Solution 1:** Pre-record Left & Right Boundary

Total water accumulation equals sum of each position's accumulation. For any position at index `i`, the reason why current position has water accumulation is due to the fact that there exist `l < i` such that `water_height[l] > water_height[i]` as well as `i < r` such that `water_height[r] > water_height[i]`. 

We probably don't care about what l, r are. However, we do care about how much each position can accumulate at maximum. `min(left_boundary[i], right_boundary[i]) - water_height[i]`. Where left_boundary represents max height on the left of i and right_boundary represents max height on the right of i.

Use the following as example: 

![rainwatertrap](https://assets.leetcode.com/uploads/2018/10/22/rainwatertrap.png)

| index                     | 0   | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   | 10  | 11  |
| :------------------------ | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| water_height              | 0   | 1   | 0   | 2   | 1   | 0   | 1   | 3   | 2   | 1   | 2   | 1   |
| left_boundary             | 0   | 0   | 1   | 2   | 2   | 2   | 2   | 2   | 3   | 3   | 3   | 3   |
| right_boundary            | 3   | 3   | 3   | 3   | 3   | 3   | 3   | 2   | 2   | 2   | 1   | 0   |
| accumulation(blue reigon) | 0   | 0   | 1   | 0   | 1   | 2   | 1   | 0   | 0   | 1   | 0   | 0   |

**Python Solution:** [https://repl.it/@trsong/trappingRainWater](https://repl.it/@trsong/trappingRainWater)
```py
def trapping_rain_water_solver(water_height):
    n = len(water_height)
    left_boundary = [0] * n
    right_boundary = [0] * n
    left = 0
    right = 0
    for i in xrange(n):
        left_boundary[i] = left
        right_boundary[n-1-i] = right
        left = max(left, water_height[i])
        right = max(right, water_height[n-i-1])
    sum = 0
    for i in xrange(n):
        # How much water can be accumulated at index i is depend on max boundary height on the left and max boundary on the right.
        sum += max(0, min(left_boundary[i], right_boundary[i]) - water_height[i])
    return sum

def main():
    assert trapping_rain_water_solver([0,1,0,2,1,0,1,3,2,1,2,1]) == 6
    assert trapping_rain_water_solver([1, 1, 1, 2]) == 0
    assert trapping_rain_water_solver([1, 1, 1, 1]) == 0
    assert trapping_rain_water_solver([2, 1, 1, 1]) == 0
    assert trapping_rain_water_solver([]) == 0
    assert trapping_rain_water_solver([1, 1, 2, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 2, 1]) == 8
    assert trapping_rain_water_solver([3, 2, 1, 2, 3]) == 4

if __name__ == '__main__':
    main()
```

**Solution 2:** 2 pointers

Maintain two pointers and always move pointer w/ smaller height to the other one. How this works is because if left pointer is smaller: `water_height[left] < water_height[right]`, then there must be an index k between left and right represents local right boundary such that its value `water_height[left] < water_height[k] <= water_height[right]`. Note that heights between left and k is either increasing or decresing. We can calculate water accumulation between left and k. And vice versa when right pointer is smaller.

Use the following as example: 

![rainwatertrap](https://assets.leetcode.com/uploads/2018/10/22/rainwatertrap.png)

| Iteration |     |     |     |     |     |     |     |     |     |     |     |     | Direction | Accumulation  |
| :-------- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-------: | :------------ |
| 1         | 0   |     |     |     |     |     |     |     |     |     |     | 1   | start     | 0             |
| 2         | 0   | 1   |     |     |     |     |     |     |     |     |     | 1   | ->        | 0             |
| 3         | 0   | 1   | 0   | 2   |     |     |     |     |     |     |     | 1   | ->        | 1             |
| 4         | 0   | 1   | 0   | 2   |     |     |     |     |     |     | 2   | 1   | ->        | 0             |
| 5         | 0   | 1   | 0   | 2   | 1   | 0   | 1   | 3   |     |     | 2   | 1   | ->        | 1 + 2 + 1 = 4 |
| 6         | 0   | 1   | 0   | 2   | 1   | 0   | 1   | 3   | 2   | 1   | 2   | 1   | <-        | 1             |

total: 1 + 4 + 1 = 6

```py
def trapping_rain_water_solver2(water_height):
    lo = 0
    hi = len(water_height) - 1
    sum = 0
    while lo < hi:
        left_boundary = water_height[lo]
        right_boundary = water_height[hi]
        if left_boundary < right_boundary:
            # If left_boundary < right_boundary, there exists a local_right_bounary while we are moving from left to right
            # such that left_boundary < local_right_bounary <= right_boundary
            while lo < hi and left_boundary >= water_height[lo]:
                # Accumulate water between left_boundary and local_right_bounary
                sum += left_boundary - water_height[lo]
                lo += 1
        else:
            # If left_boundary >= right_boundary, there exists a local_left_boundary while moveing from right to left
            # such that left_boundary <= local_left_bounary < right_boundary
            while lo < hi and right_boundary >= water_height[hi]:
                # Accumulate water between right_boundary and local_left_bounary
                sum += right_boundary - water_height[hi]
                hi -= 1
    return sum
```

### May 10, 2019 \[Hard\] Execlusive Product
---
> **Question:**  Given an array of integers, return a new array such that each element at index i of the new array is the product of all the numbers in the original array except the one at i.
>
> For example, if our input was [1, 2, 3, 4, 5], the expected output would be [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output would be [2, 3, 6].
>
> Follow-up: what if you can't use division?

**My thoughts:** For an input array `[a[0], a[1], a[2], a[3], a[4]]`, what will be the value for output at index i?

It should be `res[i] = a[0] * a[1] * ... a[i-1] * a[i+1] * ... * a[n-1]`, which basically equals `res[i] = (a[0] * a[1] * ... * a[i-1]) * (a[i+1] * ... * a[n-1])`. We can use one array to store all left products and another array for right products. 

eg. `left_execlusive_product[i] = a[0] * a[1] * ... * a[i-1]` and `right_execlusive_product[i] = a[i+1] * ... * a[n-1]`. Thus `res[i] = left_execlusive_product[i] * right_execlusive_product[i]`

Note that `left_execlusive_product[i] = left_execlusive_product[i-1] * a[i-1]` so that we can calculate left_execlusive_product quite easily. Which take `O(N)` time. Similarly, `right_execlusive_product` can also calculate within `O(N)` time.

**Python Solution:** [https://repl.it/@trsong/execlusiveproduct](https://repl.it/@trsong/execlusiveproduct)
```py
def execlusive_product(nums):
    if not nums: return []
    n = len(nums)
    left_execlusive_product = [0] * n
    right_execlusive_product = [0] * n
    
    left = 1
    for i in xrange(n):
        left_execlusive_product[i] = left
        left *= nums[i]

    right = 1
    for j in xrange(n - 1, -1, -1):
        right_execlusive_product[j] = right
        right *= nums[j]

    return [left_execlusive_product[i] * right_execlusive_product[i] for i in xrange(n)]

def main():
    assert execlusive_product([1, 2, 3, 4, 5]) == [120, 60, 40, 30, 24]
    assert execlusive_product([]) == []
    assert execlusive_product([2]) == [1]
    assert execlusive_product([3, 2]) == [2, 3]

if __name__ == '__main__':
    main()
```
**Note:** We can further optimize the code by combining two loops into one.
```py
def execlusive_product2(nums):
    # one pass optimization
    if not nums: return []
    n = len(nums)
    left = 1
    right = 1
    res = [1] * n
    for i in xrange(n):
        res[i] *= left
        res[n-1-i] *= right
        left *= nums[i]
        right *= nums[n-1-i]
    return res
```


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

**My thoughts:** Use BFS from start to end. Each layer we proceed will contribute to distance by 1. e.g. If end is one of start's level-7 descendant then the distance will be 7. 

**Python Solution1:** [https://repl.it/@trsong/gridpath](https://repl.it/@trsong/gridpath)
```py
class Position(object):
    def __init__(self, row, col):
        self.row = row
        self.col = col
    def __repr__(self):
        return "(%d, %d)" % (self.row, self.col)

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

def grid_path(grid_wall, start, end):
    num_row = len(grid_wall)
    num_col = len(grid_wall[0])
    distance = 0
    queue = [start]
    visited = set()
    while queue:
        level_size = len(queue)
        for i in xrange(level_size):
            current = queue.pop(0)
            if current in visited: continue
            if current == end:
                return distance
            visited.add(current)

            top = Position(current.row - 1, current.col) if current.row > 0 else None
            if top and not grid_wall[top.row][top.col] and top not in visited:
                queue.append(top)

            left = Position(current.row, current.col - 1) if current.col > 0 else None
            if left and not grid_wall[left.row][left.col] and left not in visited:
                queue.append(left)
            
            bottom = Position(current.row + 1, current.col) if current.row < num_row - 1 else None
            if bottom and not grid_wall[bottom.row][bottom.col] and bottom not in visited:
                queue.append(bottom)
            
            right = Position(current.row, current.col + 1) if current.col < num_col - 1 else None
            if right and not grid_wall[right.row][right.col] and right not in visited:
                queue.append(right)
        distance += 1
    
    return None
    
def main():
    F = False
    T = True
    wall_map1 = [
        [F, F, F, F],
        [T, T, F, T],
        [F, F, F, F],
        [F, F, F, F]
    ]
    assert grid_path(wall_map1, Position(0, 0), Position(3, 0)) == 7

    wall_map2 = [
        [F, F, F, F],
        [T, T, T, T],
        [F, F, F, F],
        [F, F, F, F]
    ]
    assert grid_path(wall_map2, Position(0, 0), Position(3, 0)) is None

    wall_map3 = [
        [F, F, F, F, T, F, F, F],
        [T, T, T, F, T, F, T, T],
        [F, F, F, F, T, F, F, F],
        [F, T, T, T, T, T, T, F] ,
        [F, F, F, F, F, F, F, F]
    ]
    assert grid_path(wall_map3, Position(0, 0), Position(0, 7)) == 25

    wall_map4 = [[F, F]]
    assert grid_path(wall_map4, Position(0, 0), Position(0, 1)) == 1

if __name__ == '__main__':
    main()
```

**Note:** This question can also be solved use ***Bidirectional BFS*** (2-way BFS). 

**Python Solution2:** [https://repl.it/@trsong/gridpath2](https://repl.it/@trsong/gridpath2)

```py
def search(grid_wall, queue, visited, other_visited):
    """Search all elem in queue to find exist elem in other_visited. Add all neighbors to queue after."""
    num_row = len(grid_wall)
    num_col = len(grid_wall[0])
    level_size = len(queue)
    for i in xrange(level_size):
        current = queue.pop(0)
        if current in visited: continue
        if current in other_visited:
            return current
        visited.add(current)
        
        top = Position(current.row - 1, current.col) if current.row > 0 else None
        if top and not grid_wall[top.row][top.col] and top not in visited:
            queue.append(top)

        left = Position(current.row, current.col - 1) if current.col > 0 else None
        if left and not grid_wall[left.row][left.col] and left not in visited:
            queue.append(left)

        bottom = Position(current.row + 1, current.col) if current.row < num_row - 1 else None
        if bottom and not grid_wall[bottom.row][bottom.col] and bottom not in visited:
            queue.append(bottom)
            
        right = Position(current.row, current.col + 1) if current.col < num_col - 1 else None
        if right and not grid_wall[right.row][right.col] and right not in visited:
            queue.append(right)

    return None

def grid_path2(grid_wall, start, end):
    iteration = 0
    start_visited = set()
    end_visited = set()
    start_queue = [start]
    end_queue = [end]
    while start_queue and end_queue:
        # Forward search and add neighbors
        if search(grid_wall, start_queue, start_visited, end_visited):
            return 2 * iteration - 1

        # Backward search and add neighbors
        if search(grid_wall, end_queue, end_visited, start_visited):
            return 2 * iteration

        iteration += 1
    return None
```

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

**My thoughts:** There are mutiple ways to solve this question. Solution 1 use DFS and Solution 2 use Union-Find. 

**Solution 1:** for all 'O' cells connect to boundary of the grid, temporarily mark them as '-'. And then scan through entire grid, replace 'O' with 'X' and '-' with 'O'

1. Orginal Grid:
  ```py
  [
      ['X', 'O', 'X', 'X', 'O'],
      ['X', 'X', 'O', 'O', 'X'],
      ['X', 'O', 'X', 'X', 'O'],
      ['O', 'O', 'O', 'X', 'O']
  ]
  ```
2. Replace all 'O' connect to bounary as '-'
  ```py
  [
      ['X', '-', 'X', 'X', '-'],
      ['X', 'X', 'O', 'O', 'X'],
      ['X', '-', 'X', 'X', '-'],
      ['-', '-', '-', 'X', '-']
  ]
  ```
3. Replace 'O' with 'X' and '-' with 'O'
  ```py
  [
      ['X', 'O', 'X', 'X', 'O'],
      ['X', 'X', 'X', 'X', 'X'],
      ['X', 'O', 'X', 'X', 'O'],
      ['O', 'O', 'O', 'X', 'O']
  ]
  ```

**Python Solution 1:** [https://repl.it/@trsong/surroundedregionsdfs](https://repl.it/@trsong/surroundedregionsdfs)

```py
class SurroundedReigonSolver(object):
    def __init__(self, grid):
        self._grid = grid
        self._num_row = len(grid)
        self._num_col = len(grid[0])

    def _indexToPosition(self, row, col):
        return row * self._num_col + col

    def _positionToIndex(self, pos):
        return (pos / self._num_col, pos % self._num_col)

    def _dfs_search_replace(self, visited, source_pos, target):
        r, c = self._positionToIndex(source_pos)
        source = self._grid[r][c]
        stack = [source_pos]
        while stack:
            current = stack.pop()
            r, c = self._positionToIndex(current)
            # process current cell
            if current not in visited and self._grid[r][c] == source:
                self._grid[r][c] = target
                visited.add(current)
                # Check and add bottom to stack
                if r < self._num_row - 1:
                    stack.append(self._indexToPosition(r + 1, c))
                # Check and add right to stack
                if c < self._num_col - 1:
                    stack.append(self._indexToPosition(r, c + 1))
                # Check and add top to stack  
                if r > 0:
                    stack.append(self._indexToPosition(r - 1, c))
                # Check and add top left to stack     
                if c > 0:
                    stack.append(self._indexToPosition(r, c - 1))

    def solve(self):
        # Mark all edge-connected 'O' cells as '-'
        visited = set()
        for i in xrange(self._num_col):
            if self._grid[0][i] == 'O':
                self._dfs_search_replace(visited, self._indexToPosition(0, i), '-')
            if self._grid[self._num_row - 1][i] == 'O':
                self._dfs_search_replace(visited, self._indexToPosition(self._num_row - 1, i), '-')

        for j in xrange(self._num_row):
            if self._grid[j][0] == 'O':
                self._dfs_search_replace(visited, self._indexToPosition(j, 0), '-')
            if self._grid[j][self._num_col - 1] == 'O':
                self._dfs_search_replace(visited, self._indexToPosition(j, self._num_col - 1), '-')

        # Replace all 'O' with 'X' and '-' with 'O'
        for r in xrange(self._num_row):
            for c in xrange(self._num_col):
                if self._grid[r][c] == 'O':
                    self._grid[r][c] = 'X'
                elif self._grid[r][c] == '-':
                    self._grid[r][c] = 'O'
        
        return self._grid      

def main():
    grid1 = [
        ['X', 'X', 'X', 'X'],
        ['X', 'O', 'O', 'X'],
        ['X', 'X', 'O', 'X'],
        ['X', 'O', 'X', 'X']
    ]
    solved_grid1 = [
        ['X', 'X', 'X', 'X'],
        ['X', 'X', 'X', 'X'],
        ['X', 'X', 'X', 'X'],
        ['X', 'O', 'X', 'X']
    ]
    solver = SurroundedReigonSolver(grid1)
    assert solver.solve() == solved_grid1

    grid2 = [
        ['O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O']
    ]
    solved_grid2 = [
        ['O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O']
    ]
    solver2 = SurroundedReigonSolver(grid2)
    assert solver2.solve() == solved_grid2

    grid3 = [
        ['X', 'O', 'X', 'X', 'O'],
        ['X', 'X', 'O', 'O', 'X'],
        ['X', 'O', 'X', 'X', 'O'],
        ['O', 'O', 'O', 'X', 'O']
    ]
    solved_grid3 = [
        ['X', 'O', 'X', 'X', 'O'],
        ['X', 'X', 'X', 'X', 'X'],
        ['X', 'O', 'X', 'X', 'O'],
        ['O', 'O', 'O', 'X', 'O']
    ]
    solver3 = SurroundedReigonSolver(grid3)
    assert solver3.solve() == solved_grid3

if __name__ == '__main__':
    main()
```

**Solution 2:** There is a special data-structure called ***Union-Find*** that allow us to quickly determine if two cells are **CONNECTED** or not. 

The idea of ***Union-Find*** is as the following:

1. Suppose we have an array: `[0, 1, 2, 3, 4, 5]` represents 5 different cells. 

2. And we create a parent array which initialized to be `[-1, -1, -1, -1, -1, -1]`. That means all cell's parent is null and none of the cells are connected - they have no shared parent.

3. Now if we want to **connect** cell 0 and cell 5 by <u>marking cell 0's parent as 5</u> (we call this action **"Union"**).
Parent array will become `[5, -1, -1, -1, -1, -1]`. 

4. And if we also connect 3 and 0 so that all of 0,3,5 are connected.
we will have parent array looks like `[5, -1, -1, 0, -1, -1]`.

5. Now we defined `is_connected(index1, index2)` if they all share the same **root**. 
In our example 0,3,5 are connected and everything else is connected to itself separately.

6. <u>Since `parent[0] == 5` and `parent[3] == parent[0] == 5` and 5 has not parent indicates it is the root.</u> (We call the action of retrieving parent recursively: **"Find"**)

7. By keeping calling **union** and **find** on different cells will allow us to **connect** those cells. And quickly tell if two cells are **connected**.


The way we use Union-Find to conquer this question looks like the following:

* Step 1: Init the parent arry:
Orginal Grid:
  ```py
  [
      ['X', 'O', 'X', 'X', 'O'],
      ['X', 'X', 'O', 'O', 'X'],
      ['X', 'O', 'X', 'X', 'O'],
      ['O', 'O', 'O', 'X', 'O']
  ]
  ```

  parent array
  ```py
  [
      [-1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1]
      [-1]  <----------------- we also create a secret cell for future use
  ]
  ```
* Step 2: Now connect all connected 'O' use Union-Find
  
  parent array: instead of using parent index, I use letter to represent different connected region
  ```py
  [
      [-1,  A, -1, -1,  B],
      [-1, -1,  C,  C, -1],
      [-1,  E, -1, -1,  D],
      [ E,  E,  E, -1,  D]
      [-1]  <----------------- In the next step we will take advantage of this cell
  ]
  ```

* Step 3: Let's connect the secret cell to all edge-connected 'O' cells 

 parent array: instead of using parent index, I use letter to represent different connected region
  ```py
  [
      [-1,  O, -1, -1,  O],
      [-1, -1,  C,  C, -1],
      [-1,  O, -1, -1,  O],
      [ O,  O,  O, -1,  O]
      [ O]  <----------------- this cell is used to connect to all 'O' on the edge of the grid
  ]
  ```

* Step 4: Replace all 'O' with 'X', except for connected-to-secret-spot ones

  ```py
  [
      ['X', 'O', 'X', 'X', 'O'],
      ['X', 'X', 'X', 'X', 'X'],
      ['X', 'O', 'X', 'X', 'O'],
      ['O', 'O', 'O', 'X', 'O']
  ]
  ``` 
  
  Note for my implementation of union-find, I flatten the parent 2D array into 1D array.

**Python Solution 2:** [https://repl.it/@trsong/surroundedregionsunionfind](https://repl.it/@trsong/surroundedregionsunionfind)

```py
class SurroundedReigonSolver(object):
    def __init__(self, grid):
        self._grid = grid
        self._num_row = len(grid)
        self._num_col = len(grid[0])
        self._parent = [-1] * (self._num_row * self._num_col + 1)

    def _indexToPosition(self, row, col):
        return row * self._num_col + col

    def _find(self, pos):
        if self._parent[pos] < 0:
            return pos
        else:
            return self._find(self._parent[pos])

    def _union(self, pos1, pos2):
        parent1 = self._find(pos1)
        parent2 = self._find(pos2)
        if parent1 != parent2:
            self._parent[pos2] = pos1

    def _is_connected(self, pos1, pos2):
        return self._find(pos1) == self._find(pos2)

    def solve(self):
        # Scan through the grid to connect all 'O'
        for r in xrange(self._num_row):
            for c in xrange(self._num_col):
                current_pos = self._indexToPosition(r, c)
                # Note, as we are moving right and down, we only need to check top and left
                if self._grid[r][c] == 'O':
                    if r > 0 and self._grid[r-1][c] == 'O':
                        self._union(current_pos, self._indexToPosition(r-1, c))
                    if c > 0 and self._grid[r][c-1] == 'O':
                        self._union(current_pos, self._indexToPosition(r, c-1))

        # Connect all edge-connnected 'O' cell to this secret_pos
        secret_pos = self._num_row * self._num_col
        for i in xrange(self._num_col):
            if self._grid[0][i] == 'O':
                self._union(secret_pos, self._indexToPosition(0, i))
            if self._grid[self._num_row-1][i] == 'O':
                self._union(secret_pos, self._indexToPosition(self._num_row-1, i))
        for j in xrange(self._num_row):
            if self._grid[j][0] == 'O':
                self._union(secret_pos, self._indexToPosition(j, 0))
            if self._grid[j][self._num_col-1] == 'O':
                self._union(secret_pos, self._indexToPosition(j, self._num_col-1))
    
        for r in xrange(self._num_row):
            for c in xrange(self._num_col):
                # Only replace 'O' with 'X' unless it's connected to secret_pos
                if self._grid[r][c] == 'O' and not self._is_connected(secret_pos, self._indexToPosition(r, c)):
                    self._grid[r][c] = 'X'

        return self._grid
```

**Optimization for Solution 2:** Union-find can be optimized w/ : 

1. Find Optimization: Flatten the parent array on the fly
2. Union Optimization: Balance number of children for the same root

By doing the following optimization will allow us to have a faster `is_connected(pos1, pos2)` check.

```py
class SurroundedReigonSolver2(object):
    def __init__(self, grid):
        self._grid = grid
        self._num_row = len(grid)
        self._num_col = len(grid[0])
        self._parent = range(self._num_row * self._num_col + 1)
        self._parent_weight = [0] * len(self._parent)

    def _find(self, pos):
        if self._parent[pos] == pos:
            return pos
        else:
            # we flatten the path from current node to root
            root = self._find(self._parent[pos])
            self._parent[pos] = root
            return root

    def _union(self, pos1, pos2):
        parent1 = self._find(pos1)
        parent2 = self._find(pos2)
        if parent1 != parent2:
            # Note we always connect light branch to heavy branch
            # eg. 1 <- 2 <- 3 <- 5  and 6 <- 7
            # we will have 1 <- 2 <- 3 <- 5 and 1 <- 6 <- 7
            if self._parent_weight[parent1] == self._parent_weight[parent2]:
                self._parent[parent1] = pos2
                self._parent_weight[parent2] += 1
            elif self._parent_weight[parent1] < self._parent_weight[parent2]:
                # parent2 has more nodes recognize it as root
                self._parent[parent1] = pos2
            else:
                # parent1 has more nodes recognize it as root
                self._parent[parent2] = pos1

    def _is_connected(self, pos1, pos2):
        return self._find(pos1) == self._find(pos2)
```


### May 7, 2019 \[Hard\] Largest Sum of Non-adjacent Numbers
---

> **Question:** Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0 or negative.
>
> For example, `[2, 4, 6, 2, 5]` should return 13, since we pick 2, 6, and 5. `[5, 1, 1, 5]` should return 10, since we pick 5 and 5.
>
> Follow-up: Can you do this in O(N) time and constant space?

**My thoughts:** Try to find the pattern by examples:

Let keep it simple first, just assume all elements are positive:
```py
[] => 0  # Sum of none equals 0
[2] => 2  # the only elem is 2
[2, 4] => 4  # 2 and 4 we will choose 4
[2, 4, 6] => 2+6 => 8  # The pattern is eithe choose all odd index numbers or all even index numbers. eg.  [_, 4, _] vs [2, _, 6]. We choose 2,6 which gives 8. If it's [1, 99, 1], we will choose 99 instead.
[2, 4, 6, 2] => [_, 4, _, 2] vs [2, _, 6, _] => 6 vs 8 => 8

# If we let f[lst] represents the max non-adjacent sum when input array is lst
f[2, 4, 6] = max { f[2] + 6, f[4] } = max{6, 8} = 8
f[2, 4, 6, 2] = max { f[2, 4] + 2, f[2, 4, 6] } = max{4, 8} = 8
f[2, 4, 6, 2, 5] = max {f[2, 4, 6] + 5, f[2, 4, 6, 2]} = max{8 + 5, 8} = 13

f[a1, a2, ...., an] = max {f[a1, ..., an-2] + an, f[a1, a2, ..., an-1]}
```
What if one of the element is negative? i.e. `[-1, -1, 1]`

```py
# if we continue apply the formula above, we will end up minus a number from max sum
f[-1, -1, 1] = max{f[-1] + 1, f[-1, -1]} = max{-1+1, -1} = 0

# we can choose to not include that number
#f[-1, -1, 1] should equal to 1
f[-1, -1, 1] = max{f[-1] + 1, f[-1, -1], 1} = max{-1+1, -1, 1} = 1

# As all previous sum chould be a negative number, if that's the case, we can just include the positive number.
f[a1, a2, ...., an] = max {f[a1, ..., an-2] + an, f[a1, a2, ..., an-1], an}

# If we let dp[n] represents the max non-adjacent sum when input size is n, then 
dp[i] = max(dp[i-1], dp[i-2] + numbers[i-1], numbers[i-1])

# dp[n] will be the answer to f[a0, a2, ...., an-1]
```

**Python Solution:** [https://repl.it/@trsong/nonadjacentsum](https://repl.it/@trsong/nonadjacentsum)

```py
def non_adjacent_sum(numbers):
    # Let dp[n] represents the max non-adjacent sum when input size is n.
    # Now let's focusing on the number with index n - 1:
    # * If that number is negative, then including this number won't do us any good. i.e. dp[n] = dp[n - 1]
    # * If that number is positive, then we either include this number, i.e. dp[n] = dp[n-2] + numbers[n-1]
    #                                       or not include this number, i.e. dp[n] = dp[n-1]
    # * Moreover, what if dp[n-1] and dp[n-2] are all negative, but the number with index n - 1 is positive. 
    #   Then we will need to consider dp[n] = numbers[n-1]
    if not numbers: return 0
    if len(numbers) == 1: return numbers[0]
    n = len(numbers)
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = numbers[0]
    dp[2] = max(numbers[0], numbers[1])
    for i in xrange(3, n + 1):
        dp[i] = max(dp[i-1], dp[i-2] + numbers[i-1], numbers[i-1])
    return dp[n]

def non_adjacent_sum2(numbers):
    # Solution satisfies O(N) time and constant space
    if not numbers: return 0
    if len(numbers) == 1: return numbers[0]
    prev_prev_dp = numbers[0]
    prev_dp = max(numbers[0], numbers[1])
    current = prev_dp
    for i in xrange(3, len(numbers) + 1):
        current = max(prev_dp, prev_prev_dp + numbers[i-1], numbers[i-1])
        prev_prev_dp, prev_dp = prev_dp, current
    return current

def main():
    assert non_adjacent_sum2([]) == 0
    assert non_adjacent_sum2([2]) == 2
    assert non_adjacent_sum2([2, 4]) == 4
    assert non_adjacent_sum2([2, 4, -1]) == 4
    assert non_adjacent_sum2([2, 4, -1, 0]) == 4
    assert non_adjacent_sum2([2, 4, 6, 2, 5]) == 13
    assert non_adjacent_sum2([5, 1, 1, 5]) == 10
    assert non_adjacent_sum2([-1, -2, -3, -4]) == -1
    assert non_adjacent_sum2([1, -1, 1, -1]) == 2
    assert non_adjacent_sum2([1, -1, -1, -1]) == 1
    assert non_adjacent_sum2([1, 1, 1, 1, 1]) == 3
    assert non_adjacent_sum2([1, -1, -1, 1, 2]) == 3
    assert non_adjacent_sum2([1, -1, -1, 2, 1]) == 3

if __name__ == '__main__':
    main()
```

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
 
