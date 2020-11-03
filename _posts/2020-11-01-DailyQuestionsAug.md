---
layout: post
title:  "Daily Coding Problems 2020 Nov to Jan"
date:   2020-11-01 22:22:32 -0700
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


### Nov 3, 2020 LC 121 \[Easy\] Best Time to Buy and Sell Stock
---
> **Question:** You are given an array. Each element represents the price of a stock on that particular day. Calculate and return the maximum profit you can make from buying and selling that stock only once.

**Example:**
```py
Input: [9, 11, 8, 5, 7, 10]
Output: 5
Explanation: Here, the optimal trade is to buy when the price is 5, and sell when it is 10, so the return value should be 5 (profit = 10 - 5 = 5).
```

 
### Nov 2, 2020 LC 403 \[Hard\] Frog Jump
---
> **Question:** A frog is crossing a river. The river is divided into x units and at each unit there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.
>
> Given a list of stones' positions (in units) in sorted ascending order, determine if the frog is able to cross the river by landing on the last stone. Initially, the frog is on the first stone and assume the first jump must be 1 unit.
> 
> If the frog's last jump was k units, then its next jump must be either k - 1, k, or k + 1 units. Note that the frog can only jump in the forward direction.

**Example 1:**

```py
[0, 1, 3, 5, 6, 8, 12, 17]

There are a total of 8 stones.
The first stone at the 0th unit, second stone at the 1st unit,
third stone at the 3rd unit, and so on...
The last stone at the 17th unit.

Return true. The frog can jump to the last stone by jumping 
1 unit to the 2nd stone, then 2 units to the 3rd stone, then 
2 units to the 4th stone, then 3 units to the 6th stone, 
4 units to the 7th stone, and 5 units to the 8th stone.
```

**Example 2:**

```py
[0, 1, 2, 3, 4, 8, 9, 11]

Return false. There is no way to jump to the last stone as 
the gap between the 5th and 6th stone is too large.
```

**Solution with DFS:** [https://repl.it/@trsong/Solve-Frog-Jump-Problem](https://repl.it/@trsong/Solve-Frog-Jump-Problem)
```py
import unittest

def can_cross(stones):
    stone_set = set(stones)
    visited = set()
    stack = [(0, 0)]
    goal = stones[-1]

    while stack:
        stone, step = stack.pop()
        if stone == goal:
            return True
        visited.add((stone, step))
        for delta in [-1, 0, 1]:
            next_step = step + delta
            next_stone = stone + next_step
            if next_stone >= stone and next_stone in stone_set and (next_stone, next_step) not in visited:
                stack.append((next_stone, next_step))

    return False
        

class CanCrossSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(can_cross([0, 1, 3, 5, 6, 8, 12, 17])) # step: 1(1), 2(3), 2(5), 3(8), 4(12), 5(17)

    def test_example2(self):
        self.assertFalse(can_cross([0, 1, 2, 3, 4, 8, 9, 11]))

    def test_fast_then_slow(self):
        self.assertTrue(can_cross([0, 1, 3, 6, 10, 13, 15, 16, 16]))

    def test_fast_then_cooldown(self):
        self.assertFalse(can_cross([0, 1, 3, 6, 10, 11]))

    def test_unreachable_last_stone(self):
        self.assertFalse(can_cross([0, 1, 3, 6, 11]))

    def test_reachable_last_stone(self):
        self.assertTrue(can_cross([0, 1, 3, 6, 10]))

    def test_fall_into_water_in_the_middle(self):
        self.assertFalse(can_cross([0, 1, 10, 1000, 1000]))
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 1, 2020 \[Medium\] The Tower of Hanoi
---
> **Question:** The Tower of Hanoi is a puzzle game with three rods and n disks, each a different size.
>
> All the disks start off on the first rod in a stack. They are ordered by size, with the largest disk on the bottom and the smallest one at the top.
>
> The goal of this puzzle is to move all the disks from the first rod to the last rod while following these rules:
>
> - You can only move one disk at a time.
> - A move consists of taking the uppermost disk from one of the stacks and placing it on top of another stack.
> - You cannot place a larger disk on top of a smaller disk.
>
> Write a function that prints out all the steps necessary to complete the Tower of Hanoi. 
> - You should assume that the rods are numbered, with the first rod being 1, the second (auxiliary) rod being 2, and the last (goal) rod being 3.

**Example:** 
```py
with n = 3, we can do this in 7 moves:

Move 1 to 3
Move 1 to 2
Move 3 to 2
Move 1 to 3
Move 2 to 1
Move 2 to 3
Move 1 to 3
```

**My thoughts:** Think about the problem backwards, like what is the most significant states to reach the final state. There are three states coming into my mind: 

- First state, we move all disks except for last one from rod 1 to rod 2. i.e. `[[3], [1, 2], []]`.
- Second state, we move the last disk from rod 1 to rod 3. i.e. `[[], [1, 2], [3]]`
- Third state, we move all disks from rod 2 to rod 3. i.e. `[[], [], [1, 2, 3]]`

There is a clear recrusive relationship between game with size n and size n - 1. So we can perform above stategy recursively for game with size n - 1 which gives the following implementation.

**Solution with Divide-and-Conquer:** [https://repl.it/@trsong/Solve-the-Tower-of-Hanoi-Problem](https://repl.it/@trsong/Solve-the-Tower-of-Hanoi-Problem)
```py
import unittest

def hanoi_moves(n):
    res = []

    def hanoi_moves_recur(n, src, dst):
        if n <= 0:
            return 

        bridge = 3 - src - dst
        # use the unused rod as bridge rod
        # Step1: move n - 1 disks from src to bridge to allow last disk move to dst
        hanoi_moves_recur(n - 1, src, bridge)

        # Step2: move last disk from src to dst
        res.append((src, dst))

        # Step3: move n - 1 disks from bridge to dst
        hanoi_moves_recur(n - 1, bridge, dst)

    hanoi_moves_recur(n, 0, 2)
    return res


class HanoiMoveSpec(unittest.TestCase):
    def assert_hanoi_moves(self, n, moves):
        game = HanoiGame(n)
        # Turn on verbose for debugging
        self.assertTrue(game.can_moves_finish_game(moves, verbose=False))

    def test_three_disks(self):
        moves = hanoi_moves(3)
        self.assert_hanoi_moves(3, moves)

    def test_one_disk(self):
        moves = hanoi_moves(1)
        self.assert_hanoi_moves(1, moves)

    def test_ten_disks(self):
        moves = hanoi_moves(10)
        self.assert_hanoi_moves(10, moves)


class HanoiGame(object):
    def __init__(self, num_disks):
        self.num_disks = num_disks
        self.reset()
        
    def reset(self):
        self.rods = [[disk for disk in xrange(self.num_disks, 0, -1)], [], []]

    def move(self, src, dst):
        disk = self.rods[src].pop()
        self.rods[dst].append(disk)

    def is_feasible_move(self, src, dst):
        return 0 <= src <= 2 and 0 <= dst <= 2 and self.rods[src] and (not self.rods[dst] or self.rods[src][-1] < self.rods[dst][-1])

    def is_game_finished(self):
        return len(self.rods[-1]) == self.num_disks

    def can_moves_finish_game(self, actions, verbose=False):
        self.reset()
        for step, action in enumerate(actions):
            src, dst = action
            if verbose:
                self.display()
                print "Step %d: %d -> %d" % (step, src, dst)
            if not self.is_feasible_move(src, dst):
                return False
            else:
                self.move(src, dst)
        if verbose:
            self.display()
            
        return self.is_game_finished()

    def display(self):
        for plates in self.rods:
            print "- %s" % str(plates)
    

class HanoiGameSpec(unittest.TestCase):
    def test_example_moves(self):
        game = HanoiGame(3)
        moves = [(0, 2), (0, 1), (2, 1), (0, 2), (1, 0), (1, 2), (0, 2)]
        self.assertTrue(game.can_moves_finish_game(moves))

    def test_invalid_moves(self):
        game = HanoiGame(3)
        moves = [(0, 1), (0, 1)]
        self.assertFalse(game.can_moves_finish_game(moves))

    def test_unfinished_moves(self):
        game = HanoiGame(3)
        moves = [(0, 1)]
        self.assertFalse(game.can_moves_finish_game(moves))


if __name__ == '__main__':
    unittest.main(exit=False)
```
