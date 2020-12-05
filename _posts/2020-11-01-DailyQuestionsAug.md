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



### Dec 5, 2020 LC 274 \[Medium\] H-Index
---
> **Question:** The h-index is a metric that attempts to measure the productivity and citation impact of the publication of a scholar. The definition of the h-index is if a scholar has at least h of their papers cited h times.
>
> Given a list of publications of the number of citations a scholar has, find their h-index.

**Example:**
```py
Input: [3, 5, 0, 1, 3]
Output: 3
Explanation:
There are 3 publications with 3 or more citations, hence the h-index is 3.
```


### Dec 4, 2020 \[Hard\] Sliding Puzzle 
---
> **Question:**  An 8-puzzle is a game played on a 3 x 3 board of tiles, with the ninth tile missing. The remaining tiles are labeled 1 through 8 but shuffled randomly. Tiles may slide horizontally or vertically into an empty space, but may not be removed from the board.
>
> Design a class to represent the board, and find a series of steps to bring the board to the state `[[1, 2, 3], [4, 5, 6], [7, 8, None]]`.

**Solution with A-Star Search:** [https://repl.it/@trsong/Solve-Sliding-Puzzle](https://repl.it/@trsong/Solve-Sliding-Puzzle)
```py
import unittest
from copy import deepcopy
from Queue import PriorityQueue

class SlidingPuzzle(object):
    BLANK_VALUE = 9
    GOAL_STATE = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, None]
    ]
    DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    @staticmethod
    def solve(grid):
        """
        Given a grid and returns a sequence of moves to reach to goal state
        """
        pq = PriorityQueue()
        end_hash = SlidingPuzzle.hash(SlidingPuzzle.GOAL_STATE)
        start_hash = SlidingPuzzle.hash(grid)

        visited = set()
        pq.put((0, start_hash))
        prev_states = {start_hash: (None, None)}
        actual_cost = {start_hash: 0}

        while not pq.empty():
            _, cur_hash = pq.get()
            if cur_hash == end_hash:
                break
            if cur_hash in visited:
                continue
            visited.add(cur_hash)

            for neihbor_hash, neighbor_heuristic, move in SlidingPuzzle.neighbor_costs(cur_hash):
                if neihbor_hash in visited:
                    continue
                actual_cost[neihbor_hash] = actual_cost[cur_hash] + 1
                pq.put((actual_cost[neihbor_hash] + neighbor_heuristic, neihbor_hash))
                prev_states[neihbor_hash] = (cur_hash, move)

        moves = []
        while end_hash:
            prev_hash, move = prev_states[end_hash]
            if move:
                moves.append(move)
            end_hash = prev_hash
        moves.reverse()
        return moves

    @staticmethod
    def neighbor_costs(grid_hash):
        """
        Given a grid hash, returns next grid's hash value, cost and move 
        """
        blank_row, blank_col = 0, 0
        grid = [[None for _ in xrange(3)] for _ in xrange(3)]
        for r in xrange(2, -1, -1):
            for c in xrange(2, -1, -1):
                grid[r][c] = grid_hash % 10
                grid_hash //= 10
                if grid[r][c] == SlidingPuzzle.BLANK_VALUE:
                    grid[r][c] = None
                    blank_row, blank_col = r, c

        for dr, dc in SlidingPuzzle.DIRECTIONS:
            new_r, new_c = blank_row + dr, blank_col + dc
            if 0 <= new_r < 3 and 0 <= new_c < 3:
                move = grid[new_r][new_c]
                grid[new_r][new_c] = None
                grid[blank_row][blank_col] = move
                yield SlidingPuzzle.hash(grid), SlidingPuzzle.heuristc(grid), move
                grid[new_r][new_c] = move
                grid[blank_row][blank_col] = None

    @staticmethod
    def hash(grid):
        """
        Flatten grid and then convert to an integer
        """
        res = 0
        for row in grid:
            for num in row:
                # treat None as 9
                res = res * 10 + (num or SlidingPuzzle.BLANK_VALUE)
        return res

    @staticmethod
    def heuristc(grid):
        """
        Estimation of remaining cost based on current grid
        """
        cost = 0
        for r in xrange(3):
            for c in xrange(3):
                # treat None as 9
                num = grid[r][c] or SlidingPuzzle.BLANK_VALUE
                expected_r = (num - 1) // 3
                expected_c = (num - 1) % 3
                cost += abs(r - expected_r) + abs(c - expected_c)
        return cost
        

class SlidingPuzzleSpec(unittest.TestCase):
    ###################
    # Testing Utility
    ###################
    @staticmethod
    def validate(grid, steps):
        blank_row = map(lambda row: None in row, grid).index(True)
        blank_col = grid[blank_row].index(None)

        for num in steps:
            for dr, dc in SlidingPuzzle.DIRECTIONS:
                new_r, new_c = blank_row + dr, blank_col + dc
                if 0 <= new_r < 3 and 0 <= new_c < 3 and grid[new_r][new_c] == num:
                    grid[blank_row][blank_col] = grid[new_r][new_c]
                    grid[new_r][new_c] = None
                    blank_row = new_r
                    blank_col = new_c
                    break
        
        return grid == SlidingPuzzle.GOAL_STATE

    def assert_result(self, grid):
        user_grid = deepcopy(grid)
        steps = SlidingPuzzle.solve(user_grid)
        self.assertTrue(SlidingPuzzleSpec.validate(grid, steps), user_grid)

    def test_heuristic_function_should_not_overestimate(self):
        # Optimial solution: [1, 2, 3, 6, 5, 4, 7, 8]
        self.assert_result([
            [None, 1, 2],
            [5, 6, 3],
            [4, 7, 8],
        ])

    def test_random_grid(self):
        # Optimial solution: [7, 4, 5, 6, 2, 5, 6, 1, 4, 7, 8, 6, 5, 2, 1, 4, 7, 8]
        self.assert_result([
            [6, 2, 3],
            [5, 4, 8],
            [1, 7, None]
        ])

    def test_random_grid2(self):
        # Optimial solution: [3, 6, 8, 4, 7, 5, 2, 1, 4, 7, 5, 3, 6, 8, 7, 4, 1, 2, 3, 6]
        self.assert_result([
            [1, 2, 5],
            [8, 4, 7],
            [6, 3, None]
        ])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 3, 2020 \[Hard\] Knight's Tour Problem
---
> **Question:** A knight's tour is a sequence of moves by a knight on a chessboard such that all squares are visited once.
>
> Given N, write a function to return the number of knight's tours on an N by N chessboard.


**Solution with Backtracking:** [https://repl.it/@trsong/Knights-Tour-Problem](https://repl.it/@trsong/Knights-Tour-Problem)
```py
import unittest

KNIGHT_MOVES = [
    (1, 2), (2, 1), 
    (-1, 2), (-2, 1), 
    (-1, -2), (-2, -1),
    (1, -2), (2, -1)
]


def knights_tours(n):
    if n == 0:
        return 0

    grid = [[False for _ in xrange(n)] for _ in xrange(n)]
    count = 0
    for r in xrange(n):
        for c in xrange(n):
            grid[r][c] = True
            count += backtrack(1, grid, (r, c))
            grid[r][c] = False
    return count


def backtrack(step, grid, pos):
    n = len(grid)
    if step == n * n:
        return 1
    else:
        count = 0
        r, c = pos
        for dr, dc in KNIGHT_MOVES:
            new_r, new_c = r + dr, c + dc
            if 0 <= new_r < n and 0 <= new_c < n and not grid[new_r][new_c]:
                grid[new_r][new_c] = True
                count += backtrack(step + 1, grid, (new_r, new_c))
                grid[new_r][new_c] = False
        return count
                

class KnightsTourSpec(unittest.TestCase):
    """
    Kngiths Tours answer adapts from wiki: https://en.wikipedia.org/wiki/Knight%27s_tour
    """
    def test_size_zero_grid(self):
        self.assertEqual(0, knights_tours(0))

    def test_size_one_grid(self):
        self.assertEqual(1, knights_tours(1))

    def test_size_two_grid(self):
        self.assertEqual(0, knights_tours(2))

    def test_size_three_grid(self):
        self.assertEqual(0, knights_tours(3))

    def test_size_four_grid(self):
        self.assertEqual(0, knights_tours(4))
        
    # Long running execution: took 235 sec
    def test_size_five_grid(self):
        self.assertEqual(1728, knights_tours(5))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 2, 2020 LC 127 \[Medium\] Word Ladder
---
> **Question:** Given a `start` word, an `end` word, and a dictionary of valid words, find the shortest transformation sequence from `start` to `end` such that only one letter is changed at each step of the sequence, and each transformed word exists in the dictionary. If there is no possible transformation, return null. Each word in the dictionary have the same length as start and end and is lowercase.
>
> For example, given `start = "dog"`, `end = "cat"`, and `dictionary = {"dot", "dop", "dat", "cat"}`, return `["dog", "dot", "dat", "cat"]`.
>
> Given `start = "dog"`, `end = "cat"`, and `dictionary = {"dot", "tod", "dat", "dar"}`, return `null` as there is no possible transformation from `dog` to `cat`.

**Solution with BFS:** [https://repl.it/@trsong/Word-Ladder](https://repl.it/@trsong/Word-Ladder)
```py
import unittest
from Queue import Queue

def word_ladder(start, end, word_set):
    parents = {}
    visited = set()
    queue = Queue()
    queue.put((start, None))

    while not queue.empty():
        for _ in xrange(queue.qsize()):
            cur, prev = queue.get()
            if cur in visited:
                continue
            visited.add(cur)

            parents[cur] = prev
            if cur == end:
                return find_path(parents, end)

            for word in word_set:
                if word in visited or not is_neighbor(cur, word):
                    continue
                queue.put((word, cur))
    
    return None


def find_path(parents, start):
    res = []
    while start:
        res.append(start)
        start = parents.get(start, None)
    res.reverse()
    return res


def is_neighbor(word1, word2):
    count = 0
    for c1, c2 in zip(word1, word2):
        if c1 != c2:
            count += 1
        if count > 1:
            return False
    return count == 1


class WordLadderSpec(unittest.TestCase):
    def test_example(self):
        start = 'dog'
        end = 'cat'
        word_set = {'dot', 'dop', 'dat', 'cat'}
        expected = ['dog', 'dot', 'dat', 'cat']
        self.assertEqual(expected, word_ladder(start, end, word_set))

    def test_example2(self):
        start = 'dog'
        end = 'cat'
        word_set = {'dot', 'tod', 'dat', 'dar'}
        self.assertIsNone(word_ladder(start, end, word_set))

    def test_empty_dict(self):
        self.assertIsNone(word_ladder('start', 'end', {}))

    def test_example3(self):
        start = 'hit'
        end = 'cog'
        word_set = {'hot', 'dot', 'dog', 'lit', 'log', 'cog'}
        expected = ['hit', 'hot', 'dot', 'dog', 'cog']
        self.assertEqual(expected, word_ladder(start, end, word_set))

    def test_end_word_not_in_dictionary(self):
        start = 'hit'
        end = 'cog'
        word_set = ['hot', 'dot', 'dog', 'lot', 'log']
        self.assertIsNone(word_ladder(start, end, word_set))

    def test_long_example(self):
        start = 'coder'
        end = 'goner'
        word_set = {
            'lover', 'coder', 'comer', 'toner', 'cover', 'tower', 'coyer',
            'bower', 'honer', 'poles', 'hover', 'lower', 'homer', 'boyer',
            'goner', 'loner', 'boner', 'cower', 'never', 'sower', 'asian'
        }
        expected = ['coder', 'cower', 'lower', 'loner', 'goner']
        self.assertEqual(expected, word_ladder(start, end, word_set))

    def test_long_example2(self):
        start = 'coder'
        end = 'goner'
        word_set = {
            'lover', 'coder', 'comer', 'toner', 'cover', 'tower', 'coyer',
            'bower', 'honer', 'poles', 'hover', 'lower', 'homer', 'boyer',
            'goner', 'loner', 'boner', 'cower', 'never', 'sower', 'asian'
        }
        expected = ['coder', 'cower', 'lower', 'loner', 'goner']
        self.assertEqual(expected, word_ladder(start, end, word_set))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Dec 1, 2020 \[Easy\] Intersection of N Arrays
---
> **Question:** Given n arrays, find the intersection of them.

**Example:**
```py
intersection([1, 2, 3, 4], [2, 4, 6, 8], [3, 4, 5])  # returns [4]
```

**Solution:** [https://repl.it/@trsong/Intersection-of-N-Arrays](https://repl.it/@trsong/Intersection-of-N-Arrays)
```py
import unittest

def intersection(*num_lsts):
    if not num_lsts:
        return []

    n = len(num_lsts)
    sorted_lsts = map(sorted, num_lsts)
    positions = [0] * n

    res = []
    while True:
        min_val = float('inf')
        for i, pos in enumerate(positions):
            if pos >= len(sorted_lsts[i]):
                return res
            min_val = min(min_val, sorted_lsts[i][pos])

        count = 0
        for i, pos in enumerate(positions):
            if sorted_lsts[i][pos] == min_val:
                count += 1
                positions[i] += 1
        
        if count == n:
            res.append(min_val)

    return None


class IntersectionSpec(unittest.TestCase):
    def test_example(self):
        list1 = [1, 2, 3, 4]
        list2 = [2, 4, 6, 8]
        list3 = [3, 4, 5]
        expected = [4]
        self.assertEqual(expected, intersection(list1, list2, list3))

    def test_example2(self):
        list1 = [1, 5, 10, 20, 40, 80]
        list2 = [6, 7, 20, 80, 100]
        list3 = [3, 4, 15, 20, 30, 70, 80, 120]
        expected = [20, 80]
        self.assertEqual(expected, intersection(list1, list2, list3))

    def test_example3(self):
        list1 = [1, 5, 6, 7, 10, 20]
        list2 = [6, 7, 20, 80]
        list3 = [3, 4, 5, 7, 15, 20]
        expected = [7, 20]
        self.assertEqual(expected, intersection(list1, list2, list3))
    
    def test_empty_array(self):
        self.assertEqual([], intersection())

    def test_one_array(self):
        self.assertEqual([1, 2], intersection([1, 2]))

    def test_two_arrays(self):
        list1 = [1, 2, 3]
        list2 = [5, 3, 1]
        expected = [1, 3]
        self.assertEqual(expected, intersection(list1, list2))

    def test_reverse_order(self):
        list1 = [4, 3, 2, 1]
        list2 = [2, 4, 6, 8]
        list3 = [5, 4, 3]
        expected = [4]
        self.assertEqual(expected, intersection(list1, list2, list3))

    def test_contains_duplicate(self):
        list1 = [1, 5, 5]
        list2 = [3, 4, 5, 5, 10]
        list3 = [5, 5, 10, 20]
        expected = [5, 5]
        self.assertEqual(expected, intersection(list1, list2, list3))

    def test_different_length_lists(self):
        list1 = [1, 5, 10, 20, 30]
        list2 = [5, 13, 15, 20]
        list3 = [5, 20]
        expected = [5, 20]
        self.assertEqual(expected, intersection(list1, list2, list3))
    
    def test_empty_list(self):
        list1 = [1, 2, 3, 4, 5]
        list2 = [4, 5, 6, 7]
        list3 = []
        expected = []
        self.assertEqual(expected, intersection(list1, list2, list3))

    def test_empty_list2(self):
        self.assertEqual([], intersection([], [], []))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 30, 2020 LC 228 \[Easy\] Extract Range
---
> **Question:** Given a sorted list of numbers, return a list of strings that represent all of the consecutive numbers.

**Example:**
```py
Input: [0, 1, 2, 5, 7, 8, 9, 9, 10, 11, 15]
Output: ['0->2', '5', '7->11', '15']
```

**Solution:** [https://repl.it/@trsong/Extract-Range](https://repl.it/@trsong/Extract-Range)
```py
import unittest

def extract_range(nums):
    if not nums:
        return []

    res = []
    base = nums[0]
    n = len(nums)

    for i in xrange(1, n+1):
        if i < n and (nums[i] - nums[i-1]) <= 1:
            continue

        if base == nums[i-1]:
            res.append(str(base))
        else:
            res.append("%d->%d" % (base, nums[i-1]))

        if i < n:
            base = nums[i]

    return res


class ExtractRangeSpec(unittest.TestCase):
    def test_example(self):
        nums = [0, 1, 2, 5, 7, 8, 9, 9, 10, 11, 15]
        expected = ['0->2', '5', '7->11', '15']
        self.assertEqual(expected, extract_range(nums))

    def test_empty_array(self):
        self.assertEqual([], extract_range([]))

    def test_one_elem_array(self):
        self.assertEqual(['42'], extract_range([42]))

    def test_duplicates(self):
        nums = [1, 1, 1, 1]
        expected = ['1']
        self.assertEqual(expected, extract_range(nums))

    def test_duplicates2(self):
        nums = [1, 1, 2, 2]
        expected = ['1->2']
        self.assertEqual(expected, extract_range(nums))

    def test_duplicates3(self):
        nums = [1, 1, 3, 3, 5, 5, 5]
        expected = ['1', '3', '5']
        self.assertEqual(expected, extract_range(nums))

    def test_first_elem_in_range(self):
        nums = [1, 2, 3, 10, 11]
        expected = ['1->3', '10->11']
        self.assertEqual(expected, extract_range(nums))

    def test_first_elem_not_in_range(self):
        nums = [-5, -3, -2]
        expected = ['-5', '-3->-2']
        self.assertEqual(expected, extract_range(nums))

    def test_last_elem_in_range(self):
        nums = [0, 15, 16, 17]
        expected = ['0', '15->17']
        self.assertEqual(expected, extract_range(nums))

    def test_last_elem_not_in_range(self):
        nums = [-42, -1, 0, 1, 2, 15]
        expected = ['-42', '-1->2', '15']
        self.assertEqual(expected, extract_range(nums))

    def test_entire_array_in_range(self):
        nums = list(range(-10, 10))
        expected = ['-10->9']
        self.assertEqual(expected, extract_range(nums))

    def test_no_range_at_all(self):
        nums = [1, 3, 5]
        expected = ['1', '3', '5']
        self.assertEqual(expected, extract_range(nums))

    def test_range_and_not_range(self):
        nums = [0, 1, 3, 5, 7, 8, 9, 11, 13, 14, 15]
        expected = ['0->1', '3', '5', '7->9', '11', '13->15']
        self.assertEqual(expected, extract_range(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 29, 2020 \[Medium\] Implement Soundex
---
> **Question:** **Soundex** is an algorithm used to categorize phonetically, such that two names that sound alike but are spelled differently have the same representation.
>
> **Soundex** maps every name to a string consisting of one letter and three numbers, like M460.
>
> One version of the algorithm is as follows:
>
> 1. Remove consecutive consonants with the same sound (for example, change ck -> c).
> 2. Keep the first letter. The remaining steps only apply to the rest of the string.
> 3. Remove all vowels, including y, w, and h.
> 4. Replace all consonants with the following digits:
>    - b, f, p, v → 1
>    - c, g, j, k, q, s, x, z → 2
>    - d, t → 3
>    - l → 4
>    - m, n → 5
>    - r → 6
> 5. If you don't have three numbers yet, append zeros until you do. Keep the first three numbers.
>
> Using this scheme, Jackson and Jaxen both map to J250.

**Solution:** [https://repl.it/@trsong/Implement-Soundex](https://repl.it/@trsong/Implement-Soundex)
```py
import unittest

def soundex_code(s):
    skip_letters = {'y', 'w', 'h'}
    score_to_letter = {
        '1': 'bfpv',
        '2': 'cgjkqsxz',
        '3': 'dt',
        '4': 'l',
        '5': 'mn',
        '6': 'r'
    }
    letter_to_score = {ch: score for score, chs in score_to_letter.items() for ch in chs}
    
    initial = s[0].upper()
    res = [initial]
    prev = None
    for ch in s:
        lower_case = ch.lower()
        if lower_case in skip_letters:
            continue

        code = letter_to_score.get(lower_case, '0')
        if prev is not None and code != '0' and code != prev:
            res.append(code)
        prev = code

        if len(res) == 4:
            break

    res.extend('0000')
    return ''.join(res[:4])


class SoundexCodeSpec(unittest.TestCase):
    def test_example(self):
        # J, 2 for the C, K ignored, S ignored, 5 for the N, 0 added
        self.assertEqual('J250', soundex_code('Jackson'))

    def test_example2(self):
        self.assertEqual('J250', soundex_code('Jaxen'))

    def test_name_with_double_letters(self):
        # G, 3 for the T, 6 for the first R, second R ignored, 2 for the Z
        self.assertEqual('G362', soundex_code('Gutierrez'))

    def test_side_by_side_same_code(self):
        # P, F ignored, 2 for the S, 3 for the T, 6 for the R
        self.assertEqual('P236', soundex_code('Pfister'))

    def test_side_by_side_same_code2(self):
        # T, 5 for the M, 2 for the C, Z ignored, 2 for the K
        self.assertEqual('T522', soundex_code('Tymczak'))

    def test_append_zero_to_end(self):
        self.assertEqual('L000', soundex_code('Lee'))

    def test_discard_extra_letters(self):
        # W, 2 for the S, 5 for the N, 2 for the G, remaining letters disregarded
        self.assertEqual('W252', soundex_code('Washington')) 

    def test_separate_consonant_with_same_code(self):
        self.assertEqual('A261', soundex_code('Ashcraft'))

    def test_more_example(self):
        self.assertEqual('K530', soundex_code('Knuth'))

    def test_more_example2(self):
        self.assertEqual('K530', soundex_code('Kant'))

    def test_more_example3(self):
        self.assertEqual('J612', soundex_code('Jarovski'))

    def test_more_example4(self):
        self.assertEqual('R252', soundex_code('Resnik'))

    def test_more_example5(self):
        self.assertEqual('R252', soundex_code('Reznick'))

    def test_more_example6(self):
        self.assertEqual('E460', soundex_code('Euler'))

    def test_more_example7(self):
        self.assertEqual('P362', soundex_code('Peterson'))

    def test_more_example8(self):
        self.assertEqual('J162', soundex_code('Jefferson'))

    def test_more_example9(self):
        self.assertEqual('T526', soundex_code('Tangrui'))

    def test_more_example10(self):
        self.assertEqual('S520', soundex_code('Song'))

    def test_more_example11(self):
        self.assertEqual('J520', soundex_code('Jing'))

    def test_more_example12(self):
        self.assertEqual('Z520', soundex_code('Zhang'))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 28, 2020 LC 151 \[Medium\] Reverse Words in a String
---
> **Question:** Given an input string, reverse the string word by word.
>
> Note:
>
> - A word is defined as a sequence of non-space characters.
> - Input string may contain leading or trailing spaces. However, your reversed string should not contain leading or trailing spaces.
> - You need to reduce multiple spaces between two words to a single space in the reversed string.

**Example 1:**
```py
Input: "the sky is blue"
Output: "blue is sky the"
```

**Example 2:**
```py
Input: "  hello world!  "
Output: "world! hello"
Explanation: Your reversed string should not contain leading or trailing spaces.
```

**Example 3:**
```py
Input: "a good   example"
Output: "example good a"
Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.
```


**My thoughts:** Suppose after strip out extra whitespaces the string is `"the sky is blue"`. We just need to do the following steps:
1. reverse entire sentence. `"eulb si yks eht"`
2. reverse each word in that sentence. `"blue is sky the"`

**Solution:** [https://repl.it/@trsong/Reverse-words-in-a-string](https://repl.it/@trsong/Reverse-words-in-a-string)
```py
import unittest

def reverse_word(s):
    char_arr = remove_white_spaces(s)
    reverse_section(char_arr, 0, len(char_arr) - 1)
    reverse_each_word(char_arr)
    return ''.join(char_arr)


def remove_white_spaces(s):
    prev = ' '
    res = []
    
    for ch in s:
        if prev == ch == ' ':
            continue
        res.append(ch)
        prev = ch
    
    if res and res[-1] == ' ':
        res.pop()
    
    return res


def reverse_each_word(arr):
    i = 0
    n = len(arr)

    while i < n:
        if arr[i] == ' ':
            i += 1
            continue
        
        start = i
        while i < n and arr[i] != ' ':
            i += 1
        
        reverse_section(arr, start, i-1)


def reverse_section(arr, start, end):
    while start < end:
        arr[start], arr[end] = arr[end], arr[start]
        start += 1
        end -= 1


class ReverseWordSpec(unittest.TestCase):
    def test_example1(self):
        s = "the sky is blue"
        expected = "blue is sky the"
        self.assertEqual(expected, reverse_word(s))

    def test_example2(self):
        s = "  hello world!  "
        expected = "world! hello"
        self.assertEqual(expected, reverse_word(s))

    def test_example3(self):
        s = "a good   example"
        expected = "example good a"
        self.assertEqual(expected, reverse_word(s))

    def test_mutliple_whitespaces(self):
        s = "      "
        expected = ""
        self.assertEqual(expected, reverse_word(s))
    
    def test_mutliple_whitespaces2(self):
        s = "the sky is blue"
        expected = "blue is sky the"
        self.assertEqual(expected, reverse_word(s))

    def test_even_number_of_words(self):
        s = " car cat"
        expected = "cat car"
        self.assertEqual(expected, reverse_word(s))

    def test_even_number_of_words2(self):
        s = "car cat "
        expected = "cat car"
        self.assertEqual(expected, reverse_word(s))

    def test_no_whitespaces(self):
        s = "asparagus"
        expected = "asparagus"
        self.assertEqual(expected, reverse_word(s))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 27, 2020 \[Medium\] Direction and Position Rule Verification
---
> **Question:** A rule looks like this:
>
> ```py
> A NE B
> ``` 
> This means this means point A is located northeast of point B.
> 
> ```
> A SW C
> ```
> means that point A is southwest of C.
>
>Given a list of rules, check if the sum of the rules validate. For example:
> 
> ```
> A N B
> B NE C
> C N A
> ```
> 
> does not validate, since A cannot be both north and south of C.
> 
> ```
> A NW B
> A N B
> ```
> 
> is considered valid.

**My thoughts:** A rule is considered invalid if there exists an opposite rule either implicitly or explicitly. eg. `'A N B'`, `'B N C'` and `'C N A'`. But that doesn't mean we need to scan through the rules over and over again to check if any pair of two rules are conflicting with each other. We can simply scan through the list of rules to build a directional graph for each direction. i.e. E, S, W, N

Then what about diagonal directions. i.e. NE, NW, SE, SW.? As the nature of those rules are 'AND' relation, we can break one diagonal rules into two normal rules. eg, `'A NE B' => 'A N B' and 'A E B'`.

For each direction we build a directional graph, with nodes being the points and edge represents a rule. And each rule will be added to two graphs with opposite directions. e.g. `'A N B'` added to both 'N' graph and 'S' graph. If doing this rule forms a cycle, we simply return False. And if otherwise for all rules, then we return True in the end. 

**Solution with DFS:** [https://repl.it/@trsong/Verify-List-of-Direction-and-Position-Rules](https://repl.it/@trsong/Verify-List-of-Direction-and-Position-Rules)
```py
import unittest

def direction_rule_validate(rules):
    neighbors_by_direction = { 
        direction: {} for direction in [Direction.E, Direction.W, Direction.S, Direction.N]
    }

    for rule in rules:
        start, directions, end = rule.split()
        for direction in directions:
            neighbors = neighbors_by_direction[direction]

            if check_connectivity(neighbors, end, start):
                return False
            
            neighbors[start] = neighbors.get(start, set())
            neighbors[start].add(end)

            opposite_neighbors = neighbors_by_direction[Direction.get_opposite_direction(direction)]
            opposite_neighbors[end] = neighbors.get(end, set())
            opposite_neighbors[end].add(start)

    return True


def check_connectivity(neighbors, start, end):
    stack = [start]
    visited = set()

    while stack:
        cur = stack.pop()
        if cur == end:
            return True
        if cur in visited:
            continue
        visited.add(cur)

        if cur not in neighbors:
            continue

        for child in neighbors[cur]:
            if child not in visited:
                stack.append(child)

    return False


class Direction:
    E = 'E'
    S = 'S'
    W = 'W'
    N = 'N'
    
    @staticmethod
    def get_opposite_direction(d):
        if d == Direction.E: return Direction.W
        if d == Direction.W: return Direction.E
        if d == Direction.S: return Direction.N
        if d == Direction.N: return Direction.S
        return None


class DirectionRuleValidationSpec(unittest.TestCase):
    def test_example1(self):
        self.assertFalse(direction_rule_validate([
            "A N B",
            "B NE C",
            "C N A"
        ]))

    def test_example2(self):
        self.assertTrue(direction_rule_validate([
            "A NW B",
            "A N B"
        ]))

    def test_ambigious_rules(self):
        self.assertTrue(direction_rule_validate([
            "A SE B",
            "C SE B",
            "C SE A",
            "A N C"
        ]))

    def test_conflict_diagonal_directions(self):
        self.assertFalse(direction_rule_validate([
            "B NW A",
            "C SE A",
            "C NE B"
        ]))

    def test_paralllel_rules(self):
        self.assertTrue(direction_rule_validate([
            "A N B",
            "C N D",
            "C E B",
            "B W D",
            "B S D",
            "A N C",
            "D N B",
            "C E A"
        ]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 26, 2020 \[Hard\] Inversion Pairs
---
> **Question:**  We can determine how "out of order" an array A is by counting the number of inversions it has. Two elements `A[i]` and `A[j]` form an inversion if `A[i] > A[j]` but `i < j`. That is, a smaller element appears after a larger element. Given an array, count the number of inversions it has. Do this faster than `O(N^2)` time. You may assume each element in the array is distinct.
>
> For example, a sorted list has zero inversions. The array `[2, 4, 1, 3, 5]` has three inversions: `(2, 1)`, `(4, 1)`, and `(4, 3)`. The array `[5, 4, 3, 2, 1]` has ten inversions: every distinct pair forms an inversion.


**Trivial Solution:** 
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

Did you notice the following properties? 

1. `count_inversion_pairs([5, 4, 3, 2, 1]) = count_inversion_pairs([5, 4]) + count_inversion_pairs([3, 2, 1]) + inversion_pairs_between([5, 4], [3, 2, 1])`
2. `inversion_pairs_between([5, 4], [3, 2, 1]) = inversion_pairs_between(sorted([5, 4]), sorted([3, 2, 1])) = inversion_pairs_between([4, 5], [1, 2, 3])`

This is bascially modified version of merge sort. Consider we break `[5, 4, 3, 2, 1]` into two almost equal parts: `[5, 4]` and `[3, 2, 1]`. Notice such break won't affect inversion pairs, as whatever on the left remains on the left. However, inversion pairs between `[5, 4]` and `[3, 2, 1]` can be hard to count without doing it one-by-one. 

If only we could sort them separately as sort won't affect the inversion order between two lists. i.e. `[4, 5]` and `[1, 2, 3]`. Now let's see if we can find the pattern, if `4 < 1`, then `5 should < 1`. And we also have `4 < 2` and `4 < 3`. We can simply skip all `elem > than 4` on each iteration, i.e. we just need to calculate how many elem > 4 on each iteration. This gives us **property 2**.

And we can futher break `[5, 4]` into `[5]` and `[4]` recursively. This gives us **property 1**.

Combine property 1 and 2 gives us the modified version of ***Merge-Sort***.

**Solution with Merge Sort:** [https://repl.it/@trsong/Count-Inversion-Pairs](https://repl.it/@trsong/Count-Inversion-Pairs)
```py
import unittest

def count_inversion_pairs(nums):
    count, _ = count_and_sort(nums)
    return count


def count_and_sort(nums):
    if len(nums) < 2:
        return 0, nums
    
    mid = len(nums) // 2
    sub_res1, sorted1 = count_and_sort(nums[:mid])
    sub_res2, sorted2 = count_and_sort(nums[mid:])
    count_res, merged = count_and_merge(sorted1, sorted2)
    return sub_res1 + sub_res2 + count_res, merged


def count_and_merge(nums1, nums2):
    inversions = 0
    merged = []

    i = j = 0
    len1, len2 = len(nums1), len(nums2)
    while i < len1 and j < len2:
        if nums1[i] <= nums2[j]:
            merged.append(nums1[i])
            i += 1
        else:
            # there are len1 - i numbers one the left array greater than right
            inversions += len1 - i 
            merged.append(nums2[j])
            j += 1

    while i < len1:
        merged.append(nums1[i])
        i += 1

    while j < len2:
        merged.append(nums2[j])
        j += 1

    return inversions, merged


class CountInversionPairSpec(unittest.TestCase):
    def test_example(self):
        nums = [2, 4, 1, 3, 5]
        expected = 3  # (2, 1), (4, 1), (4, 3)
        self.assertEqual(expected, count_inversion_pairs(nums))

    def test_example2(self):
        nums = [5, 4, 3, 2, 1]
        expected = 10  # (5, 4), (5, 3), ... (2, 1) = 4 + 3 + 2 + 1 = 10 
        self.assertEqual(expected, count_inversion_pairs(nums))

    def test_empty_array(self):
        self.assertEqual(0, count_inversion_pairs([]))

    def test_one_elem_array(self):
        self.assertEqual(0, count_inversion_pairs([42]))

    def test_ascending_array(self):
        nums = [1, 4, 6, 8, 9]
        expected = 0
        self.assertEqual(expected, count_inversion_pairs(nums))

    def test_ascending_array2(self):
        nums = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        expected = 0
        self.assertEqual(expected, count_inversion_pairs(nums))

    def test_increasing_decreasing_array(self):
        nums = [1, 2, 3, 2, 5]
        expected = 1  # (3, 2)
        self.assertEqual(expected, count_inversion_pairs(nums))

    def test_decreasing_increasing_array(self):
        nums = [0, -1, -2, -2, 2, 3]
        expected = 5  # (0, -1), (0, -2), (0, -2), (-1, -2), (-1, -2)
        self.assertEqual(expected, count_inversion_pairs(nums))

    def test_unique_value_array(self):
        nums = [0, 0, 0]
        expected = 0
        self.assertEqual(expected, count_inversion_pairs(nums))

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 25, 2020 \[Hard\] Minimum Cost to Construct Pyramid with Stones
---
> **Question:** You have `N` stones in a row, and would like to create from them a pyramid. This pyramid should be constructed such that the height of each stone increases by one until reaching the tallest stone, after which the heights decrease by one. In addition, the start and end stones of the pyramid should each be one stone high.
>
> You can change the height of any stone by paying a cost of `1` unit to lower its height by `1`, as many times as necessary. Given this information, determine the lowest cost method to produce this pyramid.
>
> For example, given the stones `[1, 1, 3, 3, 2, 1]`, the optimal solution is to pay `2` to create `[0, 1, 2, 3, 2, 1]`.

**My thoughts:** To find min cost is equivalent to find max pyramid we can construct. With that being said, all we need is to figure out each position's max possible height. 

A position's height can only be 1 greater than prvious one on the left and right position. We can scan from left and right to figure out the max height for each position. 

After that, we need to find center location of pyramid. That is just the max height. And the total number of stones equals `1 + 2 + .. + n + ... + 1 = n * n`.

Therefore the `min cost` is just `sum of all stones - total stones of max pyramid`. 

**Solution:** [https://repl.it/@trsong/Minimum-Cost-to-Construct-Pyramid-with-Stones](https://repl.it/@trsong/Minimum-Cost-to-Construct-Pyramid-with-Stones)
```py
import unittest

def min_cost_pyramid(stones):
    n = len(stones)
    left_trim_heights = [None] * n
    right_trim_heights = [None] * n

    left_trim_heights[0] = min(1, stones[0])
    right_trim_heights[n - 1] = min(1, stones[n - 1])

    for i in xrange(1, n):
        left_trim_heights[i] = min(left_trim_heights[i - 1] + 1, stones[i])
        right_trim_heights[n - i - 1] = min(right_trim_heights[n - i] + 1, stones[n - i - 1])

    trim_heights = map(min, zip(left_trim_heights, right_trim_heights))
    pyramid_height = max(trim_heights)

    total_pyramid_stones = pyramid_height * pyramid_height  # 1 + 2 + ... + 2 + 1
    return sum(stones) - total_pyramid_stones


class MinCostPyramidSpec(unittest.TestCase):
    def test_example(self):
        stones = [1, 1, 3, 3, 2, 1]
        expected = 2  # [0, 1, 2, 3, 2, 1]
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_small_pyramid(self):
        stones = [1, 2, 1]
        expected = 0
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_small_pyramid2(self):
        stones = [1, 1, 1]
        expected = 2  # [0, 1, 0]
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_almost_pyramid(self):
        stones = [1, 2, 3, 4, 2, 1]
        expected = 4  # [1, 2, 3, 2, 1, 0]
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_choice_between_different_pyramid(self):
        stones = [1, 2, 1, 0, 0, 1, 2, 3, 2, 1, 0, 1, 0]
        expected = 5  # [0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0]
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_build_from_flat_plane(self):
        stones = [5, 5, 5, 5, 5]
        expected = 16  # [1, 2, 3, 2, 1]
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_concave_array(self):
        stones = [0, 0, 3, 2, 3, 0]
        expected = 4  # [0, 0, 1, 2, 1, 0]
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_multiple_layer_platforms(self):
        stones = [2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 5, 5, 5, 5, 5, 5, 2, 2]
        #        [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0]
        expected = 61
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_multiple_layer_platforms2(self):
        stones = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 1, 1, 1, 0]
        expected = 16  # [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0]
        self.assertEqual(expected, min_cost_pyramid(stones))

    def test_choose_between_two_pyramids(self):
        stones = [1, 2, 3, 2, 1, 0, 0, 1, 6, 1, 0]
        expected = 8  # [1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0]
        self.assertEqual(expected, min_cost_pyramid(stones))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 24, 2020 \[Easy\] Reconstruct a Jumbled Array
---
> **Question:** The sequence `[0, 1, ..., N]` has been jumbled, and the only clue you have for its order is an array representing whether each number is larger or smaller than the last. 
> 
> Given this information, reconstruct an array that is consistent with it. For example, given `[None, +, +, -, +]`, you could return `[1, 2, 3, 0, 4]`.

**My thoughts:** treat `+` as `+1`, `+2`, `+3` and `-` as `-1`, `-2` from `0` you can generate an array with satisfied condition yet incorrect range. Then all we need to do is to shift to range from `0` to `N` by minusing each element with global minimum value. 

**Solution:** [https://repl.it/@trsong/Reconstruct-a-Jumbled-Array](https://repl.it/@trsong/Reconstruct-a-Jumbled-Array)
```py
import unittest

def build_jumbled_array(clues):
    res = [0] * len(clues)
    upper = lower = 0

    for i in xrange(1, len(clues)):
        if clues[i] == '+':
            res[i] = upper + 1
            upper += 1
        elif clues[i] == '-':
            res[i] = lower - 1
            lower -= 1

    return map(lambda num: num - lower, res)


class BuildJumbledArraySpec(unittest.TestCase):
    @staticmethod
    def generate_clues(nums):
        nums_signs = [None] * len(nums)
        for i in xrange(1, len(nums)):
            nums_signs[i] = '+' if nums[i] > nums[i-1] else '-'
        return nums_signs

    def validate_result(self, cludes, res):
        res_set = set(res)
        res_signs = BuildJumbledArraySpec.generate_clues(res)
        msg = "Incorrect result %s. Expect %s but gives %s." % (res, cludes, res_signs)
        self.assertEqual(len(cludes), len(res_set), msg)
        self.assertEqual(0, min(res), msg)
        self.assertEqual(len(cludes) - 1, max(res), msg)
        self.assertEqual(cludes, res_signs, msg)

    def test_example(self):
        clues = [None, '+', '+', '-', '+']
        # possible solution: [1, 2, 3, 0, 4]
        res = build_jumbled_array(clues)
        self.validate_result(clues, res)

    def test_empty_array(self):
        self.assertEqual([], build_jumbled_array([]))

    def test_one_element_array(self):
        self.assertEqual([0], build_jumbled_array([None]))

    def test_two_elements_array(self):
        self.assertEqual([1, 0], build_jumbled_array([None, '-']))

    def test_ascending_array(self):
        clues = [None, '+', '+', '+']
        expected = [0, 1, 2, 3]
        self.assertEqual(expected, build_jumbled_array(clues))

    def test_descending_array(self):
        clues = [None, '-', '-', '-', '-']
        expected = [4, 3, 2, 1, 0]
        self.assertEqual(expected, build_jumbled_array(clues))

    def test_random_array(self):
        clues = [None, '+', '-', '+', '-']
        # possible solution: [1, 4, 2, 3, 0]
        res = build_jumbled_array(clues)
        self.validate_result(clues, res)

    def test_random_array2(self):
        clues = [None, '-', '+', '-', '-', '+']
        # possible solution: [3, 1, 4, 2, 0, 5]
        res = build_jumbled_array(clues)
        self.validate_result(clues, res)

    def test_random_array3(self):
        clues = [None, '+', '-', '+', '-', '+', '+', '+']
        # possible solution: [1, 7, 0, 6, 2, 3, 4, 5]
        res = build_jumbled_array(clues)
        self.validate_result(clues, res)
    
    def test_random_array4(self):
        clues = [None, '+', '+', '-', '+']
        # possible solution: [1, 2, 3, 0, 4]
        res = build_jumbled_array(clues)
        self.validate_result(clues, res)

    def test_random_array5(self):
        clues = [None, '-', '-', '-', '+']
        # possible solution: [3, 2, 1, 0, 4]
        res = build_jumbled_array(clues)
        self.validate_result(clues, res)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 23, 2020 LC 286 \[Medium\] Walls and Gates
---
> **Question:** You are given a m x n 2D grid initialized with these three possible values.
> * -1 - A wall or an obstacle.
> * 0 - A gate.
> * INF - Infinity means an empty room. We use the value `2^31 - 1 = 2147483647` to represent INF as you may assume that the distance to a gate is less than 2147483647.
> 
> Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.

**Example:**

```py
Given the 2D grid:

INF  -1  0  INF
INF INF INF  -1
INF  -1 INF  -1
  0  -1 INF INF

After running your function, the 2D grid should be:

  3  -1   0   1
  2   2   1  -1
  1  -1   2  -1
  0  -1   3   4
  ```

**My thoughts:** Most of time, the BFS you are familiar with has only one starting point and searching from that point onward will produce the shortest path from start to visited points. For multi-starting points, it works exactly as single starting point. All you need to do is to imagine a single vitual starting point connecting to all starting points. Moreover, the way we achieve that is to put all starting point into the queue before doing BFS. 


**Solution with BFS:** [https://repl.it/@trsong/Identify-Walls-and-Gates](https://repl.it/@trsong/Identify-Walls-and-Gates)
```py
import unittest
import sys
from Queue import Queue

INF = sys.maxint

def nearest_gate(grid):
    if not grid or not grid[0]:
        return

    n, m = len(grid), len(grid[0])
    queue = Queue()
    for r in xrange(n):
        for c in xrange(m):
            if grid[r][c] == 0:
                queue.put((r, c))

    depth = 0
    DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    while not queue.empty():
        for _ in xrange(queue.qsize()):
            r, c = queue.get()
            if 0 < grid[r][c] < INF:
                continue
            grid[r][c] = depth

            for dr, dc in DIRECTIONS:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < n and 0 <= new_c < m and grid[new_r][new_c] == INF:
                    queue.put((new_r, new_c))

        depth += 1


class NearestGateSpec(unittest.TestCase):
    def test_example(self):
        grid = [
            [INF,  -1,   0, INF],
            [INF, INF, INF,  -1],
            [INF,  -1, INF,  -1],
            [  0,  -1, INF, INF]
        ]
        expected_grid = [
            [  3,  -1,   0,   1],
            [  2,   2,   1,  -1],
            [  1,  -1,   2,  -1],
            [  0,  -1,   3,   4]
        ]
        nearest_gate(grid)
        self.assertEqual(expected_grid, grid)

    def test_unreachable_room(self):
        grid = [
            [INF, -1],
            [ -1,  0]
        ]
        expected_grid = [
            [INF, -1],
            [ -1,  0]
        ]
        nearest_gate(grid)
        self.assertEqual(expected_grid, grid)

    def test_no_gate_exists(self):
        grid = [
            [-1,   -1],
            [INF, INF]
        ]
        expected_grid = [
            [-1,   -1],
            [INF, INF]
        ]
        nearest_gate(grid)
        self.assertEqual(expected_grid, grid)

    def test_all_gates_no_room(self):
        grid = [
            [0, 0, 0],
            [0, 0, 0]
        ]
        expected_grid = [
            [0, 0, 0],
            [0, 0, 0]
        ]
        nearest_gate(grid)
        self.assertEqual(expected_grid, grid)

    def test_empty_grid(self):
        grid = []
        nearest_gate(grid)
        self.assertEqual([], grid)

    def test_1D_grid(self):
        grid = [[INF, 0, INF, INF, INF, 0, INF, 0, 0, -1, INF]]
        expected_grid = [[1, 0, 1, 2, 1, 0, 1, 0, 0, -1, INF]]
        nearest_gate(grid)
        self.assertEqual(expected_grid, grid)

    def test_multi_gates(self):
        grid = [
            [INF, INF,  -1,   0, INF],
            [INF, INF, INF, INF, INF],
            [  0, INF, INF, INF,   0],
            [INF, INF,  -1, INF, INF]
        ]
        expected_grid = [
            [  2,   3,  -1,   0,   1],
            [  1,   2,   2,   1,   1],
            [  0,   1,   2,   1,   0],
            [  1,   2,  -1,   2,   1]
        ]
        nearest_gate(grid)
        self.assertEqual(expected_grid, grid)

    def test_at_center(self):
        grid = [
            [INF, INF, INF, INF, INF],
            [INF, INF, INF, INF, INF],
            [INF, INF,   0, INF, INF],
            [INF, INF, INF, INF, INF]
        ]
        expected_grid = [
            [  4,   3,   2,   3,   4],
            [  3,   2,   1,   2,   3],
            [  2,   1,   0,   1,   2],
            [  3,   2,   1,   2,   3]
        ]
        nearest_gate(grid)
        self.assertEqual(expected_grid, grid)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 22, 2020 LC 212 \[Hard\] Word Search II
---
> **Question:** Given an m x n board of characters and a list of strings words, return all words on the board.
>
> Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.
 
**Example 1:**
```py
Input: words = ["oath","pea","eat","rain"], board = [
    ["o","a","a","n"],
    ["e","t","a","e"],
    ["i","h","k","r"],
    ["i","f","l","v"]]
Output: ["eat", "oath"]
```

**Example 2:**
```py
Input: words = ["abcb"], board = [
    ["a","b"],
    ["c","d"]]
Output: []
```


**Solution with Backtracking and Trie:** [https://repl.it/@trsong/Word-Search-II](https://repl.it/@trsong/Word-Search-II)
```py
import unittest

def search_word(board, words):
    if not board or not board[0] or not words:
        return []

    trie = Trie()
    for word in words:
        trie.insert(word)

    res = []
    n, m = len(board), len(board[0])
    for r in xrange(n):
        for c in xrange(m):
            backtrack(board, trie, (r, c), res)

    return res


class Trie(object):
    def __init__(self):
        self.word = None
        self.children = None
    
    def insert(self, word):
        p = self
        for ch in word:
            if not p.children:
                p.children =  {}
            
            if ch not in p.children:
                p.children[ch] = Trie()

            p = p.children[ch]
        p.word = word

DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]          

def backtrack(board, parent_node, pos, res):
    r, c = pos
    ch = board[r][c]
    if not ch or not parent_node.children or not parent_node.children.get(ch, None):
        return
    
    node = parent_node.children[ch]
    if node.word:
        res.append(node.word)
        node.word = None
    
    board[r][c] = None
    n, m = len(board), len(board[0])
    for dr, dc in DIRECTIONS:
        new_r, new_c = r + dr, c + dc
        if 0 <= new_r < n and 0 <= new_c < m and board[new_r][new_c]:
            backtrack(board, node, (new_r, new_c), res)
    board[r][c] = ch


class SearchWordSpec(unittest.TestCase):
    def assert_result(self, expected, res):
        self.assertEqual(sorted(expected), sorted(res))

    def test_example(self):
        words = ['oath','pea','eat','rain']
        board = [
            ['o','a','a','n'],
            ['e','t','a','e'],
            ['i','h','k','r'],
            ['i','f','l','v']]
        expected = ['eat', 'oath']
        self.assert_result(expected, search_word(board, words))

    def test_example2(self):
        words = ['abcb']
        board = [
            ['a','b'],
            ['c','d']]
        expected = []
        self.assert_result(expected, search_word(board, words))

    def test_unique_char(self):
        words = ['a', 'aa', 'aaa']
        board = [
            ['a','a'],
            ['a','a']]
        expected = ['a', 'aa', 'aaa']
        self.assert_result(expected, search_word(board, words))

    def test_empty_grid(self):
        self.assertEqual([], search_word([], ['a']))

    def test_empty_empty_word(self):
        self.assertEqual([], search_word(['a'], []))

    def test_word_use_all_letters(self):
        words = ['abcdef']
        board = [
            ['a','b'],
            ['f','c'],
            ['e','d']]
        expected = ['abcdef']
        self.assert_result(expected, search_word(board, words))

    
if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 21, 2020 \[Hard\] Find the Element That Appears Once While Others Occur 3 Times
---
> **Question:** Given an array of integers where every integer occurs three times except for one integer, which only occurs once, find and return the non-duplicated integer.
>
> For example, given `[6, 1, 3, 3, 3, 6, 6]`, return `1`. Given `[13, 19, 13, 13]`, return `19`.
>
> Do this in `O(N)` time and `O(1)` space.

**My thoughts:** An interger repeats 3 time, then each of its digit will repeat 3 times. If a digit repeat 1 more time on top of that, then that digit must be contributed by the unique number. 

**Solution:** [https://repl.it/@trsong/Find-the-Element-That-Appears-Once-While-Others-Occur-3-Time](https://repl.it/@trsong/Find-the-Element-That-Appears-Once-While-Others-Occur-3-Time)
```py
import unittest

INT_SIZE = 32

def find_uniq_elem(nums):
    res = 0
    count = 0
    for i in xrange(INT_SIZE):
        count = 0
        for num in nums:
            if num & 1 << i:
                count += 1

        if count % 3 == 1:
            res |= 1 << i 
    return res


class FindUniqElemSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(1, find_uniq_elem([6, 1, 3, 3, 3, 6, 6]))

    def test_example2(self):
        self.assertEqual(19, find_uniq_elem([13, 19, 13, 13]))

    def test_example3(self):
        self.assertEqual(2, find_uniq_elem([12, 1, 12, 3, 12, 1, 1, 2, 3, 3]))

    def test_example4(self):
        self.assertEqual(20, find_uniq_elem([10, 20, 10, 30, 10, 30, 30]))

    def test_ascending_array(self):
        self.assertEqual(4, find_uniq_elem([1, 1, 1, 2, 2, 2, 3, 3, 3, 4]))

    def test_descending_array(self):
        self.assertEqual(2, find_uniq_elem([2, 1, 1, 1, 0, 0, 0]))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 20, 2020 \[Easy\] Word Ordering in a Different Alphabetical Order
---
> **Question:** Given a list of words, and an arbitrary alphabetical order, verify that the words are in order of the alphabetical order.

**Example 1:**
```py
Input: 
words = ["abcd", "efgh"]
order="zyxwvutsrqponmlkjihgfedcba"

Output: False
Explanation: 'e' comes before 'a' so 'efgh' should come before 'abcd'
```

**Example 2:**
```py
Input:
words = ["zyx", "zyxw", "zyxwy"]
order="zyxwvutsrqponmlkjihgfedcba"

Output: True
Explanation: The words are in increasing alphabetical order
```

**Solution:** [https://repl.it/@trsong/Determine-Word-Ordering-in-a-Different-Alphabetical-Order](https://repl.it/@trsong/Determine-Word-Ordering-in-a-Different-Alphabetical-Order)
```py
import unittest

def is_sorted(words, order):
    char_rank = {ch: rank for rank, ch in enumerate(order)}
    prev = ""
    for word in words:
        is_smaller = False
        for ch1, ch2 in zip(prev, word):
            if char_rank[ch1] > char_rank[ch2]:
                return False

            if char_rank[ch1] < char_rank[ch2]:
                is_smaller = True
                break

        if not is_smaller and len(prev) > len(word):
            return False

        prev = word
    return True


class IsSortedSpec(unittest.TestCase):
    def test_example1(self):
        words = ["abcd", "efgh"]
        order = "zyxwvutsrqponmlkjihgfedcba"
        self.assertFalse(is_sorted(words, order))

    def test_example2(self):
        words = ["zyx", "zyxw", "zyxwy"]
        order = "zyxwvutsrqponmlkjihgfedcba"
        self.assertTrue(is_sorted(words, order))

    def test_empty_list(self):
        self.assertTrue(is_sorted([], ""))
        self.assertTrue(is_sorted([], "abc"))

    def test_one_elem_list(self):
        self.assertTrue(is_sorted(["z"], "xyz"))

    def test_empty_words(self):
        self.assertTrue(is_sorted(["", "", ""], ""))

    def test_word_of_different_length(self):
        words = ["", "1", "11", "111", "1111"]
        order = "4321"
        self.assertTrue(is_sorted(words, order))

    def test_word_of_different_length2(self):
        words = ["", "11", "", "111", "1111"]
        order = "1"
        self.assertFalse(is_sorted(words, order))

    def test_large_word_dictionary(self):
        words = ["123", "1a1b1A2ca", "ABC", "Aaa", "aaa", "bbb", "c11", "cCa"]
        order = "".join(map(chr, range(256)))
        self.assertTrue(is_sorted(words, order))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 19, 2020 \[Easy\] Count Visible Nodes in Binary Tree
---
> **Question:** In a binary tree, if in the path from root to the node A, there is no node with greater value than A’s, this node A is visible. We need to count the number of visible nodes in a binary tree.

**Example 1:**
```py
Input:
        5
     /     \
   3        10
  /  \     /
20   21   1

Output: 4
Explanation: There are 4 visible nodes: 5, 20, 21, and 10.
```

**Example 2:**
```py
Input:
  -10
    \
    -15
      \
      -1

Output: 2
Explanation: Visible nodes are -10 and -1.
```

**Solution with DFS:** [https://repl.it/@trsong/Count-Visible-Nodes-in-Binary-Tree](https://repl.it/@trsong/Count-Visible-Nodes-in-Binary-Tree)
```py
import unittest

def count_visible_nodes(root):
    if not root:
        return 0

    stack = [(root, float('-inf'))]
    res = 0
    while stack:
        cur, prev_max = stack.pop()
        if cur.val > prev_max:
            res += 1
        
        for child in [cur.left, cur.right]:
            if not child:
                continue
            stack.append((child, max(prev_max, cur.val)))
    
    return res


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class CountVisibleNodeSpec(unittest.TestCase):
    def test_example(self):
        """
                5*
             /     \
           3        10*
          /  \     /
        20*  21*  1
        """
        left_tree = TreeNode(3, TreeNode(20), TreeNode(21))
        right_tree = TreeNode(10, TreeNode(1))
        root = TreeNode(5, left_tree, right_tree)
        self.assertEqual(4, count_visible_nodes(root))

    def test_example2(self):
        """
         -10*
           \
           -15
             \
             -1*
        """
        root = TreeNode(-10, right=TreeNode(-15, TreeNode(-1)))
        self.assertEqual(2, count_visible_nodes(root))

    def test_empty_tree(self):
        self.assertEqual(0, count_visible_nodes(None))

    def test_full_tree(self):
        """
             1*
           /    \
          2*     3*
         / \    / \
        4*  5* 6*  7*
        """
        left_tree = TreeNode(2, TreeNode(4), TreeNode(5))
        right_tree = TreeNode(3, TreeNode(6), TreeNode(7))
        root = TreeNode(1, left_tree, right_tree)
        self.assertEqual(7, count_visible_nodes(root))

    def test_one_node_tree(self):
        self.assertEqual(1, count_visible_nodes(TreeNode(42)))

    def test_complete_tree(self):
        """
            10*
           /   \
          9     8
         / \   /
        7   6 5
        """
        left_tree = TreeNode(9, TreeNode(7), TreeNode(6))
        right_tree = TreeNode(8, TreeNode(5))
        root = TreeNode(10, left_tree, right_tree)
        self.assertEqual(1, count_visible_nodes(root))
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 18, 2020 \[Hard\] Non-adjacent Subset Sum
---
> **Question:** Given an array of size n with unique positive integers and a positive integer K,
check if there exists a combination of elements in the array satisfying both of below constraints:
> - The sum of all such elements is K
> - None of those elements are adjacent in the original array

**Example:**
```py
Input: K = 14, arr = [1, 9, 8, 3, 6, 7, 5, 11, 12, 4]
Output: [3, 7, 4]
```

**Solution with DP:** [https://repl.it/@trsong/Non-adjacent-Subset-Sum](https://repl.it/@trsong/Non-adjacent-Subset-Sum)
```py
import unittest

def non_adj_subset_sum(nums, k):
    if not nums:
        return None

    n = len(nums)
    # Let dp[k][n] represents whether exists subset sum k for nums[:n]
    # dp[k][n] = dp[k][n-1] or dp[k - nums[n-1]][n-2]
    dp = [[False for _ in xrange(n + 1)] for _ in xrange(k + 1)]
    for s in xrange(1, k+1):
        for i in xrange(1, n+1):
            if s == nums[i-1] or dp[s][i-1]:
                dp[s][i] = True
                continue
            if i > 2 and s >= nums[i-1]:
                dp[s][i] = dp[s - nums[i-1]][i-2]

    if not dp[k][n]:
        return None
    res = []
    for i in xrange(n, 0, -1):
        if dp[k][i-1]:
            continue

        num = nums[i-1]
        if k == num or i > 2 and dp[k - num][i - 2]:
            res.append(num)
            k -= num
    return res[::-1]
            

class NonAdjSubsetSumSpec(unittest.TestCase):
    def assert_result(self, k, nums, res):
        self.assertEqual(set(), set(res) - set(nums))
        self.assertEqual(k, sum(res))

    def test_example(self):
        k, nums = 14, [1, 9, 8, 3, 6, 7, 5, 11, 12, 4]
        # Possible solution [3, 7, 4]
        res = non_adj_subset_sum(nums, k) 
        self.assert_result(k, nums, res)

    def test_multiple_solution(self):
        k, nums = 12, [1, 2, 3, 4, 5, 6, 7]
        # Possible solution [2, 4, 6]
        res = non_adj_subset_sum(nums, k)
        self.assert_result(k, nums, res)

    def test_no_subset_satisfied(self):
        k, nums = 100, [1, 2]
        self.assertIsNone(non_adj_subset_sum(nums, k))

    def test_no_subset_satisfied2(self):
        k, nums = 3, [1, 2]
        self.assertIsNone(non_adj_subset_sum(nums, k))

    def test_should_not_pick_adjacent_elements(self):
        k, nums = 3, [1, 2, 3]
        expected = [3]
        self.assertEqual(expected, non_adj_subset_sum(nums, k))

    def test_should_not_pick_adjacent_elements2(self):
        k, nums = 4, [1, 2, 3]
        expected = [1, 3]
        self.assertEqual(expected, non_adj_subset_sum(nums, k))

    def test_pick_every_other_elements(self):
        k, nums = 11, [1, 90, 2, 80, 3, 100, 5]
        expected = [1, 2, 3, 5]
        self.assertEqual(expected, non_adj_subset_sum(nums, k))

    def test_pick_first_and_last(self):
        k, nums = 3, [1, 10, 11, 7, 4, 12, 2]
        expected = [1, 2]
        self.assertEqual(expected, non_adj_subset_sum(nums, k))

    def test_pick_every_three_elements(self):
        k, nums = 6, [1, 100, 109, 2, 101, 110, 3]
        expected = [1, 2, 3]
        self.assertEqual(expected, non_adj_subset_sum(nums, k))
  

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 17, 2020 \[Easy\] Count Number of Unival Subtrees
---
> **Question:** A unival tree is a tree where all the nodes have the same value. Given a binary tree, return the number of unival subtrees in the tree.

**Example 1:**
```py
The following tree should return 5:

   0
  / \
 1   0
    / \
   1   0
  / \
 1   1

The 5 trees are:
- The three single '1' leaf nodes. (+3)
- The single '0' leaf node. (+1)
- The [1, 1, 1] tree at the bottom. (+1)
```

**Example 2:**
```py
Input: root of below tree
              5
             / \
            1   5
           / \   \
          5   5   5
Output: 4
There are 4 subtrees with single values.
```

**Example 3:**
```py
Input: root of below tree
              5
             / \
            4   5
           / \   \
          4   4   5                
Output: 5
There are five subtrees with single values.
```

**Solution with Postorder Traversal:** [https://repl.it/@trsong/Count-Total-Number-of-Uni-val-Subtrees](https://repl.it/@trsong/Count-Total-Number-of-Uni-val-Subtrees)
```py
import unittest

class Result(object):
    def __init__(self, res, is_unival):
        self.res = res
        self.is_unival = is_unival


def count_unival_subtrees(tree):
    return count_unival_subtrees_recur(tree).res


def count_unival_subtrees_recur(tree):
    if not tree:
        return Result(0, True)

    left_res = count_unival_subtrees_recur(tree.left)
    right_res = count_unival_subtrees_recur(tree.right)
    is_current_unival = left_res.is_unival and left_res.is_unival
    if is_current_unival and tree.left and tree.left.val != tree.val:
        is_current_unival = False
    if is_current_unival and tree.right and tree.right.val != tree.val:
        is_current_unival = False

    current_count = left_res.res + right_res.res + (1 if is_current_unival else 0)
    return Result(current_count, is_current_unival)


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class CountUnivalSubTreeSpec(unittest.TestCase):
    def test_example1(self):
        """
           0
          / \
         1   0
            / \
           1   0
          / \
         1   1
        """
        rl = TreeNode(1, TreeNode(1), TreeNode(1))
        r = TreeNode(0, rl, TreeNode(0))
        root = TreeNode(0, TreeNode(1), r)
        self.assertEqual(5, count_unival_subtrees(root))

    def test_example2(self):
        """
              5
             / \
            1   5
           / \   \
          5   5   5
        """
        l = TreeNode(1, TreeNode(5), TreeNode(5))
        r = TreeNode(5, right=TreeNode(5))
        root = TreeNode(5, l, r)
        self.assertEqual(4, count_unival_subtrees(root))

    def test_example3(self):
        """
              5
             / \
            4   5
           / \   \
          4   4   5  
        """
        l = TreeNode(4, TreeNode(4), TreeNode(4))
        r = TreeNode(5, right=TreeNode(5))
        root = TreeNode(5, l, r)
        self.assertEqual(5, count_unival_subtrees(root))

    def test_empty_tree(self):
        self.assertEqual(0, count_unival_subtrees(None))

    def test_left_heavy_tree(self):
        """
            1
           /
          1
         / \ 
        1   0
        """
        root = TreeNode(1, TreeNode(1, TreeNode(1), TreeNode(0)))
        self.assertEqual(2, count_unival_subtrees(root))

    def test_right_heavy_tree(self):
        """
          0
         / \
        1   0
             \
              0
               \
                0
        """
        rr = TreeNode(0, right=TreeNode(0))
        r = TreeNode(0, right=rr)
        root = TreeNode(0, TreeNode(1), r)
        self.assertEqual(4, count_unival_subtrees(root))

    def test_unival_tree(self):
        """
            0
           / \
          0   0
         /   /
        0   0          
        """
        l = TreeNode(0, TreeNode(0))
        r = TreeNode(0, TreeNode(0))
        root = TreeNode(0, l, r)
        self.assertEqual(5, count_unival_subtrees(root))

    def test_distinct_value_trees(self):
        """
               _0_
              /   \
             1     2
            / \   / \
           3   4 5   6
          /
         7
        """
        n1 = TreeNode(1, TreeNode(3, TreeNode(7)), TreeNode(4))
        n2 = TreeNode(2, TreeNode(5), TreeNode(6))
        n0 = TreeNode(0, n1, n2)
        self.assertEqual(4, count_unival_subtrees(n0))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 16, 2020 LC 1647 \[Medium\] Minimum Deletions to Make Character Frequencies Unique
---
> **Question:** A string s is called good if there are no two different characters in s that have the same frequency.
>
> Given a string s, return the minimum number of characters you need to delete to make s good.
>
> The frequency of a character in a string is the number of times it appears in the string. For example, in the string "aab", the frequency of 'a' is 2, while the frequency of 'b' is 1.


**Example 1:**
```py
Input: s = "aab"
Output: 0
Explanation: s is already good.
```
**Example 2:**
```py
Input: s = "aaabbbcc"
Output: 2
Explanation: You can delete two 'b's resulting in the good string "aaabcc".
Another way it to delete one 'b' and one 'c' resulting in the good string "aaabbc".
```
**Example 3:**
```py
Input: s = "ceabaacb"
Output: 2
Explanation: You can delete both 'c's resulting in the good string "eabaab".
Note that we only care about characters that are still in the string at the end (i.e. frequency of 0 is ignored).
```

**My thoughts:** sort frequency in descending order, while iterate through all frequencies, keep track of biggest next frequency we can take. Then the min deletion for that letter is `freq - biggestNextFreq`. Remember to reduce the biggest next freq by 1 for each step.  

**Greedy Solution**: [https://repl.it/@trsong/Minimum-Deletions-to-Make-Character-Frequencies-Unique](https://repl.it/@trsong/Minimum-Deletions-to-Make-Character-Frequencies-Unique)
```py
import unittest

CHAR_SIZE = 26

def min_deletions(s):
    histogram = [0] * CHAR_SIZE
    ord_a = ord('a')
    for ch in s:
        histogram[ord(ch) - ord_a] += 1

    histogram.sort(reverse=True)
    next_count = histogram[0]
    res = 0

    for count in histogram:
        if count <= next_count:
            next_count = count - 1
        else:
            # reduce count to next_count
            res += count - next_count
            next_count -= 1    
        next_count = max(0, next_count)
    return res


class MinDeletionSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(0, min_deletions("aab"))
        
    def test_example2(self):
        # remove 2b's
        self.assertEqual(2, min_deletions("aaabbbcc"))
        
    def test_example3(self):
        # remove 2b's
        self.assertEqual(2, min_deletions("ceabaacb"))
        
    def test_empty_string(self):
        self.assertEqual(0, min_deletions(""))
        
    def test_string_with_same_char_freq(self):
        s = 'a' * 100 + 'b' * 100 + 'c' * 2 + 'd' * 1
        self.assertEqual(1, min_deletions(s))
        
    def test_remove_all_other_string(self):
        self.assertEqual(4, min_deletions("abcde"))
        
    def test_collision_after_removing(self):
        # remove 1b, 1c, 2d, 2e, 1f 
        s = 'a' * 10 + 'b' * 10 + 'c' * 9 + 'd' * 9 + 'e' * 8 + 'f' * 6
        self.assertEqual(7, min_deletions(s))

    def test_remove_all_of_certain_letters(self):
        # remove 3b, 1f
        s = 'a' * 3 + 'b' * 3 + 'c' * 2 + 'd' + 'f' 
        self.assertEqual(4, min_deletions(s))
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 15, 2020 \[Medium\] Number of Flips to Make Binary String
---
> **Question:** You are given a string consisting of the letters `x` and `y`, such as `xyxxxyxyy`. In addition, you have an operation called flip, which changes a single `x` to `y` or vice versa.
>
> Determine how many times you would need to apply this operation to ensure that all x's come before all y's. In the preceding example, it suffices to flip the second and sixth characters, so you should return 2.

**My thoughts:** Basically, the question is about finding a sweet cutting spot so that # flip on left plus # flip on right is minimized. We can simply scan through the array from left and right to allow constant time query for number of flip need on the left and right for a given spot. And the final answer is just the min of sum of left and right flips.


**Solution with DP:** [https://repl.it/@trsong/Find-Number-of-Flips-to-Make-Binary-String](https://repl.it/@trsong/Find-Number-of-Flips-to-Make-Binary-String)
```py
import unittest

def min_flip_to_make_binary(s):
    if not s:
        return 0

    n = len(s)
    left_y_count = 0 
    right_x_count = 0
    left_accu = [0] * n
    right_accu = [0] * n
    for i in xrange(n):
        left_accu[i] = left_y_count
        left_y_count += 1 if s[i] == 'y' else 0

        right_accu[n - 1 - i] = right_x_count
        right_x_count += 1 if s[n - 1 - i] == 'x' else 0
    
    res = float('inf')
    for left_y, right_x in zip(left_accu, right_accu):
        res = min(res, left_y + right_x)

    return res


class MinFlipToMakeBinarySpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(2, min_flip_to_make_binary('xyxxxyxyy'))  # xxxxxxxyy

    def test_empty_string(self):
        self.assertEqual(0, min_flip_to_make_binary(''))
    
    def test_already_binary(self):
        self.assertEqual(0, min_flip_to_make_binary('xxxxxxxyy'))

    def test_flipped_string(self):
        self.assertEqual(3, min_flip_to_make_binary('yyyxxx'))  # yyyyyy

    def test_flip_all_x(self):
        self.assertEqual(1, min_flip_to_make_binary('yyx'))  # yyy

    def test_flip_all_y(self):
        self.assertEqual(2, min_flip_to_make_binary('yyxxx'))  # xxxxx

    def test_flip_all_y2(self):
        self.assertEqual(4, min_flip_to_make_binary('xyxxxyxyyxxx'))  # xxxxxxxxxxxx

    def test_flip_y(self):
        self.assertEqual(2, min_flip_to_make_binary('xyxxxyxyyy'))  # xxxxxxxyyy

    def test_flip_x_and_y(self):
        self.assertEqual(3, min_flip_to_make_binary('xyxxxyxyyx'))  # xxxxxyyyyy


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 14, 2020 \[Medium\] Isolated Islands
---
> **Question:** Given a matrix of 1s and 0s, return the number of "islands" in the matrix. A 1 represents land and 0 represents water, so an island is a group of 1s that are neighboring whose perimeter is surrounded by water.
>
> For example, this matrix has 4 islands.

```py
1 0 0 0 0
0 0 1 1 0
0 1 1 0 0
0 0 0 0 0
1 1 0 0 1
1 1 0 0 1
```

**Solution with DFS:** [https://repl.it/@trsong/Count-Number-of-Isolated-Islands](https://repl.it/@trsong/Count-Number-of-Isolated-Islands)
```py
import unittest

DIRECTIONS = [-1, 0, 1]

def calc_islands(area_map):
    if not area_map or not area_map[0]:
        return 0

    n, m = len(area_map), len(area_map[0])
    visited = set()
    res = 0
    for r in xrange(n):
        for c in xrange(m):
            if area_map[r][c] == 0 or (r, c) in visited:
                continue
            res += 1
            dfs_island(area_map, (r, c), visited)
    return res


def dfs_island(area_map, pos, visited):
    n, m = len(area_map), len(area_map[0])
    stack = [pos]
    while stack:
        cur_r, cur_c = stack.pop()
        if (cur_r, cur_c) in visited:
            continue
        visited.add((cur_r, cur_c))
            
        for dr in DIRECTIONS:
            for dc in DIRECTIONS:
                new_r, new_c = cur_r + dr, cur_c + dc
                if (0 <= new_r < n and 0 <= new_c < m and 
                    area_map[new_r][new_c] == 1 and 
                    (new_r, new_c) not in visited):
                    stack.append((new_r, new_c))


class CalcIslandSpec(unittest.TestCase):
    def test_sample_area_map(self):
        self.assertEqual(4, calc_islands([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1]
        ]))
    
    def test_some_random_area_map(self):
        self.assertEqual(5, calc_islands([
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1] 
        ]))

    def test_island_edge_of_map(self):
        self.assertEqual(5, calc_islands([
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1] 
        ]))

    def test_huge_water(self):
        self.assertEqual(0, calc_islands([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]))

    def test_huge_island(self):
        self.assertEqual(1, calc_islands([
            [1, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
            [1, 1, 1, 0, 1]
        ]))

    def test_non_square_island(self):
        self.assertEqual(1, calc_islands([
            [1],
            [1],
            [1]
        ]))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 13, 2020 \[Medium\] Find Missing Positive
---
> **Question:** Given an unsorted integer array, find the first missing positive integer.

**Example 1:**
```py
Input: [1, 2, 0]
Output: 3
```

**Example 2:**
```py
Input: [3, 4, -1, 1]
Output: 2
```

**My thougths:** Ideally each positive number should map to the same index as its `value - 1`. So all we need to do is for each postion, use its value as index and swap with that element until we find correct number. Keep doing this and each postive should store in postion of its `value - 1`.  Now we just scan through the entire array until find the first missing number by checking each element's value against index. 


**Soluion:** [https://repl.it/@trsong/Find-First-Missing-Positive](https://repl.it/@trsong/Find-First-Missing-Positive)
```py
import unittest

def find_missing_positive(nums):
    n = len(nums)
    value_to_index = lambda value: value - 1

    for i in xrange(n):
        while 1 <= nums[i] <= n and value_to_index(nums[i]) != i:
            target_index = value_to_index(nums[i])
            if nums[i] == nums[target_index]:
                break
            nums[i], nums[target_index] = nums[target_index], nums[i]

    for i, num in enumerate(nums):
        if value_to_index(num) != i:
            return i + 1

    return n + 1
    

class FindMissingPositiveSpec(unittest.TestCase):
    def test_example(self):
        nums = [1, 2, 0]
        expected = 3
        self.assertEqual(expected, find_missing_positive(nums))

    def test_example2(self):
        nums = [3, 4, -1, 1]
        expected = 2
        self.assertEqual(expected, find_missing_positive(nums))
    
    def test_empty_array(self):
        nums = []
        expected = 1
        self.assertEqual(expected, find_missing_positive(nums))

    def test_all_non_positives(self):
        nums = [-1, 0, -1, -2, -1, -3, -4]
        expected = 1
        self.assertEqual(expected, find_missing_positive(nums))

    def test_number_out_of_range(self):
        nums = [101, 102, 103]
        expected = 1
        self.assertEqual(expected, find_missing_positive(nums))

    def test_duplicated_numbers(self):
        nums = [1, 1, 3, 3, 2, 2, 5]
        expected = 4
        self.assertEqual(expected, find_missing_positive(nums))

    def test_missing_positive_falls_out_of_range(self):
        nums = [5, 4, 3, 2, 1]
        expected = 6
        self.assertEqual(expected, find_missing_positive(nums))

    def test_number_off_by_one_position(self):
        nums = [0, 2, 3, 4, 7, 6, 1]
        expected = 5
        self.assertEqual(expected, find_missing_positive(nums))

    def test_positive_and_negative_numbers(self):
        nums = [-1, -3, -2, 0, 1, 2, 4, -4, 5, -6, 7]
        expected = 3
        self.assertEqual(expected, find_missing_positive(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 12, 2020 \[Medium\] Max Value of Coins to Collect in a Matrix
---
> **Question:** You are given a 2-d matrix where each cell represents number of coins in that cell. Assuming we start at `matrix[0][0]`, and can only move right or down, find the maximum number of coins you can collect by the bottom right corner.

**Example:**

```py
Given below matrix:

0 3 1 1
2 0 0 4
1 5 3 1

The most we can collect is 0 + 2 + 1 + 5 + 3 + 1 = 12 coins.
```

**My thoughts:** This problem gives you a strong feeling that this must be a DP question. 'coz for each step you can either move right or down, that is, max number of coins you can collect so far at current cell depends on top and left solution gives the following recurrence formula: 

```py
Let dp[i][j] be the max coin value collect when reach cell (i, j) in grid.
dp[i][j] = grid[i][j] + max(dp[i-1][j], dp[i][j-1])
```

You can also do it in-place using original grid. However, mutating input params in general is a bad habit as those parameters may be used in other place and might be immutable.


**Solution with DP:** [https://repl.it/@trsong/Find-Max-Value-of-Coins-to-Collect-in-a-Matrix](https://repl.it/@trsong/Find-Max-Value-of-Coins-to-Collect-in-a-Matrix)
```py
import unittest

def max_coins(grid):
    if not grid or not grid[0]:
        return 0

    n, m = len(grid), len(grid[0])
    dp = [[0 for _ in xrange(m)] for _ in xrange(n)]

    for r in xrange(n):
        for c in xrange(m):
            left_max = dp[r-1][c] if r > 0 else 0
            top_max = dp[r][c-1] if c > 0 else 0
            dp[r][c] = grid[r][c] + max(left_max, top_max)

    return dp[n-1][m-1]


class MaxCoinSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(max_coins([
            [0, 3, 1, 1],
            [2, 0, 0, 4],
            [1, 5, 3, 1]
        ]), 12)

    def test_empty_grid(self):
        self.assertEqual(max_coins([]), 0)
        self.assertEqual(max_coins([[]]), 0)

    def test_one_way_or_the_other(self):
        self.assertEqual(max_coins([
            [0, 3],
            [2, 0]
        ]), 3)

    def test_until_the_last_moment_knows(self):
        self.assertEqual(max_coins([
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [2, 0, 3, 0]
        ]), 5)

    def test_try_to_get_most_coins(self):
        self.assertEqual(max_coins([
            [1, 1, 1],
            [2, 3, 1],
            [1, 4, 5]
        ]), 15)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 11, 2020 LC 859 \[Easy\] Buddy Strings
---
> **Question:** Given two strings A and B of lowercase letters, return true if and only if we can swap two letters in A so that the result equals B.

**Example 1:**
```py
Input: A = "ab", B = "ba"
Output: true
```

**Example 2:**
```py
Input: A = "ab", B = "ab"
Output: false
```

**Example 3:**
```py
Input: A = "aa", B = "aa"
Output: true
```

**Example 4:**
```py
Input: A = "aaaaaaabc", B = "aaaaaaacb"
Output: true
```

**Example 5:**
```py
Input: A = "", B = "aa"
Output: false
```

**Solution:** [https://repl.it/@trsong/Buddy-Strings](https://repl.it/@trsong/Buddy-Strings)
```py
import unittest

def is_buddy_string(s1, s2):
    if len(s1) != len(s2):
        return False

    unmatches = []
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            unmatches.append((c1, c2))
        if len(unmatches) > 2:
            return False

    if len(unmatches) == 2:
        return unmatches[0] == unmatches[1][::-1]
    elif len(unmatches) == 0:
        return has_duplicates(s1)
    else:
        return False 


def has_duplicates(s):
    visited = set()
    for c in s:
        if c in visited:
            return True
        visited.add(c)
    return False


class IsBuddyString(unittest.TestCase):
    def test_example(self):
        self.assertTrue(is_buddy_string('ab', 'ba'))

    def test_example2(self):
        self.assertFalse(is_buddy_string('ab', 'ab'))

    def test_example3(self):
        self.assertTrue(is_buddy_string('aa', 'aa'))

    def test_example4(self):
        self.assertTrue(is_buddy_string('aaaaaaabc', 'aaaaaaacb'))

    def test_example5(self):
        self.assertFalse(is_buddy_string('ab', 'aa'))

    def test_empty_string(self):
        self.assertFalse(is_buddy_string('', ''))

    def test_single_char_string(self):
        self.assertFalse(is_buddy_string('a', 'b'))

    def test_same_string_without_duplicates(self):
        self.assertFalse(is_buddy_string('abc', 'abc'))

    def test_string_with_duplicates(self):
        self.assertFalse(is_buddy_string('aba', 'abc'))

    def test_different_length_string(self):
        self.assertFalse(is_buddy_string('aa', 'aaa'))

    def test_different_length_string2(self):
        self.assertFalse(is_buddy_string('ab', 'baa'))

    def test_different_length_string3(self):
        self.assertFalse(is_buddy_string('ab', 'abba'))

    def test_swap_failure(self):
        self.assertFalse(is_buddy_string('abcaa', 'abcbb'))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 10, 2020 \[Medium\] All Root to Leaf Paths in Binary Tree
---
> **Question:** Given a binary tree, return all paths from the root to leaves.

**Example:** 
```py
Given the tree:
   1
  / \
 2   3
    / \
   4   5

Return [[1, 2], [1, 3, 4], [1, 3, 5]]
```

**Solution with Backtracking:** [https://repl.it/@trsong/Print-All-Root-to-Leaf-Paths-in-Binary-Tree](https://repl.it/@trsong/Print-All-Root-to-Leaf-Paths-in-Binary-Tree)
```py
import unittest

def path_to_leaves(tree):
    if not tree:
        return []
    res = []
    backtrack(res, tree, [])
    return res


def backtrack(res, current_node, path):
    if not current_node.left and not current_node.right:
        res.append(path + [current_node.val])
    else:
        for child in [current_node.left, current_node.right]:
            if not child:
                continue
            path.append(current_node.val)
            backtrack(res, child, path)
            path.pop()

    
class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right= right


class PathToLeavesSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertEqual([], path_to_leaves(None))

    def test_one_level_tree(self):
        self.assertEqual([[1]], path_to_leaves(TreeNode(1)))

    def test_two_level_tree(self):
        """
          1
         / \
        2   3
        """
        tree = TreeNode(1, TreeNode(2), TreeNode(3))
        expected = [[1, 2], [1, 3]]
        self.assertEqual(expected, path_to_leaves(tree))

    def test_example(self):
        """
          1
         / \
        2   3
           / \
          4   5
        """
        n3 = TreeNode(3, TreeNode(4), TreeNode(5))
        tree = TreeNode(1, TreeNode(2), n3)
        expected = [[1, 2], [1, 3, 4], [1, 3, 5]]
        self.assertEqual(expected, path_to_leaves(tree))

    def test_complete_tree(self):
        """
               1
             /   \
            2     3
           / \   /
          4   5 6
        """
        left_tree = TreeNode(2, TreeNode(4), TreeNode(5))
        right_tree = TreeNode(3, TreeNode(6))
        tree = TreeNode(1, left_tree, right_tree)
        expected = [[1, 2, 4], [1, 2, 5], [1, 3, 6]]
        self.assertEqual(expected, path_to_leaves(tree))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 9, 2020 \[Medium\] Invert a Binary Tree
---
> **Question:** You are given the root of a binary tree. Invert the binary tree in place. That is, all left children should become right children, and all right children should become left children.

**Example:**
```py
Given the following tree:

     a
   /   \
  b     c
 / \   /
d   e f

should become:
   a
 /   \
c     b
 \   / \
  f e   d
```

**Solution:** [https://repl.it/@trsong/Invert-All-Nodes-in-Binary-Tree](https://repl.it/@trsong/Invert-All-Nodes-in-Binary-Tree)
```py
import unittest

def invert_tree(root):
    if root:
        root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __eq__(self, other):
        return (other and
            self.val == other.val and 
            self.left == other.left and 
            self.right == other.right)

    def __repr__(self):
        stack = [(self, 0)]
        res = []
        while stack:
            node, depth = stack.pop()
            res.append("\n" + "\t" * depth)
            if not node:
                res.append("* None")
                continue

            res.append("* " + str(node.val))
            for child in [node.right, node.left]:
                stack.append((child, depth+1))
        return "\n" + "".join(res) + "\n"


class InvertTreeSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertIsNone(invert_tree(None))

    def test_heavy_left_tree(self):
        """
            1
           /
          2
         /
        3
        """
        tree = TreeNode(1, TreeNode(2, TreeNode(3)))
        """
        1
         \
          2
           \
            3
        """
        expected_tree = TreeNode(1, right=TreeNode(2, right=TreeNode(3)))
        self.assertEqual(invert_tree(tree), expected_tree)

    def test_heavy_right_tree(self):
        """
          1
         / \
        2   3
           /
          4 
        """
        tree = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4)))
        """
          1
         / \
        3   2
         \ 
          4         
        """
        expected_tree = TreeNode(1, TreeNode(3, right=TreeNode(4)), TreeNode(2))
        self.assertEqual(invert_tree(tree), expected_tree)

    def test_sample_tree(self):
        """
             1
           /   \
          2     3
         / \   /
        4   5 6
        """
        n2 = TreeNode(2, TreeNode(4), TreeNode(5))
        n3 = TreeNode(3, TreeNode(6))
        n1 = TreeNode(1, n2, n3)
        """
            1
          /   \
         3     2
          \   / \
           6 5   4
        """
        en2 = TreeNode(2, TreeNode(5), TreeNode(4))
        en3 = TreeNode(3, right=TreeNode(6))
        en1 = TreeNode(1, en3, en2)
        self.assertEqual(invert_tree(n1), en1)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 8, 2020 \[Medium\] Similar Websites
---
> **Question:** You are given a list of (website, user) pairs that represent users visiting websites. Come up with a program that identifies the top k pairs of websites with the greatest similarity.
>
> **Note:** The similarity metric bewtween two sets equals intersection / union. 

**Example:**
```py
Suppose k = 1, and the list of tuples is:

[('a', 1), ('a', 3), ('a', 5),
 ('b', 2), ('b', 6),
 ('c', 1), ('c', 2), ('c', 3), ('c', 4), ('c', 5),
 ('d', 4), ('d', 5), ('d', 6), ('d', 7),
 ('e', 1), ('e', 3), ('e', 5), ('e', 6)]
 
Then a reasonable similarity metric would most likely conclude that a and e are the most similar, so your program should return [('a', 'e')].
```

**My thoughts:** The similarity metric bewtween two sets equals intersection / union. So, the way to get top k similar website is first calculate the similarity score between any two websites and after that use a priority queue to mantain top k similarity pairs.

However, as duplicate entry might occur, we have to treat normal set to multiset: treat `3, 3, 3` as `3(first), 3(second), 3(third)`. 
```py
a: 1 2 3(first) 3(second) 3(third)
b: 1 2 3(first)
The similarty between (a, b) is 3/5
```


**Solution with Priority Queue:** [https://repl.it/@trsong/Find-Similar-Websites](https://repl.it/@trsong/Find-Similar-Websites)
```py
import unittest
from collections import defaultdict
from Queue import PriorityQueue

def top_similar_websites(website_log, k):
    user_site_hits = defaultdict(lambda: defaultdict(int))
    site_hits = defaultdict(int)
    for site, user in website_log:
        user_site_hits[user][site] += 1
        site_hits[site] += 1

    cross_site_hits = defaultdict(lambda: defaultdict(int))
    for site_hits_per_user in user_site_hits.values():
        for site1 in site_hits_per_user:
            for site2 in site_hits_per_user:
                cross_site_hits[site1][site2] += min(site_hits_per_user[site1], site_hits_per_user[site2])

    min_heap = PriorityQueue()
    sites = sorted(site_hits.keys())
    for site1 in sites:
        for site2 in sites:
            if site2 == site1:
                break
            intersection = cross_site_hits[site1][site2]
            total = site_hits[site1] + site_hits[site2]
            union = total - intersection
            similarity = float(intersection) / union

            if min_heap.qsize() >= k and min_heap.queue[0][0] < similarity:
                min_heap.get()
            
            if min_heap.qsize() < k:
                min_heap.put((similarity, (site1, site2)))

    ascending_sites = [min_heap.get()[1] for _ in xrange(k)]
    return ascending_sites[::-1]


class TopSimilarWebsiteSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        # same length
        self.assertEqual(len(expected), len(result))
        for e, r in zip(expected, result):
            # pair must be the same, order doesn't matter
            self.assertEqual(set(e), set(r), "Expected %s but get %s" % (expected, result))

    def test_example(self):
        website_log = [
            ('a', 1), ('a', 3), ('a', 5),
            ('b', 2), ('b', 6),
            ('c', 1), ('c', 2), ('c', 3), ('c', 4), ('c', 5),
            ('d', 4), ('d', 5), ('d', 6), ('d', 7),
            ('e', 1), ('e', 3), ('e', 5), ('e', 6)]
        # Similarity: (a,e)=3/4, (a,c)=3/5, (c, e)=1/2
        expected = [('a', 'e'), ('a', 'c'), ('c', 'e')]
        self.assert_result(expected, top_similar_websites(website_log, len(expected)))

    def test_no_overlapping(self):
        website_log = [('a', 1), ('b', 2)]
        expected = [('a', 'b')]
        self.assert_result(expected, top_similar_websites(website_log, len(expected)))
    
    def test_should_return_correct_order(self):
        website_log = [
            ('a', 1),
            ('b', 1), ('b', 2),
            ('c', 1), ('c', 2), ('c', 3), 
            ('d', 1), ('d', 2), ('d', 3), ('d', 4),
            ('e', 1), ('e', 2), ('e', 3), ('e', 4), ('e', 5)]
        # Similarity: (d,e)=4/5, (c,d)=3/4, (b,c)=2/3, (c,e)=3/5
        expected = [('d', 'e'), ('c', 'd'), ('b', 'c'), ('c', 'e')]
        self.assert_result(expected, top_similar_websites(website_log, len(expected)))
        
    def test_duplicated_entries(self):
        website_log = [
            ('a', 1), ('a', 1),
            ('b', 1),
            ('c', 1), ('c', 1), ('c', 2),
            ('d', 1), ('d', 3), ('d', 3), ('d', 4),
            ('e', 1), ('e', 1), ('e', 5), ('e', 6),
            ('f', 1), ('f', 7), ('f', 8), ('f', 8)
        ]
        # Similarity: (a,c)=2/3
        expected = [('a', 'c')]
        self.assert_result(expected, top_similar_websites(website_log, len(expected)))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 7, 2020 \[Medium\] Largest Square
---
> **Question:** Given an N by M matrix consisting only of 1's and 0's, find the largest square matrix containing only 1's and return its dimension size.

**Example:**
```py
Given the following matrix:

[[1, 0, 0, 0],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [0, 1, 0, 0]]

Return 2. As the following 1s form the largest square matrix containing only 1s:
 [1, 1],
 [1, 1]
```

**Solution with DP:** [https://repl.it/@trsong/Largest-Square](https://repl.it/@trsong/Largest-Square)
```py
import unittest

def largest_square_dimension(grid):
    if not grid or not grid[0]:
        return 0
    n, m = len(grid), len(grid[0])

    # Let dp[r][c] represents max dimension of square matrix with bottom right corner at (r, c)
    # dp[r][c] = min(dp[r-1][c], dp[r][c-1], dp[r-1][c-1]) + 1 if (r, c) is 1
    #          = 0 otherwise
    dp = [[grid[r][c] for c in xrange(m)] for r in xrange(n)]

    for r in xrange(1, n):
        for c in xrange(1, m):
            if grid[r][c] == 0:
                continue
            dp[r][c] = 1 + min(dp[r-1][c], dp[r][c-1], dp[r-1][c-1])
    
    return max(map(max, dp))


class LargestSquareDimensionSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(2, largest_square_dimension([
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 0, 0]
        ]))

    def test_empty_grid(self):
        self.assertEqual(0, largest_square_dimension([]))

    def test_1d_grid(self):
        self.assertEqual(1, largest_square_dimension([
            [0, 0, 0, 0, 1, 0, 0]
        ]))

    def test_1d_grid2(self):
        self.assertEqual(0, largest_square_dimension([
            [0],
            [0],
            [0]
        ]))

    def test_dimond_shape(self):
        self.assertEqual(3, largest_square_dimension([
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0]
        ]))

    def test_dimond_shape2(self):
        self.assertEqual(3, largest_square_dimension([
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0]
        ]))

    def test_square_on_edge(self):
        self.assertEqual(2, largest_square_dimension([
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 0]
        ]))

    def test_dense_matrix(self):
        self.assertEqual(3, largest_square_dimension([
            [0, 1, 1, 0, 1], 
            [1, 1, 0, 1, 0], 
            [0, 1, 1, 1, 0], 
            [1, 1, 1, 1, 0], 
            [1, 1, 1, 1, 1], 
            [0, 0, 0, 0, 0]
        ]))



if __name__ == '__main__':
    unittest.main(exit=False)
```


### Nov 6, 2020 \[Medium\] M Smallest in K Sorted Lists
---
> **Question:** Given k sorted arrays of possibly different sizes, find m-th smallest value in the merged array.

**Example 1:**
```py
Input: [[1, 3], [2, 4, 6], [0, 9, 10, 11]], m = 5
Output: 4
Explanation: The merged array would be [0, 1, 2, 3, 4, 6, 9, 10, 11].  
The 5-th smallest element in this merged array is 4.
```

**Example 2:**
```py
Input: [[1, 3, 20], [2, 4, 6]], m = 2
Output: 2
```

**Example 3:**
```py
Input: [[1, 3, 20], [2, 4, 6]], m = 6
Output: 20
```

**Solution:** [https://repl.it/@trsong/Find-the-M-thSmallest-in-K-Sorted-Lists](https://repl.it/@trsong/Find-the-M-thSmallest-in-K-Sorted-Lists)
```py
import unittest
from Queue import PriorityQueue

def find_m_smallest(ksorted_list, m):
    pq = PriorityQueue()
    for lst in ksorted_list:
        if not lst:
            continue
        it = iter(lst)
        pq.put((it.next(), it))

    while not pq.empty():
        num, it = pq.get()
        if m == 1:
            return num
        m -= 1

        next_num = next(it, None)
        if next_num is not None:
            pq.put((next_num, it))
    
    return None
        

class FindMSmallestSpec(unittest.TestCase):
    def test_example1(self):
        m, ksorted_list = 5, [
            [1, 3],
            [2, 4, 6],
            [0, 9, 10, 11]
        ]
        expected = 4
        self.assertEqual(expected, find_m_smallest(ksorted_list, m))

    def test_example2(self):
        m, ksorted_list = 2, [
            [1, 3, 20],
            [2, 4, 6]
        ]
        expected = 2
        self.assertEqual(expected, find_m_smallest(ksorted_list, m))

    def test_example3(self):
        m, ksorted_list = 6, [
            [1, 3, 20],
            [2, 4, 6]
        ]
        expected = 20
        self.assertEqual(expected, find_m_smallest(ksorted_list, m))

    def test_empty_sublist(self):
        m, ksorted_list = 2, [
            [1],
            [],
            [0, 2]
        ]
        expected = 1
        self.assertEqual(expected, find_m_smallest(ksorted_list, m))

    def test_one_sublist(self):
        m, ksorted_list = 5, [
            [1, 2, 3, 4, 5],
        ]
        expected = 5
        self.assertEqual(expected, find_m_smallest(ksorted_list, m))

    def test_target_out_of_boundary(self):
        m, ksorted_list = 7, [
            [1, 2, 3],
            [4, 5, 6]
        ]
        self.assertIsNone(find_m_smallest(ksorted_list, m))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 5, 2020 \[Easy\] Maximum Subarray Sum 
---
> **Question:** You are given a one dimensional array that may contain both positive and negative integers, find the sum of contiguous subarray of numbers which has the largest sum.
>
> For example, if the given array is `[-2, -5, 6, -2, -3, 1, 5, -6]`, then the maximum subarray sum is 7 as sum of `[6, -2, -3, 1, 5]` equals 7
>
> Solve this problem with Divide and Conquer as well as DP separately.


**Solution with Divide and Conquer:** [https://repl.it/@trsong/Maximum-Subarray-Sum-Divide-and-Conquer](https://repl.it/@trsong/Maximum-Subarray-Sum-Divide-and-Conquer)
```py
def max_sub_array_sum(nums):
    if not nums:
        return 0
    res = max_sub_array_sum_recur(nums, 0, len(nums) - 1)
    return res.max


def max_sub_array_sum_recur(nums, lo, hi):
    if lo == hi:
        return Result(nums[lo])
    
    mid = lo + (hi - lo) // 2
    left_res = max_sub_array_sum_recur(nums, lo, mid)
    right_res = max_sub_array_sum_recur(nums, mid+1, hi)

    res = Result(0)
    # From left to right, expand range
    res.prefix = max(0, left_res.prefix, left_res.sum + right_res.prefix, left_res.sum + right_res.sum)
    # From right to left, expand range
    res.suffix = max(0, right_res.suffix, right_res.sum + left_res.suffix, right_res.sum + left_res.sum)
    res.sum = left_res.sum + right_res.sum
    res.max = max(res.prefix, res.suffix, left_res.suffix + right_res.prefix, left_res.max, right_res.max)
    return res


class Result(object):
    def __init__(self, res):
        self.prefix = res
        self.suffix = res
        self.sum = res
        self.max = res
```

**Solution with DP:** [https://repl.it/@trsong/Maximum-Subarray-Sum-DP](https://repl.it/@trsong/Maximum-Subarray-Sum-DP)
```py
import unittest

def max_sub_array_sum(nums):
    n = len(nums)
    # Let dp[i] represents max sub array sum ends at nums[i-1]
    # dp[i] = max(0, dp[i-1] + nums[i-1])
    dp = [0] * (n + 1)
    res = 0
    for i in xrange(1, n+1):
        dp[i] = max(0, dp[i-1] + nums[i-1])
        res = max(res, dp[i])
    return res


class MaxSubArraySum(unittest.TestCase):
    def test_empty_array(self):
        self.assertEqual(0, max_sub_array_sum([]))

    def test_ascending_array(self):
        self.assertEqual(6, max_sub_array_sum([-3, -2, -1, 0, 1, 2, 3]))
        
    def test_descending_array(self):
        self.assertEqual(6, max_sub_array_sum([3, 2, 1, 0, -1]))

    def test_example_array(self):
        self.assertEqual(7, max_sub_array_sum([-2, -5, 6, -2, -3, 1, 5, -6]))

    def test_negative_array(self):
        self.assertEqual(0, max_sub_array_sum([-2, -1]))

    def test_positive_array(self):
        self.assertEqual(3, max_sub_array_sum([1, 2]))

    def test_swing_array(self):
        self.assertEqual(5, max_sub_array_sum([-3, 3, -2, 2, -5, 5]))
        self.assertEqual(1, max_sub_array_sum([-1, 1, -1, 1, -1]))
        self.assertEqual(2, max_sub_array_sum([-100, 1, -100, 2, -100]))

    def test_converging_array(self):
        self.assertEqual(4, max_sub_array_sum([-3, 3, -2, 2, 1, 0]))

    def test_positive_negative_positive_array(self):
        self.assertEqual(8, max_sub_array_sum([7, -1, -2, 3, 1]))
        self.assertEqual(7, max_sub_array_sum([7, -1, -2, 0, 1, 1]))

    def test_negative_positive_array(self):
        self.assertEqual(3, max_sub_array_sum([-100, 1, 0, 2, -100]))
  

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 4, 2020 \[Easy\] Largest Path Sum from Root To Leaf
---
> **Question:** Given a binary tree, find and return the largest path from root to leaf.

**Example:**
```py
Input:
    1
  /   \
 3     2
  \   /
   5 4
Output: [1, 3, 5]
```

**Solution with DFS:** [https://repl.it/@trsong/Find-Largest-Path-Sum-from-Root-To-Leaf](https://repl.it/@trsong/Find-Largest-Path-Sum-from-Root-To-Leaf)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def largest_sum_path(root):
    if not root:
        return []
    parent_lookup = {}
    stack = [(root, 0)]
    max_sum = float('-inf')
    max_sum_node = root

    while stack:
        cur, prev_sum = stack.pop()
        cur_sum = prev_sum + cur.val
        if not cur.left and not cur.right and cur_sum > max_sum:
            max_sum = cur_sum
            max_sum_node = cur
        else:
            for child in [cur.left, cur.right]:
                if not child:
                    continue
                parent_lookup[child] = cur
                stack.append((child, cur_sum))
    
    res = []
    node = max_sum_node
    while node:
        res.append(node.val)
        node = parent_lookup.get(node, None)
    return res[::-1]


class LargestSumPathSpec(unittest.TestCase):
    def test_example(self):
        """
            1
          /   \
         3     2
          \   /
           5 4
        """
        left_tree = TreeNode(3, right=TreeNode(5))
        right_tree = TreeNode(2, TreeNode(4))
        root = TreeNode(1, left_tree, right_tree)
        expected_path = [1, 3, 5]
        self.assertEqual(expected_path, largest_sum_path(root))

    def test_negative_nodes(self):
        """
             10
            /  \
          -2    7
         /  \     
        8   -4    
        """
        left_tree = TreeNode(-2, TreeNode(8), TreeNode(-4))
        root = TreeNode(10, left_tree, TreeNode(7))
        expected_path = [10, 7]
        self.assertEqual(expected_path, largest_sum_path(root))

    def test_empty_tree(self):
        self.assertEqual([], largest_sum_path(None))

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
        expected_path = [1, 3, 5, 9]
        self.assertEqual(expected_path, largest_sum_path(n1))

    def test_all_paths_are_negative(self):
        """
              -1
            /     \
          -2      -3
          / \    /  \
        -4  -5 -6   -7
        """
        left_tree = TreeNode(-2, TreeNode(-4), TreeNode(-5))
        right_tree = TreeNode(-3, TreeNode(-6), TreeNode(-7))
        root = TreeNode(-1, left_tree, right_tree)
        expected_path = [-1, -2, -4]
        self.assertEqual(expected_path, largest_sum_path(root))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Nov 3, 2020 LC 121 \[Easy\] Best Time to Buy and Sell Stock
---
> **Question:** You are given an array. Each element represents the price of a stock on that particular day. Calculate and return the maximum profit you can make from buying and selling that stock only once.

**Example:**
```py
Input: [9, 11, 8, 5, 7, 10]
Output: 5
Explanation: Here, the optimal trade is to buy when the price is 5, and sell when it is 10, so the return value should be 5 (profit = 10 - 5 = 5).
```

**Solution:** [https://repl.it/@trsong/Best-Time-to-Buy-and-Sell-Stock](https://repl.it/@trsong/Best-Time-to-Buy-and-Sell-Stock)
```py
import unittest

def max_profit(stock_data):
    if not stock_data:
        return 0
    local_min = stock_data[0]
    res = 0
    for price in stock_data:
        if price < local_min:
            local_min = price
        else:
            res = max(res, price - local_min)
    return res


class MaxProfitSpec(unittest.TestCase):
    def test_blank_data(self):
        self.assertEqual(max_profit([]), 0)
    
    def test_1_day_data(self):
        self.assertEqual(max_profit([9]), 0)
        self.assertEqual(max_profit([-1]), 0)

    def test_monotonically_increase(self):
        self.assertEqual(max_profit([1, 2, 3]), 2)
        self.assertEqual(max_profit([1, 1, 1, 2, 2, 3, 3, 3]), 2)
    
    def test_monotonically_decrease(self):
        self.assertEqual(max_profit([3, 2, 1]), 0)
        self.assertEqual(max_profit([3, 3, 3, 2, 2, 1, 1, 1]), 0)

    def test_raise_suddenly(self):
        self.assertEqual(max_profit([3, 2, 1, 1, 2]), 1)
        self.assertEqual(max_profit([3, 2, 1, 1, 9]), 8)

    def test_drop_sharply(self):
        self.assertEqual(max_profit([1, 3, 0]), 2)
        self.assertEqual(max_profit([1, 3, -1]), 2)

    def test_bear_market(self):
        self.assertEqual(max_profit([10, 11, 5, 7, 1, 2]), 2)
        self.assertEqual(max_profit([10, 11, 1, 4, 2, 7, 5]), 6)

    def test_bull_market(self):
        self.assertEqual(max_profit([1, 5, 3, 7, 2, 14, 10]), 13)
        self.assertEqual(max_profit([5, 1, 11, 10, 12]), 11)


if __name__ == '__main__':
    unittest.main(exit=False)
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
