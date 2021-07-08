---
layout: post
title:  "Daily Coding Problems 2021 May to Jul"
date:   2021-05-01 22:22:32 -0700
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


### Jul 8, 2021 LC 218 \[Hard\] City Skyline
---
> **Question:** Given a list of building in the form of `(left, right, height)`, return what the skyline should look like. The skyline should be in the form of a list of `(x-axis, height)`, where x-axis is the point where there is a change in height starting from 0, and height is the new height starting from the x-axis.

**Example:**
```py
Input: [(2, 8, 3), (4, 6, 5)]
Output: [(2, 3), (4, 5), (7, 3), (9, 0)]
Explanation:
             2 2 2
             2   2
         1 1 2 1 2 1 1
         1   2   2   1
         1   2   2   1      
pos: 0 1 2 3 4 5 6 7 8 9
We have two buildings: one has height 3 and the other 5. The city skyline is just the outline of combined looking. 
The result represents the scanned height of city skyline from left to right.
```


### Jul 7, 2021 \[Easy\] Huffman Coding
---
> **Question:** Huffman coding is a method of encoding characters based on their frequency. Each letter is assigned a variable-length binary string, such as `0101` or `111110`, where shorter lengths correspond to more common letters. To accomplish this, a binary tree is built such that the path from the root to any leaf uniquely maps to a character. When traversing the path, descending to a left child corresponds to a `0` in the prefix, while descending right corresponds to 1.
>
> Here is an example tree (note that only the leaf nodes have letters):
```py
        *
      /   \
    *       *
   / \     / \
  *   a   t   *
 /             \
c               s
```
> With this encoding, cats would be represented as `0000110111`.
>
> Given a dictionary of character frequencies, build a Huffman tree, and use it to determine a mapping between characters and their encoded binary strings.

**My thoughts:** Huffman encoding has the following properties:
- Less frequent letter has longer encoding
- No encoding is prefix of any other encodings

**Solution with Priority Queue:** [https://replit.com/@trsong/Huffman-Coding](https://replit.com/@trsong/Huffman-Coding)
```py
import unittest
from queue import PriorityQueue

def huffman_encode(histogram):
    if len(histogram) < 2:
        return None

    root = build_huffman_tree(histogram)
    res = {}
    huffman_backtrack(root, res, [])
    return res


def build_huffman_tree(histogram):
    pq = PriorityQueue()
    for ch, freq in histogram.items():
        pq.put((freq, TreeNode(ch)))

    while pq.qsize() > 1:
        left_freq, left_child = pq.get()
        right_freq, right_child = pq.get()
        pq.put((left_freq + right_freq, TreeNode(None, left_child, right_child)))
        
    _, root = pq.get()
    return root


def huffman_backtrack(root, encoding_map, path):
    if root.left is None and root.right is None:
        encoding_map[root.val] = ''.join(path)

    if root.left:
        path.append('0')
        huffman_backtrack(root.left, encoding_map, path)
        path.pop()

    if root.right:
        path.append('1')
        huffman_backtrack(root.right, encoding_map, path)
        path.pop()


class TreeNode(object):
    def __init__(self, val=None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class HuffmanEncodeSpec(unittest.TestCase):
    def validate_huffman(self, histogram, encoding_map):
        # Make sure the encoding contains exact character set
        self.assertEqual(sorted(list(histogram.keys())), sorted(list(encoding_map.keys())))

        for k1, freq1 in histogram.items():
            for k2, freq2 in histogram.items():
                if k1 == k2: 
                    continue
                # Make sure less frequent letter has longer encoding
                if freq1 < freq2:
                    self.assertGreaterEqual(len(encoding_map[k1]), len(encoding_map[k2]))
                elif freq1 > freq2:
                    self.assertLessEqual(len(encoding_map[k1]), len(encoding_map[k2]))
                

        for code in encoding_map.values():
            # Make sure encoding just contains character 0 and 1
            self.assertEqual(len(code), code.count("1") + code.count("0"))

        for k1, code1 in encoding_map.items():
            for k2, code2 in encoding_map.items():
                if k1 == k2:
                    continue
                # Make sure any encoding is not prefix of any other encodings
                self.assertFalse(code2.startswith(code1))

    def test_example(self):
        """
        Step 1: Pop f(2), a(3) from heap. Create internal node n5
          (5)
         /   \
        f     a

        Step 2: Pop n5 and c(6) from heap. Create internal node n11
              (11)
             /    \  
          (5)      c
         /   \
        f     a

        Step 3: Pop e(8) and n11 from heap. Create internal node n19
              (19)
             /   \
            e   (11)      
               /    \
             (5)      c
            /   \
           f     a
        """
        histogram = {'a': 3, 'c': 6, 'e': 8, 'f': 2}
        # Possible encoding: {'e': '0', 'f': '100', 'a': '101', 'c': '11'} 
        encoding_map = huffman_encode(histogram)
        self.validate_huffman(histogram, encoding_map)

    def test_example2(self):
        """
        Step 1: Pop a(5) and b(9) from heap. Create internal node n14
         (14)
         /  \
        a    b

        Step 2: Pop c(12) and d(13) from heap. Create internal node n25
         (25)
         /  \
        c    d

        Step 3: Pop n14 and e(16) from heap. Create internal node n30.
            (30)
           /   \
         (14)   e
         /  \
        a    b

        Step 4: Pop n25 and n30 from heap. Create internal node n55.
            (55)
           /    \      
        (25)    (30)
        /  \    /   \
       c    d (14)   e
              /  \
             a    b

        Step 5: Pop f(45) and n55 from heap. Create internal node n100
                 (100)
                 /   \
                f   (55)
                   /    \      
                (25)    (30)
                /  \    /   \
               c    d (14)   e
                      /  \
                     a    b
        """
        histogram = {'a': 5, 'b': 9, 'c': 12, 'd': 13, 'e': 16, 'f': 45}
        # Possible encoding: {'f': '0', 'c': '100', 'd': '101', 'a': '1100', 'b': '1101', 'e': '111'}
        encoding_map = huffman_encode(histogram)
        self.validate_huffman(histogram, encoding_map)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### Jul 6, 2021 LC 665 \[Medium\] Off-by-One Non-Decreasing Array
---
> **Question:** Given an array of integers, write a function to determine whether the array could become non-decreasing by modifying at most 1 element.
>
> For example, given the array `[10, 5, 7]`, you should return true, since we can modify the `10` into a `1` to make the array non-decreasing.
>
> Given the array `[10, 5, 1]`, you should return false, since we can't modify any one element to get a non-decreasing array.

**Solution:** [https://replit.com/@trsong/Determine-if-Off-by-One-Non-Decreasing-Array-2](https://replit.com/@trsong/Determine-if-Off-by-One-Non-Decreasing-Array-2)
```py
import unittest

def is_off_by_one_array(nums):
    if len(nums) <= 2:
        return True

    down_pos = None
    for i in range(1, len(nums)):
        if nums[i-1] <= nums[i]:
            continue
        
        if down_pos is not None:
            return False
        down_pos = i
    
    return (down_pos is None or 
            down_pos == 1 or 
            down_pos == len(nums) - 1 or
            nums[down_pos - 1] <= nums[down_pos + 1] or 
            nums[down_pos - 2] <= nums[down_pos])


class IsOffByOneArraySpec(unittest.TestCase):
    def test_example(self):
        self.assertTrue(is_off_by_one_array([10, 5, 7]))

    def test_example2(self):
        self.assertFalse(is_off_by_one_array([10, 5, 1]))

    def test_empty_array(self):
        self.assertTrue(is_off_by_one_array([]))

    def test_one_element_array(self):
        self.assertTrue(is_off_by_one_array([1]))

    def test_two_elements_array(self):
        self.assertTrue(is_off_by_one_array([1, 1]))
        self.assertTrue(is_off_by_one_array([1, 0]))
        self.assertTrue(is_off_by_one_array([0, 1]))

    def test_decreasing_array(self):
        self.assertFalse(is_off_by_one_array([8, 2, 0]))

    def test_non_decreasing_array(self):
        self.assertTrue(is_off_by_one_array([0, 0, 1, 2, 2]))
        self.assertTrue(is_off_by_one_array([0, 1, 2]))
        self.assertTrue(is_off_by_one_array([0, 0, 0, 0]))

    def test_off_by_one_array(self):
        self.assertTrue(is_off_by_one_array([2, 10, 0]))
        self.assertTrue(is_off_by_one_array([5, 2, 10]))
        self.assertTrue(is_off_by_one_array([0, 1, 0, 0]))
        self.assertTrue(is_off_by_one_array([-1, 4, 2, 3]))
        self.assertTrue(is_off_by_one_array([0, 1, 1, 0]))

    def test_off_by_two_array(self):
        self.assertFalse(is_off_by_one_array([5, 2, 10, 3, 4]))
        self.assertTrue(is_off_by_one_array([0, 1, 0, 0, 0, 1]))
        self.assertFalse(is_off_by_one_array([1, 1, 0, 0]))
        self.assertFalse(is_off_by_one_array([0, 1, 1, 0, 0, 1]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jul 5, 2021 \[Medium\] Largest BST in a Binary Tree
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

**Solution with Recursion:** [https://replit.com/@trsong/Find-Largest-BST-in-a-Binary-Tree-2](https://replit.com/@trsong/Find-Largest-BST-in-a-Binary-Tree-2)
```py
import unittest

def largest_bst(root):
    res = largest_bst_recur(root)
    return res.max_bst


def largest_bst_recur(root):
    if root is None:
        return BSTResult()
    
    left_res = largest_bst_recur(root.left)
    right_res = largest_bst_recur(root.right)
    is_valid_bst = (left_res.is_valid and 
                    right_res.is_valid and 
                    (left_res.max_val is None or left_res.max_val <= root.val) and 
                    (right_res.min_val is None or root.val <= right_res.min_val))
    res = BSTResult()
    if is_valid_bst:
        res.min_val = left_res.min_val if root.left is not None else root.val
        res.max_val = right_res.max_val if root.right is not None else root.val
        res.max_bst_size = 1 + left_res.max_bst_size + right_res.max_bst_size
        res.max_bst = root
    else:
        res.max_bst_size = max(left_res.max_bst_size, right_res.max_bst_size)
        res.max_bst = left_res.max_bst if left_res.max_bst_size > right_res.max_bst_size else right_res.max_bst
        res.is_valid = False
    return res


class BSTResult(object):
    def __init__(self):
        self.min_val = None
        self.max_val = None
        self.max_bst_size = 0
        self.max_bst = None
        self.is_valid = True


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
    unittest.main(exit=False, verbosity=2)
```

### Jul 4, 2021 \[Hard\] Stable Marriage Problem
---
> **Question:** The stable marriage problem is defined as follows:
>
> Suppose you have `N` men and `N` women, and each person has ranked their prospective opposite-sex partners in order of preference.
>
> For example, if `N = 3`, the input could be something like this:

```py
guy_preferences = {
    'andrew': ['caroline', 'abigail', 'betty'],
    'bill': ['caroline', 'betty', 'abigail'],
    'chester': ['betty', 'caroline', 'abigail'],
}

gal_preferences = {
    'abigail': ['andrew', 'bill', 'chester'],
    'betty': ['bill', 'andrew', 'chester'],
    'caroline': ['bill', 'chester', 'andrew']
}
```

> Write an algorithm that pairs the men and women together in such a way that no two people of opposite sex would both rather be with each other than with their current partners.
 
**Gale–Shapley Algorithm:** Consider the following example.
```
Let there be two men m1 and m2 and two women w1 and w2.
Let m1's list of preferences be {w1, w2}
Let m2's list of preferences be {w1, w2}
Let w1's list of preferences be {m1, m2}
Let w2's list of preferences be {m1, m2}
```

The matching { {m1, w2}, {w1, m2} } is not stable because m1 and w1 would prefer each other over their assigned partners. The matching {m1, w1} and {m2, w2} is stable because there are no two people of opposite sex that would prefer each other over their assigned partners.

```
function stableMatching {
    Initialize all m ∈ M and w ∈ W to free
    while ∃ free man m who still has a woman w to propose to {
       w = first woman on m’s list to whom m has not yet proposed
       if w is free
         (m, w) become engaged
       else some pair (m', w) already exists
         if w prefers m to m'
            m' becomes free
           (m, w) become engaged 
         else
           (m', w) remain engaged
    }
}
```

**Solution with Gale–Shapley Algorithm:** [https://replit.com/@trsong/Solve-Stable-Marriage-Problem](https://replit.com/@trsong/Solve-Stable-Marriage-Problem)
 ```py
import unittest
from collections import deque

def stable_marriage_match(guy_preferences, gal_preferences):
    woman_preference_ranking = {}
    for woman in gal_preferences:
        woman_preference_ranking[woman] = {}
        for rank, man in enumerate(gal_preferences[woman]):
            woman_preference_ranking[woman][man] = rank

    woman_engagement = { woman: None for woman in gal_preferences }
    single_men = deque(guy_preferences)
    next_propose_index = { man: 0 for man in guy_preferences }
    while single_men:
        man = single_men.pop()
        target_woman = guy_preferences[man][next_propose_index[man]]
        if woman_engagement[target_woman] is None:
            woman_engagement[target_woman] = man
        elif woman_preference_ranking[target_woman][man] < woman_preference_ranking[target_woman][woman_engagement[target_woman]]:
            evicted_man = woman_engagement[target_woman]
            single_men.append(evicted_man)
            woman_engagement[target_woman] = man
        else:
            single_men.appendleft(man)
        next_propose_index[man] += 1

    return list(woman_engagement.items())


####################
# Testing Utilities
####################
def find_all_unstable_pairs(match_pairs, guy_preferences, gal_preferences):
    preferences = {**guy_preferences, **gal_preferences}
    matched = dict(match_pairs + list(map(reversed, match_pairs)))
    rank = lambda person, target: preferences[person].index(target)
    prefer = lambda person, target: rank(person, target) < rank(
        person, matched[person])

    unstable_pairs = []
    for person, preference_list in preferences.items():
        for candidate in preference_list:
            if matched[person] != candidate and prefer(
                    person, candidate) and prefer(candidate, person):
                # check if exists (person, candidate) not in match_pairs such that
                # rank(person, candidate) < rank(person, matched[person]) and
                # rank(candidate, person) < rank(candidate, matched[candidate])
                unstable_pairs.append((person, candidate))
    return unstable_pairs


class StableMarriageMatchSpec(unittest.TestCase):
    def assert_result(self, match_pairs, guy_preferences, gal_preferences):
        # Check if all people are married as well as all match_pairs are stable
        self.assertEqual(len(match_pairs), len(guy_preferences))
        self.assertEqual([],
                         find_all_unstable_pairs(match_pairs, guy_preferences,
                                                 gal_preferences))

    def test_unstable_marriage_should_return_correct_instability_pairs(self):
        # Test utility method find_all_unstable_pairs
        guy_preferences = {'m1': ['w1', 'w2'], 'm2': ['w1', 'w2']}
        gal_preferences = {'w1': ['m1', 'm2'], 'w2': ['m1', 'm2']}
        match_pairs = [('m1', 'w2'), ('m2', 'w1')]
        instability_pairs = [('w1', 'm1'), ('m1', 'w1')]
        self.assertEqual(
            sorted(instability_pairs),
            sorted(find_all_unstable_pairs(match_pairs, guy_preferences, gal_preferences)))

    def test_stable_marriage_should_return_none_instability_pairs(self):
        # Test utility method find_all_unstable_pairs
        guy_preferences = {'m1': ['w1', 'w2'], 'm2': ['w1', 'w2']}
        gal_preferences = {'w1': ['m1', 'm2'], 'w2': ['m1', 'm2']}
        match_pairs = [('m1', 'w1'), ('m2', 'w2')]
        self.assertEqual([],
                         find_all_unstable_pairs(match_pairs, guy_preferences, gal_preferences))

    def test_example(self):
        guy_preferences = {
            'andrew': ['caroline', 'abigail', 'betty'],
            'bill': ['caroline', 'betty', 'abigail'],
            'chester': ['betty', 'caroline', 'abigail'],
        }
        gal_preferences = {
            'abigail': ['andrew', 'bill', 'chester'],
            'betty': ['bill', 'andrew', 'chester'],
            'caroline': ['bill', 'chester', 'andrew']
        }
        # Possible Matches: [('abigail', 'andrew'), ('betty', 'chester'), ('caroline', 'bill')]
        res = stable_marriage_match(guy_preferences, gal_preferences)
        self.assert_result(res, guy_preferences, gal_preferences)

    def test_size4_problem(self):
        guy_preferences = {
            'm1': ['w1', 'w2', 'w3', 'w4'],
            'm2': ['w1', 'w2', 'w3', 'w4'],
            'm3': ['w1', 'w2', 'w3', 'w4'],
            'm4': ['w1', 'w2', 'w3', 'w4']
        }
        gal_preferences = {
            'w1': ['m4', 'm2', 'm3', 'm1'],
            'w2': ['m2', 'm1', 'm3', 'm4'],
            'w3': ['m1', 'm2', 'm3', 'm4'],
            'w4': ['m1', 'm2', 'm3', 'm4']
        }
        # Possible Matches: [('w1', 'm4'), ('w2', 'm2'), ('w3', 'm1'), ('w4', 'm3')]
        res = stable_marriage_match(guy_preferences, gal_preferences)
        self.assert_result(res, guy_preferences, gal_preferences)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
 ```

### Jul 3, 2021 \[Hard\] Elo Rating System
---
> **Question:** In chess, the Elo rating system is used to calculate player strengths based on game results.
>
> A simplified description of the Elo system is as follows. Every player begins at the same score. For each subsequent game, the loser transfers some points to the winner, where the amount of points transferred depends on how unlikely the win is. For example, a 1200-ranked player should gain much more points for beating a 2000-ranked player than for beating a 1300-ranked player.
>
> Implement this system.


**10 Match Result:** Notice that there is a diminishing return when a player beats same opponent over and over again.
```py
  round    player1    player2
-------  ---------  ---------
      0       1000       1000
      1       1016        984
      2       1030        969
      3       1043        956
      4       1055        944
      5       1066        933
      6       1076        923
      7       1086        913
      8       1094        905
      9       1102        897
     10       1110        889
```

**Solution:** [https://replit.com/@trsong/Elo-Rating-System](https://replit.com/@trsong/Elo-Rating-System)
```py
import unittest
import uuid

class EloRatingSystem(object):
    DISTRIBUTION_BASE = 10
    DISTRIBUTION_SD = 400

    class Outcome:
        WIN = 1
        DRAW = 0.5
        LOSE = 0

    def __init__(self, initial_score=1000, round_score=32):
        self.score_board = {}
        self.initial_score = initial_score
        self.round_score = round_score

    def new_player(self):
        player_id = str(uuid.uuid4())
        self.score_board[player_id] = self.initial_score
        return player_id

    def match(self, p1, p2, outcome):
        score1, score2 = self.score_board[p1], self.score_board[p2]
        prob1 = self.match_probability(score1, score2)
        prob2 = self.match_probability(score2, score1)
        outcome1 = 1 - outcome
        outcome2 = outcome

        self.score_board[p1] += self.round_score * (prob1 - outcome1)
        self.score_board[p2] += self.round_score * (prob2 - outcome2)

    def match_probability(self, score1, score2):
        # logistic distribution predict the likelihood of match win result
        return 1.0 / (1.0 + EloRatingSystem.DISTRIBUTION_BASE**(
            (score2 - score1) / EloRatingSystem.DISTRIBUTION_SD))

    def score(self, player_id):
        return self.score_board.get(player_id, -1)


class EloRatingSystemSpec(unittest.TestCase):
    @staticmethod
    def quick_match(elo_system, p1, p2, outcome):
        elo_system.match(p1, p2, outcome)
        new_score1 = elo_system.score(p1)
        new_score2 = elo_system.score(p2)
        return new_score1, new_score2

    def test_player_creation_and_initial_score(self):
        elo_system = EloRatingSystem(initial_score=100)
        player_id = elo_system.new_player()
        self.assertEqual(100, elo_system.score(player_id))

    def test_match_draw_will_not_change_score(self):
        elo_system = EloRatingSystem()
        p1 = elo_system.new_player()
        p2 = elo_system.new_player()
        score1 = elo_system.score(p1)
        score2 = elo_system.score(p2)
        self.assertEqual(score1, score2)

        round1_score1, round1_score2 = self.quick_match(
            elo_system, p1, p2, EloRatingSystem.Outcome.DRAW)
        self.assertEqual(round1_score1, score1)
        self.assertEqual(round1_score2, score2)

    def test_match_winner_add_points_and_loser_remove_points(self):
        elo_system = EloRatingSystem()
        p1 = elo_system.new_player()
        p2 = elo_system.new_player()
        score1 = elo_system.score(p1)
        score2 = elo_system.score(p2)
        
        round1_score1, round1_score2 = self.quick_match(
            elo_system, p1, p2, EloRatingSystem.Outcome.WIN)
        self.assertGreater(round1_score1, score1)
        self.assertLess(round1_score2, score2)

        round2_score2, round2_score1 = self.quick_match(
            elo_system, p2, p1, EloRatingSystem.Outcome.LOSE)
        self.assertGreater(round2_score1, round1_score1)
        self.assertLess(round2_score2, round1_score2)

    def test_diminishing_return_when_winning_over_and_over(self):
        elo_system = EloRatingSystem()
        p1 = elo_system.new_player()
        p2 = elo_system.new_player()
        score1 = elo_system.score(p1)
        
        round1_score1, _ = self.quick_match(
            elo_system, p1, p2, EloRatingSystem.Outcome.WIN)

        round2_score2, _ = self.quick_match(
            elo_system, p1, p2, EloRatingSystem.Outcome.WIN)
        
        self.assertLess(round2_score2 - round1_score1, round1_score1, score1)

    def test_demo_win_10_round_in_a_row(self):
        from tabulate import tabulate

        match_result = []
        elo_system = EloRatingSystem()
        p1 = elo_system.new_player()
        p2 = elo_system.new_player()
        score1 = elo_system.score(p1)
        score2 = elo_system.score(p2)
        match_result.append([0, score1, score2])

        for round in range(1, 11):
            s1, s2 = self.quick_match(elo_system, p1, p2, EloRatingSystem.Outcome.WIN)
            match_result.append([round, int(s1), int(s2)])
        
        print("\n10 Match Result\n")
        print(tabulate(match_result, headers=['round', 'player1', 'player2']))

        
if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jul 2, 2021 \[Easy\] Goldbach’s conjecture
---
> **Question:** Given an even number (greater than 2), return two prime numbers whose sum will be equal to the given number.
>
> A solution will always exist. See [Goldbach’s conjecture](https://en.wikipedia.org/wiki/Goldbach%27s_conjecture). If there are more than one solution possible, return the lexicographically smaller solution.

**Example:**
```py
Input: 4
Output: (2, 2)
Explanation: 2 + 2 = 4
```

**Solution:** [https://replit.com/@trsong/Goldbachs-conjecture](https://replit.com/@trsong/Goldbachs-conjecture)
```py
import unittest
from math import sqrt

def goldbach_conjecture(n):
    for term in range(n):
        if is_prime(term) and is_prime(n - term):
            return (term, n - term)
    return None


def is_prime(num):
    if num < 2:
        return False

    for d in range(2, int(sqrt(num)) + 1):
        if num % d == 0:
            return False
    
    return True


class GoldbachConjectureSpec(unittest.TestCase):
    def test_example(self):
        n, expected = 4, (2, 2)
        self.assertEqual(expected, goldbach_conjecture(n))

    def test_example2(self):
        n, expected = 6, (3, 3)
        self.assertEqual(expected, goldbach_conjecture(n))

    def test_example3(self):
        n, expected = 8, (3, 5)
        self.assertEqual(expected, goldbach_conjecture(n))

    def test_example4(self):
        n, expected = 10, (3, 7)
        self.assertEqual(expected, goldbach_conjecture(n))

    def test_example6(self):
        n, expected = 12, (5, 7)
        self.assertEqual(expected, goldbach_conjecture(n))

    def test_return_lexicographically_smaller_result(self):
        # 14 = 3 + 11 = 7 + 7
        n, expected = 14, (3, 11) 
        self.assertEqual(expected, goldbach_conjecture(n))

    def test_return_lexicographically_smaller_result2(self):
        n, expected = 100, (3, 97)
        self.assertEqual(expected, goldbach_conjecture(n))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### Jul 1, 2021 \[Hard\] Sum of Consecutive Numbers
---
> **Question:** Given a number `n`, return the number of lists of consecutive numbers that sum up to `n`.
>
> For example, for `n = 9`, you should return `3` since the lists are: `[2, 3, 4]`, `[4, 5]`, and `[9]`. Can you do it in linear time?

**Solution:** [https://replit.com/@trsong/Sum-of-Consecutive-Numbers](https://replit.com/@trsong/Sum-of-Consecutive-Numbers)
```py
import unittest

def sum_of_consecutive_numbers(n):
    # suppose sequence length is d (where d <= n) and sequence start from x (where x >= 1). 
    #     x + (x + 1) + ... + (x + d - 1) = n
    # =>  d * x + (1 + d - 1) * (d - 1) / 2 = n 
    # =>  d * x = n - d * (d - 1) / 2
    # Now, by knowing n, we want to find (d, x) pair such that above equation holds
    res = 0
    for d in range(1, n + 1):
        rhs = n - d * (d - 1) // 2
        if rhs <= 0:
            break
        
        if rhs % d == 0:
            # x is rhs / d
            res += 1
    return res


class SumOfConsecutiveNumberSpec(unittest.TestCase):
    def test_example(self):
        # [2, 3, 4], [4, 5], and [9]
        self.assertEqual(3, sum_of_consecutive_numbers(9))

    def test_example2(self):
        # [2]
        self.assertEqual(1, sum_of_consecutive_numbers(2))

    def test_example3(self):
        # [1]
        self.assertEqual(1, sum_of_consecutive_numbers(1))

    def test_example4(self):
        # [4]
        self.assertEqual(1, sum_of_consecutive_numbers(4))

    def test_example5(self):
        # [1, 2, 3, 4, 5], [4, 5, 6], [7, 8], and [15]
        self.assertEqual(4, sum_of_consecutive_numbers(15))

    def test_example6(self):
        # [2, 3], and [5]
        self.assertEqual(2, sum_of_consecutive_numbers(5))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### June 30, 2021 \[Hard\] Array Shuffle
---
> **Question:** Given an array, write a program to generate a random permutation of array elements. This question is also asked as “shuffle a deck of cards” or “randomize a given array”. Here shuffle means that every permutation of array element should equally likely.

```
Input: [1, 2, 3, 4, 5, 6]
Output: [3, 4, 1, 5, 6, 2]
The output can be any random permutation of the input such that all permutation are equally likely.
```
> **Hint:** Given a function that generates perfectly random numbers between 1 and k (inclusive) where k is an input, write a function that shuffles the input array using only swaps.

**My thoughts:** It's quite tricky to figure out how to evenly shuffle an array unless you understand there is no difference between shuffle an array versus randomly generate a permutation of that array.

So, how to randomly generate a permutation of an array? That's simple. Just imagine you hold a deck of cards on your hand, and each time you randomly draw a card until not more cards available.

| Round | Remaining Cards | Chosen Cards |
|:------|:----------------|:-------------| 
| 0     | `[1, 2, 3, 4, 5]` | `[]`           |
| 1     | `[1, 3, 4, 5]`    | `[2]`          |
| 2     | `[1, 3, 5]`       | `[4, 2]`       |
| 3     | `[3, 5]`          | `[1, 4, 2]`    |
| 4     | `[3]`             | `[5, 1, 4, 2]` |
| 1     | `[]`              | `[3, 5, 1, 4, 2]` |

It seems we can do that in-place:

| Round | Remaining Cards \| Chosen Cards |
|:------|:-----------------------------| 
| 0     | `[1, 2, 3, 4, 5|]`           |
| 1     | `[1, 3, 4, 5 | 2]`          |
| 2     | `[1, 3, 5 | 4, 2]`       |
| 3     | `[3, 5 | 1, 4, 2]`    |
| 4     | `[3 | 5, 1, 4, 2]` |
| 1     | `[|3, 5, 1, 4, 2]` |



**Solution:** [https://replit.com/@trsong/Randomly-Shuffle-Array](https://replit.com/@trsong/Randomly-Shuffle-Array)
```py
from random import randint

def array_shuffle(nums):
    for last in range(len(nums) - 1, 0, -1):
        chosen = randint(0, last)
        # move the chosen number to last and move on
        nums[chosen], nums[last] = nums[last], nums[chosen]


def print_shuffle_histogram(nums, repeat):
    """Print the frequency of each position get swapped"""
    n = len(nums)
    original = nums[:]
    swap_freq = [0] * n
    for _ in range(repeat):
        array_shuffle(nums)
        for i in range(n):
            if original[i] != nums[i]:
                swap_freq[i] += 1
    print(swap_freq)


if __name__ == '__main__':
    nums = list(range(10))
    # The frequency map for position get swapped should look like:
    # [9010, 9036, 9015, 9035, 9006, 8935, 8990, 8951, 8926, 8985]
    # Indicates each postion has same probability to be shuffled
    print_shuffle_histogram(nums, repeat=10000)
```


### June 29, 2021 \[Easy\] Balanced Brackets
---
> **Question:** Given a string of round, curly, and square open and closing brackets, return whether the brackets are balanced (well-formed).
> For example, given the string `"([])[]({})"`, you should return true.
> 
> Given the string `"([)]"` or `"((()"`, you should return false.

**My thoughts:** Whenever there is an open bracket, there should be a corresponding close bracket. Likewise, whenver we encounter a close bracket that does not match corresponding open braket, such string is not valid.

So the idea is to iterate through the string, store all the open bracket, and whenever we see a close bracket we check and see if it matches the most rent open breaket we stored early. The data structure, **Stack**, staisfies all of our requirements.


**Solution with Stack:** [https://replit.com/@trsong/Is-Balanced-Brackets](https://replit.com/@trsong/Is-Balanced-Brackets)
```py
import unittest

def is_balanced_brackets(input):
    bracket_mapping = {
        '(': ')',
        '[': ']',
        '{': '}'
    }

    stack = []
    for ch in input:
        if ch in bracket_mapping:
            stack.append(ch)
        elif stack and ch == bracket_mapping[stack[-1]]:
            stack.pop()
        else:
            return False

    return not stack


class IsBalancedBracketSpec(unittest.TestCase):
    def test_example(self):
        self.assertTrue(is_balanced_brackets('([])[]({})'))

    def test_example2(self):
        self.assertFalse(is_balanced_brackets('([)]'))

    def test_example3(self):
        self.assertFalse(is_balanced_brackets('((()'))

    def test_empty_input(self):
        self.assertTrue(is_balanced_brackets(''))

    def test_close_bracket_only(self):
        self.assertFalse(is_balanced_brackets(')'))

    def test_mismatch_input(self):
        self.assertFalse(is_balanced_brackets('(]'))

    def test_mismatch_input2(self):
        self.assertFalse(is_balanced_brackets('[}'))

    def test_mismatch_input3(self):
        self.assertFalse(is_balanced_brackets('((]]'))

    def test_mismatch_input4(self):
        self.assertFalse(is_balanced_brackets('(][)'))

    def test_balanced_brackets(self):
        self.assertTrue(is_balanced_brackets('(([]))'))

    def test_balanced_brackets2(self):
        self.assertTrue(is_balanced_brackets('[]{}()'))

    def test_balanced_brackets3(self):
        self.assertTrue(is_balanced_brackets('[](())'))

    def test_imbalanced_brackets(self):
        self.assertFalse(is_balanced_brackets('())))'))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### June 28, 2021  LC 65 \[Medium\]  Determine if number
---
> **Question:** Given a string that may represent a number, determine if it is a number. Here's some of examples of how the number may be presented:
```py
"123" # Integer
"12.3" # Floating point
"-123" # Negative numbers
"-.3" # Negative floating point
"1.5e5" # Scientific notation
```

> Here's some examples of what isn't a proper number:
```py
"12a" # No letters
"1 2" # No space between numbers
"1e1.2" # Exponent can only be an integer (positive or negative or 0)
```
> Scientific notation requires the first number to be less than 10, however to simplify the solution assume the first number can be greater than 10. Do not parse the string with int() or any other python functions.

**Solution:** [https://replit.com/@trsong/Determine-valid-number](https://replit.com/@trsong/Determine-valid-number)
```py
import unittest

def is_valid_number(raw_num):
    """
    (WHITE_SPACE) (SIGN) DIGITS (DOT DIGITS) (e (SIGN) DIGITS) (WHITE_SPACE)
    or
     (WHITE_SPACE) (SIGN) DOT DIGITS (e (SIGN) DIGITS) (WHITE_SPACE)
    """
    if not raw_num:
        return False
    n = len(raw_num)
    start = 0
    end = len(raw_num) - 1
    has_digit = False
    has_exponent = False
    has_dot = False
    has_sign = False

    # Skip Whitespaces
    while start < n and raw_num[start] == ' ':
        start += 1
    
    while end >= 0 and raw_num[end] == ' ':
        end -= 1
    
    if start > end:
        return False
    
    for i in range(start, end+1):
        char = raw_num[i]
        if char == '-' or char == '+':
            # SIGN must be before DIGIT OR DOT
            if has_sign or has_digit or has_dot:
                return False
            has_sign = True
        elif char == 'e':
            # Must exits DIGIT before EXPONENT
            if has_exponent or not has_digit: return False
            has_exponent = True
            has_sign = False
            has_digit = False
            has_dot = False
        elif char == '.':
            # Dot cannot go after EXPONENT
            if has_exponent or has_dot: return False
            has_dot = True
        elif '0' <= char <= '9':
            has_digit = True
        else:
            return False
    
    return has_digit


class IsValidNumberSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(is_valid_number("123"))  # Integer

    def test_example2(self):
        self.assertTrue(is_valid_number("12.3"))  # Floating point
    
    def test_example3(self):
        self.assertTrue(is_valid_number("-123"))  # Negative numbers
    
    def test_example4(self):
        self.assertTrue(is_valid_number("-.3"))  # Negative floating point
    
    def test_example5(self):
        self.assertTrue(is_valid_number("1.5e5")) # Scientific notation
    
    def test_example6(self):
        self.assertFalse(is_valid_number("12a"))  # No letters
    
    def test_example7(self):
        self.assertFalse(is_valid_number("1 2")) # No space between numbers
    
    def test_example8(self):
        self.assertFalse(is_valid_number("1e1.2")) # Exponent can only be an integer (positive or negative or 0)

    def test_empty_string(self):
        self.assertFalse(is_valid_number(""))

    def test_blank_string(self):
        self.assertFalse(is_valid_number("   "))

    def test_just_signs(self):
        self.assertFalse(is_valid_number("+"))

    def test_zero(self):
        self.assertTrue(is_valid_number("0"))

    def test_contains_no_number(self):
        self.assertFalse(is_valid_number("e"))

    def test_contains_white_spaces(self):
        self.assertTrue(is_valid_number(" -123.456  "))

    def test_scientific_notation(self):
        self.assertTrue(is_valid_number("2e10"))

    def test_scientific_notation2(self):
        self.assertFalse(is_valid_number("10e5.4"))

    def test_scientific_notation3(self):
        self.assertTrue(is_valid_number("-24.35e-10"))

    def test_scientific_notation4(self):
        self.assertFalse(is_valid_number("1e1e1"))

    def test_scientific_notation5(self):
        self.assertTrue(is_valid_number("+.5e-23"))

    def test_scientific_notation6(self):
        self.assertFalse(is_valid_number("+e-23"))

    def test_scientific_notation7(self):
        self.assertFalse(is_valid_number("0e"))

    def test_multiple_signs(self):
        self.assertFalse(is_valid_number("+-2"))

    def test_multiple_signs2(self):
        self.assertFalse(is_valid_number("-2-2-2-2"))

    def test_multiple_signs3(self):
        self.assertFalse(is_valid_number("6+1"))

    def test_multiple_dots(self):
        self.assertFalse(is_valid_number("10.24.25"))

    def test_sign_and_dot(self):
        self.assertFalse(is_valid_number(".-4"))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### June 27, 2021  \[Easy\] Swap Even and Odd Nodes
---
> **Question:** Given the head of a singly linked list, swap every two nodes and return its head.
>
> **Note:** Make sure it’s acutally nodes that get swapped not value.

**Example:**
```py
Given 1 -> 2 -> 3 -> 4, return 2 -> 1 -> 4 -> 3.
```

**Solution:** [https://replit.com/@trsong/Swap-Every-Even-and-Odd-Nodes-in-Linked-List-2](https://replit.com/@trsong/Swap-Every-Even-and-Odd-Nodes-in-Linked-List-2)
```py
import unittest

def swap_list(lst):
    prev = dummy = ListNode(-1, lst)
    
    while prev and prev.next and prev.next.next:
        first = prev.next
        second = first.next

        first.next = second.next
        second.next = first
        prev.next = second

        prev = first
    return dummy.next


class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class SwapListSpec(unittest.TestCase):
    def assert_lists(self, lst, node_seq):
        p = lst
        for node in node_seq:
            if p != node: print (p.data if p else "None"), (node.data if node else "None")
            self.assertTrue(p == node)
            p = p.next
        self.assertTrue(p is None)

    def test_empty(self):
        self.assert_lists(swap_list(None), [])

    def test_one_elem_list(self):
        n1 = ListNode(1)
        self.assert_lists(swap_list(n1), [n1])

    def test_two_elems_list(self):
        # 1 -> 2
        n2 = ListNode(2)
        n1 = ListNode(1, n2)
        self.assert_lists(swap_list(n1), [n2, n1])

    def test_three_elems_list(self):
        # 1 -> 2 -> 3
        n3 = ListNode(3)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)
        self.assert_lists(swap_list(n1), [n2, n1, n3])

    def test_four_elems_list(self):
        # 1 -> 2 -> 3 -> 4
        n4 = ListNode(4)
        n3 = ListNode(3, n4)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)
        self.assert_lists(swap_list(n1), [n2, n1, n4, n3])

    def test_five_elems_list(self):
        # 1 -> 2 -> 3 -> 4 -> 5
        n5 = ListNode(5)
        n4 = ListNode(4, n5)
        n3 = ListNode(3, n4)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)
        self.assert_lists(swap_list(n1), [n2, n1, n4, n3, n5])


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### June 26, 2021 \[Medium\] LRU Cache
---
> **Question:** Implement an LRU (Least Recently Used) cache. It should be able to be initialized with a cache size n, and contain the following methods:
>
> - `put(key, value)`: sets key to value. If there are already n items in the cache and we are adding a new item, then it should also remove the least recently used item.
> - `get(key)`: gets the value at key. If no such key exists, return null.
>  
> Each operation should run in O(1) time.

**Example:**
```py
cache = LRUCache(2)
cache.put(3, 3)
cache.put(4, 4)
cache.get(3)  # returns 3
cache.get(2)  # returns None

cache.put(2, 2)
cache.get(4)  # returns None (pre-empted by 2)
cache.get(3)  # returns 3
```

**Solution:** [https://replit.com/@trsong/Design-LRU-Cache-2](https://replit.com/@trsong/Design-LRU-Cache-2)
```py
import unittest

class LRUCache(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.lookup = {}
        self.list = DoubleLinkedList()

    def get(self, key):
        if key not in self.lookup:
            return None

        node = self.lookup[key]
        self.list.remove(node)
        self.list.append(node)
        return node.data.val

    def put(self, key, val):
        if key in self.lookup:
            self.list.remove(self.lookup[key])
        
        self.lookup[key] = ListNode(CacheEntry(key, val))
        self.list.append(self.lookup[key])

        if len(self.lookup) > self.capacity:
            evit_node = self.list.head.next
            del self.lookup[evit_node.data.key]
            self.list.remove(evit_node)


class CacheEntry(object):
    def __init__(self, key, val):
        self.key = key
        self.val = val


class ListNode(object):
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None


class DoubleLinkedList(object):
    def __init__(self):
        self.head = ListNode()
        self.tail = ListNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def append(self, node):
        prev_tail = self.tail.prev

        prev_tail.next = node
        node.next = self.tail

        self.tail.prev = node
        node.prev = prev_tail

    def remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.next = None
        node.prev = None


class LRUCacheSpec(unittest.TestCase):
    def test_example(self):
        cache = LRUCache(2)
        cache.put(3, 3)
        cache.put(4, 4)
        self.assertEqual(3, cache.get(3))
        self.assertIsNone(cache.get(2))

        cache.put(2, 2)
        self.assertIsNone(cache.get(4))  # returns None (pre-empted by 2)
        self.assertEqual(3, cache.get(3))

    def test_get_element_from_empty_cache(self):
        cache = LRUCache(1)
        self.assertIsNone(cache.get(-1))

    def test_cachecapacity_is_one(self):
        cache = LRUCache(1)
        cache.put(-1, 42)
        self.assertEqual(42, cache.get(-1))

        cache.put(-1, 10)
        self.assertEqual(10, cache.get(-1))

        cache.put(2, 0)
        self.assertIsNone(cache.get(-1))
        self.assertEqual(0, cache.get(2))

    def test_evict_most_inactive_element_when_cache_is_full(self):
        cache = LRUCache(3)
        cache.put(1, 1)
        cache.put(1, 1)
        cache.put(2, 1)
        cache.put(3, 1)

        cache.put(4, 1)
        self.assertIsNone(cache.get(1))

        cache.put(2, 1)
        cache.put(5, 1)
        self.assertIsNone(cache.get(3))

    def test_update_element_should_get_latest_value(self):
        cache = LRUCache(2)
        cache.put(3, 10)
        cache.put(3, 42)
        cache.put(1, 1)
        cache.put(1, 2)
        self.assertEqual(42, cache.get(3))

    def test_end_to_end_workflow(self):
        cache = LRUCache(3)
        cache.put(0, 0)  # Least Recent -> 0 -> Most Recent
        cache.put(1, 1)  # Least Recent -> 0, 1 -> Most Recent
        cache.put(2, 2)  # Least Recent -> 0, 1, 2 -> Most Recent
        cache.put(3, 3)  # Least Recent -> 1, 2, 3 -> Most Recent. Evict 0
        self.assertIsNone(cache.get(0))  
        self.assertEqual(2, cache.get(2))  # Least Recent -> 1, 3, 2 -> Most Recent
        cache.put(4, 4)  # Least Recent -> 3, 2, 4 -> Most Recent. Evict 1 
        self.assertIsNone(cache.get(1))
        self.assertEqual(2, cache.get(2))  # Least Recent -> 3, 4, 2 -> Most Recent 
        self.assertEqual(3, cache.get(3))  # Least Recent -> 4, 2, 3 -> Most Recent
        self.assertEqual(2, cache.get(2))  # Least Recent -> 4, 3, 2 -> Most Recent
        cache.put(5, 5)  # Least Recent -> 3, 2, 5 -> Most Recent. Evict 4
        cache.put(6, 6)  # Least Recent -> 2, 5, 6 -> Most Recent. Evict 3
        self.assertIsNone(cache.get(4))
        self.assertIsNone(cache.get(3))
        cache.put(7, 7)  # Least Recent -> 5, 6, 7 -> Most Recent. Evict 2
        self.assertIsNone(cache.get(2))

    def test_end_to_end_workflow2(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        self.assertEqual(cache.get(1), 1)
        cache.put(3, 3)  # evicts key 2
        self.assertIsNone(cache.get(2))
        cache.put(4, 4)  # evicts key 1
        self.assertIsNone(cache.get(1))
        self.assertEqual(3, cache.get(3))
        self.assertEqual(4, cache.get(4))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### June 25, 2021 \[Easy\] Spreadsheet Columns
--- 
> **Question:** In many spreadsheet applications, the columns are marked with letters. From the 1st to the 26th column the letters are A to Z. Then starting from the 27th column it uses AA, AB, ..., ZZ, AAA, etc.
>
> Given a number n, find the n-th column name.

**Examples:**
```py
Input          Output
 26             Z
 51             AY
 52             AZ
 80             CB
 676            YZ
 702            ZZ
 705            AAC
```

**Solution:** [https://replit.com/@trsong/Find-Spreadsheet-Columns-2](https://replit.com/@trsong/Find-Spreadsheet-Columns-2)
```py
import unittest

def spreadsheet_columns(n):
    ord_a = ord('A')
    alphabet_size = 26
    res = []
    while n > 0:
        remainder = n % alphabet_size
        # 1 -> A, 2 -> B, 26 -> Z
        letter = (remainder - 1) % alphabet_size
        res.append(chr(ord_a + letter))
        n //= alphabet_size
        if remainder == 0:
            n -= 1
    res.reverse()
    return "".join(res)

    
class SpreadsheetColumnSpec(unittest.TestCase):
    def test_trivial_example(self):
        self.assertEqual("A", spreadsheet_columns(1))
    
    def test_example1(self):
        self.assertEqual("Z", spreadsheet_columns(26))
    
    def test_example2(self):
        self.assertEqual("AY", spreadsheet_columns(51))
    
    def test_example3(self):
        self.assertEqual("AZ", spreadsheet_columns(52))
    
    def test_example4(self):
        self.assertEqual("CB", spreadsheet_columns(80))
    
    def test_example5(self):
        self.assertEqual("YZ", spreadsheet_columns(676))
    
    def test_example6(self):
        self.assertEqual("ZZ", spreadsheet_columns(702))
    
    def test_example7(self):
        self.assertEqual("AAC", spreadsheet_columns(705))


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```

### June 24, 2021 \[Easy\] Permutations
---
> **Question:** Given a number in the form of a list of digits, return all possible permutations.
>
> For example, given `[1,2,3]`, return `[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]`.

**My thoughts:** total number of permutations equals `n * n-1 * n-2 * ... * 2 * 1`. So in **Step 1**: swap all `n` number with index `0`. And in **Step 2**: swap the rest `n - 1` numbers with index `1` ... and so on.

For problem with size `k`, we swap the `n - k` th element with the result from problem with size `n - k`. 

**Example:**
```py
Suppose the input is [0, 1, 2]
├ swap(0, 0) 
│ ├ swap(1, 1) 
│ │ └ swap(2, 2)  gives [0, 1, 2]
│ └ swap(1, 2)  
│   └ swap(2, 2)  gives [0, 2, 1]
├ swap(0, 1) 
│ ├ swap(1, 1) 
│ │ └ swap(2, 2)  gives [1, 0, 2]
│ └ swap(1, 2)  
│   └ swap(2, 2)  gives [1, 2, 0]
└ swap(0, 2)  
  ├ swap(1, 1)
  │ └ swap(2, 2)  gives [2, 1, 0]
  └ swap(1, 2)
    └ swap(2, 2)  gives [2, 0, 1]
```

**Solution with Backtracking:** [https://replit.com/@trsong/Generate-Permutations-2](https://replit.com/@trsong/Generate-Permutations-2)
```py
import unittest

def generate_permutations(nums):
    res = []
    backtrack(res, 0, nums)
    return res


def backtrack(res, index, nums):
    n = len(nums)
    if index >= n:
        res.append(nums[:])
    else:
        for i in range(index, n):
            nums[i], nums[index] = nums[index], nums[i]
            backtrack(res, index + 1, nums)
            nums[i], nums[index] = nums[index], nums[i]


class CalculatePermutationSpec(unittest.TestCase):
    def test_permuation_of_empty_array(self):
        self.assertEqual( [[]], generate_permutations([]))

    def test_permuation_of_2(self):
        self.assertEqual(
            sorted([[0, 1], [1, 0]]),
            sorted(generate_permutations([0, 1])))

    def test_permuation_of_3(self):
        self.assertEqual(
            sorted([[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]),
            sorted(generate_permutations([1, 2, 3])))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### June 23, 2021 \[Easy\] Implement Prefix Map Sum
---
> **Question:** Implement a PrefixMapSum class with the following methods:
>
> - `insert(key: str, value: int)`: Set a given key's value in the map. If the key already exists, overwrite the value.
> - `sum(prefix: str)`: Return the sum of all values of keys that begin with a given prefix.

**Example:**
```py
mapsum.insert("columnar", 3)
assert mapsum.sum("col") == 3

mapsum.insert("column", 2)
assert mapsum.sum("col") == 5
```

**Solution with Trie:** [https://replit.com/@trsong/Implement-Prefix-Map-Sum-3](https://replit.com/@trsong/Implement-Prefix-Map-Sum-3)
```py
import unittest

class Trie(object):
    def __init__(self):
        self.children = None
        self.accu_sum = 0


class PrefixMap(object):
    def __init__(self):
        self.trie = Trie()
        self.record = {}

    def insert(self, word, val):
        updated_val = val - self.record.get(word, 0) 
        self.record[word] = val

        p = self.trie
        for ch in word:
            p.accu_sum += updated_val
            p.children = p.children or {}
            p.children[ch] = p.children.get(ch, Trie())
            p = p.children[ch]
        p.accu_sum += updated_val

    def sum(self, word):
        p = self.trie
        for ch in word:
            if not p or not p.children or ch not in p.children:
                return 0
            p = p.children[ch]
        return p.accu_sum if p else 0


class PrefixMapSpec(unittest.TestCase):
    def test_example(self):
        prefix_map = PrefixMap()
        prefix_map.insert("columnar", 3)
        self.assertEqual(3, prefix_map.sum("col"))
        prefix_map.insert("column", 2)
        self.assertEqual(5, prefix_map.sum("col"))

    def test_empty_map(self):
        prefix_map = PrefixMap()
        self.assertEqual(0, prefix_map.sum(""))
        self.assertEqual(0, prefix_map.sum("unknown"))

    def test_same_prefix(self):
        prefix_map = PrefixMap()
        prefix_map.insert("a", 1)
        prefix_map.insert("aa", 2)
        prefix_map.insert("aaa", 3)
        self.assertEqual(0, prefix_map.sum("aaaa"))
        self.assertEqual(3, prefix_map.sum("aaa"))
        self.assertEqual(5, prefix_map.sum("aa"))
        self.assertEqual(6, prefix_map.sum("a"))
        self.assertEqual(6, prefix_map.sum(""))

    def test_same_prefix2(self):
        prefix_map = PrefixMap()
        prefix_map.insert("aa", 1)
        prefix_map.insert("a", 2)
        prefix_map.insert("b", 1)
        self.assertEqual(0, prefix_map.sum("aaa"))
        self.assertEqual(1, prefix_map.sum("aa"))
        self.assertEqual(3, prefix_map.sum("a"))
        self.assertEqual(4, prefix_map.sum(""))

    def test_double_prefix(self):
        prefix_map = PrefixMap()
        prefix_map.insert("abc", 1)
        prefix_map.insert("abd", 2)
        prefix_map.insert("abzz", 1)
        prefix_map.insert("bazz", 1)
        self.assertEqual(4, prefix_map.sum("ab"))
        self.assertEqual(0, prefix_map.sum("abq"))
        self.assertEqual(4, prefix_map.sum("a"))
        self.assertEqual(1, prefix_map.sum("b"))

    def test_update_value(self):
        prefix_map = PrefixMap()
        prefix_map.insert('a', 1)
        prefix_map.insert('ab', 1)
        prefix_map.insert('abc', 1)
        prefix_map.insert('ab', 100)
        self.assertEqual(102, prefix_map.sum('a'))


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```

### June 22, 2021 \[Medium\] Circle of Chained Words
---
> **Question:** Two words can be 'chained' if the last character of the first word is the same as the first character of the second word.
>
> Given a list of words, determine if there is a way to 'chain' all the words in a circle.

**Example:**
```py
Input: ['eggs', 'karat', 'apple', 'snack', 'tuna']
Output: True
Explanation:
The words in the order of ['apple', 'eggs', 'snack', 'karat', 'tuna'] creates a circle of chained words.
```

**My thoughts:** Treat each non-empty word as an edge in a directed graph with vertices being the first and last letter of the word. Now, pick up any letter as a starting point. Perform DFS and remove any edge we visited from the graph. Check if all edges are used. And make sure the vertex we stop at is indeed the starting point. If all above statisfied, then there exists a cycle that chains all words. 

**Solution with DFS:** [https://replit.com/@trsong/Contains-Circle-of-Chained-Words-2](https://replit.com/@trsong/Contains-Circle-of-Chained-Words-2)
```py
import unittest

def exists_cycle(words):
    neighbors = {}
    for word in words:
        if not word:
            continue
        start, end = word[0], word[-1]
        neighbors[start] = neighbors.get(start, {})
        neighbors[start][end] = neighbors[start].get(end, 0) + 1

    if not neighbors:
        return False

    start = next(neighbors.iterkeys())
    stack = [start]

    while stack and neighbors:
        cur = stack.pop()
        if cur not in neighbors:
            return False

        nb = next(neighbors[cur].iterkeys())
        neighbors[cur][nb] -= 1
        if neighbors[cur][nb] == 0:
            del neighbors[cur][nb]
            if not neighbors[cur]:
                del neighbors[cur]
        stack.append(nb)
    return not neighbors and len(stack) == 1 and stack[0] == start
    
            

class ExistsCycleSpec(unittest.TestCase):
    def test_example(self):
        words = ['eggs', 'karat', 'apple', 'snack', 'tuna']
        self.assertTrue(exists_cycle(words)) # ['apple', 'eggs', 'snack', 'karat', 'tuna']

    def test_empty_words(self):
        words = []
        self.assertFalse(exists_cycle(words))
    
    def test_not_contains_cycle(self):
        words = ['ab']
        self.assertFalse(exists_cycle(words))

    def test_not_contains_cycle2(self):
        words = ['']
        self.assertFalse(exists_cycle(words))

    def test_not_exist_cycle(self):
        words = ['ab', 'c', 'c', 'def', 'gh']
        self.assertFalse(exists_cycle(words))

    def test_exist_cycle_but_not_chaining_all_words(self):
        words = ['ab', 'be', 'bf', 'bc', 'ca']
        self.assertFalse(exists_cycle(words))
    
    def test_exist_cycle_but_not_chaining_all_words2(self):
        words = ['ab', 'ba', 'bc', 'ca']
        self.assertFalse(exists_cycle(words))

    def test_duplicate_words_with_cycle(self):
        words = ['ab', 'bc', 'ca', 'ab', 'bd', 'da' ]
        self.assertTrue(exists_cycle(words))

    def test_contains_mutiple_cycles(self):
        words = ['ab', 'ba', 'ac', 'ca']
        self.assertTrue(exists_cycle(words))

    def test_disconnect_graph(self):
        words = ['ab', 'ba', 'cd', 'de', 'ec']
        self.assertFalse(exists_cycle(words))

    def test_conains_empty_string(self):
        words2 = ['', 'a', '', '', 'a']
        self.assertTrue(exists_cycle(words2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### June 21, 2021 \[Easy\] Zig-Zag Distinct LinkedList
--- 
> **Question:** Given a linked list with DISTINCT value, rearrange the node values such that they appear in alternating `low -> high -> low -> high ...` form. For example, given `1 -> 2 -> 3 -> 4 -> 5`, you should return `1 -> 3 -> 2 -> 5 -> 4`.


**Solution:** [https://replit.com/@trsong/Zig-Zag-Order-of-Distinct-LinkedList-2](https://replit.com/@trsong/Zig-Zag-Order-of-Distinct-LinkedList-2)
```py
import unittest
import copy

def zig_zag_order(lst):
    should_increase = True
    dummy = prev = ListNode(-1, lst)
    p = lst

    while p and p.next:
        has_increase = p.next.val > p.val
        if should_increase == has_increase:
            prev = p
            p = p.next
        else:
            second = p.next

            p.next = second.next
            second.next = p
            prev.next = second

            prev = second
        should_increase = not should_increase
    return dummy.next
            

###################
# Testing Utilities
###################
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
    
    @staticmethod  
    def List(*vals):
        dummy = ListNode(-1)
        p = dummy
        for elem in vals:
            p.next = ListNode(elem)
            p = p.next
        return dummy.next  

    def __repr__(self):
        return "{} -> {}".format(str(self.val), str(self.next))

    def to_list(self):
        res = []
        p = self
        while p:
            res.append(p.val)
            p = p.next
        return res


class ZigZagOrderSpec(unittest.TestCase):
    def verify_order(self, original_lst):
        lst = zig_zag_order(copy.deepcopy(original_lst))
        self.assertIsNotNone(lst)
        self.assertEqual(set(original_lst.to_list()), set(lst.to_list()))

        isLessThanPrevious = False
        p = lst.next
        prev = lst
        while p:
            if isLessThanPrevious:
                self.assertLess(p.val, prev.val, "%d in %s" % (p.val, lst))
            else:
                self.assertGreater(p.val, prev.val, "%d in %s" % (p.val, lst))

            isLessThanPrevious = not isLessThanPrevious
            prev = p
            p = p.next

    def test_example(self):
        lst = ListNode.List(1, 2, 3, 4, 5)
        self.verify_order(lst)

    def test_empty_array(self):
        self.assertIsNone(zig_zag_order(None))

    def test_unsorted_list1(self):
        lst = ListNode.List(10, 5, 6, 3, 2, 20, 100, 80)
        self.verify_order(lst)

    def test_unsorted_list2(self):
        lst = ListNode.List(2, 4, 6, 8, 10, 20)
        self.verify_order(lst)

    def test_unsorted_list3(self):
        lst = ListNode.List(3, 6, 5, 10, 7, 20)
        self.verify_order(lst)

    def test_unsorted_list4(self):
        lst = ListNode.List(20, 10, 8, 6, 4, 2)
        self.verify_order(lst)

    def test_unsorted_list5(self):
        lst = ListNode.List(6, 4, 2, 1, 8, 3)
        self.verify_order(lst)

    def test_sorted_list(self):
        lst = ListNode.List(6, 5, 4, 3, 2, 1)
        self.verify_order(lst)
    

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### June 20, 2021 \[Easy\] Count Total Set Bits from 1 to n
---
> **Question:** Write an algorithm that finds the total number of set bits in all integers between 1 and N.

**Examples:**
```py
Input: n = 3  
Output:  4
Explanation: The binary representation (01, 10, 11) contains 4 1s.

Input: n = 6
Output: 9
Explanation: The binary representation (01, 10, 11, 100, 101, 110) contains 9 1s.

Input: n = 7
Output: 12

Input: n = 8
Output: 13
```

**Solution with DP:** [https://replit.com/@trsong/Count-Total-Number-of-Set-Bits-from-1-to-n](https://replit.com/@trsong/Count-Total-Number-of-Set-Bits-from-1-to-n)
```py
import unittest

def count_bits(target):
    # Let dp[num] represents number of bits for number num
    dp = [None] * (target + 1)
    dp[0] = 0
    res = 0
    for num in range(1, target + 1):
        if num % 2 == 0:
            dp[num] = dp[num // 2]
        else:
            dp[num] = dp[num - 1] + 1
        res += dp[num]
    return res


class CountBitSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(4, count_bits(3))  # 1, 10, 11

    def test_example2(self):
        self.assertEqual(9, count_bits(6))  # 1, 10, 11, 100, 101, 110

    def test_example3(self):
        self.assertEqual(12, count_bits(7))

    def test_example4(self):
        self.assertEqual(13, count_bits(8))

    def test_example5(self):
        self.assertEqual(35, count_bits(17))

    def test_zero(self):
        self.assertEqual(0, count_bits(0))

    def test_one(self):
        self.assertEqual(1, count_bits(1))

    def test_power_of_two(self):
        self.assertEqual(5121, count_bits(1024))

    def test_all_ones(self):
        self.assertEqual(1024, count_bits(0b11111111))

    def test_mixed_digits(self):
        self.assertEqual(515, count_bits(0b10010101))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)

```


### June 19, 2021 \[Hard\] Construct Cartesian Tree from Inorder Traversal
---
> **Question:** A Cartesian tree with sequence S is a binary tree defined by the following two properties:
>
> - It is heap-ordered, so that each parent value is strictly less than that of its children.
> - An in-order traversal of the tree produces nodes with values that correspond exactly to S.
>
> Given a sequence S, construct the corresponding Cartesian tree.

**Example:**
```py
Given the sequence [3, 2, 6, 1, 9], the resulting Cartesian tree would be:
      1
    /   \   
  2       9
 / \
3   6
```

**My thoughts:** The root of min heap is always the smallest element. In order to maintain the given inorder traversal order: we can find the min element and recursively build the tree based on subarray on the left and right.

**Trivial Solution:** [https://repl.it/@trsong/Construct-Cartesian-Tree-from-Inorder-Traversal](https://repl.it/@trsong/Construct-Cartesian-Tree-from-Inorder-Traversal)
```py
import unittest

def construct_cartesian_tree(nums):
    return construct_cartesian_tree_recur(nums, 0, len(nums) - 1)


def construct_cartesian_tree_recur(nums, start, end):
    if start > end:
        return None

    local_min, local_min_index = find_local_min_and_index(nums, start, end)
    left_res = construct_cartesian_tree_recur(nums, start, local_min_index - 1)
    right_res = construct_cartesian_tree_recur(nums, local_min_index + 1, end)
    return TreeNode(local_min, left_res, right_res)


def find_local_min_and_index(nums, start, end):
    min_val, min_index = nums[start], start
    for i in xrange(start + 1, end + 1):
        if nums[i] < min_val:
            min_val = nums[i]
            min_index = i
    return min_val, min_index
```

**Optimal Solution:** [https://replit.com/@trsong/Construct-Cartesian-Tree-from-Inorder-Traversal-2](https://replit.com/@trsong/Construct-Cartesian-Tree-from-Inorder-Traversal-2)
```py
import unittest

def construct_cartesian_tree(nums):
    if not nums:
        return None
    n = len(nums)
    parent = [None] * n
    left = [None] * n
    right = [None] * n
    root = 0

    for i in xrange(1, n):
        prev = i - 1

        while nums[prev] > nums[i] and prev != root:
            prev = parent[prev]

        if nums[prev] > nums[i]:
            #  
            # 1                     0
            #  \  insert 0 gives   /
            #   2                 1
            #                      \
            #                       2
            left[i] = root
            parent[root] = i
            root = i
        else:
            if right[prev] is not None:
                # 0                     0
                #   \  insert 1 gives    \
                #    2                    1
                #                        /
                #                       2
                left[i] = right[prev]
                parent[right[prev]] = i
            right[prev] = i
            parent[i] = prev
    
    nodes = map(TreeNode, nums)
    for i in xrange(n):
        if left[i] is not None:
            nodes[i].left = nodes[left[i]]
        if right[i] is not None:
            nodes[i].right = nodes[right[i]]
    return nodes[root]



#####################
# Testing Utilities
#####################
class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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
                stack.append((child, depth + 1))
        return "\n" + "".join(res) + "\n"

    def is_heap(self):
        for child in [self.left, self.right]:
            if child and (self.val > child.val or not child.is_heap()):
                return False
        return True

    def traversal(self):
        res = []
        if self.left:
            res.extend(self.left.traversal())
        res.append(self.val)
        if self.right:
            res.extend(self.right.traversal())
        return res


class ConstructCartesianTree(unittest.TestCase):
    def assert_result(self, res, nums):
        self.assertEqual(nums, res.traversal())
        self.assertTrue(res.is_heap(), res)

    def test_example(self):
        """
             1
            / \
           2   9
          / \
         3   6
        """
        nums = [3, 2, 6, 1, 9]
        res = construct_cartesian_tree(nums)
        self.assert_result(res, nums)

    def test_example2(self):
        """
             1
           /   \
          3     5
         / \   /
        9   7 8
               \
                10
               /  \
              12  15
                  / \
                20  18
        """
        nums = [9, 3, 7, 1, 8, 12, 10, 20, 15, 18, 5]
        res = construct_cartesian_tree(nums)
        self.assert_result(res, nums)

    def test_empty_array(self):
        self.assertEqual(None, construct_cartesian_tree([]))

    def test_ascending_array(self):
        """
        1
         \
          2
           \
            3
        """
        nums = [1, 2, 3]
        res = construct_cartesian_tree(nums)
        self.assert_result(res, nums)

    def test_descending_array(self):
        """
              1
             /
            2
           /
          3
         / 
        4
        """
        nums = [1, 2, 3, 4]
        res = construct_cartesian_tree(nums)
        self.assert_result(res, nums)

    def test_ascending_descending_array(self):
        """
          1
         / 
        1   
         \
          2
           \
            2
           /
          3  
        """
        nums = [1, 2, 3, 2, 1]
        res = construct_cartesian_tree(nums)
        self.assert_result(res, nums)

    def test_descending_ascending_array(self):
        """
           1
          / \
         2   2
        /     \
       3       3
        """
        nums = [3, 2, 1, 2, 3]
        res = construct_cartesian_tree(nums)
        self.assert_result(res, nums)   


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### June 18, 2021 \[Medium\] Sorting a List With 3 Unique Numbers
---
> **Question:** Given a list of numbers with only `3` unique numbers `(1, 2, 3)`, sort the list in `O(n)` time.

**Example:**
```py
Input: [3, 3, 2, 1, 3, 2, 1]
Output: [1, 1, 2, 2, 3, 3, 3]
```

**Solution with 3-way Quick Select:** [https://replit.com/@trsong/Sorting-a-List-With-3-Unique-Numbers-2](https://replit.com/@trsong/Sorting-a-List-With-3-Unique-Numbers-2)
```py
import unittest

def sort(nums):
    lo = mid = 0
    hi = len(nums) - 1

    while mid <= hi:
        if nums[mid] == 2:
            mid += 1
        elif nums[mid] == 3:
            nums[mid], nums[hi] = nums[hi], nums[mid]
            hi -= 1
        else:
            nums[mid], nums[lo] = nums[lo], nums[mid]
            lo += 1
            mid += 1
    return nums
        

class SortSpec(unittest.TestCase):
    def test_example(self):
        nums = [3, 3, 2, 1, 3, 2, 1]
        expected = [1, 1, 2, 2, 3, 3, 3]
        self.assertEqual(expected, sort(nums))

    def test_empty_arry(self):
        self.assertEqual([], sort([]))

    def test_descending_array(self):
        nums = [3, 2, 2, 1]
        expected = [1, 2, 2, 3]
        self.assertEqual(expected, sort(nums))

    def test_sorted_array(self):
        nums = [1, 1, 2, 3]
        expected = [1, 1, 2, 3]
        self.assertEqual(expected, sort(nums))

    def test_array_without_one(self):
        nums = [2, 3, 2, 2, 3, 3, 2]
        expected = [2, 2, 2, 2, 3, 3, 3]
        self.assertEqual(expected, sort(nums))

    def test_array_without_two(self):
        nums = [1, 3, 3, 1, 1, 1, 1]
        expected = [1, 1, 1, 1, 1, 3, 3]
        self.assertEqual(expected, sort(nums))
    
    def test_array_without_three(self):
        nums = [2, 1, 1, 1, 2]
        expected = [1, 1, 1, 2, 2]
        self.assertEqual(expected, sort(nums))
    
    def test_last_elem_is_one(self):
        nums = [2, 1, 1]
        expected = [1, 1, 2]
        self.assertEqual(expected, sort(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### June 17, 2021 \[Easy\] String Compression
---
> **Question:** Given an array of characters with repeats, compress it in place. The length after compression should be less than or equal to the original array.

**Example:**
```py
Input: ['a', 'a', 'b', 'c', 'c', 'c']
Output: ['a', '2', 'b', 'c', '3']
```

**Solution with Two Pointers:** [https://replit.com/@trsong/String-Array-Compression](https://replit.com/@trsong/String-Array-Compression)
```py
import unittest

def string_compression(msg):
    n = len(msg)
    count = 1
    slow = 0
    for fast in range(n):
        if fast < n - 1 and msg[fast] == msg[fast + 1]:
            count += 1
        else:
            msg[slow] = msg[fast]
            slow += 1
            if count > 1:
                for ch in str(count):
                    msg[slow] = ch
                    slow += 1
            count = 1
    
    for _ in range(slow, n):
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
    unittest.main(exit=False, verbosity=2)
```


### June 16, 2021 \[Medium\] Minimum Number of Jumps to Reach End
---
> **Question:** You are given an array of integers, where each element represents the maximum number of steps that can be jumped going forward from that element. 
> 
> Write a function to return the minimum number of jumps you must take in order to get from the start to the end of the array.
>
> For example, given `[6, 2, 4, 0, 5, 1, 1, 4, 2, 9]`, you should return `2`, as the optimal solution involves jumping from `6 to 5`, and then from `5 to 9`.

**My thoughts:** Instead of using DP to calculate min step required to reach current index, we can treat this problem as climbing floor with ladders. For each floor you reach, you will get a new ladder with length `i + step[i]`. Now all you need to do is to greedily use the max length ladder you have seen so far and swap to the next one when the current one reaches end. The answer will be the total number of max length ladder you have used. 

**Greedy Solution:** [https://replit.com/@trsong/Calculate-Minimum-Number-of-Jumps-to-Reach-End](https://replit.com/@trsong/Calculate-Minimum-Number-of-Jumps-to-Reach-End)
```py
import unittest

def min_jump_to_reach_end(steps):
    if not steps:
        return None
    
    ladder_usage = 0
    max_ladder = cur_ladder = 0
    for level, jump in enumerate(steps):
        if level > max_ladder:
            # Max ladder cannot reach current level
            return None
        elif level > cur_ladder:
            # Need switch to a different ladder
            cur_ladder = max_ladder
            ladder_usage += 1

        new_ladder = level + jump
        max_ladder = max(max_ladder, new_ladder)
    return ladder_usage


class MinJumpToReachEndSpec(unittest.TestCase):
    def test_example(self):
        steps = [6, 2, 4, 0, 5, 1, 1, 4, 2, 9]
        expected = 2  # 6 -> 5 -> 9
        self.assertEqual(expected, min_jump_to_reach_end(steps))

    def test_empty_steps(self):
        self.assertIsNone(min_jump_to_reach_end([]))
    
    def test_trivial_case(self):
        self.assertEqual(0, min_jump_to_reach_end([0]))

    def test_multiple_ways_to_reach_end(self):
        steps = [1, 3, 5, 6, 8, 12, 17]
        expected = 3  # 1 -> 3 -> 5 -> 17
        self.assertEqual(expected, min_jump_to_reach_end(steps)) 

    def test_should_return_min_step_to_reach_end(self):
        steps = [1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9]
        expected = 3  # 1 -> 3 -> 9 -> 9
        self.assertEqual(expected, min_jump_to_reach_end(steps))

    def test_should_return_min_step_to_reach_end2(self):
        steps = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        expected = 4
        self.assertEqual(expected, min_jump_to_reach_end(steps))

    def test_should_return_min_step_to_reach_end3(self):
        steps = [1, 3, 6, 3, 2, 3, 6, 8, 9, 5]
        expected = 4  # 1 -> 3 -> 6 -> 9 -> 5
        self.assertEqual(expected, min_jump_to_reach_end(steps))

    def test_should_return_min_step_to_reach_end4(self):
        steps = [1, 3, 6, 1, 0, 9]
        expected = 3  # 1 -> 3 -> 6 -> 9
        self.assertEqual(expected, min_jump_to_reach_end(steps))

    def test_unreachable_end(self):
        steps = [1, -2, -3, 4, 8, 9, 11]
        self.assertIsNone(min_jump_to_reach_end(steps))

    def test_unreachable_end2(self):
        steps = [1, 3, 2, -11, 0, 1, 0, 0, -1]
        self.assertIsNone(min_jump_to_reach_end(steps))

    def test_reachable_end(self):
        steps = [1, 3, 6, 10]
        expected = 2  # 1 -> 3 -> 10
        self.assertEqual(expected, min_jump_to_reach_end(steps))

    def test_stop_in_the_middle(self):
        steps = [1, 2, 0, 0, 0, 1000, 1000]
        self.assertIsNone(min_jump_to_reach_end(steps))

    def test_stop_in_the_middle2(self):
        steps = [2, 1, 0, 9]
        self.assertIsNone(min_jump_to_reach_end(steps))

    def test_greedy_solution_fails(self):
        steps = [5, 3, 3, 3, 4, 2, 1, 1, 1]
        expected = 2  # 5 -> 4 -> 1
        self.assertEqual(expected, min_jump_to_reach_end(steps))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### June 15, 2021 LC 403 \[Hard\] Frog Jump
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

**Solution with DFS:** [https://replit.com/@trsong/Solve-Frog-Jump-Problem-2](https://replit.com/@trsong/Solve-Frog-Jump-Problem-2)
```py
import unittest

def can_cross(stones):
    stone_set = set(stones)
    visited = set()
    stack = [(0, 0)]
    goal = stones[-1]

    while stack:
        cur, step = stack.pop()
        if cur == goal:
            return True
        visited.add((cur, step))
        for step_delta in [-1, 0, 1]:
            next_step = step + step_delta
            next_stone = cur + next_step
            if (next_stone >= cur and 
                next_stone in stone_set and 
                (next_stone, next_step) not in visited):
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
    unittest.main(exit=False, verbosity=2)
```

### June 14, 2021 LC 228 \[Easy\] Extract Range
---
> **Question:** Given a sorted list of numbers, return a list of strings that represent all of the consecutive numbers.

**Example:**
```py
Input: [0, 1, 2, 5, 7, 8, 9, 9, 10, 11, 15]
Output: ['0->2', '5', '7->11', '15']
```

**Solution:** [https://replit.com/@trsong/Extract-Range-2](https://replit.com/@trsong/Extract-Range-2)
```py
import unittest

def extract_range(nums):
    if not nums:
        return []
    prev = float('inf')
    record = nums[0]
    res = []
    for num in nums:
        if num - prev > 1:
            res.append(generate_interval(record, prev))
            record = num
        prev = num
    res.append(generate_interval(record, prev))
    return res
    

def generate_interval(start, end):
    if start == end:
        return str(start)
    else:
        return "%d->%d" % (start, end)


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
    unittest.main(exit=False, verbosity=2)
```


### June 13, 2021 \[Easy\] Find the K-th Largest Number
---
> **Question:** Find the k-th largest number in a sequence of unsorted numbers. Can you do this in linear time?

**Example:**
```py
Input: 3, [8, 7, 2, 3, 4, 1, 5, 6, 9, 0]
Output: 7
```

**Solution with Quick Select:** [https://replit.com/@trsong/Find-the-K-th-Largest-Number-2](https://replit.com/@trsong/Find-the-K-th-Largest-Number-2)
```py
import unittest
import random

def find_kth_max(nums, k):
    n = len(nums)
    if k > n:
        return None

    lo, hi = 0, n - 1
    while True:
        pivot = quick_select(nums, lo, hi)
        if pivot == n - k:
            return nums[pivot]
        elif pivot < n - k:
            lo = pivot + 1
        else:
            hi = pivot - 1
    return None


def quick_select(nums, lo, hi):
    pivot = random.randint(lo, hi)
    pivot_num = nums[pivot]
    nums[hi], nums[pivot] = nums[pivot], nums[hi]
    
    i = lo
    for j in range(lo, hi):
        if nums[j] < pivot_num:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
    
    nums[i], nums[hi] = nums[hi], nums[i]
    return i

    
class FindKthMaxSpec(unittest.TestCase):
    def test_example(self):
        k, nums = 3, [8, 7, 2, 3, 4, 1, 5, 6, 9, 0]
        expected = 7
        self.assertEqual(expected, find_kth_max(nums, k))

    def test_k_out_of_bound(self):
        k, nums = 4, [1, 2, 3]
        self.assertIsNone(find_kth_max(nums, k))

    def test_find_max(self):
        k, nums = 1, [1, 2, 3]
        expected = 3
        self.assertEqual(expected, find_kth_max(nums, k))

    def test_find_min(self):
        k, nums = 5, [1, 2, 3, 4, 5]
        expected = 1
        self.assertEqual(expected, find_kth_max(nums, k))

    def test_array_with_duplicated_elements(self):
        k, nums = 3, [1, 1, 3, 5, 5]
        expected = 3
        self.assertEqual(expected, find_kth_max(nums, k))

    def test_array_with_duplicated_elements2(self):
        k, nums = 4, [1, 1, 1, 1, 1, 1, 1, 1]
        expected = 1
        self.assertEqual(expected, find_kth_max(nums, k)) 

    def test_array_with_duplicated_elements3(self):
        k, nums = 2, [1, 2, 3, 1, 2, 3, 1, 2, 3]
        expected = 3
        self.assertEqual(expected, find_kth_max(nums, k)) 


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### June 12, 2021 \[Medium\] Toss Biased Coin
---
> **Question:** Assume you have access to a function toss_biased() which returns 0 or 1 with a probability that's not 50-50 (but also not 0-100 or 100-0). You do not know the bias of the coin. Write a function to simulate an unbiased coin toss.

**Solution:** [https://replit.com/@trsong/Toss-Biased-Coins](https://replit.com/@trsong/Toss-Biased-Coins)
```py
from random import randint

def toss_unbiased():
    # Let P(T1, T2) represents probability to get T1, T2 in first and second toss: 
    # P(0, 0) = p * p
    # P(1, 1) = (1 - p) * (1 - p)
    # P(1, 0) = (1 - p) * p
    # P(0, 1) = p * (1 - p)
    # Notice that P(1, 0) = P(0, 1)
    while True:
        t1 = toss_biased()
        t2 = toss_biased()
        if t1 != t2:
            return t1
    

def toss_biased():
    # suppose the toss has 1/4 chance to get 0 and 3/4 to get 1
    return 0 if randint(0, 3) == 0 else 1


def print_distribution(repeat):
    histogram = {}
    for _ in range(repeat):
        res = toss_unbiased()
        if res not in histogram:
            histogram[res] = 0
        histogram[res] += 1
    print(histogram)


if __name__ == '__main__':
     # Distribution looks like {0: 99931, 1: 100069}
    print_distribution(repeat=200000)
```

### June 11, 2021  \[Hard\] Minimum Cost to Construct Pyramid with Stones
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

**Solution:** [https://replit.com/@trsong/Minimum-Cost-to-Construct-Pyramid-with-Stones-2](https://replit.com/@trsong/Minimum-Cost-to-Construct-Pyramid-with-Stones-2)
```py
import unittest

def min_cost_pyramid(stones):
    if not stones:
        return 0

    n = len(stones)
    left_heights = [0] * n
    right_heights = [0] * n

    left_heights[0] = min(1, stones[0])
    right_heights[n - 1] = min(1, stones[n - 1])

    for i in range(1, n):
        left_heights[i] = min(left_heights[i - 1] + 1, stones[i])
        right_heights[n - 1 - i] = min(right_heights[n - i] + 1, stones[n - 1 - i])

    actual_height = max(map(min, left_heights, right_heights))
    stone_usage = actual_height * actual_height
    return sum(stones) - stone_usage


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
    unittest.main(exit=False, verbosity=2)
```


### June 10, 2021 \[Medium\] Maximum Edge Removal to Make Even Forest
---
> **Questions:** You are given a tree with an even number of nodes. Consider each connection between a parent and child node to be an "edge". You would like to remove some of these edges, such that the disconnected subtrees that remain each have an even number of nodes.
>
> For example, suppose your input was the following tree:

```py
   1
  / \ 
 2   3
    / \ 
   4   5
 / | \
6  7  8
```

> In this case, removing the edge (3, 4) satisfies our requirement.
>
> Write a function that returns the maximum number of edges you can remove while still satisfying this requirement.

**Solution with DFS:** [https://replit.com/@trsong/Maximum-Edge-Removal-to-Make-Even-Forest](https://replit.com/@trsong/Maximum-Edge-Removal-to-Make-Even-Forest)
```py
import unittest
from collections import defaultdict
from functools import reduce

def max_edge_removal(graph):
    if not graph:
        return 0

    node_count_map = defaultdict(int)
    start = next(iter(graph))
    count_nodes_recur(start, graph, node_count_map)

    # count node with odd number of children
    return reduce(
        lambda accu, node: accu + 1 if node_count_map[node] % 2 else accu,
        node_count_map.keys(), 0)


def count_nodes_recur(node, graph, node_count_map):
    res = 1
    for child in graph[node]:
        num_nodes = count_nodes_recur(child, graph, node_count_map)
        res += num_nodes
        node_count_map[child] += num_nodes - 1
    return res


class MaxEdgeRemovalSpec(unittest.TestCase):
    def test_example(self):
        """
           1
          / \ 
         2   3
            / \ 
           4   5
         / | \
        6  7  8
        """
        graph = {
            1: [2, 3],
            2: [],
            3: [4, 5],
            4: [6, 7, 8],
            5: [],
            6: [],
            7: [],
            8: []
        }
        expected = 2  # remove edge (3, 4) and (1, 3)
        self.assertEqual(expected, max_edge_removal(graph))

    def test_example2(self):
        """
            0
           /|\
          2 4 1
         /  |
        3   5
           / \
          6   7
        """
        graph = {
            0: [2, 4, 1],
            1: [],
            2: [3],
            3: [],
            4: [5],
            5: [6, 7],
            6: [],
            7: []
        }
        expected = 2  # remove edge (0, 2) and (0, 4)
        self.assertEqual(expected, max_edge_removal(graph))

    def test_empty_graph(self):
        self.assertEqual(0, max_edge_removal({}))

    def test_one_node_graph(self):
        self.assertEqual(0, max_edge_removal({0: []}))

    def test_no_need_to_remove(self):
        """
          0
         /|\
        1 2 3
        """
        graph = {
            0: [1, 2, 3],
            1: [],
            2: [],
            3: []
        }
        expected = 0
        self.assertEqual(expected, max_edge_removal(graph))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### June 9, 2021 \[Easy\] ZigZag Binary Tree
---
> **Questions:** In Ancient Greece, it was common to write text with the first line going left to right, the second line going right to left, and continuing to go back and forth. This style was called "boustrophedon".
>
> Given a binary tree, write an algorithm to print the nodes in boustrophedon order.

**Example:**
```py
Given the following tree:
       1
    /     \
  2         3
 / \       / \
4   5     6   7
You should return [1, 3, 2, 4, 5, 6, 7].
```

**Solution with BFS:** [https://replit.com/@trsong/ZigZag-Order-of-Binary-Tree-2](https://replit.com/@trsong/ZigZag-Order-of-Binary-Tree-2)
```py
import unittest
from queue import deque

def zigzag_traversal(tree):
    if not tree:
        return []

    reverse_order = False
    dq = deque([tree])
    res = []

    while dq:
        if reverse_order:
            res.extend(reversed(dq))
        else:
            res.extend(dq)
            
        for _ in range(len(dq)):
            cur = dq.popleft()
            if cur.left:
                dq.append(cur.left)
            if cur.right:
                dq.append(cur.right)
        reverse_order = not reverse_order
    return list(map(lambda node: node.val, res))
    

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ZigzagTraversalSpec(unittest.TestCase):
    def test_example(self):
        """
               1
            /     \
          2         3
         / \       / \
        4   5     6   7
        """
        left = TreeNode(2, TreeNode(4), TreeNode(5))
        right = TreeNode(3, TreeNode(6), TreeNode(7))
        root = TreeNode(1, left, right)
        expected_traversal = [1, 3, 2, 4, 5, 6, 7]
        self.assertEqual(expected_traversal, zigzag_traversal(root))

    def test_empty(self):
        self.assertEqual([], zigzag_traversal(None))

    def test_right_heavy_tree(self):
        """
            3
           / \
          9  20
            /  \
           15   7
        """
        n20 = TreeNode(20, TreeNode(15), TreeNode(7))
        n3 = TreeNode(3, TreeNode(9), n20)
        expected_traversal = [3, 20, 9, 15, 7]
        self.assertEqual(expected_traversal, zigzag_traversal(n3))
    
    def test_complete_tree(self):
        """
             1
           /   \
          3     2
         / \   /  
        4   5 6  
        """
        n3 = TreeNode(3, TreeNode(4), TreeNode(5))
        n2 = TreeNode(2, TreeNode(6))
        n1 = TreeNode(1, n3, n2)
        expected_traversal = [1, 2, 3, 4, 5, 6]
        self.assertEqual(expected_traversal, zigzag_traversal(n1))

    def test_sparse_tree(self):
        """
             1
           /   \
          3     2
           \   /  
            4 5
           /   \
          7     6
           \   /  
            8 9
        """
        n3 = TreeNode(3, right=TreeNode(4, TreeNode(7, right=TreeNode(8))))
        n2 = TreeNode(2, TreeNode(5, right=TreeNode(6, TreeNode(9))))
        n1 = TreeNode(1, n3, n2)
        expected_traversal = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(expected_traversal, zigzag_traversal(n1))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### June 8, 2021 \[Hard\] Find Next Greater Permutation
---
> **Question:** Given a number represented by a list of digits, find the next greater permutation of a number, in terms of lexicographic ordering. If there is not greater permutation possible, return the permutation with the lowest value/ordering.
>
> For example, the list `[1,2,3]` should return `[1,3,2]`. The list `[1,3,2]` should return `[2,1,3]`. The list `[3,2,1]` should return `[1,2,3]`.
>
> Can you perform the operation without allocating extra memory (disregarding the input memory)?

**My thoughts:** Imagine the list as a number, if it’s in descending order, then there will be no number greater than that and we have to return the number in ascending order, that is, the smallest number. e.g. `321` will become `123`.

Leave first part untouched. If the later part of array are first increasing then decreasing, like `1321`, then based on previous observation, we know the descending part will change from largest to smallest, we want the last increasing digit to increase as little as possible, i.e. slightly larger number on the right. e.g. `2113`

Here are all the steps:

1. Find last increase number, name it _pivot_
2. Find the slightly larger number _pivot_plus_. i.e. the smallest one among all number greater than the last increase number on the right
3. Swap the slightly larger number _pivot_plus_ with last increase number _pivot_
4. Turn the descending array on right to be ascending array


**Solution:** [https://replit.com/@trsong/Find-the-Next-Greater-Permutation-2](https://replit.com/@trsong/Find-the-Next-Greater-Permutation-2)
```py
import unittest

def find_next_permutation(nums):
    if not nums:
        return nums

    n = len(nums)
    pivot = n - 2
    while pivot >= 0 and nums[pivot] >= nums[pivot + 1]:
        pivot -= 1

    if pivot >= 0:
        pivot_plus = pivot
        for i in range(pivot + 1, n):
            if nums[i] <= nums[pivot]:
                break
            pivot_plus = i
        nums[pivot], nums[pivot_plus] = nums[pivot_plus], nums[pivot]
    
    lo, hi = pivot + 1, n - 1
    while lo < hi:
        nums[lo], nums[hi] = nums[hi], nums[lo]
        lo += 1
        hi -= 1
    return nums


class FindNextPermutationSpec(unittest.TestCase):
    def test_example1(self):
        nums = [1, 2, 3]
        expected = [1, 3, 2]
        self.assertEqual(expected, find_next_permutation(nums))
    
    def test_example2(self):
        nums = [1, 3, 2]
        expected = [2, 1, 3]
        self.assertEqual(expected, find_next_permutation(nums))

    def test_example3(self):
        nums = [3, 2, 1]
        expected = [1, 2, 3]
        self.assertEqual(expected, find_next_permutation(nums))

    def test_empty_array(self):
        self.assertEqual([], find_next_permutation([]))

    def test_one_elem_array(self):
        self.assertEqual([1], find_next_permutation([1]))

    def test_decrease_increase_decrease_array(self):
        nums = [3, 2, 1, 6, 5, 4]
        expected = [3, 2, 4, 1, 5, 6]
        self.assertEqual(expected, find_next_permutation(nums))       

    def test_decrease_increase_decrease_array2(self):    
        nums = [3, 2, 4, 6, 5, 4]
        expected = [3, 2, 5, 4, 4, 6]
        self.assertEqual(expected, find_next_permutation(nums))

    def test_increasing_decreasing_increasing_array(self):
        nums = [4, 5, 6, 1, 2, 3]
        expected = [4, 5, 6, 1, 3, 2]
        self.assertEqual(expected, find_next_permutation(nums))

    def test_increasing_decreasing_increasing_array2(self):
        nums =     [1, 1, 2, 3, 4, 4, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 3, 2, 1]
        expected = [1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10]
        self.assertEqual(expected, find_next_permutation(nums))

    def test_multiple_decreasing_and_increasing_array(self):
        nums = [5, 3, 4, 9, 7, 6]
        expected = [5, 3, 6, 4, 7, 9]
        self.assertEqual(expected, find_next_permutation(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### June 7, 2021 \[Medium\] Maximum Number of Connected Colors
---
> **Question:** Given a grid with cells in different colors, find the maximum number of same color  cells that are connected.
>
> Note: two cells are connected if they are of the same color and adjacent to each other: left, right, top or bottom. To stay simple, we use integers to represent colors:
>
> The following grid have max 4 connected colors. [color 3: (1, 2), (1, 3), (2, 1), (2, 2)]

```py
 [
    [1, 1, 2, 2, 3], 
    [1, 2, 3, 3, 1],
    [2, 3, 3, 1, 2]
 ]
```

**My thoughts:** Perform BFS/DFS or Union-Find on all unvisited cells, count its neighbors of same color and mark them as visited.

**Solution with DFS:** [https://replit.com/@trsong/Find-Maximum-Number-of-Connected-Colors-2](https://replit.com/@trsong/Find-Maximum-Number-of-Connected-Colors-2)
```py
import unittest

def max_connected_colors(grid):
    if not grid or not grid[0]:
        return 0
    
    n, m = len(grid), len(grid[0])
    visited = [[False for _ in range(m)] for _ in range(n)]
    res = 0
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    for r in range(n):
        for c in range(m):
            if visited[r][c]:
                continue

            start_color = grid[r][c]
            num_colors = 0
            stack = [(r, c)]
            while stack:
                cur_r, cur_c = stack.pop()
                if visited[cur_r][cur_c]:
                    continue
                num_colors += 1
                visited[cur_r][cur_c] = True

                for dr, dc in directions:
                    new_r, new_c = cur_r + dr, cur_c + dc
                    if (0 <= new_r < n and 
                        0 <= new_c < m and 
                        not visited[new_r][new_c] and 
                        grid[new_r][new_c] == start_color):
                        stack.append((new_r, new_c))
            res = max(res, num_colors)
    return res


class MaxConnectedColorSpec(unittest.TestCase):
    def test_empty_graph(self):
        self.assertEqual(max_connected_colors([[]]), 0)   

    def test_example(self):
        self.assertEqual(max_connected_colors([
            [1, 1, 2, 2, 3],
            [1, 2, 3, 3, 1],
            [2, 3, 3, 1, 2]
        ]), 4)

    def test_disconnected_colors(self):
        self.assertEqual(max_connected_colors([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]), 1)

    def test_cross_shap(self):
        self.assertEqual(max_connected_colors([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ]), 5)

    def test_boundary(self):
        self.assertEqual(max_connected_colors([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]), 8)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### June 6, 2021 LC 1155 \[Medium\] Number of Dice Rolls With Target Sum
---
> **Question:** You have `d` dice, and each die has `f` faces numbered `1, 2, ..., f`.
>
> Return the number of possible ways (out of `f^d` total ways) modulo `10^9 + 7` to roll the dice so the sum of the face up numbers equals target.


**Example 1:**
```py
Input: d = 1, f = 6, target = 3
Output: 1
Explanation: 
You throw one die with 6 faces.  There is only one way to get a sum of 3.
```

**Example 2:**
```py
Input: d = 2, f = 6, target = 7
Output: 6
Explanation: 
You throw two dice, each with 6 faces.  There are 6 ways to get a sum of 7:
1+6, 2+5, 3+4, 4+3, 5+2, 6+1.
```

**Example 3:**
```py
Input: d = 2, f = 5, target = 10
Output: 1
Explanation: 
You throw two dice, each with 5 faces.  There is only one way to get a sum of 10: 5+5.
```

**Example 4:**
```py
Input: d = 1, f = 2, target = 3
Output: 0
Explanation: 
You throw one die with 2 faces.  There is no way to get a sum of 3.
```

**Example 5:**
```py
Input: d = 30, f = 30, target = 500
Output: 222616187
Explanation: 
The answer must be returned modulo 10^9 + 7.
```

**Solution with DP:** [https://replit.com/@trsong/Number-of-Dice-Rolls-With-Target-Sum-2](https://replit.com/@trsong/Number-of-Dice-Rolls-With-Target-Sum-2)
```py
import unittest

MODULE_NUM = 1000000007 

def throw_dice(d, f, target):
    if d * f < target:
        return 0
    return throw_dice_with_cache(d, f, target, {})


def throw_dice_with_cache(d, f, target, cache):
    if d == 0 and target == 0:
        return 1
    elif d == 0:
        return 0

    if (d, target) not in cache:
        res = 0
        for v in range(1, f + 1):
            res += throw_dice_with_cache(d - 1, f, target - v, cache)
        res %= MODULE_NUM
        cache[(d, target)] = res
    return cache[(d, target)]


class ThrowDiceSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(15, throw_dice(3, 6, 7))

    def test_example1(self):
        self.assertEqual(1, throw_dice(1, 6, 3))

    def test_example2(self):
        self.assertEqual(6, throw_dice(2, 6, 7))

    def test_example3(self):
        self.assertEqual(1, throw_dice(2, 5, 10))

    def test_example4(self):
        self.assertEqual(0, throw_dice(1, 2, 3))

    def test_example5(self):
        self.assertEqual(222616187, throw_dice(30, 30, 500))

    def test_target_total_too_large(self):
        self.assertEqual(0, throw_dice(1, 6, 12))

    def test_target_total_too_small(self):
        self.assertEqual(0, throw_dice(4, 2, 1))

    def test_throw_dice1(self):
        self.assertEqual(2, throw_dice(2, 2, 3))

    def test_throw_dice2(self):
        self.assertEqual(21, throw_dice(6, 3, 8))

    def test_throw_dice3(self):
        self.assertEqual(4, throw_dice(4, 2, 5))

    def test_throw_dice4(self):
        self.assertEqual(6, throw_dice(3, 4, 5))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### June 5, 2021 \[Easy\] Step Word Anagram
---
> **Question:** A step word is formed by taking a given word, adding a letter, and anagramming the result. For example, starting with the word `"APPLE"`, you can add an `"A"` and anagram to get `"APPEAL"`.
>
> Given a dictionary of words and an input word, create a function that returns all valid step words.


**Solution:** [https://replit.com/@trsong/Step-Word-Anagram-2](https://replit.com/@trsong/Step-Word-Anagram-2)
```py
import unittest

def find_step_anagrams(word, dictionary):
    histogram = {}
    for ch in word:
        histogram[ch] = histogram.get(ch, 0) + 1
    
    res = []
    for anagram in dictionary:
        if len(anagram) - len(word) != 1:
            continue
            
        for ch in anagram:
            histogram[ch] = histogram.get(ch, 0) - 1
            if histogram[ch] == 0:
                del histogram[ch]

        if len(histogram) == 1:
            res.append(anagram)

        for ch in anagram:
            histogram[ch] = histogram.get(ch, 0) + 1
            if histogram[ch] == 0:
                del histogram[ch]
    return res

        
class FindStepAnagramSpec(unittest.TestCase):
    def test_example(self):
        word = 'APPLE'
        dictionary = ['APPEAL', 'CAPPLE', 'PALPED']
        expected = ['APPEAL', 'CAPPLE', 'PALPED']
        self.assertEqual(sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_empty_word(self):
        word = ''
        dictionary = ['A', 'B', 'AB', 'ABC']
        expected = ['A', 'B']
        self.assertEqual(sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_empty_dictionary(self):
        word = 'ABC'
        dictionary = []
        expected = []
        self.assertEqual(sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_no_match(self):
        word = 'ABC'
        dictionary = ['BBB', 'ACCC']
        expected = []
        self.assertEqual(sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_no_match2(self):
        word = 'AA'
        dictionary = ['ABB']
        expected = []
        self.assertEqual(sorted(expected), sorted(find_step_anagrams(word, dictionary)))

    def test_repeated_chars(self):
        word = 'AAA'
        dictionary = ['A', 'AA', 'AAA', 'AAAA', 'AAAAB', 'AAB', 'AABA']
        expected = ['AAAA', 'AABA']
        self.assertEqual(sorted(expected), sorted(find_step_anagrams(word, dictionary)))



if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### June 4, 2021  LC 872 \[Easy\] Leaf-Similar Trees
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

**Solution with DFS:** [https://replit.com/@trsong/Leaf-Similar-Trees-2](https://replit.com/@trsong/Leaf-Similar-Trees-2)
```py
import unittest

def is_leaf_similar(t1, t2):
    traversal1 = dfs_leaf_traversal(t1)
    traversal2 = dfs_leaf_traversal(t2)
    return traversal1 == traversal2


def dfs_leaf_traversal(root):
    if not root:
        return []
    
    stack = [root]
    res = []
    while stack:
        cur = stack.pop()
        if not cur.left and not cur.right:
            res.append(cur.val)
        
        if cur.right:
            stack.append(cur.right)
        if cur.left:
            stack.append(cur.left)
    
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


### June 3, 2021 \[Hard\] Find Next Sparse Number
---
> **Question:** We say a number is sparse if there are no adjacent ones in its binary representation. For example, `21 (10101)` is sparse, but `22 (10110)` is not. 
> 
> For a given input `N`, find the smallest sparse number greater than or equal to `N`.
>
> Do this in faster than `O(N log N)` time.

**My thoughts:** Whenever we see sub-binary string `011` mark it as `100` and set all bit on the right to 0. eg. `100110` => `101000`, `101101` => `1000000`

**Solution:** [https://replit.com/@trsong/Find-the-Next-Spare-Number-2](https://replit.com/@trsong/Find-the-Next-Spare-Number-2)
```py
import unittest

def next_sparse_number(num):
    window = 0b111
    match_pattern = 0b011

    last_match = i = 0
    while (match_pattern << i) <= num:
        if (window << i) & num == (match_pattern << i):
            num ^= window << i
            last_match = i
        i += 1
    
    num &= ~0 << last_match
    return num


class NextSparseNumberSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(0b10101, next_sparse_number(0b10101))

    def test_no_bit_is_set(self):
        self.assertEqual(0, next_sparse_number(0))

    def test_next_sparse_is_itself(self):
        self.assertEqual(0b100, next_sparse_number(0b100))

    def test_adjacent_bit_is_set(self):
        self.assertEqual(0b1000, next_sparse_number(0b110))
    
    def test_adjacent_bit_is_set2(self):
        self.assertEqual(0b101000, next_sparse_number(0b100110))

    def test_bit_shift_cause_another_bit_shift(self):
        self.assertEqual(0b1000000, next_sparse_number(0b101101))
    
    def test_complicated_number(self):
        self.assertEqual(0b1010010000000, next_sparse_number(0b1010001011101))

    def test_all_bit_is_one(self):
        self.assertEqual(0b1000, next_sparse_number(0b111))


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
```

### June 2, 2021 \[Medium\] Find Next Biggest Integer
---
> **Question:** Given an integer `n`, find the next biggest integer with the same number of 1-bits on. For example, given the number `6 (0110 in binary)`, return `9 (1001)`.

> **Question:** Given an integer `n`, find the next biggest integer with the same number of 1-bits on. For example, given the number `6 (0110 in binary)`, return `9 (1001)`.

**My thoughts:** The idea is to find the leftmost of rightmost ones, swap it with left zero and push remaining rightmost ones all the way till the end.

**Example:**
```py
   10011100
      ^      swap with left zero
=> 10101100 
       ^^    push till the end
=> 10100011 
```

**Solution:** [https://replit.com/@trsong/Find-the-Next-Biggest-Integer-2](https://replit.com/@trsong/Find-the-Next-Biggest-Integer-2)
```py
import unittest

def next_higher_number(num):
    if num == 0:
        return None

    # Step1: count last chunk of 1s
    last_one = num & -num
    count_ones = 0
    while num & last_one:
        num ^= last_one
        last_one <<= 1
        count_ones += 1
    
    # Step2: shift 1st bit of last chunk to left by 1 position
    num |= last_one

    # Step3: pull rest of last chunk all the way right 
    if count_ones > 0:
        num |= (1 << count_ones - 1) - 1
    return num


class NextHigherNumberSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(bin(expected), bin(result))

    def test_example(self):
        self.assert_result(0b1001, next_higher_number(0b0110))

    def test_example2(self):
        self.assert_result(0b110, next_higher_number(0b101))

    def test_example3(self):
        self.assert_result(0b1101, next_higher_number(0b1011))

    def test_zero(self):
        self.assertIsNone(next_higher_number(0))

    def test_end_in_one(self):
        self.assert_result(0b10, next_higher_number(0b01))

    def test_end_in_one2(self):
        self.assert_result(0b1011, next_higher_number(0b111))

    def test_end_in_one3(self):
        self.assert_result(0b110001101101, next_higher_number(0b110001101011))

    def test_end_in_zero(self):
        self.assert_result(0b100, next_higher_number(0b010))

    def test_end_in_zero2(self):
        self.assert_result(0b1000011, next_higher_number(0b0111000))

    def test_end_in_zero3(self):
        self.assert_result(0b1101110001, next_higher_number(0b1101101100))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### June 1, 2021 \[Medium\] Integer Exponentiation
---
> **Question:** Implement integer exponentiation. That is, implement the `pow(x, y)` function, where `x` and `y` are integers and returns `x^y`.
>
> Do this faster than the naive method of repeated multiplication.
>
> For example, `pow(2, 10)` should return `1024`.

**Solution:** [https://replit.com/@trsong/Implement-Integer-Exponentiation](https://replit.com/@trsong/Implement-Integer-Exponentiation)
```py
import unittest

def pow(x, y):
    if y == 0:
        return 1
    elif y < 0:
        return 1.0 / pow(x, -y)
    elif y % 2 == 0:
        return pow(x * x, y / 2)
    else:
        return x * pow(x, y - 1)


class PowSpec(unittest.TestCase):
    def test_power_of_zero(self):
        self.assertAlmostEqual(1, pow(-2, 0))

    def test_power_of_zero2(self):
        self.assertAlmostEqual(1, pow(3, 0))

    def test_power_of_zero3(self):
        self.assertAlmostEqual(1, pow(0, 0))
        
    def test_power_of_zero4(self):
        self.assertAlmostEqual(1, pow(0.5, 0))
        
    def test_power_of_zero5(self):
        self.assertAlmostEqual(1, pow(0.6, 0))

    def test_negative_power(self):
        self.assertAlmostEqual(0.25, pow(-2, -2))
    
    def test_negative_power2(self):
        self.assertAlmostEqual(4, pow(0.5, -2))

    def test_negative_power3(self):
        self.assertAlmostEqual(1.0/27, pow(3, -3))

    def test_positive_power(self):
        self.assertAlmostEqual(4, pow(-2, 2))

    def test_positive_power2(self):
        self.assertAlmostEqual(1024, pow(2, 10))

    def test_positive_power3(self):
        self.assertAlmostEqual(0.25, pow(-0.5, 2))

    def test_positive_power4(self):
        self.assertAlmostEqual(-1.0/512, pow(-2, -9))

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 31, 2021 LC 417 \[Medium\] Pacific Atlantic Water Flow
---
> **Question:** You are given an m x n integer matrix heights representing the height of each unit cell in a continent. The Pacific ocean touches the continent's left and top edges, and the Atlantic ocean touches the continent's right and bottom edges.
>
> Water can only flow in four directions: up, down, left, and right. Water flows from a cell to an adjacent one with an equal or lower height.
>
> Return a list of grid coordinates where water can flow to both the Pacific and Atlantic oceans.

**Example 1:**
```py
Input: heights = [
    [1,2,2,3,5],
    [3,2,3,4,4],
    [2,4,5,3,1],
    [6,7,1,4,5],
    [5,1,1,2,4]]
Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
```

**Example 2:**
```py
Input: heights = [
    [2,1],
    [1,2]]
Output: [[0,0],[0,1],[1,0],[1,1]]
```

**Solution with DFS:** [https://replit.com/@trsong/Pacific-Atlantic-Water-Flow](https://replit.com/@trsong/Pacific-Atlantic-Water-Flow)
```py
import unittest

DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

def ocean_channel(grid):
    if not grid or not grid[0]:
        return []

    n, m = len(grid), len(grid[0])
    pacific_highland = [[False for _ in range(m)] for _ in range(n)]
    atlantic_highland = [[False for _ in range(m)] for _ in range(n)]

    for r in range(n):
        dfs_highland(grid, (r, 0), pacific_highland)
        dfs_highland(grid, (r, m - 1), atlantic_highland)

    for c in range(m):
        dfs_highland(grid, (0, c), pacific_highland)
        dfs_highland(grid, (n - 1, c), atlantic_highland)

    res = [[r, c] for r in range(n) for c in range(m)
           if pacific_highland[r][c] and atlantic_highland[r][c]]
    return res


def dfs_highland(grid, start, visited):
    n, m = len(grid), len(grid[0])
    stack = [start]
    while stack:
        r, c = stack.pop()
        if visited[r][c]:
            continue
        visited[r][c] = True
        for dr, dc in DIRECTIONS:
            new_r, new_c = r + dr, c + dc
            if (0 <= new_r < n and 0 <= new_c < m and not visited[new_r][new_c]
                    and grid[r][c] <= grid[new_r][new_c]):
                stack.append((new_r, new_c))


class OceanChannelSpec(unittest.TestCase):
    def test_example(self):
        grid = [
            [1, 2, 2, 3, 5],
            [3, 2, 3, 4, 4],
            [2, 4, 5, 3, 1],
            [6, 7, 1, 4, 5],
            [5, 1, 1, 2, 4]]
        expected = [[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]]
        self.assertCountEqual(expected, ocean_channel(grid))

    def test_example2(self):
        grid = [
            [2, 1],
            [1, 2]]
        expected = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.assertCountEqual(expected, ocean_channel(grid))

    def test_empty_grid(self):
        self.assertEqual([], ocean_channel([]))
        self.assertEqual([], ocean_channel([[]]))

    def test_one_cell_grid(self):
        grid = [[42]]
        expected = [[0, 0]]
        self.assertCountEqual(expected, ocean_channel(grid))
        self.assertEqual([], ocean_channel([[]]))

    def test_one_row_grid(self):
        grid = [[1, 2, 3, 4]]
        expected = [[0, 0], [0, 1], [0, 2], [0, 3]]
        self.assertCountEqual(expected, ocean_channel(grid))
        self.assertEqual([], ocean_channel([[]]))

    def test_one_column_grid(self):
        grid = [
            [3],
            [2],
            [1]]
        expected = [[0, 0], [1, 0], [2, 0]]
        self.assertCountEqual(expected, ocean_channel(grid))

    def test_bottom_right_is_high(self):
        grid = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4]
        ]
        expected = [[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]]
        self.assertCountEqual(expected, ocean_channel(grid))

    def test_top_left_is_high(self):
        grid = [
            [4, 3, 2],
            [3, 2, 1],
            [2, 1, 0]
        ]
        expected = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]
        self.assertCountEqual(expected, ocean_channel(grid))
        self.assertCountEqual(expected, ocean_channel(grid))

    def test_top_right_is_high(self):
        grid = [
            [2, 3, 4],
            [1, 2, 3],
            [0, 1, 2]
        ]
        expected = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        self.assertCountEqual(expected, ocean_channel(grid))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### May 30, 2021 \[Medium\] H-Index II
---
> **Question:** The h-index is a metric that attempts to measure the productivity and citation impact of the publication of a scholar. The definition of the h-index is if a scholar has at least h of their papers cited h times.
>
> Given a list of publications of the number of citations a scholar has, find their h-index.
>
> Follow up for H-Index: What if the citations array is sorted in ascending order? Could you optimize your algorithm?

**Example:**
```py
Input: [0, 1, 3, 3, 5]
Output: 3
Explanation:
There are 3 publications with 3 or more citations, hence the h-index is 3.
```

**Solution with Binary Search:** [https://replit.com/@trsong/H-Index-II](https://replit.com/@trsong/H-Index-II)
```py
import unittest

def calculate_h_index(citations):
    n = len(citations)
    lo, hi = 0, n - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if citations[mid] < n - mid:
            lo = mid + 1
        else:
            hi = mid
    return n - lo


class CalculateHIndexSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(3, calculate_h_index([0, 1, 3, 3, 5]))

    def test_another_citation_array(self):
        self.assertEqual(3, calculate_h_index([0, 1, 3, 5, 6]))

    def test_empty_citations(self):
        self.assertEqual(0, calculate_h_index([]))

    def test_only_one_publications(self):
        self.assertEqual(1, calculate_h_index([42]))
    
    def test_h_index_appear_once(self):
        self.assertEqual(5, calculate_h_index([5, 10, 10, 10, 10]))

    def test_balanced_citation_counts(self):
        self.assertEqual(5, calculate_h_index([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    def test_duplicated_citations(self):
        self.assertEqual(3, calculate_h_index([2, 2, 2, 2, 2, 3, 3, 3]))
    
    def test_zero_citations_not_count(self):
        self.assertEqual(2, calculate_h_index([0, 0, 0, 0, 10, 10]))
    
    def test_citations_number_greater_than_publications(self):
        self.assertEqual(4, calculate_h_index([6, 7, 8, 9]))

    def test_citations_number_greater_than_publications2(self):
        self.assertEqual(3, calculate_h_index([1, 4, 7, 9]))   


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### May 29, 2021 \[Medium\] Similar Websites
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

**Notes:** duplicate entry might occur, we have to treat normal set to multiset: treat `3, 3, 3` as `3(first), 3(second), 3(third)`. 
```py
a: 1 2 3(first) 3(second) 3(third)
b: 1 2 3(first)
The similarty between (a, b) is 3/5
```

**Solution:** [https://replit.com/@trsong/Find-Similar-Websites-2](https://replit.com/@trsong/Find-Similar-Websites-2)
```py
import unittest

def top_similar_websites(website_log, k):
    site_hits = {}
    user_site_hits = {}
    for site, user in website_log:
        site_hits[site] = site_hits.get(site, 0) + 1
        user_site_hits[user] = user_site_hits.get(user, {})
        user_site_hits[user][site] = user_site_hits[user].get(site, 0) + 1

    cross_site_hits = {}
    for hits_per_site in user_site_hits.values():
        sites = sorted(hits_per_site.keys())
        for i in range(len(sites)):
            site1 = sites[i]
            for j in range(i):
                site2 = sites[j]
                cross_site_hits[(site1, site2)] = (
                    cross_site_hits.get((site1, site2), 0) + 
                    min(hits_per_site[site1], hits_per_site[site2]))

    site_pairs = cross_site_hits.keys()
    site_pairs.sort(key=lambda pair: calculate_similarity(
        pair[0], pair[1], site_hits, cross_site_hits),
        reverse=True)
    if len(site_pairs) >= k:
        return site_pairs[:k]
    else:
        return site_pairs + padding_dissimilar_sites(
            site_hits, cross_site_hits)[k-len(site):]


def calculate_similarity(site1, site2, site_hits, cross_site_hits):
    total = site_hits[site1] + site_hits[site2]
    intersection = cross_site_hits.get((site1, site2), 0)
    union = total - intersection
    return float(intersection) / union


def padding_dissimilar_sites(site_hits, cross_site_hits):
    res = []
    sites = sorted(site_hits.keys())
    for i in range(len(sites)):
        site1 = sites[i]
        for j in range(i):
            site2 = sites[j]
            if (site1, site2) not in cross_site_hits:
                res.append((site1, site2))
    return res


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
    unittest.main(exit=False, verbosity=2)
```


### May 28, 2021 \[Medium\] Detect Linked List Cycle
---
> **Question:** Given a linked list, determine if the linked list has a cycle in it. 

**Example:**
```py
Input: 4 -> 3 -> 2 -> 1 -> 3 ... 
Output: True
```

**Solution with Fast-Slow Pointers:** [https://replit.com/@trsong/Detect-Linked-List-Cycle-2](https://replit.com/@trsong/Detect-Linked-List-Cycle-2)
```py
import unittest

def contains_cycle(lst):
    fast = slow = lst
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if fast == slow:
            return True
    return False


class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    @staticmethod
    def List(intersect_index, vals):
        dummy = p = ListNode(-1)
        intersect_node = None
        for val in vals:
            p.next = ListNode(val)
            p = p.next
            if intersect_index == 0:
                intersect_node = p
            intersect_index -= 1
        p.next = intersect_node
        return dummy.next


class ContainsCycleSpec(unittest.TestCase):
    def test_example(self):
        lst = ListNode.List(intersect_index = 1, vals=[4, 3, 2, 1])
        self.assertTrue(contains_cycle(lst))

    def test_empty_list(self):
        self.assertFalse(contains_cycle(None))

    def test_list_with_self_pointing_node(self):
        lst = ListNode.List(intersect_index = 0, vals=[1])
        self.assertTrue(contains_cycle(lst))

    def test_acyclic_list_with_duplicates(self):
        lst = ListNode.List(intersect_index = -1, vals=[1, 1, 1, 1, 1, 1])
        self.assertFalse(contains_cycle(lst))

    def test_acyclic_list_with_duplicates2(self):
        lst = ListNode.List(intersect_index = -1, vals=[1, 2, 3, 1, 2, 3])
        self.assertFalse(contains_cycle(lst))

    def test_cyclic_list_with_duplicates(self):
        lst = ListNode.List(intersect_index = 0, vals=[1, 2, 3, 1, 2, 3])
        self.assertTrue(contains_cycle(lst))

    def test_cyclic_list(self):
        lst = ListNode.List(intersect_index = 6, vals=[0, 1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(contains_cycle(lst))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 27, 2021 \[Medium\] Largest Rectangular Area in a Histogram
---
> **Question:** You are given a histogram consisting of rectangles of different heights. Determine the area of the largest rectangle that can be formed only from the bars of the histogram.

**Example:**
```py
These heights are represented in an input list, such that [1, 3, 2, 5] corresponds to the following diagram:

      x
      x  
  x   x
  x x x
x x x x

For the diagram above, for example, this would be six, representing the 2 x 3 area at the bottom right.
```

**Solution with Stack:** [https://replit.com/@trsong/Find-Largest-Rectangular-Area-in-a-Histogram-2](https://replit.com/@trsong/Find-Largest-Rectangular-Area-in-a-Histogram-2)
```py
import unittest

def largest_rectangle_in_histogram(histogram):
    n = len(histogram)
    stack = []
    res = 0
    i = 0

    while i < n or stack:
        if not stack or i < n and histogram[i] > histogram[stack[-1]]:
            # mantain stack in non-descending order
            stack.append(i)
            i += 1
        else:
            # if stack starts decreasing,
            # then left boundary must be stack[-2] and right boundary must be i. Note both boundaries are exclusive
            # and height is stack[-1]
            height = histogram[stack.pop()]
            left = stack[-1] if stack else -1
            width = i - left - 1
            res = max(res, width * height)
    return res


class LargestRectangleInHistogramSpec(unittest.TestCase):
    def test_example(self):
        """
              x
              x  
          x   x
          X X X
        x X X X
        """
        histogram = [1, 3, 2, 5]
        expected = 2 * 3
        self.assertEqual(expected, largest_rectangle_in_histogram(histogram))

    def test_empty_histogram(self):
        self.assertEqual(0, largest_rectangle_in_histogram([]))

    def test_width_one_rectangle(self):
        """
              X
              X
              X
              X 
          x   X
          x x X
        x x x X
        """
        histogram = [1, 3, 2, 7]
        expected = 7 * 1
        self.assertEqual(expected, largest_rectangle_in_histogram(histogram))

    def test_ascending_sequence(self):
        """
                x  
              x x
            X X X
          x X X X
        """
        histogram = [0, 1, 2, 3, 4]
        expected = 2 * 3
        self.assertEqual(expected, largest_rectangle_in_histogram(histogram))

    def test_descending_sequence(self):
        """
        x  
        X X
        X X x    
        X X x x
        """
        histogram = [4, 3, 2, 1, 0]
        expected = 3 * 2
        self.assertEqual(expected, largest_rectangle_in_histogram(histogram))

    def test_sequence3(self):
        """
            x
          x x x 
        X X X X X 
        X X X X X
        X X X X X
        """       
        histogram = [3, 4, 5, 4, 3]
        expected = 3 * 5
        self.assertEqual(expected, largest_rectangle_in_histogram(histogram))

    def test_sequence4(self):
        """
          x   x   x   x
          x   x   x   x
          x   x   x   x
          x   x   x   x
          x   x x x   x
          x   x x x   x
          X X X X X X X 
        x X X X X X X X x
        x X X X X X X X x
        x X X X X X X X x
        """      
        histogram = [3, 10, 4, 10, 5, 10, 4, 10, 3]
        expected = 4 * 7
        self.assertEqual(expected, largest_rectangle_in_histogram(histogram))

    def test_sequence5(self):
        """
        x           x
        x   x   x   x
        x   X X X   x
        x   X X X   x
        x x X X X   x
        x x X X X x x 
        """
        histogram = [6, 2, 5, 4, 5, 1, 6]
        expected = 4 * 3
        self.assertEqual(expected, largest_rectangle_in_histogram(histogram))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 26, 2021 \[Easy\] Remove k-th Last Element From Linked List
---
> **Question:** You are given a singly linked list and an integer k. Return the linked list, removing the k-th last element from the list. 

**My thoughts:** Use two pointers, faster one are k position ahead of slower one. When fast one hit last element, the slow one will become the one before the kth last element.

**Solution with Fast-Slow Pointers:** [https://replit.com/@trsong/Remove-the-k-th-Last-Element-From-LinkedList](https://replit.com/@trsong/Remove-the-k-th-Last-Element-From-LinkedList)
```py
import unittest

def remove_last_kth_elem(k, lst):
    if not lst:
        return None
    
    dummy = fast = slow = ListNode(-1, lst)
    for _ in range(k):
        fast = fast.next
        if not fast:
            return lst

    while fast.next:
        slow = slow.next
        fast = fast.next
    
    slow.next = slow.next.next
    return dummy.next
    

###################
# Testing Utilities
###################
class ListNode(object):
    def __init__(self, x, next=None):
        self.val = x
        self.next = next

    def __eq__(self, other):
        return other and self.val == other.val and self.next == other.next

    def __repr__(self):
        return "{} -> {}".format(str(self.val), str(self.next))

    @staticmethod
    def List(*vals):
        p = dummy = ListNode(-1)
        for elem in vals:
            p.next = ListNode(elem)
            p = p.next
        return dummy.next


class RemoveLastKthElementSpec(unittest.TestCase):
    def test_empty_list(self):
        self.assertIsNone(remove_last_kth_elem(0, None))

    def test_remove_the_only_element(self):
        k, lst = 1, ListNode.List(42)
        self.assertIsNone(remove_last_kth_elem(k, lst))
    
    def test_remove_the_last_element(self):
        k, lst = 1, ListNode.List(1, 2, 3)
        expected = ListNode.List(1, 2)
        self.assertEqual(expected, remove_last_kth_elem(k, lst))

    def test_remove_the_first_element(self):
        k, lst = 4, ListNode.List(1, 2, 3, 4)
        expected = ListNode.List(2, 3, 4)
        self.assertEqual(expected, remove_last_kth_elem(k, lst))

    def test_remove_element_in_the_middle(self):
        k, lst = 3, ListNode.List(5, 4, 3, 2, 1)
        expected = ListNode.List(5, 4, 2, 1)
        self.assertEqual(expected, remove_last_kth_elem(k, lst))
    
    def test_k_beyond_the_range(self):
        k, lst = 10, ListNode.List(3, 2, 1)
        expected = ListNode.List(3, 2, 1)
        self.assertEqual(expected, remove_last_kth_elem(k, lst))
    
    def test_k_beyond_the_range2(self):
        k, lst = 4, ListNode.List(3, 2, 1)
        expected = ListNode.List(3, 2, 1)
        self.assertEqual(expected, remove_last_kth_elem(k, lst))

    def test_remove_the_second_last_element(self):
        k, lst = 2, ListNode.List(4, 3, 2, 1)
        expected = ListNode.List(4, 3, 1)
        self.assertEqual(expected, remove_last_kth_elem(k, lst))

    
if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 25, 2021 \[Easy\] Valid Mountain Array
---
> **Question:** Given an array of heights, determine whether the array forms a "mountain" pattern. A mountain pattern goes up and then down.

**Example:**
```py
validMountainArray([1, 2, 3, 2, 1])  # True
validMountainArray([1, 2, 3])  # False
```

**Solution:** [https://replit.com/@trsong/Valid-Mountain-Array-2](https://replit.com/@trsong/Valid-Mountain-Array-2)
```py
import unittest

def is_valid_mountain(arr):
    if len(arr) < 3:
        return False
    
    sign = 1
    flip = 0
    for i in range(1, len(arr)):
        delta = arr[i] - arr[i - 1]
        if sign * delta < 0:
            flip += 1
            sign *= -1
        if flip > 1 or delta == 0:
            return False
    return flip == 1


class IsValidMountainSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(is_valid_mountain([1, 2, 3, 2, 1]))

    def test_example2(self):
        self.assertFalse(is_valid_mountain([1, 2, 3]))

    def test_empty_array(self):
        self.assertFalse(is_valid_mountain([]))

    def test_one_element_array(self):
        self.assertFalse(is_valid_mountain([1]))

    def test_two_elements_array(self):
        self.assertFalse(is_valid_mountain([1, 2]))

    def test_three_elements_array(self):
        self.assertFalse(is_valid_mountain([1, 2, 3]))

    def test_three_elements_array2(self):
        self.assertTrue(is_valid_mountain([1, 2, 1]))

    def test_duplicted_elements(self):
        self.assertFalse(is_valid_mountain([1, 2, 2]))
    
    def test_duplicted_element2(self):
        self.assertFalse(is_valid_mountain([1, 1, 2]))

    def test_duplicted_elements2(self):
        self.assertFalse(is_valid_mountain([0, 0, 0]))

    def test_mutiple_mountains(self):
        self.assertFalse(is_valid_mountain([1, 2, 1, 2]))

    def test_mutiple_mountains2(self):
        self.assertFalse(is_valid_mountain([1, 2, 1, 2, 1]))

    def test_concave_array(self):
        self.assertFalse(is_valid_mountain([0, -1, 1]))

    def test_no_ascending(self):
        self.assertFalse(is_valid_mountain([0, 0, -1]))

    def test_no_ascending2(self):
        self.assertFalse(is_valid_mountain([0, -1, -1]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 24, 2021 \[Medium\] Satisfactory Playlist
---
> **Question:** You have access to ranked lists of songs for various users. Each song is represented as an integer, and more preferred songs appear earlier in each list. For example, the list `[4, 1, 7]` indicates that a user likes song `4` the best, followed by songs `1` and `7`.
>
> Given a set of these ranked lists, interleave them to create a playlist that satisfies everyone's priorities.
>
> For example, suppose your input is `[[1, 7, 3], [2, 1, 6, 7, 9], [3, 9, 5]]`. In this case a satisfactory playlist could be `[2, 1, 6, 7, 3, 9, 5]`.

**My thoughts:** create a graph with vertex being song and edge `(u, v)` representing that u is more preferred than v. A topological order will make sure that all more preferred song will go before less preferred ones. Thus gives a list that satisfies everyone's priorities, if there is one (no cycle).

**Solution with Topological Sort:** [https://replit.com/@trsong/Satisfactory-Playlist-for-Everyone-2](https://replit.com/@trsong/Satisfactory-Playlist-for-Everyone-2)
```py
import unittest

def calculate_satisfactory_playlist(preference):
    song_set = set([song for songs in preference for song in songs])
    neighbors = {}
    inbound_edges = {}
    
    for song_lst in preference:
        for i in range(1, len(song_lst)):
            prev, cur = song_lst[i-1], song_lst[i]
            neighbors[prev] = neighbors.get(prev, set())
            if cur not in neighbors[prev]:
                neighbors[prev].add(cur)
                inbound_edges[cur] = inbound_edges.get(cur, 0) + 1

    queue = filter(lambda song: inbound_edges.get(song, 0) == 0, song_set)

    top_order = []
    while queue:
        cur = queue.pop(0)
        top_order.append(cur)

        if cur not in neighbors:
            continue

        for neighbor in neighbors[cur]:
            inbound_edges[neighbor] -= 1
            if inbound_edges[neighbor] == 0:
                del inbound_edges[neighbor]
                queue.append(neighbor)
    return top_order if len(top_order) == len(song_set) else None


class CalculateSatisfactoryPlaylistSpec(unittest.TestCase):
    def validate_result(self, preference, suggested_order):
        song_set = set([song for songs in preference for song in songs])
        self.assertEqual(
            song_set,
            set(suggested_order),
            "Missing song: " + str(str(song_set - set(suggested_order))))

        for i in xrange(len(suggested_order)):
            for j in xrange(i+1, len(suggested_order)):
                for lst in preference:
                    song1, song2 = suggested_order[i], suggested_order[j]
                    if song1 in lst and song2 in lst:
                        self.assertLess(
                            lst.index(song1), 
                            lst.index(song2),
                            "Suggested order {} conflict: {} cannot be more popular than {}".format(suggested_order, song1, song2))

    def test_example(self):
        preference = [[1, 7, 3], [2, 1, 6, 7, 9], [3, 9, 5]]
        # possible order: 2, 1, 6, 7, 3, 9, 5
        suggested_order = calculate_satisfactory_playlist(preference) 
        self.validate_result(preference, suggested_order)
    
    def test_preference_contains_duplicate(self):
        preference = [[1, 2], [1, 2], [1, 2]]
        # possible order: 1, 2
        suggested_order = calculate_satisfactory_playlist(preference) 
        self.validate_result(preference, suggested_order)

    def test_empty_graph(self):
        self.assertEqual([], calculate_satisfactory_playlist([]))

    def test_cyclic_graph(self):
        preference = [[1, 2, 3], [1, 3, 2]]
        self.assertIsNone(calculate_satisfactory_playlist(preference))

    def test_acyclic_graph(self):
        preference = [[1, 2], [2, 3], [1, 3, 5], [2, 5], [2, 4]]
        # possible order: 1, 2, 3, 4, 5
        suggested_order = calculate_satisfactory_playlist(preference)
        self.validate_result(preference, suggested_order)

    def test_disconnected_graph(self):
        preference = [[0, 1], [2, 3], [3, 4]]
        # possible order: 0, 1, 2, 3, 4
        suggested_order = calculate_satisfactory_playlist(preference)
        self.validate_result(preference, suggested_order)

    def test_disconnected_graph2(self):
        preference = [[0, 1], [2], [3]]
        # possible order: 0, 1, 2, 3
        suggested_order = calculate_satisfactory_playlist(preference)
        self.validate_result(preference, suggested_order)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 23, 2021 LC 438 \[Medium\] Anagrams in a String
---
> **Question:** Given 2 strings s and t, find and return all indexes in string s where t is an anagram.

**Example:** 
```py
find_anagrams('acdbacdacb', 'abc')  # gives [3, 7], anagrams: bac, acb
```

**Solution:** [https://replit.com/@trsong/Find-All-Anagram-Indices-2](https://replit.com/@trsong/Find-All-Anagram-Indices-2)
```py
import unittest

def find_anagrams(word, s):
    if not s:
        return []
        
    histogram = {}
    for ch in s:
        histogram[ch] = histogram.get(ch, 0) + 1
    
    res = []
    for end, incoming_char in enumerate(word):
        histogram[incoming_char] = histogram.get(incoming_char, 0) - 1
        if histogram[incoming_char] == 0:
            del histogram[incoming_char]

        start = end - len(s)
        if start >= 0:
            outgoing_char = word[start]
            histogram[outgoing_char] = histogram.get(outgoing_char, 0) + 1
            if histogram[outgoing_char] == 0:
                del histogram[outgoing_char]

        if not histogram:
            res.append(start + 1)
    return res


class FindAnagramSpec(unittest.TestCase):
    def test_example(self):
        word = 'abxaba'
        s = 'ab'
        self.assertEqual([0, 3, 4], find_anagrams(word, s))

    def test_example2(self):
        word = 'acdbacdacb'
        s = 'abc'
        self.assertEqual([3, 7], find_anagrams(word, s))

    def test_empty_source(self):
        self.assertEqual([], find_anagrams('', 'a'))
    
    def test_empty_pattern(self):
        self.assertEqual([], find_anagrams('a', ''))

    def test_pattern_contains_unseen_characters_in_source(self):
        word = "abcdef"
        s = "123"
        self.assertEqual([], find_anagrams(word, s))
    
    def test_pattern_not_in_source(self):
        word = 'ab9cd9abc9d'
        s = 'abcd'
        self.assertEqual([], find_anagrams(word, s))
    
    def test_matching_strings_have_overlapping_positions_in_source(self):
        word = 'abab'
        s = 'ab'
        self.assertEqual([0, 1, 2], find_anagrams(word, s))
    
    def test_find_all_matching_positions(self):
        word = 'cbaebabacd'
        s = 'abc'
        self.assertEqual([0, 6], find_anagrams(word, s))
    
    def test_find_all_matching_positions2(self):
        word = 'BACDGABCDA'
        s = 'ABCD'
        self.assertEqual([0, 5, 6], find_anagrams(word, s))
    
    def test_find_all_matching_positions3(self):
        word = 'AAABABAA'
        s = 'AABA'
        self.assertEqual([0, 1, 4], find_anagrams(word, s))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 22, 2021 LC 166 \[Medium\] Fraction to Recurring Decimal
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

**Solution:** [https://replit.com/@trsong/Convert-Fraction-to-Recurring-Decimal-2](https://replit.com/@trsong/Convert-Fraction-to-Recurring-Decimal-2)
```py
import unittest

def fraction_to_decimal(numerator, denominator):
    if numerator == 0:
        return '0'

    res = []
    is_negative = (numerator > 0) ^ (denominator > 0)
    if is_negative:
        res.append('-')

    numerator = abs(numerator)
    denominator = abs(denominator)
    res.append(str(numerator // denominator))

    numerator %= denominator
    if numerator > 0:
        res.append('.')

    repeat_location = {}
    while numerator:
        if numerator in repeat_location:
            pos = repeat_location[numerator]
            res = res[:pos] + ['('] + res[pos:] + [')']
            break
        repeat_location[numerator] = len(res)
        numerator *= 10
        res.append(str(numerator // denominator))
        numerator %= denominator
    return ''.join(res)


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
    unittest.main(exit=False, verbosity=2)
```


### May 21, 2021 LC 421 \[Medium\] Maximum XOR of Two Numbers in an Array
---
> **Question:** Given an array of integers, find the maximum XOR of any two elements.

**Example:**
```py
Input: nums = [3, 10, 5, 25, 2, 8]
Output: 28
Explanation: The maximum result is 5 XOR 25 = 28.
```

**My thoughts:** The idea is to efficiently get the other number for each number such that xor is max. We can parellel this process with Trie: insert all number into trie, and for each number greedily choose the largest number that has different digit. 

**Solution with Trie:** [https://replit.com/@trsong/Maximum-XOR-of-Two-Numbers-in-an-Array-2](https://replit.com/@trsong/Maximum-XOR-of-Two-Numbers-in-an-Array-2)
```py
import unittest

def find_max_xor(nums):
    trie = Trie()
    for num in nums:
        trie.insert(num)
    return max(map(trie.max_xor_value, nums))


class Trie(object):
    def __init__(self):
        self.children = [None, None]

    def insert(self, number):
        p = self
        for i in xrange(32, -1, -1):
            bit = 1 & (number >> i)
            p.children[bit] = p.children[bit] or Trie()
            p = p.children[bit]
        
    def max_xor_value(self, number):
        p = self
        accu = 0
        for i in xrange(32, -1, -1):
            bit = 1 & (number >> i)
            if p.children[1 - bit]:
                accu ^= 1 << i
                p = p.children[1 - bit]
            else:
                p = p.children[bit]
        return accu
            

class FindMaxXORSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(bin(expected), bin(result))

    def test_example(self):
        nums = [  0b11,
                0b1010,
                 0b101,
               0b11001,
                  0b10,
                0b1000]
        expected = 0b101 ^ 0b11001
        self.assert_result(expected, find_max_xor(nums))

    def test_only_one_element(self):
        nums = [0b11]
        expected = 0b11 ^ 0b11
        self.assert_result(expected, find_max_xor(nums))

    def test_two_elements(self):
        nums = [0b10, 0b100]
        expected = 0b10 ^ 0b100
        self.assert_result(expected, find_max_xor(nums))

    def test_example2(self):
        nums = [ 0b1110,
              0b1000110,
               0b110101,
              0b1010011,
               0b110001,
              0b1011011,
               0b100100,
              0b1010000,
              0b1011100,
               0b110011,
              0b1000010,
              0b1000110]
        expected = 0b1011011 ^ 0b100100
        self.assert_result(expected, find_max_xor(nums))

    def test_return_max_number_of_set_bit(self):
        nums = [ 0b111,
                  0b11,
                   0b1,
                     0]
        expected = 0b111 ^ 0
        self.assert_result(expected, find_max_xor(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### May 20, 2021 \[Hard\] Inversion Pairs
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

**Solution with Merge Sort:** [https://replit.com/@trsong/Count-Inversion-Pairs-2](https://replit.com/@trsong/Count-Inversion-Pairs-2)
```py
import unittest

def count_inversion_pairs(nums):
    count, _ = merge_sort_and_count(nums)
    return count


def merge_sort_and_count(nums):
    n = len(nums)
    if n <= 1:
        return 0, nums

    count1, lst1 = merge_sort_and_count(nums[:n//2])
    count2, lst2 = merge_sort_and_count(nums[n//2:])
    combined_count, lst = merge(lst1, lst2)
    return combined_count + count1 + count2, lst


def merge(lst1, lst2):
    n, m = len(lst1), len(lst2)
    i = j = 0
    res = []
    count = 0

    while i < n and j < m:
        if lst1[i] <= lst2[j]:
            res.append(lst1[i])
            i += 1
        else:
            res.append(lst2[j])
            j += 1
            count += n - i

    if i < n:
        res.append(lst1[i])
        i += 1
    
    if j < m:
        res.append(lst2[j])
        j += 1
    
    return count, res


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
    unittest.main(exit=False, verbosity=2)
```

### May 19, 2021 \[Easy\] Add Two Numbers as a Linked List
---
> **Question:** You are given two linked-lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

**Example:**
```py
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```

**Solution:** [https://replit.com/@trsong/Add-Two-Numbers-and-Return-as-a-Linked-List-2](https://replit.com/@trsong/Add-Two-Numbers-and-Return-as-a-Linked-List-2)
```py
import unittest

def lists_addition(l1, l2):
    dummy = p = ListNode(-1)
    carry = 0
    while l1 or l2 or carry:
        if l1:
            carry += l1.val
            l1 = l1.next
        if l2:
            carry += l2.val
            l2 = l2.next
        p.next = ListNode(carry % 10)
        p = p.next
        carry //= 10
    return dummy.next


###################
# Testing Utilities
###################
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return "{} -> {}".format(str(self.val), str(self.next))

    def __eq__(self, other):
        return other and self.val == other.val and self.next == other.next

    @staticmethod
    def build_list(*nums):
        node = dummy = ListNode(-1)
        for num in nums:
            node.next = ListNode(num)
            node = node.next
        return dummy.next


class ListsAdditionSpec(unittest.TestCase):
    def test_example(self):
        l1 = ListNode.build_list(2, 4, 3)
        l2 = ListNode.build_list(5, 6, 4)
        expected = ListNode.build_list(7, 0, 8)
        self.assertEqual(expected, lists_addition(l1, l2))

    def test_add_empty_list(self):
        self.assertEqual(None, lists_addition(None, None))

    def test_add_nonempty_to_empty_list(self):
        l1 = None
        l2 = ListNode.build_list(1, 2, 3)
        expected = ListNode.build_list(1, 2, 3)
        self.assertEqual(expected, lists_addition(l1, l2))

    def test_add_empty_to_nonempty_list(self):
        l1 = ListNode.build_list(1)
        l2 = None
        expected = ListNode.build_list(1)
        self.assertEqual(expected, lists_addition(l1, l2))

    def test_addition_with_carryover(self):
        l1 = ListNode.build_list(1, 1)
        l2 = ListNode.build_list(9, 9, 9, 9)
        expected = ListNode.build_list(0, 1, 0, 0, 1)
        self.assertEqual(expected, lists_addition(l1, l2))

    def test_addition_with_carryover2(self):
        l1 = ListNode.build_list(7, 5, 9, 4, 6)
        l2 = ListNode.build_list(8, 4)
        expected = ListNode.build_list(5, 0, 0, 5, 6)
        self.assertEqual(expected, lists_addition(l1, l2))

    def test_add_zero_to_number(self):
        l1 = ListNode.build_list(4, 2)
        l2 = ListNode.build_list(0)
        expected = ListNode.build_list(4, 2)
        self.assertEqual(expected, lists_addition(l1, l2))
    
    def test_same_length_lists(self):
        l1 = ListNode.build_list(1, 2, 3)
        l2 = ListNode.build_list(9, 8, 7)
        expected = ListNode.build_list(0, 1, 1, 1)
        self.assertEqual(expected, lists_addition(l1, l2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### May 18, 2021 \[Medium\] Boggle Game
--- 
> **Question:** Given a dictionary, a method to do a lookup in the dictionary and a M x N board where every cell has one character. Find all possible words that can be formed by a sequence of adjacent characters. Note that we can move to any of 8 adjacent characters, but a word should not have multiple instances of the same cell.

**Example 1:**
```py
Input: dictionary = ["GEEKS", "FOR", "QUIZ", "GO"]
       boggle = [['G', 'I', 'Z'],
                 ['U', 'E', 'K'],
                 ['Q', 'S', 'E']]
Output: Following words of the dictionary are present
         GEEKS
         QUIZ
```

**Example 2:**
```py
Input: dictionary = ["GEEKS", "ABCFIHGDE"]
       boggle = [['A', 'B', 'C'],
                 ['D', 'E', 'F'],
                 ['G', 'H', 'I']]
Output: Following words of the dictionary are present
         ABCFIHGDE
```

**Solution with Trie and Backtracking:** [https://replit.com/@trsong/Boggle-Game](https://replit.com/@trsong/Boggle-Game)
```py
import unittest

def boggle_game(boggle, dictionary):
    if not boggle or not boggle[0]:
        return []

    trie = Trie()
    for word in dictionary:
        trie.insert(word)

    res = []
    for r in range(len(boggle)):
        for c in range(len(boggle[0])):
            backtrack(res, boggle, trie, r, c)
    return res


class Trie(object):
    def __init__(self):
        self.word = None
        self.children = None
    
    def insert(self, word):
        p = self
        for ch in word:
            p.children = p.children or {}
            p.children[ch] = p.children.get(ch, Trie())
            p = p.children[ch]
        p.word = word


DIRECTIONS = [-1, 0, 1]
def backtrack(res, boggle, parent_node, r, c):
    ch = boggle[r][c]
    if not ch or not parent_node.children or not parent_node.children.get(ch, None):
        return
    
    node = parent_node.children[ch]
    if node.word:
        res.append(node.word)
        node.word = None
    
    boggle[r][c] = None
    n, m = len(boggle), len(boggle[0])
    for dr in DIRECTIONS:
        for dc in DIRECTIONS:
            new_r, new_c = r + dr, c + dc
            if 0 <= new_r < n and 0 <= new_c < m and boggle[new_r][new_c]:
                backtrack(res, boggle, node, new_r, new_c)
    boggle[r][c] = ch


class BoggleGamedSpec(unittest.TestCase):
    def assert_result(self, expected, res):
        self.assertEqual(sorted(expected), sorted(res))

    def test_example(self):
        dictionary = ["GEEKS", "FOR", "QUIZ", "GO"]
        boggle = [['G', 'I', 'Z'],
                 ['U', 'E', 'K'],
                 ['Q', 'S', 'E']]
        expected = ["GEEKS", "QUIZ"]
        self.assert_result(expected, boggle_game(boggle, dictionary))

    def test_example2(self):
        dictionary = ["GEEKS", "ABCFIHGDE"]
        boggle = [['A', 'B', 'C'],
                 ['D', 'E', 'F'],
                 ['G', 'H', 'I']]
        expected = ['ABCFIHGDE']
        self.assert_result(expected, boggle_game(boggle, dictionary))

    def test_example3(self):
        dictionary = ['oath','pea','eat','rain']
        boggle = [
            ['o','a','a','n'],
            ['e','t','a','e'],
            ['i','h','k','r'],
            ['i','f','l','v']]
        expected = ['eat', 'oath']
        self.assert_result(expected, boggle_game(boggle, dictionary))

    def test_example4(self):
        dictionary = ['abcb']
        boggle = [
            ['a','b'],
            ['c','d']]
        expected = []
        self.assert_result(expected, boggle_game(boggle, dictionary))
    
    def test_example5(self):
        dictionary = ['DATA', 'HALO', 'HALT', 'SAG', 'BEAT', 'TOTAL', 'GLOT', 'DAG', 'DAGCD', 'DOG']
        boggle = [
            ['D', 'A', 'T', 'H'],
            ['C', 'G', 'O', 'A'],
            ['S', 'A', 'T', 'L'],
            ['B', 'E', 'E', 'G']]
        expected = ['DATA', 'HALO', 'HALT', 'SAG', 'BEAT', 'TOTAL', 'GLOT', 'DAG']
        self.assert_result(expected, boggle_game(boggle, dictionary))

    def test_unique_char(self):
        dictionary = ['a', 'aa', 'aaa']
        boggle = [
            ['a','a'],
            ['a','a']]
        expected = ['a', 'aa', 'aaa']
        self.assert_result(expected, boggle_game(boggle, dictionary))

    def test_empty_grid(self):
        self.assertEqual([], boggle_game([], ['a']))

    def test_empty_empty_word(self):
        self.assertEqual([], boggle_game(['a'], []))

    def test_word_use_all_letters(self):
        dictionary = ['abcdef']
        boggle = [
            ['a','b'],
            ['f','c'],
            ['e','d']]
        expected = ['abcdef']
        self.assert_result(expected, boggle_game(boggle, dictionary))

    
if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### May 17, 2021 \[Easy\] Phone Number to Words Based on The Dictionary
--- 
> **Question:** Given a phone number, return all valid words that can be created using that phone number.
>
> For instance, given the phone number `364` and dictionary `['dog', 'fish', 'cat', 'fog']`, we can construct the words `['dog', 'fog']`.


**Solution:** [https://replit.com/@trsong/Phone-Number-to-Words-Based-on-The-Dictionary](https://replit.com/@trsong/Phone-Number-to-Words-Based-on-The-Dictionary)
```py
import unittest
from functools import reduce

def phone_number_to_words(phone, dictionary):
    letter_map = {
        1: [],
        2: ['a', 'b', 'c'],
        3: ['d', 'e', 'f'],
        4: ['g', 'h', 'i'],
        5: ['j', 'k', 'l'],
        6: ['m', 'n', 'o'],
        7: ['p', 'q', 'r', 's'],
        8: ['t', 'u', 'v'],
        9: ['w', 'x', 'y', 'z'],
        0: []
    }
    letter_to_digit = {
        letter: digit
        for digit in letter_map for letter in letter_map[digit]
    }
    word_to_phone = lambda word: reduce(
        lambda accu, ch: 10 * accu + letter_to_digit[ch], word, 0)
    validate_word = lambda word: phone == word_to_phone(word)
    return list(filter(validate_word, dictionary))


class PhoneNumberToWordSpec(unittest.TestCase):
    def test_example(self):
        phone = 364
        dictionary = ['dog', 'fish', 'cat', 'fog']
        expected = ['dog', 'fog']
        self.assertCountEqual(expected,
                              phone_number_to_words(phone, dictionary))

    def test_empty_dictionary(self):
        phone = 42
        dictionary = []
        expected = []
        self.assertCountEqual(expected,
                              phone_number_to_words(phone, dictionary))

    def test_single_digit(self):
        phone = 5
        dictionary = ['a', 'b', 'cd', 'ef', 'g', 'k', 'j', 'kl']
        expected = ['j', 'k']
        self.assertCountEqual(expected,
                              phone_number_to_words(phone, dictionary))

    def test_contains_empty_word(self):
        phone = 222
        dictionary = [
            "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"
        ]
        expected = ['abc']
        self.assertCountEqual(expected,
                              phone_number_to_words(phone, dictionary))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### May 16, 2021 \[Easy\] Maximum Time
--- 
> **Question:** You are given a string that represents time in the format `hh:mm`. Some of the digits are blank (represented by ?). Fill in ? such that the time represented by this string is the maximum possible. Maximum time: `23:59`, minimum time: `00:00`. You can assume that input string is always valid.

**Example 1:**
```py
Input: "?4:5?"
Output: "14:59"
```

**Example 2:**
```py
Input: "23:5?"
Output: "23:59"
```

**Example 3:**
```py
Input: "2?:22"
Output: "23:22"
```

**Example 4:**
```py
Input: "0?:??"
Output: "09:59"
```

**Example 5:**
```py
Input: "??:??"
Output: "23:59"
```

**Solution:** [https://replit.com/@trsong/Maximum-Time](https://replit.com/@trsong/Maximum-Time)
```py
import unittest

def max_time(s):
    parts = s.split(':')
    return max_hour(parts[0]) + ':' + max_minute(parts[1])


def max_hour(s):
    d0, d1 = s[0], s[1]
    if d0 == '?':
        d0 = '2' if '0' <= d1 <= '3' or d1 == '?' else '1'
    if d1 == '?':
        d1 = '3' if d0 == '2' else '9'
    return d0 + d1


def max_minute(s):
    d0 = '5' if s[0] == '?' else s[0]
    d1 = '9' if s[1] == '?' else s[1]
    return d0 + d1


class MaxTimeSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual('14:59', max_time('?4:5?'))

    def test_example2(self):
        self.assertEqual('23:59', max_time('23:5?'))

    def test_example3(self):
        self.assertEqual('23:22', max_time('2?:22'))

    def test_example4(self):
        self.assertEqual('09:59', max_time('0?:??'))

    def test_example5(self):
        self.assertEqual('23:59', max_time('??:??'))

    def test_no_question_mark(self):
        self.assertEqual('00:00', max_time('00:00'))

    def test_hour(self):
        self.assertEqual('23:49', max_time('??:49'))

    def test_hour2(self):
        self.assertEqual('23:03', max_time('?3:03'))

    def test_hour3(self):
        self.assertEqual('17:19', max_time('?7:19'))

    def test_hour4(self):
        self.assertEqual('09:50', max_time('0?:50'))

    def test_minute(self):
        self.assertEqual('18:59', max_time('18:??'))

    def test_minute2(self):
        self.assertEqual('18:54', max_time('18:?4'))

    def test_minute3(self):
        self.assertEqual('18:39', max_time('18:3?'))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 15, 2021 LC 975 \[Hard\] Odd Even Jump
--- 
> **Question:** You are given an integer array `A`.  From some starting index, you can make a series of jumps.  The (1st, 3rd, 5th, ...) jumps in the series are called odd numbered jumps, and the (2nd, 4th, 6th, ...) jumps in the series are called even numbered jumps. 
> 
> You may from index `i` jump forward to index `j` (with `i < j`) in the following way:
>
> - During odd numbered jumps (ie. jumps 1, 3, 5, ...), you jump to the index `j` such that `A[i] <= A[j]` and `A[j]` is the smallest possible value.  If there are multiple such indexes `j`, you can only jump to the smallest such index `j`.
> - During even numbered jumps (ie. jumps 2, 4, 6, ...), you jump to the index `j` such that `A[i] >= A[j]` and `A[j]` is the largest possible value.  If there are multiple such indexes `j`, you can only jump to the smallest such index `j`.
> - (It may be the case that for some index `i`, there are no legal jumps.)
> 
> A starting index is good if, starting from that index, you can reach the end of the array (index `A.length - 1)` by jumping some number of times (possibly 0 or more than once.)
> 
> Return the number of good starting indexes.

**Example 1:**
```py
Input: [10,13,12,14,15]
Output: 2
Explanation: 
From starting index i = 0, we can jump to i = 2 (since A[2] is the smallest among A[1], A[2], A[3], A[4] that is greater or equal to A[0]), then we can't jump any more.
From starting index i = 1 and i = 2, we can jump to i = 3, then we can't jump any more.
From starting index i = 3, we can jump to i = 4, so we've reached the end.
From starting index i = 4, we've reached the end already.
In total, there are 2 different starting indexes (i = 3, i = 4) where we can reach the end with some number of jumps.
```

**Example 2:**
```py
Input: arr = [2,3,1,1,4]
Output: 3
Explanation: 
From starting index i = 0, we make jumps to i = 1, i = 2, i = 3:
During our 1st jump (odd-numbered), we first jump to i = 1 because arr[1] is the smallest value in [arr[1], arr[2], arr[3], arr[4]] that is greater than or equal to arr[0].
During our 2nd jump (even-numbered), we jump from i = 1 to i = 2 because arr[2] is the largest value in [arr[2], arr[3], arr[4]] that is less than or equal to arr[1]. arr[3] is also the largest value, but 2 is a smaller index, so we can only jump to i = 2 and not i = 3
During our 3rd jump (odd-numbered), we jump from i = 2 to i = 3 because arr[3] is the smallest value in [arr[3], arr[4]] that is greater than or equal to arr[2].
We can't jump from i = 3 to i = 4, so the starting index i = 0 is not good.
In a similar manner, we can deduce that:
From starting index i = 1, we jump to i = 4, so we reach the end.
From starting index i = 2, we jump to i = 3, and then we can't jump anymore.
From starting index i = 3, we jump to i = 4, so we reach the end.
From starting index i = 4, we are already at the end.
In total, there are 3 different starting indices i = 1, i = 3, and i = 4, where we can reach the end with some
number of jumps.
```

**Example 3:**
```py
Input: arr = [5,1,3,4,2]
Output: 3
Explanation: We can reach the end from starting indices 1, 2, and 4.
```



**My thoughts:** For each index i, two scenarios will happen:
- If it's odd turn, then whether start from i can reach the end depends on next greater index of i can reach the end in even turn
- If it's even turn, then whether start from i can reach the end depends on next smaller index of i can reach the end in odd turn

Based on above recursive definition, we can backtrack from the end and use dp to avoid calculate sub-problem over and over again. 

Yet, to make the algorithem efficient we need to pre-compute next greater and smaller index. This can be done with sorting and using stack.  

Finally, we just count how many indice can reach the end in odd turn. 


**Solution with DP, Stack and Sorting:** [https://replit.com/@trsong/Odd-Even-Jump](https://replit.com/@trsong/Odd-Even-Jump)
```py
import unittest

def count_odd_even_jumps(nums):
    if not nums:
        return 0

    n = len(nums)
    next_greater = [None] * n
    next_smaller = [None] * n
    smaller_stack = []
    for i in sorted(range(n), key=lambda i: (nums[i], i)):
        while smaller_stack and smaller_stack[-1] < i:
            next_greater[smaller_stack.pop()] = i
        smaller_stack.append(i)

    bigger_stack = []
    for j in sorted(range(n), key=lambda j: (-nums[j], j)):
        while bigger_stack and bigger_stack[-1] < j:
            next_smaller[bigger_stack.pop()] = j
        bigger_stack.append(j)

    # Let dp[i][odd_res], dp[i][even_res] represents whether start from i and take odd steps or even steps can reach the end
    odd_res, even_res = 1, 0
    dp = [[False, False] for _ in range(n)]
    dp[n - 1][odd_res] = True
    dp[n - 1][even_res] = True
    res = 1  # index n - 1 is always true
    for i in range(n - 2, -1, -1):
        if next_greater[i] is not None and dp[next_greater[i]][even_res]:
            dp[i][odd_res] = True
            res += 1

        if next_smaller[i] is not None and dp[next_smaller[i]][odd_res]:
            dp[i][even_res] = True
    return res


class CountOddEvenJumpSpec(unittest.TestCase):
    def test_example(self):
        nums = [10, 13, 12, 14, 15]
        # 14 -> 15
        # 15
        expected = 2
        self.assertEqual(expected, count_odd_even_jumps(nums))

    def test_example2(self):
        nums = [2, 3, 1, 1, 4]
        # 3 -> 4
        # 1 -> 4
        # 4
        expected = 3
        self.assertEqual(expected, count_odd_even_jumps(nums))

    def test_example3(self):
        nums = [5, 1, 3, 4, 2]
        # 1 -> 3 -> 2
        # 3 -> 4 -> 2
        # 2
        expected = 3
        self.assertEqual(expected, count_odd_even_jumps(nums))

    def test_example4(self):
        nums = [1, 2, 3, 2, 1, 4, 4, 5]
        # 2 -> 2 -> 1 -> 4 -> 4 -> 5
        # 3 -> 4 -> 4 -> 5
        # 2 -> 4 -> 4 -> 5
        # 1 -> 4 -> 4 -> 5
        # 4 -> 5
        # 5
        expected = 6
        self.assertEqual(expected, count_odd_even_jumps(nums))

    def test_increasing_array(self):
        nums = [1, 2, 3, 4, 5, 6, 7]
        # 6 -> 7
        # 7
        expected = 2
        self.assertEqual(expected, count_odd_even_jumps(nums))

    def test_decreasing_array(self):
        nums = [7, 6, 5, 4, 3, 2, 1]
        # 1
        expected = 1
        self.assertEqual(expected, count_odd_even_jumps(nums))

    def test_array_with_duplicated_values(self):
        nums = [1, 1, 1]
        # 1 -> 1 -> 1
        # 1 -> 1
        # 1
        expected = 3
        self.assertEqual(expected, count_odd_even_jumps(nums))

    def test_array_with_duplicated_values2(self):
        nums = [1, 1, 1, 2, 2, 2]
        # 1 -> 1 -> 2 -> 2 -> 2
        # 1 -> 2 -> 2 -> 2
        # 2 -> 2 -> 2
        # 2 -> 2
        # 2
        expected = 5
        self.assertEqual(expected, count_odd_even_jumps(nums))

    def test_array_with_duplicated_values3(self):
        nums = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        # 2 -> 2 -> 3 -> 3 -> 3
        # 2 -> 3 -> 3 -> 3
        # 3 -> 3 -> 3
        # 3 -> 3
        # 3
        expected = 5
        self.assertEqual(expected, count_odd_even_jumps(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 14, 2021 \[Easy\] Compare Version Numbers
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

**Solution:** [https://replit.com/@trsong/Compare-Two-Version-Numbers-2](https://replit.com/@trsong/Compare-Two-Version-Numbers-2)
```py
import unittest

def compare(v1, v2):
    stream1 = generate_stream(v1)
    stream2 = generate_stream(v2)

    for num1, num2 in zip(stream1, stream2):
        if num1 < num2:
            return -1
        elif num1 > num2:
            return 1
    
    for num1 in stream1:
        if num1 != 0:
            return 1

    for num2 in stream2:
        if num2 != 0:
            return -1

    return 0


def generate_stream(version):
    num = 0
    for ch in version:
        if ch == '.':
            yield num
            num = 0
        else:
            num = 10 * num + int(ch)
    yield num


class VersionNumberCompareSpec(unittest.TestCase):
    def test_example1(self):
        version1 = "1.0.33"
        version2 = "1.0.27"
        self.assertEqual(1, compare(version1, version2))

    def test_example2(self):
        version1 = "0.1"
        version2 = "1.1"
        self.assertEqual(-1, compare(version1, version2))

    def test_example3(self):
        version1 = "1.01"
        version2 = "1.001"
        self.assertEqual(0, compare(version1, version2))

    def test_example4(self):
        version1 = "1.0"
        version2 = "1.0.0"
        self.assertEqual(0, compare(version1, version2))

    def test_unspecified_version_numbers(self):
        self.assertEqual(0, compare("", ""))
        self.assertEqual(-1, compare("", "1"))
        self.assertEqual(1, compare("2", ""))

    def test_unaligned_zeros(self):
        version1 = "00000.00000.00000.0"
        version2 = "0.00000.000.00.00000.000.000.0"
        self.assertEqual(0, compare(version1, version2))

    def test_same_version_yet_unaligned(self):
        version1 = "00001.001"
        version2 = "1.000001.0000000.0000"
        self.assertEqual(0, compare(version1, version2))

    def test_different_version_numbers(self):
        version1 = "1.2.3.4"
        version2 = "1.2.3.4.5"
        self.assertEqual(-1, compare(version1, version2))

    def test_different_version_numbers2(self):
        version1 = "3.2.1"
        version2 = "3.1.2.3"
        self.assertEqual(1, compare(version1, version2))

    def test_different_version_numbers3(self):
        version1 = "00001.001.0.1"
        version2 = "1.000001.0000000.0000"
        self.assertEqual(1, compare(version1, version2))

    def test_without_dots(self):
        version1 = "32123"
        version2 = "3144444"
        self.assertEqual(-1, compare(version1, version2))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### May 13, 2021 \[Easy\] Rand7
---
> **Question:** Given a function `rand5()`, use that function to implement a function `rand7()` where `rand5()` returns an integer from `1` to `5` (inclusive) with uniform probability and `rand7()` is from `1` to `7` (inclusive). Also, use of any other library function and floating point arithmetic are not allowed.


**My thoughts:** You might ask how is it possible that a random number ranges from 1 to 5 can generate another random number ranges from 1 to 7? Well, think about how binary number works. For example, any number ranges from 0 to 3 can be represents in binary: 00, 01, 10 and 11. Each digit ranges from 0 to 1. Yet it can represents any number. Moreover, all digits are independent which means all number have the same probability to generate.

Just like the idea of a binary system, we can design a quinary (base-5) numeral system. And 2 digits is sufficient: `00, 01, 02, 03, 04, 10, 11, ..., 33, 34, 40, 41, 42, 43, 44.` (25 numbers in total) In decimal, "d1d0" base-5 equals `5 * d1 + d0` where d0, d1 ranges from 0 to 4. And entire "d1d0" ranges from 0 to 24. That should be sufficient to cover 1 to 7.

So whenever we get a random number in 1 to 7, we can simply return otherwise replay the same process over and over again until get a random number in 1 to 7.

> But, what if rand5 is expensive to call? Can we limit the call to rand5?

Yes, we can. We can just break the interval into the multiple of the modules. eg. `[0, 6]`, `[7, 13]` and `[14, 20]`. Once mod 7, all of them will be `[0, 6]`. And whenever we encounter 21 to 24, we simply discard it and replay the same algorithem mentioned above.


**Solution:** [https://replit.com/@trsong/Implement-Rand7-with-Rand5](https://replit.com/@trsong/Implement-Rand7-with-Rand5)
```py
from random import randint

def rand7():
    # d0, d2 range from 0 to 4 inclusive
    d0 = rand5() - 1
    d1 = rand5() - 1

    # base5 num ranges from 0 to 24 inclusive
    base5_num = 5 * d1 + d0

    if base5_num >= 21:
        # retry
        return rand7()
    
    # now base5 num can only have range 0 to 20 inclusive
    return base5_num % 7 + 1


####################
# Testing Utilities
####################
def rand5():
    return randint(1, 5)

def print_distribution(func, repeat):
    histogram = {}
    for _ in range(repeat):
        res = func()
        histogram[res] = histogram.get(res, 0) + 1
    print(histogram)


def main():
    # Distribution looks like {1: 10058, 2: 9977, 3: 10039, 4: 10011, 5: 9977, 6: 9998, 7: 9940}
    print_distribution(rand7, repeat=70000)


if __name__ == '__main__':
    main()
```


### May 12, 2021 \[Easy\] Three Equal Sums
---
> **Question:** Given an array of numbers, determine whether it can be partitioned into 3 arrays of equal sums.

**Example:**
```py
[0, 2, 1, -6, 6, -7, 9, 1, 2, 0, 1] can be partitioned into:
[0, 2, 1], [-6, 6, -7, 9, 1], [2, 0, 1] all of which sum to 3.
```

**Solution:** [https://replit.com/@trsong/Three-Equal-Sums](https://replit.com/@trsong/Three-Equal-Sums)
```py
import unittest

def has_three_equal_sums(nums):
    if len(nums) < 3:
        return False
    
    total = sum(nums)
    if total % 3 != 0:
        return False
    
    subtotal = total // 3
    count = 0
    accu = 0
    for num in nums:
        accu += num
        if accu == subtotal:
            accu = 0
            count += 1
        if count >= 3:
            return True
    
    return False


class HasThreeEqualSumSpec(unittest.TestCase):
    def test_example(self):
        nums = [0, 2, 1, -6, 6, -7, 9, 1, 2, 0, 1]
        # [0, 2, 1], [-6, 6, -7, 9, 1], [2, 0, 1]
        self.assertTrue(has_three_equal_sums(nums))

    def test_not_enough_elements(self):
        self.assertFalse(has_three_equal_sums([]))
        self.assertFalse(has_three_equal_sums([0]))
        self.assertFalse(has_three_equal_sums([0, 0]))
    
    def test_contains_zero(self):
        self.assertTrue(has_three_equal_sums([0, 0, 0, 0]))

    def test_total_sum_not_divisible_by_three(self):
        nums = [1, 1, 2, 1]
        self.assertFalse(has_three_equal_sums(nums))

    def test_total_sum_ok_yet_cannot_break(self):
        nums = [1, 1, 1, 3]
        self.assertFalse(has_three_equal_sums(nums))

    def test_qualified_array(self):
        nums = [1, 1, 1, 3, 3]
        self.assertTrue(has_three_equal_sums(nums))

    def test_qualified_array2(self):
        nums = [1, -1, 3, -3, 0, 0, 1, 0, -1, 2, -1, -1]
        # [1, -1], [3, -3], [0, 0, 1, 0, -1, 2, -1, -1]
        self.assertTrue(has_three_equal_sums(nums))

    def test_qualified_array3(self):
        nums = [10, 2, 2, 2, 2, 2, 9, 1]
        # [10], [2, 2, 2, 2, 2], [9, 1]
        self.assertTrue(has_three_equal_sums(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 11, 2021  LC 239 \[Medium\] Sliding Window Maximum
---
> **Question:** Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.
> 

**Example:**

```py
Input: nums = [1, 3, -1, -3, 5, 3, 6, 7], and k = 3
Output: [3, 3, 5, 5, 6, 7] 
```

**Explanation:**
```
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
 ```

**My thoughts:** The idea is to efficiently keep track of **INDEX** of 1st max, 2nd max, 3rd max and potentially k-th max elem. The reason for storing index is for the sake of avoiding index out of window. We can achieve that by using ***Double-Ended Queue*** which allow us to efficiently push and pop from both ends of the queue. 

The queue looks like `[index of 1st max, index of 2nd max, ...., index of k-th max]`

We might run into the following case as we progress:
- index of 1st max is out of bound of window: we pop left and index of 2nd max because 1st max within window
- the next elem become j-th max: evict old j-th max all the way to index of k-th max on the right of dequeue, i.e. pop right: `[index of 1st max, index of 2nd max, ..., index of j-1-th max, index of new elem]`



**Solution with Double-ended Queue:** [https://replit.com/@trsong/Calculate-Sliding-Window-Maximum](https://replit.com/@trsong/Calculate-Sliding-Window-Maximum)
```py
from collections import deque
import unittest

def max_sliding_window(nums, k):
    res = []
    dq = deque()

    for i, num in enumerate(nums):
        while len(dq) > 0 and nums[dq[-1]] <= num:
            # mantain an increasing double-ended queue
            dq.pop()
        dq.append(i)

        if dq[0] <= i - k:
            dq.popleft()

        if i >= k - 1:
            res.append(nums[dq[0]])
    return res


class MaxSlidingWindowSpec(unittest.TestCase):
    def test_example_array(self):
        k, nums = 3, [1, 3, -1, -3, 5, 3, 6, 7]
        expected = [3, 3, 5, 5, 6, 7]
        self.assertEqual(expected, max_sliding_window(nums, k))

    def test_empty_array(self):
        self.assertEqual([], max_sliding_window([], 1))

    def test_window_has_same_size_as_array(self):
        self.assertEqual([3], max_sliding_window([3, 2, 1], 3))

    def test_window_has_same_size_as_array2(self):
        self.assertEqual([2], max_sliding_window([1, 2], 2))

    def test_window_has_same_size_as_array3(self):
        self.assertEqual([-1], max_sliding_window([-1], 1))

    def test_non_ascending_array(self):
        k, nums = 2, [4, 3, 3, 2, 2, 1]
        expected = [4, 3, 3, 2, 2]
        self.assertEqual(expected, max_sliding_window(nums, k))

    def test_non_ascending_array2(self):
        k, nums = 2, [1, 1, 1]
        expected = [1, 1]
        self.assertEqual(expected, max_sliding_window(nums, k))

    def test_non_descending_array(self):
        k, nums = 3, [1, 1, 2, 2, 2, 3]
        expected = [2, 2, 2, 3]
        self.assertEqual(expected, max_sliding_window(nums, k))
    
    def test_non_descending_array2(self):
        self.assertEqual(max_sliding_window([1, 1, 2, 3], 1), [1, 1, 2 ,3])

    def test_first_decreasing_then_increasing_array(self):
        k, nums = 3, [5, 4, 1, 1, 1, 2, 2, 2]
        expected = [5, 4, 1, 2, 2, 2]
        self.assertEqual(expected, max_sliding_window(nums, k))
    
    def test_first_decreasing_then_increasing_array2(self):
        k, nums = 2, [3, 2, 1, 2, 3]
        expected = [3, 2, 2, 3]
        self.assertEqual(expected, max_sliding_window(nums, k))

    def test_first_decreasing_then_increasing_array3(self):
        k, nums = 3, [3, 2, 1, 2, 3]
        expected = [3, 2, 3]
        self.assertEqual(expected, max_sliding_window(nums, k))
    
    def test_first_increasing_then_decreasing_array(self):
        k, nums = 2, [1, 2, 3, 2, 1]
        expected = [2, 3, 3, 2]
        self.assertEqual(expected, max_sliding_window(nums, k))
    
    def test_first_increasing_then_decreasing_array2(self):
        k, nums = 3, [1, 2, 3, 2, 1]
        expected = [3, 3, 3]
        self.assertEqual(expected, max_sliding_window(nums, k))

    def test_oscillation_array(self):
        k, nums = 2, [1, -1, 1, -1, -1, 1, 1]
        expected = [1, 1, 1, -1, 1, 1]
        self.assertEqual(expected, max_sliding_window(nums, k))
    
    def test_oscillation_array2(self):
        k, nums = 3, [1, 3, 1, 2, 0, 5]
        expected = [3, 3, 2, 5]
        self.assertEqual(expected, max_sliding_window(nums, k))
 

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 10, 2021 LC 1007 \[Medium\] Minimum Domino Rotations For Equal Row
--- 
> **Question:** In a row of dominoes, `A[i]` and `B[i]` represent the top and bottom halves of the ith domino.  (A domino is a tile with two numbers from 1 to 6 - one on each half of the tile.)
>
> We may rotate the ith domino, so that `A[i]` and `B[i]` swap values.
>
> Return the minimum number of rotations so that all the values in A are the same, or all the values in B are the same.
>
> If it cannot be done, return `-1`.

**Example 1:**
```py
Input: A = [2,1,2,4,2,2], B = [5,2,6,2,3,2]
Output: 2
Explanation: 
The first figure represents the dominoes as given by A and B: before we do any rotations.
If we rotate the second and fourth dominoes, we can make every value in the top row equal to 2, as indicated by the second figure.
```

**Example 2:**
```py
Input: A = [3,5,1,2,3], B = [3,6,3,3,4]
Output: -1
Explanation: In this case, it is not possible to rotate the dominoes to make one row of values equal.
```

**My thoughts:** If there exists a solution then it can only be one of the following cases:

1. `A[0]` is the target value: need to rotate rest of `B` dominos to match `A[0]`;
2. `A[0]` is the target value, yet position is not correct: rotate `A[0]` as well as remaining dominos;
3. `B[0]` is the target value: need to rotate rest of `A` dominos to match `B[0]`
4. `B[0]` is the target value, yet position is not correct: rotate `B[0]` as well as remaining dominos.


**Solution:** [https://replit.com/@trsong/Minimum-Domino-Rotations-For-Equal-Row](https://replit.com/@trsong/Minimum-Domino-Rotations-For-Equal-Row)
```py
import unittest

def min_domino_rotations(A, B):
    if not A or not B:
        return 0

    res = min(
        count_rotations(A, B, A[0]),
        count_rotations(A, B, B[0]),
        count_rotations(B, A, A[0]),
        count_rotations(B, A, B[0]))
    return res if res < float('inf') else -1


def count_rotations(A, B, target):
    res = 0
    for num1, num2 in zip(A, B):
        if num1 == target:
            continue
        elif num2 == target:
            res += 1
        else:
            return float('inf')
    return res


class MinDominoRotationSpec(unittest.TestCase):
    def test_example(self):
        A = [2, 1, 2, 4, 2, 2]
        B = [5, 2, 6, 2, 3, 2]
        expected = 2
        self.assertEqual(expected, min_domino_rotations(A, B))

    def test_example2(self):
        A = [3, 5, 1, 2, 3]
        B = [3, 6, 3, 3, 4]
        expected = -1
        self.assertEqual(expected, min_domino_rotations(A, B))

    def test_empty_domino_lists(self):
        self.assertEqual(0, min_domino_rotations([], []))

    def test_rotate_towards_A0(self):
        A = [1, 2, 3, 4]
        B = [3, 1, 1, 1]
        expected = 1
        self.assertEqual(expected, min_domino_rotations(A, B))

    def test_rotate_towards_B0(self):
        A = [0, 3, 3, 3, 4]
        B = [3, 0, 1, 0, 3]
        expected = 2
        self.assertEqual(expected, min_domino_rotations(A, B))

    def test_rotate_A0(self):
        A = [0, 2, 3, 4, 0]
        B = [-1, 0, 0, 0, 1]
        expected = 2
        self.assertEqual(expected, min_domino_rotations(A, B))

    def test_rotate_B0(self):
        A = [1, 1, 2, 2, 2]
        B = [2, 2, 1, 1, 1]
        expected = 2
        self.assertEqual(expected, min_domino_rotations(A, B))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### May 9, 2021 LT 1859 \[Easy\] Minimum Amplitude
--- 
> **Question:** Given an array A consisting of N integers. In one move, we can choose any element in this array and replace it with any value. The amplitude of an array is the difference between the largest and the smallest values it contains.
> 
> Return the smallest amplitude of array A that we can achieve by performing at most three moves.

**Example 1:**
```py
Input: A = [-9, 8, -1]
Output: 0
Explanation: We can replace -9 and 8 with -1 so that all element are equal to -1, and then the amplitude is 0
```

**Example 2:**
```py
Input: A = [14, 10, 5, 1, 0]
Output: 1
Explanation: To achieve an amplitude of 1, we can replace 14, 10 and 5 with 1 or 0.
```

**Example 3:**
```py
Input: A = [11, 0, -6, -1, -3, 5]
Output: 3
Explanation: This can be achieved by replacing 11, -6 and 5 with three values of -2.
```

**Solution with Heap:** [https://replit.com/@trsong/Minimum-Amplitude](https://replit.com/@trsong/Minimum-Amplitude)
```py
import unittest
from queue import PriorityQueue

def min_amplitute(nums):
    heap_size = 4
    if len(nums) <= heap_size:
        return 0

    max_heap = PriorityQueue()
    min_heap = PriorityQueue()

    for i in range(heap_size):
        min_heap.put(nums[i])
        max_heap.put(-nums[i])
    
    for i in range(heap_size, len(nums)):
        num = nums[i]
        if num > min_heap.queue[0]:
            min_heap.get()
            min_heap.put(num)
        if num < abs(max_heap.queue[0]):
            max_heap.get()
            max_heap.put(-num)

    max4, max3, max2, max1 = [min_heap.get() for _ in range(heap_size)]
    min4, min3, min2, min1 = [-max_heap.get() for _ in range(heap_size)]
    return min(max4 - min1, max3 - min2, max2 - min3, max1 - min4)


class MinAmplitute(unittest.TestCase):
    def test_example(self):
        nums = [-9, 8, -1]
        expected = 0
        self.assertEqual(expected, min_amplitute(nums))

    def test_example2(self):
        nums = [14, 10, 5, 1, 0]
        # 5 -> 1, 10 -> 1, 14 -> 1
        expected = 1
        self.assertEqual(expected, min_amplitute(nums))

    def test_example3(self):
        nums = [11, 0, -6, -1, -3, 5]
        # 11 -> 0, -6 -> 0, 5 -> 0
        expected = 3
        self.assertEqual(expected, min_amplitute(nums))

    def test_empty_array(self):
        self.assertEqual(0, min_amplitute([]))

    def test_one_elem_array(self):
        self.assertEqual(0, min_amplitute([42]))

    def test_two_elem_array(self):
        self.assertEqual(0, min_amplitute([42, -43]))

    def test_change_max3_outliers(self):
        nums = [0, 0, 0, 99, 100, 101]
        expected = 0
        self.assertEqual(expected, min_amplitute(nums))

    def test_change_min3_outliers(self):
        nums = [0, 0, 0, -99, -100, -101]
        expected = 0
        self.assertEqual(expected, min_amplitute(nums))

    def test_change_min1_and_max2_outliers(self):
        nums = [0, 0, 0, -99, 100, 101]
        expected = 0
        self.assertEqual(expected, min_amplitute(nums))

    def test_change_min2_and_max1_outliers(self):
        nums = [0, 0, 0, -99, -100, 101]
        expected = 0
        self.assertEqual(expected, min_amplitute(nums))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### May 8, 2021 LC 1525 \[Medium\] Number of Good Ways to Split a String
--- 
> **Question:** Given a string S, we can split S into 2 strings: S1 and S2. Return the number of ways S can be split such that the number of unique characters between S1 and S2 are the same.

**Example 1:**
```py
Input: "aaaa"
Output: 3
Explanation: we can get a - aaa, aa - aa, aaa- a
```

**Example 2:**
```py
Input: "bac"
Output: 0
```

**Example 3:**
```py
Input: "ababa"
Output: 2
Explanation: ab - aba, aba - ba
```

**Solution with Sliding Window:** [https://replit.com/@trsong/Number-of-Good-Ways-to-Split-a-String](https://replit.com/@trsong/Number-of-Good-Ways-to-Split-a-String)
```py
import unittest

def count_equal_splits(s):
    n = len(s)
    last_occurance = {}
    forward_bit_map = 0
    backward_bit_map = 0

    for i in range(n - 1, -1, -1):
        ch = s[i]
        mask = 1 << ord(ch)
        if ~backward_bit_map & mask:
            last_occurance[ch] = i
            backward_bit_map |= mask

    res = 0
    for i, ch in enumerate(s):
        mask = 1 << ord(ch)
        forward_bit_map |= mask
        if last_occurance[ch] == i:
            backward_bit_map ^= mask
        
        if forward_bit_map == backward_bit_map:
            res += 1

    return res
            

class CountEqualSplitSpec(unittest.TestCase):
    def test_example(self):
        s = "aaaa"
        # a - aaa, aa - aa, aaa- a
        expected = 3
        self.assertEqual(expected, count_equal_splits(s))

    def test_example2(self):
        s = "bac"
        expected = 0
        self.assertEqual(expected, count_equal_splits(s))

    def test_example3(self):
        s = "ababa"
        # ab - aba, aba - ba
        expected = 2
        self.assertEqual(expected, count_equal_splits(s))
    
    def test_empty_string(self):
        s = ""
        expected = 0
        self.assertEqual(expected, count_equal_splits(s))
    
    def test_string_with_unique_characters(self):
        s = "abcdef"
        expected = 0
        self.assertEqual(expected, count_equal_splits(s))

    def test_palindrome(self):
        s = "123454321"
        expected = 0
        self.assertEqual(expected, count_equal_splits(s))

    def test_palindrome2(self):
        s = "1234554321"
        expected = 1
        self.assertEqual(expected, count_equal_splits(s))

    def test_string_with_duplicates(self):
        s = "123123112233"
        # 123-123112233, 1231-23112233, 12312-3112233, 123123-112233, 1231231-12233
        expected = 5
        self.assertEqual(expected, count_equal_splits(s))
    

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 7, 2021 \[Medium\] Minimum Days to Bloom Roses
--- 
> **Question:** Given an array of roses. `roses[i]` means rose `i` will bloom on day `roses[i]`. Also given an int `k`, which is the minimum number of adjacent bloom roses required for a bouquet, and an int `n`, which is the number of bouquets we need. Return the earliest day that we can get `n` bouquets of roses.

**Example:**
```py
Input: roses = [1, 2, 4, 9, 3, 4, 1], k = 2, n = 2
Output: 4
Explanation:
day 1: [b, n, n, n, n, n, b]
The first and the last rose bloom.

day 2: [b, b, n, n, n, n, b]
The second rose blooms. Here the first two bloom roses make a bouquet.

day 3: [b, b, n, n, b, n, b]

day 4: [b, b, b, n, b, b, b]
Here the last three bloom roses make a bouquet, meeting the required n = 2 bouquets of bloom roses. So return day 4.
```

**My thoughts:** Unless rose field cannot produce `n * k` roses, the final answer lies between `1` and `max(roses)`. And if on `x` day we can get expected number of bloom roses then any `day > x` can also be. So we can use the binary search to guess the min day to get target number of rose bunquets. 

**Solution with Binary Search:** [https://replit.com/@trsong/Minimum-Days-to-Bloom-Roses](https://replit.com/@trsong/Minimum-Days-to-Bloom-Roses)
```py
import unittest

def min_day_rose_bloom(roses, n, k):
    if n * k > len(roses):
        return -1
    
    lo = 1
    hi = max(roses)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if can_bloom_rose(roses, mid, n, k):
            hi = mid
        else:
            lo = mid + 1
    return lo


def can_bloom_rose(roses, target, n, k):
    bonquet = 0
    count = 0

    for day in roses:
        if day > target:
            count = 0
            continue
        
        count += 1
        if count >= k:
            count = 0
            bonquet += 1
        
        if bonquet >= n:
            break

    return bonquet >= n


class MinDayRoseBloomSpec(unittest.TestCase):
    def test_example(self):
        n, k, roses = 2, 2, [1, 2, 4, 9, 3, 4, 1]
        # [b, n, n, n, n, n, b]
        # [b, b, n, n, n, n, b]
        # [b, b, n, n, b, n, b]
        # [b, b, b, n, b, b, b]
        expected = 4
        self.assertEqual(expected, min_day_rose_bloom(roses, n, k))

    def test_window_size_one(self):
        n, k, roses = 3, 1, [1, 10, 3, 10, 2]
        # [b, n, n, n, n]
        # [b, n, n, n, b]
        # [b, n, b, n, b]
        expected = 3
        self.assertEqual(expected, min_day_rose_bloom(roses, n, k))

    def test_required_size_greater_than_array(self):
        n, k, roses = 3, 2, [1, 1, 1, 1, 1]
        expected = -1
        self.assertEqual(expected, min_day_rose_bloom(roses, n, k))

    def test_just_meet_required_size(self):
        n, k, roses = 2, 3, [1, 2, 3, 1, 2, 3]
        # [b, n, n, b, n, n]
        # [b, b, n, b, b, n]
        # [b, b, b, b, b, b]
        expected = 3
        self.assertEqual(expected, min_day_rose_bloom(roses, n, k))

    def test_array_with_outlier_number(self):
        n, k, roses = 2, 3, [7, 7, 7, 7, 12, 7, 7]
        expected = 12
        self.assertEqual(expected, min_day_rose_bloom(roses, n, k))

    def test_array_with_extreme_large_number(self):
        n, k, roses = 1, 1, [10000, 9999999]
        expected = 10000
        self.assertEqual(expected, min_day_rose_bloom(roses, n, k))

    def test_continuous_bonquet(self):
        n, k, roses = 4, 2, [1, 10, 2, 9, 3, 8, 4, 7, 5, 6]
        expected = 9
        self.assertEqual(expected, min_day_rose_bloom(roses, n, k))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### May 6, 2021 \[Easy\] Find Corresponding Node in Cloned Tree
--- 
> **Question:** Given two binary trees that are duplicates of one another, and given a node in one tree, find that corresponding node in the second tree. 
> 
> There can be duplicate values in the tree (so comparing node1.value == node2.value isn't going to work).

**Solution with DFS Traversal:** [https://replit.com/@trsong/Find-Corresponding-Node-in-Cloned-Tree-2](https://replit.com/@trsong/Find-Corresponding-Node-in-Cloned-Tree-2)
```py
from copy import deepcopy
import unittest

def find_node(root1, root2, node1):
    if root1 is None or root2 is None or node1 is None:
        return None
    
    traversal1 = dfs_traversal(root1)
    traversal2 = dfs_traversal(root2)

    for n1, n2 in zip(traversal1, traversal2):
        if n1 == node1:
            return n2
    return None


def dfs_traversal(root):
    stack = [root]
    while stack:
        cur = stack.pop()
        yield cur
        for child in [cur.left, cur.right]:
            if child is None:
                continue
            stack.append(child)


class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return "TreeNode(%d, %s, %s)" % (self.val, self.left, self.right)


class FindNodeSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertIsNone(find_node(None, None, None))

    def test_one_node_tree(self):
        root1 = TreeNode(1)
        root2 = deepcopy(root1)
        self.assertEqual(root2, find_node(root1, root2, root1))

    def test_leaf_node(self):
        """
          1
         / \
        2   3
             \
              4
        """
        root1 = TreeNode(1, TreeNode(2), TreeNode(3, right=TreeNode(4)))
        root2 = deepcopy(root1)
        f = lambda root: root.right.right
        self.assertEqual(f(root2), find_node(root1, root2, f(root1)))

    def test_internal_node(self):
        """
            1
           / \
          2   3
         /   /
        0   1
        """
        left_tree = TreeNode(2, TreeNode(0))
        right_tree = TreeNode(3, TreeNode(1))
        root1 = TreeNode(1, left_tree, right_tree)
        root2 = deepcopy(root1)
        f = lambda root: root.left
        self.assertEqual(f(root2), find_node(root1, root2, f(root1))) 

    def test_duplicated_value_in_tree(self):
        """
          1
           \
            1
           /
          1
         /
        1
        """
        root1 = TreeNode(1, right=TreeNode(1, TreeNode(1, TreeNode(1))))
        root2 = deepcopy(root1)
        f = lambda root: root.right.left
        self.assertEqual(f(root2), find_node(root1, root2, f(root1))) 
    
    def test_find_root_node(self):
        """
          1
         / \
        2   3
        """
        root1 = TreeNode(1, TreeNode(2), TreeNode(3))
        root2 = deepcopy(root1)
        self.assertEqual(root2, find_node(root1, root2, root1))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 5, 2021 LC 93 \[Medium\] All Possible Valid IP Address Combinations
---
> **Question:** Given a string of digits, generate all possible valid IP address combinations.
>
> IP addresses must follow the format A.B.C.D, where A, B, C, and D are numbers between `0` and `255`. Zero-prefixed numbers, such as `01` and `065`, are not allowed, except for `0` itself.
>
> For example, given `"2542540123"`, you should return `['254.25.40.123', '254.254.0.123']`

**Solution with Backtracking:** [https://replit.com/@trsong/Find-All-Possible-Valid-IP-Address-Combinations-3](https://replit.com/@trsong/Find-All-Possible-Valid-IP-Address-Combinations-3)
```py
import unittest

def all_ip_combinations(raw_str):
    res = []
    accu = []
    n = len(raw_str)

    def backtrack(i):
        if len(accu) > 4:
            return
    
        if i == n and len(accu) == 4:
            res.append('.'.join(accu))
        else:
            for num_digit in [1, 2, 3]:
                if i + num_digit > n: 
                    break
                num = int(raw_str[i: i + num_digit])
                
                # From 0 to 9
                case1 = num_digit == 1

                # From 10 to 99
                case2 = case1 or num_digit == 2 and num >= 10

                # From 100 to 255
                case3 = case2 or num_digit == 3 and 100 <= num <= 255
                if case3:
                    accu.append(str(num))
                    backtrack(i + num_digit)
                    accu.pop()
    
    backtrack(0)
    return res
                    

class AllIpCombinationSpec(unittest.TestCase):
    def test_example(self):
        raw_str = '2542540123'
        expected = ['254.25.40.123', '254.254.0.123']
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_empty_string(self):
        self.assertItemsEqual([], all_ip_combinations(''))

    def test_no_valid_ips(self):
        raw_str = '25505011535'
        expected = []
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_multiple_outcomes(self):
        raw_str = '25525511135'
        expected = ['255.255.11.135', '255.255.111.35']
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_multiple_outcomes2(self):
        raw_str = '25011255255'
        expected = ['250.112.55.255', '250.11.255.255']
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_multiple_outcomes3(self):
        raw_str = '10101010'
        expected = [
            '10.10.10.10', '10.10.101.0', '10.101.0.10', '101.0.10.10',
            '101.0.101.0'
        ]
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_multiple_outcomes4(self):
        raw_str = '01010101'
        expected = ['0.10.10.101', '0.101.0.101']
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_unique_outcome(self):
        raw_str = '111111111111'
        expected = ['111.111.111.111']
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_unique_outcome2(self):
        raw_str = '0000'
        expected = ['0.0.0.0']
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))

    def test_missing_parts(self):
        raw_str = '000'
        expected = []
        self.assertItemsEqual(expected, all_ip_combinations(raw_str))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 4, 2021 \[Medium\] In-place Array Rotation
---
> **Question:** Write a function that rotates an array by `k` elements.
>
> For example, `[1, 2, 3, 4, 5, 6]` rotated by two becomes `[3, 4, 5, 6, 1, 2]`.
>
> Try solving this without creating a copy of the array. How many swap or move operations do you need?

**Solution:** [https://replit.com/@trsong/Rotate-Array-In-place-2](https://replit.com/@trsong/Rotate-Array-In-place-2)
```py
import unittest

def rotate(nums, k):
    if not nums:
        return []
        
    n = len(nums)
    k %= n
    reverse(nums, 0, k - 1)
    reverse(nums, k , n - 1)
    reverse(nums, 0, n - 1)
    return nums


def reverse(nums, start, end):
    while start < end:
        nums[start], nums[end] = nums[end], nums[start]
        start += 1
        end -= 1


class RotateSpec(unittest.TestCase):
    def test_example(self):
        k, nums = 2, [1, 2, 3, 4, 5, 6]
        expected = [3, 4, 5, 6, 1, 2]
        self.assertEqual(expected, rotate(nums, k))

    def test_rotate_0_position(self):
        k, nums = 0, [0, 1, 2, 3]
        expected = [0, 1, 2, 3]
        self.assertEqual(expected, rotate(nums, k))

    def test_empty_array(self):
        self.assertEqual([], rotate([], k=10))

    def test_shift_negative_position(self):
        k, nums = -1, [0, 1, 2, 3]
        expected = [3, 0, 1, 2]
        self.assertEqual(expected, rotate(nums, k))

    def test_shift_more_than_array_size(self):
        k, nums = 8,  [1, 2, 3, 4, 5, 6]
        expected = [3, 4, 5, 6, 1, 2]
        self.assertEqual(expected, rotate(nums, k))

    def test_multiple_round_of_forward_and_backward_shift(self):
        k, nums = 5, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        expected = [5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4]
        self.assertEqual(expected, rotate(nums, k))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 3, 2021 \[Easy\] Find Duplicates
---
> **Question:** Given an array of size n, and all values in the array are in the range 1 to n, find all the duplicates.

**Example:**
```py
Input: [4, 3, 2, 7, 8, 2, 3, 1]
Output: [2, 3]
```

**Solution:** [https://replit.com/@trsong/Find-Duplicates](https://replit.com/@trsong/Find-Duplicates)
```py
import unittest

def find_duplicates(nums):
    res = []
    for num in nums:
        index = abs(num) - 1
        if nums[index] < 0:
            res.append(abs(num))
        else:
            nums[index] *= - 1

    for i in range(len(nums)):
        nums[i] = abs(nums[i])
    return res


class FindDuplicateSpec(unittest.TestCase):
    def assert_result(self, expected, result):
        self.assertEqual(sorted(expected), sorted(result))

    def test_example1(self):
        input = [4, 3, 2, 7, 8, 2, 3, 1]
        expected = [2, 3]
        self.assert_result(expected, find_duplicates(input))

    def test_example2(self):
        input = [4, 5, 2, 6, 8, 2, 1, 5]
        expected = [2, 5]
        self.assert_result(expected, find_duplicates(input))

    def test_empty_array(self):
        self.assertEqual([], find_duplicates([]))

    def test_no_duplicated_numbers(self):
        input = [6, 1, 4, 3, 2, 5]
        expected = []
        self.assert_result(expected, find_duplicates(input))

    def test_duplicated_number(self):
        input = [1, 1, 2]
        expected = [1]
        self.assert_result(expected, find_duplicates(input))

    def test_duplicated_number2(self):
        input = [1, 1, 3, 5, 6, 8, 8, 1, 1]
        expected = [1, 8, 1, 1]
        self.assert_result(expected, find_duplicates(input))
  
    def test_duplicated_number3(self):
        input = [1, 3, 3]
        expected = [3]
        self.assert_result(expected, find_duplicates(input))
    
    def test_duplicated_number4(self):
        input = [3, 2, 3, 2, 3, 2, 7]
        expected = [3, 2, 3, 2]
        self.assert_result(expected, find_duplicates(input))

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```

### May 2, 2021 \[Hard\] Increasing Subsequence of Length K
---
> **Question:** Given an int array nums of length n and an int k. Return an increasing subsequence of length k (KIS). Expected time complexity `O(nlogk)`.

**Example 1:**
```py
Input: nums = [10, 1, 4, 8, 2, 9], k = 3
Output: [1, 4, 8] or [1, 4, 9] or [1, 8, 9]
```

**Example 2:**
```py
Input: nums = [10, 1, 4, 8, 2, 9], k = 4
Output: [1, 4, 8, 9]
```

**Solution with DP and Binary Search:** [https://replit.com/@trsong/Increasing-Subsequence-of-Length-K-2](https://replit.com/@trsong/Increasing-Subsequence-of-Length-K-2)
```py
import unittest

def increasing_sequence(nums, k):
    if not nums:
        return []

    # Let dp[i] represents i-th element in a length i + 1 length subsequence
    dp = []
    prev_elem = {}

    for num in nums:
        insert_pos = binary_search(dp, num)
        if insert_pos == len(dp):
            dp.append(num)
        else:
            dp[insert_pos] = num
        prev_elem[num] = dp[insert_pos - 1] if insert_pos > 0 else None

        if len(dp) == k:
            break

    res = []
    num = dp[-1]
    while num is not None:
        res.append(num)
        num = prev_elem[num]
    return res[::-1]


def binary_search(dp, target):
    lo = 0
    hi = len(dp)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if dp[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


class IncreasingSequenceSpec(unittest.TestCase):
    def validate_result(self, nums, k):
        subseq = increasing_sequence(nums, k)
        self.assertEqual(k, len(subseq), str(subseq) + " Is not of length " + str(k))

        i = 0
        for num in nums:
            if i < len(subseq) and num == subseq[i]:
                i += 1
        self.assertEqual(len(subseq), i, str(subseq) + " Is not valid subsequence.")
        
        for i in xrange(1, len(subseq)):
            self.assertLessEqual(subseq[i-1], subseq[i], str(subseq) + " Is not increasing subsequene.")

    def test_example(self):
        k, nums = 3, [10, 1, 4, 8, 2, 9]
        self.validate_result(nums, k)  # possible result: [1, 4, 8]

    def test_example2(self):
        k, nums = 4, [10, 1, 4, 8, 2, 9]
        self.validate_result(nums, k)  # possible result: [1, 4, 8, 9]

    def test_empty_sequence(self):
        k, nums = 0, []
        self.validate_result(nums, k)

    def test_longest_increasing_subsequence(self):
        k, nums = 4, [10, 9, 2, 5, 3, 7, 101, 18]
        self.validate_result(nums, k)  # possible result: [2, 3, 7, 101]

    def test_longest_increasing_subsequence_in_second_half_sequence(self):
        k, nums = 4, [1, 2, 3, -2, -1, 0, 1]
        self.validate_result(nums, k)  # possible result: [-2, -1, 0, 1]

    def test_should_return_valid_subsequene(self):
        k, nums = 3, [8, 9, 7, 6, 10]
        self.validate_result(nums, k)  # possible result: [8, 9, 10]


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


### May 1, 2021 \[Hard\] Decreasing Subsequences
---
> **Question:** Given an int array nums of length n. Split it into strictly decreasing subsequences. Output the min number of subsequences you can get by splitting.

**Example 1:**
```py
Input: [5, 2, 4, 3, 1, 6]
Output: 3
Explanation:
You can split this array into: [5, 2, 1], [4, 3], [6]. And there are 3 subsequences you get.
Or you can split it into [5, 4, 3], [2, 1], [6]. Also 3 subsequences.
But [5, 4, 3, 2, 1], [6] is not legal because [5, 4, 3, 2, 1] is not a subsuquence of the original array.
```

**Example 2:**
```py
Input: [2, 9, 12, 13, 4, 7, 6, 5, 10]
Output: 4
Explanation: [2], [9, 4], [12, 10], [13, 7, 6, 5]
```

**Example 3:**
```py
Input: [1, 1, 1]
Output: 3
Explanation: Because of the strictly descending order you have to split it into 3 subsequences: [1], [1], [1]
```

**My thoughts:** This question is equivalent to [Longest Increasing Subsequence](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-19-2020-lc-300-hard-the-longest-increasing-subsequence). Can be solved with greedy approach.

Imagine we are to create a list of stacks each in descending order (stack top is smallest). And those stacks are sorted by each stack's top element. 

Then for each element from input sequence, we just need to figure out (using binary search) the stack such that by pushing this element into stack, result won't affect the order of stacks and decending property of each stack. 

Finally, the total number of stacks equal to min number of subsequence we can get by splitting. Each stack represents a decreasing subsequence.

**Greedy Solution with Descending Stack and Binary Search:** [https://replit.com/@trsong/Decreasing-Subsequences](https://replit.com/@trsong/Decreasing-Subsequences)
```py
import unittest

def min_decreasing_subsequences(sequence):
    # maintain a list of descending stacks sorted by each stack top 
    stack_list = []
    for num in sequence:
        stack_index = binary_search_stack_top(stack_list, num)
        if stack_index == len(stack_list):
            stack_list.append([])
        stack_list[stack_index].append(num)
    return len(stack_list)


def binary_search_stack_top(stack_list, target):
    lo = 0
    hi = len(stack_list)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        stack_top = stack_list[mid][-1]
        if stack_top <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


class MinDecreasingSubsequnceSpec(unittest.TestCase):
    def test_example(self):
        sequence = [5, 2, 4, 3, 1, 6]
        # [5, 2, 1]
        # [4, 3]
        # [6]
        expected = 3
        self.assertEqual(expected, min_decreasing_subsequences(sequence))

    def test_example2(self):
        sequence = [2, 9, 12, 13, 4, 7, 6, 5, 10]
        # [2]
        # [9, 4]
        # [12, 7, 6, 5]
        # [13, 10]
        expected = 4
        self.assertEqual(expected, min_decreasing_subsequences(sequence))

    def test_example3(self):
        sequence = [1, 1, 1]
        # [1]
        # [1]
        # [1]
        expected = 3
        self.assertEqual(expected, min_decreasing_subsequences(sequence))

    def test_empty_sequence(self):
        self.assertEqual(0, min_decreasing_subsequences([]))

    def test_last_elem_is_local_max(self):
        seq = [1, 2, 3, 0, 2]
        # [1, 0]
        # [2]
        # [3, 2]
        expected = 3 
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_last_elem_is_global_max(self):
        seq = [1, 2, 3, 0, 6]
        # [1, 0]
        # [2]
        # [3]
        # [6]
        expected = 4
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_min_decreasing_subsequences_in_first_half_sequence(self):
        seq = [4, 5, 6, 7, 1, 2, 3]
        # [4, 1]
        # [5, 2]
        # [6, 3]
        # [7]
        expected = 4 
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_min_decreasing_subsequences_in_second_half_sequence(self):
        seq = [1, 2, 3, -2, -1, 0, 1]
        # [1, -2]
        # [2, -1]
        # [3, 0]
        # [1]
        expected = 4
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_sequence_in_up_down_up_pattern(self):
        seq = [1, 2, 3, 2, 4]
        # [1]
        # [2]
        # [3, 2]
        # [4]
        expected = 4 
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_sequence_in_up_down_up_pattern2(self):
        seq = [1, 2, 3, -1, 0]
        # [1, -1, 0]
        # [2]
        # [3]
        expected = 3 
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_sequence_in_down_up_down_pattern(self):
        seq = [4, 3, 5]
        # [4, 3]
        # [5]
        expected = 2
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_sequence_in_down_up_down_pattern2(self):
        seq = [4, 0, 1]
        # [4, 0]
        # [1]
        expected = 2
        self.assertEqual(expected, min_decreasing_subsequences(seq))

    def test_multiple_result(self):
        seq = [10, 9, 2, 5, 3, 7, 101, 18]
        # [10, 9, 2]
        # [5, 3]
        # [7]
        # [101, 18]
        expected = 4 
        self.assertEqual(expected, min_decreasing_subsequences(seq))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
```


