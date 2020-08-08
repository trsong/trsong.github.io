---
layout: post
title:  "Daily Coding Problems 2020 Aug to Oct"
date:   2020-08-01 22:22:32 -0700
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



### Aug 8, 2020 \[Easy\] Flatten a Nested Dictionary
---
> **Question:** Write a function to flatten a nested dictionary. Namespace the keys with a period.

**Example:**

```py
Given the following dictionary:
{
    "key": 3,
    "foo": {
        "a": 5,
        "bar": {
            "baz": 8
        }
    }
}

it should become:
{
    "key": 3,
    "foo.a": 5,
    "foo.bar.baz": 8
}

You can assume keys do not contain dots in them, i.e. no clobbering will occur.
```

### Aug 7, 2020 \[Hard\] Shortest Uphill and Downhill Route
---
> **Question:** A competitive runner would like to create a route that starts and ends at his house, with the condition that the route goes entirely uphill at first, and then entirely downhill.
>
> Given a dictionary of places of the form `{location: elevation}`, and a dictionary mapping paths between some of these locations to their corresponding distances, find the length of the shortest route satisfying the condition above. Assume the runner's home is location `0`.

**Example:**
```py
Suppose you are given the following input:

elevations = {0: 5, 1: 25, 2: 15, 3: 20, 4: 10}
paths = {
    (0, 1): 10,
    (0, 2): 8,
    (0, 3): 15,
    (1, 3): 12,
    (2, 4): 10,
    (3, 4): 5,
    (3, 0): 17,
    (4, 0): 10
}

In this case, the shortest valid path would be 0 -> 2 -> 4 -> 0, with a distance of 28.
```

**My thoughts:** Break the graph in to uphill and downhill graph separately. For uphill graph, we can calculate shortest distance from 0 to target node. For downhill graph, we want to calculate shortest distance from node to 0. But how? One way is that we can reverse the edge of graph and convert the problem to find shortest distance from 0 to node. Once get shorest distance from and to node, we can sum them up and find the shorest combined distance which will give the final solution.

Then how to do we find shortest distance in directed graph? We can use Dijkstra's algorithm that will be `O(V + E log E)` due to priority queue used in the algorithm. However, notice that in this problem neither uphill graph nor downhil graph can form a cycle. Then we will have a DAG. The shortest distance in DAG can be found in `O(V + E)`. Therefore our solution will have `O(V + E)` in time complexity.

**Solution with Topological Sort:** [https://repl.it/@trsong/Shortest-Uphill-and-Downhill-Route](https://repl.it/@trsong/Shortest-Uphill-and-Downhill-Route)
```py
import unittest
from collections import defaultdict

def shortest_uphill_downhill_cycle(elevations, paths):
    uphill_neighbors = defaultdict(list)
    reverse_downhill_neighbors = defaultdict(list)
    for u, v in paths:
        if elevations[u] < elevations[v]:
            uphill_neighbors[u].append(v)
        else:
            reverse_downhill_neighbors[v].append(u)
    
    uphill_order = topological_sort(uphill_neighbors, elevations)
    shorest_distance_to = shortest_distance(uphill_order, uphill_neighbors, paths)

    reverse_downhill_order = topological_sort(reverse_downhill_neighbors, elevations)
    shortest_distance_from = shortest_distance(reverse_downhill_order, reverse_downhill_neighbors, paths, reversed=True)

    del shorest_distance_to[0]
    del shortest_distance_from[0]
    shortest_cycle_through = map(lambda u: shorest_distance_to[u] + shortest_distance_from[u], elevations.keys())
    return min(shortest_cycle_through) 

    
def topological_sort(neighbors, elevations):
    class Vertex_State:
        UNVISITED = 0
        VISITING = 1
        VISITED = 2

    vertext_states = {node: Vertex_State.UNVISITED for node in elevations}
    reverse_top_order = []

    for node, node_state in vertext_states.items():
        if node_state is not Vertex_State.UNVISITED:
            continue
        
        stack = [node]
        while stack:
            cur = stack[-1]
            if vertext_states[cur] is Vertex_State.VISITED:
                stack.pop()
            elif vertext_states[cur] is Vertex_State.VISITING:
                vertext_states[cur] = Vertex_State.VISITED
                reverse_top_order.append(cur)
            else:
                # vertext_states[cur] is Vertex_State.UNVISITED
                vertext_states[cur] = Vertex_State.VISITING
                for nb in neighbors[cur]:
                    if vertext_states[nb] is Vertex_State.UNVISITED:
                        stack.append(nb)
    
    return reverse_top_order[::-1]


def shortest_distance(top_order, neighbors, paths, reversed=False):
    node_distance = defaultdict(lambda: float('inf'))
    node_distance[0] = 0
    for u in top_order:
        for v in neighbors[u]:
            w = float('inf')
            if reversed:
                w = paths.get((v, u), float('inf'))
            else:
                w = paths.get((u, v), float('inf'))
            node_distance[v] = min(node_distance[v], node_distance[u] + w)
    return node_distance


class ShortestUphillDownhillCycleSpec(unittest.TestCase):
    def test_example(self):
        elevations = {0: 5, 1: 25, 2: 15, 3: 20, 4: 10}
        paths = {
            (0, 1): 10,
            (0, 2): 8,
            (0, 3): 15,
            (1, 3): 12,
            (2, 4): 10,
            (3, 4): 5,
            (3, 0): 17,
            (4, 0): 10
        }
        expected = 28  # 0 -> 2 -> 4 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))

    def test_choose_between_downhill_routes(self):
        """
         1
       / | \
      2  |  3
       \ | /
         0
        """
        elevations = {0: 0, 1: 10, 2: 5, 3: 7}
        paths = {
            (0, 1): 10,
            (1, 2): 20,
            (2, 0): 30,
            (1, 3): 5,
            (3, 0): 6
        }
        expected = 21  # 0 -> 1 -> 3 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))
    
    def test_star_graph(self):
        """
          3
          |
          0
         / \
        1   2
        """
        elevations = {0: 0, 1: 1, 2: 2, 3: 3}
        paths = {
            (0, 1): 1,
            (1, 0): 10,
            (0, 2): 8,
            (2, 0): 5,
            (0, 3): 4,
            (3, 0): 6
        }
        expected = 10  # 0 -> 3 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))
    
    def test_clockwise_vs_conterclockwise(self):
        """
        0 - 1
        |   |
        3 - 2 
        """
        elevations = {0: 0, 1: 1, 2: 2, 3: 3}
        paths = {
            (0, 1): 10,
            (1, 2): 20,
            (2, 3): 30,
            (3, 0): 99,
            (0, 3): 1,
            (3, 2): 2,
            (2, 1): 3,
            (1, 0): 4
        }
        expected = 10  # 0 -> 3 -> 2 -> 1 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))
    
    def test_choose_downhill_or_uphill(self):
        """
         1
       / | \
      2  |  3
       \ | /
         0
        """
        elevations = {0: 0, 1: 50, 2: 25, 3: 100}
        paths = {
            (0, 1): 10,
            (1, 0): 9999,
            (1, 2): 5,
            (2, 0): 3,
            (1, 3): 1,
            (3, 0): 0,
            (0, 3): 999,
            (0, 2): 999
        }
        expected = 11  # 0 -> 1 -> 3 -> 0
        self.assertEqual(expected, shortest_uphill_downhill_cycle(elevations, paths))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 6, 2020 LC 78 \[Medium\] Generate All Subsets
---
> **Question:** Given a list of unique numbers, generate all possible subsets without duplicates. This includes the empty set as well.

**Example:**
```py
generate_all_subsets([1, 2, 3])
# [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]
```

**Solution with Recursion:** [https://repl.it/@trsong/Generate-All-Subsets-with-recursion](https://repl.it/@trsong/Generate-All-Subsets-with-recursion)
```py
from functools import reduce

def generate_all_subsets(nums):
    return reduce(
        lambda accu_subsets, elem: accu_subsets + map(lambda subset: subset + [elem], accu_subsets),
        nums,
        [[]])
```

**Solution with Backtracking:** [https://repl.it/@trsong/Generate-All-the-Subsets](https://repl.it/@trsong/Generate-All-the-Subsets)
```py
import unittest

def generate_all_subsets(nums):
    res = []
    backtrack(0, res, [], nums)
    return res


def backtrack(next_index, res, accu, nums):
    res.append(accu[:])
    for i in xrange(next_index, len(nums)):
        accu.append(nums[i])
        backtrack(i+1, res, accu, nums)
        accu.pop()


class GenerateAllSubsetSpec(unittest.TestCase):
    def test_example(self):
        nums = [1, 2, 3]
        expected = [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]
        self.assertItemsEqual(expected, generate_all_subsets(nums))

    def test_empty_list(self):
        nums = []
        expected = [[]]
        self.assertItemsEqual(expected, generate_all_subsets(nums))

    def test_one_elem_list(self):
        nums = [1]
        expected = [[], [1]]
        self.assertItemsEqual(expected, generate_all_subsets(nums))

    def test_two_elem_list(self):
        nums = [1, 2]
        expected = [[1], [2], [1, 2], []]
        self.assertItemsEqual(expected, generate_all_subsets(nums))

    def test_four_elem_list(self):
        nums = [1, 2, 3, 4]
        expected = [
            [], 
            [1], [2], [3],  [4],
            [1, 2], [1, 3], [2, 3], [1, 4], [2, 4], [3, 4],
            [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4], [1, 2, 3, 4]
        ]
        self.assertItemsEqual(expected, generate_all_subsets(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 5, 2020 \[Easy\] Array of Equal Parts
---
> **Question:** Given an array containing only positive integers, return if you can pick two integers from the array which cuts the array into three pieces such that the sum of elements in all pieces is equal.

**Example 1:**
```py
Input: [2, 4, 5, 3, 3, 9, 2, 2, 2]
Output: True
Explanation: choosing the number 5 and 9 results in three pieces [2, 4], [3, 3] and [2, 2, 2]. Sum = 6.
```

**Example 2:**
```py
Input: [1, 1, 1, 1]
Output: False
```

**Solution with Prefix Sum and Two Pointers:** [https://repl.it/@trsong/Array-of-Equal-Parts](https://repl.it/@trsong/Array-of-Equal-Parts)
```py
import unittest

def contains_3_equal_parts(nums):
    if not nums:
        return False

    prefix_sum = nums[:]
    n = len(nums)
    for i in xrange(1, n):
        prefix_sum[i] += prefix_sum[i-1]

    i, j = 0, n-1
    total = prefix_sum[n-1]
    while i < j:
        left_sum = prefix_sum[i] - nums[i]
        right_sum = total - prefix_sum[j]
        if left_sum < right_sum:
            i += 1
        elif left_sum > right_sum:
            j -= 1
        else:
            # left_sum == right_sum
            mid_sum = prefix_sum[j] - nums[j] - prefix_sum[i]
            if mid_sum == left_sum:
                return True
            i += 1
            j -= 1

    return False
        

class Contains3EqualPartSpec(unittest.TestCase):
    def test_example(self):
        nums = [2, 4, 5, 3, 3, 9, 2, 2, 2]
        # Remove 5, 9 to break array into [2, 4], [3, 3] and [2, 2, 2]
        self.assertTrue(contains_3_equal_parts(nums))

    def test_example2(self):
        nums = [1, 1, 1, 1]
        self.assertFalse(contains_3_equal_parts(nums))

    def test_empty_array(self):
        self.assertFalse(contains_3_equal_parts([]))

    def test_two_element_array(self):
        nums = [1, 2]
        # [], [], []
        self.assertTrue(contains_3_equal_parts(nums))

    def test_three_element_array(self):
        nums = [1, 2, 3]
        self.assertFalse(contains_3_equal_parts(nums))

    def test_symmetic_array(self):
        nums = [1, 2, 4, 3, 5, 2, 1]
        # remove 4, 5 gives [1, 2], [3], [2, 1]
        self.assertTrue(contains_3_equal_parts(nums))

    def test_sum_not_divisiable_by_3(self):
        nums = [2, 2, 2, 2, 2, 2]
        self.assertFalse(contains_3_equal_parts(nums))

    def test_ascending_array(self):
        nums = [1, 2, 3, 3, 3, 3, 4, 6]
        # remove 3, 4 gives [1, 2, 3], [3, 3], [6]
        self.assertTrue(contains_3_equal_parts(nums))


if __name__ == '__main__':
    unittest.main(exit=False)
```



### Aug 4, 2020 \[Medium\] M Smallest in K Sorted Lists
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

**My thoughts:** This problem is almost the same as merge k sorted list. The idea is to leverage priority queue to keep track of minimum element among all k sorted list.

**Solution with Priority Queue:** [https://repl.it/@trsong/Find-M-Smallest-in-K-Sorted-Lists](https://repl.it/@trsong/Find-M-Smallest-in-K-Sorted-Lists)
```py
import unittest
from Queue import PriorityQueue

def find_m_smallest(ksorted_list, m):
    sorted_iterators = map(iter, ksorted_list)
    pq = PriorityQueue()

    for it in sorted_iterators:
        num = next(it, None)
        if num is not None:
            pq.put((num, it))

    while m > 1 and not pq.empty():
        _, it = pq.get()
        next_num = next(it, None)
        if next_num is not None:
            pq.put((next_num, it))
        m -= 1

    return None if pq.empty() else pq.get()[0]


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


### Aug 3, 2020 \[Hard\] Graph Coloring
---
> **Question:** Given an undirected graph represented as an adjacency matrix and an integer `k`, determine whether each node in the graph can be colored such that no two adjacent nodes share the same color using at most `k` colors.

**My thoughts:** Solve this problem with backtracking. For each node, testing all colors one-by-one; if it turns out there is something wrong with current color, we will backtrack to test other colors.

**Solution with Backtracking:** [https://repl.it/@trsong/k-Graph-Coloring](https://repl.it/@trsong/k-Graph-Coloring)
```py
import unittest

def solve_graph_coloring(neighbor_matrix, k):
    n = len(neighbor_matrix)
    node_color = [None] * n

    def backtrack(next_node):
        if next_node >= n:
            return True
        
        processed_neighbors = [i for i in xrange(next_node) if neighbor_matrix[next_node][i]]
        for color in xrange(k):
            if any(color == node_color[nb] for nb in processed_neighbors):
                continue
            node_color[next_node] = color
            if backtrack(next_node+1):
                return True
            node_color[next_node] = None
        return False

    return backtrack(0)
        

class SolveGraphColoringSpec(unittest.TestCase):
    @staticmethod
    def generateCompleteGraph(n):
        return [[1 if i != j else 0 for i in xrange(n)] for j in xrange(n)] 

    def test_k2_graph(self):
        k2 = SolveGraphColoringSpec.generateCompleteGraph(2)
        self.assertFalse(solve_graph_coloring(k2, 1))
        self.assertTrue(solve_graph_coloring(k2, 2))
        self.assertTrue(solve_graph_coloring(k2, 3))

    def test_k3_graph(self):
        k3 = SolveGraphColoringSpec.generateCompleteGraph(3)
        self.assertFalse(solve_graph_coloring(k3, 2))
        self.assertTrue(solve_graph_coloring(k3, 3))
        self.assertTrue(solve_graph_coloring(k3, 4))

    def test_k4_graph(self):
        k4 = SolveGraphColoringSpec.generateCompleteGraph(4)
        self.assertFalse(solve_graph_coloring(k4, 3))
        self.assertTrue(solve_graph_coloring(k4, 4))
        self.assertTrue(solve_graph_coloring(k4, 5))

    def test_square_graph(self):
        square = [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ]
        self.assertFalse(solve_graph_coloring(square, 1))
        self.assertTrue(solve_graph_coloring(square, 2))
        self.assertTrue(solve_graph_coloring(square, 3))

    def test_star_graph(self):
        star = [
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0]
        ]
        self.assertFalse(solve_graph_coloring(star, 2))
        self.assertTrue(solve_graph_coloring(star, 3))
        self.assertTrue(solve_graph_coloring(star, 4))

    def test_disconnected_graph(self):
        disconnected = [[0 for _ in xrange(10)] for _ in xrange(10)]
        self.assertTrue(solve_graph_coloring(disconnected, 1))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 2, 2020 \[Medium\] Maximum Circular Subarray Sum
---
> **Question:** Given a circular array, compute its maximum subarray sum in `O(n)` time. A subarray can be empty, and in this case the sum is 0.

**Example 1:**
```py
Input: [8, -1, 3, 4]
Output: 15 
Explanation: we choose the numbers 3, 4, and 8 where the 8 is obtained from wrapping around.
```

**Example 2:**
```py
Input: [-4, 5, 1, 0]
Output: 6 
Explanation: we choose the numbers 5 and 1.
```

**My thoughts:** The max circular subarray sum can be divied into sub-problems: max non-circular subarray sum and max circular-only subarray sum. 

For max non-circular subarray sum problem, we can use `dp[i]` to represent max subarray sum end at index `i` and `max(dp)` will be the answer.

For max circular-only subarray sum problem, we want to find `i`, `j` where `i < j` such that `nums[0] + nums[1] + ... + nums[i] + nums[j] + .... + nums[n-1]` reaches maximum. The way we can handle it is to calculate prefix sum and suffix sum array and find max accumulated sum on the left and on the right. The max circular-only subarray sum equals the sum of those two accumulated sum. 

Finally, the answer to the original problem is the larger one between answers to above two sub-problems. And one thing worth to notice is that if all elements are negative, then the answer should be `0`.

**Solution with DP and Prefix-Sum:** [https://repl.it/@trsong/Find-Maximum-Circular-Subarray-Sum](https://repl.it/@trsong/Find-Maximum-Circular-Subarray-Sum)
```py
import unittest

def max_circular_sum(nums):
    return max(max_subarray_sum(nums), max_circular_subarray_sum(nums))


def max_subarray_sum(nums):
    n = len(nums)
    # Let dp[n] represents max subarray max ends at index n-1
    dp = [0] * (n+1)
    for i in xrange(1, n+1):
        dp[i] = nums[i-1] + max(dp[i-1], 0)
    return max(dp)


def max_circular_subarray_sum(nums):
    if not nums:
        return 0

    left_max_sums = max_prefix_sums(nums)
    right_max_sums = reversed(max_prefix_sums(reversed(nums)))
    combined_sum = map(sum, zip(left_max_sums, right_max_sums))
    return max(combined_sum)


def max_prefix_sums(stream):
    res = []
    prefix_sum = 0
    max_prefix_sum = 0

    for num in stream:
        max_prefix_sum = max(max_prefix_sum, prefix_sum)
        res.append(max_prefix_sum)
        prefix_sum += num

    return res


class MaxCircularSumSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(15, max_circular_sum([8, -1, 3, 4]))  # 3 + 4 + 8
    
    def test_example2(self):
        self.assertEqual(6, max_circular_sum([-4, 5, 1, 0]))  # 5 + 1

    def test_empty_array(self):
        self.assertEqual(0, max_circular_sum([]))

    def test_negative_array(self):
        self.assertEqual(0, max_circular_sum([-1, -2, -3]))

    def test_circular_array1(self):
        self.assertEqual(22, max_circular_sum([8, -8, 9, -9, 10, -11, 12]))  # 12 + 8 - 8 + 9 - 9 + 10

    def test_circular_array2(self):
        self.assertEqual(23, max_circular_sum([10, -3, -4, 7, 6, 5, -4, -1]))  # 7 + 6 + 5 - 4 -1 + 10

    def test_circular_array3(self):
        self.assertEqual(52, max_circular_sum([-1, 40, -14, 7, 6, 5, -4, -1]))  # 7 + 6 + 5 - 4 - 1 - 1 + 40

    def test_all_positive_array(self):
        self.assertEqual(10, max_circular_sum([1, 2, 3, 4]))


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 1, 2020 LC 934 \[Medium\] Shortest Bridge
---
> **Question:** In a given 2D binary array A, there are two islands.  (An island is a 4-directionally connected group of 1s not connected to any other 1s.)
>
> Now, we may change 0s to 1s so as to connect the two islands together to form 1 island.
>
> Return the smallest number of 0s that must be flipped.  (It is guaranteed that the answer is at least 1.)

**Example 1:**
```py
Input: 
[
    [0, 1],
    [1, 0]
]
Output: 1
```

**Example 2:**
```py
Input: 
[
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 1]
]
Output: 2
```

**Example 3:**
```py
Input: 
[
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]
Output: 1
```


**My thoughts:** First color each island with 2 and -2. Then search from each island by bi-directional bfs (2-way bfs). Like below

```py
Color Each Island, mark coastline1 and coastline2 to be start queue and end queue

  .   .   .   .   .   .   .   .
  .   .   .   2   .   2   2   .
  .   .   2   2   2   2   .   .
  .   2   2   2   2   2   .   .
  .   2   .   .   2   2   .   .
  .   .   .   .   2   2   .   .
  .   .   .   .   2   .   .   .
  .   .   .   .   2   .   .   .
  .   .   .   .   .   .   .   .
  .  -2   .   .   .   .   .   .
  .  -2  -2   .   .   .   .   .
  .   .  -2   .   .   .   .   .
  .   .  -2  -2  -2   .   .   .


======= Iteration 1 ======
=> Start Queue Move 

  .   .   .   3   .   3   3   .
  .   .   3   2   3   2   2   3
  .   3   2   2   2   2   3   .
  3   2   2   2   2   2   3   .
  3   2   3   3   2   2   3   .
  .   3   .   3   2   2   3   .
  .   .   .   3   2   3   .   .
  .   .   .   3   2   3   .   .
  .   .   .   .   3   .   .   .
  .  -2   .   .   .   .   .   .
  .  -2  -2   .   .   .   .   .
  .   .  -2   .   .   .   .   .
  .   .  -2  -2  -2   .   .   .

=> End Queue Move 

  .   .   .   3   .   3   3   .
  .   .   3   2   3   2   2   3
  .   3   2   2   2   2   3   .
  3   2   2   2   2   2   3   .
  3   2   3   3   2   2   3   .
  .   3   .   3   2   2   3   .
  .   .   .   3   2   3   .   .
  .   .   .   3   2   3   .   .
  .  -3   .   .   3   .   .   .
 -3  -2  -3   .   .   .   .   .
 -3  -2  -2  -3   .   .   .   .
  .  -3  -2  -3  -3   .   .   .
  .  -3  -2  -2  -2  -3   .   .


======= Iteration 2 ======
=> Start Queue Move 

  .   .   4   3   4   3   3   4
  .   4   3   2   3   2   2   3
  4   3   2   2   2   2   3   4
  3   2   2   2   2   2   3   4
  3   2   3   3   2   2   3   4
  4   3   4   3   2   2   3   4
  .   4   4   3   2   3   4   .
  .   .   4   3   2   3   4   .
  .  -3   .   4   3   4   .   .
 -3  -2  -3   .   4   .   .   .
 -3  -2  -2  -3   .   .   .   .
  .  -3  -2  -3  -3   .   .   .
  .  -3  -2  -2  -2  -3   .   .

=> End Queue Move 

  .   .   4   3   4   3   3   4
  .   4   3   2   3   2   2   3
  4   3   2   2   2   2   3   4
  3   2   2   2   2   2   3   4
  3   2   3   3   2   2   3   4
  4   3   4   3   2   2   3   4
  .   4   4   3   2   3   4   .
  .  -4   4   3   2   3   4   .
 -4  -3  -4   4   3   4   .   .
 -3  -2  -3  -4   4   .   .   .
 -3  -2  -2  -3  -4   .   .   .
 -4  -3  -2  -3  -3  -4   .   .
 -4  -3  -2  -2  -2  -3  -4   .

```

**Solution with 2-way BFS:** [https://repl.it/@trsong/Shortest-Bridge](https://repl.it/@trsong/Shortest-Bridge)
```py
import unittest
from Queue import deque


SEARCH_DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

def shortest_bridge(grid):
    coastline1, coastline2 = dfs_partition(grid)
    distance = bi_directional_bfs(grid, coastline1, coastline2)
    return distance


def dfs_partition(grid):
    """
    Color first island with 2 and second island with -2. Return coastline of each.
    """
    n, m = len(grid), len(grid[0])
    sea_color = 0
    original_island_color = 1
    current_color = 2
    max_num_island = 2
    res = []

    for r in xrange(n):
        for c in xrange(m):
            if grid[r][c] != original_island_color:
                continue
            
            island_costline = []
            stack = [(r, c)]
            while stack:
                cur_r, cur_c = stack.pop()

                # visited pos has same color
                if grid[cur_r][cur_c] is current_color:
                    continue
                grid[cur_r][cur_c] = current_color

                is_coast = False
                for dr, dc in SEARCH_DIRECTIONS:
                    new_r = cur_r + dr
                    new_c = cur_c + dc
                    if not (0 <= new_r < n and 0 <= new_c < m):
                        continue
                    
                    if grid[new_r][new_c] is sea_color:
                        is_coast = True
                    elif grid[new_r][new_c] is original_island_color:
                        # unvisited pos has original color
                        stack.append((new_r, new_c))

                if is_coast:
                    island_costline.append((cur_r, cur_c))
            
            # the other island has negative color
            current_color = -2
            res.append(island_costline)
            if len(res) == max_num_island:
                return res

    return res


def bi_directional_bfs(grid, coastline1, coastline2):
    iteration = 0
    color = 2
    start_queue, end_queue = deque(coastline1), deque(coastline2)
    while start_queue and end_queue:
        if search_by_level(grid, start_queue, color):
            return 2 * iteration - 2

        if search_by_level(grid, end_queue, -color):
            return 2 * iteration - 1

        iteration += 1
        color += 1

    return None


def search_by_level(grid, queue, current_color):
    n, m = len(grid), len(grid[0])
    sign = 1 if current_color > 0 else -1
    sea_color = 0

    for _ in xrange(len(queue)):
        cur_r, cur_c = queue.popleft()

        if grid[cur_r][cur_c] * sign < 0:
            # found the other island
            return True

        if grid[cur_r][cur_c] != sea_color and abs(grid[cur_r][cur_c]) < abs(current_color):
            # visited pos has smaller color 
            continue
        
        grid[cur_r][cur_c] = current_color
        for dr, dc in SEARCH_DIRECTIONS:
            new_r = cur_r + dr
            new_c = cur_c + dc
            if not (0 <= new_r < n and 0 <= new_c < m):
                continue 
            
            if grid[new_r][new_c] is sea_color or grid[new_r][new_c] * sign < 0:
                queue.append((new_r, new_c))
    
    return False
    

class ShortestBridgeSpec(unittest.TestCase):
    @staticmethod
    def print_grid(grid):
        print
        for row in grid:
            print ' '.join('{:3}'.format(v if v != 0 else '.') for v in row)
        print

    def test_example(self):
        grid = [
            [0, 1], 
            [1, 0]
        ]
        self.assertEqual(1, shortest_bridge(grid))  # flip (1, 1)

    def test_example2(self):
        grid = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1]
        ]
        self.assertEqual(2, shortest_bridge(grid))  # flip (0, 2) and (1, 2)

    def test_example3(self):
        grid = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
        self.assertEqual(1, shortest_bridge(grid))  # flip (1, 1)

    def test_1D_grid(self):
        grid = [[1, 0, 1]]
        self.assertEqual(1, shortest_bridge(grid))

    def test_1D_grid2(self):
        grid = [[1, 0, 0, 1]]
        self.assertEqual(2, shortest_bridge(grid))
    
    def test_not_all_0_are_connected(self):
        grid = [
            [0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 0],
        ]
        self.assertEqual(2, shortest_bridge(grid))  # flip (1, 2) and (1, 3)

    def test_complicated_islands(self):
        grid = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
        ]
        self.assertEqual(4, shortest_bridge(grid))  # flip (5, 1), (6, 1), (7, 1) and (8, 1)

    def test_find_shortest_bridge(self):
        grid = [
            [1, 1, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
        self.assertEqual(1, shortest_bridge(grid))  # flip (3, 0)

    def test_performance(self):
        n, m = 15, 15
        grid = [[0 for _ in xrange(m)] for _ in xrange(n)]
        grid[0][0] = 1
        grid[n-1][m-1] = 1
        self.assertEqual(n+m-3, shortest_bridge(grid))


if __name__ == '__main__':
    unittest.main(exit=False)
```