---
layout: post
title:  "Interval Search Algorithm"
date:   2017-11-04 22:00:00 -0700
categories: Scala TypeScript
---
* This will become a table of contents (this text will be scraped).
{:toc}

## Interval Search Algorithm

### 1d Interval Search

Data structure to hold set of (Overlapping) intervals

- Insert an interval *(lo, hi)*
- Search for an interval *(lo, hi)*
- Delete an interval *(lo, hi)*
- Interval intersection query: given an interval *(lo, hi)*, find all intervals in data structure overlapping *(lo, hi)*

**Interface**

```scala
class IntervalST[Key <: Comparable[Key], Value] {
    def put(lo: Key, hi: Key, value: Value)
    def get(lo: Key, hi: Key): Value
    def delete(lo: Key, hi: Key)
    // For this implementation, suppose there's no two intervals w/ same two ends
    def intersects(lo: Key, hi: Key): Iterable[Value]
}
```

**Interval Search Tree**

Create BST, where each node stores an interval (lo, hi). 

- Use **left** endpoint as BST key

```
                 [17, 19]
                /        \
          [5, 8]          [21, 24]
         /      \         /       \
   [4, 8]       [15, 18]
  /      \     /        \
            [7, 10]
           /       \
```

- Store **max endpoint** in subtree rooted at node.

```
                 [17, 19] (24)
                /        \
          [5, 8] (18)     [21, 24] (24)
         /      \         /       \
   [4, 8] (8)   [15, 18] (18)
  /      \     /        \
            [7, 10] (10)
           /       \
```

Example: insert interval(16, 22)

Step 1, BSearch and insert (Top-down)

```
                 [17, 19] (24)
                /        \
          [5, 8] (18)     [21, 24] (24)
         /      \         /       \
   [4, 8] (8)   [15, 18] (18)
  /      \     /        \
            [7, 10] (10) [16, 22]
           /       \
```

Step 2, update max endpoint (Bottom-up)

```
                 [17, 19] (24)
                /        \
          [5, 8] (22)     [21, 24] (24)
         /      \         /       \
   [4, 8] (8)   [15, 18] (22)
  /      \     /        \
            [7, 10] (10) [16, 22] (22)
           /       \
```

**Search for an intersecting interval implementation**

To search for any one interval that intersects query interval *(lo, hi)*:

- If interval in node intersects query interval, return it.
- Else if left subtreee is null, go right.
- Else if max endpoint in left subtree is less than *lo*, go right.
- Else go left.

```scala
var x: Node = root
while (x != null) {
    if (x.inverval.intersects(lo, hi)) return x.interval
    else if (x.left == null) x = x.right
    else if (x.left.max < lo) x = x.right 
        // if goes right, there's no intersection on the left
    else x = x.left
        // if goes left, there's eiter an intersection in left subtree or no intersections either
}
return null
```

**Use Red-black Tree to ensure the performance of BST**
