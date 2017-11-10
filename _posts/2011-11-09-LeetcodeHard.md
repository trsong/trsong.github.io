---
layout: post
title:  "LeetCode Questions - Hard"
date:   2017-11-09 22:36:32 -0700
categories: Scala TypeScript
---
* This will become a table of contents (this text will be scraped).
{:toc}

## LeetCode Questions - Hard

### Environment Setup

TypeScript Playground: [https://www.typescriptlang.org/play/](https://www.typescriptlang.org/play/)

Scala Playground: [https://scastie.scala-lang.org/](https://scastie.scala-lang.org/)

Scala/Js/CodeSnippet Playground2: [https://leetcode.com/playground/new](https://leetcode.com/playground/new)

Covert Tabs to Spaces in Code Snippets: [http://tabstospaces.com/](http://tabstospaces.com/)


### 146. LRU Cache

Source: [https://leetcode.com/problems/lru-cache/description/](https://leetcode.com/problems/lru-cache/description/)

Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: `get` and `put`.

`get(key)` - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
`put(key, value)` - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

**Follow up:**

Could you do both operations in **O(1)** time complexity?

**Example:**

```java
LRUCache cache = new LRUCache( 2 /* capacity */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4
```

**Hint:**

1. The key to solve this problem is using a double linked list which enables us to quickly move nodes.
2. The LRU cache is a hash table of keys and double linked nodes. The hash table makes the time of get() to be O(1). The list of double linked nodes make the nodes adding/removal operations O(1).


**Scala Solution:**

```scala
import scala.collection.mutable.HashMap

case class Node(key: Int, var value: Int, var prev: Node = null, var next: Node = null)

class LRUCache(capacity: Int) {
    private val map = HashMap.empty[Int, Node]
    private var count = 0
    private var head = Node(0, 0)
    private var tail = Node(0, 0)
    head.next = tail
    tail.prev = head
    
    def get(key: Int): Int = {
        if (map.contains(key)) {
            val node = map(key)
            val result = node.value
            populate(node)
            result
        } else {
            -1
        }
    }
    
    def set(key: Int, value: Int): Unit = {
        if (map.contains(key)) {
            val node = map(key)
            node.value = value
            populate(node)
        } else {
            val newNode = Node(key, value)
            map(key) = newNode
            if (count < capacity) {
                count += 1
            } else {
                map -= tail.prev.key
                deleteNode(tail.prev)
            }
            
            addToHead(newNode)
        }
    }
    
    private def populate(node: Node): Unit = {
        deleteNode(node)
        addToHead(node)
    }
    
    private def deleteNode(node: Node): Unit = {
        node.prev.next = node.next
        node.next.prev = node.prev
    }
    
    // Insert node between head and head.next
    private def addToHead(node: Node): Unit = {
        // set up node <-> head.next
        node.next = head.next   // used to be null
        node.next.prev = node   // used to be head
        
        // set up head <-> node 
        node.prev = head        // used to be null
        head.next = node        // used to be head.next
    }
}

object Main extends App {
    val cache = new LRUCache(2)
    
    cache.set(1, 1)
    cache.set(2, 2)
    println(cache.get(1))      // returns 1
    cache.set(3, 3)            // evicts key 2
    println(cache.get(2))      // returns -1 (not found)
    cache.set(4, 4)            // evicts key 1
    println(cache.get(1))      // returns -1 (not found)
    println(cache.get(3))      // returns 3
    println(cache.get(4))      // returns 4
}
```
