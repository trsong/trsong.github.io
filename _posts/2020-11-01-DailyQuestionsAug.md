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
