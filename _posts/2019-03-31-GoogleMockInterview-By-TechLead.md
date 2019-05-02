---
layout: post
title:  "Mock Google Interview Question"
date:   2019-03-31 21:53:32 -0700
categories: Scala
---
* This will become a table of contents (this text will be scraped).
{:toc}

## Mock Google Interview Question
Randomly browsering youtube videos online, found an interesting video posted by my favorite youtuber: the [TechLead](https://www.youtube.com/channel/UC4xKdmAXFh4ACyhpiQ_3qBw).

### Question: Maximum Number of Connected Colors

> Question: Given a grid with cells in different colors, find the maximum number of same color cells that are connected. 
> 
> Note: two cells are connected if they are of the same color and adjacent to each other: left, right, top or bottom. 

 **To stay simple, we use integers to represent colors:**
 
 ```scala
// The following grid have max 4 connected colors. [color 3: (1, 2), (1, 3), (2, 1), (2, 2)]
 [
    [1, 1, 2, 2, 3], 
    [1, 2, 3, 3, 1],
    [2, 3, 3, 1, 2]
 ]
 ```

Source: [Mock Google interview (for Software Engineer job) - coding & algorithms tips](https://youtu.be/IWvbPIYQPFM?t=319). The question starts at 5'19''.

### My Thoughts

### Implementation
