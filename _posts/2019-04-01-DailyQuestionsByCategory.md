---
layout: post
title:  "Daily Coding Problems by Category"
date:   2019-04-01 22:22:32 -0700
categories: Python/Java
---
* This will become a table of contents (this text will be scraped).
{:toc}

<!--
<details>
<summary class="lc_h">

- [****]() -- ** [*\(Try ME\)*]()

</summary>
<div>


</div>
</details>
-->

{::options parse_block_html="true" /}


## Math
---

<details>
<summary class="lc_m">

- [**\[Medium\] Count Occurrence in Multiplication Table**](https://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-3-2021-medium-count-occurrence-in-multiplication-table) -- *Given integers N and X, write a function that returns the number of times X appears as a value in an N by N multiplication table.* [*\(Try ME\)*](https://repl.it/@trsong/Count-Occurrence-in-Multiplication-Table-1)

</summary>
<div>

**Question:**  Suppose you have a multiplication table that is `N` by `N`. That is, a 2D array where the value at the `i-th` row and `j-th` column is `(i + 1) * (j + 1)` (if 0-indexed) or `i * j` (if 1-indexed).

Given integers `N` and `X`, write a function that returns the number of times `X` appears as a value in an `N` by `N` multiplication table.

For example, given `N = 6` and `X = 12`, you should return `4`, since the multiplication table looks like this:

```py
| 1 |  2 |  3 |  4 |  5 |  6 |

| 2 |  4 |  6 |  8 | 10 | 12 |

| 3 |  6 |  9 | 12 | 15 | 18 |

| 4 |  8 | 12 | 16 | 20 | 24 |

| 5 | 10 | 15 | 20 | 25 | 30 |

| 6 | 12 | 18 | 24 | 30 | 36 |
```

And there are `4` 12's in the table.
 
</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Palindrome Integers**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-22-2021-easy-palindrome-integers) -- *Write a program that checks whether an integer is a palindrome.* [*\(Try ME\)*](https://repl.it/@trsong/Determine-Palindrome-Integer-1)

</summary>
<div>

**Question:** Write a program that checks whether an integer is a palindrome. For example, `121` is a palindrome, as well as `888`. But neither `678` nor `80` is a palindrome. Do not convert the integer into a string.

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] Regular Numbers**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-18-2021-medium-regular-numbers) -- *A regular number in mathematics is defined as one which evenly divides some power of 60.  Returns, in order, the first N regular numbers.* [*\(Try ME\)*](https://repl.it/@trsong/Regular-Numbers-1)

</summary>
<div>

**Question:**  A regular number in mathematics is defined as one which evenly divides some power of `60`. Equivalently, we can say that a regular number is one whose only prime divisors are `2`, `3`, and `5`.

These numbers have had many applications, from helping ancient Babylonians keep time to tuning instruments according to the diatonic scale.

Given an integer N, write a program that returns, in order, the first N regular numbers.


</div>
</details>


<details>

<summary class="lc_e">

- [**\[Easy\] LC 1344. Angle between Clock Hands**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-14-2021-lc-1344-easy-angle-between-clock-hands) -- *Given a clock time in `hh:mm` format, determine, to the nearest degree, the angle between the hour and the minute hands.* [*\(Try ME\)*](https://repl.it/@trsong/Calculate-Angle-between-Clock-Hands-1)

</summary>
<div>

**Question:** Given a clock time in `hh:mm` format, determine, to the nearest degree, the angle between the hour and the minute hands.

**Example:**
```py
Input: "9:00"
Output: 90
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Spreadsheet Columns**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-11-2021-easy-spreadsheet-columns) -- *Given a number n, find the n-th column name. From the 1st to the 26th column the letters are A to Z.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Spreadsheet-Columns-1)

</summary>
<div>

**Question:** In many spreadsheet applications, the columns are marked with letters. From the 1st to the 26th column the letters are A to Z. Then starting from the 27th column it uses AA, AB, ..., ZZ, AAA, etc.

Given a number n, find the n-th column name.

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

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Add Digits**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-26-2020-easy-add-digits) -- *Given a number, add the digits repeatedly until you get a single number.* [*\(Try ME\)*](https://repl.it/@trsong/Add-Digits-1)

</summary>
<div>

**Question:** Given a number, add the digits repeatedly until you get a single number.

**Example:**
```py
Input: 159
Output: 6
Explanation:
1 + 5 + 9 = 15.
1 + 5 = 6.
So the answer is 6.
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] LC 273. Integer to English Words**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-25-2020-lc-273-hard-integer-to-english-words) -- *Convert a non-negative integer to its English word representation.* [*\(Try ME\)*](https://repl.it/@trsong/Convert-Int-to-English-Words-1)

</summary>
<div>

**Question:** Convert a non-negative integer to its English word representation. 

**Example 1:**
```py
Input: 123
Output: "One Hundred Twenty Three"
```

**Example 2:**
```py
Input: 12345
Output: "Twelve Thousand Three Hundred Forty Five"
```

**Example 3:**
```py
Input: 1234567
Output: "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
```

**Example 4:**
```py
Input: 1234567891
Output: "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One"
```

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] LC 13. Convert Roman Numerals to Decimal**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-8-2020-lc-13-easy-convert-roman-numerals-to-decimal) -- *Given a Roman numeral, find the corresponding decimal value. Inputs will be between 1 and 3999.* [*\(Try ME\)*](https://repl.it/@trsong/Convert-Roman-Format-Number-1)

</summary>
<div>

**Question:** Given a Roman numeral, find the corresponding decimal value. Inputs will be between 1 and 3999.
 
**Note:** Numbers are strings of these symbols in descending order. In some cases, subtractive notation is used to avoid repeated characters. The rules are as follows:
1. I placed before V or X is one less, so 4 = IV (one less than 5), and 9 is IX (one less than 10)
2. X placed before L or C indicates ten less, so 40 is XL (10 less than 50) and 90 is XC (10 less than 100).
3. C placed before D or M indicates 100 less, so 400 is CD (100 less than 500), and 900 is CM (100 less than 1000).

**Example:**
```py
Input: IX
Output: 9

Input: VII
Output: 7

Input: MCMIV
Output: 1904

Roman numerals are based on the following symbols:
I     1
IV    4
V     5
IX    9 
X     10
XL    40
L     50
XC    90
C     100
CD    400
D     500
CM    900
M     1000
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 8. String to Integer (atoi)**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-6-2020-lc-8-medium-string-to-integer-atoi) -- *Given a string, convert it to an integer without using the builtin str function.* [*\(Try ME\)*](https://repl.it/@trsong/String-to-Integer-atoi-1)

</summary>
<div>

**Question:** Given a string, convert it to an integer without using the builtin str function. You are allowed to use ord to convert a character to ASCII code.

Consider all possible cases of an integer. In the case where the string is not a valid integer, return `0`.

**Example:**
```py
atoi('-105')  # -105
```

</div>
</details>


<details>
<summary class="lc_m">


- [**\[Medium\] LC 166. Fraction to Recurring Decimal**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-30-2020-lc-166-medium-fraction-to-recurring-decimal) -- *Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.* [*\(Try ME\)*](https://repl.it/@trsong/Convert-Fraction-to-Recurring-Decimal-1)

</summary>
<div>

**Question:** Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.

If the fractional part is repeating, enclose the repeating part in parentheses.

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

</div>
</details>



<details>
<summary class="lc_e">

- [**\[Easy\] Overlapping Rectangles**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-17-2020-medium-overlapping-rectangles) -- *Find the area of overlapping rectangles.* [*\(Try ME\)*](https://repl.it/@trsong/Overlapping-Rectangle-Areas-1)

</summary>
<div>

**Question:** You’re given 2 over-lapping rectangles on a plane. For each rectangle, you’re given its bottom-left and top-right points. How would you find the area of their overlap?

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Exists Overlap Rectangle**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-16-2021-easy-exists-overlap-rectangle) -- *Given a list of rectangles represented by min and max x- and y-coordinates. Compute whether or not a pair of rectangles overlap each other.* [*\(Try ME\)*](https://repl.it/@trsong/Exists-Overlap-Rectangles-1)

</summary>
<div>

**Question:** You are given a list of rectangles represented by min and max x- and y-coordinates. Compute whether or not a pair of rectangles overlap each other. If one rectangle completely covers another, it is considered overlapping.

For example, given the following rectangles:
```py
{
    "top_left": (1, 4),
    "dimensions": (3, 3) # width, height
},
{
    "top_left": (-1, 3),
    "dimensions": (2, 1)
},
{
    "top_left": (0, 5),
    "dimensions": (4, 3)
}
```
Return true as the first and third rectangle overlap each other.
 
</div>
</details>


### Puzzle
---

<details>
<summary class="lc_e">

- [**\[Easy\] LC 766. Toeplitz Matrix**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-31-2021-lc-766-easy-toeplitz-matrix) -- *Write a program to determine whether a given input is a Toeplitz matrix whose diagonal elements are the same.* [*\(Try ME\)*](https://repl.it/@trsong/Toeplitz-Matrix-1)

</summary>
<div>

**Question:** In linear algebra, a Toeplitz matrix is one in which the elements on any given diagonal from top left to bottom right are identical.

Write a program to determine whether a given input is a Toeplitz matrix.

**Example of Toeplitz Matrix:**
```py
1 2 3 4 8
5 1 2 3 4
4 5 1 2 3
7 4 5 1 2
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 89. Generate Gray Code**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-27-2021-lc-89-medium-generate-gray-code) -- *Gray code is a binary code where each successive value differ in only one bit* [*\(Try ME\)*](https://repl.it/@trsong/Generate-the-Gray-Code-1)

</summary>
<div>

**Question:**  Gray code is a binary code where each successive value differ in only one bit, as well as when wrapping around. Gray code is common in hardware so that we don't see temporary spurious values during transitions.

Given a number of bits n, generate a possible gray code for it.

**Example:**
```py
For n = 2, one gray code would be [00, 01, 11, 10].
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Minimum Cost to Construct Pyramid with Stones**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-25-2020-hard-minimum-cost-to-construct-pyramid-with-stones) -- *Given N stones in a row, you can change the height of any stone by paying a cost of 1 unit. Determine the lowest cost  or produce a pyramid.* [*\(Try ME\)*](https://repl.it/@trsong/Minimum-Cost-to-Construct-Pyramid-with-Stones-1)

</summary>
<div>

**Question:** You have `N` stones in a row, and would like to create from them a pyramid. This pyramid should be constructed such that the height of each stone increases by one until reaching the tallest stone, after which the heights decrease by one. In addition, the start and end stones of the pyramid should each be one stone high.

You can change the height of any stone by paying a cost of `1` unit to lower its height by `1`, as many times as necessary. Given this information, determine the lowest cost method to produce this pyramid.

For example, given the stones `[1, 1, 3, 3, 2, 1]`, the optimal solution is to pay `2` to create `[0, 1, 2, 3, 2, 1]`.

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Reconstruct a Jumbled Array**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-24-2020-easy-reconstruct-a-jumbled-array) -- *The sequence 0 to N is jumbled, and given its order is an array with each number is larger or smaller than the previous one. Return an array.* [*\(Try ME\)*](https://repl.it/@trsong/Reconstruct-a-Jumbled-Array-1)

</summary>
<div>

**Question:** The sequence `[0, 1, ..., N]` has been jumbled, and the only clue you have for its order is an array representing whether each number is larger or smaller than the last. 
 
Given this information, reconstruct an array that is consistent with it. For example, given `[None, +, +, -, +]`, you could return `[1, 2, 3, 0, 4]`.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Count Attacking Bishop Pairs**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-12-2020-medium-count-attacking-bishop-pairs) -- *Given N bishops, represented as (row, column) tuples on a M by M chessboard, count the number of pairs of bishops attacking each other.* [*\(Try ME\)*](https://repl.it/@trsong/Count-Number-of-Attacking-Bishop-Pairs-1)

</summary>
<div>

**Question:** On our special chessboard, two bishops attack each other if they share the same diagonal. This includes bishops that have another bishop located between them, i.e. bishops can attack through pieces.
 
You are given N bishops, represented as (row, column) tuples on a M by M chessboard. Write a function to count the number of pairs of bishops that attack each other. The ordering of the pair doesn't matter: `(1, 2)` is considered the same as `(2, 1)`.

For example, given `M = 5` and the list of bishops:

```py
(0, 0)
(1, 2)
(2, 2)
(4, 0)
```
The board would look like this:

```py
[b 0 0 0 0]
[0 0 b 0 0]
[0 0 b 0 0]
[0 0 0 0 0]
[b 0 0 0 0]
```

You should return 2, since bishops 1 and 3 attack each other, as well as bishops 3 and 4.

</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] Find Next Greater Permutation**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-2-2020-hard-find-next-greater-permutation) -- *Given a number represented by a list of digits, find the next greater permutation of a number, in terms of lexicographic ordering.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Next-Greater-Permutation-1)

</summary>
<div>

**Question:** Given a number represented by a list of digits, find the next greater permutation of a number, in terms of lexicographic ordering. If there is not greater permutation possible, return the permutation with the lowest value/ordering.

For example, the list `[1,2,3]` should return `[1,3,2]`. The list `[1,3,2]` should return `[2,1,3]`. The list `[3,2,1]` should return `[1,2,3]`.

Can you perform the operation without allocating extra memory (disregarding the input memory)?

</div>
</details>


### Bitwise Operations
---


<details>
<summary class="lc_m">

- [**\[Medium\] Bitwise AND of a Range**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#mar-2-2021-medium-bitwise-and-of-a-range) -- *Write a function that returns the bitwise AND of all integers between M and N, inclusive.* [*\(Try ME\)*](https://repl.it/@trsong/Calculate-Bitwise-AND-of-a-Range-1)

</summary>
<div>

**Question:** Write a function that returns the bitwise AND of all integers between M and N, inclusive.

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Valid UTF-8 Encoding**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-23-2021-easy-valid-utf-8-encoding) -- *Takes in an array of integers representing byte values, and returns whether it is a valid UTF-8 encoding.* [*\(Try ME\)*](https://repl.it/@trsong/Valid-UTF-8-Encoding-1)

</summary>
<div>

**Question:** UTF-8 is a character encoding that maps each symbol to one, two, three, or four bytes.

Write a program that takes in an array of integers representing byte values, and returns whether it is a valid UTF-8 encoding.
 
For example, the Euro sign, `€`, corresponds to the three bytes 11100010 10000010 10101100. The rules for mapping characters are as follows:

- For a single-byte character, the first bit must be zero.
- For an n-byte character, the first byte starts with n ones and a zero. The other n - 1 bytes all start with 10.

Visually, this can be represented as follows.

```py
 Bytes   |           Byte format
-----------------------------------------------
   1     | 0xxxxxxx
   2     | 110xxxxx 10xxxxxx
   3     | 1110xxxx 10xxxxxx 10xxxxxx
   4     | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
```   


</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Longest Consecutive 1s in Binary Representation**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-20-2020-easy-longest-consecutive-1s-in-binary-representation) -- *Given an integer n, return the length of the longest consecutive run of 1s in its binary representation.* [*\(Try ME\)*](https://repl.it/@trsong/Longest-Consecutive-1s-in-Binary-Representation-1)

</summary>
<div>

**Question:**  Given an integer n, return the length of the longest consecutive run of 1s in its binary representation.

**Example:**
```py
Input: 156
Output: 3
Exaplanation: 156 in binary is 10011100
```


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 29. Divide Two Integers**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-23-2020-lc-29-medium-divide-two-integers) -- *Implement integer division without using the division operator: take two numbers,  return a tuple of (dividend, remainder)* [*\(Try ME\)*](https://repl.it/@trsong/Divide-Two-Integers-1)

</summary>
<div>

**Question:** Implement integer division without using the division operator. Your function should return a tuple of `(dividend, remainder)` and it should take two numbers, the product and divisor.

For example, calling `divide(10, 3)` should return `(3, 1)` since the divisor is `3` and the remainder is `1`.

</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] Find the Element That Appears Once While Others Occur 3 Times**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-21-2020-hard-find-the-element-that-appears-once-while-others-occur-3-times) -- *Given an array of integers where every integer occurs three times except for one integer, which only occurs once, find that element.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Element-That-Appears-Once-While-Others-Occur-3-Ti-1)

</summary>
<div>

**Question:** Given an array of integers where every integer occurs three times except for one integer, which only occurs once, find and return the non-duplicated integer.

For example, given `[6, 1, 3, 3, 3, 6, 6]`, return `1`. Given `[13, 19, 13, 13]`, return `19`.

Do this in `O(N)` time and `O(1)` space.

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Find Next Sparse Number**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-11-2020-hard-find-next-sparse-number) -- *For a given input N, find the smallest sparse number greater than or equal to N.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Next-Spare-Number-1)

</summary>
<div>

**Question:** We say a number is sparse if there are no adjacent ones in its binary representation. For example, `21 (10101)` is sparse, but `22 (10110)` is not. 
 
For a given input `N`, find the smallest sparse number greater than or equal to `N`.

Do this in faster than `O(N log N)` time.
 
</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Find Next Biggest Integer**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-1-2020-medium-find-next-biggest-integer) -- *Given an integer n, find the next biggest integer with the same number of 1-bits on.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Next-Biggest-Integer-1)

</summary>
<div>

**Question:** Given an integer `n`, find the next biggest integer with the same number of 1-bits on. For example, given the number `6 (0110 in binary)`, return `9 (1001)`.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Find Two Elements Appear Once**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-27-2020-medium-find-two-elements-appear-once) -- *Given an array with two elements appear exactly once and all other elements appear exactly twice,find the two elements that appear only once* [*\(Try ME\)*](https://repl.it/@trsong/Find-Two-Elements-Appear-Once-1)

</summary>
<div>

**Question:** Given an array of integers in which two elements appear exactly once and all other elements appear exactly twice, find the two elements that appear only once. Can you do this in linear time and constant space?

**Example:**
```py
Input: [2, 4, 6, 8, 10, 2, 6, 10]
Output: [4, 8] order does not matter
 ```

</div>
</details>

### Hashing
---

<details>
<summary class="lc_e">

- [**\[Easy\] URL Shortener**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-28-2021-easy-url-shortener) -- *Implement a URL shortener with the following methods: shorten(url), restore(short).* [*\(Try ME\)*](https://repl.it/@trsong/URL-Shortener-1)

</summary>
<div>


**Question:** Implement a URL shortener with the following methods:

- `shorten(url)`, which shortens the url into a six-character alphanumeric string, such as `zLg6wl`.
- `restore(short)`, which expands the shortened string into the original url. If no such shortened string exists, return `null`.

**Follow-up:** What if we enter the same URL twice?

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 652. Find Duplicate Subtrees**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-30-2020-lc-652-medium-find-duplicate-subtrees) -- *Given a binary tree, find all duplicate subtrees (subtrees with the same value and same structure)* [*\(Try ME\)*](https://repl.it/@trsong/Find-Duplicate-Subtree-Nodes-1)

</summary>
<div>

**Question:** Given a binary tree, find all duplicate subtrees (subtrees with the same value and same structure) and return them as a list of list `[subtree1, subtree2, ...]` where `subtree1` is a duplicate of `subtree2` etc.

**Example1:**
```py
Given the following tree:
     1
    / \
   2   2
  /   /
 3   3

The duplicate subtrees are 
  2
 /    And  3
3
```

**Example2:**
```py
Given the following tree:
        1
       / \
      2   3
     /   / \
    4   2   4
       /
      4
      
The duplicate subtrees are 
      2
     /  And  4
    4
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 796. Shift-Equivalent Strings**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-25-2020-lc-796-easy-shift-equivalent-strings) -- *Given two strings A and B, return whether or not A can be shifted some number of times to get B.* [*\(Try ME\)*](https://repl.it/@trsong/Check-if-Strings-are-Shift-Equivalent-1)

</summary>
<div>

**Question:** Given two strings A and B, return whether or not A can be shifted some number of times to get B.

For example, if A is `'abcde'` and B is `'cdeab'`, return `True`. If A is `'abc'` and B is `'acb'`, return `False`.

</div>
</details>

## Array
---

<details>
<summary class="lc_m">

- [**\[Medium\] LC 227. Basic Calculator II**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-26-2021-lc-227-medium-basic-calculator-ii) -- *Implement a basic calculator to evaluate a simple expression string.* [*\(Try ME\)*](https://repl.it/@trsong/Implement-Basic-Calculator-II-1)

</summary>
<div>

**Question:** Implement a basic calculator to evaluate a simple expression string.

The expression string contains only non-negative integers, +, -, *, / operators and empty spaces. The integer division should truncate toward zero.

**Example 1:**
```py
Input: "3+2*2"
Output: 7
```

**Example 2:**
```py
Input: " 3/2 "
Output: 1
```

**Example 3:**
```py
Input: " 3+5 / 2 "
Output: 5
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Compare Version Numbers**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-9-2021-easy-compare-version-numbers) -- *Given two version numbers version1 and version2, conclude which is the latest version number.* [*\(Try ME\)*](https://repl.it/@trsong/Compare-Two-Version-Numbers-1)

</summary>
<div>

**Question:** Version numbers are strings that are used to identify unique states of software products. A version number is in the format a.b.c.d. and so on where a, b, etc. are numeric strings separated by dots. These generally represent a hierarchy from major to minor changes. 
 
Given two version numbers version1 and version2, conclude which is the latest version number. Your code should do the following:
- If version1 > version2 return 1.
- If version1 < version2 return -1.
- Otherwise return 0.

Note that the numeric strings such as a, b, c, d, etc. may have leading zeroes, and that the version strings do not start or end with dots. Unspecified level revision numbers default to 0.

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

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 554. Brick Wall**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-14-2020-lc-554-medium-brick-wall) -- *Find a vertical line going from the top to the bottom of the wall that cuts through the fewest* [*\(Try ME\)*](https://repl.it/@trsong/Min-Cut-of-Wall-Bricks-1)

</summary>
<div>

**Question:** A wall consists of several rows of bricks of various integer lengths and uniform height. Your goal is to find a vertical line going from the top to the bottom of the wall that cuts through the fewest number of bricks. If the line goes through the edge between two bricks, this does not count as a cut.

For example, suppose the input is as follows, where values in each row represent the lengths of bricks in that row:

```py
[[3, 5, 1, 1],
 [2, 3, 3, 2],
 [5, 5],
 [4, 4, 2],
 [1, 3, 3, 3],
 [1, 1, 6, 1, 1]]
 
```

The best we can we do here is to draw a line after the eighth brick, which will only require cutting through the bricks in the third and fifth row.

Given an input consisting of brick lengths for each row such as the one above, return the fewest number of bricks that must be cut to create a vertical line.


</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Distribute Bonuses**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-10-2020-easy-distribute-bonuses) -- *Give the smallest positive amount to each worker and if a developer has written more lines of code than their neighbors that dev earns more.* [*\(Try ME\)*](https://repl.it/@trsong/Distribute-Bonuses-1)

</summary>
<div>

**Question:** MegaCorp wants to give bonuses to its employees based on how many lines of codes they have written. They would like to give the smallest positive amount to each worker consistent with the constraint that if a developer has written more lines of code than their neighbor, they should receive more money.

Given an array representing a line of seats of employees at MegaCorp, determine how much each one should get paid.

**Example:**
```py
Input: [10, 40, 200, 1000, 60, 30]
Output: [1, 2, 3, 4, 2, 1].
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] Find Missing Positive**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-13-2020-medium-find-missing-positive) -- *Given an unsorted integer array, find the first missing positive integer.* [*\(Try ME\)*](https://repl.it/@trsong/Find-First-Missing-Positive-1)

</summary>
<div>

**Question:** Given an unsorted integer array, find the first missing positive integer.

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

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 121. Best Time to Buy and Sell Stock**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-3-2020-lc-121-easy-best-time-to-buy-and-sell-stock) -- *Given an array of numbers representing the stock prices of a company, write a function that calculates the maximum profit.* [*\(Try ME\)*](https://repl.it/@trsong/Best-Time-to-Buy-and-Sell-Stock-1)

</summary>
<div>

**Question:** You are given an array. Each element represents the price of a stock on that particular day. Calculate and return the maximum profit you can make from buying and selling that stock only once.

**Example:**
```py
Input: [9, 11, 8, 5, 7, 10]
Output: 5
Explanation: Here, the optimal trade is to buy when the price is 5, and sell when it is 10, so the return value should be 5 (profit = 10 - 5 = 5).
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Exclusive Product**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-28-2020-hard-exclusive-product) -- *Given an array of integers, return a new array such that each element at index i of the new array is the product of all the numbers except i.* [*\(Try ME\)*](https://repl.it/@trsong/Calculate-Exclusive-Product-1)

</summary>
<div>

**Question:** Given an array of integers, return a new array such that each element at index i of the new array is the product of all the numbers in the original array except the one at i.

For example, if our input was `[1, 2, 3, 4, 5]`, the expected output would be `[120, 60, 40, 30, 24]`. If our input was `[3, 2, 1]`, the expected output would be `[2, 3, 6]`.

Follow-up: what if you can't use division?

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Delete Columns to Make Sorted I**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-14-2020-easy-delete-columns-to-make-sorted-i) -- *Given a 2D matrix of lowercase letters. Determine the minimum number of columns that can be removed to ensure that each column is sorted.* [*\(Try ME\)*](https://repl.it/@trsong/Delete-Columns-From-Table-to-Make-Sorted-I-1)

</summary>
<div>

**Question:** You are given an N by M 2D matrix of lowercase letters. Determine the minimum number of columns that can be removed to ensure that each row is ordered from top to bottom lexicographically. That is, the letter at each column is lexicographically later as you go down each row. It does not matter whether each row itself is ordered lexicographically.

**Example 1:**
```py
Given the following table:
cba
daf
ghi

This is not ordered because of the a in the center. We can remove the second column to make it ordered:
ca
df
gi

So your function should return 1, since we only needed to remove 1 column.
```

**Example 2:**
```py
Given the following table:
abcdef

Your function should return 0, since the rows are already ordered (there's only one row).
```

**Example 3:**
```py
Given the following table:
zyx
wvu
tsr

Your function should return 3, since we would need to remove all the columns to order it.
```


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 665. Off-by-One Non-Decreasing Array**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-9-2020-lc-665-medium-off-by-one-non-decreasing-array) -- *Given an array of integers, write a function to determine whether the array could become non-decreasing by modifying at most 1 element.* [*\(Try ME\)*](https://repl.it/@trsong/Determine-if-Off-by-One-Non-Decreasing-Array-1)

</summary>
<div>

**Question:** Given an array of integers, write a function to determine whether the array could become non-decreasing by modifying at most 1 element.

For example, given the array `[10, 5, 7]`, you should return true, since we can modify the `10` into a `1` to make the array non-decreasing.

Given the array `[10, 5, 1]`, you should return false, since we can't modify any one element to get a non-decreasing array.

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Witness of The Tall People**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-30-2020-easy-witness-of-the-tall-people) -- *There are n people lined up,  a murder has happened right in front of them. How many witnesses are there?* [*\(Try ME\)*](https://repl.it/@trsong/Witness-of-The-Tall-People-1)

</summary>
<div>

**Question:** There are `n` people lined up, and each have a height represented as an integer. A murder has happened right in front of them, and only people who are taller than everyone in front of them are able to see what has happened. How many witnesses are there?

**Example:**
```py
Input: [3, 6, 3, 4, 1]  
Output: 3
Explanation: Only [6, 4, 1] were able to see in front of them.
 #
 #
 # #
####
####
#####
36341  
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 54. Spiral Matrix**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-5-2020-lc-54-medium-spiral-matrix) -- *Given a matrix of n x m elements (n rows, m columns), return all elements of the matrix in spiral order.* [*\(Try ME\)*](https://repl.it/@trsong/Spiral-Matrix-Traversal-1)

</summary>
<div>

**Question:** Given a matrix of n x m elements (n rows, m columns), return all elements of the matrix in spiral order.

**Example 1:**

```py
Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1, 2, 3, 6, 9, 8, 7, 4, 5]
```

**Example 2:**

```py
Input:
[
  [1,  2,  3,  4],
  [5,  6,  7,  8],
  [9, 10, 11, 12]
]
Output: [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
```

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Record the Last N Orders**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-15-2020-easy-record-the-last-n-orders) -- *You run an e-commerce website and want to record the last N order ids in a log.* [*\(Try ME\)*](https://repl.it/@trsong/Record-the-Last-N-Orders-1)

</summary>
<div>

**Question:** You run an e-commerce website and want to record the last `N` order ids in a log. Implement a data structure to accomplish this, with the following API:

- `record(order_id)`: adds the order_id to the log
- `get_last(i)`: gets the ith last element from the log. `i` is guaranteed to be smaller than or equal to `N`.
 
You should be as efficient with time and space as possible.


</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] In-place Array Rotation**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-3-2020-medium-in-place-array-rotation) -- *Write a function that rotates an array by k elements.* [*\(Try ME\)*](https://repl.it/@trsong/Rotate-Array-In-place-1)

</summary>
<div>

Write a function that rotates an array by `k` elements.

For example, `[1, 2, 3, 4, 5, 6]` rotated by two becomes [`3, 4, 5, 6, 1, 2]`.

Try solving this without creating a copy of the array. How many swap or move operations do you need?

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Longest Subarray with Sum Divisible by K**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-11-2020-medium-longest-subarray-with-sum-divisible-by-k) -- *Find the length of the longest subarray with sum of the elements divisible by the given value k.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Longest-Subarray-with-Sum-Divisible-by-K-1)

</summary>
<div>

**Question:** Given an arr[] containing n integers and a positive integer k. The problem is to find the length of the longest subarray with sum of the elements divisible by the given value k.

**Example:**
```py
Input : arr[] = {2, 7, 6, 1, 4, 5}, k = 3
Output : 4
The subarray is {7, 6, 1, 4} with sum 18, which is divisible by 3.
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 152. Maximum Product Subarray**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-14-2020-lc-152-medium-maximum-product-subarray) -- *Given an array that contains both positive and negative integers, find the product of the maximum product subarray.* [*\(Try ME\)*](https://repl.it/@trsong/Maximum-Product-Subarray-1)

</summary>
<div>

**Question:** Given an array that contains both positive and negative integers, find the product of the maximum product subarray. 

**Example 1:**
```py
Input: [6, -3, -10, 0, 2]
Output:  180
Explanation: The subarray is [6, -3, -10]
```

**Example 2:**
```py
Input: [-1, -3, -10, 0, 60]
Output:   60 
Explanation: The subarray is [60]
```

**Example 3:**
```py
Input: [-2, -3, 0, -2, -40]
Output: 80
Explanation: The subarray is [-2, -40]
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] Majority Element**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-28-2020-medium-majority-element) -- *A majority element is an element that appears more than half the time. Given a list with a majority element, find the majority element.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Majority-Element-1)

</summary>
<div>

**Question:** A majority element is an element that appears more than half the time. Given a list with a majority element, find the majority element.

**Example:**
```py
majority_element([3, 5, 3, 3, 2, 4, 3])  # gives 3
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Valid Mountain Array**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-6-2020-easy-valid-mountain-array) -- *Given an array of heights, determine whether the array forms a “mountain” pattern. A mountain pattern goes up and then down.* [*\(Try ME\)*](https://repl.it/@trsong/Valid-Mountain-Array-1)

</summary>
<div>

**Question:** Given an array of heights, determine whether the array forms a "mountain" pattern. A mountain pattern goes up and then down.

**Example:**
```py
validMountainArray([1, 2, 3, 2, 1])  # True
validMountainArray([1, 2, 3])  # False
```

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Matrix Rotation**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-7-2020-medium-matrix-rotation) -- *Given an N by N matrix, rotate it by 90 degrees clockwise.* [*\(Try ME\)*](https://repl.it/@trsong/Matrix-Rotation-In-place-with-flip-1)

</summary>
<div>

**Question:** Given an N by N matrix, rotate it by 90 degrees clockwise.

For example, given the following matrix:

```py
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
 ```

you should return:

```py
[[7, 4, 1],
 [8, 5, 2],
 [9, 6, 3]]
 ```

Follow-up: What if you couldn't use any extra space?
 
</div>
</details>

### Interval
---

<details>
<summary class="lc_h">

- [**\[Hard\] LC 352. Data Stream as Disjoint Intervals**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-17-2020-lc-352-hard-data-stream-as-disjoint-intervals) -- *Given a data stream input of non-negative integers a1, a2, ..., an, ..., summarize the numbers seen so far as a list of disjoint intervals.* [*\(Try ME\)*](https://repl.it/@trsong/Print-Data-Stream-as-Disjoint-Intervals-1)

</summary>
<div>

**Question:** Given a data stream input of non-negative integers `a1, a2, ..., an, ...`, summarize the numbers seen so far as a list of disjoint intervals.

For example, suppose the integers from the data stream are `1, 3, 7, 2, 6, ...`, then the summary will be:

```py
[1, 1]
[1, 1], [3, 3]
[1, 1], [3, 3], [7, 7]
[1, 3], [7, 7]
[1, 3], [6, 7]
```

</div>
</details>



<details>
<summary class="lc_e">

- [**\[Easy\] LC 228. Extract Range**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-30-2020-lc-228-easy-extract-range) -- *Given a sorted list of numbers, return a list of strings that represent all of the consecutive numbers.* [*\(Try ME\)*](https://repl.it/@trsong/Extract-Range-1)

</summary>
<div>

**Question:** Given a sorted list of numbers, return a list of strings that represent all of the consecutive numbers.

**Example:**
```py
Input: [0, 1, 2, 5, 7, 8, 9, 9, 10, 11, 15]
Output: ['0->2', '5', '7->11', '15']
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Merge Overlapping Intervals**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-18-2020-easy-merge-overlapping-intervals) -- *Given a list of possibly overlapping intervals, return a new list of intervals where all overlapping intervals have been merged.* [*\(Try ME\)*](https://repl.it/@trsong/Merge-All-Overlapping-Intervals-1)

</summary>
<div>

**Question:** Given a list of possibly overlapping intervals, return a new list of intervals where all overlapping intervals have been merged.

The input list is not necessarily ordered in any way.

For example, given `[(1, 3), (5, 8), (4, 10), (20, 25)]`, you should return `[(1, 3), (4, 10), (20, 25)]`.

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] LC 253. Minimum Lecture Rooms**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-27-2020-lc-253-easy-minimum-lecture-rooms) -- *Given an array of time intervals (start, end) for classroom lectures (possibly overlapping), find the minimum number of rooms required.* [*\(Try ME\)*](https://repl.it/@trsong/Minimum-Required-Lecture-Rooms-1)

</summary>
<div>

**Questions:** Given an array of time intervals `(start, end)` for classroom lectures (possibly overlapping), find the minimum number of rooms required.

For example, given `[(30, 75), (0, 50), (60, 150)]`, you should return `2`.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 986. Interval List Intersections**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-15-2020-lc-986-medium-interval-list-intersections) -- *Given two lists of closed intervals, each list of intervals is pairwise disjoint and in sorted order.* [*\(Try ME\)*](https://repl.it/@trsong/Interval-List-Intersections-1)

</summary>
<div>

**Question:** Given two lists of closed intervals, each list of intervals is pairwise disjoint and in sorted order.

Return the intersection of these two interval lists.

**Example:**
```py
Input: A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
```

</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] LC 57. Insert Interval**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-14-2020-lc-57-hard-insert-interval) -- *Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).* [*\(Try ME\)*](https://repl.it/@trsong/Insert-Interval-1)

</summary>
<div>

**Question:** Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

**Example 1:**
```py
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
```

**Example 2:**
```py
Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
```

</div>
</details>

### Binary Search
---

<details>
<summary class="lc_h">

- [**\[Hard\] Maximize the Minimum of Subarray Sum**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#mar-3-2021-hard-maximize-the-minimum-of-subarray-sum) -- *Given an array of numbers N and integer k, split N into k partitions such that the maximize sum of any partition is minimized.* [*\(Try ME\)*](https://repl.it/@trsong/Maximize-the-Minimum-of-Subarray-Sum-1)

</summary>
<div>

**Question:** Given an array of numbers `N` and an integer `k`, your task is to split `N` into `k` partitions such that the maximum sum of any partition is minimized. Return this sum.

For example, given `N = [5, 1, 2, 7, 3, 4]` and `k = 3`, you should return `8`, since the optimal partition is `[5, 1, 2], [7], [3, 4]`.


</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Fixed Point**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-15-2021-easy-fixed-point) -- *Given a sorted array of distinct elements, return a fixed point, if one exists. Otherwise, return False.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Fixed-Point-1)

</summary>
<div>

**Questions:** A fixed point in an array is an element whose value is equal to its index. Given a sorted array of distinct elements, return a fixed point, if one exists. Otherwise, return `False`.

For example, given `[-6, 0, 2, 40]`, you should return `2`. Given `[1, 5, 7, 8]`, you should return `False`.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Find Minimum Element in a Sorted and Rotated Array**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-31-2020-medium-find-minimum-element-in-a-sorted-and-rotated-array) -- *Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand. Find the minimum element.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Minimum-Element-in-a-Sorted-and-Rotated-Array-1)

</summary>
<div>

**Question:** Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand. Find the minimum element in `O(log N)` time. You may assume the array does not contain duplicates.

For example, given `[5, 7, 10, 3, 4]`, return `3`.


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Searching in Rotated Array**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-12-2020-medium-searching-in-rotated-array) -- *A sorted array of integers was rotated an unknown number of times. Find the index of the element in the array.* [*\(Try ME\)*](https://repl.it/@trsong/Searching-Elem-in-Rotated-Array-1)

</summary>
<div>

**Question:** A sorted array of integers was rotated an unknown number of times. Given such an array, find the index of the element in the array in faster than linear time. If the element doesn't exist in the array, return null.
 
 For example, given the array `[13, 18, 25, 2, 8, 10]` and the element 8, return 4 (the index of 8 in the array).
 
 You can assume all the integers in the array are unique.
 
</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] LC 668. Kth Smallest Number in Multiplication Table**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-11-2020-lc-668-hard-kth-smallest-number-in-multiplication-table) -- *Find out the k-th smallest number quickly from the multiplication table.* [*\(Try ME\)*](https://repl.it/@trsong/Kth-Smallest-Number-in-Multiplication-Table-1)

</summary>
<div>

**Question:** Find out the k-th smallest number quickly from the multiplication table.

Given the height m and the length n of a m * n Multiplication Table, and a positive integer k, you need to return the k-th smallest number in this table.

**Example 1:**
```py
Input: m = 3, n = 3, k = 5
Output: 
Explanation: 
The Multiplication Table:
1	2	3
2	4	6
3	6	9
The 5-th smallest number is 3 (1, 2, 2, 3, 3).
```

**Example 2:**
```py
Input: m = 2, n = 3, k = 6
Output: 
Explanation: 
The Multiplication Table:
1	2	3
2	4	6
The 6-th smallest number is 6 (1, 2, 2, 3, 4, 6).
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] First and Last Indices of an Element in a Sorted Array**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-6-2020-easy-first-and-last-indices-of-an-element-in-a-sorted-array) -- *Given a sorted array, A, with possibly duplicated elements, find the indices of the first and last occurrences of a target element.* [*\(Try ME\)*](https://repl.it/@trsong/Find-First-and-Last-Indices-of-an-Element-in-a-Sorted-Arr-1)

</summary>
<div>

**Question:** Given a sorted array, A, with possibly duplicated elements, find the indices of the first and last occurrences of a target element, x. Return -1 if the target is not found.

**Examples:**
```py
Input: A = [1, 3, 3, 5, 7, 8, 9, 9, 9, 15], target = 9
Output: [6, 8]

Input: A = [100, 150, 150, 153], target = 150
Output: [1, 2]

Input: A = [1, 2, 3, 4, 5, 6, 10], target = 9
Output: [-1, -1]
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] LC 300. The Longest Increasing Subsequence**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-19-2020-lc-300-hard-the-longest-increasing-subsequence) -- *Given an array of numbers, find the length of the longest increasing subsequence in the array.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Longest-Increasing-Subsequence-1)

</summary>
<div>

**Question:** Given an array of numbers, find the length of the longest increasing **subsequence** in the array. The subsequence does not necessarily have to be contiguous.

For example, given the array `[0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]`, the longest increasing subsequence has length `6` ie. `[0, 2, 6, 9, 11, 15]`.

</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] Increasing Subsequence of Length K**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-20-2020-hard-increasing-subsequence-of-length-k) -- *Given an int array nums of length n and an int k. Return an increasing subsequence of length k (KIS).* [*\(Try ME\)*](https://repl.it/@trsong/Increasing-Subsequence-of-Length-K-1)

</summary>
<div>

**Question:** Given an int array nums of length n and an int k. Return an increasing subsequence of length k (KIS). Expected time complexity `O(nlogk)`.

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

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 162. Find a Peak Element**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-26-2020-lc-162-easy-find-a-peak-element) -- *Given an unsorted array, in which all elements are distinct, find a “peak” element in O(log N) time.* [*\(Try ME\)*](https://repl.it/@trsong/Find-a-Peak-Element-2)

</summary>
<div>

**Question:** Given an unsorted array, in which all elements are distinct, find a "peak" element in `O(log N)` time.

An element is considered a peak if it is greater than both its left and right neighbors. It is guaranteed that the first and last elements are lower than all others.

**Example 1:**
```py
Input: [5, 10, 20, 15]
Output: 20
The element 20 has neighbours 10 and 15,
both of them are less than 20.
```

**Example 2:**
```py
Input: [10, 20, 15, 2, 23, 90, 67]
Output: 20 or 90
The element 20 has neighbours 10 and 15, 
both of them are less than 20, similarly 90 has neighbous 23 and 67.
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 163. Missing Ranges**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-29-2020-lc-163-medium-missing-ranges) -- *Given a sorted list of numbers, and two integers low and high, return a list of (inclusive) ranges where the numbers are missing.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Missing-Ranges-1)

</summary>
<div>

**Question:** Given a sorted list of numbers, and two integers low and high representing the lower and upper bound of a range, return a list of (inclusive) ranges where the numbers are missing. A range should be represented by a tuple in the format of (lower, upper).

**Example:**
```py
missing_ranges(nums=[1, 3, 5, 10], lower=1, upper=10)
# returns [(2, 2), (4, 4), (6, 9)]
```

</div>
</details>


### Two Pointers
---

<details>
<summary class="lc_e">

- [**\[Easy\] Minimum Distance between Two Words**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-24-2021-easy-minimum-distance-between-two-words) -- *Find an efficient algorithm to find the smallest distance (measured in number of words) between any two given words in a string.* [*\(Try ME\)*](https://repl.it/@trsong/Minimum-Distance-between-Two-Words-1)

</summary>
<div>

**Question:** Find an efficient algorithm to find the smallest distance (measured in number of words) between any two given words in a string.

For example, given words `"hello"`, and `"world"` and a text content of `"dog cat hello cat dog dog hello cat world"`, return `1` because there's only one word `"cat"` in between the two words.

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Count Elements in Sorted Matrix**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-17-2021-hard-count-elements-in-sorted-matrix) -- *Let A be a sorted matrix. Given i1, j1, i2, and j2, compute the number of elements smaller than A[i1, j1] and larger than A[i2, j2].* [*\(Try ME\)*](https://repl.it/@trsong/Count-Elements-in-Sorted-Matrix-1)

</summary>
<div>

**Question:** Let A be an `N` by `M` matrix in which every row and every column is sorted.

Given `i1`, `j1`, `i2`, and `j2`, compute the number of elements smaller than `A[i1, j1]` and larger than `A[i2, j2]`.

**Example:**
```py
Given the following matrix:
[[ 1,  3,  6, 10, 15, 20],
 [ 2,  7,  9, 14, 22, 25],
 [ 3,  8, 10, 15, 25, 30],
 [10, 11, 12, 23, 30, 35],
 [20, 25, 30, 35, 40, 45]]
And i1 = 1, j1 = 1, i2 = 3, j2 = 3
 
return 15 as there are 15 numbers in the matrix smaller than 7 or greater than 23.
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 859. Buddy Strings**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-11-2020-lc-859-easy-buddy-strings) -- *Given two string A, B. Return true if and only if we can swap two letters in A so that the result equals B.* [*\(Try ME\)*](https://repl.it/@trsong/Buddy-Strings-1)

</summary>
<div>

**Question:** Given two strings A and B of lowercase letters, return true if and only if we can swap two letters in A so that the result equals B.

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


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Find Pythagorean Triplets**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-22-2020-medium-find-pythagorean-triplets) -- *Given a list of numbers, find if there exists a Pythagorean triplet in that list. * [*\(Try ME\)*](https://repl.it/@trsong/Find-Pythagorean-Triplets-1)

</summary>
<div>

**Question:** Given a list of numbers, find if there exists a pythagorean triplet in that list. A pythagorean triplet is `3` variables `a`, `b`, `c` where `a * a + b * b = c * c`.

**Example:**
```py
Input: [3, 5, 12, 5, 13]
Output: True
Here, 5 * 5 + 12 * 12 = 13 * 13.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LT 640. One Edit Distance**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-21-2020-lt-640-medium-one-edit-distance) -- *Given two strings S and T, determine if they are both one edit distance apart.* [*\(Try ME\)*](https://repl.it/@trsong/Is-One-Edit-Distance-1)

</summary>
<div>

**Question:** Given two strings S and T, determine if they are both one edit distance apart.

**Example 1:**
```
Input: s = "aDb", t = "adb" 
Output: True
```

**Example 2:**
```
Input: s = "ab", t = "ab" 
Output: False
Explanation:
s=t, so they aren't one edit distance apart
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Array of Equal Parts**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-5-2020-easy-array-of-equal-parts) -- *Given an array with only positive integers, determine if exist two numbers such that removing them can break array into 3 equal-sum sub-arrays.* [*\(Try ME\)*](https://repl.it/@trsong/Array-of-Equal-Parts-1)

</summary>
<div>

**Question:** Given an array containing only positive integers, return if you can pick two integers from the array which cuts the array into three pieces such that the sum of elements in all pieces is equal.

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

</div>
</details>


### Sliding Window
---


<details>
<summary class="lc_m">

- [**\[Medium\] LC 438. Anagram Indices Problem**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-13-2020-lc-438-medium-anagram-indices-problem) -- *Given a word W and a string S, find all starting indices in S which are anagrams of W.* [*\(Try ME\)*](https://repl.it/@trsong/Find-All-Anagram-Indices-1)

</summary>
<div>

**Question:**  Given a word W and a string S, find all starting indices in S which are anagrams of W.

For example, given that W is `"ab"`, and S is `"abxaba"`, return `0`, `3`, and `4`.
 
</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 239. Sliding Window Maximum**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-14-2020-lc-239-medium-sliding-window-maximum) -- *Given an array nums, return the max sliding window.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Sliding-Window-Maximum-1)

</summary>
<div>

**Question:** Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.
 

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

</div>
</details>



<details>
<summary class="lc_m">

- [**\[Medium\] Substrings with Exactly K Distinct Characters**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-17-2021-medium-substrings-with-exactly-k-distinct-characters) -- *Given a string s and an int k, return an int representing the number of substrings (not unique) of s with exactly k distinct characters.* [*\(Try ME\)*](https://repl.it/@trsong/Substrings-with-Exactly-K-Distinct-Characters-1)

</summary>
<div>

**Question:** Given a string `s` and an int `k`, return an int representing the number of substrings (not unique) of `s` with exactly `k` distinct characters. 

If the given string doesn't have `k` distinct characters, return `0`.

**Example 1:**
```py
Input: s = "pqpqs", k = 2
Output: 7
Explanation: ["pq", "pqp", "pqpq", "qp", "qpq", "pq", "qs"]
```

**Example 2:**
```py
Input: s = "aabab", k = 3
Output: 0
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LT 386. Longest Substring with At Most K Distinct Characters**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-16-2020-lt-386-medium-longest-substring-with-at-most-k-distinct-characters) -- *Given a string, find the longest substring that contains at most k unique characters.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Longest-Substring-with-At-Most-K-Distinct-Characters-1)

</summary>
<div>

**Question:** Given a string, find the longest substring that contains at most k unique characters. 
 
For example, given `"abcbbbbcccbdddadacb"`, the longest substring that contains 2 unique character is `"bcbbbbcccb"`.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Longest Substring without Repeating Characters**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-17-2020-medium-longest-substring-without-repeating-characters) -- *Given a string, find the length of the longest substring without repeating characters.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Longest-Substring-without-Repeating-Characters-1)

</summary>
<div>

**Question:** Given a string, find the length of the longest substring without repeating characters.

**Note:** Can you find a solution in linear time?
 
**Example:**
```py
lengthOfLongestSubstring("abrkaabcdefghijjxxx") # => 10 as len("abcdefghij") == 10
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 209. Minimum Size Subarray Sum**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-23-2020-lc-209-medium-minimum-size-subarray-sum) -- *Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum >= s.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Minimum-Size-Subarray-Sum-1)

</summary>
<div>

**Question:** Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum >= s. If there isn't one, return 0 instead.

**Example:**
```py
Input: s = 7, nums = [2, 3, 1, 2, 4, 3]
Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.
```

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] LC 383. Ransom Note**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-10-2020-lc-383-easy-ransom-note) -- *Given a magazine of letters and the note he wants to write, determine whether he can construct the word.* [*\(Try ME\)*](https://repl.it/@trsong/Ransom-Note-1)

</summary>
<div>

**Question:** A criminal is constructing a ransom note. In order to disguise his handwriting, he is cutting out letters from a magazine.

Given a magazine of letters and the note he wants to write, determine whether he can construct the word.

**Example 1:**
```py
Input: ['a', 'b', 'c', 'd', 'e', 'f'], 'bed'
Output: True
```

**Example 2:**
```py
Input: ['a', 'b', 'c', 'd', 'e', 'f'], 'cat'
Output: False
```

</div>
</details>

 
## List
---

### Singly Linked List
---

<details>
<summary class="lc_m">

- [**\[Medium\] LC 138. Deepcopy List with Random Pointer**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-12-2021-lc-138-medium-deepcopy-list-with-random-pointer) -- *A linked list is given such that each node contains an additional random pointer points to any node in the list or null. Deepcopy that list.* [*\(Try ME\)*](https://repl.it/@trsong/Create-Deepcopy-List-with-Random-1)

</summary>
<div>

**Question:** A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.

Return a deep copy of the list.

**Example:**
```py
Input:
{"$id":"1","next":{"$id":"2","next":null,"random":{"$ref":"2"},"val":2},"random":{"$ref":"2"},"val":1}

Explanation:
Node 1's value is 1, both of its next and random pointer points to Node 2.
Node 2's value is 2, its next pointer points to null and its random pointer points to itself.
```

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Reverse a Linked List**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-8-2020-easy-reverse-a-linked-list) -- *Given a singly-linked list, reverse the list. This can be done iteratively or recursively. Can you get both solutions?* [*\(Try ME\)*](https://repl.it/@trsong/Reverse-a-Linked-List-1)

</summary>
<div>

**Question:** Given a singly-linked list, reverse the list. This can be done iteratively or recursively. Can you get both solutions?

**Example:**
```py
Input: 4 -> 3 -> 2 -> 1 -> 0 -> NULL
Output: 0 -> 1 -> 2 -> 3 -> 4 -> NULL
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Add Two Numbers as a Linked List**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-4-2020--easy-add-two-numbers-as-a-linked-list) -- *Given two linked-lists representing two non-negative integers. Add the two numbers and return it as a linked list.* [*\(Try ME\)*](https://repl.it/@trsong/Add-Two-Numbers-and-Return-as-a-Linked-List-1)

</summary>
<div>

**Question:** You are given two linked-lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

**Example:**
```py
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Remove Duplicates From Linked List**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-10-2020-easy-remove-duplicates-from-linked-list) -- *Given a linked list, remove all duplicate values from the linked list.* [*\(Try ME\)*](https://repl.it/@trsong/Remove-Duplicates-From-Linked-List-1)

</summary>
<div>

**Question:** Given a linked list, remove all duplicate values from the linked list.

For instance, given `1 -> 2 -> 3 -> 3 -> 4`, then we wish to return the linked list `1 -> 2 -> 4`.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 86. Partitioning Linked List**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-28-2020--lc-86-medium-partitioning-linked-list) -- *Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.* [*\(Try ME\)*](https://repl.it/@trsong/Partitioning-Singly-Linked-List-1)

</summary>
<div>

**Question:** Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.

You should preserve the original relative order of the nodes in each of the two partitions.

**Example:**
```py
Input: head = 1->4->3->2->5->2, x = 3
Output: 1->2->2->4->3->5
```


</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Zig-Zag Distinct LinkedList**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-16-2020-easy-zig-zag-distinct-linkedlist) -- *Rearrange distinct linked-list such that they appear in alternating order.* [*\(Try ME\)*](https://repl.it/@trsong/Zig-Zag-Order-of-Distinct-LinkedList-1)

</summary>
<div>

**Question:** Given a linked list with DISTINCT value, rearrange the node values such that they appear in alternating `low -> high -> low -> high ...` form. For example, given `1 -> 2 -> 3 -> 4 -> 5`, you should return `1 -> 3 -> 2 -> 5 -> 4`.

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Remove K-th Last Element from Singly Linked-list**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#may-31-2020-medium-remove-k-th-last-element-from-singly-linked-list) -- *Given a singly linked list and an integer k, remove the kth last element from the list.* [*\(Try ME\)*](https://repl.it/@trsong/Remove-the-K-th-Last-Element-from-Singly-Linked-list-1)

</summary>
<div>

**Question:** Given a singly linked list and an integer k, remove the kth last element from the list. k is guaranteed to be smaller than the length of the list.

**Note:**
- The list is very long, so making more than one pass is prohibitively expensive.
- Do this in constant space and in one pass.

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Intersection of N Arrays**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-1-2020-easy-intersection-of-n-arrays) -- *Given n arrays, find the intersection of them.* [*\(Try ME\)*](https://repl.it/@trsong/Intersection-of-N-Arrays-1)

</summary>
<div>

**Question:** Given n arrays, find the intersection of them.

**Example:**
```py
intersection([1, 2, 3, 4], [2, 4, 6, 8], [3, 4, 5])  # returns [4]
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Intersection of Linked Lists**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-19-2020-easy-intersection-of-linked-lists) -- *Given two singly linked lists. The lists intersect at some node. Find, and return the node. Note: the lists are non-cyclical.* [*\(Try ME\)*](https://repl.it/@trsong/Intersection-of-Linked-Lists-1)

</summary>
<div>

**Question:** You are given two singly linked lists. The lists intersect at some node. Find, and return the node. Note: the lists are non-cyclical.

**Example:**
```py
A = 1 -> 2 -> 3 -> 4
B = 6 -> 3 -> 4
# This should return 3 (you may assume that any nodes with the same value are the same node)
```

</div>
</details>

### Doubly Linked List
---

<details>
<summary class="lc_e">

- [**\[Easy\] Determine If Linked List is Palindrome**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-2-2021-easy-determine-if-linked-list-is-palindrome) -- *You are given a doubly linked list. Determine if it is a palindrome.* [*\(Try ME\)*](https://repl.it/@trsong/Determine-If-Linked-List-is-Palindrome-1)

</summary>
<div>

**Question:** You are given a doubly linked list. Determine if it is a palindrome. 

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] First Unique Character from a Stream**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-24-2020-hard-first-unique-character-from-a-stream) -- *Given a stream of characters, find the first unique (non-repeating) character from stream.* [*\(Try ME\)*](https://repl.it/@trsong/First-Unique-Character-from-a-Stream-1)

</summary>
<div>

**Question:** Given a stream of characters, find the first unique (non-repeating) character from stream. You need to tell the first unique character in O(1) time at any moment.

**Example:**
```py
Input: Stream.of('abracadabra')
Output: Stream.of('aaabbbbbrcc')
Explanation:
a => a
abr => a
abra => b
abracadab => r
abracadabra => c
```

</div>
</details>


### Circular Linked List
---

### Fast-slow Pointers
---

<details>
<summary class="lc_e">

- [**\[Easy\] Rotate Linked List**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-26-2021-easy-rotate-linked-list) -- *Given a linked list and a number k, rotate the linked list by k places.* [*\(Try ME\)*](https://repl.it/@trsong/Rotate-Singly-Linked-List-1)

</summary>
<div>

**Question:** Given a linked list and a positive integer `k`, rotate the list to the right by k places.

For example, given the linked list `7 -> 7 -> 3 -> 5` and `k = 2`, it should become `3 -> 5 -> 7 -> 7`.

Given the linked list `1 -> 2 -> 3 -> 4 -> 5` and `k = 3`, it should become `3 -> 4 -> 5 -> 1 -> 2`.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Detect Linked List Cycle**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-3-2021-medium-detect-linked-list-cycle) -- *Given a linked list, determine if the linked list has a cycle in it.* [*\(Try ME\)*](https://repl.it/@trsong/Detect-Linked-List-Cycle-1)

</summary>
<div>

**Question:** Given a linked list, determine if the linked list has a cycle in it. 

**Example:**
```py
Input: 4 -> 3 -> 2 -> 1 -> 3 ... 
Output: True
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 287. Find the Duplicate Number**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-31-2020-lc-287-medium-find-the-duplicate-number) -- *Given an array of length `n + 1` whose elements belong to the set `{1, 2, ..., n}`. Find any duplicate element.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Duplicate-Number-from-Array-1)

</summary>
<div>

You are given an array of length `n + 1` whose elements belong to the set `{1, 2, ..., n}`. By the pigeonhole principle, there must be a duplicate. Find it in linear time and space.
 

</div>
</details>


## Sort
---

<details>
<summary class="lc_m">

- [**\[Medium\] LC 274. H-Index**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-5-2020-lc-274-medium-h-index) -- *The definition of the h-index is if a scholar has at least h of their papers cited h times.* [*\(Try ME\)*](https://repl.it/@trsong/Calculate-H-Index-1)

</summary>
<div>

**Question:** The h-index is a metric that attempts to measure the productivity and citation impact of the publication of a scholar. The definition of the h-index is if a scholar has at least h of their papers cited h times.

Given a list of publications of the number of citations a scholar has, find their h-index.

**Example:**
```py
Input: [3, 5, 0, 1, 3]
Output: 3
Explanation:
There are 3 publications with 3 or more citations, hence the h-index is 3.
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Sorted Square of Integers**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-20-2020-easy-sorted-square-of-integers) -- *Given a sorted list of integers, square the elements and give the output in sorted order.* [*\(Try ME\)*](https://repl.it/@trsong/Calculate-Sorted-Square-of-Integers-1)

</summary>
<div>

**Question:** Given a sorted list of integers, square the elements and give the output in sorted order.

For example, given `[-9, -2, 0, 2, 3]`, return `[0, 4, 4, 9, 81]`.

Additonal Requirement: Do it in-place. i.e. Space Complexity O(1).  

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Make the Largest Number**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-21-2020-easy-make-the-largest-number) -- *Given a number of integers, combine them so it would create the largest number.* [*\(Try ME\)*](https://repl.it/@trsong/Construct-Largest-Number-1)

</summary>
<div>

**Question:** Given a number of integers, combine them so it would create the largest number.

**Example:**
```py
Input: [17, 7, 2, 45, 72]
Output: 77245217
```


</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Sorting Window Range**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-21-2020-medium-sorting-window-range) -- *Given a list of numbers, find the smallest window to sort such that the whole list will be sorted.* [*\(Try ME\)*](https://repl.it/@trsong/Min-Window-Range-to-Sort-1)

</summary>
<div>

**Question:** Given a list of numbers, find the smallest window to sort such that the whole list will be sorted. If the list is already sorted return (0, 0). 

**Example:**
```py
Input: [2, 4, 7, 5, 6, 8, 9]
Output: (2, 4)
Explanation: Sorting the window (2, 4) which is [7, 5, 6] will also means that the whole list is sorted.
```

</div>
</details>

### Merge Sort
---

<details>
<summary class="lc_h">

- [**\[Hard\] Inversion Pairs**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-26-2020-hard-inversion-pairs) -- *Count total number of pairs such that a smaller element appears after a larger element.* [*\(Try ME\)*](https://repl.it/@trsong/Count-Inversion-Pairs-1)

</summary>
<div>

**Question:**  We can determine how "out of order" an array A is by counting the number of inversions it has. Two elements `A[i]` and `A[j]` form an inversion if `A[i] > A[j]` but `i < j`. That is, a smaller element appears after a larger element. Given an array, count the number of inversions it has. Do this faster than `O(N^2)` time. You may assume each element in the array is distinct.

For example, a sorted list has zero inversions. The array `[2, 4, 1, 3, 5]` has three inversions: `(2, 1)`, `(4, 1)`, and `(4, 3)`. The array `[5, 4, 3, 2, 1]` has ten inversions: every distinct pair forms an inversion.

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] Sort Linked List**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-14-2020-medium-sort-linked-list) -- *Given a linked list, sort it in O(n log n) time and constant space.* [*\(Try ME\)*](https://repl.it/@trsong/Sort-Linked-List-1)

</summary>
<div>

**Question:** Given a linked list, sort it in `O(n log n)` time and constant space.

For example, the linked list `4 -> 1 -> -3 -> 99` should become `-3 -> 1 -> 4 -> 99`.


</div>
</details>


### Quick Select & Quick Sort
---

<details>
<summary class="lc_e">

- [**\[Easy\] Find the K-th Largest Number**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-24-2020-easy-find-the-k-th-largest-number) -- *Find the k-th largest number in a sequence of unsorted numbers.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-K-th-Largest-Number-1)

</summary>
<div>

**Question:** Find the k-th largest number in a sequence of unsorted numbers. Can you do this in linear time?

**Example:**
```py
Input: 3, [8, 7, 2, 3, 4, 1, 5, 6, 9, 0]
Output: 7
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Sorting a List With 3 Unique Numbers**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-18-2020-medium-sorting-a-list-with-3-unique-numbers) -- *Given a list of numbers with only 3 unique numbers (1, 2, 3), sort the list in O(n) time.* [*\(Try ME\)*](https://repl.it/@trsong/Sorting-a-List-With-3-Unique-Numbers-1)

</summary>
<div>

**Question:** Given a list of numbers with only `3` unique numbers `(1, 2, 3)`, sort the list in `O(n)` time.

**Example:**
```py
Input: [3, 3, 2, 1, 3, 2, 1]
Output: [1, 1, 2, 2, 3, 3, 3]
```

</div>
</details>

### Counting Sort
---

<details>
<summary class="lc_e">

- [**\[Easy\] LC 451. Sort Characters By Frequency**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-22-2020-lc-451-easy-sort-characters-by-frequency) -- *Given a string, sort it in decreasing order based on the frequency of characters.* [*\(Try ME\)*](https://repl.it/@trsong/Sort-Characters-By-Frequency-1)

</summary>
<div>

**Question:** Given a string, sort it in decreasing order based on the frequency of characters. If there are multiple possible solutions, return any of them.

For example, given the string `tweet`, return `tteew`. `eettw` would also be acceptable.


</div>
</details>


## String
---


<details>
<summary class="lc_m">

- [**\[Medium\] LC 821. Shortest Distance to Character**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#mar-10-2021-lc-821-medium-shortest-distance-to-character) -- *Given a string s and a character c, find the distance for all characters in the string to the character c in the string s.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Shortest-Distance-to-Characters-1)

</summary>
<div>

**Question:**  Given a string s and a character c, find the distance for all characters in the string to the character c in the string s. 

You can assume that the character c will appear at least once in the string.

**Example:**
```py
shortest_dist('helloworld', 'l') 
# returns [2, 1, 0, 0, 1, 2, 2, 1, 0, 1]
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Efficiently Manipulate a Very Long String**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#mar-6-2021-hard-efficiently-manipulate-a-very-long-string) -- *Design a tree-based data structure to efficiently manipulate a very long string* [*\(Try ME\)*](https://repl.it/@trsong/Efficiently-Manipulate-a-Very-Long-String-1)

</summary>
<div>

**Question:** Design a tree-based data structure to efficiently manipulate a very long string that supports the following operations:

- `char char_at(int index)`, return char at index
- `LongString substring_at(int start_index, int end_index)`, return substring based on start and end index
- `void delete(int start_index, int end_index)`, deletes the substring 

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Reverse Words Keep Delimiters**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#mar-5-2021-hard-reverse-words-keep-delimiters) -- *Given a string and a set of delimiters, reverse the words in the string while maintaining the relative order of the delimiters.* [*\(Try ME\)*](https://repl.it/@trsong/Reverse-Words-and-Keep-Delimiters-1)

</summary>
<div>

**Question:** Given a string and a set of delimiters, reverse the words in the string while maintaining the relative order of the delimiters. For example, given "hello/world:here", return "here/world:hello"

Follow-up: Does your solution work for the following cases: "hello/world:here/", "hello//world:here"

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Regular Expression: Period and Asterisk**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-7-2021-hard-regular-expression-period-and-asterisk) -- *Implement regular expression matching with the following special characters: . (period) and \* (asterisk)* [*\(Try ME\)*](https://repl.it/@trsong/Regular-Expression-Period-and-Asterisk-1)

</summary>
<div>

**Question:** Implement regular expression matching with the following special characters:

- `.` (period) which matches any single character
 
- `*` (asterisk) which matches zero or more of the preceding element
 
That is, implement a function that takes in a string and a valid regular expression and returns whether or not the string matches the regular expression.

For example, given the regular expression "ra." and the string "ray", your function should return true. The same regular expression on the string "raymond" should return false.

Given the regular expression ".*at" and the string "chat", your function should return true. The same regular expression on the string "chats" should return false.

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] LC 65. Valid Number**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-30-2021-lc-65-hard-valid-number) -- *Given a string that may represent a number, determine if it is a number.* [*\(Try ME\)*](https://repl.it/@trsong/Valid-Number-1)

</summary>
<div>

**Question:** Given a string, return whether it represents a number. Here are the different kinds of numbers:

- "10", a positive integer
- "-10", a negative integer
- "10.1", a positive real number
- "-10.1", a negative real number
- "1e5", a number in scientific notation

And here are examples of non-numbers:

- "a"
- "x 1"
- "a -2"
- "-"

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 151. Reverse Words in a String**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-28-2020-lc-151-medium-reverse-words-in-a-string) -- *Given an input string, reverse the string word by word.* [*\(Try ME\)*](https://repl.it/@trsong/Reverse-words-in-a-string-1)

</summary>
<div>


**Question:** Given an input string, reverse the string word by word.

Note:

- A word is defined as a sequence of non-space characters.
- Input string may contain leading or trailing spaces. However, your reversed string should not contain leading or trailing spaces.
- You need to reduce multiple spaces between two words to a single space in the reversed string.

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



</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] Implement Soundex**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-29-2020-medium-implement-soundex) -- *Soundex is an algorithm used to categorize phonetically, such that two names that sound alike but are spelled differently have the same repr...* [*\(Try ME\)*](https://repl.it/@trsong/Implement-Soundex-1)

</summary>
<div>

**Question:** **Soundex** is an algorithm used to categorize phonetically, such that two names that sound alike but are spelled differently have the same representation.

**Soundex** maps every name to a string consisting of one letter and three numbers, like M460.

One version of the algorithm is as follows:

1. Remove consecutive consonants with the same sound (for example, change ck -> c).
2. Keep the first letter. The remaining steps only apply to the rest of the string.
3. Remove all vowels, including y, w, and h.
4. Replace all consonants with the following digits:
   - b, f, p, v → 1
   - c, g, j, k, q, s, x, z → 2
   - d, t → 3
   - l → 4
   - m, n → 5
   - r → 6
5. If you don't have three numbers yet, append zeros until you do. Keep the first three numbers.

Using this scheme, Jackson and Jaxen both map to J250.


</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Run-length String Encode and Decode**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-21-2020-easy-run-length-string-encode-and-decode) -- *Implement run-length encoding and decoding.* [*\(Try ME\)*](https://repl.it/@trsong/Run-length-String-Encode-and-Decode-1)

</summary>
<div>

**Question:** Run-length encoding is a fast and simple method of encoding strings. The basic idea is to represent repeated successive characters as a single count and character. For example, the string `"AAAABBBCCDAA"` would be encoded as `"4A3B2C1D2A"`.

Implement run-length encoding and decoding. You can assume the string to be encoded have no digits and consists solely of alphabetic characters. You can assume the string to be decoded is valid.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Zig-Zag String**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-4-2020-medium-zig-zag-string) -- *Given a string and a number of lines k, print the string in zigzag form.* [*\(Try ME\)*](https://repl.it/@trsong/Print-Zig-Zag-String-1)

</summary>
<div>

**Question:** Given a string and a number of lines k, print the string in zigzag form. In zigzag, characters are printed out diagonally from top left to bottom right until reaching the kth line, then back up to top right, and so on.

**Example:**
```py
Given the sentence "thisisazigzag" and k = 4, you should print:
t     a     g
 h   s z   a
  i i   i z
   s     g
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Tokenization**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-20-2020-medium-tokenization) -- *Given a dictionary of words and a string made up of those words (no spaces), return the original sentence in a list.* [*\(Try ME\)*](https://repl.it/@trsong/String-Tokenization-1)

</summary>
<div>

**Questions:** Given a dictionary of words and a string made up of those words (no spaces), return the original sentence in a list. If there is more than one possible reconstruction, return any of them. If there is no possible reconstruction, then return null.

**Example 1:**
```py
Input: ['quick', 'brown', 'the', 'fox'], 'thequickbrownfox'
Output: ['the', 'quick', 'brown', 'fox']
```

**Example 2:**
```py
Input: ['bed', 'bath', 'bedbath', 'and', 'beyond'], 'bedbathandbeyond'
Output:  Either ['bed', 'bath', 'and', 'beyond'] or ['bedbath', 'and', 'beyond']
```

</div>
</details>


### Parenthesis
---

<details>
<summary class="lc_e">

- [**\[Easy\] Invalid Parentheses to Remove**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-21-2020-easy-invalid-parentheses-to-remove) -- *Write a function to compute the minimum number of parentheses to be removed to make the string valid.* [*\(Try ME\)*](https://repl.it/@trsong/Count-Invalid-Parentheses-to-Remove-1)

</summary>
<div>

**Question:** Given a string of parentheses, write a function to compute the minimum number of parentheses to be removed to make the string valid (i.e. each open parenthesis is eventually closed).

For example, given the string `"()())()"`, you should return `1`. Given the string `")("`, you should return `2`, since we must remove all of them.

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] LC 301. Remove Invalid Parentheses**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-22-2020-lc-301-hard-remove-invalid-parentheses) -- *Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.* [*\(Try ME\)*](https://repl.it/@trsong/Ways-to-Remove-Invalid-Parentheses-1)

</summary>
<div>

**Question:** Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.

**Note:** The input string may contain letters other than the parentheses `(` and `)`.

**Example 1:**
```py
Input: "()())()"
Output: ["()()()", "(())()"]
```

**Example 2:**
```py
Input: "(a)())()"
Output: ["(a)()()", "(a())()"]
```

**Example 3:**
```py
Input: ")("
Output: [""]
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 678. Balanced Parentheses with Wildcard**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-2-2020-lc-678-medium-balanced-parentheses-with-wildcard) -- *Given string contains * that can represent either a (, ), or an empty string. Determine whether the parentheses are balanced.* [*\(Try ME\)*](https://repl.it/@trsong/Determine-Balanced-Parentheses-with-Wildcard-1)

</summary>
<div>

**Question:** You're given a string consisting solely of `(`, `)`, and `*`. `*` can represent either a `(`, `)`, or an empty string. Determine whether the parentheses are balanced.

For example, `(()*` and `(*)` are balanced. `)*(` is not balanced.

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 1021. Remove One Layer of Parenthesis**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-12-2020-lc-1021-easy-remove-one-layer-of-parenthesis) -- *Given a valid parenthesis string, remove the outermost layers of the parenthesis string and return the new parenthesis string.* [*\(Try ME\)*](https://repl.it/@trsong/Remove-One-Layer-of-Parenthesis-1)

</summary>
<div>

**Question:** Given a valid parenthesis string (with only '(' and ')', an open parenthesis will always end with a close parenthesis, and a close parenthesis will never start first), remove the outermost layers of the parenthesis string and return the new parenthesis string.

If the string has multiple outer layer parenthesis (ie (())()), remove all outer layers and construct the new string. So in the example, the string can be broken down into (()) + (). By removing both components outer layer we are left with () + '' which is simply (), thus the answer for that input would be ().

**Example 1:**
```py
Input: '(())()'
Output: '()'
```

**Example 2:**
```py
Input: '(()())'
Output: '()()'
```

**Example 3:**
```py
Input: '()()()'
Output: ''
```


</div>
</details>




### Anagram
---

<details>
<summary class="lc_e">

- [**\[Easy\] Step Word Anagram**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-8-2021-easy-step-word-anagram) -- *Given a dictionary of words and an input word, create a function that returns all valid step words (word adding a letter, and anagramming).* [*\(Try ME\)*](https://repl.it/@trsong/Step-Word-Anagram-1)

</summary>
<div>

**Question:** A step word is formed by taking a given word, adding a letter, and anagramming the result. For example, starting with the word `"APPLE"`, you can add an `"A"` and anagram to get `"APPEAL"`.

Given a dictionary of words and an input word, create a function that returns all valid step words.

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Is Anagram of Palindrome**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-1-2020-easy-is-anagram-of-palindrome) -- *Given a string, determine whether any permutation of it is a palindrome.* [*\(Try ME\)*](https://repl.it/@trsong/Is-Anagram-of-Palindrome-1)

</summary>
<div>

**Question:** Given a string, determine whether any permutation of it is a palindrome.

For example, `'carrace'` should return True, since it can be rearranged to form `'racecar'`, which is a palindrome. `'daily'` should return False, since there's no rearrangement that can form a palindrome.

</div>
</details>


### Palindrome
---


<details>
<summary class="lc_h">

- [**\[Hard\] LC 336. Palindrome Pairs**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#mar-1-2021-lc-336-hard-palindrome-pairs) -- *Given a list of words, find all pairs of unique indices such that the concatenation of the two words is a palindrome.* [*\(Try ME\)*](https://repl.it/@trsong/Find-All-Palindrome-Pairs-1)

</summary>
<div>

**Question:** Given a list of words, find all pairs of unique indices such that the concatenation of the two words is a palindrome.

For example, given the list `["code", "edoc", "da", "d"]`, return `[(0, 1), (1, 0), (2, 3)]`.

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] K-Palindrome**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-23-2021-hard-k-palindrome) -- *Given a string which we can delete at most k, return whether you can make a palindrome.* [*\(Try ME\)*](https://repl.it/@trsong/Find-K-Palindrome-1)

</summary>
<div>

**Question:** Given a string which we can delete at most k, return whether you can make a palindrome.

For example, given `'waterrfetawx'` and a k of 2, you could delete f and x to get `'waterretaw'`.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 131. Palindrome Partitioning**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-21-2021-lc-131-medium-palindrome-partitioning) -- *Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.* [*\(Try ME\)*](https://repl.it/@trsong/Palindrome-Partitioning-1)

</summary>
<div>

**Question:** Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.

A palindrome string is a string that reads the same backward as forward.

**Example 1:**
```py
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]
```

**Example 2:**
```py
Input: s = "a"
Output: [["a"]]
```

</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] Minimum Palindrome Substring**](https://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-20-2021-hard-minimum-palindrome-substring) -- *Given a string, split it into as few strings as possible such that each string is a palindrome.* [*\(Try ME\)*](https://repl.it/@trsong/Minimum-Palindrome-Substring-1)

</summary>
<div>

**Question:** Given a string, split it into as few strings as possible such that each string is a palindrome.

For example, given the input string `"racecarannakayak"`, return `["racecar", "anna", "kayak"]`.

Given the input string `"abc"`, return `["a", "b", "c"]`.

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Longest Palindromic Substring**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-10-2020-hard-longest-palindromic-substring) -- *Given a string, find the longest palindromic contiguous substring. If there are more than one with the maximum length, return any one.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Longest-Palindromic-Substring-1)

</summary>
<div>

**Question:** Given a string, find the longest palindromic contiguous substring. If there are more than one with the maximum length, return any one.

For example, the longest palindromic substring of `"aabcdcb"` is `"bcdcb"`. The longest palindromic substring of `"bananas"` is `"anana"`.


</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] LC 680. Remove Character to Create Palindrome**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-1-2020-lc-680-easy-remove-character-to-create-palindrome) -- *Given a string, determine if you can remove any character to create a palindrome.* [*\(Try ME\)*](https://repl.it/@trsong/Remove-Character-to-Create-Palindrome-1)

</summary>
<div>

**Question:** Given a string, determine if you can remove any character to create a palindrome.

**Example 1:**
```py
Input: "abcdcbea"
Output: True 
Explanation: Remove 'e' gives "abcdcba"
```

**Example 2:**
```py
Input: "abccba"
Output: True
```

**Example 3:**
```py
Input: "abccaa"
Output: False
```

</div>
</details>


## Tree
---

<details>
<summary class="lc_m">

- [**\[Medium\] Lazy Binary Tree Generation**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-22-2021-medium-lazy-binary-tree-generation) -- *Generate a finite, but an arbitrarily large binary tree quickly in `O(1)`.* [*\(Try ME\)*](https://repl.it/@trsong/Lazy-Binary-Tree-Generation-1)

</summary>
<div>

**Question:** Generate a finite, but an arbitrarily large binary tree quickly in `O(1)`.

That is, `generate()` should return a tree whose size is unbounded but finite.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 388. Longest Absolute File Path**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-6-2021-lc-388-medium-longest-absolute-file-path) -- *Given a string representing the file system in certain format, return the length of the longest absolute path to a file in that fs.* [*\(Try ME\)*](https://repl.it/@trsong/Longest-Absolute-File-Path-1)

</summary>
<div>

**Question:** Suppose we represent our file system by a string in the following manner:

The string `"dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"` represents:

```py
dir
    subdir1
    subdir2
        file.ext
```

The directory dir contains an empty sub-directory subdir1 and a sub-directory subdir2 containing a file file.ext.

The string `"dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"` represents:

```py
dir
    subdir1
        file1.ext
        subsubdir1
    subdir2
        subsubdir2
            file2.ext
```

The directory dir contains two sub-directories subdir1 and subdir2. subdir1 contains a file file1.ext and an empty second-level sub-directory subsubdir1. subdir2 contains a second-level sub-directory subsubdir2 containing a file file2.ext.

We are interested in finding the longest (number of characters) absolute path to a file within our file system. For example, in the second example above, the longest absolute path is `"dir/subdir2/subsubdir2/file2.ext"`, and its length is `32` (not including the double quotes).

Given a string representing the file system in the above format, return the length of the longest absolute path to a file in the abstracted file system. If there is no file in the system, return `0`.

**Note:**
- The name of a file contains at least a period and an extension.
- The name of a directory or sub-directory will not contain a period.


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Invert a Binary Tree**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-9-2020-medium-invert-a-binary-tree) -- *Given the root of a binary tree. Invert the binary tree in place.* [*\(Try ME\)*](https://repl.it/@trsong/Invert-All-Nodes-in-Binary-Tree-1)

</summary>
<div>

**Question:** You are given the root of a binary tree. Invert the binary tree in place. That is, all left children should become right children, and all right children should become left children.

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

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Locking in Binary Tree**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-13-2020-medium-locking-in-binary-tree) -- *Implement locking in a binary tree. A binary tree node can be locked or unlocked only if all of its descendants or ancestors are not locked.* [*\(Try ME\)*](https://repl.it/@trsong/Locking-and-Unlocking-in-Binary-Tree-1)

</summary>
<div>

**Question:** Implement locking in a binary tree. A binary tree node can be locked or unlocked only if all of its descendants or ancestors are not locked.
 
Design a binary tree node class with the following methods:

- `is_locked`, which returns whether the node is locked
- `lock`, which attempts to lock the node. If it cannot be locked, then it should return false. Otherwise, it should lock it and return true.
- `unlock`, which unlocks the node. If it cannot be unlocked, then it should return false. Otherwise, it should unlock it and return true.

You may augment the node to add parent pointers or any other property you would like. You may assume the class is used in a single-threaded program, so there is no need for actual locks or mutexes. Each method should run in O(h), where h is the height of the tree.


</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Depth of Binary Tree in Peculiar String Representation**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-31-2020-easy-depth-of-binary-tree-in-peculiar-string-representation) -- *Given a binary tree in a peculiar string representation, determine the depth of the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Depth-of-Binary-Tree-in-Peculiar-String-Representation-1)

</summary>
<div>

**Question:** You are given a binary tree in a peculiar string representation. Each node is written in the form `(lr)`, where `l` corresponds to the left child and `r` corresponds to the right child.

If either `l` or `r` is null, it will be represented as a zero. Otherwise, it will be represented by a new `(lr)` pair.
 
Given this representation, determine the depth of the tree.

**Here are a few examples:**
```py
A root node with no children: (00)
A root node with two children: ((00)(00))
An unbalanced tree with three consecutive left children: ((((00)0)0)0)
```

</div>
</details>


### BST
---

<details>
<summary class="lc_m">

- [**\[Medium\] Distance Between 2 Nodes in BST**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#mar-7-2021-medium-distance-between-2-nodes-in-bst) -- *Write a function that given a BST, it will return the distance (number of edges) between 2 nodes.* [*\(Try ME\)*](https://repl.it/@trsong/Distance-Between-2-Nodes-in-BST-1)

</summary>
<div>

**Question:** Write a function that given a BST, it will return the distance (number of edges) between 2 nodes.

**Example:**
```py
Given the following tree:

         5
        / \
       3   6
      / \   \
     2   4   7
    /         \
   1           8
The distance between 1 and 4 is 3: [1 -> 2 -> 3 -> 4]
The distance between 1 and 8 is 6: [1 -> 2 -> 3 -> 5 -> 6 -> 7 -> 8]
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Largest BST in a Binary Tree**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-3-2020-medium-largest-bst-in-a-binary-tree) -- *You are given the root of a binary tree. Find and return the largest subtree of that tree, which is a valid binary search tree.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Largest-BST-in-a-Binary-Tree-1)

</summary>
<div>

**Question:** You are given the root of a binary tree. Find and return the largest subtree of that tree, which is a valid binary search tree.

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

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Floor and Ceiling of BST**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-23-2020-easy-floor-and-ceiling-of-bst) -- *Given an integer k and a binary search tree, find the floor (less than or equal to) of k, and the ceiling (larger than or equal to) of k.* [*\(Try ME\)*](https://repl.it/@trsong/Floor-and-Ceiling-of-BST-1)

</summary>
<div>

**Question:** Given an integer `k` and a binary search tree, find the `floor` (less than or equal to) of `k`, and the `ceiling` (larger than or equal to) of `k`. If either does not exist, then print them as None.

**Example:**
```py
          8
        /   \    
      4      12
    /  \    /  \
   2    6  10   14

k: 11  Floor: 10  Ceil: 12
k: 1   Floor: None  Ceil: 2
k: 6   Floor: 6   Ceil: 6
k: 15  Floor: 14  Ceil: None
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Number of Smaller Elements to the Right**](https://repl.it/@trsong/Find-Number-of-Smaller-Elements-to-the-Right) -- *Given number array,  return the number of smaller elements to the right of each element in the original input array.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Number-of-Smaller-Elements-to-the-Right-1)

</summary>
<div>

**Question:** Given an array of integers, return a new array where each element in the new array is the number of smaller elements to the right of that element in the original input array.

For example, given the array `[3, 4, 9, 6, 1]`, return `[1, 1, 2, 1, 0]`, since:

* There is 1 smaller element to the right of 3
* There is 1 smaller element to the right of 4
* There are 2 smaller elements to the right of 9
* There is 1 smaller element to the right of 6
* There are no smaller elements to the right of 1
 
</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Generate Binary Search Trees**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-11-2020-easy-generate-binary-search-trees) -- *Given an integer N, construct all possible binary search trees with N nodes.* [*\(Try ME\)*](https://repl.it/@trsong/Generate-Binary-Search-Trees-with-N-Nodes-1)

</summary>
<div>

**Question:** Given an integer N, construct all possible binary search trees with N nodes.

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Second Largest in BST**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-24-2020-medium-second-largest-in-bst) -- *Given the root to a binary search tree, find the second largest node in the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Second-Largest-in-BST-1)

</summary>
<div>

**Question:** Given the root to a binary search tree, find the second largest node in the tree.

</div>
</details>

### Recursion
---


<details>
<summary class="lc_m">

- [**\[Medium\] Split a Binary Search Tree**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#mar-14-2021--medium-split-a-binary-search-tree) -- *Given a BST and a value s, split the BST into 2 trees, where one tree has all values less than or equal to s, and the others.* [*\(Try ME\)*](https://replit.com/@trsong/Split-BST-into-Two-BSTs-1)

</summary>
<div>

**Question:** Given a binary search tree (BST) and a value s, split the BST into 2 trees, where one tree has all values less than or equal to s, and the other tree has all values greater than s while maintaining the tree structure of the original BST. You can assume that s will be one of the node's value in the BST. Return both tree's root node as a tuple.

**Example:**
```py
Given the following tree, and s = 2
     3
   /   \
  1     4
   \     \
    2     5

Split into two trees:
 1    And   3
  \          \
   2          4
               \
                5
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Filter Binary Tree Leaves**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-14-2021-easy-filter-binary-tree-leaves) -- *Given a binary tree and an integer k, filter the binary tree such that its leaves don’t contain the value k.* [*\(Try ME\)*](https://repl.it/@trsong/Filter-Binary-Tree-Leaves-of-Certain-Value-1)

</summary>
<div>

**Questions:** Given a binary tree and an integer k, filter the binary tree such that its leaves don't contain the value k. Here are the rules:

- If a leaf node has a value of k, remove it.
- If a parent node has a value of k, and all of its children are removed, remove it.

**Example:**
```py
Given the tree to be the following and k = 1:
     1
    / \
   1   1
  /   /
 2   1

After filtering the result should be:
     1
    /
   1
  /
 2
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Making a Height Balanced Binary Search Tree**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-10-2021-easy-making-a-height-balanced-binary-search-tree) -- *Given a sorted list, create a height balanced binary search tree, meaning the height differences of each node can only differ by at most 1.* [*\(Try ME\)*](https://repl.it/@trsong/Making-a-Height-Balanced-Binary-Search-Tree-1)

</summary>
<div>

**Question:** Given a sorted list, create a height balanced binary search tree, meaning the height differences of each node can only differ by at most 1.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Maximum Path Sum in Binary Tree**](https://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-1-2021-medium-maximum-path-sum-in-binary-tree) -- *Given the root of a binary tree. Find the path between 2 nodes that maximizes the sum of all the nodes in the path, and return the sum.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Maximum-Path-Sum-in-Binary-Tree-1)

</summary>
<div>

**Question:** You are given the root of a binary tree. Find the path between 2 nodes that maximizes the sum of all the nodes in the path, and return the sum. The path does not necessarily need to go through the root.

**Example:**
```py
Given the following binary tree, the result should be 42 = 20 + 2 + 10 + 10.
       *10
       /  \
     *2   *10
     / \     \
   *20  1    -25
             /  \
            3    4
(* denotes the max path)
```


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Most Frequent Subtree Sum**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-20-2021-medium-most-frequent-subtree-sum) -- *Given the root of a binary tree, find the most frequent subtree sum.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Most-Frequent-Subtree-Sum-1)

</summary>
<div>

**Question:** Given a binary tree, find the most frequent subtree sum.

If there is a tie between the most frequent sum, return the smaller one.

**Example:**
```
   3
  / \
 1   -3

The above tree has 3 subtrees.:
The root node with 3, and the 2 leaf nodes, which gives us a total of 3 subtree sums.
The root node has a sum of 1 (3 + 1 + -3).
The left leaf node has a sum of 1, and the right leaf node has a sum of -3. 
Therefore the most frequent subtree sum is 1.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 236. Lowest Common Ancestor of a Binary Tree**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-23-2020-lc-236-medium-lowest-common-ancestor-of-a-binary-tree) -- *Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Lowest-Common-Ancestor-of-a-Given-Binary-Tree-1)

</summary>
<div>

**Question:** Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

**Example:**

```py
     1
   /   \
  2     3
 / \   / \
4   5 6   7

LCA(4, 5) = 2
LCA(4, 6) = 1
LCA(3, 4) = 1
LCA(2, 4) = 2
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Tree Isomorphism Problem**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-29-2020-easy-tree-isomorphism-problem) -- *Write a function to detect if two trees are isomorphic.* [*\(Try ME\)*](https://repl.it/@trsong/Is-Binary-Tree-Isomorphic-1)

</summary>
<div>

**Question:** Write a function to detect if two trees are isomorphic. Two trees are called isomorphic if one of them can be obtained from other by a series of flips, i.e. by swapping left and right children of a number of nodes. Any number of nodes at any level can have their children swapped. Two empty trees are isomorphic.


**Example:** 
```py
The following two trees are isomorphic with following sub-trees flipped: 2 and 3, NULL and 6, 7 and 8.

Tree1:
     1
   /   \
  2     3
 / \   /
4   5 6
   / \
  7   8

Tree2:
   1
 /   \
3     2
 \   / \
  6 4   5
       / \
      8   7
```


</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Full Binary Tree**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-11-2020-easy-full-binary-tree) -- *Given a binary tree, remove the nodes in which there is only 1 child, so that the binary tree is a full binary tree.* [*\(Try ME\)*](https://repl.it/@trsong/Prune-to-Full-Binary-Tree-1)

</summary>
<div>

**Question:** Given a binary tree, remove the nodes in which there is only 1 child, so that the binary tree is a full binary tree.

So leaf nodes with no children should be kept, and nodes with 2 children should be kept as well.

**Example:**
```py
Given this tree:
     1
    / \ 
   2   3
  /   / \
 0   9   4

We want a tree like:
     1
    / \ 
   0   3
      / \
     9   4
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 938. Range Sum of BST**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-22-2020-lc-938-easy-range-sum-of-bst) -- *Given a binary search tree and a range [a, b] (inclusive), return the sum of the elements of the binary search tree within the range.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Range-Sum-of-BST-1)

</summary>
<div>

**Question:** Given a binary search tree and a range `[a, b]` (inclusive), return the sum of the elements of the binary search tree within the range.

**Example:**
```py
Given the range [4, 9] and the following tree:

    5
   / \
  3   8
 / \ / \
2  4 6  10

return 23 (5 + 4 + 6 + 8).
```

</div>
</details>




### Traversal
---

<details>
<summary class="lc_h">

- [**\[Hard\] Construct Cartesian Tree from Inorder Traversal**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-12-2020-hard-construct-cartesian-tree-from-inorder-traversal) -- *Given in-order traversal, construct a Cartesian tree. A Cartesian tree is heap-ordered, so that each parent value is smaller than children.* [*\(Try ME\)*](https://repl.it/@trsong/Construct-Cartesian-Tree-from-Inorder-Traversal-1)

</summary>
<div>

**Question:** A Cartesian tree with sequence S is a binary tree defined by the following two properties:

- It is heap-ordered, so that each parent value is strictly less than that of its children.
- An in-order traversal of the tree produces nodes with values that correspond exactly to S.

Given a sequence S, construct the corresponding Cartesian tree.

**Example:**
```py
Given the sequence [3, 2, 6, 1, 9], the resulting Cartesian tree would be:
      1
    /   \   
  2       9
 / \
3   6
```


</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Count Number of Unival Subtrees**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-17-2020-easy-count-number-of-unival-subtrees) -- *A unival tree is a tree where all the nodes have the same value. Given a binary tree, return the number of unival subtrees in the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Count-Total-Number-of-Uni-val-Subtrees-1)

</summary>
<div>

**Question:** A unival tree is a tree where all the nodes have the same value. Given a binary tree, return the number of unival subtrees in the tree.

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


</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Find Corresponding Node in Cloned Tree**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-27-2020-easy-find-corresponding-node-in-cloned-tree) -- *Given two binary trees that are duplicates of one another, and given a node in one tree, find that corresponding node in the second tree.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Corresponding-Node-in-Cloned-Tree-1)

</summary>
<div>

**Question:** Given two binary trees that are duplicates of one another, and given a node in one tree, find that corresponding node in the second tree. 
 
There can be duplicate values in the tree (so comparing node1.value == node2.value isn't going to work).
 
</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 653. Two Sum in BST**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-19-2020-easy-two-sum-in-bst) -- *Given the root of a binary search tree, and a target K, return two nodes in the tree whose sum equals K.* [*\(Try ME\)*](https://repl.it/@trsong/Two-Sum-in-BST-1)

</summary>
<div>

**Question:** Given the root of a binary search tree, and a target K, return two nodes in the tree whose sum equals K.

**Example:**
```py
Given the following tree and K of 20:

    10
   /   \
 5      15
       /  \
     11    15

Return the nodes 5 and 15.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 230. Kth Smallest Element in a BST**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-1-2020-lc-230-medium-kth-smallest-element-in-a-bst) -- *Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Kth-Smallest-Element-in-a-BST-1)

</summary>
<div>

**Question:** Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

**Example 1:**

```py
Input: 
   3
  / \
 1   4
  \
   2

k = 1
Output: 1
```

**Example 2:**
```py
Input: 
       5
      / \
     3   6
    / \
   2   4
  /
 1

k = 3
Output: 3
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Tree Serialization**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-20-2020-medium-tree-serialization) -- *Given the root of a binary tree. You need to implement 2 functions: Serialize and Deserialize.* [*\(Try ME\)*](https://repl.it/@trsong/Serialize-and-Deserialize-the-Binary-Tree-1)

</summary>
<div>

**Question:** You are given the root of a binary tree. You need to implement 2 functions:

1. `serialize(root)` which serializes the tree into a string representation
2. `deserialize(s)` which deserializes the string back to the original tree that it represents

For this problem, often you will be asked to design your own serialization format. However, for simplicity, let's use the pre-order traversal of the tree.

**Example:**
```py
     1
    / \
   3   4
  / \   \
 2   5   7

serialize(tree)
# returns "1 3 2 # # 5 # # 4 # 7 # #"
```


</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Symmetric K-ary Tree**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-5-2020-easy-symmetric-k-ary-tree) -- *Given a k-ary tree, figure out if the tree is symmetrical.* [*\(Try ME\)*](https://repl.it/@trsong/Is-Symmetric-K-ary-Tree-1)

</summary>
<div>

**Question:** Given a k-ary tree, figure out if the tree is symmetrical.
 
A k-ary tree is a tree with k-children, and a tree is symmetrical if the data of the left side of the tree is the same as the right side of the tree. 

Here's an example of a symmetrical k-ary tree.

```py
        4
     /     \
    3        3
  / | \    / | \
9   4  1  1  4  9
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 872. Leaf-Similar Trees**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-8-2020-lc-872-easy-leaf-similar-trees) -- *Given two trees, whether they are "leaf similar". Two trees are considered "leaf-similar" if their leaf orderings are the same.* [*\(Try ME\)*](https://repl.it/@trsong/Leaf-Similar-Trees-1)

</summary>
<div>

**Question:** Given two trees, whether they are `"leaf similar"`. Two trees are considered `"leaf-similar"` if their leaf orderings are the same. 

For instance, the following two trees are considered leaf-similar because their leaves are `[2, 1]`:

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

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Construct BST from Post-order Traversal**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-17-2020-medium-construct-bst-from-post-order-traversal) -- *Given the sequence of keys visited by a postorder traversal of a binary search tree, reconstruct the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Construct-Binary-Search-Tree-from-Post-order-Traversal-1)

</summary>
<div>

**Question:** Given the sequence of keys visited by a postorder traversal of a binary search tree, reconstruct the tree.

**Example:**
```py
Given the sequence 2, 4, 3, 8, 7, 5, you should construct the following tree:

    5
   / \
  3   7
 / \   \
2   4   8
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] In-order & Post-order Binary Tree Traversal**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-29-2020-medium-in-order--post-order-binary-tree-traversal) -- *Given Postorder and Inorder traversals, construct the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Test-Build-Binary-Tree-with-In-order-and-Post-order-Traversal)

</summary>
<div>

**Question:** Given Postorder and Inorder traversals, construct the tree.

**Examples 1:**
```py
Input: 
in_order = [2, 1, 3]
post_order = [2, 3, 1]

Output: 
      1
    /   \
   2     3 
```

**Example 2:**
```py
Input: 
in_order = [4, 8, 2, 5, 1, 6, 3, 7]
post_order = [8, 4, 5, 2, 6, 7, 3, 1]

Output:
          1
       /     \
     2        3
   /    \   /   \
  4     5   6    7
    \
      8
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Pre-order & In-order Binary Tree Traversal**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-28-2020-medium-pre-order--in-order-binary-tree-traversal) -- *Given pre-order and in-order traversals of a binary tree, write a function to reconstruct the tree.* [*\(Try ME\)*](https://repl.it/@trsong/Test-Pre-order-and-In-order-Binary-Tree-Traversal)

</summary>
<div>

**Question:** Given pre-order and in-order traversals of a binary tree, write a function to reconstruct the tree.

For example, given the following preorder traversal:

```py
[a, b, d, e, c, f, g]
```

And the following inorder traversal:

```py
[d, b, e, a, f, c, g]
```

You should return the following tree:

```py
    a
   / \
  b   c
 / \ / \
d  e f  g
```

</div>
</details>

### Range Query


<details>
<summary class="lc_m">

- [**\[Medium\] 24-Hour Hit Counter**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-28-2021-medium-24-hour-hit-counter) -- *You are given an array of length 24, where each element represents the number of new subscribers during the corresponding hour.* [*\(Try ME\)*](https://repl.it/@trsong/24-Hour-Hit-Counter-1)

</summary>
<div>

**Question:** You are given an array of length 24, where each element represents the number of new subscribers during the corresponding hour. Implement a data structure that efficiently supports the following:

- `update(hour: int, value: int)`: Increment the element at index hour by value.
- `query(start: int, end: int)`: Retrieve the number of subscribers that have signed up between start and end (inclusive). You can assume that all values get cleared at the end of the day, and that you will not be asked for start and end values that wrap around midnight.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 307. Range Sum Query - Mutable**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-15-2020-lc-307-medium-range-sum-query---mutable) -- *Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.* [*\(Try ME\)*](https://repl.it/@trsong/Mutable-Range-Sum-Query-1)

</summary>
<div>

**Question:** Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.

The update(i, val) function modifies nums by updating the element at index i to val.

**Example:**
```py
Given nums = [1, 3, 5]

sumRange(0, 2) -> 9
update(1, 2)
sumRange(0, 2) -> 8
```

</div>
</details>


## Trie
---

<details>
<summary class="lc_m">

- [**\[Medium\] Shortest Unique Prefix**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-16-2021-medium-shortest-unique-prefix) -- *Given an array of words, find all shortest unique prefixes to represent each word in the given array. Assume no word is prefix of another.* [*\(Try ME\)*](https://repl.it/@trsong/Find-All-Shortest-Unique-Prefix-1)

</summary>
<div>

**Question:** Given an array of words, find all shortest unique prefixes to represent each word in the given array. Assume that no word is prefix of another.

**Example:**
```py
Input: ['zebra', 'dog', 'duck', 'dove']
Output: ['z', 'dog', 'du', 'dov']
Explanation: dog => dog
             dove = dov 
             duck = du
             z   => zebra 
```

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Implement Prefix Map Sum**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-6-2020-easy-implement-prefix-map-sum) -- *Implement a PrefixMapSum class with the following methods: insert(key: str, value: int), sum(prefix: str)* [*\(Try ME\)*](https://repl.it/@trsong/Implement-Prefix-Map-Sum-1)

</summary>
<div>

**Question:** Implement a PrefixMapSum class with the following methods:

- `insert(key: str, value: int)`: Set a given key's value in the map. If the key already exists, overwrite the value.
- `sum(prefix: str)`: Return the sum of all values of keys that begin with a given prefix.

**Example:**
```py
mapsum.insert("columnar", 3)
assert mapsum.sum("col") == 3

mapsum.insert("column", 2)
assert mapsum.sum("col") == 5
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] LC 212. Word Search II**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-22-2020-lc-212-hard-word-search-ii) -- *Given an m x n board of characters and a list of strings words, return all words on the board.* [*\(Try ME\)*](https://repl.it/@trsong/Word-Search-II-1)

</summary>
<div>

**Question:** Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.
 
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


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 421. Maximum XOR of Two Numbers in an Array**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-20-2020-lc-421-medium-maximum-xor-of-two-numbers-in-an-array) -- *Given an array of integers, find the maximum XOR of any two elements.* [*\(Try ME\)*](https://repl.it/@trsong/Maximum-XOR-of-Two-Numbers-in-an-Array-1)

</summary>
<div>

**Question:** Given an array of integers, find the maximum XOR of any two elements.

**Example:**
```py
Input: nums = [3,10,5,25,2,8]
Output: 28
Explanation: The maximum result is 5 XOR 25 = 28.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Group Words that are Anagrams**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-8-2020-medium-group-words-that-are-anagrams) -- *Given a list of words, group the words that are anagrams of each other. (An anagram are words made up of the same letters).* [*\(Try ME\)*](https://repl.it/@trsong/Group-Anagrams-1)

</summary>
<div>

**Question:** Given a list of words, group the words that are anagrams of each other. (An anagram are words made up of the same letters).

**Example:**
```py
Input: ['abc', 'bcd', 'cba', 'cbd', 'efg']
Output: [['abc', 'cba'], ['bcd', 'cbd'], ['efg']]
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Autocompletion**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-16-2020-medium-autocompletion) -- *Given a large set of words and then a single word prefix, find all words that it can complete to.* [*\(Try ME\)*](https://repl.it/@trsong/Autocompletion-1)

</summary>
<div>

**Question:**  Implement auto-completion. Given a large set of words (for instance 1,000,000 words) and then a single word prefix, find all words that it can complete to.

**Example:**
```py
class Solution:
  def build(self, words):
    # Fill this in.
    
  def autocomplete(self, word):
    # Fill this in.

s = Solution()
s.build(['dog', 'dark', 'cat', 'door', 'dodge'])
s.autocomplete('do')  # Return ['dog', 'door', 'dodge']
```

</div>
</details>

## Stack
---

<details>
<summary class="lc_m">

- [**\[Medium\] Evaluate Expression in Reverse Polish Notation**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-25-2021-medium-evaluate-expression-in-reverse-polish-notation) -- *Given an arithmetic expression in Reverse Polish Notation, write a program to evaluate it.* [*\(Try ME\)*](https://repl.it/@trsong/Evaluate-Expression-Represented-in-Reverse-Polish-Notation-1)

</summary>
<div>

**Question:** Given an arithmetic expression in **Reverse Polish Notation**, write a program to evaluate it.

The expression is given as a list of numbers and operands. 

**Example 1:** 
```py
[5, 3, '+'] should return 5 + 3 = 8.
```

**Example 2:**
```py
 [15, 7, 1, 1, '+', '-', '/', 3, '*', 2, 1, 1, '+', '+', '-'] should return 5, 
 since it is equivalent to ((15 / (7 - (1 + 1))) * 3) - (2 + (1 + 1)) = 5.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Normalize Pathname**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-7-2021-medium-normalize-pathname) -- *Given an absolute pathname that may have . or .. as part of it, return the shortest standardized path.* [*\(Try ME\)*](https://repl.it/@trsong/Normalize-to-Absolute-Pathname-1)

</summary>
<div>

**Question:** Given an absolute pathname that may have `"."` or `".."` as part of it, return the shortest standardized path.

For example, given `"/usr/bin/../bin/./scripts/../"`, return `"/usr/bin"`.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Largest Rectangular Area in a Histogram**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-26-2020-medium-largest-rectangular-area-in-a-histogram) -- *Given a histogram consisting of rectangles of different heights. Determine the area of the largest rectangle that can be formed.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Largest-Rectangular-Area-in-a-Histogram-1)

</summary>
<div>

**Question:** You are given a histogram consisting of rectangles of different heights. Determine the area of the largest rectangle that can be formed only from the bars of the histogram.

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


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Interleave Stacks**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-27-2020-medium-interleave-stacks) -- *Given a stack of N elements, interleave the first half of the stack with the second half reversed using only one other queue.* [*\(Try ME\)*](https://repl.it/@trsong/Interleave-First-and-Second-Half-of-Stacks-1)

</summary>
<div>

**Question:** Given a stack of N elements, interleave the first half of the stack with the second half reversed using only one other queue. This should be done in-place.

Recall that you can only push or pop from a stack, and enqueue or dequeue from a queue.

For example, if the stack is `[1, 2, 3, 4, 5]`, it should become `[1, 5, 2, 4, 3]`. If the stack is `[1, 2, 3, 4]`, it should become `[1, 4, 2, 3]`.

Hint: Try working backwards from the end state.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Maximum In A Stack**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-16-2020-medium-maximum-in-a-stack) -- *Implement a class for a stack that supports all the regular functions (push, pop) and an additional function of max().* [*\(Try ME\)*](https://repl.it/@trsong/Maximum-In-A-Stack-1)

</summary>
<div>

**Question:** Implement a class for a stack that supports all the regular functions (`push`, `pop`) and an additional function of `max()` which returns the maximum element in the stack (return None if the stack is empty). Each method should run in constant time.

**Example**:
```py
s = MaxStack()
s.push(1)
s.push(2)
s.push(3)
s.push(2)
print s.max()  # 3
s.pop()
s.pop()
print s.max()  # 2
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Nearest Larger Number**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-13-2020-medium-nearest-larger-number) -- *Given an array of numbers and an index i, return the index of the nearest larger number of the number at index i.* [*\(Try ME\)*](https://repl.it/@trsong/Nearest-Larger-Number-1)

</summary>
<div>

**Question:** Given an array of numbers and an index `i`, return the index of the nearest larger number of the number at index `i`, where distance is measured in array indices.

For example, given `[4, 1, 3, 5, 6]` and index `0`, you should return `3`.

If two distances to larger numbers are the equal, then return any one of them. If the array at i doesn't have a nearest larger integer, then return null.

**Follow-up:** If you can preprocess the array, can you do this in constant time?

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 1047. Remove Adjacent Duplicate Characters**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-15-2020-lc-1047-easy-remove-adjacent-duplicate-characters) -- *Given a string, we want to remove 2 adjacent characters that are the same, and repeat the process with the new string until we can no longer* [*\(Try ME\)*](https://repl.it/@trsong/Remove-2-Adjacent-Duplicate-Characters-1)

</summary>
<div>

**Question:** Given a string, we want to remove 2 adjacent characters that are the same, and repeat the process with the new string until we can no longer perform the operation.

**Example:**
```py
remove_adjacent_dup("cabba")
# Start with cabba
# After remove bb: caa
# After remove aa: c
# Returns c
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Index of Larger Next Number**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-17-2020-medium-index-of-larger-next-number) -- *Given a list of numbers, for each element find the next element that is larger than the current element.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Index-of-Larger-Next-Number-1)

</summary>
<div>

**Question:** Given a list of numbers, for each element find the next element that is larger than the current element. Return the answer as a list of indices. If there are no elements larger than the current element, then use -1 instead.

**Example:** 
```py
larger_number([3, 2, 5, 6, 9, 8])
# return [2, 2, 3, 4, -1, -1]
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] The Celebrity Problem**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-27-2020-medium-the-celebrity-problem) -- *Given a list of N people and the above operation, find a way to identify the celebrity in O(N) time.* [*\(Try ME\)*](https://repl.it/@trsong/Test-The-Celebrity-Problem)

</summary>
<div>

**Question:** At a party, there is a single person who everyone knows, but who does not know anyone in return (the "celebrity"). To help figure out who this is, you have access to an `O(1)` method called `knows(a, b)`, which returns `True` if person `a` knows person `b`, else `False`.

Given a list of `N` people and the above operation, find a way to identify the celebrity in `O(N)` time.
 
</div>
</details>


## Heap / Priority Queue
---

<details>
<summary class="lc_m">

- [**\[Medium\] Running Median of a Number Stream**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-29-2021-medium-running-median-of-a-number-stream) -- *Print the running median of a number stream* [*\(Try ME\)*](https://repl.it/@trsong/Running-Median-of-a-Number-Stream-1)

</summary>
<div>

**Question:** Compute the running median of a sequence of numbers. That is, given a stream of numbers, print out the median of the list so far on each new element.
 
Recall that the median of an even-numbered list is the average of the two middle numbers.
 
For example, given the sequence `[2, 1, 5, 7, 2, 0, 5]`, your algorithm should print out:

```py
2
1.5
2
3.5
2
2
2
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 692. Top K Frequent words**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-6-2021-lc-692-medium-top-k-frequent-words) -- *Given a non-empty list of words, return the k most frequent words.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Top-K-Frequent-Elements-1)

</summary>
<div>

**Question:** Given a non-empty list of words, return the k most frequent words. The output should be sorted from highest to lowest frequency, and if two words have the same frequency, the word with lower alphabetical order comes first. Input will contain only lower-case letters.

**Example 1:**
```py
Input: ["i", "love", "leapcode", "i", "love", "coding"], k = 2
Output: ["i", "love"]
Explanation: "i" and "love" are the two most frequent words.
    Note that "i" comes before "love" due to a lower alphabetical order.
```

**Example 2:**
```py
Input: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
Output: ["the", "is", "sunny", "day"]
Explanation: "the", "is", "sunny" and "day" are the four most frequent words,
    with the number of occurrence being 4, 3, 2 and 1 respectively.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Sort a K-Sorted Array**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-4-2021-medium-sort-a-k-sorted-array) -- *Given a list of N numbers, in which each number is located at most k places away from its sorted position. Sort this array in O(N log k).* [*\(Try ME\)*](https://repl.it/@trsong/Sort-a-K-Sorted-Array-1)

</summary>
<div>

**Question:** You are given a list of `N` numbers, in which each number is located at most `k` places away from its sorted position. For example, if `k = 1`, a given element at index `4` might end up at indices `3`, `4`, or `5`.

Come up with an algorithm that sorts this list in O(N log k) time.

**Example:**
```py
Input: [3, 2, 6, 5, 4], k=2
Output: [2, 3, 4, 5, 6]
As seen above, every number is at most 2 indexes away from its proper sorted index.
``` 

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] LC 218. City Skyline**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-27-2020--lc-218-hard-city-skyline) -- *Given a list of building in the form of (left, right, height), return what the skyline should look like.* [*\(Try ME\)*](https://repl.it/@trsong/Find-City-Skyline-1)

</summary>
<div>

**Question:** Given a list of building in the form of `(left, right, height)`, return what the skyline should look like. The skyline should be in the form of a list of `(x-axis, height)`, where x-axis is the point where there is a change in height starting from 0, and height is the new height starting from the x-axis.

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

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Similar Websites**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-8-2020-medium-similar-websites) -- *You are given a list of (website, user) pairs that represent users visiting websites. Come up with a program that identifies the top k pairs* [*\(Try ME\)*](https://repl.it/@trsong/Find-Similar-Websites-1)

</summary>
<div>

**Question:** You are given a list of (website, user) pairs that represent users visiting websites. Come up with a program that identifies the top k pairs of websites with the greatest similarity.

**Note:** The similarity metric bewtween two sets equals intersection / union. 

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

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] M Smallest in K Sorted Lists**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-4-2020-medium-m-smallest-in-k-sorted-lists) -- *Given k sorted arrays of possibly different sizes, find m-th smallest value in the merged array.* [*\(Try ME\)*](https://repl.it/@trsong/Find-M-Smallest-in-K-Sorted-Lists-1)

</summary>
<div>

**Question:** Given k sorted arrays of possibly different sizes, find m-th smallest value in the merged array.

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

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Largest Product of 3 Elements**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-25-2020-easy-largest-product-of-3-elements) -- *You are given an array of integers. Return the largest product that can be made by multiplying any 3 integers in the array.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Largest-Product-of-3-Elements-in-Array-1)

</summary>
<div>

**Question:** You are given an array of integers. Return the largest product that can be made by multiplying any 3 integers in the array.

**Example:**
```py
Input: [-4, -4, 2, 8]
Output: 128
Explanation: the largest product can be made by multiplying -4 * -4 * 8 = 128.
```

</div>
</details>

### Scheduling
---

<details>
<summary class="lc_m">

- [**\[Medium\] Craft Sentence**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-19-2020-medium-craft-sentence) -- *Given a sequence of words and an integer line length k, write an algorithm to justify text.* [*\(Try ME\)*](https://repl.it/@trsong/Craft-Sentence-and-Adjust-Text-Width-1)

</summary>
<div>

**Question:** Write an algorithm to justify text. Given a sequence of words and an integer line length k, return a list of strings which represents each line, fully justified.

More specifically, you should have as many words as possible in each line. There should be at least one space between each word. Pad extra spaces when necessary so that each line has exactly length k. Spaces should be distributed as equally as possible, with the extra spaces, if any, distributed starting from the left.

If you can only fit one word on a line, then you should pad the right-hand side with spaces.

Each word is guaranteed not to be longer than k.

For example, given the list of words `["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]` and `k = 16`, you should return the following:

```py
["the  quick brown",
 "fox  jumps  over",
 "the   lazy   dog"]
```


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Rearrange String with Repeated Characters**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-7-2020-medium-rearrange-string-with-repeated-characters) -- *Given a string with repeated characters, rearrange the string so that no two adjacent characters are the same.* [*\(Try ME\)*](https://repl.it/@trsong/Rearrange-String-with-Repeated-Characters-1)

</summary>
<div>

**Question:** Given a string with repeated characters, rearrange the string so that no two adjacent characters are the same. If this is not possible, return None.

For example, given `"aaabbc"`, you could return `"ababac"`. Given `"aaab"`, return `None`.

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] LC 358. Rearrange String K Distance Apart**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-11-2020-lc-358-hard-rearrange-string-k-distance-apart) -- *Given a non-empty string str and an integer k, rearrange the string such that the same characters are at least distance k from each other.* [*\(Try ME\)*](https://repl.it/@trsong/Rearrange-Strings-K-Distance-Apart-1)

</summary>
<div>

**Question:** Given a non-empty string str and an integer k, rearrange the string such that the same characters are at least distance k from each other.

All input strings are given in lowercase letters. If it is not possible to rearrange the string, return an empty string "".

**Example 1:**
```py
str = "aabbcc", k = 3
Result: "abcabc"
The same letters are at least distance 3 from each other.
```

**Example 2:**
```py
str = "aaabc", k = 3 
Answer: ""
It is not possible to rearrange the string.
```

**Example 3:**
```py
str = "aaadbbcc", k = 2
Answer: "abacabcd"
Another possible answer is: "abcabcda"
The same letters are at least distance 2 from each other.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 621. Task Scheduler**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-10-2020--lc-621-medium-task-scheduler) -- *Given a char array representing tasks CPU need to do.You need to return the least number of intervals the CPU will take to finish all tasks.* [*\(Try ME\)*](https://repl.it/@trsong/LC-621-Task-Scheduler-1)

</summary>
<div>

**Question:** Given a char array representing tasks CPU need to do. It contains capital letters A to Z where different letters represent different tasks. Tasks could be done without original order. Each task could be done in one interval. For each interval, CPU could finish one task or just be idle.

However, there is a non-negative cooling interval n that means between two same tasks, there must be at least n intervals that CPU are doing different tasks or just be idle.

You need to return the least number of intervals the CPU will take to finish all the given tasks.

**Example:**
```py
Input: tasks = ["A", "A", "A", "B", "B", "B"], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B.
```


</div>
</details>



## Hashmap
---

<details>
<summary class="lc_e">

- [**\[Easy\] Word Ordering in a Different Alphabetical Order**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-20-2020-easy-word-ordering-in-a-different-alphabetical-order) -- *Given a list of words, and an arbitrary alphabetical order, verify that the words are in order of the alphabetical order.* [*\(Try ME\)*](https://repl.it/@trsong/Determine-Word-Ordering-in-a-Different-Alphabetical-Order-1)

</summary>
<div>

**Question:** Given a list of words, and an arbitrary alphabetical order, verify that the words are in order of the alphabetical order.

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

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LRU Cache**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-9-2021-medium-lru-cache) -- *Design and implement an LRU cache class with the 2 functions 'put' and 'get'.* [*\(Try ME\)*](https://repl.it/@trsong/Design-LRU-Cache-1)

</summary>
<div>

**Question:** Implement an LRU (Least Recently Used) cache. It should be able to be initialized with a cache size n, and contain the following methods:

- `put(key, value)`: sets key to value. If there are already n items in the cache and we are adding a new item, then it should also remove the least recently used item.
- `get(key)`: gets the value at key. If no such key exists, return null.
  
Each operation should run in O(1) time.

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

</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] LFU Cache**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-8-2021-hard-lfu-cache) -- *Designs and implements data structures that use the least frequently used (LFU) cache.* [*\(Try ME\)*](https://repl.it/@trsong/Design-LFU-Cache-1)

</summary>
<div>

**Question:** Implement an LFU (Least Frequently Used) cache. It should be able to be initialized with a cache size n, and contain the following methods:

- `put(key, value)`: sets key to value. If there are already n items in the cache and we are adding a new item, then it should also remove the least frequently used item. If there is a tie, then the least recently used key should be removed.
- `get(key)`: gets the value at key. If no such key exists, return null.
Each operation should run in O(1) time.


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Fixed Order Task Scheduler with Cooldown**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-26-2020-medium-fixed-order-task-scheduler-with-cooldown) -- *Given a list of tasks to perform, with a cooldown period. Return execution order.* [*\(Try ME\)*](Fixed Order Task Scheduler with Cooldown-1)

</summary>
<div>

**Question:** We have a list of tasks to perform, with a cooldown period. We can do multiple of these at the same time, but we cannot run the same task simultaneously.

Given a list of tasks, find how long it will take to complete the tasks in the order they are input.

**Example:**
```py
tasks = [1, 1, 2, 1]
cooldown = 2
output: 7 (order is 1 _ _ 1 2 _ 1)
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 763. Partition Labels**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-7-2020-lc-763-medium-partition-labels) -- *Partition the given string into as many parts as possible so that each letter appears in at most one part. Return length for each part.* [*\(Try ME\)*](https://repl.it/@trsong/Partition-Labels-1)

</summary>
<div>

**Question:** A string S of lowercase English letters is given. We want to partition this string into as many parts as possible so that each letter appears in at most one part, and return a list of integers representing the size of these parts.

**Example:**
```py
Input: S = "ababcbacadefegdehijhklij"
Output: [9, 7, 8]
Explanation: The partition is "ababcbaca", "defegde", "hijhklij". 
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits S into less parts.
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 1171. Remove Consecutive Nodes that Sum to 0**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-26-2020-lc-1171-medium-remove-consecutive-nodes-that-sum-to-0) -- *Given a linked list of integers, remove all consecutive nodes that sum up to 0.* [*\(Try ME\)*](https://repl.it/@trsong/Remove-Consecutive-Nodes-that-Sum-to-Zero-1)

</summary>
<div>

**Question:** Given a linked list of integers, remove all consecutive nodes that sum up to 0.

**Example 1:**
```py
Input: 10 -> 5 -> -3 -> -3 -> 1 -> 4 -> -4
Output: 10
Explanation: The consecutive nodes 5 -> -3 -> -3 -> 1 sums up to 0 so that sequence should be removed. 4 -> -4 also sums up to 0 too so that sequence should also be removed.
```

**Example 2:**

```py
Input: 1 -> 2 -> -3 -> 3 -> 1
Output: 3 -> 1
Note: 1 -> 2 -> 1 would also be accepted.
```

**Example 3:**
```py
Input: 1 -> 2 -> 3 -> -3 -> 4
Output: 1 -> 2 -> 4
```

**Example 4:**
```py
Input: 1 -> 2 -> 3 -> -3 -> -2
Output: 1
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 560. Subarray Sum Equals K**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-24-2020-lc-560-medium-subarray-sum-equals-k) -- *Given a list of integers and a number K, return which contiguous elements of the list sum to K.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Number-of-Sub-array-Sum-Equals-K-1)

</summary>
<div>

**Question:** Given a list of integers and a number `K`, return which contiguous elements of the list sum to `K`.

For example, if the list is `[1, 2, 3, 4, 5]` and `K` is `9`, then it should return `[2, 3, 4]`, since `2 + 3 + 4 = 9`.

</div>
</details>


## Greedy
---

<details>
<summary class="lc_h">

- [**\[Hard\] Smallest Stab Set**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-24-2021-hard-smallest-stab-set) -- *P “stabs” X if every interval in X contains at least one point in P. Compute the smallest set of points that stabs X.* [*\(Try ME\)*](https://repl.it/@trsong/Smallest-Stab-Set-1)

</summary>
<div>

**Question:** Let `X` be a set of `n` intervals on the real line. We say that a set of points `P` "stabs" `X` if every interval in `X` contains at least one point in `P`. Compute the smallest set of points that stabs X.

For example, given the intervals `[(1, 4), (4, 5), (7, 9), (9, 12)]`, you should return `[4, 9]`.

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 1647. Minimum Deletions to Make Character Frequencies Unique**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-16-2020-lc-1647-medium-minimum-deletions-to-make-character-frequencies-unique) -- *Given a string s, return the minimum number of characters you need to delete to make s good. s is good if no two letters have same count.* [*\(Try ME\)*](https://repl.it/@trsong/Minimum-Deletions-to-Make-Character-Frequencies-Unique-1)

</summary>
<div>


**Question:** A string s is called good if there are no two different characters in s that have the same frequency.

Given a string s, return the minimum number of characters you need to delete to make s good.

The frequency of a character in a string is the number of times it appears in the string. For example, in the string "aab", the frequency of 'a' is 2, while the frequency of 'b' is 1.


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

</div>
</details>




## Divide and Conquer
---

<details>
<summary class="lc_e">

- [**\[Easy\] Max and Min with Limited Comparisons**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-16-2020-easy-max-and-min-with-limited-comparisons) -- *Find the maximum and minimum of the list using less than `2 * (n - 1)` comparisons.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Max-and-Min-with-Limited-Comparisons-1)

</summary>
<div>

**Question:** Given a list of numbers of size `n`, where `n` is greater than `3`, find the maximum and minimum of the list using less than `2 * (n - 1)` comparisons.

**Example:**
```py
Input: [3, 5, 1, 2, 4, 8]
Output: (1, 8)
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 240. Search a 2D Matrix II**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-13-2020-lc-240-medium-search-a-2d-matrix-ii) -- *Write an efficient algorithm that searches for a value in an m x n matrix.* [*\(Try ME\)*](https://repl.it/@trsong/Search-in-a-Sorted-2D-Matrix-1)

</summary>
<div>

**Question:** Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

- Integers in each row are sorted in ascending from left to right.
- Integers in each column are sorted in ascending from top to bottom.

**Example:**
```py
Consider the following matrix:

[
  [ 1,  4,  7, 11, 15],
  [ 2,  5,  8, 12, 19],
  [ 3,  6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
Given target = 5, return True.
Given target = 20, return False.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Maximum Subarray Sum**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-5-2020-easy-maximum-subarray-sum) -- *Given array that may contain both positive and negative integers, find the sum of contiguous subarray of numbers which has the largest sum.* [*\(Try ME\)*](https://repl.it/@trsong/Maximum-Subarray-Sum-Divide-and-Conquer-1)

</summary>
<div>

**Question:** You are given a one dimensional array that may contain both positive and negative integers, find the sum of contiguous subarray of numbers which has the largest sum.

For example, if the given array is `[-2, -5, 6, -2, -3, 1, 5, -6]`, then the maximum subarray sum is 7 as sum of `[6, -2, -3, 1, 5]` equals 7

Solve this problem with Divide and Conquer as well as DP separately.


</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] The Tower of Hanoi**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-1-2020-medium-the-tower-of-hanoi) -- *The Tower of Hanoi is a puzzle game with three rods.Write a function that prints out all the steps necessary to complete the Tower of Hanoi.* [*\(Try ME\)*](https://repl.it/@trsong/Solve-the-Tower-of-Hanoi-Problem-1)

</summary>
<div>

**Question:** The Tower of Hanoi is a puzzle game with three rods and n disks, each a different size.

All the disks start off on the first rod in a stack. They are ordered by size, with the largest disk on the bottom and the smallest one at the top.

The goal of this puzzle is to move all the disks from the first rod to the last rod while following these rules:

- You can only move one disk at a time.
- A move consists of taking the uppermost disk from one of the stacks and placing it on top of another stack.
- You cannot place a larger disk on top of a smaller disk.

Write a function that prints out all the steps necessary to complete the Tower of Hanoi. 
- You should assume that the rods are numbered, with the first rod being 1, the second (auxiliary) rod being 2, and the last (goal) rod being 3.

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

</div>
</details>


## Graph
---

### BFS
---

<details>
<summary class="lc_e">

- [**\[Easy\] ZigZag Binary Tree**](https://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-2-2021-easy-zigzag-binary-tree) -- *Given a binary tree, write an algorithm to print the nodes in zigzag order.* [*\(Try ME\)*](https://repl.it/@trsong/ZigZag-Order-of-Binary-Tree-1)

</summary>
<div>

**Questions:** In Ancient Greece, it was common to write text with the first line going left to right, the second line going right to left, and continuing to go back and forth. This style was called "boustrophedon".

Given a binary tree, write an algorithm to print the nodes in boustrophedon order.

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


</div>
</details>



<details>
<summary class="lc_m">

- [**\[Medium\] LC 127. Word Ladder**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-2-2020-lc-127-medium-word-ladder) -- *Find the shortest transformation sequence from start to end such that only one letter is changed at each step of the sequence.* [*\(Try ME\)*](https://repl.it/@trsong/Word-Ladder-1)

</summary>
<div>

**Question:** Given a `start` word, an `end` word, and a dictionary of valid words, find the shortest transformation sequence from `start` to `end` such that only one letter is changed at each step of the sequence, and each transformed word exists in the dictionary. If there is no possible transformation, return null. Each word in the dictionary have the same length as start and end and is lowercase.

For example, given `start = "dog"`, `end = "cat"`, and `dictionary = {"dot", "dop", "dat", "cat"}`, return `["dog", "dot", "dat", "cat"]`.

Given `start = "dog"`, `end = "cat"`, and `dictionary = {"dot", "tod", "dat", "dar"}`, return `null` as there is no possible transformation from `dog` to `cat`.
 

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 286. Walls and Gates**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-23-2020-lc-286-medium-walls-and-gates) -- *Given a m x n 2D grid, fill each empty room with the distance to its nearest gate.* [*\(Try ME\)*](https://repl.it/@trsong/Identify-Walls-and-Gates-1)

</summary>
<div>

**Question:** You are given a m x n 2D grid initialized with these three possible values.

* -1 - A wall or an obstacle.
* 0 - A gate.
* INF - Infinity means an empty room. We use the value `2^31 - 1 = 2147483647` to represent INF as you may assume that the distance to a gate is less than 2147483647.
 
Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.

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

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Deepest Node in a Binary Tree**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-2-2020-easy-deepest-node-in-a-binary-tree) -- *You are given the root of a binary tree. Return the deepest node (the furthest node from the root).* [*\(Try ME\)*](https://repl.it/@trsong/Find-Deepest-Node-in-a-Binary-Tree-1)

</summary>
<div>

**Question:** You are given the root of a binary tree. Return the deepest node (the furthest node from the root).

**Example:**
```py
    a
   / \
  b   c
 /
d
The deepest node in this tree is d at depth 3.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Maximum Number of Connected Colors**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-30-2020-medium-maximum-number-of-connected-colors) -- *Given a grid with cells in different colors, find the maximum number of same color cells that are connected.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Maximum-Number-of-Connected-Colors-1)

</summary>
<div>

**Question:** Given a grid with cells in different colors, find the maximum number of same color  cells that are connected.

Note: two cells are connected if they are of the same color and adjacent to each other: left, right, top or bottom. To stay simple, we use integers to represent colors:

The following grid have max 4 connected colors. `[color 3: (1, 2), (1, 3), (2, 1), (2, 2)]`

```py
 [
    [1, 1, 2, 2, 3], 
    [1, 2, 3, 3, 1],
    [2, 3, 3, 1, 2]
 ]
 ```


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Find All Cousins in Binary Tree**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-21-2020-medium-find-all-cousins-in-binary-tree) -- *Given a binary tree and a particular node, find all cousins of that node.* [*\(Try ME\)*](https://repl.it/@trsong/All-Cousins-in-Binary-Tree-1)

</summary>
<div>

**Question:** Two nodes in a binary tree can be called cousins if they are on the same level of the tree but have different parents. 

Given a binary tree and a particular node, find all cousins of that node.


**Example:**
```py
In the following diagram 4 and 6 are cousins:

    1
   / \
  2   3
 / \   \
4   5   6
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Longest Path in Binary Tree**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-31-2020-hard-longest-path-in-binary-tree) -- *Given a binary tree, return any of the longest path.* [*\(Try ME\)*](https://repl.it/@trsong/Test-Longest-Path-in-Binary-Tree)

</summary>
<div>

**Question:** Given a binary tree, return any of the longest path.

**Example 1:**
```py
Input:      1
          /   \
        2      3
      /  \
    4     5

Output: [4, 2, 1, 3] or [5, 2, 1, 3]  
```

**Example 2:**
```py
Input:      1
          /   \
        2      3
      /  \      \
    4     5      6

Output: [4, 2, 1, 3, 6] or [5, 2, 1, 3, 6] 
```

</div>
</details>

### Level-based BFS
---

<details>
<summary class="lc_h">

- [**\[Hard\] LC 934. Shortest Bridge**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-1-2020-lc-934-medium-shortest-bridge) -- *In a given 2D binary array A, there are two islands.Return the smallest number of 0s that must be flipped.* [*\(Try ME\)*](https://repl.it/@trsong/Shortest-Bridge-1)

</summary>
<div>

**Question:** In a given 2D binary array A, there are two islands.  (An island is a 4-directionally connected group of 1s not connected to any other 1s.)

Now, we may change 0s to 1s so as to connect the two islands together to form 1 island.

Return the smallest number of 0s that must be flipped.  (It is guaranteed that the answer is at least 1.)

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


</div>
</details>


### DFS
---

<details>
<summary class="lc_h">

- [**\[Hard\] Critical Routers (Articulation Point)**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#mar-13-2021-hard-critical-routers-articulation-point) -- *Given an undirected connected graph, find all articulation points in the given graph.* [*\(Try ME\)*](https://replit.com/@trsong/Find-the-Critical-Routers-Articulation-Point-1)

</summary>
<div>

**Question:** You are given an undirected connected graph. An articulation point (or cut vertex) is defined as a vertex which, when removed along with associated edges, makes the graph disconnected (or more precisely, increases the number of connected components in the graph). The task is to find all articulation points in the given graph.

**Example1:**
```py
Input: vertices = 4, edges = [[0, 1], [1, 2], [2, 3]]
Output: [1, 2]
Explanation: 
Removing either vertex 0 or 3 along with edges [0, 1] or [2, 3] does not increase number of connected components. 
But removing 1, 2 breaks graph into two components.
```

**Example2:**
```py
Input: vertices = 5, edges = [[0, 1], [0, 2], [1, 2], [0, 3], [3, 4]]
Output: [0, 3]
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 332. Reconstruct Itinerary**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-12-2021-lc-332-medium-reconstruct-itinerary) -- *Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], reconstruct the itinerary in order.* [*\(Try ME\)*](https://repl.it/@trsong/Reconstruct-Flight-Itinerary-1)

</summary>
<div>

**Questions:** Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK. Thus, the itinerary must begin with JFK.

Note:

1. If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string. For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].
2. All airports are represented by three capital letters (IATA code).
3. You may assume all tickets form at least one valid itinerary.
   
**Example 1:**
```java
Input: [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
Output: ["JFK", "MUC", "LHR", "SFO", "SJC"]
```

**Example 2:**
```java
Input: [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
Explanation: Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"].
             But it is larger in lexical order.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Longest Consecutive Sequence in an Unsorted Array**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-25-2020-medium-longest-consecutive-sequence-in-an-unsorted-array) -- *Given an array of integers, return the largest range, inclusive, of integers that are all included in the array.* [*\(Try ME\)*](https://repl.it/@trsong/Longest-Consecutive-Sequence-1)

</summary>
<div>

**Question:** Given an array of integers, return the largest range, inclusive, of integers that are all included in the array.

For example, given the array `[9, 6, 1, 3, 8, 10, 12, 11]`, return `(8, 12)` since `8, 9, 10, 11, and 12` are all in the array.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Direction and Position Rule Verification**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-27-2020-medium-direction-and-position-rule-verification) -- *Given a list of rules, check if the sum of the rules validate.* [*\(Try ME\)*](https://repl.it/@trsong/Verify-List-of-Direction-and-Position-Rules-1)

</summary>
<div>

**Question:** A rule looks like this:

```py
A NE B
``` 
This means this means point A is located northeast of point B.
 
```
A SW C
```
means that point A is southwest of C.

Given a list of rules, check if the sum of the rules validate. For example:

```
A N B
B NE C
C N A
```

does not validate, since A cannot be both north and south of C.

```
A NW B
A N B
```
 
is considered valid.

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Count Visible Nodes in Binary Tree**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-19-2020-easy-count-visible-nodes-in-binary-tree) -- *In a binary tree, if no node with greater value than A’s along the path, this node is visible. Count total number of such nodes.* [*\(Try ME\)*](https://repl.it/@trsong/Count-Visible-Nodes-in-Binary-Tree-1)

</summary>
<div>

**Question:** In a binary tree, if in the path from root to the node A, there is no node with greater value than A’s, this node A is visible. We need to count the number of visible nodes in a binary tree.

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

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Isolated Islands**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-14-2020-medium-isolated-islands) -- *Given a matrix of 1s and 0s, return the number of "islands" in the matrix.* [*\(Try ME\)*](https://repl.it/@trsong/Count-Number-of-Isolated-Islands-1)

</summary>
<div>

**Question:** Given a matrix of 1s and 0s, return the number of "islands" in the matrix. A 1 represents land and 0 represents water, so an island is a group of 1s that are neighboring whose perimeter is surrounded by water.

For example, this matrix has 4 islands.
```py
1 0 0 0 0
0 0 1 1 0
0 1 1 0 0
0 0 0 0 0
1 1 0 0 1
1 1 0 0 1
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Largest Path Sum from Root To Leaf**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-4-2020-easy-largest-path-sum-from-root-to-leaf) -- *Given a binary tree, find and return the largest path from root to leaf.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Largest-Path-Sum-from-Root-To-Leaf-1)

</summary>
<div>

**Question:** Given a binary tree, find and return the largest path from root to leaf.

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

</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] LC 403. Frog Jump**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-2-2020-lc-403-hard-frog-jump) -- *A frog is crossing a river. The river is divided into x units and at each unit there may or may not exist a stone.* [*\(Try ME\)*](https://repl.it/@trsong/Solve-Frog-Jump-Problem-1)

</summary>
<div>

**Question:** A frog is crossing a river. The river is divided into x units and at each unit there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.

Given a list of stones' positions (in units) in sorted ascending order, determine if the frog is able to cross the river by landing on the last stone. Initially, the frog is on the first stone and assume the first jump must be 1 unit.
 
If the frog's last jump was k units, then its next jump must be either k - 1, k, or k + 1 units. Note that the frog can only jump in the forward direction.

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


</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] De Bruijn Sequence**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-24-2020-hard-de-bruijn-sequence) -- *Given a set of characters C and an integer k,  find a De Bruijn Sequence.* [*\(Try ME\)*](https://repl.it/@trsong/Find-De-Bruijn-Sequence-1)

</summary>
<div>

**Question:** Given a set of characters `C` and an integer `k`, a **De Bruijn** Sequence is a cyclic sequence in which every possible k-length string of characters in C occurs exactly once.
 
**Background:** De Bruijn Sequence can be used to shorten a brute-force attack on a PIN-like code lock that does not have an "enter" key and accepts the last n digits entered. For example, a digital door lock with a 4-digit code would have B (10, 4) solutions, with length 10000. Therefore, only at most 10000 + 3 = 10003 (as the solutions are cyclic) presses are needed to open the lock. Trying all codes separately would require 4 × 10000 = 40000 presses.

**Example1:**
```py
Input: C = [0, 1], k = 3
Output: 0011101000
All possible strings of length three (000, 001, 010, 011, 100, 101, 110 and 111) appear exactly once as sub-strings in C.
```

**Example2:**
```py
Input: C = [0, 1], k = 2
Output: 01100
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Level of tree with Maximum Sum**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-3-2020-easy-level-of-tree-with-maximum-sum) -- *Given a binary tree, find the level in the tree where the sum of all nodes on that level is the greatest.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Level-of-tree-with-Maximum-Sum-1)

</summary>
<div>

**Question:** Given a binary tree, find the level in the tree where the sum of all nodes on that level is the greatest.

**Example:**
```py
The following tree should return level 1:
    1          Level 0 - Sum: 1
   / \
  4   5        Level 1 - Sum: 9 
 / \ / \
3  2 4 -1      Level 2 - Sum: 8
```

</div>
</details>




<details>
<summary class="lc_e">

- [**\[Easy\] Flatten Nested List Iterator**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-27-2021-easy-flatten-nested-list-iterator) -- *Implement a 2D iterator class. It will be initialized with an array of arrays, and should implement the following methods: next(), hasNext()* [*\(Try ME\)*](https://repl.it/@trsong/Flatten-Nested-List-Iterator-1)

</summary>
<div>

**Question:** Implement a 2D iterator class. It will be initialized with an array of arrays, and should implement the following methods:

- `next()`: returns the next element in the array of arrays. If there are no more elements, raise an exception.
- `has_next()`: returns whether or not the iterator still has elements left.

For example, given the input `[[1, 2], [3], [], [4, 5, 6]]`, calling `next()` repeatedly should output `1, 2, 3, 4, 5, 6`.

Do not use flatten or otherwise clone the arrays. Some of the arrays can be empty.

</div>
</details>



<details>
<summary class="lc_e">

- [**\[Easy\] Flatten a Nested Dictionary**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-8-2020-easy-flatten-a-nested-dictionary) -- *Write a function to flatten a nested dictionary. Namespace the keys with a period.* [*\(Try ME\)*](https://repl.it/@trsong/Flatten-Nested-Dictionary-1)

</summary>
<div>

**Question:** Write a function to flatten a nested dictionary. Namespace the keys with a period.

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

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Circle of Chained Words**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-9-2020-medium-circle-of-chained-words) -- *Given a list of words, determine if there is a way to ‘chain’ all the words in a circle.* [*\(Try ME\)*](https://repl.it/@trsong/Contains-Circle-of-Chained-Words-1)

</summary>
<div>

**Question:** Two words can be 'chained' if the last character of the first word is the same as the first character of the second word.

Given a list of words, determine if there is a way to 'chain' all the words in a circle.

**Example:**
```py
Input: ['eggs', 'karat', 'apple', 'snack', 'tuna']
Output: True
Explanation:
The words in the order of ['apple', 'eggs', 'snack', 'karat', 'tuna'] creates a circle of chained words.
```

</div>
</details>


### Topological Sort
---

<details>
<summary class="lc_h">

- [**\[Hard\] Max Path Value in Directed Graph**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-13-2021-hard-max-path-value-in-directed-graph) -- *Given a directed graph, return the largest value path of the graph. Define a path’s value as the number of most frequent letters along path.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Max-Letter-Path-Value-in-Directed-Graph-1)

</summary>
<div>

**Question:** In a directed graph, each node is assigned an uppercase letter. We define a path's value as the number of most frequently-occurring letter along that path. For example, if a path in the graph goes through "ABACA", the value of the path is 3, since there are 3 occurrences of 'A' on the path.

Given a graph with n nodes and m directed edges, return the largest value path of the graph. If the largest value is infinite, then return null.

The graph is represented with a string and an edge list. The i-th character represents the uppercase letter of the i-th node. Each tuple in the edge list (i, j) means there is a directed edge from the i-th node to the j-th node. Self-edges are possible, as well as multi-edges.

**Example1:** 
```py
Input: "ABACA", [(0, 1), (0, 2), (2, 3), (3, 4)]
Output: 3
Explanation: maximum value 3 using the path of vertices [0, 2, 3, 4], ie. A -> A -> C -> A.
```

**Example2:**
```py
Input: "A", [(0, 0)]
Output: None
Explanation: we have an infinite loop.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Satisfactory Playlist**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-15-2021-medium-satisfactory-playlist) -- *Given a set of these ranked lists, interleave them to create a playlist that satisfies everyone’s priorities.* [*\(Try ME\)*](https://repl.it/@trsong/Satisfactory-Playlist-for-Everyone-1)

</summary>
<div>

**Question:** You have access to ranked lists of songs for various users. Each song is represented as an integer, and more preferred songs appear earlier in each list. For example, the list `[4, 1, 7]` indicates that a user likes song `4` the best, followed by songs `1` and `7`.

Given a set of these ranked lists, interleave them to create a playlist that satisfies everyone's priorities.

For example, suppose your input is `[[1, 7, 3], [2, 1, 6, 7, 9], [3, 9, 5]]`. In this case a satisfactory playlist could be `[2, 1, 6, 7, 3, 9, 5]`.

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Order of Alien Dictionary**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-25-2020-hard-order-of-alien-dictionary) -- *Give a dictionary of sorted words in an alien language, returns the correct order of letters in this language.* [*\(Try ME\)*](https://repl.it/@trsong/Alien-Dictionary-Order-1)

</summary>
<div>

**Question:** You come across a dictionary of sorted words in a language you've never seen before. Write a program that returns the correct order of letters in this language.

For example, given `['xww', 'wxyz', 'wxyw', 'ywx', 'ywz']`, you should return `['x', 'z', 'w', 'y']`.


</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Order of Course Prerequisites**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-18-2020-hard-order-of-course-prerequisites) -- *Given a hashmap of courseId key to a list of courseIds values. Return a sorted ordering of courses such that we can finish all courses.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Order-of-Course-Prerequisites-1)

</summary>
<div>

We're given a hashmap associating each courseId key with a list of courseIds values, which represents that the prerequisites of courseId are courseIds. Return a sorted ordering of courses such that we can finish all courses.

Return null if there is no such ordering.

For example, given `{'CSC300': ['CSC100', 'CSC200'], 'CSC200': ['CSC100'], 'CSC100': []}`, should return `['CSC100', 'CSC200', 'CSC300']`.

</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] LC 329. Longest Increasing Path in a Matrix**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-16-2020-lc-329-hard-longest-increasing-path-in-a-matrix) -- *Given an integer matrix, find the length of the longest increasing path.* [*\(Try ME\)*](https://repl.it/@trsong/Longest-Increasing-Path-in-a-Matrix-1)

</summary>
<div>

**Question:** Given an integer matrix, find the length of the longest increasing path.

From each cell, you can either move to four directions: left, right, up or down. You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).

**Example 1:**
```py
Input: nums = 
[
  [9,9,4],
  [6,6,8],
  [2,1,1]
] 
Output: 4 
Explanation: The longest increasing path is [1, 2, 6, 9].
```

**Example 2:**
```py
Input: nums = 
[
  [3,4,5],
  [3,2,6],
  [2,2,1]
] 
Output: 4 
Explanation: The longest increasing path is [3, 4, 5, 6]. Moving diagonally is not allowed.
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Shortest Uphill and Downhill Route**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-7-2020-hard-shortest-uphill-and-downhill-route) -- *A runner decide the route goes entirely uphill at first, and then entirely downhill.  Find the length of the shortest of such route.* [*\(Try ME\)*](https://repl.it/@trsong/Shortest-Uphill-and-Downhill-Route-1)

</summary>
<div>

**Question:** A competitive runner would like to create a route that starts and ends at his house, with the condition that the route goes entirely uphill at first, and then entirely downhill.

Given a dictionary of places of the form `{location: elevation}`, and a dictionary mapping paths between some of these locations to their corresponding distances, find the length of the shortest route satisfying the condition above. Assume the runner's home is location `0`.

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


</div>
</details>


### Union Find
---

<details>
<summary class="lc_m">

- [**\[Medium\] Number of Connected Components**](https://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#mar-12-2021-medium-number-of-connected-components) -- *Given a list of undirected edges which represents a graph, find out the number of connected components.* [*\(Try ME\)*](https://replit.com/@trsong/Number-of-Connected-Components-1)

</summary>
<div>

**Question:** Given a list of undirected edges which represents a graph, find out the number of connected components.

**Example:**
```py
Input: [(1, 2), (2, 3), (4, 1), (5, 6)]
Output: 2
Explanation: In the above example, vertices 1, 2, 3, 4 are all connected, and 5, 6 are connected, and thus there are 2 connected components in the graph above.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Friend Cycle Problem**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-19-2021-medium-friend-cycle-problem) -- *Given a friendship list, determine the number of friend groups in the class.* [*\(Try ME\)*](https://repl.it/@trsong/Friend-Cycle-Problem-1)

</summary>
<div>

**Question:** A classroom consists of `N` students, whose friendships can be represented in an adjacency list. For example, the following descibes a situation where `0` is friends with `1` and `2`, `3` is friends with `6`, and so on.

```py
{0: [1, 2],
 1: [0, 5],
 2: [0],
 3: [6],
 4: [],
 5: [1],
 6: [3]} 
```

Each student can be placed in a friend group, which can be defined as the transitive closure of that student's friendship relations. In other words, this is the smallest set such that no student in the group has any friends outside this group. For the example above, the friend groups would be `{0, 1, 2, 5}`, `{3, 6}`, `{4}`.

Given a friendship list such as the one above, determine the number of friend groups in the class.

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Power Supply to All Cities**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-9-2020-hard-power-supply-to-all-cities) -- *Given a graph of possible electricity connections(each with their own cost)between cities in an area, find the cheapest way to supply power.* [*\(Try ME\)*](https://repl.it/@trsong/Design-Power-Supply-to-All-Cities-1)

</summary>
<div>

**Question:** Given a graph of possible electricity connections (each with their own cost) between cities in an area, find the cheapest way to supply power to all cities in the area. 

**Example 1:**
```py
Input: cities = ['Vancouver', 'Richmond', 'Burnaby']
       cost_btw_cities = [
           ('Vancouver', 'Richmond', 1),
           ('Vancouver', 'Burnaby', 1),
           ('Richmond', 'Burnaby', 2)]
Output: 2  
Explanation: 
Min cost to supply all cities is to connect the following cities with total cost 1 + 1 = 2: 
(Vancouver, Burnaby), (Vancouver, Richmond)
```

**Example 2:**
```py
Input: cities = ['Toronto', 'Mississauga', 'Waterloo', 'Hamilton']
       cost_btw_cities = [
           ('Mississauga', 'Toronto', 1),
           ('Toronto', 'Waterloo', 2),
           ('Waterloo', 'Hamilton', 3),
           ('Toronto', 'Hamilton', 2),
           ('Mississauga', 'Hamilton', 1),
           ('Mississauga', 'Waterloo', 2)]
Output: 4
Explanation: Min cost to connect to all cities is 4:
(Toronto, Mississauga), (Toronto, Waterloo), (Mississauga, Hamilton)
```


</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 130. Surrounded Regions**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-18-2020-lc-130-medium-surrounded-regions) -- *A region is captured by flipping all ‘O’s into ‘X’s in that surrounded region.* [*\(Try ME\)*](https://repl.it/@trsong/Flip-Surrounded-Regions-1)

</summary>
<div>

**Question:**  Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.
A region is captured by flipping all 'O's into 'X's in that surrounded region.
 
**Example:**
```py
X X X X
X O O X
X X O X
X O X X

After running your function, the board should be:

X X X X
X X X X
X X X X
X O X X

Explanation:
Surrounded regions shouldn’t be on the border, which means that any 'O' on the border of the board are not flipped to 'X'. Any 'O' that is not on the border and it is not connected to an 'O' on the border will be flipped to 'X'. Two cells are connected if they are adjacent cells connected horizontally or vertically.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Is Bipartite**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-22-2020-medium-is-bipartite) -- *Given an undirected graph G, check whether it is bipartite.* [*\(Try ME\)*](https://repl.it/@trsong/Is-a-Graph-Bipartite-1)

</summary>
<div>

**Question:** Given an undirected graph G, check whether it is bipartite. Recall that a graph is bipartite if its vertices can be divided into two independent sets, U and V, such that no edge connects vertices of the same set.

**Example:**
```py
is_bipartite(vertices=3, edges=[(0, 1), (1, 2), (2, 0)])  # returns False 
is_bipartite(vertices=2, edges=[(0, 1), (1, 0)])  # returns True. U = {0}. V = {1}. 
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 261. Graph Valid Tree**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-10-2020-lc-261-medium-graph-valid-tree) -- *Given n nodes and a list of undirected edges, write a function to check whether these edges make up a valid tree.* [*\(Try ME\)*](https://repl.it/@trsong/Graph-Valid-Tree-1)

</summary>
<div>

**Question:** Given n nodes labeled from 0 to n-1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.

**Example 1:**
```py
Input: n = 5, and edges = [[0,1], [0,2], [0,3], [1,4]]
Output: True
```

**Example 2:**
```py
Input: n = 5, and edges = [[0,1], [1,2], [2,3], [1,3], [1,4]]
Output: False
```

</div>
</details>



### Uniform-Cost Search / Dijkstra

<details>
<summary class="lc_m">

- [**\[Medium\] LC 743. Network Delay Time**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-8-2020-lc-743-medium-network-delay-time) -- *A network consists of nodes labeled 0 to N. Determine how long it will take for every node to receive a message that begins at node 0.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Network-Delay-Time-1)

</summary>
<div>

**Question:** A network consists of nodes labeled 0 to N. You are given a list of edges `(a, b, t)`, describing the time `t` it takes for a message to be sent from node `a` to node `b`. Whenever a node receives a message, it immediately passes the message on to a neighboring node, if possible.

Assuming all nodes are connected, determine how long it will take for every node to receive a message that begins at node 0.

**Example:** 
```py
given N = 5, and the following edges:

edges = [
    (0, 1, 5),
    (0, 2, 3),
    (0, 5, 4),
    (1, 3, 8),
    (2, 3, 1),
    (3, 5, 10),
    (3, 4, 5)
]

You should return 9, because propagating the message from 0 -> 2 -> 3 -> 4 will take that much time.
```

</div>
</details>

### A-Star Search
---

<details>
<summary class="lc_h">

- [**\[Hard\] Sliding Puzzle**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-4-2020-hard-sliding-puzzle) -- *An 8-puzzle is a game played on a 3 x 3 board of tiles, with the ninth tile missing. Design the board and solve the puzzle.* [*\(Try ME\)*](https://repl.it/@trsong/Solve-Sliding-Puzzle-1)

</summary>
<div>

**Question:**  An 8-puzzle is a game played on a 3 x 3 board of tiles, with the ninth tile missing. The remaining tiles are labeled 1 through 8 but shuffled randomly. Tiles may slide horizontally or vertically into an empty space, but may not be removed from the board.

Design a class to represent the board, and find a series of steps to bring the board to the state `[[1, 2, 3], [4, 5, 6], [7, 8, None]]`.

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Minimum Step to Reach One**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-26-2020-easy-minimum-step-to-reach-one) -- *Given a positive integer N, find the smallest number of steps it will take to reach 1.* [*\(Try ME\)*](https://repl.it/@trsong/Test-Find-Minimum-Step-to-Reach-One)

</summary>
<div>

**Question:** Given a positive integer `N`, find the smallest number of steps it will take to reach `1`.

There are two kinds of permitted steps:
- You may decrement `N` to `N - 1`.
- If `a * b = N`, you may decrement `N` to the larger of `a` and `b`.
 
For example, given `100`, you can reach `1` in five steps with the following route: `100 -> 10 -> 9 -> 3 -> 2 -> 1`.

</div>
</details>


## Backtracking
---

<details>
<summary class="lc_m">

- [**\[Medium\] Number of Android Lock Patterns**](https://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#mar-9-2021-medium-number-of-android-lock-patterns) -- *One way to unlock an Android phone is through a pattern of swipes across a 1-9 keypad.* [*\(Try ME\)*](https://repl.it/@trsong/Count-Number-of-Android-Lock-Patterns-1)

</summary>
<div>

**Question:** One way to unlock an Android phone is through a pattern of swipes across a 1-9 keypad.

For a pattern to be valid, it must satisfy the following:

- All of its keys must be distinct.
- It must not connect two keys by jumping over a third key, unless that key has already been used.

For example, 4 - 2 - 1 - 7 is a valid pattern, whereas 2 - 1 - 7 is not.

Find the total number of valid unlock patterns of length N, where 1 <= N <= 9.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Lazy Bartender**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#mar-8-2021-medium-lazy-bartender) -- *Given a dictionary input such as the one above, return the fewest number of drinks he must learn in order to satisfy all customers.* [*\(Try ME\)*](https://repl.it/@trsong/Lazy-Bartender-Problem-1)

</summary>
<div>

**Question:** At a popular bar, each customer has a set of favorite drinks, and will happily accept any drink among this set. 

For example, in the following situation, customer 0 will be satisfied with drinks 0, 1, 3, or 6.

```py
preferences = {
    0: [0, 1, 3, 6],
    1: [1, 4, 7],
    2: [2, 4, 7, 5],
    3: [3, 2, 5],
    4: [5, 8]
}
```

A lazy bartender working at this bar is trying to reduce his effort by limiting the drink recipes he must memorize. 

Given a dictionary input such as the one above, return the fewest number of drinks he must learn in order to satisfy all customers.

For the input above, the answer would be 2, as drinks 1 and 5 will satisfy everyone.


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 131. Palindrome Partitioning**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-21-2021-lc-131-medium-palindrome-partitioning) -- *Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.* [*\(Try ME\)*](https://repl.it/@trsong/Palindrome-Partitioning-1)

</summary>
<div>

**Question:** Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.

A palindrome string is a string that reads the same backward as forward.

**Example 1:**
```py
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]
```

**Example 2:**
```py
Input: s = "a"
Output: [["a"]]
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Anagram to Integer**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-5-2021-hard-anagram-to-integer) -- *Given a string formed by concatenating several words corresponding to the integers zero through nine and then anagramming,return the integer* [*\(Try ME\)*](https://repl.it/@trsong/Convert-Anagram-to-Integer-1)

</summary>
<div>

**Question:** You are given a string formed by concatenating several words corresponding to the integers `zero` through `nine` and then anagramming.

For example, the input could be `'niesevehrtfeev'`, which is an anagram of `'threefiveseven'`. Note that there can be multiple instances of each integer.

Given this string, return the original integers in sorted order. In the example above, this would be `357`.

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Map Digits to Letters*](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-1-2021-easy-map-digits-to-letters) -- *Given a mapping of digits to letters (as in a phone number), and a digit string, return all possible letters the number could represent.* [*\(Try ME\)*](https://repl.it/@trsong/Map-Digits-to-All-Possible-Letters-1)

</summary>
<div>

**Question:** Given a mapping of digits to letters (as in a phone number), and a digit string, return all possible letters the number could represent. You can assume each valid number in the mapping is a single digit.

**Example:**
```py
Input: {'2': ['a', 'b', 'c'], '3': ['d', 'e', 'f']}, '23'
Output: ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
```

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Power Set**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-21-2021-easy-power-set) -- *Given a set, generate its power set.* [*\(Try ME\)*](https://repl.it/@trsong/Power-Set-1)

</summary>
<div>

**Question:** The power set of a set is the set of all its subsets. Write a function that, given a set, generates its power set.

For example, given a set represented by a list `[1, 2, 3]`, it should return `[[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]` representing the power set.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 77. Combinations**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-30-2020-lc-77-medium-combinations) -- *Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.* [*\(Try ME\)*](https://repl.it/@trsong/Combinations-1)

</summary>
<div>

**Question:** Given two integers `n` and `k`, return all possible combinations of `k` numbers out of `1 ... n`.

**Example:**
```py
Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Permutations**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-29-2020-easy-permutations) -- *Given a number in the form of a list of digits, return all possible permutations.* [*\(Try ME\)*](https://repl.it/@trsong/Generate-Permutations-1)

</summary>
<div>

**Question:** Given a number in the form of a list of digits, return all possible permutations.

For example, given `[1,2,3]`, return `[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]`.

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] LC 301. Remove Invalid Parentheses**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-22-2020-lc-301-hard-remove-invalid-parentheses) -- *Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.* [*\(Try ME\)*](https://repl.it/@trsong/Ways-to-Remove-Invalid-Parentheses-1)

</summary>
<div>

**Question:** Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.

**Note:** The input string may contain letters other than the parentheses `(` and `)`.

**Example 1:**
```py
Input: "()())()"
Output: ["()()()", "(())()"]
```

**Example 2:**
```py
Input: "(a)())()"
Output: ["(a)()()", "(a())()"]
```

**Example 3:**
```py
Input: ")("
Output: [""]
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Knight's Tour Problem**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-3-2020-hard-knights-tour-problem) -- *A knight’s tour is a sequence of moves by a knight on a chessboard such that all squares are visited once.* [*\(Try ME\)*](https://repl.it/@trsong/Knights-Tour-Problem-1)

</summary>
<div>

**Question:** A knight's tour is a sequence of moves by a knight on a chessboard such that all squares are visited once.

Given N, write a function to return the number of knight's tours on an N by N chessboard.

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] LC 212. Word Search II**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-22-2020-lc-212-hard-word-search-ii) -- *Given an m x n board of characters and a list of strings words, return all words on the board.* [*\(Try ME\)*](https://repl.it/@trsong/Word-Search-II-1)

</summary>
<div>

**Question:** Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.
 
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


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] All Root to Leaf Paths in Binary Tree**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-10-2020-medium-all-root-to-leaf-paths-in-binary-tree) -- *Given a binary tree, return all paths from the root to leaves.* [*\(Try ME\)*](https://repl.it/@trsong/Print-All-Root-to-Leaf-Paths-in-Binary-Tree-1)

</summary>
<div>

**Question:** Given a binary tree, return all paths from the root to leaves.

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

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] LC 37. Sudoku Solver**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-13-2020-lc-37-hard-sudoku-solver) -- *Write a program to solve a Sudoku puzzle by filling the empty cells.* [*\(Try ME\)*](https://repl.it/@trsong/Sudoku-Solver-1)

</summary>
<div>

**Question:** Write a program to solve a Sudoku puzzle by filling the empty cells.

A sudoku solution must satisfy all of the following rules:
- Each of the digits `1-9` must occur exactly once in each row.
- Each of the digits `1-9` must occur exactly once in each column.
- Each of the the digits `1-9` must occur exactly once in each of the 9 `3x3` sub-boxes of the grid.
- Empty cells are indicated `0`.

**Example:**
```py
Input: [
    [3, 0, 6, 5, 0, 8, 4, 0, 0], 
    [5, 2, 0, 0, 0, 0, 0, 0, 0], 
    [0, 8, 7, 0, 0, 0, 0, 3, 1], 
    [0, 0, 3, 0, 1, 0, 0, 8, 0], 
    [9, 0, 0, 8, 6, 3, 0, 0, 5], 
    [0, 5, 0, 0, 9, 0, 6, 0, 0], 
    [1, 3, 0, 0, 0, 0, 2, 5, 0], 
    [0, 0, 0, 0, 0, 0, 0, 7, 4], 
    [0, 0, 5, 2, 0, 6, 3, 0, 0]
]

Possible output:
[
    [3, 1, 6, 5, 7, 8, 4, 9, 2],
    [5, 2, 9, 1, 3, 4, 7, 6, 8],
    [4, 8, 7, 6, 2, 9, 5, 3, 1],
    [2, 6, 3, 4, 1, 5, 9, 8, 7],
    [9, 7, 4, 8, 6, 3, 1, 2, 5],
    [8, 5, 1, 7, 9, 2, 6, 4, 3],
    [1, 3, 8, 9, 4, 7, 2, 5, 6],
    [6, 9, 2, 3, 5, 1, 8, 7, 4],
    [7, 4, 5, 2, 8, 6, 3, 1, 9]
]
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 78. Generate All Subsets**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-6-2020-lc-78-medium-generate-all-subsets) -- *Given a list of unique numbers, generate all possible subsets without duplicates. This includes the empty set as well.* [*\(Try ME\)*](https://repl.it/@trsong/Generate-All-the-Subsets-1)

</summary>
<div>

**Question:** Given a list of unique numbers, generate all possible subsets without duplicates. This includes the empty set as well.

**Example:**
```py
generate_all_subsets([1, 2, 3])
# [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Graph Coloring**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-3-2020-hard-graph-coloring) -- *Given an undirected graph and an integer k, determine whether color with k colors such that no two adjacent nodes share the same color.* [*\(Try ME\)*](https://repl.it/@trsong/k-Graph-Coloring-1)

</summary>
<div>

**Question:** Given an undirected graph represented as an adjacency matrix and an integer `k`, determine whether each node in the graph can be colored such that no two adjacent nodes share the same color using at most `k` colors.

</div>
</details>

<details>
<summary class="lc_e">

- [**\[Easy\] Generate All Possible Subsequences**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-7-2020-easy-generate-all-possible-subsequences) -- *Given a string, generate all possible subsequences of the string.* [*\(Try ME\)*](https://repl.it/@trsong/Generate-All-Possible-Subsequences-1)

</summary>
<div>

**Question:** Given a string, generate all possible subsequences of the string.

For example, given the string `xyz`, return an array or set with the following strings:

```py
x
y
z
xy
xz
yz
xyz
```

Note that `zx` is not a valid subsequence since it is not in the order of the given string.


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 93. All Possible Valid IP Address Combinations**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-23-2020-lc-93-medium-all-possible-valid-ip-address-combinations) -- *Given a string of digits, generate all possible valid IP address combinations.* [*\(Try ME\)*](https://repl.it/@trsong/Find-All-Possible-Valid-IP-Address-Combinations-1)

</summary>
<div>

**Question:** Given a string of digits, generate all possible valid IP address combinations.

IP addresses must follow the format A.B.C.D, where A, B, C, and D are numbers between `0` and `255`. Zero-prefixed numbers, such as `01` and `065`, are not allowed, except for `0` itself.

For example, given `"2542540123"`, you should return `['254.25.40.123', '254.254.0.123']`
 
</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] The N Queens Puzzle**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-2-2020-hard-the-n-queens-puzzle) -- *Given N, returns the number of possible arrangements of the board where N queens can be placed on the board without attacking each other.* [*\(Try ME\)*](https://repl.it/@trsong/Solve-the-N-Queen-Problem-1)

</summary>
<div>

**Question:** You have an N by N board. Write a function that, given N, returns the number of possible arrangements of the board where N queens can be placed on the board without threatening each other, i.e. no two queens share the same row, column, or diagonal.

</div>
</details>


## Dynamic Programming
---

### 1D DP
---

<details>
<summary class="lc_m">

- [**\[Medium\] LC 1155. Number of Dice Rolls With Target Sum**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-18-2021-lc-1155-medium-number-of-dice-rolls-with-target-sum) -- *Write a function that determines how many ways it is possible to throw N dice with some number of faces each to get a specific total.* [*\(Try ME\)*](https://repl.it/@trsong/Number-of-Dice-Rolls-With-Target-Sum-1)

</summary>
<div>

**Question:** You have `d` dice, and each die has `f` faces numbered `1, 2, ..., f`.

Return the number of possible ways (out of `f^d` total ways) modulo `10^9 + 7` to roll the dice so the sum of the face up numbers equals target.


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

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] LC 120. Max Path Sum in Triangle**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-5-2021-lc-120-easy-max-path-sum-in-triangle) -- *Given an array of arrays of integers, where each array corresponds to a row in a triangle of number, returns the weight of the maximum path.* [*\(Try ME\)*](https://repl.it/@trsong/Max-Path-Sum-in-Triangle-1)

</summary>
<div>

**Question:** You are given an array of arrays of integers, where each array corresponds to a row in a triangle of numbers. For example, `[[1], [2, 3], [1, 5, 1]]` represents the triangle:

```py
  1
 2 3
1 5 1
```

We define a path in the triangle to start at the top and go down one row at a time to an adjacent value, eventually ending with an entry on the bottom row. For example, `1 -> 3 -> 5`. The weight of the path is the sum of the entries.

Write a program that returns the weight of the maximum weight path.


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 279. Minimum Number of Squares Sum to N**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-4-2021-lc-279-medium-minimum-number-of-squares-sum-to-n) -- *Given a positive integer n, find the smallest number of squared integers which sum to n.* [*\(Try ME\)*](https://repl.it/@trsong/Minimum-Squares-Sum-to-N-1)

</summary>
<div>

**Question:** Given a positive integer n, find the smallest number of squared integers which sum to n.

For example, given `n = 13`, return `2` since `13 = 3^2 + 2^2 = 9 + 4`.
 
Given `n = 27`, return `3` since `27 = 3^2 + 3^2 + 3^2 = 9 + 9 + 9`.


</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] Multiset Partition**](https://repl.it/@trsong/Multiset-Partition-into-Equal-Sum) -- *Given a multiset of integers, return whether it can be partitioned into two subsets whose sums are the same.* [*\(Try ME\)*](https://repl.it/@trsong/Multiset-Partition-into-Equal-Sum-1)

</summary>
<div>

**Question:** Given a multiset of integers, return whether it can be partitioned into two subsets whose sums are the same.

For example, given the multiset `{15, 5, 20, 10, 35, 15, 10}`, it would return true, since we can split it up into `{15, 5, 10, 15, 10}` and `{20, 35}`, which both add up to `55`.

Given the multiset `{15, 5, 20, 10, 35}`, it would return `false`, since we can't split it up into two subsets that add up to the same sum.

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Smallest Sum Not Subset Sum**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-24-2020-easy-smallest-sum-not-subset-sum) -- *Given a sorted list of positive numbers, find the smallest positive number that cannot be a sum of any subset in the list.* [*\(Try ME\)*](https://repl.it/@trsong/Smallest-Sum-Not-Subset-Sum-1)

</summary>
<div>

**Question:** Given a sorted list of positive numbers, find the smallest positive number that cannot be a sum of any subset in the list.

**Example:**
```py
Input: [1, 2, 3, 8, 9, 10]
Output: 7
Numbers 1 to 6 can all be summed by a subset of the list of numbers, but 7 cannot.
```


</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LT 879. NBA Playoff Matches**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-23-2020-lt-879-medium-nba-playoff-matches) -- *Now, you're given n teams, and you need to output their final contest matches in the form of a string.* [*\(Try ME\)*](https://repl.it/@trsong/Print-NBA-Playoff-Matches-1)

</summary>
<div>

**Question:** During the NBA playoffs, we always arrange the rather strong team to play with the rather weak team, like make the rank 1 team play with the rank nth team, which is a good strategy to make the contest more interesting. Now, you're given n teams, and you need to output their final contest matches in the form of a string.
>
> The n teams are given in the form of positive integers from 1 to n, which represents their initial rank. (Rank 1 is the strongest team and Rank n is the weakest team.) We'll use parentheses () and commas , to represent the contest team pairing - parentheses () for pairing and commas , for partition. During the pairing process in each round, you always need to follow the strategy of making the rather strong one pair with the rather weak one.
>
> We ensure that the input n can be converted into the form `2^k`, where k is a positive integer.


**Example 1:**
```py
Input: 2
Output: "(1,2)"
```

**Example 2:**
```py
Input: 4
Output: "((1,4),(2,3))"
Explanation: 
  In the first round, we pair the team 1 and 4, the team 2 and 3 together, as we need to make the strong team and weak team together.
  And we got (1,4),(2,3).
  In the second round, the winners of (1,4) and (2,3) need to play again to generate the final winner, so you need to add the paratheses outside them.
  And we got the final answer ((1,4),(2,3)).
```

**Example 3:**
```py
Input: 8
Output: "(((1,8),(4,5)),((2,7),(3,6)))"
Explanation:
  First round: (1,8),(2,7),(3,6),(4,5)
  Second round: ((1,8),(4,5)),((2,7),(3,6))
  Third round: (((1,8),(4,5)),((2,7),(3,6)))
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Number of Flips to Make Binary String**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-15-2020-medium-number-of-flips-to-make-binary-string) -- *You are given a string consisting of the letters x and y, such as xyxxxyxyy. Find min flip so that all x goes before y.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Number-of-Flips-to-Make-Binary-String-1)

</summary>
<div>

**Question:** You are given a string consisting of the letters `x` and `y`, such as `xyxxxyxyy`. In addition, you have an operation called flip, which changes a single `x` to `y` or vice versa.

Determine how many times you would need to apply this operation to ensure that all x's come before all y's. In the preceding example, it suffices to flip the second and sixth characters, so you should return 2.

</div>
</details>


<details>
<summary class="lc_e">

- [**\[Easy\] Maximum Subarray Sum**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-5-2020-easy-maximum-subarray-sum) -- *Given array that may contain both positive and negative integers, find the sum of contiguous subarray of numbers which has the largest sum.* [*\(Try ME\)*](https://repl.it/@trsong/Maximum-Subarray-Sum-Divide-and-Conquer-1)

</summary>
<div>

**Question:** You are given a one dimensional array that may contain both positive and negative integers, find the sum of contiguous subarray of numbers which has the largest sum.

For example, if the given array is `[-2, -5, 6, -2, -3, 1, 5, -6]`, then the maximum subarray sum is 7 as sum of `[6, -2, -3, 1, 5]` equals 7

Solve this problem with Divide and Conquer as well as DP separately.


</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Largest Sum of Non-adjacent Numbers**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-17-2020-hard-largest-sum-of-non-adjacent-numbers) -- *Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0 or negative.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Largest-Sum-of-Non-adjacent-Numbers-1)

</summary>
<div>

**Question:** Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0 or negative.

For example, `[2, 4, 6, 2, 5]` should return 13, since we pick 2, 6, and 5. `[5, 1, 1, 5]` should return 10, since we pick 5 and 5.

Follow-up: Can you do this in O(N) time and constant space?

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Delete Columns to Make Sorted II**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-15-2020-medium-delete-columns-to-make-sorted-ii) -- *Given 2D matrix, the task is to count the number of columns to be deleted so that all the rows are lexicographically sorted.* [*\(Try ME\)*](https://repl.it/@trsong/Delete-Columns-to-Make-Row-Sorted-II-1)

</summary>
<div>

**Question:** You are given an N by M 2D matrix of lowercase letters. The task is to count the number of columns to be deleted so that all the rows are lexicographically sorted.

**Example 1:**
```
Given the following table:
hello
geeks

Your function should return 1 as deleting column 1 (index 0)
Now both strings are sorted in lexicographical order:
ello
eeks
```

**Example 2:**
```
Given the following table:
xyz
lmn
pqr

Your function should return 0. All rows are already sorted lexicographically.
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Largest Divisible Pairs Subset**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-28-2020-hard-largest-divisible-pairs-subset) -- *Given a set of distinct positive integers, find the largest subset such that every pair of elements has divisible counterpart in the subset.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Largest-Divisible-Pairs-Subset-1)

</summary>
<div>

**Question:** Given a set of distinct positive integers, find the largest subset such that every pair of elements in the subset `(i, j)` satisfies either `i % j = 0` or `j % i = 0`.

For example, given the set `[3, 5, 10, 20, 21]`, you should return `[5, 10, 20]`. Given `[1, 3, 6, 24]`, return `[1, 3, 6, 24]`.

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 139. Word Break**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#sep-9-2020-lc-139-medium-word-break) -- *Given a non-empty string s and a dictionary with a list of non-empty words, determine if s can be segmented into a space-separated sequence.* [*\(Try ME\)*](https://repl.it/@trsong/Word-Break-Problem-1)

</summary>
<div>

**Question:** Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words.

**Note:**
* The same word in the dictionary may be reused multiple times in the segmentation.
* You may assume the dictionary does not contain duplicate words.

**Example 1:**
```py
Input: s = "Pseudocode", wordDict = ["Pseudo", "code"]
Output: True
Explanation: Return true because "Pseudocode" can be segmented as "Pseudo code".
```

**Example 2:**
```py
Input: s = "applepenapple", wordDict = ["apple", "pen"]
Output: True
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
             Note that you are allowed to reuse a dictionary word.
```

**Example 3:**
```py
Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
Output: False
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 91. Decode Ways**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-19-2020-lc-91-medium-decode-ways) -- *A message containing letters from A-Z is being encoded to numbers using the following mapping: 'A' -> 1 'B' -> 2 ... 'Z' -> 26.* [*\(Try ME\)*](https://repl.it/@trsong/Number-of-Decode-Ways-1)

</summary>
<div>

**Question:** A message containing letters from `A-Z` is being encoded to numbers using the following mapping:

 ```py
 'A' -> 1
 'B' -> 2
 ...
 'Z' -> 26
 ```
 Given an encoded message containing digits, determine the total number of ways to decode it.

**Example 1:**
```py
Input: "12"
Output: 2
Explanation: It could be decoded as AB (1 2) or L (12).
```

**Example 2:**
```py
Input: "10"
Output: 1
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] Making Changes**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-9-2020-medium-making-changes) -- *Given a list of possible coins in cents, and an amount (in cents) n, return the minimum number of coins needed to create the amount n.* [*\(Try ME\)*](https://repl.it/@trsong/Making-Changes-Problem-1)

</summary>
<div>

**Question:** Given a list of possible coins in cents, and an amount (in cents) n, return the minimum number of coins needed to create the amount n. If it is not possible to create the amount using the given coin denomination, return None.

**Example:**
```py
make_change([1, 5, 10, 25], 36)  # gives 3 coins (25 + 10 + 1) 
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Reverse Coin Change**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#dec-18-2020-medium-reverse-coin-change) -- *Given an array represents the number of ways we can produce `i` units of change, determine the denominations that must be in use.* [*\(Try ME\)*](https://repl.it/@trsong/Reverse-Coin-Change-1)

</summary>
<div>

**Question:** You are given an array of length `N`, where each element `i` represents the number of ways we can produce `i` units of change. For example, `[1, 0, 1, 1, 2]` would indicate that there is only one way to make `0, 2, or 3` units, and two ways of making `4` units.

Given such an array, determine the denominations that must be in use. In the case above, for example, there must be coins with value `2, 3, and 4`.


</div>
</details>


### 2D DP
---

<details>
<summary class="lc_h">

- [**\[Hard\] Maximize Sum of the Minimum of K Subarrays**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#mar-4-2021-hard-maximize-sum-of-the-minimum-of-k-subarrays) -- *Given an array a of size N and an integer K, divide the array into K segments such that sum of the minimum of K segments is maximized.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Maximize-Sum-of-the-Minimum-of-K-Subarrays-1)

</summary>
<div>

**Question:** Given an array a of size N and an integer K, the task is to divide the array into K segments such that sum of the minimum of K segments is maximized.

**Example1:**
```py
Input: [5, 7, 4, 2, 8, 1, 6], K = 3
Output: 13
Divide the array at indexes 0 and 1. Then the segments are [5], [7], [4, 2, 8, 1, 6] Sum of the minimus is 5 + 7 + 1 = 13
```

**Example2:**
```py
Input: [6, 5, 3, 8, 9, 10, 4, 7, 10], K = 4
Output: 27
[6, 5, 3, 8, 9], [10], [4, 7], [10] => 3 + 10 + 4 + 10 = 27
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Minimum Palindrome Substring**](https://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-20-2021-hard-minimum-palindrome-substring) -- *Given a string, split it into as few strings as possible such that each string is a palindrome.* [*\(Try ME\)*](https://repl.it/@trsong/Minimum-Palindrome-Substring-1)

</summary>
<div>

**Question:** Given a string, split it into as few strings as possible such that each string is a palindrome.

For example, given the input string `"racecarannakayak"`, return `["racecar", "anna", "kayak"]`.

Given the input string `"abc"`, return `["a", "b", "c"]`.

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Partition Array to Reach Mininum Difference**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-25-2021-hard-partition-array-to-reach-mininum-difference) -- *Given an array of positive integers, divide the array into two subsets such that the difference between the sum of the subsets reaches min.* [*\(Try ME\)*](https://repl.it/@trsong/Partition-Array-to-Reach-Min-Difference-1)

</summary>
<div>

**Question:** Given an array of positive integers, divide the array into two subsets such that the difference between the sum of the subsets is as small as possible.

For example, given `[5, 10, 15, 20, 25]`, return the sets `[10, 25]` and `[5, 15, 20]`, which has a difference of `5`, which is the smallest possible difference. 

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Longest Common Subsequence of 3 Strings**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#jan-13-2021-hard-longest-common-subsequence-of-3-strings) -- *Write a program that computes the length of the longest common subsequence of three given strings.* [*\(Try ME\)*](https://repl.it/@trsong/Longest-Common-Subsequence-of-3-Strings-1)

</summary>
<div>

**Question:** Write a program that computes the length of the longest common subsequence of three given strings. 
 
For example, given `"epidemiologist"`, `"refrigeration"`, and `"supercalifragilisticexpialodocious"`, it should return `5`, since the longest common subsequence is `"eieio"`.


</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] Non-adjacent Subset Sum**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-18-2020-hard-non-adjacent-subset-sum) -- *Given an array of unique positive numbers. Find non-adjacent subset of elements sum up to k.* [*\(Try ME\)*](https://repl.it/@trsong/Non-adjacent-Subset-Sum-1)

</summary>
<div>

**Question:** Given an array of size n with unique positive integers and a positive integer K,
check if there exists a combination of elements in the array satisfying both of below constraints:
- The sum of all such elements is K
- None of those elements are adjacent in the original array

**Example:**
```py
Input: K = 14, arr = [1, 9, 8, 3, 6, 7, 5, 11, 12, 4]
Output: [3, 7, 4]
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Max Value of Coins to Collect in a Matrix**](https://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-12-2020-medium-max-value-of-coins-to-collect-in-a-matrix) -- *You are given a 2-d matrix. Find the maximum number of coins you can collect by the bottom right corner.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Max-Value-of-Coins-to-Collect-in-a-Matrix-1)

</summary>
<div>

**Question:** You are given a 2-d matrix where each cell represents number of coins in that cell. Assuming we start at `matrix[0][0]`, and can only move right or down, find the maximum number of coins you can collect by the bottom right corner.

**Example:**

```py
Given below matrix:

0 3 1 1
2 0 0 4
1 5 3 1

The most we can collect is 0 + 2 + 1 + 5 + 3 + 1 = 12 coins.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Largest Square**](http://trsong.github.io/python/java/2020/11/02/DailyQuestionsAug/#nov-7-2020-medium-largest-square) -- *Given matrix with 1 and 0. Find the largest square matrix containing only 1’s and return its dimension size.* [*\(Try ME\)*](https://repl.it/@trsong/Largest-Square-1)

</summary>
<div>

**Question:** Given an N by M matrix consisting only of 1's and 0's, find the largest square matrix containing only 1's and return its dimension size.

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

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] LC 76. Minimum Window Substring**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-12-2020-lc-76-hard-minimum-window-substring) -- *Given a string and a set of characters, return the shortest substring containing all the characters in the set.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Minimum-Window-Substring-1)

</summary>
<div>

**Question:** Given a string and a set of characters, return the shortest substring containing all the characters in the set.

For example, given the string `"figehaeci"` and the set of characters `{a, e, i}`, you should return `"aeci"`.

If there is no substring containing all the characters in the set, return null.
 
</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] LC 727. Minimum Window Subsequence**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-15-2020-lc-727-hard-minimum-window-subsequence) -- *Given strings S and T, find the minimum (contiguous) substring W of S, so that T is a subsequence of W.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Minimum-Window-Subsequence-1)

</summary>
<div>

**Question:** Given strings S and T, find the minimum (contiguous) substring W of S, so that T is a subsequence of W.

If there is no such window in S that covers all characters in T, return the empty string "". If there are multiple such minimum-length windows, return the one with the left-most starting index.

**Example:**
```py
Input: 
S = "abcdebdde", T = "bde"
Output: "bcde"

Explanation: 
"bcde" is the answer because it occurs before "bdde" which has the same length.
"deb" is not a smaller window because the elements of T in the window must occur in order.
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] Longest Common Subsequence**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-18-2020-medium-longest-common-subsequence) -- *Given two sequences, find the length of longest subsequence present in both of them.* [*\(Try ME\)*](https://repl.it/@trsong/Find-the-Longest-Common-Subsequence-1)

</summary>
<div>

**Question:** Given two sequences, find the length of longest subsequence present in both of them. 

A subsequence is a sequence that appears in the same relative order, but not necessarily contiguous.

**Example 1:**

```py
Input:  "ABCD" and "EDCA"
Output:  1
	
Explanation:
LCS is 'A' or 'D' or 'C'
```

**Example 2:**

```py
Input: "ABCD" and "EACB"
Output:  2
	
Explanation: 
LCS is "AC"
```

</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] K-Palindrome**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-23-2021-hard-k-palindrome) -- *Given a string which we can delete at most k, return whether you can make a palindrome.* [*\(Try ME\)*](https://repl.it/@trsong/Find-K-Palindrome-1)

</summary>
<div>

**Question:** Given a string which we can delete at most k, return whether you can make a palindrome.

For example, given `'waterrfetawx'` and a k of 2, you could delete f and x to get `'waterretaw'`.

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] LC 446. Count Arithmetic Subsequences**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#june-22-2020-lc-446-medium-count-arithmetic-subsequences) -- *Given an array of n positive integers. The task is to count the number of Arithmetic Subsequence in the array.* [*\(Try ME\)*](https://repl.it/@trsong/Count-Number-of-Arithmetic-Subsequences-1)

</summary>
<div>

**Question:** Given an array of n positive integers. The task is to count the number of Arithmetic Subsequence in the array. Note: Empty sequence or single element sequence is also Arithmetic Sequence. 

**Example 1:**
```py
Input : arr[] = [1, 2, 3]
Output : 8
Arithmetic Subsequence from the given array are:
[], [1], [2], [3], [1, 2], [2, 3], [1, 3], [1, 2, 3].
```

**Example 2:**
```py
Input : arr[] = [10, 20, 30, 45]
Output : 12
```

**Example 3:**
```py
Input : arr[] = [1, 2, 3, 4, 5]
Output : 23
```

</div>
</details>


<details>
<summary class="lc_m">

- [**\[Medium\] LC 718. Longest Common Sequence of Browsing Histories**](https://trsong.github.io/python/java/2020/05/02/DailyQuestionsMay/#jul-9-2020-lc-718-medium-longest-common-sequence-of-browsing-histories) -- *Write a function that takes two users’ browsing histories as input and returns the longest contiguous sequence of URLs that appear in both.* [*\(Try ME\)*](https://repl.it/@trsong/Longest-Common-Sequence-of-Browsing-Histories-1)

</summary>
<div>

**Question:** We have some historical clickstream data gathered from our site anonymously using cookies. The histories contain URLs that users have visited in chronological order.

Write a function that takes two users' browsing histories as input and returns the longest contiguous sequence of URLs that appear in both.

For example, given the following two users' histories:

```py
user1 = ['/home', '/register', '/login', '/user', '/one', '/two']
user2 = ['/home', '/red', '/login', '/user', '/one', '/pink']
```

You should return the following:

```py
['/login', '/user', '/one']
```

</div>
</details>


### DP+ 
---

<details>
<summary class="lc_h">

- [**\[Hard\] Max Path Value in Directed Graph**](http://trsong.github.io/python/java/2021/02/02/DailyQuestionsFeb/#feb-13-2021-hard-max-path-value-in-directed-graph) -- *Given a directed graph, return the largest value path of the graph. Define a path’s value as the number of most frequent letters along path.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Max-Letter-Path-Value-in-Directed-Graph-1)

</summary>
<div>

**Question:** In a directed graph, each node is assigned an uppercase letter. We define a path's value as the number of most frequently-occurring letter along that path. For example, if a path in the graph goes through "ABACA", the value of the path is 3, since there are 3 occurrences of 'A' on the path.

Given a graph with n nodes and m directed edges, return the largest value path of the graph. If the largest value is infinite, then return null.

The graph is represented with a string and an edge list. The i-th character represents the uppercase letter of the i-th node. Each tuple in the edge list (i, j) means there is a directed edge from the i-th node to the j-th node. Self-edges are possible, as well as multi-edges.

**Example1:** 
```py
Input: "ABACA", [(0, 1), (0, 2), (2, 3), (3, 4)]
Output: 3
Explanation: maximum value 3 using the path of vertices [0, 2, 3, 4], ie. A -> A -> C -> A.
```

**Example2:**
```py
Input: "A", [(0, 0)]
Output: None
Explanation: we have an infinite loop.
```

</div>
</details>

<details>
<summary class="lc_h">

- [**\[Hard\] Largest Rectangle**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-29-2020-hard-largest-rectangle) -- *Given an N by M matrix consisting only of 1’s and 0’s, find the largest rectangle containing only 1’s and return its area.* [*\(Try ME\)*](https://repl.it/@trsong/Largest-Rectangle-in-a-Grid-1)

</summary>
<div>

**Question:** Given an N by M matrix consisting only of 1's and 0's, find the largest rectangle containing only 1's and return its area.

**For Example:**

```py
Given the following matrix:

[[1, 0, 0, 0],
 [1, 0, 1, 1],
 [1, 0, 1, 1],
 [0, 1, 0, 0]]

Return 4. As the following 1s form the largest rectangle containing only 1s:
 [1, 1],
 [1, 1]
```

</div>
</details>


<details>
<summary class="lc_h">

- [**\[Hard\] Find Arbitrage Opportunities**](http://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#oct-3-2020--hard-find-arbitrage-opportunities) -- *Given a table of currency exchange rates, represented as a 2D array. Determine whether there is a possible arbitrage.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Arbitrage-Opportunities-for-Currency-Exchange-1)

</summary>
<div>

**Question:** Suppose you are given a table of currency exchange rates, represented as a 2D array. Determine whether there is a possible arbitrage: that is, whether there is some sequence of trades you can make, starting with some amount A of any currency, so that you can end up with some amount greater than A of that currency.
>
> There are no transaction costs and you can trade fractional quantities.

**Example:**
```py
Given the following matrix:
#       RMB,   USD,  CAD
# RMB     1, 0.14, 0.19
# USD  6.97,    1,  1.3
# CAD  5.37, 0.77,    1

# Since RMB -> CAD -> RMB:  1 Yuan * 0.19 * 5.37 = 1.02 Yuan
# If we keep exchange RMB to CAD and exchange back, we can make a profit eventually.
```

</div>
</details>

<details>
<summary class="lc_m">

- [**\[Medium\] Maximum Circular Subarray Sum**](https://trsong.github.io/python/java/2020/08/02/DailyQuestionsAug/#aug-2-2020-medium-maximum-circular-subarray-sum) -- *Given a circular array, compute its maximum subarray sum in O(n) time. A subarray can be empty, and in this case the sum is 0.* [*\(Try ME\)*](https://repl.it/@trsong/Find-Maximum-Circular-Subarray-Sum-1)

</summary>
<div>

**Question:** Given a circular array, compute its maximum subarray sum in `O(n)` time. A subarray can be empty, and in this case the sum is 0.

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


</div>
</details>


{::options parse_block_html="false" /}
