---
layout: post
title:  "Daily Coding Problems Aug to Oct"
date:   2019-08-01 22:22:32 -0700
categories: Python/Java
---
* This will become a table of contents (this text will be scraped).
{:toc}

## Daily Coding Problems

### Enviroment Setup
---

**Python 2.7 Playground:** [https://repl.it/languages/python](https://repl.it/languages/python)

<!--
### Jul 11, 2019 \[Medium\]
---
> **Question:** A rule looks like this:

```
A NE B
``` 
> This means this means point A is located northeast of point B.
> 
```
A SW C
```
> means that point A is southwest of C.
>
>Given a list of rules, check if the sum of the rules validate. For example:

```
A N B
B NE C
C N A
```

> does not validate, since A cannot be both north and south of C.

```
A NW B
A N B
```

> is considered valid.

--->
### Sep 2, 2019 LC 937 \[Easy\] Reorder Log Files
---
> **Question:** You have an array of logs. Each log is a space delimited string of words.
>
> For each log, the first word in each log is an alphanumeric identifier.  Then, either:
>
> - Each word after the identifier will consist only of lowercase letters, or;
> - Each word after the identifier will consist only of digits.
We will call these two varieties of logs letter-logs and digit-logs.  It is guaranteed that each log has at least one word after its identifier.
>
> Reorder the logs so that all of the letter-logs come before any digit-log.  The letter-logs are ordered lexicographically ignoring identifier, with the identifier used in case of ties.  The digit-logs should be put in their original order.
>
> Return the final order of the logs.

**Example:**

```py
Input: ["a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo"]
Output: ["g1 act car","a8 act zoo","ab1 off key dog","a1 9 2 3 1","zo4 4 7"]
```

### Sep 1, 2019 LT 892 \[Medium\] Alien Dictionary
---
> **Question:** There is a new alien language which uses the latin alphabet. However, the order among letters are unknown to you. You receive a list of non-empty words from the dictionary, where words are sorted lexicographically by the rules of this new language. Derive the order of letters in this language.
>
> - You may assume all letters are in lowercase.
> 
> - You may assume that if a is a prefix of b, then a must appear before b in the given dictionary.
> 
> - If the order is invalid, return an empty string.
There may be multiple valid order of letters, return the smallest in normal lexicographical order

**Example 1:**
```py
Input: ["wrt", "wrf", "er", "ett", "rftt"]
Output: "wertf"
Explanation：
from "wrt" and "wrf", we can get 't'<'f'
from "wrt" and "er", we can get 'w'<'e'
from "er" and "ett", we can get 'r'<'t'
from "ett" and "rtff", we can get 'e'<'r'
So return "wertf"
```

**Example 2:**
```py
Input: ["z", "x"]
Output: "zx"
Explanation：
from "z" and "x"，we can get 'z' < 'x'
So return "zx"
```

### Aug 31, 2019 \[Hard\] Encode and Decode Array of Strings
---
> **Question:** Given an array of string, write function "encode" to convert an array of strings into a single string and function "decode" to restore the original array.
> 
> Hint: string can be encoded as `<length>:<contents>`
> 
> Follow-up: what about array of integers, strings, and dictionaries?

**My thoughts:** There are many ways to encode/decode. Our solution use BEncode, the encoding used by BitTorrent. 

According to BEncode Wiki: [https://en.wikipedia.org/wiki/Bencode](https://en.wikipedia.org/wiki/Bencode) and specification of BitTorrent: [http://www.bittorrent.org/beps/bep_0003.html](http://www.bittorrent.org/beps/bep_0003.html). BEncode works as following:

- Integer num is encoded as `"i{num}e"`. eg. `42 => "i42e"`, `-32 => "i-32e"`, `0 => "i0e"`
- String s is encoded as `"{len(s)}:{s}"`. eg. `"abc" => "3:abc"`, `"" => "0:"`, `"s" => "1:s"`, `"doge" => "4:doge"`
- List lst is encoded as `"l{encode(lst[0])}{encode(lst[1])}....{encode(lst[n-1])}e"`.  e.g. `[] => "le"`, `[42, "cat"]` => `"li42e3:cate"`, `[[11], [22], [33]] => lli11eeli22eeli33eee`
- Dictionary d is encoded as `"d{encode(key1)}{encode(val1)}...{encode(key_n)}{encode(val_n)}e"` e.g. `{'bar': 'spam','foo': 42} => "d3:bar4:spam3:fooi42ee"`


**Solution with BEncode:** [https://repl.it/@trsong/Encode-and-Decode-Array-of-Strings](https://repl.it/@trsong/Encode-and-Decode-Array-of-Strings)
```py
import unittest

"""
Encode Utilities
"""
def encode_int(num):
    return ["i", str(num), "e"]

def encode_str(string):
    return [str(len(string)), ":", string]

def encode_list(lst):
    res = ["l"]
    for e in lst:
        res.extend(encode_func[type(e)](e))
    res.append("e")
    return res

def encode_dict(dicionary):
    res = ["d"]
    for k, v in sorted(dicionary.items()):
        res.extend(encode_str(k))
        res.extend(encode_func[type(v)](v))
    res.append("e")
    return res

encode_func = {
    int: encode_int,
    str: encode_str,
    list: encode_list,
    dict: encode_dict
}

"""
Decode Utilities
"""
def decode_int(encoded, pos):
    pos += 1
    new_pos = encoded.index('e', pos)
    num = int(encoded[pos:new_pos])
    return (num, new_pos + 1)

def decode_str(encoded, pos):
    colon = encoded.index(':', pos)
    n = int(encoded[pos:colon])
    str_start = colon + 1
    return (encoded[str_start:str_start + n], str_start + n)

def decode_list(encoded, pos):
    res = []
    pos += 1
    while encoded[pos] != 'e':
        val, new_pos = decode_func[encoded[pos]](encoded, pos)
        pos = new_pos
        res.append(val)
    return (res, pos + 1)

def decode_dict(encoded, pos):
    res = {}
    pos += 1
    while encoded[pos] != 'e':
        key, key_end_pos = decode_str(encoded, pos)
        val, val_end_pos = decode_func[encoded[key_end_pos]](encoded, key_end_pos)
        pos = val_end_pos
        res[key] = val
    return (res, pos + 1)

decode_func = {
    'l': decode_list,
    'd': decode_dict,
    'i': decode_int
}
decode_func.update({
    chr(ord('0') + i): decode_str for i in xrange(10) # 0-9 all use decode_str
})


class BEncode(object):
    @staticmethod
    def encode(obj):
        return "".join(encode_func[type(obj)](obj))

    
    @staticmethod
    def decode(encoded):
        obj, _ = decode_func[encoded[0]](encoded, 0)
        return obj


class BEncodeSpec(unittest.TestCase):
    def assert_BEncode_result(self, obj):
        encoded_str = BEncode.encode(obj)
        decoded_obj = BEncode.decode(encoded_str)
        self.assertEqual(decoded_obj, obj)

    def test_positive_integer(self):
        self.assert_BEncode_result(42)  # encode as "i42e"
    
    def test_negative_integer(self):
        self.assert_BEncode_result(-32)  # encode as "i-32e"

    def test_zero(self):
        self.assert_BEncode_result(0)  # encode as "i0e"

    def test_empty_string(self):
        self.assert_BEncode_result("")  # encode as "0:""

    def test_string_with_whitespaces(self):
        self.assert_BEncode_result(" ")  # encode as "1: "
        self.assert_BEncode_result(" a ")  # encode as "3: a "
        self.assert_BEncode_result("a b  c   1 2 3 ")  # encode as "15:a b  c   1 2 3 "

    def test_string_with_delimiter_characters(self):
        self.assert_BEncode_result("i42ei42e")  # encode as "8:i42ei42e"
        self.assert_BEncode_result("4:spam")  # encode as "6:4:spam"
        self.assert_BEncode_result("d3:bar4:spam3:fooi42ee")  # encode as "22:d3:bar4:spam3:fooi42ee"
    
    def test_string_with_special_characters(self):
        self.assert_BEncode_result("!@#$%^&*(){}[]|\;:'',.?/`~") # encode as "26:!@#$%^&*(){}[]|\;:'',.?/`~"

    def test_empty_list(self):
        self.assert_BEncode_result([])  # encode as "le"

    def test_list_of_empty_strings(self):
        self.assert_BEncode_result(["", "", ""])  # encode as "l0:0:0:e"

    def test_nested_empty_lists(self):
        self.assert_BEncode_result([[], [[]], [[[]]]]) # encoded as "llelleellleee"

    def test_list_of_strings(self):
        self.assert_BEncode_result(['a', '', 'abc']) # encode as "l1:a0:3:abce"

    def test_nested_lists(self):
        self.assert_BEncode_result([0, ["a", 1], "ab", [[2], "c"]]) # encode as "li0el1:ai1ee2:ablli2ee1:cee"

    def test_empty_dictionary(self):
        self.assert_BEncode_result({}) # encode as "de"

    def test_dictionary(self):
        self.assert_BEncode_result({
            'bar': 'spam',
            'foo': 42
        })  # encode as "d3:bar4:spam3:fooi42ee"

    def test_nested_dictionary(self):
        self.assert_BEncode_result([
            "s",
            42, 
            { 
                'a': 12,
                'list': [
                    'b', 
                    {
                        'c': [[1], 2, [3, [4]]],
                        'd': 12
                    }]
            }]) # encode as 'l1:si42ed1:ai12e4:listl1:bd1:clli1eei2eli3eli4eeee1:di12eeeee'
 

if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 30, 2019 LT 623 \[Hard\] K Edit Distance
---
> **Question:** Given a set of strings which just has lower case letters and a target string, output all the strings for each the edit distance with the target no greater than k.
You have the following 3 operations permitted on a word:
> - Insert a character
> - Delete a character
> - Replace a character

**Example 1:**
```py
Given words = ["abc", "abd", "abcd", "adc"] and target = "ac", k = 1
Return ["abc", "adc"]
Explanation:
- "abc" remove "b"
- "adc" remove "d"
```

**Example 2:**
```py
Given words = ["acc","abcd","ade","abbcd"] and target = "abc", k = 2
Return ["acc","abcd","ade","abbcd"]
Explanation:
- "acc" turns "c" into "b"
- "abcd" remove "d"
- "ade" turns "d" into "b" turns "e" into "c"
- "abbcd" gets rid of "b" and "d"
```

**My thoughts:** The brutal force way is to calculate the distance between each word and target and filter those qualified words. However, notice that word might have exactly the same prefix and that share the same DP array. So we can build a prefix tree that contains all words and calculate the DP array along the way.

**Solution with Trie and DFS:** [https://repl.it/@trsong/K-Edit-Distance](https://repl.it/@trsong/K-Edit-Distance)
```py
import unittest

class Trie(object):
    def __init__(self):
        self.count = 0
        self.word = None
        self.edit_distance_dp = None
        self.children = None

    def insert(self, word): 
        t = self
        for char in word:
            if not t.children:
                t.children = {}
            if char not in t.children:
                t.children[char] = Trie()
            t = t.children[char]
        t.count += 1
        t.word = word


def filter_k_edit_distance(words, target, k):
    trie = Trie()
    n = len(target)
    filtered_word = filter(lambda word: n - k <= len(word) <= n + k, words)
    for word in filtered_word:
        trie.insert(word)
        
    trie.edit_distance_dp = [i for i in xrange(n+1)] # edit distance between "" and target[:i] equals i (insert i letters)
    stack = [trie]
    res = []
    while stack:
        parent = stack.pop()

        parent_dp = parent.edit_distance_dp
        if parent.word is not None and parent_dp[n] <= k:
            res.extend([parent.word] * parent.count)

        if not parent.children:
            continue

        for char, child in parent.children.items():
            dp = [0] * (n+1)
            dp[0] = parent_dp[0] + 1
            for j in xrange(1, n+1):
                if char == target[j-1]:
                    dp[j] = parent_dp[j-1]
                else:
                    dp[j] = min(1 + parent_dp[j-1], 1 + dp[j-1], 1 + parent_dp[j])
            child.edit_distance_dp = dp
            stack.append(child)
    
    return res


class FilterKEditDistance(unittest.TestCase):
    def assert_k_distance_array(self, res, expected):
        self.assertEqual(sorted(res), sorted(expected))

    def test_example1(self):
        words =["abc", "abd", "abcd", "adc"] 
        target = "ac"
        k = 1
        expected = ["abc", "adc"]
        self.assert_k_distance_array(filter_k_edit_distance(words, target, k), expected)
    
    def test_example2(self):
        words = ["acc","abcd","ade","abbcd"]
        target = "abc"
        k = 2
        expected = ["acc","abcd","ade","abbcd"]
        self.assert_k_distance_array(filter_k_edit_distance(words, target, k), expected)

    def test_duplicated_words(self):
        words = ["a","b","a","c", "bb", "cc"]
        target = ""
        k = 1
        expected = ["a","b","a","c"]
        self.assert_k_distance_array(filter_k_edit_distance(words, target, k), expected)

    def test_empty_words(self):
        words = ["", "", "", "c", "bbbbb", "cccc"]
        target = "ab"
        k = 2
        expected = ["", "", "", "c"]
        self.assert_k_distance_array(filter_k_edit_distance(words, target, k), expected)

    def test_same_word(self):
        words = ["ab", "ab", "ab"]
        target = "ab"
        k = 1000
        expected = ["ab", "ab", "ab"]
        self.assert_k_distance_array(filter_k_edit_distance(words, target, k), expected)

    def test_unqualified_words(self):
        words = ["", "a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa"]
        target = "aaaaa"
        k = 2
        expected = ["aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa"]
        self.assert_k_distance_array(filter_k_edit_distance(words, target, k), expected)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 29, 2019 \[Easy\] Flip Bit to Get Longest Sequence of 1s
---
> **Question:** Given an integer, can you flip exactly one bit from a 0 to a 1 to get the longest sequence of 1s? Return the longest possible length of 1s after flip.

**Example:**
```py
Input: 183 (or binary: 10110111)
Output: 6
Explanation: 10110111 => 10111111. The longest sequence of 1s is of length 6.
```

**Solution:** [https://repl.it/@trsong/Flip-Bit-to-Get-Longest-Sequence-of-1s](https://repl.it/@trsong/Flip-Bit-to-Get-Longest-Sequence-of-1s)
```py
import unittest

def flip_bits(num):
    prev = 0
    cur = 0
    max_len = 0
    while num > 0:
        last_digit = num & 1
        if last_digit == 1:
            cur += 1
        else:
            max_len = max(max_len, cur + prev + 1)
            prev = cur
            cur = 0
        num >>= 1
    return max(max_len, cur + prev + 1)


class FlipBitSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(flip_bits(0b10110111), 6)  # 10110111 => 10111111

    def test_not_exist_ones(self):
        self.assertEqual(flip_bits(0), 1)  # 0 => 1

    def test_flip_last_digit(self):
        self.assertEqual(flip_bits(0b100110), 3)  # 100110 => 100111

    def test_three_zeros(self):
        self.assertEqual(flip_bits(0b1011110110111), 7)  # 1011110110111 => 1011111110111

    def test_one(self):
        self.assertEqual(flip_bits(1), 2)  # 01 => 11


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 28, 2019 LC 103 \[Medium\] Binary Tree Zigzag Level Order Traversal
---
> **Question:** Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

**For example:**
```py
Given following binary tree:
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]
```

**Solution with BFS and Two Stacks:** [https://repl.it/@trsong/Binary-Tree-Zigzag-Level-Order-Traversal](https://repl.it/@trsong/Binary-Tree-Zigzag-Level-Order-Traversal)
```py
import unittest

class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def zig_zag_level_order(root):
    if not root:
        return []
    stack = [root]
    res = []
    is_left_first = True
    while stack:
        level = []
        next_stack = []
        for _ in xrange(len(stack)):
            cur = stack.pop()
            if cur:
                level.append(cur.val)
                if is_left_first:
                    next_stack.append(cur.left)
                    next_stack.append(cur.right)
                else:
                    next_stack.append(cur.right)
                    next_stack.append(cur.left)
        stack = next_stack
        is_left_first ^= True # flip the boolean flag
        if level:
            res.append(level)
    return res


class ZigZagLevelOrderSpec(unittest.TestCase):
    def test_example(self):
        """
            3
           / \
          9  20
            /  \
           15   7
        """
        n20 = Node(20, Node(15), Node(7))
        n3 = Node(3, Node(9), n20)
        self.assertEqual(zig_zag_level_order(n3), [
            [3],
            [20, 9],
            [15, 7]
        ])
    
    def test_complete_tree(self):
        """
             1
           /   \
          3     2
         / \   /  
        4   5 6  
        """
        n3 = Node(3, Node(4), Node(5))
        n2 = Node(2, Node(6))
        n1 = Node(1, n3, n2)
        self.assertEqual(zig_zag_level_order(n1), [
            [1],
            [2, 3],
            [4, 5, 6]
        ])

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
        n3 = Node(3, right=Node(4, Node(7, right=Node(8))))
        n2 = Node(2, Node(5, right=Node(6, Node(9))))
        n1 = Node(1, n3, n2)
        self.assertEqual(zig_zag_level_order(n1), [
            [1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]
        ])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 27, 2019 \[Hard\] Minimum Appends to Craft a Palindrome
---
> **Question:** Given a string s we need to append (insertion at end) minimum characters to make a string palindrome.
>
> Follow-up: Don't use Manacher’s Algorithm, even though Longest Palindromic Substring can be efficiently solved with that algorithm.  

**Example 1:**

```
Input : s = "abede"
Output : "abedeba"
We can make string palindrome as "abedeba" by adding ba at the end of the string.
```

**Example 2:**
```
Input : s = "aabb"
Output : "aabbaa"
We can make string palindrome as"aabbaa" by adding aa at the end of the string.
```

**My thoughts:** An efficient way to solve this problem is to find the max len of suffix that is a palindrome. We can use a rolling hash function to quickly convert string into a number and by comparing the forward and backward hash value we can easily tell if a string is a palidrome or not. 
Example 1:
```py
Hash("123") = 123, Hash("321") = 321. Not Palindrome 
```
Example 2:
```py
Hash("101") = 101, Hash("101") = 101. Palindrome.
```
Rolling hashes are amazing, they provide you an ability to calculate the hash values without rehashing the whole string. eg. Hash("123") = Hash("12") ~ 3.  ~ is some function that can efficient using previous hashing value to build new caching value. 

However, we should not use Hash("123") = 123 as when the number become too big, the hash value be come arbitrarily big. Thus we use the following formula for rolling hash:

```py
hash("1234") = (1*p0^3 + 2*p0^2 + 3*p0^1 + 4*p0^0) % p1. where p0 is a much smaller prime and p1 is relatively large prime. 
```

There might be some hashing collision. However by choosing a much smaller p0 and relatively large p1, such collison is highly unlikely. Here I choose to remember a special large prime number `666667` and smaller number you can just use any smaller prime number, it shouldn't matter.


**Solution with Rolling Hash:** [https://repl.it/@trsong/Minimum-Appends-to-Craft-a-Palindrome](https://repl.it/@trsong/Minimum-Appends-to-Craft-a-Palindrome)
```py
import unittest

def craft_palindrome_with_min_appends(input_string):
    p0 = 17
    p1 = 666667  # large prime number worth to remember 
    reversed_string = input_string[::-1]
    forward_hash = 0  # right-most is most significant digit
    backward_hash = 0  # left-most is most significant digit
    max_len_palindrome_suffix = 0
    for i, char in enumerate(reversed_string):
        ord_char = ord(char)
        forward_hash = (forward_hash + ord_char * pow(p0, i)) % p1
        backward_hash = (p0 * backward_hash + ord_char) % p1
        if forward_hash == backward_hash:
            max_len_palindrome_suffix = i + 1

    return input_string + reversed_string[max_len_palindrome_suffix:]
  

class CraftPalindromeWithMinAppendSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(craft_palindrome_with_min_appends('abede'), 'abedeba')

    def test_example2(self):
        self.assertEqual(craft_palindrome_with_min_appends('aabb'), 'aabbaa')

    def test_empty_string(self):
        self.assertEqual(craft_palindrome_with_min_appends(''), '')
    
    def test_already_palindrome(self):
        self.assertEqual(craft_palindrome_with_min_appends('147313741'), '147313741')
        self.assertEqual(craft_palindrome_with_min_appends('328823'), '328823')

    def test_ascending_sequence(self):
        self.assertEqual(craft_palindrome_with_min_appends('12345'), '123454321')

    def test_binary_sequence(self):
        self.assertEqual(craft_palindrome_with_min_appends('10001001'), '100010010001')
        self.assertEqual(craft_palindrome_with_min_appends('100101'), '100101001')
        self.assertEqual(craft_palindrome_with_min_appends('010101'), '0101010')


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 26, 2019 \[Hard\] Find Next Greater Permutation
---
> **Question:** Given a number represented by a list of digits, find the next greater permutation of a number, in terms of lexicographic ordering. If there is not greater permutation possible, return the permutation with the lowest value/ordering.
>
> For example, the list `[1,2,3]` should return `[1,3,2]`. The list `[1,3,2]` should return `[2,1,3]`. The list `[3,2,1]` should return `[1,2,3]`.
>
> Can you perform the operation without allocating extra memory (disregarding the input memory)?

**My thoughts:** Imagine the list as a number, if it's in descending order, then there will be no number greater than that and we have to return the number in ascending order, that is, the smallest number. e.g. 321 will become 123. 

Leave first part untouched. If the later part of array are first increasing then decreasing, like 1321, then based on previous observation, we know the descending part will change from largest to smallest, we want the last increasing digit to increase as little as possible, i.e. slightly larger number on the right. e.g. 2113

Here are all the steps:
1. Find last increase number
2. Find the slightly larger number. i.e. the smallest one among all number greater than the last increase number on the right
3. Swap the slightly larger number with last increase number
4. Turn the descending array on right to be ascending array 

**Solution:** [https://repl.it/@trsong/Find-Next-Greater-Permutation](https://repl.it/@trsong/Find-Next-Greater-Permutation)
```py
import unittest

def next_greater_permutation(num_lst):
    n = len(num_lst)
    last_increase_index = n - 2

    # Step1: Find last increase number
    while last_increase_index >= 0:
        if num_lst[last_increase_index] >= num_lst[last_increase_index + 1]:
            last_increase_index -= 1
        else:
            break

    if last_increase_index >= 0:
        # Step2: Find the slightly larger number. i.e. the smallest one among all number greater than the last increase number on the right
        larger_num_index = n - 1
        while num_lst[larger_num_index] <= num_lst[last_increase_index]:
            larger_num_index -= 1

        # Step3: Swap the slightly larger number with last increase number
        num_lst[larger_num_index], num_lst[last_increase_index] = num_lst[last_increase_index], num_lst[larger_num_index]

    # Step4: Turn the descending array on right to be ascending array 
    i, j = last_increase_index + 1, n - 1
    while i < j:
        num_lst[i], num_lst[j] = num_lst[j], num_lst[i]
        i += 1
        j -= 1
    return num_lst


class NextGreaterPermutationSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(next_greater_permutation([1, 2, 3]), [1, 3, 2])
    
    def test_example2(self):
        self.assertEqual(next_greater_permutation([1, 3, 2]), [2, 1, 3])

    def test_example3(self):
        self.assertEqual(next_greater_permutation([3, 2, 1]), [1, 2, 3])

    def test_empty_array(self):
        self.assertEqual(next_greater_permutation([]), [])

    def test_one_elem_array(self):
        self.assertEqual(next_greater_permutation([1]), [1])

    def test_decrease_increase_decrease_array(self):
        self.assertEqual(next_greater_permutation([3, 2, 1, 6, 5, 4]), [3, 2, 4, 1, 5, 6])
        self.assertEqual(next_greater_permutation([3, 2, 4, 6, 5, 4]), [3, 2, 5, 4, 4, 6])

    def test_increasing_decreasing_increasing_array(self):
        self.assertEqual(next_greater_permutation([4, 5, 6, 1, 2, 3]), [4, 5, 6, 1, 3, 2])

    def test_multiple_decreasing_and_increasing_array(self):
        self.assertEqual(next_greater_permutation([5, 3, 4, 9, 7, 6]), [5, 3, 6, 4, 7, 9])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 25, 2019 \[Medium\] Longest Subarray with Sum Divisible by K
---
> **Question:** Given an arr[] containing n integers and a positive integer k. The problem is to find the length of the longest subarray with sum of the elements divisible by the given value k.

**Example:**
```py
Input : arr[] = {2, 7, 6, 1, 4, 5}, k = 3
Output : 4
The subarray is {7, 6, 1, 4} with sum 18, which is divisible by 3.
```

**My thoughts:** Recall the way to efficiently calculate subarray sum is to calculate prefix sum, `prefix_sum[i] = arr[0] + arr[1] + ... + arr[i] and prefix_sum[i] = prefix_sum[i-1] + arr[i]`. The subarray sum between index i and j, `arr[i] + arr[i+1] + ... + arr[j] = prefix_sum[j] - prefix_sum[i-1]`.

But this question is asking to find subarray whose sum is divisible by 3, that is,  `(prefix_sum[j] - prefix_sum[i-1]) mod k == 0 ` which implies `prefix_sum[j] % k == prefix_sum[j-1] % k`. So we just need to generate prefix_modulo array and find i,j such that `j - i reaches max` and `prefix_modulo[j] == prefix_modulo[i-1]`. As `j > i` and we must have value of `prefix_modulo[i-1]` already when we reach j. We can use a map to store the first occurance of certain prefix_modulo. This feels similar to Two-Sum question in a sense that we use map to store previous reached element and is able to quickly tell if current element satisfies or not.

**Solution:** [https://repl.it/@trsong/Longest-Subarray-with-Sum-Divisible-by-K](https://repl.it/@trsong/Longest-Subarray-with-Sum-Divisible-by-K)
```py
import unittest

def longest_subarray(nums, k):
    if not nums: return 0
    n = len(nums)
    prefix_modulo = [0] * n
    mod_so_far = 0
    for i in xrange(n):
        mod_so_far = (mod_so_far + nums[i] % k) % k
        prefix_modulo[i] = mod_so_far
    
    mod_first_occur_map = {0: -1}
    max_len = 0
    for i, prefix_mod in enumerate(prefix_modulo):
        if prefix_mod not in mod_first_occur_map:
            mod_first_occur_map[prefix_mod] = i
        else:
            max_len = max(max_len, i - mod_first_occur_map[prefix_mod])
    return max_len


class LongestSubarraySpec(unittest.TestCase):
    def test_example(self):
        # Modulo 3, prefix array = [2, 0, 0, 1, 2, 1]. max (end - start) = 4 such that prefix[end] - prefix[start-1] = 0
        self.assertEqual(longest_subarray([2, 7, 6, 1, 4, 5], 3), 4)  # sum([7, 6, 1, 4]) = 18 (18 % 3 = 0)

    def test_empty_array(self):
        self.assertEqual(longest_subarray([], 10), 0)
    
    def test_no_existance_of_such_subarray(self):
        self.assertEqual(longest_subarray([1, 2, 3, 4], 11), 0)
    
    def test_entire_array_qualify(self):
        # Modulo 4, prefix array: [0, 1, 2, 3, 3, 2, 1, 0]. max (end - start) = 8 such that prefix[end] - prefix[start-1] = 0
        self.assertEqual(longest_subarray([0, 1, 1, 1, 0, -1, -1, -1], 4), 8)  # entire array sum = 0
        self.assertEqual(longest_subarray([4, 5, 9, 17, 8, 3, 7, -1], 4), 8)  # entire array sum = 52 (52 % 4 = 0)

    def test_unique_subarray(self):
        # Modulo 6, prefix array: [0, 1, 1, 2, 3, 2, 4, 4, 5]. max (end - start) = 2 such that prefix[end] - prefix[start-1] = 0
        self.assertEqual(longest_subarray([0, 1, 0, 1, 1, -1, 2, 0, 1], 6), 2)  #  sum([1, -1]) = 0
        self.assertEqual(longest_subarray([6, 7, 12, 7, 13, 5, 8, 36, 19], 6), 2)  #  sum([13, 5]) = 18 (18 % 6 = 0)
    

if __name__ == '__main__':
    unittest.main(exit=False)

```

### Additional Question: LC 560 \[Medium\] Subarray Sum Equals K
---
> **Question:** Given an array of integers and an integer k, you need to find the total number of continuous subarrays whose sum equals to k.

**Example:**
```py
Input: nums = [1, 1, 1], k = 2
Output: 2
```

**My thoughts:** Just like how we efficiently calculate prefix_sum in previous question. We want to find how many index i exists such that `prefix[j] - prefix[i] = k`. As `j > i`, when we reach j, we pass i already, so we can store `prefix[i]` in a map and put value as occurance of `prefix[i]`, that is why this question feels similar to Two Sum question.

**Solution:** [https://repl.it/@trsong/Subarray-Sum-Equals-K](https://repl.it/@trsong/Subarray-Sum-Equals-K)
```py
import unittest

def subarray_sum(nums, k):
    sum_so_far = 0
    prefix_sum_occur_map = {0: 1}
    res = 0
    for num in nums:
        sum_so_far += num
        target_sum = sum_so_far - k
        res += prefix_sum_occur_map.get(target_sum, 0)
        prefix_sum_occur_map[sum_so_far] = prefix_sum_occur_map.get(sum_so_far, 0) + 1
    return res


class SubarraySumSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(subarray_sum([1, 1, 1], 2), 2)  # [1, 1] and [1, 1]
    
    def test_empty_array(self):
        self.assertEqual(subarray_sum([], 2), 0) 

    def test_target_is_zero(self):
        self.assertEqual(subarray_sum([0, 0], 0), 3) # [0], [0], [0, 0]

    def test_array_with_one_elem(self):
        self.assertEqual(subarray_sum([1], 0), 0)
        self.assertEqual(subarray_sum([1], 1), 1) # [1]

    def test_array_with_unique_target_prefix(self):
        # suppose the prefix_sum = [1, 2, 3, 3, 2, 1]
        self.assertEqual(subarray_sum([1, 1, 1, 0, -1, -1], 2), 4)  # [1, 1], [1, ,1], [1, 1, 0], [1, 1, 1, 0, -1]


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 24, 2019 LC 358 \[Hard\] Rearrange String K Distance Apart
---
> **Question:** Given a non-empty string str and an integer k, rearrange the string such that the same characters are at least distance k from each other.
>
> All input strings are given in lowercase letters. If it is not possible to rearrange the string, return an empty string "".

**Example 1:**
```
str = "aabbcc", k = 3
Result: "abcabc"
The same letters are at least distance 3 from each other.
```

**Example 2:**
```
str = "aaabc", k = 3 
Answer: ""
It is not possible to rearrange the string.
```

**Example 3:**
```
str = "aaadbbcc", k = 2
Answer: "abacabcd"
Another possible answer is: "abcabcda"
The same letters are at least distance 2 from each other.
```

**My thoughts:** The problem is just a variant of yesterday's Task Scheduler problem. The idea is to greedily choose the character with max remaining number for each window k. If no such character satisfy return empty string directly.

**Solution with Greedy Algorithm:** [https://repl.it/@trsong/Rearrange-String-K-Distance-Apart](https://repl.it/@trsong/Rearrange-String-K-Distance-Apart)
```py
import unittest
from Queue import PriorityQueue

def rearrange_string(input_string, k):
    if not input_string or k <= 0: return ""
    histogram = {}
    for c in input_string:
        # use negative key with min-heap to achieve max heap
        histogram[c] = histogram.get(c, 0) + 1
    
    max_heap = PriorityQueue()
    for c, count in histogram.items():
        max_heap.put((-count, c))

    res = []
    while not max_heap.empty():
        remaining_char = []
        for _ in xrange(k):
            # Greedily choose the char with max remaining count
            if max_heap.empty() and not remaining_char:
                break
            elif max_heap.empty():
                return ""
            neg_count, char = max_heap.get()
            count = -neg_count - 1
            res.append(char)
            if count > 0:
                remaining_char.append((-count, char))
        for count_char in remaining_char:
            max_heap.put(count_char)
    return ''.join(res)


class RearrangeStringSpec(unittest.TestCase):
    def assert_k_distance_apart(self, rearranged_string, original_string, k):
        # Test same length
        self.assertTrue(len(original_string) == len(rearranged_string))

        # Test containing all characters
        self.assertTrue(sorted(original_string) == sorted(rearranged_string))

        # Test K distance apart
        last_occur_map = {}
        for i, c in enumerate(rearranged_string):
            last_occur = last_occur_map.get(c, float('-inf'))
            self.assertTrue(i - last_occur >= k)
            last_occur_map[c] = i
    
    def test_utility_function_is_correct(self):
        original_string = "aaadbbcc"
        k = 2
        ans1 = "abacabcd"
        ans2 = "abcabcda"
        self.assert_k_distance_apart(ans1, original_string, k)
        self.assert_k_distance_apart(ans2, original_string, k)
        self.assertRaises(AssertionError, self.assert_k_distance_apart, original_string, original_string, k)

    def test_example1(self):
        original_string = "aabbcc"
        k = 3
        target_string = rearrange_string(original_string, k)
        self.assert_k_distance_apart(target_string, original_string, k)
    
    def test_example2(self):
        original_string = "aaabc"
        self.assertEqual(rearrange_string(original_string, 3),"")
    
    def test_example3(self):
        original_string = "aaadbbcc"
        k = 2
        target_string = rearrange_string(original_string, k)
        self.assert_k_distance_apart(target_string, original_string, k)
    
    def test_large_distance(self):
        original_string = "abcd"
        k = 10
        rearranged_string = rearrange_string(original_string, k)
        self.assert_k_distance_apart(rearranged_string, original_string, k)

    def test_empty_input_string(self):
        self.assertEqual(rearrange_string("", 1),"")
    
    def test_impossible_to_rearrange(self):
        self.assertEqual(rearrange_string("aaaabbbcc", 3), "")

    def test_k_too_small(self):
        self.assertEqual(rearrange_string("a", 0), "")
        self.assertEqual(rearrange_string("a", -1), "")
    

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 23, 2019 LC 621 \[Medium\] Task Scheduler
---
> **Question:** Given a char array representing tasks CPU need to do. It contains capital letters A to Z where different letters represent different tasks. Tasks could be done without original order. Each task could be done in one interval. For each interval, CPU could finish one task or just be idle.
>
> However, there is a non-negative cooling interval n that means between two same tasks, there must be at least n intervals that CPU are doing different tasks or just be idle.
>
> You need to return the least number of intervals the CPU will take to finish all the given tasks.

**Example:**
```py
Input: tasks = ["A", "A", "A", "B", "B", "B"], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B.
```

**My thoughts:** Treat n+1 as the size of each window. For each window, we try to fit as many tasks as possible following the max number of remaining tasks. If all tasks are chosen, we instead use idle. 

**Solution with Greedy Algorithm:** [https://repl.it/@trsong/Task-Scheduler](https://repl.it/@trsong/Task-Scheduler)
```py
import unittest
from Queue import PriorityQueue

def least_interval(tasks, n):
    if not tasks: return 0
    occurrence = {}
    for task in tasks:
        occurrence[task] = occurrence.get(task, 0) + 1
    
    max_heap = PriorityQueue()
    for task, occur in occurrence.items():
        # use negative key with min-heap to achieve max heap
        max_heap.put((-occur, task))
    
    res = 0
    while not max_heap.empty():
        remaining_tasks = []
        for _ in xrange(n+1):
            if max_heap.empty() and not remaining_tasks:
                break
            elif not max_heap.empty():
                # Greedily choose the task with max occurrence 
                negative_occur, task = max_heap.get()
                occur = -negative_occur - 1
                if occur > 0:
                    remaining_tasks.append((-occur, task))
            res += 1
        for task_occur in remaining_tasks:
            max_heap.put(task_occur)
    
    return res


class LeastIntervalSpec(unittest.TestCase):
    def test_example(self):
        tasks = ["A", "A", "A", "B", "B", "B"]
        n = 2
        self.assertEqual(least_interval(tasks, n), 8) # A -> B -> idle -> A -> B -> idle -> A -> B

    def test_no_tasks(self):
        self.assertEqual(least_interval([], 0), 0)
        self.assertEqual(least_interval([], 2), 0)
    
    def test_same_task_and_idle(self):
        tasks = ["A", "A", "A"]
        n = 1
        self.assertEqual(least_interval(tasks, n), 5)  # A -> idle -> A -> idle -> A

    def test_three_kind_tasks_no_idle(self):
        tasks = ["A", "B", "A", "C"]
        n = 1
        self.assertEqual(least_interval(tasks, n), 4)  # A -> B -> A -> C
    
    def test_three_kind_tasks_with_one_idle(self):
        tasks = ["A", "A", "A", "B", "C", "C"]
        n = 2
        self.assertEqual(least_interval(tasks, n), 7)  # A -> C -> B -> A -> C -> idle -> A

    def test_each_kind_one_task(self):
        tasks = ["A", "B", "C", "D"]
        n = 10
        self.assertEqual(least_interval(tasks, n), 4)  # A -> B -> C -> D


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 22, 2019 \[Medium\] Amazing Number
---
> **Question:** Define amazing number as: its value is less than or equal to its index. Given a circular array, find the starting position, such that the total number of amazing numbers in the array is maximized. 
> 
> Follow-up: Should get a solution with time complexity less than O(N^2)

**Example 1:**
```py
Input: [0, 1, 2, 3]
Ouptut: 0. When starting point at position 0, all the elements in the array are equal to its index. So all the numbers are amazing number.
```

**Example 2:** 
```py
Input: [1, 0, 0]
Output: 1. When starting point at position 1, the array becomes 0, 0, 1. All the elements are amazing number.
If there are multiple positions, return the smallest one.
```

**My thoughts:** We can use Brute-force Solution to get answer in O(n^2). But can we do better? Well, some smart observation is needed.

- First, we know that 0 and all negative numbers are amazing numbers. 
- Second, if a number is too big, i.e. greater than the length of array, then there is no way for that number to be an amazing number. 
- Finally, if a number is neither too small nor too big, i.e. between (0, n-1), then we can define "dangerous" range as [i - nums[i] + 1, i] and its complement: "safe" range [i + 1, i - nums[i]] should be safe. So we store all safe intervals to an array.

We accumlate those intervals by using interval counting technique: define interval_accu array, for each interval (start, end), interval_accu[start] += 1 and interval_accu[end+1] -= 1 so that when we can make interval accumulation by interval_accu[i] += interval_accu[i-1] for all i. 

Find max safe interval along the interval accumulation, i.e. the index that has maximum safe interval overlapping. 

**Brute-force Solution:** [https://repl.it/@trsong/Amazing-Number-Brute-force](https://repl.it/@trsong/Amazing-Number-Brute-force)
```py
def max_amazing_number_index(nums):
    n = len(nums)
    max_count = 0
    max_count_index = 0
    for i in xrange(n):
        count = 0
        for j in xrange(i, i + n):
            index = (j - i) % n
            if nums[j % n] <= index:
                count += 1
        
        if count > max_count:
            max_count = count
            max_count_index = i
    return max_count_index
```

**Efficient Solution with Interval Count:** [https://repl.it/@trsong/Amazing-Number](https://repl.it/@trsong/Amazing-Number)
```py
import unittest

def max_amazing_number_index(nums):
    n = len(nums)
    valid_intervals = []
    for i in xrange(n):
        # invalid zone starts from i - nums[i] + 1 and ends at i
        # 0 0 0 0 0 3 0 0 0 0 0 
        #       ^ ^ ^
        #       invalid
        # thus the valid zone is the complement [i + 1, i - nums[i]]
        if nums[i] > n:
            continue
        elif nums[i] < 0:
            valid_intervals.append([0, n-1])
        else:
            valid_intervals.append([(i + 1) % n, (i - nums[i]) % n])

    interval_accumulation = [0] * n
    for start, end in valid_intervals:
        # valid interval [start, end] is circular, i.e. end < start
        # thus can be broken into [0, end] and [start, n-1]
        interval_accumulation[start] += 1

        # with one exception: end > start, when the number is too small, like smaller than 0,
        # if that's the case, we don't count 
        if start > end:
            interval_accumulation[0] += 1

        if end + 1 < n:
            interval_accumulation[end + 1] -= 1

    max_count = interval_accumulation[0]
    max_count_index = 0
    for i in xrange(1, n):
        interval_accumulation[i] += interval_accumulation[i-1]
        if interval_accumulation[i] > max_count:
            max_count = interval_accumulation[i]
            max_count_index = i
    return max_count_index 


class MaxAmazingNumberIndexSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(max_amazing_number_index([0, 1, 2, 3]), 0)  # max # amazing number = 4 at [0, 1, 2, 3]

    def test_example2(self):
        self.assertEqual(max_amazing_number_index([1, 0, 0]), 1)  # max # amazing number = 3 at [0, 0, 1]

    def test_non_descending_array(self):
        self.assertEqual(max_amazing_number_index([0, 0, 0, 1, 2, 3]), 0)  # max # amazing number = 0 at [0, 0, 0, 1, 2, 3]

    def test_random_array(self):
        self.assertEqual(max_amazing_number_index([1, 4, 3, 2]), 1)  # max # amazing number = 2 at [4, 3, 2, 1]

    def test_non_ascending_array(self):
        self.assertEqual(max_amazing_number_index([3, 3, 2, 1, 0]), 2)  # max # amazing number = 4 at [2, 1, 0, 3, 3]

    def test_return_smallest_index_when_no_amazing_number(self):
        self.assertEqual(max_amazing_number_index([99, 99, 99, 99]), 0)  # max # amazing number = 0 thus return smallest possible index

    def test_negative_number(self):
        self.assertEqual(max_amazing_number_index([3, -99, -99, -99]), 1)  # max # amazing number = 4 at [-1, -1, -1, 3])

if __name__ == '__main__':
    unittest.main(exit=False)
```




### Aug 21, 2019 LC 273 \[Hard\] Integer to English Words
---
> **Question:** Convert a non-negative integer to its English word representation. 

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

**My thoughts:** The main difficulty of this problem comes from edge case scenarios from breaking large number into smaller ones and conquer them separately. Includes but not limit to "Zero", "Ten", "Twenty One" and other edge cases from missing millions, thousands and hundreds, like "Thirty Billion Two Million" and "Fifty Billion Two Hundred".


**Solution:** [https://repl.it/@trsong/Integer-to-English-Words](https://repl.it/@trsong/Integer-to-English-Words)
```py
import unittest

word_lookup = {
    0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five',
    6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten',
    11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', 15: 'Fifteen',
    16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', 19: 'Nineteen', 20: 'Twenty',
    30: 'Thirty', 40: 'Forty', 50: 'Fifty', 60: 'Sixty', 
    70: 'Seventy', 80: 'Eighty', 90: 'Ninety',
    100: 'Hundred', 1000: 'Thousand', 1000000: 'Million', 1000000000: 'Billion'
}

def read_hundreds(num):
    # Helper function to read num between 1 and 999
    global word_lookup
    if num == 0: return []
    res = []
    if num >= 100:
        res.append(word_lookup[num / 100])
        res.append(word_lookup[100])
    
    num %= 100
    if 21 <= num <= 99:
        res.append(word_lookup[num - num % 10])
        if num % 10 > 0:
            res.append(word_lookup[num % 10])
    elif num > 0:
        res.append(word_lookup[num])
        
    return res


def number_to_words(num):
    global word_lookup
    if num == 0: return word_lookup[0]
    res = []
    separators = [1000000000, 1000000, 1000] # 'Billion', 'Million', 'Thousand'
    for sep in separators:
        if num >= sep:
            res += read_hundreds(num/sep)
            res.append(word_lookup[sep])
            num %= sep
    if num > 0:
        res += read_hundreds(num)
    return ' '.join(res)


class NumberToWordSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(number_to_words(123), "One Hundred Twenty Three")

    def test_example2(self):
        self.assertEqual(number_to_words(12345), "Twelve Thousand Three Hundred Forty Five")

    def test_example3(self):
        self.assertEqual(number_to_words(1234567), "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven")

    def test_example4(self):
        self.assertEqual(number_to_words(1234567891), "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One")

    def test_zero(self):
        self.assertEqual(number_to_words(0), "Zero")

    def test_one_digit(self):
        self.assertEqual(number_to_words(8), "Eight")

    def test_two_digits(self):
        self.assertEqual(number_to_words(21), "Twenty One")
        self.assertEqual(number_to_words(10), "Ten")
        self.assertEqual(number_to_words(20), "Twenty")
        self.assertEqual(number_to_words(16), "Sixteen")
        self.assertEqual(number_to_words(32), "Thirty Two")
        self.assertEqual(number_to_words(30), "Thirty")

    def test_ignore_thousand_part(self):
        self.assertEqual(number_to_words(30002000000), "Thirty Billion Two Million")

    def test_ignore_million_part(self):
        self.assertEqual(number_to_words(50000000200), "Fifty Billion Two Hundred")


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 20, 2019 LC 297 \[Hard\] Serialize and Deserialize Binary Tree
---
> **Question:** Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.
>
> Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

**Example 1:**

```py
You may serialize the following tree:

    1
   / \
  2   3
     / \
    4   5

as "[1,2,3,null,null,4,5]"
```

**Example 2:**

```py
You may serialize the following tree:

       5
      / \ 
     4   7
    /   /
   3   2
  /   /
-1   9

as "[5,4,7,3,null,2,null,-1,null,9]"
```

**Solution with BFS:** [https://repl.it/@trsong/Serialize-and-Deserialize-Binary-Tree](https://repl.it/@trsong/Serialize-and-Deserialize-Binary-Tree)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __eq__(self, other):
        return other and other.val == self.val and other.left == self.left and other.right == self.right

class BinaryTreeSerializer(object):
    @staticmethod
    def serialize(tree):
        if not tree: return "[]"
        res = []
        queue = [tree]
        is_done = False
        while queue and not is_done:
            level_size = len(queue)
            is_done = True
            for _ in xrange(level_size):
                cur = queue.pop(0)
                if not cur:
                    res.append("null")
                else:
                    res.append(str(cur.val))
                    if cur.left or cur.right:
                        is_done = False
                    queue.append(cur.left)
                    queue.append(cur.right)
        return "[" + ",".join(res) + "]"

    @staticmethod
    def deserialize(encoded_string):
        if encoded_string == "[]": return None
        nums = encoded_string[1:-1].split(',')
        tree = TreeNode(int(nums[0]))
        i = 1
        queue = [tree]
        while queue:
            level_size = len(queue)
            for _ in xrange(level_size):
                cur = queue.pop(0)
                if i < len(nums) and nums[i] != "null":
                    cur.left = TreeNode(int(nums[i]))
                    queue.append(cur.left)
                i += 1

                if i < len(nums) and nums[i] != "null":
                    cur.right = TreeNode(int(nums[i]))
                    queue.append(cur.right)
                i += 1

                if i > len(nums):
                    break
        return tree
   

class BinaryTreeSerializerSpec(unittest.TestCase):
    def test_example1(self):
        """
            1
           / \
          2   3
             / \
            4   5
        """
        n3 = TreeNode(3, TreeNode(4), TreeNode(5))
        n1 = TreeNode(1, TreeNode(2), n3)
        encoded = BinaryTreeSerializer.serialize(n1)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(decoded, n1)

    def test_example2(self):
        """
               5
              / \ 
             4   7
            /   /
           3   2
          /   /
        -1   9
        """
        n4 = TreeNode(4, TreeNode(3, TreeNode(-1)))
        n7 = TreeNode(7, TreeNode(2, TreeNode(9)))
        n5 = TreeNode(5, n4, n7)
        encoded = BinaryTreeSerializer.serialize(n5)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(decoded, n5)

    def test_serialize_empty_tree(self):
        encoded = BinaryTreeSerializer.serialize(None)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertIsNone(decoded)

    def test_serialize_left_heavy_tree(self):
        """
            1
           /
          2
         /
        3
        """
        tree = TreeNode(1, TreeNode(2, TreeNode(3)))
        encoded = BinaryTreeSerializer.serialize(tree)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(decoded, tree)

    def test_serialize_right_heavy_tree(self):
        """
        1
         \
          2
         /
        3
        """
        tree = TreeNode(1, right=TreeNode(2, TreeNode(3)))
        encoded = BinaryTreeSerializer.serialize(tree)
        decoded = BinaryTreeSerializer.deserialize(encoded)
        self.assertEqual(decoded, tree) 
        

        
if __name__ == '__main__':
    unittest.main(exit=False)
```

### Additional Question: \[Medium\] M Smallest in K Sorted Lists
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

**Solution:** [https://repl.it/@trsong/M-Smallest-in-K-Sorted-Lists](https://repl.it/@trsong/M-Smallest-in-K-Sorted-Lists)
```py
import unittest
from Queue import PriorityQueue

def find_m_smallest(ksorted_list, m):
    pq = PriorityQueue()
    for lst in ksorted_list:
        if lst:
            pq.put((lst[0], [0, lst]))
    
    res = None
    while not pq.empty() and m >= 1:
        index, lst = pq.get()[1]
        if m == 1:
            res = lst[index]
            break
        m -= 1
        if index + 1 < len(lst):
            pq.put((lst[index + 1], [index + 1, lst]))
    return res


class FindMSmallestSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(find_m_smallest([[1, 3], [2, 4, 6], [0, 9, 10, 11]], m=5), 4) 

    def test_example2(self):
        self.assertEqual(find_m_smallest([[1, 3, 20], [2, 4, 6]], m=2), 2) 

    def test_example3(self):
        self.assertEqual(find_m_smallest([[1, 3, 20], [2, 4, 6]], m=6), 20)

    def test_empty_sublist(self):
        self.assertEqual(find_m_smallest([[1], [], [0, 2]], m=2), 1)

    def test_one_sublist(self):
        self.assertEqual(find_m_smallest([[1, 2, 3, 4, 5]], m=5), 5)

    def test_target_out_of_boundary(self):
        self.assertIsNone(find_m_smallest([[1, 2, 3], [4, 5, 6]], 7))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 19, 2019 \[Medium\] Jumping Numbers
---
> **Question:** Given a positive int n, print all jumping numbers smaller than or equal to n. A number is called a jumping number if all adjacent digits in it differ by 1. For example, 8987 and 4343456 are jumping numbers, but 796 and 89098 are not. All single digit numbers are considered as jumping numbers.

**Example:**

```py
Input: 105
Output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 21, 23, 32, 34, 43, 45, 54, 56, 65, 67, 76, 78, 87, 89, 98, 101]
```

**My thoughts:** We can use Brutal Force to search from 0 to given upperbound to find jumping numbers which might not be so efficient. Or we can take advantage of the property of jummping number: a jumping number is:
- either one of all single digit numbers 
- or 10 * some jumping number + last digit of that jumping number +/- by 1

For example, 

```
1 -> 10, 12.
2 -> 21, 23.
3 -> 32, 34.
...
10 -> 101
12 -> 121, 123
```

We can get all qualified jumping number by BFS searching for all candidates.

**Solution with BFS:** [https://repl.it/@trsong/Jumping-Numbers](https://repl.it/@trsong/Jumping-Numbers)
```py
import unittest

def generate_jumping_numbers(upper_bound):
    if upper_bound < 0: return []
    queue = [x for x in xrange(1, 10)]
    res = [0]
    while queue:
        # Apply BFS to search for jumping numbers
        cur = queue.pop(0)
        if cur > upper_bound:
            break
        res.append(cur)
        last_digit = cur % 10
        if last_digit > 0:
            queue.append(10 * cur + last_digit - 1)
        
        if last_digit < 9:
            queue.append(10 * cur + last_digit + 1)
    return res


class GenerateJumpingNumberSpec(unittest.TestCase):
    def test_zero_as_upperbound(self):
        self.assertEqual(generate_jumping_numbers(0), [0])

    def test_single_digits_are_all_jummping_numbers(self):
        self.assertEqual(generate_jumping_numbers(9), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_five_as_upperbound(self):
        self.assertEqual(generate_jumping_numbers(5), [0, 1, 2, 3, 4, 5])

    def test_not_always_contains_upperbound(self):
        self.assertEqual(generate_jumping_numbers(13), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12])

    def test_example(self):
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 21, 23, 32, 34, 43, 45, 54, 56, 65, 67, 76, 78, 87, 89, 98, 101]
        self.assertEqual(generate_jumping_numbers(105), expected)
    
    def test_negative_upperbound(self):
        self.assertEqual(generate_jumping_numbers(-1), [])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Additional Question: \[Easy\] Swap Even and Odd Nodes
---
> **Question:** Given the head of a singly linked list, swap every two nodes and return its head.
> 
> Note: Make sure it's acutally nodes that get swapped not value. 


**Example:**
```py
given 1 -> 2 -> 3 -> 4, return 2 -> 1 -> 4 -> 3.
```

**Solution with Recursion:** [https://repl.it/@trsong/Swap-Even-and-Odd-Nodes](https://repl.it/@trsong/Swap-Even-and-Odd-Nodes)
```py
import unittest

class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


def swap_list(lst):
    if not lst or not lst.next:
        return lst
    first = lst
    second = first.next
    third = second.next

    second.next = first
    first.next = swap_list(third)
    return second
  

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


if __name__ == '__main__':
    unittest.main(exit=False)
```

**Solution2 with Iteration:** [https://repl.it/@trsong/Swap-Even-and-Odd-Nodes-Iterative](https://repl.it/@trsong/Swap-Even-and-Odd-Nodes-Iterative)

```py
def swap_list(lst):
    dummy = ListNode(-1, lst)
    prev = dummy
    p = lst
    while p and p.next:
        first = p
        second = first.next

        first.next = second.next
        second.next = first
        prev.next = second
        prev = first
        p = prev.next
    return dummy.next
```


### Aug 18, 2019 LT 612 \[Medium\] K Closest Points
--- 
> **Question:** Given some points and a point origin in two dimensional space, find k points out of the some points which are nearest to origin.
> 
> Return these points sorted by distance, if they are same with distance, sorted by x-axis, otherwise sorted by y-axis.


**Example:**

```py
Given points = [[4, 6], [4, 7], [4, 4], [2, 5], [1, 1]], origin = [0, 0], k = 3
return [[1, 1], [2, 5], [4, 4]]
```

**My thoguhts:** This problem can be easily solved with k Max-heap with key being the distance and value being the point. First heapify first k elements to form a k max-heap. Then for the remaining n - k element, replace top of heap with smaller-distance point.

**Solution with k Max-Heap:** [https://repl.it/@trsong/K-Closest-Points](https://repl.it/@trsong/K-Closest-Points)
```py
import unittest
from Queue import PriorityQueue

def distance2(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy 

def k_closest_points(points, origin, k):
    if not points and k == 0: return []
    elif len(points) < k: return None

    max_heap = PriorityQueue()
    for i in xrange(k):
        max_heap.put((-distance2(points[i], origin), points[i]))

    for i in xrange(k, len(points)):
        dist = distance2(points[i], origin)
        top = max_heap.queue[0]
        if -top[0] > dist:   
            max_heap.get()
            max_heap.put((-dist, points[i]))
            
    res = [None] * k
    for i in xrange(k-1, -1, -1):
        res[i] = max_heap.get()[1]
    return res


class KClosestPointSpec(unittest.TestCase):
    def assert_points(self, result, expected):
        self.assertEqual(sorted(result), sorted(expected))

    def test_example(self):
        points = [[4, 6], [4, 7], [4, 4], [2, 5], [1, 1]]
        origin = [0, 0]
        k = 3
        expected = [[1, 1], [2, 5], [4, 4]]
        self.assert_points(k_closest_points(points, origin, k), expected)

    def test_empty_points(self):
        self.assert_points(k_closest_points([], [0, 0], 0), [])
        self.assertIsNone(k_closest_points([], [0, 0], 1))

    def test_descending_distance(self):
        points = [[1, 6], [1, 5], [1, 4], [1, 3], [1, 2], [1, 1]]
        origin = [1, 1]
        k = 2
        expected = [[1, 2], [1, 1]]
        self.assert_points(k_closest_points(points, origin, k), expected)

    def test_ascending_distance(self):
        points = [[-1, -1], [-2, -1], [-3, -1], [-4, -1], [-5, -1], [-6, -1]]
        origin = [-1, -1]
        k = 1
        expected = [[-1, -1]]
        self.assert_points(k_closest_points(points, origin, k), expected)

    def test_duplicated_distance(self):
        points = [[1, 0], [0, 1], [-1, -1], [1, 1], [2, 1], [-2, 0]]
        origin = [0, 0]
        k = 5
        expected = [[1, 0], [0, 1], [-1, -1], [1, 1], [-2, 0]]
        self.assert_points(k_closest_points(points, origin, k), expected)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Additional Question: \[Medium\] Swap Even and Odd Bits
---
> **Question:** Given an unsigned 8-bit integer, swap its even and odd bits. The 1st and 2nd bit should be swapped, the 3rd and 4th bit should be swapped, and so on.

**Example:**

```py
10101010 should be 01010101. 11100010 should be 11010001.
```
> Bonus: Can you do this in one line?

**Solution:** [https://repl.it/@trsong/Swap-Even-and-Odd-Bits](https://repl.it/@trsong/Swap-Even-and-Odd-Bits)
```py
import unittest

def swap_bits(num):
    # 1010 is 0xa and  0101 is 0x5
    # 32 bit has 8 bits (4 * 8 = 32)
    return (num & 0xaaaaaaaa) >> 1 | (num & 0x55555555) << 1

class SwapBitSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(swap_bits(0b10101010), 0b01010101)

    def test_example2(self):
        self.assertEqual(swap_bits(0b11100010), 0b11010001)

    def test_zero(self):
        self.assertEqual(swap_bits(0), 0)
    
    def test_one(self):
        self.assertEqual(swap_bits(0b1), 0b10)

    def test_odd_digits(self):
        self.assertEqual(swap_bits(0b111), 0b1011)

    def test_large_number(self):
        self.assertEqual(swap_bits(0xffffffff), 0xffffffff)
        self.assertEqual(swap_bits(0x55555555), 0xaaaaaaaa)
        self.assertEqual(swap_bits(0xaaaaaaaa), 0x55555555)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 17, 2019 \[Medium\] Deep Copy Linked List with Pointer to Random Node
---
> **Question:** Make a deep copy of a linked list that has a random link pointer and a next pointer.

**My thoughts:** The way we solve this problem is to mingle old nodes and cloned nodes so that every odd node is original node and every even node is clone node which will allow us to access both nodes through `node.next` and `node.next.next`. And we then build random pointer and finally connect every other node to build cloned node's next as well as restore original node's next. 

**Solution:** [https://repl.it/@trsong/Deep-Copy-Linked-List-with-Pointer-to-Random-Node](https://repl.it/@trsong/Deep-Copy-Linked-List-with-Pointer-to-Random-Node)
```py
import unittest

class ListNode(object):
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random
    
    def __eq__(self, other):
        if other is not None:
            is_random_valid = self.random is None and other.random is None or self.random is not None and other.random is not None and self.random.val == other.random.val
            return is_random_valid and self.val == other.val and self.next == other.next
        else:
            return False

def deep_copy(lst):
    if not lst: return None
    
    # Insert cloned node into original list
    # Now every odd pointer is old node and every even pointer is cloned node
    node = lst
    while node:
        node.next = ListNode(node.val, node.next)
        node = node.next.next

    # Build cloned node's random pointer
    node = lst
    while node:
        node.next.random = node.random.next if node.random else None
        node = node.next.next

    # Build cloned node's next and restore old node's next
    node = lst
    res = lst.next
    while node:
        cloned_node = node.next
        old_next = cloned_node.next
        if old_next:
            cloned_node.next = old_next.next
        node.next = old_next
        node = old_next
    
    return res


class DeepCopySpec(unittest.TestCase):
    def test_empty_list(self):
        self.assertIsNone(deep_copy(None))
    
    def test_list_with_random_point_to_itself(self):
        n = ListNode(1)
        n.random = n
        self.assertEqual(deep_copy(n), n)

    def test_list_with_forward_random_pointers(self):
        # 1 -> 2 -> 3 -> 4
        n4 = ListNode(4)
        n3 = ListNode(3, n4)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)

        # random pointer:
        # 1 -> 3
        # 2 -> 3
        # 3 -> 4
        n1.random = n3
        n2.random = n3
        n3.random = n4
        self.assertEqual(deep_copy(n1), n1)

    def test_list_with_backward_random_pointers(self):
        # 1 -> 2 -> 3 -> 4
        n4 = ListNode(4)
        n3 = ListNode(3, n4)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)

        # random pointer:
        # 1 -> 1
        # 2 -> 1
        # 3 -> 2
        # 4 -> 1
        n1.random = n1
        n2.random = n1
        n3.random = n2
        n4.random = n1
        self.assertEqual(deep_copy(n1), n1)

    def test_list_with_both_forward_and_backward_pointers(self):
        # 1 -> 2 -> 3 -> 4
        n4 = ListNode(4)
        n3 = ListNode(3, n4)
        n2 = ListNode(2, n3)
        n1 = ListNode(1, n2)

        # random pointer:
        # 1 -> 3
        # 2 -> 1
        # 3 -> 4
        # 4 -> 3
        n1.random = n3
        n2.random = n2
        n3.random = n4
        n4.random = n3
        self.assertEqual(deep_copy(n1), n1)
        

if __name__ == '__main__':
    unittest.main(exit=False)
``` 

### Aug 16, 2019 \[Medium\] Longest Substring without Repeating Characters
---
> **Question:** Given a string, find the length of the longest substring without repeating characters.
>
> **Note:** Can you find a solution in linear time?
 
**Example:**
```py
lengthOfLongestSubstring("abrkaabcdefghijjxxx") # => 10 as len("abcdefghij") == 10
```

**My thoughts:** This is a typical sliding window problem. The idea is to mantain a last occurance map while proceeding the sliding window. Such window is bounded by indices `(i, j)`, whenever we process next character j, we check the last occurance map to see if the current character `a[j]` is duplicated within the window `(i, j)`, ie. `i <= k < j`, if that's the case, we move `i` to `k + 1` so that `a[j]` no longer exists in window. And we mantain the largest window size `j - i + 1` as the longest substring without repeating characters.

**Solution with Sliding Window:** [https://repl.it/@trsong/Longest-Substring-without-Repeating-Characters](https://repl.it/@trsong/Longest-Substring-without-Repeating-Characters)
```py
import unittest

def longest_nonrepeated_substring(input_string):
    if not input_string: return 0
    last_occur = {}
    max_len = 0
    i = 0
    for j in xrange(len(input_string)):
        cur = input_string[j]
        last_index = last_occur.get(cur, -1)
        if last_index >= i:
            i = last_index + 1
        last_occur[cur] = j
        max_len = max(max_len, j - i + 1)
    return max_len
    

class LongestNonrepeatedSubstringSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(longest_nonrepeated_substring("abrkaabcdefghijjxxx"), 10) # "abcdefghij"

    def test_empty_string(self):
        self.assertEqual(longest_nonrepeated_substring(""), 0)

    def test_string_with_repeated_characters(self):
        self.assertEqual(longest_nonrepeated_substring("aabbafacbbcacbfa"), 4) # "facb"

    def test_some_random_string(self):
        self.assertEqual(longest_nonrepeated_substring("ABDEFGABEF"), 6) # "ABDEFG"

    def test_all_repated_characters(self):
        self.assertEqual(longest_nonrepeated_substring("aaa"), 1) # "a"
    
    def test_non_repated_characters(self):
        self.assertEqual(longest_nonrepeated_substring("abcde"), 5) # "abcde"

if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 15, 2019 \[Hard\] Largest Rectangle
---
> **Question:** Given an N by M matrix consisting only of 1's and 0's, find the largest rectangle containing only 1's and return its area.

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

**My thoughts:** This problem is an application of finding largest rectangle in histogram. That question gives you an array of height of bar in histogram and find the largest area of rectangle bounded by the bar. (consider bar width as 1)

Example:
```py
largest_rectangle_in_histogram([3, 4, 5, 4, 3]) # return 15 as max at height 3 * width 5
```

Now the way we take advantage of largest_rectangle_in_histogram is that, we can calculate the histogram of each row with each cell value being the accumulated value since last saw 1. 

Example:

```py
Suppose the table looks like the following:

[
    [0, 1, 0, 1],
    [1, 1, 1, 0],
    [0, 1, 1, 0]
]

The histogram of first row is just itself:
    [0, 1, 0, 1]  # largest_rectangle_in_histogram => 1 as max at height 1 * width 1
The histogram of second row is:
    [0, 1, 0, 1]
    +
    [1, 1, 1, 0]
    =
    [1, 2, 1, 0]  # largest_rectangle_in_histogram => 3 as max at height 1 * width 3
              ^ will not accumulate 0
The histogram of third row is:
    [1, 2, 1, 0]       
    +
    [0, 1, 1, 0]
    =
    [0, 3, 2, 0]  # largest_rectangle_in_histogram => 4 as max at height 2 * width 2
              ^ will not accumulate 0
     ^ will not accumulate 0

Therefore, the largest rectangle has 4 1s in it.
```

**Solution with DP:** [https://repl.it/@trsong/Largest-Rectangle](https://repl.it/@trsong/Largest-Rectangle)
```py
import unittest

def largest_rectangle_in_histogram(histogram):
    stack = []
    i = 0
    max_area = 0
    while i < len(histogram) or stack:
        if not stack or i < len(histogram) and histogram[stack[-1]] <= histogram[i]:
            # maintain an ascending stack
            stack.append(i)
            i += 1
        else:
            # if stack starts decreasing,
            # then left boundary must be stack[-2] and right boundary must be i. Note both boundaries are exclusive
            # and height is stack[-1]
            height = histogram[stack.pop()]
            left_boundary = stack[-1] if stack else -1
            right_boundary = i
            current_area = height * (right_boundary - left_boundary - 1)
            max_area = max(max_area, current_area)
    return max_area


class LargestRectangleInHistogramSpec(unittest.TestCase):
    def test_ascending_sequence(self):
        self.assertEqual(largest_rectangle_in_histogram([0, 1, 2, 3, 4]), 6) # max at height 2 * width 3

    def test_descending_sequence(self):
        self.assertEqual(largest_rectangle_in_histogram([4, 3, 2, 1, 0]), 6) # max at height 3 * width 2

    def test_sequence3(self):
        self.assertEqual(largest_rectangle_in_histogram([3, 4, 5, 4, 3]), 15) # max at height 3 * width 5

    def test_sequence4(self):
        self.assertEqual(largest_rectangle_in_histogram([3, 10, 4, 10, 5, 10, 4, 10, 3]), 28)  # max at height 4 * width 7

    def test_sequence5(self):
        self.assertEqual(largest_rectangle_in_histogram([6, 2, 5, 4, 5, 1, 6]), 12)  # max at height 4 * width 3


def largest_rectangle(table):
    if not table or not table[0]: return 0
    n, m = len(table), len(table[0])
    max_area = largest_rectangle_in_histogram(table[0])
    for r in xrange(1, n):
        for c in xrange(m):
            # calculate the histogram of each row since last saw 1 at same column
            if table[r][c] == 1:
                table[r][c] = table[r-1][c] + 1
        max_area = max(max_area, largest_rectangle_in_histogram(table[r]))
    return max_area
    

class LargestRectangleSpec(unittest.TestCase):
    def test_empty_table(self):
        self.assertEqual(largest_rectangle([]), 0)
        self.assertEqual(largest_rectangle([[]]), 0)

    def test_example(self):
        self.assertEqual(largest_rectangle([
            [1, 0, 0, 0],
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [0, 1, 0, 0]
        ]), 4)

    def test_table2(self):
        self.assertEqual(largest_rectangle([
            [0, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 1, 1, 0]
        ]), 4)

    def test_table3(self):
        self.assertEqual(largest_rectangle([
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0],
        ]), 3)

    def test_table4(self):
        self.assertEqual(largest_rectangle([
            [0, 0, 1, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 0, 1],
        ]), 4)

    def test_table5(self):
        self.assertEqual(largest_rectangle([
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 0]
        ]), 8)


if __name__ == '__main__':
    unittest.main(exit=False)
```


### Aug 14, 2019 LC 375 \[Medium\] Guess Number Higher or Lower II
---
> **Question:** We are playing the Guess Game. The game is as follows:
>
> I pick a number from 1 to n. You have to guess which number I picked.
> 
> Every time you guess wrong, I'll tell you whether the number I picked is higher or lower.
> 
> However, when you guess a particular number x, and you guess wrong, you pay $x. You win the game when you guess the number I picked.

**Example:**

```
n = 10, I pick 8.

First round:  You guess 5, I tell you that it's higher. You pay $5.
Second round: You guess 7, I tell you that it's higher. You pay $7.
Third round:  You guess 9, I tell you that it's lower. You pay $9.

Game over. 8 is the number I picked.

You end up paying $5 + $7 + $9 = $21.
```

Given a particular n ≥ 1, find out how much money you need to have to guarantee a win.

**My thoughts:** This question looks really similar to *LC 312 [Hard] Burst Balloons* in a sense that it choose the best candidate at each step and deligate the subproblem to recursive calls. eg. `guess_number_cost_between(i, j) = max(k + guess_number_cost_between(i, k-1), guess_number_cost_between(k+1, j)) for all k between i+1 and j-1 inclusive`.

Take a look at base case:
- when 1 number to pick, the min $ to secure a win is $0
- when 2 numbers to pick, the min $ to secure a win is to choose the smaller one, 
- when 3 numbers to pick, the min $ to secure a win is to choose the middle one

**Solution with Top-down DP(Recursion with Cache):** [https://repl.it/@trsong/Guess-Number-Higher-or-Lower-II](https://repl.it/@trsong/Guess-Number-Higher-or-Lower-II)
```py
import unittest
import sys

def guess_number_cost(n):
    cache = [[None for _ in xrange(n+1)] for _ in xrange(n+1)]
    return guess_number_cost_with_cache(1, n, cache)


def guess_number_cost_between(i, j, cache):
    if i == j: return 0
    elif i + 1 == j: return i
    elif i + 2 == j: return i + 1
    cost = sys.maxint
    for k in xrange(i+1, j):
        pick_k_cost = k + guess_number_cost_with_cache(i, k-1, cache) + guess_number_cost_with_cache(k+1, j, cache)
        cost = min(cost, pick_k_cost)
    return cost


def guess_number_cost_with_cache(i, j, cache):
    if cache[i][j] is None:
        cache[i][j] = guess_number_cost_between(i, j, cache)
    return cache[i][j]


class GuessNumberCostSpec(unittest.TestCase):
    def test_n_equals_one(self):
        self.assertEqual(guess_number_cost(1), 0)

    def test_n_equals_two(self):
        self.assertEqual(guess_number_cost(2), 1) # pick: 1

    def test_n_equals_three(self):
        self.assertEqual(guess_number_cost(3), 2) # pick: 2

    def test_n_equals_four(self):
        self.assertEqual(guess_number_cost(4), 4) # pick: 1, 3

    def test_n_equals_five(self):
        self.assertEqual(guess_number_cost(5), 6) # pick: 2, 4

    def test_n_equals_six(self):
        self.assertEqual(guess_number_cost(6), 9) # pick 1, 3, 5


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 13, 2019 LC 727 \[Hard\] Minimum Window Subsequence
---
> **Question:** Given strings S and T, find the minimum (contiguous) substring W of S, so that T is a subsequence of W.
>
> If there is no such window in S that covers all characters in T, return the empty string "". If there are multiple such minimum-length windows, return the one with the left-most starting index.

**Example:**

```
Input: 
S = "abcdebdde", T = "bde"
Output: "bcde"

Explanation: 
"bcde" is the answer because it occurs before "bdde" which has the same length.
"deb" is not a smaller window because the elements of T in the window must occur in order.
```

**My thoughts:** Have you noticed the pattern that substring of s always has the same first char as t. i.e. `s = "abcdebdde", t = "bde", substring = "bcde", substring[0] == t[0]`,  we can take advantage of that to keep track of previous index such that t[0] == s[index] and we can do that recursively for the rest of t and s. We get the following recursive definition

```py
Let dp[i][j] = index where index represents index such that s[index:i] has subsequence t[0:j].

dp[i][j] = dp[i-1][j-1] if there s[i-1] matches t[j-1] 
         = dp[i-1][j]   otherwise
```

And the final solution is to find index where `len of t <= index <= len of s` such that `index - dp[index][len of t]` i.e. the length of substring, reaches minimum. 

**Solution with DP:** [https://repl.it/@trsong/Minimum-Window-Subsequence](https://repl.it/@trsong/Minimum-Window-Subsequence)
```py
import unittest
import sys

def min_window_subsequence(s, t):
    if len(s) < len(t): return ""
    n, m = len(s), len(t)

    # let dp[i][j] = index where index represents index such that s[index:i] has subsequence t[0:j]
    # Then s[start: end] where start = dp[i][m], end = i such that len = i - dp[i][m] reaches minimum
    dp = [[-1 for _ in xrange(m+1)] for _ in xrange(n+1)]
    for i in xrange(n):
        dp[i][0] = i

    for i in xrange(1, n+1):
        for j in xrange(1, m+1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = dp[i-1][j]

    start = -1
    min_len = sys.maxint
    for i in xrange(m, n+1):
        if dp[i][m] != -1:
            cur_len = i - dp[i][m]
            if cur_len < min_len:
                start = dp[i][m]
                min_len = cur_len
    return s[start: start + min_len] if start != -1 else ""


class MinWindowSubsequenceSpec(unittest.TestCase):
    def test_example(self):
        self.assertEqual(min_window_subsequence("abcdebdde", "bde"), "bcde")

    def test_target_too_long(self):
        self.assertEqual(min_window_subsequence("a", "aaa"), "")

    def test_duplicated_char_in_target(self):
        self.assertEqual(min_window_subsequence("abbbbabbbabbababbbb", "aa"), "aba")

    def test_duplicated_char_but_no_matching(self):
        self.assertEqual(min_window_subsequence("ccccabbbbabbbabbababbbbcccc", "aca"), "")

    def test_match_last_char(self):
        self.assertEqual(min_window_subsequence("abcdef", "f"), "f")

    def test_match_first_char(self):
        self.assertEqual(min_window_subsequence("abcdef", "a"), "a")

    def test_equal_length_string(self):
        self.assertEqual(min_window_subsequence("abc", "abc"), "abc")
        self.assertEqual(min_window_subsequence("abc", "bca"), "")


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 12, 2019 LC 230 \[Medium\] Kth Smallest Element in a BST
---
> **Question:** Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

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

**My thoughts:** In BST, the in-order traversal presents orders from smallest to largest. Thus you can use both recursion with global variable k or the following template for iterative in-order traversal.

**Template for Iterative In-order Traversal:**

```py
while True:
    if t is not None:
        stack.append(t)
        t = t.left
    elif stack:
        t = stack.pop()
        print t.val # this will give node value follows in-order traversal    
        t = t.right
    else:
        break
return None
```

**Solution with Iterative In-order Traversal:** [https://repl.it/@trsong/Kth-Smallest-Element-in-a-BST](https://repl.it/@trsong/Kth-Smallest-Element-in-a-BST)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def kth_smallest(tree, k):
    t = tree
    stack = []

    while True:
        if t is not None:
            stack.append(t)
            t = t.left
        elif stack:
            t = stack.pop()
            if k == 1:
                return t.val
            else:
                k -= 1
            t = t.right
        else:
            break
    return None


class KthSmallestSpec(unittest.TestCase):
    def test_example1(self):
        """
           3
          / \
         1   4
          \
           2
        """
        n1 = TreeNode(1, right=TreeNode(2))
        tree = TreeNode(3, n1, TreeNode(4))
        check_expected = [1, 2, 3, 4]
        for e in check_expected:
            self.assertEqual(kth_smallest(tree, e), e)

    def test_example2(self):
        """
              5
             / \
            3   6
           / \
          2   4
         /
        1
        """
        n2 = TreeNode(2, TreeNode(1))
        n3 = TreeNode(3, n2, TreeNode(4))
        tree = TreeNode(5, n3, TreeNode(6))
        check_expected = [1, 2, 3, 4, 5, 6]
        for e in check_expected:
            self.assertEqual(kth_smallest(tree, e), e)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 11, 2019 LC 684 \[Medium\] Redundant Connection
---
> **Question:** In this problem, a tree is an **undirected** graph that is connected and has no cycles.
>
> The given input is a graph that started as a tree with N nodes (with distinct values 1, 2, ..., N), with one additional edge added. The added edge has two different vertices chosen from 1 to N, and was not an edge that already existed.
>
> The resulting graph is given as a 2D-array of edges. Each element of edges is a pair [u, v] with u < v, that represents an undirected edge connecting nodes u and v.
>
> Return an edge that can be removed so that the resulting graph is a tree of N nodes. If there are multiple answers, return the answer that occurs last in the given 2D-array. The answer edge [u, v] should be in the same format, with u < v.

**Example 1:**

```
Input: [[1,2], [1,3], [2,3]]
Output: [2,3]
Explanation: The given undirected graph will be like this:
  1
 / \
2 - 3
```

**Example 2:**

```
Input: [[1,2], [2,3], [3,4], [1,4], [1,5]]
Output: [1,4]
Explanation: The given undirected graph will be like this:
5 - 1 - 2
    |   |
    4 - 3
```

**My thoughts:** Process edges one by one, when we encouter an edge that has two ends already connected then adding current edge will form a cycle, then we return such edge. 

In order to efficiently checking connection between any two nodes. The idea is to keep track of all nodes that are already connected. Disjoint-set(Union Find) is what we are looking for. Initially UF makes all nodes disconnected, and whenever we encounter an edge, connect both ends. And we do that for all edges. 

**Solution with Disjoint-Set(Union-Find):** [https://repl.it/@trsong/CharmingCluelessApplets](https://repl.it/@trsong/CharmingCluelessApplets)
```py
import unittest

class UnionFind(object):
    def __init__(self, size):
        self.parent = range(size)
        self.rank = [0] * size
    
    def find(self, v):
        p = v
        while self.parent[p] != p:
            p = self.parent[p]
        self.parent[v] = p
        return p
    
    def union(self, v1, v2):
        p1 = self.find(v1)
        p2 = self.find(v2)
        if p1 != p2:
            if self.rank[p1] > self.rank[p2]:
                self.parent[p2] = p1
                self.rank[p1] += 1
            else:
                self.parent[p1] = p2
                self.rank[p2] += 1            
            
    def is_connected(self, v1, v2):
        return self.find(v1) == self.find(v2)

def find_redundant_connection(edges):
    uf = UnionFind(len(edges) + 1)
    for u, v in edges:
        if uf.is_connected(u, v):
            return [u, v]
        else:
            uf.union(u, v)
    return None


class FindRedundantConnectionSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(find_redundant_connection([[1,2], [1,3], [2,3]]), [2, 3])

    def test_example2(self):
        self.assertEqual(find_redundant_connection([[1,2], [2,3], [3,4], [1,4], [1,5]]), [1,4])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 10, 2019 LC 308 \[Hard\] Range Sum Query 2D - Mutable
---
> **Question:** Given a 2D matrix matrix, find the sum of the elements inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).

**Example:**

```py
Given matrix = [
  [3, 0, 1, 4, 2],
  [5, 6, 3, 2, 1],
  [1, 2, 0, 1, 5],
  [4, 1, 0, 1, 7],
  [1, 0, 3, 0, 5]
]

sumRegion(2, 1, 4, 3) -> 8
update(3, 2, 2)
sumRegion(2, 1, 4, 3) -> 10
```

**Solution with 2D Binary Indexed Tree:** [https://repl.it/@trsong/Range-Sum-Query-2D-Mutable](https://repl.it/@trsong/Range-Sum-Query-2D-Mutable)
```py
import unittest

class RangeSumQuery(object):
    def __init__(self, matrix):
        n, m = len(matrix), len(matrix[0])
        self.bit_matrix = [[0 for _ in xrange(m+1)] for _ in xrange(n+1)]
        for r in xrange(n):
            for c in xrange(m):
                self.update(r, c, matrix[r][c])

    def sumOriginToPosition(self, position):
        row, col = position
        res = 0
        rIdx = row + 1
        while rIdx > 0:
            cIdx = col + 1
            while cIdx > 0:
                res += self.bit_matrix[rIdx][cIdx]
                cIdx -= cIdx & -cIdx
            rIdx -= rIdx & -rIdx
        return res

    def sumRegion(self, row1, col1, row2, col2):
        top_left = (row1 - 1, col1 - 1)
        top_right = (row1 - 1, col2)
        bottom_left = (row2, col1 - 1)
        bottom_right = (row2, col2)
        top_left_sum = self.sumOriginToPosition(top_left)
        top_right_sum = self.sumOriginToPosition(top_right)
        bottom_left_sum = self.sumOriginToPosition(bottom_left)
        bottom_right_sum = self.sumOriginToPosition(bottom_right)
        return bottom_right_sum - bottom_left_sum - top_right_sum + top_left_sum

    def update(self, row, col, val):
        diff = val - self.sumRegion(row, col, row, col)
        n, m = len(self.bit_matrix), len(self.bit_matrix[0])
        rIdx = row + 1
        while rIdx < n:
            cIdx = col + 1
            while cIdx < m:
                self.bit_matrix[rIdx][cIdx] += diff
                cIdx += cIdx & -cIdx
            rIdx += rIdx & -rIdx


class RangeSumQuerySpec(unittest.TestCase):
    def test_example(self):
        matrix = [
            [3, 0, 1, 4, 2],
            [5, 6, 3, 2, 1],
            [1, 2, 0, 1, 5],
            [4, 1, 0, 1, 7],
            [1, 0, 3, 0, 5]
        ]
        rsq = RangeSumQuery(matrix)
        self.assertEqual(rsq.sumRegion(2, 1, 4, 3), 8)
        rsq.update(3, 2, 2)
        self.assertEqual(rsq.sumRegion(2, 1, 4, 3), 10)

    def test_non_square_matrix(self):
        matrix = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]
        rsq = RangeSumQuery(matrix)
        self.assertEqual(rsq.sumRegion(0, 1, 1, 3), 6)
        rsq.update(0, 2, 2)
        self.assertEqual(rsq.sumRegion(0, 1, 1, 3), 7)


if __name__ == '__main__':
    unittest.main(exit=False)
```
### Additional Question: LC 54 \[Medium\] Spiral Matrix 
---
> **Question:** Given a matrix of n x m elements (n rows, m columns), return all elements of the matrix in spiral order.

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

**Solution:** [https://repl.it/@trsong/Spiral-Matrix](https://repl.it/@trsong/Spiral-Matrix)
```py
import unittest

def spiral_order(matrix):
    if not matrix or not matrix[0]: return []
    n, m = len(matrix), len(matrix[0])
    row_lower_bound = 0
    row_upper_bound = n - 1
    col_lower_bound = 0
    col_upper_bound = m - 1
    res = []
        
    while row_lower_bound < row_upper_bound and col_lower_bound < col_upper_bound:
        r, c = row_lower_bound, col_lower_bound
        # top
        while c < col_upper_bound:
            res.append(matrix[r][c])
            c += 1
        
        # right
        while r < row_upper_bound:
            res.append(matrix[r][c])
            r += 1
        
        # bottom
        while c > col_lower_bound:
            res.append(matrix[r][c])
            c -= 1
        
        # left 
        while r > row_lower_bound:
            res.append(matrix[r][c])
            r -= 1
    
        col_upper_bound -= 1
        col_lower_bound += 1
        row_upper_bound -= 1
        row_lower_bound += 1
        
    r, c = row_lower_bound, col_lower_bound
    if row_lower_bound == row_upper_bound: 
        # Edge Case 1: when remaining block is 1xk
        for col in xrange(col_lower_bound, col_upper_bound + 1):
            res.append(matrix[r][col])
    elif col_lower_bound == col_upper_bound:
        # Edge Case 2: when remaining block is kx1
        for row in xrange(row_lower_bound, row_upper_bound + 1):
            res.append(matrix[row][c])
            
    return res


class SpiralOrderSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(spiral_order([
            [ 1, 2, 3 ],
            [ 4, 5, 6 ],
            [ 7, 8, 9 ]
        ]), [1, 2, 3, 6, 9, 8, 7, 4, 5])

    def test_example2(self):
        self.assertEqual(spiral_order([
            [1,  2,  3,  4],
            [5,  6,  7,  8],
            [9, 10, 11, 12]
        ]), [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7])

    def test_empty_table(self):
        self.assertEqual(spiral_order([]), [])
        self.assertEqual(spiral_order([[]]), [])

    def test_two_by_two_table(self):
        self.assertEqual(spiral_order([
            [1, 2],
            [4, 3]
        ]), [1, 2, 3, 4])

    def test_one_element_table(self):
        self.assertEqual(spiral_order([[1]]), [1])

    def test_one_by_k_table(self):
        self.assertEqual(spiral_order([
            [1, 2, 3, 4]
        ]), [1, 2, 3, 4])

    def test_k_by_one_table(self):
        self.assertEqual(spiral_order([
            [1],
            [2],
            [3]
        ]), [1, 2, 3])


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 9, 2019 LC 307 \[Medium\] Range Sum Query - Mutable
---
> **Question:** Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.
>
> The update(i, val) function modifies nums by updating the element at index i to val.

**Example:**

```py
Given nums = [1, 3, 5]

sumRange(0, 2) -> 9
update(1, 2)
sumRange(0, 2) -> 8
```

**Solution with Binary Indexed Tree:** [https://repl.it/@trsong/Range-Sum-Query-Mutable](https://repl.it/@trsong/Range-Sum-Query-Mutable)
```py
import unittest

class RangeSumQuery(object):
    @staticmethod
    def last_bit(num):
        return num & -num

    def __init__(self, nums):
        n = len(nums)
        self._BITree = [0] * (n + 1)
        for i in xrange(n):
            self.update(i, nums[i])

    def prefixSum(self, i):
        """Get sum of value from index 0 to i """
        # BITree starts from index 1
        index = i + 1
        res = 0
        while index > 0:
            res += self._BITree[index]
            index -= RangeSumQuery.last_bit(index)
        return res

    def rangeSum(self, i, j):
        return self.prefixSum(j) - self.prefixSum(i-1)

    def update(self, i, val):
        """Update the sum by add delta on top of result"""
        # BITree starts from index 1
        delta = val - self.rangeSum(i, i)
        index = i + 1
        while index < len(self._BITree):
            self._BITree[index] += delta
            index += RangeSumQuery.last_bit(index)


class RangeSumQuerySpec(unittest.TestCase):
    def test_example(self):
        rsq = RangeSumQuery([1, 3, 5])
        self.assertEqual(rsq.rangeSum(0, 2), 9)
        rsq.update(1, 2)
        self.assertEqual(rsq.rangeSum(0, 2), 8)

    def test_one_elem_array(self):
        rsq = RangeSumQuery([8])
        rsq.update(0, 2)
        self.assertEqual(rsq.rangeSum(0, 0), 2)

    def test_update_all_elements(self):
        req = RangeSumQuery([1, 4, 2, 3])
        self.assertEqual(req.rangeSum(0, 3), 10)
        req.update(0, 0)
        req.update(2, 0)
        req.update(1, 0)
        req.update(3, 0)
        self.assertEqual(req.rangeSum(0, 3), 0)
        req.update(2, 1)
        self.assertEqual(req.rangeSum(0, 1), 0)
        self.assertEqual(req.rangeSum(1, 2), 1)
        self.assertEqual(req.rangeSum(2, 3), 1)
        self.assertEqual(req.rangeSum(3, 3), 0)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Additional Question: LC 114 \[Medium\] Flatten Binary Tree to Linked List
---
> **Question:** Given a binary tree, flatten it to a linked list in-place.
>
> For example, given the following tree:

```py
    1
   / \
  2   5
 / \   \
3   4   6
```
>
> The flattened tree should look like:

```py
1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
```

**Solution with Recursion:** [https://repl.it/@trsong/Flatten-Binary-Tree-to-Linked-List](https://repl.it/@trsong/Flatten-Binary-Tree-to-Linked-List)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __eq__(self, other):
        return other and other.val == self.val and other.left == self.left and other.right == self.right


def flatten(tree):
    flatten_and_get_leaf(tree)


def flatten_and_get_leaf(tree):
    # flatten the tree and return the leaf node of that tree
    if not tree or not tree.left and not tree.right: return tree
    right_leaf = flatten_and_get_leaf(tree.right)
    left_leaf = flatten_and_get_leaf(tree.left)

    if left_leaf:
        # append right tree to the end of left tree leaf 
        left_leaf.right = tree.right
    if tree.left:
        # move left tree to right tree
        tree.right = tree.left
    tree.left = None

    # return tree right leaf if exits otherwise return left one 
    return right_leaf if right_leaf else left_leaf


class FlattenSpec(unittest.TestCase):
    def list_to_tree(self, lst):
        p = dummy = TreeNode(-1)
        for num in lst:
            p.right = TreeNode(num)
            p = p.right
        return dummy.right

    def test_example(self):
        """
            1
           / \
          2   5
         / \   \
        3   4   6
        """
        n2 = TreeNode(2, TreeNode(3), TreeNode(4))
        n5 = TreeNode(5, right = TreeNode(6))
        tree = TreeNode(1, n2, n5)
        flatten_list = [1, 2, 3, 4, 5, 6]
        flatten(tree)
        self.assertEqual(tree, self.list_to_tree(flatten_list))

    def test_empty_tree(self):
        tree = None
        flatten(tree)
        self.assertIsNone(tree)

    def test_only_right_child(self):
        """
        1
         \
          2
           \
            3
        """
        n2 = TreeNode(2, right=TreeNode(3))
        tree = TreeNode(1, right = n2)
        flatten_list = [1, 2, 3]
        flatten(tree)
        self.assertEqual(tree, self.list_to_tree(flatten_list))

    def test_only_left_child(self):
        """
            1
           /
          2
         /
        3
        """
        n2 = TreeNode(2, TreeNode(3))
        tree = TreeNode(1, n2)
        flatten_list = [1, 2, 3]
        flatten(tree)
        self.assertEqual(tree, self.list_to_tree(flatten_list))  


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 8, 2019 \[Medium\] Delete Columns to Make Sorted II
---
> **Question:** You are given an N by M 2D matrix of lowercase letters. The task is to count the number of columns to be deleted so that all the rows are lexicographically sorted.

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

**My thoughts:** This problem feels like 2D version of ***The Longest Increasing Subsequence Problem*** (LIP) (check Jul 2, 2019 problem for details). The LIP says find longest increasing subsequence. e.g. `01212342345` gives `01[2]1234[23]45`, with `2,2,3` removed. So if we only have 1 row, we can simply find longest increasing subsequence and use that to calculate how many columns to remove i.e. `# of columns to remove = m - LIP`. Similarly, for n by m table, we can first find longest increasing sub-columns and use that to calculate which columns to remove. Which can be done using DP:

let `dp[i]` represents max number of columns to keep at ends at column i. 
- `dp[i] = max(dp[j]) + 1 where j < i` if all characters in column `i` have lexicographical order larger than column `j`
- `dp[0] = 1`


**Solution with DP:** [https://repl.it/@trsong/Delete-Columns-to-Make-Sorted-II](https://repl.it/@trsong/Delete-Columns-to-Make-Sorted-II)
```py
import unittest

def delete_column(table):
    # let dp[i] represents max number of columns to keep at ends at column i
    # i.e. column i is the last column to keep
    # dp[i] = max(dp[j]) + 1 where j < i  if all e in column i have lexicographical order larger than column j
    m = len(table[0])
    dp = [1] * m
    for i in xrange(1, m):
        for j in xrange(i):
            if all(r[j] <= r[i] for r in table):
                dp[i] = max(dp[i], dp[j] + 1)
    return m - max(dp)


class DeleteColumnSpec(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(delete_column([
            'hello',
            'geeks'
        ]), 1)

    def test_example2(self):
        self.assertEqual(delete_column([
            'xyz',
            'lmn',
            'pqr'
        ]), 0)

    def test_table_with_one_row(self):
        self.assertEqual(delete_column([
            '01212342345' # 01[2]1234[23]45   
        ]), 3)  

    def test_table_with_two_rows(self):
        self.assertEqual(delete_column([
            '01012',  # [0] 1 [0] 1 2
            '20101'   # [2] 0 [1] 0 1
        ]), 2)  


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Additional Question: LT 640 \[Medium\] One Edit Distance
---
> **Question:** Given two strings S and T, determine if they are both one edit distance apart.

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

**My thougths:** The trick for this problem is that one insertion of shorter string is equivalent to one removal of longer string. And if both string of same length, then we only allow one replacement. The other cases are treated as more than one edit distance between two strings.

**Solution:** [https://repl.it/@trsong/One-Edit-Distance](https://repl.it/@trsong/One-Edit-Distance)
```py
import unittest

def is_one_edit_distance_between(s1, s2):
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > 1: return False
    i = distance = 0
    (shorter_str, longer_str) = (s1, s2) if len1 < len2 else (s2, s1)
    len_shorter_str = len(shorter_str)
    for c in longer_str:
        if distance > 1: return False
        if i >= len_shorter_str or c != shorter_str[i]:
            distance += 1
            if len1 != len2:
                # when two strings of different length and there is mismatch, only proceed the pointer to longer string
                i -= 1
        i += 1
    return distance == 1


class IsOneEditDistanceBetweenSpec(unittest.TestCase):
    def test_example1(self):
        self.assertTrue(is_one_edit_distance_between('aDb', 'adb'))

    def test_example2(self):
        self.assertFalse(is_one_edit_distance_between('ab', 'ab'))

    def test_empty_string(self):
        self.assertFalse(is_one_edit_distance_between('', ''))
        self.assertTrue(is_one_edit_distance_between('', 'a'))
        self.assertFalse(is_one_edit_distance_between('', 'ab'))

    def test_one_insert_between_two_strings(self):
        self.assertTrue(is_one_edit_distance_between('abc', 'ac'))
        self.assertFalse(is_one_edit_distance_between('abcd', 'ad'))

    def test_one_remove_between_two_strings(self):
        self.assertTrue(is_one_edit_distance_between('abcd', 'abd'))
        self.assertFalse(is_one_edit_distance_between('abcd', 'cd'))

    def test_one_replace_between_two_string(self):
        self.assertTrue(is_one_edit_distance_between('abc', 'abd'))
        self.assertFalse(is_one_edit_distance_between('abc', 'ddc'))

    def test_length_difference_greater_than_one(self):
        self.assertFalse(is_one_edit_distance_between('abcd', 'abcdef'))


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 7, 2019 \[Easy\] Delete Columns to Make Sorted I
---
> **Question:** You are given an N by M 2D matrix of lowercase letters. Determine the minimum number of columns that can be removed to ensure that each row is ordered from top to bottom lexicographically. That is, the letter at each column is lexicographically later as you go down each row. It does not matter whether each row itself is ordered lexicographically.

**Example 1:**

```
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


```
Given the following table:
abcdef

Your function should return 0, since the rows are already ordered (there's only one row).
```

**Example 3:**

```
Given the following table:
zyx
wvu
tsr

Your function should return 3, since we would need to remove all the columns to order it.
```

**My thoughts:** Wordy this problem may be, but extremely easy to be solved. We will take a look at *Delete Columns to Make Sorted II* tomorrow.

**Solution:** [https://repl.it/@trsong/Delete-Columns-to-Make-Sorted-I](https://repl.it/@trsong/Delete-Columns-to-Make-Sorted-I)
```py
import unittest

def columns_to_delete(table):
    if not table or not table[0]: return 0
    n, m = len(table), len(table[0])
    count = 0
    for c in xrange(m):
        for r in xrange(1, n):
            if table[r][c] < table[r-1][c]:
                count += 1
                break
    return count

class ColumnToDeleteSpec(unittest.TestCase):
    def test_empty_table(self):
        self.assertEqual(columns_to_delete([]), 0)
        self.assertEqual(columns_to_delete([""]), 0)

    def test_example1(self):
        self.assertEqual(columns_to_delete([
            'cba',
            'daf',
            'ghi'
        ]), 1)

    def test_example2(self):
        self.assertEqual(columns_to_delete([
            'abcdef'
        ]), 0)

    def test_example3(self):
        self.assertEqual(columns_to_delete([
            'zyx',
            'wvu',
            'tsr'
        ]), 3)


if __name__ == '__main__':
    unittest.main(exit=False)     
```

### Aug 6, 2019 LC 236 \[Medium\] Lowest Common Ancestor of a Binary Tree
---
> **Question:** Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

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

**My thoughts:** Notice that only the nodes at the same level can find common ancestor with same number of tracking backward. e.g. Consider 3 and 4 in above example: the common ancestor is 1, 3 needs 1 tracking backward, but 4 need 2 tracking backward. So the idea is to move those two nodes to the same level and then tacking backward until hit the common ancestor. The algorithm works as below:

We can use BFS to find target nodes and their depth. And by tracking backward the parent of the deeper node, we can make sure both of nodes are on the same level. Finally, we can tracking backwards until hit a common ancestor. 

**Solution with BFS and Backward Tracking Ancestor:** [https://repl.it/@trsong/Lowest-Common-Ancestor-of-a-Binary-Tree](https://repl.it/@trsong/Lowest-Common-Ancestor-of-a-Binary-Tree)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def LCA(tree, v1, v2):
    if v1 == v2: return v1
    parent = {}
    n1 = n2 = None
    lv1 = lv2 = lv = 0
    queue = [tree]
    
    # Run BFS to find node with value v1 and v2 and its depth
    while queue and (n1 is None or n2 is None):
        level_size = len(queue)
        for _ in xrange(level_size):
            node = queue.pop(0)
            if node.val == v1:
                n1 = node
                lv1 = lv
            elif node.val == v2:
                n2 = node
                lv2 = lv
            
            if node.left:
                parent[node.left] = node
                queue.append(node.left)
            if node.right:
                parent[node.right] = node
                queue.append(node.right)
        lv += 1
    
    # Backtrack the parent of deeper node up until at the same level as the other node
    (deeper_node, other_node) = (n1, n2) if lv1 > lv2 else (n2, n1)
    for _ in xrange(abs(lv1 - lv2)):
        deeper_node = parent[deeper_node]

    # Find the ancestor of both nodes recursively until find the common ancestor
    while deeper_node != other_node:
        deeper_node = parent[deeper_node]
        other_node = parent[other_node]

    return deeper_node.val


class LCASpec(unittest.TestCase):
    def setUp(self):
        """
             1
           /   \
          2     3
         / \   / \
        4   5 6   7
        """
        n2 = TreeNode(2, TreeNode(4), TreeNode(5))
        n3 = TreeNode(3, TreeNode(6), TreeNode(7))
        self.tree = TreeNode(1, n2, n3)

    def test_both_nodes_on_leaves(self):
        self.assertEqual(LCA(self.tree, 4, 5), 2)
        self.assertEqual(LCA(self.tree, 6, 7), 3)
        self.assertEqual(LCA(self.tree, 4, 6), 1)

    def test_nodes_on_different_levels(self):
        self.assertEqual(LCA(self.tree, 4, 2), 2)
        self.assertEqual(LCA(self.tree, 4, 3), 1)
        self.assertEqual(LCA(self.tree, 4, 1), 1)

    def test_same_nodes(self):
        self.assertEqual(LCA(self.tree, 2, 2), 2)
        self.assertEqual(LCA(self.tree, 6, 6), 6)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 5, 2019 \[Easy\] Single Bit Switch
---
> **Question:** Given three 32-bit integers x, y, and b, return x if b is 1 and y if b is 0, using only mathematical or bit operations. You can assume b can only be 1 or 0.

**Solution:** [https://repl.it/@trsong/Single-Bit-Switch](https://repl.it/@trsong/Single-Bit-Switch)
```py
import unittest

def single_bit_switch_1(b, x, y):
    return b * x + (1 - b) * y


def single_bit_switch_2(b, x, y):
    # When b = 0001b,
    # -b = 1111b, ~-b = 0000b
    # When b = 0000b
    # -b = 0000b, ~-b = 1111b
    return (x & -b) | (y & ~-b) 


class SingleBitSwitchSpec(unittest.TestCase):
    def test_b_is_zero(self):
        b, x, y = 0, 8, 16
        self.assertEqual(single_bit_switch_1(b, x, y), y)
        self.assertEqual(single_bit_switch_2(b, x, y), y)

    def test_b_is_one(self):
        b, x, y = 1, 8, 16
        self.assertEqual(single_bit_switch_1(b, x, y), x)
        self.assertEqual(single_bit_switch_2(b, x, y), x)

    def test_negative_numbers(self):
        b0, b1, x, y = 0, 1, -1, -2
        self.assertEqual(single_bit_switch_1(b0, x, y), y)
        self.assertEqual(single_bit_switch_2(b1, x, y), x)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 4, 2019 LC 392 \[Medium\] Is Subsequence
---
> **Question:** Given a string s and a string t, check if s is subsequence of t.
>
> A subsequence of a string is a new string which is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (ie, "ace" is a subsequence of "abcde" while "aec" is not).

**Example 1:**
```
s = "abc", t = "ahbgdc"
Return true.
```

**Example 2:**
```
s = "axc", t = "ahbgdc"
Return false.
```

**My thoughts:** Just base on the definition of subsequence, mantaining an pointer of s to the next letter we are about to checking and check it against all letter of t.

**Solution:** [https://repl.it/@trsong/Is-Subsequence](https://repl.it/@trsong/Is-Subsequence)

```py
import unittest

def isSubsequence(s, t):
    if len(s) > len(t): return False
    if not len(s): return True
    i = 0
    for c in t:
        if i >= len(s):
            break
        if c == s[i]:
            i += 1
    return i >= len(s)


class IsSubsequenceSpec(unittest.TestCase):
    def test_empty_s(self):
        self.assertTrue(isSubsequence("", ""))
        self.assertTrue(isSubsequence("", "a"))

    def test_empty_t(self):
        self.assertFalse(isSubsequence("a", ""))

    def test_s_longer_than_t(self):
        self.assertFalse(isSubsequence("ab", "a"))

    def test_size_one_input(self):
        self.assertTrue(isSubsequence("a", "a"))
        self.assertFalse(isSubsequence("a", "b"))

    def test_end_with_same_letter(self):
        self.assertTrue(isSubsequence("ab", "aaaaccb"))

    def test_example(self):
        self.assertTrue(isSubsequence("abc", "ahbgdc"))

    def test_example2(self):
        self.assertFalse(isSubsequence("axc", "ahbgdc"))


if __name__ == '__main__':
    unittest.main(exit=False)
```
### Aug 3, 2019 \[Medium\] Toss Biased Coin
---
> **Question:** Assume you have access to a function toss_biased() which returns 0 or 1 with a probability that's not 50-50 (but also not 0-100 or 100-0). You do not know the bias of the coin. Write a function to simulate an unbiased coin toss.

**My thoughts:** Suppose the biased toss has probablilty p to return 0 and (1-p) to get 1. Then the probability to get:

- 0, 0 is `p * p`
- 1, 1 is `(1-p) * (1-p)`
- 1, 0 is `(1-p) * p`
- 0, 1 is `p * (1-p)`
  
Thus we can take advantage that 1, 0 and 0, 1 has same probility to get unbiased toss. Of course, above logic works only if p is neither 0% nor 100%. 


**Solution:** [https://repl.it/@trsong/Toss-Biased-Coin](https://repl.it/@trsong/Toss-Biased-Coin)
```py
from random import randint

def toss_biased():
    # suppose the toss has 1/4 chance to get 0 and 3/4 to get 1
    return 0 if randint(0, 3) == 0 else 1


def toss_unbiased():
    while True:
        t1 = toss_biased()
        t2 = toss_biased()
        if t1 != t2:
            return t1


def print_distribution(repeat):
    histogram = {}
    for _ in xrange(repeat):
        res = toss_unbiased()
        if res not in histogram:
            histogram[res] = 0
        histogram[res] += 1
    print histogram


def main():
    # Distribution looks like {0: 99931, 1: 100069}
    print_distribution(repeat=200000)


if __name__ == '__main__':
    main()
```

### Aug 2, 2019 \[Medium\] The Tower of Hanoi
---
> **Question:**  The Tower of Hanoi is a puzzle game with three rods and n disks, each a different size.
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

```
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

**Solution with Recursion:** [https://repl.it/@trsong/The-Tower-of-Hanoi](https://repl.it/@trsong/The-Tower-of-Hanoi)
```py
import unittest

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

    def can_moves_finish_game(self, actions):
        self.reset()
        for src, dst in actions:
            if not self.is_feasible_move(src, dst):
                return False
            else:
                self.move(src, dst)
        return self.is_game_finished()
    

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


def hanoi_moves(n):
    moves = []

    def hanoi_moves_recur(n, src, dst):
        if n <= 0: return
        other = 3 - src - dst

        # Step1: move n - 1 disks from src to 'other' to allow last disk move to dst
        hanoi_moves_recur(n-1, src, other)

        # Step2: move last disk from src to dst
        moves.append((src, dst))

        # Step3: move n - 1 disks from 'other' to dst
        hanoi_moves_recur(n-1, other, dst)

    hanoi_moves_recur(n, 0, 2)
    return moves


class HanoiMoveSpec(unittest.TestCase):
    def assert_hanoi_moves(self, n, moves):
        game = HanoiGame(n)
        self.assertTrue(game.can_moves_finish_game(moves))

    def test_three_disks(self):
        moves = hanoi_moves(3)
        self.assert_hanoi_moves(3, moves)

    def test_one_disk(self):
        moves = hanoi_moves(1)
        self.assert_hanoi_moves(1, moves)

    def test_ten_disks(self):
        moves = hanoi_moves(10)
        self.assert_hanoi_moves(10, moves)


if __name__ == '__main__':
    unittest.main(exit=False)
```

### Aug 1, 2019 \[Medium\] All Root to Leaf Paths in Binary Tree
---
> **Question:** Given a binary tree, return all paths from the root to leaves.
>
> For example, given the tree:

```py
   1
  / \
 2   3
    / \
   4   5
```
> Return `[[1, 2], [1, 3, 4], [1, 3, 5]]`.

**Solution with Recursion:** [https://repl.it/@trsong/All-Root-to-Leaf-Paths-in-Binary-Tree](https://repl.it/@trsong/All-Root-to-Leaf-Paths-in-Binary-Tree)
```py
import unittest

class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right= right


def path_to_leaves(tree):
    if not tree: return []
    res = []
    def path_to_leaves_recur(tree, root_to_parent):
        root_to_current = root_to_parent + [tree.val]
        if not tree.left and not tree.right: 
            res.append(root_to_current)
        else:
            if tree.left:
                path_to_leaves_recur(tree.left, root_to_current)
            if tree.right:
                path_to_leaves_recur(tree.right, root_to_current)
    
    path_to_leaves_recur(tree, [])
    return res


class PathToLeavesSpec(unittest.TestCase):
    def test_empty_tree(self):
        self.assertEqual(path_to_leaves(None), [])

    def test_one_level_tree(self):
        self.assertEqual(path_to_leaves(TreeNode(1)), [[1]])

    def test_two_level_tree(self):
        tree = TreeNode(1, TreeNode(2), TreeNode(3))
        self.assertEqual(path_to_leaves(tree), [[1, 2], [1, 3]])

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
        self.assertEqual(path_to_leaves(tree), [[1, 2], [1, 3, 4], [1, 3, 5]])


if __name__ == '__main__':
    unittest.main(exit=False)
```
