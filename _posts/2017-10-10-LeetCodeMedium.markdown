---
layout: post
title:  "LeetCode Questions - Medium"
date:   2017-10-10 22:36:32 -0700
categories: Scala TypeScript
---
* This will become a table of contents (this text will be scraped).
{:toc}

### Environment Setup

TypeScript Playground: [https://www.typescriptlang.org/play/](https://www.typescriptlang.org/play/)

Scala Playground: [https://scastie.scala-lang.org/](https://scastie.scala-lang.org/)

Covert Tabs to Spaces in Code Snippets: [http://tabstospaces.com/](http://tabstospaces.com/)

### 534. Design TinyURL
Source: [https://leetcode.com/problems/design-tinyurl/description/](https://leetcode.com/problems/design-tinyurl/description/)

How would you design a URL shortening service that is similar to [TinyURL](https://en.wikipedia.org/wiki/TinyURL)?

**Background:**

TinyURL is a URL shortening service where you enter a URL such as `https://leetcode.com/problems/design-tinyurl` and it returns a short URL such as `http://tinyurl.com/4e9iAk`.

**Requirements:**

For instance, `"http://tinyurl.com/4e9iAk"` is the tiny url for the page `"https://leetcode.com/problems/design-tinyurl"`. The identifier can be any string with 6 alphanumeric characters containing `0-9`, `a-z`, `A-Z`.
Each shortened URL must be unique; that is, no two different URLs can be shortened to the same URL.

**TypeScript Solution:**

```typescript
const DICTIONARY: string = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

const URL_PREFIX: string = "http://tiny.url/";

const CODE_0: number = "0".charCodeAt(0);
const CODE_9: number = "9".charCodeAt(0);
const CODE_LOWER_A: number = "a".charCodeAt(0);
const CODE_LOWER_Z: number = "z".charCodeAt(0);
const CODE_UPPER_A: number = "A".charCodeAt(0);
const CODE_UPPER_Z: number = "Z".charCodeAt(0);

class URLService {
    private _stol: Map<number, string>;
    private _counter: number;

    constructor() {
        this._stol = new Map<number, string>();
        this._counter = 1;
    }

    public longToShort(url: string): string {
        let shorturl = URLService.base10ToBase62(this._counter);
        this._stol.set(this._counter, url);
        this._counter++;
        return URL_PREFIX + shorturl;
    }

    public shortToLong(url: string): string {
        let urlParam: string = url.substring(URL_PREFIX.length);
        let n: number = URLService.base62ToBase10(urlParam);
        return this._stol.get(n);
    }

    public static convert(charCode: number): number {
        if (charCode >= CODE_0 && charCode <= CODE_9) {
            return charCode - CODE_0;
        } else if (charCode >= CODE_LOWER_A && charCode <= CODE_LOWER_Z) {
            return charCode - CODE_LOWER_A + 10;
        } else if (charCode >= CODE_UPPER_A && charCode <= CODE_UPPER_Z) {
            return charCode - CODE_UPPER_A + 36;
        } else {
            return -1;
        }
    }

    public static base62ToBase10(s: string): number {
        let n: number = 0;
        for (let i: number = 0; i < s.length; i++) {
            n = n * 62 + URLService.convert(s.charCodeAt(i));
        }
        return n;
    }

    public static base10ToBase62(n: number): string {
        let s: string = "";
        while (n !== 0) {
            let r: number = n % 62;
            n = (n - r) / 62;
            s = DICTIONARY[r] + s;
        }
        while (s.length < 6) {
            s = "0" + s;
        }
        return s;
    }
}

function exec() {
    const service: URLService = new URLService();
    let result: string = service.shortToLong(service.longToShort("http://trsong.github.io/"));

    let div: HTMLElement = document.createElement("div");
    div.innerText = result;
    document.body.appendChild(div);
}

exec();
```

**Scala Soluction:**

```scala
import scala.collection.mutable.{StringBuilder, HashMap}

class URLService {
    var counter: BigInt = BigInt(1)
    val stol = HashMap.empty[BigInt, String]
    
    def longToShort(url: String): String = {
        val shorturl = URLService.base10ToBase62(counter)
        stol(counter) = url
        counter += 1
        URLService.urlPrefix + shorturl
    }
    
    def shortToLong(url: String): String = {
        val urlParam = url.substring(URLService.urlPrefix.length)
        stol.getOrElse(URLService.base62ToBase10(urlParam), URLService.urlPrefix)
    }
}

object URLService {
    val dictionary = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    val urlPrefix = "http://tiny.url/"
    
    def convert(c: Char): Int = c match {
        case num if num >= '0' && num <= '9' => num - '0'
        case lower if lower >= 'a' && lower <= 'z' => lower - 'a' + 10
        case upper if upper >= 'A' && upper <= 'Z' => upper - 'A' + 36
        case _ => -1
    }
    
    def base62ToBase10(s: String): BigInt = {
        (BigInt(0) /: s)(62 * _ + convert(_) )
    }
    
    def base10ToBase62(n: BigInt): String = {
        val sb = new StringBuilder
        var target = n
        while (target != 0) {
            sb += dictionary((target % 62).toInt)
            target = target / 62
        }
        while (sb.size < 6) {
            sb += '0'
        }
        sb.toString().reverse
    }
}

object Main extends App {
    val service = new URLService
    println(service.shortToLong(service.longToShort("http://trsong.github.io/")))
}
```

### 8. String to Integer (atoi)

Source: [https://leetcode.com/problems/string-to-integer-atoi/description/](https://leetcode.com/problems/string-to-integer-atoi/description/)

Implement atoi to convert a string to an integer.

**Hint:** Carefully consider all possible input cases. If you want a challenge, please do not see below and ask yourself what are the possible input cases.

**Requirements for atoi:**

The function first discards as many whitespace characters as necessary until the first non-whitespace character is found. Then, starting from this character, takes an optional initial plus or minus sign followed by as many numerical digits as possible, and interprets them as a numerical value.

The string can contain additional characters after those that form the integral number, which are ignored and have no effect on the behavior of this function.

If the first sequence of non-whitespace characters in str is not a valid integral number, or if no such sequence exists because either str is empty or it contains only whitespace characters, no conversion is performed.

If no valid conversion could be performed, a zero value is returned. If the correct value is out of the range of representable values, INT_MAX (2147483647) or INT_MIN (-2147483648) is returned.

**TypeScript Solution:**

```typescript
const INT_MAX: number = 2147483647;
const INT_MIN: number = -2147483648;

function atoi(str: string): number {
    let index: number = 0, sign: number = 1, total: number = 0;
    
    // 1. Empty string
    if (!str || str.length === 0) return 0;
    
    // 2. Remove whitespace
    while(index < str.length && str[index] === ' ') index++;
    
    // 3. Handle signs
    while (index < str.length && (str[index] === '+' || str[index] === '-')) {
        sign = str[index] === '+' ? sign : -sign;
        index++;
    }
    
    // 4. Convert number and avoid overflow
    while (index < str.length) {
        let digit: number = str.charCodeAt(index) - "0".charCodeAt(0);
        if (digit < 0 || digit > 9) {
            total = 0;
            break;
        } 
        total = 10 * total + digit;
        index++;
        
        if (total > INT_MAX && sign > 0) return INT_MAX;
        else if (total > -INT_MIN && sign < 0) return INT_MIN;
    }
    return total * sign;
}

function exec() {
    let result: number = atoi("   -+-++-42");

    let div: HTMLElement = document.createElement("div");
    div.innerText = result.toString();
    document.body.appendChild(div);
}

exec();
```

**Scala Soluction:** 

```scala
object Main extends App {
    val INT_MAX: Int = 2147483647
    val INT_MIN: Int = -2147483648

    def atoi(str: String): Int = {
        var index = 0
        var sign = 1
          var total = 0
          var isValid = true
  
        // 1. Empty string
    
        // 2. Remove whitespaces
        while (index < str.length && str(index) == ' ') {
            index += 1
        }
    
        // 3. Handle sign
        while (index < str.length && (str(index) == '+' || str(index) == '-')) {
            sign = if (str(index) == '+') sign else -sign
            index += 1
        }
    
        // 4. Convert number and avoid overflow
        while (index < str.length) {
            val digit = str(index) - '0'
            if (!isValid || digit < 0 || digit > 9) {
                isValid = false
            } else if (sign > 0 && digit > (INT_MAX - digit) / 10) {
                total = INT_MAX
            } else if (sign < 0 && digit > (- digit - INT_MIN) / 10) {
                total = INT_MIN
            } else {
                total = 10 * total + digit
            }
            index += 1
        }
    
        if (isValid) total * sign else 0
    }

    println(atoi("2147483647"))
    println(atoi("-2147483648"))
    println(atoi("42"))
    println(atoi("    -++----42"))
    println(atoi(""))
    println(atoi("  +  -  - + "))
}
```

### 151. Reverse Words in a String

Source: [https://leetcode.com/problems/reverse-words-in-a-string/description/](https://leetcode.com/problems/reverse-words-in-a-string/description/)

Given an input string, reverse the string word by word.

For example,
Given s = "`the sky is blue`",
return "`blue is sky the`".

Try to solve it in-place in O(1) space.


**Clarification:**

* Q: What constitutes a word?
 A: A sequence of non-space characters constitutes a word.

* Q: Could the input string contain leading or trailing spaces?
A: Yes. However, your reversed string should not contain leading or trailing spaces.

* Q: How about multiple spaces between two words?
A: Reduce them to a single space in the reversed string.


**TypeScript Solution:**

```typescript
const CODE_WHITE_SPACE: number = " ".charCodeAt(0);

class Solution {
    public static reverseWords(s: string): string {
        if (!s) return undefined;
        let a: number[] = Array.from({length: s.length}, (_, index: number) => s.charCodeAt(index));
        
        // 1. Reverse the whole string
        Solution.reverse(a, 0, a.length - 1);
        
        // 2. Reverse each word in place
        Solution.reverseEachWordInPlace(a);
        
        // 3. Clean up spaces and return
        return Solution.cleanSpaces(a);
    }
    
    // reverse a[] from a[i] to a[j]
    private static reverse(a: number[], i: number, j: number): void {
        while (i < j) {
            [a[i++], a[j--]] = [a[j], a[i]]
        }
    }
    
    // scan through and reverse each word in place
    private static reverseEachWordInPlace(a: number[]): void {
        let i: number = 0, j: number = 0;
        
        while (i < a.length) {
            while (i < j || i < a.length && a[i] === CODE_WHITE_SPACE) i++; // Skip spaces
            while (j < i || j < a.length && a[j] !== CODE_WHITE_SPACE) j++; // Skip non spaces
            Solution.reverse(a, i, j - 1); // reverse the word
        }
    }
    
    // trim leading, trailling and multiple spaces
    private static cleanSpaces(a: number[]): string {
        let i: number = 0, j: number = 0;
        
        while (j < a.length) {
            while (j < a.length && a[j] === CODE_WHITE_SPACE) j++; // skip spaces
            while (j < a.length && a[j] !== CODE_WHITE_SPACE) a[i++] = a[j++]; // keep non spaces 
            while (j < a.length && a[j] === CODE_WHITE_SPACE) j++; // skip spaces 
            if (j < a.length) a[i++] = CODE_WHITE_SPACE; // keep only one space between consecutive word
        }
        
        return a.slice(0, i).map((code: number) => String.fromCharCode(code)).join('');
    }
}

function exec() {
    let result: string = Solution.reverseWords("     Hello    World !");
    let div: HTMLElement = document.createElement("div");
    div.innerText = result;
    document.body.appendChild(div);
}

exec();
```

**Scala Soluction:** 

```scala
import scala.collection.mutable.Buffer

object Main extends App {
    def reverseWords(s: String): String = {
        val sb = s.toBuffer
            
        // 1. reverse the input string
        reverse(sb, 0, sb.size - 1)
            
        // 2. reverse each word in place
        reverseEachWordInPlace(sb)
            
        // 3. clean spaces and return
        cleanSpaces(sb)
    }
    
    // Reverse the sb from sb(start) to sb(end)
    def reverse(sb: Buffer[Char], start: Int, end: Int): Unit = {
        var (i, j) = (start, end)
        while (i < j) {
            val tmp = sb(i)
            sb(i) = sb(j)
            sb(j) = tmp
            i += 1
            j -= 1
        }
    }
    
    // Scan through each word and reverse them in place
    def reverseEachWordInPlace(sb: Buffer[Char]): Unit = {
        var (i, j) = (0, 0)
        while (i < sb.size) {
            while (i < j || i < sb.size && sb(i) == ' ') i += 1
            while (j < i || j < sb.size && sb(j) != ' ') j += 1
            reverse(sb, i, j - 1)
        }
    }
    
    // Remove the leading, tailing and multiple white spaces
    def cleanSpaces(sb: Buffer[Char]): String = {
        var (i, j) = (0, 0)
        while (j < sb.size) {
            while (j < sb.size && sb(j) == ' ') j += 1
            while (j < sb.size && sb(j) != ' ') {
                sb(i) = sb(j)
                i += 1
                j += 1
            }
            while (j < sb.size && sb(j) == ' ') j += 1
            if (j < sb.size) {
                sb(i) = ' '
                i += 1
            }
        }
        sb.take(i).mkString
    }
    
    println(reverseWords("    Hello    World !"))
}
```

### 29. Divide Two Integers

Source: [https://leetcode.com/problems/divide-two-integers/description/](https://leetcode.com/problems/divide-two-integers/description/)

Divide two integers without using multiplication, division and mod operator.

If it is overflow, return MAX_INT.

**Hint:** 

```
n = n0 + d * 2 ^ m0            where d <= n0 < 2d, n > d
n0 = n1 + d * 2 ^ m1        where d <= n1 < 2d, n0 > d
...
n_{k-1} = nk + d * 2 ^ 0  where 0 <= nk < d, 

ans = 2^m0 + 2^m1 + ... + 1 

eg. 
divide(16, 3)

16 = 4 + 3 * 2^2  
4 = 1 + 3 * 2^0

ans = 2^2 + 2^0 = 5


Note: 
n = d * ans = d * (1 << m0 + 1 << m1 + ... + 1 << 0)

worese case: n = 2^p - 1, d = 1

Time Complexity: O((Log N) ^ 2)
N in binary has Log(N) digits
So we need to have Log(N) step with each step spend Log(N) to figure out m
```

**TypeScript Solution:**

```typescript
function divide(n: number, d: number): number {
    // Edge case optimization
    if (d == 1) return n;
    
    // Error/Overflow handling
    if (d == 0 || (n == Number.MIN_SAFE_INTEGER && d == -1)) return Number.MAX_SAFE_INTEGER;

    let result: number = 0;
    let n0: number = Math.abs(n);
    let d0: number = Math.abs(d);
    while (n0 >= d0) {
        let a: number = d0;
        let m: number = 1;
        while ((a << 1) < n0) {
            a <<= 1;
            m <<= 1;
        }
        result += m;
        n0 -= a;
    }
    
    if ((n > 0) !== (d > 0)) { // XOR
        result = -result;
    }
    return result;
}

function exec() {
    let result: number = divide(16, 3);

    let div: HTMLElement = document.createElement("div");
    div.innerText = result.toString();
    document.body.appendChild(div);
}

exec();
```

**Scala Soluction:** 

```scala
object Main extends App {
    def divide(n: Int, d: Int): Int = {
        if (d == 1) n        // edge case optimization
        else if (d == 0 || (n == Int.MinValue && d == -1)) Int.MaxValue // Handling overflow
        else {
            var result = 0
            var n0 = Math.abs(n)
            var d0 = Math.abs(d)
            while (n0 > d0) {
                var a = d0
                var m = 1
                while ((a << 1) < n0) {
                    a <<= 1
                    m <<= 1
                }
                result += m
                n0 -= a
            }
            
            if ((n > 0) ^ (d > 0)) result = -result
            result
        }
    }
    
    println(divide(72, 1))
    println(divide(16, 3))
    println(divide(Int.MinValue, -1))
    println(divide(Int.MaxValue, 1))
}
```

### 166. Fraction to Recurring Decimal

Source: [https://leetcode.com/problems/fraction-to-recurring-decimal/description/](https://leetcode.com/problems/fraction-to-recurring-decimal/description/)


Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.

If the fractional part is repeating, enclose the repeating part in parentheses.

For example,

- Given numerator = 1, denominator = 2, return "0.5".
- Given numerator = 2, denominator = 1, return "2".
- Given numerator = 2, denominator = 3, return "0.(6)".
- Given numerator = 61, denominator = 30, return "2.0(3)".



**TypeScript Solution:**

```typescript
function fractionToDecimal(numerator: number, denominator: number): string {
    let result: string = "";
    let sign: string = (numerator < 0) !== (denominator < 0) ? "-" : "";
    let n: number = Math.abs(numerator);
    let d: number = Math.abs(denominator);
    
    result += sign;
    
    let remainder: number = n % d;
    if (remainder === 0) {
        result += n / d;
        return result;
    }
    
    result += (n - remainder) / d;
    result += ".";
    
    let remainderPosMap: Map<number, number> = new Map<number, number>();
    while (!remainderPosMap.has(remainder)) {
        remainderPosMap.set(remainder, result.length);
        let nextRemainder: number = remainder * 10 % d;
        result += (remainder * 10 - nextRemainder) / d;
        remainder = nextRemainder;
    } 
    let index: number = remainderPosMap.get(remainder);
    let prifixTerm: string = result.slice(0, index);
    let repeatedTerm: string = result.slice(index);
    
    if (repeatedTerm !== "0") {
        result = `${prifixTerm}(${repeatedTerm})`;
    } else {
        result = prifixTerm;
    }
    
    return result;
}

function exec() {
    let result: string = fractionToDecimal(61, 30);

    let div: HTMLElement = document.createElement("div");
    div.innerText = result;
    document.body.appendChild(div);
}

exec();
```

**Scala Soluction:** 

```scala
import scala.collection.mutable.{HashMap, StringBuilder}

object Main extends App {
    def fractionToDecimal(numerator: Int, denominator: Int): String = {
        val sb = StringBuilder.newBuilder
        if ((numerator < 0) ^ (denominator < 0)) sb += '-'
        val n = Math.abs(numerator)
        val d = Math.abs(denominator)
        sb.append(n / d)
        var remainder = n % d
        
        if (remainder == 0) sb.result
        else {
            sb += '.'
            
            val remainderPosMap = HashMap.empty[Int, Int]
            while (!remainderPosMap.contains(remainder)) {
                remainderPosMap(remainder) = sb.size
                sb.append(10 * remainder / d)
                remainder = 10 * remainder % d
            }
            
            val index = remainderPosMap(remainder)
            val repeatedTerm = sb.slice(index, sb.size)
            if (repeatedTerm.size == 1 && repeatedTerm(0) == '0') {
                sb.dropRight(1).result    // strip the tailing 0
            } else {
                sb.insert(index, '(')
                sb += ')'
                sb.result
            }
        }
    }
    
    println(fractionToDecimal(61, 30))
    println(fractionToDecimal(61, -30))
    println(fractionToDecimal(60, 30))
    println(fractionToDecimal(61, 10))
}
```

### 220. Contains Duplicate III

Source: [https://leetcode.com/problems/contains-duplicate-iii/description/](https://leetcode.com/problems/contains-duplicate-iii/description/)

Given an array of integers, find out whether there are two distinct indices i and j in the array such that the **absolute** difference between **nums[i]** and **nums[j]** is at most t and the **absolute** difference between i and j is at most k.

Eg. Exists different indices i, j such that abs(nums[i] - num[j]) <= t and abs(i - j) <= k 

**Hint:**

The idea is like the bucket sort algorithm. Suppose we have consecutive buckets covering the range of nums with each bucket a width of (t+1). If there are two item with difference <= t, one of the two will happen:

```
(1) the two in the same bucket
(2) the two in neighbor buckets
```
- For case (1) return true directly, since they are within the same bucket of t + 1
- For case (2) check if abs(nums[i] - num[j]) <= t
- For other case, we move on

Note, while we iterator through the list, we keep a index window of k, means any nums[j] with `j < i - k` will no longer take into consideration.

**TypeScript Solution:**

```typescript
function getBucketIndex(value: number, windowSize: number): number {
    return Math.floor(value < 0 ? value / windowSize - 1 : value / windowSize);
}

// Check if exists different indices i, j such that abs(nums[i] - num[j]) <= t and abs(i - j) <= k
function containsNearbyAlmostDuplicate(nums: number[], k: number, t: number): boolean {
    if (k < 1 || t < 0) return false;
    let bucket: Map<number, number> = new Map<number, number>();
    let w: number = t + 1;
    for (let i: number = 0; i < nums.length; i++) {
        let bucketIndex: number = getBucketIndex(nums[i], w);
        if (bucket.has(bucketIndex) ||
        (bucket.has(bucketIndex - 1) && Math.abs(bucket.get(bucketIndex - 1) - nums[i]) < w) ||
        (bucket.has(bucketIndex + 1) && Math.abs(bucket.get(bucketIndex + 1) - nums[i]) < w)) {
            return true;
        }
        
        bucket.set(bucketIndex, nums[i]);
        if (i >= k) bucket.delete(getBucketIndex(nums[i - k], w));
    }
    return false;
}

function exec() {
    let result: boolean = containsNearbyAlmostDuplicate([-1, -5, 7, 3, 9, -3], 2, 2);
    let result2: boolean = containsNearbyAlmostDuplicate([-1, -5, 7, 3, 9, -3], 1, 2);

    let div: HTMLElement = document.createElement("div");
    div.innerText = result.toString();
    div.innerText += " \n " + result2.toString();
    document.body.appendChild(div);
}

exec();
```

**Scala Soluction:** 

```scala
object Main extends App {
    // Check if exists different indices i, j such that abs(nums[i] - num[j]) <= t and abs(i - j) <= k
    def containsNearbyAlmostDuplicate(nums: IndexedSeq[Int], k: Int, t: Int): Boolean = {
        val windowSize = t + 1
        def getBucketIndex(value: Int) = if (value < 0) value / windowSize - 1 else value / windowSize
    
        if (k < 1 || t < 0) false
        else nums.zipWithIndex.foldLeft((false, Map.empty[Int, Int])) { (memo, valueAndIndex) =>
            val (result, bucket) = memo
            val (value, i) = valueAndIndex
            val bucketIndex = getBucketIndex(value)
            if (result || bucket.contains(bucketIndex) ||
                (bucket.contains(bucketIndex - 1) && Math.abs(bucket(bucketIndex - 1) - value) < windowSize) || 
                (bucket.contains(bucketIndex + 1) && Math.abs(bucket(bucketIndex + 1) - value) < windowSize)) {
                (true, bucket)
            } else {
                val updatedBucket = bucket + (bucketIndex -> value) 
                (false, if (i >= k) updatedBucket - getBucketIndex(nums(i - k)) else updatedBucket)
            }
        }._1
    }
    
    println(containsNearbyAlmostDuplicate(Array(-1, -5, 7, 3, 9, -3), 2, 2))
    println(containsNearbyAlmostDuplicate(Array(-1, -5, 7, 3, 9, -3), 1, 2))
}
```

### 127. Word Ladder

Source: [https://leetcode.com/problems/word-ladder/description/](https://leetcode.com/problems/word-ladder/description/)

Given two words (*beginWord* and *endWord*), and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord, such that:

1. Only one letter can be changed at a time.
2. Each transformed word must exist in the word list. Note that beginWord is not a transformed word.

For example,

```
Given:    
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]

As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
return its length 5.
```

**Note:**

- Return 0 if there is no such transformation sequence.
- All words have the same length.
- All words contain only lowercase alphabetic characters.
- You may assume no duplicates in the word list.
- You may assume beginWord and endWord are non-empty and are not the same.

**Hint:**

Well, this problem has a nice BFS structure.

Let's see the example in the problem statement.

`start = "hit"`

`end = "cog"`

`dict = ["hot", "dot", "dog", "lot", "log"]`

Since only one letter can be changed at a time, if we start from `"hit"`, we can only change to those words which have only one different letter from it, like `"hot"`. Putting in graph-theoretic terms, we can say that `"hot"` is a neighbor of `"hit"`.

The idea is simpy to begin from `start`, then visit its neighbors, then the non-visited neighbors of its neighbors... Well, this is just the typical BFS structure.

To simplify the problem, we insert `end` into `dict`. Once we meet `end` during the BFS, we know we have found the answer. We maintain a variable `dist` for the current distance of the transformation and update it by `dist++` after we finish a round of BFS search (note that it should fit the definition of the distance in the problem statement). Also, to avoid visiting a word for more than once, we erase it from `dict` once it is visited.


**TypeScript Solution: (One-way BFS)**

```typescript
const CODE_LOWER_A: number = "a".charCodeAt(0);

function addNextWords(word: string, wordDict: Set<string>, toVisitQueue: string[]): void {
    wordDict.delete(word);
    let w: number[] = Array.from({ length: word.length }, (_, index: number) => word.charCodeAt(index));
    for (let p: number = 0; p < word.length; p++) {
        let letter: number = word.charCodeAt(p);
        for (let k: number = 0; k < 26; k++) {
            w[p] = CODE_LOWER_A + k;
            let searchString: string = w.map(c => String.fromCharCode(c)).join("");
            if (wordDict.has(searchString)) {
                toVisitQueue.push(searchString);
                wordDict.delete(searchString);
            }
        }
        w[p] = letter;
    }
}

function ladderLength(beginWord: string, endWord: string, wordList: string[]): number {
    let dist: number = 2;
    let toVisitQueue: string[] = [];
    let wordDict: Set<string> = new Set<string>(wordList);
    
    wordDict.add(endWord);
    addNextWords(beginWord, wordDict, toVisitQueue);
    
    while (toVisitQueue.length !== 0) {
        let num: number = toVisitQueue.length;
        for (let i: number = 0; i < num; i++) {
            let word: string = toVisitQueue.shift();
            if (word === endWord) return dist;
            addNextWords(word, wordDict, toVisitQueue);
        }
        dist++;
    }
}

function exec() {
    let result: number = ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log"]);

    let div: HTMLElement = document.createElement("div");
    div.innerText = result.toString();
    document.body.appendChild(div);
}

exec();
```

**Scala Solution: (One-way BFS)**

```scala
import scala.collection.mutable.{Queue, Set, StringBuilder}

object Main extends App {
    def addNewWords(word: String, wordDict: Set[String], toVisitQueue: Queue[String]): Unit = {
        wordDict -= word
        val sb = new StringBuilder(word)
        
        (word.zipWithIndex) foreach { case (c0: Char, i: Int) =>
            ('a' to 'z') foreach { c1 =>
                sb(i) = c1
                val neighbourWord = sb.toString
                if (wordDict.contains(neighbourWord)) {
                    toVisitQueue.enqueue(neighbourWord);
                    wordDict -= neighbourWord
                }
            }
            sb(i) = c0
        }
    }

    def ladderLength(beginWord: String, endWord: String, wordList: Seq[String]): Int = {
        val wordDict = scala.collection.mutable.Set() ++ (endWord +: wordList)
        val toVisitQueue = Queue.empty[String]
        var dist = 2
        
        addNewWords(beginWord, wordDict, toVisitQueue)
        while (!toVisitQueue.isEmpty) {
            (0 until toVisitQueue.length).foreach { _ =>
                val word = toVisitQueue.dequeue
                if (word == endWord) return dist
                addNewWords(word, wordDict, toVisitQueue)
            }
            dist += 1
        }
        dist
    }
    
    println(ladderLength("hit", "cog", List("hot", "dot", "dog", "lot", "log")))
}
```

**Typescript Solution: (Two-way BFS)**

```typescript
const CODE_LOWER_A: number = "a".charCodeAt(0);

function ladderLength(beginWord: string, endWord: string, wordList: string[]): number { 
    let wordDict: Set<string> = new Set<string>(wordList);
    let beginSet: Set<string> = new Set<string>();
    let endSet: Set<string> = new Set<string>();
    let visited: Set<string> = new Set<string>();
    
    let len: number = 1;
    beginSet.add(beginWord);
    endSet.add(endWord);
    
    while(beginSet.size !== 0 && endSet.size !== 0) {
        if (beginSet.size > endSet.size) {
            let tmp: Set<string> = beginSet;
            beginSet = endSet;
            endSet = tmp;
        }
        
        let neighbours: Set<string> = new Set<string>();
        for (let word of Array.from(beginSet)) {
            let chs: number[] = Array.from({ length: word.length }, (_, i: number) => word.charCodeAt(i));
            for (let i: number = 0; i < chs.length; i++) {
                for (let c: number = 0; c < 26; c++) {
                    let oldChar: number = chs[i];
                    chs[i] = CODE_LOWER_A + c;
                    let target: string = chs.map((s: number) => String.fromCharCode(s)).join("");

                    if (endSet.has(target)) {
                        return len + 1;
                    }
                    
                    if (!visited.has(target) && wordDict.has(target)) {
                        neighbours.add(target);
                        visited.add(target);
                    }
                    
                    chs[i] = oldChar;
                }
            }
        }

        beginSet = neighbours;
        len++;
    }
    
    return 0;
}

function exec() {
    let result: number = ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log"]);

    let div: HTMLElement = document.createElement("div");
    div.innerText = result.toString();
    document.body.appendChild(div);
}

exec();
```

**Scala Solution: (Two-way BFS)**

```scala
import scala.collection.mutable.{Set, StringBuilder}

object Main extends App {
    def ladderLength(beginWord: String, endWord: String, wordList: Seq[String]): Int = {
        var wordDict = wordList.toSet
        var beginSet = Set(beginWord)
        var endSet = Set(endWord)
        var len = 1
        val visited = Set.empty[String]
        
        while (!beginSet.isEmpty && !endSet.isEmpty) {
            if (beginSet.size > endSet.size) {
                var tmp = beginSet
                beginSet = endSet
                endSet = tmp
            }
            
            val neighbour = Set.empty[String]
            beginSet.foreach { word =>
                val sb = new StringBuilder(word)
                (sb.zipWithIndex) foreach { case (c0: Char, i: Int) =>
                    ('a' to 'z') foreach { c1 =>
                        sb(i) = c1
                        val target = sb.toString
                        
                        if (endSet.contains(target)) {
                            return len + 1
                        }
                        
                        if (!visited.contains(target) && wordDict.contains(target)) {
                            visited += target
                            neighbour += target
                        }
                    }
                    sb(i) = c0
                }
            }
            beginSet = neighbour
            len += 1
        }
        
        len
    }
    
    println(ladderLength("hit", "cog", List("hot", "dot", "dog", "lot", "log")))
}
```

### 91. Decode Ways

Source: [https://leetcode.com/problems/decode-ways/description/](https://leetcode.com/problems/decode-ways/description/)

A message containing letters from **A-Z** is being encoded to numbers using the following mapping:

```
'A' -> 1
'B' -> 2
...
'Z' -> 26
```

Given an encoded message containing digits, determine the total number of ways to decode it.

For example,
Given encoded message **"12"**, it could be decoded as **"AB"** (1 2) or **"L"** (12).

The number of ways decoding **"12"** is 2.

**Hint:**

Use a dp array of size n + 1 to save subproblem solutions. `dp[0]` means an empty string will have one way to decode, `dp[1]` means the way to decode a string of size 1. I then check one digit and two digit combination and save the results along the way. In the end, `dp[n]` will be the end result.

```
"12": 
    1 2
    12
"126":
    1 2 6    
   12 6
 +
   1  26
   
numDecodings("126") = numDecodings("12") + numDecodings("1")

dp[n] = dp[n-1] + d[n-2] if last two digit can form a letter
dp[n] = dp[n-1] if last two digit cannot form a letter 
```

**Typescript Solution:**

```typescript
function numDecodings(s: string): number {
    if (!s || s.length === 0) return 0;
    
    let dp: number[] = new Array(10).fill(0);    
    dp[0] = 1;
    dp[1] = s[0] !== "0" ? 1 : 0;
    
    for (let i: number = 2; i <= s.length; i++) {
        let one: number = +s.substring(i - 1, i);
        let two: number = +s.substring(i - 2, i);
        if (one >= 1 && one <= 9) {
            dp[i] = dp[i - 1];
        }
        
        if (two >= 10 && two <= 26) {
            dp[i] += dp[i - 2];
        }
    }
    
    return dp[s.length];
}

function exec() {
    let result: number = numDecodings("126");

    let div: HTMLElement = document.createElement("div");
    div.innerText = result.toString();
    document.body.appendChild(div);
}

exec();
```

**Scala Solution:**

```scala
object Main extends App {
    def numDecodings(s: String): Int = {
        if (s.isEmpty) 0
        else {
            val dp = new Array[Int](s.length + 1)
            
            dp(0) = 1
            dp(1) = if (s(0) != '0') 1 else 0
            
            for (i <- 2 to s.length) {
                val one = s.substring(i - 1, i).toInt
                val two = s.substring(i - 2, i).toInt
                
                if (one >= 1 && i <= 9) {
                    dp(i) = dp(i - 1)
                }
                
                if (two >= 10 && two <= 26) {
                    dp(i) += dp(i - 2)
                }
            }
            
            dp(s.length)
        }
    }
    
    println(numDecodings(""))
    println(numDecodings("1"))
    println(numDecodings("12"))
    println(numDecodings("126"))
    println(numDecodings("136"))
}
```

### 165. Compare Version Numbers

Source: [https://leetcode.com/problems/compare-version-numbers/description/](https://leetcode.com/problems/compare-version-numbers/description/)

Compare two version numbers version1 and version2.
If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0.

You may assume that the version strings are non-empty and contain only digits and the . character.
The `.` character does not represent a decimal point and is used to separate number sequences.
For instance, `2.5` is not "two and a half" or "half way to version three", it is the fifth second-level revision of the second first-level revision.

Here is an example of version numbers ordering:

```
0.1 < 1.1 < 1.2 < 13.37
```

**Typescript Solution:**

```typescript
function compareTo(s1: number, s2: number): number {
    if (s1 < s2) return -1;
    else if (s1 > s2) return 1;
    else return 0;
}

function compareVersion(version1: string, version2: string): number {
    let levels1: string[] = version1.split('.');
    let levels2: string[] = version2.split('.');
    
    let length: number = Math.max(levels1.length, levels2.length);
    
    for (let i: number = 0; i < length; i++) {
        let v1: number = i < levels1.length ? +levels1[i] : 0;
        let v2: number = i < levels2.length ? +levels2[i] : 0;
        
        let compareResult: number = compareTo(v1, v2);
        if (compareResult !== 0) {
            return compareResult;
        }
    }
    
    return 0;
}

function exec() {
    let result: number = compareVersion("1.10", "1.2");

    let div: HTMLElement = document.createElement("div");
    div.innerText = result.toString();
    document.body.appendChild(div);
}

exec();
```

**Scala Solution:**

```scala
object Main extends App {
    def compareVersion(version1: String, version2: String): Int = {
        val levels1 = version1.split('.')
        val levels2 = version2.split('.')
        
        levels1.zipAll(levels2, "0", "0").foldLeft(0) { (result, pair) =>
            if (result != 0) result else pair._1.compareTo(pair._2)
        }
    }
    println(compareVersion("1.1.0.0", "1.1"))
}
```

### 457. Circular Array Loop

Source: [https://leetcode.com/problems/circular-array-loop/description/](https://leetcode.com/problems/circular-array-loop/description/)

You are given an array of positive and negative integers. If a number n at an index is positive, then move forward n steps. Conversely, if it's negative (-n), move backward n steps. Assume the first element of the array is forward next to the last element, and the last element is backward next to the first element. Determine if there is a loop in this array. A loop starts and ends at a particular index with more than 1 element along the loop. The loop must be ***"forward"*** or ***"backward"***.

**Example 1:** Given the array [2, -1, 1, 2, 2], there is a loop, from index 0 -> 2 -> 3 -> 0.

**Example 2:** Given the array [-1, 2], there is no loop.

**Note:** The given array is guaranteed to contain no element "0".

Can you do it in **O(n)** time complexity and **O(1)** space complexity?

**Hint:**

Slow/Fast Pointer Solution: Just think it as finding a loop in Linked-list, except that loops with only 1 element do not count. Use a slow and fast pointer, slow pointer moves 1 step a time while fast pointer moves 2 steps a time. If there is a loop (fast == slow), we return true, else if we meet element with different directions, then the search fail, we set all elements along the way to 0. Because 0 is fail for sure so when later search meet 0 we know the search will fail.