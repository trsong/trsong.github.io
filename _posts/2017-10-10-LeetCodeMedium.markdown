---
layout: post
title:  "LeetCode Questions - Medium"
date:   2017-10-10 22:36:32 -0700
categories: Scala TypeScript
---
* This will become a table of contents (this text will be scraped).
{:toc}

## LeetCode Questions - Medium

### Environment Setup

TypeScript Playground: [https://www.typescriptlang.org/play/](https://www.typescriptlang.org/play/)

Scala Playground: [https://scastie.scala-lang.org/](https://scastie.scala-lang.org/)

Scala/Js/CodeSnippet Playground2: [https://leetcode.com/playground/new](https://leetcode.com/playground/new)

Covert Tabs to Spaces in Code Snippets: [http://tabstospaces.com/](http://tabstospaces.com/)

### All Permutations and Combinations(PowerSet)

```scala
object Permutations {
  def permutations(s: String): List[String] = {
    def merge(ins: String, c: Char): Seq[String] =
      for (i <- 0 to ins.length) yield
        ins.substring(0, i) + c + ins.substring(i, ins.length)

    if (s.length() == 1)
      List(s)
    else
      permutations(s.substring(0, s.length - 1)).flatMap { p =>
        merge(p, s.charAt(s.length - 1))
      }
  }
}
```

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
    private var counter: BigInt = BigInt(1)
    private val stol = HashMap.empty[BigInt, String]
    
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

object Solution {
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
    private def reverse(sb: Buffer[Char], start: Int, end: Int): Unit = {
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
    private def reverseEachWordInPlace(sb: Buffer[Char]): Unit = {
        var (i, j) = (0, 0)
        while (i < sb.size) {
            while (i < j || i < sb.size && sb(i) == ' ') i += 1
            while (j < i || j < sb.size && sb(j) != ' ') j += 1
            reverse(sb, i, j - 1)
        }
    }
    
    // Remove the leading, tailing and multiple white spaces
    private def cleanSpaces(sb: Buffer[Char]): String = {
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
}

object Main extends App {
    println(Solution.reverseWords("    Hello    World !"))
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

You are given an array of positive and negative integers. If a number n at an index is positive, then move forward n steps. Conversely, if it's negative (-n), move backward n steps. Assume the first element of the array is forward next to the last element, and the last element is backward next to the first element. Determine if there is a loop in this array. A loop starts and ends at a particular index with more than 1 element along the loop. The loop must be ***"forward"*** or ***"backward"*** .

**Example 1:** Given the array [2, -1, 1, 2, 2], there is a loop, from index 0 -> 2 -> 3 -> 0.

**Example 2:** Given the array [-1, 2], there is no loop.

**Note:** The given array is guaranteed to contain no element "0".

Can you do it in **O(n)** time complexity and **O(1)** space complexity?

**Hint:**

Slow/Fast Pointer Solution: Just think it as finding a loop in Linked-list, except that loops with only 1 element do not count. Use a slow and fast pointer, slow pointer moves 1 step a time while fast pointer moves 2 steps a time. If there is a loop (fast == slow), we return true, else if we meet element with different directions, then the search fail, we set all elements along the way to 0. Because 0 is fail for sure so when later search meet 0 we know the search will fail.

**Typescript Solution:**

```typescript
function circularArrayLoop(nums: number[]): boolean {
    let n: number = nums.length;
    let arr: number[] = nums.slice();  // Make a copy
    const move = (i: number): number => {
        return i + nums[i] >= 0 ? (i + nums[i]) % n : (i + nums[i]) % n + n; 
    }
        
    for (let i: number = 0; i < n; i++) {
        if (arr[i] === 0) continue; 
        
        // slow / faster pointer
        let slow: number = i, fast: number = i;
        
        // Safe to move the slow pointer once and move the fast pointer twice
        while (arr[slow] * arr[i] > 0 && arr[fast] * arr[i] > 0 && arr[move(fast)] * arr[i] > 0) {
            // Make the move
            slow = move(slow);
            fast = move(move(fast));
            
            if (slow === fast) {  // Encounter each other!
                if (slow === move(slow)) break;  // 1 element loop is not a valid loop
                return true;
            }
        }
        
        // Optimization: Mark element along the path to be 0
        let direction: number = arr[i]; // arr[i] will be set to 0 in first iteration in loop
        slow = i;
        while (direction * arr[slow] > 0) {
            let nextPos: number = move(slow);
            arr[slow] = 0;
            slow = nextPos;
        }
    }
    return false;
}

function exec() {
    let result: boolean = circularArrayLoop([2, -1, 1, 2, 2]);

    let div: HTMLElement = document.createElement("div");
    div.innerText = result.toString();
    document.body.appendChild(div);
}

exec();
```

**Scala Solution:**

```scala
object Main extends App {
    def circularArrayLoop(nums: Array[Int]): Boolean = {
        val arr = nums.clone()
        val n = nums.size
        def move(i: Int) = if (i + arr(i) >= 0) (i + arr(i)) % n else (i + arr(i)) % n + n
        
        def slowFastPointerSearch(i: Int, slow: Int, fast: Int): Boolean = {
            if (!(arr(i) * arr(slow) > 0 && arr(i) * arr(fast) > 0 && arr(move(fast)) * arr(i) > 0)) false
            else {
                val nextSlow = move(slow)
                val nextFast = move(move(fast))
                if (nextSlow != nextFast) slowFastPointerSearch(i, nextSlow, nextFast) 
                else if (nextSlow == move(nextSlow)) false
                else true
            }
        }
                
        (0 until n) exists { i =>
            if (arr(i) == 0) false
            else if (slowFastPointerSearch(i, i, i)) true
            else {
                // Optimization: Mark element along the path to be 0
                val direction = arr(i)
                var j = i
                while (direction * arr(i) > 0) {
                    val nextPos = move(j)
                    arr(j) = 0
                    j = nextPos
                }
            
                false
            }
        }
    }
    
    println(circularArrayLoop(Array(2, -1, 1, 2, 2)))
}
```

### 468. Validate IP Address

Source: [https://leetcode.com/problems/validate-ip-address/description/](https://leetcode.com/problems/validate-ip-address/description/)

Write a function to check whether an input string is a valid IPv4 address or IPv6 address or neither.

**IPv4** addresses are canonically represented in dot-decimal notation, which consists of four decimal numbers, each ranging from 0 to 255, separated by dots ("."), e.g.,`172.16.254.1`;

Besides, leading zeros in the IPv4 is invalid. For example, the address `172.16.254.01` is invalid.

**IPv6** addresses are represented as eight groups of four hexadecimal digits, each group representing 16 bits. The groups are separated by colons (":"). For example, the address `2001:0db8:85a3:0000:0000:8a2e:0370:7334` is a valid one. Also, we could omit some leading zeros among four hexadecimal digits and some low-case characters in the address to upper-case ones, so `2001:db8:85a3:0:0:8A2E:0370:7334` is also a valid IPv6 address(Omit leading zeros and using upper cases).

However, we don't replace a consecutive group of zero value with a single empty group using two consecutive colons (::) to pursue simplicity. For example, `2001:0db8:85a3::8A2E:0370:7334` is an invalid IPv6 address.

Besides, extra leading zeros in the IPv6 is also invalid. For example, the address `02001:0db8:85a3:0000:0000:8a2e:0370:7334` is invalid.

**Note:** You may assume there is no extra space or special characters in the input string.

**Example 1:**

```
Input: "172.16.254.1"

Output: "IPv4"

Explanation: This is a valid IPv4 address, return "IPv4".
```

**Example 2:**

```
Input: "2001:0db8:85a3:0:0:8A2E:0370:7334"

Output: "IPv6"

Explanation: This is a valid IPv6 address, return "IPv6".
```

**Example 3:**

```
Input: "256.256.256.256"

Output: "Neither"

Explanation: This is neither a IPv4 address nor a IPv6 address.
```

**Typescript Solution:**

```typescript
class IPAddressProcessor {
	public static validIPAddress(ip: string): string {
		if (IPAddressProcessor.isValidIPv4(ip)) return "IPv4";
		else if (IPAddressProcessor.isValidIPv6(ip)) return "IPv6";
		else return "Neither";
	}
	
	public static isValidIPv4(ip: string): boolean {
		if (ip.length < 7) return false; // edge case: 0.0.0.0
		if (ip[0] === "." || ip[ip.length - 1] === ".") return false; // edge case: .0.10.0.
		let tokens: string[] = ip.split('.');
		if (tokens.length !== 4) return false;
		return tokens.every(IPAddressProcessor.isValidIPv4Token);
	}
	
	public static isValidIPv6(ip: string): boolean {
		if (ip.length < 15) return false; // edge case: 0:0:0:0:0:0:0:0
		if (ip[0] === ":" || ip[ip.length - 1] === ":") return false;
		let tokens: string[] = ip.split(":");
		if (tokens.length !== 8) return false;
		return tokens.every(IPAddressProcessor.isValidIPv6Token);
	}
	
	private static isValidIPv4Token(token: string): boolean {
		if (token.length === 0) return false;
		if (token[0] === "0" && token.length > 1) return false;
		
		try {
			let tokenValue: number = +token;
			if (tokenValue < 0 || tokenValue > 255) return false;
			if (tokenValue === 0 && token[0] !== "0") return false;
		} catch(e) {
			return false;
		}
		return true;
	}
	
	private static isValidIPv6Token(token: string): boolean {
		if (token.length > 4 || token.length === 0) return false;
		const isBase16 = (c: string) => {
			return (c >= "0" && c <= "9") // is digit
				|| (c >= "a" && c <= "f")  // is lower a to f
				|| (c >= "A" && c <= "F");  // is uppper a to f
		};
		
		return Array.from(token).every(isBase16);
	}
}

function exec() {
    let result: string = IPAddressProcessor.validIPAddress("2001:db8:85a3:0:0:8A2E:370:7334");

    let div: HTMLElement = document.createElement("div");
    div.innerText = result;
    document.body.appendChild(div);
}

exec();
```

**Scala Solution:**

```scala
object IPAddressProcessor {
	def validIPAddress(ip: String): String = {
		if (isValidIPv4(ip)) "IPv4"
		else if (isValidIPv6(ip)) "IPv6"
		else "Neither"
	}
	
	def isValidIPv4(ip: String): Boolean = {
		if (ip.length < 7 || ip.head == '.' || ip.last == '.') false
		else {
			val tokens = ip.split('.')
			if (tokens.size != 4) false
			else tokens.forall(isValidIPv4Token)
		}
	}
	
	def isValidIPv6(ip: String): Boolean = {
		if (ip.length < 15 || ip.head == ':' || ip.last == ':') false
		else {
			val tokens = ip.split(':')
			if (tokens.size != 8) false
			else tokens.forall(isValidIPv6Token)
		}
	}
	
	private def isValidIPv4Token(token: String): Boolean = {
		if (token.isEmpty || token.head == '0' && !token.isEmpty) false
		else {
			try {
				val tokenValue: Int = token.toInt
				if (tokenValue < 0 || tokenValue > 255 || tokenValue == 0 && token.length != 1) false
				else true
			} catch {
				case _ => false
			}
		}
	}

	private def isValidIPv6Token(token: String): Boolean = {
		def isBase16(c: Char) = c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
	
		if (token.length > 4 || token.isEmpty) false
		else token.forall(isBase16)
	}
}

object Main extends App {
	println(IPAddressProcessor.validIPAddress("172.16.254.1"))
	println(IPAddressProcessor.validIPAddress("2001:db8:85a3:0:0:8A2E:0370:7334"))
	println(IPAddressProcessor.validIPAddress("255.255.256.255"))
	println(IPAddressProcessor.validIPAddress("255.0.01.255"))
	println(IPAddressProcessor.validIPAddress("255. 0 .01.255"))
	println(IPAddressProcessor.validIPAddress("2001:db8:85a3:::8A2E:0370:7334"))
}
```

### 184. Department Highest Salary

Source: [https://leetcode.com/problems/department-highest-salary/discuss/](https://leetcode.com/problems/department-highest-salary/discuss/)

The Employee table holds all employees. Every employee has an Id, a salary, and there is also a column for the department Id.

```
+----+-------+--------+--------------+
| Id | Name  | Salary | DepartmentId |
+----+-------+--------+--------------+
| 1  | Joe   | 70000  | 1            |
| 2  | Henry | 80000  | 2            |
| 3  | Sam   | 60000  | 2            |
| 4  | Max   | 90000  | 1            |
+----+-------+--------+--------------+
```

The Department table holds all departments of the company.

```
+----+----------+
| Id | Name     |
+----+----------+
| 1  | IT       |
| 2  | Sales    |
+----+----------+
```

Write a SQL query to find employees who have the highest salary in each of the departments. For the above tables, Max has the highest salary in the IT department and Henry has the highest salary in the Sales department.

```
+------------+----------+--------+
| Department | Employee | Salary |
+------------+----------+--------+
| IT         | Max      | 90000  |
| Sales      | Henry    | 80000  |
+------------+----------+--------+
```

SQL Schema

```sql
Create table If Not Exists Employee (Id int, Name varchar(255), Salary int, DepartmentId int);
Create table If Not Exists Department (Id int, Name varchar(255));
Truncate table Employee;
insert into Employee (Id, Name, Salary, DepartmentId) values ('1', 'Joe', '70000', '1');
insert into Employee (Id, Name, Salary, DepartmentId) values ('2', 'Henry', '80000', '2');
insert into Employee (Id, Name, Salary, DepartmentId) values ('3', 'Sam', '60000', '2');
insert into Employee (Id, Name, Salary, DepartmentId) values ('4', 'Max', '90000', '1');
Truncate table Department;
insert into Department (Id, Name) values ('1', 'IT');
insert into Department (Id, Name) values ('2', 'Sales');
```

**Solution:**

```sql
SELECT dep.Name as Department, emp.Name as Employee, emp.Salary 
from Department dep, Employee emp 
where emp.DepartmentId=dep.Id 
and emp.Salary=(Select max(Salary) from Employee e2 where e2.DepartmentId=dep.Id)
```

### 177. Nth Highest Salary

Write a SQL query to get the nth highest salary from the **Employee** table.

```
+----+--------+
| Id | Salary |
+----+--------+
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |
+----+--------+
```

For example, given the above Employee table, the nth highest salary where n = 2 is **200**. If there is no nth highest salary, then the query should return **null**.

```
+------------------------+
| getNthHighestSalary(2) |
+------------------------+
| 200                    |
+------------------------+
```


**Solution:**

```sql
WITH Unique_Salary AS (SELECT DISTINCT Salary FROM Employee)
SELECT e1.Salary
FROM Unique_Salary e1
WHERE N - 1 = 
    (SELECT COUNT(*) 
     FROM Unique_Salary e2 
     WHERE e2.Salary > e1.Salary)      
LIMIT 1
```

Or use `Order By` with `Limit N - 1`

```sql
SELECT IFNULL((SELECT DISTINCT Salary FROM Employee ORDER BY Salary DESC LIMIT N - 1 ,1), NULL)
```

### 15. 3Sum

Source: [https://leetcode.com/problems/3sum/description/](https://leetcode.com/problems/3sum/description/)

Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

**Note:** The solution set must not contain duplicate triplets.

```
For example, given array S = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

**Hint:**

The idea is to sort an input array and then run through all indices of a possible first element of a triplet. For each possible first element we make a standard bi-directional 2Sum sweep of the remaining part of the array. Also we want to skip equal elements to avoid duplicates in the answer without making a set or smth like that.



**Typescript Solution:**

```typescript
class ThreeSumSolution {
    public static threeSum(nums: number[]): [number, number, number][] {
        let sorted: number[] = nums.sort();
        let result: [number, number, number][] = [];
        for (let i: number = 0; i < sorted.length - 2; i++) {
            if (i === 0 || i > 0 && sorted[i] !== sorted[i-1]) { // skip the term that has same value
                ThreeSumSolution.twoSum(sorted, i, result);
            }
        }
        return result;
    }
    
    private static twoSum(sorted: number[], i: number, output: [number, number, number][]): void {
        let lo: number = i + 1, hi: number = sorted.length - 1, sum: number = 0 - sorted[i];
        while (lo < hi) {
            if (sorted[lo] + sorted[hi] === sum) {
                output.push([sorted[i], sorted[lo], sorted[hi]]);
                while (lo < hi && sorted[lo] === sorted[lo + 1]) lo++;
                while (lo < hi && sorted[hi] === sorted[hi - 1]) hi--;
                lo++; hi--;
            } else if (sorted[lo] + sorted[hi] < sum) lo++;
            else hi--;
        }
    }
}

function exec() {
    let result: [number, number, number][] = ThreeSumSolution.threeSum([-1, 0, 1, 2, -1, -4]);

    let div: HTMLElement = document.createElement("div");
    div.innerText = result.toString();
    document.body.appendChild(div);
}

exec();
```

**Scala Solution:**

```scala
object ThreeSumSolution {
  def threeSum(nums: IndexedSeq[Int]): Seq[(Int, Int, Int)] = {
    val sorted = nums.sorted
    var result = List.empty[(Int, Int, Int)]
    for (i <- 0 until sorted.size - 2) {
      if (i == 0 || i > 0 && sorted(i) != sorted(i - 1)) {
        var lo = i + 1
        var hi = sorted.size - 1
        var sum = 0 - sorted(i)
        while (lo < hi) {
          if (sorted(lo) + sorted(hi) == sum) {
            result = (sorted(i), sorted(lo), sorted(hi)) :: result
            while (lo < hi && sorted(lo) == sorted(lo + 1)) lo += 1
            while (lo < hi && sorted(hi) == sorted(hi - 1)) hi -= 1
            lo += 1
            hi -= 1
          } else if (sorted(lo) + sorted(hi) < sum) {
            lo += 1
          } else {
            hi -= 1
          }
        }
      }
    }
    result
  }
}

object Main extends App {
  println(ThreeSumSolution.threeSum(Array(-1, 0, 1, 2, -1, -4)))
}
```

### 307. Range Sum Query - Mutable

Sourece: [https://leetcode.com/problems/range-sum-query-mutable/description/](https://leetcode.com/problems/range-sum-query-mutable/description/)

Given an integer array nums, find the sum of the elements between indices *i* and *j (i ≤ j)*, inclusive.

The *update(i, val)* function modifies *nums* by updating the element at index *i* to *val*.

**Example:**

```
Given nums = [1, 3, 5]

sumRange(0, 2) -> 9
update(1, 2)
sumRange(0, 2) -> 8
```

**Note:**

1. The array is only modifiable by the update function.
2. You may assume the number of calls to update and sumRange function is distributed evenly.

**Approach #1 Sqrt decomposition**

```scala
class NumArray(private val nums: Array[Int]) {
    val (block, blockSize) = initBlock()
    
    def update(i: Int, value: Int): Unit = {
        val blockIndex = i / blockSize
        block(blockIndex) += value - nums(i)
        nums(i) = value
    }
    
    def sumRange(i: Int, j: Int): Int = {
        var sum = 0
        var startBlock = i / blockSize
        var endBlock = j / blockSize
        if (startBlock == endBlock) { // within the same block
            for (k <- i to j) sum += nums(k)
        } else {
            for (k <- i to (startBlock + 1) * blockSize - 1) sum += nums(k)
            for (k <- (startBlock + 1) to (endBlock - 1)) sum += block(k)
            for (k <- (endBlock * blockSize) to j) sum += nums(k)
        }
        sum
    }
    
    private def initBlock(): (Array[Int], Int) = {
        val blockSize = Math.ceil(Math.sqrt(nums.size)).toInt
        val block = new Array[Int](blockSize)
        for (i <- 0 until nums.size) {
            block(i / blockSize) += nums(i)
        }
        (block, blockSize)
    }
}

object Main extends App {
    val obj = new NumArray(Array(1, 3, 5))
    println(obj.sumRange(0, 2)) // 9
    obj.update(1, 2)
    println(obj.sumRange(0, 2)) // 8
}
```

**Approach #2: Segment tree**

```scala
class NumArray(private val nums: Array[Int]) {
    private val tree = buildTree()

    def update(i: Int, value: Int): Unit = {
        // update the leaf node and roll value all the way up to root
        var pos = i + nums.size
        tree(pos) = value
        while (pos > 0) {
            var left = pos
            var right = pos
            
            if (pos % 2 == 0) right = pos + 1 // pos is left child
            else left = pos - 1 // pos is right child
            
            // parent is updated after child is updated
            tree(pos / 2) = tree(left) + tree(right)
            pos /= 2
        }
    }
    
    def sumRange(i: Int, j: Int): Int = {
        val n = nums.size
        var l = i + n
        var r = j + n
        var sum = 0
        
        while (l <= r) {
            //              1~7
            //           /       \  
            //          /         \
            //         1~3        4~7
            //        /   \      /   \
            //       1    2~3  4~5   6~7
            //           /  |  | |  /   \
            //          2   3  4 5 6     7
             
            // case1: suppose l is right child that represents 2~3, we sum value(2~3) then move l right to 4~5
            // case2: suppose l is left child do nothing
            if (l % 2 == 1) { // l bound is right child
                sum += tree(l)
                l += 1
            }
            
            // case3: suppose r is right child that represents 6~7, do nothing
            // case4: suppose r is left child that represents 4~5, we sum value(4~5) then move r left to 2~3
            if ((r % 2) == 0) { // r bound is left child
                sum += tree(r)
                r -= 1
            }
            
            // Mov l and r to parent and continue
            l /= 2
            r /= 2
        }
        sum
    }
    
    private def buildTree(): Array[Int] = {
        if (nums.isEmpty) Array.empty[Int]
        else {
            val n = nums.size
            val tree = new Array[Int](2 * n)
            
            for (i <- 0 until n) {
                tree(n + i) = nums(i)
            }
            
            for (i <- n - 1 until 0 by -1) {
                tree(i) = tree(i * 2) + tree(i * 2 + 1)
            }
            
            tree
        }
    }
}

object Main extends App {
    val obj = new NumArray(Array(1, 3, 5))
    println(obj.sumRange(0, 2)) // 9
    obj.update(1, 2)
    println(obj.sumRange(0, 2)) // 8
}
```

**Apparoach #3 BIT or Fenwick tree**

```java
public class NumArray {
	/**
	 * Binary Indexed Trees (BIT or Fenwick tree):
	 * https://www.topcoder.com/community/data-science/data-science-
	 * tutorials/binary-indexed-trees/
	 * 
	 * Example: given an array a[0]...a[7], we use a array BIT[9] to
	 * represent a tree, where index [2] is the parent of [1] and [3], [6]
	 * is the parent of [5] and [7], [4] is the parent of [2] and [6], and
	 * [8] is the parent of [4]. I.e.,
	 * 
	 * BIT[] as a binary tree:
	 *            ______________*
	 *            ______*
	 *            __*     __*
	 *            *   *   *   *
	 * indices: 0 1 2 3 4 5 6 7 8
	 * 
	 * BIT[i] = ([i] is a left child) ? the partial sum from its left most
	 * descendant to itself : the partial sum from its parent (exclusive) to
	 * itself. (check the range of "__").
	 * 
	 * Eg. BIT[1]=a[0], BIT[2]=a[1]+BIT[1]=a[1]+a[0], BIT[3]=a[2],
	 * BIT[4]=a[3]+BIT[3]+BIT[2]=a[3]+a[2]+a[1]+a[0],
	 * BIT[6]=a[5]+BIT[5]=a[5]+a[4],
	 * BIT[8]=a[7]+BIT[7]+BIT[6]+BIT[4]=a[7]+a[6]+...+a[0], ...
	 * 
	 * Thus, to update a[1]=BIT[2], we shall update BIT[2], BIT[4], BIT[8],
	 * i.e., for current [i], the next update [j] is j=i+(i&-i) //double the
	 * last 1-bit from [i].
	 * 
	 * Similarly, to get the partial sum up to a[6]=BIT[7], we shall get the
	 * sum of BIT[7], BIT[6], BIT[4], i.e., for current [i], the next
	 * summand [j] is j=i-(i&-i) // delete the last 1-bit from [i].
	 * 
	 * To obtain the original value of a[7] (corresponding to index [8] of
	 * BIT), we have to subtract BIT[7], BIT[6], BIT[4] from BIT[8], i.e.,
	 * starting from [idx-1], for current [i], the next subtrahend [j] is
	 * j=i-(i&-i), up to j==idx-(idx&-idx) exclusive. (However, a quicker
	 * way but using extra space is to store the original array.)
	 */

	int[] nums;
	int[] BIT;
	int n;

	public NumArray(int[] nums) {
		this.nums = nums;

		n = nums.length;
		BIT = new int[n + 1];
		for (int i = 0; i < n; i++)
			init(i, nums[i]);
	}

	public void init(int i, int val) {
		i++;
		while (i <= n) {
			BIT[i] += val;
			i += (i & -i);
		}
	}

	void update(int i, int val) {
		int diff = val - nums[i];
		nums[i] = val;
		init(i, diff);
	}

	public int getSum(int i) {
		int sum = 0;
		i++;
		while (i > 0) {
			sum += BIT[i];
			i -= (i & -i);
		}
		return sum;
	}

	public int sumRange(int i, int j) {
		return getSum(j) - getSum(i - 1);
	}
}

// Your NumArray object will be instantiated and called as such:
// NumArray numArray = new NumArray(nums);
// numArray.sumRange(0, 1);
// numArray.update(1, 10);
// numArray.sumRange(1, 2);
```

**Q & A:**

What are differences between *segment trees*, *interval trees*, *binary indexed trees* and *range trees* in terms of:

- Key idea/definition
- Applications
- Performance/order in higher dimensions/space consumption

**Answer:**

All these data structures are used for solving different problems:

- **Segment tree** stores intervals, and optimized for "*which of these intervals contains a given point*" queries. (It is a static structure; that is, it's a structure that cannot be modified once it's built, same as **interval tree**.)
- **Interval tree** stores intervals as well, but optimized for "w*hich of these intervals overlap with a given interval*" queries. It can also be used for point queries - similar to segment tree.
- **Range tree** stores points, and optimized for "*which points fall within a given interval*" queries.
- **Binary indexed tree** stores items-count per index, and optimized for "*how many items are there between index m and n*" queries.


**One Dimension**

`k` is the number of reported results

              | Segment       | Interval   | Range          | Indexed   |
--------------|--------------:|-----------:|---------------:|----------:|
Preprocessing |        n logn |     n logn |         n logn |    n logn |
Query         |        k+logn |     k+logn |         k+logn |      logn |
Space         |             n |          n |              n |         n |
Insert/Delete |          logn |       logn |           logn |      logn |

**Higher Dimensions**

`d > 1`

              | Segment       | Interval   | Range          | Indexed   |
--------------|--------------:|-----------:|---------------:|----------:|
Preprocessing |     n(logn)^d |     n logn |      n(logn)^d | n(logn)^d |
Query         |    k+(logn)^d | k+(logn)^d |     k+(logn)^d |  (logn)^d |
Space         | n(logn)^(d-1) |     n logn | n(logn)^(d-1)) | n(logn)^d |


### 54. Spiral Matrix

Source: [https://leetcode.com/problems/spiral-matrix/description/](https://leetcode.com/problems/spiral-matrix/description/)

Given a matrix of *m x n* elements (*m* rows, *n* columns), return all elements of the matrix in spiral order.

For example,
Given the following matrix:

```
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
```

You should return `[1,2,3,6,9,8,7,4,5]`.

**Hint:**

This is a very simple and easy to understand solution. I traverse right and increment rowBegin, then traverse down and decrement colEnd, then I traverse left and decrement rowEnd, and finally I traverse up and increment colBegin.

The only tricky part is that when I traverse left or up I have to check whether the row or col still exists to prevent duplicates. 

**Solution**

```java
public class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        
        List<Integer> res = new ArrayList<Integer>();
        
        if (matrix.length == 0) {
            return res;
        }
        
        int rowBegin = 0;
        int rowEnd = matrix.length-1;
        int colBegin = 0;
        int colEnd = matrix[0].length - 1;
        
        while (rowBegin <= rowEnd && colBegin <= colEnd) {
            // Traverse Right
            for (int j = colBegin; j <= colEnd; j ++) {
                res.add(matrix[rowBegin][j]);
            }
            rowBegin++;
            
            // Traverse Down
            for (int j = rowBegin; j <= rowEnd; j ++) {
                res.add(matrix[j][colEnd]);
            }
            colEnd--;
            
            if (rowBegin <= rowEnd) {
                // Traverse Left
                for (int j = colEnd; j >= colBegin; j --) {
                    res.add(matrix[rowEnd][j]);
                }
            }
            rowEnd--;
            
            if (colBegin <= colEnd) {
                // Traver Up
                for (int j = rowEnd; j >= rowBegin; j --) {
                    res.add(matrix[j][colBegin]);
                }
            }
            colBegin ++;
        }
        
        return res;
    }
}
```

### 40. Combination Sum II

Source: [https://leetcode.com/problems/combination-sum-ii/description/](https://leetcode.com/problems/combination-sum-ii/description/)

Given a collection of candidate numbers (**C**) and a target number (**T**), find all unique combinations in **C** where the candidate numbers sums to **T**.

Each number in **C** may only be used **once** in the combination.

**Note:**

* All numbers (including target) will be positive integers.
* The solution set must not contain duplicate combinations.
* 
For example, given candidate set `[10, 1, 2, 7, 6, 1, 5]` and target `8`, 
A solution set is: 

```
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
```

**Solution**

```java
 public List<List<Integer>> combinationSum2(int[] cand, int target) {
    Arrays.sort(cand);
    List<List<Integer>> res = new ArrayList<List<Integer>>();
    List<Integer> path = new ArrayList<Integer>();
    dfs_com(cand, 0, target, path, res);
    return res;
}
void dfs_com(int[] cand, int cur, int target, List<Integer> path, List<List<Integer>> res) {
    if (target == 0) {
        res.add(new ArrayList(path));
        return ;
    }
    if (target < 0) return;
    for (int i = cur; i < cand.length; i++){
        if (i > cur && cand[i] == cand[i-1]) continue;
        path.add(path.size(), cand[i]);
        dfs_com(cand, i+1, target - cand[i], path, res);
        path.remove(path.size()-1);
    }
}
```

### 114. Flatten Binary Tree to Linked List

Source: [https://leetcode.com/problems/flatten-binary-tree-to-linked-list/description/](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/description/)

Given a binary tree, flatten it to a linked list in-place.

For example,
Given

```
         1
        / \
       2   5
      / \   \
     3   4   6
```

The flattened tree should look like:

```
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

**Hints:**

If you notice carefully in the flattened tree, each node's right child points to the next node of a pre-order traversal.

**Solution1:**

```java
private TreeNode prev = null;

public void flatten(TreeNode root) {
    if (root == null)
        return;
    flatten(root.right);
    flatten(root.left);
    root.right = prev;
    root.left = null;
    prev = root;
}
```

**Solution2:**

```java
public class Solution {
    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        if (root.left == null && root.right == null) {
            return;
        }
        while (root != null) {
            if (root.left == null) {
                root = root.right;
                continue;
            }
            TreeNode left = root.left;
            while (left.right != null) {
                left = left.right;
            }
            left.right = root.right;
            root.right = root.left;
            root.left = null;
            root = root.right;
        }
    }
}
```
