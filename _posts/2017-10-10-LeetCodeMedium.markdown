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
import scala.collection.mutable.StringBuilder
import scala.collection.mutable.HashMap

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
			sb += dictionary.charAt((target % 62).toInt)
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
	while(str[index] === ' ' && index < str.length) index++;
	
	// 3. Handle signs
	if (str[index] === '+' || str[index] === '-') {
		sign = str[index] === '+' ? 1 : -1;
		index++;
	}
	
	// 4. Convert number and avoid overflow
	while (index < str.length) {
		let digit: number = str.charCodeAt(index) - "0".charCodeAt(0);
		if (digit < 0 || digit > 9) break;
		total = 10 * total + digit;
		index++;
		
		if (total > INT_MAX && sign > 0) return INT_MAX;
		else if (total > -INT_MIN && sign < 0) return INT_MIN;
	}
	return total * sign;
}

function exec() {
    let result: number = atoi("   -42");

    let div: HTMLElement = document.createElement("div");
    div.innerText = result.toString();
    document.body.appendChild(div);
}

exec();
```

**Scala Soluction:** 

```scala
def atoi(str: String): Int = {
	if (str.isEmpty) 0
	...
}
```