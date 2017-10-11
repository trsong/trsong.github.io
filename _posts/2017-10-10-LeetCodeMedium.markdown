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
        while (n != 0) {
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
