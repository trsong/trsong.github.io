---
layout: post
title:  "Scala for the Impatient"
date:   2017-09-24 22:36:32 -0700
categories: Scala
---
* This will become a table of contents (this text will be scraped).
{:toc}

### CH1: The Basics

* Declare multiple values or variables together:

```scala
val xmax, ymax = 100
var greeting, message: String = null
```

* Unlike Java, there's no distinction between primitive types and class types.

```scala
1.toString() // yields the string "1"
"Hello".intersect("World") // yields "lo"
```

* When call a Java method with arguments of type Object, need to convert any primitive types by hand

```scala
val str = MessageFormat.format("The answer to {0} is {1}", "everything", 42.asInstanceOf[AnyRef])
```

* Unlike Java, operators can be overridden

```scala
a + b // is the shorthand for a.+(b)
a method b // means a.method(b)

// scala has + - * / and bit-wise operator: & | ^ >> <<
// however, scala has neither ++ nor --

counter += 1
counter -= 1

// In Java, when deal w/ BigInt, one can only call x.multiply(x).multiply(x)
val x: BigInt = 1234567890
x * x * x
```

* Import inline

```scala
import scala.math._ // In Scala, the _ character is a “wildcard,” like * in Java
sqrt(2)
pow(2,4)
min(3,Pi)

import scala.util.Random
Random.nextInt
```

* Call a method without parameters

```scala
"Hello".distinct // Same as .distinct()
```

* The `apply` method

```scala
"Hello"(4) // yields 'o'

// In StringOps class you will find a method
def apply(n: Int): Char

"Hello".apply(4) // can simply use "Hello"(4)

BigInt("1234567890") // is a shortcut for BigInt.apply("1234567890")
```

### CH2: Control Structures and Functions

* An `if` expression has a value

```scala
val s = if (x > 0) 1 else -1 // same as x > 0 ? 1 : -1 in Java

val mixedType = if (x > 0) "Positive" else -1 // the return type is the common supertype of both branches, in this case is Any

val withoutElse = if (x > 0) 1 
// Is equivalent to 
val withElse = if (x > 0) 1 else () // () has type Unit
```

* Semicolon ending is optional in Scala

```scala
if(n > 0) { r = r * n; n -= 1}
```

* Block expression use the last stement value as return value

```scala
val distance = {val dx = x - x0; val dy = y - y0; sqrt(dx * dx + dy * dy)}
```

* Do NOT chaining assignments statements

```scala
x = y = 1 // NO

y = 1 // result in Unit

x = () // is the final result of x = y = 1
```

* Input and output

```scala
print("Answer: ")
println(42)  

// yields the same result as 

println("Answer: " + 42)

// readInt, readDouble, readByte, readShort, readLong, readFloat, readBoolean and readChar

val name = readLine("Your name: ")
print("Your age: ")
val age = readInt()
printf("Hello, %s! Next year, you will be %d.\n", name, age + 1)
```

* Loops

```scala 
while(n > 0) {
	r = r * n
	n -= 1
}

for (i <- 1 to n)  // Note: there's no var/val before 'i', and the type of 'i' is the element type of the collection
	r = r * n
	
// for(i <- expr)
val s = "Hello"
var sum = 0
for(i <- 0 until s.length)
	sum += s(i)
	
// No need to use index
var sum = 0 
for(ch <- "Hello") sum += ch
```

* There is no 'break' in a loop

```scala
import scala.util.control.Breaks._
breakable {
	for(...) {
		if(...) break; // Exit the breakable block
		...
	}
}

// Note above break is done by throwing and catching exception
// Should avoid use break as much as possible
```

* Advanced 'for' loops

```scala
// Multiple generators separated by semicolons
for(i <- 1 to 3; j <- 1 to 3) print((10 * i + j) + " ")
	// Prints 11 12 13 21 22 23 31 32 33

// Each generators can have a guard
for(i <- 1 to 3; j <- 1 to 3 if i != j) print((10 * i + j) + " ")
	// Prints 12 13 21 23 31 32
	
// Introduce a new var that can be used inside loop
for(i <- 1 to 3; from = 4 - i; j <- from to 3) print((10 * i + j) + " ")
	//                 ^ careful here, it's not <- 
	// Prints 13 22 23 31 32 33

// Note above can also be
for { i <- 1 to 3
	from = 4 - i
	j <- from to 3 }

// for comprehension
for(i <- 1 to 10) yield i % 3
	// yields Vector(1, 2, 0, 1, 2, 0, 1, 2, 0)

// The generated collection is compactiable w/ the first generator
for(c <- "Hello"; i <- 0 to 1) yield (c + i).toChar
	// yields "HIeflmlmop"
for(i <- 0 to 1; c <- "Hello") yields (c + i).toChar
	// yields Vector('H', 'e', 'l', 'l', 'o', 'I', 'f', 'm', 'm', 'p')
```

* Create a function without return types will make compiler complains

```scala
def fac(n: Int): Int = if(n <= 0) 1 else n * fac(n - 1)
```

* Create a function with default and named arguments

```scala
def decorate(str: String, left: String = "[", right = "]"): String = left + str + right
decorate("Hello") // gets "[Hello]"
decorate("Hello", "<<<") // gets "<<<Hello]"
decorate(left = "<<<", str = "Hello", right = ">>>") // gets "<<<Hello>>>"
decorate(str = "Hello", right = ">>>") // gets "[Hello>>>"
```

* Create a function with Variable Arguments

```scala
def sum(args: Int*): Int = {
	var result = 0
	for(arg <- args) result += arg
	result
}

sum(1, 4, 9, 16, 25)

sum(1 to 5: _*) // Use : _* to convert a Seq to an argument sequence

def recurSum(args: Int*): Int = {
	if(args.length == 0) 0 
	else args.head + recurSum(args.tail: _*)
}
```

* Lazy values

```scala
lazy val words = scala.io.Source.fromFile("/usr/share/dict/words").mkString
// Note, laziness is not cost-free, each time a lazy value is access, a method is called (in thread-safe manner) to check if it has been initialized
```

* Exceptions

```scala
// Exception has type Nothing, so the final type will be the same as the other branch
if(x > 0) {
	sqrt(x)
} else throw new IllegalArgumentException("x should not be negative")

// Catching excetpion is modled after pattern-matching syntax
try {
	process(new URL("http://notExists.com/whatever.gif"))
} catch {
	case _: MalformedURLException => println("Bad URL: " + url)
	case ex: IOException => ex.printStackTrace()
}

// try/finally allow clean up
var in = new URL("http://notExists.com/whatever.gif").openStream()
try {
	process(in)
} finally {
	in.close()
}

// Nested try/catch/finally
try {...} catch {...} finally {...}
// Is the same as
try { try {...} catch {...} } finally {...}
```

### CH3: Working with Arrays

* Use Array for fixed length, and ArrayBuffer if length can vary

```scala
val nums = new Array[Int](10) // fills w/ 0
val a = new Array[String](10) // filss w/ null
val s = Array("Hello", "World")
s(0) = "Goodbye"

// ArrayBuffer is Scala equivalent for Java ArrayList, C++ vector
import scala.collection.mutable.ArrayBuffer
val b = new ArrayBuffer[Int]()
b += 1
	// ArrayBuffer(1)
b += (1, 2, 3, 5)
	// Use += to add element(s)
	// ArrayBuffer(1, 1, 2, 3, 5)
b ++= Array(8, 13, 21)
	// Use ++= to add other collections
	// ArrayBuffer(1, 1, 2, 3, 5, 8, 13, 21)
b.trimEnd(5)
	// Remove the last 5 elements
	// ArrayBuffer(1, 1, 2)
	
// Note: Adding or removing element has "amortized constant time". However, insert/remove is not efficient, both of them require the rest member to be shifted

b.insert(2, 6)
	// ArrayBuffer(1, 1, 6, 2)
b.insert(2, 7, 8, 9)
	// ArrayBuffer(1, 1, 7, 8, 9, 6, 2)
b.remove(2)
	// ArrayBuffer(1, 1, 8, 9, 6, 2)
b.remove(2, 3)
	// ArrayBuffer(1, 1, 2)
	
// Use toArray an toBuffer to convert between Array and ArrayBuffer	
b.toArray()
	// Array(1, 1, 2)
Array(1, 1, 2).toBuffer
	// ArrayBuffer(1, 1, 2)
```

* Array and ArrayBuffer Traversing

```scala
for(i <- 0 until a.length)
	println(i + ": " + a(i))

for(i <- 0 until (a.length, 2))
	// Range(0, 2, 4, ...)
	
for(i <- (0 until a.length).reverse)
	// Range(..., 2, 1, 0)
	
for(elem <- a)
	println(elem)
```

* Transforming Arrays

```scala
	val a = Array(2, 3, 5, 7, 11)
	val result = for(elem <- a) yield 2 * elem
		// result is Array(4, 6, 10, 14, 22)
	
	for(elem <- a if elem % 2 == 0) yield 2 * elem
	a.filter(_ % 2 == 0).map(2 * _)
```

Example: Given an array of Int, remove all but the first negative numbers

```scala
val indices = for(i <- 0 until a.length if a(i) <  1) yield i
for(j <- (1 until indices.length).reverse) a.remove(indices(j))
```

* Common Algorithms

```scala
Array(1, 7, 2, 9).sum // 19
ArrayBuffer("Mary", "had", "a", "little", "lamb") // "little"

val b = ArrayBuffer(1, 7, 2, 9)
val bSorted = b.sorted
val bSortedReverse = b.sorted(Ordering.Int.reverse)
	// b unchanged, bSorted is ArrayBuffer(1, 2, 7, 9)
	
val a = Array(1, 7, 2, 9)
scala.util.Sorting.quickSort(a)
	// a is now Array(1, 2, 7, 9)

a.mkString(" and ")
	// "1 and 2 and 7 and 9"
a.mkString("<", ",", ">")
	// "<1, 2, 7, 9>"

a.toString
	// "[I@73e5"
	// The useless toString from Java
b.toString
	// "ArrayBuffer(1, 7, 2, 9)"
	
Array(1, 7, 2, 9).count(_ % 2 == 0) // 1

// Behind screen Scala convert Array Class to ArrayOps before any operations is applied
```

* Multidimensional Arrays

```scala
val matrix = Array.ofDim[Double](3, 4) // matrix is 3 by 4 matrix filled w/ 0
matrix(0)(1) = 42

// Now build ragged arrays, with varying row lengths
val triangle = new Array[Array[Int]](10)
for(1 <- 0 until triangle.length)
	triangle(i) = new Array[Int](i + 1)
```

### CH4: Maps and Tuples

* Construct a map 

```scala
val scores = Map("Alice" -> 10, "Bob" -> 3, "Cindy" -> 8)
```

* Construct a mutable map

```scala
val scores = scala.collection.mutable.Map("Alice" -> 10, "Bob" -> 3, "Cindy" -> 8)

// Start out w/ a blank map
val scores = new scala.collection.mutable.HashMap[String, Int]
```

* `key -> value` will make a pair

```scala
"Alice" -> 10 // returns ("Alice", 10)
```

* `map.getOrElse(key, defaultVal)`

```scala
val bobsScore = if(scores.contains("Bob")) scores("Bob") else 0

// is equivalent to 

val bobsScore = scores.getOrElse("Bob", 0)
```

* Update map values

```scala
scores("Bob") = 10  // If scores is mutable
scores += ("Bob" -> 10, "Fred" -> 7) // Add multiple value to map
scores -= "Alice"
val newScores = scores + ("Bob" -> 10, "Fred" -> 7) // new map w/ update
val newScores2 = scores - "Alice"
```

* Iterating over Maps

```
for((k, v) <- map) yield (v, k) // Reverse a map will override v, since v might not be unique

scores.keySet
	// Set("Bob", "Cindy", "Fred", "Alice")

for(v <- scores.values) println(v)
```

* Sorted Map

```scala
val scores = scala.collection.immutable.SortedMap("Alice" -> 10, "Fred" -> 7, "Bob" -> 3, "Cindy" -> 8)

val months = scala.collection.mutable.LinkedHashMap("January" -> 1, "Feburary" -> 2, "March" -> 3, "April" -> 4, "May" -> 5, ...)
```

* Tuples

```scala
val t = (1, 3.14, "Fred")
	// has type (Int, Double, java.lang.String)
val second = t._2
val (first, second, third) = t
val (first, second, _) = t

"New York".partition(_.isUpper)
	// yields ("NY", "new ork")
```

* Zipping

```scala
val symbols = Array("<", "-", ">")
val counts = Array(2, 10, 2)
val pairs = symbols.zip(counts)
	// yields Array(("<", 2), ("-", 10), (">", 2))
for((s, n) <- pairs) print(s * n)
	// displays <<---------->>

// zip and toMap
keys.zip(values).toMap
```

### CH5: Classes

* Getter and Setter

```scala
class Person {
	private var privateAge = 0
	def age = privateAge
	def age_=(newValue: Int): Unit = {
		if(newValue > privateAge) privateAge = newValue
	}
}

val fred = new Person
fred.age = 30
fred.age = 21
println(fred.age) // 30

// Note: 
//		1) if field is private, then getter and setter is private
//		2) if field is a value, only getter is generated
//		3) declear as 'private[this]' and no getter and setter is generated

class Message {
	val timeStamp = new java.util.Date // read-only property w/ only getter
	...
}
```

* Object-private fields

```scala
class Counter {
	private var value = 0
	def increment(): Unit = {
		value += 1
	}
	
	def isLess(other: Counter): Boolean = {
	 value < other.value
		// Can access the private field of other object
	}
}

// declear value as object-private
private[this] var value = 0
	// Accessing someObject.value is not allowed
```

* Auxiliary Constructors

```scala
// Class can only have 1 primary constructor, but can have as many auxiliary constructors
class Person {
	private var name = ""
	private var age = 0
	
	def this(name: String) { // An auxiliary constructor
		this() // call primary constructor
		this.name = name
	}
	
	def this(name: String, age: Int) { // Another auxiliary constructor
		this(name) // Calls previous constructor
		this.age = age
	}
}

val p1 = new Person	// Calls primary Constructor
val p2 = new Person("Fred") // Calls first auxiliary constructor 
val p3 = new Person("Fred", 42) // Calls second auxiliary constructor
```

* The Primary Constructor

```scala
class Person(val name: String, val age: Int) {
	...
}

class PersonWithDefault(val name: String = "", val age: Int = 0) {
	...
}

class Person(val name: String, private var age: Int)

// name: String    generates object-private field, or no field if no method uses name
// private val/var name: String  private field, private getter/setter
// val/var name: String  generates private field, public getter/setter

class Person private(val id: Int) { ... } 
	// will make the primary constructor private
```

* Nested Classes

```scala
class Network {
	class Member(val name: String) {
		val contacts = new ArrayBuffer[Member]
	}
	
	private val members = new ArrayBuffer[Member]
	
	def join(name: String): Member = {
		val m = new Member(name)
		members += m
		m
	}
}

val chatter = new Network
val myFace = new Network

// Different from Java, inner class belongs to outer class
// new chatter.Member to init inner class

val fred = chatter.join("Fred")
val wilma = chatter.join("Wilma")
fred.contacts += wilma // OK

val barney = myFace.join("Barney")
fred.contacts += barney
	// No, cannot add myFace.Member to a buffer of chatter.Member elements
	
// If we want Member to be shared between two networks, there are two ways to achieve that

// Method 1: Companion Object 
object Network {
	class Member(val name: String) {
		val contacts = new ArrayBuffer[Member]
	}
}

class Network {
	private val members = new ArrayBuffer[Network.Member]
	...
}

// Method 2: Type Projection
class Network {
	class Member(val name: String) {
		val contacts = new ArrayBuffer[Network#Member]
			// means a Member of any Network
	}
	...
}
```

* Access outer class's field 

```scala
class Network(val name: String) { outer => 
	class Member(val name: String) {
		...
		def description = name + " inside " + outer.name
			// outer refer to Network.this
	}
}
```

### CH6: Objects

* Singletons

```scala
object Accounts {
	private var lastNumber = 0
	def newUniqueNumber(): Int = {
		lastNumber += 1
		lastNumber
	}
}
// The constructor of an object is executed when the object is first used
Accounts.newUniqueNumber()

// object can be used:
//		* as a home for utility function or constants
//		* as a single immutable instance being shared
// 		* the singleton design pattern
```