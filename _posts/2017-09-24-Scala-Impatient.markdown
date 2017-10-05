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
if (n > 0) { r = r * n; n -= 1}
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
while (n > 0) {
	r = r * n
	n -= 1
}

for (i <- 1 to n)  // Note: there's no var/val before 'i', and the type of 'i' is the element type of the collection
	r = r * n
	
// for (i <- expr)
val s = "Hello"
var sum = 0
for (i <- 0 until s.length)
	sum += s(i)
	
// No need to use index
var sum = 0 
for (ch <- "Hello") sum += ch
```

* There is no 'break' in a loop

```scala
import scala.util.control.Breaks._
breakable {
	for (...) {
		if (...) break; // Exit the breakable block
		...
	}
}

// Note above break is done by throwing and catching exception
// Should avoid use break as much as possible
```

* Advanced 'for' loops

```scala
// Multiple generators separated by semicolons
for (i <- 1 to 3; j <- 1 to 3) print((10 * i + j) + " ")
	// Prints 11 12 13 21 22 23 31 32 33

// Each generators can have a guard
for (i <- 1 to 3; j <- 1 to 3 if i != j) print((10 * i + j) + " ")
	// Prints 12 13 21 23 31 32
	
// Introduce a new var that can be used inside loop
for (i <- 1 to 3; from = 4 - i; j <- from to 3) print((10 * i + j) + " ")
	//                 ^ careful here, it's not <- 
	// Prints 13 22 23 31 32 33

// Note above can also be
for { i <- 1 to 3
	from = 4 - i
	j <- from to 3 }

// for comprehension
for (i <- 1 to 10) yield i % 3
	// yields Vector(1, 2, 0, 1, 2, 0, 1, 2, 0)

// The generated collection is compactiable w/ the first generator
for (c <- "Hello"; i <- 0 to 1) yield (c + i).toChar
	// yields "HIeflmlmop"
for (i <- 0 to 1; c <- "Hello") yields (c + i).toChar
	// yields Vector('H', 'e', 'l', 'l', 'o', 'I', 'f', 'm', 'm', 'p')
```

* Create a function without return types will make compiler complains

```scala
def fac(n: Int): Int = if (n <= 0) 1 else n * fac(n - 1)
```

* Create a function with default and named arguments

```scala
def decorate(str: String, left: String = "[", right = "]"): String = left + str + right
decorate("Hello") // gets "[Hello]"
decorate("Hello", "<<<") // gets "<<<Hello]"
decorate(left = "<<<", str = "Hello", right = ">>>") // gets "<<<Hello>>>"
decorate(str = "Hif (ello", right = ">>>") // gets "[Hello>>>"
```

* Create a function with Variable Arguments

```scala
def sum(args: Int*): Int = {
	var result = 0
	for (arg <- args) result += arg
	result
}

sum(1, 4, 9, 16, 25)

sum(1 to 5: _*) // Use : _* to convert a Seq to an argument sequence

def recurSum(args: Int*): Int = {
	if (args.length == 0) 0 
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
if (x > 0) {
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
for (i <- 0 until a.length)
	println(i + ": " + a(i))

for (i <- 0 until (a.length, 2))
	// Range(0, 2, 4, ...)
	
for (i <- (0 until a.length).reverse)
	// Range(..., 2, 1, 0)
	
for (elem <- a)
	println(elem)
```

* Transforming Arrays

```scala
	val a = Array(2, 3, 5, 7, 11)
	val result = for (elem <- a) yield 2 * elem
		// result is Array(4, 6, 10, 14, 22)
	
	for (elem <- a if elem % 2 == 0) yield 2 * elem
	a.filter(_ % 2 == 0).map(2 * _)
```

Example: Given an array of Int, remove all but the first negative numbers

```scala
val indices = for (i <- 0 until a.length if a(i) <  1) yield i
for (j <- (1 until indices.length).reverse) a.remove(indices(j))
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
for (1 <- 0 until triangle.length)
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
val bobsScore = if (scores.contains("Bob")) scores("Bob") else 0

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
for ((k, v) <- map) yield (v, k) // Reverse a map will override v, since v might not be unique

scores.keySet
	// Set("Bob", "Cindy", "Fred", "Alice")

for (v <- scores.values) println(v)
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
for ((s, n) <- pairs) print(s * n)
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
		if (newValue > privateAge) privateAge = newValue
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

* Companion Objects

```scala
class Account {
	val id = Account.newUniqueNumber()
	private var balance = 0.0
	def deposit(amount: Double): Unit = {
		balance += amount
	}
}

object Account {  // The companion object
	private var lastNumber = 0
	private def newUniqueNumber(): Int = {
		lastNumber += 1
		lastNumber
	}
}

// Note class and its companion object:
//		1. must share the same name
//		2. can access each others' private features
//		3. must be located in the same source file
```

* Objects extending a class or trait

```scala
abstract class UndoableAction(val description: String) {
	def undo(): Unit
	def redo(): Unit
}

object DoNothingAction extends UndoableAction("Do nothing") {
	override undo(): Unit = {}
	override redo(): Unit = {}
}

val actions = Map("open" -> DoNothingAction, "save" -> DoNothingAction, ... )
	// Open and save action not yet implemented
```

* The `apply` method

```scala
class Account private(val id: Int, initialBalance Double) {
	private var balance = initialBalance
	...
}

object Account {
	def apply(initialBalance: Double): Account = {
		new Account(newUniqueNumber(), initialBalance)
	}
	...
}

val acc = Account(1000.0)
```

* Application Objects

```
object Hello {
	def main(args: Array[String]): Unit = {
		println("Hello World!")
	}
}
```

* Enumerations

```scala
object TrafficLightColor extends Enumeration {
	val Red, Yellow, Green = Value
}

// is equivalent to
object TrafficLightColor extends Enumeration {
	val Red = Value
	val Yellow = Value
	val Green = Value
}

// Alternatively
object TrafficLightColor extends Enumeration {
	val Red = Value(0, "Stop")
	val Yellow = Value(10) // Name "Yellow"
	val Green = Value // ID 11
}

// Access the enum
val red = TrafficeLightColor.Red 

// If we want to use Red, Yellow, Green directly
import TrafficeLightColor._

// Note the return type of Enum is 
val red: TrafficLightColor.Value = TrafficeLightColor.Red

// Some people recommend to add a type alias
object TrafficLightColor extends Enumeration {
	type TrafficLightColor = Value
	val Red, Yellow, Green = Value
}

import TrafficLightColor._
def doWhat(color: TrafficLightColor): string {
	if (color == Red) {
		"stop"
	} else if (color == Yellow) {
		"hurry up"
	} else {
		"go"
	}
}

for (c <- TrafficLightColor.values) println(c.id + ": " + c)

TrafficLightColor(0) // Calls Enumeration.apply
TrafficLightColor.withname("Red")
```

### CH7: Packages

* `java.lang`, `scala`, `Predef` are always imported implicitly

```
// Every Scala program implicitly starts with
import java.lang._
import scala._
import Predef._
```

* Packages 

```scala
package com {
	package horstmann {
		package impatient {
			class Employee
			...
		}
	}
}

// class name Employee can be accessed as com.horstmann.impatient.Employee
```

* Contribute to more than one package in a single file

```scala
package com {
	package horstmann {
		package impatient {
			class Employee
			...
		}
	}
}

package org {
	package bigjava {
		class Counter
		...
	}
}
```

* Scope rules - Everything in the parent package is in scope

```scala
package com {
	package horstmann {
		object Utils {
			def percentOf(value: Double, rate: Double): Double = {
				value * rate / 100
			}
			...
		}
		
		package impatient {
			class Employee {
				...
				def giveRaise(rate: scala.Double): Unit = {
					salary += Utils.percentOf(salary, rate)
						// You could also use com.horstmann.Utils.percenOf, since com is also in scope
				}
			}
		}
	}
}

// however, consider the following 
package com {
	package horstmann {
		package impatient {
			class Manager {
				val subordinates = new collection.mutable.ArrayBuffer[Employee] // Note, scala is always imported
			}
		}
	}
}

// and in a different file
package com {
	package hortmann {
		package collection {
			...
		}
	}
}

// one solution
val subordinates = new _root_.scala.collection.mutable.ArrayBuffer[Employee]

// the other solution: chained package clauses
```

* Chained Package Clauses

```scala
package com.horstmann.impatient {
	// Members of com and com.horstmann are NOT visible here
	package people {
		class Person
		...
	}
	
	// Now com.horstmann.collection will not be accessed as collection
}
```

* Top-of-file notation

```scala
// START-OF-FILE
package com.horstmann.impatient
package people

class Person
...
// End-OF-FILE

// is equivalent to 
package com.horstmann.impatient {
	package people {
		class Person
		...
		
		// Until the end of the file
	}
}

// Notice the whole file belong to package com.horstmann.impatient.people 
// however, package com.horstmann.impatient is open up to allow refer to its content
```

* Package Objects

```scala
// A package can contain classes, objects, and traits, but not functions or varaibles, that's limitation of JVM
// However, it make sense to add utility functions or constants to package than to some Utils Object
// That's how package object come into play

// Note every package has one package object. 

package com.horstmann.impatient

package object people {
	val defaultName = "John Q. Public"
}

package people {
	class Person {
		var name = defaultName // A constant from the package
			// defaultName is in scope
			// outside the package, it is com.horstmann.impatient.people.defaultName
	}
}
```

* Package Visibility

```scala
package com.horstmann.impatient.people

class Person {
	private[people] def description = "A person with name" + name
}

// You can also exend the visibility to an enclosing package
class Person {
	private[impatient] def description = "A person with name" + name
}
```

* Imports

```scala
import java.awt.Color	// Now you can write Color instead of java.awt.Color

import java.awt._ 

def handler(evt: event.ActionEvent): Unit { // java.awt.event.ActionEvent
	...	
}
```

* Imports can be anywhere

```scala
class Manager {
	import scala.collection.mutable._
	val subordinates = new ArrayBuffer[Employee]
	...
}
```

* Renaming and hiding members

```scala
import java.awt.{Color, Font}
import java.util.{HashMap => JavaHashMap}
import scala.collection.mutable._

// JavaHashMap is java.util.HashMap, plain HashMap is scala.collection.mutable.HashMap

import java.util.{HashMap => _, _}
import scala.collection.mutable._
	// now HashMap unambiguously refer to scala.collection.mutable.HashMap 
	// since we import everything in java.util while hiding java.util.HashMap
```

### CH8: Inheritance

* Overriding methods

```scala
public class Person {
	...
	override def toString = getClass.getName + "[name=" + name + "]"
}

public class Employee extends Person {
	...
	override def toString = super.toString + "[salary=" + salary + "]"
}
```

* Type checks and casts

```scala
if (p.isInstanceOf[Employee]) {
	val s = p.asInstanceOf[Employee] // s has type Employee
}

p match {
	case s: Employee => ... // Process s as Employee
	case _ => ... // p wasn't an Employee
}
```

* Superclass Construction

```scala
class Employee(name: String, age: Int, val salary: Double) extends Person(name, age)
```

* Overriding fields

```scala
class Person(val name: String) {
	override def toString: String = getClass.getName + "[name=" + name + "]"
}

class SecretAgent(codename: String) extends Person(codename) {
	override val name = "Secret"
	override val toString = "Secret"
}

// Note:
//		1. A `def` can only override `def`
//		2. A `val` can only override `val` or a parameterless `def`
//		3. A `var` can only override abstract var
//		4. according to 3. if super class use a `var` then all subclass are stuck with it. So avoid using `var`
```

* Anonymous subclasses

```scala
val alien = new Person("Fred") {
	def greeting = "Greetings, Earthling! My name is Fred."
}

def meet(p: Person{ def greeting: String }): Unit = {
	println(p.name + " says: " + p.greeting)
}
```

* Abstract class

```scala
abstract class Person(val name: String) {
	def id: Int // No method body - this is an abstract method
}

class Employee(name: String) extends Person(name) {
	def id = name.hashCode // override an abstract method do not require override keyword
}
```

* Abstract Fields

```scala
abstract class Person {
	val id: Int	// an abstract field w/ an abstract getter method
	var name: String // an abstract field w/ an abstract getter and setter method
}

class Employee(val id: Int) extends Person { // subclass has concert id property
var name = "" // and concrete name property
}

// Note: no override is required to override an abstract field

val fred = new Person {
	val id = 1792
	var name = "Fred"
}
```

* Construction order and early definitions

```scala
class Creature {
	val range: Int = 10
	val env: Array[Int] = new Array[Int](range)
}

class Ant extends Creature {
	override val range = 2
}

val a = new Ant
a.range 	// 2
a.env 		// Array()

// why a.env is empty array?
// 1. in order to init Ant, init Creature first
// 2. Creature set range to 10
// 3. in order to set env, we call range() getter
// 4. at compile time, range() getter is override to be the one defined in Ant (range is yet uninitalized)
// 5. range() returns 0, as its default val for all uninitialized Int field
// 6. env is set to array of lenth 0
// 7. Ant constuctor begins, set range to 2

// 4 ways to solve above issue:
// 		1) declare the val as final
//		2) declare the val as lazy val
// 		3) declare the val as def
//		4) use early definition syntax

// Early definiton syntax

class Bug extends {
	override val range = 3
} with Creature
```

* Object equality

```scala
	// eq method in AnyRef checks tow references refer to same obejct
	// equals in AnyRef should be used to check its content
	
class Item(val description: String, val price: Double) {
	final override def equals(other: Any): boolean = {
		val that = other.asInstanceOf[Item]
		if (that == null) false
		else description == that.description && price == that.price
	}
}	
```

### CH9: Files and Regular Expressions

* Read lines

```scala
import scala.io.Source
val source = Source.fromFile("myfile.txt", "UTF-8")
val lineIterator = source.getLines
for (l <- lineIterator) println(l)

val lines = source.getLines.toArray
val contents = source.mkString
```

* Read tokens 

```scala
val tokens = source.mkString.split("\\s+")
	// split the string by white space
``` 

* Read source from URL or other source

```scala
val source1 = Source.fromURL("http://horstmann.com", "UTF-8")
val source2 = Source.fromString("Hello, world!")
val source3 = Source.stdin
```

* Read binary files

```scala
// scala lack of native way to read binary, need to rely on Java class
val file = new File(filename)
val in = new FileInputStream(file)
val bytes = new Array[Byte](file.length.toInt)
in.read(bypes)
in.close()
```

* Write into files

```scala
// Same as above, scala lack of support to write to a file
val out = new PrintWrite("numbers.txt")
for (i <- 1 to 100) out.println(i)
out.close()
```

* Object serialization

```
class Person extends Serializable

val fred = new Person(...)
import java.io._
val out = new ObjectOutputStream(new FileOutputStream("tmp/test.obj"))
out.writeObject(fred)
out.close()

val in = new ObjectInputStream(new FileInputSteam("tmp/test.obj"))
val savedFred = in.readObject().asInstanceOf[Person]
```

* Process control

```scala
import sys.process._
// ! will implicitly convert a string into a processBuilder
val retCode: Int = "ls -al" !
	// will execute the command in bash and print out the result
	// retCode is 0 if succeed otherwise display a non-zero value
	
val result: String = "ls -al" !!
	// will not display anything, but return the display string

"ls -al" #| "grep sec" !
	// use #| as a pipe

"ls -al" #> new File("output.txt") !
	// create and replace output file
"ls -al" #>> new File("output.txt") !
	// append or create output file 
	
"grep sec" #< new File("output.txt") !
"grep google" #< new URL("google.ca") !

val (p, q) = ("pwd", "whoami")
p #&& q !
	//	if p succeed then execute q
p #|| q !
	//  if p not succeed then execute q
```

* Regular Expression

```scala
val numPattern = "[0-9]+".r
val wsnumwsPattern = """\s+[0-9]+\s+""".r
	// If expression contains too many "\"'s
	
for (matchString <- numPattern.findAllIn("99 bottles, 98 bottles"))
val maches = numPattern.findAllIn("99 bottles, 98 bottles").toArray
	// Array(99, 98)
val m1 = wsnumwsPattern.findFirstIn("99 bottles, 98 bottles")	// Some(" 98 ")

numPattern.findPrefixOf("99 bottles, 98 bottles")
	// Some(99)
wsnumwsPattern.findPrefixOf("99 bottles, 98 bottles")
	// None
	
numPattern.replaceFirstIn("99 bottles, 98 bottles", "XX")
	// "XX bottles, 98 bottles"
"numPattern.replaceAllIn("99 bottles, 98 bottles", "XX")
	// "XX bottles, XX bottles"
```

* Regular Expression Group

```scala
val numitemPattern = "([0-9]+) ([a-z]+)".r
val numitemPattern(num, item) = "99 bottles"
	// num is 99 and item is bottles
for (numitemPattern(num, item) <- numitemPattern.findAll("99 bottles, 98 bottles"))
	println(num + item)
```

### CH10: Trait

* Trait use as an interface

```scala
trait Logger {
	def log(msg: String): Unit // abstract method
}

// unlike java interface, scala trait is more like a class
class ConsoleLogger extends Logger { // use extends instead of implements
	def log(msg: String): Unit = {	// without override
		println(msg)
	}
}

// extends more traits
class ConsoleLogger extends Logger with Clonable with Serializable
	// note all Java interface can be used as a trait in scala
```

* Trait with concrete implementations

```scala
trait ConsoleLogger {
	def log(msg: String): Unit = {
		println(msg)
	}
}

// allow implementation details to be mix-in
class SavingAccount extends Account with ConsoleLogger {
	def withdraw(amount: Double): Unit = {
		if (amount > balance) log("Insufficient funds")
		else balance -= amount
	}
}
```

* Multiple traits mixed-in

```scala
trait TimestampLogger extends Logged {
	override def log(msg: String): Unit = {
		super.log(new java.util.Date() + " " + msg)
	}
}

trait ShortLogger extends Logged {
	val maxLength = 15
	override def log(msg: String): Unit = {
		super.log(
			if (msg.length <= maxLength) msg 
			else msg.substring(0, maxLength - 3) + "..." 
		)
	}
}

val acct1 = new SavingsAccount with ConsoleLogger with TimestampLogger with ShortLogger
	// <- super direction
val acct2 = new SavingsAccount with ConsoleLogger with ShortLogger with TimestampLogger
	// <- super direction

acct1.log() // Sun Feb 06 17:45:45 ...
	// ShortLogger's super is TimeStampLogger
	// as a result, it cuts "" string and add time stamp
	
acct2.log() // Sun Feb 06 1...
	// TimeStampLogger's super is ShortLogger
	// as a result, it prepends time stamp and then truncate it
```

* override methods

```scala
trait Logger {
	def log(msg: String): Unit	 // abstract method
}

trait TimestampLogger extends Logger {
	abstract override def log(msg: String): Unit = { // override abstract method
		super.log(new java.util.Date() + " " + msg)
		// ^ we have yet not determined which is the final implementation, thus have to make it abstract
	}
}
``` 

* trait fields

```scala
trait ShortLogger extends Logged {
	val maxLength = 15
}

class Account {
	var balance = 0.0
}

class SavingAccount extends Account with ConsoleLogger with ShortLogger {
	var interest = 0.0
	def withdraw(amount: Double): Unit = {
		if (amount > balance) log("Insufficient funds")
		else ...
	}
	
	// Note maxLength is not inherited, it's added as SavingAccount's own field
}
```

* abstract trait fields

```scala
trait ShortLogger extends Logged {
	val maxLength: Int // abstract field
	override def log(msg: String): Unit = {
		super.log(
			if (msg.length <= maxLength) msg
			else msg.substring
		)
	}
}

class SavingAccount extends Account with ConsoleLogger with ShortLogger {
	val maxLength = 20 // No need to override
}
```

* trait constructor execution order

```scala
trait FileLogger extends Logger {
	val out = new PrintWriter("app.log") // part of the constructor
	out.println("# " + new Date().toString) // alos part of the constructor
	def log(msg: String): Unit = { 
		out.println(msg)
		out.flush()
	}
}

class SavingsAccount extends Account with FileLogger with ShortLogger

// Constructor Execution Order:
// 1. Account // Super class
// 2. Logger // Super trait of first trait
// 3. FileLogger // First trait
// 4. ShortLogger // Second trait. Notice its super trait Logger is called already
// 5. SavingsAccount
```

* trait cannot have constructor param 

```scala
trait FileLogger extends Logger {
	val filename: String
	val out = new PrintStream(filename)
	def log(msg: String): Unit = {
		out.println(msg)
		out.flush()
	}
}

// val acct = new Savings with FileLoger("myapp.log")  <= Wrong!

val acct = new Savings with FileLogger {
	val filename = "myapp.log" //not work because the construct order
		// val out will be evaluated too early before filename
}

// One way to solve above problem
val acct = new {
	val filename = "myapp.log"
} with SavingAccount with FileLogger

// Another way to solve above problem
trait FileLogger extends Logger {
	val filename: String
	lazy val out = new PrintStream(filename)
	def log(msg: String): Unit = {
		out.println(msg)
	}
}
```

* trait extends class

```scala
// This feature is not used so frequently in general

trait LoggedException extends Exception with Logged {
	def log(): Unit = {
		log(getMessage())
	}
}

// A class A extends a trait which extends a class B, then
// class B will be the super class of class A automatically
class UnhappyException extends LoggedException {
	override def getMessage(): String = "arrgh!"
}
```

* Self type

```scala
trait LoggedException extends Logged {
	this: Exception =>
		def log(): Unit = {
			log(getMessge())
		}
}

// Self type means, above trait can only be mixed-into Exception's derived class

trait LoggedException extends Logged {
	self: { def getMessage(): String } =>
		def log(): Unit = {
			log(getMessage())
		}
}

// Note above example shows that self type can be apply to structural type
// means any class that has the same getMessage signature can mix this trait
```

* How Scala trait get compiled into Java

A trait w/ only abstract method:

```scala
trait Logger {
	def log(msg: String): Unit
}
```

becomes 

```java
public interface Logger {
	void log(String msg);
}
```


A trait has concrete methods, scala compiler will help us create an associated class:

```scala
trait ConsoleLogger extends Logger {
	def log(msg: String): Unit = {
		println(msg)
	}
}
```

becomes

```java
public interface ConsoleLogger extends Logger {
	void log(String msg);
}

public class ConsoleLogger$class {
	public static void log(ConsoleLogger self, String msg) {
		println(msg);
	}
}
```


A trait with fields: 

```scala
trait ShortLogger extends Logger {
	val maxLength = 15
}
```

becomes 

```java
public interface ShortLogger extends Logger {
	public abstract int maxLength();
	public abstract void weird_prefix$maxLength_$eq(int int);
}

public class ShortLogger$class {
	public void $init$(ShortLogger self) {
		self.werid_prefix$maxLength_$eq(15);
	}
}
```

### CH11: Operators

* Operators

```scala
a op b // a.op(b)
a op	// a.op()
op a   // a.unary_op()
a op= b // a = a op b
	// exception for <=, >=, !=, ==, ===, =/=
```

* Operator priority

```scala
a infix b postfix

// is equal to 

(a infix b) postfix
```

* Operator associativity

```scala
// most operator is left associativity
1 + 2 + 3 // (1 + 2) + 3
	
// some operator is right associativity
1 :: 2 :: Nil // 1 :: (2 :: Nil)
	
2 :: Nil // Nil.::(2) 
```

* `apply` and `unapply`

```scala
f(arg1, arg2, ...) // f.apply(arg1, arg2, ...)
f(arg1, arg2, ...) = value // f.update(arg1, arg2, ..., value)

// Above mechanism is used by array and map
val scores = new scala.collection.mutable.HashMap[String, Int]
scores("Bob") = 100 // scores.update("Bob", 100)
val bobsScore = scores("Bob") // scores.apply("Bob")


class Fraction(n: Int, d: Int) {
	...
}

object Fraction {
	def apply(n: Int, d: Int): Fraction = new Fraction(n, d)
}

val result = Fraction(3, 4) * Fraction(2, 5)

// the unapply method can be defined as the inverse of apply method
object Fraction {
	def unapply(input: Fraction) = {
		if (input.d == 0) None
		else Some((input.n, input.d))
	}
}

// the unapply method might not be the inverse of apply method
class Name(first: String, last: String)

object Name {
	def unapply(input: String) = {
		val pos = input.indexOf(" ")
		if (pos == -1) None
		else Some((input.substring(0, pos), input.substring(pos + 1)))
	}
}

val author = "Cay Horstmann"
val Name(first, last) = author

// cas class comes with apply and unapply
case class Currency(value: Double, unit: String)
Currency(29.95, "EUR") // calls Currency.apply
case Currency(amount, "USD") => println("$" + amount) // calls Currency.unapply

object IsCompound {
	def unapply(input: String) = input.contains(" ")
}

author match {
	case Name(first, last @ IsCompound()) => ...
	case Name(first, last) => ...
}
```

* `unapplySeq`

```scala
object Name {
	def unapplySeq(input: String): Option[Seq[String]] =
		if (input.trim == "") None else Some(input.trim.split("\\s+"))
}

author match {
	case Name(first, last) => ...
	case Name(first, middle, last) => ...
	case Name(first, "van", "der", last) => ...
}
```

### CH12: High-order Functions

* Convert a method into a function

```scala
import scala.math._
val num = 3.14
val fun = ceil _ // use underscore to convert a method into a function

def valueAtOneQuarter(f: (Double) => Double) = f(0.5)
valueAtOneQuarter(ceil _) // 1.0
valueAtOneQuarter(sqrt _) // 0.5
```

* Anonymous Function

```scala
val triple = (x: Double) => 3 * x	
Array(3.14, 1.42, 2.0).map((x: Double) => 3 * x)
// can also use curly bracket
Array(3.14, 1.42, 2.0).map { x: Double =>
	x * 3
}
```

* Function produces another function

```scala
def mulBy(factor: Double) = (x: Double) => factor * x
val quintuple = mulBy(5)
quintuple(20) // 100
```

* Function type detection

```scala
valueAtOneQuarter((x: Double) => 3 * x) // 0.75

// It seems valueAtOneQuarter is expecting a (Double) => Double function
valueAtOneQuarter((x) => 3 * x)

// If the pass-in function only has one param, parenthese can be ignored
valueAtOneQuarter(x => 3 * x)

// If the param pass-in function only use once
valueAtOneQuarter(3 * _)

// Note: _ can only be used if type can be detected previously
val fun = 3 * _ // wrong, cannot detect _'s type
val fun = 3 * (_: Double) // OK
val fun: (Double) => Double = 3 * _ // Ok, because we give type
``` 

* Some useful high-order function

```scala
(1 to 9).map("*" * _).foreach(println _)
(1 to 9).filter(_ % 2 == 0)
(1 to 9).reduceLeft(_ * _)
	// same as 1 * 2 * 3 * ... * 9
"Mary has a little lamb".split(" ").sortWith(_.length < _.length)
```

* Closure

```scala
def mulBy(factor: Double) = (x: Double) => factor * x
val triple = mulBy(3)
val half = mulBy(0.5)
println(triple(14) + " " + half(14)) // display 42 7

// Note, each return function of mulBy has it's own factor
// such function is called closure, it contains code and the factor used by code
// Above is achieved by using class and apply function, scala compiler will handle that
```

* SAM(single abstract method) transform

```scala
var counter = 0
val button = new JButton("Increment")
button.addActionListener(new ActionListener {
	override def actionPerformed(event: ActionEvent) {
		counter += 1
	}
})

// There's too many codes, if only we could have the following
button.addActionListener((event: ActionEvent) => count += 1)

// That can be achieved with implicit conversion
implicit def makAction(action: (ActionEvent) => Unit): ActionListener = 
	new ActionListener {
		override def actionPerformed(event: ActionEvent): Unit = {
			action(event)
		}
	}
```

* Currying

```scala
def mulOneAtATime(x: Int) = (y: Int) => x * y
mulOneAtATime(6)(7) //42

// scala support the following
def mulOneAtATime(x: Int)(y: Int) = x * y

// curring usage
val a = Array("Hello", "World")
val b = Array("hellow", "world")
a.corresponds(b)(_.equalsIgnoreCase(_))

// note
def corresponds[B](that: Seq[B])(p: (A, B) => Boolean): Boolean
```

* Abstraction Control

```scala
def runInThread(block: () => Unit) {
	new Thread {
		override def run() {
			block()
		}
	}.start()
}

runInThread {
	() => println("Hi");
	Thread.sleep(10000)
	println("Bye")
}

// But () => does not looks good

def runInThread(block: => Unit) {
	new Thread {
		override def run() {
			block
		}
	}.start()
}

runInThread {
	println("Hi")
	Thread.sleep(10000)
	println("Bye")
}

// We can create a abstract control
def until(condition: => Boolean) (block: => Unit) {
	if (!condition) {
		block
		until(condition) (block)
	}
}

var x = 10
until (x == 0) {
	x -= 1
	println(x)
}
```

### CH13: Collections

* Main collections' trait

```scala
// main collection's trait
trait Iterable

trait Seq extends Iterable
trait IndexedSeq extends Seq

trait Set extends Iterable
trait SortedSet extends Set

trait Map extends Iterable
trait SortedMap extends Map

// Iterable 
val coll = ... // some Iterable
val iter = coll.iterator
while (iter.hasNext) {
	iter.next()
}

// All scala's collection trait/class has apply method
Iterable(0xFF, 0xFF00, 0xFF0000)
Set(Color.RED, Color.GREEN, Color.BLUE)
Map(Color.RED -> 0xFF0000, Color.GREEN -> 0xFF00, Color.BLUE -> 0xFF)
SortedSet("Hello", "World")
```

* mutable and immutable collections

```scala
// scala.collection.Map is the base type of 
// scala.collection.mutable.Map and scala.collection.immutable.Map
// Scala predefine collection use immutable implementations, ie. Predef.Map is scala.collection.immutable.Map

def digits(n: Int): Set[Int] = {
	if (n < 0) digits(-1)
	else if (n < 10) Set(n)
	else digits(n / 10) + (n % 10)
}

// immutable collections
trait Seq
class List extends Seq
class Stream extends Seq
class Stack extends Seq
class Queue extends Seq
trait IndexedSeq extends Seq
class Vector extends IndexedSeq
class Range extends IndexedSeq

// mutable collections
trait Seq
class Stack extends Seq
class Queue extends Seq
class PriorityQueue extends Seq
class LinkedList extends Seq
class DoubleLinkedList extends Seq
trait IndexedSeq extends Seq
class ArrayBuffer extends IndexedSeq
```

* List

```scala
9 :: 4 :: 2 :: Nil

def sum(lst: List[Int]): Int = {
	if (lst == Nil) 0
	else lst.head + sum(lst.tail)
}

def sum(lst: List[Int]): Int = lst match {
	case Nil => 0
	case h :: t => h + sum(t)
}

List(9, 4, 2).sum

```

* LinkedList

```scala
val lst = scala.collection.mutable.LinkedList(1, -2, 7, -9)
var cur = lst
while (cur != Nil) {
	if (cur.elem < 0) cur.elem = 0
	cur = cur.next
}

var cur = lst
while (cur != Nil && cur.next != Nil) {
	cur.next = cur.next.next
	cur = cur.next
}

```

* Set 

```scala
Set(2, 0, 1) + 1 // same as Set(2, 0 , 1)
Set(1, 2, 3, 4, 5, 6)
	// the iteration order might be 5, 1, 6 ,2, 3, 4 because underneath it's implemented use a hash table
	
val weekdays = scala.collection.mutable.LinkedHashSet("Mo", "Tu", "We", "Th", "Fr")
	// use LinkedHashSet will use a underlying linked-list to keep track of insertion order
	
scala.colection.immutable.SortedSet(1, 2, 3, 4, 5, 6)
	// is implemented use a Red-Black tree
	
val digits = Set(1, 7, 2, 9)
digits contains 0	// false
Set(1, 2) subsetOf digits	// true

val primes = Set(2, 3, 5, 7)
digits ++ primes // digits union primes
digits | primes // digits union primes

digits & primes // digits intersection primes

digits &~ primes // digits diff primes
digits -- primes // digits diff primes
```

* Operator for add/remove element on collection

```scala
collection :+ elem	// Seq
elem +: collection	// Seq

collection + elem	// Set, Map
collection + (e1, e2, ...)	// Set, Map

collection - elem	// Set, Map, ArrayBuffer
collection - (e1, e2, ...)	// Set, Map, ArrayBuffer

collection1 ++ collection2	// Iterable

collection1 -- collection2	// Set, Map, ArrayBuffer

elem :: lst // List
lst2 ::: lst1 // List

set1 | set2 // Set
set1 & set2 // Set
set1 &~ set2 // Set

collection += elem	// mutable collection
collection += (e1, e2, ...) // mutable collection
collection1 ++= collection2  // mutable collection
collection -= elem // mutable collection
collection -= (e1, e2, ...) // mutable collection
collection1 --= collection2  // mutable collection

elem +=: coll // ArrayBuffer prepend 
```

* Methods in Iterable trait

```scala
head, last, headOption, lastOption // first or last elem or option
tail, init // without head or last
length, isEmpty
map(f), foreach(f), flatMap(f), collect(pf)
reduceLeft(op), reduceRight(op), foldLeft(init)(op), foldRight(init)(op) // apply binary op with specific order
reduce(op), fold(init)(op), aggregate(init)(op, combineOp) // apply binary op without a specific order
sum, product, max, min
count(pred), forall(pred), exists(pred)
filter(pred), filterNot(pred), partition(pred)
takeWhile(pred), dropWhile(pred), span(pred)
take(n), drop(n), splitAt(n)
takeRight(n), dropRight(n)
slice(from, to)
zip(collection2), zipAll(collection2, fill, fill2), zipWithIndex
grouped(n), sliding(n) // both produce an interator. 
                                   // grouped(n) return a iterator with n elem and then shift n. 
                                   // sliding(n) return an iterator with n elem, then sift 1
mkString(before, between, after), addString(stringBuilder, before, between, after)
toIterable, toSeq, toIndexedSeq, toArray, toList, toStream, toSet, toMap
copyToArray(arr), copyToArray(arr, start, length), copyToBuffer(buf)
```

* Methods in Seq trait

```scala
contains(elem), containSlice(seq), startsWith(seq), endsWith(seq)
indexOf(elem), lastIndexOf(elem), indexOfSlice(seq), lastIndexOfSlice(seq)
indexWhere(pred)
prefixLength(pred), segmentLength(pred, n)
padTo(n, fill)
intersect(seq), diff(seq)
reverse
sorted, sortWith(less), sortBy(f)
permutations, combinations(n)
```

* Collection map

```scala
val names = List("Peter", "Paul", "Mary")
names.map(_.toUpperCase)
def ulcase(s: String) = Vector(s.toUpperCase(), s.toLowerCase())

names.flatMap(ulcase)

"-3+4".collect {
	case '+' => 1
	case '-' => -1
}  // Vector(-1, 1)

names.foreach(println)
```

* Reduce, Fold and Scan

```scala
List(1, 7, 2, 9).reduceLeft(_ - _)
	// ((1 - 7) - 2) - 9 = -17 
List(1, 7, 2, 9).reduceRight(_ - _)
	// 1 - (7 - (2 - 9)) = -13

List(1, 7, 2, 9).foldLeft(0)(_ - _)
	// (((0 - 1) - 7) - 2) - 9 = 19

(0 /: List(1, 7, 2, 9)) (_ - _)  // same as List(1, 7, 2, 9).foldLeft(0) (_ - _)
(List(1, 7, 2, 9) :\ 0) (_ - _)  // same as List(1, 7, 2, 9).foldRight(0) (_ - _) 

val freq = scala.collection.mutable.Map[Char, Int]()
for (c <- "Mississippi") freq(c) = freq.getOrElse(c, 0) + 1	

// Above loop can be replaced by foldLeft
(Map[Char, Int]() /: "Mississippi") {
	(m, c) => m + (c -> m.getOrElse(c, 0) + 1)
} // result is an immutable map

// Note: Any while loop can be replaced use foldLeft, foldRight

(1 to 10).scanLeft(0) (_ + _)
	// scan generate all intermedia result
	// Vector(0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55)
```

* zip

```scala
val prices = List(5.0, 20.0)
val quantities = List(10, 2, 1)
(prices zip quantities) map { 
	p => p._1 * p._2
} // List(50.0, 40.0, 9.95)

List(5.0, 20.0, 9.95).zipAll(List(10, 2), 0.0, 1)
	// List((5.0, 10), (20.0, 2), (9.95, 1))
	
"scala".zipWithIndex
	// Vector((s,0), (c,1), (a,2), (l,3), (a,4))

```

* iterator

```scala
while(iter.hasNext) {
	iter.next()
}

while(elem <- iter) {
	elem
}
```

* Stream

```scala
def numsFrom(n: BigInt): Steam[BigInt] = n #:: numsFrom(n + 1)
val tenOrMore = numsFrom(10)
	// Stream(10, ?)
tenOrMore.tail.tail.tail
	// Stream(13, ?)
val squares = numsFrom(1).map(x => x * x)
	// Stream(1, ?)
squares.take(5).force
	// Steam(1, 4, 9, 16, 25)
squares.force
	// Don't do this, will get OutOfMemoryError

val words = Source.fromFile("/usr/share/dict/words").getLines.toStream
words // Stream(A, ?)
words(5) // Aachen
words // Stream(A, A's, AOL, AOL's, Aachen, ?)
```

* Lazy view

```scala
val powers = (0 until 1000).view.map(pow(10, _))
	// will produce a view that is always being lazy calcuate and never cache any value
powers(100)
	// only pow(10, 100) is calculated 
	// and will not be cached
	// if call powers(100) again, pow(10, 100) will be calculated again
	
(0 to 1000).map(pow(10, _)).map(1 / _)
	// calculate map of all element and then caculate the other map

(0 to 1000).view.map(pow(10, _)).map(1 / _).force
	// calculate two maps together for each element, there's no middle temp collection	
```

* Thread-safe collections

```scala
trait SynchronizedBuffer
trait SynchronizedMap
trait SynchronizedPriorityQueue
trait SynchronizedQueue
trait SynchronizedSet
trait SynchronizedStack

val scores = new scala.collection.mutable.HashMap[String, Int] with scala.collection.mutable.SynchronizedMap[String, Int]
```

* Parallel collections

```scala
collection.par.sum
collection.par.count(_ % 2 == 0)
for(i <- (0 until 100).par) print(i + " ")
	// 1 26 51 27 ...
for(i <- (0 until 100).par) yield i + " "
	// ParVector(1 , 2 , 3 , 4 , 5 , ...)

// par methods return trait ParSeq, ParSet, ParMap, which is the subtype of ParIterable

var count = 0
for (c <- collection.par) {
	if (c % 2 == 0) {
		count += 1 // Wrong
	}
}
```

### CH14: Pattern Matching and Case Class

* Pattern matching as a better switch statement

```scala
sign = ch match {
	case '+' => 1
	case '-' => -1
	case _ => 0
}

color match {
	case Color.RED =>
	case Color.BLACK =>
	...
}
```

* Guard

```scala
ch match {
	case '+' => sign = 1
	case '-' => sign = -1
	case _ if Character.isDigit(ch) => digit = Character.digit(ch, 10)
	case _ => sign = 0
}
```

* Variables in pattern matching

```scala
str(i) match {
	case ch if Character.isDigit(ch) => digit = Character.digit(ch, 10)
	...
}
```

* Type matching

```scala
obj match {
	case x: Int => x
	case s: String => Integer.parseInt(s)
	case _: BigInt => Int.MaxValue
	case _ => 0
}
```

* Array, List and Tuple matching

```scala
arr match {
	case Array(0) => "0"
	case Array(x, y) => x + " " + y
	case Array(0, _*) => "0 ..."
	case _ => "something else"
}

lst match {
	case 0 :: Nil => "0"
	case x :: y :: Nil => x + " " + y
	case 0 :: tail => "0 ..."
	case _ => "something else"
}

pair match {
	case (0, _) => "0 ..."
	case (y, 0) => y + " 0"
	case _ => "neither is 0"
}
```

* Extractor

```scala
// Extractor use unapply or unapplySeq to extra value 
arr match {
	case Array(0, x) => ...
	...
}

val pattern = "([0-9]+) ([a-z]+)".r
"99 bottle" match {
	case pattern(num, item) => ...
		// set num to "99" and item to "bottles"
}

//	 pattern.unapplySeq("99 bottles") to assign to num and item
```

* Pattern matching in variable assignment

```scala
val (x, y) = (1, 2)
val (q, r) = BigInt(10) /% 3
val Array(first, second, _*) = arr
```

* Pattern matching in for loop

```scala
for ((k, v) <- myMap) {
	println(k + " -> " + v)
}

for ((k, "") <- myMap) {
	println(k)
}

for ((k, v) <- myMap if v == "") {
	println(k)
}
```

* case class

```scala
abstract class Amount 
case class Dollar(value: Double) extends Amount 
case class Currency(value: Double, unit: String) extends Amount
case object Nothing extends Amount

amt match {
	case Dollar(v) => "$" + v
	case Currency(_, u) => "Oh noes, I got " + u
	case Nothing => ""
}

// case class/object is really good at pattern matching 
// and it is different from normal class/object in the following way:
// 1) all param in constructor is val by default
// 2) come with the `apply` method, so `new` key-word is not a must
// 3) come with the `unapply` method for pattern matching
// 4) come with `toString`, `equals`, `hashCode` and `copy` method
```

* copy method 

```scala
val amt = Currency(29.95, "EUR")
val price = amt.copy()
val price = amt.copy(value = 19.95) // Currency(19.95, "EUR")
val price = amt.copy(unit = "CHF") // Currency(29.95, "CHF")
```

* Match decomposition on infix operator

```scala
case class %%%(x: Int, y: Int)

a match {
  case x %%% y => x + y
}

amt match {
	case a Currency u => ... // equivalent to case Currency(a, u)
}

// seems to be silly, but above example is super useful for the following

case class ::[E](head: E, tail: List[E]) extends List[E]

lst match {
	case h :: t => ...
		// case ::(h, t) and use ::.unapply(lst)
}

result match {
	case p ~ q => ... // case ~(p, q)
	case p ~ q ~ r => ...
		// case ~(~(p, q), r)
}

// Note :: is right associative 

case first :: second :: rest 
	// case ::(first, ::(second, rest))
	
case object +: {
	def unapply[T](input: List[T]) = {
		if (input.isEmpty) None else Some((input.head, input.tail))
	}
}

1 +: 7 +: 2 +: 9 +: Nil match {
	case first +: second +: rest => first + second + rest.length
}
```

* Matching recursive data structure

```scala
abstract class Item
case class Article(description: String, price: Double) extends Item
case class Bundle(description: String, discount: Double, items: Item*) extends Item

val item = Bundle("Father's day special", 20.0, 
				Article("Scala for the Impatient", 29.95),
				Bundle("Anchor Distillery Sampler", 10.0, 
					Article("Old Potrere Straight Rye Whisky", 79.95),
					Article("Junipero Gin", 32.95)))
item match {
	case Bundle(_, _, Article(descr, _), _*) => ...
	case Bundle(_, _, art @ Article(_, _), rest @ _*) => ...
	case Bundle(_, _, art @ Article(_, _), last) => ...
}

def price(it: Item): Double = it match {
	case Article(_, p) => p
	case Bundle(_, disc, its @ _*) => its.map(price _).sum - disc
}
```

* case class equals

```scala
Currency(10, "EUR") == Currency(10, "EUR") // true
```

* sealed case class

```scala
sealed abstract class Amount
case class Dollar(value: Double) extends Amount
case class Currency(value: Double, unit: String) extends Amount

// sealed case class forced all derived case class to be defined in the same file

sealed abstract class TrafficLightColor
case object Red extends TrafficLightColor
case object Yellow extends TrafficLightColor
case object Green extends TrafficLightColor

color match {
	case Red => "Stop"
	case Yellow => "hurry up"
	case Green => "go"
}
```

* Option type

```scala
// Option is either Some(x) or None
scores.get("Alice") match {
	case Some(score) => println(score)
	case None => println("No score")
}

val alicesScore = score.get("Alice")
if (alicesScore.isEmpty) println("No score")
else println(alicesScore.get)

println(alicesScore.getOrElse("No score"))

println(score.getOrElse("Alice", "No score"))

for (score <- scores.get("Alice")) println(score)

scores.get("Alice").foreach(println _)
```

* Partial Function

```scala
val f: PartialFunction[Char, Int] = {
	case '+' => 1
	case '-' => -1
}

f('-') // calls f.apply('-') return -1
f.isDefinedAt('0') // false
f('0') // throws MatchError

// Note collect function accepts partial function

"-3+4".collect {
	case '+' => 1
	case '-' => -1
} // Vector(-1, 1)
```

