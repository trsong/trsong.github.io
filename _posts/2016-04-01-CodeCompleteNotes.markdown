---
layout: post
title:  "Code Complete Notes"
date:   2016-04-01 22:36:32 -0700
categories: Design Documents
---
* This will become a table of contents (this text will be scraped).
{:toc}

## Code Complete Notes

### Purpose of routine
Reason for routine is to achieve the balance between **readability**(abstraction) and  **maintainess**(avoid duplicate and reduce complexity)
The rule of creating a good class also applies to creation of routine. 

Here’s a summary list of the valid reasons for creating a routine: 

* Reduce complexity - hide implementation detail and create a good level of abstraction so that we can reuse code 
* Make a section of code readable 
* Avoid duplicate code - maintaining two sets of code is really expensive, any change happen to one place also need to apply to the other
* Hide sequences - some combinations of function need to call in a sequence. By using routine to wrap some logical chuck of code to hide the implementation details
* Hide pointer operations - wrap low level  operation to more readable wrapper so that we can reuse code
* Improve portability 
* Simplify complicated boolean tests 
* Improve performance - avoid redundant, make it easier to refactor code and improve code performance

#### Why create a small routine
* Improve readability 
* self-documented

#### Routine Level Design - How to Write A Purposeful Code 
**Higher cohesion** = One function one goal (preferred)  -> Higher reliability (less bug)

**Lower cohesion** = One function multiple goals

High cohesion function should have a purposed named. but strong or weak cohesion is depended on implementation detail. 
A function should do what its name suggest and nothing more, nothing less otherwise it’s bad named  

**Sequential Cohesion** = contains some operation with some execution order and have some dependency

**Communication Cohesion** = functions share the same data, but don’t have any dependency on each other

**Temporal Cohesion** = functions combine temporally to execute together. (Orchestrate activities than doing it directly)

**Procedural Cohesion** = operations have some logical order, they might not have dependency

**Logical Cohesion** = whether or not execute some operations are depended on boolean flag (shouldn’t do it, except for eventHandler)

**Coincidental Cohesion** = no cohesion = chaotic cohesion = code so bad that need to do a deeper redesign or reimplement

### Good Routine Name
#### Name of the routine should

- describe everything routine does
- avoid meaningless verb, like handle, use detailed verb
- be long-name
- describe the return type if routine returns a value
- mention the object if perform operations on that object
- use opposite name: openFile vs closeFile
- consistent naming

Avoid too long files - be careful on more than 200 line code file

#### Routine Parameters 

- Input output Order:  InputParam , InputOutputParam, outputParam (in first then out)
- Type Order: Type1, Type2, Type3 (similar type stay close to each other)
- Use all parameters
- State/Err parameters go last
- Never change the value of input parameter
- Document: type, input/output, meaning, purpose, range, unit
- fewer parameters more likely to be reuse 
- passing whole object or pass each filed individually: depend on how many usage
- actual parameter match the interface 


### Defensive Programming = guarding against errors you don’t expected
A good program uses “Garbage in, nothing out”, “Garbage in, err msg out” and “no garbage allowed in”

#### Handle garbage:
- Check external source: buffer overflow, SQL injection, html/xml injection, integer overflow
- Check all routine input parameters
- Handle bad inputs

#### Assertions
- especially useful in large, complicated programs
- also useful in high-reliability programs 
- allow more quickly detecting bad interface assumption
- Use assertion to document assumptions, e.g.: check initialization, current states, non-NULL, range, empty
- Avoid executables code in assertion
- assert and document pre-condition and post-condition
- assert and then handle the err for highly robust code

Rule: err handling for expected, assertion for unexpected. ie. err handling bad inputs, assert check for bugs

Error handling techniques

- Return a neutral value
- Skip and read next valid data
- Log a warning
- Return an error code by: setting a status variable, return states, throw an exception and so on
- Call a centralized error handler. but drawbacks is: increase coupling(hard to reuse and refactor), single point of failure (easy to attack). Advantage: easy to debug
- Display error message wherever err occurs: it may give too much information to attackers
- Shutdown: in some case where we’d rather reboot than deliver wrong output
- be consistent for level of error handling

#### Generic Class for Defensive Programming
Ex. Generic Programming for Handling Exceptions

```java
public class Test {
   class Animal {}
   
   class Cat extends Animal {}
   class Dog extends Animal {}
   
   private List<Cat> catList = new ArrayList<>();
   private List<Animal> animals = catList;  // Type cast err
   
   public void test() {
      Cat[] cats = {};
      Animal[] animals = cats;
      animal[0] = new Dog();
      Cat cat = cats[0]; // Try to fool the compiler, it works
   }
   
   public void test2(Function<Cat, Animal> handler) {
      hanlder.apply(new Cat());
   }
   
   public void test3(Function<? super Cat, Animal> handler) {
      hanlder.apply(new Cat());
   }
   
   public void test3() {
      Function<Animal, Animal> bla = null
      test2(bla); // Test cast err
      test3(bla); // Works
      // However, we cannot have not way Animal subclass ? work
      // Because Java always implement Generic class as Invariance
      // there's no way to implement Co-variance
   }
   
}
```

Now let's see how Scala handls co-variance

```scala
class ScalaTest {
   class Animal()
   class Cat() extends Animal
   
   class Handler[-T, +K] {
      def handle(input: T): K
   }
   
   trait Container[+K] {
      def output(): K
   }
   
   trait Function[-K] {
      def call(input: K): Unit
   }
   
   trait MyList[K] {        // cannot put -K or +K either one blow will break
      def add(item: K): Unit
      def get(index: Int): K
   }
   
   def test(hanlder: Handler[Cat, Animal]): Unit = {
      handler.handle(new Cat)
   }
   
   def anotherTest(): Unit {
      val myHandler: Handler[Animal, Cat] = null
      test(myHandler)
   }
}

```

### Robustness vs. Correctness
Correctness = never return incorrect result
Robustness = keep running even if incorrect result occurs

#### Exception
- Throw an exception only for conditions that are truly exceptional: e.g., event that should never occurred 
- Handle it locally than throw an exception
- Throw exception at the correct level of abstraction: don’t throw too low-level exception, make sure to use correct level of exception
- never leave catch empty. because that means doing nothing when exception occurred 
- use centralized exception reporter

### Barricade
Define some part of software handling dirty data and some handle clean. so that we can create boundaries of “safe” area.
Make sure to remember to convert input data to proper type at input time.

Product version and development version of program can be different. 

We can use more offensive approach in development version to make it easy to detect bugs: assert abort program, fill all memory, fill all file/stream, abort on else statement……


### Debug - Find the root cause of defects
Keep In mind: General Principle of Software Quality: Improving quality reduces development costs 
Learn through debugging: 

- learn program you are working on
- learn the kind of mistakes you make
- learn how you solve the problem
- learn the quality of your code
- learn how you fix defects

An ineffective approach for debugging

- Find the defect by guessing
- Don’t waste time trying to understand the problem
- Fix the err with the most obvious fix
When you start debugging, 
- Assume you are the one who cause the bug, and don’t assume your code is error-free

#### The scientific method of debugging
1. Stabilize the error: find a reliable way to reproduce the scenario
2. Locate the source of the error
3. Fix the defect
4. Test the fix
5. Look for similar error

#### For Locate the source of the error:
1. Gather the data that produce the defect
2. analyze the data and form hypothesis about the defect
3. determine how to test the hypothesis: test the program or examine the code
4. prove or disprove, repeat

#### Tips for finding defects:
- use all the data to make hypothesis: your hypothesis should explain all current data gathered
- refine the test case to produce the error
- test the code in different unit test suites: smaller piece of code is easier to test
- use debug tool
- reproduce the error several different ways to find the root cause
- gather more data to make more hypothesis 
- disapprove the hypothesis by using the negative tests
- brainstorm for possible hypothesis
- narrow the suspicious region of code
- be suspicious of classes and routines that have had defects before
- check the recent changed code
- debug by increase the search are gradually
- check for common defects
- talk to someone else who know about the problem
- Take a beak from the problem to calm down and think in another direction 
- Brute Force Debugging: create list of brute force debugging goal
- Divide and conquer: divide the program into sections and debug into them one by one

#### Fixing a defect
- understand the problem before you fix: don’t try to fix the problem if you don’t really understand the problem
- understand the program not just the problem
- don’t rush into coding, spend time reproduce the problem and prove your hypothesis
- after fix, you need to verify the result to make sure the problem is solved and won’t break the program
- don’t change the program randomly and claim that is a fix
- look for similar defects