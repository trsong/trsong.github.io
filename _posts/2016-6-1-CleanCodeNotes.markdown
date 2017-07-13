---
layout: post
title:  "Clean Code Notes"
date:  2016-06-01 22:36:32 -0700
categories: Design Documents
---
* This will become a table of contents (this text will be scraped).
{:toc}

## Clean Code Notes

### CH1. Clean Code
causing code messy is easy, but maintaining messy code is expensive.
The inventor of c++ think clean code means "**elegant**":

1. code pleasing to read (well-crafted or well-designed)
2. efficient
3. focused

And the following is also important:

1. run all tests
2. no duplication
3. express all the design idea in the system
4. function name suggests its bahaviours


### CH2 Meaningful names
#### Use Intention-Revealing Names 
- Avoid Disinformation 
- Make Meaningful Distinctions 
- Use Pronounceable Names 
- eg. Dialog 

#### Use Searchable Names 

- eg. AbstractGridCellRenderer, scenarioGridCellRenderer, attributeGridCellRenderer, 
      
#### Avoid Encodings 

- eg. i18n    —> what is i18n

#### Pick One Word per Concept 

- eg. getOrElseUpdate -> computeIfAbsent

#### Use Solution Domain Names 

#### Use Problem Domain Names 


### CH3 Function
#### Small! 
function should be small and meaningful

#### Do One Thing
function should do one thing, do it well and do it only

bad example showing how a function has so many parameter to control the workflow
scenarioService: 355 clone scenario

#### One Level of Abstraction per Function 
bad example showing different level of abstraction
scenarioService.scala: 1011 updateScenarioStateWithPlan

We want the code to read like a top-down narrative.5 We want every function to be fol- lowed by those at the next level of abstraction so that we can read the program, descending one level of abstraction at a time as we read down the list of functions. I call this The Step- down Rule. 

#### Switch Statements 
why switch statement should hide behind an interface to support polymorphism ?

#### Use Descriptive Names 
using the same phrase pattern to name things
assumptionService, assumptionLoader, assumptionController,
scenario….
plan…

#### Function Arguments 
#### Have No Side Effects
A function that has a side effect will create coupling to other objects. Which is hard to detect and maintain, and it’s usually buggy
And it’s against “DO ONE THING”

#### Command Query Separation 

```
if(getOrElseSet()){
} // bad code
```

```
if(exist) {
  set
}// clear code
```

#### Prefer Exceptions to Returning Error Codes 
use try catch block is much clear than nested if statement to detect the err
 
#### Don’t Repeat Yourself 
duplicated code make is hard to maintain. if have N copy, then 1 place with bug will result in N-fold effort for fixing bug in duplicated place

### CH4 Comments
#### Comments Do Not Make Up for Bad Code 
fixing it! don’t leave comments.
But guess sometimes we have to leave a comment. e.g.. Partial refactoring

#### Explain Yourself in Code 
if we already explain everything in code then why leave a comment

#### Informative Comments 
Good usage of comments is to explain why I code like this not that

#### TODO Comments 
what ever TODO might be, it shall ever become the reason for making up for bad code

Comments is meant to help programmer get a better knowledge of the code, it should not be meant to cause noise and misleading.
We should rely on self document and IDE more often than leaving a comment.

### CH5 Formatting
Code formatting is about communication which is first order of a professional developer.

Small file (around 200 lines) are usually easier to understand than large files.

Vertical Density implies close association.  eg. gridController.js 

Interesting facts: the caller should be above the callee. 


### CH6 Objects and Data Structures
Object hides the implementation and provides abstraction.
Data structure expose its data and have no meaningful functions.

Procedural code makes it easy to add new functions without changing the data structure.
OO code, on the other hand, makes it easy to add new classes without changing the existing functions.

**Procedural code**: hard to add new data structure, because all of the existing function must change.
OO code: hard to add new function, because all the classes must change

In conclusion: 
Objects expose behaviour and hide data.
Data structures expose data and have no significant behaviour.

Note: calling a.b.c.d will add coupling so that later on make it even harder to refactor on a.

### CH7 Error Handling
Define the normal workflow. It’s better if we define the edge case class than create a try-catch logic because the logic is fully encapsulated.
Ex. NullHistoricalCache

null is dangerous. We should avoid passing null in code whenever possible to prevent from NullPointerException

Last time: catch exception and re-throw example planLoader.scala:65 

### CH8 Boundaries
#### Using Third-Party Code
**Code Provider:** strive for broad applicability
**Code User:** want an interface focused on their particular needs

**Learning Tests**: boundary tests testing our understanding of third-party api, provide example of third-party api usage and also prevent its upgrade incompatibility and change of code to meet in new needs

We should avoid letting too much of our code know about third-party code so that we can wrap the third party into Map or adapter to convert their interface to the interface we are looking for (ie. create a clear boundary).  Thus, we are able to rely on something we can have control of. 

Note: I think not only third-party api needs boundary. The framework api, or some low level api also needs boundary tests. Eg. cache
e.g. /modules/engine-play/src/main/scala/com/visiercorp/server/cache/package.scala

The default implementation of the play cache API uses EHCache
EhcacheAdapterSpec.scala is our boundary tests for Ehcache

### CH9 Unit Tests
#### Three Laws of Test-Driven Development(TDD)

- First Law You may not write production code until you have written a failing unit test. 
- Second Law You may not write more of a unit test than is sufficient to fail, and not compiling is failing. 
- Third Law You may not write more production code than is sufficient to pass the currently failing test. 

#### Clean Tests is all about READABILITY
A good example would be: BUILD-OPERATE-CHECK pattern for unit-test. And Given-when-then convention. 

Clean Tests can also be F.I.R.S.T.

- **Fast**: tests need to be fast to run; otherwise no one is willing to run tests

- **Independent**: test should be independent. Bad example: Selenium 3 year, it’s a 20-30min ui tests which features a complete 3 year workflow of planning. Drive people crazy when first part of test fails 
- **Repeatable**: test should be repeatable under any circumstances.  Use mock object avoid states. 
- **Self-validating**: Either passed or failed on which line. Need to be clear. Bad Example, can’t tell which tests and which line of that test is falling for it:Test
- **Timely**: TDD, write test before write production code. because some part of code is poorly designed and unable to be tested.
Automation: 


### CH10 Classes
```js
class MyClass {
	public static constants
	private static variables
	private instance variables
	public functions
	private utilities
}
```

Name of class: describe what responsibilities it fulfills. 
The Single Responsibility Principle states that a class or module should have one, and only one, reason to change.
Maximally cohesive: A class in which each variable is used by each method is maximally cohesive. 

A class should have single responsibility and achieve as high cohesive as possible (use as many variable as possible). 

### CH11 Systems

System should evolve appropriate levels of abstraction and modularity that make it possible for individuals and the “components” they manage to work effectively, even without understanding the big picture. 

A system should **Separate construct from usage**. (separate Init and use)
1. Move init to a function call Main, after Main, we can assume all the service is ready to be use. All other function inside the class should also assume the service is ready to be use
2. Dependency Injection
DefaultPlanningServiceLocatorFactory in PlanningServiceLocatorFactory.scala


Systems Need Domain-Specific Languages 

### CH17 Smells and Heuristics

#### Comments 
* No longer valid comments
* Redundant comments

#### Testing Environment
* One click TestAll
* One click build

#### Functions
* Too many params 
* Eliminate flag arguments
* Get rid of dead function
* Beware of return value

#### General
* Obvious behaviour is unimplemented
* write tests for boundaries
* code at wrong level of abstraction
* Base class should not depend on derivatives
* Prefer Polymorphism to IF/Else/Switch/Case
* Encapsulate Conditions

```typescript
   if (shouldBeDeleted(timer)){

	}
```

is preferable to
	
	
```typescript
	if (timer.hasExpired() && !timer.isRecurrent()){
	
   }
```

* Avoid Transitive Navigation
 `a.getB().getC().getD()`

#### Java
* Avoid import package.*, if possible should specific the list
* Don't inherit constants. Use "import static" instead


#### Review Example:
1. Lacking comments/  Too many parameters / should using val instead of var
http://reviews.visier.corp/r/6326/ 
2. Encapsulate Conditions / Update comments
http://reviews.visier.corp/r/6327/
3. Hardcoded Constant for tests / Test not follow certain rules just try to testing everything/ Test is hard to follow
http://reviews.visier.corp/r/6329/
