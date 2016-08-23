---
layout: post
title:  "Design Patterns"
date:   2016-08-04 22:23:32 -0700
categories: Design-Patterns
---
* This will become a table of contents (this text will be scraped).
{:toc}

<a name="structuralPatterns"></a>
### Design Patterns - Structural Patterns
***
Structural design patterns are design patterns that ease the design by identifying a simple way to realize **relationships between entities**.

Examples of Structural Patterns includes:

* **Adapter pattern**: 'adapts' one interface for a class into one that a client expects
* **Bridge pattern**: decouple an abstraction from its implementation so that the two can vary independently
* **Composite pattern**: a tree structure of objects where every object has the same interface
* **Decorator pattern**: add additional functionality to a class at runtime where subclassing would result in an exponential rise of new classes
<br/>

#### Adapter/Wrapper/Translator
***
The adapter pattern allows the interface of an existing class to be used as another interface. Note that: *The interface may be incompatible but the inner function must suit the need.*

```scala
implicit def adaptee2Adapter(adaptee: Adaptee): Adapter = {
  new Adapter {
    override def clientMethod: Unit = { 
    // call Adaptee's method(s) to implement Client's clientMethod */ 
    }
  }
}
```

Example in our code base: EhcacheManagerAdapter.scala

```scala
/**
 * This class adapts an Ehcache cache manager to the [[CacheManager]] interface.
 * @param underlyingManager Adapts this manager
 */
class EhcacheManagerAdapter(val underlyingManager: EhcacheManager) extends AbstractCacheManager {

}

/**
 * An internal [[Cache]] implementation that adapts an ehcache cache
 * to the [[Cache]] interface.
 * @param underlyingCache The underlying ehcache cache object
 * @tparam K Type of the key
 * @tparam V Type of the value
 */
class EhcacheAdapter[K, V](manager: EhcacheManagerAdapter, val underlyingCache: Ehcache) extends Cache[K, V] {

}

```

<br/>

#### Bridge
***
The bridge pattern decouples an abstraction from its implementation so that the two can vary independently. Note that: 

The Bridge Pattern

* uses encapsulation, aggregation, and can use inheritance to separate responsibilities into different classes.
* is useful when both the class and what it does vary often.
* the class itself can be thought of as the abstraction and what the class can do as the implementation. 
* can also be thought of as two layers of abstraction.
* is often confused with the **adapter pattern**. In fact, the **bridge pattern** is often implemented using the class **adapter pattern**.
* The implementation can be further decoupled to the point the abstraction is utilized.

```scala

trait DrawingAPI {
  def drawCircle(x: Double, y: Double, radius: Double)
}

class DrawingAPI1 extends DrawingAPI {
  def drawCircle(x: Double, y: Double, radius: Double) = println(s"API #1 $x $y $radius")
}

class DrawingAPI2 extends DrawingAPI {
  def drawCircle(x: Double, y: Double, radius: Double) = println(s"API #2 $x $y $radius")
}

abstract class Shape(drawingAPI: DrawingAPI) {
  def draw()
  def resizePercentage(pct: Double)
}

class CircleShape(x: Double, y: Double, var radius: Double, drawingAPI: DrawingAPI)
    extends Shape(drawingAPI: DrawingAPI) {

  def draw() = drawingAPI.drawCircle(x, y, radius)

  def resizePercentage(pct: Double) { radius *= pct }
}

object BridgePattern {
  def main(args: Array[String]) {
    Seq (
	new CircleShape(1, 3, 5, new DrawingAPI1),
	new CircleShape(4, 5, 6, new DrawingAPI2)
    ) foreach { x => 
        x.resizePercentage(3)
        x.draw()			
      }	
  }
}

```

Example in our code base: HeadcountForecastBridge.scala


```java
public interface HeadcountForecastDAO{
...
}
```

```scala
// planLoader, scenarioLoader, serviceLocatorFactory are injected dependency which will change the implementation
// HeadcountForecastDAO interface is the abstract interface we can use
class HeadcountForecastBridge @Inject() (planLoader: PlanLoader, scenarioLoader: ScenarioLoader, serviceLocatorFactory: WFAServiceLocatorFactory)
  extends HeadcountForecastDAO {
  
  }
```

<br/>

#### Composite
***
The Composite Pattern is a **partitioning** design pattern which can be a tree structure of objects where every object has the same interface. A composite is an object designed as a composition of one-or-more similar objects, all exhibiting similar functionality. This is known as a "has-a" relationship between objects.

Motivation: When dealing with Tree-structured data, programmers often have to discriminate between a leaf-node and a branch. This makes code more complex, and therefore, error prone. The solution is an interface that allows treating complex and primitive objects uniformly. 

Note: The operations you can perform on all the composite objects often have a **least common denominator relationship**. For example, it would be useful to define resizing a group of shapes to have the same effect (in some sense) as resizing a single shape.

```scala
trait Component { def operation(): Unit }

class Composite extends Component {
  def operation() = children foreach { _.operation() }
  def add(child: Component) = ...
  def remove(child: Component) = ...
  def getChild(index: Int) = ...
}

class Leaf extends Component {
  def operation() = ...
}

trait Component{
    def operation():Unit = this match {
      case c:Composite => c.children.foreach(_.operation())
      case leaf:Leaf => println("leaf")  
    }
}

object BridgePattern {
  def main(args: Array[String]) {
    val comp = new Composite()
    comp += child1
    comp += child2
    comp -= child1
    val firstChild = comp(0)
  }
}
```
<br/>

Example in our code base: The whole Internal DSL uses composite pattern, like ModelingSyntax.scala

```scala
def doubleMetricOf(const: Double): Exp[Metric[Double]]

def floor[T <: ModelTypes](metric: Exp[Metric[T]])

class ModelingArithmeticOps[T <: ModelTypes : TypeTag](metric: Exp[Metric[T]]) {
   def +[F <: ModelTypes](addend: Exp[Metric[F]]): Exp[Metric[T]]
}

// floor(doubleMetricOf(1.5D)) + doubleMetricOf(2.0D) is of type Exp[Metric[Double]]

```

#### Decorator
***
The Decorator Pattern allows behavior to be added to an individual object, either statically or dynamically, without affecting the behavior of other objects from the same class.

Motivation: As an example, consider a window in a windowing system. Assume the window class has no functionality for adding scrollbars. One could create a subclass ScrollingWindow that provides them, or create a ScrollingWindowDecorator that adds this functionality to existing Window objects. At this point, either solution would be fine. This problem gets worse with every new feature or window subtype to be added.

```scala
class Coffee {
  val sep = ", "
  def cost:Double  = 1
  def ingredients: String = "Coffee"
}

trait Milk extends Coffee {
  abstract override def cost = super.cost + 0.5
  abstract override def ingredients = super.ingredients + sep + "Milk"
}

trait Whip extends Coffee {
  abstract override def cost = super.cost + 0.7
  abstract override def ingredients = super.ingredients + sep + "Whip"
}

trait Sprinkles extends Coffee {
  abstract override def cost = super.cost + 0.2
  abstract override def ingredients = super.ingredients + sep + "Sprinkles"
}

object DecoratorSample {
  def main(args: Array[String]) {
    var c: Coffee = new Coffee with Sprinkles
    printf("Cost: %f Ingredients %s\n", c.cost, c.ingredients)
     
    c = new Coffee with Sprinkles with Milk
    printf("Cost: %f Ingredients %s\n", c.cost, c.ingredients)
    
    c = new Coffee with Sprinkles with Milk with Whip
    printf("Cost: %f Ingredients %s\n", c.cost, c.ingredients)
  }
}

```
<br/>


### Design Patterns - Behavioral Patterns
***
Behavioral Design Patterns are design patterns that **identify common communication patterns between objects and realize these patterns**. By doing so, these patterns increase flexibility in carrying out this communication.

Examples of Behavioral Patterns includes:

* **Chain-of-responsibility Pattern**: Command objects are *handled or passed on* to other objects by logic-containing processing objects
* **Command Pattern**: Command objects encapsulate an action and its parameters
* **Interpreter Pattern**: Implement a specialized computer language to rapidly solve a specific set of problems

<br/>

#### Chain-of-responsibility Pattern
***
Chain-of-responsibility Pattern consists of **a source of command objects** and **a series of processing objects**. Each processing object contains logic that defines the types of command objects that it can handle; the rest are passed to the next processing object in the chain.

Example would be the mechanism of exception handling. A source of command objects is the exception object and a series of processing objects are exception handlers at each level.

Note: the source don't know who should handle until the runtime.

In scala, the pattern matching use Chain-of-responsibility Pattern.

```scala
parentClassInstance match{
   case instance: SubClass1 =>
   case instance: SubClass2 => 
}

```

More examples for Chain-of-responsibility pattern in scala:

```scala
case class Event(level: Int, title: String)

//Base handler class
abstract class Handler {
  val successor: Option[Handler]
  def handleEvent(event: Event): Unit
}

//Customer service agent
class Agent(val successor: Option[Handler]) extends Handler {
  override def handleEvent(event: Event): Unit = {
    event match {
      case e if e.level < 2 => println("CS Agent Handled event: " + e.title)
      case e if e.level > 1 => {
        successor match {
          case Some(h: Handler) => h.handleEvent(e)
          case None => println("Agent: This event cannot be handled.")
        }
      }
    }
  }
}

class Supervisor(val successor: Option[Handler]) extends Handler {
  override def handleEvent(event: Event): Unit = {
    event match {
      case e if e.level < 3 => println("Supervisor handled event: " + e.title)
      case e if e.level > 2 => {
        successor match {
          case Some(h: Handler) => h.handleEvent(e)
          case None => println("Supervisor: This event cannot be handled")
        }
      }
    }
  }
}

class Boss(val successor: Option[Handler]) extends Handler {
  override def handleEvent(event: Event): Unit = {
    event match {
      case e if e.level < 4 => println("Boss handled event: " + e.title)
      case e if e.level > 3 => successor match {
        case Some(h: Handler) => h.handleEvent(e)
        case None => println("Boss: This event cannot be handled")
      }
    }
  }
}

object Main {
  def main(args: Array[String]) {
    val boss = new Boss(None)
    val supervisor = new Supervisor(Some(boss))
    val agent = new Agent(Some(supervisor))
    
    println("Passing events")
    val events = Array(
      Event(1, "Technical support"), 
      Event(2, "Billing query"),
      Event(1, "Product information query"), 
      Event(3, "Bug report"), 
      Event(5, "Police subpoena"), 
      Event(2, "Enterprise client request")
    )
    events foreach { e: Event =>
      agent.handleEvent(e)
    }
  }
}
```

In our code base: `getScope` will chain a sequence of scope, but who should be responsiblity to handle, we do not know in advance.

Chain-of-responsibility is widely used in front-end. Ex. UI components form a chain. We will iterate through the chain to discover who should be responsible to it.

<br/>


#### Command Pattern
***
Command Pattern encapsulates all information needed to perform an action or trigger an event at a later time. This information includes the method name, the object that owns the method and values for the method parameters.

In Scala, we can rely on by-name parameter to defer evaluation of any expression:

```scala
object Invoker {
  private var history: Seq[() => Unit] = Seq.empty

  def invoke(command: => Unit) { // by-name parameter
    command
    history :+= command _
  }
}

Invoker.invoke(println("foo"))

Invoker.invoke {
  println("bar 1")
  println("bar 2")
}
```

The idea is to store function and parameters as an object and to execute it later. 

```scala
abstract class CommandBase{
   def undo: Unit
   def redo: Unit
}

class MoveCommand extends CommandBase{
   override def undo: Unit = ???
   override def redo: Unit = ???
}

class DeleteCommand extends CommandBase{
   override def undo: Unit = ???
   override def redo: Unit = ???
}

val undoStack: List[CommandBase] = ???
val redoStack: List[CommandBase] = ???
```

<br/>

#### Interpreter Pattern
***
Interpreter Pattern is a design pattern that specifies how to evaluate sentences in a language. The basic idea is to have a class for each symbol (terminal or nonterminal) in a specialized computer language. The syntax tree of a sentence in the language is an instance of the composite pattern and is used to evaluate (interpret) the sentence for a client.

```scala
case class Context

abstract class AbstractExpression{
   def interpret(Context context): Unit
}

class TerminalExpression extends AbstractExpression{
   override def interpret(Context context): Unit =  ???
}

class NonTerminalExpression(expressions: AbstractExpression*) extends AbstractExpression{
   override def interpret(Context context): Unit =  ???
}

def main(): Unit = {
   val context = Context
   
   val expressionTree = List(
      new TerminalExpression, 
      new NonTerminalExpression(
          new NonTerminalExpression(new TerminalExpression),
          new TerminalExpression))
          
   expressionTree.foreach{ expression =>
      expression.interpret(context)
   }
}

```

Example in our code base: The whole Internal DSL uses Composite Pattern and Interpreter Pattern, like ModelingSyntax.scala

```scala
def doubleMetricOf(const: Double): Exp[Metric[Double]]

def floor[T <: ModelTypes](metric: Exp[Metric[T]])

class ModelingArithmeticOps[T <: ModelTypes : TypeTag](metric: Exp[Metric[T]]) {
   def +[F <: ModelTypes](addend: Exp[Metric[F]]): Exp[Metric[T]]
}

// floor(doubleMetricOf(1.5D)) + doubleMetricOf(2.0D) is of type Exp[Metric[Double]]



  /**
   * Prepare the data structures required to evaluate the model.
   *
   * @return PerThreadInitialization case class (see above)
   */
  def interpretExpTree: ExpTreeInterpretation = {

    val ExpTreeInitialization(expTree, substitutions) = initializeExpTree

    // invoke the interpreter to get the 'prepared calculation' using the input calculations we just made
    val prepared = Interpreter.interpretPlan(expTree, substitutions, PrepareModel.evaluator, new ContextVars {}, Seq.empty)
 ...
 
  }

```

<br/>

### Design Patterns - Workshop Aug 23
***
Today we will cover three design patterns.

**Creational patterns** are design patterns that deal with object creation mechanisms, trying to create objects in a manner suitable to the situation. Examples: 

* **Lazy initialization**: Tactic of delaying the creation of an object, the calculation of a value, or some other expensive process until the first time it is needed. 
* **Singleton**: Ensure a class has only one instance, and provide a global point of access to it.

**Structural patterns**: (definition already covered [here](#structuralPatterns))

* **Proxy**: Provide a placeholder for another object to control access to it.

<br />

#### Lazy initialization
***
**Lazy initialization** is the tactic of delaying the creation of an object, the calculation of a value, or some other expensive process until the first time it is needed.


In a software design pattern view, lazy initialization is often used together with a factory method pattern, so called the "lazy factory". This combines three ideas:

* Using a factory method to get instances of a class (**factory method pattern**)
* Store the instances in a map, so you get the same instance the next time you ask for an instance with same parameter (**multiton pattern**)
* Using lazy initialization to instantiate the object the first time it is requested (**lazy initialization pattern**)

In scala, the support for lazy value is built-in: 

```scala
class Example {
  lazy val x = "Value";
}
```

Above code is compiled to the code equivalent to the following java code:

```java
public class Example {

  private String x;
  private volatile boolean bitmap$0;

  public String x() {
    if(this.bitmap$0 == true) {
      return this.x;
    } else {
      return x$lzycompute();
    }
  }

  private String x$lzycompute() {
    synchronized(this) {
      if(this.bitmap$0 != true) {
        this.x = "Value";
        this.bitmap$0 = true;
      }
      return this.x;
    }
  }
}
```

However, using lazy val could be troublesome. 

##### Scenario1: Potential dead lock when accessing lazy vals

```scala
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent._
import scala.concurrent.duration._

object A {
  lazy val base = 42
  lazy val start = B.step
}

object B {
  lazy val step = A.base
}

object Scenario1 {
  def run = {
    val result = Future.sequence(Seq(
      Future { A.start },                        // (1)
      Future { B.step }                          // (2)
    ))
    Await.result(result, 1.minute)
  }
}
```

As the following figure illustrates:

![Lazy val dead lock](lazy-val-dead-lock.png)


The `A.start` val depends on `B.step` which in turn depends again on `A.base`. Although there is no cyclic relation here, running this code can lead to a **deadlock**:

```bash
scala> :paste
...
scala> Scenario1.run
java.util.concurrent.TimeoutException: Futures timed out after [1 minute]
  at scala.concurrent.impl.Promise$DefaultPromise.ready(Promise.scala:219)
  at scala.concurrent.impl.Promise$DefaultPromise.result(Promise.scala:223)
  at scala.concurrent.Await$$anonfun$result$1.apply(package.scala:190)
  at scala.concurrent.BlockContext$DefaultBlockContext$.blockOn(BlockContext.scala:53)
  at scala.concurrent.Await$.result(package.scala:190)
  at Scenario2$.run(<console>:30)
  ... 32 elided
```

Reason: when scala `lazy val` is compiled to java code, **synchronized** block is used, which will lock the current object.

- Thread 1 is holding A at `(1)` while try to access B; 
- Thread 2 is holding B at `(2)` while try to access A.

Thus deadlock occurs. 

##### Scenario 2: Deadlock in combination with synchronization

``` scala
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent._
import scala.concurrent.duration._

trait Compute {
  def compute: Future[Int] =
    Future(this.synchronized { 21 + 21 })        // (1)
}

object Scenario2 extends Compute {
  def run: Unit = {
    lazy val someVal: Int =
      Await.result(compute, 1.minute)            // (2)
    println(someVal)
  }
}
```

After execute above code in scala, we will get the following error

``` bash
scala> :paste
...
scala> Scenario2.run
java.util.concurrent.TimeoutException: Futures timed out after [1 minute]
  at scala.concurrent.impl.Promise$DefaultPromise.ready(Promise.scala:219)
  at scala.concurrent.impl.Promise$DefaultPromise.result(Promise.scala:223)
  at scala.concurrent.Await$$anonfun$result$1.apply(package.scala:190)
  at scala.concurrent.BlockContext$DefaultBlockContext$.blockOn(BlockContext.scala:53)
  at scala.concurrent.Await$.result(package.scala:190)
  at Scenario2$.someVal$lzycompute$1(<console>:30)
  at Scenario2$.someVal$1(<console>:29)
  at Scenario2$.run(<console>:31)
  ... 32 elided

```

Reason: Same as in scenario 1, when **synchronized** block is used, current object will be locked to prevent other thread from access it.

When call `println`, lazy initialization of someVal is triggered, and lock the current object; However, in order to calculate the result of `compute`, the current object also needs to be locked. Thus deadlock occurs.

<br />

#### Singleton
***

The singleton pattern is a design pattern that restricts the instantiation of a class to one object.

In java, initialization of singleton use eager initialization:

```java
/* Singleton implementation without lazy initialisation */
public class Singleton {
    // Eager initialization 
    private static final Singleton INSTANCE = new Singleton();
    private Singleton() { /* Initialisation code */ }
    
    public static Singleton getInstance() {
        return INSTANCE;
    }
}
```

In scala, it's as easy as create an object

```scala
trait Service

object Singleton extend Service{
 ...
}

functionTakesInAService(service = Singleton)

```

How to extend the functionality of a singleton object in scala:

```scala
object X{
  def x = 5
}

object Y{
  import X._
  val y = x
}
```

But most of the time, it's better to composite an object instead of extend an object. 
<br />

#### Proxy
***

A **proxy** is a wrapper or agent object that is being called by the client to access the real serving object behind the scenes. Use of the proxy can simply be forwarding to the real object, or can provide additional logic. 
Use it for lazy initialization, performance improvement by caching the object and controlling access to the client/caller

```scala

trait Proxy 

trait Service

trait ImageProvider {
  val fileUrl: URL
  def image: ImageIcon
}

private class RealImageService(val fileUrl: URL) extends Service with ImageProvider {
  private val imageIcon = new ImageIcon(fileUrl)
  def image: ImageIcon = imageIcon
}

class ImageServiceProxy(imageFileUrl: URL) extends Proxy with ImageProvider {
  val fileUrl = imageFileUrl
  private lazy val imageService = new RealImageService(imageFileUrl)
  def image: ImageIcon = imageService.image
  override def toString = "ImageServiceProxy for: " + imageFileUrl.toString()
}
```