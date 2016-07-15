### Design Patterns - Structural Patterns
<hr/>
Structural design patterns are design patterns that ease the design by identifying a simple way to realize **relationships between entities**.

Examples of Structural Patterns includes:

* **Adapter pattern**: 'adapts' one interface for a class into one that a client expects
* **Bridge pattern**: decouple an abstraction from its implementation so that the two can vary independently
* **Composite pattern**: a tree structure of objects where every object has the same interface
* **Decorator pattern**: add additional functionality to a class at runtime where subclassing would result in an exponential rise of new classes
<br/>

#### Adapter/Wrapper/Translator
<hr/>
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
<hr/>
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
<hr/>
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
<hr/>
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
