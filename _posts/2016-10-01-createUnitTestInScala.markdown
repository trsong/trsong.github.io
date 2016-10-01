---
layout: post
title:  "Create Unit Tests in Scala"
date:   2016-10-01 10:00:32 -0700
categories: Platform
---
* This will become a table of contents (this text will be scraped).
{:toc}


### Create Unit Tests in Scala
***

```scala
class SomeServiceSpec extend TestingFrameWorkUtility with MyTestUntilHelper {

/* define some function */
def testFunc(num: Int) = {
   num == 2
}

   describe("Test Suites Description"){
      it("Test Case1: description1"){
          val myValue = 1
          testFunc(myValue) should equals false
      }
      
      it("Test Case2: description2"){
          val myValue = 2
          testFunc(myValue) should equals true
      }
   }
   
   describe("Huge serive1"){
     describe("Component service"){
        describe("SubComponent Service"){
           it("test case"){
               ???
           }
        }
     }
     
     it("test"){
         ???
     }
   }

}



```

### How to run tests w/ sbt:
***

``` bash
# run all tests under the project
./sbt "project data_designer_services" "test"  

# run a specific test from terminal
./sbt "project data_designer_services" "test-only visier.designer.dataBootstrap.services.SampleDataGenerationServiceSpec"
./sbt "project data_designer_services" "test-only *SampleDataGenerationServiceSpec"

# run test with a debug port
./sbt -Ddebugport=9999 "project data_designer_services" "test-only *SampleDataGenerationServiceSpec"

```

### How to mock a service
***

Class Definition

```scala
// SomeService.scala

trait TimeStamp{
   def timeStamp: Int 
}

class SomeService[T :< TimeStamp](dbService: DBService[T]){
    def insert(entity: Entity[T]): Entity[T] = {
        dbService.insert(entity)
    }
    
    def update(entity: Entity[T]): Entity[T] = { 
        dbService.update(entity)
    }
    
    def delete(entity: Entity[T]): Unit = {
        dbService.delete(entity)
    }
    
    def get(timeStamp: Int): Entity[T] = {
        dbService.get(timeStamp)
    }
    
    def mainWorkFlow: Unit = {
        val randomEntity: Entity[T] = Entity[T].generateRandomEntity
        insert(randomEntity)
    }
}

```

Unit Test

```scala
// SomeServiceSpec.scala

class SomeServiceSpec extends FunSpec with Matchers with MockitoSugar{
     val mockedDBService = mock[DBService]
     
     class Container(value: Int) extends TimeStamp{
         val timeStamp = Random.nextInt
     }
     
     describe("SomeService"){
         describe("When the object has type Integer"){
             val service = new SomeService[IntegerWithTimeStamp](mockedDBServicece)
             
             val lst = mutable.List.empty[Container]
             
             def mockedInsert(entity: Entity[Container]): Entity[Container] ={
                lst += entity
                lst.last
             }
             
             when(mockedDBService.insert).thenReturn(mockedInsert)
            
             it("should insert the object correctly"){
                 service.insert(new Container(42))
                 lst.length should equal 1
                 lst.last should equal 42
             }
         }
     }
}
```


Run under stb:

```bash
# assume SomeServiceSpec is under MyProject/...../test/.... 
./sbt "project MyProject" "test:only *SomeServiceSpec"
```


### What is `case class`
****

Every `case class` is normal `class`. But not every `class` is `case class`.

`case class` has something extra.

#### Has `apply` function on default

```scala
// Example 1
class Container(value: Int)

val container = new Container(42)

// Example 2
class ContainerWithApply(value: Int){
    def apply(value: Int): ContainerWithApply = new ContainerWithApply(value)
}

val containerWithApply = ContainerWithApply(42) 
// Is equivalent to call ContainerWithApply.apply(42)

// Example 3
case class SpecialContainer(value: Int)

val specialContainer = SpecialContainer(42)

```

#### Has default `equal` function

```scala
case class Postion(x: Double, y:Double)

Position(1.0D, 2.0D) == Position(2.0D, 2.0D) // return false
```

#### Good to know feature: has default hash code

#### Allow pattern matching, because it implements `equal` on default

```scala
case class Container(value: Int)
case class Student(id: Int)
case class Position(x: Double, y:Double)

val myVal = Student(42)
val myVal2 = Position(1.0D, 2.0D)

var value: Any = myVal

def test(value: Any): Unit = {
     value match {
      case Student(i: Int) => print(i)
      case Postion(_, _) => print
      case Container(x) => print(x)
      case value: String => print(value)
      case _ => 
   }
}
``` 

### what is `object`
***

```scala
```