---
layout: post
title:  "Essential Of Swift"
date:   2015-10-08 22:36:32 -0700
categories: Swift Learning
---
* This will become a table of contents (this text will be scraped).
{:toc}

### Essential Of Swift 

#### Basic Types

```swift
var myVariable = 42
let myConstant = 50
let explicitDouble: Double = 42.25
// Note: option-click on variable to see its type in Xcode

let stringInterpolation = “Here’s an example: my variable equlas \(myVariable)”
let optionInt: Int? = 9
let optionNil: Int? = nil
let getOptionInt: Int = optionInt!

let convertionNil = Int(“This cannot be convert to any valid number, so result will be a nil.”)
let convertionNumber = Int(“42”)

let stringList = [“a”, “b”, “c”]
let emptyStringList = [String]()

let implicitUnWrappedOptionInt: Int! = 42
print(implicitUnWrappedOptionInt + 5)
```

#### Control Flow

```swift
for elem in stringList {
    print(elem)
}

if let num = optionInt {
    print(num)
}

if let num = optionInt where num != 0, let num2 = convertionNumber {
    print(“\(num) + \(num2) = \(num + num2)”)  
}

let num2 = 3
switch num2{
case 1,2,3:
    print("One to three")
case let x where x % 2 == 0:
    print("even")
default:
    print("default value")
}

var accumulator = 0
for i in 0..<4 {
    accumulator += 1
}
print(accumulator) // 4
accumulator = 0
for _ in 0...5 {   // _, the underscore is a wild card 
    accumulator += 1
}
print(accumulator) // 6
```

#### Functions and Methods

```swift
func numberGenerator(seed: Int) -> Int {
    return 42
}

numberGenerator(123)

func moreParamsFunc(first: Int, second: Int, third: Int) -> Int {
    return first + second + third
}

moreParamsFunc(1, second: 2, third: 3)
```


#### Classes and Initializers

```swift
class MyClass{
    func printName() {
        print(“Hello World”)
    }
}

let myInstance = MyClass()
myInstance.printName()


class ClassInitExample{
    var arg1: Int
    let arg2: Int
    var arg3: Int = 3
    init(arg1: Int, arg2: Int, arg3: Int){
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
    }
    func printParam(){
        print("ClassInitExample: arg1== \(arg1), arg2== \(arg2), arg3== \(arg3)")
    }
}
let classInitExample = ClassInitExample(arg1: 1, arg2: 2, arg3: 4)
classInitExample.arg1
classInitExample.arg2
classInitExample.arg3
classInitExample.printParam()

class ClassInheritanceExample: ClassInitExample{
    let arg4: Int
    init(arg4: Int){
        self.arg4 = arg4
        super.init(arg1:1 , arg2:2, arg3:3)
    }
    override func printParam(){
        super.printParam()
        print("ClassInheritance: arg4== \(arg4)")
    }
}

let classInheritanceExample = ClassInheritanceExample(arg4: 4)
classInheritanceExample.printParam()
```

A designated initializer indicates that it’s one of the primary initializers for a class; any initializer within a class must ultimately call through to a designated initializer. A convenience initializer is a secondary initializer, which adds additional behavior or customization, but must eventually call through to a designated initializer.
A required keyword next to an initializer indicates that every subclass of the class that has that initializer must implement its own version of the initializer (if it implements any initializer).

```swift
let upCastExample: ClassInitExample = classInheritanceExample
if let downCastExample = upCastExample as? ClassInheritanceExample {
    print("Downcast succeed, result: ")
    downCastExample.printParam()
} else {
    print("Downcast failed")
}
```

#### Enumerations and Structures 

```swift
enum FourSeason: Int {
    case Winter, Spring, Summer, Fall
    func description() -> String {
        switch self {
        case .Winter:
            return "Winter"
        case .Spring:
            return "Spring"
        case .Summer:
            return "Summer"
        case .Fall:
            return "Fall"
        }
    }
}

let season = FourSeason.Spring
let swimSeasonOption = FourSeason(rawValue: 3)
season.description()
if let swimSeason = swimSeasonOption where swimSeason == FourSeason.Summer {
    swimSeason.description()
}

struct Container{
    var weightLimit: Double
    var name: String
    func checkSize() -> Bool {
        return weightLimit >= 0 && weightLimit <= 20.0
    }
}

let bag = Container(weightLimit: 19, name: "myBag")
var bag2 = bag
bag2.name = "bag2"
bag.checkSize()
print(bag2.name)
print(bag.name)
```

#### Protocol

```swift
protocol SimpleProtocol {
    var displayValue: Int { get }
}

class SimpleImpl: SimpleProtocol{
    var displayValue: Int = 20
    init(displayValue: Int){
        self.displayValue = displayValue
    }
}

var simpleExample: SimpleProtocol = SimpleImpl(displayValue: 42)
print(simpleExample.displayValue)
if let simpleImpl = simpleExample as? SimpleImpl{
    simpleImpl.displayValue = 10
    print(simpleImpl.displayValue)
}
print(simpleExample.displayValue)
```
