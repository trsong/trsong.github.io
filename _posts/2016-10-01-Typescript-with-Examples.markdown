---
layout: post
title:  " Typescript with Examples"
date:   2016-10-01 22:23:32 -0700
categories: Platform
---
* This will become a table of contents (this text will be scraped).
{:toc}

## Typescript with Examples
<hr />
<br />

### General Usage
<hr />

```bash
npm install -g typescript    #install
tsc greeter.ts    # compile to javascript file "greeter.js"
node greeter.js   # execute the output js file 
```
<br />

#### Typescript REPL
<hr />

```bash
# Install
npm install -g ts-node
npm install -g typescript

# Execute a script as you world normally with `node`.
ts-node script.ts

# Starts the TypeScript REPL.
ts-node

# Execute code with TypeScript.
ts-node -e 'console.log("Hello, world!")'

# Execute, and print, code with TypeScript.
ts-node -p '"Hello, world!"'

```

### Quick Start
<hr />

greeter.ts

```typescript
class Student {
    fullName: string;
    constructor(public firstName, public middleInitial, public lastName) {
        this.fullName = firstName + " " + middleInitial + " " + lastName;
    }
}

interface Person {
    firstName: string;
    lastName: string;
}

function greeter(person : Person) {
    return "Hello, " + person.firstName + " " + person.lastName;
}

var user = new Student("Jane", "M.", "User");

document.body.innerHTML = greeter(user);
```

greeter.html

```html
<!DOCTYPE html>
<html>
    <head><title>TypeScript Greeter</title></head>
    <body>
        <script src="greeter.js"></script>
    </body>
</html>
```

<br />

###Classes
<hr />
<br />

#### Inheritance
<hr />

```typescript
class Animal {
    name: string;
    constructor(theName: string) { this.name = theName; }
    move(distanceInMeters: number = 0) {
        console.log(`${this.name} moved ${distanceInMeters}m.`);
    }
}

class Snake extends Animal {
    constructor(name: string) { super(name); }
    move(distanceInMeters = 5) {
        console.log("Slithering...");
        super.move(distanceInMeters);
    }
}

class Horse extends Animal {
    constructor(name: string) { super(name); }
    move(distanceInMeters = 45) {
        console.log("Galloping...");
        super.move(distanceInMeters);
    }
}

let sam = new Snake("Sammy the Python");
let tom: Animal = new Horse("Tommy the Palomino");

sam.move();
tom.move(34);
```

<br />

#### Public, private, and protected modifiers
<hr />
**Public** by default.

If wish to define as **private**, one need to define explicitly.

```typescript
class Animal {
    public name: string;
    public constructor(theName: string) { this.name = theName; }
    public move(distanceInMeters: number) {
        console.log(`${this.name} moved ${distanceInMeters}m.`);
    }
}
```
The above code needs to change to 

```typescript
class Animal {
    private name: string;
    constructor(theName: string) { this.name = theName; }
}

new Animal("Cat").name; // Error: 'name' is private;
```

**Protected** can only be accessed by instances of deriving classes.

```typescript
class Person {
    protected name: string;
    constructor(name: string) { this.name = name; }
}

class Employee extends Person {
    private department: string;

    constructor(name: string, department: string) {
        super(name);
        this.department = department;
    }

    public getElevatorPitch() {
        return `Hello, my name is ${this.name} and I work in ${this.department}.`;
    }
}

let howard = new Employee("Howard", "Sales");
console.log(howard.getElevatorPitch());
console.log(howard.name); // error
```

**Parameter properties** make it possible to have declaration and assignment in one place inside constructor.

```typescript
class Animal {
    constructor(private name: string) { }
    move(distanceInMeters: number) {
        console.log(`${this.name} moved ${distanceInMeters}m.`);
    }
}
```

**Accessors**, have control over getters and setters

```typescript
let passcode = "secret passcode";

class Employee {
    private _fullName: string;

    get fullName(): string {
        return this._fullName;
    }

    set fullName(newName: string) {
        if (passcode && passcode == "secret passcode") {
            this._fullName = newName;
        }
        else {
            console.log("Error: Unauthorized update of employee!");
        }
    }
}

let employee = new Employee();
employee.fullName = "Bob Smith";    // call the setter
if (employee.fullName) {    // call the getter
    console.log(employee.fullName);
}
```

**Static properties**

In the following example, we call the static properties by using `Grid.origin`

```typescript
class Grid {
    static origin = {x: 0, y: 0};
    calculateDistanceFromOrigin(point: {x: number; y: number;}) {
        let xDist = (point.x - Grid.origin.x);
        let yDist = (point.y - Grid.origin.y);
        return Math.sqrt(xDist * xDist + yDist * yDist) / this.scale;
    }
    constructor (public scale: number) { }
}

let grid1 = new Grid(1.0);  // 1x scale
let grid2 = new Grid(5.0);  // 5x scale

console.log(grid1.calculateDistanceFromOrigin({x: 10, y: 10}));
console.log(grid2.calculateDistanceFromOrigin({x: 10, y: 10}));
```

**Abstract Classes**

Abstract method in the abstract class need to use key word **abstract**.

```typescript
abstract class Animal {
    abstract makeSound(): void;
    move(): void {
        console.log("roaming the earth...");
    }
}
```

Use a class as an **interface**

```typescript
class Point {
    x: number;
    y: number;
}

interface Point3d extends Point {
    z: number;
}

let point3d: Point3d = {x: 1, y: 2, z: 3};
```
<br />

### Interface 
<hr />
Simple example:

```typescript
function printLabel(labelledObj: { label: string }) {
    console.log(labelledObj.label);
}

let myObj = {size: 10, label: "Size 10 Object"};
printLabel(myObj);
```

**Optional Properties**:

```typescript
interface SquareConfig {
    color?: string;
    width?: number;
    [propName: string]: any;
}

function createSquare(config: SquareConfig): { color: string; area: number } {
    let newSquare = {color: "white", area: 100};
    if (config.color) {
        newSquare.color = config.collor;
    }
    if (config.width) {
        newSquare.area = config.width * config.width;
    }
    return newSquare;
}

let squareOptions1 = { colour: "red", width: 100 };
let squareOptions2 = { colour: "red"};
let squareOptions3 = { width: 100 };
let squareOptions4 = { anyOtherField1: [1, 2, 3], anyOtherField2: "abc"};

createSquare(squareOptions1);
createSquare(squareOptions2); // Error: Property 'collor' does not exist on type 'SquareConfig'
createSquare(squareOptions3); // Error: Property 'width' does not exist on type 'SquareConfig'
createSquare(squareOptions4); // Similar compilication err 
```

**Function Types**

```typescript
interface SearchFunc {
    (source: string, subString: string): boolean;
}

let mySearch: SearchFunc;
mySearch = function(src: string, sub: string): boolean {
    let result = src.search(sub);
    if (result == -1) {
        return false;
    }
    else {
        return true;
    }
}
```

**Indexable Types**

```typescript
interface StringArray {
    [index: number]: string;
}

let myArray: StringArray;
myArray = ["Bob", "Fred"];

let myStr: string = myArray[0];
```

**Class Types**, Implementing an interface

```typescript
interface ClockInterface {
    currentTime: Date;
}

class Clock implements ClockInterface {
    currentTime: Date;
    constructor(h: number, m: number) { }
}

```

Interface for constructors

```typescript
interface ClockConstructor {
    new (hour: number, minute: number);
}

class Clock implements ClockConstructor {
    currentTime: Date;
    constructor(h: number, m: number) { }
}
``` 

**Hybrid Types**

```typescript
interface Counter {
    (start: number): string;
    interval: number;
    reset(): void;
}

function getCounter(): Counter {
    let counter = <Counter>function (start: number) { };
    counter.interval = 123;
    counter.reset = function () { };
    return counter;
}

let c = getCounter();
c(10);
c.reset();
c.interval = 5.0;
```


<br />

### Enums
<hr />

```typescript
enum Direction {
    Up = 1,
    Down,     // 2
    Left,     // 3
    Right     // 4
}

```

More examples:

```typescript
enum FileAccess {
    // constant members
    None,
    Read    = 1 << 1,
    Write   = 1 << 2,
    ReadWrite  = Read | Write,
    // computed member
    G = "123".length
}

enum Enum {
    A
}
let a = Enum.A;
let nameOfA = Enum[Enum.A]; // "A"
```


<br />