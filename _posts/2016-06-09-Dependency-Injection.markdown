---
layout: post
title:  "依赖注入(Dependency Injection)"
date:   2016-06-09 22:23:32 -0700
categories: Design-Patterns
---
* This will become a table of contents (this text will be scraped).
{:toc}

引子
---
第一次接触算法是大二的时候，总想写点什么，却总因为这样或是那样的原因搁置。这次我总算是痛下决心，决定认真地总结一下这段时间学习算法的种种心得。那么，一千零一夜的第一夜就从*依赖注入（Dependency Injection）*开始讲起吧。

<a name="originalCode"></a>
废话不多说，先上段Java代码：

~~~java
// An example without dependency injection
public class Client {
    // Internal reference to the service used by this client
    private Service service;

    // Constructor
    Client() {
        // Specify a specific implementation in the constructor instead of using dependency injection
        service = new ServiceExample();
    }

    // Method within this client that uses the services
    public String greet() {
        return "Hello " + service.getName();
    }
    
    // Below code shows how we can call greet function
    public static void main(){
        Client clientInstance = new Client();
        clientInstance.greet();
    }
}
~~~

这段代码简单改自wiki[^codeWithoutDIExample]， 构造了一个简单的客户端`Client类`， 隶属于它的只有hardcode（硬编）进去的固定服务`Service类`。 我们简单的想象一下， 好不容易用手机下载一个客户端。打开了，里面就只用一个固定功能的样子。 更可怕的事情是， 如果以后想加入新功能，`功能A`。我们就只有两个选择：

  1. 改`Client类`的代码 
  2. 写新的类`clientWithFunctionA` 然后通过继承`Client类`来覆写`Service`。

相比2，选项1还好那么一点， 至少可以使用`工厂模式`来区分不同的服务。 但缺点也很明显：每加一个新的服务， 都需要添加新的规则到`switch`。 最后`Client.java`文件会无休止地变得越来越大。就像下面的代码一样：

~~~java
public class Client {
...
    // Factory Design Pattern
    Client(String service){
        swith(service){
            case "FunctionA":
                this.service = new ServiceExampleWithFunctionA();
                break:
            default: 
                this.service = new ServiceExample();
        }
    }
    
...
    
    public static void main(){
        Client clientInstance = new Client();
        clientInstance.greet(); // using ServiceExample 
        clientInstance = new Client("FunctionA");
        clientInstance.greet(); // using ServiceExampleWithFunctionA
    }
}
~~~

如果不幸选择了2的话，那么以后代码的维系只能自求多福了。 首先，因为`Client类`不同于抽象类或者是接口，因为继承了之后任何任何与`Service类`相关的成员方法都需要override(覆写)，无形中增加了程序的不稳定性。 再者，如果在`功能A`上有新加了`功能B`， 那么只能继续构造新的类：`ClientWithFunctionAWithFunctionB`。维护代码会变得越发困难。比如以下代码：

~~~java
public class ClientWithFunctionA extends Client{
    ClientWithFunctionA(){
        this.service = new ServiceExampleWithFunctionA()
    }
    
    public static void main(){
        Client clientInstance = new Client();
        clientInstance.greet(); // using ServiceExample 
        clientInstance = new ClientWithFunctionA();
        clientInstance.greet(); // using ServiceExampleWithFunctionA
    }
}

public class ClientWithFunctionAWithFunctionB extends Client{
...
}
~~~


依赖注入--基础
----------------------------------
试想一下，如果`Client的实例`能在运行的时候任意改变`service`的Implementation(实现方法)，那么我们前面所遇到的问题就迎刃而解了。

<a name="DICode"></a>
仔细观察`Service类的一个实例`是怎么样在运行的时候被Inject(注入)的，并且它是如何实现不同`Service`之间切换的。

~~~java
public class Client {
...
    
    // Inject service at runtime
    Client(Service service){
        this.service = service;
    }

...
    
    public static void main(){
        Client clientInstance = new Client(new ServiceExample());
        clientInstance.greet(); // using ServiceExample 
        clientInstance = new Client(new ServiceExampleWithFunctionA());
        clientInstance.greet(); // using ServiceExampleWithFunctionA
    }
}
~~~

看到这里，在屏幕前的你肯定会说：“切。 看起来没什么了不起的嘛，跟前面的方法1有啥区别。方法1的工厂模式输入一个`String类`， 你这里偷懒直接给了个`Service类`罢了。” 面对这样疑问，其实我想说，我最开始也抱有类似的看法，但是随着代码库的扩大，以及功能的增多，我慢慢改变了我的看法。

先说说依赖注入的好处吧:

   1. `Client类`在初始化的时候并不用考虑`Service类`具体的实现是怎么样的，甚至也不用关心它是否初始化过。如果有一天`Service类`初始化出现了问题，你不会去`Client类`里面找Bug， 因为`Client类`根本就不负责初始化。   
   2. 由于`Client类`和`Service类`的依赖关系在Compile（编译）时是分离关系，添加新的`Service子类`并不需要修改`Client类`， 所以，`Client类`的代码行数不会无限制的疯涨下去。
   3. 单元测试变得无比容易

1，2 两点前面举例说明的已经很清晰了，下面一个章节就重点说说3。


依赖注入与单元测试 （Unit Test）
-----------------------------------------------------
在给出的[第一个例子](#originalCode)里面，我们申明了`service = new ServiceExample();`。 如果我们要对`Client类`进行测试的话，势必要对`Service进行初始化`。 可是问题来了， 这里`service` 被强行定义成 `ServiceExample()`， 如果`ServiceExample类`链接了数据库并且执行了几条SQL语句呢？ 单元测试就不可能了：因为在测试前需要建立好数据库的链接并且得保证数据库链接不出问题。最重要的是，加入了数据库的测试不能称得上是单元测试（Unit Test），应该归类成集成测试（Integration Test），因为测试的已经不再是小小一个零部件，而是整个模块。

~~~java
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class ClientJIntegrationTest{
    @Before
    public void setupDBEnvironment(){
        // set up DB connection
    }
    
    @Test
    public void test(){
        Client clientInstance = new Client();
        assert(clientInstance.greet() == "Hello " + new ServiceExample().getName());
    }
    
    @After
    public void shutdownDBEnvironment(){
        // close DB connection
    }
}
~~~

以上代码最大的问题就在于，万一哪一天`Client类`的`service成员`不再是`ServiceExample`了，那么我们必须得修改以上`ClientJIntegrationTest`集成测试。那么有没有一劳永逸的办法？

仔细观察[前面那段](#DICode)关于*依赖注入*的代码,不难得出以下答案：

~~~java
import org.junit.Test;

public class ClientJUnitTest{

    public class simpleService implements Service{
        @override
        public String getName(){
            return "simpleService";
        }
    }
    
    @Test
    public void test(){
        Client clientInstance = new Client(new simpleService());
        assert(clientInstance.greet() == "Hello " + new simpleService().getName());
    }
}
~~~

怎么样，以后无论再怎么增加新的`Service子类`，是不是都不用担心会影响到`Client类`的单元测试了？



依赖注入--进阶
---------------------------------

### 依赖注入的三种方式
依赖注入有三种注入依赖关系的方式：*Constructor 注入*， *Setter 注入* 和 *Interface 注入*。

#### 1. Constructor 注入
*Constructor注入*是我现在的单位所使用的方法：主要通过在类初始化之前就提前注入依赖关系来实现。[前面](#DICode)的所展示的例子都是通过*Constructor注入*实现的。这样做的好处是: 

* 注入的service在使用的时候永远不会是`null`
* service不会有被改动的风险
* 如果所有依赖关系都通过注入的方式可以保证*线程安全*（Thread Safe），因为所有注入的服务都可以声明成为常量*final*或者c/c++中的*const*）。

缺点就是：依赖关系一旦注入就无法更改，将导致依赖关系变得不那么灵活。

~~~java
// Constructor
Client(Service service) {
    // Save the reference to the passed-in service inside this client
    this.service = service;
}
~~~

#### 2. Setter 注入
*Setter 注入*是通过setter的方式注入和改变依赖关系。

优点： 依赖关系变得灵活

缺点： 1）线程安全无法像1那样轻轻松松保证。 2）无法强制保证调用这个方法的时候，service不是*null*， 换言之，什么时候初始化是个问题。

~~~java
// Setter method
public void setService(Service service) {
    // Save the reference to the passed-in service inside this client
    this.service = service;
}
~~~

但是缺点2可以通过以下方法避免, 虽然每次调用service里面的服务时会变得比较麻烦就是了 >_<

~~~java
// Check the service references of this client
private void validateState() {
    if (service == null) {
        throw new IllegalStateException("service must not be null");
    }
}

// Method that uses the service references
public void doSomething() {
    validateState();
    service.doSomething();
}
~~~

#### 3. Interface 注入
*Interface 注入*比*Setter 注入*强的地方在于：`ServiceSetter接口`可以成为一个`注入子（Injector）`。当使用`注入子`的时候，调用它的函数可以不知道原始`类`具体是什么：

~~~java
// Service setter interface.
public interface ServiceSetter {
    public void setService(Service service);
}

// Client class
public class Client implements ServiceSetter {
    // Internal reference to the service used by this client.
    private Service service;

    // Set the service that this client is to use.
    @Override
    public void setService(Service service) {
        this.service = service;
    }
}
~~~

下面展示如何使用注入子：

~~~java
public class ClientA implements ServiceSetter {
...
}

public class ClientB implements ServiceSetter {
...
}

public void test(){
    // Initilization
    ServiceSetter injector = null;
    injector = new ClientA();
    injector.setService(new ExampleServiceA);
    
    // Cast to ClientA to use it
    ClientA clientA = (ClientA) injector;
    clientA.doSomething();
    
    // Switch to different service
    injector = new ClientB();
    injector.setService(new ExampleServiceB);
    
    // Cast to ClientB to use it
    ClientB clientB = (ClientB) injector;
    clientB.doOtherThing();
}
~~~


### 注入子(Injector)



### 支持依赖注入的框架(Framework): Guice
像前面的例子一样，自己一点一点地写依赖注入的程序是个恼人的活：不仅写起来耗时耗力，而且由于是运行的时候动态绑定服务，容易错误频发。所以企业级程序一般是使用支持依赖注入的框架来实现其功能的。

市面上支持依赖注入的框架有很多：Spring, Guice, Play framework, Salta, Glassfish HK2, Managed Extensibility Framework (MEF) 。这里介绍一个轻量级支持Java 6的框架Guice。以下代码摘自Guice的github介绍页[^GuiceDI]。

以下代码声明了一个账单服务的类：第一步，将需要注入的方法加上`@Inject`记号：

~~~java
class BillingService {
  private final CreditCardProcessor processor;
  private final TransactionLog transactionLog;

  @Inject
  BillingService(CreditCardProcessor processor, 
      TransactionLog transactionLog) {
    this.processor = processor;
    this.transactionLog = transactionLog;
  }

  public Receipt chargeOrder(PizzaOrder order, CreditCard creditCard) {
    ...
  }
}
~~~


第二步，注入参数设定。告诉Guice，当需要某种服务的时候，该绑定何种对应的类。 这里需要定义一个类继承`AbstractModule类`，并覆写`configure方法`。 下面第一个`bind`的的意思是：当需要注入`TransactionLog接口`时，使用`DatabaseTransactionLog类`。第二个`bind`意义类似：

~~~java
public class BillingModule extends AbstractModule {
  @Override 
  protected void configure() {

     /*
      * This tells Guice that whenever it sees a dependency on a TransactionLog,
      * it should satisfy the dependency using a DatabaseTransactionLog.
      */
    bind(TransactionLog.class).to(DatabaseTransactionLog.class);

     /*
      * Similarly, this binding tells Guice that when CreditCardProcessor is used in
      * a dependency, that should be satisfied with a PaypalCreditCardProcessor.
      */
    bind(CreditCardProcessor.class).to(PaypalCreditCardProcessor.class);
  }
}
~~~


第三步，在`main函数`里用前面声明好的注入参数设定`BillingModulel类`，调用`Guice库`来生成注入子。并向注入子请求`BillingService`对应的类。这里的类不能使用 `new BillingService()`来声明， 因为`BillingService`需要使用依赖注入的方式构建：

~~~java
 public static void main(String[] args) {
    /*
     * Guice.createInjector() takes your Modules, and returns a new Injector
     * instance. Most applications will call this method exactly once, in their
     * main() method.
     */
    Injector injector = Guice.createInjector(new BillingModule());

    /*
     * Now that we've got the injector, we can build objects.
     */
    BillingService billingService = injector.getInstance(BillingService.class);
    ...
  }
~~~

### Guice框架下的单元测试
Guice 4.0 之后有两种方法对依赖注入类进行单元测试[^GuiceUnitTest], 因为本文主要目的不是介绍单元测试，所以以下的代码仅供参考:

方法1：

~~~java
public class BillingServiceJUnitTest {
  private TransactionLog logMock;
  private CreditCardProcessor processorMock;

  // BillingService depends on TransactionLog and CreditCardProcessor.
  @Inject 
  private BillingService billingService;

  @Before 
  public void setUp() {
    logMock = ...;
    processorMock = ...;
    Guice.createInjector(getTestModule()).injectMembers(this);
  }

  private Module getTestModule() {
    return new AbstractModule() {
      @Override protected void configure() {
        bind(TransactionLog.class).toInstance(logMock);
        bind(CreditCardProcessor.class).toInstance(processorMock);
      }
    };
  }
  

  @Test 
  public void testBehavior() {
    ...
  }
}
~~~

方法2： 使用`BoundFieldModule`

~~~java
public class BillingServiceJUnitTest {

  // bind(TransactionLog.class).toInstance(logMock)
  @Bind 
  private TransactionLog logMock;
  
  // bind(CreditCardProcessor.class).toInstance(processorMock);
  @Bind
  private CreditCardProcessor processorMock;

  @Inject 
  private BillingService billingService;

  @Before 
  public void setUp() {
    logMock = ...;
    processorMock = ...;
    Guice.createInjector(BoundFieldModule.of(this)).injectMembers(this);
  }
  
...
}
~~~



### 依赖注入的缺点
依赖注入不是**万金油**。 前面说了那么多优点，下面说一下它的缺点[^drawbacks]：

* 适用于**大型**或相对**复杂**的项目，小的项目只会变成杀鸡用牛刀，增加额外的复杂程度
* 即使是**大型**项目，因为依赖关系是注入的原因，理解整个程序的流向会变得困难。例如前面的例子里注入的Service -- 在现实生活中，举网络后端为例 -- 很有可能会因为不同的请求注入不同的服务，具体注入什么很有可能在XML文件里面定义。 初学者往往会对整个流程感到困惑
* 依赖注入解决的问题是如何实现**控制反转**（[Inversion of Control](https://en.wikipedia.org/wiki/Inversion_of_control)），比如[第一个例子](#originalCode)里面提到的：`Client`依赖`Service`的问题。如果，涉及的项目并不需要解决**控制反转**的问题，那么强行使用依赖注入会让本来简单的问题复杂化
* 依赖注入是在程序运行的时候动态建立依赖关系，所以很难测试到运行时发生的错误，即**Runtime Error**。 但是这种Error在会在不使用依赖注入的时候被编译器（Compiler） 捕获。但是，现在很多主流的框架（Framework）也支持编译的时候检查错误了。所以**Runtime Error**问题在框架面前也没什么大不了的了。

[^codeWithoutDIExample]: 代码（[出处](https://en.wikipedia.org/wiki/Dependency_injection)）：展示了如果不使用DI的话，代码看起来会怎么样。这段代码的主要问题在于：Client类和Service类的关系是hardcode（硬编）上去的，导致Client只能提供固定的功能。

[^GuiceDI]: 代码（[出处](https://github.com/google/guice/wiki/GettingStarted)）:介绍了Guice框架下是如何智能地使用依赖注入帮助程序员免除繁琐的工序的。

[^GuiceUnitTest]: 代码（[出处](https://github.com/google/guice/wiki/BoundFields)) ：结合了前面Guice简介里面介绍的例子，略做改动。

[^drawbacks]: 参考自MSDN文档，[来源](https://msdn.microsoft.com/en-us/library/dn178469(v=pandp.30).aspx)。总结了依赖注入的优缺点，其中对缺点的描述十分深刻，让人产生共鸣。