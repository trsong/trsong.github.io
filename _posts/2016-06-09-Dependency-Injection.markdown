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


依赖注入（Dependency Injection）基础
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


依赖注入（Dependency Injection）与单元测试 （Unit Test）
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



依赖注入（Dependency Injection）进阶
---------------------------------

### 依赖注入的方式

### 注入子(Injector)

### 支持依赖注入的框架(Framework)


[^codeWithoutDIExample]: 代码（[出处](https://en.wikipedia.org/wiki/Dependency_injection)）：展示了如果不使用DI的话，代码看起来会怎么样。这段代码的主要问题在于：Client类和Service类的关系是hardcode（硬编）上去的，导致Client只能提供固定的功能。