---
layout: post
title:  "依赖注入(Dependency Injection)"
date:   2016-06-09 22:23:32 -0700
categories: Design-Patterns
---
第一次接触算法是大二的时候，总想写点什么，却总因为这样或是那样的原因搁置。这次我总算是痛下决心，决定认真地总结一下这段时间学习算法的种种心得。那么，一千零一夜的第一夜就从*依赖注入（Dependency Injection）*开始讲起吧。

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
}
~~~
这段代码直接抄自wiki[^codeWithoutDIExample]， 构造了一个简单的客户端Client类， 隶属于它的只有hardcode（硬编）进去的固定服务Service类。 我们简单的想象一下， 好不容易用手机下载一个客户端。打开了，里面就只用一个固定功能的样子。 更可怕的事情是， 如果以后想加入新功能，功能A。我们就只有两个选择： 1） 改Client类的代码 2） 写新的类clientWithFunctionA 然后通过继承Client类来覆写Service。

相比2，选项1还好那么一点， 至少可以使用工厂模式来区分不同的服务。 如果不幸选择了2的话，那么以后代码的维系只能自求多福了。 首先，因为Client是个类，不是抽象类或者是接口，所以继承了之后任何任何与service相关的成员方法都需要覆写，无形中增加了程序的不稳定性。 再者，如果在功能A上有新加了功能B， 那么只能继续构造新的类：clientWithFunctionAWithFunctionB。维护代码会变得越发困难。


[^codeWithoutDIExample]: 代码（[出处](https://en.wikipedia.org/wiki/Dependency_injection)）：展示了如果不使用DI的话，代码看起来会怎么样。这段代码的主要问题在于：Client类和Service类的关系是hardcode（硬编）上去的，导致Client只能提供固定的功能。