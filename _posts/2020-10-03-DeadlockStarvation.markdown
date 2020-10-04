---
layout: post
title:  "Deadlock and Starvation Java Demo"
date:   2020-10-03 22:36:32 -0700
categories: Java
---
* This will become a table of contents (this text will be scraped).
{:toc}


### Enviroment Setup
---
**Java Playground:** [https://repl.it/languages/java](https://repl.it/languages/java)


### Deadlock Demo
---
Deadlock is about holding and waiting another lock forms a loop. 


**Demo:** [https://repl.it/@trsong/Deadlock-Demo#Main.java](https://repl.it/@trsong/Deadlock-Demo#Main.java)
```java
public class Main {
	public static class SyncObject1 {}
	public static class SyncObject2 {}

	/**
	 * Deadlock Demo
	 * In terminal run: java Demo
	 * In another terminal run to find java PID: jps -mv
	 * and then: jstack -l <PID>
	 *
	 * Found one Java-level deadlock:
	 * =============================
	 * "Thread1":
	 *   waiting to lock monitor 0x00007fc09fe13e00 (object 0x000000070fe1b708, a Demo$SyncObject2),
	 *   which is held by "Thread2"
	 * "Thread2":
	 *   waiting to lock monitor 0x00007fc09fe13f00 (object 0x000000070fe1aa10, a Demo$SyncObject1),
	 *   which is held by "Thread1"
	 */
	public static void main(String[] args) {
		SyncObject1 object1 = new SyncObject1();
		SyncObject2 object2 = new SyncObject2();
		Thread t1 = new Thread(new SyncAndRunnable(object1, object2, 1, 2, true));
		Thread t2 = new Thread(new SyncAndRunnable(object1, object2, 2, 1, false));

		t1.setName("Thread1");
		t1.start();
		t2.setName("Thread2");
		t2.start();
	}

	public static class SyncAndRunnable implements Runnable {
		final SyncObject1 obj1;
		final SyncObject2 obj2;
		final int a, b;
		final boolean flag;

		public SyncAndRunnable(SyncObject1 obj1, SyncObject2 obj2, int a, int b, boolean flag) {
			this.obj1 = obj1;
			this.obj2 = obj2;
			this.a = a;
			this.b = b;
			this.flag = flag;
		}

		@Override
		public void run() {
			try {
				if (flag) {
					synchronized (obj1) {
						Thread.sleep(3000);
						synchronized (obj2) {
							System.out.println(a + b);
						}
					}
				} else {
					synchronized (obj2) {
						Thread.sleep(3000);
						synchronized (obj1) {
							System.out.println(a + b);
						}
					}
				}

			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
}
```


### Starvation Demo
---
Starvation is about waiting for underlying dependency that has no chance or low priority to be scheduled. 
 
**Demo:** [https://repl.it/@trsong/Starvation-Demo#Main.java](https://repl.it/@trsong/Starvation-Demo#Main.java)
```java
import java.util.concurrent.*;

public class Demo {

	/**
	 * Starvation Demo
	 * In terminal run: java Demo
	 * In another terminal run to find java PID: jps -mv
	 * and then: jstack -l <PID>
	 *
	 * "main" #1 prio=5 os_prio=31 cpu=150.72ms elapsed=65.36s tid=0x00007fed14808800 nid=0x2803 waiting on condition  [0x0000700002d2e000]
	 *    java.lang.Thread.State: WAITING (parking)
	 *         at jdk.internal.misc.Unsafe.park(java.base@11.0.8/Native Method)
	 *         - parking to wait for  <0x000000070fec5c00> (a java.util.concurrent.FutureTask)
	 *         at java.util.concurrent.locks.LockSupport.park(java.base@11.0.8/LockSupport.java:194)
	 *         at java.util.concurrent.FutureTask.awaitDone(java.base@11.0.8/FutureTask.java:447)
	 *         at java.util.concurrent.FutureTask.get(java.base@11.0.8/FutureTask.java:190)
	 *         at Demo.main(Demo.java:36)
	 *
	 *  "pool-1-thread-1" #13 prio=5 os_prio=31 cpu=0.60ms elapsed=65.21s tid=0x00007fed14024000 nid=0x5f03 waiting on condition  [0x000070000406a000]
	 *    java.lang.Thread.State: WAITING (parking)
	 *         at jdk.internal.misc.Unsafe.park(java.base@11.0.8/Native Method)
	 *         - parking to wait for  <0x000000070fd01760> (a java.util.concurrent.FutureTask)
	 *         at java.util.concurrent.locks.LockSupport.park(java.base@11.0.8/LockSupport.java:194)
	 *         at java.util.concurrent.FutureTask.awaitDone(java.base@11.0.8/FutureTask.java:447)
	 *         at java.util.concurrent.FutureTask.get(java.base@11.0.8/FutureTask.java:190)
	 *         at Demo$MyCallable.call(Demo.java:46)
	 *         at Demo$MyCallable.call(Demo.java:41)
	 *         at java.util.concurrent.FutureTask.run(java.base@11.0.8/FutureTask.java:264)
	 *         at java.util.concurrent.ThreadPoolExecutor.runWorker(java.base@11.0.8/ThreadPoolExecutor.java:1128)
	 *         at java.util.concurrent.ThreadPoolExecutor$Worker.run(java.base@11.0.8/ThreadPoolExecutor.java:628)
	 *         at java.lang.Thread.run(java.base@11.0.8/Thread.java:834)
	 */
	public static void main(String[] args) throws ExecutionException, InterruptedException {
		Future<String> submit = single.submit(new MyCallable());
		System.out.println(submit.get());
		System.out.println("over");
		single.shutdown();
	}

	public static class MyCallable implements Callable<String> {
		@Override
		public String call() throws Exception {
			System.out.println("In MyCallable");
			Future<String> submit = single.submit(new AnotherCallable());
			return "success:" + submit.get();
		}
	}

	public static class AnotherCallable implements Callable<String> {
		@Override
		public String call() throws Exception {
			System.out.println("In another callable");
			return "another callable success";
		}
	}

	private final static ExecutorService single = Executors.newSingleThreadExecutor();
}
```


 