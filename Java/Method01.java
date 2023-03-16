package Java;

import java.util.Scanner;
public class Method01 {
	//编写一个 main 方法
	public static void main(String[] args) {
		//方法使用
		//1. 方法写好后，如果不去调用(使用)，不会输出
		//2. 先创建对象 ,然后调用方法即可

		Person p1 = new Person();
		p1.speak(); //调用方法
		p1.cal01(); //调用 cal01 方法
		System.out.println("请输入一个数字：");
		Scanner myScanner = new Scanner(System.in);
		int num=myScanner.nextInt();
		p1.cal02(num);
		p1.cal02(5); //调用 cal02 方法，同时给 n = 5
		p1.cal02(10); //调用 cal02 方法，同时给 n = 10
		
		//调用 getSum 方法，同时 num1=10, num2=20
		
		//把 方法 getSum 返回的值，赋给 变量 returnRes
		System.out.println("请输入一个数字：");
		int num1=myScanner.nextInt();
		int num2=myScanner.nextInt();
		int returnRes1 = p1.getSum(num1,num2);
		int returnRes2 = p1.getSum(10, 20);
		System.out.println("getSum 方法返回的值=" + returnRes1);
		System.out.println("getSum 方法返回的值=" + returnRes2);
		
		int returnRes3 = p1.getSum1();
		System.out.println("getSum1 方法返回的值=" + returnRes3);
	}
}
class Person {
	//String name;
	int age;
	//方法(成员方法)
	//添加 speak 成员方法,输出 “我是一个好人”
	//老韩解读
	//1. public 表示方法是公开
	//2. void： 表示方法没有返回值
	//3. speak() : speak 是方法名， () 形参列表
	//4. {} 方法体，可以写我们要执行的代码
	//5. System.out.println("我是一个好人"); 表示我们的方法就是输出一句话
	public void speak() {
		System.out.println("我是一个好人");
	}
	//添加 cal01 成员方法,可以计算从 1+..+1000 的结果
	public void cal01() {
		//循环完成
		int res = 0;
		for(int i = 1; i <= 1000; i++) {
			res += i;
		}
		System.out.println("cal01 方法 计算结果=" + res);
	}
	//添加 cal02 成员方法,该方法可以接收一个数 n，计算从 1+..+n 的结果
	//1. (int n) 形参列表， 表示当前有一个形参 n, 可以接收用户输入
	public void cal02(int n) {
		//循环完成
		int res = 0;
		for(int i = 1; i <= n; i++) {
			res += i;
		}
		System.out.println("cal02 方法 计算结果=" + res);
	}
	
	//添加 getSum 成员方法,可以计算两个数的和
	//老韩解读
	//1. public 表示方法是公开的
	//2. int :表示方法执行后，返回一个 int 值
	//3. getSum 方法名
	//4. (int num1, int num2) 形参列表，2 个形参，可以接收用户传入的两个数

	//5. return res; 表示把 res 的值， 返回
	public int getSum(int num1, int num2) {
		int res = num1 + num2;
		return res;
	}
	public int getSum1(){
		Scanner sum1=new Scanner(System.in);
		System.out.println("请输入一个数字：");
		int num1=sum1.nextInt();
		int num2=sum1.nextInt();
		int res = num1 + num2;
		return res;
	}
}