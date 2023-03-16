package Java;

public class Metho1 {
    public static void main(String[] args) {
        /*
         * 
         1) 添加 speak 成员方法,输出 “我是一个好人”
            2) 添加 cal01 成员方法,可以计算从 1+..+1000 的结果
            3) 添加 cal02 成员方法,该方法可以接收一个数 n，计算从 1+..+n 的结果
            4) 添加 getSum 成员方法,可以计算两个数的和
         */
        Person p1=new Person();
        System.out.println("使用speak方法");
       
    }
}
class Person{
    public void speak(){
        System.out.println("我是一个好人");
    }


    public void cal01(){
        int sum=0;
        for (int i = 0; i < 1000; i++) {
            sum=sum+i;
            
        }
        System.out.println("1+...+1000的和为:"+sum);
    }

    public void cal02(int n){
        int sum=0;
        for (int i = 0; i < n; i++) {
            sum=sum+i;
            
        }
        System.out.println("1+...+1000的和为:"+sum);
    }
}
