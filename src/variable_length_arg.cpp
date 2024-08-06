#include<iostream>

template<typename I>
class base
{template<typename T>
    void printArgs(T t)
    {
        std::cout << t << "\n";
    }
    
    template<typename T, typename... args>
    void printArgs(T t, args... Arg)
    {
        std::cout << t << "\n";
        printArgs(Arg...);
    }
    public:
    
    template<typename... args>
    base(args... Arg)
    {
        printArgs(Arg...);
    }
};

int main ()
{
    base<int> a(3,2,3);
}