#include <iostream>
#include <vector>
#include <type_traits>

// Base template class
template <typename T>
class Base {
protected:
    std::vector<T> baseValues;

public:
    // Variadic template constructor, enabled only if none of Args are of type Base<T>
    template <typename... Args, typename = std::enable_if_t<!(std::is_same_v<Base<T>, std::decay_t<Args>> || ...)>>
    Base(Args... args) : baseValues{static_cast<T>(args)...} {
        std::cout << "Base variadic constructor called." << std::endl;
    }

    // Explicit copy constructor
    Base(const Base& other) : baseValues(other.baseValues) {
        std::cout << "Base copy constructor called." << std::endl;
    }

    // Virtual method to print the base values
    virtual void print
