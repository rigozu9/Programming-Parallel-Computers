#include <iostream>
#include <string>
#include <stdexcept>


int comparison() {
    int x = 5;
    double y = 2.5;

    x = y;
    std::cout << x << std::endl;
    return 0;
}

int main() {
    comparison();
    try {
        std::string s = "abc";
        int x = std::stoi(s);
    }
    catch (std::invalid_argument& e) {
        std::cout << e.what() << std::endl;
    }
    return 0;
}