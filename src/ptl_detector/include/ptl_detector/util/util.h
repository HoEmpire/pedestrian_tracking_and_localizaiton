#pragma once
#include <iostream>

#include <ros/package.h>
#include <ros/ros.h>
#include "ptl_detector/detector/yolo_object_detector.h"

using namespace std;
// Integral type equal
template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type Equal(
    const T &lhs, const T &rhs)
{
    return lhs == rhs;
}

// Floating point type equal
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type Equal(
    const T &lhs, const T &rhs)
{
    return std::fabs(lhs - rhs) < std::numeric_limits<T>::epsilon();
}

/** Template function that generates a comma separated string from the contents
 * of a vector. Elements are separated by a comma and a space for readability.
 */
template <typename T>
std::string Vector2Csv(const std::vector<T> &vec)
{
    std::string s;
    for (typename std::vector<T>::const_iterator it = vec.begin();
         it != vec.end(); ++it)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << *it;
        s += ss.str();
        s += ", ";
    }
    if (s.size() >= 2)
    { // clear the trailing comma, space
        s.erase(s.size() - 2);
    }
    return s;
}