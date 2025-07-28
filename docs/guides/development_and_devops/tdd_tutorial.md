# Test-Driven Development (TDD)  



## Why TDD Matters: Motivation & Benefits  

Test-Driven Development (**TDD**) is more than just a testing technique—it’s a **development philosophy** that helps data scientists and developers to write **more reliable and maintainable code**.  

### **Real-World Scenarios Where TDD Saves the Day**  

**Catching Bugs Early:** A developer implements a new feature but unknowingly breaks an existing one. With TDD, automated tests would have **immediately flagged the issue**, preventing a costly debugging session.  

**Reducing Last-Minute Fixes:** Without TDD, testing often happens **at the end of development**, leading to **late-stage surprises** and rushed patches before deployment. TDD ensures that tests **guide the development process**, minimizing late fixes.  

**Encouraging Better Code Design:** Writing tests first forces developers to **think about design before implementation**, often leading to **cleaner, more modular, and reusable code**.  

**Building Confidence for Refactoring:** With a **strong test suite**, developers can **refactor code fearlessly**, knowing that **if something breaks, tests will catch it immediately**.  

By integrating TDD into development workflows, teams can **write higher-quality software with fewer surprises down the road**.  

---

## What is TDD & How to Implement It?  

### **Test-Driven Development in 5 Steps**  

TDD follows a simple yet powerful **Red-Green-Refactor** cycle. According to [IBM Developer](https://developer.ibm.com/articles/5-steps-of-test-driven-development/), here’s how it works:  

1. **Write a Failing Test (Red)** → Define a test for the functionality **before writing any code**. The test should fail since the implementation **doesn’t exist yet**.  

2. **Write the Minimum Code to Pass the Test (Green)** → Implement the simplest solution that makes the test **pass**.  

3. **Refactor the Code (Refactor)** → Clean up the implementation **without changing its behavior** to improve readability, efficiency, and maintainability.  

4. **Repeat the Cycle** → Add new tests, write code to pass them, and refactor.  

5. **Run All Tests Regularly** → Ensuring existing features **remain functional as the software evolves**.  

This iterative process **keeps the codebase in check** while ensuring that **new features don’t introduce hidden defects**.  

---

## Structuring Test Functions  

When writing tests, **keeping them structured** is essential for clarity and maintainability. The best practice is to follow the **Given-When-Then pattern**, also known as **Arrange-Act-Assert pattern**.  

### **The Importance of Test Structure**  

**Brian Okken** explains structuring test functions best in his book *Python Testing with pytest*:  

> *"I recommend making sure you keep assertions at the end of test functions. This is such a common recommendation that it has at least two names: Arrange-Act-Assert and Given-When-Then. Bill Wake originally named the Arrange-Act-Assert pattern in 2001. Kent Beck later popularized the practice as part of test-driven development (TDD). Behavior-driven development (BDD) uses the terms Given-When-Then, a pattern from Ivan Moore, popularized by Dan North. Regardless of the names of the steps, the goal is the same: separate a test into stages."*  

### **Given/When/Then Pattern Explained**  

- **Given (Arrange):** Set up the initial conditions for the test (mock data, dependencies, environment).  
- **When (Act):** Perform the actual action you are testing (calling a function, making a request).  
- **Then (Assert):** Verify that the expected outcome occurred.  

---

## Example: Applying the Given-When-Then Pattern in Python  

Below is a simple **TDD-based test** using `pytest` that follows **Given-When-Then**:  



```python
import pytest

# Function to test
def add_numbers(a, b):
    return a + b

# Test using Given-When-Then structure
def test_add_numbers():
    # Given (Arrange): two integer numbers.
    num1 = 3
    num2 = 5

    # When (Act): use the function add_number to sum both values.
    result = add_numbers(num1, num2)

    # Then (Assert): we are expecting 3 + 5 = 8 as a result.
    assert result == 8
```