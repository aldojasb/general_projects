## SOLID Principles

The SOLID principles are foundational for creating maintainable and scalable software. These principles guide developers in designing mid-level software structures that:

- Tolerate change
- Are easy to understand
- Serve as reusable components in various software systems

### **Single Responsibility Principle (SRP)**

**Definition:** A class should have only one reason to change, meaning it should only have one job or responsibility.

#### **Example**

```python
# Bad example
class ReportManager:
    def generate_report(self):
        # logic for generating report
        pass
    
    def save_to_file(self):
        # logic for saving report to file
        pass

# Good example (separating responsibilities)
class ReportGenerator:
    def generate_report(self):
        # logic for generating report
        pass

class ReportSaver:
    def save_to_file(self, report):
        # logic for saving report to file
        pass
```

#### **Key Takeaway**

A module should be responsible to **one and only one actor**. This ensures clarity, maintainability, and flexibility.

------

### **Open-Closed Principle (OCP)**

**Definition:** Software entities (classes, modules, functions, etc.) should be **open for extension but closed for modification**.

#### **Counter-Example (Violation of OCP)**

```python
class Shape:
    def __init__(self, shape_type: str, dimension: float):
        self.shape_type = shape_type
        self.dimension = dimension

    def calculate_area(self) -> float:
        if self.shape_type == "circle":
            return 3.14 * (self.dimension ** 2)
        elif self.shape_type == "square":
            return self.dimension ** 2
        else:
            raise ValueError("Unknown shape type")
```

Problems:

- Adding a new shape requires modifying the `calculate_area` method.

#### **Corrected Example (Applying OCP)**

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def calculate_area(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def calculate_area(self) -> float:
        return 3.14 * (self.radius ** 2)

class Square(Shape):
    def __init__(self, side: float):
        self.side = side

    def calculate_area(self) -> float:
        return self.side ** 2
```

#### **Key Takeaway**

OCP ensures new functionality is added through **extensions** rather than modifications, making the system more maintainable.

------

### **Liskov Substitution Principle (LSP)**

**Definition:** Derived classes must be substitutable for their base classes without affecting correctness.

#### **Violation Example**

```python
class Bird:
    def fly(self) -> str:
        return "I'm flying!"

class Penguin(Bird):
    def fly(self) -> str:
        raise NotImplementedError("Penguins can't fly")
```

#### **Correct Application**

```python
from abc import ABC, abstractmethod

class Bird(ABC):
    @abstractmethod
    def move(self) -> str:
        pass

class Sparrow(Bird):
    def move(self) -> str:
        return "I'm flying!"

class Penguin(Bird):
    def move(self) -> str:
        return "I'm swimming!"
```

#### **Key Takeaway**

Subclasses should extend behavior **without altering the expected behavior** of the base class.

------

### **Interface Segregation Principle (ISP)**

**Definition:** Clients should not be forced to depend on interfaces they do not use.

#### **Violation Example**

```python
class Animal:
    def fly(self) -> None:
        pass

    def swim(self) -> None:
        pass

class Bird(Animal):
    def fly(self) -> None:
        print("I'm flying!")

    def swim(self) -> None:
        raise NotImplementedError("Birds can't swim")
```

#### **Correct Application**

```python
from abc import ABC, abstractmethod

class Flyable(ABC):
    @abstractmethod
    def fly(self) -> None:
        pass

class Swimmable(ABC):
    @abstractmethod
    def swim(self) -> None:
        pass

class Bird(Flyable):
    def fly(self) -> None:
        print("I'm flying!")

class Fish(Swimmable):
    def swim(self) -> None:
        print("I'm swimming!")
```

#### **Key Takeaway**

Create **specific interfaces** rather than forcing classes to implement unnecessary methods.

------

### **Dependency Inversion Principle (DIP)**

**Definition:**

- High-level modules should not depend on low-level modules. Both should depend on abstractions.
- Abstractions should not depend on details. Details should depend on abstractions.

#### **Violation Example**

```python
class EmailSender:
    def send_email(self, message: str) -> None:
        print(f"Sending email: {message}")

class NotificationService:
    def __init__(self):
        self.email_sender = EmailSender()

    def notify(self, message: str) -> None:
        self.email_sender.send_email(message)
```

#### **Correct Application (Applying DIP)**

```python
from abc import ABC, abstractmethod

class Notifier(ABC):
    @abstractmethod
    def notify(self, message: str) -> None:
        pass

class EmailSender(Notifier):
    def notify(self, message: str) -> None:
        print(f"Sending email: {message}")

class SMSNotifier(Notifier):
    def notify(self, message: str) -> None:
        print(f"Sending SMS: {message}")

class NotificationService:
    def __init__(self, notifier: Notifier):
        self.notifier = notifier

    def notify(self, message: str) -> None:
        self.notifier.notify(message)
```

#### **Key Takeaway**

DIP helps create modular, **loosely coupled** systems that are more adaptable and maintainable.

------

### **Final Thoughts**

By applying **SOLID principles**, developers create software that is **flexible, maintainable, and scalable**. These principles provide a strong foundation for designing robust systems that can evolve efficiently with changing business needs.