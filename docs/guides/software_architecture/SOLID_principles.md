# 🏰️ SOLID Principles

The SOLID principles are foundational for creating **maintainable and scalable software**. These principles guide developers in designing software structures that:

- **Tolerate change**
- **Are easy to understand**
- **Serve as reusable components** in various software systems

Applying **SOLID principles** leads to cleaner, more adaptable code and a smoother development experience.  🚀

------

## **Single Responsibility Principle (SRP)**

**Definition:** A class should have **only one reason to change**, meaning it should have a single responsibility.

### ❌ **Counter-Example (SRP Violation)**

```python
# A class handling multiple responsibilities
class InvoiceProcessor:
    def calculate_total(self, items):
        # logic for calculate total items
        pass
    
    def generate_pdf(self, invoice):
        # logic for generate a pdf
        pass
    
    def send_email(self, invoice, email)
    	# logic to send a email
        pass   
    
```

🚨 **Problem:** If you need to change how PDFs are generate, you might risk breaking other functionality like email sender.

### ✅ **Corrected Example (Applying SRP)**

```python
# Good example: Separating responsibilities
class InvoiceCalculator:
    def calculate_total(self, items):
        # logic for generating report
        pass

class InvoicePDFGenerator:
    def generate_pdf(self, invoice):
        # logic for generate a pdf
        pass

class EmailSender:
    def send_email(self, invoice, email):
        # logic for send emails
        pass
```

### 🎯 **Key Takeaway**

A module should be responsible to **one and only one actor**. This ensures **clarity, maintainability, and flexibility**.

------

## **Open-Closed Principle (OCP)**

**Definition:** Software entities **should be open for extension but closed for modification**.

### ❌ **Counter-Example (OCP Violation)**

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

🚨 **Problem:** Adding a new shape requires modifying the `calculate_area` method, violating OCP.

### ✅ **Corrected Example (Applying OCP)**

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

### 🎯 **Key Takeaway**

OCP ensures new functionality is added through **extensions** rather than modifications, making the system more maintainable.

------

## **Liskov Substitution Principle (LSP)**

**Definition:** Derived classes must be **substitutable for their base classes** without affecting correctness.

### ❌ **Counter-Example (LSP Violation)**

```python
class Bird:
    def fly(self) -> str:
        return "I'm flying!"

class Penguin(Bird):
    def fly(self) -> str:
        raise NotImplementedError("Penguins can't fly")
```

🚨 **Problem:** `Penguin` violates LSP because it inherits behavior it cannot fulfill.

### ✅ **Corrected Example (Applying LSP)**

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

### 🎯 **Key Takeaway**

Subclasses should extend behavior **without altering the expected behavior** of the base class.

------

## **Interface Segregation Principle (ISP)**

**Definition:** Clients **should not be forced to depend on interfaces they do not use**.

### ❌ **Counter-Example (ISP Violation)**

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

🚨 **Problem:** `Bird` is forced to implement `swim()`, which it doesn't need.

### ✅ **Corrected Example (Applying ISP)**

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

### 🎯 **Key Takeaway**

Create **specific interfaces** rather than forcing classes to implement unnecessary methods.

------

## **Dependency Inversion Principle (DIP)**

**Definition:**
**High-level modules should not depend on low-level modules**. Both should depend on abstractions.
**Abstractions should not depend on details**. Details should depend on abstractions.

### ❌ **Counter-Example (DIP Violation)**

```python
# Low-level class
class EmailSender:
    def send_email(self, message: str) -> None:
        print(f"Sending email: {message}")

# High-level class
class NotificationService:
    def __init__(self):
        self.email_sender = EmailSender()

    def notify(self, message: str) -> None:
        self.email_sender.send_email(message)
```

🚨 **Problem:** `NotificationService` is tightly coupled to `EmailSender`, making it harder to extend or change notification types.

### ✅ **Corrected Example (Applying DIP)**

```python
from abc import ABC, abstractmethod

# Abstraction
class Notifier(ABC):
    @abstractmethod
    def notify(self, message: str) -> None:
        pass

# Low-level class
class EmailSender(Notifier):
    def notify(self, message: str) -> None:
        print(f"sending email: {message}")

class SMSNotifier(Notifier):
    def notify(self, message: str) -> None:
        print(f"sending SMS: {message}")

# High-level class
class NotificationService:
    def __init__(self, notifier: Notifier):
        self.notifier = notifier
    
    def notify(self, message: str) -> None:
        self.notifier.notify(message)
        
```

### 🎯 **Key Takeaway**

DIP helps create modular, **loosely coupled** systems that are more adaptable and maintainable.



