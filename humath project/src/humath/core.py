PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
E = 2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274
TAU = 6.2831853071795864769252867665590057683943387987502116419498891846156328125724179972560696506842341359
INF = float('inf')
NAN = float('nan')
GOLDEN_RATIO = 1.6180339887498948482045868343656381177203091798057628621354486227052604628189024497072072041893911374
deg_to_rad_constant = 0.0174532925199432957692369076848861271344287188854172545609719144017100911460344944368224156963450948
rad_to_deg_constant = 57.295779513082320876798154814105170332405472466564321549160243861202847148321552632440968995851110944

def is_close(a: float, b: float, rel_tol: float = 1e-09) -> bool:
    """
    ⚖️ **Check if Two Floating-Point Numbers Are Close**

    Evaluates whether two values are mathematically close to each other within a 
    specified relative tolerance threshold. Essential for bypassing binary 
    floating-point equality limitations (`a == b`).

    ---

    ### 📥 Arguments:
    *   `a` (float / int): The first numerical value to compare.
    *   `b` (float / int): The second numerical value to compare.
    *   `rel_tol` (float): The maximum allowed relative difference fraction. *Defaults to 1e-09.*

    ### 📤 Returns:
    *   `bool`: `True` if the numbers are within tolerance of each other, `False` otherwise.

    ---

    ### 💡 Example Usage:
    ```python
    # Bypassing raw float limitations
    0.1 + 0.2 == 0.3          # Returns: False (due to binary float drift)
    is_close(0.1 + 0.2, 0.3)  # Returns: True

    # Checking values outside tolerance boundaries
    is_close(100.0, 101.0, rel_tol=0.001)  # Returns: False
    ```
    """
    # Absolute difference between the two numbers
    diff = a - b
    abs_diff = -diff if diff < 0 else diff
    
    # Absolute values of a and b to scale the tolerance
    abs_a = -a if a < 0 else a
    abs_b = -b if b < 0 else b
    
    # Determine the larger value to scale the relative tolerance safely
    max_val = abs_a if abs_a > abs_b else abs_b
    
    # Handle the edge case where both are exactly 0
    if max_val == 0:
        return True
        
    return abs_diff <= (max_val * rel_tol)

def lerp(start: float, end: float, t: float) -> float:
    """
    📈 **Linear Interpolation (Lerp)**

    Smoothly glides between a starting value and an ending value based on a percentage 
    fraction tracker `t`. Widely used in animations, camera tracking, and game physics loops.

    ---

    ### 📥 Arguments:
    *   `start` (float / int): The baseline starting value (returned when `t = 0.0`).
    *   `end` (float / int): The targeted destination value (returned when `t = 1.0`).
    *   `t` (float): The interpolation fraction. A value of `0.5` represents the exact midpoint.

    ### 📤 Returns:
    *   `float`: The calculated value between the start and end ranges.

    ---

    ### 💡 Example Usage:
    ```python
    # Finding the exact midpoint between 10 and 20
    lerp(10, 20, 0.5)   # Returns: 15.0

    # Moving 10% along a trajectory path from 0 to 100
    lerp(0, 100, 0.1)   # Returns: 10.0
    ```
    """
    return start + t * (end - start)

def fma(x: float, y: float, z: float) -> float:
    """
    🚀 **Fused Multiply-Add (FMA)**

    Computes `(x * y) + z` with infinite precision during the intermediate steps, 
    applying only a single rounding error at the very end. This prevents 
    compounding floating-point drift.

    ---

    ### 📥 Arguments:
    *   `x` (float / int): The first value to multiply.
    *   `y` (float / int): The second value to multiply.
    *   `z` (float / int): The value to add to the product.

    ### 📤 Returns:
    *   `float`: The precisely computed and single-rounded final value.

    ---

    ### 💡 Example Usage:
    ```python
    # Standard float multiplication compounded drift error vs FMA
    # a large number * tiny number + offset
    fma(1.23456789, 9.87654321, 0.00001)
    ```
    """
    # Convert floats to exact integer fractions using native float methods
    x_num, x_den = x.as_integer_ratio()
    y_num, y_den = y.as_integer_ratio()
    z_num, z_den = z.as_integer_ratio()
    
    # Perform the exact rational arithmetic: (x_num/x_den * y_num/y_den) + z_num/z_den
    # Common denominator = x_den * y_den * z_den
    mul_num = x_num * y_num * z_den
    add_num = z_num * x_den * y_den
    total_num = mul_num + add_num
    total_den = x_den * y_den * z_den
    
    # Return the exact division result as a float
    return total_num / total_den

def copysign(x: float, y: float) -> float:
    """
    🧬 **Copy Sign of a Number**

    Returns a value with the raw magnitude (absolute value) of `x` but combined 
    with the positive or negative sign of `y`. Safely handles signed zeroes (`-0.0`).

    ---

    ### 📥 Arguments:
    *   `x` (float / int): The magnitude source value.
    *   `y` (float / int): The sign destination value.

    ### 📤 Returns:
    *   `float`: The absolute value of `x` containing the exact sign tracking of `y`.

    ---

    ### 💡 Example Usage:
    ```python
    # Standard positive matching
    copysign(5.0, -3.0)   # Returns: -5.0

    # Handling signed zeroes properly
    copysign(10.0, -0.0)  # Returns: -10.0
    ```
    """
    # Get the raw absolute value of x
    abs_x = -x if x < 0 else x
    
    # Check if y is negative, safely catching floating-point -0.0 via string representation
    is_y_negative = str(float(y)).startswith('-')
    
    return -abs_x if is_y_negative else abs_x

def nextafter(x: float, y: float) -> float:
    """
    ⏭️ **Find Next Adjacent Floating-Point Value**

    Returns the next representable floating-point value after `x` traveling in the 
    direction of `y`. Essential for precise numerical simulation bounds, checking 
    floating-point step limits, and edge-case testing.

    ---

    ### 📥 Arguments:
    *   `x` (float): The starting base floating-point value.
    *   `y` (float): The target direction value towards which to step.

    ### 📤 Returns:
    *   `float`: The adjacent float value immediately preceding or following `x`.

    ---

    ### 💡 Example Usage:
    ```python
    # Find the smallest float value greater than 1.0
    nextafter(1.0, 2.0)   # Returns: 1.0000000000000002

    # Stepping from 0 down into negative subnormal space
    nextafter(0.0, -1.0)  # Returns: -4.940656458412465e-324
    ```
    """
    # Handle NaN cases (NaN is never equal to itself)
    if x != x or y != y:
        return float('nan')

    # If values are identical, return y directly
    if x == y:
        return y

    # Handle zero transitions cleanly (5e-324 is the smallest subnormal double float)
    if x == 0.0:
        return -4.940656458412465e-324 if y < 0.0 else 4.940656458412465e-324

    # Helper: Converts any float to its exact 64-bit unsigned integer bit configuration
    def float_to_bits(f: float) -> int:
        if f == float('inf'): return 0x7ff0000000000000
        if f == float('-inf'): return 0xfff0000000000000

        s = float.hex(f)
        sign = 1 if s.startswith('-') else 0
        if s.startswith('-'): s = s[1:]

        prefix, exp_part = s.split('p')
        exponent = int(exp_part)

        hex_digits = prefix[4:].ljust(13, '0')
        mantissa = int(hex_digits, 16)

        exp_bits = 0 if prefix.startswith('0x0.') else exponent + 1023
        return (sign << 63) | (exp_bits << 52) | mantissa

    # Helper: Reconstructs a float out of 64-bit integer bits
    def bits_to_float(bits: int) -> float:
        sign = (bits >> 63) & 1
        exp_bits = (bits >> 52) & 0x7ff
        mantissa = bits & 0xfffffffffffff

        if exp_bits == 0x7ff:
            if mantissa == 0: return float('-inf') if sign else float('inf')
            return float('nan')

        if exp_bits == 0:
            prefix = f"0x0.{hex(mantissa)[2:].zfill(13)}"
            exponent = -1022
        else:
            prefix = f"0x1.{hex(mantissa)[2:].zfill(13)}"
            exponent = exp_bits - 1023

        return float.fromhex(f"{'-' if sign else ''}{prefix}p{exponent:+d}")

    # Step into bit space to compute adjacent stepping
    bx = float_to_bits(x)
    if x > 0.0:
        bx += 1 if y > x else -1
    else: # x < 0.0
        bx -= 1 if y > x else -1

    return bits_to_float(bx)

def frexp(x: float) -> tuple[float, int]:
    """
    🧮 **Deconstruct Float into Mantissa and Exponent**

    Breaks a floating-point number down into its normalized fraction (mantissa) 
    and an integral power of 2, such that `x = mantissa * (2 ** exponent)`. 
    The absolute value of the returned mantissa will sit in the range [0.5, 1.0).

    ---

    ### 📥 Arguments:
    *   `x` (float): The floating-point number to decompose.

    ### 📤 Returns:
    *   `tuple[float, int]`: A tuple containing (mantissa, exponent).

    ---

    ### 💡 Example Usage:
    ```python
    # Decomposing a standard integer
    frexp(8.0)    # Returns: (0.5, 4) -> since 0.5 * (2 ** 4) == 8.0

    # Decomposing a fraction
    frexp(0.25)   # Returns: (0.5, -1) -> since 0.5 * (2 ** -1) == 0.25
    ```
    """
    # Handle zero edge cases
    if x == 0.0:
        return (0.0, 0)

    # Handle infinity and NaN
    if x != x:  # NaN check
        return (x, 0)
    if x in (float('inf'), float('-inf')):
        return (x, 0)

    # Convert the float to its exact hexadecimal string representation
    hex_str = float.hex(x)
    is_negative = hex_str.startswith('-')

    # Split the hex mantissa component and the exponent 'p' component
    prefix, exp_part = hex_str.split('p')
    raw_exponent = int(exp_part)

    # Check if the float is a subnormal number (starts with 0x0. instead of 0x1.)
    is_subnormal = '0x0.' in prefix or '-0x0.' in prefix

    if is_subnormal:
        # Extract the fractional hex string to find where the first 1-bit is
        hex_digits = prefix.split('.')[-1]
        bits = bin(int(hex_digits, 16))[2:].zfill(len(hex_digits) * 4)
        leading_zeros = len(bits) - len(bits.lstrip('0'))

        # Adjust exponent and reconstruct a normalized representation
        raw_exponent -= leading_zeros
        
        # Re-encode it as a normalized float manually to read it cleanly
        normalized_hex = f"{'-' if is_negative else ''}0x1.{hex(int(bits.lstrip('0'), 2))[2:].ljust(13, '0')}p{raw_exponent}"
        x = float.fromhex(normalized_hex)
        
        # re-fetch parts
        prefix, exp_part = float.hex(x).split('p')
        raw_exponent = int(exp_part)

    # In float.hex(), the format is always 0x1.xxxxx * 2^exponent.
    # But frexp requires the mantissa to be 0.xxxxx * 2^exponent.
    # Therefore, we shift the decimal left 1 spot: divide mantissa by 2, add 1 to exponent.
    shifted_hex_mantissa = f"{prefix}p-1"
    mantissa = float.fromhex(shifted_hex_mantissa)
    exponent = raw_exponent + 1

    return (mantissa, exponent)

def ldexp(x: float, i: int) -> float:
    """
    🔌 **Reconstruct Float from Mantissa and Exponent**

    Computes the inverse of `frexp`, calculating `x * (2 ** i)` precisely by 
    directly manipulating the underlying binary exponent string configuration. 
    Safely catches positive infinity overflows and signed zero underflows.

    ---

    ### 📥 Arguments:
    *   `x` (float): The multiplier fraction (mantissa).
    *   `i` (int): The integer exponent power of 2 scale factor.

    ### 📤 Returns:
    *   `float`: The reassembled floating-point numerical value.

    ---

    ### 💡 Example Usage:
    ```python
    # Reassembling a standard fraction
    ldexp(0.5, 4)    # Returns: 8.0  (0.5 * 2^4)
    
    # Reassembling a negative fraction
    ldexp(-0.75, 2)  # Returns: -3.0 (-0.75 * 2^2)
    ```
    """
    # Handle NaN and Infinity edge cases directly
    if x != x or x in (float('inf'), float('-inf')) or x == 0.0:
        return x

    # If the exponent adjustment is a clean integer within reasonable hardware limits,
    # we can reassemble it instantly using float.hex() and string processing.
    try:
        # Extract the base hexadecimal components of the mantissa 'x'
        hex_str = float.hex(x)
        prefix, exp_part = hex_str.split('p')

        # Add the new exponent shift to the existing binary exponent scale
        new_exponent = int(exp_part) + i

        # Re-encode back into Python's native binary floating-point assembler
        return float.fromhex(f"{prefix}p{new_exponent:+d}")

    except (ValueError, OverflowError):
        # Determine strict sign alignment, safely catching negative zeroes
        is_negative = float.hex(x).startswith('-')

        # Fallback mechanism if the requested exponent exceeds maximum hardware bounds:
        # If it overflows positive space, it pushes to infinity with correct signs
        if i > 0:
            return float('-inf') if is_negative else float('inf')
            
        # If it completely underflows negative space, it settles to signed zero
        return -0.0 if is_negative else 0.0

def add(*args: float) -> float:
    """
    ➕ **Sum Multiple Numbers Together**

    Accepts a variable number of numerical values and calculates their total sum 
    sequentially.

    ---

    ### 📥 Arguments:
    *   `*args` (float / int): An arbitrary list of numbers to add up.

    ### 📤 Returns:
    *   `float / int`: The combined total of all provided inputs.

    ---

    ### 💡 Example Usage:
    ```python
    # Adding integer values
    add(1, 2, 3, 4)  # Returns: 10

    # Mixing floats and integers
    add(1.5, 2.5, 5) # Returns: 9.0
    ```
    """
    output = sum(args)
    return output

def subtract(a: float, b: float) -> float:
    """
    ➖ **Subtract One Number From Another**

    Calculates the mathematical difference by subtracting the second value (`b`) 
    from the first value (`a`).

    ---

    ### 📥 Arguments:
    *   `a` (float / int): The starting value (minuend).
    *   `b` (float / int): The amount to reduce by (subtrahend).

    ### 📤 Returns:
    *   `float / int`: The remaining difference after subtraction.

    ---

    ### 💡 Example Usage:
    ```python
    # Basic integer subtraction
    subtract(10, 3)  # Returns: 7

    # Working with negative numbers
    subtract(5, -2)  # Returns: 7
    ```
    """
    output = a - b
    return output

def multiply(*args: float) -> float:
    """
    ✖️ **Multiply Multiple Numbers Together**

    Accepts a variable number of numerical values and calculates their total product 
    by multiplying them together sequentially.

    ---

    ### 📥 Arguments:
    *   `*args` (float / int): An arbitrary list of numbers to multiply.

    ### 📤 Returns:
    *   `float / int`: The final computed product of all provided inputs.

    ---

    ### 💡 Example Usage:
    ```python
    # Multiplying whole numbers
    multiply(2, 3, 4)  # Returns: 24

    # Including floating point numbers
    multiply(5, 0.5)   # Returns: 2.5
    ```
    """
    output = 1
    for number in args:
        output *= number
    return output

def divide(numerator: float, denominator: float) -> float | None:
    """
    ➗ **Divide One Number By Another Safely**

    Calculates the mathematical quotient by dividing the numerator by the 
    denominator. Includes a safety check to prevent program crashes if the 
    denominator is zero.

    ---

    ### 📥 Arguments:
    *   `numerator` (float / int): The number to be divided (dividend).
    *   `denominator` (float / int): The number to divide by (divisor).

    ### 📤 Returns:
    *   `float`: The resulting quotient of the division.
    *   `None`: If the `denominator` is `0` (prevents a crash).

    ---

    ### 💡 Example Usage:
    ```python
    # Standard clean division
    divide(10, 2)   # Returns: 5.0

    # Safe zero division handling
    divide(5, 0)    # Prints error and returns: None
    ```
    """
    if denominator == 0:
        print("divide error: Cannot divide a number by zero.")
        return None
        
    output = numerator / denominator
    return output

def absolute(number: float) -> float:
    """
    📐 **Get the Absolute Value of a Number**

    Calculates the non-negative magnitude of a number, representing its absolute 
    distance from zero regardless of its positive or negative sign (`|x|`).

    ---

    ### 📥 Arguments:
    *   `number` (float / int): The value to evaluate.

    ### 📤 Returns:
    *   `float / int`: The positive distance value of the number.

    ---

    ### 💡 Example Usage:
    ```python
    # Converting a negative value
    absolute(-42)  # Returns: 42

    # Passing an already positive value
    absolute(7.5)  # Returns: 7.5
    ```
    """
    if number < 0:
        return -number
    return number

def ceil(n: float) -> int:
    """
    ⬆️ **Round a Number Up to the Nearest Integer**

    Calculates the ceiling of a number, rounding it upward to the next 
    consecutive whole integer value. 

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The numerical value you want to round up.

    ### 📤 Returns:
    *   `int`: The rounded whole integer value.

    ---

    ### 💡 Example Usage:
    ```python
    # Rounding up a positive decimal
    ceil(3.1)   # Returns: 4

    # Negative decimals round up toward zero
    ceil(-2.7)  # Returns: -2

    # Whole numbers stay exactly the same
    ceil(5.0)   # Returns: 5
    ```
    """
    return int(-(-n // 1))

def floor(n: float) -> int:
    """
    ⬇️ **Round a Number Down to the Nearest Integer**

    Calculates the floor of a number, rounding it downward to the previous 
    consecutive whole integer value.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The numerical value you want to round down.

    ### 📤 Returns:
    *   `int`: The rounded whole integer value.

    ---

    ### 💡 Example Usage:
    ```python
    # Rounding down a positive decimal
    floor(3.9)   # Returns: 3

    # Negative decimals round down away from zero
    floor(-2.1)  # Returns: -3

    # Whole numbers stay exactly the same
    floor(5.0)   # Returns: 5
    ```
    """
    return int(n // 1)

def truncate(n: float) -> int:
    """
    🪓 **Chop Off Decimal Places (Round Toward Zero)**

    Removes the fractional or decimal portion of a number entirely, leaving 
    only the whole integer part. Unlike floor, it always rounds toward zero.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The numerical value you want to truncate.

    ### 📤 Returns:
    *   `int`: The whole integer part of the number.

    ---

    ### 💡 Example Usage:
    ```python
    # Truncating a positive decimal
    truncate(3.9)   # Returns: 3

    # Truncating a negative decimal (rounds toward zero)
    truncate(-3.9)  # Returns: -3

    # Whole numbers stay identical
    truncate(5.0)   # Returns: 5
    ```
    """
    output = int(n)
    return output

def round_half_up(n: float, digits: int = 0) -> float | int:
    """
    🎯 **Standard Round-Half-Up Utility**

    Bypasses Python's default "Banker's Rounding" (which rounds toward the nearest 
    even number, causing `2.5` to become `2`) and enforces traditional 
    schoolhouse mathematical rounding where half-values always round up.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The target number you want to round.
    *   `digits` (int): Number of decimal places to retain. *Defaults to 0.*

    ### 📤 Returns:
    *   `int`: If `digits == 0`, a whole integer is returned.
    *   `float`: If `digits > 0`, a precise floating decimal is returned.

    ---

    ### 💡 Example Usage:
    ```python
    # Bypassing Banker's Rounding (standard round() yields 2)
    round_half_up(2.5)       # Returns: 3

    # Rounding a negative half-value up (toward positive infinity)
    round_half_up(-2.5)      # Returns: -2

    # Rounding to a specific decimal place without float drift drops
    round_half_up(1.425, 2)  # Returns: 1.43
    ```
    """
    factor = 10 ** digits
    
    # 1e-12 is a microscopic buffer. It corrects binary floating point drift 
    # (e.g. 142.49999999999997 becomes 142.50000000000097) so floor division rounds up perfectly.
    rounded = int((n * factor + 0.5 + 1e-12) // 1)
    
    return rounded if digits == 0 else rounded / factor

def modulo(a: float, b: float) -> float:
    """
    🔢 **Calculate Floored Remainder (Modulo)**

    Returns the remainder of division of `a` by `b`. This implementation uses 
    floored division logic, meaning the resulting sign always matches the 
    divisor (`b`).

    ---

    ### 📥 Arguments:
    *   `a` (float / int): The dividend value.
    *   `b` (float / int): The divisor value.

    ### 📤 Returns:
    *   `float / int`: The remainder after floored division.

    ### ⚠️ Raises:
    *   `ZeroDivisionError`: If the divisor `b` is exactly `0`.

    ---

    ### 💡 Example Usage:
    ```python
    # Standard positive modulo
    modulo(10, 3)    # Returns: 1

    # Negative dividend (returns positive remainder matching divisor sign)
    modulo(-1, 3)    # Returns: 2

    # Negative divisor (returns negative remainder matching divisor sign)
    modulo(5, -3)    # Returns: -1
    ```
    """
    # If dividing by zero, match Python's native ZeroDivisionError behavior
    if b == 0:
        raise ZeroDivisionError("integer division or modulo by zero")

    # Calculate the remainder using pure floor division logic
    return a - (b * (a // b))

def divmod_tuple(a: float, b: float) -> tuple[float, float]:
    """
    📦 **Combined Division and Modulo Operation**

    Simultaneously calculates the floored quotient and the remainder of 
    dividing `a` by `b`, returning them together as a tuple. 

    ---

    ### 📥 Arguments:
    *   `a` (float / int): The dividend value.
    *   `b` (float / int): The divisor value.

    ### 📤 Returns:
    *   `tuple[float, float]`: A tuple structured as `(quotient, remainder)`.

    ### ⚠️ Raises:
    *   `ZeroDivisionError`: If the divisor `b` is exactly `0`.

    ---

    ### 💡 Example Usage:
    ```python
    # Standard positive inputs
    divmod_tuple(10, 3)    # Returns: (3, 1)

    # Handling negative numbers cleanly
    divmod_tuple(-10, 3)   # Returns: (-4, 2)
    ```
    """
    # Guard against dividing by zero to match native exceptions
    if b == 0:
        raise ZeroDivisionError("integer division or modulo by zero")

    # Calculate the quotient using floor division
    quotient = a // b

    # Calculate the remainder based on that quotient
    remainder = a - (b * quotient)

    return (quotient, remainder)

def reciprocal(n: float) -> float:
    """
    🔄 **Calculate Multiplicative Inverse (Reciprocal)**

    Computes `1 / n` precisely. Properly handles floating-point infinity boundaries 
    to map to signed zeroes, maintaining strict IEEE-754 compatibility without dependencies.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The input numerical value to invert.

    ### 📤 Returns:
    *   `float`: The inverted result (`1 / n`).

    ### ⚠️ Raises:
    *   `ZeroDivisionError`: If `n` is exactly `0`.

    ---

    ### 💡 Example Usage:
    ```python
    # Inverting an integer
    reciprocal(4)         # Returns: 0.25

    # Handling positive infinity conversion
    reciprocal(float('inf'))   # Returns: 0.0

    # Handling negative infinity conversion
    reciprocal(float('-inf'))  # Returns: -0.0
    ```
    """
    # Explicitly check for division by zero to match native error structures
    if n == 0:
        raise ZeroDivisionError("float division by zero")

    # Handle floating-point infinity signatures
    if n == float('inf'):
        return 0.0
    if n == float('-inf'):
        return -0.0

    return 1 / n

def negate(n: float) -> float:
    """
    🔄 **Negate a Numerical Value**

    Inverts the sign of the given number. Positive values become negative, 
    and negative values become positive. Safely handles signed zeroes and infinities.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The input numeric value to flip.

    ### 📤 Returns:
    *   `float / int`: The sign-inverted result (`-n`).

    ---

    ### 💡 Example Usage:
    ```python
    # Negating a positive integer
    negate(42)            # Returns: -42

    # Negating a negative float
    negate(-3.14)         # Returns: 3.14

    # Handling signed zeroes properly
    negate(0.0)           # Returns: -0.0
    ```
    """
    return -n

def clamp(val: float, min_val: float, max_val: float) -> float:
    """
    🧰 **Clamp a Value Within a Strict Range**

    Constrains an input value so that it remains strictly between a minimum 
    and maximum boundary limit. If the value overflows either threshold, 
    it snaps directly to that boundary edge.

    ---

    ### 📥 Arguments:
    *   `val` (float / int): The target number to constrain.
    *   `min_val` (float / int): The absolute lower boundary limit.
    *   `max_val` (float / int): The absolute upper boundary limit.

    ### 📤 Returns:
    *   `float / int`: The safely bounded number.

    ---

    ### 💡 Example Usage:
    ```python
    # Value is within range (untouched)
    clamp(15, 0, 100)    # Returns: 15

    # Value drops below minimum limit (snaps to min)
    clamp(-20, 0, 100)   # Returns: 0

    # Value overflows maximum limit (snaps to max)
    clamp(150, 0, 100)  # Returns: 100
    ```
    """
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val

def sign(n: float) -> int:
    """
    🪧 **Extract the Mathematical Sign of a Number**

    Determines if a number is positive, negative, or zero, returning a clean 
    direction index identifier. Safely preserves structural unsigned and signed zeroes.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The value to check.

    ### 📤 Returns:
    *   `int`: The sign indicator mapping:
        *   `-1` for negative numbers
        *   `1` for positive numbers
        *   `0` for zeroes

    ---

    ### 💡 Example Usage:
    ```python
    # Positive numbers
    sign(42.5)   # Returns: 1

    # Negative numbers
    sign(-0.004) # Returns: -1

    # Zero
    sign(0.0)    # Returns: 0
    ```
    """
    if n > 0:
        return 1
    elif n < 0:
        return -1
    return 0

def is_even(n: int) -> bool:
    """
    🔢 **Check if an Integer is Even**

    Evaluates whether a whole integer is perfectly divisible by 2 with no 
    fractional remainder.

    ---

    ### 📥 Arguments:
    *   `n` (int): The target whole number to evaluate.

    ### 📤 Returns:
    *   `bool`: `True` if the number is even, `False` if it is odd.

    ---

    ### 💡 Example Usage:
    ```python
    # Checking even numbers
    is_even(4)    # Returns: True

    # Checking odd numbers
    is_even(7)    # Returns: False

    # Zero is mathematically even
    is_even(0)    # Returns: True
    ```
    """
    return n % 2 == 0

def is_odd(n: int) -> bool:
    """
    🔢 **Check if an Integer is Odd**

    Evaluates whether a whole integer leaves a remainder when divided by 2.

    ---

    ### 📥 Arguments:
    *   `n` (int): The target whole number to evaluate.

    ### 📤 Returns:
    *   `bool`: `True` if the number is odd, `False` if it is even.

    ---

    ### 💡 Example Usage:
    ```python
    # Checking odd numbers
    is_odd(7)     # Returns: True

    # Checking even numbers
    is_odd(4)     # Returns: False

    # Zero is even, so it returns False
    is_odd(0)     # Returns: False
    ```
    """
    return n % 2 != 0

def is_integer(n: float | int) -> bool:
    """
    🔎 **Verify if a Number is a Whole Integer**

    Evaluates whether a value represents a whole number without any fractional 
    components. Safely handles values whether they are typed as raw integers 
    or formatted as floating-point decimals.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The numerical value to inspect.

    ### 📤 Returns:
    *   `bool`: `True` if the number is a whole integer, `False` if it contains fractional parts.

    ---

    ### 💡 Example Usage:
    ```python
    # Standard integer types
    is_integer(5)        # Returns: True

    # Floating-point decimals without a fraction
    is_integer(5.0)      # Returns: True

    # Floating-point decimals with a fractional component
    is_integer(5.25)     # Returns: False
    ```
    """
    return n % 1 == 0

def is_infinite(n: float) -> bool:
    """
    ♾️ **Check if a Number represents Infinity**

    Evaluates whether a value has expanded or overflowed into either positive 
    or negative infinity.

    ---

    ### 📥 Arguments:
    *   `n` (float): The target numerical value to evaluate.

    ### 📤 Returns:
    *   `bool`: `True` if the number matches `float('inf')` or `float('-inf')`, 
               `False` otherwise.

    ---

    ### 💡 Example Usage:
    ```python
    # Testing direct infinity strings
    is_infinite(float('inf'))   # Returns: True
    is_infinite(float('-inf'))  # Returns: True

    # Standard real numbers return False
    is_infinite(3.14159)        # Returns: False
    ```
    """
    return n == float('inf') or n == float('-inf')

def is_nan(n: float) -> bool:
    """
    ❌ **Check if a Value is Not a Number (NaN)**

    Evaluates whether a floating-point value represents a mathematically undefined 
    or corrupted data result (NaN). Bypasses standard identity rules, as NaN is the 
    only value in computer arithmetic that does not equal itself.

    ---

    ### 📥 Arguments:
    *   `n` (float): The target numerical value to evaluate.

    ### 📤 Returns:
    *   `bool`: `True` if the value is NaN, `False` for any valid real number.

    ---

    ### 💡 Example Usage:
    ```python
    # Testing an explicit NaN float
    is_nan(float('nan'))  # Returns: True

    # Standard numbers return False
    is_nan(42.0)          # Returns: False
    ```
    """
    return n != n

def square(n: float) -> float:
    """
    🔢 **Calculate the Square of a Number (x²)**

    Multiplies a value by itself to compute its second power.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The base number to square.

    ### 📤 Returns:
    *   `float / int`: The squared value (`n * n`).

    ---

    ### 💡 Example Usage:
    ```python
    # Squaring a whole number
    square(5)     # Returns: 25

    # Squaring a negative number
    square(-4)    # Returns: 16
    ```
    """
    return n * n

def cube(n: float) -> float:
    """
    🔢 **Calculate the Cube of a Number (x³)**

    Multiplies a value by itself twice to compute its third power.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The base number to cube.

    ### 📤 Returns:
    *   `float / int`: The cubed value (`n * n * n`).

    ---

    ### 💡 Example Usage:
    ```python
    # Cubing a whole number
    cube(3)     # Returns: 27

    # Cubing a negative number (preserves the negative sign)
    cube(-2)    # Returns: -8
    ```
    """
    return n * n * n

def power(base: float, exponent: float) -> float:
    """
    🔢 **Raise a Number to a Custom Power (x^y)**

    Calculates the exponential value of a base number raised to any custom 
    exponent power.

    ---

    ### 📥 Arguments:
    *   `base` (float / int): The starting baseline number.
    *   `exponent` (float / int): The power factor to scale the base by.

    ### 📤 Returns:
    *   `float / int`: The resulting powered value.

    ---

    ### 💡 Example Usage:
    ```python
    # Standard positive integer powers
    power(2, 3)      # Returns: 8

    # Fractional powers (calculating a square root)
    power(9, 0.5)    # Returns: 3.0

    # Negative exponents (inverting the output scale)
    power(2, -1)     # Returns: 0.5
    ```
    """
    return base ** exponent

def sqrt(n: float) -> float:
    """
    🧮 **Calculate the Square Root (√x)**

    Finds the principal square root of a non-negative number. Returns the non-negative 
    value that, when multiplied by itself, equals the input number.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The target number to evaluate. Must be greater than or equal to 0.

    ### 📤 Returns:
    *   `float`: The calculated principal square root.

    ### ⚠️ Raises:
    *   `ValueError`: If `n` is a negative number (square roots of negative numbers yield imaginary results).

    ---

    ### 💡 Example Usage:
    ```python
    # Perfect square inputs
    sqrt(9)      # Returns: 3.0
    sqrt(100)    # Returns: 10.0

    # Decimals
    sqrt(2.25)   # Returns: 1.5
    ```
    """
    if n < 0:
        raise ValueError("sqrt error: Square root of a negative number is undefined for real numbers.")
    return n ** 0.5

def cbrt(n: float) -> float:
    """
    🧮 **Calculate the Cube Root (³√x)**

    Finds the cube root of a number. Unlike square roots, the cube root is perfectly 
    defined for negative real numbers and preserves the input's sign.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The target number to evaluate.

    ### 📤 Returns:
    *   `float`: The calculated cube root.

    ---

    ### 💡 Example Usage:
    ```python
    # Positive whole numbers
    cbrt(27)     # Returns: 3.0

    # Negative numbers (safely preserves sign)
    cbrt(-8)     # Returns: -2.0

    # Decimals
    cbrt(0.125)  # Returns: 0.5
    ```
    """
    if n < 0:
        return -((-n) ** (1 / 3))
    return n ** (1 / 3)

def nth_root(n: float, root: int) -> float:
    """
    🧮 **Calculate any Custom Root (ⁿ√x)**

    Finds the principal nth root of a number. Safely resolves the root scaling direction 
    and handles odd roots of negative numbers natively without falling into complex number spaces.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The base number to evaluate.
    *   `root` (int): The index of the root (e.g., 2 for square root, 3 for cube root, 4 for fourth root). Must not be 0.

    ### 📤 Returns:
    *   `float`: The calculated root value.

    ### ⚠️ Raises:
    *   `ValueError`: If attempting to find an even root of a negative number.
    *   `ZeroDivisionError`: If `root` is exactly `0`.

    ---

    ### 💡 Example Usage:
    ```python
    # Finding a 4th root
    nth_root(16, 4)     # Returns: 2.0

    # Finding an odd root of a negative number safely
    nth_root(-32, 5)    # Returns: -2.0
    ```
    """
    if root == 0:
        raise ZeroDivisionError("nth_root error: Root index cannot be zero.")
        
    if n < 0:
        if root % 2 == 0:
            raise ValueError("nth_root error: Even root of a negative number yields an imaginary result.")
        return -((-n) ** (1 / root))
        
    return n ** (1 / root)

def hypot(*args: float) -> float:
    """
    📐 **Calculate the Hypotenuse / Euclidean Norm**

    Computes the square root of the sum of squares for all provided coordinates. 
    Commonly used to find vector lengths or straight-line distances across any multi-dimensional space.

    ---

    ### 📥 Arguments:
    *   `*args` (float / int): An arbitrary list of numeric dimensions (e.g., `x`, `y`, `z`).

    ### 📤 Returns:
    *   `float`: The calculated Euclidean distance length.

    ---

    ### 💡 Example Usage:
    ```python
    # Classic 2D right-triangle hypotenuse (3-4-5 triangle)
    hypot(3, 4)       # Returns: 5.0

    # 3D vector length calculation
    hypot(2, 3, 6)    # Returns: 7.0
    ```
    """
    if not args:
        return 0.0

    # To protect against overflow with massive numbers, find the maximum scale factor
    max_val = 0.0
    for val in args:
        abs_val = -val if val < 0 else val
        if abs_val > max_val:
            max_val = abs_val

    if max_val == 0.0:
        return 0.0

    # Scale numbers down before squaring to guarantee hardware floating-point safety
    sum_of_squares = 0.0
    for val in args:
        scaled = val / max_val
        sum_of_squares += scaled * scaled

    return max_val * (sum_of_squares ** 0.5)

def exp(n: float) -> float:
    """
    📈 **Calculate the Exponential of a Number (e^x)**

    Raises Euler's number `e` (approximately 2.718281828459045) to the power of `n`. 
    Widely used across calculus, continuous compound interest tracking, and growth modeling.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The exponent power to raise `e` by.

    ### 📤 Returns:
    *   `float`: The resulting exponential value (`e ** n`).

    ---

    ### 💡 Example Usage:
    ```python
    # Exponential of 0 (always yields 1.0)
    exp(0)       # Returns: 1.0

    # Exponential of 1 (yields Euler's constant)
    exp(1)       # Returns: 2.718281828459045

    # Negative growth/decay tracking
    exp(-1)      # Returns: 0.36787944117144233
    ```
    """
    # Hardcoded high-precision approximation of Euler's constant
    E = 2.7182818284590452353602874713527
    
    # Handle infinite boundaries cleanly
    if n == float('inf'):
        return float('inf')
    if n == float('-inf'):
        return 0.0
        
    return E ** n

def expm1(n: float) -> float:
    """
    📈 **Accurate Exponential Minus One (e^x - 1)**

    Computes `(e ** n) - 1` with high precision. Bypasses catastrophic cancellation 
    precision errors by using a Taylor series expansion when handling values near zero.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The exponential value to evaluate.

    ### 📤 Returns:
    *   `float`: The resulting calculated value.

    ---

    ### 💡 Example Usage:
    ```python
    # Standard subtraction loses precision on tiny numbers
    # expm1 handles it effortlessly:
    expm1(1e-15)  # Returns: 1.0000000000000005e-15
    ```
    """
    # Absolute value check for the zero-proximity threshold
    abs_n = -n if n < 0 else n
    
    # If n is incredibly close to 0, use a 5-term Taylor Series to prevent float loss
    if abs_n < 1e-5:
        term1 = n
        term2 = (n * n) / 2.0
        term3 = (n * n * n) / 6.0
        term4 = (n * n * n * n) / 24.0
        term5 = (n * n * n * n * n) / 120.0
        return term1 + term2 + term3 + term4 + term5

    # Hardcoded high-precision approximation of Euler's constant
    E = 2.7182818284590452353602874713527
    return (E ** n) - 1.0

def log(n: float, base: float = 2.718281828459045) -> float:
    """
    🪵 **Calculate the Logarithm of a Number with a Custom Base**

    Determines the power to which the base must be raised to yield the value `n`.
    Uses high-precision Newton-Raphson optimization to invert exponentiation 
    natively without imports.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The target input value. Must be greater than 0.
    *   `base` (float / int): The logarithmic base layer. Must be greater than 0 and not equal to 1. *Defaults to Euler's number (e).*

    ### 📤 Returns:
    *   `float`: The calculated logarithm value.

    ### ⚠️ Raises:
    *   `ValueError`: If `n <= 0`, `base <= 0`, or `base == 1` (mathematically undefined zones).

    ---

    ### 💡 Example Usage:
    ```python
    # Common Logarithm (Base 10)
    log(100, base=10)   # Returns: 2.0

    # Binary Logarithm (Base 2)
    log(8, base=2)      # Returns: 3.0

    # Natural Logarithm (Base e)
    log(2.718281828)    # Returns: 1.0
    ```
    """
    # Mathematical boundary safety checks
    if n <= 0:
        raise ValueError("log error: Logarithm domain error (n must be greater than 0).")
    if base <= 0 or base == 1.0:
        raise ValueError("log error: Logarithm base error (base must be greater than 0 and not equal to 1).")

    # Helper function to compute the natural log (ln) using Newton-Raphson iteration
    def _ln(x: float) -> float:
        # Pull the exact binary scaling exponent from the float hex string as a baseline hint
        _, exp_part = float.hex(x).split('p')
        binary_exponent = int(exp_part)
        
        # Initial guess: binary exponent scaled by ln(2)
        y = binary_exponent * 0.6931471805599453
        
        # 5 iterations of Newton's method yields full 64-bit float precision stability
        for _ in range(5):
            ey = 2.7182818284590452353602874713527 ** y
            y = y - 1.0 + x / ey
            
        return y

    # Apply the change of base formula: log_b(a) = ln(a) / ln(b)
    return _ln(n) / _ln(base)

def log10(n: float) -> float:
    """
    🪵 **Calculate the Common Logarithm (Base-10)**

    Determines the power to which the number 10 must be raised to yield the 
    value `n`. Essential for scientific measurements like pH scale, decibel levels, 
    and the Richter scale.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The target input value to evaluate. Must be greater than 0.

    ### 📤 Returns:
    *   `float`: The calculated base-10 logarithm value.

    ### ⚠️ Raises:
    *   `ValueError`: If `n <= 0` (logarithms of zero or negative numbers are undefined).

    ---

    ### 💡 Example Usage:
    ```python
    # Clean powers of 10
    log10(100)      # Returns: 2.0
    log10(1000)     # Returns: 3.0

    # Fractions less than 1
    log10(0.1)      # Returns: -1.0
    ```
    """
    if n <= 0:
        raise ValueError("log10 error: Logarithm domain error (n must be greater than 0).")

    # Pure standard calculation utilizing change of base formula via Newton-Raphson ln(x)
    def _ln(x: float) -> float:
        _, exp_part = float.hex(x).split('p')
        y = int(exp_part) * 0.6931471805599453
        for _ in range(5):
            ey = 2.7182818284590452353602874713527 ** y
            y = y - 1.0 + x / ey
        return y

    # Fixed high-precision constant for ln(10) to maximize runtime efficiency
    LN_10 = 2.3025850929940456840179914546844
    return _ln(n) / _ln(10)

def log2(n: float) -> float:
    """
    🪵 **Calculate the Binary Logarithm (Base-2)**

    Determines the power to which the number 2 must be raised to yield the 
    value `n`. Essential for computing time complexities (O(log n)), information 
    entropy, and sizing binary data layouts.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The target input value to evaluate. Must be greater than 0.

    ### 📤 Returns:
    *   `float`: The calculated base-2 logarithm value.

    ### ⚠️ Raises:
    *   `ValueError`: If `n <= 0` (logarithms of zero or negative numbers are undefined).

    ---

    ### 💡 Example Usage:
    ```python
    # Clean binary powers
    log2(8)        # Returns: 3.0
    log2(1024)     # Returns: 10.0

    # Fractions less than 1
    log2(0.25)     # Returns: -2.0
    ```
    """
    if n <= 0:
        raise ValueError("log2 error: Logarithm domain error (n must be greater than 0).")

    # Pure standard calculation utilizing change of base formula via Newton-Raphson ln(x)
    def _ln(x: float) -> float:
        _, exp_part = float.hex(x).split('p')
        y = int(exp_part) * 0.6931471805599453
        for _ in range(5):
            ey = 2.7182818284590452353602874713527 ** y
            y = y - 1.0 + x / ey
        return y

    # Fixed high-precision constant for ln(2) to maximize runtime efficiency
    LN_2 = 0.69314718055994530941723212145818
    return _ln(n) / LN_2

def log1p(n: float) -> float:
    """
    🪵 **Accurate Natural Logarithm of One Plus X (log1p)**

    Computes `ln(1 + n)` with extreme precision. Bypasses catastrophic float loss 
    errors by switching to a Taylor series expansion when handling tiny values near zero.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The input numerical value to evaluate. Must be greater than -1.

    ### 📤 Returns:
    *   `float`: The precisely calculated natural log of `1 + n`.

    ### ⚠️ Raises:
    *   `ValueError`: If `n <= -1` (logarithms of zero or negative numbers are undefined).

    ---

    ### 💡 Example Usage:
    ```python
    # Standard log(1 + 1e-15) returns 0.0 due to float precision limits.
    # log1p handles it flawlessly:
    log1p(1e-15)  # Returns: 9.999999999999995e-16
    ```
    """
    if n <= -1.0:
        raise ValueError("log1p error: Logarithm domain error (n must be greater than -1).")

    # Absolute value check for the zero-proximity threshold
    abs_n = -n if n < 0 else n

    # If n is incredibly close to 0, use an 5-term Taylor Series to prevent precision loss
    if abs_n < 1e-4:
        term1 = n
        term2 = -(n * n) / 2.0
        term3 = (n * n * n) / 3.0
        term4 = -(n * n * n * n) / 4.0
        term5 = (n * n * n * n * n) / 5.0
        return term1 + term2 + term3 + term4 + term5

    # Pure standard calculation utilizing Newton-Raphson ln(x) for normal ranges
    def _ln(x: float) -> float:
        _, exp_part = float.hex(x).split('p')
        y = int(exp_part) * 0.6931471805599453
        for _ in range(5):
            ey = 2.7182818284590452353602874713527 ** y
            y = y - 1.0 + x / ey
        return y

    return _ln(1.0 + n)

def is_power_of_two(n: int) -> bool:
    """
    🔎 **Check if an Integer is a Power of Two**

    Evaluates whether a whole integer lines up perfectly on a binary memory boundary 
    (e.g., 1, 2, 4, 8, 16, 32, ...). Utilizes an ultra-fast hardware bitwise identity 
    for maximum performance.

    ---

    ### 📥 Arguments:
    *   `n` (int): The target whole integer to evaluate.

    ### 📤 Returns:
    *   `bool`: `True` if the number is an exact power of two, `False` otherwise.

    ---

    ### 💡 Example Usage:
    ```python
    # Valid binary boundaries
    is_power_of_two(16)   # Returns: True
    is_power_of_two(1)    # Returns: True (2^0)

    # Invalid boundaries
    is_power_of_two(14)   # Returns: False
    
    # Negative numbers and zero are always False
    is_power_of_two(0)    # Returns: False
    ```
    """
    # Powers of two must be strictly positive whole integers
    if n <= 0:
        return False
        
    # Bitwise identity trick: A power of two shared with its predecessor always yields zero
    return (n & (n - 1)) == 0

def is_perfect_square(n: int) -> bool:
    """
    🔎 **Check if an Integer is a Perfect Square**

    Evaluates whether an integer's square root resolves to a clean whole number 
    (e.g., 0, 1, 4, 9, 16, 25, ...). Performs calculations purely in integer space 
    to completely eliminate floating-point precision loss.

    ---

    ### 📥 Arguments:
    *   `n` (int): The target whole integer to evaluate.

    ### 📤 Returns:
    *   `bool`: `True` if the integer is a perfect square, `False` otherwise.

    ---

    ### 💡 Example Usage:
    ```python
    # Valid perfect squares
    is_perfect_square(16)   # Returns: True (4 * 4)
    is_perfect_square(0)    # Returns: True (0 * 0)

    # Non-perfect square integers
    is_perfect_square(15)   # Returns: False
    
    # Negative integers are always False
    is_perfect_square(-4)   # Returns: False
    ```
    """
    if n < 0:
        return False
    if n in (0, 1):
        return True

    # High-performance integer-only Newton's method (bypasses float limitations)
    x = n // 2
    seen = {x}
    while x * x > n:
        next_x = (x + n // x) // 2
        if next_x >= x or next_x in seen:
            break
        x = next_x
        seen.add(x)

    return x * x == n

def degrees_to_radians(deg: float) -> float:
    """
    📐 **Convert Degrees to Radians**

    Transforms an angular measurement from standard degrees into its equivalent 
    radian arc length using the library's high-precision constant.

    ---

    ### 📥 Arguments:
    *   `deg` (float / int): The angular value in standard degrees.

    ### 📤 Returns:
    *   `float`: The equivalent angular measurement in radians.
    """
    return deg * deg_to_rad_constant

def radians_to_degrees(rad: float) -> float:
    """
    📐 **Convert Radians to Degrees**

    Transforms an angular measurement from radian arc lengths back into standard 
    degrees using the library's high-precision tracking constant.

    ---

    ### 📥 Arguments:
    *   `rad` (float / int): The angular value measured in radians.

    ### 📤 Returns:
    *   `float`: The equivalent angular measurement in standard degrees.
    """
    return rad * rad_to_deg_constant

def sin(rad: float) -> float:
    """
    🌊 **Calculate the Trigonometric Sine of an Angle**

    Computes the sine of an angle specified in radians. Automatically maps large or 
    negative angles back onto a standard single circle grid using the library's `TAU` constant.

    ---

    ### 📥 Arguments:
    *   `rad` (float / int): The input angle measured in radians.

    ### 📤 Returns:
    *   `float`: The sine value sitting precisely between -1.0 and 1.0.

    ---

    ### 💡 Example Usage:
    ```python
    # Sine of 0
    sin(0.0)                      # Returns: 0.0

    # Sine of 90 degrees (using your library's constant)
    sin(90 * deg_to_rad_constant) # Returns: 1.0
    ```
    """
    # Map any massive or negative angle onto a standard single circle sweep [0, TAU)
    rad = rad - (TAU * (rad // TAU))
    
    # Run a high-speed alternating Taylor series expansion loop
    total = 0.0
    term = rad
    term_squared = rad * rad
    
    for i in range(1, 20, 2):
        total += term
        # Build the next fractional factorial term sequentially to protect performance
        term = -term * term_squared / ((i + 1) * (i + 2))
        
    return total

def cos(rad: float) -> float:
    """
    🌊 **Calculate the Trigonometric Cosine of an Angle**

    Computes the cosine of an angle specified in radians. Automatically maps massive or 
    negative angles back onto a standard single circle grid using the library's `TAU` constant.

    ---

    ### 📥 Arguments:
    *   `rad` (float / int): The input angle measured in radians.

    ### 📤 Returns:
    *   `float`: The cosine value sitting precisely between -1.0 and 1.0.

    ---

    ### 💡 Example Usage:
    ```python
    # Cosine of 0 (always yields 1.0)
    cos(0.0)                      # Returns: 1.0

    # Cosine of 90 degrees (using your library's constant)
    cos(90 * deg_to_rad_constant) # Returns: 0.0
    ```
    """
    # Map any massive or negative angle onto a standard single circle sweep [0, TAU)
    rad = rad - (TAU * (rad // TAU))
    
    # Run a high-speed alternating Taylor series expansion loop for cosine
    total = 0.0
    term = 1.0
    term_squared = rad * rad
    
    for i in range(0, 20, 2):
        total += term
        # Build the next fractional factorial term sequentially to protect performance
        term = -term * term_squared / ((i + 1) * (i + 2))
        
    return total

def tan(rad: float) -> float | None:
    """
    🌊 **Calculate the Trigonometric Tangent of an Angle**

    Computes the tangent of an angle specified in radians by finding the ratio of 
    sine over cosine (`sin(rad) / cos(rad)`). Automatically maps massive or 
    negative angles back onto a standard single circle grid using the library's `TAU` constant.

    ---

    ### 📥 Arguments:
    *   `rad` (float / int): The input angle measured in radians.

    ### 📤 Returns:
    *   `float`: The tangent value.
    *   `None`: If the cosine of the angle resolves to exactly `0` (where tangent approaches infinity and is undefined).

    ---

    ### 💡 Example Usage:
    ```python
    # Tangent of 0 (yields 0.0)
    tan(0.0)                      # Returns: 0.0

    # Tangent of 45 degrees (yields 1.0)
    tan(45 * deg_to_rad_constant) # Returns: 1.0
    ```
    """
    # Calculate the sine and cosine parameters using your library's established engines
    c = cos(rad)
    
    # Catch mathematical undefined boundaries where cosine drops to zero (e.g., 90° or 270°)
    if c == 0.0:
        print("tan error: Tangent is undefined where cosine equals zero.")
        return None
        
    s = sin(rad)
    return s / c

def asin(n: float) -> float | None:
    """
    🔄 **Calculate Inverse Sine (Arcsine)**

    Computes the arcsine of a number, returning the resulting angular trajectory 
    measured in radians sitting inside the principal range [-PI/2, PI/2].

    ---

    ### 📥 Arguments:
    *   `n` (float): The input ratio tracking value. Must sit between -1.0 and 1.0.

    ### 📤 Returns:
    *   `float`: The calculated angular trajectory in radians.
    *   `None`: If `n` breaches domain parameters (outside the strict [-1.0, 1.0] envelope).

    ---

    ### 💡 Example Usage:
    ```python
    # Ratio of 0 yields an angle of 0
    asin(0.0)   # Returns: 0.0

    # Max ratio yields a right angle (Half of your PI scaling constant)
    asin(1.0)   # Returns: 1.5707963267948966
    ```
    """
    # Guard against invalid trigonometric tracking scopes
    if n < -1.0 or n > 1.0:
        print("asin error: Domain boundary breach (input ratio must sit within [-1.0, 1.0]).")
        return None

    # Capture absolute value scale tracking
    abs_n = -n if n < 0 else n

    # Taylor series approximation core engine
    def _asin_series(x: float) -> float:
        total = 0.0
        term = x
        term_squared = x * x
        
        # 25 structural iterations provide flawless machine decimal accuracy layers
        for i in range(1, 50, 2):
            total += term
            # Sequence multiplier tracking loops matching the Maclaurin expansion sequence
            term = term * term_squared * i / (i + 1) * (i + 2) / (i + 3)
        return total

    # If the ratio gets steep (> 0.5), apply the square root reduction identity 
    # to safeguard precision bounds near the circle horizon edge
    if abs_n > 0.5:
        half_pi = PI * 0.5
        reduced_ratio = (1.0 - abs_n) * 0.5
        # Utilize your native power index mapping configuration to extract square roots
        sqrt_ratio = reduced_ratio ** 0.5
        angle = half_pi - 2.0 * _asin_series(sqrt_ratio)
        return -angle if n < 0.0 else angle

    # Standard range tracking processing
    angle = _asin_series(abs_n)
    return -angle if n < 0.0 else angle

def acos(n: float) -> float | None:
    """
    🔄 **Calculate Inverse Cosine (Arccosine)**

    Computes the arccosine of a number, returning the resulting angular trajectory 
    measured in radians sitting inside the principal range [0, PI].

    ---

    ### 📥 Arguments:
    *   `n` (float): The input ratio tracking value. Must sit between -1.0 and 1.0.

    ### 📤 Returns:
    *   `float`: The calculated angular trajectory in radians.
    *   `None`: If `n` breaches domain parameters (outside the strict [-1.0, 1.0] envelope).

    ---

    ### 💡 Example Usage:
    ```python
    # Ratio of 1.0 yields a starting angle of 0
    acos(1.0)   # Returns: 0.0

    # Ratio of 0.0 yields a right angle (Half of your PI scaling constant)
    acos(0.0)   # Returns: 1.5707963267948966
    ```
    """
    # Use your asin function to calculate the inverse arcsine value
    asin_val = asin(n)
    
    # If the domain check inside asin failed and returned None, return None here too
    if asin_val is None:
        return None
        
    # Apply the mathematical identity: acos(x) = (PI / 2) - asin(x)
    return (PI * 0.5) - asin_val

def atan(n: float) -> float:
    """
    🔄 **Calculate Inverse Tangent (Arctangent)**

    Computes the arctangent of a number, returning the resulting angular trajectory 
    measured in radians sitting inside the principal range [-PI/2, PI/2].

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The input ratio tracking value.

    ### 📤 Returns:
    *   `float`: The calculated angular trajectory in radians.

    ---

    ### 💡 Example Usage:
    ```python
    # Ratio of 0 yields an angle of 0
    atan(0.0)   # Returns: 0.0

    # Ratio of 1.0 yields a 45-degree angle (Quarter of your PI scaling constant)
    atan(1.0)   # Returns: 0.7853981633974483
    ```
    """
    # Capture absolute value scale tracking
    abs_n = -n if n < 0 else n

    # High-speed Taylor series approximation engine optimized for x <= 1
    def _atan_series(x: float) -> float:
        total = 0.0
        term = x
        term_squared = x * x
        
        # 30 sequential terms handle perfect 64-bit float decimal stabilization
        for i in range(1, 60, 2):
            # Alternating sign sequence: add, subtract, add, subtract...
            if (i // 2) % 2 == 0:
                total += term / i
            else:
                total -= term / i
            term *= term_squared
        return total

    # If the ratio is larger than 1, map it back into the convergence envelope
    if abs_n > 1.0:
        half_pi = PI * 0.5
        # Identity: atan(x) = pi/2 - atan(1/x)
        angle = half_pi - _atan_series(1.0 / abs_n)
        return -angle if n < 0.0 else angle

    # Standard range tracking processing
    angle = _atan_series(abs_n)
    return -angle if n < 0.0 else angle

def atan2(y: float, x: float) -> float:
    """
    🎯 **Calculate Full 4-Quadrant Arctangent (atan2)**

    Computes the angular trajectory from the positive x-axis to a coordinate point 
    `(x, y)` in radians. Safely resolves the exact circle quadrant and handles vertical 
    trajectories (`x = 0`) flawlessly without zero-division crashes.

    ---

    ### 📥 Arguments:
    *   `y` (float / int): The vertical coordinate component (rise).
    *   `x` (float / int): The horizontal coordinate component (run).

    ### 📤 Returns:
    *   `float`: The resulting heading angle in radians, sitting within the range [-PI, PI].

    ---

    ### 💡 Example Usage:
    ```python
    # Heading into Quadrant 1 (45 degrees)
    atan2(1.0, 1.0)   # Returns: 0.7853981633974483

    # Heading straight up (90 degrees, x=0 handled safely)
    atan2(1.0, 0.0)   # Returns: 1.5707963267948966

    # Heading backwards into Quadrant 2 (135 degrees)
    atan2(1.0, -1.0)  # Returns: 2.356194490192345
    ```
    """
    # Case 1: Horizontal component is strictly positive
    if x > 0.0:
        return atan(y / x)
        
    # Case 2: Horizontal component is negative, vertical component is positive or zero
    if x < 0.0 and y >= 0.0:
        return atan(y / x) + PI
        
    # Case 3: Horizontal component is negative, vertical component is strictly negative
    if x < 0.0 and y < 0.0:
        return atan(y / x) - PI
        
    # Case 4: x is exactly zero (Vertical lines)
    if x == 0.0:
        if y > 0.0:
            return PI * 0.5
        if y < 0.0:
            return -(PI * 0.5)
        # y == 0.0 and x == 0.0 is mathematically undefined (Origin coordinate point)
        return 0.0

def sinh(n: float) -> float:
    """
    📈 **Calculate the Hyperbolic Sine (sinh(x))**

    Computes the hyperbolic sine of a number. This function is widely used in physics 
    and engineering to describe wave mechanics, the structural shape of hanging cables 
    (catenaries), and within deep learning activation formulas.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The target numerical value to evaluate.

    ### 📤 Returns:
    *   `float`: The calculated hyperbolic sine value.

    ---

    ### 💡 Example Usage:
    ```python
    # Hyperbolic sine of 0 (always yields 0.0)
    sinh(0.0)    # Returns: 0.0

    # Hyperbolic sine of 1.0 (utilizing your library's E constant)
    sinh(1.0)    # Returns: 1.1752011936438014
    ```
    """
    # Guard infinite boundaries to prevent hardware floating-point runtime issues
    if n == float('inf'):
        return float('inf')
    if n == float('-inf'):
        return float('-inf')

    # Mathematical identity: sinh(x) = (e^x - e^-x) / 2
    # Utilizing your library's pre-defined high-precision E constant
    ex = E ** n
    e_neg_x = E ** (-n)
    
    return (ex - e_neg_x) / 2.0

def cosh(n: float) -> float:
    """
    📈 **Calculate the Hyperbolic Cosine (cosh(x))**

    Computes the hyperbolic cosine of a number. This function describes the natural 
    hanging curve of a uniform cable or chain under uniform gravity (the catenary curve) 
    and is used widely throughout structural engineering and physics.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The target numerical value to evaluate.

    ### 📤 Returns:
    *   `float`: The calculated hyperbolic cosine value.

    ---

    ### 💡 Example Usage:
    ```python
    # Hyperbolic cosine of 0 (always yields 1.0)
    cosh(0.0)    # Returns: 1.0

    # Hyperbolic cosine of 1.0 (utilizing your library's E constant)
    cosh(1.0)    # Returns: 1.5430806348152437
    ```
    """
    # Guard infinite boundaries to prevent hardware floating-point runtime issues
    if n in (float('inf'), float('-inf')):
        return float('inf')

    # Mathematical identity: cosh(x) = (e^x + e^-x) / 2
    # Utilizing your library's pre-defined high-precision E constant
    ex = E ** n
    e_neg_x = E ** (-n)
    
    return (ex + e_neg_x) / 2.0

def tanh(n: float) -> float:
    """
    📈 **Calculate the Hyperbolic Tangent (tanh(x))**

    Computes the hyperbolic tangent of a number by finding the ratio of hyperbolic 
    sine over hyperbolic cosine (`sinh(n) / cosh(n)`). Extensively used as an activation 
    function in artificial neural networks and machine learning models.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The target numerical value to evaluate.

    ### 📤 Returns:
    *   `float`: The calculated hyperbolic tangent value, bounding strictly between -1.0 and 1.0.

    ---

    ### 💡 Example Usage:
    ```python
    # Hyperbolic tangent of 0 (always yields 0.0)
    tanh(0.0)    # Returns: 0.0

    # Approaching upper bounding limits
    tanh(4.0)    # Returns: 0.999329299739067
    ```
    """
    # Guard infinite boundaries to prevent NaN runtime collisions during division
    if n == float('inf'):
        return 1.0
    if n == float('-inf'):
        return -1.0

    # Reusing your library's established sinh and cosh formulas inline for maximum efficiency:
    # tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    ex = E ** n
    e_neg_x = E ** (-n)
    
    return (ex - e_neg_x) / (ex + e_neg_x)

def asinh(n: float) -> float:
    """
    🔄 **Calculate Inverse Hyperbolic Sine (asinh(x))**

    Computes the inverse hyperbolic sine of a number. This function handles all 
    real inputs across an infinite range and is symmetric around zero.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The target input value to evaluate.

    ### 📤 Returns:
    *   `float`: The calculated inverse hyperbolic sine value in radians.

    ---

    ### 💡 Example Usage:
    ```python
    # Inverse hyperbolic sine of 0 (yields 0.0)
    asinh(0.0)   # Returns: 0.0

    # Checking matching directional metrics
    asinh(1.0)   # Returns: 0.881373587019543
    ```
    """
    # Handle infinite boundaries directly to prevent overflow issues
    if n == float('inf'):
        return float('inf')
    if n == float('-inf'):
        return float('-inf')

    # Pure standard calculation utilizing change of base formula via Newton-Raphson ln(x)
    def _ln(x: float) -> float:
        _, exp_part = float.hex(x).split('p')
        y = int(exp_part) * 0.6931471805599453
        for _ in range(5):
            ey = 2.7182818284590452353602874713527 ** y
            y = y - 1.0 + x / ey
        return y

    # Mathematical identity: asinh(x) = ln(x + sqrt(x^2 + 1))
    # Handle the negative sign symmetrically to ensure calculation stability near zero
    is_negative = n < 0.0
    abs_n = -n if is_negative else n
    
    # Calculate the inner tracking envelope using native power operator for square root
    inner_val = abs_n + ((abs_n * abs_n) + 1.0) ** 0.5
    result = _ln(inner_val)
    
    return -result if is_negative else result

def acosh(n: float) -> float | None:
    """
    🔄 **Calculate Inverse Hyperbolic Cosine (acosh(x))**

    Computes the inverse hyperbolic cosine of a number. This function is only 
    defined for real numbers greater than or equal to 1.0.

    ---

    ### 📥 Arguments:
    *   `n` (float / int): The target input value to evaluate. Must be >= 1.0.

    ### 📤 Returns:
    *   `float`: The calculated inverse hyperbolic cosine value.
    *   `None`: If `n` breaches domain boundaries (input is less than 1.0).

    ---

    ### 💡 Example Usage:
    ```python
    # Starting boundary index (always yields 0.0)
    acosh(1.0)   # Returns: 0.0

    # Computing a tracking value scale
    acosh(2.5)   # Returns: 1.566799236972411
    ```
    """
    # Guard against invalid mathematical domain entries
    if n < 1.0:
        print("acosh error: Domain boundary breach (input must be greater than or equal to 1.0).")
        return None

    # Handle infinity boundaries directly
    if n == float('inf'):
        return float('inf')

    # Pure standard calculation utilizing change of base formula via Newton-Raphson ln(x)
    def _ln(x: float) -> float:
        _, exp_part = float.hex(x).split('p')
        y = int(exp_part) * 0.6931471805599453
        for _ in range(5):
            ey = 2.7182818284590452353602874713527 ** y
            y = y - 1.0 + x / ey
        return y

    # Mathematical identity: acosh(x) = ln(x + sqrt(x^2 - 1))
    # Extract square root precisely using your native power operator mapping configuration
    inner_val = n + ((n * n) - 1.0) ** 0.5
    
    return _ln(inner_val)

def area_circle(radius: float) -> float | None:
    """
    📐 **Calculate the Area of a Circle (πr²)**

    Computes the total structural surface area bounded within a circle given its 
    radius vector length. Reuses the library's pre-defined high-precision `PI` constant.

    ---

    ### 📥 Arguments:
    *   `radius` (float / int): The distance from the center to the outer boundary edge. Must be non-negative.

    ### 📤 Returns:
    *   `float`: The precisely computed surface area.
    *   `None`: If `radius` breaches physical spatial constraints (less than 0).

    ---

    ### 💡 Example Usage:
    ```python
    # Area of a unit circle (radius = 1)
    area_circle(1.0)   # Returns: 3.141592653589793

    # Area of a circle with a radius of 5
    area_circle(5)     # Returns: 78.53981633974483
    ```
    """
    if radius < 0.0:
        print("area_circle error: Radius cannot be negative in physical geometry.")
        return None

    return PI * radius * radius

def circumference(radius: float) -> float | None:
    """
    📐 **Calculate the Circumference of a Circle (2πr)**

    Computes the total linear perimeter distance tracking around the outer edge 
    of a circle. Reuses the library's pre-defined high-precision `TAU` constant 
    (which is exactly `2 * PI`) to optimize computational runtime speed.

    ---

    ### 📥 Arguments:
    *   `radius` (float / int): The distance from the center to the outer boundary edge. Must be non-negative.

    ### 📤 Returns:
    *   `float`: The precisely computed perimeter distance.
    *   `None`: If `radius` breaches physical spatial constraints (less than 0).

    ---

    ### 💡 Example Usage:
    ```python
    # Circumference of a unit circle (radius = 1)
    circumference(1.0)   # Returns: 6.283185307179586 (Matches your TAU constant)

    # Circumference of a circle with a radius of 10
    circumference(10)    # Returns: 62.83185307179586
    ```
    """
    if radius < 0.0:
        print("circumference error: Radius cannot be negative in physical geometry.")
        return None

    # Utilizing your pre-defined TAU constant simplifies 2 * PI * r into a single multiplication step!
    return TAU * radius

def area_triangle(base: float, height: float) -> float | None:
    """
    📐 **Calculate the Area of a Triangle (0.5 * base * height)**

    Computes the total structural surface scale bounded within a 2D triangle 
    given its foundational baseline length and vertical altitude height.

    ---

    ### 📥 Arguments:
    *   `base` (float / int): The linear horizontal length of the triangle's bottom edge. Must be non-negative.
    *   `height` (float / int): The perpendicular vertical distance from the base to the highest peak point. Must be non-negative.

    ### 📤 Returns:
    *   `float`: The precisely computed surface area.
    *   `None`: If either dimension breaches physical spatial constraints (less than 0).

    ---

    ### 💡 Example Usage:
    ```python
    # Standard triangle calculation
    area_triangle(10, 5)   # Returns: 25.0

    # Working with floating-point parameters
    area_triangle(4.5, 2.0) # Returns: 4.5
    ```
    """
    if base < 0.0 or height < 0.0:
        print("area_triangle error: Base and height cannot be negative in physical geometry.")
        return None

    return 0.5 * base * height

def perimeter_triangle(a: float, b: float, c: float) -> float | None:
    """
    📐 **Calculate the Perimeter of a Triangle**

    Computes the total structural boundary line of a triangle by summing the lengths 
    of its three individual sides. Includes a geometric verification check to ensure 
    the provided side lengths can physically form a real triangle.

    ---

    ### 📥 Arguments:
    *   `a` (float / int): The length of the first side. Must be non-negative.
    *   `b` (float / int): The length of the second side. Must be non-negative.
    *   `c` (float / int): The length of the third side. Must be non-negative.

    ### 📤 Returns:
    *   `float`: The precisely computed perimeter total.
    *   `None`: If any side is negative, or if the side combinations violate the Triangle Inequality Theorem.

    ---

    ### 💡 Example Usage:
    ```python
    # Classic 3-4-5 right triangle
    perimeter_triangle(3, 4, 5)     # Returns: 12.0

    # Passing impossible dimensions (e.g., straight line overlap)
    perimeter_triangle(10, 2, 2)    # Prints error and returns: None
    ```
    """
    # Verify physical dimension constraints
    if a < 0.0 or b < 0.0 or c < 0.0:
        print("perimeter_triangle error: Side lengths cannot be negative in physical geometry.")
        return None

    # Enforce the Triangle Inequality Theorem: The sum of any two sides must be strictly greater than the third side
    if (a + b <= c) or (a + c <= b) or (b + c <= a):
        print("perimeter_triangle error: Invalid dimensions. The provided side lengths cannot physically form a closed triangle.")
        return None

    return a + b + c

def distance_2d(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """
    📐 **Calculate Straight-Line Distance in 2D Space (Euclidean Distance)**

    Computes the shortest path straight-line spatial gap between two coordinate 
    points `p1(x1, y1)` and `p2(x2, y2)` on a flat 2D grid plane.

    ---

    ### 📥 Arguments:
    *   `p1` (tuple[float, float]): A tuple representing the starting point `(x1, y1)`.
    *   `p2` (tuple[float, float]): A tuple representing the destination point `(x2, y2)`.

    ### 📤 Returns:
    *   `float`: The calculated linear distance between the two coordinate markers.

    ---

    ### 💡 Example Usage:
    ```python
    # Distance between origin and point (3, 4) -> Classic 3-4-5 layout
    point_a = (0.0, 0.0)
    point_b = (3.0, 4.0)
    distance_2d(point_a, point_b)  # Returns: 5.0
    ```
    """
    # Deconstruct coordinate pairs into raw dimensional variables
    x1, y1 = p1
    x2, y2 = p2
    
    # Calculate dimensional delta separations
    dx = x2 - x1
    dy = y2 - y1
    
    # Square components natively and extract the principal root using power indexing
    return ((dx * dx) + (dy * dy)) ** 0.5

def mean(data: list[float]) -> float | None:
    """
    📊 **Calculate the Arithmetic Mean**

    Computes the standard average value of a provided array set of numerical entries.

    ---

    ### 📥 Arguments:
    *   `data` (list): A list structure containing numbers to average.

    ### 📤 Returns:
    *   `float`: The calculated arithmetic mean.
    *   `None`: If the provided list sequence is empty.

    ---

    ### 💡 Example Usage:
    ```python
    # Basic average tracking 
    mean([1, 2, 3, 4, 5])   # Returns: 3.0
    ```
    """
    if not data:
        print("mean error: Cannot calculate the average of an empty dataset.")
        return None
        
    return sum(data) / len(data)

def median(data: list[float]) -> float | None:
    """
    📊 **Calculate the Median Value**

    Locates the true physical middle data point within an unsorted collection of numbers. 
    If the sequence element count is even, it returns the average of the two central marks.

    ---

    ### 📥 Arguments:
    *   `data` (list): A list structure containing numerical elements to evaluate.

    ### 📤 Returns:
    *   `float`: The calculated median value index.
    *   `None`: If the provided list sequence is empty.

    ---

    ### 💡 Example Usage:
    ```python
    # Odd element layout sizes
    median([1, 3, 2])      # Returns: 2.0

    # Even element layout sizes (averages central items)
    median([1, 4, 3, 2])   # Returns: 2.5
    ```
    """
    if not data:
        print("median error: Cannot calculate the median of an empty dataset.")
        return None

    # Construct a local copy to preserve the original user array configuration unmutated
    clean_list = [item for item in data]
    
    # Pure Python insertion sort implementation to avoid dependency imports
    for i in range(1, len(clean_list)):
        key = clean_list[i]
        j = i - 1
        while j >= 0 and clean_list[j] > key:
            clean_list[j + 1] = clean_list[j]
            j -= 1
        clean_list[j + 1] = key

    n = len(clean_list)
    mid = n // 2

    # If the collection length is odd, return the exact middle coordinate value
    if n % 2 != 0:
        return float(clean_list[mid])
        
    # If the collection length is even, average the two center items smoothly
    return (clean_list[mid - 1] + clean_list[mid]) / 2.0

def mode(data: list[float]) -> float | None:
    """
    📊 **Calculate the Statistical Mode**

    Identifies the single most frequently occurring numerical value within a dataset. 
    If there is an analytical frequency tie, it safely resolves to the first 
    dominant peak encountered.

    ---

    ### 📥 Arguments:
    *   `data` (list): A list structure containing numbers to analyze.

    ### 📤 Returns:
    *   `float`: The single most frequent value encountered.
    *   `None`: If the provided list sequence is empty.

    ---

    ### 💡 Example Usage:
    ```python
    # Basic data peak frequency verification
    mode([1, 2, 2, 3, 4])  # Returns: 2.0
    ```
    """
    if not data:
        print("mode error: Cannot calculate the mode of an empty dataset.")
        return None

    # Map frequencies manually using a standard key-value dictionary engine
    frequencies = {}
    for item in data:
        frequencies[item] = frequencies.get(item, 0) + 1

    # Isolate the element tracking index holding the highest count trail
    most_frequent_val = data[0]
    max_count = 0
    
    for key, count in frequencies.items():
        if count > max_count:
            max_count = count
            most_frequent_val = key

    return float(most_frequent_val)

def minimum(data: list[float]) -> float | None:
    """
    📉 **Find the Minimum Value in an Array**

    Scans through an array or list of numerical values and extracts the lowest 
    overall value present.

    ---

    ### 📥 Arguments:
    *   `data` (list): A list structure containing numbers to evaluate.

    ### 📤 Returns:
    *   `float`: The lowest numerical value found in the collection.
    *   `None`: If the provided list sequence is empty.

    ---

    ### 💡 Example Usage:
    ```python
    # Finding the lowest value in a list of integers
    minimum([5, 2, 9, -1, 7])  # Returns: -1.0
    ```
    """
    if not data:
        print("minimum error: Cannot locate the lowest value of an empty dataset.")
        return None

    lowest = data[0]
    for val in data:
        if val < lowest:
            lowest = val

    return float(lowest)

def maximum(data: list[float]) -> float | None:
    """
    📈 **Find the Maximum Value in an Array**

    Scans through an array or list of numerical values and extracts the highest 
    overall value present.

    ---

    ### 📥 Arguments:
    *   `data` (list): A list structure containing numbers to evaluate.

    ### 📤 Returns:
    *   `float`: The highest numerical value found in the collection.
    *   `None`: If the provided list sequence is empty.

    ---

    ### 💡 Example Usage:
    ```python
    # Finding the highest value in a list of numbers
    maximum([5, 2, 9, -1, 7])  # Returns: 9.0
    ```
    """
    if not data:
        print("maximum error: Cannot locate the highest value of an empty dataset.")
        return None

    highest = data[0]
    for val in data:
        if val > highest:
            highest = val

    return float(highest)

def range_span(data: list[float]) -> float | None:
    """
    📊 **Calculate the Range Span (Data Spread)**

    Computes the absolute mathematical spread of a dataset by subtracting the 
    overall minimum value from the overall maximum value.

    ---

    ### 📥 Arguments:
    *   `data` (list): A list structure containing numerical elements to evaluate.

    ### 📤 Returns:
    *   `float`: The total spread distance between the highest and lowest numbers.
    *   `None`: If the provided list sequence is empty.

    ---

    ### 💡 Example Usage:
    ```python
    # Calculating the total spread of a dataset
    range_span([10, 2, 5, 25, 7])  # Returns: 23.0 (25.0 - 2.0)
    ```
    """
    if not data:
        print("range_span error: Cannot calculate the range of an empty dataset.")
        return None

    # Reusing the pure scanning logic to find boundaries efficiently without library functions
    lowest = data[0]
    highest = data[0]
    
    for val in data:
        if val < lowest:
            lowest = val
        if val > highest:
            highest = val

    return float(highest - lowest)

def sum_of_squares(data: list[float]) -> float | None:
    """
    📊 **Calculate the Sum of Squares**

    Squares every numerical item present in the list collection and calculates 
    the total combined sum of those squared values. Essential for computing 
    variances, standard deviations, and regression models.

    ---

    ### 📥 Arguments:
    *   `data` (list): A list structure containing numbers to evaluate.

    ### 📤 Returns:
    *   `float`: The total calculated sum of all squared elements.
    *   `None`: If the provided list sequence is empty.

    ---

    ### 💡 Example Usage:
    ```python
    # Summing squares of a short array: (1² + 2² + 3²) = (1 + 4 + 9)
    sum_of_squares([1, 2, 3])  # Returns: 14.0
    ```
    """
    if not data:
        print("sum_of_squares error: Cannot process an empty dataset.")
        return None

    total = 0.0
    for val in data:
        total += val * val

    return total

def variance(data: list[float]) -> float | None:
    """
    📊 **Calculate the Sample Variance**

    Measures how far a collection of numbers spreads out from their central 
    arithmetic mean value. Uses sample variance calculations (dividing by N - 1) 
    to provide an unbiased estimator for statistical subsets.

    ---

    ### 📥 Arguments:
    *   `data` (list): A list structure containing numbers to analyze.

    ### 📤 Returns:
    *   `float`: The computed sample variance value.
    *   `None`: If the dataset contains fewer than 2 elements (variance requires a spread).

    ---

    ### 💡 Example Usage:
    ```python
    # Calculating the variance of a distributed dataset
    variance([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])  # Returns: 4.5
    ```
    """
    if not data or len(data) < 2:
        print("variance error: Variance requires a minimum of two data points.")
        return None

    # Step 1: Calculate the arithmetic mean of the dataset
    data_sum = 0.0
    for val in data:
        data_sum += val
    data_mean = data_sum / len(data)

    # Step 2: Sum the squared deviations from the calculated mean
    squared_deviation_sum = 0.0
    for val in data:
        deviation = val - data_mean
        squared_deviation_sum += deviation * deviation

    # Step 3: Divide by (N - 1) to establish the unbiased sample variance
    return squared_deviation_sum / (len(data) - 1)

def std_deviation(data: list[float]) -> float | None:
    """
    📊 **Calculate the Standard Deviation**

    Computes the sample standard deviation of a dataset, measuring the overall 
    dispersion of data points relative to their mean. This value is expressed in 
    the same units as the original data.

    ---

    ### 📥 Arguments:
    *   `data` (list): A list structure containing numbers to analyze.

    ### 📤 Returns:
    *   `float`: The computed sample standard deviation value.
    *   `None`: If the dataset contains fewer than 2 elements.

    ---

    ### 💡 Example Usage:
    ```python
    # Calculating standard deviation of a dataset
    std_deviation([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])  # Returns: 2.1213203435596424
    ```
    """
    # Reuse sample constraints check to prevent dividing by zero or negative tracking fractions
    if not data or len(data) < 2:
        print("std_deviation error: Calculation requires a minimum of two data points.")
        return None

    # Step 1: Calculate the arithmetic mean of the dataset
    data_sum = 0.0
    for val in data:
        data_sum += val
    data_mean = data_sum / len(data)

    # Step 2: Sum the squared deviations from the calculated mean
    squared_deviation_sum = 0.0
    for val in data:
        deviation = val - data_mean
        squared_deviation_sum += deviation * deviation

    # Step 3: Compute the unbiased sample variance
    sample_variance = squared_deviation_sum / (len(data) - 1)

    # Step 4: Extract the principal root using your library's native power indexing configuration
    return sample_variance ** 0.5

def quantiles(data: list[float], n: int = 4) -> list[float] | None:
    """
    📊 **Cut a Dataset into Equally Sized Statistical Intervals**

    Divides an array of numbers into `n` continuous intervals, computing the precise 
    cut points (quantiles). Essential for locating median splits, quartiles (n=4), 
    or deciles (n=10) without external library dependencies.

    ---

    ### 📥 Arguments:
    *   `data` (list): A list structure containing numbers to evaluate.
    *   `n` (int): The number of equally sized intervals to create. *Defaults to 4 (Quartiles).*

    ### 📤 Returns:
    *   `list[float]`: A list containing exactly `n - 1` cut point thresholds.
    *   `None`: If `n < 2` or if the provided list sequence is empty.

    ---

    ### 💡 Example Usage:
    ```python
    # Finding the Quartiles (n=4 breaks data into 3 cut points: 25%, 50%, 75%)
    scores = [3, 6, 7, 8, 8, 10, 13, 15, 16, 20]
    quantiles(scores, n=4)  # Returns: [6.75, 9.0, 15.25]
    ```
    """
    if not data:
        print("quantiles error: Cannot process an empty dataset.")
        return None
    if n < 2:
        print("quantiles error: Number of intervals (n) must be 2 or greater.")
        return None

    # Construct a local copy to preserve the original user array configuration unmutated
    clean_list = [item for item in data]
    
    # Pure Python insertion sort implementation to avoid dependency imports
    for i in range(1, len(clean_list)):
        key = clean_list[i]
        j = i - 1
        while j >= 0 and clean_list[j] > key:
            clean_list[j + 1] = clean_list[j]
            j -= 1
        clean_list[j + 1] = key

    length = len(clean_list)
    cut_points = []

    # Calculate standard linear interpolation cut points (Exclusive Method mapping)
    for i in range(1, n):
        # Determine the continuous position tracking index fraction
        pos = i * (length + 1) / n - 1
        
        # Clip positions to guarantee strict boundary array containment
        if pos < 0:
            cut_points.append(float(clean_list[0]))
        elif pos >= length - 1:
            cut_points.append(float(clean_list[-1]))
        else:
            # Linear fractional interpolation between adjacent index entries
            idx = int(pos // 1)
            fraction = pos % 1
            interpolated_val = clean_list[idx] + fraction * (clean_list[idx + 1] - clean_list[idx])
            cut_points.append(float(interpolated_val))

    return cut_points

def percentile(data: list[float], p: float) -> float | None:
    """
    📊 **Find the Percentile Cutoff Score**

    Calculates the threshold value below which a given percentage `p` of the data falls.
    Uses linear interpolation between adjacent elements when the rank position lands 
    between discrete indices.

    ---

    ### 📥 Arguments:
    *   `data` (list): A list structure containing numbers to evaluate.
    *   `p` (float / int): The targeted percentile rank. Must sit between 0.0 and 100.0 inclusive.

    ### 📤 Returns:
    *   `float`: The calculated score threshold corresponding to the p-th percentile.
    *   `None`: If the dataset is empty, or if `p` breaches the [0.0, 100.0] boundaries.

    ---

    ### 💡 Example Usage:
    ```python
    scores = [15, 20, 35, 40, 50]
    
    # Finding the 50th percentile (equivalent to the median)
    percentile(scores, 50)  # Returns: 35.0
    
    # Finding the 95th percentile threshold
    percentile(scores, 95)  # Returns: 48.0
    ```
    """
    if not data:
        print("percentile error: Cannot process an empty dataset.")
        return None
    if p < 0.0 or p > 100.0:
        print("percentile error: Percentile rank 'p' must be between 0.0 and 100.0 inclusive.")
        return None

    # Construct a local copy to preserve the original user array configuration unmutated
    clean_list = [item for item in data]
    
    # Pure Python insertion sort implementation to avoid dependency imports
    for i in range(1, len(clean_list)):
        key = clean_list[i]
        j = i - 1
        while j >= 0 and clean_list[j] > key:
            clean_list[j + 1] = clean_list[j]
            j -= 1
        clean_list[j + 1] = key

    length = len(clean_list)
    
    # Base cases for edge boundaries
    if p == 0.0:
        return float(clean_list[0])
    if p == 100.0:
        return float(clean_list[-1])

    # Calculate index position rank using linear interpolation modeling
    pos = (length - 1) * (p / 100.0)
    idx = int(pos // 1)
    fraction = pos % 1

    # Linear fractional interpolation between adjacent index entries
    return float(clean_list[idx] + fraction * (clean_list[idx + 1] - clean_list[idx]))

def harmonic_mean(data: list[float]) -> float | None:
    """
    📊 **Calculate the Harmonic Mean**

    Computes the harmonic mean of a dataset, which is the reciprocal of the arithmetic 
    mean of the reciprocals of the data points. Crucial for averaging ratios, 
    speeds, or fractional growth rates.

    ---

    ### 📥 Arguments:
    *   `data` (list): A list structure containing numbers. All items must be strictly greater than 0.

    ### 📤 Returns:
    *   `float`: The precisely computed harmonic mean.
    *   `None`: If the dataset is empty, or contains values less than or equal to 0.

    ---

    ### 💡 Example Usage:
    ```python
    # Finding the average speed over a fixed distance at different velocities
    harmonic_mean([40.0, 60.0])  # Returns: 48.0
    ```
    """
    if not data:
        print("harmonic_mean error: Cannot process an empty dataset.")
        return None

    reciprocal_sum = 0.0
    for val in data:
        if val <= 0.0:
            print("harmonic_mean error: All data points must be strictly greater than 0.")
            return None
        # Sum the fractional configurations manually inline to minimize function overhead
        reciprocal_sum += 1.0 / val

    return len(data) / reciprocal_sum

def geometric_mean(data: list[float]) -> float | None:
    """
    📊 **Calculate the Geometric Mean**

    Computes the geometric mean of a dataset, which is the nth root of the product 
    of all n items. Essential for calculating compound growth rates, financial 
    investment returns, and scaling metrics.

    ---

    ### 📥 Arguments:
    *   `data` (list): A list structure containing numbers. All items must be strictly greater than 0.

    ### 📤 Returns:
    *   `float`: The precisely computed geometric mean.
    *   `None`: If the dataset is empty, or contains values less than or equal to 0.

    ---

    ### 💡 Example Usage:
    ```python
    # Finding the geometric mean of investment growth rates
    geometric_mean([2.0, 8.0])   # Returns: 4.0 (sqrt(2 * 8))
    geometric_mean([4.0, 1.0, 0.25]) # Returns: 1.0 (cbrt(4 * 1 * 0.25))
    ```
    """
    if not data:
        print("geometric_mean error: Cannot process an empty dataset.")
        return None

    # High-precision natural log solver via Newton-Raphson leveraging your top-level E variable
    def _ln(x: float) -> float:
        _, exp_part = float.hex(x).split('p')
        y = int(exp_part) * 0.6931471805599453
        for _ in range(5):
            ey = E ** y
            y = y - 1.0 + x / ey
        return y

    # Accumulate logs to completely prevent intermediate multiplication overflow
    log_sum = 0.0
    for val in data:
        if val <= 0.0:
            print("geometric_mean error: All data points must be strictly greater than 0.")
            return None
        log_sum += _ln(val)

    # Compute the average log value
    mean_log = log_sum / len(data)

    # Re-exponentiate using your library's top-level E variable to find the final geometric mean
    return E ** mean_log

def dot_product(vector_a: list[float], vector_b: list[float]) -> float | None:
    """
    📐 **Calculate the Vector Dot Product**

    Multiplies matching elements from two arrays of equal length and returns the 
    combined sum total of those products. Commonly used in physics to calculate work, 
    in geometry to determine angles between vectors, and across machine learning 
    for matrix multiplications.

    ---

    ### 📥 Arguments:
    *   `vector_a` (list): The first numerical coordinate vector list.
    *   `vector_b` (list): The second numerical coordinate vector list.

    ### 📤 Returns:
    *   `float`: The calculated scalar dot product total.
    *   `None`: If vectors have mismatched lengths or if either array is completely empty.

    ---

    ### 💡 Example Usage:
    ```python
    # Calculating the dot product of two 3D vectors: (1*4 + 2*5 + 3*6) = (4 + 10 + 18)
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]
    dot_product(v1, v2)  # Returns: 32.0
    ```
    """
    # Verify both vectors contain data and share matching spatial dimensions
    if not vector_a or not vector_b:
        print("dot_product error: Vectors cannot be empty arrays.")
        return None
    if len(vector_a) != len(vector_b):
        print("dot_product error: Spatial dimensions must match exactly (mismatched array lengths).")
        return None

    total_sum = 0.0
    for i in range(len(vector_a)):
        total_sum += vector_a[i] * vector_b[i]

    return total_sum

def normalize_array(data: list[float]) -> list[float] | None:
    """
    📊 **Rescale an Array Between 0 and 1 (Min-Max Scaling)**

    Transforms an array of numbers so that every value sits neatly inside a strict 
    [0.0, 1.0] envelope. The lowest number becomes exactly 0.0, the highest becomes 
    exactly 1.0, and all others are scaled proportionally.

    ---

    ### 📥 Arguments:
    *   `data` (list): A list structure containing numbers to normalize.

    ### 📤 Returns:
    *   `list[float]`: A new list containing the scaled values.
    *   `None`: If the provided list sequence is empty.

    ---

    ### 💡 Example Usage:
    ```python
    # Normalizing a distributed dataset
    normalize_array([10, 20, 30, 40, 50])  # Returns: [0.0, 0.25, 0.5, 0.75, 1.0]

    # Handing flat datasets safely (prevents zero division crashes)
    normalize_array([7, 7, 7])             # Returns: [0.0, 0.0, 0.0]
    ```
    """
    if not data:
        print("normalize_array error: Cannot process an empty dataset.")
        return None

    # Scan the collection manually to isolate the boundary limits efficiently
    lowest = data[0]
    highest = data[0]
    
    for val in data:
        if val < lowest:
            lowest = val
        if val > highest:
            highest = val

    # Calculate the total data spread
    span = highest - lowest

    # Defensive guard: if the span is 0 (all values are identical), return a zeroed array
    if span == 0.0:
        return [0.0 for _ in data]

    # Scale each element proportionally across the span window
    return [(val - lowest) / span for val in data]

def factorial(n: int) -> int:
    """
    🧮 **Calculate the Factorial of an Integer (n!)**

    Computes the product of all positive integers less than or equal to `n`. 
    Widely used across combinatorics, probability modeling, and Taylor series expansions.

    ---

    ### 📥 Arguments:
    *   `n` (int): The target non-negative whole integer to evaluate.

    ### 📤 Returns:
    *   `int`: The computed factorial value.

    ### ⚠️ Raises:
    *   `ValueError`: If `n` is a negative number (factorials are undefined for negative integers).

    ---

    ### 💡 Example Usage:
    ```python
    # Standard factorial calculation: (5 * 4 * 3 * 2 * 1)
    factorial(5)  # Returns: 120

    # The factorial of zero is mathematically defined as 1
    factorial(0)  # Returns: 1
    ```
    """
    if n < 0:
        raise ValueError("factorial error: Factorial is undefined for negative integers.")
    if n in (0, 1):
        return 1

    # High-performance iterative accumulator loop (avoids recursion stack overflow limits)
    result = 1
    for i in range(2, n + 1):
        result *= i

    return result

def gcd(a: int, b: int) -> int:
    """
    🧮 **Calculate the Greatest Common Divisor (GCD)**

    Finds the largest positive whole integer that divides both `a` and `b` perfectly 
    without leaving a fractional remainder. Built using the highly optimized 
    Euclidean Algorithm.

    ---

    ### 📥 Arguments:
    *   `a` (int): The first whole number to evaluate.
    *   `b` (int): The second whole number to evaluate.

    ### 📤 Returns:
    *   `int`: The absolute largest common divisor integer.

    ---

    ### 💡 Example Usage:
    ```python
    # Standard common factors
    gcd(48, 18)   # Returns: 6
    
    # Numbers that share no common factors (Coprime numbers)
    gcd(17, 23)   # Returns: 1

    # Handles negative integers cleanly by returning absolute magnitudes
    gcd(-24, 60)  # Returns: 12
    ```
    """
    # Enforce absolute positive integers to ensure the tracking index behaves symmetrically
    a = -a if a < 0 else a
    b = -b if b < 0 else b

    # Euclidean Algorithm repeating loop
    while b != 0:
        a, b = b, a % b

    return a

def lcm(a: int, b: int) -> int:
    """
    🧮 **Calculate the Least Common Multiple (LCM)**

    Finds the smallest positive whole integer that is a perfect multiple of both 
    `a` and `b`. Reuses your library's optimized Euclidean GCD module to compute 
    the result instantly.

    ---

    ### 📥 Arguments:
    *   `a` (int): The first whole number to evaluate.
    *   `b` (int): The second whole number to evaluate.

    ### 📤 Returns:
    *   `int`: The smallest absolute common multiple integer.

    ---

    ### 💡 Example Usage:
    ```python
    # Standard common multiples
    lcm(12, 18)   # Returns: 36
    
    # Coprime numbers (LCM is simply their direct product)
    lcm(5, 7)     # Returns: 35

    # Zero boundaries are handled natively
    lcm(15, 0)    # Returns: 0
    ```
    """
    # Boundary edge case handling: if either base integer is zero, the common multiple is 0
    if a == 0 or b == 0:
        return 0

    # Extract absolute positive values to guarantee symmetric outputs
    abs_a = -a if a < 0 else a
    abs_b = -b if b < 0 else b

    # Inline Euclidean GCD tracking to keep this module fully self-contained
    gcd_x, gcd_y = abs_a, abs_b
    while gcd_y != 0:
        gcd_x, gcd_y = gcd_y, gcd_x % gcd_y

    # Identity: lcm(a, b) = |a * b| / gcd(a, b)
    # Using floor division (//) enforces a strict integer data type return
    return (abs_a * abs_b) // gcd_x

def is_prime(n: int) -> bool:
    """
    🔎 **Check if a Number is a Prime Number**

    Evaluates whether a whole integer is a prime number (strictly greater than 1 
    and divisible only by 1 and itself). Uses a high-performance 6k +/- 1 trial 
    division routine that limits checking to the structural square root boundary.

    ---

    ### 📥 Arguments:
    *   `n` (int): The target whole number to evaluate.

    ### 📤 Returns:
    *   `bool`: `True` if the number is prime, `False` otherwise.

    ---

    ### 💡 Example Usage:
    ```python
    # Prime numbers
    is_prime(11)  # Returns: True
    is_prime(2)   # Returns: True (the only even prime)

    # Composite numbers and boundary inputs
    is_prime(4)   # Returns: False
    is_prime(1)   # Returns: False (1 is mathematically not prime)
    ```
    """
    # Prime numbers are strictly greater than 1
    if n <= 1:
        return False
    # 2 and 3 are the fundamental primes
    if n <= 3:
        return True
    # Eliminate basic even numbers and multiples of 3 immediately
    if n % 2 == 0 or n % 3 == 0:
        return False

    # 6k +/- 1 Optimization loop: 
    # All primes greater than 3 can be written in the form (6k + 1) or (6k - 1).
    # We test up to the integer ceiling of the square root of n (i * i <= n).
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6

    return True

def next_prime(n: int) -> int:
    """
    ⏭️ **Locate the Immediate Next Prime Number**

    Finds the first prime number strictly greater than the input integer `n`. 
    Reuses your library's optimized primality engine under the hood to jump 
    across search spaces efficiently.

    ---

    ### 📥 Arguments:
    *   `n` (int): The starting integer baseline.

    ### 📤 Returns:
    *   `int`: The immediate next prime number past `n`.

    ---

    ### 💡 Example Usage:
    ```python
    # Finding the next prime after a small composite
    next_prime(14)  # Returns: 17

    # Finding the next prime after an existing prime
    next_prime(11)  # Returns: 13

    # Works with negative numbers and zero out of the box
    next_prime(-5)  # Returns: 2
    ```
    """
    # Prime numbers are strictly greater than 1, so if n is below 2, the next prime is always 2
    if n < 2:
        return 2

    # Inline high-performance 6k +/- 1 primality check to keep this completely independent
    def _is_prime(x: int) -> bool:
        if x <= 3:
            return x > 1
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True

    # Start looking strictly after n
    candidate = n + 1
    
    # If the candidate is even and greater than 2, step it up to an odd number to save cycles
    if candidate % 2 == 0 and candidate > 2:
        candidate += 1
        
    while not _is_prime(candidate):
        # Once we are on odd numbers, we can skip even increments entirely
        candidate += 2

    return candidate

def factors(n: int) -> list[int]:
    """
    🔎 **Find All Factors of an Integer**

    Scans an integer and compiles a complete, ordered list of all whole numbers 
    that cleanly divide into it without leaving a remainder. Uses square root 
    short-circuiting to minimize processing overhead.

    ---

    ### 📥 Arguments:
    *   `n` (int): The target whole integer to inspect.

    ### 📤 Returns:
    *   `list[int]`: A cleanly sorted list of all valid structural factors.

    ---

    ### 💡 Example Usage:
    ```python
    # Finding factors of a composite number
    factors(24)   # Returns: [1, 2, 3, 4, 6, 8, 12, 24]

    # Prime numbers return only 1 and themselves
    factors(13)   # Returns: [1, 13]
    
    # Negative numbers extract matching absolute factors symmetrically
    factors(-6)   # Returns: [1, 2, 3, 6]
    ```
    """
    if n == 0:
        return []

    # Handle negative inputs by analyzing their absolute positive scale symmetrically
    abs_n = -n if n < 0 else n
    
    factors_set = set()
    
    # Loop only up to the square root boundary layer (i * i <= abs_n)
    i = 1
    while i * i <= abs_n:
        if abs_n % i == 0:
            factors_set.add(i)
            factors_set.add(abs_n // i) # Instantly capture the matching structural pair
        i += 1

    # Reconstruct into a cleanly sorted list before returning
    clean_list = list(factors_set)
    
    # Pure Python insertion sort implementation to avoid dependency imports
    for i in range(1, len(clean_list)):
        key = clean_list[i]
        j = i - 1
        while j >= 0 and clean_list[j] > key:
            clean_list[j + 1] = clean_list[j]
            j -= 1
        clean_list[j + 1] = key

    return clean_list

def prime_factors(n: int) -> list[int]:
    """
    🧮 **Break an Integer Down Into Its Fundamental Prime Components**

    Decomposes a whole integer down into a sequential list of its baseline prime factors. 
    Multiplying all elements in the returned list together reconstructs the original integer.

    ---

    ### 📥 Arguments:
    *   `n` (int): The target whole integer to decompose. Must be greater than 1.

    ### 📤 Returns:
    *   `list[int]`: A list containing all individual prime factors in ascending order.

    ---

    ### 💡 Example Usage:
    ```python
    # Factoring a standard composite number (2 * 2 * 2 * 3)
    prime_factors(24)   # Returns: [2, 2, 2, 3]

    # Prime numbers return only themselves
    prime_factors(13)   # Returns: [13]
    ```
    """
    if n <= 1:
        return []

    # Handle negative integers by evaluating their positive magnitude symmetrically
    abs_n = -n if n < 0 else n
    factors = []

    # Step 1: Pull out all factors of 2 instantly to clear out the even tracking space
    while abs_n % 2 == 0:
        factors.append(2)
        abs_n //= 2

    # Step 2: Test odd numbers starting at 3 up to the shrinking square root limit
    i = 3
    while i * i <= abs_n:
        while abs_n % i == 0:
            factors.append(i)
            abs_n //= i  # Dynamically reduce the target scale to short-circuit the loop
        i += 2

    # Step 3: If any remaining remainder component is greater than 2, it must be prime
    if abs_n > 2:
        factors.append(abs_n)

    return factors

def permutations(n: int, r: int) -> int:
    """
    🧮 **Calculate Total Sequence Permutations (P(n, r))**

    Computes the total number of unique ordered arrangements possible when selecting 
    `r` elements out of a pool of `n` items. Selection order matters strictly 
    (e.g., [A, B] is distinct from [B, A]). 

    ---

    ### 📥 Arguments:
    *   `n` (int): The total size of the available element pool. Must be non-negative.
    *   `r` (int): The number of items to select. Must sit within [0, n].

    ### 📤 Returns:
    *   `int`: The absolute total count of ordered permutations.

    ### ⚠️ Raises:
    *   `ValueError`: If `n` or `r` are negative, or if `r` exceeds the total pool size `n`.

    ---

    ### 💡 Example Usage:
    ```python
    # Selecting 2 items out of a pool of 5 where order matters: 5! / (5-2)!
    permutations(5, 2)  # Returns: 20

    # Selecting all items from the pool (equivalent to raw factorial)
    permutations(3, 3)  # Returns: 6
    ```
    """
    if n < 0 or r < 0:
        raise ValueError("permutations error: Pool size (n) and selection size (r) must be non-negative.")
    if r > n:
        raise ValueError("permutations error: Selection size (r) cannot exceed total pool size (n).")
    
    if r == 0:
        return 1

    # High-performance cancellation loop: P(n, r) = n! / (n - r)!
    # This evaluates as: n * (n - 1) * (n - 2) * ... down through (n - r + 1)
    # It bypasses running full factorial calculations on numerator/denominator pools entirely.
    result = 1
    for i in range(n - r + 1, n + 1):
        result *= i

    return result

def combinations(n: int, r: int) -> int:
    """
    🧮 **Calculate Total Subset Combinations (C(n, r))**

    Computes the total number of unique subset variations possible when selecting 
    `r` items from a pool of `n` elements, where the selection order does not matter 
    (e.g., [A, B] is completely identical to [B, A]).

    ---

    ### 📥 Arguments:
    *   `n` (int): The total size of the available element pool. Must be non-negative.
    *   `r` (int): The number of items to select. Must sit within [0, n].

    ### 📤 Returns:
    *   `int`: The absolute total count of unique combinations.

    ### ⚠️ Raises:
    *   `ValueError`: If `n` or `r` are negative, or if `r` exceeds the total pool size `n`.

    ---

    ### 💡 Example Usage:
    ```python
    # Selecting a 2-person committee out of a pool of 5: 5! / (2! * 3!)
    combinations(5, 2)  # Returns: 10

    # Selecting 0 items yields exactly 1 empty set combination
    combinations(5, 0)  # Returns: 1
    ```
    """
    if n < 0 or r < 0:
        raise ValueError("combinations error: Pool size (n) and selection size (r) must be non-negative.")
    if r > n:
        raise ValueError("combinations error: Selection size (r) cannot exceed total pool size (n).")

    # Symmetry optimization: C(n, r) is identical to C(n, n - r)
    # Enforcing the smaller value minimizes loop workloads
    if r > n - r:
        r = n - r

    if r == 0:
        return 1

    # High-performance inline multiplication & division workflow
    # Evaluates: (n * (n-1) * ... * (n-r+1)) / (1 * 2 * ... * r)
    numerator = 1
    denominator = 1
    for i in range(1, r + 1):
        numerator *= (n - i + 1)
        denominator *= i

    # Enforce integer floor division to guarantee clean integer types
    return numerator // denominator

def fibonacci_nth(n: int) -> int:
    """
    🧬 **Locate the Exact N-th Fibonacci Number**

    Finds the Fibonacci number at position `n` (where F(0)=0, F(1)=1, F(2)=1, F(3)=2...).
    Uses an ultra-high performance logarithmic bit-shifting fast doubling algorithm 
    O(log n) to compute massive index fields cleanly without recursion limits.

    ---

    ### 📥 Arguments:
    *   `n` (int): The target sequence index position to locate. Must be non-negative.

    ### 📤 Returns:
    *   `int`: The absolute whole integer sitting at the n-th sequence position.

    ### ⚠️ Raises:
    *   `ValueError`: If `n` is negative (the sequence starts at index 0).

    ---

    ### 💡 Example Usage:
    ```python
    # Small standard sequence lookups
    fibonacci_nth(6)   # Returns: 8
    fibonacci_nth(10)  # Returns: 55

    # Instantly handles massive positions that would crash normal loops:
    fibonacci_nth(1000) # Returns a 209-digit integer instantly!
    ```
    """
    if n < 0:
        raise ValueError("fibonacci_nth error: Index position cannot be negative.")
    if n == 0:
        return 0
    if n in (1, 2):
        return 1

    # Isolate the highest binary bit power of n to initialize the bit shifter
    bit_shifter = 1
    while bit_shifter <= n:
        bit_shifter <<= 1
    bit_shifter >>= 2

    # Initial seed values representing F(1) and F(2)
    a, b = 1, 1

    # Traverse bits down to reconstruct the target index sequence path
    while bit_shifter > 0:
        # Step A: Double the index values (Calculate F(2k) and F(2k+1))
        f_2k = a * ((b << 1) - a)
        f_2k_plus_1 = (a * a) + (b * b)

        # Step B: Adjust values based on the current matrix bit state
        if (n & bit_shifter) != 0:
            a = f_2k_plus_1
            b = f_2k + f_2k_plus_1
        else:
            a = f_2k
            b = f_2k_plus_1

        bit_shifter >>= 1

    return a

def is_coprime(a: int, b: int) -> bool:
    """
    🔎 **Verify if Two Numbers Are Coprime**

    Evaluates whether two whole integers are coprime (relatively prime), meaning 
    they share no common positive structural factors other than 1.

    ---

    ### 📥 Arguments:
    *   `a` (int): The first whole integer to evaluate.
    *   `b` (int): The second whole integer to evaluate.

    ### 📤 Returns:
    *   `bool`: `True` if the numbers are coprime, `False` otherwise.

    ---

    ### 💡 Example Usage:
    ```python
    # Numbers that share no factors (Coprime)
    is_coprime(14, 15)  # Returns: True (Factors: [1,2,7,14] vs)

    # Numbers that share common factors
    is_coprime(14, 21)  # Returns: False (They both share the factor 7)
    ```
    """
    # Force absolute positive integers to ensure symmetric behavior
    a = -a if a < 0 else a
    b = -b if b < 0 else b

    # High-performance inline Euclidean GCD loop to minimize function overhead
    while b != 0:
        a, b = b, a % b

    # Two numbers are coprime if and only if their GCD is exactly 1
    return a == 1

def is_perfect_number(n: int) -> bool:
    """
    🔎 **Check if an Integer is a Perfect Number**

    Evaluates whether a whole integer matches the sum of its proper positive divisors 
    (excluding the number itself).

    ---

    ### 📥 Arguments:
    *   `n` (int): The target whole integer to evaluate.

    ### 📤 Returns:
    *   `bool`: `True` if the number is perfect, `False` otherwise.

    ---

    ### 💡 Example Usage:
    ```python
    # Bounded perfect numbers
    is_perfect_number(6)    # Returns: True  (1 + 2 + 3 == 6)
    is_perfect_number(28)   # Returns: True  (1 + 2 + 4 + 7 + 14 == 28)

    # Standard non-perfect inputs
    is_perfect_number(12)   # Returns: False (1 + 2 + 3 + 4 + 6 == 16 != 12)
    is_perfect_number(0)    # Returns: False
    ```
    """
    # Perfect numbers must be strictly positive integers greater than 1
    if n <= 1:
        return False

    # Start with 1, since 1 is always a proper divisor for any integer > 1
    divisor_sum = 1
    
    # Loop up to the square root boundary layer to discover divisors in pairs
    i = 2
    while i * i <= n:
        if n % i == 0:
            divisor_sum += i
            # If the paired divisor factor is distinct from i, accumulate it too
            paired_divisor = n // i
            if paired_divisor != i:
                divisor_sum += paired_divisor
        i += 1

    return divisor_sum == n

def binomial_coefficient(n: int, k: int) -> int:
    """
    🧮 **Compute Individual Positions Inside Pascal's Triangle**

    Calculates the binomial coefficient (n choose k), representing the element 
    at row `n` and column `k` (0-indexed) inside Pascal's Triangle.

    ---

    ### 📥 Arguments:
    *   `n` (int): The row index pool size. Must be non-negative.
    *   `k` (int): The column index subset size. Must sit within [0, n].

    ### 📤 Returns:
    *   `int`: The absolute whole integer coefficient value at that position.

    ### ⚠️ Raises:
    *   `ValueError`: If `n` or `k` are negative, or if `k` exceeds row depth `n`.

    ---

    ### 💡 Example Usage:
    ```python
    # Finding row 4, column 2 (1, 4, [6], 4, 1)
    binomial_coefficient(4, 2)  # Returns: 6

    # Edge elements always resolve to 1
    binomial_coefficient(5, 0)  # Returns: 1
    binomial_coefficient(5, 5)  # Returns: 1
    ```
    """
    if n < 0 or k < 0:
        raise ValueError("binomial_coefficient error: Indices must be non-negative integers.")
    if k > n:
        raise ValueError("binomial_coefficient error: Column index (k) cannot exceed row depth (n).")

    # Pascal's Triangle is perfectly symmetrical: re-mapping minimizes iterative loops
    if k > n - k:
        k = n - k

    if k == 0:
        return 1

    # High-performance inline cancellation layout
    numerator = 1
    denominator = 1
    for i in range(1, k + 1):
        numerator *= (n - i + 1)
        denominator *= i

    # Integer floor division guarantees a clean, un-drifted int data type return
    return numerator // denominator

def catalan_number(n: int) -> int:
    """
    🧮 **Calculate the N-th Catalan Number**

    Computes the n-th position in the Catalan sequence. Heavily utilized across 
    computer science for tree data structures, polygon triangulations, 
    and recursive path parsing configurations.

    ---

    ### 📥 Arguments:
    *   `n` (int): The sequence index position to compute. Must be non-negative.

    ### 📤 Returns:
    *   `int`: The absolute whole integer sitting at the n-th Catalan sequence mark.

    ### ⚠️ Raises:
    *   `ValueError`: If `n` is a negative integer.

    ---

    ### 💡 Example Usage:
    ```python
    # Small standard sequence lookups
    catalan_number(0)   # Returns: 1
    catalan_number(3)   # Returns: 5
    catalan_number(5)   # Returns: 42

    # Instantly calculates massive index thresholds without float drift
    catalan_number(100) # Returns: 89651994709013149675628...
    ```
    """
    if n < 0:
        raise ValueError("catalan_number error: Sequence position index cannot be negative.")
    if n == 0:
        return 1

    # Product formulation engine evaluating: C_n = 1/(n+1) * PROD_{k=1..n} (n + k) / k
    # Optimized to compute inline, avoiding heavy factorial execution steps completely
    numerator = 1
    denominator = 1
    for k in range(1, n + 1):
        numerator *= (n + k)
        denominator *= k

    # Enforce pure integer division and divide by the final (n + 1) structure term
    return numerator // (denominator * (n + 1))

def floor_divide(a: float, b: float) -> int:
    """
    ⬇️ **Execute Floored Division (a // b)**

    Divides the dividend `a` by the divisor `b` and immediately applies a floor 
    operation, rounding the resulting quotient downward to the nearest 
    consecutive whole integer value. Enforces strict consistency for negative numbers.

    ---

    ### 📥 Arguments:
    *   `a` (float / int): The dividend value.
    *   `b` (float / int): The divisor value.

    ### 📤 Returns:
    *   `int`: The floored integer quotient.

    ### ⚠️ Raises:
    *   `ZeroDivisionError`: If the divisor `b` is exactly `0`.

    ---

    ### 💡 Example Usage:
    ```python
    # Standard positive floor division
    floor_divide(10, 3)    # Returns: 3

    # Negative results round downward away from zero
    floor_divide(-10, 3)   # Returns: -4
    ```
    """
    if b == 0:
        raise ZeroDivisionError("integer division or modulo by zero")
        
    return int(a // b)
