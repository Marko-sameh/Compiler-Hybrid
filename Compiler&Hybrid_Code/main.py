import tkinter as tk
from tkinter import scrolledtext, messagebox
import re
import sys
from collections import namedtuple
from tkinter import messagebox, simpledialog

# Define token specifications
token_specification = [
    ('NUMBER', r'\b\d+\.\d+|\d+\b'),  # Integer
    ('ID', r'\b[a-zA-Z_]\w*\b'),  # Identifiers
    ('OP', r'[+\-*/=]'),  # Operators
    ('WHITESPACE', r'\s+'),  # Whitespace
]


# Convert math form to source code format
def convert_math_to_code(math_expr):
    math_expr = re.sub(r'\bpi\b', '3.14', math_expr)
    math_expr = re.sub(r'(\d+)([a-zA-Z_])', r'\1*\2', math_expr)  # handel multiplication
    tokens = []
    position = 0

    while position < len(math_expr):
        match = None

        for token_type, token_regex in token_specification:
            regex = re.compile(token_regex)
            match = regex.match(math_expr, position)
            if match:
                value = match.group(0)
                if token_type != 'WHITESPACE':
                    tokens.append((token_type, value))
                position = match.end(0)
                break

        if not match:
            messagebox.showerror("Error", f'Lexical error Unexpected character: {math_expr[position]}')
            messagebox.showerror("Error", "Enter a valid character")
            return
    code = ''
    for token_type, value in tokens:
        if token_type == 'NUMBER':
            code += value
        elif token_type == 'ID':
            code += value
        elif token_type == 'OP':
            code += f' {value} '

    return code.strip()


# Check if a string can be converted to a float (handles integers and floats)
def is_number(value):
    try:
        float(value)  # Try to convert the value to float
        return True
    except ValueError:
        return False


def creat_lexical(text):
    combined_pattern = r'\b\d+\.\d+|\d+\b|[a-zA-Z_]\w*|[+\-*/=]|\s+'
    non_matching = re.sub(combined_pattern, '', text)
    if non_matching:
        messagebox.showerror("Error", "Lexical Error.")
        return

    if re.search(r'\d+[a-zA-Z_]|[a-zA-Z_]\d+', text):
        messagebox.showerror("Error", "Lexical Error.")
        return

    if '=' in text:
        normalized_text = re.sub(r'\s+', '', text)

        left_side, right_side = map(str.strip, text.split('=', 1))
        if not re.match(r'^[a-zA-Z]$', left_side):
            messagebox.showerror("Error", "Lexical Error.")
            return

        pattern = fr'^{left_side}=[^=]*{left_side}[\+\-\*/]'
        if re.match(pattern, normalized_text):
            messagebox.showerror("Error", "Lexical Error: Self-assignment with operations is not allowed.")
            return

    id_counter = 1
    id_map = {}

    def replace_id(match):
        nonlocal id_counter
        variable = match.group(0)
        if variable not in id_map:
            id_map[variable] = f"id{id_counter}"
            id_counter += 1
        return id_map[variable]

    created_text = re.sub(r'\b[a-zA-Z_]\w*\b', replace_id, text)
    return created_text


# Tokenizer function for math expressions
def tokenize(math_expr):
    tokens = []
    position = 0

    while position < len(math_expr):
        match = None
        for token_type, token_regex in token_specification:
            regex = re.compile(token_regex)
            match = regex.match(math_expr, position)
            if match:
                value = match.group(0)
                if token_type != 'WHITESPACE':
                    tokens.append(Token(token_type, value))
                position = match.end(0)
                break
    return tokens


# Syntax analysis function to build a parse tree
class ParseTreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


# Token and ParseTreeNode definitions remain unchanged
Token = namedtuple('Token', ['type', 'value'])


def parse_expression(tokens):
    def parse_term():
        if not tokens:
            raise SyntaxError("Incomplete expression")

        token = tokens.pop(0)
        if token.type == 'NUMBER' or token.type == 'ID':
            return ParseTreeNode(token.value)
        else:
            raise SyntaxError("syntax error Unexpected token: " + token.value)

    def parse_factor():
        node = parse_term()
        while tokens and tokens[0].value in ('*', '/'):
            op = tokens.pop(0)
            if not tokens or tokens[0].type not in ('NUMBER', 'ID'):
                raise SyntaxError("syntax error, Expected a term after operator " + op.value)
            right = parse_term()
            new_node = ParseTreeNode(op.value)
            new_node.left = node
            new_node.right = right
            node = new_node
        return node

    def parse_expr():
        node = parse_factor()
        while tokens and tokens[0].value in ('+', '-'):
            op = tokens.pop(0)
            if not tokens or tokens[0].type not in ('NUMBER', 'ID'):
                raise SyntaxError("syntax error,Expected a term after operator " + op.value)
            right = parse_factor()
            new_node = ParseTreeNode(op.value)
            new_node.left = node
            new_node.right = right
            node = new_node
        return node

    # Check for assignment expression
    if tokens[0].type == 'ID' and len(tokens) > 1 and tokens[1].value == '=':
        assignment_node = ParseTreeNode('=')
        assignment_node.left = ParseTreeNode(tokens.pop(0).value)  # ID on the left
        tokens.pop(0)  # Discard '='
        assignment_node.right = parse_expr()  # Parse the expression on the right
        if tokens:
            raise SyntaxError("syntax error,Unexpected token after assignment")
        return assignment_node

    # If no assignment, parse as a regular expression
    root = parse_expr()
    if tokens:
        raise SyntaxError("Unexpected token after expression")
    return root


def should_convert_tree(node, variable_types):
    if node is None:
        return False
    return (is_number(node.value) and not node.value.isdigit()) or \
           (node.value in variable_types and variable_types[node.value] == 'float') or \
           should_convert_tree(node.left, variable_types) or \
           should_convert_tree(node.right, variable_types)


def semantic_analysis(node, variable_types):
    def convert_tree(node):
        if node is None:
            return
        if node.value in variable_types and variable_types[node.value] == 'int':
            node.value = f'inttofloat──{node.value}'
        # Convert integer to float if needed
        if node.value.isdigit():
            node.value = f'inttofloat──{node.value}'
        convert_tree(node.left)
        convert_tree(node.right)

    if should_convert_tree(node, variable_types):
        convert_tree(node)

    return node


# Intermediate code generation function
def generate_intermediate_code(node):
    temp_counter = 1
    code = []
    conversion_map = {}

    def handle_inttofloat(value):
        nonlocal temp_counter
        if value not in conversion_map:
            temp_var = f'T{temp_counter}'
            temp_counter += 1
            code.append(f'{temp_var} = {value}')
            conversion_map[value] = temp_var
        return conversion_map[value]

    def generate_code_recursive(node):
        nonlocal temp_counter
        if node is None:
            return None

        # Handle variables and constants
        if node.value.isdigit() or node.value.startswith('id'):
            return node.value

        # Handle inttofloat separately, creating a temp variable if needed
        if 'inttofloat' in node.value:
            converted_value = handle_inttofloat(node.value)
            return converted_value

        # Process left and right operands
        left_operand = generate_code_recursive(node.left)
        right_operand = generate_code_recursive(node.right)

        if node.value == '=' and isinstance(node.left.value, str) and node.left.value.startswith('id'):
            code.append(f'{node.left.value} = {right_operand}')
            return node.left.value

        # Generate a new temporary variable for the operation
        temp_var = f'T{temp_counter}'
        temp_counter += 1
        code.append(f'{temp_var} = {left_operand} {node.value} {right_operand}')
        return temp_var

    # Start recursive code generation
    generate_code_recursive(node)
    return code


"""----------------------------------------------------------------"""


# Hybrid phase: Substitute variable values into the parse tree and calculate the result
def hybrid_phase_and_evaluate(node, variable_values, variable_types):
    if node is None:
        raise ValueError("Unexpected None node encountered in the parse tree. Check syntax analysis.")

    # Handle nodes with int-to-float conversion
    if node.value.startswith("inttofloat──"):
        original_value = node.value.split("──")[1]  # Extract the numeric part
        if original_value.isdigit():
            return float(original_value)
        elif original_value in variable_values:
            value = float(variable_values[original_value])
            node.value = f"{original_value}──{value}"
            return value
        else:
            raise ValueError(f"Invalid inttofloat conversion for: {node.value}")

    # If the node is a variable (on the right-hand side), substitute its value
    if node.value in variable_values:
        if should_convert_tree(node, variable_types):
            value = float(variable_values[node.value])
        else:
            value = int(variable_values[node.value])
        node.value = f"{node.value}──{value}"
        return value

    # If the node is a number, return it as a float
    if node.value.replace('.', '', 1).isdigit():
        if should_convert_tree(node, variable_types):
            return float(node.value)
        else:
            return int(node.value)

    # If the node is an operator, recursively calculate its operands
    if node.value == '+':
        return hybrid_phase_and_evaluate(node.left, variable_values, variable_types) + hybrid_phase_and_evaluate(
            node.right,
            variable_values,
            variable_types)
    elif node.value == '-':
        return hybrid_phase_and_evaluate(node.left, variable_values, variable_types) - hybrid_phase_and_evaluate(
            node.right,
            variable_values,
            variable_types)
    elif node.value == '*':
        return hybrid_phase_and_evaluate(node.left, variable_values, variable_types) * hybrid_phase_and_evaluate(
            node.right,
            variable_values,
            variable_types)
    elif node.value == '/':
        right = hybrid_phase_and_evaluate(node.right, variable_values, variable_types)
        if right == 0:
            raise ZeroDivisionError("Division by zero encountered in the expression.")
        return hybrid_phase_and_evaluate(node.left, variable_values, variable_types) / right
    elif node.value == '=':
        # Left-hand side variable (assignment)
        left_var = node.left.value.split()[0]  # Extract variable name
        result = hybrid_phase_and_evaluate(node.right, variable_values, variable_types)
        variable_values[left_var] = result  # Assign calculated result to the variable
        return result

    # Raise an error for unrecognized nodes
    raise ValueError(f"Unexpected node value: {node.value}")


"""----------------------------------------------------------------"""


# Optimization phase
def optimize_code(intermediate_code):
    optimized_code = []
    temp_replacements = {}

    for line in intermediate_code:
        if 'inttofloat' in line:
            # Directly convert integer constants to float format
            parts = line.split(' = ')
            temp_var = parts[0]
            inttofloat_value = parts[1].replace('inttofloat──', '')
            if inttofloat_value.isdigit():
                # Replace with the float representation of the number
                temp_replacements[temp_var] = f"{float(inttofloat_value)}"
            else:
                temp_replacements[temp_var] = f"{inttofloat_value} (float)"
        elif '=' in line:
            # Process a normal assignment T1 = T2 * T3 or T1 = id2
            parts = line.split(' = ')
            left = parts[0]
            right = parts[1]
            # Substitute temporary replacements in the right side of the expression
            for temp, replacement in temp_replacements.items():
                right = right.replace(temp, replacement)
            optimized_code.append(f"{left} = {right}")
        else:
            # Directly append unmodifiable lines or unsupported cases
            optimized_code.append(line)

    # Renumber temporary variables
    final_code = []
    new_temp_count = 1
    temp_mapping = {}

    for line in optimized_code:
        parts = line.split(' = ')
        if len(parts) == 2:
            # Split left and right sides
            left, right = parts[0], parts[1]

            # If it's a temporary variable on the left side, assign a new name
            if left.startswith('T'):
                if left not in temp_mapping:
                    temp_mapping[left] = f"T{new_temp_count}"
                    new_temp_count += 1
                left = temp_mapping[left]

            # Replace any old temporary variables on the right side
            for old_temp, new_temp in temp_mapping.items():
                right = right.replace(old_temp, new_temp)

            final_code.append(f"{left} = {right}")
        else:
            final_code.append(line)

    return final_code


def generate_machine_code(optimized_code):
    machine_code = []
    reg1 = "R1"
    reg2 = "R2"
    reg_used = "R1"

    # Load a value into the specified register
    def load_value(value, reg):
        if 'id' in value:  # Load only identifiers into registers
            machine_code.append(f"LODF {reg},{value}")

    # Generate the machine code for each line in the optimized code
    for line in optimized_code:
        if '=' in line:
            left, right = line.split(' = ')
            if '*' in right:
                # Handle multiplication
                op1, op2 = right.split(' * ')
                if reg_used == reg1:
                    if is_number(op1.strip()):  # First operand is a constant
                            load_value(op2.strip(), reg2)  # Load variable into R2
                            machine_code.append(f"MULF {reg2}, {reg2}, #{op1.strip()}")  # Multiply R2 with constant
                    elif is_number(op2.strip()):  # Second operand is a constant
                        load_value(op1.strip(), reg2)  # Load variable into R2
                        machine_code.append(f"MULF {reg2}, {reg2}, #{op2.strip()}")  # Multiply R2 with constant
                    else:  # Both operands are variables
                        load_value(op1.strip(), reg2)  # Load first variable into R2
                        load_value(op2.strip(), reg1)  # Load second variable into R1
                        machine_code.append(f"MULF {reg2}, {reg2}, {reg1}")  # Multiply R2 by R1
                    reg_used = reg2
                elif reg_used == reg2:
                    if is_number(op1.strip()):  # First operand is a constant
                            load_value(op2.strip(), reg1)  # Load variable into R2
                            machine_code.append(f"MULF {reg1}, {reg1}, #{op1.strip()}")  # Multiply R2 with constant
                    elif is_number(op2.strip()):  # Second operand is a constant
                        load_value(op1.strip(), reg1)  # Load variable into R2
                        machine_code.append(f"MULF {reg1}, {reg1}, #{op2.strip()}")  # Multiply R2 with constant
                    else:  # Both operands are variables
                        load_value(op1.strip(), reg2)  # Load first variable into R2
                        load_value(op2.strip(), reg1)  # Load second variable into R1
                        machine_code.append(f"MULF {reg2}, {reg2}, {reg1}")  # Multiply R2 by R1
                    reg_used = reg1
            elif '+' in right:
                # Handle addition
                op1, op2 = right.split(' + ')
                if reg_used == reg1:
                    if is_number(op1.strip()):  # First operand is a constant
                        load_value(op2.strip(), reg2)  # Load variable into R2
                        machine_code.append(f"ADDF {reg2}, {reg2}, #{op1.strip()}")  # Multiply R2 with constant
                    elif is_number(op2.strip()):  # Second operand is a constant
                        load_value(op1.strip(), reg2)  # Load variable into R2
                        machine_code.append(f"ADDF {reg2}, {reg2}, #{op2.strip()}")  # Multiply R2 with constant
                    else:  # Both operands are variables
                        load_value(op1.strip(), reg2)  # Load first operand into R2
                        load_value(op2.strip(), reg1)  # Load second operand into R1
                        machine_code.append(f"ADDF {reg1}, {reg2}, {reg1}")  # Add R2 to R1
                    reg_used = reg2
                elif reg_used == reg2:
                    if is_number(op1.strip()):  # First operand is a constant
                        load_value(op2.strip(), reg1)  # Load variable into R2
                        machine_code.append(f"ADDF {reg1}, {reg1}, #{op1.strip()}")  # Multiply R2 with constant
                    elif is_number(op2.strip()):  # Second operand is a constant
                        load_value(op1.strip(), reg1)  # Load variable into R2
                        machine_code.append(f"ADDF {reg1}, {reg1}, #{op2.strip()}")  # Multiply R2 with constant
                    else:  # Both operands are variables
                        load_value(op1.strip(), reg2)  # Load first operand into R2
                        load_value(op2.strip(), reg1)  # Load second operand into R1
                        machine_code.append(f"ADDF {reg1}, {reg2}, {reg1}")  # Add R2 to R1
                    reg_used = reg1
            elif '-' in right:
                # Handle subtraction
                op1, op2 = right.split(' - ')
                if reg_used == reg1:
                    if is_number(op1.strip()):  # First operand is a constant
                        load_value(op2.strip(), reg2)  # Load variable into R2
                        machine_code.append(f"SUBF {reg2}, {reg2}, #{op1.strip()}")  # Multiply R2 with constant
                    elif is_number(op2.strip()):  # Second operand is a constant
                        load_value(op1.strip(), reg2)  # Load variable into R2
                        machine_code.append(f"SUBF {reg2}, {reg2}, #{op2.strip()}")  # Multiply R2 with constant
                    else:  # Both operands are variables
                        load_value(op1.strip(), reg2)  # Load first operand into R2
                        load_value(op2.strip(), reg1)  # Load second operand into R1
                        machine_code.append(f"SUBF {reg1}, {reg2}, {reg1}")  # Subtract R1 from R2
                    reg_used = reg2
                elif reg_used == reg2:
                    if is_number(op1.strip()):  # First operand is a constant
                        load_value(op2.strip(), reg1)  # Load variable into R2
                        machine_code.append(f"SUBF {reg1}, {reg1}, #{op1.strip()}")  # Multiply R2 with constant
                    elif is_number(op2.strip()):  # Second operand is a constant
                        load_value(op1.strip(), reg1)  # Load variable into R2
                        machine_code.append(f"SUBF {reg1}, {reg1}, #{op2.strip()}")  # Multiply R2 with constant
                    else:  # Both operands are variables
                        load_value(op1.strip(), reg2)  # Load first operand into R2
                        load_value(op2.strip(), reg1)  # Load second operand into R1
                        machine_code.append(f"SUBF {reg1}, {reg2}, {reg1}")  # Subtract R1 from R2
                    reg_used = reg1
            elif '/' in right:
                # Handle division
                op1, op2 = right.split(' / ')
                if reg_used == reg1:
                    if is_number(op1.strip()):  # First operand is a constant
                        load_value(op2.strip(), reg2)  # Load variable into R2
                        machine_code.append(f"DIVF {reg2}, {reg2}, #{op1.strip()}")  # Multiply R2 with constant
                    elif is_number(op2.strip()):  # Second operand is a constant
                        load_value(op1.strip(), reg2)  # Load variable into R2
                        machine_code.append(f"DIVF {reg2}, {reg2}, #{op2.strip()}")  # Multiply R2 with constant
                    else:  # Both operands are variables
                        load_value(op1.strip(), reg2)  # Load first operand into R2
                        load_value(op2.strip(), reg1)  # Load second operand into R1
                        machine_code.append(f"DIVF {reg1}, {reg2}, {reg1}")  # Divide R2 by R1
                    reg_used = reg2
                elif reg_used == reg2:
                    if is_number(op1.strip()):  # First operand is a constant
                        load_value(op2.strip(), reg1)  # Load variable into R2
                        machine_code.append(f"DIVF {reg1}, {reg1}, #{op1.strip()}")  # Multiply R2 with constant
                    elif is_number(op2.strip()):  # Second operand is a constant
                        load_value(op1.strip(), reg1)  # Load variable into R2
                        machine_code.append(f"DIVF {reg1}, {reg1}, #{op2.strip()}")  # Multiply R2 with constant
                    else:  # Both operands are variables
                        load_value(op1.strip(), reg2)  # Load first operand into R2
                        load_value(op2.strip(), reg1)  # Load second operand into R1
                        machine_code.append(f"DIVF {reg1}, {reg2}, {reg1}")  # Divide R2 by R1
                    reg_used = reg1
            if 'id' in left.strip():
                # Store the result in the left-hand variable
                machine_code.append(f"STRF {left.strip()}, {reg1}")  # Store R1 into the variable

    return machine_code


class CompilerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Compiler")
        self.create_widgets()

    def create_widgets(self):
        # Input section
        self.input_label = tk.Label(self.root, text="Chose Compiler or Hybrid:")
        self.input_label.pack(pady=5)

        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side="top", pady=10)
        # User choice for source code or math expression
        self.mode = tk.StringVar(value="Compiler")  # Default is 'Source Code'

        self.source_code_radio = tk.Radiobutton(self.top_frame, text="Compiler", variable=self.mode, value="Compiler")
        self.source_code_radio.pack(side="left", padx=20)

        self.math_expression_radio = tk.Radiobutton(self.top_frame, text="Hybrid", variable=self.mode, value="Hybrid")
        self.math_expression_radio.pack(side="left", padx=20)

        # Input section
        self.input_label = tk.Label(self.root, text="Chose Source Code or Math Expression:")
        self.input_label.pack(pady=5)

        # User choice for source code or math expression
        self.input_choice_var = tk.StringVar(value="Source Code")  # Default is 'Source Code'

        self.source_code_radio = tk.Radiobutton(self.root, text="Source Code", variable=self.input_choice_var,
                                                value="Source Code")
        self.source_code_radio.pack(anchor="w", padx=20)

        self.math_expression_radio = tk.Radiobutton(self.root, text="Math Expression", variable=self.input_choice_var,
                                                    value="Math Expression")
        self.math_expression_radio.pack(anchor="w", padx=20)

        self.input_text = scrolledtext.ScrolledText(self.root, width=70, height=2)
        self.input_text.pack(pady=5)

        # Button to process the input
        self.lexical_button = tk.Button(self.root, text="Start Analysis", command=self.start_analysis)
        self.lexical_button.pack(pady=5)

        # Output sections
        self.output_label = tk.Label(self.root, text="Analysis Output:")
        self.output_label.pack(pady=5)

        self.output_text = scrolledtext.ScrolledText(self.root, width=70, height=30)
        self.output_text.pack(pady=5)

    def start_analysis(self):
        self.output_text.delete(1.0, tk.END)  # Clear previous outputs
        user_input = self.input_text.get(1.0, tk.END).strip()

        if not user_input:
            messagebox.showwarning("Input Error", "Please enter some code or a math expression.")
            return

        input_choice = self.input_choice_var.get()

        try:
            if input_choice == "Source Code":
                # Perform lexical analysis directly on source code
                lexical_output = creat_lexical(user_input)
                self.output_text.insert(tk.END, "Lexical Analysis:\n" + lexical_output + "\n\n")

            elif input_choice == "Math Expression":
                # Convert math expression to source code and then perform lexical analysis
                source_code = convert_math_to_code(user_input)
                self.output_text.insert(tk.END, "Source Code:\n" + source_code + "\n\n")
                lexical_output = creat_lexical(source_code)
                self.output_text.insert(tk.END, "Lexical Analysis (from Math Expression):\n" + lexical_output + "\n\n")

            # Tokenize and parse
            tokens = tokenize(lexical_output)
            parse_tree_root = parse_expression(tokens)

            # Display syntax tree
            self.output_text.insert(tk.END, "Syntax Analysis:\n")
            self.collect_parse_tree(parse_tree_root)

            # Variable collection and semantic analysis
            variables = set()
            self.collect_variables(parse_tree_root, variables)

            variable_types = {}
            for var in variables:
                var_type = self.get_variable_type(var)
                variable_types[var] = var_type

            parse_tree = semantic_analysis(parse_tree_root, variable_types)
            self.output_text.insert(tk.END, "\nSemantic Analysis:\n")
            self.collect_parse_tree(parse_tree_root)

            mode_choice = self.mode.get()

            if mode_choice == "Hybrid":
                # Tokenize the expression
                tokens = tokenize(lexical_output)

                # Extract the left-hand side variable
                left_hand_side = None
                if tokens[0].type == 'ID' and len(tokens) > 1 and tokens[1].value == '=':
                    left_hand_side = tokens[0].value

                # Prompt user for variable values
                variable_values = {}
                for token in tokens:
                    if token.type == 'ID' and token.value != left_hand_side and token.value not in variable_values:
                        user_value = simpledialog.askstring("Variable Value", f"Enter the value of '{token.value}':")
                        variable_values[token.value] = user_value

                # Parse the expression
                result = hybrid_phase_and_evaluate(parse_tree, variable_values, variable_types)

                # Display the final parse tree
                self.output_text.insert(tk.END, "Direct Substitution:\n")
                self.collect_parse_tree(parse_tree)

                # Display the result
                if left_hand_side:
                    self.output_text.insert(tk.END, f"\nThe value of '{left_hand_side}' is: {result}\n")

            elif mode_choice == 'Compiler':
                # Intermediate and optimized code generation
                intermediate_code = generate_intermediate_code(parse_tree_root)
                self.output_text.insert(tk.END, "\nICG:\n" + "\n".join(intermediate_code) + "\n\n")

                optimized_code = optimize_code(intermediate_code)
                self.output_text.insert(tk.END, "\nOptimized Code:\n" + "\n".join(optimized_code) + "\n\n")

                # Machine code generation
                machine_code = generate_machine_code(optimized_code)
                self.output_text.insert(tk.END, "\nMachine Code:\n" + "\n".join(machine_code))

        except SyntaxError as e:
            messagebox.showerror("Syntax Error", str(e))

    def collect_parse_tree(self, node, prefix="", is_tail=True):
        if node is not None:
            tree_line = prefix + ("└── " if is_tail else "├──") + str(node.value) + "\n"
            self.output_text.insert(tk.END, tree_line)
            if node.left or node.right:
                if node.left:
                    self.collect_parse_tree(node.left, prefix + ("    " if is_tail else "│   "), node.right is None)
                if node.right:
                    self.collect_parse_tree(node.right, prefix + ("    " if is_tail else "│   "), True)

    def collect_variables(self, node, variables):
        if node is not None:
            if node.value.isidentifier():
                variables.add(node.value)
            self.collect_variables(node.left, variables)
            self.collect_variables(node.right, variables)

    def get_variable_type(self, var):
        # Continuously ask the user for a valid type (int or float)
        while True:
            if var == 'id1':
                return
            else:
                var_type = simpledialog.askstring("Variable Type", f"Enter type for variable '{var}' (int/float):")

            # Check if the input is either 'int' or 'float'
            if var_type and var_type.lower() in ['int', 'float']:
                return var_type.lower()  # Return the valid type in lowercase
            else:
                # Show a warning message box if the input is invalid
                messagebox.showwarning("Invalid Input", "Please enter either 'int' or 'float' as the type.")


if __name__ == "__main__":
    root = tk.Tk()
    app = CompilerGUI(root)
    root.mainloop()
