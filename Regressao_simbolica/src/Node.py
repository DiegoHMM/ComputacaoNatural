import random
import math
import sys

TERMINALS = ["X1", "X2"]#,"X3","X4","X5","X6","X7","X8"]
OPERATORS = ["+", "-", "*", "cos" , "sin"]
types = ["terminal", "operator"]

class Node:
    def __init__(self, type_, value, left=None, right=None, const_value=1, terminals=TERMINALS):
        self.type = type_
        self.value = value
        self.left = left
        self.right = right
        self.const_value = const_value
        self.terminals = terminals + ["const"]

    def evaluate(self, *args):
        if self.type == "terminal":
            if self.value == "const":
                return self.const_value
            else:
                return args[self.terminals.index(self.value)]
        elif self.type == "operator":
            if self.value == "+":
                return self.left.evaluate(*args) + self.right.evaluate(*args)
            elif self.value == "-":
                return self.left.evaluate(*args) - self.right.evaluate(*args)
            elif self.value == "*":
                return self.left.evaluate(*args) * self.right.evaluate(*args)
            elif self.value == "/":
                right_val = self.right.evaluate(*args)
                return self.left.evaluate(*args) / right_val if right_val != 0 else 0
            elif self.value == "sin":
                return math.sin(self.left.evaluate(*args))
            elif self.value == "cos":
                return math.cos(self.left.evaluate(*args))
            elif self.value == "exp":
                exp_val = self.left.evaluate(*args)
                # Limit the value passed to exp function
                exp_val = max(min(exp_val, 30), -30)
                try:
                    return math.exp(exp_val)
                except OverflowError:
                    return sys.float_info.max if exp_val > 0 else sys.float_info.min
            elif self.value == "ln":
                ln_val = self.left.evaluate(*args)
                ln_val = abs(ln_val)  # Ensure the value is positive
                # Limit the value passed to ln function
                ln_val = max(min(ln_val, sys.float_info.max), sys.float_info.min)
                try:
                    return math.log(ln_val)
                except ValueError:
                    return 0
    def __str__(self):
        if self.type == "terminal":
            return str(self.value)
        elif self.type == "operator":
            if self.value in ["sin", "cos", "exp", "ln"]:
                return f"{self.value}({self.left})"
            else:
                return f"({self.left} {self.value} {self.right})"
    
    def copy(self):
        left_copy = self.left.copy() if self.left is not None else None
        right_copy = self.right.copy() if self.right is not None else None
        return Node(self.type, self.value, left_copy, right_copy, self.const_value)
    
    def subtree(self, index, current_index=0):
        if current_index == index:
            return self
        else:
            left_result = None
            right_result = None

            if self.left is not None:
                left_result = self.left.subtree(index, current_index + 1)
                if left_result is not None:
                    return left_result

            if self.right is not None:
                right_result = self.right.subtree(index, current_index + 1 + self.left.count_nodes() if self.left is not None else current_index + 1)
                if right_result is not None:
                    return right_result
        return None

    def replace_subtree(self, index, new_subtree, current_index=0):
        if index == 0:
            return False

        if self.left is not None:
            if current_index + 1 == index:
                self.left = new_subtree
                return True
            else:
                left_result = self.left.replace_subtree(index, new_subtree, current_index + 1)
                if left_result:
                    return True

        if self.right is not None:
            right_start_index = current_index + 1 + (self.left.count_nodes() if self.left is not None else 0)
            if right_start_index == index:
                self.right = new_subtree
                return True
            else:
                right_result = self.right.replace_subtree(index, new_subtree, right_start_index)
                if right_result:
                    return True

        return False

    def node_depth(self):
        if self.left is None and self.right is None:
            return 0
        elif self.left is None:
            return self.right.node_depth() + 1
        elif self.right is None:
            return self.left.node_depth() + 1
        else:
            return max(self.left.node_depth(), self.right.node_depth()) + 1
        
    def select_subtree(self, index, current_index=0):
        if current_index == index:
            return self
        else:
            left_result = None
            right_result = None

            if self.left is not None:
                left_result = self.left.select_subtree(index, current_index + 1)
                if left_result is not None:
                    return left_result

            if self.right is not None:
                right_result = self.right.select_subtree(index, current_index + 1 + self.left.count_nodes() if self.left is not None else current_index + 1)
                if right_result is not None:
                    return right_result

    def get_operators(self):
        operators = []
        if self.type == "operator":
            operators.append(self.value)
            if self.left is not None:
                operators += self.left.get_operators()
            if self.right is not None:
                operators += self.right.get_operators()
        return operators
    def get_index(self, target_node, current_index=0):
        if self is target_node:
            return current_index

        left_index = None
        right_index = None

        if self.left is not None:
            left_index = self.left.get_index(target_node, current_index + 1)
            if left_index is not None:
                return left_index

        if self.right is not None:
            right_start_index = current_index + 1 + (self.left.count_nodes() if self.left is not None else 0)
            right_index = self.right.get_index(target_node, right_start_index)
            if right_index is not None:
                return right_index

        return None
            
    def count_nodes(self, count=0):
        if self.type == "terminal":
            count += 1
        elif self.type == "operator":
            if self.value in ["sin", "cos", "exp", "ln"]:
                count = self.left.count_nodes(count)
            else:
                count = self.left.count_nodes(count)
                count = self.right.count_nodes(count)
            count += 1
        return count




    
    def get_depth(self):
        depth = 0
        if self.left is not None:
            depth = max(depth, self.left.get_depth() + 1)
        if self.right is not None:
            depth = max(depth, self.right.get_depth() + 1)
        return depth

    @staticmethod
    def ramped_half(max_depth, terminal_prob, min_size):
        depth_range = range(2, max_depth + 1)
        while True:
            depth = random.choice(depth_range)
            if depth == max_depth:
                tree = Node.full(depth, terminal_prob, min_size)
            else:
                tree = Node.grow(depth, terminal_prob, min_size)
            if tree.count_nodes() >= min_size:
                return tree

    @staticmethod
    def grow(max_depth, terminal_prob, min_size):
        while True:
            if max_depth == 0:
                value = random.choice(TERMINALS)
                if value == "const":
                    return Node("terminal", value, const_value=random.uniform(-10, 10))
                else:
                    return Node("terminal", value)
            else:
                if random.random() < terminal_prob:
                    value = random.choice(TERMINALS)
                    if value == "const":
                        tree = Node("terminal", value, const_value=random.uniform(-10, 10))
                    else:
                        tree = Node("terminal", value)
                else:
                    value = random.choice(OPERATORS)
                    left = Node.grow(max_depth - 1, terminal_prob, min_size)
                    right = Node.grow(max_depth - 1, terminal_prob, min_size)
                    tree = Node("operator", value, left, right)
                if tree.count_nodes() >= min_size:
                    return tree

    @staticmethod
    def full(max_depth, terminal_prob, min_size):
        while True:
            if max_depth == 0:
                value = random.choice(TERMINALS)
                if value == "const":
                    return Node("terminal", value, const_value=random.uniform(-10, 10))
                else:
                    return Node("terminal", value)
            else:
                value = random.choice(OPERATORS)
                if value in ["+", "-", "*", "/"]:
                    left = Node.full(max_depth - 1, terminal_prob, min_size)
                    right = Node.full(max_depth - 1, terminal_prob, min_size)
                    tree = Node("operator", value, left, right)
                else:
                    tree = Node("operator", value, Node.full(max_depth - 1, terminal_prob, min_size))
                if tree.count_nodes() >= min_size:
                    return tree

def print_tree(node):
    if node.type == "terminal":
        if node.value == "const":
            print(f"({node.const_value})", end="")
        else:
            print(node.value, end="")
    elif node.type == "operator":
        if node.value in ["sin", "cos", "exp", "ln"]:
            print(f"{node.value}(", end="")
            print_tree(node.left)
            print(")", end="")
        else:
            print("(", end="")
            print_tree(node.left)
            print(f" {node.value} ", end="")
            print_tree(node.right)
            print(")", end="")
