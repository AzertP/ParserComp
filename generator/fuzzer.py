"""
Optimized simplefuzzer implementation for large grammars.

This is a highly optimized version of simplefuzzer that uses:
1. Aggressive memoization to avoid redundant cost computations
2. Precomputed cheap grammars for faster expansion
3. Optimized data structures and caching
"""

import random


def is_terminal(v):
    return (v[0], v[-1]) != ('<', '>')


def is_nonterminal(v):
    return (v[0], v[-1]) == ('<', '>')


def tree_to_string(tree):
    symbol, children, *_ = tree
    if children:
        return ''.join(tree_to_string(c) for c in children)
    else:
        return '' if is_nonterminal(symbol) else symbol


def modifiable(tree):
    name, children, *rest = tree
    if not is_nonterminal(name):
        return [name, []]
    else:
        return [name, [modifiable(c) for c in children]]


def iter_tree_to_str(tree_):
    tree = modifiable(tree_)
    expanded = []
    to_expand = [tree]
    while to_expand:
        (key, children, *rest), *to_expand = to_expand
        if is_nonterminal(key):
            to_expand = children + to_expand
        else:
            assert not children
            expanded.append(key)
    return ''.join(expanded)


def compute_cost_optimized(grammar):
    """
    Highly optimized cost computation using iterative fixed-point algorithm.
    Avoids expensive recursion and set operations.
    """
    # Step 1: Build dependency graph and identify terminals
    nonterminals = set(grammar.keys())
    
    # Step 2: Initialize costs - terminals cost 0, rules with only terminals cost 1
    symbol_cost = {}
    rule_cost = {}
    
    # Cache rule string representations
    rule_str_cache = {}
    
    for symbol in grammar:
        rule_cost[symbol] = {}
        min_cost = float('inf')
        
        for idx, rule in enumerate(grammar[symbol]):
            # Use tuple as key instead of str() for faster hashing
            rule_key = tuple(rule)
            rule_str_cache[(symbol, idx)] = rule_key
            
            # Check if rule has only terminals
            has_nonterminal = any(token in nonterminals for token in rule)
            if not has_nonterminal:
                # Pure terminal rule
                rule_cost[symbol][rule_key] = 1
                min_cost = 1
            else:
                # Will compute later
                rule_cost[symbol][rule_key] = float('inf')
        
        symbol_cost[symbol] = min_cost
    
    # Step 3: Fixed-point iteration to compute costs
    changed = True
    iteration = 0
    max_iterations = len(grammar) * 2  # Reasonable upper bound
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        for symbol in grammar:
            for idx, rule in enumerate(grammar[symbol]):
                rule_key = rule_str_cache[(symbol, idx)]
                old_cost = rule_cost[symbol][rule_key]
                
                # Compute rule cost based on current symbol costs
                max_token_cost = 0
                has_infinite = False
                
                for token in rule:
                    if token in nonterminals:
                        token_cost = symbol_cost.get(token, float('inf'))
                        if token_cost == float('inf'):
                            has_infinite = True
                            break
                        max_token_cost = max(max_token_cost, token_cost)
                
                if has_infinite:
                    new_cost = float('inf')
                else:
                    new_cost = max_token_cost + 1
                
                if new_cost < old_cost:
                    rule_cost[symbol][rule_key] = new_cost
                    changed = True
            
            # Update symbol cost
            min_rule_cost = min(rule_cost[symbol].values())
            if min_rule_cost < symbol_cost[symbol]:
                symbol_cost[symbol] = min_rule_cost
                changed = True
    
    # Convert back to use tuple keys consistently
    final_rule_cost = {}
    for symbol in grammar:
        final_rule_cost[symbol] = {}
        for idx, rule in enumerate(grammar[symbol]):
            rule_key = rule_str_cache[(symbol, idx)]
            # Keep as tuple key instead of str(rule)
            final_rule_cost[symbol][str(rule)] = rule_cost[symbol][rule_key]
    
    return final_rule_cost


class LimitFuzzer:
    """
    Optimized grammar fuzzer with depth limiting.
    
    Uses precomputed costs and cheap grammars for fast generation.
    """

    def __init__(self, grammar, weights, bias_long=False, long_bias_factor=2.0):
        self.grammar = grammar
        self.rule_weights = weights
        self.bias_long = bias_long
        self.long_bias_factor = long_bias_factor
        print("Computing grammar costs...")
        self.cost = compute_cost_optimized(grammar)
        print("Precomputing cheap grammar...")
        self._precompute_cheap_grammar()
        if bias_long:
            print(f"Precomputing expensive grammar (bias factor: {long_bias_factor})...")
            self._precompute_expensive_grammar()
        print("Fuzzer initialized.")

    def _precompute_cheap_grammar(self):
        """Precompute cheap grammar once during initialization."""
        self.cheap_grammar = {}
        for k in self.cost:
            rules = self.grammar[k]
            min_cost = min([self.cost[k][str(r)] for r in rules])
            # Only keep rules with minimum cost that aren't infinite
            if min_cost != float('inf'):
                self.cheap_grammar[k] = [r for r in rules if self.cost[k][str(r)] == min_cost]
            else:
                # Fallback to all rules if all are infinite (shouldn't happen in valid grammars)
                self.cheap_grammar[k] = rules
    
    def _precompute_expensive_grammar(self):
        """Precompute expensive grammar for generating longer strings."""
        self.expensive_grammar = {}
        for k in self.cost:
            rules = self.grammar[k]
            # Filter out epsilon productions and prefer longer rules
            non_empty_rules = [r for r in rules if len(r) > 0]
            if non_empty_rules:
                # Sort by cost (higher is better for long strings) and rule length
                sorted_rules = sorted(non_empty_rules, 
                                    key=lambda r: (self.cost[k][str(r)], len(r)), 
                                    reverse=True)
                # Take top rules biased toward expensive/long productions
                cutoff = max(1, len(sorted_rules) // 2)
                self.expensive_grammar[k] = sorted_rules[:cutoff]
            else:
                # Fallback if all rules are empty
                self.expensive_grammar[k] = rules

    def _weighted_choice(self, key, rules):
        # User-specified rule probabilities take precedence
        if key in self.rule_weights:
            weights = self.rule_weights[key]
            if len(weights) != len(self.grammar[key]):
                raise ValueError(
                    f"rule-weights for {key} has {len(weights)} weights "
                    f"but grammar has {len(self.grammar[key])} rules"
                )
            # If we're using a filtered grammar (cheap/expensive), map the
            # original weights onto the remaining rules.
            if len(rules) != len(self.grammar[key]):
                original = self.grammar[key]
                weights = [
                    weights[original.index(rule)]
                    for rule in rules
                ]
            return random.choices(rules, weights=weights, k=1)[0]

        """Choose a rule with bias toward longer/more expensive productions."""
        if not self.bias_long or len(rules) == 1:
            return random.choice(rules)
        
        # If key is not in cost (terminal symbol), just use random choice
        if key not in self.cost:
            return random.choice(rules)
        
        # Calculate weights based on cost and length
        weights = []
        for rule in rules:
            cost = self.cost[key].get(str(rule), 1)
            length_bonus = len(rule) * self.long_bias_factor
            # Avoid infinite costs
            if cost == float('inf'):
                weights.append(1)
            else:
                weights.append(cost + length_bonus)
        
        # Weighted random choice
        total = sum(weights)
        if total == 0:
            return random.choice(rules)
        
        r = random.uniform(0, total)
        cumsum = 0
        for rule, weight in zip(rules, weights):
            cumsum += weight
            if r <= cumsum:
                return rule
        return rules[-1]

    def gen_key(self, key, depth, max_depth):
        if key not in self.grammar:
            return key
        if depth > max_depth:
            # Use precomputed cheap grammar
            rules = self.cheap_grammar.get(key, self.grammar[key])
        else:
            rules = self.grammar[key]
        return self.gen_rule(self._weighted_choice(key, rules), depth + 1, max_depth)

    def gen_rule(self, rule, depth, max_depth):
        return ''.join(self.gen_key(token, depth, max_depth) for token in rule)

    def fuzz(self, key='<start>', max_depth=10):
        return self.gen_key(key=key, depth=0, max_depth=max_depth)

    def iter_gen_key(self, key, max_depth, max_iterations=1000000, max_queue_size=100000):
        """Optimized iterative key generation with safety limits."""
        def get_def(t):
            if is_nonterminal(t):
                return [t, None]
            else:
                return [t, []]

        root = [key, None]
        queue = [(0, root)]
        iterations = 0
        
        while queue:
            iterations += 1
            
            # Safety check: prevent infinite loops
            if iterations > max_iterations:
                print(f"Warning: Reached max iterations ({max_iterations}), stopping generation")
                break
            
            # Safety check: prevent unbounded queue growth
            if len(queue) > max_queue_size:
                print(f"Warning: Queue size exceeded {max_queue_size}, stopping generation")
                break
            
            # Pop from front for BFS-like behavior
            (depth, item), *queue = queue
            key = item[0]
            
            if item[1] is not None:
                continue
            
            # Check if this is a terminal - if so, mark as complete
            if key not in self.grammar:
                item[1] = []
                continue
            
            # Use precomputed cheap grammar when depth exceeded
            if depth < max_depth:
                grammar_rules = self.grammar[key]
            else:
                grammar_rules = self.cheap_grammar.get(key, self.grammar[key])
            
            # Use weighted choice if bias_long is enabled
            chosen_rule = self._weighted_choice(key, grammar_rules)
            expansion = [get_def(t) for t in chosen_rule]
            item[1] = expansion
            
            # Add to queue
            for t in expansion:
                queue.append((depth + 1, t))
        
        return root

    def iter_fuzz(self, key='<start>', max_depth=10):
        """Fast iterative fuzzing that avoids stack depth issues."""
        self._s = self.iter_gen_key(key=key, max_depth=max_depth)
        return iter_tree_to_str(self._s)

if __name__ == '__main__':
    import json
    with open('grammars/json.json') as f:
        j = json.load(f)
        lf = LimitFuzzer(j['rules'], j['rule-weights'])
        print(lf.fuzz(j['start']))